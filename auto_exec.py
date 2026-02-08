"""Auto-execution manager (E*TRADE).

Design goals:
 - Zero impact on existing signal logic.
 - State survives Streamlit reruns via st.session_state.
 - No cached functions or unstable return shapes.
 - Conservative defaults: LONG-only, confirm-only optional.

This module owns:
 - eligibility gating (time windows, min score, engine selection)
 - lifecycle state machine per symbol
 - order placement + reconciliation (entry -> stop + TP0)
 - end-of-day liquidation (hard close by 15:55 ET)
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from datetime import datetime, time
from typing import Any, Dict, Optional, Tuple

import math

import streamlit as st

from email_utils import send_email_alert
from etrade_client import ETradeClient
from sessions import classify_session


ET_TZ = "America/New_York"
ENTRY_TIMEOUT_MINUTES = 20  # default; can be overridden via AutoExecConfig.timeout_minutes


@dataclass
class AutoExecConfig:
    enabled: bool
    sandbox: bool
    engines: Tuple[str, ...]
    min_score: float
    max_dollars_per_trade: float
    max_pool_dollars: float
    max_concurrent_symbols: int
    lifecycles_per_symbol_per_day: int
    timeout_minutes: int
    tp0_deviation: float
    confirm_only: bool
    status_emails: bool
    hourly_pnl_emails: bool


    enforce_entry_windows: bool
    entry_grace_minutes: int
@dataclass
class TradeLifecycle:
    symbol: str
    engine: str
    created_ts: str
    stage: str  # STAGED, ENTRY_SENT, IN_POSITION, CLOSED, CANCELED
    desired_entry: float
    stop: float
    tp0: float  # already adjusted by cfg.tp0_deviation (exit limit)
    qty: int
    reserved_dollars: float
    entry_order_id: Optional[int] = None
    stop_order_id: Optional[int] = None
    tp_order_id: Optional[int] = None
    filled_qty: int = 0
    entry_sent_ts: Optional[str] = None
    bracket_qty: int = 0
    emailed_events: Dict[str, str] = field(default_factory=dict)
    notes: str = ""

def _now_et() -> datetime:
    # Streamlit environment should have tzdata; fall back to naive local if needed.
    try:
        from zoneinfo import ZoneInfo

        return datetime.now(ZoneInfo(ET_TZ))
    except Exception:
        return datetime.now()


def _in_exec_window(now: datetime, allow_opening: bool, allow_midday: bool, allow_power: bool) -> bool:
    """Two windows: 09:50–11:00 and 14:00–15:30 ET."""
    t = now.time()
    w1 = time(9, 50) <= t <= time(11, 0)
    w2 = time(14, 0) <= t <= time(15, 30)
    if w1:
        return allow_opening
    if w2:
        # Split midday vs power
        if t < time(15, 0):
            return allow_midday
        return allow_power
    return False


def _is_liquidation_time(now: datetime) -> bool:
    return now.time() >= time(15, 55)


def _get_state() -> Dict[str, Any]:
    """Return the durable autoexec state stored in st.session_state.

    Important: app.py may create st.session_state['autoexec'] during OAuth
    before this function is ever called. In that case, we *must not* wipe
    the auth tokens when we "initialize" the rest of the state.
    """

    today = _now_et().date().isoformat()

    # Start from whatever is present (OAuth flow may have created a partial dict)
    existing: Dict[str, Any] = st.session_state.get("autoexec", {}) or {}
    existing_auth = existing.get("auth", {}) if isinstance(existing, dict) else {}

    # Initialize / backfill missing keys without losing auth
    state: Dict[str, Any] = {
        "pool_reserved": float(existing.get("pool_reserved", 0.0)) if isinstance(existing, dict) else 0.0,
        "lifecycles": existing.get("lifecycles", {}) if isinstance(existing, dict) else {},
        "auth": existing_auth if isinstance(existing_auth, dict) else {},
        "day": str(existing.get("day", today)) if isinstance(existing, dict) else today,
        "skip_notices": existing.get("skip_notices", {}) if isinstance(existing, dict) else {},
        "hourly_report_last": str(existing.get("hourly_report_last", "")) if isinstance(existing, dict) else "",
        "last_action": str(existing.get("last_action", "")) if isinstance(existing, dict) else "",
    }

    # Daily reset (preserve auth so "auth before boot" remains valid)
    if state.get("day") != today:
        state = {
            "pool_reserved": 0.0,
            "lifecycles": {},
            "auth": state.get("auth", {}),
            "day": today,
            "skip_notices": {},
            "hourly_report_last": "",
            "last_action": "",
        }

    st.session_state["autoexec"] = state
    return state



def _email_settings():
    """Read SMTP settings from Streamlit secrets. Returns tuple or None."""
    try:
        cfg = st.secrets.get("email", {}) or {}
    except Exception:
        return None
    smtp_server = cfg.get("smtp_server")
    smtp_port = cfg.get("smtp_port")
    smtp_user = cfg.get("smtp_user")
    smtp_password = cfg.get("smtp_password")

    # Accept to_emails (preferred) OR to_email (string). Normalize to list[str].
    to_emails = cfg.get("to_emails")
    if to_emails is None:
        to_email = cfg.get("to_email", "")
        if isinstance(to_email, str):
            # Support comma-separated lists in legacy config.
            parts = [p.strip() for p in to_email.split(",") if p.strip()]
            to_emails = parts
        elif to_email:
            to_emails = [str(to_email).strip()]
        else:
            to_emails = []
    if isinstance(to_emails, str):
        to_emails = [e.strip() for e in to_emails.split(",") if e.strip()]
    if not (smtp_server and smtp_port and smtp_user and smtp_password and to_emails):
        return None
    try:
        smtp_port_int = int(smtp_port)
    except Exception:
        return None
    return smtp_server, smtp_port_int, str(smtp_user), str(smtp_password), [str(e).strip() for e in to_emails if str(e).strip()]


def _send_status_email(cfg: AutoExecConfig, subject: str, body: str) -> None:
    """Send auto-exec lifecycle status emails (one email per recipient)."""
    if not getattr(cfg, "status_emails", False):
        return
    settings = _email_settings()
    if settings is None:
        return
    smtp_server, smtp_port, smtp_user, smtp_password, to_emails = settings
    try:
        send_email_alert(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            to_emails=to_emails,
            subject=subject,
            body=body,
        )
    except Exception:
        # Never crash the app due to email
        return


def _should_send_hourly(now: datetime) -> Optional[str]:
    """Return an hourly key (YYYY-MM-DD:HH) if we are within the report window.

    We aim for "every hour" during the regular session. Because Streamlit reruns
    on a timer, we allow a small minute window so we don't miss the top of the hour.
    """
    # Monday=0 ... Sunday=6
    if now.weekday() > 4:
        return None

    t = now.time()
    # Regular session 09:30–16:00 ET
    if t < time(9, 30) or t > time(16, 0):
        return None

    # Send at 10:00, 11:00, ... 16:00 (inclusive). Allow minute window [0, 7].
    if now.hour < 10 or now.hour > 16:
        return None
    if not (0 <= now.minute <= 7):
        return None

    return f"{now.date().isoformat()}:{now.hour:02d}"


def _extract_positions(portfolio_json: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Best-effort extraction of position objects from E*TRADE portfolio JSON."""

    positions: list[Dict[str, Any]] = []

    def _walk(obj):
        if isinstance(obj, dict):
            # E*TRADE commonly uses key 'Position' or 'position' for lists
            for k, v in obj.items():
                if str(k).lower() == "position":
                    if isinstance(v, list):
                        for it in v:
                            if isinstance(it, dict):
                                positions.append(it)
                    elif isinstance(v, dict):
                        positions.append(v)
                else:
                    _walk(v)
        elif isinstance(obj, list):
            for it in obj:
                _walk(it)

    _walk(portfolio_json)
    return positions


def _pos_symbol(pos: Dict[str, Any]) -> str:
    # Common: pos['Product']['symbol']
    try:
        sym = pos.get("Product", {}).get("symbol")
        if sym:
            return str(sym).upper().strip()
    except Exception:
        pass
    # Sometimes: pos['product']['symbol']
    try:
        sym = pos.get("product", {}).get("symbol")
        if sym:
            return str(sym).upper().strip()
    except Exception:
        pass
    # Fallback: traverse
    def _walk(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() == "symbol" and isinstance(v, str):
                    return v
                r = _walk(v)
                if r:
                    return r
        elif isinstance(obj, list):
            for it in obj:
                r = _walk(it)
                if r:
                    return r
        return ""
    sym = _walk(pos)
    return str(sym).upper().strip() if sym else ""



def _safe_num(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _maybe_send_hourly_pnl(cfg: AutoExecConfig, state: Dict[str, Any], client: Optional[ETradeClient]) -> None:
    """Send an hourly P&L + analytics email during the regular session."""
    if not getattr(cfg, "hourly_pnl_emails", False):
        return

    now = _now_et()
    key = _should_send_hourly(now)
    if not key:
        return

    last = str(state.get("hourly_report_last", "") or "")
    if last == key:
        return

    # Must have auth bound to an account.
    account_id_key = state.get("auth", {}).get("account_id_key")
    if not (client and account_id_key):
        return

    # Build core analytics from our own state (always available)
    lifecycles = state.get("lifecycles", {}) or {}
    staged = entry_sent = in_pos = closed = canceled = 0
    active_symbols = 0
    managed_symbols: set[str] = set()
    for sym, lst in lifecycles.items():
        managed_symbols.add(str(sym).upper())
        sym_active = False
        for raw in (lst or []):
            stg = str((raw or {}).get("stage", ""))
            if stg == "STAGED":
                staged += 1
            elif stg == "ENTRY_SENT":
                entry_sent += 1
                sym_active = True
            elif stg == "IN_POSITION":
                in_pos += 1
                sym_active = True
            elif stg == "CLOSED":
                closed += 1
            elif stg == "CANCELED":
                canceled += 1
        if sym_active:
            active_symbols += 1

    # Pull portfolio snapshot from E*TRADE
    port_lines: list[str] = []
    total_mkt = total_gl = 0.0
    port_ok = False
    try:
        pj = client.get_portfolio(str(account_id_key))
        pos_list = _extract_positions(pj)
        rows = []
        for p in pos_list:
            sym = _pos_symbol(p)
            if not sym:
                continue
            if managed_symbols and sym not in managed_symbols:
                # Keep it focused to bot-managed symbols.
                continue
            qty = _safe_num(p.get("quantity") or p.get("Quantity"))
            mv = _safe_num(p.get("marketValue") or p.get("MarketValue"))
            gl = _safe_num(p.get("totalGainLoss") or p.get("TotalGainLoss") or p.get("unrealizedGainLoss") or p.get("UnrealizedGainLoss"))
            if mv is not None:
                total_mkt += mv
            if gl is not None:
                total_gl += gl
            rows.append((sym, qty, mv, gl))
        if rows:
            port_ok = True
            port_lines.append("Bot-managed positions (E*TRADE portfolio):")
            for sym, qty, mv, gl in sorted(rows, key=lambda x: x[0]):
                q = "—" if qty is None else f"{qty:.0f}"
                m = "—" if mv is None else f"${mv:,.2f}"
                g = "—" if gl is None else f"${gl:,.2f}"
                port_lines.append(f"  • {sym}: qty {q} | mkt {m} | P&L {g}")
            port_lines.append(f"Totals (managed): market ${total_mkt:,.2f} | P&L ${total_gl:,.2f}")
        else:
            port_lines.append("No bot-managed positions found in portfolio snapshot.")
    except Exception as e:
        port_lines.append(f"Portfolio snapshot unavailable: {e}")

    # Open orders count (managed symbols)
    order_lines: list[str] = []
    try:
        oj = client.list_orders(str(account_id_key), status="OPEN", count=50)
        # Walk for orders and count by symbol
        open_orders = 0
        def _walk_orders(obj):
            nonlocal open_orders
            if isinstance(obj, dict):
                # identify order objects by presence of Instrument
                if "Instrument" in obj and isinstance(obj.get("Instrument"), list):
                    sym = ""
                    try:
                        sym = str(obj.get("Instrument")[0].get("Product", {}).get("symbol", "") or "").upper().strip()
                    except Exception:
                        sym = ""
                    if not managed_symbols or (sym and sym in managed_symbols):
                        open_orders += 1
                for v in obj.values():
                    _walk_orders(v)
            elif isinstance(obj, list):
                for it in obj:
                    _walk_orders(it)
        _walk_orders(oj)
        order_lines.append(f"Open orders (managed): {open_orders}")
    except Exception as e:
        order_lines.append(f"Open orders snapshot unavailable: {e}")

    subj = f"[AUTOEXEC] Hourly P&L Update — {now.hour:02d}:00 ET"
    body = (
        f"Time (ET): {now.isoformat()}\n"
        f"Environment: {'SANDBOX' if cfg.sandbox else 'LIVE'}\n\n"
        f"Last bot action: {str(state.get('last_action', '') or '—')}\n\n"
        f"Auto‑exec today:\n"
        f"  • Active symbols: {active_symbols}\n"
        f"  • Lifecycles — STAGED: {staged}, ENTRY_SENT: {entry_sent}, IN_POSITION: {in_pos}, CLOSED: {closed}, CANCELED: {canceled}\n"
        f"  • Pool reserved: ${float(state.get('pool_reserved', 0.0) or 0.0):,.2f} / ${float(cfg.max_pool_dollars):,.2f}\n\n"
        + "\n".join(port_lines)
        + "\n\n"
        + "\n".join(order_lines)
        + "\n\n"
        "Note: This report is informational only. Execution logic remains governed by your stop-loss + TP0 rules."
    )
    _send_status_email(cfg, subj, body)
    state["hourly_report_last"] = key


def _event_once(cfg: AutoExecConfig, lifecycle: TradeLifecycle, event_key: str, subject: str, body: str) -> None:
    """Dedup: send an event email once per lifecycle per event_key."""
    try:
        sent = (lifecycle.emailed_events or {}).get(event_key)
    except Exception:
        sent = None
    if sent:
        return
    _send_status_email(cfg, subject, body)
    try:
        lifecycle.emailed_events[event_key] = _now_et().isoformat()
    except Exception:
        pass


def _active_symbols(state: Dict[str, Any]) -> int:
    n = 0
    for sym, lst in state.get("lifecycles", {}).items():
        for l in lst:
            if l.get("stage") in {"STAGED", "ENTRY_SENT", "IN_POSITION"}:
                n += 1
                break
    return n


def _symbol_lifecycle_count_today(state: Dict[str, Any], symbol: str) -> int:
    return len(state.get("lifecycles", {}).get(symbol, []))


def _reserve_pool(state: Dict[str, Any], dollars: float, max_pool: float) -> bool:
    if state["pool_reserved"] + dollars > max_pool:
        return False
    state["pool_reserved"] += dollars
    return True


def _release_pool(state: Dict[str, Any], dollars: float) -> None:
    state["pool_reserved"] = max(0.0, float(state.get("pool_reserved", 0.0)) - float(dollars))


def _set_last_action(state: Dict[str, Any], summary: str) -> None:
    """Store a short human-readable summary of the bot's most recent meaningful action.

    Included in hourly P&L emails so you can quickly see what the bot last did
    without digging through all status emails.
    """
    try:
        state["last_action"] = str(summary or "")
    except Exception:
        pass


def _parse_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, str):
            x = x.replace("$", "").strip()
        return float(x)
    except Exception:
        return None


def build_desired_entry_for_ride(pbl: float, pbh: float, stage: str) -> float:
    rng = max(0.0001, pbh - pbl)
    if str(stage).upper().startswith("PRE"):
        return pbl + 0.33 * rng
    return pbl + 0.66 * rng


def compute_qty(max_dollars: float, entry: float) -> int:
    if entry <= 0:
        return 0
    return int(math.floor(max_dollars / entry))


def should_stage_lifecycle(cfg: AutoExecConfig, payload: Dict[str, Any]) -> bool:
    # LONG-only for v1
    if str(payload.get("bias", "LONG")).upper() != "LONG":
        return False
    score = _parse_float(payload.get("score"))
    if score is None or score < cfg.min_score:
        return False
    stage = str(payload.get("stage", "")).upper()
    if cfg.confirm_only and "CONF" not in stage:
        return False
    return True


def stage_from_payload(cfg: AutoExecConfig, engine: str, payload: Dict[str, Any]) -> Optional[TradeLifecycle]:
    symbol = str(payload.get("symbol", payload.get("Symbol", "")) or "").upper().strip()
    if not symbol:
        return None

    entry = _parse_float(payload.get("entry"))
    stop = _parse_float(payload.get("stop"))
    tp0 = _parse_float(payload.get("tp0"))
    if entry is None or stop is None or tp0 is None:
        return None

    # Adjust TP0 by deviation (sell a bit early). For longs: tp0 - dev
    tp0_adj = max(0.0, tp0 - float(cfg.tp0_deviation or 0.0))

    desired_entry = entry
    if engine == "RIDE":
        pbl = _parse_float(payload.get("pb_low"))
        pbh = _parse_float(payload.get("pb_high"))
        if pbl is not None and pbh is not None:
            desired_entry = build_desired_entry_for_ride(pbl, pbh, str(payload.get("stage", "")))

    qty = compute_qty(cfg.max_dollars_per_trade, desired_entry)
    if qty <= 0:
        return None

    reserved = qty * desired_entry
    return TradeLifecycle(
        symbol=symbol,
        engine=engine,
        created_ts=_now_et().isoformat(),
        stage="STAGED",
        desired_entry=float(desired_entry),
        stop=float(stop),
        tp0=float(tp0_adj),
        qty=qty,
        reserved_dollars=float(reserved),
        notes="staged",
    )


def ensure_client(cfg: AutoExecConfig) -> Optional[ETradeClient]:
    state = _get_state()
    auth = state.get("auth", {})
    ck = auth.get("consumer_key")
    cs = auth.get("consumer_secret")
    at = auth.get("access_token")
    ats = auth.get("access_token_secret")
    if not (ck and cs and at and ats):
        return None
    try:
        return ETradeClient(
            consumer_key=ck,
            consumer_secret=cs,
            sandbox=cfg.sandbox,
            access_token=at,
            access_token_secret=ats,
        )
    except Exception:
        return None


def reconcile_and_execute(
    cfg: AutoExecConfig,
    allow_pre: bool,
    allow_opening: bool,
    allow_midday: bool,
    allow_power: bool,
    allow_after: bool,
    fetch_last_price_fn,
) -> None:
    """Runs every Streamlit rerun to reconcile order state and enforce EOD."""
    if not cfg.enabled:
        return

    state = _get_state()
    now = _now_et()

    # Hourly checkpoint email (P&L + simple analytics) is independent of the
    # execution windows. It should still fire even if we're outside the
    # opening/midday/power windows (as long as OAuth is active).
    try:
        if getattr(cfg, "hourly_pnl_emails", False):
            key = _should_send_hourly(now)
            if key and str(state.get("hourly_report_last", "") or "") != key:
                client_for_report = ensure_client(cfg)
                if client_for_report is not None:
                    _maybe_send_hourly_pnl(cfg, state, client_for_report)
                    # Persist any state mutations (dedupe key)
                    st.session_state["autoexec"] = state
    except Exception:
        # Never crash the app due to reporting.
        pass

    # Liquidation enforcement
    if _is_liquidation_time(now):
        _force_liquidate_all(cfg, state, fetch_last_price_fn)
        return

    if not _in_exec_window(now, allow_opening, allow_midday, allow_power):
        return

    client = ensure_client(cfg)
    if client is None:
        return

    account_id_key = state.get("auth", {}).get("account_id_key")
    if not account_id_key:
        return

    # Reconcile lifecycles
    lifecycles = state.get("lifecycles", {})
    for symbol, lst in list(lifecycles.items()):
        for idx, raw in enumerate(list(lst)):
            lifecycle = TradeLifecycle(**raw)
            try:
                _reconcile_one(client, account_id_key, state, lifecycle, cfg)
            except Exception as e:
                lifecycle.notes = f"reconcile_error: {e}"
            lst[idx] = asdict(lifecycle)


def handle_alert_for_autoexec(
    cfg: AutoExecConfig,
    engine: str,
    payload: Dict[str, Any],
    allow_pre: bool,
    allow_opening: bool,
    allow_midday: bool,
    allow_power: bool,
    allow_after: bool,
) -> None:
    """Called only when the app has already decided to send an email alert."""
    if not cfg.enabled:
        return
    if engine not in set(cfg.engines):
        return

    now = _now_et()
    session = classify_session(now)
    # respect allowed sessions (even though you usually toggle all on)
    allowed = {
        "PRE": allow_pre,
        "OPENING": allow_opening,
        "MIDDAY": allow_midday,
        "POWER": allow_power,
        "AFTER": allow_after,
    }
    if not allowed.get(session, True):
        return

    if not _in_exec_window(now, allow_opening, allow_midday, allow_power):
        return

    if not should_stage_lifecycle(cfg, payload):
        return

    state = _get_state()

    symbol = str(payload.get("symbol", "") or payload.get("Symbol", "")).upper().strip()

    def _skip_once(reason: str) -> None:
        key = f"{symbol}:{reason}"
        sent = state.get("skip_notices", {}).get(key)
        if sent:
            return
        state.setdefault("skip_notices", {})[key] = _now_et().isoformat()
        subj = f"[AUTOEXEC] {symbol} SKIP — {reason}"
        body = f"Time (ET): {_now_et().isoformat()}\nSymbol: {symbol}\nEngine: {engine}\nReason: {reason}\n\nThis is a one-time notice for today."
        _send_status_email(cfg, subj, body)

    # Concurrency and lifecycle limits
    if _active_symbols(state) >= int(cfg.max_concurrent_symbols):
        _skip_once("max_concurrent_symbols")
        return

    if _symbol_lifecycle_count_today(state, symbol) >= int(cfg.lifecycles_per_symbol_per_day):
        _skip_once("lifecycle_cap")
        return

    lifecycle = stage_from_payload(cfg, engine, payload)
    if lifecycle is None:
        return

    if not _reserve_pool(state, lifecycle.reserved_dollars, float(cfg.max_pool_dollars)):
        _skip_once("pool_cap")
        return

    state.setdefault("lifecycles", {}).setdefault(symbol, []).append(asdict(lifecycle))

    # Status email: staged
    subj = f"[AUTOEXEC] {symbol} {engine} STAGED"
    score = payload.get("score", payload.get("Score"))
    tier = payload.get("tier", payload.get("Tier"))
    body = (
        f"Time (ET): {now.isoformat()}\n"
        f"Symbol: {symbol}\nEngine: {engine}\nTier: {tier}\nScore: {score}\n\n"
        f"STAGED — waiting for entry conditions.\n\n"
        f"Desired entry: {lifecycle.desired_entry}\nStop: {lifecycle.stop}\nTP0 (exit limit): {lifecycle.tp0}\nQty: {lifecycle.qty}\nReserved: ${lifecycle.reserved_dollars:.2f}\n"
    )
    _event_once(cfg, lifecycle, "STAGED", subj, body)

    # _event_once mutates lifecycle.emailed_events for dedupe.
    # Persist that mutation back into session_state so reruns don't resend.
    try:
        lifelist = state.get("lifecycles", {}).get(symbol)
        if isinstance(lifelist, list) and lifelist:
            lifelist[-1] = asdict(lifecycle)
    except Exception:
        # If persistence fails, it's non-fatal; worst case is a duplicate status email.
        pass


def try_send_entries(cfg: AutoExecConfig, allow_opening: bool, allow_midday: bool, allow_power: bool, fetch_last_price_fn) -> None:
    """Places entry orders for STAGED lifecycles when price is in range.

    Safety:
      - entry is ONLY sent if last is ABOVE stop AND at/below desired entry.
      - staged/entry orders time out after cfg.timeout_minutes.
    """
    if not cfg.enabled:
        return
    state = _get_state()
    client = ensure_client(cfg)
    if client is None:
        return
    account_id_key = state.get("auth", {}).get("account_id_key")
    if not account_id_key:
        return

    now = _now_et()

    # Entry-window gating controls NEW entry submissions only.
    # Exits (stops/targets/EOD) are handled in reconcile.
    in_window_now = _in_exec_window(now, allow_opening, allow_midday, allow_power)
    enforce_windows = bool(getattr(cfg, "enforce_entry_windows", True))
    grace_min = int(getattr(cfg, "entry_grace_minutes", 0) or 0)
    for symbol, lst in list(state.get("lifecycles", {}).items()):
        for idx, raw in enumerate(list(lst)):
            lifecycle = TradeLifecycle(**raw)

            # Timeout STAGED lifecycles that never got an entry window
            if lifecycle.stage == "STAGED":
                try:
                    created = datetime.fromisoformat(lifecycle.created_ts)
                    age_min = (now - created).total_seconds() / 60.0
                except Exception:
                    age_min = 0.0
                timeout_m = int(getattr(cfg, "timeout_minutes", ENTRY_TIMEOUT_MINUTES) or ENTRY_TIMEOUT_MINUTES)
                if age_min >= timeout_m:
                    lifecycle.stage = "CANCELED"
                    lifecycle.notes = f"staged_timeout_{timeout_m}m"
                    _event_once(cfg, lifecycle, "STAGED_TIMEOUT", f"[AUTOEXEC] {lifecycle.symbol} STAGED TIMEOUT", f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nCanceled: staged timeout ({timeout_m}m)\n")
                    # release unused reserved dollars back to the pool
                    _release_pool(state, lifecycle.reserved_dollars)
                    lst[idx] = asdict(lifecycle)
                    continue

            if lifecycle.stage != "STAGED":
                continue

            # Entry window gating: prevents orders outside selected execution windows.
            # If enforce_entry_windows is ON, we only send entries during the window,
            # with an optional small grace period based on lifecycle.created_ts.
            if enforce_windows and not in_window_now:
                allow_via_grace = False
                if grace_min > 0:
                    try:
                        created_dt = datetime.fromisoformat(lifecycle.created_ts)
                        if _in_exec_window(created_dt, allow_opening, allow_midday, allow_power):
                            age_min = (now - created_dt).total_seconds() / 60.0
                            if age_min <= float(grace_min):
                                allow_via_grace = True
                    except Exception:
                        allow_via_grace = False
                if not allow_via_grace:
                    continue

            try:
                last = _parse_float(fetch_last_price_fn(symbol))
            except Exception:
                last = None
            if last is None:
                continue

            # Universal tight-entry constraint: above stop, below/at entry limit
            if not (last <= lifecycle.desired_entry and last > lifecycle.stop):
                continue

            # Place entry order (limit at desired_entry)
            try:
                oid = client.place_equity_limit_order(
                    account_id_key=account_id_key,
                    symbol=symbol,
                    qty=lifecycle.qty,
                    limit_price=float(lifecycle.desired_entry),
                    action="BUY",
                    market_session="REGULAR",
                )
                lifecycle.entry_order_id = oid
                lifecycle.entry_sent_ts = now.isoformat()
                lifecycle.stage = "ENTRY_SENT"
                lifecycle.notes = f"entry_sent@{lifecycle.desired_entry}"
                _event_once(cfg, lifecycle, "ENTRY_SENT", f"[AUTOEXEC] {symbol} ENTRY SENT", f"Time (ET): {now.isoformat()}\nSymbol: {symbol}\nEngine: {lifecycle.engine}\n\nEntry limit placed.\nQty: {lifecycle.qty}\nLimit: {lifecycle.desired_entry}\nStop (planned): {lifecycle.stop}\nTP0 (planned exit): {lifecycle.tp0}\nOrderId: {oid}\n")
            except Exception as e:
                lifecycle.notes = f"entry_send_failed: {e}"
                _event_once(cfg, lifecycle, "ENTRY_SEND_FAILED", f"[AUTOEXEC] {symbol} ENTRY SEND FAILED", f"Time (ET): {now.isoformat()}\nSymbol: {symbol}\nEngine: {lifecycle.engine}\n\nEntry placement failed: {e}\n")

            lst[idx] = asdict(lifecycle)


def _reconcile_one(client: ETradeClient, account_id_key: str, state: Dict[str, Any], lifecycle: TradeLifecycle, cfg: AutoExecConfig) -> None:
    """Update lifecycle state based on order statuses, enforcing full exits and timeouts."""
    now = _now_et()

    # ---- ENTRY SENT: monitor fill / timeout ----
    if lifecycle.stage == "ENTRY_SENT" and lifecycle.entry_order_id:
        # Timeout (blanket)
        try:
            sent = datetime.fromisoformat(lifecycle.entry_sent_ts) if lifecycle.entry_sent_ts else datetime.fromisoformat(lifecycle.created_ts)
            age_min = (now - sent).total_seconds() / 60.0
        except Exception:
            age_min = 0.0

        timeout_m = int(getattr(cfg, "timeout_minutes", ENTRY_TIMEOUT_MINUTES) or ENTRY_TIMEOUT_MINUTES)
        if age_min >= timeout_m:
            try:
                client.cancel_order(account_id_key, lifecycle.entry_order_id)
            except Exception:
                pass
            lifecycle.stage = "CANCELED"
            lifecycle.notes = f"entry_timeout_{timeout_m}m"
            _event_once(cfg, lifecycle, "ENTRY_TIMEOUT", f"[AUTOEXEC] {lifecycle.symbol} ENTRY TIMEOUT", f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nCanceled: entry order timeout ({timeout_m}m).\n")
            _release_pool(state, lifecycle.reserved_dollars)
            return

        status, filled_qty = client.get_order_status_and_filled_qty(account_id_key, lifecycle.entry_order_id)
        lifecycle.notes = f"entry_status={status}"
        prev_filled = int(lifecycle.filled_qty or 0)

        # If we got any fills, we immediately manage the position (even if partial).
        if filled_qty and filled_qty > 0:
            used = float(filled_qty) * float(lifecycle.desired_entry)
            unused = max(0.0, float(lifecycle.reserved_dollars) - used)
            if unused > 0:
                _release_pool(state, unused)
                lifecycle.reserved_dollars = used

            lifecycle.filled_qty = int(filled_qty)
            if int(filled_qty) > int(prev_filled):
                _event_once(
                    cfg,
                    lifecycle,
                    f"FILL_{int(filled_qty)}",
                    f"[AUTOEXEC] {lifecycle.symbol} FILL UPDATE",
                    f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nFilled qty: {int(filled_qty)} (prev {int(prev_filled)})\nEntry: {lifecycle.desired_entry}\nOrderId: {lifecycle.entry_order_id}\n",
                )

            # Place / resize brackets for the filled quantity
            if lifecycle.bracket_qty != int(filled_qty) or not (lifecycle.stop_order_id and lifecycle.tp_order_id):
                # Cancel any existing brackets before resizing
                for oid in [lifecycle.stop_order_id, lifecycle.tp_order_id]:
                    if oid:
                        try:
                            client.cancel_order(account_id_key, oid)
                        except Exception:
                            pass

                lifecycle.stop_order_id = client.place_equity_stop_order(
                    account_id_key=account_id_key,
                    symbol=lifecycle.symbol,
                    qty=int(filled_qty),
                    stop_price=float(lifecycle.stop),
                    action="SELL",
                    market_session="REGULAR",
                )
                lifecycle.tp_order_id = client.place_equity_limit_order(
                    account_id_key=account_id_key,
                    symbol=lifecycle.symbol,
                    qty=int(filled_qty),
                    limit_price=float(lifecycle.tp0),
                    action="SELL",
                    market_session="REGULAR",
                )
                lifecycle.bracket_qty = int(filled_qty)
                lifecycle.notes = f"bracket_sent stop={lifecycle.stop} tp0={lifecycle.tp0} qty={filled_qty}"
                _event_once(cfg, lifecycle, f"BRACKETS_{int(filled_qty)}", f"[AUTOEXEC] {lifecycle.symbol} BRACKETS PLACED", f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nBrackets placed for filled qty {int(filled_qty)}.\nSTOP: {lifecycle.stop} (order {lifecycle.stop_order_id})\nTP0: {lifecycle.tp0} (order {lifecycle.tp_order_id})\n")

            lifecycle.stage = "IN_POSITION"
            return

        # No fills and order died
        if status in {"CANCELLED", "REJECTED", "EXPIRED"}:
            lifecycle.stage = "CANCELED"
            lifecycle.notes = f"entry_{status.lower()}"
            _event_once(cfg, lifecycle, f"ENTRY_{status}", f"[AUTOEXEC] {lifecycle.symbol} ENTRY {status}", f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nEntry order ended with status: {status}.\n")
            _release_pool(state, lifecycle.reserved_dollars)
            return

    # ---- IN POSITION: monitor exits, enforce full exit ----
    if lifecycle.stage == "IN_POSITION":
        # Check TP first
        if lifecycle.tp_order_id:
            tp_status, tp_filled = client.get_order_status_and_filled_qty(account_id_key, lifecycle.tp_order_id)
            # Enforce FULL EXIT even on partial TP fills
            if tp_filled and int(tp_filled) > 0 and int(tp_filled) < int(lifecycle.bracket_qty or 0) and tp_status not in {"EXECUTED", "FILLED", "CANCELLED", "REJECTED", "EXPIRED"}:
                # Cancel remaining TP + protective stop, then market-sell any remaining shares
                try:
                    client.cancel_order(account_id_key, lifecycle.tp_order_id)
                except Exception:
                    pass
                if lifecycle.stop_order_id:
                    try:
                        client.cancel_order(account_id_key, lifecycle.stop_order_id)
                    except Exception:
                        pass
                try:
                    positions = client.get_positions_map(account_id_key)
                    rem = int(positions.get(lifecycle.symbol, 0) or 0)
                except Exception:
                    rem = 0
                if rem > 0:
                    try:
                        client.place_equity_market_order(
                            account_id_key=account_id_key,
                            symbol=lifecycle.symbol,
                            qty=rem,
                            action="SELL",
                            market_session="REGULAR",
                        )
                    except Exception:
                        pass
                lifecycle.stage = "CLOSED"
                lifecycle.notes = "tp0_partial_forced_exit"

                # Record last action for hourly P&L emails
                _set_last_action(
                    state,
                    f"{now.strftime('%H:%M:%S')} ET — {lifecycle.symbol} TP0 partial ({int(tp_filled)}/{int(lifecycle.bracket_qty or 0)}) → mkt sell rem {int(rem)}",
                )

                _event_once(
                    cfg,
                    lifecycle,
                    "TP0_PARTIAL_FORCED",
                    f"[AUTOEXEC] {lifecycle.symbol} TP0 PARTIAL → MARKET EXIT",
                    (
                        f"Time (ET): {now.isoformat()}\n"
                        f"Symbol: {lifecycle.symbol}\n"
                        f"Engine: {lifecycle.engine}\n\n"
                        f"Exit detail: TP0 order {lifecycle.tp_order_id} | status {tp_status} | filled {int(tp_filled)}/{int(lifecycle.bracket_qty or 0)} | market-sell qty {int(rem)}\n\n"
                        f"TP0 order partially filled ({int(tp_filled)}/{int(lifecycle.bracket_qty or 0)}). "
                        f"Forced full exit: canceled remainder + market-sold remaining shares.\n"
                    ),
                )

            if tp_status in {"EXECUTED", "FILLED"}:
                if lifecycle.stop_order_id:
                    try:
                        client.cancel_order(account_id_key, lifecycle.stop_order_id)
                    except Exception:
                        pass
                # Ensure no leftover shares: market-sell remaining qty (best-effort)
                try:
                    positions = client.get_positions_map(account_id_key)
                    rem = int(positions.get(lifecycle.symbol, 0) or 0)
                except Exception:
                    rem = 0
                if rem > 0:
                    try:
                        client.place_equity_market_order(
                            account_id_key=account_id_key,
                            symbol=lifecycle.symbol,
                            qty=rem,
                            action="SELL",
                            market_session="REGULAR",
                        )
                    except Exception:
                        pass
                lifecycle.stage = "CLOSED"
                lifecycle.notes = "tp0_hit"
                _event_once(cfg, lifecycle, "TP0_HIT", f"[AUTOEXEC] {lifecycle.symbol} TP0 EXIT", f"Time (ET): {now.isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nTP0 exit complete.\n")
                return

        if lifecycle.stop_order_id:
            st_status, st_filled = client.get_order_status_and_filled_qty(account_id_key, lifecycle.stop_order_id)
            # Enforce FULL EXIT even on partial STOP fills
            if st_filled and int(st_filled) > 0 and int(st_filled) < int(lifecycle.bracket_qty or 0) and st_status not in {"EXECUTED", "FILLED", "CANCELLED", "REJECTED", "EXPIRED"}:
                try:
                    client.cancel_order(account_id_key, lifecycle.stop_order_id)
                except Exception:
                    pass
                if lifecycle.tp_order_id:
                    try:
                        client.cancel_order(account_id_key, lifecycle.tp_order_id)
                    except Exception:
                        pass
                try:
                    positions = client.get_positions_map(account_id_key)
                    rem = int(positions.get(lifecycle.symbol, 0) or 0)
                except Exception:
                    rem = 0
                if rem > 0:
                    try:
                        client.place_equity_market_order(
                            account_id_key=account_id_key,
                            symbol=lifecycle.symbol,
                            qty=rem,
                            action="SELL",
                            market_session="REGULAR",
                        )
                    except Exception:
                        pass
                lifecycle.stage = "CLOSED"
                lifecycle.notes = "stop_partial_forced_exit"

                # Record last action for hourly P&L emails
                _set_last_action(
                    state,
                    f"{now.strftime('%H:%M:%S')} ET — {lifecycle.symbol} STOP partial ({int(st_filled)}/{int(lifecycle.bracket_qty or 0)}) → mkt sell rem {int(rem)}",
                )

                _event_once(
                    cfg,
                    lifecycle,
                    "STOP_PARTIAL_FORCED",
                    f"[AUTOEXEC] {lifecycle.symbol} STOP PARTIAL → MARKET EXIT",
                    (
                        f"Time (ET): {now.isoformat()}\n"
                        f"Symbol: {lifecycle.symbol}\n"
                        f"Engine: {lifecycle.engine}\n\n"
                        f"Exit detail: STOP order {lifecycle.stop_order_id} | status {st_status} | filled {int(st_filled)}/{int(lifecycle.bracket_qty or 0)} | market-sell qty {int(rem)}\n\n"
                        f"Stop order partially filled ({int(st_filled)}/{int(lifecycle.bracket_qty or 0)}). "
                        f"Forced full exit: canceled remainder + market-sold remaining shares.\n"
                    ),
                )
                return

            if st_status in {"EXECUTED", "FILLED"}:
                if lifecycle.tp_order_id:
                    try:
                        client.cancel_order(account_id_key, lifecycle.tp_order_id)
                    except Exception:
                        pass
                try:
                    positions = client.get_positions_map(account_id_key)
                    rem = int(positions.get(lifecycle.symbol, 0) or 0)
                except Exception:
                    rem = 0
                if rem > 0:
                    try:
                        client.place_equity_market_order(
                            account_id_key=account_id_key,
                            symbol=lifecycle.symbol,
                            qty=rem,
                            action="SELL",
                            market_session="REGULAR",
                        )
                    except Exception:
                        pass
                lifecycle.stage = "CLOSED"
                lifecycle.notes = "stopped_out"
                _event_once(
                    cfg,
                    lifecycle,
                    "STOP_HIT",
                    f"[AUTOEXEC] {lifecycle.symbol} STOP EXIT",
                    (
                        f"Time (ET): {now.isoformat()}\n"
                        f"Symbol: {lifecycle.symbol}\n"
                        f"Engine: {lifecycle.engine}\n\n"
                        f"Stop exit complete.\n"
                    ),
                )
                return
def _force_liquidate_all(cfg: AutoExecConfig, state: Dict[str, Any], fetch_last_price_fn) -> None:
    # One-time daily notice
    if not state.get("skip_notices", {}).get("EOD_LIQUIDATION"):
        state.setdefault("skip_notices", {})["EOD_LIQUIDATION"] = _now_et().isoformat()
        _send_status_email(cfg, "[AUTOEXEC] EOD LIQUIDATION", f"Time (ET): {_now_et().isoformat()}\n\nEnd-of-day liquidation triggered (15:55 ET). Canceling open orders and flattening managed positions.")
    client = ensure_client(cfg)
    if client is None:
        return
    account_id_key = state.get("auth", {}).get("account_id_key")
    if not account_id_key:
        return

    # Cancel any open orders first
    for symbol, lst in list(state.get("lifecycles", {}).items()):
        for idx, raw in enumerate(list(lst)):
            lifecycle = TradeLifecycle(**raw)
            for oid in [lifecycle.entry_order_id, lifecycle.stop_order_id, lifecycle.tp_order_id]:
                if oid:
                    try:
                        client.cancel_order(account_id_key, oid)
                    except Exception:
                        pass
            lifecycle.stage = "CLOSED"
            lifecycle.notes = "eod_cleanup"
            _event_once(cfg, lifecycle, "EOD_CLEANUP", f"[AUTOEXEC] {lifecycle.symbol} EOD CLOSE", f"Time (ET): {_now_et().isoformat()}\nSymbol: {lifecycle.symbol}\nEngine: {lifecycle.engine}\n\nEOD cleanup: orders canceled; flattening positions if any.\n")
            lst[idx] = asdict(lifecycle)

    # Market-sell any remaining positions we opened (best-effort) — scoped to managed symbols only
    managed = set()
    for sym, lst in state.get("lifecycles", {}).items():
        if lst:
            managed.add(sym)
    if not managed:
        return
    try:
        positions = client.get_positions_map(account_id_key)
    except Exception:
        positions = {}
    for symbol, qty in positions.items():
        if symbol not in managed:
            continue
        if qty and qty > 0:
            try:
                client.place_equity_market_order(
                    account_id_key=account_id_key,
                    symbol=symbol,
                    qty=int(qty),
                    action="SELL",
                    market_session="REGULAR",
                )
            except Exception:
                pass
