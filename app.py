import time
import os
import json
import pandas as pd
import numpy as np
import streamlit as st
from email_utils import send_email_alert, format_alert_email
import plotly.graph_objects as go
from typing import List

from av_client import AlphaVantageClient
from engine import scan_watchlist, scan_watchlist_dual, scan_watchlist_triple, scan_watchlist_quad, fetch_bundle
from indicators import vwap as calc_vwap, session_vwap as calc_session_vwap
from signals import compute_scalp_signal, PRESETS
from heavenly_engine import compute_heavenly_signal, HeavenlyConfig

# Auto-exec (E*TRADE) - isolated so it cannot affect engine logic unless enabled
from auto_exec import (
    AutoExecConfig,
    handle_alert_for_autoexec,
    reconcile_and_execute,
    try_send_entries,
)
from etrade_client import ETradeClient
# In-memory (server-side) cache for HEAVENLY 1-minute data.
# IMPORTANT: Do NOT store DataFrames in st.session_state (can cause "Bad session" / serialization issues).
HEAVENLY_1M_CACHE = {}  # {symbol: {"ts": float, "df": pd.DataFrame}}
# -------------------------
# Session-state + Arrow safety utilities
# -------------------------
import math as _math
import datetime as _dt

def _json_sanitize(obj, _depth: int = 0, _max_depth: int = 6):
    """Convert objects to JSON/Streamlit-safe primitives (no DataFrames, no numpy scalars)."""
    if _depth > _max_depth:
        return str(obj)
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        if isinstance(obj, float) and (_math.isnan(obj) or _math.isinf(obj)):
            return None
        return obj
    try:
        import numpy as _np
        import pandas as _pd
        if isinstance(obj, (_np.integer,)):
            return int(obj)
        if isinstance(obj, (_np.floating,)):
            x = float(obj)
            if _math.isnan(x) or _math.isinf(x):
                return None
            return x
        if isinstance(obj, (_pd.Timestamp, _dt.datetime, _dt.date)):
            return str(obj)
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v, _depth + 1, _max_depth) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_sanitize(v, _depth + 1, _max_depth) for v in list(obj)]
    # numpy/pandas arrays
    if hasattr(obj, "tolist"):
        try:
            return _json_sanitize(obj.tolist(), _depth + 1, _max_depth)
        except Exception:
            pass
    return str(obj)

def _arrow_safe_df(df: pd.DataFrame) -> pd.DataFrame:
    """Make a DataFrame safe for st.dataframe (pyarrow) by stringifying dict/list/object payloads."""
    if df is None or df.empty:
        return df
    out = df.copy()
    for col in out.columns:
        s = out[col]
        if str(s.dtype) == "object":
            def _conv(v):
                if v is None:
                    return None
                if isinstance(v, float) and (_math.isnan(v) or _math.isinf(v)):
                    return None
                if isinstance(v, (dict, list, tuple, set)):
                    try:
                        import json as _json
                        return _json.dumps(_json_sanitize(v), ensure_ascii=False)
                    except Exception:
                        return str(v)
                # numpy/pandas scalars
                try:
                    import numpy as _np
                    import pandas as _pd
                    if isinstance(v, (_np.integer, _np.floating, _pd.Timestamp, _dt.datetime, _dt.date)):
                        return _json_sanitize(v)
                except Exception:
                    pass
                return v
            out[col] = s.map(_conv)
        else:
            # coerce numeric NaNs/infs
            try:
                import numpy as _np
                if _np.issubdtype(s.dtype, _np.number):
                    out[col] = pd.to_numeric(s, errors="coerce")
            except Exception:
                pass
    return out


def load_email_secrets():
    """Load email settings from Streamlit Secrets."""
    email_tbl = st.secrets.get("email", {})
    smtp_server = email_tbl.get("smtp_server") or st.secrets.get("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(email_tbl.get("smtp_port") or st.secrets.get("SMTP_PORT", 587))
    smtp_user = email_tbl.get("smtp_user") or st.secrets.get("SMTP_USER", "")
    smtp_password = email_tbl.get("smtp_password") or st.secrets.get("SMTP_APP_PASSWORD", "")
    # Recipients must be provided as a list so we can send *individually*.
    # Example (Streamlit Secrets):
    # [email]
    # to_emails = ["you@domain.com", "team@domain.com"]
    to_emails = email_tbl.get("to_emails") or st.secrets.get("ALERT_TO_EMAILS") or []
    # Allow accidental comma-separated string (common when pasting)
    if isinstance(to_emails, str):
        to_emails = [e.strip() for e in to_emails.split(",") if e.strip()]
    return smtp_server, smtp_port, smtp_user, smtp_password, list(to_emails)

def send_email_safe(payload: dict, smtp_server: str, smtp_port: int, smtp_user: str, smtp_password: str, to_emails: List[str]):
    """Send an email alert and return (ok, err_msg)."""
    if not (smtp_user and smtp_password and to_emails):
        return False, "Missing SMTP secrets"
    try:
        # Defensive: payload can sometimes arrive as a pandas Series.
        if hasattr(payload, "to_dict"):
            payload = payload.to_dict()
        elif not isinstance(payload, dict):
            payload = dict(payload)
        # Payload may arrive as a row dict using either TitleCase or lowercase keys.
        sym = payload.get('Symbol') or payload.get('symbol') or '?' 
        bias = payload.get('Bias') or payload.get('bias') or ''
        stage = payload.get('Tier') or payload.get('tier') or payload.get('Stage') or payload.get('stage') or ''
        stage_tag = f"[{stage}]" if stage else ""

        # Optional alert family tag (e.g., REVERSAL vs RIDE)
        fam = payload.get('SignalFamily') or (payload.get('Extras') or {}).get('family') or payload.get('family')
        fam_tag = f"[{str(fam).upper()}]" if fam else ""

        subject = f"Ztockly Alert {fam_tag}: {sym} {bias} {stage_tag}".replace("  ", " ").strip()
        body = format_alert_email(payload)
        send_email_alert(
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            smtp_user=smtp_user,
            smtp_password=smtp_password,
            to_emails=to_emails,
            subject=subject,
            body=body,
        )
        return True, ""
    except Exception as e:
        return False, str(e)

st.set_page_config(page_title="Ztockly Scalping Scanner", layout="wide")

# Persist results across reruns
st.session_state.setdefault('last_results', None)
st.session_state.setdefault('last_df_view', None)
st.session_state.setdefault('last_scan_ts', None)

if "watchlist" not in st.session_state:
    st.session_state.watchlist = ["AAPL", "NVDA", "TSLA", "SPY", "QQQ"]
if "last_alert_ts" not in st.session_state:
    st.session_state.last_alert_ts = {}
if "pending_confirm" not in st.session_state:
    # per-symbol pending setup waiting for next-bar confirmation (only used when auto-refresh is ON)
    st.session_state.pending_confirm = {}
if "alerts" not in st.session_state:
    st.session_state.alerts = []

# Track per-symbol state so alerts only fire on *new* actionable transitions.
# This prevents re-sending the same alert on every auto-refresh rerun.
if "symbol_state" not in st.session_state:
    # {SYM: {"bias": str, "actionable": bool, "score": float}}
    st.session_state.symbol_state = {}

# Separate state for RIDE/continuation so it doesn't suppress reversal alerts.
if "ride_symbol_state" not in st.session_state:
    # {SYM: {"bias": str, "stage": str, "actionable": bool, "score": float}}
    st.session_state.ride_symbol_state = {}

# Separate state for SWING so it doesn't suppress reversal or ride alerts.
if "swing_symbol_state" not in st.session_state:
    st.session_state.swing_symbol_state = {}

# Separate state for MSS/ICT strict signals.
if "mss_symbol_state" not in st.session_state:
    st.session_state.mss_symbol_state = {}

# Separate state for HEAVENLY so it doesn't suppress other engines.
if "heavenly_symbol_state" not in st.session_state:
    st.session_state.heavenly_symbol_state = {}

st.sidebar.title("Scalping Scanner")
watchlist_text = st.sidebar.text_area("Watchlist (comma or newline separated)", value="\n".join(st.session_state.watchlist), height=150)

interval = st.sidebar.selectbox("Intraday interval", ["1min", "5min"], index=0)
interval_mins = int(interval.replace("min","").strip())
mode = st.sidebar.selectbox("Signal mode", list(PRESETS.keys()), index=list(PRESETS.keys()).index("Cleaner signals"))

st.sidebar.markdown("#### Causality / bar guards")
use_last_closed_only = st.sidebar.toggle("Use last completed bar only (snapshot)", value=True, help="Uses the last fully completed candle for indicator reads.")
bar_closed_guard = st.sidebar.toggle("Bar-closed guard (avoid partial current bar)", value=True, help="Steps back if the latest candle is still forming.")


st.sidebar.markdown("### VWAP")
vwap_logic = st.sidebar.selectbox("VWAP logic for signals", ["session", "cumulative"], index=0)
session_vwap_include_premarket = st.sidebar.toggle("Session VWAP includes Premarket (starts 04:00)", value=False, help="OFF = RTH VWAP reset at 09:30 ET. ON = Extended VWAP starts 04:00 ET.")
show_dual_vwap = st.sidebar.toggle("Dual VWAP (show both lines)", value=True)

st.sidebar.markdown("### Engine complexity")
pro_mode = st.sidebar.toggle("Pro mode", value=True, help="Enables ICT-style diagnostics + extra scoring components.")
entry_model = st.sidebar.selectbox(
    "Entry model",
    ["Last price", "Midpoint (last closed bar)", "VWAP reclaim limit"],
    index=2,
    help="Controls how the app proposes an entry price when a setup is detected."
)

slip_mode = st.sidebar.selectbox(
    "Slippage buffer",
    ["Off", "Fixed cents", "ATR fraction"],
    index=1,
    help="Adds a small buffer to entry to be more realistic for fast/volatile names."
)
slip_fixed_cents = st.sidebar.slider("Fixed slippage (cents)", 0.0, 0.25, 0.02, 0.01)
slip_atr_frac = st.sidebar.slider("ATR fraction slippage", 0.0, 1.0, 0.15, 0.05)


st.sidebar.markdown("### Time-of-day filter (ET)")
allow_opening = st.sidebar.checkbox("Opening (09:30–11:00)", value=True)
allow_midday = st.sidebar.checkbox("Midday (11:00–15:00)", value=False)
allow_power = st.sidebar.checkbox("Power hour (15:00–16:00)", value=True)
allow_premarket = st.sidebar.checkbox("Premarket (04:00–09:30)", value=False)
allow_afterhours = st.sidebar.checkbox("Afterhours (16:00+)", value=False)

# -------------------------
# AUTO-EXEC (E*TRADE) — LONG only (experimental)
# NOTE: This does not change any engine logic; it only listens to alerts that
# already fired and (when enabled) routes them into an order lifecycle.
# -------------------------
with st.sidebar.expander("AUTO‑EXEC (E*TRADE) — LONG only", expanded=False):
    autoexec_enabled = st.toggle(
        "Enable auto‑execution",
        value=False,
        help="If ON, the app may place real orders in your E*TRADE account based on eligible alerts. Default OFF.",
    )
    autoexec_sandbox = st.checkbox("Use E*TRADE SANDBOX", value=True, help="Sandbox first. Switch OFF only when ready.")

    st.markdown("**Capital controls**")
    max_dollars_per_trade = st.number_input("Max $ per trade", min_value=10.0, max_value=10000.0, value=100.0, step=10.0)
    max_pool_dollars = st.number_input(
        "Max $ pool for the day",
        min_value=10.0,
        max_value=100000.0,
        value=1000.0,
        step=50.0,
        help="When reserved capital hits this cap, no more new auto-trades will be staged that day.",
    )

    st.markdown("**Risk + throttles**")
    auto_min_score = st.slider("Min score to auto‑exec", 70, 99, 90, 1)
    auto_timeout_minutes = st.slider(
        "Auto‑exec entry timeout (minutes)",
        5,
        60,
        20,
        1,
        help="How long a STAGED or ENTRY_SENT lifecycle can wait before the bot cancels it. Default 20m.",
    )

    st.markdown("**Execution windows**")
    enforce_entry_windows = st.checkbox(
        "Enforce entry windows for order submission",
        value=True,
        help="Recommended. Prevents sending NEW entry orders outside your selected time windows. This avoids late-day fills on stale setups.",
    )
    entry_grace_minutes = st.slider(
        "Entry grace period (minutes)",
        0,
        10,
        2,
        1,
        help="If an alert was staged inside the window but your next refresh occurs slightly after the window ends, allow entry submission for this many minutes.",
    )
    max_concurrent_symbols = st.slider("Max symbols traded simultaneously", 1, 10, 3, 1)
    lifecycles_per_symbol = st.slider("Lifecycles per symbol per day", 1, 5, 1, 1)
    confirm_only = st.checkbox("Confirm‑only (safer)", value=True, help="If ON, only CONFIRMED stages can stage auto‑exec.")
    status_emails = st.checkbox(
        "Status emails (auto‑exec lifecycle)",
        value=True,
        help="Sends emails when auto‑exec stages, places orders, fills, and exits (one per event).",
    )
    hourly_pnl_emails = st.checkbox(
        "Hourly P&L checkpoint emails (9:30–4:00 ET)",
        value=False,
        help="Sends an informational P&L + simple bot analytics email once per hour during the regular session (requires OAuth to be active).",
    )
    tp0_deviation = st.slider(
        "TP0 deviation (sell early)",
        0.0,
        0.15,
        0.01,
        0.01,
        help="0.00 = place exit exactly at TP0. 0.01 means place exit at TP0 - $0.01 (longs).",
    )

    engines_for_auto = st.multiselect(
        "Engines allowed for auto‑exec",
        ["RIDE", "SCALP", "HEAVENLY"],
        default=["RIDE"],
        help="Start with RIDE only. You can add SCALP/HEAVENLY later.",
    )

    st.markdown("**E*TRADE authentication**")
    etrade_ck = st.secrets.get("etrade", {}).get("consumer_key", "")
    etrade_cs = st.secrets.get("etrade", {}).get("consumer_secret", "")
    if not etrade_ck or not etrade_cs:
        st.warning("Missing E*TRADE API keys in secrets. Add [etrade] consumer_key / consumer_secret.")
    else:
        st.caption("You can re-auth each morning. Tokens are kept in session_state only.")
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Start auth", use_container_width=True, disabled=not autoexec_enabled):
                try:
                    c = ETradeClient(etrade_ck, etrade_cs, sandbox=autoexec_sandbox)
                    req = c.get_request_token()
                    st.session_state.setdefault("autoexec", {}).setdefault("auth", {})
                    st.session_state["autoexec"]["auth"].update(
                        {
                            "consumer_key": etrade_ck,
                            "consumer_secret": etrade_cs,
                            "sandbox": bool(autoexec_sandbox),
                            "request_token": req.oauth_token,
                            "request_token_secret": req.oauth_token_secret,
                            "authorize_url": c.get_authorize_url(req.oauth_token),
                        }
                    )
                except Exception as e:
                    st.error(f"Auth start failed: {e}")

        auth = st.session_state.get("autoexec", {}).get("auth", {})
        auth_url = auth.get("authorize_url")
        if auth_url:
            st.markdown(f"Authorize here: {auth_url}")
            verifier = st.text_input("Enter verifier (PIN)", value="", disabled=not autoexec_enabled)
            with col_b:
                if st.button("Finish auth", use_container_width=True, disabled=not autoexec_enabled):
                    try:
                        c = ETradeClient(etrade_ck, etrade_cs, sandbox=autoexec_sandbox)
                        tok = c.get_access_token(
                            auth["request_token"], auth["request_token_secret"], verifier.strip()
                        )
                        # Bind access tokens for subsequent calls
                        c2 = ETradeClient(
                            etrade_ck,
                            etrade_cs,
                            sandbox=autoexec_sandbox,
                            access_token=tok.oauth_token,
                            access_token_secret=tok.oauth_token_secret,
                        )
                        acct = c2.list_accounts()
                        accounts = (
                            acct.get("AccountListResponse", {}).get("Accounts", {}).get("Account", [])
                            if isinstance(acct, dict)
                            else []
                        )
                        if isinstance(accounts, dict):
                            accounts = [accounts]
                        # Choose the first brokerage account by default
                        account_id_key = ""
                        for a in accounts:
                            if str(a.get("accountType", "")).upper() in {"BROKERAGE", "MARGIN", "CASH"}:
                                account_id_key = str(a.get("accountIdKey", ""))
                                break
                        if not account_id_key and accounts:
                            account_id_key = str(accounts[0].get("accountIdKey", ""))

                        st.session_state.setdefault("autoexec", {}).setdefault("auth", {})
                        st.session_state["autoexec"]["auth"].update(
                            {
                                "consumer_key": etrade_ck,
                                "consumer_secret": etrade_cs,
                                "sandbox": bool(autoexec_sandbox),
                                "access_token": tok.oauth_token,
                                "access_token_secret": tok.oauth_token_secret,
                                "account_id_key": account_id_key,
                            }
                        )
                        st.success("E*TRADE auth complete.")
                    except Exception as e:
                        st.error(f"Auth finish failed: {e}")

        # Lightweight status
        if st.session_state.get("autoexec", {}).get("auth", {}).get("account_id_key"):
            st.info("Auto‑exec is authenticated and ready (account bound).")

    autoexec_cfg = AutoExecConfig(
        enabled=bool(autoexec_enabled),
        sandbox=bool(autoexec_sandbox),
        engines=tuple([e for e in engines_for_auto if e]),
        min_score=float(auto_min_score),
        max_dollars_per_trade=float(max_dollars_per_trade),
        max_pool_dollars=float(max_pool_dollars),
        max_concurrent_symbols=int(max_concurrent_symbols),
        lifecycles_per_symbol_per_day=int(lifecycles_per_symbol),
        timeout_minutes=int(auto_timeout_minutes),
        tp0_deviation=float(tp0_deviation),
        confirm_only=bool(confirm_only),
        status_emails=bool(status_emails),
        hourly_pnl_emails=bool(hourly_pnl_emails),
        enforce_entry_windows=bool(enforce_entry_windows),
        entry_grace_minutes=int(entry_grace_minutes),
    )
    st.session_state["autoexec_cfg"] = autoexec_cfg

st.sidebar.markdown("#### Killzone presets")
killzone_preset = st.sidebar.selectbox(
    "Killzone preset",
    ["Custom (use toggles)", "Opening Drive", "Lunch Chop", "Power Hour", "Pre-market"],
    index=0,
    help="Quick presets that bias scoring + optionally constrain time windows."
)
liquidity_weighting = st.sidebar.slider(
    "Liquidity-weighted scoring (0–1)",
    0.0, 1.0, 0.55, 0.05,
    help="Boosts scoring during higher-liquidity windows (open/close) and de-emphasizes lunch chop."
)
orb_minutes = st.sidebar.slider(
    "ORB window (minutes)",
    5, 60, 15, 5,
    help="Opening Range Breakout window used to compute ORB high/low levels."
)


st.sidebar.markdown("### Higher‑TF bias overlay (optional)")
enable_htf = st.sidebar.toggle("Enable HTF bias", value=False)
htf_interval = st.sidebar.selectbox("HTF interval", ["15min", "30min"], index=0, disabled=not enable_htf)
htf_strict = st.sidebar.checkbox("Strict HTF alignment", value=False, disabled=not enable_htf)

st.sidebar.markdown("### ATR score normalization")
atr_norm_mode = st.sidebar.selectbox("ATR normalization", ["Auto (per ticker)", "Manual"], index=0, help="Auto uses each ticker's recent median ATR% as its baseline so high-vol names aren't punished.")
if atr_norm_mode == "Manual":
    target_atr_pct = st.sidebar.slider("Target ATR% (score normalization)", 0.001, 0.020, 0.004, 0.001, format="%.3f")
else:
    target_atr_pct = None

st.sidebar.markdown("### Fib logic")
show_fibs = st.sidebar.checkbox("Show Fibonacci retracement", value=True)
fib_lookback = st.sidebar.slider("Fib lookback bars", 60, 240, 120, 10) if show_fibs else 120

st.sidebar.markdown("### In‑App Alerts")
cooldown_minutes = st.sidebar.slider("Cooldown minutes (per ticker)", 1, 30, 7, 1)
alert_threshold = st.sidebar.slider("Alert score threshold", 60, 100, int(PRESETS[mode]["min_actionable_score"]), 1)

st.sidebar.markdown("### 💫 HEAVENLY engine")
enable_heavenly = st.sidebar.toggle(
    "Enable HEAVENLY (new engine)",
    value=True,
    help="Separate, high-selectivity swing/expansion engine. Does not alter SCALP/RIDE/SWING/MSS logic.",
)
heavenly_htf = st.sidebar.selectbox(
    "HEAVENLY HTF (suppression)",
    ["30min", "60min"],
    index=0,
    disabled=not enable_heavenly,
    help="30m reacts faster (better for small/mid-caps). 60m is stricter and slower.",
)
heavenly_conditional_1m = st.sidebar.toggle(
    "HEAVENLY conditional 1m intent",
    value=True,
    disabled=not enable_heavenly,
    help="Fetches 1m only when price is near the TSZ and a setup/trigger is possible.",
)
heavenly_min_evs = st.sidebar.slider(
    "HEAVENLY min EVS (ATR)",
    1.0, 4.0, 2.0, 0.25,
    disabled=not enable_heavenly,
    help="Minimum Expected Value Span (room to next obstacle) in ATR.",
)
heavenly_prox = st.sidebar.slider(
    "HEAVENLY proximity to TSZ (ATR)",
    0.25, 2.0, 0.75, 0.25,
    disabled=not enable_heavenly,
    help="How close price must be to the TSZ to upgrade WATCH → SETUP.",
)

# Debug/product controls: makes it obvious when cooldown is suppressing email alerts.
col_cd1, col_cd2 = st.sidebar.columns(2)
with col_cd1:
    if st.button("Clear cooldowns", use_container_width=True):
        st.session_state.last_alert_ts = {}
        st.sidebar.success("Cooldowns cleared")
with col_cd2:
    if st.button("Clear signal state", use_container_width=True):
        st.session_state.symbol_state = {}
        st.session_state.pending_confirm = {}
        st.sidebar.success("Signal state cleared")
# Bias strictness tuning
st.sidebar.markdown("#### Bias strictness")
bias_strictness = st.sidebar.slider(
    "Bias strictness (looser ↔ stricter)",
    0.0, 1.0, 0.65, 0.05,
    help="Higher = fewer signals, stronger confirmation requirements."
)
split_long_short = st.sidebar.toggle(
    "Separate LONG vs SHORT thresholds",
    value=False,
    help="If enabled, you can require different score thresholds for LONG vs SHORT."
)
long_threshold = int(alert_threshold)
short_threshold = int(alert_threshold)
if split_long_short:
    long_threshold = st.sidebar.slider("LONG score threshold", 50, 99, int(alert_threshold), 1)
    short_threshold = st.sidebar.slider("SHORT score threshold", 50, 99, int(alert_threshold), 1)

capture_alerts = st.sidebar.checkbox("Capture alerts in-app", value=True)
max_alerts_kept = st.sidebar.slider("Max alerts kept", 10, 300, 60, 10)
smtp_server, smtp_port, smtp_user, smtp_password, to_emails = load_email_secrets()
enable_email_alerts = st.sidebar.toggle(
    "Send email alerts",
    value=False,
    help="Sends alerts via Gmail SMTP (requires Secrets). You can keep in-app alerts ON too.",
)

# If the user turns email alerts ON mid-session, arm the system so already-actionable
# rows can trigger on the next scan (instead of requiring a fresh transition).
prev_email_enabled = bool(st.session_state.get("_email_enabled_prev", False))
if enable_email_alerts and not prev_email_enabled:
    # Reset per-symbol state so threshold/actionable crossings are re-evaluated.
    st.session_state.symbol_state = {}
    st.session_state.last_alert_ts = {}
st.session_state["_email_enabled_prev"] = bool(enable_email_alerts)

if enable_email_alerts:
    if not (smtp_user and smtp_password and to_emails):
        st.sidebar.warning('Email is ON but Secrets are missing. Add [email] smtp_user/smtp_password/to_emails in Streamlit Secrets.')
    else:
        st.sidebar.success(f"Email enabled → {', '.join(to_emails)}")




st.sidebar.markdown("### API pacing / refresh")
# Premium entitlement: ensures intraday candles are real-time if your plan supports it.
env_ent = (os.getenv("ALPHAVANTAGE_ENTITLEMENT") or os.getenv("AV_ENTITLEMENT") or "").strip()
entitlement_ui = st.sidebar.selectbox(
    "Alpha Vantage entitlement",
    ["(auto)" if env_ent else "(none)", "realtime", "delayed"],
    index=0,
    help="Premium customers should use 'realtime' so intraday candles don't look like yesterday's feed. '(auto)' uses ALPHAVANTAGE_ENTITLEMENT env var if set."
)
min_between_calls = st.sidebar.slider("Seconds between API calls", 0.5, 8.0, 1.5, 0.5)
auto_refresh = st.sidebar.checkbox("Auto-refresh scanner", value=False)
refresh_seconds = st.sidebar.slider("Refresh every (seconds)", 10, 180, 30, 5) if auto_refresh else None

st.sidebar.markdown("---")
st.sidebar.caption("Required env var: ALPHAVANTAGE_API_KEY")

symbols = [s.strip().upper() for s in watchlist_text.replace(",", "\n").splitlines() if s.strip()]
st.session_state.watchlist = symbols

st.title("Ztockly — Intraday Reversal Scalping Engine (v7)")
st.caption("Basic: VWAP + RSI‑5 event + MACD histogram turn + volume. Pro adds sweeps/OB/breaker/FVG/EMA. v7 adds fib‑anchored TPs, liquidity‑weighted scoring, ATR score normalization, and optional HTF bias.")

@st.cache_resource
def get_client(min_seconds_between_calls: float, entitlement_choice: str):
    client = AlphaVantageClient()
    client.cfg.min_seconds_between_calls = float(min_seconds_between_calls)
    # Allow UI override, otherwise rely on env var configured in AlphaVantageClient.
    try:
        if entitlement_choice == "realtime":
            client.cfg.entitlement = "realtime"
        elif entitlement_choice == "delayed":
            client.cfg.entitlement = "delayed"
        else:
            # (auto)/(none): keep whatever av_client initialized
            pass
    except Exception:
        pass
    return client

client = get_client(min_between_calls, entitlement_ui)

def _now_label() -> str:
    return pd.Timestamp.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

def can_alert(key: str, now_ts: float, cooldown_min: int) -> bool:
    """Cooldown guard. `key` can be a symbol or a symbol+family composite."""
    last = st.session_state.last_alert_ts.get(key)
    if last is None:
        return True
    return (now_ts - float(last)) >= cooldown_min * 60.0

def add_in_app_alert(row: dict) -> None:
    alert = {
        "ts_unix": time.time(),
        "time": _now_label(),
        "symbol": row["Symbol"],
        "bias": row["Bias"],
        "score": int(row["Score"]),
        "session": row.get("Session"),
        "last": row.get("Last"),
        "tier": row.get("Tier"),
        "entry_limit": row.get("Entry"),
        "entry": row.get("Entry"),
        "entry_chase_line": row.get("Chase"),
        "stop": row.get("Stop"),
        "tp0": row.get("TP0"),
        "t1": row.get("TP1"),
        "tp1": row.get("TP1"),
        "t2": row.get("TP2"),
        "tp2": row.get("TP2"),
        "t3": row.get("TP3"),
        "tp3": row.get("TP3"),
        "eta_tp0_min": row.get("ETA TP0 (min)"),
        "why": row.get("Why"),
        "as_of": row.get("AsOf"),
        "mode": mode,
        "interval": interval,
        "pro_mode": pro_mode,
        "extras": _json_sanitize(row.get("Extras", {})),
    }
    st.session_state.alerts.insert(0, alert)
    st.session_state.alerts = st.session_state.alerts[: int(max_alerts_kept)]

def render_alerts_panel():
    st.subheader("🚨 Live Alerts")
    left, right = st.columns([2, 1])

    def _fmt(v):
        """Safe numeric formatter for alert fields that may be missing/None."""
        return f"{float(v):.4f}" if isinstance(v, (int, float)) else "—"

    with right:
        st.metric("Alerts stored", len(st.session_state.alerts))
        if st.button("Clear alerts", type="secondary"):
            st.session_state.alerts = []
            st.session_state.last_alert_ts = {}
            st.rerun()
        st.markdown("**Filters**")
        f_bias = st.multiselect("Bias", ["LONG", "SHORT"], default=["LONG", "SHORT"])
        min_score = st.slider("Min score", 0, 100, 80, 1)

    with left:
        # Be defensive: alerts can come from multiple producers (scan, test email, legacy state).
        alerts = [
            a for a in st.session_state.alerts
            if (a.get("bias") in f_bias) and (float(a.get("score") or 0) >= float(min_score))
        ]
        if not alerts:
            st.info("No alerts yet. Turn on auto-refresh + capture alerts, then let it scan.")
            return

        for a in alerts[:30]:
            bias = (a.get("bias") or "").upper()
            badge = "🟢" if bias == "LONG" else "🔴"
            pro_badge = "⚡ Pro" if a.get("pro_mode") else "🧱 Basic"
            title = f"{badge} **{a.get('symbol','?')}** — **{bias or '—'}** — Score **{a.get('score','—')}** ({a.get('session','')}) • {pro_badge}"
            with st.container(border=True):
                st.markdown(title)
                cols = st.columns(7)
                cols[0].metric("Last", _fmt(a.get("last")))
                cols[1].metric("Entry", _fmt(a.get("entry")))
                cols[2].metric("Stop", _fmt(a.get("stop")))
                cols[3].metric("TP0", _fmt(a.get("tp0") if a.get("tp0") is not None else a.get("t1")))
                cols[4].metric("TP1", _fmt(a.get("tp1") if a.get("tp1") is not None else a.get("t2")))
                cols[5].metric("TP2", _fmt(a.get("tp2") if a.get("tp2") is not None else a.get("tp3") if a.get("tp3") is not None else a.get("t3")))
                fib_tp1 = (a.get("extras") or {}).get("fib_tp1")
                cols[6].metric("Fib TP1", f"{fib_tp1:.4f}" if isinstance(fib_tp1, (float,int)) else "—")
                st.caption(
                    f"{a.get('time','')} • interval={a.get('interval','')} • mode={a.get('mode','')} "
                    f"• VWAP={(a.get('extras') or {}).get('vwap_logic')} "
                    f"• liquidity={(a.get('extras') or {}).get('liquidity_phase')} "
                    f"• as_of={a.get('as_of')}"
                )
                st.write(a.get("why") or "")

                ex = a.get("extras") or {}
                chips = []
                if ex.get("bull_liquidity_sweep"): chips.append("Liquidity sweep (low)")
                if ex.get("bear_liquidity_sweep"): chips.append("Liquidity sweep (high)")
                if ex.get("bull_ob_retest"): chips.append("Bull OB retest")
                if ex.get("bear_ob_retest"): chips.append("Bear OB retest")
                if ex.get("bull_breaker_retest"): chips.append("Bull breaker retest")
                if ex.get("bear_breaker_retest"): chips.append("Bear breaker retest")
                if ex.get("fib_near_long") or ex.get("fib_near_short"): chips.append("Near Fib")
                if ex.get("htf_bias_value") in ("BULL","BEAR"): chips.append(f"HTF {ex.get('htf_bias_value')}")
                if chips:
                    st.markdown("**Chips:** " + " • ".join([f"`{c}`" for c in chips]))
                with st.expander("Raw payload"):
                    st.json(a)

tab_scan, tab_alerts = st.tabs(["📡 Scanner", "🚨 Alerts"])

with tab_alerts:
    render_alerts_panel()


def run_scan():
    """One full scan pass.

    Returns (reversal_results, ride_results, swing_results, mss_results).
    """
    if not symbols:
        st.warning("Add at least one ticker to your watchlist.")
        return [], [], [], []
    with st.spinner("Scanning watchlist..."):
        return scan_watchlist_quad(
            client, symbols,
            interval=interval,
            mode=mode,
            pro_mode=pro_mode,
            allow_opening=allow_opening,
            allow_midday=allow_midday,
            allow_power=allow_power,
            allow_premarket=allow_premarket,
            allow_afterhours=allow_afterhours,
            vwap_logic=vwap_logic,
            session_vwap_include_premarket=session_vwap_include_premarket,
            fib_lookback_bars=fib_lookback,
            enable_htf_bias=enable_htf,
            htf_interval=htf_interval,
            htf_strict=htf_strict,
            target_atr_pct=target_atr_pct,
            use_last_closed_only=use_last_closed_only,
            bar_closed_guard=bar_closed_guard,
            killzone_preset=killzone_preset,
            liquidity_weighting=liquidity_weighting,
            orb_minutes=orb_minutes,
            entry_model=entry_model,
            slippage_mode=slip_mode,
            fixed_slippage_cents=slip_fixed_cents,
            atr_fraction_slippage=slip_atr_frac,
        )


with tab_scan:
    col_a, col_b, col_c, col_d = st.columns([1, 1, 2, 1])
    with col_a:
        scan_now = st.button("Scan Watchlist", type="primary")
    with col_b:
        if st.button("Capture test alert", use_container_width=True):
            test = {
                "Symbol": "TEST",
                "Bias": "LONG",
                "Tier": "CONFIRMED",
                "Score": 95,
                "Session": "TEST",
                "Last": 100.00,
                "Entry": 100.00,
                "Stop": 99.50,
                "TP0": 100.25,
                "TP1": 100.50,
                "TP2": 101.00,
                "Why": "Test alert (wiring check).",
                "AsOf": pd.Timestamp.utcnow().isoformat(),
                "Time": pd.Timestamp.utcnow().isoformat(),
                "Extras": {"family": "REV"},
            }

            if capture_alerts:
                add_in_app_alert(test)
                st.success("Test alert captured in-app.")
            else:
                st.info("In-app capture is OFF; test alert not stored.")

            if enable_email_alerts:
                ok, err = send_email_safe(test, smtp_server, smtp_port, smtp_user, smtp_password, to_emails)
                if ok:
                    st.success(f"Test email sent to {', '.join(to_emails)}.")
                else:
                    st.error(f"Test email failed: {err}")
    with col_c:
        st.write("Tip: Keep watchlist small (5–15) to stay within API limits.")
    with col_d:
        st.write(f"Now: {_now_label()}")

    # --- Scan driver ---
    results_rev = st.session_state.get("last_results_rev", [])
    results_ride = st.session_state.get("last_results_ride", [])
    results_swing = st.session_state.get("last_results_swing", [])
    results_mss = st.session_state.get("last_results_mss", [])
    results_heavenly = st.session_state.get("last_results_heavenly", [])
    if scan_now or auto_refresh:
        # --- Auto-exec reconciliation (does not affect engine logic) ---
        try:
            _aecfg = st.session_state.get("autoexec_cfg")
            if _aecfg is not None and getattr(_aecfg, "enabled", False):
                _px_client = AlphaVantageClient()  # uses env key; throttled internally
                reconcile_and_execute(_aecfg, allow_premarket, allow_opening, allow_midday, allow_powerhour, allow_afterhours, lambda sym: _px_client.fetch_quote(sym))
                # Place entries only when price reaches the desired area
                try_send_entries(_aecfg, allow_opening, allow_midday, allow_powerhour, lambda sym: _px_client.fetch_quote(sym))
        except Exception as _e:
            # Never crash the app because of auto-exec; surface in sidebar via Streamlit logs
            st.sidebar.warning(f"Auto-exec warning: {_e}")

        results_rev, results_ride, results_swing, results_mss = run_scan()
        st.session_state["last_results_rev"] = results_rev
        st.session_state["last_results_ride"] = results_ride
        st.session_state["last_results_swing"] = results_swing
        st.session_state["last_results_mss"] = results_mss

        # HEAVENLY: computed separately so it cannot affect other engines' logic.
        if enable_heavenly:
            now_ts = time.time()
            htf_interval = "30min" if heavenly_htf == "30min" else "60min"

            cfg = HeavenlyConfig(
                enable=True,
                allow_opening=allow_opening,
                allow_midday=allow_midday,
                allow_power=allow_power,
                allow_premarket=allow_premarket,
                allow_afterhours=allow_afterhours,
                session_vwap_include_premarket=session_vwap_include_premarket,
                session_vwap_include_afterhours=False,
                price_to_zone_proximity_atr=float(heavenly_prox),
                min_evs=float(heavenly_min_evs),
            )

            def _get_1m(symbol: str) -> pd.DataFrame:
                # cache per symbol (avoids needless 1m calls on every rerun)
                c = HEAVENLY_1M_CACHE.get(symbol)
                if c and (now_ts - float(c.get("ts") or 0)) <= float(cfg.one_min_ttl_seconds):
                    dfc = c.get("df")
                    if isinstance(dfc, pd.DataFrame):
                        return dfc
                df1 = client.fetch_intraday(symbol, interval="1min", outputsize="full")
                HEAVENLY_1M_CACHE[symbol] = {"ts": now_ts, "df": df1}
                return df1

            h_rows: List[dict] = []
            for sym in symbols:
                try:
                    df5 = client.fetch_intraday(sym, interval="5min", outputsize="full")
                    dfh = client.fetch_intraday(sym, interval=htf_interval, outputsize="full")

                    # First pass (no 1m) so we can decide whether 1m is warranted.
                    p = compute_heavenly_signal(sym, df_5m=df5, df_30m=dfh, df_1m=None, cfg=cfg, now_ts=now_ts)

                    need_1m = False
                    try:
                        ex = p.get("extras") or {}
                        dist = float(ex.get("distance_to_zone_atr") or 9e9)
                        stg = (p.get("stage") or "").upper()
                        if stg in ("SETUP", "ENTRY"):
                            need_1m = True
                        elif dist <= 1.0:
                            need_1m = True
                    except Exception:
                        need_1m = False

                    if heavenly_conditional_1m and need_1m:
                        df1 = _get_1m(sym)
                        p = compute_heavenly_signal(sym, df_5m=df5, df_30m=dfh, df_1m=df1, cfg=cfg, now_ts=now_ts)

                    h_rows.append(p)
                except Exception as e:
                    h_rows.append({"symbol": sym, "family": "HEAVENLY", "stage": "OFF", "bias": "NEUTRAL", "score": 0, "why": f"Error: {e}"})

            results_heavenly = h_rows
            st.session_state["last_results_heavenly"] = results_heavenly
        else:
            st.session_state["last_results_heavenly"] = []
            results_heavenly = []

    if results_rev:


        # Build ranked table
        df = pd.DataFrame([{
            "Symbol": r.symbol,
            "Bias": r.bias,
            # UI-friendly label: PRE vs CONFIRMED
            "Tier": (r.extras or {}).get("stage"),
            "Actionable": (r.bias in ["LONG", "SHORT"] and (r.extras or {}).get("stage") in ("PRE", "CONFIRMED")),
            # Product rule: never hide the score.
            # Actionability is expressed by Bias/Actionable + Entry/Stop/TP.
            "Score": int(r.setup_score),
            "Potential": int(r.setup_score),
            "Session": r.session,
            "Last": r.last_price,
            "Entry": (r.extras or {}).get("entry_limit", r.entry),
            "Chase": (r.extras or {}).get("entry_chase_line"),
            "Stop": r.stop,
            "TP0": (r.extras or {}).get("tp0"),
            "TP1": r.target_1r,
            "TP2": r.target_2r,
            "TP3": (r.extras or {}).get("tp3"),
            "ETA TP0 (min)": (r.extras or {}).get("eta_tp0_min"),
            "ATR%": (r.extras or {}).get("atr_pct"),
            "ATR baseline%": (r.extras or {}).get("atr_ref_pct"),
            "Score scale": (r.extras or {}).get("atr_score_scale"),
            "Why": r.reason,
            # Show the candle timestamp (ET) and help diagnose stale feeds.
            "AsOf": str(r.timestamp) if r.timestamp is not None else None,
            "Extras": r.extras,
        } for r in results_rev])

        # Data freshness diagnostics (helps catch stale intraday feeds).
        def _age_minutes(ts):
            try:
                t = pd.to_datetime(ts)
                # treat naive as ET (Alpha Vantage timestamps are US/Eastern strings)
                if t.tzinfo is None:
                    t = t.tz_localize("America/New_York")
                else:
                    t = t.tz_convert("America/New_York")
                now_et = pd.Timestamp.now(tz="America/New_York")
                return float((now_et - t).total_seconds() / 60.0)
            except Exception:
                return None

        df["Data age (min)"] = df["AsOf"].map(_age_minutes)

        try:
            oldest = df["Data age (min)"].dropna().max()
        except Exception:
            oldest = None
        if isinstance(oldest, (float, int)) and oldest >= 30:
            st.warning(
                f"Heads up: intraday feed looks stale (oldest AsOf is ~{oldest:.0f} min ago). "
                "This can happen with free/delayed APIs, rate limits, or extended-hours gaps. "
                "Scores may still rank, but actionability/alerts can be misleading until data refreshes."
            )

        # Styling: color scale column + per-row tooltip explaining normalization
        df_view = df.drop(columns=["Extras"]).copy()

        def _scale_tooltip(row):
            atrp = row.get("ATR%")
            basep = row.get("ATR baseline%")
            sc = row.get("Score scale")
            if isinstance(atrp, (float, int)) and isinstance(basep, (float, int)) and isinstance(sc, (float, int)):
                return (
                    f"Score normalized because ATR% differs from baseline. "
                    f"Current ATR%={atrp:.3f}, Baseline={basep:.3f}. "
                    f"Scale={sc:.2f} (clipped to 0.75–1.25)."
                )
            return "No ATR normalization data."

        # Add human-readable sanity check columns (Streamlit Cloud safe)
        df_view["Scale note"] = df_view.apply(_scale_tooltip, axis=1)

        def _flag(sc):
            try:
                x = float(sc)
            except Exception:
                return ""
            if x < 0.90:
                return "🔻 scaled down"
            if x > 1.10:
                return "🔺 scaled up"
            return "•"

        df_view["Scale flag"] = df_view["Score scale"].map(_flag)

        st.subheader("Ranked Setups")
        df_view = _arrow_safe_df(df_view)
        st.dataframe(
            df_view,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Tier": st.column_config.TextColumn("Tier", help="PRE = early heads-up; CONFIRMED = full confluence."),
                "Score": st.column_config.ProgressColumn("Score", min_value=0, max_value=100),
                "Potential": st.column_config.NumberColumn("Potential", format="%d", help="Unblocked scoring potential before hard requirements are satisfied."),
                "Actionable": st.column_config.CheckboxColumn("Actionable", help="True only when the engine can produce Entry/Stop/TP (bias LONG/SHORT)."),
                "Data age (min)": st.column_config.NumberColumn("Data age (min)", format="%.0f", help="How old the most recent candle is (ET). Large values usually mean delayed/stale feed."),
                "ATR%": st.column_config.NumberColumn("ATR%", format="%.3f"),
                "ATR baseline%": st.column_config.NumberColumn("ATR baseline%", format="%.3f"),
                "Entry": st.column_config.NumberColumn("Entry", format="%.4f"),
                "Chase": st.column_config.NumberColumn("Chase", format="%.4f", help="If price crosses this line, you're late — reassess execution."),
                "TP0": st.column_config.NumberColumn("TP0", format="%.4f", help="Nearest structure/liquidity target."),
                "TP3": st.column_config.NumberColumn("TP3", format="%.4f", help="Runner target based on expected excursion (MFE p95) for similar historical signals."),
                "ETA TP0 (min)": st.column_config.NumberColumn("ETA TP0 (min)", format="%.0f", help="Rough minutes-to-TP0 using ATR + liquidity phase."),
                "Score scale": st.column_config.NumberColumn(
                    "Score scale",
                    format="%.2f",
                    help="ATR normalization scale. <0.90 means more volatile than baseline (score scaled down). >1.10 means less volatile than baseline (score scaled up).",
                ),
                "Scale flag": st.column_config.TextColumn("Scale", help="Quick visual: scaled up/down based on ATR normalization."),
                "Scale note": st.column_config.TextColumn("Scale note", help="Why this ticker was scaled (sanity-check ATR normalization)."),
            },
        )
        # --- Ride / Continuation table ---
        st.subheader("🚀 Drive / Continuation (RIDE)")
        if results_ride:
            ride_rows = []
            for rr in results_ride:
                # results_ride may contain either our Signal object, a plain dict,
                # or (rarely) None if a fetch failed mid-scan. Be defensive.
                if rr is None:
                    continue
                if hasattr(rr, "to_dict"):
                    d = rr.to_dict()
                elif isinstance(rr, dict):
                    d = rr
                else:
                    # Fallback for simple namespace-like objects
                    d = getattr(rr, "__dict__", {}) or {}
                ex = d.get("extras") or {}
                ride_rows.append({
                    "Symbol": d.get("symbol"),
                    "Bias": d.get("bias"),
                    "Stage": ex.get("stage") or "—",
                    "Score": d.get("setup_score"),
                    "Last": d.get("last_price"),
                    "PullbackEntry": ex.get("pullback_entry") or d.get("entry"),
                    "BreakTrigger": ex.get("break_trigger"),
                    "Stop": d.get("stop"),
                    "TP0": ex.get("tp0") or d.get("target_1r"),
                    "TP1": ex.get("tp1") or d.get("target_2r"),
                    "ETA_TP0_min": ex.get("eta_tp0_min"),
                    "Why": d.get("reason"),
                })
            ride_df = pd.DataFrame(ride_rows)
            ride_df = _arrow_safe_df(ride_df)
            st.dataframe(
                ride_df,
                use_container_width=True,
                hide_index=True,
                height=min(520, 34 + 30 * (len(ride_df) + 1)),
            )
        else:
            st.caption("No RIDE setups (or fetch errors).")
        # --- Swing / Intraday Swing table ---
        st.subheader("🧭 Intraday Swing (SWING)")
        if results_swing:
            swing_rows = []
            for ss in results_swing:
                if ss is None:
                    continue
                if hasattr(ss, "to_dict"):
                    d = ss.to_dict()
                elif isinstance(ss, dict):
                    d = ss
                else:
                    d = getattr(ss, "__dict__", {}) or {}

                ex = d.get("extras") or {}
                pb = ex.get("pullback_band")
                if (
                    isinstance(pb, (tuple, list))
                    and len(pb) == 2
                    and all(isinstance(x, (int, float, np.number)) for x in pb)
                ):
                    pb_disp = f"{float(min(pb)):.4f}–{float(max(pb)):.4f}"
                elif pb is None:
                    pb_disp = "—"
                else:
                    pb_disp = str(pb)

                swing_rows.append({
                    "Symbol": d.get("symbol"),
                    "Bias": d.get("bias"),
                    "SwingStage": ex.get("swing_stage") or ("ENTRY" if (ex.get("stage")=="CONFIRMED") else ("WATCH" if (ex.get("stage")=="PRE") else "OFF")),
                    "AlertStage": ex.get("stage") or "—",
                    "Score": d.get("setup_score"),
                    "PullbackBand": pb_disp,
                    "BreakTrigger": ex.get("break_trigger"),
                    "Entry": ex.get("pullback_entry") or d.get("entry"),
                    "Stop": d.get("stop"),
                    "TP0": ex.get("tp0") or d.get("target_1r"),
                    "TP1": ex.get("tp1") or d.get("target_2r"),
                    "TP2": ex.get("tp2"),
                    "ETA_TP0_min": ex.get("eta_tp0_min"),
                    "Why": d.get("reason"),
                })

            swing_df = pd.DataFrame(swing_rows)
            swing_df = _arrow_safe_df(swing_df)
            st.dataframe(
                swing_df,
                use_container_width=True,
                hide_index=True,
                height=min(520, 34 + 30 * (len(swing_df) + 1)),
            )
        else:
            st.caption("No SWING setups (or fetch errors).")

        # --- HEAVENLY table ---
        st.subheader("💫 Heavenly (HEAVENLY)")
        if results_heavenly:
            h_rows = []
            for p in results_heavenly:
                if p is None:
                    continue
                ex = (p.get("extras") or {}) if isinstance(p, dict) else {}
                h_rows.append({
                    "Symbol": p.get("symbol") if isinstance(p, dict) else None,
                    "Bias": p.get("bias") if isinstance(p, dict) else None,
                    "Stage": (p.get("stage") if isinstance(p, dict) else None) or "OFF",
                    "Score": p.get("score") if isinstance(p, dict) else None,
                    "Session": p.get("session") if isinstance(p, dict) else None,
                    "Last": p.get("last") if isinstance(p, dict) else None,
                    "TSZ": ex.get("tsz"),
                    "EVS_ATR": ex.get("evs"),
                    "Intent": ex.get("intent_label"),
                    "Entry": p.get("entry") if isinstance(p, dict) else None,
                    "Stop": p.get("stop") if isinstance(p, dict) else None,
                    "TP1": p.get("tp0") if isinstance(p, dict) else None,
                    "TP2": p.get("tp1") if isinstance(p, dict) else None,
                    "TP3": p.get("tp2") if isinstance(p, dict) else None,
                    "AsOf": p.get("as_of") if isinstance(p, dict) else None,
                    "Why": p.get("why") if isinstance(p, dict) else None,
                })
            heavenly_df = pd.DataFrame(h_rows)
            heavenly_df = _arrow_safe_df(heavenly_df)
            st.dataframe(
                heavenly_df,
                use_container_width=True,
                hide_index=True,
                height=min(520, 34 + 30 * (len(heavenly_df) + 1)),
            )
        else:
            st.caption("HEAVENLY is off (or no results yet).")

        # --- MSS / ICT strict table ---
        st.subheader("🧱 MSS / ICT (Strict)")
        if results_mss:
            mss_rows = []
            for mm in results_mss:
                if mm is None:
                    continue
                if hasattr(mm, "to_dict"):
                    d = mm.to_dict()
                elif isinstance(mm, dict):
                    d = mm
                else:
                    d = getattr(mm, "__dict__", {}) or {}

                ex = d.get("extras") or {}
                pb = ex.get("pullback_band")
                if (
                    isinstance(pb, (tuple, list))
                    and len(pb) == 2
                    and all(isinstance(x, (int, float, np.number)) for x in pb)
                ):
                    pb_disp = f"{float(min(pb)):.4f}–{float(max(pb)):.4f}"
                elif pb is None:
                    pb_disp = "—"
                else:
                    pb_disp = str(pb)

                mss_rows.append({
                    "Symbol": d.get("symbol"),
                    "Bias": d.get("bias"),
                    "Stage": ex.get("stage") or "—",
                    "Score": d.get("setup_score"),
                    "Last": d.get("last_price"),
                    "POI": ex.get("poi_src") or "—",
                    "PullbackBand": pb_disp,
                    "BreakTrigger": ex.get("break_trigger"),
                    "Entry": ex.get("pullback_entry") or d.get("entry"),
                    "Stop": d.get("stop"),
                    "TP0": ex.get("tp0"),
                    "TP1": ex.get("tp1"),
                    "TP2": ex.get("tp2"),
                    "ETA_TP0_min": ex.get("eta_tp0_min"),
                    "Why": d.get("reason"),
                })
            mss_df = pd.DataFrame(mss_rows)
            mss_df = _arrow_safe_df(mss_df)
            st.dataframe(
                mss_df,
                use_container_width=True,
                hide_index=True,
                height=min(520, 34 + 30 * (len(mss_df) + 1)),
            )
        else:
            st.caption("No MSS setups (or fetch errors).")


        top = next((x for x in results_rev if x.bias in ["LONG", "SHORT"]), results_rev[0])
        pro_badge = "⚡ Pro" if pro_mode else "🧱 Basic"
        st.success(f"Top setup: **{top.symbol}** — **{top.bias}** (Score {top.setup_score}, {top.session}) • {pro_badge}")

        if top.bias == "NEUTRAL":
            st.info("Top-ranked row is **non-actionable** right now (hard requirement not met), so no Entry/TP and no alert will fire. Check the *Why* column for the blocker.")

        now = time.time()
        suppressed_by_cooldown = []

        # Pre-alerts are intentionally NOT optional: traders should see forming setups
        # *and* the confirmed triggers. We gate pre-alerts slightly below the main
        # threshold so we avoid spam, while still being early.
        pre_alert_threshold = max(60.0, float(alert_threshold) - 10.0)

        for r in results_rev:
            stage = (r.extras or {}).get("stage")
            actionable = (r.bias in ["LONG", "SHORT"] and stage in ("PRE", "CONFIRMED"))
            score_now = float(r.setup_score or 0)
            prev = st.session_state.symbol_state.get(r.symbol)

            # Persist state for next rerun (we track even NEUTRAL rows so we can detect threshold crossings).
            st.session_state.symbol_state[r.symbol] = {
                "bias": r.bias,
                "stage": stage,
                "actionable": actionable,
                "score": score_now,
            }

            # Alert eligibility:
            #   - CONFIRMED: actionable + above main threshold
            #   - PRE: actionable + above pre-threshold (still early)
            if not actionable:
                continue
            if stage == "PRE" and not (score_now >= pre_alert_threshold):
                continue
            if stage == "CONFIRMED" and not (score_now >= alert_threshold):
                continue

            # Fire alerts when the setup:
            #  - becomes actionable (NEUTRAL -> LONG/SHORT)
            #  - flips direction (LONG <-> SHORT)
            #  - crosses the score threshold (below -> above)
            prev_actionable = bool(prev.get("actionable")) if prev else False
            prev_bias = prev.get("bias") if prev else None
            prev_stage = prev.get("stage") if prev else None
            prev_score = float(prev.get("score") or 0) if prev else 0.0

            # Threshold crossing depends on stage.
            th = pre_alert_threshold if stage == "PRE" else float(alert_threshold)
            crossed_threshold = (prev is None) or (prev_score < th)
            became_actionable = (prev is None) or (not prev_actionable)
            flipped_direction = (prev is not None) and (prev_bias in ["LONG", "SHORT"]) and (prev_bias != r.bias)

            # Always alert when PRE -> CONFIRMED (even if threshold was already exceeded)
            promoted_to_confirmed = (prev_stage == "PRE") and (stage == "CONFIRMED")

            if not (crossed_threshold or became_actionable or flipped_direction or promoted_to_confirmed):
                continue

            alert_key = f"{r.symbol}::REV"
            if not can_alert(alert_key, now, cooldown_minutes):
                suppressed_by_cooldown.append(r.symbol)
                continue

            row = df.loc[df['Symbol'] == r.symbol].iloc[0].to_dict()
            # Help the email/alert payload include stage + a clean timestamp key.
            row["Time"] = row.get("AsOf")

            # In-app capture
            if capture_alerts:
                add_in_app_alert(row)

            # Auto-exec hook (LONG-only) — listens to this alert but does not change alert logic.
            try:
                _aecfg = st.session_state.get("autoexec_cfg")
                if _aecfg is not None and getattr(_aecfg, "enabled", False):
                    handle_alert_for_autoexec(_aecfg, "SCALP", row, allow_premarket, allow_opening, allow_midday, allow_powerhour, allow_afterhours)
            except Exception:
                pass

            # Email delivery
            if enable_email_alerts:
                ok, err = send_email_safe(row, smtp_server, smtp_port, smtp_user, smtp_password, to_emails)
                if not ok:
                    st.warning(f"Email alert failed for {r.symbol}: {err}")

            st.session_state.last_alert_ts[alert_key] = now

        # --- MSS alerts (Strict ICT/MSS) ---
        suppressed_by_cooldown_mss = []
        pre_mss_threshold = max(60.0, float(alert_threshold) - 10.0)

        for mm in results_mss:
            if mm is None:
                continue
            ex = mm.extras or {}
            stage = ex.get("stage")
            actionable = bool(mm.bias in ("MSS_LONG", "MSS_SHORT") and stage in ("PRE", "CONFIRMED") and ex.get("actionable"))
            score_now = float(mm.setup_score or 0)

            prev = st.session_state.get("mss_symbol_state", {}).get(mm.symbol)
            if "mss_symbol_state" not in st.session_state:
                st.session_state["mss_symbol_state"] = {}
            st.session_state["mss_symbol_state"][mm.symbol] = {
                "bias": mm.bias,
                "stage": stage,
                "actionable": actionable,
                "score": score_now,
            }

            if not actionable:
                continue
            if stage == "PRE" and not (score_now >= pre_mss_threshold):
                continue
            if stage == "CONFIRMED" and not (score_now >= float(alert_threshold)):
                continue

            prev_actionable = bool(prev.get("actionable")) if prev else False
            prev_bias = prev.get("bias") if prev else None
            prev_stage = prev.get("stage") if prev else None
            prev_score = float(prev.get("score") or 0) if prev else 0.0

            th = pre_mss_threshold if stage == "PRE" else float(alert_threshold)
            crossed_threshold = (prev is None) or (prev_score < th)
            became_actionable = (prev is None) or (not prev_actionable)
            flipped_direction = (prev is not None) and (prev_bias in ("MSS_LONG", "MSS_SHORT")) and (prev_bias != mm.bias)
            promoted_to_confirmed = (prev_stage == "PRE") and (stage == "CONFIRMED")

            if not (crossed_threshold or became_actionable or flipped_direction or promoted_to_confirmed):
                continue

            alert_key = f"{mm.symbol}::MSS"
            if not can_alert(alert_key, now, cooldown_minutes):
                suppressed_by_cooldown_mss.append(mm.symbol)
                continue

            bias_simple = "LONG" if mm.bias == "MSS_LONG" else "SHORT"
            payload = {
                "Symbol": mm.symbol,
                "Bias": bias_simple,
                "Tier": stage,
                "Score": int(score_now),
                "Session": mm.session,
                "Last": mm.last_price,
                "Entry": ex.get("pullback_entry") or mm.entry,
                "PullbackBand": ex.get("pullback_band"),
                "BreakTrigger": ex.get("break_trigger"),
                "Chase": ex.get("chase_line") or ex.get("break_trigger"),
                "Stop": mm.stop,
                "TP0": ex.get("tp0"),
                "TP1": ex.get("tp1"),
                "TP2": ex.get("tp2"),
                "TP3": ex.get("tp3"),
                "ETA TP0 (min)": ex.get("eta_tp0_min"),
                "Why": mm.reason,
                "AsOf": str(mm.timestamp) if mm.timestamp is not None else None,
                "Time": str(mm.timestamp) if mm.timestamp is not None else None,
                "Extras": {**ex, "family": "MSS"},
            }

            if capture_alerts:
                add_in_app_alert(payload)

            if enable_email_alerts:
                payload["Stage"] = stage
                payload["SignalFamily"] = "MSS"
                ok, err = send_email_safe(payload, smtp_server, smtp_port, smtp_user, smtp_password, to_emails)
                if not ok:
                    st.warning(f"Email MSS alert failed for {mm.symbol}: {err}")

            st.session_state.last_alert_ts[alert_key] = now

        if suppressed_by_cooldown_mss:
            st.info(
                "MSS alert cooldown suppressed: "
                + ", ".join(sorted(set(suppressed_by_cooldown_mss)))
                + ". Use **Clear cooldowns** in the sidebar if you want to force re-alerts."
            )

        # --- RIDE alerts (Drive/Continuation) ---
        # Keep ride alerts separate from reversal alerts: different family, different cooldown key,
        # and independent per-symbol state.
        suppressed_by_cooldown_ride = []
        pre_ride_threshold = max(60.0, float(alert_threshold) - 10.0)

        for rr in results_ride:
            ex = rr.extras or {}
            stage = ex.get("stage")
            actionable = bool(rr.bias in ("RIDE_LONG", "RIDE_SHORT") and stage in ("PRE", "CONFIRMED") and ex.get("actionable"))
            score_now = float(rr.setup_score or 0)

            # Persist ride state for transitions.
            prev = st.session_state.ride_symbol_state.get(rr.symbol)
            st.session_state.ride_symbol_state[rr.symbol] = {
                "bias": rr.bias,
                "stage": stage,
                "actionable": actionable,
                "score": score_now,
            }

            if not actionable:
                continue
            if stage == "PRE" and not (score_now >= pre_ride_threshold):
                continue
            if stage == "CONFIRMED" and not (score_now >= float(alert_threshold)):
                continue

            prev_actionable = bool(prev.get("actionable")) if prev else False
            prev_bias = prev.get("bias") if prev else None
            prev_stage = prev.get("stage") if prev else None
            prev_score = float(prev.get("score") or 0) if prev else 0.0

            th = pre_ride_threshold if stage == "PRE" else float(alert_threshold)
            crossed_threshold = (prev is None) or (prev_score < th)
            became_actionable = (prev is None) or (not prev_actionable)
            flipped_direction = (prev is not None) and (prev_bias in ("RIDE_LONG", "RIDE_SHORT")) and (prev_bias != rr.bias)
            promoted_to_confirmed = (prev_stage == "PRE") and (stage == "CONFIRMED")

            if not (crossed_threshold or became_actionable or flipped_direction or promoted_to_confirmed):
                continue

            alert_key = f"{rr.symbol}::RIDE"
            if not can_alert(alert_key, now, cooldown_minutes):
                suppressed_by_cooldown_ride.append(rr.symbol)
                continue

            # Normalize payload to the email/alert schema used everywhere else.
            bias_simple = "LONG" if rr.bias == "RIDE_LONG" else "SHORT"
            payload = {
                "Symbol": rr.symbol,
                "Bias": bias_simple,
                "Tier": stage,
                "Score": int(score_now),
                "Session": rr.session,
                "Last": rr.last_price,
                "Entry": ex.get("pullback_entry") or rr.entry,
                "PullbackEntry": ex.get("pullback_entry"),
                "BreakTrigger": ex.get("break_trigger"),
                "Chase": ex.get("chase_line") or ex.get("break_trigger"),
                "Stop": rr.stop,
                "TP0": ex.get("tp0") or rr.target_1r,
                "TP1": ex.get("tp1") or rr.target_2r,
                "TP2": ex.get("tp2"),
                "TP3": ex.get("tp3"),
                "ETA TP0 (min)": ex.get("eta_tp0_min"),
                "Why": rr.reason,
                "AsOf": str(rr.timestamp) if rr.timestamp is not None else None,
                "Time": str(rr.timestamp) if rr.timestamp is not None else None,
                "Extras": {**ex, "family": "RIDE"},
            }

            if capture_alerts:
                add_in_app_alert(payload)

            # Auto-exec hook (LONG-only) — listens to this alert but does not change alert logic.
            try:
                _aecfg = st.session_state.get("autoexec_cfg")
                if _aecfg is not None and getattr(_aecfg, "enabled", False):
                    handle_alert_for_autoexec(_aecfg, "RIDE", payload, allow_premarket, allow_opening, allow_midday, allow_powerhour, allow_afterhours)
            except Exception:
                pass

            if enable_email_alerts:
                # Tag subject so you can filter/route ride alerts.
                payload["Stage"] = stage
                payload["SignalFamily"] = "RIDE"
                ok, err = send_email_safe(payload, smtp_server, smtp_port, smtp_user, smtp_password, to_emails)
                if not ok:
                    st.warning(f"Email RIDE alert failed for {rr.symbol}: {err}")

            st.session_state.last_alert_ts[alert_key] = now

        # --- SWING alerts (Intraday Swing) ---
        suppressed_by_cooldown_swing = []
        pre_swing_threshold = max(60.0, float(alert_threshold) - 10.0)

        for ss in results_swing:
            if ss is None:
                continue
            ex = ss.extras or {}
            stage = ex.get("stage")
            actionable = bool(ss.bias in ("SWING_LONG", "SWING_SHORT") and stage in ("PRE", "CONFIRMED") and ex.get("actionable"))
            score_now = float(ss.setup_score or 0)

            prev = st.session_state.swing_symbol_state.get(ss.symbol)
            st.session_state.swing_symbol_state[ss.symbol] = {
                "bias": ss.bias,
                "stage": stage,
                "actionable": actionable,
                "score": score_now,
            }

            if not actionable:
                continue
            if stage == "PRE" and not (score_now >= pre_swing_threshold):
                continue
            if stage == "CONFIRMED" and not (score_now >= float(alert_threshold)):
                continue

            prev_actionable = bool(prev.get("actionable")) if prev else False
            prev_bias = prev.get("bias") if prev else None
            prev_stage = prev.get("stage") if prev else None
            prev_score = float(prev.get("score") or 0) if prev else 0.0

            th = pre_swing_threshold if stage == "PRE" else float(alert_threshold)
            crossed_threshold = (prev is None) or (prev_score < th)
            became_actionable = (prev is None) or (not prev_actionable)
            flipped_direction = (prev is not None) and (prev_bias in ("SWING_LONG", "SWING_SHORT")) and (prev_bias != ss.bias)
            promoted_to_confirmed = (prev_stage == "PRE") and (stage == "CONFIRMED")

            if not (crossed_threshold or became_actionable or flipped_direction or promoted_to_confirmed):
                continue

            alert_key = f"{ss.symbol}::SWING"
            if not can_alert(alert_key, now, cooldown_minutes):
                suppressed_by_cooldown_swing.append(ss.symbol)
                continue

            bias_simple = "LONG" if ss.bias == "SWING_LONG" else "SHORT"
            payload = {
                "Symbol": ss.symbol,
                "Bias": bias_simple,
                "Tier": stage,
                "Score": int(score_now),
                "Session": ss.session,
                "Last": ss.last_price,
                "Entry": ex.get("pullback_entry") or ss.entry,
                "PullbackBand": ex.get("pullback_band"),
                "BreakTrigger": ex.get("break_trigger"),
                "Chase": ex.get("chase_line") or ex.get("break_trigger"),
                "Stop": ss.stop,
                "TP0": ex.get("tp0") or ss.target_1r,
                "TP1": ex.get("tp1") or ss.target_2r,
                "TP2": ex.get("tp2"),
                "TP3": ex.get("tp3"),
                "ETA TP0 (min)": ex.get("eta_tp0_min"),
                "Why": ss.reason,
                "AsOf": str(ss.timestamp) if ss.timestamp is not None else None,
                "Time": str(ss.timestamp) if ss.timestamp is not None else None,
                "Extras": {**ex, "family": "SWING"},
            }

            if capture_alerts:
                add_in_app_alert(payload)

            if enable_email_alerts:
                payload["Stage"] = stage
                payload["SignalFamily"] = "SWING"
                ok, err = send_email_safe(payload, smtp_server, smtp_port, smtp_user, smtp_password, to_emails)
                if not ok:
                    st.warning(f"Email SWING alert failed for {ss.symbol}: {err}")

            st.session_state.last_alert_ts[alert_key] = now

        # --- HEAVENLY alerts (new engine) ---
        suppressed_by_cooldown_heavenly = []
        if enable_heavenly and results_heavenly:
            for p in results_heavenly:
                if not isinstance(p, dict):
                    continue
                stage = (p.get("stage") or "").upper()
                direction = (p.get("bias") or "NEUTRAL").upper()
                if stage != "ENTRY" or direction not in ("LONG", "SHORT"):
                    continue

                ex = p.get("extras") or {}
                trig_ts = ex.get("trigger_bar_ts")

                prev = st.session_state.heavenly_symbol_state.get(p.get("symbol"))
                st.session_state.heavenly_symbol_state[p.get("symbol")] = {
                    "stage": stage,
                    "bias": direction,
                    "trigger_bar_ts": trig_ts,
                }

                # Only alert when we first reach ENTRY for a given trigger bar.
                if prev and prev.get("stage") == "ENTRY" and prev.get("trigger_bar_ts") == trig_ts:
                    continue

                alert_key = f"{p.get('symbol')}::HEAVENLY"
                if not can_alert(alert_key, now, cooldown_minutes):
                    suppressed_by_cooldown_heavenly.append(p.get('symbol'))
                    continue

                payload = {
                    "Symbol": p.get("symbol"),
                    "Bias": direction,
                    "Tier": stage,
                    "Score": p.get("score"),
                    "Session": p.get("session"),
                    "Last": p.get("last"),
                    "Entry": p.get("entry"),
                    "Chase": None,
                    "Stop": p.get("stop"),
                    "TP0": p.get("tp0"),
                    "TP1": p.get("tp1"),
                    "TP2": p.get("tp2"),
                    "TP3": None,
                    "Why": p.get("why"),
                    "AsOf": p.get("as_of"),
                    "Time": p.get("as_of"),
                    "Extras": {**ex, "family": "HEAVENLY"},
                }

                if capture_alerts:
                    add_in_app_alert(payload)

                # Auto-exec hook (LONG-only) — listens to this alert but does not change alert logic.
                try:
                    _aecfg = st.session_state.get("autoexec_cfg")
                    if _aecfg is not None and getattr(_aecfg, "enabled", False):
                        handle_alert_for_autoexec(_aecfg, "HEAVENLY", payload, allow_premarket, allow_opening, allow_midday, allow_powerhour, allow_afterhours)
                except Exception:
                    pass

                if enable_email_alerts:
                    payload["Stage"] = stage
                    payload["SignalFamily"] = "HEAVENLY"
                    ok, err = send_email_safe(payload, smtp_server, smtp_port, smtp_user, smtp_password, to_emails)
                    if not ok:
                        st.warning(f"Email HEAVENLY alert failed for {p.get('symbol')}: {err}")

                st.session_state.last_alert_ts[alert_key] = now

        if suppressed_by_cooldown_heavenly:
            st.info(
                "HEAVENLY alert cooldown suppressed: "
                + ", ".join(sorted(set([x for x in suppressed_by_cooldown_heavenly if x])))
                + ". Use **Clear cooldowns** in the sidebar if you want to force re-alerts."
            )

        if suppressed_by_cooldown_swing:
            st.info(
                "SWING alert cooldown suppressed: "
                + ", ".join(sorted(set(suppressed_by_cooldown_swing)))
                + ". Use **Clear cooldowns** in the sidebar if you want to force re-alerts."
            )

        if suppressed_by_cooldown_ride:
            st.info(
                "RIDE alert cooldown suppressed: "
                + ", ".join(sorted(set(suppressed_by_cooldown_ride)))
                + ". Use **Clear cooldowns** in the sidebar if you want to force re-alerts."
            )

        if suppressed_by_cooldown:
            st.info(
                "Alert cooldown suppressed: "
                + ", ".join(sorted(set(suppressed_by_cooldown)))
                + ". Use **Clear cooldowns** in the sidebar if you want to force re-alerts."
            )


        st.subheader("Chart & Signal Detail")
        pick = st.selectbox("Select ticker", [r.symbol for r in results_rev], index=0)

        with st.spinner(f"Loading chart data for {pick}..."):
            ohlcv, rsi5, rsi14, macd_hist, quote = fetch_bundle(client, pick, interval=interval)

        sig = compute_scalp_signal(
            pick, ohlcv, rsi5, rsi14, macd_hist,
            mode=mode,
            pro_mode=pro_mode,
            allow_opening=allow_opening,
            allow_midday=allow_midday,
            allow_power=allow_power,
            allow_premarket=allow_premarket,
            allow_afterhours=allow_afterhours,
            use_last_closed_only=use_last_closed_only,
            bar_closed_guard=bar_closed_guard,
            interval=interval,
            vwap_logic=vwap_logic,
            session_vwap_include_premarket=session_vwap_include_premarket,
            fib_lookback_bars=fib_lookback,
            target_atr_pct=target_atr_pct,
            killzone_preset=killzone_preset,
            liquidity_weighting=liquidity_weighting,
            orb_minutes=orb_minutes,
            entry_model=entry_model,
            slippage_mode=slip_mode,
            fixed_slippage_cents=slip_fixed_cents,
            atr_fraction_slippage=slip_atr_frac,
        )
        plot_df = ohlcv.sort_index().copy().tail(260)
        plot_df["vwap_cum"] = calc_vwap(plot_df)
        plot_df["vwap_sess"] = calc_session_vwap(plot_df, include_premarket=session_vwap_include_premarket)

        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df["open"], high=plot_df["high"], low=plot_df["low"], close=plot_df["close"], name="Price"))

        if show_dual_vwap:
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["vwap_sess"], mode="lines", name="VWAP (Session)"))
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["vwap_cum"], mode="lines", name="VWAP (Cumulative)"))
        else:
            key = "vwap_sess" if vwap_logic == "session" else "vwap_cum"
            nm = "VWAP (Session)" if vwap_logic == "session" else "VWAP (Cumulative)"
            fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[key], mode="lines", name=nm))

        # Fib lines (visual)
        if show_fibs:
            seg = plot_df.tail(int(min(fib_lookback, len(plot_df))))
            hi = float(seg["high"].max())
            lo = float(seg["low"].min())
            if hi > lo:
                for name, level in [("Fib 0.382", hi - 0.382*(hi-lo)), ("Fib 0.5", hi - 0.5*(hi-lo)), ("Fib 0.618", hi - 0.618*(hi-lo)), ("Fib 0.786", hi - 0.786*(hi-lo))]:
                    fig.add_hline(y=level, line_dash="dot", annotation_text=name, annotation_position="top left")

        # Entry/Stop/Targets
        if sig.entry and sig.stop:
            fig.add_hline(y=sig.entry, line_dash="dot", annotation_text="Entry", annotation_position="top left")
            fig.add_hline(y=sig.stop, line_dash="dash", annotation_text="Stop", annotation_position="bottom left")
        if sig.target_1r:
            fig.add_hline(y=sig.target_1r, line_dash="dot", annotation_text="1R", annotation_position="top right")
        if sig.target_2r:
            fig.add_hline(y=sig.target_2r, line_dash="dot", annotation_text="2R", annotation_position="top right")
        fib_tp1 = (sig.extras or {}).get("fib_tp1")
        fib_tp2 = (sig.extras or {}).get("fib_tp2")
        if isinstance(fib_tp1, (float, int)):
            fig.add_hline(y=float(fib_tp1), line_dash="dash", annotation_text="Fib TP1", annotation_position="top right")
        if isinstance(fib_tp2, (float, int)):
            fig.add_hline(y=float(fib_tp2), line_dash="dash", annotation_text="Fib TP2", annotation_position="top right")

        fig.update_layout(height=540, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
        with c1: st.metric("Bias", sig.bias)
        with c2: st.metric("Score", sig.setup_score)
        with c3: st.metric("Session", sig.session)
        with c4: st.metric("Liquidity", (sig.extras or {}).get("liquidity_phase", ""))
        with c5:
            lp = quote if quote is not None else sig.last_price
            st.metric("Last", f"{lp:.4f}" if lp is not None else "N/A")
        with c6:
            atrp = (sig.extras or {}).get("atr_pct")
            basep = (sig.extras or {}).get("atr_ref_pct")
            st.metric("ATR% / Base", f"{atrp:.3f} / {basep:.3f}" if isinstance(atrp, (float,int)) and isinstance(basep, (float,int)) else "N/A")
        with c7:
            sc = (sig.extras or {}).get("atr_score_scale")
            st.metric("Score scale", f"{sc:.2f}" if isinstance(sc, (float,int)) else "N/A")

        st.write("**Reasoning:**", sig.reason)

        st.markdown("### Trade Plan")
        if sig.bias in ["LONG", "SHORT"] and sig.entry and sig.stop:
            st.write(f"- **Entry:** {sig.entry:.4f}")
            st.write(f"- **Stop:** {sig.stop:.4f}")
            st.write(f"- **Targets (R):** 1R={sig.target_1r:.4f} • 2R={sig.target_2r:.4f}")
            if isinstance(fib_tp1, (float,int)) or isinstance(fib_tp2, (float,int)):
                st.write(f"- **Fib partials:** TP1={fib_tp1 if fib_tp1 is not None else '—'} • TP2={fib_tp2 if fib_tp2 is not None else '—'}")
            st.write("- **Fail-safe exit:** if price loses VWAP and MACD histogram turns against you, flatten remainder.")
            st.warning("Analytics tool only — always position-size and respect stops.")
        else:
            st.info("No clean confluence signal right now (or time-of-day filter blocking).")

        with st.expander("Diagnostics"):
            st.json(sig.extras)

    else:
        st.info("Add your watchlist in the sidebar, then click **Scan Watchlist** or enable auto-refresh.")

    if auto_refresh:
        # Streamlit doesn't have a native timer; we sleep then rerun.
        # Keep refresh >=10s and enforce API pacing via 'Seconds between API calls'.
        time.sleep(float(refresh_seconds or 30))
        st.rerun()