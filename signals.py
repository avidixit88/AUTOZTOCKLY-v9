from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import pandas as pd
import numpy as np
import math

from indicators import (
    vwap as calc_vwap,
    session_vwap as calc_session_vwap,
    atr as calc_atr,
    ema as calc_ema,
    adx as calc_adx,
    rolling_swing_lows,
    rolling_swing_highs,
    detect_fvg,
    find_order_block,
    find_breaker_block,
    in_zone,
)
from sessions import classify_session, classify_liquidity_phase


def _cap_score(x: float | int | None) -> int:
    """Scores are treated as 0..100 for UI + alerting.

    The internal point system can temporarily exceed 100 when multiple features
    stack or when ATR normalization scales up. We cap here so the UI never
    shows impossible percentages (e.g., 113%).
    """
    try:
        if x is None:
            return 0
        return int(np.clip(float(x), 0.0, 100.0))
    except Exception:
        return 0


@dataclass
class SignalResult:
    symbol: str
    bias: str                      # "LONG", "SHORT", "NEUTRAL"
    setup_score: int               # 0..100 (calibrated)
    reason: str
    entry: Optional[float]
    stop: Optional[float]
    target_1r: Optional[float]
    target_2r: Optional[float]
    last_price: Optional[float]
    timestamp: Optional[pd.Timestamp]
    session: str                   # OPENING/MIDDAY/POWER/PREMARKET/AFTERHOURS/OFF
    extras: Dict[str, Any]


# ---------------------------
# SWING / Intraday-Swing (structure-first) signal family
# ---------------------------

def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample intraday OHLCV to a higher timeframe without additional API calls.

    We use this for Swing alerts so we don't add extra Alpha Vantage calls.
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    if not isinstance(df.index, pd.DatetimeIndex):
        return df.copy()
    out = (
        df[["open", "high", "low", "close", "volume"]]
        .resample(rule)
        .agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        })
        .dropna()
    )
    return out


def compute_swing_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    rsi5: pd.Series,
    rsi14: pd.Series,
    macd_hist: pd.Series,
    *,
    interval: str = "1min",
    pro_mode: bool = False,
    # Time filters
    allow_opening: bool = True,
    allow_midday: bool = True,
    allow_power: bool = True,
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
    use_last_closed_only: bool = False,
    bar_closed_guard: bool = True,
    # Shared options
    vwap_logic: str = "session",
    session_vwap_include_premarket: bool = False,
    fib_lookback_bars: int = 240,
    orb_minutes: int = 15,
    liquidity_weighting: float = 0.55,
    target_atr_pct: float | None = None,
) -> SignalResult:
    """SWING v2 — HTF dip-buy positioning with 5m confirmation (CONFIRM-only emails).

    Design principles:
      - SWING is a *positioning* engine (define cheap-risk buy zones) not an execution engine.
      - Use HTF structure (30m derived from intraday bars) to define the impulse leg and retrace band.
      - Stages:
          STALK   : candidate is valid; waiting for price to enter the buy zone
          BUY ZONE: price has entered the retrace band; waiting for confirmation
          CONFIRM : 5m confirmation (2-of-3); actionable (email is sent on CONFIRM only)
      - Streamlit safety: extras contain primitives only (no dict/list objects).
    """
    # -------------------------
    # Basic guards
    # -------------------------
    if ohlcv is None or ohlcv.empty or len(ohlcv) < 120:
        return SignalResult(
            symbol, "CHOP", 0, "Not enough data",
            None, None, None, None,
            None, None, "OFF",
            {"family": "SWING", "stage": "OFF", "swing_stage": "OFF"}
        )

    df = ohlcv.copy()
    if use_last_closed_only and len(df) >= 2:
        df = df.iloc[:-1].copy()

    last_ts = df.index[-1] if isinstance(df.index, pd.DatetimeIndex) else None
    last_price = float(df["close"].iloc[-1])

    # -------------------------
    # Session gating (respects allowed_sessions toggles)
    # -------------------------
    try:
        sess = classify_session(
            last_ts,
            allow_opening=allow_opening,
            allow_midday=allow_midday,
            allow_power=allow_power,
            allow_premarket=allow_premarket,
            allow_afterhours=allow_afterhours,
        )
    except Exception:
        sess = "OFF"

    if sess == "OFF":
        return SignalResult(
            symbol, "CHOP", 0, "Outside allowed session",
            None, None, None, None,
            last_price, last_ts, "OFF",
            {"family": "SWING", "stage": "OFF", "swing_stage": "OFF"}
        )

    # -------------------------
    # HTF (30m) context — derived via resample (no extra API call)
    #   - When running 5m, resampling preserves your interval choice.
    # -------------------------
    htf = _resample_ohlcv(df, "30T").tail(5 * 13 * 10).copy()  # ~10 trading days of 30m bars
    if len(htf) < 60:
        return SignalResult(
            symbol, "CHOP", 0, "Not enough HTF bars",
            None, None, None, None,
            last_price, last_ts, sess,
            {"family": "SWING", "stage": "OFF", "swing_stage": "OFF"}
        )

    # Core HTF measures
    htf["ema20"] = calc_ema(htf["close"], 20)
    htf["ema50"] = calc_ema(htf["close"], 50)
    htf["atr"] = calc_atr(htf, 14)

    atr30 = float(htf["atr"].iloc[-1]) if np.isfinite(htf["atr"].iloc[-1]) else float(calc_atr(df, 14).iloc[-1])
    atr30 = atr30 if np.isfinite(atr30) and atr30 > 0 else max(1e-9, float(df["high"].tail(30).max() - df["low"].tail(30).min()) / 30.0)

    # HTF VWAP proxy (cumulative on HTF bars)
    try:
        htf_vwap = calc_vwap(htf).iloc[-1]
        htf_vwap = float(htf_vwap) if np.isfinite(htf_vwap) else None
    except Exception:
        htf_vwap = None

    # -------------------------
    # HTF pivot structure
    # -------------------------
    is_low = rolling_swing_lows(htf["low"], left=3, right=3)
    is_high = rolling_swing_highs(htf["high"], left=3, right=3)

    pivot_lows = htf.loc[is_low]
    pivot_highs = htf.loc[is_high]

    if len(pivot_lows) < 2 or len(pivot_highs) < 2:
        return SignalResult(
            symbol, "CHOP", 0, "Awaiting structure",
            None, None, None, None,
            last_price, last_ts, sess,
            {"family": "SWING", "stage": "PRE", "actionable": False, "swing_stage": "STALK"}
        )

    last_low = float(pivot_lows["low"].iloc[-1])
    prev_low = float(pivot_lows["low"].iloc[-2])
    last_high = float(pivot_highs["high"].iloc[-1])
    prev_high = float(pivot_highs["high"].iloc[-2])

    uptrend = (last_high > prev_high) and (last_low >= prev_low)
    downtrend = (last_low < prev_low) and (last_high <= prev_high)

    if not uptrend and not downtrend:
        return SignalResult(
            symbol, "CHOP", 0, "CHOP (HTF trend unclear)",
            None, None, None, None,
            last_price, last_ts, sess,
            {"family": "SWING", "stage": "PRE", "actionable": False, "swing_stage": "STALK"}
        )

    # IMPORTANT WIRING NOTE:
    # Downstream (tables + email gating) expects SWING bias labels:
    #   SWING_LONG / SWING_SHORT / CHOP
    # We keep a separate direction helper for internal comparisons.
    bias_dir = "LONG" if uptrend else "SHORT"
    bias = "SWING_LONG" if bias_dir == "LONG" else "SWING_SHORT"

    # -------------------------
    # Trend lock score (0..5)
    # -------------------------
    tl = 0
    # (1) EMA stack
    ema20 = float(htf["ema20"].iloc[-1]) if np.isfinite(htf["ema20"].iloc[-1]) else None
    ema50 = float(htf["ema50"].iloc[-1]) if np.isfinite(htf["ema50"].iloc[-1]) else None
    if ema20 is not None and ema50 is not None:
        if bias_dir == "LONG" and ema20 >= ema50:
            tl += 1
        if bias_dir == "SHORT" and ema20 <= ema50:
            tl += 1
    # (2) Price vs EMA20
    if ema20 is not None:
        if bias_dir == "LONG" and float(htf["close"].iloc[-1]) >= ema20:
            tl += 1
        if bias_dir == "SHORT" and float(htf["close"].iloc[-1]) <= ema20:
            tl += 1
    # (3) HTF VWAP alignment
    if htf_vwap is not None:
        if bias_dir == "LONG" and float(htf["close"].iloc[-1]) >= htf_vwap:
            tl += 1
        if bias_dir == "SHORT" and float(htf["close"].iloc[-1]) <= htf_vwap:
            tl += 1
    # (4) Structure confirmation
    tl += 1  # uptrend/downtrend already established
    # (5) DI spread proxy via candle range expansion in trend direction (robust, no extra indicators)
    # A small boost if the last HTF candle expanded in the trend direction.
    last_htf = htf.iloc[-1]
    if bias_dir == "LONG" and float(last_htf["close"]) >= float(last_htf["open"]):
        tl += 1
    if bias_dir == "SHORT" and float(last_htf["close"]) <= float(last_htf["open"]):
        tl += 1
    trend_lock_score = int(max(0, min(5, tl)))

    if trend_lock_score < 2:
        return SignalResult(
            symbol, bias, 0, "WATCH - Trend not locked",
            None, None, None, None,
            last_price, last_ts, sess,
            {"family": "SWING", "stage": "PRE", "actionable": False, "swing_stage": "STALK", "trend_lock_score": trend_lock_score}
        )

    # -------------------------
    # Define impulse leg from HTF pivots
    # -------------------------
    # Build ordered pivot list
    piv = []
    for ts, row in pivot_lows.tail(12).iterrows():
        piv.append((ts, "L", float(row["low"])))
    for ts, row in pivot_highs.tail(12).iterrows():
        piv.append((ts, "H", float(row["high"])))
    piv.sort(key=lambda x: x[0])

    impulse_start = None
    impulse_end = None
    # Scan from the end to find the latest complete impulse in the trend direction.
    if bias_dir == "LONG":
        # need L then later H
        for i in range(len(piv) - 1, 0, -1):
            if piv[i][1] == "H":
                # find prior L
                for j in range(i - 1, -1, -1):
                    if piv[j][1] == "L":
                        impulse_start = piv[j][2]
                        impulse_end = piv[i][2]
                        break
            if impulse_start is not None:
                break
    else:
        # need H then later L
        for i in range(len(piv) - 1, 0, -1):
            if piv[i][1] == "L":
                for j in range(i - 1, -1, -1):
                    if piv[j][1] == "H":
                        impulse_start = piv[j][2]
                        impulse_end = piv[i][2]
                        break
            if impulse_start is not None:
                break

    if impulse_start is None or impulse_end is None:
        return SignalResult(
            symbol, bias, 0, "WATCH - Awaiting impulse leg",
            None, None, None, None,
            last_price, last_ts, sess,
            {"family": "SWING", "stage": "PRE", "actionable": False, "swing_stage": "STALK", "trend_lock_score": trend_lock_score}
        )

    impulse_range = abs(impulse_end - impulse_start)
    if not np.isfinite(impulse_range) or impulse_range < 1.2 * atr30:
        return SignalResult(
            symbol, bias, 0, "WATCH - Impulse too small",
            None, None, None, None,
            last_price, last_ts, sess,
            {
                "family": "SWING",
                "stage": "PRE",
                "swing_stage": "STALK",
                "trend_lock_score": trend_lock_score,
                "impulse_start": float(impulse_start),
                "impulse_end": float(impulse_end),
                "retrace_mode": "pivot-leg",
            }
        )

    # -------------------------
    # Retrace band (38.2%..61.8%) + sweet spot 50%
    # -------------------------
    if bias_dir == "LONG":
        # Retrace from impulse_end downwards
        pb1 = float(impulse_end - 0.382 * impulse_range)
        pb2 = float(impulse_end - 0.618 * impulse_range)
        band_high = max(pb1, pb2)
        band_low = min(pb1, pb2)
        retrace_pct = 100.0 * max(0.0, min(1.0, (impulse_end - last_price) / impulse_range))
        invalidated = last_price < (impulse_start - 0.10 * atr30)
    else:
        pb1 = float(impulse_end + 0.382 * impulse_range)
        pb2 = float(impulse_end + 0.618 * impulse_range)
        band_high = max(pb1, pb2)
        band_low = min(pb1, pb2)
        retrace_pct = 100.0 * max(0.0, min(1.0, (last_price - impulse_end) / impulse_range))
        invalidated = last_price > (impulse_start + 0.10 * atr30)

    if invalidated:
        return SignalResult(
            symbol, bias, 0, "Invalidated (broke impulse start)",
            None, None, None, None,
            last_price, last_ts, sess,
            {
                "family": "SWING",
                "stage": "PRE",
                "swing_stage": "FAIL",
                "trend_lock_score": trend_lock_score,
                "impulse_start": float(impulse_start),
                "impulse_end": float(impulse_end),
                "retrace_pct": float(retrace_pct),
                "pb1": float(band_low),
                "pb2": float(band_high),
                "retrace_mode": "pivot-leg",
            }
        )

    # Pullback quality (0..6)
    # - 6 if in band & near 50% retrace; lower if shallow/deep or outside.
    target_retrace = 0.50
    if bias_dir == "LONG":
        retr = (impulse_end - last_price) / impulse_range
    else:
        retr = (last_price - impulse_end) / impulse_range
    retr = float(retr) if np.isfinite(retr) else 0.0
    pb_in_band = (last_price >= band_low) and (last_price <= band_high)
    dist_to_mid = abs(retr - target_retrace)
    pbq = 0
    pbq_reasons = []
    if pb_in_band:
        pbq = 4
        pbq_reasons.append("In band")
        if dist_to_mid <= 0.08:
            pbq += 2
            pbq_reasons.append("Near 50%")
        elif dist_to_mid <= 0.14:
            pbq += 1
            pbq_reasons.append("Near mid")
    else:
        # shallow pullback
        if retr < 0.30:
            pbq = 1
            pbq_reasons.append("Too shallow")
        # deep pullback
        elif retr > 0.75:
            pbq = 1
            pbq_reasons.append("Too deep")
        else:
            pbq = 2
            pbq_reasons.append("Approaching band")

    pullback_quality = int(max(0, min(6, pbq)))

    # -------------------------
    # Confluence (capped)
    # -------------------------
    confluences = []
    conf_n = 0
    tol = 0.25 * atr30

    # (1) HTF VWAP overlap
    if htf_vwap is not None and (band_low - tol) <= htf_vwap <= (band_high + tol):
        conf_n += 1
        confluences.append("HTF VWAP")

    # (2) Prior day levels from intraday df
    try:
        if isinstance(df.index, pd.DatetimeIndex):
            d = df.copy()
            # Use last completed date
            last_date = d.index[-1].date()
            prev_days = sorted({x.date() for x in d.index if x.date() < last_date})
            if prev_days:
                pd_date = prev_days[-1]
                d_prev = d[d.index.date == pd_date]
                if len(d_prev) >= 5:
                    pdh = float(d_prev["high"].max())
                    pdl = float(d_prev["low"].min())
                    pdo = float(d_prev["open"].iloc[0])
                    for name, lvl in [("PDH", pdh), ("PDL", pdl), ("PDO", pdo)]:
                        if (band_low - tol) <= lvl <= (band_high + tol):
                            conf_n += 1
                            confluences.append(name)
    except Exception:
        pass

    # (3) Pivot cluster overlap (nearest prior pivot opposite side)
    try:
        # nearest HTF pivot level near band center
        band_mid = (band_low + band_high) / 2.0
        # use recent pivot highs/lows levels
        levels = []
        levels += [float(x) for x in pivot_highs["high"].tail(8).values]
        levels += [float(x) for x in pivot_lows["low"].tail(8).values]
        if levels:
            closest = min(levels, key=lambda x: abs(x - band_mid))
            if abs(closest - band_mid) <= tol:
                conf_n += 1
                confluences.append("HTF Pivot")
    except Exception:
        pass

    confluence_count = int(min(3, conf_n))
    confluences_str = ", ".join(confluences[:6]) if confluences else ""

    # -------------------------
    # Stage determination: STALK / BUY ZONE / CONFIRM
    # -------------------------
    stage = "STALK"
    entry_trigger_reason = ""
    # Zone entry check (within band OR within proximity)
    prox = 0.20 * atr30
    in_zone_now = (last_price >= (band_low - prox)) and (last_price <= (band_high + prox))
    if in_zone_now:
        stage = "BUY ZONE"

    # Confirmation checks on execution timeframe (df assumed to match selected interval, usually 5m)
    # We only allow confirm if we have a recent zone touch.
    recent = df.tail(20).copy()
    touched = False
    if bias_dir == "LONG":
        touched = float(recent["low"].min()) <= (band_high + prox)
    else:
        touched = float(recent["high"].max()) >= (band_low - prox)

    # Compute 5m ATR for stop sizing
    atr5 = float(calc_atr(df, 14).iloc[-1]) if len(df) >= 20 else atr30
    atr5 = atr5 if np.isfinite(atr5) and atr5 > 0 else atr30

    # Confirm rules (2-of-3)
    confirm_hits = 0
    confirm_reasons = []

    # (1) Reclaim VWAP after zone touch
    vwap_s = None
    try:
        vwap_s = calc_session_vwap(df).iloc[-1]
        vwap_s = float(vwap_s) if np.isfinite(vwap_s) else None
    except Exception:
        vwap_s = None

    if touched and vwap_s is not None and len(df) >= 2:
        prev_close = float(df["close"].iloc[-2])
        cur_close = float(df["close"].iloc[-1])
        if bias_dir == "LONG" and prev_close < vwap_s <= cur_close:
            confirm_hits += 1
            confirm_reasons.append("VWAP reclaim")
        if bias_dir == "SHORT" and prev_close > vwap_s >= cur_close:
            confirm_hits += 1
            confirm_reasons.append("VWAP reject")

    # (2) Two-candle reversal (robust)
    if touched and len(df) >= 3:
        c1 = df.iloc[-2]
        c2 = df.iloc[-1]
        if bias_dir == "LONG":
            if float(c1["close"]) < float(c1["open"]) and float(c2["close"]) > float(c2["open"]) and float(c2["close"]) > float(c1["high"]):
                confirm_hits += 1
                confirm_reasons.append("2-bar reversal")
        else:
            if float(c1["close"]) > float(c1["open"]) and float(c2["close"]) < float(c2["open"]) and float(c2["close"]) < float(c1["low"]):
                confirm_hits += 1
                confirm_reasons.append("2-bar reversal")

    # (3) RSI regime cross (compute from provided rsi14 if aligned; else compute on df closes)
    rsi_now = None
    rsi_prev = None
    try:
        if rsi14 is not None and len(rsi14) >= len(df):
            rsi_now = float(rsi14.iloc[-1])
            rsi_prev = float(rsi14.iloc[-2]) if len(rsi14) >= 2 else None
        else:
            # lightweight recompute using calc_rsi provided by engine (if available) is outside this module
            # so we compute locally here
            closes = df["close"].astype(float)
            delta = closes.diff()
            up = delta.clip(lower=0).rolling(14).mean()
            down = (-delta.clip(upper=0)).rolling(14).mean()
            rs = up / down.replace(0, np.nan)
            rsi_series = 100 - (100 / (1 + rs))
            rsi_now = float(rsi_series.iloc[-1])
            rsi_prev = float(rsi_series.iloc[-2]) if len(rsi_series) >= 2 else None
    except Exception:
        rsi_now = None
        rsi_prev = None

    if touched and rsi_now is not None and rsi_prev is not None:
        if bias_dir == "LONG" and (rsi_prev < 50 <= rsi_now):
            confirm_hits += 1
            confirm_reasons.append("RSI reclaim")
        if bias_dir == "SHORT" and (rsi_prev > 50 >= rsi_now):
            confirm_hits += 1
            confirm_reasons.append("RSI roll")

    confirmed = touched and (confirm_hits >= 2)

    # -------------------------
    # Score (0..100) — optimized for entry quality
    # -------------------------
    score = 0
    score += 10 * trend_lock_score  # 0..50
    if impulse_range >= 2.0 * atr30:
        score += 8
    if impulse_range >= 3.0 * atr30:
        score += 6  # stack a bit more for bigger displacement
    # Pullback quality boost
    if pullback_quality > 2:
        score += 12 * (pullback_quality - 2)  # max +48
    # Confluence
    score += 8 * confluence_count  # max +24
    # Stage boosts
    if stage == "BUY ZONE":
        score += 10
    if confirmed:
        score += 18
    # Penalties
    # "Actionable" here is a scoring concept (do we have enough quality to consider the setup real?).
    # Email/alert gating is handled separately via extras["actionable"].
    actionable_ok = (confluence_count >= 1) and (pullback_quality >= 2)
    if not actionable_ok:
        score -= 8
    # Character ok: avoid over-extended RSI > 80 longs / <20 shorts
    character_ok = True
    if rsi_now is not None:
        if bias_dir == "LONG" and rsi_now > 80:
            character_ok = False
        if bias_dir == "SHORT" and rsi_now < 20:
            character_ok = False
    if not character_ok:
        score -= 20

    score = int(_cap_score(score))

    # -------------------------
    # Entry / stop / targets (only actionable on CONFIRM)
    # -------------------------
    entry = None
    stop = None
    tp0 = None
    tp1 = None

    # TP0 preference: mean-reversion to session VWAP if available; else impulse end
    if vwap_s is not None and np.isfinite(vwap_s):
        tp0 = float(vwap_s)
    else:
        tp0 = float(impulse_end)

    tp1 = float(impulse_end)

    if confirmed:
        stage = "CONFIRM"
        entry_trigger_reason = ", ".join(confirm_reasons)
        entry = last_price
        if bias_dir == "LONG":
            stop = float(band_low - 0.25 * atr5)
        else:
            stop = float(band_high + 0.25 * atr5)

        # Ensure targets are in the correct direction for R-multiples
        if bias_dir == "LONG":
            tp0 = max(tp0, entry + 0.5 * atr5)
            tp1 = max(tp1, entry + 1.0 * atr5)
        else:
            tp0 = min(tp0, entry - 0.5 * atr5)
            tp1 = min(tp1, entry - 1.0 * atr5)

        return SignalResult(
            symbol,
            bias,
            score,
            "CONFIRM - Dip-buy confirmed",
            float(entry),
            float(stop),
            float(tp0),
            float(tp1),
            last_price,
            last_ts,
            sess,
            {
                "family": "SWING",
                "stage": "CONFIRMED",
                # Email gating expects an explicit boolean.
                # For SWING we only want emails on CONFIRMED signals.
                "actionable": True,
                "swing_stage": "CONFIRM",
                "trend_lock_score": trend_lock_score,
                "retrace_pct": float(retrace_pct),
                "impulse_start": float(impulse_start),
                "impulse_end": float(impulse_end),
                "retrace_mode": "pivot-leg",
                "pullback_quality": pullback_quality,
                "pullback_quality_reasons": "; ".join(pbq_reasons),
                "confluence_count": confluence_count,
                "confluences": confluences_str,
                "pb1": float(band_low),
                "pb2": float(band_high),
                "pullback_band": [float(band_low), float(band_high)],
                "entry_zone": f"{band_low:.4f}–{band_high:.4f}",
                "entry_trigger_reason": entry_trigger_reason,
            }
        )

    # Not confirmed yet (no email)
    why = "WATCH - Trend locked, awaiting zone entry"
    if stage == "BUY ZONE":
        why = "SETUP - In buy zone, awaiting confirmation"
    return SignalResult(
        symbol,
        bias,
        score,
        why,
        None,
        None,
        None,
        None,
        last_price,
        last_ts,
        sess,
        {
            "family": "SWING",
            "stage": "PRE",
            "actionable": False,
            "swing_stage": stage,
            "trend_lock_score": trend_lock_score,
            "retrace_pct": float(retrace_pct),
            "impulse_start": float(impulse_start),
            "impulse_end": float(impulse_end),
            "retrace_mode": "pivot-leg",
            "pullback_quality": pullback_quality,
            "pullback_quality_reasons": "; ".join(pbq_reasons),
            "confluence_count": confluence_count,
            "confluences": confluences_str,
            "pb1": float(band_low),
            "pb2": float(band_high),
            "pullback_band": [float(band_low), float(band_high)],
            "entry_zone": f"{band_low:.4f}–{band_high:.4f}",
        }
    )


def _mfe_percentile_from_history(
    df: pd.DataFrame,
    *,
    direction: str,
    occur_mask: pd.Series,
    horizon_bars: int,
    pct: float,
) -> tuple[float | None, int]:
    """Compute a percentile of forward MFE for occurrences marked by occur_mask.

    LONG MFE is max(high fwd) - close at signal bar.
    SHORT MFE is close - min(low fwd).
    Returns (mfe_pct, n_samples).
    """
    try:
        h = int(horizon_bars)
        if h <= 0:
            return None, 0
    except Exception:
        return None, 0

    if occur_mask is None or df is None or len(df) == 0:
        return None, 0

    try:
        close = df["close"].astype(float)
        hi = df["high"].astype(float)
        lo = df["low"].astype(float)
    except Exception:
        return None, 0

    idxs = [i for i, ok in enumerate(occur_mask.values.tolist()) if bool(ok)]
    idxs = [i for i in idxs if i + h < len(df)]
    if len(idxs) < 10:
        return None, len(idxs)

    mfes: list[float] = []
    for i in idxs:
        ref = float(close.iloc[i])
        if direction.upper() == "LONG":
            fwd_max = float(hi.iloc[i + 1 : i + h + 1].max())
            mfes.append(max(0.0, fwd_max - ref))
        else:
            fwd_min = float(lo.iloc[i + 1 : i + h + 1].min())
            mfes.append(max(0.0, ref - fwd_min))

    if not mfes:
        return None, 0

    mfes.sort()
    k = int(round((pct / 100.0) * (len(mfes) - 1)))
    k = max(0, min(len(mfes) - 1, k))
    return float(mfes[k]), len(mfes)


def _tp3_from_expected_excursion(
    df: pd.DataFrame,
    *,
    direction: str,
    signature: dict,
    entry_px: float,
    interval_mins: int,
    lookback_bars: int = 600,
    horizon_bars: int | None = None,
) -> tuple[float | None, dict]:
    """Compute TP3 using expected excursion (rolling MFE) for similar historical signatures.

    Lightweight rolling backtest per symbol+interval:
    - Find prior bars where the same boolean signature fired
    - Compute forward Max Favorable Excursion (MFE) over horizon
    - Use a high percentile (95th) as TP3 (runner/lottery)

    Returns (tp3, diagnostics).
    """
    diag = {
        "tp3_mode": "mfe_p95",
        "samples": 0,
        "horizon_bars": None,
        "signature": signature,
    }
    if df is None or len(df) < 60:
        return None, diag

    try:
        n = int(lookback_bars)
    except Exception:
        n = 600
    n = max(120, min(len(df), n))
    d = df.iloc[-n:].copy()

    # Default horizon: 1m -> 15 bars (15m); 5m -> 6 bars (~30m)
    if horizon_bars is None:
        hb = 15 if int(interval_mins) <= 1 else 6
    else:
        hb = int(horizon_bars)
    hb = max(3, hb)
    diag["horizon_bars"] = hb

    # vwap series for signature matching (prefer a precomputed 'vwap_use')
    if "vwap_use" in d.columns:
        vwap_use = d["vwap_use"].astype(float)
    elif "vwap_sess" in d.columns:
        vwap_use = d["vwap_sess"].astype(float)
    elif "vwap_cum" in d.columns:
        vwap_use = d["vwap_cum"].astype(float)
    else:
        return None, diag

    close = d["close"].astype(float)

    # Recompute simple boolean events in-window to find prior occurrences.
    was_below = (close.shift(3) < vwap_use.shift(3)) | (close.shift(5) < vwap_use.shift(5))
    reclaim = (close > vwap_use) & (close.shift(1) <= vwap_use.shift(1))
    was_above = (close.shift(3) > vwap_use.shift(3)) | (close.shift(5) > vwap_use.shift(5))
    reject = (close < vwap_use) & (close.shift(1) >= vwap_use.shift(1))

    rsi5 = d.get("rsi5")
    rsi14 = d.get("rsi14")
    macd_hist = d.get("macd_hist")
    vol = d.get("volume")

    if rsi5 is not None:
        rsi5 = rsi5.astype(float)
    if rsi14 is not None:
        rsi14 = rsi14.astype(float)
    if macd_hist is not None:
        macd_hist = macd_hist.astype(float)

    # RSI events (match current engine semantics approximately)
    rsi_snap = None
    rsi_down = None
    if rsi5 is not None:
        rsi_snap = ((rsi5 >= 30) & (rsi5.shift(1) < 30)) | ((rsi5 >= 25) & (rsi5.shift(1) < 25))
        rsi_down = ((rsi5 <= 70) & (rsi5.shift(1) > 70)) | ((rsi5 <= 75) & (rsi5.shift(1) > 75))

    # MACD turns
    macd_up = None
    macd_dn = None
    if macd_hist is not None:
        macd_up = (macd_hist > macd_hist.shift(1)) & (macd_hist.shift(1) > macd_hist.shift(2))
        macd_dn = (macd_hist < macd_hist.shift(1)) & (macd_hist.shift(1) < macd_hist.shift(2))

    # Volume confirm: last bar volume >= multiplier * rolling median(30)
    vol_ok = None
    if vol is not None:
        v = vol.astype(float)
        med = v.rolling(30, min_periods=10).median()
        mult = float(signature.get("vol_mult") or 1.25)
        vol_ok = v >= (mult * med)

    # Micro-structure: higher-low / lower-high
    hl_ok = None
    lh_ok = None
    try:
        lows = d["low"].astype(float)
        highs = d["high"].astype(float)
        hl_ok = lows.iloc[-1] > lows.rolling(10, min_periods=5).min()
        lh_ok = highs.iloc[-1] < highs.rolling(10, min_periods=5).max()
    except Exception:
        pass

    # Build occurrence mask to match the CURRENT signature
    diru = direction.upper()
    if diru == "LONG":
        m = (was_below & reclaim)
        if signature.get("rsi_event") and rsi_snap is not None:
            m = m & rsi_snap
        if signature.get("macd_event") and macd_up is not None:
            m = m & macd_up
        if signature.get("vol_event") and vol_ok is not None:
            m = m & vol_ok
        if signature.get("struct_event") and hl_ok is not None:
            m = m & hl_ok
    else:
        m = (was_above & reject)
        if signature.get("rsi_event") and rsi_down is not None:
            m = m & rsi_down
        if signature.get("macd_event") and macd_dn is not None:
            m = m & macd_dn
        if signature.get("vol_event") and vol_ok is not None:
            m = m & vol_ok
        if signature.get("struct_event") and lh_ok is not None:
            m = m & lh_ok

    mfe95, n_samples = _mfe_percentile_from_history(d, direction=diru, occur_mask=m.fillna(False), horizon_bars=hb, pct=95.0)
    diag["samples"] = int(n_samples)
    if mfe95 is None or not np.isfinite(mfe95):
        return None, diag

    try:
        mfe95 = float(mfe95)
        if diru == "LONG":
            return float(entry_px) + mfe95, diag
        return float(entry_px) - mfe95, diag
    except Exception:
        return None, diag

def _candidate_levels_from_context(
    *,
    levels: Dict[str, Any],
    recent_swing_high: float,
    recent_swing_low: float,
    hi: float,
    lo: float,
) -> Dict[str, float]:
    """Collect common structure/liquidity levels into a flat dict of floats.

    We use these as *potential* scalp targets (TP0). We intentionally favor
    levels that are meaningful to traders (prior day hi/lo, ORB, swing pivots),
    but fall back gracefully when some session levels aren't available.
    """
    out: Dict[str, float] = {}

    def _add(name: str, v: Any):
        try:
            if v is None:
                return
            fv = float(v)
            if np.isfinite(fv):
                out[name] = fv
        except Exception:
            return

    # Session liquidity levels (may be None)
    _add("orb_high", levels.get("orb_high"))
    _add("orb_low", levels.get("orb_low"))
    _add("prior_high", levels.get("prior_high"))
    _add("prior_low", levels.get("prior_low"))
    _add("premarket_high", levels.get("premarket_high"))
    _add("premarket_low", levels.get("premarket_low"))

    # Swing + range context
    _add("recent_swing_high", recent_swing_high)
    _add("recent_swing_low", recent_swing_low)
    _add("range_high", hi)
    _add("range_low", lo)
    return out


def _pick_tp0(
    direction: str,
    *,
    entry_px: float,
    last_px: float,
    atr_last: float,
    levels: Dict[str, float],
) -> Optional[float]:
    """Pick TP0 as the nearest meaningful level beyond entry.

    For scalping, TP0 should usually be *closer* than 1R/2R and should map to
    real structure. If no structure exists in-range, we fall back to an ATR-based
    objective.
    """
    try:
        entry_px = float(entry_px)
        last_px = float(last_px)
    except Exception:
        return None

    max_dist = None
    if atr_last and atr_last > 0:
        # Don't pick a target 10 ATR away for a scalp; keep it sane.
        max_dist = 3.0 * float(atr_last)

    cands: List[float] = []
    if direction == "LONG":
        for _, lvl in levels.items():
            if lvl > entry_px:
                cands.append(float(lvl))
        if cands:
            tp0 = min(cands, key=lambda x: abs(x - entry_px))
            if max_dist is None or abs(tp0 - entry_px) <= max_dist:
                return float(tp0)
        # Fallback: small objective beyond last/entry
        bump = 0.8 * float(atr_last) if atr_last else max(0.001 * last_px, 0.01)
        return float(max(entry_px, last_px) + bump)

    # SHORT
    for _, lvl in levels.items():
        if lvl < entry_px:
            cands.append(float(lvl))
    if cands:
        tp0 = min(cands, key=lambda x: abs(x - entry_px))
        if max_dist is None or abs(tp0 - entry_px) <= max_dist:
            return float(tp0)
    bump = 0.8 * float(atr_last) if atr_last else max(0.001 * last_px, 0.01)
    return float(min(entry_px, last_px) - bump)


def _eta_minutes_to_tp0(
    *,
    last_px: float,
    tp0: Optional[float],
    atr_last: float,
    interval_mins: int,
    liquidity_mult: float,
) -> Optional[float]:
    """Rough expected minutes to TP0 using ATR as a speed proxy.

    This is not meant to be precise. It's a UI helper to detect *slow* setups
    (common midday / low-liquidity conditions).
    """
    try:
        if tp0 is None:
            return None
        if not atr_last or atr_last <= 0:
            return None
        dist = abs(float(tp0) - float(last_px))
        bars = dist / float(atr_last)
        # liquidity_mult >1 means faster; <1 slower.
        speed = max(0.5, float(liquidity_mult))
        mins = bars * float(interval_mins) / speed
        return float(min(max(mins, 0.0), 999.0))
    except Exception:
        return None


def _entry_limit_and_chase(
    direction: str,
    *,
    entry_px: float,
    last_px: float,
    atr_last: float,
    slippage_mode: str,
    fixed_slippage_cents: float,
    atr_fraction_slippage: float,
) -> Tuple[float, float]:
    """Return (limit_entry, chase_line).

    - limit_entry: your planned limit.
    - chase_line: a "max pain" price where, if crossed, you're late and should
      reassess or switch to a different execution model.
    """
    slip = _slip_amount(
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_last=float(atr_last or 0.0),
        atr_fraction_slippage=float(atr_fraction_slippage or 0.0),
    )
    try:
        entry_px = float(entry_px)
        last_px = float(last_px)
    except Exception:
        return entry_px, entry_px

    # "Chase" is intentionally tight for scalps.
    chase_pad = 0.25 * float(atr_last) if atr_last else max(0.001 * last_px, 0.01)
    if direction == "LONG":
        chase = max(entry_px, last_px) + chase_pad + slip
        return float(entry_px), float(chase)
    chase = min(entry_px, last_px) - chase_pad - slip
    return float(entry_px), float(chase)


def _is_rising(series: pd.Series, bars: int = 3) -> bool:
    """Simple monotonic rise check over the last N bars."""
    try:
        s = series.dropna().tail(int(bars))
        if len(s) < int(bars):
            return False
        return bool(all(float(s.iloc[i]) > float(s.iloc[i - 1]) for i in range(1, len(s))))
    except Exception:
        return False


def _is_falling(series: pd.Series, bars: int = 3) -> bool:
    """Simple monotonic fall check over the last N bars."""
    try:
        s = series.dropna().tail(int(bars))
        if len(s) < int(bars):
            return False
        return bool(all(float(s.iloc[i]) < float(s.iloc[i - 1]) for i in range(1, len(s))))
    except Exception:
        return False


PRESETS: Dict[str, Dict[str, float]] = {
    "Fast scalp": {
        "min_actionable_score": 70,
        "vol_multiplier": 1.15,
        "require_volume": 0,
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
    },
    "Cleaner signals": {
        "min_actionable_score": 80,
        "vol_multiplier": 1.35,
        "require_volume": 1,
        "require_macd_turn": 1,
        "require_vwap_event": 1,
        "require_rsi_event": 1,
    },
}


def _fib_retracement_levels(hi: float, lo: float) -> List[Tuple[str, float]]:
    ratios = [0.382, 0.5, 0.618, 0.786]
    rng = hi - lo
    if rng <= 0:
        return []
    # "pullback" levels for an up-move: hi - r*(hi-lo)
    return [(f"Fib {r:g}", hi - r * rng) for r in ratios]


def _fib_extensions(hi: float, lo: float) -> List[Tuple[str, float]]:
    # extensions above hi for longs, below lo for shorts (we'll mirror in logic)
    ratios = [1.0, 1.272, 1.618]
    rng = hi - lo
    if rng <= 0:
        return []
    return [(f"Ext {r:g}", hi + (r - 1.0) * rng) for r in ratios]


def _closest_level(price: float, levels: List[Tuple[str, float]]) -> Tuple[Optional[str], Optional[float], Optional[float]]:
    if not levels:
        return None, None, None
    name, lvl = min(levels, key=lambda x: abs(price - x[1]))
    return name, float(lvl), float(abs(price - lvl))


def _session_liquidity_levels(df: pd.DataFrame, interval_mins: int, orb_minutes: int):
    """Compute simple liquidity levels: prior session high/low, today's premarket high/low, and ORB high/low."""
    if df is None or len(df) < 5:
        return {}
    # normalize timestamps to ET
    if "time" in df.columns:
        ts = pd.to_datetime(df["time"])
    else:
        ts = pd.to_datetime(df.index)

    try:
        ts = ts.dt.tz_localize("America/New_York") if getattr(ts.dt, "tz", None) is None else ts.dt.tz_convert("America/New_York")
    except Exception:
        try:
            ts = ts.dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
        except Exception:
            # if tz ops fail, fall back to naive dates
            pass

    d = df.copy()
    d["_ts"] = ts
    # derive dates
    try:
        cur_date = d["_ts"].iloc[-1].date()
        dates = sorted({x.date() for x in d["_ts"] if pd.notna(x)})
    except Exception:
        cur_date = pd.to_datetime(df.index[-1]).date()
        dates = sorted({pd.to_datetime(x).date() for x in df.index})

    prev_date = dates[-2] if len(dates) >= 2 else cur_date

    def _t(x):
        try:
            return x.time()
        except Exception:
            return None

    def _is_pre(x):
        t = _t(x)
        return t is not None and (t >= pd.Timestamp("04:00").time()) and (t < pd.Timestamp("09:30").time())

    def _is_rth(x):
        t = _t(x)
        return t is not None and (t >= pd.Timestamp("09:30").time()) and (t <= pd.Timestamp("16:00").time())

    prev = d[d["_ts"].dt.date == prev_date] if "_ts" in d else df.iloc[:0]
    prev_rth = prev[prev["_ts"].apply(_is_rth)] if len(prev) else prev
    prior_high = float(prev_rth["high"].max()) if len(prev_rth) else (float(prev["high"].max()) if len(prev) else None)
    prior_low = float(prev_rth["low"].min()) if len(prev_rth) else (float(prev["low"].min()) if len(prev) else None)

    cur = d[d["_ts"].dt.date == cur_date] if "_ts" in d else df
    cur_pre = cur[cur["_ts"].apply(_is_pre)] if len(cur) else cur
    pre_hi = float(cur_pre["high"].max()) if len(cur_pre) else None
    pre_lo = float(cur_pre["low"].min()) if len(cur_pre) else None

    cur_rth = cur[cur["_ts"].apply(_is_rth)] if len(cur) else cur
    orb_bars = max(1, int(math.ceil(float(orb_minutes) / max(float(interval_mins), 1.0))))
    orb_slice = cur_rth.head(orb_bars)
    orb_hi = float(orb_slice["high"].max()) if len(orb_slice) else None
    orb_lo = float(orb_slice["low"].min()) if len(orb_slice) else None

    return {
        "prior_high": prior_high, "prior_low": prior_low,
        "premarket_high": pre_hi, "premarket_low": pre_lo,
        "orb_high": orb_hi, "orb_low": orb_lo,
    }

def _asof_slice(df: pd.DataFrame, interval_mins: int, use_last_closed_only: bool, bar_closed_guard: bool) -> pd.DataFrame:
    """Return df truncated so the last row represents the 'as-of' bar we can legally use."""
    if df is None or len(df) < 3:
        return df
    asof_idx = len(df) - 1

    # Always allow "snapshot mode" to use last fully completed bar
    if use_last_closed_only:
        asof_idx = max(0, len(df) - 2)

    if bar_closed_guard and len(df) >= 2:
        try:
            # Determine timestamp of latest bar
            if "time" in df.columns:
                last_ts = pd.to_datetime(df["time"].iloc[-1], utc=False)
            else:
                last_ts = pd.to_datetime(df.index[-1], utc=False)

            # Normalize to ET if timezone-naive
            now = pd.Timestamp.now(tz="America/New_York")
            if last_ts.tzinfo is None:
                last_ts = last_ts.tz_localize("America/New_York")
            else:
                last_ts = last_ts.tz_convert("America/New_York")

            bar_end = last_ts + pd.Timedelta(minutes=int(interval_mins))
            # If bar hasn't ended yet, step back one candle (avoid partial)
            if now < bar_end:
                asof_idx = min(asof_idx, len(df) - 2)
        except Exception:
            # If anything goes sideways, be conservative
            asof_idx = min(asof_idx, len(df) - 2)

    asof_idx = max(0, int(asof_idx))
    return df.iloc[: asof_idx + 1].copy()


def _detect_liquidity_sweep(df: pd.DataFrame, levels: dict, *, atr_last: float | None = None, buffer: float = 0.0):
    """Liquidity sweep with confirmation (reclaim + displacement).

    We only count a sweep when ALL are true on the latest bar:
      1) Liquidity grab (wick through a key level)
      2) Reclaim (close back on the 'correct' side of the level)
      3) Displacement (range >= ~1.2x ATR) to filter chop/fakes

    Returns:
      {"type": "...", "level": float(level), "confirmed": bool}
    or None.
    """
    if df is None or len(df) < 2 or not levels:
        return None

    h = float(df["high"].iloc[-1])
    l = float(df["low"].iloc[-1])
    c = float(df["close"].iloc[-1])

    # Displacement filter (keep it mild; still allow if ATR isn't available)
    disp_ok = True
    if atr_last is not None and np.isfinite(float(atr_last)) and float(atr_last) > 0:
        disp_ok = float(h - l) >= 1.2 * float(atr_last)

    def _bull(level: float) -> Optional[dict]:
        # wick below, reclaim above
        if l < level - buffer and c > level + buffer and disp_ok:
            return {"type": "bull_sweep", "level": float(level), "confirmed": True}
        return None

    def _bear(level: float) -> Optional[dict]:
        # wick above, reclaim below
        if h > level + buffer and c < level - buffer and disp_ok:
            return {"type": "bear_sweep", "level": float(level), "confirmed": True}
        return None

    # Priority: prior day hi/lo, then premarket hi/lo
    ph = levels.get("prior_high")
    pl = levels.get("prior_low")
    if ph is not None:
        out = _bear(float(ph))
        if out:
            out["type"] = "bear_sweep_prior_high"
            return out
    if pl is not None:
        out = _bull(float(pl))
        if out:
            out["type"] = "bull_sweep_prior_low"
            return out

    pmah = levels.get("premarket_high")
    pmal = levels.get("premarket_low")
    if pmah is not None:
        out = _bear(float(pmah))
        if out:
            out["type"] = "bear_sweep_premarket_high"
            return out
    if pmal is not None:
        out = _bull(float(pmal))
        if out:
            out["type"] = "bull_sweep_premarket_low"
            return out

    return None


def _orb_three_stage(
    df: pd.DataFrame,
    *,
    orb_high: float | None,
    orb_low: float | None,
    buffer: float,
    lookback_bars: int = 30,
    accept_bars: int = 2,
) -> Dict[str, bool]:
    """ORB as a 3-stage sequence: break -> accept -> retest.

    Bull:
      - break: close crosses above orb_high
      - accept: next `accept_bars` closes stay above orb_high
      - retest: subsequent bar(s) touch orb_high (within buffer) and close back above

    Bear mirrors below orb_low.

    Returns dict with:
      {"bull_orb_seq": bool, "bear_orb_seq": bool, "bull_break": bool, "bear_break": bool}
    """
    out = {"bull_orb_seq": False, "bear_orb_seq": False, "bull_break": False, "bear_break": False}
    if df is None or len(df) < 8:
        return out

    d = df.tail(int(min(max(10, lookback_bars), len(df)))).copy()
    c = d["close"].astype(float)
    h = d["high"].astype(float)
    l = d["low"].astype(float)

    # --- Bull sequence ---
    if orb_high is not None and np.isfinite(float(orb_high)):
        level = float(orb_high)
        broke_idx = None
        for i in range(1, len(d)):
            if c.iloc[i] > level + buffer and c.iloc[i - 1] <= level + buffer:
                broke_idx = i
        if broke_idx is not None:
            out["bull_break"] = True
            # accept: next N closes remain above
            end_acc = min(len(d), broke_idx + 1 + int(accept_bars))
            acc_ok = True
            for j in range(broke_idx + 1, end_acc):
                if c.iloc[j] <= level + buffer:
                    acc_ok = False
                    break
            if acc_ok and end_acc <= len(d) - 1:
                # retest: any later bar tags level (low <= level+buffer) and closes back above
                for k in range(end_acc, len(d)):
                    if l.iloc[k] <= level + buffer and c.iloc[k] > level + buffer:
                        out["bull_orb_seq"] = True
                        break

    # --- Bear sequence ---
    if orb_low is not None and np.isfinite(float(orb_low)):
        level = float(orb_low)
        broke_idx = None
        for i in range(1, len(d)):
            if c.iloc[i] < level - buffer and c.iloc[i - 1] >= level - buffer:
                broke_idx = i
        if broke_idx is not None:
            out["bear_break"] = True
            end_acc = min(len(d), broke_idx + 1 + int(accept_bars))
            acc_ok = True
            for j in range(broke_idx + 1, end_acc):
                if c.iloc[j] >= level - buffer:
                    acc_ok = False
                    break
            if acc_ok and end_acc <= len(d) - 1:
                for k in range(end_acc, len(d)):
                    if h.iloc[k] >= level - buffer and c.iloc[k] < level - buffer:
                        out["bear_orb_seq"] = True
                        break

    return out



def _detect_rsi_divergence(
    df: pd.DataFrame,
    rsi_fast: pd.Series,
    rsi_slow: pd.Series | None = None,
    *,
    lookback: int = 160,
    pivot_lr: int = 3,
    min_price_delta_atr: float = 0.20,
    min_rsi_delta: float = 3.0,
) -> Optional[Dict[str, float | str]]:
    """Pivot-based RSI divergence with RSI-5 timing + RSI-14 validation.

    We use PRICE pivots (swing highs/lows) and compare RSI values at those pivots.
    - RSI-5 provides the timing (fast divergence signal)
    - RSI-14 acts as a validator (should not *contradict* the divergence)

    Bullish divergence:
      price pivot low2 < low1 by >= min_price_delta_atr * ATR
      AND RSI-5 at low2 > RSI-5 at low1 by >= min_rsi_delta
      AND RSI-14 at low2 >= RSI-14 at low1 - 1 (soft validation)

    Bearish divergence:
      price pivot high2 > high1 by >= min_price_delta_atr * ATR
      AND RSI-5 at high2 < RSI-5 at high1 by >= min_rsi_delta
      AND RSI-14 at high2 <= RSI-14 at high1 + 1 (soft validation)

    Returns dict like:
      {"type": "bull"|"bear", "strength": float, ...}
    """
    if df is None or len(df) < 25 or rsi_fast is None or len(rsi_fast) < 25:
        return None

    d = df.tail(int(min(max(60, lookback), len(df)))).copy()
    r5 = rsi_fast.reindex(d.index).ffill()
    if r5.isna().all():
        return None
    r14 = None
    if rsi_slow is not None:
        r14 = rsi_slow.reindex(d.index).ffill()

    # ATR for scaling (fallback to price*0.002 if missing)
    atr_last = None
    try:
        if "atr14" in d.columns and np.isfinite(float(d["atr14"].iloc[-1])):
            atr_last = float(d["atr14"].iloc[-1])
    except Exception:
        atr_last = None
    atr_scale = atr_last if (atr_last is not None and atr_last > 0) else float(d["close"].iloc[-1]) * 0.002

    # Price pivots
    lows_mask = rolling_swing_lows(d["low"], left=int(pivot_lr), right=int(pivot_lr))
    highs_mask = rolling_swing_highs(d["high"], left=int(pivot_lr), right=int(pivot_lr))
    piv_lows = d.loc[lows_mask, ["low"]].tail(6)
    piv_highs = d.loc[highs_mask, ["high"]].tail(6)

    # --- Bull divergence on the last two pivot lows ---
    if len(piv_lows) >= 2:
        a_idx = piv_lows.index[-2]
        b_idx = piv_lows.index[-1]
        p_a = float(d.loc[a_idx, "low"])
        p_b = float(d.loc[b_idx, "low"])
        r_a = float(r5.loc[a_idx])
        r_b = float(r5.loc[b_idx])

        price_ok = (p_b < p_a) and ((p_a - p_b) >= float(min_price_delta_atr) * atr_scale)
        rsi_ok = (r_b > r_a) and ((r_b - r_a) >= float(min_rsi_delta))
        slow_ok = True
        if r14 is not None and not r14.isna().all():
            try:
                s_a = float(r14.loc[a_idx])
                s_b = float(r14.loc[b_idx])
                slow_ok = (s_b >= s_a - 1.0)  # don't contradict
            except Exception:
                slow_ok = True

        if price_ok and rsi_ok and slow_ok:
            strength = float((r_b - r_a) / max(1.0, min_rsi_delta)) + float((p_a - p_b) / max(1e-9, atr_scale))
            return {"type": "bull", "strength": float(strength), "price_a": p_a, "price_b": p_b, "rsi_a": r_a, "rsi_b": r_b}

    # --- Bear divergence on the last two pivot highs ---
    if len(piv_highs) >= 2:
        a_idx = piv_highs.index[-2]
        b_idx = piv_highs.index[-1]
        p_a = float(d.loc[a_idx, "high"])
        p_b = float(d.loc[b_idx, "high"])
        r_a = float(r5.loc[a_idx])
        r_b = float(r5.loc[b_idx])

        price_ok = (p_b > p_a) and ((p_b - p_a) >= float(min_price_delta_atr) * atr_scale)
        rsi_ok = (r_b < r_a) and ((r_a - r_b) >= float(min_rsi_delta))
        slow_ok = True
        if r14 is not None and not r14.isna().all():
            try:
                s_a = float(r14.loc[a_idx])
                s_b = float(r14.loc[b_idx])
                slow_ok = (s_b <= s_a + 1.0)
            except Exception:
                slow_ok = True

        if price_ok and rsi_ok and slow_ok:
            strength = float((r_a - r_b) / max(1.0, min_rsi_delta)) + float((p_b - p_a) / max(1e-9, atr_scale))
            return {"type": "bear", "strength": float(strength), "price_a": p_a, "price_b": p_b, "rsi_a": r_a, "rsi_b": r_b}

    return None


def _compute_atr_pct_series(df: pd.DataFrame, period: int = 14):
    if df is None or len(df) < period + 2:
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr / close.replace(0, np.nan)


def _apply_atr_score_normalization(score: float, df: pd.DataFrame, lookback: int = 200, period: int = 14):
    atr_pct = _compute_atr_pct_series(df, period=period)
    if atr_pct is None:
        return score, None, None, 1.0
    cur = atr_pct.iloc[-1]
    if pd.isna(cur) or float(cur) <= 0:
        return score, (None if pd.isna(cur) else float(cur)), None, 1.0
    tail = atr_pct.dropna().tail(int(lookback))
    baseline = float(tail.median()) if len(tail) else None
    if baseline is None or baseline <= 0:
        return score, float(cur), baseline, 1.0
    scale = float(baseline / float(cur))
    scale = max(0.75, min(1.35, scale))
    return max(0.0, min(100.0, float(score) * scale)), float(cur), baseline, scale

def compute_scalp_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    rsi_fast: pd.Series,
    rsi_slow: pd.Series,
    macd_hist: pd.Series,
    *,
    mode: str = "Cleaner signals",
    pro_mode: bool = False,

    # Time / bar guards
    allow_opening: bool = True,
    allow_midday: bool = False,
    allow_power: bool = True,
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
    use_last_closed_only: bool = False,
    bar_closed_guard: bool = True,
    interval: str = "1min",

    # VWAP / Fib / HTF
    lookback_bars: int = 180,
    vwap_logic: str = "session",
    session_vwap_include_premarket: bool = False,
    session_vwap_include_afterhours: bool = False,
    fib_lookback_bars: int = 120,
    htf_bias: Optional[Dict[str, object]] = None,   # {bias, score, details}
    htf_strict: bool = False,

    # Liquidity / ORB / execution model
    killzone_preset: str = "Custom (use toggles)",
    liquidity_weighting: float = 0.55,
    orb_minutes: int = 15,
    entry_model: str = "VWAP reclaim limit",
    slippage_mode: str = "Off",
    fixed_slippage_cents: float = 0.02,
    atr_fraction_slippage: float = 0.15,

    # Score normalization
    target_atr_pct: float | None = None,
) -> SignalResult:
    if len(ohlcv) < 60:
        return SignalResult(symbol, "NEUTRAL", 0, "Not enough data", None, None, None, None, None, None, "OFF", {})

    # --- Interval parsing ---
    # interval is typically like "1min", "5min", "15min", "30min", "60min"
    interval_mins = 1
    try:
        s = str(interval).lower().strip()
        if s.endswith("min"):
            interval_mins = int(float(s.replace("min", "").strip()))
        elif s.endswith("m"):
            interval_mins = int(float(s.replace("m", "").strip()))
        else:
            interval_mins = int(float(s))
    except Exception:
        interval_mins = 1

    # --- Killzone presets ---
    # Presets can optionally override the time-of-day allow toggles.
    kz = (killzone_preset or "Custom (use toggles)").strip()
    if kz == "Opening Drive":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = True, False, False, False, False
    elif kz == "Lunch Chop":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = False, True, False, False, False
    elif kz == "Power Hour":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = False, False, True, False, False
    elif kz == "Pre-market":
        allow_opening, allow_midday, allow_power, allow_premarket, allow_afterhours = False, False, False, True, False

    # --- Snapshot / bar-closed guards ---
    try:
        df_asof = _asof_slice(ohlcv.copy(), interval_mins=interval_mins, use_last_closed_only=use_last_closed_only, bar_closed_guard=bar_closed_guard)
    except Exception:
        df_asof = ohlcv.copy()

    cfg = PRESETS.get(mode, PRESETS["Cleaner signals"])

    df = df_asof.copy().tail(int(lookback_bars)).copy()
    # --- Attach indicator series onto df for downstream helpers that expect columns ---
    # Some callers pass RSI/MACD as separate Series; downstream logic may reference df["rsi5"]/df["rsi14"]/df["macd_hist"].
    # Align by index when possible; otherwise fall back to tail-alignment by length.
    def _attach_series(_df: pd.DataFrame, col: str, s) -> None:
        if s is None:
            return
        try:
            if isinstance(s, pd.Series):
                # Prefer index alignment
                if _df.index.equals(s.index):
                    _df[col] = s
                else:
                    _df[col] = s.reindex(_df.index)
                    # If reindex produced all-NaN (e.g., different tz), tail-align values
                    if _df[col].isna().all() and len(s) >= len(_df):
                        _df[col] = pd.Series(s.values[-len(_df):], index=_df.index)
            else:
                # list/np array
                arr = list(s)
                if len(arr) >= len(_df):
                    _df[col] = pd.Series(arr[-len(_df):], index=_df.index)
        except Exception:
            # Last resort: do nothing
            return

    _attach_series(df, "rsi5", rsi_fast)
    _attach_series(df, "rsi14", rsi_slow)
    _attach_series(df, "macd_hist", macd_hist)
    # Session VWAP windows are session-dependent. If the user enables scanning PM/AH but keeps
    # session VWAP restricted to RTH, VWAP-based logic becomes NaN during those windows.
    # As a product guardrail, automatically extend session VWAP to the scanned session(s).
    auto_vwap_fix = False
    if vwap_logic == "session":
        if allow_premarket and not session_vwap_include_premarket:
            session_vwap_include_premarket = True
            auto_vwap_fix = True
        if allow_afterhours and not session_vwap_include_afterhours:
            session_vwap_include_afterhours = True
            auto_vwap_fix = True

    df["vwap_cum"] = calc_vwap(df)
    df["vwap_sess"] = calc_session_vwap(
        df,
        include_premarket=session_vwap_include_premarket,
        include_afterhours=session_vwap_include_afterhours,
    )
    df["atr14"] = calc_atr(df, 14)
    df["ema20"] = calc_ema(df["close"], 20)
    df["ema50"] = calc_ema(df["close"], 50)

    # Pro: Trend strength (ADX) + direction (DI+/DI-)
    adx14 = plus_di = minus_di = None
    try:
        adx_s, pdi_s, mdi_s = calc_adx(df, 14)
        df["adx14"] = adx_s
        df["plus_di14"] = pdi_s
        df["minus_di14"] = mdi_s
        adx14 = float(adx_s.iloc[-1]) if len(adx_s) and np.isfinite(adx_s.iloc[-1]) else None
        plus_di = float(pdi_s.iloc[-1]) if len(pdi_s) and np.isfinite(pdi_s.iloc[-1]) else None
        minus_di = float(mdi_s.iloc[-1]) if len(mdi_s) and np.isfinite(mdi_s.iloc[-1]) else None
    except Exception:
        adx14 = plus_di = minus_di = None

    rsi_fast = rsi_fast.reindex(df.index).ffill()
    rsi_slow = rsi_slow.reindex(df.index).ffill()
    macd_hist = macd_hist.reindex(df.index).ffill()

    close = df["close"]
    vol = df["volume"]
    vwap_use = df["vwap_sess"] if vwap_logic == "session" else df["vwap_cum"]
    df["vwap_use"] = vwap_use  # unify VWAP ref for downstream TP/expected-excursion logic

    last_ts = df.index[-1]
    # Feed freshness diagnostics (ET): this helps catch the "AsOf is yesterday" case.
    try:
        now_et = pd.Timestamp.now(tz="America/New_York")
        ts_et = last_ts.tz_convert("America/New_York") if last_ts.tzinfo is not None else last_ts.tz_localize("America/New_York")
        data_age_min = float((now_et - ts_et).total_seconds() / 60.0)
        extras_feed = {"data_age_min": data_age_min, "data_date": str(ts_et.date())}
    except Exception:
        extras_feed = {"data_age_min": None, "data_date": None}
    session = classify_session(last_ts)
    phase = classify_liquidity_phase(last_ts)

    # IMPORTANT PRODUCT RULE:
    # Time-of-day toggles should NOT *block* scoring/alerts.
    # They are preference hints used for liquidity weighting and optional UI filtering.
    # A great setup is a great setup regardless of clock-time.
    allowed = (
        (session == "OPENING" and allow_opening)
        or (session == "MIDDAY" and allow_midday)
        or (session == "POWER" and allow_power)
        or (session == "PREMARKET" and allow_premarket)
        or (session == "AFTERHOURS" and allow_afterhours)
    )
    last_price = float(close.iloc[-1])

    # --- Safety: define reference VWAP early so it is always in-scope ---
    # The PRE-alert logic and entry/TP models reference `ref_vwap`. In some code paths
    # (depending on toggles/returns), `ref_vwap` can otherwise be referenced before it
    # is assigned, causing UnboundLocalError.
    try:
        _rv = vwap_use.iloc[-1]
        ref_vwap: float | None = float(_rv) if _rv is not None and np.isfinite(_rv) else None
    except Exception:
        ref_vwap = None

    atr_last = float(df["atr14"].iloc[-1]) if np.isfinite(df["atr14"].iloc[-1]) else 0.0
    buffer = 0.25 * atr_last if atr_last else 0.0
    atr_pct = (atr_last / last_price) if last_price else 0.0

    # Liquidity weighting: scale contributions based on the current liquidity phase.
    # liquidity_weighting in [0..1] controls how strongly we care about time-of-day liquidity.
    #  - OPENING / POWER: boost
    #  - MIDDAY: discount
    #  - PREMARKET / AFTERHOURS: heavier discount
    base = 1.0
    if phase in ("OPENING", "POWER"):
        base = 1.15
    elif phase in ("MIDDAY",):
        base = 0.85
    elif phase in ("PREMARKET", "AFTERHOURS"):
        base = 0.75
    try:
        w = max(0.0, min(1.0, float(liquidity_weighting)))
    except Exception:
        w = 0.55
    liquidity_mult = 1.0 + w * (base - 1.0)

    extras: Dict[str, Any] = {
        "vwap_logic": vwap_logic,
        "session_vwap_include_premarket": bool(session_vwap_include_premarket),
        "session_vwap_include_afterhours": bool(session_vwap_include_afterhours),
        "auto_vwap_session_fix": bool(auto_vwap_fix),
        "vwap_session": float(df["vwap_sess"].iloc[-1]) if np.isfinite(df["vwap_sess"].iloc[-1]) else None,
        "vwap_cumulative": float(df["vwap_cum"].iloc[-1]) if np.isfinite(df["vwap_cum"].iloc[-1]) else None,
        "ema20": float(df["ema20"].iloc[-1]) if np.isfinite(df["ema20"].iloc[-1]) else None,
        "ema50": float(df["ema50"].iloc[-1]) if np.isfinite(df["ema50"].iloc[-1]) else None,
        "adx14": adx14,
        "plus_di14": plus_di,
        "minus_di14": minus_di,
        "atr14": atr_last,
        "atr_pct": atr_pct,
        "adx14": adx14,
        "plus_di14": plus_di,
        "minus_di14": minus_di,
        "liquidity_phase": phase,
        "liquidity_mult": liquidity_mult,
        "fib_lookback_bars": int(fib_lookback_bars),
        "htf_bias": htf_bias,
        "htf_strict": bool(htf_strict),
        "target_atr_pct": (float(target_atr_pct) if target_atr_pct is not None else None),
        # Diagnostics: whether the current session is inside the user's preferred windows.
        # This is NEVER used to block actionability.
        "time_filter_allowed": bool(allowed),
    }

    # Attach feed diagnostics (age/date) to every result.
    try:
        extras.update(extras_feed)
    except Exception:
        pass

    # merge feed freshness fields
    extras.update(extras_feed)

    # Do not early-return when outside preferred windows.
    # We keep scoring normally and simply annotate the result.

    # VWAP event
    was_below_vwap = (close.shift(3) < vwap_use.shift(3)).iloc[-1] or (close.shift(5) < vwap_use.shift(5)).iloc[-1]
    reclaim_vwap = (close.iloc[-1] > vwap_use.iloc[-1]) and (close.shift(1).iloc[-1] <= vwap_use.shift(1).iloc[-1])

    was_above_vwap = (close.shift(3) > vwap_use.shift(3)).iloc[-1] or (close.shift(5) > vwap_use.shift(5)).iloc[-1]
    reject_vwap = (close.iloc[-1] < vwap_use.iloc[-1]) and (close.shift(1).iloc[-1] >= vwap_use.shift(1).iloc[-1])

    # RSI + MACD events
    rsi5 = float(rsi_fast.iloc[-1])
    rsi14 = float(rsi_slow.iloc[-1])

    # Pro: RSI divergence (RSI-5 vs price pivots)
    rsi_div = None
    if pro_mode:
        try:
            rsi_div = _detect_rsi_divergence(df, rsi_fast, rsi_slow, lookback=int(min(220, max(80, lookback_bars))))
        except Exception:
            rsi_div = None
    extras["rsi_divergence"] = rsi_div

    rsi_snap = (rsi5 >= 30 and float(rsi_fast.shift(1).iloc[-1]) < 30) or (rsi5 >= 25 and float(rsi_fast.shift(1).iloc[-1]) < 25)
    rsi_downshift = (rsi5 <= 70 and float(rsi_fast.shift(1).iloc[-1]) > 70) or (rsi5 <= 75 and float(rsi_fast.shift(1).iloc[-1]) > 75)

    macd_turn_up = (macd_hist.iloc[-1] > macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] > macd_hist.shift(2).iloc[-1])
    macd_turn_down = (macd_hist.iloc[-1] < macd_hist.shift(1).iloc[-1]) and (macd_hist.shift(1).iloc[-1] < macd_hist.shift(2).iloc[-1])

    # Volume confirmation (liquidity weighted)
    vol_med = vol.rolling(30, min_periods=10).median().iloc[-1]
    vol_ok = (vol.iloc[-1] >= float(cfg["vol_multiplier"]) * vol_med) if np.isfinite(vol_med) else False

    # Swings
    swing_low_mask = rolling_swing_lows(df["low"], left=3, right=3)
    recent_swing_lows = df.loc[swing_low_mask, "low"].tail(6)
    recent_swing_low = float(recent_swing_lows.iloc[-1]) if len(recent_swing_lows) else float(df["low"].tail(12).min())

    swing_high_mask = rolling_swing_highs(df["high"], left=3, right=3)
    recent_swing_highs = df.loc[swing_high_mask, "high"].tail(6)
    recent_swing_high = float(recent_swing_highs.iloc[-1]) if len(recent_swing_highs) else float(df["high"].tail(12).max())

    # Trend context (EMA)
    trend_long_ok = bool((close.iloc[-1] >= df["ema20"].iloc[-1]) and (df["ema20"].iloc[-1] >= df["ema50"].iloc[-1]))
    trend_short_ok = bool((close.iloc[-1] <= df["ema20"].iloc[-1]) and (df["ema20"].iloc[-1] <= df["ema50"].iloc[-1]))
    extras["trend_long_ok"] = trend_long_ok
    extras["trend_short_ok"] = trend_short_ok

    # Fib context (scoring + fib-anchored take profits)
    seg = df.tail(int(min(max(60, fib_lookback_bars), len(df))))
    hi = float(seg["high"].max())
    lo = float(seg["low"].min())
    rng = hi - lo

    fib_name = fib_level = fib_dist = None
    fib_near_long = fib_near_short = False
    fib_bias = "range"
    retr = _fib_retracement_levels(hi, lo) if rng > 0 else []
    fib_name, fib_level, fib_dist = _closest_level(last_price, retr)

    if rng > 0:
        pos = (last_price - lo) / rng
        if pos >= 0.60:
            fib_bias = "up"
        elif pos <= 0.40:
            fib_bias = "down"
        else:
            fib_bias = "range"

    if fib_level is not None and fib_dist is not None:
        # Volatility-aware proximity: tighter when ATR is small, wider when ATR is large.
        # For scalping, we don't want "near fib" firing when price is far away in ATR terms.
        prox = None
        if atr_last is not None and np.isfinite(float(atr_last)) and float(atr_last) > 0:
            prox = max(0.35 * float(atr_last), 0.0015 * float(last_price))
        else:
            prox = 0.002 * float(last_price)
        near = float(fib_dist) <= max(float(buffer), float(prox))
        if near:
            if fib_bias == "up":
                fib_near_long = True
            elif fib_bias == "down":
                fib_near_short = True

    extras["fib_hi"] = hi if rng > 0 else None
    extras["fib_lo"] = lo if rng > 0 else None
    extras["fib_bias"] = fib_bias
    extras["fib_closest"] = {"name": fib_name, "level": fib_level, "dist": fib_dist}
    extras["fib_near_long"] = fib_near_long
    extras["fib_near_short"] = fib_near_short

    # Liquidity sweeps + ORB context
    # Use session-aware levels (prior day high/low, premarket high/low, ORB high/low) when possible.
    try:
        levels = _session_liquidity_levels(df, interval_mins=interval_mins, orb_minutes=int(orb_minutes))
    except Exception:
        levels = {}

    extras["liq_levels"] = levels

    # Fallback swing-based levels (always available)
    prior_swing_high = float(recent_swing_highs.iloc[-1]) if len(recent_swing_highs) else float(df["high"].tail(30).max())
    prior_swing_low = float(recent_swing_lows.iloc[-1]) if len(recent_swing_lows) else float(df["low"].tail(30).min())

    # Sweep definition:
    # - Primary: wick through a key level, then close back inside (ICT-style)
    # - Secondary fallback: take + reclaim against recent swing
    bull_sweep = False
    bear_sweep = False
    if pro_mode and levels:
        sweep = _detect_liquidity_sweep(df, levels, atr_last=atr_last, buffer=buffer)
        extras["liquidity_sweep"] = sweep
        if isinstance(sweep, dict) and sweep.get("type"):
            stype = str(sweep.get("type")).lower()
            bull_sweep = stype.startswith("bull")
            bear_sweep = stype.startswith("bear")
    else:
        bull_sweep = bool((df["low"].iloc[-1] < prior_swing_low) and (df["close"].iloc[-1] > prior_swing_low))
        bear_sweep = bool((df["high"].iloc[-1] > prior_swing_high) and (df["close"].iloc[-1] < prior_swing_high))

    extras["bull_liquidity_sweep"] = bool(bull_sweep)
    extras["bear_liquidity_sweep"] = bool(bear_sweep)

    # ORB bias (upgraded): 3-stage sequence (break → accept → retest)
    orb_high = levels.get("orb_high")
    orb_low = levels.get("orb_low")
    extras["orb_high"] = orb_high
    extras["orb_low"] = orb_low

    orb_seq = _orb_three_stage(
        df,
        orb_high=float(orb_high) if orb_high is not None else None,
        orb_low=float(orb_low) if orb_low is not None else None,
        buffer=float(buffer),
        lookback_bars=int(max(24, orb_minutes * 3)),  # ~last ~2 hours on 5m, ~6 bars on 1m
        accept_bars=2,
    )
    orb_bull = bool(orb_seq.get("bull_orb_seq"))
    orb_bear = bool(orb_seq.get("bear_orb_seq"))
    # keep break-only flags for diagnostics/UI
    extras["orb_bull_break"] = bool(orb_seq.get("bull_break"))
    extras["orb_bear_break"] = bool(orb_seq.get("bear_break"))
    extras["orb_bull_seq"] = orb_bull
    extras["orb_bear_seq"] = orb_bear


    # FVG + OB + Breaker
    bull_fvg, bear_fvg = detect_fvg(df.tail(60))
    extras["bull_fvg"] = bull_fvg
    extras["bear_fvg"] = bear_fvg

    ob_bull = find_order_block(df, df["atr14"], side="bull", lookback=35)
    ob_bear = find_order_block(df, df["atr14"], side="bear", lookback=35)
    extras["bull_ob"] = ob_bull
    extras["bear_ob"] = ob_bear
    bull_ob_retest = bool(ob_bull[0] is not None and in_zone(last_price, ob_bull[0], ob_bull[1], buffer=buffer))
    bear_ob_retest = bool(ob_bear[0] is not None and in_zone(last_price, ob_bear[0], ob_bear[1], buffer=buffer))
    extras["bull_ob_retest"] = bull_ob_retest
    extras["bear_ob_retest"] = bear_ob_retest

    brk_bull = find_breaker_block(df, df["atr14"], side="bull", lookback=60)
    brk_bear = find_breaker_block(df, df["atr14"], side="bear", lookback=60)
    extras["bull_breaker"] = brk_bull
    extras["bear_breaker"] = brk_bear
    bull_breaker_retest = bool(brk_bull[0] is not None and in_zone(last_price, brk_bull[0], brk_bull[1], buffer=buffer))
    bear_breaker_retest = bool(brk_bear[0] is not None and in_zone(last_price, brk_bear[0], brk_bear[1], buffer=buffer))
    extras["bull_breaker_retest"] = bull_breaker_retest
    extras["bear_breaker_retest"] = bear_breaker_retest

    displacement = bool(atr_last and float(df["high"].iloc[-1] - df["low"].iloc[-1]) >= 1.5 * atr_last)
    extras["displacement"] = displacement

    # HTF bias overlay
    htf_b = None
    if isinstance(htf_bias, dict):
        htf_b = htf_bias.get("bias")
    extras["htf_bias_value"] = htf_b

    # --- Scoring (raw) ---
    contrib: Dict[str, Dict[str, int]] = {"LONG": {}, "SHORT": {}}

    def _add(side: str, key: str, pts: int, why: str | None = None):
        nonlocal long_points, short_points
        if side == "LONG":
            long_points += int(pts)
            contrib["LONG"][key] = contrib["LONG"].get(key, 0) + int(pts)
            if why:
                long_reasons.append(why)
        else:
            short_points += int(pts)
            contrib["SHORT"][key] = contrib["SHORT"].get(key, 0) + int(pts)
            if why:
                short_reasons.append(why)

    long_points = 0
    long_reasons: List[str] = []
    if was_below_vwap and reclaim_vwap:
        _add("LONG", "vwap_event", 35, f"VWAP reclaim ({vwap_logic})")
    if rsi_snap and rsi14 < 60:
        _add("LONG", "rsi_snap", 20, "RSI-5 snapback (RSI-14 ok)")
    if macd_turn_up:
        _add("LONG", "macd_turn", 20, "MACD hist turning up")
    if vol_ok:
        _add("LONG", "volume", int(round(15 * liquidity_mult)), "Volume confirmation")
    if df["low"].tail(12).iloc[-1] > df["low"].tail(12).min():
        _add("LONG", "micro_structure", 10, "Higher-low micro structure")

    short_points = 0
    short_reasons: List[str] = []
    if was_above_vwap and reject_vwap:
        _add("SHORT", "vwap_event", 35, f"VWAP rejection ({vwap_logic})")
    if rsi_downshift and rsi14 > 40:
        _add("SHORT", "rsi_downshift", 20, "RSI-5 downshift (RSI-14 ok)")
    if macd_turn_down:
        _add("SHORT", "macd_turn", 20, "MACD hist turning down")
    if vol_ok:
        _add("SHORT", "volume", int(round(15 * liquidity_mult)), "Volume confirmation")
    if df["high"].tail(12).iloc[-1] < df["high"].tail(12).max():
        _add("SHORT", "micro_structure", 10, "Lower-high micro structure")

    # Fib scoring (volatility-aware, cluster-gated)
    # Fib/FVG should only matter when clustered with structure + volatility context.
    micro_hl = bool(df["low"].tail(12).iloc[-1] > df["low"].tail(12).min())
    micro_lh = bool(df["high"].tail(12).iloc[-1] < df["high"].tail(12).max())
    long_structure_ok = bool((was_below_vwap and reclaim_vwap) or micro_hl or orb_bull)
    short_structure_ok = bool((was_above_vwap and reject_vwap) or micro_lh or orb_bear)
    vol_context_ok = bool(vol_ok or displacement)

    if fib_near_long and fib_name is not None and long_structure_ok and vol_context_ok:
        add = 15 if ("0.5" in fib_name or "0.618" in fib_name) else 8
        _add("LONG", "fib", add, f"Fib cluster ({fib_name})")
    if fib_near_short and fib_name is not None and short_structure_ok and vol_context_ok:
        add = 15 if ("0.5" in fib_name or "0.618" in fib_name) else 8
        _add("SHORT", "fib", add, f"Fib cluster ({fib_name})")


    # Pro structure scoring
    if pro_mode:
        if isinstance(rsi_div, dict) and rsi_div.get("type") == "bull":
            _add("LONG", "rsi_divergence", 22, "RSI bullish divergence")
        if isinstance(rsi_div, dict) and rsi_div.get("type") == "bear":
            _add("SHORT", "rsi_divergence", 22, "RSI bearish divergence")
        if bull_sweep:
            _add("LONG", "liquidity_sweep", int(round(20 * liquidity_mult)), "Liquidity sweep (low)")
        if bear_sweep:
            _add("SHORT", "liquidity_sweep", int(round(20 * liquidity_mult)), "Liquidity sweep (high)")
        if orb_bull:
            _add("LONG", "orb", int(round(12 * liquidity_mult)), f"ORB seq (break→accept→retest, {orb_minutes}m)")
        if orb_bear:
            _add("SHORT", "orb", int(round(12 * liquidity_mult)), f"ORB seq (break→accept→retest, {orb_minutes}m)")
        if bull_ob_retest:
            _add("LONG", "order_block", 15, "Bullish order block retest")
        if bear_ob_retest:
            _add("SHORT", "order_block", 15, "Bearish order block retest")
                # FVG only matters when price is actually interacting with the gap AND structure/vol context agrees.
        if bull_fvg is not None and isinstance(bull_fvg, (tuple, list)) and len(bull_fvg) == 2:
            z0, z1 = float(min(bull_fvg)), float(max(bull_fvg))
            near_fvg = (last_price >= z0 - buffer) and (last_price <= z1 + buffer)
            if near_fvg and long_structure_ok and vol_context_ok:
                _add("LONG", "fvg", 10, "Bullish FVG cluster")
        if bear_fvg is not None and isinstance(bear_fvg, (tuple, list)) and len(bear_fvg) == 2:
            z0, z1 = float(min(bear_fvg)), float(max(bear_fvg))
            near_fvg = (last_price >= z0 - buffer) and (last_price <= z1 + buffer)
            if near_fvg and short_structure_ok and vol_context_ok:
                _add("SHORT", "fvg", 10, "Bearish FVG cluster")
        if bull_breaker_retest:
            _add("LONG", "breaker", 20, "Bullish breaker retest")
        if bear_breaker_retest:
            _add("SHORT", "breaker", 20, "Bearish breaker retest")
        if displacement:
            _add("LONG", "displacement", 5, None)
            _add("SHORT", "displacement", 5, None)

        # ADX trend-strength bonus (directional): helps avoid low-energy chop.
        # - If ADX is strong and DI agrees with direction => small bonus.
        # - If ADX is very low => mild penalty (but don't over-filter reversal setups).
        try:
            adx_val = float(adx14) if adx14 is not None else None
            pdi_val = float(plus_di) if plus_di is not None else None
            mdi_val = float(minus_di) if minus_di is not None else None
        except Exception:
            adx_val = pdi_val = mdi_val = None

        if adx_val is not None and np.isfinite(adx_val):
            if adx_val >= 20 and pdi_val is not None and mdi_val is not None:
                if pdi_val > mdi_val:
                    _add("LONG", "adx_trend", 8, "ADX trend strength (DI+)")
                elif mdi_val > pdi_val:
                    _add("SHORT", "adx_trend", 8, "ADX trend strength (DI-)")
            elif adx_val <= 15:
                # Penalize both slightly during very low trend strength
                long_points = max(0, long_points - 5)
                short_points = max(0, short_points - 5)
                contrib["LONG"]["adx_chop_penalty"] = contrib["LONG"].get("adx_chop_penalty", 0) - 5
                contrib["SHORT"]["adx_chop_penalty"] = contrib["SHORT"].get("adx_chop_penalty", 0) - 5

        if not trend_long_ok and not (was_below_vwap and reclaim_vwap):
            long_points = max(0, long_points - 15)
        if not trend_short_ok and not (was_above_vwap and reject_vwap):
            short_points = max(0, short_points - 15)

    # HTF overlay scoring
    if htf_b in ("BULL", "BEAR"):
        if htf_b == "BULL":
            long_points += 10; long_reasons.append("HTF bias bullish")
            short_points = max(0, short_points - 10)
        elif htf_b == "BEAR":
            short_points += 10; short_reasons.append("HTF bias bearish")
            long_points = max(0, long_points - 10)

    # Requirements / Gatekeeping (product-safe)
    #
    # Product philosophy:
    #   - Score represents *setup quality*.
    #   - Actionability represents *tradeability* (do we have enough confirmation to plan an entry/stop/targets).
    #
    # We do this with a "confirmation score" (count of independent confirmations) and a
    # "soft-hard" volume requirement:
    #   - Volume is still required for alerting *unless* we have strong Pro confluence
    #     (sweep/OB/breaker/ORB + divergence), so we don't miss real money-makers.
    #
    # Confirmation components are boolean (0/1) and deliberately simple:
    #   confirmation_score = vwap + orb + rsi + micro_structure + volume + divergence + liquidity + fib
    #
    # NOTE: Time-of-day filters do NOT block actionability. They only affect liquidity weighting
    # (via liquidity_mult) and UI display.

    vwap_event = bool((was_below_vwap and reclaim_vwap) or (was_above_vwap and reject_vwap))
    rsi_event = bool(rsi_snap or rsi_downshift)
    macd_event = bool(macd_turn_up or macd_turn_down)
    volume_event = bool(vol_ok)

    # Micro-structure flags (used for confirmation, not direction)
    micro_hl = bool(df["low"].tail(12).iloc[-1] > df["low"].tail(12).min())
    micro_lh = bool(df["high"].tail(12).iloc[-1] < df["high"].tail(12).max())
    micro_structure_event = bool(micro_hl or micro_lh)

    is_extended_session = session in ("PREMARKET", "AFTERHOURS")

    # Pro structural trigger (if enabled)
    pro_trigger = False
    divergence_event = False
    if pro_mode:
        divergence_event = bool(isinstance(rsi_div, dict) and rsi_div.get("type") in ("bull", "bear"))
        pro_trigger = bool(
            bull_sweep or bear_sweep
            or bull_ob_retest or bear_ob_retest
            or bull_breaker_retest or bear_breaker_retest
            or orb_bull or orb_bear
            or divergence_event
        )
    extras["pro_trigger"] = bool(pro_trigger)

    # Strong Pro confluence: 2+ independent Pro triggers (plus divergence counts as a trigger)
    # This is the override that can allow alerts even without the simplistic volume flag.
    pro_triggers_count = 0
    if pro_mode:
        pro_triggers_count += 1 if (bull_sweep or bear_sweep) else 0
        pro_triggers_count += 1 if (bull_ob_retest or bear_ob_retest) else 0
        pro_triggers_count += 1 if (bull_breaker_retest or bear_breaker_retest) else 0
        pro_triggers_count += 1 if (orb_bull or orb_bear) else 0
        pro_triggers_count += 1 if divergence_event else 0
    strong_pro_confluence = bool(pro_mode and pro_triggers_count >= 2)

    # Confirmation score (0..8)
    orb_event = bool(orb_bull or orb_bear)
    liquidity_event = bool((bull_sweep or bear_sweep) or (bull_ob_retest or bear_ob_retest) or (bull_breaker_retest or bear_breaker_retest))
    fib_event = bool(fib_near_long or fib_near_short)

    confirmation_components = {
        "vwap": int(vwap_event),
        "orb": int(orb_event),
        "rsi": int(rsi_event),
        "micro_structure": int(micro_structure_event),
        "volume": int(volume_event),
        "divergence": int(divergence_event),
        "liquidity": int(liquidity_event),
        "fib": int(fib_event),
    }
    confirmation_score = int(sum(confirmation_components.values()))
    extras["confirmation_components"] = confirmation_components
    extras["confirmation_score"] = confirmation_score
    extras["strong_pro_confluence"] = bool(strong_pro_confluence)

    # Preserve gate diagnostics (used in UI/why strings)
    extras["gates"] = {
        "vwap_event": vwap_event,
        "rsi_event": rsi_event,
        "macd_event": macd_event,
        "volume_event": volume_event,
        "extended_session": bool(is_extended_session),
        "confirmation_score": confirmation_score,
        "strong_pro_confluence": bool(strong_pro_confluence),
    }

    # Confirm threshold: require multiple independent confirmations before we emit entry/TP or alert.
    # Pro mode gets a slightly lower threshold because we have more independent features.
    confirm_threshold = 4 if not pro_mode else 3
    extras["confirm_threshold"] = int(confirm_threshold)

    # PRE vs CONFIRMED stages
    # ----------------------
    # Goal: fire *earlier* (pre-trigger) alerts when a high-quality setup is forming,
    # without removing the confirmed (fully gated) alert. We do this by allowing a
    # PRE stage when price is approaching the planned trigger (usually VWAP) with
    # supportive momentum/structure, but before the reclaim/rejection event prints.
    #
    # Stages are stored in extras["stage"]:
    #   - "PRE"        : forming setup, provides an entry/stop/TP plan
    #   - "CONFIRMED"  : classic gated setup (confirm_threshold met + hard gates)
    stage: str | None = None
    stage_note: str = ""

    # Trigger-proximity used for PRE alerts
    # -------------------------------
    # PRE alerts should be *trigger proximity* driven (distance to the trigger line, normalized by ATR),
    # not only score thresholds or "actionable transition".
    #
    # Today the most common trigger line is VWAP (session or cumulative). If VWAP is unavailable (NaN)
    # we still allow PRE when Pro structural trigger exists, but proximity math is skipped.
    prox_atr = None
    prox_abs = None
    try:
        prox_abs = max(0.35 * float(atr_last or 0.0), 0.0008 * float(last_price or 0.0))
    except Exception:
        prox_abs = None

    trigger_near = False
    if isinstance(ref_vwap, (float, int)) and isinstance(last_price, (float, int)) and isinstance(prox_abs, (float, int)) and prox_abs > 0:
        dist = abs(float(last_price) - float(ref_vwap))
        trigger_near = bool(dist <= float(prox_abs))
        try:
            if atr_last and float(atr_last) > 0:
                prox_atr = float(dist) / float(atr_last)
        except Exception:
            prox_atr = None

    extras["trigger_proximity_atr"] = prox_atr
    extras["trigger_proximity_abs"] = float(prox_abs) if isinstance(prox_abs, (float, int)) else None
    extras["trigger_near"] = bool(trigger_near)

    # Momentum/structure "pre" hints
    rsi_pre_long = bool(_is_rising(df["rsi5"], 3) and float(df["rsi5"].iloc[-1]) < 60)
    rsi_pre_short = bool(_is_falling(df["rsi5"], 3) and float(df["rsi5"].iloc[-1]) > 40)
    macd_pre_long = bool(_is_rising(df["macd_hist"], 3))
    macd_pre_short = bool(_is_falling(df["macd_hist"], 3))
    struct_pre_long = bool(micro_hl)
    struct_pre_short = bool(micro_lh)

    # Primary trigger must exist (otherwise we have nothing to anchor a plan).
    # NOTE: this is used by both PRE and CONFIRMED routing.
    primary_trigger = bool(vwap_event or rsi_event or macd_event or pro_trigger)
    extras["primary_trigger"] = primary_trigger

    # PRE condition: near trigger line on the "wrong" side, with momentum/structure pointing toward a flip.
    pre_long_ok = bool(
        isinstance(ref_vwap, (float, int))
        and isinstance(last_price, (float, int))
        and float(last_price) < float(ref_vwap)
        and trigger_near
        and (rsi_event or rsi_pre_long or macd_event or macd_pre_long or pro_trigger)
        and (struct_pre_long or liquidity_event or orb_event)
        and (confirmation_score >= max(2, confirm_threshold - 1))
    )
    pre_short_ok = bool(
        isinstance(ref_vwap, (float, int))
        and isinstance(last_price, (float, int))
        and float(last_price) > float(ref_vwap)
        and trigger_near
        and (rsi_event or rsi_pre_short or macd_event or macd_pre_short or pro_trigger)
        and (struct_pre_short or liquidity_event or orb_event)
        and (confirmation_score >= max(2, confirm_threshold - 1))
    )

    # If we're near the trigger line and the setup quality is already strong, emit PRE even if we are
    # one confirmation short (so you don't get the alert *after* the move already started).
    # This is intentionally conservative: requires proximity + at least 2 confirmations + a real trigger anchor.
    try:
        setup_quality_points = float(max(long_points_cal, short_points_cal))
    except Exception:
        setup_quality_points = float(max(long_points, short_points))
    pre_proximity_quality = bool(
        trigger_near
        and primary_trigger
        and confirmation_score >= 2
        and setup_quality_points >= float(cfg.get("min_actionable_score", 60)) * 0.85
    )
    extras["pre_proximity_quality"] = bool(pre_proximity_quality)

    # "Soft-hard" volume requirement:
    # If the preset says volume is required, we still require it UNLESS strong Pro confluence exists.
    if int(cfg.get("require_volume", 0)) == 1 and (not volume_event) and (not strong_pro_confluence):
        return SignalResult(
            symbol, "NEUTRAL", _cap_score(max(long_points, short_points)),
            "No volume confirmation",
            None, None, None, None,
            last_price, last_ts, session, extras,
        )

    if not primary_trigger:
        return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_points, short_points)), "No primary trigger (VWAP/RSI/MACD/Pro)", None, None, None, None, last_price, last_ts, session, extras)

    # Stage selection:
    #   - CONFIRMED requires full confirmation_score + hard gates.
    #   - PRE can be emitted one notch earlier (approaching VWAP) so traders can be ready.
    if confirmation_score < confirm_threshold:
        if pre_long_ok or pre_short_ok or pre_proximity_quality:
            stage = "PRE"
            stage_note = f"PRE: trigger proximity (confirmations {confirmation_score}/{confirm_threshold})"
        else:
            return SignalResult(
                symbol, "NEUTRAL", _cap_score(max(long_points, short_points)),
                f"Not enough confirmations ({confirmation_score}/{confirm_threshold})",
                None, None, None, None,
                last_price, last_ts, session, extras,
            )
    else:
        stage = "CONFIRMED"
        stage_note = f"CONFIRMED ({confirmation_score}/{confirm_threshold})"

    # Optional: keep classic hard requirements during RTH when Pro confluence is absent.
    # (These protect the "Cleaner signals" preset from becoming too loose.)
    hard_vwap = (int(cfg.get("require_vwap_event", 0)) == 1) and (not is_extended_session)
    hard_rsi  = (int(cfg.get("require_rsi_event", 0)) == 1) and (not is_extended_session)
    hard_macd = (int(cfg.get("require_macd_turn", 0)) == 1) and (not is_extended_session)

    # Hard gates apply to CONFIRMED only (PRE is allowed to form *before* these print).
    if stage == "CONFIRMED":
        if hard_vwap and (not vwap_event) and (not pro_trigger):
            # If the setup is *almost* there, degrade to PRE instead of dropping it.
            if pre_long_ok or pre_short_ok:
                stage = "PRE"; stage_note = "PRE: VWAP event not printed yet"
            else:
                return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_points, short_points)), "No VWAP reclaim/rejection event", None, None, None, None, last_price, last_ts, session, extras)
        if hard_rsi and (not rsi_event) and (not pro_trigger):
            if pre_long_ok or pre_short_ok:
                stage = "PRE"; stage_note = "PRE: RSI event not printed yet"
            else:
                return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_points, short_points)), "No RSI-5 snap/downshift event", None, None, None, None, last_price, last_ts, session, extras)
        if hard_macd and (not macd_event) and (not pro_trigger):
            if pre_long_ok or pre_short_ok:
                stage = "PRE"; stage_note = "PRE: MACD turn not printed yet"
            else:
                return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_points, short_points)), "No MACD histogram turn event", None, None, None, None, last_price, last_ts, session, extras)

    # For extended sessions (PM/AH), mark missing classic triggers for transparency.
    if is_extended_session:
        if int(cfg.get("require_vwap_event", 0)) == 1 and (not vwap_event) and (not pro_trigger):
            extras["soft_gate_missing_vwap"] = True
        if int(cfg.get("require_rsi_event", 0)) == 1 and (not rsi_event) and (not pro_trigger):
            extras["soft_gate_missing_rsi"] = True
        if int(cfg.get("require_macd_turn", 0)) == 1 and (not macd_event) and (not pro_trigger):
            extras["soft_gate_missing_macd"] = True

    # ATR-normalized score calibration (per ticker)
    # If target_atr_pct is None => auto-tune per ticker using median ATR% over a recent window.
    # Otherwise => use the manual target ATR% as a global anchor.
    scale = 1.0
    ref_atr_pct = None
    if atr_pct:
        if target_atr_pct is None:
            atr_series = df["atr14"].tail(120)
            close_series = df["close"].tail(120).replace(0, np.nan)
            atr_pct_series = (atr_series / close_series).replace([np.inf, -np.inf], np.nan).dropna()
            if len(atr_pct_series) >= 20:
                ref_atr_pct = float(np.nanmedian(atr_pct_series.values))
        else:
            ref_atr_pct = float(target_atr_pct)

        if ref_atr_pct and ref_atr_pct > 0:
            scale = ref_atr_pct / atr_pct
            # Keep calibration gentle; we want comparability, not distortion.
            scale = float(np.clip(scale, 0.75, 1.25))

    extras["atr_score_scale"] = scale
    extras["atr_ref_pct"] = ref_atr_pct

    long_points_cal = int(round(long_points * scale))
    short_points_cal = int(round(short_points * scale))
    extras["long_points_raw"] = long_points
    extras["short_points_raw"] = short_points
    extras["long_points_cal"] = long_points_cal
    extras["short_points_cal"] = short_points_cal
    extras["contrib_points"] = contrib

    min_score = int(cfg["min_actionable_score"])

    # Entry/stop + targets
    tighten_factor = 1.0
    if pro_mode:
        # Tighten stops a bit when we have structural confluence.
        # NOTE: We intentionally do NOT mutate the setup_score here; scoring is handled above.
        confluence = bool(
            (isinstance(rsi_div, dict) and rsi_div.get("type") in ("bull", "bear"))
            or bull_sweep or bear_sweep
            or orb_bull or orb_bear
            or bull_ob_retest or bear_ob_retest
            or bull_breaker_retest or bear_breaker_retest
            or (bull_fvg is not None) or (bear_fvg is not None)
        )
        if confluence:
            tighten_factor = 0.85
        extras["stop_tighten_factor"] = float(tighten_factor)

    def _fib_take_profits_long(entry_px: float) -> Tuple[Optional[float], Optional[float]]:
        if rng <= 0:
            return None, None
        exts = _fib_extensions(hi, lo)
        # Partial at recent high if above entry, else at ext 1.272
        tp1 = hi if entry_px < hi else next((lvl for _, lvl in exts if lvl > entry_px), None)
        tp2 = next((lvl for _, lvl in exts if lvl and tp1 and lvl > tp1), None)
        return (float(tp1) if tp1 else None, float(tp2) if tp2 else None)

    def _fib_take_profits_short(entry_px: float) -> Tuple[Optional[float], Optional[float]]:
        if rng <= 0:
            return None, None
        # Mirror extensions below lo
        ratios = [1.0, 1.272, 1.618]
        exts_dn = [ (f"Ext -{r:g}", lo - (r - 1.0) * rng) for r in ratios ]
        tp1 = lo if entry_px > lo else next((lvl for _, lvl in exts_dn if lvl < entry_px), None)
        tp2 = next((lvl for _, lvl in exts_dn if lvl and tp1 and lvl < tp1), None)
        return (float(tp1) if tp1 else None, float(tp2) if tp2 else None)

    def _long_entry_stop(entry_px: float):
        stop_px = float(min(recent_swing_low, entry_px - max(atr_last, 0.0) * 0.8))
        if pro_mode and tighten_factor < 1.0:
            stop_px = float(entry_px - (entry_px - stop_px) * tighten_factor)
        if bull_breaker_retest and brk_bull[0] is not None:
            stop_px = float(min(stop_px, brk_bull[0] - buffer))
        if fib_near_long and fib_level is not None:
            stop_px = float(min(stop_px, fib_level - buffer))
        return entry_px, stop_px

    def _short_entry_stop(entry_px: float):
        stop_px = float(max(recent_swing_high, entry_px + max(atr_last, 0.0) * 0.8))
        if pro_mode and tighten_factor < 1.0:
            stop_px = float(entry_px + (stop_px - entry_px) * tighten_factor)
        if bear_breaker_retest and brk_bear[1] is not None:
            stop_px = float(max(stop_px, brk_bear[1] + buffer))
        if fib_near_short and fib_level is not None:
            stop_px = float(max(stop_px, fib_level + buffer))
        return entry_px, stop_px
    # Final decision + trade levels
    long_score = int(round(float(long_points_cal))) if 'long_points_cal' in locals() else int(round(float(long_points)))
    short_score = int(round(float(short_points_cal))) if 'short_points_cal' in locals() else int(round(float(short_points)))

    # Never allow scores outside 0..100.
    long_score = _cap_score(long_score)
    short_score = _cap_score(short_score)

    if long_score < min_score and short_score < min_score:
        reason = "Score below threshold"
        extras["decision"] = {"long": long_score, "short": short_score, "min": min_score}
        return SignalResult(symbol, "NEUTRAL", _cap_score(max(long_score, short_score)), reason, None, None, None, None, last_price, last_ts, session, extras)

    # Stage + direction
    extras["stage"] = stage
    extras["stage_note"] = stage_note

    # For PRE alerts, prefer the directional pre-condition when it is unambiguous.
    if stage == "PRE" and pre_long_ok and not pre_short_ok:
        bias = "LONG"
    elif stage == "PRE" and pre_short_ok and not pre_long_ok:
        bias = "SHORT"
    else:
        bias = "LONG" if long_score >= short_score else "SHORT"
    setup_score = _cap_score(max(long_score, short_score))

    # Assemble reason text from the winning side
    if bias == "LONG":
        reasons = long_reasons[:] if 'long_reasons' in locals() else []
    else:
        reasons = short_reasons[:] if 'short_reasons' in locals() else []

    core_reason = "; ".join(reasons) if reasons else "Actionable setup"
    reason = (stage_note + " — " if stage_note else "") + core_reason

    # Entry model context
    ref_vwap = None
    try:
        ref_vwap = float(vwap_use.iloc[-1])
    except Exception:
        ref_vwap = None

    mid_price = None
    try:
        mid_price = float((df["high"].iloc[-1] + df["low"].iloc[-1]) / 2.0)
    except Exception:
        mid_price = None

    entry_px = _entry_from_model(
        bias,
        entry_model=entry_model,
        last_price=float(last_price),
        ref_vwap=ref_vwap,
        mid_price=mid_price,
        atr_last=float(atr_last) if atr_last is not None else 0.0,
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_fraction_slippage=atr_fraction_slippage,
    )

    # Entry model upgrade: expose both a limit entry and a chase-line.
    entry_limit, chase_line = _entry_limit_and_chase(
        bias,
        entry_px=float(entry_px),
        last_px=float(last_price),
        atr_last=float(atr_last) if atr_last is not None else 0.0,
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_fraction_slippage=atr_fraction_slippage,
    )

    # Entry model upgrade: adapt when the planned limit is already stale.
    # If price has already moved beyond the limit by a meaningful fraction of ATR,
    # we flip the plan to a chase-based execution so we don't alert *after* the move.
    #
    # - LONG: if last is above the limit by > stale_buffer => use chase line as the new entry.
    # - SHORT: if last is below the limit by > stale_buffer => use chase line as the new entry.
    #
    # This keeps entry/stop/TP coherent (all are computed off entry_limit) while preserving
    # the informational chase line for the trader.
    stale_buffer = None
    try:
        stale_buffer = max(0.25 * float(atr_last or 0.0), 0.0006 * float(last_price or 0.0))
    except Exception:
        stale_buffer = None

    exec_mode = "LIMIT"
    entry_stale = False
    if isinstance(stale_buffer, (float, int)) and stale_buffer and stale_buffer > 0:
        try:
            if bias == "LONG" and float(last_price) > float(entry_limit) + float(stale_buffer):
                exec_mode = "CHASE"; entry_stale = True
                entry_limit = float(chase_line)
            elif bias == "SHORT" and float(last_price) < float(entry_limit) - float(stale_buffer):
                exec_mode = "CHASE"; entry_stale = True
                entry_limit = float(chase_line)
        except Exception:
            pass

    extras["execution_mode"] = exec_mode
    extras["entry_stale"] = bool(entry_stale)
    extras["entry_stale_buffer"] = float(stale_buffer) if isinstance(stale_buffer, (float, int)) else None
    extras["entry_limit"] = float(entry_limit)
    extras["entry_chase_line"] = float(chase_line)

    # PRE tier risk tightening: smaller risk ⇒ closer TP ⇒ more hits.
    interval_mins_i = int(interval_mins) if isinstance(interval_mins, (int, float)) else 1
    pre_stop_tighten = 0.70 if stage == "PRE" else 1.0
    extras["pre_stop_tighten"] = float(pre_stop_tighten)

    if bias == "LONG":
        entry_px, stop_px = _long_entry_stop(float(entry_limit))
        if stage == "PRE":
            stop_px = float(entry_px - (entry_px - stop_px) * pre_stop_tighten)
        risk = max(1e-9, entry_px - stop_px)
        # Targeting overhaul (structure-first): TP0/TP1/TP2
        lvl_map = _candidate_levels_from_context(
            levels=levels if isinstance(levels, dict) else {},
            recent_swing_high=float(recent_swing_high),
            recent_swing_low=float(recent_swing_low),
            hi=float(hi),
            lo=float(lo),
        )
        tp0 = _pick_tp0("LONG", entry_px=entry_px, last_px=float(last_price), atr_last=float(atr_last), levels=lvl_map)
        tp1 = (tp0 + 0.9 * risk) if tp0 is not None else (entry_px + risk)
        tp2 = (tp0 + 1.8 * risk) if tp0 is not None else (entry_px + 2 * risk)
        # Optional TP3: expected excursion (rolling MFE) for similar historical signatures
        sig_key = {
            "rsi_event": bool(rsi_snap and rsi14 < 60),
            "macd_event": bool(macd_turn_up),
            "vol_event": bool(vol_ok),
            "struct_event": bool(micro_hl),
            "vol_mult": float(cfg.get("vol_multiplier", 1.25)),
        }
        tp3, tp3_diag = _tp3_from_expected_excursion(
            df, direction="LONG", signature=sig_key, entry_px=float(entry_px), interval_mins=int(interval_mins_i)
        )
        extras["tp3"] = float(tp3) if tp3 is not None else None
        extras["tp3_diag"] = tp3_diag

        # If fib extension helper is available, prefer it for pro mode.
        if pro_mode and "_fib_take_profits_long" in locals():
            f1, f2 = _fib_take_profits_long(entry_px)
            # Use fib as TP2 (runner) when it is further than our structure target.
            if f1 is not None and (tp0 is None or float(f1) > float(tp0)):
                tp1 = float(f1)
            if f2 is not None and float(f2) > float(tp1):
                tp2 = float(f2)
            extras["fib_tp1"] = float(tp1) if tp1 is not None else None
            extras["fib_tp2"] = float(tp2) if tp2 is not None else None
    else:
        entry_px, stop_px = _short_entry_stop(float(entry_limit))
        if stage == "PRE":
            stop_px = float(entry_px + (stop_px - entry_px) * pre_stop_tighten)
        risk = max(1e-9, stop_px - entry_px)
        lvl_map = _candidate_levels_from_context(
            levels=levels if isinstance(levels, dict) else {},
            recent_swing_high=float(recent_swing_high),
            recent_swing_low=float(recent_swing_low),
            hi=float(hi),
            lo=float(lo),
        )
        tp0 = _pick_tp0("SHORT", entry_px=entry_px, last_px=float(last_price), atr_last=float(atr_last), levels=lvl_map)
        tp1 = (tp0 - 0.9 * risk) if tp0 is not None else (entry_px - risk)
        tp2 = (tp0 - 1.8 * risk) if tp0 is not None else (entry_px - 2 * risk)
        sig_key = {
            "rsi_event": bool(rsi_downshift and rsi14 > 40),
            "macd_event": bool(macd_turn_down),
            "vol_event": bool(vol_ok),
            "struct_event": bool(micro_lh),
            "vol_mult": float(cfg.get("vol_multiplier", 1.25)),
        }
        tp3, tp3_diag = _tp3_from_expected_excursion(
            df, direction="SHORT", signature=sig_key, entry_px=float(entry_px), interval_mins=int(interval_mins_i)
        )
        extras["tp3"] = float(tp3) if tp3 is not None else None
        extras["tp3_diag"] = tp3_diag

        if pro_mode and "_fib_take_profits_short" in locals():
            f1, f2 = _fib_take_profits_short(entry_px)
            if f1 is not None and (tp0 is None or float(f1) < float(tp0)):
                tp1 = float(f1)
            if f2 is not None and float(f2) < float(tp1):
                tp2 = float(f2)
            extras["fib_tp1"] = float(tp1) if tp1 is not None else None
            extras["fib_tp2"] = float(tp2) if tp2 is not None else None
            extras["fib_tp1"] = float(tp1) if tp1 is not None else None
            extras["fib_tp2"] = float(tp2) if tp2 is not None else None

    # Expected time-to-TP0 UI helper
    extras["tp0"] = float(tp0) if "tp0" in locals() and tp0 is not None else None
    extras["eta_tp0_min"] = _eta_minutes_to_tp0(
        last_px=float(last_price),
        tp0=tp0 if "tp0" in locals() else None,
        atr_last=float(atr_last) if atr_last else 0.0,
        interval_mins=interval_mins_i,
        liquidity_mult=float(liquidity_mult) if "liquidity_mult" in locals() else 1.0,
    )

    extras["decision"] = {"bias": bias, "long": long_score, "short": short_score, "min": min_score}
    return SignalResult(
        symbol,
        bias,
        setup_score,
        reason,
        float(entry_px),
        float(stop_px),
        float(tp1) if tp1 is not None else None,
        float(tp2) if tp2 is not None else None,
        last_price,
        last_ts,
        session,
        extras,
    )

def _slip_amount(*, slippage_mode: str, fixed_slippage_cents: float, atr_last: float, atr_fraction_slippage: float) -> float:
    """Return slippage amount in price units (not percent)."""
    try:
        mode = (slippage_mode or "Off").strip()
    except Exception:
        mode = "Off"

    if mode == "Off":
        return 0.0

    if mode == "Fixed cents":
        try:
            return max(0.0, float(fixed_slippage_cents)) / 100.0
        except Exception:
            return 0.0

    if mode == "ATR fraction":
        try:
            return max(0.0, float(atr_last)) * max(0.0, float(atr_fraction_slippage))
        except Exception:
            return 0.0

    return 0.0
def _entry_from_model(
    direction: str,
    *,
    entry_model: str,
    last_price: float,
    ref_vwap: float | None,
    mid_price: float | None,
    atr_last: float,
    slippage_mode: str,
    fixed_slippage_cents: float,
    atr_fraction_slippage: float,
) -> float:
    """Compute an execution-realistic entry based on the selected entry model."""
    slip = _slip_amount(
        slippage_mode=slippage_mode,
        fixed_slippage_cents=fixed_slippage_cents,
        atr_last=atr_last,
        atr_fraction_slippage=atr_fraction_slippage,
    )

    model = (entry_model or "Last price").strip()

    # 1) VWAP-based: place a limit slightly beyond VWAP in the adverse direction (more realistic fills).
    if model == "VWAP reclaim limit" and isinstance(ref_vwap, (float, int)):
        return (float(ref_vwap) + slip) if direction == "LONG" else (float(ref_vwap) - slip)

    # 2) Midpoint of the last completed bar
    if model == "Midpoint (last closed bar)" and isinstance(mid_price, (float, int)):
        return (float(mid_price) + slip) if direction == "LONG" else (float(mid_price) - slip)

    # 3) Default: last price with slippage in the adverse direction
    return (float(last_price) + slip) if direction == "LONG" else (float(last_price) - slip)

# ===========================
# RIDE / Continuation signals
# ===========================

def _last_swing_level(series: pd.Series, *, kind: str, lookback: int = 60) -> float | None:
    """Return the most recent swing high/low level in the lookback window (excluding the last bar)."""
    if series is None or len(series) < 10:
        return None
    s = series.astype(float).tail(int(min(len(series), max(12, lookback))))
    flags = rolling_swing_highs(s, left=3, right=3) if kind == "high" else rolling_swing_lows(s, left=3, right=3)

    # exclude last bar (cannot be a confirmed pivot yet)
    flags = flags.iloc[:-1]
    s2 = s.iloc[:-1]

    idx = None
    for i in range(len(flags) - 1, -1, -1):
        if bool(flags.iloc[i]):
            idx = flags.index[i]
            break
    if idx is None:
        return None
    try:
        return float(s2.loc[idx])
    except Exception:
        return None


def compute_ride_signal(
    symbol: str,
    ohlcv: pd.DataFrame,
    rsi5: pd.Series,
    rsi14: pd.Series,
    macd_hist: pd.Series,
    *,
    pro_mode: bool = False,
    allow_opening: bool = True,
    allow_midday: bool = False,
    allow_power: bool = True,
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
    use_last_closed_only: bool = False,
    bar_closed_guard: bool = True,
    interval: str = "1min",
    vwap_logic: str = "session",
    session_vwap_include_premarket: bool = False,
    # kept for engine/app parity (even if not used directly in RIDE yet)
    fib_lookback_bars: int = 200,
    killzone_preset: str = "none",
    target_atr_pct: float = 0.004,
    htf_bias: dict | None = None,
    orb_minutes: int = 15,
    liquidity_weighting: float = 0.55,
    **_ignored: object,
) -> SignalResult:
    """Continuation / Drive signal family.

    Returns bias:
      - RIDE_LONG / RIDE_SHORT when trend + impulse/acceptance exists (actionable proximity)
      - CHOP when trend is insufficient or setup is not actionable yet
    """
    try:
        df = ohlcv.sort_index().copy()
    except Exception:
        df = ohlcv.copy()

    # interval mins
    try:
        interval_mins = int(str(interval).replace("min", "").strip())
    except Exception:
        interval_mins = 1

    # bar-closed guard (avoid partial last bar)
    df = _asof_slice(df, interval_mins, use_last_closed_only, bar_closed_guard)

    if df is None or len(df) < 60:
        return SignalResult(symbol, "CHOP", 0, "Not enough data for continuation scan.", None, None, None, None, None, None, "OFF", {"mode": "RIDE"})

    # attach indicators (aligned)
    df["rsi5"] = pd.to_numeric(rsi5.reindex(df.index).ffill(), errors="coerce")
    df["rsi14"] = pd.to_numeric(rsi14.reindex(df.index).ffill(), errors="coerce")
    df["macd_hist"] = pd.to_numeric(macd_hist.reindex(df.index).ffill(), errors="coerce")

    session = classify_session(df.index[-1])
    liquidity_phase = classify_liquidity_phase(df.index[-1])
    liquidity_mult = float(np.clip(0.75 + liquidity_weighting, 0.75, 1.25))

    last_ts = pd.to_datetime(df.index[-1])
    last_price = float(df["close"].iloc[-1])

    # VWAP reference
    vwap_sess = calc_session_vwap(df, include_premarket=session_vwap_include_premarket)
    vwap_cum = calc_vwap(df)
    ref_vwap_series = vwap_sess if str(vwap_logic).lower() == "session" else vwap_cum
    ref_vwap = float(ref_vwap_series.iloc[-1]) if len(ref_vwap_series) else None

    # ATR + trend stats
    atr_s = calc_atr(df, period=14).reindex(df.index).ffill()
    atr_last = float(atr_s.iloc[-1]) if len(atr_s) else None
    if atr_last is None or not np.isfinite(atr_last) or atr_last <= 0:
        atr_last = max(1e-6, float(df["high"].iloc[-10:].max() - df["low"].iloc[-10:].min()) / 10.0)

    close = df["close"].astype(float)
    ema20 = calc_ema(close, 20)
    ema50 = calc_ema(close, 50)
    adx, di_plus, di_minus = calc_adx(df, period=14)

    adx_last = float(adx.reindex(df.index).ffill().iloc[-1]) if len(adx) else float("nan")
    di_p = float(di_plus.reindex(df.index).ffill().iloc[-1]) if len(di_plus) else float("nan")
    di_m = float(di_minus.reindex(df.index).ffill().iloc[-1]) if len(di_minus) else float("nan")

    adx_floor = 20.0 if interval_mins <= 1 else 18.0
    di_gap_floor = 6.0 if interval_mins <= 1 else 5.0

    pass_adx = bool(np.isfinite(adx_last) and adx_last >= adx_floor)
    pass_di_gap = bool(np.isfinite(di_p) and np.isfinite(di_m) and abs(di_p - di_m) >= di_gap_floor)
    pass_ema_up = bool(float(ema20.iloc[-1]) > float(ema50.iloc[-1]))
    pass_ema_dn = bool(float(ema20.iloc[-1]) < float(ema50.iloc[-1]))

    trend_votes = int(pass_adx) + int(pass_di_gap) + int(pass_ema_up or pass_ema_dn)
    trend_ok = trend_votes >= 2

    if not trend_ok:
        return SignalResult(
            symbol=symbol,
            bias="CHOP",
            setup_score=0,
            reason=f"Too choppy for RIDE (trend {trend_votes}/3).",
            entry=None, stop=None, target_1r=None, target_2r=None,
            last_price=last_price, timestamp=last_ts, session=session,
            extras={"mode": "RIDE", "stage": None, "trend_votes": trend_votes, "adx": adx_last, "di_plus": di_p, "di_minus": di_m, "liquidity_phase": liquidity_phase},
        )

    # ORB / pivots / displacement
    levels = _session_liquidity_levels(df, interval_mins, orb_minutes)
    orb_high = levels.get("orb_high")
    orb_low = levels.get("orb_low")
    buffer = 0.15 * float(atr_last)

    orb_seq = _orb_three_stage(df, orb_high=orb_high, orb_low=orb_low, buffer=buffer, lookback_bars=60, accept_bars=2)
    swing_hi = _last_swing_level(df["high"], kind="high", lookback=60)
    swing_lo = _last_swing_level(df["low"], kind="low", lookback=60)

    # --- Impulse quality inputs ---
    # Displacement (range vs ATR) is a key filter for rideability.
    last_range = float(df["high"].iloc[-1] - df["low"].iloc[-1])
    disp_ratio = float(last_range / max(1e-9, float(atr_last)))
    disp_ok = disp_ratio >= 1.2
    prev_close = float(df["close"].iloc[-2])

    vwap_reclaim = bool(ref_vwap is not None and prev_close <= ref_vwap and last_price > ref_vwap and disp_ok)
    vwap_reject = bool(ref_vwap is not None and prev_close >= ref_vwap and last_price < ref_vwap and disp_ok)

    pivot_break_up = bool(swing_hi is not None and last_price > float(swing_hi) + buffer)
    pivot_break_dn = bool(swing_lo is not None and last_price < float(swing_lo) - buffer)

    orb_break_up = bool(orb_high is not None and orb_seq.get("bull_break") and last_price > float(orb_high) + buffer)
    orb_break_dn = bool(orb_low is not None and orb_seq.get("bear_break") and last_price < float(orb_low) - buffer)

    impulse_long = orb_break_up or pivot_break_up or vwap_reclaim
    impulse_short = orb_break_dn or pivot_break_dn or vwap_reject

    if not impulse_long and not impulse_short:
        return SignalResult(
            symbol=symbol,
            bias="CHOP",
            setup_score=0,
            reason="Trend present but no impulse/drive signature yet.",
            entry=None, stop=None, target_1r=None, target_2r=None,
            last_price=last_price, timestamp=last_ts, session=session,
            extras={"mode": "RIDE", "stage": None, "trend_votes": trend_votes, "liquidity_phase": liquidity_phase},
        )

    direction = None
    if impulse_long and not impulse_short:
        direction = "LONG"
    elif impulse_short and not impulse_long:
        direction = "SHORT"
    else:
        direction = "LONG" if di_p >= di_m else "SHORT"

    # accept line priority
    if direction == "LONG":
        if vwap_reclaim and ref_vwap is not None:
            accept_line, accept_src = float(ref_vwap), "VWAP"
        elif orb_high is not None and orb_break_up:
            accept_line, accept_src = float(orb_high), "ORB"
        else:
            accept_line, accept_src = float(ema20.iloc[-1]), "EMA20"
    else:
        if vwap_reject and ref_vwap is not None:
            accept_line, accept_src = float(ref_vwap), "VWAP"
        elif orb_low is not None and orb_break_dn:
            accept_line, accept_src = float(orb_low), "ORB"
        else:
            accept_line, accept_src = float(ema20.iloc[-1]), "EMA20"

    # --- Acceptance / retest logic ---
    # NOTE: In live tape, ORB/VWAP levels can be *valid* but still too far from price
    # to be a realistic pullback entry for a continuation scalp.
    #
    # Example: price is actionable via break trigger proximity, but the selected
    # accept line sits far away (stale ORB from earlier in the session). In these
    # cases we still want to surface the breakout plan, and we clamp the accept
    # line used for pullback bands into a sane ATR window around last.
    accept_line_raw = float(accept_line)
    if direction == "LONG":
        # accept line should be below last, but not absurdly far.
        lo = float(last_price - 1.20 * atr_last)
        hi = float(last_price - 0.05 * atr_last)
        accept_line = float(np.clip(accept_line_raw, lo, hi))
    else:
        # accept line should be above last, but not absurdly far.
        lo = float(last_price + 0.05 * atr_last)
        hi = float(last_price + 1.20 * atr_last)
        accept_line = float(np.clip(accept_line_raw, lo, hi))
    # Accept = closes remain on the correct side of the accept line.
    look = int(min(3, len(df) - 1))
    recent_closes = df["close"].astype(float).iloc[-look:]
    if direction == "LONG":
        accept_ok = bool((recent_closes > float(accept_line) - buffer).all())
    else:
        accept_ok = bool((recent_closes < float(accept_line) + buffer).all())

    # Retest/hold = within the last few bars, price *tests* the accept line band and holds.
    retest_look = int(min(6, len(df) - 1))
    recent_lows = df["low"].astype(float).iloc[-retest_look:]
    recent_highs = df["high"].astype(float).iloc[-retest_look:]
    if direction == "LONG":
        retest_seen = bool((recent_lows <= float(accept_line) + buffer).any())
        hold_ok = bool((recent_closes >= float(accept_line) - buffer).all())
    else:
        retest_seen = bool((recent_highs >= float(accept_line) - buffer).any())
        hold_ok = bool((recent_closes <= float(accept_line) + buffer).all())

    stage = "CONFIRMED" if (accept_ok and retest_seen and hold_ok) else "PRE"

    # volume pattern: impulse expansion + hold compression
    vol = df["volume"].astype(float)
    med30 = float(vol.tail(60).rolling(30).median().iloc[-1]) if len(vol) >= 30 else float(vol.median())
    vol_impulse = float(vol.iloc[-1])
    vol_hold = float(vol.tail(3).mean()) if len(vol) >= 3 else vol_impulse
    vol_ok = bool(med30 > 0 and (vol_impulse >= 1.5 * med30) and (vol_hold <= 1.1 * vol_impulse))

    # exhaustion guard
    r5 = float(df["rsi5"].iloc[-1]) if np.isfinite(df["rsi5"].iloc[-1]) else None
    r14 = float(df["rsi14"].iloc[-1]) if np.isfinite(df["rsi14"].iloc[-1]) else None
    exhausted = False
    if direction == "LONG" and r5 is not None and r14 is not None:
        exhausted = bool(r5 > 85 and r14 > 70)
    if direction == "SHORT" and r5 is not None and r14 is not None:
        exhausted = bool(r5 < 15 and r14 < 30)

    # RSI rideability context (not a trigger, a realism guard):
    # - Continuations are best when short-term momentum is strong *but not blown out*.
    # - We use RSI-5 for timing and RSI-14 as a validation/backdrop.
    rsi_q = 0.5
    try:
        if r5 is not None and r14 is not None:
            if direction == "LONG":
                # Ideal: RSI-5 45..78 with RSI-14 >= ~45.
                base = 1.0 if (45.0 <= r5 <= 78.0 and r14 >= 45.0) else 0.6
                # If it's very hot, require pullback/retest; otherwise reduce quality.
                if r5 >= 85.0 and r14 >= 70.0:
                    base = 0.2
                rsi_q = float(np.clip(base, 0.0, 1.0))
            else:
                base = 1.0 if (22.0 <= r5 <= 55.0 and r14 <= 55.0) else 0.6
                if r5 <= 15.0 and r14 <= 30.0:
                    base = 0.2
                rsi_q = float(np.clip(base, 0.0, 1.0))
    except Exception:
        rsi_q = 0.5

    # --- Impulse/Hold quality score (0..1) ---
    # We want Score 100 to *mean* something tradeable:
    #   - displacement strength
    #   - close in the direction of travel
    #   - impulse volume expansion + hold compression
    #   - (for CONFIRMED) accept+retest/hold quality
    try:
        close_pos = (float(df["close"].iloc[-1]) - float(df["low"].iloc[-1])) / max(1e-9, last_range)
    except Exception:
        close_pos = 0.5

    # Directional close quality: long wants close near highs; short near lows.
    close_q = float(close_pos) if direction == "LONG" else float(1.0 - close_pos)
    close_q = float(np.clip(close_q, 0.0, 1.0))

    disp_q = float(np.clip((disp_ratio - 1.0) / 1.5, 0.0, 1.0))  # ~0 at 1.0ATR, ~1 at 2.5ATR
    vol_q = 1.0 if vol_ok else 0.0
    retest_q = 1.0 if (stage == "CONFIRMED") else (0.5 if accept_ok else 0.0)
    impulse_quality = float(np.clip(0.35 * disp_q + 0.25 * close_q + 0.20 * vol_q + 0.20 * retest_q, 0.0, 1.0))

    # Fold in RSI rideability (timing realism). This does NOT create the signal;
    # it just prevents weak/overextended moves from scoring like perfect rides.
    # Keep it gentle: at most ~15% adjustment.
    impulse_quality = float(np.clip(impulse_quality * (0.85 + 0.15 * float(rsi_q)), 0.0, 1.0))

    # If we're exhausted, don't allow CONFIRMED without a retest/hold.
    if exhausted and stage == "CONFIRMED":
        stage = "PRE"

    # If the impulse/accept sequence is low quality, don't label it "rideable".
    # This keeps 100 scores from appearing on flimsy moves.
    if impulse_quality < 0.35:
        # Too weak to trade as continuation.
        return SignalResult(
            symbol=symbol,
            bias="CHOP",
            setup_score=0.0,
            reason="Not rideable (low impulse quality)",
            entry=None,
            stop=None,
            target_1r=None,
            target_2r=None,
            last_price=last_price,
            timestamp=last_ts,
            session=session,
            extras={"mode": "RIDE", "stage": None, "trend_votes": trend_votes, "impulse_quality": impulse_quality,
                    "disp_ratio": disp_ratio, "liquidity_phase": liquidity_phase},
        )

    # If quality is mediocre, allow PRE but not CONFIRMED.
    if stage == "CONFIRMED" and impulse_quality < 0.55:
        stage = "PRE"

    # --- Robustness guardrails ---
    # If the impulse/hold quality is weak, a RIDE alert isn't "rideable".
    # - Very weak quality -> CHOP (no alert)
    # - Borderline quality -> PRE only (avoid overconfident CONFIRMED labels)
    if impulse_quality < 0.35:
        return SignalResult(
            symbol=symbol,
            bias="CHOP",
            setup_score=0,
            reason=f"Trend present but impulse/hold quality too weak (Q={impulse_quality:.2f}).",
            entry=None, stop=None, target_1r=None, target_2r=None,
            last_price=last_price, timestamp=last_ts, session=session,
            extras={"mode": "RIDE", "stage": None, "trend_votes": trend_votes, "adx": adx_last, "di_plus": di_p, "di_minus": di_m,
                   "liquidity_phase": liquidity_phase, "impulse_quality": impulse_quality, "disp_ratio": disp_ratio, "vol_ok": vol_ok,
                   "accept_src": accept_src, "accept_line": accept_line},
        )
    if stage == "CONFIRMED" and impulse_quality < 0.55:
        stage = "PRE"

    # scoring (quality-weighted)
    pts = 0.0
    pts += 22.0  # base for being in a trend-filtered universe
    pts += 18.0 if pass_adx else 0.0
    pts += 12.0 if pass_di_gap else 0.0
    pts += 15.0 if (direction == "LONG" and pass_ema_up) or (direction == "SHORT" and pass_ema_dn) else 0.0

    # Impulse + acceptance are amplified by quality; weak impulses shouldn't look like 100s.
    pts += (26.0 * impulse_quality)
    pts += (14.0 * impulse_quality) if stage == "CONFIRMED" else (7.0 * impulse_quality)
    pts += (10.0 * liquidity_mult) if vol_ok else 0.0
    pts -= 12.0 if exhausted else 0.0

    if isinstance(htf_bias, dict) and "bias" in htf_bias:
        hb = str(htf_bias.get("bias", "")).upper()
        if direction == "LONG" and hb in ("BULL", "BULLISH"):
            pts += 6.0
        if direction == "SHORT" and hb in ("BEAR", "BEARISH"):
            pts += 6.0

    score = _cap_score(pts)

    # --- Entries: pullback band (PB1/PB2) + break trigger ---
    # A single-line pullback is too brittle. Bands are more realistic for continuation execution.
    #
    # Also: if the setup is actionable because we're *near the break trigger* (not near pullback),
    # we should not keep showing a stale pullback limit far away. In that case we surface a
    # breakout-style entry (stop/trigger + a small chase line).
    pb_inner = 0.25 * float(atr_last)
    pb_outer = 0.60 * float(atr_last)

    # IMPORTANT: break_trigger must be stable across refreshes.
    # Previously it was set to the latest bar's high/low, which *moves* every scan and can
    # prevent alerts from firing when price actually hits the original trigger.
    # We anchor the trigger to the most-recent "impulse" break bar.
    impulse_idx: Optional[int] = None
    try:
        if direction == "LONG":
            if impulse_type == "ORB" and isinstance(orb_hi, (float, int)):
                crossed = (df["close"] > float(orb_hi)) & (df["close"].shift(1) <= float(orb_hi))
                if crossed.any():
                    impulse_idx = int(df.index.get_loc(df.index[crossed].max()))
            elif impulse_type == "PIVOT" and isinstance(last_swh, (float, int)):
                lvl = float(last_swh)
                crossed = (df["close"] > lvl) & (df["close"].shift(1) <= lvl)
                if crossed.any():
                    impulse_idx = int(df.index.get_loc(df.index[crossed].max()))
            elif impulse_type == "VWAP" and isinstance(vwap, (float, int)):
                lvl = float(vwap)
                crossed = (df["close"] > lvl) & (df["close"].shift(1) <= lvl)
                if crossed.any():
                    impulse_idx = int(df.index.get_loc(df.index[crossed].max()))
        else:
            if impulse_type == "ORB" and isinstance(orb_lo, (float, int)):
                crossed = (df["close"] < float(orb_lo)) & (df["close"].shift(1) >= float(orb_lo))
                if crossed.any():
                    impulse_idx = int(df.index.get_loc(df.index[crossed].max()))
            elif impulse_type == "PIVOT" and isinstance(last_swl, (float, int)):
                lvl = float(last_swl)
                crossed = (df["close"] < lvl) & (df["close"].shift(1) >= lvl)
                if crossed.any():
                    impulse_idx = int(df.index.get_loc(df.index[crossed].max()))
            elif impulse_type == "VWAP" and isinstance(vwap, (float, int)):
                lvl = float(vwap)
                crossed = (df["close"] < lvl) & (df["close"].shift(1) >= lvl)
                if crossed.any():
                    impulse_idx = int(df.index.get_loc(df.index[crossed].max()))
    except Exception:
        impulse_idx = None

    if direction == "LONG":
        break_trigger = float(df["high"].iloc[impulse_idx]) if impulse_idx is not None else float(df["high"].iloc[-1])
        pb1 = float(accept_line) + pb_inner
        pb2 = float(accept_line) - pb_outer
        pullback_entry = float(np.clip(float(accept_line), pb2, pb1))
        # PRE tier: tighter stop (smaller risk ⇒ closer TP ⇒ more hits)
        stop_mult = 0.55 if stage == "PRE" else 0.80
        stop = float(pullback_entry - stop_mult * atr_last)
    else:
        break_trigger = float(df["low"].iloc[impulse_idx]) if impulse_idx is not None else float(df["low"].iloc[-1])
        pb1 = float(accept_line) - pb_inner
        pb2 = float(accept_line) + pb_outer
        pullback_entry = float(np.clip(float(accept_line), pb1, pb2))
        stop_mult = 0.55 if stage == "PRE" else 0.80
        stop = float(pullback_entry + stop_mult * atr_last)

    # --- Actionability gating by ATR-distance to entry/trigger ---
    # Prevent stale alerts: we only fire when price is *near* either the pullback band or break trigger.
    prox_atr = 0.45
    # Distance to pullback band (0 if inside the band)
    if direction == "LONG":
        dist_pb_band = 0.0 if (last_price >= pb2 and last_price <= pb1) else min(abs(last_price - pb2), abs(last_price - pb1))
        # Stale breakout if price is far above break trigger and not near pullback.
        stale_breakout = bool(last_price > break_trigger + 0.60 * atr_last and dist_pb_band > prox_atr * atr_last)
    else:
        dist_pb_band = 0.0 if (last_price <= pb2 and last_price >= pb1) else min(abs(last_price - pb2), abs(last_price - pb1))
        stale_breakout = bool(last_price < break_trigger - 0.60 * atr_last and dist_pb_band > prox_atr * atr_last)

    dist_br = abs(last_price - break_trigger)
    near_pullback = bool(dist_pb_band <= prox_atr * atr_last)
    near_break = bool(dist_br <= prox_atr * atr_last)
    actionable = bool((near_pullback or near_break) and not stale_breakout)

    # Choose execution plan based on what made it actionable.
    # - If we're near pullback band: limit-style pullback entry.
    # - If we're near break trigger: breakout-style (trigger + small chase line).
    entry_mode = None
    entry_price = None
    chase_line = None
    if actionable:
        if near_pullback and (not near_break or dist_pb_band <= dist_br):
            entry_mode = "PULLBACK"
            entry_price = float(pullback_entry)
            chase_line = float(break_trigger)
            # stop already computed relative to pullback_entry
        else:
            entry_mode = "BREAKOUT"
            # For breakouts, the practical entry is the trigger itself; if we've already pushed through,
            # we treat it as a small chase entry rather than a deep pullback limit.
            if direction == "LONG":
                entry_price = float(max(break_trigger, last_price))
                chase_line = float(entry_price + 0.10 * atr_last)
                stop = float(min(stop, accept_line - 0.80 * atr_last))
            else:
                entry_price = float(min(break_trigger, last_price))
                chase_line = float(entry_price - 0.10 * atr_last)
                stop = float(max(stop, accept_line + 0.80 * atr_last))

    # --- Targets: structure-first + monotonicity ---
    # TP0 should be a *real* liquidity/structure level (not a tiny tick), and TP ordering
    # must be monotonic (TP0 -> TP1 -> TP2 in the trade direction).
    hold_rng = float(df["high"].tail(6).max() - df["low"].tail(6).min())
    min_step = max(0.60 * float(atr_last), 0.35 * float(hold_rng))

    if direction == "LONG":
        cands = [x for x in [levels.get("prior_high"), levels.get("premarket_high"), swing_hi] if isinstance(x, (float, int))]
        cands = [float(x) for x in cands if float(x) > break_trigger + 0.10 * atr_last]
        tp0 = float(min(cands)) if cands else float(break_trigger + 0.90 * atr_last)
        # ensure tp0 isn't a meaningless "tick" target
        if float(tp0) - float(last_price) < 0.25 * float(atr_last):
            tp0 = float(last_price + 0.80 * atr_last)

        tp1 = float(tp0 + max(min_step, 0.70 * hold_rng))
        tp2 = float(tp1 + max(1.00 * atr_last, 0.90 * hold_rng))
    else:
        cands = [x for x in [levels.get("prior_low"), levels.get("premarket_low"), swing_lo] if isinstance(x, (float, int))]
        cands = [float(x) for x in cands if float(x) < break_trigger - 0.10 * atr_last]
        tp0 = float(max(cands)) if cands else float(break_trigger - 0.90 * atr_last)
        if float(last_price) - float(tp0) < 0.25 * float(atr_last):
            tp0 = float(last_price - 0.80 * atr_last)

        tp1 = float(tp0 - max(min_step, 0.70 * hold_rng))
        tp2 = float(tp1 - max(1.00 * atr_last, 0.90 * hold_rng))

    # Optional runner target (TP3): simple, monotonic extension.
    if direction == "LONG":
        tp3 = float(tp2 + max(1.25 * atr_last, 1.10 * hold_rng))
    else:
        tp3 = float(tp2 - max(1.25 * atr_last, 1.10 * hold_rng))

    # ETA to TP0 (minutes)
    liq_factor = 1.0
    if str(liquidity_phase).upper() in ("AFTERHOURS", "PREMARKET"):
        liq_factor = 1.6
    elif str(liquidity_phase).upper() in ("MIDDAY",):
        liq_factor = 1.25
    elif str(liquidity_phase).upper() in ("OPENING", "POWER"):
        liq_factor = 0.9
    eta_min = None
    try:
        dist = abs(float(tp0) - float(last_price))
        bars = dist / max(1e-6, float(atr_last))
        eta_min = float(bars * float(interval_mins) * liq_factor)
    except Exception:
        eta_min = None

    why = []
    why.append(f"Trend {trend_votes}/3 (ADX {adx_last:.1f})")
    why.append("Impulse: " + ("ORB" if (orb_break_up or orb_break_dn) else ("Pivot" if (pivot_break_up or pivot_break_dn) else "VWAP+Disp")))
    why.append(f"Accept: {accept_src}" + (" + retest" if stage == "CONFIRMED" else ""))
    if vol_ok:
        why.append("Vol: expand→compress")
    if exhausted:
        why.append("Exhaustion guard")
    if not actionable:
        why.append("Not near entry lines yet")
    # compact quality hint
    why.append(f"Q={impulse_quality:.2f}")

    bias = "RIDE_LONG" if direction == "LONG" else "RIDE_SHORT"

    return SignalResult(
        symbol=symbol,
        bias=bias if actionable else "CHOP",
        setup_score=score,
        reason="; ".join(why),
        entry=float(entry_price) if (actionable and entry_price is not None) else None,
        stop=stop if actionable else None,
        target_1r=tp0 if actionable else None,
        target_2r=tp1 if actionable else None,
        last_price=last_price,
        timestamp=last_ts,
        session=session,
        extras={
            "mode": "RIDE",
            "stage": stage if actionable else None,
            "actionable": actionable,
            "accept_line": float(accept_line),
            "accept_src": accept_src,
            "break_trigger": float(break_trigger),
            "pullback_entry": float(pullback_entry),
            "pb1": float(pb1),
            "pb2": float(pb2),
            "tp0": float(tp0),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "tp3": float(tp3),
            "entry_mode": entry_mode,
            "chase_line": float(chase_line) if chase_line is not None else None,
            "eta_tp0_min": eta_min,
            "liquidity_phase": liquidity_phase,
            "trend_votes": trend_votes,
            "adx": adx_last,
            "di_plus": di_p,
            "di_minus": di_m,
            "impulse_quality": impulse_quality,
            "disp_ratio": disp_ratio,
            "vwap_logic": vwap_logic,
            "session_vwap_include_premarket": session_vwap_include_premarket,
        },
    )

# =========================
# MSS / ICT (Strict) alerts
# =========================

def _last_pivot_level(df: pd.DataFrame, piv_bool: pd.Series, col: str, *, before_idx: int) -> Tuple[Optional[float], Optional[int]]:
    """Return the most recent pivot level and its index position strictly before `before_idx`."""
    try:
        idxs = np.where(piv_bool.values)[0]
        idxs = idxs[idxs < before_idx]
        if len(idxs) == 0:
            return None, None
        i = int(idxs[-1])
        return float(df[col].iloc[i]), i
    except Exception:
        return None, None


def _first_touch_after(df: pd.DataFrame, *, start_i: int, zone_low: float, zone_high: float) -> Optional[int]:
    """First index >= start_i where candle overlaps the zone."""
    try:
        h = df["high"].values
        l = df["low"].values
        for i in range(max(0, start_i), len(df)):
            if (l[i] <= zone_high) and (h[i] >= zone_low):
                return i
        return None
    except Exception:
        return None


def compute_mss_signal(
    symbol: str,
    df: pd.DataFrame,
    rsi5: Optional[pd.Series] = None,
    rsi14: Optional[pd.Series] = None,
    macd_hist: Optional[pd.Series] = None,
    *,
    interval: str = "1min",
    # Time / bar guards
    allow_opening: bool = True,
    allow_midday: bool = True,
    allow_power: bool = True,
    allow_premarket: bool = False,
    allow_afterhours: bool = False,
    use_last_closed_only: bool = False,
    bar_closed_guard: bool = True,
    # VWAP config (for context + some POI ranking)
    vwap_logic: str = "session",
    session_vwap_include_premarket: bool = False,
    # Fib/vol knobs
    fib_lookback_bars: int = 240,
    orb_minutes: int = 15,
    liquidity_weighting: float = 0.55,
    target_atr_pct: float | None = None,
) -> SignalResult:
    """Strict MSS/ICT alert family.

    Philosophy:
      - Very selective: only fire when we can explicitly see
        raid -> displacement -> MSS break -> POI retest/accept.
      - Output is actionability-oriented (pullback band + trigger + monotonic targets).

    Returns SignalResult with bias in {MSS_LONG, MSS_SHORT, CHOP}.
    """

    if df is None or len(df) < 80:
        return SignalResult(symbol, "CHOP", 0, "Not enough data", None, None, None, None, None, None, None, {"family": "MSS"})

    # Use last closed bar if requested (prevents half-formed candle artifacts)
    dfx = df.copy()
    if use_last_closed_only and len(dfx) >= 2:
        dfx = dfx.iloc[:-1].copy()

    last_ts = dfx.index[-1]
    session = classify_session(last_ts)
    liquidity_phase = classify_liquidity_phase(last_ts)

    # Respect user time-of-day filters (same pattern as other engines).
    allow = {
        "OPENING": allow_opening,
        "MIDDAY": allow_midday,
        "POWER": allow_power,
        "PREMARKET": allow_premarket,
        "AFTERHOURS": allow_afterhours,
        "CLOSED": allow_afterhours,
    }.get(session, True)
    if not allow:
        return SignalResult(symbol, "CHOP", 0, f"Time filter blocks MSS ({session})", None, None, None, None, None, None, session, {"family": "MSS", "session": session})

    # --- Core series ---
    atr14 = calc_atr(dfx[["high", "low", "close"]], period=14)
    atr_last = float(atr14.iloc[-1]) if len(atr14) else 0.0

    # Vol normalization baseline (optional)
    atr_pct = float(atr_last / float(dfx["close"].iloc[-1])) if float(dfx["close"].iloc[-1]) else 0.0
    atr_score_scale = 1.0
    baseline_atr_pct = None
    if target_atr_pct is not None and isinstance(target_atr_pct, (float, int)) and target_atr_pct > 0:
        baseline_atr_pct = float(target_atr_pct)
        try:
            atr_score_scale = float(np.clip(baseline_atr_pct / max(atr_pct, 1e-9), 0.75, 1.25))
        except Exception:
            atr_score_scale = 1.0

    # --- Pivots: external (structure) + internal (MSS) ---
    ext_l = 6 if interval in ("1min", "5min") else 8
    ext_r = ext_l
    int_l = 2
    int_r = 2

    piv_low_ext = rolling_swing_lows(dfx["low"], left=ext_l, right=ext_r)
    piv_high_ext = rolling_swing_highs(dfx["high"], left=ext_l, right=ext_r)
    piv_low_int = rolling_swing_lows(dfx["low"], left=int_l, right=int_r)
    piv_high_int = rolling_swing_highs(dfx["high"], left=int_l, right=int_r)

    # --- Find most recent raid (liquidity sweep) ---
    raid_search = min(180, len(dfx) - 10)
    raid_i = None
    raid_side = None  # "bull" means swept lows
    raid_level = None

    # scan from near-end backwards for a clean sweep
    lows = dfx["low"].values
    highs = dfx["high"].values
    closes = dfx["close"].values

    for i in range(len(dfx) - 2, max(10, len(dfx) - raid_search), -1):
        # bullish raid: take external pivot low, wick below, close back above pivot (reclaim)
        pl, pl_i = _last_pivot_level(dfx, piv_low_ext, "low", before_idx=i)
        if pl is not None and pl_i is not None:
            if lows[i] < pl and closes[i] > pl:
                # require meaningful sweep size
                if atr_last > 0 and (pl - lows[i]) >= 0.15 * atr_last:
                    raid_i = i
                    raid_side = "bull"
                    raid_level = float(pl)
                    break
        # bearish raid: take external pivot high, wick above, close back below pivot
        ph, ph_i = _last_pivot_level(dfx, piv_high_ext, "high", before_idx=i)
        if ph is not None and ph_i is not None:
            if highs[i] > ph and closes[i] < ph:
                if atr_last > 0 and (highs[i] - ph) >= 0.15 * atr_last:
                    raid_i = i
                    raid_side = "bear"
                    raid_level = float(ph)
                    break

    if raid_i is None or raid_side is None:
        return SignalResult(symbol, "CHOP", 0, "No clean liquidity raid found", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF", "liquidity_phase": liquidity_phase})

    # --- Displacement after raid ---
    tr = (dfx["high"] - dfx["low"]).rolling(20).median().fillna(method="bfill")
    disp_i = None
    disp_ratio = None

    for j in range(raid_i + 1, min(len(dfx), raid_i + 15)):
        rng = float(dfx["high"].iloc[j] - dfx["low"].iloc[j])
        med = float(tr.iloc[j]) if float(tr.iloc[j]) else 0.0
        if med <= 0:
            continue
        body = float(abs(dfx["close"].iloc[j] - dfx["open"].iloc[j]))
        dr = rng / med
        # directionality
        bull_dir = dfx["close"].iloc[j] > dfx["open"].iloc[j]
        bear_dir = dfx["close"].iloc[j] < dfx["open"].iloc[j]
        if raid_side == "bull" and bull_dir and dr >= 1.35 and (body / max(rng, 1e-9)) >= 0.55:
            disp_i = j
            disp_ratio = dr
            break
        if raid_side == "bear" and bear_dir and dr >= 1.35 and (body / max(rng, 1e-9)) >= 0.55:
            disp_i = j
            disp_ratio = dr
            break

    if disp_i is None:
        return SignalResult(symbol, "CHOP", 0, "Raid found but no displacement", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF", "liquidity_phase": liquidity_phase, "raid_i": raid_i})

    # --- MSS break: break of internal pivot in displacement direction ---
    if raid_side == "bull":
        # internal pivot high between raid and displacement
        mss_level, mss_piv_i = _last_pivot_level(dfx, piv_high_int, "high", before_idx=disp_i)
        if mss_level is None:
            return SignalResult(symbol, "CHOP", 0, "No internal pivot for MSS", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF"})
        mss_break_i = None
        for k in range(disp_i, min(len(dfx), disp_i + 20)):
            if float(dfx["close"].iloc[k]) > float(mss_level):
                mss_break_i = k
                break
        if mss_break_i is None:
            return SignalResult(symbol, "CHOP", 0, "No MSS break yet", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF", "mss_level": float(mss_level)})
        bias = "MSS_LONG"
    else:
        mss_level, mss_piv_i = _last_pivot_level(dfx, piv_low_int, "low", before_idx=disp_i)
        if mss_level is None:
            return SignalResult(symbol, "CHOP", 0, "No internal pivot for MSS", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF"})
        mss_break_i = None
        for k in range(disp_i, min(len(dfx), disp_i + 20)):
            if float(dfx["close"].iloc[k]) < float(mss_level):
                mss_break_i = k
                break
        if mss_break_i is None:
            return SignalResult(symbol, "CHOP", 0, "No MSS break yet", None, None, None, None, None, None, session, {"family": "MSS", "stage": "OFF", "mss_level": float(mss_level)})
        bias = "MSS_SHORT"

    # --- POI selection (order block / FVG / breaker) from raid->break window ---
    window_df = dfx.iloc[max(0, raid_i - 5): mss_break_i + 1].copy()

    poi_low = None
    poi_high = None
    poi_src = None

    # Order block
    try:
        ob = find_order_block(window_df, atr14.loc[window_df.index], side=("bull" if raid_side == "bull" else "bear"))
        if ob and isinstance(ob, dict):
            poi_low = float(ob.get("low"))
            poi_high = float(ob.get("high"))
            poi_src = "OB"
    except Exception:
        pass

    # FVG (prefer if present and tighter)
    try:
        fvg = detect_fvg(window_df)
        if fvg and isinstance(fvg, dict):
            fl = float(fvg.get("low"))
            fh = float(fvg.get("high"))
            if (poi_low is None) or (fh - fl) < (poi_high - poi_low):
                poi_low, poi_high, poi_src = fl, fh, "FVG"
    except Exception:
        pass

    # Breaker fallback
    if poi_low is None or poi_high is None:
        try:
            br = find_breaker_block(window_df, atr14.loc[window_df.index], side=("bull" if raid_side == "bull" else "bear"))
            if br and isinstance(br, dict):
                poi_low = float(br.get("low"))
                poi_high = float(br.get("high"))
                poi_src = "BREAKER"
        except Exception:
            pass

    if poi_low is None or poi_high is None:
        # last resort: midpoint of displacement candle
        poi_low = float(min(window_df["open"].iloc[-1], window_df["close"].iloc[-1]))
        poi_high = float(max(window_df["open"].iloc[-1], window_df["close"].iloc[-1]))
        poi_src = "DISP_BODY"

    poi_low, poi_high = float(min(poi_low, poi_high)), float(max(poi_low, poi_high))
    poi_mid = 0.5 * (poi_low + poi_high)

    # --- Retest + accept (CONFIRMED) ---
    touch_i = _first_touch_after(dfx, start_i=mss_break_i, zone_low=poi_low, zone_high=poi_high)
    retest_ok = touch_i is not None
    accept_ok = False
    if retest_ok:
        after = min(len(dfx) - 1, int(touch_i) + 3)
        if bias == "MSS_LONG":
            accept_ok = float(dfx["close"].iloc[after]) >= poi_mid
        else:
            accept_ok = float(dfx["close"].iloc[after]) <= poi_mid

    # --- Actionability: ATR-distance to POI band or trigger ---
    last_price = float(dfx["close"].iloc[-1])
    atr = max(atr_last, 1e-9)

    # trigger is the MSS break level (for long, above; for short, below)
    break_trigger = float(mss_level)
    dist_to_poi = 0.0
    if last_price < poi_low:
        dist_to_poi = (poi_low - last_price)
    elif last_price > poi_high:
        dist_to_poi = (last_price - poi_high)

    dist_to_trigger = abs(last_price - break_trigger)
    actionable_gate = (min(dist_to_poi, dist_to_trigger) <= 0.75 * atr)

    stage = None
    if actionable_gate:
        stage = "PRE"
        if retest_ok and accept_ok:
            stage = "CONFIRMED"

    # --- Entries / stops (strict + practical) ---
    pullback_entry = float(poi_mid)
    pb1 = float(poi_high)
    pb2 = float(poi_low)

    raid_extreme = float(dfx["low"].iloc[raid_i]) if bias == "MSS_LONG" else float(dfx["high"].iloc[raid_i])
    strict_stop = raid_extreme - 0.05 * atr if bias == "MSS_LONG" else raid_extreme + 0.05 * atr
    practical_stop = (poi_low - 0.10 * atr) if bias == "MSS_LONG" else (poi_high + 0.10 * atr)
    stop = float(practical_stop)

    # --- Targets (monotonic, structure-first) ---
    tp0 = None
    tp1 = None
    tp2 = None

    if bias == "MSS_LONG":
        # nearest internal pivot high above last
        candidates = []
        for i in np.where(piv_high_int.values)[0]:
            if i < len(dfx) and float(dfx["high"].iloc[i]) > last_price:
                candidates.append(float(dfx["high"].iloc[i]))
        candidates = sorted(set(candidates))
        tp0 = candidates[0] if candidates else float(last_price + 1.0 * atr)

        # next external pivot high (bigger pool)
        ext_cand = []
        for i in np.where(piv_high_ext.values)[0]:
            if i < len(dfx) and float(dfx["high"].iloc[i]) > float(tp0):
                ext_cand.append(float(dfx["high"].iloc[i]))
        ext_cand = sorted(set(ext_cand))
        tp1 = ext_cand[0] if ext_cand else float(tp0 + 1.0 * atr)

        # measured move from displacement
        disp_range = float(dfx["high"].iloc[disp_i] - dfx["low"].iloc[disp_i])
        tp2 = float(max(tp1, pullback_entry + max(disp_range, 1.2 * atr)))

    else:
        candidates = []
        for i in np.where(piv_low_int.values)[0]:
            if i < len(dfx) and float(dfx["low"].iloc[i]) < last_price:
                candidates.append(float(dfx["low"].iloc[i]))
        candidates = sorted(set(candidates), reverse=True)
        tp0 = candidates[0] if candidates else float(last_price - 1.0 * atr)

        ext_cand = []
        for i in np.where(piv_low_ext.values)[0]:
            if i < len(dfx) and float(dfx["low"].iloc[i]) < float(tp0):
                ext_cand.append(float(dfx["low"].iloc[i]))
        ext_cand = sorted(set(ext_cand), reverse=True)
        tp1 = ext_cand[0] if ext_cand else float(tp0 - 1.0 * atr)

        disp_range = float(dfx["high"].iloc[disp_i] - dfx["low"].iloc[disp_i])
        tp2 = float(min(tp1, pullback_entry - max(disp_range, 1.2 * atr)))

    # ensure monotonic ordering
    if bias == "MSS_LONG":
        tp0 = float(max(tp0, last_price))
        tp1 = float(max(tp1, tp0))
        tp2 = float(max(tp2, tp1))
    else:
        tp0 = float(min(tp0, last_price))
        tp1 = float(min(tp1, tp0))
        tp2 = float(min(tp2, tp1))

    # --- Score (quality-driven) ---
    score = 0.0
    why_bits = []

    # Raid quality (size)
    try:
        if raid_side == "bull":
            raid_size = float(raid_level - lows[raid_i])
        else:
            raid_size = float(highs[raid_i] - raid_level)
        raid_q = float(np.clip(raid_size / max(atr, 1e-9), 0.0, 2.0))
    except Exception:
        raid_q = 0.0

    score += 20.0 * min(1.0, raid_q)
    why_bits.append("Raid+reclaim")

    # Displacement quality
    dq = float(np.clip((disp_ratio or 0.0) / 2.0, 0.0, 1.0))
    score += 25.0 * dq
    why_bits.append("Displacement")

    # MSS break
    score += 20.0
    why_bits.append("MSS break")

    # POI quality
    if poi_src in ("FVG", "OB", "BREAKER"):
        score += 10.0
        why_bits.append(f"POI={poi_src}")

    # Retest/accept
    if retest_ok:
        score += 10.0
        why_bits.append("Retest")
    if accept_ok:
        score += 10.0
        why_bits.append("Accept")

    # RSI exhaustion guard (prevents buying top / selling bottom)
    if rsi5 is not None and rsi14 is not None:
        try:
            r5 = float(rsi5.iloc[-1])
            r14 = float(rsi14.iloc[-1])
            if bias == "MSS_LONG" and (r5 > 88 and r14 > 72):
                score -= 12.0
                why_bits.append("RSI exhausted")
            if bias == "MSS_SHORT" and (r5 < 12 and r14 < 28):
                score -= 12.0
                why_bits.append("RSI exhausted")
            else:
                score += 5.0
        except Exception:
            pass

    score *= float(atr_score_scale)
    score_i = _cap_score(score)

    actionable = stage in ("PRE", "CONFIRMED") and bias in ("MSS_LONG", "MSS_SHORT")

    reason = " ".join(why_bits)
    if stage is None:
        reason = reason + "; Too far from POI/trigger (ATR gating)"

    # ETA TP0 using same concept as other engines
    eta_min = None
    try:
        if atr_last > 0:
            dist = abs(float(tp0) - last_price)
            # rough minutes per ATR based on liquidity phase
            pace = 7.0 if liquidity_phase == "RTH" else 11.0
            eta_min = float(max(1.0, (dist / atr_last) * pace))
    except Exception:
        eta_min = None

    return SignalResult(
        symbol=symbol,
        # Keep the MSS family bias namespace intact so app-side routing/alerting
        # can key off (MSS_LONG/MSS_SHORT) without ambiguity.
        bias=bias if actionable else "CHOP",
        setup_score=score_i,
        reason=(f"MSS {stage or 'OFF'} — {reason}"),
        last_price=last_price,
        entry=pullback_entry if actionable else None,
        stop=stop if actionable else None,
        target_1r=float(tp0) if actionable else None,
        target_2r=float(tp1) if actionable else None,
        timestamp=last_ts,
        session=session,
        extras={
            "family": "MSS",
            "stage": stage,
            "actionable": actionable,
            "poi_src": poi_src,
            "pb1": pb1,
            "pb2": pb2,
            "pullback_entry": pullback_entry,
            "break_trigger": break_trigger,
            "strict_stop": float(strict_stop),
            "tp0": float(tp0),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "eta_tp0_min": eta_min,
            "liquidity_phase": liquidity_phase,
            "raid_i": int(raid_i),
            "disp_i": int(disp_i),
            "mss_level": float(mss_level),
            "disp_ratio": float(disp_ratio) if disp_ratio is not None else None,
            "atr_pct": atr_pct,
            "baseline_atr_pct": baseline_atr_pct,
            "atr_score_scale": atr_score_scale,
            "vwap_logic": vwap_logic,
            "session_vwap_include_premarket": session_vwap_include_premarket,
        },
    )
