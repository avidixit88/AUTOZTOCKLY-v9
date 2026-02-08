from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

from indicators import atr as calc_atr
from indicators import ema as calc_ema
from indicators import adx as calc_adx
from indicators import session_vwap as calc_session_vwap
from indicators import detect_fvg
from indicators import rolling_swing_highs, rolling_swing_lows
from sessions import classify_session


@dataclass
class HeavenlyConfig:
    enable: bool = True
    # allowed sessions (passed from app toggles)
    allow_opening: bool = True
    allow_midday: bool = True
    allow_power: bool = True
    allow_premarket: bool = False
    allow_afterhours: bool = False
    # vwap session config (passed from app)
    session_vwap_include_premarket: bool = False
    session_vwap_include_afterhours: bool = False
    # numeric knobs
    one_min_ttl_seconds: int = 90
    zone_tol_atr: float = 0.35
    zone_max_width_atr: float = 0.45
    price_to_zone_proximity_atr: float = 0.75
    min_evs: float = 2.0
    min_evs_tp3: float = 2.5
    max_risk_atr: float = 0.8


def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        if isinstance(x, (np.floating, np.integer)):
            x = float(x)
        if isinstance(x, (int, float)) and np.isfinite(x):
            return float(x)
    except Exception:
        return None
    return None


def _ensure_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize expected OHLCV columns and sort ascending by time."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])  # empty
    out = df.copy()
    out = out.sort_index()
    # Ensure lowercase columns
    cols = {c: c.lower() for c in out.columns}
    out = out.rename(columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        if c not in out.columns:
            raise ValueError(f"Missing column {c} in OHLCV")
    return out


def _last_closed_bar(df: pd.DataFrame, interval_seconds: int, now_ts: float) -> Tuple[pd.Series, pd.Timestamp]:
    """Return (bar, ts) for the most recently *closed* bar.

    AlphaVantage sometimes includes a still-forming bar at the end.
    We treat the last bar as "forming" if it's too recent vs interval.
    """
    if df is None or df.empty:
        raise ValueError("empty df")
    ts_last: pd.Timestamp = df.index[-1]
    # tz-aware timestamps expose .timestamp(); naive assumed UTC-like
    try:
        age = now_ts - ts_last.timestamp()
    except Exception:
        age = float("inf")
    # If last bar is younger than 80% of interval, step back one.
    if age < (0.8 * interval_seconds) and len(df) >= 2:
        ts = df.index[-2]
        return df.iloc[-2], ts
    return df.iloc[-1], ts_last


def _volume_profile_levels(df: pd.DataFrame, bins: int = 24) -> Dict[str, Optional[float]]:
    """Compute simple volume-by-price proxy levels (HVN/LVN centers).

    Uses typical price * volume accumulation into equal-width bins.
    Returns hvn_center, lvn_center.
    """
    if df is None or df.empty or len(df) < 20:
        return {"hvn": None, "lvn": None}
    lo = float(df["low"].min())
    hi = float(df["high"].max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return {"hvn": None, "lvn": None}
    edges = np.linspace(lo, hi, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].astype(float).clip(lower=0)
    idx = np.clip(np.digitize(tp.values, edges) - 1, 0, bins - 1)
    agg = np.zeros(bins, dtype=float)
    for i, v in zip(idx, vol.values):
        agg[int(i)] += float(v)
    if agg.sum() <= 0:
        return {"hvn": None, "lvn": None}
    hvn = float(centers[int(np.argmax(agg))])
    lvn = float(centers[int(np.argmin(agg))])
    return {"hvn": hvn, "lvn": lvn}


def compute_30m_suppression(df_30m: pd.DataFrame) -> Dict[str, Any]:
    """Return suppression grade and a compact diagnostic."""
    df = _ensure_ohlcv(df_30m)
    if len(df) < 80:
        return {
            "grade": "weak",
            "range_ratio": None,
            "adx": None,
            "ema_slope": None,
            "box_low": None,
            "box_high": None,
            "why": "Not enough 30m bars",
        }

    tail = df.tail(40).copy()
    atr30 = calc_atr(df, 14)
    atr_last = _safe_float(atr30.iloc[-1])
    if not atr_last or atr_last <= 0:
        return {
            "grade": "weak",
            "range_ratio": None,
            "adx": None,
            "ema_slope": None,
            "box_low": None,
            "box_high": None,
            "why": "ATR unavailable",
        }

    box_low = float(tail["low"].min())
    box_high = float(tail["high"].max())
    range_width = float(box_high - box_low)
    range_ratio = range_width / float(atr_last)

    adx, _, _ = calc_adx(df, 14)
    adx_last = _safe_float(adx.iloc[-1])

    ema20 = calc_ema(df["close"].astype(float), 20)
    # slope over last 6 bars
    ema_slope = _safe_float((ema20.iloc[-1] - ema20.iloc[-7]) / 6.0) if len(ema20) >= 8 else None

    grade = "weak"
    if range_ratio <= 2.0:
        grade = "moderate"
    if range_ratio <= 1.5:
        grade = "strong"

    why_bits = [f"range/ATR={range_ratio:.2f}"]
    if adx_last is not None:
        why_bits.append(f"ADX={adx_last:.1f}")
    if ema_slope is not None:
        why_bits.append(f"EMA20_slope={ema_slope:.4f}")
    return {
        "grade": grade,
        "range_ratio": float(range_ratio),
        "adx": adx_last,
        "ema_slope": ema_slope,
        "box_low": box_low,
        "box_high": box_high,
        "why": ", ".join(why_bits),
    }


def _find_pivots(df: pd.DataFrame, left: int = 2, right: int = 2) -> Dict[str, List[float]]:
    df = _ensure_ohlcv(df)
    if len(df) < (left + right + 5):
        return {"highs": [], "lows": []}
    is_high = rolling_swing_highs(df["high"], left=left, right=right)
    is_low = rolling_swing_lows(df["low"], left=left, right=right)
    highs = [float(v) for v in df.loc[is_high, "high"].tail(20).values]
    lows = [float(v) for v in df.loc[is_low, "low"].tail(20).values]
    return {"highs": highs, "lows": lows}


def compute_5m_tsz(df_5m: pd.DataFrame, df_30m: pd.DataFrame, cfg: HeavenlyConfig, *, now_ts: float) -> Dict[str, Any]:
    """Compute triangulated supply/demand zone (TSZ) from multi-constraint clustering."""
    d5 = _ensure_ohlcv(df_5m)
    d30 = _ensure_ohlcv(df_30m)
    if len(d5) < 80:
        return {"exists": False, "why": "Not enough 5m bars"}

    atr5 = calc_atr(d5, 14)
    atr5_last = _safe_float(atr5.iloc[-1])
    if not atr5_last or atr5_last <= 0:
        return {"exists": False, "why": "ATR(5m) unavailable"}

    last_bar, last_ts = _last_closed_bar(d5, 300, now_ts)
    price = float(last_bar["close"])

    # Constraint 1: session vwap
    try:
        vwap_s = calc_session_vwap(
            d5,
            include_premarket=bool(cfg.session_vwap_include_premarket),
            include_afterhours=bool(cfg.session_vwap_include_afterhours),
        )
        vwap_last = _safe_float(vwap_s.dropna().iloc[-1]) if not vwap_s.dropna().empty else None
    except Exception:
        vwap_last = None

    levels: List[Tuple[float, str]] = []
    if vwap_last is not None:
        levels.append((float(vwap_last), "SVWAP"))

    # Constraint 2: last major pivots (5m)
    piv5 = _find_pivots(d5.tail(240), left=3, right=3)
    if piv5["highs"]:
        levels.append((float(piv5["highs"][-1]), "5m_pivot_high"))
    if piv5["lows"]:
        levels.append((float(piv5["lows"][-1]), "5m_pivot_low"))

    # Constraint 3: volume profile proxy (5m)
    vp5 = _volume_profile_levels(d5.tail(240), bins=24)
    if vp5.get("hvn") is not None:
        levels.append((float(vp5["hvn"]), "5m_HVN"))
    if vp5.get("lvn") is not None:
        levels.append((float(vp5["lvn"]), "5m_LVN"))

    # Constraint 4: projected 30m suppression box edges
    sup = compute_30m_suppression(d30)
    if sup.get("box_low") is not None and sup.get("box_high") is not None:
        levels.append((float(sup["box_low"]), "30m_box_low"))
        levels.append((float(sup["box_high"]), "30m_box_high"))

    # Constraint 5: 30m FVG bounds (if present)
    bull, bear = detect_fvg(d30.tail(120))
    if bull is not None:
        levels.append((float(bull[0]), "30m_bull_fvg_low"))
        levels.append((float(bull[1]), "30m_bull_fvg_high"))
    if bear is not None:
        levels.append((float(bear[0]), "30m_bear_fvg_low"))
        levels.append((float(bear[1]), "30m_bear_fvg_high"))

    # Cluster within tolerance
    tol = float(cfg.zone_tol_atr) * float(atr5_last)
    if len(levels) < 3 or tol <= 0:
        return {"exists": False, "why": "Insufficient constraints"}

    best_cluster: List[Tuple[float, str]] = []
    for base, _lbl in levels:
        cluster = [(lv, lb) for lv, lb in levels if abs(lv - base) <= tol]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    if len(best_cluster) < 3:
        return {"exists": False, "why": "No 3+ constraint cluster"}

    zone_low = float(min(lv for lv, _ in best_cluster))
    zone_high = float(max(lv for lv, _ in best_cluster))
    width = zone_high - zone_low
    if width > float(cfg.zone_max_width_atr) * float(atr5_last):
        return {"exists": False, "why": "Cluster too wide"}

    mid = (zone_low + zone_high) / 2.0
    dist_atr = 0.0
    if price < zone_low:
        dist_atr = (zone_low - price) / float(atr5_last)
    elif price > zone_high:
        dist_atr = (price - zone_high) / float(atr5_last)
    else:
        dist_atr = 0.0

    used = sorted({lb for _lv, lb in best_cluster})
    return {
        "exists": True,
        "low": zone_low,
        "high": zone_high,
        "mid": float(mid),
        "width_atr": float(width / float(atr5_last)),
        "constraints": used,
        "confluence_count": int(len(used)),
        "price": float(price),
        "distance_to_zone_atr": float(dist_atr),
        "last_closed_ts": last_ts,
    }


def compute_evs(df_30m: pd.DataFrame, sup: Dict[str, Any], *, now_ts: float) -> Dict[str, Any]:
    """Expected-Value Space: distance (in ATR) to next obstacle in candidate direction."""
    d30 = _ensure_ohlcv(df_30m)
    if len(d30) < 80:
        return {"evs": 0.0, "direction": "NEUTRAL", "obstacle": None, "why": "Not enough 30m bars"}

    atr30 = calc_atr(d30, 14)
    atr_last = _safe_float(atr30.iloc[-1])
    if not atr_last or atr_last <= 0:
        return {"evs": 0.0, "direction": "NEUTRAL", "obstacle": None, "why": "ATR unavailable"}

    last_bar, _ts = _last_closed_bar(d30, 1800, now_ts)
    price = float(last_bar["close"])

    ema20 = calc_ema(d30["close"].astype(float), 20)
    slope = (ema20.iloc[-1] - ema20.iloc[-7]) / 6.0 if len(ema20) >= 8 else 0.0

    # Candidate direction: follow slope if meaningful; else use position in suppression box.
    direction = "NEUTRAL"
    if abs(float(slope)) > (0.02 * float(atr_last)):
        direction = "LONG" if slope > 0 else "SHORT"
    else:
        lo = sup.get("box_low")
        hi = sup.get("box_high")
        if lo is not None and hi is not None and hi > lo:
            mid = (float(lo) + float(hi)) / 2.0
            direction = "LONG" if price <= mid else "SHORT"
        else:
            direction = "LONG"  # default bias

    piv = _find_pivots(d30.tail(240), left=2, right=2)
    obstacle_price: Optional[float] = None
    obstacle_type: Optional[str] = None
    if direction == "LONG":
        above = [p for p in piv["highs"] if p > price]
        if above:
            obstacle_price = float(min(above))
            obstacle_type = "30m_pivot_high"
    elif direction == "SHORT":
        below = [p for p in piv["lows"] if p < price]
        if below:
            obstacle_price = float(max(below))
            obstacle_type = "30m_pivot_low"

    # Fallback: use suppression box edge
    if obstacle_price is None:
        if direction == "LONG" and sup.get("box_high") is not None:
            obstacle_price = float(sup["box_high"])
            obstacle_type = "30m_box_high"
        if direction == "SHORT" and sup.get("box_low") is not None:
            obstacle_price = float(sup["box_low"])
            obstacle_type = "30m_box_low"

    if obstacle_price is None:
        return {"evs": 0.0, "direction": direction, "obstacle": None, "why": "No obstacle"}

    dist = abs(float(obstacle_price) - price)
    evs = dist / float(atr_last)
    return {
        "evs": float(evs),
        "direction": direction,
        "obstacle_price": float(obstacle_price),
        "obstacle_type": obstacle_type,
        "why": f"dist={dist:.4f} ({evs:.2f} ATR)",
    }


def should_fetch_1m(sup: Dict[str, Any], tsz: Dict[str, Any], evs: Dict[str, Any], cfg: HeavenlyConfig) -> Tuple[bool, str]:
    if sup.get("grade") not in ("moderate", "strong"):
        return False, "suppression weak"
    if not tsz.get("exists"):
        return False, "no TSZ"
    if float(evs.get("evs") or 0.0) < float(cfg.min_evs):
        return False, "EVS too small"
    if float(tsz.get("distance_to_zone_atr") or 9e9) > float(cfg.price_to_zone_proximity_atr):
        return False, "too far from zone"
    return True, "gate passed"


def compute_1m_intent(df_1m: pd.DataFrame) -> Dict[str, Any]:
    d1 = _ensure_ohlcv(df_1m)
    if len(d1) < 60:
        return {"intent_score": 0.0, "intent_label": "neutral", "why": "Not enough 1m bars"}

    atr1 = calc_atr(d1, 14)
    a_now = _safe_float(atr1.iloc[-1]) or 0.0
    a_prev = _safe_float(atr1.iloc[-30:-15].median()) or 0.0
    if a_prev <= 0:
        a_prev = a_now if a_now > 0 else 1.0
    atr_ratio = a_now / a_prev

    vol = d1["volume"].astype(float)
    v_now = float(vol.iloc[-3:].sum())
    v_prev = float(vol.iloc[-30:-3].mean() * 3.0) if len(vol) >= 40 else float(vol.mean() * 3.0)
    if v_prev <= 0:
        v_prev = max(v_now, 1.0)
    vol_ratio = v_now / v_prev

    # 2-bar directional strength
    c = d1["close"].astype(float)
    dir_strength = float(abs(c.iloc[-1] - c.iloc[-3]) / (a_now + 1e-9))

    score = 0.0
    score += min(40.0, 20.0 * max(0.0, atr_ratio - 1.0))
    score += min(40.0, 20.0 * max(0.0, vol_ratio - 1.0))
    score += min(20.0, 10.0 * max(0.0, dir_strength - 0.5))
    score = float(max(0.0, min(100.0, score)))

    label = "neutral"
    if score >= 70:
        label = "hot"
    elif score >= 40:
        label = "waking"
    return {
        "intent_score": float(score),
        "intent_label": label,
        "atr_ratio": float(atr_ratio),
        "vol_ratio": float(vol_ratio),
        "why": f"atr×{atr_ratio:.2f}, vol×{vol_ratio:.2f}",
    }


def detect_5m_entry_trigger(df_5m: pd.DataFrame, tsz: Dict[str, Any], direction: str, *, now_ts: float) -> Dict[str, Any]:
    d5 = _ensure_ohlcv(df_5m)
    if len(d5) < 50 or not tsz.get("exists"):
        return {"triggered": False, "why": "no TSZ"}
    bar, ts = _last_closed_bar(d5, 300, now_ts)
    prev = d5.iloc[-2] if len(d5) >= 2 else bar

    atr5 = calc_atr(d5, 14)
    atr_last = _safe_float(atr5.iloc[-1]) or 0.0
    if atr_last <= 0:
        return {"triggered": False, "why": "ATR missing"}

    rng = float(bar["high"] - bar["low"])
    close = float(bar["close"])
    open_ = float(bar["open"])
    high = float(bar["high"])
    low = float(bar["low"])

    vol = d5["volume"].astype(float)
    v_ma = float(vol.tail(25).mean()) if len(vol) >= 25 else float(vol.mean())
    v_now = float(bar["volume"])
    vol_ok = (v_ma > 0) and (v_now >= 1.5 * v_ma)

    # Trigger A: expansion break
    exp_ok = rng >= 1.25 * atr_last
    if direction == "LONG":
        close_pos = (high - close) <= (0.25 * rng + 1e-9)
    elif direction == "SHORT":
        close_pos = (close - low) <= (0.25 * rng + 1e-9)
    else:
        close_pos = False

    if exp_ok and close_pos and vol_ok:
        return {
            "triggered": True,
            "type": "expansion_break",
            "bar_ts": ts,
            "why": f"range={rng/atr_last:.2f}ATR, vol×{(v_now/(v_ma+1e-9)):.2f}",
        }

    # Trigger B: TSZ reclaim / break
    zl = float(tsz["low"])
    zh = float(tsz["high"])
    prev_close = float(prev["close"])
    prev_in = (prev_close >= zl) and (prev_close <= zh)
    now_out_long = close > zh
    now_out_short = close < zl
    if prev_in and direction == "LONG" and now_out_long:
        return {"triggered": True, "type": "tsz_reclaim", "bar_ts": ts, "why": "reclaimed above TSZ"}
    if prev_in and direction == "SHORT" and now_out_short:
        return {"triggered": True, "type": "tsz_reclaim", "bar_ts": ts, "why": "reclaimed below TSZ"}

    return {"triggered": False, "bar_ts": ts, "why": "no trigger"}


def compute_stops_targets(entry: float, direction: str, tsz: Dict[str, Any], evs: Dict[str, Any], df_5m: pd.DataFrame, df_30m: pd.DataFrame, cfg: HeavenlyConfig) -> Dict[str, Any]:
    d5 = _ensure_ohlcv(df_5m)
    atr5 = calc_atr(d5, 14)
    atr5_last = _safe_float(atr5.iloc[-1]) or 0.0
    if atr5_last <= 0:
        return {"valid": False, "why": "ATR5 missing"}

    zl = float(tsz["low"])
    zh = float(tsz["high"])
    zone_w = zh - zl
    buf = max(0.15 * atr5_last, 0.15 * zone_w)

    if direction == "LONG":
        stop = zl - buf
        risk = entry - stop
        tp1 = entry + 1.25 * risk
    else:
        stop = zh + buf
        risk = stop - entry
        tp1 = entry - 1.25 * risk

    risk_atr = risk / atr5_last
    if risk <= 0 or risk_atr > float(cfg.max_risk_atr):
        return {"valid": False, "stop": stop, "risk_atr": float(risk_atr), "why": "risk too large"}

    # TP2: obstacle if >=2R else 2R
    ob = evs.get("obstacle_price")
    tp2 = None
    if ob is not None:
        if abs(float(ob) - entry) >= 2.0 * risk:
            tp2 = float(ob)
    if tp2 is None:
        tp2 = entry + (2.0 * risk if direction == "LONG" else -2.0 * risk)

    # TP3: only if EVS supports
    tp3 = None
    if float(evs.get("evs") or 0.0) >= float(cfg.min_evs_tp3):
        d30 = _ensure_ohlcv(df_30m)
        atr30 = calc_atr(d30, 14)
        atr30_last = _safe_float(atr30.iloc[-1]) or atr5_last
        stretch = float(evs.get("evs") or 0.0) * float(atr30_last)
        tp3 = entry + (stretch if direction == "LONG" else -stretch)

    return {
        "valid": True,
        "stop": float(stop),
        "risk_atr": float(risk_atr),
        "tp1": float(tp1),
        "tp2": float(tp2),
        "tp3": _safe_float(tp3),
        "why": f"risk={risk_atr:.2f}ATR",
    }


def compute_heavenly_signal(
    symbol: str,
    *,
    df_5m: pd.DataFrame,
    df_30m: pd.DataFrame,
    df_1m: Optional[pd.DataFrame],
    cfg: HeavenlyConfig,
    now_ts: float,
) -> Dict[str, Any]:
    """Compute HEAVENLY engine output.

    Returns a dict ready for UI display and email payload.
    """
    d5 = _ensure_ohlcv(df_5m)
    d30 = _ensure_ohlcv(df_30m)
    if d5.empty or d30.empty:
        return {
            "symbol": symbol,
            "family": "HEAVENLY",
            "stage": "OFF",
            "bias": "NEUTRAL",
            "score": 0,
            "why": "Missing data",
        }

    last5, last5_ts = _last_closed_bar(d5, 300, now_ts)
    session = classify_session(
        last5_ts,
        allow_opening=cfg.allow_opening,
        allow_midday=cfg.allow_midday,
        allow_power=cfg.allow_power,
        allow_premarket=cfg.allow_premarket,
        allow_afterhours=cfg.allow_afterhours,
    )
    if session == "OFF":
        return {
            "symbol": symbol,
            "family": "HEAVENLY",
            "stage": "OFF",
            "bias": "NEUTRAL",
            "score": 0,
            "session": "OFF",
            "last": float(last5["close"]),
            "as_of": last5_ts.isoformat(),
            "why": "Outside allowed session",
            "extras": {"session": "OFF"},
        }

    sup = compute_30m_suppression(d30)
    tsz = compute_5m_tsz(d5, d30, cfg, now_ts=now_ts)
    evs = compute_evs(d30, sup, now_ts=now_ts)
    direction = evs.get("direction") if evs else "NEUTRAL"
    if direction not in ("LONG", "SHORT"):
        direction = "LONG"
    last_price = float(last5["close"])

    # Stage gating
    if sup.get("grade") not in ("moderate", "strong") or not tsz.get("exists") or float(evs.get("evs") or 0.0) < float(cfg.min_evs):
        why = []
        if sup.get("grade") not in ("moderate", "strong"):
            why.append(f"Suppression {sup.get('grade')}")
        if not tsz.get("exists"):
            why.append("No TSZ")
        if float(evs.get("evs") or 0.0) < float(cfg.min_evs):
            why.append(f"EVS<{cfg.min_evs}")
        return {
            "symbol": symbol,
            "family": "HEAVENLY",
            "stage": "OFF",
            "bias": direction,
            "score": 0,
            "session": session,
            "last": last_price,
            "as_of": last5_ts.isoformat(),
            "why": " - ".join(why) if why else "Not ready",
            "extras": {
                "family": "HEAVENLY",
                "suppression": sup,
                "evs": evs,
            },
        }

    dist_atr = float(tsz.get("distance_to_zone_atr") or 9e9)
    stage = "WATCH"
    if dist_atr <= float(cfg.price_to_zone_proximity_atr):
        stage = "SETUP"

    trig = detect_5m_entry_trigger(d5, tsz, direction, now_ts=now_ts)
    entry = float(tsz.get("mid") or last_price)
    stops_targets: Dict[str, Any] = {"valid": False}
    if stage == "SETUP" and trig.get("triggered"):
        # Use entry at TSZ mid for limit
        stops_targets = compute_stops_targets(entry, direction, tsz, evs, d5, d30, cfg)
        if stops_targets.get("valid"):
            stage = "ENTRY"

    # Optional 1m intent
    intent: Dict[str, Any] = {}
    if df_1m is not None and isinstance(df_1m, pd.DataFrame) and not df_1m.empty:
        intent = compute_1m_intent(df_1m)

    # Score: compact 0-100
    score = 0.0
    if sup.get("grade") == "moderate":
        score += 20
    if sup.get("grade") == "strong":
        score += 30
    conf = int(tsz.get("confluence_count") or 0)
    score += min(40.0, 10.0 * conf)
    score += min(30.0, 10.0 * float(evs.get("evs") or 0.0) / 1.0)  # 1 ATR => +10
    if intent:
        lbl = intent.get("intent_label")
        if lbl == "waking":
            score += 5
        if lbl == "hot":
            score += 10
    if stage == "ENTRY":
        score += 20
    score = float(min(100.0, max(0.0, score)))

    why_parts = []
    why_parts.append(f"{stage} - suppression {sup.get('grade')} ({sup.get('why')})")
    why_parts.append(f"TSZ {tsz.get('low'):.4f}–{tsz.get('high'):.4f} ({tsz.get('confluence_count')} confluences)")
    why_parts.append(f"EVS {float(evs.get('evs') or 0.0):.2f} ATR to {evs.get('obstacle_type')}")
    if stage in ("WATCH", "SETUP"):
        why_parts.append("awaiting 5m expansion / reclaim trigger")
    if stage == "ENTRY":
        why_parts.append(f"trigger={trig.get('type')} ({trig.get('why')})")
    if intent:
        why_parts.append(f"1m intent={intent.get('intent_label')} ({intent.get('why')})")

    extras = {
        "family": "HEAVENLY",
        "stage": stage,
        "session": session,
        "suppression_grade": sup.get("grade"),
        "suppression_ratio": sup.get("range_ratio"),
        "tsz_low": tsz.get("low"),
        "tsz_high": tsz.get("high"),
        "tsz_mid": tsz.get("mid"),
        "tsz_width_atr": tsz.get("width_atr"),
        "tsz_constraints": tsz.get("constraints"),
        "evs": evs.get("evs"),
        "evs_obstacle": evs.get("obstacle_type"),
        "evs_obstacle_price": evs.get("obstacle_price"),
        "distance_to_zone_atr": tsz.get("distance_to_zone_atr"),
        "trigger_type": trig.get("type"),
        "trigger_bar_ts": trig.get("bar_ts").isoformat() if trig.get("bar_ts") is not None else None,
        "intent_label": intent.get("intent_label") if intent else None,
        "intent_score": intent.get("intent_score") if intent else None,
    }

    payload = {
        "symbol": symbol,
        "family": "HEAVENLY",
        "bias": direction,
        "stage": stage,
        "score": round(score, 2),
        "session": session,
        "last": last_price,
        "entry": float(entry) if stage in ("SETUP", "ENTRY") else None,
        "stop": stops_targets.get("stop") if stage == "ENTRY" else None,
        "tp0": stops_targets.get("tp1") if stage == "ENTRY" else None,
        "tp1": stops_targets.get("tp2") if stage == "ENTRY" else None,
        "tp2": stops_targets.get("tp3") if stage == "ENTRY" else None,
        "as_of": last5_ts.isoformat(),
        "why": " | ".join(why_parts),
        "extras": extras,
    }
    return payload
