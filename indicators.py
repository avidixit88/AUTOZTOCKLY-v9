from __future__ import annotations

import numpy as np
import pandas as pd


def vwap(df: pd.DataFrame) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    return pv.cumsum() / df["volume"].cumsum().replace(0, np.nan)


def session_vwap(df: pd.DataFrame, tz: str = "America/New_York") -> pd.Series:
    if df.empty:
        return pd.Series(dtype="float64")

    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx_et = idx.tz_localize(tz)
    else:
        idx_et = idx.tz_convert(tz)

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]
    dates = pd.Series(idx_et.date, index=df.index)

    out = pd.Series(index=df.index, dtype="float64")
    for _, g in df.groupby(dates):
        gtp = tp.loc[g.index]
        gpv = pv.loc[g.index]
        out.loc[g.index] = gpv.cumsum() / g["volume"].cumsum().replace(0, np.nan)
    return out


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rolling_swing_lows(series: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    s = series
    is_low = pd.Series(False, index=s.index)
    for i in range(left, len(s) - right):
        window = s.iloc[i - left: i + right + 1]
        if s.iloc[i] == window.min():
            is_low.iloc[i] = True
    return is_low


def rolling_swing_highs(series: pd.Series, left: int = 3, right: int = 3) -> pd.Series:
    s = series
    is_high = pd.Series(False, index=s.index)
    for i in range(left, len(s) - right):
        window = s.iloc[i - left: i + right + 1]
        if s.iloc[i] == window.max():
            is_high.iloc[i] = True
    return is_high


def detect_fvg(df: pd.DataFrame):
    if len(df) < 3:
        return None, None
    h = df["high"].values
    l = df["low"].values
    bull = None
    bear = None
    for i in range(2, len(df)):
        if l[i] > h[i - 2]:
            bull = (float(h[i - 2]), float(l[i]))
        if h[i] < l[i - 2]:
            bear = (float(h[i]), float(l[i - 2]))
    return bull, bear


def find_order_block(df: pd.DataFrame, atr_series: pd.Series, side: str = "bull", lookback: int = 30):
    if len(df) < 10:
        return None, None, None
    df = df.tail(lookback).copy()
    atr_series = atr_series.reindex(df.index).ffill()

    o = df["open"].values
    c = df["close"].values
    h = df["high"].values
    l = df["low"].values
    at = atr_series.values
    idx = df.index

    if side == "bull":
        for i in range(len(df) - 4, -1, -1):
            if c[i] < o[i]:
                atr_i = at[i] if np.isfinite(at[i]) else None
                if not atr_i:
                    continue
                for j in range(i + 1, min(i + 4, len(df))):
                    if c[j] > h[i] and (c[j] - c[i]) > 1.0 * atr_i:
                        zone_low = float(l[i])
                        zone_high = float(o[i])
                        return min(zone_low, zone_high), max(zone_low, zone_high), idx[i]
    else:
        for i in range(len(df) - 4, -1, -1):
            if c[i] > o[i]:
                atr_i = at[i] if np.isfinite(at[i]) else None
                if not atr_i:
                    continue
                for j in range(i + 1, min(i + 4, len(df))):
                    if c[j] < l[i] and (c[i] - c[j]) > 1.0 * atr_i:
                        zone_high = float(h[i])
                        zone_low = float(o[i])
                        return min(zone_low, zone_high), max(zone_low, zone_high), idx[i]
    return None, None, None


def find_breaker_block(df: pd.DataFrame, atr_series: pd.Series, side: str = "bull", lookback: int = 60):
    if len(df) < 20:
        return None, None, None

    df = df.tail(lookback).copy()
    atr_series = atr_series.reindex(df.index).ffill()
    atr_last = float(atr_series.iloc[-1]) if np.isfinite(atr_series.iloc[-1]) else 0.0
    pad = 0.2 * atr_last if atr_last else 0.0

    if side == "bull":
        zl, zh, ts = find_order_block(df, atr_series, side="bear", lookback=lookback)
        if zl is None:
            return None, None, None
        if not (df["close"] > (zh + pad)).any():
            return None, None, None
        last = float(df["close"].iloc[-1])
        if (last >= (zl - pad)) and (last <= (zh + pad)):
            return float(zl), float(zh), ts
        return None, None, None
    else:
        zl, zh, ts = find_order_block(df, atr_series, side="bull", lookback=lookback)
        if zl is None:
            return None, None, None
        if not (df["close"] < (zl - pad)).any():
            return None, None, None
        last = float(df["close"].iloc[-1])
        if (last >= (zl - pad)) and (last <= (zh + pad)):
            return float(zl), float(zh), ts
        return None, None, None


def in_zone(price: float, zone_low: float, zone_high: float, buffer: float = 0.0) -> bool:
    return (price >= (zone_low - buffer)) and (price <= (zone_high + buffer))


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1/period, adjust=False).mean()
    roll_down = down.ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def adx(df: pd.DataFrame, period: int = 14):
    """Average Directional Index (ADX) with DI+ and DI-.

    This is a lightweight Wilder-style implementation suitable for intraday bars.
    Returns a tuple: (adx, plus_di, minus_di) as Series aligned to df.index.
    """
    if df is None or df.empty or len(df) < period + 2:
        empty = pd.Series(index=(df.index if df is not None else None), dtype="float64")
        return empty, empty, empty

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    # Wilder smoothing via rolling sums/means (good enough for scanner usage)
    atr = tr.rolling(period, min_periods=period).mean()
    plus_di = 100.0 * (plus_dm.rolling(period, min_periods=period).mean() / atr.replace(0, np.nan))
    minus_di = 100.0 * (minus_dm.rolling(period, min_periods=period).mean() / atr.replace(0, np.nan))

    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(period, min_periods=period).mean()

    return adx, plus_di, minus_di


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd - sig


def session_vwap(
    df: pd.DataFrame,
    tz: str = "America/New_York",
    include_premarket: bool = False,
    include_afterhours: bool = False,
) -> pd.Series:
    """
    Session VWAP that **resets each trading day at the US cash open (09:30 ET)** by default.

    - include_premarket=False => VWAP starts at 09:30 (RTH-only VWAP).
    - include_premarket=True  => VWAP starts at 04:00 (Extended VWAP, useful for PM scalps).
    - include_afterhours=True => continues after 16:00 (extended into AH).

    Bars outside the included session window receive NaN.
    """
    if df.empty:
        return pd.Series(dtype="float64")

    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx_et = idx.tz_localize(tz)
    else:
        idx_et = idx.tz_convert(tz)

    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    pv = tp * df["volume"]

    # session window
    start_h, start_m = (4, 0) if include_premarket else (9, 30)
    end_h, end_m = (20, 0) if include_afterhours else (16, 0)

    t = pd.Series([d.time() for d in idx_et], index=df.index)
    # Build boolean mask for included times
    def _in_window(tt):
        return (tt.hour, tt.minute) >= (start_h, start_m) and (tt.hour, tt.minute) < (end_h, end_m)

    mask = t.map(_in_window).astype(bool)

    # Group by ET date and compute VWAP cumulatives only within included session window
    dates = pd.Series(idx_et.date, index=df.index)
    out = pd.Series(index=df.index, dtype="float64")

    for d, g in df.groupby(dates):
        gi = g.index
        gmask = mask.loc[gi]
        if not gmask.any():
            continue
        gi_in = gi[gmask.values]
        gpv = pv.loc[gi_in].cumsum()
        gvol = g.loc[gi_in, "volume"].cumsum().replace(0, np.nan)
        out.loc[gi_in] = gpv / gvol

    return out
