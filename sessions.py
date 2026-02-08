from __future__ import annotations

import pandas as pd
import pytz
from datetime import time


ET = pytz.timezone("America/New_York")


def _to_et(ts: pd.Timestamp) -> pd.Timestamp:
    if ts.tzinfo is None:
        return ts.tz_localize(ET)
    return ts.tz_convert(ET)


def classify_liquidity_phase(ts: pd.Timestamp) -> str:
    """
    Returns: PREMARKET / RTH / AFTERHOURS
    Premarket: 04:00–09:30 ET
    RTH:       09:30–16:00 ET
    After:     16:00–20:00 ET (approx; AV can include extended)
    """
    t = _to_et(ts).time()
    if time(4, 0) <= t < time(9, 30):
        return "PREMARKET"
    if time(9, 30) <= t < time(16, 0):
        return "RTH"
    return "AFTERHOURS"


def classify_session(
    ts: pd.Timestamp | None,
    *,
    allow_opening: bool = True,
    allow_midday: bool = True,
    allow_power: bool = True,
    # Defaults preserve legacy behavior for call sites that do NOT pass allow_* flags
    # (REV/RIDE/MSS), where session classification should never be implicitly blocked.
    allow_premarket: bool = True,
    allow_afterhours: bool = True,
) -> str:
    """Classify a timestamp into a session bucket.

    Returns one of: PREMARKET / OPENING / MIDDAY / POWER / AFTERHOURS / OFF

    IMPORTANT:
    - This function is called in two ways across the codebase:
      1) classify_session(ts)  -> legacy call sites (REV/RIDE/MSS)
      2) classify_session(ts, allow_opening=..., ...) -> SWING call site
    - To preserve existing behavior, all new parameters are optional
      keyword-only flags with defaults matching prior assumptions.
    """
    if ts is None:
        return "OFF"

    phase = classify_liquidity_phase(ts)
    if phase == "PREMARKET":
        return "PREMARKET" if allow_premarket else "OFF"
    if phase == "AFTERHOURS":
        return "AFTERHOURS" if allow_afterhours else "OFF"

    # Regular Trading Hours sub-buckets
    t = _to_et(ts).time()
    if time(9, 30) <= t < time(11, 0):
        return "OPENING" if allow_opening else "OFF"
    if time(11, 0) <= t < time(15, 0):
        return "MIDDAY" if allow_midday else "OFF"
    if time(15, 0) <= t < time(16, 0):
        return "POWER" if allow_power else "OFF"
    return "OFF"
