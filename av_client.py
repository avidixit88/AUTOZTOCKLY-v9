from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd
import requests

BASE_URL = "https://www.alphavantage.co/query"


@dataclass
class AVConfig:
    api_key: str
    min_seconds_between_calls: float = 1.0
    entitlement: Optional[str] = None  # e.g., "realtime" for premium plans


class AlphaVantageClient:
    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or os.getenv("ALPHAVANTAGE_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ALPHAVANTAGE_API_KEY env var.")
        self.cfg = AVConfig(api_key=api_key)
        # Premium plans require the `entitlement` parameter (commonly "realtime")
        # to avoid stale/delayed intraday candles.
        env_ent = (os.getenv("ALPHAVANTAGE_ENTITLEMENT") or os.getenv("AV_ENTITLEMENT") or "").strip()
        if env_ent:
            self.cfg.entitlement = env_ent
        self._last_call = 0.0

    def _throttle(self):
        wait = self.cfg.min_seconds_between_calls - (time.time() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def _get(self, params: Dict[str, Any]) -> Dict[str, Any]:
        self._throttle()
        params = dict(params)
        params["apikey"] = self.cfg.api_key
        if self.cfg.entitlement:
            params["entitlement"] = self.cfg.entitlement
        r = requests.get(BASE_URL, params=params, timeout=30)
        self._last_call = time.time()
        r.raise_for_status()
        j = r.json()
        if "Error Message" in j:
            raise RuntimeError(j["Error Message"])
        if "Note" in j:
            # Rate limit note
            raise RuntimeError(j["Note"])
        return j

    def fetch_intraday(self, symbol: str, interval: str = "1min", outputsize: str = "compact") -> pd.DataFrame:
        j = self._get({
            "function": "TIME_SERIES_INTRADAY",
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "datatype": "json",
        })
        key = next((k for k in j.keys() if "Time Series" in k), None)
        if not key:
            raise RuntimeError(f"Unexpected response keys: {list(j.keys())}")
        ts = j[key]
        df = pd.DataFrame.from_dict(ts, orient="index").rename(columns={
            "1. open": "open",
            "2. high": "high",
            "3. low": "low",
            "4. close": "close",
            "5. volume": "volume",
        })
        df.index = pd.to_datetime(df.index)
        # Alpha Vantage timestamps are in US/Eastern (ET). Make the index tz-aware
        # immediately so every downstream time filter / session logic is consistent.
        try:
            if df.index.tz is None:
                df.index = df.index.tz_localize("America/New_York")
            else:
                df.index = df.index.tz_convert("America/New_York")
        except Exception:
            # If tz ops fail, keep naive; downstream code will localize defensively.
            pass

        df = df.astype(float).sort_index()
        return df

    def fetch_quote(self, symbol: str) -> Optional[float]:
        j = self._get({"function": "GLOBAL_QUOTE", "symbol": symbol})
        q = j.get("Global Quote") or {}
        px = q.get("05. price")
        try:
            return float(px) if px is not None else None
        except Exception:
            return None
