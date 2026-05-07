"""Thin Polygon REST adapter for the free tier (5 req/min, 2-year history).
Reads POLYGON_API_KEY from environment via python-dotenv if available."""
from __future__ import annotations

import os
import time
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.polygon.io"
FREE_TIER_RPM = 5  # requests per minute on the free tier
DEFAULT_TIMEOUT_S = 30.0

_AGG_COLUMNS = ["open", "high", "low", "close", "volume", "vwap", "transactions"]
_AGG_FIELD_MAP = {
    "o": "open",
    "h": "high",
    "l": "low",
    "c": "close",
    "v": "volume",
    "vw": "vwap",
    "n": "transactions",
}
_CONTRACT_COLUMNS = [
    "ticker",
    "underlying_ticker",
    "contract_type",
    "strike_price",
    "expiration_date",
    "exercise_style",
    "shares_per_contract",
]


def get_api_key() -> str | None:
    """Returns POLYGON_API_KEY from environment (loads .env if available)."""
    try:
        from pathlib import Path

        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    except ImportError:
        pass
    return os.environ.get("POLYGON_API_KEY")


class PolygonError(Exception):
    """Generic Polygon API error."""


class PolygonAuthError(PolygonError):
    """Raised on missing/invalid credentials (401, 403)."""


class PolygonRateLimitError(PolygonError):
    """Raised on HTTP 429 from Polygon (free tier exceeded)."""


class PolygonFreeClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        rate_limit_per_min: int = FREE_TIER_RPM,
        session: requests.Session | None = None,
    ) -> None:
        self._api_key = api_key or get_api_key()
        if not self._api_key:
            raise PolygonAuthError(
                "POLYGON_API_KEY not set; add it to .env or pass api_key= to PolygonFreeClient"
            )
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        if rate_limit_per_min <= 0:
            raise ValueError("rate_limit_per_min must be > 0")
        self._min_interval_s = 60.0 / rate_limit_per_min
        self._last_request_at = 0.0
        self._session = session or requests.Session()

    def _throttle(self) -> None:
        """Sleep just enough to stay under rate_limit_per_min."""
        now = time.monotonic()
        elapsed = now - self._last_request_at
        wait = self._min_interval_s - elapsed
        if wait > 0:
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    def _get(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self._throttle()
        url = f"{self._base_url}{endpoint}"
        merged_params: dict[str, Any] = {"apiKey": self._api_key}
        if params:
            for k, v in params.items():
                if v is not None:
                    merged_params[k] = v
        response = self._session.get(url, params=merged_params, timeout=self._timeout_s)
        status = response.status_code
        if status == 429:
            raise PolygonRateLimitError(
                f"Polygon rate limit hit (429) on {endpoint}; free tier is {FREE_TIER_RPM}/min"
            )
        if status in (401, 403):
            raise PolygonAuthError(f"Polygon auth failure ({status}) on {endpoint}")
        if status >= 400:
            raise PolygonError(f"Polygon HTTP {status} on {endpoint}: {response.text[:200]}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise PolygonError(f"Polygon returned non-JSON on {endpoint}: {exc}") from exc
        if not isinstance(payload, dict):
            raise PolygonError(f"Polygon returned unexpected payload type on {endpoint}")
        return payload

    def fetch_aggregates(
        self,
        ticker: str,
        from_date: str,
        to_date: str,
        timespan: str = "day",
        multiplier: int = 1,
        adjusted: bool = True,
    ) -> pd.DataFrame:
        endpoint = (
            f"/v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from_date}/{to_date}"
        )
        params = {"adjusted": "true" if adjusted else "false"}
        payload = self._get(endpoint, params=params)
        results = payload.get("results") or []
        return _aggregates_to_df(results)

    def fetch_options_contracts(
        self,
        underlying_ticker: str,
        expiration_date: str | None = None,
        contract_type: str | None = None,
        as_of: str | None = None,
        limit: int = 1000,
    ) -> pd.DataFrame:
        endpoint = "/v3/reference/options/contracts"
        params: dict[str, Any] = {
            "underlying_ticker": underlying_ticker,
            "expiration_date": expiration_date,
            "contract_type": contract_type,
            "as_of": as_of,
            "limit": limit,
        }
        payload = self._get(endpoint, params=params)
        results = payload.get("results") or []
        return _contracts_to_df(results)

    def fetch_options_aggregates(
        self,
        option_ticker: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        return self.fetch_aggregates(
            ticker=option_ticker,
            from_date=from_date,
            to_date=to_date,
            timespan="day",
            multiplier=1,
            adjusted=True,
        )


def _aggregates_to_df(results: list[dict[str, Any]]) -> pd.DataFrame:
    if not results:
        empty = pd.DataFrame(columns=_AGG_COLUMNS)
        empty.index = pd.DatetimeIndex([], name="date")
        return empty
    rows = []
    timestamps = []
    for bar in results:
        timestamps.append(bar.get("t"))
        rows.append({_AGG_FIELD_MAP[k]: bar.get(k) for k in _AGG_FIELD_MAP if k in bar})
    df = pd.DataFrame(rows)
    for col in _AGG_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[_AGG_COLUMNS]
    # Polygon returns 't' as Unix milliseconds in UTC.
    df.index = pd.to_datetime(timestamps, unit="ms", utc=True).tz_convert(None)
    df.index.name = "date"
    return df


def _contracts_to_df(results: list[dict[str, Any]]) -> pd.DataFrame:
    if not results:
        return pd.DataFrame(columns=_CONTRACT_COLUMNS)
    df = pd.DataFrame(results)
    for col in _CONTRACT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[_CONTRACT_COLUMNS].copy()
