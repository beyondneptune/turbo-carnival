"""Earnings calendar gate for ATHENA v2.1.

Fetches upcoming earnings dates via yfinance and blocks entries on
symbols inside their earnings blackout window (default ±1 trading day).

Wiring into the host (after the news-sentiment gate, around line 825):

    from scripts.earnings_calendar_gate import is_earnings_blackout
    if is_earnings_blackout(sym):
        log_decision(
            conn, lane=lane.nickname, symbol=sym, action="skip",
            reason="earnings_blackout", ...
        )
        continue

Earnings dates are cached on disk for 24h (data/cache/earnings.json) so
we don't hammer yfinance from every tick. Re-fetch happens at startup
and at midnight UTC.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

REPO = Path("/Users/weeoto/turbo-carnival")
CACHE_PATH = REPO / "data" / "cache" / "earnings.json"
CACHE_TTL_SEC = 24 * 60 * 60  # 24h

# Days to block around earnings date (1 = block day-before, day-of, day-after)
BLACKOUT_DAYS = int(os.environ.get("ATHENA_EARNINGS_BLACKOUT_DAYS", "1"))

# Symbols to keep earnings dates for (matches host universe)
UNIVERSE = ("SPY", "QQQ", "NVDA", "TSLA", "AMD", "AAPL", "META")


def _load_cache() -> dict:
    if CACHE_PATH.exists():
        try:
            return json.loads(CACHE_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_cache(data: dict) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    CACHE_PATH.write_text(json.dumps(data, indent=2, default=str))


def _fetch_earnings_dates(symbol: str) -> list[str]:
    """Return list of ISO date strings (next 12 months) of earnings dates."""
    try:
        import yfinance as yf
    except ImportError:
        return []
    try:
        t = yf.Ticker(symbol)
        cal = t.calendar
        out = []
        # yfinance returns a dict {"Earnings Date": [Timestamp, ...]} or DataFrame
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
        else:
            try:
                dates = cal.loc["Earnings Date"].tolist()
            except Exception:
                dates = []
        for d in dates:
            try:
                if hasattr(d, "strftime"):
                    out.append(d.strftime("%Y-%m-%d"))
                else:
                    out.append(str(d)[:10])
            except Exception:
                continue
        return out
    except Exception:
        return []


def refresh_cache(force: bool = False) -> dict:
    """Re-fetch all symbols' earnings if cache is stale or missing."""
    cache = _load_cache()
    now = time.time()
    cache_age = now - cache.get("_fetched_at", 0)
    if not force and cache_age < CACHE_TTL_SEC and all(s in cache for s in UNIVERSE):
        return cache

    new_cache = {"_fetched_at": now}
    for sym in UNIVERSE:
        dates = _fetch_earnings_dates(sym)
        new_cache[sym] = dates
    _save_cache(new_cache)
    return new_cache


def is_earnings_blackout(symbol: str, *, now: datetime | None = None) -> bool:
    """True if `symbol` is within BLACKOUT_DAYS of its next earnings."""
    cache = _load_cache()
    if (time.time() - cache.get("_fetched_at", 0)) > CACHE_TTL_SEC:
        cache = refresh_cache()
    if now is None:
        now = datetime.now(timezone.utc)
    dates = cache.get(symbol, [])
    if not dates:
        return False
    for d in dates:
        try:
            earnings_date = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        delta = abs((earnings_date - now).days)
        if delta <= BLACKOUT_DAYS:
            return True
    return False


def next_earnings(symbol: str) -> str | None:
    """Return the next future earnings date string, or None."""
    cache = _load_cache()
    now = datetime.now(timezone.utc)
    dates = cache.get(symbol, [])
    future = []
    for d in dates:
        try:
            dt = datetime.strptime(d, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            continue
        if dt > now:
            future.append(d)
    return min(future) if future else None


if __name__ == "__main__":
    print("Refreshing earnings cache...")
    cache = refresh_cache(force=True)
    print()
    print(f"Earnings calendar · blackout ±{BLACKOUT_DAYS} day(s)")
    print("-" * 70)
    for sym in UNIVERSE:
        nxt = next_earnings(sym)
        blocked = is_earnings_blackout(sym)
        flag = "BLOCK" if blocked else " ok  "
        all_dates = cache.get(sym, [])
        print(f"  {sym:6s}  next: {nxt or 'n/a'}  {flag}  (all known: {all_dates})")
