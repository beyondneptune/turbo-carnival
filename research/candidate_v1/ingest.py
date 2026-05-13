"""Daily OHLCV ingest for candidate_v1 universe.

Source preference: Alpaca historical (IEX free feed) -> yfinance fallback.
Cache layout: data/cache/candidate_v1/<symbol>.parquet (one file per symbol).

Run:
    .venv/bin/python -m research.candidate_v1.ingest --refresh
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "candidate_v1"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = [
    "USO", "UNG", "GLD", "SLV", "DBA", "DBC",
    "CORN", "WEAT", "CPER", "PPLT",
]

START_DATE = "2015-01-01"

ALPACA_DATA = "https://data.alpaca.markets"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("candidate_v1.ingest")


def _alpaca_headers() -> dict[str, str] | None:
    key = os.environ.get("ALPACA_AEGIS_KEY")
    sec = os.environ.get("ALPACA_AEGIS_SECRET")
    if not key or not sec:
        return None
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}


def fetch_alpaca_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fetch daily bars from Alpaca IEX feed (free tier)."""
    h = _alpaca_headers()
    if not h:
        raise RuntimeError("ALPACA_AEGIS_KEY/SECRET not in env")

    rows: list[dict] = []
    page_token: str | None = None
    while True:
        params = {
            "timeframe": "1Day",
            "start": f"{start}T00:00:00Z",
            "end": f"{end}T23:59:59Z",
            "limit": 10000,
            "feed": "iex",
            "adjustment": "split",
        }
        if page_token:
            params["page_token"] = page_token
        r = requests.get(
            f"{ALPACA_DATA}/v2/stocks/{symbol}/bars",
            headers=h, params=params, timeout=20,
        )
        if r.status_code != 200:
            raise RuntimeError(f"alpaca {symbol} HTTP {r.status_code}: {r.text[:200]}")
        body = r.json()
        for bar in body.get("bars", []) or []:
            rows.append({
                "date": bar["t"][:10],
                "open": bar["o"], "high": bar["h"],
                "low": bar["l"], "close": bar["c"],
                "volume": bar["v"], "vwap": bar.get("vw"),
            })
        page_token = body.get("next_page_token")
        if not page_token:
            break
        time.sleep(0.25)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
    return df


def fetch_yfinance_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Fallback: yfinance daily bars."""
    import yfinance as yf
    t = yf.Ticker(symbol)
    df = t.history(start=start, end=end, interval="1d", auto_adjust=False)
    if df.empty:
        return df
    df = df.reset_index()
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"adj close": "adj_close"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df[["date", "open", "high", "low", "close", "volume"]].sort_values("date").reset_index(drop=True)


def cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.parquet"


def needs_refresh(symbol: str, max_age_days: int = 1) -> bool:
    p = cache_path(symbol)
    if not p.exists():
        return True
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    age = (datetime.now(timezone.utc) - mtime).days
    return age >= max_age_days


def load_cached(symbol: str) -> pd.DataFrame:
    p = cache_path(symbol)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def save_cache(symbol: str, df: pd.DataFrame) -> None:
    df.to_parquet(cache_path(symbol), index=False)


def fetch_one(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Primary: yfinance (full 2014+ daily history). Fallback: Alpaca IEX (2020+ only)."""
    try:
        df = fetch_yfinance_daily(symbol, start, end)
        if not df.empty:
            df["source"] = "yfinance"
            return df
        log.warning("yfinance returned empty for %s, falling back to alpaca", symbol)
    except Exception as e:
        log.warning("yfinance failed for %s (%s), falling back", symbol, e)
    df = fetch_alpaca_daily(symbol, start, end)
    if not df.empty:
        df["source"] = "alpaca_iex"
    return df


def validate(symbol: str, df: pd.DataFrame, expected_start: str) -> dict:
    """Quality checks: coverage, gap count, look-ahead screen."""
    if df.empty:
        return {"ok": False, "reason": "empty"}
    expected = pd.bdate_range(start=expected_start, end=df["date"].max())
    actual = pd.to_datetime(df["date"]).dt.normalize()
    missing = expected.difference(actual)
    gap_pct = len(missing) / len(expected) if len(expected) else 0.0
    has_future = (actual > pd.Timestamp.utcnow().tz_localize(None)).any()
    return {
        "ok": gap_pct < 0.05 and not has_future,
        "n_rows": int(len(df)),
        "first_date": str(df["date"].min())[:10],
        "last_date": str(df["date"].max())[:10],
        "missing_bdays": int(len(missing)),
        "gap_pct": round(gap_pct, 4),
        "future_dates_present": bool(has_future),
        "source": df["source"].iloc[0] if "source" in df.columns else "unknown",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true",
                   help="Refetch even if cache fresh")
    p.add_argument("--symbols", nargs="*", default=UNIVERSE,
                   help="Subset of universe to ingest")
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    args = p.parse_args(argv)

    summary: dict[str, dict] = {}
    for sym in args.symbols:
        if not args.refresh and not needs_refresh(sym):
            cached = load_cached(sym)
            summary[sym] = validate(sym, cached, args.start)
            summary[sym]["from_cache"] = True
            log.info("%s: using cache (%d rows)", sym, len(cached))
            continue
        log.info("%s: fetching %s..%s", sym, args.start, args.end)
        df = fetch_one(sym, args.start, args.end)
        if df.empty:
            log.error("%s: empty fetch", sym)
            summary[sym] = {"ok": False, "reason": "empty_fetch"}
            continue
        save_cache(sym, df)
        summary[sym] = validate(sym, df, args.start)
        summary[sym]["from_cache"] = False

    all_ok = all(v.get("ok") for v in summary.values())
    print("\n=== INGEST SUMMARY ===")
    for sym, v in summary.items():
        flag = "✓" if v.get("ok") else "✗"
        print(f"  {flag} {sym:6} {v}")
    print(f"\noverall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
