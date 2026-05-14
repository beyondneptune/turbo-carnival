"""5-minute bar ingest for MFTv1 from Alpaca IEX historical.

Pulls SPY/QQQ/IWM/DIA/MDY 5-min OHLCV bars 2017-01-01 to present.
Stores per-symbol partitioned by year for fast filtering.

Run:
    .venv/bin/python -m research.mft_v1.ingest --refresh
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
CACHE_DIR = ROOT / "data" / "cache" / "mft_v1"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "MDY"]
ALPACA_DATA = "https://data.alpaca.markets"
START_DATE = "2017-01-01"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("mft_v1.ingest")


def _headers() -> dict:
    key = os.environ.get("ALPACA_AEGIS_KEY")
    sec = os.environ.get("ALPACA_AEGIS_SECRET")
    if not key or not sec:
        raise SystemExit("ALPACA_AEGIS_KEY/SECRET missing")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}


def fetch_5min_bars(symbol: str, start: str, end: str,
                    feed: str = "iex") -> pd.DataFrame:
    """Paginate Alpaca 5-min bars between [start, end]. Returns one DataFrame."""
    rows = []
    page_token = None
    h = _headers()
    while True:
        params = {
            "timeframe": "5Min",
            "start": f"{start}T13:30:00Z",   # 09:30 ET = 13:30 UTC (winter); summer 14:30
            "end": f"{end}T21:00:00Z",
            "limit": 10000,
            "feed": feed,
            "adjustment": "split",
        }
        if page_token:
            params["page_token"] = page_token
        r = requests.get(f"{ALPACA_DATA}/v2/stocks/{symbol}/bars",
                         headers=h, params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"alpaca {symbol} HTTP {r.status_code}: {r.text[:300]}")
        body = r.json()
        for bar in body.get("bars", []) or []:
            rows.append({
                "ts_utc": bar["t"],
                "open": bar["o"], "high": bar["h"],
                "low": bar["l"], "close": bar["c"],
                "volume": bar["v"], "vwap": bar.get("vw"),
                "trade_count": bar.get("n"),
            })
        page_token = body.get("next_page_token")
        if not page_token:
            break
        time.sleep(0.05)  # well under 200/min rate limit
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["ts_utc"] = pd.to_datetime(df["ts_utc"])
    df["ts_et"] = df["ts_utc"].dt.tz_convert("America/New_York")
    df["session_date"] = df["ts_et"].dt.date
    df["bar_time_et"] = df["ts_et"].dt.strftime("%H:%M")
    # RTH filter — keep 09:30 to 16:00 ET only
    df = df[(df["bar_time_et"] >= "09:30") & (df["bar_time_et"] < "16:00")].reset_index(drop=True)
    return df


def cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{symbol}.parquet"


def needs_refresh(symbol: str, max_age_hours: int = 18) -> bool:
    p = cache_path(symbol)
    if not p.exists():
        return True
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).total_seconds() > max_age_hours * 3600


def validate(symbol: str, df: pd.DataFrame) -> dict:
    if df.empty:
        return {"ok": False, "reason": "empty"}
    n_days = df["session_date"].nunique()
    bars_per_day = df.groupby("session_date").size()
    median_bars = bars_per_day.median()
    return {
        "ok": median_bars >= 70,  # 78 bars in full RTH; allow some half-days
        "n_bars": int(len(df)),
        "n_sessions": int(n_days),
        "first_session": str(df["session_date"].min()),
        "last_session": str(df["session_date"].max()),
        "median_bars_per_session": int(median_bars),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--symbols", nargs="*", default=UNIVERSE)
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    args = p.parse_args(argv)

    # Load env
    env_file = ROOT / "council" / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v.strip('"').strip("'"))

    summary: dict[str, dict] = {}
    for sym in args.symbols:
        if not args.refresh and not needs_refresh(sym):
            cached = pd.read_parquet(cache_path(sym))
            summary[sym] = validate(sym, cached)
            summary[sym]["from_cache"] = True
            log.info("%s: cached (%d rows)", sym, len(cached))
            continue
        log.info("%s: fetching 5-min bars %s..%s", sym, args.start, args.end)
        t0 = time.time()
        df = fetch_5min_bars(sym, args.start, args.end)
        log.info("%s: %d bars in %.1fs", sym, len(df), time.time() - t0)
        if df.empty:
            summary[sym] = {"ok": False, "reason": "empty_fetch"}
            continue
        df.to_parquet(cache_path(sym), index=False)
        summary[sym] = validate(sym, df)
        summary[sym]["from_cache"] = False

    all_ok = all(v.get("ok") for v in summary.values())
    print("\n=== MFTv1 INGEST SUMMARY ===")
    for sym, v in summary.items():
        ok = "✓" if v.get("ok") else "✗"
        print(f"  {ok} {sym:4} {v}")
    print(f"\noverall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
