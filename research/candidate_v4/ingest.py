"""Daily ingest for candidate_v4 — ^VIX, ^VIX3M, SVXY, ^GSPC."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from research.candidate_v1.ingest import fetch_yfinance_daily, validate

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "candidate_v4"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = ["^VIX", "^VIX3M", "SVXY", "^GSPC"]
START_DATE = "2015-01-01"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("candidate_v4.ingest")


def safe_filename(sym: str) -> str:
    return sym.replace("^", "_").replace("/", "_")


def cache_path(symbol: str) -> Path:
    return CACHE_DIR / f"{safe_filename(symbol)}.parquet"


def needs_refresh(symbol: str, max_age_days: int = 1) -> bool:
    p = cache_path(symbol)
    if not p.exists():
        return True
    mtime = datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)
    return (datetime.now(timezone.utc) - mtime).days >= max_age_days


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--symbols", nargs="*", default=UNIVERSE)
    p.add_argument("--start", default=START_DATE)
    p.add_argument("--end", default=datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    args = p.parse_args(argv)

    summary: dict[str, dict] = {}
    for sym in args.symbols:
        if not args.refresh and not needs_refresh(sym):
            cached = pd.read_parquet(cache_path(sym))
            summary[sym] = validate(sym, cached, args.start)
            log.info("%s: cached (%d rows)", sym, len(cached))
            continue
        log.info("%s: fetching %s..%s", sym, args.start, args.end)
        df = fetch_yfinance_daily(sym, args.start, args.end)
        if df.empty:
            summary[sym] = {"ok": False, "reason": "empty_fetch"}
            log.error("%s: empty", sym); continue
        df["source"] = "yfinance"
        df.to_parquet(cache_path(sym), index=False)
        summary[sym] = validate(sym, df, args.start)

    all_ok = all(v.get("ok") for v in summary.values())
    print("\n=== INGEST SUMMARY (v4) ===")
    for sym, v in summary.items():
        print(f"  {'✓' if v.get('ok') else '✗'} {sym:10} {v}")
    print(f"\noverall: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
