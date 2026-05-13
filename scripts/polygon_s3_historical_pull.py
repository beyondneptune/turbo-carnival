"""Pull minute bars from Polygon S3 flat files (free tier).

Polygon publishes daily flat files at s3://flatfiles.polygon.io/us_stocks_sip/
in CSV.gz format. Free tier gives 5 years of historical minute bars at zero
incremental cost beyond bandwidth.

This script:
  1. Walks the universe + date range
  2. Downloads daily files
  3. Filters to our symbols
  4. Writes consolidated per-symbol minute parquet to data/cache/minute_bars/

Run in tmux for the overnight pull:

    tmux new -d -s polygon_pull "/usr/bin/caffeinate -dimsu \\
        .venv/bin/python scripts/polygon_s3_historical_pull.py \\
            --start 2025-01-01 --end 2026-05-01 \\
            2>&1 | tee data/logs/polygon_pull.log"

Requires:
  - POLYGON_S3_ACCESS_KEY / POLYGON_S3_SECRET (free tier)
  - pip install boto3

See polygon.io/docs/flat-files for access setup (free account → S3 keys).

Note: requires Polygon S3 credentials. If you don't have them yet, sign up
at https://polygon.io/ (free tier is sufficient for flat files).
"""
from __future__ import annotations

import argparse
import gzip
import io
import os
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path("/Users/weeoto/turbo-carnival")
OUT = REPO / "data" / "cache" / "minute_bars"
OUT.mkdir(parents=True, exist_ok=True)

UNIVERSE = ("SPY", "QQQ", "NVDA", "TSLA", "AMD", "AAPL", "META")
S3_PREFIX = "us_stocks_sip/minute_aggs_v1/"
# These can be overridden via env (account-specific endpoint/bucket)
S3_ENDPOINT_DEFAULT = "https://files.polygon.io"
S3_BUCKET_DEFAULT = "flatfiles"


def _load_env_if_present():
    """Read council/.env.local into os.environ if not already set."""
    env_path = REPO / "council" / ".env.local"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _get_s3_client():
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise RuntimeError("boto3 not installed. pip install boto3") from e
    _load_env_if_present()
    key = os.environ.get("POLYGON_S3_ACCESS_KEY")
    sec = os.environ.get("POLYGON_S3_SECRET")
    endpoint = os.environ.get("POLYGON_S3_ENDPOINT", S3_ENDPOINT_DEFAULT)
    if not key or not sec:
        raise RuntimeError("POLYGON_S3_ACCESS_KEY/POLYGON_S3_SECRET env vars required")
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=key,
        aws_secret_access_key=sec,
        config=Config(signature_version="s3v4", s3={"addressing_style": "path"}),
    )


def _get_bucket() -> str:
    return os.environ.get("POLYGON_S3_BUCKET", S3_BUCKET_DEFAULT)


def _date_iter(start: str, end: str):
    s = datetime.strptime(start, "%Y-%m-%d")
    e = datetime.strptime(end, "%Y-%m-%d")
    while s <= e:
        # Polygon flat files are weekdays only.
        # Key layout is: YYYY/MM/YYYY-MM-DD.csv.gz
        if s.weekday() < 5:
            yield s.strftime("%Y/%m/%Y-%m-%d")
        s += timedelta(days=1)


def _fetch_day(s3, date_path: str) -> bytes | None:
    """Download one day's minute_aggs CSV.gz from Polygon S3."""
    key = f"{S3_PREFIX}{date_path}.csv.gz"
    try:
        obj = s3.get_object(Bucket=_get_bucket(), Key=key)
        return obj["Body"].read()
    except Exception as exc:
        print(f"  fetch {date_path}: skip ({exc})")
        return None


def _filter_to_universe(gz_bytes: bytes, universe: set[str]):
    """Parse gzipped CSV and yield rows for our universe."""
    import csv
    raw = gzip.decompress(gz_bytes).decode("utf-8")
    reader = csv.DictReader(io.StringIO(raw))
    for row in reader:
        sym = row.get("ticker") or row.get("symbol")
        if sym in universe:
            yield row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-01-01")
    parser.add_argument("--end", default=datetime.now().strftime("%Y-%m-%d"))
    parser.add_argument("--universe", nargs="*", default=list(UNIVERSE))
    parser.add_argument("--dry-run", action="store_true",
                          help="Skip actual downloads, just count")
    args = parser.parse_args()

    print(f"Polygon S3 historical pull · {args.start} → {args.end}")
    print(f"Universe: {args.universe}")
    print(f"Output: {OUT}")
    print()

    if args.dry_run:
        dates = list(_date_iter(args.start, args.end))
        print(f"Would download {len(dates)} weekdays (~{len(dates)*5}MB compressed)")
        return

    try:
        s3 = _get_s3_client()
    except RuntimeError as e:
        print(f"ERROR: {e}")
        print()
        print("To get Polygon S3 credentials (free):")
        print("  1. Sign up at https://polygon.io/")
        print("  2. Settings → API Keys → S3 Access Keys")
        print("  3. Add to council/.env.local:")
        print("       POLYGON_S3_ACCESS_KEY=<your_key>")
        print("       POLYGON_S3_SECRET=<your_secret>")
        return

    import pandas as pd
    universe_set = set(args.universe)
    by_symbol: dict[str, list] = {s: [] for s in universe_set}
    n_days = 0
    for date_path in _date_iter(args.start, args.end):
        gz = _fetch_day(s3, date_path)
        if gz is None:
            continue
        n_days += 1
        n_rows = 0
        for row in _filter_to_universe(gz, universe_set):
            by_symbol[row.get("ticker") or row.get("symbol")].append(row)
            n_rows += 1
        if n_days % 20 == 0:
            print(f"  {date_path}: cumulative {n_days} days, {sum(len(v) for v in by_symbol.values())} bars")

    print(f"\nCompleted {n_days} days. Writing per-symbol parquet...")
    for sym, rows in by_symbol.items():
        if not rows:
            print(f"  {sym}: no data")
            continue
        df = pd.DataFrame(rows)
        out_path = OUT / f"{sym}.parquet"
        df.to_parquet(out_path, index=False)
        print(f"  {sym}: {len(df):,} bars → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
