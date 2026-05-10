"""Fetch non-Polygon offline context data for paper_v1 research.

This intentionally does not call Polygon. It fills a local context cache with
daily OHLCV from yfinance, corporate actions from yfinance, and macro/rates/
volatility series from FRED CSV downloads. These datasets can help later
feature engineering and postmortems without competing with the Polygon Free
options-pricing rate limit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date, timedelta, timezone, datetime
from io import StringIO
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingest.yfinance_adapter import fetch_daily_prices  # noqa: E402
from signals.paper_v1_universe import PAPER_V1_CORE, PAPER_V1_SATELLITE  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "offline_context"
DEFAULT_STATUS_FILE = REPO_ROOT / "data" / "offline_context_status.json"

START_DATE = date(2024, 5, 7)
END_DATE = date(2026, 5, 6)

MARKET_ETFS = (
    "SPY", "QQQ", "IWM", "DIA",
    "TLT", "IEF", "SHY", "HYG", "LQD",
    "GLD", "USO", "SMH", "KRE",
    "UUP", "FXE", "FXY", "VIXY", "SVXY",
)
SECTOR_ETFS = (
    "XLB", "XLC", "XLE", "XLF", "XLI", "XLK",
    "XLP", "XLRE", "XLU", "XLV", "XLY",
)
VOL_TICKERS = ("^VIX", "^VVIX", "^VIX3M", "^VXN", "^RVX", "^MOVE")

# Preserve the frozen paper_v1 ticker on disk while querying current Yahoo
# symbols when a company has changed tickers during/after the study window.
YFINANCE_ALIASES = {
    "SQ": "XYZ",  # Block changed its listed ticker; Yahoo serves history on XYZ.
}

FRED_SERIES = {
    # Rates / curves
    "DGS1MO": "1-month Treasury yield",
    "DGS3MO": "3-month Treasury yield",
    "DGS2": "2-year Treasury yield",
    "DGS10": "10-year Treasury yield",
    "T10Y2Y": "10Y minus 2Y Treasury spread",
    "T10Y3M": "10Y minus 3M Treasury spread",
    "SOFR": "Secured Overnight Financing Rate",
    "EFFR": "Effective Federal Funds Rate (daily)",
    "FEDFUNDS": "Effective federal funds rate",
    "IORB": "Interest on reserve balances",
    "WALCL": "Federal Reserve total assets",
    "RRPONTSYD": "Overnight reverse repurchase agreements",
    # Vol / risk / macro context
    "VIXCLS": "CBOE VIX close",
    "VXNCLS": "CBOE NASDAQ-100 Volatility Index",
    "RVXCLS": "CBOE Russell 2000 Volatility Index",
    "OVXCLS": "CBOE Crude Oil ETF Volatility Index",
    "GVZCLS": "CBOE Gold ETF Volatility Index",
    "BAMLH0A0HYM2": "ICE BofA US high yield option-adjusted spread",
    "BAMLH0A0HYM2EY": "ICE BofA US high yield effective yield",
    "BAMLC0A0CM": "ICE BofA US corporate option-adjusted spread",
    "BAMLC0A0CMEY": "ICE BofA US corporate effective yield",
    "NFCI": "Chicago Fed National Financial Conditions Index",
    "ANFCI": "Chicago Fed Adjusted National Financial Conditions Index",
    "STLFSI4": "St. Louis Fed Financial Stress Index",
    "DCOILWTICO": "WTI crude oil spot price",
    "DTWEXBGS": "Nominal broad dollar index",
    "DEXUSEU": "US / Euro foreign exchange rate",
    "DEXJPUS": "Japan / US foreign exchange rate",
    "USEPUINDXD": "US economic policy uncertainty index",
    "CPIAUCSL": "CPI all urban consumers",
    "UNRATE": "Civilian unemployment rate",
    "ICSA": "Initial unemployment claims",
    "CCSA": "Continued unemployment claims",
    "PAYEMS": "Total nonfarm payrolls",
    "INDPRO": "Industrial production index",
    "UMCSENT": "University of Michigan consumer sentiment",
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def safe_symbol_name(symbol: str) -> str:
    """Filesystem-safe ticker/series name."""
    keep = []
    for ch in symbol.upper():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {".", "-", "_"}:
            keep.append(ch)
        elif ch == "^":
            keep.append("IDX_")
        else:
            keep.append("_")
    return "".join(keep).strip("_")


def atomic_write_json(target: Path, obj: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))
    os.replace(tmp, target)


def write_status(stage: str, status: str, *, status_file: Path, **fields) -> None:
    existing = {}
    if status_file.exists():
        try:
            existing = json.loads(status_file.read_text())
        except (OSError, ValueError):
            existing = {}
    stages = existing.get("stages") if isinstance(existing.get("stages"), dict) else {}
    stages[stage] = {"status": status, "updated_at_utc": utc_now_iso(), **fields}
    atomic_write_json(
        status_file,
        {
            "phase": "offline_context",
            "started_at_utc": existing.get("started_at_utc") or utc_now_iso(),
            "last_update_utc": utc_now_iso(),
            "stages": stages,
        },
    )


def unique_ordered(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.upper()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def default_tickers() -> list[str]:
    return unique_ordered(
        list(PAPER_V1_CORE)
        + list(PAPER_V1_SATELLITE)
        + list(MARKET_ETFS)
        + list(SECTOR_ETFS)
        + list(VOL_TICKERS)
    )


def write_frame(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    if isinstance(out.index, pd.DatetimeIndex):
        out.index.name = out.index.name or "date"
        out = out.reset_index()
    tmp = target.with_suffix(target.suffix + ".tmp")
    out.to_parquet(tmp, index=False)
    os.replace(tmp, target)


def fetch_yfinance_daily(
    tickers: list[str],
    *,
    start: date,
    end: date,
    output_dir: Path,
    status_file: Path,
    chunk_size: int,
    refresh: bool,
) -> dict:
    target_dir = output_dir / "yfinance" / "daily"
    failures: dict[str, str] = {}
    written = 0
    skipped = 0
    chunks = [tickers[i:i + chunk_size] for i in range(0, len(tickers), chunk_size)]
    for idx, chunk in enumerate(chunks, start=1):
        needed = [
            t for t in chunk
            if refresh or not (target_dir / f"{safe_symbol_name(t)}.parquet").exists()
        ]
        if not needed:
            skipped += len(chunk)
            continue
        query_tickers = unique_ordered(YFINANCE_ALIASES.get(t, t) for t in needed)
        write_status(
            "yfinance_daily",
            "running",
            status_file=status_file,
            chunk=idx,
            chunks=len(chunks),
            tickers=",".join(needed),
            query_tickers=",".join(query_tickers),
            written=written,
            skipped=skipped,
        )
        try:
            # yfinance treats end as exclusive; add one day so END_DATE is included.
            prices = fetch_daily_prices(query_tickers, start, end + timedelta(days=1), auto_adjust=True)
        except Exception as exc:  # pragma: no cover - external service defensive
            for ticker in needed:
                failures[ticker] = repr(exc)
            continue
        for ticker in needed:
            source_ticker = YFINANCE_ALIASES.get(ticker, ticker)
            df = prices.get(source_ticker)
            if df is None or df.empty:
                failures[ticker] = "empty"
                continue
            if source_ticker != ticker:
                df = df.copy()
                df["source_ticker"] = source_ticker
            write_frame(df, target_dir / f"{safe_symbol_name(ticker)}.parquet")
            written += 1
    write_status(
        "yfinance_daily",
        "complete",
        status_file=status_file,
        written=written,
        skipped=skipped,
        failures=failures,
    )
    return {"written": written, "skipped": skipped, "failures": failures}


def fetch_yfinance_actions(
    tickers: list[str],
    *,
    start: date,
    end: date,
    output_dir: Path,
    status_file: Path,
    refresh: bool,
) -> dict:
    target_dir = output_dir / "yfinance" / "actions"
    failures: dict[str, str] = {}
    written = 0
    skipped = 0
    for idx, ticker in enumerate(tickers, start=1):
        target = target_dir / f"{safe_symbol_name(ticker)}.parquet"
        if target.exists() and not refresh:
            skipped += 1
            continue
        write_status(
            "yfinance_actions",
            "running",
            status_file=status_file,
            done=idx - 1,
            total=len(tickers),
            ticker=ticker,
            written=written,
            skipped=skipped,
        )
        try:
            source_ticker = YFINANCE_ALIASES.get(ticker, ticker)
            actions = yf.Ticker(source_ticker).actions
        except Exception as exc:  # pragma: no cover - external service defensive
            failures[ticker] = repr(exc)
            continue
        if actions is None:
            actions = pd.DataFrame()
        if source_ticker != ticker:
            actions = actions.copy()
            actions["source_ticker"] = source_ticker
        if not actions.empty:
            actions = actions.copy()
            actions.index = pd.DatetimeIndex(actions.index).tz_localize(None)
            actions = actions[(actions.index.date >= start) & (actions.index.date <= end)]
        write_frame(actions, target)
        written += 1
    write_status(
        "yfinance_actions",
        "complete",
        status_file=status_file,
        done=len(tickers),
        total=len(tickers),
        written=written,
        skipped=skipped,
        failures=failures,
    )
    return {"written": written, "skipped": skipped, "failures": failures}


def fetch_fred_series(
    series_map: dict[str, str],
    *,
    start: date,
    end: date,
    output_dir: Path,
    status_file: Path,
    refresh: bool,
    timeout_s: float = 30.0,
) -> dict:
    target_dir = output_dir / "fred"
    failures: dict[str, str] = {}
    written = 0
    skipped = 0
    for idx, (series_id, description) in enumerate(series_map.items(), start=1):
        target = target_dir / f"{safe_symbol_name(series_id)}.parquet"
        if target.exists() and not refresh:
            skipped += 1
            continue
        write_status(
            "fred",
            "running",
            status_file=status_file,
            done=idx - 1,
            total=len(series_map),
            series_id=series_id,
            written=written,
            skipped=skipped,
        )
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        try:
            resp = requests.get(url, timeout=timeout_s)
            resp.raise_for_status()
            df = pd.read_csv(StringIO(resp.text))
        except Exception as exc:  # pragma: no cover - external service defensive
            failures[series_id] = repr(exc)
            continue
        date_col = "observation_date"
        if date_col not in df.columns or series_id not in df.columns:
            failures[series_id] = "unexpected csv shape"
            continue
        out = pd.DataFrame({
            "date": pd.to_datetime(df[date_col]),
            "value": pd.to_numeric(df[series_id].replace(".", pd.NA), errors="coerce"),
            "series_id": series_id,
            "description": description,
        })
        out = out[(out["date"].dt.date >= start) & (out["date"].dt.date <= end)]
        write_frame(out, target)
        written += 1
    write_status(
        "fred",
        "complete",
        status_file=status_file,
        done=len(series_map),
        total=len(series_map),
        written=written,
        skipped=skipped,
        failures=failures,
    )
    return {"written": written, "skipped": skipped, "failures": failures}


def write_manifest(output_dir: Path, summaries: dict) -> dict:
    files = sorted(p for p in output_dir.rglob("*") if p.is_file())
    manifest = {
        "generated_at_utc": utc_now_iso(),
        "output_dir": str(output_dir),
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "summaries": summaries,
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch non-Polygon offline context datasets.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--status-file", type=Path, default=DEFAULT_STATUS_FILE)
    p.add_argument("--start", type=lambda s: date.fromisoformat(s), default=START_DATE)
    p.add_argument("--end", type=lambda s: date.fromisoformat(s), default=END_DATE)
    p.add_argument("--chunk-size", type=int, default=15)
    p.add_argument("--max-tickers", type=int, default=None)
    p.add_argument("--refresh", action="store_true")
    p.add_argument("--skip-yfinance", action="store_true")
    p.add_argument("--skip-actions", action="store_true")
    p.add_argument("--skip-fred", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    started = time.monotonic()
    tickers = default_tickers()
    if args.max_tickers is not None:
        tickers = tickers[: args.max_tickers]
    summaries: dict[str, dict] = {}

    write_status(
        "init",
        "running",
        status_file=args.status_file,
        output_dir=str(args.output_dir),
        start=args.start.isoformat(),
        end=args.end.isoformat(),
        tickers=len(tickers),
    )

    if not args.skip_yfinance:
        summaries["yfinance_daily"] = fetch_yfinance_daily(
            tickers,
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            status_file=args.status_file,
            chunk_size=args.chunk_size,
            refresh=args.refresh,
        )
    if not args.skip_actions:
        summaries["yfinance_actions"] = fetch_yfinance_actions(
            tickers,
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            status_file=args.status_file,
            refresh=args.refresh,
        )
    if not args.skip_fred:
        summaries["fred"] = fetch_fred_series(
            FRED_SERIES,
            start=args.start,
            end=args.end,
            output_dir=args.output_dir,
            status_file=args.status_file,
            refresh=args.refresh,
        )

    manifest = write_manifest(args.output_dir, summaries)
    write_status(
        "complete",
        "complete",
        status_file=args.status_file,
        elapsed_seconds=int(time.monotonic() - started),
        file_count=manifest["file_count"],
        total_bytes=manifest["total_bytes"],
    )
    print("offline context fetch: COMPLETE")
    print(f"  output_dir: {args.output_dir}")
    print(f"  files: {manifest['file_count']}")
    print(f"  bytes: {manifest['total_bytes']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
