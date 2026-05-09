"""Read-only readiness checker for the paper_v1 priced chain cache.

This script performs no Polygon/network calls and writes no files. It checks
whether the priced chain cache is ready for ``scripts/run_paper_v1_evidence.py``
in strict mode:

* Year 1 and Year 2 earnings events have matching ``chains_priced`` parquets.
* Priced parquets include the columns the evidence runner requires.
* Priced parquets have finite entry and exit ``mid`` values plus finite
  ``underlying_close`` and ``atm_iv``.
* SPY core priced files are counted and reported separately.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from signals.paper_v1_universe import PAPER_V1_SATELLITE  # noqa: E402

DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"

YEARS = ("year1", "year2")
YEAR_WINDOWS: dict[str, tuple[date, date]] = {
    "year1": (date(2024, 5, 7), date(2025, 5, 6)),
    "year2": (date(2025, 5, 7), date(2026, 5, 6)),
}

REQUIRED_PRICED_COLUMNS = (
    "bid",
    "ask",
    "mid",
    "underlying_close",
    "atm_iv",
    "timing",
    "is_atm",
)


@dataclass(frozen=True)
class Finding:
    level: str
    code: str
    message: str
    year: str | None = None
    ticker: str | None = None
    event_date: str | None = None
    path: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


def _year_dir(cache_dir: Path, year: str) -> Path:
    if year not in YEAR_WINDOWS:
        raise ValueError(f"unknown year {year!r}")
    return Path(cache_dir) / year


def _priced_path(cache_dir: Path, year: str, ticker: str, event_date: date) -> Path:
    return (
        _year_dir(cache_dir, year)
        / "chains_priced"
        / ticker
        / f"{event_date.isoformat()}.parquet"
    )


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return pd.Timestamp(value).date()
    except (TypeError, ValueError):
        return None


def _read_earnings_dates(path: Path) -> tuple[list[date], Finding | None]:
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return [], Finding(
            level="blocker",
            code="unreadable_earnings_file",
            message=f"cannot read earnings_dates file: {exc}",
            path=str(path),
        )

    if "earnings_date" not in df.columns:
        return [], Finding(
            level="blocker",
            code="missing_earnings_date_column",
            message="earnings_dates parquet is missing earnings_date column",
            path=str(path),
        )

    dates = sorted({d for d in (_parse_date(v) for v in df["earnings_date"]) if d})
    return dates, None


def _collect_earnings_dates(
    cache_dir: Path,
    year: str,
    universe: Iterable[str],
) -> tuple[dict[str, list[date]], list[Finding]]:
    earnings_dir = _year_dir(cache_dir, year) / "earnings_dates"
    findings: list[Finding] = []
    events: dict[str, list[date]] = {}

    if not earnings_dir.exists():
        findings.append(
            Finding(
                level="blocker",
                code="missing_earnings_dir",
                message="missing earnings_dates directory",
                year=year,
                path=str(earnings_dir),
            )
        )
        return events, findings

    missing_files = 0
    for ticker in universe:
        path = earnings_dir / f"{ticker}.parquet"
        if not path.exists():
            missing_files += 1
            findings.append(
                Finding(
                    level="blocker",
                    code="missing_earnings_file",
                    message=f"missing earnings_dates file for {ticker}",
                    year=year,
                    ticker=ticker,
                    path=str(path),
                )
            )
            continue

        dates, finding = _read_earnings_dates(path)
        if finding is not None:
            findings.append(
                Finding(
                    level=finding.level,
                    code=finding.code,
                    message=finding.message,
                    year=year,
                    ticker=ticker,
                    path=finding.path,
                )
            )
            continue
        events[ticker] = dates

    if not any(events.values()) and missing_files == 0:
        findings.append(
            Finding(
                level="blocker",
                code="no_satellite_earnings_events",
                message="no valid satellite earnings events found",
                year=year,
                path=str(earnings_dir),
            )
        )

    return events, findings


def _finite_count(series: pd.Series) -> int:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    return int(np.isfinite(values).sum())


def _timing_slice(df: pd.DataFrame, timing: str) -> pd.DataFrame:
    if "timing" not in df.columns:
        return pd.DataFrame()
    return df[df["timing"] == timing]


def _atm_slice(df: pd.DataFrame) -> pd.DataFrame:
    if "is_atm" not in df.columns or df.empty:
        return pd.DataFrame()
    return df[df["is_atm"].astype(bool)]


def _validate_priced_parquet(path: Path, *, year: str, ticker: str) -> list[Finding]:
    findings: list[Finding] = []
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        return [
            Finding(
                level="blocker",
                code="unreadable_priced_file",
                message=f"cannot read priced parquet: {exc}",
                year=year,
                ticker=ticker,
                event_date=path.stem,
                path=str(path),
            )
        ]

    if df.empty:
        return [
            Finding(
                level="blocker",
                code="empty_priced_file",
                message="priced parquet has no rows",
                year=year,
                ticker=ticker,
                event_date=path.stem,
                path=str(path),
            )
        ]

    missing = [col for col in REQUIRED_PRICED_COLUMNS if col not in df.columns]
    if missing:
        findings.append(
            Finding(
                level="blocker",
                code="missing_priced_columns",
                message="priced parquet missing required columns: " + ", ".join(missing),
                year=year,
                ticker=ticker,
                event_date=path.stem,
                path=str(path),
            )
        )
        return findings

    entry = _timing_slice(df, "entry")
    exit_rows = _timing_slice(df, "exit")
    if entry.empty:
        findings.append(
            Finding(
                level="blocker",
                code="missing_entry_rows",
                message="priced parquet has no entry timing rows",
                year=year,
                ticker=ticker,
                event_date=path.stem,
                path=str(path),
            )
        )
    if exit_rows.empty:
        findings.append(
            Finding(
                level="blocker",
                code="missing_exit_rows",
                message="priced parquet has no exit timing rows",
                year=year,
                ticker=ticker,
                event_date=path.stem,
                path=str(path),
            )
        )

    if entry.empty or _finite_count(entry["mid"]) == 0:
        findings.append(
            Finding(
                level="blocker",
                code="no_finite_entry_mid",
                message="priced parquet has no finite entry mid values",
                year=year,
                ticker=ticker,
                event_date=path.stem,
                path=str(path),
            )
        )

    if exit_rows.empty or _finite_count(exit_rows["mid"]) == 0:
        findings.append(
            Finding(
                level="blocker",
                code="no_finite_exit_mid",
                message="priced parquet has no finite exit mid values",
                year=year,
                ticker=ticker,
                event_date=path.stem,
                path=str(path),
            )
        )

    if _finite_count(df["underlying_close"]) == 0:
        findings.append(
            Finding(
                level="blocker",
                code="no_finite_underlying_close",
                message="priced parquet has no finite underlying_close values",
                year=year,
                ticker=ticker,
                event_date=path.stem,
                path=str(path),
            )
        )

    atm_entry = _atm_slice(entry)
    atm_exit = _atm_slice(exit_rows)
    if atm_entry.empty or atm_exit.empty:
        findings.append(
            Finding(
                level="blocker",
                code="missing_atm_rows",
                message="priced parquet needs ATM rows at entry and exit",
                year=year,
                ticker=ticker,
                event_date=path.stem,
                path=str(path),
            )
        )

    if atm_entry.empty or _finite_count(atm_entry["atm_iv"]) == 0:
        findings.append(
            Finding(
                level="blocker",
                code="no_finite_atm_iv",
                message="priced parquet has no finite ATM entry atm_iv values",
                year=year,
                ticker=ticker,
                event_date=path.stem,
                path=str(path),
            )
        )

    return findings


def _spy_core_entry_dates(year: str) -> list[date]:
    start, end = YEAR_WINDOWS[year]
    return [
        ts.date()
        for ts in pd.bdate_range(start, end)
        if ts.date().weekday() == 0
    ]


def _scan_priced_files(cache_dir: Path, year: str) -> list[Path]:
    priced_dir = _year_dir(cache_dir, year) / "chains_priced"
    if not priced_dir.exists():
        return []
    return sorted(priced_dir.glob("*/*.parquet"))


def _summarize_spy_core(
    cache_dir: Path,
    year: str,
    *,
    require_spy_core: bool,
) -> tuple[dict[str, Any], list[Finding]]:
    spy_dir = _year_dir(cache_dir, year) / "chains_priced" / "SPY"
    files = sorted(spy_dir.glob("*.parquet")) if spy_dir.exists() else []
    expected_dates = _spy_core_entry_dates(year)
    expected_names = {f"{d.isoformat()}.parquet" for d in expected_dates}
    present_names = {p.name for p in files}
    missing_dates = sorted(
        d.isoformat()
        for d in expected_dates
        if f"{d.isoformat()}.parquet" not in present_names
    )

    findings: list[Finding] = []
    level = "blocker" if require_spy_core else "warning"
    if not files:
        findings.append(
            Finding(
                level=level,
                code="no_spy_priced_core",
                message="no SPY priced core files found",
                year=year,
                ticker="SPY",
                path=str(spy_dir),
            )
        )
    elif missing_dates:
        findings.append(
            Finding(
                level=level,
                code="partial_spy_priced_core",
                message=(
                    f"SPY priced core is partial: {len(files)}/"
                    f"{len(expected_dates)} expected Monday files present"
                ),
                year=year,
                ticker="SPY",
                path=str(spy_dir),
            )
        )

    for path in files:
        findings.extend(_validate_priced_parquet(path, year=year, ticker="SPY"))

    return {
        "present": bool(files),
        "priced_files": int(len(files)),
        "expected_monday_files": int(len(expected_dates)),
        "missing_expected_monday_files": int(len(missing_dates)),
        "missing_expected_monday_dates": missing_dates,
        "extra_files": int(len(present_names - expected_names)),
    }, findings


def _summarize_year(
    cache_dir: Path,
    year: str,
    universe: Iterable[str],
    *,
    require_spy_core: bool,
) -> tuple[dict[str, Any], list[Finding]]:
    universe = tuple(universe)
    findings: list[Finding] = []
    year_dir = _year_dir(cache_dir, year)
    priced_dir = year_dir / "chains_priced"

    if not year_dir.exists():
        findings.append(
            Finding(
                level="blocker",
                code="missing_year_dir",
                message="missing year cache directory",
                year=year,
                path=str(year_dir),
            )
        )

    if not priced_dir.exists():
        findings.append(
            Finding(
                level="blocker",
                code="missing_chains_priced_dir",
                message="missing chains_priced directory",
                year=year,
                path=str(priced_dir),
            )
        )

    earnings_by_ticker, earnings_findings = _collect_earnings_dates(
        cache_dir, year, universe
    )
    findings.extend(earnings_findings)

    expected_satellite: set[tuple[str, date]] = {
        (ticker, event_date)
        for ticker, dates in earnings_by_ticker.items()
        for event_date in dates
    }
    present_satellite: set[tuple[str, date]] = set()
    missing_satellite: list[dict[str, str]] = []

    for ticker, event_date in sorted(expected_satellite):
        path = _priced_path(cache_dir, year, ticker, event_date)
        if path.exists():
            present_satellite.add((ticker, event_date))
            findings.extend(_validate_priced_parquet(path, year=year, ticker=ticker))
        else:
            missing_satellite.append(
                {
                    "ticker": ticker,
                    "event_date": event_date.isoformat(),
                    "path": str(path),
                }
            )
            findings.append(
                Finding(
                    level="blocker",
                    code="missing_priced_file",
                    message=(
                        f"missing priced chain for {ticker} "
                        f"{event_date.isoformat()}"
                    ),
                    year=year,
                    ticker=ticker,
                    event_date=event_date.isoformat(),
                    path=str(path),
                )
            )

    all_priced_files = _scan_priced_files(cache_dir, year)
    for path in all_priced_files:
        ticker = path.parent.name
        if ticker == "SPY":
            continue
        parsed_date = _parse_date(path.stem)
        if (ticker, parsed_date) in present_satellite:
            continue
        findings.extend(_validate_priced_parquet(path, year=year, ticker=ticker))
        if ticker in universe:
            findings.append(
                Finding(
                    level="blocker",
                    code="extra_satellite_priced_file",
                    message=(
                        "priced satellite file is not listed in earnings_dates; "
                        "official sample must not be padded by orphan chains"
                    ),
                    year=year,
                    ticker=ticker,
                    event_date=path.stem,
                    path=str(path),
                )
            )

    spy_core, spy_findings = _summarize_spy_core(
        cache_dir, year, require_spy_core=require_spy_core
    )
    findings.extend(spy_findings)

    priced_count = len(all_priced_files)
    satellite_event_count = len(expected_satellite)
    satellite_present_count = len(present_satellite)

    if priced_count == 0:
        findings.append(
            Finding(
                level="blocker",
                code="no_priced_chain_files",
                message="no chains_priced parquet files found for year",
                year=year,
                path=str(priced_dir),
            )
        )

    return {
        "year": year,
        "window": [d.isoformat() for d in YEAR_WINDOWS[year]],
        "chains_priced_dir": str(priced_dir),
        "chains_priced_dir_present": priced_dir.exists(),
        "priced_file_count": int(priced_count),
        "satellite_tickers": int(len(universe)),
        "satellite_earnings_tickers_with_files": int(len(earnings_by_ticker)),
        "satellite_event_count": int(satellite_event_count),
        "satellite_priced_file_count": int(satellite_present_count),
        "satellite_missing_priced_count": int(len(missing_satellite)),
        "satellite_missing_priced": missing_satellite,
        "spy_core": spy_core,
    }, findings


def check_readiness(
    cache_dir: Path = DEFAULT_CACHE_DIR,
    *,
    universe: Iterable[str] = PAPER_V1_SATELLITE,
    require_spy_core: bool = False,
) -> dict[str, Any]:
    """Return a machine-readable readiness summary without side effects."""
    cache_dir = Path(cache_dir)
    year_summaries: dict[str, dict[str, Any]] = {}
    findings: list[Finding] = []

    for year in YEARS:
        summary, year_findings = _summarize_year(
            cache_dir,
            year,
            universe,
            require_spy_core=require_spy_core,
        )
        year_summaries[year] = summary
        findings.extend(year_findings)

    blockers = [f for f in findings if f.level == "blocker"]
    warnings = [f for f in findings if f.level == "warning"]
    return {
        "ready": len(blockers) == 0,
        "cache_dir": str(cache_dir),
        "blocker_count": int(len(blockers)),
        "warning_count": int(len(warnings)),
        "years": year_summaries,
        "blockers": [f.to_json() for f in blockers],
        "warnings": [f.to_json() for f in warnings],
    }


def _format_findings(
    label: str,
    findings: list[dict[str, Any]],
    *,
    max_findings: int,
) -> list[str]:
    lines: list[str] = [f"{label}: {len(findings)}"]
    for item in findings[:max_findings]:
        prefix = item["level"].upper()
        year = f" {item['year']}" if "year" in item else ""
        ticker = f" {item['ticker']}" if "ticker" in item else ""
        event_date = f" {item['event_date']}" if "event_date" in item else ""
        lines.append(
            f"  {prefix}{year}{ticker}{event_date} {item['code']}: "
            f"{item['message']}"
        )
    remaining = len(findings) - max_findings
    if remaining > 0:
        lines.append(f"  ... {remaining} more")
    return lines


def format_text_summary(summary: dict[str, Any], *, max_findings: int = 20) -> str:
    status = "READY" if summary["ready"] else "BLOCKED"
    if summary["ready"] and summary["warning_count"]:
        status = "READY with warnings"

    lines = [
        f"paper_v1 readiness: {status}",
        f"cache_dir: {summary['cache_dir']}",
    ]
    for year in YEARS:
        y = summary["years"][year]
        spy = y["spy_core"]
        lines.append(
            f"{year}: satellite priced {y['satellite_priced_file_count']}/"
            f"{y['satellite_event_count']}; "
            f"chains_priced files={y['priced_file_count']}; "
            f"SPY core files={spy['priced_files']}/"
            f"{spy['expected_monday_files']}"
        )

    lines.extend(
        _format_findings(
            "blockers",
            summary["blockers"],
            max_findings=max_findings,
        )
    )
    lines.extend(
        _format_findings(
            "warnings",
            summary["warnings"],
            max_findings=max_findings,
        )
    )
    return "\n".join(lines)


def _parse_universe(value: str | None) -> tuple[str, ...]:
    if not value:
        return PAPER_V1_SATELLITE
    tickers = tuple(t.strip().upper() for t in value.split(",") if t.strip())
    if not tickers:
        raise argparse.ArgumentTypeError("universe must include at least one ticker")
    return tickers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Read-only paper_v1 priced-cache readiness checker. "
            "No network calls; no files written."
        )
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache root to inspect (default: {DEFAULT_CACHE_DIR}).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of concise text.",
    )
    p.add_argument(
        "--require-spy-core",
        action="store_true",
        help="Promote missing/partial SPY core priced files from warning to blocker.",
    )
    p.add_argument(
        "--universe",
        default=None,
        help=(
            "Comma-separated satellite tickers to check "
            "(default: frozen paper_v1 universe)."
        ),
    )
    p.add_argument(
        "--max-findings",
        type=int,
        default=20,
        help="Maximum findings to print per level in text mode (default: 20).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        universe = _parse_universe(args.universe)
    except argparse.ArgumentTypeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = check_readiness(
        args.cache_dir,
        universe=universe,
        require_spy_core=bool(args.require_spy_core),
    )
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_text_summary(summary, max_findings=max(0, args.max_findings)))
    return 0 if summary["ready"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
