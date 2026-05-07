"""Cache-first data fetcher for paper_v1.

Pulls Polygon Free data for both Year 1 (training) and Year 2 (held-out OOS)
to local Parquet. Run ONCE (probably overnight at 5 rpm — ~42 hrs wall).
Idempotent: re-running skips files already on disk. Resumable: hard interrupt
mid-run resumes cleanly.

Year 1 / Year 2 windows match metadata/paper_v1_prereg.md § 13 (subject to
CEO confirmation at tag time). Constants are hardcoded here as the local
source-of-truth so this script never has to parse the prereg.

Cache layout under ``data/cache/`` mirrors the year split required by the
held-out evidence rule (no peeking at Year 2 during training):

    data/cache/
      year1/                           # 2024-05-07 -> 2025-05-06
        spy_daily.parquet              # SPY daily OHLCV
        spy_iv_daily.parquet           # SPY ATM IV per trading day
        earnings_dates/{TICKER}.parquet
        chains/{TICKER}/{YYYY-MM-DD}.parquet
      year2/                           # 2025-05-07 -> 2026-05-06
        ...same shape...
      manifest.json                    # what's cached, when, sizes

Status (one JSON file at ``data/paper_v1_status.json``) is rewritten
atomically after every step so the dashboard agent can render live progress
during the multi-day pull. Schema:

    {
        "started_at_utc": "...",
        "last_update_utc": "...",
        "phase": "running" | "cache_complete" | "error",
        "stages": {
            "fetch_year1_spy_daily": {"status": "complete", "rows": ...},
            "fetch_year1_earnings": {"status": "complete", "events": ...},
            "fetch_year1_chains":   {"status": "running", "done": ..., "total": ...},
            "fetch_year1_iv_daily": {"status": "complete", "days": ...},
            ...same five stages with year2_ prefix...
        }
    }

Idempotency / resumability discipline:
    * SPY daily: a single parquet per year. Existing file -> load + return; no
      API call. (One bar per day; refetching wastes a precious request.)
    * Earnings dates: one parquet per ticker via the existing
      ``ingest.earnings_calendar`` cache. ``fetch_calendar`` already round-trips
      cache hits without calling yfinance. We just point it at our year-scoped
      directory.
    * Per-event chains: ``chains/{TICKER}/{YYYY-MM-DD}.parquet`` per event. If
      the file exists we skip the underlying chain fetch. This is the natural
      shard size for a 5 rpm budget — losing one event re-pulls one event.
    * SPY IV daily: row-level resumable. We append, never rewrite. On restart we
      load the existing parquet, take ``max(date)``, and only fetch later days.
    * Every parquet write goes to ``<target>.tmp`` and then ``rename(target)``
      so a kill -9 in the middle never leaves a half-written file.
    * Status JSON also writes via ``.tmp`` + rename for the same reason.

CLI:
    python scripts/fetch_paper_v1_data.py [--year year1|year2|both]
                                          [--skip-iv] [--skip-chains]
                                          [--max-events-per-ticker N]
                                          [--cache-dir PATH]

Tests in ``tests/test_fetch_paper_v1_data.py`` mock PolygonFreeClient and
yfinance — no real network calls.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ingest.earnings_calendar import fetch_calendar  # noqa: E402
from ingest.polygon_free import (  # noqa: E402
    PolygonAuthError,
    PolygonError,
    PolygonFreeClient,
)
from signals.paper_v1_universe import (  # noqa: E402
    PAPER_V1_SATELLITE,
)

logger = logging.getLogger("fetch_paper_v1_data")

DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"
STATUS_FILE = REPO_ROOT / "data" / "paper_v1_status.json"
MANIFEST_FILENAME = "manifest.json"

# Pinned per metadata/paper_v1_prereg.md § 13 (proposed; CEO confirms at tag).
YEAR1_START = date(2024, 5, 7)
YEAR1_END = date(2025, 5, 6)
YEAR2_START = date(2025, 5, 7)
YEAR2_END = date(2026, 5, 6)

# Stage names persisted in STATUS_FILE — keep stable for the dashboard agent.
STAGE_SPY_DAILY = "spy_daily"
STAGE_EARNINGS = "earnings"
STAGE_CHAINS = "chains"
STAGE_IV_DAILY = "iv_daily"


# ---------------------------------------------------------------------------
# Atomic IO helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    """Write payload to <target>.tmp then rename(target). Caller ensures parent dir exists."""
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    """Write a parquet file atomically: <target>.tmp -> rename(target).

    pandas' to_parquet doesn't expose a direct stream, so we write the temp
    file by hand and rename. parent dirs are created by the caller.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_parquet(tmp, index=True if isinstance(df.index, pd.DatetimeIndex) else False)
    os.replace(tmp, target)


def _atomic_write_json(target: Path, obj: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True, default=str).encode("utf-8")
    _atomic_write_bytes(target, payload)


def _read_json(target: Path) -> dict:
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text())
    except (OSError, ValueError):
        # Corrupt status from prior crash; start fresh rather than abort the run.
        logger.warning("status file %s unreadable; starting fresh", target)
        return {}


# ---------------------------------------------------------------------------
# Status reporting
# ---------------------------------------------------------------------------


def write_status(
    stage: str,
    status: str,
    *,
    status_file: Path = STATUS_FILE,
    phase: str | None = None,
    **fields,
) -> dict:
    """Merge a stage update into STATUS_FILE atomically.

    Existing stages are preserved; only the named ``stage`` is replaced with
    a fresh ``{"status": status, **fields}`` block. ``phase`` overrides the
    top-level phase (e.g. ``"running"`` -> ``"cache_complete"``).

    Returns the merged document for tests / inspection.
    """
    existing = _read_json(status_file)
    stages = existing.get("stages") or {}
    stage_entry = {"status": status, "updated_at_utc": _utc_now_iso()}
    stage_entry.update(fields)
    stages[stage] = stage_entry

    doc = {
        "started_at_utc": existing.get("started_at_utc") or _utc_now_iso(),
        "last_update_utc": _utc_now_iso(),
        "phase": phase if phase is not None else existing.get("phase", "running"),
        "stages": stages,
    }
    _atomic_write_json(status_file, doc)
    return doc


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------


def cache_path_year1(rel: str, base_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Return ``base_dir / 'year1' / rel`` without creating directories."""
    return Path(base_dir) / "year1" / rel


def cache_path_year2(rel: str, base_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Return ``base_dir / 'year2' / rel`` without creating directories."""
    return Path(base_dir) / "year2" / rel


def _year_dir(year: str, base_dir: Path) -> Path:
    if year == "year1":
        return Path(base_dir) / "year1"
    if year == "year2":
        return Path(base_dir) / "year2"
    raise ValueError(f"unknown year {year!r}")


def _year_window(year: str) -> tuple[date, date]:
    if year == "year1":
        return YEAR1_START, YEAR1_END
    if year == "year2":
        return YEAR2_START, YEAR2_END
    raise ValueError(f"unknown year {year!r}")


# ---------------------------------------------------------------------------
# Fetchers (each idempotent, each persists before returning)
# ---------------------------------------------------------------------------


def fetch_spy_daily(
    client: PolygonFreeClient,
    year_start: date,
    year_end: date,
    target: Path,
) -> pd.DataFrame:
    """Idempotent: if target exists, load and return; else fetch, save, return.

    SPY daily OHLCV across the year. One Polygon request total.
    """
    if target.exists():
        logger.debug("spy_daily cache hit at %s", target)
        return pd.read_parquet(target)

    df = client.fetch_aggregates(
        ticker="SPY",
        from_date=year_start.isoformat(),
        to_date=year_end.isoformat(),
        timespan="day",
        multiplier=1,
        adjusted=True,
    )
    _atomic_write_parquet(df, target)
    return df


def fetch_earnings_dates_for_universe(
    tickers: Iterable[str],
    year_start: date,
    year_end: date,
    target_dir: Path,
) -> dict[str, list[date]]:
    """Wrapper around ``ingest.earnings_calendar.fetch_calendar``.

    The earnings_calendar module already implements per-ticker cache files at
    ``target_dir/{ticker}.parquet``; calling it with ``use_cache=True`` makes
    the call idempotent on a per-ticker basis.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    return fetch_calendar(
        list(tickers),
        start=year_start,
        end=year_end,
        base_dir=target_dir,
        use_cache=True,
    )


def _trading_day_on_or_before(d: date) -> date:
    """Return the most recent weekday on-or-before ``d``.

    Used to derive a T-1 close anchor for chain snapshots — Polygon's chain
    endpoint is queryable with ``as_of`` on a real trading day. Holidays will
    still slip through (we don't bake an exchange calendar in here) but the
    fetched chain handles them: an empty result means the file is skipped.
    """
    cur = d
    while cur.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        cur -= timedelta(days=1)
    return cur


def fetch_chain_at_event(
    client: PolygonFreeClient,
    underlying: str,
    event_date: date,
    target_dir: Path,
) -> pd.DataFrame | None:
    """Fetch options chain at T-1 close for a given earnings event.

    File layout: ``target_dir/{underlying}/{event_date}.parquet``.

    Idempotent — existing file -> load + return without API call.
    Returns ``None`` (and writes no file) when Polygon returns no contracts:
    a small/illiquid name on a particular date legitimately has no listed
    chain. Caller should treat ``None`` as "skip this event" rather than fail.
    """
    target_dir = Path(target_dir) / underlying
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{event_date.isoformat()}.parquet"
    if target.exists():
        logger.debug("chain cache hit %s %s", underlying, event_date)
        return pd.read_parquet(target)

    as_of = _trading_day_on_or_before(event_date - timedelta(days=1))
    try:
        df = client.fetch_options_contracts(
            underlying_ticker=underlying,
            as_of=as_of.isoformat(),
            limit=1000,
        )
    except PolygonError as exc:
        logger.warning("chain fetch failed %s %s: %s", underlying, event_date, exc)
        return None

    if df is None or df.empty:
        logger.info("no chain for %s as_of %s; skipping", underlying, as_of)
        return None

    # Stamp the as_of date on the cached file so downstream code knows the snap.
    df = df.copy()
    df["as_of_date"] = as_of.isoformat()
    df["event_date"] = event_date.isoformat()
    _atomic_write_parquet(df, target)
    return df


def fetch_spy_iv_daily(
    client: PolygonFreeClient,
    year_start: date,
    year_end: date,
    target: Path,
) -> pd.DataFrame:
    """Build a daily SPY ATM IV history for ``[year_start, year_end]``.

    For each trading day we'd compute IV from an ATM straddle on the closest
    listed expiry; in practice (free tier) we approximate: fetch SPY contracts
    as_of each trading day at limit=1 just to confirm chain availability, then
    record an IV proxy from the previously fetched daily close. The exact IV
    derivation is downstream — this script's job is just to populate one row
    per trading day so the IV history is complete and resumable.

    Resumability:
        * If ``target`` already has rows up to date D, only fetch days strictly
          after D.
        * Each day's row is appended atomically: load -> append -> _atomic_write.
        * A kill -9 between days loses at most the in-flight day's bookkeeping
          (which is a no-op API call, repeated harmlessly on resume).
    """
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        existing = pd.read_parquet(target)
    else:
        existing = pd.DataFrame(columns=["date", "atm_iv"])

    if not existing.empty and "date" in existing.columns:
        try:
            already = pd.to_datetime(existing["date"]).dt.date.tolist()
            cursor = max(already)
            start_from = cursor + timedelta(days=1)
        except (TypeError, ValueError):
            start_from = year_start
    else:
        start_from = year_start

    if start_from > year_end:
        return existing

    # Trading-day calendar via pandas business-day range; weekends only —
    # holidays will produce zero-result Polygon responses which we record as
    # NaN to keep the row index dense for the dashboard.
    days = pd.bdate_range(start_from, year_end)
    if len(days) == 0:
        return existing

    rows = existing.to_dict(orient="records")
    for ts in days:
        d: date = ts.date()
        try:
            df = client.fetch_options_contracts(
                underlying_ticker="SPY",
                as_of=d.isoformat(),
                limit=1,
            )
            available = not (df is None or df.empty)
        except PolygonError as exc:
            logger.warning("iv probe failed for %s: %s", d, exc)
            available = False

        # IV computation is downstream; record a placeholder so a restart can
        # resume from the next day. NaN signals "fetch but no IV computed yet"
        # rather than "fetch missing".
        rows.append({"date": d.isoformat(), "atm_iv": float("nan"), "available": bool(available)})

        # Persist every row immediately for true row-level resumability.
        df_out = pd.DataFrame(rows)
        _atomic_write_parquet(df_out, target)

    return pd.read_parquet(target)


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def _manifest_entry(path: Path) -> dict:
    if not path.exists():
        return {"present": False}
    stat = path.stat()
    return {
        "present": True,
        "size_bytes": int(stat.st_size),
        "mtime_utc": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(timespec="seconds"),
    }


def write_manifest(base_dir: Path) -> dict:
    """Walk ``base_dir`` and emit a manifest of cached files (size + mtime)."""
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    target = base_dir / MANIFEST_FILENAME
    items: dict[str, dict] = {}
    for path in sorted(base_dir.rglob("*.parquet")):
        rel = path.relative_to(base_dir).as_posix()
        items[rel] = _manifest_entry(path)
    doc = {
        "generated_at_utc": _utc_now_iso(),
        "base_dir": str(base_dir),
        "files": items,
        "count": len(items),
    }
    _atomic_write_json(target, doc)
    return doc


# ---------------------------------------------------------------------------
# Year orchestration
# ---------------------------------------------------------------------------


def _stage_key(year: str, kind: str) -> str:
    """e.g. ``year1`` + ``spy_daily`` -> ``fetch_year1_spy_daily``."""
    return f"fetch_{year}_{kind}"


def _fetch_year(
    *,
    year: str,
    client: PolygonFreeClient,
    base_dir: Path,
    skip_iv: bool,
    skip_chains: bool,
    max_events_per_ticker: int | None,
    status_file: Path,
) -> dict:
    """Run the full pipeline for one year, persisting status after each stage."""
    year_start, year_end = _year_window(year)
    yd = _year_dir(year, base_dir)
    yd.mkdir(parents=True, exist_ok=True)

    summary: dict = {"year": year, "window": [year_start.isoformat(), year_end.isoformat()]}

    # 1) SPY daily ----------------------------------------------------------
    spy_target = yd / "spy_daily.parquet"
    write_status(_stage_key(year, STAGE_SPY_DAILY), "running", status_file=status_file)
    spy_df = fetch_spy_daily(client, year_start, year_end, spy_target)
    write_status(
        _stage_key(year, STAGE_SPY_DAILY),
        "complete",
        status_file=status_file,
        rows=int(len(spy_df)),
        path=str(spy_target),
    )
    summary["spy_daily_rows"] = int(len(spy_df))

    # 2) Earnings dates -----------------------------------------------------
    earnings_dir = yd / "earnings_dates"
    write_status(_stage_key(year, STAGE_EARNINGS), "running", status_file=status_file)
    universe = list(PAPER_V1_SATELLITE)
    calendar = fetch_earnings_dates_for_universe(
        universe, year_start, year_end, earnings_dir
    )
    total_events = sum(len(v) for v in calendar.values())
    write_status(
        _stage_key(year, STAGE_EARNINGS),
        "complete",
        status_file=status_file,
        tickers=len(calendar),
        events=int(total_events),
        path=str(earnings_dir),
    )
    summary["earnings_events"] = int(total_events)
    summary["earnings_tickers"] = len(calendar)

    # 3) Per-event chains ---------------------------------------------------
    chains_dir = yd / "chains"
    if skip_chains:
        write_status(
            _stage_key(year, STAGE_CHAINS),
            "skipped",
            status_file=status_file,
            reason="--skip-chains",
        )
        summary["chains_files"] = 0
    else:
        # Build the work list, applying the per-ticker cap if requested.
        work: list[tuple[str, date]] = []
        for ticker, dates in calendar.items():
            ev_dates = sorted(dates)
            if max_events_per_ticker is not None:
                ev_dates = ev_dates[:max_events_per_ticker]
            for ed in ev_dates:
                work.append((ticker, ed))

        write_status(
            _stage_key(year, STAGE_CHAINS),
            "running",
            status_file=status_file,
            done=0,
            total=len(work),
        )
        done = 0
        files = 0
        for ticker, event_date in work:
            df = fetch_chain_at_event(client, ticker, event_date, chains_dir)
            done += 1
            if df is not None:
                files += 1
            # Status update every 5 events to reduce IO at 5 rpm cadence.
            if done % 5 == 0 or done == len(work):
                write_status(
                    _stage_key(year, STAGE_CHAINS),
                    "running",
                    status_file=status_file,
                    done=done,
                    total=len(work),
                    files_written=files,
                )
        write_status(
            _stage_key(year, STAGE_CHAINS),
            "complete",
            status_file=status_file,
            done=done,
            total=len(work),
            files_written=files,
            path=str(chains_dir),
        )
        summary["chains_files"] = files
        summary["chains_attempts"] = done

    # 4) SPY IV daily -------------------------------------------------------
    iv_target = yd / "spy_iv_daily.parquet"
    if skip_iv:
        write_status(
            _stage_key(year, STAGE_IV_DAILY),
            "skipped",
            status_file=status_file,
            reason="--skip-iv",
        )
        summary["iv_daily_rows"] = 0
    else:
        write_status(_stage_key(year, STAGE_IV_DAILY), "running", status_file=status_file)
        iv_df = fetch_spy_iv_daily(client, year_start, year_end, iv_target)
        write_status(
            _stage_key(year, STAGE_IV_DAILY),
            "complete",
            status_file=status_file,
            days=int(len(iv_df)),
            path=str(iv_target),
        )
        summary["iv_daily_rows"] = int(len(iv_df))

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Cache-first Polygon Free fetcher for paper_v1. Run once; idempotent. "
            "Pulls Year 1 (training) and Year 2 (held-out OOS) into data/cache/. "
            "Throttled to 5 rpm; expect ~42 hours on a cold cache."
        ),
    )
    p.add_argument(
        "--year",
        choices=["year1", "year2", "both"],
        default="both",
        help="Which window to fetch (default: both).",
    )
    p.add_argument(
        "--skip-iv",
        action="store_true",
        help="Skip the slow daily IV history fetch (the ~12,500-call bottleneck).",
    )
    p.add_argument(
        "--skip-chains",
        action="store_true",
        help="Skip per-event options chains (still pulls SPY daily + earnings dates).",
    )
    p.add_argument(
        "--max-events-per-ticker",
        type=int,
        default=None,
        help="Cap events per ticker for debugging (default: no cap).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache root (default: {DEFAULT_CACHE_DIR}).",
    )
    p.add_argument(
        "--status-file",
        type=Path,
        default=STATUS_FILE,
        help=f"Status JSON path (default: {STATUS_FILE}).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)

    started = time.monotonic()
    write_status(
        "init",
        "running",
        status_file=args.status_file,
        phase="running",
        cache_dir=str(args.cache_dir),
        year_choice=args.year,
        skip_iv=bool(args.skip_iv),
        skip_chains=bool(args.skip_chains),
    )

    try:
        client = PolygonFreeClient()
    except PolygonAuthError as exc:
        logger.error("polygon auth failure: %s", exc)
        write_status(
            "init",
            "error",
            status_file=args.status_file,
            phase="error",
            error=str(exc),
        )
        return 2

    years_to_run = ["year1", "year2"] if args.year == "both" else [args.year]

    summaries: list[dict] = []
    try:
        for year in years_to_run:
            logger.info("=== fetching %s ===", year)
            summary = _fetch_year(
                year=year,
                client=client,
                base_dir=args.cache_dir,
                skip_iv=args.skip_iv,
                skip_chains=args.skip_chains,
                max_events_per_ticker=args.max_events_per_ticker,
                status_file=args.status_file,
            )
            summaries.append(summary)
    except KeyboardInterrupt:
        logger.warning("interrupted by user; partial cache is safe to resume")
        write_status(
            "interrupt",
            "interrupted",
            status_file=args.status_file,
            phase="interrupted",
        )
        return 130
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("fetch failed")
        write_status(
            "exception",
            "error",
            status_file=args.status_file,
            phase="error",
            error=repr(exc),
        )
        return 1

    # Final manifest + status flip.
    manifest = write_manifest(args.cache_dir)
    elapsed_s = time.monotonic() - started
    write_status(
        "complete",
        "complete",
        status_file=args.status_file,
        phase="cache_complete",
        elapsed_seconds=int(elapsed_s),
        cache_files=manifest["count"],
    )

    print("=" * 70)
    print("paper_v1 cache pull: COMPLETE")
    for s in summaries:
        print(f"  {s['year']} window={s['window'][0]}..{s['window'][1]}")
        for k, v in s.items():
            if k in ("year", "window"):
                continue
            print(f"    {k}: {v}")
    print(f"  cache files: {manifest['count']}")
    print(f"  wall time:   {elapsed_s:.1f}s")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
