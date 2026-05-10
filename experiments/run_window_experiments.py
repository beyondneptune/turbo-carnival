"""Rolling-window experiment harness for the paper options stack.

This script is deliberately isolated from live paper artifacts:

- It never writes data/trading_system.db.
- It never writes data/paper_v1_status.json.
- It never uses run ids that match paper_v1_evidence_*.

Each window gets a scratch audit DB and status JSON under experiments/runs/.
The goal is fast research over alternate train/test windows while the current
paper month and official evidence path remain stable.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import run_paper_v1_evidence as evidence  # noqa: E402


DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiments" / "runs"


@dataclass(frozen=True)
class WindowSpec:
    name: str
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_days: int
    test_days: int


def _parse_days_csv(value: str) -> list[int]:
    out: list[int] = []
    for part in value.split(","):
        text = part.strip()
        if not text:
            continue
        days = int(text)
        if days <= 0:
            raise argparse.ArgumentTypeError("window day counts must be positive")
        out.append(days)
    if not out:
        raise argparse.ArgumentTypeError("at least one day count is required")
    return out


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (date, datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return str(obj)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default))


def _cache_years(base_dir: Path, *, allow_synthetic: bool) -> list[dict[str, Any]]:
    return [
        evidence.load_year_cache(base_dir, "year1", allow_synthetic=allow_synthetic),
        evidence.load_year_cache(base_dir, "year2", allow_synthetic=allow_synthetic),
    ]


def combine_caches(caches: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge year1/year2 cache dicts into one event-addressable cache."""
    chains: dict[tuple[str, date], pd.DataFrame] = {}
    chain_sources: dict[tuple[str, date], str] = {}
    earnings_dates: dict[str, list[date]] = {}
    events_dropped: list[dict[str, Any]] = []
    spy_daily_parts: list[pd.DataFrame] = []
    spy_iv_parts: list[pd.DataFrame] = []
    window_starts: list[date] = []
    window_ends: list[date] = []

    for cache in caches:
        chains.update(cache.get("chains") or {})
        chain_sources.update(cache.get("chain_sources") or {})
        for ticker, dates in (cache.get("earnings_dates") or {}).items():
            earnings_dates.setdefault(ticker, []).extend(dates)
        events_dropped.extend(cache.get("events_dropped") or [])

        spy_daily = cache.get("spy_daily")
        if isinstance(spy_daily, pd.DataFrame) and not spy_daily.empty:
            spy_daily_parts.append(spy_daily)
        spy_iv = cache.get("spy_iv_daily")
        if isinstance(spy_iv, pd.DataFrame) and not spy_iv.empty:
            spy_iv_parts.append(spy_iv)

        win = cache.get("window")
        if isinstance(win, tuple) and len(win) == 2:
            window_starts.append(win[0])
            window_ends.append(win[1])

    for ticker, dates in earnings_dates.items():
        earnings_dates[ticker] = sorted(set(dates))

    spy_daily_all = _concat_time_indexed(spy_daily_parts)
    spy_iv_all = _concat_time_indexed(spy_iv_parts)
    window = (
        min(window_starts) if window_starts else None,
        max(window_ends) if window_ends else None,
    )
    return {
        "year": "combined",
        "window": window,
        "spy_daily": spy_daily_all,
        "spy_iv_daily": spy_iv_all,
        "earnings_dates": earnings_dates,
        "chains": chains,
        "chain_sources": chain_sources,
        "events_dropped": events_dropped,
    }


def _concat_time_indexed(parts: list[pd.DataFrame]) -> pd.DataFrame | None:
    if not parts:
        return None
    out = pd.concat(parts).sort_index()
    return out[~out.index.duplicated(keep="last")]


def filter_cache_window(cache: dict[str, Any], start: date, end: date, *, label: str) -> dict[str, Any]:
    """Return a cache dict containing only events whose event_date is in [start, end]."""
    chains = {
        key: value
        for key, value in (cache.get("chains") or {}).items()
        if start <= key[1] <= end
    }
    chain_sources = {
        key: value
        for key, value in (cache.get("chain_sources") or {}).items()
        if start <= key[1] <= end
    }
    earnings_dates: dict[str, list[date]] = {}
    for ticker, dates in (cache.get("earnings_dates") or {}).items():
        kept = [d for d in dates if start <= d <= end]
        if kept:
            earnings_dates[ticker] = sorted(set(kept))

    events_dropped = []
    for item in cache.get("events_dropped") or []:
        ev = item.get("event_date")
        if isinstance(ev, str):
            try:
                ev = date.fromisoformat(ev)
            except ValueError:
                ev = None
        if isinstance(ev, date) and start <= ev <= end:
            events_dropped.append(item)

    return {
        "year": label,
        "window": (start, end),
        "spy_daily": _slice_time_indexed(cache.get("spy_daily"), start, end),
        "spy_iv_daily": _slice_time_indexed(cache.get("spy_iv_daily"), start, end),
        "earnings_dates": earnings_dates,
        "chains": chains,
        "chain_sources": chain_sources,
        "events_dropped": events_dropped,
    }


def _slice_time_indexed(frame: Any, start: date, end: date) -> pd.DataFrame | None:
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return None
    idx_dates = pd.to_datetime(frame.index, errors="coerce").date
    mask = [(d is not pd.NaT) and start <= d <= end for d in idx_dates]
    return frame.loc[mask].copy()


def cache_event_bounds(cache: dict[str, Any]) -> tuple[date, date]:
    dates = [event_date for (_ticker, event_date) in (cache.get("chains") or {}).keys()]
    if not dates:
        raise RuntimeError("cache has no priced/loaded events to window")
    return min(dates), max(dates)


def generate_windows(
    *,
    first_event: date,
    last_event: date,
    train_days: list[int],
    test_days: list[int],
    step_days: int,
    max_windows: int | None = None,
) -> list[WindowSpec]:
    specs: list[WindowSpec] = []
    for train_len in train_days:
        for test_len in test_days:
            test_start = first_event + timedelta(days=train_len)
            while test_start + timedelta(days=test_len - 1) <= last_event:
                train_start = test_start - timedelta(days=train_len)
                train_end = test_start - timedelta(days=1)
                test_end = test_start + timedelta(days=test_len - 1)
                name = (
                    f"exp{len(specs) + 1:03d}_"
                    f"train{train_len}_test{test_len}_{test_start.isoformat()}"
                )
                specs.append(
                    WindowSpec(
                        name=name,
                        train_start=train_start,
                        train_end=train_end,
                        test_start=test_start,
                        test_end=test_end,
                        train_days=train_len,
                        test_days=test_len,
                    )
                )
                if max_windows is not None and len(specs) >= max_windows:
                    return specs
                test_start += timedelta(days=step_days)
    return specs


def _insert_experiment_run_row(conn: sqlite3.Connection, run_id: str, spec: WindowSpec) -> None:
    conn.execute(
        """
        INSERT INTO runs (
            run_id, run_type, prereg_version, methodology_commit,
            started_at_utc, ended_at_utc, status, initial_nav_usd, notes
        ) VALUES (?, 'research_experiment', NULL, ?, ?, NULL, 'running', ?, ?)
        """,
        (
            run_id,
            evidence._current_methodology_commit(),
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            evidence.INITIAL_NAV_USD,
            (
                "offline rolling-window experiment; "
                f"train={spec.train_start.isoformat()}..{spec.train_end.isoformat()} "
                f"test={spec.test_start.isoformat()}..{spec.test_end.isoformat()}"
            ),
        ),
    )
    conn.commit()


def _materialize_spy_priced_window(
    *,
    cache_dir: Path,
    dest_dir: Path,
    start: date,
    end: date,
) -> Path | None:
    """Symlink available SPY priced files for a test window into one directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for year in ("year1", "year2"):
        src_dir = cache_dir / year / "chains_priced" / "SPY"
        if not src_dir.exists():
            continue
        for src in sorted(src_dir.glob("*.parquet")):
            try:
                entry_date = date.fromisoformat(src.stem)
            except ValueError:
                continue
            if not (start <= entry_date <= end):
                continue
            dst = dest_dir / src.name
            if not dst.exists():
                try:
                    dst.symlink_to(src.resolve())
                except OSError:
                    shutil.copy2(src, dst)
            count += 1
    return dest_dir if count else None


def run_window(
    *,
    full_cache: dict[str, Any],
    cache_dir: Path,
    run_dir: Path,
    spec: WindowSpec,
    seed: int,
    allow_synthetic: bool,
    min_train_events: int,
    min_test_events: int,
    cost_stress_multiplier: float,
    synthetic_spy_core: bool,
) -> dict[str, Any]:
    win_dir = run_dir / spec.name
    win_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"research_window_{run_dir.name}_{spec.name}"
    db_path = win_dir / "experiment.db"
    status_path = win_dir / "status.json"
    train_cache = filter_cache_window(
        full_cache, spec.train_start, spec.train_end, label="train"
    )
    test_cache = filter_cache_window(
        full_cache, spec.test_start, spec.test_end, label="test"
    )

    train_events = evidence.build_earnings_events(train_cache)
    test_events = evidence.build_earnings_events(test_cache)
    row: dict[str, Any] = {
        "name": spec.name,
        "run_id": run_id,
        "train_start": spec.train_start,
        "train_end": spec.train_end,
        "test_start": spec.test_start,
        "test_end": spec.test_end,
        "train_days": spec.train_days,
        "test_days": spec.test_days,
        "train_events": len(train_events),
        "test_events": len(test_events),
        "db_path": db_path,
        "status_path": status_path,
    }
    if len(train_events) < min_train_events or len(test_events) < min_test_events:
        row.update(
            {
                "status": "skipped",
                "reason": (
                    f"events below floor train={len(train_events)}/{min_train_events} "
                    f"test={len(test_events)}/{min_test_events}"
                ),
            }
        )
        _write_json(win_dir / "window.json", {"spec": spec, "summary": row})
        return row

    rng = np.random.default_rng(seed)
    try:
        X, y_a, y_b, realized = evidence.build_year1_training_frame(
            train_cache,
            rng=rng,
            allow_synthetic=allow_synthetic,
        )
        if len(X) < 8:
            raise RuntimeError(f"insufficient train rows after target build: {len(X)}")
        model_a, model_b, comparison = evidence.train_and_compare(
            X, y_a, y_b, realized, random_state=seed
        )
        selected_model = model_b if comparison.b_beats_a else model_a
        selected_version = (
            evidence.MODEL_VERSION
            if comparison.b_beats_a
            else "lightgbm_modelA_window_experiment"
        )

        conn = evidence.open_db(db_path)
        try:
            _insert_experiment_run_row(conn, run_id, spec)
            status = evidence.StatusWriter(status_path, run_id=run_id)
            status.set_phase("window_experiment")
            status.set_stage("train_window", "complete", rows=int(len(X)))

            spy_priced_dir = None
            if not synthetic_spy_core:
                spy_priced_dir = _materialize_spy_priced_window(
                    cache_dir=cache_dir,
                    dest_dir=win_dir / "spy_priced",
                    start=spec.test_start,
                    end=spec.test_end,
                )

            replay = evidence.replay_year2(
                conn=conn,
                run_id=run_id,
                cache=test_cache,
                status=status,
                rng=rng,
                cost_stress_multiplier=cost_stress_multiplier,
                synthetic_spy_chains=synthetic_spy_core or spy_priced_dir is None,
                allow_synthetic=allow_synthetic or synthetic_spy_core,
                spy_priced_dir=spy_priced_dir,
                satellite_model=selected_model,
                satellite_model_version=selected_version,
                satellite_signal_threshold=evidence.SATELLITE_SIGNAL_THRESHOLD,
            )
            base_inputs = evidence.build_cost_regime_inputs(
                strategy_returns=replay["strategy_returns"],
                underlying_moves=replay["underlying_moves"],
                pre_event_iv=replay["pre_event_iv"],
                features_for_b3=replay["features_for_b3"],
                cost_multiplier=1.0,
            )
            plus50_inputs = evidence.build_cost_regime_inputs(
                strategy_returns=replay["strategy_returns"],
                underlying_moves=replay["underlying_moves"],
                pre_event_iv=replay["pre_event_iv"],
                features_for_b3=replay["features_for_b3"],
                cost_multiplier=1.5,
            )
            report = evidence.evaluate_full(base_inputs, plus50_inputs)
            evidence.update_run_status(conn, run_id, "completed")
        finally:
            conn.close()
    except Exception as exc:
        row.update({"status": "error", "reason": str(exc)})
        _write_json(win_dir / "window.json", {"spec": spec, "summary": row})
        return row

    row.update(
        {
            "status": "complete",
            "train_rows": int(len(X)),
            "selected_model": "B" if comparison.b_beats_a else "A",
            "twin_b_beats_a": bool(comparison.b_beats_a),
            "twin_net_excess_bps_ann": float(comparison.net_excess_b_minus_a_bps_ann),
            "twin_sharpe_delta": float(comparison.sharpe_delta_b_minus_a),
            "twin_t_stat": float(comparison.paired_t_stat),
            "trades_placed": int(replay["trades_placed"]),
            "trades_blocked": int(replay["trades_blocked"]),
            "events_dropped_no_pricing": int(replay.get("events_dropped_no_pricing", 0)),
            "strategy_events": int(len(replay["strategy_returns"])),
            "strategy_ann_return": float(base_inputs.strategy_annualised_return),
            "best_baseline_ann_return": float(base_inputs.best_baseline_annualised_return),
            "net_excess_bps": float(
                (base_inputs.strategy_annualised_return - base_inputs.best_baseline_annualised_return)
                * 10_000.0
            ),
            "max_drawdown": float(base_inputs.max_drawdown),
            "dsr": float(base_inputs.dsr),
            "pbo": float(base_inputs.pbo),
            "gates_pass": bool(report.overall_pass),
            "failed_gates": ";".join(report.failed_gates_summary),
            "trade_source_counts": replay.get("trade_source_counts", {}),
        }
    )
    _write_json(
        win_dir / "window.json",
        {
            "spec": spec,
            "summary": row,
            "twin_model_comparison": comparison,
            "base_inputs": base_inputs,
            "plus50_inputs": plus50_inputs,
            "gate_report": report,
        },
    )
    return row


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run isolated rolling-window experiments over the paper cache."
    )
    p.add_argument("--cache-dir", type=Path, default=evidence.DEFAULT_CACHE_DIR)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument("--run-set-id", default=None)
    p.add_argument("--train-days", type=_parse_days_csv, default=_parse_days_csv("180,252,365"))
    p.add_argument("--test-days", type=_parse_days_csv, default=_parse_days_csv("60,90"))
    p.add_argument("--step-days", type=int, default=30)
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument(
        "--jobs",
        type=int,
        default=min(4, os.cpu_count() or 1),
        help="Parallel window workers. Each worker writes to its own scratch DB.",
    )
    p.add_argument("--min-train-events", type=int, default=20)
    p.add_argument("--min-test-events", type=int, default=8)
    p.add_argument("--seed", type=int, default=evidence.SYNTHETIC_SEED)
    p.add_argument("--allow-synthetic", action="store_true")
    p.add_argument("--synthetic-spy-core", action="store_true")
    p.add_argument("--cost-stress-multiplier", type=float, default=1.0)
    p.add_argument("--dry-list", action="store_true", help="Only write generated windows.json.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.step_days <= 0:
        raise SystemExit("--step-days must be positive")
    if args.jobs <= 0:
        raise SystemExit("--jobs must be positive")
    if args.max_windows is not None and args.max_windows <= 0:
        max_windows = None
    else:
        max_windows = int(args.max_windows)

    run_set_id = args.run_set_id or (
        "window_grid_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    run_dir = args.output_root / run_set_id
    run_dir.mkdir(parents=True, exist_ok=True)

    caches = _cache_years(args.cache_dir, allow_synthetic=args.allow_synthetic)
    full_cache = combine_caches(caches)
    first_event, last_event = cache_event_bounds(full_cache)
    windows = generate_windows(
        first_event=first_event,
        last_event=last_event,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        max_windows=max_windows,
    )
    _write_json(run_dir / "windows.json", {"run_set_id": run_set_id, "windows": windows})
    if args.dry_list:
        print(f"listed {len(windows)} windows under {run_dir}")
        return 0

    jobs = min(args.jobs, max(1, len(windows)))
    rows_by_name: dict[str, dict[str, Any]] = {}

    if jobs == 1:
        for idx, spec in enumerate(windows, start=1):
            print(f"[{idx}/{len(windows)}] {spec.name}", flush=True)
            row = run_window(
                full_cache=full_cache,
                cache_dir=args.cache_dir,
                run_dir=run_dir,
                spec=spec,
                seed=args.seed + idx,
                allow_synthetic=args.allow_synthetic,
                min_train_events=args.min_train_events,
                min_test_events=args.min_test_events,
                cost_stress_multiplier=args.cost_stress_multiplier,
                synthetic_spy_core=args.synthetic_spy_core,
            )
            rows_by_name[spec.name] = row
            _write_summaries(run_dir, run_set_id, windows, rows_by_name)
    else:
        print(f"running {len(windows)} windows with {jobs} parallel workers", flush=True)
        with concurrent.futures.ProcessPoolExecutor(max_workers=jobs) as pool:
            futures = {
                pool.submit(
                    run_window,
                    full_cache=full_cache,
                    cache_dir=args.cache_dir,
                    run_dir=run_dir,
                    spec=spec,
                    seed=args.seed + idx,
                    allow_synthetic=args.allow_synthetic,
                    min_train_events=args.min_train_events,
                    min_test_events=args.min_test_events,
                    cost_stress_multiplier=args.cost_stress_multiplier,
                    synthetic_spy_core=args.synthetic_spy_core,
                ): (idx, spec)
                for idx, spec in enumerate(windows, start=1)
            }
            for fut in concurrent.futures.as_completed(futures):
                idx, spec = futures[fut]
                try:
                    row = fut.result()
                except Exception as exc:  # pragma: no cover - worker crash guard
                    row = {
                        "name": spec.name,
                        "status": "error",
                        "reason": f"worker crashed: {exc}",
                    }
                rows_by_name[spec.name] = row
                status = row.get("status")
                print(f"[{idx}/{len(windows)}] {spec.name}: {status}", flush=True)
                _write_summaries(run_dir, run_set_id, windows, rows_by_name)

    rows = _ordered_rows(windows, rows_by_name)

    complete = sum(1 for row in rows if row.get("status") == "complete")
    skipped = sum(1 for row in rows if row.get("status") == "skipped")
    errors = sum(1 for row in rows if row.get("status") == "error")
    print(
        f"window experiments complete: {complete} complete, {skipped} skipped, "
        f"{errors} errors -> {run_dir}"
    )
    return 0 if errors == 0 else 1


def _ordered_rows(
    windows: list[WindowSpec],
    rows_by_name: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [rows_by_name[name] for name in [w.name for w in windows] if name in rows_by_name]


def _write_summaries(
    run_dir: Path,
    run_set_id: str,
    windows: list[WindowSpec],
    rows_by_name: dict[str, dict[str, Any]],
) -> None:
    rows = _ordered_rows(windows, rows_by_name)
    pd.DataFrame(rows).to_csv(run_dir / "summary.csv", index=False)
    _write_json(run_dir / "summary.json", {"run_set_id": run_set_id, "results": rows})


if __name__ == "__main__":
    raise SystemExit(main())
