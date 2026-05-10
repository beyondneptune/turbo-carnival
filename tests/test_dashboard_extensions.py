"""Tests for the Paper v1 / System view helpers added to dashboard/app.py.

We only exercise the data-fetch helpers; rendering Streamlit widgets is
intentionally out of scope (heavy + brittle).
"""
from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dashboard import app as dashboard_app
from dashboard.app import (
    _is_simulation_running,
    count_satellite_events,
    count_priced_chain_files,
    count_spy_core_files,
    fetch_latest_run_id,
    fetch_live_simulation_summary,
    fetch_prediction_actuals,
    fetch_run_trades,
    fetch_run_equity_curve,
    filter_paper_v1_processes,
    get_system_snapshot,
    is_status_stale,
    live_log_sources,
    paper_v1_log_path_from_status,
    read_offline_context_manifest,
    read_offline_context_status,
    read_paper_v1_status,
    read_paper_v1_unofficial_status,
    resolve_live_run_id,
    summarize_offline_context_cache,
    tail_text_file,
    total_polygon_api_calls,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "sql" / "schema.sql"


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_db_with_run(
    path: Path,
    *,
    run_id: str = "paper_v1_evidence_test",
    n_satellite: int = 0,
    n_demo: int = 0,
    pnl_sequence: list[float] | None = None,
) -> None:
    """Create a populated DB with a run and a configurable number of trades."""
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text())
    conn.execute(
        "INSERT INTO runs (run_id, run_type, status, initial_nav_usd, started_at_utc) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, "paper_frozen", "running", 100_000.0, "2026-05-07T03:00:00Z"),
    )
    pnl_iter = iter(pnl_sequence or [])

    def _add(sleeve: str, idx: int, pnl: float | None) -> None:
        cur = conn.execute(
            "INSERT INTO gate_decisions (run_id, decision_at_utc, underlying_ticker, sleeve, "
            "candidate_summary, signal_score, signal_threshold, signal_pass, "
            "risk_spread_pass, risk_vega_pass, risk_scenario_pass, "
            "risk_liquidity_pass, risk_pin_pass, risk_concentration_pass, final_decision) "
            "VALUES (?, ?, ?, ?, ?, ?, 0.0, 1, 1, 1, 1, 1, 1, 1, 'approved')",
            (
                run_id,
                f"2026-05-07T04:{idx:02d}:00Z",
                "SPY",
                sleeve,
                "test",
                float(idx) / 10.0,
            ),
        )
        decision_id = cur.lastrowid
        status = "closed" if pnl is not None else "open"
        strategy = "iron_condor" if sleeve == "satellite" else "demo_long_equity"
        conn.execute(
            "INSERT INTO trades (run_id, decision_id, sleeve, strategy_type, underlying_ticker, "
            "status, entry_at_utc, exit_at_utc, realized_pnl_usd) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                decision_id,
                sleeve,
                strategy,
                "SPY",
                status,
                f"2026-05-07T05:{idx:02d}:00Z",
                f"2026-05-07T06:{idx:02d}:00Z" if pnl is not None else None,
                pnl,
            ),
        )

    counter = 0
    for _ in range(n_satellite):
        pnl = next(pnl_iter, None)
        _add("satellite", counter, pnl)
        counter += 1
    for _ in range(n_demo):
        pnl = next(pnl_iter, None)
        _add("demo", counter, pnl)
        counter += 1
    conn.commit()
    conn.close()


def _sample_status_payload() -> dict:
    return {
        "run_id": "paper_v1_evidence_20260507T030000Z",
        "phase": "year2_replay",
        "started_at_utc": "2026-05-07T03:00:00Z",
        "last_update_utc": "2026-05-07T05:42:13Z",
        "stages": {
            "fetch_year1": {"status": "complete", "events": 200, "api_calls": 950, "cache_hits": 0},
            "fetch_year2": {"status": "complete", "events": 195, "api_calls": 920, "cache_hits": 0},
            "train_models": {
                "status": "complete",
                "model_a_val_mse": 0.0042,
                "model_b_val_mse": 0.0038,
            },
            "year2_replay": {
                "status": "running",
                "events_processed": 87,
                "events_total": 195,
                "trades_placed": 62,
                "trades_blocked": 25,
            },
        },
        "twin_model_comparison": {
            "net_excess_bps_ann": 78.4,
            "sharpe_delta": 0.31,
            "paired_t_stat": 2.42,
            "b_beats_a": True,
        },
        "gates": None,
        "errors": [],
    }


def _sample_offline_manifest() -> dict:
    return {
        "generated_at_utc": "2026-05-09T20:29:08+00:00",
        "output_dir": "/tmp/offline_context",
        "file_count": 6,
        "total_bytes": 1536,
        "summaries": {
            "yfinance_daily": {
                "written": 2,
                "skipped": 3,
                "failures": {"BAD": "empty"},
            },
            "yfinance_actions": {
                "written": 5,
                "skipped": 0,
                "failures": {},
            },
            "fred": {
                "written": 4,
                "skipped": 1,
                "failures": {"DGS10": "timeout"},
            },
        },
    }


def _sample_offline_status() -> dict:
    return {
        "phase": "offline_context",
        "started_at_utc": "2026-05-09T20:27:25+00:00",
        "last_update_utc": "2026-05-09T20:29:08+00:00",
        "stages": {
            "yfinance_daily": {
                "status": "complete",
                "written": 2,
                "skipped": 3,
                "failures": {"BAD": "empty"},
            },
            "yfinance_actions": {
                "status": "running",
                "done": 2,
                "total": 5,
                "written": 2,
                "skipped": 0,
                "failures": {},
            },
            "fred": {
                "status": "complete",
                "done": 5,
                "total": 5,
                "written": 4,
                "skipped": 1,
                "failures": {"DGS10": "timeout"},
            },
            "complete": {
                "status": "complete",
                "file_count": 5,
                "total_bytes": 1024,
            },
        },
    }


# ---------------------------------------------------------------------------
# read_paper_v1_status
# ---------------------------------------------------------------------------


def test_read_paper_v1_status_missing(tmp_path):
    assert read_paper_v1_status(tmp_path / "not_there.json") is None


def test_read_paper_v1_status_parses_synthetic_file(tmp_path):
    payload = _sample_status_payload()
    target = tmp_path / "paper_v1_status.json"
    target.write_text(json.dumps(payload))
    parsed = read_paper_v1_status(target)
    assert parsed is not None
    assert parsed["run_id"] == "paper_v1_evidence_20260507T030000Z"
    assert parsed["phase"] == "year2_replay"
    assert parsed["stages"]["year2_replay"]["events_processed"] == 87
    assert parsed["twin_model_comparison"]["b_beats_a"] is True


def test_read_paper_v1_status_invalid_json(tmp_path):
    target = tmp_path / "broken.json"
    target.write_text("{not valid json")
    assert read_paper_v1_status(target) is None


def test_read_paper_v1_status_non_dict_root(tmp_path):
    target = tmp_path / "list.json"
    target.write_text(json.dumps([1, 2, 3]))
    assert read_paper_v1_status(target) is None


def test_read_paper_v1_unofficial_status_parses_smoke_file(tmp_path):
    payload = {
        "run_id": "paper_v1_unofficial_smoke_test",
        "phase": "complete",
        "errors": [],
    }
    target = tmp_path / "paper_v1_unofficial_status.json"
    target.write_text(json.dumps(payload))
    parsed = read_paper_v1_unofficial_status(target)
    assert parsed is not None
    assert parsed["run_id"] == "paper_v1_unofficial_smoke_test"
    assert parsed["phase"] == "complete"


# ---------------------------------------------------------------------------
# offline context cache helpers
# ---------------------------------------------------------------------------


def test_read_offline_context_manifest_missing(tmp_path):
    assert read_offline_context_manifest(tmp_path / "not_there.json") is None


def test_read_offline_context_status_invalid_json(tmp_path):
    target = tmp_path / "offline_context_status.json"
    target.write_text("{not valid json")
    assert read_offline_context_status(target) is None


def test_summarize_offline_context_cache_from_manifest_and_status(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    status_path = tmp_path / "offline_context_status.json"
    manifest_path.write_text(json.dumps(_sample_offline_manifest()))
    status_path.write_text(json.dumps(_sample_offline_status()))

    summary = summarize_offline_context_cache(manifest_path, status_path)

    assert summary["manifest_present"] is True
    assert summary["status_present"] is True
    assert summary["file_count"] == 6
    assert summary["total_bytes"] == 1536
    assert summary["size_label"] == "1.5 KB"
    assert summary["status_label"] == "running"
    assert summary["stage_summary"] == "running: YF actions"
    assert summary["generated_at_utc"] == "2026-05-09T20:29:08+00:00"
    assert summary["last_update_utc"] == "2026-05-09T20:29:08+00:00"

    daily = summary["sources"]["yfinance_daily"]
    assert daily["label"] == "YF daily"
    assert daily["count"] == 5
    assert daily["written"] == 2
    assert daily["skipped"] == 3
    assert daily["failures"] == 1
    assert daily["status"] == "complete"

    actions = summary["sources"]["yfinance_actions"]
    assert actions["count"] == 5
    assert actions["failures"] == 0
    assert actions["status"] == "running"

    fred = summary["sources"]["fred"]
    assert fred["label"] == "FRED"
    assert fred["count"] == 5
    assert fred["failures"] == 1


def test_summarize_offline_context_cache_falls_back_to_status(tmp_path):
    status_path = tmp_path / "offline_context_status.json"
    status = {
        "phase": "offline_context",
        "last_update_utc": "2026-05-09T20:29:08+00:00",
        "stages": {
            "fred": {
                "status": "complete",
                "total": 17,
                "written": 0,
                "skipped": 17,
                "failures": {},
            },
            "complete": {
                "status": "complete",
                "file_count": 18,
                "total_bytes": 2048,
            },
        },
    }
    status_path.write_text(json.dumps(status))

    summary = summarize_offline_context_cache(tmp_path / "missing.json", status_path)

    assert summary["manifest_present"] is False
    assert summary["status_present"] is True
    assert summary["file_count"] == 18
    assert summary["total_bytes"] == 2048
    assert summary["size_label"] == "2.0 KB"
    assert summary["status_label"] == "complete"
    assert summary["sources"]["fred"]["count"] == 17
    assert summary["sources"]["fred"]["failures"] == 0


def test_summarize_offline_context_cache_missing_files(tmp_path):
    summary = summarize_offline_context_cache(
        tmp_path / "missing_manifest.json",
        tmp_path / "missing_status.json",
    )

    assert summary["manifest_present"] is False
    assert summary["status_present"] is False
    assert summary["file_count"] is None
    assert summary["total_bytes"] is None
    assert summary["size_label"] == "—"
    assert summary["status_label"] == "missing"
    assert summary["stage_summary"] == "—"


# ---------------------------------------------------------------------------
# is_status_stale
# ---------------------------------------------------------------------------


def test_is_status_stale_recent_update():
    now = datetime(2026, 5, 7, 6, 0, 0, tzinfo=timezone.utc)
    payload = _sample_status_payload()
    payload["last_update_utc"] = (now - timedelta(minutes=5)).isoformat().replace("+00:00", "Z")
    assert is_status_stale(payload, now=now) is False


def test_is_status_stale_old_update():
    now = datetime(2026, 5, 7, 6, 0, 0, tzinfo=timezone.utc)
    payload = _sample_status_payload()
    payload["last_update_utc"] = (now - timedelta(minutes=45)).isoformat().replace("+00:00", "Z")
    assert is_status_stale(payload, now=now) is True


def test_is_status_stale_missing_field():
    assert is_status_stale({}) is True


def test_is_status_stale_terminal_phase_never_stale():
    """Terminal phases (cache_complete, complete, error, interrupted) are never
    stale — they're done; old timestamps are expected. Per the dashboard rules."""
    now = datetime(2026, 5, 9, 17, 0, 0, tzinfo=timezone.utc)
    very_old = (now - timedelta(days=2)).isoformat().replace("+00:00", "Z")
    for phase in ("cache_complete", "complete", "error", "interrupted"):
        payload = {"phase": phase, "last_update_utc": very_old}
        assert is_status_stale(payload, now=now) is False, (
            f"phase={phase!r} should not be stale even at 2 days old"
        )


def test_is_status_stale_running_phase_old_update_is_stale():
    """Non-terminal phases still go stale on old timestamps (regression check)."""
    now = datetime(2026, 5, 9, 17, 0, 0, tzinfo=timezone.utc)
    very_old = (now - timedelta(hours=2)).isoformat().replace("+00:00", "Z")
    payload = {"phase": "year2_replay", "last_update_utc": very_old}
    assert is_status_stale(payload, now=now) is True


# ---------------------------------------------------------------------------
# count_satellite_events
# ---------------------------------------------------------------------------


def test_count_satellite_events_empty_db(tmp_path):
    db_path = tmp_path / "empty.db"
    _make_db_with_run(db_path, run_id="r0", n_satellite=0, n_demo=0)
    conn = sqlite3.connect(db_path)
    assert count_satellite_events(conn, "r0") == 0
    conn.close()


def test_count_satellite_events_populated(tmp_path):
    db_path = tmp_path / "p.db"
    _make_db_with_run(db_path, run_id="r0", n_satellite=4, n_demo=2)
    conn = sqlite3.connect(db_path)
    assert count_satellite_events(conn, "r0") == 4
    # demo trades should not be counted
    assert count_satellite_events(conn, "no_such_run") == 0
    conn.close()


def test_count_satellite_events_handles_none_conn():
    assert count_satellite_events(None, "anything") == 0


def test_count_satellite_events_handles_empty_run_id(tmp_path):
    db_path = tmp_path / "p.db"
    _make_db_with_run(db_path, run_id="r0", n_satellite=2)
    conn = sqlite3.connect(db_path)
    assert count_satellite_events(conn, "") == 0
    conn.close()


# ---------------------------------------------------------------------------
# fetch_run_equity_curve
# ---------------------------------------------------------------------------


def test_fetch_run_equity_curve_returns_cumulative(tmp_path):
    db_path = tmp_path / "p.db"
    _make_db_with_run(
        db_path,
        run_id="r0",
        n_satellite=3,
        pnl_sequence=[100.0, -25.0, 50.0],
    )
    conn = sqlite3.connect(db_path)
    df = fetch_run_equity_curve(conn, "r0")
    assert len(df) == 3
    assert list(df["cumulative_pnl_usd"]) == [100.0, 75.0, 125.0]
    conn.close()


def test_fetch_run_equity_curve_no_closed_trades(tmp_path):
    db_path = tmp_path / "p.db"
    _make_db_with_run(db_path, run_id="r0", n_satellite=2, pnl_sequence=[])
    conn = sqlite3.connect(db_path)
    df = fetch_run_equity_curve(conn, "r0")
    assert df.empty
    conn.close()


def test_fetch_run_equity_curve_handles_none_conn():
    df = fetch_run_equity_curve(None, "anything")
    assert df.empty


# ---------------------------------------------------------------------------
# live simulation helpers
# ---------------------------------------------------------------------------


def test_fetch_latest_run_id_returns_newest(tmp_path):
    db_path = tmp_path / "runs.db"
    _make_db_with_run(db_path, run_id="old")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO runs (run_id, run_type, status, initial_nav_usd, started_at_utc) "
        "VALUES ('new', 'paper_demo', 'running', 1000.0, '2026-05-08T00:00:00Z')"
    )
    conn.commit()

    assert fetch_latest_run_id(conn) == "new"
    conn.close()


def test_resolve_live_run_id_prefers_status_then_db(tmp_path):
    db_path = tmp_path / "runs.db"
    _make_db_with_run(db_path, run_id="db_run")
    conn = sqlite3.connect(db_path)

    assert resolve_live_run_id(conn, {"run_id": "status_run"}) == "status_run"
    assert resolve_live_run_id(conn, {}) == "db_run"
    conn.close()


def test_fetch_live_simulation_summary_aggregates_decisions_and_trades(tmp_path):
    db_path = tmp_path / "sim.db"
    _make_db_with_run(
        db_path,
        run_id="r0",
        n_satellite=3,
        n_demo=1,
        pnl_sequence=[100.0, -40.0, 0.0, 20.0],
    )
    conn = sqlite3.connect(db_path)

    out = fetch_live_simulation_summary(conn, "r0")

    assert out["decisions_total"] == 4
    assert out["approved"] == 4
    assert out["blocked"] == 0
    assert out["trades_total"] == 4
    assert out["closed_trades"] == 4
    assert out["satellite_trades"] == 3
    assert out["realized_pnl_usd"] == 80.0
    assert out["win_rate"] == 0.5
    conn.close()


def test_fetch_prediction_actuals_joins_decisions_to_trades(tmp_path):
    db_path = tmp_path / "sim.db"
    _make_db_with_run(
        db_path,
        run_id="r0",
        n_satellite=2,
        pnl_sequence=[100.0, -20.0],
    )
    conn = sqlite3.connect(db_path)

    df = fetch_prediction_actuals(conn, "r0")

    assert len(df) == 2
    assert {"signal_score", "signal_threshold", "realized_pnl_usd"}.issubset(df.columns)
    assert list(df["realized_pnl_usd"]) == [100.0, -20.0]
    assert list(df["signal_score"]) == [0.0, 0.1]
    conn.close()


def test_fetch_run_trades_returns_recent_trades(tmp_path):
    db_path = tmp_path / "sim.db"
    _make_db_with_run(
        db_path,
        run_id="r0",
        n_satellite=2,
        pnl_sequence=[100.0, -20.0],
    )
    conn = sqlite3.connect(db_path)

    df = fetch_run_trades(conn, "r0")

    assert len(df) == 2
    assert {"trade_id", "sleeve", "realized_pnl_usd"}.issubset(df.columns)
    assert list(df["realized_pnl_usd"]) == [100.0, -20.0]
    conn.close()


def test_paper_v1_log_path_from_status_uses_init_log_file(tmp_path):
    log_file = tmp_path / "active.log"
    payload = {"stages": {"init": {"log_file": str(log_file)}}}
    assert paper_v1_log_path_from_status(payload) == log_file.resolve()


# ---------------------------------------------------------------------------
# total_polygon_api_calls
# ---------------------------------------------------------------------------


def test_total_polygon_api_calls_sums_stages():
    payload = _sample_status_payload()
    assert total_polygon_api_calls(payload) == 1870  # 950 + 920


def test_total_polygon_api_calls_returns_none_when_absent():
    assert total_polygon_api_calls(None) is None
    assert total_polygon_api_calls({}) is None
    assert total_polygon_api_calls({"stages": {"x": {"status": "ok"}}}) is None


# ---------------------------------------------------------------------------
# Behind-the-scenes helpers
# ---------------------------------------------------------------------------


def test_count_priced_chain_files_by_year(tmp_path):
    for path in (
        tmp_path / "year1" / "chains_priced" / "AAPL" / "2024-08-01.parquet",
        tmp_path / "year1" / "chains_priced" / "MSFT" / "2024-10-24.parquet",
        tmp_path / "year2" / "chains_priced" / "NVDA" / "2025-08-27.parquet",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"marker")

    out = count_priced_chain_files(tmp_path)
    assert out == {"year1": 2, "year2": 1, "total": 3}


def test_count_spy_core_files_by_year(tmp_path):
    for path in (
        tmp_path / "year1" / "chains_priced" / "SPY" / "2024-05-13.parquet",
        tmp_path / "year2" / "chains_priced" / "SPY" / "2025-05-12.parquet",
        tmp_path / "year2" / "chains_priced" / "SPY" / "2025-05-19.parquet",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"marker")

    out = count_spy_core_files(tmp_path)
    assert out == {"year1": 1, "year2": 2, "total": 3}


def test_tail_text_file_returns_last_lines(tmp_path):
    log = tmp_path / "paper_v1_augment.log"
    log.write_text("\n".join(f"line {i}" for i in range(10)))

    assert tail_text_file(log, max_lines=3) == ["line 7", "line 8", "line 9"]
    assert tail_text_file(tmp_path / "missing.log", max_lines=3) == []


def test_live_log_sources_prefers_active_log_and_dedupes(monkeypatch, tmp_path):
    active = tmp_path / "active.log"
    err = tmp_path / "active.err"
    out = tmp_path / "active.out"
    default = tmp_path / "default.log"
    active.write_text("hello\n")
    err.write_text("err\n")
    out.write_text("")
    default.write_text("default\n")
    monkeypatch.delenv("PAPER_V1_LOG_PATH", raising=False)
    monkeypatch.setattr(dashboard_app, "PAPER_V1_SPY_CORE_LOG_PATH", active)
    monkeypatch.setattr(dashboard_app, "PAPER_V1_SPY_CORE_ERR_PATH", err)
    monkeypatch.setattr(dashboard_app, "PAPER_V1_SPY_CORE_OUT_PATH", out)
    monkeypatch.setattr(dashboard_app, "PAPER_V1_WATCHDOG_LOG_PATH", tmp_path / "watchdog.log")
    monkeypatch.setattr(dashboard_app, "PAPER_V1_LOG_PATH", default)

    status = {"stages": {"init": {"log_file": str(active)}}}
    sources = live_log_sources(status)
    paths = [row["path"] for row in sources]

    assert paths.count(active.resolve()) == 1
    assert paths[:4] == [
        (tmp_path / "watchdog.log").resolve(),
        active.resolve(),
        err.resolve(),
        out.resolve(),
    ]
    assert sources[1]["exists"] is True
    assert sources[1]["size_bytes"] == len("hello\n")


def test_filter_paper_v1_processes_keeps_relevant_commands():
    processes = [
        {"pid": 1, "cmdline": "python scripts/augment_chain_pricing.py"},
        {"pid": 2, "cmdline": "python scripts/run_paper_v1_evidence.py"},
        {"pid": 3, "cmdline": "python worker.py"},
        {"pid": 4, "cmdline": "streamlit run dashboard/app.py"},
    ]

    out = filter_paper_v1_processes(processes)
    assert [p["pid"] for p in out] == [1, 2, 4]


# ---------------------------------------------------------------------------
# _is_simulation_running (live indicator)
# ---------------------------------------------------------------------------


class _FakeProc:
    def __init__(self, cmdline: list[str]):
        self.info = {"cmdline": cmdline}


def test_is_simulation_running_detects_augment(monkeypatch):
    fake = [
        _FakeProc(["python", "scripts/augment_chain_pricing.py"]),
        _FakeProc(["bash", "-l"]),
    ]
    monkeypatch.setattr(
        dashboard_app.psutil,
        "process_iter",
        lambda attrs=None: iter(fake),
    )
    assert _is_simulation_running() is True


def test_is_simulation_running_detects_evidence_run(monkeypatch):
    fake = [_FakeProc(["python", "scripts/run_paper_v1_evidence.py", "--allow-synthetic"])]
    monkeypatch.setattr(
        dashboard_app.psutil,
        "process_iter",
        lambda attrs=None: iter(fake),
    )
    assert _is_simulation_running() is True


def test_is_simulation_running_returns_false_when_no_match(monkeypatch):
    fake = [
        _FakeProc(["python", "manage.py", "runserver"]),
        _FakeProc(["streamlit", "run", "other.py"]),
        _FakeProc([]),
    ]
    monkeypatch.setattr(
        dashboard_app.psutil,
        "process_iter",
        lambda attrs=None: iter(fake),
    )
    assert _is_simulation_running() is False


# ---------------------------------------------------------------------------
# get_system_snapshot
# ---------------------------------------------------------------------------


def test_get_system_snapshot_has_expected_keys():
    snap = get_system_snapshot()
    expected_keys = {
        "cpu_percent",
        "cpu_per_core",
        "memory_total_gb",
        "memory_used_gb",
        "memory_free_gb",
        "memory_percent",
        "disk_total_gb",
        "disk_used_gb",
        "disk_free_gb",
        "processes",
        "timestamp_utc",
    }
    assert expected_keys.issubset(snap.keys())


def test_get_system_snapshot_numeric_values_are_finite():
    snap = get_system_snapshot()
    numeric_keys = [
        "cpu_percent",
        "memory_total_gb",
        "memory_used_gb",
        "memory_free_gb",
        "memory_percent",
        "disk_total_gb",
        "disk_used_gb",
        "disk_free_gb",
    ]
    for key in numeric_keys:
        value = snap[key]
        assert isinstance(value, float), f"{key} must be float, got {type(value)}"
        assert math.isfinite(value), f"{key} must be finite, got {value}"
        assert value >= 0.0, f"{key} must be >= 0, got {value}"


def test_get_system_snapshot_processes_shape():
    snap = get_system_snapshot()
    procs = snap["processes"]
    assert isinstance(procs, list)
    for proc in procs:
        assert {"pid", "name", "cpu_percent", "rss_mb"}.issubset(proc.keys())
        assert isinstance(proc["pid"], int)
        assert isinstance(proc["cpu_percent"], float)
        assert math.isfinite(proc["cpu_percent"])
        assert isinstance(proc["rss_mb"], float)
        assert math.isfinite(proc["rss_mb"])
        assert proc["rss_mb"] >= 0.0


def test_get_system_snapshot_cpu_per_core_list():
    snap = get_system_snapshot()
    cpu_per_core = snap["cpu_per_core"]
    assert isinstance(cpu_per_core, list)
    assert len(cpu_per_core) >= 1
    for value in cpu_per_core:
        assert isinstance(value, float)
        assert math.isfinite(value)
        assert value >= 0.0
