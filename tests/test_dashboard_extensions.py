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

from dashboard.app import (
    count_satellite_events,
    fetch_run_equity_curve,
    get_system_snapshot,
    is_status_stale,
    read_paper_v1_status,
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
            "candidate_summary, signal_pass, risk_spread_pass, risk_vega_pass, risk_scenario_pass, "
            "risk_liquidity_pass, risk_pin_pass, risk_concentration_pass, final_decision) "
            "VALUES (?, ?, ?, ?, ?, 1, 1, 1, 1, 1, 1, 1, 'approved')",
            (run_id, f"2026-05-07T04:{idx:02d}:00Z", "SPY", sleeve, "test"),
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
