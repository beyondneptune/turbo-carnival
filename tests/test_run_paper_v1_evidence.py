"""Tests for `scripts/run_paper_v1_evidence.py` -- the paper_v1 evidence orchestrator.

Strategy: import the script as a module, drive it end-to-end with synthetic
data via `--allow-empty-cache`. Each test gets its own tmp_path-backed
SQLite file initialised from `sql/schema.sql` so we exercise real schema
constraints (FKs, CHECK clauses).

No real network: the orchestrator never calls Polygon. Synthetic data is
generated deterministically inside the module under test.
"""
from __future__ import annotations

import importlib
import json
import sqlite3
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCHEMA_PATH = REPO_ROOT / "sql" / "schema.sql"


def _import_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    if "run_paper_v1_evidence" in sys.modules:
        return importlib.reload(sys.modules["run_paper_v1_evidence"])
    return importlib.import_module("run_paper_v1_evidence")


@pytest.fixture
def mod():
    return _import_module()


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    """Empty parquet path that becomes a schema-initialised DB on first open()."""
    return tmp_path / "trading_system.db"


@pytest.fixture
def tmp_status(tmp_path) -> Path:
    return tmp_path / "paper_v1_status.json"


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults(mod):
    ns = mod.parse_args([])
    assert ns.cache_dir == mod.DEFAULT_CACHE_DIR
    assert ns.db_path == mod.DEFAULT_DB_PATH
    assert ns.run_id is None
    assert ns.allow_empty_cache is False
    assert ns.cost_stress_multiplier == 1.0


def test_parse_args_custom(mod, tmp_path):
    ns = mod.parse_args(
        [
            "--cache-dir", str(tmp_path / "cache"),
            "--db-path", str(tmp_path / "evidence.db"),
            "--run-id", "custom_run",
            "--allow-empty-cache",
            "--cost-stress-multiplier", "1.25",
            "--seed", "11",
        ]
    )
    assert ns.cache_dir == tmp_path / "cache"
    assert ns.db_path == tmp_path / "evidence.db"
    assert ns.run_id == "custom_run"
    assert ns.allow_empty_cache is True
    assert ns.cost_stress_multiplier == pytest.approx(1.25)
    assert ns.seed == 11


def test_help_exits_zero(mod):
    with pytest.raises(SystemExit) as exc:
        mod.parse_args(["--help"])
    assert exc.value.code == 0


# ---------------------------------------------------------------------------
# StatusWriter / atomic JSON
# ---------------------------------------------------------------------------


def test_status_writer_creates_file_with_required_keys(mod, tmp_status):
    writer = mod.StatusWriter(tmp_status, run_id="test_run")
    assert tmp_status.exists()
    doc = json.loads(tmp_status.read_text())
    for key in ("run_id", "phase", "started_at_utc", "last_update_utc", "stages", "errors"):
        assert key in doc
    assert doc["run_id"] == "test_run"
    assert doc["phase"] == "loading"
    # No leftover .tmp file (atomic write).
    assert not tmp_status.with_suffix(tmp_status.suffix + ".tmp").exists()
    # Avoid unused-variable lint by exercising the writer.
    writer.set_phase("loading")


def test_status_writer_merges_stages_and_atomic_write(mod, tmp_status):
    writer = mod.StatusWriter(tmp_status, run_id="r")
    writer.set_stage("stage_a", "running", events_total=5)
    writer.update_stage("stage_a", events_processed=2, trades_placed=1)
    writer.set_stage("stage_b", "complete", events=42)

    doc = json.loads(tmp_status.read_text())
    assert set(doc["stages"].keys()) == {"stage_a", "stage_b"}
    a = doc["stages"]["stage_a"]
    assert a["status"] == "running"
    assert a["events_total"] == 5
    assert a["events_processed"] == 2
    assert a["trades_placed"] == 1
    assert doc["stages"]["stage_b"]["status"] == "complete"
    assert doc["stages"]["stage_b"]["events"] == 42


def test_status_writer_errors_appended(mod, tmp_status):
    writer = mod.StatusWriter(tmp_status, run_id="r")
    writer.add_error("bad thing happened", traceback="trace1")
    writer.add_error("another error")
    doc = json.loads(tmp_status.read_text())
    assert isinstance(doc["errors"], list) and len(doc["errors"]) == 2
    assert doc["errors"][0]["message"] == "bad thing happened"
    assert doc["errors"][0]["traceback"] == "trace1"


# ---------------------------------------------------------------------------
# Synthetic cache + load
# ---------------------------------------------------------------------------


def test_synthesize_year_cache_year1_has_events(mod):
    cache = mod.synthesize_year_cache("year1", seed=1)
    assert cache["year"] == "year1"
    assert cache["window"] == (mod.YEAR1_START, mod.YEAR1_END)
    assert isinstance(cache["spy_daily"], pd.DataFrame) and not cache["spy_daily"].empty
    assert isinstance(cache["spy_iv_daily"], pd.DataFrame)
    # 50 tickers * 4 events default = 200 events
    n_events = sum(len(v) for v in cache["earnings_dates"].values())
    assert n_events >= 50  # comfortably above the sample-size gate floor
    assert len(cache["chains"]) == n_events
    # Each chain must have at least call+put rows with strike/expiry/option_type.
    sample = next(iter(cache["chains"].values()))
    for col in ("strike", "expiry", "option_type", "bid", "ask", "mid"):
        assert col in sample.columns


def test_synthesize_year_cache_year2_distinct_from_year1(mod):
    y1 = mod.synthesize_year_cache("year1", seed=1)
    y2 = mod.synthesize_year_cache("year2", seed=1)
    # Different windows.
    assert y1["window"] != y2["window"]
    # SPY daily indexed by distinct date ranges.
    y1_dates = {pd.Timestamp(d).date() for d in y1["spy_daily"].index}
    y2_dates = {pd.Timestamp(d).date() for d in y2["spy_daily"].index}
    assert y1_dates.isdisjoint(y2_dates)


def test_load_year_cache_with_synthetic_files(mod, tmp_path):
    """Round-trip: write synthetic parquets to disk, then load_year_cache reads."""
    # Write a tiny "year1" parquet layout.
    yd = tmp_path / "year1"
    (yd / "earnings_dates").mkdir(parents=True)
    (yd / "chains" / "AAPL").mkdir(parents=True)
    pd.DataFrame({"earnings_date": [date(2024, 8, 1)]}).to_parquet(
        yd / "earnings_dates" / "AAPL.parquet", index=False
    )
    sample = pd.DataFrame(
        {
            "strike": [100.0, 100.0],
            "expiry": [date(2024, 8, 9), date(2024, 8, 9)],
            "option_type": ["call", "put"],
            "bid": [1.0, 1.0],
            "ask": [1.1, 1.1],
            "mid": [1.05, 1.05],
            "underlying_close": [100.0, 100.0],
            "atm_iv": [0.30, 0.30],
        }
    )
    sample.to_parquet(yd / "chains" / "AAPL" / "2024-08-01.parquet", index=False)

    cache = mod.load_year_cache(tmp_path, "year1", universe=("AAPL",))
    assert cache["earnings_dates"]["AAPL"] == [date(2024, 8, 1)]
    assert ("AAPL", date(2024, 8, 1)) in cache["chains"]


def test_cache_summary_zero_when_empty(mod, tmp_path):
    empty = mod.load_year_cache(tmp_path, "year1", universe=("AAPL",))
    summary = mod.cache_summary(empty)
    assert summary["spy_daily_rows"] == 0
    assert summary["earnings_events"] == 0
    assert summary["chain_files"] == 0


# ---------------------------------------------------------------------------
# Feature extraction + event building
# ---------------------------------------------------------------------------


def test_build_earnings_events_returns_sorted(mod):
    cache = mod.synthesize_year_cache("year2", seed=2)
    events = mod.build_earnings_events(cache)
    assert len(events) >= 10
    # Sorted by event_date then ticker.
    pairs = [(e.event_date, e.ticker) for e in events]
    assert pairs == sorted(pairs)
    sample = events[0]
    assert sample.spot_at_decision > 0
    assert sample.pre_event_atm_iv_5d > 0
    assert sample.nearest_expiry > sample.event_date


def test_build_feature_row_returns_expected_columns(mod):
    cache = mod.synthesize_year_cache("year1", seed=3)
    events = mod.build_earnings_events(cache)
    assert events
    ev = events[0]
    chain = cache["chains"][(ev.ticker, ev.event_date)]
    feats = mod.build_feature_row(ev, chain)
    expected = {"pre_event_atm_iv_5d", "log_spot", "dte", "atm_premium", "chain_width"}
    assert set(feats.keys()) == expected
    for v in feats.values():
        assert np.isfinite(v)


# ---------------------------------------------------------------------------
# Year-1 training pipeline
# ---------------------------------------------------------------------------


def test_build_year1_training_frame_shapes(mod):
    cache = mod.synthesize_year_cache("year1", seed=4)
    X, y_a, y_b, realized = mod.build_year1_training_frame(cache)
    n = len(X)
    assert n > 50
    assert len(y_a) == len(y_b) == len(realized) == n
    assert set(X.columns) == {"pre_event_atm_iv_5d", "log_spot", "dte", "atm_premium", "chain_width"}
    assert np.all(np.isfinite(y_a))
    assert np.all(np.isfinite(y_b))
    assert np.all(np.isfinite(realized))


# ---------------------------------------------------------------------------
# DB helpers (open + insert)
# ---------------------------------------------------------------------------


def test_open_db_initialises_schema(mod, tmp_db):
    conn = mod.open_db(tmp_db)
    try:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"runs", "gate_decisions", "trades", "trade_legs", "agents_log"} <= names
    finally:
        conn.close()


def test_insert_run_row_roundtrip(mod, tmp_db):
    conn = mod.open_db(tmp_db)
    try:
        mod.insert_run_row(conn, "run_a")
        row = conn.execute(
            "SELECT run_id, run_type, status, initial_nav_usd FROM runs WHERE run_id = 'run_a'"
        ).fetchone()
    finally:
        conn.close()
    assert row[0] == "run_a"
    assert row[1] == "paper_frozen"
    assert row[2] == "running"
    assert float(row[3]) == pytest.approx(mod.INITIAL_NAV_USD)


def test_insert_gate_decision_and_trade(mod, tmp_db):
    conn = mod.open_db(tmp_db)
    try:
        mod.insert_run_row(conn, "run_b")
        decision_id = mod.insert_gate_decision(
            conn,
            run_id="run_b",
            decision_at_utc="2026-01-01T14:30:00+00:00",
            underlying="AAPL",
            sleeve="satellite",
            candidate_summary="ENTRY:long_straddle AAPL test",
            risk_pass=True,
            block_reason=None,
            final_decision="approved",
            model_version="lightgbm_test",
            signal_score=1.23,
            signal_threshold=0.0,
        )
        legs = [
            {
                "leg_number": 1, "instrument_type": "call", "direction": "long",
                "quantity": 1, "multiplier": 100, "strike": 150.0, "expiry": "2026-01-09",
                "entry_intended_price": 1.0, "entry_fill_price": 1.05,
                "exit_intended_price": 1.0, "exit_fill_price": 1.0,
            },
            {
                "leg_number": 2, "instrument_type": "put", "direction": "long",
                "quantity": 1, "multiplier": 100, "strike": 150.0, "expiry": "2026-01-09",
                "entry_intended_price": 1.0, "entry_fill_price": 1.04,
                "exit_intended_price": 1.0, "exit_fill_price": 1.0,
            },
        ]
        trade_id = mod.insert_trade_with_legs(
            conn,
            run_id="run_b",
            decision_id=decision_id,
            sleeve="satellite",
            strategy_type="long_straddle",
            underlying="AAPL",
            entry_at_utc="2026-01-01T14:30:00+00:00",
            exit_at_utc="2026-01-09T20:00:00+00:00",
            legs=legs,
            entry_greeks={"delta": 0.0, "vega": 1.0, "theta": -1.0},
            commission_usd=1.30,
            spread_cost_usd=0.05,
            slippage_usd=0.02,
            realized_pnl_usd=12.34,
            notes="test",
        )
        n_legs = conn.execute(
            "SELECT COUNT(*) FROM trade_legs WHERE trade_id = ?", (trade_id,)
        ).fetchone()[0]
        status = conn.execute(
            "SELECT status, realized_pnl_usd FROM trades WHERE trade_id = ?", (trade_id,)
        ).fetchone()
    finally:
        conn.close()
    assert n_legs == 2
    assert status[0] == "closed"
    assert float(status[1]) == pytest.approx(12.34)


# ---------------------------------------------------------------------------
# End-to-end main(...) with synthetic cache
# ---------------------------------------------------------------------------


def test_main_end_to_end_with_allow_empty_cache(mod, tmp_path):
    db_path = tmp_path / "evidence.db"
    status_path = tmp_path / "status.json"
    cache_dir = tmp_path / "cache"
    rc = mod.main(
        [
            "--allow-empty-cache",
            "--db-path", str(db_path),
            "--cache-dir", str(cache_dir),
            "--status-path", str(status_path),
            "--run-id", "test_evidence",
        ]
    )
    assert rc == 0
    assert status_path.exists()
    doc = json.loads(status_path.read_text())
    assert doc["run_id"] == "test_evidence"
    assert doc["phase"] == "complete"
    assert isinstance(doc["stages"], dict) and doc["stages"]
    # Expected stages.
    for stage_name in (
        "load_year1_cache", "load_year2_cache", "train_models", "year2_replay", "summary"
    ):
        assert stage_name in doc["stages"], f"missing stage {stage_name}"
    # Errors list present and empty.
    assert doc.get("errors") == []

    # DB rows.
    conn = sqlite3.connect(db_path)
    try:
        n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
        n_decisions = conn.execute("SELECT COUNT(*) FROM gate_decisions").fetchone()[0]
        n_trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        n_legs = conn.execute("SELECT COUNT(*) FROM trade_legs").fetchone()[0]
        n_agents = conn.execute("SELECT COUNT(*) FROM agents_log").fetchone()[0]
        run_status = conn.execute(
            "SELECT status FROM runs WHERE run_id = 'test_evidence'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert n_runs == 1
    assert n_decisions > 0
    assert n_trades > 0
    assert n_legs >= n_trades * 2  # straddles have 2 legs, condors 4
    assert n_agents >= 2  # at least coder + verifier logs
    assert run_status == "completed"


def test_main_status_has_twin_model_comparison_keys(mod, tmp_path):
    rc = mod.main(
        [
            "--allow-empty-cache",
            "--db-path", str(tmp_path / "evidence.db"),
            "--cache-dir", str(tmp_path / "cache"),
            "--status-path", str(tmp_path / "status.json"),
            "--run-id", "test_twin",
        ]
    )
    assert rc == 0
    doc = json.loads((tmp_path / "status.json").read_text())
    twin = doc.get("twin_model_comparison")
    assert isinstance(twin, dict)
    for key in (
        "net_excess_bps_ann",
        "sharpe_delta",
        "paired_t_stat",
        "paired_p_value",
        "b_beats_a",
        "n_events",
    ):
        assert key in twin
    # Numeric sanity.
    assert isinstance(twin["b_beats_a"], bool)
    assert isinstance(twin["n_events"], int)


def test_main_status_has_gate_evaluation_shape(mod, tmp_path):
    rc = mod.main(
        [
            "--allow-empty-cache",
            "--db-path", str(tmp_path / "evidence.db"),
            "--cache-dir", str(tmp_path / "cache"),
            "--status-path", str(tmp_path / "status.json"),
            "--run-id", "test_gates",
        ]
    )
    assert rc == 0
    doc = json.loads((tmp_path / "status.json").read_text())
    gates = doc.get("gates")
    assert isinstance(gates, dict)
    assert "base" in gates and "plus50" in gates
    assert "overall_pass" in gates
    assert "failed_gates_summary" in gates
    # Each regime must carry a `gates` mapping with all 7 gate names.
    expected_gates = {
        "dsr", "pbo", "net_excess_bps", "max_drawdown",
        "adversarial_stress", "regime_stability", "sample_size",
    }
    for regime in ("base", "plus50"):
        sub = gates[regime]
        assert "gates" in sub
        assert set(sub["gates"].keys()) == expected_gates
        for gate in sub["gates"].values():
            for key in ("name", "passed", "observed", "threshold", "detail"):
                assert key in gate


def test_main_idempotency_blocks_duplicate_run_id(mod, tmp_path):
    """Re-running with the same run_id must fail (UNIQUE on runs.run_id)."""
    args = [
        "--allow-empty-cache",
        "--db-path", str(tmp_path / "evidence.db"),
        "--cache-dir", str(tmp_path / "cache"),
        "--status-path", str(tmp_path / "status.json"),
        "--run-id", "dup_run",
    ]
    rc1 = mod.main(args)
    assert rc1 == 0
    rc2 = mod.main(args)
    assert rc2 != 0  # IntegrityError


def test_main_status_json_is_valid_json(mod, tmp_path):
    rc = mod.main(
        [
            "--allow-empty-cache",
            "--db-path", str(tmp_path / "evidence.db"),
            "--cache-dir", str(tmp_path / "cache"),
            "--status-path", str(tmp_path / "status.json"),
            "--run-id", "test_json",
        ]
    )
    assert rc == 0
    raw = (tmp_path / "status.json").read_text()
    doc = json.loads(raw)
    # Required top-level keys per the dashboard contract.
    for key in (
        "run_id", "phase", "started_at_utc", "last_update_utc",
        "stages", "twin_model_comparison", "gates", "errors",
    ):
        assert key in doc, f"missing key {key}"


def test_main_status_dashboard_contract_year2_replay_progress(mod, tmp_path):
    rc = mod.main(
        [
            "--allow-empty-cache",
            "--db-path", str(tmp_path / "evidence.db"),
            "--cache-dir", str(tmp_path / "cache"),
            "--status-path", str(tmp_path / "status.json"),
            "--run-id", "test_replay",
        ]
    )
    assert rc == 0
    doc = json.loads((tmp_path / "status.json").read_text())
    replay = doc["stages"]["year2_replay"]
    # Dashboard reads these fields to render the progress bar.
    for field in ("status", "events_total", "events_processed", "trades_placed", "trades_blocked"):
        assert field in replay
    assert replay["events_processed"] == replay["events_total"]
    assert replay["status"] == "complete"


# ---------------------------------------------------------------------------
# Helper: build_cost_regime_inputs sanity
# ---------------------------------------------------------------------------


def test_build_cost_regime_inputs_handles_empty(mod):
    inp = mod.build_cost_regime_inputs(
        strategy_returns=np.array([]),
        underlying_moves=np.array([]),
        pre_event_iv=np.array([]),
        features_for_b3=pd.DataFrame(columns=["pre_event_atm_iv_5d", "log_spot", "atm_premium"]),
        cost_multiplier=1.0,
    )
    assert inp.n_events == 0
    assert inp.dsr == 0.0
    assert inp.pbo == 1.0


def test_build_cost_regime_inputs_with_data(mod):
    rng = np.random.default_rng(0)
    n = 200
    strat = rng.normal(0.005, 0.03, n)
    moves = rng.normal(0.0, 0.04, n)
    iv = np.clip(0.30 + 0.05 * rng.standard_normal(n), 0.10, 0.60)
    feats = pd.DataFrame(
        {
            "pre_event_atm_iv_5d": iv,
            "log_spot": np.log(np.linspace(50.0, 250.0, n)),
            "atm_premium": np.abs(rng.normal(2.0, 0.5, n)),
        }
    )
    inp = mod.build_cost_regime_inputs(
        strategy_returns=strat,
        underlying_moves=moves,
        pre_event_iv=iv,
        features_for_b3=feats,
        cost_multiplier=1.5,
    )
    assert inp.n_events == n
    assert np.isfinite(inp.dsr)
    assert 0.0 <= inp.pbo <= 1.0
