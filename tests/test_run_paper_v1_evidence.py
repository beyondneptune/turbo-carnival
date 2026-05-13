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
    assert ns.allow_synthetic is False  # strict / fail-closed by default
    assert ns.cost_stress_multiplier == 1.0


def test_parse_args_allow_synthetic_flag(mod):
    """The new opt-in --allow-synthetic flag must be off by default and
    settable on the command line."""
    ns = mod.parse_args(["--allow-synthetic"])
    assert ns.allow_synthetic is True
    assert ns.allow_empty_cache is False


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
    """Round-trip: write a legacy chains/ parquet to disk and load it.

    Strict mode drops the metadata-only event; this test exercises the
    `--allow-synthetic` fallback that the smoke-test cohort uses.
    """
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

    cache = mod.load_year_cache(
        tmp_path, "year1", universe=("AAPL",), allow_synthetic=True
    )
    assert cache["earnings_dates"]["AAPL"] == [date(2024, 8, 1)]
    assert ("AAPL", date(2024, 8, 1)) in cache["chains"]
    # Source is metadata since it sits in chains/ rather than chains_priced/.
    assert cache["chain_sources"][("AAPL", date(2024, 8, 1))] == "metadata"


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


# ---------------------------------------------------------------------------
# Priced cache wiring (chains_priced/) -- new tests for fail-closed behavior
# ---------------------------------------------------------------------------


def _build_priced_parquet_rows(
    *,
    ticker: str,
    event_date: date,
    spot: float = 100.0,
    iv: float = 0.30,
    expiry: date | None = None,
    strikes: tuple[float, ...] = (95.0, 100.0, 105.0),
    atm_strike: float = 100.0,
    atm_iv_value: float | None = None,
    atm_mid_value: float | None = None,
    include_exit: bool = True,
) -> pd.DataFrame:
    """Build a long-format priced parquet matching the augment-script schema.

    Columns: ticker, underlying_ticker, contract_type, strike_price,
    expiration_date, as_of_date, timing, event_date, bid, ask, mid,
    underlying_close, is_atm, atm_iv.
    """
    expiry = expiry or (event_date + pd.Timedelta(days=9)).date() if isinstance(
        event_date + pd.Timedelta(days=9), pd.Timestamp
    ) else (expiry or pd.Timestamp(event_date).date())
    if expiry is None:
        expiry = (pd.Timestamp(event_date) + pd.Timedelta(days=9)).date()
    rows: list[dict] = []
    timings = ("entry", "exit") if include_exit else ("entry",)
    for timing in timings:
        as_of = event_date if timing == "entry" else (
            (pd.Timestamp(event_date) + pd.Timedelta(days=5)).date()
        )
        for K in strikes:
            for ctype in ("call", "put"):
                contract = (
                    f"O:{ticker}{event_date.strftime('%y%m%d')}"
                    f"{ctype[0].upper()}{int(K * 1000):08d}"
                )
                # Synthetic-ish ATM-aware mid.
                base_mid = max(0.05, abs(spot - K) + 1.0)
                is_atm = K == atm_strike
                # Optional injection of NaN at the ATM strike.
                if is_atm and timing == "entry" and atm_mid_value is not None:
                    mid = atm_mid_value
                else:
                    mid = base_mid
                if is_atm and atm_iv_value is not None:
                    row_iv = atm_iv_value
                else:
                    row_iv = iv
                spread = max(0.05, mid * 0.02) if mid is not None and not (
                    isinstance(mid, float) and np.isnan(mid)
                ) else 0.05
                bid = (mid - spread / 2.0) if mid is not None and not (
                    isinstance(mid, float) and np.isnan(mid)
                ) else float("nan")
                ask = (mid + spread / 2.0) if mid is not None and not (
                    isinstance(mid, float) and np.isnan(mid)
                ) else float("nan")
                rows.append(
                    {
                        "ticker": contract,
                        "underlying_ticker": ticker,
                        "contract_type": ctype,
                        "strike_price": float(K),
                        "expiration_date": expiry.isoformat(),
                        "as_of_date": as_of.isoformat(),
                        "timing": timing,
                        "event_date": event_date.isoformat(),
                        "bid": bid,
                        "ask": ask,
                        "mid": mid,
                        "underlying_close": float(spot),
                        "is_atm": bool(is_atm),
                        "atm_iv": float(row_iv) if row_iv is not None else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def _write_priced_parquet(
    base_dir: Path,
    year: str,
    *,
    ticker: str,
    event_date: date,
    **kwargs,
) -> Path:
    """Write a priced parquet under chains_priced/ and return the path."""
    target_dir = base_dir / year / "chains_priced" / ticker
    target_dir.mkdir(parents=True, exist_ok=True)
    df = _build_priced_parquet_rows(ticker=ticker, event_date=event_date, **kwargs)
    out = target_dir / f"{event_date.isoformat()}.parquet"
    df.to_parquet(out, index=False)
    return out


def _write_metadata_parquet(
    base_dir: Path,
    year: str,
    *,
    ticker: str,
    event_date: date,
) -> Path:
    """Write a legacy metadata-only parquet under chains/."""
    target_dir = base_dir / year / "chains" / ticker
    target_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "ticker": [
                f"O:{ticker}{event_date.strftime('%y%m%d')}C00100000",
                f"O:{ticker}{event_date.strftime('%y%m%d')}P00100000",
            ],
            "underlying_ticker": [ticker, ticker],
            "contract_type": ["call", "put"],
            "strike_price": [100.0, 100.0],
            "expiration_date": [
                (pd.Timestamp(event_date) + pd.Timedelta(days=9)).date().isoformat(),
                (pd.Timestamp(event_date) + pd.Timedelta(days=9)).date().isoformat(),
            ],
            "as_of_date": [event_date.isoformat(), event_date.isoformat()],
            "event_date": [event_date.isoformat(), event_date.isoformat()],
        }
    )
    out = target_dir / f"{event_date.isoformat()}.parquet"
    df.to_parquet(out, index=False)
    return out


def _write_earnings_dates(
    base_dir: Path,
    year: str,
    *,
    ticker: str,
    dates: list[date],
) -> None:
    target_dir = base_dir / year / "earnings_dates"
    target_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"earnings_date": dates}).to_parquet(
        target_dir / f"{ticker}.parquet", index=False
    )


def test_load_year_cache_prefers_priced_over_metadata(mod, tmp_path):
    """When both a metadata parquet and a priced parquet exist for an event,
    the loader picks the priced one and tags it source='priced'."""
    ev = date(2025, 7, 15)
    _write_earnings_dates(tmp_path, "year2", ticker="AAPL", dates=[ev])
    _write_metadata_parquet(tmp_path, "year2", ticker="AAPL", event_date=ev)
    _write_priced_parquet(tmp_path, "year2", ticker="AAPL", event_date=ev)

    cache = mod.load_year_cache(
        tmp_path, "year2", universe=("AAPL",), allow_synthetic=False
    )
    key = ("AAPL", ev)
    assert key in cache["chains"]
    assert cache["chain_sources"][key] == "priced"
    assert cache["events_dropped"] == []
    chain = cache["chains"][key]
    # Normalized columns should be present.
    for col in ("strike", "expiry", "option_type", "bid", "ask", "mid", "atm_iv"):
        assert col in chain.columns, f"missing col {col}"
    assert "exit_mid" in chain.columns
    assert np.isfinite(pd.to_numeric(chain["exit_mid"], errors="coerce")).any()


def test_build_year1_training_frame_uses_real_exit_mids_in_strict_mode(mod):
    """Strict training targets come from priced entry/exit mids, not RNG."""
    ev = date(2024, 8, 1)
    raw = _build_priced_parquet_rows(
        ticker="AAPL",
        event_date=ev,
        expiry=(pd.Timestamp(ev) + pd.Timedelta(days=9)).date(),
        strikes=(100.0,),
        atm_strike=100.0,
    )
    raw.loc[raw["timing"] == "entry", "mid"] = 1.0
    raw.loc[raw["timing"] == "exit", "mid"] = 2.0
    chain = mod._normalize_priced_chain(raw)
    key = ("AAPL", ev)
    cache = {
        "year": "year1",
        "window": (mod.YEAR1_START, mod.YEAR1_END),
        "spy_daily": pd.DataFrame(),
        "spy_iv_daily": pd.DataFrame(),
        "earnings_dates": {"AAPL": [ev]},
        "chains": {key: chain},
        "chain_sources": {key: "priced"},
        "events_dropped": [],
    }

    X, y_a, y_b, realized = mod.build_year1_training_frame(
        cache,
        rng=np.random.default_rng(123),
        allow_synthetic=False,
    )

    assert len(X) == 1
    assert realized[0] == pytest.approx(1.0)
    assert y_a[0] == pytest.approx(1.0)
    assert np.isfinite(y_b[0])


def test_load_year_cache_falls_back_to_metadata_when_allow_synthetic(mod, tmp_path):
    """In allow-synthetic mode, a metadata-only event is loaded with
    source='metadata' so the orchestrator can still smoke-test plumbing."""
    ev = date(2025, 7, 15)
    _write_earnings_dates(tmp_path, "year2", ticker="AAPL", dates=[ev])
    _write_metadata_parquet(tmp_path, "year2", ticker="AAPL", event_date=ev)
    # No chains_priced/AAPL/2025-07-15.parquet -> must fall back.

    cache = mod.load_year_cache(
        tmp_path, "year2", universe=("AAPL",), allow_synthetic=True
    )
    key = ("AAPL", ev)
    assert key in cache["chains"]
    assert cache["chain_sources"][key] == "metadata"
    assert cache["events_dropped"] == []


def test_load_year_cache_strict_drops_event_when_only_metadata(mod, tmp_path):
    """Strict mode (default): no chains_priced/ parquet => event dropped from
    the universe and recorded in events_dropped with reason='no_pricing'."""
    ev = date(2025, 7, 15)
    _write_earnings_dates(tmp_path, "year2", ticker="AAPL", dates=[ev])
    _write_metadata_parquet(tmp_path, "year2", ticker="AAPL", event_date=ev)

    cache = mod.load_year_cache(
        tmp_path, "year2", universe=("AAPL",), allow_synthetic=False
    )
    key = ("AAPL", ev)
    assert key not in cache["chains"]
    assert key not in cache["chain_sources"]
    drop_keys = [(d["ticker"], d["event_date"]) for d in cache["events_dropped"]]
    assert ("AAPL", ev) in drop_keys
    reasons = {d["reason"] for d in cache["events_dropped"] if d["ticker"] == "AAPL"}
    assert "no_pricing" in reasons


def test_load_year_cache_drops_priced_with_nan_atm_iv(mod, tmp_path):
    """A priced parquet with NaN atm_iv at the ATM strike must be dropped
    with reason='no_atm_iv' even though the file exists."""
    ev = date(2025, 7, 15)
    _write_earnings_dates(tmp_path, "year2", ticker="AAPL", dates=[ev])
    _write_priced_parquet(
        tmp_path, "year2", ticker="AAPL", event_date=ev,
        atm_iv_value=float("nan"),
    )

    cache = mod.load_year_cache(
        tmp_path, "year2", universe=("AAPL",), allow_synthetic=False
    )
    assert ("AAPL", ev) not in cache["chains"]
    drop = [d for d in cache["events_dropped"] if d["ticker"] == "AAPL"]
    assert drop and drop[0]["reason"] == "no_atm_iv"


def test_load_year_cache_drops_priced_with_nan_entry_mid(mod, tmp_path):
    """A priced parquet with NaN mid at the ATM strike triggers
    drop reason='no_entry_price'."""
    ev = date(2025, 7, 15)
    _write_earnings_dates(tmp_path, "year2", ticker="AAPL", dates=[ev])
    _write_priced_parquet(
        tmp_path, "year2", ticker="AAPL", event_date=ev,
        atm_mid_value=float("nan"),
    )

    cache = mod.load_year_cache(
        tmp_path, "year2", universe=("AAPL",), allow_synthetic=False
    )
    assert ("AAPL", ev) not in cache["chains"]
    drop = [d for d in cache["events_dropped"] if d["ticker"] == "AAPL"]
    assert drop and drop[0]["reason"] == "no_entry_price"


def test_synthetic_spy_chains_auto_disables_when_priced_exists(mod, tmp_path):
    """When chains_priced/SPY/<date>.parquet exists for the year and
    --allow-synthetic is off, _resolve_synthetic_spy_chains must return False."""
    spy_dir = tmp_path / "year2" / "chains_priced" / "SPY"
    spy_dir.mkdir(parents=True)
    # Create one parquet so the dir is non-empty.
    df = _build_priced_parquet_rows(
        ticker="SPY",
        event_date=date(2025, 7, 14),
        spot=500.0,
        iv=0.16,
        strikes=(495.0, 500.0, 505.0),
        atm_strike=500.0,
    )
    df.to_parquet(spy_dir / "2025-07-14.parquet", index=False)

    assert (
        mod._resolve_synthetic_spy_chains(tmp_path, "year2", allow_synthetic=False)
        is False
    )
    # --allow-synthetic must NOT override real priced files. Real data wins.
    # Otherwise the verdict silently substitutes synthetic SPY for real prices,
    # bypassing the SPY-condor validator and producing zero priced core trades.
    assert (
        mod._resolve_synthetic_spy_chains(tmp_path, "year2", allow_synthetic=True)
        is False
    )


def test_synthetic_spy_chains_default_when_no_priced_dir(mod, tmp_path):
    """No chains_priced/SPY/ directory => synthetic_spy_chains stays True."""
    assert (
        mod._resolve_synthetic_spy_chains(tmp_path, "year2", allow_synthetic=False)
        is True
    )


def test_replay_year2_accepts_priced_spy_condor_without_atm_rows(mod, tmp_path):
    """Regression: SPY core condor files are OTM legs, not ATM straddles."""
    entry = date(2025, 7, 14)  # Monday
    expiry = date(2025, 8, 22)
    spy_dir = tmp_path / "spy_priced"
    spy_dir.mkdir()

    rows = []
    legs = (
        ("put", 490.0, 1.20, 0.10),
        ("put", 495.0, 1.60, 0.20),
        ("call", 530.0, 1.50, 0.25),
        ("call", 535.0, 1.10, 0.10),
    )
    for timing in ("entry", "exit"):
        for contract_type, strike, entry_mid, exit_mid in legs:
            mid = entry_mid if timing == "entry" else exit_mid
            rows.append(
                {
                    "ticker": f"O:SPY250822{contract_type[0].upper()}{int(strike * 1000):08d}",
                    "underlying_ticker": "SPY",
                    "contract_type": contract_type,
                    "strike_price": strike,
                    "expiration_date": expiry.isoformat(),
                    "as_of_date": (entry if timing == "entry" else expiry).isoformat(),
                    "timing": timing,
                    "event_date": entry.isoformat(),
                    "bid": mid - 0.01,
                    "ask": mid + 0.01,
                    "mid": mid,
                    "underlying_close": 512.0,
                    "is_atm": False,
                    "atm_iv": 0.18,
                }
            )
    pd.DataFrame(rows).to_parquet(spy_dir / f"{entry.isoformat()}.parquet", index=False)

    cache = {
        "year": "year2",
        "window": (entry, entry),
        "spy_daily": pd.DataFrame(
            {"close": [512.0]},
            index=pd.DatetimeIndex([entry], name="date"),
        ),
        "spy_iv_daily": pd.DataFrame(),
        "earnings_dates": {},
        "chains": {},
        "chain_sources": {},
        "events_dropped": [],
    }

    conn = mod.open_db(tmp_path / "evidence.db")
    try:
        mod.insert_run_row(conn, "run_spy_core")
        status = mod.StatusWriter(tmp_path / "status.json", run_id="run_spy_core")
        replay = mod.replay_year2(
            conn=conn,
            run_id="run_spy_core",
            cache=cache,
            status=status,
            rng=np.random.default_rng(1),
            synthetic_spy_chains=False,
            allow_synthetic=False,
            spy_priced_dir=spy_dir,
        )
        trades = conn.execute(
            "SELECT sleeve, strategy_type, underlying_ticker FROM trades"
        ).fetchall()
    finally:
        conn.close()

    assert replay["trades_placed"] == 1
    assert replay["trade_source_counts"]["priced"] == 1
    assert replay["events_dropped_no_pricing"] == 0
    assert trades == [("core", "iron_condor", "SPY")]


def test_strict_mode_year2_replay_drops_events_without_pricing(mod, tmp_path):
    """Mixed cache: priced + metadata events. Strict mode loads only priced
    events; metadata-only events are dropped and the year2_replay stage's
    events_dropped_no_pricing counter is non-zero."""
    cache_dir = tmp_path / "cache"

    # Year2 only has 1 priced + 1 metadata event; we focus the strict-mode
    # behaviour on Year 2 by inspecting the loader's output directly (we don't
    # need to invoke main() here; a separate end-to-end test covers that).
    y2_priced = date(2025, 7, 15)
    y2_meta = date(2025, 8, 12)
    _write_earnings_dates(cache_dir, "year2", ticker="AAPL", dates=[y2_priced])
    _write_earnings_dates(cache_dir, "year2", ticker="MSFT", dates=[y2_meta])
    _write_priced_parquet(
        cache_dir, "year2", ticker="AAPL", event_date=y2_priced,
    )
    _write_metadata_parquet(cache_dir, "year2", ticker="MSFT", event_date=y2_meta)

    cache = mod.load_year_cache(
        cache_dir, "year2", universe=("AAPL", "MSFT"), allow_synthetic=False
    )
    # Priced AAPL loaded; metadata-only MSFT dropped.
    assert ("AAPL", y2_priced) in cache["chains"]
    assert ("MSFT", y2_meta) not in cache["chains"]
    assert cache["chain_sources"][("AAPL", y2_priced)] == "priced"
    drop_pairs = {(d["ticker"], d["event_date"]) for d in cache["events_dropped"]}
    assert ("MSFT", y2_meta) in drop_pairs
    # Status JSON: events_dropped_no_pricing must surface to the loader summary.
    summary = mod.cache_summary(cache)
    assert summary["events_dropped_no_pricing"] >= 1


def test_replay_year2_uses_model_signal_to_block_satellite_trade(mod, tmp_path):
    """Year 2 replay must use the frozen model prediction, not premium alone."""
    ev = date(2025, 7, 15)
    raw = _build_priced_parquet_rows(
        ticker="AAPL",
        event_date=ev,
        expiry=(pd.Timestamp(ev) + pd.Timedelta(days=9)).date(),
        strikes=(100.0,),
        atm_strike=100.0,
    )
    raw.loc[raw["timing"] == "entry", "mid"] = 1.0
    raw.loc[raw["timing"] == "exit", "mid"] = 2.0
    chain = mod._normalize_priced_chain(raw)
    key = ("AAPL", ev)
    cache = {
        "year": "year2",
        "window": (mod.YEAR2_START, mod.YEAR2_END),
        "spy_daily": pd.DataFrame(),
        "spy_iv_daily": pd.DataFrame(),
        "earnings_dates": {"AAPL": [ev]},
        "chains": {key: chain},
        "chain_sources": {key: "priced"},
        "events_dropped": [],
    }

    class NegativeBooster:
        def predict(self, X, num_iteration=None):  # noqa: N803, ARG002
            return np.full(len(X), -0.01)

    class NegativeModel:
        booster = NegativeBooster()
        best_iteration = 1
        feature_names = [
            "pre_event_atm_iv_5d",
            "log_spot",
            "dte",
            "atm_premium",
            "chain_width",
        ]

    conn = mod.open_db(tmp_path / "evidence.db")
    try:
        mod.insert_run_row(conn, "run_model_signal")
        status = mod.StatusWriter(tmp_path / "status.json", run_id="run_model_signal")
        replay = mod.replay_year2(
            conn=conn,
            run_id="run_model_signal",
            cache=cache,
            status=status,
            rng=np.random.default_rng(1),
            synthetic_spy_chains=True,
            allow_synthetic=False,
            satellite_model=NegativeModel(),
        )
        decision = conn.execute(
            "SELECT final_decision, signal_score, signal_pass FROM gate_decisions"
        ).fetchone()
        trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    finally:
        conn.close()

    assert replay["trades_blocked"] == 1
    assert replay["trades_placed"] == 0
    assert replay["strategy_returns"].tolist() == [0.0]
    assert decision[0] == "blocked"
    assert float(decision[1]) == pytest.approx(-0.01)
    assert int(decision[2]) == 0
    assert trades == 0


def test_compute_gate_data_source_priced_when_all_priced(mod):
    """Helper: gate data_source = 'priced' iff every contributing trade is."""
    assert mod._compute_gate_data_source(["priced", "priced"]) == "priced"
    assert mod._compute_gate_data_source(["priced", "synthetic"]) == "synthetic"
    assert mod._compute_gate_data_source(["priced", "metadata"]) == "synthetic"
    assert mod._compute_gate_data_source([]) == "synthetic"  # empty => fail-closed


def test_should_refuse_completion_when_synthetic_and_strict(mod):
    """Refusal logic: any synthetic data_source in gates_payload triggers refusal
    unless allow_synthetic is True."""
    payload = {
        "base": {"data_source": "synthetic", "gates": {}},
        "plus50": {"data_source": "priced", "gates": {}},
        "data_source": "synthetic",
    }
    refuse, reason = mod._should_refuse_completion(payload, allow_synthetic=False)
    assert refuse is True
    assert "synthetic" in (reason or "")

    refuse2, _ = mod._should_refuse_completion(payload, allow_synthetic=True)
    assert refuse2 is False


def test_should_refuse_completion_priced_passes(mod):
    payload = {
        "base": {"data_source": "priced", "gates": {
            "dsr": {"data_source": "priced"},
        }},
        "plus50": {"data_source": "priced", "gates": {
            "dsr": {"data_source": "priced"},
        }},
        "data_source": "priced",
    }
    refuse, reason = mod._should_refuse_completion(payload, allow_synthetic=False)
    assert refuse is False
    assert reason is None


def test_run_refuses_complete_phase_with_synthetic_data_source(mod, tmp_path, monkeypatch):
    """End-to-end: when the gate data_source resolves to 'synthetic' and
    --allow-synthetic was not passed, main exits with rc != 0 and the status
    JSON's phase is 'error'."""
    db_path = tmp_path / "evidence.db"
    status_path = tmp_path / "status.json"
    cache_dir = tmp_path / "cache"

    # Force the data_source computation to return 'synthetic' regardless of
    # the trade_sources list. We still need a working cache; --allow-empty-cache
    # provides that, but we also have to keep effective_allow_synthetic=False
    # for the refusal to fire. We achieve this by monkey-patching the helper
    # main() uses to derive allow_synthetic from args (no clean hook for that),
    # so instead we monkey-patch `_should_refuse_completion` to always refuse.
    monkeypatch.setattr(
        mod,
        "_should_refuse_completion",
        lambda payload, allow_synthetic: (True, "test-injected refusal"),
    )
    rc = mod.main(
        [
            "--allow-empty-cache",
            "--db-path", str(db_path),
            "--cache-dir", str(cache_dir),
            "--status-path", str(status_path),
            "--run-id", "test_refusal",
        ]
    )
    assert rc != 0
    doc = json.loads(status_path.read_text())
    assert doc["phase"] == "error"
    assert any("test-injected" in (e.get("message") or "") for e in doc["errors"])


def test_run_with_allow_synthetic_completes(mod, tmp_path):
    """Smoke-test: --allow-empty-cache (which auto-implies --allow-synthetic)
    still produces phase=complete with the new wiring."""
    db_path = tmp_path / "evidence.db"
    status_path = tmp_path / "status.json"
    cache_dir = tmp_path / "cache"
    rc = mod.main(
        [
            "--allow-empty-cache",
            "--allow-synthetic",  # explicit, redundant with --allow-empty-cache
            "--db-path", str(db_path),
            "--cache-dir", str(cache_dir),
            "--status-path", str(status_path),
            "--run-id", "test_allow_synth",
        ]
    )
    assert rc == 0
    doc = json.loads(status_path.read_text())
    assert doc["phase"] == "complete"
    # Gates carry data_source field on each gate.
    gates = doc["gates"]
    for regime in ("base", "plus50"):
        for gate in gates[regime]["gates"].values():
            assert gate["data_source"] in ("priced", "synthetic")
    # Top-level data_source should also be present.
    assert gates["data_source"] in ("priced", "synthetic")
