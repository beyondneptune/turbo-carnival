"""Tests for `scripts/run_daily_paper.py` — the paper_v1 daily forward orchestrator.

Strategy: import the script as a module and monkeypatch at the module-level
boundaries (Polygon chain fetch, earnings calendar, DB path). Each test gets
its own tmp_path-backed SQLite file initialised from sql/schema.sql so we
exercise real schema constraints (FKs, CHECK clauses, triggers if any).
"""
from __future__ import annotations

import importlib
import sqlite3
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
SCHEMA_PATH = REPO_ROOT / "sql" / "schema.sql"


def _import_module():
    if str(SCRIPTS_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_DIR))
    if "run_daily_paper" in sys.modules:
        return importlib.reload(sys.modules["run_daily_paper"])
    return importlib.import_module("run_daily_paper")


def _init_db(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.executescript(SCHEMA_PATH.read_text())
        conn.commit()
    finally:
        conn.close()


def _seed_run(db_path: Path, run_id: str = "paper_v1", initial_nav: float = 10000.0) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            "INSERT INTO runs (run_id, run_type, prereg_version, started_at_utc, status, initial_nav_usd) "
            "VALUES (?, 'paper_frozen', 'paper_v1.0-frozen', ?, 'running', ?)",
            (run_id, "2026-05-06T13:30:00+00:00", initial_nav),
        )
        conn.commit()
    finally:
        conn.close()


def _seed_nav_history(db_path: Path, run_id: str, rows: list[tuple[str, float, float]]) -> None:
    """Create a `nav_history` table on the fly (the orchestrator reads it if present).

    Each row: (date_iso, nav, intraday_low).
    """
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS nav_history (
                run_id TEXT NOT NULL,
                date TEXT NOT NULL,
                nav REAL NOT NULL,
                intraday_low REAL NOT NULL
            )
        """)
        for d, nav, low in rows:
            conn.execute(
                "INSERT INTO nav_history (run_id, date, nav, intraday_low) VALUES (?, ?, ?, ?)",
                (run_id, d, nav, low),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def fresh_db(tmp_path, monkeypatch):
    """Isolated DB: schema applied, env var pointed at tmp file."""
    db_path = tmp_path / "trading_system.db"
    _init_db(db_path)
    monkeypatch.setenv("TRADING_SYSTEM_DB", str(db_path))
    mod = _import_module()
    monkeypatch.setattr(mod, "DEFAULT_DB_PATH", db_path)
    return db_path, mod


def _synth_spy_chain(today: date) -> pd.DataFrame:
    """Synthetic SPY chain at one expiry ~35 DTE with a band of strikes.

    Hand-crafted deltas place the short call (delta=+0.12) at 580 and short put
    (delta=-0.12) at 540, with $5 wings out at 585 (long call) and 535 (long put).
    """
    expiry = today + pd.Timedelta(days=35).to_pytimedelta()
    spot = 560.0
    # Explicit delta map keyed by strike. Calls: declining from 0.95 ATM-OTM down.
    call_delta_map = {
        530: 0.95, 535: 0.90, 540: 0.82, 545: 0.72, 550: 0.60,
        555: 0.45, 560: 0.50,  # ATM
        565: 0.35, 570: 0.24, 575: 0.18,
        580: 0.12,  # short call target
        585: 0.06,  # long call wing
        590: 0.03, 595: 0.01,
    }
    # Puts: -1 + call_delta gives put_delta, but we override the key short_put strike.
    rows = []
    for strike, cd in call_delta_map.items():
        # Symmetric put delta — but override 540 to exact -0.12 and 535 to -0.06.
        if strike == 540:
            pd_delta = -0.12
        elif strike == 535:
            pd_delta = -0.06
        else:
            pd_delta = cd - 1.0
        intrinsic_call = max(0.0, spot - strike)
        intrinsic_put = max(0.0, strike - spot)
        extrinsic = max(0.5, 8.0 - abs(strike - spot) * 0.15)
        call_mid = intrinsic_call + extrinsic
        put_mid = intrinsic_put + extrinsic
        rows.append({
            "strike": float(strike), "option_type": "call", "expiry": expiry,
            "delta": cd,
            "bid": call_mid - 0.05, "ask": call_mid + 0.05,
        })
        rows.append({
            "strike": float(strike), "option_type": "put", "expiry": expiry,
            "delta": pd_delta,
            "bid": put_mid - 0.05, "ask": put_mid + 0.05,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------


def test_parse_args_defaults():
    mod = _import_module()
    ns = mod.parse_args([])
    assert ns.run_id == "paper_v1"
    assert ns.dry_run is False
    assert ns.date is None


def test_parse_args_custom_date_and_dry_run():
    mod = _import_module()
    ns = mod.parse_args(["--date", "2026-06-01", "--dry-run", "--run-id", "paper_v1_x"])
    assert ns.date == "2026-06-01"
    assert ns.dry_run is True
    assert ns.run_id == "paper_v1_x"


def test_now_utc_iso_uses_new_york_market_open_with_dst():
    mod = _import_module()

    assert mod._now_utc_iso(date(2026, 5, 11)) == "2026-05-11T13:30:00+00:00"
    assert mod._now_utc_iso(date(2026, 1, 5)) == "2026-01-05T14:30:00+00:00"
    assert (
        mod._now_utc_iso(date(2026, 5, 11), hour=mod.EXIT_LOCAL_HOUR, minute=mod.EXIT_LOCAL_MINUTE)
        == "2026-05-11T20:00:00+00:00"
    )


def test_load_cycle_state_bootstraps_nav_history(fresh_db):
    db_path, mod = fresh_db
    _seed_run(db_path, initial_nav=12345.0)
    conn = mod.open_connection(db_path)
    try:
        cycle = mod.load_cycle_state(conn, "paper_v1")
        rows = conn.execute(
            "SELECT date, nav, intraday_low FROM nav_history WHERE run_id = 'paper_v1'"
        ).fetchall()
    finally:
        conn.close()

    assert cycle.current_nav_usd == 12345.0
    assert rows == [("2026-05-06", 12345.0, 12345.0)]


# ---------------------------------------------------------------------------
# kill switch
# ---------------------------------------------------------------------------


def test_run_kill_switch_no_history(fresh_db):
    db_path, mod = fresh_db
    _seed_run(db_path)
    conn = mod.open_connection(db_path)
    try:
        status = mod.run_kill_switch(conn, "paper_v1")
    finally:
        conn.close()
    assert status.suspended is False
    assert status.triggered_at is None
    assert "intraday_loss_8pct" not in status.triggered_reasons
    assert "cycle_drawdown_15pct" not in status.triggered_reasons


def test_kill_switch_trips_main_exits_early(fresh_db, capsys, monkeypatch):
    db_path, mod = fresh_db
    _seed_run(db_path, initial_nav=10000.0)
    # 9% intraday loss between day 1 and day 2 -> hard trigger.
    _seed_nav_history(
        db_path, "paper_v1",
        [("2026-05-04", 10000.0, 9990.0), ("2026-05-05", 9950.0, 9100.0)],
    )

    # Make sure no entries happen even if invoked: stub the chain & calendar.
    monkeypatch.setattr(mod, "fetch_spy_chain", lambda today: pd.DataFrame())
    monkeypatch.setattr(mod, "fetch_todays_earnings", lambda today: [])

    rc = mod.main(["--run-id", "paper_v1", "--date", "2026-05-11"])  # Monday
    assert rc == 0
    out = capsys.readouterr().out
    assert "kill_switch suspended" in out

    # Confirm no trades were inserted.
    conn = sqlite3.connect(db_path)
    try:
        n_trades = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE run_id = 'paper_v1'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert n_trades == 0


def test_main_skips_weekends_before_touching_db(fresh_db, capsys):
    _db_path, mod = fresh_db

    rc = mod.main(["--run-id", "missing", "--date", "2026-05-10"])

    assert rc == 0
    assert "daily run skipped" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# core sleeve
# ---------------------------------------------------------------------------


def test_manage_core_exits_no_open_positions(fresh_db):
    db_path, mod = fresh_db
    _seed_run(db_path)
    conn = mod.open_connection(db_path)
    try:
        closed = mod.manage_core_exits(conn, "paper_v1", date(2026, 5, 11))
    finally:
        conn.close()
    assert closed == []


def test_enter_core_condor_skipped_on_non_monday(fresh_db, monkeypatch):
    db_path, mod = fresh_db
    _seed_run(db_path)
    # Synthetic chain would otherwise yield a trade.
    today = date(2026, 5, 12)  # Tuesday
    monkeypatch.setattr(mod, "fetch_spy_chain", lambda d: _synth_spy_chain(d))
    conn = mod.open_connection(db_path)
    try:
        result = mod.enter_core_condor(conn, "paper_v1", today)
    finally:
        conn.close()
    assert result is None


def test_enter_core_condor_produces_trade_on_monday(fresh_db, monkeypatch):
    db_path, mod = fresh_db
    _seed_run(db_path)
    today = date(2026, 5, 11)  # Monday
    monkeypatch.setattr(mod, "fetch_spy_chain", lambda d: _synth_spy_chain(d))
    conn = mod.open_connection(db_path)
    try:
        trade_id = mod.enter_core_condor(conn, "paper_v1", today)
        conn.commit()
    finally:
        conn.close()
    assert isinstance(trade_id, int) and trade_id > 0

    # Verify schema rows: 1 trade, 4 legs.
    conn = sqlite3.connect(db_path)
    try:
        n_trades = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE run_id = 'paper_v1' AND sleeve = 'core'"
        ).fetchone()[0]
        n_legs = conn.execute(
            "SELECT COUNT(*) FROM trade_legs WHERE trade_id = ?", (trade_id,)
        ).fetchone()[0]
        n_decisions = conn.execute(
            "SELECT COUNT(*) FROM gate_decisions WHERE run_id = 'paper_v1' AND sleeve = 'core'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert n_trades == 1
    assert n_legs == 4
    assert n_decisions == 1


def test_enter_core_condor_no_chain_returns_none(fresh_db, monkeypatch):
    db_path, mod = fresh_db
    _seed_run(db_path)
    today = date(2026, 5, 11)  # Monday
    monkeypatch.setattr(mod, "fetch_spy_chain", lambda d: None)
    conn = mod.open_connection(db_path)
    try:
        result = mod.enter_core_condor(conn, "paper_v1", today)
    finally:
        conn.close()
    assert result is None


# ---------------------------------------------------------------------------
# idempotency
# ---------------------------------------------------------------------------


def test_main_idempotent_on_same_date(fresh_db, monkeypatch):
    db_path, mod = fresh_db
    _seed_run(db_path)
    monkeypatch.setattr(mod, "fetch_spy_chain", lambda d: _synth_spy_chain(d))
    monkeypatch.setattr(mod, "fetch_satellite_chain", lambda t, d: None)
    monkeypatch.setattr(mod, "fetch_todays_earnings", lambda d: [])

    rc1 = mod.main(["--run-id", "paper_v1", "--date", "2026-05-11"])
    rc2 = mod.main(["--run-id", "paper_v1", "--date", "2026-05-11"])
    assert rc1 == 0 and rc2 == 0

    conn = sqlite3.connect(db_path)
    try:
        n_trades = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE run_id = 'paper_v1'"
        ).fetchone()[0]
        n_decisions = conn.execute(
            "SELECT COUNT(*) FROM gate_decisions WHERE run_id = 'paper_v1' AND "
            "candidate_summary LIKE 'ENTRY:%'"
        ).fetchone()[0]
    finally:
        conn.close()
    # Exactly one core entry — second run sees the existing decision and skips.
    assert n_trades == 1
    assert n_decisions == 1


def test_main_help_exits_zero(monkeypatch):
    mod = _import_module()
    with pytest.raises(SystemExit) as exc_info:
        mod.parse_args(["--help"])
    assert exc_info.value.code == 0
