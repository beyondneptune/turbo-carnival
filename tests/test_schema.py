import sqlite3
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "sql" / "schema.sql"


@pytest.fixture
def db():
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text())
    yield conn
    conn.close()


def _insert_run(conn: sqlite3.Connection, run_id: str = "r1") -> None:
    conn.execute(
        "INSERT INTO runs (run_id, run_type, status, initial_nav_usd, started_at_utc) "
        "VALUES (?, ?, ?, ?, ?)",
        (run_id, "paper_demo", "completed", 10000.0, "2026-05-06T00:00:00Z"),
    )


def _insert_decision(conn: sqlite3.Connection, run_id: str = "r1") -> int:
    cur = conn.execute(
        "INSERT INTO gate_decisions (run_id, decision_at_utc, underlying_ticker, sleeve, "
        "candidate_summary, signal_pass, risk_spread_pass, risk_vega_pass, risk_scenario_pass, "
        "risk_liquidity_pass, risk_pin_pass, risk_concentration_pass, final_decision) "
        "VALUES (?, ?, ?, ?, ?, 1, 1, 1, 1, 1, 1, 1, 'approved')",
        (run_id, "2026-05-06T01:00:00Z", "SPY", "demo", "demo trade"),
    )
    return cur.lastrowid


def _insert_trade(
    conn: sqlite3.Connection,
    decision_id: int,
    run_id: str = "r1",
    sleeve: str = "demo",
    strategy_type: str = "demo_long_equity",
    status: str = "open",
    commission_usd: float = 1.0,
    spread_cost_usd: float = 0.5,
) -> int:
    cur = conn.execute(
        "INSERT INTO trades (run_id, decision_id, sleeve, strategy_type, underlying_ticker, "
        "status, entry_at_utc, commission_usd, spread_cost_usd) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            decision_id,
            sleeve,
            strategy_type,
            "SPY",
            status,
            "2026-05-06T02:00:00Z",
            commission_usd,
            spread_cost_usd,
        ),
    )
    return cur.lastrowid


def _insert_leg(
    conn: sqlite3.Connection,
    trade_id: int,
    leg_number: int = 1,
    instrument_type: str = "stock",
    direction: str = "long",
    quantity: int = 1,
    multiplier: int = 1,
    entry_fill_price: float = 100.0,
) -> int:
    cur = conn.execute(
        "INSERT INTO trade_legs (trade_id, leg_number, instrument_type, direction, "
        "quantity, multiplier, entry_fill_price) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (trade_id, leg_number, instrument_type, direction, quantity, multiplier, entry_fill_price),
    )
    return cur.lastrowid


def test_tables_exist(db):
    expected = {"runs", "gate_decisions", "agents_log", "trades", "trade_legs"}
    rows = db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    found = {r[0] for r in rows}
    missing = expected - found
    assert not missing, f"missing tables: {missing}"


def test_foreign_keys_enabled(db):
    assert db.execute("PRAGMA foreign_keys").fetchone()[0] == 1


def test_happy_path_insert(db):
    _insert_run(db)
    decision_id = _insert_decision(db)
    trade_id = _insert_trade(db, decision_id)
    _insert_leg(db, trade_id)
    db.commit()

    assert db.execute("SELECT COUNT(*) FROM runs").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM gate_decisions").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM trades").fetchone()[0] == 1
    assert db.execute("SELECT COUNT(*) FROM trade_legs").fetchone()[0] == 1


def test_check_rejects_bad_sleeve_in_trades(db):
    _insert_run(db)
    decision_id = _insert_decision(db)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_trade(db, decision_id, sleeve="invalid")


def test_check_rejects_bad_strategy_type_in_trades(db):
    _insert_run(db)
    decision_id = _insert_decision(db)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_trade(db, decision_id, strategy_type="not_a_strategy")


def test_check_rejects_bad_status_in_trades(db):
    _insert_run(db)
    decision_id = _insert_decision(db)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_trade(db, decision_id, status="bogus")


def test_check_rejects_bad_instrument_type_in_legs(db):
    _insert_run(db)
    decision_id = _insert_decision(db)
    trade_id = _insert_trade(db, decision_id)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_leg(db, trade_id, instrument_type="future")


def test_check_rejects_bad_direction_in_legs(db):
    _insert_run(db)
    decision_id = _insert_decision(db)
    trade_id = _insert_trade(db, decision_id)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_leg(db, trade_id, direction="sideways")


def test_check_rejects_negative_commission_usd(db):
    _insert_run(db)
    decision_id = _insert_decision(db)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_trade(db, decision_id, commission_usd=-0.01)


def test_check_rejects_negative_spread_cost_usd(db):
    _insert_run(db)
    decision_id = _insert_decision(db)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_trade(db, decision_id, spread_cost_usd=-1.0)


def test_check_rejects_multiplier_zero_in_legs(db):
    _insert_run(db)
    decision_id = _insert_decision(db)
    trade_id = _insert_trade(db, decision_id)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_leg(db, trade_id, multiplier=0)


def test_check_rejects_multiplier_negative_in_legs(db):
    _insert_run(db)
    decision_id = _insert_decision(db)
    trade_id = _insert_trade(db, decision_id)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_leg(db, trade_id, multiplier=-1)


def test_check_rejects_quantity_zero_in_legs(db):
    _insert_run(db)
    decision_id = _insert_decision(db)
    trade_id = _insert_trade(db, decision_id)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_leg(db, trade_id, quantity=0)


def test_unique_decision_id_in_trades(db):
    _insert_run(db)
    decision_id = _insert_decision(db)
    _insert_trade(db, decision_id)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_trade(db, decision_id)


def test_unique_trade_id_leg_number_in_legs(db):
    _insert_run(db)
    decision_id = _insert_decision(db)
    trade_id = _insert_trade(db, decision_id)
    _insert_leg(db, trade_id, leg_number=1)
    with pytest.raises(sqlite3.IntegrityError):
        _insert_leg(db, trade_id, leg_number=1)


def test_fk_enforcement_on_trades_run_id(db):
    # decision exists under run 'r1' but trade references missing 'ghost' run_id
    _insert_run(db, run_id="r1")
    decision_id = _insert_decision(db, run_id="r1")
    with pytest.raises(sqlite3.IntegrityError):
        db.execute(
            "INSERT INTO trades (run_id, decision_id, sleeve, strategy_type, underlying_ticker, "
            "status, entry_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("ghost_run", decision_id, "demo", "demo_long_equity", "SPY", "open", "2026-05-06T02:00:00Z"),
        )
