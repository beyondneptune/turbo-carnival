import sqlite3
from pathlib import Path

import pytest

from dashboard.app import (
    DEFAULT_DB_PATH,
    connect_readonly,
    count_rows,
    fetch_recent,
    get_db_path,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "sql" / "schema.sql"


def _make_populated_db(path: Path, n_runs: int = 1, n_trades: int = 0) -> None:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_PATH.read_text())
    for i in range(n_runs):
        conn.execute(
            "INSERT INTO runs (run_id, run_type, status, initial_nav_usd, started_at_utc) "
            "VALUES (?, ?, ?, ?, ?)",
            (f"r{i}", "paper_demo", "completed", 10000.0, f"2026-05-0{(i % 9) + 1}T00:00:00Z"),
        )
    for i in range(n_trades):
        run_id = f"r{i % max(n_runs, 1)}"
        cur = conn.execute(
            "INSERT INTO gate_decisions (run_id, decision_at_utc, underlying_ticker, sleeve, "
            "candidate_summary, signal_pass, risk_spread_pass, risk_vega_pass, risk_scenario_pass, "
            "risk_liquidity_pass, risk_pin_pass, risk_concentration_pass, final_decision) "
            "VALUES (?, ?, ?, ?, ?, 1, 1, 1, 1, 1, 1, 1, 'approved')",
            (run_id, f"2026-05-06T01:{i:02d}:00Z", "SPY", "demo", "demo trade"),
        )
        decision_id = cur.lastrowid
        conn.execute(
            "INSERT INTO trades (run_id, decision_id, sleeve, strategy_type, underlying_ticker, "
            "status, entry_at_utc) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, decision_id, "demo", "demo_long_equity", "SPY", "open",
             f"2026-05-06T02:{i:02d}:00Z"),
        )
    conn.commit()
    conn.close()


def _empty_schema_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA_PATH.read_text())
    return conn


def test_get_db_path_default(monkeypatch):
    monkeypatch.delenv("TRADING_SYSTEM_DB", raising=False)
    assert get_db_path() == DEFAULT_DB_PATH


def test_get_db_path_override(monkeypatch):
    monkeypatch.setenv("TRADING_SYSTEM_DB", "/tmp/my.db")
    result = get_db_path()
    assert result == Path("/tmp/my.db").expanduser().resolve()


def test_get_db_path_expands_tilde(monkeypatch):
    monkeypatch.setenv("TRADING_SYSTEM_DB", "~/foo.db")
    result = get_db_path()
    home = str(Path.home())
    assert "~" not in str(result)
    assert str(result).startswith(home)


def test_connect_readonly_missing_file(tmp_path):
    assert connect_readonly(tmp_path / "nope.db") is None


def test_connect_readonly_real_db(tmp_path):
    db_path = tmp_path / "real.db"
    _make_populated_db(db_path)
    conn = connect_readonly(db_path)
    assert conn is not None
    rows = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    assert rows == 1
    conn.close()


def test_connect_readonly_is_read_only(tmp_path):
    db_path = tmp_path / "real.db"
    _make_populated_db(db_path)
    conn = connect_readonly(db_path)
    assert conn is not None
    with pytest.raises(sqlite3.OperationalError):
        conn.execute(
            "INSERT INTO runs (run_id, run_type, status, initial_nav_usd, started_at_utc) "
            "VALUES ('x', 'paper_demo', 'planned', 0.0, '2026-05-06T00:00:00Z')"
        )
    conn.close()


def test_count_rows_empty():
    conn = _empty_schema_conn()
    assert count_rows(conn, "runs") == 0
    conn.close()


def test_count_rows_populated(tmp_path):
    db_path = tmp_path / "p.db"
    _make_populated_db(db_path, n_runs=3)
    conn = sqlite3.connect(db_path)
    assert count_rows(conn, "runs") == 3
    conn.close()


def test_count_rows_with_where(tmp_path):
    db_path = tmp_path / "p.db"
    _make_populated_db(db_path)
    conn = sqlite3.connect(db_path)
    # add 2 more runs with different status
    conn.execute(
        "INSERT INTO runs (run_id, run_type, status, initial_nav_usd, started_at_utc) "
        "VALUES ('a', 'paper_demo', 'running', 1.0, '2026-05-06T00:00:00Z')"
    )
    conn.execute(
        "INSERT INTO runs (run_id, run_type, status, initial_nav_usd, started_at_utc) "
        "VALUES ('b', 'paper_demo', 'running', 1.0, '2026-05-06T00:00:00Z')"
    )
    conn.commit()
    assert count_rows(conn, "runs", "status = 'completed'") == 1
    assert count_rows(conn, "runs", "status = 'running'") == 2
    conn.close()


def test_count_rows_nonexistent_table():
    conn = _empty_schema_conn()
    assert count_rows(conn, "no_such_table") == 0
    conn.close()


def test_fetch_recent_empty():
    conn = _empty_schema_conn()
    df = fetch_recent(conn, "runs")
    assert df.empty
    conn.close()


def test_fetch_recent_populated(tmp_path):
    db_path = tmp_path / "p.db"
    _make_populated_db(db_path, n_runs=2, n_trades=2)
    conn = sqlite3.connect(db_path)
    df = fetch_recent(conn, "trades")
    assert len(df) == 2
    expected_cols = {"trade_id", "run_id", "decision_id", "sleeve", "strategy_type",
                     "underlying_ticker", "status", "entry_at_utc"}
    assert expected_cols.issubset(set(df.columns))
    conn.close()


def test_fetch_recent_honors_limit(tmp_path):
    db_path = tmp_path / "p.db"
    _make_populated_db(db_path, n_runs=60)
    conn = sqlite3.connect(db_path)
    df = fetch_recent(conn, "runs")
    assert len(df) == 50
    conn.close()


def test_fetch_recent_nonexistent_table():
    # fresh in-memory conn without the schema; the SELECT will hit a missing table
    conn = sqlite3.connect(":memory:")
    df = fetch_recent(conn, "runs")
    assert df.empty
    conn.close()
