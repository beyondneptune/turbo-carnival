"""Phase 0 dashboard skeleton — read-only view of the paper trading audit DB.

No trading logic, no LLM calls, no DB writes. Reads from the SQLite path given
by env var TRADING_SYSTEM_DB or the default `data/trading_system.db`.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "trading_system.db"

TABLE_QUERIES: dict[str, str] = {
    "runs": (
        "SELECT run_id, run_type, status, prereg_version, "
        "initial_nav_usd, started_at_utc, ended_at_utc "
        "FROM runs ORDER BY started_at_utc DESC LIMIT ?"
    ),
    "gate_decisions": (
        "SELECT decision_id, run_id, decision_at_utc, underlying_ticker, "
        "sleeve, final_decision, block_reason, candidate_summary "
        "FROM gate_decisions ORDER BY decision_at_utc DESC LIMIT ?"
    ),
    "trades": (
        "SELECT trade_id, run_id, decision_id, sleeve, strategy_type, "
        "underlying_ticker, status, entry_at_utc, exit_at_utc, "
        "realized_pnl_usd FROM trades ORDER BY entry_at_utc DESC LIMIT ?"
    ),
    "trade_legs": (
        "SELECT leg_id, trade_id, leg_number, instrument_type, direction, "
        "quantity, strike, expiry, entry_fill_price, exit_fill_price "
        "FROM trade_legs ORDER BY leg_id DESC LIMIT ?"
    ),
    "agents_log": (
        "SELECT log_id, run_id, invoked_at_utc, role, model, prompt_version, "
        "tokens_in, tokens_out, cost_usd, error "
        "FROM agents_log ORDER BY invoked_at_utc DESC LIMIT ?"
    ),
}


def get_db_path() -> Path:
    override = os.environ.get("TRADING_SYSTEM_DB")
    if override:
        return Path(override).expanduser().resolve()
    return DEFAULT_DB_PATH


def connect_readonly(path: Path) -> sqlite3.Connection | None:
    if not path.exists():
        return None
    try:
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        return sqlite3.connect(uri, uri=True)
    except sqlite3.OperationalError:
        return None


def count_rows(conn: sqlite3.Connection, table: str, where: str = "") -> int:
    sql = f"SELECT COUNT(*) FROM {table}"
    if where:
        sql += f" WHERE {where}"
    try:
        return int(conn.execute(sql).fetchone()[0])
    except sqlite3.OperationalError:
        return 0


def fetch_recent(conn: sqlite3.Connection, table: str, limit: int = 50) -> pd.DataFrame:
    try:
        return pd.read_sql_query(TABLE_QUERIES[table], conn, params=(limit,))
    except (pd.errors.DatabaseError, sqlite3.OperationalError):
        return pd.DataFrame()


def show_overview(conn: sqlite3.Connection | None, db_path: Path) -> None:
    st.subheader("Overview")
    st.caption(f"DB path: `{db_path}`")
    if conn is None:
        st.warning(
            f"Database file not found at `{db_path}`. "
            f"Initialise with `sqlite3 {db_path} < sql/schema.sql` "
            "(create the parent directory first if needed)."
        )
        return

    cols = st.columns(4)
    cols[0].metric("Runs", count_rows(conn, "runs"))
    cols[1].metric("Gate decisions", count_rows(conn, "gate_decisions"))
    cols[2].metric("Trades", count_rows(conn, "trades"))
    cols[3].metric("Open trades", count_rows(conn, "trades", "status = 'open'"))


def show_table_view(conn: sqlite3.Connection | None, label: str, table: str) -> None:
    st.subheader(label)
    if conn is None:
        st.info("DB not available.")
        return
    df = fetch_recent(conn, table)
    if df.empty:
        st.info(f"No rows in `{table}` yet.")
    else:
        st.dataframe(df, use_container_width=True)


def show_gate_decisions(conn: sqlite3.Connection | None) -> None:
    st.subheader("Gate Decisions")
    if conn is None:
        st.info("DB not available.")
        return

    approved = count_rows(conn, "gate_decisions", "final_decision = 'approved'")
    blocked = count_rows(conn, "gate_decisions", "final_decision = 'blocked'")
    if approved or blocked:
        cols = st.columns(2)
        cols[0].metric("Approved", approved)
        cols[1].metric("Blocked", blocked)

    df = fetch_recent(conn, "gate_decisions")
    if df.empty:
        st.info("No gate decisions yet.")
    else:
        st.dataframe(df, use_container_width=True)


def show_trades(conn: sqlite3.Connection | None) -> None:
    st.subheader("Trades")
    if conn is None:
        st.info("DB not available.")
        return

    trades = fetch_recent(conn, "trades")
    if trades.empty:
        st.info("No trades yet.")
    else:
        st.dataframe(trades, use_container_width=True)

    st.subheader("Recent legs")
    legs = fetch_recent(conn, "trade_legs")
    if legs.empty:
        st.info("No trade legs yet.")
    else:
        st.dataframe(legs, use_container_width=True)


def main() -> None:
    st.set_page_config(page_title="trading_system — Phase 0", layout="wide")
    st.title("trading_system — Phase 0 dashboard")
    st.caption("Read-only view of the paper trading audit DB. No trade execution.")

    db_path = get_db_path()
    conn = connect_readonly(db_path)

    view = st.sidebar.radio(
        "View",
        ["Overview", "Runs", "Gate Decisions", "Trades", "Agent Activity"],
    )

    if view == "Overview":
        show_overview(conn, db_path)
    elif view == "Runs":
        show_table_view(conn, "Runs", "runs")
    elif view == "Gate Decisions":
        show_gate_decisions(conn)
    elif view == "Trades":
        show_trades(conn)
    elif view == "Agent Activity":
        show_table_view(conn, "Agent activity", "agents_log")


if __name__ == "__main__":
    main()
