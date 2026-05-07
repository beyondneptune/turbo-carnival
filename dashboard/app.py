"""Phase 0 dashboard skeleton — read-only view of the paper trading audit DB.

No trading logic, no LLM calls, no DB writes. Reads from the SQLite path given
by env var TRADING_SYSTEM_DB or the default `data/trading_system.db`.

Two extra views (`Paper v1 Run`, `System`) are appended to the sidebar to let
the user monitor a long-running paper_v1 evidence/forward run from a phone on
the LAN. The Paper v1 status file lives at `data/paper_v1_status.json` and is
written by the orchestrators; the dashboard treats it strictly as read-only.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "trading_system.db"
PAPER_V1_STATUS_PATH = REPO_ROOT / "data" / "paper_v1_status.json"
STALE_THRESHOLD_SECONDS = 30 * 60  # 30 minutes
SATELLITE_SAMPLE_TARGET = 50  # gate red/green threshold for sample size

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


# ---------------------------------------------------------------------------
# Paper v1 status file helpers
# ---------------------------------------------------------------------------


def get_paper_v1_status_path() -> Path:
    """Return the configured paper_v1 status path (env override allowed)."""
    override = os.environ.get("PAPER_V1_STATUS_PATH")
    if override:
        return Path(override).expanduser().resolve()
    return PAPER_V1_STATUS_PATH


def read_paper_v1_status(path: Path | None = None) -> dict[str, Any] | None:
    """Read the paper_v1 status JSON file. Returns None if absent or unreadable."""
    target = path if path is not None else get_paper_v1_status_path()
    if not target.exists():
        return None
    try:
        raw = target.read_text()
    except OSError:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return data


def _parse_iso_utc(value: str | None) -> datetime | None:
    if not value or not isinstance(value, str):
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def is_status_stale(
    status: dict[str, Any],
    *,
    now: datetime | None = None,
    threshold_seconds: int = STALE_THRESHOLD_SECONDS,
) -> bool:
    """True if `last_update_utc` is older than `threshold_seconds` (or missing)."""
    last = _parse_iso_utc(status.get("last_update_utc"))
    if last is None:
        return True
    current = now if now is not None else datetime.now(tz=timezone.utc)
    return (current - last).total_seconds() > threshold_seconds


def count_satellite_events(conn: sqlite3.Connection | None, run_id: str) -> int:
    """Count rows in `trades` for the given run with sleeve='satellite'."""
    if conn is None or not run_id:
        return 0
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE sleeve = 'satellite' AND run_id = ?",
            (run_id,),
        ).fetchone()
    except sqlite3.OperationalError:
        return 0
    return int(row[0]) if row else 0


def fetch_run_equity_curve(
    conn: sqlite3.Connection | None,
    run_id: str,
) -> pd.DataFrame:
    """Per-event cumulative realised P&L for a run (closed trades only)."""
    if conn is None or not run_id:
        return pd.DataFrame(columns=["entry_at_utc", "realized_pnl_usd", "cumulative_pnl_usd"])
    try:
        df = pd.read_sql_query(
            "SELECT entry_at_utc, realized_pnl_usd FROM trades "
            "WHERE run_id = ? AND realized_pnl_usd IS NOT NULL "
            "ORDER BY entry_at_utc ASC",
            conn,
            params=(run_id,),
        )
    except (pd.errors.DatabaseError, sqlite3.OperationalError):
        return pd.DataFrame(columns=["entry_at_utc", "realized_pnl_usd", "cumulative_pnl_usd"])
    if df.empty:
        return df
    df["cumulative_pnl_usd"] = df["realized_pnl_usd"].astype(float).cumsum()
    return df


def total_polygon_api_calls(status: dict[str, Any] | None) -> int | None:
    """Sum `api_calls` across stages in the status payload, if present."""
    if not status:
        return None
    stages = status.get("stages")
    if not isinstance(stages, dict):
        return None
    total = 0
    found_any = False
    for stage in stages.values():
        if not isinstance(stage, dict):
            continue
        api_calls = stage.get("api_calls")
        if isinstance(api_calls, (int, float)):
            total += int(api_calls)
            found_any = True
    return total if found_any else None


# ---------------------------------------------------------------------------
# System snapshot (psutil)
# ---------------------------------------------------------------------------


def get_system_snapshot(target_path: Path | None = None) -> dict[str, Any]:
    """Collect a single Mac mini health snapshot via psutil.

    Returns a dict with keys: `cpu_percent`, `cpu_per_core` (list[float]),
    `memory_total_gb`, `memory_used_gb`, `memory_free_gb`, `memory_percent`,
    `disk_total_gb`, `disk_used_gb`, `disk_free_gb`, `processes` (list of dict),
    `timestamp_utc` (ISO string).
    """
    bytes_per_gb = 1024 ** 3
    cpu_percent = float(psutil.cpu_percent(interval=None))
    cpu_per_core = [float(x) for x in psutil.cpu_percent(interval=None, percpu=True)]
    vm = psutil.virtual_memory()
    disk_target = target_path if target_path is not None else (REPO_ROOT / "data")
    try:
        usage = shutil.disk_usage(disk_target if disk_target.exists() else REPO_ROOT)
        disk_total = usage.total / bytes_per_gb
        disk_used = usage.used / bytes_per_gb
        disk_free = usage.free / bytes_per_gb
    except OSError:
        disk_total = disk_used = disk_free = 0.0

    processes: list[dict[str, Any]] = []
    for proc in psutil.process_iter(attrs=["pid", "name", "cmdline"]):
        try:
            info = proc.info
            name = (info.get("name") or "").lower()
            cmdline = " ".join(info.get("cmdline") or [])
            cmd_lower = cmdline.lower()
            if not (
                "python" in name
                or "streamlit" in name
                or "python" in cmd_lower
                or "streamlit" in cmd_lower
            ):
                continue
            cpu = float(proc.cpu_percent(interval=None))
            rss_mb = float(proc.memory_info().rss) / (1024 * 1024)
            processes.append({
                "pid": int(info.get("pid") or 0),
                "name": info.get("name") or "",
                "cmdline": cmdline[:160],
                "cpu_percent": cpu,
                "rss_mb": rss_mb,
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    return {
        "cpu_percent": cpu_percent,
        "cpu_per_core": cpu_per_core,
        "memory_total_gb": float(vm.total) / bytes_per_gb,
        "memory_used_gb": float(vm.used) / bytes_per_gb,
        "memory_free_gb": float(vm.available) / bytes_per_gb,
        "memory_percent": float(vm.percent),
        "disk_total_gb": float(disk_total),
        "disk_used_gb": float(disk_used),
        "disk_free_gb": float(disk_free),
        "processes": processes,
        "timestamp_utc": datetime.now(tz=timezone.utc).isoformat(),
    }


# ---------------------------------------------------------------------------
# Existing views
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# New views: Paper v1 Run + System
# ---------------------------------------------------------------------------


def _stages_to_dataframe(stages: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for name, payload in stages.items():
        if isinstance(payload, dict):
            row: dict[str, Any] = {"stage": name}
            row.update({k: v for k, v in payload.items()})
            rows.append(row)
        else:
            rows.append({"stage": name, "status": str(payload)})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    # promote the most useful columns to the front when present
    preferred = [
        "stage",
        "status",
        "events",
        "events_processed",
        "events_total",
        "trades_placed",
        "trades_blocked",
        "api_calls",
        "cache_hits",
        "model_a_val_mse",
        "model_b_val_mse",
    ]
    ordered = [c for c in preferred if c in df.columns]
    extras = [c for c in df.columns if c not in ordered]
    return df[ordered + extras]


def _gates_to_dataframe(gates: Any) -> pd.DataFrame:
    """Render the 7-gate report regardless of whether it was serialised from
    `eval/gates.py::GateEvaluationReport` or stored as a raw dict.
    """
    if not isinstance(gates, dict):
        return pd.DataFrame()
    base = gates.get("base") if isinstance(gates.get("base"), dict) else {}
    plus50 = gates.get("plus50") if isinstance(gates.get("plus50"), dict) else {}

    base_gates = base.get("gates") if isinstance(base.get("gates"), dict) else {}
    plus50_gates = plus50.get("gates") if isinstance(plus50.get("gates"), dict) else {}
    gate_names = sorted(set(base_gates) | set(plus50_gates))
    if not gate_names:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for name in gate_names:
        b = base_gates.get(name) or {}
        p = plus50_gates.get(name) or {}
        rows.append({
            "gate": name,
            "base_pass": bool(b.get("passed")) if "passed" in b else None,
            "base_detail": b.get("detail", ""),
            "plus50_pass": bool(p.get("passed")) if "passed" in p else None,
            "plus50_detail": p.get("detail", ""),
        })
    return pd.DataFrame(rows)


def show_paper_v1_run(conn: sqlite3.Connection | None) -> None:
    st.subheader("Paper v1 Run")
    status_path = get_paper_v1_status_path()
    st.caption(f"Status file: `{status_path}`")
    status = read_paper_v1_status(status_path)
    if status is None:
        st.info("No paper_v1 run in progress (status file not found).")
        return

    run_id = str(status.get("run_id") or "")
    phase = str(status.get("phase") or "unknown")
    started_at = status.get("started_at_utc") or "—"
    last_update = status.get("last_update_utc") or "—"
    stale = is_status_stale(status)

    header_cols = st.columns(4)
    header_cols[0].metric("Run ID", run_id or "—")
    header_cols[1].metric("Phase", phase)
    header_cols[2].metric("Started (UTC)", started_at)
    header_cols[3].metric("Last update (UTC)", last_update)
    if stale:
        st.warning(
            f"Status is stale (no update in > {STALE_THRESHOLD_SECONDS // 60} min). "
            "The orchestrator may be hung."
        )

    # Stage table
    stages = status.get("stages") or {}
    if isinstance(stages, dict) and stages:
        st.markdown("**Stages**")
        st.dataframe(_stages_to_dataframe(stages), use_container_width=True)
    else:
        st.info("No stage records yet.")

    # Twin-model comparison
    twin = status.get("twin_model_comparison")
    if isinstance(twin, dict):
        st.markdown("**Twin-model comparison (B vs A)**")
        cols = st.columns(4)
        cols[0].metric("Net excess (bps/yr)", twin.get("net_excess_bps_ann", "—"))
        cols[1].metric("Sharpe Δ", twin.get("sharpe_delta", "—"))
        cols[2].metric("Paired t-stat", twin.get("paired_t_stat", "—"))
        verdict = twin.get("b_beats_a")
        if verdict is True:
            cols[3].markdown(":green[**PASS — B beats A**]")
        elif verdict is False:
            cols[3].markdown(":red[**FAIL — B does not beat A**]")
        else:
            cols[3].markdown("**verdict: pending**")

    # 7-gate evaluation
    gates_payload = status.get("gates")
    if gates_payload:
        st.markdown("**7-Gate evaluation (base vs +50% costs)**")
        gates_df = _gates_to_dataframe(gates_payload)
        if not gates_df.empty:
            st.dataframe(gates_df, use_container_width=True)
            overall = gates_payload.get("overall_pass") if isinstance(gates_payload, dict) else None
            if overall is True:
                st.success("Overall: ALL GATES PASS in base AND +50% regimes")
            elif overall is False:
                failed = gates_payload.get("failed_gates_summary") or []
                if failed:
                    st.error(f"Overall: FAIL — {', '.join(failed)}")
                else:
                    st.error("Overall: FAIL")
        else:
            st.json(gates_payload)

    # Live equity curve
    st.markdown("**Live equity curve (cumulative realised P&L)**")
    if conn is None:
        st.info("DB not available — cannot draw equity curve.")
    elif not run_id:
        st.info("No run_id in status payload.")
    else:
        equity_df = fetch_run_equity_curve(conn, run_id)
        if equity_df.empty:
            st.info("No closed trades yet for this run.")
        else:
            chart_df = equity_df.set_index("entry_at_utc")[["cumulative_pnl_usd"]]
            st.line_chart(chart_df)

    # Sample-size counter
    st.markdown("**Satellite sample size**")
    n_satellite = count_satellite_events(conn, run_id) if run_id else 0
    if n_satellite >= SATELLITE_SAMPLE_TARGET:
        st.markdown(f":green[**{n_satellite}** satellite trades (target ≥ {SATELLITE_SAMPLE_TARGET})]")
    else:
        st.markdown(f":red[**{n_satellite}** satellite trades (target ≥ {SATELLITE_SAMPLE_TARGET})]")

    # Errors
    errors = status.get("errors") or []
    if isinstance(errors, list) and errors:
        st.markdown("**Errors**")
        err_rows: list[dict[str, Any]] = []
        for err in errors:
            if isinstance(err, dict):
                err_rows.append(err)
            else:
                err_rows.append({"message": str(err)})
        st.dataframe(pd.DataFrame(err_rows), use_container_width=True)


def show_system() -> None:
    st.subheader("System")
    snap = get_system_snapshot()

    # CPU total + delta
    prev_cpu = st.session_state.get("system_cpu_prev")
    delta = None
    if prev_cpu is not None:
        delta = snap["cpu_percent"] - prev_cpu
    st.session_state["system_cpu_prev"] = snap["cpu_percent"]

    cols = st.columns(4)
    cols[0].metric(
        "CPU total %",
        f"{snap['cpu_percent']:.1f}",
        delta=f"{delta:+.1f}" if delta is not None else None,
    )
    cols[1].metric("Memory used (GiB)", f"{snap['memory_used_gb']:.1f}")
    cols[2].metric("Memory free (GiB)", f"{snap['memory_free_gb']:.1f}")
    cols[3].metric("Disk free (GiB)", f"{snap['disk_free_gb']:.1f}")

    # CPU per core
    st.markdown("**CPU per core (%)**")
    cpu_per_core = snap.get("cpu_per_core") or []
    if cpu_per_core:
        per_core_df = pd.DataFrame({
            "core": [f"core{i}" for i in range(len(cpu_per_core))],
            "cpu_percent": cpu_per_core,
        }).set_index("core")
        st.bar_chart(per_core_df)

    # Memory + disk detail
    mem_disk_df = pd.DataFrame([
        {
            "resource": "memory",
            "total_gb": round(snap["memory_total_gb"], 2),
            "used_gb": round(snap["memory_used_gb"], 2),
            "free_gb": round(snap["memory_free_gb"], 2),
            "percent": round(snap["memory_percent"], 1),
        },
        {
            "resource": "disk(/Users/weeoto/turbo-carnival/data)",
            "total_gb": round(snap["disk_total_gb"], 2),
            "used_gb": round(snap["disk_used_gb"], 2),
            "free_gb": round(snap["disk_free_gb"], 2),
            "percent": (
                round(100 * snap["disk_used_gb"] / snap["disk_total_gb"], 1)
                if snap["disk_total_gb"]
                else 0.0
            ),
        },
    ])
    st.markdown("**Memory & disk**")
    st.dataframe(mem_disk_df, use_container_width=True)

    # Process tree
    st.markdown("**Python / Streamlit processes**")
    procs = snap.get("processes") or []
    if procs:
        proc_df = pd.DataFrame(procs).sort_values("cpu_percent", ascending=False)
        st.dataframe(proc_df, use_container_width=True)
    else:
        st.info("No matching python/streamlit processes found.")

    # Polygon API call counter
    status = read_paper_v1_status()
    api_total = total_polygon_api_calls(status)
    if api_total is None:
        st.markdown("**Polygon API calls**: N/A (no status file or no api_calls in stages)")
    else:
        st.markdown(f"**Polygon API calls (cumulative across stages)**: {api_total:,}")

    # GPU placeholder
    st.markdown(
        "**GPU**: monitoring not available on macOS without `sudo powermetrics`. "
        "Run `sudo powermetrics --samplers gpu_power -n 1` separately if needed."
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(page_title="trading_system — Phase 0", layout="wide")
    st.title("trading_system — Phase 0 dashboard")
    st.caption("Read-only view of the paper trading audit DB. No trade execution.")
    st.caption(
        "Tip: serve on LAN with `.venv/bin/streamlit run dashboard/app.py "
        "--server.address 0.0.0.0 --server.port 8501` — then access from "
        "phone at http://<mac-ip>:8501."
    )

    db_path = get_db_path()
    conn = connect_readonly(db_path)

    view = st.sidebar.radio(
        "View",
        [
            "Overview",
            "Runs",
            "Gate Decisions",
            "Trades",
            "Agent Activity",
            "Paper v1 Run",
            "System",
        ],
    )
    refresh_interval = int(
        st.sidebar.number_input(
            "Refresh interval (s)",
            min_value=0,
            max_value=300,
            value=5,
            step=1,
            help="0 disables auto-refresh.",
        )
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
    elif view == "Paper v1 Run":
        show_paper_v1_run(conn)
    elif view == "System":
        show_system()

    if refresh_interval > 0:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
