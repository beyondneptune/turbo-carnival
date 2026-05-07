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
import plotly.graph_objects as go
import psutil
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = REPO_ROOT / "data" / "trading_system.db"
PAPER_V1_STATUS_PATH = REPO_ROOT / "data" / "paper_v1_status.json"
STALE_THRESHOLD_SECONDS = 30 * 60  # 30 minutes
SATELLITE_SAMPLE_TARGET = 50  # gate red/green threshold for sample size

# Color palette (hex). Streamlit-friendly accent tones, work in light + dark.
COLOR_GREEN = "#22c55e"
COLOR_RED = "#ef4444"
COLOR_AMBER = "#f59e0b"
COLOR_BLUE = "#3b82f6"
COLOR_GRAY = "#6b7280"
COLOR_BG_MUTED = "#1f2937"

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
# Visual primitives
# ---------------------------------------------------------------------------


def _render_status_pill(text: str, color: str, *, big: bool = False) -> str:
    """Return raw HTML for a colored pill. Use with st.markdown(unsafe_allow_html=True)."""
    font_size = "1.4rem" if big else "0.85rem"
    padding = "0.35rem 1rem" if big else "0.15rem 0.6rem"
    weight = "700" if big else "600"
    return (
        f'<span style="display:inline-block;background:{color};color:#ffffff;'
        f'padding:{padding};border-radius:999px;font-weight:{weight};'
        f'font-size:{font_size};letter-spacing:0.02em;">{text}</span>'
    )


def _render_dot(passed: bool | None) -> str:
    """Return HTML for a small status dot (green/red/gray)."""
    if passed is True:
        return f'<span style="color:{COLOR_GREEN};font-size:1.2rem;">●</span>'
    if passed is False:
        return f'<span style="color:{COLOR_RED};font-size:1.2rem;">●</span>'
    return f'<span style="color:{COLOR_GRAY};font-size:1.2rem;">○</span>'


def _empty_state(icon: str, title: str, hint: str) -> None:
    """Polished empty state — icon + title + hint."""
    st.markdown(
        f'<div style="text-align:center;padding:2rem 0.5rem;color:{COLOR_GRAY};">'
        f'<div style="font-size:3rem;line-height:1;">{icon}</div>'
        f'<div style="font-size:1.1rem;font-weight:600;margin-top:0.5rem;">{title}</div>'
        f'<div style="font-size:0.9rem;margin-top:0.3rem;opacity:0.8;">{hint}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def _paper_v1_health(status: dict[str, Any] | None) -> tuple[str, str, str]:
    """Compute (label, color, css-color-band) for the top banner from status."""
    if status is None:
        return ("IDLE", COLOR_GRAY, COLOR_GRAY)
    if status.get("errors"):
        return ("ERROR", COLOR_RED, COLOR_RED)
    if is_status_stale(status):
        return ("STALE", COLOR_AMBER, COLOR_AMBER)
    phase = str(status.get("phase") or "").lower()
    if phase in {"complete", "completed", "done", "finished"}:
        return ("HEALTHY", COLOR_GREEN, COLOR_GREEN)
    return ("RUNNING", COLOR_BLUE, COLOR_BLUE)


def _format_elapsed(started_at: str | None, last_update: str | None) -> str:
    """Return a short human-readable elapsed time, or em-dash if unknown."""
    start_dt = _parse_iso_utc(started_at)
    end_dt = _parse_iso_utc(last_update) or datetime.now(tz=timezone.utc)
    if start_dt is None:
        return "—"
    seconds = max(0.0, (end_dt - start_dt).total_seconds())
    if seconds < 60:
        return f"{int(seconds)}s"
    if seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    hours = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    return f"{hours}h {mins}m"


def _render_top_banner(status: dict[str, Any] | None) -> None:
    """Banner at the top of every page: name, refresh time, LAN tip, color strip."""
    label, color, _band = _paper_v1_health(status)
    now_str = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    # Color band line at top.
    st.markdown(
        f'<div style="height:6px;background:{color};border-radius:4px;margin-bottom:0.6rem;"></div>',
        unsafe_allow_html=True,
    )

    title_col, status_col = st.columns([3, 1])
    with title_col:
        st.markdown(
            '<div style="font-size:1.6rem;font-weight:700;line-height:1.1;">'
            "📊 turbo-carnival · paper_v1"
            "</div>"
            f'<div style="font-size:0.78rem;color:{COLOR_GRAY};margin-top:0.2rem;">'
            f"Last refresh {now_str} · LAN tip: "
            "<code>.venv/bin/streamlit run dashboard/app.py "
            "--server.address 0.0.0.0 --server.port 8501</code>"
            "</div>",
            unsafe_allow_html=True,
        )
    with status_col:
        st.markdown(
            '<div style="text-align:right;padding-top:0.4rem;">'
            + _render_status_pill(label, color, big=False)
            + "</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Existing views (visually polished)
# ---------------------------------------------------------------------------


def show_overview(conn: sqlite3.Connection | None, db_path: Path) -> None:
    st.subheader("📊 Overview")
    if conn is None:
        st.markdown(
            f'<div style="background:#fef3c7;color:#78350f;padding:0.9rem 1rem;'
            f'border-left:4px solid {COLOR_AMBER};border-radius:6px;">'
            f"<b>⚠️ Database not found</b><br>"
            f"<code>{db_path}</code><br>"
            f"<span style='font-size:0.85rem;'>Initialise with "
            f"<code>sqlite3 {db_path} &lt; sql/schema.sql</code> "
            f"(create the parent directory first if needed).</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        return

    n_runs = count_rows(conn, "runs")
    n_gates = count_rows(conn, "gate_decisions")
    n_trades = count_rows(conn, "trades")
    n_open = count_rows(conn, "trades", "status = 'open'")

    cols = st.columns(4)
    cols[0].metric("Runs", n_runs, border=True, help="Total run rows in the audit DB.")
    cols[1].metric(
        "Gate decisions",
        n_gates,
        border=True,
        help="All approve/block decisions logged.",
    )
    cols[2].metric(
        "Trades",
        n_trades,
        border=True,
        help="All paper trades executed across runs.",
    )
    cols[3].metric(
        "Open trades",
        n_open,
        border=True,
        help="Trades with status = 'open'.",
    )

    st.divider()

    with st.container(border=True):
        st.markdown(
            f'<div style="font-size:0.95rem;">'
            f'<span style="color:{COLOR_GREEN};">✅</span> '
            f"<b>Database connected</b> — <code>{db_path}</code>"
            f"</div>",
            unsafe_allow_html=True,
        )


def show_table_view(conn: sqlite3.Connection | None, label: str, table: str) -> None:
    st.subheader(label)
    if conn is None:
        _empty_state("🗄️", "Database not available", "Initialise the audit DB first.")
        return
    df = fetch_recent(conn, table)
    if df.empty:
        _empty_state(
            "📭",
            f"No rows in `{table}` yet",
            "Rows will appear here as the orchestrator records them.",
        )
    else:
        st.caption(f"Showing the {len(df)} most recent rows from `{table}`.")
        st.dataframe(df, use_container_width=True, hide_index=True)


def show_gate_decisions(conn: sqlite3.Connection | None) -> None:
    st.subheader("🎯 Gate Decisions")
    if conn is None:
        _empty_state("🗄️", "Database not available", "Initialise the audit DB first.")
        return

    approved = count_rows(conn, "gate_decisions", "final_decision = 'approved'")
    blocked = count_rows(conn, "gate_decisions", "final_decision = 'blocked'")
    total = approved + blocked

    if total:
        approve_rate = (approved / total) * 100.0 if total else 0.0
        cols = st.columns(3)
        cols[0].metric("Approved", approved, border=True)
        cols[1].metric("Blocked", blocked, border=True)
        cols[2].metric("Approve rate", f"{approve_rate:.0f}%", border=True)
        st.divider()

    df = fetch_recent(conn, "gate_decisions")
    if df.empty:
        _empty_state(
            "🎯",
            "No gate decisions yet",
            "Rows appear here when the gates evaluate a candidate.",
        )
    else:
        st.caption(f"Showing the {len(df)} most recent decisions.")
        st.dataframe(df, use_container_width=True, hide_index=True)


def show_trades(conn: sqlite3.Connection | None) -> None:
    st.subheader("💰 Trades")
    if conn is None:
        _empty_state("🗄️", "Database not available", "Initialise the audit DB first.")
        return

    tab_trades, tab_legs = st.tabs(["💼 Trades", "🦴 Legs"])

    with tab_trades:
        trades = fetch_recent(conn, "trades")
        if trades.empty:
            _empty_state(
                "💼",
                "No trades yet",
                "Approved gate decisions become trades here.",
            )
        else:
            st.caption(f"Showing the {len(trades)} most recent trades.")
            st.dataframe(trades, use_container_width=True, hide_index=True)

    with tab_legs:
        legs = fetch_recent(conn, "trade_legs")
        if legs.empty:
            _empty_state(
                "🦴",
                "No trade legs yet",
                "Each multi-leg trade records its individual legs here.",
            )
        else:
            st.caption(f"Showing the {len(legs)} most recent legs.")
            st.dataframe(legs, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# Paper v1 Run + System
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


def _stage_status_icon(status: str) -> str:
    s = (status or "").lower()
    if s in {"complete", "completed", "done", "finished", "ok"}:
        return "✅"
    if s in {"running", "in_progress", "active"}:
        return "🔄"
    if s in {"error", "failed", "fail"}:
        return "❌"
    if s in {"pending", "queued", "waiting"}:
        return "⏸️"
    return "•"


def _render_stage_progress(stages: dict[str, Any]) -> None:
    """Render each stage with a progress bar (if events_total) or status row."""
    for name, payload in stages.items():
        if not isinstance(payload, dict):
            with st.container(border=True):
                st.markdown(f"**{_stage_status_icon('')} {name}** — {payload}")
            continue
        status = str(payload.get("status") or "")
        icon = _stage_status_icon(status)
        events_total = payload.get("events_total")
        events_processed = payload.get("events_processed")

        with st.container(border=True):
            head_cols = st.columns([3, 1])
            with head_cols[0]:
                st.markdown(f"**{icon} {name}** &nbsp; <span style='color:{COLOR_GRAY};font-size:0.85rem;'>{status}</span>", unsafe_allow_html=True)
            with head_cols[1]:
                api_calls = payload.get("api_calls")
                if isinstance(api_calls, (int, float)):
                    st.markdown(
                        f"<div style='text-align:right;font-size:0.8rem;color:{COLOR_GRAY};'>"
                        f"📡 {int(api_calls):,} api calls</div>",
                        unsafe_allow_html=True,
                    )

            if (
                isinstance(events_total, (int, float))
                and events_total > 0
                and isinstance(events_processed, (int, float))
            ):
                pct = max(0.0, min(1.0, float(events_processed) / float(events_total)))
                st.progress(pct, text=f"{int(events_processed)} / {int(events_total)} events ({pct*100:.0f}%)")

            # Compact key/value strip for other useful fields.
            extras: list[str] = []
            for key in ("trades_placed", "trades_blocked", "cache_hits"):
                v = payload.get(key)
                if isinstance(v, (int, float)):
                    extras.append(f"<b>{key.replace('_',' ')}</b>: {int(v):,}")
            for key in ("model_a_val_mse", "model_b_val_mse"):
                v = payload.get(key)
                if isinstance(v, (int, float)):
                    extras.append(f"<b>{key}</b>: {v:.4g}")
            if extras:
                st.markdown(
                    f"<div style='font-size:0.82rem;color:{COLOR_GRAY};'>"
                    + " &nbsp;·&nbsp; ".join(extras)
                    + "</div>",
                    unsafe_allow_html=True,
                )


def _render_twin_model(twin: dict[str, Any]) -> None:
    """Twin-model verdict — three metric cards + verdict banner."""

    def _num(v: Any) -> float | None:
        if isinstance(v, (int, float)):
            return float(v)
        return None

    excess = _num(twin.get("net_excess_bps_ann"))
    sharpe = _num(twin.get("sharpe_delta"))
    tstat = _num(twin.get("paired_t_stat"))

    # Common gate thresholds for verdict cues.
    excess_ok = excess is not None and excess > 0
    sharpe_ok = sharpe is not None and sharpe > 0
    tstat_ok = tstat is not None and tstat >= 1.96

    cols = st.columns(3)
    cols[0].metric(
        "Net excess (bps/yr)",
        f"{excess:+.1f}" if excess is not None else "—",
        delta=("✅ positive" if excess_ok else ("❌ ≤ 0" if excess is not None else None)),
        delta_color="off",
        border=True,
        help="Annualised net excess return of B over A in bps.",
    )
    cols[1].metric(
        "Sharpe Δ",
        f"{sharpe:+.3f}" if sharpe is not None else "—",
        delta=("✅ positive" if sharpe_ok else ("❌ ≤ 0" if sharpe is not None else None)),
        delta_color="off",
        border=True,
        help="Difference in Sharpe ratio (B − A).",
    )
    cols[2].metric(
        "Paired t-stat",
        f"{tstat:+.2f}" if tstat is not None else "—",
        delta=("✅ ≥ 1.96" if tstat_ok else ("❌ < 1.96" if tstat is not None else None)),
        delta_color="off",
        border=True,
        help="Two-sided significance threshold ≈ 1.96.",
    )

    verdict = twin.get("b_beats_a")
    if verdict is True:
        st.success("✅ **B beats A — adopt Model B**", icon=None)
    elif verdict is False:
        st.error("❌ **B does not beat A — falsified**", icon=None)
    else:
        st.info("⏳ Verdict pending — awaiting full evaluation.")


def _render_gate_cards(gates_payload: dict[str, Any]) -> None:
    """Render each of the 7 gates as a compact pill row (gate name + base/+50% dots)."""
    df = _gates_to_dataframe(gates_payload)
    if df.empty:
        st.json(gates_payload)
        return

    # Header row.
    st.markdown(
        f'<div style="display:flex;align-items:center;font-size:0.78rem;color:{COLOR_GRAY};'
        f'padding:0 0.4rem 0.4rem 0.4rem;letter-spacing:0.04em;">'
        f'<div style="flex:2;">GATE</div>'
        f'<div style="flex:1;text-align:center;">BASE</div>'
        f'<div style="flex:1;text-align:center;">+50%</div>'
        f"</div>",
        unsafe_allow_html=True,
    )
    rows_html: list[str] = []
    for _, row in df.iterrows():
        gate_name = str(row["gate"])
        base_pass = row.get("base_pass")
        plus_pass = row.get("plus50_pass")

        # Choose pill color from base+plus50 combined state.
        if base_pass is True and plus_pass is True:
            pill_bg = COLOR_GREEN
        elif base_pass is False or plus_pass is False:
            pill_bg = COLOR_RED
        else:
            pill_bg = COLOR_GRAY

        rows_html.append(
            f'<div style="display:flex;align-items:center;'
            f'background:{pill_bg}22;border-left:4px solid {pill_bg};'
            f'border-radius:6px;padding:0.45rem 0.6rem;margin-bottom:0.3rem;">'
            f'<div style="flex:2;font-weight:600;font-size:0.92rem;">{gate_name}</div>'
            f'<div style="flex:1;text-align:center;">{_render_dot(base_pass)}</div>'
            f'<div style="flex:1;text-align:center;">{_render_dot(plus_pass)}</div>'
            f"</div>"
        )
    st.markdown("\n".join(rows_html), unsafe_allow_html=True)

    overall = gates_payload.get("overall_pass")
    if overall is True:
        st.success("✅ All 7 gates pass in BOTH base and +50% cost regimes.")
    elif overall is False:
        failed = gates_payload.get("failed_gates_summary") or []
        if failed:
            st.error(f"❌ Overall: FAIL — {', '.join(failed)}")
        else:
            st.error("❌ Overall: FAIL")


def _render_equity_curve(equity_df: pd.DataFrame) -> None:
    """Plotly line chart with gradient fill, hover, terminal-P&L coloring."""
    if equity_df.empty:
        return
    cum = equity_df["cumulative_pnl_usd"].astype(float)
    terminal = float(cum.iloc[-1])
    line_color = COLOR_GREEN if terminal >= 0 else COLOR_RED
    fill_rgba = "rgba(34,197,94,0.20)" if terminal >= 0 else "rgba(239,68,68,0.20)"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=equity_df["entry_at_utc"],
            y=cum,
            mode="lines+markers",
            line={"color": line_color, "width": 2.5},
            marker={"size": 5, "color": line_color},
            fill="tozeroy",
            fillcolor=fill_rgba,
            name="Cumulative P&L",
            hovertemplate="<b>%{x}</b><br>Cum P&L: $%{y:,.2f}<extra></extra>",
        )
    )
    fig.add_hline(y=0, line_dash="dot", line_color=COLOR_GRAY, opacity=0.5)
    fig.update_layout(
        template="plotly_dark",
        margin={"l": 10, "r": 10, "t": 10, "b": 10},
        height=300,
        xaxis={"title": None, "showgrid": True, "gridcolor": "rgba(255,255,255,0.06)"},
        yaxis={
            "title": "Cumulative P&L (USD)",
            "showgrid": True,
            "gridcolor": "rgba(255,255,255,0.06)",
            "zeroline": False,
        },
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_sample_size_pill(n: int, target: int = SATELLITE_SAMPLE_TARGET) -> None:
    """Big pill showing satellite sample-size progress."""
    pct = min(1.0, n / target) if target else 1.0
    color = COLOR_GREEN if n >= target else COLOR_RED
    label = "✅ TARGET REACHED" if n >= target else f"NEED {target - n} MORE"
    cols = st.columns([1, 2])
    with cols[0]:
        st.markdown(
            f'<div style="text-align:center;padding:0.6rem;'
            f"background:{color}22;border:2px solid {color};"
            f"border-radius:12px;\">"
            f'<div style="font-size:2.4rem;font-weight:800;color:{color};line-height:1;">{n}</div>'
            f'<div style="font-size:0.75rem;color:{COLOR_GRAY};margin-top:0.2rem;">'
            f"of {target} satellite trades"
            f"</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.progress(pct, text=label)


def show_paper_v1_run(conn: sqlite3.Connection | None) -> None:
    st.subheader("🚀 Paper v1 Run")
    status_path = get_paper_v1_status_path()
    status = read_paper_v1_status(status_path)
    if status is None:
        _empty_state(
            "🚀",
            "No paper_v1 run in progress",
            f"Status file not found at `{status_path}`. "
            "Start a run to populate this view.",
        )
        return

    run_id = str(status.get("run_id") or "")
    phase = str(status.get("phase") or "unknown")
    started_at = status.get("started_at_utc") or "—"
    last_update = status.get("last_update_utc") or "—"
    stale = is_status_stale(status)
    elapsed = _format_elapsed(status.get("started_at_utc"), status.get("last_update_utc"))

    # ---- Top status banner (big pill + meta strip) ----
    label, color, _band = _paper_v1_health(status)
    with st.container(border=True):
        head_cols = st.columns([1, 2])
        with head_cols[0]:
            st.markdown(
                '<div style="text-align:center;padding:0.4rem;">'
                + _render_status_pill(label, color, big=True)
                + "</div>",
                unsafe_allow_html=True,
            )
        with head_cols[1]:
            meta_cols = st.columns(2)
            meta_cols[0].markdown(
                f'<div style="font-size:0.78rem;color:{COLOR_GRAY};">RUN ID</div>'
                f'<div style="font-size:0.95rem;font-weight:600;word-break:break-all;">{run_id or "—"}</div>'
                f'<div style="font-size:0.78rem;color:{COLOR_GRAY};margin-top:0.5rem;">PHASE</div>'
                f'<div style="font-size:0.95rem;font-weight:600;">{phase}</div>',
                unsafe_allow_html=True,
            )
            meta_cols[1].markdown(
                f'<div style="font-size:0.78rem;color:{COLOR_GRAY};">STARTED (UTC)</div>'
                f'<div style="font-size:0.85rem;font-family:monospace;">{started_at}</div>'
                f'<div style="font-size:0.78rem;color:{COLOR_GRAY};margin-top:0.5rem;">ELAPSED</div>'
                f'<div style="font-size:0.95rem;font-weight:600;">{elapsed}</div>',
                unsafe_allow_html=True,
            )
        if stale:
            st.warning(
                f"⚠️ Status is stale (no update in > {STALE_THRESHOLD_SECONDS // 60} min). "
                f"Last update: `{last_update}`. The orchestrator may be hung."
            )
        else:
            st.caption(f"Last update {last_update} · status file `{status_path.name}`")

    st.divider()

    # ---- Stages ----
    stages = status.get("stages") or {}
    st.markdown("### 🧭 Stages")
    if isinstance(stages, dict) and stages:
        _render_stage_progress(stages)
    else:
        _empty_state("🧭", "No stage records yet", "Stages appear once the orchestrator boots.")

    st.divider()

    # ---- Twin-model comparison ----
    st.markdown("### 🤝 Twin-model verdict (B vs A)")
    twin = status.get("twin_model_comparison")
    if isinstance(twin, dict):
        _render_twin_model(twin)
    else:
        _empty_state(
            "🤝",
            "No twin-model comparison yet",
            "Verdict appears after model B finishes year-2 replay.",
        )

    st.divider()

    # ---- 7-Gate evaluation ----
    st.markdown("### 🛡️ 7-Gate evaluation (base vs +50% costs)")
    gates_payload = status.get("gates")
    if isinstance(gates_payload, dict) and gates_payload:
        _render_gate_cards(gates_payload)
    else:
        _empty_state(
            "🛡️",
            "Gate evaluation pending",
            "Gates run after the twin-model comparison completes.",
        )

    st.divider()

    # ---- Live equity curve ----
    st.markdown("### 📈 Live equity curve (cumulative realised P&L)")
    if conn is None:
        _empty_state("🗄️", "DB not available", "Cannot draw equity curve without the audit DB.")
    elif not run_id:
        _empty_state("📈", "No run_id in status payload", "Set `run_id` in the status JSON.")
    else:
        equity_df = fetch_run_equity_curve(conn, run_id)
        if equity_df.empty:
            _empty_state(
                "📈",
                "No closed trades yet",
                "The curve appears once at least one trade has realised P&L.",
            )
        else:
            terminal = float(equity_df["cumulative_pnl_usd"].iloc[-1])
            cols_top = st.columns(3)
            cols_top[0].metric(
                "Closed trades",
                len(equity_df),
                border=True,
            )
            cols_top[1].metric(
                "Terminal P&L",
                f"${terminal:,.2f}",
                delta=f"{'positive' if terminal >= 0 else 'negative'}",
                delta_color="normal" if terminal >= 0 else "inverse",
                border=True,
            )
            wins = int((equity_df["realized_pnl_usd"].astype(float) > 0).sum())
            cols_top[2].metric(
                "Win count",
                f"{wins} / {len(equity_df)}",
                border=True,
            )
            _render_equity_curve(equity_df)

    st.divider()

    # ---- Sample-size counter ----
    st.markdown("### 🎯 Satellite sample size")
    n_satellite = count_satellite_events(conn, run_id) if run_id else 0
    _render_sample_size_pill(n_satellite, SATELLITE_SAMPLE_TARGET)

    # ---- Errors ----
    errors = status.get("errors") or []
    if isinstance(errors, list) and errors:
        st.divider()
        st.markdown("### 🚨 Errors")
        err_rows: list[dict[str, Any]] = []
        for err in errors:
            if isinstance(err, dict):
                err_rows.append(err)
            else:
                err_rows.append({"message": str(err)})
        st.dataframe(pd.DataFrame(err_rows), use_container_width=True, hide_index=True)


def _cpu_load_color(pct: float) -> str:
    if pct < 50:
        return COLOR_GREEN
    if pct < 80:
        return COLOR_AMBER
    return COLOR_RED


def _render_cpu_per_core(cpu_per_core: list[float]) -> None:
    if not cpu_per_core:
        return
    labels = [f"core {i}" for i in range(len(cpu_per_core))]
    colors = [_cpu_load_color(v) for v in cpu_per_core]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=cpu_per_core,
            marker={"color": colors},
            text=[f"{v:.0f}%" for v in cpu_per_core],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        margin={"l": 10, "r": 10, "t": 24, "b": 10},
        height=240,
        yaxis={
            "range": [0, max(100, max(cpu_per_core) + 8)],
            "title": "%",
            "gridcolor": "rgba(255,255,255,0.06)",
        },
        xaxis={"title": None},
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_donut(used_pct: float, label: str, used_text: str, free_text: str) -> None:
    """Render a single small donut indicator with center text."""
    color = _cpu_load_color(used_pct)
    fig = go.Figure(
        data=[
            go.Pie(
                values=[used_pct, max(0.0, 100.0 - used_pct)],
                hole=0.66,
                marker={"colors": [color, "rgba(255,255,255,0.08)"]},
                textinfo="none",
                hoverinfo="skip",
                sort=False,
                direction="clockwise",
                showlegend=False,
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        margin={"l": 6, "r": 6, "t": 6, "b": 6},
        height=180,
        annotations=[
            {
                "text": f"<b>{used_pct:.0f}%</b>",
                "x": 0.5,
                "y": 0.5,
                "font": {"size": 22, "color": color},
                "showarrow": False,
            }
        ],
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.markdown(
        f'<div style="text-align:center;font-size:0.85rem;font-weight:600;'
        f'margin-bottom:0.2rem;">{label}</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f'<div style="text-align:center;font-size:0.78rem;color:{COLOR_GRAY};">'
        f"used {used_text} · free {free_text}"
        f"</div>",
        unsafe_allow_html=True,
    )


def show_system() -> None:
    st.subheader("🖥️ System")
    snap = get_system_snapshot()

    # ---- Top metric row ----
    prev_cpu = st.session_state.get("system_cpu_prev")
    delta = None
    if prev_cpu is not None:
        delta = snap["cpu_percent"] - prev_cpu
    st.session_state["system_cpu_prev"] = snap["cpu_percent"]

    procs = snap.get("processes") or []
    n_procs = len(procs)

    cols = st.columns(4)
    cols[0].metric(
        "CPU total",
        f"{snap['cpu_percent']:.1f}%",
        delta=f"{delta:+.1f}%" if delta is not None else None,
        border=True,
    )
    cols[1].metric(
        "Memory used",
        f"{snap['memory_used_gb']:.1f} GiB",
        delta=f"{snap['memory_percent']:.0f}% of total",
        delta_color="off",
        border=True,
    )
    cols[2].metric(
        "Disk free",
        f"{snap['disk_free_gb']:.1f} GiB",
        delta=f"of {snap['disk_total_gb']:.0f} GiB",
        delta_color="off",
        border=True,
    )
    cols[3].metric(
        "Python procs",
        n_procs,
        delta="active",
        delta_color="off",
        border=True,
    )

    st.divider()

    # ---- CPU per core (plotly bar, color-coded) ----
    st.markdown("### 🧠 CPU per core")
    cpu_per_core = snap.get("cpu_per_core") or []
    if cpu_per_core:
        _render_cpu_per_core(cpu_per_core)
    else:
        st.info("No per-core CPU data available.")

    st.divider()

    # ---- Memory + Disk donuts side-by-side ----
    st.markdown("### 💾 Memory & disk")
    donut_cols = st.columns(2)
    with donut_cols[0]:
        _render_donut(
            snap["memory_percent"],
            "Memory",
            f"{snap['memory_used_gb']:.1f} GiB",
            f"{snap['memory_free_gb']:.1f} GiB",
        )
    with donut_cols[1]:
        disk_used_pct = (
            (100.0 * snap["disk_used_gb"] / snap["disk_total_gb"])
            if snap["disk_total_gb"]
            else 0.0
        )
        _render_donut(
            disk_used_pct,
            "Disk (data/)",
            f"{snap['disk_used_gb']:.1f} GiB",
            f"{snap['disk_free_gb']:.1f} GiB",
        )

    st.divider()

    # ---- Polygon API call / cache progress counter ----
    # The fetcher writes per-stage `done`/`events`/`rows`/`files_written`
    # rather than a literal `api_calls` field, so fall back to summing those
    # if the explicit key isn't present.
    status = read_paper_v1_status()
    api_total = total_polygon_api_calls(status)
    fallback_total = 0
    if status and isinstance(status.get("stages"), dict):
        for stage in status["stages"].values():
            if not isinstance(stage, dict):
                continue
            for key in ("files_written", "done", "events", "rows"):
                v = stage.get(key)
                if isinstance(v, (int, float)):
                    fallback_total += int(v)
    api_cols = st.columns(2)
    with api_cols[0]:
        if api_total is not None:
            value_str = f"{api_total:,}"
            delta_str = "Polygon api_calls (cumulative)"
        elif fallback_total > 0:
            value_str = f"{fallback_total:,}"
            delta_str = "stage progress (events + rows + done + files)"
        else:
            value_str = "—"
            delta_str = "no status"
        st.metric(
            "Cache progress",
            value_str,
            delta=delta_str,
            delta_color="off",
            border=True,
        )
    with api_cols[1]:
        st.markdown(
            f'<div style="border:1px solid rgba(255,255,255,0.1);padding:0.7rem;border-radius:8px;">'
            f'<div style="font-size:0.85rem;font-weight:600;">📊 Snapshot age</div>'
            f'<div style="font-size:0.78rem;color:{COLOR_GRAY};margin-top:0.3rem;">'
            f"{snap['timestamp_utc']}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # ---- Process tree (compact) ----
    st.markdown("### 🐍 Python / Streamlit processes")
    if procs:
        proc_df = pd.DataFrame(procs).sort_values("cpu_percent", ascending=False).reset_index(drop=True)
        st.dataframe(
            proc_df,
            use_container_width=True,
            hide_index=True,
            height=min(360, 40 + 32 * len(proc_df)),
        )
    else:
        _empty_state(
            "🐍",
            "No matching processes",
            "Run an orchestrator and refresh to see live python procs.",
        )

    # ---- GPU placeholder (expander) ----
    with st.expander("🎮 GPU monitoring (macOS)"):
        st.markdown(
            "macOS does not expose GPU metrics through `psutil`. To peek at GPU power draw "
            "manually, run:\n\n"
            "```sh\nsudo powermetrics --samplers gpu_power -n 1\n```\n\n"
            "This dashboard intentionally avoids `sudo` calls — keep GPU monitoring out-of-band."
        )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="turbo-carnival paper_v1",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    db_path = get_db_path()
    conn = connect_readonly(db_path)
    status = read_paper_v1_status()

    # ---- Top banner with health color strip ----
    _render_top_banner(status)

    # ---- Sidebar: grouped views ----
    st.sidebar.markdown("### 🔍 Audit")
    audit_view = st.sidebar.radio(
        "Audit views",
        [
            "📊 Overview",
            "📋 Runs",
            "🎯 Gate Decisions",
            "💰 Trades",
            "🤖 Agent Activity",
        ],
        label_visibility="collapsed",
        key="audit_view",
    )

    st.sidebar.markdown("### 📡 Live")
    live_view = st.sidebar.radio(
        "Live views",
        [
            "🚀 Paper v1 Run",
            "🖥️ System",
        ],
        label_visibility="collapsed",
        key="live_view",
    )

    # Determine the active view from the most-recently-clicked group.
    last_group = st.session_state.get("last_group", "live")
    # Detect change in either radio.
    last_audit = st.session_state.get("_last_audit_value")
    last_live = st.session_state.get("_last_live_value")
    if audit_view != last_audit and last_audit is not None:
        last_group = "audit"
    elif live_view != last_live and last_live is not None:
        last_group = "live"
    st.session_state["_last_audit_value"] = audit_view
    st.session_state["_last_live_value"] = live_view
    st.session_state["last_group"] = last_group
    active = audit_view if last_group == "audit" else live_view

    st.sidebar.divider()

    # ---- Sidebar: refresh controls ----
    st.sidebar.markdown("### ⚙️ Refresh")
    refresh_interval = int(
        st.sidebar.number_input(
            "Auto-refresh (s)",
            min_value=0,
            max_value=300,
            value=5,
            step=1,
            help="0 disables auto-refresh.",
            key="refresh_interval_seconds",
        )
    )
    if st.sidebar.button("🔄 Refresh now", use_container_width=True):
        st.rerun()

    # Auto-refresh badge near the top.
    if refresh_interval > 0:
        st.markdown(
            f'<div style="display:inline-block;background:{COLOR_BLUE}22;color:{COLOR_BLUE};'
            f'padding:0.2rem 0.7rem;border-radius:999px;font-size:0.75rem;font-weight:600;'
            f'border:1px solid {COLOR_BLUE}66;margin-bottom:0.5rem;">'
            f"⏱️ Auto-refresh: {refresh_interval}s"
            "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="display:inline-block;background:{COLOR_GRAY}22;color:{COLOR_GRAY};'
            f'padding:0.2rem 0.7rem;border-radius:999px;font-size:0.75rem;font-weight:600;'
            f'border:1px solid {COLOR_GRAY}66;margin-bottom:0.5rem;">'
            "⏸️ Auto-refresh paused"
            "</div>",
            unsafe_allow_html=True,
        )

    # ---- Route to view by suffix-match (icon prefixes are decorative) ----
    if "Overview" in active:
        show_overview(conn, db_path)
    elif "Runs" in active:
        show_table_view(conn, "📋 Runs", "runs")
    elif "Gate Decisions" in active:
        show_gate_decisions(conn)
    elif "Trades" in active:
        show_trades(conn)
    elif "Agent Activity" in active:
        show_table_view(conn, "🤖 Agent activity", "agents_log")
    elif "Paper v1 Run" in active:
        show_paper_v1_run(conn)
    elif "System" in active:
        show_system()

    if refresh_interval > 0:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
