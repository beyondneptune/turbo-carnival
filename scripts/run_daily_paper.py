"""Daily forward orchestrator for paper_v1 — runs once per trading day.

Per metadata/paper_v1_prereg.md § 13, this is the OPTIONAL 90-day forward
confirmation. Cycle 1 verdict comes from the held-out Year 2 evaluation
(scripts/run_paper_v1_evidence.py). This script is for real-market
operational validation only.

Daily flow (one run per trading day):
    1. Connect to DB and load the active paper_v1 cycle's run + NAV history.
    2. Evaluate the kill switch; on hard trigger, exit without placing trades.
    3. Core sleeve (SPY iron condors):
         - Mondays: structure a fresh 30-45 DTE, $5-wide, ~12-delta condor.
         - Each open core position: evaluate exit; close if signaled.
    4. Satellite sleeve (Russell 1000 earnings straddles):
         - For each ticker with earnings today: ATM straddle, T-1 entry.
         - Close any straddles whose 5-trading-day window ends today.
    5. Risk envelope gates every candidate before order submission.
    6. Audit: gate_decisions row, then trades + trade_legs rows on approval.

Idempotency: re-running on the same date does not duplicate trades — we check
gate_decisions for an existing entry for (run_id, sleeve, ticker, today) and
skip if present.
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paper_exec.simulator import LegOrder, Quote, simulate_trade_fill  # noqa: E402
from risk.envelope import (  # noqa: E402
    CandidatePosition,
    PortfolioState,
    evaluate_envelope,
)
from risk.kill_switch import KillSwitchStatus, evaluate_kill_switch  # noqa: E402
from signals.vrp_core_spy import (  # noqa: E402
    IronCondorEntry,
    IronCondorSpec,
    compute_entry_credit,
    construct_iron_condor,
    evaluate_exit,
    select_expiry,
)

DEFAULT_RUN_ID = "paper_v1"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "trading_system.db"
SCHEMA_PATH = REPO_ROOT / "sql" / "schema.sql"
PREREG_VERSION = "paper_v1.0-frozen"
MODEL_VERSION_CORE = "vrp_core_spy_v1"
MODEL_VERSION_SAT = "earnings_satellite_twin_v1"

CORE_TICKER = "SPY"
CORE_TARGET_SHORT_DELTA = 0.12
CORE_WING_WIDTH_USD = 5.0
CORE_DTE_MIN = 30
CORE_DTE_MAX = 45
CORE_QUANTITY = 1
CORE_MULTIPLIER = 100
SATELLITE_HOLD_TRADING_DAYS = 5
MARKET_TZ = ZoneInfo("America/New_York")
ENTRY_LOCAL_HOUR = 9
ENTRY_LOCAL_MINUTE = 30
EXIT_LOCAL_HOUR = 16
EXIT_LOCAL_MINUTE = 0

logger = logging.getLogger("run_daily_paper")


# ---------------------------------------------------------------------------
# State containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CycleState:
    """Snapshot of the active paper_v1 cycle at the start of the daily run."""

    run_id: str
    initial_nav_usd: float
    current_nav_usd: float
    current_cash_usd: float
    nav_history: pd.DataFrame  # columns: date, nav, intraday_low
    open_core_trade_ids: list[int] = field(default_factory=list)
    open_satellite_trade_ids: list[int] = field(default_factory=list)


def get_db_path() -> Path:
    """Resolve the DB path from TRADING_SYSTEM_DB env var or default."""
    env = os.environ.get("TRADING_SYSTEM_DB")
    return Path(env) if env else DEFAULT_DB_PATH


def open_connection(db_path: Path) -> sqlite3.Connection:
    """Open a connection with FK pragma enabled. Schema must already exist."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# ---------------------------------------------------------------------------
# Cycle state
# ---------------------------------------------------------------------------


def load_cycle_state(conn: sqlite3.Connection, run_id: str) -> CycleState:
    """Pull the run row, derive NAV/cash, and list open trade ids by sleeve.

    NAV history is read from a `nav_history` table if it exists; otherwise we
    construct a single-row frame from the run's initial NAV (typical for the
    very first daily run after tag time).
    """
    row = conn.execute(
        "SELECT initial_nav_usd, started_at_utc, status FROM runs WHERE run_id = ?",
        (run_id,),
    ).fetchone()
    if row is None:
        raise ValueError(f"run_id {run_id!r} not found in runs table")
    initial_nav, started_at, status = row
    if status not in ("running", "planned"):
        raise ValueError(
            f"run_id {run_id!r} has status={status!r}; not active for daily orchestration"
        )

    nav_history = _load_nav_history(conn, run_id, float(initial_nav), started_at)

    if nav_history.empty:
        current_nav = float(initial_nav)
    else:
        current_nav = float(nav_history["nav"].iloc[-1])
    # Approximation: cash = NAV minus sum of open-position max losses.
    cash_estimate = _estimate_cash(conn, run_id, current_nav)

    open_core = _select_open_trade_ids(conn, run_id, "core")
    open_sat = _select_open_trade_ids(conn, run_id, "satellite")

    return CycleState(
        run_id=run_id,
        initial_nav_usd=float(initial_nav),
        current_nav_usd=current_nav,
        current_cash_usd=cash_estimate,
        nav_history=nav_history,
        open_core_trade_ids=open_core,
        open_satellite_trade_ids=open_sat,
    )


def _load_nav_history(
    conn: sqlite3.Connection,
    run_id: str,
    initial_nav: float,
    started_at: str,
) -> pd.DataFrame:
    """Return a NAV history frame (columns date/nav/intraday_low).

    The live paper run needs a NAV anchor even before the first closed trade.
    Older DBs may not have this table, so create it lazily and seed one row
    from the run metadata when empty.
    """
    _ensure_nav_history_table(conn)

    df = pd.read_sql_query(
        "SELECT date, nav, intraday_low FROM nav_history WHERE run_id = ? "
        "ORDER BY date ASC",
        conn,
        params=(run_id,),
    )
    if df.empty:
        seed_date = _iso_to_date(started_at)
        conn.execute(
            """
            INSERT INTO nav_history (run_id, date, nav, intraday_low)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, seed_date.isoformat(), initial_nav, initial_nav),
        )
        return pd.DataFrame(
            [{"date": seed_date, "nav": float(initial_nav), "intraday_low": float(initial_nav)}]
        )
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df


def _ensure_nav_history_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS nav_history (
            run_id TEXT NOT NULL REFERENCES runs(run_id),
            date TEXT NOT NULL,
            nav REAL NOT NULL,
            intraday_low REAL NOT NULL,
            created_at_utc TEXT NOT NULL DEFAULT (datetime('now')),
            PRIMARY KEY (run_id, date)
        )
        """
    )


def _iso_to_date(value: str) -> date:
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return date.today()
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).date()


def _estimate_cash(conn: sqlite3.Connection, run_id: str, nav: float) -> float:
    """Estimate available cash = NAV minus reserved max-loss for open trades.

    Iron condors: max loss = wing width * 100 * qty (read from trade_legs).
    Straddles: max loss = entry premium * 100 * qty (long premium = full debit).
    """
    rows = conn.execute(
        """
        SELECT t.trade_id, t.strategy_type
        FROM trades t
        WHERE t.run_id = ? AND t.status = 'open'
        """,
        (run_id,),
    ).fetchall()
    reserved = 0.0
    for trade_id, strategy_type in rows:
        legs = conn.execute(
            "SELECT direction, quantity, multiplier, strike, entry_fill_price "
            "FROM trade_legs WHERE trade_id = ? ORDER BY leg_number",
            (trade_id,),
        ).fetchall()
        if strategy_type == "iron_condor" and len(legs) == 4:
            strikes = sorted({float(leg[3]) for leg in legs if leg[3] is not None})
            if len(strikes) >= 2:
                width = max(
                    strikes[-1] - strikes[-2],  # call wing
                    strikes[1] - strikes[0],    # put wing
                )
                qty = legs[0][1]
                mult = legs[0][2]
                reserved += width * qty * mult
        elif strategy_type in ("long_straddle", "short_straddle"):
            premium = sum(
                (leg[4] or 0.0) * leg[1] * leg[2]
                for leg in legs
                if leg[0] == "long"
            )
            reserved += premium
    return max(0.0, nav - reserved)


def _select_open_trade_ids(
    conn: sqlite3.Connection, run_id: str, sleeve: str
) -> list[int]:
    rows = conn.execute(
        "SELECT trade_id FROM trades WHERE run_id = ? AND sleeve = ? AND status = 'open' "
        "ORDER BY trade_id",
        (run_id, sleeve),
    ).fetchall()
    return [int(r[0]) for r in rows]


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


def run_kill_switch(conn: sqlite3.Connection, run_id: str) -> KillSwitchStatus:
    """Evaluate the deterministic kill switch over the cycle's NAV history."""
    state = load_cycle_state(conn, run_id)
    return evaluate_kill_switch(
        nav_history=state.nav_history,
        current_cash_usd=state.current_cash_usd,
        current_nav_usd=state.current_nav_usd,
    )


# ---------------------------------------------------------------------------
# Idempotency check
# ---------------------------------------------------------------------------


def _has_decision_today(
    conn: sqlite3.Connection,
    run_id: str,
    sleeve: str,
    ticker: str,
    today: date,
    candidate_kind: str,
) -> bool:
    """Return True if an entry-side gate decision for (sleeve, ticker, kind) exists today.

    `candidate_kind` is matched against the leading prefix of `candidate_summary`
    (e.g. "ENTRY:iron_condor", "ENTRY:long_straddle"). This lets us distinguish
    entries from exit-related decisions even though both can target the same
    (run, sleeve, ticker) on the same day.
    """
    today_iso = today.isoformat()
    rows = conn.execute(
        """
        SELECT decision_id FROM gate_decisions
        WHERE run_id = ? AND sleeve = ? AND underlying_ticker = ?
          AND substr(decision_at_utc, 1, 10) = ?
          AND candidate_summary LIKE ?
        LIMIT 1
        """,
        (run_id, sleeve, ticker, today_iso, f"{candidate_kind}%"),
    ).fetchall()
    return len(rows) > 0


# ---------------------------------------------------------------------------
# Polygon chain shim
# ---------------------------------------------------------------------------


def fetch_spy_chain(today: date) -> pd.DataFrame | None:
    """Return a SPY option chain DataFrame or None on failure / no API key.

    Network errors, missing keys, or rate-limits all degrade to None — the
    daily orchestrator treats that as "skip new entries today, manage existing
    positions only". Tests monkeypatch this whole function.
    """
    try:
        from ingest.polygon_free import PolygonAuthError, PolygonError, PolygonFreeClient

        client = PolygonFreeClient()
    except Exception as exc:  # auth or import error
        logger.warning("polygon client unavailable: %s", exc)
        return None
    try:
        # Free tier: in production we'd fetch contracts then per-ticker aggregates.
        # The actual chain assembly for SPY across multiple expiries spans dozens
        # of requests, which exceeds the 5/min free-tier budget — we report
        # graceful failure here and leave the live wiring for a later patch.
        del client
        del PolygonAuthError, PolygonError  # noqa: F841 — imports for future use
        logger.info("polygon SPY chain fetch is stubbed; returning None")
        return None
    except Exception as exc:
        logger.warning("polygon SPY chain fetch failed: %s", exc)
        return None


def fetch_satellite_chain(ticker: str, today: date) -> pd.DataFrame | None:
    """Stub mirror of fetch_spy_chain for the satellite sleeve."""
    try:
        from ingest.polygon_free import PolygonFreeClient

        PolygonFreeClient()  # presence check
    except Exception as exc:
        logger.warning("polygon client unavailable for %s: %s", ticker, exc)
        return None
    logger.info("polygon %s chain fetch is stubbed; returning None", ticker)
    return None


def fetch_todays_earnings(today: date) -> list[str]:
    """Return tickers reporting earnings today.

    Backed by ingest.earnings_calendar.fetch_calendar; on any failure we
    return an empty list so the orchestrator treats today as "no satellite
    candidates" rather than aborting the whole run.
    """
    try:
        from ingest.earnings_calendar import fetch_calendar
        from signals.paper_v1_universe import PAPER_V1_SATELLITE
    except Exception as exc:
        logger.warning("earnings calendar unavailable: %s", exc)
        return []

    try:
        cal = fetch_calendar(
            tickers=list(PAPER_V1_SATELLITE),
            start=today,
            end=today,
            use_cache=True,
        )
    except Exception as exc:
        logger.warning("fetch_calendar failed: %s", exc)
        return []

    return sorted([t for t, dates in cal.items() if today in dates])


# ---------------------------------------------------------------------------
# Insert helpers
# ---------------------------------------------------------------------------


def _now_utc_iso(
    today: date,
    hour: int = ENTRY_LOCAL_HOUR,
    minute: int = ENTRY_LOCAL_MINUTE,
) -> str:
    """Return a New-York-market timestamp converted to UTC.

    `hour` and `minute` are interpreted in America/New_York, so market-open
    and market-close timestamps survive US daylight-saving shifts.
    """
    local = datetime(today.year, today.month, today.day, hour, minute, tzinfo=MARKET_TZ)
    return local.astimezone(timezone.utc).isoformat()


def _insert_gate_decision(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    decision_at_utc: str,
    underlying: str,
    sleeve: str,
    candidate_summary: str,
    risk_pass: bool,
    block_reason: str | None,
    final_decision: str,
    model_version: str,
    signal_pass: int = 1,
    signal_score: float | None = None,
    signal_threshold: float | None = None,
) -> int:
    flag = 1 if risk_pass else 0
    cur = conn.execute(
        """
        INSERT INTO gate_decisions (
            run_id, decision_at_utc, underlying_ticker, sleeve, candidate_summary,
            signal_score, signal_threshold, signal_pass,
            risk_spread_pass, risk_vega_pass, risk_scenario_pass,
            risk_liquidity_pass, risk_pin_pass, risk_concentration_pass,
            critic_verdict, critic_reason, final_decision, block_reason,
            trade_id, feature_snapshot_path, model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id, decision_at_utc, underlying, sleeve, candidate_summary,
            signal_score, signal_threshold, signal_pass,
            flag, flag, flag, flag, flag, flag,
            None, None, final_decision, block_reason,
            None, None, model_version,
        ),
    )
    return int(cur.lastrowid)


def _insert_trade_with_legs(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    decision_id: int,
    sleeve: str,
    strategy_type: str,
    underlying: str,
    entry_at_utc: str,
    legs: list[dict],
    entry_greeks: dict,
    commission_usd: float,
    spread_cost_usd: float,
    slippage_usd: float,
    notes: str | None = None,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO trades (
            run_id, decision_id, sleeve, strategy_type, underlying_ticker,
            status, entry_at_utc, exit_at_utc,
            entry_delta, entry_gamma, entry_theta, entry_vega, entry_rho,
            exit_delta, exit_gamma, exit_theta, exit_vega, exit_rho,
            commission_usd, spread_cost_usd, slippage_usd, realized_pnl_usd, notes
        ) VALUES (?, ?, ?, ?, ?, 'open', ?, NULL,
                  ?, ?, ?, ?, ?,
                  NULL, NULL, NULL, NULL, NULL,
                  ?, ?, ?, NULL, ?)
        """,
        (
            run_id, decision_id, sleeve, strategy_type, underlying,
            entry_at_utc,
            entry_greeks.get("delta", 0.0),
            entry_greeks.get("gamma", 0.0),
            entry_greeks.get("theta", 0.0),
            entry_greeks.get("vega", 0.0),
            entry_greeks.get("rho", 0.0),
            commission_usd, spread_cost_usd, slippage_usd, notes,
        ),
    )
    trade_id = int(cur.lastrowid)
    for leg in legs:
        conn.execute(
            """
            INSERT INTO trade_legs (
                trade_id, leg_number, instrument_type, direction,
                quantity, multiplier, strike, expiry,
                entry_intended_price, entry_fill_price,
                exit_intended_price, exit_fill_price,
                entry_delta, entry_gamma, entry_theta, entry_vega, entry_rho
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                leg["leg_number"],
                leg["instrument_type"],
                leg["direction"],
                leg["quantity"],
                leg["multiplier"],
                leg.get("strike"),
                leg.get("expiry"),
                leg.get("entry_intended_price"),
                leg["entry_fill_price"],
                leg.get("entry_delta", 0.0),
                leg.get("entry_gamma", 0.0),
                leg.get("entry_theta", 0.0),
                leg.get("entry_vega", 0.0),
                leg.get("entry_rho", 0.0),
            ),
        )
    return trade_id


def _close_trade(
    conn: sqlite3.Connection,
    *,
    trade_id: int,
    exit_at_utc: str,
    realized_pnl_usd: float,
    exit_commission_usd: float,
    exit_spread_cost_usd: float,
    exit_slippage_usd: float,
    exit_legs: list[dict],
) -> None:
    """Mark a trade closed and stamp exit-side leg fills."""
    cur = conn.execute(
        "SELECT commission_usd, spread_cost_usd, slippage_usd FROM trades WHERE trade_id = ?",
        (trade_id,),
    )
    row = cur.fetchone()
    entry_comm = float(row[0] or 0.0)
    entry_spread = float(row[1] or 0.0)
    entry_slip = float(row[2] or 0.0)
    conn.execute(
        """
        UPDATE trades SET
            status = 'closed',
            exit_at_utc = ?,
            commission_usd = ?,
            spread_cost_usd = ?,
            slippage_usd = ?,
            realized_pnl_usd = ?
        WHERE trade_id = ?
        """,
        (
            exit_at_utc,
            entry_comm + exit_commission_usd,
            entry_spread + exit_spread_cost_usd,
            entry_slip + exit_slippage_usd,
            realized_pnl_usd,
            trade_id,
        ),
    )
    for leg in exit_legs:
        conn.execute(
            "UPDATE trade_legs SET exit_intended_price = ?, exit_fill_price = ? "
            "WHERE trade_id = ? AND leg_number = ?",
            (
                leg.get("exit_intended_price"),
                leg["exit_fill_price"],
                trade_id,
                leg["leg_number"],
            ),
        )


# ---------------------------------------------------------------------------
# Core sleeve
# ---------------------------------------------------------------------------


def enter_core_condor(
    conn: sqlite3.Connection,
    run_id: str,
    today: date,
) -> int | None:
    """Open a fresh SPY iron condor on Mondays. Returns trade_id or None."""
    if today.weekday() != 0:  # Monday only
        logger.debug("core entry skipped: %s is weekday %d", today, today.weekday())
        return None

    if _has_decision_today(conn, run_id, "core", CORE_TICKER, today, "ENTRY:iron_condor"):
        logger.info("core entry skipped: decision already exists for %s", today)
        return None

    chain = fetch_spy_chain(today)
    if chain is None or chain.empty:
        logger.info("core entry skipped: no SPY chain available")
        return None

    expiries = sorted({pd.Timestamp(e).date() for e in chain["expiry"].unique()})
    try:
        target_expiry = select_expiry(
            expiries, today, target_dte_min=CORE_DTE_MIN, target_dte_max=CORE_DTE_MAX
        )
    except ValueError as exc:
        logger.info("core entry skipped: no eligible expiry (%s)", exc)
        return None

    expiry_chain = chain[chain["expiry"] == target_expiry]
    try:
        spec = construct_iron_condor(
            expiry_chain,
            underlying=CORE_TICKER,
            expiry=target_expiry,
            target_short_delta=CORE_TARGET_SHORT_DELTA,
            wing_width=CORE_WING_WIDTH_USD,
        )
    except ValueError as exc:
        logger.info("core entry skipped: cannot construct condor (%s)", exc)
        return None
    spec = IronCondorSpec(
        underlying=spec.underlying,
        expiry=spec.expiry,
        short_call_strike=spec.short_call_strike,
        long_call_strike=spec.long_call_strike,
        short_put_strike=spec.short_put_strike,
        long_put_strike=spec.long_put_strike,
        quantity=CORE_QUANTITY,
        multiplier=CORE_MULTIPLIER,
    )

    try:
        credit = compute_entry_credit(expiry_chain, spec)
    except ValueError as exc:
        logger.info("core entry skipped: credit calc failed (%s)", exc)
        return None

    state = load_cycle_state(conn, run_id)
    candidate_notional = spec.max_loss
    # Approximate iron-condor Greeks for sizing: short-vega ~$0.5 / vol point per
    # spread (well within the 5% NAV cap), neutral delta/gamma at entry. These
    # are deliberately small — a $5-wing condor on a $10k account cannot move
    # the portfolio by more than ~$400 max loss, regardless of Greeks.
    candidate = CandidatePosition(
        underlying=CORE_TICKER,
        sleeve="core",
        notional_usd=candidate_notional,
        delta=0.0,
        gamma=0.0,
        theta=10.0,
        vega=-0.5,  # short ~$0.5 per vol point
        rho=0.0,
        max_loss_usd=spec.max_loss,
    )
    portfolio = PortfolioState(
        nav_usd=state.current_nav_usd,
        cash_usd=state.current_cash_usd,
        open_positions=[],
    )
    risk = evaluate_envelope(candidate, portfolio)

    decision_at_utc = _now_utc_iso(today)
    summary = (
        f"ENTRY:iron_condor SPY {spec.long_put_strike:g}P/{spec.short_put_strike:g}P/"
        f"{spec.short_call_strike:g}C/{spec.long_call_strike:g}C exp={target_expiry.isoformat()} "
        f"credit={credit:.2f}"
    )
    decision_id = _insert_gate_decision(
        conn,
        run_id=run_id,
        decision_at_utc=decision_at_utc,
        underlying=CORE_TICKER,
        sleeve="core",
        candidate_summary=summary,
        risk_pass=risk.passed,
        block_reason=None if risk.passed else f"risk gates failed: {risk.failed_gates}",
        final_decision="approved" if risk.passed else "blocked",
        model_version=MODEL_VERSION_CORE,
        signal_score=credit,
        signal_threshold=0.0,
    )
    if not risk.passed:
        logger.info("core entry blocked: %s", risk.failed_gates)
        return None

    leg_orders = _condor_leg_orders(spec, expiry_chain)
    leg_quotes = _condor_leg_quotes(spec, expiry_chain)
    fill = simulate_trade_fill(leg_orders, leg_quotes)

    legs = []
    spec_legs = (
        ("put", "long", spec.long_put_strike),
        ("put", "short", spec.short_put_strike),
        ("call", "short", spec.short_call_strike),
        ("call", "long", spec.long_call_strike),
    )
    for idx, ((kind, direction, strike), order, quote, leg_fill) in enumerate(
        zip(spec_legs, leg_orders, leg_quotes, fill.leg_fills), start=1
    ):
        legs.append({
            "leg_number": idx,
            "instrument_type": kind,
            "direction": direction,
            "quantity": CORE_QUANTITY,
            "multiplier": CORE_MULTIPLIER,
            "strike": float(strike),
            "expiry": target_expiry.isoformat(),
            "entry_intended_price": order.intended_mid_price,
            "entry_fill_price": leg_fill.fill_price,
        })

    trade_id = _insert_trade_with_legs(
        conn,
        run_id=run_id,
        decision_id=decision_id,
        sleeve="core",
        strategy_type="iron_condor",
        underlying=CORE_TICKER,
        entry_at_utc=decision_at_utc,
        legs=legs,
        entry_greeks={"delta": candidate.delta, "vega": candidate.vega, "theta": candidate.theta},
        commission_usd=fill.total_commission_usd,
        spread_cost_usd=fill.total_spread_cost_usd,
        slippage_usd=fill.total_slippage_usd,
        notes=f"paper_v1 core entry; credit={credit:.2f}",
    )
    logger.info("core entry: trade_id=%d expiry=%s credit=%.2f", trade_id, target_expiry, credit)
    return trade_id


def _condor_leg_orders(spec: IronCondorSpec, chain: pd.DataFrame) -> list[LegOrder]:
    rows = (
        ("put", "long", spec.long_put_strike),
        ("put", "short", spec.short_put_strike),
        ("call", "short", spec.short_call_strike),
        ("call", "long", spec.long_call_strike),
    )
    orders = []
    for kind, direction, strike in rows:
        sub = chain[(chain["strike"] == strike) & (chain["option_type"] == kind)]
        if sub.empty:
            raise ValueError(f"missing {kind} at {strike}")
        bid = float(sub.iloc[0]["bid"])
        ask = float(sub.iloc[0]["ask"])
        mid = 0.5 * (bid + ask)
        orders.append(LegOrder(
            instrument_type=kind,  # type: ignore[arg-type]
            direction=direction,   # type: ignore[arg-type]
            quantity=spec.quantity,
            multiplier=spec.multiplier,
            intended_mid_price=mid,
        ))
    return orders


def _condor_leg_quotes(spec: IronCondorSpec, chain: pd.DataFrame) -> list[Quote]:
    rows = (
        ("put", spec.long_put_strike),
        ("put", spec.short_put_strike),
        ("call", spec.short_call_strike),
        ("call", spec.long_call_strike),
    )
    quotes = []
    for kind, strike in rows:
        sub = chain[(chain["strike"] == strike) & (chain["option_type"] == kind)]
        bid = float(sub.iloc[0]["bid"])
        ask = float(sub.iloc[0]["ask"])
        quotes.append(Quote(bid=bid, ask=ask))
    return quotes


def manage_core_exits(
    conn: sqlite3.Connection,
    run_id: str,
    today: date,
) -> list[int]:
    """Walk open core trades, close any with a triggered exit. Returns closed ids."""
    state = load_cycle_state(conn, run_id)
    closed: list[int] = []
    for trade_id in state.open_core_trade_ids:
        decision = _evaluate_core_trade_exit(conn, trade_id, today)
        if decision is None or not decision.should_exit:
            continue
        exit_legs = _stub_exit_fill_legs(conn, trade_id)
        # Approximate realized P&L using stub exit fills (mid prices).
        realized = _approx_pnl(conn, trade_id, exit_legs)
        _close_trade(
            conn,
            trade_id=trade_id,
            exit_at_utc=_now_utc_iso(today, hour=EXIT_LOCAL_HOUR, minute=EXIT_LOCAL_MINUTE),
            realized_pnl_usd=realized,
            exit_commission_usd=0.65 * len(exit_legs),
            exit_spread_cost_usd=0.0,
            exit_slippage_usd=0.0,
            exit_legs=exit_legs,
        )
        closed.append(trade_id)
        logger.info(
            "core exit: trade_id=%d reason=%s realized=%.2f",
            trade_id, decision.reason, realized,
        )
    return closed


def _evaluate_core_trade_exit(
    conn: sqlite3.Connection, trade_id: int, today: date
):
    """Synthetic exit eval: assume current_pnl == 0 and use leg expiries for DTE.

    A production implementation would price the live chain; here we lean on the
    21 DTE rule being deterministic. Profit/stop branches require a live price
    feed, which the daily orchestrator does not have today.
    """
    legs = conn.execute(
        "SELECT direction, strike, expiry, entry_fill_price, quantity, multiplier "
        "FROM trade_legs WHERE trade_id = ? ORDER BY leg_number",
        (trade_id,),
    ).fetchall()
    if len(legs) != 4:
        return None
    expiries = [pd.Timestamp(leg[2]).date() for leg in legs if leg[2]]
    if not expiries:
        return None
    expiry = expiries[0]
    dte = (expiry - today).days
    short_strikes = sorted({float(leg[1]) for leg in legs if leg[0] == "short"})
    long_strikes = sorted({float(leg[1]) for leg in legs if leg[0] == "long"})
    if len(short_strikes) < 2 or len(long_strikes) < 2:
        return None
    spec = IronCondorSpec(
        underlying=CORE_TICKER,
        expiry=expiry,
        short_call_strike=short_strikes[1],
        long_call_strike=long_strikes[1],
        short_put_strike=short_strikes[0],
        long_put_strike=long_strikes[0],
        quantity=int(legs[0][4]),
        multiplier=int(legs[0][5]),
    )
    short_prices = [float(leg[3]) for leg in legs if leg[0] == "short"]
    long_prices = [float(leg[3]) for leg in legs if leg[0] == "long"]
    entry_credit = sum(short_prices) - sum(long_prices)
    entry = IronCondorEntry(
        spec=spec,
        entry_credit=entry_credit,
        entry_at=today,
        short_call_delta=0.12,
        short_put_delta=-0.12,
        expected_pnl_50pct_target=0.5 * entry_credit * spec.quantity * spec.multiplier,
    )
    return evaluate_exit(entry, today, current_pnl=0.0, days_to_expiry=dte)


def _stub_exit_fill_legs(conn: sqlite3.Connection, trade_id: int) -> list[dict]:
    """Return per-leg exit fill at entry-mid (deterministic stub)."""
    rows = conn.execute(
        "SELECT leg_number, entry_intended_price, entry_fill_price "
        "FROM trade_legs WHERE trade_id = ? ORDER BY leg_number",
        (trade_id,),
    ).fetchall()
    legs = []
    for leg_number, intended, fill in rows:
        # Treat exit at mid (pessimistic-neutral). A real run would re-quote.
        price = float(intended if intended is not None else fill)
        legs.append({
            "leg_number": int(leg_number),
            "exit_intended_price": price,
            "exit_fill_price": price,
        })
    return legs


def _approx_pnl(
    conn: sqlite3.Connection, trade_id: int, exit_legs: list[dict]
) -> float:
    """Approximate realized P&L from the closing fills."""
    rows = conn.execute(
        "SELECT leg_number, direction, quantity, multiplier, entry_fill_price "
        "FROM trade_legs WHERE trade_id = ?",
        (trade_id,),
    ).fetchall()
    by_num = {int(r[0]): r for r in rows}
    total = 0.0
    for leg in exit_legs:
        n = leg["leg_number"]
        if n not in by_num:
            continue
        _, direction, qty, mult, entry_fill = by_num[n]
        sign = 1.0 if direction == "short" else -1.0
        # Closing reverses sign: short -> we buy back (cost); long -> we sell (credit).
        total += sign * (entry_fill - leg["exit_fill_price"]) * qty * mult
    return total


# ---------------------------------------------------------------------------
# Satellite sleeve
# ---------------------------------------------------------------------------


def enter_satellite_straddles(
    conn: sqlite3.Connection,
    run_id: str,
    today: date,
) -> list[int]:
    """Open one ATM long straddle per ticker reporting earnings today."""
    tickers = fetch_todays_earnings(today)
    opened: list[int] = []
    if not tickers:
        return opened
    state = load_cycle_state(conn, run_id)

    for ticker in tickers:
        if _has_decision_today(conn, run_id, "satellite", ticker, today, "ENTRY:long_straddle"):
            continue
        chain = fetch_satellite_chain(ticker, today)
        if chain is None or chain.empty:
            continue
        try:
            target_expiry = min(
                pd.Timestamp(e).date()
                for e in chain["expiry"].unique()
                if pd.Timestamp(e).date() >= today + timedelta(days=3)
            )
        except ValueError:
            continue
        expiry_chain = chain[chain["expiry"] == target_expiry]
        if expiry_chain.empty:
            continue
        # ATM = strike with |strike - mid_spot| min; use cheapest call+put pair.
        strikes = sorted({float(s) for s in expiry_chain["strike"].unique()})
        spot_proxy = expiry_chain["underlying_close"].iloc[0] if "underlying_close" in expiry_chain.columns else strikes[len(strikes) // 2]
        atm_strike = min(strikes, key=lambda s: abs(s - float(spot_proxy)))
        call = expiry_chain[(expiry_chain["strike"] == atm_strike) & (expiry_chain["option_type"] == "call")]
        put = expiry_chain[(expiry_chain["strike"] == atm_strike) & (expiry_chain["option_type"] == "put")]
        if call.empty or put.empty:
            continue
        call_quote = Quote(bid=float(call.iloc[0]["bid"]), ask=float(call.iloc[0]["ask"]))
        put_quote = Quote(bid=float(put.iloc[0]["bid"]), ask=float(put.iloc[0]["ask"]))
        premium = call_quote.mid + put_quote.mid
        max_loss = premium * 100  # one contract pair, multiplier=100

        candidate = CandidatePosition(
            underlying=ticker,
            sleeve="satellite",
            notional_usd=max_loss,
            delta=0.0, gamma=0.5, theta=-2.0,
            vega=state.current_nav_usd * 0.01,
            rho=0.0,
            max_loss_usd=max_loss,
        )
        portfolio = PortfolioState(
            nav_usd=state.current_nav_usd,
            cash_usd=state.current_cash_usd,
            open_positions=[],
        )
        risk = evaluate_envelope(candidate, portfolio)

        decision_at_utc = _now_utc_iso(today)
        summary = (
            f"ENTRY:long_straddle {ticker} {atm_strike:g}C+{atm_strike:g}P "
            f"exp={target_expiry.isoformat()} premium={premium:.2f}"
        )
        decision_id = _insert_gate_decision(
            conn,
            run_id=run_id,
            decision_at_utc=decision_at_utc,
            underlying=ticker,
            sleeve="satellite",
            candidate_summary=summary,
            risk_pass=risk.passed,
            block_reason=None if risk.passed else f"risk gates failed: {risk.failed_gates}",
            final_decision="approved" if risk.passed else "blocked",
            model_version=MODEL_VERSION_SAT,
            signal_score=premium,
        )
        if not risk.passed:
            continue

        leg_orders = [
            LegOrder("call", "long", 1, 100, call_quote.mid),
            LegOrder("put", "long", 1, 100, put_quote.mid),
        ]
        fill = simulate_trade_fill(leg_orders, [call_quote, put_quote])
        legs = [
            {
                "leg_number": 1, "instrument_type": "call", "direction": "long",
                "quantity": 1, "multiplier": 100, "strike": atm_strike,
                "expiry": target_expiry.isoformat(),
                "entry_intended_price": call_quote.mid,
                "entry_fill_price": fill.leg_fills[0].fill_price,
            },
            {
                "leg_number": 2, "instrument_type": "put", "direction": "long",
                "quantity": 1, "multiplier": 100, "strike": atm_strike,
                "expiry": target_expiry.isoformat(),
                "entry_intended_price": put_quote.mid,
                "entry_fill_price": fill.leg_fills[1].fill_price,
            },
        ]
        trade_id = _insert_trade_with_legs(
            conn,
            run_id=run_id,
            decision_id=decision_id,
            sleeve="satellite",
            strategy_type="long_straddle",
            underlying=ticker,
            entry_at_utc=decision_at_utc,
            legs=legs,
            entry_greeks={"vega": candidate.vega, "gamma": candidate.gamma, "theta": candidate.theta},
            commission_usd=fill.total_commission_usd,
            spread_cost_usd=fill.total_spread_cost_usd,
            slippage_usd=fill.total_slippage_usd,
            notes=f"paper_v1 satellite entry; premium={premium:.2f}",
        )
        opened.append(trade_id)
        logger.info("satellite entry: trade_id=%d ticker=%s premium=%.2f", trade_id, ticker, premium)
    return opened


def manage_satellite_exits(
    conn: sqlite3.Connection,
    run_id: str,
    today: date,
) -> list[int]:
    """Close any straddle whose entry was 5 trading days ago (T+5 hold)."""
    rows = conn.execute(
        """
        SELECT trade_id, entry_at_utc FROM trades
        WHERE run_id = ? AND sleeve = 'satellite' AND strategy_type = 'long_straddle'
          AND status = 'open'
        """,
        (run_id,),
    ).fetchall()
    closed: list[int] = []
    for trade_id, entry_at_utc in rows:
        entry_date = pd.Timestamp(entry_at_utc).date()
        if _trading_days_between(entry_date, today) < SATELLITE_HOLD_TRADING_DAYS:
            continue
        exit_legs = _stub_exit_fill_legs(conn, trade_id)
        realized = _approx_pnl(conn, trade_id, exit_legs)
        _close_trade(
            conn,
            trade_id=trade_id,
            exit_at_utc=_now_utc_iso(today, hour=EXIT_LOCAL_HOUR, minute=EXIT_LOCAL_MINUTE),
            realized_pnl_usd=realized,
            exit_commission_usd=0.65 * len(exit_legs),
            exit_spread_cost_usd=0.0,
            exit_slippage_usd=0.0,
            exit_legs=exit_legs,
        )
        closed.append(int(trade_id))
        logger.info("satellite exit: trade_id=%d realized=%.2f", trade_id, realized)
    return closed


def _trading_days_between(start: date, end: date) -> int:
    """Count business days strictly between start and end (exclusive of start, inclusive of end)."""
    if end <= start:
        return 0
    rng = pd.bdate_range(start + timedelta(days=1), end)
    return len(rng)


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Daily forward orchestrator for paper_v1. Runs once per trading day."
    )
    p.add_argument(
        "--run-id", default=DEFAULT_RUN_ID,
        help="run_id of the active paper_v1 cycle (default: paper_v1).",
    )
    p.add_argument(
        "--dry-run", action="store_true",
        help="Evaluate signals/risk and log, but roll back all DB writes.",
    )
    p.add_argument(
        "--date", default=None,
        help="ISO date YYYY-MM-DD (default: today UTC). Used for testing.",
    )
    return p.parse_args(argv)


def _resolve_today(arg: str | None) -> date:
    if arg is None:
        return datetime.now(timezone.utc).date()
    return datetime.strptime(arg, "%Y-%m-%d").date()


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    args = parse_args(argv)
    today = _resolve_today(args.date)
    if today.weekday() >= 5:
        print(f"daily run skipped: {today.isoformat()} is not a weekday")
        return 0

    db_path = get_db_path()
    if not db_path.exists():
        logger.error("DB not found at %s", db_path)
        return 1

    conn = open_connection(db_path)
    try:
        try:
            cycle = load_cycle_state(conn, args.run_id)
        except ValueError as exc:
            logger.error("cannot load cycle %s: %s", args.run_id, exc)
            return 1

        ks = evaluate_kill_switch(
            nav_history=cycle.nav_history,
            current_cash_usd=cycle.current_cash_usd,
            current_nav_usd=cycle.current_nav_usd,
        )
        if ks.suspended:
            logger.warning(
                "kill switch SUSPENDED for run %s: reasons=%s triggered_at=%s",
                args.run_id, ks.triggered_reasons, ks.triggered_at,
            )
            print(f"kill_switch suspended: {ks.triggered_reasons}")
            return 0

        if args.dry_run:
            logger.info("dry-run: opening DB savepoint; rollback at end")

        new_core = enter_core_condor(conn, args.run_id, today)
        closed_core = manage_core_exits(conn, args.run_id, today)
        new_sat = enter_satellite_straddles(conn, args.run_id, today)
        closed_sat = manage_satellite_exits(conn, args.run_id, today)

        if args.dry_run:
            conn.rollback()
            print("dry-run complete; no rows persisted.")
        else:
            conn.commit()

        print(
            f"daily run complete: today={today.isoformat()} "
            f"new_core={new_core} closed_core={closed_core} "
            f"new_satellite={new_sat} closed_satellite={closed_sat}"
        )
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())
