"""Phase 0 dummy SPY-momentum demo seed: exercises full audit path through the schema."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = REPO_ROOT / "data" / "trading_system.db"
SCHEMA_PATH = REPO_ROOT / "sql" / "schema.sql"

RUN_ID = "paper_v0_demo"
PREREG_VERSION = "phase0_demo_v1"
MODEL_VERSION = "phase0_dummy_momentum_v1"
SLEEVE = "demo"
TICKER = "SPY"

N_DAYS = 60
WINDOW = 10  # last ~10 trading days that we evaluate as candidates
MOMENTUM_WIN = 5
MOMENTUM_THRESHOLD = 0.005  # 0.5%
SEED = 42


def init_db(db_path: Path, schema_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    schema_sql = schema_path.read_text()
    conn.executescript(schema_sql)
    # Re-assert FK pragma — executescript can reset pragmas in some sqlite builds
    conn.execute("PRAGMA foreign_keys = ON")
    conn.commit()
    return conn


def cleanup_existing(conn: sqlite3.Connection, run_id: str) -> None:
    cur = conn.cursor()
    cur.execute(
        "DELETE FROM trade_legs WHERE trade_id IN "
        "(SELECT trade_id FROM trades WHERE run_id = ?)",
        (run_id,),
    )
    cur.execute("DELETE FROM trades WHERE run_id = ?", (run_id,))
    cur.execute("DELETE FROM gate_decisions WHERE run_id = ?", (run_id,))
    cur.execute("DELETE FROM agents_log WHERE run_id = ?", (run_id,))
    cur.execute("DELETE FROM runs WHERE run_id = ?", (run_id,))


def synth_spy_prices(n_days: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    # geometric-ish random walk around 500
    daily_returns = rng.normal(loc=0.0005, scale=0.01, size=n_days)
    prices = 500.0 * np.cumprod(1.0 + daily_returns)
    end_date = datetime(2026, 5, 5, tzinfo=timezone.utc).date()
    # produce business days ending at end_date
    dates = pd.bdate_range(end=end_date, periods=n_days)
    return pd.DataFrame({"date": dates, "close": prices})


def compute_momentum_signals(
    prices: pd.DataFrame, window: int, momentum_win: int, threshold: float
) -> pd.DataFrame:
    df = prices.copy()
    df["mom"] = df["close"].pct_change(periods=momentum_win)
    df = df.iloc[-window:].reset_index(drop=True)
    df["signal_pass"] = (df["mom"] > threshold).astype(int)
    return df


def seed_run(
    conn: sqlite3.Connection,
    run_id: str,
    started_at: datetime,
    ended_at: datetime,
) -> None:
    conn.execute(
        """
        INSERT INTO runs (
            run_id, run_type, prereg_version, methodology_commit,
            started_at_utc, ended_at_utc, status, initial_nav_usd, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            "paper_demo",
            PREREG_VERSION,
            None,
            started_at.isoformat(),
            ended_at.isoformat(),
            "completed",
            10000.0,
            "Phase 0 dummy SPY-momentum demo seed (synthetic prices, no live data).",
        ),
    )


def _utc_at(date: pd.Timestamp, hour: int = 14, minute: int = 30) -> str:
    # treat date as UTC market session anchor (~14:30 UTC ~ 09:30 ET pre-DST)
    dt = datetime(
        year=int(date.year),
        month=int(date.month),
        day=int(date.day),
        hour=hour,
        minute=minute,
        tzinfo=timezone.utc,
    )
    return dt.isoformat()


def seed_decisions_and_trades(
    conn: sqlite3.Connection,
    run_id: str,
    signals: pd.DataFrame,
) -> tuple[int, int, int, int]:
    """Insert ~10 gate decisions; for approved decisions insert trade + leg.
    Mark ~5 of the approved trades as closed.
    Returns (n_decisions, n_approved, n_trades_closed, n_legs).
    """
    cur = conn.cursor()

    # Force the mix to ~7 approved / ~3 blocked regardless of synthetic signal.
    # Take indices 0..n-1; deterministically choose 3 to block (indices 1, 4, 7).
    n = len(signals)
    blocked_idx = {1, 4, 7}

    decision_records: list[tuple[int, int, float, str]] = []  # (decision_id, idx, mom, ts)
    approved_decision_ids: list[int] = []

    for i, row in signals.iterrows():
        decision_ts = _utc_at(row["date"], hour=14, minute=30)
        mom = float(row["mom"]) if pd.notna(row["mom"]) else 0.0
        is_blocked = i in blocked_idx

        # Risk gate flags
        risk_flags = {
            "risk_spread_pass": 1,
            "risk_vega_pass": 1,
            "risk_scenario_pass": 1,
            "risk_liquidity_pass": 1,
            "risk_pin_pass": 1,
            "risk_concentration_pass": 1,
        }
        signal_pass_val = 1
        block_reason = None

        if is_blocked:
            # Rotate which gate fails so each block has a distinct reason
            failure_modes = [
                ("signal_pass", "signal below momentum threshold"),
                ("risk_liquidity_pass", "underlying ADV below liquidity floor"),
                ("risk_concentration_pass", "demo sleeve concentration cap reached"),
            ]
            mode_idx = sorted(blocked_idx).index(i) % len(failure_modes)
            failed_field, block_reason = failure_modes[mode_idx]
            if failed_field == "signal_pass":
                signal_pass_val = 0
            else:
                risk_flags[failed_field] = 0

        final_decision = "blocked" if is_blocked else "approved"

        cur.execute(
            """
            INSERT INTO gate_decisions (
                run_id, decision_at_utc, underlying_ticker, sleeve, candidate_summary,
                signal_score, signal_threshold, signal_pass,
                risk_spread_pass, risk_vega_pass, risk_scenario_pass,
                risk_liquidity_pass, risk_pin_pass, risk_concentration_pass,
                critic_verdict, critic_reason,
                final_decision, block_reason,
                trade_id, feature_snapshot_path, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                decision_ts,
                TICKER,
                SLEEVE,
                f"SPY long equity demo @ 5-day momentum {mom:.4f}",
                mom,
                MOMENTUM_THRESHOLD,
                signal_pass_val,
                risk_flags["risk_spread_pass"],
                risk_flags["risk_vega_pass"],
                risk_flags["risk_scenario_pass"],
                risk_flags["risk_liquidity_pass"],
                risk_flags["risk_pin_pass"],
                risk_flags["risk_concentration_pass"],
                "pass" if not is_blocked else "warn",
                None if not is_blocked else "critic flagged borderline candidate",
                final_decision,
                block_reason,
                None,  # trade_id denormalized convenience field — leave NULL
                None,
                MODEL_VERSION,
            ),
        )
        decision_id = cur.lastrowid
        decision_records.append((decision_id, int(i), mom, decision_ts))
        if not is_blocked:
            approved_decision_ids.append(decision_id)

    # Decide which approved trades close: first 5 of 7 close, last 2 stay open
    close_decision_ids = set(approved_decision_ids[:5])

    n_legs = 0
    n_closed = 0
    for decision_id, idx, mom, decision_ts in decision_records:
        if decision_id not in set(approved_decision_ids):
            continue

        signal_row = signals.iloc[idx]
        entry_intended = float(signal_row["close"])
        # small slippage on entry: paid 0.02 above intended
        entry_fill = entry_intended + 0.02
        quantity = 10
        multiplier = 1
        # delta for long stock = +qty*multiplier
        entry_delta = float(quantity * multiplier)

        is_closed = decision_id in close_decision_ids
        exit_at_utc = None
        exit_fill = None
        exit_intended = None
        exit_delta_agg = None
        commission = None
        spread_cost = None
        slippage = None
        realized_pnl = None
        trade_status = "open"

        if is_closed:
            # exit two trading days later — use synthetic price drift
            exit_idx = min(idx + 2, len(signals) - 1)
            exit_row = signals.iloc[exit_idx]
            exit_intended = float(exit_row["close"])
            # small slippage on exit: received 0.03 below intended
            exit_fill = exit_intended - 0.03
            # Slippage convention: intended − actual, summed per leg.
            # Entry leg: bought, intended − actual = entry_intended − entry_fill (negative if paid up)
            # Exit leg: sold, intended − actual = exit_intended − exit_fill (positive if received less)
            slippage = (entry_intended - entry_fill) + (exit_intended - exit_fill)
            commission = 0.65 * 2  # round-trip
            spread_cost = 0.05  # small bid-ask cost
            realized_pnl = (
                (exit_fill - entry_fill) * quantity * multiplier
                - commission
                - spread_cost
            )
            exit_at_utc = _utc_at(exit_row["date"], hour=20, minute=0)
            exit_delta_agg = 0.0  # position closed → no remaining delta
            trade_status = "closed"

        cur.execute(
            """
            INSERT INTO trades (
                run_id, decision_id, sleeve, strategy_type, underlying_ticker,
                status, entry_at_utc, exit_at_utc,
                entry_delta, entry_gamma, entry_theta, entry_vega, entry_rho,
                exit_delta, exit_gamma, exit_theta, exit_vega, exit_rho,
                commission_usd, spread_cost_usd, slippage_usd, realized_pnl_usd,
                notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                decision_id,
                SLEEVE,
                "demo_long_equity",
                TICKER,
                trade_status,
                decision_ts,
                exit_at_utc,
                entry_delta,
                0.0,
                0.0,
                0.0,
                0.0,
                exit_delta_agg if is_closed else None,
                0.0 if is_closed else None,
                0.0 if is_closed else None,
                0.0 if is_closed else None,
                0.0 if is_closed else None,
                commission,
                spread_cost,
                slippage,
                realized_pnl,
                "demo: long SPY shares (synthetic price)",
            ),
        )
        trade_id = cur.lastrowid

        cur.execute(
            """
            INSERT INTO trade_legs (
                trade_id, leg_number, instrument_type, direction,
                quantity, multiplier, strike, expiry,
                entry_intended_price, entry_fill_price,
                exit_intended_price, exit_fill_price,
                entry_delta, entry_gamma, entry_theta, entry_vega, entry_rho
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                1,
                "stock",
                "long",
                quantity,
                multiplier,
                None,
                None,
                entry_intended,
                entry_fill,
                exit_intended,
                exit_fill,
                entry_delta,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
        )
        n_legs += 1
        if is_closed:
            n_closed += 1

    n_decisions = len(decision_records)
    n_approved = len(approved_decision_ids)
    return n_decisions, n_approved, n_closed, n_legs


def seed_agents_log(conn: sqlite3.Connection, run_id: str, anchor_ts: datetime) -> int:
    rows = [
        {
            "role": "summariser",
            "model": "claude-opus-4-7",
            "prompt_version": "summariser_v1.0",
            "input_summary": "summarise SPY 5-day momentum candidates",
            "output_summary": "10 candidates summarised; 7 approved, 3 blocked",
            "tokens_in": 1200,
            "tokens_out": 240,
            "cost_usd": 0.014,
            "duration_ms": 1850,
        },
        {
            "role": "critic",
            "model": "claude-opus-4-7",
            "prompt_version": "critic_v1.0",
            "input_summary": "review approved demo candidates",
            "output_summary": "all approved candidates pass critic; 1 borderline",
            "tokens_in": 980,
            "tokens_out": 180,
            "cost_usd": 0.011,
            "duration_ms": 1620,
        },
        {
            "role": "critic",
            "model": "gpt-5.5",
            "prompt_version": "critic_v1.0",
            "input_summary": "second-pass critic on blocked candidates",
            "output_summary": "blocks confirmed; no override",
            "tokens_in": 760,
            "tokens_out": 120,
            "cost_usd": 0.008,
            "duration_ms": 1310,
        },
        {
            "role": "summariser",
            "model": "gpt-5.5",
            "prompt_version": "summariser_v1.0",
            "input_summary": "end-of-cycle summary",
            "output_summary": "5 trades closed; 2 still open at cycle end",
            "tokens_in": 540,
            "tokens_out": 110,
            "cost_usd": 0.006,
            "duration_ms": 980,
        },
    ]
    cur = conn.cursor()
    for offset, row in enumerate(rows):
        invoked = (anchor_ts + timedelta(minutes=offset * 7)).isoformat()
        cur.execute(
            """
            INSERT INTO agents_log (
                run_id, invoked_at_utc, role, model, prompt_version,
                input_summary, output_summary,
                tokens_in, tokens_out, cost_usd,
                json_schema_valid, artifact_path, duration_ms, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                invoked,
                row["role"],
                row["model"],
                row["prompt_version"],
                row["input_summary"],
                row["output_summary"],
                row["tokens_in"],
                row["tokens_out"],
                row["cost_usd"],
                1,
                None,
                row["duration_ms"],
                None,
            ),
        )
    return len(rows)


def count_rows(conn: sqlite3.Connection, run_id: str) -> dict[str, int]:
    cur = conn.cursor()
    counts: dict[str, int] = {}
    counts["runs"] = cur.execute(
        "SELECT COUNT(*) FROM runs WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    counts["gate_decisions"] = cur.execute(
        "SELECT COUNT(*) FROM gate_decisions WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    counts["trades"] = cur.execute(
        "SELECT COUNT(*) FROM trades WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    counts["trade_legs"] = cur.execute(
        "SELECT COUNT(*) FROM trade_legs WHERE trade_id IN "
        "(SELECT trade_id FROM trades WHERE run_id = ?)",
        (run_id,),
    ).fetchone()[0]
    counts["agents_log"] = cur.execute(
        "SELECT COUNT(*) FROM agents_log WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    return counts


def main() -> None:
    conn = init_db(DB_PATH, SCHEMA_PATH)
    try:
        prices = synth_spy_prices(N_DAYS, SEED)
        signals = compute_momentum_signals(prices, WINDOW, MOMENTUM_WIN, MOMENTUM_THRESHOLD)

        # Anchor timestamps to a realistic cycle
        started_at = datetime(2026, 5, 1, 13, 30, tzinfo=timezone.utc)
        ended_at = datetime(2026, 5, 6, 21, 0, tzinfo=timezone.utc)

        with conn:  # single transaction
            cleanup_existing(conn, RUN_ID)
            seed_run(conn, RUN_ID, started_at, ended_at)
            n_dec, n_appr, n_closed, n_legs = seed_decisions_and_trades(
                conn, RUN_ID, signals
            )
            n_agents = seed_agents_log(conn, RUN_ID, started_at)

        counts = count_rows(conn, RUN_ID)
        print(f"Phase 0 demo seed complete (run_id={RUN_ID})")
        print(f"  runs:            {counts['runs']}")
        print(f"  gate_decisions:  {counts['gate_decisions']}")
        print(f"  trades:          {counts['trades']}")
        print(f"  trade_legs:      {counts['trade_legs']}")
        print(f"  agents_log:      {counts['agents_log']}")
        print(f"DB: {DB_PATH}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
