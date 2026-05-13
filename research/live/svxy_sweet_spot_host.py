"""SVXY Sweet-Spot Paper-Shadow Deployment Host.

================================================================================
STRATEGY (validated by /research/results/svxy_sweep_top.json — sweet spot row):
  Threshold:      VIX / VIX3M < 1.00   (contango — positive roll yield expected)
  Size:           25% of account NAV in SVXY when signal active
  Cash leg:       75% of NAV held as cash (or in SGOV if --sgov)
  Circuit breaker: flatten SVXY if VIX intraday > 40
  Hard kill:      flatten + halt if SVXY intraday drop > 15%
                  flatten + halt if account drawdown > 12% rolling peak

  Backtest 2019-01-01 to 2026-05-11 (1849 days):
    total_return_pct: 70.11
    ann_return_pct:   7.51
    ann_vol_pct:      7.88
    sharpe:           0.96
    max_drawdown_pct: -8.36
    rotations/year:   11.3
    days_active:      92.81%

================================================================================
DEPLOYMENT
--------------------------------------------------------------------------------
First run (interactive confirm on first 5 trades, recommended):
    /Users/weeoto/turbo-carnival/.venv/bin/python \
        /Users/weeoto/turbo-carnival/research/live/svxy_sweet_spot_host.py \
        --account AEGIS --confirm

Daemon mode (after first 5 trades confirmed, runs forever):
    /Users/weeoto/turbo-carnival/.venv/bin/python \
        /Users/weeoto/turbo-carnival/research/live/svxy_sweet_spot_host.py \
        --account AEGIS

One-shot cron-style:
    /Users/weeoto/turbo-carnival/.venv/bin/python \
        /Users/weeoto/turbo-carnival/research/live/svxy_sweet_spot_host.py \
        --account AEGIS --once

Dry run (compute signal, no orders):
    /Users/weeoto/turbo-carnival/research/live/svxy_sweet_spot_host.py \
        --account AEGIS --dry-run

================================================================================
ENVIRONMENT
--------------------------------------------------------------------------------
Reads /Users/weeoto/turbo-carnival/council/.env.local then process env.

Required env vars (or substitute ALPACA_API_KEY / ALPACA_API_SECRET as fallback):
    ALPACA_<ACCOUNT>_KEY      e.g. ALPACA_AEGIS_KEY
    ALPACA_<ACCOUNT>_SECRET   e.g. ALPACA_AEGIS_SECRET

Defaults:
    ALPACA_PAPER_ACCOUNT      AEGIS   (override via --account or this env var)
    SVXY_DECISION_TIME_ET     15:30   (override via --decision-time-et)

================================================================================
KILL-SWITCH INSTRUCTIONS
--------------------------------------------------------------------------------
Automatic kills (executed by the script):
    1. VIX intraday > 40                  -> flatten SVXY this loop only
    2. SVXY intraday return < -15%         -> flatten + halt daemon
    3. Account drawdown > 12% from peak    -> flatten + halt daemon
    4. /Users/weeoto/turbo-carnival/research/live/STOP_SVXY exists -> halt

Manual halt (any of the following):
    A. Create the kill file (does NOT flatten — just halts the loop):
         touch /Users/weeoto/turbo-carnival/research/live/STOP_SVXY
    B. SIGINT / SIGTERM the process (Ctrl-C in tmux, or `kill <pid>`).
    C. Force-flatten now:
         /Users/weeoto/turbo-carnival/.venv/bin/python \
             /Users/weeoto/turbo-carnival/research/live/svxy_sweet_spot_host.py \
             --account AEGIS --flatten-now

================================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests

# ---------------------------------------------------------------------------
# Constants & paths
# ---------------------------------------------------------------------------
REPO = Path("/Users/weeoto/turbo-carnival")
ENV_FILE = REPO / "council" / ".env.local"
LIVE_DIR = REPO / "research" / "live"
DB_PATH = LIVE_DIR / "svxy_decisions.sqlite3"
LOG_DIR = REPO / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LIVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "svxy_sweet_spot_host.log"
STOP_FILE = LIVE_DIR / "STOP_SVXY"

NY = ZoneInfo("America/New_York")
ENDPOINT = "https://paper-api.alpaca.markets/v2"
DATA_ENDPOINT = "https://data.alpaca.markets/v2"

SVXY = "SVXY"
SGOV = "SGOV"  # 0-3M T-bill ETF (cash-equivalent)

# Sweet-spot parameters (sourced from svxy_sweep_top.json)
THRESHOLD_RATIO = 1.00
SIZE_PCT = 0.25
CB_VIX_LEVEL = 40.0

# Hard kills
HARD_KILL_SVXY_INTRADAY_DROP = -0.15  # -15%
HARD_KILL_ACCOUNT_DRAWDOWN = -0.12    # -12% from rolling peak

# Trading thresholds
MIN_REBALANCE_USD = 50.0
ORDER_PRICE_SLIP_BPS = 25  # 25 bps marketable-limit aggression
FIRST_N_TRADES_REQUIRE_CONFIRM = 5

# Backtest verification tolerance
VERIFY_LOOKBACK_DAYS = 90
VERIFY_DIVERGENCE_TOLERANCE = 0.10  # 10%

logger = logging.getLogger("svxy_sweet_spot_host")
STOPPED = False


# ---------------------------------------------------------------------------
# Env loading
# ---------------------------------------------------------------------------
def load_env() -> None:
    """Merge council/.env.local into os.environ (does not override existing)."""
    if not ENV_FILE.exists():
        logger.warning(f"env file not found: {ENV_FILE} — relying on process env")
        return
    for raw in ENV_FILE.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def resolve_credentials(account: str) -> tuple[str, str]:
    """Return (key, secret) for the account, falling back to generic ALPACA_API_*."""
    account = account.upper()
    key = os.environ.get(f"ALPACA_{account}_KEY") or os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get(f"ALPACA_{account}_SECRET") or os.environ.get("ALPACA_API_SECRET")
    if not key or not sec:
        raise SystemExit(
            f"Missing Alpaca paper credentials. Expected ALPACA_{account}_KEY/"
            f"ALPACA_{account}_SECRET (or ALPACA_API_KEY / ALPACA_API_SECRET)."
        )
    return key, sec


# ---------------------------------------------------------------------------
# Alpaca REST helpers (paper)
# ---------------------------------------------------------------------------
def _headers(key: str, secret: str) -> dict:
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def get_account(key: str, secret: str) -> Optional[dict]:
    try:
        r = requests.get(f"{ENDPOINT}/account", headers=_headers(key, secret), timeout=10)
        if r.status_code == 200:
            return r.json()
        logger.warning(f"account GET {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        logger.warning(f"account exception: {exc}")
    return None


def get_position(key: str, secret: str, symbol: str) -> Optional[dict]:
    try:
        r = requests.get(f"{ENDPOINT}/positions/{symbol}",
                         headers=_headers(key, secret), timeout=10)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 404:
            return None  # no position
        logger.warning(f"position GET {symbol} {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        logger.warning(f"position exception: {exc}")
    return None


def get_latest_trade(key: str, secret: str, symbol: str) -> Optional[float]:
    try:
        r = requests.get(f"{DATA_ENDPOINT}/stocks/{symbol}/trades/latest",
                         headers=_headers(key, secret), timeout=10)
        if r.status_code == 200:
            j = r.json()
            if j.get("trade") and j["trade"].get("p"):
                return float(j["trade"]["p"])
    except Exception as exc:
        logger.warning(f"trade GET {symbol} exception: {exc}")
    return None


def get_asset_tradable(key: str, secret: str, symbol: str) -> bool:
    try:
        r = requests.get(f"{ENDPOINT}/assets/{symbol}",
                         headers=_headers(key, secret), timeout=8)
        if r.status_code == 200:
            j = r.json()
            return bool(j.get("tradable")) and j.get("status") == "active"
    except Exception:
        pass
    return False


def submit_marketable_limit_ioc(
    key: str, secret: str, symbol: str, qty: float, side: str, ref_price: float,
    client_tag: str,
) -> Optional[dict]:
    """Submit a marketable IOC limit order at ref_price +/- ORDER_PRICE_SLIP_BPS."""
    slip = ref_price * (ORDER_PRICE_SLIP_BPS / 10_000.0)
    if side == "buy":
        limit_price = round(ref_price + slip, 2)
    else:
        limit_price = round(ref_price - slip, 2)
    payload = {
        "symbol": symbol,
        "qty": str(round(qty, 4)),
        "side": side,
        "type": "limit",
        "limit_price": str(limit_price),
        "time_in_force": "ioc",
        "client_order_id": f"svxy-ss-{client_tag}-{int(time.time() * 1000)}",
    }
    try:
        r = requests.post(f"{ENDPOINT}/orders",
                          headers=_headers(key, secret), json=payload, timeout=10)
        if r.status_code in (200, 201):
            return r.json()
        logger.error(f"order failed {r.status_code}: {r.text[:300]}")
    except Exception as exc:
        logger.error(f"order exception: {exc}")
    return None


def get_order(key: str, secret: str, order_id: str) -> Optional[dict]:
    try:
        r = requests.get(f"{ENDPOINT}/orders/{order_id}",
                         headers=_headers(key, secret), timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Market data: VIX, VIX3M, intraday SVXY
# ---------------------------------------------------------------------------
def fetch_vix_signal() -> Optional[tuple[float, float, float]]:
    """Return (vix_close_or_intraday, vix3m_close_or_intraday, ratio) via yfinance.

    Uses 1d period; yfinance gives latest available bar even during RTH.
    """
    try:
        import yfinance as yf
        vix_hist = yf.Ticker("^VIX").history(period="5d", interval="1d")
        v3_hist = yf.Ticker("^VIX3M").history(period="5d", interval="1d")
        if vix_hist.empty or v3_hist.empty:
            return None
        vix = float(vix_hist["Close"].iloc[-1])
        v3 = float(v3_hist["Close"].iloc[-1])
        if v3 <= 0:
            return None
        return vix, v3, vix / v3
    except Exception as exc:
        logger.warning(f"vix fetch failed: {exc}")
        return None


def fetch_vix_history(days: int) -> Optional["pd.DataFrame"]:
    """Pull VIX + VIX3M daily closes for the last `days` calendar days."""
    try:
        import pandas as pd
        import yfinance as yf
        # Add buffer for weekends/holidays
        start = (datetime.now(NY).date() - timedelta(days=days + 30)).isoformat()
        end = (datetime.now(NY).date() + timedelta(days=1)).isoformat()
        vix = yf.download("^VIX", start=start, end=end, progress=False, auto_adjust=False)
        v3 = yf.download("^VIX3M", start=start, end=end, progress=False, auto_adjust=False)
        if vix.empty or v3.empty:
            return None
        # Flatten potential multi-index columns from yfinance
        vix_close = vix["Close"]
        if hasattr(vix_close, "columns"):
            vix_close = vix_close.iloc[:, 0]
        v3_close = v3["Close"]
        if hasattr(v3_close, "columns"):
            v3_close = v3_close.iloc[:, 0]
        df = pd.DataFrame({"vix": vix_close, "vix3m": v3_close}).dropna()
        df["ratio"] = df["vix"] / df["vix3m"]
        return df.tail(days)
    except Exception as exc:
        logger.warning(f"vix history fetch failed: {exc}")
        return None


def fetch_svxy_intraday_change(key: str, secret: str) -> Optional[float]:
    """Return intraday % change of SVXY vs yesterday's close (decimal, e.g. -0.04)."""
    last = get_latest_trade(key, secret, SVXY)
    if last is None:
        return None
    try:
        import yfinance as yf
        hist = yf.Ticker(SVXY).history(period="5d", interval="1d")
        if hist.empty or len(hist) < 2:
            return None
        prior_close = float(hist["Close"].iloc[-2])
        if prior_close <= 0:
            return None
        return (last - prior_close) / prior_close
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Backtest verification on startup
# ---------------------------------------------------------------------------
def verify_signal_against_recent(days: int = VERIFY_LOOKBACK_DAYS) -> tuple[bool, dict]:
    """Replay the last `days` of VIX/VIX3M and count signal-active days.

    The sweep showed days_active = 92.81% over 2019-2026. Recent regimes can
    diverge, so we compare against a wide tolerance band centered on 85% +/- a
    healthy margin. We exit only if the recent active% is < (sweep_avg - 10%).
    """
    df = fetch_vix_history(days)
    out: dict = {"days": days, "ok": False}
    if df is None or df.empty:
        out["reason"] = "vix_history_unavailable"
        return False, out
    active = (df["ratio"] < THRESHOLD_RATIO).sum()
    n = len(df)
    active_pct = active / n if n else 0.0
    expected_pct = 0.9281  # from sweep
    divergence = abs(active_pct - expected_pct)
    out.update(
        actual_active_pct=round(active_pct * 100, 2),
        expected_active_pct=round(expected_pct * 100, 2),
        divergence_pct=round(divergence * 100, 2),
        n_days_sampled=int(n),
    )
    # Only fail if divergence > 10% AND recent active is BELOW expected (signal
    # collapse). Higher-than-expected activity is fine — that's just persistent
    # contango.
    if divergence > VERIFY_DIVERGENCE_TOLERANCE and active_pct < expected_pct:
        out["ok"] = False
        out["reason"] = f"recent active {active_pct:.1%} diverges > 10% from sweep {expected_pct:.1%}"
        return False, out
    out["ok"] = True
    return True, out


# ---------------------------------------------------------------------------
# SQLite
# ---------------------------------------------------------------------------
def ensure_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            ts_et  TEXT NOT NULL,
            account TEXT NOT NULL,
            vix REAL, vix3m REAL, ratio REAL,
            vix_intraday_max REAL,
            svxy_intraday_chg REAL,
            target TEXT,                 -- 'long' or 'flat'
            target_alloc_pct REAL,
            equity REAL,
            cash REAL,
            svxy_qty_held REAL,
            svxy_qty_target REAL,
            svxy_price REAL,
            sgov_qty_held REAL,
            sgov_price REAL,
            order_action TEXT,           -- buy/sell/none/halt/dry_run/...
            order_id TEXT,
            order_status TEXT,
            order_filled_qty REAL,
            order_filled_avg REAL,
            killswitch TEXT,             -- which kill if any
            equity_peak REAL,
            drawdown_pct REAL,
            confirmed INTEGER,           -- 1 if interactive confirm passed
            note TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS state (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_summary (
            date_et TEXT PRIMARY KEY,
            account TEXT,
            signal TEXT,
            decision TEXT,
            equity_open REAL,
            equity_close REAL,
            svxy_qty_close REAL,
            cash_close REAL,
            day_pnl_pct REAL,
            note TEXT
        )
    """)
    conn.commit()
    return conn


def state_get(conn: sqlite3.Connection, key: str, default: Optional[str] = None) -> Optional[str]:
    cur = conn.execute("SELECT value FROM state WHERE key=?", (key,))
    row = cur.fetchone()
    return row[0] if row else default


def state_set(conn: sqlite3.Connection, key: str, value: str) -> None:
    conn.execute("INSERT INTO state(key,value) VALUES(?,?) "
                 "ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, value))
    conn.commit()


def count_executed_trades(conn: sqlite3.Connection) -> int:
    cur = conn.execute(
        "SELECT COUNT(*) FROM decisions WHERE order_id IS NOT NULL AND order_id != ''"
    )
    return int(cur.fetchone()[0])


# ---------------------------------------------------------------------------
# Core decision logic
# ---------------------------------------------------------------------------
def compute_target(ratio: float, vix: float) -> tuple[str, float, Optional[str]]:
    """Return (target, alloc_pct, killswitch_reason).

    target ∈ {'long','flat'}. killswitch is non-None only if circuit breaker
    forces a flat target this iteration.
    """
    if vix > CB_VIX_LEVEL:
        return "flat", 0.0, f"VIX>{CB_VIX_LEVEL:.0f} (vix={vix:.2f})"
    if ratio < THRESHOLD_RATIO:
        return "long", SIZE_PCT, None
    return "flat", 0.0, None


def interactive_confirm(prompt: str) -> bool:
    """Block on user input. Returns True only on explicit 'y'."""
    try:
        sys.stdout.write(f"\n>>> CONFIRM: {prompt}  [y/N]: ")
        sys.stdout.flush()
        ans = sys.stdin.readline().strip().lower()
        return ans in {"y", "yes"}
    except Exception:
        return False


def decide_and_execute(
    conn: sqlite3.Connection,
    account: str,
    key: str,
    secret: str,
    *,
    dry_run: bool,
    require_confirm: bool,
    use_sgov: bool,
) -> str:
    """Run one decision cycle. Returns a status string."""
    now_utc = datetime.now(timezone.utc)
    now_et = datetime.now(NY)

    # --- 1. Pull signal ---
    sig = fetch_vix_signal()
    if sig is None:
        _log_decision_skip(conn, account, now_utc, now_et, "vix_unavailable")
        return "vix_unavailable"
    vix, vix3m, ratio = sig

    # --- 2. Account snapshot ---
    acct = get_account(key, secret)
    if not acct:
        _log_decision_skip(conn, account, now_utc, now_et, "account_unavailable")
        return "account_unavailable"
    equity = float(acct.get("equity", 0) or 0)
    cash = float(acct.get("cash", 0) or 0)
    if equity <= 0:
        _log_decision_skip(conn, account, now_utc, now_et, "equity<=0")
        return "equity_zero"

    # --- 3. Update equity peak / check rolling drawdown ---
    prior_peak = float(state_get(conn, "equity_peak", "0") or 0)
    equity_peak = max(prior_peak, equity)
    state_set(conn, "equity_peak", str(equity_peak))
    drawdown = (equity - equity_peak) / equity_peak if equity_peak > 0 else 0.0

    # --- 4. Position snapshots (broker truth) ---
    svxy_pos = get_position(key, secret, SVXY)
    held_svxy_qty = float(svxy_pos["qty"]) if svxy_pos else 0.0
    sgov_pos = get_position(key, secret, SGOV) if use_sgov else None
    held_sgov_qty = float(sgov_pos["qty"]) if sgov_pos else 0.0

    # --- 5. Prices ---
    svxy_px = get_latest_trade(key, secret, SVXY)
    if svxy_px is None or svxy_px <= 0:
        _log_decision_skip(conn, account, now_utc, now_et, "svxy_price_unavailable")
        return "svxy_price_unavailable"
    sgov_px = get_latest_trade(key, secret, SGOV) if use_sgov else None

    # --- 6. Intraday hard kills ---
    killswitch: Optional[str] = None
    halt_daemon = False
    svxy_chg = fetch_svxy_intraday_change(key, secret)
    if svxy_chg is not None and svxy_chg <= HARD_KILL_SVXY_INTRADAY_DROP:
        killswitch = f"SVXY intraday {svxy_chg:.2%} <= {HARD_KILL_SVXY_INTRADAY_DROP:.0%}"
        halt_daemon = True
    if drawdown <= HARD_KILL_ACCOUNT_DRAWDOWN:
        killswitch = (killswitch + " + " if killswitch else "") + \
            f"account dd {drawdown:.2%} <= {HARD_KILL_ACCOUNT_DRAWDOWN:.0%}"
        halt_daemon = True

    # --- 7. Compute target ---
    if killswitch and halt_daemon:
        target, alloc_pct = "flat", 0.0
    else:
        target, alloc_pct, cb_reason = compute_target(ratio, vix)
        if cb_reason and not killswitch:
            killswitch = cb_reason

    # --- 8. Decide SVXY trade ---
    target_usd = equity * alloc_pct
    target_qty = round(target_usd / svxy_px, 2) if svxy_px > 0 else 0.0
    delta_qty = target_qty - held_svxy_qty
    delta_usd = delta_qty * svxy_px

    msg = (f"account={account} vix={vix:.2f} vix3m={vix3m:.2f} ratio={ratio:.3f} "
           f"target={target} alloc={alloc_pct:.0%} eq=${equity:,.2f} "
           f"held={held_svxy_qty:.2f} target_qty={target_qty:.2f} "
           f"px=${svxy_px:.2f} delta=${delta_usd:+,.2f} "
           f"dd={drawdown:.2%} ks={killswitch or 'none'}")
    logger.info(msg)

    action = "no_change"
    order_id: Optional[str] = None
    order_status: Optional[str] = None
    filled_qty: Optional[float] = None
    filled_avg: Optional[float] = None
    confirmed_flag = 0
    note: Optional[str] = None

    if abs(delta_usd) < MIN_REBALANCE_USD:
        action = "no_change_threshold"
        note = f"|delta|=${abs(delta_usd):.2f} < ${MIN_REBALANCE_USD:.0f}"
    elif dry_run:
        action = "dry_run"
        note = f"would {'buy' if delta_qty > 0 else 'sell'} qty={delta_qty:+.2f}"
    else:
        side = "buy" if delta_qty > 0 else "sell"
        order_qty = abs(delta_qty)
        # Never sell more than we hold
        if side == "sell" and order_qty > held_svxy_qty:
            order_qty = held_svxy_qty
        if order_qty < 0.01:
            action = "no_change_too_small"
        else:
            # Interactive confirm for first N live trades
            executed_so_far = count_executed_trades(conn)
            if require_confirm and executed_so_far < FIRST_N_TRADES_REQUIRE_CONFIRM:
                ok = interactive_confirm(
                    f"{side.upper()} {order_qty:.2f} SVXY @ ~${svxy_px:.2f} "
                    f"(delta ${delta_usd:+,.2f}, equity ${equity:,.2f}) "
                    f"[trade #{executed_so_far + 1} of first {FIRST_N_TRADES_REQUIRE_CONFIRM}]"
                )
                if not ok:
                    action = "user_declined"
                    note = "interactive confirm rejected"
                    _record(conn, account, now_utc, now_et, vix, vix3m, ratio, None,
                            svxy_chg, target, alloc_pct, equity, cash, held_svxy_qty,
                            target_qty, svxy_px, held_sgov_qty, sgov_px, action,
                            order_id, order_status, filled_qty, filled_avg,
                            killswitch, equity_peak, drawdown, confirmed_flag, note)
                    return action
                confirmed_flag = 1
            tag = "buy" if side == "buy" else "sell"
            result = submit_marketable_limit_ioc(
                key, secret, SVXY, order_qty, side, svxy_px, tag,
            )
            if result:
                action = f"order_{side}"
                order_id = result.get("id", "")
                order_status = result.get("status")
                logger.info(f"submitted {side} {order_qty:.2f} SVXY id={order_id} status={order_status}")
                # Poll briefly for fill status (IOC: filled or canceled within seconds)
                if order_id:
                    for _ in range(6):
                        time.sleep(1)
                        upd = get_order(key, secret, order_id)
                        if upd:
                            order_status = upd.get("status")
                            if upd.get("filled_qty"):
                                try:
                                    filled_qty = float(upd["filled_qty"])
                                except Exception:
                                    pass
                            if upd.get("filled_avg_price"):
                                try:
                                    filled_avg = float(upd["filled_avg_price"])
                                except Exception:
                                    pass
                            if order_status in {"filled", "canceled", "expired", "rejected"}:
                                break
            else:
                action = "order_failed"
                note = "submit returned None"

    # --- 9. Optional SGOV cash-leg rebalance (75% of NAV) ---
    if use_sgov and sgov_px and sgov_px > 0 and not dry_run and not halt_daemon:
        cash_alloc = 1.0 - alloc_pct  # 0.75 when long, 1.0 when flat
        target_sgov_usd = equity * cash_alloc
        # leave a small cash buffer (1% of equity) to cover SVXY price drift
        target_sgov_usd = max(0.0, target_sgov_usd - 0.01 * equity)
        target_sgov_qty = round(target_sgov_usd / sgov_px, 2)
        sgov_delta_qty = target_sgov_qty - held_sgov_qty
        sgov_delta_usd = sgov_delta_qty * sgov_px
        if abs(sgov_delta_usd) >= MIN_REBALANCE_USD:
            side = "buy" if sgov_delta_qty > 0 else "sell"
            qty = abs(sgov_delta_qty)
            if side == "sell" and qty > held_sgov_qty:
                qty = held_sgov_qty
            if qty >= 0.01:
                submit_marketable_limit_ioc(key, secret, SGOV, qty, side, sgov_px, f"sgov-{side}")
                logger.info(f"SGOV rebalance {side} {qty:.2f} @ ${sgov_px:.2f}")

    # --- 10. Persist decision row ---
    _record(conn, account, now_utc, now_et, vix, vix3m, ratio, None, svxy_chg,
            target, alloc_pct, equity, cash, held_svxy_qty, target_qty, svxy_px,
            held_sgov_qty, sgov_px, action, order_id, order_status, filled_qty,
            filled_avg, killswitch, equity_peak, drawdown, confirmed_flag, note)

    # --- 11. Halt daemon if hard-kill fired ---
    if halt_daemon:
        global STOPPED
        STOPPED = True
        logger.error(f"HARD KILL: {killswitch} — halting daemon")
        return f"halt:{killswitch}"

    return action


def _record(conn, account, now_utc, now_et, vix, vix3m, ratio, vix_intraday_max,
            svxy_chg, target, alloc_pct, equity, cash, held_svxy_qty, target_qty,
            svxy_px, held_sgov_qty, sgov_px, action, order_id, order_status,
            filled_qty, filled_avg, killswitch, equity_peak, drawdown,
            confirmed_flag, note) -> None:
    conn.execute(
        """INSERT INTO decisions
           (ts_utc, ts_et, account, vix, vix3m, ratio, vix_intraday_max,
            svxy_intraday_chg, target, target_alloc_pct, equity, cash,
            svxy_qty_held, svxy_qty_target, svxy_price, sgov_qty_held, sgov_price,
            order_action, order_id, order_status, order_filled_qty,
            order_filled_avg, killswitch, equity_peak, drawdown_pct, confirmed, note)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (now_utc.isoformat(), now_et.isoformat(), account, vix, vix3m, ratio,
         vix_intraday_max, svxy_chg, target, alloc_pct, equity, cash,
         held_svxy_qty, target_qty, svxy_px, held_sgov_qty, sgov_px, action,
         order_id, order_status, filled_qty, filled_avg, killswitch,
         equity_peak, drawdown, confirmed_flag, note),
    )
    conn.commit()


def _log_decision_skip(conn, account, now_utc, now_et, reason: str) -> None:
    logger.warning(f"{account}: skip — {reason}")
    conn.execute(
        "INSERT INTO decisions (ts_utc, ts_et, account, order_action, note) "
        "VALUES (?,?,?,?,?)",
        (now_utc.isoformat(), now_et.isoformat(), account, "skip", reason),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Daily 16:30 ET summary
# ---------------------------------------------------------------------------
def write_daily_summary(conn: sqlite3.Connection, account: str, key: str, secret: str) -> None:
    now_et = datetime.now(NY)
    date_str = now_et.date().isoformat()
    cur = conn.execute(
        "SELECT target, order_action, ratio, equity, svxy_qty_held, cash "
        "FROM decisions WHERE account=? AND ts_et LIKE ? "
        "ORDER BY id DESC LIMIT 1",
        (account, f"{date_str}%"),
    )
    row = cur.fetchone()
    if not row:
        logger.info("no decisions today — skipping summary")
        return
    target, action, ratio, equity, qty, cash = row
    # Compute day-over-day PnL from yesterday's daily_summary if present
    cur2 = conn.execute(
        "SELECT equity_close FROM daily_summary "
        "WHERE account=? AND date_et < ? ORDER BY date_et DESC LIMIT 1",
        (account, date_str),
    )
    prev_row = cur2.fetchone()
    prev_equity = float(prev_row[0]) if prev_row else float(equity or 0)
    day_pnl_pct = ((equity - prev_equity) / prev_equity) if prev_equity > 0 else 0.0

    note = (f"signal_ratio={ratio:.3f} action={action} svxy_qty={qty}")
    conn.execute(
        """INSERT INTO daily_summary
           (date_et, account, signal, decision, equity_open, equity_close,
            svxy_qty_close, cash_close, day_pnl_pct, note)
           VALUES (?,?,?,?,?,?,?,?,?,?)
           ON CONFLICT(date_et) DO UPDATE SET
             account=excluded.account, signal=excluded.signal,
             decision=excluded.decision, equity_open=excluded.equity_open,
             equity_close=excluded.equity_close, svxy_qty_close=excluded.svxy_qty_close,
             cash_close=excluded.cash_close, day_pnl_pct=excluded.day_pnl_pct,
             note=excluded.note""",
        (date_str, account, target, action, prev_equity, equity, qty, cash,
         day_pnl_pct, note),
    )
    conn.commit()
    logger.info(
        f"DAILY SUMMARY {date_str} acct={account} signal={target} action={action} "
        f"eq=${equity:,.2f} day_pnl={day_pnl_pct:+.2%} qty_svxy={qty}"
    )


# ---------------------------------------------------------------------------
# Flatten-now utility
# ---------------------------------------------------------------------------
def flatten_now(account: str, key: str, secret: str) -> None:
    pos = get_position(key, secret, SVXY)
    if not pos:
        logger.info("no SVXY position to flatten")
        return
    qty = float(pos["qty"])
    if qty <= 0:
        logger.info(f"SVXY qty={qty} — nothing to flatten")
        return
    px = get_latest_trade(key, secret, SVXY) or 1.0
    result = submit_marketable_limit_ioc(key, secret, SVXY, qty, "sell", px, "flatten-now")
    logger.warning(f"FLATTEN-NOW: sold {qty:.2f} SVXY id={result.get('id') if result else 'FAIL'}")


# ---------------------------------------------------------------------------
# Daemon loop
# ---------------------------------------------------------------------------
def _signal_handler(_signum, _frame) -> None:
    global STOPPED
    STOPPED = True
    logger.info("stop signal received")


def daemon_loop(conn: sqlite3.Connection, account: str, key: str, secret: str,
                decision_time_et: str, summary_time_et: str, *,
                dry_run: bool, require_confirm: bool, use_sgov: bool) -> int:
    dh, dm = [int(x) for x in decision_time_et.split(":")]
    sh, sm = [int(x) for x in summary_time_et.split(":")]
    last_decision_date = None
    last_summary_date = None
    logger.info(f"daemon: decision at {decision_time_et} ET, summary at {summary_time_et} ET")
    while not STOPPED:
        if STOP_FILE.exists():
            logger.warning(f"STOP_FILE present at {STOP_FILE} — halting daemon (no flatten)")
            break
        now_et = datetime.now(NY)
        is_weekday = now_et.weekday() < 5
        dec_dt = now_et.replace(hour=dh, minute=dm, second=0, microsecond=0)
        sum_dt = now_et.replace(hour=sh, minute=sm, second=0, microsecond=0)

        if is_weekday and now_et >= dec_dt and now_et.date() != last_decision_date:
            decide_and_execute(conn, account, key, secret, dry_run=dry_run,
                               require_confirm=require_confirm, use_sgov=use_sgov)
            last_decision_date = now_et.date()
        if is_weekday and now_et >= sum_dt and now_et.date() != last_summary_date:
            write_daily_summary(conn, account, key, secret)
            last_summary_date = now_et.date()

        for _ in range(60):
            if STOPPED or STOP_FILE.exists():
                break
            time.sleep(1)
    logger.info("daemon exited cleanly")
    return 0


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(
        description="SVXY sweet-spot paper-shadow deployment host.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--account", default=os.environ.get("ALPACA_PAPER_ACCOUNT", "AEGIS"),
                   help="Alpaca paper account nickname (default: AEGIS).")
    p.add_argument("--once", action="store_true",
                   help="Run a single decision cycle and exit (cron-style).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute decisions but submit no orders.")
    p.add_argument("--confirm", action="store_true",
                   help=f"Require interactive [y/N] confirmation on the first "
                        f"{FIRST_N_TRADES_REQUIRE_CONFIRM} live trades.")
    p.add_argument("--decision-time-et",
                   default=os.environ.get("SVXY_DECISION_TIME_ET", "15:30"),
                   help="Daily signal-check time in ET (HH:MM). Default 15:30.")
    p.add_argument("--summary-time-et", default="16:30",
                   help="Daily end-of-day summary time in ET (HH:MM). Default 16:30.")
    p.add_argument("--sgov", action="store_true",
                   help="Place cash leg into SGOV (default: hold as cash).")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip the 90-day backtest verification on startup.")
    p.add_argument("--flatten-now", action="store_true",
                   help="Sell entire SVXY position immediately and exit.")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
    )

    load_env()
    key, secret = resolve_credentials(args.account)
    conn = ensure_db()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info("=" * 78)
    logger.info(f"SVXY sweet-spot host starting — account={args.account} "
                f"threshold<{THRESHOLD_RATIO} size={SIZE_PCT:.0%} cb=VIX>{CB_VIX_LEVEL:.0f}")
    logger.info(f"db={DB_PATH} log={LOG_FILE}")

    # Flatten-now path
    if args.flatten_now:
        flatten_now(args.account, key, secret)
        return 0

    # Sanity: SGOV asset support check
    use_sgov = args.sgov
    if use_sgov:
        if not get_asset_tradable(key, secret, SGOV):
            logger.warning("SGOV is not tradable on this paper account — falling back to cash leg")
            use_sgov = False

    # Backtest verification on startup
    if not args.skip_verify:
        ok, details = verify_signal_against_recent()
        logger.info(f"verify_90d: {json.dumps(details)}")
        if not ok:
            logger.error(f"backtest verification FAILED — divergence exceeds tolerance. {details}")
            return 2

    # First-trade confirm warning
    if args.confirm:
        executed = count_executed_trades(conn)
        if executed < FIRST_N_TRADES_REQUIRE_CONFIRM:
            logger.warning(f"interactive confirmation ENABLED for first "
                           f"{FIRST_N_TRADES_REQUIRE_CONFIRM - executed} live trades")
        else:
            logger.info("already passed first-N trades window — confirm is a no-op")

    if args.once:
        status = decide_and_execute(conn, args.account, key, secret,
                                    dry_run=args.dry_run,
                                    require_confirm=args.confirm,
                                    use_sgov=use_sgov)
        logger.info(f"once-result: {status}")
        # Also run summary if past summary time
        sh, sm = [int(x) for x in args.summary_time_et.split(":")]
        now_et = datetime.now(NY)
        if now_et >= now_et.replace(hour=sh, minute=sm, second=0, microsecond=0):
            write_daily_summary(conn, args.account, key, secret)
        return 0

    return daemon_loop(conn, args.account, key, secret,
                       args.decision_time_et, args.summary_time_et,
                       dry_run=args.dry_run, require_confirm=args.confirm,
                       use_sgov=use_sgov)


if __name__ == "__main__":
    raise SystemExit(main())
