"""iter_26 UK / FTSE 100 Weekly Mean-Reversion Paper-Shadow Deployment Host.

================================================================================
STATUS:   PAPER_SHADOW_READY  (NOT started — operator decides activation)
STRATEGY: see /research/candidates/iter_26_uk_ftse_meanrev.md
SOURCE:   /research/scripts/iter_26_backtest.py
PURPOSE:  fill the operator's 08:00-14:30 BST coverage gap by trading the
          EWU US-listed UK proxy during US RTH (no new broker subscription).

Rules (frozen on IS 2010-2022; do not edit after first run):
  Signal:        Monday 15:00 ET; r5 = EWU[t] / EWU[t-5] - 1
  Entry long:    r5 <= -0.030  -> BUY 20% NAV
  Entry short:   r5 >= +0.030  -> SHORT 20% NAV   (live-margin only)
  Hold:          5 trading days, exit MOC of day 5
  Take profit:   cum return  >= +3.0% (long) / <= -3.0% (short) at any close
  Stop loss:     cum return  <= -2.5% (long) / >= +2.5% (short) at any close
  Cost budget:   5 bps round-trip
  Size cap:      hard cap 25% NAV per position
  Daily DD cap:  -1.5% rolling 24h on the position
  Account DD:    -8% trailing peak-to-trough over 90d -> halt
  Kill file:     /Users/weeoto/turbo-carnival/research/live/STOP_UK_FTSE

================================================================================
DEPLOYMENT (operator-issued only — do NOT auto-launch)

Dry run (no orders, just compute signal & log):
    /Users/weeoto/turbo-carnival/.venv/bin/python \
        /Users/weeoto/turbo-carnival/research/live/iter_26_uk_ftse_host.py \
        --account AEGIS --dry-run

First live runs (interactive confirm on first 5 orders):
    /Users/weeoto/turbo-carnival/.venv/bin/python \
        /Users/weeoto/turbo-carnival/research/live/iter_26_uk_ftse_host.py \
        --account AEGIS --confirm

Daemon mode (after operator approves):
    /Users/weeoto/turbo-carnival/.venv/bin/python \
        /Users/weeoto/turbo-carnival/research/live/iter_26_uk_ftse_host.py \
        --account AEGIS

One-shot cron-style:
    /Users/weeoto/turbo-carnival/.venv/bin/python \
        /Users/weeoto/turbo-carnival/research/live/iter_26_uk_ftse_host.py \
        --account AEGIS --once

================================================================================
ENVIRONMENT
  Reads /Users/weeoto/turbo-carnival/council/.env.local then process env.

Required env vars (or fall back to ALPACA_API_KEY / ALPACA_API_SECRET):
    ALPACA_<ACCOUNT>_KEY     e.g. ALPACA_AEGIS_KEY
    ALPACA_<ACCOUNT>_SECRET  e.g. ALPACA_AEGIS_SECRET

Optional:
    ALPACA_PAPER_ACCOUNT     default: AEGIS
    UK_FTSE_DECISION_TIME_ET default: 15:00 (HH:MM Eastern)
    UK_FTSE_DECISION_DOW     default: 0     (Monday=0 ... Sunday=6)

================================================================================
KILL-SWITCH INSTRUCTIONS

Automatic kills (executed by the script):
    1. Daily P&L on position <= -1.5% NAV (24h)  -> close + halt new for 5 days
    2. Account drawdown < -8% trailing 90d        -> close + halt daemon
    3. /Users/weeoto/turbo-carnival/research/live/STOP_UK_FTSE exists -> halt

Manual halt:
    A. touch /Users/weeoto/turbo-carnival/research/live/STOP_UK_FTSE
    B. SIGINT / SIGTERM the process.
    C. Force-flatten now:
         /Users/weeoto/turbo-carnival/.venv/bin/python \
             /Users/weeoto/turbo-carnival/research/live/iter_26_uk_ftse_host.py \
             --account AEGIS --flatten-now

================================================================================
"""
from __future__ import annotations

import argparse
import json
import logging
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
DB_PATH = LIVE_DIR / "iter_26_uk_ftse_decisions.sqlite3"
LOG_DIR = REPO / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LIVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "iter_26_uk_ftse_host.log"
STOP_FILE = LIVE_DIR / "STOP_UK_FTSE"

NY = ZoneInfo("America/New_York")
ENDPOINT = "https://paper-api.alpaca.markets/v2"
DATA_ENDPOINT = "https://data.alpaca.markets/v2"

TICKER = "EWU"

# Strategy parameters (frozen on IS 2010-2022; mirror iter_26_backtest.py)
LOOKBACK_DAYS = 5
ENTRY_LONG_THRESHOLD = -0.030
ENTRY_SHORT_THRESHOLD = 0.030
HOLD_DAYS = 5
TP_LONG = 0.030
TP_SHORT = -0.030
SL_LONG = -0.025
SL_SHORT = 0.025
SIZE_NAV_PCT = 0.20
HARD_CAP_NAV_PCT = 0.25

# Risk envelope
DAILY_LOSS_CAP_NAV = -0.015         # -1.5% rolling 24h on position
ACCOUNT_DD_HALT = -0.08             # -8% trailing 90d account DD
HALT_DAYS_AFTER_LOSS_CAP = 5        # halt new entries for 5 days

# Execution
ORDER_PRICE_SLIP_BPS = 25           # 25 bps marketable-limit aggression
MIN_REBALANCE_USD = 50.0
FIRST_N_TRADES_REQUIRE_CONFIRM = 5

logger = logging.getLogger("iter_26_uk_ftse_host")
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
# Alpaca REST helpers
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
            return None
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


def submit_marketable_limit_ioc(
    key: str, secret: str, symbol: str, qty: float, side: str, ref_price: float,
    client_tag: str,
) -> Optional[dict]:
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
        "client_order_id": f"iter26-uk-{client_tag}-{int(time.time() * 1000)}",
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
# Market data: EWU 5-day return via yfinance
# ---------------------------------------------------------------------------
def fetch_r5_signal() -> Optional[tuple[float, float, float]]:
    """Return (latest_close, close_5d_ago, r5). Uses yfinance daily bars."""
    try:
        import yfinance as yf
        hist = yf.Ticker(TICKER).history(period="30d", interval="1d")
        if hist.empty or len(hist) < LOOKBACK_DAYS + 1:
            return None
        latest = float(hist["Close"].iloc[-1])
        prior = float(hist["Close"].iloc[-(LOOKBACK_DAYS + 1)])
        if prior <= 0:
            return None
        return latest, prior, latest / prior - 1.0
    except Exception as exc:
        logger.warning(f"yfinance fetch failed: {exc}")
        return None


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
            ewu_close REAL,
            ewu_close_5d_ago REAL,
            r5 REAL,
            signal TEXT,                 -- 'long' / 'short' / 'flat'
            equity REAL,
            cash REAL,
            held_qty REAL,
            target_qty REAL,
            entry_price REAL,
            cum_return REAL,
            held_bars INTEGER,
            exit_reason TEXT,            -- 'tp' / 'sl' / 'time' / null
            order_action TEXT,           -- buy/sell/short/cover/none/dry_run/...
            order_id TEXT,
            order_status TEXT,
            order_filled_qty REAL,
            order_filled_avg REAL,
            killswitch TEXT,
            equity_peak REAL,
            drawdown_pct REAL,
            confirmed INTEGER,
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
            action TEXT,
            equity_open REAL,
            equity_close REAL,
            qty_close REAL,
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
# Decision logic
# ---------------------------------------------------------------------------
def compute_signal(r5: float) -> str:
    if r5 <= ENTRY_LONG_THRESHOLD:
        return "long"
    if r5 >= ENTRY_SHORT_THRESHOLD:
        return "short"
    return "flat"


def interactive_confirm(prompt: str) -> bool:
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
) -> str:
    """One decision cycle. Returns status string."""
    now_utc = datetime.now(timezone.utc)
    now_et = datetime.now(NY)

    # --- 1. Pull signal ---
    sig = fetch_r5_signal()
    if sig is None:
        _log_skip(conn, account, now_utc, now_et, "r5_unavailable")
        return "r5_unavailable"
    latest_px, prior_px, r5 = sig

    # --- 2. Account snapshot ---
    acct = get_account(key, secret)
    if not acct:
        _log_skip(conn, account, now_utc, now_et, "account_unavailable")
        return "account_unavailable"
    equity = float(acct.get("equity", 0) or 0)
    cash = float(acct.get("cash", 0) or 0)
    if equity <= 0:
        _log_skip(conn, account, now_utc, now_et, "equity<=0")
        return "equity_zero"

    # --- 3. Equity peak / DD ---
    prior_peak = float(state_get(conn, "equity_peak", "0") or 0)
    equity_peak = max(prior_peak, equity)
    state_set(conn, "equity_peak", str(equity_peak))
    drawdown = (equity - equity_peak) / equity_peak if equity_peak > 0 else 0.0

    # --- 4. Position snapshot (broker truth) ---
    pos = get_position(key, secret, TICKER)
    held_qty = float(pos["qty"]) if pos else 0.0
    entry_price = float(pos["avg_entry_price"]) if pos else 0.0
    pos_side = 0
    if held_qty > 0:
        pos_side = 1
    elif held_qty < 0:
        pos_side = -1

    # --- 5. Current price ---
    px = get_latest_trade(key, secret, TICKER)
    if px is None or px <= 0:
        # Fall back to yfinance latest close
        px = latest_px

    # --- 6. If holding, evaluate exit conditions ---
    exit_reason: Optional[str] = None
    held_bars = 0
    cum_ret = 0.0
    if pos_side != 0 and entry_price > 0:
        cum_ret = pos_side * (px / entry_price - 1.0)
        # Held-bars from local state (we record entry timestamp)
        entry_ts_iso = state_get(conn, "open_entry_ts_et")
        if entry_ts_iso:
            try:
                entry_ts = datetime.fromisoformat(entry_ts_iso)
                # NaiveET -> compare same tz
                if entry_ts.tzinfo is None:
                    entry_ts = entry_ts.replace(tzinfo=NY)
                held_bars = max(0, (now_et.date() - entry_ts.date()).days)
            except Exception:
                held_bars = 0

        if pos_side == 1:
            if cum_ret <= SL_LONG:
                exit_reason = "sl"
            elif cum_ret >= TP_LONG:
                exit_reason = "tp"
            elif held_bars >= HOLD_DAYS:
                exit_reason = "time"
        else:
            if cum_ret <= -SL_SHORT:
                exit_reason = "sl"
            elif cum_ret >= -TP_SHORT:
                exit_reason = "tp"
            elif held_bars >= HOLD_DAYS:
                exit_reason = "time"

    # --- 7. Kill checks ---
    killswitch: Optional[str] = None
    halt_daemon = False
    if drawdown <= ACCOUNT_DD_HALT:
        killswitch = f"account_dd {drawdown:.2%} <= {ACCOUNT_DD_HALT:.0%}"
        halt_daemon = True

    # Daily loss-cap on position: pos_size * cum_ret <= -1.5% NAV
    if pos_side != 0:
        pos_pct_nav = (pos_side * held_qty * px) / equity if equity > 0 else 0.0
        position_dd = abs(pos_pct_nav) * cum_ret
        if position_dd <= DAILY_LOSS_CAP_NAV:
            killswitch = (killswitch + " + " if killswitch else "") + \
                f"position_dd {position_dd:.2%} <= {DAILY_LOSS_CAP_NAV:.1%}"
            exit_reason = exit_reason or "sl"
            # Mark a halt-new-entries window
            state_set(conn, "halt_entries_until_iso",
                      (now_et + timedelta(days=HALT_DAYS_AFTER_LOSS_CAP)).isoformat())

    # --- 8. Decide action ---
    signal_label = compute_signal(r5)
    target_side = 0
    if signal_label == "long":
        target_side = 1
    elif signal_label == "short":
        target_side = -1

    # If new entries are temporarily halted, suppress fresh entries (still allow exits).
    halt_until = state_get(conn, "halt_entries_until_iso")
    halt_active = False
    if halt_until:
        try:
            until_dt = datetime.fromisoformat(halt_until)
            if until_dt.tzinfo is None:
                until_dt = until_dt.replace(tzinfo=NY)
            halt_active = now_et < until_dt
        except Exception:
            halt_active = False

    action = "no_change"
    order_id: Optional[str] = None
    order_status: Optional[str] = None
    filled_qty: Optional[float] = None
    filled_avg: Optional[float] = None
    confirmed_flag = 0
    note: Optional[str] = None
    target_qty = 0.0

    msg = (f"account={account} ewu={px:.2f} r5={r5:+.4f} sig={signal_label} "
           f"held_side={pos_side} held_qty={held_qty:.2f} eq=${equity:,.2f} "
           f"cum_ret={cum_ret:+.2%} held_bars={held_bars} "
           f"dd={drawdown:.2%} exit_reason={exit_reason or 'none'} "
           f"halt_active={halt_active} ks={killswitch or 'none'}")
    logger.info(msg)

    if halt_daemon:
        # Force-flatten any position; do not enter new.
        if pos_side != 0 and not dry_run:
            side = "sell" if pos_side == 1 else "buy"
            qty = abs(held_qty)
            r = submit_marketable_limit_ioc(key, secret, TICKER, qty, side, px, "halt-flatten")
            if r:
                order_id = r.get("id", "")
                order_status = r.get("status")
                action = f"halt_flatten_{side}"
            else:
                action = "halt_flatten_failed"
        else:
            action = "halt_flatten_dryrun" if dry_run else "halt_no_pos"
        _record(conn, account, now_utc, now_et, latest_px, prior_px, r5,
                signal_label, equity, cash, held_qty, target_qty, entry_price,
                cum_ret, held_bars, exit_reason, action, order_id, order_status,
                filled_qty, filled_avg, killswitch, equity_peak, drawdown,
                confirmed_flag, note)
        global STOPPED
        STOPPED = True
        return f"halt:{killswitch}"

    # Priority 1: if we have a position with an exit_reason, EXIT.
    if pos_side != 0 and exit_reason is not None:
        side = "sell" if pos_side == 1 else "buy"
        qty = abs(held_qty)
        if dry_run:
            action = f"dry_run_exit_{side}_{exit_reason}"
            note = f"would {side} {qty:.2f} EWU on {exit_reason}"
        else:
            # interactive confirm for first N live trades
            executed_so_far = count_executed_trades(conn)
            if require_confirm and executed_so_far < FIRST_N_TRADES_REQUIRE_CONFIRM:
                ok = interactive_confirm(
                    f"{side.upper()} (EXIT/{exit_reason}) {qty:.2f} {TICKER} @ ~${px:.2f} "
                    f"cum_ret={cum_ret:+.2%} [trade #{executed_so_far + 1} of "
                    f"first {FIRST_N_TRADES_REQUIRE_CONFIRM}]"
                )
                if not ok:
                    action = "user_declined_exit"
                    _record(conn, account, now_utc, now_et, latest_px, prior_px, r5,
                            signal_label, equity, cash, held_qty, target_qty,
                            entry_price, cum_ret, held_bars, exit_reason, action,
                            order_id, order_status, filled_qty, filled_avg,
                            killswitch, equity_peak, drawdown, confirmed_flag,
                            "user declined exit")
                    return action
                confirmed_flag = 1
            r = submit_marketable_limit_ioc(key, secret, TICKER, qty, side, px,
                                            f"exit-{exit_reason}")
            if r:
                action = f"exit_{side}_{exit_reason}"
                order_id = r.get("id", "")
                order_status = r.get("status")
                state_set(conn, "open_entry_ts_et", "")  # clear entry timestamp
            else:
                action = "exit_failed"

    # Priority 2: if we have no position and the signal is non-flat (and we're
    # not in a post-loss-cap halt window), ENTER.
    elif pos_side == 0 and target_side != 0 and not halt_active:
        side = "buy" if target_side == 1 else "sell"  # 'sell' = short on Alpaca margin
        target_usd = equity * SIZE_NAV_PCT
        target_qty_raw = target_usd / px if px > 0 else 0.0
        # Apply hard cap
        hard_cap_usd = equity * HARD_CAP_NAV_PCT
        target_qty_raw = min(target_qty_raw, hard_cap_usd / px)
        qty = round(target_qty_raw, 2)
        delta_usd = qty * px
        if delta_usd < MIN_REBALANCE_USD:
            action = "no_change_too_small"
            note = f"target ${delta_usd:.2f} < ${MIN_REBALANCE_USD:.0f}"
        elif dry_run:
            action = f"dry_run_entry_{side}"
            note = f"would {side} {qty:.2f} EWU (signal={signal_label})"
        else:
            executed_so_far = count_executed_trades(conn)
            if require_confirm and executed_so_far < FIRST_N_TRADES_REQUIRE_CONFIRM:
                ok = interactive_confirm(
                    f"{side.upper()} (ENTRY/{signal_label}) {qty:.2f} {TICKER} "
                    f"@ ~${px:.2f} = ${delta_usd:,.2f} (equity ${equity:,.2f}) "
                    f"[trade #{executed_so_far + 1} of first "
                    f"{FIRST_N_TRADES_REQUIRE_CONFIRM}]"
                )
                if not ok:
                    action = "user_declined_entry"
                    _record(conn, account, now_utc, now_et, latest_px, prior_px, r5,
                            signal_label, equity, cash, held_qty, qty, entry_price,
                            cum_ret, held_bars, exit_reason, action, order_id,
                            order_status, filled_qty, filled_avg, killswitch,
                            equity_peak, drawdown, confirmed_flag,
                            "user declined entry")
                    return action
                confirmed_flag = 1
            r = submit_marketable_limit_ioc(key, secret, TICKER, qty, side, px,
                                            f"entry-{signal_label}")
            if r:
                action = f"entry_{side}_{signal_label}"
                order_id = r.get("id", "")
                order_status = r.get("status")
                state_set(conn, "open_entry_ts_et", now_et.isoformat())
                target_qty = qty if side == "buy" else -qty
            else:
                action = "entry_failed"
    else:
        # No-op cases: holding & no exit, or flat & no signal, or halt_active.
        if halt_active and target_side != 0:
            action = "halt_window_skip_entry"
            note = f"entries halted until {halt_until}"
        elif pos_side != 0:
            action = "hold_no_exit"
        else:
            action = "no_signal"

    # --- 9. Poll for fill if we placed an order ---
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

    # --- 10. Persist decision row ---
    _record(conn, account, now_utc, now_et, latest_px, prior_px, r5, signal_label,
            equity, cash, held_qty, target_qty, entry_price, cum_ret, held_bars,
            exit_reason, action, order_id, order_status, filled_qty, filled_avg,
            killswitch, equity_peak, drawdown, confirmed_flag, note)
    return action


def _record(conn, account, now_utc, now_et, ewu_close, ewu_close_5d_ago, r5,
            signal_label, equity, cash, held_qty, target_qty, entry_price,
            cum_ret, held_bars, exit_reason, order_action, order_id, order_status,
            filled_qty, filled_avg, killswitch, equity_peak, drawdown,
            confirmed_flag, note) -> None:
    conn.execute(
        """INSERT INTO decisions
           (ts_utc, ts_et, account, ewu_close, ewu_close_5d_ago, r5, signal,
            equity, cash, held_qty, target_qty, entry_price, cum_return,
            held_bars, exit_reason, order_action, order_id, order_status,
            order_filled_qty, order_filled_avg, killswitch, equity_peak,
            drawdown_pct, confirmed, note)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (now_utc.isoformat(), now_et.isoformat(), account, ewu_close,
         ewu_close_5d_ago, r5, signal_label, equity, cash, held_qty, target_qty,
         entry_price, cum_ret, held_bars, exit_reason, order_action, order_id,
         order_status, filled_qty, filled_avg, killswitch, equity_peak,
         drawdown, confirmed_flag, note),
    )
    conn.commit()


def _log_skip(conn, account, now_utc, now_et, reason: str) -> None:
    logger.warning(f"{account}: skip — {reason}")
    conn.execute(
        "INSERT INTO decisions (ts_utc, ts_et, account, order_action, note) "
        "VALUES (?,?,?,?,?)",
        (now_utc.isoformat(), now_et.isoformat(), account, "skip", reason),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Flatten-now utility
# ---------------------------------------------------------------------------
def flatten_now(account: str, key: str, secret: str) -> None:
    pos = get_position(key, secret, TICKER)
    if not pos:
        logger.info(f"no {TICKER} position to flatten")
        return
    qty = float(pos["qty"])
    if qty == 0:
        logger.info(f"{TICKER} qty={qty} — nothing to flatten")
        return
    side = "sell" if qty > 0 else "buy"
    px = get_latest_trade(key, secret, TICKER) or 1.0
    r = submit_marketable_limit_ioc(key, secret, TICKER, abs(qty), side, px, "flatten-now")
    logger.warning(f"FLATTEN-NOW: {side} {abs(qty):.2f} {TICKER} "
                   f"id={r.get('id') if r else 'FAIL'}")


# ---------------------------------------------------------------------------
# Daemon
# ---------------------------------------------------------------------------
def _signal_handler(_signum, _frame) -> None:
    global STOPPED
    STOPPED = True
    logger.info("stop signal received")


def daemon_loop(conn: sqlite3.Connection, account: str, key: str, secret: str,
                decision_time_et: str, decision_dow: int, *,
                dry_run: bool, require_confirm: bool) -> int:
    dh, dm = [int(x) for x in decision_time_et.split(":")]
    last_decision_date = None
    logger.info(f"daemon: weekly signal on dow={decision_dow} at {decision_time_et} ET")
    logger.info("DAILY check still runs (for exits / TP / SL / time-stop)")
    while not STOPPED:
        if STOP_FILE.exists():
            logger.warning(f"STOP_FILE present at {STOP_FILE} — halting daemon")
            break
        now_et = datetime.now(NY)
        is_weekday = now_et.weekday() < 5
        dec_dt = now_et.replace(hour=dh, minute=dm, second=0, microsecond=0)

        # Weekly entry signal: only fire on the configured dow.
        # Daily exit check: fires every weekday at the same time (cheap pull).
        if is_weekday and now_et >= dec_dt and now_et.date() != last_decision_date:
            decide_and_execute(conn, account, key, secret,
                               dry_run=dry_run, require_confirm=require_confirm)
            last_decision_date = now_et.date()

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
        description="iter_26 UK/FTSE EWU mean-reversion paper-shadow host.",
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
                   default=os.environ.get("UK_FTSE_DECISION_TIME_ET", "15:00"),
                   help="Daily signal-check time in ET (HH:MM). Default 15:00.")
    p.add_argument("--decision-dow", type=int,
                   default=int(os.environ.get("UK_FTSE_DECISION_DOW", "0")),
                   help="Weekly signal day of week (0=Mon ... 4=Fri). Default 0.")
    p.add_argument("--flatten-now", action="store_true",
                   help=f"Close entire {TICKER} position immediately and exit.")
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
    logger.info(f"iter_26 UK/FTSE host starting — account={args.account} "
                f"ticker={TICKER} thresholds=+/-{abs(ENTRY_LONG_THRESHOLD):.3f} "
                f"hold={HOLD_DAYS}d size={SIZE_NAV_PCT:.0%}")
    logger.info(f"db={DB_PATH} log={LOG_FILE}")

    if args.flatten_now:
        flatten_now(args.account, key, secret)
        return 0

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
                                    require_confirm=args.confirm)
        logger.info(f"once-result: {status}")
        return 0

    return daemon_loop(conn, args.account, key, secret,
                       args.decision_time_et, args.decision_dow,
                       dry_run=args.dry_run, require_confirm=args.confirm)


if __name__ == "__main__":
    raise SystemExit(main())
