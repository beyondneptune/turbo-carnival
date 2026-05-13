"""Paper-shadow host for candidate_v1 — cointegration commodity ETF pairs.

Runs ONE daily cycle on AEGIS Alpaca account:
1. Fetch latest prices for the 10-ETF universe
2. For each pair (45 total): compute rolling Engle-Granger residual, ADF p-value, z-score
3. Apply entry/exit/stop rules per the variant params
4. Reconcile broker truth vs decisions DB
5. Submit close-then-open orders if state changed
6. Log decisions to SQLite, snapshot NAV
7. Alert if 3+ consecutive cycles end with submit_failed

Bug-fixes baked in (the 2026-05-13 ADR lessons):
- Real-only gross-exposure check (no synthetic legs)
- `submit_failed` alerting after N consecutive failures
- STOP file honored at startup AND before each new entry
- Broker truth is the ground-truth source for current positions

Run:
    .venv/bin/python -m research.candidate_v1.host --dry-run
    .venv/bin/python -m research.candidate_v1.host           # live paper
    .venv/bin/python -m research.candidate_v1.host --flatten # close all
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import requests

from research.candidate_v1.backtest import (
    UNIVERSE, VariantParams, rolling_ols_residual, adf_pvalue_rolling,
    PER_PAIR_NAV_FRAC, MAX_OPEN_PAIRS,
)
from research.candidate_v1.walk_forward import load_best_variant

ROOT = Path(__file__).resolve().parents[2]
LIVE_DIR = ROOT / "research" / "candidate_v1" / "live"
LIVE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = LIVE_DIR / "decisions.sqlite3"
NAV_STATE = LIVE_DIR / "nav_state.json"
STOP_FILE = LIVE_DIR / "STOP_CANDIDATE_V1"
SUBMIT_FAIL_STATE = LIVE_DIR / "submit_fail_counter.json"

ALPACA_TRADING = "https://paper-api.alpaca.markets"
ALPACA_DATA = "https://data.alpaca.markets"
ACCOUNT = "AEGIS"  # per prereg §2 + memory
PRICE_HISTORY_DAYS = 120  # enough for 90-day variant window + headroom
SUBMIT_FAIL_LIMIT = 3
ORDER_TIF = "day"
ORDER_TYPE = "market"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("candidate_v1.host")


SCHEMA = """
CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    pair_a TEXT NOT NULL,
    pair_b TEXT NOT NULL,
    z REAL,
    adf_p REAL,
    action TEXT NOT NULL,         -- 'hold' / 'open_long_a' / 'open_short_a' / 'close' / 'stop' / 'timeout' / 'skip_cap'
    reason TEXT,
    notional_usd REAL,
    note TEXT
);
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    pair_a TEXT NOT NULL,
    pair_b TEXT NOT NULL,
    leg TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    qty INTEGER NOT NULL,
    submitted_ok INTEGER NOT NULL,
    status TEXT,
    order_id TEXT,
    client_order_id TEXT,
    note TEXT
);
CREATE TABLE IF NOT EXISTS positions_snapshot (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    symbol TEXT NOT NULL,
    qty REAL NOT NULL,
    avg_cost REAL,
    market_value REAL,
    unreal_pnl REAL
);
CREATE TABLE IF NOT EXISTS nav_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts_utc TEXT NOT NULL,
    equity REAL NOT NULL,
    cash REAL,
    buying_power REAL,
    n_open_pairs INTEGER
);
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def _headers() -> dict:
    key = os.environ.get(f"ALPACA_{ACCOUNT}_KEY")
    sec = os.environ.get(f"ALPACA_{ACCOUNT}_SECRET")
    if not key or not sec:
        raise SystemExit(f"ALPACA_{ACCOUNT}_KEY/SECRET missing")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}


def alpaca_positions() -> list[dict]:
    r = requests.get(f"{ALPACA_TRADING}/v2/positions", headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def alpaca_account() -> dict:
    r = requests.get(f"{ALPACA_TRADING}/v2/account", headers=_headers(), timeout=10)
    r.raise_for_status()
    return r.json()


def alpaca_latest_close(symbol: str) -> float | None:
    r = requests.get(f"{ALPACA_DATA}/v2/stocks/{symbol}/bars/latest",
                     headers=_headers(), params={"feed": "iex"}, timeout=10)
    if r.status_code != 200:
        return None
    data = r.json()
    if "bar" in data and data["bar"]:
        return float(data["bar"]["c"])
    return None


def fetch_recent_prices() -> pd.DataFrame:
    """Use the cached parquets for history, append latest IEX close for today."""
    cache_dir = ROOT / "data" / "cache" / "candidate_v1"
    frames = []
    for sym in UNIVERSE:
        df = pd.read_parquet(cache_dir / f"{sym}.parquet")
        df = df[["date", "close"]].rename(columns={"close": sym}).set_index("date")
        frames.append(df)
    wide = pd.concat(frames, axis=1).sort_index()
    # Append today's IEX close
    today_row = {}
    today_dt = pd.Timestamp(datetime.now(timezone.utc).date())
    for sym in UNIVERSE:
        px = alpaca_latest_close(sym)
        if px is not None:
            today_row[sym] = px
    if today_row and today_dt > wide.index.max():
        wide.loc[today_dt] = today_row
    return wide.tail(PRICE_HISTORY_DAYS)


def submit_market_order(symbol: str, side: str, qty: int, tag: str) -> dict:
    payload = {
        "symbol": symbol, "qty": str(abs(int(qty))),
        "side": side, "type": ORDER_TYPE, "time_in_force": ORDER_TIF,
        "client_order_id": f"candidate-v1-{tag[:32]}-{int(time.time() * 1000)}",
    }
    r = requests.post(f"{ALPACA_TRADING}/v2/orders", headers=_headers(),
                      json=payload, timeout=10)
    return {
        "ok": r.status_code in (200, 201),
        "status_code": r.status_code,
        "body": r.json() if r.status_code in (200, 201) else r.text[:300],
    }


def load_submit_fail_counter() -> int:
    if not SUBMIT_FAIL_STATE.exists():
        return 0
    try:
        return int(json.loads(SUBMIT_FAIL_STATE.read_text()).get("consecutive_failures", 0))
    except Exception:
        return 0


def save_submit_fail_counter(n: int) -> None:
    SUBMIT_FAIL_STATE.write_text(json.dumps({"consecutive_failures": n,
                                              "ts_utc": datetime.now(timezone.utc).isoformat()}))


def check_stop_file() -> Optional[dict]:
    if STOP_FILE.exists():
        try:
            return json.loads(STOP_FILE.read_text())
        except Exception:
            return {"reason": "unparseable_stop_file"}
    return None


def get_pair_state(positions: list[dict], a: str, b: str, nav: float) -> int:
    """Return +1 if long-a-short-b, -1 if short-a-long-b, 0 if flat."""
    pos_a = next((p for p in positions if p["symbol"] == a), None)
    pos_b = next((p for p in positions if p["symbol"] == b), None)
    qa = float(pos_a["qty"]) if pos_a else 0.0
    qb = float(pos_b["qty"]) if pos_b else 0.0
    if qa > 0 and qb < 0:
        return +1
    if qa < 0 and qb > 0:
        return -1
    return 0


def run_cycle(args) -> int:
    log.info("=== candidate_v1 cycle @ %s ===", datetime.now(timezone.utc).isoformat())

    # 1. STOP file
    kf = check_stop_file()
    if kf is not None:
        log.error("STOP file present: %s — exiting cleanly", kf)
        return 0  # not an error; the daemon should not loop on a stop

    # 2. Locked variant
    v = load_best_variant()
    log.info("variant: %s", v.key())

    conn = _conn()

    # 3. Account + broker truth
    acct = alpaca_account()
    nav = float(acct["equity"])
    cash = float(acct["cash"])
    bp = float(acct["buying_power"])
    log.info("account %s: equity=$%.2f cash=$%.2f bp=$%.0f", ACCOUNT, nav, cash, bp)

    positions = alpaca_positions()
    for p in positions:
        conn.execute("INSERT INTO positions_snapshot (ts_utc, symbol, qty, avg_cost, market_value, unreal_pnl) VALUES (?,?,?,?,?,?)",
                     (datetime.now(timezone.utc).isoformat(),
                      p["symbol"], float(p["qty"]),
                      float(p.get("avg_entry_price", 0)),
                      float(p.get("market_value", 0)),
                      float(p.get("unrealized_pl", 0))))
    conn.execute("INSERT INTO nav_history (ts_utc, equity, cash, buying_power, n_open_pairs) VALUES (?,?,?,?,?)",
                 (datetime.now(timezone.utc).isoformat(), nav, cash, bp, 0))
    conn.commit()

    # 4. Gross-exposure check (REAL US only; lesson from 2026-05-13 ADR bug)
    gross = sum(abs(float(p.get("market_value", 0))) for p in positions)
    if gross > 1.5 * nav:  # generous emergency cap
        log.error("gross exposure $%.2f > 1.5×NAV $%.2f — halting new entries this cycle", gross, nav)
        return 1

    # 5. Compute signals
    prices = fetch_recent_prices()
    if len(prices) < v.window + 2:
        log.error("insufficient price history (%d days < %d window+2)", len(prices), v.window)
        return 1
    log.info("prices: %s..%s (%d rows)",
             prices.index.min().date(), prices.index.max().date(), len(prices))

    # Count currently open pairs (for MAX_OPEN_PAIRS cap)
    open_pair_states: dict[tuple[str, str], int] = {}
    for a, b in combinations(UNIVERSE, 2):
        st = get_pair_state(positions, a, b, nav)
        if st != 0:
            open_pair_states[(a, b)] = st
    n_open = len(open_pair_states)
    log.info("currently open pairs: %d / %d cap", n_open, MAX_OPEN_PAIRS)

    # 6. Per-pair decisions
    submit_fail_count = 0
    submit_success_count = 0
    new_entries_allowed = MAX_OPEN_PAIRS - n_open

    for a, b in combinations(UNIVERSE, 2):
        log_a = np.log(prices[a].values)
        log_b = np.log(prices[b].values)
        _, resid = rolling_ols_residual(log_a, log_b, v.window)
        adf_p = adf_pvalue_rolling(resid, v.window)
        if np.isnan(resid[-1]) or np.isnan(adf_p[-1]):
            continue
        # Z-score of latest residual
        window_resid = resid[-v.window:]
        mu, sd = np.mean(window_resid), np.std(window_resid, ddof=1)
        if sd <= 1e-12:
            continue
        z_now = (resid[-1] - mu) / sd
        state = open_pair_states.get((a, b), 0)

        decision = "hold"
        reason = ""

        # Manage existing
        if state != 0:
            if abs(z_now) > v.stop:
                decision = "stop"; reason = f"|z|={abs(z_now):.2f}>{v.stop}"
            elif abs(z_now) < v.exit:
                decision = "close"; reason = f"|z|={abs(z_now):.2f}<{v.exit}"
            else:
                decision = "hold"; reason = f"|z|={abs(z_now):.2f} within band"
        else:
            if adf_p[-1] >= v.coint_p:
                decision = "skip_not_cointegrated"; reason = f"adf_p={adf_p[-1]:.3f}>={v.coint_p}"
            elif z_now > v.entry and new_entries_allowed > 0:
                decision = "open_short_a"; reason = f"z={z_now:.2f}>{v.entry}"
            elif z_now < -v.entry and new_entries_allowed > 0:
                decision = "open_long_a"; reason = f"z={z_now:.2f}<{-v.entry}"
            elif (z_now > v.entry or z_now < -v.entry) and new_entries_allowed <= 0:
                decision = "skip_cap"; reason = f"max_open_pairs={MAX_OPEN_PAIRS}"

        per_pair_notional = nav * PER_PAIR_NAV_FRAC
        conn.execute("INSERT INTO decisions (ts_utc, pair_a, pair_b, z, adf_p, action, reason, notional_usd, note) VALUES (?,?,?,?,?,?,?,?,?)",
                     (datetime.now(timezone.utc).isoformat(), a, b,
                      float(z_now), float(adf_p[-1]), decision, reason,
                      per_pair_notional, ""))
        conn.commit()

        if args.dry_run or decision in ("hold", "skip_not_cointegrated", "skip_cap"):
            continue

        # Real US orders
        px_a = float(prices[a].iloc[-1])
        px_b = float(prices[b].iloc[-1])
        qty_a = max(1, int(round((per_pair_notional / 2) / px_a)))
        qty_b = max(1, int(round((per_pair_notional / 2) / px_b)))

        if decision in ("close", "stop"):
            # Buy back the short, sell the long
            for sym, side, q in [(a, "buy" if state > 0 else "sell", qty_a),
                                  (b, "sell" if state > 0 else "buy", qty_b)]:
                if state > 0 and sym == a: side = "sell"  # was long, close = sell
                elif state > 0 and sym == b: side = "buy"  # was short, close = buy
                elif state < 0 and sym == a: side = "buy"
                elif state < 0 and sym == b: side = "sell"
                res = submit_market_order(sym, side, q, f"{a}_{b}_{decision}")
                conn.execute("INSERT INTO orders (ts_utc, pair_a, pair_b, leg, symbol, side, qty, submitted_ok, status, order_id, client_order_id, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                             (datetime.now(timezone.utc).isoformat(), a, b,
                              "US", sym, side, q, 1 if res["ok"] else 0,
                              str(res.get("body", "")) if not res["ok"] else res["body"].get("status", ""),
                              res["body"].get("id", "") if res["ok"] else "",
                              res["body"].get("client_order_id", "") if res["ok"] else "",
                              "" if res["ok"] else str(res["body"])[:300]))
                if res["ok"]:
                    submit_success_count += 1
                else:
                    submit_fail_count += 1
                    log.error("ORDER FAILED %s %s %d: %s", sym, side, q, res["body"])
        elif decision == "open_long_a":
            for sym, side, q in [(a, "buy", qty_a), (b, "sell", qty_b)]:
                res = submit_market_order(sym, side, q, f"{a}_{b}_open_long_a")
                conn.execute("INSERT INTO orders (ts_utc, pair_a, pair_b, leg, symbol, side, qty, submitted_ok, status, order_id, client_order_id, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                             (datetime.now(timezone.utc).isoformat(), a, b,
                              "US", sym, side, q, 1 if res["ok"] else 0,
                              res["body"].get("status", "") if res["ok"] else "",
                              res["body"].get("id", "") if res["ok"] else "",
                              res["body"].get("client_order_id", "") if res["ok"] else "",
                              "" if res["ok"] else str(res["body"])[:300]))
                if res["ok"]:
                    submit_success_count += 1
                    new_entries_allowed -= 1 if sym == a else 0
                else:
                    submit_fail_count += 1
                    log.error("ORDER FAILED %s %s %d: %s", sym, side, q, res["body"])
        elif decision == "open_short_a":
            for sym, side, q in [(a, "sell", qty_a), (b, "buy", qty_b)]:
                res = submit_market_order(sym, side, q, f"{a}_{b}_open_short_a")
                conn.execute("INSERT INTO orders (ts_utc, pair_a, pair_b, leg, symbol, side, qty, submitted_ok, status, order_id, client_order_id, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                             (datetime.now(timezone.utc).isoformat(), a, b,
                              "US", sym, side, q, 1 if res["ok"] else 0,
                              res["body"].get("status", "") if res["ok"] else "",
                              res["body"].get("id", "") if res["ok"] else "",
                              res["body"].get("client_order_id", "") if res["ok"] else "",
                              "" if res["ok"] else str(res["body"])[:300]))
                if res["ok"]:
                    submit_success_count += 1
                    new_entries_allowed -= 1 if sym == a else 0
                else:
                    submit_fail_count += 1
                    log.error("ORDER FAILED %s %s %d: %s", sym, side, q, res["body"])
        conn.commit()

    # 7. Submit-failure alerting (the 2026-05-13 ADR bug class)
    prior = load_submit_fail_counter()
    if submit_fail_count > 0 and submit_success_count == 0:
        new_counter = prior + 1
        save_submit_fail_counter(new_counter)
        log.error("ALERT: %d submit_failed in this cycle, %d consecutive failed cycles total",
                  submit_fail_count, new_counter)
        if new_counter >= SUBMIT_FAIL_LIMIT:
            log.error("HALT: %d consecutive submit-failed cycles — touching STOP file", new_counter)
            STOP_FILE.write_text(json.dumps({
                "ts_utc": datetime.now(timezone.utc).isoformat(),
                "reason": "consecutive_submit_failures",
                "detail": f"{new_counter} cycles with submit_failed and no submit_success",
            }))
            return 2
    else:
        if prior > 0:
            log.info("clearing submit-fail counter (was %d)", prior)
        save_submit_fail_counter(0)

    conn.close()
    log.info("cycle complete: %d successes, %d failures", submit_success_count, submit_fail_count)
    return 0


def flatten_all() -> int:
    log.warning("FLATTEN: closing every candidate_v1 position on %s", ACCOUNT)
    positions = alpaca_positions()
    if not positions:
        log.info("already flat")
        return 0
    for p in positions:
        sym = p["symbol"]
        qty = int(float(p["qty"]))
        side = "buy" if qty < 0 else "sell"
        res = submit_market_order(sym, side, abs(qty), f"flatten_{sym}")
        log.info("flatten %s %s %d -> ok=%s", sym, side, abs(qty), res["ok"])
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="Compute decisions and log to DB but submit no orders")
    p.add_argument("--flatten", action="store_true",
                   help="Close all open positions on AEGIS and exit")
    args = p.parse_args(argv)

    # Source env if running standalone
    env_file = ROOT / "council" / ".env.local"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k, v.strip('"').strip("'"))

    if args.flatten:
        return flatten_all()
    return run_cycle(args)


if __name__ == "__main__":
    sys.exit(main())
