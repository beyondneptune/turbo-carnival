"""SVXY regime rotation live trading host.

Base strategy (validated walk-forward 2019-2026: +218% / +17.1%/yr / Sharpe 0.66):
  - Fetch VIX, VIX3M at 09:35 ET each market day (post-open)
  - ratio = VIX / VIX3M
  - target = "long SVXY" if ratio < 1.0 (contango — positive roll yield)
  - target = "flat" if ratio >= 1.0 (backwardation — vol-spike risk)
  - If current position doesn't match target, execute trade.

Lane policy:
  - Hermes: core SVXY contango rotation, 30% allocation when ratio < 1.00.
  - Atlas: strict/deep-contango control lane, 30% allocation only when ratio < 0.85.
  - TERMINATOR: HFT-only account; no SVXY overlay.

Run as: .venv/bin/python scripts/svxy_rotation_host.py --lane HERMES
                                                       --lane ATLAS
                                                       --lane TERMINATOR
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
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

REPO = Path("/Users/weeoto/turbo-carnival")
ENV = REPO / "council" / ".env.local"
DB = REPO / "data" / "svxy_rotation_host.sqlite3"
LOG_DIR = REPO / "data" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
NY = ZoneInfo("America/New_York")
ENDPOINT = "https://paper-api.alpaca.markets/v2"
DATA_ENDPOINT = "https://data.alpaca.markets/v2"
SVXY = "SVXY"
TARGET_ALLOCATION = 0.30  # 30% of equity in SVXY when contango; rest cash
STOPPED = False

LANE_POLICIES = {
    "HERMES": {
        "allocation": 0.30,
        "entry_ratio": 1.00,
        "description": "core_contango_rotation",
    },
    "ATLAS": {
        "allocation": 0.30,
        "entry_ratio": 0.85,
        "description": "deep_contango_control",
    },
    "TERMINATOR": {
        "allocation": 0.00,
        "entry_ratio": 0.00,
        "description": "hft_only_no_svxy_overlay",
    },
}

logger = logging.getLogger("svxy_rotation_host")


def load_env() -> None:
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def _headers(prefix: str) -> dict:
    return {
        "APCA-API-KEY-ID": os.environ[f"ALPACA_{prefix}_KEY"],
        "APCA-API-SECRET-KEY": os.environ[f"ALPACA_{prefix}_SECRET"],
    }


def fetch_vix_ratio() -> tuple[float, float, float] | None:
    """Use yfinance to grab today's VIX + VIX3M close.

    Returns (vix, vix3m, ratio) or None if unavailable.
    """
    try:
        import yfinance as yf
        vix = yf.Ticker("^VIX").history(period="2d", interval="1d")
        vix3m = yf.Ticker("^VIX3M").history(period="2d", interval="1d")
        if vix.empty or vix3m.empty:
            return None
        v = float(vix["Close"].iloc[-1])
        v3 = float(vix3m["Close"].iloc[-1])
        return v, v3, v / v3
    except Exception as exc:
        logger.warning(f"vix fetch failed: {exc}")
        return None


def fetch_svxy_price(prefix: str) -> float | None:
    try:
        r = requests.get(f"{DATA_ENDPOINT}/stocks/{SVXY}/trades/latest",
                          headers=_headers(prefix), timeout=8)
        if r.status_code == 200 and r.json().get("trade"):
            return float(r.json()["trade"]["p"])
    except Exception:
        pass
    return None


def get_account(prefix: str) -> dict | None:
    try:
        r = requests.get(f"{ENDPOINT}/account", headers=_headers(prefix), timeout=8)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def get_svxy_position(prefix: str) -> dict | None:
    try:
        r = requests.get(f"{ENDPOINT}/positions/{SVXY}",
                          headers=_headers(prefix), timeout=8)
        if r.status_code == 200:
            return r.json()
        return None
    except Exception:
        return None


def submit_order(prefix: str, qty: float, side: str, lane: str) -> dict | None:
    payload = {
        "symbol": SVXY, "qty": str(qty), "side": side,
        "type": "market", "time_in_force": "day",
        "client_order_id": f"svxy-{lane}-{int(time.time())}",
    }
    try:
        r = requests.post(f"{ENDPOINT}/orders", headers=_headers(prefix),
                           json=payload, timeout=10)
        if r.status_code in (200, 201):
            return r.json()
        logger.error(f"order failed {r.status_code}: {r.text[:200]}")
    except Exception as exc:
        logger.error(f"order exception: {exc}")
    return None


def lane_policy(lane: str) -> dict:
    return LANE_POLICIES.get(
        lane.upper(),
        {
            "allocation": TARGET_ALLOCATION,
            "entry_ratio": 1.00,
            "description": "default_contango_rotation",
        },
    )


def ensure_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            lane TEXT NOT NULL,
            vix REAL, vix3m REAL, ratio REAL,
            target TEXT,
            held_qty REAL,
            equity REAL, svxy_price REAL,
            action TEXT,
            order_id TEXT,
            note TEXT
        )
    """)
    conn.commit()
    return conn


def decide_and_execute(lane: str, prefix: str, conn: sqlite3.Connection,
                       dry_run: bool = False) -> None:
    """Make today's decision for `lane`, execute if needed."""
    now = datetime.now(timezone.utc)
    vix_data = fetch_vix_ratio()
    if vix_data is None:
        logger.warning(f"{lane}: VIX unavailable, skipping")
        conn.execute("INSERT INTO decisions(ts_utc, lane, action, note) VALUES (?,?,?,?)",
                     (now.isoformat(), lane, "skip", "vix unavailable"))
        conn.commit()
        return

    vix, vix3m, ratio = vix_data
    policy = lane_policy(lane)
    allocation = float(policy["allocation"])
    entry_ratio = float(policy["entry_ratio"])
    policy_name = str(policy["description"])
    target = "long" if allocation > 0 and ratio < entry_ratio else "flat"
    acct = get_account(prefix)
    if not acct:
        logger.warning(f"{lane}: account unavailable")
        return
    equity = float(acct.get("equity", 0))
    cash = float(acct.get("cash", 0))
    pos = get_svxy_position(prefix)
    held_qty = float(pos["qty"]) if pos else 0.0

    svxy_px = fetch_svxy_price(prefix)
    if svxy_px is None or svxy_px <= 0:
        logger.warning(f"{lane}: SVXY price unavailable")
        return

    # Target position in dollars
    target_usd = equity * allocation if target == "long" else 0.0
    target_qty = target_usd / svxy_px if svxy_px > 0 else 0.0
    # Fractional shares OK on Alpaca paper. Round to 2 decimals.
    target_qty = round(target_qty, 2)
    delta_qty = target_qty - held_qty
    abs_delta_usd = abs(delta_qty) * svxy_px

    msg = (f"policy={policy_name} alloc={allocation:.0%} trigger<{entry_ratio:.2f}  "
            f"vix={vix:.2f} vix3m={vix3m:.2f} ratio={ratio:.3f} target={target}  "
            f"eq=${equity:,.2f} held_qty={held_qty:.2f} target_qty={target_qty:.2f}  "
            f"px=${svxy_px:.2f} delta=${delta_qty * svxy_px:+,.2f}")
    logger.info(f"{lane}: {msg}")

    action = "no_change"
    order_id = None
    note = None

    # Only trade if delta_usd > $50 (avoid micro-trades)
    if abs_delta_usd < 50:
        action = "no_change_threshold"
        note = f"delta ${abs_delta_usd:.2f} below $50 threshold"
    elif dry_run:
        action = "dry_run"
        note = f"would have traded qty={delta_qty:+.2f}"
    else:
        side = "buy" if delta_qty > 0 else "sell"
        # If selling more than we hold, sell only what we have
        order_qty = abs(delta_qty)
        if side == "sell" and order_qty > held_qty:
            order_qty = held_qty
        if order_qty < 0.01:
            action = "no_change_too_small"
        else:
            result = submit_order(prefix, round(order_qty, 2), side, lane)
            if result:
                action = f"order_{side}"
                order_id = result.get("id", "")
                logger.info(f"{lane}: submitted {side} {order_qty:.2f} SVXY  oid={order_id}")
            else:
                action = "order_failed"
                note = "submit returned None"

    conn.execute(
        "INSERT INTO decisions(ts_utc, lane, vix, vix3m, ratio, target, held_qty, "
        "equity, svxy_price, action, order_id, note) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (now.isoformat(), lane, vix, vix3m, ratio, target, held_qty, equity, svxy_px,
         action, order_id, note),
    )
    conn.commit()


def _signal_handler(_signum, _frame) -> None:
    global STOPPED
    STOPPED = True
    logger.info("stop signal received")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--lane", action="append", default=[],
                   help="Lane nickname (HERMES, TERMINATOR). Repeatable.")
    p.add_argument("--once", action="store_true",
                   help="Make one decision per lane and exit (cron-style).")
    p.add_argument("--dry-run", action="store_true",
                   help="Compute decisions but don't submit orders.")
    p.add_argument("--decision-time-et", default="09:40",
                   help="Daily decision time in ET (default 09:40 — 10 min after open).")
    args = p.parse_args()

    if not args.lane:
        args.lane = ["HERMES", "TERMINATOR"]

    # Configure logging
    log_file = LOG_DIR / "svxy_rotation_host.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    load_env()
    conn = ensure_db()
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    logger.info(f"SVXY rotation host starting — lanes: {args.lane} · dry_run={args.dry_run}")

    if args.once:
        for lane in args.lane:
            decide_and_execute(lane, lane, conn, dry_run=args.dry_run)
        return 0

    # Daemon mode: sleep until next decision time, then act, repeat
    decision_hour, decision_minute = [int(x) for x in args.decision_time_et.split(":")]
    last_decision_date = None
    while not STOPPED:
        ny_now = datetime.now(NY)
        is_weekday = ny_now.weekday() < 5
        decision_dt = ny_now.replace(hour=decision_hour, minute=decision_minute,
                                      second=0, microsecond=0)
        if is_weekday and ny_now >= decision_dt and ny_now.date() != last_decision_date:
            for lane in args.lane:
                decide_and_execute(lane, lane, conn, dry_run=args.dry_run)
            last_decision_date = ny_now.date()
        # Sleep 60s and re-check
        for _ in range(60):
            if STOPPED:
                break
            time.sleep(1)
    logger.info("stopped cleanly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
