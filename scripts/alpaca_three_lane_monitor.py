"""Monitor and exit Alpaca three-lane paper condors.

This watches the orders created by scripts/alpaca_three_lane_host.py and closes
the condor when one of the strategy exit rules fires:
  - profit target: close debit <= entry_credit * (1 - profit_take)
  - stop loss: close debit >= entry_credit * (1 + stop_loss_credit_mult)
  - DTE close: days to expiry <= close_dte

Default mode is read-only. Pass --place-orders to submit paper close orders.
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
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.alpaca_three_lane_host import (  # noqa: E402
    DEFAULT_DB,
    DEFAULT_ENV,
    DEFAULT_LOG,
    LANES,
    AlpacaClient,
    init_db,
    load_env,
    setup_logging,
    utc_now,
)

DATA_BASE = "https://data.alpaca.markets"
NY = ZoneInfo("America/New_York")


def ensure_monitor_tables(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS monitor_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            lane TEXT NOT NULL,
            trade_date TEXT NOT NULL,
            event_type TEXT NOT NULL,
            close_reason TEXT,
            close_debit REAL,
            entry_credit REAL,
            pnl_estimate REAL,
            response_json TEXT
        )
        """
    )
    conn.commit()


def lane_keys(lane_name: str) -> tuple[str, str]:
    for lane in LANES:
        if lane.nickname == lane_name:
            return lane.key_env, lane.secret_env
    raise KeyError(lane_name)


def lane_config(lane_name: str):
    for lane in LANES:
        if lane.nickname == lane_name:
            return lane
    raise KeyError(lane_name)


def quote_mid(q: dict[str, Any]) -> tuple[float, float, float]:
    bid = float(q.get("bp") or 0.0)
    ask = float(q.get("ap") or 0.0)
    if bid <= 0 or ask <= 0 or ask < bid:
        return bid, ask, 0.0
    return bid, ask, (bid + ask) / 2.0


def latest_option_quotes(client: AlpacaClient, symbols: list[str]) -> dict[str, Any]:
    payload = client.get(f"{DATA_BASE}/v1beta1/options/quotes/latest", symbols=",".join(symbols))
    return payload.get("quotes", {})


def current_option_position_qtys(client: AlpacaClient, symbols: list[str]) -> dict[str, int]:
    """Return current signed quantities for the monitored option symbols."""
    out: dict[str, int] = {}
    for sym in symbols:
        try:
            pos = client.get(f"/positions/{sym}")
            out[sym] = int(float(pos.get("qty") or 0))
        except Exception:
            out[sym] = 0
    return out


def payload_legs(open_payload: dict[str, Any]) -> list[dict[str, Any]]:
    legs = open_payload.get("legs")
    if legs:
        return list(legs)
    nested = open_payload.get("payload") or {}
    return list(nested.get("legs") or [])


def payload_client_order_id(open_payload: dict[str, Any]) -> str | int:
    nested = open_payload.get("payload") or {}
    return open_payload.get("client_order_id") or nested.get("client_order_id") or int(time.time())


def live_condor_qty(open_payload: dict[str, Any], qtys: dict[str, int]) -> tuple[int, str | None]:
    """Infer the live condor quantity from actual account positions.

    The opening order row may be stale after manual scale-downs. Before any
    monitor-triggered close, require all four legs to be present with the
    expected sign and use the smallest absolute live quantity.
    """
    expected: dict[str, int] = {}
    for leg in payload_legs(open_payload):
        sign = 1 if leg["side"] == "buy" else -1
        expected[leg["symbol"]] = sign

    abs_qtys: list[int] = []
    for sym, sign in expected.items():
        live_qty = qtys.get(sym, 0)
        if live_qty == 0:
            return 0, f"missing_leg:{sym}"
        if live_qty * sign <= 0:
            return 0, f"wrong_sign:{sym}:{live_qty}"
        abs_qtys.append(abs(live_qty))

    if not abs_qtys:
        return 0, "no_legs"
    if len(set(abs_qtys)) != 1:
        return min(abs_qtys), f"uneven_legs:{qtys}"
    return abs_qtys[0], None


def recent_open_orders(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    conn.row_factory = sqlite3.Row
    return conn.execute(
        """
        SELECT *
        FROM lane_orders
        WHERE status IN ('submitted', 'filled', 'close_submitted')
        ORDER BY created_at_utc DESC
        """
    ).fetchall()


def already_close_submitted(conn: sqlite3.Connection, lane: str, trade_date: str) -> bool:
    row = conn.execute(
        """
        SELECT 1 FROM monitor_events
        WHERE lane = ? AND trade_date = ? AND event_type = 'close_submitted'
        LIMIT 1
        """,
        (lane, trade_date),
    ).fetchone()
    return row is not None


def log_event(
    conn: sqlite3.Connection,
    *,
    lane: str,
    trade_date: str,
    event_type: str,
    close_reason: str | None,
    close_debit: float,
    entry_credit: float,
    pnl_estimate: float,
    response: dict[str, Any] | None = None,
) -> None:
    conn.execute(
        """
        INSERT INTO monitor_events (
            ts_utc, lane, trade_date, event_type, close_reason,
            close_debit, entry_credit, pnl_estimate, response_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            utc_now(),
            lane,
            trade_date,
            event_type,
            close_reason,
            close_debit,
            entry_credit,
            pnl_estimate,
            json.dumps(response or {}, default=str),
        ),
    )
    conn.commit()


def order_status(client: AlpacaClient, order_id: str) -> dict[str, Any]:
    return client.get(f"/orders/{order_id}")


def close_payload(open_payload: dict[str, Any], qty: int, limit_debit: float) -> dict[str, Any]:
    close_legs = []
    for leg in payload_legs(open_payload):
        side = leg["side"]
        close_side = "buy" if side == "sell" else "sell"
        intent = "buy_to_close" if side == "sell" else "sell_to_close"
        close_legs.append(
            {
                "symbol": leg["symbol"],
                "ratio_qty": leg.get("ratio_qty", "1"),
                "side": close_side,
                "position_intent": intent,
            }
        )
    return {
        "order_class": "mleg",
        "qty": str(qty),
        "type": "limit",
        "time_in_force": "day",
        "limit_price": f"{limit_debit:.2f}",
        "client_order_id": f"close-{payload_client_order_id(open_payload)}",
        "legs": close_legs,
    }


def evaluate_row(
    conn: sqlite3.Connection,
    row: sqlite3.Row,
    endpoint: str,
    *,
    place_orders: bool,
    force_close: bool = False,
) -> None:
    lane_name = str(row["lane"])
    lane = lane_config(lane_name)
    key_env, secret_env = lane_keys(lane_name)
    client = AlpacaClient(endpoint, os.environ[key_env], os.environ[secret_env])
    response = json.loads(row["response_json"] or "{}")
    order_id = response.get("id")
    if not order_id:
        return
    status = order_status(client, order_id)
    if status.get("status") != "filled":
        logging.info("%s opening order status=%s; no exit eval yet", lane_name, status.get("status"))
        return

    open_payload = json.loads(row["payload_json"])
    symbols = [
        open_payload["symbols"]["long_put"],
        open_payload["symbols"]["short_put"],
        open_payload["symbols"]["short_call"],
        open_payload["symbols"]["long_call"],
    ]
    quotes = latest_option_quotes(client, symbols)
    q_lp = quote_mid(quotes[symbols[0]])
    q_sp = quote_mid(quotes[symbols[1]])
    q_sc = quote_mid(quotes[symbols[2]])
    q_lc = quote_mid(quotes[symbols[3]])
    # Conservative executable debit: buy shorts at ask, sell longs at bid.
    close_debit = q_sp[1] + q_sc[1] - q_lp[0] - q_lc[0]
    mid_debit = q_sp[2] + q_sc[2] - q_lp[2] - q_lc[2]
    entry_credit = abs(float(status.get("filled_avg_price") or row["credit"] or 0.0))
    live_qtys = current_option_position_qtys(client, symbols)
    qty, structure_warning = live_condor_qty(open_payload, live_qtys)
    if qty <= 0:
        logging.warning("%s monitor structure mismatch; skipping close eval: %s", lane_name, structure_warning)
        log_event(
            conn,
            lane=lane_name,
            trade_date=str(row["trade_date"]),
            event_type="structure_mismatch",
            close_reason=structure_warning,
            close_debit=close_debit,
            entry_credit=entry_credit,
            pnl_estimate=0.0,
            response={"live_qtys": live_qtys},
        )
        return
    pnl_est = (entry_credit - close_debit) * 100.0 * qty
    expiry = datetime.fromisoformat(str(row["expiry"])).date()
    dte = (expiry - datetime.now(NY).date()).days

    reason = None
    if force_close:
        reason = "manual_stop"
    elif close_debit <= entry_credit * (1.0 - lane.profit_take):
        reason = "profit_target"
    elif close_debit >= entry_credit * (1.0 + lane.stop_loss_credit_mult):
        reason = "stop_loss"
    elif dte <= lane.close_dte:
        reason = "dte_close"

    logging.info(
        "%s monitor close_debit=%.2f mid_debit=%.2f entry=%.2f live_qty=%s pnl_est=%.2f dte=%s reason=%s structure=%s",
        lane_name,
        close_debit,
        mid_debit,
        entry_credit,
        qty,
        pnl_est,
        dte,
        reason,
        structure_warning,
    )
    log_event(
        conn,
        lane=lane_name,
        trade_date=str(row["trade_date"]),
        event_type="mark",
        close_reason=reason,
        close_debit=close_debit,
        entry_credit=entry_credit,
        pnl_estimate=pnl_est,
    )
    if not reason or already_close_submitted(conn, lane_name, str(row["trade_date"])):
        return

    limit_debit = max(0.01, round(max(close_debit, mid_debit) + 0.02, 2))
    payload = close_payload(open_payload, qty, limit_debit)
    if place_orders:
        try:
            close_resp = client.submit_order(payload)
            log_event(
                conn,
                lane=lane_name,
                trade_date=str(row["trade_date"]),
                event_type="close_submitted",
                close_reason=reason,
                close_debit=limit_debit,
                entry_credit=entry_credit,
                pnl_estimate=pnl_est,
                response=close_resp,
            )
            logging.warning("%s close submitted reason=%s id=%s", lane_name, reason, close_resp.get("id"))
        except Exception as exc:
            log_event(
                conn,
                lane=lane_name,
                trade_date=str(row["trade_date"]),
                event_type="close_error",
                close_reason=reason,
                close_debit=limit_debit,
                entry_credit=entry_credit,
                pnl_estimate=pnl_est,
                response={"error": str(exc)},
            )
            logging.exception("%s close submit failed", lane_name)
    else:
        logging.warning("%s would close reason=%s limit_debit=%.2f", lane_name, reason, limit_debit)


def monitor_loop(
    conn: sqlite3.Connection,
    endpoint: str,
    *,
    place_orders: bool,
    interval_sec: int,
    once: bool,
    lanes: set[str] | None,
    force_close_lanes: set[str],
) -> None:
    while True:
        rows = recent_open_orders(conn)
        if not rows:
            logging.info("no submitted lane orders to monitor")
        for row in rows:
            lane_name = str(row["lane"])
            if lanes is not None and lane_name not in lanes:
                continue
            try:
                evaluate_row(
                    conn,
                    row,
                    endpoint,
                    place_orders=place_orders,
                    force_close=lane_name in force_close_lanes,
                )
            except Exception:
                logging.exception("monitor row failed lane=%s", row["lane"])
        if once:
            return
        time.sleep(interval_sec)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--log", type=Path, default=REPO_ROOT / "data" / "logs" / "alpaca_three_lane_monitor.log")
    parser.add_argument("--place-orders", action="store_true")
    parser.add_argument("--interval-sec", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--lanes", default="", help="Comma-separated lane names to monitor; empty means all")
    parser.add_argument("--force-close-lanes", default="", help="Comma-separated lane names to close immediately")
    args = parser.parse_args()

    load_env(args.env)
    setup_logging(args.log)
    conn = init_db(args.db)
    ensure_monitor_tables(conn)
    endpoint = os.environ.get("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")
    lanes = {x.strip() for x in args.lanes.split(",") if x.strip()} or None
    force_close_lanes = {x.strip() for x in args.force_close_lanes.split(",") if x.strip()}
    logging.info(
        "monitor starting place_orders=%s interval=%s lanes=%s force_close_lanes=%s",
        args.place_orders,
        args.interval_sec,
        lanes or "ALL",
        force_close_lanes,
    )
    monitor_loop(
        conn,
        endpoint,
        place_orders=args.place_orders,
        interval_sec=args.interval_sec,
        once=args.once,
        lanes=lanes,
        force_close_lanes=force_close_lanes,
    )
    logging.info("monitor done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
