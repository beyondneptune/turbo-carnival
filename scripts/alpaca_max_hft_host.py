"""Max-risk Alpaca paper HFT-ish equity host.

Paper-only high-turnover host for the user's "maxed out HFT" lane.

This is not true exchange-colocated HFT. It is retail Alpaca REST scalping:
batch quotes, fast decision loop, market entries/exits, tight TP/SL/time exits,
and independent DB/logs so it can run beside the options-condor monitor.

Defaults are intentionally aggressive:
  - interval: 2 seconds
  - notional: 95% of equity per entry
  - max positions: 1 per account
  - max daily trades: 2000 per account
  - tight 5 bps TP / 4 bps stop / 45 sec time exit

Run dry:
    .venv/bin/python scripts/alpaca_max_hft_host.py --once

Run paper orders:
    .venv/bin/python scripts/alpaca_max_hft_host.py --place-orders
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import signal
import sqlite3
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.alpaca_three_lane_host import AlpacaClient, load_env, setup_logging  # noqa: E402


NY = ZoneInfo("America/New_York")
DEFAULT_ENV = REPO_ROOT / "council" / ".env.local"
DEFAULT_DB = REPO_ROOT / "data" / "alpaca_max_hft_host.sqlite3"
DEFAULT_LOG = REPO_ROOT / "data" / "logs" / "alpaca_max_hft_host.log"
DATA_BASE = "https://data.alpaca.markets"
DEFAULT_SYMBOLS = ("SPY", "QQQ", "NVDA", "TSLA", "AMD", "AAPL", "META")


@dataclass(frozen=True)
class HftLane:
    nickname: str
    account_env: str
    key_env: str
    secret_env: str
    mode: str
    min_signal_bps: float
    take_profit_bps: float
    stop_loss_bps: float
    max_hold_sec: float
    notional_pct: float = 0.95


LANE_TEMPLATES = [
    # ATHENA v2 stack: composite C3 gate + vol-inverse sizing + tightened thresholds
    # Aegis: HFT mean-reversion, tighter signal (z>=25 bps as before but blocked by C3)
    HftLane("Aegis", "ALPACA_AEGIS_ACCOUNT", "ALPACA_AEGIS_KEY", "ALPACA_AEGIS_SECRET", "mean_reversion", 25.0, 100.0, 35.0, 1200.0),
    # Falcon: HFT momentum/breakout
    HftLane("Falcon", "ALPACA_FALCON_ACCOUNT", "ALPACA_FALCON_KEY", "ALPACA_FALCON_SECRET", "breakout", 18.0, 70.0, 25.0, 900.0),
    HftLane("Valk", "ALPACA_VALK_ACCOUNT", "ALPACA_VALK_KEY", "ALPACA_VALK_SECRET", "turbo_hybrid", 18.0, 70.0, 25.0, 900.0),
    # TERMINATOR: $1k test bed running mean-reversion + composite C3 + Half-Kelly
    HftLane("TERMINATOR", "ALPACA_TERMINATOR_ACCOUNT", "ALPACA_TERMINATOR_KEY", "ALPACA_TERMINATOR_SECRET", "mean_reversion", 25.0, 100.0, 35.0, 1200.0, 0.30),
    HftLane("HFT1", "ALPACA_HFT1_ACCOUNT", "ALPACA_HFT1_KEY", "ALPACA_HFT1_SECRET", "momentum", 18.0, 70.0, 25.0, 900.0),
    HftLane("HFT2", "ALPACA_HFT2_ACCOUNT", "ALPACA_HFT2_KEY", "ALPACA_HFT2_SECRET", "mean_reversion", 25.0, 100.0, 35.0, 1200.0),
    HftLane("HFT3", "ALPACA_HFT3_ACCOUNT", "ALPACA_HFT3_KEY", "ALPACA_HFT3_SECRET", "breakout", 18.0, 70.0, 25.0, 900.0),
]


class StopFlag:
    stop = False


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ny_now() -> datetime:
    return datetime.now(NY)


def regular_market_open() -> bool:
    now = ny_now()
    if now.weekday() >= 5:
        return False
    t = now.time()
    return t >= datetime.strptime("09:30", "%H:%M").time() and t <= datetime.strptime("15:59", "%H:%M").time()


def near_eod() -> bool:
    return ny_now().time() >= datetime.strptime("15:55", "%H:%M").time()


def discover_lanes(requested: list[str] | None = None) -> list[HftLane]:
    wanted = {x.strip().lower() for x in requested} if requested else None
    lanes = []
    for lane in LANE_TEMPLATES:
        if wanted and lane.nickname.lower() not in wanted:
            continue
        if os.environ.get(lane.key_env) and os.environ.get(lane.secret_env):
            lanes.append(lane)
        else:
            logging.info("Skipping %s: missing %s/%s", lane.nickname, lane.key_env, lane.secret_env)
    return lanes


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            lane TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            reason TEXT NOT NULL,
            bid REAL,
            ask REAL,
            mid REAL,
            spread_bps REAL,
            signal_bps REAL,
            equity REAL,
            buying_power REAL,
            open_positions INTEGER,
            payload_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            lane TEXT NOT NULL,
            symbol TEXT NOT NULL,
            action TEXT NOT NULL,
            side TEXT NOT NULL,
            qty INTEGER,
            notional REAL,
            status TEXT NOT NULL,
            client_order_id TEXT,
            response_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS position_marks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_utc TEXT NOT NULL,
            lane TEXT NOT NULL,
            symbol TEXT NOT NULL,
            side TEXT NOT NULL,
            qty REAL,
            avg_entry REAL,
            mid REAL,
            pnl_bps REAL,
            age_sec REAL,
            exit_reason TEXT,
            raw_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS entry_state (
            lane TEXT NOT NULL,
            symbol TEXT NOT NULL,
            entry_ts_utc TEXT NOT NULL,
            side TEXT NOT NULL,
            client_order_id TEXT,
            PRIMARY KEY (lane, symbol)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS heartbeats (
            ts_utc TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            lanes TEXT NOT NULL,
            message TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def log_decision(
    conn: sqlite3.Connection,
    *,
    lane: str,
    symbol: str,
    action: str,
    reason: str,
    quote: dict[str, float] | None,
    signal_bps: float,
    equity: float,
    buying_power: float,
    open_positions: int,
    payload: dict[str, Any] | None = None,
) -> None:
    q = quote or {}
    conn.execute(
        """
        INSERT INTO decisions (
            ts_utc, lane, symbol, action, reason, bid, ask, mid, spread_bps,
            signal_bps, equity, buying_power, open_positions, payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            utc_now(),
            lane,
            symbol,
            action,
            reason,
            q.get("bid"),
            q.get("ask"),
            q.get("mid"),
            q.get("spread_bps"),
            signal_bps,
            equity,
            buying_power,
            open_positions,
            json.dumps(payload or {}, default=str),
        ),
    )
    conn.commit()


def log_order(
    conn: sqlite3.Connection,
    *,
    lane: str,
    symbol: str,
    action: str,
    side: str,
    qty: int,
    notional: float,
    status: str,
    client_order_id: str,
    response: dict[str, Any] | None,
) -> None:
    conn.execute(
        """
        INSERT INTO orders (
            ts_utc, lane, symbol, action, side, qty, notional,
            status, client_order_id, response_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            utc_now(),
            lane,
            symbol,
            action,
            side,
            qty,
            notional,
            status,
            client_order_id,
            json.dumps(response or {}, default=str),
        ),
    )
    conn.commit()


def remember_entry(conn: sqlite3.Connection, lane: str, symbol: str, side: str, client_order_id: str) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO entry_state (lane, symbol, entry_ts_utc, side, client_order_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (lane, symbol, utc_now(), side, client_order_id),
    )
    conn.commit()


def forget_entry(conn: sqlite3.Connection, lane: str, symbol: str) -> None:
    conn.execute("DELETE FROM entry_state WHERE lane = ? AND symbol = ?", (lane, symbol))
    conn.commit()


def entry_age_sec(conn: sqlite3.Connection, lane: str, symbol: str) -> float:
    row = conn.execute(
        "SELECT entry_ts_utc FROM entry_state WHERE lane = ? AND symbol = ?",
        (lane, symbol),
    ).fetchone()
    if not row:
        return 0.0
    try:
        t = datetime.fromisoformat(str(row[0]))
        return max(0.0, (datetime.now(timezone.utc) - t).total_seconds())
    except Exception:
        return 0.0


def latest_quotes(data_client: AlpacaClient, symbols: list[str]) -> dict[str, dict[str, float]]:
    payload = data_client.get(
        f"{DATA_BASE}/v2/stocks/quotes/latest",
        symbols=",".join(symbols),
        feed="iex",
    )
    out: dict[str, dict[str, float]] = {}
    for sym, q in (payload.get("quotes") or {}).items():
        bid = float(q.get("bp") or 0.0)
        ask = float(q.get("ap") or 0.0)
        if bid <= 0 or ask <= 0 or ask < bid:
            continue
        mid = (bid + ask) / 2.0
        out[sym] = {
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread_bps": (ask - bid) / mid * 10000.0 if mid else 9999.0,
        }
    return out


def parse_positions(raw: list[dict[str, Any]], universe: set[str]) -> dict[str, dict[str, Any]]:
    out = {}
    for p in raw:
        sym = str(p.get("symbol") or "")
        if not sym or sym not in universe:
            continue
        out[sym] = p
    return out


def update_history(quotes: dict[str, dict[str, float]], history: dict[str, deque[float]]) -> None:
    for sym, quote in quotes.items():
        history[sym].append(quote["mid"])


def signal_from_history(mode: str, symbol: str, history: dict[str, deque[float]]) -> tuple[float, str]:
    h = history[symbol]
    if len(h) < 4:
        return 0.0, "warming"
    one = (h[-1] / h[-2] - 1.0) * 10000.0
    three = (h[-1] / h[-4] - 1.0) * 10000.0
    if len(h) >= 12:
        twelve = (h[-1] / h[-12] - 1.0) * 10000.0
    else:
        twelve = three

    momentum = 0.65 * three + 0.35 * one
    reversion = -0.80 * one - 0.35 * three if abs(three) > 1.5 else -0.20 * twelve
    breakout = twelve + 0.40 * three if abs(twelve) > 3.0 else 0.35 * momentum
    if mode == "momentum":
        score = momentum
    elif mode == "mean_reversion":
        score = reversion
    elif mode == "breakout":
        score = breakout
    elif mode == "turbo_hybrid":
        if abs(twelve) > 8.0:
            score = breakout
        elif abs(three) > 2.0:
            score = momentum
        else:
            score = reversion
    else:
        score = momentum
    reason = f"mode={mode} one={one:+.2f}bps three={three:+.2f}bps twelve={twelve:+.2f}bps"
    return score, reason


def best_signal(
    mode: str,
    quotes: dict[str, dict[str, float]],
    history: dict[str, deque[float]],
    *,
    min_signal_bps: float,
    max_spread_bps: float,
    allow_shorts: bool,
    open_symbols: set[str],
) -> tuple[str | None, int, float, str]:
    best_sym = None
    best_score = 0.0
    best_reason = ""
    for sym, quote in quotes.items():
        score, reason = signal_from_history(mode, sym, history)
        if sym in open_symbols:
            continue
        if quote["spread_bps"] > max_spread_bps:
            continue
        if abs(score) < min_signal_bps:
            continue
        if score < 0 and not allow_shorts:
            continue
        if abs(score) > abs(best_score):
            best_sym = sym
            best_score = score
            best_reason = reason
    if best_sym is None:
        return None, 0, 0.0, "no_signal"
    return best_sym, 1 if best_score > 0 else -1, best_score, best_reason


def client_order_id(lane: str, action: str, symbol: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")[:-3]
    rand = random.randint(100, 999)
    return f"hft-{lane.lower()}-{action}-{symbol.lower()}-{stamp}-{rand}"[:48]


def submit_market_order(
    client: AlpacaClient,
    *,
    lane: str,
    symbol: str,
    side: str,
    qty: int,
    action: str,
    place_orders: bool,
) -> tuple[str, str, dict[str, Any]]:
    cid = client_order_id(lane, action, symbol)
    payload = {
        "symbol": symbol,
        "qty": str(qty),
        "side": side,
        "type": "market",
        "time_in_force": "day",
        "client_order_id": cid,
    }
    if not place_orders:
        return cid, "dry_run", {"dry_run": True, "payload": payload}
    try:
        response = client.post("/orders", payload)
        return cid, str(response.get("status") or "submitted"), response
    except Exception as exc:
        return cid, "error", {"error": str(exc), "payload": payload}


def open_order_symbols(client: AlpacaClient) -> set[str]:
    try:
        orders = client.get("/orders", status="open", limit=100, nested="false")
    except Exception:
        return set()
    out = set()
    if isinstance(orders, list):
        for order in orders:
            sym = str(order.get("symbol") or "")
            if sym:
                out.add(sym)
    return out


def position_pnl_bps(position: dict[str, Any], quote: dict[str, float]) -> tuple[float, str, float]:
    side = str(position.get("side") or "long").lower()
    qty = abs(float(position.get("qty") or 0.0))
    avg = float(position.get("avg_entry_price") or 0.0)
    if avg <= 0 or qty <= 0:
        return 0.0, side, qty
    mid = quote["mid"]
    direction = 1.0 if side == "long" else -1.0
    return (mid / avg - 1.0) * 10000.0 * direction, side, qty


def evaluate_exits(
    conn: sqlite3.Connection,
    client: AlpacaClient,
    lane: HftLane,
    positions: dict[str, dict[str, Any]],
    quotes: dict[str, dict[str, float]],
    *,
    take_profit_bps: float,
    stop_loss_bps: float,
    max_hold_sec: float,
    place_orders: bool,
    pending_symbols: set[str] | None = None,
) -> int:
    exits = 0
    pending_symbols = pending_symbols or set()
    for sym, pos in positions.items():
        if sym not in quotes:
            continue
        if sym in pending_symbols:
            continue
        pnl_bps, side, qty_f = position_pnl_bps(pos, quotes[sym])
        qty = int(abs(qty_f))
        if qty <= 0:
            continue
        age = entry_age_sec(conn, lane.nickname, sym)
        reason = None
        if near_eod():
            reason = "eod_flat"
        elif pnl_bps >= take_profit_bps:
            reason = "take_profit"
        elif pnl_bps <= -abs(stop_loss_bps):
            reason = "stop_loss"
        elif age >= max_hold_sec > 0:
            reason = "time_exit"

        conn.execute(
            """
            INSERT INTO position_marks (
                ts_utc, lane, symbol, side, qty, avg_entry, mid, pnl_bps,
                age_sec, exit_reason, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_now(),
                lane.nickname,
                sym,
                side,
                qty_f,
                float(pos.get("avg_entry_price") or 0.0),
                quotes[sym]["mid"],
                pnl_bps,
                age,
                reason,
                json.dumps(pos, default=str),
            ),
        )
        conn.commit()

        if not reason:
            continue
        close_side = "sell" if side == "long" else "buy"
        cid, status, response = submit_market_order(
            client,
            lane=lane.nickname,
            symbol=sym,
            side=close_side,
            qty=qty,
            action=reason,
            place_orders=place_orders,
        )
        log_order(
            conn,
            lane=lane.nickname,
            symbol=sym,
            action=reason,
            side=close_side,
            qty=qty,
            notional=qty * quotes[sym]["mid"],
            status=status,
            client_order_id=cid,
            response=response,
        )
        if status != "error":
            forget_entry(conn, lane.nickname, sym)
        logging.info(
            "%s EXIT %s %s qty=%s pnl=%.2fbps reason=%s status=%s",
            lane.nickname,
            close_side.upper(),
            sym,
            qty,
            pnl_bps,
            reason,
            status,
        )
        exits += 1
    return exits


def trades_today(conn: sqlite3.Connection, lane: str) -> int:
    prefix = datetime.now(timezone.utc).date().isoformat()
    row = conn.execute(
        """
        SELECT COUNT(*) FROM orders
        WHERE lane = ? AND ts_utc >= ? AND action IN ('entry_long', 'entry_short')
        """,
        (lane, prefix),
    ).fetchone()
    return int(row[0] or 0)


def run_loop(args: argparse.Namespace) -> None:
    load_env(DEFAULT_ENV)
    setup_logging(Path(args.log))
    conn = init_db(Path(args.db))
    endpoint = os.environ.get("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")
    requested_lanes = [x.strip() for x in args.lanes.split(",") if x.strip()] if args.lanes else None
    lanes = discover_lanes(requested_lanes)
    if not lanes:
        raise RuntimeError("No HFT lanes found. Add ALPACA_VALK_KEY/SECRET or ALPACA_HFT1_KEY/SECRET to council/.env.local")

    clients = {
        lane.nickname: AlpacaClient(endpoint, os.environ[lane.key_env], os.environ[lane.secret_env])
        for lane in lanes
    }
    data_client = clients[lanes[0].nickname]
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=max(args.history_ticks, 12)))
    logging.info(
        "START max_hft lanes=%s symbols=%s interval=%.2fs place_orders=%s lane_presets=%s",
        [f"{l.nickname}:{l.mode}" for l in lanes],
        symbols,
        args.interval_sec,
        args.place_orders,
        not args.no_lane_presets,
    )

    while not StopFlag.stop:
        loop_start = time.time()
        try:
            if not regular_market_open() and not args.ignore_market_hours:
                logging.info("Market closed; sleeping %.1fs", args.closed_sleep_sec)
                if args.once:
                    break
                time.sleep(args.closed_sleep_sec)
                continue

            quotes = latest_quotes(data_client, symbols)
            if not quotes:
                logging.warning("No quotes returned")
                if args.once:
                    break
                time.sleep(args.interval_sec)
                continue
            update_history(quotes, history)

            for lane in lanes:
                client = clients[lane.nickname]
                account = client.get("/account")
                equity = float(account.get("equity") or 0.0)
                buying_power = float(account.get("buying_power") or 0.0)
                positions = parse_positions(client.get("/positions"), set(symbols))
                pending_symbols = open_order_symbols(client)
                open_symbols = set(positions) | pending_symbols

                evaluate_exits(
                    conn,
                    client,
                    lane,
                    positions,
                    quotes,
                    take_profit_bps=lane.take_profit_bps if not args.no_lane_presets else args.take_profit_bps,
                    stop_loss_bps=lane.stop_loss_bps if not args.no_lane_presets else args.stop_loss_bps,
                    max_hold_sec=lane.max_hold_sec if not args.no_lane_presets else args.max_hold_sec,
                    place_orders=args.place_orders,
                    pending_symbols=pending_symbols,
                )

                if near_eod() and not args.allow_entries_near_eod:
                    continue
                if trades_today(conn, lane.nickname) >= args.max_trades_per_day:
                    log_decision(
                        conn,
                        lane=lane.nickname,
                        symbol="*",
                        action="skip",
                        reason="max_trades_per_day",
                        quote=None,
                        signal_bps=0.0,
                        equity=equity,
                        buying_power=buying_power,
                        open_positions=len(positions),
                    )
                    continue
                if len(positions) >= args.max_positions:
                    continue

                sym, direction, score, reason = best_signal(
                    lane.mode,
                    quotes,
                    history,
                    min_signal_bps=lane.min_signal_bps if not args.no_lane_presets else args.min_signal_bps,
                    max_spread_bps=args.max_spread_bps,
                    allow_shorts=not args.long_only,
                    open_symbols=open_symbols,
                )
                if not sym:
                    log_decision(
                        conn,
                        lane=lane.nickname,
                        symbol="*",
                        action="skip",
                        reason=reason,
                        quote=None,
                        signal_bps=0.0,
                        equity=equity,
                        buying_power=buying_power,
                        open_positions=len(positions),
                    )
                    continue

                # Regime-filter gate (vol-of-vol, ATR, news blackout, abs-return).
                # Source: 60-model swarm consensus (Opus max #9 top filter).
                # Config: data/regime_filters_config.json — edit to tune.
                try:
                    from datetime import datetime as _dt, timezone as _tz
                    from scripts.regime_filters import block_entry as _block_entry
                    blocked, why = _block_entry(
                        lane.nickname, sym, history[sym], now_utc=_dt.now(_tz.utc),
                    )
                    if blocked:
                        log_decision(
                            conn,
                            lane=lane.nickname,
                            symbol=sym,
                            action="skip",
                            reason=f"regime_filter:{why}",
                            quote=quotes.get(sym),
                            signal_bps=score,
                            equity=equity,
                            buying_power=buying_power,
                            open_positions=len(positions),
                        )
                        continue
                except Exception as _filt_exc:
                    logging.warning("regime_filter error (continuing): %s", _filt_exc)

                # ATHENA v2: COMPOSITE C3 GATE
                # Trained on 21 days (validated via walk-forward at +$1,288 honest delta
                # vs yesterday's +$2,030 lookahead-biased number).
                # Block rule: ANY of {016 LOF, 017 OCSVM, 018 EllipticEnvelope, 102 RBF+LR}.
                # Fails open (allows entry) if models can't load — graceful degradation.
                try:
                    from scripts.athena_ml_gates import composite_c3_blocks as _c3_blocks
                    import numpy as _np
                    # Build a feature dict from rolling history. We use last bar's stats.
                    # If history hasn't accumulated 20+ samples, skip the gate (allow).
                    if len(history[sym]) >= 20:
                        prices = list(history[sym])
                        ret1 = (prices[-1] / prices[-2] - 1) if prices[-2] > 0 else 0
                        ret3 = (prices[-1] / prices[-4] - 1) if len(prices) >= 4 and prices[-4] > 0 else 0
                        ret10 = (prices[-1] / prices[-11] - 1) if len(prices) >= 11 and prices[-11] > 0 else 0
                        # rough range_pct approximation (intra-bar high-low not in history)
                        recent = prices[-20:]
                        range_pct = (max(recent) - min(recent)) / max(recent) if max(recent) > 0 else 0
                        # Volume z and ret1 z — approximate with rolling stdev
                        rets = [(prices[i] / prices[i-1] - 1) for i in range(1, len(prices))]
                        if len(rets) >= 5:
                            mean_r = sum(rets[-20:]) / min(20, len(rets))
                            std_r = (sum((r - mean_r)**2 for r in rets[-20:]) / min(20, len(rets))) ** 0.5
                            ret1_z = (ret1 - mean_r) / std_r if std_r > 0 else 0
                        else:
                            ret1_z = 0
                        ny_t = ny_now()
                        mins = max(0, min(390, (ny_t.hour - 9) * 60 + ny_t.minute - 30))
                        tod_sin = _np.sin(2 * _np.pi * mins / 390)
                        tod_cos = _np.cos(2 * _np.pi * mins / 390)
                        feats = {
                            "ret1": float(ret1), "ret3": float(ret3), "ret10": float(ret10),
                            "range_pct": float(range_pct),
                            "volume_z": 0.0, "ret1_z": float(ret1_z),
                            "vwap_dist": 0.0,
                            "tod_sin": float(tod_sin), "tod_cos": float(tod_cos),
                            "vov": 1.0,
                        }
                        c3_block, c3_info = _c3_blocks(feats)
                        if c3_block:
                            log_decision(
                                conn, lane=lane.nickname, symbol=sym, action="skip",
                                reason=f"composite_c3:{','.join(c3_info.get('blockers', []))}",
                                quote=quotes.get(sym), signal_bps=score,
                                equity=equity, buying_power=buying_power,
                                open_positions=len(positions),
                                payload={"c3_info": c3_info},
                            )
                            continue
                except Exception as _c3_exc:
                    logging.warning("composite_c3 error (continuing — fail open): %s", _c3_exc)

                quote = quotes[sym]
                # ATHENA v2: VOL-INVERSE SIZING (vs old fixed notional_pct)
                # Sizes position inversely to recent realized vol — quiet bars get more,
                # noisy bars get less. For TERMINATOR ($1k), uses Half-Kelly cap.
                try:
                    from scripts.voltgt_sizer import compute_notional as _vt_notional, compute_realized_vol_bps as _rv_bps
                    if len(history[sym]) >= 6:
                        recent_rets = [(history[sym][i] / history[sym][i-1] - 1)
                                       for i in range(-5, 0) if history[sym][i-1] > 0]
                        rv = _rv_bps(recent_rets)
                        # TERMINATOR uses much smaller base ($12k → $150 since $1k account)
                        is_tt = lane.nickname == "TERMINATOR"
                        base_notional = 150 if is_tt else 12000
                        floor = 50 if is_tt else 4000
                        ceiling = 300 if is_tt else 20000
                        notional = _vt_notional(rv, base_notional=base_notional,
                                                 floor=floor, ceiling=ceiling)
                    else:
                        # Fall back to old sizing if no vol estimate yet
                        lane_notional_pct = lane.notional_pct if not args.no_lane_presets else args.notional_pct
                        notional = min(equity * lane_notional_pct, buying_power * args.buying_power_pct)
                except Exception as _sz_exc:
                    logging.warning("voltgt_sizer error (fall back to default): %s", _sz_exc)
                    lane_notional_pct = lane.notional_pct if not args.no_lane_presets else args.notional_pct
                    notional = min(equity * lane_notional_pct, buying_power * args.buying_power_pct)
                qty = int(notional // quote["mid"])
                if qty < 1:
                    log_decision(
                        conn,
                        lane=lane.nickname,
                        symbol=sym,
                        action="skip",
                        reason="qty_zero",
                        quote=quote,
                        signal_bps=score,
                        equity=equity,
                        buying_power=buying_power,
                        open_positions=len(positions),
                    )
                    continue

                action = "entry_long" if direction > 0 else "entry_short"
                side = "buy" if direction > 0 else "sell"
                payload = {"score_reason": reason, "notional": notional, "qty": qty}
                log_decision(
                    conn,
                    lane=lane.nickname,
                    symbol=sym,
                    action=action,
                    reason=reason,
                    quote=quote,
                    signal_bps=score,
                    equity=equity,
                    buying_power=buying_power,
                    open_positions=len(positions),
                    payload=payload,
                )
                cid, status, response = submit_market_order(
                    client,
                    lane=lane.nickname,
                    symbol=sym,
                    side=side,
                    qty=qty,
                    action=action,
                    place_orders=args.place_orders,
                )
                log_order(
                    conn,
                    lane=lane.nickname,
                    symbol=sym,
                    action=action,
                    side=side,
                    qty=qty,
                    notional=qty * quote["mid"],
                    status=status,
                    client_order_id=cid,
                    response=response,
                )
                if status != "error":
                    remember_entry(conn, lane.nickname, sym, side, cid)
                logging.info(
                    "%s[%s] ENTRY %s %s qty=%s notional=$%.0f score=%.2fbps spread=%.2fbps tp=%.1f sl=%.1f hold=%.0fs status=%s",
                    lane.nickname,
                    lane.mode,
                    side.upper(),
                    sym,
                    qty,
                    qty * quote["mid"],
                    score,
                    quote["spread_bps"],
                    lane.take_profit_bps if not args.no_lane_presets else args.take_profit_bps,
                    lane.stop_loss_bps if not args.no_lane_presets else args.stop_loss_bps,
                    lane.max_hold_sec if not args.no_lane_presets else args.max_hold_sec,
                    status,
                )

            conn.execute(
                "INSERT OR REPLACE INTO heartbeats (ts_utc, mode, lanes, message) VALUES (?, ?, ?, ?)",
                (
                    utc_now(),
                    "place_orders" if args.place_orders else "dry_run",
                    ",".join(l.nickname for l in lanes),
                    "loop_ok",
                ),
            )
            conn.commit()
        except Exception as exc:
            logging.exception("loop_error: %s", exc)
            if args.once:
                raise

        if args.once:
            break
        elapsed = time.time() - loop_start
        time.sleep(max(0.05, args.interval_sec - elapsed))


def install_signal_handlers() -> None:
    def _handle(_signum, _frame) -> None:
        StopFlag.stop = True

    signal.signal(signal.SIGINT, _handle)
    signal.signal(signal.SIGTERM, _handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default=str(DEFAULT_ENV))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--log", default=str(DEFAULT_LOG))
    parser.add_argument("--lanes", default="Valk,HFT1,HFT2,HFT3")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    parser.add_argument("--place-orders", action="store_true")
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--interval-sec", type=float, default=2.0)
    parser.add_argument("--closed-sleep-sec", type=float, default=30.0)
    parser.add_argument("--ignore-market-hours", action="store_true")
    parser.add_argument("--allow-entries-near-eod", action="store_true")
    parser.add_argument("--notional-pct", type=float, default=0.95)
    parser.add_argument("--buying-power-pct", type=float, default=0.95)
    parser.add_argument("--max-positions", type=int, default=1)
    parser.add_argument("--max-trades-per-day", type=int, default=2000)
    parser.add_argument("--take-profit-bps", type=float, default=5.0)
    parser.add_argument("--stop-loss-bps", type=float, default=4.0)
    parser.add_argument("--max-hold-sec", type=float, default=45.0)
    parser.add_argument("--min-signal-bps", type=float, default=1.0)
    parser.add_argument("--max-spread-bps", type=float, default=35.0)
    parser.add_argument("--history-ticks", type=int, default=80)
    parser.add_argument("--long-only", action="store_true")
    parser.add_argument("--no-lane-presets", action="store_true", help="Use global TP/SL/hold/signal args instead of lane-specific sim winners")
    args = parser.parse_args()

    load_env(Path(args.env))
    install_signal_handlers()
    run_loop(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
