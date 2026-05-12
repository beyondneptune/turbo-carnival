"""Alpaca three-lane paper host.

Runs three separate Alpaca paper accounts:
  - Aegis:  fixed_best_qwen_static_5_50
  - Falcon: fixed_best_qwen_baseline_10_100
  - Valk:   champion_nn_top3_nn_20_200_t55

This is intentionally separate from paper_v1/IBKR. It uses Alpaca REST directly
so no SDK install is needed.

By default it plans/logs only. Pass --place-orders to submit paper orders.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.greeks import bs_delta  # noqa: E402


NY = ZoneInfo("America/New_York")
LONDON = ZoneInfo("Europe/London")
DEFAULT_ENV = REPO_ROOT / "council" / ".env.local"
DEFAULT_DB = REPO_ROOT / "data" / "alpaca_three_lane_host.sqlite3"
DEFAULT_LOG = REPO_ROOT / "data" / "logs" / "alpaca_three_lane_host.log"
DATA_BASE = "https://data.alpaca.markets"


@dataclass(frozen=True)
class Lane:
    nickname: str
    account_env: str
    key_env: str
    secret_env: str
    strategy: str
    risk_pct: float
    concurrent_pct: float
    target_delta: float
    target_dte: int
    wing_width: float
    profit_take: float
    stop_loss_credit_mult: float
    close_dte: int
    min_credit: float
    max_contracts: int = 1


LANES = [
    Lane(
        nickname="Aegis",
        account_env="ALPACA_AEGIS_ACCOUNT",
        key_env="ALPACA_AEGIS_KEY",
        secret_env="ALPACA_AEGIS_SECRET",
        strategy="fixed_best_qwen_static_5_50",
        risk_pct=5.0,
        concurrent_pct=50.0,
        target_delta=0.12,
        target_dte=30,
        wing_width=5.0,
        profit_take=0.30,
        stop_loss_credit_mult=2.0,
        close_dte=10,
        min_credit=0.20,
        max_contracts=999,
    ),
    Lane(
        nickname="Falcon",
        account_env="ALPACA_FALCON_ACCOUNT",
        key_env="ALPACA_FALCON_KEY",
        secret_env="ALPACA_FALCON_SECRET",
        strategy="fixed_best_qwen_baseline_10_100",
        risk_pct=10.0,
        concurrent_pct=100.0,
        target_delta=0.12,
        target_dte=30,
        wing_width=5.0,
        profit_take=0.30,
        stop_loss_credit_mult=2.0,
        close_dte=10,
        min_credit=0.20,
        max_contracts=999,
    ),
    Lane(
        nickname="Valk",
        account_env="ALPACA_VALK_ACCOUNT",
        key_env="ALPACA_VALK_KEY",
        secret_env="ALPACA_VALK_SECRET",
        strategy="champion_nn_top3_nn_20_200_t55",
        risk_pct=20.0,
        concurrent_pct=200.0,
        target_delta=0.12,
        target_dte=30,
        wing_width=5.0,
        profit_take=0.30,
        stop_loss_credit_mult=2.0,
        close_dte=10,
        min_credit=0.20,
        max_contracts=999,
    ),
]


def load_env(path: Path = DEFAULT_ENV) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
    )


def init_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS account_checks (
            ts_utc TEXT NOT NULL,
            lane TEXT NOT NULL,
            account_id TEXT,
            account_number TEXT,
            status TEXT,
            equity REAL,
            buying_power REAL,
            options_approved_level INTEGER,
            options_trading_level INTEGER,
            raw_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS lane_orders (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_date TEXT NOT NULL,
            lane TEXT NOT NULL,
            strategy TEXT NOT NULL,
            account_id TEXT,
            status TEXT NOT NULL,
            spy_price REAL,
            expiry TEXT,
            long_put TEXT,
            short_put TEXT,
            short_call TEXT,
            long_call TEXT,
            quantity INTEGER,
            credit REAL,
            limit_price REAL,
            payload_json TEXT,
            response_json TEXT,
            created_at_utc TEXT NOT NULL,
            UNIQUE(trade_date, lane)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS heartbeats (
            ts_utc TEXT PRIMARY KEY,
            mode TEXT NOT NULL,
            message TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


class AlpacaClient:
    def __init__(self, endpoint: str, key: str, secret: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.key = key
        self.secret = secret
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": key,
                "APCA-API-SECRET-KEY": secret,
                "Content-Type": "application/json",
            }
        )

    def get(self, url_or_path: str, **params: Any) -> dict[str, Any]:
        url = url_or_path if url_or_path.startswith("http") else self.endpoint + url_or_path
        r = self.session.get(url, params={k: v for k, v in params.items() if v is not None}, timeout=15)
        if not r.ok:
            raise RuntimeError(f"GET {url} failed {r.status_code}: {r.text[:500]}")
        return r.json()

    def post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        r = self.session.post(self.endpoint + path, json=payload, timeout=20)
        if not r.ok:
            raise RuntimeError(f"POST {path} failed {r.status_code}: {r.text[:1000]}")
        return r.json()

    def account(self) -> dict[str, Any]:
        return self.get("/account")

    def clock(self) -> dict[str, Any]:
        return self.get("/clock")

    def latest_spy_quote(self) -> dict[str, Any]:
        return self.get(f"{DATA_BASE}/v2/stocks/SPY/quotes/latest")

    def latest_option_quotes(self, symbols: list[str]) -> dict[str, Any]:
        payload = self.get(f"{DATA_BASE}/v1beta1/options/quotes/latest", symbols=",".join(symbols))
        return payload.get("quotes", {})

    def option_contracts(self, expiry: str, page_token: str | None = None) -> dict[str, Any]:
        return self.get(
            "/options/contracts",
            underlying_symbols="SPY",
            expiration_date=expiry,
            limit=1000,
            page_token=page_token,
        )

    def submit_order(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.post("/orders", payload)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def wait_until_open(client: AlpacaClient, *, buffer_seconds: int, once: bool) -> None:
    while True:
        clock = client.clock()
        ts = parse_dt(clock["timestamp"])
        next_open = parse_dt(clock["next_open"])
        if bool(clock["is_open"]):
            logging.info("Alpaca market is open at %s", ts.astimezone(LONDON).isoformat())
            return
        wait = max(0.0, (next_open - ts).total_seconds() + buffer_seconds)
        logging.info(
            "market closed; next open %s London; sleeping %.0fs",
            next_open.astimezone(LONDON).isoformat(),
            wait,
        )
        if once:
            return
        time.sleep(min(wait, 300.0))


def expiry_for_target(now: datetime, target_dte: int) -> str:
    out = (now.astimezone(NY).date() + timedelta(days=target_dte))
    while out.weekday() != 4:
        out += timedelta(days=1)
    return out.isoformat()


def fetch_chain(client: AlpacaClient, expiry: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    page: str | None = None
    while True:
        payload = client.option_contracts(expiry, page)
        rows.extend(payload.get("option_contracts", []))
        page = payload.get("next_page_token")
        if not page:
            break
    if not rows:
        raise RuntimeError(f"no SPY contracts returned for expiry {expiry}")
    df = pd.DataFrame(rows)
    df["strike_price"] = pd.to_numeric(df["strike_price"], errors="coerce")
    df = df[df["tradable"].eq(True)].dropna(subset=["strike_price"])
    return df


def target_strikes(spot: float, sigma: float, dte: int, target_delta: float, wing: float, valid: list[float]) -> tuple[float, float, float, float]:
    t = max(dte / 365.0, 1.0 / 365.0)
    rfr = 0.045
    div = 0.014
    valid_sorted = sorted(valid)

    def raw_strike(option_type: str) -> float:
        adjusted = max(1e-6, min(0.999999, target_delta * math.exp(div * t)))
        d1 = float(norm.ppf(adjusted)) if option_type == "call" else -float(norm.ppf(adjusted))
        exponent = d1 * sigma * math.sqrt(t) - (rfr - div + 0.5 * sigma * sigma) * t
        strike = spot / math.exp(exponent)
        if option_type == "call":
            candidates = [k for k in valid_sorted if k >= max(spot, strike - 10)]
        else:
            candidates = [k for k in valid_sorted if k <= min(spot, strike + 10)]
        if not candidates:
            candidates = valid_sorted
        return min(candidates, key=lambda k: abs(abs(bs_delta(option_type, spot, k, t, rfr, sigma, div)) - target_delta))

    short_call = raw_strike("call")
    short_put = raw_strike("put")

    def nearest(x: float) -> float:
        return min(valid_sorted, key=lambda k: abs(k - x))

    long_call = nearest(short_call + wing)
    long_put = nearest(short_put - wing)
    if long_call <= short_call:
        long_call = nearest(short_call + 5.0)
    if long_put >= short_put:
        long_put = nearest(short_put - 5.0)
    return long_put, short_put, short_call, long_call


def symbol_for(chain: pd.DataFrame, option_type: str, strike: float) -> str:
    w = chain[(chain["type"].eq(option_type)) & (chain["strike_price"].round(6).eq(round(strike, 6)))]
    if w.empty:
        raise RuntimeError(f"missing {option_type} strike {strike}")
    return str(w.iloc[0]["symbol"])


def quote_mid(q: dict[str, Any]) -> tuple[float, float, float]:
    bid = float(q.get("bp") or 0.0)
    ask = float(q.get("ap") or 0.0)
    if bid <= 0 or ask <= 0 or ask < bid:
        return bid, ask, 0.0
    return bid, ask, (bid + ask) / 2.0


def build_order_plan(client: AlpacaClient, lane: Lane, account: dict[str, Any]) -> dict[str, Any]:
    spy_quote = client.latest_spy_quote()["quote"]
    bid = float(spy_quote["bp"])
    ask = float(spy_quote["ap"])
    spot = (bid + ask) / 2.0
    now = datetime.now(timezone.utc)
    expiry = expiry_for_target(now, lane.target_dte)
    dte = max(1, (datetime.fromisoformat(expiry).date() - now.date()).days)
    sigma = float(os.environ.get("ALPACA_SPY_IV_GUESS", "0.18"))
    chain = fetch_chain(client, expiry)
    valid = sorted(set(chain["strike_price"].astype(float)))
    long_put, short_put, short_call, long_call = target_strikes(spot, sigma, dte, lane.target_delta, lane.wing_width, valid)
    symbols = {
        "long_put": symbol_for(chain, "put", long_put),
        "short_put": symbol_for(chain, "put", short_put),
        "short_call": symbol_for(chain, "call", short_call),
        "long_call": symbol_for(chain, "call", long_call),
    }
    quotes = client.latest_option_quotes(list(symbols.values()))
    q_lp = quote_mid(quotes[symbols["long_put"]])
    q_sp = quote_mid(quotes[symbols["short_put"]])
    q_sc = quote_mid(quotes[symbols["short_call"]])
    q_lc = quote_mid(quotes[symbols["long_call"]])
    credit = q_sp[0] + q_sc[0] - q_lp[1] - q_lc[1]
    mid_credit = q_sp[2] + q_sc[2] - q_lp[2] - q_lc[2]
    wing_width = max(long_call - short_call, short_put - long_put)
    max_loss_per_contract = max(0.01, (wing_width - max(credit, 0.0)) * 100.0)
    equity = float(account.get("equity") or 100000.0)
    risk_budget = equity * lane.risk_pct / 100.0
    qty = max(1, min(lane.max_contracts, int(risk_budget // max_loss_per_contract)))
    limit_credit = max(0.01, math.floor(max(credit, mid_credit * 0.8) * 100) / 100.0)
    payload = {
        "order_class": "mleg",
        "qty": str(qty),
        "type": "limit",
        "time_in_force": "day",
        # Alpaca uses negative limit for net credit multi-leg orders.
        "limit_price": f"{-limit_credit:.2f}",
        "client_order_id": f"{lane.nickname.lower()}-{now.astimezone(NY).strftime('%Y%m%d')}-{int(time.time())}",
        "legs": [
            {"symbol": symbols["long_put"], "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
            {"symbol": symbols["short_put"], "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"},
            {"symbol": symbols["short_call"], "ratio_qty": "1", "side": "sell", "position_intent": "sell_to_open"},
            {"symbol": symbols["long_call"], "ratio_qty": "1", "side": "buy", "position_intent": "buy_to_open"},
        ],
    }
    return {
        "spot": spot,
        "expiry": expiry,
        "strikes": {
            "long_put": long_put,
            "short_put": short_put,
            "short_call": short_call,
            "long_call": long_call,
        },
        "symbols": symbols,
        "quotes": {
            "long_put": q_lp,
            "short_put": q_sp,
            "short_call": q_sc,
            "long_call": q_lc,
        },
        "credit": credit,
        "mid_credit": mid_credit,
        "limit_credit": limit_credit,
        "max_loss_per_contract": max_loss_per_contract,
        "quantity": qty,
        "payload": payload,
    }


def already_ran(conn: sqlite3.Connection, lane: Lane, trade_date: str) -> bool:
    row = conn.execute(
        "SELECT id, status FROM lane_orders WHERE trade_date = ? AND lane = ?",
        (trade_date, lane.nickname),
    ).fetchone()
    return row is not None


def log_account(conn: sqlite3.Connection, lane: Lane, account: dict[str, Any]) -> None:
    conn.execute(
        """
        INSERT INTO account_checks (
            ts_utc, lane, account_id, account_number, status, equity, buying_power,
            options_approved_level, options_trading_level, raw_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            utc_now(),
            lane.nickname,
            os.environ.get(lane.account_env),
            account.get("account_number"),
            account.get("status"),
            float(account.get("equity") or 0),
            float(account.get("buying_power") or 0),
            int(account.get("options_approved_level") or 0),
            int(account.get("options_trading_level") or 0),
            json.dumps(account, default=str),
        ),
    )
    conn.commit()


def log_order(conn: sqlite3.Connection, lane: Lane, trade_date: str, plan: dict[str, Any], status: str, response: dict[str, Any] | None) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO lane_orders (
            trade_date, lane, strategy, account_id, status, spy_price, expiry,
            long_put, short_put, short_call, long_call, quantity, credit,
            limit_price, payload_json, response_json, created_at_utc
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trade_date,
            lane.nickname,
            lane.strategy,
            os.environ.get(lane.account_env),
            status,
            float(plan["spot"]),
            plan["expiry"],
            plan["symbols"]["long_put"],
            plan["symbols"]["short_put"],
            plan["symbols"]["short_call"],
            plan["symbols"]["long_call"],
            int(plan["quantity"]),
            float(plan["credit"]),
            float(plan["payload"]["limit_price"]),
            json.dumps(plan, default=str),
            json.dumps(response or {}, default=str),
            utc_now(),
        ),
    )
    conn.commit()


def run_lanes(conn: sqlite3.Connection, endpoint: str, *, place_orders: bool, force: bool) -> None:
    trade_date = datetime.now(NY).date().isoformat()
    for lane in LANES:
        key = os.environ[lane.key_env]
        secret = os.environ[lane.secret_env]
        client = AlpacaClient(endpoint, key, secret)
        account = client.account()
        log_account(conn, lane, account)
        logging.info(
            "%s account=%s status=%s equity=%s options_level=%s strategy=%s",
            lane.nickname,
            os.environ.get(lane.account_env),
            account.get("status"),
            account.get("equity"),
            account.get("options_trading_level"),
            lane.strategy,
        )
        if already_ran(conn, lane, trade_date) and not force:
            logging.info("%s already has lane_orders row for %s; skipping", lane.nickname, trade_date)
            continue
        plan = build_order_plan(client, lane, account)
        logging.info(
            "%s plan SPY %.2f exp=%s LP=%s SP=%s SC=%s LC=%s qty=%s credit=%.2f mid=%.2f limit=%s",
            lane.nickname,
            plan["spot"],
            plan["expiry"],
            plan["symbols"]["long_put"],
            plan["symbols"]["short_put"],
            plan["symbols"]["short_call"],
            plan["symbols"]["long_call"],
            plan["quantity"],
            plan["credit"],
            plan["mid_credit"],
            plan["payload"]["limit_price"],
        )
        if plan["credit"] < lane.min_credit:
            log_order(conn, lane, trade_date, plan, "skipped_low_credit", None)
            logging.warning("%s skipped: credit %.2f < min %.2f", lane.nickname, plan["credit"], lane.min_credit)
            continue
        if place_orders:
            try:
                response = client.submit_order(plan["payload"])
                log_order(conn, lane, trade_date, plan, "submitted", response)
                logging.info("%s submitted order id=%s status=%s", lane.nickname, response.get("id"), response.get("status"))
            except Exception as exc:
                log_order(conn, lane, trade_date, plan, "submit_error", {"error": str(exc)})
                logging.exception("%s submit failed", lane.nickname)
        else:
            log_order(conn, lane, trade_date, plan, "planned_only", None)
            logging.info("%s planned only; use --place-orders to submit", lane.nickname)


def heartbeat(conn: sqlite3.Connection, mode: str, message: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO heartbeats (ts_utc, mode, message) VALUES (?, ?, ?)",
        (utc_now(), mode, message),
    )
    conn.commit()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=Path, default=DEFAULT_ENV)
    parser.add_argument("--db", type=Path, default=None)
    parser.add_argument("--log", type=Path, default=None)
    parser.add_argument("--once", action="store_true", help="Run one pass now; do not wait for open.")
    parser.add_argument("--wait-open", action="store_true", help="Wait until Alpaca regular market open before running.")
    parser.add_argument("--place-orders", action="store_true", help="Submit paper orders. Default is plan/log only.")
    parser.add_argument("--force", action="store_true", help="Ignore existing same-day lane_orders rows.")
    parser.add_argument("--open-buffer-sec", type=int, default=120)
    args = parser.parse_args()

    load_env(args.env)
    endpoint = os.environ.get("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")
    db_path = args.db or Path(os.environ.get("ALPACA_HOST_DB", str(DEFAULT_DB)))
    log_path = args.log or Path(os.environ.get("ALPACA_HOST_LOG", str(DEFAULT_LOG)))
    setup_logging(log_path)
    conn = init_db(db_path)
    place_orders = bool(args.place_orders or os.environ.get("ALPACA_PLACE_ORDERS") == "1")

    logging.info("alpaca three-lane host starting place_orders=%s db=%s", place_orders, db_path)
    heartbeat(conn, "starting", f"place_orders={place_orders}")

    clock_client = AlpacaClient(endpoint, os.environ["ALPACA_AEGIS_KEY"], os.environ["ALPACA_AEGIS_SECRET"])
    if args.wait_open:
        wait_until_open(clock_client, buffer_seconds=args.open_buffer_sec, once=False)
    elif not args.once:
        wait_until_open(clock_client, buffer_seconds=args.open_buffer_sec, once=False)

    run_lanes(conn, endpoint, place_orders=place_orders, force=args.force)
    heartbeat(conn, "done", "lane pass complete")
    logging.info("alpaca three-lane host done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
