"""Daily-loss circuit breaker (Rec #6 — Kimi K2 explicit suggestion).

Runs as a side process. Polls each Alpaca paper account every 30s. If any
lane's intraday P&L breaches the threshold (default −0.5% of starting NAV),
it sets a flag file that the HFT host reads and immediately stops opening
new positions for that lane. Open positions are NOT force-closed (that's a
separate operator decision).

Designed to never increase exposure. Pure defensive.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = Path("/Users/weeoto/turbo-carnival")
ENV = REPO / "council" / ".env.local"
FLAG_DIR = REPO / "data" / "circuit_breaker"
LOG_DIR = REPO / "data" / "logs"
FLAG_DIR.mkdir(parents=True, exist_ok=True)

STOPPED = False


def load_env() -> None:
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def check_lane(lane: str, endpoint: str, key: str, sec: str,
                threshold_pct: float, starting_nav: float) -> dict:
    H = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    try:
        acct = requests.get(f"{endpoint}/account", headers=H, timeout=8).json()
        eq = float(acct.get("equity") or 0.0)
        last_eq = float(acct.get("last_equity") or 0.0)
        pnl = eq - last_eq
        pnl_pct = (pnl / starting_nav) * 100.0 if starting_nav else 0.0
        breached = pnl_pct <= -abs(threshold_pct)
        return {
            "lane": lane, "ok": True, "equity": eq, "last_equity": last_eq,
            "pnl_usd": pnl, "pnl_pct": pnl_pct, "breached": breached,
        }
    except Exception as exc:
        return {"lane": lane, "ok": False, "error": str(exc)}


def set_flag(lane: str, payload: dict) -> Path:
    p = FLAG_DIR / f"{lane.lower()}_paused.json"
    p.write_text(json.dumps({**payload, "set_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2))
    return p


def clear_flag(lane: str) -> None:
    p = FLAG_DIR / f"{lane.lower()}_paused.json"
    if p.exists():
        p.unlink()


def _handle_sig(_s, _f) -> None:
    global STOPPED
    STOPPED = True


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold-pct", type=float, default=0.5,
                   help="Pause a lane when intraday loss >= this %% of starting NAV (default 0.5)")
    p.add_argument("--starting-nav", type=float, default=100_000.0)
    p.add_argument("--interval-sec", type=float, default=30.0)
    p.add_argument("--lanes", default="Aegis,Falcon,Valk")
    p.add_argument("--once", action="store_true")
    args = p.parse_args()

    load_env()
    endpoint = os.environ.get("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")
    lanes = [x.strip() for x in args.lanes.split(",") if x.strip()]
    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    log = LOG_DIR / "circuit_breaker.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"circuit breaker started — threshold {args.threshold_pct}% of "
          f"${args.starting_nav:,.0f} starting NAV, polling {args.interval_sec}s")

    while not STOPPED:
        snap = []
        for lane in lanes:
            k = os.environ.get(f"ALPACA_{lane.upper()}_KEY")
            s = os.environ.get(f"ALPACA_{lane.upper()}_SECRET")
            if not k or not s:
                continue
            r = check_lane(lane, endpoint, k, s, args.threshold_pct, args.starting_nav)
            snap.append(r)
            if r.get("ok") and r["breached"]:
                set_flag(lane, r)
                msg = f"CIRCUIT_BREAKER lane={lane} pnl_pct={r['pnl_pct']:+.2f}% PAUSED"
            elif r.get("ok") and not r["breached"]:
                # Auto-clear if previously set and now back under threshold (intraday recovery rare on stocks
                # but harmless here)
                clear_flag(lane)
                msg = f"  ok      lane={lane} pnl_pct={r['pnl_pct']:+.2f}%"
            else:
                msg = f"  ERR     lane={lane} err={r.get('error','?')}"
            print(msg, flush=True)
            with log.open("a") as f:
                f.write(f"{datetime.now(timezone.utc).isoformat()} {msg}\n")

        if args.once:
            break
        time.sleep(args.interval_sec)
    print("circuit breaker stopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
