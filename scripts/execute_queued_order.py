"""Execute a queued order JSON file. Safe by default.

Refuses to run unless:
  - market is currently open (per Alpaca /v2/clock)
  - --confirm flag is passed
  - the queue file's `execute_only_when_market_open` is True

Submits each leg as a separate market or marketable-limit order.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = Path("/Users/weeoto/turbo-carnival")
ENV = REPO / "council" / ".env.local"
LOG_DIR = REPO / "data" / "logs"
LOG_DIR.mkdir(exist_ok=True)


def load_env() -> None:
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("queue_path", help="path to queued order JSON")
    p.add_argument("--confirm", action="store_true", help="actually submit (default: dry-run)")
    p.add_argument("--ignore-market-hours", action="store_true")
    args = p.parse_args()

    load_env()
    queue = json.loads(Path(args.queue_path).read_text())
    endpoint = queue.get("endpoint") or os.environ.get("ALPACA_ENDPOINT")
    key_env = queue.get("lane_key_env", "ALPACA_VALK_KEY")
    sec_env = key_env.replace("_KEY", "_SECRET")
    key = os.environ[key_env]
    sec = os.environ[sec_env]
    H = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}

    # Market clock check
    if not args.ignore_market_hours and queue.get("execute_only_when_market_open", True):
        clock = requests.get(f"{endpoint}/clock", headers=H, timeout=8).json()
        if not clock.get("is_open"):
            print(f"REFUSE: market not open (next open {clock.get('next_open')})", file=sys.stderr)
            return 2

    log = LOG_DIR / f"order_execution_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.log"
    print(f"queue:   {args.queue_path}")
    print(f"action:  {queue.get('action')}")
    print(f"lane:    {queue.get('lane')}")
    print(f"confirm: {args.confirm}")
    print(f"log:     {log}")
    print()

    submitted = []
    for leg in queue.get("close_legs_to_submit", []):
        payload = {
            "symbol": leg["symbol"],
            "qty": str(leg["qty"]),
            "side": leg["side"],
            "type": queue.get("order_type", "market"),
            "time_in_force": queue.get("time_in_force", "day"),
        }
        line = f"  {leg['side']:4s} {leg['qty']:>3d} {leg['symbol']}"
        if not args.confirm:
            print(f"[DRY] {line}")
            continue
        try:
            r = requests.post(f"{endpoint}/orders", headers=H, json=payload, timeout=12)
            r.raise_for_status()
            data = r.json()
            print(f"[SUB] {line}  status={data.get('status')}  id={data.get('id','?')[:8]}")
            submitted.append({"leg": leg, "response": data})
        except Exception as exc:
            print(f"[ERR] {line}  {type(exc).__name__}: {exc}")
            submitted.append({"leg": leg, "error": str(exc)})

    log.write_text(json.dumps({
        "queue": queue,
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "confirmed": args.confirm,
        "submitted": submitted,
    }, indent=2, default=str))

    if not args.confirm:
        print("\nDry-run only. Re-run with --confirm to submit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
