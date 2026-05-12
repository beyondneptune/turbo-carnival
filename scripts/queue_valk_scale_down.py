"""Queue the Valk scale-down (44 → 22 contracts) for tomorrow's open.

Does NOT submit at execution time. Writes a queued-order JSON file that an
operator can review then submit at next market open via the companion
`execute_queued_order.py` runner (which checks market-is-open first).

Per 60-model swarm consensus + Opus xhigh #7 (largest unbounded tail risk on
the book overnight). Action class: defensive, defined-risk.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = Path("/Users/weeoto/turbo-carnival")
ENV = REPO / "council" / ".env.local"
QUEUE_DIR = REPO / "data" / "order_queue"
QUEUE_DIR.mkdir(parents=True, exist_ok=True)


def load_env() -> None:
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def main() -> int:
    load_env()
    endpoint = os.environ.get("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")
    key = os.environ["ALPACA_VALK_KEY"]
    sec = os.environ["ALPACA_VALK_SECRET"]
    H = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}

    # Read current Valk positions
    pos = requests.get(f"{endpoint}/positions", headers=H, timeout=10).json()
    if not isinstance(pos, list):
        print(f"ERROR: positions returned {pos}", file=sys.stderr)
        return 1

    legs = []
    for p in pos:
        sym = str(p.get("symbol") or "")
        if not sym.startswith("SPY"):
            continue
        qty = int(float(p.get("qty") or 0))
        side = str(p.get("side") or "").lower()
        avg = float(p.get("avg_entry_price") or 0)
        legs.append({"symbol": sym, "current_qty": qty, "side": side, "avg_entry": avg})

    if len(legs) != 4:
        print(f"WARN expected 4 legs, got {len(legs)}: {[l['symbol'] for l in legs]}", file=sys.stderr)
        # Continue anyway — write whatever we have

    # Build close-half legs (qty 22 of each, opposite side to flatten half)
    # For an iron condor: long puts/calls are positive qty (we sold to close);
    # short puts/calls are negative qty (we buy to close).
    close_legs = []
    for leg in legs:
        sym = leg["symbol"]
        cur_qty = leg["current_qty"]
        # Half the position; sign matches what we need to flatten.
        # If we hold long 44, close 22 by selling 22.
        # If we hold short 44 (qty -44), close 22 by buying 22.
        if cur_qty > 0:
            close_qty = min(cur_qty, 22)
            close_action = "sell"
        else:
            close_qty = min(abs(cur_qty), 22)
            close_action = "buy"
        close_legs.append({
            "symbol": sym,
            "side": close_action,
            "qty": close_qty,
            "note": f"close {close_qty} of current {abs(cur_qty)} ({'long' if cur_qty > 0 else 'short'})",
        })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = {
        "queued_at_utc": datetime.now(timezone.utc).isoformat(),
        "action": "valk_scale_down_half",
        "lane": "Valk",
        "rationale": (
            "60-model swarm consensus + Opus xhigh #7: halve qty 44 → 22 contracts at "
            "tomorrow's open. Reduces short-gamma concentration. Largest unbounded "
            "overnight tail per panel."
        ),
        "submit_window_et": "09:30-10:00",  # earliest part of session for tighter spreads
        "execute_only_when_market_open": True,
        "endpoint": endpoint,
        "lane_key_env": "ALPACA_VALK_KEY",
        "legs_observed_at_queue": legs,
        "close_legs_to_submit": close_legs,
        "order_type": "market",   # tight markets, but options can have wide spreads; consider limit at mid + 0.05
        "time_in_force": "day",
        "fallback_limit_offset_usd": 0.05,
    }

    out_path = QUEUE_DIR / f"valk_scale_down_{stamp}.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"queued: {out_path}")
    print(f"observed positions:")
    for leg in legs:
        print(f"  {leg['symbol']:25s} qty {leg['current_qty']:+d}  side {leg['side']}  avg ${leg['avg_entry']:.2f}")
    print(f"\nclose legs (to submit at open):")
    for cl in close_legs:
        print(f"  {cl['symbol']:25s} {cl['side']:4s} {cl['qty']}   # {cl['note']}")
    print(f"\nTo execute tomorrow:")
    print(f"  .venv/bin/python scripts/execute_queued_order.py {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
