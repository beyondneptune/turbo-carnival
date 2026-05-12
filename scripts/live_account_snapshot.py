"""Print a compact live snapshot for the six Alpaca paper lanes."""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

REPO = Path("/Users/weeoto/turbo-carnival")
ENV = REPO / "council" / ".env.local"
ENDPOINT = "https://paper-api.alpaca.markets/v2"
NY = ZoneInfo("America/New_York")

ACCOUNTS = [
    ("Aegis", "AEGIS"),
    ("Falcon", "FALCON"),
    ("Valk", "VALK"),
    ("Hermes", "HERMES"),
    ("Atlas", "ATLAS"),
    ("TERMINATOR", "TERMINATOR"),
]


def load_env() -> None:
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def headers(prefix: str) -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": os.environ[f"ALPACA_{prefix}_KEY"],
        "APCA-API-SECRET-KEY": os.environ[f"ALPACA_{prefix}_SECRET"],
    }


def print_accounts() -> None:
    total = 0.0
    for name, prefix in ACCOUNTS:
        h = headers(prefix)
        acct = requests.get(f"{ENDPOINT}/account", headers=h, timeout=8).json()
        pos = requests.get(f"{ENDPOINT}/positions", headers=h, timeout=8).json() or []
        eq = float(acct.get("equity") or 0.0)
        last = float(acct.get("last_equity") or 0.0)
        total += eq
        print(
            f"{name:<11} eq=${eq:>10,.2f} day=${eq-last:+9,.2f} "
            f"pos={len(pos):<2} daytrades={acct.get('daytrade_count')}"
        )
        for p in pos:
            print(
                f"  {p.get('symbol'):<24} qty={p.get('qty'):>7} "
                f"side={p.get('side'):<5} upl={p.get('unrealized_pl')}"
            )
    print(f"{'TOTAL':<11} eq=${total:>10,.2f}")


def print_hft_counts() -> None:
    db = REPO / "data" / "alpaca_max_hft_host.sqlite3"
    if not db.exists():
        return
    conn = sqlite3.connect(db)
    today = datetime.now(timezone.utc).date().isoformat()
    print("HFT orders today:")
    for lane, action, status, count in conn.execute(
        """
        SELECT lane, action, status, COUNT(*)
        FROM orders
        WHERE date(ts_utc) = ?
        GROUP BY lane, action, status
        ORDER BY lane, action, status
        """,
        (today,),
    ):
        print(f"  {lane:<11} {action:<12} {status:<12} {count}")
    conn.close()


def main() -> int:
    load_env()
    print()
    print(f"=== {datetime.now(NY).strftime('%Y-%m-%d %H:%M:%S %Z')} ===")
    print_accounts()
    print_hft_counts()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
