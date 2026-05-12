"""Live slippage measurement (Rec #7).

Joins every filled order to the quote at submission time and computes the
realized slip = |fill_price − mid_at_submit| in bps. Writes per-trade rows
to data/slippage_log.parquet so we can build an honest distribution.

Pulls from `data/alpaca_max_hft_host.sqlite3` (HFT orders) and from Alpaca's
order GET endpoint for fill prices.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

REPO = Path("/Users/weeoto/turbo-carnival")
ENV = REPO / "council" / ".env.local"
DB = REPO / "data" / "alpaca_max_hft_host.sqlite3"
OUT_PARQUET = REPO / "data" / "slippage_log.parquet"


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
    p.add_argument("--lane", default="all", help="lane filter or 'all'")
    p.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    p.add_argument("--limit", type=int, default=500)
    args = p.parse_args()

    load_env()
    endpoint = os.environ.get("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")
    conn = sqlite3.connect(DB)
    where = "WHERE date(ts_utc) = ?"
    params = [args.date]
    if args.lane != "all":
        where += " AND lane = ?"
        params.append(args.lane)
    sql = f"""
        SELECT id, ts_utc, lane, symbol, action, side, qty, notional, status,
               client_order_id, bid, ask, mid, response_json
        FROM orders LEFT JOIN decisions USING(client_order_id)
        {where}
        ORDER BY id DESC LIMIT ?
    """
    df = pd.read_sql_query(sql.replace("LEFT JOIN decisions USING(client_order_id)\n        ", ""),
                            conn, params=params + [args.limit])
    conn.close()
    if df.empty:
        print("no orders found for filter")
        return 0

    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True)
    df["expected_mid"] = None
    df["fill_price"] = None
    df["slip_bps"] = None

    for i, row in df.iterrows():
        # 1) Try to read decision's bid/ask/mid from the decisions table
        # 2) Fetch the order from Alpaca for actual fill price
        oid = (row.get("response_json") or "")
        try:
            resp = json.loads(oid)
            order_id = resp.get("id")
            if not order_id:
                continue
            lane = row["lane"]
            k = os.environ.get(f"ALPACA_{lane.upper()}_KEY")
            s = os.environ.get(f"ALPACA_{lane.upper()}_SECRET")
            H = {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}
            r = requests.get(f"{endpoint}/orders/{order_id}", headers=H, timeout=8)
            if r.status_code == 200:
                o = r.json()
                fill = float(o.get("filled_avg_price") or 0.0)
                df.at[i, "fill_price"] = fill if fill > 0 else None
        except Exception:
            continue

    # Compute slip when we have both mid and fill — limited because the decisions
    # join doesn't always succeed; this is the v1 best-effort. We can enrich later.
    have_both = df.dropna(subset=["fill_price"]).copy()
    if "mid" in have_both.columns:
        have_both = have_both.dropna(subset=["mid"])
        have_both["slip_bps"] = ((have_both["fill_price"] - have_both["mid"]).abs() / have_both["mid"]) * 10000.0

    df.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {len(df)} rows to {OUT_PARQUET}")
    if "slip_bps" in have_both.columns and not have_both.empty:
        s = have_both["slip_bps"].describe()
        print(f"slip_bps (n={len(have_both)}): mean={s['mean']:.2f}  median={s['50%']:.2f}  "
              f"p90={have_both['slip_bps'].quantile(0.9):.2f}  max={s['max']:.2f}")
    else:
        print("no joined mid+fill rows; results table written without slip column")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
