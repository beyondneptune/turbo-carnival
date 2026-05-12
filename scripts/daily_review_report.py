"""Daily post-EOD review report (Rec #14).

Pulls today's account state, trade log, news context, and circuit-breaker
events. Writes a markdown summary to reports/daily_review_<date>.md.

Designed to be cron'd at 21:30 BST / 16:30 ET each weekday.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import requests

REPO = Path("/Users/weeoto/turbo-carnival")
ENV = REPO / "council" / ".env.local"
REPORTS = REPO / "reports"
REPORTS.mkdir(exist_ok=True)


def load_env() -> None:
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def fetch_lane(lane: str, endpoint: str) -> dict:
    k = os.environ.get(f"ALPACA_{lane.upper()}_KEY")
    s = os.environ.get(f"ALPACA_{lane.upper()}_SECRET")
    H = {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s}
    try:
        acct = requests.get(f"{endpoint}/account", headers=H, timeout=8).json()
        pos = requests.get(f"{endpoint}/positions", headers=H, timeout=8).json() or []
        eq = float(acct.get("equity") or 0)
        last = float(acct.get("last_equity") or 0)
        return {"lane": lane, "equity": eq, "pnl": eq - last, "positions": pos}
    except Exception as exc:
        return {"lane": lane, "error": str(exc)}


def hft_trades(date_str: str) -> tuple[int, dict[str, int], int, int, int]:
    db = REPO / "data" / "alpaca_max_hft_host.sqlite3"
    if not db.exists():
        return 0, {}, 0, 0, 0
    conn = sqlite3.connect(db)
    df = conn.execute(
        f"SELECT lane, action, COUNT(*) FROM orders WHERE date(ts_utc)='{date_str}' GROUP BY lane, action"
    ).fetchall()
    conn.close()
    total = 0
    by_lane: dict[str, int] = {}
    tp = sl = te = 0
    for lane, action, c in df:
        total += c
        by_lane[lane] = by_lane.get(lane, 0) + c
        if action == "take_profit": tp += c
        elif action == "stop_loss": sl += c
        elif action == "time_exit": te += c
    return total, by_lane, tp, sl, te


def news_count(date_str: str) -> dict[str, int]:
    out = {}
    for src in ("cnbc", "marketwatch", "yahoo", "stocktwits", "twitter"):
        base = REPO / "data" / "news" / src / date_str
        if not base.exists():
            continue
        n = 0
        for p in base.glob("*.jsonl"):
            n += sum(1 for line in p.read_text().splitlines() if line.strip())
        out[src] = n
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    args = p.parse_args()

    load_env()
    endpoint = os.environ.get("ALPACA_ENDPOINT", "https://paper-api.alpaca.markets/v2")
    lanes = [fetch_lane(l, endpoint) for l in ("Aegis", "Falcon", "Valk")]
    total, by_lane, tp, sl, te = hft_trades(args.date)
    news = news_count(args.date)

    total_pnl = sum(l.get("pnl", 0) for l in lanes)
    total_eq = sum(l.get("equity", 0) for l in lanes)

    lines = [
        f"# Daily Review — {args.date}",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Headline",
        "",
        f"- Combined P&L today: **${total_pnl:+,.2f}**",
        f"- Combined NAV: ${total_eq:,.2f}",
        f"- Total HFT trades: {total}",
        f"- Exit mix: {tp} take-profit / {sl} stop-loss / {te} time-exit",
        f"- TP hit rate: {tp/(tp+sl+te)*100 if (tp+sl+te) else 0:.1f}%",
        "",
        "## Per-lane",
        "",
        "| Lane | Equity | P&L | # Positions |",
        "|---|---:|---:|---:|",
    ]
    for l in lanes:
        if "error" in l:
            lines.append(f"| {l['lane']} | err | err | {l.get('error', '')} |")
            continue
        lines.append(f"| {l['lane']} | ${l['equity']:,.2f} | ${l['pnl']:+,.2f} | {len(l['positions'])} |")

    lines += ["", "## Trades by lane", ""]
    for lane, n in sorted(by_lane.items()):
        lines.append(f"- {lane}: {n}")

    lines += ["", "## News collected today", ""]
    for src, n in sorted(news.items()):
        lines.append(f"- {src}: {n} items")

    # Circuit breaker state
    cb_dir = REPO / "data" / "circuit_breaker"
    if cb_dir.exists():
        flags = list(cb_dir.glob("*_paused.json"))
        if flags:
            lines += ["", "## ⚠ Circuit breaker — paused lanes", ""]
            for f in flags:
                obj = json.loads(f.read_text())
                lines.append(f"- {obj.get('lane')} paused at pnl_pct {obj.get('pnl_pct', 0):+.2f}%")

    out = REPORTS / f"daily_review_{args.date}.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
