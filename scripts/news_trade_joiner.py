"""Join trades to news events (Rec #8).

Reads today's HFT orders + condor lane orders, joins to news events from
data/news/{cnbc,marketwatch,yahoo,stocktwits}/<date>/, and writes a wide
attribution table to data/trade_attribution.parquet.

For each trade row we attach:
- news events within ±15 min of entry timestamp
- count of stocktwits messages mentioning the ticker in the prior 60 min
- sentiment tilt (bull - bear) at entry time
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
DB_HFT = REPO / "data" / "alpaca_max_hft_host.sqlite3"
DB_CONDOR = REPO / "data" / "alpaca_three_lane_host.sqlite3"
NEWS_DIR = REPO / "data" / "news"
OUT_PATH = REPO / "data" / "trade_attribution.parquet"


def load_trades(date_str: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_HFT)
    df = pd.read_sql_query(
        "SELECT ts_utc, lane, symbol, action, side, qty, notional, status FROM orders "
        f"WHERE date(ts_utc) = '{date_str}' ORDER BY id",
        conn,
    )
    conn.close()
    df["ts"] = pd.to_datetime(df["ts_utc"], utc=True)
    return df


def load_news(date_str: str) -> pd.DataFrame:
    rows = []
    # Headlines: cnbc, marketwatch
    for src in ("cnbc", "marketwatch"):
        p = NEWS_DIR / src / date_str / "headlines.jsonl"
        if not p.exists():
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ts_raw = obj.get("timestamp") or obj.get("published_at") or obj.get("time") or ""
                try:
                    ts = pd.Timestamp(ts_raw, tz="UTC") if ts_raw else None
                except Exception:
                    ts = None
                rows.append({
                    "source": src,
                    "ticker": None,
                    "ts": ts,
                    "headline": obj.get("headline") or obj.get("title") or "",
                    "url": obj.get("url", ""),
                })
            except Exception:
                continue
    # Per-ticker: yahoo
    for path in (NEWS_DIR / "yahoo" / date_str).glob("*.jsonl") if (NEWS_DIR / "yahoo" / date_str).exists() else []:
        ticker = path.stem
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                pub = obj.get("providerPublishTime") or obj.get("pubDate") or obj.get("published_at")
                if isinstance(pub, (int, float)):
                    ts = pd.Timestamp(pub, unit="s", tz="UTC")
                elif isinstance(pub, str):
                    try:
                        ts = pd.Timestamp(pub, tz="UTC")
                    except Exception:
                        ts = None
                else:
                    ts = None
                rows.append({
                    "source": "yahoo",
                    "ticker": ticker.upper(),
                    "ts": ts,
                    "headline": obj.get("title") or "",
                    "url": obj.get("link") or "",
                })
            except Exception:
                continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def load_stocktwits(date_str: str) -> pd.DataFrame:
    base = NEWS_DIR / "stocktwits" / date_str
    if not base.exists():
        return pd.DataFrame()
    rows = []
    for p in base.glob("*.jsonl"):
        ticker = p.stem
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                ts_raw = obj.get("created_at")
                ts = pd.Timestamp(ts_raw, tz="UTC") if ts_raw else None
                sent = ((obj.get("entities") or {}).get("sentiment") or {}).get("basic")
                rows.append({"ticker": ticker, "ts": ts, "sentiment": sent})
            except Exception:
                continue
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)


def attribute(trades: pd.DataFrame, news: pd.DataFrame, twits: pd.DataFrame) -> pd.DataFrame:
    if trades.empty:
        return trades
    delta = pd.Timedelta(minutes=15)
    rows = []
    for _, t in trades.iterrows():
        ts = t["ts"]
        sym = t["symbol"]
        relevant_news = news[
            (news["ts"] >= ts - delta)
            & (news["ts"] <= ts + delta)
            & ((news["ticker"].isna()) | (news["ticker"] == sym))
        ] if not news.empty else news
        # Sentiment in prior 60 min
        sym_twits = twits[(twits["ticker"] == sym) & (twits["ts"] >= ts - pd.Timedelta(hours=1)) & (twits["ts"] <= ts)] if not twits.empty else twits
        bull = int((sym_twits["sentiment"] == "Bullish").sum()) if not sym_twits.empty else 0
        bear = int((sym_twits["sentiment"] == "Bearish").sum()) if not sym_twits.empty else 0
        rows.append({
            **t.to_dict(),
            "n_news_within_15min": int(len(relevant_news)) if not relevant_news.empty else 0,
            "news_headlines": " | ".join(relevant_news["headline"].head(3).tolist()) if not relevant_news.empty else "",
            "stocktwits_bull_60m": bull,
            "stocktwits_bear_60m": bear,
            "sentiment_tilt": bull - bear,
        })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    args = p.parse_args()

    trades = load_trades(args.date)
    news = load_news(args.date)
    twits = load_stocktwits(args.date)
    print(f"trades={len(trades)}  news={len(news)}  stocktwits={len(twits)}")

    if trades.empty:
        print("no trades for date")
        return 0

    attributed = attribute(trades, news, twits)
    attributed.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}  ({len(attributed)} rows)")
    # Quick summary
    with_news = attributed[attributed["n_news_within_15min"] > 0]
    print(f"  trades with news in ±15min window: {len(with_news)} ({len(with_news)/max(len(attributed),1)*100:.1f}%)")
    if not with_news.empty:
        by_action = with_news.groupby("action").size()
        print("  breakdown by action:")
        for a, n in by_action.items():
            print(f"    {a}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
