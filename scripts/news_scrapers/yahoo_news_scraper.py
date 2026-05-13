"""Yahoo Finance ticker news scraper for paper-trading research.

Wraps `yfinance.Ticker(...).news` to fetch the latest headlines per ticker
and appends them to a per-day JSONL log. Dedupes by uuid (or by id/link if
uuid is missing — Yahoo's payload shape has shifted across yfinance
versions, so we are defensive). Research-only; not `paper_v1` evidence.

Output layout:
    data/news/yahoo/{YYYY-MM-DD}/{ticker}.jsonl

CLI mirrors stocktwits_scraper.py:
    --once
    --loop --interval-sec N      (default 300)
    --tickers SPY,NVDA
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "news" / "yahoo"
LOG_DIR = REPO_ROOT / "data" / "logs"
LOG_FILE = LOG_DIR / "yahoo_news_scraper.log"

DEFAULT_TICKERS = ("SPY", "NVDA", "TSLA", "AAPL", "META", "AMD", "QQQ")
PER_TICKER_SLEEP_SEC = 1.0


def _setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("yahoo_news_scraper")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _output_path(ticker: str) -> Path:
    day = _today_str()
    out_dir = DATA_DIR / day
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{ticker.upper()}.jsonl"


def _load_existing_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    if not path.exists():
        return keys
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                k = rec.get("uuid") or rec.get("link") or rec.get("title")
                if isinstance(k, str):
                    keys.add(k)
    except OSError:
        pass
    return keys


def _flatten_item(item: dict[str, Any]) -> dict[str, Any]:
    """yfinance >= 0.2.5x wraps fields under {"content": {...}} in some
    versions. Normalize so downstream code sees a flat dict regardless."""
    if not isinstance(item, dict):
        return {}
    if "content" in item and isinstance(item["content"], dict):
        merged = {**item, **item["content"]}
        return merged
    return item


def _extract_publish_ts(item: dict[str, Any]) -> int | None:
    # Old shape: providerPublishTime (epoch seconds)
    v = item.get("providerPublishTime")
    if isinstance(v, (int, float)):
        return int(v)
    # New shape: pubDate (ISO8601 string)
    pd = item.get("pubDate") or item.get("displayTime")
    if isinstance(pd, str):
        try:
            dt = datetime.fromisoformat(pd.replace("Z", "+00:00"))
            return int(dt.timestamp())
        except ValueError:
            return None
    return None


def _extract_link(item: dict[str, Any]) -> str | None:
    link = item.get("link")
    if isinstance(link, str) and link:
        return link
    cu = item.get("canonicalUrl")
    if isinstance(cu, dict):
        u = cu.get("url")
        if isinstance(u, str) and u:
            return u
    cl = item.get("clickThroughUrl")
    if isinstance(cl, dict):
        u = cl.get("url")
        if isinstance(u, str) and u:
            return u
    return None


def _extract_publisher(item: dict[str, Any]) -> str | None:
    p = item.get("publisher")
    if isinstance(p, str) and p:
        return p
    prov = item.get("provider")
    if isinstance(prov, dict):
        name = prov.get("displayName") or prov.get("name")
        if isinstance(name, str):
            return name
    return None


def _extract_summary(item: dict[str, Any]) -> str | None:
    for k in ("summary", "description"):
        v = item.get(k)
        if isinstance(v, str) and v:
            return v
    return None


def _extract_related(item: dict[str, Any]) -> list[str]:
    rel = item.get("relatedTickers")
    if isinstance(rel, list):
        return [str(x) for x in rel if isinstance(x, (str, int))]
    # Newer shape: list of dicts with "symbol"
    fin = item.get("finance")
    if isinstance(fin, dict):
        syms = fin.get("relatedTickers") or fin.get("symbols")
        if isinstance(syms, list):
            return [str(x) for x in syms if isinstance(x, (str, int))]
    return []


def _normalize(item: dict[str, Any], ticker: str) -> dict[str, Any]:
    flat = _flatten_item(item)
    return {
        "uuid": flat.get("uuid") or flat.get("id"),
        "title": flat.get("title"),
        "publisher": _extract_publisher(flat),
        "providerPublishTime": _extract_publish_ts(flat),
        "summary": _extract_summary(flat),
        "link": _extract_link(flat),
        "relatedTickers": _extract_related(flat),
        "ticker": ticker.upper(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_ticker(ticker: str, logger: logging.Logger) -> int:
    try:
        news = yf.Ticker(ticker).news or []
    except Exception as exc:  # noqa: BLE001 - yfinance can raise broadly
        logger.warning("%s: yfinance error %s", ticker, exc)
        return 0

    if not isinstance(news, list):
        logger.warning("%s: unexpected news shape %s", ticker, type(news).__name__)
        return 0

    out_path = _output_path(ticker)
    existing = _load_existing_keys(out_path)

    new_count = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for raw in news:
            if not isinstance(raw, dict):
                continue
            row = _normalize(raw, ticker)
            key = row.get("uuid") or row.get("link") or row.get("title")
            if not isinstance(key, str) or not key:
                continue
            if key in existing:
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing.add(key)
            new_count += 1

    logger.info("%s: wrote %d new items -> %s", ticker, new_count, out_path)
    return new_count


def run_once(tickers: Iterable[str], logger: logging.Logger) -> int:
    tickers = list(tickers)
    total = 0
    for t in tickers:
        try:
            total += fetch_ticker(t, logger)
        except Exception as exc:  # noqa: BLE001 - defensive
            logger.exception("%s: unhandled error %s", t, exc)
        # Add tiny jitter so we are not perfectly periodic.
        time.sleep(PER_TICKER_SLEEP_SEC + random.uniform(0, 0.5))
    logger.info("cycle complete: %d new items across %d tickers", total, len(tickers))
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Yahoo Finance ticker news scraper")
    p.add_argument("--once", action="store_true", help="Run a single pull cycle and exit")
    p.add_argument("--loop", action="store_true", help="Run continuously")
    p.add_argument("--interval-sec", type=int, default=300, help="Seconds between cycles in --loop mode")
    p.add_argument(
        "--tickers",
        type=str,
        default=",".join(DEFAULT_TICKERS),
        help="Comma-separated tickers",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not args.once and not args.loop:
        args.once = True

    logger = _setup_logger()
    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    logger.info("starting yahoo news scraper; tickers=%s mode=%s", tickers, "loop" if args.loop else "once")

    if args.once:
        run_once(tickers, logger)
        return 0

    while True:
        try:
            run_once(tickers, logger)
        except KeyboardInterrupt:
            logger.info("interrupted; exiting")
            return 0
        except Exception as exc:  # noqa: BLE001 - keep loop alive
            logger.exception("cycle error: %s", exc)
        time.sleep(max(30, args.interval_sec))


if __name__ == "__main__":
    raise SystemExit(main())
