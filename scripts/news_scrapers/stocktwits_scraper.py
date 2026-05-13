"""StockTwits public stream scraper for paper-trading research.

Pulls the latest messages for each ticker from the public StockTwits stream
API and appends them to a per-day JSONL log. Dedupes by message id so it is
safe to run repeatedly. This is research-only context data; it is never
treated as `paper_v1` evidence (see CLAUDE.md).

Output layout:
    data/news/stocktwits/{YYYY-MM-DD}/{ticker}.jsonl

CLI:
    --once                 Pull every ticker once and exit.
    --loop                 Pull repeatedly with --interval-sec between cycles.
    --interval-sec N       Seconds between cycles in --loop mode (default 300).
    --tickers SPY,NVDA     Override the default ticker list (comma-separated).
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import requests

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data" / "news" / "stocktwits"
LOG_DIR = REPO_ROOT / "data" / "logs"
LOG_FILE = LOG_DIR / "stocktwits_scraper.log"

DEFAULT_TICKERS = ("SPY", "NVDA", "TSLA", "AAPL", "META", "AMD", "QQQ")
STREAM_URL = "https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"

# Real-looking Chrome-on-macOS UA so the public endpoint does not 403 us.
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://stocktwits.com",
    "Referer": "https://stocktwits.com/",
}

REQUEST_TIMEOUT_SEC = 15
MAX_BACKOFF_SEC = 120
PER_TICKER_SLEEP_SEC = 1.5  # be polite between tickers


def _setup_logger() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("stocktwits_scraper")
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


def _load_existing_ids(path: Path) -> set[int]:
    ids: set[int] = set()
    if not path.exists():
        return ids
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    mid = rec.get("id")
                    if isinstance(mid, int):
                        ids.add(mid)
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return ids


def _extract_message(msg: dict, ticker: str) -> dict:
    user = msg.get("user") or {}
    entities = msg.get("entities") or {}
    sentiment = (entities.get("sentiment") or {}) if isinstance(entities, dict) else {}
    return {
        "id": msg.get("id"),
        "created_at": msg.get("created_at"),
        "body": msg.get("body"),
        "user": {
            "username": user.get("username"),
            "followers": user.get("followers"),
        },
        "entities": {
            "sentiment": {
                "basic": sentiment.get("basic") if isinstance(sentiment, dict) else None,
            },
        },
        "ticker": ticker.upper(),
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def fetch_ticker(session: requests.Session, ticker: str, logger: logging.Logger) -> int:
    """Fetch latest messages for one ticker. Returns count of new rows written."""
    url = STREAM_URL.format(ticker=ticker.upper())
    backoff = 5
    for attempt in range(1, 5):
        try:
            resp = session.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT_SEC)
        except requests.RequestException as exc:
            logger.warning("%s: network error %s (attempt %d)", ticker, exc, attempt)
            time.sleep(min(backoff, MAX_BACKOFF_SEC))
            backoff *= 2
            continue

        if resp.status_code == 200:
            break
        if resp.status_code in (429, 403):
            logger.warning(
                "%s: HTTP %s, backing off %ds (attempt %d)",
                ticker, resp.status_code, backoff, attempt,
            )
            time.sleep(min(backoff + random.uniform(0, 2), MAX_BACKOFF_SEC))
            backoff *= 2
            continue
        # Other non-200 -> log and skip
        logger.error("%s: HTTP %s, skipping. Body: %s", ticker, resp.status_code, resp.text[:200])
        return 0
    else:
        logger.error("%s: gave up after retries", ticker)
        return 0

    try:
        data = resp.json()
    except ValueError:
        logger.error("%s: invalid JSON response", ticker)
        return 0

    messages = data.get("messages") or []
    if not isinstance(messages, list):
        logger.error("%s: unexpected payload shape", ticker)
        return 0

    out_path = _output_path(ticker)
    existing = _load_existing_ids(out_path)

    new_count = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for msg in messages[:30]:
            if not isinstance(msg, dict):
                continue
            mid = msg.get("id")
            if not isinstance(mid, int) or mid in existing:
                continue
            row = _extract_message(msg, ticker)
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            existing.add(mid)
            new_count += 1

    logger.info("%s: wrote %d new messages -> %s", ticker, new_count, out_path)
    return new_count


def run_once(tickers: Iterable[str], logger: logging.Logger) -> int:
    session = requests.Session()
    total = 0
    for t in tickers:
        try:
            total += fetch_ticker(session, t, logger)
        except Exception as exc:  # noqa: BLE001 - defensive
            logger.exception("%s: unhandled error %s", t, exc)
        time.sleep(PER_TICKER_SLEEP_SEC)
    logger.info("cycle complete: %d new messages across %d tickers", total, len(list(tickers)))
    return total


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="StockTwits public stream scraper")
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
    logger.info("starting stocktwits scraper; tickers=%s mode=%s", tickers, "loop" if args.loop else "once")

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
