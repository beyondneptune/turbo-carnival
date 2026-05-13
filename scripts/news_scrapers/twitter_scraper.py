#!/usr/bin/env python3
"""Twitter/X scraper for paper-trading research context.

Uses Playwright with a persistent browser context so the user logs in once
(via ``--login-mode``) and subsequent runs reuse the saved cookies headlessly.

Outputs JSONL per ticker per day under ``data/news/twitter/{YYYY-MM-DD}/{ticker}.jsonl``,
deduped across runs by ``tweet_id``.

Does NOT touch live trading code or tmux sessions.

Usage examples
--------------
One-time interactive login (headed browser, user logs in to x.com):
    .venv/bin/python scripts/news_scrapers/twitter_scraper.py --login-mode

Single sweep across the default ticker set (headless, after login):
    .venv/bin/python scripts/news_scrapers/twitter_scraper.py --once

Continuous loop, 5 minutes between sweeps:
    .venv/bin/python scripts/news_scrapers/twitter_scraper.py --loop --interval-sec 300
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

# Resolve repo root so we can run from anywhere.
REPO_ROOT = Path(__file__).resolve().parents[2]
BROWSER_PROFILE_DIR = REPO_ROOT / "data" / "browser_profiles" / "twitter"
NEWS_OUT_DIR = REPO_ROOT / "data" / "news" / "twitter"
LOG_DIR = REPO_ROOT / "data" / "logs"
LOG_FILE = LOG_DIR / "twitter_scraper.log"

DEFAULT_TICKERS = ["SPY", "NVDA", "TSLA", "AAPL", "META", "AMD", "QQQ"]
DEFAULT_MAX_TWEETS = 60
SCROLL_COUNT = 5
SCROLL_WAIT_SEC = 1.5
NAV_TIMEOUT_MS = 45_000
PAGE_LOAD_SETTLE_SEC = 3.0

TWEET_URL_RE = re.compile(r"^/(?P<author>[A-Za-z0-9_]+)/status/(?P<tweet_id>\d+)")


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("twitter_scraper")
    logger.setLevel(logging.INFO)
    # Avoid duplicate handlers if re-imported.
    if logger.handlers:
        return logger
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    fh = logging.FileHandler(LOG_FILE)
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stderr)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


log = _setup_logging()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _out_path(ticker: str) -> Path:
    day_dir = NEWS_OUT_DIR / _today_str()
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir / f"{ticker.upper()}.jsonl"


def _load_seen_ids(path: Path) -> set[str]:
    """Load all tweet_ids already written to this file (cheap, line-by-line)."""
    if not path.exists():
        return set()
    seen: set[str] = set()
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                tid = obj.get("tweet_id")
                if tid:
                    seen.add(str(tid))
    except OSError as exc:
        log.warning("Could not read %s for dedupe: %s", path, exc)
    return seen


def _parse_count(text: str | None) -> int:
    """Parse Twitter's compact counts like '1.2K', '3M', '847'."""
    if not text:
        return 0
    t = text.strip().replace(",", "").upper()
    if not t:
        return 0
    mult = 1.0
    if t.endswith("K"):
        mult, t = 1_000.0, t[:-1]
    elif t.endswith("M"):
        mult, t = 1_000_000.0, t[:-1]
    elif t.endswith("B"):
        mult, t = 1_000_000_000.0, t[:-1]
    try:
        return int(float(t) * mult)
    except ValueError:
        return 0


def _search_url(ticker: str) -> str:
    # %24 == $ for cashtag; lang:en; f=live for latest.
    q = f"${ticker.upper()} lang:en"
    return f"https://x.com/search?q={urllib.parse.quote(q)}&f=live"


# --------------------------------------------------------------------------- #
# Login mode
# --------------------------------------------------------------------------- #
def run_login(playwright) -> None:
    """Open headed browser, let the user log in, persist cookies, exit."""
    BROWSER_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    log.info("Launching headed Chromium for interactive login -> %s", BROWSER_PROFILE_DIR)
    ctx = playwright.chromium.launch_persistent_context(
        user_data_dir=str(BROWSER_PROFILE_DIR),
        headless=False,
        viewport={"width": 1280, "height": 900},
    )
    try:
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.goto("https://x.com/login", timeout=NAV_TIMEOUT_MS)
        print(
            "\n>>> A browser window has opened. Log in to x.com, then come back\n"
            ">>> to this terminal and press Enter to save cookies and exit.\n",
            file=sys.stderr,
            flush=True,
        )
        try:
            input()
        except EOFError:
            log.warning("No stdin available; sleeping 120s for manual login instead.")
            time.sleep(120)
        log.info("Saving persistent context and exiting.")
    finally:
        ctx.close()


# --------------------------------------------------------------------------- #
# Scrape
# --------------------------------------------------------------------------- #
def _is_blocked_page(page) -> str | None:
    """Detect interstitials we should skip rather than crash on."""
    try:
        body_text = page.locator("body").inner_text(timeout=3_000)
    except Exception:  # noqa: BLE001 - playwright TimeoutError etc.
        return None
    if not body_text:
        return None
    low = body_text.lower()
    needles = [
        "rate limit",
        "rate-limited",
        "log in to twitter",
        "log in to x",
        "sign up to see",
        "something went wrong",
        "try reloading",
    ]
    for n in needles:
        if n in low:
            return n
    return None


def _extract_tweets_from_page(page, ticker: str, limit: int) -> list[dict]:
    """Return a list of tweet dicts scraped from the current search page."""
    tweets: list[dict] = []
    seen_ids_local: set[str] = set()

    articles = page.locator("article[data-testid='tweet']")
    try:
        count = articles.count()
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] Could not count tweets: %s", ticker, exc)
        return tweets

    for i in range(count):
        if len(tweets) >= limit:
            break
        art = articles.nth(i)
        try:
            # Time element -> ISO timestamp + status URL.
            time_el = art.locator("time").first
            datetime_attr = time_el.get_attribute("datetime", timeout=2_000)
            # The <time> is wrapped in an <a> to /<author>/status/<id>.
            link_el = time_el.locator("xpath=ancestor::a[1]")
            href = link_el.get_attribute("href", timeout=2_000) or ""
        except Exception:  # noqa: BLE001
            continue

        m = TWEET_URL_RE.match(href)
        if not m:
            continue
        tweet_id = m.group("tweet_id")
        author = m.group("author")
        if tweet_id in seen_ids_local:
            continue

        # Tweet text (may be missing for media-only tweets).
        text = ""
        try:
            txt_el = art.locator("div[data-testid='tweetText']").first
            if txt_el.count() > 0:
                text = txt_el.inner_text(timeout=1_500)
        except Exception:  # noqa: BLE001
            text = ""

        # Engagement counts via aria-labels on the action buttons.
        def _btn_count(testid: str) -> int:
            try:
                btn = art.locator(f"button[data-testid='{testid}']").first
                if btn.count() == 0:
                    return 0
                inner = btn.locator(
                    "span[data-testid='app-text-transition-container'] span span"
                ).first
                if inner.count() > 0:
                    return _parse_count(inner.inner_text(timeout=1_000))
                # Fallback: parse the aria-label of the button itself.
                label = btn.get_attribute("aria-label", timeout=1_000) or ""
                m2 = re.search(r"([\d.,KMB]+)", label)
                return _parse_count(m2.group(1)) if m2 else 0
            except Exception:  # noqa: BLE001
                return 0

        reply = _btn_count("reply")
        retweet = _btn_count("retweet")
        like = _btn_count("like")

        url = f"https://x.com{href}" if href.startswith("/") else href
        tweets.append(
            {
                "tweet_id": tweet_id,
                "timestamp": datetime_attr,
                "author": author,
                "text": text,
                "like_count": like,
                "reply_count": reply,
                "retweet_count": retweet,
                "url": url,
                "ticker": ticker.upper(),
                "scraped_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        seen_ids_local.add(tweet_id)
    return tweets


def scrape_ticker(page, ticker: str, max_tweets: int) -> tuple[int, int]:
    """Scrape one ticker. Returns (new_written, total_seen_on_page)."""
    url = _search_url(ticker)
    log.info("[%s] navigating: %s", ticker, url)
    try:
        page.goto(url, timeout=NAV_TIMEOUT_MS, wait_until="domcontentloaded")
    except Exception as exc:  # noqa: BLE001
        log.warning("[%s] navigation failed: %s", ticker, exc)
        return (0, 0)

    time.sleep(PAGE_LOAD_SETTLE_SEC)

    block = _is_blocked_page(page)
    if block:
        log.warning("[%s] interstitial detected ('%s') - skipping", ticker, block)
        return (0, 0)

    # Lazy-load by scrolling.
    for i in range(SCROLL_COUNT):
        try:
            page.mouse.wheel(0, 2200)
        except Exception:  # noqa: BLE001
            try:
                page.evaluate("window.scrollBy(0, 2200)")
            except Exception:  # noqa: BLE001
                pass
        time.sleep(SCROLL_WAIT_SEC + random.uniform(0, 0.4))

    tweets = _extract_tweets_from_page(page, ticker, limit=max_tweets)
    if not tweets:
        log.info("[%s] no tweets extracted", ticker)
        return (0, 0)

    out_path = _out_path(ticker)
    seen = _load_seen_ids(out_path)
    new_count = 0
    with out_path.open("a", encoding="utf-8") as fh:
        for tw in tweets:
            if tw["tweet_id"] in seen:
                continue
            fh.write(json.dumps(tw, ensure_ascii=False) + "\n")
            seen.add(tw["tweet_id"])
            new_count += 1
    log.info(
        "[%s] wrote %d new tweets (saw %d on page) -> %s",
        ticker,
        new_count,
        len(tweets),
        out_path,
    )
    return (new_count, len(tweets))


def run_sweep(playwright, tickers: Iterable[str], headed: bool, max_tweets: int) -> None:
    if not BROWSER_PROFILE_DIR.exists() or not any(BROWSER_PROFILE_DIR.iterdir()):
        log.warning(
            "Browser profile %s is empty - run with --login-mode first.",
            BROWSER_PROFILE_DIR,
        )

    ctx = playwright.chromium.launch_persistent_context(
        user_data_dir=str(BROWSER_PROFILE_DIR),
        headless=not headed,
        viewport={"width": 1280, "height": 900},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/127.0.0.0 Safari/537.36"
        ),
    )
    try:
        page = ctx.pages[0] if ctx.pages else ctx.new_page()
        page.set_default_timeout(NAV_TIMEOUT_MS)
        totals_new = 0
        for ticker in tickers:
            try:
                new_n, _ = scrape_ticker(page, ticker, max_tweets)
                totals_new += new_n
            except Exception as exc:  # noqa: BLE001 - never crash the sweep.
                log.exception("[%s] unexpected error: %s", ticker, exc)
            # Small jitter between tickers to look less robotic.
            time.sleep(2.0 + random.uniform(0, 1.5))
        log.info("sweep complete: %d new tweets across %d tickers", totals_new, len(list(tickers)))
    finally:
        ctx.close()


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_tickers(raw: str | None) -> list[str]:
    if not raw:
        return list(DEFAULT_TICKERS)
    out = [t.strip().upper() for t in raw.split(",") if t.strip()]
    return out or list(DEFAULT_TICKERS)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="twitter_scraper.py",
        description=(
            "Playwright-based X/Twitter scraper for paper-trading research context. "
            "Uses a persistent browser profile so you log in once and then scrape "
            "headlessly."
        ),
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--login-mode",
        action="store_true",
        help="Open headed Chromium for interactive login, save cookies, exit.",
    )
    mode.add_argument(
        "--once",
        action="store_true",
        help="Run a single sweep across all tickers and exit.",
    )
    mode.add_argument(
        "--loop",
        action="store_true",
        help="Repeat the sweep on an interval (requires --interval-sec).",
    )

    p.add_argument(
        "--interval-sec",
        type=int,
        default=300,
        help="Seconds between sweeps when --loop is set (default: 300).",
    )
    p.add_argument(
        "--tickers",
        type=str,
        default=None,
        help=(
            "Comma-separated ticker list, e.g. 'SPY,NVDA'. "
            f"Default: {','.join(DEFAULT_TICKERS)}"
        ),
    )
    p.add_argument(
        "--headed",
        action="store_true",
        help="Show the browser during scraping (default: headless).",
    )
    p.add_argument(
        "--max-tweets",
        type=int,
        default=DEFAULT_MAX_TWEETS,
        help=f"Max tweets to record per ticker per sweep (default: {DEFAULT_MAX_TWEETS}).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # Import lazily so --help works even if playwright fails to import.
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        log.error("playwright is not installed: %s", exc)
        log.error("Install with: .venv/bin/pip install playwright && .venv/bin/playwright install chromium")
        return 2

    tickers = _parse_tickers(args.tickers)

    if args.login_mode:
        with sync_playwright() as pw:
            run_login(pw)
        return 0

    if not (args.once or args.loop):
        # Default behaviour if no mode flag: single sweep.
        log.info("no mode flag given; defaulting to --once")
        args.once = True

    if args.once:
        with sync_playwright() as pw:
            run_sweep(pw, tickers, headed=args.headed, max_tweets=args.max_tweets)
        return 0

    if args.loop:
        if args.interval_sec <= 0:
            log.error("--interval-sec must be > 0 when using --loop")
            return 2
        log.info(
            "starting loop: tickers=%s interval=%ds max_per_ticker=%d headed=%s",
            tickers,
            args.interval_sec,
            args.max_tweets,
            args.headed,
        )
        while True:
            t0 = time.monotonic()
            try:
                with sync_playwright() as pw:
                    run_sweep(pw, tickers, headed=args.headed, max_tweets=args.max_tweets)
            except Exception as exc:  # noqa: BLE001 - keep the loop alive.
                log.exception("sweep raised: %s", exc)
            elapsed = time.monotonic() - t0
            sleep_for = max(5.0, args.interval_sec - elapsed)
            log.info("sleeping %.1fs until next sweep", sleep_for)
            time.sleep(sleep_for)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
