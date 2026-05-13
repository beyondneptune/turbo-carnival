"""Unified financial news scraper for Reuters, MarketWatch, CNBC.

Purpose: log news catalysts so paper trading losses (Aegis/Falcon/Valk on SPY/HFT)
can be attributed to real-world events later.

This is a passive, read-only logger. It does NOT touch live trading code, models,
features, thresholds, risk gates, or paper_v1 evidence. CLAUDE.md hard rules
respected: no synthetic prices, no `paper_v1` modifications, no commits implied.

Output layout:
    data/news/{source}/{YYYY-MM-DD}/headlines.jsonl
        (append-only, deduped by url within file)

Usage:
    .venv/bin/python scripts/news_scrapers/financial_news_scraper.py --once
    .venv/bin/python scripts/news_scrapers/financial_news_scraper.py --loop --interval-sec 600
    .venv/bin/python scripts/news_scrapers/financial_news_scraper.py --once --sources reuters,cnbc
    .venv/bin/python scripts/news_scrapers/financial_news_scraper.py --once --headed

Sources are isolated: if one fails (paywall, layout change, network), the others
still run. Per-source results are reported on stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence

try:
    from playwright.sync_api import (
        BrowserContext,
        Page,
        TimeoutError as PlaywrightTimeoutError,
        sync_playwright,
    )
except ImportError as e:  # pragma: no cover - explicit guidance for setup
    print(
        "ERROR: playwright not installed. Run:\n"
        "  .venv/bin/pip install playwright\n"
        "  .venv/bin/playwright install chromium",
        file=sys.stderr,
    )
    raise SystemExit(2) from e


# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
NEWS_DIR = REPO_ROOT / "data" / "news"
PROFILE_DIR = REPO_ROOT / "data" / "browser_profiles" / "news"
LOG_DIR = REPO_ROOT / "data" / "logs"

DEFAULT_INTERVAL_SEC = 600
INTER_SOURCE_DELAY_SEC = 2.5

# Tickers we care about (used for related_tickers extraction & Reuters per-ticker pages)
TICKERS = ["SPY", "NVDA", "TSLA", "AAPL", "META", "AMD", "QQQ"]

# Realistic recent Chrome UA (macOS) so sites don't throw us into a bot path.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/130.0.0.0 Safari/537.36"
)

# Common consent-button text variants. Lowercased compared.
CONSENT_TEXTS = (
    "accept all",
    "i accept",
    "accept cookies",
    "accept",
    "agree",
    "i agree",
    "got it",
    "continue",
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "news_scraper.log"
    logger = logging.getLogger("news_scraper")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh = logging.FileHandler(log_path)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


log = _setup_logging()


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclass
class Headline:
    source: str
    url: str
    headline: str
    summary: str = ""
    timestamp: str = ""  # site-reported timestamp string (raw)
    category: str = ""
    related_tickers: list[str] = field(default_factory=list)
    scraped_at: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "url": self.url,
            "headline": self.headline,
            "summary": self.summary,
            "timestamp": self.timestamp,
            "category": self.category,
            "related_tickers": self.related_tickers,
            "scraped_at": self.scraped_at,
        }


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------


def _today_str() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _out_path(source: str) -> Path:
    p = NEWS_DIR / source / _today_str() / "headlines.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_existing_urls(path: Path) -> set[str]:
    if not path.exists():
        return set()
    urls: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                u = obj.get("url")
                if u:
                    urls.add(u)
            except json.JSONDecodeError:
                continue
    return urls


def append_headlines(source: str, items: Iterable[Headline]) -> int:
    """Append new headlines to today's per-source jsonl. Dedupe by url. Returns count saved."""
    path = _out_path(source)
    existing = _load_existing_urls(path)
    written = 0
    with path.open("a", encoding="utf-8") as fh:
        for h in items:
            if not h.url or not h.headline:
                continue
            if h.url in existing:
                continue
            existing.add(h.url)
            fh.write(json.dumps(h.to_dict(), ensure_ascii=False) + "\n")
            written += 1
    return written


# ---------------------------------------------------------------------------
# Page helpers
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_goto(page: Page, url: str, timeout_ms: int = 30000) -> tuple[bool, int | None]:
    """Navigate to url. Returns (ok, http_status). ok is False on timeout/exception
    or on >=400 status (caller can decide to bail)."""
    try:
        resp = page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        status = resp.status if resp is not None else None
        if status is not None and status >= 400:
            log.warning("goto %s -> HTTP %d", url, status)
            return False, status
        return True, status
    except PlaywrightTimeoutError:
        log.warning("goto timeout: %s", url)
        return False, None
    except Exception as e:  # noqa: BLE001
        log.warning("goto error %s: %s", url, e)
        return False, None


def _wait_for_content(page: Page, max_wait_ms: int = 3500) -> None:
    """Wait for content. Prefer network idle but bounded; fall back to a short sleep."""
    try:
        page.wait_for_load_state("networkidle", timeout=max_wait_ms)
    except PlaywrightTimeoutError:
        pass
    # belt + suspenders
    page.wait_for_timeout(500)


def _click_consent(page: Page) -> None:
    """Best-effort cookie / consent dismissal. Tries main frame and child frames."""
    targets = list(page.frames)
    for frame in targets:
        for text in CONSENT_TEXTS:
            try:
                btn = frame.get_by_role("button", name=re.compile(text, re.IGNORECASE))
                if btn.count() > 0:
                    btn.first.click(timeout=1500)
                    page.wait_for_timeout(500)
                    return
            except Exception:  # noqa: BLE001
                pass
        # text-based fallback
        for text in CONSENT_TEXTS:
            try:
                loc = frame.get_by_text(re.compile(rf"^\s*{text}\s*$", re.IGNORECASE))
                if loc.count() > 0:
                    loc.first.click(timeout=1500)
                    page.wait_for_timeout(500)
                    return
            except Exception:  # noqa: BLE001
                pass


def _scroll_once(page: Page) -> None:
    try:
        page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.6)")
        page.wait_for_timeout(800)
        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        page.wait_for_timeout(800)
    except Exception:  # noqa: BLE001
        pass


def _absolutize(base: str, href: str) -> str:
    if not href:
        return ""
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if href.startswith("//"):
        return "https:" + href
    if href.startswith("/"):
        # strip path off base
        m = re.match(r"^(https?://[^/]+)", base)
        if m:
            return m.group(1) + href
    return href


def _extract_tickers(text: str) -> list[str]:
    if not text:
        return []
    found: list[str] = []
    upper = text.upper()
    for t in TICKERS:
        # word-boundary so we don't catch "AAPLE"
        if re.search(rf"\b{re.escape(t)}\b", upper):
            found.append(t)
    return found


# ---------------------------------------------------------------------------
# Source: Reuters Markets
# ---------------------------------------------------------------------------


def scrape_reuters(context: BrowserContext) -> list[Headline]:
    page = context.new_page()
    out: list[Headline] = []
    try:
        url = "https://www.reuters.com/markets/"
        ok, status = _safe_goto(page, url)
        if not ok:
            raise RuntimeError(
                f"reuters blocked/unreachable (status={status}); likely DataDome CAPTCHA"
            )
        # Reuters serves a 200 captcha page too — detect by title length
        try:
            content_snippet = page.content()[:2000]
            if "captcha-delivery.com" in content_snippet or "DataDome" in content_snippet:
                raise RuntimeError("reuters served DataDome CAPTCHA challenge")
        except Exception:
            pass
        _wait_for_content(page)
        _click_consent(page)
        _scroll_once(page)
        _wait_for_content(page, max_wait_ms=2000)

        # Reuters story links live under /world, /markets, /business, /technology, etc.
        # Strategy: collect anchors with story-like hrefs, then derive headline/summary.
        anchors = page.evaluate(
            r"""
            () => {
                const out = [];
                const seen = new Set();
                const links = document.querySelectorAll('a[href]');
                for (const a of links) {
                    const href = a.getAttribute('href') || '';
                    if (!href) continue;
                    // Reuters article URLs typically contain a -YYYY-MM-DD slug and end with /
                    if (!/^\/(?:world|markets|business|technology|legal|sustainability|breakingviews|sports|lifestyle|graphics)\//.test(href)) continue;
                    if (!/\-\d{4}-\d{2}-\d{2}\/?$/.test(href) && !/-20\d{2}-\d{2}-\d{2}/.test(href)) {
                        // also allow generic article slugs
                        if (!/[a-z]+-[a-z0-9-]{8,}/.test(href)) continue;
                    }
                    if (seen.has(href)) continue;
                    seen.add(href);
                    const text = (a.innerText || a.textContent || '').trim();
                    if (!text || text.length < 8) continue;
                    // Walk up to a parent that might hold a timestamp/summary
                    let p = a.parentElement;
                    let summary = '';
                    let ts = '';
                    for (let i = 0; i < 4 && p; i++) {
                        const t = p.querySelector('time');
                        if (t && !ts) ts = t.getAttribute('datetime') || t.innerText || '';
                        const ps = p.querySelectorAll('p');
                        for (const pp of ps) {
                            const txt = (pp.innerText || '').trim();
                            if (txt && txt.length > 30 && txt !== text) { summary = txt; break; }
                        }
                        if (summary) break;
                        p = p.parentElement;
                    }
                    out.push({href, text, summary, ts});
                }
                return out;
            }
            """
        )

        now = _now_iso()
        for a in anchors:
            href = _absolutize("https://www.reuters.com", a.get("href", ""))
            headline = (a.get("text") or "").strip()
            if not headline or not href:
                continue
            # Filter obvious nav links
            if len(headline) < 12:
                continue
            cat = ""
            m = re.match(r"https?://[^/]+/([^/]+)/", href)
            if m:
                cat = m.group(1)
            out.append(
                Headline(
                    source="reuters",
                    url=href,
                    headline=headline,
                    summary=(a.get("summary") or "").strip(),
                    timestamp=(a.get("ts") or "").strip(),
                    category=cat,
                    related_tickers=_extract_tickers(headline + " " + (a.get("summary") or "")),
                    scraped_at=now,
                )
            )

        # Per-ticker pages (best-effort)
        for tkr in TICKERS:
            tkr_url = f"https://www.reuters.com/companies/{tkr}.OQ"
            try:
                resp = page.goto(tkr_url, wait_until="domcontentloaded", timeout=15000)
                if resp and resp.status >= 400:
                    log.info("reuters ticker page %s -> %d (skip)", tkr, resp.status)
                    continue
                _wait_for_content(page, max_wait_ms=2000)
                _scroll_once(page)
                tkr_anchors = page.evaluate(
                    r"""
                    () => {
                        const out = [];
                        const seen = new Set();
                        const links = document.querySelectorAll('a[href]');
                        for (const a of links) {
                            const href = a.getAttribute('href') || '';
                            if (!/^\/(?:world|markets|business|technology|legal|sustainability|breakingviews)\//.test(href)) continue;
                            if (seen.has(href)) continue;
                            seen.add(href);
                            const text = (a.innerText || '').trim();
                            if (!text || text.length < 12) continue;
                            out.push({href, text});
                        }
                        return out.slice(0, 25);
                    }
                    """
                )
                for a in tkr_anchors:
                    href = _absolutize("https://www.reuters.com", a.get("href", ""))
                    headline = (a.get("text") or "").strip()
                    if not href or not headline:
                        continue
                    out.append(
                        Headline(
                            source="reuters",
                            url=href,
                            headline=headline,
                            summary="",
                            timestamp="",
                            category=f"ticker:{tkr}",
                            related_tickers=[tkr],
                            scraped_at=now,
                        )
                    )
            except PlaywrightTimeoutError:
                log.info("reuters ticker page %s timeout (skip)", tkr)
            except Exception as e:  # noqa: BLE001
                log.info("reuters ticker page %s error: %s (skip)", tkr, e)

    finally:
        page.close()
    return out


# ---------------------------------------------------------------------------
# Source: MarketWatch latest-news
# ---------------------------------------------------------------------------


def scrape_marketwatch(context: BrowserContext) -> list[Headline]:
    page = context.new_page()
    out: list[Headline] = []
    try:
        url = "https://www.marketwatch.com/latest-news"
        ok, status = _safe_goto(page, url)
        if not ok:
            raise RuntimeError(f"marketwatch goto failed (status={status})")
        _wait_for_content(page)
        _click_consent(page)
        _scroll_once(page)
        _wait_for_content(page, max_wait_ms=2000)

        items = page.evaluate(
            r"""
            () => {
                const out = [];
                const seen = new Set();
                const articles = document.querySelectorAll('div.article__content, .article__content, article');
                for (const art of articles) {
                    const a = art.querySelector('a.link, a[href*="/story/"], h3 a, h2 a, a');
                    if (!a) continue;
                    const href = a.getAttribute('href') || '';
                    if (!href || seen.has(href)) continue;
                    seen.add(href);
                    const headline = (a.innerText || a.textContent || '').trim();
                    if (!headline || headline.length < 8) continue;
                    const summaryEl = art.querySelector('.article__summary, p.article__summary, p');
                    const summary = summaryEl ? (summaryEl.innerText || '').trim() : '';
                    const timeEl = art.querySelector('.article__timestamp, time, span.article__timestamp');
                    const ts = timeEl ? (timeEl.getAttribute('data-est') || timeEl.getAttribute('datetime') || timeEl.innerText || '').trim() : '';
                    // related tickers (MarketWatch sometimes links them)
                    const tickers = [];
                    art.querySelectorAll('a').forEach(x => {
                        const t = (x.innerText || '').trim();
                        if (/^[A-Z]{1,5}$/.test(t)) tickers.push(t);
                    });
                    out.push({href, headline, summary, ts, tickers});
                }
                return out;
            }
            """
        )

        now = _now_iso()
        for it in items:
            href = _absolutize("https://www.marketwatch.com", it.get("href", ""))
            headline = (it.get("headline") or "").strip()
            if not href or not headline:
                continue
            related = list(dict.fromkeys(it.get("tickers", [])))  # de-dup, preserve order
            # also extract from text just in case
            related = list(dict.fromkeys(related + _extract_tickers(headline + " " + (it.get("summary") or ""))))
            out.append(
                Headline(
                    source="marketwatch",
                    url=href,
                    headline=headline,
                    summary=(it.get("summary") or "").strip(),
                    timestamp=(it.get("ts") or "").strip(),
                    category="latest-news",
                    related_tickers=[t for t in related if t in TICKERS or len(t) <= 5],
                    scraped_at=now,
                )
            )
    finally:
        page.close()
    return out


# ---------------------------------------------------------------------------
# Source: CNBC Markets
# ---------------------------------------------------------------------------


def scrape_cnbc(context: BrowserContext) -> list[Headline]:
    page = context.new_page()
    out: list[Headline] = []
    try:
        url = "https://www.cnbc.com/markets/"
        ok, status = _safe_goto(page, url)
        if not ok:
            raise RuntimeError(f"cnbc goto failed (status={status})")
        _wait_for_content(page)
        _click_consent(page)
        _scroll_once(page)
        _wait_for_content(page, max_wait_ms=2000)

        items = page.evaluate(
            r"""
            () => {
                const out = [];
                const seen = new Set();
                // CNBC card structure: LatestNews-headline, RiverHeadline, Card-title, etc.
                const anchors = document.querySelectorAll('a[href*="/2024/"], a[href*="/2025/"], a[href*="/2026/"], a.LatestNews-headline, a.Card-title, a.RiverHeadline-headline');
                for (const a of anchors) {
                    const href = a.getAttribute('href') || '';
                    if (!href || seen.has(href)) continue;
                    seen.add(href);
                    const headline = (a.innerText || a.textContent || '').trim();
                    if (!headline || headline.length < 8) continue;
                    // climb to a container for summary/ts
                    let p = a.parentElement;
                    let summary = '';
                    let ts = '';
                    for (let i = 0; i < 4 && p; i++) {
                        if (!summary) {
                            const sEl = p.querySelector('.Card-description, .RiverHeadline-description, p');
                            if (sEl) {
                                const t = (sEl.innerText || '').trim();
                                if (t && t !== headline && t.length > 20) summary = t;
                            }
                        }
                        if (!ts) {
                            const tEl = p.querySelector('time, .LatestNews-timestamp, .Card-time');
                            if (tEl) ts = (tEl.getAttribute('datetime') || tEl.innerText || '').trim();
                        }
                        if (summary && ts) break;
                        p = p.parentElement;
                    }
                    out.push({href, headline, summary, ts});
                }
                return out;
            }
            """
        )

        now = _now_iso()
        for it in items:
            href = _absolutize("https://www.cnbc.com", it.get("href", ""))
            headline = (it.get("headline") or "").strip()
            if not href or not headline:
                continue
            # CNBC quote/video pages: skip pure video/quote pages without article slug? keep them all for now.
            out.append(
                Headline(
                    source="cnbc",
                    url=href,
                    headline=headline,
                    summary=(it.get("summary") or "").strip(),
                    timestamp=(it.get("ts") or "").strip(),
                    category="markets",
                    related_tickers=_extract_tickers(headline + " " + (it.get("summary") or "")),
                    scraped_at=now,
                )
            )
    finally:
        page.close()
    return out


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


SourceFn = Callable[[BrowserContext], list[Headline]]

SOURCES: dict[str, SourceFn] = {
    "reuters": scrape_reuters,
    "marketwatch": scrape_marketwatch,
    "cnbc": scrape_cnbc,
}


def run_once(selected_sources: Sequence[str], headed: bool) -> dict[str, dict]:
    """Run a single sweep across selected sources. Returns per-source results dict.

    Uses a separate persistent browser context per source so anti-bot cookies
    (e.g. DataDome) don't leak between sites and cascade failures. Each source
    has its own profile under data/browser_profiles/news/{source}/ .
    """
    NEWS_DIR.mkdir(parents=True, exist_ok=True)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    with sync_playwright() as p:
        first = True
        for name in selected_sources:
            fn = SOURCES.get(name)
            if fn is None:
                log.warning("unknown source: %s (skip)", name)
                results[name] = {"error": "unknown source", "saved": 0, "collected": 0}
                continue
            if not first:
                time.sleep(INTER_SOURCE_DELAY_SEC)
            first = False

            per_source_profile = PROFILE_DIR / name
            per_source_profile.mkdir(parents=True, exist_ok=True)

            log.info("scraping %s ...", name)
            started = time.time()
            context = None
            try:
                context = p.chromium.launch_persistent_context(
                    user_data_dir=str(per_source_profile),
                    headless=not headed,
                    user_agent=USER_AGENT,
                    viewport={"width": 1366, "height": 900},
                    locale="en-US",
                    args=["--disable-blink-features=AutomationControlled"],
                )
                items = fn(context)
                saved = append_headlines(name, items)
                elapsed = time.time() - started
                sample = [
                    {"headline": h.headline[:140], "url": h.url}
                    for h in items[:2]
                ]
                results[name] = {
                    "collected": len(items),
                    "saved": saved,
                    "elapsed_sec": round(elapsed, 1),
                    "sample": sample,
                }
                log.info(
                    "%s: collected=%d saved_new=%d elapsed=%.1fs",
                    name,
                    len(items),
                    saved,
                    elapsed,
                )
            except Exception as e:  # noqa: BLE001
                log.exception("source %s failed: %s", name, e)
                results[name] = {
                    "error": f"{type(e).__name__}: {e}",
                    "saved": 0,
                    "collected": 0,
                }
            finally:
                if context is not None:
                    try:
                        context.close()
                    except Exception:  # noqa: BLE001
                        pass
    return results


def run_loop(selected_sources: Sequence[str], interval_sec: int, headed: bool) -> None:
    log.info(
        "loop mode: sources=%s interval=%ds headed=%s",
        ",".join(selected_sources),
        interval_sec,
        headed,
    )
    while True:
        sweep_started = time.time()
        try:
            results = run_once(selected_sources, headed=headed)
            totals = sum(r.get("saved", 0) for r in results.values())
            log.info("sweep complete: new_total=%d", totals)
        except Exception as e:  # noqa: BLE001
            log.exception("sweep failed: %s", e)
        elapsed = time.time() - sweep_started
        sleep_for = max(5, interval_sec - int(elapsed))
        log.info("sleeping %ds before next sweep", sleep_for)
        time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true", help="single sweep across selected sources then exit")
    ap.add_argument("--loop", action="store_true", help="repeat sweeps forever")
    ap.add_argument(
        "--interval-sec",
        type=int,
        default=DEFAULT_INTERVAL_SEC,
        help=f"loop interval seconds (default {DEFAULT_INTERVAL_SEC})",
    )
    ap.add_argument(
        "--sources",
        type=str,
        default="reuters,marketwatch,cnbc",
        help="comma-separated source names (reuters,marketwatch,cnbc)",
    )
    ap.add_argument("--headed", action="store_true", help="show browser window (default headless)")
    args = ap.parse_args(argv)
    if not args.once and not args.loop:
        args.once = True
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    sources = [s.strip() for s in args.sources.split(",") if s.strip()]
    invalid = [s for s in sources if s not in SOURCES]
    if invalid:
        log.error("invalid sources: %s. valid: %s", invalid, list(SOURCES))
        return 2

    if args.loop:
        try:
            run_loop(sources, args.interval_sec, headed=args.headed)
        except KeyboardInterrupt:
            log.info("loop interrupted by user")
        return 0

    results = run_once(sources, headed=args.headed)
    print("\n=== SCRAPE RESULTS ===")
    for name, info in results.items():
        print(f"\n[{name}]")
        if "error" in info:
            print(f"  ERROR: {info['error']}")
            continue
        print(f"  collected: {info.get('collected', 0)}")
        print(f"  saved_new: {info.get('saved', 0)}")
        print(f"  elapsed:   {info.get('elapsed_sec', 0)}s")
        for s in info.get("sample", []):
            print(f"   - {s['headline'][:120]}")
            print(f"     {s['url']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
