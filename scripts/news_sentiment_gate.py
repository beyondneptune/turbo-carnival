"""News-sentiment entry gate for ATHENA v2.1.

Reads recent JSONL from the 3 live scrapers (stocktwits, yahoo, cnbc/marketwatch
under news_scraper) and produces a per-symbol sentiment score in [-1, 1].

Wiring into the host (in alpaca_max_hft_host.py, after the spread gate
around line 815):

    from scripts.news_sentiment_gate import sentiment_score_for_symbol, NEWS_BLACKOUT_THRESHOLD
    _sent = sentiment_score_for_symbol(sym)
    if _sent is not None and _sent < NEWS_BLACKOUT_THRESHOLD:
        log_decision(
            conn, lane=lane.nickname, symbol=sym, action="skip",
            reason=f"news_sentiment:{_sent:+.2f}", ...
        )
        continue

The threshold default of -0.30 means "skip entry if recent posts skew strongly
negative." Tune via the env var ATHENA_NEWS_BLACKOUT_THRESHOLD.

Sentiment scoring uses a fast lexicon-based approach (VADER-like) so we don't
need to load a transformer per-tick. Cached results expire every 60s.
"""
from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path("/Users/weeoto/turbo-carnival")
NEWS_ROOT = REPO / "data" / "news"

# Lookback window (only count posts/articles within this many seconds)
LOOKBACK_SEC = 60 * 60   # 1 hour

# Block entries when score below this threshold (range [-1, 1])
NEWS_BLACKOUT_THRESHOLD = float(os.environ.get("ATHENA_NEWS_BLACKOUT_THRESHOLD", "-0.30"))

# Cache to avoid re-reading JSONL on every tick (host loop is 2s, news arrives slower)
_CACHE: dict[str, tuple[float, float]] = {}   # symbol -> (score, computed_at)
_CACHE_TTL_SEC = 60

# Lightweight sentiment lexicon (extend as needed)
_POSITIVE_WORDS = {
    "moon", "bullish", "buy", "long", "calls", "rip", "pump", "rally", "breakout",
    "squeeze", "beat", "beats", "strong", "surge", "soar", "win", "wins", "winning",
    "great", "excellent", "love", "support", "resist", "uptrend", "green", "gain",
    "gains", "rip", "rocket", "going up", "all time high", "ath", "moonshot",
}
_NEGATIVE_WORDS = {
    "bear", "bearish", "sell", "short", "puts", "dump", "crash", "tank", "miss",
    "missed", "weak", "drop", "down", "fall", "falling", "fell", "lose", "loss",
    "losses", "bad", "terrible", "hate", "doom", "rekt", "fucked", "broken",
    "downtrend", "red", "below", "support broken", "resistance", "rejected",
    "bagholder", "bagholding", "bottom", "puke", "puked", "panic",
}


def _tokenize(text: str) -> list[str]:
    """Lowercase + strip punctuation, return word tokens."""
    return re.findall(r"[a-z']+", text.lower())


def _score_text(text: str) -> float:
    """Lexicon sentiment in [-1, 1]. Empty/neutral text returns 0."""
    if not text:
        return 0.0
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    pos = sum(1 for t in tokens if t in _POSITIVE_WORDS)
    neg = sum(1 for t in tokens if t in _NEGATIVE_WORDS)
    if pos == 0 and neg == 0:
        return 0.0
    return (pos - neg) / max(1, pos + neg)


def _load_recent_posts(symbol: str, lookback_sec: int = LOOKBACK_SEC) -> list[tuple[float, float]]:
    """Read recent (timestamp, text-score) pairs from all sources for one symbol.

    Returns list of (created_at_epoch, sentiment_score) for posts in the
    lookback window.
    """
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cutoff = time.time() - lookback_sec
    results: list[tuple[float, float]] = []

    # Stocktwits — high volume, retail sentiment
    st_path = NEWS_ROOT / "stocktwits" / today / f"{symbol}.jsonl"
    if st_path.exists():
        try:
            for line in st_path.read_text().splitlines():
                obj = json.loads(line)
                ts = obj.get("created_at", "")
                if not ts:
                    continue
                try:
                    epoch = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    continue
                if epoch < cutoff:
                    continue
                body = obj.get("body", "") or ""
                # Stocktwits sometimes embeds bull/bear tag
                tag = obj.get("entities", {}).get("sentiment", {}).get("basic")
                if tag == "Bullish":
                    results.append((epoch, 1.0))
                elif tag == "Bearish":
                    results.append((epoch, -1.0))
                else:
                    results.append((epoch, _score_text(body)))
        except Exception:
            pass

    # Yahoo — news article titles
    yh_path = NEWS_ROOT / "yahoo" / today / f"{symbol}.jsonl"
    if yh_path.exists():
        try:
            for line in yh_path.read_text().splitlines():
                obj = json.loads(line)
                epoch = obj.get("providerPublishTime", 0)
                if not epoch or epoch < cutoff:
                    continue
                title = obj.get("title", "") or ""
                summary = obj.get("summary", "") or ""
                results.append((epoch, _score_text(f"{title} {summary}")))
        except Exception:
            pass

    # CNBC / MarketWatch / Reuters from news_scraper — different format, skip for now
    # (the financial_news_scraper writes to data/news/cnbc/<date>/ but format
    # may differ; can extend here when needed)

    return results


def sentiment_score_for_symbol(symbol: str) -> float | None:
    """Aggregate sentiment score in [-1, 1] for a symbol, or None if no data.

    Returns the (recency-weighted) mean across all posts/articles in the
    lookback window. Cached for 60s.
    """
    now = time.time()
    if symbol in _CACHE:
        score, computed_at = _CACHE[symbol]
        if now - computed_at < _CACHE_TTL_SEC:
            return score

    posts = _load_recent_posts(symbol)
    if not posts:
        _CACHE[symbol] = (0.0, now)
        return None

    # Recency-weight: more recent posts get higher weight
    cutoff = now - LOOKBACK_SEC
    total_weight = 0.0
    weighted_sum = 0.0
    for epoch, score in posts:
        # Linear weight from 0 (oldest) to 1 (now)
        age = max(0, now - epoch)
        weight = max(0, 1 - age / LOOKBACK_SEC)
        weighted_sum += score * weight
        total_weight += weight

    final = weighted_sum / total_weight if total_weight > 0 else 0.0
    _CACHE[symbol] = (final, now)
    return final


def explain_sentiment(symbol: str) -> dict:
    """Diagnostic: return per-source counts + final score for one symbol."""
    posts = _load_recent_posts(symbol)
    if not posts:
        return {"symbol": symbol, "n_posts": 0, "score": None}
    scores = [s for _, s in posts]
    return {
        "symbol": symbol,
        "n_posts": len(posts),
        "mean_score": sum(scores) / len(scores),
        "positive_count": sum(1 for s in scores if s > 0),
        "negative_count": sum(1 for s in scores if s < 0),
        "neutral_count": sum(1 for s in scores if s == 0),
        "final_weighted_score": sentiment_score_for_symbol(symbol),
        "blackout_threshold": NEWS_BLACKOUT_THRESHOLD,
    }


if __name__ == "__main__":
    # CLI: python -m scripts.news_sentiment_gate SPY QQQ NVDA
    import sys
    syms = sys.argv[1:] or ["SPY", "QQQ", "NVDA", "AAPL", "META", "TSLA", "AMD"]
    print(f"News sentiment gate · lookback {LOOKBACK_SEC}s · blackout < {NEWS_BLACKOUT_THRESHOLD:+.2f}")
    print("-" * 78)
    for sym in syms:
        info = explain_sentiment(sym)
        if info["n_posts"] == 0:
            print(f"  {sym:6s}  no recent posts")
            continue
        score = info["final_weighted_score"]
        flag = "BLOCK" if score is not None and score < NEWS_BLACKOUT_THRESHOLD else " ok  "
        print(f"  {sym:6s}  n={info['n_posts']:4d}  pos={info['positive_count']:3d}  "
              f"neg={info['negative_count']:3d}  neut={info['neutral_count']:3d}  "
              f"score={score:+.3f}  {flag}")
