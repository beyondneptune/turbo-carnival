"""Order book imbalance signal for ATHENA v2.1.

Alpaca's /v2/stocks/quotes/latest returns bid_size and ask_size on the
inside quote (L1). Imbalance is:

    imbalance = (bid_size - ask_size) / (bid_size + ask_size)

Range [-1, 1]:
  +1.0  → all liquidity on bid side (strong buy pressure short-term)
   0.0  → balanced
  -1.0  → all liquidity on ask side (strong sell pressure short-term)

Empirical: imbalance > 0.3 predicts +0.5-2 bps short-term return over 30s.
This is well-documented and survives at single-name level.

Wiring into latest_quotes() in alpaca_max_hft_host.py (around line 319-336)
will populate quote["imbalance"]; then use it as a feature in C3 or as a
direct gate.

This module exports two helpers:
  - extract_imbalance(raw_quote)        → float in [-1, 1]
  - imbalance_blocks(quote, direction)  → True if entry direction is opposite of imbalance
"""
from __future__ import annotations

import os
from typing import Optional

# Block entry if imbalance opposes our direction by more than this magnitude
IMBALANCE_BLOCK_THRESHOLD = float(os.environ.get("ATHENA_IMBALANCE_BLOCK", "0.30"))


def extract_imbalance(raw_quote: dict) -> Optional[float]:
    """Compute imbalance from raw Alpaca quote dict.

    Alpaca's latest quote response has fields:
      bp (bid price), bs (bid size), ap (ask price), as (ask size)
    """
    bs = raw_quote.get("bs") or raw_quote.get("bid_size") or 0
    asz = raw_quote.get("as") or raw_quote.get("ask_size") or 0
    try:
        bs = float(bs)
        asz = float(asz)
    except (TypeError, ValueError):
        return None
    total = bs + asz
    if total <= 0:
        return None
    return (bs - asz) / total


def imbalance_blocks(quote: dict, direction: int) -> tuple[bool, str]:
    """Return (block, reason) for a proposed entry.

    direction = +1 (long) or -1 (short).
    Blocks if imbalance opposes direction by more than threshold.
    """
    imb = quote.get("imbalance")
    if imb is None:
        return False, "no_imbalance"
    # Long entry blocked if heavy ask-side pressure (imb < -threshold)
    if direction > 0 and imb < -IMBALANCE_BLOCK_THRESHOLD:
        return True, f"ob_imbalance_short_pressure({imb:+.2f})"
    # Short entry blocked if heavy bid-side pressure (imb > +threshold)
    if direction < 0 and imb > IMBALANCE_BLOCK_THRESHOLD:
        return True, f"ob_imbalance_long_pressure({imb:+.2f})"
    return False, "ok"


def imbalance_as_feature(quote: dict) -> float:
    """Return imbalance as a float feature for ML gate (0.0 if missing)."""
    imb = quote.get("imbalance")
    if imb is None:
        return 0.0
    return float(imb)


# --- Patch for latest_quotes() ---
# Replace the existing build of quote dict in scripts/alpaca_max_hft_host.py
# around lines 327-340 with code that also stores bid_size, ask_size, imbalance:
#
#     bid = float(item.get("bp") or 0)
#     ask = float(item.get("ap") or 0)
#     bid_size = float(item.get("bs") or 0)
#     ask_size = float(item.get("as") or 0)
#     mid = (bid + ask) / 2 if bid and ask else 0
#     imbalance = (bid_size - ask_size) / (bid_size + ask_size) if (bid_size + ask_size) > 0 else 0
#     out[sym] = {
#         "bid": bid, "ask": ask,
#         "bid_size": bid_size, "ask_size": ask_size,
#         "mid": mid,
#         "spread_bps": (ask - bid) / mid * 10000.0 if mid else 9999.0,
#         "imbalance": imbalance,
#     }


if __name__ == "__main__":
    # Demo
    sample_quotes = [
        {"bp": 100, "ap": 100.05, "bs": 800, "as": 200},   # strong bid side
        {"bp": 100, "ap": 100.05, "bs": 200, "as": 800},   # strong ask side
        {"bp": 100, "ap": 100.05, "bs": 500, "as": 500},   # balanced
    ]
    for q in sample_quotes:
        imb = extract_imbalance(q)
        long_block, long_reason = imbalance_blocks({**q, "imbalance": imb}, +1)
        short_block, short_reason = imbalance_blocks({**q, "imbalance": imb}, -1)
        print(f"  bs={q['bs']}  as={q['as']}  imb={imb:+.3f}  "
              f"long_block={long_block} ({long_reason})  short_block={short_block} ({short_reason})")
