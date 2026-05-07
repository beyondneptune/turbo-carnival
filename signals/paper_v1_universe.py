"""Locked Top-50 Russell 1000 universe for paper_v1 satellite sleeve.

Snapshot list, not a live feed — frozen at prereg authoring time. The CEO
accepts this as the cycle 1 universe. Survivorship bias acknowledged."""
from __future__ import annotations

import pandas as pd


PAPER_V1_CORE: tuple[str, ...] = ("SPY",)

# 50 US-listed names with liquid weekly options as of 2026-05-06.
# Pick mega-caps and high-options-ADV names: AAPL, MSFT, NVDA, GOOGL, AMZN,
# META, TSLA, JPM, V, MA, BAC, WMT, UNH, JNJ, PG, XOM, CVX, HD, KO, PEP,
# DIS, NFLX, AMD, INTC, CRM, ORCL, CSCO, ADBE, AVGO, QCOM, COST, ABBV,
# PFE, MRK, LLY, BA, GS, MS, WFC, BX, COIN, SQ (Block), PLTR, SHOP,
# UBER, AMAT, MU, TXN, GE, F. (50 total — verify count.)
PAPER_V1_SATELLITE: tuple[str, ...] = (
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN",
    "META", "TSLA", "JPM", "V", "MA",
    "BAC", "WMT", "UNH", "JNJ", "PG",
    "XOM", "CVX", "HD", "KO", "PEP",
    "DIS", "NFLX", "AMD", "INTC", "CRM",
    "ORCL", "CSCO", "ADBE", "AVGO", "QCOM",
    "COST", "ABBV", "PFE", "MRK", "LLY",
    "BA", "GS", "MS", "WFC", "BX",
    "COIN", "SQ", "PLTR", "SHOP", "UBER",
    "AMAT", "MU", "TXN", "GE", "F",
)


def get_universe_candidates() -> pd.DataFrame:
    """Returns the candidates DataFrame in the format consumed by
    `signals/earnings_universe.py::apply_liquidity_filters`.

    Synthetic but realistic shaped values (ADV ≈ 50_000, OI ≈ 200_000,
    spread ≈ 4-6%, events ≈ 12) so all 50 names PASS the liquidity gates
    deterministically. This is a placeholder for paper_v1; a real Year-1
    backfill would replace it with measured Polygon data."""
    n = len(PAPER_V1_SATELLITE)
    # Deterministic per-row variation so frame is non-degenerate but every row
    # comfortably clears the gates in `UniverseFilters` (defaults: ADV>=5_000,
    # OI>=10_000, spread<=0.08, events>=8). Spreads stay in [0.04, 0.06],
    # events stay at 12, ADV/OI scale up slightly with index.
    return pd.DataFrame(
        {
            "ticker": list(PAPER_V1_SATELLITE),
            "options_adv_30d": [50_000 + i * 100 for i in range(n)],
            "options_oi_30d": [200_000 + i * 500 for i in range(n)],
            "atm_spread_pct_30d": [0.04 + (i % 21) * 0.001 for i in range(n)],
            "earnings_events_count": [12] * n,
        }
    )
