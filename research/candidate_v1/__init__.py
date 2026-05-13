"""candidate_v1 — Cointegration commodity ETF pairs strategy.

Preregistered: specs/candidate_v1_cointegration_pairs.md
Frozen 2026-05-13. Variant universe, hurdles, and train/test split are immutable
per §16 of the preregistration.
"""

UNIVERSE = [
    "USO",   # oil
    "UNG",   # natgas
    "GLD",   # gold
    "SLV",   # silver
    "DBA",   # broad agriculture
    "DBC",   # broad commodity
    "CORN",  # corn
    "WEAT",  # wheat
    "CPER",  # copper (replaces delisted JJC)
    "PPLT",  # platinum (replaces delisted JJN; covers industrial-metal slot)
]

TRAIN_START = "2015-01-01"
TRAIN_END = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END = "2025-12-31"

VARIANT_UNIVERSE = {
    "WINDOW": [20, 30, 45, 60, 90],
    "ENTRY_THRESHOLD": [1.5, 2.0, 2.5],
    "STOP_THRESHOLD": [3.0, 4.0, 5.0],
    "EXIT_THRESHOLD": [0.0, 0.5],
    "COINT_PVALUE_THRESHOLD": [0.01, 0.05, 0.10],
    "MAX_HOLD": [30, 60, 90],
}
TOTAL_VARIANTS = 5 * 3 * 3 * 2 * 3 * 3  # 810
