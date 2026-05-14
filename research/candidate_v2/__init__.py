"""candidate_v2 — Sector ETF cointegration pairs strategy.

Preregistered: specs/candidate_v2_sector_etf_cointegration.md
Differs from candidate_v1 in universe ONLY. Same variant grid, same hurdles,
same DSR formula (now fixed), same train/test split.
"""

UNIVERSE = ["XLE", "XLF", "XLK", "XLY", "XLP", "XLI", "XLB", "XLV", "XLU"]
TRAIN_START = "2015-01-01"
TRAIN_END = "2022-12-31"
OOS_START = "2023-01-01"
OOS_END = "2025-12-31"
