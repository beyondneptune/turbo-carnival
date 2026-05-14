"""candidate_v3 — Hierarchical Risk Parity on a diversified ETF basket.

Preregistered: specs/candidate_v3_hrp_diversified.md
Source: López de Prado (2016) "Building Diversified Portfolios that Outperform OOS"
"""

UNIVERSE = [
    "SPY",  # US large equity
    "QQQ",  # US growth
    "IWM",  # US small
    "EFA",  # developed ex-US
    "EEM",  # emerging
    "TLT",  # long Treasuries
    "IEF",  # intermediate Treasuries
    "LQD",  # IG corp
    "HYG",  # HY corp
    "GLD",  # gold
    "DBC",  # broad commodities
    "VNQ",  # REITs
]
