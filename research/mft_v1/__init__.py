"""MFTv1 — Opening Range Breakout on liquid US ETFs (5-min bars).

Preregistered: specs/mft_v1_opening_range_breakout.md
Operator-authorized 2026-05-14 after 5/5 daily candidates failed benchmarks.
"""
UNIVERSE = ["SPY", "QQQ", "IWM", "DIA", "MDY"]
BAR_TIMEFRAME = "5Min"
TRAIN_START = "2017-01-01"
TRAIN_END = "2022-12-31"
OOS_START = "2023-01-01"
RTH_OPEN = "09:30"
RTH_CLOSE = "16:00"
FORCE_FLATTEN = "15:55"
