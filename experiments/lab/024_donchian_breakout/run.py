"""Donchian breakout — buy on N-bar high, sell on N-bar low. Turtle-trader classical."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from shared.data import load_raw_bars

df = load_raw_bars()
all_trades = []
WIN = 30  # 30-bar Donchian
for sym, g in df.groupby("symbol"):
    g = g.sort_values("t").reset_index(drop=True)
    g["donch_high"] = g["high"].rolling(WIN).max().shift(1)
    g["donch_low"] = g["low"].rolling(WIN).min().shift(1)
    g = g.dropna().reset_index(drop=True)
    in_pos = None
    for i, row in g.iterrows():
        if in_pos is None:
            if row["close"] > row["donch_high"]:
                in_pos = {"side": "long", "entry_px": row["close"], "entry_i": i,
                           "stop": row["donch_low"]}
            elif row["close"] < row["donch_low"]:
                in_pos = {"side": "short", "entry_px": row["close"], "entry_i": i,
                           "stop": row["donch_high"]}
        else:
            # Exit on opposite breakout, stop, or timeout 20 bars
            timeout = i - in_pos["entry_i"] > 20
            stop_hit = (in_pos["side"] == "long" and row["close"] < in_pos["stop"]) or \
                       (in_pos["side"] == "short" and row["close"] > in_pos["stop"])
            if stop_hit or timeout:
                sign = 1 if in_pos["side"] == "long" else -1
                ret_pct = sign * (row["close"] - in_pos["entry_px"]) / in_pos["entry_px"]
                notional = 12000
                all_trades.append(ret_pct * notional - notional * 0.0008)
                in_pos = None

n = len(all_trades)
gross = sum(all_trades)
wins = sum(1 for r in all_trades if r > 0)
out_path = Path(sys.argv[0]).resolve().parent / "results.json"
out_path.write_text(json.dumps({
    "experiment": "024_donchian_breakout",
    "description": "30-bar Donchian breakout, opposite-channel stop, 20-bar timeout",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "n_trades": n, "net_pnl_8bps": float(gross), "win_rate": float(wins/n) if n else 0,
    "passes_overfit_gate": bool(gross > 0 and n >= 10),
    "test_auc": 0,
}, indent=2))
print(f"[024_donchian_breakout]  n_trades={n}  net_pnl_8bps={gross:.2f}  win_rate={wins/n*100 if n else 0:.1f}%")
