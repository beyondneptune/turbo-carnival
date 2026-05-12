"""Bollinger Band mean-reversion — buy 2σ below SMA, sell 2σ above. Classical strategy."""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from shared.data import load_raw_bars

df = load_raw_bars()
results_all = []
for sym, g in df.groupby("symbol"):
    g = g.sort_values("t").reset_index(drop=True)
    g["sma20"] = g["close"].rolling(20).mean()
    g["sd20"] = g["close"].rolling(20).std()
    g["upper"] = g["sma20"] + 2 * g["sd20"]
    g["lower"] = g["sma20"] - 2 * g["sd20"]
    g = g.dropna().reset_index(drop=True)
    trades = []
    in_pos = None
    for i, row in g.iterrows():
        if in_pos is None:
            if row["close"] < row["lower"]:
                in_pos = {"side": "long", "entry_px": row["close"], "entry_i": i}
            elif row["close"] > row["upper"]:
                in_pos = {"side": "short", "entry_px": row["close"], "entry_i": i}
        else:
            # Exit when price crosses SMA back
            mean_cross = (in_pos["side"] == "long" and row["close"] >= row["sma20"]) or \
                         (in_pos["side"] == "short" and row["close"] <= row["sma20"])
            time_out = i - in_pos["entry_i"] > 20
            if mean_cross or time_out:
                sign = 1 if in_pos["side"] == "long" else -1
                ret_pct = sign * (row["close"] - in_pos["entry_px"]) / in_pos["entry_px"]
                notional = 12000
                trades.append(ret_pct * notional - notional * 0.0008)  # 8 bps slip
                in_pos = None
    results_all.extend(trades)

n = len(results_all)
gross = sum(results_all)
wins = sum(1 for r in results_all if r > 0)
out_path = Path(sys.argv[0]).resolve().parent / "results.json"
out_path.write_text(json.dumps({
    "experiment": "023_bollinger_meanrev",
    "description": "Bollinger 2σ mean-rev, entry at band touch, exit at SMA cross or 20-bar timeout",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "n_trades": n, "net_pnl_8bps": float(gross), "win_rate": float(wins/n) if n else 0,
    "passes_overfit_gate": bool(gross > 0 and n >= 10),
    "test_auc": 0,
}, indent=2))
print(f"[023_bollinger_meanrev]  n_trades={n}  net_pnl_8bps={gross:.2f}  win_rate={wins/n*100 if n else 0:.1f}%")
