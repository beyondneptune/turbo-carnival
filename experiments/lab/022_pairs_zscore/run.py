"""Pairs trading on SPY/QQQ — z-score of log-ratio, entry at |z|>2, exit at |z|<0.5."""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from shared.data import load_raw_bars

df = load_raw_bars()
spy = df[df["symbol"] == "SPY"].sort_values("t").reset_index(drop=True)
qqq = df[df["symbol"] == "QQQ"].sort_values("t").reset_index(drop=True)
merged = pd.merge(spy[["t", "close"]], qqq[["t", "close"]], on="t",
                   suffixes=("_spy", "_qqq")).reset_index(drop=True)
merged["log_ratio"] = np.log(merged["close_spy"] / merged["close_qqq"])
merged["z"] = (merged["log_ratio"] - merged["log_ratio"].rolling(60).mean()) / \
              merged["log_ratio"].rolling(60).std()
merged = merged.dropna().reset_index(drop=True)

trades = []
in_pos = None
for i, row in merged.iterrows():
    z = row["z"]
    if in_pos is None:
        if z > 2.0:
            in_pos = {"side": "short_spread", "entry_i": i, "entry_z": z,
                       "entry_spy": row["close_spy"], "entry_qqq": row["close_qqq"]}
        elif z < -2.0:
            in_pos = {"side": "long_spread", "entry_i": i, "entry_z": z,
                       "entry_spy": row["close_spy"], "entry_qqq": row["close_qqq"]}
    else:
        if abs(z) < 0.5 or i - in_pos["entry_i"] > 75:
            sign = -1 if in_pos["side"] == "short_spread" else 1
            spy_ret = (row["close_spy"] - in_pos["entry_spy"]) / in_pos["entry_spy"]
            qqq_ret = (row["close_qqq"] - in_pos["entry_qqq"]) / in_pos["entry_qqq"]
            # Spread P&L = long-leg return - short-leg return
            pnl_pct = sign * (spy_ret - qqq_ret)
            notional = 15000  # per leg
            gross_usd = pnl_pct * notional
            slip = 2 * notional * 0.0008  # 8 bps × 2 legs
            trades.append({"net": gross_usd - slip, "hold_bars": i - in_pos["entry_i"]})
            in_pos = None

n = len(trades)
gross = sum(t["net"] for t in trades) if trades else 0
wins = sum(1 for t in trades if t["net"] > 0)
results = {
    "n_trades": n,
    "net_pnl_8bps": float(gross),
    "win_rate": float(wins / n) if n else 0,
    "avg_hold_bars": float(np.mean([t["hold_bars"] for t in trades])) if trades else 0,
    "passes_overfit_gate": bool(gross > 0 and n >= 5),
    "test_auc": 0,  # not applicable
}
out_path = Path(sys.argv[0]).resolve().parent / "results.json"
out_path.write_text(json.dumps({
    "experiment": "022_pairs_zscore",
    "description": "SPY/QQQ z-score pairs, |z|>2 entry, |z|<0.5 exit, 60-bar lookback",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    **results}, indent=2))
print(f"[022_pairs_zscore]  n_trades={n}  net_pnl_8bps={gross:.2f}  win_rate={wins/n*100 if n else 0:.1f}%")
