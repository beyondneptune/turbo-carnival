"""Prepare the 162 yesterday-entries context that 100 Haiku voters will see.

Writes:
  /tmp/haiku_voting_entries.md — the entries in a compact, readable form
  /tmp/haiku_voting_meta.json — entry metadata (for later aggregation join)

Each Haiku agent will read /tmp/haiku_voting_entries.md and output a JSON
list of 162 yes/no decisions.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path("/Users/weeoto/turbo-carnival")
ZOO_PER_ENTRY = REPO / "experiments" / "ml_lab" / "runs" / "ml_zoo_per_entry_latest.parquet"
VOTE_DIR = REPO / "experiments" / "ml_lab" / "haiku_vote_runs"
VOTE_DIR.mkdir(parents=True, exist_ok=True)

def load_env():
    for line in (REPO / "council" / ".env.local").read_text().splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip().strip('"').strip("'")

def fetch_yesterday_bars():
    load_env()
    H = {"APCA-API-KEY-ID": os.environ["ALPACA_AEGIS_KEY"],
         "APCA-API-SECRET-KEY": os.environ["ALPACA_AEGIS_SECRET"]}
    rows = []
    for sym in ("SPY", "QQQ", "AAPL", "AMD", "META", "NVDA", "TSLA"):
        r = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=H,
                         params={"symbols": sym, "timeframe": "1Min",
                                  "start": "2026-05-11T13:30:00Z",
                                  "end": "2026-05-11T20:00:00Z",
                                  "limit": 10000, "adjustment": "raw"}, timeout=20)
        for b in r.json().get("bars", {}).get(sym, []):
            rows.append({"symbol": sym, "t": b["t"], "open": b["o"], "high": b["h"],
                          "low": b["l"], "close": b["c"], "volume": b["v"]})
    return pd.DataFrame(rows).assign(t=lambda d: pd.to_datetime(d["t"], utc=True))

# Load yesterday's matched entries
per_entry = pd.read_parquet(ZOO_PER_ENTRY)
one_model = per_entry["model"].iloc[0]
entries = per_entry[per_entry["model"] == one_model].copy().reset_index(drop=True)
entries["entry_ts"] = pd.to_datetime(entries["entry_ts"], utc=True)
entries["entry_minute"] = entries["entry_ts"].dt.floor("1min")
print(f"loaded {len(entries)} entries")

# Fetch fresh bars for context (vwap, range, vol z-score)
yest = fetch_yesterday_bars()
yest["t_ny"] = yest["t"].dt.tz_convert("America/New_York")
yest = yest.sort_values(["symbol", "t"]).reset_index(drop=True)
yest["ret1"] = yest.groupby("symbol")["close"].pct_change(1).fillna(0)
yest["range_pct"] = (yest["high"] - yest["low"]) / yest["close"]
yest["volume_z"] = yest.groupby("symbol")["volume"].transform(
    lambda s: (s - s.rolling(20, min_periods=5).mean()) / s.rolling(20, min_periods=5).std()
).fillna(0)
yest["ret1_z"] = yest.groupby("symbol")["ret1"].transform(
    lambda s: (s - s.rolling(20, min_periods=5).mean()) / s.rolling(20, min_periods=5).std()
).fillna(0)
yest["date"] = yest["t"].dt.date
yest["session_vwap"] = yest.groupby(["symbol", "date"]).apply(
    lambda g: (g["close"] * g["volume"]).cumsum() / g["volume"].cumsum().replace(0, np.nan),
    include_groups=False,
).reset_index(level=[0, 1], drop=True)
yest["vwap_dist_bps"] = (yest["close"] - yest["session_vwap"]) / yest["session_vwap"] * 10000
yest["minute"] = yest["t"].dt.floor("1min")
ctx = yest.set_index(["symbol", "minute"])

# Build per-entry feature row
lines = []
meta = []
for i, e in entries.iterrows():
    try:
        c = ctx.loc[(e["symbol"], e["entry_minute"])]
        if isinstance(c, pd.DataFrame): c = c.iloc[0]
        ny = pd.Timestamp(e["entry_ts"]).tz_convert("America/New_York")
        line = (
            f"i={i:>3}  {ny.strftime('%H:%M')} ET  {e['symbol']:>4} {e['side_dir']:>5}  "
            f"lane={e['lane']:>6}  "
            f"ret1z={c['ret1_z']:>+5.2f}  vwap_dist={c['vwap_dist_bps']:>+6.1f}bps  "
            f"range_pct={c['range_pct']*10000:>5.1f}bps  vol_z={c['volume_z']:>+4.2f}"
        )
        lines.append(line)
        meta.append({"i": int(i), "lane": e["lane"], "symbol": e["symbol"],
                      "side_dir": e["side_dir"], "entry_ts": str(e["entry_ts"]),
                      "exit_ts": str(e["exit_ts"]), "exit_action": e["exit_action"],
                      "pnl": float(e["pnl"]), "qty": int(e["qty"])})
    except (KeyError, ValueError) as ex:
        lines.append(f"i={i:>3}  [bar data missing — vote NO]")
        meta.append({"i": int(i), "skip": True, "pnl": float(e["pnl"])})

# Write the entries context
ctx_md = f"""# Yesterday's 162 paper-trading entries — vote YES (allow) or NO (block)

You are evaluating 162 entries from yesterday (2026-05-11) on a paper-trading
HFT system. The system has TWO HFT lanes:
  - Aegis: mean-reversion (expects high z-score ret1 to revert to VWAP)
  - Falcon: momentum/breakout (expects high ret5_z to continue)

For each entry, vote YES (would have allowed this trade) or NO (would have
blocked it). Apply a trader's judgment. Examples of when to vote NO:
  - Very wide range_pct (microstructure noise dominating)
  - Extreme ret1_z combined with extreme vol_z (likely news-driven, hard to reverse)
  - Far from VWAP in the WRONG direction for the strategy (e.g. Aegis trying mean-rev
    when the bar is moving with the trend rather than away from it)
  - Open auction (9:30-10:00 ET) or close auction (15:30-16:00 ET) — wider spreads
  - Falcon entering when range_pct is below median (no breakout to ride)

Examples of when to vote YES:
  - Mid-session (10:30-15:00 ET) with moderate ret1_z (1-2.5)
  - Vwap_dist consistent with the side (Aegis short when vwap_dist > 0, mean-rev down)
  - Tight range_pct (normal vol)

Per-entry data (lane = which strategy was firing, side = long/short):

```
{chr(10).join(lines)}
```

## Output format

Write your decisions as a JSON list of 162 objects to your assigned slot file.
Each object: `{{"i": <0-161>, "vote": "YES" or "NO"}}`. Order doesn't matter.

NO reasoning text in the JSON — keep output tight.

Example output:
```json
[
  {{"i": 0, "vote": "YES"}},
  {{"i": 1, "vote": "NO"}},
  ...
]
```

Aim for a 40-70% YES rate. If you vote YES on everything or NO on everything,
your output is useless — be discriminating based on the per-entry data.
"""

Path("/tmp/haiku_voting_entries.md").write_text(ctx_md)
Path("/tmp/haiku_voting_meta.json").write_text(json.dumps(meta, indent=2))
print(f"wrote /tmp/haiku_voting_entries.md ({len(ctx_md):,} chars)")
print(f"wrote /tmp/haiku_voting_meta.json ({len(meta)} entries)")
print(f"\n--- preview ---\n{ctx_md[:1500]}\n...")
