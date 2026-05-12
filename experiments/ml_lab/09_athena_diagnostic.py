"""Diagnose which ATHENA filter is too strict for today's regime.

Walks today's 1564 bars (4 allowed symbols × ~390 mins) and counts how many
SURVIVE each filter stage cumulatively. Shows where the funnel kills entries
so we can rebalance thresholds for v1.1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path("/Users/weeoto/turbo-carnival")
sys.path.insert(0, str(REPO / "scripts"))
from athena_ml_gates import ae_decision, logreg_p_volatile  # noqa


def load_env():
    ENV = REPO / "council" / ".env.local"
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def fetch():
    load_env()
    H = {"APCA-API-KEY-ID": os.environ["ALPACA_AEGIS_KEY"],
         "APCA-API-SECRET-KEY": os.environ["ALPACA_AEGIS_SECRET"]}
    rows = []
    for sym in ("SPY", "QQQ", "AAPL", "NVDA"):
        r = requests.get("https://data.alpaca.markets/v2/stocks/bars",
                         headers=H,
                         params={"symbols": sym, "timeframe": "1Min",
                                  "start": "2026-05-11T13:30:00Z",
                                  "end": "2026-05-11T20:00:00Z",
                                  "limit": 10000, "adjustment": "raw"},
                         timeout=20)
        for b in r.json().get("bars", {}).get(sym, []):
            rows.append({"symbol": sym, "t": b["t"], "o": b["o"], "h": b["h"],
                          "l": b["l"], "close": b["c"], "v": b["v"]})
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["t_ny"] = df["t"].dt.tz_convert("America/New_York")
    df["date"] = df["t"].dt.date
    df = df.sort_values(["symbol", "t"]).reset_index(drop=True)
    df["ret1"] = df.groupby("symbol")["close"].pct_change(1).fillna(0)
    df["ret3"] = df.groupby("symbol")["close"].pct_change(3).fillna(0)
    df["ret5"] = df.groupby("symbol")["close"].pct_change(5).fillna(0)
    df["ret10"] = df.groupby("symbol")["close"].pct_change(10).fillna(0)
    df["range_pct"] = (df["h"] - df["l"]) / df["close"]
    df["volume_z"] = df.groupby("symbol")["v"].transform(
        lambda s: (s - s.rolling(20, min_periods=5).mean()) / s.rolling(20, min_periods=5).std()
    ).fillna(0)
    df["ret1_z"] = df.groupby("symbol")["ret1"].transform(
        lambda s: (s - s.rolling(20, min_periods=5).mean()) / s.rolling(20, min_periods=5).std()
    ).fillna(0)
    df["ret5_z"] = df.groupby("symbol")["ret5"].transform(
        lambda s: (s - s.rolling(20, min_periods=5).mean()) / s.rolling(20, min_periods=5).std()
    ).fillna(0)
    df["session_vwap"] = df.groupby(["symbol", "date"]).apply(
        lambda g: (g["close"] * g["v"]).cumsum() / g["v"].cumsum().replace(0, np.nan),
        include_groups=False,
    ).reset_index(level=[0, 1], drop=True)
    df["vwap_dist"] = (df["close"] - df["session_vwap"]) / df["session_vwap"]
    mins = ((df["t_ny"].dt.hour - 9) * 60 + df["t_ny"].dt.minute - 30).clip(0, 390)
    df["tod_sin"] = np.sin(2 * np.pi * mins / 390)
    df["tod_cos"] = np.cos(2 * np.pi * mins / 390)

    def vov(g):
        s = g["ret1"].rolling(5).std()
        m = g["ret1"].abs().rolling(20).median()
        return (s / m).fillna(0).clip(0, 50)
    df["vov"] = df.groupby("symbol", group_keys=False).apply(vov, include_groups=False)
    df["range_pct_p60"] = df.groupby("symbol")["range_pct"].transform(
        lambda s: s.rolling(60, min_periods=10).quantile(0.6)
    )
    return df


def in_window(ts_ny: pd.Timestamp) -> bool:
    h, m = ts_ny.hour, ts_ny.minute
    return 10 * 60 <= h * 60 + m <= 15 * 60 + 30  # 10:00 ET - 15:30 ET inclusive


def main() -> int:
    df = fetch()
    print(f"Today's bars across 4 symbols: {len(df):,}\n")

    LR_FEATS = ["ret1", "ret3", "ret10", "range_pct", "volume_z", "ret1_z",
                "vwap_dist", "tod_sin", "tod_cos", "vov"]

    # Stage-by-stage funnel — Lane-1 (Aegis mean-rev)
    print("=" * 70)
    print("LANE-1 (Aegis-α mean-rev) funnel — bars surviving each stage:")
    print("=" * 70)

    stage = df.copy()
    print(f"  Stage 0: all bars                      {len(stage):>6,}")

    stage = stage[stage["symbol"].isin(["SPY", "QQQ", "AAPL"])]
    print(f"  Stage 1: symbol in {{SPY,QQQ,AAPL}}      {len(stage):>6,}")

    stage = stage[stage["t_ny"].apply(in_window)]
    print(f"  Stage 2: 10:00-15:30 ET                 {len(stage):>6,}")

    stage = stage[abs(stage["ret1"]) * 10000 < 60]
    print(f"  Stage 3: |ret1| < 60 bps                {len(stage):>6,}")

    # LogReg
    p_vols = []
    for _, r in stage.iterrows():
        feats = {k: float(r[k]) for k in LR_FEATS}
        p_vols.append(logreg_p_volatile(feats) or 0)
    stage = stage.copy()
    stage["p_vol"] = p_vols
    print(f"      LogReg p_vol distribution: "
          f"min={stage['p_vol'].min():.3f} med={stage['p_vol'].median():.3f} "
          f"max={stage['p_vol'].max():.3f} p25={stage['p_vol'].quantile(0.25):.3f}")
    stage_lr = stage[stage["p_vol"] < 0.50]
    print(f"  Stage 4: LogReg p_vol < 0.50            {len(stage_lr):>6,}")
    stage_lr_65 = stage[stage["p_vol"] < 0.65]
    print(f"           (or < 0.65 — v1.1 candidate)   {len(stage_lr_65):>6,}")

    # Signal strength
    stage = stage_lr
    stage_z18 = stage[abs(stage["ret1_z"]) > 1.8]
    stage_z14 = stage[abs(stage["ret1_z"]) > 1.4]
    print(f"  Stage 5: |ret1_z| > 1.8 (v1)            {len(stage_z18):>6,}")
    print(f"           |ret1_z| > 1.4 (v1.1)          {len(stage_z14):>6,}")

    # VWAP filter
    final_v1 = stage_z18[abs(stage_z18["vwap_dist"]) * 10000 > 8]
    final_v11 = stage_z14[abs(stage_z14["vwap_dist"]) * 10000 > 5]
    print(f"  Stage 6: |vwap_dist| > 8 bps (v1)       {len(final_v1):>6,}")
    print(f"           |vwap_dist| > 5 bps (v1.1)     {len(final_v11):>6,}")

    print(f"\n  v1 FINAL Lane-1 entries:  {len(final_v1):>6,}")
    print(f"  v1.1 FINAL Lane-1 entries: {len(final_v11):>6,}")
    if len(final_v11) > 0:
        print(f"  v1.1 LogReg p_vol on surviving entries: "
              f"med={final_v11['p_vol'].median():.3f}")

    # ============================================
    print()
    print("=" * 70)
    print("LANE-2 (Falcon-α momentum) funnel:")
    print("=" * 70)

    stage = df.copy()
    print(f"  Stage 0: all bars                      {len(stage):>6,}")
    stage = stage[stage["symbol"].isin(["SPY", "QQQ", "NVDA"])]
    print(f"  Stage 1: symbol in {{SPY,QQQ,NVDA}}      {len(stage):>6,}")
    stage = stage[stage["t_ny"].apply(in_window)]
    print(f"  Stage 2: 10:00-15:30 ET                 {len(stage):>6,}")
    stage = stage[abs(stage["ret1"]) * 10000 < 80]
    print(f"  Stage 3: |ret1| < 80 bps                {len(stage):>6,}")

    p_vols = []
    for _, r in stage.iterrows():
        feats = {k: float(r[k]) for k in LR_FEATS}
        p_vols.append(logreg_p_volatile(feats) or 0)
    stage = stage.copy()
    stage["p_vol"] = p_vols
    stage_lr = stage[stage["p_vol"] < 0.50]
    print(f"  Stage 4: LogReg p_vol < 0.50            {len(stage_lr):>6,}")
    stage_lr_65 = stage[stage["p_vol"] < 0.65]
    print(f"           (or < 0.65 — v1.1 candidate)   {len(stage_lr_65):>6,}")

    stage = stage_lr
    stage_z14 = stage[abs(stage["ret5_z"]) > 1.4]
    stage_z10 = stage[abs(stage["ret5_z"]) > 1.0]
    print(f"  Stage 5: |ret5_z| > 1.4 (v1)            {len(stage_z14):>6,}")
    print(f"           |ret5_z| > 1.0 (v1.1)          {len(stage_z10):>6,}")

    final_v1 = stage_z14[stage_z14["range_pct"] > stage_z14["range_pct_p60"]]
    print(f"  Stage 6: range_pct > p60 (v1)           {len(final_v1):>6,}")
    final_v11 = stage_z10[stage_z10["range_pct"] > stage_z10["range_pct"].quantile(0.5)]
    print(f"           range_pct > p50 (v1.1)         {len(final_v11):>6,}")

    print(f"\n  v1 FINAL Lane-2 entries:  {len(final_v1):>6,}")
    print(f"  v1.1 FINAL Lane-2 entries: {len(final_v11):>6,}")

    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    print("""
The LogReg gate at p_vol < 0.50 is heavily restrictive today.
Today's median p_vol may be elevated, suggesting the model SEES today's
regime as more volatile than 21-day average. This is the model doing its
job — but a 'block everything' day is operationally useless.

ATHENA v1.1 proposes:
  - LogReg threshold: 0.50 → 0.65 (let p_vol up to 0.65 pass)
  - ret1_z threshold (L1): 1.8 → 1.4
  - vwap_dist (L1): 8 bps → 5 bps
  - ret5_z (L2): 1.4 → 1.0
  - range_pct (L2): p60 → p50

This keeps the ATHENA architecture (gates + sizing + symbol drop) but
relaxes thresholds to produce ~15-25 entries/lane/day, which is the
target. Re-run 08_athena_v1_signal_scanner.py with v1.1 thresholds.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
