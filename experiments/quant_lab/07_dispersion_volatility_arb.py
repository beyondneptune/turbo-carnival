"""Dispersion volatility arbitrage 2024-2026.

Theory: Index variance is a weighted sum of single-name variances +
covariances. If the market PRICES index vol below the implied portfolio of
single-name vols (i.e. assumes high correlation), dispersion exists.

Trade: SHORT index straddle (SPY), LONG basket of single-name straddles.
  - Vega-neutral by construction (matched vega)
  - Profits from realized DISPERSION > implied dispersion
  - Hedged against absolute vol level

Simple version: at each entry/exit pair in our chain data, compare:
  - SPY ATM IV (implied index vol)
  - Average single-name ATM IV (implied basket vol)

If basket IV > SPY IV by enough → dispersion premium exists → trade
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
OUT = REPO / "experiments" / "quant_lab" / "runs"
CHAIN_PATH = REPO / "data" / "final_dataset" / "priced_chain_legs.parquet"


def main():
    print("=" * 70)
    print("Dispersion volatility arbitrage · 2024-2026")
    print("=" * 70)

    chain = pd.read_parquet(CHAIN_PATH)
    atm = chain[chain["is_atm"] == True].copy()
    # For each (event_date, ticker, timing), get ATM IV
    iv = atm.groupby(["event_date", "underlying_ticker", "timing"])["atm_iv"].first().reset_index()

    # Build dispersion analysis: at each event_date, compare SPY IV vs basket IV
    pivot = iv.pivot_table(index=["event_date", "timing"], columns="underlying_ticker",
                            values="atm_iv").reset_index()
    if "SPY" not in pivot.columns:
        print("  no SPY in chain — cannot test dispersion")
        return

    # For each row, compute: SPY_iv, mean(non-SPY non-NaN), dispersion_gap
    non_spy_cols = [c for c in pivot.columns if c not in ("event_date", "timing", "SPY")]
    pivot["basket_iv"] = pivot[non_spy_cols].mean(axis=1, skipna=True)
    pivot["dispersion_gap"] = pivot["basket_iv"] - pivot["SPY"]

    # Stats by timing
    entry = pivot[pivot["timing"] == "entry"]
    exit_ = pivot[pivot["timing"] == "exit"]
    print(f"\nEntry snapshots (pre-event): {len(entry)}")
    print(f"  SPY IV:        mean={entry['SPY'].mean():.3f} median={entry['SPY'].median():.3f}")
    print(f"  basket IV:     mean={entry['basket_iv'].mean():.3f} median={entry['basket_iv'].median():.3f}")
    print(f"  dispersion gap mean={entry['dispersion_gap'].mean():.3f} (basket > SPY IV by this much)")
    print(f"\nExit snapshots (post-event): {len(exit_)}")
    print(f"  SPY IV:        mean={exit_['SPY'].mean():.3f}")
    print(f"  basket IV:     mean={exit_['basket_iv'].mean():.3f}")
    print(f"  dispersion gap mean={exit_['dispersion_gap'].mean():.3f}")

    # Trade signal: enter when entry dispersion_gap > threshold
    # Profit = realized_dispersion - implied_dispersion = entry_gap - exit_gap (roughly)
    # We don't have a clean P&L without straddle prices linked, but we can show
    # the dispersion signal exists.

    # Did the dispersion shrink between entry and exit?
    merged = entry.merge(exit_, on="event_date", suffixes=("_entry", "_exit"))
    merged["gap_change"] = merged["dispersion_gap_entry"] - merged["dispersion_gap_exit"]
    print(f"\nDispersion-gap CHANGE (entry → exit):")
    print(f"  mean: {merged['gap_change'].mean():+.4f}")
    print(f"  median: {merged['gap_change'].median():+.4f}")
    print(f"  positive change events (compression): {(merged['gap_change'] > 0).sum()} / {len(merged)}")
    print(f"  pct of events where dispersion compressed: {(merged['gap_change'] > 0).mean() * 100:.1f}%")

    # Approximate P&L per dollar of vega:
    # If dispersion compresses, short-basket+long-index loses, vice versa
    # We want LONG basket + SHORT index (capture dispersion expansion)
    # P&L per vega = (exit_gap - entry_gap) * 100 (simplified)
    merged["approx_pnl_per_vega"] = merged["dispersion_gap_exit"] - merged["dispersion_gap_entry"]
    approx_total = merged["approx_pnl_per_vega"].sum()
    print(f"\nApprox cumulative dispersion-gap P&L per vega: {approx_total:+.4f}")
    if approx_total > 0:
        print(f"  → going LONG basket + SHORT index averages POSITIVE — dispersion premium captured")
    else:
        print(f"  → dispersion CONTRACTED on average — long-dispersion strategy unprofitable")

    summary = {
        "strategy": "dispersion_volatility_arbitrage",
        "data_note": "limited chain data; analysis is dispersion-signal feasibility, not full P&L",
        "n_events": len(merged),
        "entry_avg_dispersion_gap": float(entry["dispersion_gap"].mean()),
        "exit_avg_dispersion_gap": float(exit_["dispersion_gap"].mean()),
        "entry_basket_iv": float(entry["basket_iv"].mean()),
        "entry_spy_iv": float(entry["SPY"].mean()),
        "approx_total_pnl_per_vega": float(approx_total),
        "pct_events_dispersion_compressed": float((merged["gap_change"] > 0).mean() * 100),
    }
    (OUT / "07_dispersion_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/07_dispersion_summary.json")


if __name__ == "__main__":
    main()
