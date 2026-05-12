"""VPIN (Volume-synchronized Probability of Informed trading) — Easley/Lopez de Prado/O'Hara.

Compute VPIN on the 21-day minute-bar cache for 7 symbols, then test:
  Does high VPIN predict near-term unfavorable price moves?

Method:
  1. Bucket bars into VOLUME-time buckets (V_bucket = avg daily volume / 50)
  2. Within each bucket, classify volume as BUY or SELL via tick rule
     (close > prev close = buy, < = sell, == = split 50/50)
  3. VPIN = sum_buckets |V_buy - V_sell| / (V_buy + V_sell) over rolling 50 buckets
  4. Correlate VPIN at time t with realized |return| in next 5 bars

If VPIN is informative, high VPIN should correlate with high future |return|
(informed traders moved the market) — and trading against high VPIN is dangerous.

Test: is the Pearson correlation r > 0.1? Then VPIN gives signal.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
OUT = REPO / "experiments" / "quant_lab" / "runs"
BARS = REPO / "experiments" / "runs" / "valk_hft_demo_20260511T140548Z" / "minute_bars.parquet"
BUCKET_VOL = None   # set per symbol below
ROLLING_BUCKETS = 50
FORWARD_BARS = 5


def compute_vpin(df: pd.DataFrame, bucket_vol: float) -> pd.DataFrame:
    """Bucket bars by cumulative volume, compute VPIN per bucket."""
    df = df.copy().sort_values("t").reset_index(drop=True)
    # Tick-rule classification: price up = buy, down = sell, flat = split
    df["dp"] = df["close"].diff().fillna(0)
    df["buy_vol"] = np.where(df["dp"] > 0, df["volume"],
                              np.where(df["dp"] < 0, 0, df["volume"] / 2))
    df["sell_vol"] = df["volume"] - df["buy_vol"]
    # Cumulative volume — break into buckets
    df["cum_vol"] = df["volume"].cumsum()
    df["bucket"] = (df["cum_vol"] // bucket_vol).astype(int)

    # Per-bucket: total buy / sell
    buckets = df.groupby("bucket").agg(
        buy=("buy_vol", "sum"), sell=("sell_vol", "sum"),
        last_t=("t", "last"), close=("close", "last"),
    ).reset_index()
    buckets["imbalance"] = abs(buckets["buy"] - buckets["sell"]) / (buckets["buy"] + buckets["sell"])
    # VPIN = rolling mean of imbalance over ROLLING_BUCKETS
    buckets["vpin"] = buckets["imbalance"].rolling(ROLLING_BUCKETS).mean()
    return buckets


def main():
    print("=" * 70)
    print("VPIN order-flow toxicity · 21-day minute cache · 7 symbols")
    print("=" * 70)

    if not BARS.exists():
        print(f"  no bars: {BARS}")
        return
    bars = pd.read_parquet(BARS)
    bars["t"] = pd.to_datetime(bars["t"], utc=True)
    print(f"\n{len(bars):,} bars total")

    results = []
    for sym in sorted(bars["symbol"].unique()):
        sub = bars[bars["symbol"] == sym].sort_values("t").reset_index(drop=True)
        if len(sub) < ROLLING_BUCKETS * 50:
            continue
        # Bucket size: total volume / (n_bars / 50)
        avg_daily_vol = sub["volume"].sum() / 21  # 21 days
        bucket_vol = avg_daily_vol / 50
        b = compute_vpin(sub, bucket_vol)
        b = b.dropna().reset_index(drop=True)
        if len(b) < 100:
            continue
        # Compute forward |return| from bucket close
        b["fwd_abs_ret_bps"] = (b["close"].shift(-FORWARD_BARS) / b["close"] - 1).abs() * 10000
        b = b.dropna().reset_index(drop=True)
        # Correlation
        corr = b["vpin"].corr(b["fwd_abs_ret_bps"])
        # Top-decile VPIN vs bottom-decile: average forward |return|
        cut_lo, cut_hi = b["vpin"].quantile([0.1, 0.9])
        avg_lo = b.loc[b["vpin"] <= cut_lo, "fwd_abs_ret_bps"].mean()
        avg_hi = b.loc[b["vpin"] >= cut_hi, "fwd_abs_ret_bps"].mean()
        results.append({
            "symbol": sym,
            "n_buckets": int(len(b)),
            "vpin_fwd_corr": float(corr),
            "avg_fwd_abs_ret_lo_vpin_bps": float(avg_lo),
            "avg_fwd_abs_ret_hi_vpin_bps": float(avg_hi),
            "hi_lo_ratio": float(avg_hi / avg_lo) if avg_lo > 0 else 0,
        })
        print(f"  {sym}: n={len(b):>4}  corr={corr:+.3f}  "
              f"avg|ret|_lo_VPIN={avg_lo:.1f}bps  avg|ret|_hi_VPIN={avg_hi:.1f}bps  "
              f"hi/lo={avg_hi/avg_lo:.2f}x" if avg_lo > 0 else "")

    print(f"\n=== Result interpretation ===")
    avg_corr = np.mean([r["vpin_fwd_corr"] for r in results])
    avg_ratio = np.mean([r["hi_lo_ratio"] for r in results if r["hi_lo_ratio"] > 0])
    print(f"Average VPIN→|future ret| correlation across symbols: {avg_corr:+.3f}")
    print(f"Average high-VPIN to low-VPIN forward |return| ratio: {avg_ratio:.2f}x")
    if avg_corr > 0.1 and avg_ratio > 1.3:
        print("VPIN APPEARS PREDICTIVE — using it as a regime filter is justified")
    elif avg_corr > 0.05:
        print("Weak signal — VPIN has some predictive value but small")
    else:
        print("VPIN is NOT predictive on this dataset — likely tick-rule classification too noisy at minute granularity")

    summary = {"strategy": "vpin_order_flow_toxicity",
               "data_limitation": "21 days minute bars, tick-rule classification",
               "per_symbol": results,
               "avg_correlation": avg_corr, "avg_hi_lo_ratio": avg_ratio}
    (OUT / "05_vpin_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/05_vpin_summary.json")


if __name__ == "__main__":
    main()
