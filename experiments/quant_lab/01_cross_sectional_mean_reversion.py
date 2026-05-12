"""Cross-sectional mean reversion (CSMR) backtest 2019-2026.

Strategy:
  At each rebalance day, rank N liquid stocks by their N-day return.
  Long the BOTTOM quintile (recent losers).
  Short the TOP quintile (recent winners).
  Equal-weight within each leg, dollar-neutral.

Classical Lehmann (1990) / Lo-MacKinlay (1990) result: short-horizon
cross-sectional reversion is a documented effect in US equities.

Walk-forward: signal known at close of day T-1, trades executed at open T.
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
OUT = REPO / "experiments" / "quant_lab" / "runs"
OUT.mkdir(parents=True, exist_ok=True)

# Universe: 10 highly liquid US stocks/ETFs spanning sectors
UNIVERSE = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA",
             "META", "TSLA", "AMD"]
LOOKBACK = 5    # rank by last 5-day return
HOLD = 5        # hold for 5 days before rebalancing
START = "2019-01-01"
END = "2026-05-12"
STARTING_CAPITAL = 100_000.0


def fetch_yf(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    df = yf.download(ticker, start=START, end=END, progress=False, auto_adjust=True)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
    return df[["Date", "Close"]].rename(columns={"Date": "date", "Close": ticker})


def main():
    print("=" * 70)
    print("Cross-sectional mean reversion · 2019-2026")
    print("=" * 70)
    print(f"\nUniverse: {UNIVERSE}")
    print(f"Lookback {LOOKBACK}d, hold {HOLD}d, equal-weight quintiles, dollar-neutral")

    # Fetch all tickers
    px = None
    for t in UNIVERSE:
        d = fetch_yf(t)
        print(f"  {t}: {len(d)} bars")
        if d.empty:
            continue
        if px is None:
            px = d
        else:
            px = px.merge(d, on="date", how="inner")
    px = px.sort_values("date").reset_index(drop=True)
    print(f"\nAligned: {len(px)} trading days × {len(UNIVERSE)} symbols")

    rets = px[UNIVERSE].pct_change().fillna(0)
    # N-day cumulative return at each day, computed AT END OF DAY (signal for next day)
    lookback_ret = px[UNIVERSE].pct_change(LOOKBACK).fillna(0)
    # At each rebalance day, lookback_ret is the signal — known at close of T-1
    # Position taken at open T (we approximate with close T)
    n_long = max(2, len(UNIVERSE) // 5)  # bottom 20%
    n_short = max(2, len(UNIVERSE) // 5)

    daily_pnl = []
    weights_history = []
    rebalance_days = list(range(LOOKBACK + 1, len(px), HOLD))
    current_weights = pd.Series(0.0, index=UNIVERSE)

    for i, day in enumerate(range(LOOKBACK + 1, len(px))):
        # On rebalance day, recompute weights from PRIOR day's lookback returns
        if day in rebalance_days:
            signal = lookback_ret.iloc[day - 1]  # signal known at T-1 close
            ranked = signal.rank(method="first")
            longs = ranked.nsmallest(n_long).index.tolist()
            shorts = ranked.nlargest(n_short).index.tolist()
            current_weights = pd.Series(0.0, index=UNIVERSE)
            for s in longs:
                current_weights[s] = 1 / n_long
            for s in shorts:
                current_weights[s] = -1 / n_short
        # Apply weights to today's returns (T)
        day_ret = (current_weights * rets.iloc[day]).sum()
        daily_pnl.append({"date": px["date"].iloc[day], "ret": day_ret})

    p = pd.DataFrame(daily_pnl)
    p["equity"] = STARTING_CAPITAL * (1 + p["ret"]).cumprod()
    p["bh_spy"] = STARTING_CAPITAL * (1 + rets["SPY"].iloc[LOOKBACK + 1:].reset_index(drop=True)).cumprod()

    # Metrics
    total_ret = p["equity"].iloc[-1] / STARTING_CAPITAL - 1
    bh_ret = p["bh_spy"].iloc[-1] / STARTING_CAPITAL - 1
    annualized = (1 + total_ret) ** (252 / len(p)) - 1
    sharpe = p["ret"].mean() / p["ret"].std() * np.sqrt(252) if p["ret"].std() > 0 else 0
    max_dd = (p["equity"] / p["equity"].cummax() - 1).min()
    max_dd_bh = (p["bh_spy"] / p["bh_spy"].cummax() - 1).min()
    win_rate = (p["ret"] > 0).mean()

    print(f"\n=== Results ({len(p)} trading days) ===")
    print(f"Strategy total return:    {total_ret*100:+.1f}%")
    print(f"Buy-hold SPY:             {bh_ret*100:+.1f}%")
    print(f"Annualized:               {annualized*100:+.1f}%")
    print(f"Sharpe:                   {sharpe:.2f}")
    print(f"Max drawdown (strat):     {max_dd*100:.1f}%")
    print(f"Max drawdown (SPY):       {max_dd_bh*100:.1f}%")
    print(f"Daily win rate:           {win_rate*100:.1f}%")
    print(f"\nScaled to $1k:            $1,000 → ${(STARTING_CAPITAL + STARTING_CAPITAL * total_ret) * 1000/STARTING_CAPITAL:.2f}")

    p.to_parquet(OUT / "01_csmr_results.parquet", index=False)
    summary = {
        "strategy": "cross_sectional_mean_reversion",
        "universe": UNIVERSE, "lookback": LOOKBACK, "hold": HOLD,
        "trading_days": len(p),
        "total_return_pct": float(total_ret * 100),
        "bh_spy_return_pct": float(bh_ret * 100),
        "annualized_pct": float(annualized * 100),
        "sharpe": float(sharpe),
        "max_dd_strategy_pct": float(max_dd * 100),
        "max_dd_bh_spy_pct": float(max_dd_bh * 100),
        "win_rate_pct": float(win_rate * 100),
    }
    (OUT / "01_csmr_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/01_csmr_*.{{parquet,json}}")


if __name__ == "__main__":
    main()
