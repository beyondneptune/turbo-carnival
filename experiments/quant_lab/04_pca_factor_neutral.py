"""PCA factor-neutral long/short 2019-2026.

Strategy:
  1. Fit PCA on rolling 60-day window of cross-sectional returns
  2. Project out top 3 factors (market + sector + style)
  3. Trade residuals: long bottom-quintile residual, short top-quintile residual
  4. Rebalance every 5 days

The bet: idiosyncratic moves not explained by common factors mean-revert.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

REPO = Path("/Users/weeoto/turbo-carnival")
OUT = REPO / "experiments" / "quant_lab" / "runs"
START = "2019-01-01"
END = "2026-05-12"
STARTING_CAPITAL = 100_000.0

# Diversified 25-stock universe across sectors
UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",   # mega tech
    "NVDA", "AMD", "INTC", "AVGO", "ORCL",     # semis + soft
    "JPM", "BAC", "GS", "WFC", "MS",           # financials
    "XOM", "CVX", "COP",                       # energy
    "JNJ", "PFE", "MRK", "UNH",                # healthcare
    "WMT", "PG", "KO",                         # consumer staples
]
LOOKBACK = 60        # PCA window
N_FACTORS = 3        # remove market + sector + style
REBALANCE = 5        # days
QUINTILE = 5


def fetch(t: str) -> pd.DataFrame:
    import yfinance as yf
    try:
        d = yf.download(t, start=START, end=END, progress=False, auto_adjust=True)
        d = d.reset_index()
        if isinstance(d.columns, pd.MultiIndex):
            d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
        return d[["Date", "Close"]].rename(columns={"Date": "date", "Close": t})
    except Exception:
        return pd.DataFrame()


def main():
    print("=" * 70)
    print("PCA factor-neutral L/S · 2019-2026 · 25-stock universe")
    print("=" * 70)

    px = None
    valid = []
    for t in UNIVERSE:
        d = fetch(t)
        if d.empty:
            print(f"  {t}: SKIP (no data)")
            continue
        valid.append(t)
        px = d if px is None else px.merge(d, on="date", how="outer")
    px = px.sort_values("date").dropna().reset_index(drop=True)
    print(f"\nAligned: {len(px)} days × {len(valid)} symbols")

    rets = px[valid].pct_change().fillna(0)
    daily_pnl = []
    current_weights = pd.Series(0.0, index=valid)
    rebalance_days = list(range(LOOKBACK, len(px), REBALANCE))

    for t in range(LOOKBACK, len(px)):
        if t in rebalance_days:
            # Fit PCA on rolling window of returns (use ≤ t-1 to avoid lookahead)
            window = rets.iloc[t - LOOKBACK:t]
            try:
                pca = PCA(n_components=N_FACTORS).fit(window)
                factor_loadings = pca.components_  # (n_factors, n_stocks)
                # Project each row of window onto factors, then compute residuals
                projected = window.values @ factor_loadings.T @ factor_loadings
                residuals = window.values - projected
                # Use most recent residual (last row) as the "alpha signal"
                last_resid = residuals[-1]  # idiosyncratic move on day t-1
                # Rank: bottom = oversold, top = overbought
                series = pd.Series(last_resid, index=valid)
                n_each = max(2, len(valid) // QUINTILE)
                bottom = series.nsmallest(n_each).index
                top = series.nlargest(n_each).index
                current_weights = pd.Series(0.0, index=valid)
                for s in bottom: current_weights[s] = 1 / n_each
                for s in top: current_weights[s] = -1 / n_each
            except Exception as e:
                pass
        daily_pnl.append({
            "date": px["date"].iloc[t],
            "ret": float((current_weights * rets.iloc[t]).sum()),
        })

    p = pd.DataFrame(daily_pnl)
    p["equity"] = STARTING_CAPITAL * (1 + p["ret"]).cumprod()
    p["spy"] = STARTING_CAPITAL * (1 + rets["AAPL" if "SPY" not in valid else "SPY"].iloc[LOOKBACK:].reset_index(drop=True)).cumprod()

    total = p["equity"].iloc[-1] / STARTING_CAPITAL - 1
    annual = (1 + total) ** (252 / len(p)) - 1
    sharpe = p["ret"].mean() / p["ret"].std() * np.sqrt(252) if p["ret"].std() > 0 else 0
    mdd = (p["equity"] / p["equity"].cummax() - 1).min()
    win = (p["ret"] > 0).mean()

    print(f"\n=== Results ({len(p)} days) ===")
    print(f"Total return:     {total*100:+.1f}%")
    print(f"Annualized:       {annual*100:+.1f}%")
    print(f"Sharpe:           {sharpe:.2f}")
    print(f"Max DD:           {mdd*100:.1f}%")
    print(f"Daily win rate:   {win*100:.1f}%")
    print(f"$1k scaled:       $1,000 → ${1000 * (1 + total):.2f}")

    p.to_parquet(OUT / "04_pca_factor_neutral_results.parquet", index=False)
    summary = {
        "strategy": "pca_factor_neutral",
        "universe_size": len(valid), "lookback": LOOKBACK, "n_factors": N_FACTORS,
        "rebalance_days": REBALANCE,
        "trading_days": len(p),
        "total_return_pct": float(total * 100),
        "annualized_pct": float(annual * 100),
        "sharpe": float(sharpe),
        "max_dd_pct": float(mdd * 100),
        "win_rate_pct": float(win * 100),
    }
    (OUT / "04_pca_factor_neutral_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/04_pca_factor_neutral_*.{{parquet,json}}")


if __name__ == "__main__":
    main()
