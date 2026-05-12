"""HMM regime detection 2019-2026.

Fit a 2-state Gaussian HMM on SPY daily returns + VIX level. The two states
naturally cluster as bull (low vol, positive drift) and bear (high vol,
negative drift).

Strategy:
  - Detect regime at close of day T-1 using HMM forward filter
  - If state = bull: LONG SPY at open T
  - If state = bear: FLAT (cash) or SHORT SPY

Walk-forward refit every 252 days (yearly) to capture regime drift.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from hmmlearn import hmm

REPO = Path("/Users/weeoto/turbo-carnival")
OUT = REPO / "experiments" / "quant_lab" / "runs"
OUT.mkdir(parents=True, exist_ok=True)

START = "2019-01-01"
END = "2026-05-12"
STARTING_CAPITAL = 100_000.0
N_STATES = 2
REFIT_EVERY = 252


def fetch(t: str) -> pd.DataFrame:
    import yfinance as yf
    d = yf.download(t, start=START, end=END, progress=False, auto_adjust=True)
    d = d.reset_index()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    return d[["Date", "Close"]].rename(columns={"Date": "date", "Close": t})


def main():
    print("=" * 70)
    print("HMM regime detection · 2-state Gaussian · SPY long/flat 2019-2026")
    print("=" * 70)

    spy = fetch("SPY")
    vix = fetch("^VIX").rename(columns={"^VIX": "vix"})
    df = spy.merge(vix, on="date", how="inner").sort_values("date").reset_index(drop=True)
    df["ret"] = df["SPY"].pct_change().fillna(0)
    print(f"\n{len(df)} trading days")

    # Walk-forward HMM
    features = ["ret", "vix"]
    df["state"] = -1  # unknown until enough history
    df["bull_state"] = -1

    burn = REFIT_EVERY
    model = None
    bull_state = None
    last_refit = 0

    for t in range(burn, len(df)):
        # Refit every REFIT_EVERY days
        if (t - last_refit) >= REFIT_EVERY or model is None:
            X_train = df[features].iloc[:t].values
            try:
                model = hmm.GaussianHMM(n_components=N_STATES, covariance_type="full",
                                          n_iter=100, random_state=42)
                model.fit(X_train)
                # Identify bull state: the one with higher mean ret + lower vol
                state_means = model.means_  # shape (N_STATES, n_features)
                # state with higher 0-th feature (ret) AND lower 1st feature (vix)
                bull_score = state_means[:, 0] - 0.001 * state_means[:, 1]
                bull_state = int(np.argmax(bull_score))
                last_refit = t
            except Exception:
                continue

        # Predict state for day t (using data ≤ t-1)
        try:
            X_pred = df[features].iloc[:t].values  # includes up to t-1
            state = int(model.predict(X_pred)[-1])
            df.at[t, "state"] = state
            df.at[t, "bull_state"] = bull_state
        except Exception:
            continue

    # Strategy: long SPY when state == bull_state
    df["signal"] = (df["state"] == df["bull_state"]).astype(int)
    # Position taken at open T based on signal known at T-1 close (already shifted)
    df["strat_ret"] = df["signal"].shift(1).fillna(0) * df["ret"]
    df["strat_equity"] = STARTING_CAPITAL * (1 + df["strat_ret"]).cumprod()
    df["bh_equity"] = STARTING_CAPITAL * (1 + df["ret"]).cumprod()

    # Stats — only post-burn
    p = df.iloc[burn:].copy()
    total_ret = p["strat_equity"].iloc[-1] / p["strat_equity"].iloc[0] - 1
    bh_ret = p["bh_equity"].iloc[-1] / p["bh_equity"].iloc[0] - 1
    annualized = (1 + total_ret) ** (252 / len(p)) - 1
    days_long = int(p["signal"].sum())
    days_flat = len(p) - days_long
    sharpe = p["strat_ret"].mean() / p["strat_ret"].std() * np.sqrt(252) if p["strat_ret"].std() > 0 else 0
    max_dd = (p["strat_equity"] / p["strat_equity"].cummax() - 1).min()
    max_dd_bh = (p["bh_equity"] / p["bh_equity"].cummax() - 1).min()

    print(f"\n=== Results ({len(p)} trading days post-burn) ===")
    print(f"Strategy total return:    {total_ret*100:+.1f}%")
    print(f"Buy-hold SPY:             {bh_ret*100:+.1f}%")
    print(f"Annualized:               {annualized*100:+.1f}%")
    print(f"Days long: {days_long} ({days_long/len(p)*100:.0f}%)  Days flat: {days_flat} ({days_flat/len(p)*100:.0f}%)")
    print(f"Sharpe:                   {sharpe:.2f}")
    print(f"Max drawdown (strat):     {max_dd*100:.1f}%")
    print(f"Max drawdown (SPY):       {max_dd_bh*100:.1f}%")
    print(f"\nScaled to $1k:            $1,000 → ${1000 * (1 + total_ret):.2f}")

    df.to_parquet(OUT / "02_hmm_results.parquet", index=False)
    summary = {
        "strategy": "hmm_regime_detection_2state",
        "burn_days": burn, "refit_every": REFIT_EVERY,
        "trading_days": len(p),
        "days_long_pct": float(100 * days_long / len(p)),
        "total_return_pct": float(total_ret * 100),
        "bh_spy_return_pct": float(bh_ret * 100),
        "annualized_pct": float(annualized * 100),
        "sharpe": float(sharpe),
        "max_dd_strategy_pct": float(max_dd * 100),
        "max_dd_bh_pct": float(max_dd_bh * 100),
    }
    (OUT / "02_hmm_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/02_hmm_*.{{parquet,json}}")


if __name__ == "__main__":
    main()
