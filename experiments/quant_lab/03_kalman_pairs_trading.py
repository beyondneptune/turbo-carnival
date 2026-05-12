"""Kalman filter pairs trading 2019-2026.

For each cointegrated pair, Kalman filter dynamically estimates hedge ratio:
  y_t = beta_t * x_t + epsilon_t,  beta_t = beta_{t-1} + w_t

Trade when normalized residual z-score crosses thresholds:
  entry: |z| > 2.0
  exit:  |z| < 0.5
  stop:  |z| > 3.5 (bail — pair likely broke)

Pairs tested: SPY/QQQ, AMD/NVDA, AAPL/MSFT.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
OUT = REPO / "experiments" / "quant_lab" / "runs"
START = "2019-01-01"
END = "2026-05-12"
STARTING_CAPITAL = 100_000.0

PAIRS = [
    ("SPY", "QQQ"),
    ("AMD", "NVDA"),
    ("AAPL", "MSFT"),
]


def fetch(t: str) -> pd.DataFrame:
    import yfinance as yf
    d = yf.download(t, start=START, end=END, progress=False, auto_adjust=True)
    d = d.reset_index()
    if isinstance(d.columns, pd.MultiIndex):
        d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    return d[["Date", "Close"]].rename(columns={"Date": "date", "Close": t})


def kalman_pairs(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (beta_t, residual_t) via 1-D Kalman filter on y_t = beta * x_t + e."""
    n = len(y)
    beta = np.zeros(n)
    resid = np.zeros(n)
    # Kalman state: beta. Observation model: y = beta * x + v, v~N(0, R)
    # State model: beta_t = beta_{t-1} + w,  w~N(0, Q)
    Q = 1e-4   # process noise (allows beta to drift)
    R = 1e-2   # observation noise
    P = 1.0    # initial variance
    beta_curr = 1.0  # initial guess
    for t in range(n):
        # Predict
        P = P + Q
        # Update
        K = P * x[t] / (x[t]**2 * P + R)  # Kalman gain
        residual = y[t] - beta_curr * x[t]
        beta_curr = beta_curr + K * residual
        P = (1 - K * x[t]) * P
        beta[t] = beta_curr
        resid[t] = residual
    return beta, resid


def backtest_pair(t_y: str, t_x: str) -> dict:
    print(f"\n=== Pair: {t_y} / {t_x} ===")
    y_data = fetch(t_y).rename(columns={t_y: "y"})
    x_data = fetch(t_x).rename(columns={t_x: "x"})
    df = y_data.merge(x_data, on="date").sort_values("date").reset_index(drop=True)
    print(f"  {len(df)} aligned days")

    beta, resid = kalman_pairs(df["y"].values, df["x"].values)
    df["beta"] = beta
    df["spread"] = resid
    # 60-day rolling z-score of spread
    df["spread_mean"] = df["spread"].rolling(60).mean()
    df["spread_std"] = df["spread"].rolling(60).std()
    df["z"] = (df["spread"] - df["spread_mean"]) / df["spread_std"]
    df = df.dropna().reset_index(drop=True)
    print(f"  After 60-day burn: {len(df)} days")

    # Position rules
    pos = np.zeros(len(df))   # +1 = long spread, -1 = short spread
    for i in range(1, len(df)):
        prev = pos[i - 1]
        z = df["z"].iloc[i - 1]  # use prior z (no lookahead)
        if prev == 0:
            if z > 2.0: pos[i] = -1   # spread too high -> short
            elif z < -2.0: pos[i] = 1  # spread too low -> long
        elif prev == 1:
            if z > -0.5: pos[i] = 0    # close
            elif z > 3.5: pos[i] = 0   # stop (pair broke)
            else: pos[i] = 1
        elif prev == -1:
            if z < 0.5: pos[i] = 0
            elif z < -3.5: pos[i] = 0
            else: pos[i] = -1
    df["pos"] = pos

    # P&L: spread change × position. Spread = y - beta * x, so
    # delta_spread = delta_y - beta * delta_x (approximately)
    # P&L per dollar notional per leg = (pos_yest) * (delta_spread / spread)
    df["y_ret"] = df["y"].pct_change().fillna(0)
    df["x_ret"] = df["x"].pct_change().fillna(0)
    df["spread_ret"] = df["y_ret"] - df["beta"].shift(1) * df["x_ret"]
    df["pnl"] = df["pos"].shift(1).fillna(0) * df["spread_ret"]

    df["equity"] = STARTING_CAPITAL * (1 + df["pnl"]).cumprod()
    total_ret = df["equity"].iloc[-1] / STARTING_CAPITAL - 1
    annualized = (1 + total_ret) ** (252 / len(df)) - 1
    sharpe = df["pnl"].mean() / df["pnl"].std() * np.sqrt(252) if df["pnl"].std() > 0 else 0
    max_dd = (df["equity"] / df["equity"].cummax() - 1).min()
    n_trades = int((df["pos"].diff() != 0).sum())

    print(f"  Trades: {n_trades}")
    print(f"  Total return: {total_ret*100:+.1f}%")
    print(f"  Annualized:   {annualized*100:+.1f}%")
    print(f"  Sharpe:       {sharpe:.2f}")
    print(f"  Max DD:       {max_dd*100:.1f}%")

    return {
        "pair": f"{t_y}/{t_x}",
        "n_days": len(df), "n_trades": n_trades,
        "total_return_pct": float(total_ret * 100),
        "annualized_pct": float(annualized * 100),
        "sharpe": float(sharpe),
        "max_dd_pct": float(max_dd * 100),
    }


def main():
    print("=" * 70)
    print("Kalman filter pairs trading · 2019-2026")
    print("=" * 70)
    results = []
    for y, x in PAIRS:
        try:
            results.append(backtest_pair(y, x))
        except Exception as e:
            print(f"  ERROR on {y}/{x}: {e}")
            results.append({"pair": f"{y}/{x}", "error": str(e)[:100]})

    summary = {"strategy": "kalman_pairs_trading", "pairs": results}
    (OUT / "03_kalman_pairs_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/03_kalman_pairs_summary.json")


if __name__ == "__main__":
    main()
