"""SVXY regime rotation — proper walk-forward backtest 2019-2026.

Strategy (from Agent 3's research):
  - Compute VIX/VIX3M ratio daily
  - LONG SVXY when ratio < 1.0 (contango — vol futures roll yield is positive)
  - FLAT when ratio > 1.0 (backwardation — roll yield negative, vol spike risk)
  - Optional: hedge during backwardation by going to cash + small SPY long

Walk-forward: at each day T, decide based on VIX/VIX3M observed at T close,
execute at T+1 open. No lookahead.

Data:
  - VIX, VIX3M: try yfinance or fetch from Alpaca (^VIX, ^VIX3M as indices)
  - SVXY: Alpaca historical bars

Output: full daily equity curve from inception to today.
"""
from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path("/Users/weeoto/turbo-carnival")
OUT = REPO / "experiments" / "svxy_lab" / "runs"
OUT.mkdir(parents=True, exist_ok=True)

STARTING_CAPITAL = 100_000.0  # backtest at $100k for clean math; scale at end
START_DATE = "2019-01-01"
END_DATE = "2026-05-11"


def load_env():
    for line in (REPO / "council" / ".env.local").read_text().splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def try_yfinance(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Try yfinance first — has VIX/VIX3M as indices and works without API key."""
    try:
        import yfinance as yf
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
        if df.empty:
            return pd.DataFrame()
        df = df.reset_index()
        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
    except Exception as e:
        print(f"  yfinance failed for {ticker}: {e}")
        return pd.DataFrame()


def fetch_alpaca_daily(symbol: str, start: str, end: str) -> pd.DataFrame:
    load_env()
    H = {"APCA-API-KEY-ID": os.environ["ALPACA_AEGIS_KEY"],
         "APCA-API-SECRET-KEY": os.environ["ALPACA_AEGIS_SECRET"]}
    rows = []
    page_token = None
    while True:
        params = {"symbols": symbol, "timeframe": "1Day", "start": start, "end": end,
                  "limit": 10000, "adjustment": "split"}
        if page_token:
            params["page_token"] = page_token
        try:
            r = requests.get("https://data.alpaca.markets/v2/stocks/bars",
                              headers=H, params=params, timeout=30)
            if r.status_code != 200:
                print(f"  Alpaca {symbol}: HTTP {r.status_code}")
                break
            j = r.json()
            for b in j.get("bars", {}).get(symbol, []):
                rows.append({"date": b["t"][:10], "close": b["c"]})
            page_token = j.get("next_page_token")
            if not page_token:
                break
        except Exception as e:
            print(f"  Alpaca error: {e}")
            break
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["date"] = pd.to_datetime(df["date"])
    return df


def main():
    print("=" * 70)
    print("SVXY regime rotation walk-forward: 2019-01 → 2026-05")
    print("=" * 70)

    # Fetch VIX
    print("\nFetching VIX (^VIX)...")
    vix = try_yfinance("^VIX", START_DATE, END_DATE)
    print(f"  VIX: {len(vix)} bars")
    if vix.empty:
        print("  yfinance failed; cannot proceed without VIX. Aborting.")
        return

    # Fetch VIX3M (3-month VIX index)
    print("\nFetching VIX3M (^VIX3M)...")
    vix3m = try_yfinance("^VIX3M", START_DATE, END_DATE)
    print(f"  VIX3M: {len(vix3m)} bars")

    # Fetch SVXY (the inverse-VIX-futures ETF)
    print("\nFetching SVXY...")
    svxy = try_yfinance("SVXY", START_DATE, END_DATE)
    print(f"  SVXY (yfinance): {len(svxy)} bars")
    if svxy.empty:
        print("  trying Alpaca for SVXY...")
        svxy = fetch_alpaca_daily("SVXY", START_DATE, END_DATE)
        print(f"  SVXY (Alpaca): {len(svxy)} bars")

    if vix.empty or vix3m.empty or svxy.empty:
        print("\nMissing data. Aborting.")
        return

    # Align dates
    vix = vix.rename(columns={"close": "vix"})
    vix3m = vix3m.rename(columns={"close": "vix3m"})
    svxy = svxy.rename(columns={"close": "svxy"})
    df = vix.merge(vix3m, on="date").merge(svxy, on="date").sort_values("date").reset_index(drop=True)
    print(f"\nAligned: {len(df)} trading days")

    df["ratio"] = df["vix"] / df["vix3m"]
    df["contango"] = df["ratio"] < 1.0
    df["svxy_ret"] = df["svxy"].pct_change().fillna(0)

    # Walk-forward strategy: at day T, decide based on ratio[T-1] (no lookahead),
    # take position at SVXY[T] open ≈ SVXY[T-1] close (approximation)
    df["signal"] = df["contango"].shift(1).fillna(False)  # signal known at T from T-1
    df["strategy_ret"] = np.where(df["signal"], df["svxy_ret"], 0)

    # Equity curves
    df["svxy_equity"] = STARTING_CAPITAL * (1 + df["svxy_ret"]).cumprod()
    df["strategy_equity"] = STARTING_CAPITAL * (1 + df["strategy_ret"]).cumprod()

    # Stats
    total_days = len(df)
    days_long = df["signal"].sum()
    days_flat = total_days - days_long
    total_strategy_return = df["strategy_equity"].iloc[-1] / STARTING_CAPITAL - 1
    total_buy_hold_return = df["svxy_equity"].iloc[-1] / STARTING_CAPITAL - 1
    annualized = (1 + total_strategy_return) ** (252 / total_days) - 1
    max_dd_strategy = (df["strategy_equity"] / df["strategy_equity"].cummax() - 1).min()
    max_dd_bh = (df["svxy_equity"] / df["svxy_equity"].cummax() - 1).min()
    sharpe = df["strategy_ret"].mean() / df["strategy_ret"].std() * np.sqrt(252) if df["strategy_ret"].std() > 0 else 0

    print(f"\n=== Results ===")
    print(f"Total trading days:     {total_days}")
    print(f"Long days:              {days_long} ({100*days_long/total_days:.0f}%)")
    print(f"Flat days:              {days_flat} ({100*days_flat/total_days:.0f}%)")
    print(f"Strategy total return:  {total_strategy_return*100:+.1f}%")
    print(f"Buy-and-hold SVXY:      {total_buy_hold_return*100:+.1f}%")
    print(f"Annualized:             {annualized*100:+.1f}%")
    print(f"Max drawdown (strat):   {max_dd_strategy*100:.1f}%")
    print(f"Max drawdown (B&H):     {max_dd_bh*100:.1f}%")
    print(f"Sharpe (annual):        {sharpe:.2f}")

    # Scale to $1k for TERMINATOR view
    scale = 1_000 / STARTING_CAPITAL
    final_1k = STARTING_CAPITAL * scale * (1 + total_strategy_return)
    print(f"\nScaled to $1k starting:  $1,000.00 → ${final_1k:.2f}")

    # Save
    df.to_parquet(OUT / "walk_forward_results.parquet", index=False)
    summary = {
        "start_date": START_DATE, "end_date": END_DATE,
        "total_days": int(total_days),
        "days_long": int(days_long), "days_flat": int(days_flat),
        "pct_in_market": 100 * days_long / total_days,
        "total_return_pct": float(total_strategy_return * 100),
        "buy_hold_return_pct": float(total_buy_hold_return * 100),
        "annualized_pct": float(annualized * 100),
        "max_drawdown_strategy_pct": float(max_dd_strategy * 100),
        "max_drawdown_bh_pct": float(max_dd_bh * 100),
        "sharpe": float(sharpe),
        "scaled_1k_final": float(final_1k),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/walk_forward_results.parquet + summary.json")

    # Quick HTML rendering of equity curve
    csv = df[["date", "vix", "vix3m", "ratio", "signal", "svxy_equity", "strategy_equity"]].copy()
    csv["date"] = csv["date"].dt.strftime("%Y-%m-%d")
    csv.to_csv(OUT / "walk_forward_results.csv", index=False)
    print(f"wrote {OUT}/walk_forward_results.csv  ({len(csv)} rows for plotting)")


if __name__ == "__main__":
    main()
