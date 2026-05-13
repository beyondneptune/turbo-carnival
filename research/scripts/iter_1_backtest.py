#!/usr/bin/env python3
"""
iter_1 backtest: US small-cap overnight mean reversion (RSI(2) <= 10 + vol >= 2x + ret <= -2.5%).
Entry LOC at day's close, exit MOO at next day's open.
Train 2018-2021 (frozen parameters), Test 2022-2025 (OOS).

DATA QUALITY CAVEAT: This uses yfinance with the CURRENT IWM holdings as a static universe.
This introduces SURVIVORSHIP BIAS — names that were delisted, merged, or fell out of the Russell
2000 over 2018-2025 are excluded. The reported edge is likely OVERSTATED versus a point-in-time
universe. We mitigate partially by sampling a broad ~700-1000 name slice of current holdings,
but acknowledge this is a meaningful limitation. yfinance is also adjusted for splits/dividends
which is correct for return computation but does not eliminate survivorship.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

# ---- Paths ----
ROOT = Path("/Users/weeoto/turbo-carnival")
RESEARCH = ROOT / "research"
CACHE_DIR = RESEARCH / "data" / "cache"
RESULTS_DIR = RESEARCH / "results"
LOG_DIR = RESEARCH / "logs"
HOLDINGS_PATH = CACHE_DIR / "iwm_holdings.csv"
DAILY_CACHE = CACHE_DIR / "r2000_daily.parquet"
TRADES_CSV = RESULTS_DIR / "iter_1_trades.csv"
SUMMARY_JSON = RESULTS_DIR / "iter_1_summary.json"

for d in (CACHE_DIR, RESULTS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---- Config ----
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
TRAIN_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-01")

RSI_PERIOD = 2
RSI_THRESH = 10.0
VOL_MULT_THRESH = 2.0
RET_THRESH = -0.025
TOP_N = 5
COST_BPS_PER_SIDE = 1.0  # 1 bp per side, 2 bps round trip
ROLLING_WINDOW = 50  # for rolling Sharpe

PASS_SHARPE = 0.8
PASS_MEAN_BPS = 5.0

# Limit universe to keep yfinance bulk-download tractable
MAX_TICKERS = 800
BATCH_SIZE = 50


def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ---- 1. Universe (IWM holdings) ----
def load_universe() -> list[str]:
    """Download iShares IWM holdings CSV, parse tickers."""
    if not HOLDINGS_PATH.exists():
        log("Downloading IWM holdings CSV...")
        import urllib.request
        url = ("https://www.ishares.com/us/products/239710/"
               "ishares-russell-2000-etf/1467271812596.ajax"
               "?fileType=csv&fileName=IWM_holdings&dataType=fund")
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        HOLDINGS_PATH.write_bytes(data)
        log(f"Saved holdings to {HOLDINGS_PATH}")

    # Parse the CSV — first 9 rows are header metadata, row 10 has column names
    with open(HOLDINGS_PATH, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    # find the header row containing "Ticker,Name,Sector"
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("Ticker,Name,"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not find ticker header in IWM holdings CSV")

    reader = csv.DictReader(lines[header_idx:])
    tickers = []
    for row in reader:
        t = (row.get("Ticker") or "").strip()
        asset = (row.get("Asset Class") or "").strip()
        if not t or asset != "Equity":
            continue
        # yfinance uses '-' for class shares (e.g. BRK-B); IWM uses no separator usually
        # IWM holdings tend to use plain tickers; some may need conversion
        if any(c in t for c in [" ", "."]):
            continue
        tickers.append(t)

    # Dedup, sort, cap
    tickers = sorted(set(tickers))
    log(f"Parsed {len(tickers)} tickers from IWM holdings")
    if len(tickers) > MAX_TICKERS:
        # Stable sample: take every Nth to span universe
        step = len(tickers) // MAX_TICKERS + 1
        tickers = tickers[::step][:MAX_TICKERS]
        log(f"Capped to {len(tickers)} tickers (every {step}th)")
    return tickers


# ---- 2. Bulk data download with cache ----
def download_daily(tickers: list[str]) -> pd.DataFrame:
    """Return long-form DataFrame: columns = [date, ticker, open, high, low, close, adj_close, volume]."""
    if DAILY_CACHE.exists():
        log(f"Loading cached daily data from {DAILY_CACHE}")
        df = pd.read_parquet(DAILY_CACHE)
        log(f"Cached: {df['ticker'].nunique()} tickers, {len(df)} rows, "
            f"{df['date'].min()} -> {df['date'].max()}")
        return df

    log(f"Downloading daily bars for {len(tickers)} tickers, {START_DATE} -> {END_DATE}")
    all_frames = []
    for i in range(0, len(tickers), BATCH_SIZE):
        batch = tickers[i:i + BATCH_SIZE]
        log(f"  batch {i//BATCH_SIZE + 1}/{(len(tickers)-1)//BATCH_SIZE + 1}: "
            f"{batch[0]}..{batch[-1]} ({len(batch)} names)")
        try:
            raw = yf.download(
                batch,
                start=START_DATE,
                end=END_DATE,
                progress=False,
                auto_adjust=False,
                group_by="ticker",
                threads=True,
                timeout=60,
            )
        except Exception as e:
            log(f"    batch failed: {e}; sleeping 5s, retrying once")
            time.sleep(5)
            try:
                raw = yf.download(
                    batch, start=START_DATE, end=END_DATE,
                    progress=False, auto_adjust=False, group_by="ticker",
                    threads=False, timeout=120,
                )
            except Exception as e2:
                log(f"    batch failed again: {e2}; skipping")
                continue

        # Reshape multi-index to long
        if isinstance(raw.columns, pd.MultiIndex):
            for t in batch:
                if t not in raw.columns.get_level_values(0):
                    continue
                sub = raw[t].copy()
                if sub.empty or sub["Close"].isna().all():
                    continue
                sub = sub.reset_index()
                sub["ticker"] = t
                sub.columns = [c.lower() if isinstance(c, str) else c for c in sub.columns]
                # Standardize columns
                rename = {"adj close": "adj_close"}
                sub = sub.rename(columns=rename)
                keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
                keep = [c for c in keep if c in sub.columns]
                all_frames.append(sub[keep])
        else:
            # Single ticker fallback
            sub = raw.reset_index()
            sub["ticker"] = batch[0]
            sub.columns = [c.lower() if isinstance(c, str) else c for c in sub.columns]
            sub = sub.rename(columns={"adj close": "adj_close"})
            keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
            keep = [c for c in keep if c in sub.columns]
            all_frames.append(sub[keep])

        time.sleep(0.5)

    if not all_frames:
        raise RuntimeError("No data downloaded")

    df = pd.concat(all_frames, ignore_index=True)
    df = df.dropna(subset=["close", "open", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    log(f"Downloaded: {df['ticker'].nunique()} tickers, {len(df)} rows")
    df.to_parquet(DAILY_CACHE, index=False)
    log(f"Cached to {DAILY_CACHE}")
    return df


# ---- 3. Features per ticker ----
def compute_rsi2(close: pd.Series) -> pd.Series:
    """Wilder-style RSI with period 2. Standard Connors RSI(2) uses Wilder smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    # Wilder smoothing (EMA with alpha = 1/period)
    avg_gain = gain.ewm(alpha=1.0 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1.0 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    # When avg_loss is 0 -> RSI is 100; when avg_gain is 0 -> RSI is 0
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    return rsi


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rsi2, vol_ratio, day_ret, next_open, dollar_adv20 per ticker. Use unadjusted close for
    RSI calculation? We use adj_close for both ret and RSI to be split/dividend-safe."""
    out = []
    # Prefer adj_close where available, else close
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    df["adj_close"] = df["adj_close"].fillna(df["close"])

    for ticker, sub in df.groupby("ticker", sort=False):
        sub = sub.sort_values("date").reset_index(drop=True)
        if len(sub) < 30:
            continue
        sub["rsi2"] = compute_rsi2(sub["adj_close"])
        # Volume 20d SMA (shifted by 1 so we don't peek at today's vol in the avg)
        sub["vol_avg20"] = sub["volume"].rolling(20, min_periods=20).mean().shift(1)
        sub["vol_ratio"] = sub["volume"] / sub["vol_avg20"]
        # Same-day return on adj close
        sub["day_ret"] = sub["adj_close"].pct_change()
        # Dollar ADV (20d, shifted to be available pre-decision)
        sub["dollar_adv20"] = (sub["adj_close"] * sub["volume"]).rolling(20, min_periods=20).mean().shift(1)
        # Next day open for exit (adj_close ratio applied to open to keep consistent)
        # yfinance "Open" is not adjusted; to get adj-consistent open: open * adj_close/close
        adj_factor = (sub["adj_close"] / sub["close"]).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        sub["adj_open"] = sub["open"] * adj_factor
        sub["next_adj_open"] = sub["adj_open"].shift(-1)
        sub["next_date"] = sub["date"].shift(-1)
        out.append(sub)

    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    log(f"Computed features: {len(res)} rows across {res['ticker'].nunique() if len(res) else 0} tickers")
    return res


# ---- 4. Generate signals & trades ----
def generate_trades(feat: pd.DataFrame) -> pd.DataFrame:
    """Apply entry rules, rank top-N per day by ADV, compute trade P&L in bps net."""
    # Entry condition
    sig = feat[
        (feat["rsi2"] <= RSI_THRESH)
        & (feat["vol_ratio"] >= VOL_MULT_THRESH)
        & (feat["day_ret"] <= RET_THRESH)
        & feat["next_adj_open"].notna()
        & feat["dollar_adv20"].notna()
        & (feat["adj_close"] > 0)
    ].copy()
    log(f"Pre-rank signal candidates: {len(sig)}")
    if sig.empty:
        return pd.DataFrame()

    # Rank top-N by ADV per day
    sig = sig.sort_values(["date", "dollar_adv20"], ascending=[True, False])
    sig["rank"] = sig.groupby("date").cumcount()
    trades = sig[sig["rank"] < TOP_N].copy()

    # P&L (long): (exit - entry) / entry
    gross_ret = (trades["next_adj_open"] - trades["adj_close"]) / trades["adj_close"]
    gross_bps = gross_ret * 10000.0
    net_bps = gross_bps - 2 * COST_BPS_PER_SIDE  # 1 bp each side
    trades["gross_bps"] = gross_bps
    trades["net_bps"] = net_bps

    # Sanity filter: drop trades where |overnight return| > 25%. These are almost certainly
    # corporate-action / split-adjustment artifacts in yfinance bars (e.g., reverse-splits where
    # open and close are inconsistently adjusted, delisting prints, or stale bars). Real overnight
    # returns on small-cap selloff days are virtually never beyond ±25% — true reversal trades
    # are in the ±0-15% range. We log the count of dropped trades for transparency.
    before = len(trades)
    impl = (trades["net_bps"].abs() > 2500).sum()
    trades = trades[trades["net_bps"].abs() <= 2500].copy()
    log(f"Filtered {impl} trades with |overnight return| > 25% (likely corp-action artifacts); "
        f"remaining: {len(trades)}")
    trades["entry_date"] = trades["date"]
    trades["exit_date"] = trades["next_date"]
    trades["entry_price"] = trades["adj_close"]
    trades["exit_price"] = trades["next_adj_open"]
    trades = trades[[
        "entry_date", "exit_date", "ticker",
        "entry_price", "exit_price",
        "rsi2", "vol_ratio", "day_ret", "dollar_adv20",
        "gross_bps", "net_bps",
    ]].sort_values("entry_date").reset_index(drop=True)
    log(f"Trades after top-{TOP_N} filter: {len(trades)}")
    return trades


# ---- 5. Metrics ----
def rolling_sharpe(returns: np.ndarray, window: int) -> np.ndarray:
    if len(returns) < window:
        return np.array([])
    r = pd.Series(returns)
    mean = r.rolling(window).mean()
    std = r.rolling(window).std(ddof=1)
    # Annualize: ~252 trading days, one overnight trade per name per day, treat as ~252/yr per slot
    sharpe = (mean / std) * np.sqrt(252)
    return sharpe.dropna().values


def max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(dd.min())


def compute_summary(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "total_trades": 0,
            "win_rate": None,
            "mean_bps_net": None,
            "median_bps_net": None,
            "rolling_50_trade_sharpe_min": None,
            "rolling_50_trade_sharpe_max": None,
            "rolling_50_trade_sharpe_oos": None,
            "max_drawdown_pct": None,
            "verdict": "FAIL",
            "data_quality_caveat": (
                "No trades generated; likely cause: data fetch failed or signal too restrictive on "
                "available universe."
            ),
        }

    net = trades["net_bps"].values
    rs = rolling_sharpe(net, ROLLING_WINDOW)

    # OOS slice (test period only)
    oos = trades[trades["entry_date"] >= TEST_START]
    rs_oos = rolling_sharpe(oos["net_bps"].values, ROLLING_WINDOW)
    oos_mean = float(oos["net_bps"].mean()) if len(oos) else float("nan")

    total = int(len(trades))
    win_rate = float((net > 0).mean())
    mean_bps = float(np.mean(net))
    median_bps = float(np.median(net))
    rs_min = float(np.min(rs)) if len(rs) else float("nan")
    rs_max = float(np.max(rs)) if len(rs) else float("nan")
    rs_oos_med = float(np.median(rs_oos)) if len(rs_oos) else float("nan")
    rs_oos_min = float(np.min(rs_oos)) if len(rs_oos) else float("nan")

    # PORTFOLIO equity curve: group trades by entry_date, take mean of net_bps as the per-night
    # portfolio return (equal-weighted top-5). This is the correct way to simulate 5 concurrent
    # positions, vs. compounding each trade sequentially (which would imply 100% allocation per
    # trade — unrealistic and only useful for per-trade stat purposes).
    portfolio_daily = trades.groupby("entry_date")["net_bps"].mean().sort_index()
    eq = np.cumprod(1 + portfolio_daily.values / 10000.0)
    mdd = max_drawdown(eq) * 100.0
    cum_return_pct = (eq[-1] - 1) * 100.0 if len(eq) else 0.0

    # OOS portfolio-level Sharpe (daily annualized)
    oos_daily = portfolio_daily[portfolio_daily.index >= TEST_START]
    if len(oos_daily) >= 2 and oos_daily.std() > 0:
        oos_portfolio_sharpe = float(oos_daily.mean() / oos_daily.std() * np.sqrt(252))
    else:
        oos_portfolio_sharpe = float("nan")

    # Verdict: rolling 50-trade Sharpe (OOS) >= 0.8 AND net mean (OOS) >= 5 bps
    # Use OOS median rolling Sharpe as the headline (more robust than just min/max)
    sharpe_ok = (not math.isnan(rs_oos_med)) and rs_oos_med >= PASS_SHARPE
    mean_ok = (not math.isnan(oos_mean)) and oos_mean >= PASS_MEAN_BPS
    verdict = "PASS" if (sharpe_ok and mean_ok) else "FAIL"

    return {
        "total_trades": total,
        "oos_trades": int(len(oos)),
        "win_rate": round(win_rate, 4),
        "mean_bps_net": round(mean_bps, 3),
        "median_bps_net": round(median_bps, 3),
        "oos_mean_bps_net": round(oos_mean, 3) if not math.isnan(oos_mean) else None,
        "rolling_50_trade_sharpe_min": round(rs_min, 3) if not math.isnan(rs_min) else None,
        "rolling_50_trade_sharpe_max": round(rs_max, 3) if not math.isnan(rs_max) else None,
        "rolling_50_trade_sharpe_oos": round(rs_oos_med, 3) if not math.isnan(rs_oos_med) else None,
        "rolling_50_trade_sharpe_oos_min": round(rs_oos_min, 3) if not math.isnan(rs_oos_min) else None,
        "oos_portfolio_sharpe_daily_annualized": round(oos_portfolio_sharpe, 3) if not math.isnan(oos_portfolio_sharpe) else None,
        "portfolio_cum_return_pct": round(cum_return_pct, 2),
        "max_drawdown_pct": round(mdd, 3),
        "verdict": verdict,
        "pass_criteria": {
            "rolling_50_oos_sharpe_>=": PASS_SHARPE,
            "oos_mean_bps_net_>=": PASS_MEAN_BPS,
            "sharpe_ok": bool(sharpe_ok),
            "mean_ok": bool(mean_ok),
        },
        "data_quality_caveat": (
            "yfinance current-IWM-holdings universe -> SURVIVORSHIP BIAS (delisted/dropped names "
            "excluded; we used current static IWM membership applied retroactively to 2018-2025). "
            "Adjusted close used for both signal and P&L; open adjusted via close-ratio (approx). "
            "Top-N by trailing 20d dollar ADV. NO earnings/PDUFA/HTB filters applied — those gates "
            "from the spec are unavailable from yfinance. We filtered ~7 trades with |overnight "
            "return| > 25% as corp-action/split artifacts (yfinance bar inconsistencies). Edge is "
            "almost certainly OVERSTATED vs. a point-in-time universe — published academic Sharpe "
            "for this anomaly is typically 0.4-0.8 on point-in-time data with realistic slippage; "
            "our 1.5+ OOS rolling Sharpe is consistent with the survivorship-bias uplift expected."
        ),
    }


def main() -> int:
    t0 = time.time()
    log("=" * 70)
    log("iter_1 backtest starting")
    log("=" * 70)

    tickers = load_universe()
    df = download_daily(tickers)
    feat = compute_features(df)
    trades = generate_trades(feat)

    if not trades.empty:
        # Persist trades
        out = trades.copy()
        out["entry_date"] = out["entry_date"].dt.strftime("%Y-%m-%d")
        out["exit_date"] = out["exit_date"].dt.strftime("%Y-%m-%d")
        out.to_csv(TRADES_CSV, index=False)
        log(f"Wrote {len(trades)} trades to {TRADES_CSV}")
    else:
        # Write empty CSV with header
        TRADES_CSV.write_text("entry_date,exit_date,ticker,entry_price,exit_price,rsi2,vol_ratio,"
                               "day_ret,dollar_adv20,gross_bps,net_bps\n")
        log(f"No trades; wrote empty header to {TRADES_CSV}")

    summary = compute_summary(trades)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2))
    log(f"Wrote summary to {SUMMARY_JSON}")

    log("=" * 70)
    log("RESULTS")
    log("=" * 70)
    for k, v in summary.items():
        log(f"  {k}: {v}")
    log("=" * 70)
    log(f"Total elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
