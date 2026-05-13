"""
iter_23 backtest: Global Risk Premia Multi-Factor (carry + momentum + value)
Simplified universe: 20 ETF proxies for futures.
- Monthly rebalance, long top 5 / short bottom 5, equal weight.
- Vol-scaled portfolio to 10% ann vol (60d realized).
- 10 bps per name per rebalance.
- IS 2005-2014, OOS 2015-2025.

Honesty caveats (logged):
- ETFs are imperfect proxies for futures: returns include dividends, but
  futures roll yield / convenience yield are NOT captured.
- "Carry" is approximated by trailing 12m dividend yield (yfinance .info).
  This is a WEAK proxy for true carry (no rates carry, no futures-basis carry).
- No FX carry differential, no bond curve carry. ETF dividend yield captures
  the cash-flow yield of the ETF wrapper only.
"""

import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path("/Users/weeoto/turbo-carnival/research")
DATA_CACHE = ROOT / "data" / "cache" / "global_etfs_daily.parquet"
LOG_PATH = ROOT / "logs" / "iter_23_backtest.log"
PNL_PATH = ROOT / "results" / "iter_23_monthly_pnl.csv"
SUMMARY_PATH = ROOT / "results" / "iter_23_summary.json"

UNIVERSE = [
    # Equity indices
    "SPY", "EFA", "EEM", "EWJ", "EWY", "EWG", "EWU", "EWA",
    # Bonds
    "TLT", "IEF", "BND", "EMB",
    # Commodities
    "USO", "GLD", "SLV", "DBA",
    # REITs
    "VNQ",
    # FX
    "UUP", "FXE", "FXY",
]

START = "2005-01-01"
END = "2025-12-31"
IS_END = "2014-12-31"
OOS_START = "2015-01-01"

LONG_N = 5
SHORT_N = 5
TARGET_VOL = 0.10
VOL_LOOKBACK = 60
COST_BPS_PER_NAME = 10.0  # per name per rebalance

# ---------- logging ----------
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("iter_23")


def fetch_prices() -> pd.DataFrame:
    """Daily adjusted closes for the ETF universe."""
    DATA_CACHE.parent.mkdir(parents=True, exist_ok=True)
    if DATA_CACHE.exists():
        try:
            df = pd.read_parquet(DATA_CACHE)
            if isinstance(df.index, pd.DatetimeIndex) and len(df.columns) >= len(UNIVERSE) - 2:
                log.info(f"Loaded cached daily prices: shape={df.shape}, range={df.index.min().date()}..{df.index.max().date()}")
                return df
            else:
                log.info("Cache present but schema doesn't match; refetching")
        except Exception as e:
            log.warning(f"Cache read failed: {e}; refetching")

    log.info(f"Downloading {len(UNIVERSE)} tickers from yfinance: {START}..{END}")
    raw = yf.download(
        tickers=UNIVERSE,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    # Build close-price frame (auto_adjust=True puts the adjusted total-return-ish price in 'Close')
    closes = {}
    for t in UNIVERSE:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                s = raw[t]["Close"]
            else:
                s = raw["Close"]
            closes[t] = s
        except Exception as e:
            log.warning(f"Failed to extract close for {t}: {e}")
    df = pd.DataFrame(closes).sort_index()
    df = df.dropna(how="all")
    df.to_parquet(DATA_CACHE)
    log.info(f"Cached daily prices to {DATA_CACHE}: shape={df.shape}")
    return df


def fetch_dividend_yields() -> dict:
    """Trailing dividend yield per ticker (snapshot from yfinance .info).

    This is the simplified carry proxy. NOTE: this is a single static snapshot,
    not a time series. We use it as a per-instrument cross-sectional carry
    signal that doesn't change month-to-month (the cross-sectional rank
    matters more than the absolute level for this simplified backtest).
    A more honest implementation would compute trailing 12m distributions
    from yfinance's actions data — we do that below as a refinement.
    """
    yields = {}
    for t in UNIVERSE:
        try:
            tk = yf.Ticker(t)
            info = tk.info
            # yfinance keys vary; try several
            y = info.get("dividendYield") or info.get("trailingAnnualDividendYield") or info.get("yield")
            if y is None or y == 0:
                y = 0.0
            else:
                y = float(y)
                # yfinance returns these as fractions (0.02) sometimes, percentages (2.0) other times
                if y > 1.0:
                    y = y / 100.0
            yields[t] = y
            log.info(f"  {t} static div yield = {y*100:.2f}%")
        except Exception as e:
            log.warning(f"  {t} info fetch failed: {e}; using 0")
            yields[t] = 0.0
    return yields


def compute_trailing_div_yield(daily_prices: pd.DataFrame) -> pd.DataFrame:
    """Compute trailing 12-month distribution yield per ticker per month.

    This is the more accurate carry proxy: cumulative dividends paid in the
    last 252 trading days, divided by current price. Uses yfinance .actions
    for the dividend series.

    Returns DataFrame indexed by month-end, columns = tickers.
    """
    log.info("Computing trailing 12m dividend yield series per ticker...")
    # monthly month-end index aligned to prices
    month_end = daily_prices.resample("ME").last().index

    out = pd.DataFrame(index=month_end, columns=daily_prices.columns, dtype=float)

    for t in daily_prices.columns:
        try:
            tk = yf.Ticker(t)
            divs = tk.dividends
            if divs is None or len(divs) == 0:
                out[t] = 0.0
                continue
            # remove tz
            if divs.index.tz is not None:
                divs.index = divs.index.tz_localize(None)
            divs = divs.sort_index()
            # for each month-end, sum dividends in trailing 365 days, divide by price
            for me in month_end:
                window_start = me - pd.Timedelta(days=365)
                trailing_div = divs.loc[(divs.index > window_start) & (divs.index <= me)].sum()
                px = daily_prices[t].asof(me)
                if pd.isna(px) or px <= 0:
                    out.loc[me, t] = np.nan
                else:
                    out.loc[me, t] = trailing_div / px
            n_nonzero = (out[t] > 0).sum()
            log.info(f"  {t}: trailing div-yield series, non-zero months={n_nonzero}, "
                     f"mean={out[t].mean()*100:.2f}%, last={out[t].iloc[-1]*100:.2f}%")
        except Exception as e:
            log.warning(f"  {t} dividends fetch failed: {e}; setting 0")
            out[t] = 0.0
    return out.astype(float)


def compute_signals(daily: pd.DataFrame, div_yield_m: pd.DataFrame):
    """Compute monthly momentum, value, carry signals.

    Returns: (mom_m, val_m, car_m, mret_m) all indexed by month-end with ticker cols.
    """
    log.info("Computing monthly signals...")
    # monthly close = last trading day of each month
    monthly = daily.resample("ME").last()
    log.info(f"Monthly price frame: shape={monthly.shape}, range={monthly.index.min().date()}..{monthly.index.max().date()}")

    # Monthly simple returns (used as the target/realized series)
    mret = monthly.pct_change()

    # Momentum: 12-1 month return computed on monthly closes
    # signal at month t uses prices from t-12 to t-1 (so signal known at t end)
    log_p = np.log(monthly)
    mom = log_p.shift(1) - log_p.shift(12)  # log return from t-12 to t-1
    log.info(f"Momentum signal: shape={mom.shape}, sample non-nan at 2010-06: "
             f"{mom.loc['2010-06-30'].notna().sum() if '2010-06-30' in mom.index else 'n/a'}")

    # Value: 5-year mean reversion z-score, signed so cheap = positive
    # value_i(t) = -(current_price - 5y_avg) / 5y_std
    # ~60 monthly observations = 5 years
    rolling_mean = monthly.rolling(window=60, min_periods=36).mean()
    rolling_std = monthly.rolling(window=60, min_periods=36).std()
    val = -((monthly - rolling_mean) / rolling_std)
    val = val.shift(1)  # use only info available before the month's return
    log.info(f"Value signal: shape={val.shape}")

    # Carry: trailing 12m dividend yield (already monthly)
    # Align index and shift by 1 to avoid using current month's dividend (look-ahead)
    car = div_yield_m.reindex(monthly.index).shift(1)
    log.info(f"Carry signal: shape={car.shape}, non-nan sample at 2010-06: "
             f"{car.loc['2010-06-30'].notna().sum() if '2010-06-30' in car.index else 'n/a'}")

    return mom, val, car, mret


def zscore_xs(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional z-score per row, clipped at +/- 2."""
    mu = df.mean(axis=1)
    sd = df.std(axis=1).replace(0, np.nan)
    z = df.sub(mu, axis=0).div(sd, axis=0)
    return z.clip(-2, 2)


def compute_ic(signal: pd.DataFrame, fwd_ret: pd.DataFrame) -> float:
    """Average cross-sectional Spearman rank correlation between signal and
    the next month's return.
    """
    common_idx = signal.index.intersection(fwd_ret.index)
    ics = []
    for dt in common_idx:
        s = signal.loc[dt]
        r = fwd_ret.loc[dt]
        merged = pd.concat([s, r], axis=1, keys=["sig", "ret"]).dropna()
        if len(merged) < 5:
            continue
        rs = merged["sig"].rank()
        rr = merged["ret"].rank()
        if rs.std() == 0 or rr.std() == 0:
            continue
        ic = rs.corr(rr)
        if pd.notna(ic):
            ics.append(ic)
    return float(np.mean(ics)) if ics else float("nan")


def run_backtest(daily, mom, val, car, mret):
    """Run the monthly long-short backtest with vol-scaling and costs."""
    # Z-score each factor cross-sectionally
    z_mom = zscore_xs(mom)
    z_val = zscore_xs(val)
    z_car = zscore_xs(car)

    # Combined signal
    F = (z_mom + z_val + z_car) / 3.0

    # Targets: long top 5 / short bottom 5, equal weight (gross 1.0 per side)
    weights = pd.DataFrame(0.0, index=F.index, columns=F.columns)
    for dt, row in F.iterrows():
        clean = row.dropna()
        if len(clean) < (LONG_N + SHORT_N):
            continue
        ranked = clean.sort_values(ascending=False)
        longs = ranked.index[:LONG_N]
        shorts = ranked.index[-SHORT_N:]
        weights.loc[dt, longs] = 1.0 / LONG_N  # +20% each (gross long 100%)
        weights.loc[dt, shorts] = -1.0 / SHORT_N

    # Daily portfolio returns assuming the targets are held for the month
    # We compute monthly portfolio return as sum(weights * mret_next_month)
    # weights at month-end t -> apply to return from t to t+1
    next_mret = mret.shift(-1)
    raw_port_m = (weights * next_mret).sum(axis=1)

    # Realized vol from past 60 trading days of daily portfolio returns
    # Build daily proxy portfolio returns using stale weights between rebalances
    daily_ret = daily.pct_change()
    # Align: at each rebalance, weight applies until the next rebalance
    daily_weights = weights.reindex(daily_ret.index, method="ffill").shift(1).fillna(0)
    daily_port = (daily_weights * daily_ret).sum(axis=1)

    # Realized 60-day vol at each month-end (annualized)
    rolling_vol_daily = daily_port.rolling(window=VOL_LOOKBACK, min_periods=20).std() * np.sqrt(252)
    vol_at_rebalance = rolling_vol_daily.reindex(weights.index, method="ffill")

    # Scale factor at each month: target_vol / realized_vol; clipped to [0.1, 5.0]
    scale = (TARGET_VOL / vol_at_rebalance).clip(lower=0.1, upper=5.0).fillna(1.0)

    # Apply scale to monthly portfolio returns
    scaled_port_m = raw_port_m * scale

    # Costs: turnover-based. Cost per name per rebalance = 10 bps on that name's notional.
    # Approximation: each month, total cost = sum_i |w_new_i - w_old_i| * COST_BPS_PER_NAME * 1e-4 * scale
    prev_w_scaled = (weights.shift(1) * scale.shift(1).values.reshape(-1, 1)).fillna(0)
    cur_w_scaled = weights.mul(scale, axis=0).fillna(0)
    # Re-do with proper alignment
    cur_w_scaled = weights.multiply(scale, axis=0).fillna(0)
    prev_w_scaled = cur_w_scaled.shift(1).fillna(0)
    turnover = (cur_w_scaled - prev_w_scaled).abs()
    cost_per_month = turnover.sum(axis=1) * (COST_BPS_PER_NAME * 1e-4)

    net_port_m = scaled_port_m - cost_per_month

    return F, weights, raw_port_m, scaled_port_m, net_port_m, scale, cost_per_month


def metrics(returns: pd.Series, label: str):
    r = returns.dropna()
    if len(r) == 0:
        return {"months": 0, "ann_return_pct": 0, "ann_vol_pct": 0, "sharpe": 0, "max_dd_pct": 0}
    mu_ann = r.mean() * 12
    sd_ann = r.std() * np.sqrt(12)
    sharpe = mu_ann / sd_ann if sd_ann > 0 else 0.0
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum / peak - 1).min()
    out = {
        "months": int(len(r)),
        "ann_return_pct": float(mu_ann * 100),
        "ann_vol_pct": float(sd_ann * 100),
        "sharpe": float(sharpe),
        "max_dd_pct": float(dd * 100),
    }
    log.info(f"{label}: months={out['months']}, ann_ret={out['ann_return_pct']:.2f}%, "
             f"ann_vol={out['ann_vol_pct']:.2f}%, sharpe={out['sharpe']:.3f}, "
             f"max_dd={out['max_dd_pct']:.2f}%")
    return out


def main():
    log.info("=" * 80)
    log.info("iter_23 simplified backtest START")
    log.info(f"Universe ({len(UNIVERSE)}): {UNIVERSE}")
    log.info(f"Window: {START} .. {END}")
    log.info(f"IS end: {IS_END}, OOS start: {OOS_START}")
    log.info(f"Long top {LONG_N}, short bottom {SHORT_N}; target vol = {TARGET_VOL:.0%}; "
             f"cost = {COST_BPS_PER_NAME} bps/name/rebal")
    log.info("=" * 80)

    daily = fetch_prices()
    # Drop tickers that are entirely missing
    keep = [c for c in daily.columns if daily[c].notna().sum() > 200]
    dropped = [c for c in daily.columns if c not in keep]
    if dropped:
        log.warning(f"Dropping sparse tickers: {dropped}")
    daily = daily[keep]
    log.info(f"Active tickers ({len(keep)}): {keep}")

    div_yield_m = compute_trailing_div_yield(daily)

    mom, val, car, mret = compute_signals(daily, div_yield_m)

    F, weights, raw_m, scaled_m, net_m, scale, costs = run_backtest(daily, mom, val, car, mret)

    # Save monthly PnL
    pnl_df = pd.DataFrame({
        "raw_return": raw_m,
        "vol_scale": scale,
        "scaled_return": scaled_m,
        "cost": costs,
        "net_return": net_m,
    })
    pnl_df.index.name = "date"
    pnl_df = pnl_df.dropna(subset=["net_return"])
    PNL_PATH.parent.mkdir(parents=True, exist_ok=True)
    pnl_df.to_csv(PNL_PATH)
    log.info(f"Saved monthly PnL to {PNL_PATH}: {len(pnl_df)} rows")

    # Metrics IS / OOS
    is_mask = pnl_df.index <= IS_END
    oos_mask = pnl_df.index >= OOS_START
    is_metrics = metrics(pnl_df.loc[is_mask, "net_return"], "IS (2005-2014)")
    oos_metrics = metrics(pnl_df.loc[oos_mask, "net_return"], "OOS (2015-2025)")

    # Factor ICs on OOS: signal at month t vs realized next-month return
    fwd_ret = mret.shift(-1)
    oos_idx_mask = (mom.index >= OOS_START)
    mom_oos = mom.loc[oos_idx_mask]
    val_oos = val.loc[oos_idx_mask]
    car_oos = car.loc[oos_idx_mask]
    fwd_oos = fwd_ret.loc[oos_idx_mask]
    ic_mom = compute_ic(mom_oos, fwd_oos)
    ic_val = compute_ic(val_oos, fwd_oos)
    ic_car = compute_ic(car_oos, fwd_oos)
    log.info(f"OOS factor IC: mom={ic_mom:.4f}, val={ic_val:.4f}, car={ic_car:.4f}")

    # Verdict
    sharpe_oos = oos_metrics["sharpe"]
    all_ic_positive = (ic_mom > 0) and (ic_val > 0) and (ic_car > 0)
    passes = (sharpe_oos >= 0.8) and all_ic_positive
    verdict = "PASS" if passes else "FAIL"

    summary = {
        "iter": "iter_23",
        "universe": list(daily.columns),
        "n_instruments": int(len(daily.columns)),
        "start": START,
        "end": END,
        "is_end": IS_END,
        "oos_start": OOS_START,
        "months_is": is_metrics["months"],
        "months_oos": oos_metrics["months"],
        "ann_return_pct": oos_metrics["ann_return_pct"],
        "ann_vol_pct": oos_metrics["ann_vol_pct"],
        "sharpe_is": is_metrics["sharpe"],
        "sharpe_oos": sharpe_oos,
        "max_dd_pct": oos_metrics["max_dd_pct"],
        "factor_ic_momentum": ic_mom,
        "factor_ic_value": ic_val,
        "factor_ic_carry": ic_car,
        "verdict": verdict,
        "verdict_rule": "PASS if sharpe_oos >= 0.8 AND all three factor ICs > 0",
        "caveats": [
            "Simplified universe: 20 ETF proxies for 50+ futures.",
            "ETFs include dividends but NOT futures roll yield / convenience yield.",
            "Carry proxy = trailing 12m dividend yield (a weak proxy; no rates carry, no curve carry, no FX OIS differential).",
            "No purge/embargo, no expanding-window walk-forward; single IS/OOS split.",
            "Costs = 10 bps per name per rebalance, applied to vol-scaled turnover.",
        ],
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Saved summary to {SUMMARY_PATH}")

    print("\n" + "=" * 80)
    print(f"RESULT_LINE: iter_23 | months_oos={oos_metrics['months']} | "
          f"sharpe={sharpe_oos:.3f} | ann_ret={oos_metrics['ann_return_pct']:.2f}% | "
          f"factor_IC={ic_mom:.4f}/{ic_val:.4f}/{ic_car:.4f} | VERDICT={verdict}")
    print(f"CAVEAT: 20 ETF proxies (not 50+ futures); dividend yield as carry proxy; no rates/curve carry.")
    print(f"PATH: {SUMMARY_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
