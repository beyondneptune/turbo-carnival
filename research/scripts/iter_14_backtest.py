"""iter_14 backtest: LightGBM cross-section ranker on US large-cap equities.

Spec: /Users/weeoto/turbo-carnival/research/candidates/iter_14_lightgbm_cross_section_russell1000.md

SIMPLIFIED SCOPE (data constraints — full point-in-time fundamentals are not free):
  - Universe: S&P 500 constituents (used as proxy for Russell 1000 — biased; see CAVEAT)
  - Daily OHLCV 2014-2025 via yfinance bulk download (cached)
  - Features (price-only, 10 features — vs. spec's ~70 with fundamentals/alt-data):
      1.  12-1 momentum (252d return ex last 21d)
      2.  1-month reversal (trailing 21d return)
      3.  Realized vol 60d (annualized stdev of daily returns)
      4.  Idiosyncratic vol vs SPY 60d (residual stdev from rolling beta regression)
      5.  Dollar volume 20d ADV (log scale)
      6.  Beta to SPY 60d
      7.  Skewness 60d
      8.  Kurtosis 60d
      9.  6-month momentum (126d return)
      10. 6-month vol (126d annualized stdev)
  - Target: next-month (21-trading-day) forward excess return vs universe equal-weight mean
  - Model: LightGBM regressor on cross-sectional rank target (continuous)
  - Walk-forward: train expanding window starting Jan 2014, retrain every 12 months, OOS 2020-2025
  - Portfolio: long top-decile (~50 names), short bottom-decile (~50 names), equal-weight, monthly rebalance
  - Cost: 5 bps per side per name traded (assumes 50% turnover monthly per side)
  - Metrics: Spearman rank IC each month + Sharpe of long-short portfolio

This is a HONESTLY DEGRADED test of the spec. Without point-in-time fundamentals (value, quality)
and alt-data overlays (Trends, sentiment, option-skew, short-interest, 13F), we test only the
price-derived sub-signal of the strategy. The spec calls for 70 features; we have 10. PASS bar
must therefore be regarded as a *sub-strategy floor*: if even the price-only sub-strategy can't
deliver rank IC >= 0.025 (t>2) and Sharpe >= 0.8, the full spec is unlikely to either.
"""

from __future__ import annotations

import json
import logging
import sys
import urllib.request
from io import StringIO
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

ROOT = Path("/Users/weeoto/turbo-carnival/research")
CACHE_PATH = ROOT / "data" / "cache" / "sp500_daily.parquet"
SPY_CACHE = ROOT / "data" / "cache" / "spy_daily.parquet"
TICKERS_CACHE = ROOT / "data" / "cache" / "sp500_tickers.csv"
LOG_PATH = ROOT / "logs" / "iter_14_backtest.log"
PNL_CSV = ROOT / "results" / "iter_14_monthly_pnl.csv"
SUMMARY_JSON = ROOT / "results" / "iter_14_summary.json"

START = "2014-01-01"
END = "2025-12-31"
OOS_START = pd.Timestamp("2020-01-01")
RETRAIN_FREQ_MONTHS = 12

# Feature horizons (trading days)
LB_MOM_LONG = 252  # 12m
LB_MOM_SKIP = 21   # skip last month for 12-1
LB_MOM_6M = 126
LB_REV = 21
LB_VOL = 60
LB_VOL_6M = 126
LB_ADV = 20
LB_HIGHER_MOMS = 60  # skew/kurt window

FWD_RET_DAYS = 21  # monthly forward return
DECILE_N = 50  # top/bottom 50 names
COST_BPS_PER_SIDE = 5.0
ANN_DAYS = 252


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_PATH, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("iter_14")


def get_sp500_tickers(log: logging.Logger) -> list[str]:
    if TICKERS_CACHE.exists():
        df = pd.read_csv(TICKERS_CACHE)
        log.info(f"Loaded cached tickers n={len(df)} from {TICKERS_CACHE}")
        return df["Symbol"].tolist()

    log.info("Scraping S&P 500 constituents from Wikipedia")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    tables = pd.read_html(StringIO(html))
    df = tables[0][["Symbol", "GICS Sector"]].copy()
    # yfinance uses '-' for class shares (BRK-B, BF-B); wiki uses '.'
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    TICKERS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(TICKERS_CACHE, index=False)
    log.info(f"Saved tickers n={len(df)} to {TICKERS_CACHE}")
    return df["Symbol"].tolist()


def fetch_prices(tickers: list[str], log: logging.Logger) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (close_df, volume_df) wide-format indexed by date, columns=ticker."""
    if CACHE_PATH.exists():
        log.info(f"Loading price cache {CACHE_PATH}")
        cache = pd.read_parquet(CACHE_PATH)
        close = cache.xs("Close", axis=1, level=0)
        volume = cache.xs("Volume", axis=1, level=0)
        log.info(
            f"Cache rows={len(close)} cols={close.shape[1]} "
            f"range={close.index.min().date()}..{close.index.max().date()}"
        )
        return close, volume

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading yfinance bulk OHLCV: {len(tickers)} tickers {START}..{END}")
    raw = yf.download(
        tickers,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="column",
    )
    # raw should be multi-index (column_type, ticker)
    if not isinstance(raw.columns, pd.MultiIndex):
        raise RuntimeError("Expected multi-index from yfinance bulk; got flat")
    close = raw["Close"].copy()
    volume = raw["Volume"].copy()
    # Drop columns that are all-NaN (delisted/missing tickers)
    close = close.dropna(how="all", axis=1)
    volume = volume.dropna(how="all", axis=1)
    # Align both to same columns
    common = close.columns.intersection(volume.columns)
    close = close[common]
    volume = volume[common]
    log.info(
        f"Downloaded close shape={close.shape} volume shape={volume.shape} "
        f"range={close.index.min().date()}..{close.index.max().date()}"
    )
    # Save combined parquet (need MultiIndex columns)
    combined = pd.concat({"Close": close, "Volume": volume}, axis=1)
    combined.to_parquet(CACHE_PATH)
    log.info(f"Saved {CACHE_PATH}")
    return close, volume


def fetch_spy(log: logging.Logger) -> pd.Series:
    if SPY_CACHE.exists():
        df = pd.read_parquet(SPY_CACHE)
        log.info(f"Loaded SPY cache rows={len(df)}")
        return df["Close"]

    SPY_CACHE.parent.mkdir(parents=True, exist_ok=True)
    log.info("Downloading SPY")
    raw = yf.download("SPY", start=START, end=END, auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        s = raw["Close"]["SPY"]
    else:
        s = raw["Close"]
    df = pd.DataFrame({"Close": s})
    df.to_parquet(SPY_CACHE)
    log.info(f"Saved SPY cache rows={len(df)}")
    return df["Close"]


def compute_features(close: pd.DataFrame, volume: pd.DataFrame, spy: pd.Series, log: logging.Logger) -> dict[str, pd.DataFrame]:
    """Return dict of feature name -> wide DataFrame (date x ticker)."""
    log.info("Computing features")
    ret = close.pct_change()
    spy_ret = spy.pct_change().reindex(ret.index)

    features: dict[str, pd.DataFrame] = {}

    # 1. 12-1 momentum: cumulative return over t-252..t-21
    # Equivalent: close.shift(LB_MOM_SKIP) / close.shift(LB_MOM_LONG) - 1
    mom_12_1 = close.shift(LB_MOM_SKIP) / close.shift(LB_MOM_LONG) - 1.0
    features["mom_12_1"] = mom_12_1

    # 2. 1-month reversal: trailing 21d return (use as negative-of-momentum signal at short end)
    features["rev_1m"] = close / close.shift(LB_REV) - 1.0

    # 3. Realized vol 60d
    features["vol_60d"] = ret.rolling(LB_VOL, min_periods=40).std() * np.sqrt(ANN_DAYS)

    # 4. Idio vol vs SPY 60d. Compute via residual stdev of rolling regression.
    # For each ticker: idio_vol = sqrt(var(r_i) - beta^2 * var(r_spy))
    # over 60d window, using realized covariance.
    var_spy = spy_ret.rolling(LB_VOL, min_periods=40).var()
    var_i = ret.rolling(LB_VOL, min_periods=40).var()
    # cov per ticker with SPY: (mean(r_i * r_spy) - mean(r_i)*mean(r_spy))
    spy_aligned = spy_ret.values[:, None]  # (T,1)
    cov_i_spy = (ret.mul(spy_ret, axis=0)).rolling(LB_VOL, min_periods=40).mean() - \
                ret.rolling(LB_VOL, min_periods=40).mean().mul(
                    spy_ret.rolling(LB_VOL, min_periods=40).mean(), axis=0)
    beta = cov_i_spy.div(var_spy, axis=0)
    idio_var = var_i - beta.pow(2).mul(var_spy, axis=0)
    idio_var = idio_var.clip(lower=0)
    features["idio_vol_60d"] = np.sqrt(idio_var) * np.sqrt(ANN_DAYS)

    # 5. Dollar volume 20d ADV (log)
    dollar_vol = (close * volume).rolling(LB_ADV, min_periods=10).mean()
    features["log_dollar_vol_20d"] = np.log(dollar_vol.replace(0, np.nan))

    # 6. Beta to SPY 60d (already computed)
    features["beta_60d"] = beta

    # 7. Skewness 60d
    features["skew_60d"] = ret.rolling(LB_HIGHER_MOMS, min_periods=40).skew()

    # 8. Kurtosis 60d
    features["kurt_60d"] = ret.rolling(LB_HIGHER_MOMS, min_periods=40).kurt()

    # 9. 6-month momentum
    features["mom_6m"] = close / close.shift(LB_MOM_6M) - 1.0

    # 10. 6-month vol
    features["vol_6m"] = ret.rolling(LB_VOL_6M, min_periods=80).std() * np.sqrt(ANN_DAYS)

    log.info(f"Features computed: {list(features.keys())}")
    return features


def get_month_end_dates(index: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    """Last trading day of each calendar month within [start, end]."""
    df = pd.DataFrame(index=index)
    df = df.loc[(df.index >= start) & (df.index <= end)]
    grouped = df.groupby([df.index.year, df.index.month])
    return [g.index.max() for _, g in grouped]


def build_panel(
    features: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    month_ends: list[pd.Timestamp],
    log: logging.Logger,
) -> pd.DataFrame:
    """Build long-format panel: rows = (date, ticker), columns = features + fwd_ret + ex_ret."""
    feat_names = list(features.keys())

    # Forward 21-day return computed at each month-end using close prices.
    # ret_fwd[t] = close[t + 21d] / close[t] - 1  (approximate; uses index-based shift)
    # We need to use actual trading days. Use index shift -21 on close.
    fwd_close = close.shift(-FWD_RET_DAYS)
    fwd_ret = fwd_close / close - 1.0

    rows = []
    for t in month_ends:
        if t not in close.index:
            continue
        # Get cross-section at t
        row = {"date": t}
        per_ticker = {}
        for name in feat_names:
            f_t = features[name].loc[t]
            per_ticker[name] = f_t
        fr_t = fwd_ret.loc[t]
        # Filter tickers that have all features non-NaN and forward return present
        df_t = pd.DataFrame(per_ticker)
        df_t["fwd_ret"] = fr_t
        df_t = df_t.dropna()
        if len(df_t) < 100:
            continue
        # Cross-sectional demeaning of forward return -> excess return
        df_t["ex_ret"] = df_t["fwd_ret"] - df_t["fwd_ret"].mean()
        df_t["date"] = t
        df_t["ticker"] = df_t.index
        rows.append(df_t.reset_index(drop=True))

    panel = pd.concat(rows, ignore_index=True)
    log.info(f"Panel built: {len(panel)} rows across {panel['date'].nunique()} month-ends")
    return panel


def cross_section_rank(s: pd.Series) -> pd.Series:
    """Cross-sectional rank scaled to [-1, +1]."""
    r = s.rank(pct=True)
    return 2.0 * r - 1.0


def train_and_predict(
    panel: pd.DataFrame,
    feat_names: list[str],
    log: logging.Logger,
) -> pd.DataFrame:
    """Walk-forward training. Return panel with added 'pred' column for OOS months."""
    panel = panel.copy()
    panel["pred"] = np.nan

    # Per-cross-section rank target (continuous)
    panel["rank_target"] = panel.groupby("date")["ex_ret"].transform(
        lambda x: cross_section_rank(x)
    )

    oos_dates = sorted([d for d in panel["date"].unique() if d >= OOS_START])
    if not oos_dates:
        raise RuntimeError("No OOS dates found")
    log.info(f"OOS dates: {len(oos_dates)} from {oos_dates[0].date()} to {oos_dates[-1].date()}")

    # Retrain calendar: group OOS dates into 12-month blocks
    # Train uses all dates strictly < earliest OOS date in block, minus 1-month embargo (drop last month).
    current_model = None
    current_block_start = None
    n_retrains = 0

    for oos_date in oos_dates:
        # Determine if we need to retrain: every 12 calendar months from first OOS date
        if current_block_start is None:
            need_retrain = True
        else:
            months_since_block = (oos_date.year - current_block_start.year) * 12 + (oos_date.month - current_block_start.month)
            need_retrain = months_since_block >= RETRAIN_FREQ_MONTHS

        if need_retrain:
            current_block_start = oos_date
            # Train data: dates < oos_date minus an embargo of FWD_RET_DAYS (so labels can't leak)
            # Embargo: drop the most recent month before oos_date so that the fwd_ret on the last
            # training cross-section doesn't overlap with current OOS month.
            train_cutoff = oos_date - pd.Timedelta(days=35)
            train = panel[panel["date"] < train_cutoff].copy()
            if len(train) < 1000:
                log.warning(f"  Skipping retrain at {oos_date.date()}: only {len(train)} train rows")
                continue
            X_train = train[feat_names].values
            y_train = train["rank_target"].values
            model = lgb.LGBMRegressor(
                num_leaves=63,
                learning_rate=0.02,
                n_estimators=400,
                min_data_in_leaf=200,
                feature_fraction=0.7,
                bagging_fraction=0.7,
                bagging_freq=1,
                lambda_l2=1.0,
                random_state=42,
                verbose=-1,
                n_jobs=-1,
            )
            model.fit(X_train, y_train)
            current_model = model
            n_retrains += 1
            log.info(
                f"  Retrain #{n_retrains} at OOS={oos_date.date()}: "
                f"train_rows={len(train)} train_dates={train['date'].nunique()} "
                f"cutoff<{train_cutoff.date()}"
            )

        if current_model is None:
            continue

        cs = panel[panel["date"] == oos_date]
        if len(cs) == 0:
            continue
        Xt = cs[feat_names].values
        preds = current_model.predict(Xt)
        panel.loc[cs.index, "pred"] = preds

    log.info(f"Total retrains: {n_retrains}")
    return panel


def compute_portfolio_pnl(panel: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """Compute monthly long-short P&L and rank IC.

    Returns DataFrame with columns: date, n_universe, rank_ic, long_ret, short_ret, ls_gross,
    cost, ls_net, cumulative
    """
    rows = []
    oos = panel[panel["pred"].notna()].copy()
    cumulative = 0.0
    prev_long: set[str] = set()
    prev_short: set[str] = set()

    for date, cs in oos.groupby("date"):
        cs = cs.dropna(subset=["pred", "fwd_ret"])
        if len(cs) < 100:
            continue
        cs = cs.sort_values("pred", ascending=False)
        # Rank IC: Spearman of pred vs realized ex_ret
        try:
            ic, _ = stats.spearmanr(cs["pred"], cs["ex_ret"])
        except Exception:
            ic = np.nan

        n = len(cs)
        # Top decile cap at DECILE_N; or 10% whichever smaller
        top_n = min(DECILE_N, max(10, n // 10))
        long_basket = set(cs.head(top_n)["ticker"].tolist())
        short_basket = set(cs.tail(top_n)["ticker"].tolist())

        long_ret = cs.head(top_n)["fwd_ret"].mean()
        short_ret = cs.tail(top_n)["fwd_ret"].mean()
        ls_gross = long_ret - short_ret

        # Turnover: fraction of names changed since prior month, on each side
        if prev_long:
            turnover_long = 1.0 - len(long_basket & prev_long) / top_n
            turnover_short = 1.0 - len(short_basket & prev_short) / top_n
        else:
            # First month: fully new positions on both sides
            turnover_long = 1.0
            turnover_short = 1.0
        # Cost: bps per name traded, applied to each side's notional fraction
        # Each side is 1.0x notional, so cost on side = 2 * turnover * top_n * (1/top_n) * cost_bps
        # = 2 * turnover * cost_bps (factor 2 for buy+sell of the traded fraction)
        cost_bps = (turnover_long + turnover_short) * COST_BPS_PER_SIDE
        cost = cost_bps / 1e4
        ls_net = ls_gross - cost
        cumulative += ls_net

        rows.append({
            "date": date,
            "n_universe": n,
            "rank_ic": ic,
            "top_n": top_n,
            "long_ret": long_ret,
            "short_ret": short_ret,
            "ls_gross": ls_gross,
            "turnover_long": turnover_long,
            "turnover_short": turnover_short,
            "cost": cost,
            "ls_net": ls_net,
            "cumulative_net": cumulative,
        })
        prev_long, prev_short = long_basket, short_basket

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    log.info(f"OOS months evaluated: {len(df)}")
    return df


def summarize(pnl: pd.DataFrame, log: logging.Logger) -> dict:
    months = len(pnl)
    if months == 0:
        return {"months_oos": 0, "verdict": "FAIL"}

    ics = pnl["rank_ic"].dropna().values
    mean_ic = float(np.mean(ics))
    # t-stat: mean / (std / sqrt(n))
    if len(ics) > 1:
        se = np.std(ics, ddof=1) / np.sqrt(len(ics))
        ic_t = float(mean_ic / se) if se > 0 else float("nan")
    else:
        ic_t = float("nan")

    # Newey-West 3-lag adjustment (per spec)
    if len(ics) > 4:
        def nw_se(x: np.ndarray, lags: int = 3) -> float:
            x = x - x.mean()
            n = len(x)
            gamma0 = np.dot(x, x) / n
            s = gamma0
            for k in range(1, lags + 1):
                w = 1.0 - k / (lags + 1)
                gk = np.dot(x[k:], x[:-k]) / n
                s += 2 * w * gk
            return float(np.sqrt(max(s, 1e-12) / n))
        nw = nw_se(ics, 3)
        ic_t_nw = float(mean_ic / nw) if nw > 0 else float("nan")
    else:
        ic_t_nw = float("nan")

    # Sharpe of monthly net L/S returns
    net = pnl["ls_net"].values
    mean_m = float(np.mean(net))
    std_m = float(np.std(net, ddof=1)) if len(net) > 1 else float("nan")
    ann_ret_pct = float(mean_m * 12 * 100)
    ann_vol_pct = float(std_m * np.sqrt(12) * 100) if std_m and not np.isnan(std_m) else float("nan")
    sharpe = float((mean_m / std_m) * np.sqrt(12)) if std_m and std_m > 0 else float("nan")

    # Max drawdown of cumulative net
    cum = pnl["cumulative_net"].values
    if len(cum) > 0:
        # cum is sum of monthly returns; convert to equity curve assuming additive returns on capital 1
        eq = 1.0 + cum
        roll_max = np.maximum.accumulate(eq)
        dd = (eq - roll_max) / roll_max
        max_dd_pct = float(np.min(dd) * 100)
    else:
        max_dd_pct = float("nan")

    pass_ic = (mean_ic >= 0.025) and (ic_t_nw is not None) and (not np.isnan(ic_t_nw)) and (ic_t_nw > 2.0)
    pass_sharpe = (sharpe is not None) and (not np.isnan(sharpe)) and (sharpe >= 0.8)
    verdict = "PASS" if (pass_ic and pass_sharpe) else "FAIL"

    summary = {
        "iter": "iter_14",
        "scope_note": "Simplified: S&P 500 proxy (not Russell 1000), price-only features (10 vs spec's 70), no fundamentals/alt-data",
        "months_oos": int(months),
        "mean_rank_ic": float(mean_ic),
        "ic_t_stat_simple": float(ic_t) if ic_t == ic_t else None,
        "ic_t_stat_newey_west_3lag": float(ic_t_nw) if ic_t_nw == ic_t_nw else None,
        "ic_t_stat": float(ic_t_nw) if ic_t_nw == ic_t_nw else (float(ic_t) if ic_t == ic_t else None),
        "ann_return_long_short_pct": ann_ret_pct,
        "ann_vol_pct": ann_vol_pct,
        "sharpe_oos": sharpe,
        "max_dd_pct": max_dd_pct,
        "mean_monthly_turnover_long": float(pnl["turnover_long"].mean()),
        "mean_monthly_turnover_short": float(pnl["turnover_short"].mean()),
        "mean_monthly_cost_bps": float(pnl["cost"].mean() * 1e4),
        "pass_ic_threshold": bool(pass_ic),
        "pass_sharpe_threshold": bool(pass_sharpe),
        "verdict": verdict,
        "criteria": "PASS iff mean_rank_ic>=0.025 AND ic_t_newey_west>2.0 AND sharpe_oos>=0.8",
    }
    log.info(f"Summary: {json.dumps(summary, default=str, indent=2)}")
    return summary


def main():
    log = setup_logging()
    log.info("=" * 80)
    log.info("iter_14 backtest start")
    log.info("=" * 80)

    tickers = get_sp500_tickers(log)
    close, volume = fetch_prices(tickers, log)
    spy = fetch_spy(log)

    # Align everything
    common_idx = close.index.intersection(volume.index).intersection(spy.index)
    close = close.loc[common_idx]
    volume = volume.loc[common_idx]
    spy = spy.loc[common_idx]
    log.info(f"Aligned shape: close={close.shape} spy={len(spy)}")

    features = compute_features(close, volume, spy, log)

    # Month-end dates within the full sample (need OOS_START..end)
    me = get_month_end_dates(
        close.index,
        pd.Timestamp("2015-01-01"),  # need ~1y warm-up for features before training
        close.index.max() - pd.Timedelta(days=FWD_RET_DAYS + 5),  # ensure fwd_ret available
    )
    log.info(f"Month-end dates: n={len(me)} from {me[0].date()} to {me[-1].date()}")

    panel = build_panel(features, close, me, log)
    feat_names = list(features.keys())

    panel = train_and_predict(panel, feat_names, log)
    pnl = compute_portfolio_pnl(panel, log)

    PNL_CSV.parent.mkdir(parents=True, exist_ok=True)
    pnl.to_csv(PNL_CSV, index=False)
    log.info(f"Wrote {PNL_CSV} rows={len(pnl)}")

    summary = summarize(pnl, log)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Wrote {SUMMARY_JSON}")

    # Final result line
    rl = (
        f"RESULT_LINE: iter_14 | months_oos={summary['months_oos']} | "
        f"rank_IC={summary['mean_rank_ic']:.4f} | "
        f"sharpe={summary['sharpe_oos']:.3f} | {summary['verdict']}"
    )
    log.info(rl)


if __name__ == "__main__":
    main()
