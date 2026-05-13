"""
iter_20 backtest: LLM Thematic ETF Basket Rotation
Spec: /Users/weeoto/turbo-carnival/research/candidates/iter_20_llm_thematic_basket_rotation.md

SIMPLIFIED APPROACH (no per-week LLM calls; price-momentum as proxy for "narrative momentum"):
- 20 thematic ETFs
- Weekly (Friday close) ranking by combined momentum:
    combined_rank = 0.5 * rank(4w_return) + 0.5 * rank(12w_return)
- Long top 5 ETFs, equal weight (long-only)
- Hold 4 weeks (monthly rebalance), then re-rank
- 5 bps per ETF per rebalance, applied on turnover
- IS 2019-2022, OOS 2023-2025

This tests the underlying "theme momentum" signal. An LLM ranker could plausibly improve over
price-only ranking by 10-30% (representing 60-70% of the upside if the price signal works).
A failing price proxy is suggestive (not proof) that LLM-driven narrative scoring would also
struggle; a passing price proxy is necessary-but-not-sufficient for the full LLM variant.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path("/Users/weeoto/turbo-carnival/research")
DATA_CACHE = ROOT / "data" / "cache" / "thematic_etfs_daily.parquet"
SPY_CACHE = ROOT / "data" / "cache" / "spy_daily.parquet"
LOG_PATH = ROOT / "logs" / "iter_20_backtest.log"
PNL_PATH = ROOT / "results" / "iter_20_monthly_pnl.csv"
SUMMARY_PATH = ROOT / "results" / "iter_20_summary.json"

UNIVERSE = [
    "BOTZ", "IBB", "URA", "ICLN", "BLOK",
    "ARKK", "SOXX", "JETS", "ITA", "COPX",
    "SLX", "TAN", "LIT", "KWEB", "FXI",
    "XBI", "XLE", "XLK", "XLF", "XLP",
]

START = "2019-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")
IS_END = "2022-12-31"
OOS_START = "2023-01-01"

LONG_N = 5
MOM_SHORT_WEEKS = 4
MOM_LONG_WEEKS = 12
HOLD_WEEKS = 4  # monthly rebalance
COST_BPS_PER_NAME = 5.0  # per name per rebalance (round-trip approximation per side)
TRADING_DAYS_PER_WEEK = 5
ANN_WEEKS = 52
ANN_MONTHS = 12

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
log = logging.getLogger("iter_20")


def fetch_prices(tickers: list[str], cache_path: Path, label: str) -> pd.DataFrame:
    """Daily adjusted closes (auto_adjust=True). Returns wide frame date x ticker."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        try:
            df = pd.read_parquet(cache_path)
            if isinstance(df.index, pd.DatetimeIndex):
                # Always refresh if cache is stale by more than ~3 days vs target END.
                latest = df.index.max()
                target_end = pd.Timestamp(END)
                if (target_end - latest).days <= 5:
                    log.info(f"Loaded cached {label} prices: shape={df.shape}, "
                             f"range={df.index.min().date()}..{df.index.max().date()}")
                    return df
                else:
                    log.info(f"{label} cache stale (latest={latest.date()}, target={target_end.date()}); refetching")
        except Exception as e:
            log.warning(f"{label} cache read failed: {e}; refetching")

    log.info(f"Downloading {label} from yfinance: {len(tickers)} tickers, {START}..{END}")
    raw = yf.download(
        tickers=tickers,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    closes: dict[str, pd.Series] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            try:
                s = raw[t]["Close"]
                if s.notna().sum() > 0:
                    closes[t] = s
            except Exception as e:
                log.warning(f"  Failed to extract {t}: {e}")
    else:
        # single-ticker fallback
        closes[tickers[0]] = raw["Close"]

    df = pd.DataFrame(closes).sort_index().dropna(how="all")
    df.to_parquet(cache_path)
    log.info(f"Cached {label} prices to {cache_path}: shape={df.shape}")
    return df


def to_weekly_friday(daily: pd.DataFrame) -> pd.DataFrame:
    """Resample to Friday close (last trading day of each ISO week)."""
    # 'W-FRI' aligns to Fridays; .last() picks the last available close in the week.
    wk = daily.resample("W-FRI").last()
    # Drop weeks entirely missing
    wk = wk.dropna(how="all")
    return wk


def compute_momentum_signals(weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Returns (mom_4w, mom_12w, combined_rank) DataFrames indexed by Friday."""
    # Simple total return over the window
    mom_4w = weekly / weekly.shift(MOM_SHORT_WEEKS) - 1.0
    mom_12w = weekly / weekly.shift(MOM_LONG_WEEKS) - 1.0

    # Cross-sectional ranks (higher = better); pct ranks 0..1
    r_4w = mom_4w.rank(axis=1, pct=True)
    r_12w = mom_12w.rank(axis=1, pct=True)

    combined = 0.5 * r_4w + 0.5 * r_12w
    return mom_4w, mom_12w, combined


def build_weekly_targets(combined_rank: pd.DataFrame) -> pd.DataFrame:
    """For each Friday in the rebalance schedule (every HOLD_WEEKS weeks once enough
    history exists), pick top LONG_N tickers from the combined rank and assign
    equal weight 1/LONG_N. Between rebalances, weights are held (set NaN here,
    forward-filled later when applied to daily returns).

    Returns wide DataFrame indexed by rebalance Friday, cols = tickers,
    rows are zeros + LONG_N entries of 1/LONG_N.
    """
    fridays = combined_rank.index
    # Establish first valid Friday: need MOM_LONG_WEEKS of history => first valid index is the row
    # where mom_12w is not all NaN.
    first_valid = combined_rank.dropna(how="all").index
    if len(first_valid) == 0:
        raise RuntimeError("No valid combined-rank rows; check data.")
    first_idx = combined_rank.index.get_loc(first_valid[0])

    # Rebalance every HOLD_WEEKS Fridays starting from the first valid one
    rebalance_idx = list(range(first_idx, len(fridays), HOLD_WEEKS))
    rebalance_fridays = fridays[rebalance_idx]

    weights = pd.DataFrame(0.0, index=rebalance_fridays, columns=combined_rank.columns)
    for dt in rebalance_fridays:
        row = combined_rank.loc[dt].dropna()
        if len(row) < LONG_N:
            # Not enough names this Friday; skip (leave zeros — means flat that period).
            continue
        top = row.sort_values(ascending=False).index[:LONG_N]
        weights.loc[dt, top] = 1.0 / LONG_N

    return weights


def run_backtest(daily: pd.DataFrame, weights_rebalance: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Simulate daily, return:
      daily_port: daily portfolio return (net of costs)
      monthly_port: monthly-resampled portfolio return (net)
      monthly_gross: monthly portfolio return before costs
      monthly_cost: monthly cost drag
      turnover_series: turnover at each rebalance Friday
    """
    daily = daily.copy().sort_index()
    daily_ret = daily.pct_change()

    # Forward-fill weights from each rebalance Friday until the next.
    # Note: positions are *entered* at the close of the rebalance Friday, so the
    # first return earned is the next trading day's return. Use shift(1) accordingly.
    aligned = weights_rebalance.reindex(daily.index, method="ffill")
    # shift by 1 so the position returns are earned the day AFTER the close-of-Friday signal
    aligned_lagged = aligned.shift(1).fillna(0.0)

    gross_daily = (aligned_lagged * daily_ret).sum(axis=1)

    # Costs: at each rebalance Friday, compute |w_new - w_old| total turnover,
    # multiply by cost bps. Attribute cost on the rebalance day itself (next trading day).
    prev_w = weights_rebalance.shift(1).fillna(0.0)
    turnover = (weights_rebalance - prev_w).abs().sum(axis=1)
    # Cost as fraction of NAV: turnover * bps. Each rebalance touches sum_i |dw_i| of NAV.
    cost_per_rebalance = turnover * (COST_BPS_PER_NAME * 1e-4)

    # Attribute the cost on the trading day immediately after the rebalance Friday
    cost_daily = pd.Series(0.0, index=daily.index)
    for fri, cost in cost_per_rebalance.items():
        if cost <= 0:
            continue
        # Find next trading day in daily index
        candidates = daily.index[daily.index > fri]
        if len(candidates) == 0:
            continue
        attribution_day = candidates[0]
        cost_daily.loc[attribution_day] = cost_daily.loc[attribution_day] + cost

    net_daily = gross_daily - cost_daily

    # Monthly aggregation (calendar month end) for reporting
    daily_index = daily.index
    monthly_gross = (1 + gross_daily).resample("ME").prod() - 1
    monthly_cost = cost_daily.resample("ME").sum()
    monthly_net = (1 + net_daily).resample("ME").prod() - 1
    # Drop incomplete trailing month if we don't have at least 5 trading days in it
    last_month = monthly_net.index[-1]
    last_month_days = (daily_index.to_series().dt.to_period("M") == last_month.to_period("M")).sum()
    if last_month_days < 5:
        monthly_net = monthly_net.iloc[:-1]
        monthly_gross = monthly_gross.iloc[:-1]
        monthly_cost = monthly_cost.iloc[:-1]

    return net_daily, monthly_net, monthly_gross, monthly_cost, turnover


def perf_metrics(monthly_returns: pd.Series, label: str) -> dict:
    r = monthly_returns.dropna()
    n = int(len(r))
    if n == 0:
        return {"months": 0, "ann_return_pct": 0.0, "ann_vol_pct": 0.0,
                "sharpe": 0.0, "max_dd_pct": 0.0, "win_rate_monthly": 0.0}
    mu_ann = r.mean() * ANN_MONTHS
    sd_ann = r.std() * np.sqrt(ANN_MONTHS)
    sharpe = float(mu_ann / sd_ann) if sd_ann > 0 else 0.0
    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = float((cum / peak - 1).min())
    win_rate = float((r > 0).mean())
    out = {
        "months": n,
        "ann_return_pct": float(mu_ann * 100),
        "ann_vol_pct": float(sd_ann * 100),
        "sharpe": sharpe,
        "max_dd_pct": float(dd * 100),
        "win_rate_monthly": win_rate,
    }
    log.info(f"{label}: months={n}, ann_ret={out['ann_return_pct']:.2f}%, "
             f"ann_vol={out['ann_vol_pct']:.2f}%, sharpe={sharpe:.3f}, "
             f"max_dd={out['max_dd_pct']:.2f}%, win_rate={win_rate:.3f}")
    return out


def spy_buy_hold_monthly(spy_daily: pd.Series, oos_start: pd.Timestamp, oos_end: pd.Timestamp) -> pd.Series:
    """SPY monthly returns over [oos_start, oos_end]."""
    spy_d = spy_daily.copy().sort_index()
    daily_ret = spy_d.pct_change().dropna()
    monthly = (1 + daily_ret).resample("ME").prod() - 1
    return monthly[(monthly.index >= oos_start) & (monthly.index <= oos_end)]


def main():
    log.info("=" * 80)
    log.info("iter_20 backtest START (price-momentum proxy for LLM narrative momentum)")
    log.info(f"Universe ({len(UNIVERSE)}): {UNIVERSE}")
    log.info(f"Window: {START} .. {END}")
    log.info(f"IS end: {IS_END}, OOS start: {OOS_START}")
    log.info(f"Signal: 0.5 * rank(4w_ret) + 0.5 * rank(12w_ret) (Friday close)")
    log.info(f"Long top {LONG_N}, equal weight; hold {HOLD_WEEKS} weeks; "
             f"cost {COST_BPS_PER_NAME} bps/name/rebalance on turnover")
    log.info("=" * 80)

    daily = fetch_prices(UNIVERSE, DATA_CACHE, "thematic ETFs")
    # Drop tickers with insufficient coverage (< 200 trading days)
    keep = [c for c in daily.columns if daily[c].notna().sum() > 200]
    dropped = [c for c in daily.columns if c not in keep]
    if dropped:
        log.warning(f"Dropping sparse tickers: {dropped}")
    daily = daily[keep]
    log.info(f"Active tickers ({len(keep)}): {keep}")
    log.info(f"Coverage by ticker (non-NaN count):")
    for c in keep:
        first = daily[c].first_valid_index()
        last = daily[c].last_valid_index()
        log.info(f"  {c:6s}: {int(daily[c].notna().sum())} days, "
                 f"{first.date() if first is not None else 'n/a'} -> "
                 f"{last.date() if last is not None else 'n/a'}")

    spy_daily_df = fetch_prices(["SPY"], SPY_CACHE, "SPY")
    spy = spy_daily_df["SPY"]

    weekly = to_weekly_friday(daily)
    log.info(f"Weekly Friday frame: shape={weekly.shape}, "
             f"range={weekly.index.min().date()}..{weekly.index.max().date()}")

    mom_4w, mom_12w, combined = compute_momentum_signals(weekly)
    log.info(f"Momentum signals computed. First valid 12w mom row: "
             f"{mom_12w.dropna(how='all').index.min().date()}")

    weights = build_weekly_targets(combined)
    log.info(f"Rebalance schedule: {len(weights)} Fridays, "
             f"first={weights.index.min().date()}, last={weights.index.max().date()}")
    # Show a few sample rebalances
    sample_idx = [0, len(weights) // 2, len(weights) - 1] if len(weights) >= 3 else list(range(len(weights)))
    for i in sample_idx:
        if i < 0 or i >= len(weights):
            continue
        dt = weights.index[i]
        top = weights.loc[dt][weights.loc[dt] > 0].index.tolist()
        log.info(f"  rebalance@{dt.date()} top-{LONG_N}: {top}")

    net_daily, monthly_net, monthly_gross, monthly_cost, turnover = run_backtest(daily, weights)

    # Save monthly PnL
    pnl_df = pd.DataFrame({
        "gross_return": monthly_gross,
        "cost": monthly_cost,
        "net_return": monthly_net,
    })
    pnl_df.index.name = "date"
    pnl_df = pnl_df.dropna(subset=["net_return"])
    PNL_PATH.parent.mkdir(parents=True, exist_ok=True)
    pnl_df.to_csv(PNL_PATH)
    log.info(f"Saved monthly PnL to {PNL_PATH}: {len(pnl_df)} rows")

    # IS / OOS split
    is_end_ts = pd.Timestamp(IS_END)
    oos_start_ts = pd.Timestamp(OOS_START)
    is_returns = pnl_df.loc[pnl_df.index <= is_end_ts, "net_return"]
    oos_returns = pnl_df.loc[pnl_df.index >= oos_start_ts, "net_return"]

    is_metrics = perf_metrics(is_returns, "IS 2019-2022")
    oos_metrics = perf_metrics(oos_returns, "OOS 2023-2025")

    # SPY buy-and-hold benchmark over the same OOS window
    if len(oos_returns) > 0:
        oos_first = oos_returns.index.min()
        oos_last = oos_returns.index.max()
        spy_oos = spy_buy_hold_monthly(spy, oos_first, oos_last)
        spy_metrics = perf_metrics(spy_oos, "SPY OOS 2023-2025")
    else:
        spy_metrics = perf_metrics(pd.Series(dtype=float), "SPY OOS (empty)")
        spy_oos = pd.Series(dtype=float)

    # Average turnover and rebalance count
    rebalance_count = int((turnover > 0).sum())
    avg_turnover = float(turnover[turnover > 0].mean()) if rebalance_count > 0 else 0.0
    log.info(f"Rebalances={rebalance_count}, avg turnover (sum |dw|)={avg_turnover:.3f}")

    # Verdict
    sharpe_oos = oos_metrics["sharpe"]
    sharpe_spy = spy_metrics["sharpe"]
    sharpe_gap = sharpe_oos - sharpe_spy
    ann_alpha_pct = oos_metrics["ann_return_pct"] - spy_metrics["ann_return_pct"]
    passes = (sharpe_oos >= 0.7) and (sharpe_gap >= 0.3)
    verdict = "PASS" if passes else "FAIL"

    summary = {
        "iter": "iter_20",
        "method": "price_momentum_proxy_for_llm_narrative",
        "universe": list(daily.columns),
        "n_instruments": int(len(daily.columns)),
        "start": START,
        "end": END,
        "is_end": IS_END,
        "oos_start": OOS_START,
        "signal": "0.5 * rank(4w_return) + 0.5 * rank(12w_return)",
        "rebalance_freq_weeks": HOLD_WEEKS,
        "long_n": LONG_N,
        "cost_bps_per_name": COST_BPS_PER_NAME,
        "rebalance_count_total": int(len(weights)),
        "rebalance_count_with_positions": rebalance_count,
        "avg_turnover_sum_abs_dw": avg_turnover,
        "months_is": is_metrics["months"],
        "months_oos": oos_metrics["months"],
        "ann_return_pct": oos_metrics["ann_return_pct"],
        "ann_vol_pct": oos_metrics["ann_vol_pct"],
        "sharpe_is": is_metrics["sharpe"],
        "sharpe_oos": sharpe_oos,
        "max_dd_pct": oos_metrics["max_dd_pct"],
        "win_rate_monthly": oos_metrics["win_rate_monthly"],
        "spy_oos_ann_return_pct": spy_metrics["ann_return_pct"],
        "spy_oos_sharpe": sharpe_spy,
        "spy_oos_max_dd_pct": spy_metrics["max_dd_pct"],
        "vs_spy_alpha": ann_alpha_pct,
        "vs_spy_sharpe_gap": sharpe_gap,
        "verdict": verdict,
        "verdict_rule": "PASS if sharpe_oos >= 0.7 AND sharpe_oos - sharpe_spy >= 0.3",
        "caveats": [
            "Price-momentum proxy (NOT an LLM). An LLM-driven narrative ranker COULD do better OR worse.",
            "Spec calls for weekly LLM scoring on assembled news + earnings + social corpus.",
            "yfinance close-to-close fills; no intraday slippage modeling beyond 5 bps/name flat.",
            "Single IS/OOS split (no walk-forward), 2019-2022 IS / 2023-onward OOS.",
            "Equal-weight long-only top-5 (short sleeve from spec is OFF; spec also defaults short OFF).",
            "Universe is 20 thematic ETFs as specified by the runner (subset of spec's 21).",
        ],
    }
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Saved summary to {SUMMARY_PATH}")

    print("\n" + "=" * 80)
    print(f"RESULT_LINE: iter_20 (price proxy) | sharpe={sharpe_oos:.3f} | "
          f"ann_ret={oos_metrics['ann_return_pct']:.2f}% | "
          f"vs SPY alpha={ann_alpha_pct:.2f}% (sharpe gap={sharpe_gap:.3f}) | VERDICT={verdict}")
    print("CAVEAT: Price-momentum proxy. LLM-driven version COULD do better OR worse - needs separate test.")
    print(f"PATH: {SUMMARY_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
