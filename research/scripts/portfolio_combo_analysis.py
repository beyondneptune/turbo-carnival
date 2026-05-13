"""Portfolio combination analysis for tonight's three surviving edges.

Combines:
  1. SVXY aggressive (100% / threshold 1.0 / no CB / 2019-2026): +17.01% / Sharpe 0.66
  2. iter_3 BTC funding cash-and-carry: +7.95% ann net
  3. iter_18 ADR clean pairs (TCOM/9961 + BIDU/9888 only)

Approach:
  - Reconstruct daily P&L for each strategy over 2021-01-01 .. 2025-12-31
    (intersection where all three have data).
  - Normalize each to constant 10% annualized vol (so allocation is fair).
  - Equal-weight 1/3 each on a $1M paper allocation.
  - Compute individual + combined Sharpe, max DD, correlations, diversification uplift.

No fabrication: if a strategy has no signal in a window, that day's pnl = 0
(genuine "flat" days are real).
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
RESULTS = REPO / "research" / "results"
LOGS = REPO / "research" / "logs"
LOGS.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOGS / "portfolio_combo.log"
SUMMARY_PATH = RESULTS / "portfolio_combo_summary.json"

START = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2025-12-31")
TARGET_ANN_VOL = 0.10
NOTIONAL = 1_000_000.0
EW = 1.0 / 3.0
TRADING_DAYS = 252

# ----------------------------------------------------------------------------
# logging
# ----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("portfolio_combo")


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def daily_business_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end, freq="B")


def normalize_to_vol(series: pd.Series, target: float = TARGET_ANN_VOL) -> tuple[pd.Series, float]:
    realized = series.std() * np.sqrt(TRADING_DAYS)
    if realized <= 0 or np.isnan(realized):
        return series.copy(), 1.0
    k = target / realized
    return series * k, float(k)


def perf_stats(returns: pd.Series, label: str) -> dict:
    if returns.std() <= 0:
        sharpe = 0.0
    else:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS))
    ann_ret = float((1 + returns).prod() ** (TRADING_DAYS / max(len(returns), 1)) - 1)
    equity = (1 + returns).cumprod()
    dd = equity / equity.cummax() - 1
    max_dd = float(dd.min())
    log.info(
        "%-20s | sharpe=%6.3f | ann_ret=%+.2f%% | max_dd=%+.2f%% | n=%d | vol=%.2f%%",
        label, sharpe, ann_ret * 100, max_dd * 100, len(returns), returns.std() * np.sqrt(TRADING_DAYS) * 100,
    )
    return {
        "sharpe": sharpe,
        "ann_ret": ann_ret,
        "max_dd": max_dd,
        "n_days": int(len(returns)),
        "ann_vol": float(returns.std() * np.sqrt(TRADING_DAYS)),
    }


# ----------------------------------------------------------------------------
# Strategy 1: SVXY aggressive — re-use parquet from svxy_lab
# ----------------------------------------------------------------------------
def load_svxy_aggressive() -> pd.Series:
    """Daily strategy_ret series filtered to [2021-01-01, 2025-12-31]."""
    pq = REPO / "experiments" / "svxy_lab" / "runs" / "walk_forward_results.parquet"
    log.info("SVXY: reading %s", pq)
    df = pd.read_parquet(pq)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.set_index("date").sort_index()
    s = df["strategy_ret"].astype(float)
    s = s.loc[(s.index >= START) & (s.index <= END)]
    log.info("SVXY: %d rows, %s .. %s", len(s), s.index.min(), s.index.max())
    return s.rename("svxy_aggressive")


# ----------------------------------------------------------------------------
# Strategy 2: BTC funding cash-and-carry
# ----------------------------------------------------------------------------
def load_btc_funding() -> pd.Series:
    """Reconstruct daily P&L from BTC funding history during open positions.

    Strategy = perp short + spot long (delta-neutral). When perp funding > 0,
    short side receives funding. Daily P&L = sum of the 3 funding ticks that day
    (8h ticks * 3 = 1 day). When out of position, pnl=0.

    We treat the trades from iter_3_positions.csv as the on/off mask and the
    iter_3_funding_history.csv as the actual per-tick rates. This gives real
    daily variance (funding rates fluctuate hour-to-hour).

    Round-trip cost (30 bps) is amortized at trade entry across the holding
    window — matches the iter_3 net_carry construction.
    """
    pos_fp = RESULTS / "iter_3_positions.csv"
    fund_fp = RESULTS / "iter_3_funding_history.csv"
    log.info("BTC funding: positions=%s, history=%s", pos_fp.name, fund_fp.name)

    pos = pd.read_csv(pos_fp)
    pos["entry_dt"] = pd.to_datetime(pos["entry_dt"], utc=True, format="ISO8601").dt.tz_convert(None)
    pos["exit_dt"] = pd.to_datetime(pos["exit_dt"], utc=True, format="ISO8601").dt.tz_convert(None)

    fh = pd.read_csv(fund_fp)
    fh["datetime"] = pd.to_datetime(fh["datetime"], utc=True, format="ISO8601").dt.tz_convert(None)
    fh["date"] = fh["datetime"].dt.normalize()
    fh = fh.set_index("datetime").sort_index()

    idx = daily_business_index(START, END)
    series = pd.Series(0.0, index=idx, dtype=float)

    n_positions_in_window = 0
    for _, r in pos.iterrows():
        e, x = r["entry_dt"], r["exit_dt"]
        # cap to window
        e_clip = max(e, START.tz_localize(None) if hasattr(START, "tz_localize") and START.tz else START)
        x_clip = min(x, END)
        if e_clip > x_clip:
            continue
        n_positions_in_window += 1

        # funding ticks within [e_clip, x_clip]: short side EARNS the funding rate
        ticks = fh.loc[(fh.index >= e_clip) & (fh.index <= x_clip), "funding_rate"]
        if ticks.empty:
            continue
        # group ticks by business day (use floor to nearest business day)
        by_day = ticks.copy()
        by_day.index = by_day.index.normalize()
        daily_funding = by_day.groupby(by_day.index).sum()

        # amortize 30 bps round-trip cost across the holding business days
        days_held = pd.bdate_range(e_clip.normalize(), x_clip.normalize(), freq="B")
        days_held = days_held.intersection(idx)
        n_days = max(len(days_held), 1)
        cost_per_day = -0.003 / n_days  # 30 bps = 0.003

        # reindex daily_funding to business days
        daily_funding = daily_funding.reindex(idx).fillna(0.0)
        # add cost only on actual holding days
        for d in days_held:
            series.loc[d] += float(daily_funding.loc[d]) + cost_per_day

    log.info("BTC funding: %d positions in window, %d non-zero days, total_ret=%+.2f%%",
             n_positions_in_window, (series != 0).sum(), 100 * ((1 + series).prod() - 1))
    return series.rename("btc_funding")


# ----------------------------------------------------------------------------
# Strategy 3: ADR clean pairs (TCOM/9961, BIDU/9888 only)
# ----------------------------------------------------------------------------
def load_adr_clean() -> pd.Series:
    """Daily P&L for the two clean pairs only.

    iter_18_trades.csv has individual closed trades with net_bps per trade. For
    each trade, spread net_bps (basis points = pct*100) over the holding
    business days. Sum across pairs and trades per day.

    bps to fractional return: bps / 10000.
    """
    fp = RESULTS / "iter_18_trades.csv"
    log.info("ADR clean: reading %s", fp)
    tr = pd.read_csv(fp)
    keep_pairs = {"TCOM/9961.HK", "BIDU/9888.HK"}
    tr = tr[tr["pair"].isin(keep_pairs)].copy()
    log.info("ADR clean: %d trades after filter (TCOM/9961, BIDU/9888)", len(tr))

    tr["entry_date"] = pd.to_datetime(tr["entry_date"]).dt.normalize()
    tr["exit_date"] = pd.to_datetime(tr["exit_date"]).dt.normalize()
    tr["net_ret"] = tr["net_bps"].astype(float) / 10_000.0

    idx = daily_business_index(START, END)
    series = pd.Series(0.0, index=idx, dtype=float)

    # Each trade is a fraction of capital. Since we have two pairs running
    # potentially simultaneously, we equal-weight 50% each within the ADR
    # sleeve so the sleeve doesn't double-stack. Realize P&L on the exit day
    # (most trades hold 1-2 bars; lump-realized matches a discrete trade book
    # and preserves real daily variance, which uniform-smearing destroys).
    n_clipped = 0
    pair_weight = 0.5
    for _, r in tr.iterrows():
        x = r["exit_date"]
        if x < START or x > END:
            continue
        ret = float(r["net_ret"]) * pair_weight
        # find next business day on or after exit (in case exit fell on weekend)
        # iter_18 exits are bar-aligned so this is typically exact.
        if x in idx:
            d = x
        else:
            after = idx[idx >= x]
            if len(after) == 0:
                continue
            d = after[0]
        n_clipped += 1
        series.loc[d] += ret

    log.info("ADR clean: %d trades within window, %d non-zero days", n_clipped, (series != 0).sum())
    series = series.rename("adr_clean")
    return series


# ----------------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------------
def main():
    log.info("=" * 70)
    log.info("Portfolio combination analysis — tonight's surviving edges")
    log.info("Window: %s .. %s  |  vol-target=%.0f%%  |  ew=%.4f  |  notional=$%s",
             START.date(), END.date(), TARGET_ANN_VOL * 100, EW, f"{NOTIONAL:,.0f}")
    log.info("=" * 70)

    # ---- load strategies ----
    svxy = load_svxy_aggressive()
    btc = load_btc_funding()
    adr = load_adr_clean()

    idx = daily_business_index(START, END)
    df = pd.DataFrame(index=idx)
    df["svxy_aggressive"] = svxy.reindex(idx).fillna(0.0)
    df["btc_funding"] = btc.reindex(idx).fillna(0.0)
    df["adr_clean"] = adr.reindex(idx).fillna(0.0)

    log.info("Aligned matrix: %s", df.shape)
    log.info("Non-zero counts: %s", (df != 0).sum().to_dict())

    # ---- raw stats (before vol normalization) ----
    log.info("--- raw per-strategy stats (before vol normalization) ---")
    raw_stats = {}
    for col in df.columns:
        raw_stats[col] = perf_stats(df[col], f"raw {col}")

    # ---- vol-normalize each to 10% ann vol ----
    log.info("--- vol-normalized per-strategy stats (target=%.0f%% ann vol) ---", TARGET_ANN_VOL * 100)
    df_n = pd.DataFrame(index=idx)
    scales = {}
    norm_stats = {}
    for col in df.columns:
        s, k = normalize_to_vol(df[col])
        df_n[col] = s
        scales[col] = k
        norm_stats[col] = perf_stats(s, f"normalized {col}")
    log.info("Vol-normalization scales: %s", scales)

    # ---- correlation matrix (on normalized series) ----
    corr = df_n.corr()
    log.info("--- correlation matrix (normalized) ---")
    log.info("\n%s", corr.round(4).to_string())

    # ---- combined portfolio (equal weight, 1/3 each) ----
    combined = (df_n * EW).sum(axis=1).rename("combined")
    combined_stats = perf_stats(combined, "COMBINED (1/3 each)")

    # ---- monthly distribution ----
    monthly = (1 + combined).resample("ME").prod() - 1
    monthly_summary = {
        "best_month_pct": float(monthly.max() * 100) if len(monthly) else 0.0,
        "worst_month_pct": float(monthly.min() * 100) if len(monthly) else 0.0,
        "mean_month_pct": float(monthly.mean() * 100) if len(monthly) else 0.0,
        "median_month_pct": float(monthly.median() * 100) if len(monthly) else 0.0,
        "std_month_pct": float(monthly.std() * 100) if len(monthly) else 0.0,
        "pct_positive_months": float((monthly > 0).mean() * 100) if len(monthly) else 0.0,
        "n_months": int(len(monthly)),
    }
    log.info("Monthly summary: %s", monthly_summary)

    # ---- diversification uplift ----
    individual_sharpes = [norm_stats[c]["sharpe"] for c in df.columns]
    weighted_avg_sharpe = float(np.mean(individual_sharpes))
    max_individual_sharpe = float(max(individual_sharpes))
    div_uplift_vs_max = combined_stats["sharpe"] - max_individual_sharpe
    div_uplift_vs_avg = combined_stats["sharpe"] - weighted_avg_sharpe
    log.info("Diversification: combined_sharpe=%.3f | max_individual=%.3f | uplift_vs_max=%+.3f | uplift_vs_avg=%+.3f",
             combined_stats["sharpe"], max_individual_sharpe, div_uplift_vs_max, div_uplift_vs_avg)

    # ---- dollar P&L on $1M ----
    final_equity = float(NOTIONAL * (1 + combined).prod())
    total_pnl = final_equity - NOTIONAL
    log.info(
        f"$1M notional final equity: ${final_equity:,.0f}  "
        f"(P&L=${total_pnl:,.0f}, {100 * total_pnl / NOTIONAL:.1f}%)"
    )

    # ---- save summary ----
    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "window": {"start": str(START.date()), "end": str(END.date())},
        "method": {
            "vol_normalize_target": TARGET_ANN_VOL,
            "equal_weight": EW,
            "notional_usd": NOTIONAL,
            "trading_days_per_year": TRADING_DAYS,
        },
        "data_sources": {
            "svxy_aggressive": "experiments/svxy_lab/runs/walk_forward_results.parquet (strategy_ret column)",
            "btc_funding": "research/results/iter_3_positions.csv (on/off mask) + iter_3_funding_history.csv (daily sum of 3x8h funding ticks during open positions, 30bps round-trip cost amortized over holding days)",
            "adr_clean": "research/results/iter_18_trades.csv (filtered to TCOM/9961.HK + BIDU/9888.HK, net_bps realized on exit date, 50% per pair within sleeve)",
        },
        "caveats": {
            "btc_funding_sharpe_high": "Raw daily P&L for BTC cash-and-carry has very low variance (funding ticks are small + deterministic in sign when active), producing a high daily Sharpe. The iter_3 summary's sharpe_proxy of 0.62 was computed on per-position carry returns (n=11) — a coarser metric. Real-world execution would add basis-risk variance not present here.",
            "vol_normalization_amplifies_btc": "Vol-normalizing the low-noise BTC series to 10% ann vol requires ~12x scale; this is a model artifact of the smooth-carry assumption, not a realistic deployable leverage level.",
            "adr_realization": "ADR P&L is realized lump-sum on the trade exit day. Most trades hold 1-2 bars so this is close to truth.",
        },
        "per_strategy_raw": raw_stats,
        "per_strategy_normalized": norm_stats,
        "vol_normalization_scales": scales,
        "correlation_matrix": corr.round(6).to_dict(),
        "combined_portfolio": {
            "sharpe": combined_stats["sharpe"],
            "ann_ret": combined_stats["ann_ret"],
            "max_dd": combined_stats["max_dd"],
            "ann_vol": combined_stats["ann_vol"],
            "n_days": combined_stats["n_days"],
            "monthly_return_distribution_summary": monthly_summary,
            "dollar_pnl_on_notional": {
                "notional": NOTIONAL,
                "final_equity": final_equity,
                "total_pnl": total_pnl,
                "total_return_pct": 100 * total_pnl / NOTIONAL,
            },
        },
        "diversification_uplift": {
            "combined_sharpe": combined_stats["sharpe"],
            "max_individual_sharpe": max_individual_sharpe,
            "uplift_vs_max": div_uplift_vs_max,
            "weighted_avg_individual_sharpe": weighted_avg_sharpe,
            "uplift_vs_avg": div_uplift_vs_avg,
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    log.info("wrote summary: %s", SUMMARY_PATH)

    # ---- terminal report ----
    print("\n" + "=" * 70)
    print("REPORT")
    print("=" * 70)
    print(
        f"PORTFOLIO_LINE: Combined Sharpe={combined_stats['sharpe']:.3f} | "
        f"Combined ann_ret={combined_stats['ann_ret']*100:+.2f}% | "
        f"Combined max_dd={combined_stats['max_dd']*100:+.2f}% | "
        f"Diversification uplift={div_uplift_vs_max:+.3f}"
    )
    pair_labels = [
        ("svxy_aggressive", "btc_funding", "SVXY <-> BTC"),
        ("svxy_aggressive", "adr_clean", "SVXY <-> ADR"),
        ("btc_funding", "adr_clean", "BTC <-> ADR"),
    ]
    print("CORRELATION_SUMMARY:")
    for a, b, label in pair_labels:
        print(f"  {label}: {corr.loc[a, b]:+.3f}")
    print(
        f"MONTHLY_BEST_WORST: Best month={monthly_summary['best_month_pct']:+.2f}%, "
        f"Worst month={monthly_summary['worst_month_pct']:+.2f}%, "
        f"% positive months={monthly_summary['pct_positive_months']:.1f}%"
    )
    print(f"PATH: {SUMMARY_PATH}")
    return summary


if __name__ == "__main__":
    main()
