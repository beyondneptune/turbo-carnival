"""Forward walk-forward — Agent A22.

Tests the 4-strategy combined portfolio on 2025-ONLY data so we can see
whether the prior 2021-2025 combo Sharpe is a real edge stack or overfit
theater.

The 4 surviving edges this evening:
  1. SVXY sweet spot:   thr=1.00 / size=25% / no CB (from svxy_parameter_sweep)
     -- the prior 2019-2025 SVXY run used the AGGRESSIVE 100% variant. The
        "sweet spot" is the 25% size variant with the same threshold; that is
        the deployable config. Re-running on 2025+ panel data here.
  2. iter_3 BTC funding: cash-and-carry, daily P&L from funding history.
  3. iter_18 CLEANUP ADR: ALL 5 dual-listing pairs (BABA, JD, BIDU, NTES, TCOM).
     -- prior combo only used the 2 "clean" pairs (TCOM, BIDU) from the
        FAIL-era iter_18_trades.csv. The CLEANUP run fixed NTES ratio and
        ex-div, producing 5 passing pairs in iter_18_cleanup_trades.csv.
  4. iter_6 Nikkei daily proxy: threshold 1.0% gap signal (best OOS variant).

Method:
  - Reconstruct daily P&L for each of the 4 strategies over both:
      * 2021-01-01 .. 2025-12-31  (BASELINE for apples-to-apples comparison)
      * 2025-01-01 .. today (2026-05-12)  (FORWARD OOS test)
  - Vol-normalize each leg to 10% annualized vol (using each window's own
    realized vol — vol-normalization is by definition window-local).
  - Equal-weight 4-strategy portfolio.
  - Compute per-leg sharpe / ann_ret / max_dd, combined stats, pairwise
    correlations, and monthly return distribution on each window.
  - Compute Sharpe decay = (baseline_sharpe - 2025_sharpe) / baseline_sharpe.

Verdict thresholds (per A22 brief):
  - HOLDS UP:    Sharpe decay < 30%
  - MILD DECAY:  30% <= decay < 50%
  - COLLAPSES:   decay >= 50%
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

LOG_PATH = LOGS / "portfolio_forward_walkforward.log"
SUMMARY_PATH = RESULTS / "portfolio_forward_walkforward.json"
PRIOR_COMBO_PATH = RESULTS / "portfolio_combo_summary.json"

# windows
BASELINE_START = pd.Timestamp("2021-01-01")
BASELINE_END = pd.Timestamp("2025-12-31")
FORWARD_START = pd.Timestamp("2025-01-01")
FORWARD_END = pd.Timestamp("2026-05-12")  # today

TARGET_ANN_VOL = 0.10
EW = 0.25  # 4 legs, equal weight
TRADING_DAYS = 252

# SVXY sweet-spot params (from svxy_parameter_sweep_top.json sweet_spot_max_dd_le_20)
SVXY_THRESHOLD = 1.00
SVXY_SIZE = 0.25
SVXY_CB = None  # no circuit breaker
SVXY_COST_BPS = 1.0
SGOV_FALLBACK_APY = 0.04

# Nikkei best OOS variant
NIKKEI_THRESHOLD = 0.010  # 1.0% gap
NIKKEI_ROUND_TRIP_BPS = 5.0

# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("forward_wf")


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def daily_business_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    return pd.bdate_range(start=start, end=end, freq="B")


def normalize_to_vol(series: pd.Series, target: float = TARGET_ANN_VOL) -> tuple[pd.Series, float]:
    realized = series.std() * np.sqrt(TRADING_DAYS)
    if realized <= 0 or np.isnan(realized):
        return series.copy(), 1.0
    k = target / realized
    return series * k, float(k)


def perf_stats(returns: pd.Series, label: str) -> dict:
    if returns.std() <= 0 or len(returns) == 0:
        sharpe = 0.0
    else:
        sharpe = float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS))
    ann_ret = float((1 + returns).prod() ** (TRADING_DAYS / max(len(returns), 1)) - 1)
    equity = (1 + returns).cumprod()
    if len(equity) > 0:
        dd = equity / equity.cummax() - 1
        max_dd = float(dd.min())
    else:
        max_dd = 0.0
    ann_vol = float(returns.std() * np.sqrt(TRADING_DAYS)) if len(returns) else 0.0
    log.info(
        "%-32s | sharpe=%6.3f | ann_ret=%+.2f%% | max_dd=%+.2f%% | n=%4d | vol=%.2f%%",
        label, sharpe, ann_ret * 100, max_dd * 100, len(returns), ann_vol * 100,
    )
    return {
        "sharpe": sharpe,
        "ann_ret": ann_ret,
        "max_dd": max_dd,
        "n_days": int(len(returns)),
        "ann_vol": ann_vol,
    }


# -----------------------------------------------------------------------------
# Strategy 1: SVXY sweet spot — re-run on cached VIX/SVXY panel
# -----------------------------------------------------------------------------
def load_svxy_sweet_spot_series(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Reconstruct daily P&L for SVXY sweet spot (thr=1.00 / size=25% / no CB)."""
    cache_fp = REPO / "research" / "data" / "cache" / "vix_svxy_daily.parquet"
    log.info("SVXY sweet spot: reading %s", cache_fp)
    df = pd.read_parquet(cache_fp)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["VIX", "VIX3M", "SVXY", "SGOV"]].sort_index()
    df = df[df["VIX"].notna() & df["VIX3M"].notna() & df["SVXY"].notna()]

    sgov_ret = df["SGOV"].pct_change()
    sgov_daily_fb = (1.0 + SGOV_FALLBACK_APY) ** (1.0 / 252.0) - 1.0
    sgov_ret = sgov_ret.fillna(sgov_daily_fb)
    df["SGOV_ret"] = sgov_ret
    df["SVXY_ret"] = df["SVXY"].pct_change().fillna(0.0)
    df["ratio"] = df["VIX"] / df["VIX3M"]

    contango = df["ratio"] < SVXY_THRESHOLD
    if SVXY_CB is None:
        cb_active = pd.Series(False, index=df.index)
    else:
        cb_active = df["VIX"] > SVXY_CB
    target_w = np.where(contango & ~cb_active, SVXY_SIZE, 0.0)
    df["target_svxy_w"] = target_w
    df["pos_svxy_w"] = df["target_svxy_w"].shift(1).fillna(0.0)
    df["pos_sgov_w"] = 1.0 - df["pos_svxy_w"]
    weight_chg = df["pos_svxy_w"].diff().abs().fillna(df["pos_svxy_w"].abs())
    df["rotation_cost"] = weight_chg * (SVXY_COST_BPS / 1e4)
    df["gross_ret"] = (
        df["pos_svxy_w"] * df["SVXY_ret"].fillna(0.0)
        + df["pos_sgov_w"] * df["SGOV_ret"].fillna(0.0)
    )
    df["net_ret"] = df["gross_ret"] - df["rotation_cost"]

    s = df["net_ret"].dropna()
    s = s.loc[(s.index >= start) & (s.index <= end)]
    log.info("  SVXY sweet spot: %d rows in [%s, %s]", len(s), s.index.min(), s.index.max())
    return s.rename("svxy_sweet_spot")


# -----------------------------------------------------------------------------
# Strategy 2: BTC funding cash-and-carry — same reconstruction as original
# -----------------------------------------------------------------------------
def load_btc_funding_series(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Reconstruct daily P&L for BTC cash-and-carry over [start, end]."""
    pos_fp = RESULTS / "iter_3_positions.csv"
    fund_fp = RESULTS / "iter_3_funding_history.csv"
    log.info("BTC funding: positions=%s, funding=%s", pos_fp.name, fund_fp.name)

    pos = pd.read_csv(pos_fp)
    pos["entry_dt"] = pd.to_datetime(pos["entry_dt"], utc=True, format="ISO8601").dt.tz_convert(None)
    pos["exit_dt"] = pd.to_datetime(pos["exit_dt"], utc=True, format="ISO8601").dt.tz_convert(None)

    fh = pd.read_csv(fund_fp)
    fh["datetime"] = pd.to_datetime(fh["datetime"], utc=True, format="ISO8601").dt.tz_convert(None)
    fh = fh.set_index("datetime").sort_index()

    idx = daily_business_index(start, end)
    series = pd.Series(0.0, index=idx, dtype=float)

    n_pos_in_window = 0
    for _, r in pos.iterrows():
        e, x = r["entry_dt"], r["exit_dt"]
        # clip the position holding window to [start, end]
        e_clip = max(e, start)
        x_clip = min(x, end)
        if e_clip > x_clip:
            continue
        n_pos_in_window += 1

        ticks = fh.loc[(fh.index >= e_clip) & (fh.index <= x_clip), "funding_rate"]
        if ticks.empty:
            continue
        by_day = ticks.copy()
        by_day.index = by_day.index.normalize()
        daily_funding = by_day.groupby(by_day.index).sum()

        # holding business days within the clipped window
        days_held = pd.bdate_range(e_clip.normalize(), x_clip.normalize(), freq="B")
        days_held = days_held.intersection(idx)
        n_days = max(len(days_held), 1)
        cost_per_day = -0.0030 / n_days

        daily_funding = daily_funding.reindex(idx).fillna(0.0)
        for d in days_held:
            series.loc[d] += float(daily_funding.loc[d]) + cost_per_day

    log.info("  BTC funding [%s..%s]: %d positions, %d non-zero days, total_ret=%+.4f%%",
             start.date(), end.date(), n_pos_in_window,
             int((series != 0).sum()), 100 * ((1 + series).prod() - 1))
    return series.rename("btc_funding")


# -----------------------------------------------------------------------------
# Strategy 3: iter_18 CLEANUP ADR — all 5 pairs
# -----------------------------------------------------------------------------
def load_adr_cleanup_series(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Daily P&L from iter_18 CLEANUP (all 5 pairs).

    Each trade's net_bps is realized on its exit_date. Sleeve weights: equal
    1/5 per pair so the full sleeve sums to 100% sleeve capital across pairs.
    Trades that span [start, end] are included only if exit_date is in window.
    """
    fp = RESULTS / "iter_18_cleanup_trades.csv"
    log.info("ADR cleanup: reading %s", fp)
    tr = pd.read_csv(fp)
    tr["entry_date"] = pd.to_datetime(tr["entry_date"]).dt.normalize()
    tr["exit_date"] = pd.to_datetime(tr["exit_date"]).dt.normalize()
    tr["net_ret"] = tr["net_bps"].astype(float) / 10_000.0

    n_pairs = tr["pair"].nunique()
    pair_weight = 1.0 / n_pairs
    log.info("  ADR cleanup: %d total trades across %d pairs (each %.1f%% sleeve weight)",
             len(tr), n_pairs, pair_weight * 100)

    idx = daily_business_index(start, end)
    series = pd.Series(0.0, index=idx, dtype=float)

    n_used = 0
    for _, r in tr.iterrows():
        x = r["exit_date"]
        if x < start or x > end:
            continue
        ret = float(r["net_ret"]) * pair_weight
        if x in idx:
            d = x
        else:
            after = idx[idx >= x]
            if len(after) == 0:
                continue
            d = after[0]
        n_used += 1
        series.loc[d] += ret

    log.info("  ADR cleanup [%s..%s]: %d trades realized, %d non-zero days",
             start.date(), end.date(), n_used, int((series != 0).sum()))
    return series.rename("adr_cleanup")


# -----------------------------------------------------------------------------
# Strategy 4: iter_6 Nikkei daily proxy — threshold 1.0% gap signal
# -----------------------------------------------------------------------------
def load_nikkei_series(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Daily P&L from iter_6 Nikkei gap proxy at threshold=1.0%."""
    cache_fp = REPO / "research" / "data" / "cache" / "nikkei_daily.parquet"
    log.info("Nikkei: reading %s", cache_fp)
    df = pd.read_parquet(cache_fp)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["Open", "High", "Low", "Close", "Volume"]].sort_index()
    df = df.dropna(subset=["Open", "Close"])

    df["prev_close"] = df["Close"].shift(1)
    df["gap"] = df["Open"] / df["prev_close"] - 1.0
    df["intraday_ret"] = df["Close"] / df["Open"] - 1.0
    signal = np.where(
        df["gap"] > NIKKEI_THRESHOLD, 1.0,
        np.where(df["gap"] < -NIKKEI_THRESHOLD, -1.0, 0.0),
    )
    df["signal"] = signal
    cost_per_trade = NIKKEI_ROUND_TRIP_BPS / 1e4
    df["cost"] = np.where(df["signal"] != 0.0, cost_per_trade, 0.0)
    df["net_ret"] = df["signal"] * df["intraday_ret"] - df["cost"]
    df = df.dropna(subset=["gap", "intraday_ret"])

    s = df["net_ret"]
    s = s.loc[(s.index >= start) & (s.index <= end)]
    n_traded = int((df.loc[(df.index >= start) & (df.index <= end), "signal"] != 0).sum())
    log.info("  Nikkei [%s..%s]: %d rows, %d trade days at threshold=%.1f%%",
             start.date(), end.date(), len(s), n_traded, NIKKEI_THRESHOLD * 100)
    return s.rename("nikkei_gap")


# -----------------------------------------------------------------------------
# Build aligned matrix on a window
# -----------------------------------------------------------------------------
def build_matrix(start: pd.Timestamp, end: pd.Timestamp, label: str) -> pd.DataFrame:
    log.info("=" * 78)
    log.info("Building matrix for window: %s  (%s .. %s)", label, start.date(), end.date())
    log.info("=" * 78)
    svxy = load_svxy_sweet_spot_series(start, end)
    btc = load_btc_funding_series(start, end)
    adr = load_adr_cleanup_series(start, end)
    nik = load_nikkei_series(start, end)

    idx = daily_business_index(start, end)
    df = pd.DataFrame(index=idx)
    df["svxy_sweet_spot"] = svxy.reindex(idx).fillna(0.0)
    df["btc_funding"] = btc.reindex(idx).fillna(0.0)
    df["adr_cleanup"] = adr.reindex(idx).fillna(0.0)
    df["nikkei_gap"] = nik.reindex(idx).fillna(0.0)

    nz_counts = (df != 0).sum().to_dict()
    log.info("Aligned matrix shape=%s, non-zero counts=%s", df.shape, nz_counts)
    return df


def analyze_window(df: pd.DataFrame, label: str) -> dict:
    """Compute per-leg + combined stats on an aligned daily P&L matrix."""
    log.info("--- RAW per-strategy stats [%s] ---", label)
    raw_stats = {}
    for col in df.columns:
        raw_stats[col] = perf_stats(df[col], f"raw {label} {col}")

    log.info("--- VOL-NORMALIZED per-strategy stats [%s, target=%.0f%%] ---",
             label, TARGET_ANN_VOL * 100)
    df_n = pd.DataFrame(index=df.index)
    scales = {}
    norm_stats = {}
    for col in df.columns:
        s, k = normalize_to_vol(df[col])
        df_n[col] = s
        scales[col] = k
        norm_stats[col] = perf_stats(s, f"norm {label} {col}")

    corr = df_n.corr()
    log.info("--- Correlation matrix [%s] ---\n%s", label, corr.round(4).to_string())

    combined = (df_n * EW).sum(axis=1).rename("combined")
    combined_stats = perf_stats(combined, f"COMBINED 1/4 each [{label}]")

    if len(combined) > 0:
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
    else:
        monthly_summary = {}
    log.info("Monthly summary [%s]: %s", label, monthly_summary)

    return {
        "label": label,
        "window": {
            "start": str(df.index.min().date()) if len(df) else None,
            "end": str(df.index.max().date()) if len(df) else None,
        },
        "per_strategy_raw": raw_stats,
        "per_strategy_normalized": norm_stats,
        "vol_normalization_scales": scales,
        "correlation_matrix": {k: {kk: float(round(vv, 6)) for kk, vv in v.items()}
                               for k, v in corr.to_dict().items()},
        "combined_portfolio": {
            "sharpe": combined_stats["sharpe"],
            "ann_ret": combined_stats["ann_ret"],
            "max_dd": combined_stats["max_dd"],
            "ann_vol": combined_stats["ann_vol"],
            "n_days": combined_stats["n_days"],
            "monthly_return_distribution_summary": monthly_summary,
        },
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 78)
    log.info("Forward walk-forward — 4-strategy portfolio OOS test on 2025-only data")
    log.info("Baseline window: %s .. %s", BASELINE_START.date(), BASELINE_END.date())
    log.info("Forward window:  %s .. %s", FORWARD_START.date(), FORWARD_END.date())
    log.info("EW=%.2f per leg | vol_target=%.0f%% | trading_days/yr=%d",
             EW, TARGET_ANN_VOL * 100, TRADING_DAYS)
    log.info("=" * 78)

    # baseline (apples-to-apples 4-strategy build over 2021-2025)
    df_base = build_matrix(BASELINE_START, BASELINE_END, "2021_2025")
    base_result = analyze_window(df_base, "2021_2025")

    # forward (2025-only OOS)
    df_fwd = build_matrix(FORWARD_START, FORWARD_END, "2025_only")
    fwd_result = analyze_window(df_fwd, "2025_only")

    # decay computation
    def decay_pct(base: float, fwd: float) -> float:
        if abs(base) < 1e-12:
            return 0.0
        return float(100.0 * (base - fwd) / base)

    base_norm = base_result["per_strategy_normalized"]
    fwd_norm = fwd_result["per_strategy_normalized"]
    leg_decays = {}
    for col in ["svxy_sweet_spot", "btc_funding", "adr_cleanup", "nikkei_gap"]:
        b = base_norm[col]["sharpe"]
        f = fwd_norm[col]["sharpe"]
        d = decay_pct(b, f)
        leg_decays[col] = {
            "baseline_sharpe": b,
            "forward_sharpe": f,
            "decay_pct": d,
            "baseline_ann_ret": base_norm[col]["ann_ret"],
            "forward_ann_ret": fwd_norm[col]["ann_ret"],
            "baseline_max_dd": base_norm[col]["max_dd"],
            "forward_max_dd": fwd_norm[col]["max_dd"],
        }
        log.info("LEG DECAY %-20s: base=%.3f -> fwd=%.3f (decay=%+.1f%%)",
                 col, b, f, d)

    base_combo_shr = base_result["combined_portfolio"]["sharpe"]
    fwd_combo_shr = fwd_result["combined_portfolio"]["sharpe"]
    combo_decay = decay_pct(base_combo_shr, fwd_combo_shr)
    log.info("COMBINED DECAY: base_sharpe=%.3f -> fwd_sharpe=%.3f (decay=%+.1f%%)",
             base_combo_shr, fwd_combo_shr, combo_decay)

    if combo_decay < 30.0:
        verdict = "HOLDS UP"
    elif combo_decay < 50.0:
        verdict = "MILD DECAY"
    else:
        verdict = "COLLAPSES"
    log.info("VERDICT: %s (combined sharpe decay=%+.1f%%)", verdict, combo_decay)

    # also load prior combo summary for context (3-strategy 2021-2025)
    prior_combo = None
    try:
        prior_combo = json.loads(PRIOR_COMBO_PATH.read_text())
    except Exception as e:
        log.warning("Could not load prior combo file: %s", e)

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "method": {
            "vol_normalize_target": TARGET_ANN_VOL,
            "equal_weight_per_leg": EW,
            "trading_days_per_year": TRADING_DAYS,
            "n_strategies": 4,
            "verdict_thresholds": {
                "HOLDS_UP_max_decay_pct": 30.0,
                "MILD_DECAY_max_decay_pct": 50.0,
            },
        },
        "data_sources": {
            "svxy_sweet_spot": (
                "research/data/cache/vix_svxy_daily.parquet "
                "(threshold=1.00, size=25%, no CB, 1bp cost, SGOV cash leg)"
            ),
            "btc_funding": (
                "research/results/iter_3_positions.csv (on/off mask) + "
                "iter_3_funding_history.csv (daily sum of 3x8h funding ticks, "
                "30bps round-trip amortized over holding business days)"
            ),
            "adr_cleanup": (
                "research/results/iter_18_cleanup_trades.csv "
                "(all 5 dual-listing pairs from the CLEANUP run with corrected "
                "NTES 1:5 ratio and ex-div skip, net_bps realized on exit_date, "
                "20% per pair within sleeve)"
            ),
            "nikkei_gap": (
                "research/data/cache/nikkei_daily.parquet "
                "(threshold=1.0% gap signal, sign(gap)*intraday_ret minus "
                "5bp round-trip cost, daily proxy for spec's 5-min OR breakout)"
            ),
        },
        "2021_2025": base_result,
        "2025_only": fwd_result,
        "sharpe_decay_pct": {
            "combined": combo_decay,
            "per_leg": {k: v["decay_pct"] for k, v in leg_decays.items()},
        },
        "decay_detail": leg_decays,
        "verdict": verdict,
        "verdict_combined_decay_pct": combo_decay,
        "prior_3_strategy_combo_2021_2025": (
            {
                "combined_sharpe": prior_combo["combined_portfolio"]["sharpe"],
                "combined_ann_ret": prior_combo["combined_portfolio"]["ann_ret"],
                "combined_max_dd": prior_combo["combined_portfolio"]["max_dd"],
                "note": "Prior file is 3-leg (no Nikkei) and used iter_18_trades.csv 2-pair subset.",
            }
            if prior_combo
            else None
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Wrote summary: %s", SUMMARY_PATH)

    # ---- terminal report ----
    print("\n" + "=" * 78)
    print("FORWARD WALK-FORWARD REPORT")
    print("=" * 78)
    base_combo = base_result["combined_portfolio"]
    fwd_combo = fwd_result["combined_portfolio"]
    print(f"BASELINE 2021-2025: sharpe={base_combo['sharpe']:.3f} "
          f"ann_ret={base_combo['ann_ret']*100:+.2f}% "
          f"max_dd={base_combo['max_dd']*100:+.2f}%")
    print(f"FORWARD 2025-only:  sharpe={fwd_combo['sharpe']:.3f} "
          f"ann_ret={fwd_combo['ann_ret']*100:+.2f}% "
          f"max_dd={fwd_combo['max_dd']*100:+.2f}%")
    print(
        f"FORWARD_LINE: 2025-only combined sharpe={fwd_combo['sharpe']:.3f} | "
        f"2021-2025 was {base_combo['sharpe']:.3f} | "
        f"decay={combo_decay:+.1f}% | {verdict}"
    )
    print(
        "PER_LEG_DECAY: "
        f"SVXY={leg_decays['svxy_sweet_spot']['decay_pct']:+.1f}%, "
        f"BTC={leg_decays['btc_funding']['decay_pct']:+.1f}%, "
        f"ADR={leg_decays['adr_cleanup']['decay_pct']:+.1f}%, "
        f"Nikkei={leg_decays['nikkei_gap']['decay_pct']:+.1f}%"
    )
    print(f"PATH: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
