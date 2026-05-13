"""Leverage stress test — Agent A23.

Sweeps leverage multipliers across the 4-strategy combined portfolio on
2021-2025 daily P&L and answers: at what leverage level does mean monthly
return >= 14% with acceptable max drawdown (<= 20%)?

Approach:
  1. Reconstruct the un-leveraged daily P&L of the 4-strategy combined
     portfolio using the same methodology as portfolio_forward_walkforward.py
     (vol-normalize each leg to 10% ann vol, equal-weight 1/4 each).
  2. For each L in {1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 6, 7}:
       - Scale daily returns by L (additive arithmetic leverage; classic
         portfolio leverage approximation for daily returns).
       - Subtract a 20 bps/month implementation friction (borrow / IBKR
         margin / slippage).
       - Compute ann_return, ann_vol, sharpe, max_dd, monthly distribution.
       - Compute the historical empirical probability of monthly return >=
         +14% and probability of monthly drawdown > -15%.
  3. Apply a REALISTIC adjustment: discount BTC funding Sharpe by 50%
     (deterministic-stream model artifact per A22 caveat) by scaling the
     BTC leg's daily mean by 0.5 BEFORE vol-norm and recombining.
  4. Identify the optimal leverage level where mean monthly >= 14% AND
     max DD <= 20%. Document both "raw" and "realistic" optima.

Outputs:
  - per_leverage_level: list of dicts (one per L) with all metrics.
  - optimal_for_14pct_monthly: best L hitting the goal (raw stack).
  - realistic_optimal: best L hitting the goal under the haircut.
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

LOG_PATH = LOGS / "portfolio_leverage_stress.log"
OUT_PATH = RESULTS / "portfolio_leverage_stress.json"

# windows (match portfolio_forward_walkforward.py 2021_2025 baseline)
START = pd.Timestamp("2021-01-01")
END = pd.Timestamp("2025-12-31")

TARGET_ANN_VOL = 0.10
EW = 0.25  # 4 legs, equal weight
TRADING_DAYS = 252

# SVXY sweet-spot params
SVXY_THRESHOLD = 1.00
SVXY_SIZE = 0.25
SVXY_CB = None
SVXY_COST_BPS = 1.0
SGOV_FALLBACK_APY = 0.04

# Nikkei
NIKKEI_THRESHOLD = 0.010
NIKKEI_ROUND_TRIP_BPS = 5.0

# Leverage sweep + targets
LEVERAGES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0]
TARGET_MONTHLY_RET = 0.14   # +14% per month
TARGET_MAX_DD_BAR = 0.20    # 20% drawdown ceiling
MONTHLY_DD_THRESHOLD = -0.15  # for prob of "bad month" calc

# Realistic adjustments
BTC_SHARPE_HAIRCUT = 0.50    # discount BTC funding Sharpe by 50%
MONTHLY_FRICTION_BPS = 20.0  # 20 bps/month implementation friction at each L
MONTHLY_FRICTION = MONTHLY_FRICTION_BPS / 1e4  # 0.0020


# -----------------------------------------------------------------------------
# logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("lev_stress")


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


def compute_stats(returns: pd.Series) -> dict:
    """Daily-return -> sharpe / ann_ret / max_dd / ann_vol."""
    n = len(returns)
    if n == 0 or returns.std() <= 0:
        return {
            "sharpe": 0.0,
            "ann_ret": 0.0,
            "max_dd": 0.0,
            "ann_vol": 0.0,
            "n_days": int(n),
        }
    sharpe = float(returns.mean() / returns.std() * np.sqrt(TRADING_DAYS))
    ann_ret = float((1 + returns).prod() ** (TRADING_DAYS / n) - 1)
    equity = (1 + returns).cumprod()
    dd = equity / equity.cummax() - 1
    max_dd = float(dd.min())
    ann_vol = float(returns.std() * np.sqrt(TRADING_DAYS))
    return {
        "sharpe": sharpe,
        "ann_ret": ann_ret,
        "max_dd": max_dd,
        "ann_vol": ann_vol,
        "n_days": int(n),
    }


def monthly_distribution(returns: pd.Series) -> dict:
    """Compound daily->monthly, return distribution summary + monthly drawdowns."""
    if len(returns) == 0:
        return {}
    monthly = (1 + returns).resample("ME").prod() - 1
    n = len(monthly)
    if n == 0:
        return {}
    # also compute "intra-month worst running drawdown" (max DD that opens and
    # closes within the same calendar month) — proxy by min-of-month daily DD
    # from running peak within the month.
    eq = (1 + returns).cumprod()
    monthly_dd = []
    for m, grp in returns.groupby(returns.index.to_period("M")):
        if len(grp) == 0:
            continue
        e = (1 + grp).cumprod()
        d = e / e.cummax() - 1
        monthly_dd.append(float(d.min()))
    monthly_dd = pd.Series(monthly_dd)

    summary = {
        "best_month_pct": float(monthly.max() * 100),
        "worst_month_pct": float(monthly.min() * 100),
        "mean_month_pct": float(monthly.mean() * 100),
        "median_month_pct": float(monthly.median() * 100),
        "std_month_pct": float(monthly.std() * 100),
        "pct_positive_months": float((monthly > 0).mean() * 100),
        "n_months": int(n),
        # empirical probabilities
        "prob_monthly_14pct_plus": float((monthly >= TARGET_MONTHLY_RET).mean()),
        "prob_monthly_dd_worse_than_neg15pct": float(
            (monthly_dd < MONTHLY_DD_THRESHOLD).mean()
        ) if len(monthly_dd) else 0.0,
        "worst_intramonth_dd_pct": float(monthly_dd.min() * 100) if len(monthly_dd) else 0.0,
    }
    return summary


# -----------------------------------------------------------------------------
# Strategy reconstructions — copied verbatim from forward_walkforward
# -----------------------------------------------------------------------------
def load_svxy_sweet_spot_series(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
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
    cb_active = pd.Series(False, index=df.index)
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


def load_btc_funding_series(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    pos_fp = RESULTS / "iter_3_positions.csv"
    fund_fp = RESULTS / "iter_3_funding_history.csv"
    log.info("BTC funding: %s + %s", pos_fp.name, fund_fp.name)

    pos = pd.read_csv(pos_fp)
    pos["entry_dt"] = pd.to_datetime(pos["entry_dt"], utc=True, format="ISO8601").dt.tz_convert(None)
    pos["exit_dt"] = pd.to_datetime(pos["exit_dt"], utc=True, format="ISO8601").dt.tz_convert(None)

    fh = pd.read_csv(fund_fp)
    fh["datetime"] = pd.to_datetime(fh["datetime"], utc=True, format="ISO8601").dt.tz_convert(None)
    fh = fh.set_index("datetime").sort_index()

    idx = daily_business_index(start, end)
    series = pd.Series(0.0, index=idx, dtype=float)

    n_pos = 0
    for _, r in pos.iterrows():
        e, x = r["entry_dt"], r["exit_dt"]
        e_clip = max(e, start)
        x_clip = min(x, end)
        if e_clip > x_clip:
            continue
        n_pos += 1
        ticks = fh.loc[(fh.index >= e_clip) & (fh.index <= x_clip), "funding_rate"]
        if ticks.empty:
            continue
        by_day = ticks.copy()
        by_day.index = by_day.index.normalize()
        daily_funding = by_day.groupby(by_day.index).sum()
        days_held = pd.bdate_range(e_clip.normalize(), x_clip.normalize(), freq="B")
        days_held = days_held.intersection(idx)
        n_days = max(len(days_held), 1)
        cost_per_day = -0.0030 / n_days
        daily_funding = daily_funding.reindex(idx).fillna(0.0)
        for d in days_held:
            series.loc[d] += float(daily_funding.loc[d]) + cost_per_day

    log.info("  BTC funding: %d positions, %d non-zero days, total_ret=%+.4f%%",
             n_pos, int((series != 0).sum()), 100 * ((1 + series).prod() - 1))
    return series.rename("btc_funding")


def load_adr_cleanup_series(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    fp = RESULTS / "iter_18_cleanup_trades.csv"
    log.info("ADR cleanup: reading %s", fp)
    tr = pd.read_csv(fp)
    tr["entry_date"] = pd.to_datetime(tr["entry_date"]).dt.normalize()
    tr["exit_date"] = pd.to_datetime(tr["exit_date"]).dt.normalize()
    tr["net_ret"] = tr["net_bps"].astype(float) / 10_000.0

    n_pairs = tr["pair"].nunique()
    pair_weight = 1.0 / n_pairs
    log.info("  ADR cleanup: %d trades / %d pairs / %.1f%% each",
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

    log.info("  ADR cleanup: %d trades realized, %d non-zero days",
             n_used, int((series != 0).sum()))
    return series.rename("adr_cleanup")


def load_nikkei_series(start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
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
    log.info("  Nikkei: %d rows", len(s))
    return s.rename("nikkei_gap")


# -----------------------------------------------------------------------------
# Build the combined daily P&L series (raw + realistic-haircut variants)
# -----------------------------------------------------------------------------
def build_combined_pnl(haircut_btc: bool) -> tuple[pd.Series, dict]:
    """Return the equal-weight, vol-normalized combined daily P&L series.

    If haircut_btc=True, the BTC funding leg's daily mean is multiplied by
    BTC_SHARPE_HAIRCUT BEFORE vol-normalization. This shrinks the BTC daily
    drift so the leg contributes ~half the Sharpe but the same realized vol
    after vol-norm (so the haircut affects portfolio mean, not vol target).
    """
    label = "haircut" if haircut_btc else "raw"
    log.info("---- Building 4-strategy combined P&L (variant=%s) ----", label)
    svxy = load_svxy_sweet_spot_series(START, END)
    btc = load_btc_funding_series(START, END)
    adr = load_adr_cleanup_series(START, END)
    nik = load_nikkei_series(START, END)

    idx = daily_business_index(START, END)
    df = pd.DataFrame(index=idx)
    df["svxy_sweet_spot"] = svxy.reindex(idx).fillna(0.0)
    df["btc_funding"] = btc.reindex(idx).fillna(0.0)
    df["adr_cleanup"] = adr.reindex(idx).fillna(0.0)
    df["nikkei_gap"] = nik.reindex(idx).fillna(0.0)

    if haircut_btc:
        # Reduce daily mean by 50% — preserves the day-to-day variance pattern
        # but cuts the drift in half. After vol-norm to 10% ann vol the leg's
        # sharpe also halves.
        m = df["btc_funding"].mean()
        df["btc_funding"] = df["btc_funding"] - m * (1.0 - BTC_SHARPE_HAIRCUT)
        log.info("  BTC haircut applied: subtract %.6f from each day's mean (drift halved)",
                 m * (1.0 - BTC_SHARPE_HAIRCUT))

    # vol-normalize each to 10% ann vol then equal-weight 1/4 each
    df_n = pd.DataFrame(index=idx)
    scales = {}
    leg_stats = {}
    for col in df.columns:
        s, k = normalize_to_vol(df[col])
        df_n[col] = s
        scales[col] = k
        leg_stats[col] = compute_stats(s)
        log.info("  %-20s [%s] sharpe=%.3f ann_ret=%+.2f%% ann_vol=%.2f%% (scale=%.3f)",
                 col, label, leg_stats[col]["sharpe"], leg_stats[col]["ann_ret"] * 100,
                 leg_stats[col]["ann_vol"] * 100, k)

    combined = (df_n * EW).sum(axis=1).rename("combined_un_lev")
    combined_stats = compute_stats(combined)
    log.info("  COMBINED un-leveraged [%s]: sharpe=%.3f ann_ret=%+.2f%% max_dd=%+.2f%% ann_vol=%.2f%%",
             label, combined_stats["sharpe"], combined_stats["ann_ret"] * 100,
             combined_stats["max_dd"] * 100, combined_stats["ann_vol"] * 100)

    meta = {
        "variant": label,
        "scales": scales,
        "leg_stats": leg_stats,
        "combined_unlev_stats": combined_stats,
    }
    return combined, meta


# -----------------------------------------------------------------------------
# Leverage stress sweep
# -----------------------------------------------------------------------------
def apply_leverage_and_friction(combined: pd.Series, L: float) -> pd.Series:
    """Scale daily return by L, then subtract 20 bps/month friction spread evenly.

    Friction is applied per business day: 20 bps/month / (252/12) trading days
    per month = ~0.0020 / 21 = ~0.95 bps/day.
    """
    daily_friction = MONTHLY_FRICTION / (TRADING_DAYS / 12.0)
    return (combined * L) - daily_friction


def sweep_leverage(combined: pd.Series, variant_label: str) -> list[dict]:
    """For each L in LEVERAGES, compute the metrics dict."""
    log.info("---- Sweeping leverage [%s variant] ----", variant_label)
    rows = []
    for L in LEVERAGES:
        adj = apply_leverage_and_friction(combined, L)
        stats = compute_stats(adj)
        monthly = monthly_distribution(adj)
        row = {
            "leverage": L,
            "variant": variant_label,
            "ann_return": stats["ann_ret"],
            "ann_vol": stats["ann_vol"],
            "sharpe": stats["sharpe"],
            "max_dd": stats["max_dd"],
            "mean_monthly_ret": monthly.get("mean_month_pct", 0.0) / 100.0,
            "median_monthly_ret": monthly.get("median_month_pct", 0.0) / 100.0,
            "std_monthly_ret": monthly.get("std_month_pct", 0.0) / 100.0,
            "best_monthly_ret": monthly.get("best_month_pct", 0.0) / 100.0,
            "worst_monthly_ret": monthly.get("worst_month_pct", 0.0) / 100.0,
            "pct_positive_months": monthly.get("pct_positive_months", 0.0) / 100.0,
            "prob_monthly_14pct_plus": monthly.get("prob_monthly_14pct_plus", 0.0),
            "prob_monthly_dd_worse_than_neg15pct": monthly.get(
                "prob_monthly_dd_worse_than_neg15pct", 0.0
            ),
            "worst_intramonth_dd_pct": monthly.get("worst_intramonth_dd_pct", 0.0) / 100.0,
            "frictions_applied": {
                "monthly_friction_bps": MONTHLY_FRICTION_BPS,
                "daily_friction_bps": float(MONTHLY_FRICTION / (TRADING_DAYS / 12.0) * 1e4),
            },
        }
        log.info(
            "  L=%.1fx [%s]: ann_ret=%+.2f%% sharpe=%.2f max_dd=%+.2f%% "
            "mean_mo=%+.2f%% P(>=14%%mo)=%.2f P(mo_dd<-15%%)=%.2f",
            L, variant_label, row["ann_return"] * 100, row["sharpe"], row["max_dd"] * 100,
            row["mean_monthly_ret"] * 100, row["prob_monthly_14pct_plus"],
            row["prob_monthly_dd_worse_than_neg15pct"],
        )
        rows.append(row)
    return rows


def pick_optimal(rows: list[dict]) -> dict | None:
    """Pick the lowest L with mean_monthly_ret >= 14% AND max_dd >= -20%."""
    feasible = [
        r for r in rows
        if r["mean_monthly_ret"] >= TARGET_MONTHLY_RET
        and r["max_dd"] >= -TARGET_MAX_DD_BAR
    ]
    if not feasible:
        return None
    return min(feasible, key=lambda r: r["leverage"])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    log.info("=" * 78)
    log.info("Portfolio Leverage Stress — 4-strategy combo, window=%s..%s",
             START.date(), END.date())
    log.info("Leverages: %s", LEVERAGES)
    log.info("Target: mean_monthly >= %.1f%% AND max_dd >= -%.0f%%",
             TARGET_MONTHLY_RET * 100, TARGET_MAX_DD_BAR * 100)
    log.info("Frictions: %.0f bps/month (~%.2f bps/day across %d trading days)",
             MONTHLY_FRICTION_BPS, MONTHLY_FRICTION / (TRADING_DAYS / 12.0) * 1e4, TRADING_DAYS)
    log.info("BTC Sharpe haircut for realistic variant: %.0f%% (drift halved)",
             BTC_SHARPE_HAIRCUT * 100)
    log.info("=" * 78)

    # raw stack
    combined_raw, meta_raw = build_combined_pnl(haircut_btc=False)
    rows_raw = sweep_leverage(combined_raw, "raw")

    # realistic stack (BTC sharpe haircut)
    combined_real, meta_real = build_combined_pnl(haircut_btc=True)
    rows_real = sweep_leverage(combined_real, "realistic")

    opt_raw = pick_optimal(rows_raw)
    opt_real = pick_optimal(rows_real)

    # Honest-path: best L under realistic variant that gets us closest to 14%
    # without exceeding 20% DD (even if it doesn't fully hit 14%)
    def closest_to_target(rows: list[dict]) -> dict:
        # filter to acceptable DD first
        eligible = [r for r in rows if r["max_dd"] >= -TARGET_MAX_DD_BAR]
        if not eligible:
            return min(rows, key=lambda r: abs(r["mean_monthly_ret"] - TARGET_MONTHLY_RET))
        return max(eligible, key=lambda r: r["mean_monthly_ret"])
    honest_real = closest_to_target(rows_real)

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "agent": "A23_leverage_stress",
        "window": {"start": str(START.date()), "end": str(END.date())},
        "method": {
            "vol_normalize_target": TARGET_ANN_VOL,
            "equal_weight_per_leg": EW,
            "n_strategies": 4,
            "trading_days_per_year": TRADING_DAYS,
            "leverages_swept": LEVERAGES,
            "target_monthly_ret": TARGET_MONTHLY_RET,
            "target_max_dd_bar": TARGET_MAX_DD_BAR,
            "monthly_friction_bps": MONTHLY_FRICTION_BPS,
            "btc_sharpe_haircut_pct": (1.0 - BTC_SHARPE_HAIRCUT) * 100,
            "leverage_model": (
                "Daily portfolio P&L scaled linearly by L (additive arithmetic "
                "leverage). 20 bps/month friction spread evenly across "
                "business days (= 0.95 bps/day). BTC haircut applied as a "
                "shift of the BTC daily mean by 50% BEFORE vol-norm."
            ),
            "leverage_caveat": (
                "Linear daily scaling is a first-order approximation. Real "
                "geometric compounding under leverage suffers from volatility "
                "drag (-0.5 * L^2 * sigma^2) which is NOT captured here; the "
                "ann_ret reported is the geometric compound of leveraged "
                "daily returns so volatility drag IS reflected in ann_ret "
                "and max_dd via the compounding, but mean_monthly_ret is the "
                "arithmetic mean of monthly compound returns."
            ),
        },
        "unlevered_combined_raw": meta_raw["combined_unlev_stats"],
        "unlevered_combined_realistic": meta_real["combined_unlev_stats"],
        "vol_normalization_scales_raw": meta_raw["scales"],
        "vol_normalization_scales_realistic": meta_real["scales"],
        "per_leverage_level": rows_raw,
        "per_leverage_level_realistic": rows_real,
        "optimal_for_14pct_monthly": opt_raw,
        "realistic_optimal": opt_real,
        "honest_realistic_closest": honest_real,
    }
    OUT_PATH.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Wrote summary: %s", OUT_PATH)

    # -------------- terminal report --------------
    print("\n" + "=" * 78)
    print("LEVERAGE STRESS REPORT — A23")
    print("=" * 78)
    print(
        f"UNLEVERED RAW:        sharpe={meta_raw['combined_unlev_stats']['sharpe']:.3f} "
        f"ann_ret={meta_raw['combined_unlev_stats']['ann_ret']*100:+.2f}% "
        f"max_dd={meta_raw['combined_unlev_stats']['max_dd']*100:+.2f}%"
    )
    print(
        f"UNLEVERED REALISTIC:  sharpe={meta_real['combined_unlev_stats']['sharpe']:.3f} "
        f"ann_ret={meta_real['combined_unlev_stats']['ann_ret']*100:+.2f}% "
        f"max_dd={meta_real['combined_unlev_stats']['max_dd']*100:+.2f}%"
    )

    print("\nRAW STACK SWEEP:")
    for r in rows_raw:
        print(
            f"  L={r['leverage']:.1f}x sharpe={r['sharpe']:.2f} "
            f"ann_ret={r['ann_return']*100:+.1f}% max_dd={r['max_dd']*100:+.1f}% "
            f"mean_monthly={r['mean_monthly_ret']*100:+.2f}% "
            f"P(>=14%/mo)={r['prob_monthly_14pct_plus']:.2f}"
        )

    print("\nREALISTIC (BTC sharpe halved + 20bp/mo friction):")
    for r in rows_real:
        print(
            f"  L={r['leverage']:.1f}x sharpe={r['sharpe']:.2f} "
            f"ann_ret={r['ann_return']*100:+.1f}% max_dd={r['max_dd']*100:+.1f}% "
            f"mean_monthly={r['mean_monthly_ret']*100:+.2f}% "
            f"P(>=14%/mo)={r['prob_monthly_14pct_plus']:.2f}"
        )

    if opt_raw:
        print(
            f"\nLEVERAGE_LINE: To hit 14%/month (raw stack): "
            f"leverage={opt_raw['leverage']:.1f}x, "
            f"max_dd={opt_raw['max_dd']*100:.1f}%, sharpe={opt_raw['sharpe']:.2f}"
        )
    else:
        print(
            "\nLEVERAGE_LINE: NO raw-stack leverage hits 14%/month within max_dd ≤ 20%"
        )

    if opt_real:
        print(
            f"HONEST_PATH: At realistic (BTC sharpe halved), leverage="
            f"{opt_real['leverage']:.1f}x gives mean monthly={opt_real['mean_monthly_ret']*100:.2f}% "
            f"with DD={opt_real['max_dd']*100:.1f}%"
        )
    else:
        print(
            f"HONEST_PATH: NO realistic leverage hits 14%/month within DD ≤ 20%. "
            f"Best feasible: L={honest_real['leverage']:.1f}x → "
            f"mean monthly={honest_real['mean_monthly_ret']*100:.2f}% "
            f"DD={honest_real['max_dd']*100:.1f}% "
            f"sharpe={honest_real['sharpe']:.2f}"
        )

    print(f"PATH: {OUT_PATH}")


if __name__ == "__main__":
    main()
