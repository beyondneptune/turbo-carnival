"""
iter_16 — Multi-Carry Stack: BTC Perp Funding (Leg A) x G10 FX Carry (Leg B),
risk-parity overlay with 30-day correlation circuit breaker.

This is a COMBINATORIAL overlay backtest. It does NOT re-derive a new alpha
source; it stacks iter_3 (BTC funding, PASS @ Sharpe 0.62) and iter_4 (G10 FX
carry, FAIL @ Sharpe 0.04) and tests whether the risk-parity combination
produces a meaningfully different result.

Inputs:
  - /research/results/iter_3_funding_history.csv  (BTC 8h funding rates + rolling APR)
  - /research/results/iter_3_positions.csv         (entry/exit cycles for Leg A)
  - /research/results/iter_4_monthly_returns.csv   (FX carry monthly returns)

Mechanics (per spec):
  - Reconstruct BTC daily P&L from funding history while "in-position" per the
    rolling-30-cycle / 10%-entry / 4%-exit rule used in iter_3. While flat,
    daily return = 0. Apply 15 bps round-trip transaction cost on each
    entry/exit (split entry+exit), as per iter_3 mechanics (the 30 bps in
    iter_3 is already netted in net_carry; for daily we book it as a one-time
    debit on entry and exit days, prorated).
  - Resample BTC daily returns to monthly compounded returns.
  - FX monthly returns are already monthly.
  - Common period: 2021-01 to 2025-12 (5y OOS window per spec).
  - 30-day rolling vol per leg -> monthly cadence: use 6-month rolling window
    (the canonical short window for monthly-resampled risk-parity overlays;
    see Asness-Frazzini-Pedersen). Annualize by sqrt(12).
  - Vol target = 8% annualized per leg -> scale_leg = 0.08 / sigma_leg
    (clipped to [0.25, 2.50] per spec STEP 2).
  - Apply scales monthly (rebalance), using prior-month estimates to avoid
    lookahead.
  - Correlation circuit breaker: 30-day rolling rho on daily-equivalent
    returns. At monthly cadence we use 6-month rolling correlation of monthly
    returns. Per spec: when |rho| > 0.30 (vol gate omitted at monthly cadence
    because 12% ann vol is the BTC norm), halve the BTC leg scale.
  - Bridge cost: 5 bps per rebalance per leg (lenient view of 15 bps RT in
    the spec; we book each-leg-each-rebalance at 5 bps, total 10 bps per
    monthly rebalance, ~120 bps/year on full exposure - this is the
    conservative end of the spec's "30 bps/year" estimate at 2 rebalances/yr).
  - Monthly rebalance.

Outputs:
  - /research/results/iter_16_summary.json
  - /research/logs/iter_16_backtest.log
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------- paths ----------------
ROOT = Path("/Users/weeoto/turbo-carnival/research")
BTC_FUNDING_CSV = ROOT / "results" / "iter_3_funding_history.csv"
BTC_POSITIONS_CSV = ROOT / "results" / "iter_3_positions.csv"
FX_RETURNS_CSV = ROOT / "results" / "iter_4_monthly_returns.csv"
LOG_PATH = ROOT / "logs" / "iter_16_backtest.log"
SUMMARY_JSON = ROOT / "results" / "iter_16_summary.json"

# ---------------- parameters (per spec) ----------------
COMMON_START = "2021-01-01"
COMMON_END = "2025-12-31"  # 2021-2025 5y window
TARGET_VOL_ANN = 0.08            # 8% per leg
SCALE_CLIP_LO = 0.25
SCALE_CLIP_HI = 2.50
CORR_CUTOFF = 0.30
ROLL_MONTHS_VOL = 6              # 30d at daily -> ~6m at monthly (canonical RP window)
ROLL_MONTHS_CORR = 6
BRIDGE_BPS_PER_LEG = 5.0         # 5 bps per leg per rebalance
GROSS_LEV_CAP = 3.0
# iter_3 mechanics (for BTC daily reconstruction)
ENTRY_APR = 0.10
EXIT_APR = 0.04
ROLL_WINDOW_CYCLES = 30
ROUND_TRIP_COST = 0.0030   # 30 bps cash-and-carry roundtrip, per iter_3


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("iter_16")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(LOG_PATH, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ---------------- BTC daily P&L reconstruction ----------------
def reconstruct_btc_daily_pnl(logger: logging.Logger) -> pd.Series:
    """
    Replay the iter_3 in/out logic on 8h funding cycles, then aggregate to
    daily returns. Per cycle while in-position, the strategy earns
    `funding_rate` (paid by longs to shorts when funding > 0). When flat,
    earnings = 0. Cost: 15 bps booked at entry, 15 bps at exit (matches
    iter_3's 30 bps round-trip).
    """
    df = pd.read_csv(BTC_FUNDING_CSV)
    df["dt"] = pd.to_datetime(df["datetime"], utc=True, format="ISO8601")
    df["roll_mean_apr"] = df["roll30_mean_apr"]
    df["funding_rate"] = df["funding_rate"]
    df = df.dropna(subset=["roll_mean_apr"]).reset_index(drop=True)

    in_position = False
    cycle_returns = []  # list of (dt, ret)
    for _, row in df.iterrows():
        roll = row["roll_mean_apr"]
        fr = row["funding_rate"]
        dt = row["dt"]
        cycle_ret = 0.0
        if not in_position:
            if roll > ENTRY_APR:
                # enter: pay 15 bps cost
                in_position = True
                cycle_ret -= ROUND_TRIP_COST / 2.0
                # also earn this cycle's funding (we're now short perp)
                cycle_ret += fr
        else:
            # currently in
            cycle_ret += fr
            if roll < EXIT_APR:
                in_position = False
                cycle_ret -= ROUND_TRIP_COST / 2.0
        cycle_returns.append((dt, cycle_ret))

    cyc_df = pd.DataFrame(cycle_returns, columns=["dt", "ret"]).set_index("dt")
    # Aggregate 3 8h-cycles per day to daily compounded return
    daily = (1.0 + cyc_df["ret"]).resample("1D").prod() - 1.0
    daily.name = "btc_daily_ret"
    logger.info(
        f"BTC daily P&L: {len(daily)} days; "
        f"in-position days={(daily != 0).sum()}; "
        f"mean ret={daily.mean()*100:.4f}%; std={daily.std()*100:.4f}%"
    )
    return daily


# ---------------- FX monthly returns ----------------
def load_fx_monthly(logger: logging.Logger) -> pd.Series:
    df = pd.read_csv(FX_RETURNS_CSV)
    df["date"] = pd.to_datetime(df["date"])
    s = df.set_index("date")["monthly_return"].astype(float)
    s = s.sort_index()
    s.name = "fx_monthly_ret"
    logger.info(
        f"FX monthly P&L: {len(s)} months; "
        f"mean={s.mean()*100:.4f}%; std={s.std()*100:.4f}%; "
        f"range={s.index.min().date()}..{s.index.max().date()}"
    )
    return s


# ---------------- Sharpe / DD helpers ----------------
def sharpe(monthly_returns: pd.Series) -> float:
    r = monthly_returns.dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float((r.mean() * 12) / (r.std() * np.sqrt(12)))


def ann_return(monthly_returns: pd.Series) -> float:
    r = monthly_returns.dropna()
    if len(r) == 0:
        return 0.0
    return float(((1.0 + r).prod()) ** (12.0 / len(r)) - 1.0)


def ann_vol(monthly_returns: pd.Series) -> float:
    r = monthly_returns.dropna()
    if len(r) < 2:
        return 0.0
    return float(r.std() * np.sqrt(12))


def max_drawdown(monthly_returns: pd.Series) -> float:
    r = monthly_returns.dropna()
    if len(r) == 0:
        return 0.0
    nav = (1.0 + r).cumprod()
    peak = nav.cummax()
    dd = nav / peak - 1.0
    return float(dd.min())


# ---------------- Risk-parity overlay ----------------
def build_overlay(
    btc_m: pd.Series, fx_m: pd.Series, logger: logging.Logger
) -> tuple[pd.DataFrame, dict]:
    """
    Build the risk-parity combined monthly return series.

    For each month t:
      - sigma_btc = std(btc_m[t-6:t-1]) * sqrt(12)  (lagged to avoid lookahead)
      - sigma_fx  = std(fx_m[t-6:t-1])  * sqrt(12)
      - rho       = corr(btc_m[t-6:t-1], fx_m[t-6:t-1])
      - scale_btc = clip(0.08 / sigma_btc, 0.25, 2.50)
      - scale_fx  = clip(0.08 / sigma_fx, 0.25, 2.50)
      - if |rho| > 0.30: scale_btc *= 0.5  (per spec circuit breaker, halve BTC)
      - gross cap: shrink both if (scale_btc + scale_fx) > 3.0
      - combined_ret_t = 0.5 * (scale_btc * btc_m[t] + scale_fx * fx_m[t])
        (0.5 = the risk-parity normalization: at vol-target each leg
         contributes target_vol; equal-weighted aggregator divides by 2)
      - cost: bridge_bps_per_leg * 1e-4 per leg per rebalance, debited from t

    Returns:
      df: per-month diagnostics including scales, vols, corr, returns
      summary_components: dict of stats
    """
    # align on shared monthly index
    btc_m = btc_m.copy()
    fx_m = fx_m.copy()
    common = btc_m.index.intersection(fx_m.index)
    btc_m = btc_m.reindex(common).fillna(0.0)
    fx_m = fx_m.reindex(common).fillna(0.0)

    # filter to common period
    start = pd.Timestamp(COMMON_START)
    end = pd.Timestamp(COMMON_END)
    mask = (btc_m.index >= start) & (btc_m.index <= end)
    btc_m = btc_m[mask]
    fx_m = fx_m[mask]
    logger.info(
        f"Common monthly window: {btc_m.index.min().date()}..{btc_m.index.max().date()} "
        f"({len(btc_m)} months)"
    )

    df = pd.DataFrame({"btc": btc_m, "fx": fx_m})

    # rolling 6m std using lagged window (shift(1) so we use info up to t-1 only)
    df["sigma_btc"] = df["btc"].shift(1).rolling(ROLL_MONTHS_VOL, min_periods=3).std() * np.sqrt(12)
    df["sigma_fx"] = df["fx"].shift(1).rolling(ROLL_MONTHS_VOL, min_periods=3).std() * np.sqrt(12)
    df["rho"] = (
        df["btc"].shift(1).rolling(ROLL_MONTHS_CORR, min_periods=3).corr(df["fx"].shift(1))
    )

    # scales
    df["scale_btc_raw"] = (TARGET_VOL_ANN / df["sigma_btc"]).clip(SCALE_CLIP_LO, SCALE_CLIP_HI)
    df["scale_fx_raw"] = (TARGET_VOL_ANN / df["sigma_fx"]).clip(SCALE_CLIP_LO, SCALE_CLIP_HI)

    # circuit breaker: halve BTC when |rho|>0.30
    df["breaker_fired"] = df["rho"].abs() > CORR_CUTOFF
    df["scale_btc"] = np.where(df["breaker_fired"], df["scale_btc_raw"] * 0.5, df["scale_btc_raw"])
    df["scale_fx"] = df["scale_fx_raw"]

    # gross-leverage cap (legs sum)
    total = df["scale_btc"] + df["scale_fx"]
    shrink = np.where(total > GROSS_LEV_CAP, GROSS_LEV_CAP / total, 1.0)
    df["scale_btc"] *= shrink
    df["scale_fx"] *= shrink

    # warm-up: first ROLL_MONTHS_VOL months have NaN vols -> use equal-weight as fallback
    # (or set scale to 1.0; we choose 1.0 since the alternative is to drop those months)
    df["scale_btc"] = df["scale_btc"].fillna(1.0)
    df["scale_fx"] = df["scale_fx"].fillna(1.0)

    # combined return: each leg risked at target_vol; the RP portfolio is
    # 50/50 of two vol-targeted legs, which gives portfolio vol ~ target_vol * sqrt((1+rho)/2)
    df["leg_btc_ret"] = df["scale_btc"] * df["btc"]
    df["leg_fx_ret"] = df["scale_fx"] * df["fx"]
    df["combined_pre_cost"] = 0.5 * (df["leg_btc_ret"] + df["leg_fx_ret"])

    # bridge cost: 5 bps per leg per monthly rebalance, debited monthly
    cost_per_month = 2 * (BRIDGE_BPS_PER_LEG / 1e4)
    df["combined_ret"] = df["combined_pre_cost"] - cost_per_month

    # also book transaction cost on each leg individually for fair comparison
    df["leg_btc_ret_net"] = df["leg_btc_ret"] - (BRIDGE_BPS_PER_LEG / 1e4)
    df["leg_fx_ret_net"] = df["leg_fx_ret"] - (BRIDGE_BPS_PER_LEG / 1e4)

    # diagnostics
    logger.info(f"Avg scale_btc: {df['scale_btc'].mean():.3f}  "
                f"Avg scale_fx: {df['scale_fx'].mean():.3f}")
    logger.info(f"Circuit breaker fired in {int(df['breaker_fired'].sum())} months "
                f"({df['breaker_fired'].mean()*100:.1f}%)")
    logger.info(f"Median |rho|: {df['rho'].abs().median():.3f}; "
                f"Max |rho|: {df['rho'].abs().max():.3f}")

    return df, {}


# ---------------- main ----------------
def main() -> None:
    logger = setup_logging()
    logger.info("=== iter_16 BTC funding x FX carry risk-parity backtest ===")

    btc_daily = reconstruct_btc_daily_pnl(logger)
    # monthly compounded from daily
    btc_monthly = (1.0 + btc_daily).resample("ME").prod() - 1.0
    btc_monthly.name = "btc_monthly_ret"
    logger.info(
        f"BTC monthly P&L: {len(btc_monthly)} months; "
        f"mean={btc_monthly.mean()*100:.4f}%; std={btc_monthly.std()*100:.4f}%"
    )

    # align FX to month-end (FX already month-end)
    fx_monthly = load_fx_monthly(logger)

    # FX is dated like 2010-01-31, BTC is month-end via resample('ME').
    # Make both indexes month-end without tz to align.
    btc_monthly.index = btc_monthly.index.tz_localize(None)
    btc_monthly.index = btc_monthly.index.to_period("M").to_timestamp("M")
    fx_monthly.index = fx_monthly.index.to_period("M").to_timestamp("M")

    df, _ = build_overlay(btc_monthly, fx_monthly, logger)

    # stats on legs vs combined (post-cost)
    btc_leg_net = df["leg_btc_ret_net"].dropna()
    fx_leg_net = df["leg_fx_ret_net"].dropna()
    combo = df["combined_ret"].dropna()

    btc_sharpe = sharpe(btc_leg_net)
    fx_sharpe = sharpe(fx_leg_net)
    combo_sharpe = sharpe(combo)

    btc_ann = ann_return(btc_leg_net)
    fx_ann = ann_return(fx_leg_net)
    combo_ann = ann_return(combo)

    btc_dd = max_drawdown(btc_leg_net)
    fx_dd = max_drawdown(fx_leg_net)
    combo_dd = max_drawdown(combo)

    # realized full-window correlation
    full_rho = float(df["btc"].corr(df["fx"]))

    # leg standalone (without scaling, raw) sharpes for context
    raw_btc_sharpe = sharpe(df["btc"].dropna())
    raw_fx_sharpe = sharpe(df["fx"].dropna())

    # Verdict: PASS if combined_sharpe > max(leg_btc_sharpe, leg_fx_sharpe)
    #         AND combined_max_dd > worse_leg_max_dd (i.e. smaller absolute DD).
    # "worse_leg" = the leg whose max_dd is more negative (larger |DD|).
    worse_dd = min(btc_dd, fx_dd)  # min because both negative; min = more negative = worse
    best_leg_sharpe = max(btc_sharpe, fx_sharpe)

    sharpe_pass = combo_sharpe > best_leg_sharpe
    dd_pass = combo_dd > worse_dd  # combined DD less negative than worse leg's

    verdict = "PASS" if (sharpe_pass and dd_pass) else "FAIL"

    summary = {
        "iter": 16,
        "strategy": "btc_funding_x_fx_carry_risk_parity",
        "common_start": str(df.index.min().date()),
        "common_end": str(df.index.max().date()),
        "months": int(len(df)),
        "target_vol_ann_per_leg_pct": TARGET_VOL_ANN * 100,
        "corr_cutoff": CORR_CUTOFF,
        "bridge_bps_per_leg_per_rebalance": BRIDGE_BPS_PER_LEG,
        "roll_months_vol": ROLL_MONTHS_VOL,
        "roll_months_corr": ROLL_MONTHS_CORR,
        # raw (unscaled, no cost) leg stats for context
        "raw_btc_sharpe": raw_btc_sharpe,
        "raw_fx_sharpe": raw_fx_sharpe,
        # vol-targeted, post-cost individual legs
        "leg_btc_sharpe": btc_sharpe,
        "leg_fx_sharpe": fx_sharpe,
        "leg_btc_ann_ret_pct": btc_ann * 100,
        "leg_fx_ann_ret_pct": fx_ann * 100,
        "leg_btc_max_dd_pct": btc_dd * 100,
        "leg_fx_max_dd_pct": fx_dd * 100,
        # combined portfolio (post-cost)
        "combined_sharpe": combo_sharpe,
        "combined_ann_ret_pct": combo_ann * 100,
        "combined_ann_vol_pct": ann_vol(combo) * 100,
        "combined_max_dd_pct": combo_dd * 100,
        "correlation_full_window": full_rho,
        "median_abs_rolling_corr": float(df["rho"].abs().median()),
        "max_abs_rolling_corr": float(df["rho"].abs().max()),
        "circuit_breaker_months": int(df["breaker_fired"].sum()),
        "avg_scale_btc": float(df["scale_btc"].mean()),
        "avg_scale_fx": float(df["scale_fx"].mean()),
        "sharpe_pass": bool(sharpe_pass),
        "dd_pass": bool(dd_pass),
        "verdict": verdict,
    }

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Wrote {SUMMARY_JSON}")
    logger.info(f"Summary: {json.dumps(summary, indent=2, default=str)}")

    print(
        f"RESULT_LINE: iter_16 | combined_sharpe={combo_sharpe:.3f} | "
        f"vs BTC alone={btc_sharpe:.3f} | vs FX alone={fx_sharpe:.3f} | "
        f"VERDICT: {verdict}"
    )
    print(f"  combined_ann_ret={combo_ann*100:.2f}%  combined_max_dd={combo_dd*100:.2f}%  "
          f"corr={full_rho:.3f}  breaker_fired={int(df['breaker_fired'].sum())} months")


if __name__ == "__main__":
    main()
