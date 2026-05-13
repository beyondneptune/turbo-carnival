"""
iter_9 backtest: NIFTY cross-section momentum -- ACCESSIBILITY FALLBACK FORM.

The canonical iter_9 strategy is cross-sectional 12-1 momentum on NIFTY 50
constituents with weekly long top-5 / short NIFTY-futures rebalance
(see research/candidates/iter_9_nifty_cross_section_momentum.md).  That
requires direct NSE / Kite Connect access plus an FPI / Indian-broker stack
that US retail accounts generally cannot get.

This backtest implements the simplified, US-accessible variant explicitly
called out in the spec (section 2 "US-accessible fallback" and section 12
"degraded but executable form"):

  - Single asset: iShares MSCI India ETF (INDA).
  - Momentum-gate signal computed on INDA itself (proxy for the index).
  - Long-only with cash fallback (effectively SGOV at 4% yield in flat / risk-off).
  - Vol-target 12% annualized on past 60-day realized vol.
  - Weekly rebalance, 10 bps round-trip cost per rebalance.
  - Walk-forward: IS 2014-2019, OOS 2020-2025.

Signal:
  sig = sign(mom_12_1) + sign(mom_6m) + sign(mom_3m)
    where mom_X = ln(P[t-21] / P[t-X-21+21]) with X in {252, 126, 63} (skipping
    the trailing 21d to mimic the Jegadeesh-Titman short-reversal control)
  +2 or +3 -> 100% INDA exposure (then vol-scaled)
  +1 or  0 -> 50% INDA + 50% cash (then vol-scaled)
   <0      -> 0% INDA, 100% cash (assumed 4% APR -> ~0.00756%/day)

Vol target: target_lev = clip(0.12 / realized_vol_60d, 0.3, 2.0).
Realized vol uses past 60d daily INDA returns annualized.

Outputs:
  - cache:    research/data/cache/inda_nifty_daily.parquet
  - log:      research/logs/iter_9_backtest.log
  - summary:  research/results/iter_9_summary.json
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf


# ----------------------------- config ------------------------------------- #

REPO = Path("/Users/weeoto/turbo-carnival")
CACHE = REPO / "research" / "data" / "cache" / "inda_nifty_daily.parquet"
LOG_PATH = REPO / "research" / "logs" / "iter_9_backtest.log"
SUMMARY_JSON = REPO / "research" / "results" / "iter_9_summary.json"

# Pull both for diagnostics; trade only INDA (USD-tradable).
TICKERS = ["INDA", "^NSEI"]

START = "2014-01-01"
END = "2026-01-01"

# Momentum lookbacks (trading days)
LB_LONG = 252       # 12 months
LB_MID = 126        # 6 months
LB_SHORT = 63       # 3 months
SKIP = 21           # 1-month skip (Jegadeesh-Titman style)

# Vol targeting
TARGET_VOL = 0.12
VOL_LOOKBACK = 60
LEV_FLOOR = 0.3
LEV_CAP = 2.0
TRADING_DAYS_PER_YEAR = 252

# Rebalance / costs
REBAL_DAYS = 5                   # weekly on every 5th trading day
COST_BP_PER_REBAL = 10.0         # 10 bp round trip per rebalance scaled by turnover

# Cash yield (SGOV proxy, 4% APR)
CASH_APR = 0.04
CASH_DAILY = (1.0 + CASH_APR) ** (1.0 / TRADING_DAYS_PER_YEAR) - 1.0

# Walk-forward windows
IS_START = "2014-01-01"
IS_END = "2019-12-31"
OOS_START = "2020-01-01"
OOS_END = "2025-12-31"


# ----------------------------- logging ------------------------------------ #

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
CACHE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("iter_9")
logger.setLevel(logging.INFO)
for h in list(logger.handlers):
    logger.removeHandler(h)
fh = logging.FileHandler(LOG_PATH, mode="w")
sh = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
fh.setFormatter(fmt)
sh.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(sh)


# ----------------------------- data load ---------------------------------- #

def load_prices(refresh: bool = False) -> pd.DataFrame:
    """Return wide DataFrame of adjusted close prices for INDA and ^NSEI."""
    if CACHE.exists() and not refresh:
        try:
            df = pd.read_parquet(CACHE)
            missing = [t for t in TICKERS if t not in df.columns]
            cached_end = df.index.max()
            today = pd.Timestamp.now(tz=None).normalize()
            cend = cached_end.tz_localize(None) if getattr(cached_end, "tzinfo", None) else cached_end
            stale = (today - cend).days > 7
            if not missing and not stale:
                logger.info(f"Loaded cached prices: {df.shape} from {CACHE}")
                return df
            logger.info(f"Cache present but missing={missing} stale={stale}; refreshing.")
        except Exception as e:
            logger.warning(f"Failed to read cache ({e}); refetching.")

    logger.info(f"Downloading {TICKERS} from yfinance {START}..{END}")
    raw = yf.download(
        tickers=TICKERS,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned empty dataframe")

    closes = {}
    for t in TICKERS:
        try:
            s = raw[t]["Close"].rename(t)
        except Exception:
            try:
                s = raw["Close"][t].rename(t)
            except Exception:
                logger.warning(f"Failed to find Close for {t}")
                continue
        closes[t] = s
    df = pd.concat(closes.values(), axis=1)
    df = df.dropna(how="all").sort_index()
    df = df.dropna(axis=1, how="all")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.to_parquet(CACHE)
    logger.info(f"Wrote cache {CACHE}  shape={df.shape}  range={df.index.min().date()}..{df.index.max().date()}")
    return df


# ----------------------------- signal ------------------------------------- #

def compute_momentum_signal(price: pd.Series) -> pd.DataFrame:
    """Compute 12-1, 6m, 3m momentum and aggregated signal score.

    Each mom_X is the log return from t-X to t-SKIP (skip last SKIP days).
    """
    log_p = np.log(price)
    # mom_X = log_p[t-SKIP] - log_p[t-X]; equivalently shift trick.
    p_skip = log_p.shift(SKIP)
    mom_12_1 = p_skip - log_p.shift(LB_LONG)
    mom_6m = p_skip - log_p.shift(LB_MID)
    mom_3m = p_skip - log_p.shift(LB_SHORT)

    signal = (
        np.sign(mom_12_1).fillna(0).astype(float)
        + np.sign(mom_6m).fillna(0).astype(float)
        + np.sign(mom_3m).fillna(0).astype(float)
    )

    # Map signal score to target INDA exposure pre-vol-scaling
    def to_target(s: float) -> float:
        if s >= 2:
            return 1.0
        if s >= 0:
            return 0.5
        return 0.0

    target_exposure = signal.apply(to_target)
    # Mask the warm-up period
    valid = (~mom_12_1.isna()) & (~mom_6m.isna()) & (~mom_3m.isna())
    target_exposure = target_exposure.where(valid, np.nan)

    return pd.DataFrame(
        {
            "mom_12_1": mom_12_1,
            "mom_6m": mom_6m,
            "mom_3m": mom_3m,
            "signal_score": signal,
            "target_exposure_pre_vol": target_exposure,
        }
    )


def compute_realized_vol(returns: pd.Series, lookback: int = VOL_LOOKBACK) -> pd.Series:
    """Annualized rolling realized vol using past N daily simple returns."""
    return returns.rolling(lookback, min_periods=lookback).std() * math.sqrt(TRADING_DAYS_PER_YEAR)


# ----------------------------- backtest core ------------------------------ #

def run_backtest(prices: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Walk daily series, rebalance every REBAL_DAYS, return per-day frame."""
    if "INDA" not in prices.columns:
        raise RuntimeError("INDA not in price panel")
    inda = prices["INDA"].dropna()
    inda_ret = inda.pct_change()

    sig = compute_momentum_signal(inda)
    rv = compute_realized_vol(inda_ret).reindex(inda.index)

    # Combine into one frame
    frame = pd.DataFrame(
        {
            "inda": inda,
            "inda_ret": inda_ret,
            "realized_vol_60d_ann": rv,
        }
    )
    frame = frame.join(sig)

    # Determine first usable trading index: need both momentum and realized vol.
    valid_mask = (~frame["target_exposure_pre_vol"].isna()) & (~frame["realized_vol_60d_ann"].isna())
    if not valid_mask.any():
        raise RuntimeError("No valid signal/vol days; data too short.")
    start_idx = frame.index.get_loc(valid_mask[valid_mask].index[0])
    logger.info(f"First tradeable day: {frame.index[start_idx].date()}  (after warmup)")

    # Containers
    n = len(frame)
    held_w = np.zeros(n)            # actual portfolio weight on INDA each day (post vol scale)
    target_w = np.zeros(n)          # target weight on INDA at most recent rebalance
    cash_w = np.zeros(n)            # weight in cash
    lev = np.zeros(n)                # leverage multiplier applied (informational)
    daily_ret = np.zeros(n)
    turnover = np.zeros(n)
    rebal_flag = np.zeros(n, dtype=bool)

    days_since_rebal = REBAL_DAYS  # force rebalance on first valid day

    # Snap weights track
    cur_inda_w = 0.0
    cur_cash_w = 1.0

    inda_arr = frame["inda"].values
    ret_arr = frame["inda_ret"].fillna(0.0).values
    tgt_arr = frame["target_exposure_pre_vol"].values
    rv_arr = frame["realized_vol_60d_ann"].values

    rebal_dates = []
    for i in range(n):
        # Apply yesterday-to-today drift first
        if i > 0:
            # INDA leg moves by INDA return; cash earns daily yield
            cur_inda_w = cur_inda_w * (1.0 + ret_arr[i])
            cur_cash_w = cur_cash_w * (1.0 + CASH_DAILY)
            total = cur_inda_w + cur_cash_w
            day_ret = total - 1.0
            # Normalize so weights remain interpretable post-drift
            if total > 0:
                cur_inda_w = cur_inda_w / total
                cur_cash_w = cur_cash_w / total
        else:
            day_ret = 0.0

        # Skip warmup
        if i < start_idx:
            held_w[i] = 0.0
            cash_w[i] = 1.0
            daily_ret[i] = CASH_DAILY  # park in cash during warmup
            # carry weights consistent with cash parking
            cur_inda_w = 0.0
            cur_cash_w = 1.0
            continue

        # Rebalance check (every REBAL_DAYS trading days from start_idx, or first valid day)
        do_rebal = days_since_rebal >= REBAL_DAYS

        if do_rebal:
            tgt_pre = tgt_arr[i]
            rv_i = rv_arr[i]
            if not np.isfinite(tgt_pre) or not np.isfinite(rv_i) or rv_i <= 0:
                # Defer rebalance
                do_rebal = False

        if do_rebal:
            lev_i = TARGET_VOL / rv_i
            lev_i = float(min(max(lev_i, LEV_FLOOR), LEV_CAP))
            new_inda_w = float(tgt_pre * lev_i)
            # Cap gross INDA exposure at 1.0 to remain long-only ETF accessible.
            new_inda_w = min(new_inda_w, 1.0)
            new_cash_w = 1.0 - new_inda_w
            # Turnover = |delta INDA weight| + |delta cash weight| / 2 (one-sided)
            tov = abs(new_inda_w - cur_inda_w)
            # Cost: 10 bp on the side that traded (one-sided turnover).
            cost = (COST_BP_PER_REBAL / 10_000.0) * tov
            # Apply cost as a deduction from today's return
            day_ret = day_ret - cost
            cur_inda_w = new_inda_w
            cur_cash_w = new_cash_w
            lev[i] = lev_i
            target_w[i] = float(tgt_pre)
            turnover[i] = tov
            rebal_flag[i] = True
            days_since_rebal = 0
            rebal_dates.append(frame.index[i])
        else:
            # Carry through previous lev / target for the record
            lev[i] = lev[i - 1] if i > 0 else 0.0
            target_w[i] = target_w[i - 1] if i > 0 else 0.0
            days_since_rebal += 1

        held_w[i] = cur_inda_w
        cash_w[i] = cur_cash_w
        daily_ret[i] = day_ret

    out = frame.copy()
    out["held_w_inda"] = held_w
    out["cash_w"] = cash_w
    out["target_w_pre_vol"] = target_w
    out["leverage"] = lev
    out["daily_ret"] = daily_ret
    out["turnover"] = turnover
    out["rebal"] = rebal_flag

    info = {
        "num_rebalances_total": int(rebal_flag.sum()),
        "first_tradeable_date": str(frame.index[start_idx].date()),
        "rebal_dates_sample": [str(d.date()) for d in rebal_dates[:5]],
    }
    return out, info


# ----------------------------- metrics ------------------------------------ #

def perf_metrics(daily_ret: pd.Series, label: str) -> dict:
    daily_ret = daily_ret.dropna()
    if len(daily_ret) < 30:
        return {
            "label": label,
            "days": int(len(daily_ret)),
            "sharpe": float("nan"),
            "ann_return_pct": float("nan"),
            "ann_vol_pct": float("nan"),
            "max_dd_pct": float("nan"),
            "total_return_pct": float("nan"),
        }
    mu = float(daily_ret.mean())
    sd = float(daily_ret.std(ddof=1))
    ann_ret = mu * TRADING_DAYS_PER_YEAR
    ann_vol = sd * math.sqrt(TRADING_DAYS_PER_YEAR)
    sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else float("nan")
    equity = (1.0 + daily_ret).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min())
    total_ret = float(equity.iloc[-1] - 1.0)
    return {
        "label": label,
        "days": int(len(daily_ret)),
        "sharpe": sharpe,
        "ann_return_pct": float(ann_ret * 100.0),
        "ann_vol_pct": float(ann_vol * 100.0),
        "max_dd_pct": float(max_dd * 100.0),
        "total_return_pct": float(total_ret * 100.0),
    }


def summarize(frame: pd.DataFrame, info: dict) -> dict:
    is_slice = frame.loc[IS_START:IS_END]
    oos_slice = frame.loc[OOS_START:OOS_END]

    is_perf = perf_metrics(is_slice["daily_ret"], "IS_2014_2019")
    oos_perf = perf_metrics(oos_slice["daily_ret"], "OOS_2020_2025")
    full_perf = perf_metrics(frame["daily_ret"].loc[frame["daily_ret"].first_valid_index():], "FULL")

    # Days-in-market: any day INDA weight > 0.01
    in_mkt = (frame["held_w_inda"] > 0.01)
    in_mkt_oos = (oos_slice["held_w_inda"] > 0.01)
    days_in_mkt_pct_oos = float(in_mkt_oos.mean() * 100.0) if len(oos_slice) else float("nan")
    days_in_mkt_pct_full = float(in_mkt.mean() * 100.0) if len(frame) else float("nan")

    num_rebals_oos = int(oos_slice["rebal"].sum())
    num_rebals_is = int(is_slice["rebal"].sum())

    months_oos = float(len(oos_slice) / 21.0)
    months_is = float(len(is_slice) / 21.0)

    sharpe_oos = oos_perf["sharpe"]
    verdict = "PASS" if (np.isfinite(sharpe_oos) and sharpe_oos >= 0.7) else "FAIL"

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "spec": "iter_9_nifty_cross_section_momentum (accessibility fallback: INDA single-asset momentum)",
        "in_sample_window": [IS_START, IS_END],
        "out_of_sample_window": [OOS_START, OOS_END],
        "months_is": months_is,
        "months_oos": months_oos,
        "ann_return_pct": oos_perf["ann_return_pct"],
        "ann_vol_pct": oos_perf["ann_vol_pct"],
        "sharpe_oos": sharpe_oos,
        "max_dd_pct": oos_perf["max_dd_pct"],
        "total_return_pct_oos": oos_perf["total_return_pct"],
        "days_in_market_pct": days_in_mkt_pct_oos,
        "days_in_market_pct_full": days_in_mkt_pct_full,
        "num_rebalances": num_rebals_oos,
        "num_rebalances_is": num_rebals_is,
        "num_rebalances_total": info["num_rebalances_total"],
        "first_tradeable_date": info["first_tradeable_date"],
        "is_perf": is_perf,
        "oos_perf": oos_perf,
        "full_perf": full_perf,
        "verdict": verdict,
        "target_vol_ann": TARGET_VOL,
        "vol_lookback_days": VOL_LOOKBACK,
        "cost_bp_per_rebal_onesided": COST_BP_PER_REBAL,
        "lookbacks_days": {"long_12_1": LB_LONG, "mid_6m": LB_MID, "short_3m": LB_SHORT, "skip": SKIP},
        "leverage_band": [LEV_FLOOR, LEV_CAP],
        "cash_apr_assumed": CASH_APR,
    }
    return summary


# ----------------------------- entry point -------------------------------- #

def main() -> int:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("iter_9 backtest: NIFTY cross-section momentum -- ACCESSIBILITY FALLBACK")
    logger.info(f"Asset: INDA (^NSEI loaded for diagnostics only)")
    logger.info(f"Window: {START} .. {END}")
    logger.info(f"IS: {IS_START}..{IS_END}  OOS: {OOS_START}..{OOS_END}")
    logger.info(
        f"Lookbacks(days): 12-1={LB_LONG}  6m={LB_MID}  3m={LB_SHORT}  skip={SKIP}"
    )
    logger.info(
        f"Vol target: {TARGET_VOL*100:.1f}% ann  lookback={VOL_LOOKBACK}d  lev=[{LEV_FLOOR},{LEV_CAP}]"
    )
    logger.info(f"Rebalance every {REBAL_DAYS} trading days  cost={COST_BP_PER_REBAL} bp one-sided")
    logger.info(f"Cash APR assumed: {CASH_APR*100:.1f}%")
    logger.info("=" * 70)

    prices = load_prices()
    logger.info(f"Price panel: shape={prices.shape}  cols={list(prices.columns)}")
    if "^NSEI" in prices.columns:
        common = prices.dropna()
        if len(common) > 60:
            corr = float(common["INDA"].pct_change().corr(common["^NSEI"].pct_change()))
            logger.info(f"INDA-^NSEI daily-return correlation (overlap N={len(common)}): {corr:.3f}")

    frame, info = run_backtest(prices)
    logger.info(f"Backtest frame: shape={frame.shape}  rebals_total={info['num_rebalances_total']}")

    summary = summarize(frame, info)

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else o,
        )
    )
    logger.info(f"Wrote {SUMMARY_JSON}")

    logger.info("-" * 70)
    logger.info(f"IS  ({IS_START}..{IS_END}) : "
                f"sharpe={summary['is_perf']['sharpe']:.3f}  "
                f"ann_ret={summary['is_perf']['ann_return_pct']:.2f}%  "
                f"vol={summary['is_perf']['ann_vol_pct']:.2f}%  "
                f"maxDD={summary['is_perf']['max_dd_pct']:.2f}%  "
                f"rebals={summary['num_rebalances_is']}")
    logger.info(f"OOS ({OOS_START}..{OOS_END}) : "
                f"sharpe={summary['sharpe_oos']:.3f}  "
                f"ann_ret={summary['ann_return_pct']:.2f}%  "
                f"vol={summary['ann_vol_pct']:.2f}%  "
                f"maxDD={summary['max_dd_pct']:.2f}%  "
                f"rebals={summary['num_rebalances']}")
    logger.info(f"OOS days-in-market: {summary['days_in_market_pct']:.1f}%  months_oos={summary['months_oos']:.1f}")
    logger.info(f"VERDICT={summary['verdict']}  (PASS if OOS sharpe >= 0.70)")
    logger.info(f"DONE in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
