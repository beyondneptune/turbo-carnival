"""Phase 3 backtest for candidate_v5 — TS-MOM × HRP × Regime Gate.

Reuses v3 ETF data (already in cache) and FRED macro series. Tests 432 variants
of the combined strategy with full multi-testing correction (White's RC + PBO
+ DSR + Harvey-Liu-Zhu t > 3 hurdle) and comparison against TWO benchmarks:
naive 1/N and SPY buy-and-hold.

Run:
    .venv/bin/python -m research.candidate_v5.backtest --quick   # 12 variants
    .venv/bin/python -m research.candidate_v5.backtest           # full 432
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from research.candidate_v1.backtest import (
    deflated_sharpe, whites_reality_check_bootstrap,
    pbo_combinatorial, metrics,
)
from research.candidate_v3.hrp import hrp_weights, equal_weight

ROOT = Path(__file__).resolve().parents[2]
V3_CACHE = ROOT / "data" / "cache" / "candidate_v3"
FRED_DIR = ROOT / "data" / "offline_context" / "fred"
RESULTS_DIR = ROOT / "research" / "candidate_v5" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = ["SPY","QQQ","IWM","EFA","EEM","TLT","IEF","LQD","HYG","GLD","DBC","VNQ"]
TRAIN_END = pd.Timestamp("2022-12-31")
OOS_START = pd.Timestamp("2023-01-01")

COST_PER_SIDE_BPS = 5.0

VARIANT_GRID = {
    "LOOKBACK_DAYS": [126, 189, 252],          # 6 / 9 / 12 months
    "HRP_LOOKBACK": [252, 504],
    "REGIME_STRESS_COUNT": [1, 2, 3],
    "DEFENSIVE_EXPOSURE_PCT": [0.0, 0.3, 0.5, 0.7],
    "VIX_STRESS_TH": [25.0, 30.0, 35.0],
    "REBAL_FREQ": ["M", "Q"],
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("candidate_v5.backtest")


@dataclass
class V5Variant:
    lookback: int
    hrp_lookback: int
    regime_count: int
    defensive_pct: float
    vix_th: float
    rebal_freq: str

    def key(self) -> str:
        return (f"L{self.lookback}_H{self.hrp_lookback}_R{self.regime_count}"
                f"_D{self.defensive_pct}_V{self.vix_th}_F{self.rebal_freq}")


def all_variants() -> list[V5Variant]:
    return [V5Variant(*c) for c in product(*VARIANT_GRID.values())]


def load_prices() -> pd.DataFrame:
    frames = []
    for sym in UNIVERSE:
        df = pd.read_parquet(V3_CACHE / f"{sym}.parquet")[["date", "close"]]
        df = df.rename(columns={"close": sym}).set_index("date")
        frames.append(df)
    wide = pd.concat(frames, axis=1).sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide = wide.ffill(limit=2).dropna()
    return wide


def load_fred_series(name: str) -> pd.Series:
    """Load one FRED series and return as a daily-aligned Series."""
    p = FRED_DIR / f"{name}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"FRED series missing: {p}")
    df = pd.read_parquet(p)
    # Schema is typically (date, value) — try both layouts
    if "value" in df.columns:
        df = df[["date","value"]] if "date" in df.columns else df.reset_index()[["index","value"]].rename(columns={"index":"date"})
    elif "close" in df.columns:
        df = df[["date","close"]].rename(columns={"close":"value"})
    else:
        # Assume the series is the FRED-standard layout from fredapi
        df = df.reset_index()
        date_col = df.columns[0]
        val_col = df.columns[1]
        df = df[[date_col, val_col]].rename(columns={date_col:"date", val_col:"value"})
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")["value"].sort_index()


def build_regime_gate(prices_index: pd.DatetimeIndex, vix_th: float,
                      regime_count: int) -> pd.Series:
    """Return a boolean Series indexed by `prices_index`: True = stress regime."""
    try:
        vix = load_fred_series("VIXCLS")
    except FileNotFoundError:
        log.warning("VIXCLS missing — falling back to no VIX signal")
        vix = pd.Series(np.nan, index=prices_index)
    try:
        curve = load_fred_series("T10Y2Y")
    except FileNotFoundError:
        curve = pd.Series(np.nan, index=prices_index)
    try:
        nfci = load_fred_series("NFCI")
    except FileNotFoundError:
        nfci = pd.Series(np.nan, index=prices_index)
    try:
        rec = load_fred_series("USREC")
    except FileNotFoundError:
        rec = pd.Series(np.nan, index=prices_index)

    # Reindex to daily, forward-fill so the signal is "asof today"
    vix_a = vix.reindex(prices_index, method="ffill")
    curve_a = curve.reindex(prices_index, method="ffill")
    nfci_a = nfci.reindex(prices_index, method="ffill")
    rec_a = rec.reindex(prices_index, method="ffill")

    # Stress signals
    s1 = (vix_a > vix_th).fillna(False).astype(int)
    s2 = (curve_a < 0).fillna(False).astype(int)   # inverted curve
    s3 = (nfci_a > 0.5).fillna(False).astype(int)  # tight financial conditions
    s4 = (rec_a == 1).fillna(False).astype(int)    # NBER recession

    stress_count = s1 + s2 + s3 + s4
    return (stress_count >= regime_count).astype(bool)


def load_rf_rate() -> pd.Series:
    """Daily risk-free rate from DGS3MO (annualised %). Convert to daily fraction."""
    try:
        s = load_fred_series("DGS3MO")
    except FileNotFoundError:
        log.warning("DGS3MO missing — using zero risk-free rate")
        return pd.Series(dtype=float)
    # FRED reports as annual percent; convert to daily fraction
    return (s / 100.0) / 252.0


def rebalance_dates(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    if freq == "M":
        marker = index.to_series().groupby([index.year, index.month]).last()
    elif freq == "Q":
        marker = index.to_series().groupby([index.year, index.quarter]).last()
    else:
        raise ValueError(freq)
    return pd.DatetimeIndex(marker.values)


def simulate_v5(prices: pd.DataFrame, v: V5Variant,
                cost_bps: float = COST_PER_SIDE_BPS) -> pd.Series:
    """Walk-forward simulation of TS-MOM × HRP × Regime Gate."""
    returns = prices.pct_change().dropna()
    rf_daily = load_rf_rate()
    rf_aligned = rf_daily.reindex(returns.index, method="ffill").fillna(0.0)
    stress = build_regime_gate(returns.index, v.vix_th, v.regime_count)

    rebal = rebalance_dates(returns.index, v.rebal_freq)
    # Need enough warmup for both TS-MOM and HRP
    warmup = max(v.lookback, v.hrp_lookback) + 5
    rebal = rebal[rebal >= returns.index[0] + pd.Timedelta(days=warmup)]
    if len(rebal) < 4:
        return pd.Series(0.0, index=returns.index)

    weights_ts = pd.DataFrame(0.0, index=returns.index, columns=returns.columns)
    prev_w = pd.Series(0.0, index=returns.columns)
    turnover_per_rebal = []

    for rd in rebal:
        window_end = returns.index[returns.index <= rd][-1]
        end_idx = returns.index.get_loc(window_end)
        if end_idx < warmup:
            continue

        # TS-MOM: 12m excess return per asset
        lookback_start = end_idx - v.lookback
        if lookback_start < 0:
            continue
        r_period = (prices.iloc[end_idx] / prices.iloc[lookback_start]) - 1
        rf_period = rf_aligned.iloc[lookback_start:end_idx + 1].sum()
        excess = r_period - rf_period

        survivors = [s for s in UNIVERSE if excess[s] > 0]

        # HRP weights on survivors
        if len(survivors) == 0:
            w = pd.Series(dtype=float)
        elif len(survivors) == 1:
            w = pd.Series([1.0], index=survivors)
        else:
            hrp_window = returns[survivors].iloc[end_idx - v.hrp_lookback:end_idx + 1]
            try:
                w = hrp_weights(hrp_window, linkage_method="single",
                                weight_floor=0.02, weight_cap=0.25)
            except Exception:
                # Fall back to equal weight on survivors if HRP fails
                w = pd.Series(1.0 / len(survivors), index=survivors)

        # Regime gate
        is_stress_today = bool(stress.iloc[end_idx])
        if is_stress_today:
            w = w * v.defensive_pct

        # Build full-universe weight vector (zeros for non-survivors)
        full_w = pd.Series(0.0, index=returns.columns)
        for sym in w.index:
            full_w[sym] = w[sym]

        # Apply weights from next bar until next rebal
        next_idx = end_idx + 1
        next_rd_idx_list = [i for i, x in enumerate(rebal) if x == rd]
        if next_rd_idx_list and next_rd_idx_list[0] < len(rebal) - 1:
            next_rd = rebal[next_rd_idx_list[0] + 1]
            end_apply = returns.index.get_loc(returns.index[returns.index <= next_rd][-1])
        else:
            end_apply = len(returns)

        for col in full_w.index:
            weights_ts.iloc[next_idx:end_apply, weights_ts.columns.get_loc(col)] = full_w[col]

        turnover = (full_w - prev_w).abs().sum()
        turnover_per_rebal.append(float(turnover))
        prev_w = full_w

    # Daily portfolio return + costs
    port_ret = (weights_ts * returns).sum(axis=1)
    cost_series = pd.Series(0.0, index=returns.index)
    for rd, turn in zip(rebal, turnover_per_rebal):
        cands = returns.index[returns.index > rd]
        if len(cands):
            cost_series.loc[cands[0]] = turn * (cost_bps / 1e4) * 2

    port_ret_net = port_ret - cost_series
    port_ret_net.attrs["mean_turnover"] = float(np.mean(turnover_per_rebal)) if turnover_per_rebal else 0.0
    port_ret_net.attrs["n_rebal"] = len(rebal)
    return port_ret_net


def simulate_equal_weight(prices: pd.DataFrame, rebal_freq: str = "M",
                          cost_bps: float = COST_PER_SIDE_BPS) -> pd.Series:
    returns = prices.pct_change().dropna()
    n = len(returns.columns)
    rebal = rebalance_dates(returns.index, rebal_freq)
    weights = pd.DataFrame(1.0/n, index=returns.index, columns=returns.columns)
    port = (weights * returns).sum(axis=1)
    # Negligible turnover for 1/N held — only price drift
    return port


def simulate_spy_buyhold(prices: pd.DataFrame) -> pd.Series:
    return prices["SPY"].pct_change().dropna()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="all")
    p.add_argument("--bootstrap", type=int, default=2000)
    p.add_argument("--block", type=int, default=10)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    log.info("loading prices (v5 uses v3 cache, 12 ETFs)...")
    prices = load_prices()
    train = prices[prices.index <= TRAIN_END]
    oos = prices[prices.index >= OOS_START]
    log.info("train: %s..%s (%d days)  oos: %s..%s (%d days)",
             train.index.min().date(), train.index.max().date(), len(train),
             oos.index.min().date(), oos.index.max().date(), len(oos))

    # Benchmarks
    log.info("computing benchmarks (1/N + SPY buy-hold)...")
    ew_train = simulate_equal_weight(train, "M")
    ew_oos = simulate_equal_weight(oos, "M")
    spy_train = simulate_spy_buyhold(train)
    spy_oos = simulate_spy_buyhold(oos)
    ew_oos_m = metrics(ew_oos, "1N_oos")
    spy_oos_m = metrics(spy_oos, "SPY_oos")
    log.info("1/N OOS Sharpe: %.3f  SPY OOS Sharpe: %.3f",
             ew_oos_m["sharpe"], spy_oos_m["sharpe"])

    variants = all_variants()
    if args.quick:
        variants = variants[:12]; args.bootstrap = 200
    elif args.variants != "all":
        variants = variants[:int(args.variants)]
    log.info("running %d v5 variants", len(variants))

    t0 = time.time()
    train_ret_cols, oos_ret_cols = {}, {}
    per_variant_meta = []
    for i, v in enumerate(variants):
        if i % 20 == 0:
            log.info("variant %d/%d (%.1fs)", i, len(variants), time.time() - t0)
        try:
            tr = simulate_v5(train, v)
            oo = simulate_v5(oos, v)
        except Exception as e:
            log.warning("variant %s failed: %s", v.key(), e); continue
        train_ret_cols[v.key()] = tr
        oos_ret_cols[v.key()] = oo
        per_variant_meta.append({
            "variant": v.key(),
            "mean_turnover": tr.attrs.get("mean_turnover", 0.0),
            "n_rebal": tr.attrs.get("n_rebal", 0),
            **{f"train_{k}": val for k, val in metrics(tr, "train").items() if k != "label"},
            **{f"oos_{k}": val for k, val in metrics(oo, "oos").items() if k != "label"},
        })

    elapsed = time.time() - t0
    log.info("variant sweep done in %.1fs", elapsed)

    train_ret_df = pd.DataFrame(train_ret_cols).fillna(0)
    oos_ret_df = pd.DataFrame(oos_ret_cols).fillna(0)

    # Best by train Sharpe
    train_sharpes = train_ret_df.apply(lambda r: metrics(r, "")["sharpe"])
    best_variant_key = train_sharpes.idxmax()
    best_train_sharpe = train_sharpes.max()
    log.info("best v5 variant: %s (train Sharpe %.3f)", best_variant_key, best_train_sharpe)

    log.info("White RC bootstrap (B=%d)...", args.bootstrap)
    rc = whites_reality_check_bootstrap(train_ret_df, B=args.bootstrap, block=args.block)
    log.info("White RC p-value: %.4f", rc["p_RC"])

    log.info("PBO (16 partitions)...")
    pbo = pbo_combinatorial(train_ret_df, n_partitions=16)
    log.info("PBO: %.3f", pbo["PBO"])

    dsr = deflated_sharpe(best_train_sharpe, n_trials=len(variants), T=len(train_ret_df))
    log.info("DSR: %.4f", dsr)

    best_train = train_ret_df[best_variant_key]
    best_oos = oos_ret_df[best_variant_key]
    t_stat = (best_train.mean() / (best_train.std() / math.sqrt(len(best_train)))) if best_train.std() > 0 else 0.0
    best_oos_m = metrics(best_oos, "oos")
    best_train_m = metrics(best_train, "train")

    summary = {
        "ran_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "candidate": "v5_tsmom_hrp_regime",
        "n_variants": len(variants),
        "elapsed_sec": elapsed,
        "best_variant": best_variant_key,
        "best_train_sharpe": float(best_train_sharpe),
        "best_t_stat": float(t_stat),
        "white_RC_p": rc["p_RC"],
        "PBO": pbo["PBO"],
        "DSR": dsr,
        "benchmarks_oos": {
            "1N": ew_oos_m,
            "SPY_buyhold": spy_oos_m,
        },
        "best_train_metrics": best_train_m,
        "best_oos_metrics": best_oos_m,
        "vs_1N_sharpe_diff_oos": float(best_oos_m["sharpe"] - ew_oos_m["sharpe"]),
        "vs_SPY_sharpe_diff_oos": float(best_oos_m["sharpe"] - spy_oos_m["sharpe"]),
        "hurdles": {
            "white_RC_p_lt_0.05": rc["p_RC"] < 0.05,
            "PBO_lt_0.50": pbo["PBO"] < 0.50,
            "DSR_gt_0.80": dsr > 0.80,
            "t_stat_gt_3.0": t_stat > 3.0,
            "OOS_sharpe_gt_1.0": best_oos_m["sharpe"] > 1.0,
            "OOS_sharpe_beats_1N_by_0.10": (best_oos_m["sharpe"] - ew_oos_m["sharpe"]) >= 0.10,
            "OOS_sharpe_beats_SPY_by_0.10": (best_oos_m["sharpe"] - spy_oos_m["sharpe"]) >= 0.10,
            "OOS_max_DD_lt_25pct": abs(best_oos_m["max_dd"]) < 0.25,
            "OOS_calmar_gt_1.0": best_oos_m["calmar"] > 1.0,
            "shrinkage_lt_50pct": abs(best_train_sharpe - best_oos_m["sharpe"]) / max(abs(best_train_sharpe), 1e-9) < 0.5,
        },
    }
    summary["all_hurdles_pass"] = all(summary["hurdles"].values())

    out = RESULTS_DIR / f"backtest_summary_{int(time.time())}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame(per_variant_meta).to_csv(RESULTS_DIR / "variant_metrics.csv", index=False)
    train_ret_df.to_parquet(RESULTS_DIR / "train_returns.parquet")
    oos_ret_df.to_parquet(RESULTS_DIR / "oos_returns.parquet")

    print("\n=== CANDIDATE V5 BACKTEST SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["all_hurdles_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
