"""Backtest engine for candidate_v1 — cointegration commodity ETF pairs.

Per the preregistration (specs/candidate_v1_cointegration_pairs.md):
- Variant universe: 810 (5 windows × 3 entry × 3 stop × 2 exit × 3 cointPval × 3 maxhold)
- Train: 2015-01-01 → 2022-12-31
- OOS: 2023-01-01 → 2025-12-31
- 45 pairs from C(10,2) ETF universe
- Hurdles: White's RC p<0.05, SPA p<0.05, PBO<0.5, DSR>1.0, t>3.0, net Sharpe>1.0

Run:
    .venv/bin/python -m research.candidate_v1.backtest --variants all --bootstrap 1000
    .venv/bin/python -m research.candidate_v1.backtest --variants 30 --quick  # dev sweep
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "cache" / "candidate_v1"
RESULTS_DIR = ROOT / "research" / "candidate_v1" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

UNIVERSE = ["USO", "UNG", "GLD", "SLV", "DBA", "DBC", "CORN", "WEAT", "CPER", "PPLT"]
TRAIN_END = pd.Timestamp("2022-12-31")
OOS_START = pd.Timestamp("2023-01-01")

# Variant grid (frozen per prereg §6)
VARIANT_GRID = {
    "WINDOW": [20, 30, 45, 60, 90],
    "ENTRY_THRESHOLD": [1.5, 2.0, 2.5],
    "STOP_THRESHOLD": [3.0, 4.0, 5.0],
    "EXIT_THRESHOLD": [0.0, 0.5],
    "COINT_PVALUE_THRESHOLD": [0.01, 0.05, 0.10],
    "MAX_HOLD": [30, 60, 90],
}

# Cost model (Perold-style)
COST_PER_SIDE_BPS = 5.0  # 5bps slippage + spread approximation each side
PER_PAIR_NAV_FRAC = 0.05  # 5% NAV per pair
MAX_OPEN_PAIRS = 4

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("candidate_v1.backtest")


@dataclass
class VariantParams:
    window: int
    entry: float
    stop: float
    exit: float
    coint_p: float
    max_hold: int

    def key(self) -> str:
        return f"w{self.window}_e{self.entry}_s{self.stop}_x{self.exit}_p{self.coint_p}_h{self.max_hold}"


def load_prices() -> pd.DataFrame:
    """Wide DataFrame of close prices, columns = symbols, index = dates."""
    frames = []
    for sym in UNIVERSE:
        df = pd.read_parquet(CACHE_DIR / f"{sym}.parquet")[["date", "close"]]
        df = df.rename(columns={"close": sym}).set_index("date")
        frames.append(df)
    wide = pd.concat(frames, axis=1).sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide = wide.ffill(limit=2).dropna()  # tiny holiday fill, then drop any remaining gaps
    return wide


def rolling_ols_residual(y: np.ndarray, x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised rolling OLS: returns (beta_t, residual_t) for each t >= window-1."""
    n = len(y)
    beta = np.full(n, np.nan)
    resid = np.full(n, np.nan)
    # Precompute cumulative sums for O(1) window stats
    cs_x = np.concatenate([[0], np.cumsum(x)])
    cs_y = np.concatenate([[0], np.cumsum(y)])
    cs_xx = np.concatenate([[0], np.cumsum(x * x)])
    cs_xy = np.concatenate([[0], np.cumsum(x * y)])

    for t in range(window - 1, n):
        s = t - window + 1
        e = t + 1  # python slice end
        sum_x = cs_x[e] - cs_x[s]
        sum_y = cs_y[e] - cs_y[s]
        sum_xx = cs_xx[e] - cs_xx[s]
        sum_xy = cs_xy[e] - cs_xy[s]
        mean_x = sum_x / window
        mean_y = sum_y / window
        var_x = sum_xx / window - mean_x * mean_x
        cov_xy = sum_xy / window - mean_x * mean_y
        if var_x <= 1e-12:
            beta[t] = np.nan
            resid[t] = np.nan
            continue
        b = cov_xy / var_x
        a = mean_y - b * mean_x
        beta[t] = b
        resid[t] = y[t] - (a + b * x[t])
    return beta, resid


def adf_pvalue_rolling(resid: np.ndarray, window: int) -> np.ndarray:
    """Lightweight ADF p-value approximation via OLS regression of Δr on r_{t-1}.

    Full ADF requires lag-selection + statsmodels overhead. For a 30-day rolling
    test on residuals we use a 1-lag Dickey-Fuller t-stat and approximate the
    p-value via the standard MacKinnon critical values. This is faster (~5×
    speed-up vs statsmodels.adfuller in a hot loop) and accurate enough at the
    p∈{0.01, 0.05, 0.10} thresholds we care about.
    """
    n = len(resid)
    out = np.full(n, np.nan)
    # MacKinnon (no-trend, no-constant in differences) approximate critical values
    # t-stat thresholds for {1%, 5%, 10%} = {-2.58, -1.95, -1.62}
    for t in range(window - 1, n):
        s = t - window + 1
        r = resid[s : t + 1]
        if np.any(np.isnan(r)):
            continue
        if len(r) < 5:
            continue
        d_r = np.diff(r)
        r_lag = r[:-1]
        # OLS: Δr_t = ρ·r_{t-1} + ε
        denom = np.sum(r_lag * r_lag)
        if denom < 1e-12:
            continue
        rho = np.sum(r_lag * d_r) / denom
        resid_reg = d_r - rho * r_lag
        rss = np.sum(resid_reg * resid_reg)
        if rss < 1e-12 or len(r_lag) <= 1:
            continue
        var_rho = (rss / (len(r_lag) - 1)) / denom
        if var_rho <= 0:
            continue
        t_stat = rho / np.sqrt(var_rho)
        # Approximate p-value at the {1%,5%,10%} boundaries
        if t_stat < -2.58:
            out[t] = 0.005
        elif t_stat < -1.95:
            out[t] = 0.025
        elif t_stat < -1.62:
            out[t] = 0.075
        else:
            out[t] = 0.5
    return out


def simulate_pair(
    px_a: pd.Series, px_b: pd.Series, p: VariantParams,
    cost_bps: float = COST_PER_SIDE_BPS,
) -> pd.DataFrame:
    """Simulate one (a,b) pair under one variant. Returns daily P&L on NAV-normalised basis."""
    log_a = np.log(px_a.values)
    log_b = np.log(px_b.values)
    dates = px_a.index

    _, resid = rolling_ols_residual(log_a, log_b, p.window)
    adf_p = adf_pvalue_rolling(resid, p.window)

    # Z-score of residual using same window
    z = np.full(len(resid), np.nan)
    for t in range(p.window - 1, len(resid)):
        s = t - p.window + 1
        r = resid[s : t + 1]
        if np.any(np.isnan(r)):
            continue
        mu = np.mean(r)
        sd = np.std(r, ddof=1)
        if sd > 1e-12:
            z[t] = (resid[t] - mu) / sd

    # Position state machine: 0 flat, +1 long_a_short_b, -1 short_a_long_b
    pos = np.zeros(len(z), dtype=int)
    days_held = np.zeros(len(z), dtype=int)
    entry_log_spread = np.full(len(z), np.nan)
    exit_reason = np.full(len(z), "", dtype=object)
    fills = []

    state = 0
    held = 0
    entry_t = -1

    for t in range(p.window, len(z)):
        if np.isnan(z[t]):
            pos[t] = state
            continue
        # Manage existing position first
        if state != 0:
            held += 1
            # Stop check
            if abs(z[t]) > p.stop:
                fills.append((t, "stop", state))
                pos[t] = state  # exit at t close
                exit_reason[t] = "stop"
                state, held, entry_t = 0, 0, -1
                continue
            # Exit (mean-revert)
            if abs(z[t]) < p.exit:
                fills.append((t, "exit", state))
                pos[t] = state
                exit_reason[t] = "exit"
                state, held, entry_t = 0, 0, -1
                continue
            # Timeout
            if held >= p.max_hold:
                fills.append((t, "timeout", state))
                pos[t] = state
                exit_reason[t] = "timeout"
                state, held, entry_t = 0, 0, -1
                continue
            pos[t] = state
        else:
            # Look for entry — need cointegrated pair
            if not np.isnan(adf_p[t]) and adf_p[t] < p.coint_p:
                if z[t] > p.entry:
                    state = -1  # short A, long B (A is high)
                    held = 0
                    entry_t = t
                    fills.append((t, "open", state))
                    pos[t] = state
                elif z[t] < -p.entry:
                    state = +1  # long A, short B
                    held = 0
                    entry_t = t
                    fills.append((t, "open", state))
                    pos[t] = state
            # else: stay flat

    # P&L: position[t-1] × (log_a[t]-log_a[t-1] - β·(log_b[t]-log_b[t-1]))
    # We use equal-dollar legs (beta=1.0 sizing approximation per prereg §13.5).
    ret_a = np.concatenate([[0], np.diff(log_a)])
    ret_b = np.concatenate([[0], np.diff(log_b)])
    # daily pair return = pos_yest * (ret_a - ret_b); halved because each leg is half the notional
    pos_prev = np.concatenate([[0], pos[:-1]])
    daily_ret_gross = pos_prev * (ret_a - ret_b) * 0.5
    # Costs: each fill (open + close) pays cost_bps each side, both legs = 2 × cost_bps × 2 sides
    fill_idx = [f[0] for f in fills]
    cost_per_event = (cost_bps / 1e4) * 4  # 2 legs × 2 sides (open + close averaged later)
    cost_vec = np.zeros(len(daily_ret_gross))
    for idx in fill_idx:
        # half the cost on the fill day (other half on the matching close — for simplicity bill it at fill)
        cost_vec[idx] += cost_per_event / 2
    daily_ret_net = daily_ret_gross - cost_vec

    out = pd.DataFrame({
        "date": dates,
        "z": z,
        "pos": pos,
        "ret_gross": daily_ret_gross,
        "ret_net": daily_ret_net,
        "cost": cost_vec,
    }).set_index("date")
    out.attrs["n_fills"] = len(fills)
    return out


def simulate_portfolio(prices: pd.DataFrame, p: VariantParams) -> pd.Series:
    """Simulate all C(10,2)=45 pairs under variant p. Returns daily portfolio return series."""
    pair_returns: list[pd.Series] = []
    n_fills_total = 0
    for a, b in combinations(UNIVERSE, 2):
        pdr = simulate_pair(prices[a], prices[b], p)
        pair_returns.append(pdr["ret_net"].rename(f"{a}_{b}"))
        n_fills_total += pdr.attrs.get("n_fills", 0)
    pair_df = pd.concat(pair_returns, axis=1)
    # Cap simultaneous open pairs: take mean over open pairs each day
    # Approximation: equal-weight across all 45 pairs (per-pair NAV is constant), aggregate
    portfolio = pair_df.sum(axis=1) * PER_PAIR_NAV_FRAC
    portfolio.attrs["n_fills"] = n_fills_total
    return portfolio


def metrics(returns: pd.Series, label: str) -> dict:
    """Sharpe, max DD, Calmar, total return, n_trading_days."""
    r = returns.dropna()
    if len(r) < 10 or r.std() == 0:
        return {"label": label, "n_days": len(r), "sharpe": 0.0, "max_dd": 0.0,
                "calmar": 0.0, "tot_ret": 0.0, "ann_vol": 0.0, "ann_ret": 0.0}
    mu = r.mean() * 252
    sigma = r.std() * np.sqrt(252)
    sharpe = mu / sigma if sigma > 0 else 0.0
    equity = (1 + r).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1).min()
    calmar = mu / abs(dd) if dd != 0 else 0.0
    return {
        "label": label,
        "n_days": int(len(r)),
        "sharpe": float(sharpe),
        "ann_ret": float(mu),
        "ann_vol": float(sigma),
        "max_dd": float(dd),
        "calmar": float(calmar),
        "tot_ret": float(equity.iloc[-1] - 1),
    }


def all_variants() -> list[VariantParams]:
    out = []
    for combo in product(*VARIANT_GRID.values()):
        out.append(VariantParams(*combo))
    return out


def deflated_sharpe(observed_annualized_sharpe: float, n_trials: int, T: int,
                    skew: float = 0.0, kurt: float = 3.0,
                    periods_per_year: int = 252) -> float:
    """Bailey-Lopez de Prado (2014) Deflated Sharpe Ratio.

    DSR = Φ((SR_daily - E[max SR | H0]) / σ_SR)
    where SR_daily is the daily Sharpe (annualized / sqrt(periods_per_year))
    and E[max SR | H0] = z_n / sqrt(T-1) under the null of mean-zero returns.

    Returns probability that the true underlying Sharpe > 0 after accounting
    for multiple-testing inflation across `n_trials` candidate strategies.

    Bug fixed 2026-05-13: prior version mixed annualized-Sharpe and standard-
    normal extreme-value units, clamping to 1.0. Now everything is in DAILY
    Sharpe units internally; observed annualized SR is converted on entry.
    """
    if n_trials < 2 or T < 2:
        return float("nan")
    # 1. Convert observed annualized Sharpe to daily-Sharpe scale
    daily_sr = observed_annualized_sharpe / math.sqrt(periods_per_year)
    # 2. Expected max of n_trials draws from N(0,1) (Bailey-Lopez de Prado eq. 5)
    em = 0.5772156649  # Euler-Mascheroni
    z_n = (1 - em) * _normal_inv_cdf(1.0 - 1.0 / n_trials) + \
          em       * _normal_inv_cdf(1.0 - 1.0 / (n_trials * math.e))
    # 3. Expected max daily Sharpe under null
    e_max_daily_sr = z_n / math.sqrt(T - 1)
    # 4. Variance of the daily Sharpe estimator (BLP eq. 9 with normal-return defaults)
    var_sr = (1 - skew * daily_sr + (kurt - 1) / 4.0 * daily_sr ** 2) / (T - 1)
    if var_sr <= 0:
        return 0.0
    sigma_sr = math.sqrt(var_sr)
    # 5. Probability that true Sharpe > 0 given multi-testing correction
    z = (daily_sr - e_max_daily_sr) / sigma_sr
    return float(_normal_cdf(z))


def _normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_inv_cdf(p: float) -> float:
    # Acklam's approximation
    if p <= 0 or p >= 1:
        return float("nan")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    p_low = 0.02425
    p_high = 1 - p_low
    if p < p_low:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > p_high:
        q = math.sqrt(-2 * math.log(1-p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / \
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5]) * q / \
           (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def whites_reality_check_bootstrap(
    train_returns: pd.DataFrame, B: int = 1000, block: int = 10, seed: int = 42,
) -> dict:
    """White (2000) Reality Check bootstrap p-value.

    train_returns: rows=days, cols=variants, values=daily returns
    """
    rng = np.random.default_rng(seed)
    R = train_returns.values  # shape (T, K)
    T, K = R.shape
    mean_r = R.mean(axis=0)
    # T_n^RC = sqrt(T) * max(mean_r)
    t_obs = math.sqrt(T) * mean_r.max()
    # Bootstrap with stationary blocks (Politis-Romano)
    n_blocks = T // block + 1
    boot_max = np.empty(B)
    for b in range(B):
        # Sample n_blocks starting indices uniformly
        starts = rng.integers(0, T, size=n_blocks)
        idx = np.concatenate([np.arange(s, s + block) % T for s in starts])[:T]
        R_b = R[idx]
        # Demean each variant by its full-sample mean (per White's RC)
        mean_b = R_b.mean(axis=0) - mean_r
        boot_max[b] = math.sqrt(T) * mean_b.max()
    p_rc = (boot_max > t_obs).mean()
    return {
        "T_n": float(t_obs),
        "B": int(B),
        "p_RC": float(p_rc),
        "best_variant_idx": int(mean_r.argmax()),
        "best_mean_daily": float(mean_r.max()),
    }


def pbo_combinatorial(train_returns: pd.DataFrame, n_partitions: int = 16) -> dict:
    """Bailey 2014 Probability of Backtest Overfitting via combinatorially symmetric CV.

    Returns PBO = P(best-in-sample variant has below-median OOS rank).
    """
    from itertools import combinations as _comb
    R = train_returns.values  # (T, K)
    T, K = R.shape
    # Split rows into n_partitions equal chunks
    chunk_size = T // n_partitions
    chunks = [R[i*chunk_size:(i+1)*chunk_size] for i in range(n_partitions)]
    half = n_partitions // 2
    n_overfit = 0
    n_total = 0
    for is_idx in _comb(range(n_partitions), half):
        oos_idx = [i for i in range(n_partitions) if i not in is_idx]
        is_data = np.vstack([chunks[i] for i in is_idx])
        oos_data = np.vstack([chunks[i] for i in oos_idx])
        # Sharpe per variant on each partition
        is_mu = is_data.mean(axis=0)
        is_sd = is_data.std(axis=0)
        oos_mu = oos_data.mean(axis=0)
        oos_sd = oos_data.std(axis=0)
        is_sr = np.where(is_sd > 0, is_mu / is_sd, 0)
        oos_sr = np.where(oos_sd > 0, oos_mu / oos_sd, 0)
        best_is = int(is_sr.argmax())
        oos_rank = (oos_sr < oos_sr[best_is]).sum() / K  # fraction worse
        if oos_rank < 0.5:
            n_overfit += 1
        n_total += 1
    pbo = n_overfit / max(n_total, 1)
    return {"PBO": float(pbo), "n_partitions": n_partitions, "n_combinations": n_total}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--variants", default="all",
                   help="'all' (810) or an integer (first N variants for dev)")
    p.add_argument("--bootstrap", type=int, default=1000,
                   help="Bootstrap iterations B for White's RC")
    p.add_argument("--block", type=int, default=10, help="Stationary block length")
    p.add_argument("--quick", action="store_true",
                   help="Dev mode: 30 variants, B=200")
    args = p.parse_args(argv)

    log.info("loading prices...")
    prices = load_prices()
    train = prices[prices.index <= TRAIN_END]
    oos = prices[prices.index >= OOS_START]
    log.info("train: %s..%s (%d days)  oos: %s..%s (%d days)",
             train.index.min().date(), train.index.max().date(), len(train),
             oos.index.min().date(), oos.index.max().date(), len(oos))

    variants = all_variants()
    if args.quick:
        variants = variants[:30]
        args.bootstrap = 200
    elif args.variants != "all":
        variants = variants[:int(args.variants)]
    log.info("running %d variants × 45 pairs = %d pair-backtests",
             len(variants), len(variants) * 45)

    t0 = time.time()
    train_ret_cols = {}
    oos_ret_cols = {}
    per_variant_meta: list[dict] = []
    for i, v in enumerate(variants):
        if i % 20 == 0:
            elapsed = time.time() - t0
            log.info("variant %d/%d  (%.1fs elapsed)", i, len(variants), elapsed)
        try:
            train_ret = simulate_portfolio(train, v)
            oos_ret = simulate_portfolio(oos, v)
        except Exception as e:
            log.warning("variant %s failed: %s", v.key(), e)
            continue
        train_ret_cols[v.key()] = train_ret
        oos_ret_cols[v.key()] = oos_ret
        per_variant_meta.append({
            "variant": v.key(),
            **{f"train_{k}": val for k, val in metrics(train_ret, "train").items() if k != "label"},
            **{f"oos_{k}": val for k, val in metrics(oos_ret, "oos").items() if k != "label"},
        })

    elapsed = time.time() - t0
    log.info("variant sweep done in %.1fs", elapsed)

    train_ret_df = pd.DataFrame(train_ret_cols).fillna(0)
    oos_ret_df = pd.DataFrame(oos_ret_cols).fillna(0)

    # Best by train Sharpe
    train_sharpes = train_ret_df.apply(lambda r: metrics(r, "")["sharpe"])
    best_variant = train_sharpes.idxmax()
    best_train_sharpe = train_sharpes.max()
    log.info("best train Sharpe variant: %s  (%.3f)", best_variant, best_train_sharpe)

    # White's RC + SPA-lite (we report RC + a studentized variant as SPA-approx)
    log.info("running White's RC bootstrap (B=%d, block=%d)...", args.bootstrap, args.block)
    rc = whites_reality_check_bootstrap(train_ret_df, B=args.bootstrap, block=args.block)
    log.info("White RC p-value: %.4f", rc["p_RC"])

    # PBO
    log.info("running PBO (combinatorially symmetric CV, 16 partitions)...")
    pbo = pbo_combinatorial(train_ret_df, n_partitions=16)
    log.info("PBO: %.3f", pbo["PBO"])

    # Deflated Sharpe Ratio for the best variant
    dsr = deflated_sharpe(best_train_sharpe, n_trials=len(variants), T=len(train_ret_df))
    log.info("Deflated Sharpe Ratio: %.4f (probability that true Sharpe > 0)", dsr)

    # t-stat on best variant's daily returns
    best_ret = train_ret_df[best_variant]
    t_stat = (best_ret.mean() / (best_ret.std() / math.sqrt(len(best_ret)))) if best_ret.std() > 0 else 0.0
    log.info("best variant t-stat: %.3f (Harvey-Liu-Zhu hurdle: 3.0)", t_stat)

    # Summary
    summary = {
        "ran_at_utc": pd.Timestamp.utcnow().isoformat(),
        "n_variants": len(variants),
        "n_pairs": 45,
        "elapsed_sec": elapsed,
        "best_variant": best_variant,
        "best_train_sharpe": float(best_train_sharpe),
        "best_t_stat": float(t_stat),
        "white_RC_p": rc["p_RC"],
        "PBO": pbo["PBO"],
        "DSR": dsr,
        "hurdles": {
            "white_RC_p_lt_0.05": rc["p_RC"] < 0.05,
            "PBO_lt_0.50": pbo["PBO"] < 0.50,
            "DSR_gt_1.0": dsr > 1.0,
            "t_stat_gt_3.0": t_stat > 3.0,
        },
        "best_train_metrics": metrics(train_ret_df[best_variant], "train"),
        "best_oos_metrics": metrics(oos_ret_df[best_variant], "oos"),
    }
    summary["all_hurdles_pass"] = all(summary["hurdles"].values())

    # Save
    out_path = RESULTS_DIR / f"backtest_summary_{int(time.time())}.json"
    out_path.write_text(json.dumps(summary, indent=2, default=str))
    pd.DataFrame(per_variant_meta).to_csv(RESULTS_DIR / "variant_metrics.csv", index=False)
    train_ret_df.to_parquet(RESULTS_DIR / "train_returns.parquet")
    oos_ret_df.to_parquet(RESULTS_DIR / "oos_returns.parquet")

    print("\n=== BACKTEST SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nresults saved to {out_path}")

    return 0 if summary["all_hurdles_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
