"""
iter_13 - HMM Regime-Conditioned Ensemble (SPY + sector ETFs)

Spec: research/candidates/iter_13_hmm_regime_conditioned_ensemble.md

Approach (per assignment from Backtest Agent A8 brief):
- Pull daily prices via yfinance: SPY, XLK, XLF, XLE, XLP, ^VIX
- Period 2010-01-01 to today
- Cache: data/cache/spy_sectors_vix_daily.parquet
- HMM features: daily SPY return, VIX level, MA50/MA200 cross indicator
- 3-state Gaussian HMM (hmmlearn GaussianHMM, n_components=3, full covariance)
- Train HMM on 2010-2018; freeze parameters; apply to 2019-2025 OOS
- Label regimes by mean SPY return: Trending-Up / Trending-Down / Choppy
- Per-regime substrategy:
  * Trending-Up: TSMOM on sector ETFs (sign of 1m return), long-only, equal-weight top 3 sectors
  * Trending-Down: 100% cash
  * Choppy: mean-rev on SPY (RSI(14) < 30 buy / > 70 sell)
- Daily rebalance with 5 bps cost per change
- Smooth regime: 3-day rolling mode
- Walk-forward: 2010-2018 IS (HMM training + sanity), 2019-2025 OOS

Verdict: PASS if Sharpe_oos >= 0.7 AND HMM_uplift >= 3pp AND ensemble_vs_5050_sharpe_diff >= 0.25
"""

from __future__ import annotations

import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings("ignore")

# ---------- paths ----------
ROOT = Path("/Users/weeoto/turbo-carnival/research")
CACHE = ROOT / "data" / "cache" / "spy_sectors_vix_daily.parquet"
LOG = ROOT / "logs" / "iter_13_backtest.log"
PNL_CSV = ROOT / "results" / "iter_13_daily_pnl.csv"
SUMMARY_JSON = ROOT / "results" / "iter_13_summary.json"

# ---------- logging ----------
LOG.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG, mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("iter_13")

# ---------- constants ----------
TICKERS = ["SPY", "XLK", "XLF", "XLE", "XLP"]
VIX_TICKER = "^VIX"
SECTORS = ["XLK", "XLF", "XLE", "XLP"]

START = "2010-01-01"
IS_END = "2018-12-31"
OOS_START = "2019-01-01"

COST_BPS = 5.0  # 5 bps per change in position weight
SEED = 42


# ---------- data ----------
def load_data() -> pd.DataFrame:
    """Pull daily Close for SPY, sectors and VIX. Cache as parquet."""
    if CACHE.exists():
        df = pd.read_parquet(CACHE)
        log.info(
            "Loaded cache: %s rows=%d range=%s..%s cols=%s",
            CACHE.name, len(df), df.index[0].date(), df.index[-1].date(),
            list(df.columns),
        )
        return df

    log.info("Downloading %s + %s from yfinance starting %s", TICKERS, VIX_TICKER, START)
    all_syms = TICKERS + [VIX_TICKER]
    raw = yf.download(
        all_syms,
        start=START,
        end=None,
        progress=False,
        auto_adjust=False,
        group_by="ticker",
    )
    if raw is None or raw.empty:
        raise RuntimeError("yfinance returned no data")

    out = {}
    for sym in all_syms:
        try:
            sub = raw[sym]
            close_col = "Adj Close" if "Adj Close" in sub.columns else "Close"
            s = sub[close_col].dropna()
        except (KeyError, TypeError):
            # single-ticker layout
            close_col = "Adj Close" if "Adj Close" in raw.columns else "Close"
            s = raw[close_col].dropna()
        out[sym] = s

    df = pd.DataFrame(out)
    df = df.dropna(how="all")
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    # rename ^VIX -> VIX for downstream convenience
    df = df.rename(columns={VIX_TICKER: "VIX"})

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CACHE)
    log.info("Cached %d rows -> %s (cols=%s)", len(df), CACHE, list(df.columns))
    return df


# ---------- features ----------
def build_hmm_features(spy: pd.Series, vix: pd.Series) -> pd.DataFrame:
    """3 features: SPY log return, VIX level, MA50/MA200 cross indicator."""
    r_spy = np.log(spy / spy.shift(1))
    ma50 = spy.rolling(50, min_periods=50).mean()
    ma200 = spy.rolling(200, min_periods=200).mean()
    ma_cross = pd.Series(np.where(ma50 > ma200, 1.0, -1.0), index=spy.index)
    # ma_cross is undefined while ma200 is NaN -> mask those rows out via dropna below
    ma_cross[ma200.isna() | ma50.isna()] = np.nan

    feats = pd.DataFrame({
        "r_spy": r_spy,
        "vix": vix.astype(float),
        "ma_cross": ma_cross,
    })
    feats = feats.dropna()
    return feats


# ---------- HMM ----------
def fit_hmm_and_label(feats_is: pd.DataFrame) -> tuple[GaussianHMM, dict[int, str]]:
    """
    Fit a 3-state Gaussian HMM with full covariance on IS features.

    After fitting, label states by mean SPY return:
      - highest mean r_spy -> Trending-Up
      - lowest mean r_spy  -> Trending-Down
      - middle             -> Choppy
    """
    X = feats_is.values.astype(float)
    hmm = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=200,
        random_state=SEED,
        tol=1e-4,
    )
    hmm.fit(X)

    means = hmm.means_  # shape (3, 3) -> rows are states, cols are (r_spy, vix, ma_cross)
    # rank by mean r_spy (col 0)
    r_means = means[:, 0]
    order = np.argsort(r_means)  # ascending: [lowest, middle, highest]
    label_map = {
        int(order[0]): "Trending-Down",
        int(order[1]): "Choppy",
        int(order[2]): "Trending-Up",
    }
    log.info("HMM state means (r_spy, vix, ma_cross):")
    for s in range(3):
        log.info(
            "  state %d (%s): r_spy=%.5f vix=%.2f ma_cross=%.2f",
            s, label_map[s], means[s, 0], means[s, 1], means[s, 2],
        )
    log.info("HMM transition matrix:")
    for s in range(3):
        log.info("  %s", " ".join(f"{p:.3f}" for p in hmm.transmat_[s]))
    log.info("HMM converged: %s; score(IS)=%.2f",
             hmm.monitor_.converged, hmm.score(X))
    return hmm, label_map


def predict_regimes(
    hmm: GaussianHMM,
    feats: pd.DataFrame,
    label_map: dict[int, str],
) -> pd.DataFrame:
    """Apply frozen HMM, return DataFrame with raw state, raw label, and 3-day rolling mode label."""
    X = feats.values.astype(float)
    states = hmm.predict(X)
    raw_labels = pd.Series([label_map[int(s)] for s in states], index=feats.index, name="regime_raw")

    # 3-day rolling mode for smoothing
    def _mode_window(window):
        vals, counts = np.unique(window, return_counts=True)
        return vals[np.argmax(counts)]

    # Map labels to ints for rolling operation
    label_to_int = {"Trending-Down": -1, "Choppy": 0, "Trending-Up": 1}
    int_to_label = {v: k for k, v in label_to_int.items()}
    raw_int = raw_labels.map(label_to_int).astype(float)
    smooth_int = raw_int.rolling(3, min_periods=1).apply(_mode_window, raw=True)
    smooth_labels = smooth_int.map(int_to_label).rename("regime_smooth")

    return pd.DataFrame({"regime_raw": raw_labels, "regime_smooth": smooth_labels})


# ---------- substrategies ----------
def tsmom_weights(sector_px: pd.DataFrame) -> pd.DataFrame:
    """
    Trending-Up: TSMOM on each sector ETF.
    Score = sign of 1m (21d) log return per spec from agent brief.
    Long-only, equal-weight TOP 3 sectors with positive sign (cap 3, per spec).
    If fewer than 3 are positive, equal-weight whatever is positive.
    Returns daily weights for each sector.
    """
    log_ret = np.log(sector_px / sector_px.shift(1))
    r_1m = log_ret.rolling(21).sum()  # 1-month cumulative log return
    sig = np.sign(r_1m).fillna(0.0)

    # Equal-weight top 3 with positive signal (ties broken by stronger r_1m)
    w = pd.DataFrame(0.0, index=sector_px.index, columns=sector_px.columns)
    for dt in sector_px.index:
        if dt not in r_1m.index:
            continue
        scores = r_1m.loc[dt]
        if scores.isna().all():
            continue
        # Only positive scores qualify
        pos = scores[scores > 0]
        if len(pos) == 0:
            continue
        top = pos.sort_values(ascending=False).head(3)
        n = len(top)
        for sec in top.index:
            w.loc[dt, sec] = 1.0 / n
    return w


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Standard Wilder RSI(14)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    # Wilder smoothing (EMA with alpha=1/period)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_v = 100 - (100 / (1 + rs))
    return rsi_v


def choppy_spy_weight(spy_px: pd.Series) -> pd.Series:
    """
    Choppy: SPY RSI(14) mean-reversion.
    RSI < 30 -> long SPY (100%)
    RSI > 70 -> flat
    Between: hold prior position (state machine).
    """
    rsi_v = rsi(spy_px, 14)
    pos = pd.Series(0.0, index=spy_px.index)
    cur = 0.0
    for dt, r in rsi_v.items():
        if pd.isna(r):
            pos.loc[dt] = cur
            continue
        if r < 30:
            cur = 1.0
        elif r > 70:
            cur = 0.0
        pos.loc[dt] = cur
    return pos


# ---------- ensemble ----------
def build_ensemble_weights(
    px: pd.DataFrame,
    regime: pd.Series,
) -> pd.DataFrame:
    """
    Per-day weights across [SPY, XLK, XLF, XLE, XLP].
    Trending-Up -> tsmom weights on sectors
    Trending-Down -> 100% cash (all zero)
    Choppy -> RSI mean-rev on SPY (long SPY when triggered, else 0)
    """
    cols = ["SPY", "XLK", "XLF", "XLE", "XLP"]
    sector_px = px[SECTORS]
    spy_px = px["SPY"]

    w_tsmom = tsmom_weights(sector_px)  # cols = SECTORS
    w_choppy_spy = choppy_spy_weight(spy_px)  # SPY only

    w = pd.DataFrame(0.0, index=px.index, columns=cols)

    # Align regime index to price index (regime is computed on feats which may start later
    # because of MA200 warmup). For pre-regime dates, weight = 0 (cash).
    regime_aligned = regime.reindex(px.index)

    up_mask = regime_aligned == "Trending-Up"
    choppy_mask = regime_aligned == "Choppy"
    # down: leave zero

    # Up: copy tsmom sector weights
    for sec in SECTORS:
        w.loc[up_mask, sec] = w_tsmom.loc[up_mask, sec].values

    # Choppy: SPY weight from RSI rule
    w.loc[choppy_mask, "SPY"] = w_choppy_spy.loc[choppy_mask].values

    return w


def compute_pnl(
    px: pd.DataFrame,
    weights: pd.DataFrame,
    cost_bps: float = COST_BPS,
) -> pd.DataFrame:
    """
    Compute daily portfolio P&L.
    weights at day t are taken from end of day t-1 (i.e. shifted by 1 day) to earn day-t return.
    Cost = cost_bps * sum(|delta weight|) per day.
    """
    simple_ret = px.pct_change().fillna(0.0)
    # Use yesterday's weight to earn today's return (no look-ahead).
    w_lag = weights.shift(1).fillna(0.0)
    gross = (w_lag * simple_ret).sum(axis=1)

    # Cost on weight changes (turnover). Applied at the day of rebalance.
    turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turnover * (cost_bps / 1e4)
    net = gross - cost

    out = pd.DataFrame({
        "gross_pnl": gross,
        "cost": cost,
        "net_pnl": net,
        "turnover": turnover,
    })
    return out


# ---------- stats ----------
def perf_stats(net_ret: pd.Series) -> dict:
    net_ret = net_ret.dropna()
    if net_ret.empty:
        return dict(months=0.0, years=0.0, ann_return_pct=0.0, ann_vol_pct=0.0,
                    sharpe=0.0, max_drawdown_pct=0.0)
    n = len(net_ret)
    years = n / 252.0
    months = years * 12.0
    ann_ret = net_ret.mean() * 252
    ann_vol = net_ret.std(ddof=0) * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0.0
    cum = (1.0 + net_ret).cumprod()
    peak = cum.cummax()
    dd = cum / peak - 1.0
    max_dd = float(dd.min())
    return dict(
        months=float(months),
        years=float(years),
        ann_return_pct=float(ann_ret * 100),
        ann_vol_pct=float(ann_vol * 100),
        sharpe=float(sharpe),
        max_drawdown_pct=float(max_dd * 100),
    )


def realized_regime_oracle(spy: pd.Series, vix: pd.Series, window: int = 21) -> pd.Series:
    """
    Forward-looking realized-regime oracle per spec §6.2(i):
    label day t by next 21 trading days' SPY return and VIX path:
      Up if SPY return > +3% and VIX avg < 18
      Down if SPY return < -3% and VIX avg > 25
      Choppy otherwise
    """
    fwd_ret = spy.shift(-window) / spy - 1.0
    fwd_vix = vix.rolling(window, min_periods=window).mean().shift(-window)

    label = pd.Series("Choppy", index=spy.index)
    label[(fwd_ret > 0.03) & (fwd_vix < 18)] = "Trending-Up"
    label[(fwd_ret < -0.03) & (fwd_vix > 25)] = "Trending-Down"
    label[fwd_ret.isna()] = np.nan  # last 21 days have no oracle
    return label


# ---------- baselines ----------
def baseline_5050(px: pd.DataFrame) -> pd.DataFrame:
    """
    Unconditioned 50/50 baseline: 50% TSMOM (top-3 sectors) + 50% SPY RSI mean-rev.
    Same sub-strategies as conditioned ensemble, but no regime gate.
    """
    cols = ["SPY", "XLK", "XLF", "XLE", "XLP"]
    sector_px = px[SECTORS]
    spy_px = px["SPY"]

    w_tsmom = tsmom_weights(sector_px) * 0.5  # half allocation
    w_choppy_spy = choppy_spy_weight(spy_px) * 0.5  # half allocation

    w = pd.DataFrame(0.0, index=px.index, columns=cols)
    for sec in SECTORS:
        w[sec] = w_tsmom[sec].fillna(0.0)
    w["SPY"] = w_choppy_spy.fillna(0.0)
    return w


# ---------- main ----------
def main() -> int:
    log.info("=== iter_13 HMM Regime-Conditioned Ensemble backtest ===")
    log.info("Period: %s -> today; IS: %s..%s; OOS: %s..", START, START, IS_END, OOS_START)

    df = load_data()
    # Sanity
    missing = [c for c in (TICKERS + ["VIX"]) if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns: {missing}")
    log.info("Data rows: %d range=%s..%s",
             len(df), df.index[0].date(), df.index[-1].date())

    # Drop any rows where any sector / SPY / VIX is missing (HMM needs complete observations)
    px_all = df[TICKERS].dropna()
    vix = df["VIX"].reindex(px_all.index).ffill()
    log.info("After dropna alignment: %d rows", len(px_all))

    # ---------- build HMM features ----------
    feats = build_hmm_features(px_all["SPY"], vix)
    log.info("HMM features rows: %d (after MA200 warmup) range=%s..%s",
             len(feats), feats.index[0].date(), feats.index[-1].date())

    # IS / OOS split
    is_feats = feats.loc[:IS_END]
    oos_feats = feats.loc[OOS_START:]
    log.info("IS feature rows: %d, OOS feature rows: %d", len(is_feats), len(oos_feats))

    # ---------- fit HMM on IS, freeze ----------
    hmm, label_map = fit_hmm_and_label(is_feats)

    # Apply to full sample for downstream substrategy execution and reporting
    regimes_full = predict_regimes(hmm, feats, label_map)
    log.info("Regime counts (full sample, raw):")
    log.info("  %s", regimes_full["regime_raw"].value_counts().to_dict())
    log.info("Regime counts (full sample, 3d-mode smoothed):")
    log.info("  %s", regimes_full["regime_smooth"].value_counts().to_dict())

    # ---------- score HMM vs persistence ----------
    oracle = realized_regime_oracle(px_all["SPY"], vix)
    oracle_oos = oracle.loc[OOS_START:].dropna()

    regime_smooth_oos = regimes_full.loc[OOS_START:, "regime_smooth"].dropna()
    common_idx = oracle_oos.index.intersection(regime_smooth_oos.index)
    hmm_pred = regime_smooth_oos.loc[common_idx]
    oracle_true = oracle_oos.loc[common_idx]
    hmm_acc = float((hmm_pred == oracle_true).mean())

    # Persistence baseline: today's regime = yesterday's HMM label
    persistence_pred = regime_smooth_oos.shift(1).loc[common_idx]
    pers_mask = persistence_pred.notna()
    persistence_acc = float((persistence_pred[pers_mask] == oracle_true[pers_mask]).mean())

    hmm_uplift_pp = (hmm_acc - persistence_acc) * 100.0
    log.info(
        "HMM OOS accuracy vs oracle: %.4f; Persistence baseline: %.4f; uplift = %.2fpp",
        hmm_acc, persistence_acc, hmm_uplift_pp,
    )

    # ---------- conditioned ensemble weights & P&L ----------
    w_cond = build_ensemble_weights(px_all, regimes_full["regime_smooth"])
    pnl_cond = compute_pnl(px_all, w_cond, COST_BPS)

    # ---------- 50/50 unconditioned baseline ----------
    w_5050 = baseline_5050(px_all)
    pnl_5050 = compute_pnl(px_all, w_5050, COST_BPS)

    # Combine
    out = px_all.copy()
    out["regime_raw"] = regimes_full["regime_raw"]
    out["regime_smooth"] = regimes_full["regime_smooth"]
    out["w_cond_SPY"] = w_cond["SPY"]
    for sec in SECTORS:
        out[f"w_cond_{sec}"] = w_cond[sec]
    out["gross_pnl_cond"] = pnl_cond["gross_pnl"]
    out["cost_cond"] = pnl_cond["cost"]
    out["net_pnl_cond"] = pnl_cond["net_pnl"]
    out["turnover_cond"] = pnl_cond["turnover"]
    out["net_pnl_5050"] = pnl_5050["net_pnl"]
    out["turnover_5050"] = pnl_5050["turnover"]

    PNL_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(PNL_CSV, index_label="date")
    log.info("Wrote daily P&L: %s (rows=%d)", PNL_CSV, len(out))

    # ---------- stats ----------
    is_mask = (out.index >= START) & (out.index <= IS_END)
    oos_mask = (out.index >= OOS_START)

    is_cond = perf_stats(out.loc[is_mask, "net_pnl_cond"])
    oos_cond = perf_stats(out.loc[oos_mask, "net_pnl_cond"])
    is_5050 = perf_stats(out.loc[is_mask, "net_pnl_5050"])
    oos_5050 = perf_stats(out.loc[oos_mask, "net_pnl_5050"])

    log.info("IS  conditioned : %s", is_cond)
    log.info("OOS conditioned : %s", oos_cond)
    log.info("IS  5050 baseline: %s", is_5050)
    log.info("OOS 5050 baseline: %s", oos_5050)

    ensemble_vs_5050_diff = oos_cond["sharpe"] - oos_5050["sharpe"]
    log.info("Ensemble OOS Sharpe - 5050 OOS Sharpe = %.3f", ensemble_vs_5050_diff)

    # ---------- regime stability sanity ----------
    n_regime_switches_oos = int(
        (out.loc[oos_mask, "regime_smooth"]
         != out.loc[oos_mask, "regime_smooth"].shift(1)).sum()
    )
    n_oos_days = int(oos_mask.sum())
    log.info(
        "OOS regime switches (smoothed): %d over %d days (avg %.1f/month)",
        n_regime_switches_oos, n_oos_days,
        n_regime_switches_oos / max(n_oos_days, 1) * 21.0,
    )

    # ---------- multi-condition verdict ----------
    pass_sharpe = oos_cond["sharpe"] >= 0.70
    pass_uplift = hmm_uplift_pp >= 3.0
    pass_ensemble_vs_5050 = ensemble_vs_5050_diff >= 0.25
    verdict = "PASS" if (pass_sharpe and pass_uplift and pass_ensemble_vs_5050) else "FAIL"

    log.info(
        "Conditions: Sharpe>=0.70: %s (got %.3f) | HMM uplift>=3pp: %s (got %.2fpp) | "
        "Ensemble-5050>=0.25: %s (got %.3f) -> %s",
        pass_sharpe, oos_cond["sharpe"],
        pass_uplift, hmm_uplift_pp,
        pass_ensemble_vs_5050, ensemble_vs_5050_diff,
        verdict,
    )

    summary = {
        "instrument": "SPY + XLK/XLF/XLE/XLP + ^VIX (yfinance daily)",
        "period_full": f"{out.index[0].date()}..{out.index[-1].date()}",
        "is_window": f"{START}..{IS_END}",
        "oos_window": f"{OOS_START}..{out.index[-1].date()}",
        "months_oos": oos_cond["months"],
        "years_oos": oos_cond["years"],
        "ann_return_pct": oos_cond["ann_return_pct"],
        "ann_vol_pct": oos_cond["ann_vol_pct"],
        "sharpe_oos": oos_cond["sharpe"],
        "max_dd_pct": oos_cond["max_drawdown_pct"],
        "ann_return_pct_is": is_cond["ann_return_pct"],
        "sharpe_is": is_cond["sharpe"],
        "max_dd_pct_is": is_cond["max_drawdown_pct"],
        "hmm_oos_accuracy_vs_oracle": hmm_acc,
        "regime_accuracy_persistence_baseline": persistence_acc,
        "hmm_vs_persistence_pp_uplift": hmm_uplift_pp,
        "ensemble_oos_sharpe": oos_cond["sharpe"],
        "baseline_5050_oos_sharpe": oos_5050["sharpe"],
        "ensemble_vs_5050_sharpe_diff": ensemble_vs_5050_diff,
        "ensemble_vs_unconditioned_5050_sharpe_diff": ensemble_vs_5050_diff,
        "ensemble_oos_ann_ret_pct": oos_cond["ann_return_pct"],
        "baseline_5050_oos_ann_ret_pct": oos_5050["ann_return_pct"],
        "ensemble_oos_max_dd_pct": oos_cond["max_drawdown_pct"],
        "baseline_5050_oos_max_dd_pct": oos_5050["max_drawdown_pct"],
        "n_regime_switches_oos": n_regime_switches_oos,
        "n_oos_days": n_oos_days,
        "cost_bps_per_change": COST_BPS,
        "pass_threshold_sharpe_oos": 0.70,
        "pass_threshold_hmm_uplift_pp": 3.0,
        "pass_threshold_ensemble_vs_5050": 0.25,
        "pass_sharpe": bool(pass_sharpe),
        "pass_uplift": bool(pass_uplift),
        "pass_ensemble_vs_5050": bool(pass_ensemble_vs_5050),
        "verdict": verdict,
    }

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))
    log.info("Wrote summary: %s", SUMMARY_JSON)
    log.info("VERDICT: %s", verdict)

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
