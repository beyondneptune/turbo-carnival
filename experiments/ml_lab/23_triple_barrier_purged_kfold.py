"""Triple-barrier labeling + purged k-fold CV for composite C3 retrain.

Replaces the naive fixed-bps stop/TP labeling with the de Prado triple-barrier
method, and uses purged k-fold cross-validation to avoid time-series leakage.

Methodology (Marcos López de Prado, "Advances in Financial ML"):

1. TRIPLE-BARRIER LABELING
   For each entry candidate at time t, define three barriers:
     - Upper: entry_price × (1 + take_profit_bps/10000)
     - Lower: entry_price × (1 - stop_loss_bps/10000)
     - Time:  t + max_hold_min minutes
   Label by which barrier hits first:
     +1 if upper hit first   (winner)
      0 if time barrier hit  (neither — wash)
     -1 if lower hit first   (loser)

2. PURGED K-FOLD CV
   For 5-fold CV on time series:
     - Train on folds {0,1,2,3,4} \\ {i}
     - Test on fold i
     - PURGE: remove training samples whose [t, t+max_hold_min] overlaps the
       test fold's time range (prevents label leakage)
     - EMBARGO: remove training samples within 60 min after the test fold ends
       (further reduces leakage from serial correlation)

3. C3 RETRAIN
   Retrain the 4 C3 sub-models (LOF, OCSVM, EllipticEnvelope, RBF+LR) on the
   purged training data, predict on the test fold. Aggregate metrics across folds.

Output:
  experiments/ml_lab/runs/triple_barrier_purged_cv/summary.json
  experiments/ml_lab/runs/triple_barrier_purged_cv/per_fold.parquet

Requires minute bars in data/cache/minute_bars/<symbol>.parquet (from
scripts/polygon_s3_historical_pull.py).
"""
from __future__ import annotations

import argparse
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
DATA = REPO / "data" / "cache" / "minute_bars"
OUT = REPO / "experiments" / "ml_lab" / "runs" / "triple_barrier_purged_cv"
OUT.mkdir(parents=True, exist_ok=True)

UNIVERSE = ("SPY", "QQQ", "NVDA", "AAPL", "META")  # TSLA/AMD blacklisted
TAKE_PROFIT_BPS = 100.0
STOP_LOSS_BPS = 35.0
MAX_HOLD_MIN = 20
N_FOLDS = 5
EMBARGO_MIN = 60


def triple_barrier_label(bars: pd.DataFrame, entry_idx: int) -> tuple[int, float, int]:
    """Label one entry candidate using triple-barrier method.

    Returns (label, exit_price, hold_min).
    label: +1 (TP), 0 (time), -1 (SL)
    """
    if entry_idx >= len(bars) - 1:
        return 0, bars["close"].iloc[entry_idx], 0
    entry_price = bars["close"].iloc[entry_idx]
    upper = entry_price * (1 + TAKE_PROFIT_BPS / 10000)
    lower = entry_price * (1 - STOP_LOSS_BPS / 10000)
    horizon = min(entry_idx + MAX_HOLD_MIN, len(bars) - 1)
    for j in range(entry_idx + 1, horizon + 1):
        h = bars["high"].iloc[j]
        l = bars["low"].iloc[j]
        if h >= upper:
            return +1, upper, j - entry_idx
        if l <= lower:
            return -1, lower, j - entry_idx
    return 0, bars["close"].iloc[horizon], horizon - entry_idx


def compute_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Compute the 10 features the C3 gate uses, per bar."""
    feats = pd.DataFrame(index=bars.index)
    close = bars["close"]
    feats["ret1"] = close.pct_change(1)
    feats["ret3"] = close.pct_change(3)
    feats["ret10"] = close.pct_change(10)
    feats["range_pct"] = (bars["high"].rolling(20).max() - bars["low"].rolling(20).min()) / close
    feats["volume_z"] = (bars["volume"] - bars["volume"].rolling(20).mean()) / bars["volume"].rolling(20).std()
    rets = close.pct_change()
    feats["ret1_z"] = (rets - rets.rolling(20).mean()) / rets.rolling(20).std()
    feats["vwap_dist"] = (close - (close * bars["volume"]).rolling(20).sum() / bars["volume"].rolling(20).sum()) / close
    # Intraday time encoding (assumes index is datetime)
    mins = pd.Series(bars.index.hour * 60 + bars.index.minute - 570, index=bars.index)
    feats["tod_sin"] = np.sin(2 * np.pi * mins / 390)
    feats["tod_cos"] = np.cos(2 * np.pi * mins / 390)
    feats["vov"] = rets.rolling(60).std() / rets.rolling(20).std()  # vol-of-vol proxy
    return feats.fillna(0)


def purged_split(n_samples: int, fold: int, n_folds: int, hold_min: int, embargo_min: int):
    """Return (train_idx, test_idx) with purging + embargo applied.

    Test fold = block of n/n_folds contiguous samples starting at fold*n/n_folds.
    Training samples that overlap [test_start, test_end + hold_min] are removed.
    """
    fold_size = n_samples // n_folds
    test_start = fold * fold_size
    test_end = test_start + fold_size if fold < n_folds - 1 else n_samples
    test_idx = np.arange(test_start, test_end)

    purge_buffer = hold_min  # entries within hold_min of test fold leak labels
    purge_start = max(0, test_start - purge_buffer)
    purge_end = min(n_samples, test_end + embargo_min)

    train_idx = np.concatenate([
        np.arange(0, purge_start),
        np.arange(purge_end, n_samples),
    ])
    return train_idx, test_idx


def fit_c3_subset(X_train: np.ndarray, y_train: np.ndarray):
    """Fit the 4 C3 sub-models on a training subset."""
    from sklearn.covariance import EllipticEnvelope
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.svm import OneClassSVM
    from sklearn.kernel_approximation import RBFSampler
    from sklearn.pipeline import make_pipeline

    # 016 LOF
    lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
    lof.fit(X_train)

    # 017 OCSVM
    ocsvm = OneClassSVM(gamma="auto", nu=0.05)
    ocsvm.fit(X_train)

    # 018 EllipticEnvelope
    env = EllipticEnvelope(contamination=0.05, random_state=42)
    env.fit(X_train)

    # 102 RBF + LR
    rbf_lr = make_pipeline(RBFSampler(n_components=100, random_state=42),
                              LogisticRegression(max_iter=300))
    rbf_lr.fit(X_train, (y_train == 1).astype(int))  # predict "winner"

    return lof, ocsvm, env, rbf_lr


def predict_c3_blocks(models, X_test: np.ndarray) -> np.ndarray:
    """Return boolean array: True where C3 would BLOCK entry."""
    lof, ocsvm, env, rbf_lr = models
    block = np.zeros(len(X_test), dtype=bool)
    block |= (lof.predict(X_test) == -1)
    block |= (ocsvm.predict(X_test) == -1)
    block |= (env.predict(X_test) == -1)
    block |= (rbf_lr.predict_proba(X_test)[:, 1] < 0.5)
    return block


def run_one_symbol(sym: str, n_folds: int) -> dict:
    """Full triple-barrier + purged k-fold for one symbol."""
    bars_path = DATA / f"{sym}.parquet"
    if not bars_path.exists():
        print(f"  {sym}: no minute bars at {bars_path}")
        return {"symbol": sym, "error": "no_data"}
    bars = pd.read_parquet(bars_path)
    if "timestamp" in bars.columns:
        bars["timestamp"] = pd.to_datetime(bars["timestamp"])
        bars = bars.set_index("timestamp")
    bars = bars.sort_index()
    print(f"  {sym}: {len(bars):,} bars")

    feats = compute_features(bars)
    labels = pd.Series([triple_barrier_label(bars, i)[0] for i in range(len(bars))], index=bars.index)
    valid = feats.notna().all(axis=1) & (labels != 0)
    X = feats[valid].values
    y = labels[valid].values
    if len(X) < 1000:
        return {"symbol": sym, "error": f"too_few_samples_{len(X)}"}

    fold_results = []
    for fold in range(n_folds):
        train_idx, test_idx = purged_split(len(X), fold, n_folds, MAX_HOLD_MIN, EMBARGO_MIN)
        models = fit_c3_subset(X[train_idx], y[train_idx])
        blocked = predict_c3_blocks(models, X[test_idx])
        y_test = y[test_idx]
        # Metrics: how does blocking affect realized P&L?
        kept = ~blocked
        n_kept = int(kept.sum())
        if n_kept == 0:
            continue
        winrate_kept = float((y_test[kept] == 1).mean())
        winrate_blocked = float((y_test[blocked] == 1).mean()) if blocked.any() else 0
        fold_results.append({
            "symbol": sym, "fold": fold, "n_train": len(train_idx), "n_test": len(test_idx),
            "n_kept": n_kept, "n_blocked": int(blocked.sum()),
            "winrate_kept": winrate_kept, "winrate_blocked": winrate_blocked,
            "lift": winrate_kept - (y_test == 1).mean(),
        })
        print(f"    fold {fold}: train {len(train_idx)} test {len(test_idx)} "
              f"keep {n_kept} ({winrate_kept:.1%} win)  block {blocked.sum()} ({winrate_blocked:.1%} win)")

    return {"symbol": sym, "folds": fold_results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbols", nargs="*", default=list(UNIVERSE))
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    args = parser.parse_args()

    print("=" * 70)
    print("Triple-barrier + purged k-fold CV for composite C3")
    print("=" * 70)
    print(f"Symbols: {args.symbols}  Folds: {args.folds}  Hold: {MAX_HOLD_MIN}m  Embargo: {EMBARGO_MIN}m")
    print()

    all_results = []
    for sym in args.symbols:
        result = run_one_symbol(sym, args.folds)
        all_results.append(result)

    # Aggregate
    all_folds = []
    for r in all_results:
        if "folds" in r:
            all_folds.extend(r["folds"])
    if all_folds:
        df = pd.DataFrame(all_folds)
        df.to_parquet(OUT / "per_fold.parquet", index=False)
        summary = {
            "mean_winrate_kept": float(df["winrate_kept"].mean()),
            "mean_winrate_blocked": float(df["winrate_blocked"].mean()),
            "mean_lift": float(df["lift"].mean()),
            "n_folds_total": len(df),
            "params": {
                "max_hold_min": MAX_HOLD_MIN, "embargo_min": EMBARGO_MIN,
                "tp_bps": TAKE_PROFIT_BPS, "sl_bps": STOP_LOSS_BPS,
            },
        }
        (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
        print()
        print("=== Summary ===")
        print(f"  Mean win-rate of kept entries:    {summary['mean_winrate_kept']:.1%}")
        print(f"  Mean win-rate of blocked entries: {summary['mean_winrate_blocked']:.1%}")
        print(f"  Mean lift from C3 gate:           {summary['mean_lift']:+.1%}")
        print(f"  Output: {OUT}")
    else:
        print("No folds completed — need minute bars in data/cache/minute_bars/")
        print("Run scripts/polygon_s3_historical_pull.py first.")


if __name__ == "__main__":
    main()
