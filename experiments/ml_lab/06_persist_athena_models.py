"""Train + persist the two ATHENA v1 models so the live host can load them.

Writes:
  experiments/ml_lab/runs/athena/logreg_regime.joblib
  experiments/ml_lab/runs/athena/ae_anomaly.joblib
  experiments/ml_lab/runs/athena/ae_scaler.joblib
  experiments/ml_lab/runs/athena/manifest.json (paths + percentile thresholds)

Re-run nightly. The manifest is the authoritative source — never load
joblibs directly from the live host, always go through scripts/athena_ml_gates.py.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

REPO = Path("/Users/weeoto/turbo-carnival")
BARS = REPO / "experiments" / "runs" / "valk_hft_demo_20260511T140548Z" / "minute_bars.parquet"
OUT = REPO / "experiments" / "ml_lab" / "runs" / "athena"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW = 30
FORWARD_BARS = 5
THRESHOLD_BPS = 30.0
AE_FEATS = ["ret1", "ret3", "range_pct", "volume_z", "ret1_z", "vwap_dist"]
LR_FEATS = ["ret1", "ret3", "ret10", "range_pct", "volume_z", "ret1_z",
             "vwap_dist", "tod_sin", "tod_cos", "vov"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["symbol", "t"]).reset_index(drop=True)
    df["t_ny"] = pd.to_datetime(df["t_ny"], errors="coerce")
    mins = ((df["t_ny"].dt.hour - 9) * 60 + df["t_ny"].dt.minute - 30).clip(0, 390)
    df["tod_sin"] = np.sin(2 * np.pi * mins / 390)
    df["tod_cos"] = np.cos(2 * np.pi * mins / 390)

    def vov(g):
        s = g["ret1"].rolling(5).std()
        m = g["ret1"].abs().rolling(20).median()
        return (s / m).fillna(0).clip(0, 50)
    df["vov"] = df.groupby("symbol", group_keys=False).apply(vov, include_groups=False)
    df["fwd_max_abs_ret"] = (
        df.groupby("symbol")["close"].pct_change(FORWARD_BARS).shift(-FORWARD_BARS).abs()
    )
    df["y"] = (df["fwd_max_abs_ret"] * 10000.0 > THRESHOLD_BPS).astype(int)
    return df


def train_logreg(df: pd.DataFrame) -> Pipeline:
    sub = df.dropna(subset=LR_FEATS + ["y"]).reset_index(drop=True)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced")),
    ])
    pipe.fit(sub[LR_FEATS].values, sub["y"].values)
    return pipe


def build_windows(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    df = df.copy().sort_values(["symbol", "t"]).reset_index(drop=True)
    df = df.dropna(subset=AE_FEATS + ["y"]).reset_index(drop=True)
    wins, labels = [], []
    for sym, g in df.groupby("symbol"):
        arr = g[AE_FEATS].values
        ys = g["y"].values
        for i in range(WINDOW, len(arr)):
            wins.append(arr[i - WINDOW:i].flatten())
            labels.append(ys[i])
    return np.array(wins), np.array(labels)


def train_ae(X: np.ndarray, y: np.ndarray) -> tuple[MLPRegressor, StandardScaler, np.ndarray]:
    """Train AE only on non-anomalous windows, return model+scaler+recon errors."""
    fit_mask = y == 0
    X_fit = X[fit_mask]
    scaler = StandardScaler()
    X_fit_s = scaler.fit_transform(X_fit)
    bottleneck = max(4, X.shape[1] // 4)
    ae = MLPRegressor(
        hidden_layer_sizes=(X.shape[1] // 2, bottleneck, X.shape[1] // 2),
        activation="relu", solver="adam", max_iter=200,
        learning_rate_init=0.001, random_state=42, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=10,
    )
    ae.fit(X_fit_s, X_fit_s)
    # Compute recon errors on the FULL training set (all bars, not just normal)
    # so the p95/p99 thresholds reflect what live data will look like
    X_all_s = scaler.transform(X)
    recon = ae.predict(X_all_s)
    errs = ((recon - X_all_s) ** 2).mean(axis=1)
    return ae, scaler, errs


def main() -> int:
    print(f"loading {BARS}")
    df = pd.read_parquet(BARS)
    df = make_features(df)
    print(f"  {len(df):,} bars, {df['symbol'].nunique()} symbols")

    print("\ntraining LogReg regime classifier…")
    lr = train_logreg(df)
    lr_path = OUT / "logreg_regime.joblib"
    joblib.dump(lr, lr_path)
    print(f"  → {lr_path}")

    print("\nbuilding windows + training AE…")
    X, y = build_windows(df)
    ae, scaler, recon = train_ae(X, y)
    p95 = float(np.quantile(recon, 0.95))
    p99 = float(np.quantile(recon, 0.99))
    print(f"  windows: {len(X):,}, recon errs p95={p95:.4f} p99={p99:.4f}")
    ae_path = OUT / "ae_anomaly.joblib"
    sc_path = OUT / "ae_scaler.joblib"
    joblib.dump(ae, ae_path)
    joblib.dump(scaler, sc_path)
    print(f"  → {ae_path}")
    print(f"  → {sc_path}")

    manifest = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "logreg_file": "logreg_regime.joblib",
        "logreg_features": LR_FEATS,
        "ae_file": "ae_anomaly.joblib",
        "ae_scaler_file": "ae_scaler.joblib",
        "ae_features": AE_FEATS,
        "ae_window": WINDOW,
        "ae_p95": p95,
        "ae_p99": p99,
        "training_rows_logreg": int(df.dropna(subset=LR_FEATS + ["y"]).shape[0]),
        "training_rows_ae_total": int(len(X)),
        "training_rows_ae_normal_only": int((y == 0).sum()),
        "forward_bars": FORWARD_BARS,
        "threshold_bps": THRESHOLD_BPS,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"\n  → {OUT / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
