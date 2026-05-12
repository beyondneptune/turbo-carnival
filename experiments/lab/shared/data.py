"""Shared data loader — features + labels from the 21-day minute bar cache.

All experiments in `experiments/lab/` should import from here so they're
guaranteed to use the SAME train/test split, SAME features, SAME target.
That's how the leaderboard is apples-to-apples.

Target: P(next-5-min |ret| > 30 bps) — same as ML lab regime classifier.
Train: days 1-14 of cache. Test: days 15-21.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
BARS = REPO / "experiments" / "runs" / "valk_hft_demo_20260511T140548Z" / "minute_bars.parquet"

FORWARD_BARS = 5
THRESHOLD_BPS = 30.0

CLASSIFIER_FEATURES = [
    "ret1", "ret3", "ret10", "range_pct", "volume_z", "ret1_z",
    "vwap_dist", "tod_sin", "tod_cos", "vov",
]
AE_FEATURES = ["ret1", "ret3", "range_pct", "volume_z", "ret1_z", "vwap_dist"]
WINDOW = 30


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
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
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["date"] = df["t"].dt.date
    return df


def load_classifier_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return X_train, y_train, X_test, y_test for the regime classifier task."""
    df = _enrich(pd.read_parquet(BARS))
    df = df.dropna(subset=CLASSIFIER_FEATURES + ["y"]).reset_index(drop=True)
    dates = sorted(df["date"].unique())
    n_train = int(len(dates) * 0.7)
    train_dates = set(dates[:n_train])
    train_mask = df["date"].isin(train_dates)
    test_mask = ~train_mask
    X = df[CLASSIFIER_FEATURES].values.astype(np.float32)
    y = df["y"].values.astype(np.int8)
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


def load_window_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/test windows for autoencoder / sequence models."""
    df = _enrich(pd.read_parquet(BARS))
    df = df.dropna(subset=AE_FEATURES + ["y"]).reset_index(drop=True)
    wins, labels, dates_arr = [], [], []
    for sym, g in df.groupby("symbol"):
        arr = g[AE_FEATURES].values
        ys = g["y"].values
        ds = g["date"].values
        for i in range(WINDOW, len(arr)):
            wins.append(arr[i - WINDOW:i].flatten())
            labels.append(ys[i])
            dates_arr.append(ds[i])
    X = np.array(wins, dtype=np.float32)
    y = np.array(labels, dtype=np.int8)
    dates = np.array(dates_arr)
    unique_dates = sorted(set(dates.tolist()))
    n_train = int(len(unique_dates) * 0.7)
    train_dates = set(unique_dates[:n_train])
    train_mask = np.array([d in train_dates for d in dates])
    test_mask = ~train_mask
    return X[train_mask], y[train_mask], X[test_mask], y[test_mask]


def load_raw_bars() -> pd.DataFrame:
    """For experiments that need full bar data (pairs, bollinger, etc)."""
    return _enrich(pd.read_parquet(BARS))
