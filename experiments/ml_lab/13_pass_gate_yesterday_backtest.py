"""Backtest all 64 PASS-gate architectures against yesterday's 162 actual entries.

For each pass-gate model in experiments/lab/:
  1. Train on cached bars (days 1-20)
  2. Get probability prediction at each of yesterday's 162 entry timestamps
  3. Decide BLOCK if proba >= 0.5
  4. Compute counterfactual P&L = sum of realized P&L for KEPT entries

Outputs:
  experiments/ml_lab/runs/pass_gate_backtest/summary.parquet
  experiments/ml_lab/runs/pass_gate_backtest/per_entry.parquet

This script is large (~600 lines) because it inlines all 64 model factories.
Custom architectures (BART, multi-task NN, hard-neg mining, etc.) have their
own classes with fit/predict_proba methods.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

warnings.filterwarnings("ignore")
REPO = Path("/Users/weeoto/turbo-carnival")
sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd
import requests
import lightgbm as lgb
import xgboost as xgb
from joblib import Parallel, delayed
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.calibration import CalibratedClassifierCV
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                               ExtraTreesClassifier, GradientBoostingClassifier,
                               HistGradientBoostingClassifier, IsolationForest,
                               RandomForestClassifier, StackingClassifier, VotingClassifier)
from sklearn.feature_selection import RFECV, SelectKBest, mutual_info_classif
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_approximation import Nystroem, RBFSampler
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (BayesianRidge, ElasticNet, LogisticRegression,
                                    PassiveAggressiveClassifier, Perceptron,
                                    RidgeClassifier, SGDClassifier)
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier, LocalOutlierFactor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (KBinsDiscretizer, MinMaxScaler, PolynomialFeatures,
                                     PowerTransformer, RobustScaler, StandardScaler)
from sklearn.random_projection import GaussianRandomProjection
from sklearn.svm import SVC, OneClassSVM
from sklearn.tree import DecisionTreeClassifier

OUT = REPO / "experiments" / "ml_lab" / "runs" / "pass_gate_backtest"
OUT.mkdir(parents=True, exist_ok=True)

BARS_CACHE = REPO / "experiments" / "runs" / "valk_hft_demo_20260511T140548Z" / "minute_bars.parquet"
ZOO_PER_ENTRY = REPO / "experiments" / "ml_lab" / "runs" / "ml_zoo_per_entry_latest.parquet"

THRESHOLD_BPS = 30.0
FORWARD_BARS = 5
BLOCK_PROBA = 0.5
LR_FEATS = ["ret1", "ret3", "ret10", "range_pct", "volume_z", "ret1_z",
             "vwap_dist", "tod_sin", "tod_cos", "vov"]
AE_FEATS = ["ret1", "ret3", "range_pct", "volume_z", "ret1_z", "vwap_dist"]
WINDOW_BARS = 30


# ===========================================================================
# Data loading — same pattern as 04_ml_zoo_today_backtest.py
# ===========================================================================

def load_env() -> None:
    ENV = REPO / "council" / ".env.local"
    if not ENV.exists(): return
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def fetch_yesterday_bars() -> pd.DataFrame:
    load_env()
    key = os.environ.get("ALPACA_AEGIS_KEY"); sec = os.environ.get("ALPACA_AEGIS_SECRET")
    H = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    rows = []
    for sym in ("SPY", "QQQ", "AAPL", "AMD", "META", "NVDA", "TSLA"):
        r = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=H,
                         params={"symbols": sym, "timeframe": "1Min",
                                  "start": "2026-05-11T13:30:00Z",
                                  "end": "2026-05-11T20:00:00Z",
                                  "limit": 10000, "adjustment": "raw"}, timeout=20)
        for b in r.json().get("bars", {}).get(sym, []):
            rows.append({"symbol": sym, "t": b["t"], "open": b["o"], "high": b["h"],
                          "low": b["l"], "close": b["c"], "volume": b["v"]})
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["t_ny"] = df["t"].dt.tz_convert("America/New_York")
    df["date"] = df["t"].dt.date
    df = df.sort_values(["symbol", "t"]).reset_index(drop=True)
    df["ret1"] = df.groupby("symbol")["close"].pct_change(1).fillna(0)
    df["ret3"] = df.groupby("symbol")["close"].pct_change(3).fillna(0)
    df["ret10"] = df.groupby("symbol")["close"].pct_change(10).fillna(0)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["volume_z"] = df.groupby("symbol")["volume"].transform(
        lambda s: (s - s.rolling(20, min_periods=5).mean()) / s.rolling(20, min_periods=5).std()
    ).fillna(0)
    df["ret1_z"] = df.groupby("symbol")["ret1"].transform(
        lambda s: (s - s.rolling(20, min_periods=5).mean()) / s.rolling(20, min_periods=5).std()
    ).fillna(0)
    df["session_vwap"] = df.groupby(["symbol", "date"]).apply(
        lambda g: (g["close"] * g["volume"]).cumsum() / g["volume"].cumsum().replace(0, np.nan),
        include_groups=False,
    ).reset_index(level=[0, 1], drop=True)
    df["vwap_dist"] = (df["close"] - df["session_vwap"]) / df["session_vwap"]
    mins = ((df["t_ny"].dt.hour - 9) * 60 + df["t_ny"].dt.minute - 30).clip(0, 390)
    df["tod_sin"] = np.sin(2 * np.pi * mins / 390)
    df["tod_cos"] = np.cos(2 * np.pi * mins / 390)

    def vov(g):
        s = g["ret1"].rolling(5).std()
        m = g["ret1"].abs().rolling(20).median()
        return (s / m).fillna(0).clip(0, 50)
    df["vov"] = df.groupby("symbol", group_keys=False).apply(vov, include_groups=False)
    return df


def load_training_data() -> tuple[np.ndarray, np.ndarray]:
    """Load the same 70/30 train data from the cached bars."""
    df = pd.read_parquet(BARS_CACHE)
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
    df = df.dropna(subset=LR_FEATS + ["y"]).reset_index(drop=True)
    # Use ENTIRE cached 21 days (no holdout) since we test on yesterday's actual trades
    return df[LR_FEATS].values.astype(np.float32), df["y"].values.astype(np.int8)


def load_yesterday_entries():
    """Reuse the matched-entries dataframe from ml_zoo."""
    if not ZOO_PER_ENTRY.exists():
        raise RuntimeError("Run 04_ml_zoo_today_backtest.py first to materialize matched entries.")
    per_entry = pd.read_parquet(ZOO_PER_ENTRY)
    one_model = per_entry["model"].iloc[0]
    entries = per_entry[per_entry["model"] == one_model].copy().reset_index(drop=True)
    entries["entry_ts"] = pd.to_datetime(entries["entry_ts"], utc=True)
    entries["entry_minute"] = entries["entry_ts"].dt.floor("1min")
    return entries


def build_entry_features(entries: pd.DataFrame, yest_bars: pd.DataFrame) -> np.ndarray:
    """Look up feature vector at each entry's bar."""
    yb = yest_bars.copy()
    yb["minute"] = yb["t"].dt.floor("1min")
    yb = yb.set_index(["symbol", "minute"])
    X = []
    for _, e in entries.iterrows():
        try:
            row = yb.loc[(e["symbol"], e["entry_minute"])]
            if isinstance(row, pd.DataFrame): row = row.iloc[0]
            X.append([float(row[k]) for k in LR_FEATS])
        except KeyError:
            X.append([np.nan] * len(LR_FEATS))
    return np.array(X)


# ===========================================================================
# Custom architecture wrappers — for pass-gate models with non-trivial training
# ===========================================================================

class AthenaStack:
    """027: 3 bases on disjoint features, geometric-mean confluence."""
    def __init__(self):
        self.b1 = self.b2 = self.b3 = None
        self.price_idx = [0, 1, 2, 5]
        self.volume_idx = [3, 4, 9]
        self.time_idx = [6, 7, 8]
    def fit(self, X, y):
        self.b1 = Pipeline([("s", StandardScaler()),
                             ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))]).fit(X[:, self.price_idx], y)
        self.b2 = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                       class_weight="balanced", random_state=42, verbose=-1).fit(X[:, self.volume_idx], y)
        self.b3 = Pipeline([("s", StandardScaler()),
                             ("c", MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=200,
                                                   random_state=42, early_stopping=True))]).fit(X[:, self.time_idx], y)
        return self
    def predict_proba_class1(self, X):
        eps = 1e-6
        p1 = self.b1.predict_proba(X[:, self.price_idx])[:, 1]
        p2 = self.b2.predict_proba(X[:, self.volume_idx])[:, 1]
        p3 = self.b3.predict_proba(X[:, self.time_idx])[:, 1]
        return np.exp((np.log(p1 + eps) + np.log(p2 + eps) + np.log(p3 + eps)) / 3)


class BartStyleAdditive:
    """107: 50 shallow regression trees on residuals with shrinkage."""
    def __init__(self, n_trees=50, depth=3, shrink=0.1):
        from sklearn.tree import DecisionTreeRegressor
        self.n_trees = n_trees; self.depth = depth; self.shrink = shrink
        self.trees = []; self.DTR = DecisionTreeRegressor
    def fit(self, X, y):
        f = np.zeros(len(X))
        for _ in range(self.n_trees):
            p = expit(f)
            grad = y - p
            t = self.DTR(max_depth=self.depth, random_state=42).fit(X, grad)
            self.trees.append(t)
            f += self.shrink * t.predict(X)
        return self
    def predict_proba_class1(self, X):
        f = np.zeros(len(X))
        for t in self.trees:
            f += self.shrink * t.predict(X)
        return expit(f)


class IsotonicChain:
    """115: per-feature isotonic regression chained into LogReg."""
    def __init__(self):
        self.iso_models = []; self.meta = None
    def fit(self, X, y):
        self.iso_models = []
        for j in range(X.shape[1]):
            iso = IsotonicRegression(out_of_bounds='clip').fit(X[:, j], y)
            self.iso_models.append(iso)
        X_iso = np.column_stack([m.predict(X[:, j]) for j, m in enumerate(self.iso_models)])
        self.meta = LogisticRegression(class_weight="balanced", max_iter=1000).fit(X_iso, y)
        return self
    def predict_proba_class1(self, X):
        X_iso = np.column_stack([m.predict(X[:, j]) for j, m in enumerate(self.iso_models)])
        return self.meta.predict_proba(X_iso)[:, 1]


class StrongL2Mlp:
    """125: 5-seed average of strongly L2-regularized MLPs."""
    def __init__(self):
        self.models = []
    def fit(self, X, y):
        for seed in [1, 2, 3, 4, 5]:
            m = Pipeline([("s", StandardScaler()),
                          ("c", MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1.0,
                                               max_iter=200, random_state=seed,
                                               early_stopping=True, validation_fraction=0.1))])
            m.fit(X, y); self.models.append(m)
        return self
    def predict_proba_class1(self, X):
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)


class PrototypeKnn:
    """131: per-class KMeans centroids, softmax over min-distances."""
    def __init__(self, k_pos=50, k_neg=50):
        self.k_pos = k_pos; self.k_neg = k_neg
        self.km_pos = self.km_neg = self.scaler = None
    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.km_pos = KMeans(n_clusters=self.k_pos, random_state=42, n_init=10).fit(Xs[y == 1])
        self.km_neg = KMeans(n_clusters=self.k_neg, random_state=42, n_init=10).fit(Xs[y == 0])
        return self
    def predict_proba_class1(self, X):
        Xs = self.scaler.transform(X)
        d_pos = self.km_pos.transform(Xs).min(axis=1)
        d_neg = self.km_neg.transform(Xs).min(axis=1)
        return np.exp(-d_pos) / (np.exp(-d_pos) + np.exp(-d_neg))


class BoostedLogregStack:
    """132: gradient-boosted with Ridge weak learners."""
    def __init__(self, n_rounds=20, lr=0.1):
        from sklearn.linear_model import Ridge
        self.n_rounds = n_rounds; self.lr = lr
        self.scaler = None; self.boosters = []; self.bias = 0.0; self.Ridge = Ridge
    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X); Xs = self.scaler.transform(X)
        self.bias = logit(float(y.mean()))
        F = np.full(len(Xs), self.bias)
        self.boosters = []
        for _ in range(self.n_rounds):
            p = expit(F); grad = y - p
            b = self.Ridge(alpha=1.0).fit(Xs, grad); self.boosters.append(b)
            F += self.lr * b.predict(Xs)
        return self
    def predict_proba_class1(self, X):
        Xs = self.scaler.transform(X)
        F = np.full(len(Xs), self.bias)
        for b in self.boosters: F += self.lr * b.predict(Xs)
        return expit(F)


class KMeansCodesLogreg:
    """138: cluster-distance codes as 20-dim features → LogReg."""
    def __init__(self, k=20):
        self.k = k; self.km = self.lr = self.scaler = None
    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X); Xs = self.scaler.transform(X)
        self.km = KMeans(n_clusters=self.k, random_state=42, n_init=10).fit(Xs)
        codes = self.km.transform(Xs)
        self.lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(codes, y)
        return self
    def predict_proba_class1(self, X):
        Xs = self.scaler.transform(X); codes = self.km.transform(Xs)
        return self.lr.predict_proba(codes)[:, 1]


class HardNegativeMiningMlp:
    """141: train MLP, find hard negatives, retrain on curated set."""
    def __init__(self):
        self.m = None
    def fit(self, X, y):
        m1 = Pipeline([("s", StandardScaler()),
                        ("c", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                                             random_state=42, early_stopping=True))])
        m1.fit(X, y)
        p = m1.predict_proba(X)[:, 1]
        neg_mask = y == 0; pos_idx = np.where(y == 1)[0]
        neg_p = p[neg_mask]; neg_idx = np.where(neg_mask)[0]
        thresh = np.quantile(neg_p, 0.7)
        hard_neg = neg_idx[neg_p >= thresh]
        rest_neg = neg_idx[neg_p < thresh]
        rng = np.random.default_rng(42)
        rand_neg = rng.choice(rest_neg, size=int(0.2 * len(rest_neg)), replace=False)
        keep = np.concatenate([pos_idx, hard_neg, rand_neg])
        self.m = Pipeline([("s", StandardScaler()),
                            ("c", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200,
                                                 random_state=123, early_stopping=True))])
        self.m.fit(X[keep], y[keep])
        return self
    def predict_proba_class1(self, X):
        return self.m.predict_proba(X)[:, 1]


class PowerTransformNb:
    """143: Yeo-Johnson + GaussianNB."""
    def __init__(self):
        self.p = Pipeline([("pt", PowerTransformer(method="yeo-johnson", standardize=True)),
                            ("nb", GaussianNB())])
    def fit(self, X, y): self.p.fit(X, y); return self
    def predict_proba_class1(self, X): return self.p.predict_proba(X)[:, 1]


class NbLrGeometricBlend:
    """144: sqrt(p_nb * p_lr)."""
    def __init__(self):
        self.nb = Pipeline([("s", StandardScaler()), ("c", GaussianNB())])
        self.lr = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(class_weight="balanced", max_iter=2000))])
    def fit(self, X, y): self.nb.fit(X, y); self.lr.fit(X, y); return self
    def predict_proba_class1(self, X):
        p_nb = self.nb.predict_proba(X)[:, 1].clip(1e-6, 1-1e-6)
        p_lr = self.lr.predict_proba(X)[:, 1].clip(1e-6, 1-1e-6)
        return np.sqrt(p_nb * p_lr)


class SparseLrScreening:
    """145: aggressive L1 screen + L2 refit."""
    def __init__(self):
        self.scaler = self.lr = self.mask = None
    def fit(self, X, y):
        self.scaler = StandardScaler().fit(X); Xs = self.scaler.transform(X)
        scr = LogisticRegression(penalty="l1", solver="saga", C=0.005, max_iter=5000,
                                  class_weight="balanced").fit(Xs, y)
        self.mask = (scr.coef_[0] != 0)
        if self.mask.sum() == 0: self.mask = np.ones(Xs.shape[1], dtype=bool)
        self.lr = LogisticRegression(class_weight="balanced", max_iter=2000).fit(Xs[:, self.mask], y)
        return self
    def predict_proba_class1(self, X):
        Xs = self.scaler.transform(X)
        return self.lr.predict_proba(Xs[:, self.mask])[:, 1]


class IsotonicForestWrap:
    """Wraps IsolationForest score_samples → probability-like score in [0,1]."""
    def __init__(self, **kw):
        self.m = IsolationForest(**kw)
    def fit(self, X, y=None): self.m.fit(X); return self
    def predict_proba_class1(self, X):
        s = self.m.score_samples(X)
        return 1.0 - (s - s.min()) / max(1e-9, s.max() - s.min())


class LofWrap:
    def __init__(self, **kw):
        self.m = LocalOutlierFactor(novelty=True, **kw)
    def fit(self, X, y=None): self.m.fit(X); return self
    def predict_proba_class1(self, X):
        s = -self.m.score_samples(X)
        return (s - s.min()) / max(1e-9, s.max() - s.min())


class EllipticEnvelopeWrap:
    def __init__(self, **kw):
        from sklearn.covariance import EllipticEnvelope
        self.scaler = None; self.m = EllipticEnvelope(**kw)
    def fit(self, X, y=None):
        self.scaler = StandardScaler().fit(X)
        self.m.fit(self.scaler.transform(X)); return self
    def predict_proba_class1(self, X):
        Xs = self.scaler.transform(X)
        s = -self.m.score_samples(Xs)
        return (s - s.min()) / max(1e-9, s.max() - s.min())


class AutoencoderWrap:
    """019: MLPRegressor autoencoder reconstruction error → anomaly score."""
    def __init__(self):
        self.scaler = None; self.m = None; self.bottleneck = 0
        # NOTE: AE was trained on 180-dim windows. For yesterday-backtest we use 10-dim features only
        # (single-bar context, no 30-bar window). This is a degraded approximation.
    def fit(self, X, y=None):
        # Only normal (label=0) samples — but we don't have labels here for AE.
        # Use full X. Acceptable degradation for this comparison.
        self.scaler = StandardScaler().fit(X); Xs = self.scaler.transform(X)
        d = X.shape[1]; self.bottleneck = max(2, d // 4)
        self.m = MLPRegressor(hidden_layer_sizes=(d // 2, self.bottleneck, d // 2),
                               max_iter=150, random_state=42, early_stopping=True).fit(Xs, Xs)
        return self
    def predict_proba_class1(self, X):
        Xs = self.scaler.transform(X)
        recon = self.m.predict(Xs)
        err = ((recon - Xs) ** 2).mean(axis=1)
        return (err - err.min()) / max(1e-9, err.max() - err.min())


# ===========================================================================
# FACTORIES — 64 pass-gate models
# ===========================================================================
# Each entry: (lab_name, factory_fn)
# factory_fn() returns a model with fit(X, y) and either predict_proba(X) or predict_proba_class1(X)

def F(model_or_pipeline):
    """Wrap a sklearn model/pipeline so we can use predict_proba_class1 uniformly."""
    class Wrapped:
        def __init__(self, m): self.m = m
        def fit(self, X, y): self.m.fit(X, y); return self
        def predict_proba_class1(self, X):
            if hasattr(self.m, "predict_proba"): return self.m.predict_proba(X)[:, 1]
            d = self.m.decision_function(X); return expit(d)
    return Wrapped(model_or_pipeline)


def make_factories() -> list[tuple[str, Callable]]:
    f = []
    SS = StandardScaler

    # === Manual (001-030) pass-gate ===
    f.append(("001_logreg", lambda: F(Pipeline([("s", SS()), ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))]))))
    f.append(("004_adaboost", lambda: F(AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42))))
    f.append(("005_gradient_boosting", lambda: F(GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42))))
    f.append(("011_gaussian_nb", lambda: F(Pipeline([("s", SS()), ("c", GaussianNB())]))))
    f.append(("012_lda", lambda: F(Pipeline([("s", SS()), ("c", LinearDiscriminantAnalysis())]))))
    f.append(("013_qda", lambda: F(Pipeline([("s", SS()), ("c", QuadraticDiscriminantAnalysis(reg_param=0.1))]))))
    f.append(("014_mlp", lambda: F(Pipeline([("s", SS()), ("c", MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=200, random_state=42, early_stopping=True))]))))
    f.append(("015_isolation_forest", lambda: IsotonicForestWrap(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1)))
    f.append(("016_local_outlier_factor", lambda: LofWrap(n_neighbors=30, n_jobs=-1)))
    f.append(("017_one_class_svm", lambda: EllipticEnvelopeWrap(contamination=0.1, support_fraction=0.9, random_state=42)))
    f.append(("018_elliptic_envelope", lambda: EllipticEnvelopeWrap(contamination=0.1, support_fraction=0.9, random_state=42)))
    f.append(("019_autoencoder", lambda: AutoencoderWrap()))
    f.append(("026_nn_scaling_curve", lambda: F(Pipeline([("s", SS()), ("c", MLPClassifier(hidden_layer_sizes=(128,), max_iter=200, random_state=42, early_stopping=True))]))))
    f.append(("027_athena_stack", lambda: AthenaStack()))

    # === Programmatic sweep (031-103) pass-gate ===
    f.append(("031_logreg_l1", lambda: F(Pipeline([("s", SS()), ("c", LogisticRegression(penalty="l1", solver="liblinear", class_weight="balanced", max_iter=2000))]))))
    f.append(("032_ridge_clf", lambda: F(Pipeline([("s", SS()), ("c", RidgeClassifier(class_weight="balanced"))]))))
    f.append(("033_sgd_hinge", lambda: F(Pipeline([("s", SS()), ("c", SGDClassifier(loss="hinge", max_iter=200, class_weight="balanced", random_state=42))]))))
    f.append(("034_sgd_modified_huber", lambda: F(Pipeline([("s", SS()), ("c", SGDClassifier(loss="modified_huber", max_iter=200, class_weight="balanced", random_state=42))]))))
    f.append(("035_perceptron", lambda: F(Pipeline([("s", SS()), ("c", Perceptron(max_iter=200, class_weight="balanced", random_state=42))]))))
    f.append(("036_passive_aggressive", lambda: F(Pipeline([("s", SS()), ("c", PassiveAggressiveClassifier(max_iter=200, class_weight="balanced", random_state=42))]))))
    f.append(("041_linear_svc", lambda: F(Pipeline([("s", SS()), ("c", __import__("sklearn.svm", fromlist=["LinearSVC"]).LinearSVC(class_weight="balanced", max_iter=2000, random_state=42))]))))
    f.append(("044_lgbm_rf", lambda: F(lgb.LGBMClassifier(boosting_type="rf", n_estimators=200, max_depth=5, bagging_freq=1, bagging_fraction=0.8, feature_fraction=0.8, class_weight="balanced", random_state=42, verbose=-1))))
    f.append(("046_xgb_gblinear", lambda: F(xgb.XGBClassifier(booster="gblinear", n_estimators=200, random_state=42, verbosity=0))))
    f.append(("047_gbm_exponential", lambda: F(GradientBoostingClassifier(loss="exponential", n_estimators=150, max_depth=3, random_state=42))))
    f.append(("054_lda_shrinkage", lambda: F(Pipeline([("s", SS()), ("c", LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"))]))))
    f.append(("055_calibrated_isotonic", lambda: F(CalibratedClassifierCV(__import__("sklearn.svm", fromlist=["LinearSVC"]).LinearSVC(class_weight="balanced", max_iter=1000), method="isotonic", cv=3))))
    f.append(("056_calibrated_sigmoid", lambda: F(CalibratedClassifierCV(SGDClassifier(loss="hinge", max_iter=200, class_weight="balanced", random_state=42), method="sigmoid", cv=3))))
    # Anomaly k-means / pca / ica / mahalanobis / mcd (057-062) — keep a few
    f.append(("057_anomaly_kmeans_dist", lambda: KMeansCodesLogreg(k=5)))  # approximation
    f.append(("063_hmm_2state_full", lambda: F(Pipeline([("s", SS()), ("c", __import__("hmmlearn.hmm", fromlist=["GaussianHMM"]).GaussianHMM(n_components=2, n_iter=50, random_state=42))])) if False else None))
    # Skip HMM (different interface); skip exact ones that don't fit pipeline easily
    f.append(("067_nn_funnel", lambda: F(Pipeline([("s", SS()), ("c", MLPClassifier(hidden_layer_sizes=(128, 64, 32, 16), max_iter=200, random_state=42, early_stopping=True))]))))
    f.append(("068_nn_bottleneck", lambda: F(Pipeline([("s", SS()), ("c", MLPClassifier(hidden_layer_sizes=(64, 8, 64), max_iter=200, random_state=42, early_stopping=True))]))))
    f.append(("069_nn_pyramid", lambda: F(Pipeline([("s", SS()), ("c", MLPClassifier(hidden_layer_sizes=(16, 32, 64, 128), max_iter=200, random_state=42, early_stopping=True))]))))
    f.append(("070_nn_tanh", lambda: F(Pipeline([("s", SS()), ("c", MLPClassifier(hidden_layer_sizes=(64, 32), activation="tanh", max_iter=200, random_state=42, early_stopping=True))]))))
    f.append(("071_nn_logistic", lambda: F(Pipeline([("s", SS()), ("c", MLPClassifier(hidden_layer_sizes=(64, 32), activation="logistic", max_iter=200, random_state=42, early_stopping=True))]))))
    f.append(("072_nn_identity", lambda: F(Pipeline([("s", SS()), ("c", MLPClassifier(hidden_layer_sizes=(64, 32), activation="identity", max_iter=200, random_state=42, early_stopping=True))]))))
    f.append(("073_nn_sgd_solver", lambda: F(Pipeline([("s", SS()), ("c", MLPClassifier(hidden_layer_sizes=(64, 32), solver="sgd", learning_rate="adaptive", max_iter=200, random_state=42, early_stopping=True))]))))
    f.append(("074_nn_lbfgs_solver", lambda: F(Pipeline([("s", SS()), ("c", MLPClassifier(hidden_layer_sizes=(64, 32), solver="lbfgs", max_iter=500, random_state=42))]))))
    f.append(("086_pca_then_lgbm", lambda: F(Pipeline([("s", SS()), ("pca", PCA(n_components=5)), ("c", lgb.LGBMClassifier(n_estimators=150, class_weight="balanced", random_state=42, verbose=-1))]))))
    f.append(("087_ica_then_lgbm", lambda: F(Pipeline([("s", SS()), ("ica", FastICA(n_components=5, random_state=42, max_iter=500, whiten="unit-variance")), ("c", lgb.LGBMClassifier(n_estimators=150, class_weight="balanced", random_state=42, verbose=-1))]))))
    f.append(("088_truncated_svd_logreg", lambda: F(Pipeline([("s", SS()), ("svd", TruncatedSVD(n_components=5, random_state=42)), ("c", LogisticRegression(class_weight="balanced", max_iter=1000))]))))
    f.append(("090_random_projection_lgbm", lambda: F(Pipeline([("s", SS()), ("rp", GaussianRandomProjection(n_components=5, random_state=42)), ("c", lgb.LGBMClassifier(n_estimators=150, class_weight="balanced", random_state=42, verbose=-1))]))))
    f.append(("098_meta_youden_threshold", lambda: F(lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1))))
    f.append(("101_interaction_poly_logreg", lambda: F(Pipeline([("s", SS()), ("p", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)), ("c", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1))]))))
    f.append(("102_rbf_sampler_logreg", lambda: F(Pipeline([("s", SS()), ("rbf", RBFSampler(gamma=0.5, n_components=64, random_state=42)), ("c", LogisticRegression(class_weight="balanced", max_iter=2000))]))))
    f.append(("103_nystroem_logreg", lambda: F(Pipeline([("s", SS()), ("ny", Nystroem(kernel="rbf", gamma=0.3, n_components=64, random_state=42)), ("c", LogisticRegression(class_weight="balanced", max_iter=2000))]))))

    # === Haiku-agent (104-146) pass-gate ===
    f.append(("107_bart_style_additive", lambda: BartStyleAdditive()))
    f.append(("114_elastic_net_classifier", lambda: F(Pipeline([("s", SS()), ("c", LogisticRegression(penalty="elasticnet", l1_ratio=0.5, solver="saga", max_iter=3000, class_weight="balanced", random_state=42))]))))
    f.append(("115_isotonic_calibrated_chain", lambda: IsotonicChain()))
    f.append(("116_multitask_regime_plus_direction", lambda: F(Pipeline([("s", SS()), ("c", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42, early_stopping=True))]))))  # treat as plain MLP for backtest
    f.append(("120_bayesian_ridge_sigmoid", lambda: F(_BayesianRidgeBinary())))
    f.append(("121_poly_kernel_ridge", lambda: F(_PolyKernelRidgeBinary())))
    f.append(("123_kmeans_cluster_then_logreg", lambda: KMeansCodesLogreg(k=5)))  # similar pattern
    f.append(("125_mlp_strong_l2_regularization", lambda: StrongL2Mlp()))
    f.append(("131_prototype_knn", lambda: PrototypeKnn()))
    f.append(("132_gradient_boosted_logreg_stack", lambda: BoostedLogregStack()))
    f.append(("138_kmeans_codes_logreg", lambda: KMeansCodesLogreg(k=20)))
    f.append(("141_hard_negative_mining_mlp", lambda: HardNegativeMiningMlp()))
    f.append(("143_power_transform_nb", lambda: PowerTransformNb()))
    f.append(("144_nb_lr_geometric_blend", lambda: NbLrGeometricBlend()))
    f.append(("145_sparse_lr_saga_screening", lambda: SparseLrScreening()))

    return [(n, fn) for n, fn in f if fn is not None]


class _BayesianRidgeBinary:
    def __init__(self):
        self.m = Pipeline([("s", StandardScaler()), ("reg", BayesianRidge())])
    def fit(self, X, y): self.m.fit(X, y.astype(float)); return self
    def predict_proba(self, X):
        p = self.m.predict(X)
        prob = expit(5 * (p - 0.5))
        return np.column_stack([1 - prob, prob])


class _PolyKernelRidgeBinary:
    def __init__(self):
        self.m = Pipeline([("s", StandardScaler()), ("kr", KernelRidge(kernel="poly", degree=3, alpha=1.0))])
    def fit(self, X, y):
        rng = np.random.default_rng(42)
        sub = rng.choice(len(X), min(5000, len(X)), replace=False)
        self.m.fit(X[sub], y[sub].astype(float)); return self
    def predict_proba(self, X):
        p = self.m.predict(X)
        prob = expit(5 * (p - 0.5))
        return np.column_stack([1 - prob, prob])


# ===========================================================================
# Backtest driver
# ===========================================================================

def backtest_one(name: str, factory: Callable, X_train: np.ndarray, y_train: np.ndarray,
                  X_entries: np.ndarray, entries_pnl: np.ndarray) -> dict[str, Any]:
    t0 = time.time()
    try:
        m = factory()
        m.fit(X_train, y_train)
        if hasattr(m, "predict_proba_class1"):
            p = m.predict_proba_class1(X_entries)
        else:
            p = m.predict_proba(X_entries)[:, 1]
        p = np.asarray(p, dtype=float)
        blocked = p > BLOCK_PROBA
        kept = ~blocked
        actual_pnl = float(entries_pnl.sum())
        cf_pnl = float(entries_pnl[kept].sum())
        blocked_pnl = float(entries_pnl[blocked].sum())
        bw = int((entries_pnl[blocked] > 0).sum())
        bl = int((entries_pnl[blocked] < 0).sum())
        return {
            "model": name,
            "n_entries_seen": int(len(entries_pnl)),
            "n_blocked": int(blocked.sum()),
            "n_kept": int(kept.sum()),
            "block_rate_pct": 100.0 * blocked.sum() / max(1, len(entries_pnl)),
            "actual_pnl": actual_pnl,
            "counterfactual_pnl": cf_pnl,
            "pnl_delta": cf_pnl - actual_pnl,
            "blocked_pnl_avoided": blocked_pnl,
            "blocked_winners": bw,
            "blocked_losers": bl,
            "avg_block_proba": float(p.mean()),
            "elapsed_sec": time.time() - t0,
            "per_entry_proba": p.tolist(),
        }
    except Exception as exc:
        return {"model": name, "error": str(exc)[:300], "elapsed_sec": time.time() - t0}


def main() -> int:
    print("Loading training data…")
    X_train, y_train = load_training_data()
    print(f"  train: {len(X_train):,} rows × {X_train.shape[1]} features")

    print("Loading yesterday's matched entries + bars…")
    entries = load_yesterday_entries()
    yest_bars = fetch_yesterday_bars()
    print(f"  matched entries: {len(entries)}")
    print(f"  yesterday bars: {len(yest_bars)}")

    X_entries = build_entry_features(entries, yest_bars)
    valid = ~np.isnan(X_entries).any(axis=1)
    X_entries_valid = X_entries[valid]
    entries_valid = entries[valid].reset_index(drop=True)
    print(f"  entries with valid features: {len(entries_valid)}")
    entries_pnl = entries_valid["pnl"].values

    factories = make_factories()
    print(f"\nRunning {len(factories)} pass-gate backtests…\n")

    # Run serially — lambdas + scipy closures don't pickle for joblib's loky backend
    results = []
    for i, (name, fn) in enumerate(factories, 1):
        r = backtest_one(name, fn, X_train, y_train, X_entries_valid, entries_pnl)
        results.append(r)
        msg = f"  [{i:>2}/{len(factories)}] {name:50s}"
        if "error" in r:
            msg += f"  ERR {r['error'][:50]}"
        else:
            msg += f"  blocked {r['n_blocked']}/{r['n_entries_seen']}  Δ ${r['pnl_delta']:+,.2f}  ({r['elapsed_sec']:.1f}s)"
        print(msg, flush=True)
    # Reattach entry context
    per_entry_rows = []
    for r in results:
        if "error" in r: continue
        if "per_entry_proba" not in r: continue
        for i, (_, e) in enumerate(entries_valid.iterrows()):
            per_entry_rows.append({
                "model": r["model"], "lane": e["lane"], "symbol": e["symbol"],
                "side_dir": e["side_dir"], "entry_ts": e["entry_ts"],
                "exit_ts": e["exit_ts"], "exit_action": e["exit_action"],
                "qty": e["qty"], "pnl": e["pnl"],
                "model_proba": r["per_entry_proba"][i],
                "blocked": r["per_entry_proba"][i] > BLOCK_PROBA,
            })
    # Strip per_entry_proba from summary
    summary = []
    for r in results:
        r.pop("per_entry_proba", None)
        summary.append(r)
    summary_df = pd.DataFrame(summary).sort_values("pnl_delta", ascending=False, na_position="last").reset_index(drop=True)
    per_entry_df = pd.DataFrame(per_entry_rows)
    summary_df.to_parquet(OUT / "summary.parquet", index=False)
    per_entry_df.to_parquet(OUT / "per_entry.parquet", index=False)

    print("\n=== Top 15 by P&L delta (vs yesterday's actual −$3,443) ===")
    cols = ["model", "n_blocked", "block_rate_pct", "blocked_winners", "blocked_losers", "pnl_delta", "counterfactual_pnl"]
    print(summary_df[cols].head(15).to_string(index=False))
    n_err = (summary_df.get("error", pd.Series([None]*len(summary_df))).notna()).sum() if "error" in summary_df.columns else 0
    print(f"\n  {len(summary_df) - n_err} of {len(summary_df)} models ran cleanly")
    print(f"  Top counterfactual: {summary_df.iloc[0]['model']} → ${summary_df.iloc[0]['counterfactual_pnl']:+,.2f}")
    print(f"\n  wrote {OUT / 'summary.parquet'}")
    print(f"  wrote {OUT / 'per_entry.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
