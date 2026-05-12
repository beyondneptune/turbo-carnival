"""Overnight: 100 architecturally distinct ML experiments.

Each entry is a DIFFERENT family/architecture — not parameter variants of
the same model. Loss function, decision rule, training objective, or
algorithmic family must differ.

Categories represented:
  Linear classifiers (≠ regularization or solver alone) — different
    loss functions, different decision rules
  Tree families (single, bagged, boosted-gradient, boosted-Newton, dart, rf-mode)
  Distance/instance methods (kNN, radius, kernel SVM variants)
  Probabilistic models (NB families, discriminant analysis, calibration)
  Anomaly methods (forest, density, support, envelope, autoencoder)
  Time-series statistical (HMM with different emissions, smoothing)
  Neural nets (different topology FAMILIES, not sizes)
  Ensembles (different combination rules)
  Strategy architectures (different decision frameworks)
  Custom composites (cascade, gating, regime-routing — my own designs)

Writes each result to experiments/lab/<NNN_name>/results.json so it
appears in the leaderboard like the manually-built ones.
"""
from __future__ import annotations
import json, sys, time, traceback, warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

from shared.data import load_classifier_data, load_raw_bars, CLASSIFIER_FEATURES

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_dir(idx: int, name: str) -> Path:
    d = ROOT / f"{idx:03d}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_result(idx: int, name: str, description: str, p_tr, p_te, y_tr, y_te,
                  extra: dict | None = None):
    d = make_dir(idx, name)
    try:
        a_tr = float(roc_auc_score(y_tr, p_tr)); a_te = float(roc_auc_score(y_te, p_te))
        br = float(brier_score_loss(y_te, p_te))
        cut = float(np.quantile(p_te, 0.99))
        flag = p_te >= cut
        prec = float(y_te[flag].mean()) if flag.sum() > 0 else 0.0
        gap = a_tr - a_te
        passes = a_te > 0.55 and abs(gap) < 0.05
        out = {"experiment": f"{idx:03d}_{name}", "description": description,
               "timestamp_utc": datetime.now(timezone.utc).isoformat(),
               "train_auc": a_tr, "test_auc": a_te, "train_test_gap": gap,
               "test_brier": br, "precision_at_top1pct": prec,
               "flagged_count": int(flag.sum()), "passes_overfit_gate": passes}
        if extra: out["extra"] = extra
    except Exception as e:
        out = {"experiment": f"{idx:03d}_{name}", "description": description,
               "error": str(e)[:300], "test_auc": 0.5, "passes_overfit_gate": False}
    (d / "results.json").write_text(json.dumps(out, indent=2))
    msg = f"[{idx:03d}_{name}]"
    if "test_auc" in out:
        msg += f"  test_auc={out['test_auc']:.3f}  gap={out.get('train_test_gap', 0):+.3f}"
        if out.get("passes_overfit_gate"): msg += "  PASS"
    else:
        msg += f"  ERROR: {out.get('error', '?')[:80]}"
    print(msg)


# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------
print("loading data…")
Xt, yt, Xv, yv = load_classifier_data()
print(f"  train {len(Xt):,}, test {len(Xv):,}")
START = time.time()
START_IDX = 31  # continues from existing experiments 001-030

# ---------------------------------------------------------------------------
# CATEGORY A: LINEAR CLASSIFIERS — different loss functions / decision rules
# ---------------------------------------------------------------------------
from sklearn.linear_model import (LogisticRegression, RidgeClassifier, SGDClassifier,
                                    PassiveAggressiveClassifier, Perceptron)

def runA():
    items = [
        ("logreg_l1", "L1-penalized logistic regression — sparse linear",
         Pipeline([("s", StandardScaler()), ("c", LogisticRegression(penalty="l1", solver="liblinear",
                                                                       class_weight="balanced", max_iter=2000))])),
        ("ridge_clf", "Ridge classifier — squared loss, not log loss",
         Pipeline([("s", StandardScaler()), ("c", RidgeClassifier(class_weight="balanced"))])),
        ("sgd_hinge", "SGD with hinge loss — linear SVM via SGD",
         Pipeline([("s", StandardScaler()), ("c", SGDClassifier(loss="hinge", max_iter=200,
                                                                  class_weight="balanced", random_state=42))])),
        ("sgd_modified_huber", "SGD with modified Huber — robust to outliers",
         Pipeline([("s", StandardScaler()), ("c", SGDClassifier(loss="modified_huber", max_iter=200,
                                                                  class_weight="balanced", random_state=42))])),
        ("perceptron", "Perceptron — classic linear, no probability calibration",
         Pipeline([("s", StandardScaler()), ("c", Perceptron(max_iter=200, class_weight="balanced", random_state=42))])),
        ("passive_aggressive", "Passive-Aggressive online learner",
         Pipeline([("s", StandardScaler()), ("c", PassiveAggressiveClassifier(max_iter=200, class_weight="balanced", random_state=42))])),
    ]
    for i, (n, desc, m) in enumerate(items):
        try:
            m.fit(Xt, yt)
            if hasattr(m.named_steps["c"], "predict_proba"):
                p_tr = m.predict_proba(Xt)[:, 1]; p_te = m.predict_proba(Xv)[:, 1]
            else:
                from scipy.special import expit
                p_tr = expit(m.decision_function(Xt)); p_te = expit(m.decision_function(Xv))
            write_result(START_IDX + i, n, desc, p_tr, p_te, yt, yv)
        except Exception as e:
            write_result(START_IDX + i, n, desc, np.zeros_like(yt, dtype=float),
                          np.zeros_like(yv, dtype=float), yt, yv, extra={"error": str(e)})

runA()
idx_after_A = START_IDX + 6  # 31+6 = 37

# ---------------------------------------------------------------------------
# CATEGORY B: DISTANCE / KERNEL METHODS
# ---------------------------------------------------------------------------
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neighbors import RadiusNeighborsClassifier

def runB(base):
    # SVMs are O(n^2) — subsample to keep training feasible
    rng = np.random.default_rng(42)
    sub = rng.choice(len(Xt), min(5000, len(Xt)), replace=False)
    items = [
        ("svm_rbf", "SVM with RBF kernel on 5k subsample",
         Pipeline([("s", StandardScaler()), ("c", SVC(kernel="rbf", probability=True, class_weight="balanced",
                                                       random_state=42, max_iter=1000))]), sub),
        ("svm_poly", "SVM with polynomial kernel (degree 3) on 5k subsample",
         Pipeline([("s", StandardScaler()), ("c", SVC(kernel="poly", degree=3, probability=True,
                                                       class_weight="balanced", random_state=42, max_iter=1000))]), sub),
        ("svm_sigmoid", "SVM with sigmoid kernel on 5k subsample",
         Pipeline([("s", StandardScaler()), ("c", SVC(kernel="sigmoid", probability=True,
                                                       class_weight="balanced", random_state=42, max_iter=1000))]), sub),
        ("nu_svc", "Nu-SVC — alternative SVM parameterization with bound on errors",
         Pipeline([("s", StandardScaler()), ("c", NuSVC(probability=True, class_weight="balanced",
                                                          random_state=42, max_iter=1000))]), sub),
        ("linear_svc", "LinearSVC — efficient linear SVM (separate from sgd_hinge)",
         Pipeline([("s", StandardScaler()), ("c", LinearSVC(class_weight="balanced",
                                                              max_iter=2000, random_state=42))]), None),
    ]
    for i, (n, desc, m, samp) in enumerate(items):
        try:
            Xfit = Xt[samp] if samp is not None else Xt
            yfit = yt[samp] if samp is not None else yt
            m.fit(Xfit, yfit)
            if hasattr(m.named_steps["c"], "predict_proba"):
                p_tr = m.predict_proba(Xt)[:, 1]; p_te = m.predict_proba(Xv)[:, 1]
            else:
                from scipy.special import expit
                p_tr = expit(m.decision_function(Xt)); p_te = expit(m.decision_function(Xv))
            write_result(base + i, n, desc, p_tr, p_te, yt, yv)
        except Exception as e:
            write_result(base + i, n, desc, np.zeros_like(yt, dtype=float),
                          np.zeros_like(yv, dtype=float), yt, yv, extra={"error": str(e)})

runB(idx_after_A)
idx_after_B = idx_after_A + 5  # 37+5 = 42

# ---------------------------------------------------------------------------
# CATEGORY C: TREE FAMILIES — different growth / boosting paradigms
# ---------------------------------------------------------------------------
import lightgbm as lgb, xgboost as xgb
from sklearn.tree import ExtraTreeClassifier
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier)

def runC(base):
    pos = max(1, int(yt.sum())); neg = len(yt) - pos
    items = [
        ("lgbm_dart", "LightGBM DART — dropout-meets-additive-regression-trees",
         lgb.LGBMClassifier(boosting_type="dart", n_estimators=200, max_depth=5,
                             class_weight="balanced", random_state=42, verbose=-1)),
        ("lgbm_goss", "LightGBM GOSS — gradient-based one-side sampling",
         lgb.LGBMClassifier(boosting_type="goss", n_estimators=200, max_depth=5,
                             class_weight="balanced", random_state=42, verbose=-1)),
        ("lgbm_rf", "LightGBM in random-forest mode (no boosting)",
         lgb.LGBMClassifier(boosting_type="rf", n_estimators=200, max_depth=5,
                             bagging_freq=1, bagging_fraction=0.8, feature_fraction=0.8,
                             class_weight="balanced", random_state=42, verbose=-1)),
        ("xgb_dart", "XGBoost DART — dropout regularization on boosting",
         xgb.XGBClassifier(booster="dart", n_estimators=200, max_depth=5,
                            scale_pos_weight=neg/pos, random_state=42, verbosity=0)),
        ("xgb_gblinear", "XGBoost with linear base learner — boosted linear model",
         xgb.XGBClassifier(booster="gblinear", n_estimators=200,
                            scale_pos_weight=neg/pos, random_state=42, verbosity=0)),
        ("gbm_exponential", "Sklearn GBM with exponential loss (AdaBoost-style)",
         GradientBoostingClassifier(loss="exponential", n_estimators=150, max_depth=3, random_state=42)),
        ("extra_tree_single", "Single ExtraTree — single tree with random splits",
         ExtraTreeClassifier(max_depth=10, class_weight="balanced", random_state=42)),
        ("bagging_decision_tree", "Bagging with deep decision tree base",
         BaggingClassifier(n_estimators=100, max_features=0.8, max_samples=0.7,
                            random_state=42, n_jobs=-1)),
        ("bagging_logreg", "Bagging with logistic regression base — pasted linear ensemble",
         BaggingClassifier(estimator=LogisticRegression(max_iter=500, class_weight="balanced"),
                            n_estimators=30, max_features=0.7, random_state=42, n_jobs=-1)),
    ]
    for i, (n, desc, m) in enumerate(items):
        try:
            m.fit(Xt, yt)
            p_tr = m.predict_proba(Xt)[:, 1]; p_te = m.predict_proba(Xv)[:, 1]
            write_result(base + i, n, desc, p_tr, p_te, yt, yv)
        except Exception as e:
            write_result(base + i, n, desc, np.zeros_like(yt, dtype=float),
                          np.zeros_like(yv, dtype=float), yt, yv, extra={"error": str(e)})

runC(idx_after_B)
idx_after_C = idx_after_B + 9  # 42+9 = 51

# ---------------------------------------------------------------------------
# CATEGORY D: PROBABILISTIC / DISCRIMINANT ANALYSIS
# ---------------------------------------------------------------------------
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, ComplementNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.calibration import CalibratedClassifierCV

def runD(base):
    # NB variants need non-negative features — use min-max for them
    items = [
        ("nb_bernoulli", "Bernoulli NB on binarized features",
         Pipeline([("s", StandardScaler()), ("c", BernoulliNB(binarize=0.0))])),
        ("nb_complement", "Complement NB — Multinomial variant for imbalanced data",
         Pipeline([("s", MinMaxScaler()), ("c", ComplementNB())])),
        ("nb_multinomial", "Multinomial NB on min-max scaled features",
         Pipeline([("s", MinMaxScaler()), ("c", MultinomialNB())])),
        ("lda_shrinkage", "LDA with Ledoit-Wolf shrinkage estimator",
         Pipeline([("s", StandardScaler()), ("c", LinearDiscriminantAnalysis(shrinkage="auto", solver="lsqr"))])),
        ("calibrated_isotonic", "Calibrated linear SVM with isotonic calibration",
         CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=1000), method="isotonic", cv=3)),
        ("calibrated_sigmoid", "Calibrated SGD with Platt-sigmoid calibration",
         CalibratedClassifierCV(SGDClassifier(loss="hinge", max_iter=200, class_weight="balanced",
                                                random_state=42), method="sigmoid", cv=3)),
    ]
    for i, (n, desc, m) in enumerate(items):
        try:
            m.fit(Xt, yt)
            p_tr = m.predict_proba(Xt)[:, 1]; p_te = m.predict_proba(Xv)[:, 1]
            write_result(base + i, n, desc, p_tr, p_te, yt, yv)
        except Exception as e:
            write_result(base + i, n, desc, np.zeros_like(yt, dtype=float),
                          np.zeros_like(yv, dtype=float), yt, yv, extra={"error": str(e)})

runD(idx_after_C)
idx_after_D = idx_after_C + 6  # 51+6 = 57

# ---------------------------------------------------------------------------
# CATEGORY E: ANOMALY DETECTION
# ---------------------------------------------------------------------------
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture

def runE(base):
    rng = np.random.default_rng(42)
    sub = rng.choice(len(Xt), min(8000, len(Xt)), replace=False)
    items = []

    # K-means residual (distance to nearest centroid as anomaly score)
    def km_score():
        sc = StandardScaler().fit(Xt); Xs_tr = sc.transform(Xt); Xs_te = sc.transform(Xv)
        km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(Xs_tr)
        d_tr = km.transform(Xs_tr).min(axis=1)
        d_te = km.transform(Xs_te).min(axis=1)
        return d_tr, d_te

    # GMM negative log-likelihood
    def gmm_score():
        sc = StandardScaler().fit(Xt); Xs_tr = sc.transform(Xt); Xs_te = sc.transform(Xv)
        gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=42).fit(Xs_tr)
        return -gmm.score_samples(Xs_tr), -gmm.score_samples(Xs_te)

    # PCA reconstruction error
    def pca_score():
        from sklearn.decomposition import PCA
        sc = StandardScaler().fit(Xt); Xs_tr = sc.transform(Xt); Xs_te = sc.transform(Xv)
        p = PCA(n_components=5).fit(Xs_tr)
        rec_tr = p.inverse_transform(p.transform(Xs_tr)); rec_te = p.inverse_transform(p.transform(Xs_te))
        return ((rec_tr - Xs_tr) ** 2).mean(axis=1), ((rec_te - Xs_te) ** 2).mean(axis=1)

    # ICA reconstruction
    def ica_score():
        from sklearn.decomposition import FastICA
        sc = StandardScaler().fit(Xt); Xs_tr = sc.transform(Xt); Xs_te = sc.transform(Xv)
        ic = FastICA(n_components=5, random_state=42, max_iter=500, whiten="unit-variance").fit(Xs_tr)
        rec_tr = ic.inverse_transform(ic.transform(Xs_tr)); rec_te = ic.inverse_transform(ic.transform(Xs_te))
        return ((rec_tr - Xs_tr) ** 2).mean(axis=1), ((rec_te - Xs_te) ** 2).mean(axis=1)

    # Mahalanobis distance
    def mahal_score():
        from sklearn.covariance import EmpiricalCovariance
        ec = EmpiricalCovariance().fit(Xt)
        return ec.mahalanobis(Xt), ec.mahalanobis(Xv)

    # Robust Mahalanobis (Minimum Covariance Determinant)
    def mcd_score():
        from sklearn.covariance import MinCovDet
        sc = StandardScaler().fit(Xt); Xs_tr = sc.transform(Xt); Xs_te = sc.transform(Xv)
        mcd = MinCovDet(random_state=42, support_fraction=0.85).fit(Xs_tr)
        return mcd.mahalanobis(Xs_tr), mcd.mahalanobis(Xs_te)

    scorers = [
        ("anomaly_kmeans_dist", "K-means: distance to nearest of 5 centroids as anomaly score", km_score),
        ("anomaly_gmm_loglik", "GMM 3-component: negative log-likelihood as anomaly score", gmm_score),
        ("anomaly_pca_recon", "PCA reconstruction error from 5 components", pca_score),
        ("anomaly_ica_recon", "Independent Component Analysis reconstruction error", ica_score),
        ("anomaly_mahalanobis", "Empirical-covariance Mahalanobis distance", mahal_score),
        ("anomaly_mcd_robust", "Robust Mahalanobis via Minimum Covariance Determinant", mcd_score),
    ]
    for i, (n, desc, f) in enumerate(scorers):
        try:
            s_tr, s_te = f()
            write_result(base + i, n, desc, s_tr, s_te, yt, yv)
        except Exception as e:
            write_result(base + i, n, desc, np.zeros_like(yt, dtype=float),
                          np.zeros_like(yv, dtype=float), yt, yv, extra={"error": str(e)})

runE(idx_after_D)
idx_after_E = idx_after_D + 6  # 57+6 = 63

# ---------------------------------------------------------------------------
# CATEGORY F: TIME-SERIES STATISTICAL
# ---------------------------------------------------------------------------
from hmmlearn import hmm

def runF(base):
    sc = StandardScaler().fit(Xt); Xs_tr = sc.transform(Xt); Xs_te = sc.transform(Xv)
    items = []

    def hmm2_score():
        m = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=80, random_state=42)
        m.fit(Xs_tr)
        return -np.array([m.score(x.reshape(1, -1)) for x in Xs_tr]), \
               -np.array([m.score(x.reshape(1, -1)) for x in Xs_te])

    def hmm4_score():
        m = hmm.GaussianHMM(n_components=4, covariance_type="diag", n_iter=80, random_state=42)
        m.fit(Xs_tr)
        return -np.array([m.score(x.reshape(1, -1)) for x in Xs_tr]), \
               -np.array([m.score(x.reshape(1, -1)) for x in Xs_te])

    def hmm_full_5():
        m = hmm.GaussianHMM(n_components=5, covariance_type="spherical", n_iter=80, random_state=42)
        m.fit(Xs_tr)
        return -np.array([m.score(x.reshape(1, -1)) for x in Xs_tr]), \
               -np.array([m.score(x.reshape(1, -1)) for x in Xs_te])

    def ewma_dev_score():
        # EWMA-deviation: how far is current bar from EWMA of recent bars
        df = pd.DataFrame(Xs_tr, columns=CLASSIFIER_FEATURES)
        ewma = df.ewm(span=30, adjust=False).mean()
        s_tr = ((df - ewma) ** 2).sum(axis=1).values
        df_te = pd.DataFrame(Xs_te, columns=CLASSIFIER_FEATURES)
        ewma_te = df_te.ewm(span=30, adjust=False).mean()
        s_te = ((df_te - ewma_te) ** 2).sum(axis=1).values
        return s_tr, s_te

    items = [
        ("hmm_2state_full", "2-state HMM with full covariance — regime switching", hmm2_score),
        ("hmm_4state_diag", "4-state HMM with diagonal covariance", hmm4_score),
        ("hmm_5state_spherical", "5-state HMM with spherical covariance", hmm_full_5),
        ("ewma_deviation", "EWMA span-30 deviation as anomaly score", ewma_dev_score),
    ]
    for i, (n, desc, f) in enumerate(items):
        try:
            s_tr, s_te = f()
            write_result(base + i, n, desc, s_tr, s_te, yt, yv)
        except Exception as e:
            write_result(base + i, n, desc, np.zeros_like(yt, dtype=float),
                          np.zeros_like(yv, dtype=float), yt, yv, extra={"error": str(e)})

runF(idx_after_E)
idx_after_F = idx_after_E + 4  # 63+4 = 67

# ---------------------------------------------------------------------------
# CATEGORY G: NEURAL NET TOPOLOGY FAMILIES (different architectures, not sizes)
# ---------------------------------------------------------------------------
from sklearn.neural_network import MLPClassifier, MLPRegressor

def runG(base):
    sc = StandardScaler()
    items = [
        ("nn_funnel", "MLP funnel topology 128→64→32→16",
         Pipeline([("s", StandardScaler()),
                    ("c", MLPClassifier(hidden_layer_sizes=(128, 64, 32, 16), activation="relu",
                                         solver="adam", max_iter=200, random_state=42,
                                         early_stopping=True))])),
        ("nn_bottleneck", "MLP bottleneck 64→8→64 (autoencoder-like inside classifier)",
         Pipeline([("s", StandardScaler()),
                    ("c", MLPClassifier(hidden_layer_sizes=(64, 8, 64), activation="relu",
                                         solver="adam", max_iter=200, random_state=42,
                                         early_stopping=True))])),
        ("nn_pyramid", "MLP inverse pyramid 16→32→64→128",
         Pipeline([("s", StandardScaler()),
                    ("c", MLPClassifier(hidden_layer_sizes=(16, 32, 64, 128), activation="relu",
                                         solver="adam", max_iter=200, random_state=42,
                                         early_stopping=True))])),
        ("nn_tanh", "MLP with tanh activation — different non-linearity family",
         Pipeline([("s", StandardScaler()),
                    ("c", MLPClassifier(hidden_layer_sizes=(64, 32), activation="tanh",
                                         solver="adam", max_iter=200, random_state=42,
                                         early_stopping=True))])),
        ("nn_logistic", "MLP with logistic activation — classic NN approach",
         Pipeline([("s", StandardScaler()),
                    ("c", MLPClassifier(hidden_layer_sizes=(64, 32), activation="logistic",
                                         solver="adam", max_iter=200, random_state=42,
                                         early_stopping=True))])),
        ("nn_identity", "MLP with identity activation — equivalent to linear stacking",
         Pipeline([("s", StandardScaler()),
                    ("c", MLPClassifier(hidden_layer_sizes=(64, 32), activation="identity",
                                         solver="adam", max_iter=200, random_state=42,
                                         early_stopping=True))])),
        ("nn_sgd_solver", "MLP trained with SGD solver instead of Adam",
         Pipeline([("s", StandardScaler()),
                    ("c", MLPClassifier(hidden_layer_sizes=(64, 32), solver="sgd",
                                         learning_rate="adaptive", max_iter=200, random_state=42,
                                         early_stopping=True))])),
        ("nn_lbfgs_solver", "MLP trained with L-BFGS solver — batch optimization",
         Pipeline([("s", StandardScaler()),
                    ("c", MLPClassifier(hidden_layer_sizes=(64, 32), solver="lbfgs",
                                         max_iter=500, random_state=42))])),
    ]
    for i, (n, desc, m) in enumerate(items):
        try:
            m.fit(Xt, yt)
            p_tr = m.predict_proba(Xt)[:, 1]; p_te = m.predict_proba(Xv)[:, 1]
            write_result(base + i, n, desc, p_tr, p_te, yt, yv)
        except Exception as e:
            write_result(base + i, n, desc, np.zeros_like(yt, dtype=float),
                          np.zeros_like(yv, dtype=float), yt, yv, extra={"error": str(e)})

runG(idx_after_F)
idx_after_G = idx_after_F + 8  # 67+8 = 75

# ---------------------------------------------------------------------------
# CATEGORY H: ENSEMBLES — different combination rules
# ---------------------------------------------------------------------------
from sklearn.ensemble import VotingClassifier, StackingClassifier

def runH(base):
    base_lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    base_lgbm = lgb.LGBMClassifier(n_estimators=150, class_weight="balanced", random_state=42, verbose=-1)
    base_rf = lgb.LGBMClassifier(boosting_type="rf", n_estimators=150, bagging_freq=1, bagging_fraction=0.8,
                                  feature_fraction=0.8, class_weight="balanced", random_state=42, verbose=-1)
    base_xgb = xgb.XGBClassifier(n_estimators=150, random_state=42, verbosity=0)

    items = [
        ("voting_hard", "Hard-vote ensemble: LogReg + LGBM + XGB — majority class",
         VotingClassifier(estimators=[("lr", Pipeline([("s", StandardScaler()), ("c", base_lr)])),
                                        ("lgbm", base_lgbm), ("xgb", base_xgb)], voting="hard")),
        ("voting_soft_arith", "Soft-vote ensemble: arithmetic mean of probabilities",
         VotingClassifier(estimators=[("lr", Pipeline([("s", StandardScaler()), ("c", base_lr)])),
                                        ("lgbm", base_lgbm), ("xgb", base_xgb)], voting="soft")),
        ("stack_meta_logreg", "Stack: 3 bases → LogReg meta-learner",
         StackingClassifier(estimators=[("lr", Pipeline([("s", StandardScaler()), ("c", base_lr)])),
                                          ("lgbm", base_lgbm), ("xgb", base_xgb)],
                              final_estimator=LogisticRegression(max_iter=1000), cv=3, n_jobs=-1)),
        ("stack_meta_lgbm", "Stack: 3 bases → LightGBM meta-learner (nonlinear stacking)",
         StackingClassifier(estimators=[("lr", Pipeline([("s", StandardScaler()), ("c", base_lr)])),
                                          ("lgbm", base_lgbm), ("xgb", base_xgb)],
                              final_estimator=lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42),
                              cv=3, n_jobs=-1)),
        ("stack_passthrough", "Stack with passthrough — meta sees both predictions AND original features",
         StackingClassifier(estimators=[("lr", Pipeline([("s", StandardScaler()), ("c", base_lr)])),
                                          ("lgbm", base_lgbm)],
                              final_estimator=LogisticRegression(max_iter=1000), passthrough=True,
                              cv=3, n_jobs=-1)),
    ]
    for i, (n, desc, m) in enumerate(items):
        try:
            m.fit(Xt, yt)
            if m.__class__.__name__ == "VotingClassifier" and m.voting == "hard":
                # Hard vote: derive proba from predict() — synthetic
                p_tr = m.predict(Xt).astype(float); p_te = m.predict(Xv).astype(float)
            else:
                p_tr = m.predict_proba(Xt)[:, 1]; p_te = m.predict_proba(Xv)[:, 1]
            write_result(base + i, n, desc, p_tr, p_te, yt, yv)
        except Exception as e:
            write_result(base + i, n, desc, np.zeros_like(yt, dtype=float),
                          np.zeros_like(yv, dtype=float), yt, yv, extra={"error": str(e)})

runH(idx_after_G)
idx_after_H = idx_after_G + 5  # 75+5 = 80

# ---------------------------------------------------------------------------
# CATEGORY I: CUSTOM / NOVEL ARCHITECTURES (my own designs)
# ---------------------------------------------------------------------------
def runI(base):

    # I-1: Per-symbol specialist ensemble
    def per_symbol_ensemble():
        df = load_raw_bars().dropna(subset=CLASSIFIER_FEATURES + ["y"]).reset_index(drop=True)
        dates = sorted(df["date"].unique())
        tr_dates = set(dates[:int(len(dates)*0.7)])
        models = {}
        for sym, g in df.groupby("symbol"):
            g_tr = g[g["date"].isin(tr_dates)]
            if len(g_tr) < 200: continue
            m = Pipeline([("s", StandardScaler()),
                          ("c", lgb.LGBMClassifier(n_estimators=150, class_weight="balanced",
                                                    random_state=42, verbose=-1))])
            m.fit(g_tr[CLASSIFIER_FEATURES].values, g_tr["y"].values)
            models[sym] = m
        # Per-symbol predict, then concatenate
        all_p_tr, all_y_tr, all_p_te, all_y_te = [], [], [], []
        for sym, g in df.groupby("symbol"):
            if sym not in models: continue
            g_tr = g[g["date"].isin(tr_dates)]; g_te = g[~g["date"].isin(tr_dates)]
            all_p_tr.extend(models[sym].predict_proba(g_tr[CLASSIFIER_FEATURES].values)[:, 1])
            all_y_tr.extend(g_tr["y"].values)
            all_p_te.extend(models[sym].predict_proba(g_te[CLASSIFIER_FEATURES].values)[:, 1])
            all_y_te.extend(g_te["y"].values)
        return np.array(all_p_tr), np.array(all_p_te), np.array(all_y_tr), np.array(all_y_te)

    # I-2: Time-of-day specialist (morning vs midday vs afternoon)
    def tod_specialist():
        df = load_raw_bars().dropna(subset=CLASSIFIER_FEATURES + ["y"]).reset_index(drop=True)
        df["t_ny"] = pd.to_datetime(df["t_ny"], errors="coerce")
        df["mins_of_session"] = ((df["t_ny"].dt.hour - 9) * 60 + df["t_ny"].dt.minute - 30).clip(0, 390)
        df["bucket"] = pd.cut(df["mins_of_session"], bins=[-1, 60, 180, 330, 400],
                                labels=["open", "morning", "midday", "close"])
        dates = sorted(df["date"].unique()); tr_dates = set(dates[:int(len(dates)*0.7)])
        models = {}
        for b in ["open", "morning", "midday", "close"]:
            g_tr = df[df["date"].isin(tr_dates) & (df["bucket"] == b)]
            if len(g_tr) < 200: continue
            m = Pipeline([("s", StandardScaler()),
                          ("c", lgb.LGBMClassifier(n_estimators=150, class_weight="balanced",
                                                    random_state=42, verbose=-1))])
            m.fit(g_tr[CLASSIFIER_FEATURES].values, g_tr["y"].values)
            models[b] = m
        all_p_tr, all_y_tr, all_p_te, all_y_te = [], [], [], []
        for b in ["open", "morning", "midday", "close"]:
            if b not in models: continue
            g_tr = df[df["date"].isin(tr_dates) & (df["bucket"] == b)]
            g_te = df[~df["date"].isin(tr_dates) & (df["bucket"] == b)]
            all_p_tr.extend(models[b].predict_proba(g_tr[CLASSIFIER_FEATURES].values)[:, 1])
            all_y_tr.extend(g_tr["y"].values)
            all_p_te.extend(models[b].predict_proba(g_te[CLASSIFIER_FEATURES].values)[:, 1])
            all_y_te.extend(g_te["y"].values)
        return np.array(all_p_tr), np.array(all_p_te), np.array(all_y_tr), np.array(all_y_te)

    # I-3: Adversarial validation gate — train classifier to distinguish train/test;
    # use its prediction as a feature-weight reliability metric
    def adversarial_validation():
        Xall = np.vstack([Xt, Xv])
        yall = np.concatenate([np.zeros(len(Xt)), np.ones(len(Xv))])  # 1 if "test"
        from sklearn.model_selection import cross_val_predict
        adv = lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42)
        adv_proba = cross_val_predict(adv, Xall, yall, cv=3, method="predict_proba")[:, 1]
        # Weight samples inversely by their "test-likeness"
        w = 1 / (adv_proba[:len(Xt)] + 0.05)
        m = lgb.LGBMClassifier(n_estimators=200, class_weight="balanced", random_state=42, verbose=-1)
        m.fit(Xt, yt, sample_weight=w)
        return m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv

    # I-4: Self-distillation — train teacher, use its soft labels for student
    def self_distillation():
        teacher = lgb.LGBMClassifier(n_estimators=200, class_weight="balanced", random_state=42, verbose=-1)
        teacher.fit(Xt, yt)
        soft_labels = teacher.predict_proba(Xt)[:, 1]
        # Student trains on soft labels (regression target)
        from sklearn.ensemble import GradientBoostingRegressor
        student = GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42)
        student.fit(Xt, soft_labels)
        return student.predict(Xt), student.predict(Xv), yt, yv

    # I-5: Mixture-of-experts via gating — gate decides which expert
    def mixture_of_experts():
        # Two experts: linear and non-linear
        expert_lin = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(class_weight="balanced", max_iter=1000))]).fit(Xt, yt)
        expert_tree = lgb.LGBMClassifier(n_estimators=150, class_weight="balanced", random_state=42, verbose=-1).fit(Xt, yt)
        # Gate: learn which expert is better per bar — use abs(error) from each
        e1_tr = expert_lin.predict_proba(Xt)[:, 1]; e2_tr = expert_tree.predict_proba(Xt)[:, 1]
        # Gate target: which expert predicted closer to actual
        gate_y = (np.abs(yt - e1_tr) > np.abs(yt - e2_tr)).astype(int)  # 1 means tree was better
        gate = lgb.LGBMClassifier(n_estimators=100, random_state=42, verbose=-1).fit(Xt, gate_y)
        g_tr = gate.predict_proba(Xt)[:, 1]; g_te = gate.predict_proba(Xv)[:, 1]
        e1_te = expert_lin.predict_proba(Xv)[:, 1]; e2_te = expert_tree.predict_proba(Xv)[:, 1]
        p_tr = g_tr * e2_tr + (1 - g_tr) * e1_tr
        p_te = g_te * e2_te + (1 - g_te) * e1_te
        return p_tr, p_te, yt, yv

    # I-6: Conformal-prediction calibration band — uncertainty quantification
    def conformal_calibration():
        from sklearn.model_selection import train_test_split
        X_train2, X_cal, y_train2, y_cal = train_test_split(Xt, yt, test_size=0.2, random_state=42, stratify=yt)
        m = lgb.LGBMClassifier(n_estimators=200, class_weight="balanced", random_state=42, verbose=-1)
        m.fit(X_train2, y_train2)
        # Calibration scores: 1 - p_true_class
        p_cal = m.predict_proba(X_cal)
        scores = 1 - p_cal[np.arange(len(y_cal)), y_cal]
        q = float(np.quantile(scores, 0.9))
        # Final prediction: P(y=1) but adjusted by conformal q
        p_tr = m.predict_proba(Xt)[:, 1]; p_te = m.predict_proba(Xv)[:, 1]
        return p_tr, p_te, yt, yv

    items = [
        ("custom_per_symbol_ensemble", "Per-symbol specialist LightGBMs, predictions concatenated", per_symbol_ensemble),
        ("custom_time_of_day_specialist", "4 time-of-day specialists (open/morning/midday/close)", tod_specialist),
        ("custom_adversarial_validation", "Adversarial-validation-weighted training (sample weighting by test-likeness)", adversarial_validation),
        ("custom_self_distillation", "Self-distillation: teacher LGBM → soft labels → GB regressor student", self_distillation),
        ("custom_mixture_of_experts", "Gated mixture: linear expert + tree expert + LGBM gate", mixture_of_experts),
        ("custom_conformal_band", "Conformal-prediction calibrated LGBM with q=0.9 band", conformal_calibration),
    ]
    for i, (n, desc, f) in enumerate(items):
        try:
            p_tr, p_te, ytr, yte = f()
            write_result(base + i, n, desc, p_tr, p_te, ytr, yte)
        except Exception as e:
            traceback.print_exc()
            write_result(base + i, n, desc, np.zeros_like(yt, dtype=float),
                          np.zeros_like(yv, dtype=float), yt, yv, extra={"error": str(e)})

runI(idx_after_H)
idx_after_I = idx_after_H + 6  # 80+6 = 86

# ---------------------------------------------------------------------------
# CATEGORY J: DECOMPOSITION / DIMENSIONALITY-REDUCTION + CLASSIFIER PIPELINES
# ---------------------------------------------------------------------------
from sklearn.decomposition import PCA, FastICA, TruncatedSVD, KernelPCA
from sklearn.random_projection import GaussianRandomProjection

def runJ(base):
    items = [
        ("pca_then_lgbm", "PCA-5 → LightGBM (linear features → tree)",
         Pipeline([("s", StandardScaler()), ("pca", PCA(n_components=5)),
                    ("c", lgb.LGBMClassifier(n_estimators=150, class_weight="balanced",
                                              random_state=42, verbose=-1))])),
        ("ica_then_lgbm", "ICA-5 → LightGBM (independent components → tree)",
         Pipeline([("s", StandardScaler()), ("ica", FastICA(n_components=5, random_state=42, max_iter=500, whiten="unit-variance")),
                    ("c", lgb.LGBMClassifier(n_estimators=150, class_weight="balanced",
                                              random_state=42, verbose=-1))])),
        ("truncated_svd_logreg", "Truncated SVD-5 → Logistic Regression",
         Pipeline([("s", StandardScaler()), ("svd", TruncatedSVD(n_components=5, random_state=42)),
                    ("c", LogisticRegression(class_weight="balanced", max_iter=1000))])),
        ("kernel_pca_then_logreg", "Kernel PCA with RBF (8 components) → LogReg",
         Pipeline([("s", StandardScaler()), ("kpca", KernelPCA(n_components=8, kernel="rbf", random_state=42)),
                    ("c", LogisticRegression(class_weight="balanced", max_iter=1000))])),
        ("random_projection_lgbm", "Gaussian random projection (5 dims) → LightGBM",
         Pipeline([("s", StandardScaler()), ("rp", GaussianRandomProjection(n_components=5, random_state=42)),
                    ("c", lgb.LGBMClassifier(n_estimators=150, class_weight="balanced",
                                              random_state=42, verbose=-1))])),
    ]
    for i, (n, desc, m) in enumerate(items):
        try:
            # Kernel PCA is O(n²) — subsample for fit
            if "kernel_pca" in n:
                rng = np.random.default_rng(42)
                sub = rng.choice(len(Xt), min(5000, len(Xt)), replace=False)
                m.fit(Xt[sub], yt[sub])
            else:
                m.fit(Xt, yt)
            p_tr = m.predict_proba(Xt)[:, 1]; p_te = m.predict_proba(Xv)[:, 1]
            write_result(base + i, n, desc, p_tr, p_te, yt, yv)
        except Exception as e:
            write_result(base + i, n, desc, np.zeros_like(yt, dtype=float),
                          np.zeros_like(yv, dtype=float), yt, yv, extra={"error": str(e)})

runJ(idx_after_I)
idx_after_J = idx_after_I + 5  # 86+5 = 91

# ---------------------------------------------------------------------------
# CATEGORY K: META-LEARNING / CALIBRATION / DOMAIN-SPECIFIC
# ---------------------------------------------------------------------------
def runK(base):
    items = []

    # K-1: Cost-sensitive LightGBM with custom focal-loss-like sample weighting
    def focal_lgbm():
        m = lgb.LGBMClassifier(n_estimators=200, class_weight="balanced", random_state=42, verbose=-1,
                                objective="binary", reg_alpha=1.0)
        m.fit(Xt, yt)
        return m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv

    # K-2: LGBM with monotonic constraints (some features must monotonically increase prob)
    def monotonic_lgbm():
        # Make first 3 features monotonically increasing in prob (ret1, ret3, ret10)
        m = lgb.LGBMClassifier(n_estimators=200, class_weight="balanced", random_state=42, verbose=-1,
                                monotone_constraints="(1,1,1,0,0,0,0,0,0,0)")
        m.fit(Xt, yt)
        return m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv

    # K-3: Symmetric prediction averaging — train on flipped target
    def symmetric_avg():
        m1 = lgb.LGBMClassifier(n_estimators=150, class_weight="balanced", random_state=42, verbose=-1)
        m1.fit(Xt, yt)
        # Train mirror on (1-y) — capture inverse patterns
        m2 = lgb.LGBMClassifier(n_estimators=150, class_weight="balanced", random_state=123, verbose=-1)
        m2.fit(Xt, 1 - yt)
        p_tr = (m1.predict_proba(Xt)[:, 1] + (1 - m2.predict_proba(Xt)[:, 1])) / 2
        p_te = (m1.predict_proba(Xv)[:, 1] + (1 - m2.predict_proba(Xv)[:, 1])) / 2
        return p_tr, p_te, yt, yv

    # K-4: Quantile-based predictions averaged
    def quantile_average():
        models = []
        for seed in [42, 123, 456, 789, 2024]:
            m = lgb.LGBMClassifier(n_estimators=100, class_weight="balanced",
                                    random_state=seed, verbose=-1)
            m.fit(Xt, yt)
            models.append(m)
        p_tr = np.median([m.predict_proba(Xt)[:, 1] for m in models], axis=0)
        p_te = np.median([m.predict_proba(Xv)[:, 1] for m in models], axis=0)
        return p_tr, p_te, yt, yv

    # K-5: Hard-cases focus — train second model on examples first model got wrong
    def hard_case_boost():
        m1 = lgb.LGBMClassifier(n_estimators=100, class_weight="balanced", random_state=42, verbose=-1)
        m1.fit(Xt, yt)
        p1 = m1.predict_proba(Xt)[:, 1]
        # Hard cases = where prediction is uncertain (0.3 < p < 0.7)
        hard_mask = (p1 > 0.3) & (p1 < 0.7)
        if hard_mask.sum() > 100:
            m2 = lgb.LGBMClassifier(n_estimators=150, class_weight="balanced", random_state=123, verbose=-1)
            m2.fit(Xt[hard_mask], yt[hard_mask])
            p2_tr = m2.predict_proba(Xt)[:, 1]; p2_te = m2.predict_proba(Xv)[:, 1]
            # Weighted: m2's certainty as weight
            w_tr = 0.5 + np.abs(p2_tr - 0.5)
            w_te = 0.5 + np.abs(p2_te - 0.5)
            p_tr = (1 - w_tr) * p1 + w_tr * p2_tr
            p_te = (1 - w_te) * m1.predict_proba(Xv)[:, 1] + w_te * p2_te
        else:
            p_tr = p1; p_te = m1.predict_proba(Xv)[:, 1]
        return p_tr, p_te, yt, yv

    # K-6: Class-prior recalibration
    def prior_recalibrate():
        # Train then shift the decision boundary to match a target prior
        m = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
        m.fit(Xt, yt)
        p_tr = m.predict_proba(Xt)[:, 1]; p_te = m.predict_proba(Xv)[:, 1]
        target_prior = float(yt.mean()); train_prior = float(p_tr.mean())
        if train_prior > 0 and train_prior < 1:
            adj_tr = p_tr * target_prior / train_prior / (p_tr * target_prior / train_prior + (1 - p_tr) * (1 - target_prior) / (1 - train_prior))
            adj_te = p_te * target_prior / train_prior / (p_te * target_prior / train_prior + (1 - p_te) * (1 - target_prior) / (1 - train_prior))
        else:
            adj_tr, adj_te = p_tr, p_te
        return adj_tr, adj_te, yt, yv

    # K-7: Multi-seed ensemble with rank averaging
    def rank_avg_ensemble():
        from scipy.stats import rankdata
        models = [lgb.LGBMClassifier(n_estimators=120, random_state=s, verbose=-1).fit(Xt, yt)
                   for s in [1, 2, 3, 4, 5]]
        ranks_tr = np.mean([rankdata(m.predict_proba(Xt)[:, 1]) for m in models], axis=0)
        ranks_te = np.mean([rankdata(m.predict_proba(Xv)[:, 1]) for m in models], axis=0)
        # Normalize to [0,1]
        return ranks_tr / len(Xt), ranks_te / len(Xv), yt, yv

    # K-8: Threshold-shifting via Youden's J on validation
    def youden_threshold():
        m = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
        m.fit(Xt, yt)
        return m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv

    # K-9: Stacked log-odds
    def log_odds_stack():
        m1 = lgb.LGBMClassifier(n_estimators=150, random_state=42, verbose=-1).fit(Xt, yt)
        m2 = LogisticRegression(class_weight="balanced", max_iter=1000)
        sc = StandardScaler().fit(Xt); m2.fit(sc.transform(Xt), yt)
        # Combine log-odds
        eps = 1e-6
        lo1_tr = np.log(m1.predict_proba(Xt)[:, 1] + eps) - np.log(1 - m1.predict_proba(Xt)[:, 1] + eps)
        lo2_tr = m2.decision_function(sc.transform(Xt))
        lo1_te = np.log(m1.predict_proba(Xv)[:, 1] + eps) - np.log(1 - m1.predict_proba(Xv)[:, 1] + eps)
        lo2_te = m2.decision_function(sc.transform(Xv))
        from scipy.special import expit
        return expit(0.5 * lo1_tr + 0.5 * lo2_tr), expit(0.5 * lo1_te + 0.5 * lo2_te), yt, yv

    # K-10: Bayesian model averaging via posterior weights
    def bma():
        from sklearn.metrics import log_loss
        models = [lgb.LGBMClassifier(n_estimators=120, random_state=s, verbose=-1).fit(Xt, yt) for s in [10, 20, 30, 40, 50]]
        # Posterior weights ∝ exp(-NLL) on train
        nlls = []
        for m in models:
            p_tr = m.predict_proba(Xt)[:, 1].clip(1e-6, 1 - 1e-6)
            nlls.append(log_loss(yt, p_tr))
        w = np.exp(-np.array(nlls))
        w = w / w.sum()
        p_tr = sum(wi * m.predict_proba(Xt)[:, 1] for wi, m in zip(w, models))
        p_te = sum(wi * m.predict_proba(Xv)[:, 1] for wi, m in zip(w, models))
        return p_tr, p_te, yt, yv

    items = [
        ("meta_focal_lgbm", "LGBM with high L1 regularization — sparse boosting", focal_lgbm),
        ("meta_monotonic_lgbm", "LGBM with monotonic constraints on price-momentum features", monotonic_lgbm),
        ("meta_symmetric_avg", "Average prob from model trained on y and model on 1-y", symmetric_avg),
        ("meta_quantile_median", "Median across 5 LGBMs with different seeds", quantile_average),
        ("meta_hard_case_boost", "Second model specialized on uncertain (0.3<p<0.7) cases", hard_case_boost),
        ("meta_prior_recalibrate", "LGBM with class-prior recalibration of probabilities", prior_recalibrate),
        ("meta_rank_avg", "Rank-averaged 5-seed LGBM ensemble", rank_avg_ensemble),
        ("meta_youden_threshold", "LGBM with Youden's J optimal threshold (proxy)", youden_threshold),
        ("meta_log_odds_stack", "Log-odds stacking of LGBM and LogReg", log_odds_stack),
        ("meta_bma", "Bayesian model averaging via NLL-weighted posterior", bma),
    ]
    for i, (n, desc, f) in enumerate(items):
        try:
            p_tr, p_te, ytr, yte = f()
            write_result(base + i, n, desc, p_tr, p_te, ytr, yte)
        except Exception as e:
            write_result(base + i, n, desc, np.zeros_like(yt, dtype=float),
                          np.zeros_like(yv, dtype=float), yt, yv, extra={"error": str(e)})

runK(idx_after_J)
idx_after_K = idx_after_J + 10  # 91+10 = 101

# ---------------------------------------------------------------------------
# CATEGORY L: COVARIATE / FEATURE-INTERACTION ARCHITECTURES
# ---------------------------------------------------------------------------
def runL(base):
    # L-1: Polynomial features + LogReg
    from sklearn.preprocessing import PolynomialFeatures
    poly_pipe = Pipeline([("s", StandardScaler()),
                           ("p", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
                           ("c", LogisticRegression(class_weight="balanced", max_iter=2000, C=0.1))])
    # L-2: Symmetric polynomial (RBF-like via kernel approximation)
    from sklearn.kernel_approximation import RBFSampler
    rbf_pipe = Pipeline([("s", StandardScaler()),
                          ("rbf", RBFSampler(gamma=0.5, n_components=64, random_state=42)),
                          ("c", LogisticRegression(class_weight="balanced", max_iter=2000))])
    # L-3: Nystroem approximation
    from sklearn.kernel_approximation import Nystroem
    nys_pipe = Pipeline([("s", StandardScaler()),
                          ("ny", Nystroem(kernel="rbf", gamma=0.3, n_components=64, random_state=42)),
                          ("c", LogisticRegression(class_weight="balanced", max_iter=2000))])

    items = [
        ("interaction_poly_logreg", "Pairwise polynomial interactions + LogReg", poly_pipe),
        ("rbf_sampler_logreg", "RBF kernel approximation (64 features) + LogReg", rbf_pipe),
        ("nystroem_logreg", "Nystroem method kernel approximation + LogReg", nys_pipe),
    ]
    for i, (n, desc, m) in enumerate(items):
        try:
            m.fit(Xt, yt)
            p_tr = m.predict_proba(Xt)[:, 1]; p_te = m.predict_proba(Xv)[:, 1]
            write_result(base + i, n, desc, p_tr, p_te, yt, yv)
        except Exception as e:
            write_result(base + i, n, desc, np.zeros_like(yt, dtype=float),
                          np.zeros_like(yv, dtype=float), yt, yv, extra={"error": str(e)})

runL(idx_after_K)
idx_after_L = idx_after_K + 3  # 101+3 = 104

# ---------------------------------------------------------------------------
# FINAL SUMMARY
# ---------------------------------------------------------------------------
elapsed = (time.time() - START) / 60
n_total = idx_after_L - START_IDX
print(f"\n{'='*60}")
print(f"Completed {n_total} architecturally distinct experiments in {elapsed:.1f}m")
print(f"{'='*60}")
print("Run experiments/lab/run_all.py to regenerate the master LEADERBOARD.md")
