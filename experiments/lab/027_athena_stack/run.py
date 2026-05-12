"""Custom: ATHENA-Stack — feature-decorrelated heterogeneous confluence.

My own design. Three base predictors trained on DISJOINT feature subsets:
  - LogReg on price-momentum features (linear, fast, low variance)
  - LightGBM on volume/range features (non-linear, captures interactions)
  - MLP on time-context features (relu network, handles cyclic features well)

Then 2-of-3 confluence vote: probability is geometric mean of base preds.
The novelty: forced feature decorrelation across base learners reduces
correlated errors that pure ensemble averaging can't escape.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.data import CLASSIFIER_FEATURES, load_classifier_data
from shared.eval import evaluate_proba, report

# Feature partitions
FEATS = CLASSIFIER_FEATURES  # ret1, ret3, ret10, range_pct, volume_z, ret1_z, vwap_dist, tod_sin, tod_cos, vov
PRICE_IDX = [0, 1, 2, 5]  # ret1, ret3, ret10, ret1_z
VOLUME_IDX = [3, 4, 9]    # range_pct, volume_z, vov
TIME_IDX = [6, 7, 8]      # vwap_dist, tod_sin, tod_cos

Xt, yt, Xv, yv = load_classifier_data()
Xt_p, Xv_p = Xt[:, PRICE_IDX], Xv[:, PRICE_IDX]
Xt_v, Xv_v = Xt[:, VOLUME_IDX], Xv[:, VOLUME_IDX]
Xt_tm, Xv_tm = Xt[:, TIME_IDX], Xv[:, TIME_IDX]

# Base 1: LogReg on price features
b1 = Pipeline([("s", StandardScaler()),
                ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])
b1.fit(Xt_p, yt)
p1_tr = b1.predict_proba(Xt_p)[:, 1]
p1_te = b1.predict_proba(Xv_p)[:, 1]

# Base 2: LightGBM on volume features
b2 = lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                          class_weight="balanced", random_state=42, verbose=-1)
b2.fit(Xt_v, yt)
p2_tr = b2.predict_proba(Xt_v)[:, 1]
p2_te = b2.predict_proba(Xv_v)[:, 1]

# Base 3: MLP on time-context features
b3 = Pipeline([("s", StandardScaler()),
                ("c", MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=200,
                                     random_state=42, early_stopping=True))])
b3.fit(Xt_tm, yt)
p3_tr = b3.predict_proba(Xt_tm)[:, 1]
p3_te = b3.predict_proba(Xv_tm)[:, 1]

# Geometric mean confluence — penalizes any single base disagreement
eps = 1e-6
def geom_mean(a, b, c):
    return np.exp((np.log(a + eps) + np.log(b + eps) + np.log(c + eps)) / 3)

p_tr = geom_mean(p1_tr, p2_tr, p3_tr)
p_te = geom_mean(p1_te, p2_te, p3_te)

# Verify base correlations
corr_train = np.corrcoef([p1_tr, p2_tr, p3_tr])
print(f"Base predictor correlations on train:")
print(f"  price↔volume:  {corr_train[0,1]:.3f}")
print(f"  price↔time:    {corr_train[0,2]:.3f}")
print(f"  volume↔time:   {corr_train[1,2]:.3f}")

report("027_athena_stack",
        "Custom: 3 bases on disjoint features (price/volume/time), geometric mean confluence",
        evaluate_proba(p_tr, p_te, yt, yv),
        extra={"base_correlations": corr_train.tolist(),
               "base1_auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(yv, p1_te)),
               "base2_auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(yv, p2_te)),
               "base3_auc": float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(yv, p3_te))})
