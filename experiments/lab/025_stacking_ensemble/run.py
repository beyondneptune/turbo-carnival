"""Stacking Ensemble — top 3 base classifiers fed into a logistic meta-learner.

Bases: LogReg + LightGBM + HistGBM. Meta: LogReg on out-of-fold predictions.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()

base = [
    ("logreg", Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])),
    ("lgbm", lgb.LGBMClassifier(n_estimators=200, max_depth=5, class_weight="balanced",
                                  random_state=42, verbose=-1)),
    ("hgbm", HistGradientBoostingClassifier(max_iter=200, max_depth=5,
                                              class_weight="balanced", random_state=42)),
]
meta = LogisticRegression(max_iter=1000)
m = StackingClassifier(estimators=base, final_estimator=meta, cv=3, n_jobs=-1)
m.fit(Xt, yt)
report("025_stacking_ensemble",
        "Stack: LogReg + LightGBM + HistGBM, meta: LogReg on 3-fold OOF preds",
        evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv))
