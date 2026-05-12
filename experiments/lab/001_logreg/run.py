"""Logistic Regression — linear baseline. Architecture: linear decision boundary."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()
m = Pipeline([("s", StandardScaler()), ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])
m.fit(Xt, yt)
report("001_logreg", "linear decision boundary, scaled features, balanced class weight",
        evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv))
