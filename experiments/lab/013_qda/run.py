"""Quadratic Discriminant Analysis — per-class covariance, quadratic boundary."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()
m = Pipeline([("s", StandardScaler()), ("c", QuadraticDiscriminantAnalysis(reg_param=0.1))])
m.fit(Xt, yt)
report("013_qda", "QDA — per-class covariance with 0.1 shrinkage regularization",
        evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv))
