"""Gaussian Naive Bayes — probabilistic, assumes feature independence + Gaussian likelihood."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()
m = Pipeline([("s", StandardScaler()), ("c", GaussianNB())])
m.fit(Xt, yt)
report("011_gaussian_nb", "Gaussian NB — independent Gaussian likelihoods per feature",
        evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv))
