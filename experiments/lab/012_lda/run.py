"""Linear Discriminant Analysis — generative linear classifier with shared covariance."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()
m = Pipeline([("s", StandardScaler()), ("c", LinearDiscriminantAnalysis())])
m.fit(Xt, yt)
report("012_lda", "LDA — Bayes-optimal linear under Gaussian-with-shared-cov assumption",
        evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv))
