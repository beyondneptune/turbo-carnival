"""HistGradientBoosting — histogram-based, faster than vanilla GBM on big tabular data."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import HistGradientBoostingClassifier
from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()
m = HistGradientBoostingClassifier(max_iter=300, max_depth=6, learning_rate=0.05,
                                    l2_regularization=0.5, class_weight="balanced",
                                    random_state=42)
m.fit(Xt, yt)
report("006_hist_gbm", "histogram-binned GBM, 300 iters, L2=0.5",
        evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv))
