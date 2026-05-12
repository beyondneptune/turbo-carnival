"""AdaBoost — boosted decision stumps. Reweights misclassified examples each round."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import AdaBoostClassifier
from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()
m = AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=42)
m.fit(Xt, yt)
report("004_adaboost", "200 boosted stumps (depth-1), AdaBoost.R2",
        evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv))
