"""Random Forest — bagged decision trees with feature subsampling."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.ensemble import RandomForestClassifier
from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()
m = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                            random_state=42, n_jobs=-1)
m.fit(Xt, yt)
report("002_random_forest", "300 bagged trees, depth-8, sqrt(n_features) per split",
        evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv))
