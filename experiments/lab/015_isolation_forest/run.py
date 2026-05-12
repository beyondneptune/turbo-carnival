"""Isolation Forest — anomaly detection via random tree path lengths."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.ensemble import IsolationForest
from shared.data import load_classifier_data
from shared.eval import evaluate_anomaly, report

Xt, yt, Xv, yv = load_classifier_data()
m = IsolationForest(n_estimators=300, contamination="auto", random_state=42, n_jobs=-1)
m.fit(Xt)
# Higher score_samples = more "normal"; invert so high = anomalous
sc_train = -m.score_samples(Xt)
sc_test = -m.score_samples(Xv)
report("015_isolation_forest", "300 isolation trees, contamination=auto",
        evaluate_anomaly(sc_train, sc_test, yt, yv))
