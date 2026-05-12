"""Elliptic Envelope — robust Gaussian envelope, Mahalanobis distance to centroid."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from shared.data import load_classifier_data
from shared.eval import evaluate_anomaly, report

Xt, yt, Xv, yv = load_classifier_data()
sc = StandardScaler().fit(Xt)
Xt_s = sc.transform(Xt)
Xv_s = sc.transform(Xv)
try:
    m = EllipticEnvelope(contamination=0.1, support_fraction=0.9, random_state=42)
    m.fit(Xt_s)
    sc_train = -m.score_samples(Xt_s)
    sc_test = -m.score_samples(Xv_s)
    report("018_elliptic_envelope", "robust Gaussian envelope, contamination=0.1",
            evaluate_anomaly(sc_train, sc_test, yt, yv))
except Exception as e:
    report("018_elliptic_envelope", "FAILED — likely singular covariance",
            {"error": str(e), "passes_overfit_gate": False, "test_auc": 0.5})
