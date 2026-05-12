"""Hidden Markov Model — Gaussian emissions, 3 hidden states (regimes)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from shared.data import load_classifier_data
from shared.eval import evaluate_anomaly, report

Xt, yt, Xv, yv = load_classifier_data()
sc = StandardScaler().fit(Xt)
Xt_s = sc.transform(Xt)
Xv_s = sc.transform(Xv)
try:
    m = hmm.GaussianHMM(n_components=3, covariance_type="diag",
                         n_iter=100, random_state=42)
    m.fit(Xt_s)
    # Anomaly score: -log-likelihood per bar (low likelihood = anomalous)
    sc_train = -m.score_samples(Xt_s)[0] if hasattr(m, "score_samples") else \
                -np.array([m.score(x.reshape(1, -1)) for x in Xt_s])
    sc_test = -np.array([m.score(x.reshape(1, -1)) for x in Xv_s])
    sc_train = np.array([m.score(x.reshape(1, -1)) for x in Xt_s])
    sc_test_arr = np.array([m.score(x.reshape(1, -1)) for x in Xv_s])
    # Negate so high = anomalous
    sc_train = -sc_train
    sc_test = -sc_test_arr
    report("020_hmm_regime", "3-state Gaussian HMM with diagonal covariance",
            evaluate_anomaly(sc_train, sc_test, yt, yv))
except Exception as e:
    report("020_hmm_regime", "FAILED",
            {"error": str(e), "passes_overfit_gate": False, "test_auc": 0.5})
