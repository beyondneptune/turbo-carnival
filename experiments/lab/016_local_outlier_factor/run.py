"""Local Outlier Factor — density-based anomaly, comparing local vs neighborhood density."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from shared.data import load_classifier_data
from shared.eval import evaluate_anomaly, report

Xt, yt, Xv, yv = load_classifier_data()
sc = StandardScaler().fit(Xt)
Xt_s = sc.transform(Xt)
Xv_s = sc.transform(Xv)
m = LocalOutlierFactor(n_neighbors=30, novelty=True, n_jobs=-1)
m.fit(Xt_s)
sc_train = -m.score_samples(Xt_s)
sc_test = -m.score_samples(Xv_s)
report("016_local_outlier_factor", "LOF k=30 novelty mode, scaled features",
        evaluate_anomaly(sc_train, sc_test, yt, yv))
