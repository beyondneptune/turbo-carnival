"""One-Class SVM — boundary-based anomaly, RBF kernel, trained on subsample."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from shared.data import load_classifier_data
from shared.eval import evaluate_anomaly, report

Xt, yt, Xv, yv = load_classifier_data()
sc = StandardScaler().fit(Xt)
Xt_s = sc.transform(Xt)
Xv_s = sc.transform(Xv)
# Subsample to keep training feasible (OCSVM is O(n^2))
rng = np.random.default_rng(42)
idx = rng.choice(len(Xt_s), size=min(5000, len(Xt_s)), replace=False)
m = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
m.fit(Xt_s[idx])
sc_train = -m.score_samples(Xt_s)
sc_test = -m.score_samples(Xv_s)
report("017_one_class_svm", "OCSVM RBF kernel, trained on 5k subsample, nu=0.1",
        evaluate_anomaly(sc_train, sc_test, yt, yv))
