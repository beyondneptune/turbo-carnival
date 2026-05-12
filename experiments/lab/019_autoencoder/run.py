"""Autoencoder — unsupervised reconstruction; high error = anomaly. 180-dim windows."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from shared.data import load_window_data
from shared.eval import evaluate_anomaly, report

Xt, yt, Xv, yv = load_window_data()
# Train only on non-anomalous windows
fit_mask = yt == 0
sc = StandardScaler().fit(Xt[fit_mask])
Xt_s = sc.transform(Xt)
Xv_s = sc.transform(Xv)
bottleneck = max(4, Xt.shape[1] // 4)
m = MLPRegressor(hidden_layer_sizes=(Xt.shape[1] // 2, bottleneck, Xt.shape[1] // 2),
                  activation="relu", solver="adam", max_iter=200, learning_rate_init=0.001,
                  random_state=42, early_stopping=True, validation_fraction=0.1)
m.fit(sc.transform(Xt[fit_mask]), sc.transform(Xt[fit_mask]))
err_train = ((m.predict(Xt_s) - Xt_s) ** 2).mean(axis=1)
err_test = ((m.predict(Xv_s) - Xv_s) ** 2).mean(axis=1)
report("019_autoencoder", f"MLP-AE 180->90->{bottleneck}->90->180, trained on normal windows only",
        evaluate_anomaly(err_train, err_test, yt, yv))
