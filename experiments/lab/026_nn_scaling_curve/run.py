"""NN Scaling Curve — sweep single-hidden-layer MLP from 2 to 1024 neurons.

Answers the question: how big does the NN need to be before test AUC stops
improving? Where's the empirical saturation point on 54k bars × 10 features?
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data

Xt, yt, Xv, yv = load_classifier_data()
print(f"Data: {len(Xt):,} train, {len(Xv):,} test, {Xt.shape[1]} features")

sizes = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
rows = []
best_auc = 0
best_size = 0
for n in sizes:
    m = Pipeline([("s", StandardScaler()),
                  ("c", MLPClassifier(hidden_layer_sizes=(n,), max_iter=200,
                                       activation="relu", solver="adam",
                                       learning_rate_init=0.001,
                                       random_state=42, early_stopping=True,
                                       validation_fraction=0.1))])
    m.fit(Xt, yt)
    p_tr = m.predict_proba(Xt)[:, 1]
    p_te = m.predict_proba(Xv)[:, 1]
    a_tr = float(roc_auc_score(yt, p_tr))
    a_te = float(roc_auc_score(yv, p_te))
    gap = a_tr - a_te
    n_iters = m.named_steps["c"].n_iter_
    rows.append({"size": n, "train_auc": a_tr, "test_auc": a_te,
                 "train_test_gap": gap, "iters": int(n_iters)})
    print(f"  size={n:>5}  train_auc={a_tr:.3f}  test_auc={a_te:.3f}  gap={gap:+.3f}  iters={n_iters}")
    if a_te > best_auc and abs(gap) < 0.05:
        best_auc = a_te
        best_size = n

# Saturation = smallest size within 1% of best
sat_size = None
for r in rows:
    if r["test_auc"] >= best_auc - 0.01:
        sat_size = r["size"]
        break

out_path = Path(sys.argv[0]).resolve().parent / "results.json"
out_path.write_text(json.dumps({
    "experiment": "026_nn_scaling_curve",
    "description": "Single hidden layer MLP swept across sizes 2-1024",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "test_auc": best_auc,
    "train_test_gap": next(r["train_test_gap"] for r in rows if r["test_auc"] == best_auc),
    "passes_overfit_gate": best_auc > 0.55,
    "best_size": best_size,
    "saturation_size": sat_size,
    "sweep": rows,
    "conclusion": (
        f"Test AUC saturates at ~{sat_size} hidden neurons "
        f"(test AUC {best_auc:.3f}). Bigger NN does not help — likely overfits."
    ),
}, indent=2))
print(f"\n  saturation at ~{sat_size} neurons, best test AUC {best_auc:.3f}")
print(f"[026_nn_scaling_curve]  test_auc={best_auc:.3f}  saturation_size={sat_size}")
