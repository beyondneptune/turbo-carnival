"""
MLP with very strong L2 regularization (alpha=1.0) as dropout-proxy + soft confidence ensemble.

Novel architecture combining two ideas:
1. Heavy L2 regularization (alpha=1.0, vs default 0.0001) as a dropout proxy to heavily constrain
   weight magnitudes and reduce overfitting, functionally similar to dropout in sklearn.
2. Seed-averaged ensemble: train 5 different random_state initializations (seeds 1-5) and average
   their probability predictions to get an implicit ensemble effect.

This differs from existing MLPs (014, 067-074) which use small or default alpha values.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()

# Train 5 strongly-regularized MLPs with different random seeds
models = []
for seed in [1, 2, 3, 4, 5]:
    m = Pipeline([
        ('s', StandardScaler()),
        ('c', MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=1.0,  # VERY STRONG L2 regularization (vs default 0.0001)
            activation='relu',
            solver='adam',
            max_iter=200,
            random_state=seed,
            early_stopping=True,
            validation_fraction=0.1
        ))
    ])
    m.fit(Xt, yt)
    models.append(m)

# Average predictions across all 5 seeds for soft ensemble effect
p_train_list = [m.predict_proba(Xt)[:, 1] for m in models]
p_test_list = [m.predict_proba(Xv)[:, 1] for m in models]

p_train = np.mean(p_train_list, axis=0)
p_test = np.mean(p_test_list, axis=0)

report(
    "125_mlp_strong_l2_regularization",
    "5-seed ensemble of MLPs (64-32) with alpha=1.0 (very strong L2 as dropout-proxy)",
    evaluate_proba(p_train, p_test, yt, yv)
)
