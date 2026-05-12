"""Snapshot Ensemble MLP — one network, 3 snapshots at different epochs, ensemble predictions.

Architecture: A single MLP trained for 150 epochs via partial_fit, with model parameters
saved at epochs 50, 100, and 150. At inference, each snapshot generates predictions
independently, then predictions are averaged to form the ensemble output.

Novelty: Unlike training 5 independent MLPs (e.g., exp 125), snapshots share the same
training trajectory through weight space, creating a novel form of ensemble diversity.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

# Load data
Xt, yt, Xv, yv = load_classifier_data()

# Fit scaler on training data
sc = StandardScaler()
Xt_scaled = sc.fit_transform(Xt)
Xv_scaled = sc.transform(Xv)

# Create MLP with warm_start=True for incremental training
m = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    random_state=42,
    max_iter=1,
    warm_start=True,
    early_stopping=False,
)

# Train for 150 epochs via partial_fit, saving snapshots at epochs 50, 100, 150
snapshots = []
for epoch in range(150):
    m.partial_fit(Xt_scaled, yt, classes=[0, 1])
    if (epoch + 1) in (50, 100, 150):
        # Save model parameters as a snapshot
        snapshot = {
            'coefs': [c.copy() for c in m.coefs_],
            'intercepts': [b.copy() for b in m.intercepts_],
            'epoch': epoch + 1,
        }
        snapshots.append(snapshot)

print(f"Saved {len(snapshots)} snapshots at epochs {[s['epoch'] for s in snapshots]}")


def predict_from_snapshot(snap, X):
    """Restore a snapshot into a temporary model and predict probabilities."""
    m_temp = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        random_state=42,
    )
    # Manually set the fitted attributes
    m_temp.coefs_ = snap['coefs']
    m_temp.intercepts_ = snap['intercepts']
    m_temp.n_layers_ = len(snap['coefs']) + 1
    m_temp.classes_ = np.array([0, 1])
    m_temp.n_outputs_ = 1
    m_temp.out_activation_ = 'logistic'

    return m_temp.predict_proba(X)[:, 1]


# Generate ensemble predictions by averaging across all snapshots
p_train_snapshots = [predict_from_snapshot(s, Xt_scaled) for s in snapshots]
p_test_snapshots = [predict_from_snapshot(s, Xv_scaled) for s in snapshots]

p_train = np.mean(p_train_snapshots, axis=0)
p_test = np.mean(p_test_snapshots, axis=0)

# Evaluate and report
report(
    "137_snapshot_mlp_ensemble",
    "single MLP trained for 150 epochs; snapshots at epochs 50/100/150; averaged predictions",
    evaluate_proba(p_train, p_test, yt, yv)
)
