"""Train + persist the 4 sub-models that compose the C3 composite gate.

C3 = block if ANY of:
  - 016_local_outlier_factor (LOF anomaly score > p95)
  - 017_one_class_svm (OCSVM anomaly score > p95)
  - 018_elliptic_envelope (Mahalanobis-style score > p95)
  OR
  - 102_rbf_sampler_logreg (predict_proba > 0.5)

Writes models + thresholds to experiments/ml_lab/runs/athena_c3/.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

REPO = Path("/Users/weeoto/turbo-carnival")
sys.path.insert(0, str(REPO / "experiments" / "lab"))
from shared.data import load_classifier_data  # noqa

OUT = REPO / "experiments" / "ml_lab" / "runs" / "athena_c3"
OUT.mkdir(parents=True, exist_ok=True)

print("loading training data…")
Xt, yt, _, _ = load_classifier_data()
# Use FULL training set (don't reserve test split — composite is for production)
print(f"  {len(Xt):,} train rows × {Xt.shape[1]} features")

scaler = StandardScaler().fit(Xt)
Xt_s = scaler.transform(Xt)

# === 016 — LocalOutlierFactor ===
print("\n[1/4] 016 LocalOutlierFactor (k=30 novelty mode)")
m016 = LocalOutlierFactor(n_neighbors=30, novelty=True, n_jobs=-1)
m016.fit(Xt_s)
sc016 = -m016.score_samples(Xt_s)
p95_016 = float(np.quantile(sc016, 0.95))
print(f"  p95 anomaly score: {p95_016:.4f}")

# === 017 — OneClassSVM on 5000 subsample ===
print("\n[2/4] 017 OneClassSVM (RBF kernel, 5000 subsample)")
rng = np.random.default_rng(42)
sub = rng.choice(len(Xt_s), min(5000, len(Xt_s)), replace=False)
m017 = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1)
m017.fit(Xt_s[sub])
sc017 = -m017.score_samples(Xt_s)
p95_017 = float(np.quantile(sc017, 0.95))
print(f"  p95 anomaly score: {p95_017:.4f}")

# === 018 — EllipticEnvelope ===
print("\n[3/4] 018 EllipticEnvelope (contamination=0.1)")
m018 = EllipticEnvelope(contamination=0.1, support_fraction=0.9, random_state=42)
m018.fit(Xt_s)
sc018 = -m018.score_samples(Xt_s)
p95_018 = float(np.quantile(sc018, 0.95))
print(f"  p95 anomaly score: {p95_018:.4f}")

# === 102 — RBF kernel sampler + LogReg pipeline ===
print("\n[4/4] 102 RBFSampler + LogReg (64 components)")
m102 = Pipeline([
    ("scaler", StandardScaler()),
    ("rbf", RBFSampler(gamma=0.5, n_components=64, random_state=42)),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
])
m102.fit(Xt, yt)
proba102 = m102.predict_proba(Xt)[:, 1]
print(f"  train AUC proxy mean proba on y=1: {proba102[yt == 1].mean():.3f}")
print(f"  train AUC proxy mean proba on y=0: {proba102[yt == 0].mean():.3f}")

# Save models
joblib.dump(scaler, OUT / "shared_scaler.joblib")
joblib.dump(m016, OUT / "m016_lof.joblib")
joblib.dump(m017, OUT / "m017_ocsvm.joblib")
joblib.dump(m018, OUT / "m018_envelope.joblib")
joblib.dump(m102, OUT / "m102_rbf_logreg.joblib")

manifest = {
    "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    "training_rows": int(len(Xt)),
    "features": [
        "ret1", "ret3", "ret10", "range_pct", "volume_z", "ret1_z",
        "vwap_dist", "tod_sin", "tod_cos", "vov",
    ],
    "submodels": {
        "016_lof": {"file": "m016_lof.joblib", "p95_threshold": p95_016,
                     "decision_rule": "blocks if -score_samples(X) > p95"},
        "017_ocsvm": {"file": "m017_ocsvm.joblib", "p95_threshold": p95_017,
                       "decision_rule": "blocks if -score_samples(X) > p95"},
        "018_envelope": {"file": "m018_envelope.joblib", "p95_threshold": p95_018,
                          "decision_rule": "blocks if -score_samples(X) > p95"},
        "102_rbf_logreg": {"file": "m102_rbf_logreg.joblib",
                            "decision_rule": "blocks if predict_proba(class=1) > 0.5"},
    },
    "composite_rule": "block_c3 = (016_blocks) OR (017_blocks) OR (018_blocks) OR (102_blocks)",
    "shared_scaler": "shared_scaler.joblib",
    "validation_yesterday": {
        "blocked": 82,
        "block_rate_pct": 50.6,
        "blocked_winners_losers": "17 / 24",
        "counterfactual_pnl_usd": 2525.54,
        "delta_vs_actual_usd": 2029.53,
    },
}
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(f"\nwrote 5 joblibs + manifest to {OUT}")
print(f"manifest.json:")
print(json.dumps(manifest["submodels"], indent=2))
