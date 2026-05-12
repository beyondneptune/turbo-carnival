"""Custom: Cascade-Reject — sequential filter cascade with progressive complexity.

My own design. Inspired by Viola-Jones face detection cascades. Three stages:
  Stage 1: GaussianNB (cheap, removes obvious non-events at high recall)
  Stage 2: LogReg on Stage-1 survivors (linear refinement)
  Stage 3: LightGBM on Stage-2 survivors (non-linear, focused final)

Each stage trains on the SURVIVORS of the previous (above its cutoff). The
novelty: model capacity is concentrated on the hard cases that earlier
stages couldn't classify confidently. Classic boosting-by-rejection.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()

# Stage 1: GaussianNB. Target: keep top 50% by predicted-probability.
s1 = Pipeline([("s", StandardScaler()), ("c", GaussianNB())])
s1.fit(Xt, yt)
p1_tr = s1.predict_proba(Xt)[:, 1]
p1_te = s1.predict_proba(Xv)[:, 1]
cut1 = float(np.quantile(p1_tr, 0.50))
keep1_tr = p1_tr >= cut1
keep1_te = p1_te >= cut1
print(f"Stage 1: kept {keep1_tr.sum():,} / {len(Xt):,} ({100*keep1_tr.mean():.1f}%) — cutoff {cut1:.3f}")

# Stage 2: LogReg on Stage-1 survivors
s2 = Pipeline([("s", StandardScaler()),
                ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])
s2.fit(Xt[keep1_tr], yt[keep1_tr])
p2_tr = np.full(len(Xt), 0.0)
p2_te = np.full(len(Xv), 0.0)
p2_tr[keep1_tr] = s2.predict_proba(Xt[keep1_tr])[:, 1]
p2_te[keep1_te] = s2.predict_proba(Xv[keep1_te])[:, 1]
cut2 = float(np.quantile(p2_tr[keep1_tr], 0.30))  # keep top 70% of stage-1 survivors
keep2_tr = keep1_tr & (p2_tr >= cut2)
keep2_te = keep1_te & (p2_te >= cut2)
print(f"Stage 2: kept {keep2_tr.sum():,} / {keep1_tr.sum():,} survivors of S1 — cutoff {cut2:.3f}")

# Stage 3: LightGBM on Stage-2 survivors. Final prediction.
s3 = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                          class_weight="balanced", random_state=42, verbose=-1)
s3.fit(Xt[keep2_tr], yt[keep2_tr])
p3_tr = np.full(len(Xt), 0.0)
p3_te = np.full(len(Xv), 0.0)
p3_tr[keep2_tr] = s3.predict_proba(Xt[keep2_tr])[:, 1]
p3_te[keep2_te] = s3.predict_proba(Xv[keep2_te])[:, 1]
print(f"Stage 3: trained on {keep2_tr.sum():,} hard-case bars")

# Final probability = stage-3 output for survivors, stage-2 for stage-1-survivors-stage-2-rejects, stage-1 for stage-1-rejects
final_tr = np.where(keep2_tr, p3_tr, np.where(keep1_tr, p2_tr, p1_tr))
final_te = np.where(keep2_te, p3_te, np.where(keep1_te, p2_te, p1_te))

report("028_cascade_reject",
        "Custom: 3-stage cascade (NB→LR→LGBM), each trained on prior-stage survivors",
        evaluate_proba(final_tr, final_te, yt, yv),
        extra={"stage1_survivors_pct": float(keep1_tr.mean() * 100),
               "stage2_survivors_pct": float(keep2_tr.mean() * 100),
               "stage1_cutoff": cut1,
               "stage2_cutoff": cut2})
