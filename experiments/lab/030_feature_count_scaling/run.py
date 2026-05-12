"""Feature Count Scaling — when does adding more features stop helping?

Companion to 026 (NN size scaling). Trains the SAME model (LogReg) with
N features from 1 to 10, where N is selected by permutation importance
on the full model. Plots test AUC vs N.

Answers: is there an information ceiling per feature added? Or do gains
keep coming as we add features?
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from shared.data import CLASSIFIER_FEATURES, load_classifier_data

Xt, yt, Xv, yv = load_classifier_data()

# First fit on all features to get permutation importance ranking
full = Pipeline([("s", StandardScaler()),
                 ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])
full.fit(Xt, yt)
imp = permutation_importance(full, Xv, yv, n_repeats=3, random_state=42, scoring="roc_auc", n_jobs=-1)
order = np.argsort(-imp.importances_mean)
feat_ranking = [CLASSIFIER_FEATURES[i] for i in order]
print("Feature ranking (most → least useful):")
for i, f in enumerate(feat_ranking, 1):
    print(f"  {i:>2}. {f}  (importance {imp.importances_mean[order[i-1]]:+.4f})")

rows = []
for k in range(1, len(CLASSIFIER_FEATURES) + 1):
    idx = order[:k]
    Xt_k, Xv_k = Xt[:, idx], Xv[:, idx]
    m = Pipeline([("s", StandardScaler()),
                  ("c", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    m.fit(Xt_k, yt)
    a_tr = float(roc_auc_score(yt, m.predict_proba(Xt_k)[:, 1]))
    a_te = float(roc_auc_score(yv, m.predict_proba(Xv_k)[:, 1]))
    rows.append({"n_features": k, "features": [CLASSIFIER_FEATURES[i] for i in idx],
                 "train_auc": a_tr, "test_auc": a_te, "train_test_gap": a_tr - a_te})
    print(f"  k={k:>2}  test_auc={a_te:.3f}  gap={a_tr - a_te:+.3f}  "
          f"add: {CLASSIFIER_FEATURES[idx[-1]]}")

best = max(rows, key=lambda r: r["test_auc"])
# Saturation = smallest k within 1% of best
sat_k = next((r["n_features"] for r in rows if r["test_auc"] >= best["test_auc"] - 0.01), best["n_features"])

out_path = Path(sys.argv[0]).resolve().parent / "results.json"
out_path.write_text(json.dumps({
    "experiment": "030_feature_count_scaling",
    "description": "LogReg with feature subset size 1→10 added in importance-rank order",
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "test_auc": best["test_auc"],
    "train_test_gap": best["train_test_gap"],
    "passes_overfit_gate": bool(best["test_auc"] > 0.55 and abs(best["train_test_gap"]) < 0.05),
    "best_k": best["n_features"],
    "saturation_k": sat_k,
    "feature_ranking": feat_ranking,
    "sweep": rows,
}, indent=2))
print(f"\n  saturation at k={sat_k} features (test AUC plateaus)")
print(f"[030_feature_count_scaling]  test_auc={best['test_auc']:.3f}  saturation_k={sat_k}")
