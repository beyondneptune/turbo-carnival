"""XGBoost — level-wise tree growth, second-order gradients, hist-based splitting."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import xgboost as xgb
from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report

Xt, yt, Xv, yv = load_classifier_data()
# Compute scale_pos_weight for class imbalance
pos = max(1, int(yt.sum())); neg = len(yt) - pos
m = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                       scale_pos_weight=neg / pos, tree_method="hist",
                       random_state=42, n_jobs=-1, verbosity=0)
m.fit(Xt, yt)
report("008_xgboost", "XGBoost hist, 300 trees, depth-6, scale_pos_weight balanced",
        evaluate_proba(m.predict_proba(Xt)[:, 1], m.predict_proba(Xv)[:, 1], yt, yv))
