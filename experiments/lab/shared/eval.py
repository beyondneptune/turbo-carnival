"""Common evaluation harness. All experiments call `report` to produce
standardized JSON output so the leaderboard is apples-to-apples.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import brier_score_loss, roc_auc_score


def evaluate_proba(p_train: np.ndarray, p_test: np.ndarray,
                    y_train: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    """Standard supervised classifier metrics."""
    out: dict[str, Any] = {}
    out["train_auc"] = float(roc_auc_score(y_train, p_train))
    out["test_auc"] = float(roc_auc_score(y_test, p_test))
    out["train_test_gap"] = out["train_auc"] - out["test_auc"]
    out["test_brier"] = float(brier_score_loss(y_test, p_test))
    cut = float(np.quantile(p_test, 0.99))
    flagged = p_test >= cut
    out["precision_at_top1pct"] = float(y_test[flagged].mean()) if flagged.sum() > 0 else 0.0
    out["flagged_count"] = int(flagged.sum())
    overfit_ok = abs(out["train_test_gap"]) < 0.05
    has_signal = out["test_auc"] > 0.55
    out["passes_overfit_gate"] = bool(overfit_ok and has_signal)
    return out


def evaluate_anomaly(score_train: np.ndarray, score_test: np.ndarray,
                      y_train: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    """Unsupervised anomaly score → AUC vs label."""
    out: dict[str, Any] = {}
    try:
        out["train_auc"] = float(roc_auc_score(y_train, score_train))
        out["test_auc"] = float(roc_auc_score(y_test, score_test))
        out["train_test_gap"] = out["train_auc"] - out["test_auc"]
    except ValueError as e:
        out["error"] = str(e)
        return out
    cut = float(np.quantile(score_test, 0.99))
    flagged = score_test >= cut
    out["precision_at_top1pct"] = float(y_test[flagged].mean()) if flagged.sum() > 0 else 0.0
    out["flagged_count"] = int(flagged.sum())
    overfit_ok = abs(out["train_test_gap"]) < 0.05
    has_signal = out["test_auc"] > 0.55
    out["passes_overfit_gate"] = bool(overfit_ok and has_signal)
    return out


def report(name: str, description: str, results: dict[str, Any],
            extra: dict[str, Any] | None = None) -> None:
    """Write standardized result JSON to the experiment's own dir."""
    out_path = Path(sys.argv[0]).resolve().parent / "results.json"
    payload = {
        "experiment": name,
        "description": description,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        **results,
    }
    if extra:
        payload["extra"] = extra
    out_path.write_text(json.dumps(payload, indent=2))
    # Print summary
    pretty = ""
    for k in ("test_auc", "train_test_gap", "passes_overfit_gate",
              "precision_at_top1pct", "net_pnl_8bps", "win_rate"):
        if k in results:
            v = results[k]
            if isinstance(v, float):
                pretty += f"  {k}={v:.3f}"
            else:
                pretty += f"  {k}={v}"
    print(f"[{name}] {pretty}")
