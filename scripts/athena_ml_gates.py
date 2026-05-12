"""LogReg regime gate + AE anomaly veto loaders for ATHENA v1.

Both ML artifacts are persisted to `experiments/ml_lab/runs/athena/` by
`experiments/ml_lab/06_persist_athena_models.py`. This module exposes the
hot-path predict functions the live host calls.

Designed to fail gracefully: if the model file is missing or corrupt,
each gate falls back to "allow everything" (the rule-based filters still
apply). The live host should still run, the gate just becomes a no-op.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np

REPO = Path("/Users/weeoto/turbo-carnival")
ATHENA_DIR = REPO / "experiments" / "ml_lab" / "runs" / "athena"


@dataclass
class GateState:
    logreg: object = None  # sklearn Pipeline
    logreg_feats: list[str] = None
    ae: object = None      # sklearn MLPRegressor
    ae_scaler: object = None
    ae_p95: float = float("inf")
    ae_p99: float = float("inf")
    loaded: bool = False
    error: Optional[str] = None


_state = GateState()


def load_gates() -> GateState:
    """Load both models from disk. Idempotent — call once at host startup."""
    global _state
    if _state.loaded:
        return _state
    try:
        meta_path = ATHENA_DIR / "manifest.json"
        if not meta_path.exists():
            _state.error = f"manifest missing: {meta_path}"
            _state.loaded = True
            return _state
        meta = json.loads(meta_path.read_text())
        _state.logreg = joblib.load(ATHENA_DIR / meta["logreg_file"])
        _state.logreg_feats = meta["logreg_features"]
        _state.ae = joblib.load(ATHENA_DIR / meta["ae_file"])
        _state.ae_scaler = joblib.load(ATHENA_DIR / meta["ae_scaler_file"])
        _state.ae_p95 = float(meta["ae_p95"])
        _state.ae_p99 = float(meta["ae_p99"])
        _state.loaded = True
    except Exception as exc:
        _state.error = f"load failed: {exc}"
        _state.loaded = True
    return _state


def logreg_p_volatile(features: dict[str, float]) -> Optional[float]:
    """Return P(next-5-min |ret| > 30 bps) for the current bar.

    `features` must contain the keys listed in `logreg_features` in manifest.
    Returns None if the gate is not loaded — caller should treat as pass.
    """
    s = load_gates()
    if s.logreg is None:
        return None
    try:
        x = np.array([[features[k] for k in s.logreg_feats]])
        return float(s.logreg.predict_proba(x)[0, 1])
    except (KeyError, ValueError):
        return None


def logreg_blocks(features: dict[str, float], p_block: float = 0.50) -> bool:
    """True if the LogReg gate says BLOCK this entry."""
    p = logreg_p_volatile(features)
    if p is None:
        return False  # fail-open
    return p >= p_block


def ae_recon_error(window_30x6: np.ndarray) -> Optional[float]:
    """Return MSE reconstruction error for a 30-bar × 6-feature window.

    Window shape: (30, 6) — flattened internally. Order of features must
    match training: [ret1, ret3, range_pct, volume_z, ret1_z, vwap_dist].
    """
    s = load_gates()
    if s.ae is None or s.ae_scaler is None:
        return None
    try:
        flat = window_30x6.reshape(1, -1)
        scaled = s.ae_scaler.transform(flat)
        recon = s.ae.predict(scaled)
        err = float(((recon - scaled) ** 2).mean())
        return err
    except Exception:
        return None


def ae_decision(window_30x6: np.ndarray) -> tuple[str, float]:
    """Return ('veto'|'size_down'|'pass', recon_err).

    'veto'      → recon_err > p99: hard block the trade
    'size_down' → p95 < err ≤ p99: halve the size
    'pass'      → err ≤ p95: full size
    """
    err = ae_recon_error(window_30x6)
    if err is None:
        return ("pass", float("nan"))
    s = load_gates()
    if err > s.ae_p99:
        return ("veto", err)
    if err > s.ae_p95:
        return ("size_down", err)
    return ("pass", err)


def health() -> dict:
    """Diagnostic — call from dashboard / health checks."""
    s = load_gates()
    return {
        "loaded": s.loaded,
        "error": s.error,
        "logreg_present": s.logreg is not None,
        "ae_present": s.ae is not None,
        "ae_p95": s.ae_p95,
        "ae_p99": s.ae_p99,
        "logreg_features": s.logreg_feats,
    }


if __name__ == "__main__":
    # Smoke test — won't pass until 06_persist_athena_models.py has run
    print(json.dumps(health(), indent=2))
