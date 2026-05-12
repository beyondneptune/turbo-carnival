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
C3_DIR = REPO / "experiments" / "ml_lab" / "runs" / "athena_c3"


@dataclass
class GateState:
    logreg: object = None  # sklearn Pipeline
    logreg_feats: list[str] = None
    ae: object = None      # sklearn MLPRegressor
    ae_scaler: object = None
    ae_p95: float = float("inf")
    ae_p99: float = float("inf")
    # C3 composite — 4 sub-models
    c3_scaler: object = None
    c3_m016: object = None  # LocalOutlierFactor
    c3_m017: object = None  # OneClassSVM
    c3_m018: object = None  # EllipticEnvelope
    c3_m102: object = None  # RBFSampler + LogReg pipeline
    c3_p95_016: float = float("inf")
    c3_p95_017: float = float("inf")
    c3_p95_018: float = float("inf")
    loaded: bool = False
    error: Optional[str] = None


_state = GateState()


def load_gates() -> GateState:
    """Load all gate models from disk. Idempotent — call once at host startup."""
    global _state
    if _state.loaded:
        return _state
    errors = []
    # === Original ATHENA: LogReg + AE ===
    try:
        meta_path = ATHENA_DIR / "manifest.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            _state.logreg = joblib.load(ATHENA_DIR / meta["logreg_file"])
            _state.logreg_feats = meta["logreg_features"]
            _state.ae = joblib.load(ATHENA_DIR / meta["ae_file"])
            _state.ae_scaler = joblib.load(ATHENA_DIR / meta["ae_scaler_file"])
            _state.ae_p95 = float(meta["ae_p95"])
            _state.ae_p99 = float(meta["ae_p99"])
        else:
            errors.append(f"ATHENA manifest missing: {meta_path}")
    except Exception as exc:
        errors.append(f"ATHENA load failed: {exc}")

    # === C3 composite gate ===
    try:
        c3_meta_path = C3_DIR / "manifest.json"
        if c3_meta_path.exists():
            c3_meta = json.loads(c3_meta_path.read_text())
            sub = c3_meta["submodels"]
            _state.c3_scaler = joblib.load(C3_DIR / c3_meta["shared_scaler"])
            _state.c3_m016 = joblib.load(C3_DIR / sub["016_lof"]["file"])
            _state.c3_m017 = joblib.load(C3_DIR / sub["017_ocsvm"]["file"])
            _state.c3_m018 = joblib.load(C3_DIR / sub["018_envelope"]["file"])
            _state.c3_m102 = joblib.load(C3_DIR / sub["102_rbf_logreg"]["file"])
            _state.c3_p95_016 = float(sub["016_lof"]["p95_threshold"])
            _state.c3_p95_017 = float(sub["017_ocsvm"]["p95_threshold"])
            _state.c3_p95_018 = float(sub["018_envelope"]["p95_threshold"])
        else:
            errors.append(f"C3 manifest missing: {c3_meta_path}")
    except Exception as exc:
        errors.append(f"C3 load failed: {exc}")

    if errors:
        _state.error = " | ".join(errors)
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


def composite_c3_blocks(features: dict[str, float]) -> tuple[bool, dict]:
    """Composite C3 regime gate — empirically the best gate on yesterday's data.

    Returns (block, info_dict).
    info_dict contains which sub-models voted to block and their scores.

    Block rule: ANY of {016 LOF, 017 OCSVM, 018 EllipticEnvelope, 102 RBF+LR}.
    The anomaly detectors block if score > p95 of training. 102 blocks if predict_proba > 0.5.

    Fails open (returns (False, {...error: ...})) if models aren't loaded.
    """
    s = load_gates()
    if s.c3_m016 is None or s.c3_m102 is None:
        return (False, {"error": "c3 not loaded", "loaded": s.loaded})
    try:
        # All sub-models use the same 10-feature LR_FEATS schema
        x = np.array([[features[k] for k in (s.logreg_feats or [
            "ret1", "ret3", "ret10", "range_pct", "volume_z", "ret1_z",
            "vwap_dist", "tod_sin", "tod_cos", "vov",
        ])]])
    except KeyError as e:
        return (False, {"error": f"missing feature {e}"})
    info = {}
    blocks = []
    try:
        x_scaled = s.c3_scaler.transform(x)
        sc016 = -float(s.c3_m016.score_samples(x_scaled)[0])
        sc017 = -float(s.c3_m017.score_samples(x_scaled)[0])
        sc018 = -float(s.c3_m018.score_samples(x_scaled)[0])
        info["sc016"] = sc016; info["sc017"] = sc017; info["sc018"] = sc018
        if sc016 > s.c3_p95_016: blocks.append("016_lof")
        if sc017 > s.c3_p95_017: blocks.append("017_ocsvm")
        if sc018 > s.c3_p95_018: blocks.append("018_envelope")
        # 102 — supervised classifier
        p102 = float(s.c3_m102.predict_proba(x)[0, 1])
        info["p102"] = p102
        if p102 > 0.5: blocks.append("102_rbf_logreg")
    except Exception as exc:
        return (False, {"error": str(exc)[:200]})
    info["blockers"] = blocks
    return (len(blocks) > 0, info)


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
        "c3_loaded": all([s.c3_m016, s.c3_m017, s.c3_m018, s.c3_m102, s.c3_scaler]),
        "c3_p95_016": s.c3_p95_016,
        "c3_p95_017": s.c3_p95_017,
        "c3_p95_018": s.c3_p95_018,
    }


if __name__ == "__main__":
    # Smoke test — won't pass until 06_persist_athena_models.py has run
    print(json.dumps(health(), indent=2))
