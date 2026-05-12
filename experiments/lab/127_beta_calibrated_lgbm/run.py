"""Beta-calibrated LGBM — novel calibration via 3-parameter Beta function.

Architecture: LGBM classifier + post-hoc Beta calibration (sklearn-free).
The Beta calibration function is: 1 / (1 + ((1-p)^a * e^c) / (p^b))
Parameters a, b, c are fit on a held-out calibration split via maximum likelihood.
This differs from Platt sigmoid (2-parameter) and isotonic (non-parametric).
Strong calibration option for boosting models.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import lightgbm as lgb
from scipy.optimize import minimize
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from shared.data import load_classifier_data
from shared.eval import evaluate_proba, report


def beta_calibration_function(p: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Apply Beta calibration: 1 / (1 + ((1-p)^a * e^c) / (p^b))

    Args:
        p: Raw probabilities (clipped to avoid numerics)
        a, b, c: Beta calibration parameters

    Returns:
        Calibrated probabilities
    """
    p_clipped = np.clip(p, 1e-6, 1 - 1e-6)
    numerator = (1 - p_clipped) ** a * np.exp(c)
    denominator = p_clipped ** b
    # Avoid division by zero
    ratio = numerator / (denominator + 1e-10)
    return 1.0 / (1.0 + ratio)


def beta_calibration_nll(params: tuple[float, float, float],
                         p_cal_raw: np.ndarray,
                         y_cal: np.ndarray) -> float:
    """Negative log-likelihood for Beta calibration on calibration set.

    Args:
        params: (a, b, c)
        p_cal_raw: Raw probabilities on calibration split
        y_cal: True labels on calibration split

    Returns:
        Negative log-likelihood (for minimization)
    """
    a, b, c = params

    # Apply Beta calibration
    p_cal_calibrated = beta_calibration_function(p_cal_raw, a, b, c)
    p_cal_calibrated = np.clip(p_cal_calibrated, 1e-7, 1 - 1e-7)

    # Log-loss
    eps = 1e-7
    loss = -(y_cal * np.log(p_cal_calibrated + eps) +
             (1 - y_cal) * np.log(1 - p_cal_calibrated + eps))
    return loss.mean()


def compute_calibration_metrics(y_true: np.ndarray, p_pred: np.ndarray) -> tuple[float, float]:
    """Compute calibration slope and intercept via logistic regression.

    Fit: logit(p_pred) ~ intercept + slope * logit(p_true_proxy)
    We use a simple proxy: bin p_pred and compute mean y_true per bin.
    """
    from scipy.special import logit, expit

    # Create calibration bins
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(p_pred, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_pred_means = []
    bin_true_means = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_pred_means.append(p_pred[mask].mean())
            bin_true_means.append(y_true[mask].mean())

    if len(bin_pred_means) < 2:
        return 1.0, 0.0  # Perfect calibration fallback

    bin_pred_means = np.array(bin_pred_means)
    bin_true_means = np.array(bin_true_means)

    # Fit logistic calibration: logit(true) ~ intercept + slope * logit(pred)
    logit_pred = logit(np.clip(bin_pred_means, 1e-6, 1 - 1e-6))
    logit_true = logit(np.clip(bin_true_means, 1e-6, 1 - 1e-6))

    # Simple linear regression
    A = np.vstack([np.ones(len(logit_pred)), logit_pred]).T
    try:
        params, _, _, _ = np.linalg.lstsq(A, logit_true, rcond=None)
        intercept, slope = params
    except np.linalg.LinAlgError:
        intercept, slope = 0.0, 1.0

    return slope, intercept


# Load data
Xt, yt, Xv, yv = load_classifier_data()
scaler = StandardScaler()
Xt_scaled = scaler.fit_transform(Xt)
Xv_scaled = scaler.transform(Xv)

# Split training data: 80% for LGBM fit, 20% for calibration
X_fit, X_cal, y_fit, y_cal = train_test_split(
    Xt_scaled, yt, test_size=0.2, random_state=42, stratify=yt
)

# Train LGBM
lgbm = lgb.LGBMClassifier(
    n_estimators=200,
    class_weight='balanced',
    random_state=42,
    verbose=-1,
    learning_rate=0.1,
    max_depth=5,
    num_leaves=31,
)
lgbm.fit(X_fit, y_fit)

# Get raw probabilities on calibration split
p_cal_raw = lgbm.predict_proba(X_cal)[:, 1].astype(np.float64)
p_cal_raw = np.clip(p_cal_raw, 1e-6, 1 - 1e-6)

# Fit Beta calibration parameters on calibration split
initial_params = (1.0, 1.0, 0.0)  # a, b, c
result = minimize(
    beta_calibration_nll,
    initial_params,
    args=(p_cal_raw, y_cal),
    method='Nelder-Mead',
    options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6}
)

a_opt, b_opt, c_opt = result.x

# Define calibrator function
def calibrate_proba(p_raw: np.ndarray) -> np.ndarray:
    """Apply learned Beta calibration to raw probabilities."""
    return beta_calibration_function(p_raw, a_opt, b_opt, c_opt)


# Get calibrated probabilities on train and test
p_train_raw = lgbm.predict_proba(Xt_scaled)[:, 1]
p_test_raw = lgbm.predict_proba(Xv_scaled)[:, 1]

p_train_calibrated = calibrate_proba(p_train_raw)
p_test_calibrated = calibrate_proba(p_test_raw)

# Compute metrics
metrics = evaluate_proba(p_train_calibrated, p_test_calibrated, yt, yv)

# Compute calibration metrics (slope, intercept)
cal_slope, cal_intercept = compute_calibration_metrics(yv, p_test_calibrated)

# Add extra calibration info
extra = {
    "beta_a": float(a_opt),
    "beta_b": float(b_opt),
    "beta_c": float(c_opt),
    "calibration_slope": float(cal_slope),
    "calibration_intercept": float(cal_intercept),
    "nll_on_cal_split": float(result.fun),
}

report(
    "127_beta_calibrated_lgbm",
    "LGBM classifier with 3-parameter Beta calibration (sklearn-free). Beta function: 1/(1+((1-p)^a * e^c)/(p^b)). Params fit on 20% held-out calibration split via max-likelihood.",
    metrics,
    extra=extra
)
