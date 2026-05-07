"""Phase 3 evaluation baselines -- B0/B1/B2/B3 per `metadata/strategic_lock.md`.

The pre-registered gates require a strategy's net excess return to clear
`max(B0, B1, B2, B3)` by at least 200 bps annualised at both Base and +50%
costs. This module supplies the four baselines as pure functions:

    B0 = no-trade (cash at risk-free rate)
    B1 = equal-weight long straddle (every event)
    B2 = defined-risk short vol (iron butterfly, closed-form approximation)
    B3 = dumb logistic baseline (3 features only)

Each baseline is a pure function from (events, features) -> array of per-event
realized returns (in fraction units, e.g. 0.05 = +5%). A 'no-trade' event in
any baseline is encoded as 0.0 return for that index. All functions are
deterministic given a fixed `random_state` where a model is involved.

Conventions
-----------
  - `realized_5d_straddle_returns`: net 5-day return of a long ATM straddle,
    in fraction units, signed (positive = the straddle won).
  - `realized_5d_underlying_moves`: signed % move of the underlying T-1 -> T+5,
    fraction units (e.g. -0.03 = down 3%).
  - `pre_event_atm_iv_5d`: pre-event ATM IV (annualised, decimal e.g. 0.30).
  - All return arrays are length n_events, aligned by index.

No data fetching, no I/O; pure numpy / pandas / scikit-learn computations.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal  # noqa: F401  (kept for parity with other modules)

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# B0 -- no-trade (cash at risk-free rate)
# ---------------------------------------------------------------------------


def baseline_b0_no_trade(
    n_events: int,
    *,
    risk_free_rate_annual: float = 0.04,
    holding_days: int = 5,
) -> np.ndarray:
    """B0: cash held at risk-free rate over the holding period.

    Returns array of per-event returns, all equal to (1 + r)^(d / 365) - 1.
    With r = 0.04 and d = 5, this is approx 0.000538 per event.
    """
    if n_events < 0:
        raise ValueError(f"n_events must be non-negative, got {n_events}")
    per_event_return = (1.0 + risk_free_rate_annual) ** (holding_days / 365.0) - 1.0
    return np.full(n_events, per_event_return, dtype=np.float64)


# ---------------------------------------------------------------------------
# B1 -- equal-weight long straddle (every event)
# ---------------------------------------------------------------------------


def baseline_b1_long_straddle(
    realized_5d_straddle_returns: np.ndarray,
) -> np.ndarray:
    """B1: take every event with the SAME long straddle.

    Returns are the realized straddle returns passed through. There is no
    selection in B1; every event is traded with equal sizing.
    """
    arr = np.asarray(realized_5d_straddle_returns, dtype=np.float64)
    return arr.copy()


# ---------------------------------------------------------------------------
# B2 -- defined-risk short vol (iron butterfly)
# ---------------------------------------------------------------------------

# Black-Scholes constants for a 0-rate, 0-drift ATM/OTM approximation.
# ATM straddle premium ~ 2 * S * sigma * sqrt(T) * phi(0) = S * sigma * sqrt(T) * sqrt(2/pi).
_ATM_STRADDLE_COEF = math.sqrt(2.0 / math.pi)  # ~ 0.79788

# A 1-sigma OTM call premium under the same approximation:
#   C(K=S+S*sigma*sqrt(T)) ~ S * (phi(d1) - sigma*sqrt(T)*Phi(-d1))
# With d1 ~ -sigma*sqrt(T)/2 + 1, this simplifies for small T to roughly
# (phi(1) - 1 * (1 - Phi(1))) which is approximately 0.0833 in normalised
# vol-units. Doubled across both wings the total wing debit is ~ 0.1666 *
# sigma * sqrt(T). We expose this as a module-level constant so the formula
# is transparent for review.
_WING_DEBIT_COEF_PER_SIGMA = 2.0 * (
    1.0 / math.sqrt(2.0 * math.pi) * math.exp(-0.5)  # phi(1)
    - (1.0 - 0.8413447460685429)                     # 1 - Phi(1)
)
# The 0.8413... is Phi(1) under the standard normal. We hard-code it rather
# than calling scipy.stats just to keep this module dependency-free beyond
# numpy / pandas / sklearn.


def baseline_b2_iron_butterfly(
    realized_5d_underlying_moves: np.ndarray,    # signed % move T-1 to T+5
    pre_event_atm_iv_5d: np.ndarray,             # IV at decision time (annualised)
    *,
    short_strike_offset_iv_units: float = 0.0,   # short ATM (offset 0)
    long_wing_offset_iv_units: float = 1.0,      # long wings 1 sigma (i.e. expected move) away
    notional_per_event: float = 1.0,
    holding_days: int = 5,
    trading_days_per_year: int = 252,
) -> np.ndarray:
    """B2: defined-risk short vol (iron butterfly) per-event return.

    Approximation (closed-form, dependency-free):
        - sell ATM straddle (offset 0 IV-units below/above spot)
        - buy wings 1 expected-move away on each side (sigma * sqrt(5/252))

    Per-event PnL (in fraction-of-notional units) is computed as:
        expected_move = iv * sqrt(holding_days / trading_days_per_year)
        wing_width    = long_wing_offset_iv_units * expected_move
        atm_straddle_premium = ATM_STRADDLE_COEF * iv * sqrt(T)
        wing_debit           = WING_DEBIT_COEF_PER_SIGMA * iv * sqrt(T)
                               (1-sigma OTM wings on each side)
        credit  = atm_straddle_premium - wing_debit
        loss    = min(|move|, wing_width)        # short vol pays this out
        pnl     = (credit - loss) * notional_per_event
        return  = pnl / notional_per_event       # fraction-of-notional

    Per-event PnL is bounded above by the credit and below by `wing_width -
    credit` (i.e. the maximum loss is the wing distance net of credit). The
    `short_strike_offset_iv_units` parameter is exposed for symmetry with the
    spec but is unused by this approximation -- shifting the short strike off
    ATM under a 0-skew BS world only re-prices both legs by an equal amount.
    """
    moves = np.asarray(realized_5d_underlying_moves, dtype=np.float64)
    iv = np.asarray(pre_event_atm_iv_5d, dtype=np.float64)
    if moves.shape != iv.shape:
        raise ValueError(
            f"shape mismatch: moves {moves.shape} vs iv {iv.shape}"
        )
    if notional_per_event <= 0:
        raise ValueError(
            f"notional_per_event must be > 0, got {notional_per_event}"
        )
    # short-strike offset is currently a no-op under the 0-skew approximation;
    # surfaced so future skew-aware versions can plug in without API churn.
    _ = short_strike_offset_iv_units

    sqrt_t = math.sqrt(holding_days / trading_days_per_year)
    expected_move = iv * sqrt_t
    wing_width = long_wing_offset_iv_units * expected_move

    atm_straddle_premium = _ATM_STRADDLE_COEF * iv * sqrt_t
    wing_debit = _WING_DEBIT_COEF_PER_SIGMA * iv * sqrt_t
    credit = atm_straddle_premium - wing_debit

    loss = np.minimum(np.abs(moves), wing_width)
    pnl_fraction = credit - loss  # already in fraction-of-notional units

    # Multiply by notional and divide by notional -> identity, but keep the
    # parameter so the caller can scale absolute dollar PnL elsewhere.
    return (pnl_fraction * notional_per_event) / notional_per_event


# ---------------------------------------------------------------------------
# B3 -- dumb logistic baseline (3 features only)
# ---------------------------------------------------------------------------


def baseline_b3_logistic(
    features: pd.DataFrame,                       # exactly 3 columns
    realized_5d_straddle_returns: np.ndarray,
    *,
    train_test_split_idx: int | None = None,
    threshold: float = 0.5,
    random_state: int = 42,
) -> np.ndarray:
    """B3: dumb logistic baseline -- exactly 3 features, train/test split.

    Trains on the first half (or rows 0..split_idx-1), predicts on the second
    half. Label: 1 if realized_5d_straddle_return > 0 else 0. Selection: take
    the trade if predicted P(positive) > threshold, else 0 return.

    Train rows are masked to NaN (so summary stats can `np.nanmean` cleanly
    without contaminating in-sample with realized returns).

    Parameters
    ----------
    features
        DataFrame with exactly 3 columns. Order is preserved as feature order.
    realized_5d_straddle_returns
        Per-event realized straddle return (fraction units, signed).
    train_test_split_idx
        Index where test set begins. Defaults to n // 2.
    threshold
        Decision threshold on predicted P(positive). Default 0.5.
    random_state
        Seed for the underlying logistic regression solver.

    Returns
    -------
    np.ndarray, length n_events. Train rows are NaN. Test rows are the
    realized return when the model would have selected the trade, else 0.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError(f"features must be a DataFrame, got {type(features)}")
    if features.shape[1] != 3:
        raise ValueError(
            f"B3 requires exactly 3 features, got {features.shape[1]}"
        )
    returns = np.asarray(realized_5d_straddle_returns, dtype=np.float64)
    n = len(returns)
    if len(features) != n:
        raise ValueError(
            f"length mismatch: features {len(features)} vs returns {n}"
        )
    if n < 4:
        raise ValueError(f"need at least 4 events for train/test split, got {n}")

    split = train_test_split_idx if train_test_split_idx is not None else n // 2
    if not (1 <= split < n):
        raise ValueError(
            f"split must be in [1, {n - 1}], got {split}"
        )

    X = features.to_numpy(dtype=np.float64)
    y = (returns > 0).astype(np.int64)

    # Train rows: 0..split-1; test rows: split..n-1.
    X_train, y_train = X[:split], y[:split]
    X_test = X[split:]

    out = np.full(n, np.nan, dtype=np.float64)

    # Edge case: training labels all one class -> no learning possible. Fall
    # back to "trade nothing" on the test set (return 0 for every test row).
    unique_train_labels = np.unique(y_train)
    if len(unique_train_labels) < 2:
        out[split:] = 0.0
        return out

    model = LogisticRegression(random_state=random_state, max_iter=1000)
    model.fit(X_train, y_train)
    proba_positive = model.predict_proba(X_test)[:, 1]
    selected = proba_positive > threshold

    test_returns = returns[split:]
    out[split:] = np.where(selected, test_returns, 0.0)
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineSet:
    """Collected per-event returns from all four baselines.

    All four arrays are aligned by index and have equal length. B3 may
    contain NaN entries for the train portion; B0/B1/B2 are fully populated.
    """

    b0: np.ndarray
    b1: np.ndarray
    b2: np.ndarray
    b3: np.ndarray


def compute_all_baselines(
    realized_5d_straddle_returns: np.ndarray,
    realized_5d_underlying_moves: np.ndarray,
    pre_event_atm_iv_5d: np.ndarray,
    features_for_b3: pd.DataFrame,
    *,
    risk_free_rate_annual: float = 0.04,
) -> BaselineSet:
    """Run all four baselines and return as a `BaselineSet`.

    Lengths must be consistent across the three event-aligned arrays
    (`realized_5d_straddle_returns`, `realized_5d_underlying_moves`,
    `pre_event_atm_iv_5d`) and `features_for_b3`.
    """
    n = len(realized_5d_straddle_returns)
    if len(realized_5d_underlying_moves) != n:
        raise ValueError(
            f"underlying_moves length {len(realized_5d_underlying_moves)} != {n}"
        )
    if len(pre_event_atm_iv_5d) != n:
        raise ValueError(
            f"pre_event_atm_iv_5d length {len(pre_event_atm_iv_5d)} != {n}"
        )
    if len(features_for_b3) != n:
        raise ValueError(
            f"features_for_b3 length {len(features_for_b3)} != {n}"
        )

    b0 = baseline_b0_no_trade(n, risk_free_rate_annual=risk_free_rate_annual)
    b1 = baseline_b1_long_straddle(realized_5d_straddle_returns)
    b2 = baseline_b2_iron_butterfly(
        realized_5d_underlying_moves, pre_event_atm_iv_5d
    )
    b3 = baseline_b3_logistic(features_for_b3, realized_5d_straddle_returns)
    return BaselineSet(b0=b0, b1=b1, b2=b2, b3=b3)


def best_baseline_per_event(baselines: BaselineSet) -> np.ndarray:
    """Returns max(b0, b1, b2, b3) per index -- the 'easy alpha' the strategy
    must beat.

    NaNs (e.g. B3 train rows) are treated as missing and ignored at each
    index via `np.nanmax` so the train portion of B3 doesn't poison the max.
    If every baseline is NaN at some index (cannot happen unless someone
    invents a new B3 mode), the result is NaN there too.
    """
    stacked = np.vstack([baselines.b0, baselines.b1, baselines.b2, baselines.b3])
    return np.nanmax(stacked, axis=0)


def best_baseline_summary(baselines: BaselineSet) -> dict:
    """Returns a dict of mean per-event return for each baseline plus the
    pointwise best.

    All means are computed with `np.nanmean` so NaN entries (B3 train rows)
    do not corrupt the summary. Output keys: `b0_mean`, `b1_mean`, `b2_mean`,
    `b3_mean`, `best_per_event_mean` -- each a finite float.
    """
    best = best_baseline_per_event(baselines)
    return {
        "b0_mean": float(np.nanmean(baselines.b0)),
        "b1_mean": float(np.nanmean(baselines.b1)),
        "b2_mean": float(np.nanmean(baselines.b2)),
        "b3_mean": float(np.nanmean(baselines.b3)),
        "best_per_event_mean": float(np.nanmean(best)),
    }
