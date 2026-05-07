"""Phase 3 evaluation metrics — DSR, PBO, Sharpe, drawdown, regime stability.

All functions take pre-computed return series. No model training, no I/O.

Reference papers:
    - Bailey & Lopez de Prado (2014). "The Deflated Sharpe Ratio: Correcting for
      Selection Bias, Backtest Overfitting, and Non-Normality."
    - Bailey, Borwein, Lopez de Prado, Zhu (2017). "The Probability of Backtest
      Overfitting." Journal of Computational Finance.

These metrics implement the Pre-Registered Gates from
``metadata/strategic_lock.md`` (Cycle 1 gates 1, 2, 4, 6).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

__all__ = [
    "annualized_sharpe",
    "annualized_return",
    "max_drawdown",
    "deflated_sharpe_ratio",
    "probability_backtest_overfitting",
    "evaluate_regime_stability",
    "RegimeStabilityResult",
]


# ----------------------------------------------------------------------------
# Sharpe + drawdown basics
# ----------------------------------------------------------------------------


def _to_1d_array(returns) -> np.ndarray:
    """Coerce input to a 1-D float64 numpy array, dropping NaNs."""
    arr = np.asarray(returns, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim > 1:
        arr = arr.ravel()
    return arr[~np.isnan(arr)]


_NEAR_ZERO_STD = 1e-12


def _is_effectively_constant(arr: np.ndarray) -> bool:
    """Return True if ``arr`` has effectively zero spread.

    Floating-point representation of a "constant" list (e.g. ``[0.01] * N``)
    yields ``np.std`` on the order of 1e-18 even though the values are
    semantically identical. We treat any std below ``_NEAR_ZERO_STD`` and
    any peak-to-peak below ``_NEAR_ZERO_STD * max(1, |mean|)`` as constant.
    """
    if arr.size < 2:
        return True
    sigma = float(np.std(arr, ddof=1))
    if not np.isfinite(sigma):
        return True
    if sigma <= _NEAR_ZERO_STD:
        return True
    peak_to_peak = float(np.ptp(arr))
    scale = max(1.0, float(np.abs(np.mean(arr))))
    return peak_to_peak <= _NEAR_ZERO_STD * scale


def annualized_sharpe(
    returns,
    *,
    periods_per_year: int = 252,
) -> float:
    """Annualized Sharpe ratio: ``mean(r) / std(r) * sqrt(periods_per_year)``.

    Uses sample standard deviation (``ddof=1``) so a length-1 input or a
    constant series yields ``0.0`` rather than raising.

    Returns ``0.0`` for empty input or zero/non-finite std.
    """
    arr = _to_1d_array(returns)
    if _is_effectively_constant(arr):
        return 0.0
    sigma = float(np.std(arr, ddof=1))
    mu = float(np.mean(arr))
    return mu / sigma * float(np.sqrt(periods_per_year))


def max_drawdown(returns) -> float:
    """Maximum peak-to-trough drawdown of the cumulative-return curve.

    Returns a non-negative fraction (e.g. ``0.10`` for a 10% drawdown).
    Returns ``0.0`` for empty or strictly monotone-up curves.

    Convention: ``returns`` is interpreted as additive period returns. The
    equity curve is ``[1.0, 1+r1, (1+r1)(1+r2), ...]`` so a single negative
    return is correctly measured against the implicit starting capital
    of ``1.0``.
    """
    arr = _to_1d_array(returns)
    if arr.size == 0:
        return 0.0
    equity = np.concatenate(([1.0], np.cumprod(1.0 + arr)))
    running_max = np.maximum.accumulate(equity)
    drawdowns = (running_max - equity) / running_max
    return float(np.max(drawdowns))


def annualized_return(
    returns,
    *,
    periods_per_year: int = 252,
) -> float:
    """Arithmetic annualized return: ``mean(r) * periods_per_year``.

    For additive return series this is the standard simple annualization.
    Returns ``0.0`` for empty input.
    """
    arr = _to_1d_array(returns)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr)) * float(periods_per_year)


# ----------------------------------------------------------------------------
# Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014)
# ----------------------------------------------------------------------------


def deflated_sharpe_ratio(
    returns,
    *,
    n_trials: int = 1,
    periods_per_year: int = 252,
    target_sr: float = 0.0,
) -> tuple[float, float]:
    """Deflated Sharpe Ratio (Bailey & Lopez de Prado 2014).

    Returns ``(DSR, t_stat)`` where:
        - ``DSR`` is the deflated test statistic (a probit-style value).
        - ``t_stat`` is the equivalent classical Sharpe t-statistic against
          ``H0: SR <= target_sr`` (the ``n_trials = 1`` baseline).

    Formula::

        DSR = (SR_obs - SR_0) * sqrt(T - 1)
              / sqrt(1 - gamma_3 * SR_obs + ((gamma_4 - 1) / 4) * SR_obs**2)

    where ``SR_obs`` and ``SR_0`` are expressed in the **per-period** scale
    (i.e. annualized SR divided by ``sqrt(periods_per_year)``), and
    ``gamma_3`` / ``gamma_4`` are the sample skewness and kurtosis of the
    raw returns (kurtosis is the *non-excess* fourth moment).

    Approximation for the noise floor ``SR_0``:

    For ``n_trials = 1`` we set ``SR_0 = target_sr / sqrt(periods_per_year)``
    (i.e. translate the desired annualized hurdle into per-period units).

    For ``n_trials > 1`` we use the standard ``sqrt(2 * ln(n_trials))`` upper
    bound on the expected maximum of ``N`` standard-normal random variates,
    rescaled to the per-period Sharpe scale via division by ``sqrt(T)``::

        SR_0 = sqrt(2 * ln(n_trials)) / sqrt(T)

    The textbook Bailey/LdP form additionally involves the Euler-Mascheroni
    constant and the inverse normal CDF; the ``sqrt(2 ln N)`` bound is a
    conservative, widely-used approximation that has the right asymptotic
    growth rate. We document this choice rather than masking it.

    The function returns ``(0.0, 0.0)`` for empty / single / zero-variance
    inputs. ``DSR`` is the raw test statistic; convert to a probability via
    ``scipy.stats.norm.cdf(DSR)`` if desired.
    """
    arr = _to_1d_array(returns)
    T = arr.size
    if T < 2:
        return 0.0, 0.0
    if _is_effectively_constant(arr):
        return 0.0, 0.0
    sigma = float(np.std(arr, ddof=1))
    mu = float(np.mean(arr))

    # Per-period observed Sharpe.
    sr_obs = mu / sigma

    # Skewness and (non-excess) kurtosis of returns. ``stats.kurtosis``
    # with ``fisher=False`` returns the raw fourth standardized moment so
    # ``gamma_4 = 3`` for a normal distribution.
    gamma_3 = float(stats.skew(arr, bias=True))
    gamma_4 = float(stats.kurtosis(arr, fisher=False, bias=True))

    # Noise floor SR_0 in per-period units.
    if n_trials <= 1:
        sr_0 = float(target_sr) / float(np.sqrt(periods_per_year))
    else:
        sr_0 = float(np.sqrt(2.0 * np.log(float(n_trials)))) / float(np.sqrt(T))

    # Classical t-stat against target_sr (per-period).
    t_stat = (sr_obs - float(target_sr) / float(np.sqrt(periods_per_year))) * float(
        np.sqrt(T - 1)
    )

    denom = 1.0 - gamma_3 * sr_obs + (gamma_4 - 1.0) / 4.0 * sr_obs * sr_obs
    if not np.isfinite(denom) or denom <= 0.0:
        # Pathological higher moments — fall back to the classical t-stat
        # against SR_0 with no variance correction.
        return float((sr_obs - sr_0) * np.sqrt(T - 1)), t_stat

    dsr = (sr_obs - sr_0) * float(np.sqrt(T - 1)) / float(np.sqrt(denom))
    return float(dsr), float(t_stat)


# ----------------------------------------------------------------------------
# Probability of Backtest Overfitting (CSCV)
# ----------------------------------------------------------------------------


def probability_backtest_overfitting(
    returns_matrix,
    *,
    n_splits: int = 16,
) -> float:
    """Probability of Backtest Overfitting via Combinatorially Symmetric CV.

    Implements Bailey, Borwein, Lopez de Prado, and Zhu (2017).

    Parameters
    ----------
    returns_matrix : array-like, shape (T, N)
        ``T`` time observations of ``N >= 2`` candidate strategies. Rows are
        time, columns are strategies.
    n_splits : int, default 16
        Number of equal-sized contiguous groups ``S`` to partition rows into.
        Must be even (we form ``C(S, S/2)`` train/test splits). The function
        will trim the leading rows if ``T`` is not divisible by ``S``.

    Returns
    -------
    float
        PBO in ``[0, 1]``. Lower is better; the gate threshold is ``< 0.5``.

    Method
    ------
    For each combinatorial split into IS / OOS halves:

    1. Compute Sharpe of every strategy in IS rows; pick the IS-best
       strategy index ``n*``.
    2. Rank strategies by Sharpe in OOS rows (rank 0 = worst, ``N - 1`` = best).
    3. Convert OOS rank to a relative rank ``omega = (rank + 1) / (N + 1)``,
       then to a logit ``lambda = log(omega / (1 - omega))``.
    4. PBO is the fraction of splits where ``lambda <= 0`` (i.e. the chosen
       strategy ranks at or below the OOS median).
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2:
        raise ValueError(
            f"returns_matrix must be 2-D (T, N); got ndim={M.ndim}"
        )
    T, N = M.shape
    if N < 2:
        raise ValueError(f"PBO requires at least 2 strategies; got N={N}")
    if n_splits < 2 or n_splits % 2 != 0:
        raise ValueError(
            f"n_splits must be an even integer >= 2; got n_splits={n_splits}"
        )
    if T < n_splits:
        raise ValueError(
            f"Need at least n_splits={n_splits} rows; got T={T}"
        )

    # Trim leading rows so T is divisible by n_splits.
    rows_per_group = T // n_splits
    usable = rows_per_group * n_splits
    if usable < T:
        M = M[T - usable :]
        T = usable

    # Group-index assignment for each row.
    group_of_row = np.repeat(np.arange(n_splits), rows_per_group)

    half = n_splits // 2
    all_groups = list(range(n_splits))

    logit_below_median = 0
    total_splits = 0

    for is_groups in combinations(all_groups, half):
        is_set = set(is_groups)
        is_mask = np.array([g in is_set for g in group_of_row])
        oos_mask = ~is_mask

        is_block = M[is_mask]
        oos_block = M[oos_mask]

        # Per-strategy Sharpe within block (ignore zero-std columns).
        is_sharpe = _columnwise_sharpe(is_block)
        oos_sharpe = _columnwise_sharpe(oos_block)

        # Pick best in-sample strategy.
        n_star = int(np.argmax(is_sharpe))
        # OOS rank of n_star (0 = worst, N - 1 = best). Ties broken by index
        # so that the same strategy gets a deterministic rank across splits.
        oos_order = np.argsort(oos_sharpe, kind="mergesort")  # ascending
        rank = int(np.where(oos_order == n_star)[0][0])

        omega = (rank + 1.0) / (N + 1.0)
        # logit lambda; lambda <= 0  <=>  omega <= 0.5  <=>  rank below median.
        lam = np.log(omega / (1.0 - omega))
        if lam <= 0.0:
            logit_below_median += 1
        total_splits += 1

    if total_splits == 0:
        return 0.0
    return float(logit_below_median) / float(total_splits)


def _columnwise_sharpe(block: np.ndarray) -> np.ndarray:
    """Per-column Sharpe (mean / std). Returns ``-inf`` for zero-std columns
    so they cannot be selected as IS-best."""
    if block.shape[0] < 2:
        return np.full(block.shape[1], -np.inf)
    mu = np.mean(block, axis=0)
    sigma = np.std(block, axis=0, ddof=1)
    out = np.full(block.shape[1], -np.inf)
    nz = sigma > _NEAR_ZERO_STD
    out[nz] = mu[nz] / sigma[nz]
    return out


# ----------------------------------------------------------------------------
# Regime stability (HMM)
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class RegimeStabilityResult:
    """Outcome of an HMM-partitioned regime-stability check.

    Attributes
    ----------
    n_states : int
        Number of latent states fitted.
    state_assignments : np.ndarray
        Shape ``(T,)``: most-likely hidden state index per period.
    state_returns : dict[int, float]
        Mean strategy return per state.
    benchmark_state_returns : dict[int, float]
        Mean benchmark return per state.
    underperformance_per_state : dict[int, float]
        ``state_returns[s] - benchmark_state_returns[s]``. Negative means
        the strategy underperforms the benchmark in state ``s``.
    worst_underperformance : float
        ``min(underperformance_per_state.values())``. Negative implies
        underperformance in at least one regime.
    passed : bool
        ``True`` iff ``worst_underperformance > -threshold``, i.e. the
        strategy never underperforms the benchmark by more than ``threshold``
        in any single regime (Pre-Registered Gate 6).
    """

    n_states: int
    state_assignments: np.ndarray
    state_returns: dict[int, float]
    benchmark_state_returns: dict[int, float]
    underperformance_per_state: dict[int, float]
    worst_underperformance: float
    passed: bool


def evaluate_regime_stability(
    strategy_returns,
    benchmark_returns,
    *,
    n_states: int = 2,
    threshold: float = 0.05,
    random_state: int = 42,
) -> RegimeStabilityResult:
    """Per-regime relative-performance check.

    Fits a Gaussian HMM with ``n_states`` on the **benchmark** return series,
    assigns each period to its most-likely latent state, then for each state
    computes the mean strategy and benchmark return. The strategy "passes"
    if its underperformance against the benchmark in **every** state is no
    worse than ``-threshold``.

    Parameters
    ----------
    strategy_returns, benchmark_returns : array-like
        Aligned return series of equal length.
    n_states : int, default 2
        Number of HMM states.
    threshold : float, default 0.05
        Underperformance tolerance (5%).
    random_state : int, default 42
        Seed for HMM initialization (deterministic).

    Returns
    -------
    RegimeStabilityResult

    Raises
    ------
    ValueError
        If inputs are misaligned, too short, or non-finite.
    """
    s = np.asarray(strategy_returns, dtype=float)
    b = np.asarray(benchmark_returns, dtype=float)
    if s.ndim != 1 or b.ndim != 1:
        raise ValueError("strategy_returns and benchmark_returns must be 1-D")
    if s.shape != b.shape:
        raise ValueError(
            f"shape mismatch: strategy={s.shape}, benchmark={b.shape}"
        )
    if s.size < max(2 * n_states, 10):
        raise ValueError(
            f"Need at least {max(2 * n_states, 10)} observations to fit a "
            f"{n_states}-state HMM; got {s.size}"
        )
    if not (np.all(np.isfinite(s)) and np.all(np.isfinite(b))):
        raise ValueError("returns must be finite")
    if n_states < 1:
        raise ValueError(f"n_states must be >= 1; got {n_states}")

    # Lazy import: hmmlearn is only required when this function is invoked.
    from hmmlearn.hmm import GaussianHMM

    if n_states == 1:
        # Trivial: one regime → unconditional mean comparison.
        states = np.zeros(s.size, dtype=int)
    else:
        hmm = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=100,
            tol=1e-3,
            random_state=random_state,
        )
        hmm.fit(b.reshape(-1, 1))
        states = hmm.predict(b.reshape(-1, 1))

    state_returns: dict[int, float] = {}
    bench_returns: dict[int, float] = {}
    underperf: dict[int, float] = {}
    for k in range(n_states):
        mask = states == k
        if mask.sum() == 0:
            # Empty state: define returns as 0.0 so the gate is not
            # accidentally tripped by an unobserved regime.
            state_returns[k] = 0.0
            bench_returns[k] = 0.0
            underperf[k] = 0.0
            continue
        sr = float(np.mean(s[mask]))
        br = float(np.mean(b[mask]))
        state_returns[k] = sr
        bench_returns[k] = br
        underperf[k] = sr - br

    worst = float(min(underperf.values())) if underperf else 0.0
    passed = worst > -float(threshold)

    return RegimeStabilityResult(
        n_states=n_states,
        state_assignments=states,
        state_returns=state_returns,
        benchmark_state_returns=bench_returns,
        underperformance_per_state=underperf,
        worst_underperformance=worst,
        passed=passed,
    )
