"""Phase 2 deterministic kill switch. Pure Python; no global mutable state.

Inputs: NAV history (DataFrame). Outputs: KillSwitchStatus + reasons.

Once triggered (suspended), the trigger is sticky for the cycle — re-evaluating
on later data with the same trigger date returns the same suspension. This
"stickiness" is a property of the input data (append-only NAV history), not of
any in-memory state. There is no `disable()` method by design: the kill switch
cannot be silenced by an LLM or a human; the only way to "reset" it is to
reset the cycle (i.e. start from a new history).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd


KillReason = Literal[
    "intraday_loss_8pct",
    "cycle_drawdown_15pct",
    "cash_buffer_below_20pct",
]


@dataclass(frozen=True)
class KillSwitchStatus:
    """Immutable snapshot of kill-switch evaluation."""

    suspended: bool  # True if any HARD kill triggered
    soft_block: bool  # True if soft (cash buffer) block triggered
    triggered_reasons: list[KillReason] = field(default_factory=list)
    triggered_at: date | None = None  # date when first hard trigger happened
    details: dict[str, float] = field(default_factory=dict)


def _coerce_date(value) -> date:
    """Coerce a pandas/numpy/datetime/date scalar to a stdlib date."""
    if isinstance(value, date) and not isinstance(value, pd.Timestamp):
        # date but not Timestamp (Timestamp is a subclass of datetime, not date)
        return value
    ts = pd.Timestamp(value)
    return ts.date()


def check_intraday_loss(
    nav_history: pd.DataFrame,
    threshold_pct: float = 0.08,
) -> tuple[bool, date | None, dict]:
    """Check whether any single-day intraday loss exceeds ``threshold_pct``.

    ``nav_history`` must have columns ``date``, ``nav``, ``intraday_low`` and be
    sorted ascending by date. Each row's "open" is taken to be the *previous*
    day's close (``nav`` of the row before). The first row has no previous day,
    so it cannot trigger an intraday breach.

    Intraday loss for day ``i`` is computed as
    ``(open_i - intraday_low_i) / open_i`` where ``open_i = nav_{i-1}``.

    Returns ``(True, first_breach_date, details)`` if any day breaches, else
    ``(False, None, {})``. ``details`` includes ``max_intraday_loss_pct`` and
    ``breach_date`` (as an ISO string when present).
    """
    if nav_history is None or len(nav_history) < 2:
        return False, None, {}

    df = nav_history.reset_index(drop=True)

    # open_i = nav_{i-1}; first row has no open and so cannot breach.
    opens = df["nav"].shift(1)
    lows = df["intraday_low"]

    # Avoid div-by-zero — opens > 0 expected for paper NAV, but be defensive.
    with np.errstate(divide="ignore", invalid="ignore"):
        loss_pct = (opens - lows) / opens

    loss_pct = loss_pct.where(opens > 0)

    # First row has NaN open; ignore by construction.
    breach_mask = loss_pct >= threshold_pct
    breach_mask = breach_mask.fillna(False)

    max_loss = float(loss_pct.max(skipna=True)) if loss_pct.notna().any() else 0.0
    if not bool(breach_mask.any()):
        return False, None, {"max_intraday_loss_pct": max_loss}

    first_idx = int(breach_mask.idxmax())  # idxmax on boolean returns first True
    breach_date = _coerce_date(df.loc[first_idx, "date"])
    details = {
        "max_intraday_loss_pct": max_loss,
        "breach_date": breach_date.isoformat(),
    }
    return True, breach_date, details


def check_cycle_drawdown(
    nav_history: pd.DataFrame,
    threshold_pct: float = 0.15,
) -> tuple[bool, date | None, dict]:
    """Check cumulative drawdown vs running peak NAV.

    For each row, drawdown = ``(running_peak - nav) / running_peak``. If any row
    has drawdown ``>= threshold_pct``, return ``(True, first_breach_date,
    details)``. ``details`` always reports ``max_drawdown_pct`` and (when a
    breach occurs) ``peak_nav``, ``breach_nav``, ``breach_date``.
    """
    if nav_history is None or len(nav_history) == 0:
        return False, None, {}

    df = nav_history.reset_index(drop=True)
    nav = df["nav"].astype(float)
    running_peak = nav.cummax()

    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = (running_peak - nav) / running_peak
    drawdown = drawdown.where(running_peak > 0).fillna(0.0)

    max_dd = float(drawdown.max())
    breach_mask = drawdown >= threshold_pct
    if not bool(breach_mask.any()):
        return False, None, {"max_drawdown_pct": max_dd}

    first_idx = int(breach_mask.idxmax())
    breach_date = _coerce_date(df.loc[first_idx, "date"])
    details = {
        "max_drawdown_pct": max_dd,
        "peak_nav": float(running_peak.iloc[first_idx]),
        "breach_nav": float(nav.iloc[first_idx]),
        "breach_date": breach_date.isoformat(),
    }
    return True, breach_date, details


def check_cash_buffer(
    cash_usd: float,
    nav_usd: float,
    threshold_pct: float = 0.20,
) -> tuple[bool, dict]:
    """Soft check: does NOT suspend the cycle, but blocks new positions.

    Returns ``(breached, details)``. ``breached=True`` means cash is *strictly
    less than* ``threshold_pct`` of NAV (i.e. exactly 20% is OK).
    """
    if nav_usd <= 0:
        # Degenerate: zero-or-negative NAV — treat as breached, ratio = 0.0.
        return True, {"cash_pct_of_nav": 0.0}

    ratio = float(cash_usd) / float(nav_usd)
    breached = ratio < threshold_pct
    return breached, {"cash_pct_of_nav": ratio}


def evaluate_kill_switch(
    nav_history: pd.DataFrame,
    current_cash_usd: float,
    current_nav_usd: float,
) -> KillSwitchStatus:
    """Run all 3 checks and return an immutable :class:`KillSwitchStatus`.

    - ``suspended = (intraday_breach OR drawdown_breach)``
    - ``soft_block = cash_buffer_breach``
    - ``triggered_reasons`` lists every breach (including the soft one).
    - ``triggered_at`` is the earliest date among hard trigger dates (or None).
    - ``details`` merges all numeric values from each check.

    The function is pure: re-running with the same inputs yields an equal
    ``KillSwitchStatus`` (verified in tests). Suspension persists ("sticks")
    because the input history is append-only; new rows do not erase past
    breaches.
    """
    intraday_hit, intraday_date, intraday_details = check_intraday_loss(nav_history)
    drawdown_hit, drawdown_date, drawdown_details = check_cycle_drawdown(nav_history)
    cash_hit, cash_details = check_cash_buffer(current_cash_usd, current_nav_usd)

    reasons: list[KillReason] = []
    if intraday_hit:
        reasons.append("intraday_loss_8pct")
    if drawdown_hit:
        reasons.append("cycle_drawdown_15pct")
    if cash_hit:
        reasons.append("cash_buffer_below_20pct")

    hard_dates = [d for d in (intraday_date, drawdown_date) if d is not None]
    triggered_at = min(hard_dates) if hard_dates else None

    details: dict[str, float] = {}
    # Prefix to avoid key collisions across the three checks.
    for key, value in intraday_details.items():
        details[f"intraday_{key}"] = value
    for key, value in drawdown_details.items():
        details[f"drawdown_{key}"] = value
    for key, value in cash_details.items():
        details[f"cash_{key}"] = value

    return KillSwitchStatus(
        suspended=bool(intraday_hit or drawdown_hit),
        soft_block=bool(cash_hit),
        triggered_reasons=reasons,
        triggered_at=triggered_at,
        details=details,
    )
