"""Russell 1000 universe filter for the earnings satellite sleeve.
Pure function — takes a name-level fundamentals snapshot, returns the top-50 eligible names."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniverseFilters:
    """Liquidity gates and top-N cap for the satellite earnings sleeve.

    Defaults match `metadata/strategic_lock.md` § Satellite Sleeve. Thresholds are
    inclusive on the passing side (>= for floors, <= for the spread ceiling).
    """

    min_options_adv: int = 5_000
    min_options_oi: int = 10_000
    max_atm_spread_pct: float = 0.08
    min_historical_events: int = 8
    top_n_by_adv: int = 50


REQUIRED_COLUMNS = [
    "ticker",
    "options_adv_30d",
    "options_oi_30d",
    "atm_spread_pct_30d",
    "earnings_events_count",
]


# Stable mapping from filter-attribute name to the column it gates. Used by both
# `apply_liquidity_filters` (for the gate masks) and `explain_exclusion` (for the
# audit-line names). Keeping this in one place keeps the gate names that surface
# in logs identical to the dataclass field names.
_GATE_SPECS: list[tuple[str, str, Literal["ge", "le"]]] = [
    ("min_options_adv", "options_adv_30d", "ge"),
    ("min_options_oi", "options_oi_30d", "ge"),
    ("max_atm_spread_pct", "atm_spread_pct_30d", "le"),
    ("min_historical_events", "earnings_events_count", "ge"),
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate_columns(candidates: pd.DataFrame) -> None:
    """Raise ValueError if any required column is absent from candidates."""
    missing = [c for c in REQUIRED_COLUMNS if c not in candidates.columns]
    if missing:
        raise ValueError(
            f"candidates DataFrame missing required columns: {missing} "
            f"(required: {REQUIRED_COLUMNS})"
        )


def _gate_mask(
    column: pd.Series,
    threshold: float,
    direction: Literal["ge", "le"],
) -> pd.Series:
    """Return boolean mask: rows passing the gate. NaN values always fail.

    `ge` => column >= threshold;  `le` => column <= threshold.
    NaN propagation in pandas comparisons normally yields False, but we make the
    NaN-fails contract explicit here for safety against pandas version changes.
    """
    notna = column.notna()
    if direction == "ge":
        return notna & (column >= threshold)
    if direction == "le":
        return notna & (column <= threshold)
    raise ValueError(f"unknown gate direction: {direction!r}")  # defensive


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def apply_liquidity_filters(
    candidates: pd.DataFrame,
    filters: UniverseFilters | None = None,
) -> pd.DataFrame:
    """Returns rows passing ALL liquidity gates. Validates required columns first.

    Gates (all must pass; NaN fails):
        options_adv_30d         >= filters.min_options_adv
        options_oi_30d          >= filters.min_options_oi
        atm_spread_pct_30d      <= filters.max_atm_spread_pct
        earnings_events_count   >= filters.min_historical_events

    Returns a copy with the original index preserved. Caller can re-index later.
    """
    if filters is None:
        filters = UniverseFilters()
    _validate_columns(candidates)

    thresholds: dict[str, float] = {
        "min_options_adv": filters.min_options_adv,
        "min_options_oi": filters.min_options_oi,
        "max_atm_spread_pct": filters.max_atm_spread_pct,
        "min_historical_events": filters.min_historical_events,
    }

    mask = pd.Series(True, index=candidates.index)
    for gate_name, col, direction in _GATE_SPECS:
        mask &= _gate_mask(candidates[col], thresholds[gate_name], direction)

    return candidates.loc[mask].copy()


def select_top_n(
    eligible: pd.DataFrame,
    filters: UniverseFilters | None = None,
) -> pd.DataFrame:
    """Returns top-N rows by options_adv_30d (descending). Caller can subset further.

    If fewer than N rows are supplied, returns all of them. A `rank` column is
    added (1..len, where 1 = highest ADV). The returned frame's index is reset.
    """
    if filters is None:
        filters = UniverseFilters()
    if "options_adv_30d" not in eligible.columns:
        raise ValueError("eligible DataFrame missing 'options_adv_30d' column")

    n = filters.top_n_by_adv
    if n <= 0:
        raise ValueError(f"top_n_by_adv must be positive, got {n}")

    # Stable sort so ties retain incoming order (callers that pre-sort by some
    # secondary key — e.g. alphabetical ticker — get deterministic output).
    sorted_df = eligible.sort_values(
        "options_adv_30d", ascending=False, kind="mergesort"
    ).head(n).reset_index(drop=True)
    sorted_df["rank"] = np.arange(1, len(sorted_df) + 1, dtype=np.int64)
    return sorted_df


def build_universe(
    candidates: pd.DataFrame,
    filters: UniverseFilters | None = None,
    *,
    as_of: date | None = None,
) -> pd.DataFrame:
    """Convenience: apply_liquidity_filters → select_top_n.

    Returns DataFrame with columns: ticker, options_adv_30d, options_oi_30d,
    atm_spread_pct_30d, earnings_events_count, rank (1..N), as_of.

    `as_of` is broadcast as a constant column when supplied. Omit to return a
    point-in-time snapshot without that column (callers that don't care about
    snapshot dating shouldn't pay for the column).
    """
    if filters is None:
        filters = UniverseFilters()
    eligible = apply_liquidity_filters(candidates, filters=filters)
    top = select_top_n(eligible, filters=filters)

    # Project to the documented column order so downstream consumers can rely
    # on layout. Any extra columns the caller passed in are dropped here.
    output_cols = [
        "ticker",
        "options_adv_30d",
        "options_oi_30d",
        "atm_spread_pct_30d",
        "earnings_events_count",
        "rank",
    ]
    top = top.loc[:, output_cols].copy()

    if as_of is not None:
        top["as_of"] = as_of

    return top


def explain_exclusion(
    candidate_row: pd.Series,
    filters: UniverseFilters | None = None,
) -> list[str]:
    """Returns list of failed-gate names for a single candidate (empty if eligible).

    Useful for audit/log lines like 'AAPL excluded: atm_spread_pct_30d=0.10 > 0.08'.
    Gate names returned match the `UniverseFilters` field names: 'min_options_adv',
    'min_options_oi', 'max_atm_spread_pct', 'min_historical_events'. Order is
    stable (matches the order in `_GATE_SPECS`).
    """
    if filters is None:
        filters = UniverseFilters()

    missing = [c for c in REQUIRED_COLUMNS if c not in candidate_row.index]
    if missing:
        raise ValueError(
            f"candidate_row missing required fields: {missing} "
            f"(required: {REQUIRED_COLUMNS})"
        )

    thresholds: dict[str, float] = {
        "min_options_adv": filters.min_options_adv,
        "min_options_oi": filters.min_options_oi,
        "max_atm_spread_pct": filters.max_atm_spread_pct,
        "min_historical_events": filters.min_historical_events,
    }

    failed: list[str] = []
    for gate_name, col, direction in _GATE_SPECS:
        value = candidate_row[col]
        threshold = thresholds[gate_name]
        # NaN fails every gate.
        if pd.isna(value):
            failed.append(gate_name)
            continue
        if direction == "ge" and not (value >= threshold):
            failed.append(gate_name)
        elif direction == "le" and not (value <= threshold):
            failed.append(gate_name)
    return failed
