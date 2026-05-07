"""Tests for the Phase 2 deterministic kill switch.

Coverage:
- ``check_intraday_loss``: below threshold, single breach, multi-day breach
  (returns FIRST breach date), empty/short history.
- ``check_cycle_drawdown``: monotone-up, breach at 15.5%, no-breach at 14.5%,
  multi-peak (drawdown vs running max).
- ``check_cash_buffer``: above, below, exactly-at threshold.
- ``evaluate_kill_switch``: healthy, intraday-only, drawdown-only, both,
  cash-only, and determinism (re-running with same data yields equal status).
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from risk.kill_switch import (
    KillSwitchStatus,
    check_cash_buffer,
    check_cycle_drawdown,
    check_intraday_loss,
    evaluate_kill_switch,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dates(n: int, start: date = date(2026, 5, 6)) -> list[date]:
    return [start + timedelta(days=i) for i in range(n)]


def _nav_df(navs: list[float], lows: list[float] | None = None) -> pd.DataFrame:
    """Build a NAV history frame with intraday_low defaulting to nav (no intraday move)."""
    if lows is None:
        lows = list(navs)
    return pd.DataFrame(
        {
            "date": _dates(len(navs)),
            "nav": navs,
            "intraday_low": lows,
        }
    )


# ---------------------------------------------------------------------------
# check_intraday_loss
# ---------------------------------------------------------------------------


def test_intraday_loss_below_threshold_not_triggered():
    # Five days; max intraday loss is 6%, below 8% threshold.
    navs = [10000, 10100, 10200, 10100, 10050]
    # Day 4: open=10200, low=10200*(1-0.06)=9588 -> 6% loss
    lows = [10000, 10100, 10200, 9588.0, 10050]
    df = _nav_df(navs, lows)
    triggered, breach_date, details = check_intraday_loss(df)
    assert triggered is False
    assert breach_date is None
    # max_intraday_loss_pct should be ~0.06 (within float tolerance)
    assert details["max_intraday_loss_pct"] == pytest.approx(0.06, abs=1e-9)


def test_intraday_loss_single_breach_triggered():
    # Day 2 (idx=2): open = nav[idx=1] = 10100, low = 10100*(1-0.09) = 9191
    # -> (10100-9191)/10100 = 0.09 (9% loss)
    navs = [10000, 10100, 10200, 9500, 9600]
    lows = [10000, 10100, 9191.0, 9500, 9600]
    df = _nav_df(navs, lows)
    triggered, breach_date, details = check_intraday_loss(df)
    assert triggered is True
    assert breach_date == date(2026, 5, 6) + timedelta(days=2)
    assert details["max_intraday_loss_pct"] == pytest.approx(0.09, abs=1e-9)
    assert details["breach_date"] == breach_date.isoformat()


def test_intraday_loss_multi_breach_returns_first():
    # Two distinct days breach; expect FIRST.
    # Day 2: open=10100, low=9090  -> (10100-9090)/10100 ≈ 0.10 (10% loss)
    # Day 4: open=10000, low=8500  -> (10000-8500)/10000 = 0.15 (15% loss)
    navs = [10000, 10100, 9500, 10000, 8800]
    lows = [10000, 9090, 9500, 10000, 8500]
    df = _nav_df(navs, lows)
    triggered, breach_date, details = check_intraday_loss(df)
    assert triggered is True
    # First breach is day index 1 (second day in the series).
    assert breach_date == date(2026, 5, 6) + timedelta(days=1)
    # max_intraday_loss_pct reflects the worst observed across all days
    assert details["max_intraday_loss_pct"] == pytest.approx(0.15, abs=1e-9)


def test_intraday_loss_empty_history():
    df = pd.DataFrame({"date": [], "nav": [], "intraday_low": []})
    triggered, breach_date, details = check_intraday_loss(df)
    assert triggered is False
    assert breach_date is None
    assert details == {}


def test_intraday_loss_single_row_history_cannot_breach():
    # Single row: no previous-day open, so cannot evaluate intraday.
    df = _nav_df([10000], [9000])  # would be 10% if there were an open
    triggered, breach_date, details = check_intraday_loss(df)
    assert triggered is False
    assert breach_date is None


# ---------------------------------------------------------------------------
# check_cycle_drawdown
# ---------------------------------------------------------------------------


def test_cycle_drawdown_monotone_up_not_triggered():
    df = _nav_df([10000, 10100, 10250, 10400, 10500])
    triggered, breach_date, details = check_cycle_drawdown(df)
    assert triggered is False
    assert breach_date is None
    assert details["max_drawdown_pct"] == pytest.approx(0.0, abs=1e-12)


def test_cycle_drawdown_15_5_pct_triggered():
    # Peak 11000, low 9300 -> (11000-9300)/11000 ≈ 0.1545
    df = _nav_df([10000, 10500, 11000, 10000, 9300])
    triggered, breach_date, details = check_cycle_drawdown(df)
    assert triggered is True
    assert breach_date == date(2026, 5, 6) + timedelta(days=4)
    assert details["max_drawdown_pct"] == pytest.approx(1700 / 11000, abs=1e-9)
    assert details["peak_nav"] == pytest.approx(11000.0, abs=1e-9)
    assert details["breach_nav"] == pytest.approx(9300.0, abs=1e-9)
    assert details["breach_date"] == breach_date.isoformat()


def test_cycle_drawdown_14_5_pct_not_triggered():
    # Peak 11000, low 9405 -> (11000-9405)/11000 = 0.145
    df = _nav_df([10000, 10500, 11000, 10000, 9405])
    triggered, breach_date, details = check_cycle_drawdown(df)
    assert triggered is False
    assert breach_date is None
    assert details["max_drawdown_pct"] == pytest.approx(0.145, abs=1e-9)


def test_cycle_drawdown_multi_peak_uses_running_max():
    # First peak 10500, then drawdown then NEW peak 12000 then drop to 10100.
    # If measured from start (10000) -> only -0.49% to 10100; not breach.
    # From running peak 12000 -> (12000-10100)/12000 ≈ 0.158 -> breach.
    navs = [10000, 10300, 10500, 10200, 10800, 12000, 11500, 10100]
    df = _nav_df(navs)
    triggered, breach_date, details = check_cycle_drawdown(df)
    assert triggered is True
    # Last index = 7
    assert breach_date == date(2026, 5, 6) + timedelta(days=7)
    assert details["peak_nav"] == pytest.approx(12000.0)
    assert details["breach_nav"] == pytest.approx(10100.0)
    assert details["max_drawdown_pct"] == pytest.approx(1900 / 12000, abs=1e-9)


# ---------------------------------------------------------------------------
# check_cash_buffer
# ---------------------------------------------------------------------------


def test_cash_buffer_above_threshold_not_breached():
    breached, details = check_cash_buffer(cash_usd=2500, nav_usd=10000)
    assert breached is False
    assert details["cash_pct_of_nav"] == pytest.approx(0.25)


def test_cash_buffer_below_threshold_breached():
    breached, details = check_cash_buffer(cash_usd=1500, nav_usd=10000)
    assert breached is True
    assert details["cash_pct_of_nav"] == pytest.approx(0.15)


def test_cash_buffer_exactly_at_threshold_not_breached():
    # >= 20% is OK per spec.
    breached, details = check_cash_buffer(cash_usd=2000, nav_usd=10000)
    assert breached is False
    assert details["cash_pct_of_nav"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# evaluate_kill_switch
# ---------------------------------------------------------------------------


def test_evaluate_healthy_portfolio_all_green():
    # Monotone up, healthy cash.
    df = _nav_df([10000, 10050, 10100, 10200, 10300])
    status = evaluate_kill_switch(df, current_cash_usd=2500, current_nav_usd=10300)
    assert isinstance(status, KillSwitchStatus)
    assert status.suspended is False
    assert status.soft_block is False
    assert status.triggered_reasons == []
    assert status.triggered_at is None


def test_evaluate_intraday_breach_only():
    # Intraday breach on day 2; healthy cash; no cumulative drawdown.
    navs = [10000, 10100, 10050, 10100, 10200]
    # Day 2: open=10100, low=9282 -> 8.1% intraday
    lows = [10000, 10100, 9282, 10100, 10200]
    df = _nav_df(navs, lows)
    status = evaluate_kill_switch(df, current_cash_usd=3000, current_nav_usd=10200)
    assert status.suspended is True
    assert status.soft_block is False
    assert status.triggered_reasons == ["intraday_loss_8pct"]
    assert status.triggered_at == date(2026, 5, 6) + timedelta(days=2)


def test_evaluate_drawdown_breach_only():
    # Gradual drop so no single-day intraday loss reaches 8%, but cumulative
    # drawdown from peak 11000 to 9300 ≈ 15.45% does breach.
    # Per-day intraday loss = (prev_close - this_low)/prev_close:
    #   day 1: (10000 - 10500)/10000 = -0.05 (gain)
    #   day 2: (10500 - 11000)/10500 = -0.048 (gain)
    #   day 3: (11000 - 10300)/11000 ≈ 0.0636
    #   day 4: (10300 - 9300)/10300 ≈ 0.0971  -> would breach. So step gentler.
    # Use 5 steps down from 11000 -> 9300 with each step ≤ 7%.
    navs = [10000, 10500, 11000, 10300, 9700, 9300]
    df = _nav_df(navs)
    status = evaluate_kill_switch(df, current_cash_usd=3000, current_nav_usd=9300)
    assert status.suspended is True
    assert status.soft_block is False
    assert status.triggered_reasons == ["cycle_drawdown_15pct"]
    assert status.triggered_at == date(2026, 5, 6) + timedelta(days=5)


def test_evaluate_both_intraday_and_drawdown_uses_earliest_date():
    # Intraday breach on day 1; drawdown breach not until day 4.
    # Day 1: open=10000, low=9100 -> 9% intraday loss
    # Cumulative: peak 10100 at idx 2, low 8500 at idx 4 -> drawdown ≈ 15.84%
    navs = [10000, 9200, 10100, 9500, 8500]
    lows = [10000, 9100, 10100, 9500, 8500]
    df = _nav_df(navs, lows)
    status = evaluate_kill_switch(df, current_cash_usd=3000, current_nav_usd=8500)
    assert status.suspended is True
    assert "intraday_loss_8pct" in status.triggered_reasons
    assert "cycle_drawdown_15pct" in status.triggered_reasons
    # Earliest hard trigger date wins (intraday on day 1).
    assert status.triggered_at == date(2026, 5, 6) + timedelta(days=1)


def test_evaluate_cash_low_only_soft_block():
    df = _nav_df([10000, 10050, 10100, 10200, 10300])
    status = evaluate_kill_switch(df, current_cash_usd=1500, current_nav_usd=10300)
    assert status.suspended is False
    assert status.soft_block is True
    assert status.triggered_reasons == ["cash_buffer_below_20pct"]
    assert status.triggered_at is None


def test_evaluate_is_deterministic_same_data_same_status():
    """Stickiness from data alone: re-evaluating with the same history returns
    an equal status. There is no internal state that would let a second call
    "forget" an earlier breach."""
    navs = [10000, 10500, 11000, 10000, 9300]
    df = _nav_df(navs)
    first = evaluate_kill_switch(df, 3000, 9300)
    second = evaluate_kill_switch(df, 3000, 9300)
    # Frozen dataclasses with identical fields compare equal.
    assert first == second
    assert first.suspended is True
    assert first.triggered_at == second.triggered_at
    assert first.triggered_reasons == second.triggered_reasons


def test_evaluate_appending_safe_data_preserves_suspension():
    """Once a breach occurs, appending later 'good' rows must not unsuspend."""
    breach_navs = [10000, 10500, 11000, 10000, 9300]
    df_before = _nav_df(breach_navs)
    status_before = evaluate_kill_switch(df_before, 3000, 9300)
    assert status_before.suspended is True

    # Add four recovery days; the historical breach must still be detected.
    extended_navs = breach_navs + [9400, 9500, 9700, 10000]
    df_after = _nav_df(extended_navs)
    status_after = evaluate_kill_switch(df_after, 3000, 10000)
    assert status_after.suspended is True
    # Same trigger date: the breach point in history hasn't moved.
    assert status_after.triggered_at == status_before.triggered_at
    assert "cycle_drawdown_15pct" in status_after.triggered_reasons


def test_evaluate_status_is_immutable_frozen_dataclass():
    df = _nav_df([10000, 10050])
    status = evaluate_kill_switch(df, 3000, 10050)
    with pytest.raises((AttributeError, Exception)):
        status.suspended = True  # type: ignore[misc]
