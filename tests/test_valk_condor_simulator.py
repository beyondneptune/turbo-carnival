from __future__ import annotations

import pandas as pd
import pytest

from experiments.valk_condor_simulator import (
    IronCondorSpec,
    condor_debit_to_close,
    load_condor_spec_json,
    mark_condor,
    max_loss_usd,
    simulate_condor_path,
    terminal_pnl_usd,
)


def _valk_spec(entry_credit: float | None = 0.54) -> IronCondorSpec:
    return IronCondorSpec(
        long_put=691.0,
        short_put=696.0,
        short_call=790.0,
        long_call=795.0,
        quantity=44,
        entry_credit=entry_credit,
    )


def test_max_loss_matches_credit_net_width() -> None:
    spec = _valk_spec()
    assert max_loss_usd(spec, 0.54) == pytest.approx((5.0 - 0.54) * 100 * 44)


def test_terminal_pnl_inside_body_keeps_credit() -> None:
    spec = _valk_spec()
    assert terminal_pnl_usd(spec, 738.0, 0.54) == pytest.approx(0.54 * 100 * 44)


def test_terminal_pnl_beyond_call_wing_hits_max_loss() -> None:
    spec = _valk_spec()
    assert terminal_pnl_usd(spec, 820.0, 0.54) == pytest.approx(-max_loss_usd(spec, 0.54))


def test_condor_debit_to_close_is_positive() -> None:
    spec = _valk_spec()
    debit = condor_debit_to_close(spec, spot=738.0, days_to_expiry=32.0, iv=0.18)
    assert debit > 0.0


def test_mark_condor_returns_position_level_greeks() -> None:
    spec = _valk_spec()
    mark = mark_condor(spec, spot=738.0, days_to_expiry=32.0, iv=0.18, entry_credit=0.54)

    assert mark.spot == 738.0
    assert mark.days_to_expiry == 32.0
    assert mark.debit_to_close > 0.0
    assert mark.theta_per_day > 0.0
    assert mark.vega_per_vol_point < 0.0


def test_simulate_condor_path_terminal_risk_worsens_near_short_call() -> None:
    spec = _valk_spec()
    path = pd.DataFrame(
        {
            "t": pd.to_datetime(["2026-05-11T14:00:00Z", "2026-05-12T14:00:00Z", "2026-05-13T14:00:00Z"]),
            "close": [738.0, 790.0, 800.0],
        }
    )

    marks, summary = simulate_condor_path(path, spec, dte_at_entry=32.0, iv=0.18)

    assert len(marks) == 3
    assert marks["terminal_pnl_usd"].iloc[-1] < marks["terminal_pnl_usd"].iloc[0]
    assert summary["max_loss_usd"] == pytest.approx((5.0 - 0.54) * 100 * 44)


def test_load_condor_spec_json(tmp_path) -> None:
    p = tmp_path / "valk.json"
    p.write_text(
        """{
          "kind": "iron_condor",
          "underlying": "SPY",
          "dte_at_entry": 32,
          "qty": 44,
          "entry_credit": 0.54,
          "strikes": {
            "long_put": 691,
            "short_put": 696,
            "short_call": 790,
            "long_call": 795
          }
        }"""
    )

    spec, dte = load_condor_spec_json(p)

    assert spec == _valk_spec()
    assert dte == 32.0
