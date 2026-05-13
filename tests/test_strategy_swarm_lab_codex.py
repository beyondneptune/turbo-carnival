from __future__ import annotations

from collections import deque

import pandas as pd

from experiments.strategy_swarm_lab import compute_signal


def test_volume_fade_longs_low_volume_drop() -> None:
    history = deque([100.0, 100.0, 100.0, 99.80])
    row = pd.Series({"volume_z": -0.50})

    score, reason = compute_signal(
        "volume_fade",
        history,
        row=row,
        params={"max_volume_z": 0.10, "min_abs_move_bps": 6.0},
    )

    assert score > 0
    assert "volz=-0.50" in reason


def test_volume_fade_ignores_high_volume_drop() -> None:
    history = deque([100.0, 100.0, 100.0, 99.80])
    row = pd.Series({"volume_z": 2.00})

    score, _ = compute_signal(
        "volume_fade",
        history,
        row=row,
        params={"max_volume_z": 0.10, "min_abs_move_bps": 6.0},
    )

    assert score == 0.0


def test_vwap_reversion_shorts_price_above_vwap() -> None:
    history = deque([100.0, 100.0, 100.0, 100.05])
    row = pd.Series({"volume_z": -0.25, "vwap_dist": 0.0020})

    score, reason = compute_signal(
        "vwap_reversion",
        history,
        row=row,
        params={"max_volume_z": 0.25, "min_vwap_dist_bps": 10.0},
    )

    assert score < 0
    assert "vwap=+20.0" in reason


def test_dispersion_reversion_fades_basket_outlier() -> None:
    histories = {
        "AAPL": deque([100.0] * 10 + [102.0]),
        "AMD": deque([100.0] * 11),
        "META": deque([100.0] * 11),
        "NVDA": deque([100.0] * 11),
        "QQQ": deque([100.0] * 11),
    }

    score, reason = compute_signal(
        "dispersion_reversion",
        histories["AAPL"],
        sym="AAPL",
        all_histories=histories,
        params={"lookback_bars": 10, "min_z": 1.20, "min_dispersion_bps": 5.0},
    )

    assert score < 0
    assert "z=+" in reason
