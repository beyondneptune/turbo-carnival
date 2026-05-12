"""Vol-inverse notional calculator (ATHENA v1, VOLTGT specialist).

Pure function. Tested against today's 163 entries' realized-vol windows.
"""
from __future__ import annotations

import math


def compute_notional(
    realized_vol_5min_bps: float,
    base_notional: float = 12_000.0,
    target_vol_bps: float = 15.0,
    floor: float = 4_000.0,
    ceiling: float = 20_000.0,
) -> float:
    """Return $-notional for one entry, scaled inversely to realized vol.

    Args:
        realized_vol_5min_bps: stdev(log returns last 5 1-min bars) × sqrt(5)
                               × 10000. Use NaN sentinel `math.nan` for
                               first 5 min after open (then default applied).
        base_notional: base position size in USD ($12k default).
        target_vol_bps: vol level that yields base notional (15 bps default).
        floor, ceiling: hard min/max in USD.

    Returns:
        Notional in USD, clamped to [floor, ceiling].
    """
    if realized_vol_5min_bps is None or (isinstance(realized_vol_5min_bps, float) and math.isnan(realized_vol_5min_bps)):
        # First minutes of session or stale data — use half base
        return base_notional * 0.5
    if realized_vol_5min_bps <= 0:
        return ceiling
    raw = base_notional * (target_vol_bps / realized_vol_5min_bps)
    return max(floor, min(ceiling, raw))


def compute_realized_vol_bps(returns_5_bars: list[float]) -> float:
    """5-min realized vol in bps from a list of 5 1-min log returns."""
    import statistics
    if len(returns_5_bars) < 2:
        return math.nan
    s = statistics.pstdev(returns_5_bars)
    return s * math.sqrt(5.0) * 10_000.0


if __name__ == "__main__":
    # Smoke test
    print("vol 5 bps  →", compute_notional(5))     # quiet → ceiling
    print("vol 15 bps →", compute_notional(15))    # baseline → base_notional
    print("vol 30 bps →", compute_notional(30))    # busy → halved
    print("vol 100 bps→", compute_notional(100))   # chaos → floor
    print("nan        →", compute_notional(math.nan))  # default
    # Test realized_vol calc
    rets = [0.0005, -0.0003, 0.0008, -0.0002, 0.0004]
    print("rv from rets→", compute_realized_vol_bps(rets), "bps")
