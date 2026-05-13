"""SVXY parameter sweep — Backtest Agent A14.

Goal: find the defensible balance between the aggressive variant that produced
the headline +17.01% / Sharpe 0.66 number (original walk-forward: threshold=1.0,
size=100%, no circuit breaker, 0 cost, 0% cash, 2019-2026) and the conservative
iter_5 variant that collapsed to -0.63% / Sharpe -0.08 (threshold=0.92, size=25%,
VIX>30 OR +50% breaker, 1bp cost, SGOV ~4% cash, 2019-2026).

The forensic audit
(/Users/weeoto/turbo-carnival/research/results/svxy_backtest_forensic_audit.md)
established that the gap is driven by the *spec* parameters, not lookahead or
data integrity. So we now sweep:

- threshold: 0.92, 0.95, 0.97, 1.00 (4 levels)
- size:      0.25, 0.50, 1.00       (3 levels)
- CB:        NONE, VIX>30, VIX>40   (3 levels)
=> 36 combos

Each combo runs:
  - 2019-01-01 to today, post -0.5x leverage SVXY only
  - shift(1) on signal (no lookahead)
  - 1 bp cost per change in SVXY weight (per leg)
  - cash leg earns SGOV daily return where available; else assume ~4% APY
    (constant daily return = (1.04)^(1/252) - 1) to mimic SGOV behavior in
    the pre-2020 window. This matches the spirit of the iter_5 spec.

Output:
  - /Users/weeoto/turbo-carnival/research/results/svxy_sweep_full.csv
  - /Users/weeoto/turbo-carnival/research/results/svxy_sweep_top.json

Tail test (Volmageddon): the *best* sweet-spot combo (best Sharpe with
max_dd <= 20%) is replayed against the 2018-02 window from the same cached
data; we report what THAT combo would have done in Feb 2018. SVXY was already
-0.5x leverage by Feb 2018 (the leverage cut took effect 2018-02-27, so the
2018-02-05 Volmageddon day itself was still -1x; we flag the period instead of
re-scaling -- the result is an honest worst-case lower bound).
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/weeoto/turbo-carnival/research")
CACHE_PATH = ROOT / "data" / "cache" / "vix_svxy_daily.parquet"
OUT_CSV = ROOT / "results" / "svxy_sweep_full.csv"
OUT_TOP_JSON = ROOT / "results" / "svxy_sweep_top.json"

SWEEP_START = "2019-01-01"
TAIL_START = "2018-02-01"
TAIL_END = "2018-04-01"

THRESHOLDS = [0.92, 0.95, 0.97, 1.00]
SIZES = [0.25, 0.50, 1.00]
# CB[None] = no circuit breaker
# CB[30]   = flatten when VIX > 30
# CB[40]   = flatten when VIX > 40
CB_LEVELS = [None, 30.0, 40.0]

COST_BPS = 1.0  # 1 bp per traded notional change
SGOV_FALLBACK_APY = 0.04  # ~4% annual, used pre-SGOV (before 2020-06)


@dataclass
class Combo:
    threshold: float
    size: float
    cb: float | None  # None = no CB, else VIX > cb triggers flatten


def cb_label(cb: float | None) -> str:
    return "NONE" if cb is None else f"VIX>{int(cb)}"


def load_panel() -> pd.DataFrame:
    df = pd.read_parquet(CACHE_PATH)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df[["VIX", "VIX3M", "SVXY", "SGOV"]].copy()
    df = df.sort_index()
    # drop rows missing the index-derived inputs
    df = df[df["VIX"].notna() & df["VIX3M"].notna() & df["SVXY"].notna()]
    # synth SGOV daily return for pre-SGOV days
    sgov_ret = df["SGOV"].pct_change()
    sgov_daily_fallback = (1.0 + SGOV_FALLBACK_APY) ** (1.0 / 252.0) - 1.0
    sgov_ret = sgov_ret.fillna(sgov_daily_fallback)
    df["SGOV_ret"] = sgov_ret
    df["SVXY_ret"] = df["SVXY"].pct_change().fillna(0.0)
    df["ratio"] = df["VIX"] / df["VIX3M"]
    return df


def run_one(panel: pd.DataFrame, combo: Combo) -> dict:
    """Run a single backtest combo on panel slice. Returns metrics dict.

    Position rule (decided at close of day t-1, applied to return on day t):
      target_svxy_w[t-1] = size  if (ratio[t-1] < threshold) AND not CB[t-1]
                          else 0
      target_sgov_w[t-1] = 1 - target_svxy_w[t-1]
      pos_svxy_w[t]      = target_svxy_w[t-1]   (i.e. shift(1))
    """
    df = panel.copy()
    contango = df["ratio"] < combo.threshold
    if combo.cb is None:
        cb_active = pd.Series(False, index=df.index)
    else:
        cb_active = df["VIX"] > combo.cb
    target_svxy_w = np.where(contango & ~cb_active, combo.size, 0.0)
    df["target_svxy_w"] = target_svxy_w
    df["pos_svxy_w"] = df["target_svxy_w"].shift(1).fillna(0.0)
    df["pos_sgov_w"] = 1.0 - df["pos_svxy_w"]

    weight_chg = df["pos_svxy_w"].diff().abs().fillna(df["pos_svxy_w"].abs())
    df["rotation_cost"] = weight_chg * (COST_BPS / 1e4)

    df["gross_ret"] = (
        df["pos_svxy_w"] * df["SVXY_ret"].fillna(0.0)
        + df["pos_sgov_w"] * df["SGOV_ret"].fillna(0.0)
    )
    df["net_ret"] = df["gross_ret"] - df["rotation_cost"]
    df["portfolio"] = (1.0 + df["net_ret"].fillna(0.0)).cumprod()

    daily = df["net_ret"].dropna()
    n = len(daily)
    years = n / 252.0
    total_ret = df["portfolio"].iloc[-1] / df["portfolio"].iloc[0] - 1.0 if len(df) else 0.0
    ann_ret = (1.0 + total_ret) ** (1.0 / years) - 1.0 if years > 0 else 0.0
    ann_vol = daily.std(ddof=0) * math.sqrt(252) if n else 0.0
    sharpe = (daily.mean() * 252) / ann_vol if ann_vol > 0 else 0.0
    roll_max = df["portfolio"].cummax()
    drawdown = df["portfolio"] / roll_max - 1.0
    max_dd = drawdown.min()
    rotations = int((df["pos_svxy_w"].diff().fillna(0.0).abs() > 1e-9).sum())
    days_active = int((df["pos_svxy_w"] > 0.0).sum())
    pct_active = days_active / n if n else 0.0
    cb_events = int(cb_active.sum())

    # ann_return/max_dd ratio — bigger = more return per unit of drawdown
    ret_dd_ratio = (ann_ret * 100.0) / abs(max_dd * 100.0) if max_dd < 0 else float("inf")

    return {
        "threshold": combo.threshold,
        "size_pct": int(round(combo.size * 100)),
        "cb": cb_label(combo.cb),
        "trading_days": int(n),
        "years": round(years, 3),
        "total_return_pct": round(total_ret * 100, 4),
        "ann_return_pct": round(ann_ret * 100, 4),
        "ann_vol_pct": round(ann_vol * 100, 4),
        "sharpe": round(sharpe, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
        "ret_dd_ratio": round(ret_dd_ratio, 4),
        "num_rotations": rotations,
        "rotations_per_year": round(rotations / years, 3) if years > 0 else 0.0,
        "days_active": days_active,
        "pct_days_active": round(pct_active * 100, 2),
        "cb_event_days": cb_events,
    }


def tail_test(panel_full: pd.DataFrame, combo: Combo) -> dict:
    """Apply the chosen combo to Feb-Mar 2018 (Volmageddon)."""
    tail = panel_full.loc[TAIL_START:TAIL_END].copy()
    if tail.empty:
        return {"error": "no Volmageddon data in cache"}
    # rebuild SGOV_ret with fallback for this slice
    sgov_daily_fb = (1.0 + SGOV_FALLBACK_APY) ** (1.0 / 252.0) - 1.0
    sgov_ret = tail["SGOV"].pct_change() if "SGOV" in tail.columns else pd.Series(index=tail.index, dtype=float)
    sgov_ret = sgov_ret.fillna(sgov_daily_fb)
    tail["SGOV_ret"] = sgov_ret
    tail["SVXY_ret"] = tail["SVXY"].pct_change().fillna(0.0)
    tail["ratio"] = tail["VIX"] / tail["VIX3M"]
    contango = tail["ratio"] < combo.threshold
    if combo.cb is None:
        cb_active = pd.Series(False, index=tail.index)
    else:
        cb_active = tail["VIX"] > combo.cb
    target_svxy_w = np.where(contango & ~cb_active, combo.size, 0.0)
    tail["target_svxy_w"] = target_svxy_w
    tail["pos_svxy_w"] = tail["target_svxy_w"].shift(1).fillna(0.0)
    tail["pos_sgov_w"] = 1.0 - tail["pos_svxy_w"]
    weight_chg = tail["pos_svxy_w"].diff().abs().fillna(tail["pos_svxy_w"].abs())
    tail["rotation_cost"] = weight_chg * (COST_BPS / 1e4)
    tail["gross_ret"] = (
        tail["pos_svxy_w"] * tail["SVXY_ret"].fillna(0.0)
        + tail["pos_sgov_w"] * tail["SGOV_ret"].fillna(0.0)
    )
    tail["net_ret"] = tail["gross_ret"] - tail["rotation_cost"]
    tail["portfolio"] = (1.0 + tail["net_ret"].fillna(0.0)).cumprod()
    rollmax = tail["portfolio"].cummax()
    dd = tail["portfolio"] / rollmax - 1.0
    total_ret = tail["portfolio"].iloc[-1] - 1.0
    max_dd = dd.min()
    worst_day = tail["net_ret"].min()
    worst_day_date = str(tail["net_ret"].idxmin().date()) if not tail["net_ret"].isna().all() else "n/a"
    # was the strategy IN SVXY on or after the spike?
    svxy_spike_date = pd.Timestamp("2018-02-05")
    if svxy_spike_date in tail.index:
        pos_on_spike = float(tail.loc[svxy_spike_date, "pos_svxy_w"])
        ret_on_spike = float(tail.loc[svxy_spike_date, "net_ret"])
    else:
        # closest trading day
        idx = tail.index.searchsorted(svxy_spike_date)
        idx = min(idx, len(tail.index) - 1)
        nearest = tail.index[idx]
        pos_on_spike = float(tail.loc[nearest, "pos_svxy_w"])
        ret_on_spike = float(tail.loc[nearest, "net_ret"])
    # the next-day reaction (Feb 6 / Feb 7 fallout)
    feb6 = pd.Timestamp("2018-02-06")
    if feb6 in tail.index:
        pos_feb6 = float(tail.loc[feb6, "pos_svxy_w"])
        ret_feb6 = float(tail.loc[feb6, "net_ret"])
    else:
        pos_feb6 = float("nan")
        ret_feb6 = float("nan")

    return {
        "tail_window": f"{TAIL_START} to {TAIL_END}",
        "tail_trading_days": int(len(tail)),
        "tail_total_return_pct": round(total_ret * 100, 4),
        "tail_max_dd_pct": round(max_dd * 100, 4),
        "tail_worst_day_pct": round(worst_day * 100, 4),
        "tail_worst_day_date": worst_day_date,
        "feb_5_2018_svxy_weight": round(pos_on_spike, 4),
        "feb_5_2018_pnl_pct": round(ret_on_spike * 100, 4),
        "feb_6_2018_svxy_weight": round(pos_feb6, 4),
        "feb_6_2018_pnl_pct": round(ret_feb6 * 100, 4),
        "note": (
            "SVXY was -1x pre 2018-02-27, then cut to -0.5x. Feb 5-6 2018 "
            "data reflects the -1x regime; treat as worst-case lower bound."
        ),
    }


def main() -> None:
    print("=" * 78)
    print("SVXY parameter sweep (Backtest Agent A14)")
    print("=" * 78)
    panel_full = load_panel()
    print(f"Cache loaded: {len(panel_full)} rows, {panel_full.index.min().date()} to {panel_full.index.max().date()}")

    # main sweep window
    panel_sweep = panel_full.loc[SWEEP_START:].copy()
    print(f"Sweep window: {panel_sweep.index.min().date()} to {panel_sweep.index.max().date()} ({len(panel_sweep)} rows)")

    rows = []
    combos = list(itertools.product(THRESHOLDS, SIZES, CB_LEVELS))
    print(f"Running {len(combos)} combos ...")
    for thr, sz, cb in combos:
        combo = Combo(threshold=thr, size=sz, cb=cb)
        m = run_one(panel_sweep, combo)
        rows.append(m)
        print(
            f"  thr={thr:.2f} sz={int(sz*100):3d}% cb={cb_label(cb):>6s}: "
            f"ann={m['ann_return_pct']:+7.2f}% vol={m['ann_vol_pct']:6.2f}% "
            f"shr={m['sharpe']:+6.3f} dd={m['max_drawdown_pct']:+7.2f}% "
            f"rot/y={m['rotations_per_year']:6.2f} act={m['pct_days_active']:5.1f}%"
        )
    df = pd.DataFrame(rows)
    df = df.sort_values(by=["sharpe", "ann_return_pct"], ascending=[False, False])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nWrote full sweep CSV: {OUT_CSV} ({len(df)} rows)")

    # top 5 by Sharpe (unfiltered)
    top_sharpe = df.head(5).to_dict(orient="records")
    # top 5 by ret/dd ratio
    df_ratio = df.sort_values(by=["ret_dd_ratio"], ascending=False)
    top_ratio = df_ratio.head(5).to_dict(orient="records")

    # sweet spot: best Sharpe among combos with |max_dd| <= 20%
    safe = df[df["max_drawdown_pct"] >= -20.0].copy()
    if safe.empty:
        # fall back to looser DD limit
        print("WARN: no combo has |max_dd| <= 20%; relaxing to <= 30%")
        safe = df[df["max_drawdown_pct"] >= -30.0].copy()
    safe = safe.sort_values(by=["sharpe", "ann_return_pct"], ascending=[False, False])
    sweet = safe.iloc[0].to_dict() if not safe.empty else None

    # tail test on the sweet spot
    tail = None
    if sweet is not None:
        cb_val = None if sweet["cb"] == "NONE" else float(sweet["cb"].replace("VIX>", ""))
        sweet_combo = Combo(
            threshold=float(sweet["threshold"]),
            size=float(sweet["size_pct"]) / 100.0,
            cb=cb_val,
        )
        tail = tail_test(panel_full, sweet_combo)
        print("\nTail (Volmageddon Feb-Mar 2018) replay on SWEET SPOT:")
        for k, v in tail.items():
            print(f"  {k}: {v}")

    out = {
        "sweep_window": f"{SWEEP_START} to {panel_sweep.index.max().date()}",
        "n_combos": len(rows),
        "top5_by_sharpe": top_sharpe,
        "top5_by_ret_dd_ratio": top_ratio,
        "sweet_spot_max_dd_le_20": sweet,
        "tail_test_on_sweet_spot": tail,
        "sweep_csv": str(OUT_CSV),
        "params": {
            "thresholds": THRESHOLDS,
            "sizes": SIZES,
            "cb_levels": [cb_label(c) for c in CB_LEVELS],
            "cost_bps": COST_BPS,
            "sgov_fallback_apy_pre_2020_06": SGOV_FALLBACK_APY,
        },
    }
    OUT_TOP_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TOP_JSON, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWrote sweep top JSON: {OUT_TOP_JSON}")

    print("\nTop 5 by Sharpe:")
    for r in top_sharpe:
        print(
            f"  thr={r['threshold']:.2f} sz={r['size_pct']:3d}% cb={r['cb']:>6s} "
            f"-> shr={r['sharpe']:+.3f} ann={r['ann_return_pct']:+.2f}% dd={r['max_drawdown_pct']:+.2f}%"
        )
    print("\nTop 5 by ret/|dd|:")
    for r in top_ratio:
        print(
            f"  thr={r['threshold']:.2f} sz={r['size_pct']:3d}% cb={r['cb']:>6s} "
            f"-> ratio={r['ret_dd_ratio']:.3f} ann={r['ann_return_pct']:+.2f}% dd={r['max_drawdown_pct']:+.2f}% shr={r['sharpe']:+.3f}"
        )

    if sweet is not None:
        print("\nSWEET SPOT (best Sharpe with |max_dd| <= 20%):")
        print(
            f"  thr={sweet['threshold']:.2f} sz={sweet['size_pct']:3d}% cb={sweet['cb']:>6s} "
            f"-> shr={sweet['sharpe']:+.3f} ann={sweet['ann_return_pct']:+.2f}% dd={sweet['max_drawdown_pct']:+.2f}%"
        )


if __name__ == "__main__":
    main()
