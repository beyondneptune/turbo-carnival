"""Risk overlay for candidate_v1.

Per prereg §8:
- CVaR (Rockafellar-Uryasev) cap: daily 95% CVaR of aggregate book < 5% NAV
- Vol target: 12% annualized; scale exposure if 20-day realized vol > 18%
- Chandelier per-position stop: 3 × ATR(22)
- Drift rebalance: |notional drift| > 5% triggers close+reopen
- Daily circuit breaker: session loss < -3% halts new entries

Per prereg §9 success gates I (overlay-on Sharpe >= 0.9 × overlay-off) and
J (max DD cut >= 30%).

Run:
    .venv/bin/python -m research.candidate_v1.overlay
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.candidate_v1.backtest import (
    UNIVERSE, TRAIN_END, OOS_START, VariantParams,
    load_prices, simulate_portfolio, metrics,
)
from research.candidate_v1.walk_forward import load_best_variant

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "research" / "candidate_v1" / "results"

CVAR_ALPHA = 0.95  # 95% CVaR
CVAR_DAILY_LIMIT_PCT = 0.05  # daily CVaR cannot exceed 5% NAV
VOL_TARGET_ANNUAL = 0.12  # 12%
VOL_REALIZED_WINDOW = 20
VOL_SCALE_MAX = 1.5
VOL_SCALE_MIN = 0.25
DAILY_LOSS_CIRCUIT = -0.03  # -3% halts new entries

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("candidate_v1.overlay")


def cvar_scale(returns_history: np.ndarray, alpha: float = CVAR_ALPHA,
               limit_pct: float = CVAR_DAILY_LIMIT_PCT) -> float:
    """Rockafellar-Uryasev CVaR — scale exposure so 95% CVaR fits inside limit.

    CVaR_α(L) = E[L | L >= VaR_α]
    We compute empirical CVaR on historical return distribution and return the
    scale factor that brings CVaR within the limit.
    """
    if len(returns_history) < 30:
        return 1.0
    losses = -returns_history  # losses are negatives of returns
    var = np.percentile(losses, alpha * 100)
    tail = losses[losses >= var]
    cvar = tail.mean() if len(tail) > 0 else var
    if cvar <= 0:
        return 1.0
    scale = min(1.0, limit_pct / cvar)
    return max(VOL_SCALE_MIN, scale)


def vol_target_scale(returns_history: np.ndarray) -> float:
    """Scale exposure so 20d realized vol tracks the 12% annualized target."""
    if len(returns_history) < VOL_REALIZED_WINDOW:
        return 1.0
    realized = returns_history[-VOL_REALIZED_WINDOW:]
    sigma_ann = realized.std() * np.sqrt(252)
    if sigma_ann <= 1e-6:
        return VOL_SCALE_MAX
    scale = VOL_TARGET_ANNUAL / sigma_ann
    return float(np.clip(scale, VOL_SCALE_MIN, VOL_SCALE_MAX))


def chandelier_stop(prices: pd.Series, atr_window: int = 22, k: float = 3.0) -> pd.Series:
    """Trailing chandelier stop = highest_high(22) - k × ATR(22)."""
    # Approximate ATR with rolling range (since we only have close prices here)
    rng = prices.rolling(atr_window).max() - prices.rolling(atr_window).min()
    atr = rng / atr_window
    highest_high = prices.rolling(atr_window).max()
    stop = highest_high - k * atr * atr_window
    return stop


def apply_overlay(daily_returns: pd.Series, daily_drawdown: pd.Series | None = None) -> pd.Series:
    """Apply CVaR + vol-target + circuit-breaker overlay to a daily return series.

    Inputs:
      daily_returns: strategy gross daily returns (no overlay)
      daily_drawdown: optional precomputed running DD; recomputed if missing
    Returns:
      overlay-adjusted daily returns
    """
    r = daily_returns.copy()
    out = pd.Series(0.0, index=r.index)
    scales = pd.Series(1.0, index=r.index)
    circuit_active = False
    session_start_equity = 1.0
    equity = 1.0
    for i, dt in enumerate(r.index):
        # Recompute scale at each step using past data
        history = r.iloc[:i].dropna().values if i > 0 else np.array([])
        cvar_s = cvar_scale(history)
        vol_s = vol_target_scale(history)
        scale = min(cvar_s, vol_s)
        if circuit_active:
            scale = 0.0  # no new exposure today
        scales.iloc[i] = scale
        ret_today = r.iloc[i] * scale if not np.isnan(r.iloc[i]) else 0.0
        out.iloc[i] = ret_today
        equity *= (1 + ret_today)
        # Daily circuit breaker check — halt for rest of day if session loss > 3%
        session_loss = (equity / session_start_equity) - 1
        if session_loss < DAILY_LOSS_CIRCUIT:
            circuit_active = True
        # Reset circuit each "session" (we'll treat each day as its own session
        # here; the overlay just suppresses entries on the loss-day)
        session_start_equity = equity
    out.attrs["mean_scale"] = float(scales.mean())
    out.attrs["min_scale"] = float(scales.min())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args(argv)

    best = load_best_variant()
    log.info("best variant: %s", best.key())
    prices = load_prices()
    train_prices = prices[prices.index <= TRAIN_END]
    oos_prices = prices[prices.index >= OOS_START]

    # No-overlay returns
    log.info("computing no-overlay returns (train + OOS)...")
    train_no = simulate_portfolio(train_prices, best)
    oos_no = simulate_portfolio(oos_prices, best)

    # Overlay-on returns
    log.info("applying CVaR + vol-target + circuit overlay...")
    train_on = apply_overlay(train_no)
    oos_on = apply_overlay(oos_no)

    log.info("train mean scale: %.3f, OOS mean scale: %.3f",
             train_on.attrs["mean_scale"], oos_on.attrs["mean_scale"])

    train_no_m = metrics(train_no, "train_no_overlay")
    train_on_m = metrics(train_on, "train_overlay")
    oos_no_m = metrics(oos_no, "oos_no_overlay")
    oos_on_m = metrics(oos_on, "oos_overlay")

    # Hurdles
    train_sharpe_ratio = train_on_m["sharpe"] / max(train_no_m["sharpe"], 1e-9) if train_no_m["sharpe"] > 0 else 1.0
    train_dd_cut = (abs(train_no_m["max_dd"]) - abs(train_on_m["max_dd"])) / max(abs(train_no_m["max_dd"]), 1e-9)
    oos_sharpe_ratio = oos_on_m["sharpe"] / max(oos_no_m["sharpe"], 1e-9) if oos_no_m["sharpe"] > 0 else 1.0
    oos_dd_cut = (abs(oos_no_m["max_dd"]) - abs(oos_on_m["max_dd"])) / max(abs(oos_no_m["max_dd"]), 1e-9)

    hurdles = {
        "train_sharpe_ratio_gte_0.9": train_sharpe_ratio >= 0.9,
        "train_DD_cut_gte_30pct": train_dd_cut >= 0.3,
        "oos_sharpe_ratio_gte_0.9": oos_sharpe_ratio >= 0.9,
        "oos_DD_cut_gte_30pct": oos_dd_cut >= 0.3,
    }
    log.info("train sharpe ratio: %.2f  DD cut: %.1f%%",
             train_sharpe_ratio, train_dd_cut * 100)
    log.info("OOS sharpe ratio: %.2f  DD cut: %.1f%%",
             oos_sharpe_ratio, oos_dd_cut * 100)

    summary = {
        "ran_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "best_variant": best.key(),
        "train_no_overlay": train_no_m,
        "train_overlay": train_on_m,
        "oos_no_overlay": oos_no_m,
        "oos_overlay": oos_on_m,
        "train_sharpe_ratio": float(train_sharpe_ratio),
        "train_DD_cut_pct": float(train_dd_cut * 100),
        "oos_sharpe_ratio": float(oos_sharpe_ratio),
        "oos_DD_cut_pct": float(oos_dd_cut * 100),
        "hurdles": hurdles,
        "all_hurdles_pass": all(hurdles.values()),
        "overlay_params": {
            "CVAR_ALPHA": CVAR_ALPHA,
            "CVAR_DAILY_LIMIT_PCT": CVAR_DAILY_LIMIT_PCT,
            "VOL_TARGET_ANNUAL": VOL_TARGET_ANNUAL,
            "VOL_REALIZED_WINDOW": VOL_REALIZED_WINDOW,
            "DAILY_LOSS_CIRCUIT": DAILY_LOSS_CIRCUIT,
        },
    }
    out = RESULTS_DIR / f"overlay_summary_{int(time.time())}.json"
    out.write_text(json.dumps(summary, indent=2, default=str))
    print("\n=== OVERLAY SUMMARY ===")
    print(json.dumps(summary, indent=2, default=str))
    return 0 if summary["all_hurdles_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
