"""Execute LLM-voted decisions against historical option pricing.

Uses Black-Scholes from `experiments/run_spy_core_strategy_grid.py` for prices.
Tracks open positions, daily P&L, NAV, drawdown.

Aggressive defaults: NAV-based sizing per LLM's tier, no defensive caps beyond
physical possibility (can't sell what NAV can't margin).
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_spy_core_strategy_grid import price_condor_value  # noqa: E402

from fullonriskmodelterminator.menu import (  # noqa: E402
    DTE_TO_DAYS,
    SIZE_TO_PCT,
    TradeDecision,
)

# Approx friction model (per leg, per side)
SLIP_PER_LEG = 0.03      # $0.03 bid/ask half-spread
COMMISSION_PER_LEG = 0.65  # IBKR retail
MULTIPLIER = 100


@dataclass
class OpenPosition:
    open_date: date
    expiry: date
    action: str
    legs: list[dict]   # each: {right: C/P, strike, qty, direction (long/short)}
    entry_credit: float     # net credit (positive) or debit (negative)
    max_loss_per_contract: float
    quantity: int           # contracts in this combo


@dataclass
class TradeStats:
    realized_pnl_usd: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    blowups: int = 0   # trades that went past max_loss * 2


def _bs_price(spot: float, strike: float, t: float, sigma: float, right: Literal["C", "P"],
               rfr: float = 0.045, div: float = 0.014) -> float:
    """Black-Scholes price for one leg, in dollars (per share)."""
    if t <= 0:
        if right == "C":
            return max(0.0, spot - strike)
        return max(0.0, strike - spot)
    sigma = max(sigma, 0.05)
    d1 = (math.log(spot / strike) + (rfr - div + 0.5 * sigma * sigma) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    from scipy.stats import norm  # local import to avoid hard dep at module load
    if right == "C":
        return spot * math.exp(-div * t) * norm.cdf(d1) - strike * math.exp(-rfr * t) * norm.cdf(d2)
    return strike * math.exp(-rfr * t) * norm.cdf(-d2) - spot * math.exp(-div * t) * norm.cdf(-d1)


def _round_strike(spot: float, offset_pct: float) -> float:
    """Round to nearest $1 strike (SPY has $1 strikes around ATM)."""
    return round(spot * (1 + offset_pct))


def construct_position(decision: TradeDecision, market: pd.Series, nav: float) -> OpenPosition | None:
    """Translate an LLM decision into a concrete option position."""
    if decision.action in ("no_trade", "close_all"):
        return None

    spot = float(market["spy_close"])
    sigma = float(market["iv"])
    days_to_expiry = DTE_TO_DAYS[decision.dte_bucket]
    t = max(days_to_expiry, 1) / 365.0
    expiry = pd.Timestamp(market.name).date() + timedelta(days=max(days_to_expiry, 1))

    atm = _round_strike(spot, 0.0)
    offset_strike = _round_strike(spot, decision.strike_offset_pct)
    wide_call = _round_strike(spot, 0.04)
    wide_put = _round_strike(spot, -0.04)
    near_call = _round_strike(spot, 0.02)
    near_put = _round_strike(spot, -0.02)
    far_call = _round_strike(spot, 0.06)
    far_put = _round_strike(spot, -0.06)

    risk_pct = SIZE_TO_PCT[decision.size_tier]
    risk_dollars = nav * risk_pct

    legs = []
    entry_credit = 0.0   # positive = received cash, negative = paid cash
    max_loss_per_contract = 0.0

    if decision.action == "long_call":
        price = _bs_price(spot, atm, t, sigma, "C")
        legs = [{"right": "C", "strike": atm, "qty": 1, "direction": "long"}]
        entry_credit = -price  # paid
        max_loss_per_contract = price * MULTIPLIER

    elif decision.action == "long_put":
        price = _bs_price(spot, atm, t, sigma, "P")
        legs = [{"right": "P", "strike": atm, "qty": 1, "direction": "long"}]
        entry_credit = -price
        max_loss_per_contract = price * MULTIPLIER

    elif decision.action == "long_straddle":
        c = _bs_price(spot, atm, t, sigma, "C")
        p = _bs_price(spot, atm, t, sigma, "P")
        legs = [
            {"right": "C", "strike": atm, "qty": 1, "direction": "long"},
            {"right": "P", "strike": atm, "qty": 1, "direction": "long"},
        ]
        entry_credit = -(c + p)
        max_loss_per_contract = (c + p) * MULTIPLIER

    elif decision.action == "short_strangle":
        c = _bs_price(spot, wide_call, t, sigma, "C")
        p = _bs_price(spot, wide_put, t, sigma, "P")
        legs = [
            {"right": "C", "strike": wide_call, "qty": 1, "direction": "short"},
            {"right": "P", "strike": wide_put, "qty": 1, "direction": "short"},
        ]
        entry_credit = (c + p)
        # Undefined risk — approximate max loss as 5% move × multiplier
        max_loss_per_contract = max(spot * 0.05 * MULTIPLIER, 200.0)

    elif decision.action == "short_iron_condor":
        sc = _bs_price(spot, near_call, t, sigma, "C")
        lc = _bs_price(spot, far_call, t, sigma, "C")
        sp = _bs_price(spot, near_put, t, sigma, "P")
        lp = _bs_price(spot, far_put, t, sigma, "P")
        legs = [
            {"right": "C", "strike": near_call, "qty": 1, "direction": "short"},
            {"right": "C", "strike": far_call, "qty": 1, "direction": "long"},
            {"right": "P", "strike": near_put, "qty": 1, "direction": "short"},
            {"right": "P", "strike": far_put, "qty": 1, "direction": "long"},
        ]
        credit = (sc - lc) + (sp - lp)
        entry_credit = credit
        wing = max(far_call - near_call, near_put - far_put)
        max_loss_per_contract = (wing - credit) * MULTIPLIER

    elif decision.action == "credit_put_spread":
        short_strike = _round_strike(spot, -0.02)
        long_strike = _round_strike(spot, -0.05)
        sp = _bs_price(spot, short_strike, t, sigma, "P")
        lp = _bs_price(spot, long_strike, t, sigma, "P")
        legs = [
            {"right": "P", "strike": short_strike, "qty": 1, "direction": "short"},
            {"right": "P", "strike": long_strike, "qty": 1, "direction": "long"},
        ]
        credit = sp - lp
        entry_credit = credit
        max_loss_per_contract = (short_strike - long_strike - credit) * MULTIPLIER

    elif decision.action == "credit_call_spread":
        short_strike = _round_strike(spot, 0.02)
        long_strike = _round_strike(spot, 0.05)
        sc = _bs_price(spot, short_strike, t, sigma, "C")
        lc = _bs_price(spot, long_strike, t, sigma, "C")
        legs = [
            {"right": "C", "strike": short_strike, "qty": 1, "direction": "short"},
            {"right": "C", "strike": long_strike, "qty": 1, "direction": "long"},
        ]
        credit = sc - lc
        entry_credit = credit
        max_loss_per_contract = (long_strike - short_strike - credit) * MULTIPLIER

    elif decision.action == "long_calendar":
        # Buy 30-DTE ATM, sell 7-DTE ATM (positive theta if SPY pins)
        short_t = max(7, 1) / 365.0
        long_t = max(30, 1) / 365.0
        short_price = _bs_price(spot, atm, short_t, sigma, "C")
        long_price = _bs_price(spot, atm, long_t, sigma, "C")
        legs = [
            {"right": "C", "strike": atm, "qty": 1, "direction": "short"},
            {"right": "C", "strike": atm, "qty": 1, "direction": "long"},
        ]
        net = long_price - short_price
        entry_credit = -net
        max_loss_per_contract = net * MULTIPLIER
    else:
        return None

    if max_loss_per_contract <= 0:
        max_loss_per_contract = max(100.0, abs(entry_credit) * MULTIPLIER)

    # Quantity scaled by NAV-based risk
    qty = max(1, int(risk_dollars // max_loss_per_contract))
    return OpenPosition(
        open_date=pd.Timestamp(market.name).date(),
        expiry=expiry,
        action=decision.action,
        legs=legs,
        entry_credit=entry_credit,
        max_loss_per_contract=max_loss_per_contract,
        quantity=qty,
    )


def mark_position(pos: OpenPosition, current_market: pd.Series) -> float:
    """Mark-to-market the position; return current value in USD."""
    today_date = pd.Timestamp(current_market.name).date()
    days_to_exp = (pos.expiry - today_date).days
    if days_to_exp <= 0:
        return _terminal_value(pos, current_market)
    t = days_to_exp / 365.0
    spot = float(current_market["spy_close"])
    sigma = float(current_market["iv"])
    cur_value = 0.0
    for leg in pos.legs:
        price = _bs_price(spot, leg["strike"], t, sigma, leg["right"])
        sign = 1.0 if leg["direction"] == "long" else -1.0
        cur_value += sign * price * leg["qty"] * MULTIPLIER
    return cur_value * pos.quantity


def _terminal_value(pos: OpenPosition, market_at_expiry: pd.Series) -> float:
    """Intrinsic value at expiry."""
    spot = float(market_at_expiry["spy_close"])
    intrinsic = 0.0
    for leg in pos.legs:
        if leg["right"] == "C":
            val = max(0.0, spot - leg["strike"])
        else:
            val = max(0.0, leg["strike"] - spot)
        sign = 1.0 if leg["direction"] == "long" else -1.0
        intrinsic += sign * val * leg["qty"] * MULTIPLIER
    return intrinsic * pos.quantity


def trade_pnl(pos: OpenPosition, close_value: float, close_reason: str = "expiry") -> float:
    """Compute realized P&L: entry credit + (close_value or -intrinsic) - friction."""
    n_legs = len(pos.legs) * pos.quantity
    friction = n_legs * 2 * (SLIP_PER_LEG * MULTIPLIER / 2 + COMMISSION_PER_LEG)
    # entry_credit is per-contract (per shares); convert to dollars
    entry_cash = pos.entry_credit * MULTIPLIER * pos.quantity
    return entry_cash + close_value - friction
