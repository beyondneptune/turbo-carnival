"""Phase 2 paper execution simulator -- deterministic order fill simulation.

Computes simulated fill price, commission, and slippage for each leg given
a market quote (bid/ask) and a cost-stress multiplier. Does NOT touch the
DB; consumers are responsible for persisting the resulting Trade/Leg rows.

Convention (signed costs):
    - When BUYING (long): we pay ask + slippage. fill = mid + half_spread*stress + slippage.
    - When SELLING (short): we receive bid - slippage. fill = mid - half_spread*stress - slippage.

Costs reported as POSITIVE dollar amounts (commission, spread cost, slippage).
The signed cash flow at entry (`net_credit_or_debit_usd`) is computed by:
    sum over legs of (sign * fill_price * quantity * multiplier)
    where sign = +1 for SHORT (we receive cash), -1 for LONG (we pay cash).
Then commissions/spread_costs/slippage are subtracted to produce the realized
net cash flow (positive => credit received, negative => debit paid).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


COMMISSION_PER_CONTRACT_USD = 0.65   # IBKR Pro
SLIPPAGE_PER_LEG_USD = 0.01          # default per-share slippage on top of spread


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LegOrder:
    """An order for a single leg.

    Attributes:
        instrument_type: 'call', 'put', or 'stock'.
        direction: 'long' (buy) or 'short' (sell).
        quantity: positive integer count of contracts (or shares for stock).
        multiplier: 100 for options, 1 for stock.
        intended_mid_price: decision-time mid price per share.
    """
    instrument_type: Literal['call', 'put', 'stock']
    direction: Literal['long', 'short']
    quantity: int
    multiplier: int
    intended_mid_price: float


@dataclass(frozen=True)
class Quote:
    """Bid-ask quote at fill time."""
    bid: float
    ask: float

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread(self) -> float:
        return self.ask - self.bid

    @property
    def half_spread(self) -> float:
        return self.spread / 2.0


@dataclass(frozen=True)
class LegFill:
    """Simulated fill for one leg.

    All cost fields are POSITIVE dollar amounts (what we paid).
    `fill_price` is the per-share execution price.
    """
    intended_mid_price: float
    fill_price: float
    commission_usd: float
    spread_cost_usd: float
    slippage_usd: float


@dataclass(frozen=True)
class TradeFill:
    """Simulated fill for a multi-leg trade.

    `net_credit_or_debit_usd`:
        positive => net credit received at entry
        negative => net debit paid at entry
    """
    leg_fills: list[LegFill]
    total_commission_usd: float
    total_spread_cost_usd: float
    total_slippage_usd: float
    net_credit_or_debit_usd: float


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_order(order: LegOrder) -> None:
    if order.quantity <= 0:
        raise ValueError(f"quantity must be positive, got {order.quantity}")
    if order.multiplier <= 0:
        raise ValueError(f"multiplier must be positive, got {order.multiplier}")
    if order.direction not in ('long', 'short'):
        raise ValueError(f"direction must be 'long' or 'short', got {order.direction!r}")
    if order.instrument_type not in ('call', 'put', 'stock'):
        raise ValueError(
            f"instrument_type must be 'call', 'put', or 'stock', got {order.instrument_type!r}"
        )


def _validate_quote(quote: Quote) -> None:
    if quote.bid < 0 or quote.ask < 0:
        raise ValueError(f"bid/ask must be non-negative, got bid={quote.bid}, ask={quote.ask}")
    if quote.ask < quote.bid:
        raise ValueError(f"ask ({quote.ask}) must be >= bid ({quote.bid})")


# ---------------------------------------------------------------------------
# Core simulation
# ---------------------------------------------------------------------------


def simulate_leg_fill(
    order: LegOrder,
    quote: Quote,
    *,
    cost_stress_multiplier: float = 1.0,
    slippage_per_share: float = SLIPPAGE_PER_LEG_USD,
    rng_seed: int | None = None,
) -> LegFill:
    """Compute a simulated fill for one leg.

    Mechanics:
        effective_half_spread = quote.half_spread * cost_stress_multiplier
        long  (buy):  fill = mid + effective_half_spread + slippage
        short (sell): fill = mid - effective_half_spread - slippage

    Cost reporting (all positive dollars):
        commission_usd  = COMMISSION_PER_CONTRACT_USD * quantity
        spread_cost_usd = effective_half_spread * quantity * multiplier
        slippage_usd    = slippage_per_share * quantity * multiplier

    `rng_seed` is reserved for future RNG-based slippage variance; default
    behavior is fully deterministic (a fixed `slippage_per_share` is applied).
    If `rng_seed` is provided, slippage becomes a noisy draw with mean
    `slippage_per_share` and std `slippage_per_share * 0.25` (still seeded
    so repeated calls with the same seed return identical fills).
    """
    _validate_order(order)
    _validate_quote(quote)
    if cost_stress_multiplier < 0:
        raise ValueError(
            f"cost_stress_multiplier must be non-negative, got {cost_stress_multiplier}"
        )
    if slippage_per_share < 0:
        raise ValueError(f"slippage_per_share must be non-negative, got {slippage_per_share}")

    mid = quote.mid
    effective_half_spread = quote.half_spread * cost_stress_multiplier

    # Per-share slippage (deterministic by default; seeded RNG draw if requested).
    if rng_seed is None:
        slip_per_share = slippage_per_share
    else:
        rng = np.random.default_rng(rng_seed)
        # Truncate at zero so slippage stays a cost, not a benefit.
        slip_per_share = float(
            max(0.0, rng.normal(loc=slippage_per_share, scale=slippage_per_share * 0.25))
        )

    if order.direction == 'long':
        fill_price = mid + effective_half_spread + slip_per_share
    else:  # short
        fill_price = mid - effective_half_spread - slip_per_share

    notional_units = order.quantity * order.multiplier
    commission_usd = COMMISSION_PER_CONTRACT_USD * order.quantity
    spread_cost_usd = effective_half_spread * notional_units
    slippage_usd = slip_per_share * notional_units

    return LegFill(
        intended_mid_price=order.intended_mid_price,
        fill_price=fill_price,
        commission_usd=commission_usd,
        spread_cost_usd=spread_cost_usd,
        slippage_usd=slippage_usd,
    )


def simulate_trade_fill(
    orders: list[LegOrder],
    quotes: list[Quote],
    *,
    cost_stress_multiplier: float = 1.0,
    slippage_per_share: float = SLIPPAGE_PER_LEG_USD,
) -> TradeFill:
    """Apply `simulate_leg_fill` to each leg, sum costs.

    Cash-flow convention: for each leg,
        signed_cash = sign * fill_price * quantity * multiplier
        where sign = +1 for SHORT (cash received), -1 for LONG (cash paid).

    Then `net_credit_or_debit_usd = sum(signed_cash) - total_commission`.

    Note: BOTH spread cost AND slippage are already embedded in fill_price
    (`fill = mid ± effective_half_spread ± slip_per_share`), so neither is
    subtracted again here. Only commission is subtracted explicitly because
    it is the only cost not reflected in fill_price.
    """
    if len(orders) != len(quotes):
        raise ValueError(
            f"orders and quotes must have same length, got {len(orders)} and {len(quotes)}"
        )

    if not orders:
        return TradeFill(
            leg_fills=[],
            total_commission_usd=0.0,
            total_spread_cost_usd=0.0,
            total_slippage_usd=0.0,
            net_credit_or_debit_usd=0.0,
        )

    leg_fills: list[LegFill] = []
    total_commission = 0.0
    total_spread = 0.0
    total_slippage = 0.0
    signed_cash_total = 0.0

    for order, quote in zip(orders, quotes):
        leg_fill = simulate_leg_fill(
            order,
            quote,
            cost_stress_multiplier=cost_stress_multiplier,
            slippage_per_share=slippage_per_share,
        )
        leg_fills.append(leg_fill)
        total_commission += leg_fill.commission_usd
        total_spread += leg_fill.spread_cost_usd
        total_slippage += leg_fill.slippage_usd

        sign = 1.0 if order.direction == 'short' else -1.0
        signed_cash_total += sign * leg_fill.fill_price * order.quantity * order.multiplier

    net_credit_or_debit = signed_cash_total - total_commission

    return TradeFill(
        leg_fills=leg_fills,
        total_commission_usd=total_commission,
        total_spread_cost_usd=total_spread,
        total_slippage_usd=total_slippage,
        net_credit_or_debit_usd=net_credit_or_debit,
    )


def compute_total_costs(fill: TradeFill) -> dict:
    """Return a flat dict summarizing trade-level costs.

    Keys:
        commission: total commission across legs
        spread:     total spread cost (half_spread * units, positive)
        slippage:   total slippage cost (positive)
        total:      commission + spread + slippage (all-in cost in dollars)
    """
    return {
        'commission': fill.total_commission_usd,
        'spread': fill.total_spread_cost_usd,
        'slippage': fill.total_slippage_usd,
        'total': (
            fill.total_commission_usd
            + fill.total_spread_cost_usd
            + fill.total_slippage_usd
        ),
    }
