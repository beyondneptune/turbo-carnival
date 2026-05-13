"""IBKR combo-order construction for paper trading.

Builds 2/4-leg combo orders (vertical spreads, iron condors, strangles) via
ib_insync's Bag contract and submits as LimitOrder. Refuses to place if not on
a paper port.

The base IBKRClient in ingest/ibkr_client.py raises NotImplementedError for
combo orders. This module provides the real implementation, separated so it
can be unit-tested without dragging in the whole adapter.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from ib_insync import (
    Bag,
    ComboLeg,
    Contract,
    IB,
    LimitOrder,
    Option,
    Order,
    Stock,
    Trade,
)

logger = logging.getLogger(__name__)

DEFAULT_EXCHANGE = "SMART"
DEFAULT_CURRENCY = "USD"
PAPER_PORTS = {4002, 7497}  # IB Gateway paper, TWS paper


@dataclass
class LegSpec:
    right: Literal["C", "P"]   # call or put
    strike: float
    expiry: str                 # YYYYMMDD
    direction: Literal["long", "short"]
    quantity: int = 1


@dataclass
class ComboOrderRequest:
    underlying: str             # e.g. "SPY"
    legs: list[LegSpec]
    limit_price: float          # net credit (positive) or debit (negative)
    tag: str                    # for tracking (e.g. "lane1_2026-05-11")


@dataclass
class ComboOrderResult:
    ok: bool
    submitted: bool
    filled: bool
    fill_price: float | None
    leg_fills: list[dict]
    error: str | None
    raw_trade: object | None = None


def _qualify_legs(ib: IB, underlying: str, legs: list[LegSpec]) -> list[Option]:
    """Resolve each leg into an Option contract with conId."""
    contracts = [
        Option(underlying, leg.expiry, leg.strike, leg.right,
                exchange=DEFAULT_EXCHANGE, currency=DEFAULT_CURRENCY)
        for leg in legs
    ]
    ib.qualifyContracts(*contracts)
    for leg, c in zip(legs, contracts):
        if not c.conId:
            raise RuntimeError(
                f"could not resolve leg {underlying} {leg.expiry} "
                f"{leg.strike} {leg.right} {leg.direction}"
            )
    return contracts


def _build_bag(ib: IB, underlying: str, legs: list[LegSpec],
                contracts: list[Option]) -> Bag:
    """Build a Bag (combo) contract from qualified leg contracts."""
    combo_legs = []
    for leg, c in zip(legs, contracts):
        cl = ComboLeg()
        cl.conId = c.conId
        cl.ratio = leg.quantity
        cl.action = "BUY" if leg.direction == "long" else "SELL"
        cl.exchange = DEFAULT_EXCHANGE
        combo_legs.append(cl)
    bag = Bag(
        symbol=underlying,
        exchange=DEFAULT_EXCHANGE,
        currency=DEFAULT_CURRENCY,
        comboLegs=combo_legs,
    )
    return bag


def place_combo_paper(ib: IB, request: ComboOrderRequest,
                       host_port: tuple[str, int],
                       fill_timeout_sec: float = 15.0) -> ComboOrderResult:
    """Place a combo limit order on a paper account, wait briefly for fill.

    Returns ComboOrderResult either way (filled/unfilled/error).
    The caller should NOT retry on its own — leave decision to caller.
    """
    _host, port = host_port
    if port not in PAPER_PORTS:
        return ComboOrderResult(
            ok=False, submitted=False, filled=False, fill_price=None,
            leg_fills=[], error=f"refusing to place on non-paper port {port}"
        )

    try:
        contracts = _qualify_legs(ib, request.underlying, request.legs)
    except Exception as exc:
        return ComboOrderResult(
            ok=False, submitted=False, filled=False, fill_price=None,
            leg_fills=[], error=f"leg qualification failed: {exc}"
        )

    bag = _build_bag(ib, request.underlying, request.legs, contracts)
    # For a net-credit position the order ACTION is BUY the combo at a NEGATIVE
    # limit price (you receive the credit). For a net-debit it's BUY at a
    # positive limit. ib_insync convention: BUY combo always; price is the net.
    order: Order = LimitOrder(
        action="BUY",
        totalQuantity=1,                          # 1 combo unit; leg ratios encode contract counts
        lmtPrice=round(request.limit_price, 2),
    )
    order.tif = "DAY"
    order.account = ""                            # default paper account
    order.orderRef = request.tag

    trade: Trade = ib.placeOrder(bag, order)
    logger.info("submitted combo order tag=%s legs=%s limit=%s",
                 request.tag, len(request.legs), request.limit_price)

    deadline = time.time() + fill_timeout_sec
    while time.time() < deadline:
        ib.sleep(0.5)
        if trade.isDone():
            break

    filled = trade.orderStatus.status == "Filled"
    fill_price = float(trade.orderStatus.avgFillPrice) if trade.orderStatus.avgFillPrice else None
    leg_fills = []
    for f in trade.fills:
        leg_fills.append({
            "symbol": f.contract.symbol,
            "strike": getattr(f.contract, "strike", None),
            "right": getattr(f.contract, "right", None),
            "price": f.execution.price,
            "shares": f.execution.shares,
            "time": str(f.execution.time),
        })

    return ComboOrderResult(
        ok=True,
        submitted=True,
        filled=filled,
        fill_price=fill_price,
        leg_fills=leg_fills,
        error=None if filled else f"not filled within {fill_timeout_sec}s; status={trade.orderStatus.status}",
        raw_trade=trade,
    )


def construct_iron_condor_request(
    *,
    underlying: str,
    expiry: str,
    long_put: float,
    short_put: float,
    short_call: float,
    long_call: float,
    quantity: int = 1,
    limit_credit: float = 0.0,
    tag: str = "",
) -> ComboOrderRequest:
    """Build an iron condor combo: short_call/long_call/short_put/long_put."""
    legs = [
        LegSpec(right="C", strike=short_call, expiry=expiry, direction="short", quantity=quantity),
        LegSpec(right="C", strike=long_call, expiry=expiry, direction="long", quantity=quantity),
        LegSpec(right="P", strike=short_put, expiry=expiry, direction="short", quantity=quantity),
        LegSpec(right="P", strike=long_put, expiry=expiry, direction="long", quantity=quantity),
    ]
    # Iron condor pays credit — sign convention: positive limit_credit means
    # we receive cash, so we BUY combo at NEGATIVE limit price.
    return ComboOrderRequest(
        underlying=underlying,
        legs=legs,
        limit_price=-abs(limit_credit),
        tag=tag,
    )


def construct_straddle_request(
    *,
    underlying: str,
    expiry: str,
    strike: float,
    direction: Literal["long", "short"],
    quantity: int = 1,
    limit_premium: float = 0.0,
    tag: str = "",
) -> ComboOrderRequest:
    legs = [
        LegSpec(right="C", strike=strike, expiry=expiry, direction=direction, quantity=quantity),
        LegSpec(right="P", strike=strike, expiry=expiry, direction=direction, quantity=quantity),
    ]
    # Long straddle: pay debit (BUY combo at positive limit)
    # Short straddle: receive credit (BUY combo at negative limit)
    sign = 1.0 if direction == "long" else -1.0
    return ComboOrderRequest(
        underlying=underlying,
        legs=legs,
        limit_price=sign * abs(limit_premium),
        tag=tag,
    )


def construct_strangle_request(
    *,
    underlying: str,
    expiry: str,
    call_strike: float,
    put_strike: float,
    direction: Literal["long", "short"],
    quantity: int = 1,
    limit_premium: float = 0.0,
    tag: str = "",
) -> ComboOrderRequest:
    legs = [
        LegSpec(right="C", strike=call_strike, expiry=expiry, direction=direction, quantity=quantity),
        LegSpec(right="P", strike=put_strike, expiry=expiry, direction=direction, quantity=quantity),
    ]
    sign = 1.0 if direction == "long" else -1.0
    return ComboOrderRequest(
        underlying=underlying,
        legs=legs,
        limit_price=sign * abs(limit_premium),
        tag=tag,
    )


def construct_vertical_spread_request(
    *,
    underlying: str,
    expiry: str,
    right: Literal["C", "P"],
    short_strike: float,
    long_strike: float,
    quantity: int = 1,
    limit_credit: float = 0.0,
    tag: str = "",
) -> ComboOrderRequest:
    """Credit vertical spread (bull put spread = right=P, short_strike > long_strike)."""
    legs = [
        LegSpec(right=right, strike=short_strike, expiry=expiry, direction="short", quantity=quantity),
        LegSpec(right=right, strike=long_strike, expiry=expiry, direction="long", quantity=quantity),
    ]
    return ComboOrderRequest(
        underlying=underlying,
        legs=legs,
        limit_price=-abs(limit_credit),
        tag=tag,
    )
