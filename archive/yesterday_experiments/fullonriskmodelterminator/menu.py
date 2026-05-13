"""Action menu the LLM council picks from each day.

Constrained menu so LLMs can't hallucinate impossible orders. Every action has
a deterministic cost/payoff model so the backtest is reproducible.

10 actions × 4 sizing tiers × 3 DTE buckets = 120 possible daily decisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Action = Literal[
    "no_trade",
    "long_call",          # buy ATM call — bullish directional
    "long_put",           # buy ATM put — bearish directional
    "long_straddle",      # buy ATM straddle — vol expansion bet
    "short_strangle",    # sell wide-OTM strangle — premium harvest
    "short_iron_condor", # sell defined-risk condor
    "credit_put_spread",  # short put spread — bullish premium
    "credit_call_spread", # short call spread — bearish premium
    "long_calendar",      # long calendar — flat vol with positive theta
    "close_all",          # liquidate all open positions
]

SizingTier = Literal["1pct", "5pct", "15pct", "30pct"]
DteBucket = Literal["0dte", "weekly", "monthly"]


@dataclass(frozen=True)
class TradeDecision:
    action: Action
    size_tier: SizingTier = "5pct"
    dte_bucket: DteBucket = "weekly"
    strike_offset_pct: float = 0.0    # 0 = ATM, +0.02 = 2% OTM call side, -0.02 = 2% OTM put
    reasoning: str = ""

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "size_tier": self.size_tier,
            "dte_bucket": self.dte_bucket,
            "strike_offset_pct": self.strike_offset_pct,
            "reasoning": self.reasoning,
        }


SIZE_TO_PCT = {"1pct": 0.01, "5pct": 0.05, "15pct": 0.15, "30pct": 0.30}
DTE_TO_DAYS = {"0dte": 0, "weekly": 7, "monthly": 30}


def validate(decision_dict: dict) -> TradeDecision:
    """Force a dict into a valid TradeDecision; raises on garbage."""
    action = str(decision_dict.get("action", "no_trade")).lower().replace(" ", "_")
    valid_actions = {
        "no_trade", "long_call", "long_put", "long_straddle", "short_strangle",
        "short_iron_condor", "credit_put_spread", "credit_call_spread",
        "long_calendar", "close_all",
    }
    if action not in valid_actions:
        action = "no_trade"

    size = str(decision_dict.get("size_tier", "5pct")).lower()
    if size not in SIZE_TO_PCT:
        size = "5pct"

    dte = str(decision_dict.get("dte_bucket", "weekly")).lower()
    if dte not in DTE_TO_DAYS:
        dte = "weekly"

    try:
        offset = float(decision_dict.get("strike_offset_pct", 0.0))
        offset = max(-0.10, min(0.10, offset))
    except (TypeError, ValueError):
        offset = 0.0

    return TradeDecision(
        action=action,  # type: ignore
        size_tier=size,  # type: ignore
        dte_bucket=dte,  # type: ignore
        strike_offset_pct=offset,
        reasoning=str(decision_dict.get("reasoning", ""))[:300],
    )
