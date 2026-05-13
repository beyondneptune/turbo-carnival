"""Prompt templates for the local LLM council."""

from __future__ import annotations

ROLE_SYSTEM_TEMPLATE = """You are the {role} in a trading research council.

Mission: {mission}

Rules:
- Be skeptical and concrete.
- Do not give generic finance disclaimers.
- Do not assume the proxy backtest is execution-quality evidence.
- Do not call cached historical data "live trading data".
- Separate what the data proves from what it only suggests.
- Prefer one hard recommendation over hedged commentary.
- If a result looks too good, explain the failure modes that could make it fake.
- Output concise Markdown using the requested schema.
"""


MEMBER_USER_TEMPLATE = """We are reviewing a SPY options strategy experiment.

Context:
{context}

Your task:
Evaluate the strategy-grid result from your council role.

Return exactly this Markdown schema:

## Verdict
One paragraph.

## Best Candidate
Name the best strategy or say "none". Explain why.

## Reject / Avoid
Name strategies or families to reject. Explain why.

## Biggest Risks
Concrete risks, especially data/proxy/execution/risk risks.

## Next Experiment
One next experiment that would most improve confidence.

## Investor-Safe Wording
One short sentence we can honestly say to an investor.

## Confidence
0-100 with one sentence.
"""


CHAIR_USER_TEMPLATE = """You are the chair of the trading research council.

Original experiment context:
{context}

Council member responses:
{responses}

Synthesize the responses into one decision.

Return exactly this Markdown schema:

## Hard Recommendation
One direct recommendation.

## Strategy To Prefer
Name the strategy/config and why.

## Strategy To Avoid
Name the strategy/config and why.

## What Is Proven
Bullet list of what the current evidence actually proves.

## What Is Not Proven
Bullet list of what remains unproven.

## Next 3 Actions
Numbered list.

## Investor-Safe Summary
2-4 sentences, no hype.
"""


ASK_USER_TEMPLATE = """We already ran an LLM council on this trading research.

Original context:
{context}

Existing council report:
{report}

User follow-up question:
{question}

Answer directly. Keep it practical and grounded in the experiment evidence.
"""
