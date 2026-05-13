"""Multi-LLM voting council. Aggressively uses NIM.

Each trading day:
  1. Build market-state prompt (SPY price, VIX, IV, recent returns, open positions, NAV)
  2. Fan out to 8 NIM models in parallel
  3. Each returns a TradeDecision JSON
  4. Aggregate via majority vote with tiebreak rules
  5. Return single decision

No safety gates inside the council itself — sizing/risk overlays applied
downstream in execution.py. The user explicitly waived defensive constraints.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fullonriskmodelterminator.menu import TradeDecision, validate  # noqa: E402


# 8 diverse NIM models — different families, different sizes, different lenses
COUNCIL = [
    {"role": "qwen_coder_480b", "model": "qwen/qwen3-coder-480b-a35b-instruct"},
    {"role": "mistral_large", "model": "mistralai/mistral-large-3-675b-instruct-2512"},
    {"role": "llama_maverick", "model": "meta/llama-4-maverick-17b-128e-instruct"},
    {"role": "kimi_instruct", "model": "moonshotai/kimi-k2-instruct"},
    {"role": "qwen_thinking", "model": "qwen/qwen3-next-80b-a3b-thinking"},
    {"role": "mistral_nemotron", "model": "mistralai/mistral-nemotron"},
    {"role": "step_flash", "model": "stepfun-ai/step-3.5-flash"},
    {"role": "minimax_m27", "model": "minimaxai/minimax-m2.7"},
]


PROMPT_TEMPLATE = """You are an aggressive options trader running a paper account.

TODAY: {today}
SPY price: ${spy_price:.2f}  (5-day return: {spy_5d_ret:+.2%}, 20-day return: {spy_20d_ret:+.2%})
VIX: {vix:.2f}  (5-day ROC: {vix_5d_roc:+.2%})
ATM IV: {iv:.4f}  (IV rank: {iv_rank:.0%})
Realized vol (20d): {rv20:.2%}
VRP (IV - RV): {vrp:+.2%}

Account NAV: ${nav:,.2f}
Cash available: ${cash:,.2f}
Open positions: {n_open} positions, total max-loss exposure ${total_risk:,.2f} ({pct_at_risk:.1%} of NAV)
Recent NAV trend: {nav_trend}

You are aiming for HIGH RETURNS — this is paper trading, the user has set max aggression. The boring 1-2% monthly strategies are off the table. We are hunting for 5%+ monthly.

Pick ONE action from this menu (no other actions valid):
  - no_trade
  - long_call            (bet SPY goes UP — buy ATM call)
  - long_put             (bet SPY goes DOWN — buy ATM put)
  - long_straddle        (bet vol EXPANDS — IV crush hurts you)
  - short_strangle       (bet SPY stays in range — IV crush helps)
  - short_iron_condor    (defined-risk version of short_strangle)
  - credit_put_spread    (mildly bullish premium harvest)
  - credit_call_spread   (mildly bearish premium harvest)
  - long_calendar        (flat vol bet with positive theta)
  - close_all            (liquidate everything)

Sizing tier (% of NAV at risk on THIS trade):
  - "1pct"   conservative
  - "5pct"   standard aggressive
  - "15pct"  high conviction
  - "30pct"  YOLO — only if you're very confident

DTE bucket:
  - "0dte"     same-day expiry (highest theta, highest gamma risk)
  - "weekly"   ~7 DTE
  - "monthly"  ~30 DTE

Strike offset (for spreads/strangles): -0.10 to +0.10 representing % from spot.
  0 = ATM. -0.02 = 2% OTM put side. +0.02 = 2% OTM call side.

Return strict JSON only:
{{
  "action": "...",
  "size_tier": "...",
  "dte_bucket": "...",
  "strike_offset_pct": 0.0,
  "reasoning": "one sentence on why"
}}"""


def load_env() -> None:
    p = REPO_ROOT / "council" / ".env.local"
    if not p.exists():
        return
    for ln in p.read_text().splitlines():
        if "=" in ln and not ln.strip().startswith("#"):
            k, v = ln.split("=", 1)
            if k.strip() and k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def _parse_json_loose(text: str) -> dict | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _ask_one(client: OpenAI, model: str, prompt: str, timeout_sec: float = 25.0) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output strict JSON trading decisions. No markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=400,
            timeout=timeout_sec,
        )
        return _parse_json_loose(resp.choices[0].message.content or "")
    except Exception:
        return None


def vote(market_state: dict, client: OpenAI | None = None, workers: int = 8) -> tuple[TradeDecision, list[dict]]:
    """Return aggregated decision plus per-model raw votes."""
    if client is None:
        load_env()
        key = os.environ.get("NVIDIA_API_KEY")
        if not key:
            # No NIM available — return no_trade
            return TradeDecision(action="no_trade", reasoning="NIM key missing"), []
        client = OpenAI(
            api_key=key,
            base_url="https://integrate.api.nvidia.com/v1",
            timeout=30.0,
        )

    prompt = PROMPT_TEMPLATE.format(**market_state)

    raw_votes: list[dict] = []
    decisions: list[TradeDecision] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_ask_one, client, m["model"], prompt): m for m in COUNCIL}
        for fut in concurrent.futures.as_completed(futures):
            m = futures[fut]
            j = fut.result()
            if j is None:
                raw_votes.append({"role": m["role"], "model": m["model"], "raw": None, "decision": None})
                continue
            d = validate(j)
            raw_votes.append({"role": m["role"], "model": m["model"], "raw": j, "decision": d.to_dict()})
            decisions.append(d)

    if not decisions:
        return TradeDecision(action="no_trade", reasoning="all NIM calls failed"), raw_votes

    # Majority vote on action
    action_counter = Counter(d.action for d in decisions)
    top_action, top_count = action_counter.most_common(1)[0]

    # Among voters who picked the winning action, take median sizing tier and dte
    winners = [d for d in decisions if d.action == top_action]
    size_counter = Counter(d.size_tier for d in winners)
    dte_counter = Counter(d.dte_bucket for d in winners)
    median_size = size_counter.most_common(1)[0][0]
    median_dte = dte_counter.most_common(1)[0][0]
    median_offset = sorted(d.strike_offset_pct for d in winners)[len(winners) // 2]
    reasoning = "; ".join(d.reasoning[:80] for d in winners[:3])

    return (
        TradeDecision(
            action=top_action,  # type: ignore
            size_tier=median_size,  # type: ignore
            dte_bucket=median_dte,  # type: ignore
            strike_offset_pct=median_offset,
            reasoning=f"[{top_count}/{len(decisions)} voted] {reasoning}",
        ),
        raw_votes,
    )
