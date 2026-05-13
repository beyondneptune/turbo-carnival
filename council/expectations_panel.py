"""NIM panel: ask experts what's realistic for $10k-$100k options strategies.

Specifically address the 14% monthly target the investor is expecting.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]


PANEL = [
    {"role": "quant_realism_q",   "model": "qwen/qwen3-coder-480b-a35b-instruct"},
    {"role": "options_specialist", "model": "mistralai/mistral-nemotron"},
    {"role": "hedge_fund_lens",   "model": "meta/llama-4-maverick-17b-128e-instruct"},
    {"role": "academic_lens",     "model": "moonshotai/kimi-k2-instruct"},
    {"role": "fast_independent",  "model": "stepfun-ai/step-3.5-flash"},
    {"role": "risk_perspective",  "model": "qwen/qwen3-coder-480b-a35b-instruct"},
]


PROMPT = """You are a finance expert.

The user is running a SPY iron-condor + earnings-straddle paper-trading system on Interactive Brokers paper account ($100k).

Their backtest on 7 years of historical data showed:
- Proxy backtest (no friction, $10k, 1 contract): +264% (~20% CAGR)
- Realistic backtest (real bid/ask spread, commissions, $100k NAV with 1% per-trade risk): **+29% over 7 years (~3.7% CAGR), -4% max drawdown**
- Walk-forward IS Sharpe 4.11, OOS Sharpe 5.70 (annualized, assumes trade independence — real Sharpe likely 1.5-2.5)

The investor is expecting **14% monthly returns** sustained.

Answer concisely (JSON only, no markdown):

{{
  "is_14pct_monthly_achievable_sustainably": true|false,
  "annualized_equivalent_pct": 382.0,
  "realistic_monthly_for_this_strategy_pct": 0.0,
  "realistic_annual_for_this_strategy_pct": 0.0,
  "reference_points": [
    {{ "what": "e.g. Renaissance Medallion gross", "annual_return_pct": 66.0 }}
  ],
  "what_it_would_take_for_14pct_monthly": "specific risk profile / leverage / strategy type",
  "honest_one_sentence_to_user": "what to tell them",
  "honest_one_sentence_to_investor": "what to set expectations at"
}}

Be brutal. Cite real numbers. No platitudes. The user has full latitude to push back on the investor; help them do that with facts."""


def load_env():
    p = REPO_ROOT / "council" / ".env.local"
    if not p.exists(): return
    for ln in p.read_text().splitlines():
        if "=" in ln and not ln.strip().startswith("#"):
            k, v = ln.split("=", 1)
            if k.strip() and k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def call(client, m):
    try:
        r = client.chat.completions.create(
            model=m["model"],
            messages=[
                {"role": "system", "content": "You output strict JSON. No markdown."},
                {"role": "user", "content": PROMPT},
            ],
            temperature=0.3,
            max_tokens=900,
        )
        return {"ok": True, "role": m["role"], "model": m["model"], "content": r.choices[0].message.content or ""}
    except Exception as exc:
        return {"ok": False, "role": m["role"], "model": m["model"], "error": f"{type(exc).__name__}: {exc}"}


def parse(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try: return json.loads(text)
    except: pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: return None
    return None


def main():
    load_env()
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        print("NVIDIA_API_KEY missing", file=sys.stderr); return 1
    client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1", timeout=120.0)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO_ROOT / "council" / "expectations_runs" / stamp
    raw = out / "raw_responses"
    raw.mkdir(parents=True, exist_ok=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(call, client, m): m for m in PANEL}
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            results.append(r)
            (raw / f"{r['role']}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
            print(f"  {r['role']:25} ok={r.get('ok')} via {r['model']}", flush=True)

    # synthesize
    print("\n=== Panel verdicts ===\n")
    achievable = []
    realistic_monthly = []
    realistic_annual = []
    for r in results:
        if not r.get("ok"): continue
        p = parse(str(r.get("content", "")))
        if p is None: continue
        a = p.get("is_14pct_monthly_achievable_sustainably")
        if a is not None: achievable.append(a)
        m = p.get("realistic_monthly_for_this_strategy_pct")
        if isinstance(m, (int, float)): realistic_monthly.append(float(m))
        ann = p.get("realistic_annual_for_this_strategy_pct")
        if isinstance(ann, (int, float)): realistic_annual.append(float(ann))
        print(f"  {r['role']:25}")
        print(f"    14%/mo achievable? {a}")
        print(f"    realistic monthly: {m}%  realistic annual: {ann}%")
        print(f"    to user:     {p.get('honest_one_sentence_to_user','?')[:200]}")
        print(f"    to investor: {p.get('honest_one_sentence_to_investor','?')[:200]}")
        print()

    print("=== AGGREGATE ===")
    print(f"  '14% monthly is achievable' votes: True={achievable.count(True)}  False={achievable.count(False)}")
    if realistic_monthly:
        rm = sorted(realistic_monthly)
        print(f"  Realistic monthly range: {min(rm):.2f}% to {max(rm):.2f}%, median {rm[len(rm)//2]:.2f}%")
    if realistic_annual:
        ra = sorted(realistic_annual)
        print(f"  Realistic annual range:  {min(ra):.1f}% to {max(ra):.1f}%, median {ra[len(ra)//2]:.1f}%")

    (out / "panel_summary.json").write_text(json.dumps({
        "results": results,
        "votes_14pct_achievable": {"True": achievable.count(True), "False": achievable.count(False)},
        "realistic_monthly_range": realistic_monthly,
        "realistic_annual_range": realistic_annual,
    }, indent=2), encoding="utf-8")

    print(f"\nReport: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
