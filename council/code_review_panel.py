"""NIM panel of hostile code reviewers attacking the backtest engine.

Different from research_lab.py and red_team_winner.py:
  research_lab proposes new strategies.
  red_team_winner attacks a chosen strategy candidate.
  code_review_panel attacks the BACKTEST CODE itself for bugs that would
  silently inflate results.

This is what LLMs are genuinely good at: spotting subtle bugs in code.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]


REVIEWERS = [
    {
        "role": "lookahead_bias_hunter",
        "model": "qwen/qwen3-coder-480b-a35b-instruct",
        "mission": "Find any place where future data leaks into a decision. Examples: rolling windows that use forward-looking pandas params, gates computed using future returns, exit triggers that 'know' the future, IV / vol proxies that aren't truly causal.",
    },
    {
        "role": "off_by_one_hunter",
        "model": "qwen/qwen3-coder-480b-a35b-instruct",
        "mission": "Find off-by-one errors in date handling, expiry calculations, profit/stop trigger math, position lifecycle (entry day vs exit day), and 'as of' boundaries.",
    },
    {
        "role": "exit_logic_hunter",
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "mission": "Audit the exit logic. When the proxy says 'stop_loss hit', does it use the current day's high/low, or just the close? Could profit_target be checking against an unrealistic intraday extreme? Are exit prices realistic given that we entered at mid?",
    },
    {
        "role": "double_count_hunter",
        "model": "moonshotai/kimi-k2-instruct",
        "mission": "Audit for double-counting. Are entry credits being counted as P&L AND held as cash? Are commissions and slippage modeled or assumed zero? Is the same trade ever closed twice? Is multiple-position concurrency handled correctly?",
    },
    {
        "role": "regime_artifact_hunter",
        "model": "stepfun-ai/step-3.5-flash",
        "mission": "Per a walk-forward test, 57 of 60 strategy variants had HIGHER out-of-sample Sharpe than in-sample. That uniformity is suspicious. Find the code path that could cause uniform OOS outperformance: changes in IV data quality, trade count, daily NAV update frequency, or compounding behavior between 2019-2022 vs 2023-2026.",
    },
    {
        "role": "sharpe_annualization_hunter",
        "model": "mistralai/mistral-nemotron",
        "mission": "Audit the Sharpe annualization. The proxy reports Sharpe 4.85 for an iron-condor strategy, which is unrealistically high. Is the annualization formula using sqrt(N_trades_per_year) correctly? Does it assume independence that doesn't hold for concurrent overlapping condors? What's a realistic Sharpe estimate after correcting for trade correlation?",
    },
]

PROMPT = """You are a hostile code reviewer.
Role: {role}
Mission: {mission}

You are reviewing this backtest engine. Read the code carefully and find concrete bugs or inflated assumptions.

```python
{code}
```

Return JSON only. No markdown. Schema:
{{
  "role": "{role}",
  "bugs_found": [
    {{
      "severity": "critical|high|medium|low",
      "file": "experiments/run_spy_core_strategy_grid.py",
      "line_or_function": "specific line number or function name",
      "bug": "1-2 sentence description",
      "evidence": "specific code snippet that proves it",
      "impact_on_results": "how this would inflate or distort the reported Sharpe / return / DD",
      "fix": "concrete code change to repair"
    }}
  ],
  "false_alarm_areas": ["areas that LOOK suspicious but are actually correct"],
  "honest_summary": "one sentence: should we trust the numbers from this engine?"
}}

Rules:
- Cite specific line numbers or function names. Vague accusations are useless.
- If you can't find a critical bug, say so. Don't invent.
- Distinguish between code bugs and modeling assumptions. Both matter, but they have different fixes.
- Be brutally honest. Confirmation bias here costs real money."""


def load_local_env() -> None:
    env_path = REPO_ROOT / "council" / ".env.local"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def call_reviewer(client: OpenAI, reviewer: dict, code: str) -> dict:
    prompt = PROMPT.format(role=reviewer["role"], mission=reviewer["mission"], code=code)
    try:
        resp = client.chat.completions.create(
            model=reviewer["model"],
            messages=[
                {"role": "system", "content": "You output strict JSON code reviews. No markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2000,
        )
        return {"ok": True, "role": reviewer["role"], "model": reviewer["model"],
                "content": resp.choices[0].message.content or ""}
    except Exception as exc:
        return {"ok": False, "role": reviewer["role"], "model": reviewer["model"],
                "error": f"{type(exc).__name__}: {exc}", "content": ""}


def parse_json_loose(text: str):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        return json.loads(m.group(0)) if m else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target",
                        default="experiments/run_spy_core_strategy_grid.py")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    load_local_env()
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        print("NVIDIA_API_KEY not set", file=sys.stderr)
        return 1
    client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1", timeout=180.0)

    code_path = REPO_ROOT / args.target
    code = code_path.read_text(encoding="utf-8")
    # Truncate if too long for context
    if len(code) > 50_000:
        code = code[:50_000] + "\n\n# [TRUNCATED]"

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "council" / "code_review_runs" / stamp
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reviewing {args.target} ({len(code):,} chars)")
    print(f"Output: {out_dir}\n")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call_reviewer, client, r, code): r for r in REVIEWERS}
        for fut in concurrent.futures.as_completed(futures):
            payload = fut.result()
            results.append(payload)
            stem = payload["role"]
            (raw_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            (raw_dir / f"{stem}.md").write_text(str(payload.get("content", "")), encoding="utf-8")
            print(f"  {stem:30} via {payload['model']:50} ok={payload.get('ok')}", flush=True)

    # Synthesize
    all_bugs = []
    for r in results:
        if not r.get("ok"):
            continue
        parsed = parse_json_loose(str(r.get("content", "")))
        if parsed is None:
            continue
        for b in (parsed.get("bugs_found") or []):
            if isinstance(b, dict):
                b = dict(b); b["from_role"] = parsed.get("role", r["role"])
                all_bugs.append(b)

    sev = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_bugs.sort(key=lambda x: sev.get(str(x.get("severity", "medium")).lower(), 4))

    summary = {
        "target": args.target,
        "n_reviewers_attempted": len(REVIEWERS),
        "n_reviewers_completed": sum(1 for r in results if r.get("ok")),
        "all_bugs_ranked": all_bugs,
    }
    (out_dir / "code_review_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== Top 10 bugs found ({len(all_bugs)} total) ===")
    for b in all_bugs[:10]:
        print(f"\n[{str(b.get('severity','?')).upper():8}] {b.get('bug','?')[:120]}")
        print(f"  by: {b.get('from_role','?')}")
        print(f"  location: {b.get('line_or_function','?')[:120]}")
        if b.get('impact_on_results'):
            print(f"  impact: {b['impact_on_results'][:200]}")

    # Honest summaries from each reviewer
    print("\n=== Per-reviewer verdicts ===")
    for r in results:
        if not r.get("ok"):
            print(f"  {r['role']}: ERROR {r.get('error', '')[:80]}")
            continue
        parsed = parse_json_loose(str(r.get("content", ""))) or {}
        print(f"  {r['role']:30} -> {(parsed.get('honest_summary') or '?')[:150]}")

    print(f"\nReport: {out_dir / 'code_review_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
