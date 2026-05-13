"""NIM council review for the final paper_v1 pre-deploy simulation."""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]

REVIEWERS = [
    {
        "role": "risk_manager",
        "model": "qwen/qwen3-coder-480b-a35b-instruct",
        "mission": "Attack drawdown, tail risk, position sizing, and whether the system can be safely paper-deployed tomorrow.",
    },
    {
        "role": "execution_engineer",
        "model": "mistralai/mistral-nemotron",
        "mission": "Evaluate broker/API readiness, IBKR connection requirements, order-routing risk, quote/fill reconciliation, and operational blockers.",
    },
    {
        "role": "investor_comms",
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "mission": "Recommend exactly what should and should not be said to an investor watching a 30-day paper trial.",
    },
    {
        "role": "quant_reviewer",
        "model": "mistralai/mistral-medium-3-instruct",
        "mission": "Review statistical meaning of the Year 2 result, sample size, monthly return quality, and overfitting risk.",
    },
    {
        "role": "red_team",
        "model": "moonshotai/kimi-k2-instruct",
        "mission": "Find the strongest reason not to proceed, and separate paper-trading risk from live-capital risk.",
    },
    {
        "role": "portfolio_allocator",
        "model": "bytedance/seed-oss-36b-instruct",
        "mission": "Recommend capital allocation/risk limits for a $10k paper month and future £100k live account.",
    },
]

PROMPT = """You are the {role}. Mission: {mission}

We are doing a final go/no-go review for TOMORROW'S IBKR PAPER deployment.

Simulation source:
- Real held-out Year 2 evidence trade ledger from data/trading_system.db
- Run id: {run_id}
- This is realized P&L, not full daily option mark-to-market.
- Strategy: paper_v1 hybrid: SPY core iron condor + earnings long straddle satellite.
- Capital for paper test: $10,000.
- IBKR connection status: NOT CONNECTED YET. Local port 4002 refused. IB Gateway/TWS must be started and API enabled.
- Order placement status: IBKR adapter can connect/fetch account/chain once gateway runs, but 4-leg combo order placement is scaffolded/not fully implemented yet.

Final pre-deploy simulation metrics:
{summary_json}

Question:
1. Should we proceed with PAPER deployment tomorrow if IBKR connectivity and order routing are finished first?
2. Should any REAL capital be used after a good first month?
3. What exact risk limits and investor wording should be used?

Return strict JSON only:
{{
  "role": "{role}",
  "paper_go": true,
  "paper_go_conditions": ["condition 1", "condition 2"],
  "live_capital_go": false,
  "live_capital_blockers": ["blocker 1", "blocker 2"],
  "risk_limits": {{
    "max_initial_notional_usd": 10000,
    "max_risk_per_trade_pct": 0,
    "max_daily_loss_pct": 0,
    "max_cycle_drawdown_pct": 0,
    "kill_switch_rule": "..."
  }},
  "investor_message": "one paragraph, honest and non-promissory",
  "one_sentence_verdict": "..."
}}
"""


def load_env() -> None:
    env = REPO_ROOT / "council" / ".env.local"
    if not env.exists():
        return
    for line in env.read_text(encoding="utf-8").splitlines():
        if "=" not in line or line.strip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def call_reviewer(client: OpenAI, reviewer: dict[str, str], summary: dict[str, Any]) -> dict[str, Any]:
    prompt = PROMPT.format(
        role=reviewer["role"],
        mission=reviewer["mission"],
        run_id=summary["run_id"],
        summary_json=json.dumps(summary, indent=2),
    )
    try:
        response = client.chat.completions.create(
            model=reviewer["model"],
            messages=[
                {"role": "system", "content": "You output strict JSON. No markdown, no preamble."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=1400,
        )
        return {
            "ok": True,
            "role": reviewer["role"],
            "model": reviewer["model"],
            "content": response.choices[0].message.content or "",
        }
    except Exception as exc:
        return {
            "ok": False,
            "role": reviewer["role"],
            "model": reviewer["model"],
            "error": f"{type(exc).__name__}: {exc}",
            "content": "",
        }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    load_env()
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY missing")

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1", timeout=120.0)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "council" / "predeploy_runs" / stamp
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call_reviewer, client, r, summary): r for r in REVIEWERS}
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            (raw_dir / f"{result['role']}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"done {result['role']} via {result['model']} ok={result.get('ok')}", flush=True)

    parsed = []
    for result in results:
        if not result.get("ok"):
            continue
        payload = parse_json(str(result.get("content", "")))
        if payload:
            payload["_model"] = result["model"]
            parsed.append(payload)

    paper_go_votes = sum(1 for p in parsed if p.get("paper_go") is True)
    live_go_votes = sum(1 for p in parsed if p.get("live_capital_go") is True)
    synthesis = {
        "summary_used": str(args.summary),
        "reviewers": len(REVIEWERS),
        "ok_responses": sum(1 for r in results if r.get("ok")),
        "parsed_responses": len(parsed),
        "paper_go_votes": paper_go_votes,
        "live_capital_go_votes": live_go_votes,
        "responses": parsed,
    }
    (out_dir / "predeploy_council_summary.json").write_text(json.dumps(synthesis, indent=2), encoding="utf-8")

    md = [
        "# Predeploy Council Review",
        "",
        f"Summary used: `{args.summary}`",
        f"Responses parsed: {len(parsed)}/{len(REVIEWERS)}",
        f"Paper-go votes: {paper_go_votes}/{len(parsed)}",
        f"Live-capital-go votes: {live_go_votes}/{len(parsed)}",
        "",
        "## Verdicts",
        "",
    ]
    for p in parsed:
        md.append(f"### {p.get('role')} ({p.get('_model')})")
        md.append(f"- Paper go: `{p.get('paper_go')}`")
        md.append(f"- Live capital go: `{p.get('live_capital_go')}`")
        md.append(f"- Verdict: {p.get('one_sentence_verdict')}")
        if p.get("paper_go_conditions"):
            md.append("- Paper conditions: " + "; ".join(map(str, p["paper_go_conditions"][:5])))
        if p.get("live_capital_blockers"):
            md.append("- Live blockers: " + "; ".join(map(str, p["live_capital_blockers"][:5])))
        if p.get("investor_message"):
            md.append(f"- Investor wording: {p.get('investor_message')}")
        md.append("")
    (out_dir / "predeploy_council_report.md").write_text("\n".join(md), encoding="utf-8")

    print(json.dumps({k: synthesis[k] for k in ("ok_responses", "parsed_responses", "paper_go_votes", "live_capital_go_votes")}, indent=2))
    print(f"Report: {out_dir / 'predeploy_council_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
