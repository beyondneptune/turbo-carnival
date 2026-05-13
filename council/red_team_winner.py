"""Adversarial red-team of the council's recommended winner.

Different angle from research_lab.py:
  research_lab asks models to PROPOSE strategies.
  red_team_winner asks models to ATTACK a proposed winner — find why it fails
  in real markets, what proxy artifacts inflate its score, and what stress
  tests would expose it.

Saves each model's response under council/redteam_runs/<timestamp>/ alongside
a synthesized table of attack vectors ranked by severity.
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
from typing import Any

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]

WINNER_NAME = "risk_qwen_risk_designer_risktest_d12_30d_lowdelta_pt30_sl20_c10"
WINNER_SPEC = {
    "name": WINNER_NAME,
    "entry_mode": "daily_ladder",
    "gate": "vrp_positive",
    "target_delta": 0.12,
    "target_dte": 30,
    "wing_width": 5.0,
    "profit_take": 0.30,
    "stop_loss_credit_mult": 2.0,
    "close_dte": 10,
    "min_credit": 0.20,
}

WINNER_STATS = {
    "n_trades": 1495,
    "total_pnl_usd": 26472,
    "mean_per_trade_usd": 17.7,
    "median_per_trade_usd": 28.9,
    "std_per_trade_usd": 52.4,
    "win_rate_pct": 91.4,
    "max_win_usd": 70.8,
    "max_loss_usd": -298.1,
    "max_drawdown_pct": -14.7,
    "sharpe": 1.34,
    "exit_reason_counts": {
        "profit_target": 1343,
        "stop_loss": 46,
        "dte_close": 98,
        "end_forced": 8,
    },
    "yearly_pnl_usd": {
        "2019": 3535, "2020": 2332, "2021": 6602, "2022": -561,
        "2023": 4658, "2024": 3673, "2025": 4091, "2026_partial": 2144,
    },
    "covid_march_2020_trades": 1,  # vrp gate kept it out
    "data_source": "PROXY: SPY daily closes + saved ATM IV / VIX proxy + Black-Scholes pricing",
    "execution_assumptions": "Mid-price fills. No bid/ask spread, no commissions, no slippage.",
}

RED_TEAMERS = [
    {
        "role": "execution_realism_attacker",
        "model": "qwen/qwen3-coder-480b-a35b-instruct",
        "mission": (
            "You are a hostile execution-realism reviewer. Find every assumption in "
            "the proxy backtest that would degrade in real markets: bid/ask spreads, "
            "slippage, partial fills, market hours, commission, exchange routing. "
            "Quantify the realistic drag in basis points or dollars per trade if you can."
        ),
    },
    {
        "role": "regime_overfit_attacker",
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "mission": (
            "You are a hostile regime-overfit reviewer. The strategy was tested 2019-2026 "
            "which was largely a low-vol bull market with one COVID spike. Identify "
            "specific regimes where this strategy concept (delta-0.12 daily-ladder iron "
            "condor with profit_take=0.30 and stop=2x credit) would systematically fail. "
            "Be concrete with historical examples (1987, 2000, 2008, 2018-VIXmageddon, etc)."
        ),
    },
    {
        "role": "selection_bias_attacker",
        "model": "moonshotai/kimi-k2-instruct",
        "mission": (
            "You are a hostile selection-bias reviewer. The council ran 61 model-proposed "
            "variants and picked the best risk-adjusted one. Quantify the multiple-comparisons "
            "/ p-hacking concern. How likely is it that a random walk with the same "
            "constraints would also produce a 'winner' with Sharpe >1.3? What walk-forward "
            "tests would isolate true edge from overfitting?"
        ),
    },
    {
        "role": "vrp_gate_attacker",
        "model": "mistralai/mistral-nemotron",
        "mission": (
            "You are a hostile vol-gate reviewer. The strategy's risk control depends on "
            "a 'vrp_positive' gate that kept it out of the COVID crash (only 1 trade Mar-Apr "
            "2020). Examine: how is vrp computed? Is it lookahead-biased? Does it use "
            "future-looking data? What happens if the gate is laggy/stale in live trading? "
            "Could the gate be an artifact that wouldn't trigger correctly in real-time?"
        ),
    },
    {
        "role": "liquidity_attacker",
        "model": "qwen/qwen3-coder-480b-a35b-instruct",
        "mission": (
            "You are a hostile liquidity reviewer. Daily-ladder iron condors at delta=0.12 "
            "with $5 wings on SPY at the 1495 trade rate means routinely placing 4-leg "
            "orders at unusual strikes. Examine real SPY option-chain liquidity at those "
            "OTM strikes during 2019-2026. Can you actually fill 1495 trades at mid-price "
            "with $5 wings? At what trade-rate does the strategy's own activity move the market?"
        ),
    },
    {
        "role": "investor_devil_advocate",
        "model": "bytedance/seed-oss-36b-instruct",
        "mission": (
            "You are a hostile investor-asking-hard-questions reviewer. The user wants to "
            "show this strategy to an investor tomorrow as a paper-month demo. The strategy "
            "shows +264% on PROXY data (not real fills). What 5 questions would a sophisticated "
            "investor (e.g., from a hedge fund FoF) ask that would expose weaknesses? What "
            "should the user NOT say? What honest disclosure is required?"
        ),
    },
]

PROMPT_TEMPLATE = """You are a hostile reviewer in a trading strategy red-team.
Your role: {role}.
Mission: {mission}

Strategy under review: {winner_name}
Spec:
{spec}

Backtest results (PROXY data only):
{stats}

Important caveats for your review:
- Data source: {data_source}
- Execution assumptions: {execution}
- This is NOT live broker fills. Bid/ask spreads, commissions, slippage are absent.
- No live option-chain fetching; the live runner script's chain fetch is currently a stub returning None.
- 60% of the council's 61 proposed variants share the same entry_mode + gate (daily_ladder + vrp_positive).

Return JSON only. No markdown.

Schema:
{{
  "role": "{role}",
  "primary_attack": "one-sentence headline weakness",
  "attack_vectors": [
    {{
      "name": "short name",
      "severity": "critical|high|medium|low",
      "mechanism": "how this would degrade real-world performance",
      "estimated_drag": "dollar amount per trade or percentage haircut to the +264% result",
      "test_to_falsify": "specific test that would prove this is/isn't a real risk"
    }}
  ],
  "would_you_paper_trade_this": true,
  "paper_trade_caveats": ["list of caveats if you would paper trade"],
  "would_you_use_real_capital": false,
  "real_capital_blockers": ["specific blockers before real capital deployment"],
  "honest_one_sentence_summary": "what to tell the investor, brutally honest"
}}

Be specific. Cite numbers. No platitudes. No safety boilerplate. The user has already heard the standard 'this is risky' framing and wants concrete attacks."""


def load_local_env() -> None:
    env_path = REPO_ROOT / "council" / ".env.local"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        if line.strip().startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def call_redteamer(client: OpenAI, attacker: dict[str, str]) -> dict[str, Any]:
    prompt = PROMPT_TEMPLATE.format(
        role=attacker["role"],
        mission=attacker["mission"],
        winner_name=WINNER_NAME,
        spec=json.dumps(WINNER_SPEC, indent=2),
        stats=json.dumps(WINNER_STATS, indent=2),
        data_source=WINNER_STATS["data_source"],
        execution=WINNER_STATS["execution_assumptions"],
    )
    try:
        resp = client.chat.completions.create(
            model=attacker["model"],
            messages=[
                {"role": "system", "content": "You output strict JSON red-team analyses. No markdown, no preamble."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=1600,
        )
        content = resp.choices[0].message.content or ""
        return {"ok": True, "role": attacker["role"], "model": attacker["model"], "content": content}
    except Exception as exc:
        return {"ok": False, "role": attacker["role"], "model": attacker["model"], "error": f"{type(exc).__name__}: {exc}", "content": ""}


def parse_json_loose(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    load_local_env()
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        print("NVIDIA_API_KEY not set", file=sys.stderr)
        return 1
    client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1", timeout=120.0)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "council" / "redteam_runs" / stamp
    raw_dir = out_dir / "raw_responses"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Red-team run: {out_dir}", flush=True)
    print(f"Reviewing winner: {WINNER_NAME}", flush=True)
    print(f"Calling {len(RED_TEAMERS)} hostile reviewers via NIM\n", flush=True)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(call_redteamer, client, a): a for a in RED_TEAMERS}
        for fut in concurrent.futures.as_completed(futures):
            payload = fut.result()
            results.append(payload)
            stem = payload["role"]
            (raw_dir / f"{stem}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            (raw_dir / f"{stem}.md").write_text(str(payload.get("content", "")), encoding="utf-8")
            print(f"  done {stem} via {payload['model']} ok={payload.get('ok')}", flush=True)

    # Synthesize attack vectors
    all_attacks = []
    for r in results:
        if not r.get("ok"):
            continue
        parsed = parse_json_loose(str(r.get("content", "")))
        if parsed is None:
            continue
        for v in parsed.get("attack_vectors", []) or []:
            if isinstance(v, dict):
                v = dict(v)
                v["from_role"] = parsed.get("role", r["role"])
                v["from_model"] = r["model"]
                all_attacks.append(v)

    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_attacks.sort(key=lambda x: severity_order.get(str(x.get("severity", "medium")).lower(), 4))

    summary = {
        "winner_reviewed": WINNER_NAME,
        "spec": WINNER_SPEC,
        "n_reviewers_attempted": len(RED_TEAMERS),
        "n_reviewers_completed": sum(1 for r in results if r.get("ok")),
        "all_attack_vectors_ranked": all_attacks,
        "individual_summaries": [
            {
                "role": parse_json_loose(str(r.get("content", ""))).get("role")
                if r.get("ok") and parse_json_loose(str(r.get("content", ""))) else r["role"],
                "model": r["model"],
                "primary_attack": (parse_json_loose(str(r.get("content", ""))) or {}).get("primary_attack"),
                "would_paper_trade": (parse_json_loose(str(r.get("content", ""))) or {}).get("would_you_paper_trade_this"),
                "would_use_real_capital": (parse_json_loose(str(r.get("content", ""))) or {}).get("would_you_use_real_capital"),
                "honest_summary": (parse_json_loose(str(r.get("content", ""))) or {}).get("honest_one_sentence_summary"),
            }
            for r in results if r.get("ok")
        ],
    }
    (out_dir / "redteam_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [
        f"# Red-Team Review: {WINNER_NAME}",
        "",
        f"Reviewers: {summary['n_reviewers_completed']}/{summary['n_reviewers_attempted']} completed",
        "",
        "## Individual reviewer summaries",
        "",
    ]
    for s in summary["individual_summaries"]:
        md_lines.append(f"**{s['role']}** ({s['model']})")
        md_lines.append(f"- Primary attack: {s['primary_attack']}")
        md_lines.append(f"- Paper trade? {s['would_paper_trade']}")
        md_lines.append(f"- Real capital? {s['would_use_real_capital']}")
        md_lines.append(f"- Verdict: {s['honest_summary']}")
        md_lines.append("")
    md_lines.append("## Top attack vectors ranked by severity")
    md_lines.append("")
    for v in all_attacks[:25]:
        md_lines.append(
            f"- **{v.get('severity','?').upper()}**: {v.get('name','?')} "
            f"(from {v.get('from_role','?')}) — {v.get('mechanism','?')[:200]}"
        )
        if v.get('estimated_drag'):
            md_lines.append(f"  - Estimated drag: {v['estimated_drag']}")
        if v.get('test_to_falsify'):
            md_lines.append(f"  - Test: {v['test_to_falsify']}")
        md_lines.append("")
    (out_dir / "redteam_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nReport: {out_dir / 'redteam_report.md'}", flush=True)
    print(f"JSON: {out_dir / 'redteam_summary.json'}", flush=True)
    print(f"\nTop 5 attack vectors:", flush=True)
    for v in all_attacks[:5]:
        print(f"  [{v.get('severity','?').upper():8}] {v.get('name','?')} (from {v.get('from_role','?')})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
