"""Aggressive NVIDIA NIM research lab for generating experiment ideas.

This is intentionally broader than `run_council.py`.

It asks many models for structured experiment proposals, extracts validated
SPY-core strategy specs, and writes a JSON file that can be executed by:

  .venv/bin/python experiments/run_spy_core_strategy_grid.py \
    --spec-json council/lab_runs/<timestamp>/validated_spy_core_specs.json \
    --output-dir experiments/runs/council_spy_grid_<timestamp>

The LLMs do not directly trade or edit the live strategy. They propose tests;
local deterministic code validates and runs only allowed parameter ranges.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from council.run_council import (  # noqa: E402
    DEFAULT_CONFIG,
    DEFAULT_GRID_RUN,
    load_local_env,
    read_json,
)
from experiments.run_spy_core_strategy_grid import validate_strategy_spec  # noqa: E402

DEFAULT_OUTPUT_ROOT = REPO_ROOT / "council" / "lab_runs"

TASKS = {
    "spy_core": "Design SPY iron-condor variants that improve risk-adjusted return versus the existing 30-strategy grid.",
    "risk": "Design risk-control experiments that reduce drawdown without killing edge.",
    "execution": "Design execution-realism tests: slippage, bid/ask, fill timing, mark-to-market, and stop behavior.",
    "satellite": "Design earnings-satellite experiments and features to test next.",
    "portfolio": "Design portfolio allocation experiments across SPY core, satellite, cash, and SPY benchmark.",
    "data": "Design data-quality experiments to distinguish true edge from proxy-data artifacts.",
}


@dataclass(frozen=True)
class LabCall:
    role: str
    model: str
    task: str
    mission: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    propose = sub.add_parser("propose", help="Ask many NIM models for structured experiment proposals")
    propose.add_argument("--mode", choices=["core", "expanded"], default="expanded")
    propose.add_argument("--tasks", default="spy_core,risk,execution,satellite,portfolio,data")
    propose.add_argument("--max-calls", type=int, default=30)
    propose.add_argument("--workers", type=int, default=6)
    propose.add_argument("--dry-run", action="store_true")
    propose.add_argument("--skip-unavailable", action="store_true")
    propose.add_argument("--grid-run-dir", type=Path, default=DEFAULT_GRID_RUN)
    propose.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    finalize = sub.add_parser("finalize", help="Build validated specs from an existing partial/complete lab run")
    finalize.add_argument("--run", default="latest", help="Lab run directory or 'latest'")
    finalize.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    return parser.parse_args()


def main() -> int:
    load_local_env()
    args = parse_args()
    cfg = read_json(args.config)
    if args.command == "propose":
        return propose(args, cfg)
    if args.command == "finalize":
        return finalize(args)
    raise AssertionError(args.command)


def propose(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    output_dir = make_run_dir(args.output_root)
    context = build_compact_context(args.grid_run_dir)
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    unknown = [t for t in tasks if t not in TASKS]
    if unknown:
        raise ValueError(f"unknown task(s): {unknown}; allowed={sorted(TASKS)}")

    members = lab_members_from_config(cfg)
    calls = build_calls(members, tasks, args.max_calls)
    write_json(
        output_dir / "manifest.json",
        {
            "created_at_utc": now_utc(),
            "mode": args.mode,
            "tasks": tasks,
            "max_calls": args.max_calls,
            "calls": [c.__dict__ for c in calls],
        },
    )
    (output_dir / "context.md").write_text(context, encoding="utf-8")

    prompt_dir = output_dir / "prompts"
    raw_dir = output_dir / "raw_responses"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    for call in calls:
        prompt = proposal_prompt(context, call)
        (prompt_dir / f"{call.role}__{call.task}.md").write_text(prompt, encoding="utf-8")

    if args.dry_run:
        print(f"dry lab packet written: {output_dir}", flush=True)
        return 0

    client = client_from_config(cfg)
    if args.skip_unavailable:
        available = {m.id for m in client.models.list().data}
        before = len(calls)
        calls = [c for c in calls if c.model in available]
        print(f"using {len(calls)}/{before} available calls", flush=True)

    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(call_lab_model, client, cfg, context, call): call for call in calls}
        for fut in concurrent.futures.as_completed(futures):
            call = futures[fut]
            payload = fut.result()
            results.append(payload)
            stem = f"{call.role}__{call.task}"
            write_json(raw_dir / f"{stem}.json", payload)
            (raw_dir / f"{stem}.md").write_text(str(payload.get("content", "")), encoding="utf-8")
            print(f"done {stem} via {call.model} ok={payload.get('ok')}", flush=True)

    specs, rejected = collect_validated_specs(results)
    write_json(output_dir / "validated_spy_core_specs.json", {"strategies": specs})
    write_json(output_dir / "rejected_specs.json", rejected)
    write_markdown_summary(output_dir / "lab_summary.md", results, specs, rejected)

    print(f"\nLab complete: {output_dir}", flush=True)
    print(f"Validated SPY specs: {len(specs)}", flush=True)
    print(f"Run them with:", flush=True)
    print(
        ".venv/bin/python experiments/run_spy_core_strategy_grid.py "
        f"--spec-json {output_dir / 'validated_spy_core_specs.json'} "
        f"--output-dir experiments/runs/council_spy_grid_{output_dir.name}",
        flush=True,
    )
    return 0


def finalize(args: argparse.Namespace) -> int:
    output_dir = resolve_run_dir(args.run, args.output_root)
    raw_dir = output_dir / "raw_responses"
    if not raw_dir.exists():
        raise RuntimeError(f"missing raw_responses directory: {raw_dir}")
    results: list[dict[str, Any]] = []
    for path in sorted(raw_dir.glob("*.json")):
        results.append(read_json(path))
    specs, rejected = collect_validated_specs(results)
    write_json(output_dir / "validated_spy_core_specs.json", {"strategies": specs})
    write_json(output_dir / "rejected_specs.json", rejected)
    write_markdown_summary(output_dir / "lab_summary.md", results, specs, rejected)
    print(f"Finalized: {output_dir}", flush=True)
    print(f"Raw responses: {len(results)}", flush=True)
    print(f"Validated SPY specs: {len(specs)}", flush=True)
    print(
        ".venv/bin/python experiments/run_spy_core_strategy_grid.py "
        f"--spec-json {output_dir / 'validated_spy_core_specs.json'} "
        f"--output-dir experiments/runs/council_spy_grid_{output_dir.name}",
        flush=True,
    )
    return 0


def resolve_run_dir(value: str, output_root: Path) -> Path:
    if value != "latest":
        path = Path(value)
        return path if path.is_absolute() else (REPO_ROOT / path)
    runs = sorted([p for p in output_root.glob("*") if p.is_dir()])
    if not runs:
        raise RuntimeError(f"no lab runs found under {output_root}")
    return runs[-1]


def lab_members_from_config(cfg: dict[str, Any]) -> list[Any]:
    raw = cfg.get("lab_swarm") or []
    if not raw:
        from council.run_council import configured_members

        return configured_members(cfg, "expanded")

    @dataclass(frozen=True)
    class Member:
        role: str
        model: str
        mission: str

    return [Member(str(item["role"]), str(item["model"]), str(item["mission"])) for item in raw]


def build_calls(members: list[Any], tasks: list[str], max_calls: int) -> list[LabCall]:
    calls: list[LabCall] = []
    for task in tasks:
        for member in members:
            calls.append(
                LabCall(
                    role=str(member.role),
                    model=str(member.model),
                    task=task,
                    mission=str(member.mission),
                )
            )
            if len(calls) >= max_calls:
                return calls
    return calls


def build_compact_context(grid_run_dir: Path) -> str:
    summary_path = grid_run_dir / "summary.csv"
    if not summary_path.exists():
        raise RuntimeError(f"missing grid summary: {summary_path}")
    import pandas as pd

    summary = pd.read_csv(summary_path)
    top = summary.head(12)[
        [
            "strategy",
            "entry_mode",
            "gate",
            "target_delta",
            "target_dte",
            "wing_width",
            "trades",
            "final_nav",
            "return_pct",
            "max_drawdown_pct",
            "sharpe",
            "max_open_trades",
        ]
    ]
    groups = (
        summary.sort_values("final_nav", ascending=False)
        .groupby(["entry_mode", "gate"], as_index=False)
        .first()[["entry_mode", "gate", "strategy", "final_nav", "return_pct", "max_drawdown_pct", "sharpe"]]
    )
    return f"""# Current Research State

We have a proxy SPY iron-condor grid from 2019-01-31 to 2026-05-08.

Important caveat:
- This is not live broker execution data.
- It uses real SPY daily closes, saved ATM IV where available, VIX proxy otherwise, and Black-Scholes pricing.
- Current live runner does not yet have real option-chain fetching wired.

SPY buy-and-hold benchmark over same period: $10,000 -> $30,468.08 (+204.68%).

Top strategies:
{top.to_string(index=False)}

Best by family:
{groups.to_string(index=False)}

Existing council preference:
- Continue testing daily_ladder_vrp_positive_d16_45d_w5.
- Avoid aggressive_daily_nogate_d20_30d_w10 for now because of ~-50% drawdown.

Allowed executable SPY-core parameters:
- entry_mode: single_reentry, daily_ladder, mwf_ladder
- gate: none, vrp_positive, iv_rank_50, high_iv_volume
- target_delta: 0.05 to 0.35
- target_dte: 14 to 75
- wing_width: 1 to 25
- profit_take: 0.10 to 0.90
- stop_loss_credit_mult: 0.50 to 5.00
- close_dte: >=1 and less than target_dte
- min_credit: 0.01 to 3.00
"""


def proposal_prompt(context: str, call: LabCall) -> str:
    return f"""You are part of an aggressive trading research lab.

Role: {call.role}
Model: {call.model}
Mission: {call.mission}
Task: {call.task} - {TASKS[call.task]}

{context}

Return ONLY valid JSON. No Markdown.

Schema:
{{
  "role": "{call.role}",
  "task": "{call.task}",
  "verdict": "short practical verdict",
  "research_ideas": [
    {{
      "name": "short name",
      "hypothesis": "what this tests",
      "why_it_might_help": "reason",
      "failure_mode": "how this can fool us"
    }}
  ],
  "spy_core_strategies": [
    {{
      "name": "safe_unique_name",
      "entry_mode": "single_reentry|daily_ladder|mwf_ladder",
      "gate": "none|vrp_positive|iv_rank_50|high_iv_volume",
      "target_delta": 0.12,
      "target_dte": 30,
      "wing_width": 5,
      "profit_take": 0.5,
      "stop_loss_credit_mult": 2.0,
      "close_dte": 21,
      "min_credit": 0.20,
      "rationale": "why this exact configuration deserves a test"
    }}
  ],
  "non_spy_tests": [
    {{
      "name": "test name",
      "category": "satellite|portfolio|execution|data_quality",
      "description": "specific test to implement later"
    }}
  ],
  "investor_note": "one sentence"
}}

Rules:
- Propose 2 to 5 SPY-core strategies when relevant.
- If your task is not SPY-core, keep spy_core_strategies empty and focus on non_spy_tests.
- Do not propose live capital deployment.
- Do not claim proxy data is live data.
- Prefer tests that can run tonight.
"""


def client_from_config(cfg: dict[str, Any]) -> OpenAI:
    key_name = cfg.get("env_api_key", "NVIDIA_API_KEY")
    key = os.environ.get(key_name)
    if not key:
        raise RuntimeError(f"{key_name} is not set")
    return OpenAI(
        api_key=key,
        base_url=cfg.get("base_url", "https://integrate.api.nvidia.com/v1"),
        timeout=float(cfg.get("request_timeout_seconds", 120)),
    )


def call_lab_model(client: OpenAI, cfg: dict[str, Any], context: str, call: LabCall) -> dict[str, Any]:
    prompt = proposal_prompt(context, call)
    try:
        response = client.chat.completions.create(
            model=call.model,
            messages=[
                {
                    "role": "system",
                    "content": "You output strict JSON experiment proposals. No Markdown.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.35,
            max_tokens=min(int(cfg.get("max_tokens", 1800)), 1600),
        )
        content = response.choices[0].message.content or ""
        return {
            "ok": True,
            "role": call.role,
            "model": call.model,
            "task": call.task,
            "content": content,
        }
    except Exception as exc:
        return {
            "ok": False,
            "role": call.role,
            "model": call.model,
            "task": call.task,
            "error": f"{type(exc).__name__}: {exc}",
            "content": "",
        }


def collect_validated_specs(results: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    specs: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for result in results:
        if not result.get("ok"):
            continue
        parsed = parse_jsonish(str(result.get("content", "")))
        if parsed is None:
            rejected.append({"source": result_meta(result), "reason": "could_not_parse_json"})
            continue
        for raw in parsed.get("spy_core_strategies", []) if isinstance(parsed, dict) else []:
            if not isinstance(raw, dict):
                rejected.append({"source": result_meta(result), "raw": raw, "reason": "strategy_not_object"})
                continue
            raw = dict(raw)
            raw["name"] = prefix_name(result, str(raw.get("name", "strategy")))
            try:
                spec = validate_strategy_spec(raw, len(specs) + 1)
            except Exception as exc:
                rejected.append({"source": result_meta(result), "raw": raw, "reason": str(exc)})
                continue
            spec_dict = spec.__dict__.copy()
            if spec_dict["name"] in seen:
                continue
            seen.add(spec_dict["name"])
            specs.append(spec_dict)
    return specs, rejected


def parse_jsonish(text: str) -> dict[str, Any] | None:
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


def prefix_name(result: dict[str, Any], name: str) -> str:
    base = f"{result.get('task', 'task')}_{result.get('role', 'role')}_{name}"
    base = base.lower()
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in base).strip("_")[:96]


def result_meta(result: dict[str, Any]) -> dict[str, Any]:
    return {"role": result.get("role"), "model": result.get("model"), "task": result.get("task")}


def write_markdown_summary(path: Path, results: list[dict[str, Any]], specs: list[dict[str, Any]], rejected: list[dict[str, Any]]) -> None:
    ok = [r for r in results if r.get("ok")]
    lines = [
        "# Council Research Lab Summary",
        "",
        f"Completed model calls: {len(ok)}/{len(results)}",
        f"Validated SPY-core specs: {len(specs)}",
        f"Rejected specs: {len(rejected)}",
        "",
        "## Validated Specs",
        "",
    ]
    for spec in specs:
        lines.append(
            f"- `{spec['name']}`: {spec['entry_mode']} {spec['gate']} "
            f"delta={spec['target_delta']} dte={spec['target_dte']} wing={spec['wing_width']} "
            f"pt={spec['profit_take']} sl={spec['stop_loss_credit_mult']} close_dte={spec['close_dte']}"
        )
    lines.append("")
    lines.append("## Failed Calls")
    lines.append("")
    for result in results:
        if not result.get("ok"):
            lines.append(f"- {result.get('role')} / {result.get('task')} / {result.get('model')}: {result.get('error')}")
    path.write_text("\n".join(lines), encoding="utf-8")


def make_run_dir(root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = root / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
