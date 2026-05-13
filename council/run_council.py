"""Run an NVIDIA NIM LLM council over local trading research outputs.

Usage:
  export NVIDIA_API_KEY="..."
  .venv/bin/python council/run_council.py list-models
  .venv/bin/python council/run_council.py run --mode core
  .venv/bin/python council/run_council.py run --mode expanded --max-members 12
  .venv/bin/python council/run_council.py ask --run latest --question "Which strategy should we paper next?"

No API key is ever written to disk.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sqlite3
import sys
import textwrap
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from council.prompts import ASK_USER_TEMPLATE, CHAIR_USER_TEMPLATE, MEMBER_USER_TEMPLATE, ROLE_SYSTEM_TEMPLATE

DEFAULT_CONFIG = REPO_ROOT / "council" / "config.default.json"
DEFAULT_GRID_RUN = REPO_ROOT / "experiments" / "runs" / "spy_core_30_strategy_grid_2019_2026"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "council" / "runs"
LOCAL_ENV = REPO_ROOT / "council" / ".env.local"
OFFICIAL_RUN_ID = "paper_v1_evidence_20260510T131633Z"


@dataclass(frozen=True)
class CouncilMember:
    role: str
    model: str
    mission: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    sub = parser.add_subparsers(dest="command", required=True)

    list_models = sub.add_parser("list-models", help="List model ids available to your NVIDIA key")
    list_models.add_argument("--contains", default="", help="Optional case-insensitive filter")

    validate = sub.add_parser("validate-models", help="Check configured council models against /v1/models")
    validate.add_argument("--mode", choices=["core", "expanded", "all"], default="core")

    run = sub.add_parser("run", help="Run the council")
    run.add_argument("--mode", choices=["core", "expanded"], default="core")
    run.add_argument("--max-members", type=int, default=None)
    run.add_argument("--grid-run-dir", type=Path, default=DEFAULT_GRID_RUN)
    run.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    run.add_argument("--dry-run", action="store_true", help="Write prompts only; do not call NVIDIA")
    run.add_argument("--skip-unavailable", action="store_true", help="Validate and skip unavailable model ids before running")

    ask = sub.add_parser("ask", help="Ask a follow-up question against a previous council run")
    ask.add_argument("--run", default="latest", help="Run directory or 'latest'")
    ask.add_argument("--question", required=True)
    ask.add_argument("--model", default=None, help="Override answer model; default uses chair model")
    ask.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    synth = sub.add_parser("synthesize", help="Build a chair report from completed member responses in an existing run")
    synth.add_argument("--run", default="latest", help="Run directory or 'latest'")
    synth.add_argument("--model", default=None, help="Override chair model")
    synth.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    return parser.parse_args()


def main() -> int:
    load_local_env()
    args = parse_args()
    cfg = read_json(args.config)
    if args.command == "list-models":
        return list_models(cfg, contains=args.contains)
    if args.command == "validate-models":
        return validate_models(cfg, args.mode)
    if args.command == "run":
        return run_council(args, cfg)
    if args.command == "ask":
        return ask_followup(args, cfg)
    if args.command == "synthesize":
        return synthesize_existing(args, cfg)
    raise AssertionError(args.command)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def load_local_env(path: Path = LOCAL_ENV) -> None:
    """Load simple KEY=VALUE pairs from council/.env.local without overwriting env."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def client_from_config(cfg: dict[str, Any]) -> OpenAI:
    key_name = cfg.get("env_api_key", "NVIDIA_API_KEY")
    api_key = os.environ.get(key_name)
    if not api_key:
        raise RuntimeError(f"{key_name} is not set. Run: export {key_name}='your_nvidia_key'")
    return OpenAI(
        api_key=api_key,
        base_url=cfg.get("base_url", "https://integrate.api.nvidia.com/v1"),
        timeout=float(cfg.get("request_timeout_seconds", 120)),
    )


def list_models(cfg: dict[str, Any], *, contains: str = "") -> int:
    client = client_from_config(cfg)
    models = sorted(m.id for m in client.models.list().data)
    if contains:
        needle = contains.lower()
        models = [m for m in models if needle in m.lower()]
    for model in models:
        print(model)
    print(f"\n{len(models)} model(s)")
    return 0


def configured_members(cfg: dict[str, Any], mode: str) -> list[CouncilMember]:
    raw: list[dict[str, str]] = []
    if mode in ("core", "all"):
        raw.extend(cfg.get("core_council", []))
    if mode in ("expanded", "all"):
        raw.extend(cfg.get("core_council", []))
        raw.extend(cfg.get("expanded_pool", []))
    seen: set[tuple[str, str]] = set()
    members: list[CouncilMember] = []
    for item in raw:
        key = (item["role"], item["model"])
        if key in seen:
            continue
        seen.add(key)
        members.append(CouncilMember(item["role"], item["model"], item["mission"]))
    return members


def validate_models(cfg: dict[str, Any], mode: str) -> int:
    client = client_from_config(cfg)
    available = {m.id for m in client.models.list().data}
    members = configured_members(cfg, mode)
    missing = []
    for member in members:
        ok = member.model in available
        print(f"{'OK' if ok else 'MISS'} {member.role:30s} {member.model}")
        if not ok:
            missing.append(member.model)
    print(f"\n{len(members) - len(missing)}/{len(members)} configured model ids available")
    return 1 if missing else 0


def run_council(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    members = configured_members(cfg, args.mode)
    if args.max_members is not None:
        members = members[: args.max_members]
    if not members:
        raise RuntimeError("no council members selected")

    if args.skip_unavailable and not args.dry_run:
        client = client_from_config(cfg)
        available = {m.id for m in client.models.list().data}
        before = len(members)
        members = [m for m in members if m.model in available]
        print(f"using {len(members)}/{before} available configured model ids")
        if not members:
            raise RuntimeError("all configured models were unavailable")

    run_dir = make_run_dir(args.output_root)
    context = build_context(args.grid_run_dir)
    (run_dir / "context.md").write_text(context, encoding="utf-8")
    write_json(
        run_dir / "manifest.json",
        {
            "created_at_utc": now_utc(),
            "mode": args.mode,
            "dry_run": args.dry_run,
            "grid_run_dir": str(args.grid_run_dir),
            "members": [m.__dict__ for m in members],
        },
    )

    prompt_dir = run_dir / "prompts"
    response_dir = run_dir / "member_responses"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    response_dir.mkdir(parents=True, exist_ok=True)

    for member in members:
        system, user = prompts_for_member(member, context)
        (prompt_dir / f"{member.role}.system.md").write_text(system, encoding="utf-8")
        (prompt_dir / f"{member.role}.user.md").write_text(user, encoding="utf-8")

    if args.dry_run or not os.environ.get(cfg.get("env_api_key", "NVIDIA_API_KEY")):
        print(f"dry prompt packet written: {run_dir}")
        print("Set NVIDIA_API_KEY and rerun without --dry-run to call NIM.")
        return 0

    client = client_from_config(cfg)
    temperature = float(cfg.get("temperature", 0.2))
    max_tokens = int(cfg.get("max_tokens", 1800))
    workers = min(int(cfg.get("parallel_workers", 4)), len(members))

    responses: dict[str, dict[str, Any]] = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(call_member, client, member, context, temperature, max_tokens): member
            for member in members
            if member.role != "chair_synthesizer"
        }
        for fut in concurrent.futures.as_completed(futures):
            member = futures[fut]
            payload = fut.result()
            responses[member.role] = payload
            write_json(response_dir / f"{member.role}.json", payload)
            (response_dir / f"{member.role}.md").write_text(payload.get("content", ""), encoding="utf-8")
            print(f"done {member.role} via {member.model}")

    chair = next((m for m in members if m.role == "chair_synthesizer"), members[0])
    chair_payload = call_chair(client, chair, context, responses, temperature, max_tokens, cfg)
    responses[chair.role] = chair_payload
    write_json(response_dir / f"{chair.role}.json", chair_payload)
    (response_dir / f"{chair.role}.md").write_text(chair_payload.get("content", ""), encoding="utf-8")
    (run_dir / "council_report.md").write_text(chair_payload.get("content", ""), encoding="utf-8")

    write_json(run_dir / "responses_index.json", responses)
    print(f"\nCouncil complete: {run_dir}")
    print(f"Report: {run_dir / 'council_report.md'}")
    return 0


def prompts_for_member(member: CouncilMember, context: str) -> tuple[str, str]:
    system = ROLE_SYSTEM_TEMPLATE.format(role=member.role, mission=member.mission)
    user = MEMBER_USER_TEMPLATE.format(context=context)
    return system, user


def call_member(
    client: OpenAI,
    member: CouncilMember,
    context: str,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    system, user = prompts_for_member(member, context)
    return call_chat(client, member.model, system, user, temperature, max_tokens, member.role)


def call_chair(
    client: OpenAI,
    chair: CouncilMember,
    context: str,
    responses: dict[str, dict[str, Any]],
    temperature: float,
    max_tokens: int,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    limit = int((cfg or {}).get("chair_response_char_limit", 3000))
    response_text = "\n\n".join(
        f"## {role}\nModel: {payload.get('model')}\n\n{trim_for_chair(str(payload.get('content', '')), limit)}"
        for role, payload in sorted(responses.items())
    )
    system = ROLE_SYSTEM_TEMPLATE.format(role=chair.role, mission=chair.mission)
    user = CHAIR_USER_TEMPLATE.format(context=context, responses=response_text)
    return call_chat(client, chair.model, system, user, temperature, max_tokens + 800, chair.role)


def trim_for_chair(content: str, limit: int) -> str:
    content = content.strip()
    if len(content) <= limit:
        return content
    return content[:limit].rstrip() + "\n\n[trimmed for chair synthesis]"


def call_chat(
    client: OpenAI,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    role: str,
) -> dict[str, Any]:
    started = now_utc()
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        content = completion.choices[0].message.content or ""
        return {
            "role": role,
            "model": model,
            "started_at_utc": started,
            "completed_at_utc": now_utc(),
            "ok": True,
            "content": content,
            "usage": getattr(completion, "usage", None).model_dump() if getattr(completion, "usage", None) else None,
        }
    except Exception as exc:  # keep council runs auditable when one endpoint fails
        return {
            "role": role,
            "model": model,
            "started_at_utc": started,
            "completed_at_utc": now_utc(),
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "content": f"ERROR from {model}: {type(exc).__name__}: {exc}",
        }


def ask_followup(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    run_dir = resolve_run_dir(args.run, args.output_root)
    context = (run_dir / "context.md").read_text(encoding="utf-8")
    report = (run_dir / "council_report.md").read_text(encoding="utf-8")
    chair = configured_members(cfg, "core")[0]
    model = args.model or chair.model
    system = "You answer follow-up questions for a trading research council. Be direct, skeptical, and practical."
    user = ASK_USER_TEMPLATE.format(context=context, report=report, question=args.question)

    client = client_from_config(cfg)
    payload = call_chat(
        client,
        model,
        system,
        user,
        float(cfg.get("temperature", 0.2)),
        int(cfg.get("max_tokens", 1800)),
        "followup",
    )
    out_dir = run_dir / "followups"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    write_json(out_dir / f"{stamp}.json", {"question": args.question, **payload})
    md_path = out_dir / f"{stamp}.md"
    md_path.write_text(f"# Question\n\n{args.question}\n\n# Answer\n\n{payload['content']}\n", encoding="utf-8")
    print(payload["content"])
    print(f"\nSaved: {md_path}")
    return 0


def synthesize_existing(args: argparse.Namespace, cfg: dict[str, Any]) -> int:
    run_dir = resolve_run_dir(args.run, args.output_root)
    context_path = run_dir / "context.md"
    response_dir = run_dir / "member_responses"
    if not context_path.exists():
        raise RuntimeError(f"missing context file: {context_path}")
    if not response_dir.exists():
        raise RuntimeError(f"missing response directory: {response_dir}")

    responses: dict[str, dict[str, Any]] = {}
    for path in sorted(response_dir.glob("*.json")):
        payload = read_json(path)
        if payload.get("role") == "chair_synthesizer":
            continue
        content = str(payload.get("content", "")).strip()
        if not content or content.startswith("ERROR from"):
            continue
        responses[str(payload.get("role") or path.stem)] = payload

    if not responses:
        raise RuntimeError(f"no completed member responses found in {response_dir}")

    context = context_path.read_text(encoding="utf-8")
    chair = configured_members(cfg, "core")[0]
    if args.model:
        chair = CouncilMember("chair_synthesizer", args.model, chair.mission)

    client = client_from_config(cfg)
    payload = call_chair(
        client,
        chair,
        context,
        responses,
        float(cfg.get("temperature", 0.2)),
        int(cfg.get("max_tokens", 1800)),
        cfg,
    )
    write_json(response_dir / "chair_synthesizer.json", payload)
    (response_dir / "chair_synthesizer.md").write_text(payload.get("content", ""), encoding="utf-8")
    (run_dir / "council_report.md").write_text(payload.get("content", ""), encoding="utf-8")
    print(f"synthesized from {len(responses)} completed member response(s)")
    print(f"Report: {run_dir / 'council_report.md'}")
    return 0


def resolve_run_dir(value: str, output_root: Path) -> Path:
    if value != "latest":
        path = Path(value)
        return path if path.is_absolute() else (REPO_ROOT / path)
    runs = sorted([p for p in output_root.glob("*") if p.is_dir()])
    if not runs:
        raise RuntimeError(f"no council runs found under {output_root}")
    return runs[-1]


def build_context(grid_run_dir: Path) -> str:
    summary = pd.read_csv(grid_run_dir / "summary.csv")
    nav = pd.read_csv(grid_run_dir / "daily_nav.csv")
    trades = pd.read_csv(grid_run_dir / "trades.csv")

    start = nav["date"].min()
    end = nav["date"].max()
    bench = benchmark_summary(nav)
    top10 = summary.head(10)[
        [
            "strategy",
            "trades",
            "final_nav",
            "return_pct",
            "max_drawdown_pct",
            "sharpe",
            "win_rate_pct",
            "max_open_trades",
        ]
    ]
    grouped = (
        summary.sort_values("final_nav", ascending=False)
        .groupby(["entry_mode", "gate"], as_index=False)
        .first()[
            [
                "entry_mode",
                "gate",
                "strategy",
                "trades",
                "final_nav",
                "return_pct",
                "max_drawdown_pct",
                "sharpe",
                "max_open_trades",
            ]
        ]
        .sort_values("final_nav", ascending=False)
    )
    official = official_evidence_summary()
    exit_counts = trades.groupby(["strategy", "reason"]).size().reset_index(name="count")
    top_exit_counts = exit_counts[exit_counts["strategy"].isin(top10["strategy"].head(5))].to_string(index=False)

    context = textwrap.dedent(
        f"""
        # Trading Research Context

        ## Current Question
        We tested 30 SPY iron-condor strategy variants from 2019 to current and need a council recommendation:
        - Which strategy, if any, deserves further paper testing?
        - Which strategies are too risky or likely overfit?
        - What should we honestly tell an investor?

        ## Important Evidence Caveat
        This 2019-current grid is a proxy experiment, not official execution evidence.
        It uses SPY daily closes plus an IV proxy / saved ATM IV and Black-Scholes pricing.
        It does not use real bid/ask option chains for every date from 2019-2026.

        ## Data Vocabulary
        Be precise about data labels:
        - Proxy grid: real SPY daily closes plus IV proxy / saved ATM IV, model-priced with Black-Scholes.
        - Real-priced evidence snapshot: cached historical option pricing files where available, not live broker fills.
        - Live trading data: real-time/near-real-time option quotes, marks, and broker/paper-broker fills collected during forward paper trading.
        We do not yet have true live options execution data for this strategy; the daily live runner's chain fetch path was previously found to be stubbed.

        ## Proxy Grid Period
        Start: {start}
        End: {end}
        Strategies: {len(summary)}
        Trade rows: {len(trades)}

        ## Benchmark
        {bench}

        ## Top 10 Strategies By Final NAV
        {top10.to_string(index=False)}

        ## Best Strategy In Each Group
        {grouped.to_string(index=False)}

        ## Exit Reason Counts For Top 5 Strategies
        {top_exit_counts}

        ## Official / Real-Priced Evidence Snapshot
        {official}

        ## Local Artifacts
        Full 30-strategy chart:
        {grid_run_dir / "equity_curves_30_strategies.html"}

        Grouped charts:
        {grid_run_dir / "grouped" / "index.html"}
        {grid_run_dir / "grouped" / "best_by_group.html"}

        CSVs:
        {grid_run_dir / "summary.csv"}
        {grid_run_dir / "trades.csv"}
        {grid_run_dir / "daily_nav.csv"}
        """
    ).strip()
    return "\n".join(line[8:] if line.startswith("        ") else line for line in context.splitlines())


def benchmark_summary(nav: pd.DataFrame) -> str:
    bench = nav[["date", "spy_close"]].drop_duplicates("date").sort_values("date")
    first = float(bench["spy_close"].iloc[0])
    last = float(bench["spy_close"].iloc[-1])
    final_nav = 10_000.0 * last / first
    return f"SPY buy-and-hold benchmark: $10,000 -> ${final_nav:,.2f} ({(final_nav / 10000 - 1) * 100:+.2f}%)."


def official_evidence_summary() -> str:
    db = REPO_ROOT / "data" / "trading_system.db"
    if not db.exists():
        return "No local trading_system.db found."
    try:
        conn = sqlite3.connect(db)
        rows = conn.execute(
            """
            SELECT sleeve, count(*), sum(realized_pnl_usd), avg(realized_pnl_usd)
            FROM trades
            WHERE run_id = ? AND realized_pnl_usd IS NOT NULL
            GROUP BY sleeve
            """,
            (OFFICIAL_RUN_ID,),
        ).fetchall()
        total = conn.execute(
            """
            SELECT count(*), sum(realized_pnl_usd)
            FROM trades
            WHERE run_id = ? AND realized_pnl_usd IS NOT NULL
            """,
            (OFFICIAL_RUN_ID,),
        ).fetchone()
        conn.close()
    except Exception as exc:
        return f"Could not read official evidence DB: {type(exc).__name__}: {exc}"
    if not total or total[0] == 0:
        return f"No trades found for {OFFICIAL_RUN_ID}."
    lines = [f"Run: {OFFICIAL_RUN_ID}", f"Total: {total[0]} trades, P&L ${total[1]:,.2f}, return on $10k {(total[1] / 10000) * 100:+.2f}%"]
    for sleeve, count, pnl, avg in rows:
        lines.append(f"{sleeve}: {count} trades, P&L ${pnl:,.2f}, avg ${avg:,.2f}")
    return "\n".join(lines)


def make_run_dir(output_root: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = output_root / stamp
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    raise SystemExit(main())
