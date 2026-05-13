"""NVIDIA LLM-only gate for the six aggressive SPY strategies.

This does not use the trained LightGBM model. It gives several NIM LLMs:
  - 70% train-window evidence,
  - the six frozen strategy specs,
  - held-out test trade setups available at entry time,
and asks them to vote TAKE/BLOCK for each candidate entry.

The aggregated LLM gate is then replayed with the same capital-aware simulator
used by `generate_aggressive_six_capital_aware_report.py`.
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

import numpy as np
import pandas as pd
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.generate_aggressive_six_capital_aware_report import (  # noqa: E402
    MODEL_GATE_THRESHOLD,
    add_trade_refs,
    bench_summary,
    benchmark,
    load_specs,
    overview_chart,
    replay_all,
    strategy_chart,
    summary_table,
    trade_table,
)


DEFAULT_ROOT = REPO_ROOT / "experiments" / "runs" / "aggressive_six_70_30_20260510T234527Z"
RUNS_DIR = REPO_ROOT / "experiments" / "runs"

MODELS = [
    {"role": "qwen_coder_480b", "model": "qwen/qwen3-coder-480b-a35b-instruct"},
    {"role": "llama_maverick", "model": "meta/llama-4-maverick-17b-128e-instruct"},
    {"role": "kimi_k2_instruct", "model": "moonshotai/kimi-k2-instruct"},
]


def load_env() -> None:
    env_path = REPO_ROOT / "council" / ".env.local"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = val.strip().strip('"').strip("'")


def parse_json_loose(text: str) -> Any | None:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"(\[.*\]|\{.*\})", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def train_evidence(root: Path) -> str:
    train = pd.read_csv(root / "train" / "summary.csv")
    reason = pd.read_csv(root / "train" / "trades.csv")
    reason_tab = (
        reason.groupby(["strategy", "reason"])
        .agg(trades=("pnl_usd", "size"), total_pnl=("pnl_usd", "sum"), avg_pnl=("pnl_usd", "mean"))
        .reset_index()
    )
    cols = ["strategy", "return_pct", "max_drawdown_pct", "trades", "win_rate_pct", "worst_trade_usd", "max_open_trades"]
    lines = ["TRAIN WINDOW EVIDENCE (70%, proxy backtest, one-contract base):"]
    lines.append(train[cols].to_string(index=False))
    lines.append("\nTRAIN EXIT REASON BREAKDOWN:")
    lines.append(reason_tab.to_string(index=False))
    lines.append(
        "\nObjective: choose aggressive but survivable SPY short-vol / iron-condor entries for the held-out test period. "
        "Favor high return, but avoid setups that look likely to trigger stop_loss or deep drawdown."
    )
    return "\n".join(lines)


def strategy_map() -> dict[str, str]:
    return {
        "S1": "aggressive_daily_nogate_d20_30d_w10",
        "S2": "aggressive_daily_nogate_d16_30d_w10",
        "S3": "daily_ladder_vrp_positive_d16_45d_w5",
        "S4": "daily_ladder_vrp_positive_d16_30d_w5",
        "S5": "risk_qwen_risk_designer_risktest_d12_30d_lowdelta_pt30_sl20_c10",
        "S6": "execution_llama_allocator_designer_daily_ladder_vrp_positive_d12_30d_w5_stop_loss_1_5",
    }


def candidate_rows(root: Path, specs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    trades = add_trade_refs(pd.read_csv(root / "test" / "trades.csv"), specs)
    daily = pd.read_csv(root / "test" / "daily_nav.csv", parse_dates=["date"])
    market = daily[["date", "spy_close", "iv", "vrp20", "iv_rank_252", "open_trades"]].drop_duplicates("date")
    out = trades.merge(market, left_on="entry_date", right_on="date", how="left", suffixes=("", "_mkt"))
    out = out.drop(columns=["date"])
    inv = {v: k for k, v in strategy_map().items()}
    out["sid"] = out["strategy"].map(inv)
    out["cid"] = out["entry_date"].dt.strftime("%Y%m%d") + "|" + out["sid"]
    out["dte"] = (out["expiry"] - out["entry_date"]).dt.days
    return out.sort_values(["entry_date", "sid"]).reset_index(drop=True)


def batch_payload(batch: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for _, r in batch.iterrows():
        rows.append(
            {
                "id": r["cid"],
                "date": r["entry_date"].date().isoformat(),
                "sid": r["sid"],
                "spy": round(float(r["entry_spy"]), 2),
                "iv": round(float(r["entry_iv"]), 4),
                "vrp20": None if pd.isna(r.get("vrp20")) else round(float(r["vrp20"]), 4),
                "iv_rank": None if pd.isna(r.get("iv_rank_252")) else round(float(r["iv_rank_252"]), 4),
                "dte": int(r["dte"]),
                "credit": round(float(r["entry_credit"]), 3),
                "max_loss": round(float(r["max_loss_per_contract"]), 2),
                "short_put": round(float(r["short_put"]), 2),
                "short_call": round(float(r["short_call"]), 2),
                "profit_target_debit": round(float(r["profit_target_debit"]), 3),
                "stop_loss_debit": round(float(r["stop_loss_debit"]), 3),
            }
        )
    return rows


def prompt_for_batch(evidence: str, rows: list[dict[str, Any]]) -> str:
    smap = strategy_map()
    return f"""You are an aggressive SPY options strategy gate. This is paper-trading research.

You must decide whether to TAKE or BLOCK each held-out test entry. You are not allowed to see future exit/P&L.
Use only the training evidence and entry-time setup.

Strategy IDs:
{json.dumps(smap, indent=2)}

{evidence}

Held-out entry candidates:
{json.dumps(rows, indent=2)}

Return strict JSON array only. One object per candidate:
[
  {{"id": "YYYYMMDD|S1", "decision": "TAKE" or "BLOCK", "confidence": 0.0-1.0, "reason": "short reason"}},
  ...
]

Rules:
- Be aggressive: if the setup resembles high-performing train trades, TAKE it.
- BLOCK only when IV/VRP/credit/risk structure looks poor relative to train evidence.
- Do not invent ids. Return exactly the candidate ids provided.
"""


def ask_one(client: OpenAI, model: str, prompt: str) -> list[dict[str, Any]]:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You output strict JSON only. No markdown."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.35,
        max_tokens=7000,
        timeout=75.0,
    )
    parsed = parse_json_loose(resp.choices[0].message.content or "")
    if isinstance(parsed, dict) and "decisions" in parsed:
        parsed = parsed["decisions"]
    if not isinstance(parsed, list):
        return []
    out = []
    for item in parsed:
        if not isinstance(item, dict) or "id" not in item:
            continue
        dec = str(item.get("decision", "BLOCK")).upper()
        out.append(
            {
                "id": str(item["id"]),
                "decision": "TAKE" if dec == "TAKE" else "BLOCK",
                "confidence": float(item.get("confidence", 0.5) or 0.5),
                "reason": str(item.get("reason", ""))[:220],
            }
        )
    return out


def run_llm_votes(root: Path, out_dir: Path, chunk_dates: int, force: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    load_env()
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY is missing from environment/council/.env.local")
    client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1", timeout=100.0)

    specs = load_specs()
    candidates = candidate_rows(root, specs)
    evidence = train_evidence(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "llm_raw_votes.jsonl"
    decisions_path = out_dir / "llm_gate_decisions.csv"

    if decisions_path.exists() and not force:
        return candidates, pd.read_csv(decisions_path)

    if force and raw_path.exists():
        raw_path.unlink()

    dates = sorted(candidates["entry_date"].drop_duplicates())
    all_votes: list[dict[str, Any]] = []
    total_chunks = (len(dates) + chunk_dates - 1) // chunk_dates
    for i in range(0, len(dates), chunk_dates):
        date_chunk = dates[i : i + chunk_dates]
        batch = candidates[candidates["entry_date"].isin(date_chunk)]
        rows = batch_payload(batch)
        prompt = prompt_for_batch(evidence, rows)
        chunk_no = i // chunk_dates + 1
        print(f"LLM chunk {chunk_no}/{total_chunks}: {date_chunk[0].date()}..{date_chunk[-1].date()} candidates={len(rows)}", flush=True)

        pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(MODELS))
        futures = {pool.submit(ask_one, client, m["model"], prompt): m for m in MODELS}
        done, not_done = concurrent.futures.wait(futures, timeout=90.0)
        for fut in done:
            meta = futures[fut]
            try:
                votes = fut.result()
            except Exception as exc:
                votes = []
                print(f"  {meta['role']} failed: {type(exc).__name__}: {str(exc)[:120]}", flush=True)
            print(f"  {meta['role']} votes={len(votes)}", flush=True)
            for v in votes:
                rec = {"role": meta["role"], "model": meta["model"], "chunk": chunk_no, **v}
                all_votes.append(rec)
                with raw_path.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec) + "\n")
        for fut in not_done:
            meta = futures[fut]
            fut.cancel()
            print(f"  {meta['role']} timed out; skipping this chunk", flush=True)
        pool.shutdown(wait=False, cancel_futures=True)

    raw_votes = pd.DataFrame(all_votes)
    ids = candidates["cid"].tolist()
    rows = []
    for cid in ids:
        w = raw_votes[raw_votes["id"] == cid] if not raw_votes.empty else pd.DataFrame()
        take_votes = int((w["decision"] == "TAKE").sum()) if not w.empty else 0
        block_votes = int((w["decision"] == "BLOCK").sum()) if not w.empty else 0
        # Aggressive tie-break: TAKE. If all models fail, TAKE so API failure does not erase the strategy.
        llm_take = take_votes >= block_votes
        conf = float(w["confidence"].mean()) if not w.empty else 0.5
        reason = " | ".join(w.head(3)["reason"].astype(str).tolist()) if not w.empty else "no votes; default TAKE"
        rows.append(
            {
                "cid": cid,
                "llm_take": llm_take,
                "take_votes": take_votes,
                "block_votes": block_votes,
                "vote_count": int(len(w)),
                "llm_confidence": conf,
                "llm_reason": reason[:700],
            }
        )
    decisions = pd.DataFrame(rows)
    decisions.to_csv(decisions_path, index=False)
    return candidates, decisions


def build_report(root: Path, out_dir: Path, candidates: pd.DataFrame, decisions: pd.DataFrame, capital: float, risk_pct: float, concurrent_pct: float) -> Path:
    train_template = pd.read_csv(root / "train" / "daily_nav.csv", parse_dates=["date"])
    test_template = pd.read_csv(root / "test" / "daily_nav.csv", parse_dates=["date"])
    specs = load_specs()
    train_trades = add_trade_refs(pd.read_csv(root / "train" / "trades.csv"), specs)
    test_trades = candidates.merge(decisions, on="cid", how="left")
    test_trades["model_take"] = test_trades["llm_take"].fillna(True).astype(bool)
    # Reuse capital-aware chart/table helpers, but store LLM confidence in the generic probability column used by hover/table code.
    test_trades["prob_range_lgbm"] = test_trades["llm_confidence"].fillna(0.5)

    strategies = pd.read_csv(root / "test" / "summary.csv")["strategy"].tolist()
    train_nav, train_exec, train_summary = replay_all(
        train_template,
        train_trades,
        strategies,
        capital=capital,
        risk_pct_per_trade=risk_pct,
        max_concurrent_risk_pct=concurrent_pct,
        use_model_gate=False,
    )
    test_raw_nav, test_raw_exec, test_raw_summary = replay_all(
        test_template,
        test_trades,
        strategies,
        capital=capital,
        risk_pct_per_trade=risk_pct,
        max_concurrent_risk_pct=concurrent_pct,
        use_model_gate=False,
    )
    test_llm_nav, test_llm_exec, test_llm_summary = replay_all(
        test_template,
        test_trades,
        strategies,
        capital=capital,
        risk_pct_per_trade=risk_pct,
        max_concurrent_risk_pct=concurrent_pct,
        use_model_gate=True,
    )

    stem = f"llm_gate_capital_aware_{int(capital)}_risk{str(risk_pct).replace('.', '_')}_cap{str(concurrent_pct).replace('.', '_')}"
    train_summary.to_csv(out_dir / f"{stem}_train_summary.csv", index=False)
    test_raw_summary.to_csv(out_dir / f"{stem}_test_raw_summary.csv", index=False)
    test_llm_summary.to_csv(out_dir / f"{stem}_test_llm_summary.csv", index=False)
    test_trades.to_csv(out_dir / f"{stem}_test_trades_with_llm_votes.csv", index=False)
    test_llm_exec.to_csv(out_dir / f"{stem}_test_llm_executed_trades.csv", index=False)

    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #17202a; }
      h1 { margin-bottom: 4px; }
      h2 { margin-top: 38px; padding-top: 20px; border-top: 1px solid #d7dbdd; }
      h3 { margin-top: 24px; }
      .note { color: #566573; max-width: 1200px; line-height: 1.45; }
      .navlinks a { margin-right: 14px; font-size: 13px; }
      .scroll { overflow-x: auto; border: 1px solid #eee; padding: 8px; margin-bottom: 20px; }
      table.data-table { border-collapse: collapse; width: 100%; font-size: 12px; }
      table.data-table th, table.data-table td { border: 1px solid #d6dbdf; padding: 6px 8px; text-align: right; white-space: nowrap; }
      table.data-table th { background: #f2f3f4; position: sticky; top: 0; z-index: 1; }
      table.data-table td:first-child, table.data-table th:first-child,
      table.data-table td:nth-child(4), table.data-table th:nth-child(4),
      table.data-table td:nth-child(5), table.data-table th:nth-child(5),
      table.data-table td:nth-child(6), table.data-table th:nth-child(6),
      table.data-table td:nth-child(7), table.data-table th:nth-child(7) { text-align: left; }
      .small { font-size: 12px; color: #566573; }
    </style>
    """
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>NVIDIA LLM Gate Aggressive Six</title>",
        css,
        "</head><body>",
        f"<h1>NVIDIA LLM-Only Gate: Aggressive Six (${capital:,.0f})</h1>",
        f"<p class='note'>No LightGBM trained model is used here. NVIDIA NIM models vote TAKE/BLOCK from train-window evidence and entry-time test setup only. "
        f"Council models: {', '.join(m['role'] for m in MODELS)}. Aggressive tie-break defaults to TAKE. "
        f"Capital-aware replay: {risk_pct:.1f}% max-loss risk per new trade, {concurrent_pct:.1f}% max concurrent open max-loss exposure.</p>",
        "<div class='navlinks'>",
    ]
    for s in strategies:
        html.append(f"<a href='#{s}'>{s}</a>")
    html.extend(
        [
            "</div>",
            "<h2>Overview</h2>",
            overview_chart(train_nav, test_raw_nav, test_llm_nav, train_summary, test_raw_summary, test_llm_summary, capital),
            "<h3>Train Capital-Aware Raw Summary (70%) + Benchmark</h3><div class='scroll'>",
            summary_table(train_summary, bench_summary(train_template, capital)),
            "</div>",
            "<h3>Held-Out Test Raw Summary (30%) + Benchmark</h3><div class='scroll'>",
            summary_table(test_raw_summary, bench_summary(test_template, capital)),
            "</div>",
            "<h3>Held-Out Test With NVIDIA LLM Gate</h3><div class='scroll'>",
            summary_table(test_llm_summary, bench_summary(test_template, capital)),
            "</div>",
        ]
    )
    for s in strategies:
        html.extend(
            [
                f"<h2 id='{s}'>{s}</h2>",
                strategy_chart(s, test_raw_nav, test_llm_nav, test_raw_exec, test_llm_exec, capital),
                "<p class='small'>Table below shows LLM-gated capital-aware executed/skipped records. The generic confidence column is the LLM vote confidence, not LightGBM probability.</p>",
                "<div class='scroll'>",
                trade_table(s, test_llm_exec),
                "</div>",
            ]
        )
    html.extend(["</body></html>"])
    out_path = out_dir / f"{stem}.html"
    out_path.write_text("\n".join(html), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--risk-pct", type=float, default=10.0)
    parser.add_argument("--concurrent-pct", type=float, default=100.0)
    parser.add_argument("--chunk-dates", type=int, default=10)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS_DIR / f"aggressive_six_llm_gate_{stamp}"
    candidates, decisions = run_llm_votes(args.root, out_dir, args.chunk_dates, args.force)
    report = build_report(args.root, out_dir, candidates, decisions, args.capital, args.risk_pct, args.concurrent_pct)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
