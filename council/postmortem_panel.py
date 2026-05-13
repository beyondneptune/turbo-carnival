"""NIM panel: post-mortem analysis of biggest losing trade dates.

For each catastrophic-loss date in the council winner's history:
  - SPY % move during the trade
  - IV spike
  - Trade reason (stop_loss / dte_close)

Send to ~6 NIM models with different lenses:
  - macro_analyst: what macro event caused this?
  - vol_analyst: was there an early-warning signal in vol structure?
  - news_analyst: what was in the news in the 5 days before?
  - feature_engineer: what feature could have predicted/avoided this?
  - signal_designer: what gate/filter would have skipped this entry?

Aggregate suggested features across all models + dates → rank by frequency.
The output is a prioritized list of features the strategy is missing.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]


LENSES = [
    {
        "role": "macro_analyst",
        "model": "qwen/qwen3-coder-480b-a35b-instruct",
        "mission": "Identify the macroeconomic catalyst (Fed action, geopolitical event, earnings, COVID-style shock, etc.) that caused this SPY move.",
    },
    {
        "role": "vol_analyst",
        "model": "mistralai/mistral-nemotron",
        "mission": "Examine whether IV term structure, VIX spike pattern, or skew was abnormal BEFORE the move began. What early-warning signal in options vol was visible 5-10 days before the crash?",
    },
    {
        "role": "news_analyst",
        "model": "meta/llama-4-maverick-17b-128e-instruct",
        "mission": "What major news/headlines hit markets in the 5 days BEFORE this date? List specific known events from your training data. Be honest about what you don't know.",
    },
    {
        "role": "feature_engineer",
        "model": "moonshotai/kimi-k2-instruct",
        "mission": "Propose ONE concrete feature/signal that could have been computed BEFORE entry that would have flagged this trade as risky. The feature must be computable from public market data only (no insider info).",
    },
    {
        "role": "signal_designer",
        "model": "qwen/qwen3-coder-480b-a35b-instruct",
        "mission": "Design a gate/filter rule that would have BLOCKED this trade entry. Format: 'don't enter SPY iron condor if X > Y'. Be specific with thresholds.",
    },
]


PROMPT = """You are the {role}. Your mission: {mission}

Trade post-mortem:
- Entry date: {entry_date}
- Exit date: {exit_date}
- SPY at entry: ${entry_spy:.2f}
- SPY at exit: ${exit_spy:.2f}
- SPY % move during trade: {spy_move_pct:+.2f}%
- IV at entry: {entry_iv:.4f}
- IV at exit: {exit_iv:.4f}
- IV change: {iv_change:+.4f}
- Trade reason: {reason}
- Realized P&L: ${pnl_usd:.2f}
- Strategy: 30-DTE SPY iron condor, delta=0.12, opened daily, vrp_positive gate

Other strategies took losses on this date too — this was a clustered loss event across many parameter sets.

Return JSON only:
{{
  "role": "{role}",
  "verdict_one_sentence": "what happened in 1 sentence",
  "specific_cause": "concrete cause (Fed meeting, COVID, war, earnings, etc) — or 'unknown if outside my training data'",
  "early_warning_signals": [
    {{
      "signal_name": "short name",
      "what_to_compute": "specific math, e.g. 'VIX 5-day rate of change'",
      "threshold_or_pattern": "specific trigger condition",
      "would_have_avoided_this_trade": true|false
    }}
  ],
  "blocking_rule": "single rule: 'do not enter if X', concrete and computable",
  "confidence_pct": 0
}}"""


def load_env():
    p = REPO_ROOT / "council" / ".env.local"
    if not p.exists(): return
    for ln in p.read_text().splitlines():
        if "=" in ln and not ln.strip().startswith("#"):
            k, v = ln.split("=", 1)
            if k.strip() and k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def call(client, lens, ctx):
    prompt = PROMPT.format(role=lens["role"], mission=lens["mission"], **ctx)
    try:
        r = client.chat.completions.create(
            model=lens["model"],
            messages=[
                {"role": "system", "content": "You output strict JSON market post-mortems. No markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=900,
        )
        return {"ok": True, "role": lens["role"], "model": lens["model"],
                "entry_date": ctx["entry_date"], "content": r.choices[0].message.content or ""}
    except Exception as exc:
        return {"ok": False, "role": lens["role"], "model": lens["model"],
                "entry_date": ctx["entry_date"], "error": f"{type(exc).__name__}: {exc}"}


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
        print("NVIDIA_API_KEY missing"); return 1
    client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1", timeout=120.0)

    df = pd.read_csv(REPO_ROOT / "experiments/runs/council_spy_grid_20260510T153306Z/trades.csv")
    df["spy_move_pct"] = (df["exit_spy"] / df["entry_spy"] - 1) * 100
    df["iv_change"] = df["exit_iv"] - df["entry_iv"]

    # Cluster losses by entry_date and pick worst dates (loss clusters across strategies)
    loss_dates = (df[df["pnl_usd"] < -50]
                  .groupby("entry_date")
                  .agg(losses=("pnl_usd", "size"),
                       total_loss=("pnl_usd", "sum"),
                       spy_move=("spy_move_pct", "mean"),
                       iv_change=("iv_change", "mean"))
                  .sort_values("total_loss")
                  .head(12))
    target_dates = list(loss_dates.index)
    print(f"=== Analyzing {len(target_dates)} worst loss dates ===")
    for d in target_dates:
        r = loss_dates.loc[d]
        print(f"  {d}: {r['losses']:.0f} losses, total -${abs(r['total_loss']):,.0f}, "
              f"SPY {r['spy_move']:+.1f}%, IV {r['iv_change']:+.3f}")

    # Build per-date contexts using a representative trade per date
    contexts = []
    for d in target_dates:
        sample = df[(df["entry_date"] == d) & (df["pnl_usd"] < -50)].iloc[0]
        contexts.append({
            "entry_date": d,
            "exit_date": sample["exit_date"],
            "entry_spy": float(sample["entry_spy"]),
            "exit_spy": float(sample["exit_spy"]),
            "spy_move_pct": float(sample["spy_move_pct"]),
            "entry_iv": float(sample["entry_iv"]),
            "exit_iv": float(sample["exit_iv"]),
            "iv_change": float(sample["iv_change"]),
            "reason": str(sample["reason"]),
            "pnl_usd": float(sample["pnl_usd"]),
        })

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "council" / "postmortem_runs" / stamp
    raw = out_dir / "raw_responses"
    raw.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput: {out_dir}")

    # Fan out: 12 dates × 5 lenses = 60 NIM calls in parallel
    work = [(lens, ctx) for ctx in contexts for lens in LENSES]
    print(f"\nFiring {len(work)} NIM calls in parallel (8 workers)...\n")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(call, client, lens, ctx): (lens, ctx) for lens, ctx in work}
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            results.append(r)
            stem = f"{r['entry_date']}__{r['role']}"
            (raw / f"{stem}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
            print(f"  ok={r.get('ok')}  {stem}  via {r['model']}", flush=True)

    # Aggregate causes + features
    print("\n=== Aggregating ===\n")
    causes = []
    features_proposed = []
    blocking_rules = []
    per_date_summary = {}
    for r in results:
        if not r.get("ok"): continue
        p = parse(str(r.get("content", "")))
        if not p: continue
        d = r["entry_date"]
        per_date_summary.setdefault(d, []).append({
            "role": p.get("role"),
            "verdict": p.get("verdict_one_sentence"),
            "cause": p.get("specific_cause"),
            "blocking_rule": p.get("blocking_rule"),
        })
        if p.get("specific_cause"):
            causes.append((d, str(p["specific_cause"])[:200]))
        for s in (p.get("early_warning_signals") or []):
            if isinstance(s, dict):
                features_proposed.append(str(s.get("signal_name", "")))
        if p.get("blocking_rule"):
            blocking_rules.append((d, str(p["blocking_rule"])[:300]))

    # Rank features by frequency
    feature_counts = Counter(f for f in features_proposed if f.strip())
    print("=== Top 15 features proposed (by frequency across panel) ===")
    for feat, n in feature_counts.most_common(15):
        print(f"  {n:3}x  {feat[:100]}")

    print("\n=== Per-date causes (one verdict per date, picked from macro_analyst) ===")
    for d in sorted(per_date_summary.keys()):
        macro = next((s for s in per_date_summary[d] if s["role"] == "macro_analyst"), per_date_summary[d][0])
        print(f"\n  {d}")
        print(f"    cause: {(macro.get('cause') or '?')[:200]}")
        print(f"    block: {(macro.get('blocking_rule') or '?')[:180]}")

    # Save synthesis
    out_dir.joinpath("postmortem_summary.json").write_text(
        json.dumps({
            "n_dates": len(target_dates),
            "n_responses": len(results),
            "ok_responses": sum(1 for r in results if r.get("ok")),
            "feature_counts": feature_counts.most_common(30),
            "per_date": per_date_summary,
            "blocking_rules": blocking_rules,
        }, indent=2, default=str), encoding="utf-8")

    # Markdown report
    md = ["# Post-Mortem of Council Winner's Worst Loss Dates", "",
          f"Analyzed {len(target_dates)} worst clustered-loss dates across 61 council strategies.",
          f"Ran {len(work)} NIM model calls (5 lenses × {len(target_dates)} dates).",
          "", "## Top features the strategy is missing", "",
          "Ranked by how often models proposed them as early-warning signals:", ""]
    for feat, n in feature_counts.most_common(20):
        md.append(f"- **{n}× mentions**: {feat}")
    md.append("\n## Per-date analysis\n")
    for d in sorted(per_date_summary.keys()):
        ctx = next(c for c in contexts if c["entry_date"] == d)
        md.append(f"### {d}")
        md.append(f"- Trade entered with SPY=${ctx['entry_spy']:.2f}, exited at "
                  f"${ctx['exit_spy']:.2f} ({ctx['spy_move_pct']:+.2f}%), IV "
                  f"{ctx['iv_change']:+.3f}")
        for s in per_date_summary[d]:
            md.append(f"  - **{s['role']}**: {s.get('verdict','?')[:200]}")
            if s.get("blocking_rule"):
                md.append(f"    - Blocking rule: `{s['blocking_rule'][:200]}`")
        md.append("")
    out_dir.joinpath("postmortem_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nReport: {out_dir / 'postmortem_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
