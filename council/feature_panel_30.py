"""30-model NIM feature engineering panel.

Each model proposes 5 features for an iron-condor strategy with:
  - Entry-timing features (when to take positions)
  - Exit-timing features (when to close early)
  - Position-sizing features (how much to put on)

Goal: aggregate by frequency to find consensus features beyond what the
post-mortem panel proposed (which was VIX-ROC heavy).
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

from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]


# 30 diverse models across providers/sizes/specialties
PANEL = [
    # Mistral family (5)
    {"role": "mistral_large", "model": "mistralai/mistral-large-3-675b-instruct-2512"},
    {"role": "mistral_medium", "model": "mistralai/mistral-medium-3-instruct"},
    {"role": "mistral_nemotron", "model": "mistralai/mistral-nemotron"},
    {"role": "ministral_14b", "model": "mistralai/ministral-14b-instruct-2512"},
    {"role": "magistral", "model": "mistralai/magistral-small-2506"},
    # Qwen family (3)
    {"role": "qwen_coder_480b", "model": "qwen/qwen3-coder-480b-a35b-instruct"},
    {"role": "qwen_thinking", "model": "qwen/qwen3-next-80b-a3b-thinking"},
    {"role": "qwen_instruct", "model": "qwen/qwen3-next-80b-a3b-instruct"},
    # Llama family (3)
    {"role": "llama_maverick", "model": "meta/llama-4-maverick-17b-128e-instruct"},
    {"role": "llama_70b", "model": "meta/llama-3.3-70b-instruct"},
    {"role": "llama_3_1_70b", "model": "meta/llama-3.1-70b-instruct"},
    # Kimi (Moonshot) (2)
    {"role": "kimi_instruct", "model": "moonshotai/kimi-k2-instruct"},
    {"role": "kimi_thinking", "model": "moonshotai/kimi-k2-thinking"},
    # Google Gemma (3)
    {"role": "gemma_27b", "model": "google/gemma-3-27b-it"},
    {"role": "gemma_12b", "model": "google/gemma-3-12b-it"},
    {"role": "gemma_3n_e4b", "model": "google/gemma-3n-e4b-it"},
    # Microsoft Phi (1)
    {"role": "phi_4_multimodal", "model": "microsoft/phi-4-multimodal-instruct"},
    # NVIDIA Nemotron (1)
    {"role": "nemotron_nano", "model": "nvidia/nvidia-nemotron-nano-9b-v2"},
    # ByteDance (2)
    {"role": "seed_oss", "model": "bytedance/seed-oss-36b-instruct"},
    {"role": "step_flash", "model": "stepfun-ai/step-3.5-flash"},
    # Minimax (2)
    {"role": "minimax_m25", "model": "minimaxai/minimax-m2.5"},
    {"role": "minimax_m27", "model": "minimaxai/minimax-m2.7"},
    # IBM Granite (1)
    {"role": "granite_8b", "model": "ibm/granite-3.0-8b-instruct"},
    # Databricks (1)
    {"role": "dbrx", "model": "databricks/dbrx-instruct"},
    # AI21 Jamba (1)
    {"role": "jamba", "model": "ai21labs/jamba-1.5-large-instruct"},
    # Sarvam (1)
    {"role": "sarvam", "model": "sarvamai/sarvam-m"},
    # Stockmark (1)
    {"role": "stockmark", "model": "stockmark/stockmark-2-100b-instruct"},
    # Upstage (1)
    {"role": "solar", "model": "upstage/solar-10.7b-instruct"},
    # Abacus (1)
    {"role": "dracarys", "model": "abacusai/dracarys-llama-3.1-70b-instruct"},
    # Stability (1) - excluded image-only
]


PROMPT = """You are a quantitative finance researcher.

Task: propose 5 features each for a SPY iron-condor + earnings-straddle strategy.

Strategy context:
- $100k account on Interactive Brokers paper trading
- 30-DTE SPY iron condors at 0.12 delta with $5 wings, profit_take 30% of credit, stop_loss 2x credit
- Backtest 2019-2026 shows 0.30% monthly with -4% drawdown realistic
- Failures cluster on COVID 2020, Fed-shock 2022 (Sept/April), Oct 2023 rally, Feb 2025
- Existing features: VIX, IV (saved ATM), realized vol 10/20-day, volume_ratio, IV rank, vrp20

Already-tried features that did NOT help:
- Skip if VIX_5d_ROC > 20% (cut returns more than losses)
- Skip if VIX > 28 (skipped recovery trades too)

Propose features that go BEYOND vol-skip rules. Think about:
- Microstructure (volume, order flow, options OI imbalance)
- Cross-asset (treasury yields, dollar, credit spreads)
- Calendar (Fed meeting weeks, OPEX, quarter-end, earnings season)
- Market internals (advance/decline, breadth)
- Sentiment (put/call ratio, AAII)
- Term structure (yield curve, VIX curve, IV term structure)
- Options-specific (skew, butterfly, gamma exposure)

Return strict JSON only:
{{
  "role": "<your_role_string>",
  "entry_features": [
    {{"name": "...", "computation": "concrete formula", "use": "skip | size_up | size_down"}}
  ],
  "exit_features": [
    {{"name": "...", "computation": "concrete formula", "trigger": "close_now | wait"}}
  ],
  "sizing_features": [
    {{"name": "...", "computation": "concrete formula", "effect": "increase_qty | decrease_qty"}}
  ],
  "single_bold_idea": "one specific thing the panel hasn't likely thought of"
}}

Be concrete. No vague suggestions like 'use more data'. Each feature must have a computable formula."""


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
        prompt = PROMPT.replace("<your_role_string>", m["role"])
        r = client.chat.completions.create(
            model=m["model"],
            messages=[
                {"role": "system", "content": "You output strict JSON. No markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=1500,
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


def normalize_feature_name(name: str) -> str:
    """Normalize for frequency counting — lowercase, strip whitespace, common synonyms."""
    n = re.sub(r"[^a-z0-9 _]", "", str(name).lower().strip())
    # Collapse common synonyms
    repl = {
        "put-call": "put_call", "put/call": "put_call",
        "yield curve": "yield_curve",
        "vix term structure": "vix_term_structure",
        "iv term structure": "iv_term_structure",
        "advance decline": "advance_decline",
    }
    for k, v in repl.items():
        n = n.replace(k, v)
    n = re.sub(r"\s+", "_", n)
    return n[:60]


def main():
    load_env()
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        print("NVIDIA_API_KEY missing"); return 1
    client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1", timeout=120.0)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO_ROOT / "council" / "feature_runs" / stamp
    raw = out / "raw_responses"
    raw.mkdir(parents=True, exist_ok=True)
    print(f"Running {len(PANEL)}-model panel. Output: {out}\n")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(call, client, m): m for m in PANEL}
        for fut in concurrent.futures.as_completed(futures):
            r = fut.result()
            results.append(r)
            (raw / f"{r['role']}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
            print(f"  {r['role']:25} ok={r.get('ok')} via {r['model']}", flush=True)

    # Aggregate
    entry_counter = Counter()
    exit_counter = Counter()
    sizing_counter = Counter()
    bold_ideas = []
    feature_details: dict[str, list[str]] = {}

    for r in results:
        if not r.get("ok"): continue
        p = parse(str(r.get("content", "")))
        if not p: continue
        for f in p.get("entry_features", []) or []:
            if isinstance(f, dict):
                key_n = normalize_feature_name(f.get("name", ""))
                if key_n: entry_counter[key_n] += 1
                feature_details.setdefault(key_n, []).append(
                    f"[{r['role']} entry] {f.get('name')}: {f.get('computation', '')[:120]}"
                )
        for f in p.get("exit_features", []) or []:
            if isinstance(f, dict):
                key_n = normalize_feature_name(f.get("name", ""))
                if key_n: exit_counter[key_n] += 1
        for f in p.get("sizing_features", []) or []:
            if isinstance(f, dict):
                key_n = normalize_feature_name(f.get("name", ""))
                if key_n: sizing_counter[key_n] += 1
        if p.get("single_bold_idea"):
            bold_ideas.append((r["role"], str(p["single_bold_idea"])[:300]))

    print("\n=== Top 20 ENTRY features (frequency across panel) ===")
    for f, n in entry_counter.most_common(20):
        print(f"  {n:3}x  {f}")
    print("\n=== Top 15 EXIT features ===")
    for f, n in exit_counter.most_common(15):
        print(f"  {n:3}x  {f}")
    print("\n=== Top 15 SIZING features ===")
    for f, n in sizing_counter.most_common(15):
        print(f"  {n:3}x  {f}")
    print("\n=== Bold ideas (one per model) ===")
    for role, idea in bold_ideas[:15]:
        print(f"  [{role}] {idea[:200]}")

    summary = {
        "n_models": len(PANEL),
        "ok_responses": sum(1 for r in results if r.get("ok")),
        "entry_features": entry_counter.most_common(50),
        "exit_features": exit_counter.most_common(50),
        "sizing_features": sizing_counter.most_common(50),
        "bold_ideas": bold_ideas,
        "feature_details": feature_details,
    }
    out.joinpath("feature_panel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSummary: {out / 'feature_panel_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
