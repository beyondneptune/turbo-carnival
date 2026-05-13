"""Use NVIDIA NIM (free via API key) to design strategy specs in parallel.

Calls 10 different NIM models concurrently. Each returns one JSON spec.
Validates and writes to data/strategy_specs/nim_proposed/.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

REPO = Path("/Users/weeoto/turbo-carnival")
ENV = REPO / "council" / ".env.local"
OUT = REPO / "data" / "strategy_specs" / "nim_proposed"
OUT.mkdir(parents=True, exist_ok=True)


# NIM models known to work from earlier review runs
MODELS = [
    "moonshotai/kimi-k2-instruct",
    "qwen/qwen3-coder-480b-a35b-instruct",
    "meta/llama-4-maverick-17b-128e-instruct",
    "meta/llama-3.3-70b-instruct",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "google/gemma-3-27b-it",
    "nvidia/llama-3.3-nemotron-super-49b-v1.5",
    "deepseek-ai/deepseek-v4-pro",
    "microsoft/phi-4-mini-instruct",
    "abacusai/dracarys-llama-3.1-70b-instruct",
]

DESIGN_BRIEFS = [
    ("conservative_meanrev", "Design an ultra-conservative mean-reversion variant. Stack 3+ filters. Use min_signal_bps >= 35. Prioritize survival."),
    ("aggressive_momentum",  "Design an aggressive momentum variant. min_signal_bps 20-30. Use only 1 filter (news_blackout). Aim for high trade count."),
    ("news_fade_extreme",    "Design a news-fade extreme variant using signal_mode='news_fade'. Tight stops, wide TPs. Trade DURING news (don't blackout)."),
    ("chip_specialist",      "Design a chip-only momentum variant. universe=['NVDA','AMD']. Match the chip supercycle theme. min_signal_bps 25-35."),
    ("spy_only_meanrev",     "Design a SPY-only mean-reversion variant. universe=['SPY']. Tight params since SPY moves are small. Use 2 filters."),
    ("breakout_with_atr",    "Design a breakout variant with ATR confirmation. min_signal_bps 40+. Wide stops. Use atr_gate + abs_return_gate."),
    ("morning_only",         "Design a morning-only variant using time_gate 09:30-11:00. Mean-reversion. High frequency in the open window."),
    ("turbo_hybrid_v2",      "Design a turbo hybrid (signal_mode='turbo'). Stack vol_of_vol_gate + news_blackout. Try to extract edge across regimes."),
    ("multi_filter_balanced","Design a balanced mean-reversion with EXACTLY 2 filters that minimize false signals: vol_of_vol_gate AND time_gate."),
    ("tsla_volatility",      "Design a TSLA-only volatility-harvest variant. universe=['TSLA']. signal_mode='mean_reversion'. Big bps numbers (TSLA moves 5x SPY)."),
]


SYSTEM = "You output strict JSON only. No markdown, no fences, no preamble. Just one valid JSON object that matches the schema."

PROMPT_TEMPLATE = """Design a paper-trading HFT strategy spec for the turbo-carnival simulator.

CONTEXT (Day 1 of paper trading, today 2026-05-11):
- Aegis HFT mean-rev lost -$709 / -0.7% on $100k. 25/100/35 bps signal/TP/SL, 1200s hold. 52% TP hit rate.
- Falcon HFT breakout lost -$2,290 / -2.3%. 18/70/25 bps, 900s hold. Stop fired in 2-10s 82 times of 99 entries. 7% TP rate.
- 60 LLM models reviewed. 60/60 said pause Falcon. Top regime filters: VIX(38), ATR(35), news-gate(28), ADX(25), vol-of-vol(1).
- Live slippage measured 3-15 bps; baseline assumed 0.75 bps. We now use 8 bps default.
- 5 news catalysts clustered 13:30-15:00 ET caused the chop that broke Falcon.

SIMULATOR SCHEMA (these are the ONLY supported fields/values):
- signal_mode: one of "mean_reversion" | "momentum" | "breakout" | "turbo" | "news_fade" | "neutral_carry"
- min_signal_bps: number (typically 15-60)
- take_profit_bps: number (typically 50-200)
- stop_loss_bps: number (typically 20-80)
- max_hold_sec: integer (typically 300-1800)
- notional_pct: number 0.0-1.0 (typically 0.85-0.95)
- slippage_bps: number (use 8.0 default, 10.0 for fade strategies, 15.0 for stress test)
- universe (optional, default all 7): list from ["SPY","QQQ","NVDA","TSLA","AAPL","META","AMD"]
- filters (optional list, can be empty):
    - {{"type": "atr_gate", "window": <int>, "threshold_bps": <float as ratio e.g. 0.0030 = 30 bps>}}
    - {{"type": "vol_of_vol_gate", "ratio": <float, e.g. 1.4>}}
    - {{"type": "news_blackout", "cooldown_sec": <int, e.g. 300>}}
    - {{"type": "abs_return_gate", "threshold_bps": <float as ratio>}}
    - {{"type": "time_gate", "start_et": "<HH:MM>", "end_et": "<HH:MM>"}}
    - {{"type": "correlation_check", "threshold_bps": <float as ratio>, "min_count": <int>}}

YOUR DESIGN BRIEF: {brief_label}
{brief_desc}

OUTPUT STRICT JSON (no fences, no markdown, just the object):
{{
  "name": "nim_{brief_label}",
  "description": "<one-line explanation>",
  "signal_mode": "<value>",
  "min_signal_bps": <number>,
  "take_profit_bps": <number>,
  "stop_loss_bps": <number>,
  "max_hold_sec": <integer>,
  "notional_pct": <number>,
  "slippage_bps": <number>,
  "filters": [<filter objects>]
}}

If you need universe, add a "universe" field. Output the JSON now."""


def load_env() -> None:
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() and k.strip() not in os.environ:
            os.environ[k.strip()] = v.strip().strip('"').strip("'")


def parse_json_loose(text: str) -> dict | None:
    text = text.strip()
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


def validate(spec: dict) -> tuple[bool, str]:
    required = ["name", "signal_mode", "min_signal_bps", "take_profit_bps",
                 "stop_loss_bps", "max_hold_sec"]
    for k in required:
        if k not in spec:
            return False, f"missing {k}"
    if spec["signal_mode"] not in ("mean_reversion", "momentum", "breakout", "turbo", "news_fade", "neutral_carry"):
        return False, f"bad signal_mode {spec['signal_mode']}"
    return True, "ok"


def design_one(client: OpenAI, model: str, brief_label: str, brief_desc: str, i: int) -> dict:
    prompt = PROMPT_TEMPLATE.format(brief_label=brief_label, brief_desc=brief_desc)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=800,
            timeout=90.0,
        )
        text = resp.choices[0].message.content or ""
        spec = parse_json_loose(text)
        if not spec:
            return {"ok": False, "model": model, "brief": brief_label, "error": "no JSON parsed", "raw": text[:200]}
        # Force the name to include the seq
        spec["name"] = f"nim_{i:02d}_{brief_label}"
        ok, msg = validate(spec)
        if not ok:
            return {"ok": False, "model": model, "brief": brief_label, "error": msg, "raw": json.dumps(spec)[:200]}
        # Write
        path = OUT / f"{spec['name']}.json"
        path.write_text(json.dumps(spec, indent=2))
        return {"ok": True, "model": model, "brief": brief_label, "path": str(path)}
    except Exception as exc:
        return {"ok": False, "model": model, "brief": brief_label, "error": f"{type(exc).__name__}: {exc}"}


def main() -> int:
    load_env()
    key = os.environ["NVIDIA_API_KEY"]
    client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1", timeout=100.0)

    tasks = [(MODELS[i], DESIGN_BRIEFS[i][0], DESIGN_BRIEFS[i][1], i + 1)
             for i in range(len(DESIGN_BRIEFS))]

    print(f"Launching {len(tasks)} NIM designers...")
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(design_one, client, m, b, d, i): (m, b, i) for m, b, d, i in tasks}
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            results.append(r)
            tag = "OK " if r["ok"] else "ERR"
            extra = r.get("path", "") if r["ok"] else r.get("error", "?")
            print(f"  [{tag}] {r['model']:50s}  {r['brief']:25s}  {extra}")

    ok_count = sum(1 for r in results if r["ok"])
    print(f"\n{ok_count}/{len(results)} succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
