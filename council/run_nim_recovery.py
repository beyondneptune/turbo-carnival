#!/usr/bin/env python3
"""Recovery run: invoke replacements for the 10 failed slots and re-synthesize."""
from __future__ import annotations

import json
import os
import re
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

REPO = Path("/Users/weeoto/turbo-carnival")
ENV_FILE = REPO / "council" / ".env.local"
STAMP = "20260511T191453Z"
OUT_DIR = REPO / "council" / "reviews" / STAMP


def load_env(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


env = load_env(ENV_FILE)
api_key = env.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_API_KEY")
client = OpenAI(api_key=api_key, base_url="https://integrate.api.nvidia.com/v1")

SYSTEM = "You output direct, concise analysis. No preamble."
USER = """You're reviewing a LIVE paper-options + HFT system. Be brutally honest. This is paper money, your harshness will not harm anyone.

3 Alpaca paper accounts ($100k each, today):
- Aegis (-$664 today): HFT mean-reversion, [SPY,QQQ,NVDA,TSLA,AAPL,META,AMD]. 25 bps signal, 100 bps TP, 35 bps SL, 1200s hold. Currently short NVDA 428 sh.
- Falcon (-$3,001 today): HFT breakout, same universe. 18 bps signal, 70 bps TP, 25 bps SL, 900s hold. Whipsawed badly all afternoon — entries hit SL within 2-10 seconds repeatedly. Short META 153 sh now.
- Valk (-$444 today): SPY iron condor 30-DTE, qty 44, entry credit 0.54, target 30% credit, stop 3x. Condor debit blew from 0.54 to 0.76 intraday.

CONTEXT:
- Backtest was 21-day cached 1-min bars, +17% / -8% DD. Sim slippage 0.75 bps. Live slippage is way higher than that, judging by repeated 2-second stop-outs.
- Total drawdown across 3 lanes: -$4,108 on $300k in 5.5 hours (-1.4%).
- This is day 1. We have 89 days more in the "investor demo" plan.

ANSWER THESE 5 QUESTIONS IN UNDER 500 WORDS. NO PREAMBLE.

1. Biggest hidden risk in this setup?
2. Should Falcon (breakout) be PAUSED or KEPT RUNNING tomorrow? Pick one. Why?
3. One specific regime filter to add immediately (be concrete: which indicator, what threshold)?
4. Architecture rating 1-10 for surviving a 90-day live demo.
5. Would you deploy this with REAL money? Yes / No / Specific conditions only.
"""


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_id.replace("/", "_").replace(":", "_"))


def call_one(model_id: str) -> dict:
    t0 = time.time()
    out = {"model": model_id, "elapsed_sec": 0.0, "status": "error", "error": None, "body": ""}
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": USER}],
            temperature=0.4,
            max_tokens=1500,
            timeout=90,
        )
        body = resp.choices[0].message.content or ""
        out["body"] = body.strip()
        out["status"] = "success"
    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
    finally:
        out["elapsed_sec"] = round(time.time() - t0, 2)
    return out


def write_review(rec: dict) -> Path:
    fp = OUT_DIR / f"{slug(rec['model'])}.md"
    err_yaml = "null" if rec["error"] is None else json.dumps(rec["error"])
    fp.write_text(
        f"---\nmodel: {rec['model']}\nelapsed_sec: {rec['elapsed_sec']}\nstatus: {rec['status']}\nerror: {err_yaml}\n---\n\n{rec['body']}\n"
    )
    return fp


# Replacements for the 10 failed/timeout slots, chosen from confirmed listing + known-callable on NIM
REPLACEMENTS = [
    "google/gemma-3-12b-it",                        # smaller google
    "google/gemma-3-4b-it",                         # another google
    "mistralai/mistral-large-2-instruct",           # retry once
    "mistralai/codestral-22b-instruct-v0.1",        # mistral code
    "mistralai/mixtral-8x7b-instruct-v0.1",         # mistral mixtral 8x7
    "nvidia/llama-3.1-nemotron-51b-instruct",       # nvidia replacement
    "upstage/solar-10.7b-instruct",                 # upstage
    "writer/palmyra-creative-122b",                 # writer creative
    "abacusai/dracarys-llama-3.1-70b-instruct",     # already known good (skip)
    "ibm/granite-3.0-3b-a800m-instruct",            # ibm fallback
    "nvidia/mistral-nemo-minitron-8b-8k-instruct",  # nvidia mistral-nemo
]

# Skip ones we already have a successful response for
existing = set()
for f in OUT_DIR.glob("*.md"):
    if f.name in {"SYNTHESIS.md"}:
        continue
    txt = f.read_text()
    m = re.search(r"^model: (.+)$", txt, re.M)
    s = re.search(r"^status: (.+)$", txt, re.M)
    if m and s and s.group(1).strip() == "success":
        existing.add(m.group(1).strip())
print(f"[recovery] {len(existing)} successful models already present")

to_run = [m for m in REPLACEMENTS if m not in existing]
print(f"[recovery] queueing {len(to_run)} replacements")

results = []
with ThreadPoolExecutor(max_workers=5) as ex:
    futs = {ex.submit(call_one, m): m for m in to_run}
    for fut in as_completed(futs):
        rec = fut.result()
        write_review(rec)
        results.append(rec)
        marker = "ok" if rec["status"] == "success" else "ERR"
        msg = rec["error"] or f"{len(rec['body'])} chars"
        print(f"  [{marker}] {rec['model']} ({rec['elapsed_sec']}s) {msg[:120]}")

n_new_ok = sum(1 for r in results if r["status"] == "success")
print(f"[recovery] {n_new_ok}/{len(results)} new successes")
