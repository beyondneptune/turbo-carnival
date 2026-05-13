"""5-model NIM panel: post-mortem on today's ADR losses.

Reads today's actual ADR fills from Alpaca + backtest context, asks 5
different models in parallel to evaluate whether today's loss represents
(a) strategy failure, (b) bad luck within expected variance, or
(c) execution problem. Each model gets the same prompt; we synthesize.

Run:
    .venv/bin/python council/today_postmortem.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / "council" / ".env.local"


def load_env() -> dict:
    env = {}
    if ENV_FILE.exists():
        for ln in ENV_FILE.read_text().splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    return env


ENV = load_env()
NIM_KEY = ENV.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_API_KEY", "")
NIM_BASE = "https://integrate.api.nvidia.com/v1"

# 5 models from diverse families, all confirmed reachable. Each family brings a
# different training bias so the panel surfaces independent reasoning, not echo.
PANEL = [
    ("mistral-large-3",   "mistralai/mistral-large-3-675b-instruct-2512"),
    ("llama-3.3-70b",     "meta/llama-3.3-70b-instruct"),
    ("gpt-oss-120b",      "openai/gpt-oss-120b"),
    ("qwen3-thinking",    "qwen/qwen3-next-80b-a3b-thinking"),
    ("kimi-k2.6",         "moonshotai/kimi-k2.6"),
]


def fetch_today_adr_fills() -> list[dict]:
    """Pull today's ADR-symbol fills from Alpaca AEGIS."""
    key = ENV.get("ALPACA_AEGIS_KEY", "")
    sec = ENV.get("ALPACA_AEGIS_SECRET", "")
    if not key or not sec:
        return []
    try:
        r = requests.get(
            "https://paper-api.alpaca.markets/v2/account/activities",
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec},
            params={"activity_types": "FILL", "page_size": 100},
            timeout=10,
        )
        if r.status_code != 200:
            return []
        adr_syms = {"BABA", "JD", "BIDU", "NTES", "TCOM"}
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        fills = []
        for f in r.json():
            if (f.get("symbol") in adr_syms
                    and (f.get("transaction_time") or "").startswith(today)):
                fills.append({
                    "time":   f.get("transaction_time", "")[:19],
                    "symbol": f.get("symbol"),
                    "side":   f.get("side"),
                    "qty":    f.get("qty"),
                    "price":  float(f.get("price") or 0),
                    "type":   f.get("type"),
                })
        return sorted(fills, key=lambda x: x["time"])
    except Exception as exc:
        print(f"  fetch err: {exc}")
        return []


def build_prompt(fills: list[dict]) -> str:
    """Construct the diagnostic prompt with today's actual data + backtest context."""
    fill_lines = []
    for f in fills:
        fill_lines.append(
            f"  {f['time'][11:19]} {f['side']:11} {f['qty']:>4} {f['symbol']:5} @ ${f['price']:,.2f} ({f['type']})"
        )
    fills_block = "\n".join(fill_lines) if fill_lines else "  (no fills today)"

    return f"""ROLE: Senior quant analyst for a US-HK ADR statistical arbitrage strategy.

CONTEXT — TODAY (2026-05-13, paper-money, Day 1 of 30-day shadow):
  Strategy: ADR cross-listing pairs (BABA/9988.HK, JD/9618.HK, BIDU/9888.HK, NTES/9999.HK, TCOM/9961.HK).
  Action rule: Open when |dev_bps| > 80. Stop when |dev_bps| > 250.
  Cycle: every 30 min.

TODAY'S ACTUAL FILLS ON ADR SYMBOLS (Alpaca paper):
{fills_block}

KEY EVENTS TODAY:
  13:57 BST: ADR strategy opened shorts on BABA (139), JD (588), BIDU (stop -68), TCOM (376).
              Deviations at entry: BABA +396 bps, JD +233 bps, BIDU +467 bps (stop trigger), TCOM +119 bps.
  14:57 BST: Next cycle. BIDU stopped at -$269 loss. Partial fills on BABA (115/139), TCOM (343/376).
              Realized -$1,137 on AEGIS today. Open shorts still down -$258 unrealized.

BACKTEST DISTRIBUTION (1,543 historical ADR trades, cleaned data 2021-2025):
  Win rate: 66.0% (so 34% of trades lose, by design)
  Mean net P&L per trade: +65.8 bps
  Median: +49.9 bps
  Std dev: 184.3 bps
  Worst 10% of trades: ≤ -140 bps
  Worst single trade: -970 bps
  Best: +821 bps

BACKTESTED MONTHLY DISTRIBUTION (60 months OOS):
  Best month:    +23.2%
  Worst month:   -1.1%
  88% of months positive
  Mean month:    +3.1%

CRITICAL CONSTRAINTS:
  - Paper trading only. Real $ at stake = $0.
  - Day 1 of 30. We will hold positions to natural exit per strategy logic.
  - "Failure" hypothesis must be supportable from THIS DATA, not just sentiment.

YOUR TASK — Answer in EXACTLY this JSON format. No markdown, no preamble:
{{
  "verdict": "STRATEGY_FAILURE" | "EXPECTED_VARIANCE" | "EXECUTION_PROBLEM" | "INSUFFICIENT_DATA",
  "confidence_0_to_1": float,
  "primary_reason": "<one sentence, no fluff>",
  "evidence_for_verdict": [
    "<bullet point 1>",
    "<bullet point 2>"
  ],
  "evidence_against_verdict": [
    "<counter-argument 1>"
  ],
  "actionable_change_today": "<concrete change or 'none — hold and observe'>"
}}

Be calibrated. Day 1 evidence is NOT a fair test of a strategy with 1,543-trade history.
But if you see something specific (wrong stop, sizing error, bad symbol), call it out."""


def call_one(model_id: str, prompt: str) -> tuple[str, dict | None, str | None]:
    """Make one NIM call, return (model_id, parsed_json, error)."""
    try:
        r = requests.post(
            f"{NIM_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {NIM_KEY}",
                "Content-Type":  "application/json",
            },
            json={
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens":  900,
                "temperature": 0.2,
                "top_p":       0.95,
            },
            timeout=120,
        )
        if r.status_code != 200:
            return model_id, None, f"HTTP {r.status_code}: {r.text[:200]}"
        text = r.json()["choices"][0]["message"]["content"].strip()
        # Strip code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            text = text.strip()
        # Locate the first { and last }
        s, e = text.find("{"), text.rfind("}")
        if s < 0 or e < 0:
            return model_id, None, f"no JSON found: {text[:120]}"
        parsed = json.loads(text[s:e + 1])
        return model_id, parsed, None
    except Exception as exc:
        return model_id, None, str(exc)


def main() -> int:
    if not NIM_KEY:
        print("NVIDIA_API_KEY not set — abort.", file=sys.stderr)
        return 1

    print("Fetching today's ADR fills…")
    fills = fetch_today_adr_fills()
    print(f"  fills found: {len(fills)}")
    for f in fills:
        print(f"    {f['time'][11:19]} {f['side']:11} {f['qty']:>4} {f['symbol']:5} @ ${f['price']:,.2f}")

    prompt = build_prompt(fills)
    print(f"\nPrompt size: {len(prompt)} chars")
    print(f"Calling {len(PANEL)} models in parallel…\n")

    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=len(PANEL)) as ex:
        futs = {ex.submit(call_one, mid, prompt): label for label, mid in PANEL}
        for fut in as_completed(futs):
            label = futs[fut]
            mid, parsed, err = fut.result()
            elapsed = time.time() - t0
            if parsed:
                results[label] = parsed
                v = parsed.get("verdict", "?")
                c = parsed.get("confidence_0_to_1", 0)
                print(f"  [{elapsed:5.1f}s] {label:18} verdict={v:22} conf={c}")
            else:
                errors[label] = err or "no parse"
                print(f"  [{elapsed:5.1f}s] {label:18} ERROR: {err[:120] if err else 'no parse'}")

    print("\n" + "=" * 78)
    print(f"PANEL VERDICTS — {len(results)}/{len(PANEL)} models responded")
    print("=" * 78)

    verdict_counts: dict[str, int] = {}
    for lbl, r in results.items():
        v = r.get("verdict", "?")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
    print("\nVerdict tally:")
    for v, n in sorted(verdict_counts.items(), key=lambda x: -x[1]):
        print(f"  {v:25}  {n} / {len(results)}")

    print("\n" + "-" * 78)
    for lbl, r in results.items():
        print(f"\n=== {lbl} ===  verdict: {r.get('verdict')}  conf: {r.get('confidence_0_to_1')}")
        print(f"  Reason: {r.get('primary_reason')}")
        print(f"  Evidence for:")
        for e in r.get("evidence_for_verdict", []):
            print(f"    + {e}")
        print(f"  Evidence against:")
        for e in r.get("evidence_against_verdict", []):
            print(f"    - {e}")
        print(f"  Action today: {r.get('actionable_change_today')}")

    if errors:
        print("\nErrors:")
        for lbl, err in errors.items():
            print(f"  {lbl}: {err[:200]}")

    # Persist
    out_dir = ROOT / "council" / "postmortem_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"adr_today_{stamp}.json"
    out_path.write_text(json.dumps({
        "timestamp_utc": stamp,
        "fills_today":  fills,
        "panel":        results,
        "errors":       errors,
        "verdict_tally": verdict_counts,
    }, indent=2, default=str))
    print(f"\nSaved → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
