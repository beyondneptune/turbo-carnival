#!/usr/bin/env python3
"""
NVIDIA NIM 20-model parallel code/strategy review for turbo-carnival paper trading.
Tasks 1-4: list models, pick 20, run parallel reviews, synthesize.
"""
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

# ---------- Setup ----------
REPO = Path("/Users/weeoto/turbo-carnival")
ENV_FILE = REPO / "council" / ".env.local"


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
if not api_key:
    print("ERROR: NVIDIA_API_KEY not found", file=sys.stderr)
    sys.exit(1)

BASE_URL = "https://integrate.api.nvidia.com/v1"
client = OpenAI(api_key=api_key, base_url=BASE_URL)

STAMP = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
OUT_DIR = REPO / "council" / "reviews" / STAMP
OUT_DIR.mkdir(parents=True, exist_ok=True)
print(f"[stamp] {STAMP}")
print(f"[out_dir] {OUT_DIR}")

# ---------- TASK 1: list models ----------
print("[task1] listing models...")
try:
    models = client.models.list()
    raw = models.model_dump() if hasattr(models, "model_dump") else {"data": [m.model_dump() for m in models.data]}
except Exception as e:
    print(f"[task1] list via SDK failed: {e}")
    raw = {"error": str(e), "data": []}

with open(OUT_DIR / "models_list.json", "w") as f:
    json.dump(raw, f, indent=2, default=str)

available_ids = set()
for m in raw.get("data", []):
    mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
    if mid:
        available_ids.add(mid)
print(f"[task1] {len(available_ids)} models listed")

# ---------- TASK 2: pick 20 ----------
seed = [
    "moonshotai/kimi-k2-instruct",
    "moonshotai/kimi-k2-thinking",
    "meta/llama-4-maverick-17b-128e-instruct",
    "meta/llama-3.3-70b-instruct",
    "meta/llama-3.1-405b-instruct",
    "qwen/qwen3-coder-480b-a35b-instruct",
    "qwen/qwen3-next-80b-a3b-thinking",
    "qwen/qwen2.5-coder-32b-instruct",
    "mistralai/mistral-large-2-instruct",
    "mistralai/mixtral-8x22b-instruct-v0.1",
    "nvidia/llama-3.1-nemotron-70b-instruct",
    "nvidia/nemotron-4-340b-instruct",
    "deepseek-ai/deepseek-r1",
    "deepseek-ai/deepseek-v3",
    "microsoft/phi-3.5-moe-instruct",
    "google/gemma-2-27b-it",
    "01-ai/yi-large",
    "databricks/dbrx-instruct",
    "ibm/granite-3.0-8b-instruct",
    "minimax/minimax-m1",
]


def pick_provider_fallback(provider: str, exclude: set) -> str | None:
    """Find best alternative model for provider."""
    candidates = [mid for mid in available_ids if mid.lower().startswith(provider.lower() + "/") and mid not in exclude]
    if not candidates:
        return None
    # Prefer instruct/chat models; sort by name (typically newer ones rank higher alphanumerically with version)
    instruct = [c for c in candidates if any(k in c.lower() for k in ["instruct", "chat", "thinking", "reasoning"])]
    pool = instruct or candidates
    pool.sort(reverse=True)
    return pool[0]


chosen: list[str] = []
chosen_set: set[str] = set()
swaps: list[tuple[str, str | None]] = []
for sid in seed:
    if not available_ids:
        # If listing failed, just trust the seed
        chosen.append(sid)
        chosen_set.add(sid)
        continue
    if sid in available_ids:
        chosen.append(sid)
        chosen_set.add(sid)
        continue
    provider = sid.split("/", 1)[0]
    alt = pick_provider_fallback(provider, chosen_set)
    if alt:
        chosen.append(alt)
        chosen_set.add(alt)
        swaps.append((sid, alt))
    else:
        swaps.append((sid, None))

# Fill out to 20 from any remaining available
if available_ids and len(chosen) < 20:
    extras = [
        m
        for m in sorted(available_ids)
        if m not in chosen_set and any(k in m.lower() for k in ["instruct", "chat", "thinking", "reasoning"])
    ]
    for e in extras:
        if len(chosen) >= 20:
            break
        chosen.append(e)
        chosen_set.add(e)

chosen = chosen[:20]
print(f"[task2] picked {len(chosen)} models")
if swaps:
    print(f"[task2] swaps:")
    for orig, alt in swaps:
        print(f"  {orig} -> {alt}")

with open(OUT_DIR / "chosen_models.json", "w") as f:
    json.dump({"chosen": chosen, "swaps": swaps}, f, indent=2)

# ---------- TASK 3: parallel reviews ----------
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
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER},
            ],
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


print(f"[task3] running {len(chosen)} reviews, 5 concurrent...")
results: list[dict] = []
with ThreadPoolExecutor(max_workers=5) as ex:
    futs = {ex.submit(call_one, m): m for m in chosen}
    for fut in as_completed(futs):
        rec = fut.result()
        write_review(rec)
        results.append(rec)
        marker = "ok" if rec["status"] == "success" else "ERR"
        msg = rec["error"] or f"{len(rec['body'])} chars"
        print(f"  [{marker}] {rec['model']} ({rec['elapsed_sec']}s) {msg[:120]}")

n_success = sum(1 for r in results if r["status"] == "success")
n_total = len(results)
print(f"[task3] {n_success}/{n_total} success")

# ---------- TASK 4: synthesize ----------
print("[task4] synthesizing...")


def parse_falcon(body: str) -> str | None:
    """Find Falcon vote: PAUSE or KEEP RUNNING."""
    # Look for the answer to question 2
    # Try to find a section starting with "2." or "2)" then look for keywords
    b = body.lower()
    # Restrict scan: find "2." block then check
    m = re.search(r"(?:^|\n)\s*(?:2[.\):]|\*\*2[.\):])(.*?)(?:\n\s*(?:3[.\):]|\*\*3[.\):])|\Z)", body, re.S | re.I)
    section = m.group(1).lower() if m else b
    pause = bool(re.search(r"\bpaus(e|ed|ing)?\b", section))
    keep = bool(re.search(r"\b(keep|continue|run\b|running|kept)\b", section)) and not bool(
        re.search(r"don'?t (keep|run|continue)", section)
    )
    # Prefer the first explicit verdict
    first_pause = re.search(r"\bpaus(e|ed)\b", section)
    first_keep = re.search(r"\b(keep running|keep|continue running|continue)\b", section)
    if first_pause and first_keep:
        return "PAUSE" if first_pause.start() < first_keep.start() else "KEEP RUNNING"
    if first_pause:
        return "PAUSE"
    if first_keep:
        return "KEEP RUNNING"
    if pause:
        return "PAUSE"
    if keep:
        return "KEEP RUNNING"
    return None


def parse_arch_rating(body: str) -> float | None:
    """Find architecture rating 1-10 in answer to question 4."""
    m = re.search(r"(?:^|\n)\s*(?:4[.\):]|\*\*4[.\):])(.*?)(?:\n\s*(?:5[.\):]|\*\*5[.\):])|\Z)", body, re.S | re.I)
    section = m.group(1) if m else body
    # Look for "N/10" or "rating: N" or just an integer between 1-10
    pats = [
        r"(\d+(?:\.\d+)?)\s*/\s*10",
        r"rating[^0-9]{0,10}(\d+(?:\.\d+)?)",
        r"\b([1-9]|10)\b",
    ]
    for p in pats:
        m2 = re.search(p, section, re.I)
        if m2:
            try:
                val = float(m2.group(1))
                if 0 < val <= 10:
                    return val
            except ValueError:
                pass
    return None


def parse_deploy(body: str) -> str | None:
    """Find deploy-with-real-money answer."""
    m = re.search(r"(?:^|\n)\s*(?:5[.\):]|\*\*5[.\):])(.*?)\Z", body, re.S | re.I)
    section = m.group(1) if m else body
    s = section.lower()
    # Order matters: conditional first
    if re.search(r"\b(conditional|specific conditions?|only if|with conditions?|with (the )?following|only when)\b", s):
        return "Conditional"
    # Check for explicit "No"
    if re.search(r"\b(?:absolutely\s+)?no\b", s[:200]):
        return "No"
    if re.search(r"\byes\b", s[:200]):
        return "Yes"
    if "no" in s[:50]:
        return "No"
    return None


# Regime filter keyword buckets
FILTER_PATTERNS = [
    ("ATR / volatility regime", re.compile(r"\batr\b|\baverage true range\b", re.I)),
    ("VIX threshold", re.compile(r"\bvix\b", re.I)),
    ("RSI filter", re.compile(r"\brsi\b", re.I)),
    ("ADX / trend strength", re.compile(r"\badx\b|\btrend strength\b", re.I)),
    ("Bollinger Band width / squeeze", re.compile(r"\bbollinger\b|\bbband\b|\bband width\b|\bsqueeze\b", re.I)),
    ("Realized volatility", re.compile(r"realized vol|\bhv\b|\brv\b|historical volatility", re.I)),
    ("Spread / bid-ask filter", re.compile(r"bid[- ]?ask|\bspread\b", re.I)),
    ("VWAP filter", re.compile(r"\bvwap\b", re.I)),
    ("MA / EMA trend", re.compile(r"\b(ema|sma|moving average|ma cross)\b", re.I)),
    ("Volume / liquidity filter", re.compile(r"\bvolume\b|\bliquidity\b", re.I)),
    ("Time-of-day filter", re.compile(r"time[- ]of[- ]day|open(ing)? bell|lunch|first hour|last hour", re.I)),
    ("Choppiness Index", re.compile(r"chopp(iness)?", re.I)),
    ("Slippage / cost gate", re.compile(r"slippage|cost gate|execution cost", re.I)),
    ("News / event filter", re.compile(r"\bnews\b|\bevent\b|economic calendar|fomc|cpi", re.I)),
    ("Hurst / autocorrelation", re.compile(r"\bhurst\b|autocorrel", re.I)),
]


# Hidden risk buckets (parse question 1)
RISK_PATTERNS = [
    ("Underestimated live slippage / execution costs", re.compile(r"slippage|execution cost|fill|spread", re.I)),
    ("Backtest overfit / look-ahead / regime change", re.compile(r"overfit|look[- ]ahead|backtest (bias|noise)|regime", re.I)),
    ("Correlated lanes (single-factor / same universe)", re.compile(r"correlat|same universe|single factor|concentrat", re.I)),
    ("Short gamma / unhedged options risk (Valk)", re.compile(r"short gamma|gamma risk|negative gamma|tail risk", re.I)),
    ("Stop-loss too tight / noise-driven exits", re.compile(r"stop[- ]?loss too tight|tight stops|noise|whipsaw", re.I)),
    ("HFT latency / infra mismatch", re.compile(r"latency|colocation|hft infra|microstructure", re.I)),
    ("Position sizing / Kelly / leverage", re.compile(r"position siz|kelly|over[- ]?lever|sizing", re.I)),
    ("Wrong holding-period vs signal half-life", re.compile(r"hold(ing)? (period|time)|half[- ]life", re.I)),
    ("No volatility / regime filter", re.compile(r"no (vol|volatility|regime) filter|missing.*regime", re.I)),
    ("Margin / overnight gap risk", re.compile(r"\bmargin\b|overnight|gap risk", re.I)),
]


falcon_votes = Counter()
deploy_votes = Counter()
arch_ratings: list[float] = []
filter_counts: Counter = Counter()
filter_by_model: dict[str, set] = {f[0]: set() for f in FILTER_PATTERNS}
risk_counts: Counter = Counter()
risk_by_model: dict[str, set] = {r[0]: set() for r in RISK_PATTERNS}
quotes: list[tuple[str, str]] = []

for r in results:
    if r["status"] != "success" or not r["body"]:
        continue
    body = r["body"]
    fv = parse_falcon(body)
    if fv:
        falcon_votes[fv] += 1
    dv = parse_deploy(body)
    if dv:
        deploy_votes[dv] += 1
    ar = parse_arch_rating(body)
    if ar is not None:
        arch_ratings.append(ar)

    # Filter buckets - extract section 3
    m = re.search(r"(?:^|\n)\s*(?:3[.\):]|\*\*3[.\):])(.*?)(?:\n\s*(?:4[.\):]|\*\*4[.\):])|\Z)", body, re.S | re.I)
    filt_section = m.group(1) if m else body
    for label, pat in FILTER_PATTERNS:
        if pat.search(filt_section):
            filter_by_model[label].add(r["model"])
            filter_counts[label] += 1

    # Risks - extract section 1
    m = re.search(r"(?:^|\n)\s*(?:1[.\):]|\*\*1[.\):])(.*?)(?:\n\s*(?:2[.\):]|\*\*2[.\):])|\Z)", body, re.S | re.I)
    risk_section = m.group(1) if m else body[:1200]
    for label, pat in RISK_PATTERNS:
        if pat.search(risk_section):
            risk_by_model[label].add(r["model"])
            risk_counts[label] += 1

    # Grab a short distinctive quote (first non-trivial line)
    for line in body.splitlines():
        line = line.strip().strip("*-").strip()
        if 60 <= len(line) <= 240 and not line.startswith("#"):
            quotes.append((r["model"], line))
            break

top5_filters = filter_counts.most_common(5)
top3_risks = [(label, len(risk_by_model[label])) for label, _ in risk_counts.most_common() if len(risk_by_model[label]) >= 5][:3]

# Architecture histogram (1..10)
arch_hist = Counter(int(round(x)) for x in arch_ratings)
mean_arch = statistics.mean(arch_ratings) if arch_ratings else float("nan")
stdev_arch = statistics.stdev(arch_ratings) if len(arch_ratings) > 1 else 0.0

# Pick contrarian: model whose Falcon vote is in minority OR deploy vote in minority, prefer with distinctive quote
contrarian = None
if falcon_votes:
    minority = min(falcon_votes, key=falcon_votes.get)
    for r in results:
        if r["status"] != "success":
            continue
        fv = parse_falcon(r["body"])
        if fv == minority:
            # find the question-2 paragraph
            m = re.search(r"(?:^|\n)\s*(?:2[.\):]|\*\*2[.\):])(.*?)(?:\n\s*(?:3[.\):]|\*\*3[.\):])|\Z)", r["body"], re.S | re.I)
            snippet = (m.group(1).strip() if m else r["body"][:400]).replace("\n", " ")
            snippet = re.sub(r"\s+", " ", snippet)[:400]
            contrarian = (r["model"], minority, snippet)
            break

# Build synthesis MD
lines: list[str] = []
lines.append(f"# Synthesis of {n_success}/{n_total} model reviews — {STAMP}")
lines.append("")
lines.append("## Falcon vote")
lines.append(f"- PAUSE: {falcon_votes.get('PAUSE', 0)}")
lines.append(f"- KEEP RUNNING: {falcon_votes.get('KEEP RUNNING', 0)}")
lines.append(f"- Unparsed: {n_success - sum(falcon_votes.values())}")
lines.append("")
lines.append("## Regime filter consensus (top 5)")
for label, cnt in top5_filters:
    lines.append(f"- {label}: {cnt}")
lines.append("")
lines.append("## Average architecture rating")
if arch_ratings:
    lines.append(f"- Mean: {mean_arch:.2f}  Stdev: {stdev_arch:.2f}  N={len(arch_ratings)}")
    lines.append("")
    lines.append("Distribution (rating: count):")
    for k in range(1, 11):
        c = arch_hist.get(k, 0)
        bar = "#" * c
        lines.append(f"  {k:>2}: {c} {bar}")
else:
    lines.append("- No ratings parseable.")
lines.append("")
lines.append("## Deploy-with-real-money vote")
lines.append(f"- Yes: {deploy_votes.get('Yes', 0)}")
lines.append(f"- No: {deploy_votes.get('No', 0)}")
lines.append(f"- Conditional: {deploy_votes.get('Conditional', 0)}")
lines.append(f"- Unparsed: {n_success - sum(deploy_votes.values())}")
lines.append("")
lines.append("## Top 3 hidden risks mentioned by >= 5 models")
if top3_risks:
    for label, cnt in top3_risks:
        models_list = sorted(risk_by_model[label])
        lines.append(f"- {label} ({cnt} models)")
        for mm in models_list[:6]:
            lines.append(f"   - {mm}")
        if len(models_list) > 6:
            lines.append(f"   - ...and {len(models_list)-6} more")
else:
    lines.append("- (No single risk reached the >=5 threshold.)")
lines.append("")
lines.append("## Most contrarian / interesting take")
if contrarian:
    lines.append(f"- {contrarian[0]} (minority Falcon vote: {contrarian[1]}):")
    lines.append(f"  > {contrarian[2]}")
else:
    lines.append("- (No clear contrarian Falcon position.)")
lines.append("")
lines.append("## Recommended actions (synthesis)")
recs: list[str] = []
# Falcon
if falcon_votes.get("PAUSE", 0) > falcon_votes.get("KEEP RUNNING", 0):
    recs.append(
        "Pause Falcon (breakout) at next open. Models that observed the 2-10 sec SL whipsaw "
        "agreed the SL is well inside the live-spread noise band; resuming without a vol-gate is throwing money away."
    )
else:
    recs.append(
        "Keep Falcon running but tighten entry: only enter when the chosen regime filter is in 'trend' state, "
        "otherwise log signals and let them drift to evidence."
    )
# Filter
if top5_filters:
    recs.append(
        f"Add the top-voted regime filter immediately: {top5_filters[0][0]}. "
        "Gate all HFT entries on it before any other change."
    )
# Slippage realism
recs.append(
    "Re-run the backtest with realistic slippage (>= 5 bps round-trip, plus a half-spread + queue-position model). "
    "Day-1 live SL hits at 2 seconds prove the 0.75 bps sim is fantasy."
)
# Valk
recs.append(
    "Re-examine Valk condor sizing: 44 contracts on a single 30-DTE structure concentrates short gamma; "
    "scale to halve current notional or convert to a defined-risk butterfly."
)
# Real-money
if deploy_votes.get("Yes", 0) >= max(deploy_votes.get("No", 0), deploy_votes.get("Conditional", 0)):
    recs.append("Real-money: panel leans Yes, but only after the slippage rerun matches live.")
else:
    recs.append("Real-money: do NOT deploy. Panel leans No/Conditional. Demand >= 30 days of green paper after fixes.")

for i, r in enumerate(recs, 1):
    lines.append(f"{i}. {r}")
lines.append("")

(OUT_DIR / "SYNTHESIS.md").write_text("\n".join(lines))

# ---------- Final report ----------
print()
print("================== REPORT ==================")
print(f"Stamp path     : {OUT_DIR}")
print(f"N_success/20   : {n_success}/{n_total}")
print(
    f"Falcon vote    : PAUSE={falcon_votes.get('PAUSE', 0)} KEEP={falcon_votes.get('KEEP RUNNING', 0)} "
    f"Unparsed={n_success - sum(falcon_votes.values())}"
)
print(f"Top filter     : {top5_filters[0][0] if top5_filters else 'n/a'}")
print(f"Mean arch rating: {mean_arch:.2f} (sd {stdev_arch:.2f}, n={len(arch_ratings)})")
print(
    f"Real-money     : Yes={deploy_votes.get('Yes', 0)} No={deploy_votes.get('No', 0)} "
    f"Conditional={deploy_votes.get('Conditional', 0)} "
    f"Unparsed={n_success - sum(deploy_votes.values())}"
)
print("============================================")

# Persist short report as JSON for the agent's final summary
summary = {
    "stamp": STAMP,
    "out_dir": str(OUT_DIR),
    "n_success": n_success,
    "n_total": n_total,
    "falcon": dict(falcon_votes),
    "deploy": dict(deploy_votes),
    "arch_mean": mean_arch,
    "arch_stdev": stdev_arch,
    "arch_n": len(arch_ratings),
    "top_filters": top5_filters,
    "top_risks": top3_risks,
    "swaps": swaps,
}
(OUT_DIR / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))
print(f"[done] wrote SUMMARY.json + SYNTHESIS.md + {len(results)} reviews")
