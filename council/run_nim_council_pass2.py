#!/usr/bin/env python3
"""
NVIDIA NIM second-pass review.
Run the same brutal-honesty prompt against EVERY remaining chat-capable model on NIM,
skipping anything we already attempted in the first run at council/reviews/20260511T191453Z/.
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
PRIOR_DIR = REPO / "council" / "reviews" / "20260511T191453Z"


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


def slug(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", model_id.replace("/", "_").replace(":", "_"))


# ---------- STEP 1: list all models ----------
print("[step1] listing models...")
try:
    models = client.models.list()
    raw = models.model_dump() if hasattr(models, "model_dump") else {"data": [m.model_dump() for m in models.data]}
except Exception as e:
    print(f"[step1] list via SDK failed: {e}", file=sys.stderr)
    raw = {"error": str(e), "data": []}

with open(OUT_DIR / "models_list_full.json", "w") as f:
    json.dump(raw, f, indent=2, default=str)

all_ids: list[str] = []
seen = set()
for m in raw.get("data", []):
    mid = m.get("id") if isinstance(m, dict) else getattr(m, "id", None)
    if mid and mid not in seen:
        seen.add(mid)
        all_ids.append(mid)
print(f"[step1] {len(all_ids)} unique model ids listed")


# ---------- STEP 2: filter ----------
# Build the set of slugs already attempted (any .md file in the prior run)
prior_slugs: set[str] = set()
prior_ids: set[str] = set()
for f in PRIOR_DIR.glob("*.md"):
    if f.name in {"SYNTHESIS.md"}:
        continue
    prior_slugs.add(f.stem)
    # Try to extract the model id from frontmatter
    txt = f.read_text()
    m = re.search(r"^model:\s*(.+)$", txt, re.M)
    if m:
        prior_ids.add(m.group(1).strip())
print(f"[step2] prior run touched {len(prior_slugs)} models")


# Exclusion keyword groups for non-chat models
EXCLUDE_KEYWORDS = [
    # embeddings
    "embed", "embedding", "nemoretriever", "nvembed", "nv-embed", "arctic-embed", "bge-",
    # vision-only / image-only
    "vision-only", "image-tagger", "clip", "deplot", "kosmos", "vila", "neva",
    "nvclip", "fuyu", "cosmos-reason", "synthetic-video", "phi-3-vision",
    # rerankers
    "rerank",
    # audio/speech/tts
    "audio", "speech", "tts", "riva",
    # safety/guard
    "guard", "safety", "nemotron-content-safety", "agentiq", "aegis",
    "topic-control", "gliner-pii",
    # OCR / extract
    "ocr", "paddleocr",
    "nemoretriever-parse", "nemotron-parse",
    # reward
    "reward",
]

def is_chat_capable(mid: str) -> tuple[bool, str]:
    low = mid.lower()
    # Hard exclusions first
    for kw in EXCLUDE_KEYWORDS:
        if kw in low:
            return False, f"excluded:{kw}"
    # Code-completion only (without instruct/chat suffix)
    if low.endswith("-base") or low.endswith("-completion"):
        return False, "base/completion"
    # Pure code-base models (no instruct/chat suffix)
    if "starcoder" in low and "instruct" not in low and "chat" not in low:
        return False, "code-base"
    if low in {"google/codegemma-1.1-7b", "google/codegemma-7b"}:
        return False, "code-base"
    # Permissive: if no exclusion matched, keep it. Let the API error filter the rest.
    return True, "keep"


candidates: list[str] = []
drops: list[tuple[str, str]] = []
for mid in all_ids:
    if mid in prior_ids:
        drops.append((mid, "prior run"))
        continue
    if slug(mid) in prior_slugs:
        drops.append((mid, "prior slug"))
        continue
    ok, reason = is_chat_capable(mid)
    if ok:
        candidates.append(mid)
    else:
        drops.append((mid, reason))

candidates.sort()
print(f"[step2] {len(candidates)} candidates after filtering (dropped {len(drops)})")
for mid, reason in drops[:20]:
    print(f"  drop: {mid} ({reason})")
if len(drops) > 20:
    print(f"  ...and {len(drops)-20} more drops")

(OUT_DIR / "candidates.json").write_text(
    json.dumps({"candidates": candidates, "drops": drops, "prior_slugs": sorted(prior_slugs)}, indent=2)
)


# ---------- STEP 3: parallel reviews ----------
SYSTEM = "You output direct, concise analysis. No preamble."
USER = """You're reviewing a LIVE paper-options + HFT system. Be brutally honest. This is paper money, your harshness will not harm anyone.

3 Alpaca paper accounts ($100k each, today):
- Aegis (-$709 today): HFT mean-reversion, [SPY,QQQ,NVDA,TSLA,AAPL,META,AMD]. 25 bps signal, 100 bps TP, 35 bps SL, 1200s hold. Force-flat at end of day.
- Falcon (-$2,290 today): HFT breakout, same universe. 18 bps signal, 70 bps TP, 25 bps SL, 900s hold. Whipsawed badly all afternoon — entries hit SL within 2-10 seconds repeatedly. Force-flat at end of day.
- Valk (-$444 today): SPY iron condor 30-DTE, qty 44, entry credit 0.54, target 30% credit, stop 3x. Still holding.

CONTEXT:
- Backtest was 21-day cached 1-min bars, +17% / -8% DD. Sim slippage 0.75 bps. Live slippage is way higher than that, judging by repeated 2-second stop-outs.
- Total drawdown across 3 lanes: -$3,443 on $300k in 5.5 hours (-1.15%).
- This is day 1. We have 89 days more in the "investor demo" plan.
- 20 other LLMs unanimously voted to pause Falcon. We have already paused Falcon and killed the HFT host.

ANSWER THESE 5 QUESTIONS IN UNDER 500 WORDS. NO PREAMBLE.

1. Biggest hidden risk in this setup?
2. Should Falcon (breakout) stay PAUSED tomorrow, or do you have a reason to override the 20-model consensus and resume it? Pick PAUSED or RESUME. Why?
3. One specific regime filter to add immediately (be concrete: which indicator, what threshold)?
4. Architecture rating 1-10 for surviving a 90-day live demo.
5. Would you deploy this with REAL money? Yes / No / Specific conditions only.
"""


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
            timeout=120,
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


print(f"[step3] running {len(candidates)} reviews, 5 concurrent...")
results: list[dict] = []
with ThreadPoolExecutor(max_workers=5) as ex:
    futs = {ex.submit(call_one, m): m for m in candidates}
    for fut in as_completed(futs):
        rec = fut.result()
        write_review(rec)
        results.append(rec)
        marker = "ok" if rec["status"] == "success" else "ERR"
        msg = rec["error"] or f"{len(rec['body'])} chars"
        print(f"  [{marker}] {rec['model']} ({rec['elapsed_sec']}s) {msg[:160]}")

n_success = sum(1 for r in results if r["status"] == "success")
n_total = len(results)
print(f"[step3] {n_success}/{n_total} success")


# ---------- STEP 4: synthesis ----------
print("[step4] synthesizing...")


def parse_falcon(body: str) -> str | None:
    """Find Falcon vote: PAUSED or RESUME."""
    m = re.search(r"(?:^|\n)\s*(?:\*\*)?2[.\):](.*?)(?:\n\s*(?:\*\*)?3[.\):]|\Z)", body, re.S | re.I)
    section = (m.group(1) if m else body).lower()
    first_paused = re.search(r"\bpaus(?:e|ed)\b", section)
    first_resume = re.search(r"\b(?:resume|resumed|resuming|override|keep running|continue running|unpause|un-pause)\b", section)
    if first_paused and first_resume:
        return "PAUSED" if first_paused.start() < first_resume.start() else "RESUME"
    if first_paused:
        return "PAUSED"
    if first_resume:
        return "RESUME"
    return None


def parse_arch_rating(body: str) -> float | None:
    m = re.search(r"(?:^|\n)\s*(?:\*\*)?4[.\):](.*?)(?:\n\s*(?:\*\*)?5[.\):]|\Z)", body, re.S | re.I)
    section = m.group(1) if m else body
    pats = [
        r"(\d+(?:\.\d+)?)\s*/\s*10",
        r"rating[^0-9]{0,10}(\d+(?:\.\d+)?)",
        r"\b([1-9](?:\.\d+)?|10)\b",
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
    m = re.search(r"(?:^|\n)\s*(?:\*\*)?5[.\):](.*?)\Z", body, re.S | re.I)
    section = m.group(1) if m else body
    s = section.lower()
    if re.search(r"\b(?:conditional|specific conditions?|only if|with conditions?|with (?:the )?following|only when|hell no(?:t)? unconditional)\b", s):
        return "Conditional"
    head = s[:240]
    if re.search(r"\b(?:absolutely\s+)?no\b", head):
        return "No"
    if re.search(r"\byes\b", head):
        return "Yes"
    return None


FILTER_PATTERNS = [
    ("ATR / volatility regime", re.compile(r"\batr\b|\baverage true range\b", re.I)),
    ("VIX threshold", re.compile(r"\bvix\b", re.I)),
    ("RSI filter", re.compile(r"\brsi\b", re.I)),
    ("ADX / trend strength", re.compile(r"\badx\b|\btrend strength\b", re.I)),
    ("Bollinger Band width / squeeze", re.compile(r"\bbollinger\b|\bbband\b|\bband width\b|\bsqueeze\b", re.I)),
    ("Realized volatility", re.compile(r"realized vol|\bhv\b|\brv\b|historical volatility", re.I)),
    ("Spread / bid-ask filter", re.compile(r"bid[- ]?ask|\bspread\b", re.I)),
    ("VWAP filter", re.compile(r"\bvwap\b", re.I)),
    ("MA / EMA trend", re.compile(r"\b(?:ema|sma|moving average|ma cross)\b", re.I)),
    ("Volume / liquidity filter", re.compile(r"\bvolume\b|\bliquidity\b", re.I)),
    ("Time-of-day filter", re.compile(r"time[- ]of[- ]day|open(?:ing)? bell|lunch|first hour|last hour|opening 30", re.I)),
    ("Choppiness Index", re.compile(r"chopp(?:iness)?", re.I)),
    ("Slippage / cost gate", re.compile(r"slippage|cost gate|execution cost", re.I)),
    ("News / event filter", re.compile(r"\bnews\b|\bevent\b|economic calendar|fomc|cpi", re.I)),
    ("Hurst / autocorrelation", re.compile(r"\bhurst\b|autocorrel", re.I)),
    ("Realized spread / quote stability", re.compile(r"quote stability|quote noise|microstructure noise", re.I)),
    ("Order-flow imbalance", re.compile(r"order[- ]?flow|imbalance|ofi", re.I)),
]


RISK_PATTERNS = [
    ("Underestimated live slippage / execution costs", re.compile(r"slippage|execution cost|fill quality|spread cost", re.I)),
    ("Backtest overfit / look-ahead / regime mismatch", re.compile(r"overfit|look[- ]ahead|backtest (?:bias|noise|fantasy|unrealistic)|regime", re.I)),
    ("Correlated lanes (single-factor / same universe)", re.compile(r"correlat|same universe|single factor|concentrat", re.I)),
    ("Short gamma / unhedged options tail risk (Valk)", re.compile(r"short gamma|gamma risk|negative gamma|tail risk|condor blow", re.I)),
    ("Stop-loss too tight / noise-driven exits", re.compile(r"stop[- ]?loss too tight|tight stops|noise|whipsaw", re.I)),
    ("HFT latency / infra mismatch (no co-lo)", re.compile(r"latency|colocation|co-?lo|hft infra|microstructure", re.I)),
    ("Position sizing / Kelly / leverage", re.compile(r"position siz|kelly|over[- ]?lever|sizing", re.I)),
    ("Wrong holding period vs signal half-life", re.compile(r"hold(?:ing)? (?:period|time)|half[- ]life|signal decay", re.I)),
    ("No volatility / regime filter", re.compile(r"no (?:vol|volatility|regime) filter|missing.*regime|absence.*regime", re.I)),
    ("Margin / overnight gap risk", re.compile(r"\bmargin\b|overnight|gap risk", re.I)),
    ("Survivorship / look-good for investors (demo bias)", re.compile(r"demo bias|investor demo|sunk cost|career risk|narrative risk", re.I)),
    ("Pin / assignment risk", re.compile(r"\bpin risk\b|assignment", re.I)),
]


falcon_votes = Counter()
deploy_votes = Counter()
arch_ratings: list[tuple[str, float]] = []
filter_counts: Counter = Counter()
filter_by_model: dict[str, set] = {f[0]: set() for f in FILTER_PATTERNS}
risk_counts: Counter = Counter()
risk_by_model: dict[str, set] = {r[0]: set() for r in RISK_PATTERNS}
model_falcon_vote: dict[str, str] = {}
model_deploy_vote: dict[str, str] = {}

for r in results:
    if r["status"] != "success" or not r["body"]:
        continue
    body = r["body"]
    fv = parse_falcon(body)
    if fv:
        falcon_votes[fv] += 1
        model_falcon_vote[r["model"]] = fv
    dv = parse_deploy(body)
    if dv:
        deploy_votes[dv] += 1
        model_deploy_vote[r["model"]] = dv
    ar = parse_arch_rating(body)
    if ar is not None:
        arch_ratings.append((r["model"], ar))

    m = re.search(r"(?:^|\n)\s*(?:\*\*)?3[.\):](.*?)(?:\n\s*(?:\*\*)?4[.\):]|\Z)", body, re.S | re.I)
    filt_section = m.group(1) if m else body
    for label, pat in FILTER_PATTERNS:
        if pat.search(filt_section):
            filter_by_model[label].add(r["model"])
            filter_counts[label] += 1

    m = re.search(r"(?:^|\n)\s*(?:\*\*)?1[.\):](.*?)(?:\n\s*(?:\*\*)?2[.\):]|\Z)", body, re.S | re.I)
    risk_section = m.group(1) if m else body[:1500]
    for label, pat in RISK_PATTERNS:
        if pat.search(risk_section):
            risk_by_model[label].add(r["model"])
            risk_counts[label] += 1

top5_filters = filter_counts.most_common(5)
top3_risks = [(label, len(risk_by_model[label])) for label, _ in risk_counts.most_common()][:3]

arch_only_vals = [v for _, v in arch_ratings]
mean_arch = statistics.mean(arch_only_vals) if arch_only_vals else float("nan")
stdev_arch = statistics.stdev(arch_only_vals) if len(arch_only_vals) > 1 else 0.0

# Contrarian / interesting: anyone who said RESUME, OR low arch rating with sharp deploy stance, OR a particularly long+distinctive reply
defectors: list[tuple[str, str]] = []
for r in results:
    if r["status"] != "success" or not r["body"]:
        continue
    if model_falcon_vote.get(r["model"]) == "RESUME":
        m = re.search(r"(?:^|\n)\s*(?:\*\*)?2[.\):](.*?)(?:\n\s*(?:\*\*)?3[.\):]|\Z)", r["body"], re.S | re.I)
        snip = re.sub(r"\s+", " ", (m.group(1) if m else r["body"][:400]).strip())[:400]
        defectors.append((r["model"], snip))

# Most contrarian generally: pick 3 reviews — RESUME defectors first, then unusual deploy=Yes, then lowest arch rating
contrarian_picks: list[tuple[str, str]] = []
for m, s in defectors[:3]:
    contrarian_picks.append((m, f"FALCON=RESUME — {s}"))

if len(contrarian_picks) < 3:
    yes_models = [m for m, v in model_deploy_vote.items() if v == "Yes"]
    for m in yes_models:
        if any(m == cm for cm, _ in contrarian_picks):
            continue
        rec = next((rr for rr in results if rr["model"] == m), None)
        if not rec:
            continue
        m5 = re.search(r"(?:^|\n)\s*(?:\*\*)?5[.\):](.*?)\Z", rec["body"], re.S | re.I)
        snip = re.sub(r"\s+", " ", (m5.group(1) if m5 else rec["body"][:400]).strip())[:400]
        contrarian_picks.append((m, f"DEPLOY=YES — {snip}"))
        if len(contrarian_picks) >= 3:
            break

if len(contrarian_picks) < 3 and arch_ratings:
    lowest = sorted(arch_ratings, key=lambda x: x[1])[:3]
    for m, val in lowest:
        if any(m == cm for cm, _ in contrarian_picks):
            continue
        rec = next((rr for rr in results if rr["model"] == m), None)
        if not rec:
            continue
        first = next((ln.strip().strip("*-").strip() for ln in rec["body"].splitlines() if 40 <= len(ln.strip()) <= 240), rec["body"][:240])
        contrarian_picks.append((m, f"ARCH={val} — {first}"))
        if len(contrarian_picks) >= 3:
            break


# Build SYNTHESIS.md
out: list[str] = []
out.append(f"# Synthesis (pass 2): {n_success}/{n_total} model reviews — {STAMP}")
out.append("")
out.append("## Counts")
out.append(f"- Attempted: {n_total}")
out.append(f"- Succeeded: {n_success}")
out.append(f"- Failed:    {n_total - n_success}")
out.append("")
out.append("## Falcon vote (after the existing PAUSE)")
out.append(f"- PAUSED:  {falcon_votes.get('PAUSED', 0)}")
out.append(f"- RESUME:  {falcon_votes.get('RESUME', 0)}")
out.append(f"- Unparsed: {n_success - sum(falcon_votes.values())}")
if defectors:
    out.append("")
    out.append("### Models voting RESUME (override the 20/0 consensus)")
    for m, s in defectors:
        out.append(f"- **{m}**: {s}")
out.append("")
out.append("## Top 5 regime-filter consensus")
for label, cnt in top5_filters:
    out.append(f"- {label}: {cnt}")
out.append("")
out.append("## Architecture rating")
if arch_only_vals:
    out.append(f"- Mean: {mean_arch:.2f}   Stdev: {stdev_arch:.2f}   N={len(arch_only_vals)}")
    hist = Counter(int(round(v)) for v in arch_only_vals)
    out.append("")
    out.append("Distribution (rating: count):")
    for k in range(1, 11):
        c = hist.get(k, 0)
        bar = "#" * c
        out.append(f"  {k:>2}: {c} {bar}")
else:
    out.append("- No ratings parseable.")
out.append("")
out.append("## Deploy-with-real-money vote")
out.append(f"- Yes:         {deploy_votes.get('Yes', 0)}")
out.append(f"- No:          {deploy_votes.get('No', 0)}")
out.append(f"- Conditional: {deploy_votes.get('Conditional', 0)}")
out.append(f"- Unparsed:    {n_success - sum(deploy_votes.values())}")
out.append("")
out.append("## Top 3 hidden risks")
for label, cnt in top3_risks:
    out.append(f"- {label} ({cnt} models)")
    models_list = sorted(risk_by_model[label])
    for mm in models_list[:6]:
        out.append(f"   - {mm}")
    if len(models_list) > 6:
        out.append(f"   - ...and {len(models_list)-6} more")
out.append("")
out.append("## Most contrarian / specific takes")
if contrarian_picks:
    for m, s in contrarian_picks[:3]:
        out.append(f"- **{m}**: {s}")
else:
    out.append("- (No defectors and no distinctive low ratings found.)")
out.append("")

(OUT_DIR / "SYNTHESIS.md").write_text("\n".join(out))

# Persist short report as JSON
summary = {
    "stamp": STAMP,
    "out_dir": str(OUT_DIR),
    "n_attempted": n_total,
    "n_success": n_success,
    "falcon": dict(falcon_votes),
    "deploy": dict(deploy_votes),
    "arch_mean": mean_arch,
    "arch_stdev": stdev_arch,
    "arch_n": len(arch_only_vals),
    "top_filters": top5_filters,
    "top_risks": top3_risks,
    "defectors": defectors,
    "contrarian": contrarian_picks,
}
(OUT_DIR / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))

print()
print("================== REPORT ==================")
print(f"Stamp path     : {OUT_DIR}")
print(f"Success        : {n_success}/{n_total}")
print(
    f"Falcon vote    : PAUSED={falcon_votes.get('PAUSED', 0)} RESUME={falcon_votes.get('RESUME', 0)} "
    f"Unparsed={n_success - sum(falcon_votes.values())}"
)
print(f"Top filter     : {top5_filters[0][0] if top5_filters else 'n/a'}")
print(f"Mean arch rate : {mean_arch:.2f} (sd {stdev_arch:.2f}, n={len(arch_only_vals)})")
print(
    f"Real-money     : Yes={deploy_votes.get('Yes', 0)} No={deploy_votes.get('No', 0)} "
    f"Conditional={deploy_votes.get('Conditional', 0)} "
    f"Unparsed={n_success - sum(deploy_votes.values())}"
)
print("============================================")
print(f"[done] wrote SUMMARY.json + SYNTHESIS.md + {n_total} reviews to {OUT_DIR}")
