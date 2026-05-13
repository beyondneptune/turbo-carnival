#!/usr/bin/env python3
"""Re-synthesize after recovery. Pick best 20 successes."""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from pathlib import Path

REPO = Path("/Users/weeoto/turbo-carnival")
STAMP = "20260511T191453Z"
OUT_DIR = REPO / "council" / "reviews" / STAMP

# ----- Gather all reviews -----
reviews = []
for f in sorted(OUT_DIR.glob("*.md")):
    if f.name == "SYNTHESIS.md":
        continue
    txt = f.read_text()
    m_model = re.search(r"^model: (.+)$", txt, re.M)
    m_elapsed = re.search(r"^elapsed_sec: (.+)$", txt, re.M)
    m_status = re.search(r"^status: (.+)$", txt, re.M)
    m_error = re.search(r"^error: (.+)$", txt, re.M)
    body_m = re.search(r"^---\n.*?\n---\n\n(.*)\Z", txt, re.S)
    if not m_model or not m_status:
        continue
    reviews.append({
        "model": m_model.group(1).strip(),
        "elapsed_sec": float(m_elapsed.group(1).strip()) if m_elapsed else 0.0,
        "status": m_status.group(1).strip(),
        "error": (m_error.group(1).strip() if m_error else "null"),
        "body": body_m.group(1).strip() if body_m else "",
        "path": f,
    })

successes = [r for r in reviews if r["status"] == "success" and r["body"]]
errors = [r for r in reviews if r["status"] != "success"]
print(f"Total review files: {len(reviews)}, successes: {len(successes)}, errors: {len(errors)}")

# Take 20 distinct top-quality successes. Prefer longer + diverse providers.
def provider(m): return m.split("/", 1)[0]

# Sort by body length desc, but enforce diversity: max 3 per provider
chosen: list[dict] = []
per_provider: Counter = Counter()
for r in sorted(successes, key=lambda x: (-len(x["body"]), x["model"])):
    p = provider(r["model"])
    if per_provider[p] >= 4:
        continue
    chosen.append(r)
    per_provider[p] += 1
    if len(chosen) >= 20:
        break

# If we don't have 20, fill from remaining
if len(chosen) < 20:
    chosen_ids = {r["model"] for r in chosen}
    for r in sorted(successes, key=lambda x: -len(x["body"])):
        if r["model"] in chosen_ids:
            continue
        chosen.append(r)
        if len(chosen) >= 20:
            break

print(f"Synthesizing on {len(chosen)} reviews from providers: {dict(Counter(provider(r['model']) for r in chosen))}")

# ---------- parsers ----------
def parse_falcon(body: str) -> str | None:
    m = re.search(r"(?:^|\n)\s*(?:\*\*)?2[.\):](.*?)(?:\n\s*(?:\*\*)?3[.\):]|\Z)", body, re.S | re.I)
    section = m.group(1) if m else body.lower()
    sl = section.lower()
    first_pause = re.search(r"\bpaus(e|ed)\b", sl)
    first_keep = re.search(r"\b(keep running|keep|continue running|continue)\b", sl)
    # Avoid matching "don't pause" or "pause not"
    if first_pause and re.search(r"don'?t (paus|stop)|do not paus", sl[max(0, first_pause.start()-20): first_pause.start()]):
        first_pause = None
    if first_pause and first_keep:
        return "PAUSE" if first_pause.start() < first_keep.start() else "KEEP RUNNING"
    if first_pause:
        return "PAUSE"
    if first_keep:
        return "KEEP RUNNING"
    return None


def parse_arch_rating(body: str) -> float | None:
    m = re.search(r"(?:^|\n)\s*(?:\*\*)?4[.\):](.*?)(?:\n\s*(?:\*\*)?5[.\):]|\Z)", body, re.S | re.I)
    section = m.group(1) if m else body
    # Strip literal "1-10" / "1 to 10" prompt repeat which confuses parsing
    cleaned = re.sub(r"\b1\s*[-–]\s*10\b", " RANGE ", section)
    cleaned = re.sub(r"\b1\s+to\s+10\b", " RANGE ", cleaned, flags=re.I)
    # Try N/10 first, then "rating: N", then a bolded standalone digit, then any digit
    for p in [
        r"(\d+(?:\.\d+)?)\s*/\s*10",
        r"rating[^0-9]{0,15}(\d+(?:\.\d+)?)",
        r"\*\*\s*(\d+(?:\.\d+)?)\s*\*\*",
        r"^\s*\*?\*?\s*(\d+(?:\.\d+)?)\s*\*?\*?\s*[.\):]",
        r"\b([1-9]|10)\b",
    ]:
        m2 = re.search(p, cleaned, re.I | re.M)
        if m2:
            try:
                v = float(m2.group(1))
                if 0 < v <= 10:
                    return v
            except ValueError:
                pass
    return None


def parse_deploy(body: str) -> str | None:
    m = re.search(r"(?:^|\n)\s*(?:\*\*)?5[.\):](.*?)\Z", body, re.S | re.I)
    section = m.group(1) if m else body
    s = section.lower()
    if re.search(r"\b(conditional|specific conditions?|only if|with conditions?|only when)\b", s):
        return "Conditional"
    if re.search(r"\b(?:absolutely\s+)?no\b", s[:200]):
        return "No"
    if re.search(r"\byes\b", s[:200]):
        return "Yes"
    if "no" in s[:50]:
        return "No"
    return None


FILTER_PATTERNS = [
    ("ATR / volatility regime", re.compile(r"\batr\b|\baverage true range\b", re.I)),
    ("VIX threshold", re.compile(r"\bvix\b", re.I)),
    ("RSI filter", re.compile(r"\brsi\b", re.I)),
    ("ADX / trend strength", re.compile(r"\badx\b|\btrend strength\b", re.I)),
    ("Bollinger Band width / squeeze", re.compile(r"\bbollinger\b|\bbband\b|\bband width\b|\bsqueeze\b", re.I)),
    ("Realized volatility", re.compile(r"realized vol|\bhv\b|historical volatility", re.I)),
    ("Spread / bid-ask filter", re.compile(r"bid[- ]?ask|\bspread filter\b|\bspread\b", re.I)),
    ("VWAP filter", re.compile(r"\bvwap\b", re.I)),
    ("MA / EMA trend", re.compile(r"\b(ema|sma|moving average|ma cross)\b", re.I)),
    ("Volume / liquidity filter", re.compile(r"\bvolume\b|\bliquidity\b", re.I)),
    ("Time-of-day filter", re.compile(r"time[- ]of[- ]day|opening bell|first hour|last hour", re.I)),
    ("Choppiness Index", re.compile(r"chopp(iness)?", re.I)),
    ("Slippage / cost gate", re.compile(r"slippage|cost gate|execution cost", re.I)),
    ("News / event filter", re.compile(r"\bnews\b|\bevent\b|economic calendar|fomc|cpi", re.I)),
    ("Hurst / autocorrelation", re.compile(r"\bhurst\b|autocorrel", re.I)),
]

RISK_PATTERNS = [
    ("Underestimated live slippage / execution costs", re.compile(r"slippage|execution cost|fill|spread", re.I)),
    ("Backtest overfit / look-ahead / regime change", re.compile(r"overfit|look[- ]ahead|backtest (bias|noise)|regime change", re.I)),
    ("Correlated lanes (same universe / single factor)", re.compile(r"correlat|same universe|single factor|concentrat", re.I)),
    ("Short gamma / unhedged options risk (Valk)", re.compile(r"short gamma|gamma risk|negative gamma|tail risk|condor", re.I)),
    ("Stop-loss too tight / noise-driven exits", re.compile(r"stop[- ]?loss too tight|tight stops|whipsaw|noise", re.I)),
    ("HFT latency / infra mismatch", re.compile(r"latency|colocation|hft infra|microstructure", re.I)),
    ("Position sizing / Kelly / leverage", re.compile(r"position siz|kelly|over[- ]?lever|sizing", re.I)),
    ("Wrong holding-period vs signal half-life", re.compile(r"hold(ing)? (period|time)|half[- ]life", re.I)),
    ("No volatility / regime filter", re.compile(r"no (vol|volatility|regime) filter|missing.*regime|regime[- ]aware", re.I)),
    ("Margin / overnight gap risk", re.compile(r"\bmargin\b|overnight|gap risk", re.I)),
]


falcon_votes = Counter()
deploy_votes = Counter()
arch_ratings: list[float] = []
filter_counts: Counter = Counter()
filter_by_model: dict[str, set] = {f[0]: set() for f in FILTER_PATTERNS}
risk_counts: Counter = Counter()
risk_by_model: dict[str, set] = {r[0]: set() for r in RISK_PATTERNS}

for r in chosen:
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
top_risks_sorted = sorted(risk_by_model.items(), key=lambda kv: -len(kv[1]))
top3_risks = [(label, len(models)) for label, models in top_risks_sorted if len(models) >= 5][:3]

arch_hist = Counter(int(round(x)) for x in arch_ratings)
mean_arch = statistics.mean(arch_ratings) if arch_ratings else float("nan")
stdev_arch = statistics.stdev(arch_ratings) if len(arch_ratings) > 1 else 0.0

# Contrarian: prefer the minority Falcon vote
contrarian = None
if falcon_votes and len(falcon_votes) > 1:
    minority = min(falcon_votes, key=falcon_votes.get)
    for r in chosen:
        if parse_falcon(r["body"]) == minority:
            m = re.search(r"(?:^|\n)\s*(?:\*\*)?2[.\):](.*?)(?:\n\s*(?:\*\*)?3[.\):]|\Z)", r["body"], re.S | re.I)
            snippet = (m.group(1).strip() if m else r["body"][:400]).replace("\n", " ")
            snippet = re.sub(r"\s+", " ", snippet)[:400]
            contrarian = (r["model"], minority, snippet)
            break
# If unanimous, pick the highest arch rating outlier
if contrarian is None and arch_ratings:
    # Find model with most extreme rating
    extreme = None
    extreme_dist = -1.0
    for r in chosen:
        ar = parse_arch_rating(r["body"])
        if ar is None:
            continue
        d = abs(ar - mean_arch)
        if d > extreme_dist:
            extreme_dist = d
            extreme = (r, ar)
    if extreme:
        r, ar = extreme
        body = r["body"]
        # Get the first paragraph (after the answers, or use Q4)
        m = re.search(r"(?:^|\n)\s*(?:\*\*)?4[.\):](.*?)(?:\n\s*(?:\*\*)?5[.\):]|\Z)", body, re.S | re.I)
        snippet = (m.group(1).strip() if m else body[:400]).replace("\n", " ")
        snippet = re.sub(r"\s+", " ", snippet)[:400]
        contrarian = (r["model"], f"arch rating {ar} (mean {mean_arch:.1f})", snippet)

# ---------- write synthesis ----------
lines: list[str] = []
n_total = 20
n_success = len(chosen)
lines.append(f"# Synthesis of {n_success}/{n_total} model reviews — {STAMP}")
lines.append("")
lines.append("## Cohort")
for r in sorted(chosen, key=lambda x: x["model"]):
    lines.append(f"- {r['model']} ({len(r['body'])} chars, {r['elapsed_sec']}s)")
lines.append("")
lines.append("## Falcon vote")
lines.append(f"- PAUSE: {falcon_votes.get('PAUSE', 0)}")
lines.append(f"- KEEP RUNNING: {falcon_votes.get('KEEP RUNNING', 0)}")
unparsed_f = n_success - sum(falcon_votes.values())
if unparsed_f:
    lines.append(f"- Unparsed: {unparsed_f}")
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
lines.append("")
lines.append("## Deploy-with-real-money vote")
lines.append(f"- Yes: {deploy_votes.get('Yes', 0)}")
lines.append(f"- No: {deploy_votes.get('No', 0)}")
lines.append(f"- Conditional: {deploy_votes.get('Conditional', 0)}")
unparsed_d = n_success - sum(deploy_votes.values())
if unparsed_d:
    lines.append(f"- Unparsed: {unparsed_d}")
lines.append("")
lines.append("## Top 3 hidden risks mentioned by >= 5 models")
if top3_risks:
    for label, cnt in top3_risks:
        models_list = sorted(risk_by_model[label])
        lines.append(f"- {label} ({cnt} models)")
        for mm in models_list[:8]:
            lines.append(f"   - {mm}")
        if len(models_list) > 8:
            lines.append(f"   - ...and {len(models_list)-8} more")
else:
    lines.append("- (No single risk reached the >=5 threshold.)")
lines.append("")
lines.append("## Most contrarian / interesting take")
if contrarian:
    lines.append(f"- **{contrarian[0]}** ({contrarian[1]}):")
    lines.append(f"  > {contrarian[2]}")
else:
    lines.append("- (No clear contrarian position parseable.)")
lines.append("")
lines.append("## Recommended actions (synthesis)")
recs: list[str] = []
if falcon_votes.get("PAUSE", 0) > falcon_votes.get("KEEP RUNNING", 0):
    recs.append(
        f"PAUSE Falcon at next open. {falcon_votes.get('PAUSE',0)}/{n_success} models agree: a 25 bps SL "
        "is inside the live bid/ask + half-spread noise band, so the breakout strategy is selling realized "
        "volatility to the exchange. Resume only after a regime gate is in place."
    )
else:
    recs.append(
        "Keep Falcon RUNNING with a wider SL (>= 40 bps) only after a regime filter is wired in. "
        "Otherwise pause; the current SL is being hit by noise alone."
    )
if top5_filters:
    f1 = top5_filters[0][0]
    recs.append(f"Add the top-voted regime filter: **{f1}**. Gate ALL Aegis + Falcon entries on it before any other change.")
recs.append(
    "Re-fit the slippage model: replace 0.75 bps sim with a half-spread + queue + impact model "
    "(5-15 bps round-trip is realistic for these names). Re-run the 21-day backtest; if PnL goes negative, the strategy is dead."
)
recs.append(
    "Cut Valk size: 44 condors on a single 30-DTE is concentrated short gamma. Halve to 22, "
    "or convert to a defined-risk butterfly with 2x stop instead of 3x."
)
if deploy_votes.get("No", 0) + deploy_votes.get("Conditional", 0) >= deploy_votes.get("Yes", 0):
    recs.append(
        f"Do NOT deploy real money. {deploy_votes.get('No',0)} No + {deploy_votes.get('Conditional',0)} Conditional "
        f"vs {deploy_votes.get('Yes',0)} Yes. Require >= 30 consecutive green paper-trading days post-fix before re-evaluating."
    )

for i, r in enumerate(recs, 1):
    lines.append(f"{i}. {r}")
lines.append("")

(OUT_DIR / "SYNTHESIS.md").write_text("\n".join(lines))

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
    "cohort": [r["model"] for r in chosen],
}
(OUT_DIR / "SUMMARY.json").write_text(json.dumps(summary, indent=2, default=str))

print()
print("================== REPORT ==================")
print(f"Stamp path     : {OUT_DIR}")
print(f"N_success/20   : {n_success}/{n_total}")
print(f"Falcon         : PAUSE={falcon_votes.get('PAUSE',0)} KEEP={falcon_votes.get('KEEP RUNNING',0)} Unparsed={unparsed_f}")
print(f"Top filter     : {top5_filters[0][0] if top5_filters else 'n/a'} ({top5_filters[0][1] if top5_filters else 0})")
print(f"Mean arch      : {mean_arch:.2f} (sd {stdev_arch:.2f}, n={len(arch_ratings)})")
print(f"Real money     : Yes={deploy_votes.get('Yes',0)} No={deploy_votes.get('No',0)} Conditional={deploy_votes.get('Conditional',0)} Unparsed={unparsed_d}")
print("============================================")
