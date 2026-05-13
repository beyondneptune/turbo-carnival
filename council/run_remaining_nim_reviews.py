"""Finish the NIM model review sweep started by the agent.

Reads candidates.json from council/reviews/20260511T194931Z/, skips models that
already have a .md file in either review directory, then sends the same review
prompt to every remaining model in parallel. Writes synthesis at the end.
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

REPO_ROOT = Path("/Users/weeoto/turbo-carnival")
ENV_PATH = REPO_ROOT / "council" / ".env.local"
COHORT_DIR = REPO_ROOT / "council" / "reviews" / "20260511T191453Z"
NEW_DIR = REPO_ROOT / "council" / "reviews" / "20260511T194931Z"
CANDIDATES = NEW_DIR / "candidates.json"

SYSTEM = "You output direct, concise analysis. No preamble."
USER_PROMPT = """You're reviewing a LIVE paper-options + HFT system. Be brutally honest. This is paper money, your harshness will not harm anyone.

3 Alpaca paper accounts ($100k each, today):
- Aegis (-$709 today): HFT mean-reversion, [SPY,QQQ,NVDA,TSLA,AAPL,META,AMD]. 25 bps signal, 100 bps TP, 35 bps SL, 1200s hold. Force-flat at end of day.
- Falcon (-$2,290 today): HFT breakout, same universe. 18 bps signal, 70 bps TP, 25 bps SL, 900s hold. Whipsawed badly all afternoon - entries hit SL within 2-10 seconds repeatedly. Force-flat at end of day.
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

def load_env() -> None:
    for line in ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k and k not in os.environ:
            os.environ[k] = v.strip().strip('"').strip("'")


def slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def already_done(model_id: str) -> bool:
    s = slug(model_id)
    for d in (COHORT_DIR, NEW_DIR):
        p = d / f"{s}.md"
        if p.exists():
            # Check if it was a success or a 404 error
            text = p.read_text()
            if "status: success" in text:
                return True
            # If error, allow retry only if it wasn't 404 (404 won't change)
            if "404" in text or "Not Found" in text or "Function" in text:
                return True
    return False


def review_model(client: OpenAI, model_id: str) -> dict:
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": USER_PROMPT},
            ],
            temperature=0.4,
            max_tokens=1500,
            timeout=120,
        )
        text = resp.choices[0].message.content or ""
        return {
            "model": model_id,
            "elapsed_sec": round(time.time() - start, 2),
            "status": "success",
            "error": None,
            "response": text,
        }
    except Exception as exc:
        return {
            "model": model_id,
            "elapsed_sec": round(time.time() - start, 2),
            "status": "error",
            "error": f"{type(exc).__name__}: {str(exc)[:300]}",
            "response": "",
        }


def write_result(r: dict) -> None:
    out = NEW_DIR / f"{slug(r['model'])}.md"
    frontmatter = (
        "---\n"
        f"model: {r['model']}\n"
        f"elapsed_sec: {r['elapsed_sec']}\n"
        f"status: {r['status']}\n"
        f"error: {json.dumps(r['error']) if r['error'] else 'null'}\n"
        "---\n\n"
    )
    out.write_text(frontmatter + (r["response"] or ""), encoding="utf-8")


def parse_falcon_vote(text: str) -> str | None:
    t = text.upper()
    # Find the answer to question 2
    if "RESUME" in t and "PAUSED" not in t.split("RESUME")[0][-200:]:
        return "RESUME"
    if "PAUSED" in t or "PAUSE" in t:
        return "PAUSED"
    return None


def parse_rating(text: str) -> float | None:
    import re
    for m in re.finditer(r"(?:rating|score|architecture)[^\d]{0,40}(\d{1,2})(?:\s*/\s*10|\s*out of 10|\b)", text, re.I):
        try:
            v = int(m.group(1))
            if 1 <= v <= 10:
                return float(v)
        except ValueError:
            continue
    # fallback: first "N/10"
    m = re.search(r"\b(\d{1,2})\s*/\s*10\b", text)
    if m:
        v = int(m.group(1))
        if 1 <= v <= 10:
            return float(v)
    return None


def parse_deploy(text: str) -> str | None:
    t = text.lower()
    # Look near the bottom (question 5)
    bot = t[-800:]
    if "would not deploy" in bot or "do not deploy" in bot or "no, " in bot or "no." in bot.split("\n")[-1]:
        if "yes" not in bot[:120]:
            return "No"
    if "conditional" in bot or "specific conditions" in bot or "only after" in bot or "only if" in bot:
        return "Conditional"
    if "deploy" in bot and "yes" in bot[-200:]:
        return "Yes"
    if "no\n" in bot or bot.strip().endswith("no") or bot.strip().endswith("no."):
        return "No"
    return None


def parse_filter(text: str) -> set[str]:
    t = text.upper()
    out = set()
    if "ATR" in t:
        out.add("ATR")
    if "VIX" in t:
        out.add("VIX")
    if "ADX" in t:
        out.add("ADX")
    if "BOLLINGER" in t or "BBAND" in t:
        out.add("Bollinger")
    if "SMA" in t or "EMA" in t or "MOVING AVERAGE" in t:
        out.add("MA")
    if "VWAP" in t:
        out.add("VWAP")
    return out


def synth(results: list[dict]) -> str:
    success = [r for r in results if r["status"] == "success" and r.get("response", "").strip()]
    falcon_votes = {"PAUSED": 0, "RESUME": 0, "unclear": 0}
    ratings = []
    deploys = {"Yes": 0, "No": 0, "Conditional": 0, "unclear": 0}
    filter_counts: dict[str, int] = {}
    contrarian = []

    for r in success:
        v = parse_falcon_vote(r["response"])
        if v in ("PAUSED", "RESUME"):
            falcon_votes[v] += 1
        else:
            falcon_votes["unclear"] += 1
        rating = parse_rating(r["response"])
        if rating is not None:
            ratings.append(rating)
        d = parse_deploy(r["response"])
        deploys[d or "unclear"] = deploys.get(d or "unclear", 0) + 1
        for f in parse_filter(r["response"]):
            filter_counts[f] = filter_counts.get(f, 0) + 1
        if rating is not None and rating >= 6:
            contrarian.append((r["model"], rating))

    avg = statistics.mean(ratings) if ratings else 0.0
    sd = statistics.stdev(ratings) if len(ratings) > 1 else 0.0
    lines = []
    lines.append(f"# Synthesis of remaining NIM model reviews")
    lines.append("")
    lines.append(f"- Total attempted: {len(results)}")
    lines.append(f"- Successes (with parseable response): {len(success)}")
    lines.append(f"- Errors (404 / timeout / etc): {len(results) - len(success)}")
    lines.append("")
    lines.append("## Falcon vote")
    for k, v in falcon_votes.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Architecture rating")
    lines.append(f"- Mean: {avg:.2f}  Stdev: {sd:.2f}  N={len(ratings)}")
    lines.append("")
    lines.append("## Top regime filters (by mention count)")
    for f, c in sorted(filter_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {f}: {c}")
    lines.append("")
    lines.append("## Deploy with real money")
    for k, v in deploys.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    if contrarian:
        lines.append("## Contrarian / above-average raters (rating >= 6)")
        for m, r in contrarian:
            lines.append(f"- {m}: {r:.1f}/10")
    lines.append("")
    lines.append(f"_Generated at {datetime.now(timezone.utc).isoformat()}_")
    return "\n".join(lines)


def main() -> int:
    load_env()
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        print("NVIDIA_API_KEY missing", file=sys.stderr)
        return 1
    client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1", timeout=130.0)

    candidates_doc = json.loads(CANDIDATES.read_text())
    if isinstance(candidates_doc, dict):
        candidates = candidates_doc.get("candidates", [])
    else:
        candidates = candidates_doc

    # Filter to not-yet-done
    todo = [m for m in candidates if not already_done(m)]
    print(f"total candidates: {len(candidates)}", flush=True)
    print(f"already done (success or hard error): {len(candidates) - len(todo)}", flush=True)
    print(f"to review now: {len(todo)}", flush=True)

    results: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        future_to_model = {pool.submit(review_model, client, m): m for m in todo}
        done_count = 0
        for fut in concurrent.futures.as_completed(future_to_model):
            r = fut.result()
            results.append(r)
            write_result(r)
            done_count += 1
            mark = "OK" if r["status"] == "success" else "ERR"
            print(f"[{done_count}/{len(todo)}] {mark}  {r['model']}  ({r['elapsed_sec']}s)", flush=True)

    # Include all prior results too for a final synthesis
    all_results = list(results)
    for d in (COHORT_DIR, NEW_DIR):
        for p in d.glob("*.md"):
            if p.name in ("SYNTHESIS.md", "models_list_full.json"):
                continue
            text = p.read_text()
            if "status: success" not in text:
                continue
            model = ""
            for line in text.splitlines():
                if line.startswith("model:"):
                    model = line.split(":", 1)[1].strip()
                    break
            if not model or any(r["model"] == model for r in results):
                continue
            body = text.split("---", 2)[-1].strip()
            all_results.append({"model": model, "status": "success", "response": body, "error": None, "elapsed_sec": 0.0})

    (NEW_DIR / "SYNTHESIS.md").write_text(synth(results), encoding="utf-8")
    (NEW_DIR / "FULL_SYNTHESIS.md").write_text(synth(all_results), encoding="utf-8")

    print(f"\nDONE. {sum(1 for r in results if r['status']=='success')}/{len(results)} successes")
    print(f"SYNTHESIS:      {NEW_DIR/'SYNTHESIS.md'}")
    print(f"FULL_SYNTHESIS: {NEW_DIR/'FULL_SYNTHESIS.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
