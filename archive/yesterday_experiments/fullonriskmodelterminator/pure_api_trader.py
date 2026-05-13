"""Lane 3 — Pure-API trader. Maximum agentic.

Difference from Lane 2 (council):
  - Lane 2: LLM picks from constrained menu (10 actions × tiered sizing/DTE)
  - Lane 3: LLM proposes ARBITRARY position structure each day. We only enforce
    physical possibility (can't buy what cash won't allow).

LLM output is a free-form list of legs. We construct the position from that.
This is the YOLO lane — the LLM gets to write its own strategy daily.

Single-model mode by default (cheaper, faster, but more variance).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from openai import OpenAI

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_spy_core_strategy_grid import load_market_data  # noqa: E402
from fullonriskmodelterminator.council import load_env  # noqa: E402
from fullonriskmodelterminator.execution import (  # noqa: E402
    MULTIPLIER,
    OpenPosition,
    _bs_price,
    mark_position,
    trade_pnl,
)


# Use a fast, instruction-tuned model. Thinking models are too slow per-day.
DEFAULT_MODEL = "qwen/qwen3-coder-480b-a35b-instruct"


PROMPT = """You are running an aggressive paper options account on SPY.

Today: {today}
SPY: ${spy:.2f}  (5d return: {ret5:+.2%}, 20d return: {ret20:+.2%})
VIX: {vix:.2f}  (5d ROC: {vix_roc:+.2%})
ATM IV: {iv:.4f}
NAV: ${nav:,.2f}  Cash: ${cash:,.2f}  Open positions: {n_open}

You have FULL latitude. Propose any combination of SPY options legs you want.
Tomorrow you'll be asked again — manage holdings via close_legs if needed.

Constraints (physics, not policy):
  - You cannot spend more cash than available
  - Each leg is one option contract type (call or put), one strike, one expiry, long or short
  - Strikes are rounded to nearest $1
  - Expiry can be 0-60 days out
  - Total quantity across all legs of one trade must be ≤ 100 contracts

Return strict JSON only (no markdown):
{{
  "open_legs": [
    {{"right": "C|P", "strike": 530, "dte_days": 7, "direction": "long|short", "qty": 1}},
    ...
  ],
  "close_existing": false,    // true to liquidate all open positions
  "intent": "one sentence — what you're betting on"
}}

If you want to skip the day, return {{"open_legs": [], "close_existing": false, "intent": "stand aside"}}."""


def _parse(text: str) -> dict | None:
    text = (text or "").strip()
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


def _ask(client: OpenAI, model: str, prompt: str) -> dict | None:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You output strict JSON. No markdown."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=800,
            timeout=45.0,
        )
        return _parse(resp.choices[0].message.content or "")
    except Exception:
        return None


def _build_position(legs: list[dict], market: pd.Series, today: date) -> OpenPosition | None:
    if not legs:
        return None
    spot = float(market["spy_close"])
    sigma = float(market["iv"])

    valid_legs = []
    expiry_days = []
    for leg in legs:
        if not isinstance(leg, dict):
            continue
        right = str(leg.get("right", "C")).upper()[:1]
        if right not in ("C", "P"):
            continue
        try:
            strike = round(float(leg.get("strike", spot)))
            dte = max(0, min(60, int(leg.get("dte_days", 7))))
            qty = max(1, min(100, int(leg.get("qty", 1))))
        except (TypeError, ValueError):
            continue
        direction = str(leg.get("direction", "long")).lower()
        if direction not in ("long", "short"):
            direction = "long"
        valid_legs.append({"right": right, "strike": strike, "qty": qty, "direction": direction})
        expiry_days.append(dte)
    if not valid_legs:
        return None

    # Use max DTE across legs as the position expiry (proxy)
    max_dte = max(expiry_days)
    expiry = today + timedelta(days=max(max_dte, 1))
    t = max(max_dte, 1) / 365.0

    # Compute entry credit (positive = receive) and max-loss approximation
    entry_credit = 0.0
    for leg in valid_legs:
        price = _bs_price(spot, leg["strike"], t, sigma, leg["right"])
        sign = -1.0 if leg["direction"] == "long" else 1.0
        entry_credit += sign * price * leg["qty"]

    # Max loss is hard to estimate for arbitrary multi-leg structures.
    # Approximate as max of: (long premium paid) or (5% adverse SPY move impact)
    long_premium = sum(
        _bs_price(spot, leg["strike"], t, sigma, leg["right"]) * leg["qty"] * MULTIPLIER
        for leg in valid_legs if leg["direction"] == "long"
    )
    short_exposure = sum(leg["qty"] * spot * 0.05 * MULTIPLIER
                          for leg in valid_legs if leg["direction"] == "short")
    max_loss_per_contract = max(long_premium, short_exposure, 100.0)

    return OpenPosition(
        open_date=today,
        expiry=expiry,
        action="pure_api_custom",
        legs=valid_legs,
        entry_credit=entry_credit,
        max_loss_per_contract=max_loss_per_contract,
        quantity=1,  # qty is per-leg here
    )


def run_pure_api_backtest(start: str, end: str | None = None, model: str = DEFAULT_MODEL,
                           initial_capital: float = 100_000.0, max_days: int | None = None) -> dict:
    load_env()
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        raise RuntimeError("NVIDIA_API_KEY missing")
    client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1", timeout=60.0)

    market = load_market_data(
        start=start, end=end,
        hf_iv_path=REPO_ROOT / "data" / "offline_context" / "huggingface_options_iv_sp500"
                  / "processed" / "data_IV_USA.parquet",
    )
    print(f"[pure_api] Loaded {len(market)} days, model={model}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "fullonriskmodelterminator" / "runs" / f"pure_api_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    nav = initial_capital
    cash = initial_capital
    open_positions: list[OpenPosition] = []
    nav_history = []
    trades = []
    daily_decisions = []
    days_processed = 0

    for i, (idx, row) in enumerate(market.iterrows()):
        if max_days is not None and days_processed >= max_days:
            break
        today = pd.Timestamp(idx).date()

        # Mark / expire existing
        kept = []
        for p in open_positions:
            if today >= p.expiry:
                close_val = mark_position(p, row)
                pnl = trade_pnl(p, close_val, "expiry")
                cash += pnl - (p.entry_credit * MULTIPLIER * p.quantity)
                trades.append({
                    "open_date": p.open_date.isoformat(),
                    "close_date": today.isoformat(),
                    "n_legs": len(p.legs),
                    "entry_credit": p.entry_credit,
                    "pnl_usd": pnl,
                    "close_reason": "expiry",
                })
            else:
                kept.append(p)
        open_positions = kept

        unrealized = sum(mark_position(p, row) for p in open_positions)
        nav = cash + unrealized

        # Prompt LLM
        prev5 = market.iloc[i - 5] if i >= 5 else None
        prev20 = market.iloc[i - 20] if i >= 20 else None
        prompt = PROMPT.format(
            today=today.isoformat(),
            spy=float(row["spy_close"]),
            ret5=(row["spy_close"] / prev5["spy_close"] - 1) if prev5 is not None else 0.0,
            ret20=(row["spy_close"] / prev20["spy_close"] - 1) if prev20 is not None else 0.0,
            vix=float(row.get("vix_close", 20.0)),
            vix_roc=float(row.get("vix_5d_roc", 0.0) or 0.0),
            iv=float(row["iv"]),
            nav=nav, cash=cash, n_open=len(open_positions),
        )
        decision = _ask(client, model, prompt) or {"open_legs": [], "close_existing": False, "intent": "API failed"}
        daily_decisions.append({
            "date": today.isoformat(),
            "decision": decision,
            "nav": nav,
        })

        if decision.get("close_existing"):
            for p in open_positions:
                close_val = mark_position(p, row)
                pnl = trade_pnl(p, close_val, "llm_close")
                cash += pnl - (p.entry_credit * MULTIPLIER * p.quantity)
                trades.append({
                    "open_date": p.open_date.isoformat(),
                    "close_date": today.isoformat(),
                    "n_legs": len(p.legs),
                    "entry_credit": p.entry_credit,
                    "pnl_usd": pnl,
                    "close_reason": "llm_close",
                })
            open_positions = []

        legs = decision.get("open_legs") or []
        if legs:
            pos = _build_position(legs, row, today)
            if pos is not None:
                entry_cash = pos.entry_credit * MULTIPLIER * pos.quantity
                # Block if we'd go negative
                if cash + entry_cash >= 0:
                    open_positions.append(pos)
                    cash += entry_cash

        unrealized = sum(mark_position(p, row) for p in open_positions)
        nav = cash + unrealized
        nav_history.append({"date": today.isoformat(), "nav": nav, "cash": cash,
                             "unrealized": unrealized, "n_open": len(open_positions),
                             "intent": (decision.get("intent") or "")[:120]})

        if days_processed % 3 == 0:
            print(f"  {today} | NAV ${nav:,.0f} | open {len(open_positions)} | "
                  f"intent: {(decision.get('intent') or '')[:60]}", flush=True)
        days_processed += 1
        if nav <= 0:
            print(f"  *** ACCOUNT LIQUIDATED on {today}")
            break

    nav_df = pd.DataFrame(nav_history)
    trades_df = pd.DataFrame(trades)
    decisions_df = pd.DataFrame(daily_decisions)
    nav_df.to_csv(out_dir / "nav_history.csv", index=False)
    trades_df.to_csv(out_dir / "trades.csv", index=False)
    decisions_df.to_csv(out_dir / "decisions.csv", index=False)

    final_nav = float(nav_df["nav"].iloc[-1]) if not nav_df.empty else initial_capital
    total_return = (final_nav / initial_capital - 1) * 100
    years = days_processed / 252.0
    monthly = ((final_nav / initial_capital) ** (1 / max(years * 12, 0.1)) - 1) * 100 if final_nav > 0 else -100
    peak = nav_df["nav"].cummax()
    dd = (nav_df["nav"] - peak) / np.maximum(peak, 1.0)
    max_dd = float(dd.min() * 100) if not dd.empty else 0.0
    win_rate = float((trades_df["pnl_usd"] > 0).mean() * 100) if not trades_df.empty else 0.0

    headline = {
        "stamp": stamp,
        "model": model,
        "final_nav": final_nav,
        "total_return_pct": total_return,
        "monthly_pct": monthly,
        "max_dd_pct": max_dd,
        "trades_placed": int(len(trades_df)),
        "win_rate_pct": win_rate,
        "days": days_processed,
        "output_dir": str(out_dir),
    }
    (out_dir / "headline.json").write_text(json.dumps(headline, indent=2, default=str))
    print("\n=== PURE-API HEADLINE ===")
    for k, v in headline.items():
        print(f"  {k}: {v}")
    return headline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-05-07")
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()

    run_pure_api_backtest(args.start, args.end, args.model, args.capital, args.max_days)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
