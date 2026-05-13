"""Backtest the council-driven aggressive trader on Year 2 data.

Loops day-by-day, polls the NIM council, executes the voted decision,
marks open positions, computes daily NAV / drawdown / trade stats.

Caching: NIM calls are slow. We cache (date, market_state_hash) → vote so
re-running is fast. Cache lives at fullonriskmodelterminator/runs/<id>/cache.json.

Run:
    .venv/bin/python fullonriskmodelterminator/backtest.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from experiments.run_spy_core_strategy_grid import load_market_data  # noqa: E402

from fullonriskmodelterminator.council import vote, load_env  # noqa: E402
from fullonriskmodelterminator.execution import (  # noqa: E402
    OpenPosition,
    construct_position,
    mark_position,
    trade_pnl,
)
from fullonriskmodelterminator.menu import TradeDecision, validate  # noqa: E402


DEFAULT_INITIAL_CAPITAL = 100_000.0


def market_state_for(row: pd.Series, prev_5d: pd.Series, prev_20d: pd.Series,
                      nav: float, cash: float, open_positions: list[OpenPosition],
                      iv_rank: float, nav_trend: str) -> dict:
    spy = float(row["spy_close"])
    spy_5d_ret = float(row["spy_close"] / prev_5d["spy_close"] - 1) if prev_5d is not None else 0.0
    spy_20d_ret = float(row["spy_close"] / prev_20d["spy_close"] - 1) if prev_20d is not None else 0.0
    n_open = len(open_positions)
    total_risk = sum(p.max_loss_per_contract * p.quantity for p in open_positions)
    return {
        "today": pd.Timestamp(row.name).date().isoformat(),
        "spy_price": spy,
        "spy_5d_ret": spy_5d_ret,
        "spy_20d_ret": spy_20d_ret,
        "vix": float(row.get("vix_close", 20.0)),
        "vix_5d_roc": float(row.get("vix_5d_roc", 0.0) or 0.0),
        "iv": float(row["iv"]),
        "iv_rank": float(iv_rank or 0.0),
        "rv20": float(row.get("rv20", 0.15) or 0.15),
        "vrp": float(row.get("vrp20", 0.0) or 0.0),
        "nav": nav,
        "cash": cash,
        "n_open": n_open,
        "total_risk": total_risk,
        "pct_at_risk": total_risk / max(nav, 1.0),
        "nav_trend": nav_trend,
    }


def _cache_key(market_state: dict) -> str:
    """Stable hash of inputs that determine the LLM decision."""
    keys = ["today", "spy_price", "vix", "vix_5d_roc", "iv", "iv_rank",
             "rv20", "vrp", "n_open", "pct_at_risk", "nav_trend"]
    payload = {k: round(market_state[k], 4) if isinstance(market_state[k], float) else market_state[k]
                for k in keys}
    return hashlib.sha1(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def run_backtest(start: str, end: str | None, initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                  use_cache: bool = True, max_days: int | None = None,
                  out_root: Path | None = None) -> dict:
    load_env()

    market = load_market_data(
        start=start,
        end=end,
        hf_iv_path=REPO_ROOT / "data" / "offline_context" / "huggingface_options_iv_sp500"
                  / "processed" / "data_IV_USA.parquet",
    )
    print(f"Loaded {len(market)} trading days")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = (out_root or REPO_ROOT / "fullonriskmodelterminator" / "runs") / f"council_backtest_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "vote_cache.json"
    cache: dict[str, dict] = {}
    if use_cache and cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except Exception:
            cache = {}

    nav = initial_capital
    cash = initial_capital  # all cash to start
    open_positions: list[OpenPosition] = []
    nav_history: list[dict] = []
    trades: list[dict] = []
    daily_decisions: list[dict] = []

    iv_rank_history: list[float] = []
    days_processed = 0

    for i, (idx, row) in enumerate(market.iterrows()):
        if max_days is not None and days_processed >= max_days:
            break

        today = pd.Timestamp(idx).date()

        # IV rank: percentile of current IV vs last 252 days
        iv_window = market["iv"].iloc[max(0, i - 252):i + 1]
        iv_rank = (iv_window <= row["iv"]).mean() if len(iv_window) > 30 else 0.5
        iv_rank_history.append(float(iv_rank))

        # Mark / close existing positions
        positions_remaining = []
        for p in open_positions:
            if today >= p.expiry:
                # Expire
                close_val = mark_position(p, row)
                pnl = trade_pnl(p, close_val, "expiry")
                cash += (pnl - (p.entry_credit * 100 * p.quantity))  # release entry cash
                nav_realized_change = pnl
                trades.append({
                    "open_date": p.open_date.isoformat(),
                    "close_date": today.isoformat(),
                    "action": p.action,
                    "qty": p.quantity,
                    "entry_credit": p.entry_credit,
                    "pnl_usd": pnl,
                    "close_reason": "expiry",
                })
            else:
                positions_remaining.append(p)
        open_positions = positions_remaining

        # Recompute NAV
        unrealized = sum(mark_position(p, row) for p in open_positions)
        nav = cash + unrealized

        # Build market state + nav_trend
        prev_5d = market.iloc[i - 5] if i >= 5 else None
        prev_20d = market.iloc[i - 20] if i >= 20 else None
        recent_navs = [h["nav"] for h in nav_history[-10:]]
        if len(recent_navs) >= 5:
            nav_trend = "rising" if recent_navs[-1] > recent_navs[0] else "falling"
        else:
            nav_trend = "neutral"
        state = market_state_for(row, prev_5d, prev_20d, nav, cash, open_positions,
                                  iv_rank, nav_trend)

        # Council vote (cached)
        ck = _cache_key(state)
        if use_cache and ck in cache:
            decision = validate(cache[ck]["decision"])
            raw_votes = cache[ck]["raw_votes"]
        else:
            decision, raw_votes = vote(state)
            cache[ck] = {"decision": decision.to_dict(), "raw_votes": raw_votes}
            # Persist cache every 10 days
            if days_processed % 10 == 0:
                cache_path.write_text(json.dumps(cache, indent=2, default=str))

        daily_decisions.append({
            "date": today.isoformat(),
            "decision": decision.to_dict(),
            "market_state": state,
            "raw_votes_n_ok": sum(1 for v in raw_votes if v.get("decision") is not None),
        })

        # Execute decision
        if decision.action == "close_all":
            for p in open_positions:
                close_val = mark_position(p, row)
                pnl = trade_pnl(p, close_val, "council_close_all")
                cash += pnl - (p.entry_credit * 100 * p.quantity)
                trades.append({
                    "open_date": p.open_date.isoformat(),
                    "close_date": today.isoformat(),
                    "action": p.action,
                    "qty": p.quantity,
                    "entry_credit": p.entry_credit,
                    "pnl_usd": pnl,
                    "close_reason": "council_close_all",
                })
            open_positions = []
        elif decision.action != "no_trade":
            pos = construct_position(decision, row, nav)
            if pos is not None and pos.quantity >= 1:
                # Pay/receive entry cash
                entry_cash = pos.entry_credit * 100 * pos.quantity
                if cash + entry_cash >= 0:  # we have enough cash to pay debit if applicable
                    open_positions.append(pos)
                    cash += entry_cash

        # Recompute NAV after trades
        unrealized = sum(mark_position(p, row) for p in open_positions)
        nav = cash + unrealized
        nav_history.append({"date": today.isoformat(), "nav": nav, "cash": cash,
                             "unrealized": unrealized, "n_open": len(open_positions),
                             "action": decision.action, "size": decision.size_tier})

        if days_processed % 5 == 0:
            print(f"  {today} | NAV ${nav:,.0f} | open {len(open_positions)} | "
                  f"action {decision.action} ({decision.size_tier})", flush=True)
        days_processed += 1
        if nav <= 0:
            print(f"  *** ACCOUNT LIQUIDATED on {today} (NAV ≤ 0). Halting.", flush=True)
            break

    # Persist final cache
    cache_path.write_text(json.dumps(cache, indent=2, default=str))

    # Close remaining positions at last market
    last_row = market.iloc[-1]
    for p in open_positions:
        close_val = mark_position(p, last_row)
        pnl = trade_pnl(p, close_val, "end_of_backtest")
        trades.append({
            "open_date": p.open_date.isoformat(),
            "close_date": pd.Timestamp(last_row.name).date().isoformat(),
            "action": p.action,
            "qty": p.quantity,
            "entry_credit": p.entry_credit,
            "pnl_usd": pnl,
            "close_reason": "end_of_backtest",
        })

    nav_df = pd.DataFrame(nav_history)
    trades_df = pd.DataFrame(trades)
    decisions_df = pd.DataFrame(daily_decisions)

    nav_df.to_csv(out_dir / "nav_history.csv", index=False)
    trades_df.to_csv(out_dir / "trades.csv", index=False)
    decisions_df.to_csv(out_dir / "decisions.csv", index=False)

    # Headline metrics
    final_nav = float(nav_df["nav"].iloc[-1]) if not nav_df.empty else initial_capital
    total_return = (final_nav / initial_capital - 1) * 100
    years = days_processed / 252.0
    cagr = (final_nav / initial_capital) ** (1 / max(years, 1 / 252)) - 1 if final_nav > 0 else -1
    monthly = (1 + cagr) ** (1 / 12) - 1 if final_nav > 0 else -1

    peak = nav_df["nav"].cummax()
    dd = (nav_df["nav"] - peak) / np.maximum(peak, 1.0)
    max_dd = float(dd.min() * 100) if not dd.empty else 0.0

    if not trades_df.empty:
        win_rate = float((trades_df["pnl_usd"] > 0).mean() * 100)
        action_dist = trades_df["action"].value_counts().to_dict()
    else:
        win_rate = 0.0
        action_dist = {}

    headline = {
        "stamp": stamp,
        "initial_capital": initial_capital,
        "final_nav": final_nav,
        "total_return_pct": total_return,
        "cagr_pct": cagr * 100,
        "monthly_pct": monthly * 100,
        "max_dd_pct": max_dd,
        "trades_placed": int(len(trades_df)),
        "win_rate_pct": win_rate,
        "days_traded": days_processed,
        "action_distribution": action_dist,
        "output_dir": str(out_dir),
    }
    (out_dir / "headline.json").write_text(json.dumps(headline, indent=2, default=str))

    print("\n=== HEADLINE ===")
    for k, v in headline.items():
        print(f"  {k}: {v}")
    print(f"\nOutput: {out_dir}")
    return headline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-05-07",
                        help="Backtest start date (Year 2 start by default)")
    parser.add_argument("--end", default=None,
                        help="Backtest end date (today if omitted)")
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL)
    parser.add_argument("--max-days", type=int, default=None,
                        help="Cap iterations (useful for fast smoke tests)")
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    run_backtest(
        start=args.start,
        end=args.end,
        initial_capital=args.capital,
        max_days=args.max_days,
        use_cache=not args.no_cache,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
