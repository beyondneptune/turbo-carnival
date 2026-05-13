"""Realistic execution simulator + position sizing.

Takes the council's proxy backtest output (trades.csv) and rebuilds the P&L
under realistic assumptions:

  * bid/ask spread per leg
  * commission per contract
  * slippage on stop-out (market orders cross the spread)
  * position sizing to a target % of NAV per trade
  * compounding equity curve

The proxy backtest assumed mid-price fills with 1 contract per trade. That
silently makes any high-frequency strategy look better than reality. This
script gives an honest range (optimistic / realistic / pessimistic) plus
a $100k IBKR-NAV equity curve.

Output:
  experiments/runs/<run_dir>/realistic_<strategy_name>.json
  experiments/runs/<run_dir>/realistic_<strategy_name>.md
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FrictionScenario:
    """All values in $ per leg per side unless otherwise noted."""
    name: str
    spread_per_leg_usd: float            # e.g. 0.03 = $0.03 bid/ask spread on the option
    entry_slippage_frac: float           # fraction of half-spread paid on entry (limit order)
    exit_limit_slippage_frac: float      # fraction of half-spread paid on profit_target / dte_close
    exit_market_slippage_frac: float     # fraction of half-spread paid on stop_loss / end_forced
    commission_per_contract_usd: float   # per contract per leg per side


SCENARIOS = [
    FrictionScenario(
        name="optimistic_$0.02_spread",
        spread_per_leg_usd=0.02,
        entry_slippage_frac=0.10,
        exit_limit_slippage_frac=0.10,
        exit_market_slippage_frac=0.40,
        commission_per_contract_usd=0.50,
    ),
    FrictionScenario(
        name="realistic_$0.03_spread",
        spread_per_leg_usd=0.03,
        entry_slippage_frac=0.25,
        exit_limit_slippage_frac=0.20,
        exit_market_slippage_frac=0.55,
        commission_per_contract_usd=0.65,
    ),
    FrictionScenario(
        name="pessimistic_$0.05_spread",
        spread_per_leg_usd=0.05,
        entry_slippage_frac=0.40,
        exit_limit_slippage_frac=0.35,
        exit_market_slippage_frac=0.75,
        commission_per_contract_usd=1.00,
    ),
]

LEGS_PER_CONDOR = 4
MULTIPLIER = 100  # standard equity options


@dataclass
class SizingPolicy:
    name: str
    initial_nav_usd: float
    risk_per_trade_pct: float       # cap risk per trade as % of NAV
    max_concurrent_pct: float       # cap aggregate at-risk across open trades as % of NAV
    min_qty: int = 1
    max_qty: int = 50


def _haircut_per_trade(row: pd.Series, scen: FrictionScenario) -> float:
    """Per-contract dollar haircut applied to one trade given its exit reason."""
    half = scen.spread_per_leg_usd / 2.0
    entry = LEGS_PER_CONDOR * scen.entry_slippage_frac * half * MULTIPLIER
    if str(row.get("reason")) in ("stop_loss", "end_forced"):
        exit_slip = scen.exit_market_slippage_frac
    else:
        exit_slip = scen.exit_limit_slippage_frac
    exit_ = LEGS_PER_CONDOR * exit_slip * half * MULTIPLIER
    commission = LEGS_PER_CONDOR * 2 * scen.commission_per_contract_usd
    return entry + exit_ + commission


def apply_friction(trades: pd.DataFrame, scen: FrictionScenario) -> pd.DataFrame:
    out = trades.copy()
    out["haircut_per_contract"] = out.apply(_haircut_per_trade, axis=1, args=(scen,))
    out["pnl_per_contract_realistic"] = out["pnl_usd"] - out["haircut_per_contract"]
    return out


def per_trade_max_loss(row: pd.Series) -> float:
    """Max possible loss per contract on a 4-leg iron condor."""
    wing_call = float(row["long_call"]) - float(row["short_call"])
    wing_put = float(row["short_put"]) - float(row["long_put"])
    wing = max(wing_call, wing_put, 0.01)
    credit = float(row["entry_credit"])
    return max((wing - credit) * MULTIPLIER, 0.0)


def size_quantity(per_trade_loss: float, sizing: SizingPolicy) -> int:
    if per_trade_loss <= 0:
        return sizing.min_qty
    risk_dollars = sizing.initial_nav_usd * sizing.risk_per_trade_pct / 100.0
    qty = int(risk_dollars // per_trade_loss)
    return max(sizing.min_qty, min(sizing.max_qty, qty))


def replay_with_sizing(
    trades: pd.DataFrame,
    sizing: SizingPolicy,
    scen: FrictionScenario,
) -> dict[str, Any]:
    """Walk trades chronologically, applying NAV-based sizing + friction.

    Tracks aggregate at-risk dollars; if a new trade would push total risk
    above the cap, scale its quantity down or skip.
    """
    df = trades.sort_values("entry_date").reset_index(drop=True).copy()
    df["max_loss_per_contract"] = df.apply(per_trade_max_loss, axis=1)
    df["entry_date_dt"] = pd.to_datetime(df["entry_date"])
    df["exit_date_dt"] = pd.to_datetime(df["exit_date"])

    nav = sizing.initial_nav_usd
    aggregate_risk_cap = nav * sizing.max_concurrent_pct / 100.0
    open_positions: list[dict[str, Any]] = []
    history = []

    for _, row in df.iterrows():
        entry_dt = row["entry_date_dt"]
        # Close anything that has matured
        still_open = []
        for p in open_positions:
            if p["exit_dt"] <= entry_dt:
                # Realize P&L into NAV
                nav += p["realized_pnl"]
            else:
                still_open.append(p)
        open_positions = still_open

        per_contract_loss = float(row["max_loss_per_contract"])
        ideal_qty = size_quantity(per_contract_loss, sizing)
        # Cap by aggregate risk
        current_risk = sum(p["risk"] for p in open_positions)
        room = aggregate_risk_cap - current_risk
        if room <= 0:
            actual_qty = 0
        else:
            actual_qty = min(ideal_qty, max(0, int(room // max(per_contract_loss, 1.0))))
        if actual_qty <= 0:
            history.append({
                "entry_date": str(row["entry_date"]),
                "exit_date": str(row["exit_date"]),
                "ideal_qty": ideal_qty,
                "actual_qty": 0,
                "skipped_reason": "concurrent_risk_cap",
                "nav_at_entry": nav,
            })
            continue

        per_contract_pnl = float(row["pnl_usd"]) - _haircut_per_trade(row, scen)
        pos = {
            "exit_dt": row["exit_date_dt"],
            "qty": actual_qty,
            "risk": actual_qty * per_contract_loss,
            "realized_pnl": actual_qty * per_contract_pnl,
        }
        open_positions.append(pos)
        history.append({
            "entry_date": str(row["entry_date"]),
            "exit_date": str(row["exit_date"]),
            "ideal_qty": ideal_qty,
            "actual_qty": actual_qty,
            "per_contract_pnl": per_contract_pnl,
            "trade_pnl": pos["realized_pnl"],
            "nav_at_entry": nav,
            "concurrent_at_entry": len(open_positions),
            "reason": str(row.get("reason", "")),
        })

    # Close anything still open at end
    for p in open_positions:
        nav += p["realized_pnl"]

    hist_df = pd.DataFrame(history)
    placed = hist_df[hist_df["actual_qty"] > 0] if not hist_df.empty else hist_df
    skipped = hist_df[hist_df["actual_qty"] == 0] if not hist_df.empty else hist_df

    if placed.empty:
        return {
            "scenario": scen.name,
            "sizing": sizing.name,
            "initial_nav_usd": sizing.initial_nav_usd,
            "final_nav_usd": nav,
            "total_pnl_usd": nav - sizing.initial_nav_usd,
            "return_pct": 0.0,
            "trades_attempted": int(len(df)),
            "trades_placed": 0,
            "trades_skipped": int(len(df)),
            "win_rate_pct": 0.0,
            "max_concurrent": 0,
        }

    # Compute equity curve, max drawdown, sharpe (per-trade approximation)
    pnls = placed["trade_pnl"].astype(float).to_numpy()
    cum = np.cumsum(pnls)
    nav_series = sizing.initial_nav_usd + cum
    peak = np.maximum.accumulate(nav_series)
    dd = (nav_series - peak) / np.maximum(peak, 1.0)
    max_dd = float(dd.min()) * 100.0

    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls)) if len(pnls) > 1 else 1.0
    sharpe_per_trade = (mean_pnl / std_pnl) if std_pnl > 0 else 0.0
    # Annualize assuming average ~213 trades/year
    annual_trade_rate = max(1.0, float(len(placed)) / 7.0)
    sharpe_annual = sharpe_per_trade * np.sqrt(annual_trade_rate)

    return {
        "scenario": scen.name,
        "sizing": sizing.name,
        "initial_nav_usd": sizing.initial_nav_usd,
        "final_nav_usd": float(nav),
        "total_pnl_usd": float(nav - sizing.initial_nav_usd),
        "return_pct": float((nav / sizing.initial_nav_usd - 1.0) * 100.0),
        "trades_attempted": int(len(df)),
        "trades_placed": int(len(placed)),
        "trades_skipped": int(len(skipped)),
        "win_rate_pct": float((placed["trade_pnl"] > 0).mean() * 100.0),
        "avg_qty": float(placed["actual_qty"].mean()),
        "median_qty": int(placed["actual_qty"].median()),
        "max_qty": int(placed["actual_qty"].max()),
        "max_concurrent": int(placed["concurrent_at_entry"].max()),
        "mean_pnl_per_trade": mean_pnl,
        "std_pnl_per_trade": std_pnl,
        "sharpe_per_trade": float(sharpe_per_trade),
        "sharpe_annualized": float(sharpe_annual),
        "max_drawdown_pct": max_dd,
    }


def naive_friction_only(trades: pd.DataFrame, scen: FrictionScenario) -> dict[str, Any]:
    """Apply friction with 1 contract per trade, no sizing — comparable to original backtest."""
    df = apply_friction(trades, scen)
    pnls = df["pnl_per_contract_realistic"].astype(float).to_numpy()
    cum = np.cumsum(pnls)
    nav = 10000.0 + cum
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / np.maximum(peak, 1.0)
    return {
        "scenario": scen.name,
        "sizing": "naive_1_contract_$10k",
        "initial_nav_usd": 10000.0,
        "final_nav_usd": float(nav[-1]) if len(nav) else 10000.0,
        "total_pnl_usd": float(np.sum(pnls)),
        "return_pct": float((nav[-1] / 10000.0 - 1.0) * 100.0) if len(nav) else 0.0,
        "trades_placed": int(len(df)),
        "win_rate_pct": float((df["pnl_per_contract_realistic"] > 0).mean() * 100.0),
        "mean_pnl_per_trade": float(np.mean(pnls)),
        "std_pnl_per_trade": float(np.std(pnls)),
        "max_drawdown_pct": float(dd.min()) * 100.0 if len(dd) else 0.0,
        "haircut_total_usd": float(df["haircut_per_contract"].sum()),
    }


SIZING_POLICIES = [
    SizingPolicy(name="conservative_0.5pct_per_trade_8pct_total",
                 initial_nav_usd=100_000.0, risk_per_trade_pct=0.5,
                 max_concurrent_pct=8.0),
    SizingPolicy(name="standard_1pct_per_trade_15pct_total",
                 initial_nav_usd=100_000.0, risk_per_trade_pct=1.0,
                 max_concurrent_pct=15.0),
    SizingPolicy(name="aggressive_2pct_per_trade_25pct_total",
                 initial_nav_usd=100_000.0, risk_per_trade_pct=2.0,
                 max_concurrent_pct=25.0),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-csv", type=Path,
                        default=REPO_ROOT / "experiments/runs/council_spy_grid_20260510T153306Z/trades.csv")
    parser.add_argument("--strategy", type=str,
                        default="risk_qwen_risk_designer_risktest_d12_30d_lowdelta_pt30_sl20_c10")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "experiments/runs/council_spy_grid_20260510T153306Z/realistic")
    args = parser.parse_args()

    trades = pd.read_csv(args.trades_csv)
    w = trades[trades["strategy"] == args.strategy].copy()
    if w.empty:
        print(f"no trades for strategy {args.strategy}")
        return 1

    print(f"Strategy: {args.strategy}")
    print(f"Proxy trades: {len(w)}, original P&L: ${w['pnl_usd'].sum():,.2f}")

    # Original (no friction, 1 contract, $10k)
    original = {
        "scenario": "PROXY_no_friction",
        "sizing": "naive_1_contract_$10k",
        "initial_nav_usd": 10000.0,
        "final_nav_usd": float(10000 + w["pnl_usd"].sum()),
        "total_pnl_usd": float(w["pnl_usd"].sum()),
        "return_pct": float(w["pnl_usd"].sum() / 10000 * 100),
        "trades_placed": int(len(w)),
        "win_rate_pct": float((w["pnl_usd"] > 0).mean() * 100.0),
        "mean_pnl_per_trade": float(w["pnl_usd"].mean()),
        "std_pnl_per_trade": float(w["pnl_usd"].std()),
        "max_drawdown_pct": -14.7,  # from Codex's report
    }

    rows = [original]
    print("\n=== Naive friction (1 contract, $10k notional) ===")
    for scen in SCENARIOS:
        r = naive_friction_only(w, scen)
        rows.append(r)
        print(f"  {scen.name:38} -> P&L ${r['total_pnl_usd']:>9,.0f} return {r['return_pct']:>7.1f}% "
              f"haircut ${r['haircut_total_usd']:>8,.0f} mean ${r['mean_pnl_per_trade']:>6.2f}")

    print("\n=== With $100k NAV position sizing ===")
    for sizing in SIZING_POLICIES:
        for scen in SCENARIOS:
            r = replay_with_sizing(w, sizing, scen)
            rows.append(r)
            print(f"  {sizing.name:50} | {scen.name:30} -> "
                  f"NAV ${r['final_nav_usd']:>9,.0f} ret {r['return_pct']:>6.1f}% "
                  f"DD {r['max_drawdown_pct']:>5.1f}% Sharpe {r['sharpe_annualized']:.2f} "
                  f"qty avg={r['avg_qty']:.1f} max_concurrent={r['max_concurrent']}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"{args.strategy}.json"
    out_md = args.out_dir / f"{args.strategy}.md"
    out_json.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")

    md = ["# Realistic Execution + Sizing Results", "",
          f"**Strategy**: `{args.strategy}`",
          f"**Source trades**: {args.trades_csv.relative_to(REPO_ROOT)}",
          f"**Proxy trades**: {len(w)}", "",
          "## Headline numbers", "",
          f"- **Proxy claim** (no friction, 1 contract, $10k): +{original['return_pct']:.1f}%",
          ""]
    rec = next((r for r in rows if r.get("scenario") == "realistic_$0.03_spread"
                and r.get("sizing") == "standard_1pct_per_trade_15pct_total"), None)
    if rec:
        md.append(f"- **Realistic + $100k NAV + 1% per-trade risk**: ${rec['final_nav_usd']:,.0f} "
                  f"({rec['return_pct']:+.1f}%, max DD {rec['max_drawdown_pct']:.1f}%, "
                  f"Sharpe {rec['sharpe_annualized']:.2f})")
    md.append("")
    md.append("## Sizing × Friction grid ($100k NAV)")
    md.append("")
    md.append("| Sizing policy | Friction | Final NAV | Return | Max DD | Sharpe | Avg qty | Max concurrent |")
    md.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["sizing"] == "naive_1_contract_$10k":
            continue
        md.append(f"| {r['sizing']} | {r['scenario']} | "
                  f"${r['final_nav_usd']:,.0f} | {r['return_pct']:+.1f}% | "
                  f"{r['max_drawdown_pct']:.1f}% | {r['sharpe_annualized']:.2f} | "
                  f"{r['avg_qty']:.1f} | {r['max_concurrent']} |")
    out_md.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote: {out_json}\nWrote: {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
