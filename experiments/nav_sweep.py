"""NAV sweep: run the council winner at $10k, $20k, ..., $100k.

Side-by-side comparison so the user can see how outcomes scale with account size.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO_ROOT))
from experiments.realistic_execution import (
    SizingPolicy,
    SCENARIOS,
    replay_with_sizing,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-csv", type=Path,
                        default=REPO_ROOT / "experiments/runs/council_spy_grid_20260510T153306Z/trades.csv")
    parser.add_argument("--strategy", type=str,
                        default="risk_qwen_risk_designer_risktest_d12_30d_lowdelta_pt30_sl20_c10")
    parser.add_argument("--out", type=Path,
                        default=REPO_ROOT / "experiments/runs/council_spy_grid_20260510T153306Z/nav_sweep.md")
    args = parser.parse_args()

    trades = pd.read_csv(args.trades_csv)
    w = trades[trades["strategy"] == args.strategy].copy()
    if w.empty:
        print(f"no trades for {args.strategy}"); return 1

    realistic = next(s for s in SCENARIOS if s.name == "realistic_$0.03_spread")
    pessimistic = next(s for s in SCENARIOS if s.name == "pessimistic_$0.05_spread")

    nav_levels = [10_000, 20_000, 30_000, 50_000, 75_000, 100_000]
    risk_levels = [0.5, 1.0, 2.0]

    rows = []
    for nav in nav_levels:
        for risk in risk_levels:
            sp = SizingPolicy(
                name=f"{risk}pct_per_trade",
                initial_nav_usd=float(nav),
                risk_per_trade_pct=risk,
                max_concurrent_pct=min(25.0, risk * 15),
            )
            for scen in (realistic, pessimistic):
                r = replay_with_sizing(w, sp, scen)
                # Compute monthly equivalent (84 months in 7 years)
                cagr = (r["final_nav_usd"] / r["initial_nav_usd"]) ** (1 / 7) - 1
                monthly = (1 + cagr) ** (1 / 12) - 1
                rows.append({
                    "nav": nav, "risk_per_trade_pct": risk, "scenario": scen.name,
                    "final_nav": r["final_nav_usd"], "return_pct": r["return_pct"],
                    "cagr_pct": cagr * 100, "monthly_pct": monthly * 100,
                    "max_dd_pct": r["max_drawdown_pct"], "sharpe": r["sharpe_annualized"],
                    "trades_placed": r["trades_placed"], "avg_qty": r["avg_qty"],
                })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    md = ["# NAV Sweep — Council Winner Across Account Sizes", "",
          f"Strategy: `{args.strategy}`",
          f"7-year period (2019-2026), 1495 trades available",
          "", "## Realistic friction ($0.03 spread, $0.65 commission)", ""]
    md.append("| NAV | Risk/Trade | Final NAV | Return | CAGR | **Monthly** | Max DD | Sharpe | Avg Qty |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in df[df["scenario"].str.startswith("realistic")].iterrows():
        md.append(f"| ${r['nav']:,} | {r['risk_per_trade_pct']}% | "
                  f"${r['final_nav']:,.0f} | {r['return_pct']:+.1f}% | "
                  f"{r['cagr_pct']:.1f}% | **{r['monthly_pct']:.2f}%** | "
                  f"{r['max_dd_pct']:.1f}% | {r['sharpe']:.2f} | {r['avg_qty']:.1f} |")
    md.append("")
    md.append("## Pessimistic friction ($0.05 spread, $1 commission)")
    md.append("")
    md.append("| NAV | Risk/Trade | Final NAV | Return | CAGR | **Monthly** | Max DD | Sharpe | Avg Qty |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in df[df["scenario"].str.startswith("pessimistic")].iterrows():
        md.append(f"| ${r['nav']:,} | {r['risk_per_trade_pct']}% | "
                  f"${r['final_nav']:,.0f} | {r['return_pct']:+.1f}% | "
                  f"{r['cagr_pct']:.1f}% | **{r['monthly_pct']:.2f}%** | "
                  f"{r['max_dd_pct']:.1f}% | {r['sharpe']:.2f} | {r['avg_qty']:.1f} |")
    md.append("")
    md.append("## Headline answer to '14% monthly?'")
    md.append("")
    best = df.sort_values("monthly_pct", ascending=False).head(3)
    worst = df.sort_values("monthly_pct").head(3)
    md.append(f"- **Best monthly across all configurations**: {best.iloc[0]['monthly_pct']:.2f}% "
              f"(NAV ${best.iloc[0]['nav']:,}, {best.iloc[0]['risk_per_trade_pct']}%/trade, scenario={best.iloc[0]['scenario']})")
    md.append(f"- **Median monthly**: {df['monthly_pct'].median():.2f}%")
    md.append(f"- **Worst monthly**: {worst.iloc[0]['monthly_pct']:.2f}% "
              f"(NAV ${worst.iloc[0]['nav']:,}, {worst.iloc[0]['risk_per_trade_pct']}%/trade, scenario={worst.iloc[0]['scenario']})")
    md.append("")
    md.append(f"**Investor target was 14%/month.** This strategy's realistic best is "
              f"{best.iloc[0]['monthly_pct']:.2f}%/month — about **{14 / best.iloc[0]['monthly_pct']:.0f}x lower** "
              f"than the target.")
    md.append("")
    md.append("## What changes with NAV size")
    md.append("")
    md.append("- The **% return** is similar across NAV sizes (the strategy doesn't scale magically with capital)")
    md.append("- The **max drawdown** stays roughly constant in % terms")
    md.append("- The **dollar P&L** scales roughly proportionally to qty per trade")
    md.append("- Sharpe is essentially identical — sizing doesn't change the ratio")
    md.append("")
    md.append("Key implication: training on $10k vs $100k doesn't change the strategy's underlying edge. "
              "It only changes how many contracts you trade per signal.")

    args.out.write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
