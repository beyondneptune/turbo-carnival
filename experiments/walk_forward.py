"""Walk-forward validation: split council winner trades into in-sample vs out-of-sample.

If in-sample Sharpe is 1.34 but out-of-sample collapses, the strategy is overfit.
The honest test is: did the 2019-2022 winning behavior generalize to 2023-2026?
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]


def stats_block(name: str, df: pd.DataFrame) -> dict:
    pnls = df["pnl_usd"].astype(float).to_numpy()
    if len(pnls) == 0:
        return {"name": name, "n_trades": 0}
    cum = np.cumsum(pnls)
    nav = 10000.0 + cum
    peak = np.maximum.accumulate(nav)
    dd = (nav - peak) / peak
    annual_trade_rate = max(1.0, len(pnls) / max(1.0, (
        (pd.to_datetime(df["entry_date"].max()) - pd.to_datetime(df["entry_date"].min())).days / 365.25)))
    mean = float(np.mean(pnls))
    std = float(np.std(pnls)) if len(pnls) > 1 else 1.0
    sharpe_per_trade = (mean / std) if std > 0 else 0.0
    return {
        "name": name,
        "n_trades": int(len(df)),
        "start": str(df["entry_date"].min()),
        "end": str(df["entry_date"].max()),
        "total_pnl_usd": float(cum[-1]),
        "final_nav_usd": float(nav[-1]),
        "return_pct": float((nav[-1] / 10000.0 - 1.0) * 100),
        "win_rate_pct": float((df["pnl_usd"] > 0).mean() * 100),
        "mean_pnl_per_trade": mean,
        "std_pnl_per_trade": std,
        "sharpe_per_trade": float(sharpe_per_trade),
        "sharpe_annualized": float(sharpe_per_trade * np.sqrt(annual_trade_rate)),
        "max_drawdown_pct": float(dd.min()) * 100,
        "annual_trade_rate": float(annual_trade_rate),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trades-csv", type=Path,
                        default=REPO_ROOT / "experiments/runs/council_spy_grid_20260510T153306Z/trades.csv")
    parser.add_argument("--strategy", type=str,
                        default="risk_qwen_risk_designer_risktest_d12_30d_lowdelta_pt30_sl20_c10")
    parser.add_argument("--split", type=str, default="2023-01-01",
                        help="Date splitting in-sample (before) vs out-of-sample (on/after).")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "experiments/runs/council_spy_grid_20260510T153306Z/walk_forward")
    args = parser.parse_args()

    trades = pd.read_csv(args.trades_csv)
    w = trades[trades["strategy"] == args.strategy].copy()
    w["entry_dt"] = pd.to_datetime(w["entry_date"])

    in_sample = w[w["entry_dt"] < args.split].copy()
    out_sample = w[w["entry_dt"] >= args.split].copy()
    full = w

    blocks = [
        stats_block(f"FULL_PERIOD_2019-2026", full),
        stats_block(f"IN_SAMPLE_pre_{args.split}", in_sample),
        stats_block(f"OUT_OF_SAMPLE_post_{args.split}", out_sample),
    ]

    # Top 5 council variants — same walk-forward
    print(f"=== Walk-forward for {args.strategy} (split at {args.split}) ===\n")
    for b in blocks:
        if b["n_trades"] == 0:
            continue
        print(f"  {b['name']:40} n={b['n_trades']:4} period={b['start']}..{b['end']}")
        print(f"    P&L ${b['total_pnl_usd']:>9,.0f}  ret {b['return_pct']:>6.1f}%  "
              f"DD {b['max_drawdown_pct']:>6.1f}%  Sharpe {b['sharpe_annualized']:.2f}  "
              f"win {b['win_rate_pct']:.1f}%")

    # Ratio: did OOS hold up?
    is_b = next((b for b in blocks if "IN_SAMPLE" in b["name"]), None)
    oos_b = next((b for b in blocks if "OUT_OF_SAMPLE" in b["name"]), None)
    if is_b and oos_b:
        sharpe_ratio = (oos_b["sharpe_annualized"] / max(0.001, is_b["sharpe_annualized"]))
        return_ratio = (oos_b["return_pct"] / max(0.001, is_b["return_pct"])) if is_b["return_pct"] else 0
        print()
        print(f"  >> OOS Sharpe / IS Sharpe = {sharpe_ratio:.2f} "
              f"(>0.7 = good generalization, <0.5 = overfit signal)")
        print(f"  >> OOS return / IS return = {return_ratio:.2f}")

    # Run for ALL strategies in the trades file at the same split
    print(f"\n=== Walk-forward across all 61 council variants (Sharpe IS vs OOS) ===\n")
    rows = []
    for strat in trades["strategy"].unique():
        s = trades[trades["strategy"] == strat].copy()
        s["entry_dt"] = pd.to_datetime(s["entry_date"])
        is_s = s[s["entry_dt"] < args.split]
        oos_s = s[s["entry_dt"] >= args.split]
        if len(is_s) < 20 or len(oos_s) < 20:
            continue
        is_stats = stats_block("is", is_s)
        oos_stats = stats_block("oos", oos_s)
        rows.append({
            "strategy": strat,
            "is_n": is_stats["n_trades"],
            "oos_n": oos_stats["n_trades"],
            "is_return": is_stats["return_pct"],
            "oos_return": oos_stats["return_pct"],
            "is_sharpe": is_stats["sharpe_annualized"],
            "oos_sharpe": oos_stats["sharpe_annualized"],
            "is_dd": is_stats["max_drawdown_pct"],
            "oos_dd": oos_stats["max_drawdown_pct"],
        })
    summary_df = pd.DataFrame(rows).sort_values("oos_sharpe", ascending=False)
    if not summary_df.empty:
        print("Top 10 by OOS Sharpe:")
        for _, r in summary_df.head(10).iterrows():
            print(f"  {r['strategy'][:60]:60} IS_S {r['is_sharpe']:.2f} -> OOS_S {r['oos_sharpe']:.2f} "
                  f"  IS_ret {r['is_return']:+.0f}% -> OOS_ret {r['oos_return']:+.0f}%")
        print()
        print(f"Strategies with OOS Sharpe > 1.0: {(summary_df['oos_sharpe'] > 1.0).sum()}/{len(summary_df)}")
        print(f"Strategies with OOS Sharpe > IS Sharpe (sign of robustness): {(summary_df['oos_sharpe'] > summary_df['is_sharpe']).sum()}/{len(summary_df)}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"focused_strategy": args.strategy, "split": args.split, "blocks": blocks, "all_strategies": rows}
    out_json = args.out_dir / f"walk_forward_split_{args.split}.json"
    out_json.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    if not summary_df.empty:
        summary_df.to_csv(args.out_dir / f"walk_forward_split_{args.split}.csv", index=False)
    print(f"\nWrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
