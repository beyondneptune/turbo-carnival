"""Run all three lanes in sequence + side-by-side comparison + charts.

Lane 1: Existing quant strategy (1.7% monthly ceiling) — replayed from existing trades
Lane 2: NIM council, constrained menu (fullonriskmodelterminator/backtest.py)
Lane 3: Pure-API, free-form legs (fullonriskmodelterminator/pure_api_trader.py)

Each lane starts with $100k on the same date range.
Output: HTML dashboard + PNG comparison chart.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from fullonriskmodelterminator.backtest import run_backtest as run_council  # noqa: E402
from fullonriskmodelterminator.pure_api_trader import run_pure_api_backtest as run_pure_api  # noqa: E402


GREEN = "#2ecc71"
RED = "#e74c3c"
BLUE = "#3498db"
ORANGE = "#e67e22"
GRAY = "#7f8c8d"
PURPLE = "#9b59b6"


def lane_1_quant_replay(start: str, end: str | None, initial_capital: float,
                        out_dir: Path) -> dict:
    """Lane 1 — replay the existing council winner's trades over Year 2 with $100k sizing.

    Pulls from data/trading_system.db (the official evidence run trades),
    applies 1% per-trade NAV-aware sizing.
    """
    import sqlite3
    db = REPO_ROOT / "data" / "trading_system.db"
    conn = sqlite3.connect(db)
    RID = "paper_v1_evidence_20260510T131633Z"  # the live re-run with SPY core fix
    trades = pd.read_sql_query(
        """SELECT entry_at_utc, exit_at_utc, sleeve, underlying_ticker as ticker,
                   realized_pnl_usd FROM trades WHERE run_id = ?
                   AND realized_pnl_usd IS NOT NULL ORDER BY entry_at_utc""",
        conn, params=(RID,),
    )
    conn.close()

    if trades.empty:
        return {"final_nav": initial_capital, "trades_placed": 0, "monthly_pct": 0.0,
                "max_dd_pct": 0.0, "lane": "quant_realistic"}

    trades["entry_dt"] = pd.to_datetime(trades["entry_at_utc"], utc=True).dt.tz_localize(None)
    trades["exit_dt"] = pd.to_datetime(trades["exit_at_utc"], utc=True).dt.tz_localize(None)
    # Filter to date range
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end) if end else trades["entry_dt"].max()
    trades = trades[(trades["entry_dt"] >= start_dt) & (trades["entry_dt"] <= end_dt)].copy()

    if trades.empty:
        return {"final_nav": initial_capital, "trades_placed": 0, "monthly_pct": 0.0,
                "max_dd_pct": 0.0, "lane": "quant_realistic"}

    # NAV-aware sizing: 1% per trade on $100k
    nav = initial_capital
    nav_history = []
    placed = []
    BASELINE_RISK_PER_TRADE = 412.0  # approximate from earlier audit
    risk_pct = 0.01

    open_pos = []
    for _, row in trades.sort_values("entry_dt").iterrows():
        # close matured
        kept = []
        for p in open_pos:
            if p["exit_dt"] <= row["entry_dt"]:
                nav += p["realized"]
            else:
                kept.append(p)
        open_pos = kept
        nav_history.append({"date": row["entry_dt"].date().isoformat(), "nav": nav})

        risk_dollars = nav * risk_pct
        qty = max(1, int(risk_dollars // BASELINE_RISK_PER_TRADE))
        # Apply ~25% friction haircut (per our realistic_execution friction model)
        pnl_per_contract = float(row["realized_pnl_usd"]) - 10.0  # $10 friction
        trade_pnl = qty * pnl_per_contract
        open_pos.append({
            "exit_dt": row["exit_dt"], "qty": qty, "realized": trade_pnl,
        })
        placed.append({
            "open_date": row["entry_dt"].date().isoformat(),
            "close_date": row["exit_dt"].date().isoformat(),
            "qty": qty, "pnl_usd": trade_pnl, "sleeve": row["sleeve"],
        })

    for p in open_pos:
        nav += p["realized"]
    nav_history.append({"date": pd.Timestamp(end_dt).date().isoformat(), "nav": nav})

    nav_df = pd.DataFrame(nav_history)
    trades_df = pd.DataFrame(placed)
    nav_df.to_csv(out_dir / "lane1_nav.csv", index=False)
    trades_df.to_csv(out_dir / "lane1_trades.csv", index=False)

    final_nav = float(nav_df["nav"].iloc[-1])
    total_return = (final_nav / initial_capital - 1) * 100
    days = (pd.to_datetime(nav_df["date"].iloc[-1]) - pd.to_datetime(nav_df["date"].iloc[0])).days
    years = max(days / 365.0, 1 / 365.0)
    monthly = ((final_nav / initial_capital) ** (1 / max(years * 12, 0.1)) - 1) * 100 if final_nav > 0 else -100
    peak = nav_df["nav"].cummax()
    dd = (nav_df["nav"] - peak) / np.maximum(peak, 1.0)
    max_dd = float(dd.min() * 100) if not dd.empty else 0.0

    return {
        "lane": "quant_realistic",
        "final_nav": final_nav,
        "total_return_pct": total_return,
        "monthly_pct": monthly,
        "max_dd_pct": max_dd,
        "trades_placed": int(len(trades_df)),
        "nav_csv": str(out_dir / "lane1_nav.csv"),
    }


def make_comparison_chart(lane_results: list[dict], out_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(15, 9), gridspec_kw={"height_ratios": [3, 1]}, sharex=False)
    colors = [BLUE, ORANGE, GREEN]
    ax = axes[0]
    for i, lane in enumerate(lane_results):
        if "nav_history" in lane:
            df = pd.DataFrame(lane["nav_history"])
        elif "nav_csv" in lane:
            df = pd.read_csv(lane["nav_csv"])
        else:
            continue
        if df.empty:
            continue
        df["date_dt"] = pd.to_datetime(df["date"])
        label = (f"{lane['lane_name']}: ${lane['final_nav']:,.0f} "
                 f"({lane['total_return_pct']:+.1f}%, DD {lane['max_dd_pct']:.1f}%)")
        ax.plot(df["date_dt"], df["nav"], color=colors[i % 3], linewidth=2, label=label)
    ax.axhline(100_000, color=GRAY, linestyle="--", alpha=0.6, label="$100k initial")
    ax.set_ylabel("NAV ($)")
    ax.set_title("3-Lane Comparison — fullonriskmodelterminator")
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    # Bar chart of monthly returns
    ax2 = axes[1]
    names = [l["lane_name"] for l in lane_results]
    monthlies = [l["monthly_pct"] for l in lane_results]
    bar_colors = [GREEN if m >= 0 else RED for m in monthlies]
    bars = ax2.bar(names, monthlies, color=bar_colors, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.6)
    ax2.axhline(14, color=PURPLE, linestyle="--", alpha=0.7, label="Investor target 14%")
    ax2.set_ylabel("Monthly return (%)")
    ax2.set_title("Monthly return — investor target vs each lane")
    ax2.legend()
    for bar, m in zip(bars, monthlies):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f" {m:+.1f}%", ha="center", va="bottom" if m >= 0 else "top", fontsize=10)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = out_dir / "three_lane_comparison.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-05-07")
    parser.add_argument("--end", default=None)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--max-days", type=int, default=60,
                        help="Backtest length for Lane 2/3 (LLM calls; keep moderate)")
    parser.add_argument("--skip-lane2", action="store_true")
    parser.add_argument("--skip-lane3", action="store_true")
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = REPO_ROOT / "fullonriskmodelterminator" / "runs" / f"three_lane_{stamp}"
    out.mkdir(parents=True, exist_ok=True)

    results = []

    print("\n=== Lane 1: Quant Realistic (1% NAV per trade, friction-modeled) ===\n")
    r1 = lane_1_quant_replay(args.start, args.end, args.capital, out)
    r1["lane_name"] = "Lane 1 — Quant"
    results.append(r1)
    print(json.dumps(r1, indent=2, default=str))

    if not args.skip_lane2:
        print("\n=== Lane 2: NIM Council (constrained menu, 15% sizing) ===\n")
        r2 = run_council(start=args.start, end=args.end, initial_capital=args.capital,
                          max_days=args.max_days, out_root=out)
        r2["lane_name"] = "Lane 2 — Council"
        nav_df = pd.read_csv(Path(r2["output_dir"]) / "nav_history.csv")
        r2["nav_history"] = nav_df.to_dict("records")
        results.append(r2)

    if not args.skip_lane3:
        print("\n=== Lane 3: Pure-API (free-form legs, max agentic) ===\n")
        r3 = run_pure_api(start=args.start, end=args.end, initial_capital=args.capital,
                           max_days=args.max_days)
        r3["lane_name"] = "Lane 3 — Pure-API"
        nav_df = pd.read_csv(Path(r3["output_dir"]) / "nav_history.csv")
        r3["nav_history"] = nav_df.to_dict("records")
        results.append(r3)

    chart = make_comparison_chart(results, out)
    print(f"\n=== Chart: {chart}")

    summary = {
        "stamp": stamp,
        "start": args.start,
        "end": args.end,
        "capital": args.capital,
        "max_days": args.max_days,
        "lanes": [
            {k: v for k, v in r.items() if k not in ("nav_history",)} for r in results
        ],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n=== 3-LANE COMPARISON ===")
    for r in results:
        print(f"  {r['lane_name']:25} | final ${r['final_nav']:,.0f} | "
              f"monthly {r['monthly_pct']:+.2f}% | DD {r['max_dd_pct']:.1f}% | "
              f"trades {r.get('trades_placed', '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
