"""Maximum-aggression search for 14% monthly target.

Designs aggressive parameter combos within the validated strategy framework,
runs them through the proxy backtester, then applies realistic friction +
max-risk position sizing at $10k AND $100k. Generates per-strategy charts
showing entry/exit/stop/profit markers.

Honest test: does ANY combination reach 14% monthly?
"""
from __future__ import annotations

import json
import subprocess
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

GREEN = "#2ecc71"
RED = "#e74c3c"
BLUE = "#3498db"
ORANGE = "#e67e22"
PURPLE = "#9b59b6"
GRAY = "#7f8c8d"


def build_aggressive_specs() -> list[dict]:
    """~25 aggressive variants within validated ranges."""
    specs = []

    # 1. High delta, short DTE, wide PT, loose stop — pure profit-grab
    for delta in (0.25, 0.30, 0.35):
        for dte in (14, 21, 30):
            for pt in (0.50, 0.75, 0.90):
                close_dte = max(1, dte // 4)
                specs.append({
                    "name": f"max_aggro_d{int(delta*100)}_{dte}d_pt{int(pt*100)}",
                    "entry_mode": "daily_ladder",
                    "gate": "none",
                    "target_delta": delta,
                    "target_dte": dte,
                    "wing_width": 3.0,
                    "profit_take": pt,
                    "stop_loss_credit_mult": 5.0,
                    "close_dte": close_dte,
                    "min_credit": 0.10,
                })

    # 2. Low delta, very high frequency, tight stop — pick up many small wins
    for delta in (0.08, 0.12, 0.15):
        for sl in (0.5, 1.0):
            specs.append({
                "name": f"high_freq_lowdelta_d{int(delta*100)}_sl{str(sl).replace('.','_')}",
                "entry_mode": "daily_ladder",
                "gate": "vrp_positive",
                "target_delta": delta,
                "target_dte": 30,
                "wing_width": 5.0,
                "profit_take": 0.25,
                "stop_loss_credit_mult": sl,
                "close_dte": 10,
                "min_credit": 0.10,
            })

    # 3. No-gate variants for raw frequency
    for delta in (0.15, 0.20, 0.25):
        specs.append({
            "name": f"no_gate_aggro_d{int(delta*100)}",
            "entry_mode": "daily_ladder",
            "gate": "none",
            "target_delta": delta,
            "target_dte": 30,
            "wing_width": 5.0,
            "profit_take": 0.50,
            "stop_loss_credit_mult": 3.0,
            "close_dte": 10,
            "min_credit": 0.10,
        })

    # Dedup
    seen = set()
    out = []
    for s in specs:
        if s["name"] in seen:
            continue
        seen.add(s["name"])
        out.append(s)
    return out


def run_grid(specs: list[dict], out_dir: Path) -> Path:
    spec_path = out_dir / "specs.json"
    spec_path.write_text(json.dumps({"strategies": specs}, indent=2), encoding="utf-8")
    grid_dir = out_dir / "grid"
    grid_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        ".venv/bin/python", "experiments/run_spy_core_strategy_grid.py",
        "--spec-json", str(spec_path),
        "--output-dir", str(grid_dir),
    ]
    print("Running:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if proc.returncode != 0:
        print("STDERR:", proc.stderr[-500:], flush=True)
        raise RuntimeError(f"grid run failed rc={proc.returncode}")
    return grid_dir


def apply_aggressive_sizing(trades: pd.DataFrame, nav_initial: float,
                             risk_pct: float, max_concurrent_pct: float,
                             friction_per_pt: float = 8.20,
                             friction_per_stop: float = 14.20) -> dict:
    """Replay trades with NAV-aware compounding + realistic friction."""
    df = trades.sort_values("entry_date").reset_index(drop=True).copy()
    df["entry_dt"] = pd.to_datetime(df["entry_date"])
    df["exit_dt"] = pd.to_datetime(df["exit_date"])
    df["wing"] = (df["long_call"] - df["short_call"]).clip(lower=0.01)
    df["max_loss_per_contract"] = (df["wing"] - df["entry_credit"]).clip(lower=0.01) * 100.0

    nav = nav_initial
    open_pos = []
    history = []
    for _, row in df.iterrows():
        # Close matured
        still = []
        for p in open_pos:
            if p["exit_dt"] <= row["entry_dt"]:
                nav += p["realized"]
            else:
                still.append(p)
        open_pos = still

        per_contract_loss = float(row["max_loss_per_contract"])
        risk_dollars = nav * risk_pct / 100.0
        ideal_qty = max(1, int(risk_dollars // max(per_contract_loss, 1.0)))

        current_risk = sum(p["risk"] for p in open_pos)
        cap = nav * max_concurrent_pct / 100.0
        room = cap - current_risk
        if room <= 0:
            actual_qty = 0
        else:
            actual_qty = min(ideal_qty, max(0, int(room // max(per_contract_loss, 1.0))))

        if actual_qty == 0:
            history.append({
                "entry_date": row["entry_date"], "exit_date": row["exit_date"],
                "actual_qty": 0, "skipped": True, "nav": nav, "reason": str(row["reason"]),
            })
            continue

        friction = friction_per_pt if str(row["reason"]) in ("profit_target", "dte_close") else friction_per_stop
        net = float(row["pnl_usd"]) - friction
        pnl = actual_qty * net
        risk_used = actual_qty * per_contract_loss
        open_pos.append({
            "exit_dt": row["exit_dt"], "qty": actual_qty,
            "risk": risk_used, "realized": pnl,
        })
        history.append({
            "entry_date": row["entry_date"], "exit_date": row["exit_date"],
            "entry_spy": float(row["entry_spy"]), "exit_spy": float(row["exit_spy"]),
            "short_call": float(row["short_call"]), "long_call": float(row["long_call"]),
            "short_put": float(row["short_put"]), "long_put": float(row["long_put"]),
            "actual_qty": actual_qty, "per_contract_pnl_net": net,
            "trade_pnl": pnl, "nav_at_entry": nav,
            "reason": str(row["reason"]), "skipped": False,
        })

    for p in open_pos:
        nav += p["realized"]

    h = pd.DataFrame(history)
    placed = h[~h["skipped"]] if not h.empty else h
    if placed.empty:
        return {"final_nav": nav, "history": h, "trades": 0}

    pnls = placed["trade_pnl"].astype(float).to_numpy()
    cum = np.cumsum(pnls)
    nav_curve = nav_initial + cum
    peak = np.maximum.accumulate(nav_curve)
    dd = (nav_curve - peak) / np.maximum(peak, 1.0)
    cagr = (nav / nav_initial) ** (1 / 7) - 1
    monthly = (1 + cagr) ** (1 / 12) - 1
    return {
        "final_nav": float(nav),
        "total_pnl": float(nav - nav_initial),
        "return_pct": float((nav / nav_initial - 1) * 100),
        "cagr_pct": float(cagr * 100),
        "monthly_pct": float(monthly * 100),
        "max_dd_pct": float(dd.min() * 100),
        "trades_placed": int(len(placed)),
        "trades_skipped": int((h["skipped"] == True).sum()),
        "win_rate_pct": float((placed["trade_pnl"] > 0).mean() * 100),
        "avg_qty": float(placed["actual_qty"].mean()),
        "max_qty": int(placed["actual_qty"].max()),
        "history": h,
    }


def plot_strategy(strategy: str, history: pd.DataFrame, nav_initial: float,
                   sizing_label: str, out_path: Path) -> None:
    placed = history[~history["skipped"]].copy() if "skipped" in history.columns else history.copy()
    if placed.empty:
        return
    placed["entry_dt"] = pd.to_datetime(placed["entry_date"])
    placed["exit_dt"] = pd.to_datetime(placed["exit_date"])
    placed["cum_pnl"] = placed["trade_pnl"].astype(float).cumsum()
    placed["nav"] = nav_initial + placed["cum_pnl"]
    peak = placed["nav"].cummax()
    placed["dd_pct"] = (placed["nav"] - peak) / peak * 100

    fig, axes = plt.subplots(3, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [3, 2, 2]}, sharex=True)
    final_nav = float(placed["nav"].iloc[-1])
    ret_pct = (final_nav / nav_initial - 1) * 100
    cagr = (final_nav / nav_initial) ** (1 / 7) - 1
    monthly = (1 + cagr) ** (1 / 12) - 1

    # Equity curve
    ax = axes[0]
    ax.plot(placed["entry_dt"], placed["nav"], color=BLUE, linewidth=1.6)
    ax.fill_between(placed["entry_dt"], placed["nav"], peak,
                     where=placed["nav"] < peak, color=RED, alpha=0.2, label="drawdown")
    ax.axhline(nav_initial, color=GRAY, linestyle="--", alpha=0.6, label=f"initial ${nav_initial:,.0f}")
    ax.set_ylabel("NAV ($)")
    ax.set_title(f"{strategy}\n[{sizing_label}]  Final ${final_nav:,.0f} ({ret_pct:+.1f}%)  "
                 f"CAGR {cagr*100:.1f}%  Monthly {monthly*100:.2f}%  "
                 f"MaxDD {placed['dd_pct'].min():.1f}%")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    # Per-trade P&L colored by reason
    ax = axes[1]
    color_map = {"profit_target": GREEN, "stop_loss": RED, "dte_close": BLUE,
                  "end_forced": GRAY}
    for reason, sub in placed.groupby("reason"):
        ax.scatter(sub["entry_dt"], sub["trade_pnl"],
                   color=color_map.get(reason, PURPLE),
                   s=20, alpha=0.7, label=f"{reason} ({len(sub)})")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Trade P&L ($)")
    ax.set_title(f"Per-trade P&L by exit reason — n={len(placed)} placed, "
                 f"win rate {(placed['trade_pnl']>0).mean()*100:.1f}%")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Drawdown
    ax = axes[2]
    ax.fill_between(placed["entry_dt"], placed["dd_pct"], 0,
                     color=RED, alpha=0.4)
    ax.plot(placed["entry_dt"], placed["dd_pct"], color=RED, linewidth=1.0)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("Drawdown (%)")
    ax.set_xlabel("Trade entry date")
    ax.set_title(f"Drawdown trajectory — max {placed['dd_pct'].min():.1f}%")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()


def main() -> int:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = REPO_ROOT / "experiments" / "runs" / f"max_aggression_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    chart_dir = out_dir / "charts"
    chart_dir.mkdir(parents=True, exist_ok=True)

    print("=== Building aggressive specs ===", flush=True)
    specs = build_aggressive_specs()
    print(f"  {len(specs)} variants", flush=True)

    print("\n=== Running backtest grid ===", flush=True)
    grid_dir = run_grid(specs, out_dir)
    trades = pd.read_csv(grid_dir / "trades.csv")

    sizing_configs = [
        ("$10k_max_risk_5pct", 10_000.0, 5.0, 50.0),
        ("$10k_extreme_risk_10pct", 10_000.0, 10.0, 80.0),
        ("$100k_max_risk_5pct", 100_000.0, 5.0, 50.0),
        ("$100k_extreme_risk_10pct", 100_000.0, 10.0, 80.0),
    ]

    rows = []
    for spec_name in trades["strategy"].unique():
        w = trades[trades["strategy"] == spec_name].copy()
        if len(w) < 30:
            continue
        for sizing_label, nav, risk_pct, conc_pct in sizing_configs:
            r = apply_aggressive_sizing(w, nav, risk_pct, conc_pct)
            if r["trades_placed"] == 0:
                continue
            rows.append({
                "strategy": spec_name, "sizing": sizing_label,
                "nav_initial": nav, "risk_pct": risk_pct,
                **{k: v for k, v in r.items() if k != "history"},
            })

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "all_results.csv", index=False)

    # The 14% question: any monthly >= 14?
    hit_14 = df[df["monthly_pct"] >= 14.0]
    near_14 = df[df["monthly_pct"] >= 5.0]
    print(f"\n=== Results: {len(df)} total runs ===")
    print(f"Hit 14% monthly target: {len(hit_14)}")
    print(f"Hit >= 5% monthly: {len(near_14)}")
    print()
    print("=== Top 10 by monthly return ===")
    top = df.sort_values("monthly_pct", ascending=False).head(10)
    print(top[["strategy", "sizing", "monthly_pct", "cagr_pct",
                "max_dd_pct", "trades_placed", "win_rate_pct"]].to_string(index=False))

    # Build charts for the actual top 5
    print("\n=== Building charts for top 5 ===", flush=True)
    for _, top_row in top.head(5).iterrows():
        spec_name = top_row["strategy"]
        sizing_label = top_row["sizing"]
        nav = float(top_row["nav_initial"])
        risk_pct = float(top_row["risk_pct"])
        # Find sizing config tuple
        sc = next(s for s in sizing_configs if s[0] == sizing_label)
        w = trades[trades["strategy"] == spec_name].copy()
        r = apply_aggressive_sizing(w, sc[1], sc[2], sc[3])
        chart_path = chart_dir / f"{spec_name}__{sizing_label}.png"
        plot_strategy(spec_name, r["history"], nav, sizing_label, chart_path)
        print(f"  saved {chart_path.relative_to(REPO_ROOT)}", flush=True)

    # Summary write
    summary = {
        "stamp": stamp,
        "total_runs": len(df),
        "hit_14pct_monthly": len(hit_14),
        "near_5pct_monthly": len(near_14),
        "best_monthly": float(df["monthly_pct"].max()),
        "best_strategy": df.loc[df["monthly_pct"].idxmax(), "strategy"],
        "best_sizing": df.loc[df["monthly_pct"].idxmax(), "sizing"],
        "investor_target_pct": 14.0,
        "gap_to_target": float(14.0 - df["monthly_pct"].max()),
    }
    (out_dir / "headline.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n=== HEADLINE ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\nOutput: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
