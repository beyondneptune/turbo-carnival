"""Sweep aggressive meta-engine selectors and sizing.

This extends `run_aggressive_meta_engine.py` without retraining repeatedly:
train scores once on the 70% window, then test many selector/risk combinations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_aggressive_meta_engine import (  # noqa: E402
    DEFAULT_ROOT,
    OUT_ROOT,
    benchmark,
    load_period,
    load_specs,
    replay_portfolio,
    train_scores,
)


def select_custom(test: pd.DataFrame, selector: str) -> pd.DataFrame:
    out = test.copy()
    out["take"] = False
    out["selector_score"] = 0.0

    if selector == "raw_stack":
        out["take"] = True
        out["selector_score"] = 1.0
        return out

    if selector.startswith("nn_top"):
        score_col = "nn_score"
        n = int(selector.replace("nn_top", ""))
        mode = "topn"
    elif selector.startswith("ensemble_top"):
        score_col = "ensemble_score"
        n = int(selector.replace("ensemble_top", ""))
        mode = "topn"
    elif selector.startswith("rule_top"):
        score_col = "rule_score"
        n = int(selector.replace("rule_top", ""))
        mode = "topn"
    elif selector.startswith("indicator_top"):
        score_col = "indicator_score"
        n = int(selector.replace("indicator_top", ""))
        mode = "topn"
    elif selector == "nn_positive":
        score_col = "nn_score"
        n = 6
        mode = "positive"
    elif selector == "ensemble_positive":
        score_col = "ensemble_score"
        n = 6
        mode = "positive"
    elif selector == "nn_above_median":
        score_col = "nn_score"
        n = 6
        mode = "above_median"
    elif selector == "ensemble_above_median":
        score_col = "ensemble_score"
        n = 6
        mode = "above_median"
    else:
        raise ValueError(f"unknown selector {selector}")

    for _, idx in out.groupby("entry_date").groups.items():
        w = out.loc[list(idx)].copy()
        if mode == "topn":
            chosen = w.sort_values(score_col, ascending=False).head(n).index
        elif mode == "positive":
            chosen = w[w[score_col] > 0].sort_values(score_col, ascending=False).head(n).index
        else:
            chosen = w[w[score_col] > w[score_col].median()].sort_values(score_col, ascending=False).head(n).index
        out.loc[chosen, "take"] = True
        out.loc[list(idx), "selector_score"] = out.loc[list(idx), score_col]
    return out


def max_dd(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min() * 100.0)


def make_html(out_dir: Path, summary: pd.DataFrame, best_nav: pd.DataFrame, market: pd.DataFrame, capital: float) -> Path:
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.65, 0.35],
        vertical_spacing=0.10,
        subplot_titles=("Top 12 equity curves", "Return vs drawdown for all sweeps"),
    )
    top = summary.sort_values("return_pct", ascending=False).head(12)
    for _, r in top.iterrows():
        key = r["run_key"]
        w = best_nav[best_nav["run_key"] == key]
        fig.add_trace(go.Scatter(x=w["date"], y=w["nav"], name=key, mode="lines"), row=1, col=1)
    bench = benchmark(market, capital)
    fig.add_trace(go.Scatter(x=bench["date"], y=bench["nav"], name="SPY benchmark", mode="lines", line=dict(color="#111", dash="dot", width=2.4)), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=summary["max_drawdown_pct"],
            y=summary["return_pct"],
            text=summary["run_key"],
            mode="markers",
            marker=dict(size=np.clip(summary["return_pct"].abs() / 10 + 8, 8, 34), color=summary["monthly_pct"], colorscale="Turbo", showscale=True),
            hovertemplate="<b>%{text}</b><br>Return %{y:.1f}%<br>DD %{x:.1f}%<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_xaxes(title_text="Max DD (%)", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    fig.update_layout(height=950, template="plotly_white", hovermode="x unified")

    table = summary.sort_values("return_pct", ascending=False).to_html(index=False, classes="data-table", escape=False)
    css = """
    <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin:24px; color:#17202a; }
    .scroll { overflow-x:auto; border:1px solid #eee; padding:8px; }
    table.data-table { border-collapse: collapse; width: 100%; font-size: 12px; }
    table.data-table th, table.data-table td { border: 1px solid #d6dbdf; padding: 6px 8px; text-align:right; white-space:nowrap; }
    table.data-table th { background:#f2f3f4; position:sticky; top:0; }
    table.data-table td:first-child, table.data-table th:first-child,
    table.data-table td:nth-child(2), table.data-table th:nth-child(2) { text-align:left; }
    </style>
    """
    path = out_dir / "aggressive_meta_sweep_report.html"
    path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'><title>Aggressive Meta Sweep</title>"
        + css
        + "</head><body><h1>Aggressive Meta-Engine Sweep</h1>"
        + "<p>70/30 held-out test, shared portfolio, integer contract sizing. Higher returns here are intentionally accepting much larger drawdowns and open risk.</p>"
        + fig.to_html(full_html=False, include_plotlyjs=True)
        + "<h2>All Sweep Results</h2><div class='scroll'>"
        + table
        + "</div></body></html>",
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--capital", type=float, default=100_000.0)
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"aggressive_meta_sweep_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = load_specs()
    train_market, train_trades = load_period(args.root, "train", specs)
    test_market, test_trades = load_period(args.root, "test", specs)
    scored_test, train_meta = train_scores(train_trades, test_trades)

    selectors = [
        "raw_stack",
        "nn_top1",
        "nn_top2",
        "nn_top3",
        "nn_top4",
        "nn_top5",
        "nn_top6",
        "nn_positive",
        "nn_above_median",
        "ensemble_top1",
        "ensemble_top2",
        "ensemble_top3",
        "ensemble_top4",
        "ensemble_top5",
        "ensemble_top6",
        "ensemble_positive",
        "ensemble_above_median",
        "rule_top2",
        "rule_top3",
        "rule_top4",
        "indicator_top2",
        "indicator_top3",
        "indicator_top4",
    ]
    risks = [10.0, 15.0, 20.0, 25.0, 30.0]
    concurrencies = [100.0, 150.0, 200.0, 250.0, 300.0]

    rows = []
    nav_frames = []
    trade_frames = []
    for selector in selectors:
        selected = select_custom(scored_test, selector)
        for risk in risks:
            for conc in concurrencies:
                run_key = f"{selector}__risk{risk:g}__cap{conc:g}"
                nav, trades, summary = replay_portfolio(
                    run_key,
                    test_market,
                    selected,
                    capital=args.capital,
                    risk_pct=risk,
                    concurrent_pct=conc,
                )
                summary["selector"] = selector
                summary["risk_pct"] = risk
                summary["concurrent_pct"] = conc
                summary["run_key"] = run_key
                rows.append(summary)
                nav["run_key"] = run_key
                trades["run_key"] = run_key
                nav_frames.append(nav)
                trade_frames.append(trades)

    summary_df = pd.DataFrame(rows).sort_values("return_pct", ascending=False)
    nav_df = pd.concat(nav_frames, ignore_index=True)
    trades_df = pd.concat(trade_frames, ignore_index=True)
    summary_df.to_csv(out_dir / "sweep_summary.csv", index=False)
    nav_df.to_csv(out_dir / "sweep_nav.csv", index=False)
    trades_df.to_csv(out_dir / "sweep_trades.csv", index=False)
    scored_test.to_csv(out_dir / "test_candidates_scored.csv", index=False)
    (out_dir / "train_meta.json").write_text(json.dumps(train_meta, indent=2))

    report = make_html(out_dir, summary_df, nav_df, test_market, args.capital)
    print(summary_df.head(20).to_string(index=False))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
