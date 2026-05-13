"""Browser report for the six aggressive SPY core candidates.

Input is a `run_spy_core_strategy_grid.py` output directory. The report keeps
all six strategies plus an SPY buy-and-hold benchmark in one HTML file and
shows:
  - equity curves vs benchmark,
  - per-strategy entry/exit markers,
  - exit reasons,
  - stop-loss/profit-target debit references,
  - trade-level P&L tables.

This is still the SPY core proxy engine: daily SPY close + IV proxy + Black-
Scholes condor marks. It is not real IBKR fill data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN = REPO_ROOT / "experiments" / "runs" / "aggressive_six_browser_20260510T233940Z"
SPEC_JSON = REPO_ROOT / "fullonriskmodelterminator" / "additional_six_spy_core_specs.json"


NOTES = {
    "aggressive_daily_nogate_d20_30d_w10": "highest original aggressive no-gate",
    "aggressive_daily_nogate_d16_30d_w10": "similar, still huge DD",
    "daily_ladder_vrp_positive_d16_45d_w5": "high return, less wild",
    "daily_ladder_vrp_positive_d16_30d_w5": "probably the one you mean",
    "risk_qwen_risk_designer_risktest_d12_30d_lowdelta_pt30_sl20_c10": "best council risk-adjusted candidate",
    "execution_llama_allocator_designer_daily_ladder_vrp_positive_d12_30d_w5_stop_loss_1_5": "another council candidate",
}

COLORS = {
    "aggressive_daily_nogate_d20_30d_w10": "#d35400",
    "aggressive_daily_nogate_d16_30d_w10": "#e67e22",
    "daily_ladder_vrp_positive_d16_45d_w5": "#27ae60",
    "daily_ladder_vrp_positive_d16_30d_w5": "#2ecc71",
    "risk_qwen_risk_designer_risktest_d12_30d_lowdelta_pt30_sl20_c10": "#2980b9",
    "execution_llama_allocator_designer_daily_ladder_vrp_positive_d12_30d_w5_stop_loss_1_5": "#8e44ad",
    "benchmark": "#111111",
}


def pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x:+.2f}%"


def usd(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"${x:,.2f}"


def load_specs(path: Path) -> dict[str, dict[str, Any]]:
    raw = json.loads(path.read_text())
    return {s["name"]: s for s in raw["strategies"]}


def add_refs(trades: pd.DataFrame, specs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = trades.copy()
    out["entry_date"] = pd.to_datetime(out["entry_date"])
    out["exit_date"] = pd.to_datetime(out["exit_date"])
    out["expiry"] = pd.to_datetime(out["expiry"])
    out["holding_days"] = (out["exit_date"] - out["entry_date"]).dt.days
    out["profit_take"] = out["strategy"].map(lambda s: float(specs[s]["profit_take"]))
    out["stop_loss_credit_mult"] = out["strategy"].map(lambda s: float(specs[s]["stop_loss_credit_mult"]))
    out["wing_width"] = out["strategy"].map(lambda s: float(specs[s]["wing_width"]))
    out["profit_target_debit"] = out["entry_credit"] * (1.0 - out["profit_take"])
    out["stop_loss_debit"] = out["entry_credit"] * (1.0 + out["stop_loss_credit_mult"])
    out["max_loss_usd"] = (out["wing_width"] - out["entry_credit"]) * 100.0
    out["pnl_pct_of_max_loss"] = out["pnl_usd"] / out["max_loss_usd"].replace(0, np.nan) * 100.0
    out["entry_structure"] = (
        "LP "
        + out["long_put"].map(lambda x: f"{x:.0f}")
        + " / SP "
        + out["short_put"].map(lambda x: f"{x:.0f}")
        + " / SC "
        + out["short_call"].map(lambda x: f"{x:.0f}")
        + " / LC "
        + out["long_call"].map(lambda x: f"{x:.0f}")
    )
    return out


def benchmark_nav(nav: pd.DataFrame) -> pd.DataFrame:
    base = (
        nav[["date", "spy_close"]]
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    base["date"] = pd.to_datetime(base["date"])
    first = float(base["spy_close"].iloc[0])
    base["nav"] = 10_000.0 * base["spy_close"] / first
    base["strategy"] = "SPY buy-and-hold benchmark"
    return base


def max_drawdown(series: pd.Series) -> float:
    peak = series.cummax()
    dd = series / peak - 1.0
    return float(dd.min() * 100.0)


def summary_with_benchmark(summary: pd.DataFrame, bench: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    bench_return = (float(bench["nav"].iloc[-1]) / float(bench["nav"].iloc[0]) - 1.0) * 100.0
    bench_row = {
        "strategy": "SPY buy-and-hold benchmark",
        "entry_mode": "benchmark",
        "gate": "none",
        "target_delta": np.nan,
        "target_dte": np.nan,
        "wing_width": np.nan,
        "trades": 1,
        "final_nav": float(bench["nav"].iloc[-1]),
        "return_pct": bench_return,
        "max_drawdown_pct": max_drawdown(bench["nav"]),
        "sharpe": np.nan,
        "win_rate_pct": np.nan,
        "avg_trade_usd": np.nan,
        "worst_trade_usd": np.nan,
        "max_open_trades": 1,
    }
    out = pd.concat([out, pd.DataFrame([bench_row])], ignore_index=True)
    return out


def overview_chart(nav: pd.DataFrame, bench: pd.DataFrame, summary: pd.DataFrame) -> str:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        row_heights=[0.72, 0.28],
        vertical_spacing=0.12,
        subplot_titles=("Equity curves: six aggressive candidates vs SPY benchmark", "Return vs max drawdown"),
    )
    for strat in summary["strategy"].tolist():
        w = nav[nav["strategy"] == strat].copy()
        if w.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=pd.to_datetime(w["date"]),
                y=w["nav"],
                name=strat,
                mode="lines",
                line=dict(color=COLORS.get(strat, "#555555"), width=1.8),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(
            x=bench["date"],
            y=bench["nav"],
            name="SPY benchmark",
            mode="lines",
            line=dict(color=COLORS["benchmark"], width=2.8, dash="dot"),
        ),
        row=1,
        col=1,
    )
    display = summary_with_benchmark(summary, bench)
    fig.add_trace(
        go.Scatter(
            x=display["max_drawdown_pct"],
            y=display["return_pct"],
            text=display["strategy"],
            mode="markers+text",
            textposition="top center",
            marker=dict(
                size=np.clip(display["return_pct"].abs() / 15.0 + 10, 10, 44),
                color=[
                    COLORS["benchmark"] if s == "SPY buy-and-hold benchmark" else COLORS.get(s, "#666666")
                    for s in display["strategy"]
                ],
                opacity=0.85,
            ),
            hovertemplate="<b>%{text}</b><br>Return %{y:.1f}%<br>Max DD %{x:.1f}%<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="NAV ($10k start)", row=1, col=1)
    fig.update_xaxes(title_text="Max drawdown (%)", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    fig.update_layout(
        height=880,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0),
        margin=dict(l=70, r=30, t=100, b=50),
    )
    return fig.to_html(include_plotlyjs=True, full_html=False)


def strategy_chart(strategy: str, nav: pd.DataFrame, trades: pd.DataFrame, bench: pd.DataFrame, spec: dict[str, Any]) -> str:
    w_nav = nav[nav["strategy"] == strategy].copy()
    w_nav["date"] = pd.to_datetime(w_nav["date"])
    w_tr = trades[trades["strategy"] == strategy].copy()
    color = COLORS.get(strategy, "#555555")

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.37, 0.33, 0.30],
        vertical_spacing=0.055,
        subplot_titles=(
            "NAV vs benchmark",
            "SPY with entries/exits and short-strike corridor",
            "Trade P&L by exit reason",
        ),
    )
    fig.add_trace(
        go.Scatter(x=w_nav["date"], y=w_nav["nav"], name=strategy, mode="lines", line=dict(color=color, width=2.2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=bench["date"],
            y=bench["nav"],
            name="SPY benchmark",
            mode="lines",
            line=dict(color=COLORS["benchmark"], width=1.8, dash="dot"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=w_nav["date"], y=w_nav["spy_close"], name="SPY close", mode="lines", line=dict(color="#111111", width=1.7)),
        row=2,
        col=1,
    )
    # Use entries as corridor markers; thousands of daily ladder points are easier to read as low-opacity markers.
    fig.add_trace(
        go.Scatter(
            x=w_tr["entry_date"],
            y=w_tr["short_call"],
            name="short call strike",
            mode="markers",
            marker=dict(color="#c0392b", size=4, opacity=0.22),
            hovertemplate="Entry %{x|%Y-%m-%d}<br>Short call %{y:.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=w_tr["entry_date"],
            y=w_tr["short_put"],
            name="short put strike",
            mode="markers",
            marker=dict(color="#27ae60", size=4, opacity=0.22),
            hovertemplate="Entry %{x|%Y-%m-%d}<br>Short put %{y:.0f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    sample = w_tr.iloc[:: max(1, len(w_tr) // 600)].copy()
    fig.add_trace(
        go.Scatter(
            x=sample["entry_date"],
            y=sample["entry_spy"],
            name="entries",
            mode="markers",
            marker=dict(color=color, size=7, symbol="triangle-up", opacity=0.75),
            customdata=np.stack(
                [
                    sample["exit_date"].dt.date.astype(str),
                    sample["reason"],
                    sample["entry_credit"].map(lambda x: f"{x:.2f}"),
                    sample["profit_target_debit"].map(lambda x: f"{x:.2f}"),
                    sample["stop_loss_debit"].map(lambda x: f"{x:.2f}"),
                    sample["pnl_usd"].map(lambda x: f"${x:,.2f}"),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "Entry %{x|%Y-%m-%d}<br>SPY $%{y:.2f}<br>"
                "Exit %{customdata[0]}<br>Reason %{customdata[1]}<br>"
                "Credit %{customdata[2]}<br>Profit target debit %{customdata[3]}<br>"
                "Stop debit %{customdata[4]}<br>P&L %{customdata[5]}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sample["exit_date"],
            y=sample["exit_spy"],
            name="exits",
            mode="markers",
            marker=dict(color="#111111", size=6, symbol="x", opacity=0.72),
            hovertemplate="Exit %{x|%Y-%m-%d}<br>SPY $%{y:.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )

    reason_colors = {
        "profit_target": "#27ae60",
        "stop_loss": "#c0392b",
        "dte_close": "#2980b9",
        "end_forced": "#7f8c8d",
    }
    for reason, sub in w_tr.groupby("reason"):
        fig.add_trace(
            go.Scatter(
                x=sub["exit_date"],
                y=sub["pnl_usd"],
                name=reason,
                mode="markers",
                marker=dict(color=reason_colors.get(reason, "#8e44ad"), size=5, opacity=0.62),
                hovertemplate="Exit %{x|%Y-%m-%d}<br>P&L $%{y:.2f}<extra></extra>",
            ),
            row=3,
            col=1,
        )
    fig.add_hline(y=0, row=3, col=1, line_width=1, line_color="#555555")
    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="SPY / strikes", row=2, col=1)
    fig.update_yaxes(title_text="Trade P&L", row=3, col=1)
    fig.update_layout(
        height=1050,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=70, r=30, t=110, b=50),
        title=dict(
            text=(
                f"{strategy}<br>"
                f"<sup>{NOTES.get(strategy, '')} | delta {spec['target_delta']} | DTE {spec['target_dte']} | "
                f"wing ${spec['wing_width']} | PT {spec['profit_take']:.0%} | stop {spec['stop_loss_credit_mult']}x credit</sup>"
            ),
            x=0.01,
            xanchor="left",
        ),
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)


def summary_table(summary: pd.DataFrame, bench: pd.DataFrame) -> str:
    display = summary_with_benchmark(summary, bench)
    rows = []
    for _, r in display.iterrows():
        strat = r["strategy"]
        rows.append(
            {
                "strategy": strat,
                "return": pct(float(r["return_pct"])),
                "max_dd": pct(float(r["max_drawdown_pct"])),
                "trades": int(r["trades"]) if pd.notna(r["trades"]) else "",
                "win_rate": "" if pd.isna(r["win_rate_pct"]) else f"{float(r['win_rate_pct']):.1f}%",
                "final_nav": usd(float(r["final_nav"])),
                "worst_trade": "" if pd.isna(r["worst_trade_usd"]) else usd(float(r["worst_trade_usd"])),
                "max_open": int(r["max_open_trades"]) if pd.notna(r["max_open_trades"]) else "",
                "note": NOTES.get(strat, "benchmark"),
            }
        )
    return pd.DataFrame(rows).to_html(index=False, escape=False, classes="data-table")


def reason_table(trades: pd.DataFrame) -> str:
    tab = (
        trades.groupby(["strategy", "reason"])
        .agg(trades=("pnl_usd", "size"), pnl_usd=("pnl_usd", "sum"), avg_pnl=("pnl_usd", "mean"))
        .reset_index()
    )
    tab["pnl_usd"] = tab["pnl_usd"].map(usd)
    tab["avg_pnl"] = tab["avg_pnl"].map(usd)
    return tab.to_html(index=False, escape=False, classes="data-table")


def trade_table(strategy: str, trades: pd.DataFrame, max_rows: int = 300) -> str:
    w = trades[trades["strategy"] == strategy].copy()
    # Show every row up to max_rows, but prioritize largest winners/losers and most recent.
    if len(w) > max_rows:
        worst = w.nsmallest(max_rows // 3, "pnl_usd")
        best = w.nlargest(max_rows // 3, "pnl_usd")
        recent = w.sort_values("exit_date").tail(max_rows - len(worst) - len(best))
        w = pd.concat([worst, best, recent]).drop_duplicates().sort_values("exit_date")
    out = pd.DataFrame(
        {
            "entry": w["entry_date"].dt.date.astype(str),
            "exit": w["exit_date"].dt.date.astype(str),
            "expiry": w["expiry"].dt.date.astype(str),
            "reason": w["reason"],
            "structure": w["entry_structure"],
            "entry_spy": w["entry_spy"].map(usd),
            "exit_spy": w["exit_spy"].map(usd),
            "credit": w["entry_credit"].map(lambda x: f"{x:.2f}"),
            "profit_target_debit": w["profit_target_debit"].map(lambda x: f"{x:.2f}"),
            "stop_loss_debit": w["stop_loss_debit"].map(lambda x: f"{x:.2f}"),
            "exit_debit": w["exit_debit"].map(lambda x: f"{x:.2f}"),
            "pnl": w["pnl_usd"].map(usd),
            "pnl_pct_max_loss": w["pnl_pct_of_max_loss"].map(lambda x: f"{x:+.1f}%"),
        }
    )
    return out.to_html(index=False, escape=False, classes="data-table")


def build_report(run_dir: Path, out_path: Path, spec_path: Path) -> Path:
    specs = load_specs(spec_path)
    nav = pd.read_csv(run_dir / "daily_nav.csv")
    trades = pd.read_csv(run_dir / "trades.csv")
    summary = pd.read_csv(run_dir / "summary.csv")
    nav["date"] = pd.to_datetime(nav["date"])
    trades = add_refs(trades, specs)
    bench = benchmark_nav(nav)

    strategy_order = summary["strategy"].tolist()

    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #17202a; }
      h1 { margin-bottom: 4px; }
      h2 { margin-top: 38px; padding-top: 20px; border-top: 1px solid #d7dbdd; }
      h3 { margin-top: 24px; }
      .note { color: #566573; max-width: 1160px; line-height: 1.45; }
      .navlinks a { margin-right: 14px; font-size: 13px; }
      .scroll { overflow-x: auto; border: 1px solid #eee; padding: 8px; margin-bottom: 20px; }
      table.data-table { border-collapse: collapse; width: 100%; font-size: 12px; }
      table.data-table th, table.data-table td { border: 1px solid #d6dbdf; padding: 6px 8px; text-align: right; white-space: nowrap; }
      table.data-table th { background: #f2f3f4; position: sticky; top: 0; z-index: 1; }
      table.data-table td:first-child, table.data-table th:first-child { text-align: left; }
      table.data-table td:nth-child(2), table.data-table th:nth-child(2),
      table.data-table td:nth-child(3), table.data-table th:nth-child(3),
      table.data-table td:nth-child(4), table.data-table th:nth-child(4),
      table.data-table td:nth-child(5), table.data-table th:nth-child(5) { text-align: left; }
      .small { font-size: 12px; color: #566573; }
    </style>
    """
    body = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Aggressive Six Strategy Report</title>",
        css,
        "</head><body>",
        "<h1>Aggressive Six Strategy Report</h1>",
        f"<p class='note'>Source run: <code>{run_dir}</code>. This keeps all six requested aggressive SPY iron-condor proxy strategies plus SPY buy-and-hold benchmark in one browser report. Entries/exits and P&L come from <code>run_spy_core_strategy_grid.py</code>. Stop-loss and profit-target levels are option-combo debit thresholds, not broker stop orders.</p>",
        "<div class='navlinks'>",
    ]
    for strat in strategy_order:
        body.append(f"<a href='#{strat}'>{strat}</a>")
    body.extend(["</div>", "<h2>Overview</h2>", overview_chart(nav, bench, summary), "<h3>Summary + Benchmark</h3><div class='scroll'>", summary_table(summary, bench), "</div>", "<h3>Exit Reason Breakdown</h3><div class='scroll'>", reason_table(trades), "</div>"])

    for strat in strategy_order:
        body.extend(
            [
                f"<h2 id='{strat}'>{strat}</h2>",
                strategy_chart(strat, nav, trades, bench, specs[strat]),
                "<p class='small'>Trade table is capped to 300 rows for browser speed: largest losers, largest winners, and most recent trades are included. Full trade log is in <code>trades.csv</code>.</p>",
                "<div class='scroll'>",
                trade_table(strat, trades),
                "</div>",
            ]
        )

    body.extend(["</body></html>"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(body), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--spec-json", type=Path, default=SPEC_JSON)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.run_dir / "aggressive_six_browser_report.html"
    path = build_report(args.run_dir, out, args.spec_json)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
