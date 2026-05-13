"""Build readable grouped charts for the SPY core strategy grid."""

from __future__ import annotations

import argparse
import html
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUN_DIR = REPO_ROOT / "experiments" / "runs" / "spy_core_30_strategy_grid_2019_2026"


@dataclass(frozen=True)
class GroupSpec:
    key: str
    title: str
    entry_mode: str
    gate: str
    note: str


GROUPS = [
    GroupSpec(
        key="single_reentry_vrp",
        title="Single Re-Entry: VRP Positive",
        entry_mode="single_reentry",
        gate="vrp_positive",
        note="Only one open trade at a time. Re-enters after the previous condor closes, when IV is above realized vol.",
    ),
    GroupSpec(
        key="single_reentry_ivrank",
        title="Single Re-Entry: IV Rank 50+",
        entry_mode="single_reentry",
        gate="iv_rank_50",
        note="Only one open trade at a time. Requires IV to be in the upper half of its trailing one-year range.",
    ),
    GroupSpec(
        key="daily_ladder_vrp",
        title="Daily Ladder: VRP Positive",
        entry_mode="daily_ladder",
        gate="vrp_positive",
        note="Can open a new condor every trading day while prior condors remain open. Uses a volatility-risk-premium gate.",
    ),
    GroupSpec(
        key="daily_ladder_ivrank",
        title="Daily Ladder: IV Rank 50+",
        entry_mode="daily_ladder",
        gate="iv_rank_50",
        note="Can open a new condor every trading day. Requires IV rank above 50%.",
    ),
    GroupSpec(
        key="mwf_ladder_highivvol",
        title="M/W/F Ladder: High IV + Volume",
        entry_mode="mwf_ladder",
        gate="high_iv_volume",
        note="Opens at most Monday, Wednesday, and Friday. Requires elevated IV and enough SPY volume.",
    ),
    GroupSpec(
        key="aggressive_daily_nogate",
        title="Aggressive Daily: No Gate",
        entry_mode="daily_ladder",
        gate="none",
        note="Most aggressive test. Opens daily without volatility filters, so returns and drawdowns are both larger.",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir
    out_dir = run_dir / "grouped"
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(run_dir / "summary.csv")
    nav = pd.read_csv(run_dir / "daily_nav.csv")
    trades = pd.read_csv(run_dir / "trades.csv")
    nav["date"] = pd.to_datetime(nav["date"])
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])

    written: list[tuple[GroupSpec, Path, pd.Series]] = []
    for group in GROUPS:
        group_summary = summary[
            (summary["entry_mode"] == group.entry_mode) & (summary["gate"] == group.gate)
        ].sort_values("final_nav", ascending=False)
        if group_summary.empty:
            continue
        group_nav = nav[nav["strategy"].isin(group_summary["strategy"])]
        group_trades = trades[trades["strategy"].isin(group_summary["strategy"])]
        path = out_dir / f"{group.key}.html"
        write_group_chart(group, group_summary, group_nav, group_trades, path)
        written.append((group, path, group_summary.iloc[0]))

    best_path = out_dir / "best_by_group.html"
    write_best_by_group_chart(nav, trades, written, best_path)
    index_path = out_dir / "index.html"
    write_index(written, best_path, index_path)

    print(index_path)
    for _group, path, _best in written:
        print(path)
    print(best_path)
    return 0


def write_group_chart(
    group: GroupSpec,
    summary: pd.DataFrame,
    nav: pd.DataFrame,
    trades: pd.DataFrame,
    out: Path,
) -> None:
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.58, 0.22, 0.20],
        vertical_spacing=0.09,
        specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "table"}]],
        subplot_titles=(group.title, "Closed trade P&L", "Ranked stats"),
    )
    palette = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#457b9d"]
    benchmark = benchmark_from_nav(nav)
    if not benchmark.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark["date"],
                y=benchmark["benchmark_nav"],
                mode="lines",
                name="SPY buy & hold benchmark",
                line=dict(color="#6b7280", width=3, dash="dash"),
                hovertemplate="SPY benchmark<br>Date %{x|%Y-%m-%d}<br>NAV $%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    for idx, row in summary.reset_index(drop=True).iterrows():
        strategy = row["strategy"]
        sdf = nav[nav["strategy"] == strategy].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=sdf["date"],
                y=sdf["nav"],
                mode="lines",
                name=f"{short_name(strategy)} (${row['final_nav']:,.0f})",
                line=dict(color=palette[idx % len(palette)], width=4 if idx == 0 else 2.4),
                opacity=1.0 if idx == 0 else 0.78,
                hovertemplate=(
                    f"{strategy}<br>"
                    "Date %{x|%Y-%m-%d}<br>NAV $%{y:,.2f}<br>"
                    f"Trades {int(row['trades'])}<br>"
                    f"Return {row['return_pct']:.1f}%<br>"
                    f"Max DD {row['max_drawdown_pct']:.1f}%<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        tdf = trades[trades["strategy"] == strategy].sort_values("exit_date")
        if not tdf.empty:
            fig.add_trace(
                go.Scatter(
                    x=tdf["exit_date"],
                    y=tdf["pnl_usd"],
                    mode="markers",
                    name=f"{short_name(strategy)} trades",
                    marker=dict(
                        color=palette[idx % len(palette)],
                        size=6 if len(tdf) > 1000 else 8,
                        opacity=0.55,
                    ),
                    customdata=tdf[["entry_date", "reason", "entry_credit", "exit_debit"]],
                    hovertemplate=(
                        f"{strategy}<br>"
                        "Entry %{customdata[0]|%Y-%m-%d}<br>"
                        "Exit %{x|%Y-%m-%d}<br>"
                        "P&L $%{y:,.2f}<br>"
                        "Reason %{customdata[1]}<br>"
                        "Credit %{customdata[2]:.2f} / exit debit %{customdata[3]:.2f}"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

    add_zero_trade_line(fig, trades, row=2)

    table = summary.copy()
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Rank", "Strategy", "Trades", "Final NAV", "Return", "Max DD", "Sharpe", "Win rate", "Max open"],
                fill_color="#264653",
                font=dict(color="white", size=13),
                align="left",
            ),
            cells=dict(
                values=[
                    list(range(1, len(table) + 1)),
                    [short_name(v) for v in table["strategy"]],
                    table["trades"].astype(int),
                    [f"${v:,.0f}" for v in table["final_nav"]],
                    [f"{v:+.1f}%" for v in table["return_pct"]],
                    [f"{v:.1f}%" for v in table["max_drawdown_pct"]],
                    [f"{v:.2f}" for v in table["sharpe"]],
                    [f"{v:.1f}%" for v in table["win_rate_pct"]],
                    table["max_open_trades"].astype(int),
                ],
                fill_color="white",
                align="left",
                height=27,
            ),
        ),
        row=3,
        col=1,
    )

    best = summary.iloc[0]
    fig.update_layout(
        title=dict(
            text=(
                f"{group.title}<br><sup>{group.note}<br>"
                f"Best: {short_name(best['strategy'])} | ${best['final_nav']:,.2f} | "
                f"{best['return_pct']:+.1f}% | trades {int(best['trades'])} | "
                f"max DD {best['max_drawdown_pct']:.1f}%</sup>"
            ),
            x=0.02,
            xanchor="left",
        ),
        width=1600,
        height=1120,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=80, r=50, t=135, b=50),
    )
    fig.update_yaxes(title_text="NAV ($)", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Trade P&L ($)", tickprefix="$", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Exit date", row=2, col=1)
    fig.write_html(out, include_plotlyjs="cdn")


def write_best_by_group_chart(
    nav: pd.DataFrame,
    trades: pd.DataFrame,
    written: list[tuple[GroupSpec, Path, pd.Series]],
    out: Path,
) -> None:
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.58, 0.22, 0.20],
        vertical_spacing=0.10,
        specs=[[{"type": "scatter"}], [{"type": "scatter"}], [{"type": "table"}]],
        subplot_titles=("Best strategy from each group", "Closed trade P&L for each group winner", "Best-by-group stats"),
    )
    palette = ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51", "#457b9d"]
    benchmark = benchmark_from_nav(nav)
    if not benchmark.empty:
        fig.add_trace(
            go.Scatter(
                x=benchmark["date"],
                y=benchmark["benchmark_nav"],
                mode="lines",
                name="SPY buy & hold benchmark",
                line=dict(color="#6b7280", width=3, dash="dash"),
                hovertemplate="SPY benchmark<br>Date %{x|%Y-%m-%d}<br>NAV $%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    rows = []
    for idx, (group, _path, best) in enumerate(written):
        strategy = best["strategy"]
        sdf = nav[nav["strategy"] == strategy].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=sdf["date"],
                y=sdf["nav"],
                mode="lines",
                name=f"{group.title}: ${best['final_nav']:,.0f}",
                line=dict(color=palette[idx % len(palette)], width=3),
                hovertemplate=(
                    f"{group.title}<br>{strategy}<br>"
                    "Date %{x|%Y-%m-%d}<br>NAV $%{y:,.2f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        tdf = trades[trades["strategy"] == strategy].sort_values("exit_date")
        if not tdf.empty:
            fig.add_trace(
                go.Scatter(
                    x=tdf["exit_date"],
                    y=tdf["pnl_usd"],
                    mode="markers",
                    name=f"{group.title} trades",
                    marker=dict(color=palette[idx % len(palette)], size=7, opacity=0.55),
                    customdata=tdf[["strategy", "entry_date", "reason"]],
                    hovertemplate=(
                        "%{customdata[0]}<br>"
                        "Entry %{customdata[1]|%Y-%m-%d}<br>"
                        "Exit %{x|%Y-%m-%d}<br>"
                        "P&L $%{y:,.2f}<br>"
                        "Reason %{customdata[2]}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=2,
                col=1,
            )
        rows.append((group.title, best))

    best_strategies = [best["strategy"] for _group, _path, best in written]
    add_zero_trade_line(fig, trades[trades["strategy"].isin(best_strategies)], row=2)

    fig.add_trace(
        go.Table(
            header=dict(
                values=["Group", "Best strategy", "Trades", "Final NAV", "Return", "Max DD", "Sharpe"],
                fill_color="#264653",
                font=dict(color="white", size=13),
                align="left",
            ),
            cells=dict(
                values=[
                    [r[0] for r in rows],
                    [short_name(r[1]["strategy"]) for r in rows],
                    [int(r[1]["trades"]) for r in rows],
                    [f"${r[1]['final_nav']:,.0f}" for r in rows],
                    [f"{r[1]['return_pct']:+.1f}%" for r in rows],
                    [f"{r[1]['max_drawdown_pct']:.1f}%" for r in rows],
                    [f"{r[1]['sharpe']:.2f}" for r in rows],
                ],
                fill_color="white",
                align="left",
                height=28,
            ),
        ),
        row=3,
        col=1,
    )
    fig.update_layout(
        title=dict(text="SPY Core Grid: Best Strategy From Each Group", x=0.02, xanchor="left"),
        width=1500,
        height=1080,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=80, r=50, t=120, b=50),
    )
    fig.update_yaxes(title_text="NAV ($)", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Trade P&L ($)", tickprefix="$", row=2, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(title_text="Exit date", row=2, col=1)
    fig.write_html(out, include_plotlyjs="cdn")


def add_zero_trade_line(fig: go.Figure, trades: pd.DataFrame, *, row: int) -> None:
    if trades.empty:
        return
    fig.add_trace(
        go.Scatter(
            x=[trades["exit_date"].min(), trades["exit_date"].max()],
            y=[0, 0],
            mode="lines",
            line=dict(color="#9ca3af", width=1, dash="dot"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )


def benchmark_from_nav(nav: pd.DataFrame) -> pd.DataFrame:
    benchmark = (
        nav[["date", "spy_close"]]
        .drop_duplicates("date")
        .sort_values("date")
        .dropna(subset=["spy_close"])
        .copy()
    )
    if benchmark.empty:
        return benchmark
    first = float(benchmark["spy_close"].iloc[0])
    benchmark["benchmark_nav"] = 10_000.0 * benchmark["spy_close"].astype(float) / first
    return benchmark


def write_index(
    written: list[tuple[GroupSpec, Path, pd.Series]],
    best_path: Path,
    index_path: Path,
) -> None:
    cards = []
    for group, path, best in written:
        cards.append(
            f"""
            <a class="card" href="{html.escape(path.name)}">
              <h2>{html.escape(group.title)}</h2>
              <p>{html.escape(group.note)}</p>
              <dl>
                <dt>Strategies</dt><dd>{strategy_count(group)}</dd>
                <dt>Best</dt><dd>{html.escape(short_name(best['strategy']))}</dd>
                <dt>Final NAV</dt><dd>${best['final_nav']:,.0f}</dd>
                <dt>Return</dt><dd>{best['return_pct']:+.1f}%</dd>
                <dt>Max DD</dt><dd>{best['max_drawdown_pct']:.1f}%</dd>
              </dl>
            </a>
            """
        )
    index_path.write_text(
        f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SPY Core Grouped Charts</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2933; }}
    h1 {{ margin: 0 0 8px; }}
    .sub {{ color: #52616b; margin-bottom: 24px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 16px; }}
    .card {{ display: block; border: 1px solid #d8dee4; border-radius: 8px; padding: 18px; color: inherit; text-decoration: none; background: #fff; }}
    .card:hover {{ border-color: #2a9d8f; box-shadow: 0 8px 24px rgba(0,0,0,0.08); }}
    h2 {{ font-size: 18px; margin: 0 0 8px; }}
    p {{ color: #52616b; min-height: 44px; }}
    dl {{ display: grid; grid-template-columns: 110px 1fr; gap: 6px 10px; margin: 0; }}
    dt {{ color: #52616b; }}
    dd {{ margin: 0; font-weight: 650; }}
    .top {{ display: inline-block; margin: 0 0 22px; padding: 10px 14px; border-radius: 8px; background: #264653; color: #fff; text-decoration: none; }}
  </style>
</head>
<body>
  <h1>SPY Core Strategy Grid, Grouped</h1>
  <div class="sub">30 strategies split into 6 readable charts. Each chart includes SPY buy-and-hold benchmark plus closed-trade P&L. Proxy test from 2019-01-31 to 2026-05-08.</div>
  <a class="top" href="{html.escape(best_path.name)}">Open best strategy from each group</a>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
""",
        encoding="utf-8",
    )


def strategy_count(group: GroupSpec) -> int:
    if group.key in {"mwf_ladder_highivvol", "aggressive_daily_nogate"}:
        return 3
    return 6


def short_name(strategy: str) -> str:
    return (
        strategy.replace("single_reentry_", "single ")
        .replace("daily_ladder_", "daily ")
        .replace("mwf_ladder_", "mwf ")
        .replace("aggressive_daily_", "aggressive ")
        .replace("vrp_positive_", "VRP ")
        .replace("iv_rank_50_", "IVrank ")
        .replace("highivvol_", "highIVvol ")
        .replace("nogate_", "noGate ")
        .replace("_", " ")
    )


if __name__ == "__main__":
    raise SystemExit(main())
