"""Final pre-deploy simulation report for the paper_v1 strategy.

This does not invent a new strategy. It reads the actual held-out Year 2
evidence trades from data/trading_system.db and renders a deploy-style report:

- $10k realized NAV path
- SPY benchmark over the same dates
- monthly returns
- sleeve contribution
- trade log with entry/exit markers

The NAV path is realized-P&L based because the trade ledger stores realized
exits, not full daily mark-to-market option marks.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = REPO_ROOT / "data" / "trading_system.db"
YEAR2_SPY_PATH = REPO_ROOT / "data" / "cache" / "year2" / "spy_daily.parquet"
DEFAULT_RUN_ID = "paper_v1_evidence_20260510T182714Z"

GREEN = "#16803c"
RED = "#b42318"
BLUE = "#1f6feb"
ORANGE = "#b35c00"
GRAY = "#667085"
DARK = "#111827"


def read_trades(run_id: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    try:
        trades = pd.read_sql(
            """
            SELECT
                trade_id,
                sleeve,
                strategy_type,
                underlying_ticker,
                entry_at_utc,
                exit_at_utc,
                commission_usd,
                spread_cost_usd,
                slippage_usd,
                realized_pnl_usd,
                notes
            FROM trades
            WHERE run_id = ?
              AND realized_pnl_usd IS NOT NULL
              AND exit_at_utc IS NOT NULL
            ORDER BY entry_at_utc, trade_id
            """,
            conn,
            params=(run_id,),
        )
    finally:
        conn.close()
    if trades.empty:
        raise RuntimeError(f"no realized trades found for {run_id}")
    trades["entry_dt"] = pd.to_datetime(trades["entry_at_utc"], utc=True).dt.tz_convert(None)
    trades["exit_dt"] = pd.to_datetime(trades["exit_at_utc"], utc=True).dt.tz_convert(None)
    trades["entry_date"] = trades["entry_dt"].dt.normalize()
    trades["exit_date"] = trades["exit_dt"].dt.normalize()
    return trades


def read_spy(start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if not YEAR2_SPY_PATH.exists():
        raise RuntimeError(f"missing local SPY Year 2 cache: {YEAR2_SPY_PATH}")
    spy = pd.read_parquet(YEAR2_SPY_PATH).copy()
    if "date" in spy.columns:
        spy["date"] = pd.to_datetime(spy["date"]).dt.tz_localize(None).dt.normalize()
    else:
        spy = spy.reset_index().rename(columns={"index": "date"})
        spy["date"] = pd.to_datetime(spy["date"]).dt.tz_localize(None).dt.normalize()
    close_col = "close" if "close" in spy.columns else "Close"
    out = spy[["date", close_col]].rename(columns={close_col: "spy_close"}).dropna()
    out = out[(out["date"] >= start.normalize()) & (out["date"] <= end.normalize())].copy()
    if out.empty:
        raise RuntimeError("SPY benchmark cache had no rows in evidence window")
    return out.drop_duplicates("date").sort_values("date")


def realized_nav(trades: pd.DataFrame, initial_nav: float) -> pd.DataFrame:
    start = trades["entry_date"].min()
    end = trades["exit_date"].max()
    days = pd.date_range(start, end, freq="D")
    exit_pnl = trades.groupby("exit_date")["realized_pnl_usd"].sum()
    rows = []
    nav = initial_nav
    for day in days:
        nav += float(exit_pnl.get(day, 0.0))
        rows.append({"date": day, "nav": nav, "daily_realized_pnl": float(exit_pnl.get(day, 0.0))})
    nav_df = pd.DataFrame(rows)
    nav_df["peak"] = nav_df["nav"].cummax()
    nav_df["drawdown_pct"] = (nav_df["nav"] / nav_df["peak"] - 1.0) * 100.0
    nav_df["cum_pnl"] = nav_df["nav"] - initial_nav
    return nav_df


def add_benchmark(nav_df: pd.DataFrame, spy: pd.DataFrame, initial_nav: float) -> pd.DataFrame:
    merged = nav_df.merge(spy, on="date", how="left").sort_values("date")
    merged["spy_close"] = merged["spy_close"].ffill().bfill()
    start_px = float(merged["spy_close"].iloc[0])
    merged["spy_benchmark"] = merged["spy_close"] / start_px * initial_nav
    merged["spy_return_pct"] = (merged["spy_benchmark"] / initial_nav - 1.0) * 100.0
    merged["spy_peak"] = merged["spy_benchmark"].cummax()
    merged["spy_drawdown_pct"] = (merged["spy_benchmark"] / merged["spy_peak"] - 1.0) * 100.0
    return merged


def monthly_table(nav_df: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    month_end = nav_df.set_index("date")["nav"].resample("ME").last().dropna()
    month_start = nav_df.set_index("date")["nav"].resample("MS").first().reindex(month_end.index, method="ffill")
    monthly = pd.DataFrame(
        {
            "month": month_end.index.strftime("%Y-%m"),
            "start_nav": month_start.values,
            "end_nav": month_end.values,
        }
    )
    monthly["return_pct"] = (monthly["end_nav"] / monthly["start_nav"] - 1.0) * 100.0
    t = trades.copy()
    t["exit_month"] = t["exit_date"].dt.strftime("%Y-%m")
    counts = t.groupby("exit_month").agg(trades=("trade_id", "size"), pnl=("realized_pnl_usd", "sum"))
    monthly = monthly.merge(counts, left_on="month", right_index=True, how="left")
    monthly[["trades", "pnl"]] = monthly[["trades", "pnl"]].fillna({"trades": 0, "pnl": 0.0})
    monthly["trades"] = monthly["trades"].astype(int)
    return monthly


def metrics(run_id: str, trades: pd.DataFrame, nav_df: pd.DataFrame, monthly: pd.DataFrame, initial_nav: float) -> dict[str, Any]:
    final_nav = float(nav_df["nav"].iloc[-1])
    duration_days = int((nav_df["date"].iloc[-1] - nav_df["date"].iloc[0]).days) + 1
    months = duration_days / 30.4375
    total_return = final_nav / initial_nav - 1.0
    geometric_monthly = (final_nav / initial_nav) ** (1 / months) - 1.0 if months > 0 and final_nav > 0 else float("nan")
    wins = trades[trades["realized_pnl_usd"] > 0]
    losses = trades[trades["realized_pnl_usd"] < 0]
    sleeve = (
        trades.groupby(["sleeve", "strategy_type"])["realized_pnl_usd"]
        .agg(["count", "sum", "mean", "min", "max"])
        .reset_index()
    )
    return {
        "run_id": run_id,
        "initial_nav": initial_nav,
        "final_nav": final_nav,
        "total_pnl": final_nav - initial_nav,
        "total_return_pct": total_return * 100.0,
        "geometric_monthly_return_pct": geometric_monthly * 100.0,
        "max_drawdown_pct": float(nav_df["drawdown_pct"].min()),
        "duration_days": duration_days,
        "trade_count": int(len(trades)),
        "win_count": int(len(wins)),
        "loss_count": int(len(losses)),
        "win_rate_pct": float(len(wins) / len(trades) * 100.0),
        "avg_win_usd": float(wins["realized_pnl_usd"].mean()) if len(wins) else 0.0,
        "avg_loss_usd": float(losses["realized_pnl_usd"].mean()) if len(losses) else 0.0,
        "best_trade_usd": float(trades["realized_pnl_usd"].max()),
        "worst_trade_usd": float(trades["realized_pnl_usd"].min()),
        "best_month_pct": float(monthly["return_pct"].max()),
        "worst_month_pct": float(monthly["return_pct"].min()),
        "positive_months": int((monthly["return_pct"] > 0).sum()),
        "negative_months": int((monthly["return_pct"] < 0).sum()),
        "sleeve_breakdown": sleeve.to_dict(orient="records"),
        "report_caveat": "Realized-P&L NAV only; not full daily option mark-to-market.",
    }


def write_report(
    *,
    out_dir: Path,
    run_id: str,
    trades: pd.DataFrame,
    nav_df: pd.DataFrame,
    monthly: pd.DataFrame,
    summary: dict[str, Any],
) -> Path:
    trade_marker = trades.merge(nav_df[["date", "nav"]], left_on="exit_date", right_on="date", how="left")
    trade_marker["color"] = trade_marker["realized_pnl_usd"].map(lambda x: GREEN if x >= 0 else RED)
    monthly["color"] = monthly["return_pct"].map(lambda x: GREEN if x >= 0 else RED)

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.07,
        row_heights=[0.38, 0.22, 0.22, 0.18],
        subplot_titles=(
            "Realized NAV vs SPY Benchmark",
            "Drawdown",
            "Trade P&L by Exit Date",
            "Monthly Returns",
        ),
    )
    fig.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["nav"], name="paper_v1 realized NAV", line={"color": BLUE, "width": 3}), row=1, col=1)
    fig.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["spy_benchmark"], name="SPY benchmark", line={"color": GRAY, "width": 2, "dash": "dash"}), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=trade_marker["exit_date"],
            y=trade_marker["nav"],
            mode="markers",
            name="trade exits",
            marker={"color": trade_marker["color"], "size": 8, "symbol": "diamond"},
            customdata=trade_marker[["trade_id", "sleeve", "strategy_type", "underlying_ticker", "realized_pnl_usd"]],
            hovertemplate="trade %{customdata[0]}<br>%{customdata[1]} / %{customdata[2]} / %{customdata[3]}<br>P&L $%{customdata[4]:,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["drawdown_pct"], name="strategy DD", fill="tozeroy", line={"color": RED}), row=2, col=1)
    fig.add_trace(go.Scatter(x=nav_df["date"], y=nav_df["spy_drawdown_pct"], name="SPY DD", line={"color": GRAY, "dash": "dash"}), row=2, col=1)
    fig.add_trace(go.Bar(x=trade_marker["exit_date"], y=trade_marker["realized_pnl_usd"], name="trade P&L", marker_color=trade_marker["color"]), row=3, col=1)
    fig.add_trace(go.Bar(x=monthly["month"], y=monthly["return_pct"], name="monthly return", marker_color=monthly["color"]), row=4, col=1)
    fig.add_hline(y=summary["initial_nav"], line_dash="dot", line_color=GRAY, row=1, col=1)
    fig.add_hline(y=0, line_color=GRAY, row=2, col=1)
    fig.add_hline(y=0, line_color=GRAY, row=3, col=1)
    fig.add_hline(y=0, line_color=GRAY, row=4, col=1)
    fig.update_yaxes(title_text="$", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="DD %", ticksuffix="%", row=2, col=1)
    fig.update_yaxes(title_text="$", tickprefix="$", row=3, col=1)
    fig.update_yaxes(title_text="return %", ticksuffix="%", row=4, col=1)
    fig.update_layout(
        title={
            "text": (
                f"Final Pre-Deploy Simulation: {run_id}<br>"
                f"$10k -> ${summary['final_nav']:,.0f} "
                f"({summary['total_return_pct']:+.2f}%), "
                f"geo monthly {summary['geometric_monthly_return_pct']:+.2f}%, "
                f"max DD {summary['max_drawdown_pct']:.2f}%"
            ),
            "x": 0.02,
            "xanchor": "left",
        },
        template="plotly_white",
        height=1200,
        hovermode="x unified",
        legend={"orientation": "h", "y": 1.02, "x": 0.01},
        margin={"l": 70, "r": 40, "t": 120, "b": 70},
    )
    html_path = out_dir / "final_predeploy_report.html"
    fig.write_html(html_path, include_plotlyjs="cdn", full_html=True, config={"responsive": True, "displaylogo": False})

    summary_rows = "".join(
        f"<tr><th>{k}</th><td>{v:,.4f}</td></tr>" if isinstance(v, float) else f"<tr><th>{k}</th><td>{v}</td></tr>"
        for k, v in summary.items()
        if k != "sleeve_breakdown"
    )
    monthly_html = monthly.drop(columns=["color"]).to_html(index=False, float_format=lambda x: f"{x:,.2f}")
    trade_cols = [
        "trade_id",
        "sleeve",
        "strategy_type",
        "underlying_ticker",
        "entry_at_utc",
        "exit_at_utc",
        "realized_pnl_usd",
        "commission_usd",
        "spread_cost_usd",
        "slippage_usd",
    ]
    trades_html = trades[trade_cols].to_html(index=False, float_format=lambda x: f"{x:,.2f}")
    sleeve_html = pd.DataFrame(summary["sleeve_breakdown"]).to_html(index=False, float_format=lambda x: f"{x:,.2f}")
    dashboard = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Final Predeploy Report</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111827; }}
    h1, h2 {{ margin: 22px 0 10px; }}
    table {{ border-collapse: collapse; width: 100%; margin: 12px 0 24px; font-size: 13px; }}
    th, td {{ border: 1px solid #e5e7eb; padding: 6px 8px; text-align: right; }}
    th {{ background: #f3f4f6; text-align: left; }}
    .note {{ padding: 12px; background: #fff7ed; border: 1px solid #fed7aa; margin: 16px 0; }}
  </style>
</head>
<body>
  <h1>Final Predeploy Simulation</h1>
  <div class="note">This report uses realized P&L from the Year 2 evidence trade ledger. It is not full daily option mark-to-market, and IBKR is still required for live/paper quote and fill verification.</div>
  <h2>Summary</h2>
  <table>{summary_rows}</table>
  <h2>Sleeve Breakdown</h2>
  {sleeve_html}
  <h2>Monthly Returns</h2>
  {monthly_html}
  <h2>Trade Log</h2>
  {trades_html}
</body>
</html>
"""
    (out_dir / "final_predeploy_tables.html").write_text(dashboard, encoding="utf-8")
    return html_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--initial-nav", type=float, default=10_000.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or REPO_ROOT / "experiments" / "runs" / f"final_predeploy_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = read_trades(args.run_id)
    nav_df = realized_nav(trades, args.initial_nav)
    spy = read_spy(nav_df["date"].min(), nav_df["date"].max())
    nav_df = add_benchmark(nav_df, spy, args.initial_nav)
    monthly = monthly_table(nav_df, trades)
    summary = metrics(args.run_id, trades, nav_df, monthly, args.initial_nav)

    trades.to_csv(out_dir / "trades.csv", index=False)
    nav_df.to_csv(out_dir / "daily_nav.csv", index=False)
    monthly.to_csv(out_dir / "monthly_returns.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    html_path = write_report(out_dir=out_dir, run_id=args.run_id, trades=trades, nav_df=nav_df, monthly=monthly, summary=summary)

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nReport: {html_path}")
    print(f"Tables: {out_dir / 'final_predeploy_tables.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
