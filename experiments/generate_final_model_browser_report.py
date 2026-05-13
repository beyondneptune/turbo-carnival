"""Build a browser dashboard for the final-dataset model tests.

Reads the outputs from `experiments/test_final_dataset_models.py` and creates
one standalone HTML page with:
  - equity curves,
  - SPY price with entry/exit markers,
  - stop/profit reference bands,
  - detailed trade tables.

The stop/profit fields are references derived from daily 5-day proxy data; this
report does not claim broker-level intraday stop execution.
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
SPY_PANEL = REPO_ROOT / "data" / "final_dataset" / "spy_training_panel.parquet"
DEFAULT_ALL = REPO_ROOT / "experiments" / "runs" / "final_dataset_model_test_all_20260510T232754Z"
DEFAULT_SPY = REPO_ROOT / "experiments" / "runs" / "final_dataset_model_test_spy_20260510T232802Z"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def fmt_pct(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"{x * 100:+.2f}%"


def fmt_money(x: float) -> str:
    if pd.isna(x):
        return ""
    return f"${x:,.2f}"


def strategy_specs() -> list[dict[str, str]]:
    return [
        {
            "key": "return",
            "name": "LGBM Return Long/Short",
            "signal": "signal_lgbm_return",
            "ret": "ret_return_model",
            "equity": "equity_ret_return_model",
            "color": "#e67e22",
        },
        {
            "key": "direction",
            "name": "LGBM Direction Long/Cash",
            "signal": "signal_lgbm_direction",
            "ret": "ret_direction_model",
            "equity": "equity_ret_direction_model",
            "color": "#3498db",
        },
        {
            "key": "range",
            "name": "LGBM Range / Short-Vol Proxy",
            "signal": "signal_lgbm_range",
            "ret": "ret_range_proxy",
            "equity": "equity_ret_range_proxy",
            "color": "#2ecc71",
        },
    ]


def spy_prices() -> pd.DataFrame:
    df = pd.read_parquet(SPY_PANEL, columns=["date", "close", "high", "low", "open"])
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.dropna(subset=["close"]).sort_values("date").reset_index(drop=True)


def add_trade_details(equity: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    eq = equity.copy()
    eq["date"] = pd.to_datetime(eq["date"]).dt.tz_localize(None)
    p = prices[["date", "close"]].copy()
    eq = eq.merge(p.rename(columns={"close": "entry_close"}), on="date", how="left")
    p_index = {d: i for i, d in enumerate(p["date"])}
    exit_dates = []
    exit_closes = []
    for d in eq["date"]:
        idx = p_index.get(d)
        if idx is None:
            exit_dates.append(pd.NaT)
            exit_closes.append(np.nan)
            continue
        out_idx = min(idx + 5, len(p) - 1)
        exit_dates.append(p.loc[out_idx, "date"])
        exit_closes.append(float(p.loc[out_idx, "close"]))
    eq["exit_date"] = exit_dates
    eq["exit_close"] = exit_closes

    eq["expected_move_low"] = eq["entry_close"] * (1.0 - eq["expected_move_5d"])
    eq["expected_move_high"] = eq["entry_close"] * (1.0 + eq["expected_move_5d"])
    eq["range_proxy_credit"] = 0.45 * eq["expected_move_5d"].clip(lower=0.005)
    return eq


def trade_table(equity: pd.DataFrame, spec: dict[str, str], capital: float) -> pd.DataFrame:
    sig = equity[spec["signal"]].fillna(0.0)
    rows = equity[sig != 0].copy()
    if rows.empty:
        return pd.DataFrame()

    if spec["key"] == "range":
        rows["action"] = "SHORT RANGE / CONDOR PROXY"
        rows["stop_loss_reference"] = (
            "breach expected-move band: "
            + rows["expected_move_low"].map(fmt_money)
            + " / "
            + rows["expected_move_high"].map(fmt_money)
        )
        rows["profit_reference"] = rows["range_proxy_credit"].map(fmt_pct)
    else:
        rows["action"] = np.where(rows[spec["signal"]] > 0, "LONG SPY PROXY", "SHORT SPY PROXY")
        long_stop = rows["entry_close"] * (1.0 - rows["expected_move_5d"])
        long_profit = rows["entry_close"] * (1.0 + rows["expected_move_5d"])
        short_stop = rows["entry_close"] * (1.0 + rows["expected_move_5d"])
        short_profit = rows["entry_close"] * (1.0 - rows["expected_move_5d"])
        rows["stop_loss_reference"] = np.where(
            rows[spec["signal"]] > 0,
            long_stop.map(fmt_money),
            short_stop.map(fmt_money),
        )
        rows["profit_reference"] = np.where(
            rows[spec["signal"]] > 0,
            long_profit.map(fmt_money),
            short_profit.map(fmt_money),
        )

    out = pd.DataFrame(
        {
            "entry_date": rows["date"].dt.date.astype(str),
            "exit_date": pd.to_datetime(rows["exit_date"]).dt.date.astype(str),
            "action": rows["action"],
            "entry": rows["entry_close"].map(fmt_money),
            "exit": rows["exit_close"].map(fmt_money),
            "stop_loss_reference": rows["stop_loss_reference"],
            "profit_reference": rows["profit_reference"],
            "expected_move": rows["expected_move_5d"].map(fmt_pct),
            "predicted_return": rows["pred_ret_lgbm"].map(fmt_pct),
            "prob_up": rows["prob_up_lgbm"].map(lambda x: f"{x:.1%}"),
            "prob_range": rows["prob_range_lgbm"].map(lambda x: f"{x:.1%}"),
            "realized_underlying_5d": rows["realized_fwd_ret_5d"].map(fmt_pct),
            "strategy_return": rows[spec["ret"]].map(fmt_pct),
            "pnl_on_100k": (rows[spec["ret"]] * capital).map(fmt_money),
        }
    )
    return out


def chart_for_run(label: str, run_dir: Path, prices: pd.DataFrame) -> tuple[str, dict[str, pd.DataFrame]]:
    metrics = load_json(run_dir / "metrics.json")
    equity = pd.read_csv(run_dir / "spy_nonoverlap_equity.csv")
    equity = add_trade_details(equity, prices)
    pred = pd.read_csv(run_dir / "predictions.csv")
    pred["date"] = pd.to_datetime(pred["date"])

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.44, 0.34, 0.22],
        subplot_titles=(
            f"{label}: SPY price with model entries/exits and expected-move bands",
            "Equity curves on non-overlapping 5-day trades",
            "Model probabilities",
        ),
    )

    px = prices[(prices["date"] >= equity["date"].min()) & (prices["date"] <= equity["exit_date"].max())].copy()
    fig.add_trace(
        go.Scatter(x=px["date"], y=px["close"], name="SPY close", mode="lines", line=dict(color="#111111", width=2)),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=equity["date"],
            y=equity["expected_move_high"],
            name="expected move high",
            mode="lines",
            line=dict(color="#bbbbbb", width=1, dash="dash"),
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=equity["date"],
            y=equity["expected_move_low"],
            name="expected move low",
            mode="lines",
            line=dict(color="#bbbbbb", width=1, dash="dash"),
            fill="tonexty",
            fillcolor="rgba(180,180,180,0.12)",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    for spec in strategy_specs():
        rows = equity[equity[spec["signal"]].fillna(0.0) != 0].copy()
        if rows.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=rows["date"],
                y=rows["entry_close"],
                name=f"{spec['name']} entry",
                mode="markers",
                marker=dict(color=spec["color"], size=10, symbol="triangle-up"),
                customdata=np.stack(
                    [
                        rows["exit_date"].astype(str),
                        rows[spec["ret"]].map(lambda x: f"{x * 100:+.2f}%"),
                        rows["prob_up_lgbm"].map(lambda x: f"{x:.1%}"),
                        rows["prob_range_lgbm"].map(lambda x: f"{x:.1%}"),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Entry %{x|%Y-%m-%d}<br>Price $%{y:.2f}<br>"
                    "Exit %{customdata[0]}<br>Strategy return %{customdata[1]}<br>"
                    "P(up) %{customdata[2]}<br>P(range) %{customdata[3]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=rows["exit_date"],
                y=rows["exit_close"],
                name=f"{spec['name']} exit",
                mode="markers",
                marker=dict(color=spec["color"], size=9, symbol="x"),
                hovertemplate="Exit %{x|%Y-%m-%d}<br>Price $%{y:.2f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=equity["date"],
                y=equity[spec["equity"]],
                name=spec["name"],
                mode="lines",
                line=dict(color=spec["color"], width=2),
            ),
            row=2,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=equity["date"],
            y=equity["equity_ret_buy_hold_5d"],
            name="Buy/hold SPY",
            mode="lines",
            line=dict(color="#7f8c8d", width=2, dash="dot"),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=equity["date"],
            y=equity["prob_up_lgbm"],
            name="P(up)",
            mode="lines+markers",
            line=dict(color="#3498db", width=1.5),
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=equity["date"],
            y=equity["prob_range_lgbm"],
            name="P(range)",
            mode="lines+markers",
            line=dict(color="#2ecc71", width=1.5),
        ),
        row=3,
        col=1,
    )
    fig.add_hline(y=0.55, row=3, col=1, line_dash="dash", line_color="#999999")
    fig.update_yaxes(title_text="SPY", row=1, col=1)
    fig.update_yaxes(title_text="Growth of $1", row=2, col=1)
    fig.update_yaxes(title_text="Probability", range=[0, 1], row=3, col=1)
    fig.update_layout(
        height=1050,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=110, b=50),
    )

    tables = {spec["key"]: trade_table(equity, spec, capital=100_000.0) for spec in strategy_specs()}
    summary = metrics["spy_nonoverlap_strategy_stats"]
    summary_rows = []
    for k, v in summary.items():
        summary_rows.append(
            {
                "strategy": k,
                "total_return": f"{v['total_return_pct']:+.2f}%",
                "monthly": f"{v['monthly_pct']:+.2f}%",
                "max_dd": f"{v['max_dd_pct']:+.2f}%",
                "trades": v["trades"],
            }
        )
    tables["summary"] = pd.DataFrame(summary_rows)
    html = fig.to_html(include_plotlyjs=True, full_html=False)
    return html, tables


def df_to_table(df: pd.DataFrame, max_rows: int = 200) -> str:
    if df.empty:
        return "<p>No trades.</p>"
    return df.head(max_rows).to_html(index=False, escape=False, classes="trade-table")


def build_html(all_dir: Path, spy_dir: Path, out_path: Path) -> Path:
    prices = spy_prices()
    all_chart, all_tables = chart_for_run("All-symbol trained model", all_dir, prices)
    spy_chart, spy_tables = chart_for_run("SPY-only trained model", spy_dir, prices)

    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #17202a; }
      h1 { margin-bottom: 4px; }
      h2 { margin-top: 36px; border-top: 1px solid #ddd; padding-top: 22px; }
      h3 { margin-top: 24px; }
      .note { color: #566573; max-width: 1100px; line-height: 1.45; }
      .grid { display: grid; grid-template-columns: repeat(2, minmax(280px, 1fr)); gap: 16px; margin: 18px 0; }
      .card { border: 1px solid #ddd; border-radius: 6px; padding: 14px 16px; background: #fafafa; }
      table.trade-table { border-collapse: collapse; width: 100%; font-size: 12px; margin: 8px 0 22px; }
      table.trade-table th, table.trade-table td { border: 1px solid #ddd; padding: 6px 8px; text-align: right; white-space: nowrap; }
      table.trade-table th { background: #f2f3f4; position: sticky; top: 0; z-index: 1; }
      table.trade-table td:nth-child(1), table.trade-table td:nth-child(2), table.trade-table td:nth-child(3),
      table.trade-table th:nth-child(1), table.trade-table th:nth-child(2), table.trade-table th:nth-child(3) { text-align: left; }
      .scroll { overflow-x: auto; border: 1px solid #eee; padding: 8px; }
    </style>
    """

    body = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Final Dataset Model Browser Report</title>",
        css,
        "</head><body>",
        "<h1>Final Dataset Model Test: Browser Report</h1>",
        "<p class='note'>70/30 chronological split. Entry/exit markers use non-overlapping 5-trading-day SPY proxy trades. "
        "Stop-loss and profit references are expected-move references from daily data, not simulated IBKR stop orders or intraday option marks.</p>",
        "<div class='grid'>",
        "<div class='card'><b>All-symbol model</b><br>trained on all 84 symbols, evaluated on SPY proxy trades.</div>",
        "<div class='card'><b>SPY-only model</b><br>trained only on SPY rows, smaller sample, evaluated on the same SPY dates.</div>",
        "</div>",
        "<h2>All-Symbol Trained Model</h2>",
        all_chart,
        "<h3>Summary</h3><div class='scroll'>",
        df_to_table(all_tables["summary"]),
        "</div>",
    ]
    for spec in strategy_specs():
        body.extend([f"<h3>{spec['name']} Trades</h3><div class='scroll'>", df_to_table(all_tables[spec["key"]]), "</div>"])

    body.extend(["<h2>SPY-Only Trained Model</h2>", spy_chart, "<h3>Summary</h3><div class='scroll'>", df_to_table(spy_tables["summary"]), "</div>"])
    for spec in strategy_specs():
        body.extend([f"<h3>{spec['name']} Trades</h3><div class='scroll'>", df_to_table(spy_tables[spec["key"]]), "</div>"])

    body.extend(["</body></html>"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(body), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all-run", type=Path, default=DEFAULT_ALL)
    parser.add_argument("--spy-run", type=Path, default=DEFAULT_SPY)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "experiments" / "runs" / "final_dataset_model_browser_report.html",
    )
    args = parser.parse_args()
    path = build_html(args.all_run, args.spy_run, args.out)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
