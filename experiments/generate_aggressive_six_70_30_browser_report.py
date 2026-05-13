"""70/30 browser report for the six aggressive strategies.

Uses:
  - train grid: first 70% window, fresh $10k simulation
  - test grid: newest 30% window, fresh $10k simulation
  - already-trained final-dataset model predictions from
    `final_dataset_model_test_all_*/predictions.csv`

The trained model is not retrained here. Its held-out SPY `prob_range_lgbm`
is used as a test-period gate for the six short-vol/iron-condor candidates.
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

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_spy_core_strategy_grid import price_condor_value  # noqa: E402


DEFAULT_ROOT = REPO_ROOT / "experiments" / "runs" / "aggressive_six_70_30_20260510T234527Z"
DEFAULT_PREDICTIONS = (
    REPO_ROOT
    / "experiments"
    / "runs"
    / "final_dataset_model_test_all_20260510T232754Z"
    / "predictions.csv"
)
SPEC_JSON = REPO_ROOT / "fullonriskmodelterminator" / "additional_six_spy_core_specs.json"

BASE_GRID_NAV = 10_000.0
INITIAL_NAV = 10_000.0
SIZE_SCALE = 1.0
MODEL_GATE_THRESHOLD = 0.55
ROUNDTRIP_COMMISSION = 0.65 * 8.0

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
    "gated": "#16a085",
    "blocked": "#bdc3c7",
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


def load_period(root: Path, period: str, specs: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = root / period
    nav = pd.read_csv(base / "daily_nav.csv")
    trades = pd.read_csv(base / "trades.csv")
    summary = pd.read_csv(base / "summary.csv")
    nav["date"] = pd.to_datetime(nav["date"])
    trades = add_trade_refs(trades, specs)
    nav, trades, summary = scale_period_to_capital(nav, trades, summary)
    return nav, trades, summary


def set_capital(capital: float) -> None:
    global INITIAL_NAV, SIZE_SCALE
    if capital <= 0:
        raise ValueError(f"capital must be positive, got {capital}")
    INITIAL_NAV = float(capital)
    SIZE_SCALE = INITIAL_NAV / BASE_GRID_NAV


def scale_period_to_capital(
    nav: pd.DataFrame,
    trades: pd.DataFrame,
    summary: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scale the 1x/$10k proxy grid to proportional exposure at target capital."""
    if abs(SIZE_SCALE - 1.0) < 1e-12:
        return nav, trades, summary

    nav = nav.copy()
    trades = trades.copy()
    summary = summary.copy()

    nav["nav"] = INITIAL_NAV + (nav["nav"] - BASE_GRID_NAV) * SIZE_SCALE
    for col in ("realized_pnl", "unrealized_pnl"):
        if col in nav.columns:
            nav[col] = nav[col] * SIZE_SCALE

    for col in ("pnl_usd", "max_loss_usd"):
        if col in trades.columns:
            trades[col] = trades[col] * SIZE_SCALE
    if "pnl_pct_max_loss" in trades.columns:
        trades["pnl_pct_max_loss"] = trades["pnl_usd"] / trades["max_loss_usd"].replace(0, np.nan) * 100.0

    summary["final_nav"] = INITIAL_NAV + (summary["final_nav"] - BASE_GRID_NAV) * SIZE_SCALE
    summary["return_pct"] = (summary["final_nav"] / INITIAL_NAV - 1.0) * 100.0
    for col in ("avg_trade_usd", "worst_trade_usd"):
        if col in summary.columns:
            summary[col] = summary[col] * SIZE_SCALE
    return nav, trades, summary


def add_trade_refs(trades: pd.DataFrame, specs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = trades.copy()
    for c in ("entry_date", "exit_date", "expiry"):
        out[c] = pd.to_datetime(out[c])
    out["profit_take"] = out["strategy"].map(lambda s: float(specs[s]["profit_take"]))
    out["stop_loss_credit_mult"] = out["strategy"].map(lambda s: float(specs[s]["stop_loss_credit_mult"]))
    out["wing_width"] = out["strategy"].map(lambda s: float(specs[s]["wing_width"]))
    out["profit_target_debit"] = out["entry_credit"] * (1.0 - out["profit_take"])
    out["stop_loss_debit"] = out["entry_credit"] * (1.0 + out["stop_loss_credit_mult"])
    out["max_loss_usd"] = (out["wing_width"] - out["entry_credit"]) * 100.0
    out["pnl_pct_max_loss"] = out["pnl_usd"] / out["max_loss_usd"].replace(0, np.nan) * 100.0
    out["structure"] = (
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


def merge_model_predictions(trades: pd.DataFrame, predictions_path: Path) -> pd.DataFrame:
    preds = pd.read_csv(predictions_path)
    preds["date"] = pd.to_datetime(preds["date"])
    spy = preds[preds["symbol"].astype(str).str.upper() == "SPY"][
        ["date", "prob_range_lgbm", "prob_up_lgbm", "pred_ret_lgbm", "expected_move_5d"]
    ].copy()
    out = trades.merge(spy, left_on="entry_date", right_on="date", how="left").drop(columns=["date"])
    out["model_take"] = out["prob_range_lgbm"] >= MODEL_GATE_THRESHOLD
    out["model_take"] = out["model_take"].fillna(False)
    return out


def benchmark(nav: pd.DataFrame) -> pd.DataFrame:
    b = nav[["date", "spy_close"]].drop_duplicates("date").sort_values("date").copy()
    b["nav"] = INITIAL_NAV * b["spy_close"] / float(b["spy_close"].iloc[0])
    return b


def max_drawdown(nav: pd.Series) -> float:
    peak = nav.cummax()
    return float((nav / peak - 1.0).min() * 100.0)


def benchmark_row(nav: pd.DataFrame) -> dict[str, Any]:
    b = benchmark(nav)
    return {
        "strategy": "SPY buy-and-hold benchmark",
        "trades": 1,
        "final_nav": float(b["nav"].iloc[-1]),
        "return_pct": (float(b["nav"].iloc[-1]) / INITIAL_NAV - 1.0) * 100.0,
        "max_drawdown_pct": max_drawdown(b["nav"]),
        "win_rate_pct": np.nan,
        "worst_trade_usd": np.nan,
        "max_open_trades": 1,
    }


def months_in_nav(nav: pd.DataFrame) -> float:
    start = pd.to_datetime(nav["date"]).min()
    end = pd.to_datetime(nav["date"]).max()
    return max((end - start).days / 30.4375, 1 / 30.4375)


def add_monthly(summary: pd.DataFrame, months: float) -> pd.DataFrame:
    out = summary.copy()
    out["monthly_pct"] = ((1.0 + out["return_pct"] / 100.0) ** (1.0 / months) - 1.0) * 100.0
    return out


def gated_nav_for_strategy(strategy: str, nav: pd.DataFrame, trades: pd.DataFrame) -> pd.DataFrame:
    dates = nav[["date", "spy_close", "iv"]].drop_duplicates("date").sort_values("date").copy()
    selected = trades[(trades["strategy"] == strategy) & (trades["model_take"])].copy()
    rows = []
    for _, day in dates.iterrows():
        dt = pd.Timestamp(day["date"])
        realized = selected[selected["exit_date"] <= dt]["pnl_usd"].sum()
        unrealized = 0.0
        open_trades = selected[(selected["entry_date"] <= dt) & (selected["exit_date"] > dt)]
        for _, pos in open_trades.iterrows():
            debit = price_condor_value(
                spot=float(day["spy_close"]),
                sigma=float(day["iv"]),
                as_of=dt,
                expiry=pd.Timestamp(pos["expiry"]),
                short_call=float(pos["short_call"]),
                long_call=float(pos["long_call"]),
                short_put=float(pos["short_put"]),
                long_put=float(pos["long_put"]),
                risk_free_rate=0.02,
                dividend_yield=0.015,
            )
            unrealized += ((float(pos["entry_credit"]) - debit) * 100.0 - ROUNDTRIP_COMMISSION) * SIZE_SCALE
        rows.append(
            {
                "date": dt,
                "strategy": strategy,
                "nav": INITIAL_NAV + float(realized) + float(unrealized),
                "realized_pnl": float(realized),
                "unrealized_pnl": float(unrealized),
                "open_trades": int(len(open_trades)),
            }
        )
    return pd.DataFrame(rows)


def gated_summary(test_nav: pd.DataFrame, test_trades: pd.DataFrame, strategies: list[str]) -> pd.DataFrame:
    rows = []
    for strategy in strategies:
        gnav = gated_nav_for_strategy(strategy, test_nav, test_trades)
        selected = test_trades[(test_trades["strategy"] == strategy) & (test_trades["model_take"])].copy()
        rows.append(
            {
                "strategy": strategy,
                "trades": int(len(selected)),
                "blocked": int(((test_trades["strategy"] == strategy) & (~test_trades["model_take"])).sum()),
                "final_nav": float(gnav["nav"].iloc[-1]),
                "return_pct": (float(gnav["nav"].iloc[-1]) / INITIAL_NAV - 1.0) * 100.0,
                "max_drawdown_pct": max_drawdown(gnav["nav"]),
                "win_rate_pct": float((selected["pnl_usd"] > 0).mean() * 100.0) if len(selected) else 0.0,
                "worst_trade_usd": float(selected["pnl_usd"].min()) if len(selected) else 0.0,
                "max_open_trades": int(gnav["open_trades"].max()) if len(gnav) else 0,
            }
        )
    return pd.DataFrame(rows).sort_values(["return_pct", "max_drawdown_pct"], ascending=[False, False])


def table_html(df: pd.DataFrame) -> str:
    return df.to_html(index=False, escape=False, classes="data-table")


def summary_table(summary: pd.DataFrame, bench_row: dict[str, Any], months: float, extra_cols: list[str] | None = None) -> str:
    display = pd.concat([add_monthly(summary.copy(), months), add_monthly(pd.DataFrame([bench_row]), months)], ignore_index=True)
    rows = []
    for _, r in display.iterrows():
        row = {
            "strategy": r["strategy"],
            "return": pct(float(r["return_pct"])),
            "monthly": pct(float(r["monthly_pct"])),
            "max_dd": pct(float(r["max_drawdown_pct"])),
            "trades": int(r["trades"]) if pd.notna(r["trades"]) else "",
            "win_rate": "" if pd.isna(r.get("win_rate_pct", np.nan)) else f"{float(r['win_rate_pct']):.1f}%",
            "final_nav": usd(float(r["final_nav"])),
            "worst_trade": "" if pd.isna(r.get("worst_trade_usd", np.nan)) else usd(float(r["worst_trade_usd"])),
            "max_open": int(r["max_open_trades"]) if pd.notna(r.get("max_open_trades", np.nan)) else "",
            "note": NOTES.get(r["strategy"], "benchmark"),
        }
        if extra_cols and "blocked" in extra_cols:
            row["blocked_by_model"] = "" if pd.isna(r.get("blocked", np.nan)) else int(r["blocked"])
        rows.append(row)
    return table_html(pd.DataFrame(rows))


def overview_chart(train_nav: pd.DataFrame, test_nav: pd.DataFrame, gated_navs: dict[str, pd.DataFrame], train_summary: pd.DataFrame, test_summary: pd.DataFrame, gated_sum: pd.DataFrame) -> str:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Train window raw NAV (70%)",
            "Held-out test raw NAV (30%)",
            "Held-out test model-gated NAV",
            "Test return vs drawdown",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    strategies = train_summary["strategy"].tolist()
    for s in strategies:
        c = COLORS.get(s, "#555555")
        w = train_nav[train_nav["strategy"] == s]
        fig.add_trace(go.Scatter(x=w["date"], y=w["nav"], name=s, mode="lines", line=dict(color=c, width=1.5), showlegend=True), row=1, col=1)
        w = test_nav[test_nav["strategy"] == s]
        fig.add_trace(go.Scatter(x=w["date"], y=w["nav"], name=s, mode="lines", line=dict(color=c, width=1.5), showlegend=False), row=1, col=2)
        gw = gated_navs[s]
        fig.add_trace(go.Scatter(x=gw["date"], y=gw["nav"], name=s, mode="lines", line=dict(color=c, width=1.5), showlegend=False), row=2, col=1)
    tb = benchmark(train_nav)
    eb = benchmark(test_nav)
    for r, c, b in [(1, 1, tb), (1, 2, eb), (2, 1, eb)]:
        fig.add_trace(go.Scatter(x=b["date"], y=b["nav"], name="SPY benchmark", mode="lines", line=dict(color=COLORS["benchmark"], dash="dot", width=2.4), showlegend=(r == 1 and c == 1)), row=r, col=c)

    fig.add_trace(
        go.Scatter(
            x=test_summary["max_drawdown_pct"],
            y=test_summary["return_pct"],
            text=test_summary["strategy"],
            name="raw test",
            mode="markers",
            marker=dict(color="#e74c3c", size=13, opacity=0.78),
            hovertemplate="<b>%{text}</b><br>Raw return %{y:.1f}%<br>DD %{x:.1f}%<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=gated_sum["max_drawdown_pct"],
            y=gated_sum["return_pct"],
            text=gated_sum["strategy"],
            name="model-gated test",
            mode="markers",
            marker=dict(color="#16a085", size=13, symbol="diamond", opacity=0.85),
            hovertemplate="<b>%{text}</b><br>Gated return %{y:.1f}%<br>DD %{x:.1f}%<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.update_yaxes(title_text="NAV ($10k start)", row=1, col=1)
    fig.update_yaxes(title_text="NAV ($10k start)", row=1, col=2)
    fig.update_yaxes(title_text="NAV ($10k start)", row=2, col=1)
    fig.update_xaxes(title_text="Max DD (%)", row=2, col=2)
    fig.update_yaxes(title_text="Return (%)", row=2, col=2)
    fig.update_layout(height=1000, template="plotly_white", legend=dict(orientation="h", y=1.06), margin=dict(l=60, r=30, t=110, b=45))
    return fig.to_html(include_plotlyjs=True, full_html=False)


def strategy_chart(strategy: str, test_nav: pd.DataFrame, test_trades: pd.DataFrame, gated_nav: pd.DataFrame) -> str:
    raw = test_nav[test_nav["strategy"] == strategy].copy()
    trades = test_trades[test_trades["strategy"] == strategy].copy()
    taken = trades[trades["model_take"]].copy()
    blocked = trades[~trades["model_take"]].copy()
    bench = benchmark(test_nav)
    color = COLORS.get(strategy, "#555555")
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.35, 0.37, 0.28],
        vertical_spacing=0.055,
        subplot_titles=(
            "Held-out test NAV: raw vs already-trained model gate vs benchmark",
            "SPY entries/exits, model take/block, and strike corridor",
            "Trade P&L by exit reason",
        ),
    )
    fig.add_trace(go.Scatter(x=raw["date"], y=raw["nav"], name="raw test", mode="lines", line=dict(color=color, width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=gated_nav["date"], y=gated_nav["nav"], name="model-gated test", mode="lines", line=dict(color=COLORS["gated"], width=2.2, dash="dash")), row=1, col=1)
    fig.add_trace(go.Scatter(x=bench["date"], y=bench["nav"], name="SPY benchmark", mode="lines", line=dict(color=COLORS["benchmark"], width=2, dash="dot")), row=1, col=1)
    fig.add_trace(go.Scatter(x=raw["date"], y=raw["spy_close"], name="SPY close", mode="lines", line=dict(color="#111111", width=1.6)), row=2, col=1)
    fig.add_trace(go.Scatter(x=trades["entry_date"], y=trades["short_call"], name="short call", mode="markers", marker=dict(color="#c0392b", size=4, opacity=0.25)), row=2, col=1)
    fig.add_trace(go.Scatter(x=trades["entry_date"], y=trades["short_put"], name="short put", mode="markers", marker=dict(color="#27ae60", size=4, opacity=0.25)), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=blocked["entry_date"],
            y=blocked["entry_spy"],
            name="blocked by trained model",
            mode="markers",
            marker=dict(color=COLORS["blocked"], size=6, symbol="circle-open"),
            customdata=np.stack([blocked["prob_range_lgbm"].fillna(-1).map(lambda x: "n/a" if x < 0 else f"{x:.1%}")], axis=-1) if len(blocked) else None,
            hovertemplate="Blocked entry %{x|%Y-%m-%d}<br>SPY $%{y:.2f}<br>P(range) %{customdata[0]}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=taken["entry_date"],
            y=taken["entry_spy"],
            name="taken by trained model",
            mode="markers",
            marker=dict(color=COLORS["gated"], size=8, symbol="triangle-up"),
            customdata=np.stack(
                [
                    taken["exit_date"].dt.date.astype(str),
                    taken["reason"],
                    taken["prob_range_lgbm"].map(lambda x: f"{x:.1%}"),
                    taken["pnl_usd"].map(usd),
                    taken["profit_target_debit"].map(lambda x: f"{x:.2f}"),
                    taken["stop_loss_debit"].map(lambda x: f"{x:.2f}"),
                ],
                axis=-1,
            )
            if len(taken)
            else None,
            hovertemplate=(
                "Taken entry %{x|%Y-%m-%d}<br>SPY $%{y:.2f}<br>Exit %{customdata[0]}<br>"
                "Reason %{customdata[1]}<br>P(range) %{customdata[2]}<br>P&L %{customdata[3]}<br>"
                "PT debit %{customdata[4]}<br>Stop debit %{customdata[5]}<extra></extra>"
            ),
        ),
        row=2,
        col=1,
    )
    fig.add_trace(go.Scatter(x=taken["exit_date"], y=taken["exit_spy"], name="taken exits", mode="markers", marker=dict(color="#111111", size=6, symbol="x")), row=2, col=1)
    reason_colors = {"profit_target": "#27ae60", "stop_loss": "#c0392b", "dte_close": "#2980b9", "end_forced": "#7f8c8d"}
    for reason, sub in trades.groupby("reason"):
        fig.add_trace(go.Scatter(x=sub["exit_date"], y=sub["pnl_usd"], name=f"raw {reason}", mode="markers", marker=dict(color=reason_colors.get(reason, "#8e44ad"), size=5, opacity=0.28)), row=3, col=1)
    for reason, sub in taken.groupby("reason"):
        fig.add_trace(go.Scatter(x=sub["exit_date"], y=sub["pnl_usd"], name=f"gated {reason}", mode="markers", marker=dict(color=reason_colors.get(reason, "#8e44ad"), size=8, symbol="diamond", opacity=0.82)), row=3, col=1)
    fig.add_hline(y=0, row=3, col=1, line_color="#555555", line_width=1)
    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="SPY / strikes", row=2, col=1)
    fig.update_yaxes(title_text="Trade P&L", row=3, col=1)
    fig.update_layout(
        height=1050,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04),
        title=dict(text=f"{strategy}<br><sup>{NOTES.get(strategy, '')}</sup>", x=0.01, xanchor="left"),
        margin=dict(l=70, r=30, t=115, b=50),
    )
    return fig.to_html(include_plotlyjs=False, full_html=False)


def trade_table(strategy: str, trades: pd.DataFrame, max_rows: int = 260) -> str:
    w = trades[trades["strategy"] == strategy].copy().sort_values("exit_date")
    if len(w) > max_rows:
        worst = w.nsmallest(max_rows // 4, "pnl_usd")
        best = w.nlargest(max_rows // 4, "pnl_usd")
        taken = w[w["model_take"]]
        recent = w.tail(max_rows)
        w = pd.concat([worst, best, taken, recent]).drop_duplicates().sort_values("exit_date").head(max_rows)
    out = pd.DataFrame(
        {
            "model": np.where(w["model_take"], "TAKE", "BLOCK"),
            "p_range": w["prob_range_lgbm"].map(lambda x: "" if pd.isna(x) else f"{x:.1%}"),
            "entry": w["entry_date"].dt.date.astype(str),
            "exit": w["exit_date"].dt.date.astype(str),
            "reason": w["reason"],
            "structure": w["structure"],
            "entry_spy": w["entry_spy"].map(usd),
            "exit_spy": w["exit_spy"].map(usd),
            "credit": w["entry_credit"].map(lambda x: f"{x:.2f}"),
            "profit_target_debit": w["profit_target_debit"].map(lambda x: f"{x:.2f}"),
            "stop_loss_debit": w["stop_loss_debit"].map(lambda x: f"{x:.2f}"),
            "exit_debit": w["exit_debit"].map(lambda x: f"{x:.2f}"),
            "pnl": w["pnl_usd"].map(usd),
            "pnl_pct_max_loss": w["pnl_pct_max_loss"].map(lambda x: f"{x:+.1f}%"),
        }
    )
    return table_html(out)


def build(root: Path, predictions: Path, specs_path: Path, out_path: Path, capital: float) -> Path:
    set_capital(capital)
    specs = load_specs(specs_path)
    train_nav, train_trades, train_summary = load_period(root, "train", specs)
    test_nav, test_trades_raw, test_summary = load_period(root, "test", specs)
    test_trades = merge_model_predictions(test_trades_raw, predictions)
    strategies = test_summary["strategy"].tolist()
    gsum = gated_summary(test_nav, test_trades, strategies)
    gated_navs = {s: gated_nav_for_strategy(s, test_nav, test_trades) for s in strategies}
    train_months = months_in_nav(train_nav)
    test_months = months_in_nav(test_nav)

    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #17202a; }
      h1 { margin-bottom: 4px; }
      h2 { margin-top: 38px; padding-top: 20px; border-top: 1px solid #d7dbdd; }
      h3 { margin-top: 24px; }
      .note { color: #566573; max-width: 1200px; line-height: 1.45; }
      .navlinks a { margin-right: 14px; font-size: 13px; }
      .scroll { overflow-x: auto; border: 1px solid #eee; padding: 8px; margin-bottom: 20px; }
      table.data-table { border-collapse: collapse; width: 100%; font-size: 12px; }
      table.data-table th, table.data-table td { border: 1px solid #d6dbdf; padding: 6px 8px; text-align: right; white-space: nowrap; }
      table.data-table th { background: #f2f3f4; position: sticky; top: 0; z-index: 1; }
      table.data-table td:first-child, table.data-table th:first-child,
      table.data-table td:nth-child(3), table.data-table th:nth-child(3),
      table.data-table td:nth-child(4), table.data-table th:nth-child(4),
      table.data-table td:nth-child(5), table.data-table th:nth-child(5),
      table.data-table td:nth-child(6), table.data-table th:nth-child(6) { text-align: left; }
      .small { font-size: 12px; color: #566573; }
    </style>
    """
    body = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Aggressive Six 70/30 Trained Model Report</title>",
        css,
        "</head><body>",
        f"<h1>Aggressive Six: 70/30 + Already-Trained Model Gate (${INITIAL_NAV:,.0f} capital)</h1>",
        f"<p class='note'>Train window: <b>{train_nav['date'].min().date()} to {train_nav['date'].max().date()}</b> "
        f"({train_months:.2f} months). "
        f"Held-out test window: <b>{test_nav['date'].min().date()} to {test_nav['date'].max().date()}</b> "
        f"({test_months:.2f} months after indicator warm-up). "
        f"The model used here is already trained; predictions are loaded from <code>{predictions}</code>. "
        f"This report scales the original $10k proxy grid by <b>{SIZE_SCALE:.1f}x</b>, equivalent to proportional contract sizing for ${INITIAL_NAV:,.0f}. "
        f"Gate: take an iron-condor trade only when <code>prob_range_lgbm >= {MODEL_GATE_THRESHOLD:.0%}</code>. "
        "Stop-loss and profit-target are combo debit thresholds from the proxy backtester, not IBKR intraday stops.</p>",
        "<div class='navlinks'>",
    ]
    for s in strategies:
        body.append(f"<a href='#{s}'>{s}</a>")
    body.extend(
        [
            "</div>",
            "<h2>Overview</h2>",
            overview_chart(train_nav, test_nav, gated_navs, train_summary, test_summary, gsum),
            "<h3>Train Raw Summary (70%) + Benchmark</h3><div class='scroll'>",
            summary_table(train_summary, benchmark_row(train_nav), train_months),
            "</div>",
            "<h3>Held-Out Test Raw Summary (30%) + Benchmark</h3><div class='scroll'>",
            summary_table(test_summary, benchmark_row(test_nav), test_months),
            "</div>",
            "<h3>Held-Out Test With Already-Trained Model Gate</h3><div class='scroll'>",
            summary_table(gsum, benchmark_row(test_nav), test_months, extra_cols=["blocked"]),
            "</div>",
        ]
    )
    for s in strategies:
        body.extend(
            [
                f"<h2 id='{s}'>{s}</h2>",
                strategy_chart(s, test_nav, test_trades, gated_navs[s]),
                "<p class='small'>Table prioritizes model-taken trades, largest winners/losers, and recent trades when capped for browser speed.</p>",
                "<div class='scroll'>",
                trade_table(s, test_trades),
                "</div>",
            ]
        )
    body.extend(["</body></html>"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(body), encoding="utf-8")
    # Also save machine-readable gated summary. Use capital-specific names so
    # $100k scaled reports do not overwrite the original $10k artifacts.
    suffix = "" if abs(INITIAL_NAV - BASE_GRID_NAV) < 1e-9 else f"_{int(INITIAL_NAV):d}"
    gsum.to_csv(root / f"test_model_gated_summary{suffix}.csv", index=False)
    test_trades.to_csv(root / f"test_trades_with_trained_model_predictions{suffix}.csv", index=False)
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--spec-json", type=Path, default=SPEC_JSON)
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.root / "aggressive_six_70_30_trained_model_report.html"
    path = build(args.root, args.predictions, args.spec_json, out, args.capital)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
