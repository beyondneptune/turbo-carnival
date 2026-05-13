"""Capital-aware 70/30 report for the six aggressive SPY candidates.

Unlike the proportional-scale report, this replays the 1-contract proxy trade
log with NAV-aware integer contract sizing:
  qty = floor((NAV * risk_pct_per_trade) / max_loss_per_contract)
  plus a max concurrent open-risk cap.

It also applies the already-trained final-dataset model gate on the held-out
test period without retraining the model.
"""
from __future__ import annotations

import argparse
import json
import math
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

ROUNDTRIP_COMMISSION = 0.65 * 8.0
MODEL_GATE_THRESHOLD = 0.55

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
}


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:+.2f}%"


def usd(x: float) -> str:
    return "" if pd.isna(x) else f"${x:,.2f}"


def load_specs() -> dict[str, dict[str, Any]]:
    raw = json.loads(SPEC_JSON.read_text())
    return {s["name"]: s for s in raw["strategies"]}


def add_trade_refs(trades: pd.DataFrame, specs: dict[str, dict[str, Any]]) -> pd.DataFrame:
    out = trades.copy()
    for c in ("entry_date", "exit_date", "expiry"):
        out[c] = pd.to_datetime(out[c])
    out["wing_width"] = out["strategy"].map(lambda s: float(specs[s]["wing_width"]))
    out["profit_take"] = out["strategy"].map(lambda s: float(specs[s]["profit_take"]))
    out["stop_loss_credit_mult"] = out["strategy"].map(lambda s: float(specs[s]["stop_loss_credit_mult"]))
    out["max_loss_per_contract"] = (out["wing_width"] - out["entry_credit"]) * 100.0
    out["profit_target_debit"] = out["entry_credit"] * (1.0 - out["profit_take"])
    out["stop_loss_debit"] = out["entry_credit"] * (1.0 + out["stop_loss_credit_mult"])
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
    out["model_take"] = (out["prob_range_lgbm"] >= MODEL_GATE_THRESHOLD).fillna(False)
    return out


def benchmark(nav: pd.DataFrame, capital: float) -> pd.DataFrame:
    b = nav[["date", "spy_close"]].drop_duplicates("date").sort_values("date").copy()
    b["nav"] = capital * b["spy_close"] / float(b["spy_close"].iloc[0])
    return b


def max_dd_pct(nav: pd.Series) -> float:
    return float((nav / nav.cummax() - 1.0).min() * 100.0)


def months(nav: pd.DataFrame) -> float:
    d = (pd.to_datetime(nav["date"]).max() - pd.to_datetime(nav["date"]).min()).days
    return max(d / 30.4375, 1 / 30.4375)


def monthly_pct(total_return_pct: float, months_count: float) -> float:
    growth = 1.0 + total_return_pct / 100.0
    if growth <= 0 or not np.isfinite(growth):
        return -100.0
    return float((growth ** (1.0 / months_count) - 1.0) * 100.0)


def mark_trade(row: pd.Series, market_row: pd.Series, as_of: pd.Timestamp) -> float:
    debit = price_condor_value(
        spot=float(market_row["spy_close"]),
        sigma=float(market_row["iv"]),
        as_of=as_of,
        expiry=pd.Timestamp(row["expiry"]),
        short_call=float(row["short_call"]),
        long_call=float(row["long_call"]),
        short_put=float(row["short_put"]),
        long_put=float(row["long_put"]),
        risk_free_rate=0.02,
        dividend_yield=0.015,
    )
    return (float(row["entry_credit"]) - debit) * 100.0 - ROUNDTRIP_COMMISSION


def replay_strategy(
    strategy: str,
    nav_template: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    capital: float,
    risk_pct_per_trade: float,
    max_concurrent_risk_pct: float,
    use_model_gate: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dates = nav_template[["date", "spy_close", "iv"]].drop_duplicates("date").sort_values("date").copy()
    w = trades[trades["strategy"] == strategy].copy().sort_values(["entry_date", "exit_date"])
    if use_model_gate:
        w = w[w["model_take"]].copy()

    open_pos: list[dict[str, Any]] = []
    placed: list[dict[str, Any]] = []
    realized = 0.0
    nav_rows: list[dict[str, Any]] = []

    for _, mrow in dates.iterrows():
        dt = pd.Timestamp(mrow["date"])

        still_open = []
        for pos in open_pos:
            if pd.Timestamp(pos["exit_date"]) <= dt:
                realized += float(pos["qty"]) * float(pos["pnl_usd"])
            else:
                still_open.append(pos)
        open_pos = still_open

        unrealized = sum(float(pos["qty"]) * mark_trade(pd.Series(pos), mrow, dt) for pos in open_pos)
        nav_now = capital + realized + unrealized
        open_risk = sum(float(pos["qty"]) * float(pos["max_loss_per_contract"]) for pos in open_pos)
        risk_cap = max(nav_now, 0.0) * max_concurrent_risk_pct / 100.0

        todays = w[w["entry_date"] == dt]
        for _, tr in todays.iterrows():
            nav_for_size = max(nav_now, 0.0)
            per_contract_loss = max(float(tr["max_loss_per_contract"]), 1.0)
            ideal_qty = int((nav_for_size * risk_pct_per_trade / 100.0) // per_contract_loss)
            remaining_risk = max(risk_cap - open_risk, 0.0)
            cap_qty = int(remaining_risk // per_contract_loss)
            qty = max(0, min(ideal_qty, cap_qty))
            row = tr.to_dict()
            row["qty"] = qty
            row["skipped_reason"] = "" if qty > 0 else "risk_cap_or_nav"
            if qty > 0:
                row["sized_pnl_usd"] = qty * float(tr["pnl_usd"])
                row["sized_max_loss_usd"] = qty * per_contract_loss
                open_pos.append(row)
                placed.append(row)
                open_risk += qty * per_contract_loss
            else:
                row["sized_pnl_usd"] = 0.0
                row["sized_max_loss_usd"] = 0.0
                placed.append(row)

        unrealized = sum(float(pos["qty"]) * mark_trade(pd.Series(pos), mrow, dt) for pos in open_pos)
        nav_now = capital + realized + unrealized
        nav_rows.append(
            {
                "date": dt,
                "strategy": strategy,
                "nav": nav_now,
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "open_trades": len(open_pos),
                "open_contracts": int(sum(pos["qty"] for pos in open_pos)),
                "open_risk_usd": float(sum(float(pos["qty"]) * float(pos["max_loss_per_contract"]) for pos in open_pos)),
                "spy_close": float(mrow["spy_close"]),
            }
        )

    # Realize anything left at the already-computed proxy exit P&L.
    final_date = pd.Timestamp(dates["date"].iloc[-1])
    for pos in open_pos:
        if pd.Timestamp(pos["exit_date"]) > final_date:
            realized += float(pos["qty"]) * float(pos["pnl_usd"])
    if nav_rows:
        nav_rows[-1]["nav"] = capital + realized
        nav_rows[-1]["realized_pnl"] = realized
        nav_rows[-1]["unrealized_pnl"] = 0.0

    nav_df = pd.DataFrame(nav_rows)
    trades_df = pd.DataFrame(placed)
    executed = trades_df[trades_df["qty"] > 0].copy() if not trades_df.empty else pd.DataFrame()
    final_nav = float(nav_df["nav"].iloc[-1]) if not nav_df.empty else capital
    ret = (final_nav / capital - 1.0) * 100.0
    m = months(nav_df)
    summary = {
        "strategy": strategy,
        "final_nav": final_nav,
        "return_pct": ret,
        "monthly_pct": monthly_pct(ret, m),
        "max_drawdown_pct": max_dd_pct(nav_df["nav"]) if not nav_df.empty else 0.0,
        "trades": int(len(executed)),
        "skipped": int((trades_df["qty"] == 0).sum()) if not trades_df.empty else 0,
        "avg_qty": float(executed["qty"].mean()) if not executed.empty else 0.0,
        "max_qty": int(executed["qty"].max()) if not executed.empty else 0,
        "win_rate_pct": float((executed["sized_pnl_usd"] > 0).mean() * 100.0) if not executed.empty else 0.0,
        "worst_trade_usd": float(executed["sized_pnl_usd"].min()) if not executed.empty else 0.0,
        "max_open_trades": int(nav_df["open_trades"].max()) if not nav_df.empty else 0,
        "max_open_contracts": int(nav_df["open_contracts"].max()) if not nav_df.empty else 0,
        "max_open_risk_usd": float(nav_df["open_risk_usd"].max()) if not nav_df.empty else 0.0,
        "model_gate": bool(use_model_gate),
    }
    return nav_df, trades_df, summary


def replay_all(
    nav_template: pd.DataFrame,
    trades: pd.DataFrame,
    strategies: list[str],
    *,
    capital: float,
    risk_pct_per_trade: float,
    max_concurrent_risk_pct: float,
    use_model_gate: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    navs, trade_logs, summaries = [], [], []
    for s in strategies:
        nav, tr, sm = replay_strategy(
            s,
            nav_template,
            trades,
            capital=capital,
            risk_pct_per_trade=risk_pct_per_trade,
            max_concurrent_risk_pct=max_concurrent_risk_pct,
            use_model_gate=use_model_gate,
        )
        navs.append(nav)
        trade_logs.append(tr.assign(strategy=s) if not tr.empty else tr)
        summaries.append(sm)
    return pd.concat(navs, ignore_index=True), pd.concat(trade_logs, ignore_index=True), pd.DataFrame(summaries).sort_values("return_pct", ascending=False)


def summary_table(summary: pd.DataFrame, bench_row: dict[str, Any]) -> str:
    rows = []
    display = pd.concat([summary.copy(), pd.DataFrame([bench_row])], ignore_index=True)
    for _, r in display.iterrows():
        rows.append(
            {
                "strategy": r["strategy"],
                "return": pct(float(r["return_pct"])),
                "monthly": pct(float(r["monthly_pct"])),
                "max_dd": pct(float(r["max_drawdown_pct"])),
                "trades": "" if pd.isna(r.get("trades", np.nan)) else int(r["trades"]),
                "skipped": "" if pd.isna(r.get("skipped", np.nan)) else int(r["skipped"]),
                "avg_qty": "" if pd.isna(r.get("avg_qty", np.nan)) else f"{float(r['avg_qty']):.1f}",
                "max_qty": "" if pd.isna(r.get("max_qty", np.nan)) else int(r["max_qty"]),
                "max_open_contracts": "" if pd.isna(r.get("max_open_contracts", np.nan)) else int(r["max_open_contracts"]),
                "final_nav": usd(float(r["final_nav"])),
                "worst_trade": "" if pd.isna(r.get("worst_trade_usd", np.nan)) else usd(float(r["worst_trade_usd"])),
                "note": NOTES.get(r["strategy"], "benchmark"),
            }
        )
    return pd.DataFrame(rows).to_html(index=False, escape=False, classes="data-table")


def bench_summary(nav_template: pd.DataFrame, capital: float) -> dict[str, Any]:
    b = benchmark(nav_template, capital)
    ret = (float(b["nav"].iloc[-1]) / capital - 1.0) * 100.0
    return {
        "strategy": "SPY buy-and-hold benchmark",
        "return_pct": ret,
        "monthly_pct": monthly_pct(ret, months(b)),
        "max_drawdown_pct": max_dd_pct(b["nav"]),
        "trades": 1,
        "final_nav": float(b["nav"].iloc[-1]),
    }


def overview_chart(
    train_nav: pd.DataFrame,
    test_raw_nav: pd.DataFrame,
    test_gated_nav: pd.DataFrame,
    train_summary: pd.DataFrame,
    test_raw_summary: pd.DataFrame,
    test_gated_summary: pd.DataFrame,
    capital: float,
) -> str:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Train capital-aware NAV",
            "Held-out test raw capital-aware NAV",
            "Held-out test model-gated capital-aware NAV",
            "Held-out test return vs drawdown",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.08,
    )
    strategies = train_summary["strategy"].tolist()
    templates = [
        (train_nav, 1, 1, True),
        (test_raw_nav, 1, 2, False),
        (test_gated_nav, 2, 1, False),
    ]
    for nav_df, row, col, showlegend in templates:
        for s in strategies:
            w = nav_df[nav_df["strategy"] == s]
            fig.add_trace(
                go.Scatter(
                    x=w["date"],
                    y=w["nav"],
                    name=s,
                    mode="lines",
                    line=dict(color=COLORS.get(s, "#555555"), width=1.5),
                    showlegend=showlegend,
                ),
                row=row,
                col=col,
            )
        b = benchmark(nav_df, capital)
        fig.add_trace(
            go.Scatter(
                x=b["date"],
                y=b["nav"],
                name="SPY benchmark",
                mode="lines",
                line=dict(color=COLORS["benchmark"], width=2.3, dash="dot"),
                showlegend=showlegend,
            ),
            row=row,
            col=col,
        )
    fig.add_trace(
        go.Scatter(
            x=test_raw_summary["max_drawdown_pct"],
            y=test_raw_summary["return_pct"],
            text=test_raw_summary["strategy"],
            name="raw test",
            mode="markers",
            marker=dict(color="#e74c3c", size=13, opacity=0.78),
            hovertemplate="<b>%{text}</b><br>Return %{y:.1f}%<br>DD %{x:.1f}%<extra></extra>",
        ),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=test_gated_summary["max_drawdown_pct"],
            y=test_gated_summary["return_pct"],
            text=test_gated_summary["strategy"],
            name="trained-model gated",
            mode="markers",
            marker=dict(color="#16a085", size=13, symbol="diamond", opacity=0.85),
            hovertemplate="<b>%{text}</b><br>Return %{y:.1f}%<br>DD %{x:.1f}%<extra></extra>",
        ),
        row=2,
        col=2,
    )
    for r, c in [(1, 1), (1, 2), (2, 1)]:
        fig.update_yaxes(title_text="NAV", row=r, col=c)
    fig.update_xaxes(title_text="Max DD (%)", row=2, col=2)
    fig.update_yaxes(title_text="Return (%)", row=2, col=2)
    fig.update_layout(height=980, template="plotly_white", legend=dict(orientation="h", y=1.07), margin=dict(l=70, r=30, t=110, b=50))
    return fig.to_html(include_plotlyjs=True, full_html=False)


def strategy_chart(strategy: str, raw_nav: pd.DataFrame, gated_nav: pd.DataFrame, raw_trades: pd.DataFrame, gated_trades: pd.DataFrame, capital: float) -> str:
    rn = raw_nav[raw_nav["strategy"] == strategy]
    gn = gated_nav[gated_nav["strategy"] == strategy]
    rt = raw_trades[raw_trades["strategy"] == strategy]
    gt = gated_trades[(gated_trades["strategy"] == strategy) & (gated_trades["qty"] > 0)]
    b = benchmark(raw_nav, capital)
    color = COLORS.get(strategy, "#555555")
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.35, 0.37, 0.28],
        vertical_spacing=0.055,
        subplot_titles=("NAV raw vs model-gated vs benchmark", "SPY entries/exits and strike corridor", "Sized trade P&L"),
    )
    fig.add_trace(go.Scatter(x=rn["date"], y=rn["nav"], name="raw", mode="lines", line=dict(color=color, width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=gn["date"], y=gn["nav"], name="model-gated", mode="lines", line=dict(color=COLORS["gated"], dash="dash", width=2.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=b["date"], y=b["nav"], name="SPY benchmark", mode="lines", line=dict(color=COLORS["benchmark"], dash="dot", width=2.0)), row=1, col=1)
    fig.add_trace(go.Scatter(x=rn["date"], y=rn["spy_close"], name="SPY", mode="lines", line=dict(color="#111111", width=1.6)), row=2, col=1)
    fig.add_trace(go.Scatter(x=rt["entry_date"], y=rt["short_call"], name="short call", mode="markers", marker=dict(color="#c0392b", size=4, opacity=0.23)), row=2, col=1)
    fig.add_trace(go.Scatter(x=rt["entry_date"], y=rt["short_put"], name="short put", mode="markers", marker=dict(color="#27ae60", size=4, opacity=0.23)), row=2, col=1)
    fig.add_trace(
        go.Scatter(
            x=gt["entry_date"],
            y=gt["entry_spy"],
            name="model-gated entries",
            mode="markers",
            marker=dict(color=COLORS["gated"], size=np.clip(gt["qty"], 5, 18), symbol="triangle-up", opacity=0.75),
            customdata=np.stack(
                [
                    gt["qty"].astype(str),
                    gt["exit_date"].dt.date.astype(str),
                    gt["reason"],
                    gt["sized_pnl_usd"].map(usd),
                    gt["prob_range_lgbm"].map(lambda x: "" if pd.isna(x) else f"{x:.1%}"),
                ],
                axis=-1,
            )
            if len(gt)
            else None,
            hovertemplate="Entry %{x|%Y-%m-%d}<br>SPY $%{y:.2f}<br>Qty %{customdata[0]}<br>Exit %{customdata[1]}<br>Reason %{customdata[2]}<br>P&L %{customdata[3]}<br>P(range) %{customdata[4]}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(go.Scatter(x=gt["exit_date"], y=gt["exit_spy"], name="model-gated exits", mode="markers", marker=dict(color="#111111", size=6, symbol="x")), row=2, col=1)
    reason_colors = {"profit_target": "#27ae60", "stop_loss": "#c0392b", "dte_close": "#2980b9", "end_forced": "#7f8c8d"}
    for reason, sub in rt[rt["qty"] > 0].groupby("reason"):
        fig.add_trace(go.Scatter(x=sub["exit_date"], y=sub["sized_pnl_usd"], name=f"raw {reason}", mode="markers", marker=dict(color=reason_colors.get(reason, "#8e44ad"), size=5, opacity=0.26)), row=3, col=1)
    for reason, sub in gt.groupby("reason"):
        fig.add_trace(go.Scatter(x=sub["exit_date"], y=sub["sized_pnl_usd"], name=f"gated {reason}", mode="markers", marker=dict(color=reason_colors.get(reason, "#8e44ad"), size=8, symbol="diamond", opacity=0.85)), row=3, col=1)
    fig.add_hline(y=0, row=3, col=1, line_color="#555555", line_width=1)
    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="SPY / strikes", row=2, col=1)
    fig.update_yaxes(title_text="P&L", row=3, col=1)
    fig.update_layout(height=1030, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.04), title=dict(text=f"{strategy}<br><sup>{NOTES.get(strategy, '')}</sup>", x=0.01, xanchor="left"), margin=dict(l=70, r=30, t=115, b=50))
    return fig.to_html(include_plotlyjs=False, full_html=False)


def trade_table(strategy: str, trades: pd.DataFrame, max_rows: int = 260) -> str:
    w = trades[trades["strategy"] == strategy].copy().sort_values("exit_date")
    if len(w) > max_rows:
        skipped = w[w["qty"] == 0].tail(max_rows // 6)
        worst = w[w["qty"] > 0].nsmallest(max_rows // 4, "sized_pnl_usd")
        best = w[w["qty"] > 0].nlargest(max_rows // 4, "sized_pnl_usd")
        recent = w.tail(max_rows)
        w = pd.concat([skipped, worst, best, recent]).drop_duplicates().sort_values("exit_date").head(max_rows)
    out = pd.DataFrame(
        {
            "qty": w["qty"].astype(int),
            "model_take": w.get("model_take", pd.Series([""] * len(w))).map(lambda x: "" if pd.isna(x) else str(bool(x))),
            "p_range": w.get("prob_range_lgbm", pd.Series([np.nan] * len(w))).map(lambda x: "" if pd.isna(x) else f"{x:.1%}"),
            "entry": w["entry_date"].dt.date.astype(str),
            "exit": w["exit_date"].dt.date.astype(str),
            "reason": w["reason"],
            "structure": w["structure"],
            "credit": w["entry_credit"].map(lambda x: f"{x:.2f}"),
            "pt_debit": w["profit_target_debit"].map(lambda x: f"{x:.2f}"),
            "stop_debit": w["stop_loss_debit"].map(lambda x: f"{x:.2f}"),
            "exit_debit": w["exit_debit"].map(lambda x: f"{x:.2f}"),
            "pnl": w["sized_pnl_usd"].map(usd),
            "risk": w["sized_max_loss_usd"].map(usd),
        }
    )
    return out.to_html(index=False, escape=False, classes="data-table")


def build(root: Path, predictions: Path, capital: float, risk_pct: float, concurrent_pct: float, out: Path) -> Path:
    specs = load_specs()
    train_nav_template = pd.read_csv(root / "train" / "daily_nav.csv", parse_dates=["date"])
    test_nav_template = pd.read_csv(root / "test" / "daily_nav.csv", parse_dates=["date"])
    train_trades = add_trade_refs(pd.read_csv(root / "train" / "trades.csv"), specs)
    test_trades = merge_model_predictions(add_trade_refs(pd.read_csv(root / "test" / "trades.csv"), specs), predictions)
    strategies = pd.read_csv(root / "test" / "summary.csv")["strategy"].tolist()

    train_nav, train_exec, train_summary = replay_all(
        train_nav_template,
        train_trades,
        strategies,
        capital=capital,
        risk_pct_per_trade=risk_pct,
        max_concurrent_risk_pct=concurrent_pct,
        use_model_gate=False,
    )
    test_raw_nav, test_raw_exec, test_raw_summary = replay_all(
        test_nav_template,
        test_trades,
        strategies,
        capital=capital,
        risk_pct_per_trade=risk_pct,
        max_concurrent_risk_pct=concurrent_pct,
        use_model_gate=False,
    )
    test_gated_nav, test_gated_exec, test_gated_summary = replay_all(
        test_nav_template,
        test_trades,
        strategies,
        capital=capital,
        risk_pct_per_trade=risk_pct,
        max_concurrent_risk_pct=concurrent_pct,
        use_model_gate=True,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    stem = f"capital_aware_{int(capital)}_risk{str(risk_pct).replace('.', '_')}_cap{str(concurrent_pct).replace('.', '_')}"
    train_summary.to_csv(out.parent / f"{stem}_train_summary.csv", index=False)
    test_raw_summary.to_csv(out.parent / f"{stem}_test_raw_summary.csv", index=False)
    test_gated_summary.to_csv(out.parent / f"{stem}_test_gated_summary.csv", index=False)
    test_gated_exec.to_csv(out.parent / f"{stem}_test_gated_trades.csv", index=False)

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
      table.data-table td:nth-child(4), table.data-table th:nth-child(4),
      table.data-table td:nth-child(5), table.data-table th:nth-child(5),
      table.data-table td:nth-child(6), table.data-table th:nth-child(6),
      table.data-table td:nth-child(7), table.data-table th:nth-child(7) { text-align: left; }
      .small { font-size: 12px; color: #566573; }
    </style>
    """
    body = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Capital-Aware Aggressive Six</title>",
        css,
        "</head><body>",
        f"<h1>Capital-Aware Aggressive Six: ${capital:,.0f}</h1>",
        f"<p class='note'>This is not proportional scaling. Each trade chooses integer contract quantity from NAV: "
        f"<code>{risk_pct:.1f}%</code> max-loss risk per new trade, with <code>{concurrent_pct:.1f}%</code> max concurrent open max-loss exposure. "
        f"The held-out model-gated run uses the already-trained range model from <code>{predictions}</code>, threshold <code>{MODEL_GATE_THRESHOLD:.0%}</code>.</p>",
        "<div class='navlinks'>",
    ]
    for s in strategies:
        body.append(f"<a href='#{s}'>{s}</a>")
    body.extend(
        [
            "</div>",
            "<h2>Overview</h2>",
            overview_chart(train_nav, test_raw_nav, test_gated_nav, train_summary, test_raw_summary, test_gated_summary, capital),
            "<h3>Train Capital-Aware Raw Summary (70%) + Benchmark</h3><div class='scroll'>",
            summary_table(train_summary, bench_summary(train_nav_template, capital)),
            "</div>",
            "<h3>Held-Out Test Capital-Aware Raw Summary (30%) + Benchmark</h3><div class='scroll'>",
            summary_table(test_raw_summary, bench_summary(test_nav_template, capital)),
            "</div>",
            "<h3>Held-Out Test Capital-Aware With Already-Trained Model Gate</h3><div class='scroll'>",
            summary_table(test_gated_summary, bench_summary(test_nav_template, capital)),
            "</div>",
        ]
    )
    for s in strategies:
        body.extend(
            [
                f"<h2 id='{s}'>{s}</h2>",
                strategy_chart(s, test_raw_nav, test_gated_nav, test_raw_exec, test_gated_exec, capital),
                "<p class='small'>Trade table shows the model-gated capital-aware run. Marker size scales with qty.</p>",
                "<div class='scroll'>",
                trade_table(s, test_gated_exec),
                "</div>",
            ]
        )
    body.extend(["</body></html>"])
    out.write_text("\n".join(body), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--risk-pct", type=float, default=10.0)
    parser.add_argument("--concurrent-pct", type=float, default=100.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    out = args.out or args.root / f"aggressive_six_capital_aware_{int(args.capital)}.html"
    print(build(args.root, args.predictions, args.capital, args.risk_pct, args.concurrent_pct, out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
