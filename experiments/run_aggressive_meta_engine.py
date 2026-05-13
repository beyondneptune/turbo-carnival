"""Aggressive meta-engine over the six SPY strategy candidates.

This is the "strategy picker" experiment:
  - Train on the first 70% window.
  - Learn which of the six strategy entries work under market regimes.
  - Test on the held-out 30% window with one shared $100k portfolio.
  - Use integer contract sizing and shared risk caps.

Selectors included:
  - raw_stack: no selector; take every candidate until risk cap blocks it.
  - rule_regime_top3: RSI/VRP/IV/trend regime table learned on train.
  - nn_top3: sklearn MLP classifier/regressor learned on train.
  - ensemble_top4: rule + NN + indicator score.
  - ensemble_max_stack: more aggressive, all positive ensemble scores.

No Kimi, no LLM. This uses local data only.
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
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_spy_core_strategy_grid import price_condor_value  # noqa: E402


DEFAULT_ROOT = REPO_ROOT / "experiments" / "runs" / "aggressive_six_70_30_20260510T234527Z"
SPEC_JSON = REPO_ROOT / "fullonriskmodelterminator" / "additional_six_spy_core_specs.json"
OUT_ROOT = REPO_ROOT / "experiments" / "runs"

ROUNDTRIP_COMMISSION = 0.65 * 8.0
RANDOM_STATE = 42

COLORS = {
    "raw_stack": "#7f8c8d",
    "rule_regime_top3": "#2980b9",
    "nn_top3": "#8e44ad",
    "ensemble_top4": "#27ae60",
    "ensemble_max_stack": "#d35400",
    "SPY benchmark": "#111111",
}


def pct(x: float) -> str:
    return "" if pd.isna(x) else f"{x:+.2f}%"


def usd(x: float) -> str:
    return "" if pd.isna(x) else f"${x:,.2f}"


def load_specs() -> dict[str, dict[str, Any]]:
    raw = json.loads(SPEC_JSON.read_text())
    return {s["name"]: s for s in raw["strategies"]}


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def market_features(nav_path: Path) -> pd.DataFrame:
    nav = pd.read_csv(nav_path, parse_dates=["date"])
    m = nav[["date", "spy_close", "iv", "vrp20", "iv_rank_252"]].drop_duplicates("date").sort_values("date")
    m["ret_1d"] = m["spy_close"].pct_change(1)
    m["ret_3d"] = m["spy_close"].pct_change(3)
    m["ret_5d"] = m["spy_close"].pct_change(5)
    m["ret_10d"] = m["spy_close"].pct_change(10)
    m["ret_20d"] = m["spy_close"].pct_change(20)
    m["rv_10d"] = m["ret_1d"].rolling(10, min_periods=5).std() * math.sqrt(252)
    m["rv_20d"] = m["ret_1d"].rolling(20, min_periods=10).std() * math.sqrt(252)
    m["rsi_14"] = rsi(m["spy_close"], 14)
    m["sma_20"] = m["spy_close"].rolling(20, min_periods=10).mean()
    m["sma_50"] = m["spy_close"].rolling(50, min_periods=20).mean()
    m["dist_sma20"] = m["spy_close"] / m["sma_20"] - 1.0
    m["dist_sma50"] = m["spy_close"] / m["sma_50"] - 1.0
    m["drawdown_20d"] = m["spy_close"] / m["spy_close"].rolling(20, min_periods=10).max() - 1.0
    m["dayofweek"] = m["date"].dt.dayofweek
    m["month"] = m["date"].dt.month
    return m


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
    out["pnl_pct_max_loss"] = out["pnl_usd"] / out["max_loss_per_contract"].replace(0, np.nan)
    out["put_distance_pct"] = out["short_put"] / out["entry_spy"] - 1.0
    out["call_distance_pct"] = out["short_call"] / out["entry_spy"] - 1.0
    out["credit_to_width"] = out["entry_credit"] / out["wing_width"].replace(0, np.nan)
    out["dte"] = (out["expiry"] - out["entry_date"]).dt.days
    out["sid"] = out["strategy"].astype("category").cat.codes
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


def load_period(root: Path, period: str, specs: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = market_features(root / period / "daily_nav.csv")
    tr = add_trade_refs(pd.read_csv(root / period / "trades.csv"), specs)
    tr = tr.merge(m, left_on="entry_date", right_on="date", how="left").drop(columns=["date"])
    return m, tr


def add_regime_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["iv_bucket"] = pd.cut(out["iv_rank_252"].fillna(0.5), bins=[-0.01, 0.33, 0.66, 1.01], labels=["iv_low", "iv_mid", "iv_high"])
    out["rsi_bucket"] = pd.cut(out["rsi_14"].fillna(50), bins=[-1, 35, 65, 101], labels=["rsi_low", "rsi_mid", "rsi_high"])
    out["trend_bucket"] = np.where(out["ret_20d"].fillna(0) >= 0, "trend_up", "trend_down")
    out["vrp_bucket"] = np.where(out["vrp20"].fillna(0) >= 0, "vrp_pos", "vrp_neg")
    out["regime_key"] = (
        out["strategy"].astype(str)
        + "|"
        + out["iv_bucket"].astype(str)
        + "|"
        + out["rsi_bucket"].astype(str)
        + "|"
        + out["trend_bucket"].astype(str)
        + "|"
        + out["vrp_bucket"].astype(str)
    )
    return out


FEATURES = [
    "entry_spy",
    "entry_iv",
    "entry_credit",
    "max_loss_per_contract",
    "credit_to_width",
    "put_distance_pct",
    "call_distance_pct",
    "dte",
    "iv",
    "vrp20",
    "iv_rank_252",
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "ret_20d",
    "rv_10d",
    "rv_20d",
    "rsi_14",
    "dist_sma20",
    "dist_sma50",
    "drawdown_20d",
    "dayofweek",
    "month",
    "sid",
]


def train_scores(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    train = add_regime_columns(train)
    test = add_regime_columns(test)

    strat_mean = train.groupby("strategy")["pnl_pct_max_loss"].mean().to_dict()
    global_mean = float(train["pnl_pct_max_loss"].mean())
    grouped = (
        train.groupby("regime_key")
        .agg(mean_pnl=("pnl_pct_max_loss", "mean"), win_rate=("pnl_usd", lambda s: float((s > 0).mean())), n=("pnl_usd", "size"))
        .reset_index()
    )
    regime_map = grouped.set_index("regime_key").to_dict("index")

    def rule_score(row: pd.Series) -> float:
        rec = regime_map.get(row["regime_key"])
        fallback = strat_mean.get(row["strategy"], global_mean)
        if not rec:
            return float(fallback)
        n = float(rec["n"])
        return float((rec["mean_pnl"] * n + fallback * 8.0) / (n + 8.0))

    train["rule_score"] = train.apply(rule_score, axis=1)
    test["rule_score"] = test.apply(rule_score, axis=1)

    X_train = pd.get_dummies(train[FEATURES + ["strategy"]], columns=["strategy"], dtype=float)
    X_test = pd.get_dummies(test[FEATURES + ["strategy"]], columns=["strategy"], dtype=float).reindex(columns=X_train.columns, fill_value=0.0)
    y_reg = train["pnl_pct_max_loss"].fillna(0.0).clip(-2.0, 2.0)
    y_cls = (train["pnl_usd"] > 0).astype(int)

    reg = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPRegressor(hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=800, random_state=RANDOM_STATE, early_stopping=True),
    )
    clf = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(64, 32), alpha=0.001, max_iter=800, random_state=RANDOM_STATE, early_stopping=True),
    )
    reg.fit(X_train, y_reg)
    clf.fit(X_train, y_cls)

    test["nn_pnl_score"] = reg.predict(X_test)
    test["nn_win_prob"] = clf.predict_proba(X_test)[:, 1]
    test["nn_score"] = test["nn_pnl_score"] * (0.5 + test["nn_win_prob"])

    # Hand-built indicator score: short-vol likes positive VRP, high-ish IV,
    # neutral RSI, small recent directional shock, and credit-to-width.
    test["indicator_score"] = (
        test["vrp20"].fillna(0) * 2.0
        + test["iv_rank_252"].fillna(0.5) * 0.25
        + test["credit_to_width"].fillna(0) * 0.8
        - (test["rsi_14"].fillna(50).sub(50).abs() / 100.0) * 0.35
        - test["ret_5d"].fillna(0).abs() * 1.5
        - test["drawdown_20d"].fillna(0).abs() * 0.5
    )
    for col in ("rule_score", "nn_score", "indicator_score"):
        mu = test[col].mean()
        sd = test[col].std()
        test[f"{col}_z"] = (test[col] - mu) / (sd if sd and np.isfinite(sd) else 1.0)
    test["ensemble_score"] = test["rule_score_z"] + test["nn_score_z"] + test["indicator_score_z"]
    train_meta = {
        "strategy_mean_pnl_pct_max_loss": {k: float(v) for k, v in strat_mean.items()},
        "global_mean_pnl_pct_max_loss": global_mean,
        "regime_rows": int(len(grouped)),
    }
    return test, train_meta


def select_variant(test: pd.DataFrame, variant: str) -> pd.DataFrame:
    out = test.copy()
    out["take"] = False
    out["selector_score"] = 0.0
    score_col = {
        "raw_stack": None,
        "rule_regime_top3": "rule_score",
        "nn_top3": "nn_score",
        "ensemble_top4": "ensemble_score",
        "ensemble_max_stack": "ensemble_score",
    }[variant]
    if variant == "raw_stack":
        out["take"] = True
        out["selector_score"] = 1.0
        return out

    for _, idx in out.groupby("entry_date").groups.items():
        w = out.loc[list(idx)].copy()
        if variant == "ensemble_max_stack":
            chosen = w[w[score_col] > w[score_col].median()].sort_values(score_col, ascending=False).head(6).index
        else:
            n = 4 if variant == "ensemble_top4" else 3
            chosen = w.sort_values(score_col, ascending=False).head(n).index
        out.loc[chosen, "take"] = True
        out.loc[list(idx), "selector_score"] = out.loc[list(idx), score_col]
    return out


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


def replay_portfolio(
    variant: str,
    market: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    capital: float,
    risk_pct: float,
    concurrent_pct: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dates = market[["date", "spy_close", "iv"]].drop_duplicates("date").sort_values("date")
    candidates = selected[selected["take"]].copy().sort_values(["entry_date", "selector_score"], ascending=[True, False])
    open_pos: list[dict[str, Any]] = []
    realized = 0.0
    nav_rows: list[dict[str, Any]] = []
    placed: list[dict[str, Any]] = []

    for _, mrow in dates.iterrows():
        dt = pd.Timestamp(mrow["date"])
        still = []
        for pos in open_pos:
            if pd.Timestamp(pos["exit_date"]) <= dt:
                realized += float(pos["qty"]) * float(pos["pnl_usd"])
            else:
                still.append(pos)
        open_pos = still

        unrealized = sum(float(pos["qty"]) * mark_trade(pd.Series(pos), mrow, dt) for pos in open_pos)
        nav_now = capital + realized + unrealized
        risk_cap = max(nav_now, 0.0) * concurrent_pct / 100.0
        open_risk = sum(float(pos["qty"]) * float(pos["max_loss_per_contract"]) for pos in open_pos)

        todays = candidates[candidates["entry_date"] == dt].sort_values("selector_score", ascending=False)
        for _, tr in todays.iterrows():
            nav_for_size = max(nav_now, 0.0)
            per_loss = max(float(tr["max_loss_per_contract"]), 1.0)
            ideal_qty = int((nav_for_size * risk_pct / 100.0) // per_loss)
            room_qty = int(max(risk_cap - open_risk, 0.0) // per_loss)
            qty = max(0, min(ideal_qty, room_qty))
            row = tr.to_dict()
            row["variant"] = variant
            row["qty"] = qty
            row["sized_pnl_usd"] = qty * float(tr["pnl_usd"])
            row["sized_max_loss_usd"] = qty * per_loss
            row["skipped_reason"] = "" if qty > 0 else "risk_cap_or_nav"
            placed.append(row)
            if qty > 0:
                open_pos.append(row)
                open_risk += qty * per_loss

        unrealized = sum(float(pos["qty"]) * mark_trade(pd.Series(pos), mrow, dt) for pos in open_pos)
        nav_now = capital + realized + unrealized
        nav_rows.append(
            {
                "date": dt,
                "variant": variant,
                "nav": nav_now,
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "open_trades": len(open_pos),
                "open_contracts": int(sum(pos["qty"] for pos in open_pos)),
                "open_risk_usd": float(sum(float(pos["qty"]) * float(pos["max_loss_per_contract"]) for pos in open_pos)),
                "spy_close": float(mrow["spy_close"]),
            }
        )

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
    executed = trades_df[trades_df["qty"] > 0] if not trades_df.empty else pd.DataFrame()
    months_count = max((nav_df["date"].max() - nav_df["date"].min()).days / 30.4375, 1 / 30.4375)
    final_nav = float(nav_df["nav"].iloc[-1])
    ret = (final_nav / capital - 1.0) * 100.0
    peak = nav_df["nav"].cummax()
    dd = nav_df["nav"] / peak - 1.0
    summary = {
        "variant": variant,
        "final_nav": final_nav,
        "return_pct": ret,
        "monthly_pct": ((1 + ret / 100.0) ** (1 / months_count) - 1) * 100.0,
        "max_drawdown_pct": float(dd.min() * 100.0),
        "trades": int(len(executed)),
        "skipped": int((trades_df["qty"] == 0).sum()) if not trades_df.empty else 0,
        "avg_qty": float(executed["qty"].mean()) if not executed.empty else 0.0,
        "max_qty": int(executed["qty"].max()) if not executed.empty else 0,
        "max_open_contracts": int(nav_df["open_contracts"].max()) if not nav_df.empty else 0,
        "max_open_risk_usd": float(nav_df["open_risk_usd"].max()) if not nav_df.empty else 0.0,
        "worst_trade_usd": float(executed["sized_pnl_usd"].min()) if not executed.empty else 0.0,
        "win_rate_pct": float((executed["sized_pnl_usd"] > 0).mean() * 100.0) if not executed.empty else 0.0,
    }
    return nav_df, trades_df, summary


def benchmark(market: pd.DataFrame, capital: float) -> pd.DataFrame:
    b = market[["date", "spy_close"]].drop_duplicates("date").sort_values("date").copy()
    b["nav"] = capital * b["spy_close"] / float(b["spy_close"].iloc[0])
    b["variant"] = "SPY benchmark"
    return b


def summary_table(summary: pd.DataFrame, bench: pd.DataFrame, capital: float) -> str:
    months_count = max((bench["date"].max() - bench["date"].min()).days / 30.4375, 1 / 30.4375)
    bret = (float(bench["nav"].iloc[-1]) / capital - 1.0) * 100.0
    bpeak = bench["nav"].cummax()
    brow = {
        "variant": "SPY benchmark",
        "final_nav": float(bench["nav"].iloc[-1]),
        "return_pct": bret,
        "monthly_pct": ((1 + bret / 100) ** (1 / months_count) - 1) * 100,
        "max_drawdown_pct": float((bench["nav"] / bpeak - 1).min() * 100),
        "trades": 1,
        "skipped": "",
        "avg_qty": "",
        "max_qty": "",
        "max_open_contracts": "",
        "worst_trade_usd": "",
        "win_rate_pct": "",
    }
    d = pd.concat([summary.copy(), pd.DataFrame([brow])], ignore_index=True)
    rows = []
    for _, r in d.iterrows():
        rows.append(
            {
                "engine": r["variant"],
                "return": pct(float(r["return_pct"])),
                "monthly": pct(float(r["monthly_pct"])),
                "max_dd": pct(float(r["max_drawdown_pct"])),
                "final_nav": usd(float(r["final_nav"])),
                "trades": r["trades"],
                "skipped": r["skipped"],
                "avg_qty": "" if r["avg_qty"] == "" else f"{float(r['avg_qty']):.1f}",
                "max_qty": r["max_qty"],
                "max_open_contracts": r["max_open_contracts"],
                "worst_trade": "" if r["worst_trade_usd"] == "" else usd(float(r["worst_trade_usd"])),
                "win_rate": "" if r["win_rate_pct"] == "" else f"{float(r['win_rate_pct']):.1f}%",
            }
        )
    return pd.DataFrame(rows).to_html(index=False, escape=False, classes="data-table")


def overview_chart(navs: pd.DataFrame, summary: pd.DataFrame, market: pd.DataFrame, capital: float) -> str:
    bench = benchmark(market, capital)
    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.70, 0.30],
        subplot_titles=("Held-out test: meta-engine equity curves", "Return vs drawdown"),
        vertical_spacing=0.10,
    )
    for variant, w in navs.groupby("variant"):
        fig.add_trace(
            go.Scatter(
                x=w["date"],
                y=w["nav"],
                name=variant,
                mode="lines",
                line=dict(color=COLORS.get(variant, "#555555"), width=2),
            ),
            row=1,
            col=1,
        )
    fig.add_trace(
        go.Scatter(x=bench["date"], y=bench["nav"], name="SPY benchmark", mode="lines", line=dict(color="#111111", width=2.4, dash="dot")),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=summary["max_drawdown_pct"],
            y=summary["return_pct"],
            text=summary["variant"],
            mode="markers+text",
            textposition="top center",
            marker=dict(color=[COLORS.get(v, "#555555") for v in summary["variant"]], size=15, opacity=0.85),
            hovertemplate="<b>%{text}</b><br>Return %{y:.1f}%<br>DD %{x:.1f}%<extra></extra>",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_xaxes(title_text="Max DD (%)", row=2, col=1)
    fig.update_yaxes(title_text="Return (%)", row=2, col=1)
    fig.update_layout(height=900, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.05), margin=dict(l=70, r=30, t=100, b=50))
    return fig.to_html(include_plotlyjs=True, full_html=False)


def trade_table(trades: pd.DataFrame, max_rows: int = 500) -> str:
    if trades.empty:
        return "<p>No trades.</p>"
    w = trades[trades["qty"] > 0].copy().sort_values("exit_date")
    if len(w) > max_rows:
        worst = w.nsmallest(max_rows // 4, "sized_pnl_usd")
        best = w.nlargest(max_rows // 4, "sized_pnl_usd")
        recent = w.tail(max_rows)
        w = pd.concat([worst, best, recent]).drop_duplicates().sort_values("exit_date").head(max_rows)
    out = pd.DataFrame(
        {
            "engine": w["variant"],
            "date": w["entry_date"].dt.date.astype(str),
            "strategy": w["strategy"],
            "qty": w["qty"].astype(int),
            "score": w["selector_score"].map(lambda x: f"{x:+.3f}"),
            "exit": w["exit_date"].dt.date.astype(str),
            "reason": w["reason"],
            "credit": w["entry_credit"].map(lambda x: f"{x:.2f}"),
            "exit_debit": w["exit_debit"].map(lambda x: f"{x:.2f}"),
            "pnl": w["sized_pnl_usd"].map(usd),
            "risk": w["sized_max_loss_usd"].map(usd),
        }
    )
    return out.to_html(index=False, escape=False, classes="data-table")


def build_html(out_dir: Path, navs: pd.DataFrame, trades: pd.DataFrame, summary: pd.DataFrame, market: pd.DataFrame, capital: float, risk_pct: float, concurrent_pct: float, train_meta: dict[str, Any]) -> Path:
    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #17202a; }
      h1 { margin-bottom: 4px; }
      h2 { margin-top: 34px; padding-top: 18px; border-top: 1px solid #d7dbdd; }
      .note { color: #566573; max-width: 1200px; line-height: 1.45; }
      .scroll { overflow-x: auto; border: 1px solid #eee; padding: 8px; margin-bottom: 20px; }
      table.data-table { border-collapse: collapse; width: 100%; font-size: 12px; }
      table.data-table th, table.data-table td { border: 1px solid #d6dbdf; padding: 6px 8px; text-align: right; white-space: nowrap; }
      table.data-table th { background: #f2f3f4; position: sticky; top: 0; z-index: 1; }
      table.data-table td:first-child, table.data-table th:first-child,
      table.data-table td:nth-child(2), table.data-table th:nth-child(2),
      table.data-table td:nth-child(3), table.data-table th:nth-child(3),
      table.data-table td:nth-child(6), table.data-table th:nth-child(6),
      table.data-table td:nth-child(7), table.data-table th:nth-child(7) { text-align: left; }
      pre { background:#f7f9f9; padding:12px; border:1px solid #e5e8e8; overflow-x:auto; }
    </style>
    """
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Aggressive Meta Engine</title>",
        css,
        "</head><body>",
        f"<h1>Aggressive Meta-Engine (${capital:,.0f})</h1>",
        f"<p class='note'>No Kimi, no LLM. This trains selectors on the 70% window and tests on the held-out 30% window. "
        f"Portfolio sizing is shared across all selected strategies: {risk_pct:.1f}% max-loss risk per new trade, {concurrent_pct:.1f}% max concurrent max-loss exposure. "
        "The NN variant uses sklearn MLP only because PyTorch is not installed locally.</p>",
        "<h2>Overview</h2>",
        overview_chart(navs, summary, market, capital),
        "<h2>Summary + Benchmark</h2><div class='scroll'>",
        summary_table(summary, benchmark(market, capital), capital),
        "</div>",
        "<h2>Training Meta</h2>",
        f"<pre>{json.dumps(train_meta, indent=2)}</pre>",
        "<h2>Selected Trade Table</h2><div class='scroll'>",
        trade_table(trades),
        "</div>",
        "</body></html>",
    ]
    out_path = out_dir / "aggressive_meta_engine_report.html"
    out_path.write_text("\n".join(html), encoding="utf-8")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--risk-pct", type=float, default=15.0)
    parser.add_argument("--concurrent-pct", type=float, default=150.0)
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"aggressive_meta_engine_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = load_specs()
    train_market, train_trades = load_period(args.root, "train", specs)
    test_market, test_trades = load_period(args.root, "test", specs)
    scored_test, train_meta = train_scores(train_trades, test_trades)

    variants = ["raw_stack", "rule_regime_top3", "nn_top3", "ensemble_top4", "ensemble_max_stack"]
    navs, trades, summaries = [], [], []
    for variant in variants:
        selected = select_variant(scored_test, variant)
        nav, tr, sm = replay_portfolio(
            variant,
            test_market,
            selected,
            capital=args.capital,
            risk_pct=args.risk_pct,
            concurrent_pct=args.concurrent_pct,
        )
        navs.append(nav)
        trades.append(tr)
        summaries.append(sm)
    nav_df = pd.concat(navs, ignore_index=True)
    trade_df = pd.concat(trades, ignore_index=True)
    summary_df = pd.DataFrame(summaries).sort_values("return_pct", ascending=False)

    nav_df.to_csv(out_dir / "meta_nav.csv", index=False)
    trade_df.to_csv(out_dir / "meta_trades.csv", index=False)
    summary_df.to_csv(out_dir / "meta_summary.csv", index=False)
    scored_test.to_csv(out_dir / "test_candidates_scored.csv", index=False)
    (out_dir / "train_meta.json").write_text(json.dumps(train_meta, indent=2))
    report = build_html(out_dir, nav_df, trade_df, summary_df, test_market, args.capital, args.risk_pct, args.concurrent_pct, train_meta)

    print(summary_df.to_string(index=False))
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
