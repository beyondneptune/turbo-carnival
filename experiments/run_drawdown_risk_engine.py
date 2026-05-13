"""Drawdown-aware risk governor for the aggressive strategies.

This tests the risk lever that actually matters: position sizing before entry.

The previous stop-engine test showed early exits cut too many recovering trades.
This script trains a local NN risk model on the 70% train window and uses it to
scale or block new trades on the 30% held-out test window.

Compared engines:
  - baseline sizing
  - static reduced-risk sizing
  - live drawdown governor
  - NN trade-risk governor
  - hybrid NN + live drawdown governor

No council, no external LLM/API calls.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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
    mark_trade,
    pct,
    train_scores,
    usd,
)
from experiments.run_optimized_combined_engine import (  # noqa: E402
    Recipe,
    add_combined_scores,
    safe_monthly,
    select_trades,
)


ENTRY_FEATURES = [
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
    "rule_score",
    "nn_score",
    "indicator_score",
    "combined_score",
    "combined_conviction",
    "risk_adjusted_score",
]


@dataclass(frozen=True)
class RiskConfig:
    name: str
    mode: str
    base_risk_pct: float
    concurrent_pct: float
    nn_threshold: float = 0.65
    nn_floor_mult: float = 0.25
    nn_strength: float = 0.85
    dd_profile: str = "none"
    max_bad_prob: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mode": self.mode,
            "base_risk_pct": self.base_risk_pct,
            "concurrent_pct": self.concurrent_pct,
            "nn_threshold": self.nn_threshold,
            "nn_floor_mult": self.nn_floor_mult,
            "nn_strength": self.nn_strength,
            "dd_profile": self.dd_profile,
            "max_bad_prob": self.max_bad_prob,
        }


def strategy_recipes() -> dict[str, Recipe]:
    return {
        "fixed_best_qwen": Recipe(
            name="fixed_best_qwen",
            selector="fixed_best_qwen",
            score_col="combined_score",
            top_n=1,
            positive_only=False,
            base_risk_pct=10.0,
            concurrent_pct=100.0,
            throttle="none",
            score_sizing=False,
        ),
        "champion_nn_top3": Recipe(
            name="champion_nn_top3",
            selector="top",
            score_col="nn_score",
            top_n=3,
            positive_only=False,
            base_risk_pct=30.0,
            concurrent_pct=250.0,
            throttle="none",
            score_sizing=False,
        ),
    }


def risk_label(df: pd.DataFrame) -> pd.Series:
    """Train target: trades that are bad enough to drive portfolio drawdown."""
    max_loss = df["max_loss_per_contract"].replace(0, np.nan).abs()
    pnl_pct = df["pnl_usd"] / max_loss
    return (
        (df["reason"].astype(str) == "stop_loss")
        | (pnl_pct < -0.08)
        | (df["pnl_usd"] < -75.0)
    ).astype(int)


def train_risk_model(train_scored: pd.DataFrame) -> tuple[Any | None, dict[str, Any]]:
    samples = train_scored.copy()
    y = risk_label(samples)
    if y.nunique() < 2:
        return None, {"status": "no_model", "reason": "single-class labels", "samples": int(len(samples))}
    X = samples[ENTRY_FEATURES].copy()
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(64, 32),
            alpha=0.002,
            max_iter=600,
            random_state=42,
            early_stopping=True,
        ),
    )
    model.fit(X, y)
    pred = model.predict_proba(X)[:, 1]
    try:
        auc = float(roc_auc_score(y, pred))
    except ValueError:
        auc = float("nan")
    return model, {
        "status": "trained",
        "samples": int(len(samples)),
        "positives": int(y.sum()),
        "positive_rate_pct": float(y.mean() * 100.0),
        "train_auc": auc,
        "label": "stop_loss OR pnl_pct_max_loss < -8% OR pnl_usd < -75",
    }


def bad_prob(model: Any | None, row: pd.Series) -> float:
    if model is None:
        return 0.0
    x = pd.DataFrame([row])[ENTRY_FEATURES]
    return float(model.predict_proba(x)[0, 1])


def dd_multiplier(profile: str, live_dd_pct: float) -> tuple[float, float]:
    """Return risk multiplier and concurrent-cap multiplier."""
    if profile == "none":
        return 1.0, 1.0
    if profile == "soft":
        if live_dd_pct <= -15:
            return 0.25, 0.45
        if live_dd_pct <= -10:
            return 0.45, 0.65
        if live_dd_pct <= -5:
            return 0.70, 0.85
        return 1.0, 1.0
    if profile == "hard":
        if live_dd_pct <= -12:
            return 0.12, 0.30
        if live_dd_pct <= -8:
            return 0.30, 0.50
        if live_dd_pct <= -4:
            return 0.60, 0.75
        return 1.0, 1.0
    if profile == "target10":
        if live_dd_pct <= -9:
            return 0.05, 0.20
        if live_dd_pct <= -7:
            return 0.20, 0.40
        if live_dd_pct <= -4:
            return 0.50, 0.70
        return 1.0, 1.0
    raise ValueError(f"unknown dd profile {profile}")


def risk_multiplier(config: RiskConfig, row: pd.Series, live_dd_pct: float, model: Any | None) -> tuple[float, bool, float]:
    dd_risk_mult, _ = dd_multiplier(config.dd_profile, live_dd_pct)
    p_bad = bad_prob(model, row) if config.mode in {"nn", "hybrid"} else 0.0
    blocked = False
    nn_mult = 1.0
    if config.mode in {"nn", "hybrid"}:
        if p_bad >= config.max_bad_prob:
            blocked = True
            nn_mult = 0.0
        elif p_bad >= config.nn_threshold:
            nn_mult = config.nn_floor_mult
        else:
            nn_mult = float(np.clip(1.0 - config.nn_strength * p_bad, config.nn_floor_mult, 1.15))
    if config.mode == "static":
        dd_risk_mult = 1.0
    return dd_risk_mult * nn_mult, blocked, p_bad


def replay_risk_engine(
    engine_name: str,
    recipe: Recipe,
    config: RiskConfig,
    market: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    capital: float,
    model: Any | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dates = market.drop_duplicates("date").sort_values("date").copy()
    candidates = selected[selected["take"]].copy().sort_values(["entry_date", "selector_score"], ascending=[True, False])
    open_pos: list[dict[str, Any]] = []
    realized = 0.0
    peak_nav = capital
    nav_rows = []
    placed = []
    blocked_by_model = 0
    reduced_entries = 0

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
        peak_nav = max(peak_nav, nav_now)
        live_dd_pct = (nav_now / max(peak_nav, 1.0) - 1.0) * 100.0
        _, cap_mult = dd_multiplier(config.dd_profile, live_dd_pct)
        risk_cap = max(nav_now, 0.0) * config.concurrent_pct * cap_mult / 100.0
        open_risk = sum(float(pos["qty"]) * float(pos["max_loss_per_contract"]) for pos in open_pos)

        todays = candidates[candidates["entry_date"] == dt].sort_values("selector_score", ascending=False)
        for _, tr in todays.iterrows():
            mult, blocked, p_bad = risk_multiplier(config, tr, live_dd_pct, model)
            if blocked:
                blocked_by_model += 1
                continue
            if mult < 0.999:
                reduced_entries += 1
            nav_for_size = max(nav_now, 0.0)
            per_loss = max(float(tr["max_loss_per_contract"]), 1.0)
            ideal_qty = int((nav_for_size * config.base_risk_pct * mult / 100.0) // per_loss)
            room_qty = int(max(risk_cap - open_risk, 0.0) // per_loss)
            qty = max(0, min(ideal_qty, room_qty))
            if qty <= 0:
                continue
            row = tr.to_dict()
            row["engine"] = engine_name
            row["qty"] = qty
            row["bad_prob"] = p_bad
            row["risk_mult"] = mult
            row["live_dd_at_entry"] = live_dd_pct
            row["risk_pct_used"] = config.base_risk_pct * mult
            row["sized_pnl_usd"] = qty * float(tr["pnl_usd"])
            row["sized_max_loss_usd"] = qty * per_loss
            placed.append(row)
            open_pos.append(row)
            open_risk += qty * per_loss

        unrealized = sum(float(pos["qty"]) * mark_trade(pd.Series(pos), mrow, dt) for pos in open_pos)
        nav_now = capital + realized + unrealized
        peak_nav = max(peak_nav, nav_now)
        nav_rows.append(
            {
                "date": dt,
                "engine": engine_name,
                "nav": nav_now,
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "open_trades": len(open_pos),
                "open_contracts": int(sum(pos["qty"] for pos in open_pos)),
                "open_risk_usd": float(sum(float(pos["qty"]) * float(pos["max_loss_per_contract"]) for pos in open_pos)),
                "spy_close": float(mrow["spy_close"]),
                "drawdown_pct": (nav_now / max(peak_nav, 1.0) - 1.0) * 100.0,
                "cap_mult": cap_mult,
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
    final_nav = float(nav_df["nav"].iloc[-1]) if not nav_df.empty else capital
    ret = (final_nav / capital - 1.0) * 100.0
    peak = nav_df["nav"].cummax()
    dd = nav_df["nav"] / peak - 1.0 if not nav_df.empty else pd.Series([0.0])
    summary = {
        "engine": engine_name,
        "recipe": recipe.name,
        "mode": config.mode,
        "base_risk_pct": config.base_risk_pct,
        "concurrent_pct": config.concurrent_pct,
        "dd_profile": config.dd_profile,
        "nn_threshold": config.nn_threshold,
        "nn_floor_mult": config.nn_floor_mult,
        "max_bad_prob": config.max_bad_prob,
        "final_nav": final_nav,
        "return_pct": ret,
        "monthly_pct": safe_monthly(final_nav, capital, pd.Timestamp(dates["date"].iloc[0]), pd.Timestamp(dates["date"].iloc[-1])),
        "max_drawdown_pct": float(dd.min() * 100.0),
        "trades": int(len(trades_df)),
        "avg_qty": float(trades_df["qty"].mean()) if not trades_df.empty else 0.0,
        "max_qty": int(trades_df["qty"].max()) if not trades_df.empty else 0,
        "max_open_contracts": int(nav_df["open_contracts"].max()) if not nav_df.empty else 0,
        "max_open_risk_usd": float(nav_df["open_risk_usd"].max()) if not nav_df.empty else 0.0,
        "worst_trade_usd": float(trades_df["sized_pnl_usd"].min()) if not trades_df.empty else 0.0,
        "win_rate_pct": float((trades_df["sized_pnl_usd"] > 0).mean() * 100.0) if not trades_df.empty else 0.0,
        "blocked_by_model": blocked_by_model,
        "reduced_entries": reduced_entries,
    }
    return nav_df, trades_df, summary


def config_grid(recipe_name: str) -> list[RiskConfig]:
    if recipe_name == "fixed_best_qwen":
        return [
            RiskConfig("baseline_10_100", "static", 10.0, 100.0),
            RiskConfig("static_7_5_75", "static", 7.5, 75.0),
            RiskConfig("static_5_50", "static", 5.0, 50.0),
            RiskConfig("dd_soft_10_100", "dd", 10.0, 100.0, dd_profile="soft"),
            RiskConfig("dd_target10_10_100", "dd", 10.0, 100.0, dd_profile="target10"),
            RiskConfig("nn_10_100_t55", "nn", 10.0, 100.0, nn_threshold=0.55, max_bad_prob=0.90),
            RiskConfig("nn_10_100_t65", "nn", 10.0, 100.0, nn_threshold=0.65, max_bad_prob=0.95),
            RiskConfig("hybrid_10_100_t55", "hybrid", 10.0, 100.0, nn_threshold=0.55, dd_profile="soft", max_bad_prob=0.90),
            RiskConfig("hybrid_target10_10_100", "hybrid", 10.0, 100.0, nn_threshold=0.55, dd_profile="target10", max_bad_prob=0.90),
            RiskConfig("hybrid_12_5_100_t55", "hybrid", 12.5, 100.0, nn_threshold=0.55, dd_profile="soft", max_bad_prob=0.90),
        ]
    return [
        RiskConfig("baseline_30_250", "static", 30.0, 250.0),
        RiskConfig("static_15_150", "static", 15.0, 150.0),
        RiskConfig("static_10_100", "static", 10.0, 100.0),
        RiskConfig("dd_soft_20_200", "dd", 20.0, 200.0, dd_profile="soft"),
        RiskConfig("dd_hard_20_200", "dd", 20.0, 200.0, dd_profile="hard"),
        RiskConfig("nn_20_200_t55", "nn", 20.0, 200.0, nn_threshold=0.55, max_bad_prob=0.90),
        RiskConfig("hybrid_20_200_t55", "hybrid", 20.0, 200.0, nn_threshold=0.55, dd_profile="soft", max_bad_prob=0.90),
        RiskConfig("hybrid_target10_15_150", "hybrid", 15.0, 150.0, nn_threshold=0.55, dd_profile="target10", max_bad_prob=0.90),
    ]


def train_choose_configs(
    recipe_name: str,
    recipe: Recipe,
    selected_train: pd.DataFrame,
    train_market: pd.DataFrame,
    capital: float,
    model: Any | None,
) -> tuple[list[RiskConfig], pd.DataFrame]:
    rows = []
    cfgs = config_grid(recipe_name)
    for cfg in cfgs:
        _, _, sm = replay_risk_engine(f"{recipe_name}_{cfg.name}", recipe, cfg, train_market, selected_train, capital=capital, model=model)
        sm["objective_return"] = sm["return_pct"]
        sm["objective_balanced"] = sm["return_pct"] - 2.25 * abs(sm["max_drawdown_pct"])
        sm["objective_dd10"] = np.where(sm["max_drawdown_pct"] >= -12.0, sm["return_pct"], sm["return_pct"] - 300.0)
        rows.append(sm)
    train_summary = pd.DataFrame(rows)
    chosen_names = []
    chosen_names.append(str(train_summary.sort_values("objective_return", ascending=False).iloc[0]["engine"]).replace(f"{recipe_name}_", ""))
    chosen_names.append(str(train_summary.sort_values("objective_balanced", ascending=False).iloc[0]["engine"]).replace(f"{recipe_name}_", ""))
    chosen_names.append(str(train_summary.sort_values("objective_dd10", ascending=False).iloc[0]["engine"]).replace(f"{recipe_name}_", ""))
    # Always keep baseline and simple half-risk for transparency.
    if cfgs:
        chosen_names.append(cfgs[0].name)
    if recipe_name == "fixed_best_qwen":
        chosen_names.append("static_5_50")
    else:
        chosen_names.append("static_10_100")
    by_name = {c.name: c for c in cfgs}
    chosen = [by_name[n] for n in dict.fromkeys(chosen_names) if n in by_name]
    return chosen, train_summary


def html_summary_table(df: pd.DataFrame) -> str:
    w = df.copy()
    for c in ("return_pct", "monthly_pct", "max_drawdown_pct", "win_rate_pct"):
        w[c] = w[c].map(lambda x: "" if pd.isna(x) else pct(float(x)))
    for c in ("final_nav", "worst_trade_usd", "max_open_risk_usd"):
        w[c] = w[c].map(lambda x: "" if pd.isna(x) else usd(float(x)))
    cols = [
        "engine",
        "recipe",
        "mode",
        "base_risk_pct",
        "concurrent_pct",
        "dd_profile",
        "return_pct",
        "monthly_pct",
        "max_drawdown_pct",
        "trades",
        "blocked_by_model",
        "reduced_entries",
        "win_rate_pct",
        "final_nav",
        "worst_trade_usd",
        "max_open_risk_usd",
    ]
    return w[cols].to_html(index=False, escape=False, classes="data-table")


def html_trade_table(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "<p>No trades.</p>"
    w = trades.copy().sort_values(["engine", "entry_date"])
    out = pd.DataFrame(
        {
            "engine": w["engine"],
            "entry": pd.to_datetime(w["entry_date"]).dt.date.astype(str),
            "exit": pd.to_datetime(w["exit_date"]).dt.date.astype(str),
            "strategy": w["strategy"],
            "reason": w["reason"],
            "qty": w["qty"].astype(int),
            "risk_mult": w["risk_mult"].map(lambda x: f"{float(x):.2f}"),
            "risk_pct": w["risk_pct_used"].map(lambda x: f"{float(x):.2f}%"),
            "bad_prob": w["bad_prob"].map(lambda x: f"{float(x):.2f}"),
            "live_dd": w["live_dd_at_entry"].map(lambda x: pct(float(x))),
            "legs": w["structure"],
            "pnl": w["sized_pnl_usd"].map(lambda x: usd(float(x))),
            "risk": w["sized_max_loss_usd"].map(lambda x: usd(float(x))),
        }
    )
    return out.to_html(index=False, escape=False, classes="data-table")


def build_report(
    out_dir: Path,
    summary: pd.DataFrame,
    navs: pd.DataFrame,
    trades: pd.DataFrame,
    test_market: pd.DataFrame,
    capital: float,
    model_meta: dict[str, Any],
    chosen_configs: dict[str, list[dict[str, Any]]],
    train_search: pd.DataFrame,
) -> Path:
    bench = benchmark(test_market, capital)
    bret = (float(bench["nav"].iloc[-1]) / capital - 1.0) * 100.0
    bpeak = bench["nav"].cummax()
    bench_row = {
        "engine": "SPY benchmark",
        "recipe": "buy_hold",
        "mode": "",
        "base_risk_pct": "",
        "concurrent_pct": "",
        "dd_profile": "",
        "final_nav": float(bench["nav"].iloc[-1]),
        "return_pct": bret,
        "monthly_pct": safe_monthly(float(bench["nav"].iloc[-1]), capital, pd.Timestamp(bench["date"].iloc[0]), pd.Timestamp(bench["date"].iloc[-1])),
        "max_drawdown_pct": float((bench["nav"] / bpeak - 1.0).min() * 100.0),
        "trades": 1,
        "blocked_by_model": 0,
        "reduced_entries": 0,
        "win_rate_pct": np.nan,
        "worst_trade_usd": np.nan,
        "max_open_risk_usd": capital,
    }
    summary_plus = pd.concat([summary.copy(), pd.DataFrame([bench_row])], ignore_index=True)

    palette = ["#111827", "#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#be123c", "#4b5563"]
    engines = summary["engine"].tolist()
    color_map = {e: palette[i % len(palette)] for i, e in enumerate(engines)}
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.55, 0.22, 0.23],
        vertical_spacing=0.08,
        subplot_titles=("Held-out NAV: drawdown risk engine", "Drawdown", "Return vs drawdown"),
    )
    for engine, w in navs.groupby("engine"):
        color = color_map.get(engine, "#334155")
        fig.add_trace(go.Scatter(x=w["date"], y=w["nav"], mode="lines", name=engine, line=dict(color=color, width=2.0)), row=1, col=1)
        peak = w["nav"].cummax()
        dd = w["nav"] / peak - 1.0
        fig.add_trace(go.Scatter(x=w["date"], y=dd * 100.0, mode="lines", name=f"{engine} DD", showlegend=False, line=dict(color=color, width=1.2)), row=2, col=1)
    fig.add_trace(go.Scatter(x=bench["date"], y=bench["nav"], mode="lines", name="SPY benchmark", line=dict(color="#64748b", dash="dot", width=2.0)), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=summary["max_drawdown_pct"],
            y=summary["return_pct"],
            text=summary["engine"],
            mode="markers+text",
            textposition="top center",
            marker=dict(size=13, color=[color_map[e] for e in summary["engine"]]),
            hovertemplate="<b>%{text}</b><br>Return %{y:.1f}%<br>DD %{x:.1f}%<extra></extra>",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="NAV", row=1, col=1)
    fig.update_yaxes(title_text="DD %", row=2, col=1)
    fig.update_xaxes(title_text="Max DD %", row=3, col=1)
    fig.update_yaxes(title_text="Return %", row=3, col=1)
    fig.update_layout(height=980, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.06), margin=dict(l=70, r=30, t=110, b=50))

    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #17202a; }
      h1 { margin-bottom: 4px; }
      h2 { margin-top: 34px; padding-top: 18px; border-top: 1px solid #d7dbdd; }
      .note { color: #566573; max-width: 1180px; line-height: 1.45; }
      .scroll { overflow-x: auto; border: 1px solid #eee; padding: 8px; margin-bottom: 18px; }
      table.data-table { border-collapse: collapse; width: 100%; font-size: 12px; }
      table.data-table th, table.data-table td { border: 1px solid #d6dbdf; padding: 6px 8px; text-align: right; white-space: nowrap; }
      table.data-table th { background: #f2f3f4; position: sticky; top: 0; z-index: 1; }
      table.data-table td:first-child, table.data-table th:first-child,
      table.data-table td:nth-child(2), table.data-table th:nth-child(2),
      table.data-table td:nth-child(3), table.data-table th:nth-child(3),
      table.data-table td:nth-child(4), table.data-table th:nth-child(4) { text-align: left; }
      pre { background:#f7f9f9; padding:12px; border:1px solid #e5e8e8; overflow-x:auto; }
    </style>
    """
    report = out_dir / "drawdown_risk_engine_report.html"
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Drawdown Risk Engine</title>",
        css,
        "</head><body>",
        f"<h1>Drawdown Risk Engine (${capital:,.0f})</h1>",
        "<p class='note'>This trains a local NN to identify bad entry regimes and uses it with live drawdown throttles to reduce max drawdown. "
        "The selected governors are optimized on train and evaluated on held-out test.</p>",
        "<h2>Held-Out Chart</h2>",
        fig.to_html(include_plotlyjs=True, full_html=False),
        "<h2>Held-Out Summary + Benchmark</h2><div class='scroll'>",
        html_summary_table(summary_plus),
        "</div>",
        "<h2>NN Risk Model</h2>",
        f"<pre>{json.dumps(model_meta, indent=2)}</pre>",
        "<h2>Chosen Risk Configs</h2>",
        f"<pre>{json.dumps(chosen_configs, indent=2)}</pre>",
        "<h2>Train Search</h2><div class='scroll'>",
        html_summary_table(train_search.sort_values("return_pct", ascending=False)),
        "</div>",
        "<h2>Entries / Risk Scaling / P&L</h2><div class='scroll'>",
        html_trade_table(trades),
        "</div>",
        "</body></html>",
    ]
    report.write_text("\n".join(html), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--capital", type=float, default=100_000.0)
    args = parser.parse_args()

    out_dir = OUT_ROOT / f"drawdown_risk_engine_{pd.Timestamp.now('UTC').strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = load_specs()
    train_market, train_trades = load_period(args.root, "train", specs)
    test_market, test_trades = load_period(args.root, "test", specs)
    train_scored, _ = train_scores(train_trades, train_trades)
    test_scored, _ = train_scores(train_trades, test_trades)
    train_scored = add_combined_scores(train_scored, train_trades)
    test_scored = add_combined_scores(test_scored, train_trades)

    model, model_meta = train_risk_model(train_scored)
    selected_train_by_recipe = {}
    selected_test_by_recipe = {}
    for name, recipe in strategy_recipes().items():
        selected_train_by_recipe[name] = select_trades(train_scored, recipe)
        selected_test_by_recipe[name] = select_trades(test_scored, recipe)

    chosen_configs: dict[str, list[dict[str, Any]]] = {}
    train_search_frames = []
    navs, trades, rows = [], [], []
    for name, recipe in strategy_recipes().items():
        chosen, train_summary = train_choose_configs(
            name,
            recipe,
            selected_train_by_recipe[name],
            train_market,
            args.capital,
            model,
        )
        chosen_configs[name] = [c.to_dict() for c in chosen]
        train_search_frames.append(train_summary)
        for cfg in chosen:
            engine_name = f"{name}_{cfg.name}"
            nav, tr, sm = replay_risk_engine(
                engine_name,
                recipe,
                cfg,
                test_market,
                selected_test_by_recipe[name],
                capital=args.capital,
                model=model,
            )
            navs.append(nav)
            trades.append(tr)
            rows.append(sm)

    nav_df = pd.concat(navs, ignore_index=True)
    trade_df = pd.concat(trades, ignore_index=True)
    summary = pd.DataFrame(rows).sort_values(["recipe", "return_pct"], ascending=[True, False])
    train_search = pd.concat(train_search_frames, ignore_index=True)

    nav_df.to_csv(out_dir / "drawdown_risk_nav.csv", index=False)
    trade_df.to_csv(out_dir / "drawdown_risk_trades.csv", index=False)
    summary.to_csv(out_dir / "drawdown_risk_summary.csv", index=False)
    train_search.to_csv(out_dir / "drawdown_risk_train_search.csv", index=False)
    train_scored.to_csv(out_dir / "train_candidates_risk_scored.csv", index=False)
    test_scored.to_csv(out_dir / "test_candidates_risk_scored.csv", index=False)
    (out_dir / "risk_model_meta.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")
    (out_dir / "chosen_risk_configs.json").write_text(json.dumps(chosen_configs, indent=2), encoding="utf-8")

    report = build_report(
        out_dir,
        summary,
        nav_df,
        trade_df,
        test_market,
        args.capital,
        model_meta,
        chosen_configs,
        train_search,
    )
    print("\n=== HELD-OUT DRAWDOWN RISK ENGINE RESULTS ===")
    print(summary.to_string(index=False))
    print(f"\nReport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
