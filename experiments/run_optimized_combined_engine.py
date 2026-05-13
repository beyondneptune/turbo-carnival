"""Optimized combined engine over the aggressive six strategy set.

This is the "combine the good ones" experiment:

1. Load the existing 70/30 split.
2. Train the local NN/rule/indicator scorers on the 70% train window.
3. Build combined selector scores across all six high-performing strategies.
4. Optimize selector + risk + concurrent exposure + drawdown throttle on train.
5. Freeze the top recipes and evaluate them on the 30% held-out test.

No council and no external LLM/API calls. This uses local data only.
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


TOP_STRATEGIES = [
    "aggressive_daily_nogate_d20_30d_w10",
    "aggressive_daily_nogate_d16_30d_w10",
    "daily_ladder_vrp_positive_d16_45d_w5",
    "daily_ladder_vrp_positive_d16_30d_w5",
    "risk_qwen_risk_designer_risktest_d12_30d_lowdelta_pt30_sl20_c10",
    "execution_llama_allocator_designer_daily_ladder_vrp_positive_d12_30d_w5_stop_loss_1_5",
]

STRATEGY_NOTES = {
    "aggressive_daily_nogate_d20_30d_w10": "highest original aggressive no-gate",
    "aggressive_daily_nogate_d16_30d_w10": "similar, still huge DD",
    "daily_ladder_vrp_positive_d16_45d_w5": "high return, less wild",
    "daily_ladder_vrp_positive_d16_30d_w5": "lower DD aggressive candidate",
    "risk_qwen_risk_designer_risktest_d12_30d_lowdelta_pt30_sl20_c10": "best council risk-adjusted fixed strategy",
    "execution_llama_allocator_designer_daily_ladder_vrp_positive_d12_30d_w5_stop_loss_1_5": "second council candidate",
}

COLORS = {
    "optimized_combined": "#111827",
    "return_max": "#e11d48",
    "balanced": "#2563eb",
    "survivable": "#059669",
    "champion_nn_top3": "#7c3aed",
    "fixed_best_qwen": "#d97706",
    "SPY benchmark": "#64748b",
}


@dataclass(frozen=True)
class Recipe:
    name: str
    selector: str
    score_col: str
    top_n: int
    positive_only: bool
    base_risk_pct: float
    concurrent_pct: float
    throttle: str
    score_sizing: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "selector": self.selector,
            "score_col": self.score_col,
            "top_n": self.top_n,
            "positive_only": self.positive_only,
            "base_risk_pct": self.base_risk_pct,
            "concurrent_pct": self.concurrent_pct,
            "throttle": self.throttle,
            "score_sizing": self.score_sizing,
        }


def safe_monthly(final_nav: float, capital: float, start: pd.Timestamp, end: pd.Timestamp) -> float:
    months = max((end - start).days / 30.4375, 1 / 30.4375)
    if final_nav <= 0 or capital <= 0:
        return -100.0
    return ((final_nav / capital) ** (1.0 / months) - 1.0) * 100.0


def zscore(s: pd.Series) -> pd.Series:
    sd = s.std()
    if not sd or not np.isfinite(sd):
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def add_combined_scores(scored: pd.DataFrame, train_trades: pd.DataFrame) -> pd.DataFrame:
    """Add combined selector scores learned only from train."""
    out = scored.copy()
    stats = (
        train_trades.groupby("strategy")
        .agg(
            train_edge=("pnl_pct_max_loss", "mean"),
            train_win=("pnl_usd", lambda s: float((s > 0).mean())),
            train_worst=("pnl_pct_max_loss", "min"),
            train_n=("pnl_usd", "size"),
        )
        .reindex(TOP_STRATEGIES)
    )
    stats["strategy_edge_z"] = zscore(stats["train_edge"].fillna(stats["train_edge"].mean()))
    stats["strategy_win_z"] = zscore(stats["train_win"].fillna(stats["train_win"].mean()))
    stats["strategy_worst_z"] = zscore(stats["train_worst"].fillna(stats["train_worst"].mean()))
    stat_map = stats[["strategy_edge_z", "strategy_win_z", "strategy_worst_z"]].to_dict("index")

    out["strategy_edge_z"] = out["strategy"].map(lambda s: float(stat_map.get(s, {}).get("strategy_edge_z", 0.0)))
    out["strategy_win_z"] = out["strategy"].map(lambda s: float(stat_map.get(s, {}).get("strategy_win_z", 0.0)))
    out["strategy_worst_z"] = out["strategy"].map(lambda s: float(stat_map.get(s, {}).get("strategy_worst_z", 0.0)))

    out["combined_score"] = (
        0.42 * out["nn_score_z"].fillna(0.0)
        + 0.24 * out["rule_score_z"].fillna(0.0)
        + 0.18 * out["indicator_score_z"].fillna(0.0)
        + 0.10 * out["strategy_edge_z"].fillna(0.0)
        + 0.06 * out["strategy_win_z"].fillna(0.0)
    )
    out["combined_conviction"] = out["combined_score"] * (0.70 + 0.65 * out["nn_win_prob"].fillna(0.5))
    out["risk_adjusted_score"] = (
        out["combined_score"]
        + 0.25 * out["strategy_worst_z"].fillna(0.0)
        - 0.15 * zscore(out["max_loss_per_contract"].fillna(out["max_loss_per_contract"].median()))
    )
    out["raw_return_score"] = out["nn_score_z"].fillna(0.0) + 0.35 * out["combined_score"].fillna(0.0)
    return out


def select_trades(scored: pd.DataFrame, recipe: Recipe) -> pd.DataFrame:
    out = scored.copy()
    out["take"] = False
    out["selector_score"] = out[recipe.score_col].fillna(0.0)

    if recipe.selector == "fixed_best_qwen":
        mask = out["strategy"] == "risk_qwen_risk_designer_risktest_d12_30d_lowdelta_pt30_sl20_c10"
        out.loc[mask, "take"] = True
        out.loc[mask, "selector_score"] = 1.0
        return out

    if recipe.selector == "all_positive":
        for _, idx in out.groupby("entry_date").groups.items():
            w = out.loc[list(idx)].copy().sort_values(recipe.score_col, ascending=False)
            if recipe.positive_only:
                w = w[w[recipe.score_col] > 0]
            chosen = w.head(recipe.top_n).index
            out.loc[chosen, "take"] = True
        return out

    for _, idx in out.groupby("entry_date").groups.items():
        w = out.loc[list(idx)].copy().sort_values(recipe.score_col, ascending=False)
        if recipe.positive_only:
            w = w[w[recipe.score_col] > 0]
        chosen = w.head(recipe.top_n).index
        out.loc[chosen, "take"] = True
    return out


def throttle_multiplier(throttle: str, dd_pct: float, market_row: pd.Series) -> float:
    """Scale new-trade risk from live portfolio drawdown and market shock."""
    if throttle == "none":
        dd_mult = 1.0
    elif throttle == "mild":
        dd_mult = 1.0 if dd_pct > -15 else 0.65 if dd_pct > -30 else 0.35 if dd_pct > -45 else 0.15
    elif throttle == "hard":
        dd_mult = 1.0 if dd_pct > -10 else 0.50 if dd_pct > -22 else 0.25 if dd_pct > -35 else 0.05
    elif throttle == "shock":
        dd_mult = 1.0 if dd_pct > -15 else 0.60 if dd_pct > -30 else 0.25
    else:
        dd_mult = 1.0

    shock_mult = 1.0
    if throttle == "shock":
        ret_5d = abs(float(market_row.get("ret_5d", 0.0) or 0.0))
        drawdown_20d = abs(float(market_row.get("drawdown_20d", 0.0) or 0.0))
        if ret_5d > 0.045 or drawdown_20d > 0.06:
            shock_mult = 0.45
        elif ret_5d > 0.030:
            shock_mult = 0.70
    return dd_mult * shock_mult


def score_size_multiplier(recipe: Recipe, row: pd.Series) -> float:
    if not recipe.score_sizing:
        return 1.0
    # Conviction sizing: keep enough aggression to matter but avoid letting one
    # extreme z-score dominate all account risk.
    score = float(row.get("selector_score", 0.0) or 0.0)
    return float(np.clip(0.75 + 0.20 * score, 0.35, 1.60))


def replay_dynamic(
    recipe: Recipe,
    market: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    capital: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dates = market.drop_duplicates("date").sort_values("date").copy()
    candidates = selected[selected["take"]].copy().sort_values(["entry_date", "selector_score"], ascending=[True, False])
    open_pos: list[dict[str, Any]] = []
    realized = 0.0
    nav_rows: list[dict[str, Any]] = []
    placed: list[dict[str, Any]] = []
    peak_nav = capital

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
        dd_pct = (nav_now / max(peak_nav, 1.0) - 1.0) * 100.0

        risk_cap = max(nav_now, 0.0) * recipe.concurrent_pct / 100.0
        open_risk = sum(float(pos["qty"]) * float(pos["max_loss_per_contract"]) for pos in open_pos)
        risk_mult = throttle_multiplier(recipe.throttle, dd_pct, mrow)

        todays = candidates[candidates["entry_date"] == dt].sort_values("selector_score", ascending=False)
        for _, tr in todays.iterrows():
            nav_for_size = max(nav_now, 0.0)
            per_loss = max(float(tr["max_loss_per_contract"]), 1.0)
            conviction_mult = score_size_multiplier(recipe, tr)
            trade_risk_pct = recipe.base_risk_pct * risk_mult * conviction_mult
            ideal_qty = int((nav_for_size * trade_risk_pct / 100.0) // per_loss)
            room_qty = int(max(risk_cap - open_risk, 0.0) // per_loss)
            qty = max(0, min(ideal_qty, room_qty))

            row = tr.to_dict()
            row["engine"] = recipe.name
            row["qty"] = qty
            row["risk_pct_used"] = trade_risk_pct
            row["throttle_mult"] = risk_mult
            row["sized_pnl_usd"] = qty * float(tr["pnl_usd"])
            row["sized_max_loss_usd"] = qty * per_loss
            row["skipped_reason"] = "" if qty > 0 else "risk_cap_or_nav"
            placed.append(row)
            if qty > 0:
                open_pos.append(row)
                open_risk += qty * per_loss

        unrealized = sum(float(pos["qty"]) * mark_trade(pd.Series(pos), mrow, dt) for pos in open_pos)
        nav_now = capital + realized + unrealized
        peak_nav = max(peak_nav, nav_now)
        nav_rows.append(
            {
                "date": dt,
                "engine": recipe.name,
                "nav": nav_now,
                "realized_pnl": realized,
                "unrealized_pnl": unrealized,
                "open_trades": len(open_pos),
                "open_contracts": int(sum(pos["qty"] for pos in open_pos)),
                "open_risk_usd": float(sum(float(pos["qty"]) * float(pos["max_loss_per_contract"]) for pos in open_pos)),
                "spy_close": float(mrow["spy_close"]),
                "drawdown_pct_live": (nav_now / max(peak_nav, 1.0) - 1.0) * 100.0,
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
    final_nav = float(nav_df["nav"].iloc[-1]) if not nav_df.empty else capital
    ret = (final_nav / capital - 1.0) * 100.0
    peak = nav_df["nav"].cummax()
    dd = nav_df["nav"] / peak - 1.0 if not nav_df.empty else pd.Series([0.0])
    summary = {
        "engine": recipe.name,
        "selector": recipe.selector,
        "score_col": recipe.score_col,
        "top_n": recipe.top_n,
        "positive_only": recipe.positive_only,
        "risk_pct": recipe.base_risk_pct,
        "concurrent_pct": recipe.concurrent_pct,
        "throttle": recipe.throttle,
        "score_sizing": recipe.score_sizing,
        "final_nav": final_nav,
        "return_pct": ret,
        "monthly_pct": safe_monthly(final_nav, capital, pd.Timestamp(dates["date"].iloc[0]), pd.Timestamp(dates["date"].iloc[-1])),
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


def build_recipe_grid() -> list[Recipe]:
    recipes: list[Recipe] = []
    specs = [
        ("combo", "top", "combined_conviction", [2, 3], [False, True], [True]),
        ("riskadj", "top", "risk_adjusted_score", [2, 3], [False], [True]),
        ("nn", "top", "nn_score", [3], [False], [False]),
        ("return", "top", "raw_return_score", [3], [False], [True]),
        ("positive_stack", "all_positive", "combined_score", [5], [True], [True]),
    ]
    for prefix, selector, score_col, top_ns, positive_flags, score_sizing_flags in specs:
        for top_n in top_ns:
            for positive_only in positive_flags:
                risk_grid = (20.0, 25.0, 30.0) if prefix != "positive_stack" else (25.0, 30.0)
                cap_grid = (150.0, 200.0, 250.0) if prefix in {"combo", "riskadj"} else (200.0, 250.0)
                throttle_grid = ("none", "mild", "shock") if prefix in {"combo", "riskadj"} else ("none", "mild")
                for risk_pct in risk_grid:
                    for concurrent_pct in cap_grid:
                        for throttle in throttle_grid:
                            for score_sizing in score_sizing_flags:
                                recipes.append(
                                    Recipe(
                                        name=f"{prefix}_{score_col}_top{top_n}_r{risk_pct:g}_c{concurrent_pct:g}_{throttle}_{'ss' if score_sizing else 'flat'}",
                                        selector=selector,
                                        score_col=score_col,
                                        top_n=top_n,
                                        positive_only=positive_only,
                                        base_risk_pct=risk_pct,
                                        concurrent_pct=concurrent_pct,
                                        throttle=throttle,
                                        score_sizing=score_sizing,
                                    )
                                )

    recipes.append(
        Recipe(
            name="champion_nn_top3",
            selector="top",
            score_col="nn_score",
            top_n=3,
            positive_only=False,
            base_risk_pct=30.0,
            concurrent_pct=250.0,
            throttle="none",
            score_sizing=False,
        )
    )
    recipes.append(
        Recipe(
            name="fixed_best_qwen",
            selector="fixed_best_qwen",
            score_col="combined_score",
            top_n=1,
            positive_only=False,
            base_risk_pct=10.0,
            concurrent_pct=100.0,
            throttle="none",
            score_sizing=False,
        )
    )
    return recipes


def objective_rows(train_df: pd.DataFrame) -> pd.DataFrame:
    out = train_df.copy()
    out["abs_dd"] = out["max_drawdown_pct"].abs()
    out["objective_return_max"] = out["return_pct"]
    out["objective_balanced"] = out["return_pct"] - 1.25 * out["abs_dd"]
    out["objective_survivable"] = np.where(out["max_drawdown_pct"] >= -45.0, out["return_pct"], out["return_pct"] - 500.0)
    return out


def choose_winners(train_summary: pd.DataFrame, all_recipes: dict[str, Recipe]) -> list[Recipe]:
    ranked = objective_rows(train_summary)
    winner_names = []
    for obj, name in [
        ("objective_return_max", "return_max"),
        ("objective_balanced", "balanced"),
        ("objective_survivable", "survivable"),
    ]:
        row = ranked.sort_values(obj, ascending=False).iloc[0]
        recipe = all_recipes[str(row["engine"])]
        winner_names.append((name, recipe))

    # Keep previous known champion and the fixed risk-adjusted strategy as anchors.
    for anchor in ("champion_nn_top3", "fixed_best_qwen"):
        if anchor in all_recipes:
            winner_names.append((anchor, all_recipes[anchor]))

    dedup: dict[str, Recipe] = {}
    for alias, recipe in winner_names:
        if recipe.name not in dedup:
            dedup[recipe.name] = Recipe(
                name=alias if alias in {"return_max", "balanced", "survivable"} else recipe.name,
                selector=recipe.selector,
                score_col=recipe.score_col,
                top_n=recipe.top_n,
                positive_only=recipe.positive_only,
                base_risk_pct=recipe.base_risk_pct,
                concurrent_pct=recipe.concurrent_pct,
                throttle=recipe.throttle,
                score_sizing=recipe.score_sizing,
            )
    return list(dedup.values())


def html_summary_table(df: pd.DataFrame, include_objectives: bool = False) -> str:
    cols = [
        "engine",
        "selector",
        "score_col",
        "top_n",
        "risk_pct",
        "concurrent_pct",
        "throttle",
        "score_sizing",
        "return_pct",
        "monthly_pct",
        "max_drawdown_pct",
        "trades",
        "win_rate_pct",
        "final_nav",
        "worst_trade_usd",
    ]
    if include_objectives:
        cols += ["objective_balanced", "objective_survivable"]
    w = df[cols].copy()
    for c in ("return_pct", "monthly_pct", "max_drawdown_pct", "win_rate_pct"):
        w[c] = w[c].map(lambda x: pct(float(x)))
    for c in ("final_nav", "worst_trade_usd"):
        w[c] = w[c].map(lambda x: usd(float(x)))
    return w.to_html(index=False, escape=False, classes="data-table")


def trade_table(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "<p>No trades.</p>"
    w = trades[trades["qty"] > 0].copy().sort_values(["engine", "entry_date"])
    out = pd.DataFrame(
        {
            "engine": w["engine"],
            "entry": pd.to_datetime(w["entry_date"]).dt.date.astype(str),
            "exit": pd.to_datetime(w["exit_date"]).dt.date.astype(str),
            "strategy": w["strategy"],
            "qty": w["qty"].astype(int),
            "score": w["selector_score"].map(lambda x: f"{x:+.3f}"),
            "reason": w["reason"],
            "legs": w["structure"],
            "entry_spy": w["entry_spy"].map(lambda x: f"{x:.2f}"),
            "exit_spy": w["exit_spy"].map(lambda x: f"{x:.2f}"),
            "credit": w["entry_credit"].map(lambda x: f"{x:.2f}"),
            "exit_debit": w["exit_debit"].map(lambda x: f"{x:.2f}"),
            "pnl": w["sized_pnl_usd"].map(lambda x: usd(float(x))),
            "risk": w["sized_max_loss_usd"].map(lambda x: usd(float(x))),
            "risk_pct": w["risk_pct_used"].map(lambda x: f"{float(x):.2f}%"),
        }
    )
    return out.to_html(index=False, escape=False, classes="data-table")


def strategy_component_table(train_trades: pd.DataFrame, test_trades: pd.DataFrame) -> str:
    rows = []
    for name in TOP_STRATEGIES:
        for label, data in (("train", train_trades), ("test", test_trades)):
            w = data[data["strategy"] == name]
            if w.empty:
                continue
            nav = 10_000.0 + w["pnl_usd"].cumsum()
            peak = nav.cummax()
            rows.append(
                {
                    "strategy": name,
                    "split": label,
                    "trades": len(w),
                    "raw_pnl": float(w["pnl_usd"].sum()),
                    "raw_return_on_10k": float(w["pnl_usd"].sum() / 10_000.0 * 100.0),
                    "win_rate": float((w["pnl_usd"] > 0).mean() * 100.0),
                    "rough_dd_on_10k": float((nav / peak - 1.0).min() * 100.0),
                    "note": STRATEGY_NOTES.get(name, ""),
                }
            )
    out = pd.DataFrame(rows)
    out["raw_pnl"] = out["raw_pnl"].map(usd)
    out["raw_return_on_10k"] = out["raw_return_on_10k"].map(pct)
    out["win_rate"] = out["win_rate"].map(lambda x: f"{x:.1f}%")
    out["rough_dd_on_10k"] = out["rough_dd_on_10k"].map(pct)
    return out.to_html(index=False, escape=False, classes="data-table")


def build_report(
    out_dir: Path,
    *,
    train_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    navs: pd.DataFrame,
    trades: pd.DataFrame,
    test_market: pd.DataFrame,
    train_trades: pd.DataFrame,
    test_trades: pd.DataFrame,
    winners: list[Recipe],
    capital: float,
    train_meta: dict[str, Any],
) -> Path:
    bench = benchmark(test_market, capital)
    bret = (float(bench["nav"].iloc[-1]) / capital - 1.0) * 100.0
    bpeak = bench["nav"].cummax()
    bench_row = {
        "engine": "SPY benchmark",
        "selector": "buy_hold",
        "score_col": "",
        "top_n": 1,
        "positive_only": "",
        "risk_pct": "",
        "concurrent_pct": "",
        "throttle": "",
        "score_sizing": "",
        "final_nav": float(bench["nav"].iloc[-1]),
        "return_pct": bret,
        "monthly_pct": safe_monthly(float(bench["nav"].iloc[-1]), capital, pd.Timestamp(bench["date"].iloc[0]), pd.Timestamp(bench["date"].iloc[-1])),
        "max_drawdown_pct": float((bench["nav"] / bpeak - 1.0).min() * 100.0),
        "trades": 1,
        "win_rate_pct": np.nan,
        "worst_trade_usd": np.nan,
    }
    test_plus_bench = pd.concat([test_summary.copy(), pd.DataFrame([bench_row])], ignore_index=True)

    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.55, 0.22, 0.23],
        vertical_spacing=0.08,
        subplot_titles=("Held-out test equity curves", "Drawdown", "Return vs drawdown"),
    )
    for engine, w in navs.groupby("engine"):
        color = COLORS.get(engine, None)
        fig.add_trace(go.Scatter(x=w["date"], y=w["nav"], mode="lines", name=engine, line=dict(width=2.2, color=color)), row=1, col=1)
        peak = w["nav"].cummax()
        dd = w["nav"] / peak - 1.0
        fig.add_trace(go.Scatter(x=w["date"], y=dd * 100.0, mode="lines", name=f"{engine} DD", showlegend=False, line=dict(width=1.4, color=color)), row=2, col=1)
    fig.add_trace(go.Scatter(x=bench["date"], y=bench["nav"], mode="lines", name="SPY benchmark", line=dict(width=2.0, dash="dot", color=COLORS["SPY benchmark"])), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=test_summary["max_drawdown_pct"],
            y=test_summary["return_pct"],
            text=test_summary["engine"],
            mode="markers+text",
            textposition="top center",
            marker=dict(size=14, color=[COLORS.get(e, "#334155") for e in test_summary["engine"]]),
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
      code, pre { background:#f7f9f9; padding:2px 4px; border:1px solid #e5e8e8; }
      pre { padding:12px; overflow-x:auto; }
    </style>
    """

    report = out_dir / "optimized_combined_engine_report.html"
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Optimized Combined Engine</title>",
        css,
        "</head><body>",
        f"<h1>Optimized Combined Engine (${capital:,.0f})</h1>",
        "<p class='note'>This combines the six high-performing aggressive strategies into one train-optimized selector. "
        "The optimizer searches selector/risk/throttle recipes on the 70% train split, then freezes the winners and evaluates on the 30% held-out test. "
        "No council, no Kimi, no external API calls.</p>",
        "<h2>Held-Out Test Chart</h2>",
        fig.to_html(include_plotlyjs=True, full_html=False),
        "<h2>Held-Out Test Summary + Benchmark</h2><div class='scroll'>",
        html_summary_table(test_plus_bench),
        "</div>",
        "<h2>Chosen Frozen Recipes</h2>",
        f"<pre>{json.dumps([r.to_dict() for r in winners], indent=2)}</pre>",
        "<h2>Top Train Search Rows</h2><div class='scroll'>",
        html_summary_table(objective_rows(train_summary).sort_values('return_pct', ascending=False).head(25), include_objectives=True),
        "</div>",
        "<h2>Six Strategy Components</h2><div class='scroll'>",
        strategy_component_table(train_trades, test_trades),
        "</div>",
        "<h2>Held-Out Selected Entries / Exits / Stops / P&L</h2><div class='scroll'>",
        trade_table(trades),
        "</div>",
        "<h2>Train Metadata</h2>",
        f"<pre>{json.dumps(train_meta, indent=2)}</pre>",
        "</body></html>",
    ]
    report.write_text("\n".join(html), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--max-train-recipes", type=int, default=0, help="Debug cap; 0 means all recipes")
    args = parser.parse_args()

    stamp = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"optimized_combined_engine_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = load_specs()
    train_market, train_trades = load_period(args.root, "train", specs)
    test_market, test_trades = load_period(args.root, "test", specs)

    train_scored, train_meta = train_scores(train_trades, train_trades)
    test_scored, _ = train_scores(train_trades, test_trades)
    train_scored = add_combined_scores(train_scored, train_trades)
    test_scored = add_combined_scores(test_scored, train_trades)

    recipes = build_recipe_grid()
    if args.max_train_recipes > 0:
        recipes = recipes[: args.max_train_recipes]
    recipe_map = {r.name: r for r in recipes}

    train_rows = []
    for i, recipe in enumerate(recipes, start=1):
        selected = select_trades(train_scored, recipe)
        _, _, summary = replay_dynamic(recipe, train_market, selected, capital=args.capital)
        train_rows.append(summary)
        if i % 50 == 0:
            print(f"searched {i}/{len(recipes)} recipes", flush=True)
    train_summary = pd.DataFrame(train_rows)
    train_summary.to_csv(out_dir / "train_recipe_search.csv", index=False)

    winners = choose_winners(train_summary, recipe_map)
    navs, trades, test_rows = [], [], []
    for recipe in winners:
        selected = select_trades(test_scored, recipe)
        nav, tr, summary = replay_dynamic(recipe, test_market, selected, capital=args.capital)
        navs.append(nav)
        trades.append(tr)
        test_rows.append(summary)

    nav_df = pd.concat(navs, ignore_index=True)
    trade_df = pd.concat(trades, ignore_index=True)
    test_summary = pd.DataFrame(test_rows).sort_values("return_pct", ascending=False)

    nav_df.to_csv(out_dir / "optimized_test_nav.csv", index=False)
    trade_df.to_csv(out_dir / "optimized_test_trades.csv", index=False)
    test_summary.to_csv(out_dir / "optimized_test_summary.csv", index=False)
    train_scored.to_csv(out_dir / "train_candidates_scored.csv", index=False)
    test_scored.to_csv(out_dir / "test_candidates_scored.csv", index=False)
    (out_dir / "winner_recipes.json").write_text(json.dumps([r.to_dict() for r in winners], indent=2), encoding="utf-8")

    report = build_report(
        out_dir,
        train_summary=train_summary,
        test_summary=test_summary,
        navs=nav_df,
        trades=trade_df,
        test_market=test_market,
        train_trades=train_trades,
        test_trades=test_trades,
        winners=winners,
        capital=args.capital,
        train_meta=train_meta,
    )

    print("\n=== HELD-OUT TEST WINNERS ===")
    print(test_summary.to_string(index=False))
    print(f"\nReport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
