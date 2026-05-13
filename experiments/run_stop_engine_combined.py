"""Early-stop engine for the aggressive combined strategy set.

Problem this tests:
    The old stop loss waits until the condor debit reaches the full stop.
    That creates the big account dips. This script adds an early-exit layer:

    1. baseline: old behavior, no early stop.
    2. mid_credit_stop: mechanical early stop at a smaller credit-multiple loss.
    3. nn_early_stop: sklearn MLP classifier trained on train-window trade paths.

It trains the NN only on the 70% train split, then evaluates the frozen stop
logic on the 30% held-out test split.
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


ROUNDTRIP_COMMISSION = 0.65 * 8.0

EXIT_FEATURES = [
    "days_held",
    "dte_now",
    "current_pnl",
    "current_pnl_pct_max_loss",
    "current_loss_credit_mult",
    "debit_to_credit",
    "spot_move_pct",
    "abs_spot_move_pct",
    "iv_change",
    "put_pressure_pct",
    "call_pressure_pct",
    "entry_credit",
    "max_loss_per_contract",
    "credit_to_width",
    "selector_score",
    "nn_score",
    "combined_score",
    "risk_adjusted_score",
    "iv",
    "vrp20",
    "iv_rank_252",
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "drawdown_20d",
    "rsi_14",
]


@dataclass(frozen=True)
class ExitConfig:
    name: str
    stop_mode: str
    nn_threshold: float = 0.65
    mid_credit_mult: float = 0.90
    hard_credit_mult: float = 1.25
    portfolio_dd_stop_pct: float = 0.0
    pause_days: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "stop_mode": self.stop_mode,
            "nn_threshold": self.nn_threshold,
            "mid_credit_mult": self.mid_credit_mult,
            "hard_credit_mult": self.hard_credit_mult,
            "portfolio_dd_stop_pct": self.portfolio_dd_stop_pct,
            "pause_days": self.pause_days,
        }


def recipes() -> dict[str, Recipe]:
    return {
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
    }


def add_entry_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["entry_id"] = (
        out["strategy"].astype(str)
        + "|"
        + pd.to_datetime(out["entry_date"]).dt.strftime("%Y-%m-%d")
        + "|"
        + out["short_put"].map(lambda x: f"{float(x):.0f}")
        + "|"
        + out["short_call"].map(lambda x: f"{float(x):.0f}")
    )
    return out


def row_features(trade: pd.Series | dict[str, Any], mrow: pd.Series, dt: pd.Timestamp) -> dict[str, float]:
    tr = pd.Series(trade)
    current_pnl = mark_trade(tr, mrow, dt)
    entry_credit = max(float(tr["entry_credit"]), 0.01)
    max_loss = max(float(tr["max_loss_per_contract"]), 1.0)
    debit = float(tr["entry_credit"]) - ((current_pnl + ROUNDTRIP_COMMISSION) / 100.0)
    spot = float(mrow["spy_close"])
    entry_spy = max(float(tr["entry_spy"]), 0.01)
    return {
        "days_held": float((dt - pd.Timestamp(tr["entry_date"])).days),
        "dte_now": float(max(0, (pd.Timestamp(tr["expiry"]) - dt).days)),
        "current_pnl": float(current_pnl),
        "current_pnl_pct_max_loss": float(current_pnl / max_loss),
        "current_loss_credit_mult": float(max(0.0, -current_pnl) / (entry_credit * 100.0)),
        "debit_to_credit": float(debit / entry_credit),
        "spot_move_pct": float(spot / entry_spy - 1.0),
        "abs_spot_move_pct": float(abs(spot / entry_spy - 1.0)),
        "iv_change": float(float(mrow["iv"]) - float(tr["entry_iv"])),
        "put_pressure_pct": float(max(0.0, float(tr["short_put"]) - spot) / entry_spy),
        "call_pressure_pct": float(max(0.0, spot - float(tr["short_call"])) / entry_spy),
        "entry_credit": float(entry_credit),
        "max_loss_per_contract": float(max_loss),
        "credit_to_width": float(tr.get("credit_to_width", 0.0) or 0.0),
        "selector_score": float(tr.get("selector_score", 0.0) or 0.0),
        "nn_score": float(tr.get("nn_score", 0.0) or 0.0),
        "combined_score": float(tr.get("combined_score", 0.0) or 0.0),
        "risk_adjusted_score": float(tr.get("risk_adjusted_score", 0.0) or 0.0),
        "iv": float(mrow.get("iv", 0.0) or 0.0),
        "vrp20": float(mrow.get("vrp20", 0.0) or 0.0),
        "iv_rank_252": float(mrow.get("iv_rank_252", 0.5) or 0.5),
        "ret_1d": float(mrow.get("ret_1d", 0.0) or 0.0),
        "ret_5d": float(mrow.get("ret_5d", 0.0) or 0.0),
        "ret_20d": float(mrow.get("ret_20d", 0.0) or 0.0),
        "drawdown_20d": float(mrow.get("drawdown_20d", 0.0) or 0.0),
        "rsi_14": float(mrow.get("rsi_14", 50.0) or 50.0),
    }


def build_exit_dataset(market: pd.DataFrame, selected_frames: list[pd.DataFrame]) -> pd.DataFrame:
    selected = pd.concat(selected_frames, ignore_index=True)
    selected = add_entry_id(selected[selected["take"]].copy())
    selected = selected.drop_duplicates("entry_id")
    dates = pd.to_datetime(market["date"]).drop_duplicates().sort_values()
    market_by_date = {pd.Timestamp(r["date"]): r for _, r in market.iterrows()}
    rows = []
    for _, tr in selected.iterrows():
        entry = pd.Timestamp(tr["entry_date"])
        exit_dt = pd.Timestamp(tr["exit_date"])
        path_dates = [d for d in dates if entry < d < exit_dt]
        if not path_dates:
            continue
        final_pnl = float(tr["pnl_usd"])
        max_loss = max(float(tr["max_loss_per_contract"]), 1.0)
        for dt in path_dates:
            mrow = market_by_date.get(dt)
            if mrow is None:
                continue
            feats = row_features(tr, mrow, dt)
            current_pnl = feats["current_pnl"]
            # Label a cut if the trade is already losing and waiting to the
            # original exit is materially worse than closing now.
            materially_worse = final_pnl < current_pnl - max(15.0, 0.04 * max_loss)
            final_bad = final_pnl < -max(15.0, 0.03 * max_loss)
            old_stop = str(tr.get("reason", "")) == "stop_loss"
            label = int(current_pnl < 0 and (materially_worse or final_bad or old_stop))
            feats.update(
                {
                    "entry_id": tr["entry_id"],
                    "strategy": tr["strategy"],
                    "date": dt,
                    "final_pnl": final_pnl,
                    "label_cut": label,
                }
            )
            rows.append(feats)
    return pd.DataFrame(rows)


def train_exit_model(samples: pd.DataFrame) -> tuple[Any | None, dict[str, Any]]:
    if samples.empty or samples["label_cut"].nunique() < 2:
        return None, {"status": "no_model", "reason": "insufficient label variety", "samples": int(len(samples))}
    X = samples[EXIT_FEATURES]
    y = samples["label_cut"].astype(int)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(48, 24),
            alpha=0.002,
            max_iter=500,
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
    meta = {
        "status": "trained",
        "samples": int(len(samples)),
        "positives": int(y.sum()),
        "positive_rate_pct": float(y.mean() * 100.0),
        "train_auc": auc,
    }
    return model, meta


def should_early_close(
    pos: dict[str, Any],
    mrow: pd.Series,
    dt: pd.Timestamp,
    config: ExitConfig,
    model: Any | None,
) -> tuple[bool, str, float, float]:
    current_pnl = mark_trade(pd.Series(pos), mrow, dt)
    credit_stop_loss = max(float(pos["entry_credit"]) * 100.0, 1.0)
    loss_mult = max(0.0, -current_pnl) / credit_stop_loss

    if config.stop_mode in ("baseline", "portfolio_circuit"):
        return False, "", current_pnl, 0.0

    if config.stop_mode == "mid_credit_stop":
        if loss_mult >= config.mid_credit_mult:
            return True, "mid_credit_stop", current_pnl, 0.0
        return False, "", current_pnl, 0.0

    if config.stop_mode == "nn_early_stop":
        if loss_mult >= config.hard_credit_mult:
            return True, "hard_credit_floor", current_pnl, 1.0
        if model is None or current_pnl >= 0:
            return False, "", current_pnl, 0.0
        feats = pd.DataFrame([row_features(pos, mrow, dt)])[EXIT_FEATURES]
        prob = float(model.predict_proba(feats)[0, 1])
        if prob >= config.nn_threshold and loss_mult >= 0.25:
            return True, "nn_early_stop", current_pnl, prob
        return False, "", current_pnl, prob

    raise ValueError(f"unknown stop mode {config.stop_mode}")


def replay_with_exit_engine(
    engine_name: str,
    recipe: Recipe,
    config: ExitConfig,
    market: pd.DataFrame,
    selected: pd.DataFrame,
    *,
    capital: float,
    model: Any | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dates = market.drop_duplicates("date").sort_values("date").copy()
    candidates = add_entry_id(selected[selected["take"]].copy()).sort_values(["entry_date", "selector_score"], ascending=[True, False])
    open_pos: list[dict[str, Any]] = []
    realized = 0.0
    nav_rows = []
    placed = []
    peak_nav = capital
    pause_until = pd.Timestamp.min
    circuit_count = 0

    for _, mrow in dates.iterrows():
        dt = pd.Timestamp(mrow["date"])
        still = []
        for pos in open_pos:
            close_now, close_reason, current_pnl, nn_prob = should_early_close(pos, mrow, dt, config, model)
            if pd.Timestamp(pos["exit_date"]) <= dt:
                realized += float(pos["qty"]) * float(pos["pnl_usd"])
                rec = dict(pos)
                rec.update(
                    {
                        "actual_exit_date": dt,
                        "actual_reason": str(pos.get("reason", "original_exit")),
                        "actual_pnl_per_contract": float(pos["pnl_usd"]),
                        "actual_sized_pnl_usd": float(pos["qty"]) * float(pos["pnl_usd"]),
                        "nn_stop_prob": nn_prob,
                        "early_closed": False,
                    }
                )
                placed.append(rec)
            elif close_now:
                realized += float(pos["qty"]) * current_pnl
                rec = dict(pos)
                rec.update(
                    {
                        "actual_exit_date": dt,
                        "actual_reason": close_reason,
                        "actual_pnl_per_contract": current_pnl,
                        "actual_sized_pnl_usd": float(pos["qty"]) * current_pnl,
                        "nn_stop_prob": nn_prob,
                        "early_closed": True,
                    }
                )
                placed.append(rec)
            else:
                still.append(pos)
        open_pos = still

        unrealized = sum(float(pos["qty"]) * mark_trade(pd.Series(pos), mrow, dt) for pos in open_pos)
        nav_now = capital + realized + unrealized
        peak_nav = max(peak_nav, nav_now)
        live_dd_pct = (nav_now / max(peak_nav, 1.0) - 1.0) * 100.0

        if (
            config.portfolio_dd_stop_pct > 0
            and open_pos
            and live_dd_pct <= -abs(config.portfolio_dd_stop_pct)
        ):
            circuit_count += 1
            for pos in open_pos:
                current_pnl = mark_trade(pd.Series(pos), mrow, dt)
                realized += float(pos["qty"]) * current_pnl
                rec = dict(pos)
                rec.update(
                    {
                        "actual_exit_date": dt,
                        "actual_reason": "portfolio_circuit_breaker",
                        "actual_pnl_per_contract": current_pnl,
                        "actual_sized_pnl_usd": float(pos["qty"]) * current_pnl,
                        "nn_stop_prob": 0.0,
                        "early_closed": True,
                    }
                )
                placed.append(rec)
            open_pos = []
            pause_until = max(pause_until, dt + pd.Timedelta(days=config.pause_days))
            unrealized = 0.0
            nav_now = capital + realized

        risk_cap = max(nav_now, 0.0) * recipe.concurrent_pct / 100.0
        open_risk = sum(float(pos["qty"]) * float(pos["max_loss_per_contract"]) for pos in open_pos)

        todays = candidates[candidates["entry_date"] == dt].sort_values("selector_score", ascending=False)
        if dt < pause_until:
            todays = todays.iloc[0:0]
        for _, tr in todays.iterrows():
            nav_for_size = max(nav_now, 0.0)
            per_loss = max(float(tr["max_loss_per_contract"]), 1.0)
            ideal_qty = int((nav_for_size * recipe.base_risk_pct / 100.0) // per_loss)
            room_qty = int(max(risk_cap - open_risk, 0.0) // per_loss)
            qty = max(0, min(ideal_qty, room_qty))
            if qty <= 0:
                continue
            row = tr.to_dict()
            row["engine"] = engine_name
            row["qty"] = qty
            row["risk_pct_used"] = recipe.base_risk_pct
            row["sized_max_loss_usd"] = qty * per_loss
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
                "paused": dt < pause_until,
                "circuit_count": circuit_count,
            }
        )

    final_row = dates.iloc[-1]
    final_dt = pd.Timestamp(final_row["date"])
    for pos in open_pos:
        current_pnl = mark_trade(pd.Series(pos), final_row, final_dt)
        realized += float(pos["qty"]) * current_pnl
        rec = dict(pos)
        rec.update(
            {
                "actual_exit_date": final_dt,
                "actual_reason": "end_forced_mark",
                "actual_pnl_per_contract": current_pnl,
                "actual_sized_pnl_usd": float(pos["qty"]) * current_pnl,
                "nn_stop_prob": 0.0,
                "early_closed": False,
            }
        )
        placed.append(rec)
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
        "stop_mode": config.stop_mode,
        "nn_threshold": config.nn_threshold,
        "mid_credit_mult": config.mid_credit_mult,
        "hard_credit_mult": config.hard_credit_mult,
        "final_nav": final_nav,
        "return_pct": ret,
        "monthly_pct": safe_monthly(final_nav, capital, pd.Timestamp(dates["date"].iloc[0]), pd.Timestamp(dates["date"].iloc[-1])),
        "max_drawdown_pct": float(dd.min() * 100.0),
        "trades": int(len(trades_df)),
        "early_exits": int(trades_df["early_closed"].sum()) if not trades_df.empty else 0,
        "circuit_breakers": circuit_count,
        "avg_qty": float(trades_df["qty"].mean()) if not trades_df.empty else 0.0,
        "max_qty": int(trades_df["qty"].max()) if not trades_df.empty else 0,
        "max_open_contracts": int(nav_df["open_contracts"].max()) if not nav_df.empty else 0,
        "max_open_risk_usd": float(nav_df["open_risk_usd"].max()) if not nav_df.empty else 0.0,
        "worst_trade_usd": float(trades_df["actual_sized_pnl_usd"].min()) if not trades_df.empty else 0.0,
        "win_rate_pct": float((trades_df["actual_sized_pnl_usd"] > 0).mean() * 100.0) if not trades_df.empty else 0.0,
    }
    return nav_df, trades_df, summary


def optimize_exit_configs_on_train(
    recipe: Recipe,
    train_market: pd.DataFrame,
    selected_train: pd.DataFrame,
    capital: float,
    model: Any | None,
) -> list[ExitConfig]:
    configs = [ExitConfig(name="baseline", stop_mode="baseline")]
    for mult in (0.50, 0.70, 0.90, 1.10):
        configs.append(ExitConfig(name=f"mid_{mult:.2f}", stop_mode="mid_credit_stop", mid_credit_mult=mult))
    for threshold in (0.55, 0.65, 0.75, 0.85):
        for hard in (0.90, 1.15, 1.40):
            configs.append(ExitConfig(name=f"nn_t{threshold:.2f}_h{hard:.2f}", stop_mode="nn_early_stop", nn_threshold=threshold, hard_credit_mult=hard))
    for dd_stop in (15.0, 20.0, 25.0, 30.0, 35.0):
        for pause in (5, 10, 20):
            configs.append(
                ExitConfig(
                    name=f"circuit_dd{dd_stop:.0f}_p{pause}",
                    stop_mode="portfolio_circuit",
                    portfolio_dd_stop_pct=dd_stop,
                    pause_days=pause,
                )
            )

    rows = []
    config_by_name = {}
    for cfg in configs:
        nav, trades, sm = replay_with_exit_engine(
            f"{recipe.name}_{cfg.name}",
            recipe,
            cfg,
            train_market,
            selected_train,
            capital=capital,
            model=model,
        )
        sm["objective"] = sm["return_pct"] - 1.75 * abs(sm["max_drawdown_pct"])
        rows.append(sm)
        config_by_name[cfg.name] = cfg
    ranked = pd.DataFrame(rows).sort_values("objective", ascending=False)
    keep_names = ["baseline"]
    keep_names.append(str(ranked[ranked["stop_mode"] == "mid_credit_stop"].iloc[0]["engine"]).replace(f"{recipe.name}_", ""))
    if model is not None and (ranked["stop_mode"] == "nn_early_stop").any():
        keep_names.append(str(ranked[ranked["stop_mode"] == "nn_early_stop"].iloc[0]["engine"]).replace(f"{recipe.name}_", ""))
    if (ranked["stop_mode"] == "portfolio_circuit").any():
        circuit_ranked = ranked[ranked["stop_mode"] == "portfolio_circuit"].copy()
        circuit_ranked["circuit_objective"] = circuit_ranked["return_pct"] - 2.75 * circuit_ranked["max_drawdown_pct"].abs()
        keep_names.append(str(circuit_ranked.sort_values("circuit_objective", ascending=False).iloc[0]["engine"]).replace(f"{recipe.name}_", ""))
    return [config_by_name[n] for n in dict.fromkeys(keep_names)]


def html_summary_table(df: pd.DataFrame) -> str:
    w = df.copy()
    for c in ("return_pct", "monthly_pct", "max_drawdown_pct", "win_rate_pct"):
        w[c] = w[c].map(lambda x: pct(float(x)))
    for c in ("final_nav", "max_open_risk_usd", "worst_trade_usd"):
        w[c] = w[c].map(lambda x: usd(float(x)))
    cols = [
        "engine",
        "recipe",
        "stop_mode",
        "return_pct",
        "monthly_pct",
        "max_drawdown_pct",
        "trades",
        "early_exits",
        "circuit_breakers",
        "win_rate_pct",
        "final_nav",
        "worst_trade_usd",
        "max_open_contracts",
        "max_open_risk_usd",
    ]
    return w[cols].to_html(index=False, escape=False, classes="data-table")


def html_trade_table(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "<p>No trades.</p>"
    w = trades.copy().sort_values(["engine", "actual_exit_date"])
    out = pd.DataFrame(
        {
            "engine": w["engine"],
            "entry": pd.to_datetime(w["entry_date"]).dt.date.astype(str),
            "exit": pd.to_datetime(w["actual_exit_date"]).dt.date.astype(str),
            "reason": w["actual_reason"],
            "original_reason": w["reason"],
            "early": w["early_closed"].map(lambda x: "yes" if bool(x) else ""),
            "strategy": w["strategy"],
            "qty": w["qty"].astype(int),
            "legs": w["structure"],
            "credit": w["entry_credit"].map(lambda x: f"{float(x):.2f}"),
            "pnl": w["actual_sized_pnl_usd"].map(lambda x: usd(float(x))),
            "risk": w["sized_max_loss_usd"].map(lambda x: usd(float(x))),
            "nn_prob": w["nn_stop_prob"].map(lambda x: f"{float(x):.2f}"),
        }
    )
    return out.to_html(index=False, escape=False, classes="data-table")


def build_report(
    out_dir: Path,
    navs: pd.DataFrame,
    trades: pd.DataFrame,
    summary: pd.DataFrame,
    test_market: pd.DataFrame,
    capital: float,
    model_meta: dict[str, Any],
    chosen_configs: dict[str, list[dict[str, Any]]],
) -> Path:
    bench = benchmark(test_market, capital)
    bret = (float(bench["nav"].iloc[-1]) / capital - 1.0) * 100.0
    bpeak = bench["nav"].cummax()
    bench_row = {
        "engine": "SPY benchmark",
        "recipe": "buy_hold",
        "stop_mode": "",
        "final_nav": float(bench["nav"].iloc[-1]),
        "return_pct": bret,
        "monthly_pct": safe_monthly(float(bench["nav"].iloc[-1]), capital, pd.Timestamp(bench["date"].iloc[0]), pd.Timestamp(bench["date"].iloc[-1])),
        "max_drawdown_pct": float((bench["nav"] / bpeak - 1.0).min() * 100.0),
        "trades": 1,
        "early_exits": 0,
        "circuit_breakers": 0,
        "win_rate_pct": np.nan,
        "worst_trade_usd": np.nan,
        "max_open_contracts": 1,
        "max_open_risk_usd": capital,
    }
    summary_plus = pd.concat([summary.copy(), pd.DataFrame([bench_row])], ignore_index=True)

    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.55, 0.22, 0.23],
        subplot_titles=("Held-out test equity curves with early-stop engine", "Drawdown", "Return vs drawdown"),
        vertical_spacing=0.08,
    )
    palette = ["#111827", "#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2", "#be123c"]
    color_map = {name: palette[i % len(palette)] for i, name in enumerate(summary["engine"].tolist())}
    for engine, w in navs.groupby("engine"):
        color = color_map.get(engine, "#334155")
        fig.add_trace(go.Scatter(x=w["date"], y=w["nav"], mode="lines", name=engine, line=dict(width=2.2, color=color)), row=1, col=1)
        peak = w["nav"].cummax()
        dd = w["nav"] / peak - 1.0
        fig.add_trace(go.Scatter(x=w["date"], y=dd * 100.0, mode="lines", name=f"{engine} DD", showlegend=False, line=dict(width=1.2, color=color)), row=2, col=1)
    fig.add_trace(go.Scatter(x=bench["date"], y=bench["nav"], mode="lines", name="SPY benchmark", line=dict(width=2.0, dash="dot", color="#64748b")), row=1, col=1)
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
      table.data-table td:nth-child(4), table.data-table th:nth-child(4),
      table.data-table td:nth-child(7), table.data-table th:nth-child(7) { text-align: left; }
      pre { background:#f7f9f9; padding:12px; border:1px solid #e5e8e8; overflow-x:auto; }
    </style>
    """
    report = out_dir / "stop_engine_combined_report.html"
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Stop Engine Combined</title>",
        css,
        "</head><body>",
        f"<h1>Stop Engine Combined (${capital:,.0f})</h1>",
        "<p class='note'>This is the final stop-loss test: baseline vs mechanical midway stop vs NN early-stop model. "
        "The NN is trained only on train-window position paths, then frozen for held-out test.</p>",
        "<h2>Held-Out Chart</h2>",
        fig.to_html(include_plotlyjs=True, full_html=False),
        "<h2>Held-Out Summary + Benchmark</h2><div class='scroll'>",
        html_summary_table(summary_plus),
        "</div>",
        "<h2>NN Stop Model Metadata</h2>",
        f"<pre>{json.dumps(model_meta, indent=2)}</pre>",
        "<h2>Train-Optimized Stop Configs</h2>",
        f"<pre>{json.dumps(chosen_configs, indent=2)}</pre>",
        "<h2>Entries / Exits / Early Stops / P&L</h2><div class='scroll'>",
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

    out_dir = OUT_ROOT / f"stop_engine_combined_{pd.Timestamp.now('UTC').strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = load_specs()
    train_market, train_trades = load_period(args.root, "train", specs)
    test_market, test_trades = load_period(args.root, "test", specs)
    train_scored, _ = train_scores(train_trades, train_trades)
    test_scored, _ = train_scores(train_trades, test_trades)
    train_scored = add_combined_scores(train_scored, train_trades)
    test_scored = add_combined_scores(test_scored, train_trades)

    selected_train_by_recipe = {}
    selected_test_by_recipe = {}
    for name, recipe in recipes().items():
        selected_train_by_recipe[name] = select_trades(train_scored, recipe)
        selected_test_by_recipe[name] = select_trades(test_scored, recipe)

    exit_samples = build_exit_dataset(train_market, list(selected_train_by_recipe.values()))
    exit_samples.to_csv(out_dir / "exit_training_samples.csv", index=False)
    model, model_meta = train_exit_model(exit_samples)

    navs, trades, rows = [], [], []
    chosen_configs: dict[str, list[dict[str, Any]]] = {}
    for name, recipe in recipes().items():
        configs = optimize_exit_configs_on_train(
            recipe,
            train_market,
            selected_train_by_recipe[name],
            args.capital,
            model,
        )
        chosen_configs[name] = [c.to_dict() for c in configs]
        for cfg in configs:
            engine_name = f"{name}_{cfg.name}"
            nav, tr, sm = replay_with_exit_engine(
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
    summary = pd.DataFrame(rows).sort_values("return_pct", ascending=False)

    nav_df.to_csv(out_dir / "stop_engine_nav.csv", index=False)
    trade_df.to_csv(out_dir / "stop_engine_trades.csv", index=False)
    summary.to_csv(out_dir / "stop_engine_summary.csv", index=False)
    (out_dir / "stop_model_meta.json").write_text(json.dumps(model_meta, indent=2), encoding="utf-8")
    (out_dir / "chosen_stop_configs.json").write_text(json.dumps(chosen_configs, indent=2), encoding="utf-8")
    report = build_report(
        out_dir,
        nav_df,
        trade_df,
        summary,
        test_market,
        args.capital,
        model_meta,
        chosen_configs,
    )

    print("\n=== HELD-OUT STOP ENGINE RESULTS ===")
    print(summary.to_string(index=False))
    print(f"\nReport: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
