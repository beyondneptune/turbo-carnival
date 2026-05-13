"""Chronological 70/30 model test on the final combined dataset.

This is a leak-aware first pass for the "primary models":
  - LightGBM return regressor for 5-day forward return.
  - LightGBM direction classifier for fwd_ret_5d > 0.
  - LightGBM range/short-vol classifier for realized 5-day move inside VIX
    expected move.
  - Ridge/logistic baselines for sanity.

Outputs metrics, predictions, feature importance, and a simple SPY test-period
equity chart. No live broker calls.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = REPO_ROOT / "data" / "final_dataset" / "model_daily_panel.parquet"
SPY_PATH = REPO_ROOT / "data" / "final_dataset" / "spy_training_panel.parquet"

TARGET_HORIZON = 5
RANDOM_STATE = 42


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return str(obj)


def load_panel(scope: str) -> pd.DataFrame:
    path = SPY_PATH if scope == "spy" else DATASET_PATH
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    return df.sort_values(["symbol", "date"]).reset_index(drop=True)


def add_targets_and_safe_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build targets and trailing features without relying on suspect columns."""
    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")
    out["_spot_close"] = close

    g = out.groupby("symbol", group_keys=False)
    out["_ret_1d_trailing"] = g["_spot_close"].pct_change(1)
    out["_ret_5d_trailing"] = g["_spot_close"].pct_change(5)
    out["_ret_10d_trailing"] = g["_spot_close"].pct_change(10)
    out["_ret_20d_trailing"] = g["_spot_close"].pct_change(20)
    out["_rv_10d_trailing"] = (
        g["_ret_1d_trailing"].rolling(10, min_periods=5).std().reset_index(level=0, drop=True)
        * math.sqrt(252)
    )
    out["_rv_20d_trailing"] = (
        g["_ret_1d_trailing"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True)
        * math.sqrt(252)
    )
    out["_rv_60d_trailing"] = (
        g["_ret_1d_trailing"].rolling(60, min_periods=30).std().reset_index(level=0, drop=True)
        * math.sqrt(252)
    )
    for w in (10, 20, 50):
        sma = g["_spot_close"].rolling(w, min_periods=max(5, w // 2)).mean().reset_index(level=0, drop=True)
        out[f"_dist_sma_{w}"] = out["_spot_close"] / sma - 1.0

    out["_target_fwd_ret_5d"] = g["_spot_close"].shift(-TARGET_HORIZON) / out["_spot_close"] - 1.0
    out["_target_direction_up"] = (out["_target_fwd_ret_5d"] > 0).astype(int)

    vix = None
    for col in ("ctx_idx_vix_close", "fred_vixcls", "hf_VIX"):
        if col in out.columns:
            s = pd.to_numeric(out[col], errors="coerce")
            vix = s if vix is None else vix.fillna(s)
    if vix is None:
        vix = pd.Series(np.nan, index=out.index)
    # Expected absolute 5-day move from VIX, used only to define a range/short-vol target.
    out["_vix_level"] = vix
    out["_vix_5d_roc_safe"] = out.groupby("symbol")["_vix_level"].pct_change(5)
    expected_move_5d = (vix / 100.0) * math.sqrt(TARGET_HORIZON / 252)
    fallback_move = out["_target_fwd_ret_5d"].abs().rolling(60, min_periods=20).median()
    out["_expected_move_5d"] = expected_move_5d.fillna(fallback_move)
    out["_target_inside_expected_move"] = (
        out["_target_fwd_ret_5d"].abs() <= out["_expected_move_5d"]
    ).astype(int)

    return out


def chronological_split(df: pd.DataFrame, train_frac: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    dates = np.array(sorted(df["date"].dropna().unique()))
    if len(dates) < 20:
        raise ValueError(f"not enough unique dates for chronological split: {len(dates)}")
    split_idx = max(1, min(len(dates) - 1, int(len(dates) * train_frac)))
    split_date = pd.Timestamp(dates[split_idx])
    train = df[df["date"] < split_date].copy()
    test = df[df["date"] >= split_date].copy()
    return train, test, split_date


def select_features(train: pd.DataFrame, test: pd.DataFrame, scope: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    leakage_prefixes = ("fwd_", "priced_exit_", "_target_")
    leakage_exact = {
        "date",
        "symbol",
        "dataset_split",
        "price_source",
        "cache_spy_iv_source",
        "ret_1d",
        "ret_5d",
        "ret_20d",
        "rv_10d",
        "rv_20d",
    }
    numeric_cols = list(train.select_dtypes(include=[np.number, "bool"]).columns)
    candidates: list[str] = []
    for col in numeric_cols:
        lower = col.lower()
        if col in leakage_exact:
            continue
        if lower.startswith(leakage_prefixes):
            continue
        if "future" in lower or "forward" in lower:
            continue
        if train[col].notna().mean() < 0.60:
            continue
        candidates.append(col)

    X_train = train[candidates].copy()
    X_test = test[candidates].copy()

    if scope == "all":
        train_symbols = pd.get_dummies(train["symbol"], prefix="sym", dtype=float)
        test_symbols = pd.get_dummies(test["symbol"], prefix="sym", dtype=float)
        test_symbols = test_symbols.reindex(columns=train_symbols.columns, fill_value=0.0)
        X_train = pd.concat([X_train.reset_index(drop=True), train_symbols.reset_index(drop=True)], axis=1)
        X_test = pd.concat([X_test.reset_index(drop=True), test_symbols.reset_index(drop=True)], axis=1)

    return X_train, X_test, list(X_train.columns)


def fit_imputer(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    imputer = SimpleImputer(strategy="median")
    Xtr = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    Xte = pd.DataFrame(imputer.transform(X_test), columns=X_train.columns)
    return Xtr, Xte


def regression_metrics(y_true: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(y_true, pred))
    corr = np.corrcoef(y_true, pred)[0, 1] if len(y_true) > 2 else np.nan
    return {
        "mae": float(mean_absolute_error(y_true, pred)),
        "rmse": float(rmse),
        "r2": float(r2_score(y_true, pred)),
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(pred))),
        "prediction_realized_corr": float(corr) if np.isfinite(corr) else 0.0,
    }


def classification_metrics(y_true: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    pred = (prob >= 0.5).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "brier": float(brier_score_loss(y_true, prob)),
        "positive_rate_realized": float(np.mean(y_true)),
        "positive_rate_predicted": float(np.mean(pred)),
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = float(roc_auc_score(y_true, prob))
    else:
        out["roc_auc"] = 0.0
    return out


def nonoverlap_spy_backtest(pred_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    spy = pred_df[pred_df["symbol"] == "SPY"].sort_values("date").reset_index(drop=True)
    if len(spy) < 10:
        return pd.DataFrame(), {}

    # Take one 5-day signal at a time so returns are not overlapping.
    rows = spy.iloc[::TARGET_HORIZON].copy()
    rows["signal_lgbm_return"] = np.where(
        rows["pred_ret_lgbm"] > 0.0025,
        1.0,
        np.where(rows["pred_ret_lgbm"] < -0.0025, -1.0, 0.0),
    )
    rows["signal_lgbm_direction"] = np.where(rows["prob_up_lgbm"] > 0.55, 1.0, 0.0)
    rows["signal_lgbm_range"] = np.where(rows["prob_range_lgbm"] > 0.55, 1.0, 0.0)

    rows["ret_return_model"] = rows["signal_lgbm_return"] * rows["realized_fwd_ret_5d"]
    rows["ret_direction_model"] = rows["signal_lgbm_direction"] * rows["realized_fwd_ret_5d"]
    rows["ret_buy_hold_5d"] = rows["realized_fwd_ret_5d"]
    # Range proxy: an iron-condor-ish payoff approximation, not a real fill model.
    credit = 0.45 * rows["expected_move_5d"].clip(lower=0.005)
    loss = np.maximum(rows["realized_fwd_ret_5d"].abs() - rows["expected_move_5d"], 0.0)
    rows["ret_range_proxy"] = rows["signal_lgbm_range"] * (credit - loss)

    for col in ("ret_return_model", "ret_direction_model", "ret_buy_hold_5d", "ret_range_proxy"):
        rows[f"equity_{col}"] = (1.0 + rows[col].fillna(0.0)).cumprod()

    def stats_for(col: str) -> dict[str, float]:
        r = rows[col].fillna(0.0).to_numpy()
        eq = (1.0 + r).cumprod()
        peak = np.maximum.accumulate(eq)
        dd = eq / np.maximum(peak, 1e-9) - 1.0
        years = max((pd.to_datetime(rows["date"].iloc[-1]) - pd.to_datetime(rows["date"].iloc[0])).days / 365.0, 1 / 365)
        cagr = eq[-1] ** (1 / years) - 1 if eq[-1] > 0 else -1.0
        monthly = (1 + cagr) ** (1 / 12) - 1 if cagr > -1 else -1.0
        return {
            "total_return_pct": float((eq[-1] - 1) * 100),
            "monthly_pct": float(monthly * 100),
            "max_dd_pct": float(dd.min() * 100),
            "trades": int(np.count_nonzero(np.abs(r) > 0)),
        }

    stats = {
        "lgbm_return_long_short": stats_for("ret_return_model"),
        "lgbm_direction_long_cash": stats_for("ret_direction_model"),
        "lgbm_range_proxy": stats_for("ret_range_proxy"),
        "buy_hold_spy_5d_steps": stats_for("ret_buy_hold_5d"),
    }
    return rows, stats


def make_chart(equity: pd.DataFrame, out_path: Path) -> None:
    if equity.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 7))
    dates = pd.to_datetime(equity["date"])
    series = {
        "LGBM return long/short": "equity_ret_return_model",
        "LGBM direction long/cash": "equity_ret_direction_model",
        "LGBM range proxy": "equity_ret_range_proxy",
        "Buy/hold SPY": "equity_ret_buy_hold_5d",
    }
    for label, col in series.items():
        ax.plot(dates, equity[col], linewidth=2, label=label)
    ax.axhline(1.0, color="#777777", linestyle="--", alpha=0.6)
    ax.set_title("SPY newest-30% test window: non-overlapping 5-day model trades")
    ax.set_ylabel("Growth of $1")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def run(scope: str, train_frac: float, output_dir: Path | None) -> dict[str, Any]:
    raw = load_panel(scope)
    panel = add_targets_and_safe_features(raw)
    usable = panel.dropna(subset=["_spot_close", "_target_fwd_ret_5d", "_expected_move_5d"]).copy()
    usable = usable.sort_values(["date", "symbol"]).reset_index(drop=True)

    train, test, split_date = chronological_split(usable, train_frac)
    X_train_raw, X_test_raw, feature_names = select_features(train, test, scope)
    X_train, X_test = fit_imputer(X_train_raw, X_test_raw)

    y_train_ret = train["_target_fwd_ret_5d"].to_numpy(float)
    y_test_ret = test["_target_fwd_ret_5d"].to_numpy(float)
    y_train_up = train["_target_direction_up"].to_numpy(int)
    y_test_up = test["_target_direction_up"].to_numpy(int)
    y_train_range = train["_target_inside_expected_move"].to_numpy(int)
    y_test_range = test["_target_inside_expected_move"].to_numpy(int)

    reg_lgbm = lgb.LGBMRegressor(
        objective="regression",
        metric="rmse",
        max_depth=4,
        n_estimators=500,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    reg_lgbm.fit(X_train, y_train_ret)
    pred_ret_lgbm = reg_lgbm.predict(X_test)

    reg_ridge = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=5.0))
    reg_ridge.fit(X_train_raw, y_train_ret)
    pred_ret_ridge = reg_ridge.predict(X_test_raw)

    clf_lgbm = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        max_depth=4,
        n_estimators=400,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    clf_lgbm.fit(X_train, y_train_up)
    prob_up_lgbm = clf_lgbm.predict_proba(X_test)[:, 1]

    clf_logit = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=0.5, random_state=RANDOM_STATE),
    )
    clf_logit.fit(X_train_raw, y_train_up)
    prob_up_logit = clf_logit.predict_proba(X_test_raw)[:, 1]

    range_lgbm = lgb.LGBMClassifier(
        objective="binary",
        metric="binary_logloss",
        max_depth=4,
        n_estimators=400,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        random_state=RANDOM_STATE,
        verbose=-1,
    )
    range_lgbm.fit(X_train, y_train_range)
    prob_range_lgbm = range_lgbm.predict_proba(X_test)[:, 1]

    pred_df = test[["date", "symbol", "_target_fwd_ret_5d", "_expected_move_5d"]].copy()
    pred_df = pred_df.rename(
        columns={
            "_target_fwd_ret_5d": "realized_fwd_ret_5d",
            "_expected_move_5d": "expected_move_5d",
        }
    )
    pred_df["pred_ret_lgbm"] = pred_ret_lgbm
    pred_df["pred_ret_ridge"] = pred_ret_ridge
    pred_df["prob_up_lgbm"] = prob_up_lgbm
    pred_df["prob_up_logit"] = prob_up_logit
    pred_df["prob_range_lgbm"] = prob_range_lgbm
    pred_df["actual_up"] = y_test_up
    pred_df["actual_inside_expected_move"] = y_test_range

    equity, strategy_stats = nonoverlap_spy_backtest(pred_df)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = output_dir or REPO_ROOT / "experiments" / "runs" / f"final_dataset_model_test_{scope}_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "predictions.csv", index=False)
    if not equity.empty:
        equity.to_csv(out_dir / "spy_nonoverlap_equity.csv", index=False)
        make_chart(equity, out_dir / "spy_test_equity.png")

    fi = pd.DataFrame(
        {
            "feature": feature_names,
            "return_model_importance": reg_lgbm.feature_importances_,
            "direction_model_importance": clf_lgbm.feature_importances_,
            "range_model_importance": range_lgbm.feature_importances_,
        }
    ).sort_values("return_model_importance", ascending=False)
    fi.to_csv(out_dir / "feature_importance.csv", index=False)

    metrics: dict[str, Any] = {
        "scope": scope,
        "train_fraction": train_frac,
        "split_date": split_date.date().isoformat(),
        "rows_total_usable": int(len(usable)),
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "symbols_train": int(train["symbol"].nunique()),
        "symbols_test": int(test["symbol"].nunique()),
        "train_date_min": train["date"].min().date().isoformat(),
        "train_date_max": train["date"].max().date().isoformat(),
        "test_date_min": test["date"].min().date().isoformat(),
        "test_date_max": test["date"].max().date().isoformat(),
        "n_features": int(len(feature_names)),
        "models": {
            "lightgbm_return_regressor": regression_metrics(y_test_ret, pred_ret_lgbm),
            "ridge_return_baseline": regression_metrics(y_test_ret, pred_ret_ridge),
            "lightgbm_direction_classifier": classification_metrics(y_test_up, prob_up_lgbm),
            "logistic_direction_baseline": classification_metrics(y_test_up, prob_up_logit),
            "lightgbm_range_classifier": classification_metrics(y_test_range, prob_range_lgbm),
        },
        "spy_nonoverlap_strategy_stats": strategy_stats,
        "top_features_return_model": fi.head(15).to_dict("records"),
        "output_dir": str(out_dir),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=_json_default))

    report = [
        "# Final Dataset Model Test",
        "",
        f"- Scope: `{scope}`",
        f"- Chronological split: {train_frac:.0%} train / {1 - train_frac:.0%} test",
        f"- Split date: `{metrics['split_date']}`",
        f"- Train rows: `{metrics['rows_train']}`; test rows: `{metrics['rows_test']}`",
        f"- Features: `{metrics['n_features']}`",
        "",
        "## Model Metrics",
        "",
        "| Model | Key result |",
        "|---|---:|",
        f"| LGBM return regressor direction accuracy | {metrics['models']['lightgbm_return_regressor']['directional_accuracy']:.2%} |",
        f"| LGBM return regressor correlation | {metrics['models']['lightgbm_return_regressor']['prediction_realized_corr']:.3f} |",
        f"| Ridge return baseline direction accuracy | {metrics['models']['ridge_return_baseline']['directional_accuracy']:.2%} |",
        f"| LGBM direction classifier ROC AUC | {metrics['models']['lightgbm_direction_classifier']['roc_auc']:.3f} |",
        f"| Logistic direction baseline ROC AUC | {metrics['models']['logistic_direction_baseline']['roc_auc']:.3f} |",
        f"| LGBM range classifier ROC AUC | {metrics['models']['lightgbm_range_classifier']['roc_auc']:.3f} |",
        "",
        "## SPY Non-Overlapping 5-Day Strategy Proxy",
        "",
        "| Strategy | Total return | Monthly | Max DD | Trades |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, stats in strategy_stats.items():
        report.append(
            f"| {name} | {stats['total_return_pct']:.2f}% | {stats['monthly_pct']:.2f}% | "
            f"{stats['max_dd_pct']:.2f}% | {stats['trades']} |"
        )
    report.extend(
        [
            "",
            "Notes: the range strategy is a proxy payoff, not a broker fill model. "
            "The split is chronological by date to avoid training on future regimes.",
        ]
    )
    (out_dir / "report.md").write_text("\n".join(report) + "\n")

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", choices=["spy", "all"], default="all")
    parser.add_argument("--train-frac", type=float, default=0.70)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    metrics = run(args.scope, args.train_frac, args.output_dir)
    print(json.dumps(metrics, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
