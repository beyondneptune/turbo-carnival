"""Phoenix two-speed engine.

Brand-new final experiment built from what we learned:

1. Slow alpha core: SPY options strategies that actually worked in the 70/30
   tests, selected by regime and protected by a local NN bad-trade model.
2. Fast scout sleeve: Valk-style intraday HFT-ish scalping on Alpaca 1-minute
   bars. It is included, but it only earns allocation if it is profitable after
   slippage in the recent rolling paper-data test.

The point is not "more knobs". It is a capital allocator:
  - core options do the heavy lifting;
  - aggressive options only run in calm/positive-VRP regimes;
  - HFT is tested continuously but quarantined if its edge is negative;
  - drawdown throttles shrink risk before account damage gets stupid.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_aggressive_meta_engine import (  # noqa: E402
    DEFAULT_ROOT,
    OUT_ROOT,
    benchmark,
    load_period,
    load_specs,
    pct,
    train_scores,
    usd,
)
from experiments.run_drawdown_risk_engine import (  # noqa: E402
    RiskConfig,
    replay_risk_engine,
    train_risk_model,
)
from experiments.run_optimized_combined_engine import (  # noqa: E402
    Recipe,
    add_combined_scores,
    safe_monthly,
)
from experiments.valk_hft_demo import (  # noqa: E402
    CONFIGS as HFT_CONFIGS,
    add_features as hft_add_features,
    fetch_minute_bars,
    make_report as make_hft_report,
    run_strategy as run_hft_strategy,
)


CORE_STRATEGY = "risk_qwen_risk_designer_risktest_d12_30d_lowdelta_pt30_sl20_c10"
AGGRESSIVE_STRATEGIES = {
    "aggressive_daily_nogate_d20_30d_w10",
    "aggressive_daily_nogate_d16_30d_w10",
    "daily_ladder_vrp_positive_d16_45d_w5",
    "daily_ladder_vrp_positive_d16_30d_w5",
    "execution_llama_allocator_designer_daily_ladder_vrp_positive_d12_30d_w5_stop_loss_1_5",
}


def regime_state(row: pd.Series) -> str:
    """Simple readable regime label used by the allocator."""
    ret_5d = abs(float(row.get("ret_5d", 0.0) or 0.0))
    dd_20d = float(row.get("drawdown_20d", 0.0) or 0.0)
    iv_rank = float(row.get("iv_rank_252", 0.5) or 0.5)
    vrp = float(row.get("vrp20", 0.0) or 0.0)
    if ret_5d > 0.045 or dd_20d < -0.07 or iv_rank > 0.88:
        return "shock"
    if ret_5d > 0.030 or dd_20d < -0.045 or iv_rank > 0.75:
        return "caution"
    if vrp > 0 and iv_rank < 0.70:
        return "calm_vrp_positive"
    return "neutral"


def select_phoenix_options(scored: pd.DataFrame) -> pd.DataFrame:
    """Select entries for the Phoenix options core.

    Rules:
    - always consider the proven fixed_best_qwen/core candidate;
    - add one or two aggressive candidates only in calm, positive-VRP regimes;
    - skip aggressive entries in shock/caution regimes;
    - score aggressive candidates using the trained combined/risk-adjusted score.
    """
    out = scored.copy()
    out["take"] = False
    out["selector_score"] = out["risk_adjusted_score"].fillna(out["combined_score"]).fillna(0.0)
    out["phoenix_regime"] = out.apply(regime_state, axis=1)
    out["phoenix_slot"] = ""

    for _, idx in out.groupby("entry_date").groups.items():
        w = out.loc[list(idx)].copy()
        regime = str(w["phoenix_regime"].iloc[0])

        core = w[w["strategy"] == CORE_STRATEGY].copy()
        if not core.empty:
            core_idx = core.sort_values("risk_adjusted_score", ascending=False).head(1).index
            out.loc[core_idx, "take"] = True
            out.loc[core_idx, "selector_score"] = 2.0 + out.loc[core_idx, "selector_score"]
            out.loc[core_idx, "phoenix_slot"] = "core"

        if regime in {"shock", "caution"}:
            continue

        candidates = w[w["strategy"].isin(AGGRESSIVE_STRATEGIES)].copy()
        candidates = candidates[candidates["risk_adjusted_score"].fillna(-999) > -0.25]
        if candidates.empty:
            continue
        n = 2 if regime == "calm_vrp_positive" else 1
        chosen = candidates.sort_values("risk_adjusted_score", ascending=False).head(n).index
        out.loc[chosen, "take"] = True
        out.loc[chosen, "phoenix_slot"] = "aggressive_booster"

    return out


def select_core_only(scored: pd.DataFrame) -> pd.DataFrame:
    """Only the best proven fixed Qwen candidate, no boosters."""
    out = scored.copy()
    out["take"] = False
    out["selector_score"] = out["risk_adjusted_score"].fillna(out["combined_score"]).fillna(0.0)
    out["phoenix_regime"] = out.apply(regime_state, axis=1)
    out["phoenix_slot"] = ""
    mask = out["strategy"] == CORE_STRATEGY
    out.loc[mask, "take"] = True
    out.loc[mask, "selector_score"] = 2.0 + out.loc[mask, "selector_score"]
    out.loc[mask, "phoenix_slot"] = "core_only"
    return out


def risk_grid() -> list[RiskConfig]:
    return [
        RiskConfig("core_static_5_50", "static", 5.0, 50.0),
        RiskConfig("core_static_10_100", "static", 10.0, 100.0),
        RiskConfig("core_hybrid_10_100", "hybrid", 10.0, 100.0, nn_threshold=0.55, max_bad_prob=0.90, dd_profile="soft"),
        RiskConfig("phoenix_guarded_8_80", "hybrid", 8.0, 80.0, nn_threshold=0.55, max_bad_prob=0.90, dd_profile="target10"),
        RiskConfig("phoenix_balanced_10_100", "hybrid", 10.0, 100.0, nn_threshold=0.55, max_bad_prob=0.90, dd_profile="soft"),
        RiskConfig("phoenix_attack_12_125", "hybrid", 12.0, 125.0, nn_threshold=0.55, max_bad_prob=0.90, dd_profile="soft"),
        RiskConfig("phoenix_static_8_80", "static", 8.0, 80.0),
        RiskConfig("phoenix_static_10_100", "static", 10.0, 100.0),
    ]


def choose_config(train_rows: pd.DataFrame) -> RiskConfig:
    ranked = train_rows.copy()
    ranked["objective"] = ranked["return_pct"] - 4.0 * ranked["max_drawdown_pct"].abs()
    ranked["target_penalty"] = np.where(ranked["max_drawdown_pct"] < -30.0, 500.0, 0.0)
    ranked["target_penalty"] += np.where(ranked["worst_trade_usd"].abs() > 10_000.0, 150.0, 0.0)
    ranked["objective"] = ranked["objective"] - ranked["target_penalty"]
    chosen_name = str(ranked.sort_values("objective", ascending=False).iloc[0]["config_name"])
    return {c.name: c for c in risk_grid()}[chosen_name]


def run_option_phoenix(root: Path, capital: float) -> dict[str, Any]:
    specs = load_specs()
    train_market, train_trades = load_period(root, "train", specs)
    test_market, test_trades = load_period(root, "test", specs)

    train_scored, train_meta = train_scores(train_trades, train_trades)
    test_scored, _ = train_scores(train_trades, test_trades)
    train_scored = add_combined_scores(train_scored, train_trades)
    test_scored = add_combined_scores(test_scored, train_trades)

    model, model_meta = train_risk_model(train_scored)
    recipe = Recipe(
        name="phoenix_options_allocator",
        selector="phoenix",
        score_col="risk_adjusted_score",
        top_n=3,
        positive_only=False,
        base_risk_pct=10.0,
        concurrent_pct=100.0,
        throttle="nn_dd",
        score_sizing=True,
    )

    selected_train_by_mode = {
        "core_only": select_core_only(train_scored),
        "core_plus_boosters": select_phoenix_options(train_scored),
    }
    train_rows = []
    for mode_name, selected_train in selected_train_by_mode.items():
        for cfg in risk_grid():
            if mode_name == "core_only" and cfg.name.startswith("phoenix_"):
                continue
            if mode_name == "core_plus_boosters" and cfg.name.startswith("core_"):
                continue
            _, _, summary = replay_risk_engine(
                f"phoenix_train_{mode_name}_{cfg.name}",
                recipe,
                cfg,
                train_market,
                selected_train,
                capital=capital,
                model=model,
            )
            summary["config_name"] = cfg.name
            summary["selection_mode"] = mode_name
            train_rows.append(summary)
    train_summary = pd.DataFrame(train_rows)
    chosen = choose_config(train_summary)
    chosen_row = train_summary[train_summary["config_name"] == chosen.name].sort_values("return_pct", ascending=False).iloc[0]
    chosen_mode = str(chosen_row["selection_mode"])

    selected_test = select_core_only(test_scored) if chosen_mode == "core_only" else select_phoenix_options(test_scored)
    nav, trades, summary = replay_risk_engine(
        f"phoenix_options_{chosen_mode}_{chosen.name}",
        recipe,
        chosen,
        test_market,
        selected_test,
        capital=capital,
        model=model,
    )
    summary["config_name"] = chosen.name
    summary["selection_mode"] = chosen_mode
    summary["chosen_config"] = chosen.to_dict()

    return {
        "train_market": train_market,
        "test_market": test_market,
        "train_scored": train_scored,
        "test_scored": test_scored,
        "selected_train": selected_train,
        "selected_test": selected_test,
        "train_summary": train_summary,
        "nav": nav,
        "trades": trades,
        "summary": summary,
        "train_meta": train_meta,
        "risk_model_meta": model_meta,
    }


def run_hft_scout(days: int, capital: float, out_dir: Path) -> dict[str, Any]:
    end = pd.Timestamp.now(tz="UTC").to_pydatetime()
    start = (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)).to_pydatetime()
    symbols = sorted({sym for cfg in HFT_CONFIGS for sym in cfg.symbols})
    bars = hft_add_features(fetch_minute_bars(symbols, start=start, end=end, feed="iex"))
    results = []
    for cfg in HFT_CONFIGS:
        subset = bars[bars["symbol"].isin(cfg.symbols)].copy()
        results.append(run_hft_strategy(subset, cfg, capital))
    hft_dir = out_dir / "hft_scout"
    report = make_hft_report(results, hft_dir, bars)
    summaries = pd.DataFrame([r["summary"] for r in results]).sort_values("return_pct", ascending=False)
    best = summaries.iloc[0].to_dict()
    enabled = (
        float(best["return_pct"]) > 0.0
        and float(best["max_drawdown_pct"]) > -8.0
        and float(best["profit_factor"]) > 1.10
        and float(best["trades_per_day"]) >= 20.0
    )
    allocation_pct = 10.0 if enabled else 0.0
    return {
        "bars": bars,
        "results": results,
        "summaries": summaries,
        "best": best,
        "enabled": enabled,
        "allocation_pct": allocation_pct,
        "report": report,
    }


def summary_table(rows: list[dict[str, Any]]) -> str:
    df = pd.DataFrame(rows)
    for c in ("return_pct", "monthly_pct", "max_drawdown_pct", "win_rate_pct"):
        if c in df:
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else pct(float(x)))
    for c in ("final_nav", "worst_trade_usd", "max_open_risk_usd"):
        if c in df:
            df[c] = df[c].map(lambda x: "" if pd.isna(x) else usd(float(x)))
    return df.to_html(index=False, escape=False, classes="data-table")


def trade_table(trades: pd.DataFrame) -> str:
    if trades.empty:
        return "<p>No trades.</p>"
    w = trades.copy().sort_values(["entry_date", "strategy"])
    cols = {
        "entry": pd.to_datetime(w["entry_date"]).dt.date.astype(str),
        "exit": pd.to_datetime(w["exit_date"]).dt.date.astype(str),
        "strategy": w["strategy"],
        "slot": w.get("phoenix_slot", ""),
        "reason": w["reason"],
        "qty": w["qty"].astype(int),
        "risk_mult": w["risk_mult"].map(lambda x: f"{float(x):.2f}"),
        "bad_prob": w["bad_prob"].map(lambda x: f"{float(x):.2f}"),
        "live_dd": w["live_dd_at_entry"].map(lambda x: pct(float(x))),
        "legs": w["structure"],
        "pnl": w["sized_pnl_usd"].map(lambda x: usd(float(x))),
        "risk": w["sized_max_loss_usd"].map(lambda x: usd(float(x))),
    }
    return pd.DataFrame(cols).to_html(index=False, escape=False, classes="data-table")


def build_report(out_dir: Path, option: dict[str, Any], hft: dict[str, Any] | None, capital: float) -> Path:
    nav = option["nav"].copy()
    trades = option["trades"].copy()
    test_market = option["test_market"]
    bench = benchmark(test_market, capital)
    bret = (float(bench["nav"].iloc[-1]) / capital - 1.0) * 100.0
    bpeak = bench["nav"].cummax()
    bench_row = {
        "engine": "SPY benchmark",
        "final_nav": float(bench["nav"].iloc[-1]),
        "return_pct": bret,
        "monthly_pct": safe_monthly(float(bench["nav"].iloc[-1]), capital, pd.Timestamp(bench["date"].iloc[0]), pd.Timestamp(bench["date"].iloc[-1])),
        "max_drawdown_pct": float((bench["nav"] / bpeak - 1.0).min() * 100.0),
        "trades": 1,
        "win_rate_pct": np.nan,
        "worst_trade_usd": np.nan,
        "max_open_risk_usd": capital,
    }
    option_row = {k: v for k, v in option["summary"].items() if not isinstance(v, dict)}
    rows = [option_row, bench_row]
    if hft is not None:
        best = hft["best"]
        rows.append(
            {
                "engine": "HFT scout best recent",
                "final_nav": float(best["final_nav"]),
                "return_pct": float(best["return_pct"]),
                "monthly_pct": float(best["monthly_pct"]),
                "max_drawdown_pct": float(best["max_drawdown_pct"]),
                "trades": int(best["trades"]),
                "win_rate_pct": float(best["win_rate_pct"]),
                "worst_trade_usd": float(best["worst_trade"]),
                "max_open_risk_usd": np.nan,
            }
        )

    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.52, 0.22, 0.26],
        vertical_spacing=0.08,
        subplot_titles=("Phoenix options held-out equity", "Options drawdown", "HFT scout recent equity"),
    )
    fig.add_trace(go.Scatter(x=nav["date"], y=nav["nav"], name="Phoenix options", line=dict(color="#111827", width=2.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=bench["date"], y=bench["nav"], name="SPY benchmark", line=dict(color="#64748b", width=2, dash="dot")), row=1, col=1)
    dd = nav["nav"] / nav["nav"].cummax() - 1.0
    fig.add_trace(go.Scatter(x=nav["date"], y=dd * 100.0, name="Phoenix DD", showlegend=False, line=dict(color="#dc2626", width=1.8)), row=2, col=1)
    if hft is not None:
        for r in hft["results"]:
            eq = r["equity"]
            fig.add_trace(
                go.Scatter(x=pd.to_datetime(eq["t"], utc=True), y=eq["nav"], name=r["summary"]["strategy"], line=dict(width=1.5)),
                row=3,
                col=1,
            )
    fig.update_layout(height=1050, template="plotly_white", hovermode="x unified", legend=dict(orientation="h", y=1.06))
    fig.update_yaxes(title_text="NAV ($)", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)", row=2, col=1)
    fig.update_yaxes(title_text="NAV ($)", row=3, col=1)

    hft_decision = "not run"
    hft_table = ""
    hft_link = ""
    if hft is not None:
        hft_decision = "ENABLED" if hft["enabled"] else "QUARANTINED"
        hft_table = hft["summaries"].to_html(index=False, escape=False, classes="data-table", float_format=lambda x: f"{x:,.2f}")
        hft_link = f"<p>HFT detail report: <a href='{os.path.relpath(hft['report'], out_dir)}'>{hft['report'].name}</a></p>"

    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; color: #111827; }
      h1 { margin-bottom: 4px; }
      h2 { margin-top: 32px; padding-top: 16px; border-top: 1px solid #d1d5db; }
      .note { color: #4b5563; max-width: 1160px; line-height: 1.45; }
      .decision { display:inline-block; padding:8px 10px; border-radius:6px; background:#f3f4f6; border:1px solid #d1d5db; font-weight:700; }
      .scroll { overflow-x:auto; border:1px solid #e5e7eb; padding:8px; }
      table.data-table { border-collapse:collapse; width:100%; font-size:12px; }
      table.data-table th, table.data-table td { border:1px solid #d1d5db; padding:6px 8px; text-align:right; white-space:nowrap; }
      table.data-table th { background:#f9fafb; }
      table.data-table td:first-child, table.data-table th:first-child,
      table.data-table td:nth-child(2), table.data-table th:nth-child(2),
      table.data-table td:nth-child(3), table.data-table th:nth-child(3) { text-align:left; }
      pre { background:#f9fafb; border:1px solid #e5e7eb; padding:12px; overflow-x:auto; }
    </style>
    """
    report = out_dir / "phoenix_two_speed_engine_report.html"
    html = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Phoenix Two-Speed Engine</title>",
        css,
        "</head><body>",
        f"<h1>Phoenix Two-Speed Engine (${capital:,.0f})</h1>",
        "<p class='note'>This is the rebuilt engine: SPY options core plus a Valk HFT scout. "
        "The HFT sleeve is remembered and tested, but it does not receive live capital unless its rolling minute-bar test is positive after slippage.</p>",
        f"<p class='decision'>HFT sleeve: {hft_decision}</p>",
        "<h2>Chart</h2>",
        fig.to_html(include_plotlyjs=True, full_html=False),
        "<h2>Summary</h2><div class='scroll'>",
        summary_table(rows),
        "</div>",
        "<h2>Chosen Options Risk Config</h2>",
        f"<pre>{json.dumps(option['summary'].get('chosen_config', {}), indent=2)}</pre>",
        "<h2>Train Config Search</h2><div class='scroll'>",
        option["train_summary"].sort_values("return_pct", ascending=False).to_html(index=False, escape=False, classes="data-table", float_format=lambda x: f"{x:,.2f}"),
        "</div>",
        "<h2>HFT Scout</h2>",
        hft_link,
        "<div class='scroll'>",
        hft_table or "<p>HFT scout skipped.</p>",
        "</div>",
        "<h2>Options Entries / Exits / Stops / P&L</h2><div class='scroll'>",
        trade_table(trades),
        "</div>",
        "<h2>Risk Model Metadata</h2>",
        f"<pre>{json.dumps(option['risk_model_meta'], indent=2)}</pre>",
        "</body></html>",
    ]
    report.write_text("\n".join(html), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--hft-days", type=int, default=30)
    parser.add_argument("--skip-hft", action="store_true")
    parser.add_argument("--open-report", action="store_true")
    args = parser.parse_args()

    stamp = pd.Timestamp.now("UTC").strftime("%Y%m%dT%H%M%SZ")
    out_dir = OUT_ROOT / f"phoenix_two_speed_engine_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Running Phoenix options allocator...")
    option = run_option_phoenix(args.root, args.capital)
    option["nav"].to_csv(out_dir / "phoenix_options_nav.csv", index=False)
    option["trades"].to_csv(out_dir / "phoenix_options_trades.csv", index=False)
    option["train_summary"].to_csv(out_dir / "phoenix_train_config_search.csv", index=False)
    pd.DataFrame([option["summary"]]).to_csv(out_dir / "phoenix_options_summary.csv", index=False)
    (out_dir / "phoenix_options_summary.json").write_text(json.dumps(option["summary"], indent=2, default=str))

    hft = None
    if not args.skip_hft:
        print("Running HFT scout...")
        hft = run_hft_scout(args.hft_days, args.capital, out_dir)
        hft["summaries"].to_csv(out_dir / "hft_scout_summary.csv", index=False)
        (out_dir / "hft_scout_decision.json").write_text(
            json.dumps(
                {
                    "enabled": hft["enabled"],
                    "allocation_pct": hft["allocation_pct"],
                    "best": hft["best"],
                    "report": str(hft["report"]),
                },
                indent=2,
                default=str,
            )
        )

    report = build_report(out_dir, option, hft, args.capital)
    print("\n=== PHOENIX OPTIONS SUMMARY ===")
    print(pd.DataFrame([option["summary"]]).to_string(index=False))
    if hft is not None:
        print("\n=== HFT SCOUT DECISION ===")
        print(f"enabled={hft['enabled']} allocation_pct={hft['allocation_pct']} best={hft['best']['strategy']} return={hft['best']['return_pct']:.2f}%")
    print(f"\nReport: {report}")
    if args.open_report:
        os.system(f"open {str(report)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
