"""Visualize candidate_v5 backtest results.

Outputs:
  research/candidate_v5/results/equity_curves.html       (interactive plotly)
  research/candidate_v5/results/drawdowns.html
  research/candidate_v5/results/regime_performance.html

Run after backtest.py — reads train_returns.parquet + oos_returns.parquet
and the latest backtest_summary_*.json.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "research" / "candidate_v5" / "results"
V3_CACHE = ROOT / "data" / "cache" / "candidate_v3"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("v5.charts")


def equity_curve(returns: pd.Series, initial: float = 100_000) -> pd.Series:
    return (1 + returns.fillna(0)).cumprod() * initial


def drawdown(returns: pd.Series) -> pd.Series:
    eq = (1 + returns.fillna(0)).cumprod()
    peak = eq.cummax()
    return (eq / peak - 1)


def load_1N_and_spy_oos() -> tuple[pd.Series, pd.Series]:
    """Load 1/N benchmark and SPY buy-hold for OOS comparison."""
    UNIVERSE = ["SPY","QQQ","IWM","EFA","EEM","TLT","IEF","LQD","HYG","GLD","DBC","VNQ"]
    frames = []
    for sym in UNIVERSE:
        df = pd.read_parquet(V3_CACHE / f"{sym}.parquet")[["date", "close"]]
        df = df.rename(columns={"close": sym}).set_index("date")
        frames.append(df)
    wide = pd.concat(frames, axis=1).sort_index()
    wide.index = pd.to_datetime(wide.index)
    wide = wide.ffill(limit=2).dropna()
    oos = wide[wide.index >= pd.Timestamp("2023-01-01")]
    returns = oos.pct_change().dropna()
    n = len(returns.columns)
    one_n = (returns * (1.0 / n)).sum(axis=1)
    spy = returns["SPY"]
    return one_n, spy


def main() -> int:
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        log.error("plotly not installed; run: pip install plotly")
        return 1

    # Load most recent summary
    summaries = sorted(RESULTS_DIR.glob("backtest_summary_*.json"))
    if not summaries:
        log.error("no backtest summary found")
        return 1
    summary = json.loads(summaries[-1].read_text())
    best_variant = summary["best_variant"]
    log.info("best variant: %s", best_variant)

    train_df = pd.read_parquet(RESULTS_DIR / "train_returns.parquet")
    oos_df = pd.read_parquet(RESULTS_DIR / "oos_returns.parquet")
    v5_train = train_df[best_variant]
    v5_oos = oos_df[best_variant]

    one_n_oos, spy_oos = load_1N_and_spy_oos()
    # Align
    common = v5_oos.index.intersection(one_n_oos.index).intersection(spy_oos.index)
    v5_oos_a = v5_oos.loc[common]
    one_n_oos_a = one_n_oos.loc[common]
    spy_oos_a = spy_oos.loc[common]

    # --- Equity curves: train ---
    fig1 = make_subplots(rows=2, cols=1, subplot_titles=("Train: 2002-2022", "OOS: 2023-2025"),
                         vertical_spacing=0.12)
    fig1.add_trace(go.Scatter(x=v5_train.index, y=equity_curve(v5_train),
                              name="v5 (best variant)", line=dict(color="blue")), row=1, col=1)
    fig1.add_trace(go.Scatter(x=v5_oos_a.index, y=equity_curve(v5_oos_a),
                              name="v5 OOS", line=dict(color="blue")), row=2, col=1)
    fig1.add_trace(go.Scatter(x=one_n_oos_a.index, y=equity_curve(one_n_oos_a),
                              name="1/N benchmark", line=dict(color="gray", dash="dash")), row=2, col=1)
    fig1.add_trace(go.Scatter(x=spy_oos_a.index, y=equity_curve(spy_oos_a),
                              name="SPY buy-hold", line=dict(color="black", dash="dot")), row=2, col=1)
    fig1.update_layout(title=f"Candidate v5 equity curves — {best_variant}",
                       height=700, hovermode="x unified")
    fig1.update_xaxes(title_text="date")
    fig1.update_yaxes(title_text="NAV ($, $100k start)")
    out1 = RESULTS_DIR / "equity_curves.html"
    fig1.write_html(out1)
    log.info("wrote %s", out1.relative_to(ROOT))

    # --- Drawdowns ---
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=v5_oos_a.index, y=drawdown(v5_oos_a) * 100,
                              name="v5", line=dict(color="blue"), fill="tozeroy"))
    fig2.add_trace(go.Scatter(x=one_n_oos_a.index, y=drawdown(one_n_oos_a) * 100,
                              name="1/N", line=dict(color="gray", dash="dash")))
    fig2.add_trace(go.Scatter(x=spy_oos_a.index, y=drawdown(spy_oos_a) * 100,
                              name="SPY", line=dict(color="black", dash="dot")))
    fig2.update_layout(title="OOS drawdowns (%) — lower (more negative) is worse",
                       xaxis_title="date", yaxis_title="drawdown (%)",
                       height=500, hovermode="x unified")
    out2 = RESULTS_DIR / "drawdowns.html"
    fig2.write_html(out2)
    log.info("wrote %s", out2.relative_to(ROOT))

    # --- Performance table ---
    def stats(r: pd.Series, label: str) -> dict:
        r = r.dropna()
        ann_ret = r.mean() * 252
        ann_vol = r.std() * (252 ** 0.5)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        eq = (1 + r).cumprod()
        dd = (eq / eq.cummax() - 1).min()
        return {
            "label": label,
            "ann_ret_pct": round(ann_ret * 100, 2),
            "ann_vol_pct": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 3),
            "max_dd_pct": round(dd * 100, 2),
            "calmar": round(ann_ret / abs(dd), 3) if dd != 0 else 0,
            "tot_ret_pct": round((eq.iloc[-1] - 1) * 100, 2) if len(eq) else 0,
        }

    print("\n=== OOS PERFORMANCE COMPARISON ===")
    print(f"{'Strategy':20} {'AnnRet%':>8} {'AnnVol%':>8} {'Sharpe':>7} {'MaxDD%':>7} {'Calmar':>7} {'TotRet%':>8}")
    print("-" * 70)
    for r, lab in [(v5_oos_a, "v5 best"), (one_n_oos_a, "1/N"), (spy_oos_a, "SPY buy-hold")]:
        s = stats(r, lab)
        print(f"{s['label']:20} {s['ann_ret_pct']:>8.2f} {s['ann_vol_pct']:>8.2f} {s['sharpe']:>7.3f} {s['max_dd_pct']:>7.2f} {s['calmar']:>7.3f} {s['tot_ret_pct']:>8.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
