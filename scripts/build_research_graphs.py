"""Build PNG graphs from experiment summaries + smoke run simulation.

Outputs to research/graphs/. Reads:
  * experiments/runs/strict_priced_stats_*/summary.csv   -> grid distribution
  * experiments/runs/explore_synth_spy_stats_*/summary.csv -> wider grid
  * data/trading_system.db (run_id paper_v1_unofficial_smoke_*) -> simulation

No network calls, no DB writes. Idempotent: overwrites graphs in place.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_DIR = REPO_ROOT / "research" / "graphs"

GREEN = "#2ecc71"
RED = "#e74c3c"
BLUE = "#3498db"
ORANGE = "#e67e22"
GRAY = "#7f8c8d"
PURPLE = "#8e44ad"
DARK = "#2c3e50"

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.dpi": 130,
    "savefig.bbox": "tight",
    "font.size": 11,
})


def _save(fig: plt.Figure, name: str) -> Path:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPH_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path.relative_to(REPO_ROOT)}")
    return path


def _save_html(fig, name: str) -> Path:
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    path = GRAPH_DIR / f"{name}.html"
    fig.write_html(
        path,
        include_plotlyjs=True,
        full_html=True,
        config={"responsive": True, "displaylogo": False},
    )
    print(f"  saved {path.relative_to(REPO_ROOT)}")
    return path


def load_all_experiment_returns() -> pd.DataFrame:
    """Load every experiment summary that carries realised return columns."""
    frames: list[pd.DataFrame] = []
    for csv_path in sorted((REPO_ROOT / "experiments" / "runs").glob("*/summary.csv")):
        try:
            df = pd.read_csv(csv_path)
        except (OSError, pd.errors.ParserError):
            continue
        if "return_on_10k" not in df.columns:
            continue
        df = df.copy()
        df["experiment_set"] = csv_path.parent.name
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["return_pct"] = pd.to_numeric(out["return_on_10k"], errors="coerce") * 100.0
    out["trade_count"] = pd.to_numeric(out.get("trade_count"), errors="coerce")
    out["test_start_dt"] = pd.to_datetime(out.get("test_start"), errors="coerce")
    return out


def graph_experiment_returns_distribution(strict_df: pd.DataFrame, synth_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, df, label, color in (
        (axes[0], strict_df, "Strict priced (96 windows, real Polygon data)", BLUE),
        (axes[1], synth_df, "Exploratory (120 windows, synthetic SPY)", ORANGE),
    ):
        if df.empty or "return_on_10k" not in df.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue
        complete = df[df["status"] == "complete"]
        rets = complete["return_on_10k"].astype(float) * 100.0
        ax.hist(rets, bins=25, color=color, alpha=0.85, edgecolor="white")
        ax.axvline(0, color=GRAY, linestyle="--", alpha=0.6)
        ax.axvline(rets.median(), color=RED, linestyle="-", linewidth=2,
                   label=f"median {rets.median():.2f}%")
        ax.axvline(rets.mean(), color="black", linestyle=":", linewidth=2,
                   label=f"mean {rets.mean():.2f}%")
        ax.set_xlabel("Return on $10k notional (%)")
        ax.set_ylabel("Window count")
        ax.set_title(f"{label}\n{len(complete)} complete, best {rets.max():.1f}%, worst {rets.min():.1f}%")
        ax.legend(loc="upper right")
    fig.suptitle("Experiment grid — return distribution per window", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, "01_experiment_returns_distribution")


def graph_experiment_winrate_vs_return(strict_df: pd.DataFrame, synth_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, df, label, color in (
        (axes[0], strict_df, "Strict priced", BLUE),
        (axes[1], synth_df, "Exploratory synthetic", ORANGE),
    ):
        if df.empty or "return_on_10k" not in df.columns:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue
        complete = df[(df["status"] == "complete") & (df["trade_count"] > 0)].copy()
        if complete.empty:
            ax.text(0.5, 0.5, "no trades", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue
        complete["return_pct"] = complete["return_on_10k"].astype(float) * 100.0
        complete["winrate_pct"] = complete["win_rate"].astype(float) * 100.0
        scatter = ax.scatter(
            complete["winrate_pct"],
            complete["return_pct"],
            c=complete["trade_count"],
            cmap="viridis",
            s=70,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.6,
        )
        ax.axhline(0, color=GRAY, linestyle="--", alpha=0.6)
        ax.axvline(50, color=GRAY, linestyle="--", alpha=0.6, label="50% win rate")
        ax.set_xlabel("Win rate (%)")
        ax.set_ylabel("Return on $10k (%)")
        ax.set_title(label)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("trade count", rotation=270, labelpad=18)
        ax.legend(loc="upper left")
    fig.suptitle("Win rate vs realised return (each dot = one window)", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, "02_winrate_vs_return")


def graph_experiment_heatmap(strict_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))
    if strict_df.empty:
        ax.text(0.5, 0.5, "No strict data", ha="center", va="center", transform=ax.transAxes)
    else:
        complete = strict_df[strict_df["status"] == "complete"].copy()
        if complete.empty:
            ax.text(0.5, 0.5, "No complete windows", ha="center", va="center", transform=ax.transAxes)
        else:
            complete["return_pct"] = complete["return_on_10k"].astype(float) * 100.0
            pivot = complete.pivot_table(
                values="return_pct",
                index="train_days",
                columns="test_days",
                aggfunc="mean",
            )
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="RdYlGn",
                center=0,
                ax=ax,
                cbar_kws={"label": "mean return (%)"},
                linewidths=0.5,
            )
            ax.set_xlabel("Test window (days)")
            ax.set_ylabel("Train window (days)")
            ax.set_title("Strict priced grid — mean return by (train_days × test_days)")
    fig.tight_layout()
    return _save(fig, "03_grid_heatmap_strict")


def graph_experiment_returns_over_time(strict_df: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(14, 5))
    if strict_df.empty:
        ax.text(0.5, 0.5, "No strict data", ha="center", va="center", transform=ax.transAxes)
    else:
        complete = strict_df[strict_df["status"] == "complete"].copy()
        if complete.empty:
            ax.text(0.5, 0.5, "No complete windows", ha="center", va="center", transform=ax.transAxes)
        else:
            complete["test_start"] = pd.to_datetime(complete["test_start"], errors="coerce")
            complete = complete.dropna(subset=["test_start"]).sort_values("test_start")
            complete["return_pct"] = complete["return_on_10k"].astype(float) * 100.0
            colors = [GREEN if v >= 0 else RED for v in complete["return_pct"]]
            ax.bar(complete["test_start"], complete["return_pct"], width=20, color=colors, alpha=0.85)
            ax.axhline(0, color="black", linewidth=0.6)
            ax.axhline(complete["return_pct"].median(), color=BLUE, linestyle="--",
                       label=f"median {complete['return_pct'].median():.2f}%")
            ax.set_xlabel("Test-window start date")
            ax.set_ylabel("Return on $10k (%)")
            ax.set_title("Strict priced grid — window returns over time")
            ax.legend()
            fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, "04_returns_over_time")


def graph_experiment_top_bottom(strict_df: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    if strict_df.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
    else:
        complete = strict_df[strict_df["status"] == "complete"].copy()
        complete["return_pct"] = complete["return_on_10k"].astype(float) * 100.0
        sorted_df = complete.sort_values("return_pct")
        top = sorted_df.tail(10)
        bottom = sorted_df.head(10)
        axes[0].barh(top["name"], top["return_pct"], color=GREEN, alpha=0.85)
        axes[0].set_xlabel("Return on $10k (%)")
        axes[0].set_title(f"Top 10 windows (best {top['return_pct'].max():.1f}%)")
        axes[1].barh(bottom["name"], bottom["return_pct"], color=RED, alpha=0.85)
        axes[1].set_xlabel("Return on $10k (%)")
        axes[1].set_title(f"Bottom 10 windows (worst {bottom['return_pct'].min():.1f}%)")
    fig.suptitle("Strict priced — best vs worst windows (cherry-picking warning)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    return _save(fig, "05_top_bottom_windows")


def graph_smoke_run_equity_curve(conn: sqlite3.Connection, run_id: str) -> Path:
    trades = pd.read_sql_query(
        """
        SELECT
            COALESCE(exit_at_utc, entry_at_utc) AS at_utc,
            sleeve, underlying_ticker, realized_pnl_usd
        FROM trades
        WHERE run_id = ? AND realized_pnl_usd IS NOT NULL
        ORDER BY at_utc
        """,
        conn, params=(run_id,),
    )
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [2, 1]})
    if trades.empty:
        for ax in axes:
            ax.text(0.5, 0.5, f"No trades for {run_id}", ha="center", va="center",
                    transform=ax.transAxes)
    else:
        trades["at_utc"] = pd.to_datetime(trades["at_utc"], utc=True)
        trades = trades.sort_values("at_utc").reset_index(drop=True)
        trades["cumpnl"] = trades["realized_pnl_usd"].cumsum()
        trades["nav"] = 10000.0 + trades["cumpnl"]
        axes[0].plot(trades["at_utc"], trades["nav"], color=BLUE, linewidth=2, marker="o",
                     markersize=4, label="NAV (USD)")
        axes[0].axhline(10000, color=GRAY, linestyle="--", alpha=0.6, label="Initial $10k")
        peak = trades["nav"].iloc[-1]
        axes[0].axhline(peak, color=GREEN, linestyle=":", alpha=0.7,
                        label=f"Final ${peak:,.0f} ({(peak/10000-1)*100:+.2f}%)")
        axes[0].set_ylabel("Equity ($)")
        axes[0].set_title(f"Smoke run — equity curve, {len(trades)} closed trades")
        axes[0].legend(loc="upper left")
        axes[0].grid(True, alpha=0.3)

        colors = [GREEN if p >= 0 else RED for p in trades["realized_pnl_usd"]]
        axes[1].bar(trades["at_utc"], trades["realized_pnl_usd"], width=2, color=colors, alpha=0.85)
        axes[1].axhline(0, color="black", linewidth=0.6)
        axes[1].set_ylabel("Trade P&L ($)")
        axes[1].set_xlabel("Trade close date")
        axes[1].set_title(f"Per-trade P&L  (mean ${trades['realized_pnl_usd'].mean():.1f}, "
                          f"std ${trades['realized_pnl_usd'].std():.1f})")
        axes[1].grid(True, alpha=0.3)
        fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, "06_smoke_equity_curve_and_trade_pnl")


def graph_smoke_run_by_sleeve(conn: sqlite3.Connection, run_id: str) -> Path:
    rows = pd.read_sql_query(
        """
        SELECT sleeve, underlying_ticker, realized_pnl_usd
        FROM trades
        WHERE run_id = ? AND realized_pnl_usd IS NOT NULL
        """,
        conn, params=(run_id,),
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if rows.empty:
        for ax in axes:
            ax.text(0.5, 0.5, "No trades", ha="center", va="center", transform=ax.transAxes)
    else:
        sleeve_summary = rows.groupby("sleeve")["realized_pnl_usd"].agg(["count", "sum", "mean"])
        axes[0].bar(sleeve_summary.index, sleeve_summary["sum"],
                    color=[GREEN if v >= 0 else RED for v in sleeve_summary["sum"]], alpha=0.85)
        axes[0].set_title("Total P&L by sleeve")
        axes[0].set_ylabel("Total realised P&L ($)")
        for i, (sleeve, val) in enumerate(zip(sleeve_summary.index, sleeve_summary["sum"])):
            n = sleeve_summary.loc[sleeve, "count"]
            avg = sleeve_summary.loc[sleeve, "mean"]
            axes[0].text(i, val, f"\n  n={n}  avg=${avg:.1f}",
                         ha="center", va="bottom" if val >= 0 else "top", fontsize=10)
        axes[0].axhline(0, color="black", linewidth=0.6)

        ticker_summary = (
            rows.groupby("underlying_ticker")["realized_pnl_usd"].sum().sort_values()
        )
        colors = [GREEN if v >= 0 else RED for v in ticker_summary]
        axes[1].barh(ticker_summary.index, ticker_summary.values, color=colors, alpha=0.85)
        axes[1].axvline(0, color="black", linewidth=0.6)
        axes[1].set_xlabel("Total realised P&L ($)")
        axes[1].set_title("P&L by underlying ticker")
    fig.suptitle("Smoke run — P&L breakdown", fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, "07_smoke_pnl_by_sleeve_and_ticker")


def graph_all_experiments_and_strategies(
    conn: sqlite3.Connection,
    run_id: str,
    all_experiments: pd.DataFrame,
) -> Path:
    """One-page view: experiment windows, latest evidence, and sleeve P&L."""
    trades = pd.read_sql_query(
        """
        SELECT
            COALESCE(exit_at_utc, entry_at_utc) AS at_utc,
            sleeve,
            underlying_ticker,
            realized_pnl_usd
        FROM trades
        WHERE run_id = ? AND realized_pnl_usd IS NOT NULL
        ORDER BY at_utc
        """,
        conn,
        params=(run_id,),
    )

    fig = plt.figure(figsize=(17, 11))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0], hspace=0.32, wspace=0.24)
    ax_timeline = fig.add_subplot(gs[0, :])
    ax_dist = fig.add_subplot(gs[1, 0])
    ax_sleeves = fig.add_subplot(gs[1, 1])

    complete = all_experiments[
        (all_experiments.get("status") == "complete")
        & all_experiments["return_pct"].notna()
    ].copy() if not all_experiments.empty else pd.DataFrame()

    if complete.empty:
        ax_timeline.text(0.5, 0.5, "No experiment return rows", ha="center", va="center",
                         transform=ax_timeline.transAxes)
        ax_dist.text(0.5, 0.5, "No experiment return rows", ha="center", va="center",
                     transform=ax_dist.transAxes)
    else:
        sets = list(dict.fromkeys(complete["experiment_set"].astype(str).tolist()))
        palette = [BLUE, ORANGE, PURPLE, GREEN, RED, GRAY, DARK]
        for idx, name in enumerate(sets):
            sub = complete[complete["experiment_set"] == name].dropna(subset=["test_start_dt"])
            if sub.empty:
                continue
            size = np.clip(sub["trade_count"].fillna(4).astype(float), 2, 50) * 3.0
            ax_timeline.scatter(
                sub["test_start_dt"],
                sub["return_pct"],
                s=size,
                alpha=0.55,
                color=palette[idx % len(palette)],
                edgecolors="white",
                linewidths=0.4,
                label=f"{name} ({len(sub)})",
            )

        ax_timeline.axhline(0, color="black", linewidth=0.8, alpha=0.7)
        ax_timeline.axhline(complete["return_pct"].median(), color=GRAY, linestyle="--",
                            linewidth=1.4, label=f"all-window median {complete['return_pct'].median():.2f}%")
        ax_timeline.set_title("All experiment windows on one axis")
        ax_timeline.set_xlabel("Test-window start date")
        ax_timeline.set_ylabel("Return on $10k (%)")
        ax_timeline.legend(loc="upper left", fontsize=8, ncols=2)
        ax_timeline.grid(True, alpha=0.25)

        box_data = [
            complete[complete["experiment_set"] == name]["return_pct"].dropna()
            for name in sets
            if not complete[complete["experiment_set"] == name]["return_pct"].dropna().empty
        ]
        box_labels = [
            name.replace("_20260510T123538Z", "").replace("_", "\n")
            for name in sets
            if not complete[complete["experiment_set"] == name]["return_pct"].dropna().empty
        ]
        ax_dist.boxplot(box_data, tick_labels=box_labels, showfliers=False)
        for i, values in enumerate(box_data, start=1):
            jitter = np.random.default_rng(42 + i).normal(0, 0.04, size=len(values))
            ax_dist.scatter(np.full(len(values), i) + jitter, values, s=18, alpha=0.35, color=BLUE)
        ax_dist.axhline(0, color="black", linewidth=0.8, alpha=0.7)
        ax_dist.set_ylabel("Return on $10k (%)")
        ax_dist.set_title("Return distribution by experiment set")
        ax_dist.tick_params(axis="x", rotation=0, labelsize=8)
        ax_dist.grid(True, axis="y", alpha=0.25)

    if trades.empty:
        ax_sleeves.text(0.5, 0.5, f"No trades for {run_id}", ha="center", va="center",
                        transform=ax_sleeves.transAxes)
    else:
        trades["at_utc"] = pd.to_datetime(trades["at_utc"], utc=True)
        trades = trades.sort_values("at_utc").reset_index(drop=True)
        sleeve_totals = trades.groupby("sleeve")["realized_pnl_usd"].sum().sort_values(ascending=False)
        total_pnl = float(trades["realized_pnl_usd"].sum())
        total_return = total_pnl / 10000.0 * 100.0
        final_date = trades["at_utc"].max()
        if complete.empty or pd.isna(final_date):
            star_x = pd.Timestamp.utcnow()
        else:
            star_x = final_date
        ax_timeline.scatter(
            [star_x],
            [total_return],
            marker="*",
            s=420,
            color="black",
            edgecolors=GREEN if total_return >= 0 else RED,
            linewidths=1.6,
            label=f"latest evidence {total_return:+.2f}%",
            zorder=6,
        )
        ax_timeline.annotate(
            f"latest evidence\n{len(trades)} trades\n{total_return:+.2f}%",
            xy=(star_x, total_return),
            xytext=(18, 18),
            textcoords="offset points",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": DARK, "lw": 1.0},
        )

        bars = list(sleeve_totals.index) + ["total"]
        values = list(sleeve_totals.values) + [total_pnl]
        colors = [GREEN if v >= 0 else RED for v in values]
        ax_sleeves.bar(bars, values, color=colors, alpha=0.85, edgecolor="white")
        ax_sleeves.axhline(0, color="black", linewidth=0.8)
        ax_sleeves.set_ylabel("Realised P&L ($)")
        ax_sleeves.set_title(f"Latest evidence run by strategy sleeve\n{run_id}")
        for i, (label, value) in enumerate(zip(bars, values)):
            count = len(trades) if label == "total" else int((trades["sleeve"] == label).sum())
            ax_sleeves.text(
                i,
                value,
                f"${value:,.0f}\nn={count}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=9,
            )
        ax_sleeves.grid(True, axis="y", alpha=0.25)

    fig.suptitle(
        "One graph: all experiment windows + latest core/satellite evidence",
        fontsize=16,
        y=0.98,
    )
    return _save(fig, "09_all_experiments_and_strategies")


def _equity_curve_from_trade_pnl(
    trades: pd.DataFrame,
    *,
    dates: pd.DatetimeIndex,
    initial_nav: float,
    pnl_scale: float,
    sleeve: str | None = None,
) -> pd.Series:
    """Daily step equity curve from realised trade P&L."""
    if sleeve is not None:
        trades = trades[trades["sleeve"] == sleeve]
    if trades.empty:
        return pd.Series(initial_nav, index=dates)
    daily_pnl = trades.groupby("date")["realized_pnl_usd"].sum().sort_index()
    daily_pnl.index = pd.to_datetime(daily_pnl.index)
    daily_pnl = daily_pnl.reindex(dates, fill_value=0.0)
    return initial_nav + daily_pnl.cumsum() * pnl_scale


def _spy_benchmark_curve(
    start: pd.Timestamp,
    end: pd.Timestamp,
    *,
    initial_nav: float,
) -> pd.Series | None:
    spy_path = REPO_ROOT / "data" / "cache" / "year2" / "spy_daily.parquet"
    if not spy_path.exists():
        return None
    try:
        spy = pd.read_parquet(spy_path)
    except (OSError, ValueError):
        return None
    if spy.empty or "close" not in spy.columns:
        return None
    spy = spy.copy()
    spy.index = pd.to_datetime(spy.index).tz_localize(None).normalize()
    spy = spy[(spy.index >= start.normalize()) & (spy.index <= end.normalize())]
    if spy.empty:
        return None
    close = pd.to_numeric(spy["close"], errors="coerce").dropna()
    if close.empty or close.iloc[0] == 0:
        return None
    return initial_nav * (close / close.iloc[0])


def _spy_start_date() -> pd.Timestamp | None:
    spy_path = REPO_ROOT / "data" / "cache" / "year2" / "spy_daily.parquet"
    if not spy_path.exists():
        return None
    try:
        spy = pd.read_parquet(spy_path)
    except (OSError, ValueError):
        return None
    if spy.empty:
        return None
    return pd.to_datetime(spy.index).tz_localize(None).normalize().min()


def graph_reference_style_equity_curves(conn: sqlite3.Connection, run_id: str) -> Path:
    """Clean portfolio-line comparison, styled like the user's reference chart."""
    trades = pd.read_sql_query(
        """
        SELECT
            COALESCE(exit_at_utc, entry_at_utc) AS at_utc,
            sleeve,
            realized_pnl_usd
        FROM trades
        WHERE run_id = ? AND realized_pnl_usd IS NOT NULL
        ORDER BY at_utc
        """,
        conn,
        params=(run_id,),
    )
    fig, ax = plt.subplots(figsize=(9.2, 5.1))
    initial_nav = 10000.0
    pnl_scale = 1.0

    if trades.empty:
        ax.text(0.5, 0.5, f"No trades for {run_id}", ha="center", va="center",
                transform=ax.transAxes)
        ax.set_title("paper_v1 evidence")
        return _save(fig, "10_reference_style_equity_curves")

    trades["at_utc"] = pd.to_datetime(trades["at_utc"], utc=True, errors="coerce")
    trades = trades.dropna(subset=["at_utc", "realized_pnl_usd"]).copy()
    trades["date"] = trades["at_utc"].dt.tz_convert(None).dt.normalize()
    spy_start = _spy_start_date()
    trade_start = trades["date"].min()
    start = min(trade_start, spy_start) if spy_start is not None else trade_start
    end = trades["date"].max()
    dates = pd.bdate_range(start, end)

    curves = {
        "paper_v1_total": _equity_curve_from_trade_pnl(
            trades, dates=dates, initial_nav=initial_nav, pnl_scale=pnl_scale
        ),
        "satellite_only": _equity_curve_from_trade_pnl(
            trades, dates=dates, initial_nav=initial_nav, pnl_scale=pnl_scale, sleeve="satellite"
        ),
        "spy_core_only": _equity_curve_from_trade_pnl(
            trades, dates=dates, initial_nav=initial_nav, pnl_scale=pnl_scale, sleeve="core"
        ),
    }
    colors = {
        "paper_v1_total": GREEN,
        "satellite_only": BLUE,
        "spy_core_only": ORANGE,
    }
    for name, curve in curves.items():
        ret = (curve.iloc[-1] / initial_nav - 1.0) * 100.0
        ax.plot(
            curve.index,
            curve.values,
            linewidth=2.0 if name == "paper_v1_total" else 1.7,
            color=colors[name],
            label=f"{name} ({ret:+.2f}%)",
        )

    spy_curve = _spy_benchmark_curve(start, end, initial_nav=initial_nav)
    if spy_curve is not None:
        spy_ret = (spy_curve.iloc[-1] / initial_nav - 1.0) * 100.0
        ax.plot(
            spy_curve.index,
            spy_curve.values,
            color=GRAY,
            linestyle="--",
            linewidth=1.6,
            label=f"SPY ({spy_ret:+.2f}%)",
        )

    ax.axhline(initial_nav, color=GRAY, linewidth=0.8, alpha=0.45)
    ax.set_title(
        f"Paper v1 Evidence - $10k Portfolio\n"
        f"{start.date()} to {end.date()} | entries and exits marked in interactive view"
    )
    ax.set_ylabel("Portfolio value ($)")
    ax.set_xlabel("")
    ax.legend(loc="upper left", fontsize=9, frameon=True)
    ax.grid(True, alpha=0.28)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    return _save(fig, "10_reference_style_equity_curves")


def _curve_value_at(curve: pd.Series, when: pd.Timestamp) -> float:
    day = pd.Timestamp(when).normalize()
    if day in curve.index:
        return float(curve.loc[day])
    earlier = curve[curve.index <= day]
    if earlier.empty:
        return float(curve.iloc[0])
    return float(earlier.iloc[-1])


def graph_interactive_equity_curves_10k(conn: sqlite3.Connection, run_id: str) -> Path:
    """Interactive HTML: $10k equity curves with entry and exit markers."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    trades = pd.read_sql_query(
        """
        SELECT
            trade_id,
            sleeve,
            strategy_type,
            underlying_ticker,
            entry_at_utc,
            exit_at_utc,
            realized_pnl_usd
        FROM trades
        WHERE run_id = ? AND realized_pnl_usd IS NOT NULL
        ORDER BY COALESCE(exit_at_utc, entry_at_utc)
        """,
        conn,
        params=(run_id,),
    )
    if trades.empty:
        fig = go.Figure()
        fig.update_layout(title=f"No closed trades for {run_id}")
        return _save_html(fig, "11_equity_curves_10k_entries_exits")

    initial_nav = 10000.0
    trades["entry_at_utc"] = pd.to_datetime(trades["entry_at_utc"], utc=True, errors="coerce")
    trades["exit_at_utc"] = pd.to_datetime(trades["exit_at_utc"], utc=True, errors="coerce")
    trades = trades.dropna(subset=["entry_at_utc", "exit_at_utc", "realized_pnl_usd"]).copy()
    trades["date"] = trades["exit_at_utc"].dt.tz_convert(None).dt.normalize()
    trades["entry_date"] = trades["entry_at_utc"].dt.tz_convert(None).dt.normalize()

    spy_start = _spy_start_date()
    start = min(trades["entry_date"].min(), spy_start) if spy_start is not None else trades["entry_date"].min()
    end = trades["date"].max()
    dates = pd.bdate_range(start, end)
    daily = (
        trades.groupby(["date", "sleeve"])["realized_pnl_usd"]
        .sum()
        .unstack(fill_value=0.0)
        .reindex(dates, fill_value=0.0)
    )
    for sleeve in ("core", "satellite"):
        if sleeve not in daily.columns:
            daily[sleeve] = 0.0
    daily["total"] = daily["core"] + daily["satellite"]
    total_curve = initial_nav + daily["total"].cumsum()
    total_pnl_curve = daily["total"].cumsum()
    core_pnl_curve = daily["core"].cumsum()
    satellite_pnl_curve = daily["satellite"].cumsum()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.56, 0.24, 0.20],
        subplot_titles=(
            "Actual portfolio NAV",
            "P&L contribution by sleeve",
            "Individual closed trades",
        ),
    )

    final_nav = float(total_curve.iloc[-1])
    final_return = (final_nav / initial_nav - 1.0) * 100.0
    core_pnl = float(daily["core"].sum())
    satellite_pnl = float(daily["satellite"].sum())
    total_pnl = float(daily["total"].sum())

    fig.add_trace(
        go.Scatter(
            x=total_curve.index,
            y=total_curve.values,
            mode="lines",
            name=f"Paper v1 NAV ({final_return:+.2f}%)",
            line={"color": GREEN, "width": 4},
            hovertemplate="%{x|%Y-%m-%d}<br>NAV %{y:$,.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    spy_curve = _spy_benchmark_curve(start, end, initial_nav=initial_nav)
    if spy_curve is not None:
        spy_ret = (spy_curve.iloc[-1] / initial_nav - 1.0) * 100.0
        fig.add_trace(
            go.Scatter(
                x=spy_curve.index,
                y=spy_curve.values,
                mode="lines",
                name=f"SPY benchmark ({spy_ret:+.2f}%)",
                line={"color": GRAY, "width": 2, "dash": "dash"},
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:$,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    entry_rows = trades.copy()
    entry_rows["marker_nav"] = entry_rows["entry_date"].apply(lambda d: _curve_value_at(total_curve, d))
    entry_rows["marker_color"] = entry_rows["sleeve"].map({"core": ORANGE, "satellite": BLUE}).fillna(PURPLE)
    exit_rows = trades.copy()
    exit_rows["marker_nav"] = exit_rows["date"].apply(lambda d: _curve_value_at(total_curve, d))
    exit_rows["marker_color"] = np.where(exit_rows["realized_pnl_usd"].astype(float) >= 0, GREEN, RED)

    fig.add_trace(
        go.Scatter(
            x=entry_rows["entry_date"],
            y=entry_rows["marker_nav"],
            mode="markers",
            name="BUY / open",
            marker={
                "symbol": "triangle-up",
                "size": 13,
                "color": entry_rows["marker_color"],
                "line": {"color": DARK, "width": 0.7},
            },
            customdata=np.stack(
                [
                    entry_rows["trade_id"],
                    entry_rows["sleeve"],
                    entry_rows["underlying_ticker"],
                    entry_rows["strategy_type"],
                    entry_rows["entry_at_utc"].dt.strftime("%Y-%m-%d %H:%M UTC"),
                    entry_rows["exit_at_utc"].dt.strftime("%Y-%m-%d %H:%M UTC"),
                    entry_rows["realized_pnl_usd"].map(lambda x: f"${float(x):,.2f}"),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "BUY / OPEN<br>"
                "trade %{customdata[0]} | %{customdata[1]} | %{customdata[2]}<br>"
                "%{customdata[3]}<br>"
                "entry: %{customdata[4]}<br>"
                "exit: %{customdata[5]}<br>"
                "realized P&L: %{customdata[6]}<br>"
                "portfolio line: %{y:$,.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=exit_rows["date"],
            y=exit_rows["marker_nav"],
            mode="markers",
            name="SELL / close",
            marker={
                "symbol": "triangle-down",
                "size": 14,
                "color": exit_rows["marker_color"],
                "line": {"color": DARK, "width": 0.7},
            },
            customdata=np.stack(
                [
                    exit_rows["trade_id"],
                    exit_rows["sleeve"],
                    exit_rows["underlying_ticker"],
                    exit_rows["strategy_type"],
                    exit_rows["entry_at_utc"].dt.strftime("%Y-%m-%d %H:%M UTC"),
                    exit_rows["exit_at_utc"].dt.strftime("%Y-%m-%d %H:%M UTC"),
                    exit_rows["realized_pnl_usd"].map(lambda x: f"${float(x):,.2f}"),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "SELL / CLOSE<br>"
                "trade %{customdata[0]} | %{customdata[1]} | %{customdata[2]}<br>"
                "%{customdata[3]}<br>"
                "entry: %{customdata[4]}<br>"
                "exit: %{customdata[5]}<br>"
                "realized P&L: %{customdata[6]}<br>"
                "portfolio line: %{y:$,.2f}<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    for name, curve, color in (
        ("Total P&L", total_pnl_curve, DARK),
        ("Satellite contribution", satellite_pnl_curve, BLUE),
        ("SPY core contribution", core_pnl_curve, ORANGE),
    ):
        fig.add_trace(
            go.Scatter(
                x=curve.index,
                y=curve.values,
                mode="lines",
                name=name,
                line={"color": color, "width": 2.5},
                hovertemplate="%{x|%Y-%m-%d}<br>%{y:$,.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    bar_colors = np.where(trades["realized_pnl_usd"].astype(float) >= 0, GREEN, RED)
    fig.add_trace(
        go.Bar(
            x=trades["date"],
            y=trades["realized_pnl_usd"],
            name="Trade P&L",
            marker={"color": bar_colors},
            customdata=np.stack(
                [
                    trades["trade_id"],
                    trades["sleeve"],
                    trades["underlying_ticker"],
                    trades["strategy_type"],
                    trades["entry_at_utc"].dt.strftime("%Y-%m-%d %H:%M UTC"),
                    trades["exit_at_utc"].dt.strftime("%Y-%m-%d %H:%M UTC"),
                ],
                axis=-1,
            ),
            hovertemplate=(
                "TRADE P&L<br>"
                "trade %{customdata[0]} | %{customdata[1]} | %{customdata[2]}<br>"
                "%{customdata[3]}<br>"
                "entry: %{customdata[4]}<br>"
                "exit: %{customdata[5]}<br>"
                "P&L: %{y:$,.2f}<extra></extra>"
            ),
        ),
        row=3,
        col=1,
    )

    start_label = pd.Timestamp(start).date().isoformat()
    end_label = pd.Timestamp(end).date().isoformat()
    fig.update_layout(
        title={
            "text": (
                "Paper v1 Evidence - $10k Paper Portfolio<br>"
                f"<sup>{start_label} to {end_label} | "
                f"$10,000 + core ${core_pnl:,.2f} + satellite ${satellite_pnl:,.2f} "
                f"= final ${final_nav:,.2f} ({final_return:+.2f}%) | "
                f"{len(trades)} closed trades | ▲ buy/open, ▼ sell/close</sup>"
            ),
            "x": 0.02,
            "xanchor": "left",
            "font": {"size": 26},
        },
        template="plotly_white",
        width=1800,
        height=1180,
        hovermode="closest",
        legend={"orientation": "h", "y": 1.015, "x": 0.01, "font": {"size": 13}},
        margin={"l": 85, "r": 45, "t": 150, "b": 95},
        bargap=0.15,
    )
    fig.update_yaxes(title_text="Portfolio value ($)", tickprefix="$", gridcolor="#e8edf3", row=1, col=1)
    fig.update_yaxes(title_text="Cumulative P&L ($)", tickprefix="$", gridcolor="#e8edf3", row=2, col=1)
    fig.update_yaxes(title_text="Trade P&L ($)", tickprefix="$", gridcolor="#e8edf3", row=3, col=1)
    fig.update_xaxes(
        rangeslider={"visible": True, "thickness": 0.06},
        rangeselector={
            "buttons": [
                {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"},
                {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                {"step": "all", "label": "all"},
            ]
        },
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.add_annotation(
        text=f"Run ID: {run_id}",
        xref="paper",
        yref="paper",
        x=1.0,
        y=1.12,
        showarrow=False,
        xanchor="right",
        font={"size": 11, "color": GRAY},
    )
    fig.add_annotation(
        text=f"Profit math: ${total_pnl:,.2f} realised = ${core_pnl:,.2f} core + ${satellite_pnl:,.2f} satellite",
        xref="paper",
        yref="paper",
        x=0.01,
        y=0.97,
        showarrow=False,
        xanchor="left",
        font={"size": 13, "color": DARK},
        bgcolor="rgba(255,255,255,0.82)",
        bordercolor="#d5dde5",
        borderwidth=1,
    )
    fig.add_hline(y=initial_nav, line_width=1, line_dash="dot", line_color=GRAY, row=1, col=1)
    fig.add_hline(y=0, line_width=1, line_color=GRAY, row=2, col=1)
    fig.add_hline(y=0, line_width=1, line_color=GRAY, row=3, col=1)
    return _save_html(fig, "11_equity_curves_10k_entries_exits")


def graph_blockers_summary() -> Path:
    """Read live readiness state and visualise the data wall."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from check_paper_v1_readiness import check_readiness  # type: ignore
    r = check_readiness(REPO_ROOT / "data" / "cache", require_spy_core=True)
    blockers = r.get("blockers", [])
    if not blockers:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "🟢 No blockers", ha="center", va="center",
                fontsize=20, transform=ax.transAxes)
        return _save(fig, "08_readiness_blockers")

    df = pd.DataFrame(blockers)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    by_code = df["code"].value_counts()
    axes[0].barh(by_code.index, by_code.values, color=RED, alpha=0.85)
    axes[0].set_title(f"Readiness blockers by code (total {len(df)})")
    axes[0].set_xlabel("Count")
    for i, (code, cnt) in enumerate(zip(by_code.index, by_code.values)):
        axes[0].text(cnt, i, f" {cnt}", va="center")
    by_ticker = df["ticker"].value_counts().head(15)
    axes[1].barh(by_ticker.index[::-1], by_ticker.values[::-1], color=ORANGE, alpha=0.85)
    axes[1].set_title("Top 15 tickers by blocker count")
    axes[1].set_xlabel("Count")
    fig.suptitle("Polygon Free data wall — what's still blocking the official run",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    return _save(fig, "08_readiness_blockers")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", default="strict_priced_stats_20260510T123538Z")
    parser.add_argument("--synth", default="explore_synth_spy_stats_20260510T123538Z")
    parser.add_argument("--smoke-run-id", default="paper_v1_unofficial_smoke_20260510T011718Z")
    args = parser.parse_args()

    print(f"Output dir: {GRAPH_DIR}")
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)

    strict_csv = REPO_ROOT / "experiments" / "runs" / args.strict / "summary.csv"
    synth_csv = REPO_ROOT / "experiments" / "runs" / args.synth / "summary.csv"
    strict_df = pd.read_csv(strict_csv) if strict_csv.exists() else pd.DataFrame()
    synth_df = pd.read_csv(synth_csv) if synth_csv.exists() else pd.DataFrame()
    all_experiments = load_all_experiment_returns()
    print(f"strict rows: {len(strict_df)}, synth rows: {len(synth_df)}")
    print(f"all experiment return rows: {len(all_experiments)}")

    paths = []
    print("\nExperiment grid graphs:")
    paths.append(graph_experiment_returns_distribution(strict_df, synth_df))
    paths.append(graph_experiment_winrate_vs_return(strict_df, synth_df))
    paths.append(graph_experiment_heatmap(strict_df))
    paths.append(graph_experiment_returns_over_time(strict_df))
    paths.append(graph_experiment_top_bottom(strict_df))

    print("\nSmoke run simulation graphs:")
    db_path = REPO_ROOT / "data" / "trading_system.db"
    conn = sqlite3.connect(db_path)
    try:
        paths.append(graph_smoke_run_equity_curve(conn, args.smoke_run_id))
        paths.append(graph_smoke_run_by_sleeve(conn, args.smoke_run_id))
        paths.append(graph_all_experiments_and_strategies(conn, args.smoke_run_id, all_experiments))
        paths.append(graph_reference_style_equity_curves(conn, args.smoke_run_id))
        paths.append(graph_interactive_equity_curves_10k(conn, args.smoke_run_id))
    finally:
        conn.close()

    print("\nReadiness blocker graphs:")
    paths.append(graph_blockers_summary())

    print("\nDone. Files:")
    for p in paths:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
