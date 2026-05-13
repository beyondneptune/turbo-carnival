"""Local simulation for scripts/alpaca_max_hft_host.py.

Uses cached Alpaca 1-minute bars from a prior `valk_hft_demo` run. No network
and no orders. This answers: if Aegis/Falcon/Valk were converted to the max-HFT
host, how many trades/day and what P&L/DD would the exact lane modes produce?
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.alpaca_max_hft_host import DEFAULT_SYMBOLS, signal_from_history  # noqa: E402


RUNS = REPO_ROOT / "experiments" / "runs"


@dataclass(frozen=True)
class SimLane:
    name: str
    mode: str


LANES = [
    SimLane("Aegis", "mean_reversion"),
    SimLane("Falcon", "breakout"),
    SimLane("Valk", "turbo_hybrid"),
]


def latest_cached_bars() -> Path:
    paths = sorted(RUNS.glob("valk_hft_demo_*/minute_bars.parquet"))
    if not paths:
        raise FileNotFoundError("No cached minute_bars.parquet found under experiments/runs/valk_hft_demo_*")
    return paths[-1]


def update_history(frame: pd.DataFrame, history: dict[str, deque[float]]) -> None:
    for row in frame.itertuples(index=False):
        history[str(row.symbol)].append(float(row.close))


def exit_position(pos: dict[str, Any], row: pd.Series, *, tp_bps: float, sl_bps: float, max_hold_bars: int, slip_bps: float) -> tuple[float | None, str | None]:
    direction = int(pos["direction"])
    entry = float(pos["entry_price"])
    held = int(pos["held_bars"])
    if direction > 0:
        stop = entry * (1.0 - sl_bps / 10000.0)
        take = entry * (1.0 + tp_bps / 10000.0)
        if float(row["low"]) <= stop:
            return stop * (1.0 - slip_bps / 10000.0), "stop_loss"
        if float(row["high"]) >= take:
            return take * (1.0 - slip_bps / 10000.0), "take_profit"
        if held >= max_hold_bars:
            return float(row["close"]) * (1.0 - slip_bps / 10000.0), "time_exit"
    else:
        stop = entry * (1.0 + sl_bps / 10000.0)
        take = entry * (1.0 - tp_bps / 10000.0)
        if float(row["high"]) >= stop:
            return stop * (1.0 + slip_bps / 10000.0), "stop_loss"
        if float(row["low"]) <= take:
            return take * (1.0 + slip_bps / 10000.0), "take_profit"
        if held >= max_hold_bars:
            return float(row["close"]) * (1.0 + slip_bps / 10000.0), "time_exit"
    return None, None


def simulate_lane(
    bars: pd.DataFrame,
    lane: SimLane,
    *,
    capital: float,
    notional_pct: float,
    min_signal_bps: float,
    take_profit_bps: float,
    stop_loss_bps: float,
    max_hold_sec: float,
    slippage_bps: float,
    max_trades_per_day: int,
) -> dict[str, Any]:
    history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=80))
    nav = float(capital)
    position: dict[str, Any] | None = None
    equity_rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    trades_today: dict[str, int] = defaultdict(int)
    max_hold_bars = max(1, int(math.ceil(max_hold_sec / 60.0)))

    for ts, frame in bars.groupby("t", sort=True):
        update_history(frame, history)
        by_symbol = {str(r["symbol"]): r for _, r in frame.iterrows()}

        if position is not None:
            pos_row = by_symbol.get(position["symbol"])
            if pos_row is not None:
                position["held_bars"] += 1
                exit_px, reason = exit_position(
                    position,
                    pos_row,
                    tp_bps=take_profit_bps,
                    sl_bps=stop_loss_bps,
                    max_hold_bars=max_hold_bars,
                    slip_bps=slippage_bps,
                )
                if exit_px is not None:
                    qty = int(position["qty"])
                    direction = int(position["direction"])
                    pnl = (exit_px - float(position["entry_price"])) * qty * direction
                    nav += pnl
                    trades.append(
                        {
                            "lane": lane.name,
                            "mode": lane.mode,
                            "symbol": position["symbol"],
                            "side": "long" if direction > 0 else "short",
                            "entry_t": position["entry_t"].isoformat(),
                            "exit_t": ts.isoformat(),
                            "entry_price": position["entry_price"],
                            "exit_price": exit_px,
                            "qty": qty,
                            "pnl": pnl,
                            "pnl_bps": pnl / max(float(position["entry_price"]) * qty, 1.0) * 10000.0,
                            "exit_reason": reason,
                            "signal_bps": position["signal_bps"],
                        }
                    )
                    position = None

        if position is None:
            date_key = pd.Timestamp(ts).date().isoformat()
            if trades_today[date_key] < max_trades_per_day:
                best: tuple[str, int, float, str] | None = None
                for sym, row in by_symbol.items():
                    if sym not in DEFAULT_SYMBOLS:
                        continue
                    score, why = signal_from_history(lane.mode, sym, history)
                    if abs(score) < min_signal_bps:
                        decisions.append({"t": ts.isoformat(), "lane": lane.name, "symbol": sym, "action": "skip", "signal_bps": score, "reason": why})
                        continue
                    if best is None or abs(score) > abs(best[2]):
                        best = (sym, 1 if score > 0 else -1, score, why)
                if best is not None:
                    sym, direction, score, why = best
                    row = by_symbol[sym]
                    entry = float(row["close"]) * (1.0 + direction * slippage_bps / 10000.0)
                    qty = int((nav * notional_pct) // entry)
                    if qty > 0:
                        position = {
                            "symbol": sym,
                            "entry_t": ts,
                            "entry_price": entry,
                            "qty": qty,
                            "direction": direction,
                            "held_bars": 0,
                            "signal_bps": score,
                        }
                        trades_today[date_key] += 1
                        decisions.append({"t": ts.isoformat(), "lane": lane.name, "symbol": sym, "action": "entry_long" if direction > 0 else "entry_short", "signal_bps": score, "reason": why})

        mark_nav = nav
        if position is not None and position["symbol"] in by_symbol:
            mark = float(by_symbol[position["symbol"]]["close"])
            mark_nav += (mark - float(position["entry_price"])) * int(position["qty"]) * int(position["direction"])
        equity_rows.append({"t": ts.isoformat(), "lane": lane.name, "nav": mark_nav, "realized_nav": nav, "open": 1 if position else 0})

    equity = pd.DataFrame(equity_rows)
    trades_df = pd.DataFrame(trades)
    decisions_df = pd.DataFrame(decisions)
    equity["t_dt"] = pd.to_datetime(equity["t"], utc=True)
    equity["peak"] = equity["nav"].cummax()
    equity["dd_pct"] = (equity["nav"] / equity["peak"] - 1.0) * 100.0
    final_nav = float(equity["nav"].iloc[-1]) if not equity.empty else capital
    days = max(1, bars["date"].nunique())
    summary = {
        "lane": lane.name,
        "mode": lane.mode,
        "final_nav": final_nav,
        "return_pct": (final_nav / capital - 1.0) * 100.0,
        "monthly_pct": ((final_nav / capital) ** (21.0 / days) - 1.0) * 100.0 if final_nav > 0 else -100.0,
        "max_dd_pct": float(equity["dd_pct"].min()) if not equity.empty else 0.0,
        "trades": int(len(trades_df)),
        "trades_per_day": float(len(trades_df) / days),
        "win_rate_pct": float((trades_df["pnl"] > 0).mean() * 100.0) if not trades_df.empty else 0.0,
        "worst_trade": float(trades_df["pnl"].min()) if not trades_df.empty else 0.0,
        "best_trade": float(trades_df["pnl"].max()) if not trades_df.empty else 0.0,
    }
    return {"summary": summary, "equity": equity, "trades": trades_df, "decisions": decisions_df}


def make_report(results: list[dict[str, Any]], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame([r["summary"] for r in results]).sort_values("return_pct", ascending=False)
    summary.to_csv(out_dir / "summary.csv", index=False)
    for r in results:
        name = r["summary"]["lane"]
        r["equity"].drop(columns=["t_dt"], errors="ignore").to_csv(out_dir / f"{name}_equity.csv", index=False)
        r["trades"].to_csv(out_dir / f"{name}_trades.csv", index=False)
        r["decisions"].to_csv(out_dir / f"{name}_decisions.csv", index=False)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.7, 0.3], subplot_titles=("Equity", "Drawdown"))
    for r in results:
        eq = r["equity"]
        fig.add_trace(go.Scatter(x=eq["t_dt"], y=eq["nav"], mode="lines", name=r["summary"]["lane"]), row=1, col=1)
        fig.add_trace(go.Scatter(x=eq["t_dt"], y=eq["dd_pct"], mode="lines", name=f"{r['summary']['lane']} DD", showlegend=False), row=2, col=1)
    fig.update_layout(height=780, template="plotly_white", title="Local Max-HFT Simulation")
    html = f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>Max HFT Local Sim</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th,td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
th:first-child,td:first-child,th:nth-child(2),td:nth-child(2) {{ text-align: left; }}
th {{ background:#f4f6f8; }}
</style></head><body>
<h1>Local Max-HFT Simulation</h1>
<p>No network. No paper orders. Uses cached Alpaca 1-minute bars.</p>
{summary.to_html(index=False, float_format=lambda x: f"{x:,.2f}")}
{fig.to_html(include_plotlyjs="cdn", full_html=False)}
</body></html>
"""
    report = out_dir / "max_hft_local_sim_report.html"
    report.write_text(html)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bars", type=Path, default=None)
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--notional-pct", type=float, default=0.95)
    parser.add_argument("--min-signal-bps", type=float, default=0.1)
    parser.add_argument("--take-profit-bps", type=float, default=5.0)
    parser.add_argument("--stop-loss-bps", type=float, default=4.0)
    parser.add_argument("--max-hold-sec", type=float, default=45.0)
    parser.add_argument("--slippage-bps", type=float, default=8.0,
                         help="Round-trip slippage in bps. Default raised from 2.0 to 8.0 "
                               "per Day 1 live evidence and 60-model swarm consensus "
                               "(19/20 NIM models flagged the original assumption as fantasy).")
    parser.add_argument("--max-trades-per-day", type=int, default=2000)
    parser.add_argument("--open-report", action="store_true")
    args = parser.parse_args()

    bars_path = args.bars or latest_cached_bars()
    bars = pd.read_parquet(bars_path).sort_values(["t", "symbol"]).copy()
    bars = bars[bars["symbol"].isin(DEFAULT_SYMBOLS)].copy()
    out_dir = RUNS / f"max_hft_local_sim_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}"
    print(f"Using bars: {bars_path}")
    print(f"Rows={len(bars):,} days={bars['date'].nunique()} {bars['t'].min()} -> {bars['t'].max()}")
    results = [
        simulate_lane(
            bars,
            lane,
            capital=args.capital,
            notional_pct=args.notional_pct,
            min_signal_bps=args.min_signal_bps,
            take_profit_bps=args.take_profit_bps,
            stop_loss_bps=args.stop_loss_bps,
            max_hold_sec=args.max_hold_sec,
            slippage_bps=args.slippage_bps,
            max_trades_per_day=args.max_trades_per_day,
        )
        for lane in LANES
    ]
    report = make_report(results, out_dir)
    summary = pd.DataFrame([r["summary"] for r in results]).sort_values("return_pct", ascending=False)
    print(summary.to_string(index=False))
    print(f"Report: {report}")
    if args.open_report:
        import os

        os.system(f"open {str(report)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
