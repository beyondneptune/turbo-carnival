"""Spec-driven HFT strategy simulator with regime filters.

Loads strategy specs from JSON files, runs each against cached 21-day 1-min
bars + today's bars, and produces a ranked comparison report.

The simulator uses realistic slippage (configurable per spec, default 8 bps
matching the 60-model consensus) and supports multiple signal modes and
regime filter primitives. The Codex signal family adds non-threshold-only
features: volume-conditioned fades, session-VWAP reversion, and basket
dispersion reversion.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO = Path("/Users/weeoto/turbo-carnival")
RUNS = REPO / "experiments" / "runs"
SPECS_BASELINE = REPO / "data" / "strategy_specs" / "baseline"
SPECS_PROPOSED = REPO / "data" / "strategy_specs" / "proposed"
SPECS_CODEX = REPO / "data" / "strategy_specs" / "codex"
SPECS_NIM = REPO / "data" / "strategy_specs" / "nim_proposed"
SPECS_STRESS = REPO / "data" / "strategy_specs" / "stress_12bps"

NEWS_EVENTS = [
    # (UTC datetime, label, scope)
    ("2026-05-11T14:07:00Z", "Trump rejects Iran proposal", "macro"),
    ("2026-05-11T17:10:00Z", "Apple-Intel chip deal", "AAPL"),
    ("2026-05-11T18:00:00Z", "AMD server CPU supercycle", "AMD"),
    ("2026-05-11T18:38:00Z", "Musk+Cook Xi summit", "TSLA,META,AAPL"),
    ("2026-05-11T18:52:00Z", "Meta lawsuit California", "META"),
]


@dataclass
class Filter:
    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategySpec:
    name: str
    universe: list[str]
    signal_mode: str
    min_signal_bps: float
    take_profit_bps: float
    stop_loss_bps: float
    max_hold_sec: float
    max_positions: int
    notional_pct: float
    slippage_bps: float
    signal_params: dict[str, Any] = field(default_factory=dict)
    filters: list[Filter] = field(default_factory=list)
    description: str = ""

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "StrategySpec":
        filters = [Filter(type=f["type"], params={k: v for k, v in f.items() if k != "type"})
                   for f in d.get("filters", [])]
        return cls(
            name=d["name"],
            universe=d.get("universe", ["SPY", "QQQ", "NVDA", "TSLA", "AAPL", "META", "AMD"]),
            signal_mode=d["signal_mode"],
            signal_params=dict(d.get("signal_params", {})),
            min_signal_bps=float(d.get("min_signal_bps", 18.0)),
            take_profit_bps=float(d.get("take_profit_bps", 70.0)),
            stop_loss_bps=float(d.get("stop_loss_bps", 25.0)),
            max_hold_sec=float(d.get("max_hold_sec", 900.0)),
            max_positions=int(d.get("max_positions", 1)),
            notional_pct=float(d.get("notional_pct", 0.95)),
            slippage_bps=float(d.get("slippage_bps", 8.0)),
            filters=filters,
            description=d.get("description", ""),
        )


# ---------------------------------------------------------------------------
def load_bars() -> pd.DataFrame:
    paths = sorted(RUNS.glob("valk_hft_demo_*/minute_bars.parquet"))
    if not paths:
        raise FileNotFoundError("No cached minute_bars.parquet found.")
    df = pd.read_parquet(paths[-1])
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df = df.sort_values(["t", "symbol"]).reset_index(drop=True)
    return df


def parse_news_events() -> list[tuple[datetime, str, list[str]]]:
    out = []
    for ts, label, scope in NEWS_EVENTS:
        t = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        scope_list = ["*"] if scope == "macro" else scope.split(",")
        out.append((t, label, scope_list))
    return out


# ---------------------------------------------------------------------------
def _clean_float(value: Any, default: float = 0.0) -> float:
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def _row_float(row: Any | None, key: str, default: float = 0.0) -> float:
    if row is None:
        return default
    getter = getattr(row, "get", None)
    if getter is None:
        return default
    return _clean_float(getter(key, default), default)


def _return_bps(history: deque[float], lookback: int) -> float | None:
    if len(history) <= lookback:
        return None
    return (history[-1] / history[-lookback - 1] - 1.0) * 10000.0


def _cross_sectional_return_bps(
    histories: dict[str, deque[float]],
    lookback: int,
) -> dict[str, float]:
    moves = {}
    for sym, h in histories.items():
        move = _return_bps(h, lookback)
        if move is not None and math.isfinite(move):
            moves[sym] = move
    return moves


def compute_signal(
    mode: str,
    history: deque[float],
    *,
    row: Any | None = None,
    sym: str | None = None,
    all_histories: dict[str, deque[float]] | None = None,
    params: dict[str, Any] | None = None,
) -> tuple[float, str]:
    """Return (score_bps, reason). History is most-recent first not enforced;
    we use deque.append so last element is newest. Need >= 12 points."""
    params = params or {}
    if len(history) < 4:
        return 0.0, "warming"
    one_bps = (history[-1] / history[-2] - 1.0) * 10000.0
    three_bps = (history[-1] / history[-4] - 1.0) * 10000.0
    twelve_bps = ((history[-1] / history[-12] - 1.0) * 10000.0) if len(history) >= 12 else three_bps

    if mode == "momentum":
        score = 0.65 * three_bps + 0.35 * one_bps
    elif mode == "mean_reversion":
        if abs(three_bps) > 1.5:
            score = -0.80 * one_bps - 0.35 * three_bps
        else:
            score = -0.20 * twelve_bps
    elif mode == "breakout":
        if abs(twelve_bps) > 3.0:
            score = twelve_bps + 0.40 * three_bps
        else:
            score = 0.35 * (0.65 * three_bps + 0.35 * one_bps)
    elif mode == "turbo":
        if abs(twelve_bps) > 8.0:
            score = twelve_bps + 0.40 * three_bps
        elif abs(three_bps) > 2.0:
            score = 0.65 * three_bps + 0.35 * one_bps
        else:
            score = -0.80 * one_bps - 0.35 * three_bps
    elif mode == "news_fade":
        # Fade large 1-bar moves (assumes news pop)
        if abs(one_bps) > 20.0:
            score = -1.5 * one_bps
        else:
            score = 0.0
    elif mode == "neutral_carry":
        # No directional bet; designed to be paired with options or only enter on extreme
        if abs(twelve_bps) > 40.0:
            score = -0.5 * twelve_bps  # fade extreme moves only
        else:
            score = 0.0
    elif mode == "volume_fade":
        # Fade weakly sponsored moves: large short-window move on low relative volume.
        # This is deliberately not another raw return-threshold variant; volume_z is
        # the gate and score multiplier, so high-conviction moves are ignored.
        max_volume_z = float(params.get("max_volume_z", 0.25))
        min_abs_move_bps = float(params.get("min_abs_move_bps", 4.0))
        volume_z = _row_float(row, "volume_z", 999.0)
        if volume_z <= max_volume_z and abs(three_bps) >= min_abs_move_bps:
            low_volume_weight = 1.0 + min(1.5, max(0.0, max_volume_z - volume_z))
            score = -(0.70 * three_bps + 0.30 * one_bps) * low_volume_weight
        else:
            score = 0.0
        return score, f"{mode}_volz={volume_z:+.2f}_1={one_bps:+.1f}_3={three_bps:+.1f}"
    elif mode == "vwap_reversion":
        # Session-VWAP anchor: fade only when price is meaningfully away from
        # session VWAP and the minute volume is not confirming the move.
        max_volume_z = float(params.get("max_volume_z", 0.75))
        min_vwap_dist_bps = float(params.get("min_vwap_dist_bps", 6.0))
        volume_z = _row_float(row, "volume_z", 999.0)
        vwap_dist_bps = _row_float(row, "vwap_dist", 0.0) * 10000.0
        if volume_z <= max_volume_z and abs(vwap_dist_bps) >= min_vwap_dist_bps:
            score = -1.15 * vwap_dist_bps - 0.20 * one_bps
        else:
            score = 0.0
        return score, f"{mode}_vwap={vwap_dist_bps:+.1f}_volz={volume_z:+.2f}_1={one_bps:+.1f}"
    elif mode == "dispersion_reversion":
        # Cross-sectional residual: fade the symbol that has most detached
        # from the basket over N minutes. This is closer to a one-leg proxy
        # for stat-arb/pairs than a single-name return threshold.
        lookback = int(params.get("lookback_bars", 10))
        min_z = float(params.get("min_z", 1.20))
        min_dispersion_bps = float(params.get("min_dispersion_bps", 4.0))
        if not sym or not all_histories:
            return 0.0, f"{mode}_missing_context"
        moves = _cross_sectional_return_bps(all_histories, lookback)
        if sym not in moves or len(moves) < 4:
            return 0.0, f"{mode}_warming"
        vals = np.array(list(moves.values()), dtype=float)
        basket_mean = float(np.mean(vals))
        basket_std = float(np.std(vals))
        if basket_std < min_dispersion_bps:
            return 0.0, f"{mode}_disp={basket_std:.1f}<min"
        z = (moves[sym] - basket_mean) / basket_std
        if abs(z) < min_z:
            return 0.0, f"{mode}_z={z:+.2f}<min"
        score = -(moves[sym] - basket_mean)
        return score, f"{mode}_move={moves[sym]:+.1f}_mean={basket_mean:+.1f}_z={z:+.2f}_disp={basket_std:.1f}"
    else:
        score = 0.0
    return score, f"{mode}_1={one_bps:+.1f}_3={three_bps:+.1f}_12={twelve_bps:+.1f}"


def compute_atr(history: deque[float], window: int = 20) -> float:
    """Approximate ATR from close-to-close abs returns over `window` bars."""
    if len(history) < window:
        return 0.0
    arr = list(history)[-window:]
    diffs = [abs(arr[i] / arr[i - 1] - 1.0) for i in range(1, len(arr))]
    return float(np.mean(diffs)) * 10000.0 if diffs else 0.0


def filter_blocks_entry(spec_filters: list[Filter], sym: str, t: datetime,
                         history: deque[float], all_histories: dict[str, deque[float]],
                         news_active_until: dict[str, datetime]) -> tuple[bool, str]:
    """Return (blocked, reason). Apply all filters; first blocking filter wins."""
    def to_bps(v: float) -> float:
        """Auto-convert: values < 1.0 are ratios, multiply by 10000 to get bps."""
        return v * 10000.0 if v < 1.0 else v

    for f in spec_filters:
        if f.type == "atr_gate":
            atr_bps_val = compute_atr(history, int(f.params.get("window", 20)))
            thresh = to_bps(float(f.params.get("threshold_bps", 30)))
            if atr_bps_val >= thresh:
                return True, f"atr_gate ATR={atr_bps_val:.1f}>={thresh}"
        elif f.type == "vol_of_vol_gate":
            # Block if recent 5-bar stdev > N x trailing 20-bar median
            if len(history) >= 25:
                arr = list(history)
                short = np.std([arr[i] / arr[i - 1] - 1 for i in range(len(arr) - 5, len(arr))])
                med = np.median([abs(arr[i] / arr[i - 1] - 1) for i in range(len(arr) - 25, len(arr) - 5)])
                ratio = short / med if med > 0 else 0
                if ratio >= float(f.params.get("ratio", 1.4)):
                    return True, f"vov ratio={ratio:.2f}"
        elif f.type == "news_blackout":
            until = news_active_until.get(sym) or news_active_until.get("*")
            if until and t < until:
                return True, f"news_blackout until {until.strftime('%H:%M')}"
        elif f.type == "abs_return_gate":
            if len(history) >= 2:
                one_bps = abs(history[-1] / history[-2] - 1.0) * 10000.0
                thresh = to_bps(float(f.params.get("threshold_bps", 30)))
                if one_bps >= thresh:
                    return True, f"abs_ret {one_bps:.1f}>={thresh}"
        elif f.type == "time_gate":
            # Only trade between start_et and end_et (ET HH:MM strings)
            from zoneinfo import ZoneInfo
            t_et = t.astimezone(ZoneInfo("America/New_York"))
            start = f.params.get("start_et", "09:30")
            end = f.params.get("end_et", "15:55")
            hh, mm = start.split(":")
            h_start = int(hh) * 60 + int(mm)
            hh, mm = end.split(":")
            h_end = int(hh) * 60 + int(mm)
            now_min = t_et.hour * 60 + t_et.minute
            if now_min < h_start or now_min >= h_end:
                return True, f"time_gate outside {start}-{end}"
        elif f.type == "correlation_check":
            threshold_bps = to_bps(float(f.params.get("threshold_bps", 20)))
            min_correlated = int(f.params.get("min_count", 3))
            same_direction = 0
            for other, h in all_histories.items():
                if len(h) >= 6:
                    mv = (h[-1] / h[-6] - 1.0) * 10000.0
                    if mv >= threshold_bps or mv <= -threshold_bps:
                        same_direction += 1
            if same_direction >= min_correlated:
                return True, f"correl {same_direction} symbols moving"
    return False, "ok"


# ---------------------------------------------------------------------------
@dataclass
class SimResult:
    spec_name: str
    description: str
    final_nav: float
    return_pct: float
    monthly_pct: float
    max_dd_pct: float
    sharpe: float
    trades: int
    win_rate_pct: float
    avg_pnl_bps: float
    best_trade_bps: float
    worst_trade_bps: float
    trades_per_day: float
    blocked_by_filter: int
    equity_curve: pd.DataFrame
    trade_log: pd.DataFrame
    spec_dict: dict[str, Any]


def simulate(spec: StrategySpec, bars: pd.DataFrame, news_events: list,
              capital: float = 100_000.0) -> SimResult:
    bars = bars[bars["symbol"].isin(spec.universe)].copy()
    bars = bars.sort_values("t").reset_index(drop=True)

    history: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=80))
    open_pos: dict[str, Any] = {}  # max 1
    nav = capital
    equity_rows = []
    trade_log = []
    blocked = 0

    # Precompute news_active windows: {symbol: until_dt}
    news_blackout_secs = 300
    for filt in spec.filters:
        if filt.type == "news_blackout":
            news_blackout_secs = int(filt.params.get("cooldown_sec", 300))

    by_t = bars.groupby("t", sort=True)

    for ts, frame in by_t:
        # Update news_active_until based on news events
        news_active_until = {}
        for news_t, _label, scope in news_events:
            end_t = pd.Timestamp(news_t.timestamp() + news_blackout_secs, unit="s", tz="UTC").to_pydatetime()
            if news_t <= ts.to_pydatetime() < end_t:
                for sc in scope:
                    news_active_until[sc] = end_t

        per_sym = {row["symbol"]: row for _, row in frame.iterrows()}
        # Update histories
        for sym, row in per_sym.items():
            history[sym].append(float(row["close"]))

        # Check exits for open position
        if open_pos:
            sym = open_pos["symbol"]
            if sym in per_sym:
                row = per_sym[sym]
                age_sec = (ts - open_pos["entry_t"]).total_seconds()
                direction = int(open_pos["direction"])
                entry_px = float(open_pos["entry_price"])
                close_px = float(row["close"])
                high = float(row["high"])
                low = float(row["low"])
                tp_px = entry_px * (1.0 + direction * spec.take_profit_bps / 10000.0)
                sl_px = entry_px * (1.0 - direction * spec.stop_loss_bps / 10000.0)

                exit_px = None
                exit_reason = None
                slip = spec.slippage_bps / 10000.0
                if direction > 0:
                    if low <= sl_px:
                        exit_px = sl_px * (1.0 - slip)
                        exit_reason = "stop_loss"
                    elif high >= tp_px:
                        exit_px = tp_px * (1.0 - slip)
                        exit_reason = "take_profit"
                else:
                    if high >= sl_px:
                        exit_px = sl_px * (1.0 + slip)
                        exit_reason = "stop_loss"
                    elif low <= tp_px:
                        exit_px = tp_px * (1.0 + slip)
                        exit_reason = "take_profit"
                if exit_px is None and age_sec >= spec.max_hold_sec:
                    exit_px = close_px * (1.0 - direction * slip)
                    exit_reason = "time_exit"

                # Force-flat near EOD
                from zoneinfo import ZoneInfo
                t_et = ts.tz_convert(ZoneInfo("America/New_York"))
                if exit_px is None and (t_et.hour, t_et.minute) >= (15, 55):
                    exit_px = close_px * (1.0 - direction * slip)
                    exit_reason = "eod_flat"

                if exit_px is not None:
                    qty = int(open_pos["qty"])
                    pnl = (exit_px - entry_px) * qty * direction
                    nav += pnl
                    pnl_bps = (exit_px / entry_px - 1.0) * 10000.0 * direction
                    trade_log.append({
                        "entry_t": open_pos["entry_t"],
                        "exit_t": ts,
                        "symbol": sym,
                        "side": "long" if direction > 0 else "short",
                        "qty": qty,
                        "entry_price": entry_px,
                        "exit_price": exit_px,
                        "pnl_usd": pnl,
                        "pnl_bps": pnl_bps,
                        "exit_reason": exit_reason,
                        "hold_sec": age_sec,
                    })
                    open_pos = {}

        # Look for entry if no open position
        if not open_pos:
            best = None
            for sym, row in per_sym.items():
                score, reason = compute_signal(
                    spec.signal_mode,
                    history[sym],
                    row=row,
                    sym=sym,
                    all_histories=history,
                    params=spec.signal_params,
                )
                if abs(score) < spec.min_signal_bps:
                    continue
                blocked_now, _ = filter_blocks_entry(
                    spec.filters, sym, ts.to_pydatetime(),
                    history[sym], history, news_active_until,
                )
                if blocked_now:
                    blocked += 1
                    continue
                if best is None or abs(score) > abs(best[2]):
                    best = (sym, 1 if score > 0 else -1, score, reason)
            if best is not None:
                sym, direction, score, reason = best
                row = per_sym[sym]
                entry = float(row["close"]) * (1.0 + direction * spec.slippage_bps / 10000.0)
                qty = max(1, int(nav * spec.notional_pct / entry))
                open_pos = {
                    "symbol": sym, "entry_t": ts, "entry_price": entry,
                    "qty": qty, "direction": direction, "score": score, "reason": reason,
                }

        # Mark to market equity
        mtm = nav
        if open_pos:
            sym = open_pos["symbol"]
            if sym in per_sym:
                mtm = nav + (float(per_sym[sym]["close"]) - open_pos["entry_price"]) * \
                       open_pos["qty"] * open_pos["direction"]
        equity_rows.append({"t": ts, "nav": mtm, "realized": nav, "open": 1 if open_pos else 0})

    equity = pd.DataFrame(equity_rows)
    trades = pd.DataFrame(trade_log)
    final_nav = float(equity["nav"].iloc[-1]) if not equity.empty else capital
    ret = (final_nav / capital - 1.0) * 100.0
    peak = equity["nav"].cummax()
    dd = (equity["nav"] / peak - 1.0) * 100.0
    days = bars["date"].nunique() if "date" in bars.columns else 21
    months = days / 21.0
    monthly = (((final_nav / capital) ** (1.0 / max(months, 0.1)) - 1.0) * 100.0) if final_nav > 0 else -100.0

    # Sharpe approx from per-trade pnl_bps
    if not trades.empty and trades["pnl_bps"].std() > 0:
        sharpe = float(trades["pnl_bps"].mean() / trades["pnl_bps"].std() * math.sqrt(252))
    else:
        sharpe = 0.0

    return SimResult(
        spec_name=spec.name,
        description=spec.description,
        final_nav=final_nav,
        return_pct=ret,
        monthly_pct=monthly,
        max_dd_pct=float(dd.min()) if not dd.empty else 0.0,
        sharpe=sharpe,
        trades=len(trades),
        win_rate_pct=float((trades["pnl_usd"] > 0).mean() * 100.0) if not trades.empty else 0.0,
        avg_pnl_bps=float(trades["pnl_bps"].mean()) if not trades.empty else 0.0,
        best_trade_bps=float(trades["pnl_bps"].max()) if not trades.empty else 0.0,
        worst_trade_bps=float(trades["pnl_bps"].min()) if not trades.empty else 0.0,
        trades_per_day=float(len(trades) / max(days, 1)),
        blocked_by_filter=blocked,
        equity_curve=equity,
        trade_log=trades,
        spec_dict={
            "name": spec.name,
            "signal_mode": spec.signal_mode,
            "signal_params": spec.signal_params,
            "min_signal_bps": spec.min_signal_bps,
            "tp_bps": spec.take_profit_bps,
            "sl_bps": spec.stop_loss_bps,
            "filters": [{"type": f.type, **f.params} for f in spec.filters],
        },
    )


# ---------------------------------------------------------------------------
def load_specs(folders: list[Path]) -> list[StrategySpec]:
    specs = []
    for folder in folders:
        if not folder.exists():
            continue
        for p in sorted(folder.glob("*.json")):
            try:
                data = json.loads(p.read_text())
                if data.get("kind") and data.get("kind") not in {"hft", "equity_intraday"}:
                    print(f"INFO skip non-HFT spec {p}: kind={data.get('kind')}")
                    continue
                specs.append(StrategySpec.from_dict(data))
            except Exception as exc:
                print(f"WARN parse {p}: {exc}")
    return specs


def make_report(results: list[SimResult], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Sort by sharpe then monthly
    results_sorted = sorted(results,
                             key=lambda r: (r.sharpe if r.trades > 0 else -999, r.monthly_pct),
                             reverse=True)

    # Summary CSV
    rows = []
    for i, r in enumerate(results_sorted, 1):
        rows.append({
            "rank": i,
            "name": r.spec_name,
            "mode": r.spec_dict.get("signal_mode", ""),
            "trades": r.trades,
            "win_rate_pct": round(r.win_rate_pct, 1),
            "avg_bps": round(r.avg_pnl_bps, 2),
            "return_pct": round(r.return_pct, 2),
            "monthly_pct": round(r.monthly_pct, 2),
            "max_dd_pct": round(r.max_dd_pct, 2),
            "sharpe": round(r.sharpe, 2),
            "blocked": r.blocked_by_filter,
            "description": r.description[:80],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "summary.csv", index=False)

    # Multi-panel comparison chart
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Equity curves",
            "Return vs Drawdown",
            "Win rate vs trade count",
            "Sharpe distribution",
        ),
        vertical_spacing=0.13,
        horizontal_spacing=0.10,
    )
    for r in results_sorted[:12]:
        if r.equity_curve.empty:
            continue
        fig.add_trace(
            go.Scatter(x=r.equity_curve["t"], y=r.equity_curve["nav"],
                        name=r.spec_name, mode="lines", line=dict(width=1.4)),
            row=1, col=1,
        )
    for r in results_sorted:
        if r.trades == 0:
            continue
        fig.add_trace(
            go.Scatter(x=[r.max_dd_pct], y=[r.monthly_pct],
                        mode="markers+text", text=[r.spec_name],
                        textposition="top center", textfont=dict(size=8),
                        marker=dict(size=10), showlegend=False),
            row=1, col=2,
        )
        fig.add_trace(
            go.Scatter(x=[r.trades], y=[r.win_rate_pct],
                        mode="markers+text", text=[r.spec_name],
                        textposition="top center", textfont=dict(size=8),
                        marker=dict(size=10), showlegend=False),
            row=2, col=1,
        )
    fig.add_trace(
        go.Bar(x=[r.spec_name for r in results_sorted if r.trades > 0],
                y=[r.sharpe for r in results_sorted if r.trades > 0],
                showlegend=False),
        row=2, col=2,
    )
    fig.update_xaxes(title="Max DD (%)", row=1, col=2)
    fig.update_yaxes(title="Monthly (%)", row=1, col=2)
    fig.update_xaxes(title="Trades", row=2, col=1)
    fig.update_yaxes(title="Win rate (%)", row=2, col=1)
    fig.update_yaxes(title="Sharpe", row=2, col=2)
    fig.update_layout(height=900, template="plotly_white",
                       title=f"Strategy Swarm Lab — {len(results)} variants tested",
                       margin=dict(t=80, l=60, r=30, b=50))

    # Build HTML with ranked table
    html_table = df[["rank", "name", "mode", "trades", "win_rate_pct", "avg_bps",
                      "monthly_pct", "max_dd_pct", "sharpe", "blocked"]].to_html(
        index=False, classes="data-table", escape=False,
    )

    html = f"""
<!doctype html>
<html><head><meta charset="utf-8"><title>Strategy Swarm Lab Results</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        margin: 24px; color: #1f2937; max-width: 1400px; }}
h1 {{ margin: 0 0 4px 0; }}
.note {{ color: #6b7280; margin-bottom: 24px; }}
table.data-table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 16px; }}
table.data-table th, table.data-table td {{ border: 1px solid #e5e7eb; padding: 6px 10px;
                                              text-align: right; white-space: nowrap; }}
table.data-table th {{ background: #f3f4f6; font-weight: 600; }}
table.data-table td:nth-child(1), table.data-table td:nth-child(2),
table.data-table td:nth-child(3) {{ text-align: left; }}
table.data-table tr:first-child td {{ background: #fef3c7; font-weight: 600; }}
</style></head><body>
<h1>Strategy Swarm Lab — Ranked Results</h1>
<p class="note">{len(results)} strategy variants simulated across 21 days of cached 1-min bars
(7 tickers). Slippage modeled per-spec. Sorted by Sharpe then monthly return.
Generated {datetime.now(timezone.utc).isoformat()}.</p>
{fig.to_html(include_plotlyjs="cdn", full_html=False)}
<h2>Ranked Summary</h2>
{html_table}
</body></html>
"""
    report_path = out_dir / "swarm_lab_report.html"
    report_path.write_text(html)
    return report_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument("--stress-only", action="store_true",
                         help="Only load stress_12bps specs (Rec #13 stress test).")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.stress_only:
        folders = [SPECS_STRESS]
    else:
        folders = [SPECS_BASELINE]
        if not args.baseline_only:
            folders.append(SPECS_PROPOSED)
            folders.append(SPECS_CODEX)
            folders.append(SPECS_NIM)

    specs = load_specs(folders)
    print(f"Loaded {len(specs)} specs from {[str(f) for f in folders]}")
    if not specs:
        print("No specs found. Drop JSON files into data/strategy_specs/baseline/ first.")
        return 1

    bars = load_bars()
    news = parse_news_events()
    print(f"Loaded {len(bars):,} bar rows, {len(news)} news events")

    results = []
    for i, spec in enumerate(specs, 1):
        print(f"  [{i}/{len(specs)}] simulating {spec.name}...")
        try:
            r = simulate(spec, bars, news)
            results.append(r)
        except Exception as exc:
            print(f"    error: {type(exc).__name__}: {exc}")

    out_dir = Path(args.out) if args.out else RUNS / f"swarm_lab_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    report = make_report(results, out_dir)
    print(f"\nDONE. Report: {report}")
    print(f"Summary CSV: {out_dir / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
