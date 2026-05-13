"""Valk HFT-ish demo.

This is a local no-order simulator for a Valk replacement lane. It uses Alpaca
1-minute stock bars and tests intraday high-turnover scalping logic before we
touch the live Valk paper account.

Important: this is not true exchange-colocated HFT. Retail Alpaca REST/WebSocket
latency makes real HFT impossible. This tests a practical "many intraday
decisions, short holds, tight exits" paper lane.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ENV_PATH = REPO_ROOT / "council" / ".env.local"
RUNS_DIR = REPO_ROOT / "experiments" / "runs"
DATA_BASE = "https://data.alpaca.markets"
NY = ZoneInfo("America/New_York")


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    mode: str
    symbols: tuple[str, ...]
    trade_notional_pct: float
    max_concurrent: int
    max_trades_per_day: int
    hold_minutes: int
    take_profit_bps: float
    stop_loss_bps: float
    trailing_stop_bps: float
    slippage_bps: float
    fee_bps: float
    min_score: int
    allow_short: bool = True


CONFIGS = [
    StrategyConfig(
        name="valk_demo_momentum_hft",
        mode="momentum",
        symbols=("SPY", "QQQ", "NVDA", "TSLA", "AMD", "AAPL", "META"),
        trade_notional_pct=0.22,
        max_concurrent=5,
        max_trades_per_day=90,
        hold_minutes=8,
        take_profit_bps=22,
        stop_loss_bps=13,
        trailing_stop_bps=10,
        slippage_bps=1.5,
        fee_bps=0.25,
        min_score=2,
    ),
    StrategyConfig(
        name="valk_demo_reversion_hft",
        mode="reversion",
        symbols=("SPY", "QQQ", "NVDA", "TSLA", "AMD", "AAPL", "META"),
        trade_notional_pct=0.20,
        max_concurrent=6,
        max_trades_per_day=120,
        hold_minutes=5,
        take_profit_bps=16,
        stop_loss_bps=10,
        trailing_stop_bps=8,
        slippage_bps=1.5,
        fee_bps=0.25,
        min_score=2,
    ),
    StrategyConfig(
        name="valk_demo_turbo_hybrid",
        mode="hybrid",
        symbols=("SPY", "QQQ", "NVDA", "TSLA", "AMD", "AAPL", "META"),
        trade_notional_pct=0.30,
        max_concurrent=8,
        max_trades_per_day=180,
        hold_minutes=4,
        take_profit_bps=14,
        stop_loss_bps=9,
        trailing_stop_bps=7,
        slippage_bps=2.0,
        fee_bps=0.35,
        min_score=1,
    ),
]


def load_env(path: Path = ENV_PATH) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def alpaca_headers() -> dict[str, str]:
    load_env()
    key = os.environ.get("ALPACA_VALK_KEY") or os.environ.get("ALPACA_AEGIS_KEY")
    secret = os.environ.get("ALPACA_VALK_SECRET") or os.environ.get("ALPACA_AEGIS_SECRET")
    if not key or not secret:
        raise RuntimeError("Missing Alpaca keys in council/.env.local")
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def fetch_minute_bars(
    symbols: list[str],
    *,
    start: datetime,
    end: datetime,
    feed: str = "iex",
) -> pd.DataFrame:
    session = requests.Session()
    session.headers.update(alpaca_headers())
    rows: list[dict[str, Any]] = []
    page_token = None
    while True:
        params = {
            "symbols": ",".join(symbols),
            "timeframe": "1Min",
            "start": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "end": end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            "limit": 10000,
            "adjustment": "raw",
            "feed": feed,
            "page_token": page_token,
        }
        params = {k: v for k, v in params.items() if v is not None}
        resp = session.get(f"{DATA_BASE}/v2/stocks/bars", params=params, timeout=30)
        if not resp.ok:
            raise RuntimeError(f"Alpaca bars failed {resp.status_code}: {resp.text[:500]}")
        payload = resp.json()
        for sym, bars in (payload.get("bars") or {}).items():
            for b in bars:
                rows.append(
                    {
                        "symbol": sym,
                        "t": pd.Timestamp(b["t"], tz="UTC"),
                        "open": float(b["o"]),
                        "high": float(b["h"]),
                        "low": float(b["l"]),
                        "close": float(b["c"]),
                        "volume": float(b["v"]),
                        "vwap_bar": float(b.get("vw") or b["c"]),
                        "trades": int(b.get("n") or 0),
                    }
                )
        page_token = payload.get("next_page_token")
        if not page_token:
            break

    if not rows:
        raise RuntimeError("No minute bars returned")
    df = pd.DataFrame(rows).sort_values(["t", "symbol"]).reset_index(drop=True)
    df["t_ny"] = df["t"].dt.tz_convert(NY)
    df["date"] = df["t_ny"].dt.date.astype(str)
    regular = (df["t_ny"].dt.time >= pd.Timestamp("09:30").time()) & (
        df["t_ny"].dt.time <= pd.Timestamp("15:59").time()
    )
    return df.loc[regular].copy()


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in df.groupby("symbol", sort=False):
        g = g.sort_values("t").copy()
        g["ret1"] = g["close"].pct_change()
        g["ret3"] = g["close"].pct_change(3)
        g["ret10"] = g["close"].pct_change(10)
        g["range_pct"] = (g["high"] - g["low"]) / g["close"].replace(0, np.nan)
        vol_mean = g["volume"].rolling(30, min_periods=10).mean()
        vol_std = g["volume"].rolling(30, min_periods=10).std().replace(0, np.nan)
        g["volume_z"] = ((g["volume"] - vol_mean) / vol_std).fillna(0)
        ret_std = g["ret1"].rolling(30, min_periods=10).std().replace(0, np.nan)
        g["ret1_z"] = (g["ret1"] / ret_std).replace([np.inf, -np.inf], np.nan).fillna(0)
        g["roll_high_20"] = g["high"].shift(1).rolling(20, min_periods=10).max()
        g["roll_low_20"] = g["low"].shift(1).rolling(20, min_periods=10).min()

        chunks = []
        for _, d in g.groupby("date", sort=False):
            d = d.copy()
            pv = (d["vwap_bar"] * d["volume"]).cumsum()
            cv = d["volume"].cumsum().replace(0, np.nan)
            d["session_vwap"] = pv / cv
            d["vwap_dist"] = d["close"] / d["session_vwap"] - 1.0
            chunks.append(d)
        out.append(pd.concat(chunks))
    return pd.concat(out).sort_values(["t", "symbol"]).reset_index(drop=True)


def score_row(row: pd.Series, cfg: StrategyConfig) -> tuple[int, str]:
    score = 0
    tags: list[str] = []

    if cfg.mode in ("momentum", "hybrid"):
        if row["ret3"] > 0.0011 and row["ret10"] > 0 and row["vwap_dist"] > 0:
            score += 2
            tags.append("up_momentum")
        if row["ret3"] < -0.0011 and row["ret10"] < 0 and row["vwap_dist"] < 0:
            score -= 2
            tags.append("down_momentum")
        if row["close"] > row["roll_high_20"] and row["volume_z"] > 0.4:
            score += 1
            tags.append("breakout_up")
        if row["close"] < row["roll_low_20"] and row["volume_z"] > 0.4:
            score -= 1
            tags.append("breakout_down")

    if cfg.mode in ("reversion", "hybrid"):
        if row["ret1_z"] < -1.8 and row["vwap_dist"] < -0.0012:
            score += 2
            tags.append("oversold_revert")
        if row["ret1_z"] > 1.8 and row["vwap_dist"] > 0.0012:
            score -= 2
            tags.append("overbought_revert")

    if row["volume_z"] < -1.5:
        score = 0
        tags.append("thin_volume_block")
    return score, ",".join(tags)


def exit_price_for_position(pos: dict[str, Any], row: pd.Series, cfg: StrategyConfig) -> tuple[float | None, str | None]:
    direction = int(pos["direction"])
    entry = float(pos["entry_price"])
    best = float(pos["best_price"])
    if direction == 1:
        best = max(best, float(row["high"]))
        pos["best_price"] = best
        stop = entry * (1.0 - cfg.stop_loss_bps / 10000.0)
        trailing = best * (1.0 - cfg.trailing_stop_bps / 10000.0)
        take = entry * (1.0 + cfg.take_profit_bps / 10000.0)
        active_stop = max(stop, trailing)
        if row["low"] <= active_stop:
            return active_stop * (1.0 - cfg.slippage_bps / 10000.0), "stop_or_trail"
        if row["high"] >= take:
            return take * (1.0 - cfg.slippage_bps / 10000.0), "take_profit"
    else:
        best = min(best, float(row["low"]))
        pos["best_price"] = best
        stop = entry * (1.0 + cfg.stop_loss_bps / 10000.0)
        trailing = best * (1.0 + cfg.trailing_stop_bps / 10000.0)
        take = entry * (1.0 - cfg.take_profit_bps / 10000.0)
        active_stop = min(stop, trailing)
        if row["high"] >= active_stop:
            return active_stop * (1.0 + cfg.slippage_bps / 10000.0), "stop_or_trail"
        if row["low"] <= take:
            return take * (1.0 + cfg.slippage_bps / 10000.0), "take_profit"

    held = int((row["t"] - pos["entry_t"]) / pd.Timedelta(minutes=1))
    if held >= cfg.hold_minutes:
        px = float(row["close"]) * (1.0 - direction * cfg.slippage_bps / 10000.0)
        return px, "time_exit"
    if row["t_ny"].time() >= pd.Timestamp("15:55").time():
        px = float(row["close"]) * (1.0 - direction * cfg.slippage_bps / 10000.0)
        return px, "eod_flat"
    return None, None


def run_strategy(df: pd.DataFrame, cfg: StrategyConfig, initial_capital: float) -> dict[str, Any]:
    nav_realized = float(initial_capital)
    positions: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    equity: list[dict[str, Any]] = []
    trades_today: dict[str, int] = {}
    cooldown_until: dict[str, pd.Timestamp] = {}
    bar_map = {
        (row.t, row.symbol): row
        for row in df.itertuples(index=False)
    }

    for ts, frame in df.groupby("t", sort=True):
        rows_by_symbol = {str(r["symbol"]): r for _, r in frame.iterrows()}

        kept = []
        for pos in positions:
            row = rows_by_symbol.get(pos["symbol"])
            if row is None:
                kept.append(pos)
                continue
            exit_px, reason = exit_price_for_position(pos, row, cfg)
            if exit_px is None:
                kept.append(pos)
                continue
            qty = int(pos["qty"])
            direction = int(pos["direction"])
            gross = (exit_px - float(pos["entry_price"])) * qty * direction
            fees = (float(pos["entry_price"]) + exit_px) * qty * cfg.fee_bps / 10000.0
            pnl = gross - fees
            nav_realized += pnl
            trades.append(
                {
                    "symbol": pos["symbol"],
                    "entry_t": pos["entry_t"].isoformat(),
                    "exit_t": row["t"].isoformat(),
                    "side": "long" if direction == 1 else "short",
                    "entry_price": float(pos["entry_price"]),
                    "exit_price": float(exit_px),
                    "qty": qty,
                    "notional": float(pos["entry_price"]) * qty,
                    "pnl": pnl,
                    "pnl_bps": pnl / max(float(pos["entry_price"]) * qty, 1.0) * 10000.0,
                    "hold_minutes": int((row["t"] - pos["entry_t"]) / pd.Timedelta(minutes=1)),
                    "exit_reason": reason,
                    "signal": pos["signal"],
                }
            )
            cooldown_until[pos["symbol"]] = row["t"] + pd.Timedelta(minutes=2)
        positions = kept

        for _, row in frame.iterrows():
            date_key = str(row["date"])
            trades_today.setdefault(date_key, 0)
            if trades_today[date_key] >= cfg.max_trades_per_day:
                continue
            if len(positions) >= cfg.max_concurrent:
                continue
            if any(p["symbol"] == row["symbol"] for p in positions):
                continue
            if row["symbol"] in cooldown_until and row["t"] < cooldown_until[row["symbol"]]:
                continue
            if row["t_ny"].time() >= pd.Timestamp("15:45").time():
                continue

            score, tags = score_row(row, cfg)
            if abs(score) < cfg.min_score:
                continue
            direction = 1 if score > 0 else -1
            if direction == -1 and not cfg.allow_short:
                continue
            entry_px = float(row["close"]) * (1.0 + direction * cfg.slippage_bps / 10000.0)
            notional = max(0.0, nav_realized * cfg.trade_notional_pct)
            qty = int(notional // entry_px)
            if qty < 1:
                continue
            positions.append(
                {
                    "symbol": row["symbol"],
                    "entry_t": row["t"],
                    "entry_price": entry_px,
                    "qty": qty,
                    "direction": direction,
                    "best_price": entry_px,
                    "signal": tags,
                }
            )
            trades_today[date_key] += 1

        unrealized = 0.0
        for pos in positions:
            row = rows_by_symbol.get(pos["symbol"])
            if row is None:
                maybe = bar_map.get((ts, pos["symbol"]))
                if maybe is None:
                    continue
                mark = float(maybe.close)
            else:
                mark = float(row["close"])
            unrealized += (mark - float(pos["entry_price"])) * int(pos["qty"]) * int(pos["direction"])
        equity.append(
            {
                "t": ts.isoformat(),
                "nav": nav_realized + unrealized,
                "realized_nav": nav_realized,
                "unrealized": unrealized,
                "open_positions": len(positions),
            }
        )

    # Force close leftovers on the last available bar.
    if positions:
        last_by_symbol = df.sort_values("t").groupby("symbol").tail(1).set_index("symbol")
        for pos in positions:
            if pos["symbol"] not in last_by_symbol.index:
                continue
            row = last_by_symbol.loc[pos["symbol"]]
            direction = int(pos["direction"])
            exit_px = float(row["close"]) * (1.0 - direction * cfg.slippage_bps / 10000.0)
            qty = int(pos["qty"])
            gross = (exit_px - float(pos["entry_price"])) * qty * direction
            fees = (float(pos["entry_price"]) + exit_px) * qty * cfg.fee_bps / 10000.0
            pnl = gross - fees
            nav_realized += pnl
            trades.append(
                {
                    "symbol": pos["symbol"],
                    "entry_t": pos["entry_t"].isoformat(),
                    "exit_t": row["t"].isoformat(),
                    "side": "long" if direction == 1 else "short",
                    "entry_price": float(pos["entry_price"]),
                    "exit_price": float(exit_px),
                    "qty": qty,
                    "notional": float(pos["entry_price"]) * qty,
                    "pnl": pnl,
                    "pnl_bps": pnl / max(float(pos["entry_price"]) * qty, 1.0) * 10000.0,
                    "hold_minutes": int((row["t"] - pos["entry_t"]) / pd.Timedelta(minutes=1)),
                    "exit_reason": "final_flat",
                    "signal": pos["signal"],
                }
            )

    equity_df = pd.DataFrame(equity)
    trades_df = pd.DataFrame(trades)
    if equity_df.empty:
        equity_df = pd.DataFrame([{"t": df["t"].min().isoformat(), "nav": initial_capital}])
    equity_df["t_dt"] = pd.to_datetime(equity_df["t"], utc=True)
    equity_df["peak"] = equity_df["nav"].cummax()
    equity_df["drawdown_pct"] = (equity_df["nav"] / equity_df["peak"] - 1.0) * 100.0

    final_nav = float(equity_df["nav"].iloc[-1])
    unique_days = max(1, df["date"].nunique())
    total_return_pct = (final_nav / initial_capital - 1.0) * 100.0
    monthly_pct = ((final_nav / initial_capital) ** (21.0 / unique_days) - 1.0) * 100.0 if final_nav > 0 else -100.0
    win_rate = float((trades_df["pnl"] > 0).mean() * 100.0) if not trades_df.empty else 0.0
    profit_factor = 0.0
    if not trades_df.empty:
        gross_win = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum())
        gross_loss = abs(float(trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()))
        profit_factor = gross_win / gross_loss if gross_loss > 0 else math.inf

    return {
        "config": asdict(cfg),
        "summary": {
            "strategy": cfg.name,
            "initial_capital": initial_capital,
            "final_nav": final_nav,
            "return_pct": total_return_pct,
            "monthly_pct": monthly_pct,
            "max_drawdown_pct": float(equity_df["drawdown_pct"].min()),
            "trades": int(len(trades_df)),
            "trading_days": int(unique_days),
            "trades_per_day": float(len(trades_df) / unique_days),
            "win_rate_pct": win_rate,
            "profit_factor": profit_factor,
            "avg_hold_minutes": float(trades_df["hold_minutes"].mean()) if not trades_df.empty else 0.0,
            "worst_trade": float(trades_df["pnl"].min()) if not trades_df.empty else 0.0,
            "best_trade": float(trades_df["pnl"].max()) if not trades_df.empty else 0.0,
        },
        "equity": equity_df,
        "trades": trades_df,
    }


def make_report(results: list[dict[str, Any]], out_dir: Path, bars: pd.DataFrame) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = pd.DataFrame([r["summary"] for r in results]).sort_values("return_pct", ascending=False)
    summaries.to_csv(out_dir / "summary.csv", index=False)
    bars.to_parquet(out_dir / "minute_bars.parquet", index=False)

    for r in results:
        name = r["summary"]["strategy"]
        r["equity"].drop(columns=["t_dt"], errors="ignore").to_csv(out_dir / f"{name}_equity.csv", index=False)
        r["trades"].to_csv(out_dir / f"{name}_trades.csv", index=False)
        (out_dir / f"{name}_config.json").write_text(json.dumps(r["config"], indent=2, default=str))

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Equity", "Drawdown", "Trades By Symbol P&L"),
    )
    colors = ["#16a085", "#2980b9", "#c0392b", "#8e44ad"]
    for i, r in enumerate(results):
        eq = r["equity"]
        name = r["summary"]["strategy"]
        fig.add_trace(
            go.Scatter(
                x=eq["t_dt"],
                y=eq["nav"],
                mode="lines",
                name=f"{name} ${r['summary']['final_nav']:,.0f}",
                line=dict(width=2, color=colors[i % len(colors)]),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=eq["t_dt"],
                y=eq["drawdown_pct"],
                mode="lines",
                name=f"{name} DD",
                showlegend=False,
                line=dict(width=1.5, color=colors[i % len(colors)]),
            ),
            row=2,
            col=1,
        )

    best_name = str(summaries.iloc[0]["strategy"]) if not summaries.empty else results[0]["summary"]["strategy"]
    best_trades = next(r["trades"] for r in results if r["summary"]["strategy"] == best_name)
    if not best_trades.empty:
        by_symbol = best_trades.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
        fig.add_trace(
            go.Bar(x=by_symbol.index, y=by_symbol.values, name=f"{best_name} symbol P&L", marker_color="#2ecc71"),
            row=3,
            col=1,
        )
    fig.update_layout(height=950, template="plotly_white", title="Valk Demo: Intraday High-Turnover Test")
    fig.update_yaxes(title_text="NAV ($)", row=1, col=1)
    fig.update_yaxes(title_text="DD (%)", row=2, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=3, col=1)

    summary_html = summaries.to_html(index=False, float_format=lambda x: f"{x:,.2f}")
    trade_tables = []
    for r in results:
        tr = r["trades"].copy()
        if not tr.empty:
            tr = tr.sort_values("pnl").head(20)
        trade_tables.append(f"<h2>{r['summary']['strategy']} worst trades</h2>{tr.to_html(index=False, float_format=lambda x: f'{x:,.2f}')}")

    html = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Valk HFT Demo</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif; margin: 24px; color: #111; }}
    h1 {{ margin-bottom: 4px; }}
    .muted {{ color: #555; }}
    table {{ border-collapse: collapse; width: 100%; margin: 14px 0 28px; font-size: 13px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    th {{ background: #f4f6f8; }}
    .warn {{ background: #fff6e5; border-left: 4px solid #e67e22; padding: 10px 12px; }}
  </style>
</head>
<body>
  <h1>Valk Demo: Intraday High-Turnover Test</h1>
  <p class="muted">No live orders. Uses Alpaca IEX 1-minute bars. Tests stock/ETF scalping, not options condors.</p>
  <div class="warn">This is HFT-ish retail paper trading. True HFT is not possible through Alpaca REST latency. The goal is more frequent entries, tight exits, and local evidence before changing live Valk.</div>
  <h2>Summary</h2>
  {summary_html}
  {fig.to_html(include_plotlyjs="cdn", full_html=False)}
  {''.join(trade_tables)}
</body>
</html>
"""
    report = out_dir / "valk_hft_demo_report.html"
    report.write_text(html)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7, help="Calendar days of minute bars to fetch")
    parser.add_argument("--capital", type=float, default=100_000.0)
    parser.add_argument("--symbols", default=None, help="Comma list; default uses config symbols")
    parser.add_argument("--feed", default="iex")
    parser.add_argument("--open-report", action="store_true")
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)
    symbols = sorted({s for cfg in CONFIGS for s in cfg.symbols})
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = RUNS_DIR / f"valk_hft_demo_{stamp}"
    print(f"Fetching {len(symbols)} symbols, {args.days} calendar days, feed={args.feed}")
    bars = add_features(fetch_minute_bars(symbols, start=start, end=end, feed=args.feed))
    print(f"Bars: {len(bars):,} | trading days: {bars['date'].nunique()} | {bars['t'].min()} to {bars['t'].max()}")

    results = []
    for cfg in CONFIGS:
        cfg_symbols = tuple(s for s in cfg.symbols if s in symbols)
        cfg = StrategyConfig(**{**asdict(cfg), "symbols": cfg_symbols})
        subset = bars[bars["symbol"].isin(cfg.symbols)].copy()
        print(f"Running {cfg.name}...")
        result = run_strategy(subset, cfg, args.capital)
        results.append(result)
        s = result["summary"]
        print(
            f"  {s['return_pct']:+.2f}% total | {s['monthly_pct']:+.2f}% monthly | "
            f"DD {s['max_drawdown_pct']:.2f}% | trades {s['trades']} ({s['trades_per_day']:.1f}/day)"
        )

    report = make_report(results, out_dir, bars)
    print(f"\nReport: {report}")
    print(f"Summary: {out_dir / 'summary.csv'}")
    if args.open_report:
        os.system(f"open {str(report)!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
