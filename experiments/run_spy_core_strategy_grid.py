"""SPY core proxy strategy grid from 2019 to current.

This is an offline research experiment, not official evidence.

Why proxy:
- The repo does not have complete contract-level SPY option bid/ask chains from
  2019 through current.
- This runner uses SPY daily closes plus an IV proxy to Black-Scholes price a
  standard iron-condor structure. It is useful for comparing strategy shapes,
  cadence, and risk. It is not an execution-quality fill record.

Outputs:
- experiments/runs/spy_core_30_strategy_grid_2019_2026/summary.csv
- experiments/runs/spy_core_30_strategy_grid_2019_2026/trades.csv
- experiments/runs/spy_core_30_strategy_grid_2019_2026/daily_nav.csv
- experiments/runs/spy_core_30_strategy_grid_2019_2026/equity_curves_30_strategies.html
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.greeks import bs_delta, bs_price  # noqa: E402

DEFAULT_OUTPUT_DIR = REPO_ROOT / "experiments" / "runs" / "spy_core_30_strategy_grid_2019_2026"
DEFAULT_HF_IV_PATH = (
    REPO_ROOT
    / "data"
    / "offline_context"
    / "huggingface_options_iv_sp500"
    / "processed"
    / "data_IV_USA.parquet"
)

EntryMode = Literal["single_reentry", "daily_ladder", "mwf_ladder"]
GateName = Literal[
    "none",
    "vrp_positive",
    "iv_rank_50",
    "high_iv_volume",
    "calm_market",
    "vix_calm_only",
    "ultra_calm",
    "macro_aware_v3",         # NEW: vrp + vix term structure + skip Fed weeks
    "macro_aware_strict",     # NEW: above + steep yield curve filter
]


@dataclass(frozen=True)
class StrategySpec:
    name: str
    entry_mode: EntryMode
    gate: GateName
    target_delta: float
    target_dte: int
    wing_width: float
    profit_take: float = 0.50
    stop_loss_credit_mult: float = 2.00
    close_dte: int = 21
    min_credit: float = 0.20


@dataclass
class OpenCondor:
    strategy: str
    entry_date: pd.Timestamp
    expiry: pd.Timestamp
    entry_spy: float
    entry_iv: float
    short_call: float
    long_call: float
    short_put: float
    long_put: float
    entry_credit: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default=None, help="Exclusive end date for yfinance. Default: tomorrow/current available.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--hf-iv-path", type=Path, default=DEFAULT_HF_IV_PATH)
    parser.add_argument(
        "--spec-json",
        type=Path,
        default=None,
        help="Optional JSON file containing council-proposed strategy specs.",
    )
    parser.add_argument("--risk-free-rate", type=float, default=0.02)
    parser.add_argument("--dividend-yield", type=float, default=0.015)
    parser.add_argument("--commission-per-leg", type=float, default=0.65)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    market = load_market_data(
        start=args.start,
        end=args.end,
        hf_iv_path=args.hf_iv_path,
    )
    specs = load_strategy_specs(args.spec_json) if args.spec_json else build_strategy_specs()

    all_nav: list[pd.DataFrame] = []
    all_trades: list[pd.DataFrame] = []
    summaries: list[dict[str, object]] = []
    for spec in specs:
        nav, trades, summary = simulate_strategy(
            market,
            spec,
            risk_free_rate=args.risk_free_rate,
            dividend_yield=args.dividend_yield,
            commission_per_leg=args.commission_per_leg,
        )
        all_nav.append(nav.assign(strategy=spec.name))
        if not trades.empty:
            all_trades.append(trades.assign(strategy=spec.name))
        summaries.append(summary)

    nav_df = pd.concat(all_nav, ignore_index=True)
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    summary_df = pd.DataFrame(summaries).sort_values(
        ["final_nav", "max_drawdown_pct"],
        ascending=[False, False],
    )

    nav_df.to_csv(output_dir / "daily_nav.csv", index=False)
    trades_df.to_csv(output_dir / "trades.csv", index=False)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "start": str(market.index.min().date()),
                "end": str(market.index.max().date()),
                "strategies": [asdict(s) for s in specs],
                "summary": summary_df.to_dict(orient="records"),
                "notes": [
                    "Proxy only: SPY daily close plus IV proxy, not real option bid/ask chains.",
                    "IV source prefers saved HuggingFace SPY ATM_IV when available; otherwise uses VIX close / 100.",
                    "Aggressive ladder variants allow overlapping open trades.",
                ],
            },
            indent=2,
            default=str,
        )
    )

    html_path = output_dir / "equity_curves_30_strategies.html"
    write_chart(nav_df, trades_df, summary_df, html_path)

    print(f"wrote {output_dir}")
    print(f"chart {html_path}")
    print(summary_df.head(10).to_string(index=False))
    return 0


def load_market_data(*, start: str, end: str | None, hf_iv_path: Path) -> pd.DataFrame:
    import yfinance as yf

    yf_data = yf.download(
        ["SPY", "^VIX", "^VIX3M", "^TNX", "^FVX"],
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        threads=False,
    )
    if yf_data.empty:
        raise RuntimeError("yfinance returned no SPY/VIX data")

    close = yf_data["Close"]
    volume = yf_data["Volume"]
    cols = {
        "spy_close": close["SPY"].astype(float),
        "vix_close": close["^VIX"].astype(float),
        "spy_volume": volume["SPY"].astype(float),
    }
    # Optional cross-asset features — Phase 2 NIM panel said these mattered.
    if "^VIX3M" in close.columns:
        cols["vix3m_close"] = close["^VIX3M"].astype(float)
    if "^TNX" in close.columns:
        cols["tnx_yield"] = close["^TNX"].astype(float)  # 10y treasury yield (%)
    if "^FVX" in close.columns:
        cols["fvx_yield"] = close["^FVX"].astype(float)  # 5y treasury yield (%)
    frame = pd.DataFrame(cols).dropna(subset=["spy_close", "vix_close"])
    frame.index = pd.to_datetime(frame.index).tz_localize(None)

    hf_iv = load_hf_spy_iv(hf_iv_path)
    frame = frame.join(hf_iv, how="left")
    frame["iv"] = frame["hf_atm_iv"].fillna(frame["vix_close"] / 100.0)
    frame["iv_source"] = np.where(frame["hf_atm_iv"].notna(), "hf_atm_iv", "vix_proxy")
    frame["iv"] = frame["iv"].clip(lower=0.05, upper=1.25)

    returns = frame["spy_close"].pct_change()
    frame["rv10"] = returns.rolling(10, min_periods=10).std() * math.sqrt(252)
    frame["rv20"] = returns.rolling(20, min_periods=20).std() * math.sqrt(252)
    frame["volume_ma20"] = frame["spy_volume"].rolling(20, min_periods=20).mean()
    frame["volume_ratio"] = frame["spy_volume"] / frame["volume_ma20"]
    frame["ma20"] = frame["spy_close"].rolling(20, min_periods=20).mean()
    frame["trend20"] = frame["spy_close"] / frame["ma20"] - 1.0
    frame["vrp20"] = frame["iv"] - frame["rv20"]
    frame["iv_rank_252"] = rolling_percentile(frame["iv"], 252)
    # NIM post-mortem panel finding: VIX-ROC + VIX-absolute + SPY drawdown
    # were the most-mentioned missing signals across 12 worst-loss dates.
    frame["vix_5d_roc"] = frame["vix_close"].pct_change(5)
    frame["vix_10d_roc"] = frame["vix_close"].pct_change(10)
    frame["spy_5d_return"] = frame["spy_close"].pct_change(5)
    frame["spy_drawdown_20d"] = (
        frame["spy_close"] / frame["spy_close"].rolling(20, min_periods=20).max() - 1.0
    )
    # NEW: VIX term structure (Phase 2 NIM panel — 3x mentions)
    if "vix3m_close" in frame.columns:
        frame["vix_term_ratio"] = frame["vix_close"] / frame["vix3m_close"]
        frame["vix_term_inverted"] = (frame["vix_term_ratio"] > 1.05).astype(int)
    # NEW: Treasury yield curve slope (Phase 2 NIM panel — 3x mentions)
    if "tnx_yield" in frame.columns and "fvx_yield" in frame.columns:
        frame["yc_slope_10y_5y"] = frame["tnx_yield"] - frame["fvx_yield"]
        frame["yc_slope_change_5d"] = frame["yc_slope_10y_5y"].diff(5)
    # NEW: Fed meeting proximity (Phase 2 NIM panel — 3x mentions)
    # Hardcoded approximate FOMC dates 2019-2026. Within ±5 trading days of
    # any FOMC = "fed_week".
    fomc = pd.to_datetime([
        "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19", "2019-07-31",
        "2019-09-18", "2019-10-30", "2019-12-11",
        "2020-01-29", "2020-03-15", "2020-04-29", "2020-06-10", "2020-07-29",
        "2020-09-16", "2020-11-05", "2020-12-16",
        "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16", "2021-07-28",
        "2021-09-22", "2021-11-03", "2021-12-15",
        "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27",
        "2022-09-21", "2022-11-02", "2022-12-14",
        "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26",
        "2023-09-20", "2023-11-01", "2023-12-13",
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31",
        "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30",
        "2025-09-17", "2025-10-29", "2025-12-10",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
    ])
    fed_window = pd.Series(0, index=frame.index)
    for d in fomc:
        # ±5 trading days around each FOMC
        mask = (frame.index >= d - pd.Timedelta(days=7)) & (frame.index <= d + pd.Timedelta(days=2))
        fed_window.loc[mask] = 1
    frame["fed_week"] = fed_window
    frame = frame.dropna(subset=["rv20", "volume_ratio", "trend20", "vrp20"]).copy()
    return frame


def load_hf_spy_iv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["hf_atm_iv"])
    raw = pd.read_parquet(path, columns=["symbol", "date", "ATM_IV"])
    raw = raw[raw["symbol"].astype(str).str.upper().eq("SPY")].copy()
    if raw.empty:
        return pd.DataFrame(columns=["hf_atm_iv"])
    raw["date"] = pd.to_datetime(raw["date"]).dt.tz_localize(None)
    raw["hf_atm_iv"] = pd.to_numeric(raw["ATM_IV"], errors="coerce")
    raw.loc[raw["hf_atm_iv"] > 1.0, "hf_atm_iv"] = raw.loc[raw["hf_atm_iv"] > 1.0, "hf_atm_iv"] / 100.0
    raw = raw.dropna(subset=["hf_atm_iv"]).set_index("date").sort_index()
    return raw[["hf_atm_iv"]]


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def pct_rank(values: np.ndarray) -> float:
        current = values[-1]
        valid = values[~np.isnan(values)]
        if len(valid) < 30:
            return np.nan
        return float((valid <= current).sum() / len(valid))

    return series.rolling(window, min_periods=63).apply(pct_rank, raw=True)


def build_strategy_specs() -> list[StrategySpec]:
    specs: list[StrategySpec] = []

    for mode in ("single_reentry", "daily_ladder"):
        for gate in ("vrp_positive", "iv_rank_50"):
            for dte in (30, 45):
                for delta in (0.10, 0.12, 0.16):
                    specs.append(
                        StrategySpec(
                            name=f"{mode}_{gate}_d{int(delta * 100)}_{dte}d_w5",
                            entry_mode=mode,  # type: ignore[arg-type]
                            gate=gate,  # type: ignore[arg-type]
                            target_delta=delta,
                            target_dte=dte,
                            wing_width=5.0,
                        )
                    )

    for delta in (0.10, 0.12, 0.16):
        specs.append(
            StrategySpec(
                name=f"mwf_ladder_highivvol_d{int(delta * 100)}_45d_w10",
                entry_mode="mwf_ladder",
                gate="high_iv_volume",
                target_delta=delta,
                target_dte=45,
                wing_width=10.0,
            )
        )

    for delta in (0.12, 0.16, 0.20):
        specs.append(
            StrategySpec(
                name=f"aggressive_daily_nogate_d{int(delta * 100)}_30d_w10",
                entry_mode="daily_ladder",
                gate="none",
                target_delta=delta,
                target_dte=30,
                wing_width=10.0,
                min_credit=0.10,
            )
        )

    if len(specs) != 30:
        raise AssertionError(f"expected 30 strategies, got {len(specs)}")
    return specs


def load_strategy_specs(path: Path) -> list[StrategySpec]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_specs = payload.get("strategies", payload) if isinstance(payload, dict) else payload
    if not isinstance(raw_specs, list):
        raise ValueError("spec JSON must be a list or an object with a 'strategies' list")

    specs: list[StrategySpec] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_specs, start=1):
        if not isinstance(raw, dict):
            raise ValueError(f"strategy #{i} must be an object")
        spec = validate_strategy_spec(raw, i)
        if spec.name in seen:
            raise ValueError(f"duplicate strategy name: {spec.name}")
        seen.add(spec.name)
        specs.append(spec)
    if not specs:
        raise ValueError("spec JSON contained no strategies")
    return specs


def validate_strategy_spec(raw: dict[str, object], idx: int) -> StrategySpec:
    name = str(raw.get("name") or f"council_strategy_{idx}")
    name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name).strip("_")
    if not name:
        raise ValueError(f"strategy #{idx} has empty name")

    entry_mode = str(raw.get("entry_mode", "daily_ladder"))
    gate = str(raw.get("gate", "vrp_positive"))
    if entry_mode not in {"single_reentry", "daily_ladder", "mwf_ladder"}:
        raise ValueError(f"{name}: invalid entry_mode {entry_mode!r}")
    if gate not in {"none", "vrp_positive", "iv_rank_50", "high_iv_volume",
                     "calm_market", "vix_calm_only", "ultra_calm",
                     "macro_aware_v3", "macro_aware_strict"}:
        raise ValueError(f"{name}: invalid gate {gate!r}")

    target_delta = float(raw.get("target_delta", 0.12))
    target_dte = int(raw.get("target_dte", 30))
    wing_width = float(raw.get("wing_width", 5.0))
    profit_take = float(raw.get("profit_take", 0.50))
    stop_loss_credit_mult = float(raw.get("stop_loss_credit_mult", 2.00))
    close_dte = int(raw.get("close_dte", 21))
    min_credit = float(raw.get("min_credit", 0.20))

    if not 0.05 <= target_delta <= 0.35:
        raise ValueError(f"{name}: target_delta must be 0.05..0.35")
    if not 14 <= target_dte <= 75:
        raise ValueError(f"{name}: target_dte must be 14..75")
    if not 1.0 <= wing_width <= 25.0:
        raise ValueError(f"{name}: wing_width must be 1..25")
    if not 0.10 <= profit_take <= 0.90:
        raise ValueError(f"{name}: profit_take must be 0.10..0.90")
    if not 0.50 <= stop_loss_credit_mult <= 5.00:
        raise ValueError(f"{name}: stop_loss_credit_mult must be 0.50..5.00")
    if not 1 <= close_dte < target_dte:
        raise ValueError(f"{name}: close_dte must be >=1 and less than target_dte")
    if not 0.01 <= min_credit <= 3.00:
        raise ValueError(f"{name}: min_credit must be 0.01..3.00")

    return StrategySpec(
        name=name,
        entry_mode=entry_mode,  # type: ignore[arg-type]
        gate=gate,  # type: ignore[arg-type]
        target_delta=target_delta,
        target_dte=target_dte,
        wing_width=wing_width,
        profit_take=profit_take,
        stop_loss_credit_mult=stop_loss_credit_mult,
        close_dte=close_dte,
        min_credit=min_credit,
    )


def simulate_strategy(
    market: pd.DataFrame,
    spec: StrategySpec,
    *,
    risk_free_rate: float,
    dividend_yield: float,
    commission_per_leg: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    initial_nav = 10_000.0
    roundtrip_commission = commission_per_leg * 8.0
    open_positions: list[OpenCondor] = []
    realized_pnl = 0.0
    nav_rows: list[dict[str, object]] = []
    trade_rows: list[dict[str, object]] = []

    for dt, row in market.iterrows():
        remaining: list[OpenCondor] = []
        for pos in open_positions:
            debit = price_condor_value(
                spot=float(row["spy_close"]),
                sigma=float(row["iv"]),
                as_of=dt,
                expiry=pos.expiry,
                short_call=pos.short_call,
                long_call=pos.long_call,
                short_put=pos.short_put,
                long_put=pos.long_put,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            )
            dte = max(0, (pos.expiry.date() - dt.date()).days)
            gross_pnl = (pos.entry_credit - debit) * 100.0
            reason = None
            if debit <= pos.entry_credit * (1.0 - spec.profit_take):
                reason = "profit_target"
            elif debit >= pos.entry_credit * (1.0 + spec.stop_loss_credit_mult):
                reason = "stop_loss"
            elif dte <= spec.close_dte:
                reason = "dte_close"
            elif dt == market.index[-1]:
                reason = "end_forced"

            if reason:
                pnl = gross_pnl - roundtrip_commission
                realized_pnl += pnl
                trade_rows.append(
                    {
                        "strategy": spec.name,
                        "entry_date": pos.entry_date.date().isoformat(),
                        "exit_date": dt.date().isoformat(),
                        "expiry": pos.expiry.date().isoformat(),
                        "reason": reason,
                        "entry_spy": pos.entry_spy,
                        "exit_spy": float(row["spy_close"]),
                        "entry_iv": pos.entry_iv,
                        "exit_iv": float(row["iv"]),
                        "short_call": pos.short_call,
                        "long_call": pos.long_call,
                        "short_put": pos.short_put,
                        "long_put": pos.long_put,
                        "entry_credit": pos.entry_credit,
                        "exit_debit": debit,
                        "pnl_usd": pnl,
                    }
                )
            else:
                remaining.append(pos)
        open_positions = remaining

        if should_enter(dt, row, spec, open_positions):
            condor = build_condor(
                dt=dt,
                row=row,
                spec=spec,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            )
            if condor is not None:
                open_positions.append(condor)

        unrealized = 0.0
        for pos in open_positions:
            debit = price_condor_value(
                spot=float(row["spy_close"]),
                sigma=float(row["iv"]),
                as_of=dt,
                expiry=pos.expiry,
                short_call=pos.short_call,
                long_call=pos.long_call,
                short_put=pos.short_put,
                long_put=pos.long_put,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
            )
            unrealized += (pos.entry_credit - debit) * 100.0 - roundtrip_commission

        nav_rows.append(
            {
                "date": dt.date().isoformat(),
                "strategy": spec.name,
                "nav": initial_nav + realized_pnl + unrealized,
                "realized_pnl": realized_pnl,
                "unrealized_pnl": unrealized,
                "open_trades": len(open_positions),
                "spy_close": float(row["spy_close"]),
                "iv": float(row["iv"]),
                "iv_source": row["iv_source"],
                "vrp20": float(row["vrp20"]),
                "iv_rank_252": float(row["iv_rank_252"]) if pd.notna(row["iv_rank_252"]) else np.nan,
            }
        )

    nav = pd.DataFrame(nav_rows)
    trades = pd.DataFrame(trade_rows)
    final_nav = float(nav["nav"].iloc[-1]) if not nav.empty else initial_nav
    max_drawdown_pct = compute_max_drawdown(nav["nav"]) * 100.0 if not nav.empty else 0.0
    daily_returns = nav["nav"].pct_change().dropna() if not nav.empty else pd.Series(dtype=float)
    sharpe = (
        float(daily_returns.mean() / daily_returns.std() * math.sqrt(252))
        if len(daily_returns) > 2 and daily_returns.std() > 0
        else 0.0
    )
    win_rate = float((trades["pnl_usd"] > 0).mean() * 100.0) if not trades.empty else 0.0
    avg_trade = float(trades["pnl_usd"].mean()) if not trades.empty else 0.0
    worst_trade = float(trades["pnl_usd"].min()) if not trades.empty else 0.0
    max_open = int(nav["open_trades"].max()) if not nav.empty else 0

    summary = {
        "strategy": spec.name,
        "entry_mode": spec.entry_mode,
        "gate": spec.gate,
        "target_delta": spec.target_delta,
        "target_dte": spec.target_dte,
        "wing_width": spec.wing_width,
        "trades": int(len(trades)),
        "final_nav": final_nav,
        "return_pct": (final_nav / initial_nav - 1.0) * 100.0,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "win_rate_pct": win_rate,
        "avg_trade_usd": avg_trade,
        "worst_trade_usd": worst_trade,
        "max_open_trades": max_open,
    }
    return nav, trades, summary


def should_enter(
    dt: pd.Timestamp,
    row: pd.Series,
    spec: StrategySpec,
    open_positions: list[OpenCondor],
) -> bool:
    if not entry_day_ok(dt, spec.entry_mode):
        return False
    if spec.entry_mode == "single_reentry" and open_positions:
        return False
    return gate_ok(row, spec.gate)


def entry_day_ok(dt: pd.Timestamp, mode: EntryMode) -> bool:
    if mode in ("single_reentry", "daily_ladder"):
        return True
    if mode == "mwf_ladder":
        return dt.weekday() in (0, 2, 4)
    raise ValueError(f"unknown entry mode {mode!r}")


def gate_ok(row: pd.Series, gate: GateName) -> bool:
    if gate == "none":
        return True
    if gate == "vrp_positive":
        return bool(row["vrp20"] > 0.0 and row["iv"] >= 0.10)
    if gate == "iv_rank_50":
        return bool(pd.notna(row["iv_rank_252"]) and row["iv_rank_252"] >= 0.50)
    if gate == "high_iv_volume":
        return bool(
            pd.notna(row["iv_rank_252"])
            and row["iv_rank_252"] >= 0.55
            and row["volume_ratio"] >= 0.90
            and row["vrp20"] > -0.02
        )
    if gate == "calm_market":
        # vrp_positive + vix not spiking + no fast SPY drawdown
        return bool(
            row["vrp20"] > 0.0
            and row["iv"] >= 0.10
            and row["vix_close"] < 28.0
            and (pd.isna(row["vix_5d_roc"]) or row["vix_5d_roc"] < 0.18)
            and (pd.isna(row["spy_5d_return"]) or row["spy_5d_return"] > -0.04)
        )
    if gate == "vix_calm_only":
        # Just the VIX-based filter — minimal change from existing strategies
        return bool(
            row["vix_close"] < 25.0
            and (pd.isna(row["vix_5d_roc"]) or row["vix_5d_roc"] < 0.20)
        )
    if gate == "ultra_calm":
        # Most aggressive risk-off — stack everything
        return bool(
            row["vrp20"] > 0.0
            and row["iv"] >= 0.12
            and row["vix_close"] < 22.0
            and (pd.isna(row["vix_5d_roc"]) or row["vix_5d_roc"] < 0.10)
            and (pd.isna(row["vix_10d_roc"]) or row["vix_10d_roc"] < 0.15)
            and (pd.isna(row["spy_5d_return"]) or row["spy_5d_return"] > -0.02)
            and (pd.isna(row["spy_drawdown_20d"]) or row["spy_drawdown_20d"] > -0.03)
        )
    if gate == "macro_aware_v3":
        # Phase 2 NIM panel features: VIX term structure + Fed week + vrp
        if not (row["vrp20"] > 0.0 and row["iv"] >= 0.10):
            return False
        if row.get("fed_week", 0) == 1:
            return False  # skip ±5 days around FOMC
        if "vix_term_ratio" in row.index and pd.notna(row["vix_term_ratio"]):
            if row["vix_term_ratio"] > 1.10:  # term structure inverted (vol > 3M vol)
                return False
        return True
    if gate == "macro_aware_strict":
        # macro_aware_v3 plus require VIX term structure in contango (calm)
        # plus steep yield curve (no recession warning)
        if not (row["vrp20"] > 0.0 and row["iv"] >= 0.10):
            return False
        if row.get("fed_week", 0) == 1:
            return False
        if "vix_term_ratio" in row.index and pd.notna(row["vix_term_ratio"]):
            if row["vix_term_ratio"] > 1.0:  # require contango
                return False
        if "yc_slope_10y_5y" in row.index and pd.notna(row["yc_slope_10y_5y"]):
            if row["yc_slope_10y_5y"] < -0.5:  # severely inverted = recession warning
                return False
        return True
    raise ValueError(f"unknown gate {gate!r}")


def build_condor(
    *,
    dt: pd.Timestamp,
    row: pd.Series,
    spec: StrategySpec,
    risk_free_rate: float,
    dividend_yield: float,
) -> OpenCondor | None:
    spot = float(row["spy_close"])
    sigma = float(row["iv"])
    expiry = next_friday(dt + timedelta(days=spec.target_dte))
    t = max((expiry.date() - dt.date()).days / 365.0, 1.0 / 365.0)
    short_call = strike_for_delta("call", spot, t, risk_free_rate, sigma, dividend_yield, spec.target_delta)
    short_put = strike_for_delta("put", spot, t, risk_free_rate, sigma, dividend_yield, spec.target_delta)
    long_call = short_call + spec.wing_width
    long_put = max(1.0, short_put - spec.wing_width)
    credit = price_condor_value(
        spot=spot,
        sigma=sigma,
        as_of=dt,
        expiry=expiry,
        short_call=short_call,
        long_call=long_call,
        short_put=short_put,
        long_put=long_put,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
    )
    max_risk = spec.wing_width - credit
    if not np.isfinite(credit) or credit < spec.min_credit or max_risk <= 0:
        return None
    return OpenCondor(
        strategy=spec.name,
        entry_date=dt,
        expiry=expiry,
        entry_spy=spot,
        entry_iv=sigma,
        short_call=short_call,
        long_call=long_call,
        short_put=short_put,
        long_put=long_put,
        entry_credit=credit,
    )


def next_friday(target: pd.Timestamp) -> pd.Timestamp:
    out = pd.Timestamp(target.date())
    while out.weekday() != 4:
        out += timedelta(days=1)
    return out


def strike_for_delta(
    option_type: Literal["call", "put"],
    spot: float,
    t: float,
    risk_free_rate: float,
    sigma: float,
    dividend_yield: float,
    target_delta: float,
) -> float:
    # Direct inverse of BSM delta; much faster than scanning every strike.
    adjusted = float(np.clip(target_delta * math.exp(dividend_yield * t), 1e-6, 0.999999))
    if option_type == "call":
        d1 = float(norm.ppf(adjusted))
    else:
        d1 = -float(norm.ppf(adjusted))
    exponent = d1 * sigma * math.sqrt(t) - (risk_free_rate - dividend_yield + 0.5 * sigma * sigma) * t
    strike = spot / math.exp(exponent)
    strike = round(strike)
    if option_type == "call":
        strike = max(strike, math.ceil(spot))
    else:
        strike = min(strike, math.floor(spot))

    # A final one-dollar neighborhood check keeps rounded strikes close to the
    # target delta while staying OTM.
    candidates = [float(max(1, strike + offset)) for offset in range(-2, 3)]
    if option_type == "call":
        candidates = [k for k in candidates if k >= spot]
    else:
        candidates = [k for k in candidates if k <= spot]
    if not candidates:
        return float(strike)
    return min(
        candidates,
        key=lambda k: abs(
            abs(bs_delta(option_type, spot, k, t, risk_free_rate, sigma, dividend_yield)) - target_delta
        ),
    )


def price_condor_value(
    *,
    spot: float,
    sigma: float,
    as_of: pd.Timestamp,
    expiry: pd.Timestamp,
    short_call: float,
    long_call: float,
    short_put: float,
    long_put: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> float:
    t = max((expiry.date() - as_of.date()).days / 365.0, 1.0 / 365.0)
    call_short = bs_price("call", spot, short_call, t, risk_free_rate, sigma, dividend_yield)
    call_long = bs_price("call", spot, long_call, t, risk_free_rate, sigma, dividend_yield)
    put_short = bs_price("put", spot, short_put, t, risk_free_rate, sigma, dividend_yield)
    put_long = bs_price("put", spot, long_put, t, risk_free_rate, sigma, dividend_yield)
    return float(call_short + put_short - call_long - put_long)


def compute_max_drawdown(nav: pd.Series) -> float:
    running_high = nav.cummax()
    dd = nav / running_high - 1.0
    return float(dd.min())


def write_chart(nav: pd.DataFrame, trades: pd.DataFrame, summary: pd.DataFrame, out: Path) -> None:
    top = summary.head(10).copy()
    ordered = summary["strategy"].tolist()
    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=[0.58, 0.22, 0.20],
        vertical_spacing=0.10,
        subplot_titles=(
            "All 30 strategy equity curves, $10k starting capital",
            "Top 10 final NAV",
            "Trade count vs max drawdown",
        ),
    )

    palette = [
        "#264653",
        "#2a9d8f",
        "#e9c46a",
        "#f4a261",
        "#e76f51",
        "#457b9d",
        "#1d3557",
        "#a8dadc",
        "#6d597a",
        "#b56576",
    ]
    for idx, strategy in enumerate(ordered):
        sdf = nav[nav["strategy"] == strategy].copy()
        row = summary[summary["strategy"] == strategy].iloc[0]
        width = 3 if idx < 5 else 1.2
        opacity = 0.95 if idx < 5 else 0.45
        fig.add_trace(
            go.Scatter(
                x=sdf["date"],
                y=sdf["nav"],
                mode="lines",
                name=f"{strategy} (${row['final_nav']:,.0f})",
                line=dict(color=palette[idx % len(palette)], width=width),
                opacity=opacity,
                hovertemplate=(
                    f"{strategy}<br>"
                    "Date %{x}<br>NAV $%{y:,.2f}<br>"
                    f"Trades {int(row['trades'])}<br>"
                    f"Max DD {row['max_drawdown_pct']:.1f}%"
                    "<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Bar(
            x=top["strategy"],
            y=top["final_nav"],
            marker_color="#2a9d8f",
            text=[f"${v:,.0f}" for v in top["final_nav"]],
            textposition="outside",
            name="top final NAV",
            hovertemplate="%{x}<br>Final NAV $%{y:,.2f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=summary["trades"],
            y=summary["max_drawdown_pct"],
            mode="markers+text",
            text=[str(i + 1) for i in range(len(summary))],
            textposition="top center",
            marker=dict(
                size=np.clip(summary["return_pct"].to_numpy() / 2.0, 8, 34),
                color=summary["final_nav"],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Final NAV"),
            ),
            name="risk/turnover",
            customdata=np.stack(
                [
                    summary["strategy"],
                    summary["return_pct"],
                    summary["win_rate_pct"],
                    summary["max_open_trades"],
                ],
                axis=-1,
            ),
            hovertemplate=(
                "%{customdata[0]}<br>"
                "Trades %{x}<br>Max DD %{y:.2f}%<br>"
                "Return %{customdata[1]:.2f}%<br>"
                "Win rate %{customdata[2]:.1f}%<br>"
                "Max open %{customdata[3]}<extra></extra>"
            ),
        ),
        row=3,
        col=1,
    )

    best = summary.iloc[0]
    subtitle = (
        f"Best: {best['strategy']} | ${best['final_nav']:,.2f} | "
        f"{best['return_pct']:+.2f}% | {int(best['trades'])} trades | "
        f"max DD {best['max_drawdown_pct']:.2f}%"
    )
    fig.update_layout(
        title=dict(
            text=f"SPY Core Strategy Grid: 2019 To Current Proxy<br><sup>{subtitle}</sup>",
            x=0.02,
            xanchor="left",
        ),
        width=1800,
        height=1200,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="v", yanchor="top", y=0.98, xanchor="left", x=1.01),
        margin=dict(l=80, r=420, t=130, b=90),
    )
    fig.update_yaxes(title_text="NAV ($)", tickprefix="$", row=1, col=1)
    fig.update_yaxes(title_text="Final NAV ($)", tickprefix="$", row=2, col=1)
    fig.update_yaxes(title_text="Max drawdown (%)", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_xaxes(tickangle=25, row=2, col=1)
    fig.update_xaxes(title_text="Closed trades", row=3, col=1)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out, include_plotlyjs="cdn")


if __name__ == "__main__":
    raise SystemExit(main())
