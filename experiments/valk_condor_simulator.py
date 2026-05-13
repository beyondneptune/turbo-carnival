"""Research-only Black-Scholes iron-condor path simulator for Valk.

This is not official paper_v1 evidence and does not synthesize option-chain
prices for the frozen evidence run. It is a local risk lens for an existing or
candidate iron condor: mark the spread through an underlying price path, hold
IV constant unless a path column is supplied, and expose P&L plus Greeks.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
RUNS = REPO / "experiments" / "runs"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from features.greeks import Greeks, Leg, aggregate_position_greeks, bs_price, compute_greeks  # noqa: E402


@dataclass(frozen=True)
class IronCondorSpec:
    long_put: float
    short_put: float
    short_call: float
    long_call: float
    quantity: int
    entry_credit: float | None = None
    multiplier: int = 100
    underlying: str = "SPY"

    def __post_init__(self) -> None:
        if not (self.long_put < self.short_put < self.short_call < self.long_call):
            raise ValueError(
                "strikes must satisfy long_put < short_put < short_call < long_call "
                f"(got {self.long_put}, {self.short_put}, {self.short_call}, {self.long_call})"
            )
        if self.quantity <= 0:
            raise ValueError(f"quantity must be positive, got {self.quantity}")
        if self.multiplier <= 0:
            raise ValueError(f"multiplier must be positive, got {self.multiplier}")
        if self.entry_credit is not None and self.entry_credit < 0:
            raise ValueError(f"entry_credit must be non-negative, got {self.entry_credit}")

    @property
    def put_width(self) -> float:
        return self.short_put - self.long_put

    @property
    def call_width(self) -> float:
        return self.long_call - self.short_call

    @property
    def max_spread_width(self) -> float:
        return max(self.put_width, self.call_width)


@dataclass(frozen=True)
class CondorMark:
    spot: float
    days_to_expiry: float
    iv: float
    debit_to_close: float
    pnl_usd: float
    terminal_pnl_usd: float
    delta: float
    gamma: float
    theta_per_day: float
    vega_per_vol_point: float
    rho_per_100_rate_bps: float


def _year_fraction(days_to_expiry: float) -> float:
    return max(float(days_to_expiry), 0.0) / 365.0


def _intrinsic(option_type: str, spot: float, strike: float) -> float:
    if option_type == "call":
        return max(spot - strike, 0.0)
    if option_type == "put":
        return max(strike - spot, 0.0)
    raise ValueError(f"option_type must be 'call' or 'put', got {option_type!r}")


def option_mark(
    option_type: str,
    spot: float,
    strike: float,
    days_to_expiry: float,
    iv: float,
    risk_free_rate: float,
    dividend_yield: float,
) -> float:
    """Black-Scholes option mark, falling back to intrinsic at expiry."""
    if spot <= 0:
        raise ValueError(f"spot must be positive, got {spot}")
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike}")
    if days_to_expiry <= 0 or iv <= 0:
        return _intrinsic(option_type, spot, strike)
    return bs_price(
        option_type,
        spot,
        strike,
        _year_fraction(days_to_expiry),
        risk_free_rate,
        iv,
        dividend_yield,
    )


def condor_debit_to_close(
    spec: IronCondorSpec,
    spot: float,
    days_to_expiry: float,
    iv: float,
    risk_free_rate: float = 0.045,
    dividend_yield: float = 0.0,
) -> float:
    """Per-condor debit required to close the short iron condor at model marks."""
    long_put = option_mark("put", spot, spec.long_put, days_to_expiry, iv, risk_free_rate, dividend_yield)
    short_put = option_mark("put", spot, spec.short_put, days_to_expiry, iv, risk_free_rate, dividend_yield)
    short_call = option_mark("call", spot, spec.short_call, days_to_expiry, iv, risk_free_rate, dividend_yield)
    long_call = option_mark("call", spot, spec.long_call, days_to_expiry, iv, risk_free_rate, dividend_yield)
    return float(short_put + short_call - long_put - long_call)


def terminal_debit_to_close(spec: IronCondorSpec, spot: float) -> float:
    """Per-condor expiry settlement debit."""
    put_spread = max(spec.short_put - spot, 0.0) - max(spec.long_put - spot, 0.0)
    call_spread = max(spot - spec.short_call, 0.0) - max(spot - spec.long_call, 0.0)
    return float(put_spread + call_spread)


def max_loss_usd(spec: IronCondorSpec, entry_credit: float) -> float:
    """Worst-case dollar loss for one-side breach, net of entry credit."""
    loss_per_condor = max(spec.max_spread_width - entry_credit, 0.0)
    return float(loss_per_condor * spec.multiplier * spec.quantity)


def terminal_pnl_usd(spec: IronCondorSpec, spot: float, entry_credit: float) -> float:
    debit = terminal_debit_to_close(spec, spot)
    return float((entry_credit - debit) * spec.multiplier * spec.quantity)


def condor_greeks(
    spec: IronCondorSpec,
    spot: float,
    days_to_expiry: float,
    iv: float,
    risk_free_rate: float = 0.045,
    dividend_yield: float = 0.0,
) -> Greeks:
    """Signed position-level Greeks for the open short iron condor."""
    if days_to_expiry <= 0 or iv <= 0:
        return Greeks(0.0, 0.0, 0.0, 0.0, 0.0)
    T = _year_fraction(days_to_expiry)
    leg_defs = [
        ("put", "long", spec.long_put),
        ("put", "short", spec.short_put),
        ("call", "short", spec.short_call),
        ("call", "long", spec.long_call),
    ]
    legs: list[Leg] = []
    for option_type, direction, strike in leg_defs:
        g = compute_greeks(option_type, spot, strike, T, risk_free_rate, iv, dividend_yield)
        legs.append(
            Leg(
                instrument_type=option_type,
                direction=direction,
                quantity=spec.quantity,
                multiplier=spec.multiplier,
                delta=g.delta,
                gamma=g.gamma,
                theta=g.theta,
                vega=g.vega,
                rho=g.rho,
            )
        )
    return aggregate_position_greeks(legs)


def mark_condor(
    spec: IronCondorSpec,
    spot: float,
    days_to_expiry: float,
    iv: float,
    entry_credit: float,
    risk_free_rate: float = 0.045,
    dividend_yield: float = 0.0,
) -> CondorMark:
    debit = condor_debit_to_close(spec, spot, days_to_expiry, iv, risk_free_rate, dividend_yield)
    pnl = (entry_credit - debit) * spec.multiplier * spec.quantity
    greeks = condor_greeks(spec, spot, days_to_expiry, iv, risk_free_rate, dividend_yield)
    return CondorMark(
        spot=float(spot),
        days_to_expiry=float(days_to_expiry),
        iv=float(iv),
        debit_to_close=debit,
        pnl_usd=float(pnl),
        terminal_pnl_usd=terminal_pnl_usd(spec, spot, entry_credit),
        delta=float(greeks.delta),
        gamma=float(greeks.gamma),
        theta_per_day=float(greeks.theta / 365.0),
        vega_per_vol_point=float(greeks.vega / 100.0),
        rho_per_100_rate_bps=float(greeks.rho * 0.01),
    )


def _normalise_price_path(price_path: pd.DataFrame | pd.Series, price_col: str) -> pd.DataFrame:
    if isinstance(price_path, pd.Series):
        df = price_path.rename("spot").to_frame().reset_index()
        if "index" in df.columns:
            df = df.rename(columns={"index": "t"})
        return df

    if price_col not in price_path.columns:
        raise ValueError(f"price path missing price_col={price_col!r}")
    df = price_path.copy()
    df["spot"] = df[price_col].astype(float)
    if "t" not in df.columns:
        df["t"] = np.arange(len(df), dtype=float)
    return df


def _iv_for_row(df: pd.DataFrame, i: int, iv: float | list[float] | pd.Series, iv_col: str | None) -> float:
    if iv_col and iv_col in df.columns:
        return float(df.iloc[i][iv_col])
    if isinstance(iv, pd.Series):
        return float(iv.iloc[i])
    if isinstance(iv, list):
        return float(iv[i])
    return float(iv)


def simulate_condor_path(
    price_path: pd.DataFrame | pd.Series,
    spec: IronCondorSpec,
    dte_at_entry: float,
    iv: float | list[float] | pd.Series,
    *,
    risk_free_rate: float = 0.045,
    dividend_yield: float = 0.0,
    price_col: str = "close",
    iv_col: str | None = None,
    bars_per_day: int = 390,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mark a condor over a price path.

    If timestamps are present in column `t`, DTE decays by actual elapsed
    calendar time. Without timestamps, DTE decays by row count / bars_per_day.
    """
    if dte_at_entry < 0:
        raise ValueError(f"dte_at_entry must be non-negative, got {dte_at_entry}")
    df = _normalise_price_path(price_path, price_col)
    if df.empty:
        raise ValueError("price_path is empty")

    t = pd.to_datetime(df["t"], utc=True, errors="coerce")
    has_real_time = bool(t.notna().all())
    if has_real_time:
        elapsed_days = (t - t.iloc[0]).dt.total_seconds().to_numpy(dtype=float) / 86400.0
    else:
        elapsed_days = np.arange(len(df), dtype=float) / float(bars_per_day)

    first_iv = _iv_for_row(df, 0, iv, iv_col)
    model_entry_debit = condor_debit_to_close(
        spec,
        float(df.iloc[0]["spot"]),
        dte_at_entry,
        first_iv,
        risk_free_rate,
        dividend_yield,
    )
    entry_credit = float(spec.entry_credit if spec.entry_credit is not None else model_entry_debit)

    rows = []
    for i, row in df.reset_index(drop=True).iterrows():
        spot = float(row["spot"])
        row_iv = _iv_for_row(df, i, iv, iv_col)
        dte = max(float(dte_at_entry) - float(elapsed_days[i]), 0.0)
        mark = mark_condor(
            spec,
            spot,
            dte,
            row_iv,
            entry_credit,
            risk_free_rate,
            dividend_yield,
        )
        out = asdict(mark)
        out["t"] = row["t"]
        rows.append(out)

    result = pd.DataFrame(rows)
    result = result[[
        "t", "spot", "days_to_expiry", "iv", "debit_to_close", "pnl_usd",
        "terminal_pnl_usd", "delta", "gamma", "theta_per_day",
        "vega_per_vol_point", "rho_per_100_rate_bps",
    ]]

    summary = {
        "underlying": spec.underlying,
        "quantity": spec.quantity,
        "strikes": {
            "long_put": spec.long_put,
            "short_put": spec.short_put,
            "short_call": spec.short_call,
            "long_call": spec.long_call,
        },
        "entry_credit": entry_credit,
        "model_entry_debit": model_entry_debit,
        "max_loss_usd": max_loss_usd(spec, entry_credit),
        "start_pnl_usd": float(result["pnl_usd"].iloc[0]),
        "end_pnl_usd": float(result["pnl_usd"].iloc[-1]),
        "min_pnl_usd": float(result["pnl_usd"].min()),
        "max_pnl_usd": float(result["pnl_usd"].max()),
        "end_debit_to_close": float(result["debit_to_close"].iloc[-1]),
        "end_terminal_pnl_usd": float(result["terminal_pnl_usd"].iloc[-1]),
        "end_greeks": {
            "delta": float(result["delta"].iloc[-1]),
            "gamma": float(result["gamma"].iloc[-1]),
            "theta_per_day": float(result["theta_per_day"].iloc[-1]),
            "vega_per_vol_point": float(result["vega_per_vol_point"].iloc[-1]),
            "rho_per_100_rate_bps": float(result["rho_per_100_rate_bps"].iloc[-1]),
        },
    }
    return result, summary


def load_latest_underlying_bars(symbol: str, *, latest_date_only: bool = True) -> pd.DataFrame:
    paths = sorted(RUNS.glob("valk_hft_demo_*/minute_bars.parquet"))
    if not paths:
        raise FileNotFoundError("No cached minute_bars.parquet found.")
    bars = pd.read_parquet(paths[-1])
    bars = bars[bars["symbol"] == symbol].sort_values("t").reset_index(drop=True)
    if bars.empty:
        raise ValueError(f"No bars found for symbol={symbol!r} in {paths[-1]}")
    if latest_date_only and "date" in bars.columns:
        latest_date = max(bars["date"])
        bars = bars[bars["date"] == latest_date].reset_index(drop=True)
    return bars


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def load_condor_spec_json(path: Path) -> tuple[IronCondorSpec, float | None]:
    data = json.loads(path.read_text())
    if data.get("kind") != "iron_condor":
        raise ValueError(f"{path} is not an iron_condor spec")
    strikes = data.get("strikes", data)
    spec = IronCondorSpec(
        long_put=float(strikes["long_put"]),
        short_put=float(strikes["short_put"]),
        short_call=float(strikes["short_call"]),
        long_call=float(strikes["long_call"]),
        quantity=int(data.get("qty", data.get("quantity", 1))),
        entry_credit=float(data["entry_credit"]) if data.get("entry_credit") is not None else None,
        underlying=data.get("underlying", "SPY"),
    )
    dte = data.get("dte_at_entry")
    return spec, (float(dte) if dte is not None else None)


def main() -> int:
    parser = argparse.ArgumentParser(description="Research-only Valk iron-condor BS simulator.")
    parser.add_argument("--spec-json", default=None, help="Optional kind=iron_condor JSON spec to load.")
    parser.add_argument("--symbol", default="SPY")
    parser.add_argument("--long-put", type=float, default=691.0)
    parser.add_argument("--short-put", type=float, default=696.0)
    parser.add_argument("--short-call", type=float, default=790.0)
    parser.add_argument("--long-call", type=float, default=795.0)
    parser.add_argument("--qty", type=int, default=44)
    parser.add_argument("--entry-credit", type=float, default=0.54)
    parser.add_argument("--dte", type=float, default=None)
    parser.add_argument("--iv", type=float, default=0.18)
    parser.add_argument("--risk-free-rate", type=float, default=0.045)
    parser.add_argument("--dividend-yield", type=float, default=0.0)
    parser.add_argument(
        "--all-cached-days",
        action="store_true",
        help="Use the full cached history instead of only the latest trading date.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.spec_json:
        spec, spec_dte = load_condor_spec_json(Path(args.spec_json))
    else:
        spec, spec_dte = IronCondorSpec(
            long_put=args.long_put,
            short_put=args.short_put,
            short_call=args.short_call,
            long_call=args.long_call,
            quantity=args.qty,
            entry_credit=args.entry_credit,
            underlying=args.symbol,
        ), None
    dte = float(args.dte if args.dte is not None else (spec_dte if spec_dte is not None else 32.0))
    bars = load_latest_underlying_bars(spec.underlying, latest_date_only=not args.all_cached_days)
    path, summary = simulate_condor_path(
        bars,
        spec,
        dte,
        args.iv,
        risk_free_rate=args.risk_free_rate,
        dividend_yield=args.dividend_yield,
    )

    out_dir = Path(args.out) if args.out else RUNS / f"valk_condor_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path.to_csv(out_dir / "path.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))

    print(json.dumps(summary, indent=2, default=_json_default))
    print(f"\nWrote {out_dir / 'path.csv'}")
    print(f"Wrote {out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
