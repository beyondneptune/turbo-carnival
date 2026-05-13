"""iter_18 backtest: US/HK dual-listing parity arbitrage.

Strategy (per /Users/weeoto/turbo-carnival/research/candidates/iter_18_us_hk_dual_listing_arbitrage.md):
  parity_USD = HK_close_HKD * ratio * USD_per_HKD     # USD_per_HKD = 1/7.80 ≈ 0.1282
  deviation_bps = 10000 * (ADR_USD - parity_USD) / parity_USD

  Pairs (US ADR : HK underlying : ADR ratio = HK shares per ADR):
    BABA  / 9988.HK / 8
    JD    / 9618.HK / 2
    BIDU  / 9888.HK / 8
    NTES  / 9999.HK / 25
    TCOM  / 9961.HK / 1

  Entry:  |deviation_bps| > 80
  Exit:   |deviation_bps| < 20  OR  holding > 5 trading days
  Stop:   |deviation_bps| > 250

  Direction:
    deviation > 0 (ADR rich vs HK)  -> short ADR + long HK underlying
    deviation < 0 (HK rich vs ADR)  -> long  ADR + short HK underlying

  Costs (round-trip per leg):
    5 bps per leg per side -> total 4 sides * 5 = 20 bps round-trip per pair-trade
    Spec also mentions "total round-trip ~30 bps" including stamp/FX/spread; we'll book
    the per-leg-per-side 5 bps and report both gross and net (net subtracts 20 bps total
    round-trip; we additionally report a "net30" view subtracting 30 bps for the spec-suggested
    fuller friction budget).

CAVEAT (intentionally honest): daily Yahoo "Close" for *.HK is the HK 16:00 HKT close,
which is ~10-12 hours BEFORE the US Close on the same calendar date. So the deviation
computed here mixes (a) true cross-venue arb mispricing and (b) the pure information drift
that occurs during the 10h US-trading-day window. A real implementation would sample at
matched gap-window snapshots (HK pre-open vs prior US close, and US pre-open vs prior HK
close) per spec §3. This backtest uses *contemporaneous calendar-day closes* as a coarse
proxy. The deviation magnitudes are realistic-ish; the persistence and convergence dynamics
are biased by daily aggregation. Treat results as a coarse first pass, NOT as production
evidence.

Period: 2020-01-01 -> today (post HK secondary listings; some pairs only become eligible
from 2020-Q2/Q3 or 2021).

Output:
  CSV: research/results/iter_18_trades.csv  (one row per closed round-trip)
  JSON: research/results/iter_18_summary.json with the keys the brief requested.

PASS rule (per brief STEP 5): mean_bps_net >= 25 AND total_trades >= 100.
"""

from __future__ import annotations

import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path("/Users/weeoto/turbo-carnival/research")
CACHE_PATH = ROOT / "data" / "cache" / "us_hk_pairs_daily.parquet"
LOG_PATH = ROOT / "logs" / "iter_18_backtest.log"
TRADES_CSV = ROOT / "results" / "iter_18_trades.csv"
SUMMARY_JSON = ROOT / "results" / "iter_18_summary.json"

# Pairs: (US ADR, HK ticker, ADR ratio = HK shares per 1 ADR)
PAIRS: list[tuple[str, str, int]] = [
    ("BABA", "9988.HK", 8),
    ("JD",   "9618.HK", 2),
    ("BIDU", "9888.HK", 8),
    ("NTES", "9999.HK", 25),
    ("TCOM", "9961.HK", 1),
]

START = "2020-01-01"
USD_HKD_PEG = 7.80  # HKMA peg midpoint (band 7.75-7.85)
USD_PER_HKD = 1.0 / USD_HKD_PEG

ENTRY_BPS = 80.0
EXIT_BPS = 20.0
STOP_BPS = 250.0
MAX_HOLD_DAYS = 5  # trading days

# Cost: 30 bps round-trip per brief (stamp + FX spread + commissions + slippage).
# Also track the lighter 20 bps view (5 bps/leg/side * 4 leg-sides) as net20_bps for reference.
COST_BPS_PER_LEG_PER_SIDE = 5.0
ROUND_TRIP_COST_BPS = 30.0  # per brief Step 3: 30 bps round-trip
ROUND_TRIP_COST_BPS_LIGHT = 4 * COST_BPS_PER_LEG_PER_SIDE  # 20 bps (legacy view)

# Notional sizing for $-pnl reporting (not used in pass/fail; bps-based)
NOTIONAL_PER_LEG = 20_000.0  # informational only

# US-only tickers we treat as "ADR side"
US_TICKERS = [p[0] for p in PAIRS]
HK_TICKERS = [p[1] for p in PAIRS]
ALL_TICKERS = US_TICKERS + HK_TICKERS


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    # Clear any prior handlers so re-runs don't double-log
    for h in list(logging.getLogger().handlers):
        logging.getLogger().removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_PATH, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("iter_18")


def fetch_prices(log: logging.Logger) -> pd.DataFrame:
    """Pull daily Close for all US ADRs and HK underlyings.

    Cache layout (wide): index=Date, columns=[BABA, JD, BIDU, NTES, TCOM, 9988.HK, ...].
    Note: HK tickers have different trading calendar than US (HK CNY week, US holidays).
    We outer-join calendars; per-pair we drop NaN rows when forming signals.
    """
    if CACHE_PATH.exists():
        df = pd.read_parquet(CACHE_PATH)
        last = df.index.max()
        # Refresh if older than 4 days (HK/US holidays + weekends could leave 3-day gap)
        if last >= pd.Timestamp.today().normalize() - pd.Timedelta(days=5):
            missing = [t for t in ALL_TICKERS if t not in df.columns]
            if not missing:
                log.info(f"Loaded cache {CACHE_PATH} rows={len(df)} last={last.date()} cols={list(df.columns)}")
                return df
            log.info(f"Cache missing tickers {missing}; refetching all")
        else:
            log.info(f"Cache stale (last={last.date()}); refetching")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading yfinance: {ALL_TICKERS} from {START}")
    # auto_adjust=True so Close is split/dividend adjusted on both sides.
    # NOTE on adjustment: adjusted closes can move parity in a way that doesn't reflect
    # what a real trader would have seen at the time (e.g. ratio change events). We use
    # adjusted closes for tractability; per spec §10(7), ratio changes are extremely rare
    # and we have no such event in 2020-present for the 5 names.
    raw = yf.download(
        ALL_TICKERS,
        start=START,
        end=None,
        auto_adjust=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )

    # yfinance multi-ticker output: with group_by='ticker' we get a top-level ticker index.
    closes: dict[str, pd.Series] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in ALL_TICKERS:
            if t in raw.columns.get_level_values(0):
                s = raw[t]["Close"].dropna()
                closes[t] = s
            else:
                log.warning(f"No data returned for {t}")
    else:
        # Single ticker case (shouldn't happen with our list)
        closes[ALL_TICKERS[0]] = raw["Close"].dropna()

    if not closes:
        raise RuntimeError("yfinance returned no usable data")

    df = pd.DataFrame(closes).sort_index()
    df.index = pd.to_datetime(df.index)
    df.to_parquet(CACHE_PATH)
    log.info(
        f"Saved cache {CACHE_PATH} rows={len(df)} "
        f"{df.index.min().date()}..{df.index.max().date()} cols={list(df.columns)}"
    )
    # Log first valid date per ticker (when each name actually starts trading on its venue)
    for t in ALL_TICKERS:
        if t in df.columns:
            fvi = df[t].first_valid_index()
            n = df[t].notna().sum()
            log.info(f"  {t}: first={fvi.date() if fvi is not None else 'NA'} n={n}")
    return df


def build_pair_frame(prices: pd.DataFrame, us: str, hk: str, ratio: int) -> pd.DataFrame:
    """Build the per-pair deviation series.

    Returns columns: us_close, hk_close, parity_usd, dev_bps, all indexed by date,
    restricted to dates where BOTH sides have a print.
    """
    if us not in prices.columns or hk not in prices.columns:
        return pd.DataFrame()
    sub = prices[[us, hk]].dropna(how="any").copy()
    sub.columns = ["us_close", "hk_close"]
    sub["parity_usd"] = sub["hk_close"] * USD_PER_HKD * ratio
    sub["dev_bps"] = 10_000.0 * (sub["us_close"] - sub["parity_usd"]) / sub["parity_usd"]
    return sub


def run_pair_backtest(
    pair: pd.DataFrame,
    us: str,
    hk: str,
    ratio: int,
    log: logging.Logger,
) -> list[dict]:
    """Walk the daily series for one pair and emit one row per closed round-trip.

    Decision model (deliberately simple, matches the brief's day-bar resolution):
      - Each row is one trading day where BOTH legs have closes.
      - At end of day t we observe dev_bps[t]. If flat and |dev| > ENTRY -> open at today's
        closes (paper assumption: we cross at the close prints, no intraday execution).
      - On subsequent days we re-check exit/stop/time on each day's close.
      - Holding-days counter uses bar count of the per-pair frame (HK+US joint sessions),
        which is the spirit of "5 HK trading sessions".
      - PnL leg by leg:
          if direction == 'short_us_long_hk':
              us_pnl = -(us_exit - us_entry) / us_entry        # short ADR
              hk_pnl = +(hk_exit - hk_entry) / hk_entry        # long HK, FX assumed pegged
          if direction == 'long_us_short_hk':
              us_pnl = +(us_exit - us_entry) / us_entry
              hk_pnl = -(hk_exit - hk_entry) / hk_entry
        Total gross return = us_pnl + hk_pnl  (dollar-neutral by construction; each leg
        sized at NOTIONAL_PER_LEG, so total notional 2*NOTIONAL; gross_bps = total_return/2 * 1e4
        i.e. we average across legs as if each leg is 50% of gross notional).

    Note on this PnL model: we are NOT modeling actual convergence-to-parity dollar PnL —
    we're modeling the *spread-trade* return where the two legs are each $NOTIONAL_PER_LEG
    and the dollar-neutral position earns the spread between leg returns. This matches the
    spec's framing (sized $20k per leg, dollar-neutral within pair, gross 2x net).
    The bps_of_gross_notional is the natural "x bps per pair-trade" comparable to the
    25 bps PASS threshold.
    """
    trades: list[dict] = []
    if pair.empty:
        log.info(f"  {us}/{hk}: empty pair frame, skipping")
        return trades

    state = "flat"
    direction = None  # 'short_us_long_hk' or 'long_us_short_hk'
    entry_idx = None  # positional index in pair frame
    entry_date = None
    entry_us = None
    entry_hk = None
    entry_dev = None

    dates = pair.index.to_list()
    n = len(dates)
    log.info(
        f"  {us}/{hk}: {n} joint sessions "
        f"{dates[0].date()}..{dates[-1].date()}  "
        f"dev_bps stats: mean={pair['dev_bps'].mean():+.1f} "
        f"std={pair['dev_bps'].std():.1f} "
        f"p5={pair['dev_bps'].quantile(0.05):+.1f} "
        f"p95={pair['dev_bps'].quantile(0.95):+.1f} "
        f"|dev|>80 count={(pair['dev_bps'].abs()>ENTRY_BPS).sum()}"
    )

    def close_trade(i: int, reason: str) -> None:
        nonlocal state, direction, entry_idx, entry_date, entry_us, entry_hk, entry_dev
        exit_date = dates[i]
        exit_row = pair.iloc[i]
        exit_us = float(exit_row["us_close"])
        exit_hk = float(exit_row["hk_close"])
        exit_dev = float(exit_row["dev_bps"])

        us_ret = (exit_us - entry_us) / entry_us
        hk_ret = (exit_hk - entry_hk) / entry_hk

        if direction == "short_us_long_hk":
            us_leg_ret = -us_ret
            hk_leg_ret = +hk_ret
        else:  # long_us_short_hk
            us_leg_ret = +us_ret
            hk_leg_ret = -hk_ret

        # Each leg is 50% of gross (2 legs total). Spread return on gross notional:
        gross_return = 0.5 * (us_leg_ret + hk_leg_ret)
        gross_bps = gross_return * 10_000.0
        net_bps = gross_bps - ROUND_TRIP_COST_BPS  # 30 bps round-trip (brief)
        net20_bps = gross_bps - ROUND_TRIP_COST_BPS_LIGHT  # 20 bps round-trip (lighter view)

        holding_bars = i - entry_idx
        holding_cal_days = (exit_date - entry_date).days

        trades.append(
            {
                "pair": f"{us}/{hk}",
                "us": us,
                "hk": hk,
                "ratio": ratio,
                "entry_date": entry_date,
                "exit_date": exit_date,
                "direction": direction,
                "entry_dev_bps": round(entry_dev, 2),
                "exit_dev_bps": round(exit_dev, 2),
                "entry_us": round(entry_us, 4),
                "exit_us": round(exit_us, 4),
                "entry_hk": round(entry_hk, 4),
                "exit_hk": round(exit_hk, 4),
                "us_leg_ret": round(us_leg_ret * 100, 4),  # percent
                "hk_leg_ret": round(hk_leg_ret * 100, 4),  # percent
                "gross_bps": round(gross_bps, 2),
                "cost_bps": ROUND_TRIP_COST_BPS,
                "net_bps": round(net_bps, 2),
                "net20_bps": round(net20_bps, 2),
                "holding_bars": int(holding_bars),
                "holding_cal_days": int(holding_cal_days),
                "exit_reason": reason,
            }
        )

        state = "flat"
        direction = None
        entry_idx = None
        entry_date = None
        entry_us = None
        entry_hk = None
        entry_dev = None

    for i in range(n):
        row = pair.iloc[i]
        dev = float(row["dev_bps"])
        us_c = float(row["us_close"])
        hk_c = float(row["hk_close"])
        ts = dates[i]

        if state == "flat":
            if abs(dev) > ENTRY_BPS:
                if dev > 0:
                    direction = "short_us_long_hk"  # ADR rich -> short ADR, long HK
                else:
                    direction = "long_us_short_hk"  # HK rich -> long ADR, short HK
                state = "open"
                entry_idx = i
                entry_date = ts
                entry_us = us_c
                entry_hk = hk_c
                entry_dev = dev
        else:
            # In trade: exit/stop/time priority
            held = i - entry_idx
            if abs(dev) > STOP_BPS:
                close_trade(i, "stop_out")
            elif abs(dev) < EXIT_BPS:
                close_trade(i, "converged")
            elif held >= MAX_HOLD_DAYS:
                close_trade(i, "time_stop")

    # End of data — close any open trade at last available bar so we record it
    if state != "flat":
        close_trade(n - 1, "end_of_data")

    log.info(f"  {us}/{hk}: trades emitted = {len(trades)}")
    return trades


def max_consecutive_losses(net_bps: pd.Series) -> int:
    """Longest run of consecutive trades with net_bps <= 0, ordered by exit_date."""
    streak = 0
    best = 0
    for v in net_bps:
        if v <= 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return int(best)


def summarize(trades_df: pd.DataFrame, log: logging.Logger, prices_meta: dict) -> dict:
    n_trades = int(len(trades_df))
    summary: dict = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "spec": "iter_18_us_hk_dual_listing_arbitrage",
        "params": {
            "pairs": [{"us": u, "hk": h, "ratio": r} for u, h, r in PAIRS],
            "start": START,
            "usd_hkd_peg": USD_HKD_PEG,
            "entry_bps": ENTRY_BPS,
            "exit_bps": EXIT_BPS,
            "stop_bps": STOP_BPS,
            "max_hold_days": MAX_HOLD_DAYS,
            "cost_bps_per_leg_per_side": COST_BPS_PER_LEG_PER_SIDE,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "round_trip_cost_bps_light": ROUND_TRIP_COST_BPS_LIGHT,
        },
        "data_meta": prices_meta,
        "total_trades": n_trades,
        "win_rate": None,
        "mean_bps_gross": None,
        "mean_bps_net": None,
        "mean_bps_net20": None,
        "total_pnl_bps": None,
        "median_holding_bars": None,
        "ann_return_pct": None,
        "sharpe_proxy": None,
        "max_consecutive_losses": None,
        "exit_reason_counts": None,
        "per_pair": {},
        "verdict": "FAIL",
        "verdict_reason": "no_trades",
    }

    if n_trades == 0:
        return summary

    # Net = gross - 30 bps round-trip (brief STEP 3)
    trades_df = trades_df.sort_values("exit_date").reset_index(drop=True)
    net = trades_df["net_bps"].astype(float)
    gross = trades_df["gross_bps"].astype(float)
    net20 = trades_df["net20_bps"].astype(float)

    win_rate = float((net > 0).mean() * 100.0)
    mean_bps_gross = float(gross.mean())
    mean_bps_net = float(net.mean())
    mean_bps_net20 = float(net20.mean())
    total_pnl_bps = float(net.sum())
    median_hold = float(trades_df["holding_bars"].median())

    # Annualized return: each trade contributes net_bps/10000 to per-trade return.
    # Convert to per-year by scaling: total_pnl_bps over the elapsed calendar years,
    # interpreted as gross-notional bp/year (NOT a portfolio CAGR — paired-trade strategy
    # where capital is recycled). This matches the spec's "bps per round-trip" framing.
    first_dt = pd.to_datetime(trades_df["entry_date"].min())
    last_dt = pd.to_datetime(trades_df["exit_date"].max())
    years = max((last_dt - first_dt).days / 365.25, 1e-9)
    ann_return_pct = float((total_pnl_bps / 10_000.0) / years * 100.0)  # % per year on gross notional

    # Sharpe proxy: per-trade net_bps mean/std * sqrt(trades_per_year).
    # We use trade-count Sharpe because trades are discrete events of variable length.
    if net.std(ddof=1) > 0:
        trades_per_year = n_trades / years
        sharpe = float(net.mean() / net.std(ddof=1) * math.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    mcl = max_consecutive_losses(net)

    exit_reason_counts = trades_df["exit_reason"].value_counts().to_dict()

    per_pair: dict[str, dict] = {}
    for pname, sub in trades_df.groupby("pair"):
        per_pair[str(pname)] = {
            "n": int(len(sub)),
            "win_rate": round(float((sub["net_bps"] > 0).mean() * 100.0), 2),
            "mean_bps_net": round(float(sub["net_bps"].mean()), 2),
            "mean_bps_gross": round(float(sub["gross_bps"].mean()), 2),
            "median_hold_bars": round(float(sub["holding_bars"].median()), 2),
            "stop_outs": int((sub["exit_reason"] == "stop_out").sum()),
            "time_stops": int((sub["exit_reason"] == "time_stop").sum()),
            "converged": int((sub["exit_reason"] == "converged").sum()),
        }

    summary.update(
        {
            "total_trades": n_trades,
            "win_rate": round(win_rate, 2),
            "mean_bps_gross": round(mean_bps_gross, 2),
            "mean_bps_net": round(mean_bps_net, 2),
            "mean_bps_net20": round(mean_bps_net20, 2),
            "total_pnl_bps": round(total_pnl_bps, 2),
            "median_holding_bars": round(median_hold, 2),
            "ann_return_pct": round(ann_return_pct, 2),
            "sharpe_proxy": round(sharpe, 3),
            "max_consecutive_losses": mcl,
            "exit_reason_counts": exit_reason_counts,
            "per_pair": per_pair,
            "elapsed_years": round(years, 2),
        }
    )

    passed = (mean_bps_net >= 25.0) and (n_trades >= 100)
    summary["verdict"] = "PASS" if passed else "FAIL"
    summary["verdict_reason"] = (
        f"mean_bps_net={mean_bps_net:.2f} (req>=25) AND total_trades={n_trades} (req>=100)"
    )

    log.info(
        "Summary: trades=%d win=%.2f%% mean_gross=%.2f mean_net30=%.2f mean_net20=%.2f "
        "total=%.1f bp ann=%.2f%% sharpe=%.3f mcl=%d -> %s",
        n_trades, win_rate, mean_bps_gross, mean_bps_net, mean_bps_net20,
        total_pnl_bps, ann_return_pct, sharpe, mcl, summary["verdict"],
    )
    return summary


def main() -> int:
    log = setup_logging()
    log.info("=== iter_18 backtest start ===")
    log.info(
        "Pairs: %s",
        ", ".join(f"{u}/{h} 1:{r}" for u, h, r in PAIRS),
    )
    prices = fetch_prices(log)

    prices_meta = {
        "rows": int(len(prices)),
        "first_date": str(prices.index.min().date()),
        "last_date": str(prices.index.max().date()),
        "columns": list(prices.columns),
        "first_valid_per_ticker": {
            t: (str(prices[t].first_valid_index().date()) if t in prices.columns and prices[t].first_valid_index() is not None else None)
            for t in ALL_TICKERS
        },
    }

    all_trades: list[dict] = []
    for us, hk, ratio in PAIRS:
        log.info(f"Building pair {us}/{hk} (ratio 1:{ratio})")
        pair = build_pair_frame(prices, us, hk, ratio)
        trades = run_pair_backtest(pair, us, hk, ratio, log)
        all_trades.extend(trades)

    trades_df = pd.DataFrame(all_trades)
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(TRADES_CSV, index=False)
    log.info(f"Wrote {TRADES_CSV} rows={len(trades_df)}")

    summary = summarize(trades_df, log, prices_meta)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"Wrote {SUMMARY_JSON}")
    log.info("=== iter_18 backtest done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
