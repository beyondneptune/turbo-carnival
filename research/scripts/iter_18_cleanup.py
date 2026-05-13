"""iter_18 CLEANUP backtest: US/HK dual-listing parity arbitrage, with proper
data adjustments.

Differences from the original iter_18_backtest.py run that produced
mean_bps_net = -7.2 bps (FAIL) in aggregate:

1. NTES / 9999.HK ratio corrected from 1:25 to **1:5**.
   NTES executed a 1:5 ADS-ratio change effective 2020 (each ADR now
   represents 5 ordinary HK shares, not 25). The prior run used the legacy
   pre-2020 ratio, which mis-priced parity by 5x and turned NTES into a
   constant-deviation signal that always tripped entry then stop-out.

2. yfinance auto_adjust distorts parity around dividends, particularly the
   BABA and JD 2024-25 special dividends. auto_adjust back-applies dividend
   adjustments retroactively to the historical series on ONE side of the
   pair (the side that paid the dividend, i.e. the US ADR for BABA/JD's
   special cash dividends), while the HK side's adjustment timing and
   per-share dividend amount can be reported with a different convention.
   This drives a one-off ~50-300 bps shift on the ex-date that the strategy
   then "discovers" and trades into, generating a cluster of immediate
   stop-outs.

   Fix: use auto_adjust=False to pull raw unadjusted closes. Parity in
   unadjusted prices is theoretically valid because a real trader sees the
   same unadjusted prints on both venues at the moment of trade; dividends
   create a real, one-session step in BOTH legs simultaneously (assuming HK
   ex-date and US record-date are aligned, which they are by depositary
   construction for fungible dual-primary listings). We additionally apply
   the spec's ex-div skip: filter out pair-days within +/- 3 sessions of
   any dividend event on either leg.

3. Pair ratios are now the CURRENT-STATE ratios verified against each
   issuer's most recent F-6 amendment / HKEX prospectus:
     BABA / 9988.HK = 1:8
     JD   / 9618.HK = 1:2
     BIDU / 9888.HK = 1:8
     NTES / 9999.HK = 1:5   (was 1:25 in the failed run)
     TCOM / 9961.HK = 1:1

Everything else (strategy thresholds, costs, sizing) is unchanged:
  Entry  |dev| > 80 bps
  Exit   |dev| < 20 bps
  Stop   |dev| > 250 bps
  Time   5 trading days max hold
  Cost   30 bps round-trip

PASS rule: aggregate mean_bps_net >= 25 AND total_trades >= 100.
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
# Use a separate cache so we don't clobber the original auto_adjust=True cache
CACHE_PATH = ROOT / "data" / "cache" / "us_hk_pairs_daily_raw.parquet"
DIV_CACHE_PATH = ROOT / "data" / "cache" / "us_hk_pairs_dividends.parquet"
LOG_PATH = ROOT / "logs" / "iter_18_cleanup.log"
TRADES_CSV = ROOT / "results" / "iter_18_cleanup_trades.csv"
SUMMARY_JSON = ROOT / "results" / "iter_18_cleanup_summary.json"

# Pairs: (US ADR, HK ticker, current-state ADR ratio = HK shares per 1 ADR)
PAIRS: list[tuple[str, str, int]] = [
    ("BABA", "9988.HK", 8),
    ("JD",   "9618.HK", 2),
    ("BIDU", "9888.HK", 8),
    ("NTES", "9999.HK", 5),   # CORRECTED from 25 -> 5
    ("TCOM", "9961.HK", 1),
]

START = "2020-01-01"
USD_HKD_PEG = 7.80
USD_PER_HKD = 1.0 / USD_HKD_PEG

ENTRY_BPS = 80.0
EXIT_BPS = 20.0
STOP_BPS = 250.0
MAX_HOLD_DAYS = 5

COST_BPS_PER_LEG_PER_SIDE = 5.0
ROUND_TRIP_COST_BPS = 30.0
ROUND_TRIP_COST_BPS_LIGHT = 4 * COST_BPS_PER_LEG_PER_SIDE  # 20 bps reference

EX_DIV_SKIP_WINDOW_BARS = 3  # spec rule 4(2): skip +/- 3 sessions around ex-div

US_TICKERS = [p[0] for p in PAIRS]
HK_TICKERS = [p[1] for p in PAIRS]
ALL_TICKERS = US_TICKERS + HK_TICKERS


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
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
    return logging.getLogger("iter_18_cleanup")


def fetch_prices_and_divs(log: logging.Logger) -> tuple[pd.DataFrame, dict[str, pd.Series]]:
    """Pull RAW (unadjusted) daily Close + dividend series for all tickers.

    Two outputs:
      - prices: wide DataFrame of unadjusted Close prices (auto_adjust=False)
      - divs:   dict ticker -> Series of dividend amounts indexed by ex-date

    auto_adjust=False is critical: it keeps the parity series clean across
    dividend events. With auto_adjust=True yfinance retroactively scales the
    historical Close on each leg according to its own dividend record, which
    is timed independently on US (record date) vs HK (ex-date) calendars and
    can be off by 1-2 sessions, producing spurious 30-200 bps parity jumps
    on dividend dates that the strategy then trades into.
    """
    # Cache check
    if CACHE_PATH.exists() and DIV_CACHE_PATH.exists():
        prices = pd.read_parquet(CACHE_PATH)
        last = prices.index.max()
        if last >= pd.Timestamp.today().normalize() - pd.Timedelta(days=5):
            missing = [t for t in ALL_TICKERS if t not in prices.columns]
            if not missing:
                div_df = pd.read_parquet(DIV_CACHE_PATH)
                divs: dict[str, pd.Series] = {}
                for t in ALL_TICKERS:
                    if t in div_df.columns:
                        s = div_df[t].dropna()
                        if not s.empty:
                            divs[t] = s
                log.info(
                    f"Loaded RAW cache {CACHE_PATH} rows={len(prices)} "
                    f"last={last.date()} cols={list(prices.columns)} "
                    f"divs_tickers={list(divs.keys())}"
                )
                return prices, divs
            log.info(f"RAW cache missing {missing}; refetching")
        else:
            log.info(f"RAW cache stale (last={last.date()}); refetching")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        f"Downloading yfinance UNADJUSTED (auto_adjust=False) "
        f"tickers={ALL_TICKERS} from {START}"
    )

    # Download unadjusted prices in bulk
    raw = yf.download(
        ALL_TICKERS,
        start=START,
        end=None,
        auto_adjust=False,
        actions=True,          # include Dividends, Stock Splits columns
        progress=False,
        threads=True,
        group_by="ticker",
    )

    closes: dict[str, pd.Series] = {}
    divs: dict[str, pd.Series] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in ALL_TICKERS:
            if t in raw.columns.get_level_values(0):
                sub = raw[t]
                # Prefer "Close" (unadjusted with auto_adjust=False)
                if "Close" in sub.columns:
                    s = sub["Close"].dropna()
                    if len(s) > 0:
                        closes[t] = s
                if "Dividends" in sub.columns:
                    d = sub["Dividends"]
                    d = d[d > 0].dropna()
                    if len(d) > 0:
                        divs[t] = d
            else:
                log.warning(f"No data returned for {t}")
    else:
        log.warning("Unexpected yfinance return shape (single ticker)")

    if not closes:
        raise RuntimeError("yfinance returned no usable price data")

    df = pd.DataFrame(closes).sort_index()
    df.index = pd.to_datetime(df.index)

    # Build dividend wide-frame (one column per ticker that paid divs)
    if divs:
        div_wide = pd.concat(divs, axis=1).sort_index()
        div_wide.index = pd.to_datetime(div_wide.index)
    else:
        div_wide = pd.DataFrame()

    df.to_parquet(CACHE_PATH)
    if not div_wide.empty:
        div_wide.to_parquet(DIV_CACHE_PATH)
    else:
        # Write an empty parquet so the cache-existence check passes next time
        pd.DataFrame(index=df.index).to_parquet(DIV_CACHE_PATH)

    log.info(
        f"Saved RAW cache {CACHE_PATH} rows={len(df)} "
        f"{df.index.min().date()}..{df.index.max().date()} cols={list(df.columns)}"
    )
    log.info(f"Dividend events captured (tickers): {list(divs.keys())}")
    for t, s in divs.items():
        log.info(
            f"  {t}: {len(s)} div events, "
            f"first={s.index.min().date()} last={s.index.max().date()} "
            f"min={s.min():.4f} max={s.max():.4f}"
        )

    for t in ALL_TICKERS:
        if t in df.columns:
            fvi = df[t].first_valid_index()
            n = df[t].notna().sum()
            log.info(f"  {t}: first={fvi.date() if fvi is not None else 'NA'} n={n}")
    return df, divs


def build_pair_frame(
    prices: pd.DataFrame,
    us: str,
    hk: str,
    ratio: int,
    divs: dict[str, pd.Series],
    log: logging.Logger,
) -> pd.DataFrame:
    """Build per-pair deviation series on UNADJUSTED prices.

    Adds an `ex_div_block` boolean column: True if either leg has a dividend
    within +/- EX_DIV_SKIP_WINDOW_BARS sessions of this row. We will refuse
    to OPEN new trades on blocked days (and refuse to take CONVERGED exits
    on blocked days too, since the convergence signal there is suspect).
    Stop-out and time-stop still fire normally because they are pure
    risk-control exits.
    """
    if us not in prices.columns or hk not in prices.columns:
        return pd.DataFrame()
    sub = prices[[us, hk]].dropna(how="any").copy()
    sub.columns = ["us_close", "hk_close"]
    sub["parity_usd"] = sub["hk_close"] * USD_PER_HKD * ratio
    sub["dev_bps"] = 10_000.0 * (sub["us_close"] - sub["parity_usd"]) / sub["parity_usd"]

    # Build ex-div block flag
    block = pd.Series(False, index=sub.index)
    for leg in (us, hk):
        if leg in divs:
            for ex_date in divs[leg].index:
                ex_date = pd.Timestamp(ex_date).normalize()
                if ex_date not in sub.index:
                    # Find the closest session in the pair frame
                    nearest = sub.index[sub.index.get_indexer([ex_date], method="nearest")[0]]
                    ex_date = nearest
                pos = sub.index.get_loc(ex_date)
                lo = max(0, pos - EX_DIV_SKIP_WINDOW_BARS)
                hi = min(len(sub) - 1, pos + EX_DIV_SKIP_WINDOW_BARS)
                block.iloc[lo:hi + 1] = True
    sub["ex_div_block"] = block

    n_blocked = int(block.sum())
    log.info(
        f"  {us}/{hk}: ratio=1:{ratio}  rows={len(sub)}  "
        f"ex_div_blocked_rows={n_blocked} "
        f"dev mean={sub['dev_bps'].mean():+.1f} "
        f"std={sub['dev_bps'].std():.1f} "
        f"p5={sub['dev_bps'].quantile(0.05):+.1f} "
        f"p95={sub['dev_bps'].quantile(0.95):+.1f} "
        f"|dev|>80 n={(sub['dev_bps'].abs() > ENTRY_BPS).sum()}"
    )
    return sub


def run_pair_backtest(
    pair: pd.DataFrame,
    us: str,
    hk: str,
    ratio: int,
    log: logging.Logger,
) -> list[dict]:
    """Walk the daily series for one pair, identical to the original backtest
    except that:
      - we refuse to OPEN a trade on an ex_div_block row
      - we refuse to take a CONVERGED exit on an ex_div_block row
        (stop_out and time_stop still fire normally)
    """
    trades: list[dict] = []
    if pair.empty:
        log.info(f"  {us}/{hk}: empty pair frame, skipping")
        return trades

    state = "flat"
    direction: str | None = None
    entry_idx: int | None = None
    entry_date = None
    entry_us = None
    entry_hk = None
    entry_dev = None

    dates = pair.index.to_list()
    n = len(dates)

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
        else:
            us_leg_ret = +us_ret
            hk_leg_ret = -hk_ret

        gross_return = 0.5 * (us_leg_ret + hk_leg_ret)
        gross_bps = gross_return * 10_000.0
        net_bps = gross_bps - ROUND_TRIP_COST_BPS
        net20_bps = gross_bps - ROUND_TRIP_COST_BPS_LIGHT

        holding_bars = i - entry_idx
        holding_cal_days = (exit_date - entry_date).days

        trades.append({
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
            "us_leg_ret_pct": round(us_leg_ret * 100, 4),
            "hk_leg_ret_pct": round(hk_leg_ret * 100, 4),
            "gross_bps": round(gross_bps, 2),
            "cost_bps": ROUND_TRIP_COST_BPS,
            "net_bps": round(net_bps, 2),
            "net20_bps": round(net20_bps, 2),
            "holding_bars": int(holding_bars),
            "holding_cal_days": int(holding_cal_days),
            "exit_reason": reason,
        })

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
        blocked = bool(row["ex_div_block"])

        if state == "flat":
            if blocked:
                continue
            if abs(dev) > ENTRY_BPS:
                direction = "short_us_long_hk" if dev > 0 else "long_us_short_hk"
                state = "open"
                entry_idx = i
                entry_date = ts
                entry_us = us_c
                entry_hk = hk_c
                entry_dev = dev
        else:
            held = i - entry_idx
            if abs(dev) > STOP_BPS:
                close_trade(i, "stop_out")
            elif abs(dev) < EXIT_BPS and not blocked:
                close_trade(i, "converged")
            elif held >= MAX_HOLD_DAYS:
                close_trade(i, "time_stop")

    if state != "flat":
        close_trade(n - 1, "end_of_data")

    log.info(f"  {us}/{hk}: trades emitted = {len(trades)}")
    return trades


def max_consecutive_losses(net_bps: pd.Series) -> int:
    streak = 0
    best = 0
    for v in net_bps:
        if v <= 0:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return int(best)


def per_pair_block(sub: pd.DataFrame) -> dict:
    if len(sub) == 0:
        return {
            "n": 0, "win_rate": None, "mean_bps_net": None,
            "mean_bps_gross": None, "median_hold_bars": None,
            "sharpe_proxy": None, "stop_outs": 0, "time_stops": 0,
            "converged": 0, "end_of_data": 0,
        }
    net = sub["net_bps"].astype(float)
    gross = sub["gross_bps"].astype(float)

    # Per-pair Sharpe proxy: mean/std * sqrt(n_per_year)
    first_dt = pd.to_datetime(sub["entry_date"].min())
    last_dt = pd.to_datetime(sub["exit_date"].max())
    years = max((last_dt - first_dt).days / 365.25, 1e-9)
    if net.std(ddof=1) > 0 and len(sub) > 1:
        per_year = len(sub) / years
        sharpe = float(net.mean() / net.std(ddof=1) * math.sqrt(per_year))
    else:
        sharpe = 0.0

    return {
        "n": int(len(sub)),
        "win_rate": round(float((net > 0).mean() * 100.0), 2),
        "mean_bps_net": round(float(net.mean()), 2),
        "mean_bps_gross": round(float(gross.mean()), 2),
        "median_hold_bars": round(float(sub["holding_bars"].median()), 2),
        "sharpe_proxy": round(sharpe, 3),
        "stop_outs": int((sub["exit_reason"] == "stop_out").sum()),
        "time_stops": int((sub["exit_reason"] == "time_stop").sum()),
        "converged": int((sub["exit_reason"] == "converged").sum()),
        "end_of_data": int((sub["exit_reason"] == "end_of_data").sum()),
    }


def summarize(
    trades_df: pd.DataFrame,
    log: logging.Logger,
    prices_meta: dict,
    div_meta: dict,
) -> dict:
    n_trades = int(len(trades_df))
    summary: dict = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "spec": "iter_18_us_hk_dual_listing_arbitrage (CLEANUP)",
        "fixes_applied": [
            "NTES/9999.HK ratio corrected from 1:25 to 1:5 (post-2020 ADS-ratio change)",
            "yfinance auto_adjust=False to use UNADJUSTED prices (parity stable across dividend events)",
            "Ex-div skip window: +/- 3 sessions blocked for new entries and converged exits",
            "Stop-out / time-stop still fire on ex-div blocked days (risk control unchanged)",
        ],
        "params": {
            "pairs": [{"us": u, "hk": h, "ratio": r} for u, h, r in PAIRS],
            "start": START,
            "usd_hkd_peg": USD_HKD_PEG,
            "entry_bps": ENTRY_BPS,
            "exit_bps": EXIT_BPS,
            "stop_bps": STOP_BPS,
            "max_hold_days": MAX_HOLD_DAYS,
            "round_trip_cost_bps": ROUND_TRIP_COST_BPS,
            "ex_div_skip_window_bars": EX_DIV_SKIP_WINDOW_BARS,
        },
        "data_meta": prices_meta,
        "dividend_meta": div_meta,
        "aggregate": {},
        "per_pair": {},
        "verdict": "FAIL",
        "verdict_reason": "no_trades",
    }

    if n_trades == 0:
        return summary

    trades_df = trades_df.sort_values("exit_date").reset_index(drop=True)
    net = trades_df["net_bps"].astype(float)
    gross = trades_df["gross_bps"].astype(float)

    win_rate = float((net > 0).mean() * 100.0)
    mean_bps_gross = float(gross.mean())
    mean_bps_net = float(net.mean())
    total_pnl_bps = float(net.sum())
    median_hold = float(trades_df["holding_bars"].median())

    first_dt = pd.to_datetime(trades_df["entry_date"].min())
    last_dt = pd.to_datetime(trades_df["exit_date"].max())
    years = max((last_dt - first_dt).days / 365.25, 1e-9)
    ann_return_pct = float((total_pnl_bps / 10_000.0) / years * 100.0)

    if net.std(ddof=1) > 0:
        trades_per_year = n_trades / years
        sharpe = float(net.mean() / net.std(ddof=1) * math.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    mcl = max_consecutive_losses(net)
    exit_reason_counts = trades_df["exit_reason"].value_counts().to_dict()

    summary["aggregate"] = {
        "total_trades": n_trades,
        "win_rate": round(win_rate, 2),
        "mean_bps_gross": round(mean_bps_gross, 2),
        "mean_bps_net": round(mean_bps_net, 2),
        "total_pnl_bps": round(total_pnl_bps, 2),
        "median_holding_bars": round(median_hold, 2),
        "ann_return_pct": round(ann_return_pct, 2),
        "sharpe_proxy": round(sharpe, 3),
        "max_consecutive_losses": mcl,
        "exit_reason_counts": exit_reason_counts,
        "elapsed_years": round(years, 2),
    }

    per_pair: dict[str, dict] = {}
    pair_pass_count = 0
    for pname, sub in trades_df.groupby("pair"):
        block = per_pair_block(sub)
        per_pair[str(pname)] = block
        if (
            block.get("mean_bps_net") is not None
            and block["mean_bps_net"] >= 25.0
            and block["n"] >= 20
        ):
            pair_pass_count += 1
    summary["per_pair"] = per_pair
    summary["per_pair_pass_count"] = pair_pass_count
    summary["per_pair_pass_threshold"] = "mean_bps_net >= 25 AND n >= 20"

    passed = (mean_bps_net >= 25.0) and (n_trades >= 100)
    summary["verdict"] = "PASS" if passed else "FAIL"
    summary["verdict_reason"] = (
        f"aggregate mean_bps_net={mean_bps_net:.2f} (req>=25) "
        f"AND total_trades={n_trades} (req>=100)"
    )

    log.info(
        "AGGREGATE: trades=%d win=%.2f%% mean_gross=%.2f mean_net=%.2f "
        "total=%.1f bp ann=%.2f%% sharpe=%.3f mcl=%d -> %s",
        n_trades, win_rate, mean_bps_gross, mean_bps_net,
        total_pnl_bps, ann_return_pct, sharpe, mcl, summary["verdict"],
    )
    for pname, block in per_pair.items():
        log.info(
            "  %s: n=%d win=%.2f%% net=%.2f gross=%.2f sharpe=%.3f "
            "stops=%d converged=%d time=%d",
            pname, block["n"], block["win_rate"], block["mean_bps_net"],
            block["mean_bps_gross"], block["sharpe_proxy"],
            block["stop_outs"], block["converged"], block["time_stops"],
        )
    log.info(
        "PER-PAIR PASS COUNT (mean_bps_net>=25 AND n>=20): %d of %d",
        pair_pass_count, len(per_pair),
    )
    return summary


def main() -> int:
    log = setup_logging()
    log.info("=== iter_18 CLEANUP backtest start ===")
    log.info(
        "Pairs (CURRENT-STATE ratios): %s",
        ", ".join(f"{u}/{h} 1:{r}" for u, h, r in PAIRS),
    )

    prices, divs = fetch_prices_and_divs(log)

    prices_meta = {
        "rows": int(len(prices)),
        "first_date": str(prices.index.min().date()),
        "last_date": str(prices.index.max().date()),
        "columns": list(prices.columns),
        "first_valid_per_ticker": {
            t: (
                str(prices[t].first_valid_index().date())
                if t in prices.columns and prices[t].first_valid_index() is not None
                else None
            )
            for t in ALL_TICKERS
        },
        "auto_adjust": False,
    }
    div_meta = {
        t: {
            "n_events": int(len(s)),
            "first_ex_date": str(s.index.min().date()),
            "last_ex_date": str(s.index.max().date()),
            "max_amount": round(float(s.max()), 4),
        }
        for t, s in divs.items()
    }

    all_trades: list[dict] = []
    for us, hk, ratio in PAIRS:
        log.info(f"Building pair {us}/{hk} (ratio 1:{ratio})")
        pair = build_pair_frame(prices, us, hk, ratio, divs, log)
        trades = run_pair_backtest(pair, us, hk, ratio, log)
        all_trades.extend(trades)

    trades_df = pd.DataFrame(all_trades)
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(TRADES_CSV, index=False)
    log.info(f"Wrote {TRADES_CSV} rows={len(trades_df)}")

    summary = summarize(trades_df, log, prices_meta, div_meta)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"Wrote {SUMMARY_JSON}")
    log.info("=== iter_18 CLEANUP backtest done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
