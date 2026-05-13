"""
iter_3 BTC Perpetual Funding-Rate Cash-and-Carry Arbitrage backtest.

Pulls BTCUSDT perp funding rate history from Binance via ccxt, then simulates
a threshold-driven cash-and-carry strategy:
  - Entry when 30-period rolling mean annualized funding > 10%
  - Exit when 30-period mean falls below 4%
  - Round-trip cost = 30 bps (taker x2 + bridge)
  - Per-position carry = sum(funding_rate) - 0.0030
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd

# --- paths ---
ROOT = Path("/Users/weeoto/turbo-carnival/research")
CACHE_PATH = ROOT / "data" / "cache" / "btc_funding.parquet"
LOG_PATH = ROOT / "logs" / "iter_3_backtest.log"
HISTORY_CSV = ROOT / "results" / "iter_3_funding_history.csv"
SUMMARY_JSON = ROOT / "results" / "iter_3_summary.json"

# --- params (per spec) ---
SYMBOL = "BTC/USDT:USDT"
START_DATE = "2021-01-01T00:00:00Z"
FUND_PER_DAY = 3          # 8h windows
FUND_PER_YEAR = 365 * FUND_PER_DAY
ENTRY_APR = 0.10          # 10% rolling APR
EXIT_APR = 0.04           # 4% rolling APR
ROLL_WINDOW = 30          # 30 funding cycles
ROUND_TRIP_COST = 0.0030  # 30 bps
PAGE_LIMIT = 1000         # ccxt default page size for funding hist
SLEEP_BETWEEN_PAGES = 0.4 # respect rate limit


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("iter_3")
    logger.setLevel(logging.INFO)
    # avoid dup handlers on rerun
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(LOG_PATH, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def fetch_full_funding_history(logger: logging.Logger) -> pd.DataFrame:
    """Page through Binance funding history from START_DATE to now."""
    if CACHE_PATH.exists():
        logger.info(f"Loading cached funding from {CACHE_PATH}")
        df = pd.read_parquet(CACHE_PATH)
        # if cache is recent (within 1 day) keep it; else extend
        last_ts = df["timestamp"].max()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        if now_ms - last_ts < 24 * 60 * 60 * 1000:
            logger.info(f"Cache fresh; {len(df)} rows, last {pd.Timestamp(last_ts, unit='ms', tz='UTC')}")
            return df.sort_values("timestamp").reset_index(drop=True)
        else:
            logger.info(f"Cache stale; extending from {pd.Timestamp(last_ts, unit='ms', tz='UTC')}")
            since = last_ts + 1
    else:
        since = int(pd.Timestamp(START_DATE).timestamp() * 1000)
        df = pd.DataFrame()

    ex = ccxt.binance({
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
    })
    ex.load_markets()
    logger.info(f"Markets loaded; fetching {SYMBOL} funding from {pd.Timestamp(since, unit='ms', tz='UTC')}")

    rows: list[dict] = []
    pages = 0
    while True:
        try:
            batch = ex.fetch_funding_rate_history(SYMBOL, since=since, limit=PAGE_LIMIT)
        except ccxt.RateLimitExceeded as e:
            logger.warning(f"Rate limit: {e}; sleeping 5s")
            time.sleep(5)
            continue
        except Exception as e:  # noqa: BLE001
            logger.error(f"Fetch failed at since={since}: {e}")
            time.sleep(2)
            # retry once
            try:
                batch = ex.fetch_funding_rate_history(SYMBOL, since=since, limit=PAGE_LIMIT)
            except Exception as e2:  # noqa: BLE001
                logger.error(f"Retry failed: {e2}; stopping")
                break

        if not batch:
            logger.info("Empty batch; done paginating.")
            break

        for r in batch:
            rows.append({
                "timestamp": r["timestamp"],
                "datetime": r["datetime"],
                "funding_rate": float(r["fundingRate"]) if r["fundingRate"] is not None else np.nan,
                "symbol": r["symbol"],
            })

        pages += 1
        last_ts = batch[-1]["timestamp"]
        if pages % 5 == 0:
            logger.info(f"page {pages}: got {len(batch)} rows, last={pd.Timestamp(last_ts, unit='ms', tz='UTC')}")

        # advance; +1ms to avoid dup on next call
        new_since = last_ts + 1
        if new_since <= since:
            logger.info("No advance; stopping.")
            break
        since = new_since

        if len(batch) < PAGE_LIMIT:
            logger.info(f"Short batch ({len(batch)} < {PAGE_LIMIT}); reached end.")
            break

        time.sleep(SLEEP_BETWEEN_PAGES)

    new_df = pd.DataFrame(rows)
    if not df.empty and not new_df.empty:
        out = pd.concat([df, new_df], ignore_index=True)
    elif not new_df.empty:
        out = new_df
    else:
        out = df

    out = (
        out.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE_PATH, index=False)
    logger.info(f"Cached {len(out)} rows to {CACHE_PATH}")
    return out


def annualize(funding_rate: pd.Series) -> pd.Series:
    return funding_rate * FUND_PER_YEAR


def run_strategy(df: pd.DataFrame, logger: logging.Logger) -> tuple[pd.DataFrame, list[dict]]:
    """Threshold cash-and-carry: enter on rolling-mean APR > 10%, exit < 4%."""
    df = df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["annualized"] = annualize(df["funding_rate"])
    df["roll_mean_apr"] = df["annualized"].rolling(ROLL_WINDOW, min_periods=ROLL_WINDOW).mean()

    in_position = False
    entry_idx: int | None = None
    cum_funding = 0.0
    positions: list[dict] = []
    pos_id = 0

    for i, row in df.iterrows():
        roll = row["roll_mean_apr"]
        # need warm-up window
        if pd.isna(roll):
            continue
        if not in_position:
            if roll > ENTRY_APR:
                in_position = True
                entry_idx = i
                cum_funding = 0.0
                pos_id += 1
                # entry funding (we open AT this window, harvest from next)
                continue
        else:
            # accumulate THIS window's funding (we are holding)
            cum_funding += float(row["funding_rate"])
            if roll < EXIT_APR:
                # close
                entry_row = df.loc[entry_idx]
                hold_cycles = i - entry_idx
                hold_days = hold_cycles / FUND_PER_DAY
                gross_carry = cum_funding
                net_carry = gross_carry - ROUND_TRIP_COST
                ann_gross = (gross_carry / hold_days) * 365 if hold_days > 0 else 0.0
                ann_net = (net_carry / hold_days) * 365 if hold_days > 0 else 0.0
                positions.append({
                    "id": pos_id,
                    "entry_idx": int(entry_idx),
                    "exit_idx": int(i),
                    "entry_dt": entry_row["dt"],
                    "exit_dt": row["dt"],
                    "hold_cycles": int(hold_cycles),
                    "hold_days": float(hold_days),
                    "gross_carry_pct": float(gross_carry * 100),
                    "net_carry_pct": float(net_carry * 100),
                    "annualized_gross_pct": float(ann_gross * 100),
                    "annualized_net_pct": float(ann_net * 100),
                    "entry_roll_apr_pct": float(entry_row["roll_mean_apr"] * 100),
                    "exit_roll_apr_pct": float(row["roll_mean_apr"] * 100),
                })
                in_position = False
                entry_idx = None
                cum_funding = 0.0

    # close any open position at end of data
    if in_position and entry_idx is not None:
        last_i = len(df) - 1
        entry_row = df.loc[entry_idx]
        row = df.loc[last_i]
        hold_cycles = last_i - entry_idx
        hold_days = hold_cycles / FUND_PER_DAY
        gross_carry = cum_funding
        net_carry = gross_carry - ROUND_TRIP_COST
        ann_gross = (gross_carry / hold_days) * 365 if hold_days > 0 else 0.0
        ann_net = (net_carry / hold_days) * 365 if hold_days > 0 else 0.0
        positions.append({
            "id": pos_id,
            "entry_idx": int(entry_idx),
            "exit_idx": int(last_i),
            "entry_dt": entry_row["dt"],
            "exit_dt": row["dt"],
            "hold_cycles": int(hold_cycles),
            "hold_days": float(hold_days),
            "gross_carry_pct": float(gross_carry * 100),
            "net_carry_pct": float(net_carry * 100),
            "annualized_gross_pct": float(ann_gross * 100),
            "annualized_net_pct": float(ann_net * 100),
            "entry_roll_apr_pct": float(entry_row["roll_mean_apr"] * 100),
            "exit_roll_apr_pct": float(row["roll_mean_apr"] * 100),
            "open_at_end": True,
        })

    logger.info(f"Closed positions: {len(positions)}")
    return df, positions


def summarize(df: pd.DataFrame, positions: list[dict], logger: logging.Logger) -> dict:
    n_cycles = int(df["funding_rate"].notna().sum())
    mean_apr = float(df["annualized"].mean() * 100) if n_cycles else 0.0
    periods_above = int((df["annualized"] > ENTRY_APR).sum())

    if positions:
        pos_df = pd.DataFrame(positions)
        total_carry_harvested_pct = float(pos_df["net_carry_pct"].sum())
        total_hold_days = float(pos_df["hold_days"].sum())
        # capital deployed during in-position days; out-of-position days earn 0
        # Define annualized as: total carry harvested / total CALENDAR span * 365.
        # Calendar span = first entry to last exit (or full data span if we want capital deployed always).
        # Per spec we annualize the harvested returns -- assume capital sits idle when flat,
        # so annualize over the full data span to be conservative-realistic.
        span_start = pd.to_datetime(df["timestamp"].min(), unit="ms", utc=True)
        span_end = pd.to_datetime(df["timestamp"].max(), unit="ms", utc=True)
        span_days = (span_end - span_start).total_seconds() / 86400
        annualized_carry_net_pct = (total_carry_harvested_pct / span_days) * 365 if span_days > 0 else 0.0
        # in-position annualized (carry while deployed)
        in_pos_annualized_pct = (total_carry_harvested_pct / total_hold_days) * 365 if total_hold_days > 0 else 0.0

        # Sharpe proxy: per-position net carry annualized, mean / std
        per_pos_ann = pos_df["annualized_net_pct"].to_numpy()
        if len(per_pos_ann) > 1 and per_pos_ann.std() > 0:
            sharpe_proxy = float(per_pos_ann.mean() / per_pos_ann.std())
        else:
            sharpe_proxy = 0.0
    else:
        total_carry_harvested_pct = 0.0
        annualized_carry_net_pct = 0.0
        in_pos_annualized_pct = 0.0
        sharpe_proxy = 0.0
        span_days = (pd.to_datetime(df["timestamp"].max(), unit="ms", utc=True) -
                     pd.to_datetime(df["timestamp"].min(), unit="ms", utc=True)).total_seconds() / 86400

    verdict = "PASS" if annualized_carry_net_pct >= 6.0 else "FAIL"

    summary = {
        "symbol": SYMBOL,
        "data_start": str(pd.to_datetime(df["timestamp"].min(), unit="ms", utc=True)),
        "data_end": str(pd.to_datetime(df["timestamp"].max(), unit="ms", utc=True)),
        "span_days": float(span_days),
        "total_funding_cycles": n_cycles,
        "mean_annualized_funding_pct": mean_apr,
        "periods_above_10pct": periods_above,
        "n_positions": len(positions),
        "total_carry_harvested_pct": total_carry_harvested_pct,
        "annualized_carry_net_pct": float(annualized_carry_net_pct),
        "in_position_annualized_pct": float(in_pos_annualized_pct),
        "sharpe_proxy": sharpe_proxy,
        "entry_threshold_apr_pct": ENTRY_APR * 100,
        "exit_threshold_apr_pct": EXIT_APR * 100,
        "round_trip_cost_bps": ROUND_TRIP_COST * 10000,
        "rolling_window_cycles": ROLL_WINDOW,
        "verdict": verdict,
    }
    logger.info(f"Summary: {summary}")
    return summary


def main() -> None:
    logger = setup_logging()
    logger.info("=== iter_3 BTC perp funding cash-and-carry backtest ===")

    df_raw = fetch_full_funding_history(logger)
    logger.info(f"Raw funding rows: {len(df_raw)}, "
                f"range {pd.to_datetime(df_raw['timestamp'].min(), unit='ms', utc=True)} "
                f"to {pd.to_datetime(df_raw['timestamp'].max(), unit='ms', utc=True)}")

    df, positions = run_strategy(df_raw, logger)
    summary = summarize(df, positions, logger)

    # save history (timestamp, rate, annualized, rolling, regime)
    HISTORY_CSV.parent.mkdir(parents=True, exist_ok=True)
    out_df = df[["timestamp", "dt", "funding_rate", "annualized", "roll_mean_apr"]].copy()
    out_df.columns = ["timestamp_ms", "datetime", "funding_rate", "annualized_apr", "roll30_mean_apr"]
    out_df.to_csv(HISTORY_CSV, index=False)
    logger.info(f"Wrote {HISTORY_CSV} ({len(out_df)} rows)")

    # save positions (for inspection)
    if positions:
        pos_df = pd.DataFrame(positions)
        pos_csv = HISTORY_CSV.with_name("iter_3_positions.csv")
        pos_df.to_csv(pos_csv, index=False)
        logger.info(f"Wrote {pos_csv} ({len(pos_df)} positions)")

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"Wrote {SUMMARY_JSON}")

    # final stdout line for caller
    print(
        f"RESULT_LINE: iter_3 | cycles={summary['total_funding_cycles']} | "
        f"mean_fund={summary['mean_annualized_funding_pct']:.2f}% APR | "
        f"net_carry={summary['annualized_carry_net_pct']:.2f}% APR | "
        f"VERDICT: {summary['verdict']}"
    )


if __name__ == "__main__":
    main()
