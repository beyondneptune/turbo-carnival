"""
iter_3 BTC + ETH Perpetual Funding-Rate Cash-and-Carry combined backtest.

Extends the BTC-only iter_3 to include ETH/USDT:USDT perp funding for
diversification within the crypto carry sleeve.

For each instrument (BTC, ETH) independently run threshold cash-and-carry:
  - Entry when 30-period rolling annualized funding > 10%
  - Exit when 30-period mean falls below 4%
  - Round-trip cost = 30 bps
Then combine as an equal-weight portfolio:
  - Equal weight when both signals active
  - Full allocation when only one active
  - Flat when neither active
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
CACHE_PATH = ROOT / "data" / "cache" / "btc_eth_funding.parquet"
LOG_PATH = ROOT / "logs" / "iter_3_btc_eth_combined.log"
SUMMARY_JSON = ROOT / "results" / "iter_3_btc_eth_combined.json"

# --- params ---
SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]
START_DATE = "2021-01-01T00:00:00Z"
FUND_PER_DAY = 3          # 8h windows
FUND_PER_YEAR = 365 * FUND_PER_DAY
ENTRY_APR = 0.10          # 10% rolling APR
EXIT_APR = 0.04           # 4% rolling APR
ROLL_WINDOW = 30          # 30 funding cycles
ROUND_TRIP_COST = 0.0030  # 30 bps
PAGE_LIMIT = 1000
SLEEP_BETWEEN_PAGES = 0.4


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("iter_3_btc_eth")
    logger.setLevel(logging.INFO)
    for h in list(logger.handlers):
        logger.removeHandler(h)
    fh = logging.FileHandler(LOG_PATH, mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    sh = logging.StreamHandler()
    sh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def _fetch_one_symbol(ex: ccxt.binance, symbol: str, since: int,
                     logger: logging.Logger) -> list[dict]:
    rows: list[dict] = []
    pages = 0
    while True:
        try:
            batch = ex.fetch_funding_rate_history(symbol, since=since,
                                                  limit=PAGE_LIMIT)
        except ccxt.RateLimitExceeded as e:
            logger.warning(f"[{symbol}] Rate limit: {e}; sleeping 5s")
            time.sleep(5)
            continue
        except Exception as e:  # noqa: BLE001
            logger.error(f"[{symbol}] Fetch failed at since={since}: {e}")
            time.sleep(2)
            try:
                batch = ex.fetch_funding_rate_history(symbol, since=since,
                                                      limit=PAGE_LIMIT)
            except Exception as e2:  # noqa: BLE001
                logger.error(f"[{symbol}] Retry failed: {e2}; stopping")
                break

        if not batch:
            logger.info(f"[{symbol}] Empty batch; done.")
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
            logger.info(f"[{symbol}] page {pages}: {len(batch)} rows, "
                        f"last={pd.Timestamp(last_ts, unit='ms', tz='UTC')}")

        new_since = last_ts + 1
        if new_since <= since:
            logger.info(f"[{symbol}] No advance; stopping.")
            break
        since = new_since

        if len(batch) < PAGE_LIMIT:
            logger.info(f"[{symbol}] Short batch ({len(batch)}); end.")
            break

        time.sleep(SLEEP_BETWEEN_PAGES)

    return rows


def fetch_full_funding_history(logger: logging.Logger) -> pd.DataFrame:
    """Fetch BTC and ETH funding from Binance; cache combined."""
    cached: pd.DataFrame
    if CACHE_PATH.exists():
        logger.info(f"Loading cached funding from {CACHE_PATH}")
        cached = pd.read_parquet(CACHE_PATH)
        last_ts_by_symbol = cached.groupby("symbol")["timestamp"].max().to_dict()
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        # if both symbols have data within 1 day, cache is fresh
        all_fresh = all(
            (sym in last_ts_by_symbol and
             now_ms - last_ts_by_symbol[sym] < 24 * 60 * 60 * 1000)
            for sym in SYMBOLS
        )
        if all_fresh:
            logger.info(f"Cache fresh; {len(cached)} rows across "
                        f"{cached['symbol'].nunique()} symbols")
            return cached.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    else:
        cached = pd.DataFrame()
        last_ts_by_symbol = {}

    ex = ccxt.binance({
        "options": {"defaultType": "future"},
        "enableRateLimit": True,
    })
    ex.load_markets()
    logger.info("Markets loaded; beginning per-symbol fetch")

    all_new_rows: list[dict] = []
    for sym in SYMBOLS:
        if sym in last_ts_by_symbol:
            since = last_ts_by_symbol[sym] + 1
            logger.info(f"[{sym}] Extending from "
                        f"{pd.Timestamp(since, unit='ms', tz='UTC')}")
        else:
            since = int(pd.Timestamp(START_DATE).timestamp() * 1000)
            logger.info(f"[{sym}] Fresh fetch from {START_DATE}")
        rows = _fetch_one_symbol(ex, sym, since, logger)
        logger.info(f"[{sym}] Fetched {len(rows)} new rows")
        all_new_rows.extend(rows)

    new_df = pd.DataFrame(all_new_rows)
    if not cached.empty and not new_df.empty:
        out = pd.concat([cached, new_df], ignore_index=True)
    elif not new_df.empty:
        out = new_df
    else:
        out = cached

    out = (
        out.drop_duplicates(subset=["symbol", "timestamp"])
        .sort_values(["symbol", "timestamp"])
        .reset_index(drop=True)
    )
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CACHE_PATH, index=False)
    logger.info(f"Cached {len(out)} rows across "
                f"{out['symbol'].nunique()} symbols to {CACHE_PATH}")
    return out


def annualize(funding_rate: pd.Series) -> pd.Series:
    return funding_rate * FUND_PER_YEAR


def run_strategy_for_symbol(df_sym: pd.DataFrame,
                            label: str,
                            logger: logging.Logger) -> tuple[pd.DataFrame, list[dict]]:
    """Threshold cash-and-carry for one symbol. Returns enriched df + positions."""
    df = df_sym.copy().reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df["annualized"] = annualize(df["funding_rate"])
    df["roll_mean_apr"] = df["annualized"].rolling(ROLL_WINDOW,
                                                    min_periods=ROLL_WINDOW).mean()
    df["in_position"] = False
    df["position_id"] = 0
    df["period_funding"] = 0.0  # funding accrued this period if in position

    in_position = False
    entry_idx: int | None = None
    cum_funding = 0.0
    positions: list[dict] = []
    pos_id = 0

    for i, row in df.iterrows():
        roll = row["roll_mean_apr"]
        if pd.isna(roll):
            continue
        if not in_position:
            if roll > ENTRY_APR:
                in_position = True
                entry_idx = i
                cum_funding = 0.0
                pos_id += 1
                df.at[i, "in_position"] = True
                df.at[i, "position_id"] = pos_id
                # entry window — pay round-trip half here (track via per-position cost)
                continue
        else:
            df.at[i, "in_position"] = True
            df.at[i, "position_id"] = pos_id
            df.at[i, "period_funding"] = float(row["funding_rate"])
            cum_funding += float(row["funding_rate"])
            if roll < EXIT_APR:
                entry_row = df.loc[entry_idx]
                hold_cycles = i - entry_idx
                hold_days = hold_cycles / FUND_PER_DAY
                gross_carry = cum_funding
                net_carry = gross_carry - ROUND_TRIP_COST
                ann_gross = (gross_carry / hold_days) * 365 if hold_days > 0 else 0.0
                ann_net = (net_carry / hold_days) * 365 if hold_days > 0 else 0.0
                positions.append({
                    "id": pos_id,
                    "symbol": label,
                    "entry_idx": int(entry_idx),
                    "exit_idx": int(i),
                    "entry_dt": str(entry_row["dt"]),
                    "exit_dt": str(row["dt"]),
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

    # close any open
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
            "symbol": label,
            "entry_idx": int(entry_idx),
            "exit_idx": int(last_i),
            "entry_dt": str(entry_row["dt"]),
            "exit_dt": str(row["dt"]),
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

    logger.info(f"[{label}] Closed positions: {len(positions)}")
    return df, positions


def per_cycle_pnl(df: pd.DataFrame) -> pd.Series:
    """Return per-cycle net pnl (fraction) on a notional 1 unit allocated to this leg.

    - When in_position is True on a non-entry, non-exit bar: pnl = funding_rate (you
      collect that period's funding).
    - On exit bar: pnl = funding_rate - ROUND_TRIP_COST (we charge round-trip cost
      once at exit for simplicity).
    - On entry bar: pnl = 0 (open at the close of that bar).
    - When out of position: pnl = 0.
    """
    df = df.copy()
    pnl = pd.Series(0.0, index=df.index)
    pos = df["in_position"].astype(bool).to_numpy()
    funding = df["funding_rate"].fillna(0.0).to_numpy()
    pid = df["position_id"].to_numpy()
    n = len(df)
    for i in range(n):
        if not pos[i]:
            continue
        # entry bar: previous bar same position? if i==0 or prev was not in pos
        is_entry = (i == 0) or (not pos[i - 1])
        is_exit = (i == n - 1) or (not pos[i + 1])
        if is_entry:
            # no funding accrued on entry bar (we open AT this bar)
            pnl.iloc[i] = 0.0
        else:
            pnl.iloc[i] = funding[i]
        if is_exit and not is_entry:
            # charge full round-trip cost once on exit
            pnl.iloc[i] = pnl.iloc[i] - ROUND_TRIP_COST
        elif is_exit and is_entry:
            # one-bar "position" — net is just -cost
            pnl.iloc[i] = -ROUND_TRIP_COST
    return pnl


def summarize_leg(df: pd.DataFrame, positions: list[dict], pnl: pd.Series,
                  label: str) -> dict:
    n_cycles = int(df["funding_rate"].notna().sum())
    mean_apr = float(df["annualized"].mean() * 100) if n_cycles else 0.0
    span_start = pd.to_datetime(df["timestamp"].min(), unit="ms", utc=True)
    span_end = pd.to_datetime(df["timestamp"].max(), unit="ms", utc=True)
    span_days = (span_end - span_start).total_seconds() / 86400

    total_carry_pct = float(pnl.sum() * 100)  # equals sum of position net carries
    ann_return_pct = (total_carry_pct / span_days) * 365 if span_days > 0 else 0.0

    # Sharpe from per-cycle pnl: annualize (FUND_PER_YEAR cycles/yr)
    if pnl.std() > 0:
        sharpe = float((pnl.mean() / pnl.std()) * np.sqrt(FUND_PER_YEAR))
    else:
        sharpe = 0.0

    # max drawdown on cumulative pnl
    cum = pnl.cumsum()
    running_max = cum.cummax()
    drawdown = cum - running_max
    max_dd_pct = float(drawdown.min() * 100)

    return {
        "symbol": label,
        "data_start": str(span_start),
        "data_end": str(span_end),
        "span_days": float(span_days),
        "total_funding_cycles": n_cycles,
        "mean_annualized_funding_pct": mean_apr,
        "positions": len(positions),
        "total_carry_pct": total_carry_pct,
        "ann_return_pct": float(ann_return_pct),
        "sharpe": sharpe,
        "max_dd_pct": max_dd_pct,
    }


def combine_portfolio(df_btc: pd.DataFrame, pnl_btc: pd.Series,
                      df_eth: pd.DataFrame, pnl_eth: pd.Series,
                      logger: logging.Logger) -> tuple[dict, pd.DataFrame]:
    """Align BTC and ETH on timestamp, build equal-weight portfolio pnl.

    Weight rule:
      - both in position: 0.5 each
      - only one in position: 1.0 that leg
      - neither: 0
    """
    a = df_btc[["timestamp", "in_position"]].rename(columns={"in_position": "in_btc"}).copy()
    a["pnl_btc"] = pnl_btc.values
    b = df_eth[["timestamp", "in_position"]].rename(columns={"in_position": "in_eth"}).copy()
    b["pnl_eth"] = pnl_eth.values

    merged = pd.merge(a, b, on="timestamp", how="outer").sort_values("timestamp").reset_index(drop=True)
    merged["in_btc"] = merged["in_btc"].fillna(False).astype(bool)
    merged["in_eth"] = merged["in_eth"].fillna(False).astype(bool)
    merged["pnl_btc"] = merged["pnl_btc"].fillna(0.0)
    merged["pnl_eth"] = merged["pnl_eth"].fillna(0.0)

    # weights
    both = merged["in_btc"] & merged["in_eth"]
    only_btc = merged["in_btc"] & ~merged["in_eth"]
    only_eth = ~merged["in_btc"] & merged["in_eth"]

    w_btc = np.where(both, 0.5, np.where(only_btc, 1.0, 0.0))
    w_eth = np.where(both, 0.5, np.where(only_eth, 1.0, 0.0))
    merged["w_btc"] = w_btc
    merged["w_eth"] = w_eth
    merged["pnl_combined"] = w_btc * merged["pnl_btc"] + w_eth * merged["pnl_eth"]

    # stats
    pnl_c = merged["pnl_combined"]
    total_carry_pct = float(pnl_c.sum() * 100)
    span_start = pd.to_datetime(merged["timestamp"].min(), unit="ms", utc=True)
    span_end = pd.to_datetime(merged["timestamp"].max(), unit="ms", utc=True)
    span_days = (span_end - span_start).total_seconds() / 86400
    ann_return_pct = (total_carry_pct / span_days) * 365 if span_days > 0 else 0.0
    sharpe = float((pnl_c.mean() / pnl_c.std()) * np.sqrt(FUND_PER_YEAR)) if pnl_c.std() > 0 else 0.0
    cum = pnl_c.cumsum()
    drawdown = cum - cum.cummax()
    max_dd_pct = float(drawdown.min() * 100)

    # positions counted: number of distinct "active" stretches across either leg
    active = (merged["in_btc"] | merged["in_eth"]).astype(int).to_numpy()
    transitions = int(np.sum((active[1:] == 1) & (active[:-1] == 0))) + int(active[0] == 1)

    # correlation across cycles where at least one leg active (informative window)
    mask_either = merged["in_btc"] | merged["in_eth"]
    if mask_either.sum() > 2 and merged.loc[mask_either, "pnl_btc"].std() > 0 \
            and merged.loc[mask_either, "pnl_eth"].std() > 0:
        corr_active = float(merged.loc[mask_either, "pnl_btc"].corr(
            merged.loc[mask_either, "pnl_eth"]))
    else:
        corr_active = float("nan")
    # full-sample corr of raw funding-driven legs (use pnl when leg in position,
    # 0 otherwise) — keeps the corr a portfolio property, not a regime artifact.
    corr_full = float(merged["pnl_btc"].corr(merged["pnl_eth"])) \
        if merged["pnl_btc"].std() > 0 and merged["pnl_eth"].std() > 0 else float("nan")

    # fraction of time each regime is active
    cycles = len(merged)
    frac_both = float(both.sum()) / cycles if cycles else 0.0
    frac_only_btc = float(only_btc.sum()) / cycles if cycles else 0.0
    frac_only_eth = float(only_eth.sum()) / cycles if cycles else 0.0
    frac_flat = 1.0 - frac_both - frac_only_btc - frac_only_eth

    out = {
        "data_start": str(span_start),
        "data_end": str(span_end),
        "span_days": float(span_days),
        "cycles": cycles,
        "positions": transitions,
        "total_carry_pct": total_carry_pct,
        "ann_return_pct": float(ann_return_pct),
        "sharpe": sharpe,
        "max_dd_pct": max_dd_pct,
        "correlation_btc_eth_active": corr_active,
        "correlation_btc_eth_full": corr_full,
        "frac_both_active": frac_both,
        "frac_only_btc": frac_only_btc,
        "frac_only_eth": frac_only_eth,
        "frac_flat": frac_flat,
    }
    logger.info(f"Combined portfolio: {out}")
    return out, merged


def main() -> None:
    logger = setup_logging()
    logger.info("=== iter_3 BTC + ETH funding combined backtest ===")

    df_raw = fetch_full_funding_history(logger)
    counts = df_raw.groupby("symbol").size().to_dict()
    logger.info(f"Funding rows per symbol: {counts}")

    # split by symbol
    df_btc_raw = df_raw[df_raw["symbol"] == "BTC/USDT:USDT"].copy()
    df_eth_raw = df_raw[df_raw["symbol"] == "ETH/USDT:USDT"].copy()

    df_btc, pos_btc = run_strategy_for_symbol(df_btc_raw, "BTC", logger)
    df_eth, pos_eth = run_strategy_for_symbol(df_eth_raw, "ETH", logger)

    pnl_btc = per_cycle_pnl(df_btc)
    pnl_eth = per_cycle_pnl(df_eth)

    sum_btc = summarize_leg(df_btc, pos_btc, pnl_btc, "BTC")
    sum_eth = summarize_leg(df_eth, pos_eth, pnl_eth, "ETH")
    sum_combo, df_combo = combine_portfolio(df_btc, pnl_btc, df_eth, pnl_eth, logger)

    logger.info(f"BTC summary: {sum_btc}")
    logger.info(f"ETH summary: {sum_eth}")

    # verdict
    best_individual = max(sum_btc["ann_return_pct"], sum_eth["ann_return_pct"])
    combo_ann = sum_combo["ann_return_pct"]
    if combo_ann > best_individual * 1.10:
        verdict = "BETTER"
    elif combo_ann < best_individual * 0.90:
        verdict = "WORSE"
    else:
        verdict = "SAME"

    # also evaluate risk-adjusted (sharpe) for context
    best_sharpe = max(sum_btc["sharpe"], sum_eth["sharpe"])
    combo_sharpe = sum_combo["sharpe"]
    if combo_sharpe > best_sharpe * 1.10:
        sharpe_verdict = "BETTER"
    elif combo_sharpe < best_sharpe * 0.90:
        sharpe_verdict = "WORSE"
    else:
        sharpe_verdict = "SAME"

    result = {
        "config": {
            "symbols": SYMBOLS,
            "start_date": START_DATE,
            "entry_apr_pct": ENTRY_APR * 100,
            "exit_apr_pct": EXIT_APR * 100,
            "roll_window_cycles": ROLL_WINDOW,
            "round_trip_cost_bps": ROUND_TRIP_COST * 10000,
        },
        "btc_only": {
            "ann_return_pct": sum_btc["ann_return_pct"],
            "sharpe": sum_btc["sharpe"],
            "max_dd_pct": sum_btc["max_dd_pct"],
            "positions": sum_btc["positions"],
            "total_carry_pct": sum_btc["total_carry_pct"],
            "mean_annualized_funding_pct": sum_btc["mean_annualized_funding_pct"],
            "span_days": sum_btc["span_days"],
        },
        "eth_only": {
            "ann_return_pct": sum_eth["ann_return_pct"],
            "sharpe": sum_eth["sharpe"],
            "max_dd_pct": sum_eth["max_dd_pct"],
            "positions": sum_eth["positions"],
            "total_carry_pct": sum_eth["total_carry_pct"],
            "mean_annualized_funding_pct": sum_eth["mean_annualized_funding_pct"],
            "span_days": sum_eth["span_days"],
        },
        "combined": {
            "ann_return_pct": sum_combo["ann_return_pct"],
            "sharpe": sum_combo["sharpe"],
            "max_dd_pct": sum_combo["max_dd_pct"],
            "positions": sum_combo["positions"],
            "total_carry_pct": sum_combo["total_carry_pct"],
            "correlation_btc_eth_active": sum_combo["correlation_btc_eth_active"],
            "correlation_btc_eth_full": sum_combo["correlation_btc_eth_full"],
            "frac_both_active": sum_combo["frac_both_active"],
            "frac_only_btc": sum_combo["frac_only_btc"],
            "frac_only_eth": sum_combo["frac_only_eth"],
            "frac_flat": sum_combo["frac_flat"],
            "span_days": sum_combo["span_days"],
            "cycles": sum_combo["cycles"],
        },
        "verdict_return": verdict,
        "verdict_sharpe": sharpe_verdict,
        "verdict": verdict,  # primary verdict per spec is on return
    }

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_JSON, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info(f"Wrote {SUMMARY_JSON}")

    corr_used = sum_combo["correlation_btc_eth_active"]
    print(
        f"RESULT_LINE: BTC={sum_btc['ann_return_pct']:.2f}% APR, "
        f"ETH={sum_eth['ann_return_pct']:.2f}% APR, "
        f"Combined={sum_combo['ann_return_pct']:.2f}% APR | "
        f"Correlation={corr_used:.3f}"
    )
    print(
        f"VERDICT(return): {verdict} | VERDICT(sharpe): {sharpe_verdict} | "
        f"Sharpe BTC={sum_btc['sharpe']:.2f}, ETH={sum_eth['sharpe']:.2f}, "
        f"Combined={sum_combo['sharpe']:.2f}"
    )
    print(f"PATH: {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
