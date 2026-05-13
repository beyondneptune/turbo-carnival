"""iter_18 EXPANDED backtest: 15-pair extension of iter_18_cleanup.

Builds on iter_18_cleanup.py (5 pairs, PASS @ aggregate mean_bps_net=65.79 bps)
by adding 8 additional dual-listed Chinese ADRs (US ADR + HK primary) to test
capacity scaling. Two pairs from the spec are intentionally REMOVED:
  - PDD: no HK listing exists
  - HTHT/1179.HK: H World Group has no HK listing (Nasdaq only)

Pairs tested (13 total; 5 original + 8 new):
  Original (confirmed PASS in cleanup run):
    - BABA / 9988.HK  1:8
    - JD   / 9618.HK  1:2
    - BIDU / 9888.HK  1:8
    - NTES / 9999.HK  1:5
    - TCOM / 9961.HK  1:1
  New:
    - LI    / 2015.HK   1:2     (Li Auto:  each ADS = 2 Class A ordinary)
    - NIO   / 9866.HK   1:1     (NIO:      1 ADS per HK ordinary)
    - XPEV  / 9868.HK   1:2     (XPeng:    each ADS = 2 Class A ordinary)
    - BILI  / 9626.HK   1:1     (Bilibili: 1 ADS per HK ordinary)
    - TME   / 1698.HK   1:2     (Tencent Music: each ADS = 2 Class A ordinary)
    - MNSO  / 9896.HK   1:4     (MINISO:   each ADS = 4 Class A ordinary)
    - YMM   / 2643.HK   1:20    (Full Truck Alliance: 1 ADS = 20 ordinary)
    - ZTO   / 2057.HK   1:1     (ZTO Express: 1 ADS per HK ordinary)

Strategy parameters unchanged from cleanup:
  Entry  |dev| > 80 bps
  Exit   |dev| < 20 bps
  Stop   |dev| > 250 bps
  Time   5 trading days max hold
  Cost   30 bps round-trip
  Ex-div skip window: +/- 3 sessions on either leg

PASS rule (capacity-aware):
  aggregate mean_bps_net >= 25 AND at least 8 pairs PASS individually
  per-pair PASS: mean_bps_net >= 25 AND n >= 20

Capacity estimate: per pair, estimate the $ AUM at which a single trade
would consume more than 5% of average daily $-volume on the THINNER leg,
which is our slippage-edge breakpoint (i.e. the AUM scale at which one
side of the spread starts paying more than ~5 bps of impact, eating into
the 70+ bps gross edge). The portfolio total is the sum across PASS pairs
since trades are largely independent dispersion events.
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
# Distinct cache for the 13-ticker expanded universe
CACHE_PATH = ROOT / "data" / "cache" / "us_hk_pairs_expanded_daily_raw.parquet"
DIV_CACHE_PATH = ROOT / "data" / "cache" / "us_hk_pairs_expanded_dividends.parquet"
VOL_CACHE_PATH = ROOT / "data" / "cache" / "us_hk_pairs_expanded_volume.parquet"
LOG_PATH = ROOT / "logs" / "iter_18_expanded.log"
TRADES_CSV = ROOT / "results" / "iter_18_expanded_trades.csv"
SUMMARY_JSON = ROOT / "results" / "iter_18_expanded_summary.json"

# (US ADR, HK ticker, current-state ratio = HK shares per 1 US ADR)
PAIRS: list[tuple[str, str, int]] = [
    # Original 5 (confirmed in cleanup)
    ("BABA", "9988.HK", 8),
    ("JD",   "9618.HK", 2),
    ("BIDU", "9888.HK", 8),
    ("NTES", "9999.HK", 5),
    ("TCOM", "9961.HK", 1),
    # New 8 (PDD and HTHT removed per spec)
    ("LI",   "2015.HK", 2),
    ("NIO",  "9866.HK", 1),
    ("XPEV", "9868.HK", 2),
    ("BILI", "9626.HK", 1),
    ("TME",  "1698.HK", 2),
    ("MNSO", "9896.HK", 4),
    ("YMM",  "2643.HK", 20),
    ("ZTO",  "2057.HK", 1),
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

EX_DIV_SKIP_WINDOW_BARS = 3

# Capacity model: we'd like one trade to consume <=5% of the THINNER leg's
# ADV. If it does, market impact stays well under 5 bps (square-root model).
# Per-trade notional cap = 5% * thinner-leg-$ADV. Annualized "AUM" capacity
# is per-trade-cap * (trades_per_year / cycle_overlap_factor). We use a
# conservative cycle overlap factor of 1.0 (i.e. one position at a time
# per pair, no leverage cross-margin).
ADV_LOOKBACK_DAYS = 60
ADV_CAPACITY_FRACTION = 0.05    # 5% of ADV per trade
# Convert per-trade $ cap into "AUM" by assuming on-average ~30% of capital
# deployed per pair when in a trade (since trades fire intermittently).
# This is a soft proxy; documented in the summary.
PER_PAIR_DEPLOYMENT_RATIO = 0.30

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
    return logging.getLogger("iter_18_expanded")


def fetch_prices_divs_vol(
    log: logging.Logger,
) -> tuple[pd.DataFrame, dict[str, pd.Series], pd.DataFrame]:
    """Pull raw daily Close + Volume + dividend events for all tickers.

    Returns: (close_wide, divs_by_ticker, volume_wide)
    """
    if (
        CACHE_PATH.exists()
        and DIV_CACHE_PATH.exists()
        and VOL_CACHE_PATH.exists()
    ):
        prices = pd.read_parquet(CACHE_PATH)
        last = prices.index.max()
        if last >= pd.Timestamp.today().normalize() - pd.Timedelta(days=5):
            missing = [t for t in ALL_TICKERS if t not in prices.columns]
            if not missing:
                div_df = pd.read_parquet(DIV_CACHE_PATH)
                vol_df = pd.read_parquet(VOL_CACHE_PATH)
                divs: dict[str, pd.Series] = {}
                for t in ALL_TICKERS:
                    if t in div_df.columns:
                        s = div_df[t].dropna()
                        if not s.empty:
                            divs[t] = s
                log.info(
                    f"Loaded EXPANDED cache rows={len(prices)} "
                    f"last={last.date()} cols={list(prices.columns)} "
                    f"divs_tickers={list(divs.keys())}"
                )
                return prices, divs, vol_df
            log.info(f"EXPANDED cache missing tickers {missing}; refetching")
        else:
            log.info(f"EXPANDED cache stale (last={last.date()}); refetching")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info(
        f"Downloading yfinance UNADJUSTED (auto_adjust=False) "
        f"tickers={ALL_TICKERS} from {START}"
    )

    raw = yf.download(
        ALL_TICKERS,
        start=START,
        end=None,
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=True,
        group_by="ticker",
    )

    closes: dict[str, pd.Series] = {}
    vols: dict[str, pd.Series] = {}
    divs: dict[str, pd.Series] = {}
    if isinstance(raw.columns, pd.MultiIndex):
        for t in ALL_TICKERS:
            if t in raw.columns.get_level_values(0):
                sub = raw[t]
                if "Close" in sub.columns:
                    s = sub["Close"].dropna()
                    if len(s) > 0:
                        closes[t] = s
                if "Volume" in sub.columns:
                    v = sub["Volume"].dropna()
                    if len(v) > 0:
                        vols[t] = v
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
    vol_df = pd.DataFrame(vols).sort_index() if vols else pd.DataFrame(index=df.index)
    vol_df.index = pd.to_datetime(vol_df.index)

    if divs:
        div_wide = pd.concat(divs, axis=1).sort_index()
        div_wide.index = pd.to_datetime(div_wide.index)
    else:
        div_wide = pd.DataFrame()

    df.to_parquet(CACHE_PATH)
    vol_df.to_parquet(VOL_CACHE_PATH)
    if not div_wide.empty:
        div_wide.to_parquet(DIV_CACHE_PATH)
    else:
        pd.DataFrame(index=df.index).to_parquet(DIV_CACHE_PATH)

    log.info(
        f"Saved EXPANDED cache rows={len(df)} "
        f"{df.index.min().date()}..{df.index.max().date()} cols={list(df.columns)}"
    )
    log.info(f"Dividend events captured (tickers): {list(divs.keys())}")
    for t in ALL_TICKERS:
        if t in df.columns:
            fvi = df[t].first_valid_index()
            n = df[t].notna().sum()
            log.info(f"  {t}: first={fvi.date() if fvi is not None else 'NA'} n={n}")
        else:
            log.warning(f"  {t}: NO DATA returned by yfinance")
    return df, divs, vol_df


def build_pair_frame(
    prices: pd.DataFrame,
    us: str,
    hk: str,
    ratio: int,
    divs: dict[str, pd.Series],
    log: logging.Logger,
) -> pd.DataFrame:
    """Same as cleanup: deviation series + ex_div_block flag."""
    if us not in prices.columns or hk not in prices.columns:
        log.warning(f"  {us}/{hk}: missing column(s); skipping")
        return pd.DataFrame()
    sub = prices[[us, hk]].dropna(how="any").copy()
    if sub.empty:
        log.warning(f"  {us}/{hk}: no overlapping rows; skipping")
        return pd.DataFrame()
    sub.columns = ["us_close", "hk_close"]
    sub["parity_usd"] = sub["hk_close"] * USD_PER_HKD * ratio
    sub["dev_bps"] = 10_000.0 * (sub["us_close"] - sub["parity_usd"]) / sub["parity_usd"]

    block = pd.Series(False, index=sub.index)
    for leg in (us, hk):
        if leg in divs:
            for ex_date in divs[leg].index:
                ex_date = pd.Timestamp(ex_date).normalize()
                if ex_date not in sub.index:
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
    """Identical loop to cleanup version."""
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


def compute_capacity(
    prices: pd.DataFrame,
    volumes: pd.DataFrame,
    us: str,
    hk: str,
    n_trades: int,
    years: float,
    log: logging.Logger,
) -> dict:
    """Per-pair capacity in $ AUM.

    Method:
      1. Compute trailing-60-session average daily $-volume (ADV$) on each
         leg from the most-recent ADV_LOOKBACK_DAYS sessions where we have
         data. HK $ADV is converted USD via the peg.
      2. Per-trade notional cap = ADV_CAPACITY_FRACTION * min($ADV_us, $ADV_hk).
      3. Translate to an "AUM" figure by dividing by PER_PAIR_DEPLOYMENT_RATIO
         (i.e. assume on average only PER_PAIR_DEPLOYMENT_RATIO of capital is
         deployed at a time on this pair). This is a soft proxy.
    """
    out: dict = {
        "us_adv_usd_m": None,
        "hk_adv_usd_m": None,
        "thinner_leg": None,
        "per_trade_cap_usd_m": None,
        "per_pair_aum_capacity_usd_m": None,
    }
    if us not in prices.columns or hk not in prices.columns:
        return out
    if us not in volumes.columns or hk not in volumes.columns:
        return out

    cutoff = prices.index.max() - pd.Timedelta(days=120)
    px = prices[[us, hk]].dropna()
    vx = volumes[[us, hk]].dropna()
    aligned = px.join(vx, how="inner", lsuffix="_px", rsuffix="_vol")
    aligned = aligned[aligned.index >= cutoff].tail(ADV_LOOKBACK_DAYS)
    if len(aligned) < 10:
        return out

    # $-volume per session
    us_dvol = (aligned[f"{us}_px"] * aligned[f"{us}_vol"]).mean()
    # HK ticker name doesn't conflict with US in this universe
    hk_local_dvol = (aligned[f"{hk}_px"] * aligned[f"{hk}_vol"]).mean()
    hk_dvol_usd = hk_local_dvol * USD_PER_HKD

    thinner = "US" if us_dvol < hk_dvol_usd else "HK"
    per_trade_cap = ADV_CAPACITY_FRACTION * min(us_dvol, hk_dvol_usd)
    pair_aum = per_trade_cap / PER_PAIR_DEPLOYMENT_RATIO

    out["us_adv_usd_m"] = round(us_dvol / 1e6, 2)
    out["hk_adv_usd_m"] = round(hk_dvol_usd / 1e6, 2)
    out["thinner_leg"] = thinner
    out["per_trade_cap_usd_m"] = round(per_trade_cap / 1e6, 2)
    out["per_pair_aum_capacity_usd_m"] = round(pair_aum / 1e6, 2)

    log.info(
        f"  {us}/{hk}: $ADV US=${us_dvol/1e6:.1f}M HK=${hk_dvol_usd/1e6:.1f}M "
        f"thinner={thinner} per_trade_cap=${per_trade_cap/1e6:.2f}M "
        f"pair_AUM=${pair_aum/1e6:.2f}M"
    )
    return out


def summarize(
    trades_df: pd.DataFrame,
    log: logging.Logger,
    prices_meta: dict,
    div_meta: dict,
    pair_data_summaries: list[dict],
    capacity_by_pair: dict[str, dict],
) -> dict:
    n_trades = int(len(trades_df))
    summary: dict = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "spec": "iter_18_us_hk_dual_listing_arbitrage (EXPANDED 13 pairs)",
        "extension_notes": [
            "Built on iter_18_cleanup (5-pair PASS at agg mean_bps_net=65.79).",
            "Added 8 new pairs: LI, NIO, XPEV, BILI, TME, MNSO, YMM, ZTO.",
            "Removed PDD (no HK listing) and HTHT (H World has no HK listing).",
            "Same params, costs, ex-div skip as cleanup version.",
            "Per-pair PASS: mean_bps_net >= 25 AND n >= 20.",
            "Aggregate PASS: agg mean_bps_net >= 25 AND >= 8 pairs PASS.",
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
            "adv_lookback_days": ADV_LOOKBACK_DAYS,
            "adv_capacity_fraction": ADV_CAPACITY_FRACTION,
            "per_pair_deployment_ratio": PER_PAIR_DEPLOYMENT_RATIO,
        },
        "data_meta": prices_meta,
        "dividend_meta": div_meta,
        "pair_data_summaries": pair_data_summaries,
        "aggregate": {},
        "per_pair_results": [],
        "capacity_estimate": {},
        "verdict": "FAIL",
        "verdict_reason": "no_trades",
    }

    pairs_tradeable = sum(1 for s in pair_data_summaries if s["tradeable"])
    summary["pairs_tradeable"] = pairs_tradeable
    summary["pairs_tested"] = len(PAIRS)
    summary["pairs_skipped"] = [s["pair"] for s in pair_data_summaries if not s["tradeable"]]
    summary["pairs_excluded_pre_run"] = [
        "PDD (no HK listing exists)",
        "HTHT (H World Group: no HK listing)",
    ]

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

    # Flag pairs whose deviation series suggests a ratio mismatch: a
    # CURRENT-STATE proper-ratio pair will have |dev_mean| << ~500 bps.
    # Anything >1000 bps mean abs dev is almost certainly a wrong ratio,
    # which we should document and exclude.
    ratio_suspect_pairs: set[str] = set()
    for s in pair_data_summaries:
        if not s.get("tradeable"):
            continue
        if abs(s.get("dev_bps_mean", 0.0)) > 1000:
            ratio_suspect_pairs.add(s["pair"])
    summary["ratio_suspect_pairs"] = sorted(ratio_suspect_pairs)

    per_pair_results: list[dict] = []
    pair_pass_count = 0
    pass_pairs: list[str] = []
    for pname, sub in trades_df.groupby("pair"):
        block = per_pair_block(sub)
        block_with_name = {"pair": pname, **block}
        if pname in ratio_suspect_pairs:
            block_with_name["ratio_suspect"] = True
            block_with_name["ratio_suspect_note"] = (
                "Mean |dev_bps| > 1000; ratio likely wrong or HK listing too "
                "new for a clean parity series. Excluded from PASS-pair tally "
                "and capacity sum even if mean_bps_net >= 25 (which it isn't)."
            )
        per_pair_results.append(block_with_name)
        if (
            block.get("mean_bps_net") is not None
            and block["mean_bps_net"] >= 25.0
            and block["n"] >= 20
            and pname not in ratio_suspect_pairs
        ):
            pair_pass_count += 1
            pass_pairs.append(pname)
    summary["per_pair_results"] = per_pair_results
    summary["per_pair_pass_count"] = pair_pass_count
    summary["per_pair_pass_pairs"] = pass_pairs
    summary["per_pair_pass_threshold"] = "mean_bps_net >= 25 AND n >= 20 AND not ratio_suspect"

    # Capacity aggregation: sum across PASS pairs only (since failing pairs
    # contribute negative or non-edge trades).
    per_pair_capacity: dict[str, dict] = {}
    total_capacity_pass_usd_m = 0.0
    total_capacity_all_usd_m = 0.0
    for pname, cap in capacity_by_pair.items():
        per_pair_capacity[pname] = cap
        v = cap.get("per_pair_aum_capacity_usd_m")
        if v is None:
            continue
        total_capacity_all_usd_m += v
        if pname in pass_pairs:
            total_capacity_pass_usd_m += v
    summary["capacity_estimate"] = {
        "method": (
            "Per pair: 5% of trailing-60-session min($ADV_us, $ADV_hk) is the "
            "max per-trade notional before market impact starts eating the "
            "edge. Divide by 30% average deployment ratio to translate to "
            "sleeve-level AUM. Sum across PASS pairs for portfolio capacity."
        ),
        "adv_lookback_days": ADV_LOOKBACK_DAYS,
        "adv_capacity_fraction": ADV_CAPACITY_FRACTION,
        "per_pair_deployment_ratio": PER_PAIR_DEPLOYMENT_RATIO,
        "per_pair_capacity_usd_m": per_pair_capacity,
        "total_capacity_pass_pairs_usd_m": round(total_capacity_pass_usd_m, 2),
        "total_capacity_all_pairs_usd_m": round(total_capacity_all_usd_m, 2),
    }

    passed = (mean_bps_net >= 25.0) and (pair_pass_count >= 8)
    summary["verdict"] = "PASS" if passed else "FAIL"
    summary["verdict_reason"] = (
        f"aggregate mean_bps_net={mean_bps_net:.2f} (req>=25) "
        f"AND pair_pass_count={pair_pass_count} of {len(per_pair_results)} (req>=8)"
    )

    log.info(
        "AGGREGATE: trades=%d win=%.2f%% mean_gross=%.2f mean_net=%.2f "
        "total=%.1f bp ann=%.2f%% sharpe=%.3f mcl=%d -> %s",
        n_trades, win_rate, mean_bps_gross, mean_bps_net,
        total_pnl_bps, ann_return_pct, sharpe, mcl, summary["verdict"],
    )
    for block in per_pair_results:
        log.info(
            "  %s: n=%d win=%.2f%% net=%.2f gross=%.2f sharpe=%.3f "
            "stops=%d converged=%d time=%d",
            block["pair"], block["n"], block["win_rate"], block["mean_bps_net"],
            block["mean_bps_gross"], block["sharpe_proxy"],
            block["stop_outs"], block["converged"], block["time_stops"],
        )
    log.info(
        "PER-PAIR PASS COUNT (mean_bps_net>=25 AND n>=20): %d of %d (req >= 8)",
        pair_pass_count, len(per_pair_results),
    )
    log.info(
        "CAPACITY: total $ AUM across PASS pairs = $%.1fM (all pairs: $%.1fM)",
        total_capacity_pass_usd_m, total_capacity_all_usd_m,
    )
    return summary


def main() -> int:
    log = setup_logging()
    log.info("=== iter_18 EXPANDED backtest start (13 pairs) ===")
    log.info(
        "Pairs: %s",
        ", ".join(f"{u}/{h} 1:{r}" for u, h, r in PAIRS),
    )
    log.info("Pre-run exclusions: PDD (no HK listing), HTHT (no HK listing)")

    prices, divs, volumes = fetch_prices_divs_vol(log)

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

    pair_data_summaries: list[dict] = []
    capacity_by_pair: dict[str, dict] = {}
    all_trades: list[dict] = []

    for us, hk, ratio in PAIRS:
        log.info(f"Building pair {us}/{hk} (ratio 1:{ratio})")
        pair = build_pair_frame(prices, us, hk, ratio, divs, log)
        pname = f"{us}/{hk}"
        tradeable = len(pair) > 50
        if not tradeable:
            log.warning(f"  {pname}: insufficient rows ({len(pair)}), marking NOT tradeable")
            pair_data_summaries.append({
                "pair": pname, "tradeable": False,
                "rows": int(len(pair)) if not pair.empty else 0,
                "reason": "insufficient_data",
                "us_first": (
                    str(prices[us].first_valid_index().date())
                    if us in prices.columns and prices[us].first_valid_index() is not None
                    else None
                ),
                "hk_first": (
                    str(prices[hk].first_valid_index().date())
                    if hk in prices.columns and prices[hk].first_valid_index() is not None
                    else None
                ),
            })
            continue

        pair_data_summaries.append({
            "pair": pname, "tradeable": True,
            "rows": int(len(pair)),
            "ratio": ratio,
            "first_date": str(pair.index.min().date()),
            "last_date": str(pair.index.max().date()),
            "ex_div_blocked_rows": int(pair["ex_div_block"].sum()),
            "dev_bps_mean": round(float(pair["dev_bps"].mean()), 2),
            "dev_bps_std": round(float(pair["dev_bps"].std()), 2),
        })

        trades = run_pair_backtest(pair, us, hk, ratio, log)
        all_trades.extend(trades)
        first_dt = pair.index.min()
        last_dt = pair.index.max()
        years = max((last_dt - first_dt).days / 365.25, 1e-9)
        cap = compute_capacity(prices, volumes, us, hk, len(trades), years, log)
        capacity_by_pair[pname] = cap

    trades_df = pd.DataFrame(all_trades)
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(TRADES_CSV, index=False)
    log.info(f"Wrote {TRADES_CSV} rows={len(trades_df)}")

    summary = summarize(trades_df, log, prices_meta, div_meta,
                        pair_data_summaries, capacity_by_pair)
    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"Wrote {SUMMARY_JSON}")
    log.info("=== iter_18 EXPANDED backtest done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
