"""
iter_8 backtest: Commodity ETF pair cointegration mean-reversion.

Pairs:
  - USO  / XLE  (oil futures vs energy equities)
  - GLD  / SLV  (gold vs silver)
  - UNG  / XOP  (nat gas vs oil & gas explorers)
  - DBA  / CORN (broad agriculture vs corn)

Period: 2010-01-01 -> today.
In-sample observation: 2010-2019.  Out-of-sample: 2020-2025+.

Rule (per pair):
  - Engle-Granger cointegration test on full history; skip if p > 0.05.
  - Hedge ratio = OLS regression of A on B over the full history.
  - Spread_t = A_t - hedge_ratio * B_t.
  - Rolling 60d mean and std on spread, using ONLY past data (no look-ahead) -> z-score.
  - OU half-life from AR(1) on spread.  Skip pair if H < 5 or H > 30 trading days.
  - Entry when no open position and |z| >= 2.0:
        z >= 2:  short A, long B  (spread is rich -> convergent = sell spread)
        z <= -2: long A, short B  (spread is cheap -> convergent = buy spread)
  - Exit when in position and any of:
        |z| < 0.5           -> mean-revert win
        |z| > 3.0           -> stop loss
        days_held > 3 * H   -> time stop
  - Cost: 2 bp per leg = 4 bp round-trip on each pair-trade.
  - Dollar-neutral: $1 long + $1 short on entry; PnL per trade is the log-spread move
    times direction, applied to the average notional.  We compute the trade return in
    return-on-gross terms (consistent with the spec's "bps per trade" framing).

Outputs:
  - cache:   research/data/cache/etf_pairs_daily.parquet
  - log:     research/logs/iter_8_backtest.log
  - trades:  research/results/iter_8_trades.csv  (one row per pair-trade)
  - summary: research/results/iter_8_summary.json
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller


# ----------------------------- config ------------------------------------- #

REPO = Path("/Users/weeoto/turbo-carnival")
CACHE = REPO / "research" / "data" / "cache" / "etf_pairs_daily.parquet"
LOG_PATH = REPO / "research" / "logs" / "iter_8_backtest.log"
TRADES_CSV = REPO / "research" / "results" / "iter_8_trades.csv"
SUMMARY_JSON = REPO / "research" / "results" / "iter_8_summary.json"

PAIRS = [
    ("USO", "XLE"),
    ("GLD", "SLV"),
    ("UNG", "XOP"),
    ("DBA", "CORN"),
]

START = "2010-01-01"
END = None  # today

# Cointegration / OU gates
COINT_P_MAX = 0.05
HALF_LIFE_MIN = 5.0
HALF_LIFE_MAX = 30.0

# Trading thresholds
Z_ENTRY = 2.0
Z_EXIT = 0.5
Z_STOP = 3.0

ROLL_WIN = 60               # rolling mean/std window for z-score
COST_BP_PER_LEG = 2.0        # 2 bp per leg -> 4 bp round trip
TRADING_DAYS_PER_YEAR = 252

IS_END = "2019-12-31"
OOS_START = "2020-01-01"


# ----------------------------- logging ------------------------------------ #

LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
CACHE.parent.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger("iter_8")
logger.setLevel(logging.INFO)
# Reset handlers in case the module is re-imported
for h in list(logger.handlers):
    logger.removeHandler(h)
fh = logging.FileHandler(LOG_PATH, mode="w")
sh = logging.StreamHandler(sys.stdout)
fmt = logging.Formatter("%(asctime)s  %(levelname)s  %(message)s")
fh.setFormatter(fmt)
sh.setFormatter(fmt)
logger.addHandler(fh)
logger.addHandler(sh)


# ----------------------------- data load ---------------------------------- #

def all_tickers() -> list[str]:
    seen, out = set(), []
    for a, b in PAIRS:
        for t in (a, b):
            if t not in seen:
                seen.add(t)
                out.append(t)
    return out


def load_prices(start: str = START, end: Optional[str] = END,
                refresh: bool = False) -> pd.DataFrame:
    """Return wide DataFrame of adjusted close prices indexed by date."""
    tickers = all_tickers()

    if CACHE.exists() and not refresh:
        try:
            df = pd.read_parquet(CACHE)
            missing = [t for t in tickers if t not in df.columns]
            cached_end = df.index.max()
            today = pd.Timestamp.now(tz=None).normalize()
            stale = (today - cached_end.tz_localize(None) if cached_end.tzinfo else today - cached_end).days > 7
            if not missing and not stale:
                logger.info(f"Loaded cached prices: {df.shape} from {CACHE}")
                return df
            logger.info(f"Cache present but missing={missing} stale={stale}; refreshing.")
        except Exception as e:
            logger.warning(f"Failed to read cache ({e}); refetching.")

    logger.info(f"Downloading {tickers} from yfinance {start}..{end or 'today'}")
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned empty dataframe")

    closes = {}
    for t in tickers:
        try:
            s = raw[t]["Close"].rename(t)
        except Exception:
            s = raw["Close"][t].rename(t) if isinstance(raw.columns, pd.MultiIndex) else raw[t].rename(t)
        closes[t] = s
    df = pd.concat(closes.values(), axis=1)
    df = df.dropna(how="all").sort_index()
    # Drop columns that are all-NaN (delisted/wrong ticker safety)
    df = df.dropna(axis=1, how="all")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.to_parquet(CACHE)
    logger.info(f"Wrote cache {CACHE}  shape={df.shape}  range={df.index.min().date()}..{df.index.max().date()}")
    return df


# ----------------------------- stats helpers ------------------------------ #

def engle_granger_pvalue(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    """Return (coint p-value via statsmodels.coint, OLS hedge ratio beta).

    Uses statsmodels.tsa.stattools.coint which performs the Engle-Granger
    two-step test internally (regress A on B, ADF on residual).  Also returns
    the OLS hedge ratio beta we'd use for trading.
    """
    a = a.dropna()
    b = b.dropna()
    ix = a.index.intersection(b.index)
    a = a.loc[ix]
    b = b.loc[ix]
    if len(ix) < 100:
        return float("nan"), float("nan")
    # Hedge ratio via OLS: A = alpha + beta * B + eps
    X = sm.add_constant(b.values)
    model = sm.OLS(a.values, X).fit()
    beta = float(model.params[1])
    try:
        _t, pval, _crit = coint(a.values, b.values)
    except Exception as e:
        logger.warning(f"coint failed for {a.name}/{b.name}: {e}")
        pval = float("nan")
    return float(pval), beta


def ou_half_life(spread: pd.Series) -> float:
    """OU half-life from AR(1) on the spread.

    Fit  spread_t - spread_{t-1} = a + b*(spread_{t-1}) + eps
    half-life = -ln(2) / ln(1 + b)   when -2 < b < 0, else NaN.
    """
    s = spread.dropna()
    if len(s) < 30:
        return float("nan")
    ds = s.diff().dropna()
    s_lag = s.shift(1).loc[ds.index]
    X = sm.add_constant(s_lag.values)
    res = sm.OLS(ds.values, X).fit()
    b = float(res.params[1])
    if b >= 0 or b <= -2:
        return float("nan")
    hl = -math.log(2) / math.log(1 + b)
    return float(hl)


# ----------------------------- backtest core ------------------------------ #

@dataclass
class Trade:
    pair: str
    leg_a: str
    leg_b: str
    direction: int             # +1 = long A short B (z<=-2), -1 = short A long B (z>=+2)
    entry_date: str
    exit_date: str
    days_held: int
    entry_z: float
    exit_z: float
    entry_spread: float
    exit_spread: float
    gross_return: float        # log-spread move * direction
    cost: float                # 4 bp = 0.0004
    net_return: float          # gross - cost
    reason: str                # 'mean_revert', 'stop_loss', 'time_stop', 'end_of_data'
    in_sample: bool            # True if entered before OOS_START
    half_life: float
    hedge_ratio: float


def rolling_coint_diagnostic(pa: pd.Series, pb: pd.Series,
                              window: int = 252, step: int = 21) -> dict:
    """Diagnostic only — rolling Engle-Granger pass rate by sub-window.

    The instruction asks for full-history coint; this is recorded alongside
    so we have an honest read on how a spec-compliant *rolling* implementation
    (which the candidate's own §4 calls for) would have looked.
    """
    n = len(pa)
    is_passes = 0; is_total = 0
    oos_passes = 0; oos_total = 0
    for end in range(window, n, step):
        sub_ix = pa.index[end-window:end]
        end_ts = sub_ix[-1]
        try:
            pv = coint(pa.loc[sub_ix].values, pb.loc[sub_ix].values)[1]
        except Exception:
            continue
        if end_ts < pd.Timestamp(OOS_START):
            is_total += 1
            if pv < COINT_P_MAX:
                is_passes += 1
        else:
            oos_total += 1
            if pv < COINT_P_MAX:
                oos_passes += 1
    return {
        "rolling_window_days": window,
        "rolling_step_days": step,
        "IS_pass": is_passes, "IS_windows": is_total,
        "IS_pass_pct": (is_passes / is_total) if is_total else None,
        "OOS_pass": oos_passes, "OOS_windows": oos_total,
        "OOS_pass_pct": (oos_passes / oos_total) if oos_total else None,
    }


def run_pair_backtest(
    pair_name: str,
    a_name: str,
    b_name: str,
    prices: pd.DataFrame,
) -> tuple[list[Trade], dict, pd.Series]:
    """Run backtest for one pair.  Returns (trades, diagnostics, daily_strategy_returns)."""
    diag: dict = {
        "pair": pair_name,
        "leg_a": a_name,
        "leg_b": b_name,
    }
    if a_name not in prices.columns or b_name not in prices.columns:
        logger.warning(f"{pair_name}: missing column(s); a={a_name in prices.columns} b={b_name in prices.columns}")
        diag["status"] = "missing_data"
        return [], diag, pd.Series(dtype=float)

    pa = np.log(prices[a_name].dropna())
    pb = np.log(prices[b_name].dropna())
    common = pa.index.intersection(pb.index)
    if len(common) < 300:
        diag["status"] = "insufficient_overlap"
        diag["overlap_days"] = int(len(common))
        return [], diag, pd.Series(dtype=float)
    pa = pa.loc[common]
    pb = pb.loc[common]
    diag["overlap_days"] = int(len(common))
    diag["start"] = str(common.min().date())
    diag["end"] = str(common.max().date())

    # 0) Rolling diagnostic (informational; does not gate trading)
    try:
        diag["rolling_coint"] = rolling_coint_diagnostic(pa, pb)
    except Exception as e:
        logger.warning(f"{pair_name}: rolling diagnostic failed: {e}")

    # In-sample p-value (informational)
    is_mask = pa.index <= pd.Timestamp(IS_END)
    if is_mask.sum() > 100:
        try:
            diag["coint_pvalue_in_sample"] = float(coint(pa[is_mask].values, pb[is_mask].values)[1])
        except Exception:
            diag["coint_pvalue_in_sample"] = float("nan")

    # 1) Cointegration test on full history (the gate per the instructions)
    pval, beta = engle_granger_pvalue(pa, pb)
    diag["coint_pvalue"] = pval
    diag["hedge_ratio"] = beta
    logger.info(f"{pair_name}: coint p_full={pval:.4f}  p_IS={diag.get('coint_pvalue_in_sample', float('nan')):.4f}  beta={beta:.4f}  N={len(common)}")
    if "rolling_coint" in diag:
        rc = diag["rolling_coint"]
        logger.info(
            f"  rolling 252d gate pass: IS={rc['IS_pass']}/{rc['IS_windows']} "
            f"({(rc['IS_pass_pct'] or 0):.1%})  OOS={rc['OOS_pass']}/{rc['OOS_windows']} "
            f"({(rc['OOS_pass_pct'] or 0):.1%})"
        )
    if not np.isfinite(pval) or pval > COINT_P_MAX:
        diag["status"] = f"coint_fail (p={pval:.4f})"
        return [], diag, pd.Series(dtype=float)

    # 2) Spread, rolling stats, z-score
    spread = pa - beta * pb
    roll_mean = spread.shift(1).rolling(ROLL_WIN, min_periods=ROLL_WIN).mean()
    roll_std = spread.shift(1).rolling(ROLL_WIN, min_periods=ROLL_WIN).std()
    z = (spread - roll_mean) / roll_std

    # 3) OU half-life on full-history spread
    hl = ou_half_life(spread)
    diag["half_life"] = hl
    logger.info(f"{pair_name}: half_life={hl:.2f} days")
    if not np.isfinite(hl) or hl < HALF_LIFE_MIN or hl > HALF_LIFE_MAX:
        diag["status"] = f"half_life_fail (H={hl})"
        return [], diag, pd.Series(dtype=float)

    diag["status"] = "tradeable"

    # 4) Walk the daily series, fire entry / exit / stop / time-stop rules
    trades: list[Trade] = []
    daily_ret = pd.Series(0.0, index=spread.index, name=pair_name)

    in_pos = False
    direction = 0
    entry_date = None
    entry_spread = float("nan")
    entry_z = float("nan")
    days_in = 0
    time_stop_days = int(round(3 * hl))

    spread_arr = spread.values
    z_arr = z.values
    dates = spread.index

    # We need both rolling stats and a sane z value to act.
    for i in range(len(dates)):
        zi = z_arr[i]
        if not np.isfinite(zi):
            continue

        date_i = dates[i]
        if in_pos:
            days_in += 1
            spread_i = spread_arr[i]
            # raw spread change since entry, signed by direction
            move = (spread_i - entry_spread) * direction

            reason = None
            if abs(zi) < Z_EXIT:
                reason = "mean_revert"
            elif abs(zi) > Z_STOP:
                reason = "stop_loss"
            elif days_in > time_stop_days:
                reason = "time_stop"
            elif i == len(dates) - 1:
                reason = "end_of_data"

            if reason is not None:
                cost = (COST_BP_PER_LEG * 2) / 10_000.0  # round trip both legs -> 4 bp
                net = move - cost
                trades.append(
                    Trade(
                        pair=pair_name,
                        leg_a=a_name,
                        leg_b=b_name,
                        direction=direction,
                        entry_date=str(entry_date.date()),
                        exit_date=str(date_i.date()),
                        days_held=days_in,
                        entry_z=float(entry_z),
                        exit_z=float(zi),
                        entry_spread=float(entry_spread),
                        exit_spread=float(spread_i),
                        gross_return=float(move),
                        cost=float(cost),
                        net_return=float(net),
                        reason=reason,
                        in_sample=entry_date < pd.Timestamp(OOS_START),
                        half_life=float(hl),
                        hedge_ratio=float(beta),
                    )
                )
                # Allocate the realised PnL on the exit date for daily-return series.
                daily_ret.iloc[i] += net
                in_pos = False
                direction = 0
                entry_date = None
                days_in = 0
                continue

        # Look for entry only when flat
        if not in_pos and abs(zi) >= Z_ENTRY:
            # z >= +2  -> spread is rich -> sell spread (short A, long B) -> direction = -1
            # z <= -2  -> spread is cheap -> buy spread (long A, short B) -> direction = +1
            direction = -1 if zi > 0 else 1
            entry_date = date_i
            entry_spread = spread_arr[i]
            entry_z = zi
            in_pos = True
            days_in = 0

    diag["n_trades"] = len(trades)
    return trades, diag, daily_ret


# ----------------------------- summary ------------------------------------ #

def summarize(trades: list[Trade], daily_ret_total: pd.Series,
              diagnostics: list[dict]) -> dict:
    df_all = pd.DataFrame([asdict(t) for t in trades])
    oos = df_all[~df_all["in_sample"]] if len(df_all) else df_all
    pairs_with_coint = sum(1 for d in diagnostics if d.get("status") in ("tradeable",) or
                           (isinstance(d.get("status"), str) and d["status"].startswith("half_life_fail")))
    pairs_tradeable = sum(1 for d in diagnostics if d.get("status") == "tradeable")

    if len(oos):
        wins = (oos["net_return"] > 0).sum()
        wr = float(wins / len(oos))
        mean_bps = float(oos["net_return"].mean() * 10_000.0)
        total_trades_oos = int(len(oos))
    else:
        wr = float("nan")
        mean_bps = float("nan")
        total_trades_oos = 0

    # Daily-return-based metrics restricted to OOS window
    dr_oos = daily_ret_total.loc[OOS_START:]
    if len(dr_oos) > 30 and dr_oos.std() > 0:
        # Sharpe based on daily strategy return; trades realise PnL on exit day,
        # so this is event-driven (sparse) daily series.
        mu = dr_oos.mean()
        sd = dr_oos.std(ddof=1)
        sharpe = float((mu / sd) * math.sqrt(TRADING_DAYS_PER_YEAR))
        ann_return_pct = float(mu * TRADING_DAYS_PER_YEAR * 100.0)
        equity = (1.0 + dr_oos).cumprod()
        peak = equity.cummax()
        dd = (equity / peak - 1.0)
        max_dd_pct = float(dd.min() * 100.0)
    else:
        sharpe = float("nan")
        ann_return_pct = float("nan")
        max_dd_pct = float("nan")

    verdict = "PASS" if (np.isfinite(sharpe) and sharpe >= 0.8) else "FAIL"

    summary = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "in_sample_window": [START, IS_END],
        "out_of_sample_window": [OOS_START, str(daily_ret_total.index.max().date()) if len(daily_ret_total) else None],
        "pairs_tested": len(PAIRS),
        "pairs_with_cointegration": int(sum(1 for d in diagnostics if np.isfinite(d.get("coint_pvalue", float("nan"))) and d.get("coint_pvalue", 1.0) <= COINT_P_MAX)),
        "pairs_tradeable": pairs_tradeable,
        "total_trades_oos": total_trades_oos,
        "win_rate": wr,
        "mean_return_per_trade_bps": mean_bps,
        "ann_return_pct": ann_return_pct,
        "sharpe_oos": sharpe,
        "max_dd_pct": max_dd_pct,
        "verdict": verdict,
        "per_pair_diagnostics": diagnostics,
        "cost_per_round_trip_bp": COST_BP_PER_LEG * 2,
    }
    return summary


# ----------------------------- entry point -------------------------------- #

def main() -> int:
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("iter_8 backtest: commodity ETF pair cointegration")
    logger.info(f"pairs={PAIRS}  start={START}  IS_end={IS_END}  OOS_start={OOS_START}")
    logger.info(f"Z_entry={Z_ENTRY}  Z_exit={Z_EXIT}  Z_stop={Z_STOP}  roll={ROLL_WIN}")
    logger.info(f"HL band=[{HALF_LIFE_MIN},{HALF_LIFE_MAX}]  cost_per_leg_bp={COST_BP_PER_LEG}")
    logger.info("=" * 70)

    prices = load_prices()
    logger.info(f"Price panel: shape={prices.shape}  cols={list(prices.columns)}")

    all_trades: list[Trade] = []
    diagnostics: list[dict] = []
    daily_total = pd.Series(0.0, index=prices.index)

    for a, b in PAIRS:
        pair_name = f"{a}/{b}"
        logger.info("-" * 70)
        logger.info(f"Pair {pair_name}")
        trades, diag, dr = run_pair_backtest(pair_name, a, b, prices)
        diagnostics.append(diag)
        all_trades.extend(trades)
        if len(dr):
            # align to total daily index
            daily_total = daily_total.add(dr.reindex(daily_total.index).fillna(0.0), fill_value=0.0)
        logger.info(f"  status={diag.get('status')}  trades={len(trades)}")

    # Save trades CSV
    if all_trades:
        tdf = pd.DataFrame([asdict(t) for t in all_trades])
        tdf = tdf.sort_values(["entry_date", "pair"]).reset_index(drop=True)
    else:
        tdf = pd.DataFrame(columns=[f.name for f in Trade.__dataclass_fields__.values()])
    tdf.to_csv(TRADES_CSV, index=False)
    logger.info(f"Wrote {TRADES_CSV}  rows={len(tdf)}")

    summary = summarize(all_trades, daily_total, diagnostics)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=lambda o: None if isinstance(o, float) and not np.isfinite(o) else o))
    logger.info(f"Wrote {SUMMARY_JSON}")

    # Per-pair OOS attribution log
    if all_trades:
        df = pd.DataFrame([asdict(t) for t in all_trades])
        oos = df[~df["in_sample"]]
        for pair_name in oos["pair"].unique():
            sub = oos[oos["pair"] == pair_name]
            wr = (sub["net_return"] > 0).mean() if len(sub) else float("nan")
            mu_bp = sub["net_return"].mean() * 10_000.0 if len(sub) else float("nan")
            logger.info(f"  OOS {pair_name}: n={len(sub)}  win_rate={wr:.2%}  mean_bps={mu_bp:.1f}")

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info(f"DONE in {elapsed:.1f}s")
    logger.info(
        f"VERDICT={summary['verdict']}  pairs_tradeable={summary['pairs_tradeable']}  "
        f"OOS_trades={summary['total_trades_oos']}  Sharpe={summary['sharpe_oos']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
