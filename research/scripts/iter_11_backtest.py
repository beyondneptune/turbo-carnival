"""iter_11 backtest: LLM FOMC hawkish/dovish classifier — ORACLE-PROXY ceiling.

Spec: /Users/weeoto/turbo-carnival/research/candidates/iter_11_llm_fomc_hawkdove_classifier.md

This run does NOT call an LLM. Instead it simulates a *perfect-oracle* classifier
by labeling each FOMC event by the realized next-day SPY return:
  SPY +1d > +0.5%  -> "dovish"  (markets rallied -> surprise dovish)
  SPY +1d < -0.5%  -> "hawkish" (markets sold off -> surprise hawkish)
  otherwise        -> "neutral" (abstain)

That is a 1-day-lookahead oracle. The Sharpe it produces is the CEILING that
any LLM hawk/dove classifier on this asset/horizon could achieve. We then
estimate the realistic 70%-accuracy LLM by mixing 70% oracle signal + 30%
random noise on the same events, to see whether a non-perfect LLM still clears
a Sharpe-0.5 hurdle.

Events
------
FOMC statement-release dates 2018-01 through 2025-05 (Fed schedule, 8/year),
plus the corresponding minutes-release dates ~21 calendar days after each
statement. ~118 events total over 7+ years.

Strategy
--------
Position entered at the close of (event_date + 1 business day) — i.e. after the
1-day return that triggered the classification — and held for 3 trading days.
  dovish  -> +100% NAV long SPY
  hawkish -> +100% NAV long TLT (the safe-haven proxy when Fed is tightening)
  neutral -> flat

Cost: 5 bps round-trip per trade.

Windows
-------
Train (skip / sanity-only): 2018-01 .. 2019-12
OOS (verdict):              2020-01 .. 2025-05

Outputs
-------
Log:      /Users/weeoto/turbo-carnival/research/logs/iter_11_backtest.log
Trades:   /Users/weeoto/turbo-carnival/research/results/iter_11_trades.csv
Summary:  /Users/weeoto/turbo-carnival/research/results/iter_11_summary.json
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path("/Users/weeoto/turbo-carnival/research")
CACHE_PATH = ROOT / "data" / "cache" / "spy_tlt_daily.parquet"
LOG_PATH = ROOT / "logs" / "iter_11_backtest.log"
TRADES_CSV = ROOT / "results" / "iter_11_trades.csv"
SUMMARY_JSON = ROOT / "results" / "iter_11_summary.json"

TICKERS = ["SPY", "TLT"]
START = "2017-09-01"  # buffer before first 2018 event for any rolling stats
END_HINT = "2025-06-30"

# Classification thresholds applied to SPY 1-day forward return
DOVISH_THRESH = 0.005   # > +0.5%
HAWKISH_THRESH = -0.005  # < -0.5%

# Strategy parameters
HOLD_DAYS = 3
COST_BPS_ROUNDTRIP = 5.0  # 5 bp per trade round-trip

# Trading-day windows
TRAIN_END = pd.Timestamp("2019-12-31")
OOS_START = pd.Timestamp("2020-01-01")
OOS_END = pd.Timestamp("2025-05-31")

# Sensitivity: hypothetical LLM accuracy on oracle classifications
LLM_ACCURACY = 0.70

# Deterministic RNG for the 70%-accuracy simulation
RNG_SEED = 20260512


# ---------------------------------------------------------------------------
# FOMC schedule — hardcoded statement-release dates (FOMC official calendar).
# Dates are the press-statement release day (statement printed 14:00 ET).
# Source: federalreserve.gov FOMC historical materials.
# ---------------------------------------------------------------------------
FOMC_STATEMENTS = [
    # 2018
    "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13",
    "2018-08-01", "2018-09-26", "2018-11-08", "2018-12-19",
    # 2019
    "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
    "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
    # 2020 (March: emergency cuts on 03-03 and 03-15; regular meeting moved/canceled)
    "2020-01-29", "2020-03-03", "2020-03-15", "2020-04-29",
    "2020-06-10", "2020-07-29", "2020-09-16", "2020-11-05",
    "2020-12-16",
    # 2021
    "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
    "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
    # 2022
    "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
    "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
    # 2023
    "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
    "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
    # 2024
    "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
    "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
    # 2025 (through May 2025)
    "2025-01-29", "2025-03-19", "2025-05-07",
]


def setup_logging() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    fmt = "%(asctime)s %(levelname)s %(message)s"
    # Reset handlers so multiple runs don't double-log.
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_PATH, mode="w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger("iter_11")


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def fetch_prices(log: logging.Logger) -> pd.DataFrame:
    if CACHE_PATH.exists():
        df = pd.read_parquet(CACHE_PATH)
        last = df.index.max()
        if last >= pd.Timestamp(OOS_END):
            log.info(f"Loaded cache {CACHE_PATH} rows={len(df)} last={last.date()}")
            return df
        log.info(f"Cache only to {last.date()}; refetching")

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    log.info(f"Downloading yfinance: {TICKERS} {START}..{END_HINT}")
    raw = yf.download(
        TICKERS,
        start=START,
        end=END_HINT,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].copy()
        close.columns = TICKERS

    close = close.dropna(how="any")
    close.columns = [c.upper() for c in close.columns]
    close = close[TICKERS]
    close.index = pd.to_datetime(close.index)
    close.to_parquet(CACHE_PATH)
    log.info(
        f"Saved cache {CACHE_PATH} rows={len(close)} "
        f"{close.index.min().date()}..{close.index.max().date()}"
    )
    return close


# ---------------------------------------------------------------------------
# Event builder
# ---------------------------------------------------------------------------
def build_events(prices: pd.DataFrame, log: logging.Logger) -> pd.DataFrame:
    """Build the full event table including minutes (statement + 21 calendar days)."""
    rows = []
    for d in FOMC_STATEMENTS:
        ts = pd.Timestamp(d)
        rows.append({"event_date": ts, "kind": "statement"})
        # Minutes are released roughly 3 weeks later. 21 calendar days is the
        # canonical lag the spec asked for. We then snap to the next trading day.
        rows.append({"event_date": ts + pd.Timedelta(days=21), "kind": "minutes"})
    ev = pd.DataFrame(rows).sort_values("event_date").reset_index(drop=True)

    # Snap each scheduled date to the next available trading day in the price index.
    trade_dates = prices.index
    snapped = []
    for d in ev["event_date"]:
        # searchsorted: first index >= d
        idx = trade_dates.searchsorted(d, side="left")
        if idx >= len(trade_dates):
            snapped.append(pd.NaT)
        else:
            snapped.append(trade_dates[idx])
    ev["trading_date"] = snapped
    ev = ev.dropna(subset=["trading_date"]).copy()
    ev["trading_date"] = pd.to_datetime(ev["trading_date"])

    # Drop minutes events that fall after the OOS_END buffer
    ev = ev[ev["trading_date"] <= OOS_END + pd.Timedelta(days=10)].copy()

    log.info(
        f"Built {len(ev)} events  "
        f"({(ev['kind']=='statement').sum()} statements, "
        f"{(ev['kind']=='minutes').sum()} minutes)"
    )
    return ev


# ---------------------------------------------------------------------------
# Classifier + backtest
# ---------------------------------------------------------------------------
def classify_event(spy_1d_ret: float) -> str:
    """Perfect-oracle classifier — uses next-day realized SPY return."""
    if spy_1d_ret > DOVISH_THRESH:
        return "dovish"
    if spy_1d_ret < HAWKISH_THRESH:
        return "hawkish"
    return "neutral"


def simulate_strategy(
    events: pd.DataFrame,
    prices: pd.DataFrame,
    log: logging.Logger,
    label_override: pd.Series | None = None,
) -> pd.DataFrame:
    """Walk through events, run oracle classification, take positions.

    Per the brief: enter at close of event_date + 1 business day (the same close
    whose 1-day return drives the classification), hold 3 trading days.

    Returns a trade-per-event DataFrame (one row per traded or skipped event).
    """
    trade_dates = prices.index
    spy = prices["SPY"]
    tlt = prices["TLT"]
    rng = np.random.default_rng(RNG_SEED)

    rows = []
    for i, ev in events.iterrows():
        d0 = ev["trading_date"]
        try:
            i0 = trade_dates.get_loc(d0)
        except KeyError:
            continue
        # Need i0+1 (label/entry day) and i0 + 1 + HOLD_DAYS (exit day)
        i_entry = i0 + 1
        i_exit = i_entry + HOLD_DAYS
        if i_exit >= len(trade_dates):
            log.info(f"  skip {d0.date()} ({ev['kind']}) — runs past end of data")
            continue

        d_entry = trade_dates[i_entry]
        d_exit = trade_dates[i_exit]

        spy_event_close = float(spy.iloc[i0])
        spy_entry = float(spy.iloc[i_entry])
        spy_exit = float(spy.iloc[i_exit])
        tlt_entry = float(tlt.iloc[i_entry])
        tlt_exit = float(tlt.iloc[i_exit])

        spy_1d_ret = (spy_entry - spy_event_close) / spy_event_close
        label = classify_event(spy_1d_ret)

        # Optional label override (used for the 70%-accuracy hypothetical)
        if label_override is not None:
            label = label_override.loc[i]

        if label == "dovish":
            # long SPY for HOLD_DAYS, sized 100% NAV
            gross_ret = (spy_exit / spy_entry) - 1.0
            asset = "SPY"
            traded = True
        elif label == "hawkish":
            gross_ret = (tlt_exit / tlt_entry) - 1.0
            asset = "TLT"
            traded = True
        else:
            gross_ret = 0.0
            asset = None
            traded = False

        net_ret = gross_ret - (COST_BPS_ROUNDTRIP / 10_000.0) if traded else 0.0

        period = "train" if d_entry <= TRAIN_END else (
            "oos" if (d_entry >= OOS_START and d_entry <= OOS_END + pd.Timedelta(days=14))
            else "buffer"
        )

        rows.append({
            "event_date": d0,
            "kind": ev["kind"],
            "entry_date": d_entry,
            "exit_date": d_exit,
            "spy_event_close": spy_event_close,
            "spy_entry_close": spy_entry,
            "spy_exit_close": spy_exit,
            "tlt_entry_close": tlt_entry,
            "tlt_exit_close": tlt_exit,
            "spy_1d_ret": spy_1d_ret,
            "label": label,
            "traded": traded,
            "asset": asset,
            "gross_ret": gross_ret,
            "net_ret": net_ret,
            "period": period,
        })

    df = pd.DataFrame(rows)
    log.info(
        f"Simulated {len(df)} events  "
        f"traded={int(df['traded'].sum())}  "
        f"abstain={int((~df['traded']).sum())}"
    )
    return df


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _ann_factor_event(events_per_year: float) -> float:
    return np.sqrt(events_per_year)


def compute_metrics(trades: pd.DataFrame, log: logging.Logger, label: str) -> dict:
    """Trades is the per-event frame; compute on OOS traded rows."""
    oos = trades[trades["period"] == "oos"].copy()
    traded = oos[oos["traded"]].copy()
    n_events_oos = len(oos)
    n_trades = len(traded)

    if n_trades == 0:
        return {
            "tag": label,
            "n_events_oos": int(n_events_oos),
            "n_trades_oos": 0,
            "win_rate_pct": None,
            "mean_return_per_trade_pct": None,
            "ann_return_pct": None,
            "sharpe_oos": None,
            "max_dd_pct": None,
        }

    rets = traded["net_ret"].values
    win_rate = float((rets > 0).mean() * 100.0)
    mean_ret = float(rets.mean())
    std_ret = float(rets.std(ddof=1)) if len(rets) > 1 else 0.0

    # OOS span in years (use entry-date span)
    span_days = (traded["entry_date"].max() - traded["entry_date"].min()).days
    span_years = max(span_days / 365.25, 1e-6)
    events_per_year = n_trades / span_years

    ann_return = mean_ret * events_per_year  # arithmetic annualization on per-event mean
    sharpe = (mean_ret / std_ret) * _ann_factor_event(events_per_year) if std_ret > 0 else float("nan")

    # Max DD on per-event compounded equity curve, in % of starting equity.
    eq = (1.0 + pd.Series(rets, index=traded["entry_date"].values)).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    max_dd_pct = float(-dd.min() * 100.0) if len(eq) else 0.0

    metrics = {
        "tag": label,
        "n_events_oos": int(n_events_oos),
        "n_trades_oos": int(n_trades),
        "abstain_rate_pct": round(float((1 - n_trades / n_events_oos) * 100.0), 2),
        "win_rate_pct": round(win_rate, 2),
        "mean_return_per_trade_pct": round(mean_ret * 100.0, 4),
        "ann_return_pct": round(ann_return * 100.0, 3),
        "sharpe_oos": round(float(sharpe), 3) if not np.isnan(sharpe) else None,
        "max_dd_pct": round(max_dd_pct, 3),
        "events_per_year": round(events_per_year, 2),
        "oos_span_years": round(span_years, 3),
    }
    log.info(
        f"[{label}] OOS: events={n_events_oos} trades={n_trades} "
        f"win={metrics['win_rate_pct']}% mean/trade={metrics['mean_return_per_trade_pct']}% "
        f"ann={metrics['ann_return_pct']}% sharpe={metrics['sharpe_oos']} "
        f"maxDD={metrics['max_dd_pct']}%"
    )
    return metrics


def make_70pct_labels(oracle_trades: pd.DataFrame, log: logging.Logger) -> pd.Series:
    """For each oracle-labeled row, with prob 0.70 keep the oracle label, with
    prob 0.30 emit a uniform-random label from {dovish, hawkish, neutral}.

    This is intentionally a coarse model — a real LLM's error structure is not
    uniform. But it gives a reasonable expectation: if the strategy still works
    after 30% label noise, the edge is robust to LLM imperfection.
    """
    rng = np.random.default_rng(RNG_SEED)
    n = len(oracle_trades)
    out = oracle_trades["label"].copy().reset_index(drop=True)
    flips = rng.random(n) >= LLM_ACCURACY
    random_choices = rng.choice(["dovish", "hawkish", "neutral"], size=n)
    out = out.where(~flips, pd.Series(random_choices))
    out.index = oracle_trades.index
    n_flipped = int(flips.sum())
    log.info(
        f"[70%-acc sim] flipped {n_flipped}/{n} labels ({n_flipped/n*100:.1f}% noise)"
    )
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    log = setup_logging()
    log.info("=== iter_11 backtest start ===")
    log.info(f"params: HOLD_DAYS={HOLD_DAYS} COST_BPS={COST_BPS_ROUNDTRIP} "
             f"OOS={OOS_START.date()}..{OOS_END.date()}")

    prices = fetch_prices(log)
    events = build_events(prices, log)

    # --- Oracle run ---
    oracle_df = simulate_strategy(events, prices, log)

    # --- 70%-accuracy hypothetical: rerun with mutated labels ---
    noisy_labels = make_70pct_labels(oracle_df, log)
    sim70_df = simulate_strategy(events, prices, log, label_override=noisy_labels)

    # Write the oracle trade frame as the canonical record
    TRADES_CSV.parent.mkdir(parents=True, exist_ok=True)
    oracle_df.to_csv(TRADES_CSV, index=False)
    log.info(f"Wrote {TRADES_CSV}")

    oracle_metrics = compute_metrics(oracle_df, log, "oracle")
    sim70_metrics = compute_metrics(sim70_df, log, "sim_70pct_accuracy")

    oracle_sharpe = oracle_metrics.get("sharpe_oos") or 0.0
    sim70_sharpe = sim70_metrics.get("sharpe_oos") or 0.0
    oracle_ann = oracle_metrics.get("ann_return_pct") or 0.0
    sim70_ann = sim70_metrics.get("ann_return_pct") or 0.0

    oracle_advantage_pp = round(float(oracle_ann - sim70_ann), 3)

    pass_gate = (oracle_sharpe >= 1.0) and (sim70_sharpe >= 0.5)
    verdict = "PASS" if pass_gate else "FAIL"
    verdict_reason = (
        f"oracle_sharpe={oracle_sharpe:.3f} (req>=1.0), "
        f"sim70_sharpe={sim70_sharpe:.3f} (req>=0.5)"
    )

    summary = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "spec": "iter_11_llm_fomc_hawkdove_classifier",
        "approach": "perfect_oracle_proxy + 70%_accuracy_simulation",
        "params": {
            "tickers": TICKERS,
            "fomc_statement_count": len(FOMC_STATEMENTS),
            "minutes_lag_days": 21,
            "dovish_thresh_1d_ret": DOVISH_THRESH,
            "hawkish_thresh_1d_ret": HAWKISH_THRESH,
            "hold_days": HOLD_DAYS,
            "cost_bps_roundtrip": COST_BPS_ROUNDTRIP,
            "llm_accuracy_sim": LLM_ACCURACY,
            "train_end": str(TRAIN_END.date()),
            "oos_start": str(OOS_START.date()),
            "oos_end": str(OOS_END.date()),
        },
        # Top-level fields required by the brief
        "events_oos": oracle_metrics["n_events_oos"],
        "trades_taken": oracle_metrics["n_trades_oos"],
        "win_rate": oracle_metrics["win_rate_pct"],
        "mean_return_per_trade_pct": oracle_metrics["mean_return_per_trade_pct"],
        "ann_return_pct": oracle_metrics["ann_return_pct"],
        "sharpe_oos": oracle_metrics["sharpe_oos"],
        "max_dd_pct": oracle_metrics["max_dd_pct"],
        "oracle_advantage_pp": oracle_advantage_pp,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        # Full per-tag detail
        "oracle_metrics": oracle_metrics,
        "sim_70pct_metrics": sim70_metrics,
    }

    SUMMARY_JSON.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=str))
    log.info(f"Wrote {SUMMARY_JSON}")
    log.info(
        f"RESULT_LINE: iter_11 (oracle proxy) | events={summary['events_oos']} "
        f"| oracle_sharpe={oracle_sharpe:.3f} "
        f"| sim70_sharpe={sim70_sharpe:.3f} | {verdict}"
    )
    log.info("=== iter_11 backtest done ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
