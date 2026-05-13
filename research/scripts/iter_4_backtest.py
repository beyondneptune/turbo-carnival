#!/usr/bin/env python3
"""
iter_4 backtest: G10 FX carry, monthly rebalance, vol-scaled.

Approach:
  - Universe: 9 non-USD G10 currencies (EUR, JPY, GBP, AUD, NZD, CAD, CHF, SEK, NOK)
    Traded against USD; all returns computed in USD terms (base currency = USD).
  - Carry signal:  carry_c = i_c - i_USD   (3-month interbank from FRED)
  - Monthly: rank by carry, long top 3, short bottom 3 (against USD numeraire).
  - Vol scaling: each leg sized so 30d realized vol -> 3.33% / month  (10% ann / 3 legs).
  - Vol filter: per-pair 30d realized vol (annualized);
        > 25%  -> weight 0
        > 15%  -> weight halved
  - Monthly rebalance on first trading day of each month.
  - Cost: 1.5 bps round-trip per leg, applied at rebalance.
  - Walk-forward: 2010-2018 IS (stats only); 2019- to today OOS.
  - PASS if OOS Sharpe >= 0.55.
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
import urllib.request
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ----- paths -----
ROOT = Path('/Users/weeoto/turbo-carnival/research')
CACHE_DIR = ROOT / 'data' / 'cache'
LOG_DIR = ROOT / 'logs'
RESULTS_DIR = ROOT / 'results'
SCRIPT_PATH = ROOT / 'scripts' / 'iter_4_backtest.py'

CACHE_FX = CACHE_DIR / 'g10_fx_daily.parquet'
CACHE_RATES = CACHE_DIR / 'g10_rates_monthly.parquet'
LOG_PATH = LOG_DIR / 'iter_4_backtest.log'
MONTHLY_CSV = RESULTS_DIR / 'iter_4_monthly_returns.csv'
SUMMARY_JSON = RESULTS_DIR / 'iter_4_summary.json'

for d in (CACHE_DIR, LOG_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ----- logging -----
logger = logging.getLogger('iter_4')
logger.setLevel(logging.INFO)
logger.handlers.clear()
fh = logging.FileHandler(LOG_PATH, mode='w')
fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
logger.addHandler(fh)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
logger.addHandler(sh)

def log(msg: str) -> None:
    logger.info(msg)


# =====================================================================
# Data fetch
# =====================================================================

# G10 currencies (excluding USD which is base).
# Each entry: ticker on yfinance, and whether the ticker is "CCY/USD" (long=long that ccy)
# yfinance pair conventions:
#   EURUSD=X -> price of 1 EUR in USD            (USD-quoted; long pair = long EUR)
#   GBPUSD=X -> price of 1 GBP in USD            (long pair = long GBP)
#   AUDUSD=X -> price of 1 AUD in USD            (long pair = long AUD)
#   NZDUSD=X -> price of 1 NZD in USD            (long pair = long NZD)
#   USDJPY=X -> price of 1 USD in JPY            (long pair = short JPY) -> we want JPY rate, so we invert
#   USDCAD=X -> price of 1 USD in CAD            (long pair = short CAD) -> invert
#   USDCHF=X -> price of 1 USD in CHF            (long pair = short CHF) -> invert
#   USDSEK=X -> price of 1 USD in SEK            (long pair = short SEK) -> invert
#   USDNOK=X -> price of 1 USD in NOK            (long pair = short NOK) -> invert
#
# We always trade "long currency c vs USD" via the synthetic CCY/USD price (inverting where needed).
# So "going long EUR vs USD" earns +d(EURUSD)/EURUSD and pays (i_USD) and receives (i_EUR).
# For symmetry we form CCY_USD = price of 1 unit of currency c in USD for all 9 currencies.

CCYS = ['EUR', 'JPY', 'GBP', 'AUD', 'NZD', 'CAD', 'CHF', 'SEK', 'NOK']

YF_TICKER = {
    'EUR': ('EURUSD=X', False),
    'JPY': ('USDJPY=X', True),
    'GBP': ('GBPUSD=X', False),
    'AUD': ('AUDUSD=X', False),
    'NZD': ('NZDUSD=X', False),
    'CAD': ('USDCAD=X', True),
    'CHF': ('USDCHF=X', True),
    'SEK': ('USDSEK=X', True),
    'NOK': ('USDNOK=X', True),
}

# 3-month interbank rates from FRED.
FRED_RATE = {
    'USD': 'IR3TIB01USM156N',
    'EUR': 'IR3TIB01EZM156N',
    'JPY': 'IR3TIB01JPM156N',
    'GBP': 'IR3TIB01GBM156N',
    'AUD': 'IR3TIB01AUM156N',
    'NZD': 'IR3TIB01NZM156N',
    'CAD': 'IR3TIB01CAM156N',
    'CHF': 'IR3TIB01CHM156N',
    'SEK': 'IR3TIB01SEM156N',
    'NOK': 'IR3TIB01NOM156N',
}

START_DATE = '2009-06-01'   # warm-up
END_DATE = datetime.utcnow().strftime('%Y-%m-%d')


def fetch_fred_csv(series: str, retries: int = 4, sleep: float = 2.0) -> pd.Series:
    url = f'https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}'
    last_exc = None
    for k in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                buf = r.read()
            df = pd.read_csv(io.BytesIO(buf))
            df.columns = ['date', 'value']
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df = df.dropna().set_index('date').sort_index()
            return df['value'].rename(series)
        except Exception as e:
            last_exc = e
            log(f'  FRED {series} attempt {k+1}/{retries} failed: {e}')
            time.sleep(sleep * (k + 1))
    raise RuntimeError(f'FRED fetch failed for {series}: {last_exc}')


def load_or_fetch_rates() -> pd.DataFrame:
    if CACHE_RATES.exists():
        log(f'Loading cached rates from {CACHE_RATES}')
        return pd.read_parquet(CACHE_RATES)
    log('Fetching FRED 3m interbank rates...')
    out = {}
    for ccy, series in FRED_RATE.items():
        log(f'  fetch {ccy} = {series}')
        out[ccy] = fetch_fred_csv(series)
        time.sleep(0.6)
    df = pd.DataFrame(out)
    df = df.sort_index()
    df.index.name = 'date'
    df.to_parquet(CACHE_RATES)
    log(f'Cached rates -> {CACHE_RATES}  shape={df.shape}  range={df.index.min().date()}..{df.index.max().date()}')
    return df


def load_or_fetch_fx() -> pd.DataFrame:
    if CACHE_FX.exists():
        log(f'Loading cached FX from {CACHE_FX}')
        return pd.read_parquet(CACHE_FX)
    import yfinance as yf
    log('Fetching FX daily from yfinance...')
    frames = {}
    for ccy, (sym, inverted) in YF_TICKER.items():
        log(f'  fetch {ccy} = {sym} (inverted={inverted})')
        df = yf.download(sym, start=START_DATE, end=END_DATE, progress=False, auto_adjust=True)
        if df.empty:
            raise RuntimeError(f'empty download for {sym}')
        # Handle MultiIndex columns from new yfinance.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        close = df['Close'].copy()
        if inverted:
            close = 1.0 / close   # convert USD/CCY -> CCY/USD
        frames[ccy] = close
        time.sleep(0.3)
    out = pd.concat(frames, axis=1)
    out.columns = list(frames.keys())
    out = out.dropna(how='all').sort_index()
    out.index.name = 'date'
    # Convert to tz-naive
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.to_parquet(CACHE_FX)
    log(f'Cached FX -> {CACHE_FX}  shape={out.shape}  range={out.index.min().date()}..{out.index.max().date()}')
    return out


# =====================================================================
# Backtest engine
# =====================================================================

# Strategy params (per spec)
TARGET_VOL_ANN = 0.10            # 10% annualized portfolio vol target
TARGET_LEG_VOL_MONTHLY = 0.0333  # 3.33% / month per leg (10% ann / 3 long legs)
VOL_LOOKBACK_DAYS = 30           # 30-day realized vol window
VOL_FILTER_HALF = 0.15           # 15% ann realized vol -> weight halved
VOL_FILTER_ZERO = 0.25           # 25% ann realized vol -> weight zero
LONG_N = 3
SHORT_N = 3
COST_BPS_PER_LEG = 1.5           # round-trip bps per leg per rebalance
TRADING_DAYS = 252


def realized_vol(rets: pd.Series, window: int) -> pd.Series:
    """Annualized rolling realized vol (sqrt(252) on daily log returns)."""
    return rets.rolling(window).std() * np.sqrt(TRADING_DAYS)


def compute_backtest(fx: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily P&L of the carry portfolio.
    Returns a DataFrame with daily and monthly portfolio returns + diagnostics.
    """
    # Restrict FX to from 2009-06 onwards (warmup) and forward-fill weekends/holidays only across business days.
    fx = fx.copy().sort_index()
    fx = fx.dropna(how='all')
    # forward-fill (FX yfinance sometimes has small gaps)
    fx = fx.ffill()

    # Daily log returns for each pair (price of CCY in USD -> long CCY/USD = +ret)
    log_rets = np.log(fx).diff()

    # Bring rates to month-end aligned (rates are monthly already, indexed at month start).
    # We want, on each rebalance date, the most recently published rate for each ccy.
    # FRED publishes monthly rates at start of month with a 1-2 week lag in practice;
    # to avoid lookahead, we shift rates forward by 1 month so a rate published in Jan
    # is used for the Feb-1 rebalance.
    rates_shifted = rates.shift(1, freq='MS')  # MS = month start
    rates_shifted.index = rates_shifted.index.normalize()

    # Build monthly rebalance schedule: first business day of each month within FX range.
    fx_idx = fx.index
    bday_index = pd.bdate_range(fx_idx.min(), fx_idx.max())
    bday_set = set(fx_idx)
    # Use FX trading days only.
    fx_bdays = fx.index
    # Build month start business days = first FX bday in each (year, month).
    rebal_dates = (
        pd.Series(fx_bdays).groupby([fx_bdays.year, fx_bdays.month]).min().tolist()
    )
    rebal_dates = pd.DatetimeIndex(sorted(set(rebal_dates)))

    log(f'FX trading days {fx_bdays.min().date()}..{fx_bdays.max().date()}  n={len(fx_bdays)}')
    log(f'Rebalance dates: {len(rebal_dates)}  first={rebal_dates[0].date()}  last={rebal_dates[-1].date()}')

    # For each rebalance, compute:
    #   - rate per ccy as of latest available month-start <= rebalance date (with shift to avoid lookahead)
    #   - 30d realized vol per pair
    #   - carry vs USD
    #   - rank, choose long top 3 / short bottom 3
    #   - vol-scale each leg
    #   - apply vol filter
    # Then carry weights forward until next rebal.

    # Build helper: as-of rate map for a given rebal date.
    rates_shifted = rates_shifted.dropna(how='all').sort_index()
    rates_idx = rates_shifted.index

    def rates_asof(d: pd.Timestamp) -> pd.Series:
        # most recent shifted-rate row <= d
        i = rates_idx.searchsorted(d, side='right') - 1
        if i < 0:
            return pd.Series(dtype=float)
        return rates_shifted.iloc[i]

    # Pre-compute per-pair realized vol series
    vol = {}
    for c in CCYS:
        vol[c] = realized_vol(log_rets[c], VOL_LOOKBACK_DAYS)
    vol_df = pd.DataFrame(vol)

    # Build weight DataFrame indexed by day; weights change only on rebalance days.
    weights = pd.DataFrame(0.0, index=fx_bdays, columns=CCYS)
    chosen_history = []   # for diagnostics

    last_w = pd.Series(0.0, index=CCYS)

    for d in fx_bdays:
        if d in set(rebal_dates):
            r = rates_asof(d)
            v = vol_df.loc[:d].iloc[-1] if d in vol_df.index else vol_df.loc[:d].iloc[-1]
            # Need full data
            if r.empty or r.isna().any() or v.isna().any():
                # If anything missing, keep previous weights
                weights.loc[d] = last_w.values
                continue

            # carry_c = i_c - i_USD  (in pct units; e.g. 4.5)
            i_usd = r['USD']
            carry = pd.Series({c: r[c] - i_usd for c in CCYS})

            # rank ascending: top 3 = highest carry; bottom 3 = lowest
            ranked = carry.sort_values()
            shorts = ranked.index[:SHORT_N].tolist()
            longs = ranked.index[-LONG_N:].tolist()

            # Size each leg so 30d realized vol equals TARGET_LEG_VOL_MONTHLY (per month).
            # daily vol = annualized vol / sqrt(252); monthly vol = ann / sqrt(12).
            # target_leg_monthly = 3.33% means target_leg_annual = 11.5% per leg (for a single leg)
            # but since we have 3 longs and 3 shorts that are imperfectly correlated, this targets
            # roughly 10% portfolio annualized for a diversified basket.
            target_leg_ann = TARGET_LEG_VOL_MONTHLY * np.sqrt(12.0)  # = 0.1155

            w = pd.Series(0.0, index=CCYS)
            for c in longs + shorts:
                pair_vol = v[c]
                if not np.isfinite(pair_vol) or pair_vol <= 0:
                    continue
                size = target_leg_ann / pair_vol
                # vol filter
                if pair_vol > VOL_FILTER_ZERO:
                    size = 0.0
                elif pair_vol > VOL_FILTER_HALF:
                    size *= 0.5
                if c in shorts:
                    size = -size
                w[c] = size

            # cap any single leg at 0.5 to avoid blow-ups in low-vol regimes
            w = w.clip(lower=-0.5, upper=0.5)

            weights.loc[d] = w.values
            last_w = w.copy()
            chosen_history.append({
                'rebal_date': d.strftime('%Y-%m-%d'),
                'longs': longs,
                'shorts': shorts,
                **{f'carry_{c}': float(carry[c]) for c in CCYS},
                **{f'vol_{c}': float(v[c]) for c in CCYS},
                **{f'w_{c}': float(w[c]) for c in CCYS},
            })
        else:
            weights.loc[d] = last_w.values

    # Daily P&L: weights from previous close * today's log return
    # (weights are set at open of rebal day; for simplicity we use prev-day weights to avoid lookahead)
    weights_lag = weights.shift(1).fillna(0.0)
    pair_pnl = weights_lag * log_rets.reindex(fx_bdays)
    pair_pnl = pair_pnl.fillna(0.0)
    daily_ret = pair_pnl.sum(axis=1)

    # Transaction costs at rebalance: 1.5 bps per leg traded
    # Compute change in weight on rebal days only
    w_change = weights.diff().abs().fillna(weights.abs())  # first day = abs
    # Only count costs on rebal days (where weights changed). w_change non-zero elsewhere == 0 since weights are flat.
    cost_per_day = w_change.sum(axis=1) * (COST_BPS_PER_LEG / 10000.0)
    daily_ret_net = daily_ret - cost_per_day

    out = pd.DataFrame({
        'daily_gross': daily_ret,
        'daily_cost': cost_per_day,
        'daily_net': daily_ret_net,
    }, index=fx_bdays)

    diagnostics_df = pd.DataFrame(chosen_history)
    return out, diagnostics_df, weights


def to_monthly_returns(daily_net: pd.Series) -> pd.Series:
    """Compound daily log returns into monthly arithmetic returns."""
    daily_simple = np.exp(daily_net) - 1.0
    # Group by year-month, compound: (1+r1)(1+r2)... - 1
    monthly = (1 + daily_simple).resample('ME').prod() - 1
    return monthly


def perf_stats(monthly_ret: pd.Series) -> dict:
    monthly_ret = monthly_ret.dropna()
    if len(monthly_ret) == 0:
        return {}
    mean_m = float(monthly_ret.mean())
    std_m = float(monthly_ret.std())
    ann_ret = (1 + monthly_ret).prod() ** (12.0 / len(monthly_ret)) - 1 if len(monthly_ret) > 0 else 0.0
    ann_vol = std_m * np.sqrt(12.0)
    sharpe = (mean_m * 12.0) / ann_vol if ann_vol > 0 else 0.0
    equity = (1 + monthly_ret).cumprod()
    dd = (equity / equity.cummax()) - 1.0
    max_dd = float(dd.min())
    worst_month = float(monthly_ret.min())
    best_month = float(monthly_ret.max())
    return {
        'n_months': int(len(monthly_ret)),
        'mean_monthly_return_pct': mean_m * 100.0,
        'ann_return_pct': float(ann_ret * 100.0),
        'ann_vol_pct': float(ann_vol * 100.0),
        'sharpe': float(sharpe),
        'max_drawdown_pct': max_dd * 100.0,
        'worst_month_pct': worst_month * 100.0,
        'best_month_pct': best_month * 100.0,
    }


def main():
    log('=' * 70)
    log('iter_4 G10 FX carry backtest')
    log('=' * 70)

    rates = load_or_fetch_rates()
    fx = load_or_fetch_fx()

    log(f'rates shape={rates.shape}  cols={list(rates.columns)}')
    log(f'rates non-null counts:\n{rates.notna().sum()}')
    log(f'fx shape={fx.shape}  cols={list(fx.columns)}')

    # Align: keep FX bdays from 2010-01-01 onwards (warmup ends).
    fx = fx.loc['2010-01-01':]
    log(f'FX after warmup trim: shape={fx.shape}  start={fx.index.min().date()}  end={fx.index.max().date()}')

    daily_pnl, diag_df, weights = compute_backtest(fx, rates)
    log(f'daily_pnl rows: {len(daily_pnl)}  mean_gross={daily_pnl["daily_gross"].mean():.6f}  mean_net={daily_pnl["daily_net"].mean():.6f}')

    monthly = to_monthly_returns(daily_pnl['daily_net'])

    # Split IS / OOS
    is_mask = monthly.index < pd.Timestamp('2019-01-01')
    oos_mask = monthly.index >= pd.Timestamp('2019-01-01')

    is_stats = perf_stats(monthly[is_mask])
    oos_stats = perf_stats(monthly[oos_mask])
    full_stats = perf_stats(monthly)

    log('---- IS 2010-2018 ----')
    for k, v in is_stats.items():
        log(f'  {k} = {v}')
    log('---- OOS 2019-now ----')
    for k, v in oos_stats.items():
        log(f'  {k} = {v}')
    log('---- FULL ----')
    for k, v in full_stats.items():
        log(f'  {k} = {v}')

    # Save monthly returns CSV
    out_df = pd.DataFrame({
        'date': monthly.index,
        'monthly_return': monthly.values,
        'sample': ['IS' if d < pd.Timestamp('2019-01-01') else 'OOS' for d in monthly.index],
    })
    out_df.to_csv(MONTHLY_CSV, index=False)
    log(f'Wrote {MONTHLY_CSV}  rows={len(out_df)}')

    # Pass criterion
    verdict = 'PASS' if oos_stats.get('sharpe', 0.0) >= 0.55 else 'FAIL'

    summary = {
        'strategy': 'iter_4_g10_fx_carry',
        'is_period': '2010-01..2018-12',
        'oos_period': '2019-01..present',
        'months_oos': oos_stats.get('n_months', 0),
        'mean_monthly_return_pct': oos_stats.get('mean_monthly_return_pct', 0.0),
        'ann_return_pct': oos_stats.get('ann_return_pct', 0.0),
        'ann_vol_pct': oos_stats.get('ann_vol_pct', 0.0),
        'sharpe_oos': oos_stats.get('sharpe', 0.0),
        'max_drawdown_pct': oos_stats.get('max_drawdown_pct', 0.0),
        'worst_month_pct': oos_stats.get('worst_month_pct', 0.0),
        'best_month_pct': oos_stats.get('best_month_pct', 0.0),
        'pass_threshold_sharpe': 0.55,
        'verdict': verdict,
        'is_sharpe': is_stats.get('sharpe', 0.0),
        'is_ann_ret_pct': is_stats.get('ann_return_pct', 0.0),
        'is_max_dd_pct': is_stats.get('max_drawdown_pct', 0.0),
        'is_n_months': is_stats.get('n_months', 0),
        'full_sharpe': full_stats.get('sharpe', 0.0),
        'full_n_months': full_stats.get('n_months', 0),
        'n_rebalances': int(len(diag_df)),
        'cost_bps_per_leg_roundtrip': COST_BPS_PER_LEG,
        'target_vol_per_leg_monthly_pct': TARGET_LEG_VOL_MONTHLY * 100.0,
        'vol_filter_half_ann_pct': VOL_FILTER_HALF * 100.0,
        'vol_filter_zero_ann_pct': VOL_FILTER_ZERO * 100.0,
        'data_sources': {
            'fx': 'yfinance daily Close',
            'rates': 'FRED IR3TIB01 (3m interbank) monthly',
            'rates_shift_days': '~1 month forward shift to avoid lookahead',
        },
        'note': (
            'Carry signal = i_c - i_USD using 3m interbank rates (proxy for policy rate). '
            'Rates lagged 1 month to avoid lookahead. '
            'FX inverted where needed so every leg expressed CCY/USD (long pair = long that ccy). '
            'IS 2010-2018 not used for parameter tuning; rules pre-registered from MSSS (2012).'
        ),
    }
    with SUMMARY_JSON.open('w') as f:
        json.dump(summary, f, indent=2, default=str)
    log(f'Wrote {SUMMARY_JSON}')

    log('=' * 70)
    log(f'RESULT_LINE: iter_4 | months_oos={oos_stats.get("n_months",0)} | '
        f'ann_ret={oos_stats.get("ann_return_pct",0.0):.2f}% | '
        f'sharpe={oos_stats.get("sharpe",0.0):.2f} | '
        f'max_dd={oos_stats.get("max_drawdown_pct",0.0):.2f}% | '
        f'{verdict}')
    log('=' * 70)


if __name__ == '__main__':
    main()
