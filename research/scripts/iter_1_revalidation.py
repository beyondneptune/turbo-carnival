#!/usr/bin/env python3
"""
iter_1 RE-VALIDATION (Agent A12) — survivorship-bias-aware re-run of the small-cap overnight
mean-reversion strategy. Three mitigations executed in sequence:

  Mitigation 1 — Different universe (S&P SmallCap 600 via IJR ETF holdings).
      The S&P 600 has a documented profitability screen that makes it a *differently* biased
      universe than current-IWM-membership-applied-retroactively. Still survivorship-biased
      (we pull the current IJR holdings, not point-in-time), but the bias is "quality small-cap"
      rather than "current Russell 2000". We re-run the same RSI(2)/vol/return signal on this
      universe over 2018-2025 and compare OOS Sharpe / mean.

  Mitigation 2 — Survival haircut on the existing iter_1_trades.csv (one-shot, seed 42).
      Per the spec, randomly flag 10% of OOS trades as "delisted" and replace their net_bps
      with -50% (-5000 bps). One-shot draw with numpy RNG(seed=42), then recompute mean and
      rolling-50 Sharpe on the haircut trade log.

  Mitigation 3 — Cost stress on the existing trade log (5 bps/side and 10 bps/side).
      The original backtest used 1 bp/side (= 2 bps round-trip), which is implausibly tight
      for small-cap MOC/MOO auctions. We recover gross_bps and recompute net at 5 and 10
      bps/side, reporting OOS mean and rolling-50 Sharpe at each level.

PASS criteria (re-validation gate, all three mitigations):
  - OOS mean net >= +5 bps
  - OOS rolling-50-trade Sharpe >= 0.5
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------- Paths ----------------
ROOT = Path("/Users/weeoto/turbo-carnival")
RESEARCH = ROOT / "research"
CACHE_DIR = RESEARCH / "data" / "cache"
RESULTS_DIR = RESEARCH / "results"
LOG_DIR = RESEARCH / "logs"

R2K_CACHE = CACHE_DIR / "r2000_daily.parquet"  # existing cache from iter_1_backtest
IJR_HOLDINGS_PATH = CACHE_DIR / "ijr_holdings.csv"
SP600_DAILY_CACHE = CACHE_DIR / "sp600_daily.parquet"  # union(R2K cache, IJR-only names)
TRADES_CSV = RESULTS_DIR / "iter_1_trades.csv"
PRIOR_SUMMARY = RESULTS_DIR / "iter_1_summary.json"
OUT_JSON = RESULTS_DIR / "iter_1_revalidation.json"
LOG_PATH = LOG_DIR / "iter_1_revalidation.log"

for d in (CACHE_DIR, RESULTS_DIR, LOG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------- Config ----------------
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"
TRAIN_END = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-01")

RSI_PERIOD = 2
RSI_THRESH = 10.0
VOL_MULT_THRESH = 2.0
RET_THRESH = -0.025
TOP_N = 5
ROLLING_WINDOW = 50

# Mitigation 2 — one-shot haircut, seed 42, as specified by Agent A12 instructions
MITIGATION_2_P_DELIST = 0.10
MITIGATION_2_LOSS_PCT = -0.50  # -50% return on delisted trades
MITIGATION_2_SEED = 42

# Mitigation 3 — cost stress at 5 bps/side and 10 bps/side
MITIGATION_3_COSTS_BPS_PER_SIDE = [5.0, 10.0]

PASS_SHARPE = 0.5     # re-validation gate
PASS_MEAN_BPS = 5.0   # net of cost


# ---------------- Logging ----------------
_log_file = None


def log(msg: str) -> None:
    global _log_file
    if _log_file is None:
        _log_file = open(LOG_PATH, "a", buffering=1)
        _log_file.write(f"\n{'=' * 70}\nRUN START {datetime.now()}\n{'=' * 70}\n")
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    _log_file.write(line + "\n")


# ---------------- Mitigation 1: S&P 600 universe ----------------
def download_ijr_holdings() -> list[str]:
    """Download current IJR (iShares Core S&P Small-Cap ETF) holdings as a proxy for S&P 600."""
    if not IJR_HOLDINGS_PATH.exists():
        log("Downloading IJR holdings (S&P 600 proxy)...")
        url = (
            "https://www.ishares.com/us/products/239774/"
            "ishares-core-sp-smallcap-etf/1467271812596.ajax"
            "?fileType=csv&fileName=IJR_holdings&dataType=fund"
        )
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
        })
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        IJR_HOLDINGS_PATH.write_bytes(data)
        log(f"Saved IJR holdings to {IJR_HOLDINGS_PATH} ({len(data)} bytes)")

    with open(IJR_HOLDINGS_PATH, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()
    header_idx = None
    for i, ln in enumerate(lines):
        if ln.startswith("Ticker,Name,"):
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Could not find ticker header in IJR holdings CSV")
    reader = csv.DictReader(lines[header_idx:])
    tickers = []
    for row in reader:
        t = (row.get("Ticker") or "").strip()
        asset = (row.get("Asset Class") or "").strip()
        if not t or asset != "Equity":
            continue
        if any(c in t for c in [" ", "."]):
            continue
        tickers.append(t)
    tickers = sorted(set(tickers))
    log(f"Parsed {len(tickers)} S&P 600 (IJR) tickers")
    return tickers


def build_sp600_panel(tickers: list[str]) -> pd.DataFrame:
    """Reuse cached r2000 data where overlap exists; download IJR-only names. Cache the result."""
    if SP600_DAILY_CACHE.exists():
        log(f"Loading S&P 600 cache: {SP600_DAILY_CACHE}")
        return pd.read_parquet(SP600_DAILY_CACHE)

    log("Building S&P 600 daily panel — reuse cached R2K rows + download IJR-only names")
    cached = pd.read_parquet(R2K_CACHE)
    cached_set = set(cached["ticker"].unique())
    target_set = set(tickers)
    reused = sorted(target_set & cached_set)
    missing = sorted(target_set - cached_set)
    log(f"  reused from cache: {len(reused)}")
    log(f"  to download: {len(missing)}")

    frames = []
    if reused:
        frames.append(cached[cached["ticker"].isin(reused)].copy())

    BATCH = 50
    for i in range(0, len(missing), BATCH):
        batch = missing[i:i + BATCH]
        log(f"  yf batch {i//BATCH + 1}/{(len(missing)-1)//BATCH + 1}: "
            f"{batch[0]}..{batch[-1]} ({len(batch)} names)")
        try:
            raw = yf.download(
                batch, start=START_DATE, end=END_DATE,
                progress=False, auto_adjust=False, group_by="ticker",
                threads=True, timeout=60,
            )
        except Exception as e:
            log(f"    batch failed: {e}; sleep 5s + single-thread retry")
            time.sleep(5)
            try:
                raw = yf.download(
                    batch, start=START_DATE, end=END_DATE,
                    progress=False, auto_adjust=False, group_by="ticker",
                    threads=False, timeout=120,
                )
            except Exception as e2:
                log(f"    skip batch (failed twice): {e2}")
                continue

        if isinstance(raw.columns, pd.MultiIndex):
            for t in batch:
                if t not in raw.columns.get_level_values(0):
                    continue
                sub = raw[t].copy()
                if sub.empty or sub["Close"].isna().all():
                    continue
                sub = sub.reset_index()
                sub["ticker"] = t
                sub.columns = [c.lower() if isinstance(c, str) else c for c in sub.columns]
                sub = sub.rename(columns={"adj close": "adj_close"})
                keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
                keep = [c for c in keep if c in sub.columns]
                frames.append(sub[keep])
        else:
            if not raw.empty and not raw["Close"].isna().all():
                sub = raw.reset_index()
                sub["ticker"] = batch[0]
                sub.columns = [c.lower() if isinstance(c, str) else c for c in sub.columns]
                sub = sub.rename(columns={"adj close": "adj_close"})
                keep = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]
                keep = [c for c in keep if c in sub.columns]
                frames.append(sub[keep])
        time.sleep(0.5)

    if not frames:
        raise RuntimeError("S&P 600 panel: no data assembled")
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["close", "open", "volume"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    log(f"S&P 600 panel: {df['ticker'].nunique()} tickers, {len(df)} rows")
    df.to_parquet(SP600_DAILY_CACHE, index=False)
    return df


# ---------------- Features & signal (mirrors iter_1_backtest) ----------------
def compute_rsi2(close: pd.Series) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    avg_loss = loss.ewm(alpha=1.0 / RSI_PERIOD, adjust=False, min_periods=RSI_PERIOD).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(avg_gain != 0, 0.0)
    return rsi


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    df["adj_close"] = df["adj_close"].fillna(df["close"])
    for ticker, sub in df.groupby("ticker", sort=False):
        sub = sub.sort_values("date").reset_index(drop=True)
        if len(sub) < 30:
            continue
        sub["rsi2"] = compute_rsi2(sub["adj_close"])
        sub["vol_avg20"] = sub["volume"].rolling(20, min_periods=20).mean().shift(1)
        sub["vol_ratio"] = sub["volume"] / sub["vol_avg20"]
        sub["day_ret"] = sub["adj_close"].pct_change()
        sub["dollar_adv20"] = (sub["adj_close"] * sub["volume"]).rolling(
            20, min_periods=20
        ).mean().shift(1)
        adj_factor = (sub["adj_close"] / sub["close"]).replace(
            [np.inf, -np.inf], np.nan
        ).fillna(1.0)
        sub["adj_open"] = sub["open"] * adj_factor
        sub["next_adj_open"] = sub["adj_open"].shift(-1)
        sub["next_date"] = sub["date"].shift(-1)
        out.append(sub)
    res = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    log(f"Features: {len(res)} rows across "
        f"{res['ticker'].nunique() if len(res) else 0} tickers")
    return res


def generate_trades(feat: pd.DataFrame, cost_bps_per_side: float) -> pd.DataFrame:
    sig = feat[
        (feat["rsi2"] <= RSI_THRESH)
        & (feat["vol_ratio"] >= VOL_MULT_THRESH)
        & (feat["day_ret"] <= RET_THRESH)
        & feat["next_adj_open"].notna()
        & feat["dollar_adv20"].notna()
        & (feat["adj_close"] > 0)
    ].copy()
    if sig.empty:
        return pd.DataFrame()
    sig = sig.sort_values(["date", "dollar_adv20"], ascending=[True, False])
    sig["rank"] = sig.groupby("date").cumcount()
    trades = sig[sig["rank"] < TOP_N].copy()
    gross_ret = (trades["next_adj_open"] - trades["adj_close"]) / trades["adj_close"]
    gross_bps = gross_ret * 10000.0
    net_bps = gross_bps - 2 * cost_bps_per_side
    trades["gross_bps"] = gross_bps
    trades["net_bps"] = net_bps
    # Filter implausible bars (split/corp-action artifacts)
    trades = trades[trades["net_bps"].abs() <= 2500].copy()
    trades["entry_date"] = trades["date"]
    trades["exit_date"] = trades["next_date"]
    trades = trades[[
        "entry_date", "exit_date", "ticker",
        "rsi2", "vol_ratio", "day_ret", "dollar_adv20",
        "gross_bps", "net_bps",
    ]].sort_values("entry_date").reset_index(drop=True)
    return trades


# ---------------- Metrics ----------------
def rolling_sharpe(returns: np.ndarray, window: int) -> np.ndarray:
    if len(returns) < window:
        return np.array([])
    r = pd.Series(returns)
    mean = r.rolling(window).mean()
    std = r.rolling(window).std(ddof=1)
    sharpe = (mean / std) * np.sqrt(252)
    return sharpe.dropna().values


def max_drawdown(equity_curve: np.ndarray) -> float:
    if len(equity_curve) == 0:
        return 0.0
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve - peak) / peak
    return float(dd.min())


def trade_summary(trades: pd.DataFrame, label: str) -> dict:
    if trades.empty:
        return {"label": label, "trades": 0, "verdict": "FAIL"}
    net = trades["net_bps"].values
    oos = trades[trades["entry_date"] >= TEST_START]
    rs_oos = rolling_sharpe(oos["net_bps"].values, ROLLING_WINDOW)
    oos_mean = float(oos["net_bps"].mean()) if len(oos) else float("nan")

    portfolio_daily = trades.groupby("entry_date")["net_bps"].mean().sort_index()
    eq = np.cumprod(1 + portfolio_daily.values / 10000.0)
    mdd = max_drawdown(eq) * 100.0
    oos_daily = portfolio_daily[portfolio_daily.index >= TEST_START]
    if len(oos_daily) >= 2 and oos_daily.std() > 0:
        oos_portfolio_sharpe = float(oos_daily.mean() / oos_daily.std() * np.sqrt(252))
    else:
        oos_portfolio_sharpe = float("nan")

    rs_oos_med = float(np.median(rs_oos)) if len(rs_oos) else float("nan")
    win_rate = float((net > 0).mean())
    sharpe_ok = (not math.isnan(rs_oos_med)) and rs_oos_med >= PASS_SHARPE
    mean_ok = (not math.isnan(oos_mean)) and oos_mean >= PASS_MEAN_BPS
    verdict = "PASS" if (sharpe_ok and mean_ok) else "FAIL"

    return {
        "label": label,
        "trades": int(len(trades)),
        "oos_trades": int(len(oos)),
        "win_rate": round(win_rate, 4),
        "mean_bps_net_all": round(float(np.mean(net)), 3),
        # Use these stable keys for the headline metrics:
        "mean_bps": round(oos_mean, 3) if not math.isnan(oos_mean) else None,
        "sharpe": round(rs_oos_med, 3) if not math.isnan(rs_oos_med) else None,
        "oos_mean_bps_net": round(oos_mean, 3) if not math.isnan(oos_mean) else None,
        "rolling_50_oos_sharpe_median": (
            round(rs_oos_med, 3) if not math.isnan(rs_oos_med) else None
        ),
        "oos_portfolio_sharpe_daily_annualized": (
            round(oos_portfolio_sharpe, 3) if not math.isnan(oos_portfolio_sharpe) else None
        ),
        "max_drawdown_pct": round(mdd, 3),
        "sharpe_ok": bool(sharpe_ok),
        "mean_ok": bool(mean_ok),
        "verdict": verdict,
    }


# ---------------- Mitigation 2: ONE-SHOT survival haircut (seed 42) ----------------
def mitigation_2_survival_haircut(trades_csv: Path) -> dict:
    """One-shot: randomly haircut 10% of OOS trades by -50% return. Seed 42."""
    log("Mitigation 2: ONE-SHOT survival haircut on OOS trades")
    log(f"  P_delist={MITIGATION_2_P_DELIST}  loss_on_delist={MITIGATION_2_LOSS_PCT*100:+.0f}%  "
        f"seed={MITIGATION_2_SEED}")
    if not trades_csv.exists():
        return {"verdict": "FAIL", "reason": "trades csv missing"}
    df = pd.read_csv(trades_csv, parse_dates=["entry_date", "exit_date"])
    if df.empty:
        return {"verdict": "FAIL", "reason": "no trades"}

    oos = df[df["entry_date"] >= TEST_START].copy().reset_index(drop=True)
    n = len(oos)
    if n == 0:
        return {"verdict": "FAIL", "reason": "no OOS trades"}

    # Deterministic one-shot draw, seed 42
    rng = np.random.default_rng(MITIGATION_2_SEED)
    flags = rng.random(n) < MITIGATION_2_P_DELIST  # boolean, ~10% True
    n_flagged = int(flags.sum())
    log(f"  OOS trades: {n}; flagged delisted: {n_flagged} ({n_flagged/n*100:.2f}%)")

    haircut_bps = MITIGATION_2_LOSS_PCT * 10000.0  # -5000 bps for -50%
    base = oos["net_bps"].values
    # Replace flagged trade returns with -5000 bps (the -50% delisting loss)
    adj = np.where(flags, haircut_bps, base)

    adj_mean = float(np.mean(adj))
    # Rolling-50 Sharpe on the haircut OOS time series (preserve trade ordering)
    rs = rolling_sharpe(adj, ROLLING_WINDOW)
    rs_med = float(np.median(rs)) if len(rs) else float("nan")
    if np.std(adj, ddof=1) > 0:
        per_trade_sharpe = float(adj_mean / np.std(adj, ddof=1) * np.sqrt(252))
    else:
        per_trade_sharpe = float("nan")

    # Also closed-form expected adjustment (sanity check):
    closed_form_mean = (
        (1 - MITIGATION_2_P_DELIST) * float(np.mean(base))
        + MITIGATION_2_P_DELIST * haircut_bps
    )

    sharpe_ok = (not math.isnan(rs_med)) and rs_med >= PASS_SHARPE
    mean_ok = (not math.isnan(adj_mean)) and adj_mean >= PASS_MEAN_BPS
    verdict = "PASS" if (sharpe_ok and mean_ok) else "FAIL"

    log(f"  baseline OOS mean: {float(np.mean(base)):+.2f} bps")
    log(f"  haircut OOS mean: {adj_mean:+.2f} bps")
    log(f"  closed-form expected: {closed_form_mean:+.2f} bps (sanity)")
    log(f"  haircut rolling-50 OOS Sharpe (median): {rs_med:.3f}")
    log(f"  haircut per-trade annualized Sharpe: {per_trade_sharpe:.3f}")
    log(f"  verdict: {verdict}")

    return {
        "method": "one_shot_haircut_seed_42",
        "oos_trades": int(n),
        "p_delist": MITIGATION_2_P_DELIST,
        "loss_on_delist_pct": MITIGATION_2_LOSS_PCT,
        "loss_on_delist_bps": haircut_bps,
        "seed": MITIGATION_2_SEED,
        "n_flagged_delisted": n_flagged,
        "baseline_oos_mean_bps": round(float(np.mean(base)), 3),
        "mean_bps": round(adj_mean, 3),
        "closed_form_mean_bps": round(closed_form_mean, 3),
        "sharpe": round(rs_med, 3) if not math.isnan(rs_med) else None,
        "rolling_50_oos_sharpe_median": (
            round(rs_med, 3) if not math.isnan(rs_med) else None
        ),
        "per_trade_annualized_sharpe": (
            round(per_trade_sharpe, 3) if not math.isnan(per_trade_sharpe) else None
        ),
        "sharpe_ok": bool(sharpe_ok),
        "mean_ok": bool(mean_ok),
        "verdict": verdict,
    }


# ---------------- Mitigation 3: cost stress at 5 and 10 bps/side ----------------
def mitigation_3_cost_stress(trades_csv: Path) -> dict:
    log("Mitigation 3: cost stress on OOS trade log at 5 and 10 bps/side")
    if not trades_csv.exists():
        return {"verdict": "FAIL", "reason": "trades csv missing"}
    df = pd.read_csv(trades_csv, parse_dates=["entry_date", "exit_date"])
    if df.empty:
        return {"verdict": "FAIL", "reason": "no trades"}

    # The original trades csv stored net_bps at 1 bp/side = 2 bps round trip.
    # Recover gross by adding back 2 bps then re-apply new round-trip cost.
    df["gross_bps_recovered"] = df["net_bps"] + 2.0 * 1.0
    oos_mask = df["entry_date"] >= TEST_START

    results = []
    for cost in MITIGATION_3_COSTS_BPS_PER_SIDE:
        df_c = df.copy()
        df_c["net_bps_at_cost"] = df_c["gross_bps_recovered"] - 2.0 * cost
        oos_c = df_c[oos_mask].copy().sort_values("entry_date").reset_index(drop=True)
        if len(oos_c) == 0:
            continue
        rs = rolling_sharpe(oos_c["net_bps_at_cost"].values, ROLLING_WINDOW)
        rs_med = float(np.median(rs)) if len(rs) else float("nan")
        portfolio_daily = oos_c.groupby("entry_date")["net_bps_at_cost"].mean().sort_index()
        if len(portfolio_daily) >= 2 and portfolio_daily.std() > 0:
            port_sharpe = float(
                portfolio_daily.mean() / portfolio_daily.std() * np.sqrt(252)
            )
        else:
            port_sharpe = float("nan")
        m = float(oos_c["net_bps_at_cost"].mean())
        wr = float((oos_c["net_bps_at_cost"].values > 0).mean())
        sharpe_ok = (not math.isnan(rs_med)) and rs_med >= PASS_SHARPE
        mean_ok = m >= PASS_MEAN_BPS
        verdict = "PASS" if (sharpe_ok and mean_ok) else "FAIL"
        log(f"  cost={cost:.1f} bps/side  oos_mean={m:+.2f} bps  "
            f"rolling50_sharpe_med={rs_med:.3f}  port_sharpe={port_sharpe:.3f}  "
            f"win_rate={wr:.3f}  verdict={verdict}")
        results.append({
            "cost_bps_per_side": cost,
            "oos_trades": int(len(oos_c)),
            "mean_bps": round(m, 3),
            "oos_mean_bps_net": round(m, 3),
            "win_rate": round(wr, 4),
            "sharpe": round(rs_med, 3) if not math.isnan(rs_med) else None,
            "rolling_50_oos_sharpe_median": (
                round(rs_med, 3) if not math.isnan(rs_med) else None
            ),
            "oos_portfolio_sharpe_daily_annualized": (
                round(port_sharpe, 3) if not math.isnan(port_sharpe) else None
            ),
            "sharpe_ok": bool(sharpe_ok),
            "mean_ok": bool(mean_ok),
            "verdict": verdict,
        })

    # Overall mitigation-3 verdict: both cost levels must pass
    overall_verdict = "FAIL"
    pass_count = sum(1 for r in results if r["verdict"] == "PASS")
    if pass_count == len(results) and len(results) > 0:
        overall_verdict = "PASS"
    elif pass_count > 0:
        overall_verdict = "BORDERLINE"

    # Headline metrics for the report at 10 bps/side (worst case)
    at_5 = next((r for r in results if r["cost_bps_per_side"] == 5.0), {})
    at_10 = next((r for r in results if r["cost_bps_per_side"] == 10.0), {})

    return {
        "by_cost_level": results,
        # Headline using the 10 bps/side worst case for the re-val report line
        "mean_bps": at_10.get("mean_bps"),
        "sharpe": at_10.get("sharpe"),
        "mean_bps_at_5bps_per_side": at_5.get("mean_bps"),
        "sharpe_at_5bps_per_side": at_5.get("sharpe"),
        "mean_bps_at_10bps_per_side": at_10.get("mean_bps"),
        "sharpe_at_10bps_per_side": at_10.get("sharpe"),
        "verdict": overall_verdict,
    }


# ---------------- Mitigation 1 wrapper ----------------
def mitigation_1_sp600_universe() -> dict:
    log("Mitigation 1: S&P 600 (IJR) universe re-run")
    tickers = download_ijr_holdings()
    df = build_sp600_panel(tickers)
    feat = compute_features(df)
    trades = generate_trades(feat, cost_bps_per_side=1.0)  # match prior cost assumption
    log(f"  trades on S&P 600 universe: {len(trades)}")
    if trades.empty:
        return {"verdict": "FAIL", "reason": "no trades generated"}
    out_csv = RESULTS_DIR / "iter_1_revalidation_sp600_trades.csv"
    persist = trades.copy()
    persist["entry_date"] = persist["entry_date"].dt.strftime("%Y-%m-%d")
    persist["exit_date"] = persist["exit_date"].dt.strftime("%Y-%m-%d")
    persist.to_csv(out_csv, index=False)
    log(f"  wrote {out_csv}")
    summary = trade_summary(trades, label="sp600_universe")
    summary["trades_csv"] = str(out_csv)
    return summary


def final_verdict(m1: dict, m2: dict, m3: dict) -> str:
    """PASS iff all three mitigations show mean >= +5 bps AND sharpe >= 0.5."""
    verdicts = [m1.get("verdict"), m2.get("verdict"), m3.get("verdict")]
    pass_count = sum(1 for v in verdicts if v == "PASS")
    if pass_count == 3:
        return "PASS"
    if pass_count == 0:
        return "FAIL"
    return "BORDERLINE"


def main() -> int:
    t0 = time.time()
    log("=" * 70)
    log("iter_1 RE-VALIDATION starting (Agent A12, survivorship-aware)")
    log("=" * 70)

    prior = json.loads(PRIOR_SUMMARY.read_text())
    prior_oos_mean = prior.get("oos_mean_bps_net")
    prior_oos_sharpe = prior.get("rolling_50_trade_sharpe_oos")
    log(f"Prior reported: OOS rolling50 Sharpe={prior_oos_sharpe} OOS mean={prior_oos_mean} bps")
    log(f"Prior verdict: {prior.get('verdict')}")

    # --- Mitigation 1: S&P 600 universe ---
    m1 = mitigation_1_sp600_universe()
    log(f"Mitigation 1 summary: trades={m1.get('trades')} oos_mean={m1.get('mean_bps')} "
        f"sharpe={m1.get('sharpe')} verdict={m1.get('verdict')}")

    # --- Mitigation 2: one-shot survival haircut (seed 42) ---
    m2 = mitigation_2_survival_haircut(TRADES_CSV)

    # --- Mitigation 3: cost stress (5, 10 bps/side) ---
    m3 = mitigation_3_cost_stress(TRADES_CSV)

    fv = final_verdict(m1, m2, m3)

    # Comparison to prior (for context only)
    m1_oos_mean = m1.get("mean_bps")
    m1_oos_sharpe = m1.get("sharpe")
    comparison_m1 = {
        "prior_oos_mean_bps": prior_oos_mean,
        "sp600_oos_mean_bps": m1_oos_mean,
        "delta_mean_bps": (
            round(m1_oos_mean - prior_oos_mean, 3)
            if (m1_oos_mean is not None and prior_oos_mean is not None)
            else None
        ),
        "prior_oos_rolling50_sharpe": prior_oos_sharpe,
        "sp600_oos_rolling50_sharpe": m1_oos_sharpe,
        "delta_sharpe": (
            round(m1_oos_sharpe - prior_oos_sharpe, 3)
            if (m1_oos_sharpe is not None and prior_oos_sharpe is not None)
            else None
        ),
    }

    out = {
        "run_timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "agent": "A12",
        "prior_summary_path": str(PRIOR_SUMMARY),
        "prior_oos_mean_bps": prior_oos_mean,
        "prior_oos_rolling50_sharpe": prior_oos_sharpe,
        "prior_verdict": prior.get("verdict"),
        "pass_thresholds_used": {
            "rolling_50_oos_sharpe_>=": PASS_SHARPE,
            "oos_mean_bps_net_>=": PASS_MEAN_BPS,
        },
        "mitigation_1": {
            **m1,
            "comparison_to_prior": comparison_m1,
        },
        "mitigation_2": m2,
        "mitigation_3": m3,
        "final_verdict": fv,
        "interpretation": (
            "PASS = all 3 mitigations show OOS mean >= +5 bps AND OOS rolling-50 Sharpe >= 0.5. "
            "BORDERLINE = some pass, some fail. FAIL = none pass under their adjustments. "
            "Note: 'PASS' here is the RE-VALIDATION gate (Sharpe>=0.5, mean>=+5 bps), intentionally "
            "weaker than the original spec gate (Sharpe>=0.8, mean>=+10 bps) — asks 'is the edge "
            "still positive under realistic adjustments?', not 'is it still as strong as the "
            "survivorship-biased reading suggested?'"
        ),
    }

    OUT_JSON.write_text(json.dumps(out, indent=2, default=str))
    log(f"Wrote {OUT_JSON}")

    log("=" * 70)
    log("FINAL")
    log("=" * 70)
    log(f"Mitigation 1 (S&P 600 universe): {m1.get('verdict')} "
        f"trades={m1.get('trades')} oos_mean={m1.get('mean_bps')} bps "
        f"rolling50_sharpe={m1.get('sharpe')}")
    log(f"Mitigation 2 (survival haircut, seed 42): {m2.get('verdict')} "
        f"adj_mean={m2.get('mean_bps')} bps adj_sharpe={m2.get('sharpe')}")
    log(f"Mitigation 3 (cost stress 5/10 bps/side): {m3.get('verdict')} "
        f"mean@5={m3.get('mean_bps_at_5bps_per_side')} mean@10={m3.get('mean_bps_at_10bps_per_side')} "
        f"sharpe@10={m3.get('sharpe_at_10bps_per_side')}")
    log(f"FINAL_VERDICT: {fv}")
    log(f"Total elapsed: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
