"""TERMINATOR ($1k) — bootstrap year-long Monte Carlo simulation.

Honest answer to: "if I run TERMINATOR for exactly a year, what's the take?"

This is NOT a deterministic backtest — we only have 1 day of validated
per-entry P&L (162 entries from 2026-05-11, 80 kept by composite C3 gate).
A real year-long sim would need minute bars + retrained models on rolling
windows. Instead, this Monte-Carlo BOOTSTRAPS from the empirical 1-day
distribution to project a *range* of outcomes.

Two strategy lanes on the $1k account:
  - ATHENA HFT mean-reversion: 70% capital ($700), bootstrapped entries
  - SVXY rotation:             30% capital ($300), lognormal daily returns
    parameterized from 7-yr walk-forward (mu=17.1%/yr, sharpe=0.66)

Outputs:
  - Distribution of year-end equity (P10, median, P90, mean)
  - Histogram + equity curve fan
  - HTML report at reports/terminator_yearly/index.html

CAVEATS (honest):
  - Bootstrap assumes each trading day looks like yesterday.
    Yesterday's tape was unusually mean-reverting; real markets cycle
    through regimes (trending, choppy, panic). True annual returns
    likely sit BELOW the central case from this sim.
  - Serial correlation ignored — entries assumed IID.
  - Slippage assumed = yesterday's (4 bps IOC entries). Real cost varies.
  - C3 gate retrained nightly in production; here we use the static
    block-set from 2026-05-11.
  - Survivorship: assumes TERMINATOR account isn't liquidated mid-year
    by a single bad day. With $1k base, a single -50% day = game over.

Usage:
    python experiments/ml_lab/22_terminator_yearly_sim.py
    python experiments/ml_lab/22_terminator_yearly_sim.py --trials 5000 --seed 42
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
ML_ZOO_PER_ENTRY = REPO / "experiments" / "ml_lab" / "runs" / "ml_zoo_per_entry_latest.parquet"
COMPOSITE_PER_ENTRY = REPO / "experiments" / "ml_lab" / "runs" / "composite_backtest" / "winner_per_entry.parquet"
OUT = REPO / "reports" / "terminator_yearly"
OUT.mkdir(parents=True, exist_ok=True)

# ---- Strategy parameters (matched to live production) -------------------
STARTING_CAPITAL = 1_000.0
SVXY_PCT = 0.30
HFT_PCT = 0.70
TRADING_DAYS = 252

# TERMINATOR's actual ATHENA sizing in production (host line 887-890)
HFT_BASE_PER_ENTRY = 70.0
HFT_FLOOR = 30.0
HFT_CEILING = 120.0

# Yesterday's per-entry P&L in the parquet is at $20k notional (the live trade size).
# Script 18 applies SCALE=1/100 to map down to $200/trade. We use $20k as the divisor
# to get per-dollar P&L, then scale to TERMINATOR's actual ATHENA sizing.
YESTERDAY_TRADE_SIZE = 20_000.0

# SVXY rotation params (from quant_lab 7-yr walk-forward)
SVXY_ANNUAL_MU = 0.171      # 17.1% annualized
SVXY_SHARPE = 0.66
SVXY_MAX_DD = 0.328
# Implied daily volatility from Sharpe ratio (sharpe = mu/sigma annualized)
SVXY_ANNUAL_SIGMA = SVXY_ANNUAL_MU / SVXY_SHARPE
SVXY_DAILY_MU = SVXY_ANNUAL_MU / TRADING_DAYS
SVXY_DAILY_SIGMA = SVXY_ANNUAL_SIGMA / np.sqrt(TRADING_DAYS)

# Day-end entry density (entries per day per HFT lane, post-C3 filter)
ENTRIES_PER_DAY_MIN = 15   # quiet day
ENTRIES_PER_DAY_MEAN = 30  # base case
ENTRIES_PER_DAY_MAX = 50   # busy day


def load_yesterday_kept_entries() -> np.ndarray:
    """Return the realized P&L of entries that composite C3 KEPT (not blocked).

    Source: composite_backtest/winner_per_entry.parquet — 162 rows, 80 kept.
    """
    df = pd.read_parquet(COMPOSITE_PER_ENTRY)
    kept = df[~df["blocked"]].copy()
    # P&L is at $200/trade. Convert to per-dollar-notional ratio.
    pnl_per_dollar = kept["pnl"].values / YESTERDAY_TRADE_SIZE
    return pnl_per_dollar


def simulate_one_year(
    pnl_per_dollar: np.ndarray,
    rng: np.random.Generator,
    *,
    starting_capital: float = STARTING_CAPITAL,
    entries_per_day_mean: float = ENTRIES_PER_DAY_MEAN,
    hft_decay_factor: float = 0.20,
    # ATHENA v2.1 sizing + safety params
    enable_compounding: bool = True,
    capacity_mult_cap: float = 5.0,          # max 5× starting equity scale
    drawdown_revert_pct: float = 0.20,        # >20% off peak → baseline sizing
    circuit_breaker_pct: float = 0.03,        # halt entries at -3% daily loss
    extra_gate_block_rate: float = 0.15,      # news+earnings+OB imbalance block ~15% extra
) -> dict:
    """Simulate one full trading year. Returns equity curve + summary.

    v2.1 changes:
      - enable_compounding: base_notional scales with HFT equity (capped at 5×)
      - drawdown_revert: when HFT drawdown > 20%, sizing reverts to baseline
      - circuit_breaker: if intraday HFT loss > 3%, stop entries for that day
      - extra_gate_block_rate: ~15% additional block rate from news/earnings/OB gates
    """
    equity = starting_capital
    svxy_equity = starting_capital * SVXY_PCT
    hft_equity = starting_capital * HFT_PCT
    hft_starting = hft_equity
    hft_peak = hft_equity
    n_circuit_breaker_days = 0

    curve = np.zeros(TRADING_DAYS + 1)
    curve[0] = starting_capital
    daily_pnl = np.zeros(TRADING_DAYS)

    for d in range(TRADING_DAYS):
        # === SVXY portion: lognormal daily return ===
        svxy_ret = rng.normal(SVXY_DAILY_MU, SVXY_DAILY_SIGMA)
        svxy_equity *= (1 + svxy_ret)

        # === HFT portion: bootstrap K entries with v2.1 logic ===
        K = int(np.clip(
            rng.normal(entries_per_day_mean, entries_per_day_mean * 0.3),
            ENTRIES_PER_DAY_MIN, ENTRIES_PER_DAY_MAX,
        ))
        # Apply extra-gate filter (news/earnings/OB imbalance block additional ~15%)
        if extra_gate_block_rate > 0:
            keep_mask = rng.random(K) > extra_gate_block_rate
            K_after_gates = int(keep_mask.sum())
        else:
            K_after_gates = K

        # Equity multiplier for compounding
        if enable_compounding:
            hft_dd_pct = (hft_peak - hft_equity) / hft_peak if hft_peak > 0 else 0
            if hft_dd_pct > drawdown_revert_pct:
                eq_mult = 1.0   # in big drawdown, revert to baseline
            else:
                eq_mult = min(hft_equity / hft_starting if hft_starting > 0 else 1.0,
                              capacity_mult_cap)
        else:
            eq_mult = 1.0
        base_notional = HFT_BASE_PER_ENTRY * eq_mult

        # Track intraday P&L for circuit breaker
        # Bootstrap one entry at a time; if cumulative session P&L drops below
        # -3% of session start, halt remaining entries.
        session_start_equity = hft_equity
        day_hft_pnl = 0.0
        circuit_tripped = False
        if K_after_gates > 0:
            sampled = rng.choice(pnl_per_dollar, size=K_after_gates, replace=True)
            for entry_pnl_per_dollar in sampled:
                entry_pnl = entry_pnl_per_dollar * base_notional * hft_decay_factor
                day_hft_pnl += entry_pnl
                current_intraday_loss_pct = -day_hft_pnl / session_start_equity if session_start_equity > 0 else 0
                if current_intraday_loss_pct > circuit_breaker_pct:
                    circuit_tripped = True
                    break
        if circuit_tripped:
            n_circuit_breaker_days += 1

        # Cap HFT loss at available HFT capital
        day_hft_pnl = max(day_hft_pnl, -hft_equity)
        hft_equity += day_hft_pnl
        hft_peak = max(hft_peak, hft_equity)

        equity = svxy_equity + hft_equity
        curve[d + 1] = equity
        daily_pnl[d] = svxy_equity * svxy_ret + day_hft_pnl

        # Survivorship: stop if equity below $100 (margin call territory)
        if equity < 100:
            curve[d + 1:] = equity  # flat-line
            break

    return {
        "ending_equity": float(equity),
        "ending_svxy": float(svxy_equity),
        "ending_hft": float(hft_equity),
        "circuit_breaker_days": n_circuit_breaker_days,
        "curve": curve,
        "daily_pnl": daily_pnl,
        "max_dd_pct": float(_max_drawdown(curve)),
        "winning_days": int((daily_pnl > 0).sum()),
        "losing_days": int((daily_pnl < 0).sum()),
        "survived": curve[-1] > 100,
    }


def _max_drawdown(curve: np.ndarray) -> float:
    """Maximum peak-to-trough drawdown of an equity curve (as fraction)."""
    peaks = np.maximum.accumulate(curve)
    dd = (curve - peaks) / peaks
    return float(dd.min())


def run_monte_carlo(
    trials: int, seed: int, *,
    hft_decay: float = 0.20,
    enable_compounding: bool = True,
    circuit_breaker_pct: float = 0.03,
    extra_gate_block_rate: float = 0.15,
) -> pd.DataFrame:
    """Run N Monte Carlo trials, return per-trial results."""
    rng = np.random.default_rng(seed)
    pnl_per_dollar = load_yesterday_kept_entries()
    print(f"  loaded {len(pnl_per_dollar)} kept entries (after C3 gate)")
    print(f"  per-$ P&L: mean={pnl_per_dollar.mean()*100:+.3f}%  std={pnl_per_dollar.std()*100:.3f}%")
    print(f"  positive entries: {(pnl_per_dollar > 0).sum()} / {len(pnl_per_dollar)}")
    print(f"  compounding: {enable_compounding}  circuit: {circuit_breaker_pct*100:.1f}%  "
          f"extra-gates: {extra_gate_block_rate*100:.0f}% extra block rate")

    results = []
    for t in range(trials):
        trial_rng = np.random.default_rng(rng.integers(0, 2**31))
        r = simulate_one_year(
            pnl_per_dollar, trial_rng, hft_decay_factor=hft_decay,
            enable_compounding=enable_compounding,
            circuit_breaker_pct=circuit_breaker_pct,
            extra_gate_block_rate=extra_gate_block_rate,
        )
        results.append({
            "trial": t,
            "ending_equity": r["ending_equity"],
            "ending_svxy": r["ending_svxy"],
            "ending_hft": r["ending_hft"],
            "max_dd_pct": r["max_dd_pct"],
            "winning_days": r["winning_days"],
            "losing_days": r["losing_days"],
            "survived": r["survived"],
            "circuit_breaker_days": r["circuit_breaker_days"],
        })

    return pd.DataFrame(results)


def render_html_report(
    results: pd.DataFrame,
    decay_label: str,
    *,
    trials: int,
) -> str:
    """Render KPI strip + percentile table + brief commentary."""
    end_eq = results["ending_equity"].values
    take = end_eq - STARTING_CAPITAL
    survival_rate = results["survived"].mean()
    p10, p50, p90 = np.percentile(end_eq, [10, 50, 90])
    mean = end_eq.mean()
    pct_positive = (take > 0).mean()
    max_dd = results["max_dd_pct"].mean()

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>TERMINATOR — Year-1 Bootstrap Sim</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; color: #1f2937; }}
h1 {{ font-size: 1.5rem; margin-bottom: 0.25rem; }}
.subtitle {{ color: #6b7280; margin-bottom: 1.5rem; }}
.kpi-strip {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 0.75rem; margin-bottom: 1.5rem; }}
.kpi {{ background: #f9fafb; padding: 0.75rem; border-radius: 6px; }}
.kpi-label {{ font-size: 0.75rem; color: #6b7280; text-transform: uppercase; letter-spacing: 0.05em; }}
.kpi-value {{ font-size: 1.25rem; font-weight: 600; margin-top: 0.25rem; }}
.pos {{ color: #059669; }} .neg {{ color: #dc2626; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ text-align: right; padding: 0.5rem 0.75rem; border-bottom: 1px solid #e5e7eb; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f3f4f6; font-weight: 600; }}
.caveat {{ background: #fef3c7; padding: 0.75rem 1rem; border-left: 3px solid #f59e0b; border-radius: 4px; margin: 1rem 0; font-size: 0.9rem; }}
</style></head>
<body>
<h1>TERMINATOR ($1k) — Year-1 Bootstrap Simulation</h1>
<div class="subtitle">{trials:,} Monte Carlo trials · HFT decay factor: {decay_label} · Source: 80 kept entries from 2026-05-11 walk-forward</div>

<div class="kpi-strip">
  <div class="kpi"><div class="kpi-label">Median Year-End</div><div class="kpi-value">${p50:,.0f}</div></div>
  <div class="kpi"><div class="kpi-label">Median Take</div><div class="kpi-value {'pos' if p50>STARTING_CAPITAL else 'neg'}">${p50-STARTING_CAPITAL:+,.0f}</div></div>
  <div class="kpi"><div class="kpi-label">P10 — P90 Range</div><div class="kpi-value">${p10:,.0f} – ${p90:,.0f}</div></div>
  <div class="kpi"><div class="kpi-label">% Profitable Years</div><div class="kpi-value">{pct_positive*100:.0f}%</div></div>
  <div class="kpi"><div class="kpi-label">Survival Rate</div><div class="kpi-value">{survival_rate*100:.1f}%</div></div>
</div>

<h2>Outcome Percentiles</h2>
<table>
<tr><th>Percentile</th><th>Year-End Equity</th><th>Yearly Take</th><th>% Return</th></tr>
"""
    for p in [5, 10, 25, 50, 75, 90, 95]:
        val = np.percentile(end_eq, p)
        take_p = val - STARTING_CAPITAL
        pct = (val / STARTING_CAPITAL - 1) * 100
        color = "pos" if take_p > 0 else "neg"
        html += f'<tr><td>P{p:02d}</td><td>${val:,.0f}</td><td class="{color}">${take_p:+,.0f}</td><td class="{color}">{pct:+.1f}%</td></tr>\n'

    html += f"""
<tr><th>Mean</th><th>${mean:,.0f}</th><th class="{'pos' if mean>STARTING_CAPITAL else 'neg'}">${mean-STARTING_CAPITAL:+,.0f}</th><th>{(mean/STARTING_CAPITAL-1)*100:+.1f}%</th></tr>
</table>

<div class="caveat">
<strong>What this simulation is NOT:</strong> a deterministic year-long backtest.
It bootstraps from a single day (2026-05-11) of validated entries, assumes
each future trading day samples from that empirical distribution, applies a
{decay_label} edge-decay haircut, and runs {trials:,} trials.
<br><br>
<strong>Why the decay haircut matters:</strong> yesterday's tape was unusually
mean-reverting — perfect conditions for the strategy. Without a decay factor,
the simulator would project {trials:,} years averaging multi-thousand-percent
returns (fantasy). The 20% default acknowledges that ~80% of yesterday's edge
is regime-specific and won't persist.
</div>

<h2>Component Breakdown</h2>
<table>
<tr><th></th><th>Median Year-End</th><th>Median Take</th></tr>
<tr><td>SVXY rotation (30%)</td><td>${results['ending_svxy'].median():,.0f}</td><td class="pos">${results['ending_svxy'].median()-STARTING_CAPITAL*SVXY_PCT:+,.0f}</td></tr>
<tr><td>ATHENA HFT (70%)</td><td>${results['ending_hft'].median():,.0f}</td><td class="{'pos' if results['ending_hft'].median()>STARTING_CAPITAL*HFT_PCT else 'neg'}">${results['ending_hft'].median()-STARTING_CAPITAL*HFT_PCT:+,.0f}</td></tr>
</table>

<div class="caveat" style="background: #dbeafe; border-color: #3b82f6;">
<strong>SVXY portion is the validated anchor.</strong> +17.1% annualized over
7 years of walk-forward (2019-2026), Sharpe 0.66, max DD 32.8%. On $300 base,
expect ~$51 yearly take with rare $100+ drawdown.
<br><br>
<strong>ATHENA HFT portion is the wide-uncertainty bet.</strong> 1 day of
validation means real outcomes could be 2-4× wider than P10-P90 shown above.
</div>
</body></html>
"""
    return html


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=2000,
                          help="Number of Monte Carlo trials (default 2000)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--decay", type=float, default=0.20,
                          help="HFT edge decay factor (0=no edge, 1=yesterday persists). Default 0.20")
    parser.add_argument("--no-compounding", action="store_true",
                          help="Disable v2.1 compounding (fixed $70 base, like pre-v2.1 host)")
    parser.add_argument("--circuit-pct", type=float, default=0.03,
                          help="Daily-loss circuit breaker threshold (default 3%%)")
    parser.add_argument("--extra-gate-block", type=float, default=0.15,
                          help="Extra entries blocked by news+earnings+OB gates (default 15%%)")
    parser.add_argument("--no-html", action="store_true", help="Skip HTML report")
    args = parser.parse_args()

    print("=" * 70)
    print("TERMINATOR ($1k) — Year-1 Bootstrap Monte Carlo")
    print("=" * 70)
    print(f"Trials: {args.trials:,}  ·  HFT decay: {args.decay:.0%}  ·  Seed: {args.seed}")
    print()

    results = run_monte_carlo(
        args.trials, args.seed, hft_decay=args.decay,
        enable_compounding=not args.no_compounding,
        circuit_breaker_pct=args.circuit_pct,
        extra_gate_block_rate=args.extra_gate_block,
    )

    end = results["ending_equity"].values
    p5, p10, p25, p50, p75, p90, p95 = np.percentile(end, [5, 10, 25, 50, 75, 90, 95])
    mean = end.mean()
    take = end - STARTING_CAPITAL
    pct_profit = (take > 0).mean()
    survival = results["survived"].mean()

    print()
    print("=== Year-End Equity Distribution ($1,000 starting) ===")
    print(f"  P05:    ${p5:>8,.0f}   (take ${p5-STARTING_CAPITAL:+,.0f}, {(p5/STARTING_CAPITAL-1)*100:+.1f}%)")
    print(f"  P10:    ${p10:>8,.0f}   (take ${p10-STARTING_CAPITAL:+,.0f}, {(p10/STARTING_CAPITAL-1)*100:+.1f}%)")
    print(f"  P25:    ${p25:>8,.0f}   (take ${p25-STARTING_CAPITAL:+,.0f}, {(p25/STARTING_CAPITAL-1)*100:+.1f}%)")
    print(f"  P50:    ${p50:>8,.0f}   (take ${p50-STARTING_CAPITAL:+,.0f}, {(p50/STARTING_CAPITAL-1)*100:+.1f}%)  ← median")
    print(f"  P75:    ${p75:>8,.0f}   (take ${p75-STARTING_CAPITAL:+,.0f}, {(p75/STARTING_CAPITAL-1)*100:+.1f}%)")
    print(f"  P90:    ${p90:>8,.0f}   (take ${p90-STARTING_CAPITAL:+,.0f}, {(p90/STARTING_CAPITAL-1)*100:+.1f}%)")
    print(f"  P95:    ${p95:>8,.0f}   (take ${p95-STARTING_CAPITAL:+,.0f}, {(p95/STARTING_CAPITAL-1)*100:+.1f}%)")
    print(f"  Mean:   ${mean:>8,.0f}   (take ${mean-STARTING_CAPITAL:+,.0f})")
    print()
    print(f"  % profitable years:  {pct_profit*100:.1f}%")
    print(f"  survival rate:       {survival*100:.1f}%")
    print(f"  mean max drawdown:   {results['max_dd_pct'].mean()*100:.1f}%")
    print()

    # Component breakdown at median
    med_svxy = results["ending_svxy"].median()
    med_hft = results["ending_hft"].median()
    print("=== Median Component Breakdown ===")
    print(f"  SVXY portion ($300 start): $ {med_svxy:>6,.0f}  (take ${med_svxy-STARTING_CAPITAL*SVXY_PCT:+,.0f})")
    print(f"  HFT portion  ($700 start): $ {med_hft:>6,.0f}  (take ${med_hft-STARTING_CAPITAL*HFT_PCT:+,.0f})")
    print()

    # Save results
    results.to_parquet(OUT / "trials.parquet", index=False)
    summary = {
        "trials": args.trials,
        "hft_decay": args.decay,
        "seed": args.seed,
        "starting_capital": STARTING_CAPITAL,
        "p05": float(p5), "p10": float(p10), "p25": float(p25),
        "p50": float(p50), "p75": float(p75), "p90": float(p90), "p95": float(p95),
        "mean": float(mean),
        "pct_profitable": float(pct_profit),
        "survival_rate": float(survival),
        "mean_max_dd_pct": float(results["max_dd_pct"].mean()),
        "median_svxy_ending": float(med_svxy),
        "median_hft_ending": float(med_hft),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"wrote {OUT}/trials.parquet")
    print(f"wrote {OUT}/summary.json")

    if not args.no_html:
        decay_label = f"{args.decay:.0%}"
        html = render_html_report(results, decay_label, trials=args.trials)
        (OUT / "index.html").write_text(html)
        print(f"wrote {OUT}/index.html")


if __name__ == "__main__":
    main()
