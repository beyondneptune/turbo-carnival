"""6-account year-1 Monte Carlo simulation — v2.1 deployed stack.

Simulates all 6 paper accounts in parallel with the strategies actually
deployed today (2026-05-12):

  - Aegis      $99,286  ATHENA v2.1 HFT mean-rev          (bootstrap)
  - Falcon     $97,701  ATHENA v2.1 HFT breakout          (bootstrap)
  - Valk       $99,552  30-DTE SPY iron condor            (parametric)
  - Hermes    $100,000  SVXY regime rotation              (validated 7-yr)
  - Atlas     $100,000  flat (no strategy)                (cash interest only)
  - TERMINATOR $1,000   70% ATHENA HFT + 30% SVXY         (bootstrap + lognormal)

Models the v2.1 deployed safety + sizing stack on the HFT lanes:
  - Compounding: base_notional × (equity / session_start), capped at 5×
  - Drawdown gate: revert to baseline if >20% off peak
  - Circuit breaker: halt entries if intraday loss > 3%
  - Extra gates (news+earnings+OB imbalance): block ~15% extra entries

Run:
    python experiments/ml_lab/26_six_account_yearly_sim.py --trials 5000
    python experiments/ml_lab/26_six_account_yearly_sim.py --decay 1.0   # full edge persistence
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
COMPOSITE_PER_ENTRY = REPO / "experiments" / "ml_lab" / "runs" / "composite_backtest" / "winner_per_entry.parquet"
OUT = REPO / "reports" / "six_account_yearly"
OUT.mkdir(parents=True, exist_ok=True)

TRADING_DAYS = 252

# Live starting capital snapshot (pulled at 13:30 UTC today)
ACCOUNTS = {
    "Aegis":      {"start": 99_285.83, "strategy": "athena_hft", "hft_pct": 1.0, "svxy_pct": 0.0},
    "Falcon":     {"start": 97_700.58, "strategy": "athena_hft", "hft_pct": 1.0, "svxy_pct": 0.0},
    "Valk":       {"start": 99_551.50, "strategy": "iron_condor", "hft_pct": 0.0, "svxy_pct": 0.0},
    "Hermes":     {"start": 100_000.0, "strategy": "svxy_rotation", "hft_pct": 0.0, "svxy_pct": 1.0},
    "Atlas":      {"start": 100_000.0, "strategy": "flat", "hft_pct": 0.0, "svxy_pct": 0.0},
    "TERMINATOR": {"start": 1_000.0, "strategy": "hybrid", "hft_pct": 0.70, "svxy_pct": 0.30},
}

# === HFT edge (from yesterday's 80 C3-kept entries, sized at $20k notional) ===
YESTERDAY_TRADE_SIZE = 20_000.0
HFT_BASE_PER_ENTRY = {  # default base from host code
    "Aegis": 12_000.0,
    "Falcon": 12_000.0,
    "TERMINATOR": 70.0,
}
ENTRIES_PER_DAY_MEAN = 30

# v2.1 sizing
CAPACITY_MULT_CAP = 5.0
DD_REVERT_PCT = 0.20
CIRCUIT_PCT = 0.03

# === SVXY rotation params (7-yr validated walk-forward) ===
SVXY_ANNUAL_MU = 0.171
SVXY_SHARPE = 0.66
SVXY_ANNUAL_SIGMA = SVXY_ANNUAL_MU / SVXY_SHARPE
SVXY_DAILY_MU = SVXY_ANNUAL_MU / TRADING_DAYS
SVXY_DAILY_SIGMA = SVXY_ANNUAL_SIGMA / np.sqrt(TRADING_DAYS)

# === Iron condor params (validated edge for 30-DTE SPY condor at our scale) ===
CONDOR_CYCLES_PER_YEAR = 12.6   # 252 days / 20 trading days
CONDOR_WIN_PROB = 0.80           # full premium kept
CONDOR_WIN_RETURN = 0.04         # +4% on margin per win
CONDOR_LOSS_RETURN = -0.12       # -12% on margin per loss (stop-out)
# Expected per cycle: 0.80×4% + 0.20×(-12%) = 0.8% → +10% annualized
# Yesterday's Valk was -$440 unrealized in the current cycle — overall slight drag

# === Atlas cash interest (rfr) ===
CASH_ANNUAL_RFR = 0.045   # 4.5% money market rate


def load_hft_pnl_distribution() -> np.ndarray:
    """Per-dollar P&L from yesterday's 80 C3-kept entries."""
    df = pd.read_parquet(COMPOSITE_PER_ENTRY)
    kept = df[~df["blocked"]]
    return kept["pnl"].values / YESTERDAY_TRADE_SIZE


def simulate_athena_hft(
    pnl_per_dollar: np.ndarray, rng: np.random.Generator,
    *, starting: float, base_per_entry: float,
    decay: float, extra_gate_block_rate: float,
    enable_compounding: bool = True,
) -> tuple[float, int]:
    """Simulate one year of ATHENA HFT lane on an account starting with `starting`.

    Returns (ending_equity, n_circuit_breaker_days).
    """
    equity = starting
    starting_eq = starting
    peak = starting
    n_circuit = 0
    for d in range(TRADING_DAYS):
        # Entries per day
        K = int(np.clip(
            rng.normal(ENTRIES_PER_DAY_MEAN, ENTRIES_PER_DAY_MEAN * 0.3),
            15, 50,
        ))
        if extra_gate_block_rate > 0:
            K = int((rng.random(K) > extra_gate_block_rate).sum())

        # Compounding multiplier
        if enable_compounding:
            dd_pct = (peak - equity) / peak if peak > 0 else 0
            if dd_pct > DD_REVERT_PCT:
                eq_mult = 1.0
            else:
                eq_mult = min(equity / starting_eq if starting_eq > 0 else 1.0,
                              CAPACITY_MULT_CAP)
        else:
            eq_mult = 1.0
        base_notional = base_per_entry * eq_mult

        # Bootstrap K entries with circuit breaker
        session_start = equity
        day_pnl = 0.0
        if K > 0:
            sampled = rng.choice(pnl_per_dollar, size=K, replace=True)
            for entry_pnl_per_dollar in sampled:
                entry_pnl = entry_pnl_per_dollar * base_notional * decay
                day_pnl += entry_pnl
                if -day_pnl / session_start > CIRCUIT_PCT:
                    n_circuit += 1
                    break

        day_pnl = max(day_pnl, -equity)
        equity += day_pnl
        peak = max(peak, equity)
        if equity < 50:
            break
    return equity, n_circuit


def simulate_svxy(starting: float, rng: np.random.Generator) -> float:
    """SVXY lognormal walk."""
    equity = starting
    for _ in range(TRADING_DAYS):
        ret = rng.normal(SVXY_DAILY_MU, SVXY_DAILY_SIGMA)
        equity *= (1 + ret)
        if equity < 50:
            break
    return equity


def simulate_condor(starting: float, rng: np.random.Generator) -> float:
    """Monthly iron condor cycles."""
    equity = starting
    for _ in range(int(CONDOR_CYCLES_PER_YEAR)):
        if rng.random() < CONDOR_WIN_PROB:
            equity *= (1 + CONDOR_WIN_RETURN)
        else:
            equity *= (1 + CONDOR_LOSS_RETURN)
    return equity


def simulate_flat(starting: float) -> float:
    """Atlas: cash interest only."""
    return starting * (1 + CASH_ANNUAL_RFR)


def simulate_one_year_all_accounts(
    pnl_per_dollar: np.ndarray, rng: np.random.Generator,
    *, decay: float, extra_gate_block_rate: float,
) -> dict:
    """Simulate all 6 accounts for one trial year."""
    results = {}
    for name, params in ACCOUNTS.items():
        start = params["start"]
        strategy = params["strategy"]
        if strategy == "athena_hft":
            end, n_circ = simulate_athena_hft(
                pnl_per_dollar, rng,
                starting=start,
                base_per_entry=HFT_BASE_PER_ENTRY.get(name, 12_000.0),
                decay=decay, extra_gate_block_rate=extra_gate_block_rate,
            )
            results[name] = {"end": end, "circuit_days": n_circ}
        elif strategy == "svxy_rotation":
            end = simulate_svxy(start, rng)
            results[name] = {"end": end, "circuit_days": 0}
        elif strategy == "iron_condor":
            end = simulate_condor(start, rng)
            results[name] = {"end": end, "circuit_days": 0}
        elif strategy == "flat":
            end = simulate_flat(start)
            results[name] = {"end": end, "circuit_days": 0}
        elif strategy == "hybrid":
            hft_start = start * params["hft_pct"]
            svxy_start = start * params["svxy_pct"]
            hft_end, n_circ = simulate_athena_hft(
                pnl_per_dollar, rng,
                starting=hft_start, base_per_entry=HFT_BASE_PER_ENTRY[name],
                decay=decay, extra_gate_block_rate=extra_gate_block_rate,
            )
            svxy_end = simulate_svxy(svxy_start, rng)
            results[name] = {"end": hft_end + svxy_end,
                              "hft_end": hft_end, "svxy_end": svxy_end,
                              "circuit_days": n_circ}
    return results


def run_monte_carlo(trials: int, seed: int, decay: float, extra_block: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pnl_per_dollar = load_hft_pnl_distribution()
    print(f"loaded {len(pnl_per_dollar)} HFT entries (per-$ mean {pnl_per_dollar.mean()*100:+.3f}%)")
    print(f"running {trials} trials · decay {decay:.0%} · extra block {extra_block:.0%}")

    rows = []
    for t in range(trials):
        r = simulate_one_year_all_accounts(
            pnl_per_dollar, np.random.default_rng(rng.integers(0, 2**31)),
            decay=decay, extra_gate_block_rate=extra_block,
        )
        row = {"trial": t}
        for name, acct in r.items():
            row[f"{name}_end"] = acct["end"]
        rows.append(row)
    return pd.DataFrame(rows)


def report(df: pd.DataFrame, label: str) -> None:
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")
    print()
    print(f"{'Account':<12} {'Start':>10} {'P10':>10} {'Median':>10} {'P90':>10} {'Take(med)':>11} {'%Profit':>8}")
    print("-" * 80)
    total_start = 0
    total_p10 = 0
    total_p50 = 0
    total_p90 = 0
    total_mean = 0
    for name, params in ACCOUNTS.items():
        col = f"{name}_end"
        start = params["start"]
        total_start += start
        ends = df[col].values
        p10, p50, p90 = np.percentile(ends, [10, 50, 90])
        mean = ends.mean()
        take = p50 - start
        pct_prof = (ends > start).mean() * 100
        total_p10 += p10
        total_p50 += p50
        total_p90 += p90
        total_mean += mean
        print(f"  {name:<10} {start:>10,.0f} {p10:>10,.0f} {p50:>10,.0f} {p90:>10,.0f} "
              f"{take:>+11,.0f} {pct_prof:>7.1f}%")
    print("-" * 80)
    total_take = total_p50 - total_start
    total_pct = (total_p50 / total_start - 1) * 100
    print(f"  {'TOTAL':<10} {total_start:>10,.0f} {total_p10:>10,.0f} {total_p50:>10,.0f} "
          f"{total_p90:>10,.0f} {total_take:>+11,.0f} ({total_pct:+.2f}%)")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--decay", type=float, default=0.20,
                          help="HFT edge decay (1.0 = yesterday's edge fully holds)")
    parser.add_argument("--extra-block", type=float, default=0.15)
    parser.add_argument("--all-scenarios", action="store_true",
                          help="Run bearish/central/optimistic side-by-side")
    args = parser.parse_args()

    if args.all_scenarios:
        for label, decay in [("BEARISH (5% decay)", 0.05),
                              ("CENTRAL (20% decay)", 0.20),
                              ("OPTIMISTIC (40% decay)", 0.40),
                              ("EDGE PERSISTS (100% decay)", 1.0)]:
            df = run_monte_carlo(args.trials, args.seed, decay, args.extra_block)
            report(df, label)
    else:
        df = run_monte_carlo(args.trials, args.seed, args.decay, args.extra_block)
        decay_label = "edge persists" if args.decay == 1.0 else f"{args.decay*100:.0f}% decay"
        report(df, f"6-ACCOUNT YEAR-1 SIM · {decay_label}")
        df.to_parquet(OUT / "trials.parquet", index=False)
        print(f"wrote {OUT}/trials.parquet")


if __name__ == "__main__":
    main()
