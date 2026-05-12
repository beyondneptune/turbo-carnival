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
) -> tuple[float, int, np.ndarray, int]:
    """Simulate one year of ATHENA HFT lane on an account starting with `starting`.

    Returns (ending_equity, n_circuit_breaker_days, daily_curve, total_entries).
    """
    equity = starting
    starting_eq = starting
    peak = starting
    n_circuit = 0
    total_entries = 0
    curve = np.zeros(TRADING_DAYS + 1)
    curve[0] = starting
    for d in range(TRADING_DAYS):
        K = int(np.clip(
            rng.normal(ENTRIES_PER_DAY_MEAN, ENTRIES_PER_DAY_MEAN * 0.3),
            15, 50,
        ))
        if extra_gate_block_rate > 0:
            K = int((rng.random(K) > extra_gate_block_rate).sum())

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

        session_start = equity
        day_pnl = 0.0
        entries_taken = 0
        if K > 0:
            sampled = rng.choice(pnl_per_dollar, size=K, replace=True)
            for entry_pnl_per_dollar in sampled:
                entry_pnl = entry_pnl_per_dollar * base_notional * decay
                day_pnl += entry_pnl
                entries_taken += 1
                if -day_pnl / session_start > CIRCUIT_PCT:
                    n_circuit += 1
                    break

        day_pnl = max(day_pnl, -equity)
        equity += day_pnl
        peak = max(peak, equity)
        total_entries += entries_taken
        curve[d + 1] = equity
        if equity < 50:
            curve[d + 1:] = equity
            break
    return equity, n_circuit, curve, total_entries


def simulate_svxy(starting: float, rng: np.random.Generator) -> tuple[float, np.ndarray, int]:
    """SVXY lognormal walk. Returns (ending, curve, n_rebalances)."""
    equity = starting
    curve = np.zeros(TRADING_DAYS + 1)
    curve[0] = starting
    # SVXY rotation rebalances at most once a day, but in practice ~12-15/yr
    # (regime flips). Simplify: 1 trade per 20 days on average.
    n_rebalances = max(1, TRADING_DAYS // 20)
    for d in range(TRADING_DAYS):
        ret = rng.normal(SVXY_DAILY_MU, SVXY_DAILY_SIGMA)
        equity *= (1 + ret)
        curve[d + 1] = equity
        if equity < 50:
            curve[d + 1:] = equity
            break
    return equity, curve, n_rebalances


def simulate_condor(starting: float, rng: np.random.Generator) -> tuple[float, np.ndarray, int]:
    """Monthly iron condor cycles. Returns (ending, curve, n_cycles)."""
    equity = starting
    curve = np.zeros(TRADING_DAYS + 1)
    curve[0] = starting
    cycle_length = TRADING_DAYS // int(CONDOR_CYCLES_PER_YEAR)   # ~20 days
    cycles = 0
    for d in range(TRADING_DAYS):
        if d > 0 and d % cycle_length == 0:
            if rng.random() < CONDOR_WIN_PROB:
                equity *= (1 + CONDOR_WIN_RETURN)
            else:
                equity *= (1 + CONDOR_LOSS_RETURN)
            cycles += 1
        curve[d + 1] = equity
    return equity, curve, cycles


def simulate_flat(starting: float) -> tuple[float, np.ndarray]:
    """Atlas: cash interest only — smooth growth."""
    daily_rate = (1 + CASH_ANNUAL_RFR) ** (1 / TRADING_DAYS) - 1
    curve = starting * np.cumprod(np.ones(TRADING_DAYS + 1) * (1 + daily_rate))
    curve[0] = starting
    return float(curve[-1]), curve


def simulate_one_year_all_accounts(
    pnl_per_dollar: np.ndarray, rng: np.random.Generator,
    *, decay: float, extra_gate_block_rate: float,
) -> dict:
    """Simulate all 6 accounts for one trial year.

    Returns dict per account with: end, curve (252+1 array), total_entries, circuit_days
    """
    results = {}
    for name, params in ACCOUNTS.items():
        start = params["start"]
        strategy = params["strategy"]
        if strategy == "athena_hft":
            end, n_circ, curve, n_entries = simulate_athena_hft(
                pnl_per_dollar, rng,
                starting=start,
                base_per_entry=HFT_BASE_PER_ENTRY.get(name, 12_000.0),
                decay=decay, extra_gate_block_rate=extra_gate_block_rate,
            )
            results[name] = {"end": end, "curve": curve,
                              "total_entries": n_entries, "circuit_days": n_circ}
        elif strategy == "svxy_rotation":
            end, curve, n_trades = simulate_svxy(start, rng)
            results[name] = {"end": end, "curve": curve,
                              "total_entries": n_trades, "circuit_days": 0}
        elif strategy == "iron_condor":
            end, curve, n_cycles = simulate_condor(start, rng)
            results[name] = {"end": end, "curve": curve,
                              "total_entries": n_cycles, "circuit_days": 0}
        elif strategy == "flat":
            end, curve = simulate_flat(start)
            results[name] = {"end": end, "curve": curve,
                              "total_entries": 0, "circuit_days": 0}
        elif strategy == "hybrid":
            hft_start = start * params["hft_pct"]
            svxy_start = start * params["svxy_pct"]
            hft_end, n_circ, hft_curve, n_entries = simulate_athena_hft(
                pnl_per_dollar, rng,
                starting=hft_start, base_per_entry=HFT_BASE_PER_ENTRY[name],
                decay=decay, extra_gate_block_rate=extra_gate_block_rate,
            )
            svxy_end, svxy_curve, n_svxy_trades = simulate_svxy(svxy_start, rng)
            results[name] = {"end": hft_end + svxy_end,
                              "curve": hft_curve + svxy_curve,
                              "hft_end": hft_end, "svxy_end": svxy_end,
                              "total_entries": n_entries + n_svxy_trades,
                              "circuit_days": n_circ}
    return results


def run_monte_carlo(trials: int, seed: int, decay: float, extra_block: float) -> tuple[pd.DataFrame, dict]:
    """Returns (per-trial DataFrame, dict of percentile curves + entry counts)."""
    rng = np.random.default_rng(seed)
    pnl_per_dollar = load_hft_pnl_distribution()
    print(f"loaded {len(pnl_per_dollar)} HFT entries (per-$ mean {pnl_per_dollar.mean()*100:+.3f}%)")
    print(f"running {trials} trials · decay {decay:.0%} · extra block {extra_block:.0%}")

    rows = []
    # Track all curves per account for percentile aggregation
    curves_per_account: dict[str, list[np.ndarray]] = {name: [] for name in ACCOUNTS}
    entries_per_account: dict[str, list[int]] = {name: [] for name in ACCOUNTS}
    for t in range(trials):
        r = simulate_one_year_all_accounts(
            pnl_per_dollar, np.random.default_rng(rng.integers(0, 2**31)),
            decay=decay, extra_gate_block_rate=extra_block,
        )
        row = {"trial": t}
        for name, acct in r.items():
            row[f"{name}_end"] = acct["end"]
            row[f"{name}_entries"] = acct["total_entries"]
            curves_per_account[name].append(acct["curve"])
            entries_per_account[name].append(acct["total_entries"])
        rows.append(row)

    # Compute percentile bands per day for each account
    bands = {}
    for name in ACCOUNTS:
        arr = np.array(curves_per_account[name])   # (trials, 253)
        bands[name] = {
            "p10": np.percentile(arr, 10, axis=0),
            "p25": np.percentile(arr, 25, axis=0),
            "p50": np.percentile(arr, 50, axis=0),
            "p75": np.percentile(arr, 75, axis=0),
            "p90": np.percentile(arr, 90, axis=0),
            "mean_entries_per_year": np.mean(entries_per_account[name]),
        }
    return pd.DataFrame(rows), bands


def render_charts_html(all_results: dict, all_bands: dict, yesterday_entries: pd.DataFrame | None = None) -> None:
    """Render interactive plotly HTML with charts + yesterday's trades."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.io as pio

    SCENARIO_COLORS = {
        "Bearish (5% decay)": "#dc2626",
        "Central (20% decay)": "#2563eb",
        "Optimistic (40% decay)": "#16a34a",
        "Edge persists (100%)": "#9333ea",
    }

    # ---- 1. Portfolio total distribution (overlay 4 scenarios) ----
    fig_portfolio = go.Figure()
    for label, df in all_results.items():
        totals = sum(df[f"{name}_end"] for name in ACCOUNTS.keys())
        fig_portfolio.add_trace(go.Histogram(
            x=totals, name=label, opacity=0.6, nbinsx=60,
            marker_color=SCENARIO_COLORS.get(label, "#666"),
        ))
    total_start = sum(p["start"] for p in ACCOUNTS.values())
    fig_portfolio.add_vline(x=total_start, line_dash="dash", line_color="black",
                              annotation_text=f"Start ${total_start:,.0f}")
    fig_portfolio.update_layout(
        barmode="overlay",
        title=f"Total Portfolio Year-End Distribution · {len(next(iter(all_results.values())))} trials/scenario",
        xaxis_title="Portfolio Equity ($)", yaxis_title="Trial count",
        height=420, template="plotly_white",
    )

    # ---- 2. Per-account box plots × 4 scenarios ----
    fig_accounts = make_subplots(rows=2, cols=3,
                                   subplot_titles=list(ACCOUNTS.keys()))
    positions = [(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    for (name, params), (r, c) in zip(ACCOUNTS.items(), positions):
        for label, df in all_results.items():
            fig_accounts.add_trace(
                go.Box(y=df[f"{name}_end"], name=label.split(" ")[0],
                       marker_color=SCENARIO_COLORS.get(label, "#666"),
                       showlegend=(r==1 and c==1),
                       boxmean="sd"),
                row=r, col=c,
            )
        # Add starting capital line
        fig_accounts.add_hline(y=params["start"], line_dash="dot",
                                 line_color="black", line_width=1,
                                 row=r, col=c)
    fig_accounts.update_layout(
        title="Per-Account Year-End Distribution by Scenario",
        height=620, template="plotly_white",
    )

    # ---- 3. Median take by account × scenario (grouped bar) ----
    fig_takes = go.Figure()
    accounts = list(ACCOUNTS.keys())
    for label, df in all_results.items():
        medians = [df[f"{name}_end"].median() - ACCOUNTS[name]["start"] for name in accounts]
        fig_takes.add_trace(go.Bar(
            x=accounts, y=medians, name=label,
            marker_color=SCENARIO_COLORS.get(label, "#666"),
            text=[f"${v:+,.0f}" for v in medians], textposition="outside",
        ))
    fig_takes.update_layout(
        title="Median Yearly Take by Account × Scenario",
        xaxis_title="Account", yaxis_title="Take ($)",
        barmode="group", height=480, template="plotly_white",
    )

    # ---- 4. Yesterday's actual trades (if available) ----
    fig_trades = None
    if yesterday_entries is not None and len(yesterday_entries) > 0:
        df = yesterday_entries.copy()
        df = df.sort_values("entry_ts").reset_index(drop=True)
        df["pnl_cum"] = df["pnl"].cumsum()
        fig_trades = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            subplot_titles=("Cumulative P&L over yesterday's session",
                              "Per-entry P&L (blocked entries shown faded)"),
            vertical_spacing=0.10, row_heights=[0.5, 0.5],
        )
        # Cumulative curve — only kept entries
        kept = df[~df.get("blocked", False)] if "blocked" in df.columns else df
        kept_cum = kept["pnl"].cumsum().values
        fig_trades.add_trace(
            go.Scatter(x=list(range(len(kept))), y=kept_cum, mode="lines+markers",
                       name="Cumulative P&L (C3-kept)",
                       line=dict(color="#16a34a", width=2)),
            row=1, col=1,
        )
        # Per-entry bars
        colors = ["#9ca3af" if b else ("#16a34a" if p > 0 else "#dc2626" if p < 0 else "#9ca3af")
                  for p, b in zip(df["pnl"], df.get("blocked", [False]*len(df)))]
        hover = [f"{lane} · {sym} · {side}<br>P&L: ${p:+.2f}<br>blocked: {b}"
                 for lane, sym, side, p, b in zip(
                     df["lane"], df["symbol"], df["side_dir"], df["pnl"],
                     df.get("blocked", [False]*len(df)))]
        fig_trades.add_trace(
            go.Bar(x=list(range(len(df))), y=df["pnl"], marker_color=colors,
                   text=df["symbol"], hovertext=hover, name="Per-entry P&L"),
            row=2, col=1,
        )
        fig_trades.update_layout(
            title="Yesterday's 162 ATHENA Entries (C3 gate blocked 82 in grey)",
            height=560, template="plotly_white", showlegend=False,
        )
        fig_trades.update_xaxes(title_text="Entry index (time-ordered)", row=2, col=1)
        fig_trades.update_yaxes(title_text="Cumulative P&L ($)", row=1, col=1)
        fig_trades.update_yaxes(title_text="P&L per entry ($)", row=2, col=1)

    # ---- Build summary table ----
    rows_html = ""
    for name, params in ACCOUNTS.items():
        start = params["start"]
        rows_html += f"<tr><td><b>{name}</b></td><td>${start:,.0f}</td>"
        for label, df in all_results.items():
            ends = df[f"{name}_end"]
            median = ends.median()
            take = median - start
            pct = (median / start - 1) * 100
            color = "#16a34a" if take > 0 else "#dc2626"
            rows_html += (f'<td style="color:{color}">'
                          f'${median:,.0f}<br><small>{take:+,.0f} ({pct:+.1f}%)</small></td>')
        rows_html += "</tr>"
    # Total row
    total_start = sum(p["start"] for p in ACCOUNTS.values())
    rows_html += f'<tr style="background:#f3f4f6;font-weight:bold"><td>TOTAL</td><td>${total_start:,.0f}</td>'
    for label, df in all_results.items():
        totals = sum(df[f"{name}_end"] for name in ACCOUNTS.keys())
        med = totals.median()
        take = med - total_start
        pct = (med / total_start - 1) * 100
        color = "#16a34a" if take > 0 else "#dc2626"
        rows_html += (f'<td style="color:{color}">${med:,.0f}<br>'
                      f'<small>{take:+,.0f} ({pct:+.1f}%)</small></td>')
    rows_html += "</tr>"

    # ---- Return-cadence table (central case) ----
    cadence_rows = ""
    central_df = all_results.get("Central (20% decay)")
    central_bands = all_bands.get("Central (20% decay)", {})
    if central_df is not None and central_bands:
        for name, params in ACCOUNTS.items():
            start = params["start"]
            ends = central_df[f"{name}_end"]
            median_end = ends.median()
            total_return = (median_end / start - 1)
            mean_entries = central_bands[name].get("mean_entries_per_year", 0)
            # Compute periodic returns from total
            daily_geometric = (1 + total_return) ** (1 / TRADING_DAYS) - 1
            weekly = (1 + total_return) ** (5 / TRADING_DAYS) - 1
            monthly = (1 + total_return) ** (21 / TRADING_DAYS) - 1
            trades_per_day = mean_entries / TRADING_DAYS
            strategy_label = params["strategy"]
            cadence_rows += (
                f"<tr><td><b>{name}</b></td><td>{strategy_label}</td>"
                f"<td>{daily_geometric*100:+.3f}%</td>"
                f"<td>{weekly*100:+.2f}%</td>"
                f"<td>{monthly*100:+.2f}%</td>"
                f"<td>{total_return*100:+.1f}%</td>"
                f"<td>{trades_per_day:.1f}</td>"
                f"<td>{mean_entries:.0f}</td>"
                f"</tr>"
            )

    # ---- 5. Timeline equity-curve fan charts (central scenario) ----
    central_label = "Central (20% decay)"
    bands = all_bands.get(central_label, {})
    fig_timeline = make_subplots(rows=2, cols=3, subplot_titles=list(ACCOUNTS.keys()))
    for (name, _), (r, c) in zip(ACCOUNTS.items(), positions):
        b = bands.get(name, {})
        if not b:
            continue
        x = list(range(len(b["p50"])))
        # P10-P90 fill
        fig_timeline.add_trace(
            go.Scatter(x=x, y=b["p90"], mode="lines", line=dict(width=0), showlegend=False),
            row=r, col=c,
        )
        fig_timeline.add_trace(
            go.Scatter(x=x, y=b["p10"], mode="lines", line=dict(width=0),
                       fill="tonexty", fillcolor="rgba(37,99,235,0.15)",
                       name="P10-P90", showlegend=(r==1 and c==1)),
            row=r, col=c,
        )
        # P25-P75 fill (darker)
        fig_timeline.add_trace(
            go.Scatter(x=x, y=b["p75"], mode="lines", line=dict(width=0), showlegend=False),
            row=r, col=c,
        )
        fig_timeline.add_trace(
            go.Scatter(x=x, y=b["p25"], mode="lines", line=dict(width=0),
                       fill="tonexty", fillcolor="rgba(37,99,235,0.35)",
                       name="P25-P75", showlegend=(r==1 and c==1)),
            row=r, col=c,
        )
        # Median line
        fig_timeline.add_trace(
            go.Scatter(x=x, y=b["p50"], mode="lines",
                       line=dict(color="#1d4ed8", width=2),
                       name="Median", showlegend=(r==1 and c==1)),
            row=r, col=c,
        )
    fig_timeline.update_layout(
        title=f"Equity Timeline Through 252 Trading Days · {central_label}",
        height=620, template="plotly_white",
    )
    fig_timeline.update_xaxes(title_text="Trading day")

    # Convert plotly figures to HTML divs
    portfolio_div = pio.to_html(fig_portfolio, full_html=False, include_plotlyjs="cdn")
    accounts_div = pio.to_html(fig_accounts, full_html=False, include_plotlyjs=False)
    takes_div = pio.to_html(fig_takes, full_html=False, include_plotlyjs=False)
    timeline_div = pio.to_html(fig_timeline, full_html=False, include_plotlyjs=False)
    trades_div = pio.to_html(fig_trades, full_html=False, include_plotlyjs=False) if fig_trades else ""

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>6-Account Year-1 Simulation</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; max-width: 1280px;
       margin: 1.5rem auto; padding: 0 1.5rem; color: #1f2937; }}
h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; }}
h2 {{ font-size: 1.2rem; margin-top: 2rem; border-bottom: 1px solid #e5e7eb; padding-bottom: 0.3rem; }}
.subtitle {{ color: #6b7280; margin-bottom: 1.5rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: 0.9rem; }}
th, td {{ text-align: right; padding: 0.5rem 0.6rem; border-bottom: 1px solid #e5e7eb; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #f3f4f6; font-weight: 600; }}
.note {{ background: #fef3c7; border-left: 3px solid #f59e0b;
        padding: 0.7rem 1rem; margin: 1rem 0; font-size: 0.88rem; }}
</style></head><body>
<h1>6-Account Year-1 Monte Carlo · v2.1 Deployed Stack</h1>
<div class="subtitle">5,000 trials per scenario · live capital snapshot 2026-05-12 · interactive plotly charts</div>

<h2>Summary Table</h2>
<table>
<tr><th>Account</th><th>Start</th><th>Bearish</th><th>Central</th><th>Optimistic</th><th>Edge persists</th></tr>
{rows_html}
</table>

<h2>Return Cadence & Trade Frequency (Central case median)</h2>
<table>
<tr><th>Account</th><th>Strategy</th><th>Avg Daily</th><th>Avg Weekly</th><th>Avg Monthly</th>
<th>Annual</th><th>Trades/day</th><th>Trades/yr</th></tr>
{cadence_rows}
</table>

<h2>Equity Timeline · Central case · P10/P25/P50/P75/P90 fan</h2>
{timeline_div}

<h2>Total Portfolio Distribution</h2>
{portfolio_div}

<h2>Per-Account Year-End Distribution</h2>
{accounts_div}

<h2>Median Yearly Take by Account × Scenario</h2>
{takes_div}

<div class="note">
<b>What the charts show:</b> bearish (red) assumes only 5% of yesterday's HFT edge
persists — most of the upside comes from Valk's iron condor + Hermes's SVXY rotation
(both independent of HFT decay). Central (blue, 20% decay) is my honest baseline.
Edge-persists (purple, 100%) is what happens if yesterday's tape is representative —
Aegis and Falcon compound until they hit the 5× capacity cap (~$500k base).
</div>

<h2>Yesterday's 162 ATHENA Entries (the data behind the bootstrap)</h2>
{trades_div if trades_div else '<p style="color:#9ca3af">No trade data available.</p>'}

<div class="note" style="background:#dbeafe;border-color:#3b82f6">
<b>What this is NOT:</b> a deterministic backtest. It bootstraps from 1 day (2026-05-11)
of validated entries to project a range of outcomes. Real outcomes depend entirely on
whether tomorrow's tape resembles yesterday's. First real signal comes at today's close.
</div>

</body></html>"""
    out_path = OUT / "index.html"
    out_path.write_text(html)


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

    # Always run all 4 scenarios for the HTML chart (cheap)
    all_results = {}
    all_bands = {}
    for label, decay in [("Bearish (5% decay)", 0.05),
                          ("Central (20% decay)", 0.20),
                          ("Optimistic (40% decay)", 0.40),
                          ("Edge persists (100%)", 1.0)]:
        df, bands = run_monte_carlo(args.trials, args.seed, decay, args.extra_block)
        report(df, label)
        all_results[label] = df
        all_bands[label] = bands

    # Load yesterday's actual trade tape for the trades chart
    yesterday_entries = pd.read_parquet(COMPOSITE_PER_ENTRY)
    render_charts_html(all_results, all_bands, yesterday_entries=yesterday_entries)
    # Save the central case parquet
    all_results["Central (20% decay)"].to_parquet(OUT / "trials.parquet", index=False)
    print(f"wrote {OUT}/trials.parquet")
    print(f"wrote {OUT}/index.html")
    import subprocess
    subprocess.run(["open", str(OUT / "index.html")])


if __name__ == "__main__":
    main()
