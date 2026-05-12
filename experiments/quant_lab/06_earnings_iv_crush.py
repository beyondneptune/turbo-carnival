"""Earnings IV crush 2024-2026.

For each earnings event in our chain data:
  - At ENTRY (day before announcement): SELL the ATM straddle (-call - put)
  - At EXIT (after announcement, when IV has crushed): BUY BACK the straddle
  - P&L per share = (entry straddle premium) - (exit straddle premium)

Pay attention to: tickers where the realized move > implied move blow up the
short straddle. Net result averages across many events should be positive
if IV crush > realized move on average.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
OUT = REPO / "experiments" / "quant_lab" / "runs"
CHAIN_PATH = REPO / "data" / "final_dataset" / "priced_chain_legs.parquet"
STARTING_CAPITAL = 100_000.0
STRADDLE_NOTIONAL = 5_000.0   # risk $5k per straddle short


def main():
    print("=" * 70)
    print("Earnings IV crush · 2024-2026 · short ATM straddle through earnings")
    print("=" * 70)

    chain = pd.read_parquet(CHAIN_PATH)
    # Keep ATM legs only
    atm = chain[chain["is_atm"] == True].copy()
    print(f"\n{len(atm)} ATM legs · {atm['event_date'].nunique()} events × {atm['underlying_ticker'].nunique()} tickers")

    # Pivot: for each event, build (entry_call, entry_put, exit_call, exit_put)
    pivoted = atm.pivot_table(
        index=["underlying_ticker", "event_date"],
        columns=["timing", "contract_type"],
        values=["mid", "underlying_close", "atm_iv"],
        aggfunc="first"
    )
    pivoted.columns = ["_".join(c) for c in pivoted.columns]
    pivoted = pivoted.reset_index()

    trades = []
    for _, r in pivoted.iterrows():
        try:
            entry_straddle = r["mid_entry_call"] + r["mid_entry_put"]
            exit_straddle = r["mid_exit_call"] + r["mid_exit_put"]
            entry_iv = r.get("atm_iv_entry_call", np.nan)
            exit_iv = r.get("atm_iv_exit_call", np.nan)
            entry_spot = r.get("underlying_close_entry_call", np.nan)
            exit_spot = r.get("underlying_close_exit_call", np.nan)
            if pd.isna(entry_straddle) or pd.isna(exit_straddle):
                continue
            # SELL at entry, BUY BACK at exit. P&L per share = entry - exit.
            pnl_per_share = entry_straddle - exit_straddle
            # Size: target $STRADDLE_NOTIONAL of underlying exposure
            shares = STRADDLE_NOTIONAL / entry_spot if entry_spot > 0 else 0
            pnl_usd = pnl_per_share * shares
            # Realized return on underlying
            real_move_pct = (exit_spot - entry_spot) / entry_spot if entry_spot > 0 else 0
            iv_crush_pct = (entry_iv - exit_iv) / entry_iv if entry_iv > 0 else 0
            trades.append({
                "ticker": r["underlying_ticker"], "event_date": r["event_date"],
                "entry_straddle": float(entry_straddle), "exit_straddle": float(exit_straddle),
                "entry_iv": float(entry_iv) if pd.notna(entry_iv) else None,
                "exit_iv": float(exit_iv) if pd.notna(exit_iv) else None,
                "iv_crush_pct": float(iv_crush_pct * 100),
                "real_move_pct": float(real_move_pct * 100),
                "pnl_per_share": float(pnl_per_share),
                "pnl_usd": float(pnl_usd),
            })
        except Exception:
            continue

    df = pd.DataFrame(trades)
    print(f"\nResolved {len(df)} earnings straddle trades")
    if df.empty:
        print("  no trades — schema mismatch")
        return

    # Per-trade stats
    df = df.sort_values("event_date").reset_index(drop=True)
    df["cum_pnl"] = df["pnl_usd"].cumsum()
    total_pnl = float(df["pnl_usd"].sum())
    wins = int((df["pnl_usd"] > 0).sum())
    losses = int((df["pnl_usd"] < 0).sum())
    avg_win = float(df.loc[df["pnl_usd"] > 0, "pnl_usd"].mean()) if wins else 0
    avg_loss = float(df.loc[df["pnl_usd"] < 0, "pnl_usd"].mean()) if losses else 0
    sharpe = df["pnl_usd"].mean() / df["pnl_usd"].std() * np.sqrt(252 / 90) if df["pnl_usd"].std() > 0 else 0
    final_equity = STARTING_CAPITAL + total_pnl
    avg_iv_crush = float(df["iv_crush_pct"].mean())
    avg_real_move = float(df["real_move_pct"].abs().mean())

    print(f"\n=== Results ===")
    print(f"Trades:                {len(df)}")
    print(f"Wins / Losses:         {wins} / {losses} ({wins/len(df)*100:.0f}% win rate)")
    print(f"Total P&L:             ${total_pnl:+,.2f}")
    print(f"Avg win / Avg loss:    ${avg_win:+.2f} / ${avg_loss:+.2f}")
    print(f"Sharpe (approx):       {sharpe:.2f}")
    print(f"Avg IV crush:          {avg_iv_crush:.1f}%")
    print(f"Avg |real move|:       {avg_real_move:.1f}%")
    print(f"Ending equity (${STARTING_CAPITAL:,.0f} start): ${final_equity:,.2f} ({total_pnl/STARTING_CAPITAL*100:+.1f}%)")
    print(f"\n$1k scaled: $1,000 → ${1000 * (1 + total_pnl/STARTING_CAPITAL):.2f}")

    df.to_parquet(OUT / "06_earnings_iv_crush_results.parquet", index=False)
    summary = {
        "strategy": "earnings_iv_crush_short_atm_straddle",
        "n_trades": len(df), "wins": wins, "losses": losses,
        "win_rate_pct": float(wins / len(df) * 100),
        "total_pnl_usd": total_pnl,
        "avg_iv_crush_pct": avg_iv_crush,
        "avg_abs_real_move_pct": avg_real_move,
        "sharpe_approx": float(sharpe),
        "starting_capital": STARTING_CAPITAL,
        "final_equity": final_equity,
        "return_pct": float(total_pnl / STARTING_CAPITAL * 100),
    }
    (OUT / "06_earnings_iv_crush_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nwrote {OUT}/06_earnings_iv_crush_*.{{parquet,json}}")


if __name__ == "__main__":
    main()
