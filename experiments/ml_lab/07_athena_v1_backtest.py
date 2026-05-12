"""Replay today's 162 entries through the ATHENA v1 cascade.

For each entry, we:
1. Pull the bar features at entry minute
2. Apply ATHENA's filter cascade (time_gate, spread_gate, abs_return_gate,
   LogReg gate, AE veto, symbol whitelist, signal check)
3. Decide BLOCK / SIZE_DOWN / FULL_SIZE
4. Compute counterfactual P&L using minute-bar close prices and the
   adjusted notional (vol-inverse sizing)
5. Aggregate per-lane and overall

Output: experiments/ml_lab/runs/athena/athena_v1_backtest_results.json
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/Users/weeoto/turbo-carnival")
sys.path.insert(0, str(REPO / "scripts"))

from athena_ml_gates import ae_decision, logreg_p_volatile  # noqa: E402
from voltgt_sizer import compute_notional, compute_realized_vol_bps  # noqa: E402

BARS = REPO / "experiments" / "runs" / "valk_hft_demo_20260511T140548Z" / "minute_bars.parquet"
ENTRIES = REPO / "experiments" / "ml_lab" / "runs" / "ml_zoo_per_entry_latest.parquet"
OUT = REPO / "experiments" / "ml_lab" / "runs" / "athena"
OUT.mkdir(parents=True, exist_ok=True)

LANE1_SYMBOLS = {"SPY", "QQQ", "AAPL"}
LANE2_SYMBOLS = {"SPY", "QQQ", "NVDA"}
LANE1_SPREAD_MAX_BPS = 5.0
LANE2_SPREAD_MAX_BPS = 6.0
LANE1_ABS_RET_MAX_BPS = 60
LANE2_ABS_RET_MAX_BPS = 80
TIME_START_ET = (10, 0)
TIME_END_ET = (15, 30)

# ATHENA tighter entry thresholds (vs baseline ~1.5)
LANE1_RET1_Z_MIN_ABS = 1.8
LANE1_VWAP_DIST_MIN_BPS = 8.0


def in_time_window(t_ny: pd.Timestamp) -> bool:
    h, m = t_ny.hour, t_ny.minute
    start = TIME_START_ET[0] * 60 + TIME_START_ET[1]
    end = TIME_END_ET[0] * 60 + TIME_END_ET[1]
    now = h * 60 + m
    return start <= now <= end


def main() -> int:
    bars = pd.read_parquet(BARS)
    bars["t"] = pd.to_datetime(bars["t"], utc=True)
    bars["t_ny"] = pd.to_datetime(bars["t_ny"], errors="coerce")
    bars["minute"] = bars["t"].dt.floor("1min")

    # Use the same enriched bars from the ml_zoo run so today's data is included
    # The ml_zoo run merged Alpaca fresh-fetch for today; re-do it briefly here
    print("loading per-entry data from ml_zoo run…")
    per_entry = pd.read_parquet(ENTRIES)
    # The per-entry has one row per (model, entry); we just need unique entries
    one_model = per_entry["model"].iloc[0]
    entries = per_entry[per_entry["model"] == one_model].copy().reset_index(drop=True)
    print(f"  {len(entries)} unique entries today")
    entries["entry_ts"] = pd.to_datetime(entries["entry_ts"], utc=True)
    entries["entry_minute"] = entries["entry_ts"].dt.floor("1min")

    # We need today's bars merged. The bars cache stops at 14:04 UTC today, but
    # ml_zoo fetched fresh ones; we redo it here using the same Alpaca path.
    # For simplicity, reuse the ml_zoo's enrichment by reading the parquet now
    # that script 04 saved features alongside entries. We'll fetch by minute_bars
    # for entry rows directly.
    import os
    import requests
    ENV = REPO / "council" / ".env.local"
    if ENV.exists():
        for line in ENV.read_text().splitlines():
            line = line.strip()
            if line and "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                k = k.strip()
                if k not in os.environ:
                    os.environ[k] = v.strip().strip('"').strip("'")

    key = os.environ.get("ALPACA_AEGIS_KEY")
    sec = os.environ.get("ALPACA_AEGIS_SECRET")
    H = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": sec}
    base = "https://data.alpaca.markets/v2/stocks/bars"
    today_rows = []
    for sym in ("SPY", "QQQ", "AAPL", "AMD", "META", "NVDA", "TSLA"):
        params = {"symbols": sym, "timeframe": "1Min",
                  "start": "2026-05-11T13:30:00Z",
                  "end": "2026-05-11T20:00:00Z",
                  "limit": 10000, "adjustment": "raw"}
        try:
            r = requests.get(base, headers=H, params=params, timeout=20)
            data = r.json().get("bars", {}).get(sym, []) if r.status_code == 200 else []
            for b in data:
                today_rows.append({
                    "symbol": sym, "t": b["t"], "open": b["o"], "high": b["h"],
                    "low": b["l"], "close": b["c"], "volume": b["v"],
                })
        except Exception:
            continue
    today_df = pd.DataFrame(today_rows)
    today_df["t"] = pd.to_datetime(today_df["t"], utc=True)
    today_df["t_ny"] = today_df["t"].dt.tz_convert("America/New_York")
    today_df["minute"] = today_df["t"].dt.floor("1min")
    today_df["date"] = today_df["t"].dt.date
    today_df = today_df.sort_values(["symbol", "t"]).reset_index(drop=True)
    today_df["ret1"] = today_df.groupby("symbol")["close"].pct_change(1).fillna(0)
    today_df["ret3"] = today_df.groupby("symbol")["close"].pct_change(3).fillna(0)
    today_df["ret10"] = today_df.groupby("symbol")["close"].pct_change(10).fillna(0)
    today_df["range_pct"] = (today_df["high"] - today_df["low"]) / today_df["close"]
    today_df["volume_z"] = today_df.groupby("symbol")["volume"].transform(
        lambda s: (s - s.rolling(20, min_periods=5).mean()) / s.rolling(20, min_periods=5).std()
    ).fillna(0)
    today_df["ret1_z"] = today_df.groupby("symbol")["ret1"].transform(
        lambda s: (s - s.rolling(20, min_periods=5).mean()) / s.rolling(20, min_periods=5).std()
    ).fillna(0)
    today_df["session_vwap"] = today_df.groupby(["symbol", "date"]).apply(
        lambda g: (g["close"] * g["volume"]).cumsum() / g["volume"].cumsum().replace(0, np.nan),
        include_groups=False,
    ).reset_index(level=[0, 1], drop=True)
    today_df["vwap_dist"] = (today_df["close"] - today_df["session_vwap"]) / today_df["session_vwap"]
    mins = ((today_df["t_ny"].dt.hour - 9) * 60 + today_df["t_ny"].dt.minute - 30).clip(0, 390)
    today_df["tod_sin"] = np.sin(2 * np.pi * mins / 390)
    today_df["tod_cos"] = np.cos(2 * np.pi * mins / 390)

    def vov(g):
        s = g["ret1"].rolling(5).std()
        m = g["ret1"].abs().rolling(20).median()
        return (s / m).fillna(0).clip(0, 50)
    today_df["vov"] = today_df.groupby("symbol", group_keys=False).apply(vov, include_groups=False)

    today_df = today_df.set_index(["symbol", "minute"])
    print(f"  enriched today bars: {len(today_df):,}")

    # Run cascade per entry
    LR_FEATS = ["ret1", "ret3", "ret10", "range_pct", "volume_z", "ret1_z",
                "vwap_dist", "tod_sin", "tod_cos", "vov"]
    rows = []
    blocks = {"time_gate": 0, "symbol_whitelist": 0, "spread_gate": 0,
              "abs_return_gate": 0, "logreg_gate": 0, "ae_veto": 0,
              "signal_strength": 0}

    for _, e in entries.iterrows():
        try:
            f = today_df.loc[(e["symbol"], e["entry_minute"])]
            if isinstance(f, pd.DataFrame):
                f = f.iloc[0]
        except KeyError:
            continue
        ts_ny = pd.Timestamp(e["entry_ts"]).tz_convert("America/New_York")
        lane = e["lane"]
        symbols = LANE1_SYMBOLS if lane == "Aegis" else LANE2_SYMBOLS
        spread_max = LANE1_SPREAD_MAX_BPS if lane == "Aegis" else LANE2_SPREAD_MAX_BPS
        abs_ret_max = LANE1_ABS_RET_MAX_BPS if lane == "Aegis" else LANE2_ABS_RET_MAX_BPS

        decisions = []
        blocked_by = None

        if not in_time_window(ts_ny):
            blocks["time_gate"] += 1
            blocked_by = "time_gate"
            decisions.append("time_gate")
        elif e["symbol"] not in symbols:
            blocks["symbol_whitelist"] += 1
            blocked_by = "symbol_whitelist"
            decisions.append("symbol_whitelist")
        else:
            # Proxy spread from range_pct of last bar (no real bid/ask in cache)
            implied_spread_bps = abs(f["range_pct"]) * 10000 * 0.5
            if implied_spread_bps > spread_max * 3:  # generous proxy
                blocks["spread_gate"] += 1
                blocked_by = "spread_gate"
                decisions.append("spread_gate")
            elif abs(f["ret1"]) * 10000 > abs_ret_max:
                blocks["abs_return_gate"] += 1
                blocked_by = "abs_return_gate"
                decisions.append("abs_return_gate")
            else:
                feats = {k: float(f[k]) for k in LR_FEATS}
                p_vol = logreg_p_volatile(feats)
                if p_vol is not None and p_vol >= 0.50:
                    blocks["logreg_gate"] += 1
                    blocked_by = "logreg_gate"
                    decisions.append(f"logreg_gate(p={p_vol:.3f})")
                else:
                    # Build 30-bar window for AE
                    sym_bars = today_df.xs(e["symbol"], level=0)
                    idx = sym_bars.index.get_indexer([e["entry_minute"]])
                    if len(idx) == 0 or idx[0] < 29:
                        ae_dec = "pass"
                        err = 0.0
                    else:
                        window = sym_bars.iloc[idx[0] - 29:idx[0] + 1][
                            ["ret1", "ret3", "range_pct", "volume_z", "ret1_z", "vwap_dist"]
                        ].values
                        ae_dec, err = ae_decision(window)
                    if ae_dec == "veto":
                        blocks["ae_veto"] += 1
                        blocked_by = "ae_veto"
                        decisions.append(f"ae_veto(err={err:.3f})")
                    else:
                        # Signal strength filter (lane-specific)
                        if lane == "Aegis":
                            if abs(f["ret1_z"]) < LANE1_RET1_Z_MIN_ABS:
                                blocks["signal_strength"] += 1
                                blocked_by = "signal_strength"
                                decisions.append(f"signal_strength(z={f['ret1_z']:.2f})")
                        else:
                            if abs(f["ret1_z"]) < 1.2:
                                blocks["signal_strength"] += 1
                                blocked_by = "signal_strength"
                                decisions.append(f"signal_strength(z={f['ret1_z']:.2f})")

        # Size adjustment
        sym_bars = today_df.xs(e["symbol"], level=0)
        idx_arr = sym_bars.index.get_indexer([e["entry_minute"]])
        if len(idx_arr) > 0 and idx_arr[0] >= 5:
            recent_rets = sym_bars["ret1"].iloc[idx_arr[0] - 4:idx_arr[0] + 1].values
            rv_bps = compute_realized_vol_bps(list(recent_rets))
        else:
            rv_bps = float("nan")
        new_notional = compute_notional(rv_bps)
        # Original notional was ~$20k; entry P&L scales linearly with notional
        scale = new_notional / 20000.0

        # ATHENA-scaled counterfactual P&L
        athena_pnl = float(e["pnl"]) * scale if blocked_by is None else 0.0

        rows.append({
            "entry_ts": e["entry_ts"], "lane": e["lane"], "symbol": e["symbol"],
            "side_dir": e["side_dir"], "exit_action": e["exit_action"],
            "original_pnl": float(e["pnl"]),
            "athena_blocked_by": blocked_by,
            "athena_decisions": ", ".join(decisions),
            "athena_notional": new_notional,
            "athena_scale": scale,
            "athena_pnl": athena_pnl,
            "rv_bps": rv_bps,
        })

    out_df = pd.DataFrame(rows)
    out_df.to_parquet(OUT / "athena_v1_per_entry.parquet", index=False)

    actual_total = out_df["original_pnl"].sum()
    athena_total = out_df["athena_pnl"].sum()
    n_kept = (out_df["athena_blocked_by"].isna()).sum()
    n_blocked = (out_df["athena_blocked_by"].notna()).sum()

    print("\n=== ATHENA v1 backtest on today's 162 entries ===")
    print(f"  Entries kept:    {n_kept} / {len(out_df)} ({100*n_kept/len(out_df):.1f}%)")
    print(f"  Entries blocked: {n_blocked} ({100*n_blocked/len(out_df):.1f}%)")
    print()
    print("  Blocks by stage:")
    for k, v in blocks.items():
        print(f"    {k:24} {v}")
    print()
    print(f"  Actual realised P&L (bar-close, $20k uniform): ${actual_total:+,.2f}")
    print(f"  ATHENA v1 P&L (bar-close, vol-inverse size):   ${athena_total:+,.2f}")
    print(f"  Delta vs actual:                                ${athena_total - actual_total:+,.2f}")

    # Per-lane breakdown
    print("\n  Per-lane:")
    for lane in ("Aegis", "Falcon"):
        sub = out_df[out_df["lane"] == lane]
        if not sub.empty:
            print(f"    {lane}: kept {(sub['athena_blocked_by'].isna()).sum()}/{len(sub)}, "
                  f"P&L ${sub['athena_pnl'].sum():+,.2f} "
                  f"(vs ${sub['original_pnl'].sum():+,.2f} actual)")

    # Apply 8 bps slippage estimate to ATHENA kept trades
    kept = out_df[out_df["athena_blocked_by"].isna()].copy()
    slippage_cost_per_trade = kept["athena_notional"] * 8 / 10000.0
    athena_at_8bps = athena_total - slippage_cost_per_trade.sum()
    print(f"\n  ATHENA P&L at 8 bps slippage: ${athena_at_8bps:+,.2f}")
    print(f"    ({len(kept)} trades × avg $\
{slippage_cost_per_trade.mean():.2f} slippage cost)")

    summary = {
        "stamp": datetime.now(timezone.utc).isoformat(),
        "n_entries_seen": int(len(out_df)),
        "n_blocked": int(n_blocked),
        "n_kept": int(n_kept),
        "block_rate_pct": 100.0 * n_blocked / len(out_df),
        "blocks_by_stage": blocks,
        "actual_pnl_bar_close": float(actual_total),
        "athena_pnl_bar_close": float(athena_total),
        "athena_delta_bar_close": float(athena_total - actual_total),
        "athena_pnl_at_8bps_slip": float(athena_at_8bps),
        "athena_slippage_cost_total": float(slippage_cost_per_trade.sum()),
        "lanes": {
            lane: {
                "kept": int((out_df[out_df["lane"] == lane]["athena_blocked_by"].isna()).sum()),
                "seen": int(len(out_df[out_df["lane"] == lane])),
                "athena_pnl": float(out_df[out_df["lane"] == lane]["athena_pnl"].sum()),
                "actual_pnl": float(out_df[out_df["lane"] == lane]["original_pnl"].sum()),
            }
            for lane in ("Aegis", "Falcon")
        },
    }
    (OUT / "athena_v1_backtest_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {OUT / 'athena_v1_backtest_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
