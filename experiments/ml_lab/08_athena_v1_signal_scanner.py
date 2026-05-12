"""Scan today's bars for ATHENA v1's OWN entries (not today's actual entries).

This is the apples-to-apples backtest: instead of replaying today's strategy's
entries through ATHENA's filter, we ask "what entries would ATHENA have
generated on today's bars, and what would their P&L be?"

For each bar in today's session:
  1. Check ATHENA's filter cascade
  2. If pass, check signal trigger (mean_rev for Lane-1, momentum for Lane-2)
  3. If trigger fires, simulate entry + time-based exit (close - close P&L)
  4. Apply 8 bps slippage
  5. Aggregate
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path("/Users/weeoto/turbo-carnival")
sys.path.insert(0, str(REPO / "scripts"))
from athena_ml_gates import ae_decision, logreg_p_volatile  # noqa
from voltgt_sizer import compute_notional, compute_realized_vol_bps  # noqa

OUT = REPO / "experiments" / "ml_lab" / "runs" / "athena"

LANE1 = {
    "symbols": ["SPY", "QQQ", "AAPL"],
    "ret1_z_min_abs": 1.8,
    "spread_max_bps": 5.0,
    "abs_ret_max_bps": 60,
    "tp_bps": 20.0,
    "sl_bps": 15.0,
    "time_exit_min": 10,
    "limit_offset_bps": 2.0,
}
LANE2 = {
    "symbols": ["SPY", "QQQ", "NVDA"],
    "ret5_z_min": 1.4,
    "range_pct_min_pctile": 60,
    "spread_max_bps": 6.0,
    "abs_ret_max_bps": 80,
    "tp_bps": 30.0,
    "sl_bps": 18.0,
    "time_exit_min": 15,
    "limit_offset_bps": 3.0,
}


def load_env():
    ENV = REPO / "council" / ".env.local"
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def fetch_today() -> pd.DataFrame:
    load_env()
    H = {"APCA-API-KEY-ID": os.environ["ALPACA_AEGIS_KEY"],
         "APCA-API-SECRET-KEY": os.environ["ALPACA_AEGIS_SECRET"]}
    rows = []
    for sym in ("SPY", "QQQ", "AAPL", "NVDA"):
        params = {"symbols": sym, "timeframe": "1Min",
                  "start": "2026-05-11T13:30:00Z",
                  "end": "2026-05-11T20:00:00Z",
                  "limit": 10000, "adjustment": "raw"}
        r = requests.get("https://data.alpaca.markets/v2/stocks/bars",
                         headers=H, params=params, timeout=20)
        for b in r.json().get("bars", {}).get(sym, []):
            rows.append({"symbol": sym, "t": b["t"], "open": b["o"],
                          "high": b["h"], "low": b["l"], "close": b["c"],
                          "volume": b["v"]})
    df = pd.DataFrame(rows)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["t_ny"] = df["t"].dt.tz_convert("America/New_York")
    df["date"] = df["t"].dt.date
    df = df.sort_values(["symbol", "t"]).reset_index(drop=True)
    df["ret1"] = df.groupby("symbol")["close"].pct_change(1).fillna(0)
    df["ret3"] = df.groupby("symbol")["close"].pct_change(3).fillna(0)
    df["ret5"] = df.groupby("symbol")["close"].pct_change(5).fillna(0)
    df["ret10"] = df.groupby("symbol")["close"].pct_change(10).fillna(0)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"]
    df["volume_z"] = df.groupby("symbol")["volume"].transform(
        lambda s: (s - s.rolling(20, min_periods=5).mean()) / s.rolling(20, min_periods=5).std()
    ).fillna(0)
    df["ret1_z"] = df.groupby("symbol")["ret1"].transform(
        lambda s: (s - s.rolling(20, min_periods=5).mean()) / s.rolling(20, min_periods=5).std()
    ).fillna(0)
    df["ret5_z"] = df.groupby("symbol")["ret5"].transform(
        lambda s: (s - s.rolling(20, min_periods=5).mean()) / s.rolling(20, min_periods=5).std()
    ).fillna(0)
    df["session_vwap"] = df.groupby(["symbol", "date"]).apply(
        lambda g: (g["close"] * g["volume"]).cumsum() / g["volume"].cumsum().replace(0, np.nan),
        include_groups=False,
    ).reset_index(level=[0, 1], drop=True)
    df["vwap_dist"] = (df["close"] - df["session_vwap"]) / df["session_vwap"]
    mins = ((df["t_ny"].dt.hour - 9) * 60 + df["t_ny"].dt.minute - 30).clip(0, 390)
    df["tod_sin"] = np.sin(2 * np.pi * mins / 390)
    df["tod_cos"] = np.cos(2 * np.pi * mins / 390)

    def vov(g):
        s = g["ret1"].rolling(5).std()
        m = g["ret1"].abs().rolling(20).median()
        return (s / m).fillna(0).clip(0, 50)
    df["vov"] = df.groupby("symbol", group_keys=False).apply(vov, include_groups=False)
    # Range percentile per symbol per session (rolling 60 bars)
    df["range_pct_p60"] = df.groupby("symbol")["range_pct"].transform(
        lambda s: s.rolling(60, min_periods=10).quantile(0.6)
    )
    return df


def in_window(ts_ny: pd.Timestamp) -> bool:
    h, m = ts_ny.hour, ts_ny.minute
    return 10 * 60 <= h * 60 + m <= 15 * 60 + 30  # 10:00 ET - 15:30 ET inclusive


def simulate_lane(df: pd.DataFrame, lane_cfg: dict, lane_name: str,
                   slippage_bps: float = 8.0) -> list[dict]:
    """Scan bars for entries; for each, simulate exit + P&L."""
    LR_FEATS = ["ret1", "ret3", "ret10", "range_pct", "volume_z", "ret1_z",
                "vwap_dist", "tod_sin", "tod_cos", "vov"]
    trades = []
    df = df[df["symbol"].isin(lane_cfg["symbols"])].copy()

    for sym, g in df.groupby("symbol"):
        g = g.reset_index(drop=True)
        i = 25
        last_entry_minute = -100
        while i < len(g) - lane_cfg["time_exit_min"]:
            bar = g.iloc[i]
            ts_ny = pd.Timestamp(bar["t_ny"])
            if not in_window(ts_ny):
                i += 1
                continue
            if i - last_entry_minute < 5:  # cooldown 5 min between entries / symbol
                i += 1
                continue
            # abs return gate
            if abs(bar["ret1"]) * 10000 > lane_cfg["abs_ret_max_bps"]:
                i += 1
                continue
            # LogReg gate
            feats = {k: float(bar[k]) for k in LR_FEATS}
            p_vol = logreg_p_volatile(feats)
            if p_vol is not None and p_vol >= 0.50:
                i += 1
                continue
            # AE veto
            if i >= 30:
                window = g.iloc[i - 29:i + 1][
                    ["ret1", "ret3", "range_pct", "volume_z", "ret1_z", "vwap_dist"]
                ].values
                ae_dec, _err = ae_decision(window)
                if ae_dec == "veto":
                    i += 1
                    continue
            # Signal trigger
            if lane_name == "Aegis":
                if abs(bar["ret1_z"]) < lane_cfg["ret1_z_min_abs"]:
                    i += 1
                    continue
                if abs(bar["vwap_dist"]) * 10000 < 8.0:
                    i += 1
                    continue
                side = "short" if bar["ret1_z"] > 0 else "long"
            else:  # Falcon momentum/breakout
                if abs(bar["ret5_z"]) < lane_cfg["ret5_z_min"]:
                    i += 1
                    continue
                if pd.notna(bar["range_pct_p60"]) and bar["range_pct"] < bar["range_pct_p60"]:
                    i += 1
                    continue
                side = "long" if bar["ret5_z"] > 0 else "short"

            # Vol-inverse sizing
            rv_bps = compute_realized_vol_bps(g["ret1"].iloc[i - 4:i + 1].tolist())
            notional = compute_notional(rv_bps)
            entry_px = bar["close"]
            qty = notional / entry_px

            # Walk forward bar-by-bar to find exit
            sign = 1.0 if side == "long" else -1.0
            tp_px = entry_px * (1 + sign * lane_cfg["tp_bps"] / 10000)
            sl_px = entry_px * (1 - sign * lane_cfg["sl_bps"] / 10000)
            exit_px = entry_px
            exit_reason = "time"
            for j in range(1, lane_cfg["time_exit_min"] + 1):
                if i + j >= len(g):
                    break
                nb = g.iloc[i + j]
                # TP/SL check using high/low for the bar
                if side == "long":
                    if nb["high"] >= tp_px:
                        exit_px = tp_px
                        exit_reason = "tp"
                        break
                    if nb["low"] <= sl_px:
                        exit_px = sl_px
                        exit_reason = "sl"
                        break
                else:
                    if nb["low"] <= tp_px:
                        exit_px = tp_px
                        exit_reason = "tp"
                        break
                    if nb["high"] >= sl_px:
                        exit_px = sl_px
                        exit_reason = "sl"
                        break
                exit_px = nb["close"]

            # P&L gross
            gross_pnl = (exit_px - entry_px) * qty * sign
            # Apply slippage: entry pays half-spread, exit pays half-spread,
            # but with marketable limit offset we estimate ~5 bps round trip
            slippage_cost = notional * (slippage_bps / 10000.0)
            net_pnl = gross_pnl - slippage_cost

            trades.append({
                "lane": lane_name, "symbol": sym, "entry_ts": bar["t"],
                "entry_minute": bar["t_ny"].strftime("%H:%M"),
                "side": side, "entry_px": entry_px,
                "exit_px": exit_px, "exit_reason": exit_reason,
                "qty": qty, "notional": notional,
                "rv_bps": rv_bps, "p_vol": p_vol,
                "gross_pnl": gross_pnl, "slippage_cost": slippage_cost,
                "net_pnl": net_pnl,
            })
            last_entry_minute = i
            i += 5  # skip 5 min after entry to avoid double-firing on same setup
        # end while
    # end for symbol
    return trades


def main() -> int:
    print("fetching today's enriched bars…")
    df = fetch_today()
    print(f"  {len(df):,} bars across {df['symbol'].nunique()} symbols")
    print()

    print("scanning Lane-1 (Aegis-α mean-reversion)…")
    lane1_trades = simulate_lane(df, LANE1, "Aegis")
    print(f"  {len(lane1_trades)} entries")
    print()
    print("scanning Lane-2 (Falcon-α momentum)…")
    lane2_trades = simulate_lane(df, LANE2, "Falcon")
    print(f"  {len(lane2_trades)} entries")

    all_trades = lane1_trades + lane2_trades
    if not all_trades:
        print("\n  NO entries triggered. Filter cascade too strict for today's regime.")
        (OUT / "athena_v1_self_scan_results.json").write_text(
            json.dumps({"n_trades": 0, "note": "no triggers today"}, indent=2)
        )
        return 0
    out_df = pd.DataFrame(all_trades)
    out_df.to_parquet(OUT / "athena_v1_self_scan_trades.parquet", index=False)

    print("\n=== ATHENA v1 self-scan results on today (2026-05-11) ===")
    print(f"  Total entries (both lanes): {len(out_df)}")
    print(f"  Lane-1 (Aegis): {len(out_df[out_df['lane']=='Aegis'])}")
    print(f"  Lane-2 (Falcon): {len(out_df[out_df['lane']=='Falcon'])}")
    by_sym = out_df.groupby("symbol").size().to_dict()
    print(f"  By symbol: {by_sym}")
    by_exit = out_df.groupby("exit_reason").size().to_dict()
    print(f"  By exit reason: {by_exit}")
    print()
    print(f"  Gross P&L (no slippage):   ${out_df['gross_pnl'].sum():+,.2f}")
    print(f"  Slippage cost (8 bps):     $-{out_df['slippage_cost'].sum():,.2f}")
    print(f"  Net P&L (after 8 bps slip):${out_df['net_pnl'].sum():+,.2f}")
    print()
    print(f"  Today's actual P&L (rule-based, market orders): $-3,443.21")
    print(f"  ATHENA v1 delta vs actual:                       ${out_df['net_pnl'].sum() - (-3443.21):+,.2f}")
    print()

    # 12 bps stress test
    stress_cost = out_df["notional"].sum() * (12.0 / 10000.0)
    stress_net = out_df["gross_pnl"].sum() - stress_cost
    print(f"  Stress at 12 bps slippage: ${stress_net:+,.2f} (was ${out_df['net_pnl'].sum():+,.2f} at 8 bps)")

    win_count = (out_df["net_pnl"] > 0).sum()
    print(f"\n  Win rate (net): {win_count}/{len(out_df)} = {100*win_count/len(out_df):.1f}%")

    summary = {
        "stamp": datetime.now(timezone.utc).isoformat(),
        "n_trades": len(out_df),
        "lane_counts": {
            "Aegis": int(len(out_df[out_df["lane"] == "Aegis"])),
            "Falcon": int(len(out_df[out_df["lane"] == "Falcon"])),
        },
        "by_symbol": by_sym,
        "by_exit_reason": by_exit,
        "gross_pnl": float(out_df["gross_pnl"].sum()),
        "slippage_cost_8bps": float(out_df["slippage_cost"].sum()),
        "net_pnl_8bps": float(out_df["net_pnl"].sum()),
        "net_pnl_12bps_stress": float(stress_net),
        "win_rate_pct": float(100 * win_count / len(out_df)),
        "actual_pnl_today": -3443.21,
        "delta_vs_actual": float(out_df["net_pnl"].sum() - (-3443.21)),
    }
    (OUT / "athena_v1_self_scan_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  wrote {OUT / 'athena_v1_self_scan_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
