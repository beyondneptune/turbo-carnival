"""PCP scanner — for each (strike, expiry) where SPY has both call+put, compute
the put-call parity residual and flag mispricings beyond bid-ask cost.

PCP (European, dividend-adjusted):
   C − P = S − K·e^(−r·T) − PV(dividends)

Equivalently: residual = C - P - S + K·e^(−r·T) + PV(divs).
If |residual| > bid-ask round-trip cost, in theory you can conversion/reversal
arbitrage. In practice retail almost never edges this — but DOCUMENTING the
distribution of residuals is useful market microstructure data.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = Path("/Users/weeoto/turbo-carnival")
OUT_DIR = REPO / "experiments" / "options_lab" / "runs"
OUT_DIR.mkdir(parents=True, exist_ok=True)

UNDERLYING = "SPY"
RISK_FREE_RATE = 0.05    # annual; ~5% Treasury approximation
SPY_DIV_YIELD = 0.013    # annual; SPY dividend yield
MAX_EXPIRIES = 5         # nearest N expiries
MAX_DTE = 60             # cap days-to-expiry


def load_env():
    ENV = REPO / "council" / ".env.local"
    for line in ENV.read_text().splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip().strip('"').strip("'")


def get_alpaca_headers():
    load_env()
    return {"APCA-API-KEY-ID": os.environ["ALPACA_AEGIS_KEY"],
            "APCA-API-SECRET-KEY": os.environ["ALPACA_AEGIS_SECRET"]}


def fetch_underlying_price_at(symbol: str, target_ts: datetime) -> float:
    """Fetch underlying price at the option quote's timestamp (or nearest bar).

    This is critical for PCP — using stale S vs current options quotes creates
    a constant offset that masquerades as parity violation.
    """
    H = get_alpaca_headers()
    # Look up the minute bar at target_ts
    start = (target_ts.replace(second=0, microsecond=0))
    end = start.replace(minute=start.minute + 1) if start.minute < 59 else start.replace(hour=start.hour + 1, minute=0)
    r = requests.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/bars",
                      headers=H, params={"timeframe": "1Min",
                                           "start": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                           "end": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                           "limit": 1}, timeout=10)
    if r.status_code == 200 and r.json().get("bars"):
        return float(r.json()["bars"][0]["c"])
    raise RuntimeError(f"could not fetch {symbol} price at {target_ts}")


def fetch_underlying_price(symbol: str) -> float:
    """Latest available (may be stale relative to options snapshot)."""
    H = get_alpaca_headers()
    r = requests.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest",
                      headers=H, timeout=10)
    if r.status_code == 200 and r.json().get("trade"):
        return float(r.json()["trade"]["p"])
    r = requests.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/bars/latest",
                      headers=H, timeout=10)
    if r.status_code == 200 and r.json().get("bar"):
        return float(r.json()["bar"]["c"])
    raise RuntimeError(f"could not fetch {symbol} price")


def fetch_option_contracts(underlying: str) -> pd.DataFrame:
    """List active option contracts for the underlying."""
    H = get_alpaca_headers()
    rows = []
    page_token = None
    seen_pages = 0
    while True:
        params = {"underlying_symbols": underlying, "limit": 10000,
                   "status": "active"}
        if page_token:
            params["page_token"] = page_token
        r = requests.get("https://paper-api.alpaca.markets/v2/options/contracts",
                          headers=H, params=params, timeout=20)
        if r.status_code != 200:
            print(f"contracts API HTTP {r.status_code}: {r.text[:200]}")
            break
        j = r.json()
        for c in j.get("option_contracts", []):
            rows.append({
                "symbol": c["symbol"], "underlying": c["underlying_symbol"],
                "type": c["type"], "strike": float(c["strike_price"]),
                "expiration": c["expiration_date"],
            })
        page_token = j.get("next_page_token")
        seen_pages += 1
        if not page_token or seen_pages > 20:
            break
    df = pd.DataFrame(rows)
    return df


def fetch_option_snapshots(symbols: list[str]) -> dict[str, dict]:
    """Batch fetch latest bid/ask for each option symbol. Up to 100 per call."""
    H = get_alpaca_headers()
    out = {}
    batch = 100
    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        params = {"symbols": ",".join(chunk)}
        r = requests.get("https://data.alpaca.markets/v1beta1/options/snapshots",
                          headers=H, params=params, timeout=30)
        if r.status_code != 200:
            print(f"  snapshots API HTTP {r.status_code} batch {i}: {r.text[:200]}")
            continue
        j = r.json()
        for sym, snap in j.get("snapshots", {}).items():
            out[sym] = snap
    return out


def main() -> int:
    print("=" * 70)
    print(f"PCP SCANNER · {UNDERLYING} · {datetime.now(timezone.utc).isoformat()}")
    print("=" * 70)

    # 1. Underlying price — match the option-quote timestamp (yesterday's close
    #    if pre-market, or current if intraday). Sample one option snapshot first
    #    to get the quote timestamp, then look up SPY at that exact minute.
    print("\nDetermining option-quote timestamp...")
    H = get_alpaca_headers()
    probe = requests.get("https://data.alpaca.markets/v1beta1/options/snapshots",
                          headers=H, params={"symbols": f"{UNDERLYING}260513C00739000"},
                          timeout=10).json()
    probe_snap = list(probe.get("snapshots", {}).values())
    quote_ts = None
    if probe_snap and probe_snap[0].get("latestQuote"):
        from datetime import datetime as _dt
        quote_ts = _dt.fromisoformat(probe_snap[0]["latestQuote"]["t"].replace("Z", "+00:00"))
        print(f"  option quotes timestamped: {quote_ts.isoformat()}")
    if quote_ts:
        S = fetch_underlying_price_at(UNDERLYING, quote_ts)
        print(f"\nUnderlying {UNDERLYING} at option-quote time: ${S:.2f}")
    else:
        S = fetch_underlying_price(UNDERLYING)
        print(f"\nUnderlying {UNDERLYING} (latest, may be stale): ${S:.2f}")

    # 2. Option chain
    print(f"\nFetching option contracts for {UNDERLYING}...")
    contracts = fetch_option_contracts(UNDERLYING)
    print(f"  total active contracts: {len(contracts):,}")
    contracts["expiration"] = pd.to_datetime(contracts["expiration"])
    today = pd.Timestamp(datetime.now(timezone.utc).date())
    contracts["dte"] = (contracts["expiration"] - today).dt.days
    contracts = contracts[(contracts["dte"] > 0) & (contracts["dte"] <= MAX_DTE)]
    print(f"  within {MAX_DTE} DTE: {len(contracts):,}")

    # Pick nearest MAX_EXPIRIES expirations
    expirations = sorted(contracts["expiration"].unique())[:MAX_EXPIRIES]
    contracts = contracts[contracts["expiration"].isin(expirations)]
    print(f"  using {len(expirations)} expirations: {[d.strftime('%Y-%m-%d') for d in expirations]}")

    # Strikes near ATM (within ±10%)
    strike_min, strike_max = S * 0.90, S * 1.10
    contracts = contracts[(contracts["strike"] >= strike_min) &
                           (contracts["strike"] <= strike_max)]
    print(f"  near-ATM (90-110%): {len(contracts):,}")

    # 3. Pair calls and puts at each (strike, expiry)
    paired = contracts.pivot_table(
        index=["expiration", "strike"], columns="type", values="symbol",
        aggfunc="first"
    ).reset_index().dropna(subset=["call", "put"])
    print(f"\nPaired (strike, expiry) with both call+put: {len(paired):,}")

    # 4. Snapshot all option symbols
    all_syms = paired["call"].tolist() + paired["put"].tolist()
    print(f"\nFetching snapshots for {len(all_syms)} options...")
    snaps = fetch_option_snapshots(all_syms)
    print(f"  got {len(snaps)} snapshots")

    # 5. Compute PCP residual per pair
    rows = []
    today_dt = datetime.now(timezone.utc).replace(hour=21, minute=0, second=0, microsecond=0)
    for _, r in paired.iterrows():
        c_sym, p_sym = r["call"], r["put"]
        if c_sym not in snaps or p_sym not in snaps:
            continue
        c_snap, p_snap = snaps[c_sym], snaps[p_sym]
        c_quote = c_snap.get("latestQuote") or {}
        p_quote = p_snap.get("latestQuote") or {}
        c_bid, c_ask = c_quote.get("bp", 0), c_quote.get("ap", 0)
        p_bid, p_ask = p_quote.get("bp", 0), p_quote.get("ap", 0)
        if c_bid <= 0 or c_ask <= 0 or p_bid <= 0 or p_ask <= 0:
            continue
        c_mid, p_mid = (c_bid + c_ask) / 2, (p_bid + p_ask) / 2
        c_spread, p_spread = c_ask - c_bid, p_ask - p_bid

        K = float(r["strike"])
        exp = pd.Timestamp(r["expiration"])
        # Time to expiry in years (assume 16:00 ET = 21:00 UTC settlement)
        seconds_to_exp = (exp.replace(hour=21).tz_localize("UTC") -
                           datetime.now(timezone.utc)).total_seconds()
        T = max(seconds_to_exp / (365.25 * 86400), 0)
        # PV(divs) — approximation using yield
        pv_divs = S * SPY_DIV_YIELD * T
        # PCP residual (using mid prices)
        residual = c_mid - p_mid - S + K * math.exp(-RISK_FREE_RATE * T) + pv_divs
        # Round-trip cost: cross spread on both legs
        rt_cost = c_spread + p_spread
        tradeable = abs(residual) > rt_cost

        rows.append({
            "expiration": exp.strftime("%Y-%m-%d"),
            "dte": int(r.get("dte", -1)) if "dte" in r else int((exp.date() - datetime.now(timezone.utc).date()).days),
            "strike": K,
            "moneyness": K / S,
            "call_sym": c_sym, "put_sym": p_sym,
            "c_bid": c_bid, "c_ask": c_ask, "c_mid": c_mid, "c_spread": c_spread,
            "p_bid": p_bid, "p_ask": p_ask, "p_mid": p_mid, "p_spread": p_spread,
            "S": S, "T_years": T, "pv_divs": pv_divs,
            "residual": residual,
            "rt_cost": rt_cost,
            "tradeable": tradeable,
            "abs_residual_over_rt": abs(residual) / max(rt_cost, 1e-6),
        })

    df = pd.DataFrame(rows).sort_values("abs_residual_over_rt", ascending=False).reset_index(drop=True)
    print(f"\nComputed PCP residuals for {len(df)} pairs")
    print(f"Tradeable (|residual| > round-trip cost): {df['tradeable'].sum()}")
    print(f"Residual stats: median ${df['residual'].median():+.3f}  "
          f"min ${df['residual'].min():+.3f}  max ${df['residual'].max():+.3f}")

    # Save
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = OUT_DIR / f"pcp_snapshot_{stamp}.parquet"
    df.to_parquet(out_path, index=False)
    df.to_parquet(OUT_DIR / "pcp_snapshot_latest.parquet", index=False)
    print(f"\nwrote {out_path}")

    # Show top 15 mispricings
    print(f"\n=== TOP 15 |residual| / round-trip-cost ratios ===")
    show_cols = ["expiration", "dte", "strike", "moneyness", "c_mid", "p_mid",
                 "residual", "rt_cost", "abs_residual_over_rt", "tradeable"]
    show = df[show_cols].head(15).copy()
    for c in ("strike", "c_mid", "p_mid", "residual", "rt_cost"):
        show[c] = show[c].round(3)
    show["moneyness"] = show["moneyness"].round(3)
    show["abs_residual_over_rt"] = show["abs_residual_over_rt"].round(2)
    print(show.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
