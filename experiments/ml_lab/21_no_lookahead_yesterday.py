"""Walk-forward no-lookahead test for yesterday.

For each strategy:
  - SVXY rotation: extract yesterday's row from the existing walk-forward
    (already no-lookahead — signal at T-1, position at T)
  - Composite C3 gate: retrain 4 sub-models on days 1-20 ONLY (excluding
    yesterday's bars), then apply to yesterday's 162 entries
  - Compare to the original "trained-on-all-21-days" composite C3 result

Shows: how much of our yesterday P&L was lookahead bias vs real signal?
"""
from __future__ import annotations

import json
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.covariance import EllipticEnvelope
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

REPO = Path("/Users/weeoto/turbo-carnival")
SVXY_RESULTS = REPO / "experiments" / "svxy_lab" / "runs" / "walk_forward_results.parquet"
BARS = REPO / "experiments" / "runs" / "valk_hft_demo_20260511T140548Z" / "minute_bars.parquet"
ML_ZOO_PER_ENTRY = REPO / "experiments" / "ml_lab" / "runs" / "ml_zoo_per_entry_latest.parquet"
ORIG_COMPOSITE = REPO / "experiments" / "ml_lab" / "runs" / "composite_backtest" / "winner_per_entry.parquet"
OUT = REPO / "reports" / "no_lookahead_yesterday"
OUT.mkdir(parents=True, exist_ok=True)

LR_FEATS = ["ret1", "ret3", "ret10", "range_pct", "volume_z", "ret1_z",
             "vwap_dist", "tod_sin", "tod_cos", "vov"]
FORWARD_BARS = 5
THRESHOLD_BPS = 30.0


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values(["symbol", "t"]).reset_index(drop=True)
    df["t_ny"] = pd.to_datetime(df["t_ny"], errors="coerce")
    mins = ((df["t_ny"].dt.hour - 9) * 60 + df["t_ny"].dt.minute - 30).clip(0, 390)
    df["tod_sin"] = np.sin(2 * np.pi * mins / 390)
    df["tod_cos"] = np.cos(2 * np.pi * mins / 390)

    def vov(g):
        s = g["ret1"].rolling(5).std()
        m = g["ret1"].abs().rolling(20).median()
        return (s / m).fillna(0).clip(0, 50)
    df["vov"] = df.groupby("symbol", group_keys=False).apply(vov, include_groups=False)
    df["fwd_max_abs_ret"] = (
        df.groupby("symbol")["close"].pct_change(FORWARD_BARS).shift(-FORWARD_BARS).abs()
    )
    df["y"] = (df["fwd_max_abs_ret"] * 10000.0 > THRESHOLD_BPS).astype(int)
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df["date"] = df["t"].dt.date
    return df


def main():
    print("=" * 70)
    print("No-lookahead yesterday test — retrain composite C3 without yesterday's data")
    print("=" * 70)

    # === SVXY: extract yesterday's slice from existing walk-forward ===
    print("\n[1/2] SVXY rotation — extracting yesterday's slice...")
    svxy = pd.read_parquet(SVXY_RESULTS)
    svxy["date"] = pd.to_datetime(svxy["date"])
    yesterday_svxy = svxy[svxy["date"].dt.date == pd.Timestamp("2026-05-11").date()]
    if yesterday_svxy.empty:
        # Use last available date as proxy
        yesterday_svxy = svxy.iloc[[-1]]
    yest_row = yesterday_svxy.iloc[0]
    print(f"  date: {yest_row['date'].strftime('%Y-%m-%d')}")
    print(f"  VIX: {yest_row['vix']:.2f}  VIX3M: {yest_row['vix3m']:.2f}  ratio: {yest_row['ratio']:.3f}")
    print(f"  Signal (was known at prior close): {'LONG' if yest_row['signal'] else 'FLAT'}")
    print(f"  SVXY return: {yest_row['svxy_ret']*100:+.2f}%")
    print(f"  Strategy return: {yest_row['strategy_ret']*100:+.2f}%")
    print(f"  Strategy equity going INTO yesterday: ${svxy[svxy['date'] < yest_row['date']].iloc[-1]['strategy_equity']:,.2f}" if len(svxy[svxy['date'] < yest_row['date']]) > 0 else "  N/A")

    # === Composite C3: retrain on days 1-20 only ===
    print("\n[2/2] Composite C3 gate — retraining 4 sub-models WITHOUT yesterday's data...")
    bars = pd.read_parquet(BARS)
    bars["t"] = pd.to_datetime(bars["t"], utc=True)
    bars = enrich(bars)
    bars = bars.dropna(subset=LR_FEATS + ["y"]).reset_index(drop=True)
    yest_date = pd.Timestamp("2026-05-11").date()
    train_mask = bars["date"] != yest_date
    print(f"  training rows (excluding yesterday): {train_mask.sum():,}")
    X_train = bars.loc[train_mask, LR_FEATS].values.astype(np.float32)
    y_train = bars.loc[train_mask, "y"].values.astype(np.int8)

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)

    # Train 4 sub-models on T-1 data
    print("  fitting m016 LOF...")
    m016 = LocalOutlierFactor(n_neighbors=30, novelty=True, n_jobs=-1).fit(X_train_s)
    sc016 = -m016.score_samples(X_train_s)
    p95_016 = float(np.quantile(sc016, 0.95))

    print("  fitting m017 OCSVM (5000 subsample)...")
    rng = np.random.default_rng(42)
    sub = rng.choice(len(X_train_s), min(5000, len(X_train_s)), replace=False)
    m017 = OneClassSVM(kernel="rbf", gamma="scale", nu=0.1).fit(X_train_s[sub])
    sc017 = -m017.score_samples(X_train_s)
    p95_017 = float(np.quantile(sc017, 0.95))

    print("  fitting m018 EllipticEnvelope...")
    m018 = EllipticEnvelope(contamination=0.1, support_fraction=0.9, random_state=42).fit(X_train_s)
    sc018 = -m018.score_samples(X_train_s)
    p95_018 = float(np.quantile(sc018, 0.95))

    print("  fitting m102 RBFSampler + LogReg...")
    m102 = Pipeline([
        ("scaler", StandardScaler()),
        ("rbf", RBFSampler(gamma=0.5, n_components=64, random_state=42)),
        ("clf", LogisticRegression(class_weight="balanced", max_iter=2000)),
    ]).fit(X_train, y_train)

    # === Apply retrained models to yesterday's 162 entries ===
    print("\n  applying to yesterday's 162 entries (need today's enriched bars)...")
    # The entry features were computed from the SAME 21-day bars; we need
    # the feature vector at each entry's exact (symbol, minute) timestamp.
    # The ml_zoo_per_entry already has entries — match against today's bars.
    import requests
    import os
    for line in (REPO / "council" / ".env.local").read_text().splitlines():
        line = line.strip()
        if line and "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            if k.strip() not in os.environ:
                os.environ[k.strip()] = v.strip().strip('"').strip("'")
    H = {"APCA-API-KEY-ID": os.environ["ALPACA_AEGIS_KEY"],
         "APCA-API-SECRET-KEY": os.environ["ALPACA_AEGIS_SECRET"]}

    # The 21-day cache cuts off PRE-MARKET yesterday (only 245 rows = first hour).
    # ALWAYS fetch fresh — the entries happened DURING market hours.
    yest_bars = bars[bars["date"] == yest_date]
    print(f"  yesterday bars in cache (pre-market only): {len(yest_bars)}")
    if True:  # ALWAYS fetch fresh — cache is incomplete for yesterday
        # Fetch fresh
        rows = []
        for sym in ("SPY", "QQQ", "AAPL", "AMD", "META", "NVDA", "TSLA"):
            r = requests.get("https://data.alpaca.markets/v2/stocks/bars", headers=H,
                              params={"symbols": sym, "timeframe": "1Min",
                                       "start": "2026-05-11T13:30:00Z",
                                       "end": "2026-05-11T20:00:00Z",
                                       "limit": 10000, "adjustment": "raw"}, timeout=20)
            for b in r.json().get("bars", {}).get(sym, []):
                rows.append({"symbol": sym, "t": b["t"], "open": b["o"], "high": b["h"],
                              "low": b["l"], "close": b["c"], "volume": b["v"]})
        fresh = pd.DataFrame(rows)
        fresh["t"] = pd.to_datetime(fresh["t"], utc=True)
        fresh["t_ny"] = fresh["t"].dt.tz_convert("America/New_York")
        fresh["date"] = fresh["t"].dt.date
        fresh = fresh.sort_values(["symbol", "t"]).reset_index(drop=True)
        fresh["ret1"] = fresh.groupby("symbol")["close"].pct_change(1).fillna(0)
        fresh["ret3"] = fresh.groupby("symbol")["close"].pct_change(3).fillna(0)
        fresh["ret10"] = fresh.groupby("symbol")["close"].pct_change(10).fillna(0)
        fresh["range_pct"] = (fresh["high"] - fresh["low"]) / fresh["close"]
        fresh["volume_z"] = fresh.groupby("symbol")["volume"].transform(
            lambda s: (s - s.rolling(20, min_periods=5).mean()) / s.rolling(20, min_periods=5).std()
        ).fillna(0)
        fresh["ret1_z"] = fresh.groupby("symbol")["ret1"].transform(
            lambda s: (s - s.rolling(20, min_periods=5).mean()) / s.rolling(20, min_periods=5).std()
        ).fillna(0)
        fresh["session_vwap"] = fresh.groupby(["symbol", "date"]).apply(
            lambda g: (g["close"] * g["volume"]).cumsum() / g["volume"].cumsum().replace(0, np.nan),
            include_groups=False,
        ).reset_index(level=[0, 1], drop=True)
        fresh["vwap_dist"] = (fresh["close"] - fresh["session_vwap"]) / fresh["session_vwap"]
        mins = ((fresh["t_ny"].dt.hour - 9) * 60 + fresh["t_ny"].dt.minute - 30).clip(0, 390)
        fresh["tod_sin"] = np.sin(2 * np.pi * mins / 390)
        fresh["tod_cos"] = np.cos(2 * np.pi * mins / 390)
        def vov(g):
            s = g["ret1"].rolling(5).std()
            m = g["ret1"].abs().rolling(20).median()
            return (s / m).fillna(0).clip(0, 50)
        fresh["vov"] = fresh.groupby("symbol", group_keys=False).apply(vov, include_groups=False)
        yest_bars = fresh
        print(f"  fetched fresh yesterday bars: {len(yest_bars)}")

    # Load yesterday's matched entries
    per = pd.read_parquet(ML_ZOO_PER_ENTRY)
    entries = per[per["model"] == per["model"].iloc[0]].copy().reset_index(drop=True)
    entries["entry_ts"] = pd.to_datetime(entries["entry_ts"], utc=True)
    entries["entry_minute"] = entries["entry_ts"].dt.floor("1min")

    yb = yest_bars.copy()
    yb["minute"] = yb["t"].dt.floor("1min")
    yb = yb.set_index(["symbol", "minute"])

    blocked_new = []
    p102_list = []
    n_each_blocker = {"016": 0, "017": 0, "018": 0, "102": 0}
    for _, e in entries.iterrows():
        try:
            row = yb.loc[(e["symbol"], e["entry_minute"])]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            x = np.array([[float(row[k]) for k in LR_FEATS]])
            x_scaled = scaler.transform(x)
            s016 = -float(m016.score_samples(x_scaled)[0])
            s017 = -float(m017.score_samples(x_scaled)[0])
            s018 = -float(m018.score_samples(x_scaled)[0])
            p102 = float(m102.predict_proba(x)[0, 1])
            p102_list.append(p102)
            block_016 = s016 > p95_016
            block_017 = s017 > p95_017
            block_018 = s018 > p95_018
            block_102 = p102 > 0.5
            if block_016: n_each_blocker["016"] += 1
            if block_017: n_each_blocker["017"] += 1
            if block_018: n_each_blocker["018"] += 1
            if block_102: n_each_blocker["102"] += 1
            blocked_new.append(block_016 or block_017 or block_018 or block_102)
        except (KeyError, ValueError):
            blocked_new.append(False)

    entries["blocked_no_lookahead"] = blocked_new
    pnl_kept_no_la = entries.loc[~entries["blocked_no_lookahead"], "pnl"].sum()
    actual_pnl = entries["pnl"].sum()
    delta_no_la = pnl_kept_no_la - actual_pnl
    n_blocked_no_la = sum(blocked_new)

    # Compare to original lookahead version
    orig = pd.read_parquet(ORIG_COMPOSITE)
    n_blocked_orig = int(orig["blocked"].sum())
    pnl_kept_orig = orig.loc[~orig["blocked"], "pnl"].sum()
    delta_orig = pnl_kept_orig - actual_pnl

    print(f"\n=== Comparison ===")
    print(f"Actual bar-close P&L (no gate):        ${actual_pnl:+,.2f}")
    print(f"")
    print(f"WITH lookahead bias (original C3):     ${pnl_kept_orig:+,.2f}  (blocked {n_blocked_orig}, Δ ${delta_orig:+,.2f})")
    print(f"WITHOUT lookahead (retrained C3):      ${pnl_kept_no_la:+,.2f}  (blocked {n_blocked_no_la}, Δ ${delta_no_la:+,.2f})")
    print(f"")
    print(f"Lookahead premium: ${delta_orig - delta_no_la:+,.2f}")
    print(f"  (the original ${delta_orig:+,.2f} number was inflated by ${delta_orig - delta_no_la:+,.2f} from lookahead)")
    print(f"")
    print(f"Per-blocker votes (out of 162):")
    for sub, n in n_each_blocker.items():
        print(f"  {sub}: blocked {n}")

    # Render HTML
    delta_lookahead = delta_orig - delta_no_la
    summary_text = (
        f"Composite C3 (retrained without yesterday): blocked {n_blocked_no_la}, Δ ${delta_no_la:+,.2f}<br>"
        f"Composite C3 (original, with lookahead): blocked {n_blocked_orig}, Δ ${delta_orig:+,.2f}<br>"
        f"Lookahead premium: ${delta_lookahead:+,.2f} (how much the original number was inflated)"
    )

    svxy_summary = (
        f"Date: {yest_row['date'].strftime('%Y-%m-%d')}<br>"
        f"VIX: {yest_row['vix']:.2f}  VIX3M: {yest_row['vix3m']:.2f}  ratio: {yest_row['ratio']:.3f}<br>"
        f"Signal (decided at prior close, NO lookahead): {'LONG SVXY' if yest_row['signal'] else 'FLAT'}<br>"
        f"SVXY return that day: {yest_row['svxy_ret']*100:+.2f}%<br>"
        f"Strategy return for the day: {yest_row['strategy_ret']*100:+.2f}%"
    )

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>No-lookahead yesterday test</title>
<style>
body {{ background:#0b1120; color:#e2e8f0; font-family:-apple-system,sans-serif;
       margin:0; padding:24px 32px; line-height:1.55; }}
h1 {{ color:#f8fafc; margin:0 0 8px 0; }}
h2 {{ color:#f1f5f9; font-size:18px; margin:24px 0 10px 0; }}
.sub {{ color:#94a3b8; font-size:14px; margin-bottom:18px; }}
.card {{ background:#0f172a; border:1px solid #1e293b; padding:18px 22px;
         border-radius:12px; margin:14px 0; }}
.win {{ border-left: 4px solid #4ade80; }}
.alert {{ border-left: 4px solid #facc15; }}
.kpis {{ display:flex; gap:12px; flex-wrap:wrap; margin:10px 0; }}
.kpi {{ background:#1e293b; padding:10px 14px; border-radius:8px; min-width:140px; }}
.kpi .l {{ color:#94a3b8; font-size:11px; text-transform:uppercase; }}
.kpi .v {{ color:#f1f5f9; font-size:20px; font-weight:700; margin-top:2px;
          font-variant-numeric:tabular-nums; }}
.pos {{ color:#4ade80; }} .neg {{ color:#f87171; }}
table {{ border-collapse:collapse; width:100%; font-size:13px;
         font-variant-numeric:tabular-nums; }}
th {{ background:#1e293b; color:#cbd5e1; padding:8px 12px; text-align:left; }}
td {{ padding:6px 12px; border-bottom:1px solid #1e293b; }}
tr.lookahead {{ background:rgba(250,204,21,0.08); }}
tr.honest {{ background:rgba(34,197,94,0.08); }}
code {{ background:#1e293b; padding:2px 6px; border-radius:4px; font-size:12px; }}
.note {{ background:#0f172a; border-left:3px solid #60a5fa; padding:10px 14px;
         margin:12px 0; font-size:13px; color:#cbd5e1; border-radius:0 8px 8px 0; }}
</style></head><body>
<h1>No-lookahead test — yesterday only, on previous training</h1>
<div class="sub">Two questions answered: (1) what did the proper walk-forward
SVXY strategy decide for yesterday? (2) how much of the composite C3 +$2,030
"win" was lookahead bias?</div>

<h2>1. SVXY rotation — yesterday's decision (walk-forward, no lookahead)</h2>
<div class="card win">
{svxy_summary}
</div>

<h2>2. Composite C3 — retrained without yesterday's data</h2>
<div class="card alert">
{summary_text}
</div>

<h2>Side-by-side</h2>
<table>
<thead><tr>
  <th>Approach</th><th>Training data</th>
  <th>Blocked</th><th>Counterfactual P&L</th><th>Δ vs actual ($+496)</th>
</tr></thead>
<tbody>
<tr><td>No gate (baseline)</td><td>—</td>
  <td>0</td><td>${actual_pnl:+,.2f}</td><td>—</td></tr>
<tr class="lookahead"><td>Composite C3 (ORIGINAL)</td><td>All 21 days (includes yesterday)</td>
  <td>{n_blocked_orig}</td><td>${pnl_kept_orig:+,.2f}</td><td>${delta_orig:+,.2f}</td></tr>
<tr class="honest"><td><strong>Composite C3 (HONEST — retrained)</strong></td><td>Days 1-20 only (no yesterday)</td>
  <td>{n_blocked_no_la}</td><td>${pnl_kept_no_la:+,.2f}</td><td><strong>${delta_no_la:+,.2f}</strong></td></tr>
</tbody></table>

<h2>Per-sub-model blocking (retrained on prior 20 days)</h2>
<table>
<thead><tr><th>Sub-model</th><th>Blocked entries</th><th>% of 162</th></tr></thead>
<tbody>
<tr><td>016 LOF (local outlier factor)</td><td>{n_each_blocker['016']}</td><td>{n_each_blocker['016']/162*100:.1f}%</td></tr>
<tr><td>017 OneClassSVM</td><td>{n_each_blocker['017']}</td><td>{n_each_blocker['017']/162*100:.1f}%</td></tr>
<tr><td>018 Elliptic Envelope</td><td>{n_each_blocker['018']}</td><td>{n_each_blocker['018']/162*100:.1f}%</td></tr>
<tr><td>102 RBF-sampler + LogReg</td><td>{n_each_blocker['102']}</td><td>{n_each_blocker['102']/162*100:.1f}%</td></tr>
</tbody></table>

<div class="note">
<strong>What this means:</strong>
<br>• If the lookahead premium is small (&lt; $200), the original +$2,030 number was mostly real signal.
<br>• If it's large (&gt; $500), much of the apparent edge came from training on yesterday's data.
<br>• Either way, this number is the realistic deployment expectation under proper walk-forward.
<br>• Today's lookahead premium: <strong>${delta_lookahead:+,.2f}</strong> — this is the bias we measured.
</div>

<div class="sub" style="margin-top:24px;">
Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ·
<code>experiments/ml_lab/21_no_lookahead_yesterday.py</code></div>
</body></html>"""

    (OUT / "index.html").write_text(html)

    # Save summary JSON for programmatic access
    summary = {
        "yesterday_date": str(yest_date),
        "svxy_decision": {
            "vix": float(yest_row["vix"]),
            "vix3m": float(yest_row["vix3m"]),
            "ratio": float(yest_row["ratio"]),
            "signal": "long" if yest_row["signal"] else "flat",
            "svxy_ret_pct": float(yest_row["svxy_ret"] * 100),
            "strategy_ret_pct": float(yest_row["strategy_ret"] * 100),
        },
        "composite_c3_with_lookahead": {
            "n_blocked": int(n_blocked_orig),
            "counterfactual_pnl": float(pnl_kept_orig),
            "delta_vs_actual": float(delta_orig),
        },
        "composite_c3_no_lookahead": {
            "n_blocked": int(n_blocked_no_la),
            "counterfactual_pnl": float(pnl_kept_no_la),
            "delta_vs_actual": float(delta_no_la),
            "per_blocker_votes": n_each_blocker,
        },
        "lookahead_premium_usd": float(delta_lookahead),
        "actual_bar_close_pnl": float(actual_pnl),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nwrote {OUT}/index.html + summary.json")
    subprocess.run(["open", f"file://{OUT}/index.html"], check=False)


if __name__ == "__main__":
    main()
