"""Build one final local dataset from all cached/manual data sources.

This is intentionally non-destructive: raw files under data/cache and
data/offline_context are left untouched. Outputs are written to
data/final_dataset/.

Main outputs:
  - model_daily_panel.parquet: symbol/date training panel with prices, IV,
    option volume, earnings flags, priced-chain aggregates, macro context,
    and forward-return labels.
  - spy_training_panel.parquet: SPY-only wide panel for strategy simulation.
  - priced_chain_legs.parquet: concatenated priced option legs from year1/year2.
  - earnings_events.parquet: concatenated earnings dates.
  - executed_trades.parquet / executed_trade_legs.parquet: local DB evidence.
  - manifest.json: source counts, row counts, date ranges, column list.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data"
DEFAULT_OUT_DIR = DEFAULT_DATA_DIR / "final_dataset"

TRAIN_START = pd.Timestamp("2024-05-07")
TRAIN_END = pd.Timestamp("2025-05-06")
EVAL_START = pd.Timestamp("2025-05-07")
EVAL_END = pd.Timestamp("2026-05-06")

YF_CONTEXT_SYMBOLS = [
    "SPY",
    "QQQ",
    "IWM",
    "DIA",
    "TLT",
    "IEF",
    "SHY",
    "HYG",
    "LQD",
    "UUP",
    "GLD",
    "USO",
    "VIXY",
    "SVXY",
    "XLF",
    "XLK",
    "XLE",
    "XLV",
    "XLY",
    "XLP",
    "XLI",
    "XLU",
    "XLB",
    "XLRE",
    "SMH",
    "KRE",
    "IDX_VIX",
    "IDX_VIX3M",
    "IDX_VVIX",
    "IDX_VXN",
    "IDX_MOVE",
]


def norm_date(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce").dt.tz_localize(None).dt.normalize()


def classify_split(date_series: pd.Series) -> pd.Series:
    dates = pd.to_datetime(date_series)
    out = pd.Series("pretrain_or_context", index=date_series.index, dtype="object")
    out[(dates >= TRAIN_START) & (dates <= TRAIN_END)] = "train_year1"
    out[(dates >= EVAL_START) & (dates <= EVAL_END)] = "eval_year2"
    out[dates > EVAL_END] = "post_eval"
    return out


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False, compression="zstd")
    except Exception:
        df.to_parquet(path, index=False, compression="snappy")


def read_parquet_safe(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception as exc:
        print(f"  WARN failed to read {path}: {type(exc).__name__}: {exc}")
        return None


def load_yfinance_daily(data_dir: Path) -> pd.DataFrame:
    root = data_dir / "offline_context" / "yfinance" / "daily"
    rows = []
    for path in sorted(root.glob("*.parquet")):
        df = read_parquet_safe(path)
        if df is None or df.empty or "date" not in df.columns:
            continue
        sym = path.stem.upper()
        df = df.copy()
        df["symbol"] = sym
        df["date"] = norm_date(df["date"])
        keep = ["symbol", "date", "open", "high", "low", "close", "volume"]
        df = df[[c for c in keep if c in df.columns]]
        df["price_source"] = "offline_yfinance"
        df["price_source_priority"] = 1
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_cache_daily(data_dir: Path) -> pd.DataFrame:
    rows = []
    for year in ("year1", "year2"):
        root = data_dir / "cache" / year
        for path in sorted(root.glob("*_daily.parquet")):
            if path.name == "spy_iv_daily.parquet":
                continue
            df = read_parquet_safe(path)
            if df is None or df.empty:
                continue
            sym = path.name.replace("_daily.parquet", "").upper()
            df = df.reset_index().copy()
            if "date" not in df.columns:
                continue
            df["symbol"] = sym
            df["date"] = norm_date(df["date"])
            keep = [
                "symbol",
                "date",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "vwap",
                "transactions",
            ]
            df = df[[c for c in keep if c in df.columns]]
            df["price_source"] = f"cache_{year}"
            df["price_source_priority"] = 2
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def build_price_panel(data_dir: Path) -> pd.DataFrame:
    print("Loading price panels...")
    prices = pd.concat(
        [load_yfinance_daily(data_dir), load_cache_daily(data_dir)],
        ignore_index=True,
    )
    if prices.empty:
        return prices
    prices = prices.dropna(subset=["symbol", "date"]).copy()
    prices = prices.sort_values(["symbol", "date", "price_source_priority"])
    prices = prices.drop_duplicates(["symbol", "date"], keep="last")
    prices = prices.drop(columns=["price_source_priority"])
    prices["dataset_split"] = classify_split(prices["date"])
    prices = prices.sort_values(["symbol", "date"]).reset_index(drop=True)

    for col in ("open", "high", "low", "close", "volume", "vwap", "transactions"):
        if col in prices.columns:
            prices[col] = pd.to_numeric(prices[col], errors="coerce")

    g = prices.groupby("symbol", group_keys=False)
    prices["ret_1d"] = g["close"].pct_change()
    prices["ret_5d"] = g["close"].pct_change(5)
    prices["ret_20d"] = g["close"].pct_change(20)
    prices["fwd_ret_1d"] = g["close"].shift(-1) / prices["close"] - 1.0
    prices["fwd_ret_5d"] = g["close"].shift(-5) / prices["close"] - 1.0
    prices["fwd_ret_20d"] = g["close"].shift(-20) / prices["close"] - 1.0
    prices["rv_10d"] = g["ret_1d"].rolling(10, min_periods=5).std().reset_index(level=0, drop=True) * np.sqrt(252)
    prices["rv_20d"] = g["ret_1d"].rolling(20, min_periods=10).std().reset_index(level=0, drop=True) * np.sqrt(252)
    return prices


def load_hf_iv(data_dir: Path) -> pd.DataFrame:
    print("Loading HuggingFace IV panel...")
    path = data_dir / "offline_context" / "huggingface_options_iv_sp500" / "processed" / "data_IV_USA.parquet"
    if not path.exists():
        return pd.DataFrame()
    cols = [
        "symbol",
        "date",
        "strikes_spread",
        "calls_contracts_traded",
        "puts_contracts_traded",
        "calls_open_interest",
        "puts_open_interest",
        "expirations_number",
        "DITM_IV",
        "ITM_IV",
        "sITM_IV",
        "ATM_IV",
        "sOTM_IV",
        "OTM_IV",
        "DOTM_IV",
        "contracts_number",
        "hv_20",
        "hv_40",
        "hv_60",
        "hv_75",
        "hv_90",
        "hv_120",
        "hv_180",
        "hv_200",
        "VIX",
    ]
    df = pd.read_parquet(path, columns=cols)
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["date"] = norm_date(df["date"])
    rename = {c: f"hf_{c}" for c in cols if c not in ("symbol", "date")}
    df = df.rename(columns=rename)
    for c in ("hf_ATM_IV", "hf_hv_20", "hf_hv_40", "hf_hv_60", "hf_hv_90", "hf_VIX"):
        if c in df.columns:
            df[f"{c}_decimal"] = pd.to_numeric(df[c], errors="coerce") / 100.0
    df["hf_put_call_oi_ratio"] = (
        pd.to_numeric(df.get("hf_puts_open_interest"), errors="coerce")
        / pd.to_numeric(df.get("hf_calls_open_interest"), errors="coerce").replace(0, np.nan)
    )
    df["hf_put_call_volume_ratio"] = (
        pd.to_numeric(df.get("hf_puts_contracts_traded"), errors="coerce")
        / pd.to_numeric(df.get("hf_calls_contracts_traded"), errors="coerce").replace(0, np.nan)
    )
    return df.drop_duplicates(["symbol", "date"], keep="last")


def load_fred_macro(data_dir: Path) -> pd.DataFrame:
    print("Loading FRED macro panel...")
    root = data_dir / "offline_context" / "fred"
    rows = []
    for path in sorted(root.glob("*.parquet")):
        df = read_parquet_safe(path)
        if df is None or df.empty or not {"date", "value", "series_id"}.issubset(df.columns):
            continue
        rows.append(df[["date", "value", "series_id"]])
    if not rows:
        return pd.DataFrame(columns=["date"])
    long = pd.concat(rows, ignore_index=True)
    long["date"] = norm_date(long["date"])
    long["value"] = pd.to_numeric(long["value"], errors="coerce")
    wide = long.pivot_table(index="date", columns="series_id", values="value", aggfunc="last").reset_index()
    wide.columns = ["date"] + [f"fred_{str(c).lower()}" for c in wide.columns[1:]]
    wide = wide.sort_values("date")
    # Daily market panel: forward fill weekly/monthly macro values.
    return wide.ffill()


def aggregate_cboe_volume(data_dir: Path, out_dir: Path, force: bool = False) -> pd.DataFrame:
    print("Aggregating CBOE options volume...")
    cache_path = out_dir / "_cache_cboe_volume_by_symbol_date.parquet"
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)
    root = data_dir / "offline_context" / "cboe_options_volume" / "raw"
    pieces = []
    for path in sorted(root.glob("daily_volume_*.csv")):
        print(f"  CBOE {path.name}")
        chunks = []
        for chunk in pd.read_csv(path, chunksize=500_000):
            if not {"Trade Date", "Underlying", "Volume"}.issubset(chunk.columns):
                continue
            chunk = chunk[["Trade Date", "Underlying", "Volume"]].copy()
            chunk["date"] = norm_date(chunk["Trade Date"])
            chunk["symbol"] = chunk["Underlying"].astype(str).str.upper()
            chunk["Volume"] = pd.to_numeric(chunk["Volume"], errors="coerce").fillna(0)
            agg = chunk.groupby(["symbol", "date"], as_index=False)["Volume"].sum()
            chunks.append(agg)
        if chunks:
            yearly = pd.concat(chunks, ignore_index=True)
            yearly = yearly.groupby(["symbol", "date"], as_index=False)["Volume"].sum()
            pieces.append(yearly)
    if not pieces:
        return pd.DataFrame(columns=["symbol", "date", "cboe_option_volume"])
    out = pd.concat(pieces, ignore_index=True)
    out = out.groupby(["symbol", "date"], as_index=False)["Volume"].sum()
    out = out.rename(columns={"Volume": "cboe_option_volume"})
    write_parquet(out, cache_path)
    return out


def load_earnings_events(data_dir: Path) -> pd.DataFrame:
    print("Loading earnings events...")
    rows = []
    for year in ("year1", "year2"):
        root = data_dir / "cache" / year / "earnings_dates"
        for path in sorted(root.glob("*.parquet")):
            df = read_parquet_safe(path)
            if df is None or df.empty or "earnings_date" not in df.columns:
                continue
            sym = path.stem.upper()
            d = pd.DataFrame({"symbol": sym, "date": norm_date(df["earnings_date"]), "earnings_source": f"cache_{year}"})
            rows.append(d)
    if not rows:
        return pd.DataFrame(columns=["symbol", "date", "earnings_source"])
    out = pd.concat(rows, ignore_index=True).dropna(subset=["date"])
    out = out.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"])
    return out


def load_cache_spy_iv(data_dir: Path) -> pd.DataFrame:
    rows = []
    for year in ("year1", "year2"):
        path = data_dir / "cache" / year / "spy_iv_daily.parquet"
        df = read_parquet_safe(path)
        if df is None or df.empty or "date" not in df.columns:
            continue
        df = df.copy()
        df["symbol"] = "SPY"
        df["date"] = norm_date(df["date"])
        df = df.rename(columns={"atm_iv": "cache_spy_atm_iv", "available": "cache_spy_iv_available"})
        df["cache_spy_iv_source"] = f"cache_{year}"
        rows.append(df[["symbol", "date", "cache_spy_atm_iv", "cache_spy_iv_available", "cache_spy_iv_source"]])
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).drop_duplicates(["symbol", "date"], keep="last")


def load_priced_chains(data_dir: Path) -> pd.DataFrame:
    print("Loading priced option-chain legs...")
    rows = []
    for year in ("year1", "year2"):
        root = data_dir / "cache" / year / "chains_priced"
        for path in sorted(root.glob("*/*.parquet")):
            df = read_parquet_safe(path)
            if df is None or df.empty:
                continue
            df = df.copy()
            df["cache_year"] = year
            df["source_path"] = str(path.relative_to(REPO_ROOT))
            rows.append(df)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    for col in ("as_of_date", "event_date", "expiration_date"):
        if col in out.columns:
            out[col] = norm_date(out[col])
    if "underlying_ticker" in out.columns:
        out["symbol"] = out["underlying_ticker"].astype(str).str.upper()
    return out


def aggregate_priced_chains(chain: pd.DataFrame) -> pd.DataFrame:
    if chain.empty:
        return pd.DataFrame(columns=["symbol", "date"])
    df = chain.copy()
    df["date"] = df["event_date"]
    df["mid"] = pd.to_numeric(df["mid"], errors="coerce")
    df["atm_iv"] = pd.to_numeric(df["atm_iv"], errors="coerce")
    df["underlying_close"] = pd.to_numeric(df["underlying_close"], errors="coerce")
    entry = df[df["timing"].astype(str).str.lower() == "entry"]
    exit_ = df[df["timing"].astype(str).str.lower() == "exit"]
    base = df.groupby(["symbol", "date"], as_index=False).agg(
        priced_chain_rows=("ticker", "count"),
        priced_chain_contracts=("ticker", "nunique"),
        priced_atm_iv=("atm_iv", "mean"),
    )
    if not entry.empty:
        e = entry.groupby(["symbol", "date"], as_index=False).agg(
            priced_entry_legs=("ticker", "count"),
            priced_entry_mid_sum=("mid", "sum"),
            priced_entry_underlying_close=("underlying_close", "mean"),
        )
        base = base.merge(e, on=["symbol", "date"], how="left")
    if not exit_.empty:
        x = exit_.groupby(["symbol", "date"], as_index=False).agg(
            priced_exit_legs=("ticker", "count"),
            priced_exit_mid_sum=("mid", "sum"),
            priced_exit_underlying_close=("underlying_close", "mean"),
        )
        base = base.merge(x, on=["symbol", "date"], how="left")
    return base


def load_context_wide(yf_prices: pd.DataFrame, macro: pd.DataFrame) -> pd.DataFrame:
    close = yf_prices[yf_prices["symbol"].isin(YF_CONTEXT_SYMBOLS)][["date", "symbol", "close"]].copy()
    if close.empty:
        context = pd.DataFrame(columns=["date"])
    else:
        context = close.pivot_table(index="date", columns="symbol", values="close", aggfunc="last").reset_index()
        context.columns = ["date"] + [f"ctx_{str(c).lower()}_close" for c in context.columns[1:]]
        for c in list(context.columns):
            if c == "date" or not c.endswith("_close"):
                continue
            context[c.replace("_close", "_ret_1d")] = context[c].pct_change()
            context[c.replace("_close", "_ret_5d")] = context[c].pct_change(5)
    if macro.empty:
        return context
    return context.merge(macro, on="date", how="outer").sort_values("date").ffill()


def export_db_tables(data_dir: Path, out_dir: Path) -> dict[str, int]:
    db = data_dir / "trading_system.db"
    counts: dict[str, int] = {}
    if not db.exists():
        return counts
    conn = sqlite3.connect(db)
    for table, out_name in (
        ("trades", "executed_trades.parquet"),
        ("trade_legs", "executed_trade_legs.parquet"),
        ("runs", "executed_runs.parquet"),
        ("nav_history", "executed_nav_history.parquet"),
        ("gate_decisions", "executed_gate_decisions.parquet"),
    ):
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
        except Exception:
            continue
        write_parquet(df, out_dir / out_name)
        counts[table] = len(df)
    conn.close()
    return counts


def build_dataset(data_dir: Path, out_dir: Path, force_cboe: bool = False) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    yf_prices = load_yfinance_daily(data_dir)
    prices = build_price_panel(data_dir)
    hf_iv = load_hf_iv(data_dir)
    macro = load_fred_macro(data_dir)
    cboe = aggregate_cboe_volume(data_dir, out_dir, force=force_cboe)
    earnings = load_earnings_events(data_dir)
    spy_iv = load_cache_spy_iv(data_dir)
    chains = load_priced_chains(data_dir)
    chain_agg = aggregate_priced_chains(chains)
    context = load_context_wide(yf_prices, macro)

    print("Merging final symbol/date panel...")
    # Start from the widest symbol/date source so manually downloaded HF IV rows
    # are preserved even when no local OHLCV file exists for that symbol.
    if not hf_iv.empty:
        panel = hf_iv.merge(prices, on=["symbol", "date"], how="outer")
    else:
        panel = prices.copy()
    for extra in (cboe, spy_iv, chain_agg):
        if extra is not None and not extra.empty:
            panel = panel.merge(extra, on=["symbol", "date"], how="left")
    if not earnings.empty:
        e = earnings.copy()
        e["has_earnings_event"] = 1
        panel = panel.merge(e[["symbol", "date", "has_earnings_event"]], on=["symbol", "date"], how="left")
    else:
        panel["has_earnings_event"] = np.nan
    panel["has_earnings_event"] = panel["has_earnings_event"].fillna(0).astype("int8")
    if "dataset_split" not in panel.columns:
        panel["dataset_split"] = classify_split(panel["date"])
    else:
        panel["dataset_split"] = panel["dataset_split"].fillna(classify_split(panel["date"]))

    if not context.empty:
        panel = panel.merge(context, on="date", how="left")

    panel = panel.sort_values(["symbol", "date"]).reset_index(drop=True)

    # SPY panel for simulations; keep the full date-wide context and SPY-specific
    # symbol features in one table.
    spy_panel = panel[panel["symbol"] == "SPY"].copy()

    print("Writing final dataset files...")
    write_parquet(panel, out_dir / "model_daily_panel.parquet")
    write_parquet(spy_panel, out_dir / "spy_training_panel.parquet")
    write_parquet(context, out_dir / "market_context_daily.parquet")
    write_parquet(macro, out_dir / "macro_fred_daily.parquet")
    write_parquet(cboe, out_dir / "cboe_option_volume_daily.parquet")
    write_parquet(earnings, out_dir / "earnings_events.parquet")
    write_parquet(chains, out_dir / "priced_chain_legs.parquet")
    write_parquet(chain_agg, out_dir / "priced_chain_daily_agg.parquet")
    db_counts = export_db_tables(data_dir, out_dir)

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(data_dir),
        "out_dir": str(out_dir),
        "split_definition": {
            "train_year1": [str(TRAIN_START.date()), str(TRAIN_END.date())],
            "eval_year2": [str(EVAL_START.date()), str(EVAL_END.date())],
        },
        "row_counts": {
            "price_panel": int(len(prices)),
            "hf_iv_panel": int(len(hf_iv)),
            "macro_fred_daily": int(len(macro)),
            "cboe_volume_by_symbol_date": int(len(cboe)),
            "earnings_events": int(len(earnings)),
            "priced_chain_legs": int(len(chains)),
            "priced_chain_daily_agg": int(len(chain_agg)),
            "model_daily_panel": int(len(panel)),
            "spy_training_panel": int(len(spy_panel)),
            **{f"db_{k}": int(v) for k, v in db_counts.items()},
        },
        "date_ranges": {
            "model_daily_panel": [
                str(panel["date"].min().date()) if len(panel) else None,
                str(panel["date"].max().date()) if len(panel) else None,
            ],
            "spy_training_panel": [
                str(spy_panel["date"].min().date()) if len(spy_panel) else None,
                str(spy_panel["date"].max().date()) if len(spy_panel) else None,
            ],
        },
        "columns": {
            "model_daily_panel": list(panel.columns),
            "spy_training_panel": list(spy_panel.columns),
        },
        "source_notes": [
            "Official cache year1/year2 OHLCV overrides offline yfinance for duplicate symbol/date rows.",
            "HuggingFace IV rows are preserved even when no local OHLCV row exists.",
            "CBOE option volume is aggregated to symbol/date across all exchanges/classes.",
            "FRED macro and yfinance context are forward-filled to daily dates and joined by date.",
            "Priced chain legs are kept separately and also summarized to symbol/date.",
            "Raw source files are untouched.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--force-cboe", action="store_true", help="Rebuild CBOE volume aggregate cache.")
    args = parser.parse_args()

    manifest = build_dataset(args.data_dir, args.out_dir, force_cboe=args.force_cboe)
    print("\n=== Final dataset built ===")
    print(json.dumps({
        "out_dir": manifest["out_dir"],
        "row_counts": manifest["row_counts"],
        "date_ranges": manifest["date_ranges"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
