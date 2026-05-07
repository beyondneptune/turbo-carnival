"""yfinance daily-price adapter with parquet caching for Phase 1 ingest."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import yfinance as yf

DEFAULT_BASE_DIR = Path(__file__).resolve().parents[1] / "data"
RUSSELL_1000_SAMPLE = [
    "SPY",
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "JPM",
    "V",
]

_OHLCV_RENAME = {
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}
_OHLCV_KEEP = ["open", "high", "low", "close", "volume"]


def _normalize_single(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=_OHLCV_RENAME)
    keep = [c for c in _OHLCV_KEEP if c in df.columns]
    df = df[keep].copy()
    df.index = pd.DatetimeIndex(df.index).tz_localize(None)
    df.index.name = "date"
    if "close" in df.columns:
        df = df.dropna(subset=["close"])
    return df


def fetch_daily_prices(
    tickers: list[str],
    start: date | str,
    end: date | str,
    *,
    auto_adjust: bool = True,
) -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        group_by="ticker",
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        # Multi-ticker shape: top-level column is ticker symbol.
        top_level = raw.columns.get_level_values(0).unique().tolist()
        for ticker in tickers:
            if ticker not in top_level:
                continue
            sub = raw[ticker]
            if not isinstance(sub, pd.DataFrame):
                continue
            sub = _normalize_single(sub)
            if not sub.empty:
                out[ticker] = sub
    else:
        # Single-ticker case: columns are OHLCV directly.
        sub = _normalize_single(raw)
        if not sub.empty and len(tickers) == 1:
            out[tickers[0]] = sub
    return out


def cache_path(ticker: str, base_dir: Path) -> Path:
    return base_dir / "prices" / f"{ticker}.parquet"


def save_to_parquet(prices: dict[str, pd.DataFrame], base_dir: Path) -> None:
    prices_dir = base_dir / "prices"
    prices_dir.mkdir(parents=True, exist_ok=True)
    for ticker, df in prices.items():
        path = cache_path(ticker, base_dir)
        # Reset index so 'date' is a real column in parquet; restored on load.
        df_to_write = df.copy()
        if df_to_write.index.name != "date":
            df_to_write.index.name = "date"
        df_to_write.reset_index().to_parquet(path, index=False)


def load_from_parquet(ticker: str, base_dir: Path) -> pd.DataFrame | None:
    path = cache_path(ticker, base_dir)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")
    df.index = pd.DatetimeIndex(df.index)
    df.index.name = "date"
    return df


def fetch_and_cache(
    tickers: list[str],
    start: date | str,
    end: date | str,
    base_dir: Path,
) -> dict[str, pd.DataFrame]:
    prices = fetch_daily_prices(tickers, start, end)
    save_to_parquet(prices, base_dir)
    return prices
