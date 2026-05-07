"""yfinance-backed earnings calendar for the paper_v1 satellite sleeve.

Fetches per-ticker earnings announcement dates and caches them to disk.
Used by the evidence orchestrator to enumerate satellite events in
Year 1 (training) and Year 2 (held-out) windows.
"""
from __future__ import annotations

import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "earnings"

_CACHE_COL = "earnings_date"


def _coerce_date(value: date | str) -> date:
    """Accept date or ISO string ('YYYY-MM-DD') and return a date."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        return datetime.strptime(value, "%Y-%m-%d").date()
    raise TypeError(f"unsupported date input: {type(value).__name__}")


def cache_path(ticker: str, base_dir: Path = DEFAULT_CACHE_DIR) -> Path:
    """Returns base_dir / '{ticker}.parquet'."""
    return Path(base_dir) / f"{ticker}.parquet"


def save_to_cache(
    ticker: str,
    dates: list[date],
    base_dir: Path = DEFAULT_CACHE_DIR,
) -> None:
    """Persist as one-column parquet at base_dir/{ticker}.parquet.

    The parquet file holds a single column 'earnings_date' carrying
    datetime.date objects (object dtype). Dates are deduplicated and
    sorted ascending before write.
    """
    base_dir = Path(base_dir)
    base_dir.mkdir(parents=True, exist_ok=True)
    normalized = sorted({_coerce_date(d) for d in dates})
    df = pd.DataFrame({_CACHE_COL: normalized})
    df.to_parquet(cache_path(ticker, base_dir), index=False)


def load_cached(
    ticker: str,
    base_dir: Path = DEFAULT_CACHE_DIR,
) -> list[date] | None:
    """Returns cached list[date] or None if no cache."""
    path = cache_path(ticker, base_dir)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if _CACHE_COL not in df.columns:
        return []
    out: list[date] = []
    for v in df[_CACHE_COL].tolist():
        if v is None or (isinstance(v, float) and pd.isna(v)):
            continue
        if isinstance(v, datetime):
            out.append(v.date())
        elif isinstance(v, date):
            out.append(v)
        else:
            # pandas Timestamp or numpy datetime64 — go through pd.Timestamp
            ts = pd.Timestamp(v)
            if pd.isna(ts):
                continue
            out.append(ts.date())
    return out


def _earnings_df_to_dates(df: pd.DataFrame | None) -> list[date]:
    """Convert yfinance.get_earnings_dates() output to a sorted list[date]."""
    if df is None or len(df) == 0:
        return []
    idx = df.index
    # The index is normally a DatetimeIndex (tz-aware); be defensive.
    try:
        ts_index = pd.to_datetime(idx, errors="coerce")
    except Exception:
        return []
    seen: set[date] = set()
    for ts in ts_index:
        if ts is None or pd.isna(ts):
            continue
        # Use the calendar date in the timestamp's own tz (typically
        # America/New_York from yfinance). Converting to UTC would shift
        # late-day announcements onto the next calendar date.
        ts = pd.Timestamp(ts)
        seen.add(ts.date())
    return sorted(seen)


def fetch_ticker_earnings(
    ticker: str,
    *,
    start: date | str | None = None,
    end: date | str | None = None,
) -> list[date]:
    """Fetch earnings dates for one ticker via yfinance.

    Returns dates in [start, end] inclusive (either bound may be None for
    open-ended). Returns [] if Yahoo returns no data, the symbol is delisted,
    or any network/parse failure occurs (the failure is logged at WARNING).
    Raises ValueError on a malformed ticker argument (empty/non-string).
    """
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError(f"invalid ticker: {ticker!r}")

    start_d = _coerce_date(start) if start is not None else None
    end_d = _coerce_date(end) if end is not None else None

    try:
        df = yf.Ticker(ticker).get_earnings_dates(limit=100)
    except Exception as exc:  # robust to flaky yfinance / parse failures
        logger.warning("fetch_ticker_earnings(%s): %s", ticker, exc)
        return []

    dates = _earnings_df_to_dates(df)
    if start_d is not None:
        dates = [d for d in dates if d >= start_d]
    if end_d is not None:
        dates = [d for d in dates if d <= end_d]
    return dates


def fetch_calendar(
    tickers: list[str],
    *,
    start: date | str,
    end: date | str,
    base_dir: Path = DEFAULT_CACHE_DIR,
    use_cache: bool = True,
) -> dict[str, list[date]]:
    """Build an earnings calendar across a ticker list.

    For each ticker: load from cache if present and use_cache=True; otherwise
    fetch via fetch_ticker_earnings, save to cache, return dict.
    Empty tickers list returns {}. Network errors per-ticker are logged and
    the ticker is omitted from the result (does not raise).
    """
    if not tickers:
        return {}

    base_dir = Path(base_dir)
    start_d = _coerce_date(start)
    end_d = _coerce_date(end)
    out: dict[str, list[date]] = {}

    for ticker in tickers:
        try:
            cached = load_cached(ticker, base_dir) if use_cache else None
            if cached is not None:
                out[ticker] = [d for d in cached if start_d <= d <= end_d]
                continue
            dates = fetch_ticker_earnings(ticker, start=start_d, end=end_d)
            # Persist the fetched window so subsequent calls are offline.
            save_to_cache(ticker, dates, base_dir)
            out[ticker] = dates
        except ValueError:
            # propagate caller-error (malformed ticker string)
            raise
        except Exception as exc:
            logger.warning("fetch_calendar(%s): %s", ticker, exc)
            continue

    return out
