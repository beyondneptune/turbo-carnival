"""Phase 1.5 — augment metadata-only chain cache with real Polygon prices.

Narrow scope per the user's go-decision: ATM call + ATM put for satellite
events at T-1 / T+5; 4 SPY iron-condor legs at entry / exit. Computes real
ATM IV from straddle prices. Writes to data/cache/year{1,2}/chains_priced/.
Resumable + idempotent: existing priced parquets skip. Status JSON writer
matches scripts/fetch_paper_v1_data.py contract.

Run: caffeinate -imsu .venv/bin/python scripts/augment_chain_pricing.py
ETA at Polygon Free 5 rpm: ~8 hours wall.

Output schema (long format — one row per contract x timing):
    ticker, underlying_ticker, contract_type, strike_price, expiration_date,
    as_of_date, timing ('entry'|'exit'), event_date,
    bid (NaN — Polygon Free aggregates don't expose bid),
    ask (NaN), mid (= close from daily aggregate), underlying_close,
    is_atm (bool), atm_iv (broadcast per event).

The orchestrator (scripts/run_paper_v1_evidence.py) detects priced parquets
by the presence of bid/ask/mid/underlying_close/atm_iv columns. The raw
metadata cache under chains/ remains untouched and auditable.

Exit-date logic:
  * Satellite: entry = T-1 (trading-day-on-or-before event-1); exit = T+5
    trading days from entry. The orchestrator's SATELLITE_HOLD_TRADING_DAYS=5.
  * SPY core: entry = each Monday in the year window (the orchestrator's
    CORE_ENTRY_WEEKDAY=0 pattern in replay_year2). Exit = the iron-condor
    expiry, which is the first Friday >= entry+37 calendar days (matching
    _synth_spy_chain_at).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from features.vol_surface import implied_vol  # noqa: E402
from ingest.polygon_free import (  # noqa: E402
    PolygonAuthError,
    PolygonError,
    PolygonFreeClient,
)
from signals.paper_v1_universe import (  # noqa: E402
    PAPER_V1_SATELLITE,
)

logger = logging.getLogger("augment_chain_pricing")

DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"
STATUS_FILE = REPO_ROOT / "data" / "paper_v1_status.json"
DEFAULT_LOG_FILE = REPO_ROOT / "data" / "logs" / "paper_v1_augment.log"

# Year windows must match scripts/fetch_paper_v1_data.py and
# scripts/run_paper_v1_evidence.py — keep these constants in sync if the
# prereg moves the boundary.
YEAR1_START = date(2024, 5, 7)
YEAR1_END = date(2025, 5, 6)
YEAR2_START = date(2025, 5, 7)
YEAR2_END = date(2026, 5, 6)

# Satellite policy — matches run_paper_v1_evidence.py.
SATELLITE_HOLD_TRADING_DAYS = 5

# SPY core policy — matches run_paper_v1_evidence.py replay_year2 + the synthetic
# SPY chain generator. Mondays = entry day; expiry = first Friday >= entry+37d.
CORE_ENTRY_WEEKDAY = 0  # Monday
CORE_DTE_TARGET_DAYS = 37
CORE_WING_WIDTH_USD = 5.0
CORE_TARGET_SHORT_DELTA = 0.12

# Risk-free rate for IV inversion. Matches eval/baselines RISK_FREE_RATE_PAPER_V1
# closely enough for IV solving (the inversion is mostly insensitive to r at
# short DTEs, but we pin at 4% for reproducibility).
IV_RISK_FREE_RATE = 0.04

# Stage names — keep stable for the dashboard.
STAGE_PREFIX = "augment"


# ---------------------------------------------------------------------------
# Atomic IO helpers (same pattern as scripts/fetch_paper_v1_data.py)
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_bytes(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


def _atomic_write_parquet(df: pd.DataFrame, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    df.to_parquet(tmp, index=True if isinstance(df.index, pd.DatetimeIndex) else False)
    os.replace(tmp, target)


def _atomic_write_json(target: Path, obj: dict) -> None:
    payload = json.dumps(obj, indent=2, sort_keys=True, default=str).encode("utf-8")
    _atomic_write_bytes(target, payload)


def _read_json(target: Path) -> dict:
    if not target.exists():
        return {}
    try:
        return json.loads(target.read_text())
    except (OSError, ValueError):
        logger.warning("status file %s unreadable; starting fresh", target)
        return {}


def configure_logging(log_file: Path | None = DEFAULT_LOG_FILE) -> None:
    """Send augment logs to stderr and, when configured, a persistent log file."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# Status writer — same merge semantics as scripts/fetch_paper_v1_data.py
# ---------------------------------------------------------------------------


def write_status(
    stage: str,
    status: str,
    *,
    status_file: Path = STATUS_FILE,
    phase: str | None = None,
    **fields,
) -> dict:
    """Atomically merge ``stage`` into the status JSON. Returns merged doc."""
    existing = _read_json(status_file)
    stages = existing.get("stages") or {}
    stage_entry = {"status": status, "updated_at_utc": _utc_now_iso()}
    stage_entry.update(fields)
    stages[stage] = stage_entry
    doc = {
        "started_at_utc": existing.get("started_at_utc") or _utc_now_iso(),
        "last_update_utc": _utc_now_iso(),
        "phase": phase if phase is not None else existing.get("phase", "augmenting"),
        "stages": stages,
    }
    _atomic_write_json(status_file, doc)
    return doc


# ---------------------------------------------------------------------------
# Cache path helpers
# ---------------------------------------------------------------------------


def cache_path_priced(
    ticker: str,
    event_date: date,
    *,
    year: str,
    base_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Return the priced-parquet path for ``(ticker, event_date)`` under year."""
    return Path(base_dir) / year / "chains_priced" / ticker / f"{event_date.isoformat()}.parquet"


def cache_path_chain(
    ticker: str,
    event_date: date,
    *,
    year: str,
    base_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    """Return the metadata chain parquet path (input to this script)."""
    return Path(base_dir) / year / "chains" / ticker / f"{event_date.isoformat()}.parquet"


def cache_path_underlying_daily(
    ticker: str,
    *,
    year: str,
    base_dir: Path = DEFAULT_CACHE_DIR,
) -> Path:
    return Path(base_dir) / year / f"{ticker}_daily.parquet"


def _year_window(year: str) -> tuple[date, date]:
    if year == "year1":
        return YEAR1_START, YEAR1_END
    if year == "year2":
        return YEAR2_START, YEAR2_END
    raise ValueError(f"unknown year {year!r}")


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def trading_day_on_or_before(d: date) -> date:
    """Return the most recent weekday on-or-before ``d`` (no holiday calendar)."""
    cur = d
    while cur.weekday() >= 5:
        cur -= timedelta(days=1)
    return cur


def add_trading_days(start: date, n: int) -> date:
    """Add ``n`` weekdays to ``start`` (skip Sat/Sun). No holiday calendar."""
    cur = start
    added = 0
    while added < n:
        cur += timedelta(days=1)
        if cur.weekday() < 5:
            added += 1
    return cur


def next_friday_on_or_after(d: date) -> date:
    """Snap to the first Friday >= ``d``."""
    cur = d
    while cur.weekday() != 4:
        cur += timedelta(days=1)
    return cur


# ---------------------------------------------------------------------------
# Underlying daily-close cache
# ---------------------------------------------------------------------------


def fetch_underlying_daily(
    client: PolygonFreeClient,
    ticker: str,
    *,
    year_start: date,
    year_end: date,
    target: Path,
) -> pd.DataFrame:
    """Pull a per-ticker daily aggregate for the year window. Idempotent.

    Re-uses scripts/fetch_paper_v1_data.py's spy_daily approach: one parquet
    per ticker, one Polygon call total. Returns the loaded frame.
    """
    if target.exists():
        return pd.read_parquet(target)
    df = client.fetch_aggregates(
        ticker=ticker,
        from_date=year_start.isoformat(),
        to_date=year_end.isoformat(),
        timespan="day",
        multiplier=1,
        adjusted=True,
    )
    _atomic_write_parquet(df, target)
    return df


def lookup_close_on(daily: pd.DataFrame, d: date) -> float:
    """Return the close on date ``d`` (or last trading day before) from daily.

    Daily index is a DatetimeIndex (per polygon_free._aggregates_to_df). We
    coerce to date and pick the last close on or before ``d``. Returns NaN
    if no such row exists.
    """
    if daily is None or daily.empty:
        return float("nan")
    if "close" not in daily.columns:
        return float("nan")
    try:
        idx_dates = pd.to_datetime(daily.index).date
    except (TypeError, ValueError):
        return float("nan")
    mask = np.array([dd <= d for dd in idx_dates])
    if not mask.any():
        return float("nan")
    candidate = daily.loc[mask].iloc[-1]
    val = candidate["close"]
    if pd.isna(val):
        return float("nan")
    return float(val)


# ---------------------------------------------------------------------------
# ATM strike selection
# ---------------------------------------------------------------------------


def pick_atm_strike(strikes: Iterable[float], underlying_close: float) -> float:
    """Return the strike closest to ``underlying_close``. Stable on ties."""
    strikes = sorted({float(s) for s in strikes})
    if not strikes:
        raise ValueError("strikes is empty")
    if not np.isfinite(underlying_close):
        # Fall back to the median strike; the orchestrator skips events
        # where atm_iv ends up NaN, which catches downstream issues.
        return float(strikes[len(strikes) // 2])
    return float(min(strikes, key=lambda s: (abs(s - underlying_close), s)))


def pick_post_event_expiry(
    chain_metadata: pd.DataFrame,
    event_date: date,
    *,
    min_days_after: int = 3,
) -> date | None:
    """Pick the nearest expiry >= event_date + min_days_after.

    Mirrors the orchestrator's expiry rule in build_earnings_events.
    """
    if chain_metadata is None or chain_metadata.empty:
        return None
    if "expiration_date" not in chain_metadata.columns:
        return None
    cutoff = event_date + timedelta(days=min_days_after)
    expiries = sorted(
        {
            pd.Timestamp(e).date()
            for e in chain_metadata["expiration_date"].tolist()
            if e is not None
        }
    )
    eligible = [e for e in expiries if e >= cutoff]
    if not eligible:
        return None
    return eligible[0]


def find_contract_ticker(
    chain_metadata: pd.DataFrame,
    *,
    expiration_date: date,
    contract_type: str,
    strike: float,
) -> str | None:
    """Locate the option ticker for (expiry, type, strike) in the metadata frame."""
    if chain_metadata is None or chain_metadata.empty:
        return None
    sub = chain_metadata[
        (pd.to_datetime(chain_metadata["expiration_date"]).dt.date == expiration_date)
        & (chain_metadata["contract_type"] == contract_type)
        & (np.isclose(chain_metadata["strike_price"].astype(float), float(strike)))
    ]
    if sub.empty:
        return None
    return str(sub.iloc[0]["ticker"])


# ---------------------------------------------------------------------------
# Pricing fetchers
# ---------------------------------------------------------------------------


def fetch_option_close(
    client: PolygonFreeClient,
    option_ticker: str,
    on_date: date,
) -> float:
    """Return the close on ``on_date`` for the given option ticker.

    Polygon Free's aggregates endpoint returns OHLC + vwap. There's no bid/ask
    on this endpoint; for EOD evidence we treat ``close`` as the mid. NaN if
    no aggregate is returned.
    """
    try:
        df = client.fetch_options_aggregates(
            option_ticker=option_ticker,
            from_date=on_date.isoformat(),
            to_date=on_date.isoformat(),
        )
    except PolygonError as exc:
        logger.warning("option agg failed %s on %s: %s", option_ticker, on_date, exc)
        return float("nan")
    if df is None or df.empty:
        return float("nan")
    if "close" not in df.columns:
        return float("nan")
    val = df["close"].iloc[0]
    if pd.isna(val):
        return float("nan")
    return float(val)


# ---------------------------------------------------------------------------
# IV inversion
# ---------------------------------------------------------------------------


def compute_atm_iv_from_straddle(
    *,
    call_mid: float,
    put_mid: float,
    spot: float,
    strike: float,
    entry_date: date,
    expiration_date: date,
    risk_free_rate: float = IV_RISK_FREE_RATE,
) -> float:
    """Average of ATM call IV and ATM put IV. NaN if any input is bad."""
    if not (np.isfinite(call_mid) and np.isfinite(put_mid)):
        return float("nan")
    if not (np.isfinite(spot) and spot > 0):
        return float("nan")
    if not np.isfinite(strike) or strike <= 0:
        return float("nan")
    days = (expiration_date - entry_date).days
    if days <= 0:
        return float("nan")
    T = days / 365.0
    try:
        iv_call = implied_vol(call_mid, "call", spot, strike, T, risk_free_rate)
    except (ValueError, RuntimeError):
        iv_call = float("nan")
    try:
        iv_put = implied_vol(put_mid, "put", spot, strike, T, risk_free_rate)
    except (ValueError, RuntimeError):
        iv_put = float("nan")
    if math.isnan(iv_call) and math.isnan(iv_put):
        return float("nan")
    if math.isnan(iv_call):
        return float(iv_put)
    if math.isnan(iv_put):
        return float(iv_call)
    return float(0.5 * (iv_call + iv_put))


# ---------------------------------------------------------------------------
# Row builders
# ---------------------------------------------------------------------------


def _build_priced_row(
    *,
    option_ticker: str,
    underlying_ticker: str,
    contract_type: str,
    strike_price: float,
    expiration_date: date,
    as_of_date: date,
    timing: str,
    event_date: date,
    mid: float,
    underlying_close: float,
    is_atm: bool,
    atm_iv: float,
) -> dict:
    """Build one priced row in the spec'd schema."""
    return {
        "ticker": option_ticker,
        "underlying_ticker": underlying_ticker,
        "contract_type": contract_type,
        "strike_price": float(strike_price),
        "expiration_date": expiration_date.isoformat(),
        "as_of_date": as_of_date.isoformat(),
        "timing": timing,
        "event_date": event_date.isoformat(),
        "bid": float("nan"),
        "ask": float("nan"),
        "mid": float(mid) if np.isfinite(mid) else float("nan"),
        "underlying_close": float(underlying_close) if np.isfinite(underlying_close) else float("nan"),
        "is_atm": bool(is_atm),
        "atm_iv": float(atm_iv) if np.isfinite(atm_iv) else float("nan"),
    }


# ---------------------------------------------------------------------------
# Per-event satellite augmentation
# ---------------------------------------------------------------------------


def augment_satellite_event(
    *,
    client: PolygonFreeClient,
    ticker: str,
    event_date: date,
    chain_metadata: pd.DataFrame,
    underlying_daily: pd.DataFrame,
    target: Path,
    risk_free_rate: float = IV_RISK_FREE_RATE,
) -> bool:
    """Price ATM call + put at T-1 entry and T+5 exit. Atomic + idempotent.

    Returns True if a parquet was written, False if skipped/failed.
    """
    if target.exists():
        logger.debug("priced cache hit %s %s", ticker, event_date)
        return False

    expiry = pick_post_event_expiry(chain_metadata, event_date, min_days_after=3)
    if expiry is None:
        logger.info("no eligible expiry for %s %s; skipping", ticker, event_date)
        return False

    entry_date = trading_day_on_or_before(event_date - timedelta(days=1))
    exit_date = add_trading_days(entry_date, SATELLITE_HOLD_TRADING_DAYS)

    underlying_close_entry = lookup_close_on(underlying_daily, entry_date)
    underlying_close_exit = lookup_close_on(underlying_daily, exit_date)

    # Strikes available at the chosen expiry.
    same_expiry = chain_metadata[
        pd.to_datetime(chain_metadata["expiration_date"]).dt.date == expiry
    ]
    if same_expiry.empty:
        return False
    strikes = same_expiry["strike_price"].astype(float).tolist()
    if not strikes:
        return False
    atm_strike = pick_atm_strike(strikes, underlying_close_entry)

    call_ticker = find_contract_ticker(
        same_expiry, expiration_date=expiry, contract_type="call", strike=atm_strike
    )
    put_ticker = find_contract_ticker(
        same_expiry, expiration_date=expiry, contract_type="put", strike=atm_strike
    )
    if call_ticker is None or put_ticker is None:
        logger.warning("missing ATM contract tickers for %s %s K=%s exp=%s",
                       ticker, event_date, atm_strike, expiry)
        return False

    # Fetch closes (= mid). Two API calls per timing.
    call_entry = fetch_option_close(client, call_ticker, entry_date)
    put_entry = fetch_option_close(client, put_ticker, entry_date)
    call_exit = fetch_option_close(client, call_ticker, exit_date)
    put_exit = fetch_option_close(client, put_ticker, exit_date)

    atm_iv = compute_atm_iv_from_straddle(
        call_mid=call_entry,
        put_mid=put_entry,
        spot=underlying_close_entry,
        strike=atm_strike,
        entry_date=entry_date,
        expiration_date=expiry,
        risk_free_rate=risk_free_rate,
    )

    rows = [
        _build_priced_row(
            option_ticker=call_ticker,
            underlying_ticker=ticker,
            contract_type="call",
            strike_price=atm_strike,
            expiration_date=expiry,
            as_of_date=entry_date,
            timing="entry",
            event_date=event_date,
            mid=call_entry,
            underlying_close=underlying_close_entry,
            is_atm=True,
            atm_iv=atm_iv,
        ),
        _build_priced_row(
            option_ticker=put_ticker,
            underlying_ticker=ticker,
            contract_type="put",
            strike_price=atm_strike,
            expiration_date=expiry,
            as_of_date=entry_date,
            timing="entry",
            event_date=event_date,
            mid=put_entry,
            underlying_close=underlying_close_entry,
            is_atm=True,
            atm_iv=atm_iv,
        ),
        _build_priced_row(
            option_ticker=call_ticker,
            underlying_ticker=ticker,
            contract_type="call",
            strike_price=atm_strike,
            expiration_date=expiry,
            as_of_date=exit_date,
            timing="exit",
            event_date=event_date,
            mid=call_exit,
            underlying_close=underlying_close_exit,
            is_atm=True,
            atm_iv=atm_iv,
        ),
        _build_priced_row(
            option_ticker=put_ticker,
            underlying_ticker=ticker,
            contract_type="put",
            strike_price=atm_strike,
            expiration_date=expiry,
            as_of_date=exit_date,
            timing="exit",
            event_date=event_date,
            mid=put_exit,
            underlying_close=underlying_close_exit,
            is_atm=True,
            atm_iv=atm_iv,
        ),
    ]
    df = pd.DataFrame(rows)
    _atomic_write_parquet(df, target)
    return True


# ---------------------------------------------------------------------------
# SPY core augmentation
# ---------------------------------------------------------------------------


def _spy_core_entry_dates(year_start: date, year_end: date) -> list[date]:
    """All Mondays in [year_start, year_end]. Mirrors replay_year2's pattern.

    The orchestrator uses spy_daily.index when available and falls back to
    business days. We use bdate_range -> filter to weekday==Monday so the
    list is reproducible without depending on which days had bars.
    """
    return [
        d.date()
        for d in pd.bdate_range(year_start, year_end)
        if d.weekday() == CORE_ENTRY_WEEKDAY
    ]


def _pick_short_strike_by_proximity(
    strikes: list[float],
    *,
    target_offset: float,
    spot: float,
) -> float:
    """Pick the strike closest to ``spot + target_offset`` (signed)."""
    target = spot + target_offset
    return float(min(sorted(strikes), key=lambda s: (abs(s - target), s)))


def augment_spy_core_entry(
    *,
    client: PolygonFreeClient,
    entry_date: date,
    chain_metadata: pd.DataFrame,
    spy_daily: pd.DataFrame,
    target: Path,
    risk_free_rate: float = IV_RISK_FREE_RATE,
) -> bool:
    """Price 4 iron-condor legs at entry + exit. Atomic + idempotent.

    Strike selection here uses spot-offset proxies for the 12-delta short
    strikes (since this metadata cache lacks deltas): short call ~ +5%,
    short put ~ -5%, with $5 wings. The orchestrator's signal layer
    (signals/vrp_core_spy.py) does the real delta-based pick at backtest
    time; this script's job is to make sure the four legs near those
    moneyness ranges have priced rows on disk so the orchestrator's
    real-pricing path doesn't fall back to synthetic.

    Returns True if a parquet was written, False if skipped/failed.
    """
    if target.exists():
        return False

    expiry = pick_post_event_expiry(chain_metadata, entry_date, min_days_after=CORE_DTE_TARGET_DAYS - 7)
    if expiry is None:
        # Fall back to next Friday >= entry+CORE_DTE_TARGET_DAYS.
        expiry = next_friday_on_or_after(entry_date + timedelta(days=CORE_DTE_TARGET_DAYS))

    underlying_close_entry = lookup_close_on(spy_daily, entry_date)
    # Exit date for an iron condor = expiry day. The orchestrator currently
    # closes at expiry (see replay_year2 spec.expiry usage).
    exit_date = expiry
    underlying_close_exit = lookup_close_on(spy_daily, exit_date)

    same_expiry = chain_metadata[
        pd.to_datetime(chain_metadata["expiration_date"]).dt.date == expiry
    ]
    if same_expiry.empty:
        return False
    available_calls = sorted(
        {float(s) for s in same_expiry[same_expiry["contract_type"] == "call"]["strike_price"].tolist()}
    )
    available_puts = sorted(
        {float(s) for s in same_expiry[same_expiry["contract_type"] == "put"]["strike_price"].tolist()}
    )
    if not available_calls or not available_puts or not np.isfinite(underlying_close_entry):
        return False

    # ~12 delta proxies — 5% OTM (short_call) / -5% OTM (short_put). Wings $5 farther OTM.
    short_call_target_offset = 0.05 * underlying_close_entry
    short_put_target_offset = -0.05 * underlying_close_entry
    short_call_strike = _pick_short_strike_by_proximity(
        available_calls, target_offset=short_call_target_offset, spot=underlying_close_entry
    )
    short_put_strike = _pick_short_strike_by_proximity(
        available_puts, target_offset=short_put_target_offset, spot=underlying_close_entry
    )
    long_call_target = short_call_strike + CORE_WING_WIDTH_USD
    long_put_target = short_put_strike - CORE_WING_WIDTH_USD
    long_call_strike = float(min(available_calls, key=lambda s: (abs(s - long_call_target), s)))
    long_put_strike = float(min(available_puts, key=lambda s: (abs(s - long_put_target), s)))

    # ATM straddle for IV inversion (broadcast on rows).
    atm_strike = pick_atm_strike(available_calls + available_puts, underlying_close_entry)
    atm_call_ticker = find_contract_ticker(
        same_expiry, expiration_date=expiry, contract_type="call", strike=atm_strike
    )
    atm_put_ticker = find_contract_ticker(
        same_expiry, expiration_date=expiry, contract_type="put", strike=atm_strike
    )
    atm_iv = float("nan")
    if atm_call_ticker is not None and atm_put_ticker is not None:
        atm_call_mid = fetch_option_close(client, atm_call_ticker, entry_date)
        atm_put_mid = fetch_option_close(client, atm_put_ticker, entry_date)
        atm_iv = compute_atm_iv_from_straddle(
            call_mid=atm_call_mid,
            put_mid=atm_put_mid,
            spot=underlying_close_entry,
            strike=atm_strike,
            entry_date=entry_date,
            expiration_date=expiry,
            risk_free_rate=risk_free_rate,
        )

    legs = (
        ("put", "long", long_put_strike),
        ("put", "short", short_put_strike),
        ("call", "short", short_call_strike),
        ("call", "long", long_call_strike),
    )

    rows: list[dict] = []
    for kind, _direction, strike in legs:
        opt_ticker = find_contract_ticker(
            same_expiry, expiration_date=expiry, contract_type=kind, strike=strike
        )
        if opt_ticker is None:
            # Fill rows with NaN mids so the parquet is structurally complete;
            # orchestrator fail-closed mode will skip events with NaN mids.
            entry_mid = float("nan")
            exit_mid = float("nan")
            opt_ticker_str = f"missing:{kind}:{strike}"
        else:
            entry_mid = fetch_option_close(client, opt_ticker, entry_date)
            exit_mid = fetch_option_close(client, opt_ticker, exit_date)
            opt_ticker_str = opt_ticker

        is_atm = bool(np.isclose(strike, atm_strike))
        rows.append(
            _build_priced_row(
                option_ticker=opt_ticker_str,
                underlying_ticker="SPY",
                contract_type=kind,
                strike_price=strike,
                expiration_date=expiry,
                as_of_date=entry_date,
                timing="entry",
                event_date=entry_date,
                mid=entry_mid,
                underlying_close=underlying_close_entry,
                is_atm=is_atm,
                atm_iv=atm_iv,
            )
        )
        rows.append(
            _build_priced_row(
                option_ticker=opt_ticker_str,
                underlying_ticker="SPY",
                contract_type=kind,
                strike_price=strike,
                expiration_date=expiry,
                as_of_date=exit_date,
                timing="exit",
                event_date=entry_date,
                mid=exit_mid,
                underlying_close=underlying_close_exit,
                is_atm=is_atm,
                atm_iv=atm_iv,
            )
        )

    df = pd.DataFrame(rows)
    _atomic_write_parquet(df, target)
    return True


# ---------------------------------------------------------------------------
# Year orchestration
# ---------------------------------------------------------------------------


def _stage_key(year: str) -> str:
    return f"{STAGE_PREFIX}_{year}_pricing"


def _build_satellite_work(
    *,
    year: str,
    base_dir: Path,
    universe: Iterable[str],
    max_events: int | None,
) -> list[tuple[str, date, Path]]:
    """Build (ticker, event_date, target_path) tuples in deterministic order."""
    items: list[tuple[str, date, Path]] = []
    yd = Path(base_dir) / year
    earnings_dir = yd / "earnings_dates"
    if not earnings_dir.exists():
        return items
    for ticker in universe:
        ed_path = earnings_dir / f"{ticker}.parquet"
        if not ed_path.exists():
            continue
        try:
            ed_df = pd.read_parquet(ed_path)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("unreadable earnings file %s: %s", ed_path, exc)
            continue
        if "earnings_date" not in ed_df.columns:
            continue
        seen: set[date] = set()
        for raw in ed_df["earnings_date"].tolist():
            if raw is None:
                continue
            try:
                d = pd.Timestamp(raw).date()
            except (TypeError, ValueError):
                continue
            seen.add(d)
        for ev in sorted(seen):
            target = cache_path_priced(ticker, ev, year=year, base_dir=base_dir)
            items.append((ticker, ev, target))
    if max_events is not None:
        items = items[: int(max_events)]
    return items


def _augment_year(
    *,
    year: str,
    client: PolygonFreeClient,
    base_dir: Path,
    status_file: Path,
    max_events: int | None,
    skip_spy_core: bool,
    universe: Iterable[str] = PAPER_V1_SATELLITE,
) -> dict:
    """Run the full augmentation pipeline for one year."""
    year_start, year_end = _year_window(year)
    stage = _stage_key(year)
    universe = tuple(universe)

    work = _build_satellite_work(
        year=year, base_dir=base_dir, universe=universe, max_events=max_events,
    )
    pending = [(t, d, p) for (t, d, p) in work if not p.exists()]

    write_status(
        stage,
        "running",
        status_file=status_file,
        phase="augmenting",
        current_substage="satellite_daily_preload",
        done=0,
        total=len(work),
        pending=len(pending),
        files_written=0,
        daily_done=0,
        daily_total=len({t for (t, _d, _p) in pending}),
    )

    # Per-ticker daily caches — one Polygon call per ticker, idempotent.
    tickers_in_play = sorted({t for (t, _d, _p) in pending})
    daily_by_ticker: dict[str, pd.DataFrame] = {}
    for daily_done, ticker in enumerate(tickers_in_play, start=1):
        target = cache_path_underlying_daily(ticker, year=year, base_dir=base_dir)
        write_status(
            stage,
            "running",
            status_file=status_file,
            phase="augmenting",
            current_substage="satellite_daily_preload",
            current_ticker=ticker,
            done=0,
            total=len(work),
            pending=len(pending),
            files_written=0,
            daily_done=daily_done - 1,
            daily_total=len(tickers_in_play),
        )
        logger.info(
            "%s daily preload %s/%s ticker=%s",
            year,
            daily_done,
            len(tickers_in_play),
            ticker,
        )
        try:
            daily_by_ticker[ticker] = fetch_underlying_daily(
                client, ticker,
                year_start=year_start,
                year_end=year_end,
                target=target,
            )
        except PolygonError as exc:
            logger.warning("daily fetch failed for %s: %s", ticker, exc)
            daily_by_ticker[ticker] = pd.DataFrame()
        write_status(
            stage,
            "running",
            status_file=status_file,
            phase="augmenting",
            current_substage="satellite_daily_preload",
            current_ticker=ticker,
            done=0,
            total=len(work),
            pending=len(pending),
            files_written=0,
            daily_done=daily_done,
            daily_total=len(tickers_in_play),
        )

    files_written = 0
    done = 0
    for ticker, event_date, target_path in work:
        if target_path.exists():
            done += 1
            if done % 5 == 0 or done == len(work):
                write_status(
                    stage,
                    "running",
                    status_file=status_file,
                    phase="augmenting",
                    current_substage="satellite_event_pricing",
                    current_ticker=ticker,
                    current_event_date=event_date.isoformat(),
                    done=done,
                    total=len(work),
                    pending=max(0, len(work) - done),
                    files_written=files_written,
                )
            continue
        write_status(
            stage,
            "running",
            status_file=status_file,
            phase="augmenting",
            current_substage="satellite_event_pricing",
            current_ticker=ticker,
            current_event_date=event_date.isoformat(),
            done=done,
            total=len(work),
            pending=max(0, len(work) - done),
            files_written=files_written,
        )
        logger.info(
            "%s satellite pricing %s/%s ticker=%s event_date=%s",
            year,
            done + 1,
            len(work),
            ticker,
            event_date,
        )
        chain_path = cache_path_chain(ticker, event_date, year=year, base_dir=base_dir)
        if not chain_path.exists():
            logger.info("no chain metadata for %s %s; skipping", ticker, event_date)
            done += 1
            if done % 5 == 0 or done == len(work):
                write_status(
                    stage,
                    "running",
                    status_file=status_file,
                    phase="augmenting",
                    current_substage="satellite_event_pricing",
                    current_ticker=ticker,
                    current_event_date=event_date.isoformat(),
                    done=done,
                    total=len(work),
                    pending=max(0, len(work) - done),
                    files_written=files_written,
                )
            continue
        try:
            chain_metadata = pd.read_parquet(chain_path)
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning("unreadable chain %s: %s", chain_path, exc)
            done += 1
            if done % 5 == 0 or done == len(work):
                write_status(
                    stage,
                    "running",
                    status_file=status_file,
                    phase="augmenting",
                    current_substage="satellite_event_pricing",
                    current_ticker=ticker,
                    current_event_date=event_date.isoformat(),
                    done=done,
                    total=len(work),
                    pending=max(0, len(work) - done),
                    files_written=files_written,
                )
            continue
        underlying_daily = daily_by_ticker.get(ticker, pd.DataFrame())
        try:
            wrote = augment_satellite_event(
                client=client,
                ticker=ticker,
                event_date=event_date,
                chain_metadata=chain_metadata,
                underlying_daily=underlying_daily,
                target=target_path,
            )
        except PolygonError as exc:
            logger.warning("augment failed %s %s: %s", ticker, event_date, exc)
            wrote = False
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception("augment unexpected error %s %s: %s", ticker, event_date, exc)
            wrote = False
        if wrote:
            files_written += 1
        done += 1
        if done % 5 == 0 or done == len(work):
            write_status(
                stage,
                "running",
                status_file=status_file,
                phase="augmenting",
                current_substage="satellite_event_pricing",
                current_ticker=ticker,
                current_event_date=event_date.isoformat(),
                done=done,
                total=len(work),
                pending=max(0, len(work) - done),
                files_written=files_written,
            )

    spy_core_summary: dict = {"skipped": True, "files_written": 0, "total": 0}
    if not skip_spy_core:
        spy_core_summary = _augment_spy_core_for_year(
            year=year,
            client=client,
            base_dir=base_dir,
            status_file=status_file,
            year_start=year_start,
            year_end=year_end,
        )

    write_status(
        stage,
        "complete",
        status_file=status_file,
        phase="augmenting",
        current_substage="year_complete",
        done=done,
        total=len(work),
        pending=0,
        files_written=files_written,
        spy_core=spy_core_summary,
    )

    return {
        "year": year,
        "satellite_total": len(work),
        "satellite_files_written": files_written,
        "spy_core": spy_core_summary,
    }


def _augment_spy_core_for_year(
    *,
    year: str,
    client: PolygonFreeClient,
    base_dir: Path,
    status_file: Path,
    year_start: date,
    year_end: date,
) -> dict:
    """Iterate every Monday entry; price the 4 IC legs entry+exit."""
    stage = _stage_key(year)
    yd = Path(base_dir) / year
    spy_daily_path = yd / "spy_daily.parquet"
    spy_daily = pd.read_parquet(spy_daily_path) if spy_daily_path.exists() else pd.DataFrame()

    entry_dates = _spy_core_entry_dates(year_start, year_end)

    # SPY chains under the existing metadata layout. The fetcher writes
    # chains/SPY/{ev_date}.parquet only when a SPY earnings event was queued
    # — but the orchestrator's core sleeve doesn't need earnings events.
    # We re-use the SPY chain at the entry date if one exists; else fall
    # back to the most recent SPY chain on or before the entry date.
    spy_chains_dir = yd / "chains" / "SPY"
    available: list[tuple[date, Path]] = []
    if spy_chains_dir.exists():
        for p in sorted(spy_chains_dir.glob("*.parquet")):
            try:
                cd = date.fromisoformat(p.stem)
            except ValueError:
                continue
            available.append((cd, p))

    def _spy_chain_for(d: date) -> pd.DataFrame | None:
        # exact match wins
        for (cd, p) in available:
            if cd == d:
                try:
                    return pd.read_parquet(p)
                except Exception:
                    return None
        # else last on or before
        prior = [(cd, p) for (cd, p) in available if cd <= d]
        if not prior:
            return None
        prior.sort(key=lambda t: t[0])
        try:
            return pd.read_parquet(prior[-1][1])
        except Exception:
            return None

    files_written = 0
    total = len(entry_dates)
    write_status(
        stage,
        "running",
        status_file=status_file,
        phase="augmenting",
        current_substage="spy_core_pricing",
        current_ticker="SPY",
        done=0,
        total=total,
        files_written=0,
    )
    for done, ed in enumerate(entry_dates, start=1):
        target = cache_path_priced("SPY", ed, year=year, base_dir=base_dir)
        logger.info("%s SPY core pricing %s/%s entry_date=%s", year, done, total, ed)
        if target.exists():
            if done % 5 == 0 or done == total:
                write_status(
                    stage,
                    "running",
                    status_file=status_file,
                    phase="augmenting",
                    current_substage="spy_core_pricing",
                    current_ticker="SPY",
                    current_entry_date=ed.isoformat(),
                    done=done,
                    total=total,
                    pending=max(0, total - done),
                    files_written=files_written,
                )
            continue
        chain = _spy_chain_for(ed)
        if chain is None or chain.empty:
            if done % 5 == 0 or done == total:
                write_status(
                    stage,
                    "running",
                    status_file=status_file,
                    phase="augmenting",
                    current_substage="spy_core_pricing",
                    current_ticker="SPY",
                    current_entry_date=ed.isoformat(),
                    done=done,
                    total=total,
                    pending=max(0, total - done),
                    files_written=files_written,
                )
            continue
        try:
            wrote = augment_spy_core_entry(
                client=client,
                entry_date=ed,
                chain_metadata=chain,
                spy_daily=spy_daily,
                target=target,
            )
        except PolygonError as exc:
            logger.warning("SPY core augment failed %s: %s", ed, exc)
            wrote = False
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception("SPY core unexpected error %s: %s", ed, exc)
            wrote = False
        if wrote:
            files_written += 1
        if done % 5 == 0 or done == total:
            write_status(
                stage,
                "running",
                status_file=status_file,
                phase="augmenting",
                current_substage="spy_core_pricing",
                current_ticker="SPY",
                current_entry_date=ed.isoformat(),
                done=done,
                total=total,
                pending=max(0, total - done),
                files_written=files_written,
            )

    return {
        "skipped": False,
        "files_written": int(files_written),
        "total": int(total),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Augment metadata-only chain cache with Polygon Free pricing. "
            "Narrow scope: ATM call+put for satellite events at T-1/T+5; "
            "4 SPY iron-condor legs at entry/exit. Idempotent + resumable."
        ),
    )
    p.add_argument(
        "--year",
        choices=["year1", "year2", "both"],
        default="both",
        help="Which window to augment (default: both).",
    )
    p.add_argument(
        "--max-events",
        type=int,
        default=None,
        help="Cap satellite events for debugging (default: no cap).",
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache root (default: {DEFAULT_CACHE_DIR}).",
    )
    p.add_argument(
        "--status-file",
        type=Path,
        default=STATUS_FILE,
        help=f"Status JSON path (default: {STATUS_FILE}).",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"Persistent log path (default: {DEFAULT_LOG_FILE}).",
    )
    p.add_argument(
        "--skip-spy-core",
        action="store_true",
        help="Debug: only run satellite augmentation; skip SPY core legs.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_file)
    started = time.monotonic()

    write_status(
        "init",
        "running",
        status_file=args.status_file,
        phase="augmenting",
        cache_dir=str(args.cache_dir),
        year_choice=args.year,
        max_events=args.max_events,
        skip_spy_core=bool(args.skip_spy_core),
        log_file=str(args.log_file),
    )
    logger.info(
        "starting chain pricing augment year=%s cache_dir=%s status_file=%s log_file=%s",
        args.year,
        args.cache_dir,
        args.status_file,
        args.log_file,
    )

    try:
        client = PolygonFreeClient()
    except PolygonAuthError as exc:
        logger.error("polygon auth failure: %s", exc)
        write_status(
            "init",
            "error",
            status_file=args.status_file,
            phase="error",
            error=str(exc),
        )
        return 2

    years_to_run = ["year1", "year2"] if args.year == "both" else [args.year]
    summaries: list[dict] = []
    try:
        for year in years_to_run:
            logger.info("=== augmenting %s ===", year)
            summary = _augment_year(
                year=year,
                client=client,
                base_dir=args.cache_dir,
                status_file=args.status_file,
                max_events=args.max_events,
                skip_spy_core=args.skip_spy_core,
            )
            summaries.append(summary)
    except KeyboardInterrupt:
        logger.warning("interrupted by user; partial cache is safe to resume")
        write_status(
            "interrupt",
            "interrupted",
            status_file=args.status_file,
            phase="interrupted",
        )
        return 130
    except Exception as exc:  # pragma: no cover — defensive
        logger.exception("augment failed")
        write_status(
            "exception",
            "error",
            status_file=args.status_file,
            phase="error",
            error=repr(exc),
        )
        return 1

    elapsed_s = time.monotonic() - started
    write_status(
        "complete",
        "complete",
        status_file=args.status_file,
        phase="augment_complete",
        elapsed_seconds=int(elapsed_s),
    )

    print("=" * 70)
    print("paper_v1 chain augmentation: COMPLETE")
    for s in summaries:
        print(f"  {s['year']}")
        for k, v in s.items():
            if k == "year":
                continue
            print(f"    {k}: {v}")
    print(f"  wall time:   {elapsed_s:.1f}s")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
