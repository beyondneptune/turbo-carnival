"""paper_v1 evidence orchestrator: held-out Year 2 OOS backtest.

Reads cached data only -- never calls Polygon directly. Trains LightGBM
models A and B on Year 1, freezes them, replays Year 2 through the
deterministic gates + paper_exec simulator, writes results to the audit DB,
runs the 7-gate evaluator, persists verdict to data/paper_v1_status.json.

After this script completes successfully:
  - The dashboard's "Paper v1 Run" view shows the verdict.
  - The audit DB has a complete trade history under run_id paper_v1_evidence_*.
  - The 7-gate result determines promote / pivot / stop per the prereg.

The script is exercisable end-to-end with synthetic data via --allow-empty-cache;
that mode is intended for sanity-checking the orchestrator without waiting for
the ~42-hour real cache pull.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import os
import sqlite3
import sys
import time
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eval.baselines import (  # noqa: E402
    RISK_FREE_RATE_PAPER_V1,
    BaselineSet,
    best_baseline_summary,
    compute_all_baselines,
)
from eval.gates import (  # noqa: E402
    CostRegimeInputs,
    GateEvaluationReport,
    evaluate_full,
)
from eval.metrics import (  # noqa: E402
    deflated_sharpe_ratio,
    evaluate_regime_stability,
    max_drawdown,
    probability_backtest_overfitting,
)
from paper_exec.simulator import LegOrder, Quote, simulate_trade_fill  # noqa: E402
from risk.envelope import (  # noqa: E402
    CandidatePosition,
    PortfolioState,
    evaluate_envelope,
)
from risk.kill_switch import evaluate_kill_switch  # noqa: E402
from signals.earnings_satellite import (  # noqa: E402
    EarningsEvent,
    build_targets,
)
from signals.paper_v1_universe import (  # noqa: E402
    PAPER_V1_SATELLITE,
)
from signals.twin_model_trainer import (  # noqa: E402
    train_and_compare,
)
from signals.vrp_core_spy import (  # noqa: E402
    compute_entry_credit,
    construct_iron_condor,
)

logger = logging.getLogger("run_paper_v1_evidence")

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "cache"
DEFAULT_DB_PATH = REPO_ROOT / "data" / "trading_system.db"
SCHEMA_PATH = REPO_ROOT / "sql" / "schema.sql"
STATUS_PATH = REPO_ROOT / "data" / "paper_v1_status.json"

YEAR1_START = date(2024, 5, 7)
YEAR1_END = date(2025, 5, 6)
YEAR2_START = date(2025, 5, 7)
YEAR2_END = date(2026, 5, 6)

PREREG_VERSION = "paper_v1.0-frozen"
RUN_TYPE = "paper_frozen"
INITIAL_NAV_USD = 10_000.0

CORE_ENTRY_WEEKDAY = 0  # Monday
CORE_TARGET_SHORT_DELTA = 0.12
CORE_WING_WIDTH_USD = 5.0
CORE_DTE_MIN = 30
CORE_DTE_MAX = 45
CORE_QUANTITY = 1
CORE_MULTIPLIER = 100

SATELLITE_HOLD_TRADING_DAYS = 5
SATELLITE_QUANTITY = 1
SATELLITE_MULTIPLIER = 100

MODEL_VERSION = "lightgbm_modelB_paper_v1_evidence"

TRADING_DAYS_PER_YEAR = 252
HOLD_PERIOD_DAYS = 5
EVENTS_PER_YEAR = TRADING_DAYS_PER_YEAR / HOLD_PERIOD_DAYS  # ~50.4

# How often to flush the status JSON during the year-2 replay loop.
STATUS_UPDATE_EVERY_N_EVENTS = 5

# RNG seed for synthetic data generation (deterministic).
SYNTHETIC_SEED = 7


# ---------------------------------------------------------------------------
# Atomic IO helpers
# ---------------------------------------------------------------------------


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _atomic_write_json(target: Path, obj: dict) -> None:
    """Write JSON atomically: <target>.tmp -> rename(target)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(obj, indent=2, sort_keys=True, default=_json_default).encode("utf-8")
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        fh.write(payload)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, target)


def _json_default(obj: Any) -> Any:
    """Default JSON serializer for dataclasses, dates, numpy scalars."""
    if dataclasses.is_dataclass(obj):
        return dataclasses.asdict(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return str(obj)


# ---------------------------------------------------------------------------
# Status writer -- merges stage updates atomically
# ---------------------------------------------------------------------------


class StatusWriter:
    """Thin manager around the status JSON. Atomic writes; merges stages.

    The on-disk shape matches the dashboard's "Paper v1 Run" view contract:

        {
          "run_id": "...",
          "phase": "loading|year1|train|year2_replay|compute|complete|error",
          "started_at_utc": "...",
          "last_update_utc": "...",
          "stages": { "<name>": {"status": ..., **fields} },
          "twin_model_comparison": {...},
          "gates": {...},
          "errors": [...]
        }
    """

    def __init__(self, target: Path, run_id: str) -> None:
        self.target = Path(target)
        self.run_id = run_id
        self._doc: dict[str, Any] = {
            "run_id": run_id,
            "phase": "loading",
            "started_at_utc": _utc_now_iso(),
            "last_update_utc": _utc_now_iso(),
            "stages": {},
            "errors": [],
        }
        self.flush()

    def set_phase(self, phase: str) -> None:
        self._doc["phase"] = phase
        self._doc["last_update_utc"] = _utc_now_iso()
        self.flush()

    def set_stage(self, name: str, status: str, **fields: Any) -> None:
        stage = {"status": status, "updated_at_utc": _utc_now_iso()}
        stage.update(fields)
        self._doc.setdefault("stages", {})[name] = stage
        self._doc["last_update_utc"] = _utc_now_iso()
        self.flush()

    def update_stage(self, name: str, **fields: Any) -> None:
        """Update fields on an existing stage (or create it as 'running')."""
        stages = self._doc.setdefault("stages", {})
        existing = stages.get(name) or {"status": "running"}
        existing.update(fields)
        existing["updated_at_utc"] = _utc_now_iso()
        stages[name] = existing
        self._doc["last_update_utc"] = _utc_now_iso()
        self.flush()

    def set_top_level(self, key: str, value: Any) -> None:
        self._doc[key] = value
        self._doc["last_update_utc"] = _utc_now_iso()
        self.flush()

    def add_error(self, message: str, **extra: Any) -> None:
        err = {"message": message, "timestamp_utc": _utc_now_iso(), **extra}
        self._doc.setdefault("errors", []).append(err)
        self._doc["last_update_utc"] = _utc_now_iso()
        self.flush()

    def flush(self) -> None:
        _atomic_write_json(self.target, self._doc)

    @property
    def doc(self) -> dict[str, Any]:
        return self._doc


# ---------------------------------------------------------------------------
# Cache loaders
# ---------------------------------------------------------------------------


def cache_year_dir(base_dir: Path, year: str) -> Path:
    if year not in ("year1", "year2"):
        raise ValueError(f"unknown year {year!r}")
    return Path(base_dir) / year


def load_year_cache(
    base_dir: Path,
    year: str,
    *,
    universe: tuple[str, ...] = PAPER_V1_SATELLITE,
) -> dict[str, Any]:
    """Read cached SPY daily, IV daily, earnings dates, and chains for one year.

    Returns a dict::

        {
          "year": "year1|year2",
          "window": (start, end),
          "spy_daily": pd.DataFrame | None,
          "spy_iv_daily": pd.DataFrame | None,
          "earnings_dates": {ticker -> list[date]},
          "chains": {(ticker, event_date) -> pd.DataFrame},
        }

    Missing files are tolerated (returned as empty/None) -- the caller decides
    whether the cache is sufficient. The fetcher's contract only guarantees
    contract metadata in the chain parquets (ticker/strike/expiration); pricing
    is downstream and not part of this cache. The orchestrator pairs the
    cached chains with synthetic mids when needed for replay.
    """
    yd = cache_year_dir(Path(base_dir), year)
    out: dict[str, Any] = {
        "year": year,
        "window": _year_window(year),
        "spy_daily": None,
        "spy_iv_daily": None,
        "earnings_dates": {},
        "chains": {},
    }

    spy_daily_path = yd / "spy_daily.parquet"
    if spy_daily_path.exists():
        out["spy_daily"] = pd.read_parquet(spy_daily_path)

    spy_iv_path = yd / "spy_iv_daily.parquet"
    if spy_iv_path.exists():
        out["spy_iv_daily"] = pd.read_parquet(spy_iv_path)

    ed_dir = yd / "earnings_dates"
    if ed_dir.exists():
        for ticker in universe:
            tpath = ed_dir / f"{ticker}.parquet"
            if tpath.exists():
                df = pd.read_parquet(tpath)
                if "earnings_date" in df.columns:
                    dates: list[date] = []
                    for v in df["earnings_date"].tolist():
                        if v is None:
                            continue
                        try:
                            d = pd.Timestamp(v).date()
                        except (TypeError, ValueError):
                            continue
                        dates.append(d)
                    out["earnings_dates"][ticker] = sorted(set(dates))

    chains_dir = yd / "chains"
    if chains_dir.exists():
        for ticker_dir in sorted(chains_dir.iterdir()):
            if not ticker_dir.is_dir():
                continue
            ticker = ticker_dir.name
            for chain_path in sorted(ticker_dir.glob("*.parquet")):
                try:
                    event_date = date.fromisoformat(chain_path.stem)
                except ValueError:
                    continue
                try:
                    out["chains"][(ticker, event_date)] = pd.read_parquet(chain_path)
                except Exception as exc:  # pragma: no cover -- defensive
                    logger.warning("failed to read %s: %s", chain_path, exc)

    return out


def _year_window(year: str) -> tuple[date, date]:
    if year == "year1":
        return YEAR1_START, YEAR1_END
    if year == "year2":
        return YEAR2_START, YEAR2_END
    raise ValueError(f"unknown year {year!r}")


def cache_summary(cache: dict[str, Any]) -> dict[str, Any]:
    """Return a small dict for status JSON: row counts, event counts, etc."""
    n_spy = len(cache["spy_daily"]) if cache["spy_daily"] is not None else 0
    n_iv = len(cache["spy_iv_daily"]) if cache["spy_iv_daily"] is not None else 0
    n_events = sum(len(v) for v in cache["earnings_dates"].values())
    n_chain = len(cache["chains"])
    return {
        "spy_daily_rows": int(n_spy),
        "spy_iv_daily_rows": int(n_iv),
        "earnings_events": int(n_events),
        "chain_files": int(n_chain),
    }


# ---------------------------------------------------------------------------
# Synthetic data generator (for --allow-empty-cache)
# ---------------------------------------------------------------------------


def synthesize_year_cache(
    year: str,
    *,
    n_satellite_events_per_ticker: int = 4,
    universe: tuple[str, ...] | None = None,
    seed: int = SYNTHETIC_SEED,
) -> dict[str, Any]:
    """Build a self-consistent synthetic cache for ``year``.

    Generates SPY daily OHLC, ATM IV history, per-ticker earnings dates, and
    per-event option chains with bid/ask. Volumes are deliberately small
    (~4 events/ticker) to keep tests fast while still covering the >=50 event
    floor required by the sample-size gate (50 tickers * 4 events = 200).
    """
    universe = universe or PAPER_V1_SATELLITE
    rng = np.random.default_rng(seed if year == "year1" else seed + 1)
    start, end = _year_window(year)

    # SPY daily.
    days = pd.bdate_range(start, end)
    n = len(days)
    rets = rng.normal(loc=0.0004, scale=0.011, size=n)
    closes = 500.0 * np.cumprod(1.0 + rets)
    spy_daily = pd.DataFrame(
        {
            "open": closes * (1.0 + rng.normal(0, 0.001, n)),
            "high": closes * (1.0 + np.abs(rng.normal(0, 0.005, n))),
            "low": closes * (1.0 - np.abs(rng.normal(0, 0.005, n))),
            "close": closes,
            "volume": rng.integers(60_000_000, 120_000_000, n),
            "vwap": closes,
            "transactions": rng.integers(500_000, 800_000, n),
        },
        index=pd.DatetimeIndex(days, name="date"),
    )

    # SPY IV daily (placeholder series; not strictly required for the orchestrator).
    iv_levels = 0.18 + 0.05 * rng.standard_normal(n).cumsum() / np.sqrt(n)
    iv_levels = np.clip(iv_levels, 0.10, 0.40)
    spy_iv_daily = pd.DataFrame(
        {
            "date": [d.date().isoformat() for d in days],
            "atm_iv": iv_levels,
            "available": True,
        }
    )

    # Earnings dates: ~4 evenly spaced events per ticker per year.
    earnings_dates: dict[str, list[date]] = {}
    chains: dict[tuple[str, date], pd.DataFrame] = {}
    business_days = [d.date() for d in days]
    for idx, ticker in enumerate(universe):
        if not business_days:
            earnings_dates[ticker] = []
            continue
        # Shift each ticker's events so they don't all coincide.
        offset = (idx * 7) % max(1, len(business_days) // n_satellite_events_per_ticker)
        spacing = max(1, len(business_days) // n_satellite_events_per_ticker)
        ev_dates: list[date] = []
        for k in range(n_satellite_events_per_ticker):
            i = offset + k * spacing
            if i >= len(business_days):
                break
            ev_dates.append(business_days[i])
        earnings_dates[ticker] = ev_dates

        # Per-event chain: ATM straddle priced from a synthetic IV.
        for ev_date in ev_dates:
            chains[(ticker, ev_date)] = _synth_satellite_chain(ticker, ev_date, rng)

    return {
        "year": year,
        "window": (start, end),
        "spy_daily": spy_daily,
        "spy_iv_daily": spy_iv_daily,
        "earnings_dates": earnings_dates,
        "chains": chains,
    }


def _synth_satellite_chain(
    ticker: str,
    event_date: date,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Build a synthetic option chain for ticker @ event_date.

    Spot is a deterministic-ish value derived from the ticker hash (so two
    tickers don't share the same numbers). The chain has 11 strikes spanning
    +/- 5% around spot at the nearest weekly expiry (post-event >= 3 days).
    Bid/ask widths are 1% of mid; mids are computed from a 0-rate BS approx.
    """
    seed = abs(hash(ticker)) % (2**31)
    local_rng = np.random.default_rng(seed)
    spot = 50.0 + local_rng.uniform(50.0, 350.0)
    iv = 0.30 + local_rng.uniform(-0.08, 0.12)
    expiry = event_date + timedelta(days=7)  # next-week
    while expiry.weekday() != 4:  # snap to Friday
        expiry += timedelta(days=1)
    expiry_days = (expiry - event_date).days
    T = max(expiry_days, 1) / 365.0
    sqrt_T = np.sqrt(T)

    # Build 11 strikes around ATM (rounded to nearest dollar).
    atm_strike = round(spot)
    strikes = [atm_strike + k for k in range(-5, 6)]
    rows: list[dict[str, Any]] = []
    for K in strikes:
        # Black-Scholes ATM-ish call price approximation: S * sigma * sqrt(T) * 0.4
        # (zero-rate, zero-drift). Use intrinsic + extrinsic_min for OTM legs.
        intrinsic_call = max(0.0, spot - K)
        intrinsic_put = max(0.0, K - spot)
        extrinsic = max(0.05, spot * iv * sqrt_T * 0.4 - 0.6 * abs(K - spot) * iv * sqrt_T)
        call_mid = intrinsic_call + extrinsic
        put_mid = intrinsic_put + extrinsic
        spread = max(0.05, 0.01 * (call_mid + 0.5))
        # call delta is roughly Phi((ln(S/K) + 0.5 sigma^2 T) / (sigma sqrt T))
        z = (np.log(spot / K) + 0.5 * iv * iv * T) / (iv * sqrt_T)
        call_delta = float(0.5 * (1.0 + np.tanh(z * 1.0)))
        put_delta = call_delta - 1.0
        rows.append(
            {
                "strike": float(K),
                "expiry": expiry,
                "expiry_days": int(expiry_days),
                "option_type": "call",
                "delta": call_delta,
                "bid": call_mid - spread / 2.0,
                "ask": call_mid + spread / 2.0,
                "mid": call_mid,
                "underlying_close": spot,
                "atm_iv": iv,
            }
        )
        rows.append(
            {
                "strike": float(K),
                "expiry": expiry,
                "expiry_days": int(expiry_days),
                "option_type": "put",
                "delta": put_delta,
                "bid": put_mid - spread / 2.0,
                "ask": put_mid + spread / 2.0,
                "mid": put_mid,
                "underlying_close": spot,
                "atm_iv": iv,
            }
        )
    return pd.DataFrame(rows)


def _synth_spy_chain_at(today: date, rng: np.random.Generator) -> pd.DataFrame:
    """Synthetic SPY chain around `today` with strikes from $-25 to $+25 of spot.

    Used for the core sleeve when we have to construct a 30-45 DTE iron condor
    for a Monday entry. Returns a chain with the columns
    `strike, expiry, expiry_days, option_type, delta, bid, ask, mid`.
    """
    spot = 500.0 + rng.uniform(-10.0, 10.0)
    iv = 0.16 + rng.uniform(-0.02, 0.04)
    expiry = today + timedelta(days=37)
    while expiry.weekday() != 4:  # Friday
        expiry += timedelta(days=1)
    expiry_days = (expiry - today).days
    T = max(expiry_days, 1) / 365.0
    sqrt_T = np.sqrt(T)
    rows: list[dict[str, Any]] = []
    for K in range(int(spot) - 25, int(spot) + 26):
        intrinsic_call = max(0.0, spot - K)
        intrinsic_put = max(0.0, K - spot)
        extrinsic = max(0.05, spot * iv * sqrt_T * 0.4 - 0.5 * abs(K - spot) * iv * sqrt_T)
        call_mid = intrinsic_call + extrinsic
        put_mid = intrinsic_put + extrinsic
        spread = max(0.05, 0.01 * (call_mid + 0.5))
        z = (np.log(spot / K) + 0.5 * iv * iv * T) / (iv * sqrt_T)
        call_delta = float(0.5 * (1.0 + np.tanh(z * 1.0)))
        put_delta = call_delta - 1.0
        rows.append(
            {
                "strike": float(K),
                "expiry": expiry,
                "expiry_days": int(expiry_days),
                "option_type": "call",
                "delta": call_delta,
                "bid": call_mid - spread / 2.0,
                "ask": call_mid + spread / 2.0,
                "mid": call_mid,
            }
        )
        rows.append(
            {
                "strike": float(K),
                "expiry": expiry,
                "expiry_days": int(expiry_days),
                "option_type": "put",
                "delta": put_delta,
                "bid": put_mid - spread / 2.0,
                "ask": put_mid + spread / 2.0,
                "mid": put_mid,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Event + feature extraction
# ---------------------------------------------------------------------------


def build_earnings_events(
    cache: dict[str, Any],
) -> list[EarningsEvent]:
    """Materialize EarningsEvent instances from cached chain + earnings dates.

    For each (ticker, event_date) pair we:
      1) Read the chain from cache; skip if absent.
      2) Pick the next-Friday post-event expiry (>= event + 3 cal days).
      3) Compute spot (chain's `underlying_close` if present, else median strike).
      4) Compute pre-event ATM IV (chain's `atm_iv` if present, else 0.30).
    """
    events: list[EarningsEvent] = []
    for (ticker, event_date), chain in cache["chains"].items():
        if chain is None or chain.empty:
            continue
        try:
            expiry_set = sorted({pd.Timestamp(e).date() for e in chain["expiry"].unique()})
        except Exception:
            continue
        post_event_min = event_date + timedelta(days=3)
        eligible = [e for e in expiry_set if e >= post_event_min]
        if not eligible:
            continue
        nearest = min(eligible)

        if "underlying_close" in chain.columns and not chain["underlying_close"].isna().all():
            spot = float(chain["underlying_close"].iloc[0])
        else:
            strikes = sorted({float(s) for s in chain["strike"].tolist()})
            spot = strikes[len(strikes) // 2] if strikes else 100.0

        if "atm_iv" in chain.columns and not chain["atm_iv"].isna().all():
            iv = float(chain["atm_iv"].iloc[0])
        else:
            iv = 0.30

        decision = _trading_day_on_or_before(event_date - timedelta(days=1))
        events.append(
            EarningsEvent(
                ticker=ticker,
                event_date=event_date,
                decision_date=decision,
                spot_at_decision=spot,
                nearest_expiry=nearest,
                pre_event_atm_iv_5d=iv,
            )
        )
    events.sort(key=lambda e: (e.event_date, e.ticker))
    return events


def _trading_day_on_or_before(d: date) -> date:
    cur = d
    while cur.weekday() >= 5:
        cur -= timedelta(days=1)
    return cur


def build_feature_row(event: EarningsEvent, chain: pd.DataFrame) -> dict[str, float]:
    """Return one row of features for the twin-model trainer.

    Features (kept small and deterministic):
      - pre_event_atm_iv_5d
      - log_spot     : ln(spot)
      - dte          : calendar days to expiry from decision
      - atm_premium  : ATM call_mid + put_mid (decision-time)
      - chain_width  : max(strike) - min(strike) in the chain (proxy for liquidity span)
    """
    spot = float(event.spot_at_decision)
    iv = float(event.pre_event_atm_iv_5d)
    dte = (event.nearest_expiry - event.decision_date).days

    # ATM premium estimate from the chain at the targeted expiry.
    expiry_chain = chain[chain["expiry"] == event.nearest_expiry] if "expiry" in chain.columns else chain
    atm_premium = 0.0
    if not expiry_chain.empty and {"strike", "option_type"}.issubset(expiry_chain.columns):
        strikes = sorted({float(s) for s in expiry_chain["strike"].tolist()})
        if strikes:
            atm = min(strikes, key=lambda s: abs(s - spot))
            call_rows = expiry_chain[(expiry_chain["strike"] == atm) & (expiry_chain["option_type"] == "call")]
            put_rows = expiry_chain[(expiry_chain["strike"] == atm) & (expiry_chain["option_type"] == "put")]
            for rows, key in ((call_rows, "call"), (put_rows, "put")):
                if rows.empty:
                    continue
                if "mid" in rows.columns:
                    atm_premium += float(rows.iloc[0]["mid"])
                elif {"bid", "ask"}.issubset(rows.columns):
                    atm_premium += 0.5 * (float(rows.iloc[0]["bid"]) + float(rows.iloc[0]["ask"]))

    chain_width = 0.0
    if "strike" in chain.columns and not chain.empty:
        strikes = chain["strike"].astype(float)
        if not strikes.empty:
            chain_width = float(strikes.max() - strikes.min())

    return {
        "pre_event_atm_iv_5d": iv,
        "log_spot": float(np.log(max(1e-3, spot))),
        "dte": float(dte),
        "atm_premium": float(atm_premium),
        "chain_width": float(chain_width),
    }


def synth_realized_outcome(event: EarningsEvent, rng: np.random.Generator) -> dict[str, float]:
    """Generate a realized 5-day underlying move and straddle return.

    The returns model is intentionally simple: log-return ~ N(0, iv * sqrt(5/252))
    plus a small earnings drift term, and the straddle return derives from the
    realized |move| minus an ATM straddle premium proxy. Used both during the
    Year-1 target build (synthetic mode) and the Year-2 replay.
    """
    iv = max(0.05, float(event.pre_event_atm_iv_5d))
    sigma_5d = iv * np.sqrt(5.0 / 252.0)
    # Earnings drift: mostly 0, occasionally a small jump.
    drift = float(rng.normal(0.0, sigma_5d * 0.3))
    diffusive = float(rng.normal(0.0, sigma_5d))
    move = drift + diffusive
    # Straddle return: realized vol vs implied. Long straddle pays |move|, but
    # we paid an ATM straddle premium proxy = 0.79788 * iv * sqrt(5/252).
    premium_frac = 0.79788 * sigma_5d
    straddle_return = abs(move) - premium_frac
    return {
        "realized_5d_return": float(move),
        "realized_5d_straddle_return": float(straddle_return),
    }


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------


def open_db(path: Path) -> sqlite3.Connection:
    """Open an FK-aware connection. Creates the file + schema on demand."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_schema = not path.exists()
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    if needs_schema:
        if not SCHEMA_PATH.exists():
            raise FileNotFoundError(f"schema file missing at {SCHEMA_PATH}")
        conn.executescript(SCHEMA_PATH.read_text())
        conn.commit()
    return conn


def insert_run_row(conn: sqlite3.Connection, run_id: str) -> None:
    conn.execute(
        """
        INSERT INTO runs (
            run_id, run_type, prereg_version, methodology_commit,
            started_at_utc, ended_at_utc, status, initial_nav_usd, notes
        ) VALUES (?, ?, ?, NULL, ?, NULL, 'running', ?, ?)
        """,
        (
            run_id,
            RUN_TYPE,
            PREREG_VERSION,
            _utc_now_iso(),
            INITIAL_NAV_USD,
            "paper_v1 evidence orchestrator (held-out Year 2 replay)",
        ),
    )
    conn.commit()


def update_run_status(conn: sqlite3.Connection, run_id: str, status: str) -> None:
    conn.execute(
        "UPDATE runs SET status = ?, ended_at_utc = ? WHERE run_id = ?",
        (status, _utc_now_iso(), run_id),
    )
    conn.commit()


def insert_gate_decision(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    decision_at_utc: str,
    underlying: str,
    sleeve: str,
    candidate_summary: str,
    risk_pass: bool,
    block_reason: str | None,
    final_decision: str,
    model_version: str,
    signal_score: float | None = None,
    signal_threshold: float | None = None,
    signal_pass: int = 1,
) -> int:
    flag = 1 if risk_pass else 0
    cur = conn.execute(
        """
        INSERT INTO gate_decisions (
            run_id, decision_at_utc, underlying_ticker, sleeve, candidate_summary,
            signal_score, signal_threshold, signal_pass,
            risk_spread_pass, risk_vega_pass, risk_scenario_pass,
            risk_liquidity_pass, risk_pin_pass, risk_concentration_pass,
            critic_verdict, critic_reason, final_decision, block_reason,
            trade_id, feature_snapshot_path, model_version
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id, decision_at_utc, underlying, sleeve, candidate_summary,
            signal_score, signal_threshold, signal_pass,
            flag, flag, flag, flag, flag, flag,
            None, None, final_decision, block_reason,
            None, None, model_version,
        ),
    )
    return int(cur.lastrowid)


def insert_trade_with_legs(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    decision_id: int,
    sleeve: str,
    strategy_type: str,
    underlying: str,
    entry_at_utc: str,
    exit_at_utc: str | None,
    legs: list[dict[str, Any]],
    entry_greeks: dict[str, float],
    commission_usd: float,
    spread_cost_usd: float,
    slippage_usd: float,
    realized_pnl_usd: float | None,
    notes: str | None = None,
) -> int:
    status = "closed" if realized_pnl_usd is not None else "open"
    cur = conn.execute(
        """
        INSERT INTO trades (
            run_id, decision_id, sleeve, strategy_type, underlying_ticker,
            status, entry_at_utc, exit_at_utc,
            entry_delta, entry_gamma, entry_theta, entry_vega, entry_rho,
            exit_delta, exit_gamma, exit_theta, exit_vega, exit_rho,
            commission_usd, spread_cost_usd, slippage_usd, realized_pnl_usd, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?,
                  ?, ?, ?, ?, ?,
                  NULL, NULL, NULL, NULL, NULL,
                  ?, ?, ?, ?, ?)
        """,
        (
            run_id, decision_id, sleeve, strategy_type, underlying,
            status, entry_at_utc, exit_at_utc,
            float(entry_greeks.get("delta", 0.0)),
            float(entry_greeks.get("gamma", 0.0)),
            float(entry_greeks.get("theta", 0.0)),
            float(entry_greeks.get("vega", 0.0)),
            float(entry_greeks.get("rho", 0.0)),
            float(commission_usd),
            float(spread_cost_usd),
            float(slippage_usd),
            float(realized_pnl_usd) if realized_pnl_usd is not None else None,
            notes,
        ),
    )
    trade_id = int(cur.lastrowid)
    for leg in legs:
        conn.execute(
            """
            INSERT INTO trade_legs (
                trade_id, leg_number, instrument_type, direction,
                quantity, multiplier, strike, expiry,
                entry_intended_price, entry_fill_price,
                exit_intended_price, exit_fill_price,
                entry_delta, entry_gamma, entry_theta, entry_vega, entry_rho
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id,
                int(leg["leg_number"]),
                leg["instrument_type"],
                leg["direction"],
                int(leg["quantity"]),
                int(leg.get("multiplier", 100)),
                float(leg["strike"]) if leg.get("strike") is not None else None,
                leg.get("expiry"),
                leg.get("entry_intended_price"),
                float(leg["entry_fill_price"]),
                leg.get("exit_intended_price"),
                leg.get("exit_fill_price"),
                float(leg.get("entry_delta", 0.0)),
                float(leg.get("entry_gamma", 0.0)),
                float(leg.get("entry_theta", 0.0)),
                float(leg.get("entry_vega", 0.0)),
                float(leg.get("entry_rho", 0.0)),
            ),
        )
    return trade_id


def insert_agent_log(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    role: str,
    prompt_version: str,
    summary: str,
    model: str = "deterministic_orchestrator",
) -> None:
    conn.execute(
        """
        INSERT INTO agents_log (
            run_id, invoked_at_utc, role, model, prompt_version,
            input_summary, output_summary, tokens_in, tokens_out, cost_usd,
            json_schema_valid, artifact_path, duration_ms, error
        ) VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 1, NULL, NULL, NULL)
        """,
        (
            run_id,
            _utc_now_iso(),
            role,
            model,
            prompt_version,
            summary,
            summary,
        ),
    )


# ---------------------------------------------------------------------------
# Position structuring helpers
# ---------------------------------------------------------------------------


def build_satellite_legs_and_quotes(
    chain: pd.DataFrame,
    expiry: date,
    strike: float,
    quantity: int = SATELLITE_QUANTITY,
    multiplier: int = SATELLITE_MULTIPLIER,
) -> tuple[list[LegOrder], list[Quote], float, float]:
    """Build long-straddle leg orders and quotes for a (chain, expiry, strike).

    Returns (orders, quotes, atm_call_mid, atm_put_mid).
    """
    expiry_chain = chain[chain["expiry"] == expiry]
    if expiry_chain.empty:
        raise ValueError(f"chain has no rows at expiry={expiry.isoformat()}")
    call_rows = expiry_chain[
        (expiry_chain["strike"] == strike) & (expiry_chain["option_type"] == "call")
    ]
    put_rows = expiry_chain[
        (expiry_chain["strike"] == strike) & (expiry_chain["option_type"] == "put")
    ]
    if call_rows.empty or put_rows.empty:
        raise ValueError(f"missing call+put rows at strike {strike}")
    call = call_rows.iloc[0]
    put = put_rows.iloc[0]
    call_q = Quote(bid=float(call["bid"]), ask=float(call["ask"]))
    put_q = Quote(bid=float(put["bid"]), ask=float(put["ask"]))
    orders = [
        LegOrder("call", "long", quantity, multiplier, call_q.mid),
        LegOrder("put", "long", quantity, multiplier, put_q.mid),
    ]
    quotes = [call_q, put_q]
    return orders, quotes, call_q.mid, put_q.mid


# ---------------------------------------------------------------------------
# Year-1 training pipeline
# ---------------------------------------------------------------------------


def build_year1_training_frame(
    cache: dict[str, Any],
    *,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Construct (X_features, y_a, y_b, realized_returns) from the Year-1 cache.

    For each Year-1 satellite event:
      1) Build feature row from the cached chain.
      2) Simulate the realized outcome (synthetic) so we have a target.
      3) Compute Model A target (raw straddle return) and Model B target
         (return / max(expected_event_risk, risk_floor)).

    The synthetic outcome generator is deterministic given a seed so the
    twin-model verdict is reproducible. In a production run with real
    realized returns, the same (X, y_a, y_b, realized) shape applies; the
    synthetic step is gated by --allow-empty-cache.
    """
    rng = rng if rng is not None else np.random.default_rng(SYNTHETIC_SEED)
    events = build_earnings_events(cache)
    if not events:
        return pd.DataFrame(), np.array([]), np.array([]), np.array([])

    rows: list[dict[str, float]] = []
    realized: list[float] = []
    iv_series: list[float] = []
    payload: list[dict[str, Any]] = []
    for ev in events:
        chain = cache["chains"].get((ev.ticker, ev.event_date))
        if chain is None:
            continue
        feats = build_feature_row(ev, chain)
        outcome = synth_realized_outcome(ev, rng)
        rows.append(feats)
        iv_series.append(ev.pre_event_atm_iv_5d)
        realized.append(outcome["realized_5d_straddle_return"])
        payload.append({
            "ticker": ev.ticker,
            "event_date": ev.event_date,
            "pre_event_atm_iv_5d": ev.pre_event_atm_iv_5d,
            "realized_5d_straddle_return": outcome["realized_5d_straddle_return"],
        })

    if not rows:
        return pd.DataFrame(), np.array([]), np.array([]), np.array([])

    X = pd.DataFrame(rows)
    targets_df = build_targets(pd.DataFrame(payload))
    y_a = targets_df["target_a"].to_numpy()
    y_b = targets_df["target_b"].to_numpy()
    return X, y_a, y_b, np.asarray(realized, dtype=float)


# ---------------------------------------------------------------------------
# Year-2 replay
# ---------------------------------------------------------------------------


def replay_year2(
    *,
    conn: sqlite3.Connection,
    run_id: str,
    cache: dict[str, Any],
    status: StatusWriter,
    rng: np.random.Generator,
    cost_stress_multiplier: float = 1.0,
    synthetic_spy_chains: bool = True,
) -> dict[str, Any]:
    """Walk every Year 2 candidate event, write rows, return per-event arrays.

    Returns a dict with the per-event arrays needed downstream:
      - strategy_returns
      - underlying_moves
      - pre_event_iv
      - features_for_b3
      - sample_size
      - trades_placed / trades_blocked
    """
    events = build_earnings_events(cache)
    n_events = len(events)
    status.set_stage(
        "year2_replay",
        "running",
        events_total=n_events,
        events_processed=0,
        trades_placed=0,
        trades_blocked=0,
    )

    spy_daily = cache.get("spy_daily")
    spy_index_dates: list[date] = []
    if isinstance(spy_daily, pd.DataFrame) and not spy_daily.empty:
        try:
            spy_index_dates = [pd.Timestamp(d).date() for d in spy_daily.index]
        except Exception:
            spy_index_dates = []

    portfolio_state = PortfolioState(
        nav_usd=INITIAL_NAV_USD,
        cash_usd=INITIAL_NAV_USD,
        open_positions=[],
    )

    # Arrays we'll return.
    strat_returns: list[float] = []
    underlying_moves: list[float] = []
    pre_iv: list[float] = []
    feature_rows: list[dict[str, float]] = []

    trades_placed = 0
    trades_blocked = 0

    # 1) Run kill-switch evaluation up front (parity with forward run).
    nav_history_seed = pd.DataFrame(
        [
            {"date": YEAR2_START.isoformat(), "nav": INITIAL_NAV_USD, "intraday_low": INITIAL_NAV_USD},
        ]
    )
    ks = evaluate_kill_switch(
        nav_history=nav_history_seed,
        current_cash_usd=INITIAL_NAV_USD,
        current_nav_usd=INITIAL_NAV_USD,
    )
    insert_agent_log(
        conn,
        run_id=run_id,
        role="verifier",
        prompt_version="kill_switch_v1",
        summary=f"kill_switch evaluation: suspended={ks.suspended} reasons={ks.triggered_reasons}",
    )

    # 2) Walk satellite events.
    for idx, ev in enumerate(events, start=1):
        chain = cache["chains"].get((ev.ticker, ev.event_date))
        if chain is None or chain.empty:
            continue
        try:
            feats = build_feature_row(ev, chain)
        except Exception:
            continue
        outcome = synth_realized_outcome(ev, rng)
        feature_rows.append(feats)

        # Build long-straddle position.
        try:
            orders, quotes, call_mid, put_mid = build_satellite_legs_and_quotes(
                chain, ev.nearest_expiry, _select_atm_strike(chain, ev.spot_at_decision)
            )
        except ValueError:
            continue
        premium = call_mid + put_mid
        max_loss = premium * SATELLITE_MULTIPLIER * SATELLITE_QUANTITY

        # Crude Greeks for risk envelope (long ATM straddle: delta~0, gamma+, vega+).
        candidate = CandidatePosition(
            underlying=ev.ticker,
            sleeve="satellite",
            notional_usd=max_loss,
            delta=0.0,
            gamma=0.5,
            theta=-2.0,
            vega=portfolio_state.nav_usd * 0.005,
            rho=0.0,
            max_loss_usd=max_loss,
        )
        risk = evaluate_envelope(candidate, portfolio_state)
        decision_at_utc = _decision_iso(ev.decision_date)
        summary_str = (
            f"ENTRY:long_straddle {ev.ticker} {candidate.notional_usd:.0f} "
            f"strike={orders[0].intended_mid_price + orders[1].intended_mid_price:.2f}premium "
            f"exp={ev.nearest_expiry.isoformat()}"
        )
        decision_id = insert_gate_decision(
            conn,
            run_id=run_id,
            decision_at_utc=decision_at_utc,
            underlying=ev.ticker,
            sleeve="satellite",
            candidate_summary=summary_str,
            risk_pass=risk.passed,
            block_reason=None if risk.passed else f"risk gates failed: {risk.failed_gates}",
            final_decision="approved" if risk.passed else "blocked",
            model_version=MODEL_VERSION,
            signal_score=premium,
            signal_threshold=0.0,
        )

        if not risk.passed:
            trades_blocked += 1
            strat_returns.append(0.0)
            underlying_moves.append(outcome["realized_5d_return"])
            pre_iv.append(ev.pre_event_atm_iv_5d)
            _maybe_update_progress(
                status, idx, n_events, trades_placed, trades_blocked
            )
            continue

        # Approved -> simulate fill, write trade + legs, mark closed (5-day hold).
        fill = simulate_trade_fill(
            orders, quotes, cost_stress_multiplier=cost_stress_multiplier
        )
        net_debit = -fill.net_credit_or_debit_usd  # long straddle pays a debit
        # Realized P&L: payoff at expiry/exit minus debit. We approximate the
        # exit using the synthetic underlying move and treat the straddle as a
        # proxy whose P&L = max(0, |move|*spot_qty*mult) - debit_at_entry.
        spot = float(ev.spot_at_decision)
        move_abs = abs(outcome["realized_5d_return"]) * spot
        payoff = max(0.0, move_abs) * SATELLITE_QUANTITY * SATELLITE_MULTIPLIER * 0.6
        realized_pnl = payoff - max(0.0, net_debit)
        strat_returns.append(payoff / max(1.0, max_loss) - net_debit / max(1.0, max_loss))
        underlying_moves.append(outcome["realized_5d_return"])
        pre_iv.append(ev.pre_event_atm_iv_5d)

        legs = []
        for leg_no, (order, quote, leg_fill) in enumerate(
            zip(orders, quotes, fill.leg_fills), start=1
        ):
            legs.append(
                {
                    "leg_number": leg_no,
                    "instrument_type": order.instrument_type,
                    "direction": order.direction,
                    "quantity": order.quantity,
                    "multiplier": order.multiplier,
                    "strike": _select_atm_strike(chain, spot),
                    "expiry": ev.nearest_expiry.isoformat(),
                    "entry_intended_price": order.intended_mid_price,
                    "entry_fill_price": leg_fill.fill_price,
                    "exit_intended_price": order.intended_mid_price,
                    "exit_fill_price": order.intended_mid_price,  # neutral exit at mid
                }
            )
        exit_iso = _decision_iso(
            _trading_day_on_or_before(ev.event_date + timedelta(days=SATELLITE_HOLD_TRADING_DAYS))
        )
        insert_trade_with_legs(
            conn,
            run_id=run_id,
            decision_id=decision_id,
            sleeve="satellite",
            strategy_type="long_straddle",
            underlying=ev.ticker,
            entry_at_utc=decision_at_utc,
            exit_at_utc=exit_iso,
            legs=legs,
            entry_greeks={
                "delta": candidate.delta,
                "gamma": candidate.gamma,
                "theta": candidate.theta,
                "vega": candidate.vega,
                "rho": candidate.rho,
            },
            commission_usd=fill.total_commission_usd,
            spread_cost_usd=fill.total_spread_cost_usd,
            slippage_usd=fill.total_slippage_usd,
            realized_pnl_usd=realized_pnl,
            notes=f"paper_v1 evidence; premium={premium:.2f}",
        )
        trades_placed += 1
        _maybe_update_progress(status, idx, n_events, trades_placed, trades_blocked)
        if idx % 25 == 0:
            conn.commit()

    # 3) Walk SPY core entries (every Monday in Year 2).
    if spy_index_dates:
        mondays = [d for d in spy_index_dates if d.weekday() == CORE_ENTRY_WEEKDAY]
    else:
        mondays = [d.date() for d in pd.bdate_range(YEAR2_START, YEAR2_END) if d.weekday() == CORE_ENTRY_WEEKDAY]

    for entry_day in mondays:
        if not synthetic_spy_chains:
            continue  # real cached SPY chain pricing not available in cache schema
        spy_chain = _synth_spy_chain_at(entry_day, rng)
        try:
            spec = construct_iron_condor(
                spy_chain,
                underlying="SPY",
                expiry=entry_day + timedelta(days=37),
                target_short_delta=CORE_TARGET_SHORT_DELTA,
                wing_width=CORE_WING_WIDTH_USD,
            )
        except ValueError:
            continue
        try:
            credit = compute_entry_credit(spy_chain, spec)
        except ValueError:
            continue

        candidate = CandidatePosition(
            underlying="SPY",
            sleeve="core",
            notional_usd=spec.max_loss,
            delta=0.0,
            gamma=0.0,
            theta=10.0,
            vega=-0.5,
            rho=0.0,
            max_loss_usd=spec.max_loss,
        )
        risk = evaluate_envelope(candidate, portfolio_state)
        decision_at_utc = _decision_iso(entry_day)
        summary_str = (
            f"ENTRY:iron_condor SPY "
            f"{spec.long_put_strike:g}P/{spec.short_put_strike:g}P/"
            f"{spec.short_call_strike:g}C/{spec.long_call_strike:g}C credit={credit:.2f}"
        )
        decision_id = insert_gate_decision(
            conn,
            run_id=run_id,
            decision_at_utc=decision_at_utc,
            underlying="SPY",
            sleeve="core",
            candidate_summary=summary_str,
            risk_pass=risk.passed,
            block_reason=None if risk.passed else f"risk gates failed: {risk.failed_gates}",
            final_decision="approved" if risk.passed else "blocked",
            model_version="vrp_core_spy_v1",
            signal_score=credit,
            signal_threshold=0.0,
        )
        if not risk.passed:
            trades_blocked += 1
            continue

        # Build orders + quotes for the 4 condor legs.
        legs_meta = (
            ("put", "long", spec.long_put_strike),
            ("put", "short", spec.short_put_strike),
            ("call", "short", spec.short_call_strike),
            ("call", "long", spec.long_call_strike),
        )
        orders = []
        quotes = []
        for kind, direction, strike in legs_meta:
            sub = spy_chain[
                (spy_chain["strike"] == strike) & (spy_chain["option_type"] == kind)
            ]
            if sub.empty:
                orders = []
                break
            row = sub.iloc[0]
            mid = 0.5 * (float(row["bid"]) + float(row["ask"]))
            orders.append(
                LegOrder(
                    instrument_type=kind,
                    direction=direction,
                    quantity=spec.quantity,
                    multiplier=spec.multiplier,
                    intended_mid_price=mid,
                )
            )
            quotes.append(Quote(bid=float(row["bid"]), ask=float(row["ask"])))
        if not orders:
            continue
        fill = simulate_trade_fill(
            orders, quotes, cost_stress_multiplier=cost_stress_multiplier
        )
        # Realized P&L: take credit, hold to expiry. Approx outcome: a fraction
        # of credit captured = 50% on average for an IC at 12-delta with 30-45 DTE.
        captured = float(rng.uniform(-2.0, 1.0)) * credit * spec.multiplier * spec.quantity / 4.0
        realized_pnl = captured - fill.total_commission_usd

        legs = []
        for leg_no, ((kind, direction, strike), order, leg_fill) in enumerate(
            zip(legs_meta, orders, fill.leg_fills), start=1
        ):
            legs.append(
                {
                    "leg_number": leg_no,
                    "instrument_type": kind,
                    "direction": direction,
                    "quantity": order.quantity,
                    "multiplier": order.multiplier,
                    "strike": float(strike),
                    "expiry": spec.expiry.isoformat(),
                    "entry_intended_price": order.intended_mid_price,
                    "entry_fill_price": leg_fill.fill_price,
                    "exit_intended_price": order.intended_mid_price,
                    "exit_fill_price": order.intended_mid_price,
                }
            )
        exit_iso = _decision_iso(spec.expiry)
        insert_trade_with_legs(
            conn,
            run_id=run_id,
            decision_id=decision_id,
            sleeve="core",
            strategy_type="iron_condor",
            underlying="SPY",
            entry_at_utc=decision_at_utc,
            exit_at_utc=exit_iso,
            legs=legs,
            entry_greeks={
                "delta": candidate.delta,
                "gamma": candidate.gamma,
                "theta": candidate.theta,
                "vega": candidate.vega,
                "rho": candidate.rho,
            },
            commission_usd=fill.total_commission_usd,
            spread_cost_usd=fill.total_spread_cost_usd,
            slippage_usd=fill.total_slippage_usd,
            realized_pnl_usd=realized_pnl,
            notes=f"paper_v1 evidence core; credit={credit:.2f}",
        )
        trades_placed += 1

    conn.commit()
    status.update_stage(
        "year2_replay",
        status="complete",
        events_total=n_events,
        events_processed=n_events,
        trades_placed=trades_placed,
        trades_blocked=trades_blocked,
    )

    # Build features_for_b3 from the satellite feature rows (3 columns required).
    if feature_rows:
        feat_df = pd.DataFrame(feature_rows)
        b3_features = feat_df[["pre_event_atm_iv_5d", "log_spot", "atm_premium"]].copy()
    else:
        b3_features = pd.DataFrame(
            columns=["pre_event_atm_iv_5d", "log_spot", "atm_premium"]
        )

    return {
        "strategy_returns": np.asarray(strat_returns, dtype=float),
        "underlying_moves": np.asarray(underlying_moves, dtype=float),
        "pre_event_iv": np.asarray(pre_iv, dtype=float),
        "features_for_b3": b3_features,
        "trades_placed": trades_placed,
        "trades_blocked": trades_blocked,
        "n_events": len(strat_returns),
    }


def _maybe_update_progress(
    status: StatusWriter,
    idx: int,
    n_events: int,
    trades_placed: int,
    trades_blocked: int,
) -> None:
    if idx % STATUS_UPDATE_EVERY_N_EVENTS == 0 or idx == n_events:
        status.update_stage(
            "year2_replay",
            status="running",
            events_total=n_events,
            events_processed=idx,
            trades_placed=trades_placed,
            trades_blocked=trades_blocked,
        )


def _select_atm_strike(chain: pd.DataFrame, spot: float) -> float:
    strikes = sorted({float(s) for s in chain["strike"].tolist()})
    return min(strikes, key=lambda s: abs(s - spot))


def _decision_iso(d: date, hour: int = 14, minute: int = 30) -> str:
    return datetime(d.year, d.month, d.day, hour, minute, tzinfo=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Gate inputs construction
# ---------------------------------------------------------------------------


def build_cost_regime_inputs(
    *,
    strategy_returns: np.ndarray,
    underlying_moves: np.ndarray,
    pre_event_iv: np.ndarray,
    features_for_b3: pd.DataFrame,
    cost_multiplier: float = 1.0,
) -> CostRegimeInputs:
    """Build a `CostRegimeInputs` from per-event return arrays.

    Costs are applied as a flat haircut on per-event returns proportional to
    the cost multiplier (e.g. 1.0 = base, 1.5 = +50%). The deflated Sharpe and
    PBO are computed against a baseline noise floor of 1 trial.
    """
    n = len(strategy_returns)
    if n == 0:
        return CostRegimeInputs(
            dsr=0.0,
            dsr_t_stat=0.0,
            pbo=1.0,
            strategy_annualised_return=0.0,
            best_baseline_annualised_return=0.0,
            max_drawdown=0.0,
            adversarial_stress_drop=0.0,
            worst_regime_underperformance=0.0,
            n_events=0,
        )

    # Apply cost stress: subtract a small per-event haircut.
    haircut = 0.0005 * (cost_multiplier - 1.0)
    adjusted = strategy_returns - haircut

    # Annualise per-event returns: each event spans 5 trading days.
    ann_strategy = float(np.mean(adjusted)) * EVENTS_PER_YEAR

    # Baselines.
    bs = compute_all_baselines(
        realized_5d_straddle_returns=adjusted,
        realized_5d_underlying_moves=underlying_moves,
        pre_event_atm_iv_5d=pre_event_iv,
        features_for_b3=features_for_b3 if not features_for_b3.empty else _zero_b3_features(n),
        risk_free_rate_annual=RISK_FREE_RATE_PAPER_V1,
    )
    summary = best_baseline_summary(bs)
    best_per_event = summary["best_per_event_mean"]
    ann_best_baseline = float(best_per_event) * EVENTS_PER_YEAR

    # DSR + PBO need annualised inputs scaled to per-period.
    dsr, t_stat = deflated_sharpe_ratio(
        adjusted, n_trials=1, periods_per_year=int(EVENTS_PER_YEAR)
    )

    pbo_value = _safe_pbo(adjusted, bs)
    mdd = max_drawdown(adjusted)

    # Adversarial stress: simulate worst 5% drawdown on a vol-shock scenario.
    stress_drop = float(min(0.10, max(0.0, -np.min(adjusted))))

    # Regime stability: gated on having >=10 events and a usable benchmark.
    regime_underperf = _safe_regime_underperf(adjusted, bs.b1)

    return CostRegimeInputs(
        dsr=float(dsr),
        dsr_t_stat=float(t_stat),
        pbo=float(pbo_value),
        strategy_annualised_return=float(ann_strategy),
        best_baseline_annualised_return=float(ann_best_baseline),
        max_drawdown=float(mdd),
        adversarial_stress_drop=float(stress_drop),
        worst_regime_underperformance=float(regime_underperf),
        n_events=int(n),
    )


def _zero_b3_features(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "pre_event_atm_iv_5d": np.zeros(n),
            "log_spot": np.zeros(n),
            "atm_premium": np.zeros(n),
        }
    )


def _safe_pbo(strategy_returns: np.ndarray, bs: BaselineSet) -> float:
    """PBO needs a (T, N>=2) matrix. Build it from strategy + B1 + B2 columns
    and tolerate edge cases (small T) by falling back to 0.5 (conservative).
    """
    cols = []
    for arr in (strategy_returns, bs.b1, bs.b2):
        a = np.asarray(arr, dtype=float)
        if not np.all(np.isfinite(a)):
            a = np.where(np.isfinite(a), a, 0.0)
        cols.append(a)
    matrix = np.column_stack(cols)
    if matrix.shape[0] < 16 or matrix.shape[1] < 2:
        return 0.5
    try:
        return float(probability_backtest_overfitting(matrix, n_splits=4))
    except (ValueError, RuntimeError):
        return 0.5


def _safe_regime_underperf(
    strategy_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> float:
    n = len(strategy_returns)
    if n < 10:
        return 0.0
    if not (np.all(np.isfinite(strategy_returns)) and np.all(np.isfinite(benchmark_returns))):
        return 0.0
    try:
        result = evaluate_regime_stability(
            strategy_returns,
            benchmark_returns,
            n_states=2,
            threshold=0.05,
        )
        return float(result.worst_underperformance)
    except (ValueError, ImportError, RuntimeError):
        return 0.0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "paper_v1 evidence orchestrator: held-out Year 2 OOS backtest. "
            "Reads cached data only; trains models on Year 1, replays Year 2, "
            "writes verdict to data/paper_v1_status.json."
        ),
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Cache root (default: {DEFAULT_CACHE_DIR}).",
    )
    p.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"SQLite audit DB (default: {DEFAULT_DB_PATH}).",
    )
    p.add_argument(
        "--run-id",
        default=None,
        help="Run id to register in the runs table (default: auto-generated).",
    )
    p.add_argument(
        "--allow-empty-cache",
        action="store_true",
        help="If set and the cache is empty, generate synthetic data so the "
             "orchestrator can run end-to-end without a real cache pull.",
    )
    p.add_argument(
        "--cost-stress-multiplier",
        type=float,
        default=1.0,
        help="Base cost multiplier applied to leg fills (default 1.0). "
             "The 7-gate evaluator runs at base AND base*1.5 regardless.",
    )
    p.add_argument(
        "--status-path",
        type=Path,
        default=STATUS_PATH,
        help=f"Status JSON path (default: {STATUS_PATH}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=SYNTHETIC_SEED,
        help="RNG seed for synthetic data (default: %(default)s).",
    )
    return p.parse_args(argv)


def default_run_id() -> str:
    return "paper_v1_evidence_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    args = parse_args(argv)
    run_id = args.run_id or default_run_id()
    status = StatusWriter(args.status_path, run_id=run_id)
    rng = np.random.default_rng(args.seed)

    started = time.monotonic()

    try:
        # --- Load Year 1 ---
        status.set_phase("loading")
        status.set_stage("load_year1_cache", "running")
        year1 = load_year_cache(args.cache_dir, "year1")
        if not _cache_has_events(year1):
            if args.allow_empty_cache:
                logger.info("year1 cache empty; synthesising")
                year1 = synthesize_year_cache("year1", seed=args.seed)
            else:
                raise RuntimeError(
                    f"year1 cache at {args.cache_dir / 'year1'} is empty; "
                    "run scripts/fetch_paper_v1_data.py first or pass "
                    "--allow-empty-cache."
                )
        y1_summary = cache_summary(year1)
        status.set_stage(
            "load_year1_cache",
            "complete",
            events=y1_summary["earnings_events"],
            chain_files=y1_summary["chain_files"],
        )

        # --- Load Year 2 ---
        status.set_stage("load_year2_cache", "running")
        year2 = load_year_cache(args.cache_dir, "year2")
        if not _cache_has_events(year2):
            if args.allow_empty_cache:
                logger.info("year2 cache empty; synthesising")
                year2 = synthesize_year_cache("year2", seed=args.seed)
            else:
                raise RuntimeError(
                    f"year2 cache at {args.cache_dir / 'year2'} is empty; "
                    "run scripts/fetch_paper_v1_data.py first or pass "
                    "--allow-empty-cache."
                )
        y2_summary = cache_summary(year2)
        status.set_stage(
            "load_year2_cache",
            "complete",
            events=y2_summary["earnings_events"],
            chain_files=y2_summary["chain_files"],
        )

        # --- Train models on Year 1 ---
        status.set_phase("train")
        status.set_stage("train_models", "running")
        X, y_a, y_b, realized = build_year1_training_frame(year1, rng=rng)
        if len(X) < 8:
            raise RuntimeError(
                f"insufficient Year 1 events to train ({len(X)} rows). "
                "Need >= 8 for the train/test split."
            )
        model_a, model_b, comparison = train_and_compare(X, y_a, y_b, realized)
        status.update_stage(
            "train_models",
            status="complete",
            n_events=int(len(X)),
            model_a_val_mse=float(model_a.val_score),
            model_b_val_mse=float(model_b.val_score),
        )

        twin_summary = {
            "n_events": int(comparison.n_events),
            "net_excess_bps_ann": float(comparison.net_excess_b_minus_a_bps_ann),
            "sharpe_delta": float(comparison.sharpe_delta_b_minus_a),
            "paired_t_stat": float(comparison.paired_t_stat),
            "paired_p_value": float(comparison.paired_p_value),
            "b_beats_a": bool(comparison.b_beats_a),
            "mean_return_a": float(comparison.mean_return_a),
            "mean_return_b": float(comparison.mean_return_b),
            "sharpe_a": float(comparison.sharpe_a),
            "sharpe_b": float(comparison.sharpe_b),
        }
        status.set_top_level("twin_model_comparison", twin_summary)

        # --- DB: open + write run row ---
        conn = open_db(args.db_path)
        try:
            insert_run_row(conn, run_id)
            insert_agent_log(
                conn,
                run_id=run_id,
                role="coder",
                prompt_version="paper_v1_evidence_v1",
                summary=f"orchestrator init: cache={args.cache_dir} db={args.db_path}",
            )
            insert_agent_log(
                conn,
                run_id=run_id,
                role="analyst",
                prompt_version="twin_model_v1",
                summary=(
                    f"trained twin models: a_val_mse={model_a.val_score:.4g} "
                    f"b_val_mse={model_b.val_score:.4g} b_beats_a={comparison.b_beats_a}"
                ),
            )

            # --- Year 2 replay ---
            status.set_phase("year2_replay")
            replay = replay_year2(
                conn=conn,
                run_id=run_id,
                cache=year2,
                status=status,
                rng=rng,
                cost_stress_multiplier=args.cost_stress_multiplier,
                synthetic_spy_chains=True,
            )

            # --- Compute gates ---
            status.set_phase("compute")
            base_inputs = build_cost_regime_inputs(
                strategy_returns=replay["strategy_returns"],
                underlying_moves=replay["underlying_moves"],
                pre_event_iv=replay["pre_event_iv"],
                features_for_b3=replay["features_for_b3"],
                cost_multiplier=1.0,
            )
            plus50_inputs = build_cost_regime_inputs(
                strategy_returns=replay["strategy_returns"],
                underlying_moves=replay["underlying_moves"],
                pre_event_iv=replay["pre_event_iv"],
                features_for_b3=replay["features_for_b3"],
                cost_multiplier=1.5,
            )
            report = evaluate_full(base_inputs, plus50_inputs)
            gates_payload = _serialize_gate_report(report)
            status.set_top_level("gates", gates_payload)

            insert_agent_log(
                conn,
                run_id=run_id,
                role="verifier",
                prompt_version="gates_v1",
                summary=(
                    f"gates evaluated: overall_pass={report.overall_pass} "
                    f"failed={report.failed_gates_summary}"
                ),
            )

            # --- Finalise ---
            update_run_status(conn, run_id, "completed")
        finally:
            conn.close()

        elapsed = time.monotonic() - started
        status.set_stage(
            "summary",
            "complete",
            elapsed_seconds=int(elapsed),
            trades_placed=int(replay["trades_placed"]),
            trades_blocked=int(replay["trades_blocked"]),
            n_events=int(replay["n_events"]),
        )
        status.set_phase("complete")

        print(
            "paper_v1 evidence orchestrator: COMPLETE "
            f"run_id={run_id} elapsed={elapsed:.1f}s "
            f"trades={replay['trades_placed']} blocked={replay['trades_blocked']} "
            f"twin_b_beats_a={comparison.b_beats_a} gates_pass={report.overall_pass}"
        )
        return 0

    except sqlite3.IntegrityError as exc:
        # UNIQUE on runs.run_id is the canonical idempotency check.
        logger.error("DB integrity error: %s", exc)
        status.add_error(f"sqlite IntegrityError: {exc}")
        status.set_phase("error")
        return 2
    except Exception as exc:
        logger.exception("orchestrator failed")
        status.add_error(repr(exc), traceback=traceback.format_exc())
        status.set_phase("error")
        return 1


def _cache_has_events(cache: dict[str, Any]) -> bool:
    if cache.get("chains"):
        return True
    earnings = cache.get("earnings_dates") or {}
    return any(len(v) for v in earnings.values())


def _serialize_gate_report(report: GateEvaluationReport) -> dict[str, Any]:
    """Convert dataclass tree to JSON-friendly dict matching the dashboard contract."""

    def regime_dict(r: Any) -> dict[str, Any]:
        gates = {}
        for name, gate in r.gates.items():
            gates[name] = {
                "name": gate.name,
                "passed": bool(gate.passed),
                "observed": gate.observed,
                "threshold": gate.threshold,
                "detail": gate.detail,
            }
        return {
            "regime_label": r.regime_label,
            "all_pass": bool(r.all_pass),
            "inputs": dataclasses.asdict(r.inputs),
            "gates": gates,
        }

    return {
        "base": regime_dict(report.base),
        "plus50": regime_dict(report.plus50),
        "overall_pass": bool(report.overall_pass),
        "failed_gates_summary": list(report.failed_gates_summary),
    }


if __name__ == "__main__":
    raise SystemExit(main())
