"""Pro live trading dashboard — TradingView-style dark theme.

Top bar      : aggregate NAV, today P&L, total open P&L
Strategy grid: per-strategy attribution + P&L
Positions    : open positions table with live mark-to-market
Activity     : recent order fills/cancels
NAV chart    : 24h sparkline per account

Run:
    .venv/bin/python dashboard/pro_live.py
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time as _time
from collections import defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import requests
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse


# ----------------------------- TTL cache --------------------------------

class TTLCache:
    """Thread-safe TTL cache. Calls expensive function once per TTL window;
    all callers within that window get the cached value instantly."""
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, tuple[float, Any]] = {}
        # Per-key locks so concurrent first-time misses serialize on the same fetch
        self._key_locks: dict[str, threading.Lock] = {}

    def get(self, key: str, ttl_secs: float, fetch: Callable[[], Any]) -> Any:
        now = _time.time()
        with self._lock:
            entry = self._data.get(key)
            if entry and (now - entry[0]) < ttl_secs:
                return entry[1]
            klock = self._key_locks.setdefault(key, threading.Lock())
        # Acquire per-key lock to avoid duplicate work
        with klock:
            now = _time.time()
            with self._lock:
                entry = self._data.get(key)
                if entry and (now - entry[0]) < ttl_secs:
                    return entry[1]
            value = fetch()
            with self._lock:
                self._data[key] = (_time.time(), value)
            return value


CACHE = TTLCache()


ROOT = Path("/Users/weeoto/turbo-carnival")
ENV_FILE = ROOT / "council" / ".env.local"


# --- account & strategy mapping ----------------------------------------

ALPACA_ACCOUNTS = [
    {"label": "AEGIS",  "key_var": "ALPACA_AEGIS_KEY",  "secret_var": "ALPACA_AEGIS_SECRET"},
    {"label": "HERMES", "key_var": "ALPACA_HERMES_KEY", "secret_var": "ALPACA_HERMES_SECRET"},
    {"label": "ATLAS",  "key_var": "ALPACA_ATLAS_KEY",  "secret_var": "ALPACA_ATLAS_SECRET"},
]

# Map symbol → strategy for per-strategy P&L attribution.
SYMBOL_STRATEGY = {
    "SVXY": "SVXY rotation",
    "BABA": "ADR arb", "JD": "ADR arb", "BIDU": "ADR arb",
    "NTES": "ADR arb", "TCOM": "ADR arb",
    "EWJ":  "Nikkei gap", "DXJ": "Nikkei gap",
    "SPY":  "HFT lead-lag",
}

STRATEGIES = [
    {"name": "SVXY rotation",  "account": "AEGIS",  "session": "svxy_sweet",         "desc": "VIX term-structure rotation"},
    {"name": "ADR arb",        "account": "AEGIS",  "session": "adr_alpaca",         "desc": "US-HK cross-listing pairs"},
    {"name": "Nikkei gap",     "account": "AEGIS",  "session": "nikkei_etf",         "desc": "Tokyo gap continuation"},
    {"name": "BTC funding",    "account": "KRAKEN", "session": "kraken_btc_funding", "desc": "Perp funding carry"},
    {"name": "HFT lead-lag",   "account": "HERMES", "session": "hft_engine",         "desc": "ES→SPY RLS (Rust, live paper)"},
    {"name": "Shadow Lab",     "account": "ATLAS",  "session": "atlas_shadow",       "desc": "Multi-horizon predictions, no orders"},
]


ATLAS_SHADOW_DB = ROOT / "research" / "live" / "atlas_shadow_rust.sqlite3"

# Hyperparameters per ensemble instance — must match config/ensemble/ens_N.yaml
# (10s cycle for 1-7, 5s for ens-8 — accelerated for max throughput)
FALCON_ENSEMBLE = [
    {"n": 1, "label": "ens-1  λ=0.9999  (slowest memory)", "lam": 0.9999, "warmup": 10, "cycle": 10},
    {"n": 2, "label": "ens-2  λ=0.999",                    "lam": 0.999,  "warmup": 10, "cycle": 10},
    {"n": 3, "label": "ens-3  λ=0.995",                    "lam": 0.995,  "warmup": 10, "cycle": 10},
    {"n": 4, "label": "ens-4  λ=0.99   (HERMES baseline)", "lam": 0.99,   "warmup": 10, "cycle": 10},
    {"n": 5, "label": "ens-5  λ=0.98",                     "lam": 0.98,   "warmup": 10, "cycle": 10},
    {"n": 6, "label": "ens-6  λ=0.95   (aggressive)",      "lam": 0.95,   "warmup": 10, "cycle": 10},
    {"n": 7, "label": "ens-7  λ=0.99   (fast warmup)",     "lam": 0.99,   "warmup": 5,  "cycle": 10},
    {"n": 8, "label": "ens-8  λ=0.99   (5s turbo)",        "lam": 0.99,   "warmup": 10, "cycle": 5},
    {"n": 9, "label": "ens-9  λ=0.99   🔁 INVERTED SIGNAL", "lam": 0.99,   "warmup": 10, "cycle": 10},
]


def _ensemble_one_stats(n: int) -> dict:
    """Read one ensemble DB and return its stats dict."""
    import sqlite3
    db = ROOT / "research" / "live" / f"falcon_ens_{n}.sqlite3"
    if not db.exists():
        return {"available": False}
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        n_total = con.execute("SELECT COUNT(*) FROM shadow_predictions").fetchone()[0]
        n_eval  = con.execute("SELECT COUNT(*) FROM shadow_predictions WHERE evaluated=1").fetchone()[0]
        n_fire  = con.execute("SELECT COUNT(*) FROM shadow_predictions WHERE would_fire=1").fetchone()[0]
        hit_row = con.execute("""
            SELECT
              COUNT(*) AS n,
              SUM(CASE WHEN predicted_bps * actual_bps_60s > 0 THEN 1 ELSE 0 END) AS hits,
              AVG(ABS(predicted_bps - actual_bps_60s)) AS mae_60s
            FROM shadow_predictions
            WHERE actual_bps_60s IS NOT NULL AND ABS(predicted_bps) > 0.1
        """).fetchone()
        n_h, n_hits, mae = (hit_row[0] or 0), (hit_row[1] or 0), (hit_row[2] or 0)
        hit_rate = (n_hits / n_h) if n_h > 0 else None
        latest = con.execute("""
            SELECT predicted_bps, confidence, samples, spy_mid
            FROM shadow_predictions ORDER BY id DESC LIMIT 1
        """).fetchone()
        con.close()
        return {
            "available":    True,
            "total":        n_total,
            "evaluated":    n_eval,
            "would_fire":   n_fire,
            "hit_rate_60s": hit_rate,
            "hit_n":        n_h,
            "mae_60s_bps":  mae or 0.0,
            "latest_pred":  latest[0] if latest else None,
            "latest_conf":  latest[1] if latest else None,
            "latest_samples": latest[2] if latest else None,
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def falcon_ensemble_stats() -> list[dict]:
    """Return per-instance stats for all 8 ensemble members."""
    out = []
    for spec in FALCON_ENSEMBLE:
        stats = _ensemble_one_stats(spec["n"])
        running = tmux_running(f"hft_ens_{spec['n']}")
        out.append({**spec, "running": running, **stats})
    return out


def atlas_shadow_stats() -> dict:
    """Pull stats from the ATLAS shadow predictions DB (Rust schema).

    Schema columns:
      ts, spy_mid, predicted_bps, confidence, latency_p99_ms, samples,
      would_fire, actual_spy_30s/60s/90s, actual_bps_30s/60s/90s, evaluated
    """
    import sqlite3
    if not ATLAS_SHADOW_DB.exists():
        return {"available": False}
    try:
        con = sqlite3.connect(f"file:{ATLAS_SHADOW_DB}?mode=ro", uri=True)
        n_total = con.execute("SELECT COUNT(*) FROM shadow_predictions").fetchone()[0]
        n_eval  = con.execute("SELECT COUNT(*) FROM shadow_predictions WHERE evaluated=1").fetchone()[0]
        n_fire  = con.execute("SELECT COUNT(*) FROM shadow_predictions WHERE would_fire=1").fetchone()[0]
        # Directional hit rate at 60s lookahead (where actual is filled)
        hit_row = con.execute("""
            SELECT
              COUNT(*) AS n,
              SUM(CASE WHEN predicted_bps * actual_bps_60s > 0 THEN 1 ELSE 0 END) AS hits,
              AVG(ABS(predicted_bps - actual_bps_60s)) AS mae_60s
            FROM shadow_predictions
            WHERE actual_bps_60s IS NOT NULL AND ABS(predicted_bps) > 0.1
        """).fetchone()
        n_h, n_hits, mae = (hit_row[0] or 0), (hit_row[1] or 0), (hit_row[2] or 0)
        hit_rate = (n_hits / n_h) if n_h > 0 else None
        # Latest sample for diagnostics
        latest = con.execute("""
            SELECT ts, predicted_bps, confidence, samples, spy_mid
            FROM shadow_predictions ORDER BY id DESC LIMIT 1
        """).fetchone()
        con.close()
        return {
            "available":      True,
            "total":          n_total,
            "evaluated":      n_eval,
            "would_fire":     n_fire,
            "hit_rate_60s":   hit_rate,
            "hit_n":          n_h,
            "mae_60s_bps":    mae or 0.0,
            "latest_pred":    latest[1] if latest else None,
            "latest_conf":    latest[2] if latest else None,
            "latest_samples": latest[3] if latest else None,
            "latest_spy":     latest[4] if latest else None,
            "engine":         "rust",
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


# --- helpers -----------------------------------------------------------

def load_env() -> dict[str, str]:
    env: dict[str, str] = {}
    if not ENV_FILE.exists():
        return env
    for line in ENV_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip().strip('"').strip("'")
    return env


ENV = load_env()


def alpaca_get(path: str, key: str, secret: str, **params) -> Any:
    try:
        r = requests.get(
            f"https://paper-api.alpaca.markets{path}",
            params=params,
            headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret},
            timeout=6,
        )
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def alpaca_get_cached(path: str, key: str, secret: str, ttl: float, **params) -> Any:
    """Cached variant. Cache key includes path + params."""
    ck = f"{path}|{secret[-6:]}|" + "|".join(f"{k}={v}" for k, v in sorted(params.items()))
    return CACHE.get(ck, ttl, lambda: alpaca_get(path, key, secret, **params))


def _tmux_running_raw(session: str) -> bool:
    import subprocess
    try:
        return subprocess.run(
            ["tmux", "has-session", "-t", session],
            capture_output=True, timeout=2,
        ).returncode == 0
    except Exception:
        return False


def _tmux_log_raw(session: str, lines: int) -> str:
    import subprocess
    try:
        out = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p"],
            capture_output=True, text=True, timeout=3,
        )
        if out.returncode != 0:
            return ""
        return "\n".join(out.stdout.splitlines()[-lines:])
    except Exception:
        return ""


def tmux_running(session: str) -> bool:
    return CACHE.get(f"tmux_run|{session}", 3.0, lambda: _tmux_running_raw(session))


def tmux_log(session: str, lines: int = 18) -> str:
    return CACHE.get(f"tmux_log|{session}|{lines}", 2.0, lambda: _tmux_log_raw(session, lines))


# ----------------------------- NIM live coach ---------------------------

NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
# Mistral Large 3 (675B) — top general-purpose reasoning model on our NIM tier.
# Picked over qwen3-coder-480b because qwen-coder is optimized for code generation,
# not financial/trading analysis. Mistral Large 3 is a generalist with strong
# analytical reasoning. Backup: meta/llama-3.3-70b-instruct, openai/gpt-oss-120b.
NIM_MODEL    = "mistralai/mistral-large-3-675b-instruct-2512"
NIM_SYS_PROMPT = (
    "You are the live trading coach for a paper-money multi-strategy portfolio "
    "(ADR stat-arb, BTC funding carry, SVXY VIX rotation, Nikkei gap, HFT lead-lag). "
    "This is Day 1 of a 30-day paper shadow. You read structured state and reply "
    "with EXACTLY 3 short lines, no preamble, no markdown: "
    "(1) HEALTH: one numeric phrase on portfolio behavior (mention which strategies are working/dragging). "
    "(2) FLAG: one specific quantitative anomaly worth investigating, else 'none'. "
    "(3) FOCUS: one concrete thing to watch in the next 60 minutes. "
    "Be terse and calibrated. Day-1 P&L is noise, distinguish that from real concerns."
)


def _nim_commentary_raw(snapshot: dict) -> dict:
    """Single NIM call summarizing current state into 3 short lines."""
    api_key = ENV.get("NVIDIA_API_KEY") or os.environ.get("NVIDIA_API_KEY")
    if not api_key:
        return {"available": False, "error": "no NVIDIA_API_KEY"}
    user_msg = (
        "CURRENT STATE (paper-mode):\n" + json.dumps(snapshot, default=str, indent=2)
        + "\n\nReply with exactly 3 lines as instructed."
    )
    try:
        r = requests.post(
            f"{NIM_BASE_URL}/chat/completions",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": NIM_MODEL,
                "messages": [
                    {"role": "system", "content": NIM_SYS_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                "max_tokens":  220,
                "temperature": 0.15,
                "top_p":       0.95,
            },
            timeout=30,
        )
        if r.status_code != 200:
            return {"available": False, "error": f"HTTP {r.status_code}: {r.text[:120]}"}
        data = r.json()
        text = data["choices"][0]["message"]["content"].strip()
        return {
            "available":  True,
            "model":      NIM_MODEL,
            "text":       text,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


_NIM_LATEST: dict[str, Any] = {"available": False, "text": "warming up…"}
_NIM_REFRESH_LOCK = threading.Lock()
_NIM_LAST_REFRESH = 0.0
_NIM_REFRESH_IN_FLIGHT = False


def _nim_refresh_worker(snapshot: dict) -> None:
    """Background thread: refresh the NIM commentary cache without blocking
    any request handler. Stale-while-revalidate."""
    global _NIM_LATEST, _NIM_LAST_REFRESH, _NIM_REFRESH_IN_FLIGHT
    try:
        result = _nim_commentary_raw(snapshot)
        with _NIM_REFRESH_LOCK:
            if result.get("available"):
                _NIM_LATEST = result
            else:
                # Preserve last successful, just update error metadata
                _NIM_LATEST = {**_NIM_LATEST, "last_error": result.get("error")}
            _NIM_LAST_REFRESH = _time.time()
    finally:
        with _NIM_REFRESH_LOCK:
            _NIM_REFRESH_IN_FLIGHT = False


def nim_commentary(snapshot: dict) -> dict:
    """Non-blocking NIM commentary. Returns the latest cached value immediately;
    if cache is stale (>60s old) and no refresh is in flight, spawns a
    background worker. The request handler NEVER waits on NIM."""
    global _NIM_REFRESH_IN_FLIGHT
    now = _time.time()
    stale = (now - _NIM_LAST_REFRESH) > 60.0
    with _NIM_REFRESH_LOCK:
        should_spawn = stale and not _NIM_REFRESH_IN_FLIGHT
        if should_spawn:
            _NIM_REFRESH_IN_FLIGHT = True
    if should_spawn:
        threading.Thread(target=_nim_refresh_worker, args=(snapshot,), daemon=True).start()
    return dict(_NIM_LATEST)


# ----------------------------- system health ----------------------------

# Process command-line patterns → display label. Matched on the full cmdline.
PROCESS_PATTERNS = [
    # More-specific patterns first; first match wins per process.
    ("dashboard",      "pro_live.py"),
    ("ADR",            "iter_18_adr"),
    ("SVXY",           "svxy_sweet"),
    ("Nikkei",         "iter_6_nikkei"),
    ("BTC funding",    "iter_3_kraken_btc"),
    ("FALCON ens",     "ensemble/ens_"),    # 8 FALCON ensemble instances
    ("Shadow Lab",     "--shadow-only"),    # ATLAS shadow (no ensemble path)
    ("HFT engine",     "target/release/hft-engine"),
    ("cloudflared",    "cloudflared tunnel"),
]

# Cache psutil.Process objects so cpu_percent() returns real values across calls
# (cpu_percent(interval=None) returns 0 on the first call per object).
_PROC_CACHE: dict[int, Any] = {}


def _system_health_raw() -> dict:
    try:
        import psutil
    except ImportError:
        return {"error": "psutil not installed"}

    # CPU — blocking 0.2s interval to capture a real delta sample
    # (interval=None returns 0 on the first call until psutil has two snapshots).
    cpu_per_core = psutil.cpu_percent(interval=0.2, percpu=True)
    cpu_total = (sum(cpu_per_core) / len(cpu_per_core)) if cpu_per_core else 0.0
    cpu_freq = None
    try:
        f = psutil.cpu_freq()
        if f:
            cpu_freq = {"current_mhz": f.current, "min_mhz": f.min, "max_mhz": f.max}
    except Exception:
        pass

    # Memory
    vm = psutil.virtual_memory()
    swap = psutil.swap_memory()

    # Load
    try:
        load1, load5, load15 = psutil.getloadavg()
    except Exception:
        load1 = load5 = load15 = 0.0

    # Battery
    battery = None
    try:
        b = psutil.sensors_battery()
        if b:
            secs = b.secsleft
            time_left = None
            if secs not in (psutil.POWER_TIME_UNLIMITED, psutil.POWER_TIME_UNKNOWN):
                time_left = int(secs)
            battery = {
                "percent":  b.percent,
                "plugged":  b.power_plugged,
                "secs_left": time_left,
            }
    except Exception:
        pass

    # Thermal (macOS pmset -g therm, no sudo needed)
    thermal = {"warning_level": None, "perf_level": None, "cpu_power_status": None}
    try:
        import subprocess
        out = subprocess.run(
            ["pmset", "-g", "therm"], capture_output=True, text=True, timeout=2,
        )
        if out.returncode == 0:
            for line in out.stdout.splitlines():
                low = line.lower()
                if "thermal warning level" in low:
                    thermal["warning_level"] = line.split("=")[-1].strip() if "=" in line else line.split(":")[-1].strip()
                elif "performance warning" in low:
                    thermal["perf_level"] = line.split("=")[-1].strip() if "=" in line else line.split(":")[-1].strip()
                elif "cpu power status" in low or "cpu_scheduler_limit" in low:
                    thermal["cpu_power_status"] = line.strip()
    except Exception:
        pass

    # Per-process metrics for known strategies. We cache the psutil.Process
    # object across calls so cpu_percent() returns a real delta (calling
    # cpu_percent on a fresh Process returns 0 the first time).
    matched: list[tuple[Any, str, int]] = []  # (process, label, pid)
    alive_pids: set[int] = set()
    for p in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = " ".join(p.info.get("cmdline") or [])
            if not cmd:
                continue
            for lbl, pat in PROCESS_PATTERNS:
                if pat in cmd:
                    pid = p.info["pid"]
                    alive_pids.add(pid)
                    cached = _PROC_CACHE.get(pid)
                    if cached is None:
                        _PROC_CACHE[pid] = p
                        # First-touch: prime cpu_percent so the next read returns delta
                        try:
                            p.cpu_percent(interval=None)
                        except Exception:
                            pass
                    matched.append((_PROC_CACHE[pid], lbl, pid))
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    # Garbage-collect dead PIDs from the cache
    for stale in [pid for pid in _PROC_CACHE if pid not in alive_pids]:
        _PROC_CACHE.pop(stale, None)

    processes = []
    for proc, lbl, pid in matched:
        try:
            cpu_pct = proc.cpu_percent(interval=None)
            mem = proc.memory_info().rss / (1024 * 1024)
            ct = proc.create_time()
            processes.append({
                "label":    lbl,
                "pid":      pid,
                "cpu":      round(cpu_pct, 1),
                "mem_mb":   round(mem, 1),
                "uptime_s": int(_time.time() - ct),
            })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    # Sort by label then pid for stable ordering
    label_order = {lbl: i for i, (lbl, _) in enumerate(PROCESS_PATTERNS)}
    processes.sort(key=lambda x: (label_order.get(x["label"], 99), x["pid"]))

    return {
        "cpu_total":     round(cpu_total, 1),
        "cpu_per_core":  [round(x, 1) for x in cpu_per_core],
        "cpu_freq":      cpu_freq,
        "n_cores":       psutil.cpu_count(logical=True),
        "memory": {
            "percent":   vm.percent,
            "used_gb":   round(vm.used / (1024 ** 3), 2),
            "total_gb":  round(vm.total / (1024 ** 3), 2),
            "free_gb":   round(vm.available / (1024 ** 3), 2),
        },
        "swap": {
            "percent":   swap.percent,
            "used_gb":   round(swap.used / (1024 ** 3), 2),
            "total_gb":  round(swap.total / (1024 ** 3), 2),
        },
        "load": {"1m": round(load1, 2), "5m": round(load5, 2), "15m": round(load15, 2)},
        "battery":   battery,
        "thermal":   thermal,
        "processes": processes,
    }


def system_health() -> dict:
    """Cached system health snapshot (2s TTL — frequent enough for the UI)."""
    return CACHE.get("system_health", 2.0, _system_health_raw)


# ----------------------------- realized P&L FIFO ------------------------

def compute_realized_per_order(fills: list[dict]) -> dict[str, float]:
    """FIFO realized P&L per order_id from Alpaca FILL activities.

    Alpaca's FILL activity has: side (buy/sell/sell_short), qty, price,
    symbol, order_id, transaction_time. Returns {order_id: realized_pnl_usd}.

    Walks fills oldest-first, maintains per-symbol FIFO inventory. Each fill
    that opposes the existing inventory's side credits realized P&L to that
    fill's order_id.
    """
    sorted_fills = sorted(fills, key=lambda f: f.get("transaction_time", ""))
    inventory: dict[str, deque] = defaultdict(deque)  # symbol -> [(qty, px, side)]
    realized_by_order: dict[str, float] = defaultdict(float)
    for f in sorted_fills:
        sym = f.get("symbol", "")
        side = (f.get("side") or "").lower()
        # sell_short is "open short", buy after a short is "close short"
        # Treat sell_short as side='sell' for inventory purposes.
        open_side = "buy" if side == "buy" else "sell"
        qty = float(f.get("qty") or 0)
        px = float(f.get("price") or 0)
        order_id = f.get("order_id") or ""
        if qty <= 0 or px <= 0:
            continue

        remaining = qty
        inv = inventory[sym]
        # If opposing side is already in inventory, close FIFO lots
        while remaining > 0 and inv and inv[0][2] != open_side:
            lot_qty, lot_px, lot_side = inv[0]
            close_qty = min(remaining, lot_qty)
            if lot_side == "buy":
                # closing a long: realized = (exit - entry) * qty
                pnl = (px - lot_px) * close_qty
            else:
                # closing a short: realized = (entry - exit) * qty
                pnl = (lot_px - px) * close_qty
            realized_by_order[order_id] += pnl
            new_lot_qty = lot_qty - close_qty
            inv.popleft()
            if new_lot_qty > 1e-9:
                inv.appendleft((new_lot_qty, lot_px, lot_side))
            remaining -= close_qty
        if remaining > 0:
            inv.append((remaining, px, open_side))
    return dict(realized_by_order)


def fetch_account_state(acc_label: str, key: str, secret: str) -> dict:
    # Hot data: refresh ~every 3s (positions, account, orders)
    acct = alpaca_get_cached("/v2/account", key, secret, 3.0) or {}
    positions = alpaca_get_cached("/v2/positions", key, secret, 3.0) or []
    orders = alpaca_get_cached("/v2/orders", key, secret, 3.0,
                                status="all", limit=25) or []
    # Cooler data: portfolio history at 5-min bars only needs to refresh every minute.
    history = alpaca_get_cached(
        "/v2/account/portfolio/history", key, secret, 60.0,
        period="1D", timeframe="5Min", extended_hours="true",
    ) or {}
    # FILL activities for realized P&L computation — fresh every 5s.
    fills = alpaca_get_cached(
        "/v2/account/activities", key, secret, 5.0,
        activity_types="FILL", page_size=100,
    ) or []
    realized_by_order = compute_realized_per_order(fills)

    eq_series = history.get("equity") or []
    ts_series = history.get("timestamp") or []
    # Drop leading None values (closed market hours)
    clean_eq = [x for x in eq_series if x is not None]
    last_equity = float(acct.get("equity", 0)) if acct else 0.0
    last_equity_yest = float(acct.get("last_equity", last_equity))
    today_pnl = last_equity - last_equity_yest
    today_pnl_pct = (today_pnl / last_equity_yest * 100) if last_equity_yest else 0.0

    unrealized_total = sum(
        float(p.get("unrealized_pl", 0)) for p in positions
    )
    unrealized_intraday = sum(
        float(p.get("unrealized_intraday_pl", 0)) for p in positions
    )

    return {
        "label":               acc_label,
        "equity":              last_equity,
        "cash":                float(acct.get("cash", 0)) if acct else 0.0,
        "buying_power":        float(acct.get("buying_power", 0)) if acct else 0.0,
        "last_equity_yest":    last_equity_yest,
        "today_pnl":           today_pnl,
        "today_pnl_pct":       today_pnl_pct,
        "unrealized_pnl":      unrealized_total,
        "unrealized_intraday": unrealized_intraday,
        "realized_today":      sum(realized_by_order.values()),
        "positions":           positions,
        "orders":              orders,
        "realized_by_order":   realized_by_order,
        "equity_history":      clean_eq[-72:] if clean_eq else [],
        "history_timestamps":  ts_series[-72:] if ts_series else [],
    }


KRAKEN_DB = ROOT / "research" / "live" / "kraken_btc_funding_decisions.sqlite3"


def _fetch_btc_spot_usd_uncached() -> float | None:
    """Public Kraken ticker (no auth) for live BTC mark price."""
    try:
        r = requests.get(
            "https://api.kraken.com/0/public/Ticker",
            params={"pair": "XXBTZUSD"},
            timeout=4,
        )
        if r.status_code != 200:
            return None
        d = r.json()
        result = (d.get("result") or {})
        # Result key is XXBTZUSD on Kraken
        for v in result.values():
            last = v.get("c", [None])[0]
            if last:
                return float(last)
    except Exception:
        pass
    # Fallback: yfinance via raw chart endpoint
    try:
        r = requests.get(
            "https://query1.finance.yahoo.com/v8/finance/chart/BTC-USD",
            params={"interval": "1m", "range": "1d"},
            headers={"User-Agent": "turbo-carnival/1.0"},
            timeout=4,
        )
        if r.status_code != 200:
            return None
        result = (r.json().get("chart") or {}).get("result") or []
        if not result:
            return None
        return float(result[0]["meta"]["regularMarketPrice"])
    except Exception:
        return None


def _fetch_btc_spot_usd() -> float | None:
    """Cached BTC spot — refreshes every 3s, smooths out the 1Hz frontend polling."""
    return CACHE.get("btc_spot_usd", 3.0, _fetch_btc_spot_usd_uncached)


def kraken_state() -> dict:
    """Kraken Futures Demo state via local SQLite + public BTC spot.

    The strategy host writes every open position to the local DB; we read
    that as the source of truth (the demo-futures REST endpoint returns 404
    for openpositions on this account). We compute live P&L by marking the
    SHORT perp positions against current public BTC spot.
    """
    import sqlite3
    out: dict[str, Any] = {"available": False}
    if not KRAKEN_DB.exists():
        out["error"] = "no kraken DB"
        return out
    today_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        con = sqlite3.connect(f"file:{KRAKEN_DB}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        cur = con.cursor()
        rows = cur.execute(
            "SELECT id, opened_ts_utc, qty_btc, perp_entry_px, "
            "funding_collected_usd FROM positions WHERE status='open'"
        ).fetchall()
        # Realized P&L from positions closed today (SHORT perp: entry > exit = profit)
        closed_today = cur.execute(
            "SELECT qty_btc, perp_entry_px, perp_exit_px, funding_collected_usd "
            "FROM positions WHERE status='closed' AND closed_ts_utc LIKE ?",
            (f"{today_utc}%",),
        ).fetchall()
        realized_today = 0.0
        closed_count = 0
        for r in closed_today:
            qty = float(r["qty_btc"] or 0)
            entry = float(r["perp_entry_px"] or 0)
            ex = r["perp_exit_px"]
            if ex is None:
                continue
            realized_today += (entry - float(ex)) * qty + float(r["funding_collected_usd"] or 0)
            closed_count += 1
        con.close()
    except Exception as exc:
        out["error"] = f"db error: {exc}"
        return out

    spot = _fetch_btc_spot_usd()
    total_qty = sum(float(r["qty_btc"] or 0) for r in rows)
    weighted_entry = (
        sum(float(r["qty_btc"] or 0) * float(r["perp_entry_px"] or 0) for r in rows) / total_qty
        if total_qty > 0 else 0.0
    )
    # Strategy is SHORT perp + virtual LONG spot (delta-neutral). The
    # NET BTC delta is ~zero, so daily mark-to-market should mostly be the
    # funding accumulator. But for the dashboard we display the SHORT perp
    # P&L (which the strategy hedges out via virtual long).
    perp_short_pnl = (weighted_entry - (spot or weighted_entry)) * total_qty if spot else 0.0
    funding_total = sum(float(r["funding_collected_usd"] or 0) for r in rows)

    positions = []
    for r in rows:
        entry = float(r["perp_entry_px"] or 0)
        qty = float(r["qty_btc"] or 0)
        leg_pnl = (entry - (spot or entry)) * qty if spot else 0.0
        positions.append({
            "symbol":   "BTC/USD:USD",
            "side":     "short",
            "contracts": qty,
            "entry":    entry,
            "mark":     spot,
            "unrealized_pnl": leg_pnl,
            "opened":   r["opened_ts_utc"],
            "id":       r["id"],
        })

    out.update({
        "available":        True,
        "spot_btc_usd":     spot,
        "equity":           0.0,
        "positions_count":  len(rows),
        "btc_pnl_usd":      perp_short_pnl,
        "funding_total":    funding_total,
        "realized_today":   realized_today,
        "closed_today_count": closed_count,
        "weighted_entry":   weighted_entry,
        "total_qty_btc":    total_qty,
        "positions":        positions,
    })
    return out


# --- API ---------------------------------------------------------------

app = FastAPI()


@app.get("/api/state")
def api_state() -> JSONResponse:
    accounts = []
    aggregate_equity = 0.0
    aggregate_today_pnl = 0.0
    aggregate_unrealized = 0.0

    for a in ALPACA_ACCOUNTS:
        key = ENV.get(a["key_var"], "")
        secret = ENV.get(a["secret_var"], "")
        if not key or not secret:
            continue
        st = fetch_account_state(a["label"], key, secret)
        accounts.append(st)
        aggregate_equity += st["equity"]
        aggregate_today_pnl += st["today_pnl"]
        aggregate_unrealized += st["unrealized_pnl"]

    kraken = kraken_state()
    if kraken.get("available"):
        aggregate_equity += float(kraken.get("equity", 0))
        aggregate_unrealized += float(kraken.get("btc_pnl_usd", 0))

    # Strategy attribution from Alpaca positions
    strat_pnl: dict[str, dict[str, float]] = defaultdict(
        lambda: {"unrealized": 0.0, "market_value": 0.0, "positions": 0}
    )
    for acc in accounts:
        for p in acc.get("positions", []):
            sym = p.get("symbol", "")
            strat_name = SYMBOL_STRATEGY.get(sym, "Unattributed")
            strat_pnl[strat_name]["unrealized"] += float(p.get("unrealized_pl", 0))
            strat_pnl[strat_name]["market_value"] += abs(float(p.get("market_value", 0)))
            strat_pnl[strat_name]["positions"] += 1
    # Kraken BTC
    if kraken.get("available"):
        strat_pnl["BTC funding"]["unrealized"] += float(kraken.get("btc_pnl_usd", 0))
        strat_pnl["BTC funding"]["positions"] = kraken.get("positions_count", 0)

    shadow_stats = atlas_shadow_stats()

    # Merge with strategies definition for full coverage
    strategy_cards = []
    for s in STRATEGIES:
        name = s["name"]
        running = tmux_running(s["session"])
        log = tmux_log(s["session"], lines=12)
        last_log = (log.splitlines()[-1] if log else "")[:180]
        # For Shadow Lab, fold in shadow stats and synthesize a status line
        if name == "Shadow Lab" and shadow_stats.get("available"):
            sh = shadow_stats
            hr = sh.get("hit_rate_60s")
            hit_txt = f"hit_rate={hr*100:.1f}% (n={sh.get('hit_n',0)})" if hr is not None else "warming up"
            shadow_line = (
                f"preds={sh.get('total',0)} eval={sh.get('evaluated',0)} "
                f"hermes_fire={sh.get('would_fire',0)} {hit_txt} "
                f"mae60={sh.get('mae_60s_bps',0):.2f}bps "
                f"latest_pred={sh.get('latest_pred') or 0:+.3f}bps "
                f"samples={sh.get('latest_samples',0)}"
            )
            last_log = shadow_line
        strategy_cards.append({
            **s,
            "running": running,
            "unrealized_pnl": strat_pnl[name]["unrealized"],
            "exposure": strat_pnl[name]["market_value"],
            "positions": strat_pnl[name]["positions"],
            "last_log": last_log,
        })

    # Flatten positions across accounts with strategy tag
    all_positions = []
    for acc in accounts:
        for p in acc.get("positions", []):
            sym = p.get("symbol", "")
            all_positions.append({
                "account":      acc["label"],
                "strategy":     SYMBOL_STRATEGY.get(sym, "—"),
                "symbol":       sym,
                "side":         p.get("side"),
                "qty":          p.get("qty"),
                "avg_entry":    float(p.get("avg_entry_price", 0)),
                "current":      float(p.get("current_price", 0) or 0),
                "market_value": float(p.get("market_value", 0)),
                "cost_basis":   float(p.get("cost_basis", 0)),
                "unrealized_pl":     float(p.get("unrealized_pl", 0)),
                "unrealized_plpc":   float(p.get("unrealized_plpc", 0)) * 100,
            })
    if kraken.get("available"):
        for p in kraken.get("positions", []):
            all_positions.append({
                "account":  "KRAKEN",
                "strategy": "BTC funding",
                "symbol":   p["symbol"],
                "side":     p["side"],
                "qty":      p["contracts"],
                "avg_entry": float(p["entry"] or 0),
                "current":   float(p["mark"] or 0),
                "market_value": float(p["contracts"] or 0) * float(p["mark"] or 0),
                "cost_basis":   float(p["contracts"] or 0) * float(p["entry"] or 0),
                "unrealized_pl":   float(p["unrealized_pnl"] or 0),
                "unrealized_plpc": 0.0,
            })

    # Merge orders across accounts, sort by submitted_at desc, take 25.
    # Attach realized P&L (computed from FILL activities FIFO) per order.
    merged_orders = []
    for acc in accounts:
        realized_map = acc.get("realized_by_order") or {}
        for o in acc.get("orders", []):
            oid = o.get("id", "")
            realized = realized_map.get(oid, 0.0)
            merged_orders.append({
                "account":     acc["label"],
                "submitted":   o.get("submitted_at", "")[:19],
                "side":        o.get("side"),
                "qty":         o.get("qty"),
                "filled":      o.get("filled_qty", "0"),
                "symbol":      o.get("symbol"),
                "type":        o.get("order_type"),
                "status":      o.get("status"),
                "filled_avg":  o.get("filled_avg_price"),
                "strategy":    SYMBOL_STRATEGY.get(o.get("symbol", ""), "—"),
                "realized_pl": realized if abs(realized) > 0.001 else None,
            })
    merged_orders.sort(key=lambda x: x["submitted"], reverse=True)
    merged_orders = merged_orders[:25]

    # Per-account stat blocks for the top bar — Kraken is treated as its own
    # account/model alongside AEGIS and HERMES. They are independent strategies
    # and should NOT be aggregated.
    account_stats = []
    for a in accounts:
        account_stats.append({
            "label":         a["label"],
            "equity":        a["equity"],
            "today_pnl":     a["today_pnl"],
            "today_pnl_pct": a["today_pnl_pct"],
            "unrealized":    a["unrealized_pnl"],
            "realized_today": a.get("realized_today", 0.0),
            "positions":     sum(1 for p in all_positions if p["account"] == a["label"]),
        })
    if kraken.get("available"):
        krk_realized = float(kraken.get("realized_today") or 0)
        krk_unr      = float(kraken.get("btc_pnl_usd") or 0)
        krk_open     = kraken.get("positions_count") or 0
        krk_closed   = kraken.get("closed_today_count") or 0
        account_stats.append({
            "label":            "KRAKEN",
            # Kraken Futures Demo doesn't expose balance via signed REST — show
            # cumulative realized P&L today as the primary number instead of $0.
            "equity":           None,                     # signal "unknown" to UI
            "today_pnl":        krk_realized + krk_unr,
            "today_pnl_pct":    0.0,
            "unrealized":       krk_unr,
            "realized_today":   krk_realized,
            "positions":        krk_open,
            "closed_today":     krk_closed,
            "spot_btc_usd":     float(kraken.get("spot_btc_usd") or 0),
            "weighted_entry":   float(kraken.get("weighted_entry") or 0),
            "is_demo_no_nav":   True,
        })

    return JSONResponse({
        "ts":                  datetime.now(timezone.utc).isoformat(),
        "host":                system_health(),
        "coach":               {"available": False, "disabled": True},
        "shadow":              shadow_stats,
        "ensemble":            falcon_ensemble_stats(),
        "account_stats":       account_stats,
        # Aggregate fields kept for backwards-compat / portfolio view, but
        # the UI no longer leads with them.
        "aggregate_equity":    aggregate_equity,
        "aggregate_today_pnl": aggregate_today_pnl,
        "aggregate_unrealized": aggregate_unrealized,
        "accounts":            [
            {k: v for k, v in a.items()
             if k not in ("positions", "orders", "realized_by_order")}
            for a in accounts
        ],
        "kraken":              kraken,
        "strategies":          strategy_cards,
        "positions":           all_positions,
        "orders":              merged_orders,
    })


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return HTML


HTML = r"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>turbo-carnival · pro</title>
<script src="https://unpkg.com/lightweight-charts@5.0.8/dist/lightweight-charts.standalone.production.js"></script>
<style>
  :root {
    --bg:        #131722;
    --bg-2:      #1e222d;
    --bg-3:      #2a2e39;
    --border:    #363c4e;
    --text:      #d1d4dc;
    --muted:     #787b86;
    --green:     #26a69a;
    --red:       #ef5350;
    --blue:      #2962ff;
    --yellow:    #f7b500;
    color-scheme: dark;
  }
  * { box-sizing: border-box; }
  body {
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", sans-serif;
    margin: 0; padding: 0; font-size: 13px; line-height: 1.4;
  }
  .topbar {
    display: flex; align-items: stretch; gap: 0;
    padding: 0; background: var(--bg-2);
    border-bottom: 1px solid var(--border);
  }
  .brand-cell {
    padding: 14px 16px; border-right: 1px solid var(--border);
    display: flex; flex-direction: column; justify-content: center;
    min-width: 180px;
  }
  .brand { font-weight: 700; font-size: 15px; letter-spacing: -0.2px; }
  .brand .pulse {
    display: inline-block; width: 7px; height: 7px;
    background: var(--green); border-radius: 50%;
    margin-right: 8px; box-shadow: 0 0 6px var(--green);
    animation: pulse 2s infinite;
  }
  .ts { font-size: 10px; color: var(--muted); font-variant-numeric: tabular-nums; margin-top: 4px; }
  @keyframes pulse { 50% { opacity: 0.3; } }
  .acct-cell {
    flex: 1; padding: 10px 14px;
    border-right: 1px solid var(--border);
    display: flex; flex-direction: column; gap: 2px;
  }
  .acct-cell:last-child { border-right: none; }
  .acct-cell .label {
    font-size: 10px; text-transform: uppercase; letter-spacing: 1px;
    color: var(--muted); margin-bottom: 1px;
  }
  .acct-cell .equity {
    font-size: 17px; font-weight: 600;
    font-variant-numeric: tabular-nums;
    transition: color 0.4s ease;
  }
  .acct-cell .breakdown {
    font-size: 11px; color: var(--muted);
    font-variant-numeric: tabular-nums;
    display: flex; gap: 10px; align-items: center;
  }
  .acct-cell .breakdown .pnl {
    transition: color 0.4s ease;
  }
  @media (max-width: 1100px) {
    .topbar { flex-wrap: wrap; }
    .acct-cell { flex: 1 1 45%; }
  }

  .grid {
    display: grid; grid-template-columns: 1fr 1.4fr 1fr;
    gap: 12px; padding: 12px;
  }
  @media (max-width: 1100px) {
    .grid { grid-template-columns: 1fr; }
    .stat-row { flex-wrap: wrap; gap: 12px; }
  }

  .panel {
    background: var(--bg-2); border: 1px solid var(--border);
    border-radius: 4px; padding: 10px 12px;
  }
  .panel h3 {
    margin: 0 0 8px; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px; color: var(--muted);
  }

  /* strategy cards */
  .strat-card {
    background: var(--bg-3); border: 1px solid var(--border);
    border-radius: 3px; padding: 8px 10px; margin-bottom: 6px;
  }
  .strat-card .row1 { display: flex; justify-content: space-between; align-items: center; }
  .strat-card .name { font-weight: 600; font-size: 13px; }
  .strat-card .meta { font-size: 11px; color: var(--muted); margin-top: 2px; }
  .strat-card .pnl { font-variant-numeric: tabular-nums; font-weight: 600; }
  .strat-card .last-log {
    margin-top: 6px; padding: 5px 7px; background: var(--bg);
    font-family: ui-monospace, "SF Mono", "Menlo", monospace;
    font-size: 10px; color: var(--muted); border-radius: 2px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
  }
  .dot { display: inline-block; width: 6px; height: 6px; border-radius: 50%; margin-right: 5px; }
  .dot.green { background: var(--green); }
  .dot.red   { background: var(--red); }

  /* positions table */
  table { width: 100%; border-collapse: collapse; font-size: 12px; }
  th {
    text-align: left; padding: 6px 8px;
    color: var(--muted); font-weight: 500; font-size: 10px;
    text-transform: uppercase; letter-spacing: 0.5px;
    border-bottom: 1px solid var(--border);
  }
  td {
    padding: 6px 8px; border-bottom: 1px solid var(--border);
    font-variant-numeric: tabular-nums;
  }
  td.num { text-align: right; font-family: ui-monospace, "SF Mono", "Menlo", monospace; transition: color 0.4s ease; }
  td { transition: background 0.6s ease; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--bg-3); }
  .pnl, .equity { transition: color 0.4s ease; }

  .pos { color: var(--green); }
  .neg { color: var(--red); }
  .mut { color: var(--muted); }

  .side-buy  { color: var(--green); }
  .side-sell { color: var(--red); }
  .side-long  { color: var(--green); }
  .side-short { color: var(--red); }

  .badge {
    display: inline-block; padding: 1px 6px; border-radius: 2px;
    font-size: 10px; font-weight: 500;
    background: var(--bg); border: 1px solid var(--border);
    color: var(--muted);
  }
  .badge.aegis  { color: #79c0ff; border-color: #1f6feb; }
  .badge.hermes { color: #f0883e; border-color: #d2691e; }
  .badge.kraken { color: #c084fc; border-color: #7c3aed; }
  .badge.filled   { color: var(--green);  border-color: var(--green); }
  .badge.canceled { color: var(--red);    border-color: var(--red); }
  .badge.new, .badge.accepted, .badge.pending_new { color: var(--yellow); border-color: var(--yellow); }
  .badge.partially_filled { color: var(--yellow); border-color: var(--yellow); }

  .sparkline { width: 100%; height: 36px; margin-top: 6px; }

  .empty { color: var(--muted); padding: 14px; text-align: center; font-size: 11px; }

  .activity-row {
    display: grid; grid-template-columns: auto 1fr auto;
    gap: 8px; padding: 5px 0;
    border-bottom: 1px solid var(--border); font-size: 11px;
    font-variant-numeric: tabular-nums;
  }
  .activity-row:last-child { border-bottom: none; }
  .activity-row .t { color: var(--muted); font-family: ui-monospace, monospace; }
  .pl-chip {
    display: inline-block; margin-left: 6px; padding: 1px 6px;
    border-radius: 3px; font-size: 10px; font-weight: 600;
    font-variant-numeric: tabular-nums;
  }
  .pl-chip.pos { background: rgba(38,166,154,0.15); color: var(--green); }
  .pl-chip.neg { background: rgba(239,83,80,0.18); color: var(--red); }

  .footer { color: var(--muted); font-size: 10px; padding: 10px 16px; text-align: center; }
  a { color: var(--blue); text-decoration: none; }

  /* NIM AI coach panel */
  .coach-row { padding: 0 12px 12px 12px; }
  .coach-panel {
    background: linear-gradient(135deg, #1a1f2e 0%, #1e222d 100%);
    border: 1px solid #2962ff44; border-radius: 4px; padding: 10px 14px;
    position: relative;
  }
  .coach-panel h3 {
    margin: 0 0 6px; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px;
    color: var(--blue); display: flex; align-items: center; gap: 8px;
  }
  .coach-panel h3 .nim-dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--blue);
    box-shadow: 0 0 6px var(--blue); animation: pulse 2s infinite;
  }
  .coach-body {
    font-family: -apple-system, "Inter", sans-serif; font-size: 13px;
    line-height: 1.55; color: var(--text); white-space: pre-wrap;
    transition: opacity 0.3s ease;
  }
  .coach-meta {
    margin-top: 6px; font-size: 10px; color: var(--muted);
    font-variant-numeric: tabular-nums;
  }

  .chart-wrap { padding: 0 12px 12px 12px; }
  .chart-panel { background: var(--bg-2); border: 1px solid var(--border); border-radius: 4px; padding: 10px 12px; }
  .chart-panel h3 { margin: 0 0 6px; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px; color: var(--muted); }
  #equity-chart { width: 100%; height: 240px; }

  /* host/system health panel */
  .host-row { padding: 0 12px 12px 12px; }
  .host-panel {
    background: var(--bg-2); border: 1px solid var(--border); border-radius: 4px;
    padding: 10px 12px;
  }
  .host-panel h3 { margin: 0 0 8px; font-size: 11px; font-weight: 600;
    text-transform: uppercase; letter-spacing: 1px; color: var(--muted); }
  .host-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 10px;
  }
  .metric-card {
    background: var(--bg-3); border: 1px solid var(--border);
    border-radius: 3px; padding: 8px 10px;
  }
  .metric-card .mt-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.8px;
    color: var(--muted); margin-bottom: 2px;
  }
  .metric-card .mt-value {
    font-size: 18px; font-weight: 600; font-variant-numeric: tabular-nums;
    transition: color 0.4s ease;
  }
  .metric-card .mt-sub {
    font-size: 11px; color: var(--muted); margin-top: 2px; font-variant-numeric: tabular-nums;
  }
  .bar-track {
    height: 5px; background: var(--bg); border-radius: 2px; margin-top: 6px;
    overflow: hidden;
  }
  .bar-fill {
    height: 100%; background: var(--green); transition: width 0.4s ease, background 0.4s ease;
  }
  .bar-fill.warn   { background: var(--yellow); }
  .bar-fill.hot    { background: var(--red); }
  .core-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(18px, 1fr));
    gap: 3px; margin-top: 5px;
  }
  .core-cell {
    height: 26px; background: var(--bg);
    border-radius: 2px; position: relative; overflow: hidden;
  }
  .core-cell .core-fill {
    position: absolute; bottom: 0; left: 0; right: 0;
    background: var(--green); transition: height 0.4s ease, background 0.4s ease;
  }
  .core-cell .core-fill.warn { background: var(--yellow); }
  .core-cell .core-fill.hot  { background: var(--red); }
  .core-cell .core-num {
    position: absolute; bottom: 1px; right: 2px;
    font-size: 8px; color: var(--text); font-variant-numeric: tabular-nums;
    text-shadow: 0 0 2px var(--bg);
  }
</style>
</head>
<body>

<div class="topbar">
  <div class="brand-cell">
    <div class="brand"><span class="pulse"></span>turbo-carnival · pro</div>
    <div class="ts" id="ts">connecting…</div>
  </div>
  <div id="account-cells" style="display:flex;flex:1;align-items:stretch"></div>
</div>

<div class="grid">

  <div>
    <div class="panel">
      <h3>Accounts</h3>
      <div id="accounts"></div>
    </div>
    <div class="panel" style="margin-top:12px">
      <h3>Strategies</h3>
      <div id="strategies"></div>
    </div>
  </div>

  <div class="panel">
    <h3>Open Positions</h3>
    <div id="positions-wrap">
      <table id="positions">
        <thead>
          <tr>
            <th>Acct</th><th>Strategy</th><th>Symbol</th><th>Side</th>
            <th class="num">Qty</th><th class="num">Entry</th><th class="num">Mark</th>
            <th class="num">Mkt Val</th><th class="num">P&amp;L $</th><th class="num">P&amp;L %</th>
          </tr>
        </thead>
        <tbody></tbody>
      </table>
    </div>
  </div>

  <div class="panel">
    <h3>Activity</h3>
    <div id="activity"></div>
  </div>

</div>

<div class="host-row">
  <div class="host-panel">
    <h3>FALCON · 8-instance Compute-Driven Ensemble</h3>
    <table id="ensemble-tbl" style="margin-top:4px">
      <thead><tr>
        <th>Instance</th>
        <th class="num">λ</th>
        <th class="num">warmup</th>
        <th class="num">cycle</th>
        <th>up</th>
        <th class="num">preds</th>
        <th class="num">eval</th>
        <th class="num">fire</th>
        <th class="num">hit% 60s</th>
        <th class="num">MAE bps</th>
        <th class="num">latest pred</th>
        <th class="num">samples</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<div class="host-row">
  <div class="host-panel">
    <h3>Host system · Mac</h3>
    <div class="host-grid">
      <div class="metric-card">
        <div class="mt-label">CPU</div>
        <div class="mt-value" id="host-cpu">—</div>
        <div class="mt-sub" id="host-cpu-sub">—</div>
        <div class="core-grid" id="host-cores"></div>
      </div>
      <div class="metric-card">
        <div class="mt-label">Memory</div>
        <div class="mt-value" id="host-mem">—</div>
        <div class="mt-sub" id="host-mem-sub">—</div>
        <div class="bar-track"><div class="bar-fill" id="host-mem-bar" style="width:0%"></div></div>
      </div>
      <div class="metric-card">
        <div class="mt-label">Load</div>
        <div class="mt-value" id="host-load">—</div>
        <div class="mt-sub" id="host-load-sub">—</div>
      </div>
      <div class="metric-card" id="host-battery-card" style="display:none">
        <div class="mt-label">Battery</div>
        <div class="mt-value" id="host-battery">—</div>
        <div class="mt-sub" id="host-battery-sub">—</div>
        <div class="bar-track"><div class="bar-fill" id="host-battery-bar" style="width:0%"></div></div>
      </div>
      <div class="metric-card">
        <div class="mt-label">Thermal</div>
        <div class="mt-value" id="host-thermal">—</div>
        <div class="mt-sub" id="host-thermal-sub">—</div>
      </div>
    </div>
    <h3 style="margin-top:14px">Processes</h3>
    <table id="host-procs" style="margin-top:4px">
      <thead><tr>
        <th>Service</th><th>PID</th>
        <th class="num">CPU%</th><th class="num">RAM (MB)</th><th class="num">Uptime</th>
      </tr></thead>
      <tbody></tbody>
    </table>
  </div>
</div>

<div class="chart-wrap">
  <div class="chart-panel">
    <h3>Portfolio Equity · last 24h · 5-min</h3>
    <div id="equity-chart"></div>
  </div>
</div>

<div class="footer">
  auto-refresh 5s · paper-mode only · all prices via Alpaca + Kraken Demo
</div>

<script>
const fmtUSD = v => {
  const sign = v < 0 ? '-' : '';
  return sign + '$' + Math.abs(v).toLocaleString(undefined, {minimumFractionDigits:2, maximumFractionDigits:2});
};
const fmtPct = v => (v >= 0 ? '+' : '') + v.toFixed(2) + '%';
const cls    = v => v > 0 ? 'pos' : (v < 0 ? 'neg' : 'mut');

// ---- diff-based DOM helpers (no innerHTML rewrites in steady state) ----
function setText(el, text) {
  if (!el) return;
  if (el.textContent !== text) el.textContent = text;
}
function setClass(el, klass) {
  if (!el) return;
  if (el.className !== klass) el.className = klass;
}
// Track DOM nodes by stable key so rows update in-place
const _domCache = {
  accountCells: new Map(),
  strategyCards: new Map(),
  positionRows: new Map(),
  activityRows: new Map(),
};
function ensureNode(parent, key, cache, builder) {
  let node = cache.get(key);
  if (node && node.isConnected && node.parentElement === parent) return node;
  node = builder();
  cache.set(key, node);
  parent.appendChild(node);
  return node;
}
function reorderChildren(parent, orderedNodes) {
  // Reorder without removing — preserves focus/transitions on unchanged rows
  for (let i = 0; i < orderedNodes.length; i++) {
    const node = orderedNodes[i];
    if (parent.children[i] !== node) parent.insertBefore(node, parent.children[i] || null);
  }
  // Remove anything beyond the new length
  while (parent.children.length > orderedNodes.length) {
    parent.lastElementChild.remove();
  }
}

function sparkline(values, color) {
  if (!values || values.length < 2) return '';
  const lo = Math.min(...values), hi = Math.max(...values);
  const range = (hi - lo) || 1;
  const w = 100, h = 30;
  const pts = values.map((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = h - ((v - lo) / range) * h;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  const startVal = values[0], endVal = values[values.length-1];
  const sparkColor = endVal >= startVal ? 'var(--green)' : 'var(--red)';
  return `<svg class="sparkline" viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">
    <polyline points="${pts}" fill="none" stroke="${color || sparkColor}" stroke-width="1.5" />
  </svg>`;
}

// ---- builders (run once per row; mutated in-place after) ----
function buildAccountCell() {
  const c = document.createElement('div'); c.className = 'acct-cell';
  c.innerHTML = `
    <div class="label" data-f="label">—</div>
    <div class="equity" data-f="equity">—</div>
    <div class="breakdown">
      <span class="pnl" data-f="todayPnl">—</span>
      <span class="mut" data-f="todayPct">—</span>
      <span class="mut" data-f="positions">—</span>
    </div>
  `;
  return c;
}

function buildStratCard() {
  const c = document.createElement('div'); c.className = 'strat-card';
  c.innerHTML = `
    <div class="row1">
      <span class="name" data-f="name">—</span>
      <span class="pnl" data-f="pnl">—</span>
    </div>
    <div class="meta" data-f="meta">—</div>
    <div class="last-log" data-f="log">—</div>
  `;
  return c;
}

function buildPositionRow() {
  const r = document.createElement('tr');
  r.innerHTML = `
    <td><span class="badge" data-f="acct">—</span></td>
    <td class="mut" data-f="strategy">—</td>
    <td><b data-f="symbol">—</b></td>
    <td data-f="side">—</td>
    <td class="num" data-f="qty">—</td>
    <td class="num" data-f="entry">—</td>
    <td class="num" data-f="mark">—</td>
    <td class="num" data-f="mktVal">—</td>
    <td class="num" data-f="pl">—</td>
    <td class="num" data-f="plPct">—</td>
  `;
  return r;
}

function buildActivityRow() {
  const r = document.createElement('div'); r.className = 'activity-row';
  r.innerHTML = `
    <div class="t" data-f="time">—</div>
    <div data-f="body">—</div>
    <div data-f="status">—</div>
  `;
  return r;
}

function field(node, name) { return node.querySelector(`[data-f="${name}"]`); }

function classForPct(pct, warn, hot) {
  if (pct >= hot)  return 'hot';
  if (pct >= warn) return 'warn';
  return '';
}

function fmtUptime(secs) {
  if (!secs || secs < 0) return '—';
  const h = Math.floor(secs / 3600);
  const m = Math.floor((secs % 3600) / 60);
  const s = secs % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

function renderEnsemble(rows) {
  const tbody = document.querySelector('#ensemble-tbl tbody');
  if (!tbody) return;
  if (!rows || rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="12" class="empty">no ensemble data</td></tr>';
    return;
  }
  _domCache.ensembleRows = _domCache.ensembleRows || new Map();
  const nodes = [];
  rows.forEach(r => {
    const key = `ens-${r.n}`;
    let node = _domCache.ensembleRows.get(key);
    if (!node || !node.isConnected || node.parentElement !== tbody) {
      node = document.createElement('tr');
      node.innerHTML = `
        <td><b data-f="label">—</b></td>
        <td class="num" data-f="lam">—</td>
        <td class="num mut" data-f="warmup">—</td>
        <td class="num mut" data-f="cycle">—</td>
        <td data-f="up">—</td>
        <td class="num" data-f="preds">—</td>
        <td class="num mut" data-f="eval">—</td>
        <td class="num" data-f="fire">—</td>
        <td class="num" data-f="hit">—</td>
        <td class="num" data-f="mae">—</td>
        <td class="num" data-f="pred">—</td>
        <td class="num mut" data-f="samples">—</td>
      `;
      _domCache.ensembleRows.set(key, node);
      tbody.appendChild(node);
    }
    nodes.push(node);
    setText(field(node, 'label'), r.label || `ens-${r.n}`);
    setText(field(node, 'lam'), String(r.lam));
    setText(field(node, 'warmup'), String(r.warmup));
    setText(field(node, 'cycle'), r.cycle + 's');
    const upEl = field(node, 'up');
    setText(upEl, r.running ? '🟢' : '🔴');
    setText(field(node, 'preds'), String(r.total || 0));
    setText(field(node, 'eval'), String(r.evaluated || 0));
    setText(field(node, 'fire'), String(r.would_fire || 0));
    const hitEl = field(node, 'hit');
    if (r.hit_rate_60s != null) {
      setText(hitEl, (r.hit_rate_60s * 100).toFixed(1) + '% (n=' + r.hit_n + ')');
      setClass(hitEl, 'num ' + (r.hit_rate_60s > 0.55 ? 'pos' : r.hit_rate_60s < 0.45 ? 'neg' : 'mut'));
    } else {
      setText(hitEl, 'warming');
      setClass(hitEl, 'num mut');
    }
    setText(field(node, 'mae'), (r.mae_60s_bps || 0).toFixed(2));
    const predEl = field(node, 'pred');
    const lp = r.latest_pred || 0;
    setText(predEl, (lp >= 0 ? '+' : '') + lp.toFixed(3));
    setClass(predEl, 'num ' + cls(lp));
    setText(field(node, 'samples'), String(r.latest_samples || 0));
  });
  reorderChildren(tbody, nodes);
}

function renderHost(h) {
  if (!h || h.error) {
    setText(document.getElementById('host-cpu'), h?.error || '—');
    return;
  }
  // CPU summary
  setText(document.getElementById('host-cpu'), h.cpu_total.toFixed(1) + '%');
  document.getElementById('host-cpu').className =
    'mt-value ' + (h.cpu_total >= 85 ? 'neg' : h.cpu_total >= 60 ? 'mut' : 'pos');
  setText(document.getElementById('host-cpu-sub'),
    `${h.n_cores} cores` + (h.cpu_freq ? ` · ${Math.round(h.cpu_freq.current_mhz)} MHz` : ''));

  // Per-core grid
  const cg = document.getElementById('host-cores');
  // Build cells on first render; just update fills after
  if (cg.children.length !== (h.cpu_per_core || []).length) {
    cg.innerHTML = '';
    (h.cpu_per_core || []).forEach((v, i) => {
      const c = document.createElement('div'); c.className = 'core-cell';
      c.innerHTML = `<div class="core-fill"></div><div class="core-num">${i}</div>`;
      cg.appendChild(c);
    });
  }
  Array.from(cg.children).forEach((cell, i) => {
    const v = (h.cpu_per_core || [])[i] || 0;
    const f = cell.querySelector('.core-fill');
    f.style.height = v + '%';
    setClass(f, 'core-fill ' + classForPct(v, 60, 85));
  });

  // Memory
  setText(document.getElementById('host-mem'),
    h.memory.used_gb.toFixed(1) + ' / ' + h.memory.total_gb.toFixed(1) + ' GB');
  setText(document.getElementById('host-mem-sub'),
    h.memory.percent + '% used · ' + h.memory.free_gb.toFixed(1) + ' GB free' +
    (h.swap.used_gb > 0.05 ? ` · swap ${h.swap.used_gb.toFixed(1)} GB` : ''));
  const mbar = document.getElementById('host-mem-bar');
  mbar.style.width = h.memory.percent + '%';
  setClass(mbar, 'bar-fill ' + classForPct(h.memory.percent, 70, 90));

  // Load
  const ncores = h.n_cores || 1;
  const lpct = (h.load['1m'] / ncores) * 100;
  setText(document.getElementById('host-load'),
    `${h.load['1m'].toFixed(2)}  ·  ${h.load['5m'].toFixed(2)}  ·  ${h.load['15m'].toFixed(2)}`);
  document.getElementById('host-load').className =
    'mt-value ' + (lpct >= 100 ? 'neg' : lpct >= 70 ? 'mut' : 'pos');
  setText(document.getElementById('host-load-sub'),
    `1m / 5m / 15m  ·  ${Math.round(lpct)}% of ${ncores}-core cap`);

  // Battery (hidden if no battery)
  const batCard = document.getElementById('host-battery-card');
  if (h.battery) {
    batCard.style.display = '';
    const b = h.battery;
    setText(document.getElementById('host-battery'),
      b.percent.toFixed(0) + '%' + (b.plugged ? ' ⚡' : ''));
    let sub = b.plugged ? 'charging' : 'on battery';
    if (b.secs_left) {
      const h_ = Math.floor(b.secs_left/3600), m_ = Math.floor((b.secs_left%3600)/60);
      sub += ` · ${h_}h ${m_}m left`;
    }
    setText(document.getElementById('host-battery-sub'), sub);
    const bb = document.getElementById('host-battery-bar');
    bb.style.width = b.percent + '%';
    setClass(bb, 'bar-fill ' + (b.percent < 20 ? 'hot' : b.percent < 40 ? 'warn' : ''));
  } else {
    batCard.style.display = 'none';
  }

  // Thermal
  const t = h.thermal || {};
  const therm = t.warning_level || t.perf_level || 'nominal';
  setText(document.getElementById('host-thermal'),
    /no thermal warning/i.test(therm) ? 'nominal' : therm);
  document.getElementById('host-thermal').className =
    'mt-value ' + (/no thermal warning|nominal/i.test(therm) ? 'pos' : 'neg');
  setText(document.getElementById('host-thermal-sub'),
    t.cpu_power_status ? t.cpu_power_status.slice(0, 60) : 'CPU power: full');

  // Processes table — diff-based
  const tbody = document.querySelector('#host-procs tbody');
  const procs = h.processes || [];
  if (procs.length === 0) {
    tbody.innerHTML = '<tr><td colspan="5" class="empty">no tracked processes</td></tr>';
    _domCache.hostProcs = new Map();
  } else {
    _domCache.hostProcs = _domCache.hostProcs || new Map();
    const nodes = [];
    procs.forEach(p => {
      const key = `${p.label}|${p.pid}`;
      let node = _domCache.hostProcs.get(key);
      if (!node || !node.isConnected || node.parentElement !== tbody) {
        node = document.createElement('tr');
        node.innerHTML = `
          <td><b data-f="label">—</b></td>
          <td class="mut" data-f="pid">—</td>
          <td class="num" data-f="cpu">—</td>
          <td class="num" data-f="mem">—</td>
          <td class="num mut" data-f="up">—</td>
        `;
        _domCache.hostProcs.set(key, node);
        tbody.appendChild(node);
      }
      nodes.push(node);
      setText(field(node, 'label'), p.label);
      setText(field(node, 'pid'), String(p.pid));
      const cpuEl = field(node, 'cpu');
      setText(cpuEl, p.cpu.toFixed(1));
      setClass(cpuEl, 'num ' + classForPct(p.cpu, 50, 90));
      setText(field(node, 'mem'), p.mem_mb.toFixed(1));
      setText(field(node, 'up'), fmtUptime(p.uptime_s));
    });
    reorderChildren(tbody, nodes);
  }
}

async function tick() {
  let d;
  try {
    d = await (await fetch('/api/state', {cache:'no-store'})).json();
  } catch (e) {
    setText(document.getElementById('ts'), 'fetch err: ' + e);
    return;
  }
  setText(document.getElementById('ts'), 'updated ' + d.ts.slice(11,19) + ' UTC');

  renderHost(d.host);
  renderEnsemble(d.ensemble);

  // ---- per-account top-bar cells (AEGIS / HERMES / KRAKEN separate) ----
  const accBar = document.getElementById('account-cells');
  const accNodes = [];
  (d.account_stats || []).forEach(a => {
    const node = ensureNode(accBar, a.label, _domCache.accountCells, buildAccountCell);
    accNodes.push(node);
    setText(field(node, 'label'), a.label);
    // Equity: show "—" if unknown (Kraken demo doesn't expose balance)
    setText(field(node, 'equity'), a.equity == null ? '—' : fmtUSD(a.equity));
    const todayEl = field(node, 'todayPnl');
    setText(todayEl, fmtUSD(a.today_pnl));
    setClass(todayEl, 'pnl ' + cls(a.today_pnl));
    setText(field(node, 'todayPct'),
      (a.today_pnl_pct ? fmtPct(a.today_pnl_pct) : ''));
    // Kraken: highlight closed-today since equity is unknown
    if (a.is_demo_no_nav) {
      let sub = `${a.positions} open · ${a.closed_today || 0} closed today`;
      if (a.realized_today) sub += ` · realized ${fmtUSD(a.realized_today)}`;
      if (a.spot_btc_usd) sub += ` · spot ${fmtUSD(a.spot_btc_usd)}`;
      setText(field(node, 'positions'), sub);
    } else {
      setText(field(node, 'positions'),
        `${a.positions} pos · unr ` + fmtUSD(a.unrealized || 0));
    }
  });
  reorderChildren(accBar, accNodes);

  // ---- accounts panel (sidebar) — per-account sparklines ----
  const accBox = document.getElementById('accounts');
  const accSideKeys = new Set();
  const accSideNodes = [];
  (d.accounts || []).forEach(a => {
    accSideKeys.add('side:' + a.label);
    const node = ensureNode(accBox, 'side:' + a.label, _domCache.strategyCards, buildStratCard);
    accSideNodes.push(node);
    setText(field(node, 'name'), a.label);
    const pnlEl = field(node, 'pnl');
    setText(pnlEl,
      fmtUSD(a.today_pnl) + '  ' + fmtPct(a.today_pnl_pct));
    setClass(pnlEl, 'pnl ' + cls(a.today_pnl));
    setText(field(node, 'meta'),
      `Equity ${fmtUSD(a.equity)} · Cash ${fmtUSD(a.cash)} · BP ${fmtUSD(a.buying_power)}`);
    // Sparkline is a per-tick innerHTML refresh (small, can't easily diff SVG)
    const log = field(node, 'log');
    log.innerHTML = sparkline(a.equity_history);
    log.className = '';  // not a log; just the sparkline holder
  });
  if (d.kraken && d.kraken.available) {
    accSideKeys.add('side:KRAKEN');
    const node = ensureNode(accBox, 'side:KRAKEN', _domCache.strategyCards, buildStratCard);
    accSideNodes.push(node);
    setText(field(node, 'name'), 'KRAKEN (futures demo)');
    const k = d.kraken;
    const realizedToday = k.realized_today || 0;
    const todayTotal = realizedToday + (k.btc_pnl_usd || 0);
    const pnlEl = field(node, 'pnl');
    setText(pnlEl, fmtUSD(todayTotal));
    setClass(pnlEl, 'pnl ' + cls(todayTotal));
    // Better Kraken meta: emphasize closed-today activity when no positions open
    let metaText;
    if ((k.positions_count || 0) === 0 && (k.closed_today_count || 0) > 0) {
      metaText = `${k.closed_today_count} closed today · realized ${fmtUSD(realizedToday)} · BTC spot ${fmtUSD(k.spot_btc_usd || 0)}`;
    } else if ((k.positions_count || 0) > 0) {
      metaText = `${k.positions_count} open · entry ${fmtUSD(k.weighted_entry || 0)} · spot ${fmtUSD(k.spot_btc_usd || 0)} · unr ${fmtUSD(k.btc_pnl_usd || 0)}`;
    } else {
      metaText = `no positions today · BTC spot ${fmtUSD(k.spot_btc_usd || 0)}`;
    }
    setText(field(node, 'meta'), metaText);
    field(node, 'log').textContent = '';
  }
  reorderChildren(accBox, accSideNodes);

  // ---- strategy cards ----
  const sBox = document.getElementById('strategies');
  const sNodes = [];
  (d.strategies || []).forEach(s => {
    const node = ensureNode(sBox, 'strat:' + s.name, _domCache.strategyCards, buildStratCard);
    sNodes.push(node);
    const dot = s.running ? '🟢 ' : '🔴 ';
    setText(field(node, 'name'), dot + s.name);
    const pnlEl = field(node, 'pnl');
    setText(pnlEl, fmtUSD(s.unrealized_pnl || 0));
    setClass(pnlEl, 'pnl ' + cls(s.unrealized_pnl));
    setText(field(node, 'meta'),
      `${s.account} · ${s.desc} · ${s.positions||0} pos · exposure ${fmtUSD(s.exposure || 0)}`);
    setText(field(node, 'log'), s.last_log || '(no log)');
    field(node, 'log').className = 'last-log';
  });
  reorderChildren(sBox, sNodes);

  // ---- positions table ----
  const pBody = document.querySelector('#positions tbody');
  const positions = d.positions || [];
  if (positions.length === 0) {
    pBody.innerHTML = '<tr><td colspan="10" class="empty">no open positions</td></tr>';
    _domCache.positionRows.clear();
  } else {
    const pNodes = [];
    positions.forEach(p => {
      const key = `${p.account}|${p.symbol}|${p.side}|${p.qty}`;
      const node = ensureNode(pBody, key, _domCache.positionRows, buildPositionRow);
      pNodes.push(node);
      const accEl = field(node, 'acct');
      setText(accEl, p.account);
      setClass(accEl, 'badge ' + (p.account || '').toLowerCase());
      setText(field(node, 'strategy'), p.strategy || '—');
      setText(field(node, 'symbol'), p.symbol || '');
      const sideEl = field(node, 'side');
      setText(sideEl, p.side || '');
      setClass(sideEl, 'side-' + (p.side || ''));
      setText(field(node, 'qty'), String(p.qty));
      setText(field(node, 'entry'), Number(p.avg_entry).toLocaleString(undefined,{maximumFractionDigits:2}));
      setText(field(node, 'mark'), p.current ? Number(p.current).toLocaleString(undefined,{maximumFractionDigits:2}) : '—');
      setText(field(node, 'mktVal'), fmtUSD(p.market_value || 0));
      const plEl = field(node, 'pl');
      setText(plEl, fmtUSD(p.unrealized_pl || 0));
      setClass(plEl, 'num ' + cls(p.unrealized_pl));
      const plPctEl = field(node, 'plPct');
      setText(plPctEl, fmtPct(p.unrealized_plpc || 0));
      setClass(plPctEl, 'num ' + cls(p.unrealized_plpc));
    });
    reorderChildren(pBody, pNodes);
  }

  // ---- activity feed ----
  const aBox = document.getElementById('activity');
  const orders = d.orders || [];
  if (orders.length === 0) {
    aBox.innerHTML = '<div class="empty">no recent activity</div>';
    _domCache.activityRows.clear();
  } else {
    const aNodes = [];
    orders.forEach(o => {
      // Use submitted-time + symbol + side as a row key (orders are ~immutable once submitted)
      const key = `${o.submitted}|${o.symbol}|${o.side}|${o.qty}`;
      const node = ensureNode(aBox, key, _domCache.activityRows, buildActivityRow);
      aNodes.push(node);
      setText(field(node, 'time'), (o.submitted||'').slice(11,19));
      const fillTxt = o.filled_avg
        ? ` @ $${Number(o.filled_avg).toLocaleString(undefined,{maximumFractionDigits:2})}`
        : '';
      const pl = o.realized_pl;
      const plChip = (pl != null && Math.abs(pl) > 0.001)
        ? `<span class="pl-chip ${pl >= 0 ? 'pos' : 'neg'}">${pl >= 0 ? '+' : ''}${fmtUSD(pl)}</span>`
        : '';
      const bodyEl = field(node, 'body');
      bodyEl.innerHTML = `
        <span class="badge ${(o.account||'').toLowerCase()}">${o.account}</span>
        <span class="side-${o.side}">${(o.side||'').toUpperCase()}</span>
        ${o.qty} <b>${o.symbol||''}</b>${fillTxt}
        ${plChip}
      `;
      const stEl = field(node, 'status');
      stEl.innerHTML = `<span class="badge ${o.status}">${o.status}</span>`;
    });
    reorderChildren(aBox, aNodes);
  }
}

// ---------- Equity Chart (TradingView lightweight-charts v5) ------------
let equityChart = null;
let equityAreaSeries = null;

function initEquityChart() {
  if (equityChart) return;
  const el = document.getElementById('equity-chart');
  if (!el || typeof LightweightCharts === 'undefined') return;
  equityChart = LightweightCharts.createChart(el, {
    layout: {
      background: { type: 'solid', color: '#1e222d' },
      textColor:  '#d1d4dc',
      fontSize:   11,
      fontFamily: '-apple-system, BlinkMacSystemFont, "Inter", sans-serif',
    },
    grid: {
      vertLines: { color: '#2a2e39' },
      horzLines: { color: '#2a2e39' },
    },
    timeScale: {
      timeVisible:    true,
      secondsVisible: false,
      borderColor:    '#363c4e',
    },
    rightPriceScale: {
      borderColor: '#363c4e',
    },
    crosshair: { mode: LightweightCharts.CrosshairMode.Normal },
    autoSize: true,
  });
  equityAreaSeries = equityChart.addSeries(LightweightCharts.AreaSeries, {
    lineColor:   '#26a69a',
    topColor:    'rgba(38, 166, 154, 0.4)',
    bottomColor: 'rgba(38, 166, 154, 0.0)',
    lineWidth:   2,
    priceFormat: { type: 'price', precision: 2, minMove: 0.01 },
  });
}

function setEquityData(accounts) {
  if (!equityAreaSeries) return;
  if (!accounts || !accounts.length) return;
  const ref = accounts[0].history_timestamps || [];
  if (!ref.length) return;
  // Sum equity at each timestamp across accounts. Treat null as last known value.
  const sums = [];
  for (let i = 0; i < ref.length; i++) {
    let total = 0;
    for (const a of accounts) {
      const eq = (a.equity_history || [])[i];
      if (eq != null) total += eq;
    }
    if (total > 0) {
      sums.push({ time: ref[i], value: total });
    }
  }
  if (sums.length < 2) return;
  // Lightweight-charts requires strictly ascending time and no dupes
  sums.sort((a, b) => a.time - b.time);
  const uniq = [];
  let lastT = -1;
  for (const p of sums) {
    if (p.time !== lastT) { uniq.push(p); lastT = p.time; }
  }
  equityAreaSeries.setData(uniq);
  equityChart.timeScale().fitContent();
}

async function tickWithChart() {
  await tick();
  // After regular tick(), update chart from cached d (we re-fetch here to keep d local)
  try {
    const d = await (await fetch('/api/state', {cache:'no-store'})).json();
    initEquityChart();
    setEquityData(d.accounts);
  } catch (e) {
    // Soft-fail — chart updates are non-critical
  }
}

tick();
initEquityChart();
// Pull chart data once after first tick
setTimeout(async () => {
  try {
    const d = await (await fetch('/api/state', {cache:'no-store'})).json();
    setEquityData(d.accounts);
  } catch (e) {}
}, 200);

// In-flight guard — never queue two simultaneous fetches.
let _inFlight = false;
async function safeTick() {
  if (_inFlight) return;
  _inFlight = true;
  try { await tick(); }
  finally { _inFlight = false; }
}
// 1 Hz refresh. Backend is TTL-cached so most ticks resolve in ~50ms.
setInterval(safeTick, 1000);
// Chart data only changes on 5-min bars — pull every 60s.
setInterval(async () => {
  try {
    const d = await (await fetch('/api/state', {cache:'no-store'})).json();
    setEquityData(d.accounts);
  } catch (e) {}
}, 60000);
</script>
</body></html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8504, log_level="warning")
