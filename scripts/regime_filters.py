"""Regime filter primitives for the live HFT host.

Mirrors the logic in `experiments/strategy_swarm_lab.py` so live decisions
match what the simulator validates. Lane-specific configuration lives in
`data/regime_filters_config.json` — edit that file to tune without code
changes.

Filters supported (60-model consensus): vol_of_vol_gate (Opus max #9),
news_blackout (Claude swarm top), atr_gate (NIM top), abs_return_gate.
"""
from __future__ import annotations

import json
import math
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO = Path("/Users/weeoto/turbo-carnival")
CONFIG_PATH = REPO / "data" / "regime_filters_config.json"

# Hard-coded today's news events (UTC). Update via tools/news_event_loader when
# we wire up the live news feed integration.
DEFAULT_NEWS_EVENTS_UTC = [
    "2026-05-11T14:07:00Z",  # Trump rejects Iran
    "2026-05-11T17:10:00Z",  # Apple-Intel
    "2026-05-11T18:00:00Z",  # AMD supercycle
    "2026-05-11T18:38:00Z",  # Musk + Cook Xi summit
    "2026-05-11T18:52:00Z",  # Meta lawsuit
]


def _load_config() -> dict[str, Any]:
    if CONFIG_PATH.exists():
        return json.loads(CONFIG_PATH.read_text())
    return {}


def _compute_atr_bps(history: deque[float], window: int = 20) -> float:
    """Return mean absolute 1-bar return over the last `window` bars, in bps."""
    if len(history) < window:
        return 0.0
    arr = list(history)[-window:]
    diffs = [abs(arr[i] / arr[i - 1] - 1.0) for i in range(1, len(arr))]
    return float(sum(diffs) / len(diffs)) * 10000.0 if diffs else 0.0


def _vol_of_vol_ratio(history: deque[float]) -> float:
    """5-bar rolling stdev / 20-bar trailing median absolute return."""
    if len(history) < 25:
        return 0.0
    arr = list(history)
    short_returns = [arr[i] / arr[i - 1] - 1.0 for i in range(len(arr) - 5, len(arr))]
    long_returns = [abs(arr[i] / arr[i - 1] - 1.0) for i in range(len(arr) - 25, len(arr) - 5)]
    if not short_returns or not long_returns:
        return 0.0
    mean = sum(short_returns) / len(short_returns)
    var = sum((r - mean) ** 2 for r in short_returns) / max(len(short_returns), 1)
    short_sd = math.sqrt(var)
    long_returns_sorted = sorted(long_returns)
    med = long_returns_sorted[len(long_returns_sorted) // 2]
    if med <= 0:
        return 0.0
    return short_sd / med


def is_within_news_blackout(now_utc: datetime, cooldown_sec: int,
                             news_events: list[str] | None = None) -> tuple[bool, str | None]:
    """Return (in_blackout, anchor_iso). Checks all known news events for the day."""
    events = news_events or DEFAULT_NEWS_EVENTS_UTC
    for ev_iso in events:
        ev = datetime.fromisoformat(ev_iso.replace("Z", "+00:00"))
        delta = (now_utc - ev).total_seconds()
        if 0 <= delta <= cooldown_sec:
            return True, ev_iso
    return False, None


def block_entry(
    lane: str,
    symbol: str,
    history: deque[float],
    *,
    now_utc: datetime | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[bool, str]:
    """Return (blocked, reason). False, "ok" if entry is permitted."""
    now_utc = now_utc or datetime.now(timezone.utc)
    cfg = config if config is not None else _load_config()
    lane_cfg = cfg.get(lane, cfg.get("_default", {}))
    if not lane_cfg.get("enabled", True):
        return False, "filters_disabled_for_lane"

    # 1) Vol-of-vol gate (Opus max #9 — top single-vote regime detector)
    vov = lane_cfg.get("vol_of_vol_gate")
    if vov and vov.get("enabled", True):
        ratio = _vol_of_vol_ratio(history)
        thresh = float(vov.get("ratio", 1.4))
        if ratio >= thresh:
            return True, f"vov_gate {ratio:.2f}>={thresh}"

    # 2) ATR gate (NIM panel top regime filter — 7 of 20)
    atr = lane_cfg.get("atr_gate")
    if atr and atr.get("enabled", True):
        atr_bps = _compute_atr_bps(history, int(atr.get("window", 20)))
        thresh_bps = float(atr.get("threshold_bps", 30))
        if atr_bps >= thresh_bps:
            return True, f"atr_gate {atr_bps:.1f}>={thresh_bps}"

    # 3) News blackout (Claude swarm top filter — 28 of 60)
    news = lane_cfg.get("news_blackout")
    if news and news.get("enabled", True):
        cooldown = int(news.get("cooldown_sec", 300))
        in_b, anchor = is_within_news_blackout(now_utc, cooldown)
        if in_b:
            return True, f"news_blackout anchored_to {anchor}"

    # 4) Absolute return gate (block one-bar pop entries)
    abs_ret = lane_cfg.get("abs_return_gate")
    if abs_ret and abs_ret.get("enabled", True) and len(history) >= 2:
        one_bps = abs(history[-1] / history[-2] - 1.0) * 10000.0
        thresh = float(abs_ret.get("threshold_bps", 30))
        if one_bps >= thresh:
            return True, f"abs_return_gate {one_bps:.1f}>={thresh}"

    # 5) Daily-loss circuit breaker is enforced elsewhere (in host loop with NAV).
    return False, "ok"


def default_config() -> dict[str, Any]:
    """Return a sensible default config: gates ON for Aegis, OFF (paused) for Falcon."""
    return {
        "_meta": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "60-model swarm + strategy_swarm_lab winners",
            "notes": "Edit this file to tune; no code changes needed to retune.",
        },
        "Aegis": {
            "enabled": True,
            "vol_of_vol_gate": {"enabled": True, "ratio": 1.4},
            "atr_gate": {"enabled": True, "window": 20, "threshold_bps": 30.0},
            "news_blackout": {"enabled": True, "cooldown_sec": 300},
            "abs_return_gate": {"enabled": False, "threshold_bps": 30.0},
        },
        "Falcon": {
            "enabled": False,
            "_paused_rationale": "60-model unanimous (60/60) pause per Day 1 review. Do not re-enable without rebuild.",
        },
        "Valk": {
            "enabled": False,
            "_note": "Valk is an options lane and uses separate exit logic; HFT regime filters don't apply.",
        },
        "_default": {
            "enabled": True,
            "vol_of_vol_gate": {"enabled": True, "ratio": 1.4},
            "news_blackout": {"enabled": True, "cooldown_sec": 300},
        },
    }


def ensure_config_exists() -> Path:
    """Write the default config if no config file is present."""
    if not CONFIG_PATH.exists():
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(default_config(), indent=2))
    return CONFIG_PATH


if __name__ == "__main__":
    # Self-check: write default config and print which filters are active.
    p = ensure_config_exists()
    cfg = _load_config()
    print(f"config: {p}")
    for lane in ("Aegis", "Falcon", "Valk"):
        lc = cfg.get(lane, {})
        if not lc.get("enabled", False):
            print(f"  {lane}: DISABLED  ({lc.get('_paused_rationale') or lc.get('_note') or ''})")
            continue
        active = [k for k in ("vol_of_vol_gate", "atr_gate", "news_blackout", "abs_return_gate")
                  if lc.get(k, {}).get("enabled")]
        print(f"  {lane}: filters active = {active}")
