"""Daily live-paper runner — fires all 3 lanes into IBKR paper account.

Architecture:
  - Single daily entry point (intended to be scheduled via launchctl ~21:30 BST,
    after US market close).
  - Per lane: build today's decision, construct combo order, submit to IBKR
    paper account with lane-specific tag.
  - All trades logged to data/trading_system.db with run_id tags:
      live_lane1_quant_YYYYMMDD
      live_lane2_council_YYYYMMDD
      live_lane3_pure_api_YYYYMMDD

Safety:
  - Refuses to run if IBKR connection is on a non-paper port.
  - Refuses to run if Gateway not connected (raises clear error).
  - Per-lane try/except: one lane's failure doesn't kill the others.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ingest.ibkr_client import IBKRClient, IBKRConfig  # noqa: E402
from ingest.ibkr_orders import (  # noqa: E402
    ComboOrderResult,
    construct_iron_condor_request,
    construct_straddle_request,
    construct_strangle_request,
    construct_vertical_spread_request,
    place_combo_paper,
)

from fullonriskmodelterminator.council import vote, load_env  # noqa: E402
from fullonriskmodelterminator.menu import TradeDecision, SIZE_TO_PCT  # noqa: E402

logger = logging.getLogger(__name__)


def _today_tag(prefix: str) -> str:
    return f"{prefix}_{date.today().strftime('%Y%m%d')}"


def _expiry_for_dte(today: date, dte_days: int) -> str:
    target = today + timedelta(days=max(dte_days, 1))
    # Snap to next Friday (most liquid SPY expiry)
    while target.weekday() != 4:
        target += timedelta(days=1)
    return target.strftime("%Y%m%d")


def _round_strike(spot: float, offset_pct: float = 0.0) -> float:
    return round(spot * (1 + offset_pct))


def get_market_state(ib_client: IBKRClient) -> dict[str, Any]:
    """Pull current market data needed for lane decisions."""
    summary = ib_client.account_summary()
    nav_str = summary.get("NetLiquidation__USD") or summary.get("NetLiquidation__")
    nav = float(nav_str) if nav_str else 100_000.0
    cash_str = summary.get("AvailableFunds__USD") or summary.get("AvailableFunds__")
    cash = float(cash_str) if cash_str else nav

    # SPY spot
    from ib_insync import Stock
    spy = Stock("SPY", "SMART", "USD")
    ib_client.ib.qualifyContracts(spy)
    ticker = ib_client.ib.reqMktData(spy, "", False, False)
    ib_client.ib.sleep(2)
    spot = ticker.last or ticker.close or ticker.bid or 0.0
    ib_client.ib.cancelMktData(spy)

    # VIX
    from ib_insync import Index
    vix = Index("VIX", "CBOE", "USD")
    try:
        ib_client.ib.qualifyContracts(vix)
        vt = ib_client.ib.reqMktData(vix, "", False, False)
        ib_client.ib.sleep(2)
        vix_val = vt.last or vt.close or 20.0
        ib_client.ib.cancelMktData(vix)
    except Exception:
        vix_val = 20.0

    return {
        "today": date.today().isoformat(),
        "spy_price": float(spot),
        "vix": float(vix_val),
        "iv": float(vix_val) / 100.0,
        "iv_rank": 0.5,    # placeholder; would need historical data
        "rv20": 0.15,      # placeholder
        "vrp": 0.02,
        "vix_5d_roc": 0.0,
        "spy_5d_ret": 0.0,
        "spy_20d_ret": 0.0,
        "nav": nav,
        "cash": cash,
        "n_open": len(ib_client.positions()),
        "total_risk": 0.0,
        "pct_at_risk": 0.0,
        "nav_trend": "neutral",
        "account_summary": summary,
    }


# ============================================================================
# Lane 1 — Quant Realistic (iron condor on Mondays, satellite straddles on earnings)
# ============================================================================

def lane1_quant_decision(state: dict) -> dict | None:
    """Conservative quant: only enter SPY iron condor on Mondays.

    Real implementation would consult the frozen strategy logic from
    scripts/run_daily_paper.py. For tomorrow's first deploy, just place
    one SPY iron condor at delta 0.12, 30 DTE if today is Monday.
    """
    today = date.today()
    if today.weekday() != 0:   # 0 = Monday
        return None
    spot = state["spy_price"]
    if spot <= 0:
        return None
    return {
        "action": "iron_condor",
        "underlying": "SPY",
        "expiry": _expiry_for_dte(today, 35),
        "long_put": _round_strike(spot, -0.06),
        "short_put": _round_strike(spot, -0.05),
        "short_call": _round_strike(spot, +0.05),
        "long_call": _round_strike(spot, +0.06),
        "quantity": max(1, int(state["nav"] * 0.01 / 400)),   # 1% NAV risk / $400 max loss per contract
        "limit_credit": 0.40,    # ~$40 credit per contract
        "tag": _today_tag("live_lane1_quant"),
    }


def lane1_execute(state: dict, decision: dict, ib_client: IBKRClient) -> ComboOrderResult:
    req = construct_iron_condor_request(
        underlying=decision["underlying"],
        expiry=decision["expiry"],
        long_put=decision["long_put"],
        short_put=decision["short_put"],
        short_call=decision["short_call"],
        long_call=decision["long_call"],
        quantity=decision["quantity"],
        limit_credit=decision["limit_credit"],
        tag=decision["tag"],
    )
    return place_combo_paper(
        ib_client.ib, req,
        (ib_client.config.host, ib_client.config.port),
        fill_timeout_sec=20.0,
    )


# ============================================================================
# Lane 2 — NIM Council (constrained menu)
# ============================================================================

def lane2_council_decision(state: dict) -> tuple[TradeDecision | None, list[dict]]:
    """8 NIM models vote on today's action."""
    load_env()
    decision, raw_votes = vote(state)
    return decision, raw_votes


def lane2_execute(state: dict, decision: TradeDecision, ib_client: IBKRClient) -> ComboOrderResult | None:
    """Translate council's menu decision into an IBKR combo order."""
    today = date.today()
    spot = state["spy_price"]
    if decision.action == "no_trade":
        return None
    if decision.action == "close_all":
        # Would need to iterate positions and close each; skip for first live day
        return None

    dte = {"0dte": 0, "weekly": 7, "monthly": 30}[decision.dte_bucket]
    expiry = _expiry_for_dte(today, dte)
    nav = state["nav"]
    risk_pct = SIZE_TO_PCT[decision.size_tier]
    risk_dollars = nav * risk_pct
    tag = _today_tag(f"live_lane2_council_{decision.action}")

    atm = _round_strike(spot, 0.0)
    near_c = _round_strike(spot, 0.02)
    far_c = _round_strike(spot, 0.05)
    near_p = _round_strike(spot, -0.02)
    far_p = _round_strike(spot, -0.05)
    wide_c = _round_strike(spot, 0.04)
    wide_p = _round_strike(spot, -0.04)

    if decision.action == "short_iron_condor":
        # Approximate credit and qty
        qty = max(1, int(risk_dollars / 400))
        req = construct_iron_condor_request(
            underlying="SPY", expiry=expiry,
            long_put=far_p, short_put=near_p,
            short_call=near_c, long_call=far_c,
            quantity=qty, limit_credit=0.40, tag=tag,
        )
    elif decision.action == "short_strangle":
        qty = max(1, int(risk_dollars / (spot * 0.05 * 100)))
        req = construct_strangle_request(
            underlying="SPY", expiry=expiry,
            call_strike=wide_c, put_strike=wide_p,
            direction="short", quantity=qty, limit_premium=2.00, tag=tag,
        )
    elif decision.action == "long_straddle":
        qty = max(1, int(risk_dollars / 800))
        req = construct_straddle_request(
            underlying="SPY", expiry=expiry, strike=atm,
            direction="long", quantity=qty, limit_premium=8.00, tag=tag,
        )
    elif decision.action == "credit_put_spread":
        qty = max(1, int(risk_dollars / 300))
        req = construct_vertical_spread_request(
            underlying="SPY", expiry=expiry, right="P",
            short_strike=near_p, long_strike=far_p,
            quantity=qty, limit_credit=0.50, tag=tag,
        )
    elif decision.action == "credit_call_spread":
        qty = max(1, int(risk_dollars / 300))
        req = construct_vertical_spread_request(
            underlying="SPY", expiry=expiry, right="C",
            short_strike=near_c, long_strike=far_c,
            quantity=qty, limit_credit=0.50, tag=tag,
        )
    else:
        # Other actions (long_call, long_put, long_calendar) — not wired for
        # first live day. Defer to Lane 3 which handles arbitrary legs.
        return None

    return place_combo_paper(
        ib_client.ib, req,
        (ib_client.config.host, ib_client.config.port),
        fill_timeout_sec=20.0,
    )


# ============================================================================
# Lane 3 — Pure-API (single NIM picks free-form legs)
# ============================================================================

def lane3_pure_api_decision(state: dict) -> dict | None:
    """Single NIM call, free-form leg list."""
    import os
    from openai import OpenAI
    from fullonriskmodelterminator.pure_api_trader import PROMPT, _ask, _parse, DEFAULT_MODEL

    load_env()
    key = os.environ.get("NVIDIA_API_KEY")
    if not key:
        return None
    client = OpenAI(api_key=key, base_url="https://integrate.api.nvidia.com/v1", timeout=60.0)
    prompt = PROMPT.format(
        today=state["today"],
        spy=state["spy_price"], ret5=state.get("spy_5d_ret", 0),
        ret20=state.get("spy_20d_ret", 0), vix=state["vix"],
        vix_roc=state.get("vix_5d_roc", 0), iv=state["iv"],
        nav=state["nav"], cash=state["cash"], n_open=state["n_open"],
    )
    return _ask(client, DEFAULT_MODEL, prompt)


def lane3_execute(state: dict, decision: dict, ib_client: IBKRClient) -> ComboOrderResult | None:
    """Translate Lane 3's free-form legs into an IBKR combo order."""
    legs = decision.get("open_legs") or []
    if not legs:
        return None
    from ingest.ibkr_orders import LegSpec, ComboOrderRequest

    today = date.today()
    leg_specs = []
    max_dte = 7
    for leg in legs:
        try:
            right = str(leg.get("right", "C")).upper()[:1]
            if right not in ("C", "P"):
                continue
            strike = round(float(leg.get("strike", state["spy_price"])))
            dte = max(0, min(60, int(leg.get("dte_days", 7))))
            qty = max(1, min(50, int(leg.get("qty", 1))))
            direction = str(leg.get("direction", "long")).lower()
            if direction not in ("long", "short"):
                direction = "long"
            max_dte = max(max_dte, dte)
            leg_specs.append(LegSpec(
                right=right,        # type: ignore
                strike=strike,
                expiry=_expiry_for_dte(today, dte),
                direction=direction,  # type: ignore
                quantity=qty,
            ))
        except (TypeError, ValueError):
            continue
    if not leg_specs:
        return None

    tag = _today_tag(f"live_lane3_pure_api_{len(leg_specs)}leg")
    # For arbitrary multi-leg structures we don't know in advance whether net
    # is credit or debit — submit at the natural mid-price. For paper trading
    # this is OK; production would compute net mid first.
    req = ComboOrderRequest(
        underlying="SPY",
        legs=leg_specs,
        limit_price=0.0,    # market-ish; let paper account fill at NBBO
        tag=tag,
    )
    return place_combo_paper(
        ib_client.ib, req,
        (ib_client.config.host, ib_client.config.port),
        fill_timeout_sec=20.0,
    )


# ============================================================================
# Main daily orchestrator
# ============================================================================

def run_one_day(lanes: tuple[bool, bool, bool] = (True, True, True),
                 dry_run: bool = False) -> dict:
    """Connect to IBKR, fetch state, execute all enabled lanes once."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    results: dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "dry_run": dry_run,
        "lanes": {},
    }

    try:
        ib_client = IBKRClient(paper=True)
        ib_client.connect()
    except Exception as exc:
        logger.error("IBKR connection failed: %s", exc)
        results["error"] = f"IBKR connection failed: {exc}"
        return results

    try:
        state = get_market_state(ib_client)
        results["market_state"] = {k: v for k, v in state.items() if k != "account_summary"}
        results["account_summary"] = state.get("account_summary", {})

        if lanes[0]:
            try:
                d = lane1_quant_decision(state)
                if d is None:
                    results["lanes"]["lane1"] = {"action": "no_trade", "reason": "not_monday"}
                elif dry_run:
                    results["lanes"]["lane1"] = {"dry_decision": d}
                else:
                    r = lane1_execute(state, d, ib_client)
                    results["lanes"]["lane1"] = {
                        "decision": d,
                        "filled": r.filled, "fill_price": r.fill_price,
                        "error": r.error,
                    }
            except Exception as exc:
                logger.exception("lane1 failed")
                results["lanes"]["lane1"] = {"error": str(exc)}

        if lanes[1]:
            try:
                decision, raw_votes = lane2_council_decision(state)
                if dry_run:
                    results["lanes"]["lane2"] = {
                        "dry_decision": decision.to_dict() if decision else None,
                        "n_votes_ok": sum(1 for v in raw_votes if v.get("decision") is not None),
                    }
                elif decision:
                    r = lane2_execute(state, decision, ib_client)
                    results["lanes"]["lane2"] = {
                        "decision": decision.to_dict(),
                        "n_votes_ok": sum(1 for v in raw_votes if v.get("decision") is not None),
                        "filled": r.filled if r else None,
                        "fill_price": r.fill_price if r else None,
                        "error": r.error if r else "no_trade_action",
                    }
            except Exception as exc:
                logger.exception("lane2 failed")
                results["lanes"]["lane2"] = {"error": str(exc)}

        if lanes[2]:
            try:
                d = lane3_pure_api_decision(state)
                if dry_run:
                    results["lanes"]["lane3"] = {"dry_decision": d}
                elif d:
                    r = lane3_execute(state, d, ib_client)
                    results["lanes"]["lane3"] = {
                        "decision": d,
                        "filled": r.filled if r else None,
                        "fill_price": r.fill_price if r else None,
                        "error": r.error if r else "no_legs",
                    }
            except Exception as exc:
                logger.exception("lane3 failed")
                results["lanes"]["lane3"] = {"error": str(exc)}
    finally:
        ib_client.disconnect()

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Generate decisions but don't submit orders")
    parser.add_argument("--lanes", default="1,2,3",
                        help="Comma-separated lanes to run (default 1,2,3)")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "fullonriskmodelterminator" / "runs" / "live")
    args = parser.parse_args()

    enabled = tuple(str(i) in args.lanes.split(",") for i in (1, 2, 3))
    result = run_one_day(lanes=enabled, dry_run=args.dry_run)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fname = args.out_dir / f"daily_{date.today().strftime('%Y%m%d')}_{datetime.now(timezone.utc).strftime('%H%M%S')}.json"
    fname.write_text(json.dumps(result, indent=2, default=str))
    print(f"Wrote: {fname}")
    print(json.dumps(result, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
