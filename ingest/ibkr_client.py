"""IBKR adapter — wraps ib_insync for paper-trading deployment.

Connects to a running IB Gateway / TWS on localhost. Does NOT use
username/password directly — those are entered into the Gateway/TWS GUI at
startup. The Gateway then exposes a socket on:

  * 4002 — IB Gateway, paper account (recommended, headless)
  * 4001 — IB Gateway, live account
  * 7497 — TWS, paper account
  * 7496 — TWS, live account

Account / paper-trading credentials should be stored ONLY in the GUI app, not
in code or env files. See ops/ibkr_setup.md for the manual setup procedure.

Usage:
    from ingest.ibkr_client import IBKRClient
    with IBKRClient(paper=True) as ib:
        chain = ib.fetch_spy_chain(target_dte=30)
        summary = ib.account_summary()
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any, Iterable

try:
    from ib_insync import IB, Contract, Index, Option, Stock, util
except Exception as exc:  # pragma: no cover -- optional dep
    IB = None  # type: ignore
    raise ImportError(
        "ib_insync is required. Install with: .venv/bin/pip install ib_insync"
    ) from exc

logger = logging.getLogger(__name__)


# Default ports — only paper modes are wired by default to prevent live oops
DEFAULT_PAPER_GATEWAY_PORT = 4002
DEFAULT_PAPER_TWS_PORT = 7497
DEFAULT_LIVE_GATEWAY_PORT = 4001
DEFAULT_LIVE_TWS_PORT = 7496


@dataclass(frozen=True)
class IBKRConfig:
    host: str = "127.0.0.1"
    port: int = DEFAULT_PAPER_GATEWAY_PORT
    client_id: int = 17  # arbitrary; just needs to be unique per simultaneous connection
    timeout_sec: float = 8.0


class IBKRClient:
    def __init__(
        self,
        *,
        paper: bool = True,
        config: IBKRConfig | None = None,
    ) -> None:
        if not paper:
            # Hard guard against accidentally connecting to live.
            raise RuntimeError(
                "IBKRClient is paper-only by default. "
                "Set paper=True. To connect live, instantiate with explicit live config "
                "via the dedicated _IBKRLiveClient interface (intentionally not provided)."
            )
        self.config = config or IBKRConfig()
        self.ib = IB()

    # ------------------------------------------------------------------ context
    def __enter__(self) -> "IBKRClient":
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.disconnect()

    def connect(self) -> None:
        cfg = self.config
        logger.info("connecting IB host=%s port=%s clientId=%s", cfg.host, cfg.port, cfg.client_id)
        self.ib.connect(cfg.host, cfg.port, clientId=cfg.client_id, timeout=cfg.timeout_sec)
        if not self.ib.isConnected():
            raise RuntimeError(
                f"could not connect to IB at {cfg.host}:{cfg.port}. "
                "Is IB Gateway / TWS running with API enabled? See ops/ibkr_setup.md."
            )
        logger.info("IB connected. server version %s", self.ib.client.serverVersion())

    def disconnect(self) -> None:
        if self.ib.isConnected():
            self.ib.disconnect()
            logger.info("IB disconnected")

    # ----------------------------------------------------------------- account
    def account_summary(self) -> dict[str, Any]:
        """Return a flat dict of key account values (NetLiquidation, BuyingPower, etc.)."""
        rows = self.ib.accountSummary()
        out: dict[str, Any] = {}
        for r in rows:
            out[f"{r.tag}__{r.currency}"] = r.value
        return out

    def positions(self) -> list[dict[str, Any]]:
        return [
            {
                "symbol": p.contract.symbol,
                "secType": p.contract.secType,
                "exchange": p.contract.exchange,
                "currency": p.contract.currency,
                "position": p.position,
                "avg_cost": p.avgCost,
            }
            for p in self.ib.positions()
        ]

    # --------------------------------------------------------------- options
    def fetch_spy_chain(self, *, target_dte: int = 30) -> list[dict[str, Any]]:
        """Fetch SPY option chain near target days-to-expiry.

        Returns a list of contract dicts with strike, right (C/P), expiry, mid quote.
        Uses the optParams + reqMktData flow; this is the standard ib_insync pattern.
        """
        spy = Stock("SPY", "SMART", "USD")
        self.ib.qualifyContracts(spy)
        chains = self.ib.reqSecDefOptParams(spy.symbol, "", spy.secType, spy.conId)
        if not chains:
            raise RuntimeError("no option chain returned for SPY")
        # Pick CBOE (or SMART if not present) and the expirations closest to target_dte
        chain = next((c for c in chains if c.exchange == "SMART"), chains[0])
        today = date.today()
        target = today + timedelta(days=target_dte)
        expirations = sorted(
            chain.expirations,
            key=lambda e: abs((date(int(e[:4]), int(e[4:6]), int(e[6:8])) - target).days),
        )
        if not expirations:
            raise RuntimeError("no expirations found")
        chosen_exp = expirations[0]
        # Pull current SPY mid first
        spy_ticker = self.ib.reqMktData(spy, "", False, False)
        self.ib.sleep(2)
        spot = (spy_ticker.last or spy_ticker.close or spy_ticker.bid)
        if spot is None:
            self.ib.cancelMktData(spy)
            raise RuntimeError("no SPY spot price available")
        # Build option contracts +/- 5% around spot, $5 strike grid
        strikes = sorted(s for s in chain.strikes if 0.92 * spot <= s <= 1.08 * spot)
        contracts = [
            Option("SPY", chosen_exp, k, right, "SMART", tradingClass=chain.tradingClass)
            for k in strikes for right in ("C", "P")
        ]
        self.ib.qualifyContracts(*contracts)
        tickers = [self.ib.reqMktData(c, "", False, False) for c in contracts]
        self.ib.sleep(3)
        out = []
        for c, t in zip(contracts, tickers):
            mid = None
            if t.bid is not None and t.ask is not None and t.bid > 0 and t.ask > 0:
                mid = (t.bid + t.ask) / 2
            out.append({
                "symbol": c.symbol,
                "expiry": chosen_exp,
                "strike": float(c.strike),
                "right": c.right,
                "bid": float(t.bid) if t.bid else None,
                "ask": float(t.ask) if t.ask else None,
                "mid": float(mid) if mid else None,
            })
            self.ib.cancelMktData(c)
        self.ib.cancelMktData(spy)
        return out

    # --------------------------------------------------------------- orders
    def place_iron_condor_paper(
        self,
        *,
        expiry: str,
        long_put: float,
        short_put: float,
        short_call: float,
        long_call: float,
        quantity: int = 1,
        limit_credit: float | None = None,
    ) -> dict[str, Any]:
        """Place a 4-leg iron condor on SPY as a single combo order.

        Refuses to place if not connected to a paper port.
        """
        if self.config.port not in (DEFAULT_PAPER_GATEWAY_PORT, DEFAULT_PAPER_TWS_PORT):
            raise RuntimeError(
                f"refusing to place order on non-paper port {self.config.port}"
            )
        # NOT IMPLEMENTED IN THIS COMMIT — scaffold only.
        # Full combo order construction requires:
        #   - qualifyContracts on each leg (4 Option contracts)
        #   - construct ComboLeg list with appropriate ratios + actions
        #   - construct Bag contract with comboLegs
        #   - place LimitOrder with limit_credit as the order price
        #   - poll for fill via ib.openOrders() / ib.fills()
        raise NotImplementedError(
            "place_iron_condor_paper scaffold only. "
            "Wire combo legs + LimitOrder once initial connection is verified."
        )


def smoketest_connection(*, paper: bool = True) -> dict[str, Any]:
    """Try connect → fetch account summary → disconnect. Returns diagnostic dict."""
    out = {"connected": False, "error": None, "summary_keys": [], "summary_sample": {}}
    try:
        with IBKRClient(paper=paper) as ib:
            out["connected"] = True
            s = ib.account_summary()
            out["summary_keys"] = sorted(s.keys())[:30]
            out["summary_sample"] = {k: s[k] for k in list(out["summary_keys"])[:5]}
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    import json
    res = smoketest_connection(paper=True)
    print(json.dumps(res, indent=2, default=str))
