# Phase Status

**Phase:** All code phases complete (0–3 + eval framework + prereg draft) — awaiting CEO sign-off + freeze tag
**Machine:** Mac mini (production target; Phase 0 development allowed)
**Frozen cycle:** Not started — code changes are permitted
**Last updated:** 2026-05-06

---

## Completed

- Project skeleton (directory layout per `README.md`)
- `metadata/strategic_lock.md` — locked hybrid VRP core + earnings satellite strategy
- `metadata/capital_policy.md` — $10k paper notional, sleeve allocation, kill switch limits
- `sql/schema.sql` — `runs`, `gate_decisions`, `agents_log`, `trades`, `trade_legs`
- `dashboard/app.py` — read-only Streamlit dashboard skeleton (overview, runs, gate decisions, trades, agent activity)
- `scripts/run_phase0_demo.py` — synthetic SPY-momentum demo seed (idempotent; populates all 5 tables under `run_id = paper_v0_demo`)
- `tests/` + root `conftest.py` — pytest coverage for schema constraints (16 tests) and dashboard helpers (14 tests)

### Phase 1 (ingest + features)

- `ingest/yfinance_adapter.py` — yfinance daily price adapter (8 tests)
- `ingest/polygon_free.py` — Polygon REST scaffold w/ rate limiting (22 tests, mocked)
- `features/greeks.py` — Black-Scholes Greeks + leg aggregation (30 tests, QuantLib-validated to 1e-14)
- `features/vol_surface.py` — non-parametric implied vol surface (32 tests)

### Phase 2 (signals + risk + paper exec)

- `signals/vrp_core_spy.py` — SPY iron condor structurer + entry/exit rules (38 tests)
- `risk/envelope.py` — pre-trade risk gates (vega caps, concentration, scenario stress) (21 tests)
- `risk/kill_switch.py` — deterministic kill switch (intraday loss, cycle DD, cash buffer) (20 tests)
- `paper_exec/simulator.py` — fill simulator with slippage + commission (42 tests)

### Phase 3 (earnings satellite)

- `signals/earnings_universe.py` — Russell 1000 liquidity filter, top-50 by options ADV (25 tests)
- `signals/earnings_satellite.py` — twin-model targets + ATM straddle structurer (41 tests)
- `signals/twin_model_trainer.py` — LightGBM A/B trainer with paired-t evaluation (19 tests)

### Eval framework

- `eval/baselines.py` — B0 (cash@rf), B1 (long straddle), B2 (iron butterfly), B3 (dumb logistic) (33 tests)
- `eval/metrics.py` — DSR, PBO via CSCV, Sharpe, max DD, HMM regime stability (38 tests)
- `eval/gates.py` — the 7 pre-registered gate evaluator at base + 50% costs (32 tests)

### Freeze prep

- `metadata/paper_v1_prereg.md` — pre-registration draft v0.1 (13 sections: hypothesis, sizing, sleeves, risk, costs, gates, baselines, twin-model decision rule, audit data model, LLM roles, decision framework, out-of-scope, sign-off)

**Total tests: 431, all passing in ~2s** under `.venv` (Python 3.11.15, libomp installed for lightgbm).

## Open questions before freeze

Flagged by the prereg agent — these need CEO answers before tagging `paper_v1.0-frozen`:

1. **Risk-free rate for B0** — 3-month T-bill, Fed funds, or SOFR? Data source?
2. **HMM regime fit specs** — number of regimes, feature inputs, fitting window for Gate 6?
3. **Adversarial stress scenarios** — explicit list (vol shock, gap, liquidity withdrawal)?
4. **Annualisation convention** — 252 trading days vs 365 calendar; arithmetic vs geometric?
5. **Sample-size scope for Gate 7** — does the ≥50 events floor include walk-forward replays, or only live-cycle trades? (Implication: 90-day cycle × 50 names × ~1 earnings/quarter ≈ 25–35 events per cycle; floor likely requires augmentation.)

## Next

**Wall-clock and decision-bound** — coding is done.

- CEO sign-off on prereg + answers to the 5 open questions above
- Bump `metadata/paper_v1_prereg.md` to v1.0, lock, `git tag paper_v1.0-frozen`
- Start the 90-day frozen cycle (week 12 onward)
- Week 25: run `eval/gates.py::evaluate_full` against cycle data → promote / pivot / stop

## Active Constraint

Once the `paper_v1` frozen cycle begins, **no code changes are permitted mid-cycle**. Any change resets the cycle clock to day 0 per `metadata/strategic_lock.md` § Modification rule. During Phase 0 (now), normal development applies.

---

**Update rule:** Edit this file as Phase 0 milestones land. When `paper_v1` is tagged and the cycle starts, switch the heading to `Cycle: paper_v1 frozen, day N of 90`.
