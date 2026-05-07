# Phase Status

**Phase:** All code phases complete + prereg v1.0 amended for held-out OOS evidence — awaiting CEO `git tag paper_v1.0-frozen`
**Machine:** Mac mini (production target; Phase 0 development allowed)
**Frozen cycle:** Not started — code changes are permitted
**Last updated:** 2026-05-07

---

## Completed

- Project skeleton (directory layout per `README.md`)
- `metadata/strategic_lock.md` — locked hybrid VRP core + earnings satellite strategy
- `metadata/capital_policy.md` — $10k paper notional, sleeve allocation, kill switch limits
- `sql/schema.sql` — `runs`, `gate_decisions`, `agents_log`, `trades`, `trade_legs`
- `dashboard/app.py` — read-only Streamlit dashboard skeleton (overview, runs, gate decisions, trades, agent activity)
- `scripts/run_phase0_demo.py` — synthetic SPY-momentum demo seed; **append-only by default** (auto-generated unique `run_id`; `--run-id <id>` and `--reset-demo` flags); populates all 5 tables for one demo run
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

- `metadata/paper_v1_prereg.md` — **pre-registration v1.0** (15 sections: Annualisation Conventions, Adversarial Stress Scenarios, Gate 6 HMM specs, B0 risk-free rate snapshot, and Historical Data / Simulation Use Policy with Year 1 train / Year 2 one-shot OOS evidence)

**Total tests: 431, all passing in ~2s** under `.venv` (Python 3.11.15, libomp installed for lightgbm).

## Locked at v1.0 (CEO has the prereg)

| # | Decision | Locked value |
|---|---|---|
| 1 | B0 risk-free rate | FRED `DGS3MO` snapshot 2026-05-05 = **3.69%**, hardcoded as `eval/baselines.py::RISK_FREE_RATE_PAPER_V1`, no refetch in cycle |
| 2 | Gate 6 HMM specs | 2-state Gaussian HMM, daily SPY returns, held-out Year 2 evidence-window fit, `random_state=42` |
| 3 | Adversarial stress scenarios | 4 pre-registered: −5%/+10vol, **−10%/+30vol (adversarial)**, +5%/+5vol, flat/−5vol |
| 4 | Annualisation conventions | 252 trading days for strategy/Sharpe; 252/5 events/year for twin-model; 365 calendar geometric for B0 cash |
| 5 | Gate 7 sample-size scope | Held-out Year 2 satellite trades from the frozen evidence run only — no Year 1, synthetic, or post-result replay padding |

## Next

**Wall-clock and decision-bound** — coding is done.

- CEO `git tag paper_v1.0-frozen && git push origin paper_v1.0-frozen` (your hand on the tag)
- After tag: run the held-out Year 2 evaluator exactly once; Year 1 remains training/calibration only
- Optional: run a 90-day forward paper confirmation with the same frozen code/constants; report separately from the official Year 2 verdict
- Operational prep still needed: Polygon Free key in `.env`, evidence/backtest orchestrator, daily forward orchestrator (`scripts/run_daily_paper.py`), cron / launchd schedule
- Weekly LLM review is read-only until post-cycle: summaries/postmortems only, no model/code/threshold changes inside `paper_v1`

## Active Constraint

Once the `paper_v1` frozen cycle begins, **no code changes are permitted mid-cycle**. Any change resets the cycle clock to day 0 per `metadata/strategic_lock.md` § Modification rule. During Phase 0 (now), normal development applies.

---

**Update rule:** Edit this file as milestones land. When `paper_v1.0-frozen` is tagged, switch the heading to `Cycle: paper_v1 frozen — Year 2 OOS evidence pending/run complete`; if the optional forward confirmation starts, track day N of 90 separately.
