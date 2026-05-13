# Global Expansion Roadmap
Date: 2026-05-13
Status: Draft v1
Owner: Operator (UK, retail/small-fund)

This document is a contingent, phase-gated plan for taking the four research-validated edges from paper-shadow to a global, multi-strategy book. Every action below is conditioned on prior-phase evidence. No phase is assumed; each must be earned.

---

## Current state (Day 0)

Four strategies deployed in paper shadow as of 2026-05-12 evening:

| iter | Strategy | Broker / Venue | Active window (BST) | Edge basis | Status |
|---|---|---|---|---|---|
| 3 | BTC perp funding carry | Kraken (paper) | 24/7 (signals sparse) | Soft pass; raw Sharpe 0.62, +8% APR; needs execution-cost realism | Paper shadow live, low signal frequency |
| 5 | SVXY sweet spot (thr=1.0 / 25% / no CB) | Alpaca AEGIS | 20:30-21:00 (US close prints) | Walk-forward 2019-2026 Sharpe 0.96, +7.5% / -8.4% DD | Paper shadow live |
| 6 | Nikkei gap continuation (EWJ proxy) | Alpaca AEGIS | 14:35 (US open) | Daily proxy Sharpe 2.51, +29% / -5.4% DD; **needs real 5-min validation on NKD** | Paper shadow live on proxy only |
| 18 | US-HK ADR convergence arb (5 pairs, simulated HK leg) | Alpaca AEGIS | US session | Cleanup Sharpe ~1.0-1.5 real, +66 bps/trade, 60-69% win rate after auto_adjust + ratio + ex-div fixes | Paper shadow live; HK leg is **simulated**, not executed |

Plus:

- **Rust HFT engine (FALCON)**: prototype, not yet launched. Intended for sub-second execution and order-book microstructure work. Out of scope for Phases 1-3.

**Combined portfolio (verified backtest):** 2025 OOS Sharpe ~3.0 adjusted, near-zero pairwise correlations, ~100% positive months. This is the central thesis — that the **portfolio** is the edge, not any single leg.

**Honest caveat:** all four strategies have backtest support but **none have completed a 30-day paper shadow**. The 14%/month target depends on (a) the edges surviving live, (b) keeping correlations near-zero, and (c) applying 3-5× leverage to the portfolio. Each of these is a separately falsifiable assumption.

---

## Time-coverage gap analysis

24-hour map of when at least one deployed strategy can fire a signal:

| BST window | Markets open | Strategies currently covering | Coverage |
|---|---|---|---|
| 00:00-07:00 | ASX (partial), Tokyo, HK | iter_3 only (rare) | **WEAK** |
| 07:00-08:00 | Tokyo close, HK still open, LSE opens at 08:00 | iter_3 only | **WEAK** |
| 08:00-14:30 | LSE, Eurex, HK morning | iter_3 only | **GAP** |
| 14:30-15:30 | US open, LSE still open | iter_5, iter_6, iter_18 staged | strong |
| 15:30-21:00 | US session | iter_5, iter_6, iter_18 | strong |
| 21:00-22:00 | US after-hours only | nothing | **GAP** |
| 22:00-00:00 | ASX opens 23:00-01:00 BST | nothing | **GAP** |

**Hours covered with non-rare signal flow today: ~6.5 of 24 (~27%).** This is the gap the roadmap addresses — but only after Phase 1 proves at least some of the current four edges are real.

---

## Phase 1 — Validation (Day 0 to Day 30)

**Goal**: Determine which of the 4 deployed strategies have real edge in live execution conditions.

**Actions**:
- Run all 4 strategies in paper shadow against the existing broker stack (Alpaca AEGIS + Kraken).
- Daily monitoring via the dashboard; broker-truth reconcile each evening (per iter_5 §8 cadence).
- Track per-strategy: realized P&L, hit rate, average slippage vs backtest, max drawdown, fill quality, broker-truth divergence count.
- No code, threshold, or risk-gate changes mid-run (90-day cycle freeze rule applies to live config).

**Exit criteria — applied independently to each of the 4 strategies**:

| Outcome | Criteria (any of) |
|---|---|
| **PROMOTE** to Phase 2 | All of: 30-day realized Sharpe ≥ 1.0; broker-truth match (zero unreconciled fills); peak-to-trough DD < 5%; ≥ 10 closed trades for statistical signal |
| **HOLD** (extend shadow) | Sharpe 0.5-1.0 AND no broker-truth failures AND DD < 5% AND trade count too low to decide |
| **KILL** | Any of: Sharpe < 0.5 over 30 days; peak-to-trough DD > 5%; broker-truth mismatch (any unreconciled fill); evidence of survivorship/lookahead/data bug in backtest reproduction |

**Realistic base-rate expectation**: prior literature on retail systematic strategies and the project's own iter_1 / iter_4 / iter_7 / iter_8 / iter_10 / iter_13 / iter_14 / iter_16 / iter_20 / iter_23 failure pattern (12 of 16 tested edges failed) suggests **~50% of the current 4 will not survive 30 days unchanged**. Plan capacity assumes 2 survivors, not 4.

**Cost**: $0 incremental (paper money, existing free data feeds, existing infrastructure).

**Decision point at Day 30**: which strategies move to Phase 2. If **zero strategies promote**, the roadmap pauses — back to research, no Phase 2 spend. If **≥1 promotes**, Phase 2 begins for the survivors only.

---

## Phase 2 — Migration to IBKR with proven strategies (Day 30 to Day 60)

**Goal**: For strategies that survived Phase 1, upgrade execution quality from Alpaca proxies to direct IBKR access, and re-validate the upgraded versions.

**Actions** (only for Phase 1 survivors):
- Subscribe to the IBKR data feeds matched to surviving strategies:
  - **HKEX cash + futures**: ~$25-30/mo, unlocks iter_18 with a real HK leg instead of simulated parity.
  - **CME real-time**: ~$10/mo, unlocks iter_6 on NKD futures instead of EWJ ETF proxy.
- Wire IBKR DUK295771 (£100k GBP paper NAV) into the AEGIS execution layer.
- Migrate validated strategies from Alpaca proxies → direct IBKR execution path.
- Run a second 30-day paper shadow on the **upgraded** versions. The first shadow validated the signal; the second validates the upgraded execution.
- Continue broker-truth reconcile cadence; specifically track slippage delta between Alpaca proxy and IBKR direct execution.

**Exit criteria for graduating to tiny live**:
- 30-day IBKR paper Sharpe within ±0.3 of the Alpaca paper Sharpe (i.e. the upgrade didn't break the edge)
- Broker-truth match across both venues for the parallel run window
- Slippage delta within ±5 bps of the modelled assumption
- Operator has rehearsed the kill-switch path on IBKR (manual flat-to-cash, hardware-side stop)

**Cost**: ~$35-40/mo data subscriptions. Tiny live capital deployment at end of phase: ~$10-20k per surviving strategy, NOT leveraged.

**Decision point at Day 60**: which upgraded strategies graduate to tiny live capital. Failure mode here: strategy survives Phase 1 on Alpaca proxy but breaks on IBKR direct (proxy slippage was hiding real costs). If this happens, return to research, do not deploy real capital.

---

## Phase 3 — Global research expansion (Day 60 to Day 120)

**Goal**: Research and validate NEW strategies for currently-uncovered time zones. Coverage is the goal, but **validation comes first** — no deployment of a coverage-motivated strategy that has not passed its own 30-day paper shadow.

**New strategy candidates to research** (priority-ordered):

| iter | Description | Time zone covered (BST) | Data sub needed | Notes |
|---|---|---|---|---|
| 26 | UK FTSE 100 mean reversion via EWU (scaffolded) | 08:00-16:30 | LSE: ~$10-15/mo (or free EOD) | Already scaffolded; closest to ready |
| 27 | DAX 1-month narrow TSMOM via DAX ETF | 08:00-16:30 | None new (free EOD) | Original DAX TSMOM failed (post-2015 momentum decay); narrower variant is a hypothesis, not a fact |
| 28 | ASX 200 carry overlay (AUD/JPY carry × ASX trend) | 23:00-05:00 | ASX EOD (free via Yahoo) + OANDA AUD/JPY | Multi-leg; carry leg is the proven premium, ASX trend is the rotation gate |
| 29 | Tokyo direct gap continuation on NKD futures | 23:00-05:30 | CME real-time (already paid in Phase 2 if iter_6 survived) | This is the **upgrade** of iter_6, not a separate strategy. Promotes iter_6 from proxy → real |
| 30 | HK direct ADR with real-time HKEX quotes | 02:00-08:00 | HKEX real-time (already paid in Phase 2 if iter_18 survived) | Upgrade of iter_18 |
| 31 | Eurex DAX/FDAX intraday OFI | 08:00-16:30 | Eurex L1 / DXFEED: ~$20-30/mo | Microstructure premium analogous to iter_2 ES OFI; only attempt after iter_2 itself is researched |

**Process per candidate**: backtest → 30-day paper shadow → tiny live → scale. The same Phase 1 / Phase 2 gates apply to each new candidate independently.

**Realistic base-rate expectation**: of these 6 new candidates, prior failure base rate of ~75% (12 of 16 in the May 2026 session) suggests **1-2 will graduate**. Plan around that, not 6.

**Cost**: +$30-50/mo additional data for the LSE / Eurex / ASX feeds. **Cumulative monthly burn**: ~$80/mo by end of Phase 3.

**Decision point at Day 120**: which subset of {survivors from Phases 1-2} + {new survivors from Phase 3} forms the Phase 4 portfolio. Minimum bar to enter Phase 4: ≥4 uncorrelated strategies, combined paper-shadow Sharpe ≥1.5 over the 30-day overlap window.

---

## Phase 4 — Full 24-hour coverage + leverage (Day 120+)

**Goal**: 6-8 validated strategies covering all time zones, with prudent leverage applied to the portfolio (not to individual legs).

**Configuration (target, contingent on Phases 1-3)**:
- 6-8 strategies live across 4 brokers (Alpaca, Kraken, IBKR, OANDA-if-FX-survives)
- Capital allocation: ~$25-30k per strategy on $200k+ NAV baseline
- Leverage: 3-5× **on the validated multi-strategy portfolio**, never on a single leg
- Drawdown discipline: deleverage if rolling 30-day portfolio DD > 12%; flat-to-SGOV if > 16%
- Target: 9-14% monthly mean return at 10-15% max drawdown

**Honest framing of the 14%/month number**:
- The 14%/month figure is **the upper end of a leverage-conditional range**, derived from the verified backtest portfolio Sharpe (~2.5-3.0 adjusted) × 4-5× leverage. It is not a forecast.
- Removing leverage drops the target to ~3-4%/month at 1×, ~9-11% at 3×.
- Removing strategies (Phase 1/3 attrition) reduces the portfolio Sharpe and shifts the leverage-vs-return curve down.
- The 14%/month is **conditional on all four 2026-05-12 verified edges surviving plus ≥2 new edges from Phase 3**. If only 2 survive, the realistic Phase 4 target is ~6-8%/month at 3× leverage.

---

## Broker / Data cost progression

| Phase | Brokers active | Monthly data cost | Cumulative capital deployed |
|---|---|---|---|
| Phase 1 (Day 0-30) | Alpaca + Kraken (paper) | $0 | $0 (paper only) |
| Phase 2 (Day 30-60) | + IBKR with HKEX + CME | ~$40/mo | $20-40k (tiny live on ≤2 strategies) |
| Phase 3 (Day 60-120) | + IBKR LSE/Eurex/ASX feeds | ~$80/mo | $100k (estimated 4-5 strategies if attrition matches expectation) |
| Phase 4 (Day 120+) | Full stack incl. OANDA if FX survives | ~$80-150/mo | $200-500k (6-8 strategies, leveraged) |

Costs assume retail-tier subs. Operator does not need any single sub > $50/mo through Phase 3.

---

## Risk gates at each phase boundary

These are the **quantitative kill criteria** at each gate. They are intentionally tight; the cost of a false promotion (real money on a dead edge) is much greater than the cost of a false kill (waiting another month).

**Gate 1 → 2 (Day 30)**: per strategy, must clear ALL of:
- 30-day realized Sharpe ≥ 1.0
- Peak-to-trough DD < 5%
- Zero broker-truth reconciliation failures
- ≥10 closed round-trip trades
- Slippage within ±10 bps of backtest assumption

**Gate 2 → 3 (Day 60)**: per strategy, must clear ALL of:
- 30-day IBKR paper Sharpe within ±0.3 of Phase 1 Alpaca paper Sharpe
- Slippage delta IBKR vs Alpaca within ±5 bps
- Operator-rehearsed kill switch on IBKR (timed: < 60 seconds to flat)
- Tiny live result (if deployed) shows zero unreconciled fills in first 7 calendar days

**Gate 3 → 4 (Day 120)**: portfolio level, must clear ALL of:
- ≥4 strategies live in paper shadow simultaneously
- Pairwise correlation ≤ 0.3 across all strategy pairs
- Combined paper-shadow Sharpe ≥ 1.5 over the 30-day overlap window
- No single strategy contributing > 50% of portfolio P&L (i.e. not a single-edge book in disguise)

**Portfolio-level kill (any phase)**: rolling 30-day portfolio DD > 16% → flat to SGOV, restart from Phase 1 evidence review.

---

## What this roadmap is NOT

- **Not retail HFT.** The Rust FALCON engine is for execution quality and microstructure research, not for competing with co-located prop shops on speed. We can't and won't.
- **Not deploying multiple unvalidated strategies simultaneously.** The May 2026 session diagnosed this as the Day-0 mistake: shipping 4 strategies in parallel without 30-day shadows on each. The roadmap explicitly inverts that — one shadow per strategy per phase, sequential validation, no batched promotions.
- **Not chasing time-zone coverage by deploying random new strategies.** Every Phase 3 candidate must clear the Phase 1/2 gates on its own merits. If iter_26 LSE fails its 30-day shadow, LSE coverage stays at zero. Empty hours are acceptable; deployed-without-evidence hours are not.
- **Not real-money before 30+ day paper shadow per strategy.** Tiny live capital is the Day-60 reward for a strategy that has cleared Gates 1 and 2. Scaled live is the Day-120+ reward for clearing Gate 3. Leverage is conditional on the Phase 4 portfolio passing all three.
- **Not a yield forecast.** The 14%/month figure in the verdict is a **leverage-conditional ceiling** on a backtest that may not survive live, not a target the operator should expect to hit. Plan for the 3-4%/month un-leveraged base case; treat anything more as bonus.
- **Not a fixed timeline.** "Day 30" means "30 days of evidence accumulated," not "30 calendar days." If signal flow is too sparse (likely for iter_3 BTC funding), the phase clock extends until the evidence bar is cleared. Don't promote on insufficient data.

---

## Open questions

These are decisions to make later, with evidence, not now.

- **iter_25 META validation framework**: activate at Phase 2 (to formalize the Gate 1→2 evaluation) or wait until Phase 3 (so it spans multiple strategies)?
- **Single-account vs multi-account organization**: keep the four research-broker accounts (Alpaca + Kraken + IBKR + OANDA) or consolidate into IBKR-only once live? Trade-off: simpler ops vs strategy-isolated risk.
- **When does the operator graduate from paper to real?** Current plan: Gate 1→2 successful + Gate 2→3 successful + operator-stated readiness. But "readiness" is the operator's call, not the spec's.
- **Does iter_3 BTC funding warrant its own broker (Deribit, OKX) beyond Kraken?** Kraken funding rates are the smallest of the major venues; the edge might require a venue with deeper basis.
- **At what NAV does data-subscription cost become a fixed-cost drag worth optimizing?** At $200k NAV, $150/mo is 0.9% APR drag. At $1M NAV, it's 0.18%. The cost-to-NAV inflection probably sits around $300-500k.
- **At what AUM does iter_23 (global multi-factor risk premia) become viable?** It strictly contains iter_4 G10 carry as a sub-component; the shortlist flagged $5M+ as the data-cost amortization point. Not a near-term decision.
- **Does the Rust FALCON engine get prioritized at Phase 3 or Phase 4?** Argues for Phase 3: improved execution quality on existing legs (slippage compression). Argues for Phase 4: it's net new infra that needs its own validation cycle.

---

## Appendix: Time-coverage matrix at end of each phase

Hours of the BST 24-hour clock with **at least one validated strategy actively producing signals**. "Validated" = has cleared at least Gate 1.

| BST window | Markets open | End of Phase 1 (Day 30) | End of Phase 2 (Day 60) | End of Phase 3 (Day 120) | End of Phase 4 (Day 120+) |
|---|---|---|---|---|---|
| 00:00-02:00 | Tokyo, HK | iter_3 sparse | iter_3 sparse | iter_29 (NKD direct, if survived) + iter_3 | iter_29 + iter_3 |
| 02:00-07:00 | HK, ASX | iter_3 sparse | iter_3 sparse | iter_30 (HK direct) + iter_28 (ASX) if survived | iter_30 + iter_28 + iter_3 |
| 07:00-08:00 | LSE opens 08:00 | iter_3 sparse | iter_3 sparse | iter_3 sparse | iter_3 sparse |
| 08:00-14:30 | LSE, Eurex, HK morning | iter_3 sparse | iter_3 sparse | iter_26 (FTSE) + iter_31 (Eurex OFI) if survived | iter_26 + iter_31 + iter_27 (DAX narrow) |
| 14:30-15:30 | US open + LSE | iter_5/6/18 (if survived) | iter_5/6/18 + IBKR upgraded | full Phase 2 + LSE/Eurex overlap | full Phase 4 portfolio |
| 15:30-20:30 | US session | iter_5/6/18 (if survived) | iter_5/6/18 IBKR | iter_5/6/18 + new candidates | full Phase 4 portfolio |
| 20:30-21:00 | US close prints | iter_5 (if survived) | iter_5 | iter_5 + late entrants | iter_5 + late entrants |
| 21:00-22:00 | post-close | nothing | nothing | nothing (acceptable gap) | nothing (acceptable gap) |
| 22:00-00:00 | ASX opens 23:00 | nothing | nothing | iter_28 (ASX) if survived | iter_28 |

**Best case at Day 120**: ~18-20 of 24 BST hours covered. **Realistic case**: ~12-15 hours, given expected attrition. **Worst case**: Phase 1 produces 0-1 survivors and the roadmap pauses for research; coverage stays at Day-0 level.

---

## Sources of record

- `/Users/weeoto/turbo-carnival/research/results/FINAL_SESSION_VERDICT_20260512.md` — the 22-backtest session verdict and the 4 surviving edges
- `/Users/weeoto/turbo-carnival/research/candidates/RANKED_SHORTLIST.md` — 25-candidate scoring rubric and shortlist rationale
- `/Users/weeoto/turbo-carnival/specs/global_market_system_v1.md` — global market system spec
- `/Users/weeoto/turbo-carnival/CLAUDE.md` — project hard-rules: 90-day cycle freeze, no synthetic evidence, no mid-run config changes
