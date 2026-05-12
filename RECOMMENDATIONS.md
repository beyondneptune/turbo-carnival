# Consolidated Recommendations — Day 1 Post-Mortem

Sources: 60-model independent review (NIM × 20 + Claude swarm × 30 + Opus × 10) +
local strategy_swarm_lab simulator results (26 specs @ 8 bps slippage).

Status updated by /loop iterations. Last touched: 2026-05-11 22:33 BST — all 14 items DONE.

## Priority queue

| # | Recommendation | Source | Status |
|---:|---|---|---|
| 1 | Queue Valk scale-down to 22 contracts for tomorrow's open | Opus xhigh #7 + 60-model | **DONE** — `scripts/queue_valk_scale_down.py` + executor + queued JSON |
| 2 | Implement vol-of-vol regime gate in `alpaca_max_hft_host.py` | Opus max #9 (top single filter) | **DONE** — `scripts/regime_filters.py` + integrated |
| 3 | Implement VIX-proxy gate in HFT host code path | NIM panel + Claude swarm | **DONE** — ATR gate in `regime_filters.py` |
| 4 | Implement news-blackout gate (300s cooldown) in HFT host | Claude swarm top filter | **DONE** — news_blackout in `regime_filters.py` |
| 5 | Patch default `slippage_bps` from 0.75 → 8.0 in HFT simulator | 19/20 NIM consensus | **DONE** — `experiments/simulate_max_hft_host.py` default 2.0→8.0 |
| 6 | Build daily-loss circuit breaker (auto-pause at −0.5% NAV/day) | Kimi K2 explicit | **DONE** — `scripts/daily_loss_circuit_breaker.py` |
| 7 | Build live slippage measurement / logging | Opus high #10 | **DONE** — `scripts/slippage_logger.py` |
| 8 | Build news → trade attribution joiner script | Opus xhigh #7 | **DONE** — `scripts/news_trade_joiner.py` |
| 9 | Author 14-DTE Valk variant spec + risk math | Opus xhigh #8 | **DONE** — `data/strategy_specs/baseline/11_valk_14dte_q22.json` |
| 10 | Document the 90-day freeze rule + acceptable DD | Sonnet 03, Opus max #9 | **DONE** — `docs/RISK_POLICY.md` |
| 11 | Twitter scraper login flow (user step) + tmux launcher | sentiment coverage | **DONE** — `scripts/launch_twitter_scraper.sh` (operator step in header) |
| 12 | Audit & critique strategy_swarm_lab simulator | Codex handoff item B | **DONE** — `docs/SIMULATOR_AUDIT.md` |
| 13 | Re-run sim across 26 specs at 12 bps (stress test) | Sonnet 09 | **DONE** — stress12 batch, best −41.47%; 8→12 bps materially worse |
| 14 | Generate daily review report generator script | self-eval skill pattern | **DONE** — `scripts/daily_review_report.py` (smoke-tested) |

## Stop conditions

- All 14 items marked DONE or explicitly tabled.
- User says stop.
- Hit 5-h session budget with > 50% utilization. Then dump remaining items as a Codex handoff.

## Phase 2 — ML research (post-freeze proposals)

| # | Recommendation | Status |
|---:|---|---|
| 15 | Scaffold `experiments/ml_lab/` with README + eval gate doc | **DONE** |
| 16 | Build regime classifier (logreg + HistGBM) on 54k bars | **DONE** — logreg 0.799 test AUC, gap +0.006 |
| 17 | Build slippage model scaffold (GBM) | **DONE** — synthetic baseline, waits for real fills |
| 18 | Build autoencoder anomaly detector | **DONE** — 0.775 test AUC, 0.001 gap, 68.7% precision |
| 19 | Document overfit-gate + post-freeze deployment plan | **DONE** — `experiments/ml_lab/RESULTS.md` |

## Source citations (anchors only)

- NIM panel synthesis: `council/reviews/20260511T191453Z/SYNTHESIS.md`
- Claude swarm synthesis (30): `council/reviews/claude_swarm_20260511T205000Z/`
- Opus tier reviews (10): `council/reviews/opus_swarm_20260511T210000Z/`
- Strategy sim ranked results: `experiments/runs/swarm_lab_20260511T204716Z/summary.csv`
- Investor PDF: `reports/day1_case_study_20260511T200925Z.pdf`
- Handoff to Codex: `HANDOFF_TO_CODEX.md`
