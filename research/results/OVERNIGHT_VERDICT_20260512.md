# Overnight Backtest Verdict — 2026-05-12

7 candidate backtests run in parallel between 18:55-20:30 ET on 2026-05-12.
All used free data (yfinance, ccxt Binance, FRED). All produced verifiable `summary.json` files in `research/results/`.

---

## Final tally

| # | Candidate | Verdict | Key numbers | Honest caveat |
|---|---|---|---|---|
| 1 | **US smallcap overnight meanrev** | **PASS** ★ | 6,154 trades, win 53.2%, mean +28.7 bps, OOS Sharpe 1.92 | Survivorship-biased universe (current IWM applied retroactively); real Sharpe likely 0.8-1.4 on clean data |
| 3 | BTC perp funding | PASS (soft) | 11 positions, net carry 7.95% APR, Sharpe proxy 0.62 | One early position contributed half the gain; regime-dependent; below spec's 1.5 promote bar |
| 4 | G10 FX carry | **FAIL** | OOS Sharpe 0.04, ann -0.04% | Post-GFC carry premium decayed; USD rally + 2015/2020/2024 crashes ate it |
| 5 | **SVXY VIX rotation** | **FAIL** ⚠ | OOS Sharpe 0.09, ann 0.38% | **Prior "+17.1% / Sharpe 0.66" reproduces ONLY with same-day lookahead bias.** SVXY changed from -1× to -0.5× in Feb 2018 — any pre-2018 backtest is invalid |
| 7 | DAX TSMOM | FAIL | OOS Sharpe 0.10 (IS was 0.51) | Documented post-2015 multi-asset TSMOM decay |
| 8 | Commodity ETF cointegration | FAIL | 0 trades fired | None of 4 pairs cleared p≤0.05 cointegration gate on full history; GLD/SLV broke after 2021 silver squeeze |
| 10 | Gold/Silver ratio | FAIL | 62 OOS trades, mean +4 bps vs +200 bps gate, win 35.5% vs 50% gate | 2022+ central-bank gold buying regime; cooldown rules from spec were not applied in this run |

**Score: 1 strong PASS + 1 soft PASS + 5 FAIL.**

---

## The headline finding

**The strategy that was supposed to anchor tomorrow's live deployment (iter_5 SVXY rotation) is NOT validated.**

The prior +17.1% / Sharpe 0.66 number that this whole project has been operating against does not reproduce under honest no-lookahead implementation. The agent ran sensitivities — sizing, period selection, looser circuit breakers — and could not produce anything above Sharpe 0.21 on legitimate framing. The +17.1% only emerges when same-day signal earns same-day return (lookahead).

Additionally, SVXY changed from -1× to -0.5× leverage in February 2018 (after the XIV blowup terminated the -1× products). Any backtest using pre-2018 returns and extrapolating to current SVXY is structurally invalid. Spec §9 of `iter_5_vix_term_structure_rotation.md` explicitly warned about this.

**Implication for tomorrow:** Do NOT deploy iter_5 SVXY rotation live. The "validated baseline" claim cannot be defended.

---

## What honestly survives

**iter_1 US smallcap overnight mean-reversion** is the strongest candidate of the night, but with one major caveat: the yfinance universe used (current IWM holdings applied retroactively) carries survivorship bias. Headline Sharpe 1.92 is inflated; real Sharpe on point-in-time data is likely 0.8-1.4 (consistent with published academic results for this anomaly).

Even at 16 bps stress slippage (2× the spec's budget), iter_1 still passes the +5 bps mean gate at 12 bps. The signal is robust to cost assumptions; it is not robust to survivorship-bias correction without re-validation.

**iter_3 BTC perp funding** is a soft pass at 7.95% net APR. Above the 6% gate but below the spec's 1.5 Sharpe promote bar (proxy Sharpe came in at 0.62). Heavily regime-dependent — one early 2021 position contributed half the lifetime gain.

---

## Recommended actions before market open tomorrow

1. **Flatten all 6 accounts to cash/SGOV.** This part of the plan stands.
2. **DO NOT deploy iter_5 SVXY rotation live.** This is the correction to the morning's planned deployment.
3. **All 6 accounts park in SGOV (or equivalent T-bill ETF) at market open.** Earning ~5% APY while we figure out what's actually live-ready.
4. **iter_1 paper shadow can start Tuesday (next business day after audit)** under either:
   - Accept survivorship-biased universe as "good enough for shadow with no capital at risk" — paper-only, document the caveat
   - Wait for Polygon/CRSP re-validation (a few days of additional work) before any deploy
5. **Locate the original SVXY rotation backtest in the repo.** Audit whether the "+17.1% / 0.66" came from:
   - Pre-2018 -1× SVXY/XIV data
   - Same-day lookahead bias in the execution model
   - A different threshold or product than what's in iter_5 spec
6. **Defer iter_3 BTC funding paper shadow** until we set up safe crypto exchange paper access (CCXT testnet or Bybit testnet) — not the existing Alpaca paper accounts.

---

## Files and paths

All backtest artifacts are in `/Users/weeoto/turbo-carnival/research/`:

```
candidates/iter_1_us_smallcap_overnight_meanrev.md        ← spec
scripts/iter_1_backtest.py                                ← code that ran
data/cache/r2000_daily.parquet                            ← cached data
results/iter_1_summary.json                               ← machine-readable verdict
results/iter_1_trades.csv                                 ← every trade
logs/iter_1_backtest.log                                  ← run log
```

Same pattern for iter_3, 4, 5, 7, 8, 10.

---

## Strategies NOT tested tonight (deferred for paid data / infra)

- iter_2 (ES futures 1-min OFI) — needs minute bars; Polygon S3 plan tier was 403
- iter_6 (Nikkei futures) — needs CME data subscription
- iter_9 (NIFTY direct) — needs Zerodha or Indian broker
- iter_11-14, 17, 20, 21, 22, 24 — LLM / ML / options chain — paid data or training infra required
- iter_15 (RL execution) — infra-heavy, requires sim env
- iter_16, 18, 19 (combinatorial / cross-listing / on-chain) — needs additional vendor APIs
- iter_23 (50+ instrument multi-factor) — feasible but a multi-day pull
- iter_25 (META framework) — software infrastructure, not a backtest

These remain available in the candidate pool. The 6 not tested in this batch are next in queue once the data access is set up.

---

## What this overnight run cost (for tracking)

7 sub-agents in parallel, ~3-18 min each, total wall-time ~30 minutes. Token usage: ~325k across all agents combined. Free data only. Real Python execution, not design generation.

What it produced: 7 verifiable PASS/FAIL stamps, the invalidation of an assumed-validated strategy before it cost real money, and one credible candidate (iter_1) for tomorrow's paper-shadow pipeline.
