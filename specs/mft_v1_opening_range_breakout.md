# MFTv1 — Opening Range Breakout on Liquid US ETFs

**Status:** PREREGISTERED · DATA INGEST PENDING · NOT YET BACKTESTED OR DEPLOYED
**Date:** 2026-05-14
**Class:** Intraday momentum (mid-frequency trading, ~5-15 trades/day)
**Predecessors:** v1-v5 all failed to beat 1/N or SPY benchmarks on daily/monthly horizons. MFTv1 is a **time-horizon pivot** — different alpha class, different microstructure regime, different literature.

This is NOT HFT (retail HFT is physics-dead, see `project_full_shutdown_2026-05-13.md`). It is mid-frequency trading on 5-minute bars — the fastest viable retail timescale where the Mac mini's 80-250ms latency doesn't kill the edge.

---

## 1. Thesis (one sentence)

The first 30 minutes of trading establish a daily opening range; when intraday price breaks above the range high (or below the range low), there is a documented continuation effect through the rest of the session, sufficient to generate a tradable post-cost edge on the most liquid US ETFs.

## 2. Market / asset class / venue

- **Universe (5 most-liquid US ETFs):** SPY, QQQ, IWM, DIA, MDY
  - All have tight bid/ask spreads (1-2 cents typically)
  - All have ADV > $1B
  - All trade in `extended_hours` but strategy uses RTH only (9:30-16:00 ET)
- **Venue:** Alpaca paper account
- **Account:** AEGIS (or new dedicated MFT account)
- **Bar interval:** 5-minute
- **Settlement:** intraday only (force-flat by 15:55 ET); no overnight risk

## 3. Time horizon

**Intraday.** Entry windows 10:00-15:30 ET. Force flatten at 15:55 ET. No positions held overnight.

## 4. Method family + specific algorithm

```python
# At 10:00 ET each trading day, define the opening range:
opening_bars = bars_between("09:30 ET", "10:00 ET")  # 6 bars of 5-min each
OR_HIGH = max(b.high for b in opening_bars)
OR_LOW = min(b.low for b in opening_bars)
OR_MID = (OR_HIGH + OR_LOW) / 2
OR_WIDTH = OR_HIGH - OR_LOW

# From 10:00 ET onwards, evaluate each new 5-min bar:
for bar in bars_after_10am:
    if bar.close > OR_HIGH + STOP_BUFFER and position == FLAT:
        # Long breakout
        enter_long(qty = (POSITION_SIZE_PCT_NAV × NAV) / bar.close)
        stop_price = OR_HIGH - STOP_BUFFER  # alt: OR_LOW
        target_price = bar.close + (OR_WIDTH × TARGET_MULT)
    elif bar.close < OR_LOW - STOP_BUFFER and position == FLAT and ALLOW_SHORT:
        # Short breakout
        enter_short(qty = (POSITION_SIZE_PCT_NAV × NAV) / bar.close)
        stop_price = OR_LOW + STOP_BUFFER
        target_price = bar.close - (OR_WIDTH × TARGET_MULT)
    elif position == LONG:
        if bar.low <= stop_price: exit_at_stop()
        elif bar.high >= target_price: exit_at_target()
    elif position == SHORT:
        if bar.high >= stop_price: exit_at_stop()
        elif bar.low <= target_price: exit_at_target()

# Hard flatten at 15:55 ET regardless
if time == "15:55 ET" and position != FLAT:
    force_exit_at_market()
```

## 5. Variant universe (LOCKED for White's RC)

| Parameter | Values | Count |
|---|---|---|
| OR_DURATION_MIN (range definition window) | 15, 30, 60 | 3 |
| STOP_BUFFER_BPS (buffer outside range for entry/stop) | 0, 5, 10 | 3 |
| TARGET_MULT (target as multiple of range width) | 1.0, 1.5, 2.0 | 3 |
| POSITION_SIZE_PCT_NAV | 0.10, 0.20 | 2 |
| ALLOW_SHORT | True, False | 2 |
| MIN_RANGE_WIDTH_BPS (skip narrow-range days) | 25, 50, 100 | 3 |

Total: **3 × 3 × 3 × 2 × 2 × 3 = 324 variants**.

Phase 3 White's RC with B=2000 bootstrap iterations over these 324 variants.

## 6. Sizing rule

Per-trade notional = NAV × POSITION_SIZE_PCT_NAV. Maximum 1 simultaneous position per ETF; multiple ETFs can have concurrent positions. Maximum gross exposure = 5 × max POSITION_SIZE_PCT_NAV = 5 × 0.20 = 100% NAV.

## 7. Risk overlay (built into the strategy)

- **Force flatten 15:55 ET** — no overnight risk by construction
- **Stop loss** on every entry — defined at the opposite range edge
- **Skip narrow-range days** (MIN_RANGE_WIDTH_BPS gate) — when the open is quiet, the breakout signal is unreliable
- **Daily loss circuit** — if intraday P&L < -3% NAV, halt new entries for the day
- **No earnings days** — exclude trading on individual-ETF earnings dates (n/a for these broad ETFs)

## 8. Success hurdles (Phase 3)

| Hurdle | Threshold | Source |
|---|---|---|
| White's Reality Check p-value | < 0.05 | White 1999 |
| PBO via combinatorial CV | < 0.50 | Bailey-Lopez de Prado |
| DSR | > 0.80 | Bailey-Lopez de Prado |
| t-stat on best variant | > 3.0 | Harvey-Liu-Zhu |
| OOS Sharpe (net of cost model) | > 1.5 | ORB published Sharpes 1.5-2.0 |
| OOS Sharpe > SPY buy-hold | by ≥ 0.20 | beats market |
| OOS max DD | < 15% | retail tolerance for active trading |
| OOS Calmar | > 1.5 | risk-adjusted |
| Sharpe shrinkage train→OOS | < 50% | overfitting guard |
| Min daily trade count | > 1 (avg) | actually active |

ORB's published edge in the literature suggests these are achievable thresholds — but if they fail on real cost model + retail spreads, MFTv1 dies just like v1-v5 and we accept it.

## 9. Kill criteria (during paper shadow)

- 5 consecutive days with NO entries — strategy filter too restrictive
- 3 consecutive days losing > 1% NAV each — strategy edge gone
- Realized 30-day Sharpe outside 1σ of backtest CI
- Implementation shortfall vs Perold model > 5 bps/trade — execution problems
- VIX > 40 spot — pause strategy (high vol regimes have different microstructure)

## 10. Data needs (this is the gating step)

**Required:** 5-min OHLCV bars for SPY, QQQ, IWM, DIA, MDY over at least 3 years (~750 trading days × 78 bars/day × 5 ETFs = ~292k bars total).

**Sources:**
- **IBKR** has 5-min bars but limited depth (~6 months for most ETFs)
- **Alpaca** has free historical bars including 5-min back to early 2016 for liquid ETFs
- **Polygon free tier** has limited rate-limit but full depth

**Plan:** Use Alpaca historical bars endpoint (free with paper account) for the primary ingest. ~30 min one-shot to pull 5y of 5-min bars across 5 tickers.

Train: 2017-01-01 → 2022-12-31 (6 years, includes 2020 COVID stress)
OOS:   2023-01-01 → 2025-12-31 (3 years, the bull regime)

## 11. Capacity bound

Per-trade dollar volume targets: 20% × $100k NAV = $20k per trade. With SPY ADV ~$25B, capacity is effectively unlimited at our scale. Realistic deployment ceiling: $5M before market impact materially affects fills.

## 12. Honest priors

- **ORB published Sharpes** in academic + practitioner literature: 1.5–2.5 net of typical costs on liquid US ETFs
- **Carver-Faith / Mark Fisher** papers document the effect across multiple markets and decades
- **Realistic prior:** 50-60% probability of passing all 10 hurdles. Higher than v1-v5 (which were 20-30% prior) because:
  - Different time horizon means different statistical regime
  - Different alpha source (intraday microstructure, not factor exposure)
  - Doesn't compete against a strong bull-market benchmark on the OOS axis (intraday ORB is uncorrelated to daily SPY direction)
- **Risk:** ORB is one of the most well-known intraday effects — it may have decayed since 2010s as algos arbitraged it away. The 2017-2025 sample will reveal this.

## 13. Why this is the right pivot

The 5-of-5 failure pattern of v1-v5 establishes that:
- Retail daily/monthly mechanical strategies don't beat passive benchmarks
- The corpus we tested was the daily/monthly subset of trading literature

MFTv1 deliberately tests a **different subset of the corpus** (intraday microstructure). If MFTv1 also fails, we'll have evidence across 2 alpha classes that retail mechanical alpha is dead — which is itself a strong conclusion.

This is not "v6 from the same corpus." It's a deliberate change of problem class.

## 14. Phase progression

- ☑ Phase 1 — this document
- ☐ Phase 2 — minute-bar ingest (Alpaca historical, ~30 min wall-clock)
- ☐ Phase 3 — 324-variant backtest with White RC + PBO + DSR + dual benchmarks
- ☐ Phase 4 — walk-forward OOS (purged k-fold on the train period)
- ☐ Phase 5 — overlay verification (the strategy already includes its own overlays)
- ☐ Phase 6 — paper shadow host on AEGIS (intraday cron, runs every 5-min from 09:30 ET to 16:00 ET)
- ☐ Phase 7 — 30-day shadow per the standard protocol
- ☐ Phase 8-10 — standard funnel

## 15. Citation block (to be expanded)

- **Carver, R. & Faith, C.** — *Mechanical Trading Systems* / *Way of the Turtle* — original ORB systems
- **Crabel, T.** (1990) — *Day Trading with Short Term Price Patterns and Opening Range Breakout* — canonical retail reference
- **Bouchaud et al.** — limit order book + intraday microstructure literature (already in corpus)
- **Bailey-Lopez de Prado**, **Harvey-Liu-Zhu**, **White** — multi-testing correction (already in our pipeline)

## 16. Sign-off

Operator authorized 2026-05-14 — "ok starting building the new one" after 5/5 candidate failures + literature pivot recommendation.

Committed at git hash _to be appended_.
