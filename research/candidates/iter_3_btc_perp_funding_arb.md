# Iteration 3 — BTC Perpetual Funding-Rate Cash-and-Carry Arbitrage

## 1. Thesis (one sentence)

When the annualized funding rate on BTC perpetual swaps exceeds 10%, simultaneously shorting one BTC of perpetual on a deep-liquidity venue and longing one BTC of spot on a hedge venue isolates the funding payment as a delta-neutral carry that historically (2020-2025) yields 8-15% annualized with drawdowns dominated by basis dislocations rather than directional BTC moves — falsifiable if, over 200 forward funding cycles, the realized net carry minus transaction costs is < 4% annualized or the Sharpe of the net-of-fees stream is < 1.0.

## 2. Market / asset class / venue

- **Instruments**:
  - Short leg: `BTCUSDT` perpetual swap (linear, USDT-margined) on Binance Futures (primary execution venue)
  - Long leg: `BTC/USDT` spot on OKX (hedge / inventory venue — different corporate entity, different jurisdiction, different hot-wallet keys)
  - Cross-venue collateral: USDT held on both venues; small BTC float kept on a third venue (Bybit) as failover inventory only
- **Primary exchange**: Binance Futures for the short perp (deepest BTC perpetual book globally, ~$15-30B daily volume on BTCUSDT perp)
- **Hedge exchange**: OKX spot (independent counterparty, regulated under VARA in Dubai, separate jurisdiction from Binance)
- **Broker / SDK**: `ccxt` (Python, unified multi-exchange interface — `ccxt.binance({...})` and `ccxt.okx({...})`)
- **Settlement**: Perpetual leg settles funding every 8 hours (00:00, 08:00, 16:00 UTC on Binance); spot leg is T+0 instant. Both legs USDT-margined / USDT-quoted, so PnL is in USDT.
- **Margin**: Perp leg uses cross-margin on Binance Futures wallet, target effective leverage **2.5x** on the perp short (i.e. 40% initial margin), well below the 125x max — leaves a wide buffer to liquidation.
- **Tax regime**: Treated as Section 1256-style 60/40 is NOT applicable to crypto perpetuals (US — they are property gains under IRS Notice 2014-21). Funding payments received are ordinary income. Account is owned by a Cayman / BVI structure (user instruction-dependent; default to LLC ordinary treatment if entity is US).

## 3. Time horizon

**8-hour funding window** — discrete events at 00:00 / 08:00 / 16:00 UTC. Position is held across exactly N funding rolls, where N is determined by whether the funding regime remains above threshold at each window.

This is **not a daily-bar strategy and not intraday HFT** — it is a **3-events-per-day discrete carry harvest**. The relevant clock is the funding clock, not the wall clock. There is no daily close in the equity sense: crypto trades 24/7/365, holidays included. The "session" boundary is the funding payment timestamp.

Concretely:
- Decision points: 3 per day, ~15 minutes before each funding settlement (compute expected funding from the funding-rate prediction API + recent premium index).
- Position duration: minimum 1 funding cycle (8h), typical 3-21 days, maximum bounded by the unwind threshold logic in §7.
- Re-evaluation: every funding window the position either rolls (funding stays >= unwind threshold) or unwinds.

## 4. Method family and specific algorithm

**Family**: Rules-based cash-and-carry arbitrage with a threshold-driven entry/exit. Not ML. Not a forecaster. The edge is structural and observable in real time (the next funding rate is published 15 min before payment); no model is needed to "predict" it — the carry is mechanical once you decide to wear the basis risk.

**Algorithm** (deterministic):

```
ENTRY:
  if annualized_funding_8h_predicted > 10%  (i.e. realized 8h rate > 0.0091%)
     and 5-window EMA of annualized funding > 8%
     and spot-perp basis premium < 0.30% (i.e. perp not already crashing into spot)
     and free margin on perp venue > 2x required initial margin
  then:
     short BTCUSDT perp 1.0 BTC (post-only, GTC, 2 bps inside best bid)
     long BTC/USDT spot 1.0 BTC on OKX (post-only, GTC, 2 bps inside best ask)
     log entry basis and size

ROLL (every 8h funding window):
  hold position; collect funding payment on perp short leg

UNWIND:
  if annualized_funding < 4% (sustained over 2 windows)
     OR annualized_funding < 0 for any single window
     OR basis (perp - spot) widens beyond +/- 1.5%
     OR exchange health signal (withdrawal latency, status page, weekly insolvency screen) trips
  then:
     buy-to-close perp (marketable-limit, IOC, 5 bps slip budget)
     sell spot on hedge venue (marketable-limit, IOC, 5 bps slip budget)
     log realized carry vs expected
```

**Why rules, not ML**: Funding rate is *published* before each window. There is nothing to "predict" — you are deciding whether the *known* carry compensates for the basis-blow-out tail. ML on a known coupon is overfitting noise. The model risk is in the unwind logic (how wide do you let basis go before bailing), not in the entry signal. Lesson 2 (binary gates, not confidence rankers) is honored: this is a binary gate by construction.

## 5. Data requirements

| Feed | Source | Purpose | Storage | Cost |
|---|---|---|---|---|
| BTCUSDT perp funding history (hourly granularity, all 3 windows) | Binance public REST `/fapi/v1/fundingRate` | Backtest + live signal | ~5 MB / year | $0 |
| BTC/USDT spot OHLCV (1-min) + tick trades | Binance + OKX public REST / WS | Basis calculation, fill simulation | ~2 GB / year (1-min); ~30 GB / year (ticks) | $0 |
| Perp mark price + premium index (8h resampled) | Binance `/fapi/v1/premiumIndex` | Pre-funding prediction (Binance publishes predicted rate ~15min ahead) | ~5 MB / year | $0 |
| L2 order book snapshots (top 20 levels, 1Hz) on both venues for execution side | ccxt watch_order_book WS | Slippage estimation, depth gating | ~50 GB / year if persisted | $0 |
| Exchange status / withdrawal latency (scraped from each venue's status page hourly) | Custom scraper | Counterparty health early-warning | <1 MB / year | $0 |

- **Total storage**: ~80 GB / year if we keep ticks; ~5 GB / year if we keep only 1-min + funding + premium. Recommended: 1-min + funding + premium for production; ticks only for slippage research.
- **Total monthly cost**: **$0** (all public Binance / OKX endpoints). The "expensive" data is in time, not dollars.
- **Latency requirement**: Round-trip < 500ms to each venue is sufficient. We are NOT HFT. Co-location is unnecessary. Standard VPS in AWS ap-northeast-1 (Tokyo) or eu-central-1 (Frankfurt) is enough — those have <50ms to Binance's matching engine in Tokyo and OKX's in Hong Kong respectively. We deliberately use one VPS per venue to limit blast radius.

## 6. Validation plan

- **Historical window**: Jan 2020 (Binance funding history begins reliably) through Apr 2026 = ~6 years × 3 funding rolls/day × 365 = **~6,570 funding-cycle observations** available.
- **Walk-forward design**:
  - Train window: 12 months
  - Test window: 3 months
  - Step: 1 month (rolling)
  - Yields ~63 walk-forward folds; each test fold contains ~270 funding cycles.
- **Out-of-sample requirement**: **minimum 200 OOS funding-cycle observations** before any verdict is rendered (one fold gives this with margin; we will require **3 consecutive non-overlapping OOS folds** = ~810 cycles before promote).
- **Purge / embargo**: Funding rate is auto-correlated over ~3-7 windows (regime persistence). Purge 7 funding windows (≈ 2.3 days) between train and test; embargo a further 3 windows after each trade exits before next eligible entry in the test set.
- **Statistical tests** (all must pass):
  1. Mean net-of-cost annualized carry (after both legs' fees, both venues' funding withhold, and 5 bps round-trip slippage) > **6% annualized**, with 95% bootstrap CI excluding 4%.
  2. Sharpe ratio of the daily PnL stream (daily-marked) > **1.5** with 95% bootstrap CI excluding 1.0.
  3. Max drawdown on the OOS PnL curve **< 4% of deployed capital** across every fold.
  4. **Deflated Sharpe ratio** (López de Prado) > 0 after correcting for the number of threshold combinations tried during training.
  5. Beta of strategy returns to BTC spot daily returns has 95% CI containing zero (delta-neutrality check).
- **Anchor evidence**: Binance funding rate annualized average 2020-2025 is ~10.3% (publicly computable). Practitioner anchors include Hougan & Lawant (2021, Bitwise) on cash-and-carry, the persistent contango in CME BTC futures basis, and the Glassnode / CryptoQuant funding dashboards which document the structural long bias of leveraged retail. The thesis is that this regime persists *conditional on bull-skewed retail leverage*; the validation must include the May-Jun 2022 and Nov 2022 (FTX) negative-funding episodes in OOS.

## 7. Risk envelope

- **Per-trade size**: **1.0 BTC notional per leg** at unit-scale (the "1×" leg). Production-scale multiplier is set by AUM tier — see capacity §9.
- **Max notional cap (one strategy slot)**: **50 BTC notional** combined (i.e. 50 BTC short perp + 50 BTC long spot) — chosen so that at $70k BTC the slot caps at $3.5M gross / $0 net delta but ≤ 1% of BTCUSDT perp 1-min ADV.
- **Max position size as % of account**: Gross notional ≤ **400% of equity** (since the position is delta-hedged, gross can be high; net BTC delta must stay within ±0.5% of equity). On a $1M account, ≤ $4M gross / ~57 BTC at $70k.
- **Margin / liquidation envelope on perp leg**: Use **cross-margin at 2.5x effective leverage**. At 2.5x, BTC would have to rally ~38% in a single 8h window with NO ability to add margin to liquidate — and we have automated margin-top-up scripts triggered at every 5% adverse move. Liquidation distance is monitored every 60s.
- **Daily loss cap**: **−1.5% of account equity** absolute; on breach, unwind both legs immediately.
- **Drawdown trigger** (strategy-level halt): **−3% peak-to-trough on the slot's PnL curve over any 7-day window** halts trading pending review.
- **Circuit breakers**:
  - 3 consecutive negative funding receipts on the short leg (i.e. WE pay funding) → unwind immediately (regime flip detection).
  - Basis (perp − spot) widens > 1.5% → unwind both legs and stop entries until basis < 0.5% for 24h.
  - Either venue's withdrawal latency > 6h or any official "deposits paused" / "withdrawals paused" status → unwind on the surviving venue, accept the basis loss, sit out for 7 days.
- **Correlation rule**: This strategy is delta-neutral and uncorrelated to any directional BTC strategy. It IS correlated to other "crypto carry" trades (basis trade on CME futures, options-funded carry). Hard cap: **combined crypto-carry exposure across all slots ≤ 60% of account equity gross-notional**.
- **Exchange counterparty risk envelope**: Keep no more than **30% of strategy capital on any single venue at any time**. Excess capital sits in cold storage off-exchange; rebalanced weekly.

## 8. Execution plan

- **Broker / SDK**: `ccxt` (Python) — `ccxt.binance({'options': {'defaultType': 'future'}})` for the perp leg and `ccxt.okx()` for the spot leg. Both use API keys with **only trade + read enabled, withdrawals disabled at the API-key level** (hardware enforced — Binance's `enableWithdrawals=false` API setting plus IP allowlist).
- **Order types**:
  - Entry on both legs: **post-only limit, 2 bps inside best price**, GTC. We pay only maker fees (Binance: 0.02% perp maker, OKX: 0.08% spot maker; Binance VIP-1+: 0.016% / 0.04%).
  - Unwind: marketable-limit IOC, 5 bps slip budget, fallback to market if IOC fails twice.
- **Expected transaction cost**:
  - Binance perp entry (maker): 0.02% = 2 bps
  - OKX spot entry (maker): 0.08% = 8 bps
  - Binance perp exit (taker): 0.05% = 5 bps
  - OKX spot exit (taker): 0.10% = 10 bps
  - Slippage budget round-trip: ~5 bps total
  - **All-in round-trip transaction cost ≈ 30 bps** of notional per round-trip. To break even on a single round-trip, we need to capture at least 30 bps of funding, i.e. ~3.3 hours of funding at 10% APR. The strategy needs to hold ≥ 1 funding cycle on average to be net positive; backtest must confirm the average holding period ≥ 3 cycles (~24h).
- **Order rejection handling**: If either leg's entry fails (post-only would-cross, insufficient margin, exchange 5xx), abort the OTHER leg as well — never end up with one leg unhedged. Use a 2-second open-order TTL with automatic cancel + retry up to 3 times, then alert and skip the window.
- **Slippage measurement**: Log (mid-at-decision − fill-price) / mid for every leg fill. Daily report: rolling 30-window slippage vs. 5 bps budget. Halt if average slippage > 8 bps for 5 consecutive days.

## 9. Capacity

- **Per-window funding payment economics**: At $70k BTC, 0.0091% per-window rate = $6.37 funding per BTC per window. 50 BTC × 3 windows/day × 365 = ~$348k/year per 50 BTC slot at 10% annualized, before fees.
- **BTC notional ceiling before market impact erodes carry**:
  - BTCUSDT perp 1-min ADV ≈ ~$20-30M; top-20 book depth on Binance perp typically ≈ 200-500 BTC each side within 5 bps.
  - At **50 BTC per leg per entry, slicing into 5-BTC child orders over ~30 min**, market impact stays ≤ 3 bps based on empirical Binance L2 depth.
  - Above **200 BTC notional per entry**, child-order slicing extends to several hours and impact climbs to 8-12 bps, eroding carry meaningfully.
  - **Practical ceiling: 200 BTC gross notional per leg per slot** = ~$14M gross at $70k, supporting ~$30-50M AUM in this single strategy.
- **AUM ceiling**: **~$50M AUM** for this strategy alone before degradation; beyond that, additional venues (Bybit, Deribit) and more aggressive slicing are required, and the cross-venue inventory management becomes the dominant operational risk.
- **ADV % consumed**: At 50 BTC entry per slot, that's ~0.15% of BTCUSDT 1-min ADV and <1% of 5-min ADV. Comfortable.
- **Concentration**: 100% concentrated in BTC by design. No diversification within the strategy. Diversification must come at the *portfolio* level (other strategies in other slots), not within this slot.

## 10. Failure modes

1. **Funding regime flip (sustained negative funding)** — Funding rates went negative for weeks in May-Jun 2022 and Nov 2022 (post-FTX) as bear sentiment dominated. The short-perp leg then *pays* funding instead of receiving. Signature in metrics: 3+ consecutive negative funding receipts, strategy carry turns negative. **Detection**: realtime funding monitor; **response**: unwind on first negative-window streak of length 3.

2. **Exchange insolvency / withdrawal freeze (FTX, Mt.Gox, QuadrigaCX style)** — Counterparty goes bust or freezes withdrawals; capital on that venue becomes encumbered for months/years. Signature: status-page "withdrawals paused"; spreading rumors on Twitter/Telegram; spike in stablecoin discount on that venue; withdrawal latency > 6h. **Detection**: hourly health probes + USDT/USDC on-venue vs off-venue price differential; **response**: pre-emptive withdrawal-then-redeposit rotation weekly to ensure withdrawals work; on incident, accept basis loss and exit the surviving leg, write off impaired venue capital ≤ 30% of strategy capital by allocation rule.

3. **Spot-perp basis blow-out** — Perp can dislocate from spot during forced liquidation cascades (March 12 2020 ETH/BTC, May 19 2021, Nov 9 2022). Perp can trade 3-8% away from spot for minutes-to-hours. If we are caught with the short perp deep in the money on the basis (perp >> spot) we have unrealized gain — but if perp >> spot and we're unwinding, slippage destroys carry. Reverse case (perp << spot) creates unrealized loss on the short leg that triggers margin calls. **Detection**: basis monitor every 1s; **response**: 1.5% basis circuit breaker; pre-funded margin buffer (40% IM means ~38% adverse move tolerated without liquidation).

4. **Regulatory action on perpetuals** — US CFTC / SEC enforcement against Binance (already partial — Binance.US settlement, CZ DoJ plea), or EU MiCA-driven restrictions, or country-specific bans (UK FCA, Singapore MAS). Could force US-touching entities off Binance with little warning; could make perpetuals unavailable to non-retail. Signature: announcement, then 24-72h migration window. **Detection**: regulatory news scraper + weekly check; **response**: pre-built failover plan to Bybit + OKX, geographic entity isolation (Cayman LLC trades, US individual does not), and ≤ 30 days of regulatory-grace-period tolerance built into capital allocation.

5. **Stablecoin depeg on the quote currency (USDT / USDC)** — Both legs are USDT-quoted. A USDT depeg (e.g. May 2022 Terra/UST contagion) would distort the carry calculation and create paper losses denominated in a depegging unit. **Detection**: USDT/USD spot monitor; **response**: if USDT depeg > 2%, unwind both legs, convert to BTC + USDC, sit out.

6. **API outage / partial fills mid-roll** — Either venue's API has historically had hour+ outages around major market moves (the exact moments we most need to unwind). **Detection**: per-venue heartbeat; **response**: fallback to web UI manual unwind procedure documented in runbook; alert operator within 60s.

## 11. Falsifiable verdict criteria

**Promote to live capital** (after walk-forward + ≥30 shadow days):
- Walk-forward (≥ 810 OOS funding cycles): mean annualized net carry > 6%, Sharpe > 1.5, max DD < 4%, 95% CI on net carry excludes 4%
- AND shadow run (≥ 30 days = ~90 funding cycles): realized net carry within 1 standard error of walk-forward expectation
- AND no broker-truth divergence event: reconciled position via ccxt `fetch_positions()` and `fetch_balance()` every cycle, zero mismatches vs local cache (Lesson 4)
- AND zero unhedged-leg episodes (one leg filled while the other failed)
- AND average slippage ≤ 5 bps round-trip

**Continue shadow**:
- Metrics ambiguous (e.g. Sharpe 1.0-1.5, carry 4-6%) but no kill triggers hit; extend shadow another 30 days; re-evaluate.

**Kill immediately** (any one):
- 7-day drawdown > 3% of slot capital
- Daily loss cap −1.5% breached
- Any broker-truth divergence (local-vs-fetched position mismatch > 0.5% in BTC terms)
- Any unhedged-leg episode lasting > 5 minutes
- 3 consecutive negative funding receipts AND auto-unwind did not fire
- Either venue triggers withdrawal freeze and our capital-on-venue rule was violated at the time

## 12. Operational risk note

The operational surface is unusually wide for a "simple" carry trade. Failure modes: (a) ccxt's unified API can mask venue-specific behavior — Binance's `reduceOnly` flag, OKX's `posSide` for hedge-mode accounts, and per-venue funding-time edge cases all need explicit handling and per-venue integration tests on testnet; (b) API key rotation and IP allowlisting must be enforced — a single leaked key on the perp venue with trade permissions is enough to drain margin via wash-trading; (c) USDT inventory must be balanced across both venues, which requires periodic on-chain transfers — those carry ETH/TRC20 gas, withdrawal latency (typically 10-60min, sometimes hours), and the irreversible-mistake risk of mis-typed addresses (mitigated by ccxt address whitelisting and 2-of-2 internal sign-off for any new whitelist entry); (d) clock skew on the VPS matters — funding payments are timestamped to the exchange's clock and our pre-funding decision logic depends on NTP-synced time within 200ms; (e) **24/7 operations** means there is no "after-hours" — alerts must page during the operator's sleep and a documented runbook must enable a non-specialist to unwind both legs at 3am.

---

## Addressing the 10 Lessons

1. **Validate before deploy** — Walk-forward + ≥30-day shadow before any live capital, with explicit numeric thresholds in §11. Honored.
2. **Binary gates, not confidence rankers** — Entry/exit are deterministic thresholds on funding rate and basis. No ML confidence score sizes positions. Honored.
3. **TP:SL asymmetry is structural** — Asymmetry is baked in: unwind at 1.5% basis blow-out (small loss tolerance) and stay in for unbounded carry runs (let winners compound across funding cycles). The asymmetric structure is the strategy. Honored.
4. **Broker truth, not local cache** — Every funding window, `ccxt.fetch_positions()` and `ccxt.fetch_balance()` reconcile against local state on both venues. Closing orders use the broker-reported quantity, never the local cache. Honored.
5. **No PDT-exposed lanes on sub-$25k US equity accounts** — **N/A: this is crypto, not US equity**. PDT is a FINRA Rule 4210 construct that applies only to US margin equity accounts. Spot crypto and perpetual swaps are outside FINRA jurisdiction entirely. No round-trip count limits apply. The lesson is honored by virtue of asset-class choice. Honored by exemption.
6. **One strategy per slot** — This iteration occupies one slot ("BTC funding carry"). Hard-reject any other iteration in this run whose primary edge is BTC perp funding. Honored.
7. **No same-data backtest** — Walk-forward with 1-month step, purge 7 windows, embargo 3 windows; 3 consecutive non-overlapping OOS folds required. Honored.
8. **Battle-tested broker SDKs only** — **ccxt** (Python) is the canonical multi-exchange library, 30k+ stars, used by every major crypto quant fund. No hand-rolled HTTP. Honored.
9. **Capacity bound named upfront** — Per-leg ceiling 200 BTC notional; strategy AUM ceiling ~$50M; ADV consumption < 0.5% at full scale. Stated in §9. Honored.
10. **Kill criteria written before deploy** — Exact numeric kill triggers in §11 (7-day DD > 3%, daily cap −1.5%, broker divergence, unhedged-leg episode, regime flip). Honored.

---

## Innovation matrix cell occupied

- **Asset class**: Crypto perpetuals + spot (delta-neutral)
- **Geography**: Crypto (Binance primary, OKX hedge, Bybit failover) — global, jurisdiction-diverse
- **Time scale**: 8-hour funding windows (3 events/day)
- **Method**: Rules-based cash-and-carry (no ML, no NN, no LLM)
- **Edge source**: Carry / funding-rate harvest (structural premium for leveraged longs)
