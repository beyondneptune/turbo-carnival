# Iteration 19 — On-Chain DEX Flow + Perp Funding + Stablecoin Inflow Composite (Directional ETH/BTC Perps)

## 1. Thesis (one sentence)

For ETH and BTC perpetual swaps on a 4-48h horizon, a directional long-or-short position taken when at least **2 of 3 orthogonal signals** align — (a) funding-rate z-score crossing ±1.5 vs 30-day rolling baseline, (b) DEX/CEX ETH-spot volume ratio crossing the 80th percentile of the 90-day distribution, (c) stablecoin (USDT+USDC) net on-chain inflow to CEX hot wallets crossing ±$50M/hr — captures lead-lag information that is **not** present in price alone, producing a hypothesized OOS Sharpe of **1.0-1.6 net of costs**, falsifiable if over **200 OOS signal-triggered trades** the net Sharpe is < 0.6 or the win-rate is < 50% at a >1.4 average payoff ratio.

This is explicitly **directional** (not delta-neutral like iter_3), with a substantially tighter risk envelope: max 15% NAV gross per trade and a hard 5% per-trade drawdown stop.

## 2. Market / asset class / venue

- **Instruments**:
  - Primary directional execution: `ETH-USDT-SWAP` (OKX) and `BTCUSDT` perp (Binance Futures) — linear USDT-margined perpetuals; ETH is the primary asset (because the DEX flow signal is ETH-native on Ethereum mainnet) and BTC is included as a correlated co-trade *only when* the signal direction agrees on both
  - On-chain reference assets observed (not traded): `ETH/USDC` 0.05% pool on Uniswap V3 (mainnet), Curve 3pool, dYdX v4 perps (Cosmos chain — observed for cross-validation only, not executed against)
  - Stablecoin flows observed: ERC-20 transfers of `USDT` (0xdAC1...), `USDC` (0xA0b8...) into/out of labeled CEX hot wallets (Binance, OKX, Coinbase, Kraken, Bybit)
- **Primary execution venue**: OKX (ETH) + Binance Futures (BTC) — both have deep books (>$5B daily ETH perp ADV on each), USDT-margined, and ccxt support. OKX is primary on ETH because it has cleaner co-location for Asia-side fills during typical signal-trigger times (Asia/Europe overlap)
- **Hedge/failover venue**: Bybit perps as a third venue for failover only (never primary)
- **Broker / SDK**: `ccxt` for CEX legs; The Graph subgraphs (Uniswap V3, Curve) + Etherscan API + Dune Analytics SQL API for on-chain data, with **Glassnode Studio** as the paid commercial fallback
- **Settlement**: Funding settles 8-hourly (00/08/16 UTC on Binance; same on OKX). Position PnL is realized at exit. Crypto trades 24/7
- **Margin**: Cross-margin, target effective leverage **3x** on the position (i.e. 33% initial margin), well below 100x max; far enough below liquidation that a 20% adverse move does not liquidate
- **Tax regime**: Crypto is property under IRS Notice 2014-21 (US). Short-term cap gains apply (positions held 4-48h). Account is owned by the same Cayman/BVI structure used in iter_3, with US-resident operator drawing salary, not trading gains

## 3. Time horizon

**Hourly signal evaluation, holding period 4-48h typical, hard maximum 72h.**

- Signal evaluation: every hour on the hour (UTC), aligned to on-chain block batches (typically 12-second blocks; one hour ≈ 300 blocks)
- Entry decision: within 5 minutes of the hourly evaluation, after on-chain data has been reorg-safe (we wait 25 block confirmations ≈ 5 minutes — see §12 ops risk for why)
- Holding period: minimum **4 hours** (one half-funding-cycle, to avoid being whipsawed by intra-hour noise), typical **12-24 hours** (a 1-2 funding-cycle expression of the thesis), maximum **72 hours** (after which signal decay dominates per backtest)
- Time-stop: if no thesis-confirmation event occurs within 24h of entry (defined: signal score remains ≥ 2/3 *or* MTM > +1.0% of position notional), tighten the stop

## 4. Method family and specific algorithm

**Family**: Rules-based composite signal stack with binary gates per Lesson 2. No ML confidence score sizes positions. Each of the three signals is a **binary gate** that fires or does not fire. Position is taken only when ≥ 2 of 3 gates fire in the same direction. Position size is fixed per trade — not scaled by signal "strength."

**The three signals (all binary)**:

### Signal A — Funding rate z-score
- Compute current 8h funding rate `f_t` and z-score against trailing 30-day rolling distribution `Z_t = (f_t − μ_30d) / σ_30d`
- **A_long fires** if `Z_t < -1.5` (funding extremely negative relative to recent history → shorts are over-paid, suggests price is poised to revert upward as short crowding unwinds)
- **A_short fires** if `Z_t > +1.5` (funding extremely positive → longs are over-paying → mean-reversion downward likely)
- Anchor: Liu/Tsyvinski/Wu 2022 documents funding-rate mean-reversion as a robust crypto factor

### Signal B — DEX/CEX volume ratio
- Define DEX share of ETH spot 1-hour volume as `D_t = ETH_DEX_vol_1h / (ETH_DEX_vol_1h + ETH_CEX_spot_vol_1h)`
- DEX volume aggregated across Uniswap V2/V3 + Sushi + Curve + Balancer + Pancake (Ethereum mainnet only for v1, with planned L2 expansion in v2 — see §10 failure mode 4)
- CEX volume aggregated across Binance + Coinbase + OKX + Kraken + Bybit ETH spot (NOT perp)
- Compute 90-day rolling percentile rank of `D_t`. Call this `P_D`
- **B_retail fires** if `P_D > 0.80` (top quintile DEX share = retail-driven move, often a *fade* signal — institutional flow on CEX is the smart side)
- **B_institutional fires** if `P_D < 0.20` (bottom quintile = institutional CEX accumulation/distribution)
- **B_long fires** if (B_institutional AND CEX-side ETH spot 1h return > +0.5%) OR (B_retail AND ETH 1h return < −1.0%, i.e. retail panic-sell to fade)
- **B_short fires** if (B_institutional AND CEX-side ETH spot 1h return < −0.5%) OR (B_retail AND ETH 1h return > +1.0%, i.e. retail FOMO to fade)
- Anchor: Lehar/Parlour 2023 on DEX-CEX informed-flow segmentation; Capponi/Jia 2021 on retail concentration in AMMs

### Signal C — Stablecoin net inflow to CEX
- Compute net inflow `S_t` = (USDT + USDC ERC-20 transfers INTO labeled CEX hot wallets) − (transfers OUT of those wallets) over the trailing 1 hour, summed across the top 5 CEX cluster
- **C_long fires** if `S_t > +$50M/hr` (stablecoins arriving at CEX = imminent buying pressure)
- **C_short fires** if `S_t < −$50M/hr` (stablecoins leaving CEX = capital exodus, often preceding a sell-off as participants move to cold storage)
- Anchor: practitioner work from Glassnode (Exchange Net Position Change) and Nansen (Smart Money flows); academic anchor Cong/Li/Wang 2022 on stablecoin flows as information

### Composite gating rule
- **Enter long** if at least 2 of {A_long, B_long, C_long} fire in the same hourly evaluation AND no contradictory short signal fires
- **Enter short** if at least 2 of {A_short, B_short, C_short} fire in the same hourly evaluation AND no contradictory long signal fires
- **Do not enter** if signals split (e.g. A_long + B_short) — wait for re-alignment
- **Co-trade BTC** at 50% of ETH notional if and only if BTC's own A signal (BTC funding z-score) agrees with ETH's directional verdict (BTC has no native DEX signal — its DEX market is ~5% the size of ETH's — so B and C are ETH-only)

### Exit logic
- **Take-profit** at +3.0% MTM of position notional, or when ≥ 2 signals flip against the position
- **Stop-loss** at −2.0% MTM of position notional **OR** a per-trade hard stop of **−5.0% of account NAV** (whichever triggers first)
- **Time-stop** at 72h regardless of P&L
- **Re-entry lockout**: 4 hours after any exit on the same asset, to avoid signal-chop

**Why composite and not single-signal**: each gate alone has too many false positives (funding z-scores ≥ |1.5| occur ~5% of the time; DEX-share extremes ~20% by construction; stablecoin spikes ~10%). Requiring 2-of-3 yields roughly 1-2 entries per asset per week, with the orthogonality of the signals (funding = microstructure, DEX share = participant-mix, stablecoin = capital-flow) cutting the false-positive rate substantially. The 2-of-3 rule is itself a binary gate (Lesson 2).

## 5. Data requirements

| Feed | Source (primary / fallback) | Purpose | Storage/yr | Cost/mo |
|---|---|---|---|---|
| ETH/BTC perp 1m OHLCV + funding history | Binance + OKX REST (free) / ccxt | Signal A, execution | ~5 GB | $0 |
| ETH spot 1m OHLCV across 5 CEX | ccxt unified / Kaiko ($) | Signal B numerator | ~3 GB | $0 free / $2k Kaiko |
| ETH DEX hourly volume per pool | The Graph subgraphs (free public endpoints) / Dune Analytics SQL ($350/mo team) / Glassnode ($799/mo Advanced) | Signal B denominator | ~1 GB | $0 / $350 / $799 |
| ERC-20 stablecoin transfer events (USDT + USDC) | Etherscan API (free, 5 req/s) / Alchemy ($49/mo growth) / Glassnode | Signal C | ~10 GB (transfer logs) | $0 / $49 / inc above |
| CEX hot-wallet address labels | Arkham Intelligence (free tier) + Etherscan labels + Nansen ($) | Signal C cluster definition | <1 MB (a labels file refreshed weekly) | $0 / $150 Nansen |
| L2 orderbook for execution | ccxt watch_order_book WS | Slippage estimation | ~30 GB (top-20, 1Hz) | $0 |
| Block-reorg / chain-head monitor | Public Ethereum RPC (Infura free tier, 100k req/day) / Alchemy fallback | Reorg-safety (25 conf wait) | <100 MB | $0 / $49 |

- **Recommended production stack**: Glassnode Advanced ($799/mo) as **primary** for on-chain signals (B + C) — it aggregates DEX volume and labeled CEX flows with documented methodology and SLA; **fallback** is The Graph + Etherscan + a Dune $350 plan, which is more brittle but free-tier survivable for a few days during a Glassnode outage. Total: ~**$850/mo on-chain data**, ~**$0 CEX data**, plus a $40/mo VPS = **~$900/mo all-in**
- **Storage**: ~50 GB/year keeping minute bars + transfer logs + DEX hourly + funding. Acceptable on a $40/mo VPS with 200GB SSD
- **Latency requirement**: on-chain data is **NOT** low-latency — it has structural delay due to block confirmations. We wait **25 block confirmations (~5 min)** before treating an inflow as "real" to avoid reorg-induced false signals. CEX execution latency is <500ms, same as iter_3. Co-location not needed

## 6. Validation plan

- **Historical window**: Uniswap V3 launched May 2021, USDT/USDC on Ethereum predate that, OKX/Binance funding data back to 2020. Effective overlapping window: **Jun 2021 → Apr 2026** ≈ 5 years
- **In-sample**: Jun 2021 – Dec 2023 (30 months) for parameter tuning (the z thresholds, DEX-share percentile cutoffs, $50M stablecoin threshold)
- **OOS**: Jan 2024 – Apr 2026 (28 months) for walk-forward held-out evaluation. No tuning on OOS — single shot
- **Walk-forward design** within IS for sensitivity:
  - Train window: 12 months
  - Test window: 3 months
  - Step: 3 months (non-overlapping test folds)
  - Yields ~7 IS folds; thresholds are chosen by best-median-Sharpe-across-folds, not best in any single fold (López de Prado purged CV style)
- **Minimum OOS trades**: **200 signal-triggered round-trips** (ETH + BTC combined; ETH expected to provide ~80% of trades). With hourly evaluation and ~1-2 entries per asset per week, 28 months yields ~150-250 ETH entries + ~50-100 BTC co-trades = 200+ comfortably
- **Purge / embargo**: Signal autocorrelation is short (funding mean-reverts in ~3-7 windows; DEX share is sticky for ~4-12 hours; stablecoin spikes are episodic). Purge **24 hours** between train and test segments; embargo **4 hours** after every trade exit before next eligible entry
- **Statistical tests** (all must pass):
  1. Net Sharpe (after 30 bps round-trip CEX cost + 5 bps slippage) > **1.0** on OOS, 95% bootstrap CI excluding 0.6
  2. Hit rate ≥ **52%** with average winner / average loser ≥ **1.4** (so the expected payoff structure passes Lesson 3's TP:SL asymmetry test)
  3. Max drawdown on OOS equity curve **< 12% of NAV** across the full 28-month OOS window
  4. **Deflated Sharpe ratio** > 0 after correcting for the ~30 threshold combinations tried in IS tuning
  5. Beta of strategy returns to ETH daily returns has 95% CI containing zero **OR** if non-zero, the strategy IR (alpha / residual vol) > 0.7 — i.e. either market-neutral by construction or genuinely producing alpha above its beta
  6. **Signal contribution decomposition**: in OOS, no single signal alone may explain >70% of P&L (else the composite is fake diversification and we should simplify)
- **Anchor evidence**: Liu/Tsyvinski/Wu 2022 (RFS) "Common Risk Factors in Cryptocurrency" documents funding-rate and momentum factors in crypto. Makarov/Schoar 2020 (JFE) "Trading and Arbitrage in Cryptocurrency Markets" documents persistent CEX-CEX price segmentation. Lehar/Parlour 2023 (WP) "Decentralized Exchange: The Uniswap Automated Market Maker" decomposes DEX flow into informed vs uninformed. Practitioner: Glassnode's Exchange Net Position Change has documented predictive power for 1-3 day BTC moves in their public research

## 7. Risk envelope

- **Per-trade gross notional**: target **10% of NAV per asset**, hard cap **15% of NAV per asset**. On a $1M NAV, that is $100k-$150k per ETH trade and $50k-$75k for the BTC co-trade (50% sizing rule)
- **Total simultaneous exposure cap**: ETH + BTC co-trade combined ≤ **20% of NAV gross**. The strategy is forbidden from holding two same-direction positions across both assets summing > 20%
- **Effective leverage on perp**: 3x (i.e. 33% initial margin). At 3x, BTC/ETH would need to move ~30% adverse in a single hour with no margin top-up to liquidate. Real liquidation distance is monitored every 60s
- **Per-trade hard stop**: **−5.0% of NAV per trade** is the absolute backstop. With max 15% gross at 3x leverage, the underlying would need to move 11.1% adverse before the −5% NAV stop triggers — well within typical daily ETH range. The −2% MTM stop on position notional triggers first in most cases (it's tighter), keeping per-trade NAV loss to ~0.3-0.6%
- **Daily loss cap**: **−2.5% of NAV** absolute (much tighter than iter_3's −1.5% because iter_3 is delta-neutral; this is directional and we expect occasional sharp losers, so the cap exists to prevent 3-bad-trades-in-a-row blowups). On breach, halt all entries for 24h
- **Weekly loss cap**: **−6% of NAV** halts entries for the rest of the week and triggers a manual review before restart
- **Strategy-level drawdown trigger**: **−12% peak-to-trough** of slot equity over any 30-day window halts the strategy pending review
- **Circuit breakers**:
  - 3 consecutive losing trades → halve position size for the next 5 trades (then restore if positive)
  - 5 consecutive losing trades → halt for 72h, manual review
  - On-chain data vendor outage (primary unavailable > 15 min) → switch to fallback; if fallback also unavailable, halt new entries until restored
  - Stablecoin (USDT or USDC) depeg > 1.0% → halt new entries; existing positions exit at next signal-flip or time-stop
- **Correlation rule with other slots**:
  - This is directional crypto: it WILL correlate with any directional-crypto slot (e.g. a momentum strategy)
  - Hard rule: **no other directional-crypto slot may run concurrently in the same account**. The delta-neutral funding-arb slot (iter_3) IS allowed concurrently because its delta is zero
  - Aggregate directional-crypto exposure across all slots ≤ 25% NAV gross

## 8. Execution plan

- **Broker / SDK**: `ccxt` (Python). `ccxt.okx({'options': {'defaultType': 'swap'}})` for ETH and `ccxt.binance({'options': {'defaultType': 'future'}})` for BTC. API keys: **trade + read only, no withdrawals, IP allowlisted, hardware enforced** at the exchange API-key configuration page
- **Order types**:
  - Entry: **marketable-limit IOC at 5 bps slip budget**, fallback to next 10 bps if first attempt unfilled, fallback to market if second attempt fails. Speed > price for directional entries because signals are time-sensitive (information may decay within 1-2 hours of trigger)
  - TP/SL: native exchange stop-limit at the calibrated thresholds; the `−5% NAV` hard stop is enforced bot-side as a back-stop in case the exchange stop fails
- **Expected transaction cost**:
  - OKX perp taker: 0.05% = 5 bps; Binance perp taker: 0.05% = 5 bps
  - Slippage budget: 5 bps per side
  - Round-trip: 5+5+5+5 = **~20 bps per round-trip per asset** (vs iter_3's 30 bps because we are taker not maker)
  - To break even on a single trade: need ≥ 20 bps of directional move ≈ 0.2% on ETH. With 3x leverage, that's 0.07% account movement to cover cost — easily achievable on 4-48h horizon
- **Order rejection handling**: directional single-leg trade, so rejection just means no fill — no unhedged-leg risk. Bot retries 3x then alerts and skips the hour. Critically: **never enter at a price > 50 bps from the decision-time mid** — if the market has moved that far while we were deciding, the signal information is stale; skip
- **Slippage measurement**: log decision-time mid vs fill for every entry/exit; rolling 30-trade slippage report; halt if avg slippage > 10 bps for 5 consecutive trades

## 9. Capacity

- **Per-trade size economics**: At 10% NAV per ETH trade, $1M NAV = $100k position. ETH perp 1-min ADV on OKX is ~$2-5B, so $100k is <0.01% of ADV — negligible impact. Even at $50M NAV (5% of OKX's 1-min ADV), a $5M-$7.5M ETH position is ~0.15% of 1-min ADV, well within slip budget
- **AUM ceiling**: **~$50-75M AUM** before market-impact starts eroding the edge. Above that, child-order slicing extends entry to 15-30 minutes, by which time the hourly signal may have decayed
- **ADV % consumed**: <0.2% of 1-min ADV at full $50M scale on either OKX or Binance
- **Concentration**: 100% concentrated in ETH + BTC by design. The whole strategy is two-asset. Diversification must come from other slots in the portfolio, not within this slot
- **Capacity-degrading mechanism specific to this strategy**: as $$$ flow into on-chain-signal strategies industry-wide, the signal half-life shortens. Practitioner reports (Nansen, Arkham) suggest stablecoin-inflow signals had ~24h half-life in 2021-22 and ~4-8h half-life in 2024-25. We assume signals decay further; **kill criterion in §11 includes a signal-decay test**

## 10. Failure modes

1. **Regulatory action on stablecoins** — A US Treasury / SEC / EU action against Tether (long-running solvency questions) or Circle (post-SVB tail) could cause a USDT or USDC depeg of 1-10%, simultaneously poisoning both the C signal (inflows become noise) and the P&L denomination. Signature: stablecoin spot < 0.99 vs USD across 3+ venues for >1 hour. **Detection**: stablecoin-price monitor; **response**: depeg circuit breaker in §7 halts entries; exit existing positions at next signal-flip or 24h time-stop, whichever first

2. **DEX volume migration to L2s breaks the mainnet flow proxy** — Arbitrum + Base + Optimism + ZK rollups have been capturing growing share of ETH/USDC DEX volume (≈ 30% as of 2026 per L2Beat). If the migration accelerates past ~50% of total DEX volume, Signal B's mainnet-only denominator becomes a biased estimator (we'd undercount DEX share, calling some retail moves "institutional"). Signature: gradual decline in Signal B's predictive contribution in OOS rolling 90-day window. **Detection**: monitor L2Beat / Dune's L2-vs-L1 DEX volume tracker quarterly; **response**: planned v2 expansion to include top-3 L2 subgraphs in the DEX denominator; until then, decay-monitor kill criterion in §11 will catch the regime change

3. **On-chain data vendor outage at signal time** — The Graph public endpoints have had multi-hour outages (Apr 2023, Jan 2025); Glassnode has had documented 2-hour outages during major market moves (the exact times we most need them). Signature: vendor 5xx errors, stale data timestamps. **Detection**: per-vendor heartbeat every 5 min; fall through primary → secondary → tertiary; **response**: if all on-chain sources unavailable > 15 min, halt new entries (Signal C cannot be computed reliably; running on A+B only is a different strategy and is not authorized)

4. **Signal decay** — As more participants subscribe to on-chain alpha (Nansen Pro has ~10k users in 2026), the alpha is increasingly priced in within minutes of signal emission. Signature: declining rolling-90d Sharpe in OOS, declining hit rate. **Detection**: monthly review of trailing-90d performance; **response**: kill criterion in §11 — if trailing 90d Sharpe < 0.5 *and* hit rate < 50%, halt and re-evaluate

5. **CEX hack / exchange insolvency** (OKX or Binance) — Same as iter_3 risk: counterparty goes bust or freezes withdrawals; capital becomes encumbered. Signature: status-page "withdrawals paused", USDT discount on-venue, social-media leak. **Detection**: hourly health probes; weekly withdrawal test. **Response**: pre-emptive failover to Bybit; 30% per-venue capital cap (same as iter_3) keeps the blast radius bounded

6. **Reorg-induced false on-chain signals** — Ethereum reorgs are rare on mainnet (<1 per month, usually 1-2 blocks deep) but possible. A stablecoin transfer that was "in" a block at time t might disappear in a reorg at time t+12s. Signature: Etherscan flips transaction status from confirmed → unconfirmed. **Detection**: we wait 25 confirmations (~5 min) before acting on any inflow; **response**: built-in by construction. A reorg deeper than 25 blocks (which would be a critical security event for Ethereum itself) would be a halt-all-crypto-trading kind of event

7. **Smart-contract event affecting stablecoin contract** (USDT/USDC contract pause, blacklist event, upgrade bug) — Both USDC and USDT have admin keys that can pause or blacklist addresses. A pause event would freeze the entire stablecoin temporarily and corrupt signals. Signature: contract `paused()` returns true; on-chain transfer count drops to ~0. **Detection**: monitor contract `paused()` flag every block; **response**: halt entries; exit existing perp positions; reconcile balances when chain operations resume

8. **Funding rate manipulation / wash-trading shocks** — Funding rate is calculated from perp-index basis, manipulable by wash-trading at thin-liquidity moments. Spike in funding z-score could be artifact, not signal. Signature: divergence between Binance and OKX funding z-scores > 2σ. **Detection**: cross-venue funding consistency check before firing Signal A; **response**: require Signal A to agree across Binance AND OKX or Signal A does not fire on that asset

## 11. Falsifiable verdict criteria

**Promote to live capital** (after walk-forward + ≥30 shadow days):
- OOS (Jan 2024 – Apr 2026, ≥ 200 trades): net Sharpe > 1.0 (95% CI excluding 0.6), max DD < 12% NAV, hit rate ≥ 52% with payoff ratio ≥ 1.4, signal-contribution decomposition shows no single signal > 70% of P&L
- AND shadow run (≥ 30 days, ~30-60 signal-triggered trades): realized Sharpe within 1 SE of OOS estimate; no broker-truth divergence; no on-chain-vendor outage causing missed exits
- AND zero events of position-size > 15% NAV per asset or aggregate > 20% NAV (risk-envelope honored)
- AND average slippage ≤ 10 bps per side
- AND signal alignment rate within ±20% of backtest expectation (sanity check that signals fire at expected frequency, ~1-2/week per asset)

**Continue shadow**:
- Metrics ambiguous (Sharpe 0.6-1.0, hit rate 48-52%) but no kill triggers; extend shadow 30 days; re-evaluate

**Kill immediately** (any one):
- 30-day drawdown > 12% NAV
- Daily loss cap −2.5% NAV breached
- Weekly loss cap −6% NAV breached
- 5 consecutive losing trades AND auto-halve-size did not fire
- Any broker-truth divergence (local-vs-fetched position mismatch > 0.5% in BTC/ETH terms)
- Any per-trade NAV loss > 5% (the hard stop failed to fire)
- Trailing 90d Sharpe < 0.5 AND hit rate < 50% (signal-decay kill trigger)
- USDT or USDC depeg > 1.0% sustained > 1h and existing positions were not exited per the depeg circuit breaker
- On-chain data vendor outage where the bot continued entering trades (failure of the 15-min halt rule)

**90-day falsification clock**: The kill triggers above ensure shutdown within 90 days of any of the following: signal decay (90d window), regime change (30d DD), or single catastrophic event (per-trade or daily). If at day 90 the strategy is alive but cumulative net return < risk-free rate, kill and re-evaluate.

## 12. Operational risk note

The operational surface is wider than iter_3 by every dimension because of the on-chain leg. (a) **On-chain data has structural latency (block confirmations) and reorg risk**; the 25-confirmation wait is non-negotiable and means our signal is always ~5 min behind chain head, which we treat as a feature (avoids reorg false signals) not a bug. (b) **Vendor differences matter**: Glassnode, Dune, The Graph, and Nansen each define "CEX hot wallet" slightly differently — same metric, different numbers; production must lock to a single primary vendor's labels and refresh weekly, with the fallback vendor used only for outage continuity (knowing values will differ ~5-15%). (c) **The Graph public endpoint rate limits** (free tier: ~1k req/day per IP) can starve us during high-traffic periods; the production stack must include the $350/mo Dune team plan or $799/mo Glassnode Advanced as paid backstop. (d) **Stablecoin contract admin events** (USDC pause, USDT blacklist) are governance risks unique to this strategy and must be monitored every block via direct RPC calls — we cannot rely on vendors to surface these in time. (e) **24/7 on-call rotation** because signals fire any hour; specifically, the most predictive signal triggers historically cluster around Asia overnight (~22:00-04:00 UTC), which is the hardest operator shift to staff. (f) **API key isolation**: separate keys per venue, each scoped to one strategy, withdrawals disabled hardware-side, IP-allowlisted to one VPS per venue. (g) **Clock skew**: NTP synced within 50ms; on-chain timestamp is from block.timestamp which can be ±15s of wall-clock — we use median of last 5 blocks for time alignment.

---

## Addressing the 10 Lessons

1. **Validate before deploy** — Walk-forward IS 2021-2023, single-shot OOS 2024-2026 with ≥200 OOS trades, then ≥30 day shadow before any live capital. Numeric promote/kill thresholds in §11. Honored.
2. **Binary gates, not confidence rankers** — Each of A, B, C is a hard binary fire/no-fire. The composite is a 2-of-3 vote, itself binary. No signal "strength" sizes positions; position size is fixed per trade. Honored.
3. **TP:SL asymmetry is structural** — TP at +3.0% MTM, SL at −2.0% MTM = 1.5:1 payoff target, with hit-rate gate at 52% so expectancy is positive. Time-stop at 72h prevents dead-money compounding losses. Asymmetry is baked into the validation test (#2 in §6). Honored.
4. **Broker truth, not local cache** — Every hourly signal evaluation reconciles `ccxt.fetch_positions()` and `fetch_balance()` against local state on both OKX and Binance. Exit orders use broker-reported size. Honored.
5. **No PDT-exposed lanes on sub-$25k US equity accounts** — N/A: crypto perpetuals, outside FINRA. Honored by exemption (same as iter_3).
6. **One strategy per slot** — This iteration occupies one slot ("ETH/BTC on-chain composite directional"). Distinct from iter_3 (delta-neutral funding-arb on BTC). Mandatory exclusion: this slot cannot run concurrently with any other directional crypto slot (enforced in §7 correlation rules). Honored.
7. **No same-data backtest** — IS 2021-2023 for threshold tuning, OOS 2024-2026 single-shot, walk-forward CV within IS for sensitivity, purge 24h, embargo 4h, deflated Sharpe. Honored.
8. **Battle-tested broker SDKs only** — `ccxt` for CEX execution (industry standard). On-chain data via well-known APIs (Etherscan / The Graph / Alchemy / Glassnode) — these are not "broker SDKs" in the FINRA sense but they are battle-tested commercial-grade endpoints used by every major crypto fund. No hand-rolled JSON-RPC against random nodes. Honored.
9. **Capacity bound named upfront** — AUM ceiling ~$50-75M before degradation; ADV consumption <0.2% at full size; signal-decay-driven capacity erosion explicitly tracked with a kill criterion. Honored.
10. **Kill criteria written before deploy** — Exact numeric kill triggers in §11 covering drawdown, per-trade limit, signal decay, broker divergence, depeg, vendor outage. 90-day falsification clock explicit. Honored.

---

## Innovation matrix cell occupied

- **Asset class**: Crypto perpetuals (directional, NOT delta-neutral — distinct from iter_3)
- **Geography**: Crypto (OKX primary ETH, Binance primary BTC, Bybit failover) + Ethereum mainnet (on-chain layer)
- **Time scale**: Hourly signal evaluation, 4-48h holds
- **Method**: Rules-based composite signal stack with 2-of-3 binary gating across orthogonal information sources (microstructure + participant-mix + capital-flow)
- **Edge source**: Cross-domain information fusion — on-chain leading indicators not present in CEX price data alone, combined with CEX microstructure
