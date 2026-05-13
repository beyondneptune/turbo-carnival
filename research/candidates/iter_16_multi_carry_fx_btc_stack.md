# Iteration 16 — Multi-Carry Stack: BTC Perp Funding × G10 FX Carry, Risk-Parity Overlay

## 0. Composition note (read first)

This is a **combinatorial / meta-strategy** iteration as defined in the spec §"Evolution pattern across iterations" (iter 16+: combinatorial — pair a candidate with a hedge from a different geography). It does **not** introduce a new alpha source. It stacks two pre-existing candidates as legs of a single portfolio:

- **Leg A** — BTC perpetual funding cash-and-carry (entire mechanics defined in `iter_3_btc_perp_funding_arb.md`).
- **Leg B** — G10 FX long-3/short-3 monthly carry (entire mechanics defined in `iter_4_g10_fx_carry_diversified.md`).

Everything leg-internal (entry/exit thresholds, single-leg risk controls, single-leg validation, single-leg failure modes) is delegated to those two files and **not re-litigated here**. This iteration only specifies:

1. The portfolio-level **risk-parity weighting overlay** that sizes each leg.
2. The **correlation circuit breaker** that flattens one leg if the two legs start co-moving.
3. The **dual-broker NAV orchestration layer** (the most operationally novel piece).
4. The portfolio-level **validation** that proves the combination beats the better of the two standalones on Sharpe AND on max DD.
5. The portfolio-level **kill criteria** that supersede leg-level kills only at the gross-portfolio layer.

Per the spec hard rule "Must not duplicate the thesis or method of an earlier iteration": the **portfolio-construction thesis is novel** (cross-asset risk-parity carry diversification with correlation gating); the leg-level theses are reused-by-reference, which is the explicit point of an iter-16+ combinatorial candidate.

---

## 1. Thesis (one sentence)

Because BTC perp funding carry and G10 FX carry are economically distinct risk premia (leveraged crypto long demand vs interest-rate differential between fiat policy regimes) and have historically exhibited <0.3 monthly return correlation outside of system-wide liquidity events (March 2020, March 2023), running them as a **risk-parity-weighted pair targeting 8% annualized vol per leg with a 30-day rolling correlation circuit breaker at 0.30** produces a combined portfolio with **Sharpe strictly greater than max(Sharpe_A, Sharpe_B)** and **max drawdown strictly lower than max(MaxDD_A, MaxDD_B)** over the 2021-2026 OOS window — falsifiable within 90 forward days if the combined live Sharpe is not at least 0.2 above the better-performing leg's standalone live Sharpe AND the realized 30-day correlation does not stay below 0.5 in the absence of any liquidity-event regime flag.

## 2. Market / asset class / venue

This iteration spans **two completely disjoint markets** with no cross-margining and no shared counterparty. The portfolio is the union of two account-isolated venues:

- **Leg A (BTC funding carry)** — see `iter_3 §2` for full details. Summary:
  - Short `BTCUSDT` perpetual on Binance Futures (USDT-margined).
  - Long `BTC/USDT` spot on OKX as the cash-and-carry hedge.
  - Failover inventory on Bybit.
  - SDK: `ccxt`.
- **Leg B (G10 FX carry)** — see `iter_4 §2` for full details. Summary:
  - Long-3 / short-3 across `EUR/USD, USD/JPY, GBP/USD, USD/CHF, USD/CAD, AUD/USD, NZD/USD, USD/NOK, USD/SEK`.
  - Primary venue: OANDA via `oanda-py` for AUM ≤ $1M; IBKR IDEALPRO via `ib_insync` for AUM > $1M.
  - SDK: `oanda-py` (oandapyV20) or `ib_insync`.

**New venue introduced by this iteration**: the **orchestration layer** itself — a single off-broker Postgres instance (`portfolio_master`) that holds the canonical leg NAVs, the risk-parity weights, the correlation state, and the cross-broker rebalance ledger. It is not a broker; it is the audit trail and the only authority for sizing the legs. Hosted on the same hardened VPS that already runs the FX leg (Tokyo region) with hot-replica in Frankfurt.

- **Settlement**: each leg settles on its own venue's clock — funding payments every 8h on Binance Futures, T+2 spot rolling for FX with daily tom-next swap. Portfolio NAV is marked end-of-day NY close (21:00 UTC) for the combined risk and correlation calculations, independent of either leg's intra-day clock.
- **Margin**: leg-internal. There is **no cross-margining** between Binance/OKX and OANDA/IBKR. This is a feature, not a bug — it caps single-broker blast radius.
- **Tax regime**: leg-internal as well. The portfolio P&L is the sum of two separately-taxed streams (crypto property gains + FX Section 988 ordinary). The orchestration layer logs both but does not commingle them.

## 3. Time horizon

**Daily portfolio-level rebalance** is the portfolio-defining clock. Inside that:

- Leg A operates on its native 8-hour funding clock (3 events/day) per iter_3 §3.
- Leg B operates on its native monthly rebalance clock (~21 trading days between rebalances) per iter_4 §3.
- The **portfolio overlay** re-evaluates daily at 22:00 UTC (one hour after NY close):
  1. Computes each leg's trailing 60-day realized P&L volatility (annualized).
  2. Computes the trailing 30-day correlation between the two daily P&L streams.
  3. Recomputes the risk-parity gross-notional multipliers for each leg.
  4. Issues a "scale up / scale down / flatten one leg" instruction to the leg controllers, which the legs apply at their next native decision point (next funding window for A, next weekly rebalance check for B).

This means the portfolio overlay is **slow** (daily, with implementation lag of up to 8h for A and up to 5 trading days for B) compared to either leg. That is intentional: a daily-rebalanced overlay on top of daily-vol estimates is well-calibrated; faster would chase noise.

## 4. Method family and specific algorithm

**Family**: Cross-strategy portfolio risk-parity with a correlation circuit breaker. Rules-based, no ML, no LLM, no NN. Sizing is mechanical and observable.

**Leg-internal mechanics are NOT redefined here. See:**
- **iter_3 §4** for the BTC perp funding cash-and-carry algorithm (entry threshold 10% annualized funding; unwind on 1.5% basis blow-out; 8h roll).
- **iter_4 §4** for the G10 long-3/short-3 carry algorithm (monthly rebalance; vol-scaled weights; CVIX-based vol-regime filter at 12%/14%).

**This iteration's portfolio-level overlay algorithm** (deterministic, runs once daily at 22:00 UTC):

```
INPUTS at day t (computed from end-of-day-marked leg P&L series):
  pnl_A[t-59 .. t]   # daily P&L of Leg A (BTC carry), in USD
  pnl_B[t-59 .. t]   # daily P&L of Leg B (FX carry), in USD
  nav_total[t]       # combined account equity across all four broker accounts
  target_vol_leg     # 8% annualized per leg (parameter)
  gross_lev_cap      # 3.0x of total NAV (parameter)
  corr_cutoff        # 0.30 (parameter)
  corr_flatten_pick  # "smaller-vol-budget leg" (rule)

STEP 1 — Realized vol per leg (60-day, annualized):
  sigma_A = stdev(pnl_A[t-59:t] / nav_A) * sqrt(252)
  sigma_B = stdev(pnl_B[t-59:t] / nav_B) * sqrt(252)

STEP 2 — Risk-parity scale factor per leg (vol target / realized vol):
  scale_A = target_vol_leg / sigma_A     # clipped to [0.25, 2.50]
  scale_B = target_vol_leg / sigma_B     # clipped to [0.25, 2.50]
  # clipping prevents the overlay from levering up infinitely after a quiet period
  # or de-levering to dust after a single bad day

STEP 3 — Gross-leverage cap enforcement:
  gross_A = scale_A * default_gross_A    # default_gross_A = 2.0x leg-A NAV per iter_3
  gross_B = scale_B * default_gross_B    # default_gross_B = 2.5x leg-B NAV per iter_4
  total_gross = gross_A + gross_B
  if total_gross / nav_total > gross_lev_cap:
      shrink = gross_lev_cap * nav_total / total_gross
      scale_A *= shrink
      scale_B *= shrink

STEP 4 — Correlation circuit breaker:
  daily_ret_A = pnl_A[t-29:t] / nav_A
  daily_ret_B = pnl_B[t-29:t] / nav_B
  rho_30 = corr(daily_ret_A, daily_ret_B)
  if abs(rho_30) > 0.30 AND (sigma_A > 12% OR sigma_B > 12%):
      # both elevated correlation AND elevated vol = liquidity-event regime
      # flatten the leg with the smaller absolute vol-budget contribution
      vol_budget_A = sigma_A * gross_A
      vol_budget_B = sigma_B * gross_B
      if vol_budget_A < vol_budget_B:
          scale_A = 0  # flatten Leg A at next funding window
      else:
          scale_B = 0  # flatten Leg B at next weekly rebalance check

STEP 5 — Emit instructions:
  publish(scale_A) to leg_A_controller   # consumed at next funding window
  publish(scale_B) to leg_B_controller   # consumed at next weekly rebalance check
  log everything to portfolio_master.daily_overlay table
```

**Why risk-parity and not 50/50 dollar weights**: BTC carry P&L vol (when active) is roughly 3-5x the daily P&L vol of FX carry. A naive 50/50 dollar weight would have BTC dominating the portfolio variance and the diversification benefit would be ~all from Leg B during BTC's quiet periods and ~all from Leg A during FX's quiet periods — i.e. no consistent diversification at all. Risk-parity by realized vol makes each leg's contribution to portfolio variance roughly equal in expectation.

**Why a correlation gate AND a vol gate together (not just correlation)**: spuriously high correlation in a calm regime is not actionable (low vol × high corr = low actual risk). The combined `|rho| > 0.3 AND max(sigma) > 12%` condition is the academic "carry-crash regime" signature — Brunnermeier-Nagel-Pedersen (2008) on carry trade unwinds and the Adrian-Boyarchenko-Shin (2014) intermediary capital model both predict simultaneous vol and correlation spikes during liquidity events. Flatten the smaller-vol-budget leg first because (a) it has the lowest expected dollar P&L cost to flatten and (b) preserving the higher-conviction-currently-active leg's carry stream is more valuable than preserving a leg already running below target.

**Why this fits this edge** (not "because diversification is cool"): the two legs are paid for **bearing different specific risks** — Leg A is paid for warehousing leveraged-long crypto demand at venues that won't or can't hedge it cheaply; Leg B is paid for taking the other side of currency-peg / intervention / monetary-regime risk in fiat. The premia have no shared compensation mechanism. Their *only* shared exposure is the common factor of "global liquidity / risk appetite," which is precisely what the correlation gate is designed to detect and de-risk. Outside of liquidity events, the legs should be near-orthogonal and diversification benefit is mechanical.

## 5. Data requirements

The data feeds are the **union** of iter_3 and iter_4, plus the portfolio-level marks. No new alpha data is introduced.

| Feed | Source | Purpose | Storage | Cost |
|---|---|---|---|---|
| **All Leg A feeds** (see iter_3 §5) | Binance + OKX public REST/WS | Leg A operation | ~80 GB/yr | $0 |
| **All Leg B feeds** (see iter_4 §5) | OANDA REST, FRED, BIS, CBOE | Leg B operation | ~100 MB/yr | $10/mo |
| **Daily leg P&L marks** (NEW, portfolio-level) | computed from each leg's broker `fetch_balance` + open-position MTM | Risk-parity sizing + correlation calc | <10 MB/yr | $0 |
| **Combined NAV ledger** (NEW) | portfolio_master Postgres | Single source of truth across two brokers | <100 MB/yr | $0 (self-host) |
| **Cross-venue stablecoin / wire rate ledger** (NEW) | manual + on-chain explorers | Audit of inter-broker NAV transfers | <1 MB/yr | $0 |

- **Total monthly cost**: ~$10/mo (inherited from FX leg's IBKR data sub if used) + ~$30/mo for the portfolio_master VPS = **~$40/mo**.
- **Total storage**: ~80 GB/yr dominated by Leg A's optional tick logs; the portfolio overlay itself adds <100 MB/yr.
- **Latency requirement**: 60-second cadence for the daily overlay job (no urgency — the overlay runs once a day). 5-minute tolerance on the cross-leg P&L marking job. Sub-second is required only inside Leg A's funding-window decisions per iter_3 §5.

## 6. Validation plan

The validation **must run all three streams in parallel on aligned daily timestamps**: standalone Leg A returns, standalone Leg B returns, and the risk-parity-combined portfolio returns. The combinatorial claim is comparative — diversification benefit is meaningful only if it outperforms the better of the two standalones.

- **Historical windows**:
  - **In-sample (parameter check, no fitting)**: **2015-2020** — chosen because (a) Leg B has clean monthly data back to 1990 but (b) Leg A's BTC perp funding history begins reliably in 2020. We use a synthetic Leg A proxy for 2015-2019 (CME BTC futures basis carry, available from Dec 2017, plus a basis-implied funding back to 2015 from OKEx and BitMEX historical) and the **real** Leg A from Jan 2020. The synthetic-A period is flagged and the verdict requires that the 2020 real-A subperiod alone meet the combined-Sharpe-beats-max-standalone threshold.
  - **Out-of-sample**: **2021-01-01 through 2026-04-30** — 5+ years, fully real data on both legs, no parameter selection done on this window.
- **Walk-forward design at the portfolio level**:
  - Train (overlay parameter check only — target_vol, gross_lev_cap, corr_cutoff are pre-registered from theory; only thing "trained" is the 60-day and 30-day window lengths, which we lock at the values stated): 2015-2020.
  - Test (rolling): 2021-2026 with 12-month folds and 3-month step → ~17 OOS folds.
  - At each fold, recompute the overlay using only train-window data, then apply to test window. The combined-portfolio Sharpe is computed on the concatenation of all test windows.
- **Purge and embargo**: 5 business days between train and test (driven by Leg B's monthly rebalance carryover); independent of Leg A's 7-funding-window purge handled inside Leg A's own validation.
- **Minimum OOS trades**: portfolio rebalances every day, so 5+ years × 250 days = **~1250 daily portfolio observations OOS**. Easily satisfies the per-spec "minimum number of OOS trades."
- **Statistical thresholds — the combinatorial test** (all must pass for the iteration to be promoted to shadow):
  1. **Portfolio annualized Sharpe (OOS, net of all costs and inter-broker bridge friction) > max(Sharpe_A_standalone, Sharpe_B_standalone) by ≥ 0.20**, with 95% bootstrap CI on the **Sharpe difference** excluding zero. Per iter_3 we expect Sharpe_A ≈ 1.5-2.0 and per iter_4 Sharpe_B ≈ 0.55-0.80; combined target Sharpe ≈ 1.8-2.3.
  2. **Portfolio max drawdown (OOS) < max(MaxDD_A, MaxDD_B)** by at least 1 percentage point. Per iter_3 MaxDD_A ≤ 4%; per iter_4 MaxDD_B ≤ 25%; combined target MaxDD < 4% in benign regimes and < 8% across the full OOS window.
  3. **Realized 30-day correlation between the two legs' daily returns has median absolute value < 0.20** across the OOS window. If median |rho| > 0.30 the diversification thesis is empirically false and the overlay should be killed.
  4. **The correlation circuit breaker must fire at least once** in OOS (we expect it to fire in March 2020 if the synthetic-A proxy is used, and in March 2023 [SVB / banking stress] and August 2024 [yen-carry unwind] in the real-A period). If it never fires across 5 years, the rule is untestable in the OOS window and we need a 12-month live shadow before promotion.
  5. **Deflated Sharpe ratio** (López de Prado) on the combined portfolio > 0 after correcting for the joint number of parameter combinations explored across both legs and the overlay. Roughly: the overlay has 3 free parameters (target_vol, gross_cap, corr_cutoff), and we'll have tested ~3 values of each = ~27 combinations on the in-sample. The deflation must reflect this.
- **Anchor evidence**: literature on multi-strategy carry diversification: Koijen-Moskowitz-Pedersen-Vrugt (2018, "Carry"), AQR's "Style Premia" series, and Asness-Frazzini-Pedersen on risk-parity. None of those papers specifically combine crypto-perp funding with G10 FX carry (the premium pair is novel), so the empirical validation is genuinely OOS for the *pair*, not for the individual premia.

## 7. Risk envelope

Portfolio-level envelope, **on top of** each leg's own envelope (which is unchanged from iter_3 §7 and iter_4 §7).

- **Vol target per leg**: **8% annualized realized P&L vol** (each leg sized to this). Two legs at 8% each, with low correlation (<0.3 outside liquidity events), produces portfolio vol of ~10-11% annualized — the diversified target.
- **Portfolio gross leverage cap**: **3.0× of total combined NAV**. With Leg A's natural gross at ~2.0x leg-NAV (per iter_3 §7's 400% gross / 50% cap blend) and Leg B's natural gross at ~2.5x leg-NAV, the unscaled combined gross is already ~2.3-2.5x of total NAV at equal NAV allocation. The cap leaves ~25-30% headroom for the risk-parity overlay to lever up the lower-vol leg without breaching the cap.
- **Correlation cutoff**: 30-day rolling correlation **|rho| > 0.30** combined with **at least one leg's 60-day realized vol > 12% annualized** triggers a single-leg flatten as in §4 STEP 4. Re-entry permitted when both: (a) |rho| has been below 0.20 for 5 consecutive trading days AND (b) the higher-vol leg has dropped below 10% annualized realized vol.
- **NAV-balancing target**: each leg should hold **40-60%** of total combined NAV (i.e. neither leg is allowed to drift below 40% or above 60% of total NAV). If breached, trigger the **inter-broker NAV rebalance** described in §8.
- **Portfolio daily loss cap**: **−1.5% of total combined NAV**. On breach, flatten BOTH legs immediately, regardless of which leg drove the loss. This overrides each leg's individual daily cap when the combined breach hits first.
- **Portfolio drawdown trigger**: **−5% peak-to-trough over any 30-day window** on the combined NAV halts both legs pending review. (Individually iter_3 halts at 3%/7d and iter_4 at 8%/30d; the combined trigger is tighter than max of either to enforce the diversification claim — if the combined loses more than 5% in 30 days, the diversification did not work and we stop.)
- **Single-broker exposure cap (NEW, critical)**: no single broker (Binance OR OKX OR OANDA OR IBKR) holds more than **35% of total combined NAV** at any time. This is stricter than iter_3's 30%-per-venue rule because here the failure of one venue could imbalance the entire portfolio's vol-budget allocation, not just lose that venue's float.
- **Circuit breakers (portfolio-level, in addition to leg-level)**:
  - Cross-broker NAV imbalance > 20% (e.g. one leg has run while the other lost) → trigger inter-broker rebalance per §8.
  - Either broker pair (Binance/OKX OR OANDA/IBKR) becomes inaccessible for > 4 hours → flatten the AVAILABLE leg if the inaccessible leg has any open exposure (avoid unhedged single-leg risk while the other side is dark).
  - Correlation gate fires twice in any 90-day window → escalate to manual review; the rule may be triggering too liberally or the regime may have shifted.
- **Correlation rule with other deployed strategies**: this iteration is one slot. It must not be combined with any other multi-strategy overlay (e.g. a 3-leg or 4-leg risk-parity that re-includes either of these legs). The "one strategy per slot" rule (Lesson 6) applies at the overlay level: this is THE multi-carry stack slot; iter_3 and iter_4 individually are **not deployable separately** if iter_16 is live (they would double-count the legs).

## 8. Execution plan — the dual-broker NAV orchestration layer

This is the operationally novel section. **There is no broker that holds the combined account.** Every dollar lives on either the crypto side (Binance + OKX + Bybit) or the FX side (OANDA or IBKR). The orchestration layer reconciles them.

### 8.1 SDKs and connectivity

- **Crypto side**: `ccxt` (Python) — `ccxt.binance({'options': {'defaultType': 'future'}})`, `ccxt.okx()`, `ccxt.bybit()`. Withdrawals disabled at API-key level on all three. Per iter_3 §8.
- **FX side**: `oanda-py` (`oandapyV20`) for AUM ≤ $1M; `ib_insync` for AUM > $1M. Per iter_4 §8.
- **Orchestrator**: custom Python service `portfolio_overlay.py` running on the same hardened VPS as Leg B's executor (Tokyo region). It reads-only from each leg's broker via the leg's existing SDK and writes only to `portfolio_master` Postgres. It does **not** place trades directly — it publishes scale factors to the leg controllers, which translate them into orders within each leg's own native order-type rules (post-only limit IOC for Leg A entries, marketable-limit IOC for Leg B rebalances).

### 8.2 NAV-balancing rule (critical)

The orchestrator runs the NAV-balance check **daily at 22:30 UTC** (30 min after the overlay job, with leg P&Ls finalized). The rule:

```
INPUTS:
  nav_A = total USD-equivalent equity at Binance + OKX + Bybit (sum)
  nav_B = total USD equity at OANDA or IBKR
  nav_total = nav_A + nav_B
  share_A = nav_A / nav_total
  share_B = nav_B / nav_total

ACTION:
  if 0.40 <= share_A <= 0.60:           # within band, do nothing
      return
  elif share_A > 0.60:                  # crypto leg has run; cash flows OUT of crypto
      target_transfer = (share_A - 0.50) * nav_total   # bring share_A back to 50%
      action = "withdraw USDT from crypto → wire/onchain → deposit to FX broker"
      route = preferred_bridge_route(target_transfer)
  else:                                  # share_A < 0.40; cash flows INTO crypto
      target_transfer = (0.50 - share_A) * nav_total
      action = "withdraw USD from FX broker → wire/onchain → deposit to crypto"
      route = preferred_bridge_route(target_transfer)

  emit_manual_approval_request(target_transfer, route, action)
  # NO AUTOMATIC TRANSFERS. Inter-broker movement always requires human 2FA + email confirmation.
```

**Rebalance frequency**: the band is wide enough (40-60%) that natural rebalance is expected **once every 3-6 months**, not daily. Daily check, infrequent action. This is critical: every rebalance has friction (see §8.3) and we do not want to whipsaw the NAV across brokers.

### 8.3 Inter-broker bridge — cost and latency

Inter-broker NAV transfers are **the most expensive friction in the whole strategy**. They are the reason naive "rebalance to 50/50 daily" would destroy the diversification benefit.

| Bridge route | Latency | Cost (one-way) | Cost (round-trip) | Notes |
|---|---|---|---|---|
| **USDT on TRC20** (crypto → user wallet → fiat off-ramp → ACH/wire → FX broker) | 24-72h end-to-end | ~$1 TRC20 fee + ~0.1% off-ramp spread + $25 wire | **~10-20 bps round-trip** at $100k size; **~5 bps** at $1M size | Default route; needs licensed off-ramp (e.g. Kraken USD, Coinbase USD) as intermediary; user's bank as transit |
| **USDC on Ethereum L1** | 24-48h | ~$10-50 gas + 0.05% off-ramp + $25 wire | **~10-15 bps round-trip** at $100k; **~3 bps** at $1M | Slightly more reliable than TRC20 in regulatory edge cases |
| **Direct broker → broker wire** (IBKR USD → Kraken USD) | 1-3 business days | $25 wire | **~5 bps** at $100k; **<1 bps** at $1M | Fastest fiat path but only works for IBKR side; OANDA retail cannot send outgoing wires same-day |
| **Stablecoin via FX broker** (some FX brokers accept USDC deposits) | 1-2 business days | $0-25 | **2-5 bps** | Not available at OANDA retail; available at some IBKR Prime tiers — verify per account |

**Working assumption for the validation backtest**: **15 bps round-trip friction per inter-broker rebalance**, with expected 2 rebalances per year → **30 bps/year drag** on combined NAV from cross-broker plumbing.

### 8.4 Order types and slippage

Order types are **unchanged from iter_3 §8 and iter_4 §8**. The overlay does not introduce new order types. It only adjusts the *size* of orders each leg places at its native cadence.

- Leg A: post-only limit IOC entries, marketable-limit IOC exits, 5 bps slip budget. (iter_3 §8)
- Leg B: marketable-limit IOC, 1.5 bps slip budget per pair. (iter_4 §8)
- Overlay-driven scaling: when the overlay reduces a leg's scale by, say, 30%, the leg controller closes 30% of each open position at the next native decision point using the leg's native exit order type. No new orders, no overlay-driven market orders.

### 8.5 Order rejection cascade

- If Leg A rejects its overlay-scaled order at a funding window → leg falls back to its **previous** scale factor (not zero — partial fills are acceptable); orchestrator alerts and logs.
- If Leg B rejects its overlay-scaled order at the monthly rebalance → leg falls back to previous month's positions; orchestrator alerts.
- **Critical asymmetry**: an order rejection NEVER causes the overlay to attempt to re-route the order through the other broker. The legs are isolated by design. A failed Leg A rebalance does not become a Leg B over-allocation.

### 8.6 Reconciliation cadence

- **Per-broker reconciliation**: each leg reconciles its own positions against its broker every funding window (Leg A) or 60s during rebalance / hourly during hold (Leg B). Unchanged from leg specs.
- **Portfolio reconciliation (NEW)**: every 5 minutes, the orchestrator pulls `fetch_balance()` from each crypto venue and `account_summary()` from the FX venue, sums to `nav_total`, and reconciles against `portfolio_master.nav_total_marked`. Discrepancy > 0.5% triggers a halt-all and an operator page (Lesson 4 applied at the portfolio level: never trust the local cache when computing risk-parity weights).

## 9. Capacity

- **AUM ceilings (inherited from legs)**:
  - Leg A standalone: ~$50M before BTC perp depth degrades carry (iter_3 §9).
  - Leg B standalone: ~$1M at OANDA, ~$50-100M at IBKR IDEALPRO (iter_4 §9).
- **Combined AUM ceiling**: bounded by the **smaller** of (2× Leg A ceiling) and (2× Leg B IBKR ceiling) at the 50/50 NAV allocation = **min($100M, $100-200M) = ~$100M total combined AUM** before either leg starts paying execution drag that exceeds the diversification benefit.
- **Below $1M AUM, the strategy is dominated by inter-broker friction** — at $200k AUM and 15 bps round-trip bridge, each rebalance costs $300, and the strategy needs 2-3 rebalances/year × $300 = ~$900 = 45 bps/year drag, which is meaningful against a target Sharpe of 1.8 at 10% vol. Practical lower bound: **$500k combined AUM**.
- **ADV consumption at the combined ceiling**:
  - Leg A: 0.15% of BTC perp 1-min ADV at $50M (iter_3 §9).
  - Leg B: 0.00075% of G10 spot ADV at $50M (iter_4 §9).
  - Combined: still negligible. Capacity is purely broker-execution-bounded, not market-impact-bounded.
- **Concentration**: by construction, ≤ 50% in each of two uncorrelated premia. Within Leg A: 100% BTC. Within Leg B: ≤ 25% per FX leg. No single instrument > 12.5% of combined NAV at full size.

## 10. Failure modes

Three categories: leg failures (delegated), portfolio failures (new), operational failures (new).

### 10.1 Leg-internal failures
See `iter_3 §10` (funding regime flip, exchange insolvency, basis blow-out, regulatory action, stablecoin depeg, API outage) and `iter_4 §10` (carry crash, monetary regime change, month-end liquidity flight, broker swap-rate drift, FX intervention shock). These are unchanged and not duplicated here.

### 10.2 Portfolio-level failure modes (NEW)

1. **Correlation regime shift — the legs converge** (e.g. March 2020 COVID liquidity crunch, March 2023 SVB / banking stress, August 2024 yen-carry unwind compounded with crypto sell-off). When global liquidity tightens, both leveraged crypto long demand AND high-yield FX carry trades unwind simultaneously. The diversification benefit collapses precisely when it would be most needed.
   - **Metric signature**: 30-day |rho| jumps from <0.2 baseline to >0.4 within 5 days; both legs' realized vol simultaneously > 12%; both legs P&L drawdown synchronously.
   - **Mitigation**: the §4 STEP 4 correlation circuit breaker fires and flattens the smaller-vol-budget leg. This caps the joint downside at roughly one-leg standalone drawdown. The strategy then sits with one leg until correlation normalizes (both: 5 days < 0.20 AND higher-vol leg < 10% realized).
   - **Residual risk**: the breaker is reactive, not predictive — by the time |rho| > 0.30 the joint loss may already be 2-3% of combined NAV. This is acknowledged and the OOS validation must show that even with this lag the combined max DD is below max(MaxDD_A, MaxDD_B).

2. **One leg's edge disappears (slow regime shift)** — e.g. BTC perp funding stays at 0-3% for 12 months as crypto retail leverage normalizes; or G10 short-rate dispersion compresses as all central banks converge to zero (the 2010-2014 ZIRP regime). The leg quietly stops earning while still consuming capital.
   - **Metric signature**: 12-month rolling annualized return for one leg < 2% AND realized vol > 6% (high vol with no return = pure noise).
   - **Mitigation**: monthly review — if a leg's 12-month Sharpe drops below 0.3, allocate that leg's capital share down to a 10% floor (the leg becomes a "tracker" so we keep the option to re-engage but stop committing meaningful capital). After 18 months at < 0.3 Sharpe, decommission that leg entirely and re-evaluate the slot.

3. **Broker outage on one side forces unbalanced exposure** — e.g. Binance maintenance window extends from 4h scheduled to 36h actual; we cannot rebalance Leg A while Leg B continues to operate normally. The portfolio drifts from 50/50 to (say) 35/65 over a single month, breaching the 40/60 band. Worse case: if Leg A had an open hedged position and one of its two legs (perp short or spot long) becomes inaccessible while the other doesn't, Leg A becomes directionally exposed.
   - **Metric signature**: any broker `fetch_*()` call fails for > 4 hours consecutive; or share_A or share_B drifts outside [0.40, 0.60] for > 7 days without a manual rebalance.
   - **Mitigation**: per §7 circuit breaker, if a broker is dark > 4h while the other is up, flatten the up-broker leg to prevent unhedged single-leg risk. Loss of 2-3 days' carry on the up-broker is preferable to running a directional bet during the dark window.

4. **Inter-broker bridge becomes unavailable** — e.g. major off-ramp (Kraken, Coinbase) suspends USDT withdrawals; or the user's bank flags an FX-broker wire as suspicious and freezes it; or stablecoin networks (TRC20, ERC20) experience prolonged congestion. The NAV-rebalance mechanism is broken and the legs cannot be re-leveled.
   - **Metric signature**: ≥ 2 attempted rebalance routes fail or stall > 5 business days.
   - **Mitigation**: tolerate temporary NAV drift up to 70/30 (one-time relaxation); if the imbalance exceeds 30/70 OR persists > 30 days, decommission the smaller-side leg entirely until the bridge is restored.

5. **Risk-parity overlay becomes degenerate** — e.g. during a period of unusually low realized vol (the calm of late 2017 or mid-2019), the overlay levers both legs up aggressively (the [0.25, 2.50] clipping in §4 STEP 2 hits the upper bound) only to face a sudden vol regime change before the daily overlay can react.
   - **Metric signature**: both legs' scale factors at the 2.50 ceiling for > 30 consecutive days; combined gross leverage at the 3.0x cap.
   - **Mitigation**: the 3.0x gross leverage cap in §4 STEP 3 is the hard backstop. Additionally, when both scales are pinned at the upper bound for > 14 days, add an "anti-Madoff" rule: shrink both scales by 20% as insurance against the false sense of safety from a quiet regime.

### 10.3 Operational failure modes (NEW)

Covered in §12; not duplicated here.

## 11. Falsifiable verdict criteria

These trigger within **90 days of live deployment**, after walk-forward + ≥30-day shadow has already passed.

**Promote to full live capital** (after combined walk-forward OOS + ≥ 30-day shadow + ≥ 60-day live $50k-scale run):
- **Diversification benefit empirically confirmed**:
  - Live combined Sharpe ≥ max(live Sharpe_A_during_period, live Sharpe_B_during_period) + 0.20.
  - Live combined max DD ≤ max(live MaxDD_A, live MaxDD_B) − 1pp absolute.
- **Correlation regime stable**: realized 30-day |rho| stayed below 0.50 in absence of any liquidity-event regime flag; or the circuit breaker fired correctly during any regime that did occur.
- **NAV-balance mechanism worked**: at least one inter-broker rebalance executed end-to-end within the published friction budget (≤ 20 bps round-trip at live scale).
- **Zero broker-truth divergences > 0.5%** across both legs over the live window.
- **No unhedged single-leg episodes** caused by broker outages > 5 minutes.
- **No portfolio daily loss cap breach** (−1.5%) over the live window.

**Continue shadow / extend test**:
- Combined Sharpe between max-of-legs and max-of-legs + 0.20 (diversification visible but small); no kill triggers hit; extend shadow another 30 days; re-evaluate.
- Correlation circuit breaker has not fired and no liquidity event regime has occurred → extend until at least one regime test.

**Kill immediately** (any one):
- **Diversification thesis broken**: combined Sharpe < min(Sharpe_A_standalone, Sharpe_B_standalone) over any 60-day live window (the combination is worse than either leg alone — the overlay is value-destructive).
- **Combined max DD > 8%** at any point in the live 90-day window.
- **30-day combined drawdown > 5%** triggers the portfolio-level halt of §7 and never auto-resumes.
- **Realized 30-day |rho| stays > 0.50** for > 30 consecutive days regardless of liquidity event flag — the legs are correlated and the thesis is false.
- **NAV-balance broken**: cross-broker share imbalance > 60/40 persists > 30 days because rebalance plumbing failed; OR a rebalance executes at > 30 bps round-trip cost (4× the budget).
- **Single-broker exposure** breaches the 35% cap and is not corrected within 5 business days.
- **Any unhedged single-leg episode** lasts > 5 minutes due to broker outage AND the surviving-leg flatten rule failed to fire.
- **Inter-broker reconciliation discrepancy** (portfolio_master vs sum-of-broker-balances) > 0.5% on any 5-minute reconciliation cycle.

## 12. One-paragraph operational risk note

The orchestration layer is **the single most operationally exposed component in the entire system**: it lives outside both brokers, it depends on read-only API access to four separate broker accounts across two SDKs, and it is the only authority for sizing decisions that drive trades on both sides. A bug in the overlay (e.g. swapped scale_A / scale_B assignment, or a numpy NaN propagated through the vol calculation, or a Postgres connection drop during the daily 22:00 UTC write) can simultaneously direct both legs into incorrect sizing — the failure mode is **silent and bilateral**. Specific risks: (a) the orchestrator runs on a single VPS with hot-replica failover, but failover during the 22:00 UTC overlay job is untested — failover testing must be a monthly fire drill; (b) clock skew across the four broker endpoints can cause the P&L marks to be off by minutes — the overlay must use the orchestrator's own NTP-synced clock as the canonical timestamp, not any broker's; (c) the **inter-broker NAV transfers cannot be fully automated for security reasons** (no API can move money between independently-owned broker accounts without human approval), so a manual 2FA + email + treasurer-sign-off process is mandatory, which means rebalance latency is bounded below by the slowest human in the loop, not the slowest network; (d) Postgres `portfolio_master` is the single source of truth and a corruption / accidental DROP TABLE there is catastrophic — daily encrypted backups to S3 + WAL streaming to the hot-replica are mandatory and must be restore-tested quarterly; (e) **24/7 obligation** is now even harder than for either leg alone: Leg A's funding windows do not pause for the orchestrator's downtime, so the overlay must publish a "last-known-good scale factor" that the leg controllers continue to apply if the overlay is down, with a hard "after 24h of overlay downtime, flatten both legs" failsafe — this failsafe must be implemented inside each leg controller, not inside the orchestrator (because the orchestrator is precisely the thing that's down).

---

## Addressing the 10 Lessons

1. **Validate before deploy** — Walk-forward 2015-2020 in-sample (with synthetic-A flagged for 2015-2019, real-A 2020), 2021-2026 OOS, then 30-day shadow at $50k-scale, then 60-day live $50k-scale run, then scale. The combinatorial Sharpe-and-DD test in §6 thresholds 3-5 are explicit numeric gates. Honored.
2. **Binary gates, not confidence rankers** — The correlation circuit breaker is a deterministic threshold (`|rho| > 0.30 AND max(sigma) > 12%`), not a continuous probability score sizing positions. The risk-parity scale factors are *deterministic functions* of realized vol, not ML confidences. No ML at any level of the overlay. Honored.
3. **TP:SL asymmetry is structural** — At the portfolio level, the asymmetry is: the overlay lets winners compound (risk-parity scales up the leg that's been winning *only as its vol normalizes*; it does not aggressively chase performance), but it cuts losers fast (correlation gate flattens one leg the day after detection; daily 1.5% combined loss cap). Asymmetric by construction; carries through from each leg's own asymmetric design (iter_3 §11 1.5% basis blow-out cap with unbounded carry upside; iter_4 §10 vol filter scaling down with no symmetric upside cap). Honored.
4. **Broker truth, not local cache** — Two layers: (a) each leg reconciles its own broker per leg spec; (b) the orchestrator's 5-minute portfolio reconciliation compares `portfolio_master.nav_total_marked` against the sum-of-broker-balances pulled via `ccxt.fetch_balance()` and OANDA/IB `account_summary()`. Discrepancy > 0.5% triggers halt-all. Honored at both layers.
5. **No PDT-exposed lanes on sub-$25k US equity accounts** — **N/A by composition**: Leg A is crypto (no FINRA jurisdiction); Leg B is FX (no FINRA jurisdiction). Neither leg is a US-equity margin account, so PDT does not apply at either leg level or at the portfolio level. The choice of these two asset classes for this iter-16 combination is partly *because* both are PDT-exempt, making the combined strategy deployable from accounts of any size. Honored by exemption.
6. **One strategy per slot** — This iteration is the unique "BTC funding × FX carry risk-parity stack" slot. Critical: **iter_3 and iter_4 must NOT be deployed in separate slots while iter_16 is live**, because that would double-count the legs and over-allocate capital to these two premia. The slot allocator must enforce that iter_16's deployment supersedes any standalone deployment of iter_3 or iter_4. Honored with explicit enforcement.
7. **No same-data backtest** — Walk-forward at the portfolio level (12-month folds, 3-month step, ≥17 OOS folds in 2021-2026), with a 5-day purge between train and test. Overlay parameters (target_vol=8%, gross_cap=3.0, corr_cutoff=0.30, windows=60/30 days) are **pre-registered from theory** (risk-parity literature, Brunnermeier-Nagel-Pedersen), not tuned to the OOS sample. The deflated Sharpe test (§6 threshold 5) corrects for in-sample parameter search. Honored.
8. **Battle-tested broker SDKs only** — `ccxt` for crypto leg (per iter_3 §8), `oanda-py` or `ib_insync` for FX leg (per iter_4 §8). The orchestrator is custom code but does not place orders directly — it only reads via the same battle-tested SDKs and writes to a standard Postgres. No hand-rolled HTTP anywhere in the trading path. Honored.
9. **Capacity bound named upfront** — Lower bound $500k combined AUM (below which inter-broker friction dominates); upper bound ~$100M combined (bounded by the smaller of the two legs' execution capacity per iter_3 §9 and iter_4 §9 at IBKR tier). Stated in §9. Honored.
10. **Kill criteria written before deploy** — §11 lists exact numeric kill triggers for the portfolio (combined Sharpe < min-of-legs, combined MaxDD > 8%, |rho| > 0.50 sustained, NAV-balance broken at > 30 bps cost, single-broker > 35% cap breach, unhedged-leg episode > 5 min). All triggerable within 90 days of live deployment. Honored.

---

## Innovation matrix cell occupied

- **Asset class**: BTC perpetuals + spot (Leg A) × G10 spot FX (Leg B) — **two disjoint asset classes in one strategy**.
- **Geography**: Crypto (global, jurisdiction-diverse: Binance/OKX/Bybit) × Global FX (G10, executed via OANDA Tokyo/NY or IBKR IDEALPRO).
- **Time scale**: Daily portfolio-level rebalance overlaying an 8-hour native leg cadence (Leg A) and a monthly native leg cadence (Leg B). **Multi-clock combinatorial.**
- **Method**: Rules-based risk-parity overlay with correlation circuit breaker; underlying legs are also pure rules-based (no ML).
- **Edge source**: Diversification across two **economically distinct carry premia** (leveraged crypto long demand + interest-rate differential between fiat policy regimes), gated by a liquidity-event detector (correlation + vol joint threshold). The combinatorial edge is not a new premium but the empirical claim that the joint Sharpe exceeds max-of-legs and the joint DD undershoots max-of-legs.

---

THESIS: Risk-parity stacking BTC perp funding carry with G10 FX carry, both vol-targeted to 8% per leg with a 30-day |rho|>0.3 + vol>12% correlation circuit breaker, produces a combined OOS Sharpe at least 0.2 above the better-performing leg standalone and a combined max DD at least 1pp below the worse-performing leg standalone — over 2021-2026 OOS — and remains net positive after 30 bps/year inter-broker bridge friction.
VALIDATION HORIZON: ~1250 daily portfolio observations OOS (2021-2026, 5+ years across 17+ walk-forward folds) + 30-day shadow at $50k-scale + 60-day live $50k-scale run before full-scale promotion = ~90 OOS live days inside the falsification window.
CAPACITY CEILING: ~$100M total combined AUM (bounded by the smaller of 2× Leg A ceiling at $50M and 2× Leg B IBKR ceiling at $50-100M, at 50/50 NAV allocation); lower bound ~$500k AUM below which inter-broker friction dominates diversification benefit.
