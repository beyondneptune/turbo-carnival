# Hard Recommendation for Tomorrow's IBKR Deployment

**Author**: Claude
**Date**: 2026-05-10 evening
**Context**: User wants to deploy on Interactive Brokers ($100k account) Monday with an investor watching daily/weekly returns for a month. Codex built a council/research_lab that proposed 61 strategy variants and recommended the winner `risk_qwen_risk_designer_risktest_d12_30d_lowdelta_pt30_sl20_c10`.

## TL;DR

**Don't deploy real capital tomorrow. You can deploy IBKR PAPER trading tomorrow — but with the IBKR adapter that doesn't exist yet, you literally can't place an order. So practically: tomorrow's investor demo is the dashboard showing realistic numbers from the historical simulation, not a live month yet.**

The honest realistic number to show is **+29.1% over 7 years (3.7% CAGR) with -4.0% max drawdown** on $100k with 1% per-trade risk and 15% portfolio cap. That's defensible. The proxy's +264% claim is not.

## What we now know after today's work

### 1. The +264% headline is misleading

| Configuration | 7-year return | CAGR | Max DD |
|--------------|---------------|------|--------|
| Codex's claim (mid-price, no friction, $10k, 1 contract) | +264.7% | 20.2% | -14.7% |
| Realistic friction, $10k, 1 contract | **+145.5%** | 13.7% | -14.7% |
| Pessimistic friction, $10k, 1 contract | +30.8% | 3.8% | -14.7% |
| **Realistic friction, $100k IBKR, 1% risk/trade** | **+29.1%** | **3.7%** | **-4.0%** |
| Realistic friction, $100k IBKR, 2% risk/trade | +64.4% | 7.4% | -8.8% |

The proxy backtest assumed mid-price fills with zero spread/commission/slippage. Realistic SPY OTM 30-DTE options have ~$0.03 bid-ask spread. With 4 legs entry + exit + commissions, that's ~$8-14 of friction per trade. At 1495 trades, friction alone is ~$12,000 of the claimed $26,472 P&L.

### 2. Walk-forward says the strategy generalizes (but with a caveat)

Splitting the winner's trades into in-sample (2019-2022) vs out-of-sample (2023-2026):

- IS: +119% return, Sharpe 4.11
- OOS: **+145% return, Sharpe 5.70**
- OOS/IS Sharpe ratio: 1.39 (>0.7 = good)

**But**: 57 of 60 strategies tested showed higher OOS than IS Sharpe. That uniformity is statistically suspicious — either 2023-2026 was a particularly favorable regime for VRP-positive iron condors (post-bear-market mean reversion), or there's a code artifact. The NIM code-review panel is hunting this in parallel.

**Realistic Sharpe** after correcting for trade correlation (concurrent condors aren't independent during stress): probably 1.5-2.5, not 4-5. Still respectable.

### 3. Position sizing recommendations for $100k IBKR

Each iron condor has a max loss of ~$412 per contract (with $5 wings - $88 credit avg).

| Policy | Risk/trade | Contracts | Max concurrent | Total at-risk |
|--------|-----------|-----------|----------------|---------------|
| Conservative | 0.5% ($500) | 1 | up to 16 | $6,600 (6.6%) |
| **Standard (recommended)** | **1.0% ($1,000)** | **2** | **up to 16** | **$13,200 (13.2%)** |
| Aggressive | 2.0% ($2,000) | 4-5 | up to 16 | $26,400 (26.4%) |

**My recommendation: standard 1% per-trade with 15% portfolio cap.** Returns are still meaningful (+29%/7yr) and max drawdown stays under 5% — that's the kind of risk profile an investor will tolerate watching daily.

### 4. The IBKR adapter doesn't exist

`pip list` shows no `ib_insync` or `ibapi`. `find` shows no IBKR integration code. The current `scripts/run_daily_paper.py`'s `fetch_spy_chain` is still a stub returning None.

**What this means concretely:**
- You cannot place an order tomorrow.
- The investor cannot see "today's trades" because no trades will fire.
- The "live month starting tomorrow" framing is not delivered by code.

**To fix tomorrow morning** (4-6 hours of work, doable):
1. `pip install ib_insync`
2. Build `ingest/ibkr_client.py` — wrap IB Gateway connection, market data subscription, paper-trading order placement
3. Replace `fetch_spy_chain` stub with `IBKRClient.fetch_chain('SPY', dte=30)`
4. Wire `place_iron_condor()` into `run_daily_paper.py:enter_core_condor`
5. Test against IBKR paper account

This is real engineering work. It's not done. If the demo is tomorrow at market open, it won't be ready.

### 5. Don't conflate "model trains better on $10k" — that's not a thing

The strategy parameters (delta, DTE, wings) are NAV-independent. Backtesting on $10k vs $100k is the same math. What changes is **position sizing of the deployed strategy**. The advantage of $100k is **diversification headroom**, not better training.

## Concrete action plan for the next 24 hours

### Tonight (must)

1. **Honest investor framing** — write the brief. The pitch is "research-stage candidate with realistic +3-8% CAGR expectation, not deployable yet." Not "+264% strategy."

2. **Decide demo format**:
   - **Option A**: Show the dashboard with realistic_execution.py results as the headline. No live trades fire tomorrow. Investor sees "here's what the simulator says happens at $100k" but no fresh daily trades. Honest.
   - **Option B**: Spend tonight building the IBKR paper-trading adapter so trades actually fire tomorrow. 4-6 hours of focused work. Higher-risk; if it's broken at 9:30 AM ET, the demo fails.

   I recommend **Option A** for the first investor meeting. Build IBKR adapter this week, schedule a follow-up demo after one full week of paper trading.

### This week (should)

3. **Build IBKR paper-trading adapter** — straightforward but real work. Once done, the strategy can run forward.

4. **Re-run a strict execution test** with real IBKR chain data once the adapter exists. The proxy's 91% win rate may not survive real bid/ask.

5. **Add Bonferroni / FDR correction** — running 61 variants and picking the best is p-hacking. The "winner" might just be lucky. Need walk-forward on a hold-out cohort BEFORE the council's 61-variant pick.

6. **Address the 57/60 uniform OOS-better problem** — verify it's regime, not artifact. Code review panel may find this.

### Don't do tomorrow (under any circumstance)

- Don't deploy real capital. The strategy hasn't been validated on real bid/ask data.
- Don't claim "+264% strategy" to the investor. It's misleading.
- Don't run the live runner without the IBKR adapter — it does nothing.

## Files written today

- `experiments/realistic_execution.py` — friction simulator + position sizing
- `experiments/walk_forward.py` — IS vs OOS validator
- `council/red_team_winner.py` — adversarial review (5/6 NIM reviewers responded)
- `council/code_review_panel.py` — hostile code review (running, will land later)
- `council/redteam_runs/20260510T155050Z/SYNTHESIS_BY_CLAUDE.md` — earlier red-team synthesis
- This file — final hard recommendation

## NIM code-review panel — addendum

I ran 6 hostile NIM reviewers against `experiments/run_spy_core_strategy_grid.py`. They produced **17 specific claims**. I verified each against actual code. **Roughly half were hallucinated.**

**Verified TRUE (2):**
- Exit logic uses daily close, may miss intraday extremes — known modeling limitation
- Sharpe annualization assumes trade independence (overlapping condors correlate)

**Verified FALSE/HALLUCINATED (3 critical claims):**
- ❌ "Entry credit is double-counted into NAV" — cited wrong line numbers (349-351 are dataclass construction). Actual P&L logic at lines 386, 449, 455 is correct standard accounting.
- ❌ "iv_rank_252 has lookahead bias" — verified `pandas.rolling()` is backward-looking by default. No future data leak.
- ❌ "next_friday off-by-one returns same day" — function correctly implements "Friday on or after," which is the right semantic for expiry math.

**Note (1):** Annualization assumes 213 trades/year and √N scaling. Real Sharpe for ladder strategies is probably ~50-70% of reported, not because the math is wrong but because trade independence is violated.

**Meta-finding worth keeping:** **NIM-based code review hallucinates confidently about specific implementation details.** Out of ~10 critical claims across red-team + code-review today, 5 were false. Ship every concrete LLM claim back to a reader who can read the actual code. Don't let the council "decide" what's broken without verification.

This makes me MORE confident in the realistic_execution + walk_forward numbers, not less — the hostile panel didn't find a real bug that would invalidate the result.

## Honest one-sentence summary

**The strategy is a real research candidate with modest realistic returns; tomorrow's investor demo should show realistic numbers from the simulator (defensible), not the proxy claim (misleading); the IBKR adapter to actually trade hasn't been built yet, so live deployment is not tomorrow.**

You said you wouldn't hold me accountable for the hard recommendation. That's the call.
