# Live vs Yesterday-Only Simulation Diagnosis

Date: 2026-05-12

## Verdict

The high-return simulations from yesterday were not a deployable signal. They were mostly one-day counterfactuals selected after seeing yesterday's tape. Today's live behavior is consistent with the stricter evidence already in the repo: the existing HFT signal family has weak or negative edge after realistic slippage, and the $1k TERMINATOR HFT idea is structurally blocked by PDT.

## Evidence

1. The handoff file already warned that realistic HFT sims failed.

   `HANDOFF_TO_CODEX.md` says that at realistic 8 bps slippage every existing strategy variant loses 20-80% over 21 days, or trades zero times when filters are stacked.

   Latest swarm lab summary:

   - Aegis baseline: -41.33% return, -10.24 bps average trade.
   - Falcon baseline: -77.36% return, -14.12 bps average trade.
   - Aegis + vol-of-vol gate: -23.84% return.
   - Only rank-1 was a one-trade opening-drift fade, not a repeatable HFT lane.

2. The optimistic yesterday reports were bar-close counterfactuals.

   `reports/pass_gate_yesterday/index.html` evaluates yesterday's 162 matched entries. The page reports:

   - Bar-close P&L: +$496.01.
   - Real fills with slippage: -$3,443.

   That gap is the core issue. Bar-close entries/exits made the tape look profitable while executable fills lost money.

3. The no-lookahead report measured meaningful bias.

   `reports/no_lookahead_yesterday/index.html` says:

   - Composite C3 original, trained with yesterday included: +$2,029.53.
   - Composite C3 retrained without yesterday: +$1,287.70.
   - Lookahead premium: +$741.84.

   The no-lookahead number was still better than baseline, but it was still a one-day test and should not have been treated as a live strategy proof.

4. The $1k TERMINATOR simulation ignored a live trading constraint.

   `reports/1k_simulation/index.html` explicitly says it scales yesterday's bar-close P&L down 100x and does not include realistic live execution. In live paper trading, the $1k account hit Alpaca PDT protection and could not close same-day HFT exits.

   Result: $1k HFT is not a valid lane unless account equity is above $25k or the strategy does not require same-day round trips.

5. Today's live account state confirms the mismatch.

   As of roughly 10:35 EDT:

   - Aegis and Falcon are trading, but not showing the simulated edge.
   - Valk is the cleanest live structure after repair: 4 legs, 22 contracts.
   - Hermes is behaving as designed as the SVXY core lane.
   - Atlas is behaving as designed as the stricter flat control lane.
   - TERMINATOR is blocked from HFT by PDT and should not be treated as an HFT probe at $1k.

## Root Cause

The mistake was not one bug. It was an evidence-quality failure:

- Too much weight was placed on a single day.
- Too many model/gate variants were searched on the same tape, creating multiple-testing/selection bias.
- Some results used bar-close counterfactual P&L instead of executable fills.
- Some models were trained or selected using data too close to the evaluation day.
- The $1k HFT lane ignored the broker's PDT constraint.
- The live host initially made lanes look different by capital size more than by actual hypothesis.

## Deployment Rule Going Forward

Do not deploy or scale a strategy because it wins yesterday.

A strategy can be considered for live paper only if it passes all of these:

- Walk-forward or leave-one-day-out validation, not same-day selection.
- Realistic spread/slippage model, with explicit stress at 8 bps and 15 bps.
- No unresolved broker/account constraint such as PDT, shorting, options permission, or minimum quantity.
- Distinct lane hypothesis, not the same model under a different account name.
- Live shadow mode first, with observed fill quality logged.
- One-day results may create a research lead, not a deployment decision.

## Current Corrective Action

- Aegis and Falcon are now distinct and trading.
- Valk monitor now uses live position quantity, avoiding stale 44-contract close logic after the 22-contract scale-down.
- SVXY lanes are now distinct: Hermes core long, Atlas strict/deep-contango control, TERMINATOR no SVXY overlay.
- TERMINATOR HFT is blocked by a PDT guard below $25k. It should be reassigned after close to a non-day-trade strategy or funded above PDT threshold.

