# SVXY Backtest Forensic Audit — Why "+17.1% / Sharpe 0.66" did not survive honest re-run

**Auditor**: Forensic Audit Agent A13
**Date**: 2026-05-12
**Question**: The repo's headline figure "+17.1% annualized / Sharpe 0.66" for the SVXY rotation strategy collapsed to **+0.38% / Sharpe 0.09** in tonight's honest re-run. Find the bug.

---

## Headline finding (TL;DR)

The +17.1% figure is **not a lookahead bug, not a hallucination, and not corrupted data**. It is the genuine output of the script at `experiments/svxy_lab/01_walk_forward_2019_2026.py` under its own parameters.

The +0.38% figure is the genuine output of a **different, much more conservative strategy spec** (`research/scripts/iter_5_backtest.py`) on a **larger time period** (2018-2026 vs 2019-2026).

The "bug" is methodological — a deployment-grade spec was written that diverges from the original validation in five material ways (sizing, threshold, circuit breakers, costs, time period). The original number was then carried forward in commit messages, README claims, and the live-trading host docstring (`scripts/svxy_rotation_host.py:3`) as if it validated the spec it does not validate.

| Variant | Period | Threshold | Sizing | Breakers | Costs | Cash leg | Ann return | Sharpe |
|---|---|---|---|---|---|---|---|---|
| Original walk-forward (as found) | 2019-01-01 → 2026-05-08 | `<1.0` | 100% SVXY | none | 0 | 0% (flat) | **+17.10%** | **0.66** |
| Original logic on 2018-2026 (period-only change) | 2018-01-01 → 2026-05 | `<1.0` | 100% SVXY | none | 0 | 0% (flat) | +9.73% | 0.454 |
| iter_5 spec on 2019-2026 (spec-only change) | 2019-01-01 → 2026-05 | `<0.92` | 25% SVXY | VIX>30 OR +50% | 1bp | SGOV ~5% | -0.63% | -0.079 |
| **iter_5 spec on 2018-2026 (the honest re-run)** | 2018-01-02 → 2026-05-11 | `<0.92` | 25% SVXY | VIX>30 OR +50% | 1bp | SGOV ~5% | **+0.38%** | **0.09** |

The Sharpe is INVARIANT under linear position sizing — so the drop from 0.66 → 0.09 is **not** about going from 100% to 25%. The Sharpe collapse comes from the **threshold change (1.0 → 0.92)** and **circuit breakers that flatten on VIX>30**, both of which eliminate the days that contributed most of the historical return.

---

## Files examined

### Original "validated" backtest
- `/Users/weeoto/turbo-carnival/experiments/svxy_lab/01_walk_forward_2019_2026.py` — the script that produced +17.1% / 0.66
- `/Users/weeoto/turbo-carnival/experiments/svxy_lab/02_render_simulation_report.py` — HTML rendering
- `/Users/weeoto/turbo-carnival/experiments/svxy_lab/runs/summary.json` — cached output (total_return 218.2%, ann 17.1%, Sharpe 0.66, max DD -32.8%)
- `/Users/weeoto/turbo-carnival/experiments/svxy_lab/runs/walk_forward_results.csv` — daily equity curve
- `/Users/weeoto/turbo-carnival/experiments/svxy_lab/runs/walk_forward_results.parquet`

### Honest re-run (tonight)
- `/Users/weeoto/turbo-carnival/research/scripts/iter_5_backtest.py` — script that produced +0.38% / 0.09
- `/Users/weeoto/turbo-carnival/research/results/iter_5_summary.json`
- `/Users/weeoto/turbo-carnival/research/results/iter_5_daily_portfolio.csv`
- `/Users/weeoto/turbo-carnival/research/logs/iter_5_backtest.log`

### Strategy spec & live host
- `/Users/weeoto/turbo-carnival/research/candidates/iter_5_vix_term_structure_rotation.md` — the deployment spec (§9 cites "+17.1% Sharpe 0.66" as the prior validation, line 93)
- `/Users/weeoto/turbo-carnival/scripts/svxy_rotation_host.py` — the LIVE trading host whose docstring (line 3) claims "Base strategy (validated walk-forward 2019-2026: +218% / +17.1%/yr / Sharpe 0.66)"
- `/Users/weeoto/turbo-carnival/research/candidates/RANKED_SHORTLIST.md` — operator-profile blurb (line 3) restates "+17.1% / Sharpe 0.66" as validated baseline

### Postmortems / verdicts (today)
- `/Users/weeoto/turbo-carnival/reports/mistakes_and_errors_postmortem_20260512.md`
- `/Users/weeoto/turbo-carnival/research/results/OVERNIGHT_VERDICT_20260512.md` — already flagged this strategy as FAIL (lines 14-15, 26-30)

---

## STEP 3 audit findings, item by item

### a. Lookahead bias

**No same-day lookahead in the original walk-forward.** Verified at `experiments/svxy_lab/01_walk_forward_2019_2026.py:141-144`:

```python
# Walk-forward strategy: at day T, decide based on ratio[T-1] (no lookahead),
# take position at SVXY[T] open ≈ SVXY[T-1] close (approximation)
df["signal"] = df["contango"].shift(1).fillna(False)  # signal known at T from T-1
df["strategy_ret"] = np.where(df["signal"], df["svxy_ret"], 0)
```

`signal[T] = contango[T-1]` so the position for day T is set from yesterday's close-of-day ratio. The strategy earns `svxy.pct_change()` from T-1 close to T close while holding that signal. **This is a legitimate close-to-close framing — no lookahead present.**

(I confirmed empirically by re-running the original logic from a fresh yfinance pull and reproducing 17.01% / 0.659, within rounding of the cached 17.10% / 0.662.)

The OVERNIGHT_VERDICT_20260512.md claim (line 28) that "+17.1% only emerges when same-day signal earns same-day return (lookahead)" is **incorrect**. The original number does NOT require lookahead. It comes from materially different parameters.

### b. SVXY leverage (-1× pre-Feb-2018 vs -0.5× after)

**Period start 2019-01-01 sidesteps the issue.** The original START_DATE in `01_walk_forward_2019_2026.py:35` is `"2019-01-01"`, which is AFTER ProShares cut SVXY leverage to -0.5× (Feb 2018). The walk-forward therefore uses only the post-change regime.

But: this also means the strategy was **never stress-tested through Volmageddon (2018-02-05)**. When tonight's honest re-run extended the window back to 2018-01-01, the strategy ate a **-31.5% year in 2018** alone (verified: applying the EXACT original logic to 2018-2026 drops annualized from +17.10% to +9.73% and Sharpe from 0.66 to 0.45). Adding 2018 alone accounts for roughly half of the gap between the headline number and reality, but only half.

The pre-2018 leverage splicing issue (`-1×` data scaled to `-0.5×`) is mentioned correctly in the spec (`iter_5_vix_term_structure_rotation.md:11, §9 line 93`) but does NOT apply to either script: both start in 2018 or later, both use the natural post-2018 SVXY series with no re-scaling needed.

### c. Instrument (XIV vs SVXY)

**SVXY only**, no XIV in either script. Verified at `01_walk_forward_2019_2026.py:50` (`yf.download("SVXY", ...)`) and `iter_5_backtest.py:49` (`"SVXY": "SVXY"`). XIV terminated Feb 2018 and is not in scope. No bug here.

### d. Rolling stats look-ahead

No rolling statistics used in either script. The signal is point-in-time `VIX/VIX3M` from each day's close. No bug here.

### e. Backtest period start (pre-2018 contaminates)

The original starts **post-2018** (2019-01-01) and so does not contaminate — but it also misses the worst regime stress that the strategy would have actually faced. The honest re-run starts 2018-01-01 and so includes the post-Feb-2018 reduced-leverage regime in full. The honest framing is the more conservative one and is the correct framing for deployment.

### f. Cost assumptions

Original: **zero rotation cost** (`01_walk_forward_2019_2026.py:139-148` — `strategy_ret` is just gross `svxy_ret` when signal=True, no cost line item).
iter_5: **1 bp per traded notional, charged on rebalance days** (`iter_5_backtest.py:41, 196`).

For a strategy with ~30 rotations/year, the cost differential is ~60 bps/year — material but not the dominant driver.

---

## The actual spec divergences (the real "bug")

The two scripts implement **different strategies**, both labeled "VIX/VIX3M SVXY rotation":

| Parameter | Original `01_walk_forward_2019_2026.py` | iter_5 `iter_5_backtest.py` | Effect |
|---|---|---|---|
| Threshold | `ratio < 1.0` (line 138) | `ratio < 0.92` (line 38) | Original triggers ~92.8% of days; iter_5 ~30%. Filters out a huge swath of high-edge contango-but-near-1.0 days. |
| Sizing | 100% SVXY (line 144) | 25% SVXY (line 39) | Linear; preserves Sharpe but quarters returns. |
| Circuit breakers | NONE | VIX > 30 OR VIX 1d > +50% (lines 42-43, 179-181) | Flattens during the highest-edge tail-roll days (counterintuitively, the *post*-spike sessions). 156 breaker events in the 2018-2026 window. |
| Rotation cost | 0 bps | 1 bp per traded notional change (line 41) | ~60 bps/year drag at 30 rotations/year. |
| Cash leg return | 0% (just flat, line 144) | SGOV daily return (lines 167-170, 199) | Adds ~5% APY × 70% of days = ~3.5% to honest run. Tonight's run benefited from SGOV; without it, the iter_5 result would be even worse. |
| Start date | 2019-01-01 (line 35) | 2018-01-01 (line 35) | Excludes Feb-2018 Volmageddon (-31.5% year). |

### The marketing-vs-spec mismatch

The headline number "+17.1% / Sharpe 0.66" was generated by a script that:
1. Goes 100% long SVXY on 92.8% of trading days
2. Has zero downside protection
3. Has zero costs
4. Starts after the Volmageddon stress event
5. Has a -32.8% max drawdown (per the cached summary.json)

The deployed live system (`scripts/svxy_rotation_host.py`) and the spec (`iter_5_vix_term_structure_rotation.md`) describe a strategy with circuit breakers, ≤30% sizing, 0.92 threshold, and tail-risk gates — which is the **right** strategy to deploy, but is **NOT what produced the headline number**. The +17.1% claim was inherited by reference without re-validation under the deployment-grade parameters.

---

## Did +17.1% reproduce?

**Yes**, with the exact original parameters: re-running the original logic on a fresh 2019-2026 yfinance pull yields **+17.01% / Sharpe 0.659** (vs cached 17.10% / 0.662 — within rounding). The number is real, reproducible, and self-consistent. **It just does not validate the strategy the user is about to deploy.**

The deployment-grade iter_5 spec applied to the original 2019-2026 period gives **-0.63% / Sharpe -0.08**. Applied to 2018-2026 it gives +0.38% / Sharpe 0.09. **There is no parameter framing of the conservative spec that produces 17.1% — the Sharpe ceiling for this strategy under realistic risk gates appears to be in the 0.0-0.2 range over the modern (post-2018) SVXY regime.**

---

## Hypothesis for the divergence

1. **Original validation = 1-parameter signal at full leverage, no cost, no breakers, post-stress-event period.** This was an upper-bound estimate of the edge, not a deployment validation.
2. **Spec writing = layered on conservative risk gates** (25% sizing, 0.92 threshold, circuit breakers) without re-validating that the gates preserve the edge they were meant to protect.
3. **The 0.92 threshold + circuit breakers eat the edge** because:
   - VIX/VIX3M `< 0.92` filters out the bulk of "shallow contango" days that historically delivered the most consistent roll-down.
   - The VIX > 30 circuit breaker flattens through the V-shaped recovery sessions (post-spike days when SVXY rallies hardest) — which is exactly the wrong direction.
4. **The 2018 inclusion is a secondary factor** — costs ~7 percentage points of annualized return on its own, but the bulk of the Sharpe collapse is the spec-tightening, not the period-extension.

---

## Recommendation

1. **Do not deploy iter_5 live tomorrow on the basis of the +17.1% / 0.66 figure.** The figure does not describe the iter_5 strategy.
2. **Treat the +0.38% / Sharpe 0.09 honest re-run as the actual baseline.** It fails the spec's own promote criteria (Sharpe ≥ 0.5 from §6/§11 of the spec).
3. **Strip the "+17.1% / 0.66 validated baseline" claim from:**
   - `scripts/svxy_rotation_host.py:3` (live-host docstring)
   - `research/candidates/RANKED_SHORTLIST.md:3` (operator-profile blurb)
   - `research/candidates/iter_5_vix_term_structure_rotation.md:55, 93` (validation references)
   - Replace with the honest figure plus the caveat that the original number came from a 100% / 1.0-threshold / no-breakers / no-cost variant that is no longer the deployed strategy.
4. **Re-design the strategy if it is to be kept**: either (a) loosen the threshold back toward 0.95-1.00 and accept higher tail risk, (b) drop the VIX>30 breaker and rely only on the +50% 1d-move breaker (which is much rarer), or (c) abandon the strategy in favor of one of the other shortlist candidates (iter_1 small-cap mean-rev is the OVERNIGHT_VERDICT pass).
5. **Add a regression test**: any future "validated baseline" claim must be reproducible from a script in the repo whose parameters EXACTLY match the parameters of the system being deployed. The script that generated 17.1% must be the same script (modulo data pull) that the live host runs against.

---

## Citations

- `experiments/svxy_lab/01_walk_forward_2019_2026.py:35` — `START_DATE = "2019-01-01"` (the period-selection choice)
- `experiments/svxy_lab/01_walk_forward_2019_2026.py:138` — `df["contango"] = df["ratio"] < 1.0` (the threshold = 1.0, not 0.92)
- `experiments/svxy_lab/01_walk_forward_2019_2026.py:143-144` — signal shift(1) (NO lookahead) and 100% sizing
- `experiments/svxy_lab/01_walk_forward_2019_2026.py:144` — `np.where(df["signal"], df["svxy_ret"], 0)` (no SGOV cash leg, no cost)
- `experiments/svxy_lab/runs/summary.json` — cached: `annualized_pct: 17.10`, `sharpe: 0.66`, `max_drawdown: -32.84%`, `pct_in_market: 92.8%`
- `research/scripts/iter_5_backtest.py:35-43` — iter_5 parameters (start=2018, threshold=0.92, sizing=0.25, breakers, 1bp cost)
- `research/scripts/iter_5_backtest.py:158-203` — full run_backtest function (signal correctly shift(1)-ed, SGOV cash leg, cost charged on weight change)
- `research/results/iter_5_summary.json` — honest output: `ann_return_pct: 0.384`, `sharpe: 0.0948`, `max_drawdown_pct: -11.31%`
- `scripts/svxy_rotation_host.py:3` — live-host docstring claiming +218% / +17.1%/yr / Sharpe 0.66 as "validated walk-forward 2019-2026"

---

## Verdict line

`AUDIT_LINE: Found code: YES | Diagnosed bug: spec-vs-validation mismatch (NOT lookahead) | File: experiments/svxy_lab/01_walk_forward_2019_2026.py:35,138,144`
