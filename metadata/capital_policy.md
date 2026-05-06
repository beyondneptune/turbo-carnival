# Capital Policy

**Version:** v1.0
**Locked:** 2026-05-06
**Status:** Active for Phase 0 → Cycle 1
**Modification rule:** Any change requires CEO approval and resets the cycle clock.

---

## Paper Notional

**$10,000 USD paper notional.** This is the smallest viable capital at which the hybrid strategy mechanically runs. Below this amount, the hybrid breaks (one sleeve must be dropped, sample size collapses, cost drag dominates returns).

Justification documented in `metadata/strategic_lock.md` § Capital Sizing Rationale.

## Sleeve Allocation

| Sleeve | Allocation | Dollar Amount | Strategy |
|---|---|---|---|
| Core | 70% | $7,000 | VRP harvesting via SPY iron condors |
| Satellite | 30% | $3,000 | Earnings straddle selection on Russell 1000 |
| Cash buffer (always) | ≥20% of NAV | ≥$2,000 | Available margin/liquidity |

The cash buffer constraint floats with NAV, not the static allocation. If NAV falls to $9,000, cash buffer must be ≥$1,800 — sleeves get re-sized down accordingly.

## Hard Risk Limits

| Limit | Trigger | Action |
|---|---|---|
| Daily intraday loss | -8% NAV | Cycle suspended, postmortem required |
| Cumulative cycle drawdown | -15% NAV | Cycle suspended, restart prereg |
| Single-position vega | >5% NAV | Position rejected by risk gate |
| Total portfolio vega | >15% NAV | New positions rejected until reduced |
| Single-name concentration | >20% NAV | Position rejected by risk gate |
| Cash buffer floor | <20% NAV | Stop opening new positions |

The kill switch is **deterministic Python code**. No LLM can override it. No human can disable it during a frozen cycle (would reset the cycle).

## Live Capital Policy

**Year 1: $0 live capital. Paper trading only.**

Live capital is considered only if:
1. ≥1 cycle passes ALL 7 pre-registered gates at base AND +50% costs
2. CEO + investor explicit written approval
3. Live deployment occurs at ≤1% of any potential treasury allocation, with the same architecture, gates, and risk limits as paper
4. First live cycle treated as paper-with-real-money — same freeze rules, same kill switch, same evaluation window

Scaling beyond initial live deployment requires its own pass at each tier (1% → 5% → 10% → 25%). No automatic scaling.

## Sizing Per Trade

### Core (VRP iron condors)
- Maximum 1 contract per position (forced by capital floor)
- $5-wide wings on SPY (max loss per condor ~$350-500)
- Maximum 4-6 simultaneous positions
- Cadence: open weekly, ~30-45 DTE entries
- Profit target: 50% of credit received, OR 21 DTE close
- Stop: 2× credit received as max loss before forced exit

### Satellite (earnings straddles)
- Maximum 2-4 simultaneous positions
- 1 contract per position
- ATM strike at decision time (T-1 close)
- Nearest weekly expiration ≥3 calendar days post-earnings
- Hold from T-1 close to T+5 close
- No partial exits

## Pre-Registered Cost Stress

Strategy must pass evaluation gates at:
- **Base costs:** measured spread + commission ($0.65/contract IBKR Pro)
- **+50% costs:** spread × 1.5
- **+100% costs:** spread × 2.0 (informative, non-blocking)

Failure at +50% costs → strategy fails the cycle.
Failure at +100% costs → reported in postmortem but does not fail the cycle.

## Authorization

| Action | Required Approval |
|---|---|
| Modify this document | CEO + cycle reset to day 0 |
| Override kill switch during cycle | Forbidden — cycle reset required |
| Increase notional within Year 1 | Forbidden — $10k locked for Year 1 |
| Deploy live capital | CEO + investor written approval + cycle pass |

---

**Signature line:** Locked under CEO authority, 2026-05-06. No modifications until evaluation phase of Cycle 1.
