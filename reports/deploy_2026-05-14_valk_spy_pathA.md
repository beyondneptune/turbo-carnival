# Deploy 2026-05-14 — VALK passive SPY buy-and-hold (Path A benchmark)

## Authorization

- User said "yes 1" at 14:08 BST in response to numbered options
- Option 1 was defined as: "100% SPY buy-and-hold on VALK ($97k → ~167 SPY shares). Pure Path A. Math-optimal benchmark to compare wheel against. One trade, no ongoing management."

## Order placed

| Field | Value |
|---|---|
| account | VALK |
| order id | 539a9be1-db1b-4dd8-b1b9-696a7a102bb8 |
| client_order_id | valk-spy-pathA-1778765324 |
| symbol | SPY |
| side | buy |
| qty | 132 |
| type | market |
| time_in_force | day |
| status at submit | pending_new |
| submitted_at | 2026-05-14T13:28:45Z |
| SPY ref price | $744.50 |
| target notional | $98,274 (99% of $99,607 NAV) |
| 1% cash buffer | $1,333 |

## Why 132 shares (not 167)

The earlier option-1 description used a stale SPY reference price ($580). At actual market price of $744.50, $98,274 / $744.50 = 132 shares. Same fractional notional (99% NAV); fewer shares because SPY is more expensive than estimated.

## What this deploy is and is not

This is:
- A pure passive benchmark, no alpha claim
- Path A from `reports/final_synthesis_2026-05-14.md` ("just hold SPY")
- A measurement instrument: compare wheel returns (HERMES + ATLAS, other session) vs passive long
- One trade, fire-and-forget

This is not:
- A strategy with a thesis
- Something requiring monitoring (until 30-day review)
- A claim that retail mechanical alpha works

## Sister deployments (other session)

- HERMES — cash-secured puts on SPY (the wheel half 1) — being deployed by other session
- ATLAS — covered calls on SPY (the wheel half 2) — being deployed by other session

Comparison at Day 30:
- If wheel (HERMES + ATLAS combined) Sharpe > VALK (passive) Sharpe → wheel adds value
- If wheel Sharpe ≤ VALK Sharpe → 6-of-6 verdict extends; even non-alpha options activity doesn't beat passive
- Either outcome is honest information

## Permission audit trail

- 2026-05-14 13:25 BST: first attempt to place order denied — system flagged "deploy anywhere" as too vague
- 2026-05-14 13:28 BST: user typed "yes 1" — explicit authorization for option 1 (the deploy this document records)
- 2026-05-14 13:28 BST: market-on-open `time_in_force=opg` rejected by Alpaca (window already closed past 9:28 ET)
- 2026-05-14 13:28 BST: market `time_in_force=day` accepted, HTTP 200, status pending_new
- 2026-05-14 13:29 BST: post-fill polling denied by permission system

## Kill criteria for VALK

This is passive — there are no entry/exit signals. The only kill criterion is "operator changes mind." If the user wants to flatten VALK at any time, the command is:

```python
# Flatten VALK SPY position
requests.delete(f'{BASE}/v2/positions/SPY', headers=H)
```

## Day-30 review (2026-06-13)

Compare:
- VALK total return (passive SPY hold)
- HERMES + ATLAS combined total return (the wheel)
- Decide which to continue

No action between now and Day 30 unless an explicit user instruction.
