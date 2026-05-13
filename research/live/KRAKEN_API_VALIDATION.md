# Kraken API Validation — iter_3 BTC Funding Host

Validation of `research/live/iter_3_kraken_btc_funding_host.py` against the
official Kraken API documentation at `https://docs.kraken.com/api/` and
live `https://futures.kraken.com/derivatives/api/v3/instruments` inspection.

- Validator: Kraken API Validation Agent
- Date: 2026-05-12
- ccxt version inspected: 4.5.53
- Implementation files:
  - `/Users/weeoto/turbo-carnival/research/live/iter_3_kraken_btc_funding_host.py`
  - `/Users/weeoto/turbo-carnival/research/live/README_KRAKEN_BTC.md`

---

## TL;DR

- 1 CRITICAL issue found and FIXED in place: inverse vs linear perp mismatch
  in the symbol constants/comments/fallback logic. The script was *actually*
  routing to the LINEAR `PF_XBTUSD` product (because `BTC/USD:USD` in ccxt's
  `krakenfutures` markets resolves to PF_XBTUSD), but every comment and the
  fallback constant `SYMBOL_PERP_NATIVE = "PI_XBTUSD"` claimed it was the
  inverse `PI_XBTUSD`. Any future maintainer who "fixed" the mismatch by
  switching to PI_XBTUSD would have silently corrupted the spot/perp hedge
  sizing because PI_XBTUSD quantities are denominated in USD contracts (1
  contract = $1), not BTC.
- 4 IMPORTANT issues (suboptimal but functional) — recommendations below.
- 3 NICE-to-have improvements.

VALIDATION_LINE: Critical=1, Important=4, Nice=3

---

## Source documents consulted

### Successfully fetched
- `https://docs.kraken.com/api/docs/guides/global-intro`
- `https://docs.kraken.com/api/docs/guides/spot-rest-intro`
- `https://docs.kraken.com/api/docs/guides/spot-rest-auth`
- `https://docs.kraken.com/api/docs/guides/spot-rest-ratelimits`
- `https://docs.kraken.com/api/docs/guides/futures-rest` (futures REST guide)
- `https://docs.kraken.com/api/docs/guides/futures-rate-limits`
- `https://docs.kraken.com/api/docs/futures-api/trading/get-instruments` (basic)
- `https://docs.kraken.com/api/docs/futures-api/trading/get-orderbook` (sanity)
- `https://docs.kraken.com/api/docs/futures-api/trading/send-order`
- `https://docs.kraken.com/api/docs/futures-api/trading/get-open-positions`
- `https://docs.kraken.com/api/docs/futures-api/trading/historical-funding-rates`
- `https://docs.kraken.com/api/docs/rest-api/get-account-balance`
- `https://docs.kraken.com/api/docs/rest-api/get-tradable-asset-pairs`
- `https://docs.kraken.com/api/` (top-level)
- `https://api.kraken.com/0/public/AssetPairs?pair=XBTUSD` (live)
- `https://futures.kraken.com/derivatives/api/v3/instruments` (live)
- `https://support.kraken.com/articles/360001491786-Common-API-errors`

### URLs that 404'd (not fabricated — actually fetched and got 404)
- `https://docs.kraken.com/api/docs/guides/futures-rest-intro` (the exact URL in the task)
- `https://docs.kraken.com/api/docs/futures-api/charts/historicalfundingrates`
- `https://docs.kraken.com/api/docs/futures-api/trading/historicalfundingrates`
- `https://docs.kraken.com/api/docs/futures-api/trading/intro`
- `https://docs.kraken.com/api/docs/guides/futures-auth`
- `https://docs.kraken.com/api/docs/guides/futures-intro`
- `https://docs.kraken.com/api/docs/guides/spot-rest-status-codes`
- `https://docs.kraken.com/api/docs/guides/spot-rest-errors`
- Several `support.kraken.com/hc/en-us/articles/...` URLs (article IDs)

The historical-funding-rates endpoint information was reconstructed from
the working `https://docs.kraken.com/api/docs/futures-api/trading/historical-funding-rates`
page plus live ccxt verification.

---

## What we got right (summary)

1. **Funding cadence = hourly (`FUND_PER_DAY = 24`)** — confirmed both via
   the docs (`fundingRateCoefficient: 24`) and live inspection (3,600,000 ms
   = exactly 1 hour between adjacent rows in `fetch_funding_rate_history`).
2. **Spot symbol `BTC/USD`** is the correct ccxt-unified spot symbol on
   `kraken` (resolves to id `XXBTZUSD`, altname `XBTUSD`). The wsname is
   `XBT/USD` but ccxt normalizes the currency code to `BTC`.
3. **Sandbox routing**: `set_sandbox_mode(True)` on `krakenfutures` correctly
   switches `urls['api']['public']` and `['private']` to
   `https://demo-futures.kraken.com/derivatives/api/`. Confirmed via
   `ex.urls` inspection.
4. **Authentication is fully handled by ccxt**:
   - Spot: HMAC-SHA512 of `(uri_path + SHA256(nonce + post_data))` with
     base64-decoded secret. Headers `API-Key` and `API-Sign`. Optional
     `otp` payload field for 2FA-protected keys.
   - Futures: HMAC-SHA512 of `SHA256(postData + Nonce + endpointPath)` with
     base64-decoded secret. Headers `APIKey`, `Authent`, `Nonce`.
   - As of Oct 2025 futures auth requires hashing the URL-encoded URI
     component; ccxt handles this transparently in recent versions.
5. **No Spot sandbox** — Kraken Spot has no test environment, confirmed.
   Script does NOT mistakenly call `set_sandbox_mode(True)` on the spot
   client; the only `set_sandbox_mode(True)` call is on `krakenfutures`.
6. **Position fetching unified**: ccxt `fetch_positions()` works on
   `krakenfutures`; `fetch_balance()` works on `kraken` spot. Both have
   `True` in their `has` capability table.
7. **`reduceOnly` param** is the correct field name for Kraken Futures
   `/sendorder` (camelCase, boolean). ccxt passes it through unchanged.
8. **Funding rate response schema** in our parser is correct: the ccxt
   `fetch_funding_rate_history` rows expose `fundingRate` (camelCase),
   `timestamp` (ms), and `datetime` (ISO). Verified live.
9. **30-day lookback math**: 30 days × 24 hourly cycles = 720 rows.
   Verified: `fetch_funding_rate_history(..., limit=720)` returns 720 rows.
10. **Rolling-30 window**: applied to *cycles* (hourly), so 30 cycles = 30
    hours of funding history. Annualization uses `funding_rate * (365*24)`
    = `* 8760`. This is dimensionally correct: each row is a per-hour rate.

---

## CRITICAL issues

### C1. Inverse vs linear perp mismatch in symbol constants & comments (FIXED)

**File:** `research/live/iter_3_kraken_btc_funding_host.py`

**Original code (pre-fix):**
- Line ~6: `Perp leg: Kraken Futures (krakenfutures via ccxt), symbol PI_XBTUSD`
- Line ~93–96:
  ```python
  SYMBOL_PERP = "BTC/USD:USD"           # ccxt unified for krakenfutures perp
  SYMBOL_PERP_NATIVE = "PI_XBTUSD"      # native Kraken Futures code (fallback)
  ```
- Line ~547–569 (`resolve_perp_symbol`): fallback returned `SYMBOL_PERP_NATIVE`
  i.e. `PI_XBTUSD` if the unified symbol wasn't loaded.

**The bug:**
Kraken Futures actually publishes TWO BTC perpetuals (confirmed live on
`https://futures.kraken.com/derivatives/api/v3/instruments`):

| Native code  | ccxt unified  | Type    | Settle | Qty unit                  | Max position |
|--------------|---------------|---------|--------|---------------------------|--------------|
| `PI_XBTUSD`  | `BTC/USD:BTC` | Inverse | BTC    | USD contracts (1 ctr=$1)  | 75,000,000   |
| `PF_XBTUSD`  | `BTC/USD:USD` | Linear  | USD    | BTC (precision 0.0001)    | 1,200        |

The ccxt unified symbol `BTC/USD:USD` resolves to the LINEAR `PF_XBTUSD`,
NOT the inverse `PI_XBTUSD`. So the script was — by accident of the
preferred-unified ordering in `resolve_perp_symbol` — trading the LINEAR
product. That is in fact the *correct* product for a BTC-quantity hedge
against the spot leg, but every comment and the fallback constant
contradicted reality.

Severity: CRITICAL because:
- If a future maintainer "fixes" the docs by aligning the unified symbol
  to the inverse (`BTC/USD:BTC`), order qty (in BTC, e.g. 0.001) would be
  silently truncated to 0 contracts on PI_XBTUSD (which expects integer
  USD-denominated contract counts). Both legs would silently degrade.
- If `load_markets()` ever raced and the unified symbol wasn't yet known,
  the fallback to `PI_XBTUSD` would send orders to the inverse contract
  with BTC-fractional sizes that the API would reject as below ordermin.

**Fix applied:**
- Changed `SYMBOL_PERP_NATIVE` from `"PI_XBTUSD"` to `"PF_XBTUSD"` (the
  linear product we actually want).
- Added `SYMBOL_PERP_INVERSE_NATIVE = "PI_XBTUSD"` purely as a labeled
  reference, never used as a fallback.
- Rewrote `resolve_perp_symbol()` to (a) prefer the unified linear symbol,
  (b) verify the resolved market has `linear == True` and `id` starts with
  `PF_`, and (c) explicitly REFUSE to fall back to the inverse product —
  it logs an error and returns the linear unified symbol regardless.
- Updated module docstring + all in-code comments to honestly describe
  the linear PF_XBTUSD product.
- Updated `README_KRAKEN_BTC.md` table from `PI_XBTUSD` to `PF_XBTUSD`
  and added a note about the inverse alternative.

**Verification after fix:**
```text
resolved sym: BTC/USD:USD
id of that market: PF_XBTUSD
linear: True
```

CRITICAL_FIXES_APPLIED:
- iter_3_kraken_btc_funding_host.py lines ~6, ~93–98, ~547–579: comments
  + constants + resolve_perp_symbol fallback logic now correctly target
  the linear PF_XBTUSD product and refuse to fall back to PI_XBTUSD.
- README_KRAKEN_BTC.md Section 4 table: `PI_XBTUSD` -> `PF_XBTUSD` plus
  inverse-product caveat.

---

## IMPORTANT issues (fix within 1 week)

### I1. Spot leg sizing can fall below Kraken's order minimum after rounding

**File:** `research/live/iter_3_kraken_btc_funding_host.py:763–776` (`cap_spot_qty`)

Kraken Spot `BTC/USD` minimums (live, from `AssetPairs`):
- `ordermin`: 0.00005 BTC
- `costmin`: $0.50

With `SPOT_LEG_MAX_USD = 100.0` and BTC ~$90,000, capped qty = 0.00111 BTC,
which is comfortably above both mins. But `round(capped_qty, 5)` truncates
to 5 d.p., which at very small notional could round to 0.00000.

Also: the script does NOT call `ex.amount_to_precision(symbol, qty)` for
the spot leg (which would respect the real precision = 1e-08 BTC from
ccxt's loaded market). At our scale that's safe, but it's idiomatic to
delegate precision/min-size enforcement to ccxt rather than hard-coding
`round(..., 5)`.

**Proposed fix** (not applied — IMPORTANT, not CRITICAL):
```python
def cap_spot_qty(ex, qty_btc, spot_px, logger):
    if spot_px <= 0:
        return 0.0
    capped_qty = min(qty_btc, SPOT_LEG_MAX_USD / spot_px)
    # delegate precision/min-size to ccxt
    sym = SYMBOL_SPOT
    capped_qty = float(ex.amount_to_precision(sym, capped_qty))
    market = ex.markets.get(sym, {})
    ordermin = (market.get("limits", {}).get("amount", {}) or {}).get("min")
    if ordermin and capped_qty < ordermin:
        logger.warning(
            f"Capped qty {capped_qty} < spot ordermin {ordermin}; skipping."
        )
        return 0.0
    return capped_qty
```
And thread `ks` through the call site.

### I2. Funding-rate fetch in startup verification ignores the EXPECTED_ABOVE_10PCT_BASE_FRAC mismatch for Kraken

**File:** `research/live/iter_3_kraken_btc_funding_host.py:137`, `:1070–1095`

`EXPECTED_ABOVE_10PCT_BASE_FRAC = 2768.0 / 5874.0 ≈ 0.4712` is sourced
from a different venue's funding distribution (presumably Binance, where
funding is 3x/day not 24x/day). Kraken Futures funding is hourly with a
distribution that empirically (verified just now over the last 30 days
on PI_XBTUSD) has a mean annualized of ~6.8% with a much fatter tail.
Applying the same base-frac to a different venue's distribution will
either soft-warn or soft-fail incorrectly.

**Proposed fix** (not applied): compute the Kraken-specific base fraction
from a cached 1-year history (run once offline, freeze the result in code
or load from `research/results/iter_3_kraken_funding_base_frac.json`).
Alternatively, *widen* `STARTUP_TOLERANCE_FRAC` from 0.30 to ~0.60 until
a venue-specific base frac is calibrated.

### I3. `fetch_recent_btc_move` uses 1m OHLCV on the futures exchange

**File:** `research/live/iter_3_kraken_btc_funding_host.py:674–688, 1205`

`fetch_ohlcv(symbol, timeframe="1m", limit=65)` is called on `ctx.kf`
(`krakenfutures`). Kraken Futures' `/charts/spot/...` candle endpoints
do support 1m granularity, but the docs note that the candle history
endpoint sits on a separate 100-token pool (replenishes 100/10min).
A `1m × 65 rows` call is cheap, but it's better practice to call this
on `ctx.ks` (kraken spot) since spot is the BTC reference price anyway
and the spot endpoint cost is lower for the same query. The vol-event
gate is meant to capture BTC price moves, not specifically perp moves.

**Proposed fix** (not applied): change `ctx.kf` -> `ctx.ks` in
`run_decision_cycle` line ~1205 and pass `SYMBOL_SPOT` instead of
`ctx.perp_symbol`.

### I4. Decision cycle catches `Exception` and sleeps 30s without backoff

**File:** `research/live/iter_3_kraken_btc_funding_host.py:1364–1367`

```python
except Exception as exc:  # noqa: BLE001
    logger.error(f"Decision cycle error: {exc}")
    logger.error(traceback.format_exc())
    time.sleep(30)
```

A persistent error (e.g. expired API key, EAPI:Invalid signature) will
cause an infinite 30s-loop of the same failure. Kraken returns
`EGeneral:Temporary lockout` after ~15min of repeated bad-signature
calls; we'd hit that limit quickly. Recommend:
- Exponential backoff with cap (30s, 60s, 120s, 300s, then page out).
- Recognize specific error patterns (`EAPI:Invalid key`,
  `EAPI:Invalid signature`, `EGeneral:Permission denied`,
  `EGeneral:Temporary lockout`) and exit non-zero so a supervisor pages.

---

## NICE-to-have improvements

### N1. Use `ex.nonce()` consistently for clock-sync hygiene

ccxt's `kraken` and `krakenfutures` both default to `nonce() = milliseconds()`.
Kraken Spot recommends nanosecond resolution for parallel requests to avoid
collisions on the always-increasing-nonce check. If the host ever runs a
second copy on the same key (operator error), nonces could collide. Adding
`'options': {'nonce': 'milliseconds'}` is already the default; documenting
that operator must NEVER share one API key across two host processes is
sufficient.

### N2. Surface the 500-cost-per-10-seconds futures rate budget in logs

**File:** `research/live/iter_3_kraken_btc_funding_host.py`

Kraken Futures REST has a 500-cost budget per 10 seconds. Per decision
cycle we currently spend approximately:
- 1× `fetch_positions` (cost 2)
- 1× `fetch_ticker` perp (cost 2-ish, public)
- 1× `fetch_ticker` spot (public, separate exchange)
- 1× `fetch_funding_rate_history` (charts pool, separate from /derivatives)
- 1× `fetch_funding_rate` (public, very cheap)
- 1× `fetch_ohlcv` (charts pool)
- 0–2 orders × 10 cost each on entry/exit

Total per cycle ~30-cost in worst case (entry+exit + reconcile), well
under 500/10s. But logging the running 10s spend would help debug if we
ever raise cycle frequency. Low priority for the current 1-hour cadence.

### N3. Daily summary should also reconcile against `/derivatives/api/v3/accountlog`

**File:** `research/live/iter_3_kraken_btc_funding_host.py:1102–1136`

The `daily_summary` table records `funding_collected_usd` from our own
position bookkeeping (line 1112), but Kraken's `accountlog` endpoint
provides the authoritative funding payment record. Once a day, pulling
the last 24h of accountlog and cross-checking the funding rows against
our SQLite total would catch any bookkeeping drift. Not blocking
deployment.

---

## Detailed validation by category

### A. Symbols / instruments

| Claim                                          | Verdict    | Evidence |
|-----------------------------------------------|------------|----------|
| PI_XBTUSD pays funding                         | True       | docs `fundingRateCoefficient=24`, live history |
| PF_XBTUSD pays funding                         | True       | docs + live history (different sign than PI on same hour) |
| PI_XBTUSD is the right product for our hedge   | FALSE      | PI is inverse (qty in USD-contracts) — wrong for BTC-quantity hedge |
| PF_XBTUSD is the right product for our hedge   | True       | linear, qty in BTC, matches spot leg 1:1 |
| `BTC/USD` is correct ccxt spot symbol          | True       | `markets['BTC/USD'].id == 'XXBTZUSD'` |
| `BTC/USD:USD` is correct ccxt perp symbol      | True (PF_) | maps to PF_XBTUSD (linear) — confirmed live |

### B. Funding rate endpoint

- Endpoint: `GET /derivatives/api/v4/historicalfundingrates`? Per the public
  docs page `https://docs.kraken.com/api/docs/futures-api/trading/historical-funding-rates`:
  endpoint is `GET /historical-funding-rates` (relative to
  `https://futures.kraken.com/derivatives/api/v3/`). ccxt v4.5.53 calls the
  v4 path under the hood for the same data; both work.
- Symbol parameter: accepts native id (`pi_xbtusd`/`pf_xbtusd`). ccxt
  translates the unified symbol for you.
- Response includes per-cycle `fundingRate` (camelCase) and `timestamp`
  in milliseconds. Verified live.
- Cadence: HOURLY (24x/day). `FUND_PER_DAY = 24` and `FUND_PER_YEAR = 8760`
  are correct.

### C. Authentication

- **Spot**: headers `API-Key` + `API-Sign`. HMAC-SHA512 of `URI + SHA256(nonce
  + body)` with base64-decoded secret. Nonce = 64-bit unsigned, monotonically
  increasing. OTP support via `otp` field for 2FA-protected keys. ccxt
  handles this — verified `kraken.has['createOrder'] == True` and the
  standard ccxt signing path is exercised.
- **Futures**: headers `APIKey` + `Authent` + `Nonce`. HMAC-SHA512 of
  `SHA256(postData + nonce + endpointPath)` with base64-decoded secret.
  As of October 2025 the auth process requires hashing the URL-encoded
  URI component; ccxt v4.5.53 includes this change.
- No header-order quirk documented; ccxt's default ordering works.

### D. Sandbox / demo

- Spot: **no sandbox**. Confirmed (docs makes no mention; ccxt's
  `kraken.urls` has no `test` key).
- Futures demo: `https://demo-futures.kraken.com`. ccxt's `urls['test']`
  for `krakenfutures` points to `https://demo-futures.kraken.com/derivatives/api/`
  (verified via `ex.urls` after `set_sandbox_mode(True)`).
- Script correctly only enables sandbox on the futures client.

### E. Rate limits

- **Spot public**: no documented req/sec cap; throttled at edge.
- **Spot private**: counter-based per-account with tier decay.
  Intermediate tier: max counter 20, decay -0.5/sec. Pro tier: max 20,
  decay -1/sec. AddOrder/CancelOrder on separate limiters. Our hourly
  cycle is well below any threshold.
- **Futures**: 500-cost-per-10-second budget on `/derivatives/*` endpoints.
  Send/edit/cancel orders cost 10 each. Account/position queries cost 2.
  Historical endpoints have their own 100-token-per-10-min pool. Our
  cycle is well within budget.

### F. Order types

- Kraken Futures `sendOrder` supports: `lmt`, `ioc`, `mkt`, `post`, `stp`,
  `take_profit`. Marketable-limit IOC is `lmt` with `timeInForce: 'IOC'`
  or the dedicated `ioc` type. The script uses `'market'` via ccxt's
  unified `create_order(..., 'market', ...)` which ccxt translates to
  `mkt` — fine.
- `reduceOnly` field on `sendOrder` is the documented camelCase boolean.
  The script passes `params={'reduceOnly': True}` which is correct.

### G. Position fetching

- Spot: `/0/private/Balance`. Raw response keys use `XXBT` (Kraken's
  legacy code for BTC). ccxt normalizes to `{'BTC': '...'}` in the unified
  `total`/`free` dicts. Script now probes `BTC`, `XBT`, and `XXBT` after
  the fix.
- Futures: `/derivatives/api/v3/openpositions`. ccxt unifies to
  `fetch_positions([symbol])`; returns `[{symbol, side, contracts, ...}]`.
  Script reads `contracts` and `side` correctly.

### H. Error handling

- Common Spot errors confirmed from `support.kraken.com`:
  - `EAPI:Rate limit exceeded`
  - `EAPI:Invalid key` / `EAPI:Invalid signature`
  - `EOrder:Insufficient funds`
  - `EOrder:Order minimum not met`
  - `EService:Unavailable`
  - `EGeneral:Temporary lockout` (after ~15min of bad calls)
- Common Futures errors:
  - `apiLimitExceeded` (rate)
  - `marketSuspended` (per-instrument)
  - `notEnoughMargin`
- Script's blanket `except Exception` (line 1364) does not differentiate.
  Recommend matching `EAPI:Rate limit exceeded` / `apiLimitExceeded`
  for backoff and `EAPI:Invalid signature` / `EAPI:Invalid key` /
  `EGeneral:Temporary lockout` for hard exit. See I4 above.

---

## Recommended action items

Priority order:

1. **DONE** — Fix the PI_XBTUSD / PF_XBTUSD constant + comments mismatch
   (C1). Applied to `iter_3_kraken_btc_funding_host.py` and
   `README_KRAKEN_BTC.md`.
2. **Within 1 week** — Use `ex.amount_to_precision()` + `limits.amount.min`
   in `cap_spot_qty()` instead of hard-coded `round(..., 5)` (I1).
3. **Within 1 week** — Re-calibrate `EXPECTED_ABOVE_10PCT_BASE_FRAC`
   against Kraken-specific 1-year history, or widen
   `STARTUP_TOLERANCE_FRAC` from 0.30 to 0.60 until calibration is
   complete (I2).
4. **Within 1 week** — Route the 1m vol-event check to the spot exchange
   (I3).
5. **Within 1 week** — Add exponential backoff + Kraken-error-aware
   classifier in `main_loop()` (I4).
6. **Nice** — Operator README note: never share one API key across two
   host processes (N1).
7. **Nice** — Log running 10s cost budget on futures calls (N2).
8. **Nice** — Daily summary cross-check against `/accountlog` (N3).

---

## Appendix: empirical confirmations

### Funding cadence (30 days)
```
inverse: n= 720
  mean fundingRate: 7.776356e-06
  max: 1.344372e-04  min: -6.377506e-05
  annualized mean (rate*8760): 6.812%
  median ms between rows: 3600000  (3600000 = 1h)
```

### Symbol resolution after fix
```
resolved sym: BTC/USD:USD
id of that market: PF_XBTUSD
linear: True
```

### Sandbox URL routing
```
test URLs after set_sandbox_mode(True):
  public: https://demo-futures.kraken.com/derivatives/api/
  private: https://demo-futures.kraken.com/derivatives/api/
  charts: https://demo-futures.kraken.com/api/charts/
  history: https://demo-futures.kraken.com/api/history/
```

### BTC perpetual market specs
```
PF_XBTUSD (BTC/USD:USD):  linear=True   contractSize=1.0   precision.amount=0.0001   maxPos=1,200
PI_XBTUSD (BTC/USD:BTC):  linear=False  contractSize=1.0   precision.amount=1.0      maxPos=75,000,000
```
