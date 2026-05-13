# Broker Setup Guide — Simple (No-Friction) Edition

This is the streamlined setup that uses ONLY brokers the operator already has verified: Alpaca (paper, already in use) and Kraken (already verified, just needs Futures enabled). No IBKR application, no Binance/OKX KYC, no waiting on identity verification.

Trade-off: some strategies run with proxy instruments (EWJ ETF instead of CME NKD; Alpaca-side ADR with simulated HK leg). See Section F for the honest list of what degrades.

Estimated total setup time: **~30–60 minutes** (the only real work is enabling Kraken Futures and generating sandbox keys).

---

## Section A: Stack overview

**Total brokers needed: 2.**

- **Alpaca paper** (already in use; works immediately)
- **Kraken Futures + Spot** (user already has verified Kraken account)

**Total new verifications needed: 0** (enabling Futures on existing Kraken is instant; demo-futures.kraken.com requires no KYC).

### Strategy → broker map

| Strategy | Broker | Mode |
|---|---|---|
| SVXY sweet spot | Alpaca | Paper |
| iter_18 ADR (US-leg real, HK-leg sim) | Alpaca | Paper |
| iter_6 Nikkei (EWJ ETF proxy) | Alpaca | Paper |
| iter_3 BTC funding | Kraken Futures Sandbox + Kraken Spot | Demo for perp, tiny-size for spot |

---

## Section B: Alpaca setup (assumes user has it)

Confirm working:

```bash
curl -X GET "https://paper-api.alpaca.markets/v2/account" \
  -H "APCA-API-KEY-ID: $ALPACA_AEGIS_KEY" \
  -H "APCA-API-SECRET-KEY: $ALPACA_AEGIS_SECRET"
```

Should return JSON with `"status": "ACTIVE"` and a `"cash"` field.

Required env vars (presumably already set in `council/.env.local` from prior deployment):

```
ALPACA_AEGIS_KEY=<key>
ALPACA_AEGIS_SECRET=<secret>
```

If the curl returns 401 or 403, regenerate the keys at https://app.alpaca.markets/paper/dashboard/overview → **API Keys**.

---

## Section C: Kraken setup

### C.1 Enable Kraken Futures on existing account

- Go to https://www.kraken.com → Login
- Click **Trade** → **Futures** — this opens `futures.kraken.com`
- Account is automatically migrated; no additional KYC if the base Kraken account is already verified
- For demo/paper: go to `demo-futures.kraken.com` (separate signup, **NO KYC required**)

### C.2 Generate API keys

**Kraken Futures Sandbox (for paper trading)**:

- Go to https://demo-futures.kraken.com
- Sign up with email (no KYC)
- **API Settings → Create new key** with full permissions
- Save key + secret to env:

```
KRAKEN_FUTURES_API_KEY=<demo_key>
KRAKEN_FUTURES_API_SECRET=<demo_secret>
```

**Kraken Spot (real account, tiny-size for testing)**:

- `futures.kraken.com` → switch to "Live" tab (or use main Kraken site)
- **Settings → API → Create new key**
- Permissions: "Query funds", "Modify orders", "Cancel orders" only (**NOT** "Withdraw funds")
- Save:

```
KRAKEN_SPOT_API_KEY=<live_key>
KRAKEN_SPOT_API_SECRET=<live_secret>
```

### C.3 Test connection

```bash
.venv/bin/python -c "
import ccxt
fut = ccxt.krakenfutures({'apiKey': '...', 'secret': '...'})
fut.set_sandbox_mode(True)
print('Sandbox balance:', fut.fetch_balance())
print('Funding:', fut.fetch_funding_rate('PI_XBTUSD'))
"
```

Expected: a balance dict + funding rate object.

---

## Section D: Account capital layout for paper shadow

Recommended allocation on the existing 6 paper accounts (consolidate to 2 active):

- **AEGIS paper account ($100k):** SVXY sweet spot + iter_18 ADR Alpaca + iter_6 Nikkei ETF (these all run on the same Alpaca account, separated by ticker)
- **Kraken sandbox ($100k virtual):** iter_3 BTC funding (perp on sandbox, spot on real-tiny)

Other 5 Alpaca paper accounts: **park in SGOV** until strategies validate. No reason to spread capital across idle accounts during paper-shadow phase.

---

## Section E: Daily startup ritual (simplified)

### Pre-market (08:00 ET)

1. Verify Alpaca connection (curl test in Section B)
2. Verify Kraken Futures sandbox connection (Python test in Section C.3)
3. Launch dashboard:

```bash
cd /Users/weeoto/turbo-carnival
.venv/bin/streamlit run research/live/portfolio_dashboard.py --server.port 8503 --server.address 0.0.0.0
```

4. Launch SVXY shadow:

```bash
.venv/bin/python research/live/svxy_sweet_spot_host.py --account AEGIS --confirm
```

5. Launch ADR-Alpaca shadow:

```bash
.venv/bin/python research/live/iter_18_adr_alpaca_host.py --account AEGIS --confirm
```

6. Launch Nikkei ETF shadow:

```bash
.venv/bin/python research/live/iter_6_nikkei_etf_host.py --account AEGIS --confirm
```

7. Launch Kraken BTC funding shadow:

```bash
.venv/bin/python research/live/iter_3_kraken_btc_funding_host.py --confirm
```

8. Verify with meta framework:

```bash
.venv/bin/python research/live/meta_framework.py list
```

### EOD (16:30 ET)

1. Review dashboard (P&L, fills, slippage vs model)
2. Run `meta_framework.py kill-check`
3. Save daily snapshots (host scripts auto-persist, but verify the SQLite DB has today's date stamp)

---

## Section F: Honest limitations of this simpler setup

What you're giving up vs IBKR + Binance/OKX:

1. **iter_18 ADR:** HK leg is simulated, not executed. Real-world edge may be **50–70% of backtest** after HKEX costs and timing slippage on the real HK side.
2. **iter_6 Nikkei:** EWJ ETF proxy adds currency drag and tracking error vs NKD futures. Expected **~50–70% of backtest edge**.
3. **iter_3 BTC:** Kraken Futures sandbox doesn't perfectly mirror real fill behavior (sandbox liquidity and queue dynamics are synthetic); real edge may differ from sandbox-measured edge.

These are paper-shadow constraints, not deployment limits. Once you want to scale to real money, migrate to IBKR + Binance/OKX for full execution accuracy.

---

## Section G: Migration path to full IBKR + Binance/OKX

When ready (probably after a 30-day paper shadow shows clean reconciliation):

1. Apply for IBKR live account (1–3 business days)
2. Apply for Binance live account (varies by jurisdiction)
3. Swap the deployment scripts:
   - `iter_18_adr_alpaca_host.py` → `iter_18_adr_host.py` (IBKR)
   - `iter_6_nikkei_etf_host.py` → `iter_6_nikkei_host.py` (IBKR CME NKD)
   - `iter_3_kraken_btc_funding_host.py` → `iter_3_btc_funding_host.py` (Binance + OKX)
4. Re-run a 30-day paper shadow on the upgraded scripts before deploying live capital

---

## Section H: Troubleshooting

| Symptom | Fix |
|---|---|
| Kraken Futures returns "Invalid API key" | Check you're using sandbox keys with `sandbox=True` (or live keys with `sandbox=False`) — they're different endpoints |
| Alpaca rejects EWJ short | Check the paper account is approved for short selling (paper default is enabled; if disabled, toggle in account settings) |
| HK leg "trades" don't appear in dashboard | Check `iter_18_adr_alpaca_host.py` is writing to its sqlite; the dashboard reads from that DB |
| Cycle interval too fast | Kraken rate limits at ~15 req/sec public, ~1/sec private — back off cycle frequency or batch calls |
| Alpaca curl returns 401/403 | Regenerate keys at https://app.alpaca.markets/paper/dashboard/overview → API Keys |
| Dashboard shows stale data | Confirm host scripts are running (`ps aux \| grep host`); check SQLite DB file permissions; verify host scripts have write access |

---

## Quick-reference: env vars summary

Final `council/.env.local` should contain at minimum:

```
# Alpaca paper
ALPACA_AEGIS_KEY=...
ALPACA_AEGIS_SECRET=...

# Kraken Futures sandbox (paper)
KRAKEN_FUTURES_API_KEY=...
KRAKEN_FUTURES_API_SECRET=...

# Kraken Spot (live, tiny-size, no withdraw permission)
KRAKEN_SPOT_API_KEY=...
KRAKEN_SPOT_API_SECRET=...
```

Keep `.env.local` out of git (already in `.gitignore`). Rotate keys if they leak.
