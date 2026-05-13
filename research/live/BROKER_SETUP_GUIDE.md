# Broker Setup Guide

This guide walks the operator through setting up paper-trading accounts for the live paper-shadow deployment of the turbo-carnival strategies. Three of the four shadowing strategies require IBKR (SVXY, iter_18 ADR, iter_6 Nikkei), and one requires crypto exchanges (iter_3 BTC funding).

Estimated total setup time: 2–3 hours (most of which is waiting on IBKR account approval).

---

## Section A: IBKR Paper Account Setup

IBKR is used by three strategies:
- **SVXY sweet-spot** — US equity (SVXY ETF)
- **iter_18 ADR pairs** — US/HK arb (e.g., BABA/9988, JD/9618, NIO/9866)
- **iter_6 Nikkei** — CME NKD futures vs OSE NKM mini

### A.1 Account creation

1. Go to https://www.interactivebrokers.com/en/home.php
2. Click **Open Account** then choose **Individual** account
3. For paper trading: after live account approval, request paper account from **Client Portal → Settings → Paper Trading Account**
4. Note: IBKR requires a real account application BEFORE giving paper credentials (unlike Alpaca where paper is instant). Plan for 1–3 business days of identity verification.
5. Alternative: TWS-only paper login at https://www.interactivebrokers.com/en/index.php?f=15875 — limited (no real account linkage) but works for connection testing.

### A.2 Data subscriptions needed

Paper accounts inherit subscriptions from the linked live account. For our three strategies, the following are required:

| Subscription | Cost | Used by |
|---|---|---|
| **HKEX Cash and Derivative Real-Time** | $24/mo | iter_18 ADR (HK side: 9988, 9618, 9866, etc.) |
| **CME Real-Time** | $10/mo (free for paper-only) | iter_6 Nikkei NKD futures |
| **US Securities NASDAQ Network B Real-Time** | usually free | SVXY + ADR US side |

Activate from **Client Portal → Settings → Market Data Subscriptions**. Subscriptions activate next business day.

### A.3 IB Gateway / TWS installation

1. Download **IB Gateway** (lightweight, command-line friendly, recommended over full TWS) from:
   https://www.interactivebrokers.com/en/trading/ib-gateway-download.php
2. Install latest stable version (avoid beta builds)
3. Launch IB Gateway, log in with paper credentials
4. **Settings → API → Settings**:
   - Check **Enable ActiveX and Socket Clients**
   - Socket port: **7497** (paper) or **7496** (live)
   - Allow connections from **localhost** (127.0.0.1)
   - Create master API client ID (any integer, e.g., **17**)
   - Uncheck "Read-Only API" so orders can be submitted
   - Save

### A.4 Environment variables

Add to `council/.env.local`:

```
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=17
```

If you need to run multiple host scripts in parallel against the same gateway, allocate different client IDs (e.g., 17 for SVXY, 18 for ADR, 19 for Nikkei) — IB Gateway allows multiple concurrent client connections but each needs a unique ID.

### A.5 Test connection

```bash
.venv/bin/python -c "
from ib_insync import IB
ib = IB()
ib.connect('127.0.0.1', 7497, clientId=17)
print('Connected:', ib.isConnected())
print('Account:', ib.managedAccounts())
ib.disconnect()
"
```

Expected output: `Connected: True`, account ID starting with `DU` (paper) — e.g., `DU1234567`. A live account would start with `U`.

If you see `ConnectionRefusedError`: IB Gateway is not running, or the port is wrong, or the firewall is blocking localhost. Confirm IB Gateway is logged in (the status indicator should be green).

---

## Section B: Crypto Exchange Setup (Binance + OKX paper)

iter_3 BTC funding uses a perpetual futures funding-rate arbitrage. We need:
- One **perp venue** (Binance testnet) for the short-perp leg
- One **spot venue** (OKX demo) for the long-spot leg
- Optional **fallback** (Bybit testnet) if OKX demo is misbehaving

### B.1 Binance Testnet

1. Sign up: https://testnet.binance.vision/
2. Log in using GitHub OAuth (no real account needed for testnet)
3. Generate API key + secret from testnet dashboard (top-right)
4. Testnet has FREE virtual balances (~$100k USDT auto-credited; can request reset)
5. Add to `council/.env.local`:

```
BINANCE_TESTNET_API_KEY=...
BINANCE_TESTNET_API_SECRET=...
```

Note: Binance has a **separate** futures testnet at https://testnet.binancefuture.com/ — for funding-rate arbitrage we need the **futures** testnet (perp contracts), not the spot testnet. Use the futures testnet URL and generate keys there.

### B.2 OKX Paper Trading

1. Sign up at https://www.okx.com (real account required first)
2. Complete identity verification (KYC level 1 is sufficient)
3. Switch to **Demo Trading** mode in the UI (top-right toggle, usually a small icon near profile menu)
4. Generate API key + secret + passphrase from **Demo Trading Settings → API**
5. The demo trading API base URL is `https://www.okx.com` with a special header `x-simulated-trading: 1` — `ccxt` handles this when you set sandbox mode
6. Add to env:

```
OKX_PAPER_API_KEY=...
OKX_PAPER_API_SECRET=...
OKX_PAPER_PASSPHRASE=...
```

### B.3 Bybit Testnet (FALLBACK if OKX paper is problematic)

1. Sign up at https://testnet.bybit.com/
2. **API → Create API key** (separate from mainnet keys)
3. Add to env:

```
BYBIT_TESTNET_API_KEY=...
BYBIT_TESTNET_API_SECRET=...
```

### B.4 Test connection

```bash
.venv/bin/python -c "
import ccxt
ex = ccxt.binance({'apiKey': '...', 'secret': '...', 'options': {'defaultType': 'future'}})
ex.set_sandbox_mode(True)
print('Balance:', ex.fetch_balance()['total'])
"
```

Expected output: a dict with `USDT` (and possibly `BTC`) entries showing the testnet balance.

For OKX:

```bash
.venv/bin/python -c "
import ccxt
ex = ccxt.okx({
    'apiKey': '...',
    'secret': '...',
    'password': '...',  # the passphrase
})
ex.set_sandbox_mode(True)
print('Balance:', ex.fetch_balance()['total'])
"
```

---

## Section C: Account Capital Layout for Tomorrow's Paper Shadow

Suggested paper-account allocation to mirror the planned live setup:

| Account | Allocation | Notes |
|---|---|---|
| **IBKR paper (DU…)** | $400k total | $100k SVXY, $200k ADR (8 pairs × $25k each), $100k Nikkei |
| **Binance testnet** | $100k USDT virtual | BTC perp short leg |
| **OKX paper (demo)** | $100k USDT virtual | BTC spot long leg |
| **Bridge cost** | ~5 bps per rebalance | 1× rebalance cycle |

Total paper book: ~$600k. This is roughly 60% of a target $1M live book — sized so individual position errors don't dwarf real signal.

IBKR paper accounts default to $1M virtual cash; you can request resets to specific amounts via Client Portal → Settings → Paper Trading Account → Reset.

---

## Section D: Daily startup ritual (once everything is set up)

### Pre-market (08:00 ET)

1. Verify IB Gateway is connected:
   ```bash
   /Users/weeoto/turbo-carnival/.venv/bin/python -c "from ib_insync import IB; ib=IB(); ib.connect('127.0.0.1', 7497, clientId=17); print('OK')"
   ```
2. Verify crypto exchanges are responsive:
   ```bash
   .venv/bin/python -c "import ccxt; print(ccxt.binance().fetch_status())"
   ```
3. Launch dashboard:
   ```bash
   streamlit run research/live/portfolio_dashboard.py --server.port 8503 --server.address 0.0.0.0
   ```
4. Launch SVXY shadow:
   ```bash
   python research/live/svxy_sweet_spot_host.py --account AEGIS --confirm
   ```
5. Launch ADR shadow:
   ```bash
   python research/live/iter_18_adr_host.py --cycle us_close --confirm
   ```
6. Launch BTC funding shadow:
   ```bash
   python research/live/iter_3_btc_funding_host.py --confirm
   ```
7. Launch Nikkei shadow:
   ```bash
   python research/live/iter_6_nikkei_host.py --confirm
   ```
8. Run meta-framework health check:
   ```bash
   python research/live/meta_framework.py list
   ```

### End-of-day (16:30 ET)

1. Review dashboard (P&L, fills, slippage vs model)
2. Run kill-check:
   ```bash
   python research/live/meta_framework.py kill-check
   ```
3. Save daily snapshot (the host scripts auto-persist, but verify the SQLite DB has today's date stamp)

---

## Section E: Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| IB Gateway disconnects every ~24h | Daily auto-logout (IBKR's nightly maintenance) | Set **Auto restart** in API settings; also use `loginRestart` flag in the host scripts to auto-reconnect |
| Binance testnet returns "Invalid API key" | Using mainnet keys against testnet endpoint | Make sure you're using **TESTNET** keys (generated at `testnet.binance.vision` or `testnet.binancefuture.com`), not main keys |
| OKX rejects orders with "permission denied" | Demo mode not selected for the API key | Check demo mode is selected in API settings **AND** in account dropdown; regenerate the key under demo if needed |
| IBKR "no data subscription" error on HK or CME instruments | HKEX or CME sub not active | Check **Client Portal → Settings → Market Data Subscriptions**; subs activate next business day after purchase |
| Dashboard shows stale data | Host scripts not writing, or DB locked | Check SQLite DB file permissions; ensure host scripts are actually running (`ps aux \| grep host`); check `journalctl` or terminal logs for write errors |
| `clientId already in use` on IB connect | Another process holds the same client ID | Either kill the other process, or use a different client ID (e.g., 18, 19) — each concurrent connection must have a unique ID |
| `ccxt.NetworkError` on Binance testnet | Testnet endpoint outage or rate-limit | Testnets are best-effort uptime; retry with backoff; if persistent, switch to Bybit testnet as fallback |
| ADR pair fills only the US side, never HK | HKEX market closed (HK trades 09:30–16:00 HKT) | Schedule ADR cycle for windows where both markets are open OR run staggered `--cycle hk_open` and `--cycle us_close` |

---

## Quick-reference: env vars summary

Final `council/.env.local` should contain at minimum:

```
# IBKR
IB_HOST=127.0.0.1
IB_PORT=7497
IB_CLIENT_ID=17

# Binance (futures testnet)
BINANCE_TESTNET_API_KEY=...
BINANCE_TESTNET_API_SECRET=...

# OKX (demo)
OKX_PAPER_API_KEY=...
OKX_PAPER_API_SECRET=...
OKX_PAPER_PASSPHRASE=...

# Bybit (testnet, fallback)
BYBIT_TESTNET_API_KEY=...
BYBIT_TESTNET_API_SECRET=...
```

Keep `.env.local` out of git (already in `.gitignore`). Rotate keys if they leak.
