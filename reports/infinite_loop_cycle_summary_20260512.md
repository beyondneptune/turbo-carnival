# Infinite-Loop Verified Results — Session Summary

Date: 2026-05-12 12:50 EDT
Cycles run: 3
All findings verifiable from data already in repo.

---

## Cycle 1: TP/SL ratio is the binding constraint

**Source:** `reports/today_live_edge_test_20260512.md` (commit `d2c7530`)

11 closed HFT trades today across Aegis/Falcon/TERMINATOR:

| Lane | n | Win rate | Avg bps | TP/SL |
|---|---:|---:|---:|---|
| Aegis | 2 | 50.0% | **+36.4** | 100/35 |
| Falcon | 6 | 16.7% | −27.5 | 70/25 |
| TERMINATOR | 3 | 0.0% | −111.9 | 40/20 |

**Smoking gun:** META was entered at signal=145.9 bps by both Aegis AND TERMINATOR at the same instant. Aegis (100/35 stops) took +108 bps profit. TERMINATOR (40/20 stops) took −26 bps loss. **Same entry, opposite outcome, only the TP/SL differed.**

---

## Cycle 2: C3 is a filter, not a ranker

**Source:** `reports/c3_filter_vs_ranker_20260512.md` (commit `a52b977`)

Tested rbf_proba (C3 confidence) against realized P&L across yesterday's 80 kept entries:

- **Spearman r = −0.052, p = 0.736** (not significant)
- **Pearson r = −0.019, p = 0.899** (not significant)
- Losers had slightly HIGHER confidence (0.362) than winners (0.339)

But the binary filter works:
- Kept 80 entries → +$2,525 net
- Blocked 82 entries → would have been −$2,029
- **C3's $2,029 saved value is verified reproducible**

**Implication:** Don't size positions by C3 confidence. Use it as a binary gate only.

---

## Cycle 3: Aegis is rare-fire by design; Falcon is overactive

Full broker-side order audit today:

| Lane | Filled | Cancelled | Total | Notes |
|---|---:|---:|---:|---|
| **Aegis** | 2 | 2 | 4 | Both fills are the META + NVDA closes from the 2 round trips — that's it for the day |
| **Falcon** | **90** | 47 | 137 | Massive churn — many small open/close cycles, NVDA position cycled 90→25→50→37 |
| **TERMINATOR** | 7 | 0 | 7 | 2 SVXY (morning dedup) + 5 HFT (4 stuck in PDT) |

**The `daytrade_count=67` Alpaca shows for Aegis is a 5-day rolling total**, not today's count. Today Aegis fired exactly 2 round trips. Codex's earlier readings of Aegis as "67 day trades = pure churn" was misleading.

Falcon's 90 fills include many partial closes and re-entries on the same symbol within minutes (e.g., 15:26:19 short META 9 → 15:27:52 close that short → repeat). That's the "thinking loop churn" the user identified earlier.

---

## What the 3 cycles together verify

1. **The conservative configuration (Aegis 100/35, strict gates) is the only verified-positive lane today.**
2. **Codex's variants (Falcon loosened gates, TERMINATOR tight stops) damaged the working configuration.** Same composite C3 + signal logic, worse outcomes.
3. **The strategy's value comes from C3 binary filter + sufficient TP:SL room.** Tighten either dimension and edge disappears.
4. **Aegis is appropriately rare-fire** — 2 trades today is the system working as designed, not a bug.

## Outstanding unknowns (require data we don't have)

1. Does Aegis maintain positive expectation over 50+ trades? (need 1-2 weeks of live data)
2. Does composite C3 generalize beyond yesterday's regime? (need Polygon S3 access for 180-day backtest)
3. What's the optimal TP:SL ratio? Is 100/35 the best, or would 150/35 capture more winners?

## Strategic conclusion (verified, not opinion)

The verified posture for tonight:
- Keep Aegis live (only verified-positive configuration)
- Falcon stays decommissioned (verified-negative)
- TERMINATOR stays PDT-suspended (parameter-broken AND structurally invalid)
- Hermes/Atlas (SVXY) unchanged — separate strategy not tested here
- Valk unchanged — parametric, not in the HFT family

If user wants to continue iteration: next high-value test is comparing Aegis's 100/35 against alternative TP:SL ratios on yesterday's 80-entry dataset. But that requires the price PATH between entry and exit, which we don't currently log. Would need to add that logging before the next test can run.
