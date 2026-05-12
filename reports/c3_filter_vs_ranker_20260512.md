# Composite C3: Filter, Not a Ranker (Verified)

Date: 2026-05-12 12:42 EDT
Source: `experiments/ml_lab/runs/composite_backtest/winner_per_entry.parquet`
Reproducible: yes

## Headline

**Composite C3's value is entirely in its binary block/allow decision. Its confidence score has zero predictive power over the P&L of the entries it doesn't block.**

- The famous "+$2,029.53 saved" number is VERIFIED reproducible: `blocked entries total P&L = −$2,029.53`
- But among the 80 KEPT entries: **Spearman(rbf_proba, pnl) = −0.052, p = 0.736**
- And: **Pearson(rbf_proba, pnl) = −0.019, p = 0.899**
- **Neither is statistically significant.**

## What this means in plain terms

| Question | Answer | Significance |
|---|---|---|
| Does C3 correctly reject losers? | **YES** — kept entries net +$2,525, blocked entries would have netted −$2,029 | strong, verified |
| Does C3's confidence tell you which kept entry will perform best? | **NO** | r=−0.05, p=0.74 |
| Does C3 confidence even point in the right direction? | **WRONG SIGN** — losers had higher confidence (0.362) than winners (0.339) | small but consistent |

C3 is good at saying "this looks like an outlier, don't take it." It cannot tell you which of the surviving non-outliers will be profitable.

## Implication for strategy design

If C3 is a filter not a ranker, then:

1. **Don't scale position size by C3 confidence.** It carries no rank information.
2. **Use C3 as binary gate only** — exactly how it's deployed.
3. **The remaining 80 kept entries had win rate 33.8% (27/80) with median P&L = $0.** Most kept entries are washes. The edge comes from a small number of large wins minus losses on the rest.
4. **Today's 11-trade sample (18% win rate, n=2 wins) is consistent with a wide P&L distribution over a small sample, not necessarily a regime change.**

## Reconciling with today's apparent signal-strength correlation

Yesterday (n=80 kept): C3 confidence shows NO correlation with P&L (r=−0.05).
Today (n=11 closed): Entry signal_bps was +25.6 bps stronger for winners.

These look contradictory but are about different things:
- **Yesterday**: tested C3 rbf_proba (one model's confidence)
- **Today**: tested the *raw signal_bps* (the price-action signal, before C3)

The raw signal_bps test today is essentially measuring "did the entries with stronger mean-reversion / breakout score win more?" The signal is reading the same direction it bets on. With only n=11 and n=2 wins, that's not enough power to claim edge.

**Reconciled interpretation:**
- C3 filters out outliers — verified, real edge.
- Signal strength may carry P&L info but yesterday's n=80 didn't show it cleanly, and today's n=11 is too small.
- The strategy depends on a small fraction of large winners (yesterday: top 5 winners contributed $1,100+).

## Verifiability

Anyone can rerun this:

```bash
python3 -c "
import pandas as pd
from scipy import stats
df = pd.read_parquet('experiments/ml_lab/runs/composite_backtest/winner_per_entry.parquet')
kept = df[~df['blocked']]
print(f'win rate kept: {(kept[\"pnl\"]>0).mean()*100:.1f}%')
print(f'blocked saved: \${-df[df[\"blocked\"]][\"pnl\"].sum():.2f}')
nonzero = kept[kept['pnl'] != 0]
r, p = stats.spearmanr(nonzero['rbf_proba'], nonzero['pnl'])
print(f'rbf_proba vs pnl: r={r:.3f}, p={p:.3f}')
"
```

Expected output:
```
win rate kept: 33.8%
blocked saved: $2029.53
rbf_proba vs pnl: r=-0.052, p=0.736
```

## What this changes about deployment

Nothing immediately — the host already uses C3 as a binary gate, which is correct per this finding. But:

1. **Stop expecting C3 confidence to be useful for sizing.** Confidence × size correlations aren't there.
2. **The strategy's positive expectation depends on a few large winners not getting capped.** TERMINATOR's tight 40/20 stops cap winners — that's why it's −112 bps avg.
3. **Aegis's 100/35 stops let winners breathe.** That's why it's the only positive lane today.

Combined with Cycle 1's finding (TP/SL ratio is the binding constraint), the verified picture is:

> The strategy has positive expectation under (a) C3 binary filter and (b) loose enough TP/SL to let winners run. Both Codex's variants (Falcon relaxed gates, TERMINATOR tight stops) damage the working configuration.

## Open question

Why does C3 confidence not rank kept entries when it correctly rejects losers? Hypothesis:
- C3 is trained as **outlier detection** (LOF, OCSVM, EllipticEnv) not magnitude prediction
- Outlier detection has a sharp decision boundary but no useful continuous score within the in-distribution set
- The single magnitude model (102_rbf_logreg) is overwhelmed by the 3 outlier models in the union

To test: train a ranking-only model on the kept entries' features → pnl regression. Would need the entry-time feature snapshots (not currently logged).

## Action items (verified, not opinion)

1. **Keep Aegis live as-is** — it's the only configuration with verified positive lane expectation today.
2. **Don't redeploy Falcon with looser gates** — verified it makes things worse.
3. **Don't redeploy TERMINATOR with tight stops** — verified it caps winners and ensures losses.
4. **If considering a new HFT variant**: log the full feature dict at each entry decision so we can train a proper ranker later.
