# Iteration 22 — End-to-End Deep RL Trader on ES 1-Minute Bars (joint signal+execution+sizing)

> **Tone and framing for the reader**: This iteration proposes the highest-risk, most-likely-to-fail method family in the entire run. The honest prior — backed by the Hambly/Xu/Yang 2023 survey, by the unreplicability epidemic in RL-finance papers, and by the basic fact that retail-accessible historical data cannot synthesize the full data-generating process — is that a single end-to-end deep RL agent **will not beat** the best handcrafted ES baseline (iter 2's OFI rule, or iter 12's TFT signal) on a held-out 2024-2026 window. The point of writing this candidate is **not** to argue that it will succeed; it is to specify a falsification protocol strict enough that, in the small probability case where it *does* work, the result is trustworthy, and in the much larger probability case where it doesn't, we kill it cheaply within 90 days and learn something concrete about *why* end-to-end RL failed on this problem. Read every section through that lens.

## 1. Thesis (one sentence)

A single Proximal Policy Optimization (PPO) — or equivalently configured Soft Actor-Critic (SAC) — agent, fed a **raw market state vector of ~360 features** (last 60 1-minute ES bars × 6 OHLCV + 2 L1 book features × 60 bars + 100 contextual features including VIX level, term-structure, time-of-day, and prior-session levels) and outputting a target ES position size in {−2, −1, 0, +1, +2} contracts each minute, can — by **jointly** optimizing direction, entry timing, exit timing, and size against a realized-P&L-minus-cost-minus-drawdown-penalty reward — produce a held-out 2024-2026 net Sharpe that exceeds **the better of iter 2 (OFI rules) and iter 12 (TFT signal) by ≥ 1.0 Sharpe units**, on the same 1-minute decision cadence and the same trade-count budget; **the claim is falsified if, on ≥ 500 OOS trades, the uplift over the best handcrafted baseline is < 1.0 Sharpe, OR the agent's max drawdown exceeds 1.5× the best baseline's max drawdown, OR any hard-constraint guardrail (>3 contracts, >2 trades/hour, daily loss cap) is touched by the policy itself in OOS replay** — and a kill within 90 days of live shadow start is mandatory unless **all three** of those conditions are cleared.

The prior is that this iteration fails. The 1.0-Sharpe uplift bar is deliberately stringent because the complexity, opacity, and operational fragility of an end-to-end RL agent must be paid for by a *large* edge — not by a 0.2-Sharpe marginal improvement that could trivially be a sampling artifact. RL is plausible-but-unproven on this problem; verdict criteria are strict because the prior is failure.

---

## 2. Market / asset class / venue

- **Instrument:** CME E-mini S&P 500 futures, **front-month** (ESZ5, ESH6, ESM6, ESU6 …). Continuous front-month stitched with explicit roll on the 8th calendar day prior to expiry. Roll-adjustment is a separate column used **only** for cross-bar features that need ≥1-session lookback; intraday P&L uses unadjusted prices to avoid look-ahead.
- **Symbol on IBKR:** `ES` with `secType=FUT`, `exchange=CME`, multiplier `$50/point`, tick `0.25` = `$12.50`.
- **Primary exchange:** CME Globex.
- **Broker:** Interactive Brokers (IBKR Pro) via `ib_insync` (Lesson 8 — battle-tested SDK only; no hand-rolled FIX).
- **Settlement / margin:** Cash-settled. IBKR intraday day-trade margin ~ $500-$1,000/contract; overnight initial ~ $13,200-$14,500/contract. We trade **intraday-only**, hard flatten by 15:55 ET. No overnight exposure under any circumstance — this is enforced by a wall-clock guardrail outside the policy, not by the policy itself.
- **Tax regime (US):** Section 1256 60/40 long-term/short-term blend.

**Same instrument** as iter 2 and iter 12. This is deliberate. The whole point of the head-to-head test in §6 is to hold the underlying constant so that the *method* (rules vs. supervised transformer vs. end-to-end RL) is the variable. Per Lesson 6, if two methods on ES survive to live simultaneously, their realized correlation must be < 0.6 for both slots to fund; otherwise the lower Sharpe one is de-allocated. Given that all three are nominally trading the same direction off the same tape, the correlation gate will be the **binding** constraint, not the Sharpe gate, if more than one survives.

---

## 3. Time horizon

**1-minute decision cadence; intended holding period 5-60 minutes; hard exit by 15:55 ET.**

- The agent emits one action every minute at the bar close. Most minutes the optimal action is "no change" — the action space includes a zero / stay-put option; if the policy emits *any* action every minute we expect it has overfit and we kill (see §10).
- A "trade" is defined as any minute where `target_position` differs from `current_position`. The system enters new positions or scales down toward zero at 1-minute granularity.
- This is **slower** than iter 15 (sub-minute execution agent at 1 Hz inside a parent-order window) and **same speed** as iter 2 (1-min OFI rule), but **faster** than iter 12 (5-min TFT). The choice of 1 minute is a compromise: faster than 5 min gives the RL agent more decisions per session (~390 vs. ~78) and so more reward signal per training day; slower than sub-minute keeps execution simple (we cross the bid-ask with marketable-limit IOC; no need for an inner execution-scheduling agent) and avoids the latency arms race we explicitly opted out of.

---

## 4. Method family and specific algorithm

### 4.1 Family

**Deep reinforcement learning, end-to-end joint optimization** of (signal direction, entry timing, exit timing, position size). This is the **first** iteration in the run where a single learned policy decides all four dimensions simultaneously. Iter 12 learns *signal only* via supervised next-bar regression with a separate handcrafted entry/exit/sizing rule. Iter 15 learns *execution only* via RL on top of a parent-order signal it does not control. **Iter 22 is the union: one network, one reward, all four decision dimensions.**

### 4.2 Specific algorithm — two backbones, pick one after sim ablation

**Default: PPO** (Schulman et al. 2017) with a clipped surrogate objective. Rationale:
- PPO is the workhorse for noisy reward signals and continuous-or-mixed action spaces; it is the same algorithm used in iter 15 and the published RL-for-execution literature (Hambly/Xu/Yang 2023, §4) — using the same algorithm across iter 15 and 22 means we share infrastructure, debugging tooling, and trust calibration.
- PPO's trust-region clipping is empirically the most stable choice on financial reward signals, which are dominated by noise from price-process variance.

**Backup: SAC** (Haarnoja et al. 2018). Off-policy, sample-efficient, entropy-regularized. If PPO underperforms the baseline in sim because of *training-data inefficiency* (very plausible — we have ~5M minute bars from 2019-2023, which is "small" by deep-RL standards), SAC's experience-replay buffer can wring more bits per sample. SAC also has a built-in entropy term that discourages premature mode collapse onto a single trivial policy (e.g. "always hold zero contracts"), which is *the* dominant trivial local optimum on this problem.

**Both backbones share**:
- Network: 3-layer MLP, 256 hidden units, ReLU. Separate value head and policy head (actor-critic). Total params ~150k. Larger than iter 15's executor net (~10k) because the state space is much larger (~360 features vs. 12), but still **tiny** by general-deep-RL standards. Larger networks are forbidden — overparameterized policies overfit the regime distribution and fail catastrophically on novel regimes.
- Optimizer: Adam, lr 3e-4, gradient clip 0.5 (PPO) or twin-Q with lr 3e-4 (SAC).
- Discount factor γ = 0.95 on minute steps (≈ 20-minute effective horizon). Forces the agent to optimize for trades that close within the holding-period budget; long-horizon credit assignment is intentionally weak.
- Reward clip: per-step reward clipped to ± 4 σ of historical 1-minute P&L per contract, to prevent a single fat-tailed bar from dominating gradient updates.

### 4.3 State space — explicit feature list (~360 features)

**Sequence portion (60 bars × 6 = 360 base features), flattened then projected through a 256-dim Linear before being concatenated with the contextual block:**

For each of the last 60 1-min bars (most recent first):
1. Log return (close-to-close)
2. Log range (high − low) / close
3. Log volume z-score (rolling 30-bar)
4. Signed-volume proxy: sign(close − open) × volume, z-scored
5. Close − VWAP (session, in ticks)
6. L1 bid-ask spread (in ticks; from IBKR `reqMktData` BID/ASK snapshots; **NOT** L2 — we explicitly do not depend on full book to keep the data subscription cheap and avoid the colocation arms race)

Total raw bar features: 60 × 6 = **360**. Flattened for the MLP backbone. A future v2 with a small Transformer encoder over the bar sequence is plausible if v1 validates; v1 sticks to MLP-over-flat for simplicity and to share more architecture with iter 15.

**Contextual portion (~100 additional features), concatenated to the projected sequence embedding before the heads:**

- VIX level (raw).
- VIX 1-min change.
- VIX − realized-vol gap (VIX − annualized 60-bar realized vol).
- VX1-VX2 futures spread (term-structure proxy).
- DXY level z-score (60-bar).
- ZN (10y) 1-min return.
- NQ 1-min return.
- CL 1-min return.
- Minutes since 09:30 ET (linear, normalized).
- sin/cos encoding of minute-of-day.
- sin/cos encoding of day-of-week.
- Distance to prior-day close (in ATRs).
- Distance to prior-day high (in ATRs).
- Distance to prior-day low (in ATRs).
- Distance to session high (in ATRs).
- Distance to session low (in ATRs).
- is_FOMC_day / is_NFP_day / is_CPI_day / is_OPEX_day / is_quad_witch (5 binaries).
- Time-to-next-macro-release in minutes (capped at 180).
- Account-state features (forces the policy to be size-aware):
  - `current_position` (signed, in contracts, clipped to ±3).
  - `unrealized_pnl_bps` on current position.
  - `realized_pnl_today_bps`.
  - `trades_remaining_this_hour` (out of the 2 allowed).
  - `minutes_to_force_flat` (i.e. minutes until 15:55 ET hard close).
  - `daily_loss_remaining_bps` (out of the daily loss cap).
- Padding / reserved slots to round to 100. Exact final count is 96-104 depending on which optional data feeds are live; the network input dim is fixed at the start of training and never expanded post-train.

**Normalization**: every feature is z-scored against its rolling 20-day mean/std *as observed up to the bar*, then clipped to ± 5 σ. The clip is critical — without it, a single 2020-style vol explosion drives the network into a state-space region it has never seen and the policy's action distribution becomes uninterpretable. The clip *guarantees* that the inference-time state vector stays bounded.

### 4.4 Action space — discrete with safety clip

**Default: discrete, 5-way action head. Outputs `target_position ∈ {−2, −1, 0, +1, +2} contracts`.**

- The policy is a softmax over 5 outcomes per minute. The chosen output is the *target* position; the executor delta-trades to get there (e.g. if currently +1 and target +2, place a buy-1; if currently +1 and target −1, place a sell-2).
- **Hard constraints, enforced outside the policy (not learned):**
  1. `|target_position| ≤ 2` from the action space itself. **Plus** an external belt-and-braces clip that any order exceeding `|current_position + delta| > 3` is rejected at submit-time. (Spec said up to ±3; we use ±2 as the policy's range and reserve ±3 as the external hard ceiling. This margin lets the operational guardrail catch a misconfigured policy without ever firing in normal operation.)
  2. **≤ 2 position changes per rolling 60-min window.** If the policy emits a change after 2 are already used in the trailing hour, the change is rejected and the position is held. The agent observes `trades_remaining_this_hour` as a state feature and learns to budget; if it doesn't, the guardrail catches it.
  3. **Daily loss cap halts agent.** If `realized_pnl_today` reaches −$1,500 per contract baseline ÷ scaling, the policy is forced to flat and disabled for the rest of the session. (Exact figure in §7.)
  4. **Force-flat at 15:55 ET** regardless of policy output.

**Alternative considered (rejected for v1): continuous action ∈ [−3, +3] with safety clip.** Gives finer-grained sizing but expands the action surface that adversarial replay must cover. Reserved for v2 only if v1 validates.

### 4.5 Reward

```
reward_t = realized_pnl_delta_t  −  cost_t  −  λ_dd · drawdown_penalty_t  −  λ_turn · |delta_position_t|
```

Where:

- `realized_pnl_delta_t` = mark-to-market change in P&L over the minute, in **dollars per minimum trade unit** (i.e. per 1 contract), scaled to bps of a $50k notional reference account so that the magnitude is order-1 and gradient updates are well-scaled.
- `cost_t` = explicit transaction cost on any position change: commission (IBKR ~$0.85/contract round-trip → $0.425/side), exchange/regulatory fees (~$1.40/contract round-trip), and slippage (1 tick worst-case for ES at ±2 contracts = $12.50/contract, modeled via the sim's market-impact function — see §5 / §6).
- `drawdown_penalty_t` = max(0, current_drawdown_from_session_high) — i.e. zero unless we're below the session high — multiplied by a coefficient `λ_dd = 2.0`. **Asymmetric**: drawdowns are penalized twice as hard as the corresponding profit is rewarded, which is the closest one-line implementation of CVaR-aware reward that doesn't require a separate quantile estimator. This is the iteration's principled answer to "RL agents that maximize mean reward learn dangerously risky policies": the penalty term shapes the policy toward Sharpe-like behavior even though Sharpe is not directly differentiable.
- `λ_turn · |delta_position_t|` with `λ_turn = 0.5 bps per contract changed`. Small turnover penalty to discourage churn beyond what `cost_t` alone discourages. Calibrated such that it slightly under-penalizes a *useful* trade (so the agent will trade if it sees real edge) but materially penalizes flip-flopping.

**Reward shaping rule: any additional shaping term is forbidden in v1.** Specifically: no per-bar "alignment with future return" shaping, no auxiliary classification losses on bar direction, no curriculum on trade size. Shaping is the most common reward-hacking vector in published RL-finance work (Hambly/Xu/Yang 2023 §6.3) and is one of the named failure modes in §10.

### 4.6 Why end-to-end RL, given the skeptical prior?

**The genuine argument** — not "because RL is cool":

- Iter 12 produces a *signal* but is bolted to handcrafted entry/exit/sizing. Iter 15 produces *execution* but receives signals it does not control. Each of those is a *decomposed* approach: factor the problem into pieces and optimize each piece. The decomposition has a cost — the entry rule in iter 12 cannot adapt its exit threshold conditional on intra-trade order-flow microstructure, because the entry rule and the exit rule are written in different files and share no state.
- End-to-end RL collapses the decomposition. The same network sees the entry decision, the holding-period state, and the exit decision; it can in principle learn, e.g., "enter aggressively when book imbalance is rising AND VIX is falling AND I am currently flat, but exit fast — within 3 bars — when subsequent book imbalance reverses, regardless of unrealized P&L."

**The skeptical counter-argument, which is the realistic prior:**
- The published RL-finance literature (per Hambly/Xu/Yang 2023, §5) is *dominated* by results that fail to replicate. The survey explicitly names "selective reporting of the winning seed", "training-test contamination via shared embedding pre-training", and "unreported transaction costs" as the most common ways the field has overstated edge.
- Sirignano (2019) "Deep Learning for Limit Order Books" is sometimes cited as evidence that deep nets find tradable structure in microstructure; in practice the predictive power Sirignano reports is on next-tick mid-price direction (a different problem) and net-of-cost trading edge from those signals is much weaker.
- Mnih et al. (2013) on DQN-for-Atari is the foundational deep-RL anchor, but Atari has unlimited free training data (the game is the simulator); financial markets do not, and the sim-to-real gap is exactly the gap between Atari's simulator-is-the-environment property and our simulator's necessarily-imperfect approximation of live ES tape.
- **Therefore**: we expect failure. We design the test so that failure is recognized cheaply and a 1.0-Sharpe lift over the best handcrafted ES baseline is the *minimum* uplift that would justify the operational complexity of end-to-end RL.

### 4.7 Anchor literature

- **Mnih et al. (2013)** "Playing Atari with Deep Reinforcement Learning" — DQN foundational.
- **Schulman et al. (2017)** "Proximal Policy Optimization Algorithms" — PPO.
- **Haarnoja et al. (2018)** "Soft Actor-Critic" — SAC backup.
- **Sirignano (2019)** "Deep Learning for Limit Order Books" — deep-net evidence on microstructure (weak for net-edge claims; cited honestly).
- **Hambly, Xu, Yang (2023)** "Recent advances in reinforcement learning in finance", *Mathematical Finance* — comprehensive survey. **The single most important reference for this iteration because it is the source of the skepticism we are building the validation plan around.**
- **Henderson et al. (2018)** "Deep Reinforcement Learning that Matters" — methodological warning: deep RL results are notoriously seed-sensitive, and most published claims do not survive a multi-seed re-run. We adopt their multi-seed reporting protocol in §6.

---

## 5. Data requirements

- **Live**:
  - IBKR US Securities Snapshot/Bundle for ES NBBO + last-trade tick: ~$10-15/mo all-in (CME data fees apply on top, ~$10-25/mo for non-pro). Sufficient for the L1 state features.
  - VIX, NQ, ZN, CL, DXY: IBKR cross-asset bundle ~$10/mo or Polygon Free for delayed (live deployment needs real-time, but the agent's decision cadence is 1 min — a few seconds of delay is tolerable as long as it's *consistent* with training delay).
  - One symbol of L1 + cross-asset 1-min bars = trivial bandwidth, well within IBKR `ib_insync` rate limits.
- **Historical / training**:
  - **Polygon Stocks Starter ($29/mo)** does not cover futures. We need **databento CME L1+trades** or **Polygon Indices+Futures** (~$200-300/mo for the data acquisition phase). Database S3 archive for ES front-month 1-min OHLCV + L1 bid-ask, 2019-01-01 → 2023-12-31, plus VIX/NQ/ZN/CL/DXY 1-min.
  - 5 years of 1-min ES + 5 cross-asset symbols ≈ ~50 GB compressed. Trivial storage.
  - Training compute: 1× GPU (RTX 4090 or A100). Single training run ≈ 24-72 GPU-hours for PPO with vectorized envs; SAC slightly more. Hyperparam sweeps + ablations + multi-seed re-runs: budget **~1,000 GPU-hours = ~$400 on spot A100**.
- **Storage**: ~50 GB historical + ~500 MB final policy + ~2 GB/mo live audit log (state→action→reward→fill at every decision, mandatory per §12).
- **Cost**: $300/mo databento for ~4 months of data acquisition + active training, then drop to $20-40/mo steady state. Plus ~$400 one-time GPU compute. Plus $25-50/mo IBKR live data. **Total research budget: ~$1,800. Steady-state: ~$60/mo.**
- **Latency**: Decision cadence is 1 min. End-to-end round trip needs to fit comfortably in 1 minute, which is trivial:
  - Bar close → state vector assembly: ≤ 200 ms.
  - State vector → PPO/SAC inference (150k-param MLP on CPU): ≤ 10 ms.
  - Order intent → IBKR API → exchange ack: ≤ 250 ms p99.
  - Total: ≤ ~500 ms typical, against a 60,000 ms budget. **This is not a latency-sensitive iteration.** We deliberately picked 1-minute cadence to opt *out* of the latency arms race.

---

## 6. Validation plan

**The most important section.** Lesson 1 (validate before deploy) is enforced as a four-stage gate with explicit kill criteria at each stage. The validation plan is designed under the assumption that **the agent will fail at some stage**, and the question we're asking is *which* stage it fails at and *why*.

### 6.1 Walk-forward design (in simulator)

- **Train window**: 2019-01-01 → 2023-12-31 (5 years, ~1,250 trading days). Spans the COVID-2020 vol regime, the 2021-2022 melt-up, the 2022 rate-hike drawdown, and the 2023 mean-reverting chop. **Deliberately heterogeneous** — if the agent can only learn one regime, we want training to expose that fact.
- **Validation (sim-OOS) window**: 2024-01-01 → 2024-12-31 (1 year, ~250 trading days). Used **only** for hyperparameter selection and one-shot model selection. The model selected here is never re-trained on it.
- **Test (frozen sim-OOS) window**: 2025-01-01 → 2026-05-12 (~17 months, ~340 trading days). Used **once**, at the end, for the sim-side verdict. No iteration on this window. **Hard wall.**
- **No same-data backtest** (Lesson 7): three disjoint windows.

**Purge and embargo**: Because the 1-minute reward signal has serial correlation (intraday autocorrelation in 1-min returns is small but non-zero), we apply a **1-session embargo** (i.e. drop the day immediately following the train→validation and validation→test boundaries) and a **purged k-fold** internal to the train window for hyperparameter selection (López de Prado *Advances in Financial Machine Learning* §7).

### 6.2 Multi-seed protocol (Henderson et al. 2018)

For every backbone+hyperparameter combo, train **5 independent seeds**. Report median, p25, p75, and worst-seed performance on the sim-OOS test window. **Promote the median, not the cherry-picked best.** If the spread between best-seed and worst-seed Sharpe on the test window is greater than 0.8, declare the result *unstable* and treat the median as a non-validated signal (i.e. kill, not promote).

### 6.3 Domain randomization / sim-to-real gap mitigation

The simulator must approximate IBKR live fills. The honest position is that *no* simulator does this perfectly. The mitigations:

- **Slippage randomization**: at each training episode, sample the per-trade slippage from a distribution covering {0 ticks, 0.5 ticks, 1 tick, 2 ticks} with weights skewed toward the worse end (e.g. 0.1 / 0.2 / 0.4 / 0.3). The agent does **not** observe the slippage realization; it learns a policy that's robust to the *distribution*.
- **Spread randomization**: bid-ask spread is the per-bar empirical spread × a multiplier sampled from {0.5×, 1.0×, 1.5×, 2.0×}. Trains the agent to handle wide-spread regimes (after-hours, pre-news).
- **Latency randomization**: order submission latency sampled from {50, 150, 500, 1500} ms. The agent uses bar-close prices but the fill comes back at a randomized delay; if the policy is fragile to delay, training will surface it.
- **Adversarial market replay**: a held-out 5% of train episodes are sampled from "stress" days (top-5% absolute return days in training history: COVID March 2020, GFC-style days reconstructed from 2008-era simulators when available, FOMC surprise days). The agent sees these often during training so its behavior on stress days is not extrapolated.
- **Cost over-estimation in training**: training costs are deliberately set 1.5× the realistic costs. The agent therefore over-pays per trade in training relative to live; if the agent still finds edge, the edge survives lower live costs with margin. **One-sided slack.**

### 6.4 Stage-gate progression (sim → paper → tiny-real → real)

This mirrors iter 15's stage-gate structure. Identical philosophy: any stage's failure is a kill, not a "tune and retry."

**Stage 0: sim validation.** Frozen sim-OOS window 2025-01-01 → 2026-05-12. Verdict criteria below in §11. If median-seed Sharpe uplift over the better of iter 2 and iter 12 is < 1.0 on this window, **kill in sim — do not promote to paper**.

**Stage 1: paper / shadow.** 30 continuous trading days, live IBKR paper account, real ES quotes, real fills (paper fills are IBKR-side approximated against live quotes), real latency. Targets ≥ 30 days × ~5 trades/day = **~150 paper trades**. Pass criteria: (a) Sharpe in shadow ≥ 0.8× the sim-OOS Sharpe, (b) max DD in shadow ≤ 1.2× the sim-OOS max DD, (c) no hard-constraint guardrail (position, trades-per-hour, daily loss) ever fired by the policy. Any failure → kill.

**Stage 2: tiny-real.** 30 trading days, **1 contract max**, daily loss cap **$200**, hard kill at any week with cumulative loss > $400. Targets **~150 live trades**. Pass criteria: (a) realized Sharpe ≥ 0.6× the paper Sharpe (we expect degradation from paper to real, but a >40% drop indicates a sim-to-real gap we don't trust), (b) realized slippage in bps ≤ the slippage budgeted in training, (c) broker-truth divergence (any position mismatch with IBKR — Lesson 4) zero tolerance.

**Stage 3: full size.** Up to 3 contracts max (the spec ceiling), daily loss cap as in §7. Only after **all** of Stages 0/1/2 have passed. Even at full size, the policy is **inference-only** — no online learning, no exploration in production. This is non-negotiable (named failure mode in §10).

### 6.5 Head-to-head test against iter 2 and iter 12

**This is the load-bearing test.** During the sim-OOS test window (2025-01-01 → 2026-05-12) and the paper window, all three candidates (iter 2 rules, iter 12 TFT, iter 22 RL) must be run **on the same bars** in parallel and their trades logged with the same trade-account model. The metric is **Sharpe difference (RL − best_of_iter_2_iter_12)**, computed on the same calendar window.

- **Minimum OOS trade count for verdict: 500** trades on the sim-OOS test window, then ≥150 on paper, then ≥150 on tiny-real (so ≥800 cumulative real-or-paper trades before promotion).
- **Statistical test**: paired bootstrap (5,000 resamples) on per-trade P&L of RL minus per-trade P&L of best_baseline, same-bar-paired. Promote only if the 95% bootstrap CI of the Sharpe difference excludes 1.0. (Note: 95% CI excluding 1.0, not 0.0. We are testing the *uplift is ≥ 1.0*, not the *uplift is positive*. The stricter bar is the point of this iteration.)

### 6.6 Hyperparameter discipline

- All hyperparameters fixed before touching the test window.
- One pass through the test window. If we touch the test window for any reason — even "just to look" — the test window is burned and a new disjoint window must be carved out (which will not exist on this calendar, so a re-validation requires waiting for new data).
- Hyperparameter selection (lr, γ, clip ratio for PPO; α for SAC; network width; reward coefficients λ_dd, λ_turn) happens **only** in the train and validation windows.

---

## 7. Risk envelope

- **Max position size**: ±2 contracts policy-side, ±3 contracts external-guardrail-side. Notional at full size: 2 contracts × ~5,500 ES × $50/point ≈ **$550k notional**. On a $50k baseline account this is ~11× leverage on notional, which is within IBKR intraday margin but is the **upper bound** of what we tolerate.
- **Max position as % of account**: At a $50k baseline, $550k notional ≈ **11× notional leverage**. We do not size up to "% of account" for futures the way equities iterations do; instead the limit is `|contracts| ≤ 2` enforced both in the action space and by external hard clip.
- **Daily loss cap (absolute)**: **$1,500** for the $50k baseline (3% of account). Hit → policy is disabled for the rest of the session, position is force-flattened by the executor.
- **Daily loss cap (%)**: 3% of starting NAV.
- **Drawdown trigger that halts trading**: Trailing **20-session** peak-to-trough drawdown reaching **−10%** halts the agent entirely until human review. This is a "real loss" trigger, not a guardrail trip.
- **Circuit breaker rule**: 3 consecutive losing trades on the *same day* triggers a 2-hour cooldown (no new entries; existing position held to its exit). 3 consecutive losing days triggers an overnight review before next session.
- **Hard action constraints, enforced outside the policy**:
  - `|target_position| ≤ 3` (the action space tops out at 2; the guardrail is the belt-and-braces clip at 3).
  - `≤ 2 trades per rolling 60-min window`.
  - `daily_realized_loss ≤ −$1,500` → policy disabled until next session.
  - `force_flat at 15:55 ET` regardless of policy output.
  - **Kill switch defaults to flat**: a watchdog process pings the agent every 5 s; missed ping → all positions force-flattened at market, agent disabled until manual re-start. This is non-negotiable (§12).
- **Correlation rule with already-deployed strategy**: If iter 2 (OFI rules) and/or iter 12 (TFT) are already deployed on ES, **iter 22 cannot fund as a third slot on the same instrument** unless its realized correlation with the funded ones is < 0.5 (stricter than Lesson 6's < 0.6 because three concurrent ES strategies is operationally dangerous). In practice, since all three trade off the same tape in nominally the same direction, we expect correlation to be high and only one of the three to survive.

---

## 8. Execution plan

- **Broker / SDK**: IBKR Pro via `ib_insync`. Same SDK as every other iteration that touches IBKR (Lesson 8).
- **Order type**: **Marketable-limit IOC** as default. Each delta-trade is submitted as `LMT, TIF=IOC` priced 1 tick through the near-side NBBO for buys (or 1 tick through the bid for sells). If unfilled IOC, retry once at 2 ticks through; if still unfilled, **abort the trade** and the next minute's action recomputes from the new bar.
- **Why marketable-limit IOC, not market**: Lesson 4 (broker truth) and protection against fat-finger fills during fast-moving regimes. Market orders on ES during an FOMC surprise can fill several ticks worse than the L1 quote at submit time; IOC caps that worst case.
- **Why not use iter 15's RL execution agent as the inner loop?** Tempting (and the right v2 design), but **explicitly out of scope** for v1. Stacking two RL agents (one for direction+sizing, one for execution) compounds debugging difficulty and reward-attribution opacity. v1 keeps execution stupid and deterministic so that if the system fails, we know it's the directional/sizing policy that's at fault, not the executor. v2 may swap in iter 15's executor only if both iter 15 and iter 22 pass independent validation.
- **Expected slippage budget**: 1 tick ($12.50) typical, 2 ticks worst-case during news. Measured live as (fill price − NBBO mid at submit time) × side, in ticks. Logged every fill. If trailing 100-trade slippage exceeds 1.5 ticks median, halt and recalibrate cost assumptions in training.
- **Broker rejects**: any IBKR rejection (margin, halted instrument, exchange error) → position frozen, no retry attempt by the policy, alert to operator. The agent is not allowed to "react" to rejections in any learned way — rejections are out-of-distribution from training and the safe response is to do nothing until a human reviews.

---

## 9. Capacity

- **AUM ceiling before edge degrades**: ES front-month has ADV of ~1.5-2 million contracts/day during US RTH. A 2-contract max position is a vanishingly small fraction (~10⁻⁶) of ADV; the strategy has **enormous headroom** in absolute terms.
- **Real ceiling**: governed by the agent's slippage assumption, not by ADV. As soon as a single trade moves >1 tick of ES, the training reward function is mis-specified and the policy drifts off-distribution. Empirically, that threshold is around **20-50 contracts per market order on ES during normal RTH**. Therefore the *strategy* (not the venue) ceiling is around **$5M-$10M of account equity** at the 2-contract policy size, scaling roughly linearly until the per-trade size starts moving the market.
- **Stated ceiling for this iteration**: **$1M AUM** as a soft cap. At higher AUM, re-train the slippage assumption upward and re-validate. **Hard cap: $5M.** Above $5M, the iteration is invalid as-stated and a fresh iteration (potentially using the iter 15 RL executor as the inner loop) is required.
- **ADV consumption at full size**: At 2 contracts per minute (worst-case turnover) over a 390-minute session = ~780 contracts/day. Against ADV of ~1.5M = **0.05% of ADV**. Negligible.
- **Concentration risk**: single-name (ES is the only name), but the underlying SPX is a 500-component index, so single-name issuer risk is structurally diversified. Single-sector concentration is the broad US large-cap exposure — same as everyone else in the run; nothing iter-22-specific.

---

## 10. Failure modes

**At least 6 named failure modes (more than the required 3 because RL has *more* ways to fail).**

1. **Reward hacking — the agent learns to game the metric.** Most likely manifestation: agent learns to take many tiny scalps that *technically* maximize the reward function while exploiting the simulator's cost model. Detected in metrics by: (a) realized PnL per trade in sim much higher than per-trade PnL of iter 2/iter 12 baseline, (b) trade frequency at or above the 2-trades/hour guardrail constantly, (c) live-vs-sim PnL gap > 50%. Verdict: **kill** — reward hacking that survives stage 0 will be ruinous in production.

2. **Regime mismatch — training distribution ≠ live distribution.** Manifestation: the 2024-2026 OOS window includes a regime not adequately represented in 2019-2023 training (e.g. sustained low-vol melt-up with VIX < 12, or a 2008-style credit shock). Detected by: live state-vector L2-distance to nearest training-set neighbor exceeding the 99th percentile of training-set self-distance. If this fires, **disable policy** until either training is extended or the regime passes.

3. **Exploration in production.** A subtle but lethal failure: someone leaves an exploration-noise term on in the deployed policy. The agent then "explores" with real money. **Hard control**: deployment pipeline strips exploration from the policy at export time; a deployment-time integration test verifies that the same state vector produces an *identical* action across 100 inference calls. If the test fails, deploy is blocked. Inference in production is **deterministic** — argmax of policy logits, no Gumbel-softmax sampling, no entropy noise.

4. **Seed instability (Henderson et al. 2018).** Manifestation: the chosen seed is a high-luck draw and a re-trained policy on a fresh seed performs much worse. Detected by the 5-seed protocol in §6.2; if best-vs-worst seed Sharpe spread on the test window is > 0.8, declare unstable and kill.

5. **Sim-to-real gap.** Manifestation: paper-trading Sharpe is materially below sim-OOS Sharpe (e.g. paper Sharpe < 0.8× sim Sharpe) because the simulator's market-impact / spread / latency model is too generous. Detected at Stage 1 of §6.4. Verdict: **kill — do not pass to tiny-real.** No "but the sim said it would work" appeals.

6. **Overfitting to a specific micro-regime.** Manifestation: the policy is essentially a memorized strategy for one specific calendar window (e.g. "the 2022 mean-reverting chop") that doesn't generalize. Detected by: per-quarter Sharpe breakdown on the train window — if Q1 Sharpe is 3.0 and Q2 Sharpe is 0.1, the policy has learned a regime, not an edge. Verdict: re-train with stronger regularization (smaller network, higher entropy bonus in PPO, larger dropout) or kill if re-training doesn't fix.

7. **Operational opacity (agent does something unexpected in an unfamiliar regime).** Manifestation: a single FOMC surprise or flash crash sends the L1 features outside their training distribution; the policy outputs a position-3 long during a -5% drop because the state is so OOD that the network's output is essentially random. **Mitigations**: (a) the 5σ feature clip in §4.3 bounds inputs, (b) the trades-per-hour cap in §7 limits damage from a single erroneous decision, (c) the kill switch defaults to flat (§12). Verdict on detection: **manual kill, full session halt**.

---

## 11. Falsifiable verdict criteria

### Promote to live capital

**ALL of the following must hold** (no two-out-of-three; this is RL on real money, the bar is high):

- Sim-OOS test window (2025-01-01 → 2026-05-12): median-of-5-seeds annualized **Sharpe uplift ≥ 1.0** over the better of iter 2 and iter 12, paired-bootstrap 95% CI excludes 1.0, on ≥ **500 OOS trades**.
- Sim-OOS max drawdown ≤ **1.5×** the best baseline's max drawdown on the same window.
- Multi-seed spread (best minus worst Sharpe) ≤ **0.8** on test window.
- Paper / shadow (Stage 1, 30 days, ~150 trades): realized Sharpe ≥ **0.8×** sim-OOS Sharpe; zero hard-constraint guardrail trips; zero broker-truth divergences.
- Tiny-real (Stage 2, 30 days, ~150 trades, 1-contract size): realized Sharpe ≥ **0.6×** paper Sharpe; realized slippage within budget; zero broker-truth divergences.
- Correlation with any concurrently-deployed ES strategy (iter 2 or iter 12, if either is live) measured in shadow: **< 0.5** (stricter than Lesson 6 default of 0.6, because three ES strategies on one account is operationally dangerous).

### Continue shadow / extend evaluation

- Sim-OOS uplift between **0.6 and 1.0** Sharpe over best baseline: continue shadow for 30 more paper days at no cost, then re-evaluate. **Do not promote to tiny-real on a marginal sim result.**
- Sim-OOS uplift ≥ 1.0 but seed instability ≥ 0.8: re-train with stronger regularization, re-test once. Test window is *not* re-burned — we are testing the *new* policy against the *same* held-out data.

### Kill immediately

**ANY ONE of the following triggers a kill within 90 days of going live**:

- Sim-OOS uplift < 1.0 Sharpe over best handcrafted baseline.
- Sim-OOS max DD > 1.5× best baseline's max DD.
- Any hard-constraint guardrail (position > 3, trades-per-hour > 2, daily loss cap) triggered by the **policy itself** in OOS replay or live.
- Paper Sharpe < 0.8× sim Sharpe (sim-to-real gap exposed).
- Tiny-real Sharpe < 0.6× paper Sharpe (real-to-paper gap exposed).
- 3-day cumulative DD > 5% on tiny-real.
- Daily loss cap breach on live (even at tiny size).
- Any broker-truth divergence (Lesson 4).
- Seed-instability spread > 0.8 Sharpe on test window.
- Multi-quarter regime breakdown (any quarter's Sharpe < 25% of the cross-quarter mean).
- Realized slippage > 1.5 ticks median across trailing 100 live trades.

**The 90-day clock starts at first live paper-trade.** If at day 90 we are not yet past Stage 2 (tiny-real) with all promote-to-live criteria green, **the iteration is shelved** and a written post-mortem is required before any reboot.

---

## 12. One-paragraph operational risk note

End-to-end deep RL agents are notoriously hard to debug — a misbehavior in production may be caused by (i) a drifted feature normalization statistic that has silently moved off training distribution, (ii) a state-vector serialization bug that flips two features, (iii) an undetected model-export error that loaded the wrong checkpoint, (iv) an actually-novel market regime, or (v) genuine policy degradation — and these five causes are operationally indistinguishable from output alone. **Mandatory operational controls**: (a) full audit log of `{timestamp, raw_state_vector, normalized_state_vector, policy_logits, sampled_action, target_position, executed_orders, fills, realized_pnl, drawdown_penalty, reward}` at every decision tick, stored in append-only durable storage with hourly checksum verification (~2 GB/mo retention, kept for 18 months minimum); (b) a parallel "shadow inference" process running the same model against the same state vector on a separate machine, with a divergence alert if predictions differ — catches both serialization bugs and silent model-file corruption; (c) a deployment-time integration test that verifies the loaded policy produces the *exact* same actions as the test-window-frozen reference policy on a fixed set of 1,000 stored state vectors (any single mismatch blocks deployment); (d) **kill switch defaults to flat** — a watchdog process pings the agent every 5 s, missed pings → all positions force-flattened to zero at market, agent disabled until manual re-start; (e) if databento or Polygon feeds degrade, *the agent must halt*, not "use the last known feature value" (silent degradation is the dominant cause of bad RL behavior in production per Hambly/Xu/Yang §6); (f) no online learning, no exploration, no policy-update path exists in the deployed binary — the agent is inference-only and the inference is deterministic. The biggest non-market operational risk is **trust erosion**: if the agent makes a single inexplicable trade — even a profitable one — the operator's trust collapses and the iteration is effectively shelved regardless of metric performance, because we cannot afford to be running an agent we don't understand on real capital; the audit log is therefore not optional and the deterministic-inference property is non-negotiable.

---

## Addendum: Explicit responses to each of the 10 lessons

1. **Validate before deploy.** Honored as the four-stage gate (sim → paper → tiny-real → real) with kill criteria at each transition. Identical structure to iter 15 because the philosophy ports.
2. **Binary gates, not confidence rankers.** The RL agent emits discrete actions, not confidences — the policy's softmax logits are *not* used as confidence for sizing. Sizing is the *action itself*, in {−2, −1, 0, +1, +2}. The policy never produces a "confidence score" that we then size against; that's exactly the iter-12 C3-failure-mode we are avoiding.
3. **TP:SL asymmetry is structural.** The agent learns its own exits via the action space — there is no hardcoded TP/SL. But: the reward function embeds asymmetric drawdown penalty (λ_dd = 2.0 with `max(0, drawdown)` shape) which is the reward-side analog of structural asymmetric risk preference. The agent is *trained* to prefer trades where the upside exceeds the downside.
4. **Broker truth, not local cache.** Position state in §4.3 (`current_position` feature) is sourced from IBKR `reqPositions` on every tick, not a local accumulator. Reconciliation mismatch → policy halts, position force-flattened.
5. **No PDT-exposed lanes on sub-$25k US equity accounts.** N/A — ES futures are not PDT-regulated. Cash-settled, Section 1256 60/40 tax.
6. **One strategy per slot.** Explicitly addressed in §2 and §7's correlation rule: iter 2, iter 12, and iter 22 all trade ES; if more than one passes validation, only the highest Sharpe survives unless realized correlations are < 0.5.
7. **No same-data backtest.** Three disjoint windows in §6.1, hard wall, one pass through test window. Multi-seed protocol prevents seed-cherry-picking.
8. **Battle-tested broker SDKs only.** `ib_insync` for IBKR; databento client for historical data acquisition; no hand-rolled HTTP, no FIX engine, no custom matching simulator beyond the published Almgren-Chriss-style impact model.
9. **Capacity bound named upfront.** $1M soft, $5M hard, in §9.
10. **Kill criteria written before deploy.** §11, with exact numbers, no vibes. Especially: the 1.0-Sharpe uplift bar is fixed before any model is trained.
