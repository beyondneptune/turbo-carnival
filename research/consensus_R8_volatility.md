# Consensus R8 — Volatility, VIX Term Structure & Variance Risk Premium
Date: 2026-05-13
Total queries planned: 4
Total queries executed: 3 (Query 4 blocked — monthly search quota exhausted; resets June 1st)
Total papers surfaced: 30 (10 per executed query)
Source: Consensus MCP (Semantic Scholar / PubMed / Scopus / ArXiv)
Operator focus: SVXY rotation strategy (short-volatility ETF that benefits from VIX-futures contango and is exposed to volatility-spike tail events).

Note: DOIs are not directly returned by Consensus. Where I could not retrieve a DOI from the result payload, the Consensus URL is provided instead — operator can use it to land on the paper's detail page and grab the DOI for PDF retrieval. Authors / years / journal / citations are exactly as returned.

Relevance scale (1–5) is specifically calibrated for an SVXY rotation strategy:
- 5 = direct VIX-futures / contango / short-vol / SVXY-style results
- 4 = variance risk premium or tail-risk pricing with clear SVXY implications
- 3 = methodological infrastructure (IV surface fitting, arbitrage constraints) that affects how we measure regimes
- 2 = adjacent volatility literature
- 1 = peripheral

---

## Topic 1: VIX term structure, contango, short volatility strategies

### Paper 1.1 — Does It Take Two to ConTango?
- Authors: Thorsten Lehnert
- Year: 2025
- Venue: Unknown Journal (working paper)
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/12f52f4856f353679804e2d4d45003c2/
- Citations: 0
- Abstract summary: The time-varying SHAPE of the VIX term structure (slope + curvature, not just slope) embeds price information for variance risk and predicts S&P 500 straddle excess returns. A managed short-straddle strategy that adjusts exposure based on shape signals earned ~20% annualized variance-risk-adjusted alpha, strongest at the short end of the term structure.
- Relevance (SVXY): 5/5 — exactly the regime our SVXY rotation needs: shape-of-VIX-curve signal that conditions short-vol exposure dynamically.

### Paper 1.2 — Statistics of VIX Futures and Applications to Trading Volatility Exchange-Traded Products
- Authors: M. Avellaneda et al.
- Year: 2019
- Venue: International Journal of Theoretical and Applied Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/3f4f467d83595eb8bd97502d0f611574/
- Citations: 7
- Abstract summary: Calibrates a 2-factor lognormal mean-reverting model to VIX and constant-maturity-futures (CMF) curve, 2011–2016. Confirms VXX/XIV/VXZ short-vol roll strategies are theoretically profitable under stationarity, but realized Sharpe ratios are modest (~0.5 or less) with high 1-year variability due to sporadic VIX/CMF backwardation surges.
- Relevance (SVXY): 5/5 — the foundational tradeable-product paper. SVXY is the post-Feb-2018 successor to XIV and is directly analyzed here.

### Paper 1.3 — Selling VIX Futures and Options for Portfolio Return Enhancement
- Authors: E. Szado
- Year: 2019
- Venue: Mutual Funds (CFA / SSRN family)
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/ea43bef9131b51eebd80bd7b2c72973a/
- Citations: 2
- Abstract summary: Persistent VIX contango supports VIX-futures-selling and VIX-call-writing strategies, but high vol-of-vol and jump risk expose uncovered short positions to extreme tail losses. Compares 2008 vs 2018 events; finds the dynamically de-levered VPD/VPN indexes provide cleaner risk-adjusted profile than naked VIX selling.
- Relevance (SVXY): 5/5 — directly discusses the 2018 volmageddon failure mode for short-VIX products and proposes the de-levering/call-overlay structure that survived it.

### Paper 1.4 — Volatility Uncertainty and VIX Futures Contango
- Authors: G. Bakshi et al.
- Year: 2021
- Venue: SSRN Electronic Journal
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/0d0d6a01338a5c4cb0aedf99b7f22af8/
- Citations: 2
- Abstract summary: Theoretical model: contango is most common because short-dated volatility uncertainty is stochastically dominated by long-dated. Backwardation arises under unfavorable market conditions. Equity price process has continuous shocks plus jump discontinuities (disaster uncertainty with time-varying probability).
- Relevance (SVXY): 4/5 — gives a structural explanation for why contango is the modal state, hence why short-VIX strategies have positive drift, plus when that drift reverses.

### Paper 1.5 — Volatility timing, sentiment, and the short-term profitability of VIX-based cross-sectional trading strategies
- Authors: Wenjie Ding et al.
- Year: 2021
- Venue: Journal of Empirical Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/8e6dd51b204b5c46bbd551e90bdf4731/
- Citations: 23
- Abstract summary: Cross-sectional stock strategy that holds sentiment-prone stocks when VIX is low and sentiment-immune stocks when VIX is high earns significant excess returns vs benchmark long–short portfolios that ignore VIX. Profits not subsumed by known risk factors or transaction costs.
- Relevance (SVXY): 3/5 — uses VIX as a regime conditioner rather than trading VIX directly, but the regime-conditioning logic transfers to SVXY rotation.

### Paper 1.6 — The VIX Futures Basis: Evidence and Trading Strategies
- Authors: David P. Simon
- Year: 2012
- Venue: Unknown Journal (working paper / Journal of Alternative Investments)
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/8d64ce1a370459d587ccbf07a1519a72/
- Citations: 28
- Abstract summary: 2006–2011: VIX futures basis lacks forecast power for spot VIX but forecasts VIX futures price changes. Shorting VIX futures in contango and buying in backwardation, hedged with ES futures, is highly profitable and robust to transaction costs and out-of-sample hedge-ratio forecasts. Confirms basis reflects a harvestable risk premium, not the mean-reverting properties of VIX.
- Relevance (SVXY): 5/5 — the canonical empirical paper for the SVXY rotation thesis: long short-vol in contango, flip in backwardation, ES-hedged.

### Paper 1.7 — A Note of Trading the Term Structure of VIX Futures
- Authors: Anusar Farooqui
- Year: 2020
- Venue: ERN: Other Econometric Modeling: Derivatives (SSRN topic)
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/6b577547bd0f5866ab11ef49e50c6f4c/
- Citations: 0
- Abstract summary: Term structure of VIX futures is a high-frequency signal of dealer risk appetite. Two systematic strategies (hyper-parameters tuned OOS) harvest the resulting risk premium; risk-adjusted returns are at least 2x the three benchmarks (long SPY, naked short short-term vol, calendar spread).
- Relevance (SVXY): 4/5 — proposes term-structure-based signals; useful as a feature recipe for SVXY rotation.

### Paper 1.8 — VIX constant maturity futures trading strategy: A walk-forward machine learning study
- Authors: Sangyuan Wang et al.
- Year: 2024
- Venue: PLOS ONE
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/d5331bfb19df59cc82b651fe605da4e7/
- Citations: 2
- Abstract summary: Seven ML approaches predict next-day VIX CMF returns from term-structure features. 11-year walk-forward backtest. Their Constrained-Mean-Variance Portfolio Optimization (C-MVO) strategy achieves information ratio 0.623 vs long-short benchmark 0.404. Confirms term-structure features have predictive power.
- Relevance (SVXY): 4/5 — modern ML approach with rigorous walk-forward methodology; useful as the methodology template for our rotation logic.

### Paper 1.9 — VIX Index Strategies: Shorting Volatility As a Portfolio Enhancing Strategy
- Authors: Alberto Dondoni et al.
- Year: 2018
- Venue: Econometric Modeling: Capital Markets - Portfolio Theory eJournal
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/f5830d34b14a568cbf87738ace489094/
- Citations: 1
- Abstract summary: Empirical analysis of VIX index and VIX futures curve dynamics, then evaluates long/short VIX-futures portfolio strategies. Confirms persistent contango as a return source, qualified by tail-event drawdowns. Published the year of the volmageddon event, useful as a contemporaneous viewpoint.
- Relevance (SVXY): 4/5 — same-vintage study as the 2018 short-vol blowup; useful for contemporaneous calibration.

### Paper 1.10 — Trading Signals in VIX Futures
- Authors: M. Avellaneda et al.
- Year: 2021
- Venue: Applied Mathematical Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/fa2f7f589a175c60932d37f17bda12ff/
- Citations: 5
- Abstract summary: Markov model for the VIX futures term structure, with a deep neural network (5 hidden layers) modeling expected utility as a function of curve shape + position. OOS backtest gives sensible portfolio performance and dynamically chooses long-or-short positions based on regime.
- Relevance (SVXY): 5/5 — same authors as the 2019 foundational paper; provides the conditional position-sizing logic an SVXY rotation needs.

---

## Topic 2: Variance risk premium and equity return predictability

### Paper 2.1 — Variance Risk Premium and Equity Returns
- Authors: Athanasios P. Fassas et al.
- Year: 2018
- Venue: Research in International Business and Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/733147876e1a5576bc0c6c9983bf4ac3/
- Citations: 22
- Abstract summary: VRP is on average negative across a wide range of stock indices, individual stocks, and ETFs — confirming variance risk is priced. VRP is a reliable predictor of equity returns, explaining up to 20% of monthly equity-return variation.
- Relevance (SVXY): 4/5 — establishes that VRP applies to ETFs broadly (relevant for SVXY's underlying logic) and quantifies predictive power.

### Paper 2.2 — Stock Return Predictability and Variance Risk Premia: Statistical Inference and International Evidence
- Authors: Tim Bollerslev et al.
- Year: 2011
- Venue: Journal of Financial and Quantitative Analysis
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/9f2114c61fb25334841c1db9e505544d/
- Citations: 312
- Abstract summary: Confirms VRP predicts aggregate stock returns is NOT a finite-sample-bias artifact. Replicates the result in France, Germany, Japan, Switzerland, Netherlands, Belgium, UK; defining a "global" VRP via panel regression yields even stronger predictability.
- Relevance (SVXY): 4/5 — provides the inferential foundation we need to defend VRP-based regime filters in our strategy.

### Paper 2.3 — Expected Stock Returns and Variance Risk Premia
- Authors: Tim Bollerslev et al.
- Year: 2007
- Venue: University of Cincinnati Lindner College of Business Research Paper Series
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/998c8556931956c6bfde887106a3529f/
- Citations: 1658
- Abstract summary: Seminal paper: VRP explains >15% of quarterly excess market-return variation (1990–2005), with high (low) VRP predicting high (low) future returns. VRP dominates P/E, dividend yield, default spread, and CAY. Combined with P/E, R^2 rises to 25%. Requires model-free IV and high-frequency realized variance.
- Relevance (SVXY): 5/5 — the foundational paper for VRP as a return predictor; combining VRP with curve shape is the canonical SVXY-rotation theoretical basis.

### Paper 2.4 — The Variance Risk Premium: Components, Term Structures, and Stock Return Predictability
- Authors: Junye Li et al.
- Year: 2016
- Venue: Journal of Business & Economic Statistics
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/bfb6a3ef920b57998fe4a65021f0f68b/
- Citations: 57
- Abstract summary: Decomposes total VRP into diffusive and jump components using a self-exciting jump-clustering pricing model. Total VRP is negative with downward-sloping term structure; jump VRP component has UPWARD-sloping term structure and short-term jump VRP responds abruptly to extreme events — acting as a "fear of crash" proxy.
- Relevance (SVXY): 5/5 — the jump-VRP decomposition is exactly the crash-warning signal that should kill an SVXY position.

### Paper 2.5 — Tail Risk Premia and Return Predictability
- Authors: Tim Bollerslev et al.
- Year: 2014
- Venue: Journal of Financial Economics
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/5342ae0976fe5647aa93e50b3074a70d/
- Citations: 355
- Abstract summary: Decomposes VRP into a "normal" and a "jump tail-risk" component using a model-free estimator. Most return predictability attaches to the jump-tail component — investor "fears" of market crashes drive the predictability.
- Relevance (SVXY): 5/5 — tail-jump VRP is the precise leading indicator an SVXY rotation needs to anticipate volmageddon-style risk.

### Paper 2.6 — Expected Stock Returns and Variance Risk Premia (2008 version)
- Authors: Tim Bollerslev et al.
- Year: 2008
- Venue: Unknown Journal
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/ec54a04e5a2f5bdb8ead82d72efcb84b/
- Citations: 87
- Abstract summary: Companion/predecessor of the 2007 paper. Same core result on VRP predicting post-1990 returns, with theoretical motivation from general-equilibrium model incorporating time-varying economic uncertainty. Predictability strongest at quarterly horizon.
- Relevance (SVXY): 4/5 — same core finding as 2.3 but with the GE motivation more developed.

### Paper 2.7 — Does Variance Risk Premium Predict Expected Returns?
- Authors: Xian-Ji Kuang et al.
- Year: 2023
- Venue: Applied Economics Letters
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/fedcb23685ec5d7fb7be9839e633c094/
- Citations: 0
- Abstract summary: VRP is a significantly priced factor in BULL markets but its predictive ability diminishes faster in bear markets when horizons lengthen. Provides state-dependent calibration for VRP-based predictors.
- Relevance (SVXY): 4/5 — directly relevant: SVXY mostly works in bull/range-bound regimes, fails in bear regimes; this paper formalizes that state-dependence.

### Paper 2.8 — Variance Risk in Aggregate Stock Returns and Time-Varying Return Predictability
- Authors: Sungjune Pyun
- Year: 2016
- Venue: Econometric Modeling: Capital Markets - Risk eJournal
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/c723776663705543b931e88fa42ac490/
- Citations: 52
- Abstract summary: New OOS forecasting methodology gives monthly OOS R^2 of 8% and ~0.1 annual Sharpe gain. Key idea: VRP predicts more accurately when the variance-innovation–market-returns relationship is steeper or variance shocks explain more of market-return variance. Extends to FX cross-section.
- Relevance (SVXY): 4/5 — practical OOS methodology for VRP-based predictors; superior to Bollerslev-Tauchen-Zhou OOS results.

### Paper 2.9 — Idiosyncratic Risk Matters
- Authors: Amit Goyal et al.
- Year: 2002
- Venue: AFA 2003 Washington Meetings
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/be459afc286e51a3abd1ac03ee539de1/
- Citations: 1175
- Abstract summary: Average stock (mostly idiosyncratic) variance positively predicts market returns; market variance has no forecasting power. Consistent with time-varying risk premia from background risk and investor heterogeneity, or capital-structure option value of equity.
- Relevance (SVXY): 2/5 — variance-decomposition paper but not specifically about VRP or short-VIX strategies.

### Paper 2.10 — Moment Risk Premia and Stock Return Predictability
- Authors: Zhenzhen Fan et al.
- Year: 2018
- Venue: Journal of Financial and Quantitative Analysis
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/f2ff392cf6a35644ad97c178ba02a0d8/
- Citations: 8
- Abstract summary: Extends VRP to higher-moment risk premia. Second-moment RP predicts at short horizons positively; third-moment (skew) RP predicts at medium horizons negatively; fourth-moment (kurt) RP predicts positively at medium horizons. Combining moments improves OOS predictability.
- Relevance (SVXY): 4/5 — skew/kurtosis risk premia are exactly the tail-distortion measures that flag pre-volmageddon regimes.

---

## Topic 3: Implied volatility surface and arbitrage

### Paper 3.1 — Simulation of Arbitrage-Free Implied Volatility Surfaces
- Authors: R. Cont et al.
- Year: 2023
- Venue: Applied Mathematical Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/d58843efa4705050b8272319aa90c5c7/
- Citations: 22
- Abstract summary: Computationally tractable method for simulating arbitrage-free IV surfaces, combined with a data-driven model trained on historical SPX IV data. Generates dynamic scenarios that respect both static no-arbitrage constraints and realistic IV co-movement statistics.
- Relevance (SVXY): 3/5 — methodological infrastructure for IV-surface-respecting scenario generation; supports backtesting/MC for the rotation rule.

### Paper 3.2 — Arbitrage-Free Implied Volatility Surface Generation with Variational Autoencoders
- Authors: Brian Ning et al.
- Year: 2021
- Venue: ArXiv (preprint)
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/41c931c2634f577db0ca52d9505c15b7/
- Citations: 19
- Abstract summary: Hybrid VAE + SDE model (regime-switching + Levy additive) to generate arbitrage-free IV surfaces consistent with history. Conditional features improve OOS generative performance.
- Relevance (SVXY): 3/5 — VAE-based surface generator; useful if we want regime-switching surface scenarios for stress testing.

### Paper 3.3 — Arbitrage-Free Smoothing of the Implied Volatility Surface
- Authors: Matthias R. Fengler
- Year: 2009
- Venue: Quantitative Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/3140609193d056c18c0001b164b73536/
- Citations: 170
- Abstract summary: Smoothing-spline approach for arbitrage-free IV-surface fitting under shape constraints. Computationally cheap, simple to implement; required to avoid negative transition probabilities and false Greeks when local-vol models are built on top.
- Relevance (SVXY): 3/5 — the canonical surface-fitter; relevant if we ever derive ATM-IV or skew measures from real chains for regime classification.

### Paper 3.4 — A Two-Step Framework for Arbitrage-Free Prediction of the Implied Volatility Surface
- Authors: Wen-Ying Zhang et al.
- Year: 2021
- Venue: Quantitative Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/64a948b57f5552a3a8297a12a3651ba4/
- Citations: 10
- Abstract summary: PCA / VAE / surface-sampling features predicted via LSTM, fed to a DNN that constructs an arbitrage-free IV surface. Trained on SPX option history; outperforms popular regression baselines and standard interpolation.
- Relevance (SVXY): 3/5 — IV-surface predictor; relevant if we expand to forecasting forward variance for the rotation logic.

### Paper 3.5 — Arbitrage-Free SVI Volatility Surfaces
- Authors: Jim Gatheral et al.
- Year: 2012
- Venue: Quantitative Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/a75d609dd41e5a5b872265c7bf1e13dc/
- Citations: 236
- Abstract summary: Calibrates the widely-used SVI parameterization for IV smile with closed-form arbitrage-free parameter restrictions. Demonstrates high-quality SVI fits on recent SPX data.
- Relevance (SVXY): 3/5 — the industry-standard SVI surface model; useful as a reference if we ever build our own skew-feature pipeline.

### Paper 3.6 — Building Arbitrage-Free Implied Volatility Surface by Entropy Minimization
- Authors: Claude Muller
- Year: 2024
- Venue: SSRN Electronic Journal
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/6ca7a1caf5be53029a9bd76b07ee4c0b/
- Citations: 0
- Abstract summary: Builds AFIVS slices by relative-entropy minimization with bounded European call prices and a chosen prior. Numerically robust, suited for high-frequency use; supports vanilla market-making and LV/LSV exotic-pricing calibration.
- Relevance (SVXY): 2/5 — primarily an exotics-pricing infrastructure paper; only tangential to SVXY rotation.

### Paper 3.7 — VolGAN: A Generative Model for Arbitrage-Free Implied Volatility Surfaces
- Authors: Milena Vuletic et al.
- Year: 2024
- Venue: Applied Mathematical Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/d8f74f6feef55cdcbc613fdea0277087/
- Citations: 8
- Abstract summary: GAN trained on time series of IV surfaces + underlying. Learns covariance of IV co-movements and generates realistic VIX-index dynamics. Used to construct data-driven hedging strategies that beat Black-Scholes delta/delta-vega hedging on option portfolios.
- Relevance (SVXY): 3/5 — generative VIX-dynamics model; could be used as a scenario engine for stressing SVXY drawdowns.

### Paper 3.8 — Arbitrage-Free Conditions for Implied Volatility Surface by Delta
- Authors: Ximei Wang et al.
- Year: 2019
- Venue: The North American Journal of Economics and Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/ad8d0e98ef1e5b5c8a71e70c3683eecf/
- Citations: 2
- Abstract summary: Develops sufficient (and under mild assumptions, necessary) static-arbitrage-free conditions for delta-quoted IV surfaces (common in FX). Tools for testing arbitrage on simulated and live data; demonstrated on EURUSD and USDJPY.
- Relevance (SVXY): 1/5 — FX-options-quoted-in-delta is largely outside SVXY's domain.

### Paper 3.9 — No-Arbitrage Global Parametrization for the eSSVI Volatility Surface
- Authors: A. Mingone
- Year: 2022
- Venue: Quantitative Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/6e7ccc4988d1529585dcec3da58b8d85/
- Citations: 9
- Abstract summary: Global (all-maturities simultaneous) calibration of the eSSVI surface with arbitrage guarantees. Improves on prior sequential-by-expiry calibration.
- Relevance (SVXY): 2/5 — surface calibration infrastructure; minor relevance for SVXY rotation.

### Paper 3.10 — Arbitrage-Free Implied Volatility Surface Smoothing via Generative Adversarial Networks
- Authors: Victor Ginart Belmonte et al.
- Year: 2025
- Venue: Proceedings of the 6th ACM International Conference on AI in Finance
- DOI: not retrievable from Consensus result
- Consensus URL: https://consensus.app/papers/details/be06c0f97de250c092f60f8e35c05fd0/
- Citations: 0
- Abstract summary: GAN-based IV-surface smoother that enforces arbitrage constraints. Transforms noisy, arbitrage-violating surfaces to smooth arbitrage-free ones with the lowest reported MSE and runtime vs traditional methods.
- Relevance (SVXY): 2/5 — practical surface-denoising; relevant only if we run on raw quote chains.

---

## Topic 4: Volatility tail risk, Black Monday, Volmageddon 2018

QUERY NOT EXECUTED — Consensus search quota for May 2026 exhausted before this query ran. Limits reset 2026-06-01. Re-run this query first thing on June 1st using:
- Query: "volatility tail risk Black Monday Volmageddon 2018"
- Expected hits: papers on Feb 5, 2018 short-VIX collapse (XIV termination, SVXY 80%+ intraday drawdown), structural ETP-rebalance literature, jump-risk pricing, and the 1987 Black Monday vol-spike literature.

Operator action: Bookmark this section. When the quota resets, append 10 entries here in the same format (Paper 4.1 through 4.10).

---

## Cross-cutting takeaways for an SVXY rotation strategy

1. The contango harvest is real and structurally explained — Bakshi et al. (1.4) provides the volatility-uncertainty-ordering theory; Simon (1.6), Avellaneda et al. (1.2 and 1.10), and Szado (1.3) provide the empirical case across multiple market regimes.
2. Naked short-vol blows up — Szado (1.3) explicitly contrasts the de-levered VPD/VPN structures (which survived 2018) with naked short-VIX (which did not). Any SVXY rotation must include a regime-off rule.
3. The best regime-off signal is jump-tail VRP, not realized vol — Bollerslev et al. (2.5) and Li et al. (2.4) show that the jump component of VRP, with its upward-sloping term structure and abrupt response to fear, is the early-warning indicator. Moment risk premia (2.10) and bull-vs-bear state dependence (2.7) reinforce this.
4. The curve SHAPE matters, not just the slope — Lehnert (1.1, 2025) shows that combining slope and curvature outperforms slope-only signals for short-vol strategies, with the largest improvement at the short end of the curve (where SVXY trades).
5. IV-surface infrastructure is mostly secondary for SVXY rotation — Topic 3 papers are useful only if we extract our own ATM-IV / skew features. If we rely on VIX-futures prices + the public VIX index, Topic 3 can be deferred.

## Top 10 most-relevant papers for immediate download (SVXY rotation focus)

Ranked by combined judgment on (a) SVXY/short-vol-product specificity, (b) regime-conditioning utility, (c) tail-risk early-warning value, (d) empirical robustness:

1. Simon, "The VIX Futures Basis: Evidence and Trading Strategies" — Topic 1.6 — Consensus URL https://consensus.app/papers/details/8d64ce1a370459d587ccbf07a1519a72/
2. Avellaneda et al., "Statistics of VIX Futures and Applications to Trading Volatility Exchange-Traded Products" — Topic 1.2 — https://consensus.app/papers/details/3f4f467d83595eb8bd97502d0f611574/
3. Szado, "Selling VIX Futures and Options for Portfolio Return Enhancement" — Topic 1.3 — https://consensus.app/papers/details/ea43bef9131b51eebd80bd7b2c72973a/
4. Lehnert, "Does It Take Two to ConTango?" — Topic 1.1 — https://consensus.app/papers/details/12f52f4856f353679804e2d4d45003c2/
5. Avellaneda et al., "Trading Signals in VIX Futures" — Topic 1.10 — https://consensus.app/papers/details/fa2f7f589a175c60932d37f17bda12ff/
6. Bollerslev et al., "Tail Risk Premia and Return Predictability" — Topic 2.5 — https://consensus.app/papers/details/5342ae0976fe5647aa93e50b3074a70d/
7. Li et al., "The Variance Risk Premium: Components, Term Structures, and Stock Return Predictability" — Topic 2.4 — https://consensus.app/papers/details/bfb6a3ef920b57998fe4a65021f0f68b/
8. Bollerslev et al., "Expected Stock Returns and Variance Risk Premia" (2007 seminal) — Topic 2.3 — https://consensus.app/papers/details/998c8556931956c6bfde887106a3529f/
9. Bakshi et al., "Volatility Uncertainty and VIX Futures Contango" — Topic 1.4 — https://consensus.app/papers/details/0d0d6a01338a5c4cb0aedf99b7f22af8/
10. Wang et al., "VIX Constant Maturity Futures Trading Strategy: A Walk-Forward Machine Learning Study" — Topic 1.8 — https://consensus.app/papers/details/d5331bfb19df59cc82b651fe605da4e7/

## Coverage notes

- Topic 1 (VIX term structure / contango): excellent coverage. Direct SVXY-style strategies are well-represented and span 2012–2025.
- Topic 2 (variance risk premium): excellent coverage. Includes Bollerslev seminal 2007 paper (1658 citations) and its tail-risk extension (355 citations), plus state-dependence and higher-moment refinements.
- Topic 3 (IV surface arbitrage): strong coverage but more methodologically peripheral to an SVXY-only rotation. Most relevance comes via skew / surface-derived regime features.
- Topic 4 (tail risk / Volmageddon 2018): NOT EXECUTED. Re-run when the Consensus quota resets on June 1st.

## DOI list

DOIs were not retrievable from the Consensus result payload for any of the 30 papers in this run. Use the Consensus URLs to land on each paper's detail page and copy the DOI for downstream PDF retrieval. The closest match to a DOI candidate appears via the journal/venue, but treat all "10.xxxx" strings as pending verification.
