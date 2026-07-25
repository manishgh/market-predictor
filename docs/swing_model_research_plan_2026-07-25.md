# Swing Model Research and Promotion Plan

Date: 2026-07-25

Scope: production-grade, long-only US equity swing prediction

Primary horizon: decision after session close, entry at the next session open, exit at the fifth session close

## 1. Executive decision

The system has strong dataset, promotion, lineage, serving, and monitoring controls, but it does not yet have a proven canonical swing model. `swing.model.v1` is implemented and promotion-gated; no real candidate has passed those gates. Older swing artifacts use different feature and target contracts and are not evidence for the canonical model.

The next work must improve the economic research design before adding model complexity. The primary prediction task should be cross-sectional selection: rank the current eligible universe by expected five-session, cost-adjusted, benchmark-relative return while controlling downside. Binary direction probability remains a useful secondary output, but it should not be the sole training objective or selection score.

The immediate recommendation is:

1. Build and audit a substantially longer point-in-time research panel.
2. Establish simple, reproducible classification and regression baselines.
3. Compare those baselines with a direct learning-to-rank model.
4. Add catalyst information only through controlled ablation.
5. Add global events through an exposure map, not as one undifferentiated sentiment value applied to every stock.
6. Promote only after cost-stressed walk-forward, unseen-ticker, regime, capacity, and prospective shadow evidence pass.

This plan does not assume that a profitable edge exists. It defines how to determine that without leakage, model shopping, or optimistic execution assumptions.

## 2. Current state and gaps

### Implemented

- Canonical five-session label with exact next-session-open entry and fifth-session-close exit.
- Exact stock, SPY, QQQ, and sector benchmark alignment.
- Point-in-time feature availability and event reconciliation.
- Purged walk-forward and unseen-ticker holdout evaluation.
- Probability calibration, selected-policy economics, regimes, drawdown, capacity, and memory gates.
- Immutable candidate evidence, causal shadow evidence, signed promotion, atomic serving bundles, and selected-policy live monitoring.
- Technical, benchmark-relative, catalyst, basic fundamental, cross-sectional rank, sector, market-cap, and liquidity feature families.

### Not yet demonstrated

- No real `swing.model.v1` candidate has passed promotion.
- No canonical real-data swing report establishes positive net excess return.
- Historical news does not yet constitute a proven observed-first-seen corpus across the intended universe.
- Real event coverage, entity relevance, deduplication, and correction timing have not been demonstrated at production scale.
- Spread, market-impact, participation, and capacity estimates have not been calibrated from real executable quote data.
- There is no sufficiently long prospective selected-policy outcome history.
- The point-in-time universe has not yet demonstrated inactive and delisted-name coverage.

### Contract limitations to test

- The current model optimizes a binary `future_net_return_5d > 0` label, while the application selects a small top-ranked portfolio.
- The current fundamental features are mostly raw values. Raw revenue, income, EPS, and cash flow are weakly comparable across firms without growth, surprise, yield, quality, and sector-relative transformations.
- Current news features are predominantly counts, mean sentiment, source counts, and relevance. They do not fully represent event type, novelty, surprise, source agreement, pre-decision price reaction, or issuer-specific exposure.
- Current regimes are coarse risk-on/risk-off flags.
- Four validation folds with a minimum of 120 training sessions are implementation minima, not adequate research evidence for a stable five-session model.
- Fixed ROC AUC thresholds should not override economic and ranking evidence.

## 3. Prediction product

### Decision

At each completed US session, score one coherent point-in-time cross section of eligible securities. The model must not score incomplete, stale, partial-feed, under-warmed, or untradeable rows.

### Proposed `swing.model.v2` outputs

- `expected_net_return_5d`
- `expected_excess_return_vs_spy_5d`
- `expected_excess_return_vs_sector_5d`
- `probability_net_positive_5d`
- `probability_outperform_spy_5d`
- `probability_outperform_sector_5d`
- `predicted_mae_5d` or a calibrated downside quantile
- `cross_sectional_rank` and `rank_percentile`
- `prediction_interval` or equivalent uncertainty estimate
- `catalyst_confirmation` and catalyst evidence, produced as a separately auditable component
- `abstain_reason` when evidence is insufficient

The initial selection score should rank expected sector-relative net return and reject candidates that fail downside, liquidity, freshness, or uncertainty constraints. SPY-relative and absolute returns remain mandatory evaluation outputs.

### Benchmark policy

- SPY: broad-market comparison.
- QQQ: growth/technology comparison where relevant.
- Sector ETF: primary stock-specific relative comparison.
- Cash/T-bill proxy: optional portfolio opportunity-cost comparison.

## 4. Data research program

### 4.1 Universe

Use a point-in-time US-listed universe rather than the current constituents of an index:

- Research breadth: at least 500 liquid names; target 1,000-1,500 when source rights and data quality permit.
- Preserve delisted, acquired, bankrupt, and temporarily inactive securities.
- Store membership start/end timestamps and the reason for exclusion.
- Keep market-cap, sector, listing, and liquidity classifications point-in-time.
- Evaluate large-, mid-, and small-cap tiers separately. Do not assume one model transfers equally across them.

### 4.2 History

Two years of daily observations is too short for robust five-session research. It provides roughly 500 sessions, far fewer effectively independent five-session outcomes, and limited regime diversity.

- Price/volume and market context: target at least 7 years; 10 years is preferred.
- Fundamentals and filings: use all point-in-time history available over the same interval.
- Historical news: target at least 3-5 years where publication and correction timestamps are trustworthy.
- Prospective news: retain both provider publication time and first-observed ingestion time from now onward.
- Prospective shadow: reserve a one-use 60-90 session minimum after a candidate is frozen; extend it when power is insufficient.

Backfilled news without observed-first-seen timing is research-grade evidence only. It must not be represented as equivalent to prospectively observed data.

### 4.3 Canonical sources

- Alpaca SIP adjusted daily bars and corporate actions for market data.
- Alpaca as the primary licensed ticker-news source.
- SEC EDGAR submissions and XBRL facts for filings and point-in-time fundamentals.
- Seeking Alpha for quant, analyst, earnings, and article-derived features only where endpoint rights, timestamp semantics, quota, and historical availability are verified.
- Finviz Elite for universe discovery, screener attributes, and supplemental headlines; do not treat an HTML scrape as the source of truth when a structured source exists.
- Reddit as a separately governed prospective chatter feature after official API approval. Treat it as attention/crowding, not verified issuer news.
- GDELT or another licensed global-event source for macro/geopolitical event discovery.

Every source record must carry source, source record ID, issuer/entity mapping, publication time, first-observed time, update time, content hash, license class, and availability time used by the feature builder.

### 4.4 Data audit

Before training, publish one report per ticker and one aggregate report containing:

- first and last daily bar; valid-bar count; adjustment and feed;
- first and last event; event count by source and event type;
- first-observed coverage rate and publication-time-only rate;
- duplicate, correction, entity-mapping, and low-relevance rates;
- SEC filing and fundamental availability;
- earnings date, estimate, and surprise coverage;
- missingness by feature family and calendar period;
- selected universe membership intervals;
- eligible feature rows and exact label rows;
- benchmark alignment failures;
- stale, future-available, and news/candle mismatch counts;
- model eligibility and explicit ineligibility reasons.

Training is blocked on any future-availability observation, label replay error, benchmark mismatch, or unresolved ticker/entity mapping.

## 5. Feature hypotheses

Every feature family is a named, predeclared hypothesis. A family is retained only when its out-of-sample incremental value survives cost and multiple-testing controls.

### H1: price, volume, and liquidity

Extend the current technical family with:

- 2-, 3-, and 40-session returns and momentum acceleration;
- overnight and regular-session return decomposition;
- distance from 52-week high and recent maximum return;
- turnover, turnover volatility, zero-volume/zero-return frequency, Amihud illiquidity, and quoted spread when available;
- market beta, sector beta, residual momentum, idiosyncratic volatility, and residual reversal;
- volume-price confirmation, abnormal volume, gap persistence, and post-gap reversal;
- stock and sector breadth, dispersion, and correlation regime;
- stable cross-sectional ranks and winsorized values computed using only that session's eligible universe.

### H2: fundamentals and revisions

Replace raw scale-dependent values with point-in-time transformations:

- revenue, EPS, margin, and free-cash-flow growth;
- earnings surprise and standardized unexpected earnings;
- estimate revision direction, breadth, magnitude, and recency;
- profitability, leverage, accruals, cash conversion, and earnings quality;
- valuation yields relative to sector and historical range;
- filing recency and material 8-K/10-Q/10-K event flags.

Quarterly values become available only at the filing or verified release timestamp, never at fiscal-period end.

### H3: issuer catalyst

Build an event representation rather than relying on average sentiment:

- event taxonomy: earnings, guidance, analyst action, financing/dilution, M&A, contract, regulatory, legal, product, management, capital return, SEC filing, and operational incident;
- issuer relevance and role: direct subject, subsidiary, customer, supplier, competitor, or sector-only mention;
- novelty versus the issuer's recent event history;
- factual surprise relative to consensus or prior guidance;
- source agreement and independent-source count;
- recency decay and market-session timing;
- pre-decision abnormal return and volume, separating unpriced news from already-reacted news;
- correction/retraction state;
- event sentiment calibrated separately by event type.

FinBERT is suitable as one frozen text encoder or sentiment component. It is not a return-prediction model and must compete against simpler event features in ablation.

### H4: global and market regime

Do not attach the same global sentiment score to every ticker. Build an exposure map:

`global event -> country/commodity/industry/supply-chain exposure -> affected sectors and issuers`

Candidate features include:

- rates, yield-curve changes, credit spreads, VIX level/change/term structure, dollar, oil, gas, gold, copper, and major index returns;
- market and sector breadth, dispersion, correlation, and volatility regimes;
- scheduled macro announcement proximity;
- geopolitical event type, severity, novelty, location, affected commodity, and source agreement;
- issuer revenue geography, supply-chain, customer, and commodity sensitivity where point-in-time data are available.

The exposure map must be deterministic and auditable. LLM-generated links may propose research candidates but cannot become production features without validated mappings.

### H5: attention and crowding

Optional, lower-priority features:

- Reddit mention acceleration, unique-author count, engagement, bot/spam score, and divergence from price;
- short interest, days to cover, utilization, and borrow cost;
- options implied volatility, skew, term structure, unusual volume, and open-interest changes;
- institutional/insider filing changes.

These features require reliable historical point-in-time vendors. Missing values must represent known source absence, not silently imputed zero.

## 6. Model experiment ladder

All experiments use the same immutable dataset splits, costs, prediction policy, and top-k portfolio construction. Only the declared hypothesis changes.

### E0: current binary baseline

- Logistic regression.
- Histogram gradient boosting.
- Target: net-positive five-session return.
- Purpose: reproduce `swing.model.v1` honestly and establish a minimum baseline.

### E1: return and downside baselines

- Robust linear/Huber or regularized regression for five-session net excess return.
- Quantile regression for downside/MAE.
- Gradient-boosted tree regression.
- Selection: predicted excess return subject to downside and liquidity gates.

### E2: direct cross-sectional ranking

- LambdaMART/XGBoost or LightGBM ranker grouped by decision session.
- Relevance label derived from sector-relative net-return rank, with top-of-list emphasis.
- Metrics: NDCG@5/@10, rank information coefficient, top-k net excess return, turnover, and drawdown.

This tests whether optimizing the actual top-k decision outperforms sorting a binary probability.

### E3: multi-output candidate

Train separate or shared estimators for:

- expected excess return;
- absolute net-positive probability;
- downside quantile or MAE.

Use a deterministic selection policy; do not average probabilities from different targets.

### E4: catalyst ablation

Compare:

1. technical + relative + regime;
2. catalyst only;
3. technical + direct catalyst features;
4. technical model with catalyst as a confirmation/ranking overlay;
5. technical model restricted to material catalyst sessions.

Catalyst enters the model directly only if it improves untouched economic evidence. Otherwise it remains an explanation and confirmation overlay, consistent with the current intraday decision.

### E5: regime and exposure ablation

- Add richer market/sector regime features.
- Add global events through the exposure map.
- Test interactions with sector, cap, and liquidity.
- Consider regime-specific models only if one global model shows repeatable, statistically supported failure modes and each expert has enough independent history.

### E6: alternative data

Test Reddit, options, short interest, and ownership one source family at a time. Do not delay promotion of a sound market-data baseline while waiting for optional data.

### Model complexity rule

Do not introduce FinGPT, end-to-end transformers, temporal fusion models, or online deep learning before shallow tabular models establish positive cost-adjusted out-of-sample economics. Published asset-pricing evidence indicates that nonlinear interactions matter, but shallow methods can outperform deep methods when financial signal-to-noise and sample size are limited.

## 7. Validation design

### Splits

- Minimum research train window: 504 sessions; prefer 756 or more.
- Walk-forward test blocks: 63-126 sessions.
- At least 6 folds across multiple market regimes when history permits.
- Purge all overlapping five-session outcomes and apply a declared embargo.
- Keep calibration strictly after training and before each test block.
- Maintain an unseen-ticker holdout in addition to chronological validation.
- Use one final untouched historical holdout once, followed by prospective shadow evaluation.

### Economic simulation

- Select from the full eligible cross section, never from prefiltered winners.
- Enter at the next session's executable open assumption.
- Apply spread, fees, slippage, market impact, and adverse cost stress.
- Cap position by ADV participation and portfolio/sector concentration.
- Model overlapping five-day holdings and capital competition.
- Compare equal-weight, rank-weighted, and risk-constrained policies only if predeclared.
- Report SPY, QQQ, and sector-relative performance using identical intervals.

### Required reports

- ROC AUC, PR AUC, Brier score, expected calibration error, slope, and intercept.
- NDCG@5/@10, rank IC, precision in the selected set, and selection stability.
- Net return, excess return, profit factor, turnover, drawdown, recovery, and return/drawdown ratio.
- Confidence intervals resampled by session blocks, not independent rows.
- Performance by year, regime, sector, cap, liquidity, catalyst type, and source coverage.
- Exposure, concentration, beta, capacity, and cost-stress curves.
- Feature missingness, drift, permutation importance, and stability by fold.

### Multiple-testing control

Maintain an append-only experiment ledger including failed trials. Compute the Deflated Sharpe Ratio and Probability of Backtest Overfitting across the actual family of attempted strategies. A candidate that passes only because many alternatives were tried is rejected.

## 8. Promotion standard

The exact thresholds must be predeclared from baseline evidence and power analysis. They must not be tuned after looking at the final holdout.

At minimum, promotion requires:

- no leakage, alignment, reconciliation, stale-source, or point-in-time universe failure;
- positive paired lower confidence bound for selected net excess return versus both the frozen baseline and SPY;
- positive economics at base costs and declared stressed costs;
- enough independent sessions, selected trades, tickers, and regime observations for powered conclusions;
- acceptable calibration and no material probability bias;
- acceptable drawdown, concentration, turnover, liquidity, and capacity;
- no result dominated by one ticker, sector, month, regime, or catalyst source;
- unseen-ticker evidence that does not collapse;
- positive or non-destructive catalyst ablation;
- one-use untouched historical holdout evidence;
- 60-90 or more sessions of prospective selected-policy shadow evidence;
- valid immutable evidence, attestation, and atomic serving bundle.

ROC AUC 0.60 is not itself proof of a tradable model, and a lower AUC is not automatically a failure if ranking and net economics are stable. The primary gate should be paired, cost-adjusted top-k economic evidence with uncertainty bounds.

## 9. Live-learning policy

Daily collection is not daily model retraining.

- Collect and validate bars, events, filings, context, predictions, and matured outcomes every session.
- Monitor selected-policy calibration, economics, drawdown, selection rate, feature drift, and freshness using the existing R7.6 path.
- Retrain challengers on a fixed monthly or quarterly schedule after labels mature.
- Re-run the full immutable research and promotion path.
- Serve a challenger only after promotion; never update model weights from prediction traffic.
- Suppress a route on severe drift, stale features, identity mismatch, or monitoring evidence failure.

## 10. Delivery sequence and checkpoints

### S0: evidence inventory

- Produce the real ticker/source/date/coverage audit.
- Identify survivorship, timestamp, entity mapping, and cost-data gaps.
- Decide the exact research universe and historical start date.

Exit: auditable inventory; no unresolved leakage-critical defect.

Status on 2026-07-25: implementation and initial audit complete.

- Audited corpus: existing `largecap_50b_2y_20260630` research artifacts.
- Tickers: 318.
- Sanitized events: 224,681.
- Median daily bars: 787; configured S1 threshold: 1,764.
- Median news history: 23.81 months; configured research threshold: 36 months.
- First-observed event coverage: 0%.
- Feed provenance: unknown for all 318 tickers.
- Point-in-time membership input: absent.
- Source-collection ledger: absent.
- Alignment: 312 pass, 5 fail, 1 unavailable because BRK-B has no matching feature file.
- Peak working set: 0.57 GiB under the 4 GiB limit.

Outcome: no existing ticker is eligible for the new canonical research panel. S1 must
collect or canonicalize longer SIP-proven daily history, construct point-in-time
membership, repair the six alignment/missing-feature cases, and preserve prospective
first-observed and source-collection evidence. Historical publication-time news remains
research-only.

### S1: canonical research panel v2

- Build point-in-time universe, market, benchmark, fundamental, and event panels.
- Add availability timestamps and exact label replay.
- Extend training history and validation windows.

Exit: immutable dataset with passing alignment, coverage, memory, and replay audits.

Status on 2026-07-25: S1 market history and technical market-panel input gates
complete; catalyst, fundamental, exact-label, and availability joins remain
pending.

- Official point-in-time membership: 659 intervals, 658 tickers, 631 security
  identities, 504 constituents before the 2022 Under Armour dual-class removal
  and 503 afterward.
- Membership-carrying corporate transitions are approved in a checked-in
  primary-source ledger. Provider mergers remain non-membership candidates
  unless a reviewed row explicitly overrides them.
- Frozen Alpaca SIP window: 2019-07-09 through 2026-07-08.
- Observed bar artifacts: 670 of 671 requested stock/benchmark symbols.
- Rows: 1,088,146.
- Benchmarks: SPY, QQQ, and all 11 sector ETFs each have all 1,759 sessions.
- Explicit source gap: RHT returned `observed_empty`.
- Membership/session coverage: 885,371 of 885,538 expected rows, or 99.9811%.
- Gap classification: 633 complete intervals, 24 terminal non-trading gaps,
  one `observed_empty` interval, one ticker-reuse exclusion, and zero initial or
  interior gaps.
- Explicit exclusions: historical RHT (four expected sessions and no provider
  history) and historical SunTrust STI (107 expected sessions; Alpaca now maps
  STI to a different security).
- Memory: 0.258 GiB peak working set under the 4 GiB hard budget.
- Outcome: the hash-replayed market-history gate passes with zero blocking
  intervals.
- Canonical market-panel input assembly: 885,371 PIT stock rows across 656
  tickers, 22,867 benchmark rows, 657 approved membership intervals, and 628
  security identities.
- Panel integrity: zero duplicate ticker/session rows; stock bars are restricted
  to approved membership windows; gaps are not filled or shifted; all source and
  audit hashes are replayed before publication.
- Panel assembly memory: 0.881 GiB peak working set.
- Research boundary: stock and benchmark bars are production-quality SIP/all
  adjusted artifacts, but historical membership availability is a
  `provider_publication_proxy`. The membership artifact and bundle are
  research-only and cannot support promotion.
- Remaining exit work: event, fundamental, observed-availability, exact-label,
  and source-path replay joins must pass before the immutable S1 panel exits.

### S2: reproducible baselines

- Run E0 and E1.
- Freeze splits, costs, selection policy, and experiment ledger.

Exit: honest baseline economics and failure attribution.

### S3: ranking and multi-output research

- Run E2 and E3.
- Select one candidate family using development folds only.

Exit: predeclared candidate beats the frozen baseline on paired development evidence.

### S4: catalyst research

- Complete issuer-event taxonomy, novelty, surprise, relevance, and pre-reaction features.
- Run E4.

Exit: evidence-based decision to use catalyst directly, as an overlay, or only as explanation.

### S5: regime and global exposure

- Add richer regime features and the audited event-exposure map.
- Run E5.

Exit: incremental value without concentration or regime collapse.

### S6: execution and capacity

- Calibrate spread, slippage, impact, participation, and portfolio constraints.
- Stress costs and data delays.

Exit: positive net evidence under realistic and stressed assumptions.

### S7: frozen candidate

- Freeze hypothesis, model, baseline, features, splits, selection policy, and final historical holdout.
- Run the one-use historical evaluation.

Exit: all development and untouched historical gates pass.

### S8: prospective shadow and promotion

- Publish candidate and baseline predictions without execution.
- Mature 60-90 or more sessions of outcomes.
- Run paired causal shadow evaluation and promotion.

Exit: a signed, atomic, promotion-authorized serving bundle, or a documented rejection.

## 11. Highest-value next action

S0 and the S1 technical market-input gate are complete. Continue S1 with the
event, fundamental, availability, and exact-label evidence gaps above, not a
new model download.

The first deliverable should answer:

- How many point-in-time eligible tickers exist on each session?
- How many years of valid SIP-adjusted stock and benchmark paths exist?
- How much issuer news is truly relevant, deduplicated, and timestamp-safe?
- How much SEC, earnings, estimate-revision, and catalyst coverage exists?
- What spread/capacity evidence is available?
- How many independent five-session test periods and market regimes can be evaluated?

After the remaining S1 joins pass, run logistic, gradient-boosted
classification, gradient-boosted regression, and direct ranking on identical
folds. This is the shortest path to learning whether the project has a real
swing-prediction edge.

## 12. Research basis

- Gu, Kelly, and Xiu, *Empirical Asset Pricing via Machine Learning*: nonlinear interactions can improve return forecasts; momentum, liquidity, and volatility are dominant predictors, while shallow learning can outperform deep learning in low-signal financial panels.

  https://www.nber.org/papers/w25398
- Kelly, Manela, and Xiu, *Predicting Returns With Text Data*: return-predictive text representations should be trained for the return task rather than assumed from generic sentiment.

  https://www.nber.org/papers/w26186
- Araci, *FinBERT: Financial Sentiment Analysis with Pre-trained Language Models*: finance-specific language models improve financial sentiment tasks; this supports FinBERT as a text component, not as proof of return prediction.

  https://arxiv.org/abs/1908.10063
- Loughran and McDonald, *When Is a Liability Not a Liability?*: generic dictionaries misinterpret specialized financial language, supporting domain- and event-aware text features.

  https://doi.org/10.1111/j.1540-6261.2010.01625.x
- Poh, Lim, Zohren, and Roberts, *Building Cross-Sectional Systematic Strategies By Learning to Rank*: direct ranking is a relevant candidate when the operational decision is top-k cross-sectional selection.

  https://arxiv.org/abs/2012.07149
- Bailey et al., *The Probability of Backtest Overfitting*: ordinary holdout evidence can be unreliable after extensive strategy search; PBO estimates this selection risk.

  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Bailey and Lopez de Prado, *The Deflated Sharpe Ratio*: reported performance should account for multiple testing, selection bias, and non-normal returns.

  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551
- SEC EDGAR APIs: official submissions and XBRL company facts are available without API keys and are updated as filings are disseminated.

  https://www.sec.gov/search-filings/edgar-application-programming-interfaces
