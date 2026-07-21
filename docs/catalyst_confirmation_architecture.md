# Market Prediction Intelligence Architecture

This document defines the operational design for `market-predictor` as a prediction intelligence system.

The system produces swing, daily momentum, and intraday setup predictions from catalysts, news, filings, sentiment, market context, and price behavior. It does not own final trade execution, portfolio state, order routing, position sizing, stops, or exits.

Scope:

- US-listed equities and ETFs
- Swing and daily momentum prediction, with large-cap/ETF context
- Five-minute intraday setup ranking, entry-path probability, and exit-risk probability
- Long-side prediction signals and watchlist ranking
- Historical and recent catalyst analysis
- Model training, validation, promotion, and audit reporting

Out of scope:

- User-visible alerts, alert persistence/deduplication, acknowledgement, and notifications
- Broker order execution
- Live portfolio/risk state
- Final trade sizing
- Stop/target management
- Backtesting engine ownership
- Production websocket ownership when `trading_flow` is the live market-data runtime

## 1. System Responsibility

`market-predictor` answers:

- Which tickers have relevant catalysts?
- How did similar catalyst plus price setups behave historically?
- Is current data complete enough to trust a prediction?
- What is the next-day or next-swing probability?
- Is a completed intraday setup likely to reach its target before its stop?
- What is the separate stop-first risk for that setup?
- Which names deserve watchlist attention?
- What data, model, and source evidence produced the score?

The output is prediction intelligence, not an order instruction.

## 2. Boundary With trading_flow

`trading_flow` is the execution, backtesting, and portfolio runtime.

`market-predictor` owns:

- Alpaca/Seeking Alpha/SEC/Reddit/Finviz data collection for ML and prediction
- Historical candles and event datasets for model features
- Feature engineering
- FinBERT sentiment scoring
- Model training and validation
- Prediction reports
- Model artifact publishing
- Data-readiness and audit reports

`trading_flow` owns:

- Wishlist observation, alerts, deduplication, acknowledgement, and web/mobile notifications
- Strategy orchestration
- Backtesting
- Position state
- Portfolio risk
- Entry and exit rules
- Stop/target logic
- Broker order lifecycle
- Trade logs and PnL
- Production live market-data stream if that is already centralized there

Integration contract:

- `market-predictor` publishes prediction reports or a prediction API response.
- `trading_flow` consumes those predictions and decides whether, when, and how to trade.
- Predictions are evidence, not TradingFlow `ExternalSignal` messages; they contain no order action, side, stop, target, or size.
- The TradingFlow order path reads cached prediction evidence and never blocks on remote model inference.
- The projects do not share databases or OHLCV storage.
- Both systems must not independently open production market-data streams for the same account unless provider limits and symbol ownership are explicitly managed.

The phased contract, failure policy, alert migration, backtest handoff, and live completed-bar design are defined in [TradingFlow integration plan](trading_flow_integration_plan.md).

The removed predictor alert rules and TradingFlow parity decisions are recorded in [Legacy Alert Rule Parity](legacy_alert_rule_parity.md).

## 3. Prediction Output Contract

Every prediction output should include enough metadata to audit the score later.

Required fields:

| Field | Meaning |
| --- | --- |
| `ticker` | Symbol being scored |
| `as_of` | Timestamp or date the prediction was generated |
| `horizon` | Prediction horizon, for example `1d`, `5d`, or `10d` |
| `signal` | Human-readable direction bucket such as `bullish`, `lean_bullish`, `neutral`, `lean_bearish`, `bearish`, or `no_signal` |
| `probability_up` | Model probability for upward movement over the horizon |
| `confidence` | Confidence bucket derived from probability spread, model agreement, data quality, and catalyst strength |
| `data_readiness_status` | `valid`, `warn`, or `invalid` |
| `data_readiness_reasons` | List of failed or warning gates |
| `model_name` | Model artifact or registry name |
| `model_version` | Version, date, commit, or artifact identifier |
| `feature_schema_version` | Feature schema used for scoring |
| `daily_bar_count` | Number of daily bars available for slow technical state |
| `intraday_bar_count` | Number of intraday bars available if intraday features are used |
| `latest_price_date` | Latest candle date used |
| `recent_event_count` | Number of recent catalyst/news/filing/social events used |
| `source_families` | Sources present, such as `alpaca`, `seeking_alpha`, `sec`, `reddit` |
| `top_catalysts` | Short list of the most important recent catalysts |
| `sector` | Configured or inferred sector bucket |
| `sector_benchmark` | Sector ETF or benchmark used |
| `market_benchmark` | Broad benchmark, normally `SPY` |
| `audit_id` | Identifier linking prediction to data/model audit artifacts |

Optional fields:

- `probability_down`
- `expected_move_bucket`
- `market_cap`
- `market_cap_bucket`
- `average_recent_sentiment`
- `volume_z_score`
- `relative_strength_vs_spy`
- `relative_strength_vs_sector`
- `news_candle_alignment_status`

## 4. Universe Management

The system maintains two universe layers for prediction readiness.

### Superset

The superset is the broad pool of tickers that could plausibly become prediction candidates.

Target:

- Approximately 1,500-2,000 US-listed symbols
- Mid/small-cap focus
- Basic liquidity, listing, and tradability filters
- ETFs and large-cap names may be included as benchmarks and market context

Refresh cadence:

- Daily or weekly

Purpose:

- Keep daily technical state ready for potential candidates
- Support audit visibility across sectors and market-cap buckets

### Active Candidate Set

The active candidate set is the narrower list scored for current prediction.

Sources:

- Finviz Elite screens
- User-provided watchlists
- Alpaca active/tradable universe filters
- Catalyst-driven candidates
- Sector or theme lists

Typical size:

- Around 100 names for daily operation
- Larger batches are acceptable for offline prediction and training

The active candidate set should usually be a subset of the superset. If it is not, the miss should be logged and the ticker should receive a backfill/audit status before predictions are trusted.

## 5. Data Sources

Primary sources:

- Alpaca premium news and market bars
- Seeking Alpha via RapidAPI for SA news, analysis, ratings/quant-like snapshots, earnings, and symbol data
- SEC EDGAR filings and company facts
- Reddit API for community attention and chatter where credentials are configured
- Finviz Elite for candidate discovery and screening

Market context sources:

- SPY
- QQQ
- Sector ETFs
- Theme ETFs when relevant
- Broad market/global news feeds

### Canonical Point-In-Time Boundary

All production training and inference inputs pass through `src/market_predictor/canonical/` before model-specific feature engineering.

- Bars carry interval start, interval end, availability, ingestion, feed, adjustment, and schema provenance. Provider timestamps are treated as left edges; closing OHLCV is never available at the left edge.
- Events carry publication, provider update, first-seen, raw availability, sentiment-scoring availability, and final feature availability. Joins use final feature availability.
- Source attempts carry typed `observed`, `observed_empty`, `partial`, `failed`, `disabled`, and `not_collected` state plus request coverage end and completion availability. Zero events is evidence only when the source was successfully queried. A successful state expires when its coverage end is older than the configured decision-time limit.
- Universe memberships carry effective windows and snapshot availability. A decision must have exactly one membership that was both effective and known.
- Fundamental and quant facts require versioned availability. Current snapshots cannot be backfilled over historical rows.
- Artifacts are written immutably with SHA-256 input/output identities and audit evidence; the manifest is published last.

Production requires observed history. Provider publication-time history collected after the fact is an explicit research proxy and cannot be promoted or served as production-ready data.

Feed-quality requirement:

- Volume-sensitive features should only be trusted when feed coverage is known.
- SIP/full consolidated coverage is required for production-grade volume features.
- IEX-only or unknown partial-feed volume should set readiness to `warn` or `invalid` for volume-heavy predictions.

## 6. Warm-Up And Feature Readiness

Warm-up is a backfill problem, not a waiting problem.

The system should reconstruct indicator state from historical bars. It should not wait for live bars to slowly accumulate enough history before making a prediction.

### Daily Warm-Up

Daily features support swing trend, regime, relative strength, volatility, and context.

Minimum:

- `daily_bar_count >= 250`

Reason:

- SMA200 requires 200 daily bars before it exists.
- A 200-day slope check needs additional prior bars.
- Six months of daily bars is insufficient for daily gates using SMA200 or 200-day slope.

### Canonical Swing Model

The production C4 swing contract is `swing.features.v1` -> `swing.model.v1`.

- Decision: after the completed daily bar and all required feature/source timestamps.
- Entry reference: next exchange session open.
- Primary horizon: fifth exchange session close.
- Target: stock return after configured round-trip costs is positive.
- Retained outcomes: gross/net return, SPY/QQQ/sector excess return, MFE, MAE, exact path, and label availability.
- Features: daily technical state, SPY/QQQ/sector regime and relative strength, ticker catalysts, observed global context, point-in-time membership, optional as-of fundamentals, and decision-group cross-sectional ranks.
- Validation: horizon-purged expanding walk-forward, cross-fitted probability calibration, deterministic unseen-ticker holdout, and non-overlapping horizon-phase top-k economics.
- Promotion: both validation scopes, conservative economics, drawdown, regime, catalyst, alignment, memory, model hash, evidence hashes, and one matching `model_run_id` must pass.

Catalyst assessment remains an explanation and ranking overlay at serving time. It does not overwrite the estimator probability. Production serving rejects every older volatile schema even if an older registry manifest says `promoted`.

### Intraday Warm-Up

Intraday prediction is a supported model view. Its readiness gates apply only when an intraday route is requested; they must not block a daily-only swing response.

Minimum:

- `intraday_bar_count >= configured_intraday_warmup`
- Default recommendation: approximately 130 bars for MACD 12/26/9 stabilization.

The production 5-minute route also requires point-in-time session features, consecutive raw bars for label construction, known feed coverage, and a promoted intraday manifest. If intraday features are not part of a requested model view, intraday warm-up does not block daily-only prediction.

## 7. Data-Readiness Gates

Prediction quality depends on data validity. The model should not hide bad input behind a probability.

Required gates:

| Gate | Required condition | Failure behavior |
| --- | --- | --- |
| Daily bars | `daily_bar_count >= 250` when daily technical features are used | `invalid` |
| Intraday bars | `intraday_bar_count >= configured_intraday_warmup` when intraday features are used | `invalid` or `warn`, depending on model |
| News/candle alignment | Event timestamps map to actual ticker trading dates/candles | `invalid` |
| Cache freshness | Cached bars/events are inside configured TTL or revalidated | `invalid` |
| Feed type | Feed type is known for volume-sensitive features | `warn` if unknown, `invalid` if known partial and feature requires SIP |
| Event relevance | Recent news must be ticker-relevant or explicitly market/sector-context tagged | `warn` or exclude event |
| Source coverage | At least one valid recent source for catalyst-dependent predictions | `no_signal` |
| Required source state | Every configured source is `observed` or `observed_empty`, completed by the decision, and fresh by request coverage end | `invalid` |
| Universe membership | Exactly one effective and already-known snapshot per decision | `invalid` |
| Fundamental timing | Fact version was available by decision time; no current-snapshot backfill | `invalid` |
| Model/schema match | Feature schema matches model expectation | `invalid` |

Output must include `data_readiness_status` and `data_readiness_reasons`.

## 8. News And Candle Alignment

Event features must be joined to a completed candle only when the event-derived feature was available.

Rules:

- Intraday decisions use explicit bar-end plus finalization availability, never the provider's left-edge timestamp.
- Event inclusion uses `feature_available_at_utc <= decision_time_utc`; publication time alone is insufficient.
- Pre-market, intraday, after-hours, weekend, and holiday buckets remain explanatory metadata, not permission to bypass the as-of join.
- Daily research rows may map events to the next valid exchange session, but production still requires observed availability and a declared decision time.
- Events with no matching historical candle inside the scoring window are excluded or marked invalid.

Audit checks:

- No historical event should have a missing feature row after alignment.
- News counts by ticker/date should match between event input and generated features.
- No joined bar, event, sentiment score, source status, membership, or fundamental timestamp may exceed the decision time.
- Prediction reports should expose alignment status when available.

## 9. Model Registry

Model artifacts must be classified by lifecycle state.

### Baseline Models

Stable reference models used for comparison.

Examples:

- Last known clean daily model
- Last known clean event model
- Simple heuristic baseline

Baseline models are not necessarily the live promoted models. They provide a floor for validation.

### Candidate Models

Newly trained models not yet trusted for default prediction.

Candidate models must include:

- Training dataset path or artifact ID
- Training date
- Feature schema version
- Target definition
- Validation metrics
- Known limitations
- Whether post-event reaction features are included

### Promoted Models

Models approved for default prediction use.

Promotion requires:

- Validation meets or beats baseline gates
- No data-readiness audit failures
- Feature schema is compatible with prediction pipeline
- No leakage flags
- Out-of-sample selected-trade profitability audit passes
- Market-regime coverage audit passes
- Catalyst/news audit passes for catalyst-dependent models
- Human-readable model registry entry

Promotion must not rely on ROC AUC alone. A candidate can rank labels correctly while still producing poor trading economics, excessive selected-trade drawdown, or a brittle result that only works in one market regime. The production promotion report must therefore include:

- Classification metrics: ROC AUC, precision/recall, top-decile lift, validated rows, ticker count
- Profitability audit: selected-trade count, average return, win rate, profit factor, max drawdown
- Regime audit: rows and selected trades across risk-on, neutral, and risk-off regimes
- Catalyst/news audit: catalyst feature presence, source coverage, low-relevance event rate, alignment errors

### Deprecated Models

Models retained only for comparison or historical reproducibility.

Deprecated models should not be used by default prediction commands.

Common reasons:

- Superseded feature schema
- Known leakage risk
- Old data alignment logic
- Reaction features included where pre-trade prediction is required
- Narrow experimental dataset

## 10. Current Model Families

The repo currently contains several useful families. Their intended roles should be documented in the model registry.

| Family | Intended use | Status guidance |
| --- | --- | --- |
| Canonical swing 5D model | Estimate cost-adjusted net-positive return from next open to fifth close | C4 implementation complete; no real-data artifact promoted yet |
| Pre-C4 volatile 1D/5D models | Historical comparison only | Deprecated and rejected by canonical production serving |
| Intraday 5-minute technical entry-path model | Estimate target-before-stop probability over 12 bars | Candidate; current API artifact fails AUC/lift promotion gates |
| Intraday opening V2 models | Non-overlapping, cost-aware 09:30-11:30 ET setup experiments | Candidate artifacts with rejected promotion decision; not production-serving models |
| ML V3 B0/B1/B2/R1/D1 and O1 overlay | Cross-sectional opportunity ranking, separate downside risk, and external catalyst confirmation | C8 development evaluation complete; all available families/overlays rejected, R2 unavailable without microstructure, and no V3 artifact is production-serving |
| ML V4-H1 120-minute B0/R1 | Test whether longer exact paths and lower decision turnover cover costs | Development experiment complete and rejected; no artifact is production-serving |
| Daily market-context models | Daily 1D/5D direction from price, news, sentiment, sector, and SPY context | Baseline or promoted if validation gates pass |
| Event market-context pre-reaction models | Event-level 1D/5D probability using catalyst features before future reaction is known | Baseline or promoted |
| Calendar-safe plus-Finviz candidate models | Broader training set with corrected event-to-candle alignment | Candidate until promoted |
| Reaction-feature event models | Analysis after some reaction has already occurred | Not clean for pre-trade prediction |
| Finviz-only expansion models | Research models trained on candidate expansion set | Candidate/research only |
| Older clean/sector/swing models | Earlier iterations | Deprecated unless explicitly revalidated |

### Current Deployment State (2026-07-21)

The manifest, not the filename, is authoritative.

| Route / artifact | Manifest state | Validation summary | Operational decision |
| --- | --- | --- | --- |
| Swing 5D, `canonical_swing` / `swing.model.v1` | no promoted artifact | C4 code and frozen gates are verified; real-data metrics do not yet exist | Route remains not-ready until a hash-bound candidate passes every C4 gate. |
| Pre-C4 volatile swing artifacts | historical manifests only | Earlier reported classification metrics are not comparable to the new target and gates | Production API rejects their type/schema; no grandfathering. |
| Intraday 12-bar API default, 2026-07-09 technical ablation | `candidate` | ROC AUC 0.6014; top-decile lift 1.4719 | Research only; fails current 0.65 AUC and 2.0 lift gates. |
| Intraday opening V2 exact-path histogram model | `candidate` | 47,543 labeled rows; 196 tickers; ROC AUC 0.5806; lift 1.1764 | Promotion rejected. |
| Intraday opening V2 Extra Trees | `candidate` | ROC AUC 0.5783; lift 1.1442 | Promotion rejected. |
| Intraday opening V2 logistic baseline | `candidate` | ROC AUC 0.5641; lift 1.1836 | Promotion rejected. |
| Intraday opening V2 net-positive direction model | `candidate` | ROC AUC 0.4890; lift 0.9162 | Promotion rejected. |
| Intraday V4-H1 exact 120-minute R1 | `candidate` | NDCG@10 0.4868/0.5131; top-10 excess -0.0802%/-0.0629% | Promotion rejected; shadow not opened. |
The V2 structural dataset itself is valid for continued research: 47,614 rows, 196 eligible tickers, 122 sessions, exact 09:30-11:25 ET bar timestamps, no duplicate ticker/timestamp keys, and no cooldown gaps below 13 bars. Catalyst context covered 21.44% of rows; market context covered 87.28%. Reddit coverage was zero in this historical V2 table, so Reddit cannot be claimed as a trained intraday signal yet.

Selected-trade economics are the decisive V2 failure: 558 capped OOS trades produced -0.184% average net realized return, 35.13% win rate, 0.7076 profit factor, 30.28% maximum drawdown, and 64.44% negative periods. All three market regimes were represented, so the rejection is not caused by missing regime coverage.

Serving rules:

- Each prediction view is produced by exactly one registered model artifact. Probabilities from different targets or horizons are never averaged.
- Model routes, feature sources, universes, and promotion policy are server-owned; they are not API request fields.
- Every serving route requires a promoted manifest and a matching model artifact SHA-256 before deserialization.
- No route may silently substitute a candidate model.
- Unified responses may be partial and must include explicit per-view errors.
- A view with `warn` or `invalid` readiness is diagnostic only and emits `not_ready`, never an actionable signal.
- Readiness checks both snapshot generation time and the latest feature timestamp; republishing stale rows does not make them fresh.
- Catalyst/news remains an intraday confirmation and ranking overlay until a predeclared ablation on fresh data proves incremental model value.
- A new intraday hypothesis must first pass both development economics scopes; only then may matured observations after 2026-07-08 be used as an untouched shadow interval.

The data, target, ranking, validation, cleanup, and Git checkpoint sequence for the next model generation is defined in [ML Model V3 Improvement Plan](ml_model_v3_plan.md).

The production serving, cleanup, canonical-data, swing, intraday, deployment, and release sequence is defined in [Production ML Rebuild Plan](production_ml_rebuild_plan.md).

V3 checkpoints C1-C7 now provide strict point-in-time contracts, immutable development/shadow partitioning, exact next-open labels and costs, batch/live feature parity, cross-sectional ranks, session-purged walk-forward validation, deterministic ticker holdout, candidate adapters for B0/B1/B2/R1/D1, disjoint classifier calibration, and session-blocked independent-event economics. This changes research capability only. V3 artifacts remain outside production serving until the gate freeze and one-time shadow evaluation are completed.

C8 data readiness passes for 546 point-in-time S&P symbols over 501 source sessions. The labeled development artifact contains 1,063,587 rows over 478 eligible sessions and 24 hash-verified monthly shards. XNYS schedules define normal and early-close session boundaries; QQQ, SPY, and every required sector ETF must cover each expected 5-minute timestamp. Training accepts this directory only after validating all shard hashes, physical row count, builder schema, and dataset fingerprint.

The B0 deterministic floor, B1 logistic baseline, B2 nonlinear baseline, and R1 grouped ranker are rejected because their top-10 cost-adjusted excess returns are negative in both purged walk-forward and deterministic ticker-holdout evidence. R1 improves ranking NDCG but still returns -0.07153%/-0.07644% in the two scopes. D1 is also rejected as a downside gate. None crosses the economic floor. They remain research comparisons, not production models.

O1 keeps catalyst evidence outside the estimator and applies one fixed ranking adjustment plus material-negative vetoes to identical R1 OOF groups. The six-month audit joins 322,291 rows with zero future matches and complete sentiment for 72,818 relevant events. It improves walk-forward top-10 excess return only from -0.05744% to -0.04871% and worsens ticker holdout from -0.06423% to -0.06690%; both paired confidence intervals include zero. O1 is rejected. R2 is unavailable because C8 contains no observed microstructure inputs. C8 therefore closes without a candidate, and shadow evaluation remains unopened.

Historical sentiment inference uses locally cached FinBERT weights and immutable headline plus provider-summary inputs. Scored files carry model, input-mode, and token-limit provenance, and resume only when that tuple matches. Publication-time backfill remains research-only. Optional global context must cover both declared interval boundaries; a stale archive fails readiness instead of becoming zero-valued context.

The development-only R1 failure-attribution audit projects all available 30/60/120-minute and close outcomes from the verified C8 shards, preserves fixed top-10 groups, and reports score deciles plus month/fold/time/regime/sector/liquidity/volatility strata. It never reads shadow data and cannot promote a model. C8 shows near-zero rank correlation, no stable positive filter across both scopes, and an edge that improves with horizon but remains below costs.

V4-H1 changed the primary horizon and decision stride to 120 minutes. Its initial audit found that 2.32% of rank-eligible rows counted 24 observed bars across ticker-level candle gaps. The production label contract now rejects such rows and records `ml_v3.labels.v2`; it never fills or re-times bars. The corrected dataset has fingerprint `c2906f10b543327cc265798ecd81e019c5365dc9ede3e432b33ba881970cc612`, 505,049 rows, exact 120-minute exits, and 24 hash-verified shards.

On identical corrected groups, R1 improves B0 but remains negative after costs: -0.08015% walk-forward and -0.06285% ticker holdout. Both paired improvement intervals cross zero. V4-H1 is rejected, no serving manifest is promoted, and C9 remains closed.

Large V3 training reads only verified required columns, compacts selected features to `float32`, trains one deterministic CPU model at a time, and releases each fold model before the next fit. R1 records current and peak process working set and fails closed at a configurable memory guard; the C8 run stayed below its 4 GiB hard budget with a 3.781 GiB measured peak.

## 11. Audit Report Specification

The system should generate an audit report before model training, promotion, or production prediction review.

Required fields:

| Field | Meaning |
| --- | --- |
| `ticker` | Symbol audited |
| `raw_source_directory` | Raw event source directory used |
| `first_news_date` | Earliest event timestamp/date |
| `last_news_date` | Latest event timestamp/date |
| `months_covered` | Approximate event coverage window |
| `event_count` | Raw/sanitized event count |
| `source_families` | Source families represented |
| `feature_rows_1d` | Daily feature rows available for 1D model |
| `feature_rows_5d` | Daily feature rows available for 5D model |
| `event_feature_rows` | Event-level feature rows available |
| `daily_bar_count` | Available daily bars |
| `intraday_bar_count` | Available intraday bars if applicable |
| `news_candle_alignment_status` | Alignment result |
| `missing_historical_feature_rows` | Count of expected event dates missing from feature set |
| `dates_with_news_count_mismatch` | Count of ticker/date news-count mismatches |
| `feed_type` | SIP/IEX/unknown |
| `cache_status` | Fresh/stale/not_used |
| `model_eligibility` | Eligible/ineligible/warn |
| `eligibility_reasons` | Explanation for ineligible or warning status |

Initial known audit shape:

- Main cleaned two-year raw event set: 187 tickers, approximately 164k events, median coverage about 23.7 months.
- Finviz expansion set through 2026-06-12: 185 tickers, approximately 21k events, median coverage about 5.7 months.
- Calendar-safe daily final features: 372 tickers and approximately 213k rows per horizon.
- Calendar-safe event final features: 369 tickers and approximately 185k rows.

Implemented audit surfaces:

- `audit-swing-alignment` verifies historical event-to-candle mapping and news-count consistency.
- `build-swing-dataset` publishes a hash-verified feature/label artifact only after timing, warm-up, benchmark, SIP, adjustment, source-freshness, cross-section, and exact-path checks pass.
- `train-swing-model` writes purged predictions, unseen-ticker predictions, folds, profitability, regime, catalyst, alignment, metrics, and an evidence hash manifest tied to the candidate.
- `promote-swing-model` verifies the candidate and every evidence hash, then leaves any failing artifact in candidate state with a rejection report.
- The older `audit-promotion-readiness` / `promote-model` path remains only for pre-C5 intraday research and cannot promote a canonical swing model.

## 12. Prediction Workflow

Recommended flow:

1. Build or load candidate universe.
2. Collect recent catalysts and required market context.
3. Sanitize and deduplicate events.
4. Verify event relevance and timestamp quality.
5. Build feature rows.
6. Run data-readiness gates.
7. Score promoted models.
8. Combine daily, event, and heuristic outputs into a prediction package.
9. Write readable report, raw report, field dictionary, and audit ID.
10. Let `trading_flow` consume the prediction package for trade decisions.

## 13. Parameter Summary

| Parameter | Value / Cadence | Notes |
| --- | --- | --- |
| Superset size | ~1,500-2,000 symbols | Broad prediction-readiness universe |
| Superset refresh | Daily or weekly | Slow-changing |
| Active candidate size | ~100 typical | Finviz/user/catalyst driven |
| Active candidate refresh | Pre-open + intraday or scheduled batch | Drives prediction workload |
| Daily warm-up depth | >= 250 daily bars | Required for SMA200/slope features |
| Intraday warm-up depth | Configured, default ~130 bars | Only required when intraday features are used |
| Provider websocket ownership | Prefer `trading_flow` in production | `market-predictor` should avoid duplicate live streams |
| Feed tier | SIP/full coverage for volume-sensitive features | Unknown/partial feed reduces readiness |
| Source coverage age | 60 minutes by default | Decision minus request coverage end; stale success is invalid |
| Cache TTL | Deployment-defined | Revalidate stale cache |
| Model states | Baseline, candidate, promoted, deprecated | Required registry language |
| Audit output | CSV/report artifact | Implement after contract approval |
