# Swing Prediction Intelligence Architecture

This document defines the operational design for `market-predictor` as a prediction intelligence system.

The system produces swing and daily momentum predictions from catalysts, news, filings, sentiment, market context, and price behavior. It does not own final trade execution, portfolio state, order routing, position sizing, stops, or exits.

Scope:

- US-listed equities and ETFs
- Mid/small-cap swing and momentum prediction, with large-cap/ETF context
- Long-side prediction signals and watchlist ranking
- Historical and recent catalyst analysis
- Model training, validation, promotion, and audit reporting

Out of scope:

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
- Both systems must not independently open production market-data streams for the same account unless provider limits and symbol ownership are explicitly managed.

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
- `daily_model_probability_up`
- `event_model_probability_up`
- `heuristic_watch_score`
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

### Intraday Warm-Up

Intraday features are optional for this repo. They are useful when the prediction includes same-day momentum confirmation, premarket movement, or intraday reaction behavior.

Minimum:

- `intraday_bar_count >= configured_intraday_warmup`
- Default recommendation: approximately 130 bars for MACD 12/26/9 stabilization.

If intraday features are not part of a model, intraday warm-up should not block daily-only prediction.

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
| Model/schema match | Feature schema matches model expectation | `invalid` |

Output must include `data_readiness_status` and `data_readiness_reasons`.

## 8. News And Candle Alignment

Event timing must map to the next valid trading candle for that ticker.

Rules:

- Pre-market events map to the current trading date if the market has not opened.
- Intraday events map to the current trading date.
- After-hours events map to the next actual trading date.
- Weekend and holiday events map to the next actual trading date.
- Events with no matching historical candle inside the scoring window are excluded or marked invalid.

Audit checks:

- No historical event should have a missing feature row after alignment.
- News counts by ticker/date should match between event input and generated features.
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
| Daily market-context models | Daily 1D/5D direction from price, news, sentiment, sector, and SPY context | Baseline or promoted if validation gates pass |
| Event market-context pre-reaction models | Event-level 1D/5D probability using catalyst features before future reaction is known | Baseline or promoted |
| Calendar-safe plus-Finviz candidate models | Broader training set with corrected event-to-candle alignment | Candidate until promoted |
| Reaction-feature event models | Analysis after some reaction has already occurred | Not clean for pre-trade prediction |
| Finviz-only expansion models | Research models trained on candidate expansion set | Candidate/research only |
| Older clean/sector/swing models | Earlier iterations | Deprecated unless explicitly revalidated |

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

Implementation note:

- The audit CSV command is intentionally not part of this document change.
- Add it later only after the report contract is accepted.

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
| Cache TTL | Deployment-defined | Revalidate stale cache |
| Model states | Baseline, candidate, promoted, deprecated | Required registry language |
| Audit output | CSV/report artifact | Implement after contract approval |
