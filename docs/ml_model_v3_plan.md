# ML Model V3 Improvement Plan

## 1. Purpose

Build a more reliable market-prediction engine for swing ranking and intraday setup ranking without tuning against the already-inspected six-month validation interval.

V3 changes the primary intraday question from:

```text
Will this row hit one fixed target before one fixed stop?
```

to:

```text
Among the stocks that were genuinely available at this decision time, which
ones have the best expected net outcome relative to their market and sector,
and what is the separate downside-path risk?
```

The model remains prediction intelligence. TradingFlow owns alerts, strategy decisions, risk, portfolio state, and execution.

## 2. Current Baseline

As of 2026-07-11:

| Model | State | Evidence |
| --- | --- | --- |
| S&P 500 5-day volatile mover | Promoted under earlier gates, conditional | ROC AUC 0.7126; top-decile lift 2.5936; requires current economics/regime re-audit. |
| S&P 500 1-day volatile mover | Candidate | ROC AUC 0.6657; top-decile lift 2.4850. |
| Intraday API technical model | Candidate | ROC AUC 0.6014; lift 1.4719. |
| Intraday opening V2 exact-path model | Candidate, promotion rejected | ROC AUC 0.5806; lift 1.1764; average selected net return -0.184%; profit factor 0.7076; max drawdown 30.28%. |
| Intraday opening V2 net-positive model | Candidate, promotion rejected | ROC AUC 0.4890; lift 0.9162. |

V2 data findings:

- 47,614 independent setup rows.
- 196 eligible tickers.
- 122 sessions.
- Catalyst/news present on 21.44% of rows.
- Historical Reddit coverage was zero.
- Monthly exact-path AUC ranged from approximately 0.497 to 0.645.

Conclusion: the main limitations are target noise, regime instability, cross-sectional information loss, limited history, sparse catalyst coverage, and missing microstructure. Model complexity alone is not the limiting factor.

## 3. Goals

1. Rank tradable candidates within each decision timestamp instead of treating every row as an unrelated binary example.
2. Predict net return and downside-path risk separately.
3. Recover sample efficiency with overlap weights while retaining independent-event evaluation.
4. Add point-in-time market, sector, session-volume, and optional microstructure context.
5. Validate temporal stability and unseen-ticker generalization.
6. Produce calibrated, versioned prediction evidence for TradingFlow shadow use.
7. Keep all promotion gates unchanged until a predeclared V3 gate specification is committed.

## 4. Non-Goals

- No FinGPT or general LLM in the numerical prediction path.
- No transformer/deep sequence model until tree/ranking baselines prove stable edge.
- No lowering AUC, lift, economics, or drawdown gates to force promotion.
- No model selection using the post-2026-07-08 shadow interval.
- No predictor-owned alerts, notifications, broker calls, stops, targets, or position sizing.
- No shared TradingFlow/Market Predictor database or OHLCV repository.
- No intraday production route until an intraday artifact is promoted.

## 5. Frozen Experimental Boundaries

### Development cutoff

All observations through `2026-07-08T23:59:59Z` are development data. They may be used for feature development, inner validation, and baseline comparison, but they can no longer provide an unbiased final performance estimate.

### Shadow interval

Observations after 2026-07-08 are appended to an immutable shadow partition. Labels may mature there, but the partition is not inspected for model or threshold selection.

The shadow partition is opened exactly once after:

- Dataset schema is frozen.
- Candidate model family is frozen.
- Feature set is frozen.
- Ranking and economic thresholds are frozen.
- Minimum shadow sessions and trade count are met.

If the shadow test fails, the candidate is rejected. The same shadow interval is never reused as an untouched test for its replacement.

## 6. Data Plan

### 6.1 Universe

Build a point-in-time US-listed universe, initially 300-500 liquid symbols, balanced across:

- Technology and semiconductors.
- Software, AI, and data-center infrastructure.
- Healthcare, biotechnology, and life sciences.
- Aerospace, defense, space, and advanced mobility.
- Financials and fintech.
- Consumer/high-beta names.
- Energy and commodity-sensitive names.

Requirements:

- Store universe membership with `effective_from` and `effective_to`.
- Preserve inactive/delisted symbols when source availability permits.
- Record sector, industry, market-cap bucket, liquidity bucket, and primary benchmark.
- Do not build historical candidate sets from today's survivors only.

### 6.2 History

Target 12-24 months of consecutive 5-minute bars for the balanced universe, with 1-minute bars for a smaller liquid subset when available. The history must include multiple volatility and market-direction regimes.

Required benchmarks:

- SPY and QQQ.
- Sector ETFs mapped point-in-time by ticker sector.
- Optional thematic ETFs where the mapping is explicit and stable.

### 6.3 Events and catalysts

Continue source-isolated collection from:

- Alpaca news.
- Seeking Alpha via RapidAPI.
- SEC filings and company facts.
- Finviz news/screener context.
- Global market/flashpoint context.
- Official Reddit access for prospective chatter.

Every event requires publication time, ingestion time, source family, normalized ticker mapping, relevance, event category, and deduplication identity.

Reddit policy:

- Treat Reddit as prospective attention data until sufficient official-history coverage exists.
- Do not silently backfill unavailable historical chatter with scraped or synthetic values.
- Missing Reddit history remains missing and is represented by source-availability features.

### 6.4 Market microstructure

Add only when source quality and licensing are confirmed:

- Bid/ask spread and spread percentage.
- Quote sizes and quote imbalance.
- Trade count and average trade size.
- Dollar volume.
- Short-sale restriction and halt state.

Microstructure features form a separate ablation. The core V3 ranker must train without them so their incremental value is measurable.

### 6.5 Data audits

No V3 training begins until audits pass:

- Consecutive-bar and session-calendar checks.
- Duplicate/gap detection.
- OHLCV sanity and corporate-action checks.
- Explicit SIP/feed provenance for volume features.
- Event publication/ingestion ordering.
- Exact news-to-bar availability alignment.
- Benchmark and sector mapping coverage.
- Point-in-time universe coverage.
- Feature null-rate and source-availability report.
- No post-decision data in any feature.

## 7. V3 Dataset And Labels

### 7.1 Decision row

A decision row represents a completed bar and a point-in-time eligible ticker. Entry reference is the next tradable bar open. Every row carries:

```text
ticker
decision_time_utc
feature_available_at_utc
entry_time_utc
session_date_et
decision_group_id
universe_snapshot_id
price_feed
feature_schema_version
```

`decision_group_id` groups all candidates that were eligible at the same decision timestamp. It is the ranking query/group identifier.

### 7.2 Continuous targets

Compute from the next-bar entry after configured costs:

- Net return over 30, 60, and 120 minutes.
- Net return to the regular-session close.
- Excess return versus QQQ over the identical interval.
- Excess return versus the mapped sector ETF over the identical interval.
- Maximum favorable excursion.
- Maximum adverse excursion.
- Time to maximum favorable/adverse excursion.

### 7.3 Path targets

Retain separate path labels:

- Target before stop.
- Stop before target.
- Timeout.
- Cost-adjusted realized return under the declared barrier policy.

Path targets support downside/risk models. They are not the sole ranking objective.

### 7.4 Ranking target

For each `decision_group_id`, rank eligible names by future net excess return. Convert the ordering into a deterministic graded relevance target suitable for ranking objectives. The grade construction is frozen before shadow evaluation and requires a minimum group size.

Primary ranking evaluation is top-k candidate quality, not row-level accuracy.

### 7.5 Overlap handling

Training keeps valid overlapping rows but assigns concurrency-aware weights so repeated horizons do not count as independent observations. Evaluation and profitability simulation use non-overlapping events with the configured cooldown.

Required audit fields:

```text
concurrent_label_count
overlap_weight
independent_event_id
cooldown_bars
```

Hard filtering is not used to discard most training information.

## 8. Feature Plan

### 8.1 Core technical/session features

- Returns and volatility over multiple intraday windows.
- EMA/MACD/RSI/ATR state and slope, using completed bars only.
- Session VWAP distance and slope.
- Running opening range and premarket range.
- Overnight gap and overnight range.
- Same-minute relative volume using prior sessions only.
- Dollar-volume and liquidity state.
- Distance from recent highs/lows normalized by ATR.
- Time-of-day encoding.

### 8.2 Cross-sectional features

At each decision group, calculate point-in-time ranks or robust z-scores for:

- Return and momentum.
- Relative volume and dollar volume.
- Volatility and ATR percentage.
- Distance from VWAP/opening range.
- Relative strength versus QQQ and sector ETF.
- Gap and overnight movement.

Cross-sectional transformations use only rows available in the current decision group.

### 8.3 Market and regime features

- SPY/QQQ returns and volatility over matching windows.
- Sector ETF returns and relative strength.
- Breadth across the current eligible universe.
- Risk-on, neutral, risk-off, and high-volatility state.
- Session type: trend, mean-reverting, gap, earnings-heavy, or macro-shock where objectively defined.

### 8.4 Catalyst overlay

Ticker catalyst/news remains outside the first technical ranker. Build an auditable overlay using:

- Event recency and materiality.
- Source diversity.
- Relevance and sentiment.
- Event taxonomy.
- Novelty/duplication state.
- Global-context direction and affected sector.

Compare technical ranking versus technical ranking plus catalyst confirmation on fresh walk-forward folds. Catalyst enters the core model only if a predeclared ablation shows stable incremental value.

## 9. Model Experiments

Run all candidates on identical splits and immutable feature sets.

| ID | Model | Purpose |
| --- | --- | --- |
| B0 | Deterministic technical heuristic | Non-ML ranking floor. |
| B1 | Regularized logistic classifier | Simple direction/path baseline and calibration reference. |
| B2 | Histogram gradient boosting | Current nonlinear classifier baseline. |
| R1 | Gradient-boosted learning-to-rank | Primary cross-sectional ranking candidate grouped by `decision_group_id`. |
| R2 | R1 plus microstructure | Measure incremental quote/trade value. |
| O1 | R1 plus catalyst overlay outside model | Test confirmation/ranking value without contaminating core probability. |
| D1 | Separate downside/path classifier | Stop-first and adverse-excursion risk. |

XGBoost LambdaMART is the first planned ranking implementation because it supports grouped ranking objectives. It is an optional dependency isolated behind a model adapter. Official reference: <https://xgboost.readthedocs.io/en/stable/tutorials/learning_to_rank.html>.

Do not add another model family unless it answers a documented failure in these baselines.

## 10. Training And Tuning

### 10.1 Walk-forward structure

- Outer folds are ordered by session date.
- Embargo covers the longest target horizon.
- Inner folds tune only within each outer training window.
- Candidate selection uses aggregate outer-fold evidence, not one favorable month.
- A separate ticker-holdout audit measures behavior on symbols excluded from fitting.

### 10.2 Reproducibility

Every run records:

- Code commit.
- Dataset fingerprint.
- Feature schema hash.
- Universe snapshot IDs.
- Exact target/cost configuration.
- Model family and parameters.
- Random seeds.
- Fold dates and embargo.
- Dependency versions.
- CPU/GPU execution mode.

GPU is an optimization, not a model-quality requirement. Establish deterministic CPU baselines first.

### 10.3 Calibration

Calibrate classifier probabilities only on observations disjoint from model fitting. Compare sigmoid and isotonic calibration, then freeze one method before shadow evaluation. Official reference: <https://scikit-learn.org/stable/modules/calibration.html>.

Ranking scores are not probabilities. Do not expose them as probabilities unless a separate validated mapping is fitted and audited.

## 11. Validation And Promotion

### 11.1 Statistical metrics

Classifiers:

- ROC AUC and precision-recall AUC.
- Top-decile lift and precision at configured selection size.
- Brier score and calibration error.
- Fold/month/regime dispersion.

Rankers:

- NDCG@k.
- Precision@k and hit rate@k.
- Mean and median top-k excess return.
- Rank correlation with net excess return.
- Performance versus B0/B1/B2 on identical decision groups.

### 11.2 Trading-economics metrics

- Selected independent trades.
- Average and median net return.
- Win rate and profit factor.
- Maximum drawdown.
- Return/drawdown ratio.
- Negative-period rate.
- Turnover, exposure, and estimated capacity.
- Results by month, sector, market-cap bucket, liquidity, and regime.

### 11.3 Uncertainty

Use session/block bootstrap confidence intervals. Report point estimates and conservative bounds. A candidate cannot be promoted because of one favorable point estimate.

### 11.4 Frozen promotion gates

Existing classifier gates remain in force:

- ROC AUC at least 0.65.
- Top-decile lift at least 2.0.
- At least 20,000 validated rows and 200 tickers where applicable.
- At least 100 selected independent trades.
- Positive average selected return.
- Profit factor at least 1.05.
- Maximum drawdown no more than 25%.
- Return/drawdown ratio at least 0.5.
- Negative-period rate no more than 55%.
- Alignment, regime, catalyst, readiness, and artifact-integrity audits pass.

The ranker gate specification is committed after B0/B1/B2/R1 development baselines are available and before the shadow interval is opened. It must include minimum improvement over baseline and economic/confidence-bound requirements.

### 11.5 TradingFlow ablation

After model-level promotion, TradingFlow must compare:

1. Strategy without ML.
2. Strategy with V3 in observe mode.
3. Strategy with the proposed optional/required gate.

Only TradingFlow can prove strategy-level incremental value after portfolio constraints and execution costs.

## 12. Serving And Monitoring

V3 serving evidence adds:

```text
ranking_score
rank_within_decision_group
expected_net_return_30m
expected_net_return_60m
expected_excess_return_vs_qqq
expected_excess_return_vs_sector
stop_first_probability
adverse_excursion_risk
uncertainty_or_calibration_status
```

Rules:

- Contract version increments for V3.
- Paper/live requests require promoted manifests.
- Candidate use remains an explicit research override.
- Model health monitoring tracks feature drift, score distribution, calibration, realized outcomes, and source coverage.
- Model health outputs are reports/metrics, not user alerts.
- TradingFlow remains the only alert and execution owner.

## 13. Code Quality And Cleanup

V3 must not be added as another layer inside the current large research modules. Cleanup is part of delivery, not an unrelated follow-up.

### 13.1 Ownership cleanup

- Remove predictor-owned runtime alert generation after preserving a rule-parity specification for TradingFlow.
- Remove `monitor-alerts`, `backtest-alerts`, `alerts.py`, and predictor alert storage/schedules.
- Rename prediction-only monitor commands/reports to ranking or analysis terminology.
- Add an architecture test preventing imports of broker, order, notification, or alert-delivery code into predictor serving/training modules.

### 13.2 Module boundaries

Split oversized modules along stable ownership boundaries:

```text
commands/data.py          collection and dataset commands
commands/train.py         training commands
commands/audit.py         validation/promotion commands
commands/serve.py         API and scoring commands
datasets/intraday.py      point-in-time row construction
labels/intraday.py        V3 labels and overlap weights
features/intraday.py      offline/live feature parity
models/baselines.py       classifier baselines
models/ranking.py         grouped ranking adapters
models/calibration.py     disjoint calibration
validation/walkforward.py splits, embargo, ticker holdout
validation/economics.py   independent-event economics
```

The exact paths may follow existing package conventions, but each module must have one primary reason to change. CLI commands remain thin orchestration layers.

### 13.3 Typed contracts and errors

- Replace loosely shaped dictionaries at module boundaries with dataclasses/Pydantic models.
- Centralize feature/target names and schema versions.
- Add domain-specific exceptions for data readiness, schema mismatch, leakage audit, artifact integrity, and promotion failure.
- Validate configuration at startup; remove hard-coded dated input/model paths from production defaults.
- Keep source failures isolated per ticker/source while making aggregate incompleteness explicit.

### 13.4 Performance and resource safety

- Use PyArrow dataset scanning, column projection, and partition filtering instead of loading multi-gigabyte Parquet files wholesale.
- Partition intermediate data by date/ticker with atomic manifests.
- Vectorize label/feature computation where correctness is preserved.
- Use bounded worker pools and deterministic group ordering.
- Measure peak memory, elapsed time, rows/second, and artifact size for representative builds.
- Keep GPU optional and isolated to workloads that benefit from it.

### 13.5 Quality gates

Add repository-level checks:

- Ruff formatting/linting.
- Incremental static typing for new/changed V3 modules.
- Unit, property-style label math, schema contract, and integration tests.
- Leakage and point-in-time invariants as tests, not comments.
- CLI help/smoke tests.
- API/OpenAPI snapshot tests.
- Secret scanning and dependency vulnerability checks in CI.
- `git diff --check`, compile, and full test suite before every checkpoint.

No checkpoint may reduce existing coverage or silence a warning without a documented reason.

### 13.6 Artifact and legacy cleanup

- Replace date-coded production defaults with configuration/registry resolution.
- Keep legacy artifacts readable for reproducibility but mark their lifecycle state explicitly.
- Add a report that identifies unmanifested, orphaned, superseded, or rejected model artifacts; cleanup remains explicit and non-destructive.
- Do not commit generated data/models while refactoring artifact discovery.
- Preserve migration notes whenever a schema or CLI command changes.

## 14. Git Checkpoints

Each checkpoint is independently reviewable, tested, and reversible. Do not combine data collection, label changes, model changes, and serving changes in one commit.

### C0 - Plan freeze

Scope:

- Add this plan.
- Link it from README and model/integration documentation.
- No source-code behavior changes.

Suggested commit:

```text
Plan ML model V3 improvements
```

Exit criteria:

- Plan is internally consistent.
- Development/shadow cutoff is explicit.
- Git and runtime artifact policies are explicit.

### C1 - Remove predictor alert ownership

Status: completed on 2026-07-11.

Scope:

- Preserve a technical-rule parity matrix for TradingFlow.
- Remove predictor runtime alert commands, module, outputs, and schedule documentation.
- Rename prediction analysis commands that use alert/monitor terminology.
- Add dependency-boundary tests.

Tests:

- Full suite after command removal.
- CLI help contains no runtime alert commands.
- Prediction API contains no notification/broker behavior.

Suggested commit:

```text
Remove predictor-owned alert runtime
```

### C2 - Code quality foundation

Scope:

- Split CLI, dataset, label, model, and validation responsibilities incrementally.
- Add typed V3 schema/error foundations.
- Add lint, formatting, typing, contract, and performance-smoke gates.
- Remove hard-coded dated production defaults where touched.

Tests:

- Existing behavior remains covered while modules move.
- CLI/API contract snapshots pass.
- Representative dataset smoke build stays within a recorded memory/time budget.

Suggested commit sequence:

```text
Modularize prediction pipeline commands
Add typed V3 schema foundations
Enforce predictor code quality gates
```

### C3 - V3 data contracts and audits

Scope:

- Add point-in-time universe schema.
- Add decision-row/source-availability schemas.
- Add bar/event/benchmark audit commands.
- Add shadow-partition write protection.

Tests:

- Invalid/future timestamps fail.
- Duplicate/gapped bars fail according to policy.
- Universe membership is point-in-time.
- Shadow rows cannot be read by development training commands.

Suggested commit:

```text
Add V3 point-in-time data contracts
```

### C4 - V3 targets and overlap weighting

Scope:

- Add continuous, excess-return, MFE/MAE, and path labels.
- Add decision groups, overlap weights, and independent event IDs.
- Preserve next-bar entry and exact benchmark intervals.

Tests:

- Synthetic path tests for every label.
- Cost and benchmark math tests.
- No same-bar or future-feature leakage.
- Overlap weights and cooldown behavior.

Suggested commit:

```text
Add V3 ranking targets and overlap weights
```

### C5 - Cross-sectional feature pipeline

Scope:

- Add point-in-time ranks/z-scores.
- Add market/sector/session features.
- Add core versus microstructure feature sets.

Tests:

- Cross-sectional values use current decision group only.
- Sector/benchmark mapping is as-of safe.
- Opening range and same-minute volume remain leak-safe.
- Feature parity between offline and live builders.

Suggested commit:

```text
Build V3 cross-sectional features
```

### C6 - Ranking and baseline model adapters

Scope:

- Add B0/B1/B2/R1/D1 training adapters.
- Add optional XGBoost ranking dependency.
- Add grouped purged walk-forward training.
- Add ticker-holdout audit.

Tests:

- Group boundaries are preserved.
- Purge/embargo excludes target overlap.
- Candidate features exist in every training fold.
- Deterministic seed/model manifest behavior.

Suggested commit:

```text
Add V3 ranking model baselines
```

### C7 - Calibration, uncertainty, and economics

Scope:

- Add disjoint calibration workflow.
- Add ranking metrics and block-bootstrap intervals.
- Extend promotion audits for rankers and multi-output evidence.

Tests:

- Calibration data is disjoint.
- Bootstrap is session-blocked and reproducible.
- Promotion rejects missing, weak, or unstable evidence.

Suggested commit:

```text
Audit V3 calibration and ranking economics
```

### C8 - Development training run and gate freeze

Scope:

- Build development dataset through the frozen cutoff.
- Run B0/B1/B2/R1/R2/O1/D1 ablations.
- Select one candidate using development evidence only.
- Commit the V3 ranker promotion-gate specification before opening shadow data.

Git artifacts:

- Code/config/docs and compact model card only.
- No raw data, feature tables, model binaries, credentials, or large reports.

Suggested commit:

```text
Freeze V3 candidate and promotion gates
```

### C9 - Shadow evaluation

Scope:

- Verify minimum shadow maturity.
- Run the single predeclared shadow evaluation.
- Record pass/reject decision without tuning.

Suggested commit:

```text
Record V3 shadow evaluation decision
```

Exit criteria:

- Failed candidates remain unpromoted.
- Passed candidates satisfy every model gate and carry immutable evidence.

### C10 - Prediction API V3

Scope:

- Version the response contract.
- Add ranking/return/risk evidence.
- Add capability/readiness endpoint and correlation ID.
- Preserve strict promoted-model enforcement.

Tests:

- Contract and OpenAPI snapshot tests.
- Point-in-time `as_of` tests.
- Candidate rejection and partial-response tests.
- Artifact SHA and feature-schema compatibility tests.

Suggested commit:

```text
Serve V3 prediction evidence
```

### C11 - TradingFlow shadow integration

This checkpoint belongs in the `trading_flow` repository after C10:

- Add typed V3 evidence client/cache/repository.
- Integrate observe mode only.
- Store prediction identity in decision audits.
- Run no-ML versus observe versus gated backtests.
- Keep alerts and execution entirely in TradingFlow.

Suggested TradingFlow commit sequence:

```text
Add Market Predictor evidence contracts
Persist point-in-time prediction evidence
Show V3 evidence in strategy audits
Backtest V3 prediction confirmation
```

## 15. Runtime Artifact Policy

Never commit:

- `.env` or API credentials.
- Raw bars/news/Reddit/SEC/Seeking Alpha data.
- Feature Parquet/CSV datasets.
- Trained `.joblib`/booster artifacts.
- API caches.
- Full OOS prediction/trade reports.

Commit:

- Source, tests, schemas, and configuration examples.
- Compact model cards and promotion/rejection summaries without secrets.
- Feature/target contracts.
- Reproducible commands and dependency locks.
- Documentation and architecture decisions.

Durable runtime artifacts belong in the Market Predictor artifact store/Azure prefix, independent of TradingFlow storage.

## 16. Definition Of Done

V3 is complete only when:

- Point-in-time universe, bar, event, benchmark, and source audits pass.
- Development and shadow partitions are cryptographically fingerprinted and isolated.
- Cross-sectional ranking and downside evidence are reproducible.
- Baseline, feature, microstructure, catalyst, and regime ablations are complete.
- Calibration and confidence-bound reports are complete.
- A candidate passes the one-time fresh shadow evaluation and every promotion gate.
- API V3 serves promoted evidence with model/schema identity.
- TradingFlow proves incremental strategy value in deterministic backtest and paper shadow mode.
- Market Predictor contains no runtime alert or execution ownership.
- V3 modules pass lint, typing, contract, leakage, performance-smoke, and full regression tests.
