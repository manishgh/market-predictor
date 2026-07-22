# Statistical and ML Validity Review - 2026-07-22

Review base: commit `48ac758` (`Resolve repository-wide lint and type debt`)

Scope: canonical swing and intraday datasets, labels, point-in-time joins, model fitting, calibration, validation, promotion, serving policy, replay, drift, tests, model manifests, and persisted research evidence. This is a review-only deliverable. No production code, tests, configuration, datasets, models, or registry entries were changed.

## P0 Findings

### P0-1 - The ticker holdout is trained with future dates and is not a causal holdout

- **Location:** `src/market_predictor/swing/model.py:88-101`, `src/market_predictor/swing/model.py:139-178`, `src/market_predictor/intraday/model.py:88-101`, `src/market_predictor/intraday/model.py:148-220`.
- **Impact:** Held-out-ticker AUC, lift, calibration, and economics can use relationships learned from dates later than the held-out prediction. Both promoters consume these contaminated metrics. A candidate can therefore pass an advertised unseen-ticker gate and conservative economic gate with look-ahead leakage.
- **Evidence:** The purged walk-forward splitter is applied only to `development`. The separate holdout estimator is then fit once on all development sessions and used to score all rows for held-out tickers, including rows earlier than the latest development sessions. Holdout economics are concatenated with causal walk-forward economics and included in the conservative aggregate. The splitter itself is causal for its outer folds (`src/market_predictor/v3/validation.py:59-114`); the holdout path bypasses it.
- **Remediation:** Produce two-dimensional out-of-sample evidence. For every chronological outer fold, fit only on non-holdout tickers in that fold's training sessions and score holdout tickers only in that fold's test sessions. Calibrate from data strictly earlier than the scored session. Reserve any final all-date ticker fit for production training, never validation.
- **Verification:** Persist fold-level `max_train_label_available_at_utc`, `min_test_decision_time_utc`, ticker sets, and row identities. Require `max_train_label_available_at_utc < min_test_decision_time_utc` for every main and ticker-holdout fold. Add a poison test where changing future labels/features cannot alter any earlier holdout probability or metric.

### P0-2 - Serving changes rank and action semantics with a catalyst heuristic that promotion never evaluates

- **Location:** `src/market_predictor/catalyst_overlay.py:138-147`, `src/market_predictor/prediction_service.py:379-415`, `src/market_predictor/prediction_service.py:459-500`, `src/market_predictor/prediction_service.py:836-895`, `src/market_predictor/swing/evaluation.py:82-120`, `src/market_predictor/intraday/evaluation.py:121-176`.
- **Impact:** Promoted model economics do not describe the rankings or action labels returned by the API. The fixed `+0.04`, `-0.08`, and `-0.20` catalyst adjustments can reorder candidates and convert otherwise actionable signals into conflicts or vetoes without out-of-sample validation. Swing also includes catalyst variables in the estimator, so the serving adjustment can double count catalyst evidence.
- **Evidence:** Swing validation ranks only `swing_probability`; intraday validation ranks only opportunity/downside probabilities. Serving ranks `overlay_decision_score` and applies catalyst-dependent signal branches. Persisted evidence explicitly rejected the tested catalyst ranking overlay: `docs/model_cards/v3_c8_o1_20260721.md:25-36` reports negative economics and confidence intervals spanning zero, then says not to serve or tune the overlay.
- **Remediation:** Immediately make serving rank and signal policy identical to the validated estimator policy. Keep catalyst as explanation/confirmation metadata only, or implement vetoes as a separately versioned policy artifact. Re-enable any ranking or action adjustment only after exact serving-policy replay, paired ablation against the unmodified model, and its own promotion gates.
- **Verification:** Replay OOF rows through the production response-building function and assert that API rank/action selection exactly matches promoted evidence. Add a policy-version hash to prediction snapshots. Require positive paired benchmark-excess-return confidence bounds for the overlay in causal walk-forward, ticker holdout, and untouched shadow evidence before activation.

### P0-3 - News/candle alignment promotion evidence is fabricated or structurally incomplete

- **Location:** `src/market_predictor/swing/model.py:181-194`, `src/market_predictor/intraday/model.py:586-610`, `src/market_predictor/swing/promotion.py:152-170`, `src/market_predictor/intraday/promotion.py:175-197`.
- **Impact:** Promotion can report zero news/candle mismatches without reconciling source events to feature rows. Missing events, unmatched events, missing historical feature rows, and daily news-count mismatches can silently pass the exact gate intended to prevent unrelated or temporally misaligned news from entering predictions.
- **Evidence:** Swing training constructs a one-row alignment table with every component hardcoded to zero. Intraday computes future-feature, path, and benchmark errors, but hardcodes `events_without_feature_row`, `missing_historical_feature_rows`, and `dates_with_news_count_mismatch` to zero. Both promoters trust these values after only run-ID/hash checks. Existing tests verify provenance and one intraday path mismatch, not event-to-feature reconciliation (`tests/test_swing_model.py:126-145`, `tests/test_intraday_model_v1.py:127-142`).
- **Remediation:** Build alignment evidence from immutable event-level lineage: source event ID, normalized ticker, publication/first-seen/scoring/feature-availability timestamps, matched decision row IDs, window membership, dedup identity, and source-attempt status. Reconcile raw accepted events to feature counts per ticker/decision window and preserve unmatched reasons.
- **Verification:** Inject future, wrong-ticker, duplicate, missing-feature, and news-count-mismatch fixtures and require nonzero component counts plus promotion rejection. For real evidence, require equality between accepted event IDs and the union of matched and explicitly rejected IDs, with zero unexplained rows.

### P0-4 - Canonical promotion has no untouched-shadow, uncertainty, or frozen-baseline gate

- **Location:** `src/market_predictor/swing/contracts.py:199-220`, `src/market_predictor/swing/promotion.py:79-170`, `src/market_predictor/intraday/contracts.py:233-260`, `src/market_predictor/intraday/promotion.py:74-197`, `docs/ml_model_v3_plan.md:74-84`.
- **Impact:** Repeated feature, model-family, horizon, threshold, and overlay experiments can overfit development walk-forward evidence. Point estimates just above zero are sufficient for promotion even when their uncertainty includes material losses. There is no enforcement against reusing an inspected interval, so multiple-testing and researcher-selection bias are unbounded.
- **Evidence:** Canonical gates require point AUC/lift/economic thresholds but no shadow partition identity, one-time-use ledger, paired baseline delta, bootstrap lower bound, or minimum confidence. The earlier V3 evaluation implementation already supports session-block confidence intervals (`src/market_predictor/v3/evaluation.py:275-343`), and current real research evidence demonstrates why they matter: `docs/model_cards/v4_h1_120m_20260721.md:35-56` has apparent model improvements whose paired intervals include zero and correctly rejects both candidates.
- **Remediation:** Add an immutable experiment registry and a one-time shadow evidence bundle to canonical promotion. Require a predeclared baseline, dataset/shadow fingerprints, hypothesis ID, model/policy hash, minimum independent sessions, session-block bootstrap intervals, and positive lower confidence bounds for benchmark-relative top-k economics. A failed shadow interval must be retired for that hypothesis family.
- **Verification:** A candidate with positive point return but a lower confidence bound at or below zero must fail. Reusing a consumed shadow fingerprint must fail. Mutation of cutoff, baseline, policy, feature list, or hypothesis after evidence generation must invalidate the bundle.

## P1 Findings

### P1-1 - Unknown relevance is treated as fully relevant and can evade low-relevance gates

- **Location:** `src/market_predictor/canonical/joins.py:150-162`, `src/market_predictor/canonical/joins.py:284-318`, `src/market_predictor/swing/dataset.py:285-321`, `src/market_predictor/intraday/dataset.py:511-560`.
- **Impact:** A ticker-tagged article with missing relevance contributes a relevance weight of `1.0`, event counts, sentiment, and source counts. Missing relevance is not counted as low relevance, so unrelated or weakly mapped news can influence swing features and serving overlays while the promotion relevance audit appears healthy.
- **Evidence:** All three event aggregators use `fillna(1.0)` for relevance. Low-relevance fractions count only values below `0.5`; unknown values converted to `1.0` are therefore reported as high quality. This compounds the Reddit relevance concern already identified in the architecture review.
- **Remediation:** Represent relevance as three states: validated numeric relevance, explicitly irrelevant, and unknown. Exclude unknown ticker events from estimator/overlay inputs by default, retain separate unknown/mapping-quality features, and require deterministic ticker-link evidence for direct catalysts. Keep market/global events in a separate context channel.
- **Verification:** Null-relevance and wrong-ticker fixtures must not increment eligible catalyst counts or sentiment. They must increment unknown/unmatched audit counters. Promotion must reject excessive unknown or unmapped event rates.

### P1-2 - Economic labels and backtests do not model executable fills, liquidity, or capacity

- **Location:** `src/market_predictor/swing/dataset.py:447-470`, `src/market_predictor/intraday/labels.py:122-173`, `src/market_predictor/swing/evaluation.py:94-120`, `src/market_predictor/intraday/evaluation.py:131-176`.
- **Impact:** Fixed 10 bps round-trip cost can materially overstate edge in volatile small/mid-cap names. Intraday target/stop outcomes assume fills exactly at barrier prices even when a one-minute bar opens through a stop or target. Top-k economics have no spread, participation, volume capacity, delayed/no-fill, halt, or price-impact model. Reported return and drawdown are therefore not deployment economics.
- **Evidence:** Labels subtract one constant cost from bar-derived returns. Barrier outcomes assign `target_price` or `stop_price` directly. Selection is unconstrained by dollar volume or order size. The ad hoc investment replay supports slippage and commission, but those parameters are not part of training/promotion economics (`src/market_predictor/investment_replay.py:263-270`).
- **Remediation:** Freeze a versioned execution-cost policy by liquidity/price/volatility bucket; use conservative gap-through fills, spread and slippage estimates, participation caps, and no-fill/halt rules. Produce economics and drawdown over a cost/capacity stress grid, with the intended capital allocation policy.
- **Verification:** Gap-through-stop tests must fill at the worse executable open, not the barrier. Promotion must remain positive at declared base and stress costs, and publish capacity curves showing trade count, turnover, and benchmark excess return by capital/participation level.

### P1-3 - Regime gates prove representation, not performance stability

- **Location:** `src/market_predictor/swing/evaluation.py:155-173`, `src/market_predictor/intraday/evaluation.py:208-226`, `src/market_predictor/swing/promotion.py:122-135`, `src/market_predictor/intraday/promotion.py:143-156`.
- **Impact:** A model may lose heavily in risk-off or high-volatility periods and still pass because the audit checks only that regimes exist and no regime owns too many rows. This does not validate market-regime robustness or drawdown concentration.
- **Evidence:** Regime detail rows contain only counts/shares. Promotion reads only `regimes_present` and `max_single_regime_share`; no per-regime AUC, calibration, selected return, benchmark excess, profit factor, drawdown, or minimum sessions are computed.
- **Remediation:** Calculate causal selected-policy metrics by frozen regime and require minimum independent sessions/trades per required regime. Gate worst-regime benchmark excess, drawdown, calibration, and a confidence interval or conservative loss bound. Version the regime label definition in model evidence.
- **Verification:** A synthetic candidate that is profitable overall but loses in one sufficiently populated required regime must fail. Sparse regimes must produce `insufficient_evidence`, not a pass.

### P1-4 - Probability evidence mixes calibrated and raw scores, and swing calibration is not promotion-gated

- **Location:** `src/market_predictor/swing/model.py:428-464`, `src/market_predictor/intraday/model.py:547-583`, `src/market_predictor/swing/model.py:240-250`, `src/market_predictor/swing/contracts.py:199-220`.
- **Impact:** The first chronological calibration chunk remains raw while later chunks are isotonic-calibrated, so aggregate Brier/ECE and threshold behavior mix different probability semantics. Swing promotion ignores its recorded Brier/log-loss entirely. API values described as probabilities may not correspond to observed event frequency, making fixed `0.55`/`0.65` action thresholds unreliable.
- **Evidence:** `_cross_fitted_calibration` initializes output with raw probability and cannot calibrate the first chunk because there are no prior OOF rows. Final serving uses one calibrator fit on all OOF predictions. Intraday gates Brier/ECE on the mixed OOF stream; swing records Brier/log-loss but has no calibration thresholds.
- **Remediation:** Within each outer fold, create a strictly earlier calibration partition or inner OOF predictions and exclude rows lacking disjoint calibration evidence from probability metrics/economics. Freeze calibration method before shadow evaluation. Add swing ECE/Brier, calibration intercept/slope, and reliability-bin gates.
- **Verification:** Every validation row must carry calibration-train cutoff and method. Assert cutoff precedes decision time. A monotonic score transformation that preserves AUC but produces materially biased probabilities must fail promotion.

### P1-5 - Severe live feature drift does not change readiness

- **Location:** `src/market_predictor/drift.py:16-44`, `src/market_predictor/drift.py:79-118`, `src/market_predictor/prediction_service.py:252-316`, `src/market_predictor/prediction_service.py:333-347`.
- **Impact:** Serving can remain globally `ready` and return valid predictions when drift is classified as `severe`. A model can continue operating outside its training support without a fail-closed or shadow-only response.
- **Evidence:** Health stores drift as a component, but the `ready` flag is never updated from drift status. Only model/features exceptions and memory change readiness. The reference profile is limited to global mean, standard deviation, and missingness, so it also misses distribution shape, interactions, score drift, and regime-specific drift.
- **Remediation:** Define versioned drift policy with warning, suppress/rank-only, and not-ready states. Compare rolling live distributions to time-of-day/regime/liquidity references, and add prediction-score, calibration, and matured-outcome drift. Make severe policy breaches affect route readiness.
- **Verification:** A severe-shift fixture must make the affected route not ready or explicitly non-actionable. Stable and warning fixtures must retain the declared behavior. Backfilled matured outcomes must trigger performance-drift gates on known degradation.

### P1-6 - Replay is an ad hoc investment calculator, not target-matured model validation

- **Location:** `src/market_predictor/investment_replay.py:72-120`, `src/market_predictor/investment_replay.py:223-283`, `src/market_predictor/telemetry.py:70-83`, `src/market_predictor/telemetry.py:90-120`.
- **Impact:** The system cannot yet measure whether live swing/intraday probabilities remain calibrated or economically useful at their declared horizons. User-selected evaluation times and forced entries produce outcomes that are not comparable with training labels, while runtime metrics retain only replay-status counts.
- **Evidence:** Replay exits at the last completed bar before arbitrary `evaluation_as_of`, not at the model horizon/policy exit. It emits one event with P&L and benchmark excess, but `snapshot()` stores only counts by status; no persistent cohort metrics, calibration, top-k economics, drawdown, or model-hash performance series exist.
- **Remediation:** Keep ad hoc replay separate. Add a deterministic outcome-maturation pipeline keyed by prediction snapshot, model/policy hash, exact label schema, decision time, entry rule, horizon, and benchmarks. Persist immutable outcomes and rolling performance by model, horizon, regime, sector, and calibration bin.
- **Verification:** Outcomes must remain pending before label availability, mature exactly once at the frozen exit, and reproduce offline label code. Aggregated live metrics must match recomputation from immutable snapshot/outcome records.

### P1-7 - Trainers do not enforce one frozen label/cost policy per dataset

- **Location:** `src/market_predictor/swing/dataset.py:143-145`, `src/market_predictor/swing/model.py:330-361`, `src/market_predictor/swing/audits.py:142-152`, `src/market_predictor/intraday/dataset.py:83-95`, `src/market_predictor/intraday/model.py:401-469`.
- **Impact:** A hash-verified file can still contain rows generated with mixed cost, target/stop, stride, or warm-up semantics. One estimator probability would then represent multiple outcomes, and the manifest would not provide a complete semantic contract for reproduction.
- **Evidence:** Builders stamp policy columns, but swing training verifies only one horizon and binary target; its audit does not recompute net return from gross return and declared cost. Intraday training verifies one horizon and decision interval but not uniform target ATR, stop ATR, cost, execution interval, or label-policy hash. Intraday model manifest extras do not persist the complete dataset/training configuration (`src/market_predictor/intraday/model.py:314-332`).
- **Remediation:** Add a canonical label-config JSON/hash and require exactly one config hash in every eligible dataset. Recompute audited labels from gross/path evidence and that config. Persist full dataset, label, split, calibration, selection, and execution-policy hashes in model and promotion evidence.
- **Verification:** Mixing rows from two cost or barrier configurations must fail dataset audit and training. Rebuilding from the persisted configs must reproduce label columns, selected rows, folds, and evidence hashes exactly.

### P1-8 - Canonical swing does not require point-in-time universe identity

- **Location:** `src/market_predictor/swing/dataset.py:24-45`, `src/market_predictor/swing/audits.py:9-41`, `src/market_predictor/swing/model.py:330-361`, `src/market_predictor/canonical/joins.py:53-120`.
- **Impact:** Swing training can accept a current-survivor universe with cap/liquidity/sector attributes but no `universe_snapshot_id` or effective membership identity. That allows survivorship and historical membership bias to enter otherwise hash-valid training evidence.
- **Evidence:** The intraday decision contract requires `universe_snapshot_id`; swing does not. Swing audit/training also omit snapshot identity. A robust point-in-time membership join exists, but the swing boundary does not prove it was used or bind its artifact hash to the model.
- **Remediation:** Require membership interval, availability time, snapshot ID, and universe artifact hash on every swing row. Audit exactly one known membership at each decision and retain inactive/delisted historical members. Bind universe hashes and coverage summaries to dataset/model/promotion evidence.
- **Verification:** A present-day-survivor-only fixture and a row without membership identity must fail. A known addition/removal history must include the removed constituent before removal, exclude it afterward, and preserve its historical labeled rows.

### P1-9 - Promotion row/trade minimums do not establish independent sample sufficiency

- **Location:** `src/market_predictor/swing/contracts.py:204-218`, `src/market_predictor/intraday/contracts.py:243-259`, `src/market_predictor/swing/evaluation.py:123-152`, `src/market_predictor/intraday/evaluation.py:179-205`.
- **Impact:** Tens of thousands of correlated cross-sectional rows or hundreds of trades from a small number of sessions can satisfy promotion. Effective sample size, session count, event independence, and uncertainty are not gated, so reported edge may be concentrated in a few days.
- **Evidence:** Promotions gate `validated_rows` and `selected_trades`, but not independent test sessions/periods, decision groups, effective sample size, minimum per-fold observations, or per-period confidence. Economics computes `periods`/`sessions`, but promoters do not gate those fields.
- **Remediation:** Gate minimum independent sessions, decision groups, non-overlapping events, per-fold rows/events, and effective sample size after overlap weighting. Require session-block uncertainty and concentration limits by date/ticker/sector.
- **Verification:** Duplicating rows within the same sessions must not improve eligibility or confidence. A large-row/few-session fixture must fail, while sufficiently distributed evidence must pass the independence gates.

## P2 Findings

### P2-1 - Top-decile lift is global rather than decision-group aware

- **Location:** `src/market_predictor/swing/evaluation.py:54-79`, `src/market_predictor/intraday/evaluation.py:71-99`, `src/market_predictor/swing/promotion.py:79-90`, `src/market_predictor/intraday/promotion.py:74-110`.
- **Impact:** Global score quantiles can reward time-varying score scale or base-rate shifts instead of within-decision candidate ordering. This metric is not the deployment selection rule, which chooses top-k within a decision group.
- **Evidence:** Lift uses one probability 90th percentile across all rows. Economics correctly groups top-k by `decision_group_id`, but promotion independently requires the global lift.
- **Remediation:** Replace or subordinate global lift with group-aware precision/lift/NDCG at the deployed `k`, reported by fold and session. Keep global AUC only as secondary diagnostic evidence.
- **Verification:** Rescaling all probabilities by session must not change the primary ranking metric. A model that ranks candidates randomly within each group must fail even if global score levels track market-wide base rates.

### P2-2 - Feature availability screening consults the full development period

- **Location:** `src/market_predictor/swing/model.py:364-382`, `src/market_predictor/intraday/model.py:472-487`.
- **Impact:** Feature inclusion can depend on missingness observed in future validation sessions. This is weaker than target leakage but makes the final feature set partially validation-informed and complicates honest experiment comparison.
- **Evidence:** Both selectors first compute non-null rate over all development rows, then check fold training subsets. Future values do not enter directly, but future availability determines whether a feature is considered at all.
- **Remediation:** Freeze the feature list before validation from schema and training-only readiness evidence, or derive it independently inside each fold and intersect only using predeclared rules that do not inspect validation periods.
- **Verification:** Altering feature missingness only in future test sessions must not change any earlier fold feature list, fit, or prediction.

### P2-3 - The deterministic ticker holdout is not stratified

- **Location:** `src/market_predictor/v3/validation.py:117-130`, `src/market_predictor/swing/model.py:88-94`, `src/market_predictor/intraday/model.py:88-94`.
- **Impact:** A hash-only holdout can underrepresent sectors, market-cap/liquidity buckets, rare targets, or event coverage. Aggregate holdout metrics may conceal weak transfer to important stock classes.
- **Evidence:** Tickers are sorted by a seeded SHA-256 score and sliced; no stratification or balance audit is applied.
- **Remediation:** Use a deterministic grouped/stratified assignment across sector, cap, liquidity, and label/event-coverage summaries, while preserving complete ticker isolation. Publish stratum coverage and metrics.
- **Verification:** Holdout construction must meet predeclared representation tolerances and remain deterministic. Missing required strata must fail readiness rather than silently produce aggregate evidence.

### P2-4 - Intraday overlap weights are an unvalidated overlap-count proxy

- **Location:** `src/market_predictor/intraday/labels.py:59-87`, `src/market_predictor/intraday/model.py:529-540`, `src/market_predictor/intraday/audits.py:181-189`.
- **Impact:** `1 / number_of_intervals_that_overlap_this_interval` is not the conventional average uniqueness over each bar of the label span. It can under- or over-weight staggered horizons and make nominal row counts look more informative than they are.
- **Evidence:** The implementation counts every interval with any overlap, assigns one reciprocal weight, and audits only positivity. It does not compare weights with per-minute concurrency or report weighted effective sample size.
- **Remediation:** Compute average uniqueness from per-bar concurrency over each exact label interval, or document and empirically validate the chosen proxy. Persist weighted effective sample size and sensitivity against unweighted and non-overlapping training.
- **Verification:** Hand-calculated staggered-interval fixtures must reproduce expected uniqueness. Weight changes must be covered by ablation and must not alter evaluation, which should remain strictly non-overlapping.

## Executive Verdict

The canonical data and model rebuild has substantially better causal primitives than the legacy research paths, but it is **not statistically safe to promote or serve a real trading model at commit `48ac758`**. Four P0 defects invalidate or disconnect promotion evidence: future-contaminated ticker holdout, unvalidated catalyst-modified serving, non-evidentiary news alignment audits, and promotion without untouched/uncertainty-controlled evidence.

This verdict is consistent with persisted reality: no canonical C4 swing or C5 intraday model has been trained and promoted from a real final dataset (`docs/production_ml_rebuild_plan.md:77`, `docs/production_ml_rebuild_plan.md:101`). Existing V3/V4 real-data candidates are correctly rejected and show negative cost-adjusted selected returns or statistically inconclusive improvements. Implementation-test success is not evidence of market edge.

## Confirmed Strengths

- Canonical bars expose interval end and availability timestamps, and point-in-time joins use backward availability rather than publication-date hindsight (`src/market_predictor/canonical/joins.py:122-163`).
- Canonical event contracts include publication, provider update, first seen, scoring, and feature availability; observed history rejects provider-publication proxy data (`src/market_predictor/canonical/audits.py:89-129`).
- Swing labels implement post-close decision, next-session-open entry, exact consecutive session path, benchmark-matched exits, and MFE/MAE (`src/market_predictor/swing/dataset.py:416-489`).
- Intraday labels use completed 5-minute decisions, next available one-minute open, exact one-minute paths, same-session enforcement, conservative same-bar stop priority, and exact benchmark intervals (`src/market_predictor/intraday/labels.py:90-173`, `src/market_predictor/intraday/labels.py:206-228`).
- The main walk-forward split is expanding, session-grouped, embargoed, and preserves decision groups (`src/market_predictor/v3/validation.py:59-114`).
- Intraday catalyst inputs are excluded from the numerical estimators by contract (`src/market_predictor/intraday/contracts.py:126-160`), although P0-2 shows that serving currently reintroduces an unvalidated adjustment.
- Canonical datasets and evidence bundles are hash-verified and fail closed on mutation (`src/market_predictor/canonical/store.py:28-110`, `src/market_predictor/swing/promotion.py:262-312`, `src/market_predictor/intraday/promotion.py:288-340`).
- Existing real-data model cards report negative results and reject candidates instead of lowering gates. O1 and V4-H1 include paired session-bootstrap intervals and do not open shadow data after development failure.
- Focused verification passed 67 tests covering canonical joins/audits, swing and intraday datasets/models, split behavior, catalyst serving, V3 evaluation, drift, replay, and prediction service. The initial `pytest` command did not run because the project virtual environment has no `pytest` module; sequential `unittest` groups were used instead.

## Missing Evidence and Unresolved Assumptions

- No real canonical swing or intraday candidate, promotion evidence bundle, calibration report, or untouched-shadow result exists.
- No production-quality historical observed-first-seen news corpus has been demonstrated across required sources and all eligible ticker/decision windows.
- No event-level reconciliation proves that every accepted news item is correctly ticker-mapped, relevant, deduplicated, and matched to its exact feature window.
- No final point-in-time universe audit proves historical inactive/delisted-member coverage for the intended production universe.
- No spread/quote/trade, halt, participation, or market-impact evidence supports the fixed 10 bps execution assumption for volatile names.
- No causal per-regime, sector, cap, liquidity, or catalyst/no-catalyst performance table exists for canonical C4/C5.
- No live target-matured outcome ledger, rolling calibration, benchmark-relative performance, or model-policy drift history exists.
- No effective-sample-size or power analysis supports the current promotion row/trade thresholds.

## Promotion Blockers

1. Replace the temporally contaminated ticker holdout with fold-local causal ticker evidence.
2. Disable catalyst changes to served rank/action policy until exact-policy evidence passes; preserve catalysts as context only meanwhile.
3. Replace hardcoded alignment zeros with immutable event-to-feature reconciliation.
4. Require one-time untouched shadow evidence, session-block lower confidence bounds, and paired baseline improvement.
5. Close P1 execution, regime, calibration, universe, semantic-config, sample-sufficiency, drift, and outcome-monitoring gaps.
6. Train and audit new real canonical candidates. Existing legacy/V3/V4 artifacts are not substitutes.

## Ordered Remediation Plan

1. **Make validation and serving identical:** remove heuristic catalyst rank/action adjustment from the serving policy and version the exact selection function.
2. **Repair causal evidence:** implement fold-local ticker holdout predictions, disjoint calibration, cutoff assertions, and independent-session accounting.
3. **Repair catalyst provenance:** enforce relevance/mapping states and generate event-level alignment/reconciliation evidence from immutable source lineage.
4. **Freeze semantic identities:** hash dataset, label, universe, split, calibration, execution, catalyst-policy, and serving-policy configurations end to end.
5. **Upgrade economics:** implement conservative executable fills, liquidity/capacity limits, stress costs, and correct capital/drawdown simulation.
6. **Upgrade promotion:** add per-regime gates, group-aware ranking metrics, effective sample requirements, paired baseline intervals, and a one-time shadow ledger.
7. **Upgrade live validation:** mature exact-horizon outcomes, persist model/policy performance cohorts, and make severe drift affect readiness.
8. **Rebuild from real canonical data:** run audits before training, train swing/intraday candidates, preserve failures, and promote only after every blocker and fresh-shadow gate passes.
