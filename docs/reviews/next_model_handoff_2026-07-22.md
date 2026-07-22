# Market Predictor Production Remediation Handoff

Date: 2026-07-22  
Repository: `C:\project\market-predictor`  
Branch: `main`  
Clean checkpoint: `9cf7b66 Freeze production prediction semantics`

## Objective

Finish every validated finding in:

- `docs/reviews/ml_statistical_validity_review_2026-07-22.md`
- `docs/reviews/ml_production_architecture_review_2026-07-22.md`
- `docs/reviews/remediation_plan_2026-07-22.md`

The system produces prediction intelligence for swing and intraday workflows. It does not own alerts, orders, positions, risk, or execution; those remain TradingFlow responsibilities.

## Non-Negotiable Constraints

- No backward compatibility is required before first production deployment. Remove superseded paths instead of preserving unsafe legacy behavior.
- Keep total Python working-set memory below 4 GiB. Use bounded fixtures and one heavy test/training process at a time.
- All training, validation, promotion, replay, and serving paths must be point-in-time and hash-reproducible.
- Serving selection and action semantics must exactly match the policy evaluated for promotion.
- Catalyst/news is explanation and confirmation metadata for intraday until a separately versioned overlay passes causal ablation and promotion.
- Do not weaken gates to obtain a passing model. A failed real candidate remains failed evidence.
- Do not inspect, print, or commit `.env` or credentials. Rotate the RapidAPI and Finviz secrets previously pasted into chat before production.
- Create a verified git checkpoint after each checkpoint below.

## Verified Completed State

### Review Baseline

- `6851d4c Record senior ML remediation baseline`
- Review inventory: 4 statistical P0, 9 statistical P1, 4 statistical P2; 2 architecture P0, 11 architecture P1, 7 architecture P2.

### R1: Frozen Time and Serving Semantics

- `2764ea5 Fix nightly prediction cutoff semantics`
- `9cf7b66 Freeze production prediction semantics`
- Swing nightly cutoff is the versioned XNYS 18:00 America/New_York policy.
- Daily bar availability is separate from prediction cutoff.
- Live swing requires exact cutoff identity, audited daily bar count, SIP/consolidated feed identity, and real source coverage watermarks.
- `60m` is canonical; `1h` is accepted only as an input alias.
- Catalyst does not modify intraday/swing model rank or action.
- API failures are typed and opaque, with correlation IDs.
- Prediction responses/snapshots contain policy, cutoff, feature, model, source-watermark, horizon, and identity evidence.
- Snapshot schema is content-addressed, but durable Blob persistence is still R5.
- Full verification at this checkpoint: 219 tests passed; scoped Ruff and strict mypy passed.

### R2 Causal Core

- `56aac9d Make model validation causal`
- Deterministic stratified ticker holdout is constructed from the first causal training window.
- Holdout rows are scored only in matching chronological outer-fold test sessions.
- Labels must mature before each outer-fold test cutoff.
- Calibration is fit strictly on earlier evidence; seed folds without valid calibration are excluded.
- Training-only feature selection, row/fold hashes, cutoff evidence, and future-mutation poison tests are present.
- Final full-data fit occurs only after validation.

## Immediate Next Checkpoint: R2 Honest Evaluation and Economics

Implement in this order so contracts settle before promotion logic:

1. Add `src/market_predictor/prediction_policy.py` with immutable policy IDs and hashes.
2. Make intraday serving and evaluation call the same score/selection function. Current intended score is opportunity probability multiplied by one minus downside probability. Swing ranks by model probability only.
3. Replace global-percentile lift with decision-group-aware top-k lift/ranking metrics.
4. Compute exact intraday average uniqueness from per-bar concurrency over each label interval. Persist weighted effective sample size; do not filter evaluation rows by training weights.
5. Add a versioned execution policy: conservative gap-through fills, spread/slippage by liquidity/price/volatility, halt/no-fill handling, participation caps, cost stress, and capacity curves.
6. Simulate allocation-aware portfolio equity and drawdown. Do not sum unconstrained overlapping trade returns.
7. Add per-regime selected-policy return, benchmark excess, drawdown, calibration, independent sessions, and evidence status. Sparse regimes must be `insufficient_evidence`.
8. Add swing ECE/Brier/calibration slope/intercept gates; exclude all uncalibrated seed rows from probability/economic evidence.
9. Gate independent decision groups, independent sessions, effective sample size, minimum fold evidence, base/stress economics, liquidity/capacity, worst-regime economics, and calibration.
10. Require the causal evidence fields created in `56aac9d`; missing cutoff, split, feature-schema, row, fold, or calibration identities must reject promotion.

Suggested write slices:

- Shared policy: `prediction_policy.py`, `prediction_service.py`.
- Swing: `swing/evaluation.py`, `swing/contracts.py`, `swing/promotion.py`.
- Intraday: `intraday/evaluation.py`, `intraday/contracts.py`, `intraday/promotion.py`.
- Execution/uniqueness: new `execution_policy.py`, `intraday/dataset.py`, `intraday/model.py`; apply the same policy semantics to swing economics.

Required tests:

- OOF rows replayed through the production policy produce identical scores, ranks, and selected actions.
- Hand-calculated staggered intervals reproduce exact average uniqueness.
- Gap-through stop fills at the worse executable open, not the barrier.
- Positive point return with non-positive stress economics fails.
- Overall profit with a sufficiently populated losing regime fails.
- Sparse regime reports insufficient evidence, never pass.
- Biased probabilities preserving AUC fail calibration gates.
- Too few sessions/effective samples fails even with many cross-sectional rows.

Checkpoint only after full tests, Ruff, strict mypy, and `git diff --check` pass.

## R3: Event, Universe, Label, and Collection Lineage

1. Add canonical event relevance state: `validated`, `irrelevant`, or `unknown`.
2. Remove every `fillna(1.0)` relevance default. Unknown direct-ticker events are excluded from counts/sentiment and separately counted.
3. Validate Reddit ticker linkage at post level using cashtags/company aliases plus a false-positive stoplist. Preserve the match evidence and policy hash.
4. Produce event-level reconciliation: every accepted event ID must be matched or explicitly rejected for duplicate, wrong ticker, unavailable/future, unknown relevance, irrelevant, or outside-window reasons. Zero unexplained rows is mandatory.
5. Replace hardcoded alignment zeros in both model trainers with summaries derived from that reconciliation artifact. Promotion verifies the artifact hash and component arithmetic.
6. Add canonical label-config JSON/hash and execution-policy hash to every eligible row and artifact. Mixed configs must fail.
7. Require universe snapshot ID, membership interval, membership availability, and universe artifact hash on every swing row. Add a canonical symbol master for aliases/delistings/corporate actions.
8. Make raw collection runs atomic: staged write, validation, manifest, then publish. Partial/failed symbols remain isolated and explicit.
9. Add quota reservation and cache locking for concurrent collectors.

Important starting points: `canonical/contracts.py`, `canonical/normalize.py`, `canonical/joins.py`, `canonical/audits.py`, `swing/dataset.py`, `intraday/dataset.py`, both model trainers, Reddit source modules, and canonical store manifests.

## R4: Trustworthy Promotion and Release

1. Candidate creation cannot set `promoted` status.
2. Add immutable promotion attestation binding candidate, evidence manifest, dataset, universe, label, split, calibration, execution, catalyst, serving policy, baseline, and gate-config hashes plus build/approver identity.
3. Add predeclared hypothesis/baseline registry.
4. Add immutable untouched-shadow bundles and a one-time-use shadow fingerprint ledger.
5. Require session-block paired confidence lower bounds above zero for benchmark-relative selected-policy improvement.
6. Failed/consumed shadow evidence cannot be reused by the same hypothesis family.
7. Publish versioned local release directories and atomically switch one active pointer only after all files and attestation verify.
8. Add Azure Blob ETag/lease compare-and-swap activation, rollback, and race tests.

## R5: Bounded Serving, Durable Outcomes, and Drift

1. Load one active model context and cache it; do not deserialize models per request.
2. Add bounded concurrency/admission control and verify RSS below 4 GiB under concurrent unified requests.
3. Persist immutable prediction snapshots and matured outcomes to Blob, keyed by snapshot/model/policy/label identity.
4. Separate ad hoc investment replay from deterministic label-horizon outcome maturation.
5. Persist calibration/economic cohorts by model, horizon, regime, sector, and calibration bin.
6. Add versioned feature, score, regime, calibration, and outcome-drift policies. Severe drift makes the affected route non-actionable/not ready.
7. Add authentication, authorization scopes, rate limits, request size limits, structured logs, traces, and metrics.
8. Keep TradingFlow integration read-only/observe-only until real promotion and shadow evidence pass.

## R6: Repository and Delivery Hardening

1. Remove generic legacy promotion and obsolete experimental runtime paths.
2. Split production, collection, and research CLI surfaces.
3. Lock Python dependencies with hashes and separate runtime/training/research extras.
4. Pin the container base image by digest; produce SBOM and vulnerability/license reports.
5. Add CI for full tests, Ruff, strict mypy, package/import boundaries, secret scanning, dependency scanning, container scanning, and deterministic artifact tests.
6. Replace secrets with workload identity/Key Vault references and least-privilege roles.
7. Complete telemetry, SLOs, backup/restore, runbooks, and DR evidence.

## R7: Final Verification and Evidence Status

Run, at minimum:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\ruff.exe check src tests
.\.venv\Scripts\mypy.exe --strict src\market_predictor
git diff --check
```

Then run focused mutation, concurrency, memory, release-race, rollback, shadow-ledger, and reproducibility tests.

Do not claim these as complete without real external evidence:

- Fresh canonical swing/intraday model passing every promotion gate.
- Untouched real shadow interval with positive confidence lower bounds.
- Azure publish/sync/rollback/DR rehearsal using actual identity, private ingress, storage, and ETag/lease controls.
- Demonstrated live source history quality and matured-outcome performance.

Mark those items `environment_pending` rather than simulating a pass.

## Resume Commands

```powershell
Set-Location C:\project\market-predictor
git status --short
git log --oneline -5
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

Expected starting state is clean at `9cf7b66`, with no Python worker processes.
