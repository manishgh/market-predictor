# Production ML Rebuild Plan

This plan is separate from the completed V3 research checkpoints. It converts the repository into a fail-closed production prediction system without retaining compatibility paths that are not used in production.

## Constraints

- `market-predictor` produces prediction intelligence only. `trading_flow` owns alerts, strategies, risk, portfolio state, and execution.
- Production routes load one server-registered promoted model per mode and horizon.
- Candidate and research artifacts never substitute for missing production artifacts.
- Historical data is point-in-time or explicitly research-only.
- Every component must be independently testable, auditable, and deployable.
- Training and validation processes must stay below the configured 4 GiB working-set limit.

## Component Sequence

### C1 - Production Serving Boundary

Status: complete on 2026-07-21 (`92efcbb`).

- Enforce server-owned routes and promoted, hash-matching artifacts.
- Separate liveness from readiness.
- Reject stale or schema-incompatible feature snapshots.
- Persist immutable prediction evidence.
- Keep catalyst assessment outside estimator probability.

### C2 - Remove Unused Legacy Paths

Status: complete on 2026-07-21 (`2becc3a`).

- Remove multi-model probability averaging, heuristic prediction paths, predictor alerts, automatic legacy retraining, and obsolete schedulers.
- Remove candidate fallback from production serving.
- Retain only research commands with an explicit current purpose.
- Do not preserve CLI or artifact compatibility for removed behavior.

### C3 - Canonical Point-In-Time Data Boundary

Status: complete on 2026-07-21 (`5ebfa83`).

- Normalize left-edge bars into explicit interval-end and availability timestamps using the XNYS calendar.
- Record event publication, provider update, first-seen, sentiment-scoring, and final feature availability.
- Record source attempts as observed, observed-empty, partial, failed, disabled, or not-collected.
- Require exactly one effective and already-known universe membership per decision.
- Join versioned fundamental facts by availability; reject current-snapshot historical backfill.
- Enforce SIP provenance for production volume features.
- Perform strict as-of joins and reject future features.
- Publish immutable Parquet artifacts with SHA-256 manifests and audit evidence.
- Remove current Seeking Alpha and SEC snapshot fields from historical model defaults.

Exit evidence:

- Repository test suite passes.
- Strict typing passes on the complete C3 production surface.
- Scoped Ruff and source compilation pass.
- Canonical CLI commands and required membership input are visible in `--help`.

### C4 - Canonical Swing Model Rebuild

Status: complete on 2026-07-21 (`53c51c2`).

- One post-close decision predicts net-positive return from the next session open through the fifth session close.
- Daily technical, SPY/QQQ/sector-relative, catalyst, observed global-context, membership, and as-of fundamental features are frozen as `swing.features.v1`.
- Feature eligibility requires at least 250 daily bars, SIP/all-adjusted bars, exact benchmark coverage, fresh successful source coverage, and the configured cross section.
- Labels retain gross/net return, SPY/QQQ/sector excess return, MFE/MAE, exact consecutive-session path, entry, exit, and label-availability timestamps.
- Training supports logistic and histogram-gradient-boosting baselines with `float32` matrices and a hard 4 GiB process-memory gate.
- Validation uses horizon-purged expanding walk-forward folds, cross-fitted calibration, and deterministic unseen-ticker holdout.
- Promotion requires classification, unseen-ticker lift, conservative phase economics, drawdown, regime, catalyst, alignment, memory, and run-provenance gates.
- Training evidence is hash inventoried and bound to the candidate artifact; modified evidence is rejected before promotion.
- Production serving accepts only `canonical_swing` / `swing.model.v1` promoted artifacts and explicit feature-availability timestamps.

Exit evidence:

- Focused canonical swing, CLI, evidence-integrity, promotion, global-context, and serving tests pass.
- Strict typing passes on the C4 production surface.
- Scoped Ruff passes.
- Legacy volatile build/train/score commands and volatile production serving were removed.

No canonical C4 model has been trained or promoted from the real production dataset yet. The configured swing route intentionally remains not-ready until a candidate passes the frozen gates.

### C5 - Canonical Intraday Model Rebuild

Status: complete on 2026-07-21 (`a844fdd`).

- Build completed-bar 1-minute and 5-minute features with exact next-bar entry semantics.
- Require consecutive paths, session boundaries, SIP volume, warm-up depth, and exact benchmark bars.
- Keep catalyst/news as a separately measured overlay until it proves incremental value.
- Train opportunity and downside models separately.
- Evaluate top-k cost-adjusted excess return, turnover, drawdown, calibration, and unseen-ticker stability.
- Do not promote the currently rejected V3/V4 research artifacts.

Exit evidence:

- Canonical `intraday.features.v2` and atomic `intraday.model.v1` contracts replace the old serving scorer. V2 delays each nominal 5-minute cross-section to one cutoff at or after every peer and benchmark availability timestamp.
- Completed 5-minute decisions use the latest fully available 1-minute state and exact subsequent 1-minute entry/path labels; missing ticker or benchmark intervals fail closed.
- Opportunity and downside estimators use session-purged walk-forward folds, deterministic unseen-ticker holdout, cross-fitted calibration, overlap weights, and non-overlapping top-k economics.
- Catalyst/news remains outside both estimators and is measured as a confirmation/ranking overlay.
- Dataset construction and training enforce configurable 4 GiB memory guards.
- Candidate and all promotion evidence are hash-bound to one model run; promotion is atomic and fail-closed.
- Focused dataset, model, evidence-integrity, promotion, serving, and architecture tests pass.
- Strict typing, scoped Ruff, source compilation, and the full repository suite pass.

No canonical C5 model has been trained or promoted from the real production dataset yet. No intraday production route is registered until one candidate passes every frozen gate.

### C6 - Deployment, Observability, And Resource Controls

Status: implementation complete 2026-07-22; real C4/C5 model promotion remains intentionally independent.

- `build-swing-live-features` and `build-intraday-live-features` reuse canonical feature engineering but emit one latest, label-free, audited inference cross section.
- `publish-live-features` accepts only the matching hash-verified canonical inference artifact and atomically records source identity, schema, feed, columns, timestamps, and hash.
- Training uses projected inputs, `float32` matrices, sequential fold-model release, and configurable 4 GiB guards. API readiness and metrics enforce/report the same configurable process ceiling.
- Readiness and `/v1/metrics` expose source/freshness/schema identity, model hashes, request errors/latency, drift, prediction readiness, replay outcomes, and memory.
- Serving publication creates immutable content-addressed model-plus-feature releases; assets precede the release manifest and active pointer. Sync verifies all bytes before local activation, and rollback accepts only a complete prior release.
- The container runs the API as non-root, has a liveness probe, and can fail-closed while hydrating the active Azure release before API import.

C6 tests use synthetic promoted artifacts and an in-memory Blob protocol. A real Azure identity/network deployment rehearsal and disaster-recovery exercise remain C7 acceptance work. No canonical model is promoted merely because the deployment path exists.

C6 checkpoint verification: 198 repository tests pass; Ruff and strict mypy pass on all C6 production modules; source/test compilation and credential scanning pass; the final process audit measured 0.182 GiB against the 4 GiB hard budget. The C7 cleanup baseline was 48 repository-wide Ruff findings and 13 strict-mypy findings in older research/source modules; the cleanup checkpoint below resolves that debt.

### C7 - Repository Cleanup And Release Audit

Status: repository lint/type cleanup complete on 2026-07-22; release rehearsal remains pending.

- Remove superseded builders, adapters, commands, configuration, tests, and docs.
- Resolve repository-wide lint and type debt.
- Verify secret scanning, dependency locking, container health, startup failure modes, and disaster recovery.
- Run complete tests, replay tests, artifact-integrity tests, and deployment smoke tests.
- Tag the production-ready release only when both serving and data pipelines fail closed.

Cleanup checkpoint evidence:

- Repository-wide Ruff passes with no ignored cleanup baseline.
- Strict mypy passes all 93 source files; narrow missing-stub overrides remain only for external libraries without usable type information.
- All 199 repository tests pass, including replay, artifact-integrity, deployment, canonical swing, and canonical intraday tests.
- Source and test compilation, whitespace validation, and supplied-credential scanning pass.
- Final process audit found no resident Python process, leaving the full 4 GiB working-set budget available.

Remaining C7 acceptance work:

- Lock production dependencies and verify a clean-environment installation from that lock.
- Exercise Azure publication, startup synchronization, rollback, and identity/network failure paths against the real deployment environment.
- Verify the built container health and fail-closed startup behavior in that environment.
- Create a production release tag only after those deployment and disaster-recovery checks pass.

## Checkpoint Policy

Each component is committed only after its own tests, audit evidence, documentation, and resource checks pass. Later components may change a completed contract only through a new version and migration; they may not silently weaken an earlier gate.
