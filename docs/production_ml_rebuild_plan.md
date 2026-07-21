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

Status: complete on 2026-07-21 (C4 checkpoint).

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

Status: next; C4 data/registry interfaces are complete.

- Build completed-bar 1-minute and 5-minute features with exact next-bar entry semantics.
- Require consecutive paths, session boundaries, SIP volume, warm-up depth, and exact benchmark bars.
- Keep catalyst/news as a separately measured overlay until it proves incremental value.
- Train opportunity and downside models separately.
- Evaluate top-k cost-adjusted excess return, turnover, drawdown, calibration, and unseen-ticker stability.
- Do not promote the currently rejected V3/V4 research artifacts.

### C6 - Deployment, Observability, And Resource Controls

Status: pending promoted C4/C5 artifacts.

- Connect canonical feature publication to registered serving routes.
- Add bounded workers, column projection, `float32` training matrices, fold-model release, and 4 GiB hard memory guards everywhere.
- Add source, freshness, audit, model, latency, drift, and prediction-quality telemetry.
- Deploy API and scheduled jobs independently on Azure Container Apps unless measured GPU inference justifies a dedicated worker.
- Add rollback to the prior promoted manifest without mutable artifact replacement.

### C7 - Repository Cleanup And Release Audit

Status: pending C6.

- Remove superseded builders, adapters, commands, configuration, tests, and docs.
- Resolve repository-wide lint and type debt.
- Verify secret scanning, dependency locking, container health, startup failure modes, and disaster recovery.
- Run complete tests, replay tests, artifact-integrity tests, and deployment smoke tests.
- Tag the production-ready release only when both serving and data pipelines fail closed.

## Checkpoint Policy

Each component is committed only after its own tests, audit evidence, documentation, and resource checks pass. Later components may change a completed contract only through a new version and migration; they may not silently weaken an earlier gate.
