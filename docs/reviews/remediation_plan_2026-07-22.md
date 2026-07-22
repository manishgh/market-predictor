# Senior Review Remediation Plan - 2026-07-22

## Objective

Close every validated finding in the statistical/ML and production-architecture reviews at base commit `48ac758`. A finding is closed only when the implementation, focused regression tests, repository-wide verification, and durable evidence agree. Test success does not establish predictive edge, and no candidate may be promoted until the statistical promotion blockers are closed and fresh real-data evidence passes.

## Non-Negotiable Constraints

- Market Predictor remains prediction-only. TradingFlow owns alerts, risk, portfolio state, orders, and execution.
- All training, validation, promotion, replay, and serving behavior must be point-in-time and reproducible from immutable identities.
- The serving policy must be exactly the policy evaluated for promotion.
- Unknown event relevance is not positive relevance.
- Runtime memory must remain below 4 GiB and overload must fail before scoring.
- Production credentials must not appear in command arguments, logs, artifacts, or reports.
- Legacy compatibility is not a requirement. Noncanonical production paths should be removed rather than maintained.

## Checkpoints

### R0 - Review Baseline

- Preserve both independent review reports and the reviewer handoff.
- Record the base commit and this finding-to-checkpoint map.

Findings: all, as immutable source evidence.

### R1 - Correct Prediction Semantics

- Preserve audited swing `daily_bar_count` through live serving.
- Introduce an explicit, calendar-aware swing prediction cutoff used consistently by collection, feature availability, labels, snapshots, and replay.
- Canonicalize intraday wire horizon to `60m`.
- Remove unvalidated catalyst changes from rank/action decisions; retain catalyst as context only.
- Version the serving policy and include its hash in evidence and snapshots.
- Add typed, non-leaking API errors and stable correlation identity.

Findings: production P0-1, P0-2, P1-1, P1-2, P2-1; statistical P0-2.

### R2 - Causal Validation and Honest Economics

- Replace all-date ticker holdout scoring with fold-local chronological ticker evidence.
- Make calibration evidence strictly earlier than each scored row and gate swing calibration.
- Freeze feature availability from training-only evidence.
- Stratify deterministic ticker holdout and publish stratum coverage.
- Use group-aware ranking metrics and correct intraday average-uniqueness weights.
- Add independent-session/effective-sample gates and per-regime economic/calibration gates.
- Add executable gap-through fills, liquidity/participation limits, cost stress, capacity curves, and allocation-aware drawdown.

Findings: statistical P0-1, P1-2, P1-3, P1-4, P1-9, P2-1, P2-2, P2-3, P2-4.

### R3 - Event, Universe, and Dataset Lineage

- Represent event relevance as validated, irrelevant, or unknown; unknown direct events are excluded.
- Validate Reddit post-level ticker linkage with cashtag/company-alias evidence and a false-positive stoplist.
- Persist event-level matching lineage and reconcile every accepted event to a matched or explicitly rejected outcome.
- Replace hardcoded news/candle audit zeros with computed evidence and fail promotion on unexplained rows.
- Require point-in-time universe snapshot identity on swing rows.
- Freeze one label/execution configuration hash per dataset and bind all semantic hashes to model evidence.
- Enforce one canonical security identity and provider-only symbol conversion.
- Publish immutable raw collection runs with atomic manifests.
- Make provider quota reservation/cache publication concurrent, atomic, and single-flight.

Findings: statistical P0-3, P1-1, P1-7, P1-8; production P1-8, P1-9, P1-10, P2-2.

### R4 - Promotion, Release, and Shadow Governance

- Add immutable promotion attestations; candidate manifests cannot self-declare promotion.
- Add a predeclared hypothesis/baseline registry, immutable shadow bundle, one-time-use shadow ledger, and session-block confidence gates.
- Bind model, dataset, universe, label, split, calibration, execution, catalyst, serving policy, baseline, and gate hashes.
- Install releases into immutable local directories and switch one atomic active pointer.
- Add Azure conditional activation using lease or ETag compare-and-swap plus append-only activation events.

Findings: statistical P0-4; production P1-4, P1-5, P2-3.

### R5 - Durable and Bounded Production Serving

- Cache one verified immutable model context per active release.
- Add inference admission control, bounded ticker batches, memory headroom rejection, and one-process deployment policy.
- Implement a durable immutable prediction/outcome repository with Blob storage and a bounded local cache.
- Separate ad hoc investment replay from deterministic target-maturation replay.
- Persist live calibration/economics cohorts and make severe drift suppress actionability/readiness.
- Add service authentication/authorization, rate limits, request limits, and separately authorized replay.
- Export durable metrics/traces with release, model, feature, and correlation identities.

Findings: statistical P1-5, P1-6; production P1-3, P1-6, P1-7, P2-4.

### R6 - Reproducible Build and Operational Surface

- Use secret wrappers and prohibit production secrets in CLI arguments.
- Split production, collection, and research command surfaces; remove generic legacy promotion.
- Split serving/training dependencies and generate reviewed hash-locked sets.
- Pin the container base image by digest and produce SBOM/vulnerability evidence.
- Expand CI to repository-wide Ruff, strict mypy, full tests, lock verification, image build/startup/readiness, and security checks.

Findings: production P1-11, P2-5, P2-6, P2-7.

### R7 - Verification and Re-Audit

- Run all unit and contract tests, repository-wide Ruff and strict mypy, compile checks, credential scans, and architecture-boundary tests.
- Run bounded concurrent quota/cache tests and serving burst/soak tests under the 4 GiB ceiling.
- Exercise immutable release switch and conflict behavior.
- Run a fresh independent statistical and production re-audit against the final commit.
- Preserve unresolved real-environment evidence explicitly: real Azure rehearsal, private ingress/identity, and fresh canonical model promotion require actual infrastructure/data and cannot be simulated into a pass.

## Closure Rule

Each original finding must appear in a final closure table with one of three states:

- `closed`: implementation and verification evidence exist.
- `environment_pending`: code is complete but a real external rehearsal or fresh market dataset is required.
- `open`: remediation is incomplete; deployment and model promotion remain blocked.

No finding may be marked closed solely by documentation or mocked infrastructure.
