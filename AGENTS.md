# Market Predictor Engineering Covenant

This file is mandatory guidance for every human or automated coding agent working in
this repository. It exists to make the system converge. Read it before changing code.

## 1. Mission And Boundary

`market-predictor` produces evidence-backed prediction intelligence for swing and
intraday workflows. It owns data curation, feature construction, model training,
validation, prediction, outcome maturation, monitoring, and model governance.

It does not own alerts, broker execution, orders, positions, portfolio risk, final
position sizing, or user notification delivery. Those responsibilities belong to
`trading_flow`. Do not add them here.

The system is not yet deployed. Do not preserve obsolete implementations, aliases,
schemas, or compatibility layers unless a currently deployed consumer is identified
with reproducible evidence.

### 1.1 Working Persona

Act as the project's Chief Quant and Senior Machine Learning Engineer. Apply senior
judgment in quantitative research, market microstructure, causal feature engineering,
time-series validation, execution-cost modeling, production Python, data lineage, and
model governance. Treat claims as hypotheses requiring reproducible evidence; do not
manufacture confidence, profitability, or readiness.

This persona does not change the repository boundary. Build prediction intelligence
for daily and intraday horizons, not high-frequency execution infrastructure. Prefer
the repository's verified contracts and current stack over tools named in generic
guides. Challenge weak assumptions explicitly, explain technical failures in concrete
terms, and distinguish observed facts, statistical estimates, and design decisions.

Communicate directly and precisely. Do not hide a data, modeling, or economic failure
behind unexplained metrics or specialist terminology. State what failed, how it was
measured, what system behavior it affects, and what evidence would resolve it.

## 2. Source-Of-Truth Order

When instructions disagree, use this order:

1. Current user instruction.
2. This `AGENTS.md`.
3. The single active execution plan:
   `docs/active_edge_rebuild_plan.md`.
4. The companion current handoff:
   `docs/reviews/active_edge_rebuild_handoff.md`.
5. The current feature acceptance audit:
   `docs/reviews/feature_engineering_audit_20260801.md`.
6. Current code contracts and tests.
7. `docs/catalyst_confirmation_architecture.md` and the implementation guide.
8. Historical evidence and rejected model cards.
9. Chat history.

Never infer current state from chat alone. Inspect the branch, working tree, recent
commits, tests, and authoritative documents first.

## 3. Convergence Rules

### 3.1 Freeze Before Building

Every substantial checkpoint must have:

- one bounded problem statement;
- explicit in-scope and out-of-scope behavior;
- named contracts or invariants affected;
- measurable exit gates;
- a verification plan;
- a rollback or fail-closed behavior.

Do not start a broad implementation while these remain ambiguous.

### 3.2 One Canonical Path

There must be one production implementation for each concern. Tests may use explicit
test-only fixtures, but production must not retain parallel legacy paths, fallback
models, shadow schemas, or alternate policy logic.

Shared semantics such as prediction selection, label evaluation, execution costs,
identity hashing, and readiness must have one source of truth.

### 3.3 Completed Means Closed

A checkpoint is closed when its exit gates pass, documentation is current, and its
commit is pushed. Do not reopen a closed checkpoint for preference, style, speculative
hardening, or a repeated general review.

A closed checkpoint may be reopened only for one of:

- a reproducible failing test or production observation;
- a demonstrated security vulnerability;
- new data proving a stated invariant false;
- a changed user requirement;
- a concrete contract conflict with a later checkpoint.

Every reopening must record the new evidence, affected invariant, blast radius, and
new exit test. "A deeper review may find more" is not sufficient.

### 3.4 Review Once, Consolidate Once

For each checkpoint:

1. Review the design before implementation.
2. Implement and run focused tests.
3. Run one consolidated code/ML review against the frozen exit gates.
4. Fix supported findings.
5. Run the full verification battery and close the checkpoint.

Independent reviewers may be used, but duplicate or speculative findings must be
deduplicated into one evidence-backed list. Do not chain unlimited reviews that each
restart the architecture.

Every accepted finding must include:

- severity;
- exact file/contract;
- reproducible evidence or a failing test;
- user or system impact;
- the smallest correct remediation;
- a verification test.

### 3.5 Control Blast Radius

Prefer the smallest change that restores the violated invariant. A review finding in
monitoring does not authorize redesigning training, serving, promotion, and deployment.

Before editing more than one subsystem, state why each subsystem must change. Avoid
opportunistic cleanup. Put unrelated cleanup in a later checkpoint.

### 3.6 No Deferred Correctness

Do not add TODO behavior, permissive fallbacks, fake production evidence, placeholder
passes, or "temporary" schema shortcuts. Either implement the required invariant now
or mark the checkpoint blocked/environment-pending with the exact missing evidence.

External facts that cannot be proven locally, such as Azure identity, Blob permissions,
live market behavior, real shadow performance, or container execution on unavailable
infrastructure, must remain `environment_pending`. Never simulate them into a pass.

### 3.7 Two-Document Continuity

There must be exactly two current continuity documents:

1. `docs/active_edge_rebuild_plan.md` is the only active execution plan. It records
   the ordered checkpoints, frozen scope, status, exit gates, and completed evidence.
2. `docs/reviews/active_edge_rebuild_handoff.md` is the only current continuation
   handoff. It records the actual branch state, last completed implementation commit,
   authoritative artifacts, blockers, exact next step, files to read, and verification
   commands.

Older dated plans and handoffs are historical evidence. Do not create another active
plan or handoff while these files exist. Replace them only when the entire edge-rebuild
program is closed or explicitly superseded by the user.

At every checkpoint boundary:

1. Read both current documents before changing code.
2. Mark only the current plan step `in_progress`; exactly one step may be in progress.
3. Implement and verify only that step.
4. Commit and push the implementation checkpoint.
5. Update the active plan with factual status, evidence, and the implementation commit.
6. Rewrite the handoff so a new LLM can resume without chat history.
7. Commit and push the documentation closure before starting the next step.

The handoff must never claim unrun tests, unverified artifacts, model profitability, or
external readiness. If interrupted before the implementation commit, record the dirty
files and exact unfinished command instead of marking the step complete.

### 3.8 Documentation And Audit Retention

Keep the repository documentation small and authoritative:

1. Update the active plan, handoff, and feature audit in place. Do not create a dated
   replacement for current state or leave two documents claiming to be current.
2. At every checkpoint, consolidate useful conclusions from temporary reviews into the
   active handoff or feature audit, then delete the superseded narrative review.
3. Delete generated reports, rejected-model outputs, and incomplete materializations
   after confirming that no source, test, manifest, model card, or active document
   references them and that they are reproducible from protected inputs.
4. Retain immutable evidence only when a governance contract, content hash, accepted
   model card, or reproducibility test depends on it. Label retained historical evidence
   as historical so it cannot be mistaken for active work.
5. Never delete raw provider data, canonical authorities, accepted model bundles, or
   hash-bound evidence as routine cleanup. Their removal requires an explicit retention
   decision recorded in the active handoff.
6. Before committing documentation cleanup, run a repository reference scan and the
   affected governance tests. Broken links or dangling evidence paths are failures.

The current continuity set is exactly the documents named in section 3.7. The current
feature audit is `docs/reviews/feature_engineering_audit_20260801.md`; replace its
contents in place until the active program closes.

### 3.9 Vertical Feature Completion

A feature is not implemented when only its calculation exists. It is complete only
when one evidence chain covers every applicable layer:

1. source collection and source-coverage semantics;
2. immutable causal authority and availability timestamp;
3. historical batch construction;
4. live construction from the same transformation;
5. ordered model-feature contract and manifest hash;
6. training, validation, promotion, and serving consumption;
7. API representation with unavailable values kept distinct from zero;
8. batch/live parity, future-poison, missingness, and artifact-tamper tests.

Before training, maintain one acceptance matrix for every proposed feature group with
those layers marked `verified`, `not_applicable`, or `blocked` and linked to concrete
artifacts/tests. `not_applicable` requires a recorded design reason, such as an
intraday catalyst being an overlay rather than an estimator input. Any other missing
layer blocks training and serving. Do not describe calculation-only, batch-only, or
API-only work as complete.

Source families are part of the model contract. A model may consume only the exact
source set represented throughout its training history. Additional live sources may
be exposed as separately evaluated overlays, but they may not silently change model
inputs. Missing optional sources remain unavailable; missing required sources force
abstention.

### 3.10 Frozen Swing Data Contract

- The first permitted swing decision date is 2019-07-09.
- Bars before 2019-07-09 are warm-up history only. They may initialize causal
  indicators but may not produce decision features, labels, training rows,
  validation rows, or test rows.
- Missing catalyst coverage before 2019-07-09 is not a readiness failure because no
  modeled decision may exist in that period.
- Ticker catalyst sources are direct-issuer Alpaca news and SEC issuer filing events.
  Each source requires reproducible provenance and an explicit availability timestamp.
- Verified open-source global and sector events are separate context overlays. They
  must not be converted into ticker catalyst.
- Reddit and Seeking Alpha are retired and prohibited. Do not collect, materialize,
  feature, train, serve, or document them as available dependencies.
- No model training may begin until all required causal source, feature, and label
  authorities pass coverage validation, immutable replay, and lineage verification.

## 4. Mandatory ML And Trading Invariants

### 4.1 Point-In-Time Causality

- Every feature must have an explicit availability timestamp.
- A feature may be used only when availability is at or before the decision cutoff.
- News publication time, first-observed time, revisions, and source coverage are
  different facts; never substitute one for another.
- Historical backfills without historical first-observed evidence are research-only.
- Corporate actions, ticker mapping, delistings, membership, and benchmark membership
  must be point-in-time.

### 4.2 News And Catalyst Integrity

- Ticker news must be demonstrably relevant to the security, not merely sector-related.
- Global and sector news must enter through explicit market/sector context, not through
  false ticker assignment.
- News and candle alignment must use exchange sessions and exact decision cutoffs.
- After-close, premarket, market-hours, and post-market events are separate regimes.
- Missing, stale, duplicate, or mismatched event evidence fails readiness; it is not
  filled with neutral sentiment.

### 4.3 Labels And Economics

- Labels must be reproduced from immutable source bars through the shared evaluator.
- Swing entry/exit and intraday target/stop/timeout semantics must remain exact.
- Benchmark returns must use the same executable interval as stock returns.
- Costs are applied exactly once.
- Missing path bars, ambiguous fills, and unsupported feed coverage fail closed.
- SPY/QQQ/sector comparisons are evidence, not optional presentation fields.

### 4.4 Validation And Selection

- Use purged, embargoed, time-ordered validation; random cross-validation is prohibited.
- Ticker holdout and temporal holdout answer different questions and remain separate.
- Calibration evidence must precede scored rows.
- Promotion metrics must evaluate the exact production ranking and selection policy.
- Intraday opportunity and downside models are evaluated separately and jointly.
- Catalyst may be an overlay unless causal ablation proves it improves the estimator.
- Optimize for calibrated, cost-adjusted, benchmark-relative economics and controlled
  drawdown, not ROC AUC alone.

### 4.5 Live Monitoring

- Monitor rolling selected/actionable cohorts, not lifetime population averages.
- Track opportunity and downside calibration, score/rank distributions, selection rate,
  economics, drawdown, pending outcomes, and last-matured freshness.
- Monitoring evidence must bind release, model, feature-source set, prediction policy,
  label policy, execution policy, cohort, and source-row identities.
- Severe, stale, insufficient, tampered, or identity-mismatched evidence cannot
  authorize actionable output.

### 4.6 Promotion And Serving

- Candidate and baseline artifacts and shadow workload are frozen before observation.
- Shadow economics come only from immutable row-level predictions and matured outcomes.
- Operator-authored aggregate returns are prohibited.
- Build and approval require distinct authenticated principals.
- Serving loads only a verified promoted release; no candidate or legacy fallback.
- One response must come from one coherent verified serving identity.

## 5. Data And Operational Rules

- Each ticker/source download fails independently; one failure must not corrupt others.
- Raw, canonical, feature, model, report, and release artifacts are separate layers.
- Every persisted production artifact is immutable or atomically replaced and
  content-addressed.
- Secrets belong in environment variables or managed secret stores, never code,
  command arguments, logs, reports, fixtures, screenshots, or Git.
- Provider limits are configuration and telemetry, not hard-coded assumptions.
- Price/volume features that require consolidated volume must fail when the feed is not
  SIP/consolidated.
- Keep swing candidate training below 5 GiB process memory. Intraday and serving
  workloads retain 4 GiB limits. Use bounded batches, projected columns, `float32`
  matrices where appropriate, and sequential model release.
- Do not leave Python workers or test servers running after verification.
- Do not run multiple heavy test/training processes concurrently.
- Every heavy CLI build or training entry point must acquire the shared
  non-queueing workspace lease before loading inputs. Do not bypass the lease
  from scripts, tests, notebooks, or deployment wrappers.

## 6. Change Workflow

For every checkpoint:

1. Inspect `git status`, recent commits, current tests, and authoritative docs.
2. Preserve unrelated user changes. Never reset or revert them.
3. Write or update the failing/poison test before or with the fix.
4. Change contracts before consumers and consumers before documentation.
5. Update the vertical feature acceptance matrix before model training.
6. Run focused tests after each subsystem.
7. Run repository-wide Ruff and strict mypy.
8. Run the complete unit test suite once after the final code change.
9. Check `git diff --check`, process state, and memory.
10. Update README, architecture, implementation guide, and handoff only where behavior
   actually changed.
11. Commit and push the implementation, then close the two continuity documents before
    starting the next checkpoint.

Do not mix unfinished work from different checkpoints in one commit. If the working
tree already contains paused work, stage only the files belonging to the current
checkpoint.

## 7. Definition Of Done

A code checkpoint is complete only when:

- the frozen exit gates are satisfied;
- focused and poison tests pass;
- the full suite passes after the last code change;
- repository-wide Ruff and strict mypy pass;
- documentation describes actual behavior and current limitations;
- no secret or generated credential is present;
- no worker remains running;
- the worktree scope is understood;
- the checkpoint is committed and pushed.

Passing tests do not prove model profitability or production readiness. Real-data
promotion, prospective shadow evidence, execution calibration, and external deployment
evidence remain separate gates.

## 8. Stop And Escalate

Stop implementation and ask for a decision when:

- a new request conflicts with a frozen strategy or repository boundary;
- correctness requires unavailable credentials, external infrastructure, or market data;
- a proposed fix would invalidate multiple closed checkpoints;
- unrelated working-tree changes make safe isolation impossible;
- the only available approach would weaken fail-closed behavior.

When escalating, state the exact blocker, evidence, affected checkpoint, and smallest
decision needed. Do not replace uncertainty with an assumption.

## 9. Current Deferral

R7.8 security, container, CI, Azure deployment, rollback, and disaster-recovery evidence
is explicitly deferred by the user. Do not start it unless the user reactivates it.
