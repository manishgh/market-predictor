# R7 Independent Re-Audit and Remediation Plan

Date: 2026-07-24  
Audited code: `r3-lineage` at `9bf8e2b`  
Current delivery checkpoint: use the branch tip; R6 CI portability repairs followed the
audited commit without changing model semantics.

## Decision

R1-R6 materially improved causal evaluation, artifact integrity, bounded serving, outcome
maturation, authentication, and reproducible delivery. The repository is not yet eligible
for a real-model production promotion. The fresh R7 review found three promotion blockers
and several high-impact statistical and release gaps.

No gate may be weakened to make a candidate pass. Azure remains excluded until explicitly
integrated. Missing real shadow, execution-cost, live outcome, and deployment evidence
remains `environment_pending`.

## Ordered Remediation

### R7.1 - Exact Prediction And Selection Policy

1. Replace the global partial policy hash with a strict content-addressed policy contract.
2. Bind swing `top_k` and intraday `top_k`, downside ceiling, and per-session cap.
3. Persist the complete policy payload and hash in model payloads, manifests, evidence,
   releases, prediction snapshots, and matured outcomes.
4. Make offline OOF/holdout evaluation and serving ranking call the same pure selection
   implementation.
5. Add policy mutation tests and exact offline/serving parity tests.

Exit gate: changing any material selection parameter changes the policy hash; a model or
response with an absent/mismatched policy fails closed.

Status: implemented and locally verified. The shared policy excludes non-finite and
non-ready rows before deterministic selection, applies the bound swing/intraday top-k,
intraday downside ceiling, and per-session cap, and persists rank/eligibility/selection
identity into maturation intents. Serving selects over the complete published decision
cross-section before returning requested symbols. Independently trained view policies are
recorded per view and content-addressed as one unified serving bundle. Policy mutation,
offline selection, serving selection, promotion rejection, and contract-invariant tests
are included.

### R7.2 - Honest Statistical And Economic Evidence

1. Report and gate on actual scored folds, excluding skipped and calibration-seed folds.
2. Replace scale-invariant Kish ESS on uniqueness weights with an overlap-aware evidence
   measure based on summed uniqueness and independent sessions/events.
3. Compute benchmark excess from gross return minus one execution cost minus benchmark
   return; prohibit double cost subtraction.
4. Require sufficient evidence for every predeclared required regime and nonnegative
   confidence bounds for populated regimes.
5. Apply capacity/no-fill behavior inside the exact phase/session selection simulation for
   both validation scopes; missing required liquidity evidence fails closed.

Exit gate: focused skipped-fold, overlap, sparse-regime, losing-regime, cost identity, and
capacity tests pass.

Status: implemented and locally verified. The first checkpoint gates `validation_folds` on the distinct
fold IDs whose audit status is `included`, retains the configured split count only as
diagnostic evidence, and records the exact scored IDs. Benchmark excess is now computed
as gross return minus exactly one execution cost minus the matched raw benchmark return;
the frozen-label fallback subtracts only the incremental stress surcharge. Hand-calculated
swing/intraday identities and a configured-versus-scored promotion regression pass.
Intraday evidence now reports summed label uniqueness and independently non-overlapping
event count, uses their minimum as effective evidence, gates development and ticker
holdout scopes, and retains independent sessions as a separate gate. Uniformly reducing
all uniqueness weights now reduces effective evidence. Swing requires risk-on, neutral,
and risk-off evidence; intraday additionally requires high-volatility evidence. Every
required regime is emitted even when absent, with explicit session/trade thresholds.
Deterministic session-block bootstrap lower bounds are computed inside each
non-overlapping phase, conservatively combined across phases, and promotion rejects
missing/thin required regimes or populated regimes whose return or SPY-excess lower bound
misses its configured threshold. Each scored fold now combines seen and held-out tickers
into one full decision cross-section before economic top-k selection. Classification
diagnostics remain separated, while profitability records the full portfolio and
post-selection seen/unseen attribution whose trade counts must reconcile exactly.
Intraday capacity now consumes that exact full selected stream, preserves every
execution-policy capital level, applies participation-scaled costs and no-fill rules,
and persists the complete curve in signed metrics. Promotion reproduces selected/fill
counts, no-fill rates, capital levels, minimum net return, and maximum no-fill rate;
missing dollar volume, price, ATR, gross return, or raw matched benchmark paths fail
closed. R7.2 is implemented and locally verified; fresh real candidate evidence is
still required before promotion.

### R7.3 - Event-To-Feature Reconciliation

1. Persist exact event-to-decision/window assignments.
2. Reconcile assigned event counts against every material aggregate feature row/window.
3. Remove hard-coded zero missing-feature counters.
4. Fail canonical production readiness and promotion on missing, extra, wrong-ticker,
   duplicate, or unexplained assignments.

Exit gate: deleting or miscounting one historically eligible event causes reconciliation
and promotion failure.

Status: implemented and locally verified. Canonical builds now persist exact
`event_assignment.v1` rows, rebuild all material event aggregates solely from
those rows, publish the assignment artifact with an integrity manifest, and bind
assignment/aggregate hashes into both model families and promotion
attestations. Alignment audits consume real missing-row and mismatch counters.
Deletion, duplication, wrong-ticker, wrong-window, and aggregate-mutation poison
tests fail closed. Repository verification at this checkpoint: 328 unit tests,
Ruff, strict mypy across 129 source files, and compileall all pass. R7.4 is next.

### R7.4 - Source-Path Label Reproduction

1. Recompute audited swing labels, costs, benchmark returns, and availability from immutable
   OHLCV paths and the bound policy.
2. Replay intraday barriers from exact one-minute paths, including missing-entry, gap,
   halt, timeout, and ambiguous-bar cases.
3. Share pure label evaluators between dataset construction, audit, and outcome maturation.

Exit gate: internally plausible but source-inconsistent labels are rejected.

### R7.5 - Causally Untouched Shadow Evidence

1. Freeze candidate and baseline artifacts before the first shadow decision.
2. Require training-data and maximum label-availability cutoffs before the shadow interval.
3. Derive paired session returns from immutable row-level predictions, outcomes, exact
   selection policy, and execution policy; reject caller-supplied aggregate returns.
4. Bind replay inputs and outputs into the one-use shadow ledger and promotion attestation.

Exit gate: a candidate trained through any shadow decision or a poisoned aggregate return
cannot be attested.

### R7.6 - Live Selected-Policy Monitoring

1. Monitor rolling selected/actionable cohorts rather than all predictions.
2. Add opportunity and downside calibration, score/rank distribution, selection rate,
   economic outcomes, and last-matured-outcome freshness.
3. Bind each report to release, policy, feature, label, execution, and cohort identities.
4. Severe, stale, insufficient, or mismatched required evidence remains non-actionable.

Exit gate: degrading only selected rows, downside calibration, or recent outcomes suppresses
the affected route.

### R7.7 - Atomic Serving Bundle

1. Publish model, calibration, policy, feature schema, and compatible feature-release
   identity as one immutable serving bundle.
2. Verify from one opened snapshot or content-addressed local copy to remove verify/read
   races.
3. Suppress every unified row when swing/intraday or model/feature identities conflict.
4. Add barrier-controlled activation, rollback, and mixed-release tests.

Exit gate: a response is produced from one verified bundle identity or is non-actionable.

### R7.8 - Delivery And Deployment Closure

1. Keep the container smoke test authenticated and add a ready-state authenticated
   prediction fixture.
2. Verify each committed dependency lock can launch its documented command surface.
3. Bind promotion approval to independently authenticated CI/OIDC and approver evidence.
4. Add external metrics/tracing and shared rate-limit state before multi-replica deployment.
5. Add OIDC/JWKS discovery and bounded key refresh.

Blob durability, OCI publication, multi-replica hydration, Azure private ingress, managed
identity, Key Vault, and DR remain excluded implementation work until Azure integration is
approved; they are not simulated as passing evidence.

## Current Verified Evidence

- 322 repository tests pass locally.
- The focused R7 trust/race/rollback/memory/idempotency suite passes 37 tests.
- Repository-wide Ruff, strict mypy on Windows and Linux targets, and compile checks pass.
- Dependency locks regenerate deterministically and the production dependency audit reports
  no known vulnerabilities.
- Git history secret scanning passes.
- CI run `30087411847` passes secret scanning, repository validation, container build,
  startup, liveness, fail-closed readiness, and memory checks. The production-container
  job remains failed at the strict critical-vulnerability Trivy gate; no scanner gate was
  weakened. Linux container evidence is not accepted until that gate passes.

## Environment-Pending Evidence

- Fresh canonical swing and intraday candidates passing all corrected promotion gates.
- Untouched real shadow interval with positive paired confidence lower bounds.
- Real spread, slippage, impact, participation, no-fill, and capacity calibration.
- Live observed-first-seen event history and matured selected-policy outcomes.
- Real combined-model burst/reload soak below the 4 GiB container ceiling.
- Azure publish, hydration, failover, rollback, backup, and DR rehearsal.
