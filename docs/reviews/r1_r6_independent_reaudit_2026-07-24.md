# R1-R6 Independent Re-Audit

Date: 2026-07-24  
Audited committed state: `63cfbcc`  
Review method: two independent read-only reviews, one quantitative ML and one
production architecture/security/code-quality. Uncommitted R7 work was excluded from
their evidence.

## Decision

R1-R6 established useful causal, registry, release, API, and delivery foundations, but
the committed state was not production-promotable. The independent review confirms one
open P0 temporal-leakage risk, one open P0 shadow-evidence authenticity risk, and multiple
P1 statistical and operational gaps.

R7.1 checkpoint `e18fd32` addresses the committed policy-parity finding by binding the
complete per-model prediction policy, selecting over the full published cross-section,
sharing offline/serving selection code, and persisting eligibility/selection identity.
It does not close the other findings below.

## P0 Findings

### 1. Intraday peer cross-sections can include unavailable peer bars

Canonical decision time follows ticker-specific availability in
`src/market_predictor/canonical/joins.py:62`, but intraday grouping is replaced by nominal
bar end in `src/market_predictor/intraday/dataset.py:339`. Breadth and rank features in
`src/market_predictor/intraday/dataset.py:491` and
`src/market_predictor/intraday/dataset.py:674` aggregate the nominal group without proving
every peer bar was available by the scored row's decision time.

Impact: a faster ticker can use slower peer information that arrived later. Row-local
future-feature audits do not detect peer leakage.

Required remediation:

1. Define one immutable cross-section cutoff or build each row's peer set strictly as-of
   its decision time.
2. Persist contributing peer availability evidence.
3. Add a staggered-availability poison test proving later peer mutations cannot alter an
   earlier row.

Status: implemented and locally verified after the audit in `intraday.features.v2`;
checkpoint `9f28da1`.

### 2. Shadow promotion authenticates supplied returns, not their derivation

`src/market_predictor/shadow_ledger.py:24` accepts caller-created candidate/baseline
session returns. `src/market_predictor/promotion_workflow.py:141` verifies hashes and
timing but does not reproduce returns from predictions, selected membership, bars, fills,
and benchmarks. Build, approver, and baseline identities are also assertions rather than
independently authenticated artifacts.

Impact: plausible but fabricated or accidentally incorrect aggregate returns can be
signed and attested.

Required remediation:

1. Freeze candidate and baseline artifacts before the first shadow decision.
2. Generate row-level predictions through the production policy.
3. Recompute paired returns from immutable canonical paths and the execution policy.
4. Bind workload/approver OIDC identities, baseline artifact, gate config, source rows,
   and ledger inclusion proof into the attestation.

Status: remediated locally in R7.5. Candidate/baseline artifacts and exact decision
groups are frozen before observation; paired source rows and session returns are derived
from the outcome repository and independently reproduced at promotion. Operator-authored
aggregate returns are no longer accepted. Build and approver identities require separate
RS256 OIDC tokens, roles, principals, issuer/audience validation, and a deployment-owned
JWKS; stable token/claims evidence is bound into the ledger transaction and attestation.

## P1 ML Findings

### 3. Benchmark excess subtracts cost twice

Swing and intraday labels already subtract a frozen cost before benchmark return in
`src/market_predictor/swing/dataset.py:467` and
`src/market_predictor/intraday/labels.py:185`. Evaluation subtracts dynamic execution cost
again in `src/market_predictor/swing/evaluation.py:124` and
`src/market_predictor/intraday/evaluation.py:340`.

Fix: compute `gross return - one execution cost - matched benchmark return` directly and
cover zero, flat, dynamic, and stressed costs with hand-calculated tests.

Status: implemented and locally verified in the first R7.2 checkpoint. Swing and
intraday evaluation now derive benchmark excess from gross return, one execution cost,
and the raw matched benchmark path. The frozen-label fallback subtracts only the
incremental stress surcharge. Hand-calculated identity tests cover both timeframes.

### 4. Event reconciliation does not reproduce event aggregates

`src/market_predictor/canonical/reconciliation.py:35` checks only whether an event could
be near a decision. Build/training mappings do not prove exact event-to-window assignment,
and missing historical feature rows are hard-coded to zero.

Fix: persist exact assignments and independently reproduce count, relevance, sentiment,
source diversity, and latest-availability aggregates. Poison one event and require
promotion rejection.

Status: implemented and locally verified in R7.3. The canonical build emits a
content-addressed event-assignment artifact, independently reproduces count,
weighted sentiment, sentiment coverage, relevance quality, source
counts/diversity, and latest availability, and binds assignment plus aggregate
hashes into training and promotion. Poison tests cover deletion, duplication,
wrong ticker, wrong window, and aggregate mutation.

### 5. Fold-count evidence includes unscored seed folds

Calibration seed folds are excluded from scoring, but metrics report configured
`n_splits`; promotion gates the inflated count.

Fix: persist and gate actual included fold IDs with per-fold rows, sessions, and economics.

Status: implemented and locally verified in the first R7.2 checkpoint. Promotion now
consumes the distinct included/scored fold count; configured splits are retained under a
separate diagnostic field and cannot satisfy the gate.

### 6. Live drift is population/lifetime rather than selected/rolling

Committed maturation and performance reporting omit selected membership, rank, signal,
and intraday downside probability, then aggregate lifetime overall cohorts.

Fix: monitor rolling selected-policy cohorts, opportunity/downside calibration, rank and
selection-rate drift, economics, drawdown, and outcome freshness.

Status: policy identity added in R7.1; monitoring remains open in R7.6.

### 7. Unseen-ticker economics are not reconstructed over the full universe

Development and held-out tickers are evaluated as separate top-k universes. Neither is
the deployed full cross-section.

Fix: reconstruct each fold's full test-time cross-section, select once, and report
full-universe economics plus seen/unseen attribution.

Status: implemented and locally verified in R7.2. Scored seen and held-out rows are
combined by fold into the deployed full decision cross-section before top-k selection.
Profitability reports the full portfolio plus post-selection seen/unseen attribution,
and tests require attributed trade counts to reconcile to the full selection.

## P1 Production Findings

### 8. Feature publication and reading are not one atomic generation

Data and manifests are replaced separately, and readers hash then reopen paths in
`src/market_predictor/feature_store.py:44`,
`src/market_predictor/canonical/store.py:57`, and
`src/market_predictor/prediction_service.py:1270`.

Fix: immutable content-addressed generation directories plus one atomic pointer; read and
verify one immutable generation. Add writer/reader race and crash-injection tests.

Status: open; R7.7.

### 9. Drift warm-up can deadlock a new release

Warming state is rank-only, but serving rejects every state other than actionable.
Without served snapshots, the release cannot generate outcomes to leave warming.

Fix: allow identity-complete rank-only responses with actionable output suppressed, and
persist their outcomes. Reserve unavailable/503 for genuinely not-ready state.

Status: open; R7.6.

### 10. Authentication and request-size controls have fail-open edges

Injected services can default auth to disabled, unknown `/v1` paths lack a default scope,
and chunked bodies are fully buffered before the 64 KiB check in
`src/market_predictor/api.py:73`, `src/market_predictor/api.py:139`, and
`src/market_predictor/api.py:423`.

Fix: protect `/v1/*` by default with explicit probe exceptions, derive production security
independently of DI, and enforce body limits in the ASGI receive stream and ingress.

Status: deferred to the post-R7 delivery and deployment program.

### 11. Model replacement and feature reads do not bound peak memory safely

The old context is dropped before unbounded deserialization; feature files are read in
full; estimates do not use cgroup limits.

Fix: artifact byte/row limits, predicate/projection pushdown, cgroup-aware accounting, and
preload/functional validation before an RCU-style context swap that retains rollback.

Status: model/feature swap work remains in R7.7; deployment-scale memory evidence
is deferred to the post-R7 delivery program.

### 12. Outcome registration is not a durable closed loop

Prediction snapshot persistence and maturation-intent registration are separate actions.
Acknowledged predictions can therefore never reach monitoring.

Fix: atomically publish a durable outbox entry with every snapshot; add a recovering
scanner, complete idempotency comparison, and pending-age/error metrics.

Status: open; R7.6.

### 13. Local operational state is single-node only

Releases, snapshots, outcomes, drift, rate limits, and metrics are local/process state.

Fix: either enforce one replica with persistent volume, backup, and tested RPO/RTO, or add
CAS-capable shared release state and a transactional durable outcome/outbox store.

Status: environment pending; no distributed/cloud claim.

## P2 Findings

1. Add threshold-local calibration slope/intercept/bias gates for both intraday targets.
2. Treat execution coefficients as rejection proxies until quote/fill/no-fill evidence is
   calibrated.
3. Add signed revocation and append-only activation history.
4. Add a fully ready authenticated container prediction/restart smoke test.
5. Remove candidate overwrite and publish unique staged candidate directories atomically.
6. Make request errors redact provider response bodies and move production signing to a
   managed KMS/HSM identity.
7. Consolidate duplicated JSON/hash/atomic-write utilities and make fixed-input build
   identity reproducible.
8. Replace monolithic Typer command copying with explicit per-surface registration.

## Verified Strengths

- Label availability is filtered before each test decision.
- Cross-fold calibration uses earlier matured OOF evidence.
- Swing next-open paths and intraday consecutive one-minute barrier paths are materially
  stronger than the legacy labels.
- Point-in-time universe identity, observed event availability, immutable label policy,
  evidence manifests, signed attestations, and fail-closed release loading are useful
  foundations.
- Existing V3/V4 cards honestly reject candidates with negative or statistically
  unsupported economics; they do not claim a production edge.

## Ordered Follow-Up

1. Fix intraday cross-sectional peer availability and add the poison audit.
2. Complete R7.2 statistical/economic corrections.
3. Complete R7.3 exact event reconciliation.
4. R7.4 source-path label reproduction completed locally; rebuild real
   candidates through the new reconciliation path.
5. Complete R7.5 causally derived shadow evidence.
6. Complete R7.6 selected-policy monitoring and durable outbox.
7. Complete R7.7 atomic feature/model serving generations.

Security hardening, container/CI delivery closure, Azure deployment, rollback,
and disaster-recovery evidence are retained as a separate deferred post-R7
program.
