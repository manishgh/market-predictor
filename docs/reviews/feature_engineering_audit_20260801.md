# Current Feature Engineering Audit

Last updated: 2026-08-15

## Scope

Active edge-rebuild swing and intraday paths only. Retired model paths are not
accepted as evidence. The audit covers feature availability, label causality,
missing-data behavior, temporal validation, estimator inputs, and experiment
control.

## Intraday

### Passed controls

- Features are computed from completed volume bars and exact one-minute market,
  SPY, QQQ, and point-in-time sector context.
- Rolling technical state resets each session. Overnight observations cannot
  enter RSI, ATR, EMA, realized-volatility, OBV, or efficiency windows.
- A row is rejected when exact decision-time context is absent. No previous or
  future minute is substituted.
- Entry is the exact next one-minute open. Target, stop, timeout, SPY, QQQ, and sector
  returns use the same executable interval. Intraday label schema V2 abstains when any
  exact benchmark interval is unavailable.
- Feature availability is at or before decision time; label availability is
  strictly after decision time and after the completed outcome path.
- Training partitions are ordered by exchange session, use an overnight
  embargo, and purge any training session whose labels are not available before
  the next partition.
- The locked temporal test is opened once after validation-only candidate
  selection. A deterministic security holdout supplies separate unseen-symbol
  evidence.
- The immutable dataset contains 4,173,230 rows, including 1,410,447 eligible
  rows. Its prior authority and partition hashes verify; it predates label schema V2
  and cannot be consumed by the V2-label trainer without causal rematerialization.

### Corrected finding

The initial trainer grid contained seven learned candidates while the frozen
experiment budget allowed six. That run was stopped before publication. Commit
`febd2d5` enforces the budget in code and reduces the grid to five learned
candidates: two logistic models, two histogram-gradient-boosting models, and
one ranking model. The deterministic score remains a baseline and is not a
learned candidate.

### Current feature profile

- Intraday schema V2 replaces price-scale ATR with normalized ATR and adds
  activation relative volume, normalized volume overshoot, volume-bar duration,
  minutes since activation, and session progress.
- The same shared feature builder serves historical batch and live decisions.
  Tests reject future evidence, missing exact benchmark context, stale decisions,
  and any mismatch in the exact 44 estimator features.
- News/catalyst remains outside the intraday entry estimator. It is a separately
  hash-bound confirmation, explanation, and ranking overlay because the earlier
  direct-feature ablation reduced validation quality.
- Intraday V2 is published and replayable but economically rejected after costs.
  It is not serveable. The later V3 branch declared five cross-sectional z-score
  columns without a valid contemporaneous decision-cohort implementation; the
  columns were undefined for asynchronous or single-member timestamps. Commit
  `e168482` removes that invalid contract. Any artifact requiring those columns is
  rejected lineage and cannot train, replay, promote, or serve. A future
  cross-sectional feature group must use one verified batch/live decision cohort
  and be backfilled for the complete intraday model horizon before acceptance.
- A4.1 collection now keeps every selected stock-session and records full-session bar
  coverage only as source metadata. Historical feature eligibility uses data through
  the decision cutoff; label eligibility uses only the managed outcome horizon. A
  later missing bar cannot remove an earlier live-equivalent decision.
- Bar-only and microstructure-enhanced profiles require an immutable matched ablation
  cohort with identical decisions, labels, folds, costs, and benchmark intervals.
  Missing microstructure makes only the enhanced row unavailable.
- Raw quote transport preserves finite nonnegative zero-sided quote states. A4.2, not
  transport, must classify valid two-sided duration, observed-zero states, and
  unavailable quote coverage.
- A5.1 now publishes a strict causal event-cohort preflight. The two Alpaca parent
  authorities contain 17,401 broker-action episodes. Correct exact-ticker and
  CIK-compatible namespace reconciliation produces 1,912 unique attached episodes /
  83,636 event-decision pairs; the earlier 19/862 result was an identity-matching
  defect. All historical source timing remains retrospective
  `provider_publication_proxy`, so production eligibility and locked-test reads remain
  zero.

## Swing

### Passed controls

- Decisions follow completed daily bars and labels enter at the next session
  open.
- Technical features include momentum, trend, pullback, volume, SPY/sector
  relative return, and residual return.
- Daily warm-up is at least 250 sessions. Cross-sectional transforms use only
  same-session eligible securities, are winsorized, and include rank, z-score,
  and sector-relative forms.
- Barrier collisions are resolved stop-first after executable overnight-gap handling:
  stop gaps fill at the worse open and target gaps use the conservative resting-limit
  target. Return labels include the frozen round-trip cost.
- Promotion comparisons use holding-aligned SPY, QQQ, and sector returns from entry
  open through the managed exit session close. Fixed-ten-session returns remain
  diagnostics because daily bars cannot identify an intraday benchmark exit instant.
- The governed split uses explicit dates with XNYS-verified counts, never percentages:
  1,231-session initial fit from `2019-07-09` through `2024-05-28`, 10-session
  validation embargo, 252-session validation, expanding 1,493-session final refit over
  every post-cutoff development session, 10-session final embargo, and the 251-session
  locked test from `2025-07-01` through `2026-06-30`. This is approximately 4.9 years
  initial fit plus one validation year plus one locked-test year; the causal-news cutoff
  is authoritative.
- Temporal generalization uses the full future point-in-time cross-section. Stable 20%
  unseen-security generalization is a held-out-symbol stress test in the same time
  window, fitted separately; both scopes must pass, but they are not independent time
  samples.
- Commit `7b61873` removes the invalid training-time catalyst cohort rewrite. The
  trainer no longer attaches SEC files from a local path, fills unknown SEC coverage
  with zero, recomputes rank labels, or bypasses sector constraints. Commit `cb2aba5`
  then separates the A2 technical baseline from the future A3 event-driven family.
- Swing and intraday evaluations now publish explicit binary diagnostics for the
  estimator target, positive after-cost stock return, and positive SPY/QQQ/sector
  excess return. A deterministic 64-repeat shuffled-label AUC control must remain at
  chance; abnormal discrimination fails evaluation. Single-class scopes are recorded
  as unavailable rather than misreported as an AUC.

### Current implementation and result

- The base materializer now publishes one catalyst-independent `technical_market`
  population. A separate A3.4 authority publishes exact matched technical-only,
  analyst-revision-only, and combined event-conditioned profiles.
- Sparse missing daily sessions now invalidate only affected 250-session warm-up
  windows and 10-session labels. They are never imputed or bridged. The 5%
  whole-security exclusion rule remains unchanged and applies only to genuinely
  unusable full histories.
- The V12 technical authority contains 853,417 rows, 604 modeled securities, and
  1,759 sessions. The first A3.4 join was defective because old event decision hashes
  were compared directly with rebuilt technical-panel hashes. Corrected A3.4 contains
  27,087 matched prediction rows from 11,720 unique latest broker announcements in
  each of three exact comparison datasets. Blocked families are absent and unknown
  source coverage abstains.
- Monthly profile partitions physically isolate locked-test outcomes. Development
  training loads only requested months and projected columns; locked outcomes remain
  unopened unless all validation gates pass. Replay verifies profile, decision and
  security identities, session bounds/counts, canonical paths, and hashes.
- Candidate v2 trained six governed logistic and histogram-gradient-boosting models.
  Diagnostic AUC reached approximately 0.55-0.57, but every candidate failed at
  least one calendar, portfolio-daily, doubled-cost, or holding-aligned benchmark
  confidence gate across temporal and unseen-security validation. The result is an
  immutable `no_candidate`; the locked test was not read.
- The swing training process remained below its 5 GiB hard memory limit.
- The A2 replacement baseline uses four nested technical groups: momentum/volatility,
  trend confirmation, pullback timing, and volume/liquidity. One regularized logistic
  candidate is evaluated per group; the full group also receives one XGBoost ranker
  and regressor. Selection remains economic and validation-only; AUC is diagnostic.
- Current Finviz snapshots do not establish point-in-time quality, profitability,
  investment, valuation, or estimate-revision history. Those feature groups are
  blocked rather than copied backward or represented as zero.
- Every fitted estimator persists its exact ordered feature subset. Serving and the
  research API slice to that subset and reject missing, duplicate, out-of-order, or
  out-of-bundle columns. No real A2 candidate has been trained yet.

## Vertical acceptance matrix

`Verified code` means implementation plus focused tests. It does not mean a real
immutable artifact exists. `Blocked` means training or serving is prohibited.

| Capability | Source and immutable authority | Batch path | Live path | Model contract | Training / promotion / serving | API | Status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Intraday technical estimator | SIP/all bar and V2 authorities verify | Verified | Verified; parity and staleness rejection | Exact 44-feature schema verified; incomplete V3 z-scores removed | V2 economically rejected; later z-score artifact invalidated | Blocked until promotion | Rejected |
| Intraday ticker-catalyst specialist | Corrected A5.1 historical preflight binds 1,912 proxy-time episodes / 83,636 event-decision pairs; prospective authorities remain the production source | Upgrade and downgrade directional cohorts trained as technical confirmation filters; coverage failed capacity | Observed-time raw/revision/coverage path verified; causal SIP outcome horizon remains incomplete | Frozen 24-hour direct-issuer broker-action cohort; unknown/stale coverage abstains; catalyst is not a direct estimator feature | Four directional experiments are `no_candidate`; A6 remains prohibited | Not serveable | Historical research rejected; prospective evidence pending |
| Intraday global overlay | GDELT collector and global authority verified in code; no production collection published | Not an estimator input | Observed-time collection, immutable query policy, and unknown/zero behavior verified | Separate global overlay contract verified | Ranking use blocked until runtime authority exists | Blocked until promoted bundle and orchestrator exist | Runtime artifact pending |
| Swing technical estimator | Daily SIP/all, point-in-time membership, and V12 panel verify | Verified | Verified; latest closed session required | A2 nested technical schema and per-model subsets verified | Replacement trainer complete; new candidate not run | Blocked until promotion | Implementation ready |
| Swing ticker-catalyst estimator | Two immutable V2 Alpaca issuer-event authorities cover `2019-07-09` through `2026-07-08`; strict replay verified | Corrected A3.4 publishes 27,087 prediction rows / 11,720 unique latest broker announcements per comparison dataset | No event specialist is live | Rating-change and coverage cohorts passed capacity; price-target/generic remains report-only | Twelve A3.5 experiments failed inner AUC and canonical portfolio gates; outer validation and locked test unopened | Blocked until a new preregistered strategy version passes | Development rejected |
| Swing global overlay | Global collector and decision authority verified in code | Separate overlay; never attached as ticker news | Verified code | Separate global authority hash and source policy | Cannot rescue or alter a rejected estimator; ranking use requires complete authority | Blocked until promoted bundle and orchestrator exist | Runtime artifact pending |

## Training decision

Technical data readiness does not block the A2 baseline. Candidate v2 was trained
correctly and rejected because its out-of-sample economic edge was not stable, not
because data was missing. The replacement trainer is verified but has not produced a
new statistical result. A3 causal event authorities now exist for the complete
development horizon. Event-family precision review currently admits only broker rating
actions in both eras; every other family remains blocked. Corrected A3.4 provides
27,087 eligible prediction rows from 11,720 unique latest announcements per comparison
dataset. A3.5 separated directional rating changes from coverage initiation and kept
price-target/generic actions report-only. Both specialists had sufficient chronological
capacity, but all 12 development experiments failed the frozen inner-selection and
canonical economic gates. Outer validation and the locked test stayed unopened;
global context remains a separate overlay.

A5.1 first found an identity defect and then corrected it without weakening CIK checks.
The corrected historical authority attaches 1,912 episodes. Research-only directional
development admitted 805 upgrades and 860 downgrades; 245 coverage initiations failed
the frozen capacity floors. All four eligible subtype/hypothesis experiments failed
discrimination and economic gates and emitted `no_candidate`. Historical publication
timestamps still cannot provide production first-observed evidence. A5 production
evaluation therefore remains blocked until the prospective event and SIP outcome
horizon is causally complete.
Commit `5530246` supplies the prospective observed-time authority needed to start that
horizon, including strict A4.3 namespace binding, current-membership extension checks,
exact HTTP request/response evidence, revision preservation, cutoff uniqueness, and
strict replay. It does not create historical observations, training rows, a model, or
serving eligibility. The first live poll remains environment-pending.
