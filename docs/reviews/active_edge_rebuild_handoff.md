# Active Edge Rebuild Handoff

Status: active

Last updated: 2026-09-07

Repository: `C:\project\market-predictor`

Branch: `er-intraday-refactoring`

Last completed implementation commit: `3b2bff5` (`Define SPY-relative swing research and verify evidence inventory`)

## Purpose

Current user request, 2026-09-07: implement the **first two checkpoints** of the
long-only swing plan, using High effort. Checkpoint one is verified and pushed;
checkpoint two accounting is next. Stop after its verification and Git closure.
Do not start issuer-feature expansion, training or intraday development in this turn.
Read the Long-Only Swing Research And Implementation Plan at the front of
`docs/active_edge_rebuild_plan.md`. No real model was trained or promoted.

## Completed Objective And Evidence Checkpoint

- `configs/swing_research.toml` and `swing/contracts/research.py` freeze the new
  SPY objective separately from historical strategy/feature identity. Config hash:
  `e37a72796bac9c08ad92a467d079d2a70b4d938f34e286f8dd1e0129a6e9007e`.
- `swing/evaluation/research_evidence.py` and `configs/swing_research_evidence.toml`
  provide a bounded metadata-only audit, exposed as
  `market-predictor-research audit-swing-research-evidence --root .`.
  Inventory hash: `883a141b03b1016bfdf97d5ebbb789cddbfe9ac51eb65262854e17b6cfd01d69`.
  Real audit passed: fifteen records, 120 ordered features and sixty recorded trial
  entries in five manifests. Duplicates and uncovered lifetime history are explicit.
  It follows no raw/prediction payloads and does not parse the exposed evaluation.
- The retained trainer refuses July 2025-June 2026 as a new final test before
  loading data or fitting. Other dates are not automatically certified fresh.
- Protocol and feature audit now match the 2019-07-09 cutoff, retained warm-up,
  current split config, exposed historical test and bounded regression campaign.
- Verification: 39 focused tests; complete suite **2,070 passed, three skipped,
  133 warnings, 27m08s**; full `ruff check src tests`; strict mypy on 315 source
  files; real CLI audit; diff checks. Two independent reviewers had no remaining
  supported finding. Full-suite integration omissions in the documented split
  sentence and reviewed CLI inventory were corrected before the passing rerun.
- Existing static debt was resolved mechanically. One reference-proven unused,
  invalid duplicate `intraday/datasets/dataset_io.py` was removed; the canonical
  loader in `intraday/training/training.py` is unchanged semantically. A4.3's
  current transformation remains identical. Import-only KS4 implementation hashes
  differ, but affected retained experiments are rejected/unserved and unchanged;
  no accepted authority or model was invalidated. Do not rebuild them.
- Sampled test working sets were approximately 0.21-0.25 GiB; no Python worker
  remained. The access-restricted historical evaluation required elevated read
  access for hashing only. No permissions or protected data were changed.

### Accounting Design Review

Reuse the existing daily ledger. Allocate purchase notional `prior NAV / 10`,
equal-weight within the cohort, then cap pro rata against cash including prepaid
costs. Entries precede exits; proceeds are available next session. Add the exact
original label cost back to cumulative net paths to recover gross holding values,
then charge the configured cost once through cash. Aggregate repeated-security
exposure without merging different exit lots. Include zero-selection/idle days and
the fixed ten-session maturation tail; validate all dates before outcome loading.

Use actual full-calendar SPY bars, not approximate barrier-exit-close comparisons.
Keep those approximations diagnostic. Account returns need not equal net labels:
upfront costs reduce affordable purchase notional. Prove normalized cash, holdings,
realized/unrealized P&L and NAV reconciliation with deterministic fixtures and a
hash-bound retained-data control, never fitting-data predictions called out-of-sample.

Both accounting implementation and source admission must be honest. Current bars
declare SIP and `adjustment=all`, but independent total-return/distribution proof
has not been established. Reviewer-approved closure is **accounting code verified;
price-basis admission blocked**. Emit `price_ratio_diagnostics` and
`price_basis_pending`; positive metrics or caller-authored passed/verified flags
must not override that blocker. No speculative bulk download or invented dividend
credit is needed. Do not claim raw executable fills or verified SPY outperformance.

This repository produces prediction intelligence and abstention. Alerts, orders,
positions, portfolio risk, and execution remain in `trading_flow`.

The objective, target, policy and validation declarations are now frozen; their
accounting implementation must be verified before fitting. This is not a promise to reach AUC 0.60.
Existing rejected artifacts remain rejected and no promoted serving bundle exists.

## Pre-Implementation Research Findings

The following records the research preceding `3b2bff5`. The completed checkpoint
above supersedes its protocol and contract gaps; accounting gaps remain current.

Two independent read-only reviewers inspected model/evaluation evidence and catalyst
research respectively. The main reviewer also reproduced the benchmark metadata/code
conflict and read the historical technical evaluation's holdout-access fields.

1. **Previously exposed holdout.**
   `data/models/swing/technical/evaluation.json` records
   `locked_test_outcomes_read=true`, `test_access_count=1`, and final-test results for
   July 2025-June 2026. Its V11-lineage bundle is not a current V12/A2 candidate. The
   independent reviewer checked its model/evaluation/card/manifest hashes and found
   historical candidate-eligibility flags inconsistent with failed economic gates.
   Current code requires the gates; a current bypass was not reproduced. Later
   specialist runs had no final-test reads, but this does not restore calendar
   independence. Earlier unqualified "locked test unopened" statements below are
   historical per-run statements, not a valid current claim for the whole project.
2. **Benchmark semantics disagree.**
   `src/market_predictor/edge_rebuild/training/economics.py::_economic_gate` and
   `training/swing_evaluation.py::_selection_key` consume approximate
   managed-exit-session-close excess. Generated `swing_training.py` metadata describes
   fixed-ten-session comparisons as the selection authority instead. The approximation
   compares a stock barrier fill to a benchmark close, not necessarily the same time.
   Correct contracts, reporting and gates together; the performance effect has not
   been quantified. Existing fixed-horizon labels and the daily-position ledger
   should be reused, not duplicated.
   Historical specialist rejections retain aggregate metrics, not immutable prediction
   rows. Do not promise to replay unavailable predictions. Verify accounting first
   with deterministic controls, then attribute learned results from new saved
   chronological out-of-fold predictions.
3. **Real weak results, not just an AUC threshold.**
   The August 20 upgrade/downgrade development experiments produced zero models; best
   worst-scope AUC was approximately 0.524/0.552. The old August 9 technical
   classifier/regressor reported positive managed portfolio returns, but negative
   approximate SPY excess per selected trade and negative unseen-security portfolio
   returns. Removing the 0.60 gate alone would not rescue those results. Full details
   and artifact boundaries are in the active plan; these are not new backtests.
4. **Data already exists.**
   Technical panel: 853,417 rows, 604 securities, 1,759 sessions. Corrected broker
   comparison: 27,087 matched rows and 11,720 unique latest announcements per profile.
   Do not return to the invalid 113/19-event interpretation. SEC manifests report
   689,467 events across 624 issuers; this review inspected manifests, not every raw
   filing. Existing canonical SEC text is form-level and does not prove that original
   exhibits, guidance values, or earnings surprises are available as model inputs.
5. **Stale split prose, resolved in `3b2bff5`.**
   `docs/model_training_validation_protocol.md` previously described a May-2019 requirement
   and missing warm-up. Current temporal config and the feature audit use
   2019-07-09 through 2024-05-28 for initial fit; the panel request includes history
   from 2018-05-29. The protocol now agrees, without downloading already-retained
   history or moving the approved news cutoff.

### Recommended Implementation Sequence

1. Define the SPY objective and reconcile evidence, split/access history and contracts.
2. Reconcile fixed/managed returns, one-account funding, costs and daily SPY accounting.
3. Complete causal issuer-news/reaction features using existing archives first.
4. Train at most six return-model specifications sequentially under 5 GiB.
5. Evaluate the predefined long-only policies on funded net equity curves.
6. Start frozen prospective research predictions, never recycle the exposed test year.
7. Evaluate at the fixed boundary and promote only verified swing inference/API.

The active plan contains the research citations, known-strategy names, six-specification
experiment, two-policy trial cap, source/feature requirements, exit tests, ownership,
and stop conditions. Earnings-reaction continuation and explicit guidance changes
are the first issuer hypotheses. News novelty, volume and the reaction already known
before the decision matter more than generic positive/negative tone. SEC form flags
and retrospective edited headlines are not substitutes for original causal content.
Reaction features must distinguish a completed post-release interval from an
announcement-window return containing pre-release trading. Recall audits include
unclassified/rejected articles; the control is no qualifying event in a covered
source window, never proof that no public news existed.

The independent economic and catalyst reviewers completed one consolidated design
review; all supported plan corrections were incorporated. Documentation verification:
`python -B -m pytest -q -p no:cacheprovider tests/test_active_continuity_documents.py`
passed both tests. No full-suite run or new model run is claimed for this docs-only
checkpoint. Historical implementation verification below remains historical.

No guarantee of beating SPY is made. New code can be correct while an experiment is
economically rejected or statistically inconclusive. Keep those statuses distinct.

## Historical Verified State

The sections below retain earlier implementation and per-run evidence. Their old
checkpoint codes, source counts and next-step wording do not override the current
long-only swing sequence above. In particular, historical per-run holdout protection
does not prove this project's July-2025 to June-2026 outcomes remain unseen.

### Retained Implementation Facts

- No promoted serving bundle exists. Production prediction paths must fail closed.
- Reddit and Seeking Alpha remain retired and prohibited.
- Swing decisions begin on `2019-07-09`; earlier bars are warm-up only.
- The current intraday data authority is
  `data/features/intraday_causal_volume_bar_dataset_20260831_v2`. Intraday estimator
  input is the ordered A4.3 bar-only technical contract, sampled on
  fixed five-minute cohorts from causal completed evidence. The V3 z-score lineage is
  invalid and prohibited.
- Current swing feature construction is owned by `swing/features` with serving
  adapters under `serving/swing_features.py`; older top-level module names below
  describe historical migrations, not current import paths.
- The swing base authority contains only `technical_market`. A3 event evidence is
  published separately and cannot alter baseline probability.
- The swing trainer no longer attaches a hard-coded SEC authority, fills unknown SEC
  coverage with zero, or bypasses sector-constrained selection.
- Intraday label schema V2 requires exact stock, SPY, QQQ, and point-in-time sector ETF
  returns over the same entry-to-managed-exit interval. Missing QQQ evidence abstains.
- Both trainers publish named binary diagnostics for estimator target, positive
  after-cost stock return, and positive SPY/QQQ/sector excess return.
- A deterministic 64-repeat global label-permutation AUC control must remain at chance;
  abnormal discrimination fails evaluation. Single-class scopes are explicit
  `not_applicable_single_class`, never fabricated metrics.
- Intraday remains capped at 4 GiB and swing candidate training at 5 GiB.
- A4.1 retains all selected stock-sessions for collection. End-of-session bar coverage
  is metadata; feature eligibility is evaluated only through a decision cutoff and
  label eligibility only through its managed outcome horizon.
- A4.1 raw collection is bounded by a required finite job count, resumes failed jobs
  from verified page tokens, hashes every job/attempt/raw page, preserves zero-sided
  quote states, and cannot publish authority until every planned job replays.

## A2: Technical Swing Baseline Verification

- Commit `cb2aba5` freezes four nested baseline groups: momentum/volatility, trend
  confirmation, pullback timing, and volume/liquidity.
- Six sequential candidates are permitted: four regularized logistic ablations and
  full-feature XGBoost ranker/regressor candidates.
- Each fitted model persists its exact ordered feature subset. Batch serving and the
  research API slice to that subset; missing or reordered inputs fail closed.
- Bundle IDs are deterministic and lineage-bound. Candidate payload, model card,
  evaluation, authority replay, and promoted bundle must agree.
- Serving selects only the signed model-family frame: baseline uses
  `technical_market`; an event specialist requires a separately promoted A3 contract.
- Current Finviz snapshots are not historical point-in-time authority. Quality,
  profitability, investment, valuation, and estimate-revision groups remain blocked.
- No real candidate was trained or promoted in A2, so AUC and economics are unchanged.

## A3.1/A3.2: Issuer Event Authority Verification

- V2 title-derived classifications require a causal issuer anchor and an approved
  source/family pair. Ambiguous bare tickers, another issuer's catalyst, and
  preview/conditional language do not become training events.
- The `2019-07-09` through `2021-07-08` authority strictly replays 9,018 classified
  events, 30,875 assignments, and 28,462 coverage rows. Peak observed memory was
  1.998 GiB.
- The `2021-07-09` through `2026-07-08` authority strictly replays 26,370 classified
  and research-eligible events across 525 securities, 90,136 assignments, and 18,333
  coverage rows. Manifest-recorded peak memory was 2.157 GiB.
- The second authority was rebuilt entirely from the existing 2,608-chunk Alpaca
  archive: 2,608 chunks verified, zero failures, and 980,034 attribution relations.
  No provider request or redownload occurred.
- Seven Alpaca-supported families are admitted. `sec_material_event` is
  `blocked_missing_source`; missing SEC authority is not represented by Alpaca
  coverage or a numeric zero.
- Both authorities are retrospective, research-only evidence. A3.3 precision review
  and A3.4 matched-dataset publication are complete. No A3 model, locked-test metric,
  or promotion exists.

## A3.3: Event Precision Audit Verification

- Commit `6ef0579` adds deterministic uniform sampling of unique
  family/headline/publication-day clusters, a separate paired issuer diagnostic,
  disk-backed population indexing, strict immutable replay, and per-family admission.
- Commit `9c8aa5b` makes malformed reviewer ledgers fail before costly authority replay.
- `2019-07-09` through `2021-07-08`: 1,788 inferential clusters and eight diagnostics
  were reviewed; only `analyst_revision` passed.
- `2021-07-09` through `2026-07-08`: 1,830 inferential clusters and 29 diagnostics
  were reviewed; only `analyst_revision` passed.
- Each sample used two independent Codex reviewers and a distinct adjudicator. This is
  model-assisted research review, not human audit evidence and not production authority.
- Earnings, guidance, offering, merger/acquisition, regulatory decision, and product
  event remain blocked by reviewed false positives, wrong-issuer observations,
  confidence bounds, agreement, or rule-variant gates. SEC remains source-missing.
- Both final audit artifacts strictly replay and remain `production_ready=false`,
  `training_eligible=false`, and `alerts_eligible=false`.

## Current Verification

- The original A3.4 result is invalid because old event decision/security hashes were
  compared directly with rebuilt technical-panel hashes. Only 210 rows joined before
  quality gates, which falsely reduced the data to 113 rows from 50 announcements.
- Corrected A3.4 independently replays 27,087 prediction rows from 11,720 unique latest
  broker announcements per comparison dataset, 81,261 physical rows total, with a
  1.954 GiB recorded peak. Exact ticker and exact prediction timestamp are required;
  conflicting CIKs fail closed. Every exclusion is persisted in
  `identity_alignment_audit.parquet`.
- Repository suite: 1,433 passed and 2 skipped after the exact SIP transport review.
- Tracked Ruff, strict mypy across 229 source files, compileall, and final observed
  membership, prospective poll, and classified horizon strict replay pass.
- Coordinated request, feature, all-profile label, dtype, partition, global identity,
  source-coverage, causal-window, and governance-hash poison tests pass.
- The real materialization and event publication remained below the 5 GiB limit;
  observed working memory was below 2 GiB during A3.4.

## Model State

| Model family | Current state | Next valid work |
| --- | --- | --- |
| Swing baseline | A2 trainer complete; prior candidates rejected; no new run or promotion | Preserve the frozen technical contract until a governed training run is approved |
| Swing event-driven | Combined rating/coverage and separate upgrade/downgrade specialists trained in development; all rejected | Preserve rejection evidence; do not open locked test or serve |
| Intraday baseline | V2 rejected; V3 invalid; A4.4 continuation and reversion both rejected | Preserve evidence; no serving or future-holdout access |
| Intraday event-driven | Historical identity correction yields 1,912 attached events and two research-only no-candidate results; prospective source horizon is 1/20 SIP sessions and 12 analyst episodes in the new chain | Continue append-only prospective SIP sessions and polls; do not serve proxy-time research outputs |

`ROC-AUC >= 0.60` is a locked-test diagnostic, not a training objective or permission
for repeated locked-test tuning. Promotion also requires ranking, calibration,
after-cost benchmark-relative economics, drawdown, turnover, capacity, stability, and
coverage.

## A3.4: Matched Broker-Action Comparison Verification

- The V12 catalyst-independent base panel strictly replays 853,417 technical rows,
  604 securities, and 1,759 sessions from `2019-07-09` through `2026-07-08`.
- The source authorities contain 17,401 direct-issuer broker announcements. Causal
  coverage produces 37,372 prediction timestamps from 16,149 unique latest
  announcements before technical-panel alignment.
- The corrected A3.4 authority publishes 27,087 matched prediction rows from 11,720
  unique latest announcements in each of three comparison datasets: technical-only,
  broker-action-only, and technical-plus-broker-action. Internal profile names retain
  `analyst_revision` for source lineage.
- Profiles share exact decision IDs, labels, execution/economic lineage, and
  episode-normalized weights. Coordinated request, feature, label, dtype, partition,
  and global-identity tampering is rejected by replay tests.
- The artifact is `production_ready=false`, `training_eligible=false`,
  `research_training_eligible=true`, and `serving_eligible=false`.

## A3.5: Broker-Action Specialist Result

- Rating upgrades and downgrades form one directional rating-change specialist.
  New/resumed coverage is separate. Price-target/generic actions are report-only.
- Rating-change capacity: 5,841 development, 1,138 chronological-validation, and 213
  unseen-security-validation announcements across eight sectors.
- Coverage capacity: 2,344 development, 502 chronological-validation, and 94
  unseen-security-validation announcements across eight sectors.
- Six experiments per specialist compare technical-only, broker-action-only, and
  combined features using logistic and histogram gradient boosting.
- Selection uses an inner 2023-06-13 through 2024-05-28 window after a ten-session
  embargo. Best worst-scope AUC was 0.546 for rating changes and 0.506 for coverage;
  every candidate also failed canonical portfolio economics. Broker-only inputs were
  near or below chance. No inner candidate passed, so outer validation was not opened.
- Artifact: `data/models/swing_broker_action_specialists_dev_20260812_v4`. Peak
  working memory: 0.414 GiB. Locked-test outcomes read: false. Models emitted: zero.
- Strict deterministic V4 replay passed. Repository verification: 1,219 tests passed,
  2 skipped; tracked Ruff, strict mypy across 221 source files, and compileall passed.
- The unseen-security scope is a stable held-out-symbol stress test fitted and scored
  inside the same inner time window. It tests transfer to new symbols; it is not a
  second independent chronological validation period.
- V2 is superseded: it reused one validation window for selection/evaluation and used
  simplified event-weighted returns instead of canonical portfolio economics.

## A3.6: Directional Swing Broker-Action Result

- Commit `82b4959` uses the existing governed swing-specialist trainer and changes
  only cohort membership: rating upgrades and rating downgrades are evaluated as
  separate specialists. Profiles, labels, folds, costs, estimators, gates, and the
  locked-test boundary are unchanged.
- Upgrade capacity passes with 3,008 development, 561 chronological-validation, and
  102 unseen-security announcements. Downgrade capacity passes with 2,833, 577, and
  111 announcements respectively. Each development cohort contains 359 securities
  across eight represented sectors.
- Twelve experiments compare technical-only, broker-action-only, and combined inputs.
  Upgrade best worst-scope inner ROC-AUC is 0.524 for combined logistic regression
  (0.533 chronological / 0.524 unseen). Downgrade best is 0.552 for combined
  histogram gradient boosting (0.552 / 0.560), only 0.002 above its technical-only
  control in the weaker scope.
- No threshold passes canonical after-cost economics in both scopes, and no experiment
  reaches the 0.60 AUC gate. The result is `no_development_candidate`; outer
  validation and the locked test remain unopened, no model was emitted, and promotion
  is prohibited.
- Artifact:
  `data/models/swing_directional_broker_action_specialists_dev_20260820_v1`. Strict
  replay passed. Peak working memory was 0.376 GiB. Final verification passed 1,463
  tests with two skipped, tracked Ruff, strict mypy across 231 source files, and
  tracked compilation.

## A4.1: SIP Trade/Quote Collector Verification

- Corrected v2 plan: 43,226 selected stock-sessions, 86,452 trade/quote jobs, 559
  observed symbols; 13 incomplete source-bar sessions are retained as metadata.
- Live bounded probe: two jobs completed across two invocations, zero failures,
  4.41 MiB written, 0.350 GiB peak RSS, and no authority published.
- Full replayable trade/quote backfill is storage-blocked on the current local drive.
  Partial raw transport cannot become a feature, zero, or training input.
- Two independent reviewers found no remaining high or critical findings after fixes
  for immutable failed-attempt inventory, path overlap, page-level resume, causal
  population selection, matched ablation, and zero-sided quotes.

## A4.3: Current Bar-Only Technical Intraday Authority

- Commits `8a76ec1`, `e76bf8d`, and `1829fce` publish and verify the fixed-cohort
  bar-only dataset authority without a provider download or any
  trade/quote/microstructure input.
- The immutable five-minute projection contains 43,226 selected stock-sessions and
  3,364,335 rows: 43,132 stock-session pairs are complete and 94 incomplete pairs are
  retained as explicit coverage metadata.
- The current immutable dataset is
  `data/features/intraday_causal_volume_bar_dataset_20260831_v2`. It contains 794
  sessions, 501 tickers, 3,095,688 rows, and 1,365,015 eligible rows.
- Dataset request SHA-256:
  `5e8c508a4237320d6ea56205502f670244a49f713b22dbff10a336d4d2dc303a`.
  Transformation SHA-256:
  `6fdfd0c8f07e4f7445b66d038cbd936e4459db68e087a5ddbcb30eac4795cb51`.
- Audit v2 is
  `data/reports/intraday_causal_volume_bar_dataset_20260831_v2_audit_v2.json`, SHA-256
  `f1b21af3704317d070f479ac6a562fdb126c21d2b7da021ddf5c7b6108be97e8`.
  It requires the exact projection path and authority/manifest/inventory hashes and
  reports zero normalized cutoff, raw source cutoff, incomplete five-minute prefix,
  duplicate, label-availability, eligible-ATR, ordered-feature-hash, schema, or
  prohibited-feature violations.
- The resumed publication occurred before mandatory per-invocation execution receipts.
  `data/reports/intraday_causal_volume_bar_dataset_20260831_v2_execution_assessment.json`
  therefore records `status=incomplete`, `recorded_scope=final_invocation_only`, and
  `complete_run_memory_proven=false`. Do not reconstruct or overstate this evidence.
- Future publication requires a separate execution-evidence directory. Every invocation
  is receipt-bound, aggregate parent/worker memory must remain within 3.25 GiB, and
  interrupted evidence-authority finalization resumes deterministically.
- Two independent reviewers reported no remaining P0, P1, or P2 findings. No model was
  trained, no locked test was opened, and no serving bundle was promoted.

## A4.4 Result

Implementation commit `a062653` trained both hypotheses from the completed A4.3
authority. Continuation's best audited seen/unseen positive-return ROC-AUC was
0.510/0.516; long reversion's was 0.513/0.508. Both controlling scopes failed
after-cost return, benchmark-excess confidence bounds, and fold stability. Stop-risk
ROC-AUC was about 0.60 but Brier score and calibration gates failed. Outputs are
`data/models/edge_rebuild_intraday_bar_continuation_dev_20260814_v1` and
`data/models/edge_rebuild_intraday_bar_long_reversion_dev_20260814_v1`. Both are
strict `no_candidate` authorities with no `candidate.joblib`; the future holdout stayed
closed. Peak RSS was below 2.1 GiB. Verification passed 1,320 tests with 2 skipped,
tracked Ruff, strict mypy over 225 source files, compileall, and independent review.

## A5.1 Result And Exact Next Step

Implementation commit `98f7a48` publishes the causal intraday event preflight.
Authority `data/research/edge_rebuild_intraday_event_preflight_20260815_v1` strictly
replays as `blocked`: 17,401 research broker-action episodes reduce to 19 unique
episodes and 862 event-decision pairs under exact `security_id` attachment. All 17,401
events use retrospective `provider_publication_proxy`; production-eligible events and
decisions are both zero. Retrospective collection completion cannot create historical
known-zero coverage. Peak working set was 2.299 GiB.

The artifact is `training_eligible=false`, `serving_eligible=false`, and
`future_holdout_opened=false`; it contains no model. A5.2 and A6 are not legal next
steps. The next valid work is a new bounded data-authority checkpoint that obtains
genuinely observed first-seen and revision timestamps plus point-in-time security
identity for future broker actions, then reruns A5.1 under a new preregistered data
horizon. Historical observation time must not be inferred from publication time.
A4.2/A4.5 trade/quote work remains storage-blocked and must not be replaced with zero.

## A5.1d Historical Identity Correction And Training Result

- Implementation commits: `4cc5d4e` (identity reconciliation) and `cda4c1f`
  (event-confirmed training and verification).
- Literal historical event `security_id` matching was defective: 17,401 direct-issuer
  events produced only 19 attached episodes because event authorities commonly use
  `cik:<value>:ticker:<symbol>` while A4.3 uses `cik:<value>`.
- Exact ticker plus CIK-compatible reconciliation publishes corrected authority
  `data/research/edge_rebuild_intraday_event_preflight_20260820_v2`: 1,912 unique
  episodes, 83,636 event/decision pairs, 487 securities, and 771 sessions. Source IDs
  remain retained; conflicting CIKs fail and ambiguity abstains.
- The research-only catalyst role remains confirmation/filtering, not a direct feature.
  Continuation trained on 14,451 rows and returned 0.513 seen / 0.509 unseen
  positive-return ROC-AUC. Long reversion trained on 12,951 rows and returned 0.535 /
  0.519. Both have negative after-cost return and benchmark excess, profit factor below
  one, unstable folds, and failed stop calibration, so both are `no_candidate`.
- The technical-only reference remains continuation 0.510 / 0.516 and long reversion
  0.513 / 0.508. Catalyst therefore slightly improves long-reversion discrimination but
  does not create a tradable edge. No future holdout was opened.
- Verification passed 1,456 tests with two skipped, tracked Ruff, strict mypy across
  231 source files, and compilation. Peak training RSS was about 3.16 GiB under the
  unchanged 4 GiB hard cap and 3.25 GiB safety threshold.

## A5.1e Directional Intraday Broker-Action Result

- Implementation commit `4821780` adds strict upgrade, downgrade, and coverage cohort
  selection to the existing 30-minute research trainer. Parent authorities replay
  once; subtype classification uses only retained, hash-verified parent event fields.
- Upgrades pass capacity with 805 announcements, 32,970 decisions, 285 securities,
  and 461 sessions. Downgrades pass with 860 announcements, 31,243 decisions, 273
  securities, and 439 sessions. Coverage fails before training with 245 announcements,
  169 securities, 168 sessions, only 38-55 announcements per validation fold, and 41
  unseen-security announcements.
- Upgrade continuation: 6,709 rows, seen/unseen ROC-AUC 0.492/0.531. Upgrade long
  reversion: 5,341 rows, 0.495/0.494. Downgrade continuation: 5,991 rows, 0.510/0.485.
  Downgrade long reversion: 6,041 rows, 0.506/0.529.
- All four outputs are `no_candidate`. Isolated positive scopes have only 14-35 unseen
  trades and fail in the paired seen scope; other scopes fail after-cost return,
  profit factor, benchmark, calibration, confidence-bound, or fold-stability gates.
  Future holdout and serving remain closed.
- All four immutable output directories strictly replay. Peak working set was 3.193
  GiB. Final verification passed 1,460 tests with two skipped, tracked Ruff, strict
  mypy across 231 source files, and tracked compilation.

## Prospective Broker-Action Authority

- Implementation commit `5530246` adds
  `collect-edge-prospective-broker-actions` and
  `publish-edge-prospective-broker-action-generation`.
- Every poll strictly binds the complete A4.3 dataset and a current membership
  authority whose history must reproduce A4.3's identity namespace through the A4.3
  cutoff. A later membership cutoff is allowed only as a verified extension.
- Raw Alpaca asset/news bodies, exact endpoint/query/final URL, no-redirect state,
  request/response times, provider revisions, source coverage, identity abstentions,
  immutable failed attempts, and a stable cutoff claim/commit registry all replay.
- Parent polls must precede the child and use the same namespace and registry. A
  membership authority may advance only through a strictly replayed monotonic
  observed-time chain; observation time, release outcomes, and events cannot move
  backward. Replay is iterative, not recursive. Identity changes remain quarantined
  until a governed transition authority resolves them.
- Polls and generations are bounded below 4 GiB; generation input is additionally
  limited by verified Parquet uncompressed size. Generations preserve earliest
  observed first-seen time and every distinct provider revision. They remain
  `training_eligible=false` and `serving_eligible=false`.
- Membership-extension implementation commit `42ebe5f` adds a hash-bound archive
  cutoff, complete base-prefix preservation, CIK conflict rejection, strict parent
  envelope verification, and the shared prospective namespace verifier.
- Final evidence: 111 focused tests passed; full suite 1,374 passed and 2 skipped;
  tracked Ruff, strict mypy across 227 source files, compileall, memory below 0.2 GiB,
  and one consolidated two-reviewer correction pass.
- The first August 15 extension artifacts were collected before that New York day
  ended and remain invalid. Do not use any `*_20260815_v1` S&P extension artifact.
- Closed-cutoff raw archive
  `data/raw/index_membership/spglobal_official_20180414_20260815_v2` contains 109
  verified releases. Event authority `spglobal_events_20180414_20260815_v3` contains
  307 events with zero unresolved releases. Transition authority
  `sp500_transitions_20180529_20260815_v2` contains 13 transitions. Membership
  authority `sp500_memberships_20180529_20260815_v2` contains 1,170 intervals, 659
  securities, and five governed exclusions, and preserves the exact July 8 prefix.
- Local Market Predictor configuration uses the paper trading host because the loaded
  key is a paper key; the stock data feed remains `sip`. No credential is committed.
- Successful poll `data/raw/prospective_broker_actions/poll_20260816T071230Z` uses
  stable registry `data/raw/prospective_broker_actions/registry_v2`. Strict replay
  verifies 503 eligible identities, 76 identity-bound observations, 46 observed and
  457 known-empty symbol collections, and 0.328 GiB peak working memory. Earlier
  failed/resumed poll directories and `registry_v1` are immutable non-evidence.
- Commit `27ab9b7` also permits only the exact Alpaca live/paper asset hosts and replaces
  transient whole-page hash exceptions with exact semantic grammar for the malformed
  2018 Twitter/Monsanto release. Final evidence is 74 focused tests, 1,382 tracked tests
  passed and 2 skipped, tracked Ruff, strict mypy over 227 source files, bytecode
  compilation, strict real-authority replay, and no remaining medium-or-higher review
  finding.
- August 21 authority
  `data/raw/index_membership/sp500_observed_20260821T071500Z_v7` and poll
  `data/raw/prospective_broker_actions/poll_20260821T071000Z` strictly replay 503
  constituents and 451 identity-bound observations. This starts a new causal chain;
  it does not conceal the missing polls after August 17.

## Prospective Analyst-Event Horizon

- Implementation commit `fe2b4c9` adds
  `publish-edge-prospective-analyst-revision-horizon`. It strictly replays and combines
  chronological, non-overlapping prospective generations while preserving every
  provider revision and earliest observed response time.
- One episode is one Alpaca provider event plus exact security identity. Revisions do
  not increase episode capacity. Provider timestamp anomalies, cross-generation
  identity conflicts, non-analyst headlines, and events without a causal issuer anchor
  remain retained but ineligible.
- Authority `data/research/prospective_analyst_revision_horizon_20260817_v2` strictly
  replays 536 revisions, 248 provider events, two polls, and three qualifying analyst
  episodes across AMCR, HBAN, and WDAY. Source capacity is `blocked`; training,
  serving, and future-holdout access are false.
- Coverage rows carry the exact poll security identity. Publication/provider-update
  times cannot replace first-observed availability. Peak publication memory was 0.478
  GiB. A consolidated reviewer reproduced one timestamp dtype replay defect; the fix,
  mixed-null chained-poll regression, real v2 replay, 1,431-test suite, Ruff, strict
  mypy across 229 source files, and compileall all pass.
- `data/research/prospective_analyst_revision_horizon_20260817_v1` predates the dtype
  normalization and is superseded. Do not use it as current evidence.
- Generation `data/research/prospective_broker_actions_generation_20260821_v1` and
  horizon `data/research/prospective_analyst_revision_horizon_20260821_v4` strictly
  replay one poll, 451 revisions, and 12 qualifying analyst episodes. Capacity remains
  blocked and training eligibility is false. Combining the August 17 and August 21
  generations failed closed because their poll chain is not contiguous.

## Exact Alpaca SIP Bar Transport

- Implementation commit `c56843e` changes the shared Alpaca bar-page client from
  parsed JSON plus headers to bounded exact HTTP bytes plus parsed output. The page
  contract retains raw bytes, requested/final URL, direct-response status, retrieval
  time, safe headers, and redirect evidence.
- Requests require SIP, explicit point-in-time `asof`, `adjustment=all`, ascending
  order, bounded symbols/pages, and an exact `https://data.alpaca.markets/v2/stocks/bars`
  query. Redirects, non-200 status, content-type changes, naive retrieval time, and
  body hash/length/representation mismatches fail closed.
- Existing swing and intraday collectors remain compatible and now receive exact
  transport evidence, but they do not yet constitute the new prospective session
  authority. No provider request or data download occurred in this checkpoint.
- Focused verification passed 44 tests; the full suite passed 1,433 tests with two
  skipped. Tracked Ruff, strict mypy across 229 source files, compileall, and one
  independent consolidated review passed with no medium-or-higher finding.

## Weekday-Safe Observed S&P Membership Authority

- Implementation commit `a5aae9b` is pushed. It adds the collection-only
  `collect-edge-observed-sp500-memberships` command and preserves the fully closed S&P
  archive/event/membership authorities as immutable parents.
- The observed authority archives exact no-redirect official search and release
  responses, confirms the complete fetched page range after collecting independent
  constituent and SEC ticker/CIK anchors, records every release outcome and pending
  effective change, and strictly replays its complete file inventory and canonical
  membership table.
- A weekday poll accepts only an observed authority captured before the poll and no
  more than the configured 60-300 seconds earlier. It must not cross a known pending
  effective change. Authority rotation must retain prior observed releases/events and
  move observation time forward; collection and strict replay use the same chain gate.
  Closed archive authorities remain weekend-only.
- Commit `a7fe60e` retains exact SEC CIK-specific fallback evidence for bulk-map
  omissions, rejects anchor/inherited CIK disagreement, and records same-ticker CIK
  successors only at first observation. Raw-unit envelopes, content-addressed body
  paths, fallback inventory, and canonical input lineage all replay exactly.
- Real authority
  `data/raw/index_membership/sp500_observed_20260817T203000Z_v3` strictly replays 503
  constituents and 10,397 SEC identities, including the AEP fallback and the XOM
  successor identity observed at `2026-08-17T20:29:05.344531Z`. Zero new membership
  releases or events were found.
- Poll `data/raw/prospective_broker_actions/poll_20260817T202950Z` immediately follows
  that authority and strictly replays 11 of 11 batches, 460 observations, and 454 exact
  production-identity events at cutoff `2026-08-17T20:30:00Z`. Peak working memory was
  0.379 GiB. Both artifacts remain non-serving and non-training authorities.
- Final evidence: 152 focused tests and 1,417 tracked tests passed with 2 skipped;
  tracked Ruff, strict mypy across 228 source files, compileall, two-reviewer
  remediation, and final strict artifact replay passed.

Implementation checkpoint `6386cb4` closes the real `EQR` to `VMRK` ticker-successor
gap without weakening identity or timing rules. Failed immutable attempt
`data/raw/index_membership/sp500_observed_20260819T200115Z_v4` remains non-authoritative.
Complete authority `data/raw/index_membership/sp500_observed_20260819T201500Z_v5`
strictly replays 503 constituents at `2026-08-19T20:06:40.779393Z`; its manifest hash
is `4a7b58f35aec5077ac7b82ce3c1a0a7675df1faedddf4390499b981d773635a6` and universe
hash is `5b6e68d4844f9b0baa00e517bf0b515ddc1648a730776bf9dba201cf9082b1b3`.
Final verification passed 1,457 tests with two skipped, tracked Ruff, strict mypy across
230 source files, tracked compilation, and independent re-review with no remaining
high- or medium-severity finding.

Historical A5.1c continuation instruction, now superseded by the structural repair:
collect each
eligible SIP session with a valid pre-open membership parent until twenty contiguous
sessions exist, then build causal features, mature outcomes, and rerun the prospective
preflight. The completed historical directional experiments cannot substitute for
observed-time production evidence. Further retrospective broker-action slicing is not
authorized without a genuinely new preregistered hypothesis.

Current horizon: `2026-08-20` is source session 1/20. The next collection window is
after the `2026-08-21` XNYS close plus the frozen finalization delay and before the
`2026-08-24` open, using the August 21 pre-open membership parent. Continue scheduled
event polls as a new contiguous chain; any missed cutoff starts another separate chain.

## Source Boundary

| Source | Permitted role |
| --- | --- |
| Alpaca SIP/all bars, trades, quotes | estimator market/microstructure data after complete causal backfill |
| Alpaca direct ticker news | ticker event data after exact attribution and availability verification |
| SEC issuer filings | separate A3 event family after causal backfill; not an A2 baseline shortcut |
| Finviz Elite | screening/current metadata; historical news needs its own causal authority |
| Verified global/sector sources | separate context overlay unless preregistered and ablated |
| Reddit | prohibited |
| Seeking Alpha | prohibited |

## Working Tree State

Implementation commit `9408515` is pushed. The working tree was clean before this
documentation closure. No provider data, model artifact, or evidence authority was
created, rewritten, or deleted by the swing catalyst decision-authority move.

## Files To Read

1. `AGENTS.md`
2. `docs/active_edge_rebuild_plan.md`
3. `docs/reviews/active_edge_rebuild_handoff.md`
4. `docs/reviews/feature_engineering_audit_20260801.md`
5. `docs/model_training_validation_protocol.md`
6. Current `git status` and recent commits

## Do Not Do

- Do not train or promote from invalidated intraday z-score or old label-schema lineage.
- Do not open locked tests for feature or hyperparameter selection.
- Do not weaken economic, sector, causality, memory, or integrity gates.
- Do not fill missing news, catalyst, quote, filing, or source coverage with zero.
- Do not add a feature without complete historical backfill for its model horizon.
- Do not expose rejected candidates through the production prediction API.
- Do not execute scratch scripts that mutate Parquet or patch lineage hashes.

## Active Structural Repair

The August 24 review found that commits through `6119fe1` left the package refactor
incomplete. Measured baseline on August 27: `ruff check . --statistics --no-cache`
reports 407 findings and `mypy src` reports 31 errors across nine files,
duplicate intraday development configuration classes, an unconstructible causal
calibration result, a direct locked-holdout `NameError`, duplicate old/new namespaces,
and an `intraday.evaluation` module/package collision. Prior claims that the complete
suite and static checks passed are therefore superseded.

Implementation commit `99f635c` completes **Holdout access and shared contract repair**.
Future access now uses one exclusive claim plus a candidate-bound immutable reservation
receipt. Candidate-only checks and registry isolation precede reservation; every
post-claim failure leaves failure evidence and retries fail closed. Successful evidence
copies and verifies the reservation before atomic publication. One
`IntradayDevelopmentConfig` remains, and `CausalCalibrationFit` is constructible.
Verification: 57 focused tests passed; touched source files passed Ruff and strict
mypy; the assigned senior reviewer accepted the bounded diff after two remediation
rounds. Repository-wide Ruff and mypy remain open by design for later repair tasks.

The **Serialized artifact and namespace inventory** is complete in implementation
commit `3026450`. `docs/model_artifact_retention_inventory.json` is the authority for
retention decisions. Its original baseline recorded 53 source/test files importing
`market_predictor.v3`, 126 importing `market_predictor.edge_rebuild`, 171
chronology-named tracked paths, and two rejected serialized files importing
`market_predictor.v3`. The later semantic package migration removed every source/test
import of `market_predictor.v3` and retired those two namespace-bound joblibs while
retaining their manifests. No serialized file in `data` imports either old namespace.

The default research catalog remains active and now exposes four explicit model IDs:
`swing_technical`, `swing_technical_with_catalyst`, `intraday_technical`, and
`intraday_technical_with_catalyst`. Their local directories use matching behavioral
paths. All authority, manifest, and candidate hashes replay after the move. Only
`swing_technical` has a research-scoring candidate; every model remains
promotion-ineligible and non-actionable. Hash-bound specialist and rejection evidence,
active-plan development evidence, raw data, canonical data, features, and research
inputs were retained. Only the two explicitly audited, rejected, unreferenced joblibs
that depended on the removed Python namespace were deleted.

Verification for `3026450`: 10 focused tests passed; touched Ruff and strict mypy
passed; the default research service reported the four expected states; all four
catalog bundle hashes and all six tracked specialist authority/manifest/request hashes
matched. The assigned senior reviewer accepted the bounded diff with no P0/P1 finding.

Implementation commit `ade847c` completes **Cross-sectional research consolidation**.
The `market_predictor.v3` source package is gone. Contracts now belong to `core`,
`evidence`, `modeling`, or `universe`; S&P Global raw archive transport belongs to
`sources/spglobal`; verified index changes and point-in-time membership belong to
`universe/sp500`; production cross-sectional transforms belong to `intraday/features`;
reusable validation and ranking economics belong to `modeling`; and development-only
training, ablation, diagnostics, readiness, and candidate acceptance remain under
`research/intraday_cross_sectional`. Production packages cannot import `research` or
`commands`, enforced by an AST test. Active Python APIs and test files use behavior
names; frozen persisted schema string values remain unchanged.

Verification for `ade847c`: 330 focused test cases passed across migrated research,
features, labels, model training, S&P archive/event reconstruction, membership, and
direct consumers. Ruff passed; strict mypy passed on 44 source files; collection and
research CLI imports succeeded; diff checks passed. Post-review cleanup restored
formatter-only consumer churn, reran 20 consumer tests and five architecture/artifact
tests, and retained only the manifests for the two retired namespace-bound joblibs.
The assigned senior reviewer accepted the final staged diff with no P0, P1, or P2
finding.

Implementation commit `60cff69` completes **Historical membership and security identity
authority migration**. Corpus-integrity checks and `IntegrityThresholds` now belong to
`evidence/corpus_integrity.py`; membership identity validation and SEC identity
authority belong to `universe`; historical S&P transition and membership authorities
belong to `universe/sp500`. All direct source, command, and test imports were updated.
The five old `edge_rebuild` modules are absent. Architecture tests enforce a temporary
universe dependency allowlist and scan source, tests, and scripts for direct, aliased,
package-module, and symbol imports of removed paths. Persisted schema strings and
authority/hash behavior remain unchanged.

Verification for `60cff69`: 60 authority tests and 106 direct-consumer tests passed.
After review, four additional poison cases brought the architecture guard to eight
passing cases. Ruff passed; strict mypy passed on 15 source files; semantic authority
and command imports succeeded; diff checks passed; no Python worker remained. The
assigned senior reviewer accepted the final staged diff with no P0, P1, or P2 finding.

Implementation commit `5259bdb` completes **Prospective observed-membership source and
authority separation**. Provider HTTP identity, no-redirect collection, exact retained
bytes, raw-unit sidecars, source parsing, and raw replay now belong to
`sources/spglobal/observed_membership_collection.py`. The unchanged authority request
hash is generated by the universe orchestrator and passed verbatim into every source
unit. The single collector lock still spans parent validation, source collection,
membership construction, publication, and final strict replay. Membership lineage,
identity reconciliation, effective intervals, canonical publication, and authority
replay now belong to `universe/sp500/observed_membership_authority.py`.

The old `edge_rebuild/sp500_observed_memberships.py` module and edge-rebuild-prefixed
test file are absent. All command, SIP-session, broker-action, analyst-horizon, and
test consumers import the semantic authority path. Architecture guards enforce the source
dependency allowlist, reject source-to-universe imports, reject every import form of
the removed module, and require the old file to remain absent. Exact raw-envelope,
request-hash, raw/root inventory, body/sidecar tamper, path-escape, and real lock
contention tests fail closed as designed.

Verification for `5259bdb`: 113 affected authority, architecture, SIP-session,
broker-action, analyst-horizon, and CLI tests passed. Ruff passed; strict mypy passed
on six affected source files; compileall, command/authority import smoke, zero-reference
scan, staged diff checks, and the process check passed. No Python process remained.
The assigned senior reviewer accepted the final diff with no P0, P1, or P2 finding.

Implementation commit `9244893` completes **SEC filing evidence and decision authority
migration**. `sources/sec.py` remains the SEC provider transport. Causal issuer filing
events, collection coverage, conservative availability, retained raw-response replay,
and immutable collection publication now belong to
`catalysts/sec_filings/collection.py`. Decision-time filing overlays, explicit unknown
coverage, monthly partitions, lineage, publication, and strict replay now belong to
`catalysts/sec_filings/decision_authority.py`.

The two old `edge_rebuild` modules and their edge-rebuild-prefixed tests are absent.
Commands and tests import the semantic catalyst package. The catalyst dependency
allowlist prohibits imports from `edge_rebuild` and upper horizon packages, while the
source allowlist prevents a reverse dependency. Persisted schemas and all behavioral
contracts remain unchanged. Verification passed 44 SEC, architecture, and CLI tests,
Ruff, strict mypy on three source files, compileall, semantic import smoke,
zero-reference/file-absence checks, and diff checks. The assigned senior reviewer
accepted the move with no P0, P1, or P2 finding.

Implementation commit `5e1f65f` completes **GDELT transport, canonical global-event
collection, and decision-authority separation**. `sources/gdelt.py` is the only GDELT
HTTP transport and owns validated requests, bounded retries, no-redirect behavior,
provider URL identity, raw response hashes, and tagged provider records. Immutable
normalization, deduplication, scoring, observed availability, coverage, publication,
and replay belong to `catalysts/global_events/collection.py`. Decision-time global
coverage and features belong to `catalysts/global_events/decision_authority.py`.
`commands/market_context.py` alone converts raw documents to the older `NewsEvent`
command output; production catalyst code does not depend on that schema.

The two old `edge_rebuild` modules and the duplicate `GdeltSource` transport are
absent. Direct consumers use semantic imports without aliases. A fixed
characterization test preserves exact request parameters, raw-response identity,
request/query/source-policy hashes, canonical event raw hash, availability, coverage,
and persisted schema values. Host, path, redirect, partial-response, and retry poison
tests fail closed. Verification passed 98 focused tests, Ruff, strict mypy on six
source files, compileall, removed-module/API guards, and diff checks. No Python worker
remained. The assigned senior reviewer accepted the final diff with no P0, P1, or P2
finding.

Implementation commit `4f271ca` completes **Alpaca issuer-news evidence collection and
audit migration**. The immutable collector and strict audit now belong to
`catalysts/issuer_events/alpaca_news_collection.py` and `alpaca_news_audit.py`;
`sources/alpaca.py` remains the sole provider transport. Persisted schema strings are
unchanged and centralized in `news_history_contracts.py`. Canonical symbol handling
belongs to `core/symbols.py`, while provider-specific symbol mappings belong to
`sources/provider_symbols.py`, so catalyst code no longer depends on the former mixed
top-level symbol module.

Old news-history and symbol modules, imports, and tests are absent. Architecture guards
reject every removed import form and require the removed files to remain absent.
Collector/audit behavior, request and work-unit identity, source coverage, canonical
normalization, availability, hashes, locking, resume behavior, memory limits, persisted
schemas, and strict replay are unchanged. Verification passed 109 focused parity tests
and the complete suite with 1,530 passed and 2 skipped. Affected-file Ruff, strict mypy
on 14 source files, compileall, dependency/file-absence guards, diff checks, and the
process/memory check passed. The assigned senior reviewer accepted the final diff with
no P0, P1, or P2 finding.

Implementation commit `2b9e195` completes **issuer-event classification and attribution
foundations migration**. Reusable event-family classification, relevance, attribution,
and attribution-history behavior now belongs to `catalysts/issuer_events`. The
rule-variant helper belongs only to `classification.py`; the precision audit calls it
through that module and cannot re-export it. An AST guard rejects old-owner definitions,
direct or aliased imports, plain or annotated assignment aliases, and other stale
consumer imports.

Exact event-family and attribution policy hashes, all persisted schema/version strings,
every rule-variant branch and fallback, representative classification/relevance/
attribution outputs, and attribution-history replay remain fixed. Old swing foundation
modules and imports are absent; direct tests use issuer-event behavior names.
Verification passed 247 focused parity tests, 88 tests after reviewer fixes, 54 final
ownership/dependency tests, and the complete suite with 1,551 passed and 2 skipped.
Affected-file Ruff, strict mypy on 12 source files, compileall, removed-module scans,
diff checks, and process checks passed. The assigned senior reviewer accepted the final
diff with no P0, P1, or P2 finding.

Implementation commit `9408515` completes **swing catalyst decision authority
migration**. The decision-time feature authority now belongs to
`swing/features/catalyst_decision_authority.py`. Commands, serving, swing feature
construction, live feature binding, and tests import that path directly; no alias or
compatibility module remains. Persisted request, authority, manifest, lineage,
decision-artifact, and coverage-artifact identity strings are frozen by tests. The old
module/file and every import form are guarded against reintroduction. A separate AST
guard rejects both `swing -> intraday` and `intraday -> swing` imports.

Verification for `9408515`: 171 focused tests passed with one skipped; affected Ruff
and strict mypy on six source files passed; compileall, old-reference scans, staged
diff checks, and process/memory checks passed. The complete isolated suite passed 1,569
tests with two skipped in 13 minutes 16 seconds. The assigned senior reviewer accepted
the final diff with no P0, P1, or P2 finding. An earlier full-suite attempt produced
setup-only errors after its repository-local pytest parent was removed; the clean rerun
used an isolated external pytest directory and had no failures.

Implementation commit `03f8233` completes the **issuer-family evidence and horizon
assignment split** as a byte-preserving projection over the retained combined v2
envelopes. `evidence/issuer_family_combined_envelope.py` strictly verifies frozen root,
child, inventory, path, schema, policy, and hash contracts and computes a neutral
identity that excludes swing decisions. `catalysts/issuer_events/family_evidence.py`
owns the single neutral semantic replay for classified events, source coverage, and
unclassified evidence. `swing/datasets/issuer_event_family_cohort.py` owns swing
assignments and cohort replay. Intraday consumes only `IssuerFamilyEvidence`; it no
longer reads or validates swing assignments. The old mixed module, test, CLI command,
and imports are absent and guarded.

No persisted data was rewritten. The retained v2 envelope schema and authority hashes
remain unchanged, so this is not a claim that neutral and swing tables are stored as
separate authorities. Such a storage migration would require new schemas, regenerated
artifacts, and downstream lineage changes and needs explicit approval. The superseded
`data/research/issuer_event_family_20190709_20210708_v1` directory is not accepted by
the v2 loaders and is retained pending a separate reference-proven deletion decision.

Strict real-data replay evidence:

- `issuer_event_family_20190709_20210708_v2`: authority
  `f6ad6fff560177e5ec3cc9f40018d2ef3bf9038e0a9d57e41ce4127e6ddf7c08`, full
  inventory `f4cc4e919b6c839e6e22c33b7fbd0f925c49ed01acd5ee52115553b516f53bb8`,
  neutral projection `f2272439b492a0fcde8ded41ab82ae2ad11756a0e540c4185e466fc27359f458`,
  9,018 events, 28,462 coverage rows, 30,875 assignments, 267 cohort rows, and
  3,982 unclassified artifacts.
- `issuer_event_family_20210709_20260708_v2`: authority
  `aa8d208f41a902bdb9f9432334dab19c6b78affaa928ac3b6794ada377b8f927`, full
  inventory `b3e292ac472c176f5cc28178dff64234234eb53587d95e116c5161518c4e7344`,
  neutral projection `8dcaac805c77515b154ed2bef681e1537f2c02dfa1c54b0a431507bc06d23fab`,
  26,370 events, 18,333 coverage rows, 90,136 assignments, 519 cohort rows, and
  2,604 unclassified artifacts.

Verification for `03f8233`: 178 focused tests passed; the isolated complete suite
passed 1,584 tests with two skipped in 12 minutes 47 seconds; affected-file Ruff and
strict mypy on six source files, compileall, removed-path scans, diff checks, and
process checks passed. Peak observed Python memory during retained-data replay stayed
below 2 GiB. The assigned senior reviewer accepted the final diff with no P0, P1, or
P2 finding. Repository-wide static cleanup remains plan task 6: current whole-tree
Ruff reports 278 pre-existing findings and strict mypy reports 16 errors in three
intraday dataset files; they were not expanded into this bounded checkpoint.

Implementation commit `7ce23a0` completes **issuer-event precision governance**.
Deterministic sample publication, blind review resolution, immutable artifact
integrity, and family/rule-variant admission now belong to
`governance/issuer_event_precision`. The old combined module and test name are absent;
architecture guards reject every old import form and prevent governance from importing
either trading horizon. Command names and swing-ablation semantics are unchanged.

Publication remains fail-closed and atomic. Child manifests are rewritten to their
intended final paths while still staged, the complete staged authority is replayed
against that intended location, and only then is the directory atomically published.
The final public loaders do not expose the staging-only path binding. Injected sample
and audit corruption leaves no output directory. Symlink rejection has both a real
filesystem test and a permission-independent inventory test.

Strict retained-data replay after the final implementation:

- `2019-07-09` through `2021-07-08`: 1,796 sample/review rows, sample authority
  `1de62f84b72d8e793b0d10de65354edaac7baab9f97433096ab3dceb1873cddd`, audit
  authority `e68f66dd47d8f156e6040ccb473556aed75b0c74acaa01065d85eff0a475946f`.
- `2021-07-09` through `2026-07-08`: 1,859 sample/review rows, sample authority
  `b4bab375d8f1cd5dcae2d349fdac5bb3d1967398cd808c932aaec873c59c37c9`, audit
  authority `4e82c21cfd4b5daf9cdc4ad85d52bea52c81fc98f3dbe6eb405becdf0985735a`.

Verification: 120 affected tests passed with one skipped; the final focused governance
suite passed 25 tests with one skipped; the complete isolated suite passed 1,594 tests
with three skipped in 15 minutes 6 seconds. Affected Ruff and strict mypy, compileall,
removed-module scans, diff checks, temporary-directory cleanup, and process checks
passed. The same assigned senior reviewer accepted the final diff with no P0, P1, or
P2 finding.

The following phase consolidates **swing and intraday packages** under descriptive
`contracts`, `datasets`, `features`, `labels`, `training`, `evaluation`, and `live`
packages. Remove the intraday evaluation module/package collision and remaining
chronology/checkpoint names without compatibility aliases. Preserve mathematical,
causal, artifact, and command behavior and obtain independent design and final diff
review before closure.

Implementation commit `a176fbb` completes **intraday module/package collision
removal**. The unreachable `intraday/contracts.py` and `intraday/evaluation.py` shadow
files are deleted. Python continues resolving the public APIs to
`intraday/contracts/__init__.py` and `intraday/evaluation/__init__.py`; configuration
classes retain their serialized owner `market_predictor.intraday.contracts.configs`.
No production import, persisted artifact, or CLI changed.

A repository-wide recursive guard now rejects any sibling module/package collision,
and a poison fixture proves nested collisions are detected. Characterization freezes
the 95-feature order hash, schema strings, default label policy and SHA-256, Pydantic
validators, pickle ownership, and representative evaluation metrics. Verification
passed 143 affected tests and the complete isolated suite with 1,601 passed and three
skipped in 14 minutes 41 seconds. Touched Ruff, compileall, deleted-path scans, diff
checks, temporary-directory cleanup, and process checks passed. The assigned senior
reviewer accepted the final diff with no P0, P1, or P2 finding.

Implementation commit `c408d58` completes the **shared strategy contract migration**.
The cross-horizon contract now has one production owner at
`modeling/strategy_contract.py`; all consumers import it directly and no compatibility
alias exists. The persisted schema string remains `edge_rebuild.strategy_contract.v2`,
the active configuration SHA-256 remains
`39213ad6bd5c1f09f30065f737ffecadf05bbb0ae81b81f2ffda7a343967e972`, and retained
artifact scans found no serialized Python owner at the removed module path. Recursive
architecture guards reject every old import form and reintroduction of the old file.

Verification passed 452 affected tests with two skipped and the complete isolated suite
with 1,605 passed and three skipped in 14 minutes 7 seconds. Ruff on the migrated
authority and boundary tests, strict mypy, compileall, import smoke, removed-path scans,
diff checks, and process-memory checks passed. The assigned senior reviewer accepted the
final diff with no P0, P1, or P2 finding.

Implementation commit `8d42d26` completes the **shared mathematical primitive
ownership** checkpoint. `FeatureStep` and `FeaturePipeline` now have one production
owner at `modeling/feature_pipeline.py`; the file is byte- and AST-identical to the
removed `edge_rebuild/pipeline.py`, both horizons import it directly, and no alias
exists. Old-path poison guards cover every Python import form and old-file
reintroduction. No retained serialized artifact references the removed owner.

The independent design review rejected moving `edge_rebuild/cross_sectional.py` or
`edge_rebuild/technical_relationships.py` into `modeling`: their actual transforms and
consumers are swing-specific, so they belong in the later `swing/features` migration.
It also requires `edge_rebuild/labeling.py` to be split: shared outcome constants move
to a horizon-neutral owner, while daily barriers and session/sector rank labels move to
`swing/labels`; intraday keeps its exact minute-path label authority.

Verification passed 119 focused tests and the complete isolated suite with 1,613 passed
and three skipped in 14 minutes 5 seconds. Touched Ruff, strict mypy, compileall,
code-hash parity, removed-path scans, diff checks, and process-memory checks passed. The
assigned senior reviewer accepted the final diff with no P0, P1, or P2 finding.
Repository-wide static verification was run and remains open: Ruff reports 198 existing
findings (107 import-order, 61 import-placement, 20 unused imports, 10 unused
redefinitions), and strict mypy reports 348 existing findings across 61 files. These
counts are the baseline for the dedicated static-quality checkpoint, not passes.

Implementation commit `09341dc` completes the **intraday history-collection contract
migration**. The full Alpaca/SIP acquisition contract now has one production owner at
`intraday/contracts/history_collection.py`. All collectors, intraday datasets, command
adapters, and tests import the new owner directly; the old module is absent and no
compatibility alias exists. The moved implementation is byte- and AST-identical with
source hash `6b5d3b42c73aeb40958ca01b5a35b2a821d1de46`.

Characterization freezes all eight Pydantic class owners and the six active
configuration identities:

- intraday history: `252886fb7b7fcfca19917a1daa8e1ea43d950e006287adca12796525c911a830`
- extended sessions: `2fb6118c448438c5ffe59a1cb3319b39f4e80bf47bca5c77df55948e204700d6`
- selected five-minute sessions: `536a8194d376cf2e6925d90b8bf22e7f071fc2854c793c9b5a365b78b2841c22`
- selected one-minute sessions: `0c2896b7e40a5c0afb502c65b6ce167f16705d1b36aa44d0a90c94f2ffe1e318`
- selected benchmarks: `4215b3f63b7b5ff0cf30c6415d35362653f4f492510a9cab9a04b971be14c2cf`
- broad intraday history: `07bd5c64ef9c1b66b09cec7122e62c3abd4cda83e3ebea1d34742b497e993832`

Architecture guards reject all old import forms and reintroduction of the removed
file. Existing package-direction tests keep `sources` independent of horizon
contracts. Readable retained artifacts contain no serialized reference to the removed
owner; four pre-existing model directories remain unreadable under their Windows ACL
and were not modified. The reviewer found no retained serialized artifact risk and
approved the final diff with no P0, P1, or P2 finding.

Verification passed 171 focused tests, all 56 intraday development tests after an
interrupted external test edit was restored to the current production owners, and the
complete isolated suite with 1,631 passed and three skipped in 13 minutes 12 seconds.
Touched Ruff, strict mypy, compileall, collection CLI import/help, exact source-hash
parity, old-path scans, diff checks, and the 4 GiB process-memory gate passed.

Implementation commit `26c048d` completes **swing contract package and materialization
schema ownership**. `swing/contracts.py` is now byte-for-byte
`swing/contracts/__init__.py`, so `FrozenConfig`, `SwingDatasetConfig`,
`SwingTrainingConfig`, and `SwingPromotionConfig` retain Python and pickle owner
`market_predictor.swing.contracts`. The source identity remains
`36b698837a09a8cd0b23e9b48e4be291afa91727`.

The two swing materialization schema constants moved byte-for-byte from
`edge_rebuild/swing_artifact_contracts.py` to
`swing/contracts/materialization.py`, retaining source identity
`c7add055ab12ab53d46988f89da862f0a631649a` and schema strings
`edge_rebuild.swing_panel_materialization.v12` and
`edge_rebuild.swing_panel_materialization_authority.v12`. Every consumer uses the
canonical module explicitly. The reviewer found and then verified the fix for one P2:
direct constant imports had temporarily exposed accidental aliases on three legacy
modules. Regression tests now prove those aliases are absent.

Characterization freezes class owners and pickle round trips, the 99-feature full
profile hash `a841554e6edb6e63e6571cf653e064f51fb9c67a893aac63b266b6e0dfe3792f`,
the 53-feature technical profile hash
`4d68fd5327f1cc535ba1458a1138cd4faac866a4c129c686c2a48bede0de81fb`,
default dataset/training/promotion hashes, and label-policy hash. Accessible retained
artifacts contain no old materialization module reference; the four pre-existing
Windows-ACL-protected intraday model directories were unchanged.

Verification passed 208 affected tests with one skipped and the complete isolated
suite with 1,645 passed and three skipped in 13 minutes 35 seconds. New-file Ruff,
changed import-order Ruff, strict mypy, compileall, import/no-alias smoke, source-hash
parity, old-path scans, diff checks, and the 4 GiB process-memory gate passed. The
known pre-existing unused re-export findings in `swing_training.py` remain part of the
Step 6 static-quality baseline. The reviewer accepted the final diff with no remaining
P0, P1, or P2 finding.

Implementation commit `dd4dbcd` completes **swing technical-relationship feature
ownership**. `edge_rebuild/technical_relationships.py` moved byte-for-byte to
`swing/features/technical_relationships.py`; both lazy pipeline imports now reference
the new owner directly, the old source and test names are absent, and no compatibility
alias exists. Source identity remains
`391bac1540b6ef414dced0338b842cedc5e54bdb`.

Characterization freezes `TechnicalRelationshipSpec` at Python/pickle owner
`market_predictor.swing.features.technical_relationships`, ordered nine-feature hash
`6fc5f34e633e3be00092da294bc86afd1d155d3898b7faff415497d67770bf38`,
strategy-derived specification hash
`9409760785ae9d31b67866e5f5f92cd118f1dd32b3a3c5a473a107b4836890a4`, and
representative output hash
`814c438377415f3255c7fcd2bb16f005243f47c75302e8a5463c173c4845d4ec`.
Existing tests continue to prove five-bar pivot confirmation timing, append-only future
causality, price/volume and trend/range calculations, session/group resets, input
validation, and row-order preservation.

Readable retained artifact scans found no old Python owner; the same four
Windows-ACL-protected intraday specialist model directories were unchanged and cannot
contain this swing-only specification. Verification passed 170 affected tests with two
skipped and the complete isolated suite with 1,651 passed and three skipped in 16
minutes 30 seconds. New-owner Ruff, changed import-order Ruff, strict mypy, compileall,
import smoke, exact source parity, old-path scans, diff checks, and the 4 GiB memory
gate passed. The reviewer accepted the final diff with no P0, P1, or P2 finding.

Implementation commit `68d9893` completes **swing cross-sectional feature ownership**.
`edge_rebuild/cross_sectional.py` moved byte-for-byte to
`swing/features/cross_sectional.py`; all production/test consumers use the canonical
module explicitly, the old source and test names are absent, and no compatibility
alias exists. Source identity remains `cfb54c43d06382235fd341d9a9713a5262715c4f`.

Characterization freezes `CrossSectionSpec` at Python/pickle owner
`market_predictor.swing.features.cross_sectional`, specification hash
`2700655375ed15afba2c3c96a49c5c794bb4cf455179df7bd49c7f22ffbec45e`, emitted
column-order hash `3cc6ebd5f00ec0a89737fe7468aac1e782706e782ad3b8e619a977b0ac4f9867`,
representative output hash
`209c3b424677ac8c282439673917fbef6cf2bae3d37c3ffb338496412ba05cec`, and exact
suffixes `_xs_z`, `_xs_rank`, and `_sector_z`. Existing and added tests prove
future-session causality, session/sector isolation, sample-standard-deviation behavior,
winsorization, minimum peers, constant/outlier behavior, row/output ordering,
collisions, missing columns, and empty frames.

Readable retained artifacts contain no old Python owner; the same four protected
intraday-only specialist model directories were unchanged. Verification passed 181
affected tests with two skipped and the resumed complete suite with 1,660 passed and
three skipped. The suite reported 2 days 2 hours because the app was closed while the
same process was suspended; it resumed and completed without duplication or failure.
New-test and changed-import Ruff, strict mypy, compileall, import smoke, exact source
parity, old-path scans, diff checks, and the 4 GiB memory gate passed. The moved
byte-identical file retains its pre-existing import-spacing Ruff finding for Step 6.
The reviewer accepted the final diff with no P0, P1, or P2 finding.

Implementation commit `61f6f4c` completes **shared label outcomes and swing
barrier/rank ownership**. `modeling/label_outcomes.py` is the sole owner of
`TARGET_HIT`, `STOP_HIT`, `TIMEOUT`, `RANK_TOP`, `RANK_BOTTOM`, and `RANK_MIDDLE`.
The compact canonical JSON hash is
`b021c7ad67fedfe5ca3685189f184488520994bb86c0e68535beb76c52d36c19`.
Intraday minute-path labels and swing daily-path labels reference that module without
re-exporting the constants.

`swing/labels.py` moved byte-for-byte to `swing/labels/__init__.py`, preserving the
existing `market_predictor.swing.labels` function owner and Git object identity
`142a32c95ca97f99a06bd807037233949e06f96b`. `BarrierSpec`, barrier/rank columns,
and daily/session-sector implementations now belong to
`swing/labels/barrier_and_rank.py`; the old mixed module and test names are absent.
Frozen identities are:

- barrier specification: `1a6bee0ccb2e5c0b8c54b6ff45b9d5e641d4e57da747092a605da39baceb960f`
- barrier columns: `2513343a01863d35bbca80c97b980666f20a2ef381c1e9ff00e619c957469cfa`
- rank columns: `f89aa0051ff32e5a4b7d8efae2aa9e9ef4876e3a03a71182bc1c40b09fd59b56`
- representative barrier plus managed-return output: `47ea63e0186f0509f1bb2e3ebf9f697026968fc20fbbd3a01344d62e346faa3d`
- representative sector-rank output: `72fbbb05ec4fedd5fbe7cc271e2f233ed33d48c7d0a8cd3592b60f48bdfe5c14`

Tests also freeze output columns/dtypes, class/pickle ownership, conservative fills,
unknown horizons, future-prefix causality, session and sector isolation, package
origin, and absence of accidental consumer aliases. Architecture tests reject the old
module in every import form and prohibit `modeling` from importing `swing` or
`intraday`, including relative imports. Readable retained artifacts contained no old
owner reference; the same four Windows-ACL-protected intraday specialist directories
were not modified and cannot contain the swing-only `BarrierSpec`.

Verification passed 155 direct label/boundary tests, 133 broader swing/intraday tests
with two skipped, 135 tests after output/boundary review fixes, and 112 final boundary
tests. The complete isolated suite passed 1,681 tests with three skipped in 21 minutes
51 seconds. Affected Ruff, strict mypy on six production files, compileall, package
byte parity, old-owner scans, diff checks, generated-temp cleanup, and the 4 GiB memory
gate passed. The assigned senior reviewer found three P2 test/guard gaps, verified all
fixes, and approved the final diff with no remaining P0, P1, or P2 finding.

## Intraday Causal Volume-Bar Ownership Result

Implementation commit `e76bf8d` is pushed. The byte-identical implementation now lives
at `intraday/datasets/volume_bars.py`; `publisher.py`, `dataset_v2.py`, and
`bar_dataset.py` import it directly. The renamed characterization test freezes
`VolumeBarBuildResult` at its new pickle owner. Architecture guards reject the removed
module and all four Python import forms. No compatibility alias or old file exists.

The direct `bar_dataset.py` import change necessarily changed the complete
transformation identity because that module hashes itself. The current contract is:

- schema: `market_predictor.intraday.bar_dataset_transformation.v2`
- aggregate SHA-256: `6fdfd0c8f07e4f7445b66d038cbd936e4459db68e087a5ddbcb30eac4795cb51`
- canonical volume-bar source SHA-256:
  `0ab5baee5f9e7d92e1592855b554d89ee35ebc8a865ab4152f82e11697c5912d`
- `VOLUME_BAR_COLUMNS` SHA-256:
  `55a343087cce34eb04f438c55cce369b0f35ffbdf1e552b675ff4507fc02f849`
- `AUDIT_COLUMNS` SHA-256:
  `a840bf2a667f85d3a78f3e4747282e64b4ee26128878587963ba86c87d8f2388`
- representative bars/audit SHA-256:
  `5a3856eda8f43ef4ea80765d5ce99268d9ceb9b8bb5a85360d864e43497d7c6f` /
  `c7b26557b7bb29a2ed5a8d59a6859a49d8b51cf29256fd583a1fac97e1b308dc`

Source identity normalizes CRLF and LF before hashing, so Windows publication and
Linux/cloud replay are identical. It does not normalize provider data or artifact
bytes. Focused verification passed 194 tests; Ruff, compileall, diff checks, package
guards, old-import scans, publication/resume/atomicity, obsolete-identity rejection,
and immutable-old-output tests passed. The code reviewer re-ran the remediation and
reported no P0, P1, or P2 finding. After both implementation commits, the complete
isolated suite passed 1,691 tests with three skipped in 13 minutes 30 seconds.

During full-suite preparation, an interrupted external test edit exposed two
AST-identical ledger implementations. Supplemental implementation commit `a4002ce`
makes `intraday/evaluation/ledger.py` the sole owner of position-ledger construction,
position closing, and ledger metrics. `gates.py` and `training/coordinator.py` import
that owner directly; `economics.py` now owns only ranking diagnostics. A runtime test
freezes all three production bindings. Verification passed 173 affected tests, Ruff,
strict mypy on four source files, compileall, and a one-owner source scan. The code
reviewer approved the remediation with no remaining P0, P1, or P2 finding.

The retained
`data/features/edge_rebuild_intraday_bar_only_causal_20260814_v1` authority remains
immutable historical evidence with 3,095,688 rows and transformation `0da898cc...`.
The current loader rejects it. It cannot be resumed, trained, promoted, or served as a
current authority. Bound historical evidence includes its audit report, both
`edge_rebuild_intraday_event_preflight_20260815_v1` and
`edge_rebuild_intraday_event_preflight_20260820_v2`, and the retained bar-only,
event-confirmed, upgrade-confirmed, and downgrade-confirmed continuation/reversion
development bundles. None is promotable. Four Windows-ACL-protected swing candidate
directories are separately classified as historical/non-promotable in
`docs/model_artifact_retention_inventory.json`; they are not intraday volume-bar
authorities and were not modified.

Deterministic rematerialization is complete at
`data/features/intraday_causal_volume_bar_dataset_20260831_v2`. Strict loader replay,
audit v2, and all 794 session units pass. The old and current row/session hashes are
identical; this is code-lineage rematerialization, not new statistical evidence. Commit
`1829fce` closes the audit and future execution-evidence defects without modifying
`bar_dataset.py`, `bar_features.py`, `bar_labels.py`, or `volume_bars.py`; transformation
SHA remains `6fdfd0c8...cb51`.

Verification after the final code change: 25 focused audit/execution tests, 63 broader
intraday/architecture tests, 57 corrected development tests, and the complete suite with
1,714 passed and three skipped. Affected Ruff and strict mypy, compileall, CLI help,
strict data replay, audit replay, diff checks, and process checks pass. Repository-wide
static cleanup remains Step 6 debt: 168 Ruff findings and 14 strict mypy findings across
three untouched files.

This registers the current intraday **data authority only**. The historical A4.4 models
remain rejected; identical rows do not justify retraining them. Historical event
preflights remain bound to the obsolete authority and retrospective provider timestamps.
Intraday training, promotion, serving, and locked-test access remain unauthorized.

Commit `d34ea25` moves selected stock-session verification and one-minute/five-minute
acquisition planning byte-for-byte to
`intraday/datasets/selected_session_history.py`. All direct consumers use the new
owner, the old module is absent and guarded against reintroduction, and
`SelectedSession` now has explicit owner and pickle characterization. Persisted plan
schemas, fingerprints, unit identities, policy hashes, and artifact bytes are
unchanged. The source Git object remains `2f43aa9f...f2bd`; current transformation
`6fdfd0c8...cb51` and the 794-session data authority replay unchanged.

Verification passed 174 focused planning/coverage/benchmark/projection/dataset/boundary
tests and the complete suite with 1,719 passed and three skipped. Affected Ruff and
strict mypy, compileall, research CLI help, import smoke, old-path and source-parity
scans, diff checks, process checks, and temporary-output cleanup passed. The
repository-wide Step 6 baseline remains exactly 168 Ruff and 14 strict mypy findings in
untouched files. Independent code and ML-design reviews closed with no P0, P1, or P2.

Commit `7e96bc4` moves canonical intraday history materialization byte-for-byte to
`intraday/datasets/history_materialization.py`. The source Git object remains
`f5acc0e7...f74a`; `SessionBounds` and the five public functions have explicit canonical
owner characterization, and all old import forms are prohibited. Exchange-calendar
segmentation, early closes, selected-session eligibility, source overlap, ticker
quarantine, memory guards, and the two persisted materialization schemas are unchanged.
No data artifact was regenerated. Current transformation `6fdfd0c8...cb51` and the
794-session authority replay unchanged.

Verification passed 200 focused tests and the complete suite with 1,724 passed and three
skipped. Affected Ruff and strict mypy, compileall, research CLI help, import smoke,
old-path and source-parity scans, diff checks, process checks, and temporary-output
cleanup passed. Repository-wide Step 6 debt remains exactly 168 Ruff and 14 strict mypy
findings in untouched files. Independent code and ML-design reviews closed with no P0,
P1, or P2.

Implementation commit `01275d4` moves bounded Alpaca/SIP intraday history collection
byte-for-byte to `intraday/datasets/history_collection.py`. The source Git object
remains `ce3c6f3...c6b2b4`; all six production consumers use the canonical owner, all
old import forms are prohibited, and no alias remains. Exact request validation,
raw-page replay, resume without network access, bounded concurrency and memory,
atomic publication, and all six persisted collection/unit/authority schemas are
unchanged.

The retained stock collection replays 2,116 units and 16,636,841 rows with authority
`63d9d714...fd5c` and manifest `625b0832...b76`. The retained benchmark collection
replays 794 units and 4,005,350 rows with authority `889dcc46...dde6` and manifest
`8b3f1e57...bd61`. The downstream 794-session authority remains 3,095,688 rows with
1,365,015 eligible rows under request `5e8c508a...303a` and transformation
`6fdfd0c8...cb51`. No provider request, artifact regeneration, training, promotion,
serving, or locked-test access occurred.

Verification passed 242 focused tests and the complete suite with 1,729 passed and
three skipped in 22 minutes 20 seconds. Affected Ruff and strict mypy, compileall,
collection CLI help, import and old-path scans, source parity, both retained collection
replays, downstream authority replay, diff checks, process checks, and temporary-output
cleanup passed. Repository-wide Step 6 debt remains exactly 168 Ruff and 14 strict
mypy findings in the same three untouched files. Independent task, code, and ML/data
reviews closed with no remaining P0, P1, or P2 finding.

Implementation commit `268c62d` moves selected-session one-minute coverage and
canonical five-minute verification byte-for-byte to
`intraday/datasets/one_minute_coverage.py`. The source Git object remains
`c770f369...06ead`; all five production consumers use the canonical owner, all old
import forms are prohibited, and no alias remains. Alpaca `1Min`/SIP/`adjustment=all`
identity, plan and selection lineage, exact XNYS and early-close counts, separate
one-minute density and five-minute continuity, whole-security exclusion, the 5%
ceiling, canonical integrity, and staged final authority publication are unchanged.

The retained authority replays ready at 43,226 stock-sessions across 502 securities,
with 13 incomplete sessions retained as metadata and zero excluded securities. Its
authority remains `e18adef7...a75b` and manifest `d21c1733...c560`. The downstream
794-session authority remains 3,095,688 rows with 1,365,015 eligible rows under request
`5e8c508a...303a` and transformation `6fdfd0c8...cb51`. No provider request, artifact
regeneration, training, promotion, serving, or locked-test access occurred.

Verification passed 249 focused tests and the complete suite with 1,734 passed and
three skipped in 22 minutes 34 seconds. Affected Ruff and strict mypy, compileall, CLI
help, import and old-path scans, source parity, retained coverage and downstream
authority replay, diff checks, process checks, and temporary-output cleanup passed.
Import ordering corrected in two touched files, so repository-wide Ruff debt is now
166 findings; strict mypy debt remains 14 findings in the same three untouched files.
Independent task, code, and ML/data reviews closed with no remaining P0, P1, or P2.

Implementation commit `9a59d6f` moves selected-session benchmark acquisition planning
byte-for-byte to `intraday/datasets/benchmark_history.py`. The source Git object
remains `1c106774cadf7fdf6c514406f72590cf0d782e62`; the command adapter and direct tests
use the canonical owner, every old import form is prohibited, and no alias remains.
SPY, QQQ, all eleven sector ETFs, exact XNYS session bounds, 390 normal-session and
210 early-close bars, SIP, `adjustment=all`, selection/membership lineage, atomic
publication, and the 4 GiB process limit with 0.75 GiB headroom are unchanged.

The retained plan at
`data/research/edge_rebuild_selected_session_benchmark_1m_plan_causal_20260801_v1`
strictly replays 794 sessions, 13 benchmarks, and eight early closes at manifest
`b855b250...e72`, authority `ffd0783d...0a0`, and fingerprint
`af77d941...6616`. No provider request, artifact regeneration, training, promotion,
serving, or locked-test access occurred.

Verification passed 228 focused tests and the complete suite with 1,739 passed and
three skipped in 12 minutes 15 seconds. Compileall, CLI help, canonical and old-path
imports, source parity, retained-plan replay, diff, process, and temporary-output
checks passed. Repository-wide Step 6 debt remains 166 Ruff findings and 14 strict
mypy findings in the same three untouched files. Independent task, code, and ML/data
reviews closed with no remaining P0, P1, or P2 finding.

Implementation commit `7affd05` moves broad-universe regular-session five-minute
acquisition planning byte-for-byte to
`intraday/datasets/broad_intraday_history.py`. The source Git object remains
`a014735fd60d6ba04d764804854649c752896cf8`; both public functions have one canonical
owner, every old import form is prohibited, and no alias remains. Alpaca SIP
`5Min`/`adjustment=all`, XNYS regular-session bounds, valid existing-corpus
subtraction, truncated-session replanning, current-snapshot broad membership
limitations, point-in-time S&P precedence, explicit fund exclusion, atomic
publication, and the 4 GiB/0.75 GiB memory controls are unchanged.

Both retained plans strictly replay with the same request SHA, policy hash, and
fingerprint `cc7ead87...3b8e`. The latest directory contains 814 sessions, 867,733
eligible symbol-sessions, 419,582 existing sessions subtracted, 448,151 missing
symbol-sessions across 579 symbols, 9,428 acquisition units, and 34,793,778 maximum
expected rows. Its manifest is `ae454a08...5d88a` and authority is
`f372997e...9e6f`; the earlier logically equivalent directory remains untouched. No
provider request, artifact regeneration, training, promotion, serving, or locked-test
access occurred.

Verification passed 213 focused tests and the complete suite with 1,744 passed and
three skipped in 12 minutes. Touched Ruff and strict mypy, compileall, CLI help,
canonical and old-path imports, source parity, both retained-plan replays, diff,
process, and temporary-output checks passed. Repository-wide Step 6 debt remains 166
Ruff findings and 14 strict-mypy findings in the same three untouched files.
Independent task, code, and ML/data reviews closed with no remaining P0, P1, or P2
finding.

Implementation commit `1702991` moves ER1B pre/post-market acquisition planning to
`intraday/datasets/extended_session_context.py`, updates the sole command consumer,
and prohibits every old import form. No alias remains. The original source Git object
before the targeted validation changes was
`63caadaba0364683accb020e9b8b06b2c0319520`.

The migration also closes two independent-review findings. New plans require all
three frozen membership identities to match ER1A: membership Parquet SHA-256,
membership-audit SHA-256, and universe snapshot ID. An explicit `first_session` must
be non-empty canonical `YYYY-MM-DD` and an exact XNYS session; omission is represented
only by `None`. Tests cover all three identity mismatches, weekend and empty suffixes,
DST conversion, and the 66-bar premarket, 48-bar normal postmarket, and 84-bar
early-close postmarket windows. Extended bars remain a separate Alpaca SIP
`5Min`/`adjustment=all` layer and cannot enter regular-session VWAP, EMA, ATR, or RVOL.

Both retained plans strictly replay without regeneration. The full plan remains 804
sessions, 17,688 units, and 47,421,498 maximum rows at fingerprint
`89c91d17...a250`. The suffix plan remains 313 sessions, 6,886 units, and 18,468,096
maximum rows at fingerprint `e005505b...f7999`, manifest `8e753572...3415`, and
authority `ce0842af...52d9`. The completed collection replays all 6,886 units with
zero failures and 1,925,863 rows at manifest `f22147ee...5658`; it remains
`coverage_status=not_evaluated` and `model_data_ready=false`. No provider request,
artifact regeneration, training, promotion, serving, or locked-test access occurred.

Verification passed 217 focused tests and the complete suite with 1,755 passed and
three skipped in 12 minutes 33 seconds. Touched Ruff and strict mypy, compileall, CLI
help, canonical and old-path imports, exact ER1A membership lineage, retained-plan
and collection replay, diff, process, and temporary-output checks passed.
Repository-wide Step 6 debt remains 166 Ruff findings and 14 strict-mypy findings in
the same three untouched files. Independent task, code, and ML/data reviews closed
with no remaining P0, P1, or P2 finding.

Implementation commit `3791541` moves the prospective closed-session SIP collector to
`intraday/datasets/prospective_sip_session.py`, updates the collection command, and
prohibits every old import form without an alias. The original source Git object was
`3ff60848019791e26ada7cb0eaee814d55e37932`.

The migration closes all independent-review findings. Completed-output replay now
matches the requested session, observed membership authority, policy files, effective
configs, and required benchmark set. Fresh collection rejects an in-memory config that
cannot be reconstructed from its recorded policy file. `maximum_units_this_run` is one
parent-wide provider-request budget across both children. Replay also compares the
retained stock membership table with the exact active observed cohort. After the next
XNYS open, an interrupted parent may be finalized only if both child authorities were
already complete and every retrieval timestamp verifies inside `[close + 60 seconds,
next open)`; any incomplete child remains barred from provider access.

The retained `data/raw/prospective_sip_sessions/session_20260820_v1` authority strictly
replays 503 securities and 39,181 SIP five-minute stock rows plus all 13 benchmark ETFs
and 5,070 SIP one-minute rows. Twenty-two sparse securities equal 4.3738%, below the
5% whole-security ceiling. Status remains `source_complete_warmup_ineligible`, with
training, selection, and serving eligibility all false. No provider request, artifact
regeneration, training, promotion, serving, or locked-test access occurred.

Verification passed 249 focused tests and the complete suite with 1,772 passed and
three skipped in 13 minutes 44 seconds. Touched Ruff and strict mypy, compileall,
collection CLI help, canonical/old-path imports, retained real-authority replay, diff,
process, and temporary-output checks passed. Repository-wide Ruff debt is 165 findings;
strict mypy debt remains 14 findings in the same three untouched files. Independent
task, Python-code, and ML/data-design reviews closed with no remaining P0, P1, or P2.

Implementation commit `4a98f29` moves prospective Alpaca broker-action polling and
immutable generation publication to
`intraday/datasets/prospective_broker_actions.py`. This owner is deliberate: the
evidence is bound to the intraday A4.3 security namespace and observed polling cadence;
issuer-event classification remains a downstream catalyst concern. The old
`edge_rebuild/prospective_broker_actions.py` path and every import form are prohibited,
with no alias.

The migration adds historical identity-only replay so retained polls remain verifiable
after the A4.3 transformation changed, while the full current bar loader still rejects
stale rows for training. Fresh collection continues to require the current complete bar
authority. Poll and generation JSON use duplicate-key, non-finite, exact-schema, strict
integer, timestamp, hash, path, count, and causal-lineage checks. Original and ancestor
Windows reparse points are rejected. Cutoff claims commit only after strict poll replay.
Generation publication uses an owned sibling staging directory and atomic rename;
unowned staging is preserved rather than deleted. Every child remains
`production_ready=false`, and top-level training/serving eligibility remains false.

The retained polls strictly replay 76, 460, and 451 observations. The retained
generations strictly replay 536 and 451 revisions. No provider request, artifact
regeneration, training, promotion, serving, or locked-test access occurred. Verification
passed 283 focused integration tests and the final complete suite with 1,810 passed and
three skipped. Touched Ruff, strict mypy, compileall, collection/research CLI help,
canonical/old-path imports, real retained evidence replay, diff, memory, and temporary
output checks passed. Repository-wide Ruff debt remains 165 findings; strict mypy debt
remains 14 findings in the same three untouched files. Independent Python and ML/data
reviews closed with no remaining P0, P1, or P2 finding.

Implementation commit `7419df8` moves prospective analyst-revision classification,
episode construction, coverage, capacity audit, publication, and strict replay to
`intraday/datasets/prospective_analyst_revision_horizon.py`. The command imports the
canonical owner, every old import form is prohibited, and no compatibility alias
remains. Immutable schema identifiers retain their existing values because they name
persisted evidence formats rather than Python ownership.

The hardened publisher supports a valid zero-news horizon, enforces memory before and
after every parent load and derivation, bounds total parent bytes and projected frame
expansion, validates exact JSON and tabular schemas, rejects duplicate/non-finite
metadata and Boolean integers, verifies security identity across generations, and
recomputes provider timestamp collisions and first-seen event time over the complete
horizon. Publication uses an owned sibling staging directory and atomic rename.
Source-event capacity never claims matched market-session capacity, and all outputs
remain explicitly ineligible for training and serving.

Both retained authorities strictly replay without regeneration. The August 17
authority contains 536 revisions, three episodes, and 1,006 coverage rows; the August
21 authority contains 451 revisions, 12 episodes, and 503 coverage rows. Verification
passed 273 focused migration/integration tests, the restored 57-test intraday
development file, and the exact complete suite with 1,846 passed and three skipped.
Touched Ruff, strict mypy, compileall, CLI help, retained-authority replay, diff, and
temporary-output checks passed. Independent Python and ML/data reviewers reported no
remaining P0, P1, or P2 finding. Repository-wide debt is 166 Ruff findings and 14
strict-mypy findings in three untouched intraday dataset files.

Implementation commit `e417960` completes the prediction-data readiness governance
boundary. Readiness orchestration and contracts now live under `governance/readiness`;
strict immutable authority replay lives under `evidence`; promoted-bundle contracts
and verification live under `governance/promotion`. The behavior-named research
command is `audit-prediction-data-readiness`, its policy is
`configs/prediction_data_readiness.toml`, and no readiness compatibility alias remains.

Current authority publication and replay bind exact swing/intraday strategy and proxy
identities, XNYS calendar package/version, costs, folds, dimensions, exclusions,
catalyst channel policy, availability, counts, causal event/assignment projections,
and canonical artifact hashes. Historical-v1 authorities replay without consulting a
mutable current calendar runtime and remain prohibited from current planning. Large
swing, intraday, and catalyst populations are verified and reduced sequentially under
the 4 GiB process limit. Pytest now collects only `tests/`, preventing local data,
models, scratch files, or caches from entering the suite.

Verification passed 354 focused tests with one skip and the exact complete suite with
1,933 passed and three skipped in 18 minutes 41 seconds. Touched Ruff and strict mypy,
compileall, CLI help, retained historical replay, and diff checks passed. Independent
Python and ML/data reviewers closed with no remaining P0, P1, or P2 finding.
Repository-wide debt is 161 Ruff findings and 14 strict-mypy findings in
`intraday/datasets/publisher.py`, `intraday/datasets/bar_dataset.py`, and
`intraday/datasets/dataset_io.py`. No provider request, data regeneration, training,
promotion, serving, or locked-test access occurred.

Implementation commit `fc2a32e` completes prediction-serving ownership. Bundle loading,
API-facing contracts, prediction orchestration, snapshots, outcome-intent registration,
investment replay, swing feature construction, and swing inference now have canonical
owners under `core`, `serving`, and `swing`. Removed top-level and `edge_rebuild` paths
are absent and guarded against reintroduction; no compatibility alias remains.

The checkpoint also closes serving correctness gaps. Strict JSON rejects duplicate and
non-finite values. Bundle and feature-manifest reads reject reparse paths, bind bytes to
their hashes, and prevent path or same-file races. Direct and unified prediction paths
reject future or changed model generations. Swing readiness binds exact drift, policy,
market, liquidity, and ten-session outcome-policy identities. Swing maturation uses the
triple barrier and records `target_first`, `stop_first`, or `timeout`; intraday keeps its
separate minute-horizon calibration and managed-outcome requirements. Invalid or
abstained rows cannot create maturation intents.

Verification passed 442 focused tests with two skipped and the exact complete suite
with 1,995 passed and three skipped in 18 minutes 42 seconds. Changed-file Ruff, strict
mypy over 46 source files, compileall, CLI help, staged diff, and import-path scans
passed. Independent architecture and ML/governance reviewers reported no remaining P0,
P1, or P2 finding. Repository-wide debt is 130 Ruff findings and 14 strict-mypy findings
in `intraday/datasets/publisher.py`, `intraday/datasets/bar_dataset.py`, and
`intraday/datasets/dataset_io.py`. No provider call, data regeneration, training,
promotion, serving process, or locked-test access occurred.

Implementation commit `880f2a8` completes outcome and drift governance ownership.
Prediction selection now belongs to `modeling`; swing and intraday maturation belong
to their horizon evaluation packages; and outcome contracts, durable persistence,
performance monitoring, feature drift, and drift policy belong to `governance`.
Removed top-level modules and the duplicate swing policy module have no aliases and are
blocked by architecture tests.

The checkpoint also closes seven independently reproduced P1 failures. Stored
observations and matured outcomes are rebound to their immutable intent every time
they are read. Outcome evidence and storage identities are verified. Swing maturation
accepts only canonical daily bars and intraday maturation only canonical one-minute
bars. Label economics and dynamic execution-policy economics are separately retained.
Feature drift requires the complete unique feature-name set and exact reference-profile
identity from the active model. Drift assessment state/actionability pairs are strict,
and serving accepts only the configured drift-policy SHA-256. Both independent
reviewers confirmed their findings closed.

Verification passed 339 focused tests and the exact full suite with 2,033 passed, three
skipped, and 132 warnings in 27 minutes. Changed-file Ruff passed; strict mypy passed
over 30 source files; compileall and diff checks passed. A sampled full-suite working
set was approximately 0.13 GiB, and no Python process remained at closure. No provider
request, data regeneration, training, promotion, serving process, or locked-test access
occurred.

## Exact Next Checkpoint

Exact next checkpoint: **Reconcile returns, capital and SPY accounting**. The first
approved checkpoint is complete in `3b2bff5`; implement only the second, verify it,
push its code and documentation, then stop. Preserve historical artifacts. No real
training or further protected outcome access is permitted. The already exposed
July-2025 to June-2026 calendar cannot supply a new untouched final test.

Read in order: `AGENTS.md`, the new current section of the active plan, this handoff's
current findings, the feature audit, temporal/training/strategy configs, and the named
evaluation owners. Do not restart unrelated structural cleanup or intraday work.
Full Ruff/mypy now pass; older debt statements below/above are historical. Code
checkpoint verification still follows the covenant. Rollback anchor is `3b2bff5`.

Research artifacts to inspect without retraining:

- `data/models/swing/technical/{evaluation.json,model_card.json,_manifest.json}`:
  historical technical evidence; not a current eligible model.
- `data/models/swing_directional_broker_action_specialists_dev_20260820_v1`:
  development specialist rejection evidence.
- `data/features/edge_rebuild_swing_technical_panel_20190709_20260708_v1`:
  current technical authority/request; no full replay was performed this turn.
- `data/features/swing_broker_action_ablation_20190709_20260708_v2`:
  corrected matched event/technical profiles.
- `data/external/sec_filings_20190709_20260708_v1/_manifest.json`:
  SEC inventory, not proof of original-text or prospective availability.

Use existing data first. Parallel reviewers and disjoint light implementation tasks
are permitted; heavy tests and training remain sequential. Stop and close owned
workers/reviewers when their bounded work is complete. No deployment, provider
collection, real model training, or promotion occurred in the contract checkpoint.
