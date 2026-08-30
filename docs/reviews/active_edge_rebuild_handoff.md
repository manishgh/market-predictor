# Active Edge Rebuild Handoff

Status: active

Last updated: 2026-08-30

Repository: `C:\project\market-predictor`

Branch: `er-intraday-refactoring`

Last completed implementation commit: `61f6f4c` (`Refactor label outcomes into domain packages`)

## Purpose

Continue the four-model prediction rebuild: swing baseline, swing event-driven,
intraday baseline, and intraday event-driven. A0 through A2 implementation are closed.
A3 issuer-event specialist development is complete with no candidate. A4.1 collector,
A4.3 bar-only causal dataset publication, and A4.4 continuation/reversion development
are complete. Both A4.4 hypotheses were rejected. A5.1 is also complete and blocked:
the available event history is retrospective and exact security identity overlap is
too small. The first real closed-session SIP authority now strictly replays for
`2026-08-20`, but it is only session 1 of the required 20-session warm-up. A separate
August 21 observed-time event chain also strictly replays with 451 identity-bound
observations and 12 qualifying analyst episodes. It is not joined to the interrupted
August 17 chain. Do not train A5.2, open a locked test, or claim model quality.

This repository produces prediction intelligence and abstention. Alerts, orders,
positions, portfolio risk, and execution remain in `trading_flow`.

Checkpoint names: A0 restores research integrity; A1 verifies labels and leakage;
A2 builds the technical swing baseline; A3 builds catalyst-driven swing specialists;
A4 builds the technical intraday baseline; A5 builds catalyst-driven intraday
specialists; A6 performs locked evaluation and promotion.

## Verified State

- No promoted serving bundle exists. Production prediction paths must fail closed.
- Reddit and Seeking Alpha remain retired and prohibited.
- Swing decisions begin on `2019-07-09`; earlier bars are warm-up only.
- Intraday estimator input is the ordered A4.3 bar-only technical contract, sampled on
  fixed five-minute cohorts from causal completed evidence. The V3 z-score lineage is
  invalid and prohibited.
- `swing_features.py` has been refactored into a `FeaturePipeline` orchestrator, with logic decoupled into `swing_pipeline_steps.py`, `swing_filters.py`, and `swing_catalyst_features.py`. Shared cross-cutting utilities reside in `edge_rebuild/utils/`.
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

## A4.3: Bar-Only Technical Intraday Authority

- Commit `8a76ec1` publishes the fixed-cohort bar-only dataset authority without a
  provider download or any trade/quote/microstructure input.
- The immutable five-minute projection contains 43,226 selected stock-sessions and
  3,364,335 rows: 43,132 stock-session pairs are complete and 94 incomplete pairs are
  retained as explicit coverage metadata.
- The immutable dataset contains 794 sessions, 501 tickers, 3,095,688 rows, and
  1,365,015 eligible rows. Aggregate publication peak upper bound was 2.218 GiB.
- Dataset request SHA-256:
  `83820269d80019a46754aa451c1f1e13773995a889a51e605595511315af4bb2`.
  Transformation SHA-256:
  `0da898cc6fd3c1e933406ce07f24de197fc1fa34c4a909c4b9c4a28e2e96f3f6`.
- The bound audit is
  `data/reports/edge_rebuild_intraday_bar_only_causal_20260814_v1_audit.json`. It
  replays the exact dataset/projection manifest, authority, and inventory hashes and
  reports zero duplicate-decision, causal-cutoff, label-availability, eligible-ATR,
  ordered-feature-hash, schema-identity, and prohibited-feature violations.
- Two independent re-reviewers reported no remaining medium-or-higher findings.
- No model was trained, no locked test was opened, and no serving bundle was promoted.

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

Exact next checkpoint: move `edge_rebuild/volume_bars.py` byte-for-byte to
`intraday/datasets/volume_bars.py` and rename
`test_edge_rebuild_volume_bars.py` to `test_intraday_causal_volume_bars.py`. Update
`intraday/datasets/publisher.py`, `dataset_v2.py`, and `bar_dataset.py`, including the
transformation-module identity, with direct canonical imports. Before deleting the old
module, scan ignored `models/` and `data/` manifests and joblib/pickle files for
`market_predictor.edge_rebuild.volume_bars` and `VolumeBarBuildResult`; any match blocks
the move and no artifact may be rewritten.

Freeze source SHA-256
`93213b79c6d3de0f2463821f3228c33408857877f8a775bfc93ef4e2bb96f900`,
`VOLUME_BAR_COLUMNS` hash
`55a343087cce34eb04f438c55cce369b0f35ffbdf1e552b675ff4507fc02f849`,
`AUDIT_COLUMNS` hash
`a840bf2a667f85d3a78f3e4747282e64b4ee26128878587963ba86c87d8f2388`,
representative bar/audit schemas, dtypes, ordering and hashes, future-prefix causality,
session/ticker isolation, threshold and incomplete-remainder behavior, model
eligibility, complete transformation identity, new pickle owner, and the 4 GiB memory
limit with 0.75 GiB headroom. Exclude all adjacent history, coverage, label, feature,
training, serving, and command migrations. No alias, schema change, regenerated
artifact, or behavioral change. Rollback anchor: `61f6f4c`.
