# Active Edge Rebuild Handoff

Status: active

Last updated: 2026-08-27

Repository: `C:\project\market-predictor`

Branch: `er-intraday-refactoring`

Last completed implementation commit: `2b9e195` (`Move issuer event foundations into catalysts`)

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

Implementation commit `9244893` is pushed. The working tree was clean before this
documentation closure. No scratch script, provider data, model artifact, or evidence
authority was created, rewritten, or deleted by the SEC filing package move.

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

Exact next checkpoint: complete **swing issuer-event family cohort and decision
authority migration**. Move the swing-specific cohort dataset authority from
`edge_rebuild` to `swing/datasets` and its decision-time feature authority to
`swing/features`. Keep reusable issuer classification and attribution in `catalysts`.
Preserve cohort eligibility, source authorization, causal availability, coverage,
artifact schemas, hashes, replay, and all decision-time values. Remove old modules and
imports without aliases, add dependency/removal guards, run focused parity verification,
and obtain the assigned reviewer's diff acceptance before checkpointing.
