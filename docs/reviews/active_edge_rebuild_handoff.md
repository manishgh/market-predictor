# Active Edge Rebuild Handoff

Status: active

Last updated: 2026-08-15

Repository: `C:\project\market-predictor`

Branch: `er-intraday-refactoring`

Last completed implementation commit: `42ebe5f` (`Extend point-in-time S&P membership lineage`)

## Purpose

Continue the four-model prediction rebuild: swing baseline, swing event-driven,
intraday baseline, and intraday event-driven. A0 through A2 implementation are closed.
A3 issuer-event specialist development is complete with no candidate. A4.1 collector,
A4.3 bar-only causal dataset publication, and A4.4 continuation/reversion development
are complete. Both A4.4 hypotheses were rejected. A5.1 is also complete and blocked:
the available event history is retrospective and exact security identity overlap is
too small. The prospective Alpaca authority and current-membership extension code are
verified, but no live poll authority exists. The August 15 membership data checkpoint
is `environment_pending` until that New York publication day has closed. Do not train
A5.2, open a locked test, or claim model quality.

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
- Repository suite: 1,374 passed and 2 skipped after the membership-extension review.
- Tracked Ruff, strict mypy across 227 source files, compileall, and final A5.1
  real-authority strict replay pass.
- Coordinated request, feature, all-profile label, dtype, partition, global identity,
  source-coverage, causal-window, and governance-hash poison tests pass.
- The real materialization and event publication remained below the 5 GiB limit;
  observed working memory was below 2 GiB during A3.4.

## Model State

| Model family | Current state | Next valid work |
| --- | --- | --- |
| Swing baseline | A2 trainer complete; prior candidates rejected; no new run or promotion | Preserve the frozen technical contract until a governed training run is approved |
| Swing event-driven | Rating-change and coverage specialists trained in development; all rejected | Preserve rejection evidence; do not open locked test or serve |
| Intraday baseline | V2 rejected; V3 invalid; A4.4 continuation and reversion both rejected | Preserve evidence; no serving or future-holdout access |
| Intraday event-driven | A5.1 blocked; prospective collector verified but no live authority/horizon exists | After the August 15 New York day closes, publish the strict membership extension and begin observed polling |

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
- Parent polls must precede the child and use the same namespace, membership authority,
  and registry. Replay is iterative, not recursive. Identity changes remain quarantined
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
- Live status is `environment_pending`. The earlier HTTP 401 did not prove absent
  credentials. Market Predictor's `.env` currently loads both Alpaca values through
  `Settings.has_alpaca=true`, and `ALPACA_STOCK_FEED=sip`. No live evidence has yet
  been published.
- The available membership authority ends `2026-07-08`. Using it for a later poll
  correctly yields `membership_authority_stale` for every symbol. The first August 15
  extension artifacts were collected at `2026-08-15T18:29Z`, before that New York
  publication day ended, and are invalid under `42ebe5f`; do not use them.

- SEC evidence `data/raw/index_membership/sec_xom_identity_20260815_v1` verifies XOM
  as CIK `0000034088`. The reviewed anchor is
  `data/universe/sp500_current_20260815_sec_reviewed_v1.csv`, SHA-256
  `d1171b3aef900ddf856d1c22d1522d1e42483232a32640922bcf99b97898acba`.

Exact next action: after `2026-08-16T04:00:00Z`, recollect the August 15 official S&P
archive into a new immutable directory, rebuild events and transitions, and publish a
new membership extension using the reviewed anchor and the July 8 base authority.
Strictly replay the complete authority and exact base prefix. Then run one Alpaca poll
with the A4.3 dataset and stable registry, strictly reload it, and inspect coverage and
memory before scheduling polls no more than 120 seconds apart. Do not run A5.2 until a
newly frozen prospective horizon meets the existing capacity floors.

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

## Working Tree Warning

Pre-existing untracked diagnostics, scratch Parquet files, ad hoc scripts, and
unfinished experimental modules remain. Several scripts directly patch Parquet
authorities or lineage hashes and are prohibited by `AGENTS.md`. Do not execute,
stage, or treat them as evidence. Cleanup requires a separate bounded reference scan;
do not delete raw or governance-bound data by assumption.

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
