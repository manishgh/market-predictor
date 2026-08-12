# Active Edge Rebuild Handoff

Status: active

Last updated: 2026-08-12

Repository: `C:\project\market-predictor`

Branch: `er-intraday-refactoring`

Last completed implementation commit: `58eed6c` (`Build governed swing broker specialists`)

## Purpose

Continue the four-model prediction rebuild: swing baseline, swing event-driven,
intraday baseline, and intraday event-driven. A0 through A2 implementation are closed.
A3 issuer-event specialist development is complete with no candidate. A4 technical
intraday rebuilding is the next checkpoint. Do not open a locked test or claim model
quality until a preregistered candidate passes both validation scopes.

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
- Intraday estimator input remains the 44-feature causal technical contract. The V3
  z-score lineage is invalid and prohibited.
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
- Repository suite: 1,207 passed and 2 skipped.
- Coordinated request, feature, all-profile label, dtype, partition, global identity,
  source-coverage, causal-window, and governance-hash poison tests pass.
- The real materialization and event publication remained below the 5 GiB limit;
  observed working memory was below 2 GiB during A3.4.

## Model State

| Model family | Current state | Next valid work |
| --- | --- | --- |
| Swing baseline | A2 trainer complete; prior candidates rejected; no new run or promotion | Preserve the frozen technical contract until a governed training run is approved |
| Swing event-driven | Rating-change and coverage specialists trained in development; all rejected | Preserve rejection evidence; do not open locked test or serve |
| Intraday baseline | V2 rejected; V3 z-score lineage invalid | A4 cohort-correct market/microstructure rebuild |
| Intraday event-driven | No eligible candidate | A5 verified event cohorts |

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

## Exact Next Step: Build A4 Technical Intraday Authority

Build the cohort-correct intraday market and microstructure authority defined in A4.
Do not tune A3 against its validation result and do not open the A3 locked test.

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
