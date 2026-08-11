# Active Edge Rebuild Handoff

Status: active

Last updated: 2026-08-11

Repository: `C:\project\market-predictor`

Branch: `er-intraday-refactoring`

Last completed implementation commit: `9c8aa5b` (`Fail fast on malformed precision reviews`)

## Purpose

Continue the four-model prediction rebuild: swing baseline, swing event-driven,
intraday baseline, and intraday event-driven. A0 through A2 implementation are closed.
A3 issuer-event specialist authority is the active checkpoint. Do not open a locked
test or claim model quality until a preregistered candidate passes both validation
scopes.

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
- The swing baseline consumes only `technical_market`. Catalyst is a confirmation and
  explanation overlay for this family; `catalyst_full` belongs to the separate A3
  event-driven family.
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
- Serving selects the frame from the signed model family: baseline uses
  `technical_market`; event-driven uses `catalyst_full`.
- Current Finviz snapshots are not historical point-in-time authority. Quality,
  profitability, investment, valuation, and estimate-revision groups remain blocked.
- No real candidate was trained or promoted in A2, so AUC and economics are unchanged.
- Focused A2 suite: 59 passed, 1 skipped. Canonical suite: 1,110 passed, 2 skipped.
  Tracked Ruff, strict mypy on five changed production modules, compileall, governance
  hash replay, and diff checks passed.

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
  is now complete. No A3 training dataset, model, locked-test metric, or promotion
  exists.

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

## A1: Label and Leakage-Control Verification

- Consolidated A1 label, dataset, outcome, and trainer tests: 87 passed, 1 skipped;
  the final corrected swing feature-time poison test also passed.
- Canonical full suite: 1,110 passed, 2 skipped.
- Ruff passed for every tracked Python file.
- Strict mypy passed for all five changed production modules.
- Compileall passed using an external bytecode cache; `git diff --check` passed.
- Repository-wide tracked strict mypy still has six pre-existing errors in
  `scripts/build_intraday_v3.py`, `scripts/build_swing_v12.py`, and
  `scripts/train_intraday_v3.py`. They are not part of A1 and require the later bounded
  legacy-script cleanup decision.
- A root ignored `scratch_test.py` contains NUL bytes, so bare `pytest -q` cannot
  collect. The authoritative `pytest tests -q` run passed. Do not delete the scratch
  file without the planned reference/retention cleanup.
- Two pre-existing uvicorn research-workbench Python processes remain at about 60 MiB
  combined. No test or training worker remains.

## Model State

| Model family | Current state | Next valid work |
| --- | --- | --- |
| Swing baseline | A2 trainer complete; prior candidates rejected; no new run or promotion | Preserve the frozen technical contract until a governed training run is approved |
| Swing event-driven | Prior broad catalyst candidates rejected | A3 event-family authorities and specialists |
| Intraday baseline | V2 rejected; V3 z-score lineage invalid | A4 cohort-correct market/microstructure rebuild |
| Intraday event-driven | No eligible candidate | A5 verified event cohorts |

`ROC-AUC >= 0.60` is a locked-test diagnostic, not a training objective or permission
for repeated locked-test tuning. Promotion also requires ranking, calibration,
after-cost benchmark-relative economics, drawdown, turnover, capacity, stability, and
coverage.

## Exact Next Step: A3.4 - Build Identical-Decision Ablations

1. **A3.4 - Build identical-decision ablations:** technical-only, analyst-revision-only,
   and technical-plus-analyst-revision must share the exact decisions and labels.
   Every blocked event family must be absent, not encoded as zero.
2. **A3.5 - Train and evaluate specialists:** train only after the complete causal
   horizon and precision gates verify; remain abstaining outside verified cohorts.

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
