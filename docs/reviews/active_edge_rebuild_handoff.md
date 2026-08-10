# Active Edge Rebuild Handoff

Status: active

Last updated: 2026-08-10

Repository: `C:\project\market-predictor`

Branch: `er-intraday-refactoring`

Last completed implementation commit: `cb2aba5` (`Build governed swing baseline ablations`)

## Purpose

Continue the four-model prediction rebuild: swing baseline, swing event-driven,
intraday baseline, and intraday event-driven. A0 through A2 implementation are closed.
A3 issuer-event specialist authority is the active checkpoint. Do not open a locked
test or claim model quality until a preregistered candidate passes both validation
scopes.

This repository produces prediction intelligence and abstention. Alerts, orders,
positions, portfolio risk, and execution remain in `trading_flow`.

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

## A2 Verification

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

## A1 Verification

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

## Exact Next Step: A3

1. Inventory issuer-event histories for earnings/guidance, SEC material events,
   analyst revisions, offerings, M&A, regulatory decisions, and product events.
2. Admit a family only when publication/acceptance time, first availability, issuer
   identity, relevance, source coverage, and exact decision assignment replay.
3. Publish event-family cohort counts by calendar period, sector, security, and event
   type. Unknown coverage remains null and causes exclusion or abstention.
4. Preregister technical-only, event-only, and technical-plus-event comparisons on
   identical decisions. Train no specialist until its complete causal horizon verifies.

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
