# Active Edge Rebuild Handoff

Status: active

Last updated: 2026-08-10

Repository: `C:\project\market-predictor`

Branch: `er-intraday-refactoring`

Last completed implementation commit: `7b61873` (`Freeze causal outcome diagnostics`)

## Purpose

Continue the four-model prediction rebuild: swing baseline, swing event-driven,
intraday baseline, and intraday event-driven. A0 and A1 are closed. A2 swing-baseline
research is the only active checkpoint. Do not train a candidate or open a locked test
until the A2 feature authority and preregistered validation contract are complete.

This repository produces prediction intelligence and abstention. Alerts, orders,
positions, portfolio risk, and execution remain in `trading_flow`.

## Verified State

- No promoted serving bundle exists. Production prediction paths must fail closed.
- Reddit and Seeking Alpha remain retired and prohibited.
- Swing decisions begin on `2019-07-09`; earlier bars are warm-up only.
- Intraday estimator input remains the 44-feature causal technical contract. The V3
  z-score lineage is invalid and prohibited.
- Swing technical and Alpaca ablation profiles must contain identical immutable
  decision and label identities. Training cannot rewrite groups or labels.
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
| Swing baseline | Prior candidates rejected; no promotion | A2 compact point-in-time feature authority and ablation |
| Swing event-driven | Prior broad catalyst candidates rejected | A3 event-family specialists after A2 |
| Intraday baseline | V2 rejected; V3 z-score lineage invalid | A4 cohort-correct market/microstructure rebuild |
| Intraday event-driven | No eligible candidate | A5 verified event cohorts |

`ROC-AUC >= 0.60` is a locked-test diagnostic, not a training objective or permission
for repeated locked-test tuning. Promotion also requires ranking, calibration,
after-cost benchmark-relative economics, drawdown, turnover, capacity, stability, and
coverage.

## Exact Next Step: A2

1. Inventory which compact swing baseline groups already have point-in-time authority
   from `2019-07-09` through the frozen end date: market/sector residual momentum,
   volatility, liquidity/turnover, quality, profitability, investment, valuation, and
   estimate revisions.
2. Update the vertical acceptance matrix. Reject any group lacking full-horizon source,
   availability, batch/live, schema, and poison evidence; do not partially backfill.
3. Preregister sequential ablations and their validation-only acceptance rules. Keep
   the locked test unopened.
4. Build or replay only accepted authorities, then train the regularized linear
   baseline before bounded tree ranker/regressor candidates.

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
