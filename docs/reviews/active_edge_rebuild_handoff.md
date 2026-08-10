# Active Edge Rebuild Handoff

Status: active

Last updated: 2026-08-10

Repository: `C:\project\market-predictor`

Branch: `er-intraday-refactoring`

Last completed implementation commit: `e168482` (`Restore fail-closed model research invariants`)

## Purpose

Continue the four-model prediction rebuild: swing baseline, swing event-driven,
intraday baseline, and intraday event-driven. The next objective is A1 label,
metric, and negative-control governance. Do not train another candidate or open a
locked test until A1 closes.

This repository produces prediction intelligence and abstention. Alerts, orders,
positions, portfolio risk, and execution remain in `trading_flow`.

## Verified State

- No promoted serving bundle exists. Production prediction paths must fail closed.
- Reddit and Seeking Alpha remain retired and prohibited.
- Swing model decisions begin on `2019-07-09`; earlier bars are warm-up only.
- Existing rejected swing and intraday artifacts remain rejection evidence, not
  serving fallbacks.
- A0 restored monthly intraday partition metadata verification.
- A0 added an explicit invariant that every temporal and unseen-security scope must
  pass its economic gate before a swing validation threshold is eligible.
- A0 removed five intraday cross-sectional z-score columns that were declared in the
  model schema but lacked a valid contemporaneous decision-cohort transformation.
  Artifacts requiring those columns are invalid and cannot be replayed or served.
- The active intraday technical contract contains 44 causal normalized features.
- Missing catalyst authority is not converted to zero-valued model inputs. Catalyst
  remains outside the broad intraday estimator until a preregistered causal ablation
  and complete historical/live authority pass.
- Sequential intraday development training retains only the current selected and
  auditable ledgers in memory and does not mutate the loaded immutable dataset frame.
- Intraday memory remains capped at 4 GiB; swing candidate training remains capped
  at 5 GiB.

## A0 Verification

- Focused dataset/features: 41 passed.
- Focused training/contracts/authority: 104 passed, 1 skipped.
- Combined focused evidence: 145 passed, 1 skipped.
- Full suite: 1,102 passed, 2 skipped. Three failures were caused solely by denied
  creation of `data/runtime/heavy-job.lock`; all three passed when rerun with normal
  repository write access.
- Ruff check passed on all changed Python and test files.
- Strict mypy passed on five changed production modules.
- Compileall passed with a writable external bytecode cache.
- `git diff --cached --check` passed before commit.
- Two pre-existing uvicorn research-workbench processes remained running at about
  67 MiB combined; no test or training worker was left running.

## Model State

| Model family | Current state | Next valid work |
| --- | --- | --- |
| Swing baseline | Prior candidates rejected; no promotion | A1 target/metric controls, then A2 compact feature ablation |
| Swing event-driven | Prior broad catalyst candidates rejected | A1 event-label controls, then A3 event-family specialists |
| Intraday baseline | V2 rejected; incomplete z-score lineage invalidated | A1 exact target controls, then A4 cohort-correct microstructure rebuild |
| Intraday event-driven | No eligible candidate | A1 event availability controls, then A5 verified event cohorts |

`ROC-AUC >= 0.60` is a locked-test binary diagnostic, not the sole objective and
not permission for repeated test tuning. Promotion still requires calibration,
ranking value, benchmark-relative economics after costs, drawdown, turnover,
capacity, stability, and explicit coverage.

## Exact Next Step: A1

1. Inventory the current swing and intraday label/evaluator implementations and
   identify the single canonical evaluator for each executable horizon.
2. Freeze comparable binary diagnostics alongside economic targets:
   5/10-session benchmark-relative swing return and 30-minute/session-managed
   intraday return after costs.
3. Add poison tests for label shuffle, one-period feature shift, future timestamp,
   duplicate events, overlapping labels, and survivorship/membership leakage.
4. Prove stock, SPY, QQQ, and sector returns use identical executable intervals and
   that costs are applied once.
5. Update the feature audit, run the full verification battery, then checkpoint A1.

## Source Boundary

| Source | Permitted role |
| --- | --- |
| Alpaca SIP/all bars, trades, quotes | estimator market and microstructure data after complete causal backfill |
| Alpaca direct ticker news | ticker event data after exact attribution and availability verification |
| SEC issuer filings | separately ablated issuer event family after causal backfill |
| Finviz Elite | screening/current metadata; ticker news requires its own historical causal authority before estimator use |
| Verified global/sector sources | separate context overlay unless independently preregistered and ablated |
| Reddit | prohibited |
| Seeking Alpha | prohibited |

Every new model feature must complete source authority, full-horizon historical
backfill, shared batch/live transformation, ordered schema/hash, training,
validation, serving representation, and parity/poison/tamper tests before use.

## Working Tree Warning

The worktree still contains pre-existing untracked diagnostics, scratch Parquet
files, ad hoc scripts, and unfinished experimental modules from the interrupted
external agent. They were not committed in A0. Several scripts directly patch or
copy Parquet authorities and are prohibited by `AGENTS.md`. Do not execute, stage,
or treat them as evidence. Cleanup requires a reference scan and a separate bounded
checkpoint; do not delete raw or governance-bound data by assumption.

## Files To Read

1. `AGENTS.md`
2. `docs/active_edge_rebuild_plan.md`
3. `docs/reviews/active_edge_rebuild_handoff.md`
4. `docs/reviews/feature_engineering_audit_20260801.md`
5. `docs/model_training_validation_protocol.md`
6. Current `git status` and recent commits

## Do Not Do

- Do not train or promote from the invalidated intraday z-score lineage.
- Do not open locked tests to guide feature or hyperparameter selection.
- Do not weaken economic, sector, causality, memory, or integrity gates.
- Do not fill missing news, catalyst, quote, or source coverage with zero.
- Do not add a feature without complete historical backfill for its model horizon.
- Do not use an LLM sentiment score as a direct return predictor without causal
  return-conditioned ablation.
- Do not expose rejected candidates through the production prediction API.
- Do not execute scratch scripts that mutate Parquet or patch lineage hashes.
