# Active Edge Rebuild Handoff

## Current checkout and checkpoint (November 22)

Use `C:/project/market-predictor`, branch `main`. Reviewed implementation state:
`cb01c63`; latest implementation commit `43e63cd`. The November 22 review changes
documentation only. No collector, trainer, scheduled task or production release
was run or changed during that review.

Step 8 remains in progress. The authoritative open review is
[feature audit R1–R10](feature_engineering_audit_20260801.md#1-review-of-research-lane-changes-november-22).
Its concrete findings supersede previous claims of exact batch/live parity and
an untouched November forward ledger. Do not reconstruct or overwrite existing
evidence to make those claims pass.

## Retained authorities and interpretation

- The [active plan](../active_edge_rebuild_plan.md) owns completed-step history,
  results and artifact locators. This handoff does not duplicate those tables.
- Historical holdout: `data/research/research_lane/sealed_holdout_20240529_20260708_20261121T193257Z`
  (`trials.jsonl` line 14). Its inspected `2024-05-29..2026-07-08` period is now
  development history. No product specification was promoted against SPY.
- Saved research model: `data/research/research_lane/models/mid_small_beta_neutral_sec_xgb`;
  manifest SHA-256 `cd18356701f35b44c884d9f587a8e14ed7473878f12f75e2ec27548b4d8ebf04`.
  The review verified that manifest identity and all six model payload hashes.
  Training completed November 21 at 21:41 UTC. Research-only; no production promotion.
- `data/research/research_lane/forward_records/` contains five November 16–20
  session snapshots. The November 16, 17 and 18 snapshots were recorded after
  their next-session entries. None of the five contains `sec_xgb` scores; each
  has 1,000 scored stocks with sector `Unknown`. Do not pool these as prospective
  performance of the persisted SEC model. Preserve them as recorded evidence.
- The Windows task `MarketPredictorForwardScoringPostClose` was documented as
  installed by `43e63cd`, weekdays at 22:35 local time. Its registered state was
  not independently inspected during this review. The runner remains unchanged.
- `C:` to `D:` storage migration and R7.8 deployment/security work remain deferred.

## Next action and verification

Exact next checkpoint: start item 1 of the
[Gemini implementation order](../active_edge_rebuild_plan.md#gemini-implementation-order-for-step-8).
Compare each of the five snapshot publication times with its next tradable open,
check its score schema, and enforce those admission checks in the forward evaluator.

Read `AGENTS.md`, the active plan and feature audit first, then
`research_lane/forward_record.py`, `sec_catalyst_model.py`, `forward_evaluate.py`,
`beta_neutral.py`, `benchmark_tilt.py` and `opra_options.py` under `src/market_predictor/`.
Reuse the existing transforms, policy accounting, artifact verification and
`EvidencePaths` rather than adding more alternative implementations/configuration.

November 22 review verification: 29 focused unit tests and 21 continuity/governance/
strategy-contract tests passed; research-lane Ruff passed; strict mypy passed for
28 source files. Actual retained metadata and model hashes were inspected.
Historical training/returns were not independently rerun. Changed documentation
links, control characters, retained artifact pins and `git diff --check` passed.
The six documentation-file changes are local review edits, not a closed code
implementation checkpoint; source code and saved evidence remain unchanged.
