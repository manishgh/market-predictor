# Active Edge Rebuild Handoff

## Current checkout and checkpoint (November 22)

Use `C:/project/market-predictor`, branch `main`. Latest implementation commit: `716f332` (closing all eight Step 8 review items `R1–R10` across `71a79d7..716f332`).

Step 8 remains in progress on its operational prospective milestones:
- **Milestone 1 (Software & retrospective-replay verification of `R1–R10`):** Complete (`716f332`). See [Step 8 Resolution Ledger](feature_engineering_audit_20260801.md#step-8-resolution-ledger-for-r1r10-november-22-implementation-head-716f332) and [active plan](../active_edge_rebuild_plan.md#step-8-milestone-status-november-22).
- **Milestone 2 (First qualifying prospective prediction):** Pending (`0/5` retained November 16–20 snapshots qualify).
- **Milestone 3 (First matured prospective portfolio outcome):** Pending (`open[t+11]` for `10d` candidates and `open[t+22]` for `21d` candidate after Milestone 2).

## Retained authorities and blockers

- **Persisted research model (`research_only = true`, `actionable = false`):** `data/research/research_lane/models/mid_small_beta_neutral_sec_xgb` (`_manifest.json` SHA-256 `cd18356701f35b44c884d9f587a8e14ed7473878f12f75e2ec27548b4d8ebf04`, feature contract SHA-256 `d244cea55bde7b79c22167da6bc83ad315f5165b33fbb3a39826b32d022b868e`).
- **Original November snapshots (preserved read-only):** `data/research/research_lane/forward_records/2026-11-16..2026-11-20`. Audited in `data/research/research_lane/forward_evaluations/prospective_audit_step8` (`3/5` recorded after entry; `0/5` qualify for `mid_small_beta_neutral_sec_xgb`).
- **Retrospective CLI parity replay (`non-prospective diagnostic`):** `data/research/research_lane/forward_records_retrospective_replay_step8/{2024-04-16,2024-04-30}` and `data/research/research_lane/forward_evaluations/retrospective_replay_step8` (`0.0` max abs score/feature/return difference vs. batch across `741` `cik:` stocks).
- **Current live blocker for Milestone 2 (`DataReadinessError`):** `record-forward-research-scores --live-prospective` fails closed for dates `> 2026-08-21` because the latest point-in-time S&P 500 membership snapshot (`data/raw/index_membership/sp500_observed_20260821T071500Z_v7`) ends at `2026-08-21` and SEC `extended_roster_spans.parquet` ends at `2026-07-08`.
- **Deferred items:** `C:` to `D:` storage migration and R7.8 deployment/security work remain deferred. No model is promoted to production or API `V1`.

## Next action and verification

Exact next checkpoint: refresh point-in-time S&P 500 membership (`> 2026-08-21`), SEC `roster_spans.parquet` (`> 2026-07-08`), and trailing SEC `company_facts` / `8-K` `2.02` coverage through the next live trading session, then publish the first qualifying post-close prospective snapshot before its next `XNYS` open (`09:30 ET`) and verify `"status": "pending_entry_bar"` via `evaluate-forward-research-scores`.

Read `AGENTS.md`, `docs/active_edge_rebuild_plan.md`, and `docs/reviews/feature_engineering_audit_20260801.md` first, then `prospective_contracts.py`, `live_cohort.py`, `forward_record.py`, `forward_evaluate.py`, `sec_catalyst_model.py`, `opra_options.py`, and `benchmark_tilt.py` under `src/market_predictor/research_lane/`.

Verification completed at `716f332`:
- `.venv\Scripts\pytest.exe tests/test_research_lane*.py tests/test_active_continuity_documents.py tests/test_strategy_governance_contracts.py` (`135 passed`).
- `.venv\Scripts\ruff.exe check src/market_predictor/research_lane src/market_predictor/commands/research_lane.py tests/test_research_lane*.py` (`All checks passed!`).
- `.venv\Scripts\mypy.exe --strict src/market_predictor/research_lane src/market_predictor/commands/research_lane.py` (`Success: no issues found in 30 source files`).
- `git diff --check` clean; all retained snapshot, scoreboard, governance, and model-bundle SHA-256 pins verified intact.
