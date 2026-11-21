# Active Edge Rebuild Handoff

## 1. Current Repository & Branch State (November 21)

- **Checkout & Branch:** `C:/project/market-predictor` on `main` (single canonical checkout per `AGENTS.md`; latest implementation commit `43e63cd`).
- **Unsealed Historical Window (`2016-01-04..2026-07-08`):** `2024-05-29..2026-07-08` (`528` sessions) was evaluated once in Step 6 (`c484658`, `data/research/research_lane/sealed_holdout_20240529_20260708_20261121T193257Z`, `trials.jsonl` line `14`) with all `<= 2024-05-28` artifacts left byte-for-byte untouched. That window is now **permanently development history**; the **prospective live forward ledger (`2026-11-16+`)** is the sole untouched out-of-sample test clock.
- **Heavy-Job Lease:** `data/runtime/heavy_job.lease` absent (`0` background workers).

## 2. Authoritative Research-Lane State (Steps 5c–6)

1. **Step 6 One-Time `2024-05-29..2026-07-08` Holdout Verdict vs. `SPY` (`c484658`, *research labels* on adjusted prices, net of `20 bps` post-drift turnover):**
   - **Pre-Declared Core-Satellite Candidates (`0` promoted vs. `SPY`):**
     - `90_10__spy__R1_backwardation_to_spy` (`10d`): `+44.10%` net vs. `SPY +44.21%` (`-0.04%/yr [NW t = -0.05]`, `1/3` yrs; `80/20` `+44.08%`, `-0.02%/yr [NW t = -0.01]`).
     - `90_10__sp500_tilt_news__R1_backwardation_to_spy` (`10d`): `+41.65%` walk-forward refit (`-1.05%/yr [NW t = -0.35]`, **higher Sharpe `1.51` vs. `SPY 1.42`**, **MaxDD `-12.52%` vs. `-12.59%`**) and `+40.31%` frozen `2024-05-28` weights (`-1.52%/yr`).
     - `80_20__spy__R2_adaptive_hedge_sec_xgb` (`63d`): `+42.29%` walk-forward refit and `+42.27%` frozen `2024-05-28` weights vs. `SPY +46.88%` (`-1.72%/yr [NW t = -1.93]`).
   - **Validated Component Signal vs. Failure Modes (`2024-05-29..2026-07-08`):**
     - **Held up out-of-sample:** Standalone `100% mid_small_ex_sp500` beta/sector/vol-neutral SEC-catalyst sleeve **`R1_base_bn_sec_xgb` (`10d`) beat both matched mid/small (`+2.22%/yr [NW t = +0.67]`) and `SPY` (`+44.59%` vs. `+44.21%`, `+0.97%/yr`)** net of `20 bps`, and `mid_small_beta_neutral_sec` Stage-2 SEC residual XGBoost achieved positive Base IC (`+0.0142..+0.0259`) and positive Paired Delta IC at all 4 horizons (`+0.0005 [NW t = +1.97]`, `3/3` positive years at `21d`).
     - **Failed out-of-sample:** S&P 500 historical Alpaca news Stage-2 added zero IC (`-0.0004` at `10d`); S&P 500 active equal-rank cap-tilt trailed narrow mega-cap `SPY` returns in `2025–2026H1`; `R2`'s un-neutralized `raw` mid/small contango basket trailed `SPY` (`+24.99%` vs. `+46.88%` at `63d`); and daily-close `VIX > VIX3M` switch turnover trimmed `R1_base_bn_sec_xgb`'s `+0.97%/yr` `10d` standalone edge during `Aug 2024` / `Apr 2025` V-bounces.

2. **Retained Research-Lane Scoreboards & Retired Failed Artifacts:**
   - **Retained authoritative scoreboards:** `data/research/research_lane/wider_beta_neutral_mid_small_ex_sp500_20261119T222600Z` (`trials.jsonl` line `11`, validated `mid_small` beta-neutral + SEC catalyst model) and `data/research/research_lane/sealed_holdout_20240529_20260708_20261121T193257Z` (`trials.jsonl` line `14`, Step 6 `2024–2026` holdout verdict), plus `data/research/research_lane/trials.jsonl` (`14` lines).
   - **Retired failed/superseded artifacts:** `127` zero-reference failed/intermediate `data/research` entries (`7.66 GB`), `3` superseded pre-holdout intermediate scoreboards, `docs/contracts/tradingflow_handoff.md`, `R3_vol_interaction_xgb`, and `6` failed standalone scorer CLI commands (`score-price-predictors`, `score-fundamental-predictors`, `score-short-reversal`, `score-benchmark-tilt`, `score-core-satellite-spy`, `score-vol-regime-core-satellite`).
   - **Model-training state (`Step 7`, commit `eb600e5`):** `0` production models are promoted (`active_serving_pointer_found = false`, API `V1` `actionable = false`). In `research_lane`, the surviving **`mid_small_beta_neutral_sec` Stage-2 `XGBoost` research bundle** (`10d` `260,081` examples / `1,525` stocks; `21d` `257,820` examples / `1,517` stocks, `2019-07-09..2026-07-08`) is trained, persisted, and SHA-256 verified at `data/research/research_lane/models/mid_small_beta_neutral_sec_xgb` (`_manifest.json` SHA-256 `cd18356701f35b44c884d9f587a8e14ed7473878f12f75e2ec27548b4d8ebf04`), and `record-forward-research-scores` verifies and scores it daily (`score_mid_small_bn_sec_xgb_10d`, `score_mid_small_bn_sec_xgb_21d`, `score_mid_small_bn_sec_xgb_opra_10d`) alongside live `OPRA` straddle/skew overlays.

3. **Operational Live Prospective Forward Recorder & Outcome Evaluator (`> 2026-07-08`):**
   - Modules & Scheduler: `src/market_predictor/research_lane/forward_record.py`, `src/market_predictor/research_lane/forward_evaluate.py`, `src/market_predictor/research_lane/opra_options.py`, `scripts/run_forward_research_scoring.ps1`, and `scripts/install_forward_research_task.ps1` (registered in Windows Task Scheduler as `MarketPredictorForwardScoringPostClose`, `Mon–Fri` at `22:35` local time, commit `43e63cd`).
   - Retained append-only snapshots in `data/research/research_lane/forward_records/`: `2026-11-16`, `2026-11-17`, `2026-11-18`, `2026-11-19`, `2026-11-20` (`5` lines in `forward_log.jsonl`), plus matured outcome evaluation `data/research/research_lane/forward_evaluations/forward_eval_20261121T185348Z` (`commit 1e65122`: `9` matured pairs through `2026-11-20` showing `OPRA` `25d` put-skew vetoed falling-knife stocks trailed `SPY` by `-1.66%` to `-6.02%` at `3d` while `catalyst_shock_to_straddle > 1.0` stocks beat `SPY` at `1d/2d/3d` by `+1.77% / +0.22% / +0.31%`).

## 3. Files To Read First When Resuming

1. `AGENTS.md`
2. `docs/active_edge_rebuild_plan.md`
3. `docs/reviews/active_edge_rebuild_handoff.md`
4. `docs/reviews/feature_engineering_audit_20260801.md`
5. `src/market_predictor/research_lane/sec_catalyst_model.py`
6. `src/market_predictor/research_lane/forward_record.py`
7. `src/market_predictor/research_lane/forward_evaluate.py`
8. `src/market_predictor/research_lane/opra_options.py`
9. `src/market_predictor/research_lane/sealed_holdout.py`

## 4. Verification Commands & Next Actions

- **Verification commands (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`):**
  - `.\.venv\Scripts\ruff.exe check src/market_predictor/config.py src/market_predictor/research_lane src/market_predictor/commands/research_lane.py`
  - `.\.venv\Scripts\mypy.exe --strict src/market_predictor/config.py src/market_predictor/research_lane src/market_predictor/commands/research_lane.py`
  - `.\.venv\Scripts\pytest.exe -q tests/test_active_continuity_documents.py tests/test_architecture_boundaries.py tests/test_strategy_governance.py tests/test_strategy_research_contracts.py tests/test_cli_surfaces.py tests/test_config_secrets.py`

Exact next checkpoint: continue daily post-close (`>= 16:20 ET`) prospective `OPRA` + Finviz Elite + Alpaca `fiscal.ai` + SEC snapshot accumulation (`record-forward-research-scores --as-of-date latest --skip-existing`) and outcome grading (`evaluate-forward-research-scores`), and resolve with the user whether to execute or close the deferred `C:` -> `D:/market-predictor/data` storage migration.
