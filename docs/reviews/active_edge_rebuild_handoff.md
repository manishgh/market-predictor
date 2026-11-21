# Active Edge Rebuild Handoff

## 1. Current Repository & Branch State (November 21)

- **Checkout & Branch:** `C:/project/market-predictor` on `main` (single canonical checkout per `AGENTS.md`; latest implementation commit `c484658`).
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

2. **Historical Pre-Holdout Benchmark Comparison (`<= 2024-05-28`, Retained Evidence):**
   - `data/research/research_lane/vol_regime_core_satellite_20261120T104017Z`, `core_satellite_spy_20261120T095345Z`, and `wider_beta_neutral_mid_small_ex_sp500_20261119T222600Z` (`585` S&P 500 + `1,559` `mid_small_ex_sp500` securities across `2016-01-04..2024-05-28`, `0` active CIK overlaps).

3. **Operational Live Prospective Forward Recorder & Outcome Evaluator (`> 2026-07-08`):**
   - Modules: `src/market_predictor/research_lane/forward_record.py`, `src/market_predictor/research_lane/forward_evaluate.py`, and `src/market_predictor/research_lane/opra_options.py`.
   - Retained append-only snapshots in `data/research/research_lane/forward_records/`: `2026-11-16`, `2026-11-17`, `2026-11-18`, `2026-11-19`, `2026-11-20` (`5` lines in `forward_log.jsonl`), plus matured outcome evaluation `data/research/research_lane/forward_evaluations/forward_eval_20261121T185348Z` (`commit 1e65122`: `9` matured pairs through `2026-11-20` showing `OPRA` `25d` put-skew vetoed falling-knife stocks trailed `SPY` by `-1.66%` to `-6.02%` at `3d` while `catalyst_shock_to_straddle > 1.0` stocks beat `SPY` at `1d/2d/3d` by `+1.77% / +0.22% / +0.31%`).

## 3. Files To Read First When Resuming

1. `AGENTS.md`
2. `docs/active_edge_rebuild_plan.md`
3. `docs/reviews/active_edge_rebuild_handoff.md`
4. `docs/reviews/feature_engineering_audit_20260801.md`
5. `src/market_predictor/research_lane/sealed_holdout.py`
6. `src/market_predictor/research_lane/forward_record.py`
7. `src/market_predictor/research_lane/forward_evaluate.py`
8. `src/market_predictor/research_lane/opra_options.py`

## 4. Verification Commands & Next Actions

- **Verification commands (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`):**
  - `.\.venv\Scripts\ruff.exe check src/market_predictor/config.py src/market_predictor/research_lane src/market_predictor/commands/research_lane.py`
  - `.\.venv\Scripts\mypy.exe --strict src/market_predictor/config.py src/market_predictor/research_lane src/market_predictor/commands/research_lane.py`
  - `.\.venv\Scripts\pytest.exe -q tests/test_active_continuity_documents.py tests/test_architecture_boundaries.py tests/test_strategy_governance.py tests/test_strategy_research_contracts.py tests/test_cli_surfaces.py tests/test_config_secrets.py`

Exact next checkpoint: continue daily post-close (`>= 16:20 ET`) prospective `OPRA` + Finviz Elite + Alpaca `fiscal.ai` + SEC snapshot accumulation (`record-forward-research-scores --as-of-date latest --skip-existing`) and outcome grading (`evaluate-forward-research-scores`), and resolve with the user whether to execute or close the deferred `C:` -> `D:/market-predictor/data` storage migration.
