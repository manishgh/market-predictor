# Active Edge Rebuild Handoff

## 1. Current Repository & Branch State (November 20)

- **Checkout & Branch:** `C:/project/market-predictor` on `main` (single canonical checkout per `AGENTS.md`).
- **Sealed Final-Test Window (`2024-05-29..2026-07-08`):** `100%` untouched and sealed (`earns_sealed_period_test = false` across all `13` historical development trials in `data/research/research_lane/trials.jsonl`).
- **Heavy-Job Lease:** `data/runtime/heavy_job.lease` absent (`0` background workers).

## 2. Authoritative Research-Lane State (Steps 5c–5g + Bucket 1 Cleanup)

1. **Historical Core-Satellite + Volatility-Regime Benchmark Comparison (`<= 2024-05-28`, *research labels* on split- and dividend-adjusted prices, net of `20 bps` exact post-drift turnover):**
   - **Universe (`0` active CIK overlaps across `2,114` sessions `2016-01-04..2024-05-28`):** `585` point-in-time S&P 500 securities (`data/research/research_lane/sp500_free_backfill`) + `1,559` survivorship-checked active and delisted mid/small-cap US common stocks (`data/research/research_lane/wider_us_universe`, `208,322` non-`8-K` `SC 13D/A`, `S-1/S-3/424B`, `NT 10-K/Q` + `159,528` `8-K` filings) + `2,635` daily CBOE volatility rows (`data/research/research_lane/cboe_vol_indices/cboe_daily.parquet`).
   - **Validated Stage-2 Catalyst Signal (`NW t >= 2.0`):** Continuous post-catalyst reaction residuals (`XGBoost`) passed §5.6's paired gate on both S&P 500 (`NW t = +2.26` at `5d`, `+2.36` at `10d` across `5/5` OOS periods via Alpaca direct-issuer news) and `mid_small_ex_sp500` (`NW t = +3.05` at `5d` in `5/5` years and `+2.76` at `63d` in `4/5` years via precise SEC filings; `data/research/research_lane/wider_beta_neutral_mid_small_ex_sp500_20261119T222600Z`).
   - **Best Volatility-Conditioned Portfolios (`data/research/research_lane/vol_regime_core_satellite_20261120T104017Z` & `core_satellite_spy_20261120T095345Z`):**
     - **`R2_adaptive_hedge_sec_xgb` (`63d`, `OOS 2021..2024`):** standalone mid/small sleeve compounded **`+98.03%` vs. `+72.68%` for `SPY`** (`+9.70%/yr [NW t = +1.77]` vs. matched mid/small); joint `80% SPY + 20% R2` compounded **`+79.81%` vs. `+72.68%` for `SPY`** (`+1.67%/yr [NW t = 0.95]`).
     - **`R1_backwardation_to_spy` (`10d`, `OOS 2020..2024`):** `90% S&P 500 tilt + 10% R1` compounded **`+162.51%` vs. `+130.57%` for `SPY`** (`+2.74%/yr [NW t = 0.93]`, `4/6` positive years, **MaxDD `-22.92%` vs. `-24.15%` for `SPY`**).
     - Fitted `R3_vol_interaction_xgb` (`-1.66%..+19.79%`) and earlier `Setup A/B/C` triple-barrier candle setups (`-0.25%..-0.34%/trade` vs. `SPY`) failed out-of-sample; isolated `setups.py`, `barrier_labels.py`, `meta_model.py`, and `9` intermediate scoreboard folders were retired while keeping `trials.jsonl` intact.

2. **Operational Live Prospective Forward Recorder (`> 2026-07-08`, Zero Future Outcomes Read):**
   - Modules: `src/market_predictor/research_lane/forward_record.py` and `src/market_predictor/research_lane/opra_options.py`.
   - Captures `1,000` SEC-verified `A..Z` US common stocks (`>= $5`), live Finviz Elite (`news_export.ashx`), Alpaca (`fiscal.ai`), and SEC EDGAR catalysts classified via `classify_event_families`, plus zero-fit `OPRA` `SPY`/`QQQ` `30d/90d` constant-maturity IV term structure, front-expiry `ATM Straddle %`, `catalyst_shock_to_straddle`, `25d` Put–Call IV Skew, `falling_knife_put_skew_veto`, and `score_r2_opra_conditioned`.
   - Retained append-only snapshots in `data/research/research_lane/forward_records/`: `2026-11-16`, `2026-11-17`, `2026-11-18`, `2026-11-19` (`4` lines in `forward_log.jsonl`).

## 3. Files To Read First When Resuming

1. `AGENTS.md`
2. `docs/active_edge_rebuild_plan.md`
3. `docs/reviews/active_edge_rebuild_handoff.md`
4. `docs/reviews/feature_engineering_audit_20260801.md`
5. `src/market_predictor/research_lane/forward_record.py`
6. `src/market_predictor/research_lane/opra_options.py`
7. `src/market_predictor/research_lane/vol_regime.py`
8. `src/market_predictor/research_lane/core_satellite.py`

## 4. Verification Commands & Next Actions

- **Verification commands (`OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`):**
  - `.\.venv\Scripts\ruff.exe check src/market_predictor/config.py src/market_predictor/research_lane src/market_predictor/commands/research_lane.py tests/test_research_lane_*.py`
  - `.\.venv\Scripts\python.exe -m mypy --strict src/market_predictor/config.py src/market_predictor/research_lane src/market_predictor/commands/research_lane.py`
  - `.\.venv\Scripts\python.exe -m pytest -q tests/test_active_continuity_documents.py tests/test_architecture_boundaries.py tests/test_model_artifact_retention.py tests/test_strategy_governance.py tests/test_strategy_research_contracts.py tests/test_cli_surfaces.py tests/test_config_secrets.py tests/test_research_lane_*.py`

Exact next checkpoint: run daily post-close (`>= 16:25 ET` on weekdays) prospective forward score, multi-source catalyst, and `OPRA` option-chain accumulation via `.\.venv\Scripts\python.exe -m market_predictor.research_cli record-forward-research-scores --root C:\project\market-predictor --as-of-date latest --live-prospective --skip-existing` while `2024-05-29..2026-07-08` remains sealed, or await explicit user approval before running a single governed final-test evaluation on one frozen candidate.
