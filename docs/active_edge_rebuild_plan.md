# Active Edge Rebuild Plan

## 1. Current Active Checkpoint

Current checkpoint: **Step 7 — Standing Daily Prospective `OPRA` + Multi-Source Catalyst Accumulation, Maturation Scoring & Optional `D:` Storage Migration Decision** (`in progress`).

- **Current state (November 21):**
  1. **Path #1 (`Step 5h`, commit `1e65122`)** and **Path #2 (`Step 6`, commit `c484658`)** are complete, verified, and recorded.
  2. Because `2024-05-29..2026-07-08` was unsealed once in Step 6 (`data/research/research_lane/sealed_holdout_20240529_20260708_20261121T193257Z`), `2016-01-04..2026-07-08` is now **permanently development history** and may never again be described as an untouched final test. None of the historical price/SEC/news-only Core-Satellite specifications cleared Step 6's `SPY` outperformance promotion gate, so API `V1` remains `actionable = false`.
  3. **Sole untouched out-of-sample clock (`2026-11-16+`):** continue daily post-close (`>= 16:20 ET`) append-only recording via `record-forward-research-scores --as-of-date latest --skip-existing` and maturation grading via `evaluate-forward-research-scores`, where live `OPRA` single-stock `ATM Straddle %` (`catalyst_shock_to_straddle`), `25d` Put–Call IV Skew (`falling_knife_put_skew_veto`), `SPY`/`QQQ` `30d/90d` IV term structure, and Finviz Elite + Alpaca `fiscal.ai` + SEC catalysts are captured without lookahead.
  4. **Path #3 user decision pending:** confirm with the user whether to resume the paused November 13 physical `C:` -> `D:/market-predictor/data` dataset migration (`--data-root`) or keep canonical data on `C:`.

---

## 2. Completed Active Research-Lane Architecture (Steps 5c–6, November 19–21)

### Step 6 — One-Time Unsealed Final-Test Holdout Evaluation (`2024-05-29..2026-07-08`) vs. `SPY` (`complete`, commit `c484658`)
- Added `src/market_predictor/research_lane/sealed_holdout.py` (`evaluate-sealed-holdout-2024-2026`) and `tests/test_research_lane_sealed_holdout.py`, keeping all `<= 2024-05-28` files untouched on disk and writing incremental `2024-05-29..2026-07-08` CBOE, SEC `8-K`/non-`8-K`, and overlap-stitched Alpaca SIP bars to `data/research/research_lane/sealed_holdout_extension_20240529_20260708/`.
- Executed single unsealed evaluation `data/research/research_lane/sealed_holdout_20240529_20260708_20261121T193257Z` (`trials.jsonl` line `14`; `528` holdout sessions `2024-05-29..2026-07-08`, `71` backwardation sessions = `13.45%`, `626` S&P 500 + `1,653` `mid_small_ex_sp500` securities across `2,642` aligned sessions `2016-01-04..2026-07-08`, `0` simultaneous active CIK overlaps, peak RAM `2.13 GiB`, *research labels* on adjusted prices, net of `20 bps` exact post-drift turnover):
  - **Core-Satellite Verdict (`0` specs promoted vs. `SPY`):**
    - `90_10__spy__R1_backwardation_to_spy` (`10d`): `+44.10%` net vs. `SPY +44.21%` (`-0.04%/yr [NW t = -0.05]`, `1/3` positive years; `80/20` `+44.08%`, `-0.02%/yr [NW t = -0.01]`).
    - `90_10__sp500_tilt_news__R1_backwardation_to_spy` (`10d`): `+41.65%` walk-forward refit (`-1.05%/yr [NW t = -0.35]`, **higher Sharpe `1.51` vs. `SPY 1.42`**, **lower MaxDD `-12.52%` vs. `-12.59%`**) and `+40.31%` frozen `2024-05-28` weights (`-1.52%/yr [NW t = -0.50]`) vs. `SPY +44.21%`.
    - `80_20__spy__R2_adaptive_hedge_sec_xgb` (`63d`): `+42.29%` walk-forward refit (`-1.72%/yr [NW t = -1.93]`, `0/3` yrs) and `+42.27%` frozen `2024-05-28` weights (`-1.72%/yr [NW t = -1.96]`) vs. `SPY +46.88%`.
  - **What genuinely held up vs. what failed on `2024-05-29..2026-07-08`:**
    - **Validated out-of-sample:** Standalone `100% mid_small_ex_sp500` beta/sector/vol-neutral SEC-catalyst sleeve **`R1_base_bn_sec_xgb` at `10d` beat both matched mid/small (`+2.22%/yr [NW t = +0.67]`) and `SPY` (`+44.59%` vs. `+44.21%`, `+0.97%/yr`)** after `20 bps` turnover (`13.02%`), and `mid_small_beta_neutral_sec` Stage-2 SEC residual XGBoost maintained positive Base IC across all 4 horizons (`+0.0142..+0.0259`) plus positive Paired Delta IC across all 4 horizons (**`+0.0005 [NW t = +1.97]` in `3/3` years at `21d`**).
    - **Failed out-of-sample:** S&P 500 historical Alpaca news Stage-2 added zero IC (`-0.0004 [NW t = -0.60]` at `10d`); S&P 500 equal-rank active tilt lagged narrow `2025–2026H1` mega-cap returns despite improving Sharpe (`1.51` vs. `1.42`); `R2`'s un-neutralized `raw` mid/small contango leg trailed `SPY` (`+24.99%` vs. `+46.88%` at `63d`); and daily-close `VIX > VIX3M` parking (`R1`) whipsawed `~50 bps` during V-bottom rebounds (`Aug 2024`, `Apr 2025`).

### Step 5h — Live `2026-11-20` Prospective Snapshot & Forward-Outcome Maturation Evaluator (`complete`, commit `1e65122`)
- Added `src/market_predictor/research_lane/forward_evaluate.py` (`evaluate-forward-research-scores`) and `tests/test_research_lane_forward_evaluate.py`.
- Captured Friday `2026-11-20` live snapshot (`data/research/research_lane/forward_records/2026-11-20`: `1,000` SEC-verified `A..Z` common stocks, `647` live catalysts = `284` Finviz Elite + `279` Alpaca `fiscal.ai` + `84` SEC, **`spy_iv_30d = 29.83%` vs. `spy_iv_90d = 25.85%` [`1.1539`]** and **`qqq_iv_30d = 39.59%` vs. `qqq_iv_90d = 31.73%` [`1.2477`]**, `opra_backwardation = true`, `2` falling-knife vetoes, `1` shock `>` straddle).
- Evaluated `9` matured `(snapshot, horizon)` pairs across `2026-11-16..2026-11-19` through `2026-11-20` (`data/research/research_lane/forward_evaluations/forward_eval_20261121T185348Z`, *research labels*, zero sealed data read):
  - **`OPRA` `falling_knife_put_skew_veto` (`25d` Put–Call IV Skew):** vetoed falling-knife stocks trailed `SPY` by **`-1.66%` (`3d` from `2026-11-16`)** and **`-6.02%` (`3d` from `2026-11-17`, `-11.96%` absolute drop vs. `-5.94%` `SPY`)**.
  - **`OPRA` `catalyst_shock_to_straddle > 1.0`:** the `3` `2026-11-17` stocks whose catalyst shock exceeded `ATM Straddle %` **beat `SPY` at all 3 matured horizons (`1d +1.77%`, `2d +0.22%`, `3d +0.31%`)** during the `-5.94%` market sell-off.
  - **`R1` backwardation shift (`2026-11-16`):** live `OPRA` backwardation (`1.0330 > 1.0`) shifted the `10%` satellite into `SPY` (`+0.00%` vs. `SPY`, avoiding `-1.07%` unhedged `beta_neutral` / `-9.21%` unhedged reversal drawdown).

### Step 5g — Frozen Prospective Candidate Registry & Idempotent `"latest"` Daily Recorder (`complete`, commit `8eae9ca`)
- Added `FROZEN_PROSPECTIVE_CANDIDATES`, `resolve_latest_completed_session` (`16:20 ET` post-close `SPY` `1Day` SIP bar check strictly `> 2026-07-08`), and `--as-of-date latest --skip-existing` fast-path idempotency in `src/market_predictor/research_lane/forward_record.py` and `src/market_predictor/commands/research_lane.py`.
- Retained immutable forward snapshots in `data/research/research_lane/forward_records/`: `2026-11-16`, `2026-11-17`, `2026-11-18`, `2026-11-19` (`forward_log.jsonl` `4` lines).

### Step 5f — Zero-Fit Live Alpaca `OPRA` Option-Chain Overlay (`complete`, commit `934b6e3`)
- Added `src/market_predictor/research_lane/opra_options.py` and `tests/test_research_lane_opra_options.py`: right-anchored OCC option parser, safeguarded Newton-Raphson + bisection Black-Scholes IV/delta solver, 3-window `/v2/options/contracts` selection (`[2..20d]` front ATM, `[21..55d]` 1M ATM + `25d` put/call delta wings, `[72..125d]` 3M ATM), variance-interpolated `SPY`/`QQQ` `30d/90d` IV term structure, front-expiry `ATM Straddle %` + `catalyst_shock_to_straddle`, `25d` Put–Call IV Skew + `falling_knife_put_skew_veto`, and `score_r2_opra_conditioned`.
- Real `2026-11-16` prospective run (`data/research/research_lane/forward_records/2026-11-16`): `1,000` SEC-verified `A..Z` US common stocks (`>= $5`), `384` live catalysts (`188` Finviz Elite + `124` Alpaca `fiscal.ai` + `72` SEC), **`spy_iv_30d = 18.25%` vs. `spy_iv_90d = 17.67%` (`1.0330`)**, **`qqq_iv_30d = 28.84%` vs. `qqq_iv_90d = 26.33%` (`1.0955`)**, `opra_backwardation = true`, `1` falling-knife reversal veto (`2026-11-17` recorded `22` straddles, `12` `25d` skews, `2` falling-knife vetoes, `3` catalyst shocks `> 1.0x` straddle).
- Provider fact recorded via bounded probe (`0` sealed rows read): `/v1beta1/options/bars` begins `2024-01-18` / `2024-02-01` (`82` sessions before `2024-05-28`), preventing pre-holdout tree fitting without unsealing `2024-05-29..2026-07-08`.

### Step 5e — Public CBOE Volatility Term-Structure & Skew Regime Conditioning (`complete`, commit `cfa52f8`)
- Added `src/market_predictor/research_lane/vol_regime.py`; retained `data/research/research_lane/cboe_vol_indices/cboe_daily.parquet` (`2,635` rows `2014-01-02..2024-05-28` across `VIX`, `VIX3M`, `VIX9D`, `VXN`, `SKEW`; `313/2,114` backwardation sessions = `14.81%`; `trials.jsonl` line `13`).
- Measured results (*research labels* on adjusted prices, net of `20 bps` post-drift turnover):
  - Zero-fit **`R2_adaptive_hedge_sec_xgb`** beat `SPY` standalone across all `4` horizons (`OOS 2021..2024`), compounding **`+98.03%` vs. `+72.68%` for `SPY` at `63d`** (`+9.70%/yr [NW t = +1.77]` vs. matched mid/small; joint `80/20` compounded **`+79.81%` vs. `+72.68%`** `[NW t = 0.95]`).
  - Zero-fit **`R1_backwardation_to_spy`** compounded **`+162.51%` vs. `+130.57%` for `SPY`** on `OOS 2020..2024` (`10d`, `90/10`, `+2.74%/yr [NW t = 0.93]`, `4/6` positive years, **MaxDD `-22.92%` vs. `-24.15%` for `SPY`**).
  - Fitted **`R3_vol_interaction_xgb`** collapsed out-of-sample (`-1.66%..+19.79%` standalone) and its code was retired per `AGENTS.md` §3.2.

### Step 5d — Unified Core `SPY` / S&P 500 Tilt + Mid/Small SEC-Catalyst Satellite & Multi-Source Live Catalysts (`complete`, commit `e247eda`)
- Added `src/market_predictor/research_lane/core_satellite.py`, `FINVIZ_API_KEY` alias support in `src/market_predictor/config.py`, and Finviz Elite (`news_export.ashx`) + Alpaca (`fiscal.ai`) + SEC live catalyst capture in `forward_record.py` (`trials.jsonl` line `12`, `585` S&P 500 + `1,559` `mid_small_ex_sp500` securities, `0` active CIK overlaps).

### Step 5c — Beta/Vol/Sector-Neutralized `mid_small_ex_sp500` & `$0` Precise SEC Catalysts (`complete`, commit `643c801`)
- Added `src/market_predictor/research_lane/beta_neutral.py`; collected `data/research/research_lane/wider_us_universe/wider_non8k_catalysts.parquet` (`208,322` `SC 13D/A`, `S-1/S-3/424B`, and `NT 10-K/Q` filings `<= 2024-05-28` across `2,253` CIKs) and retained `data/research/research_lane/wider_beta_neutral_mid_small_ex_sp500_20261119T222600Z` (`trials.jsonl` line `11`).
- **Stage-2 precise SEC catalysts passed §5.6's paired `NW t >= 2.0` gate in `4` specs** (`equal_rank_24 + news_xgboost` **`+0.0005 [NW t = +3.05]` in `5/5` OOS years at `5d`** and **`+0.0007 [NW t = +2.76]` in `4/5` OOS years at `63d`**), and `[sector, beta_252d, idio_vol_60d]` orthogonalization lifted `21d` buffered skill from `-0.77%/yr` to `+1.01%/yr` (`+3.03%/yr` at `63d`).

---

## 3. Consolidated Negative-Evidence & Superseded Trial Ledger (`trials.jsonl` Lines 1–13)

Superseded intermediate scoreboard folders, `127` unreferenced failed/intermediate `data/research` work folders (`7.66 GB`), `docs/contracts/tradingflow_handoff.md`, `R3_vol_interaction_xgb`, and failed standalone scorer CLI commands (`score-price-predictors`, `score-fundamental-predictors`, `score-short-reversal`, `score-benchmark-tilt`, `score-core-satellite-spy`, `score-vol-regime-core-satellite`, `score-swing-setups`, `score-technical-models`, `score-two-stage-models`, `score-wider-two-stage-models`) were retired per `AGENTS.md` §3.2 and §3.8. Full commit, input-hash, and metric provenance remains in `data/research/research_lane/trials.jsonl`:

| Trial(s) | Step & Scope | Key Measured Result (*research labels*, `<= 2024-05-28`) | Disposition |
|---|---|---|---|
| Line `1` | Step 1: `12` price predictors on S&P 500 (`585` securities) | Weak standalone S&P 500 ordering (`composite_all` `21d` IC near `0`). | Inputs retained in `predictors.py`; `score-price-predictors` & folder retired. |
| Line `2` | Step 2a: `12` SEC `company_facts` + `8-K` Item `2.02` inputs | Fundamental inputs verified point-in-time; standalone S&P 500 spread weak after `20 bps`. | Inputs retained in `fundamentals.py`; `score-fundamental-predictors` & folder retired. |
| Line `3` | Step 2b: `~50` technical indicators + `24` Ridge/XGBoost specs | Median OOS IC `< 0.03` across `24` specs; rule not met. | `indicators.py` & `fitted.py` retained; `score-technical-models` retired. |
| Line `4` | Step 2c: `1–3` session short-term reversal | Passed development reading rule at **`3` sessions only**; negative when cap-weighted (`-2.43%/yr [t = -2.21]`). | Math retained in `reversal.py`; `score-short-reversal` & folder retired. |
| Lines `5–6` | Steps 3a & 3b: `Setup A/B/C` candle breakouts + `+2.5/-1.0 ATR` & `MTR(14)` + Stage-2 `XGBClassifier` | `58.8%` stop-out / `22.4%` timeout rate; all Stage-1 & Stage-2 specs trailed matched `SPY` (`-0.25%..-0.34%/trade`, `t = -1.61..-3.57`); `MTR(14)` raised stop-outs to `63.0%`. | Dead-end modules (`setups.py`, `barrier_labels.py`, `meta_model.py`) deleted. |
| Line `7` | Step 4: S&P 500 cap-weighted benchmark tilt & rank hysteresis (`buy top 10% / hold top 33%`) | Hysteresis cut turnover `71.8% -> 31.3%` (`+6.10%/yr [t = 1.94]` skill at `3d`), `0/48` product specs hit `NW t >= 2.50` vs. `SPY`. | Hysteresis & tilt engine retained in `benchmark_tilt.py`; `score-benchmark-tilt` & folder retired. |
| Line `8` | Step 5a: Extended `2016–2024` S&P 500 two-stage (`556` CIKs in `sp500_free_backfill`) | Stage-1 `21d` IC `-0.0005 [t = -0.03]`; Stage-2 Alpaca news failed on `2024–2026` holdout (`-0.0004 [t = -0.60]`). | `sp500_free_backfill` & `two_stage.py` retained; `score-two-stage-models` retired. |
| Lines `9–10`, `12–13` | Steps 5b/5d/5e: Wider universe + pre-holdout Core-Satellite grids | Superseded by Step 6 `2024–2026` holdout (`sealed_holdout_20240529_20260708_20261121T193257Z`) and Step 5c `wider_beta_neutral_mid_small_ex_sp500_20261119T222600Z`. | `wider_us_universe` retained; intermediate scoreboards & `score-core-satellite-spy` / `score-vol-regime-core-satellite` retired. |

---

## 4. Historical Governed Swing, Investment & Storage Summary (Closed / Deferred)

The following pre-research-lane workstreams remain historical or user-deferred evidence; their hash-bound contracts (`docs/strategy_execution_ledger.json`, `docs/strategy_hypothesis_registry.json`, `docs/reference_model_inventory.json`, `docs/model_artifact_retention_inventory.json`, `docs/evidence/*`, `docs/model_cards/*`) are preserved intact:
- **Configurable `--data-root` Storage Infrastructure (November 13, commit `d3211ee`):** `EvidencePaths` implemented and verified; physical `C:`-to-`D:` dataset copy remains paused by user instruction (`all C: originals intact`).
- **Investment Cohort `63d/252d` Evaluation (November 6–11, `docs/research_briefs/fable_5_2_project_brief_20261110.md`):** SEC `company_facts` ratios, ICTax benchmark distributions, and `3,177` known / `715` unresolved holding outcomes audited; no investment model promoted or served.
- **Governed Swing Technical & Relationship Return Fits (September–October, `docs/reviews/feature_engineering_audit_20260801.md`):** `314,167`-row technical and relationship fits completed (`mean IC 0.0007..0.0147`, MSE worse than zero-excess baseline); intraday prediction retired (`KS4` / `KS6` historical records only); communicating APIs remain `V1` with `actionable = false`.
