# Active Edge Rebuild Plan

## 1. Standing Operational Checkpoint

Current checkpoint: **Daily prospective forward score, multi-source catalyst & OPRA option accumulation while 2024-05-29..2026-07-08 remains sealed** (`in progress`).

- **Operational command (post-close `>= 16:25 ET` on weekdays):**
  `.\.venv\Scripts\python.exe -m market_predictor.research_cli record-forward-research-scores --root C:\project\market-predictor --as-of-date latest --live-prospective --skip-existing`
- **Frozen invariants:**
  1. The historical final-test holdout window (`2024-05-29..2026-07-08`) remains `100%` untouched and sealed (`earns_sealed_period_test = false` across all `13` historical development trials in `data/research/research_lane/trials.jsonl`). Unsealing requires explicit user approval for one frozen candidate.
  2. Prospective snapshots (`> 2026-07-08`) clamp trailing SIP daily bars strictly to `>= 2026-07-09`, enforce the `16:20 America/New_York` post-close settlement cutoff (`resolve_latest_completed_session`), and record zero future outcomes.
  3. Every prospective `_manifest.json` stamps `FROZEN_PROSPECTIVE_CANDIDATES`:
     - `90_10__sp500_tilt_news__R1_backwardation_to_spy` (`10d` primary swing candidate — shifts the `10%` mid/small satellite into `SPY` when `30d/90d` IV term structure `> 1.0` and holds `score_beta_neutral_price_12` + catalyst ranker in contango).
     - `80_20__spy__R2_adaptive_hedge_sec_xgb` (`63d` core-satellite candidate — `score_r2_opra_conditioned`, holding `score_price_equal_12` in contango and switching to `0.70 * score_beta_neutral_price_12 + 0.30 * score_reversal_skew_vetoed` in backwardation).

---

## 2. Completed Active Research-Lane Architecture (Steps 5c–5g, November 19–20)

### Step 5g — Frozen Prospective Candidate Registry & Idempotent `"latest"` Daily Recorder (`complete`, commit `8eae9ca`)
- Added `FROZEN_PROSPECTIVE_CANDIDATES`, `resolve_latest_completed_session` (`16:20 ET` post-close `SPY` `1Day` SIP bar check strictly `> 2026-07-08`), and `--as-of-date latest --skip-existing` fast-path idempotency in `src/market_predictor/research_lane/forward_record.py` and `src/market_predictor/commands/research_lane.py`.
- Retained immutable forward snapshots in `data/research/research_lane/forward_records/`: `2026-11-16`, `2026-11-17`, `2026-11-18`, `2026-11-19` (`forward_log.jsonl` `4` lines).

### Step 5f — Zero-Fit Live Alpaca `OPRA` Option-Chain Overlay (`complete`, commit `934b6e3`)
- Added `src/market_predictor/research_lane/opra_options.py` and `tests/test_research_lane_opra_options.py`: right-anchored OCC option parser, safeguarded Newton-Raphson + bisection Black-Scholes IV/delta solver, 3-window `/v2/options/contracts` selection (`[2..20d]` front ATM, `[21..55d]` 1M ATM + `25d` put/call delta wings, `[72..125d]` 3M ATM), variance-interpolated `SPY`/`QQQ` `30d/90d` IV term structure, front-expiry `ATM Straddle %` + `catalyst_shock_to_straddle`, `25d` Put–Call IV Skew + `falling_knife_put_skew_veto`, and `score_r2_opra_conditioned`.
- Real `2026-11-16` prospective run (`data/research/research_lane/forward_records/2026-11-16`): `1,000` SEC-verified `A..Z` US common stocks (`>= $5`), `384` live catalysts (`188` Finviz Elite + `124` Alpaca `fiscal.ai` + `72` SEC), **`spy_iv_30d = 18.25%` vs. `spy_iv_90d = 17.67%` (`1.0330`)**, **`qqq_iv_30d = 28.84%` vs. `qqq_iv_90d = 26.33%` (`1.0955`)**, `opra_backwardation = true`, `1` falling-knife reversal veto (`2026-11-17` recorded `22` straddles, `12` `25d` skews, `2` falling-knife vetoes, `3` catalyst shocks `> 1.0x` straddle).
- Provider fact recorded via bounded probe (`0` sealed rows read): `/v1beta1/options/bars` begins `2024-01-18` / `2024-02-01` (`82` sessions before `2024-05-28`), preventing pre-holdout tree fitting without unsealing `2024-05-29..2026-07-08`.

### Step 5e — Public CBOE Volatility Term-Structure & Skew Regime Conditioning (`complete`, commit `cfa52f8`)
- Added `src/market_predictor/research_lane/vol_regime.py`; retained `data/research/research_lane/cboe_vol_indices/cboe_daily.parquet` (`2,635` rows `2014-01-02..2024-05-28` across `VIX`, `VIX3M`, `VIX9D`, `VXN`, `SKEW`; `313/2,114` backwardation sessions = `14.81%`) and `data/research/research_lane/vol_regime_core_satellite_20261120T104017Z` (`trials.jsonl` line `13`).
- Measured results (*research labels* on adjusted prices, net of `20 bps` post-drift turnover):
  - Zero-fit **`R2_adaptive_hedge_sec_xgb`** beat `SPY` standalone across all `4` horizons (`OOS 2021..2024`), compounding **`+98.03%` vs. `+72.68%` for `SPY` at `63d`** (`+9.70%/yr [NW t = +1.77]` vs. matched mid/small; joint `80/20` compounded **`+79.81%` vs. `+72.68%`** `[NW t = 0.95]`).
  - Zero-fit **`R1_backwardation_to_spy`** compounded **`+162.51%` vs. `+130.57%` for `SPY`** on `OOS 2020..2024` (`10d`, `90/10`, `+2.74%/yr [NW t = 0.93]`, `4/6` positive years, **MaxDD `-22.92%` vs. `-24.15%` for `SPY`**).
  - Fitted **`R3_vol_interaction_xgb`** collapsed out-of-sample (`-1.66%..+19.79%` standalone), proving macro term structure must be applied as a zero-fit regime conditioner rather than a tree split feature.

### Step 5d — Unified Core `SPY` / S&P 500 Tilt + Mid/Small SEC-Catalyst Satellite & Multi-Source Live Catalysts (`complete`, commit `e247eda`)
- Added `src/market_predictor/research_lane/core_satellite.py`, `FINVIZ_API_KEY` alias support in `src/market_predictor/config.py`, and Finviz Elite (`news_export.ashx`) + Alpaca (`fiscal.ai`) + SEC live catalyst capture in `forward_record.py`.
- Retained `data/research/research_lane/core_satellite_spy_20261120T095345Z` (`trials.jsonl` line `12`, `585` S&P 500 + `1,559` `mid_small_ex_sp500` securities, `0` active CIK overlaps): standalone `sat_beta_neutral_eq24_sec_xgb` (`63d`) cleared `NW t = +2.00` (`+6.64%/yr` vs. matched mid/small, `+73.20%` vs. `+72.68%` `SPY`), and `90_10__sp500_tilt_news__sat_beta_neutral_eq24_sec_xgb` (`10d`) compounded `+77.18%` vs. `+75.90%` `SPY`.

### Step 5c — Beta/Vol/Sector-Neutralized `mid_small_ex_sp500` & `$0` Precise SEC Catalysts (`complete`, commit `643c801`)
- Added `src/market_predictor/research_lane/beta_neutral.py`; collected `data/research/research_lane/wider_us_universe/wider_non8k_catalysts.parquet` (`208,322` `SC 13D/A`, `S-1/S-3/424B`, and `NT 10-K/Q` filings `<= 2024-05-28` across `2,253` CIKs) and retained `data/research/research_lane/wider_beta_neutral_mid_small_ex_sp500_20261119T222600Z` (`trials.jsonl` line `11`).
- **Stage-2 precise SEC catalysts passed §5.6's paired `NW t >= 2.0` gate in `4` specs** (`equal_rank_24 + news_xgboost` **`+0.0005 [NW t = +3.05]` in `5/5` OOS years at `5d`** and **`+0.0007 [NW t = +2.76]` in `4/5` OOS years at `63d`**), and `[sector, beta_252d, idio_vol_60d]` orthogonalization lifted `21d` buffered skill from `-0.77%/yr` to `+1.01%/yr` (`+3.03%/yr` at `63d`).

---

## 3. Consolidated Negative-Evidence & Superseded Trial Ledger (`trials.jsonl` Lines 1–10)

Superseded intermediate scoreboard folders and isolated dead-end setup modules (`setups.py`, `barrier_labels.py`, `meta_model.py`, `score-swing-setups`, `score-technical-models`, `score-two-stage-models`, `score-wider-two-stage-models`) were retired per `AGENTS.md` §3.2 and §3.8. Full commit, input-hash, and metric provenance remains in `data/research/research_lane/trials.jsonl`:

| Trial(s) | Step & Scope | Key Measured Result (*research labels*, `<= 2024-05-28`) | Disposition |
|---|---|---|---|
| Line `1` | Step 1: `12` price predictors on S&P 500 (`585` securities) | Weak standalone S&P 500 ordering (`composite_all` `21d` IC near `0`). | Inputs retained in `predictors.py`; intermediate folder retired. |
| Line `2` | Step 2a: `12` SEC `company_facts` + `8-K` Item `2.02` inputs | Fundamental inputs verified point-in-time; standalone S&P 500 spread weak after `20 bps`. | Inputs retained in `fundamentals.py`; intermediate folder retired. |
| Line `3` | Step 2b: `~50` technical indicators + `24` Ridge/XGBoost specs | Median OOS IC `< 0.03` across `24` specs; rule not met. | `indicators.py` & `fitted.py` retained; `score-technical-models` retired. |
| Line `4` | Step 2c: `1–3` session short-term reversal (`short_reversal_20261117T132343Z`) | Passed development reading rule at **`3` sessions only**; negative when cap-weighted (`-2.43%/yr [t = -2.21]`). | Retained (`reversal.py` + `short_reversal_20261117T132343Z`). |
| Lines `5–6` | Steps 3a & 3b: `Setup A/B/C` candle breakouts + `+2.5/-1.0 ATR` & `MTR(14)` + Stage-2 `XGBClassifier` | `58.8%` stop-out / `22.4%` timeout rate; all Stage-1 & Stage-2 specs trailed matched `SPY` (`-0.25%..-0.34%/trade`, `t = -1.61..-3.57`); `MTR(14)` raised stop-outs to `63.0%`. | Dead-end modules (`setups.py`, `barrier_labels.py`, `meta_model.py`) deleted. |
| Line `7` | Step 4: S&P 500 cap-weighted benchmark tilt & rank hysteresis (`buy top 10% / hold top 33%`) | Hysteresis cut turnover `71.8% -> 31.3%` (`+6.10%/yr [t = 1.94]` skill at `3d`), `0/48` product specs hit `NW t >= 2.50` vs. `SPY`. | Hysteresis & tilt engine retained in `benchmark_tilt.py`; intermediate folder retired. |
| Line `8` | Step 5a: Extended `2016–2024` S&P 500 two-stage (`556` CIKs in `sp500_free_backfill`) | Stage-1 `21d` IC `-0.0005 [t = -0.03]`; **Stage-2 Alpaca news residuals passed `NW t >= 2.0` in `5` specs** (`+2.26` at `5d`, `+2.36` at `10d`, `5/5` yrs). | Backfill & `two_stage.py` retained for Core-Satellite; `score-two-stage-models` retired. |
| Lines `9–10` | Step 5b: `$0` survivorship-checked wider US universe (`2,253` active + delisted CIKs in `wider_us_universe`) | Moving to `mid_small_ex_sp500` (`1,559` securities) lifted `equal_rank_24` IC to `5d +0.0240 [t = 2.38]`, `10d +0.0288 [t = 2.00]`, `21d +0.0386 [t = 1.83]`, `63d +0.0656 [t = 1.91]`. | `wider_us_universe` & `wider_universe.py` retained; superseded by Step 5c/5d/5e. |

---

## 4. Historical Governed Swing, Investment & Storage Summary (Closed / Deferred)

The following pre-research-lane workstreams remain historical or user-deferred evidence; their hash-bound contracts (`docs/strategy_execution_ledger.json`, `docs/strategy_hypothesis_registry.json`, `docs/reference_model_inventory.json`, `docs/model_artifact_retention_inventory.json`, `docs/evidence/*`, `docs/model_cards/*`) are preserved intact:
- **Configurable `--data-root` Storage Infrastructure (November 13, commit `d3211ee`):** `EvidencePaths` implemented and verified; physical `C:`-to-`D:` dataset copy remains paused by user instruction (`all C: originals intact`).
- **Investment Cohort `63d/252d` Evaluation (November 6–11, `docs/research_briefs/fable_5_2_project_brief_20261110.md`):** SEC `company_facts` ratios, ICTax benchmark distributions, and `3,177` known / `715` unresolved holding outcomes audited; no investment model promoted or served.
- **Governed Swing Technical & Relationship Return Fits (September–October, `docs/reviews/feature_engineering_audit_20260801.md`):** `314,167`-row technical and relationship fits completed (`mean IC 0.0007..0.0147`, MSE worse than zero-excess baseline); intraday prediction retired (`KS4` / `KS6` historical records only); communicating APIs remain `V1` with `actionable = false`.
