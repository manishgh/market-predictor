# Current Feature Engineering Audit

## 1. Research-Lane Feature Acceptance Matrix (November 21)

All exploratory and prospective features in `src/market_predictor/research_lane/` enforce strict point-in-time causality, truncated-history invariance, and future-poison unit tests (`2024-05-29..2026-07-08` unsealed once in Step 6 commit `c484658`; `2026-11-16+` prospective ledger remains the sole untouched out-of-sample test clock):

| Feature Group | Module(s) | Causal Clock & Availability | Batch (`2016–2026`) | Live Prospective (`> 2026-07-08`) | Empirical Status (*research labels*) |
|---|---|---|---|---|---|
| `12` Price & Liquidity Ranks + `[sector, beta, idio_vol]` Neutralization | `predictors.py`, `beta_neutral.py` | Session `t` close (`16:00 ET`), entry `open[t+1]` | Verified (`2016–2026` S&P 500 + `mid_small_ex_sp500`) | Verified (`forward_record.py`, `60d` beta warm-up fallback) | Held positive `mid_small` beta-neutral Base IC on `2024–2026` holdout (`5d +0.0142`, `10d +0.0162`, `21d +0.0172`, `63d +0.0259`); un-neutralized `raw` mid/small lagged mega-cap `SPY` (`+24.99%` vs. `+46.88%` at `63d`). |
| `12` SEC `company_facts` + `8-K` Item `2.02` Fundamental Ranks | `company_facts.py`, `fundamentals.py`, `unadjusted_prices.py` | Usable first session strictly after SEC filing date (`200d` staleness cap); market value = `unadjusted_close * shares` | Verified (`626` S&P 500 + `1,653` mid/small securities) | Not required by `R1`/`R2` live satellite sleeve (`price + catalyst + OPRA` only) | Combined into `equal_rank_24` across `2,253` survivorship-checked CIKs (`100%` SEC XBRL coverage through `2026-07-08`). |
| `1–3` Session Short-Term Reversal Composite | `indicators.py`, `reversal.py` | Session `t` close (`16:00 ET`) | Verified (`short_reversal_20261117T132343Z`) | Verified (`score_reversal_short` & `score_reversal_skew_vetoed`) | Passed development reading rule at `3d` equal-weight; must be gated by `OPRA` `25d` put-skew veto during backwardation sell-offs (`-11.96%` drop on vetoed knives in `2026-11-17..2026-11-20`). |
| Stage-2 Direct-Issuer News & Precise SEC Catalyst Residuals | `news_catalysts.py`, `two_stage.py`, `beta_neutral.py`, `forward_record.py` | Exact UTC publication / SEC EDGAR acceptance timestamp `<= close_utc(t)` | Verified (`2019–2026` Alpaca news + `8-K`/Form-4/`13D/G`/`S-3`/`NT` filings) | Verified (Finviz Elite `news_export.ashx` + Alpaca `fiscal.ai` + SEC EDGAR + `classify_event_families`) | **`mid_small_beta_neutral_sec` held up on `2024–2026` holdout** (`21d` Paired Delta IC `+0.0005 [NW t = +1.97]`, `3/3` yrs; standalone `10d` `R1_base_bn_sec_xgb` `+44.59%` vs. `SPY +44.21%`); historical S&P 500 Alpaca news failed on `2024–2026` (`-0.0004 [t = -0.60]` at `10d`). |
| Index Volatility Term Structure (`30d / 90d` Backwardation) | `vol_regime.py` (CBOE), `opra_options.py` (live `OPRA`) | Session `t` close (`VIX/VIX3M` historical; `SPY`/`QQQ` `OPRA` `1Day` bars live) | Verified (`2014–2026-07-08` CBOE daily rows) | Verified (`spy_iv_term_30_90`, `qqq_iv_term_30_90`, `opra_backwardation`) | `90_10__spy__R1` tied `SPY` on `2024–2026` (`+44.10%` vs. `+44.21%`, `-0.04%/yr`) and protected capital in live `2026-11-16..2026-11-20` backwardation (`1.1539`), though daily-close `VIX > VIX3M` whipsawed `~50 bps` on V-bounces. |
| Single-Stock `OPRA` `ATM Straddle %` & `25d` Put–Call IV Skew | `opra_options.py`, `forward_evaluate.py` | Session-exact `as_of_date` `1Day` `OPRA` bars (`zero` post-session quote leakage) | Available from `Jan/Feb 2024` (`/v1beta1/options/bars` limit) | Verified (`atm_straddle_pct`, `catalyst_shock_to_straddle`, `put_call_skew_25d`, `falling_knife_put_skew_veto`) | Initial live maturation (`2026-11-16..2026-11-20`, `9` pairs): `falling_knife_put_skew_veto` stocks trailed `SPY` by `-1.66%` to `-6.02%` (`3d`), while `catalyst_shock_to_straddle > 1.0` stocks beat `SPY` (`+0.22%..+1.77%`). |

---

## 2. Historical Governed Swing & Storage Audit Records (Retained Reference)

The sections below preserve exact hashes and metrics referenced by `docs/research_briefs/benchmark_outperformance_recommendations_20261115.md` and governed readiness contracts. Nothing from the exploratory research lane above is promoted to `production_cli` or served via API `V1` (`actionable` remains `false`).

### Configurable Data Storage & Scratch Cleanup (November 13, Commit `d3211ee`)
- `EvidencePaths` (`--data-root`) implemented and verified across collectors, dataset builders, and trainers (`_export.json` SHA256 `fc72cdba5f53d9d0a9a4c9eafc754f679e219fa1b1e04b6612793b2932b90377`; C preflight SHA256 `cdac785950f6d4f238b92cd13711501ce640b428226fc482daa309a3a448e5f1`). Physical `C:`-to-`D:` dataset migration remains deferred by user instruction with all `C:` originals intact.

### Relationship Return Results
Completed historical run: `data/research/swing_relationship_return_models_initial_fit`, manifest SHA256 `7a699020920024ed29b5d8547724f5355fdc2202ca176f11d38f23418bd4666e` (`314,167` final-fit rows, `124` columns, peak `2.472 GiB`):

| Learner / scope | Mean daily rank correlation | Mean squared error | Zero-excess error |
| --- | ---: | ---: | ---: |
| Linear / temporal | 0.000704 | 0.00262038 | 0.00255703 |
| Linear / unseen security | 0.005503 | 0.00292185 | 0.00286511 |
| Boosted / temporal | 0.006304 | 0.00261576 | 0.00255703 |
| Boosted / unseen security | -0.013526 | 0.00293655 | 0.00286511 |

All four mean squared errors exceeded predicting zero excess (`no tradable edge established`).

### Technical Return Experiment Acceptance
Completed historical run: `data/research/swing_technical_return_models_initial_fit/_manifest.json`, SHA256 `0a30ef2f8e3b95cee252f8c1d1bb5be3c6b98cddda4e47d68e835ad5c1892551` (`configs/swing_return_training.json` SHA256 `47155e2d6ef43e9efb6bb54621913e3df5ee0f66797153c30f71d8f66797153`, readiness report `data/reports/swing_initial_fit_training_readiness_percentage_only.json` SHA256 `7a468ed3e0662d8d6390ae575bee700cdd7da4ef8bb7c8ae921aed6b8eead39f`; `586,305` decisions / `545` securities / `1,231` sessions; `314,167` final-fit rows; `64.68%` document-proven outcome coverage):

| Learner / scope | Date-weighted MSE | Zero-excess MSE | Mean daily rank correlation |
| --- | ---: | ---: | ---: |
| Linear / temporal | 0.00260415 | 0.00255894 | 0.00449 |
| Boosted / temporal | 0.00260998 | 0.00255894 | 0.01467 |
| Linear / security transfer | 0.00290470 | 0.00286810 | 0.00696 |
| Boosted / security transfer | 0.00292961 | 0.00286810 | 0.00029 |

Both learners had higher aggregate squared error than predicting zero excess, motivating the November 15–20 shift to rank targets, `2,253`-issuer wider mid/small-cap coverage, Stage-2 catalyst residual models, and Core-Satellite volatility-regime conditioning.
