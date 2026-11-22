# Current Feature Engineering Audit

## 1. Review of Research-Lane Changes (November 22)

Reviewed checkout `main` at `cb01c63`, principally changes after `5326b51` plus
the training and portfolio-evaluation helpers they consume. This is a code,
design and retained-artifact review, not a rerun of the historical training or
an independent recomputation of the reported portfolio returns.

**Conclusion:** the research implementation and negative-result reporting have
improved, but batch/live equivalence and prospective-evaluation claims are not
established. No production model is promoted. Fix the demonstrated discrepancies
below before using forward results to assess the persisted candidate. Retain the
original snapshots and scoreboards as evidence; do not overwrite them.

### Supported findings and smallest remedies

P1 means the issue can change scores, selected holdings or the interpretation of
test evidence. P2 means a narrower reproducibility or measurement defect.

| ID | Severity and location | Observed evidence and consequence | Smallest remedy and regression check |
| --- | --- | --- | --- |
| R1 | P1: `research_lane/forward_evaluate.py:323`; `forward_record.py:958` | The evaluator accepts snapshots without comparing `recorded_at_utc` to entry time. Of the five retained session snapshots dated November 16–20, the November 16 and 17 records were created November 20, and the November 18 record was created November 19 at 17:13 UTC, after its next-session entry. These three cannot count as prospective predictions. The November 19 and 20 records precede their next-session entries, but none of the five contains any `sec_xgb` score column; the saved bundle was trained November 21 at 21:41 UTC. | Separate retrospective replay from prospective records. Require actual recording before the intended entry, a frozen model/policy identity and payload verification. Test rejection of a record created after entry, including scheduler catch-up. Start SEC-model evidence with its first qualifying new record. |
| R2 | P1: `research_lane/forward_record.py:173,212` | Both readers request `wider_roster_spans.parquet`; the collector writes `roster_spans.parquet`. The former does not exist locally. All 1,000 rows in each of the five retained snapshots have sector `Unknown`; the fallback universe even includes AAPL in three snapshots. This is not the historical `mid_small_ex_sp500` cohort, and sector neutralization has no sector distinctions. Direct `root / ...` reads also bypass relocated storage. | Reuse the existing roster identity and `EvidencePaths`; enforce the same issuer, membership and liquidity contract used by training. Missing authority must abstain. Test against the actual retained roster filename, known sector mappings, S&P exclusion and an external data root. |
| R3 | P1: `research_lane/sec_catalyst_model.py:458,620`; `forward_record.py:126,138` | Training residuals subtract the neutralized 24-predictor price-plus-fundamental baseline. Live scoring adds them to `score_beta_neutral_price_12`, allows fewer available predictors and substitutes 60-session beta when 252-session beta is unavailable. The unit parity test injects an identical synthetic baseline under the price-only column name, so it cannot detect this change of estimator. | Use the same baseline transform and history requirements in training and live scoring, or train and evaluate a separately identified price-only candidate. Compare complete batch and live scores from the same raw inputs, including insufficient-history cases. |
| R4 | P1: `research_lane/sec_catalyst_model.py:443`; `beta_neutral.py:445` | Live catalyst features are ranked only among successfully queried top candidates, whereas training ranks the eligible full cross-section. A value ranked +0.1 among six stocks becomes -0.5 when only the top three are retained, without changing its underlying observation. Same feature names and transformer do not establish feature-value parity. | Freeze one ranking reference population and source-coverage requirement. Obtain that population's data before ranking, or explicitly train a top-candidate-conditioned model. Test subset collection against the full-universe batch reference; missing coverage must not silently redefine ranks. |
| R5 | P1: `research_lane/forward_evaluate.py:148–164` | The top decile is selected after removing stocks with unknown future returns. A 20-stock synthetic unit example with returns -50% for the highest score and +1% for the others reports -24.7% net; making only the highest-scored outcome unknown changes the result to +0.8% by replacing that holding. | Select and persist holdings using decision-time scores only. Report incomplete selected outcomes rather than substituting another stock. Test that removing a selected exit never changes the selected identities or produces a complete cohort return. |
| R6 | P1: `research_lane/forward_evaluate.py:441–468,535` | The report copies the current frozen-candidate registry but ignores snapshot candidate definitions when computing portfolios. It hardcodes price-only R1 and fallback R2, different candidate IDs, and all horizons; the new SEC-XGBoost candidate is not evaluated as a candidate portfolio. Historical buffered open-to-open turnover accounting is replaced by independent top-decile open-to-close windows with a flat deduction. | Evaluate the persisted candidate definition and horizon through the existing shared selection/accounting implementation. Keep standalone signal diagnostics separately labelled. Assert that report candidate IDs, holdings, execution intervals and costs match the recorded policy; missing required signals must abstain. |
| R7 | P1: `research_lane/opra_options.py:557`; `forward_evaluate.py:382` | Missing both benchmark option term ratios becomes `opra_backwardation=False`. The evaluator also defaults missing regime evidence to false, so unavailable options are treated as a measured contango regime and may activate the satellite. | Preserve an explicit unknown regime and apply the candidate's declared missing-data policy. Test absent/failed SPY and QQQ option responses and ensure no measured-regime portfolio is claimed. |
| R8 | P1: `research_lane/benchmark_tilt.py:387–401` (inherited helper used by the new holdout) | A missing required exit is replaced by the last available close/open, or even entry price if nothing later exists. A delisting, halt or incomplete download can therefore receive an invented realized exit instead of an unknown outcome. The research-label exception for adjusted bars does not supply a missing exit. | Preserve unknown holding returns and distinguish valuation marks from executable exits; require verified handling for affected holdings. Test a missing terminal exit and a disappearing security. Quantify affected selected holdings before revising any historical performance claim; this review did not establish their contribution to the saved aggregate returns. |
| R9 | P2: `research_lane/forward_record.py:975–984`; `forward_evaluate.py:321–323`; `sec_catalyst_model.py:267` | Snapshot manifests contain no score/catalyst/options payload hashes and the evaluator trusts the files directly. The bundle loader checks model files against their colocated manifest but does not enforce a separately expected manifest identity. Changing saved scores, or swapping a self-consistent bundle at the same path, is not rejected as a frozen-candidate mismatch. | Reuse existing artifact/hash primitives, bind the expected bundle and policy identities, and verify payloads before skipping or grading a snapshot. Test score-file edits and replacement by a different valid bundle. |
| R10 | P2: `research_lane/opra_options.py:448` | Daily option VWAP is preferred over option close and then inverted using stock closing spot. These observations are not synchronized, so reported IV/skew can reflect intraday spot movement. The single-day bar filter proves date alignment, not a common observation time. | Use a declared aligned option/underlying observation and freshness policy, or label these as daily-VWAP proxies and validate them separately. Test a large intraday stock move with constant underlying volatility. |

### Training, feature engineering and design assessment

- Useful progress: deterministic issuer-grouped holdout, chronological label
  purging through `training_rows`, date-balanced weights, explicit research-label
  status, fixed XGBoost settings, native model persistence and serial heavy-job
  leases. The retained model manifest hash matches `cd183567...`, and all six
  `.ubj` payload hashes match the manifest. These checks establish persistence,
  not predictive value or complete live parity.
- Saved holdout evidence supports a small exploratory SEC-catalyst signal, not
  a demonstrated product improvement. The reported standalone +44.59% versus
  SPY +44.21% is only 0.38 percentage points of cumulative difference; paired
  21-session IC improvement has Newey-West t = 1.97. Candidate selection after
  inspecting these results makes the inspected period development evidence.
  The few November option cohorts cannot establish a mandatory trading rule.
- Treat the OPRA policy as a separate experiment from historical CBOE VIX/VIX3M
  conditioning. The former uses selected SPY/QQQ ATM options and an either-index
  rule; VIX uses a strip of SPX options, so their equivalence is not demonstrated.
  See [CBOE's methodology explanation](https://www.cboe.com/insights/posts/what-the-vix-and-vix-1-d-indices-attempt-to-measure-and-how-they-differ).
- The primary reuse problem is duplicated semantics, not a lack of abstractions:
  parallel baseline builders, candidate registries, evaluation policies, bar
  fetching loops and JSON sanitizers. Consolidate these around existing helpers
  with explicit input/output contracts. Avoid a new generic framework or a
  configuration switch for each experiment.
- Keep model-defining settings in one immutable typed candidate specification:
  cohort, ranking population, ordered features, baseline, history requirement,
  source clocks, horizon, model hashes, selection, costs and missing-data policy.
  Hardcoded dates, top-k values, blend weights and fallback rules scattered across
  modules are hidden configuration, even when there is no YAML file.
- `forward_record.py` (1,011 lines), `sealed_holdout.py` (1,454) and
  `wider_universe.py` (1,614) mix provider access, transformations, policy and
  publication. Extract only shared responsibilities needed to resolve these
  findings; keep CLI adapters thin. Step numbers and claims such as "proven" or
  "noise-free" should not substitute for function contracts and limitations.
- The custom Black-Scholes root solver duplicates standard numerical machinery.
  Consider [SciPy's bracketed `brentq`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.brentq.html)
  after validating pricing assumptions and input alignment; replacing the solver
  alone does not fix R10. Declare any directly used dependency explicitly.

### Verification and documentation scope

Executed 29 tests in the ten focused research/forward/scheduler modules: all
passed in 42.24 seconds. Another 21 continuity/governance/strategy-contract tests
passed in 44.53 seconds. Research-lane Ruff passed; strict mypy passed for 28
source files. Read actual retained snapshot manifests and parquet schemas/sectors,
verified the persisted model payload hashes, and ran small synthetic unit
reproductions for R4 and R5. Synthetic examples are not actual-data integration
evidence. No provider collection, model refit, scheduler execution or full-suite
run was performed. Existing tests mainly check isolated transformations; add the
specific integration and negative cases above before claiming the issues closed.

Documentation feedback: the previous consolidation removed over 27,000 lines
from the three continuity/audit documents and should be retained. The remaining
problem is competing truth: an 885-line README, a 737-line architecture guide,
an 891-line implementation guide and repeated result tables. Keep setup/navigation
in README, durable boundaries in architecture, commands in the implementation
guide, status in plan/handoff and acceptance evidence here. Remove copied current
metrics from the handoff in favor of links. Preserve referenced JSON contracts,
model cards, the hash-bound PDF and historical strategy documents. This review
consolidated the old external-review brief into section 11 of the existing research
brief, preserving its conclusions and all five artifact SHA-256 pins. Markdown
files under `docs/` decreased from 17 to 16, with no new report file. Duplicate
handoff result tables were replaced by links, and README's obsolete 85%/2-GiB
instructions were corrected to the current percentage-only 90% policy. Changed
document links, control characters, retained pins and `git diff --check` passed.

### Step 8 Resolution Ledger for R1–R10 (November 22, Implementation HEAD `716f332`)

All ten review findings (`R1–R10`) are implemented and regression-tested on `main` (`71a79d7..716f332`), preserving all five original retained snapshots (`data/research/research_lane/forward_records/2026-11-16..2026-11-20`), historical scoreboards, and the persisted model bundle (`cd18356701f35b44c884d9f587a8e14ed7473878f12f75e2ec27548b4d8ebf04`) byte-for-byte unchanged:

| Finding | Commit | Regression test(s) | Real-data artifact / measured verification | Remaining limitation |
|---|---|---|---|---|
| **R1** (Prediction clock & pre-entry admission) | `71a79d7` | `tests/test_research_lane_opra_options.py::test_xnys_session_clock_weekend_holiday_and_early_close_settlement`; `tests/test_research_lane_forward_evaluate.py::test_r1_post_entry_and_retrospective_replay_snapshots_rejected_from_prospective_pool`; `test_r1_retained_november_snapshots_stay_untouched_and_zero_qualify_for_sec_model` | `data/research/research_lane/forward_evaluations/prospective_audit_step8/report.json`: flags `2026-11-16`, `2026-11-17`, and `2026-11-18` as `recorded_after_entry` (`21` snapshot-horizon pairs rejected) and admits `0/5` snapshots for `mid_small_beta_neutral_sec_xgb`. | `0` qualifying prospective SEC-model records exist yet; prospective accumulation starts only on the next post-close run published before its next `XNYS` open. |
| **R2** (Decision-date `mid_small_ex_sp500` cohort & `EvidencePaths`) | `35750c8` | `tests/test_research_lane_wider_universe.py::test_r2_retained_decision_date_cohort_matches_batch_security_ids_and_sectors`; `test_r2_relocated_data_root_cohort_sp500_exclusion_ticker_change_and_early_coverage_abstention`; `tests/test_research_lane_two_stage.py::test_forward_score_recorder_writes_snapshot_and_blocks_sealed_window` | `data/research/research_lane/forward_records_retrospective_replay_step8/{2024-04-16,2024-04-30}`: `741/741` `cik:` IDs and sectors match batch `wider_universe` bit-for-bit (`0` `Unknown` sectors, `0` S&P 500 overlaps). Live `--live-prospective --as-of-date 2026-11-20` fails closed (`exit code 2`) naming `sp500_observed_20260821T071500Z_v7` (`2026-08-21`) and `extended_roster_spans.parquet` (`2026-07-08`). | Live prospective scoring on `>= 2026-11-23` stays blocked until S&P 500 point-in-time membership (`> 2026-08-21`) and SEC `roster_spans.parquet` (`> 2026-07-08`) are refreshed through the live decision date. |
| **R3** (Shared 24-predictor Stage-1 baseline & `beta_252d` warm-up) | `5d35e52` | `tests/test_research_lane_sec_catalyst_model.py::test_r3_r4_end_to_end_raw_source_batch_vs_live_parity_short_beta_and_future_poison` | Real CLI replay on `2024-04-16` and `2024-04-30` (`741` eligible CIKs): `score_equal_rank_24_beta_neutral` matches batch `compute_mid_small_stage1_baseline` with `0.0` maximum absolute difference; `< 253` trailing sessions abstain (`no` `60d` beta fallback). | Live prospective runs must also supply point-in-time `company_facts` JSONs, unadjusted daily closes (`>= 253` sessions), and `8-K` Item `2.02` releases for the eligible cohort. |
| **R4** (Full-cohort SEC catalyst ranking & unqueried tracking) | `5d35e52` | `tests/test_research_lane_sec_catalyst_model.py::test_sec_catalyst_bundle_roundtrip_batch_live_parity_and_tamper_rejection`; `test_r3_r4_end_to_end_raw_source_batch_vs_live_parity_short_beta_and_future_poison` | Real CLI replay on `2024-04-16` and `2024-04-30` (`741` queried eligible CIKs, `0` unqueried): all `12` ordered `s2__*` SEC catalyst features and `10d/21d` model ranks match batch with `0.0` maximum absolute difference; partial SEC peer coverage abstains (`incomplete_sec_peer_coverage`). | Live prospective runs must query SEC submissions across the full eligible `mid_small_ex_sp500` cohort (`~740–1,000` CIKs at `<= 10 req/s` SEC rate limit, `~90–120s`) rather than top-100 candidates. |
| **R5** (Freeze decision-time holdings before outcomes) | `88d6a6e` | `tests/test_research_lane_forward_evaluate.py::test_r5_removing_highest_ranked_exit_preserves_selected_holdings_and_marks_incomplete` | `holdings.parquet` persisted at decision time (`74` top-decile `cik:` equities on `2024-04-16` and `2024-04-30`); in `retrospective_replay_step8/report.md`, `3d/5d/10d/21d` cohorts with unresolved selected exits report `n/a` (`top10_complete = false`) while keeping `observed_subset_rank_ic` separately labelled. | Incomplete top-decile cohorts remain `None` until missing exit prices are resolved with documentary corporate-action evidence. |
| **R6** (Grade persisted candidate policy & open-to-open accounting) | `bed8b6b`, `716f332` | `tests/test_research_lane_forward_evaluate.py::test_r6_batch_vs_forward_policy_parity_across_consecutive_rebalances`; `test_r6_incomplete_candidate_definition_abstention_and_pre_entry_pending_status`; `test_step8_cli_entry_points_record_and_evaluate_wiring` | `data/research/research_lane/forward_evaluations/retrospective_replay_step8/report.json`: across consecutive `10d` rebalances (`2024-04-16 -> 2024-04-30`), `mid_small_bn_sec_xgb_10d` matches batch `benchmark_tilt` (`+9.7847%` net vs `+3.91%` `SPY`, `0.0` return diff, `6.94e-18` turnover diff: `50.00%` initial, `5.34%` buffered second rebalance); `prospective_audit_step8` flags `2026-11-20`'s legacy dicts as `incomplete_candidate_definition`. | Prospective `mid_small_bn_sec_xgb_10d` / `mid_small_bn_sec_xgb_opra_10d` (`10d`) and `mid_small_bn_sec_xgb_21d` (`21d`) require `11` and `22` post-decision sessions (`open[t+h+1]`) after the first qualifying record to mature. |
| **R7** (Tri-state `OPRA` regime: `backwardation`, `contango`, `unknown`) | `cb8e68c` | `tests/test_research_lane_opra_options.py::test_r7_tristate_opra_regime_decisive_positive_indeterminate_partial_and_outage` | `mid_small_bn_sec_xgb_opra_10d` in `retrospective_replay_step8` records explicit `abstained` (`opra_observation_not_synchronized_close:not_collected`); indeterminate partial (`SPY <= 1.0`, `QQQ = None`) and outage (`None, None`) resolve to `opra_regime = "unknown"` (`opra_backwardation = None`). | `OPRA` overlay remains a separate `10d` experiment (`mid_small_bn_sec_xgb_opra_10d`), not interchangeable with historical `CBOE` `VIX/VIX3M`. |
| **R8** (Preserve unresolved exits in `continuous_rebalance_windows`) | `88d6a6e` | `tests/test_research_lane_benchmark_tilt.py::test_r8_missing_terminal_exit_disappearing_security_unheld_nan_and_complete_portfolio` | Real-data diagnostic across retained panels (see exact counts below) + `retrospective_replay_step8` (`2024-04-16` `21d` candidate marked `matured_incomplete_exit`, `portfolio_complete = false`). | Multi-year historical compounded return claims (`+44.59%` on `2024–2026`) are incomplete (`portfolio_complete = false`) because delisted/acquired holdings lack documentary cash-out prices in daily bar files. |
| **R9** (Expected model manifest & snapshot payload SHA-256 verification) | `f09df3e` | `tests/test_research_lane_sec_catalyst_model.py::test_sec_catalyst_bundle_roundtrip_batch_live_parity_and_tamper_rejection`; `tests/test_research_lane_forward_evaluate.py::test_r9_payload_tampering_skip_existing_and_swapped_model_manifest_rejected` | `retrospective_replay_step8` manifests bind `payload_artifacts` SHA-256s for `scores.parquet` and `holdings.parquet` and verify `expected_manifest_sha256 == cd183567...` on load and `--skip-existing`. | Original `2026-11-16..2026-11-20` legacy snapshots predate `payload_artifacts` and remain `payload_hashes_not_verified:legacy_unhashed_snapshot` (preserved read-only). |
| **R10** (Synchronized closing `5Min` `OPRA` + underlying bars vs daily VWAP proxy) | `cb8e68c` | `tests/test_research_lane_opra_options.py::test_r10_intraday_underlying_move_synchronized_close_vs_daily_vwap_proxy` | Primary `OPRA` fields use closing-window `[close_utc - 15m, close_utc]` `5Min` option `close` paired with timestamp-matched `5Min` raw underlying `close` (`<= 5m` skew); `1Day` `vw` is quarantined in `vwap_proxy_*` diagnostic columns. | Thinly traded mid/small-cap options with zero trades in the final `15` minutes before `XNYS` close report `daily_vwap_proxy_only` (`None` synchronized closing IV/skew). |

#### Quantified Impact of R8 on Retained Historical Portfolios (*Research Labels*)

Before revising historical return figures, all tradable `(rebalance, security)` cells and buffered top-10% selected `(rebalance, holding)` positions (`w > 0`) were audited on the retained `Step 5c` (`2021-01-04..2024-05-28`) and `Step 6` (`2024-05-29..2026-07-08`) `mid_small_ex_sp500` (`1,653` CIKs) and S&P 500 (`626` securities) panels:

| Window & Signal | H | Rebalance Periods (`K`) | `mid_small` Tradable Cells (Unresolved Exits) | S&P 500 Tradable Cells (Unresolved Exits) | Selected Holding Positions (`w > 0`) | Affected Selected Positions (`Unresolved Exits`) | Affected Rebalance Periods (`Unresolved / K`) | Legacy Last-Bar Fallback Compounded Net `20 bps` | Observed-Periods-Only Compounded Net `20 bps` | Complete Portfolio Return |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `2021-01-04..2024-05-28` (`equal_rank_24_bn`) | `5d` | `170` | `126,122` (`172`) | `79,798` (`53`) | `12,545` | `29` (`0.23%`) | `27 / 170` (`15.9%`) | `+26.66%` | `+15.08%` (`143` periods) | `Incomplete (None)` |
| `2021-01-04..2024-05-28` (`equal_rank_24_bn`) | `10d` | `85` | `63,065` (`172`) | `39,897` (`53`) | `6,272` | `32` (`0.51%`) | `27 / 85` (`31.8%`) | `+36.49%` | `+19.42%` (`58` periods) | `Incomplete (None)` |
| `2021-01-04..2024-05-28` (`equal_rank_24_bn`) | `21d` | `40` | `29,673` (`172`) | `18,775` (`53`) | `2,949` | `30` (`1.02%`) | `21 / 40` (`52.5%`) | `+37.67%` | `+30.69%` (`19` periods) | `Incomplete (None)` |
| `2021-01-04..2024-05-28` (`equal_rank_24_bn`) | `63d` | `13` | `9,643` (`162`) | `6,102` (`52`) | `959` | `21` (`2.19%`) | `10 / 13` (`76.9%`) | `+47.55%` | `+30.15%` (`3` periods) | `Incomplete (None)` |
| `2024-05-29..2026-07-08` (`equal_rank_24_bn`) | `5d` | `105` | `78,462` (`95`) | `49,269` (`38`) | `7,813` | `12` (`0.15%`) | `12 / 105` (`11.4%`) | `+45.15%` | `+37.36%` (`93` periods) | `Incomplete (None)` |
| `2024-05-29..2026-07-08` (`persisted_sec_xgb_10d`) | `10d` | `52` | `38,841` (`93`) | `24,407` (`37`) | `3,867` | `13` (`0.34%`) | `13 / 52` (`25.0%`) | `+43.39%` | `+24.18%` (`39` periods) | `Incomplete (None)` |
| `2024-05-29..2026-07-08` (`persisted_sec_xgb_21d`) | `21d` | `25` | `18,672` (`97`) | `11,733` (`38`) | `1,860` | `11` (`0.59%`) | `10 / 25` (`40.0%`) | `+35.65%` | `+11.23%` (`15` periods) | `Incomplete (None)` |
| `2024-05-29..2026-07-08` (`equal_rank_24_bn`) | `63d` | `8` | `5,961` (`90`) | `3,756` (`35`) | `594` | `10` (`1.68%`) | `5 / 8` (`62.5%`) | `+36.63%` | `+0.60%` (`3` periods) | `Incomplete (None)` |

**Interpretation:** In a `~75`-stock top-decile mid/small-cap basket, `0.34%–0.59%` of selected holding positions (`13/3,867` at `10d`, `11/1,860` at `21d` on `2024–2026`) correspond to issuers that delisted, merged, or halted before `open[exit_row]`. Because each missing position (`~1.33%` basket weight) marks its entire rebalance period unresolved (`13/52` periods at `10d`, `10/25` at `21d`), neither the legacy fallback figure (which substituted pre-halt marks) nor the observed-periods-only figure (which omits entire market-rally rebalance periods) is a complete realized multi-year return. Both remain exploratory development diagnostics until documentary corporate-action settlement prices resolve the missing terminal exits.

## 2. Historical November 21 Feature Acceptance Claims (Not Current Sign-Off)

The table below preserves the November 21 assessment for comparison. Its live
"Verified" and untouched-forward-test claims are superseded by R1–R10 above;
they must not be used as current training/live acceptance. The historical period
`2024-05-29..2026-07-08` was unsealed in Step 6 and remains development evidence.

| Feature Group | Module(s) | Causal Clock & Availability | Batch (`2016–2026`) | Live Prospective (`> 2026-07-08`) | Empirical Status (*research labels*) |
|---|---|---|---|---|---|
| `12` Price & Liquidity Ranks + `[sector, beta, idio_vol]` Neutralization | `predictors.py`, `beta_neutral.py` | Session `t` close (`16:00 ET`), entry `open[t+1]` | Verified (`2016–2026` S&P 500 + `mid_small_ex_sp500`) | Verified (`forward_record.py`, `60d` beta warm-up fallback) | Held positive `mid_small` beta-neutral Base IC on `2024–2026` holdout (`5d +0.0142`, `10d +0.0162`, `21d +0.0172`, `63d +0.0259`); un-neutralized `raw` mid/small lagged mega-cap `SPY` (`+24.99%` vs. `+46.88%` at `63d`). |
| `12` SEC `company_facts` + `8-K` Item `2.02` Fundamental Ranks | `company_facts.py`, `fundamentals.py`, `unadjusted_prices.py` | Usable first session strictly after SEC filing date (`200d` staleness cap); market value = `unadjusted_close * shares` | Verified (`626` S&P 500 + `1,653` mid/small securities) | Not required by `R1`/`R2` live satellite sleeve (`price + catalyst + OPRA` only) | Combined into `equal_rank_24` across `2,253` survivorship-checked CIKs (`100%` SEC XBRL coverage through `2026-07-08`). |
| `1–3` Session Short-Term Reversal Composite | `indicators.py`, `reversal.py` | Session `t` close (`16:00 ET`) | Verified (`trials.jsonl` line `4`) | Verified (`score_reversal_short` & `score_reversal_skew_vetoed`) | Passed development reading rule at `3d` equal-weight; must be gated by `OPRA` `25d` put-skew veto during backwardation sell-offs (`-11.96%` drop on vetoed knives in `2026-11-17..2026-11-20`). |
| Stage-2 Direct-Issuer News & Precise SEC Catalyst Residuals | `news_catalysts.py`, `two_stage.py`, `beta_neutral.py`, `sec_catalyst_model.py`, `forward_record.py` | Exact UTC publication / SEC EDGAR acceptance timestamp `<= close_utc(t)` | Verified (`2019–2026` Alpaca news + `8-K`/Form-4/`13D/G`/`S-3`/`NT` filings; persisted `.ubj` bundle `cd183567...`) | Verified (exact batch `build_precise_sec_catalyst_features` parity + SHA-256 verified `XGBoost` `10d/21d` + Finviz Elite / Alpaca / SEC) | **`mid_small_beta_neutral_sec` held up on `2024–2026` holdout** (`21d` Paired Delta IC `+0.0005 [NW t = +1.97]`, `3/3` yrs; standalone `10d` `R1_base_bn_sec_xgb` `+44.59%` vs. `SPY +44.21%`) and is now persisted & scored live (`score_mid_small_bn_sec_xgb_opra_10d`); historical S&P 500 Alpaca news failed on `2024–2026` (`-0.0004 [t = -0.60]`). |
| Index Volatility Term Structure (`30d / 90d` Backwardation) | `vol_regime.py` (CBOE), `opra_options.py` (live `OPRA`) | Session `t` close (`VIX/VIX3M` historical; `SPY`/`QQQ` `OPRA` `1Day` bars live) | Verified (`2014–2026-07-08` CBOE daily rows) | Verified (`spy_iv_term_30_90`, `qqq_iv_term_30_90`, `opra_backwardation`) | `90_10__spy__R1` tied `SPY` on `2024–2026` (`+44.10%` vs. `+44.21%`, `-0.04%/yr`) and protected capital in live `2026-11-16..2026-11-20` backwardation (`1.1539`), though daily-close `VIX > VIX3M` whipsawed `~50 bps` on V-bounces. |
| Single-Stock `OPRA` `ATM Straddle %` & `25d` Put–Call IV Skew | `opra_options.py`, `forward_evaluate.py` | Session-exact `as_of_date` `1Day` `OPRA` bars (`zero` post-session quote leakage) | Available from `Jan/Feb 2024` (`/v1beta1/options/bars` limit) | Verified (`atm_straddle_pct`, `catalyst_shock_to_straddle`, `put_call_skew_25d`, `falling_knife_put_skew_veto`) | Initial live maturation (`2026-11-16..2026-11-20`, `9` pairs): `falling_knife_put_skew_veto` stocks trailed `SPY` by `-1.66%` to `-6.02%` (`3d`), while `catalyst_shock_to_straddle > 1.0` stocks beat `SPY` (`+0.22%..+1.77%`). |

---

## 3. Historical Governed Swing & Storage Audit Records (Retained Reference)

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
