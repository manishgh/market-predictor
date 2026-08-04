# Active Edge Rebuild Handoff

Status: active

Last updated: 2026-08-02

Repository: `C:\project\market-predictor`

Branch: `main` (fast-forwarded from `r3-lineage`)

Completed implementation checkpoint: `Modularity Refactoring`

## Purpose

Continue the prediction-only edge rebuild without reviving retired sources, invalid
artifacts, or unverified model claims. Run only one memory-heavy process at a time and
keep swing candidate training below 5 GiB process RSS; intraday and serving
workloads retain their 4 GiB limits.

## Verified Current State

- Alpaca is the sole ticker catalyst source in the current estimator.
- SEC authority distinguishes known zero filings from unknown coverage. SEC is
  planned as a separately ablated issuer-specific estimator profile after causal
  collection and exact attachment; it is not permanently restricted to overlay use.
- Finviz and verified global or sector sources remain overlay/audit only.
- Reddit and Seeking Alpha are removed and prohibited.
- Swing model decisions begin on `2019-07-09`; all earlier bars are warm-up only.
- Catalyst V5 identity rebind is published and replayed:
  - 377,778 exact ticker/decision-time matches;
  - 6,359 source-coverage rows;
  - all 604 target securities represented.
- Swing V9 is invalid. Pandas aligned managed-label series on retained global row
  indices, leaving corrupted or absent labels for much of the panel. V9 is retained
  only because V5 binds it as target-lineage evidence; never train from it.
- Corrected swing V10 is published and replayed. Candidate v1 returned
  `no_candidate` before economic gates because the 50-stock hard sector floor often
  produced four eligible sectors while the 20% sector cap required at least five.
  This is a structural selection rejection, not an economic model rejection.
- The implemented policy now uses a within-sector target of 50 and hard floor of 30.
  It persists sector peer count, rank eligibility, target status, and ranking
  reliability weight. Rows with 30-49 peers remain eligible with weight
  `decision_time_sector_peer_count / 50`.
- Portfolio construction targets a 20% sector cap, adapts to 25% with four represented
  sectors and 33.3% with three, and skips sessions with fewer than three.
- Promotion economics now use managed holding-aligned benchmarks, full-calendar
  portfolio returns including cash days and overlapping positions, doubled-cost
  portfolio stress, and a hard 33.3% active-sector exposure gate.
- Strategy contract schema is `edge_rebuild.strategy_contract.v2`. Live processing
  excludes individual unavailable securities through the governed 5% ceiling, and
  cached serving identity includes contract, trust store, promotion policy, and size.
- Swing V11 is published and strictly replayed: 853,417 rows per profile, 604
  modeled securities, 1,759 sessions, and 640,107 rank-eligible rows.
- Swing candidate v2, v3, and v4 all resulted in an immutable `no_candidate` result (none passed both temporal and unseen-security economic gates).
- **Swing V12** is published and strict-replayed, containing advanced technical indicators (`macd`, `rsi`, `dist_15m_high`, `ema_10`, etc.) and cross-sectional rankings/Z-scores.
- Swing candidate v5 trained on V12 features resulted in an immutable `no_candidate` result. All 14 governed candidates failed strict economic gates.
- Intraday V2 is published and replayable but economically rejected after costs. It
  is not a serving fallback.
- **Intraday V3** is fully materialized with technical indicators. Feature predictive efficacies on target-hits were evaluated (MACD ~0.514, EMA/SMA dist ~0.504). It is still reserved for a genuinely future holdout. No V3
  future-holdout run has occurred. **The collection of the future holdout is currently blocked (environment-pending) because the requisite canonical membership data (`sp500_memberships`) and raw market data for `2026-07-09` onwards are not available in the current environment.**
- No promoted bundle exists. All model API behavior must fail closed.
- Documentation is consolidated to this handoff, the active plan, and the current
  feature audit. Unreferenced Azure and standalone TradingFlow plans were removed;
  governance-bound evidence and rejected model cards remain.

## Verification Completed 2026-08-02

- Strict V11 panel replay passed: 853,417 rows/profile, 604 securities, 1,759
  sessions, and 640,107 rank-eligible rows.
- Strict candidate-v2 replay passed with `status=no_candidate`, no model artifact,
  and locked-test access count zero.
- Full suite run: 1,056 passed and 2 skipped. Twelve failures were isolated to three
  workspace-lock permission cases and nine stale intraday-selection fixtures. The
  three permission-affected test groups passed with normal repository access; the
  fixture was corrected for V3 membership lineage and all nine tests then passed.
- Focused swing/governance verification: 28 passed, 1 skipped, plus 10/10 research
  governance tests.
- Ruff, compileall, `git diff --check`, and strict mypy on seven active modules pass.
- Sampled swing training private memory remained below the 5 GiB hard limit.

## Source Boundary

| Source | Permitted role |
| --- | --- |
| Alpaca SIP/all bars | estimator market data |
| Alpaca direct ticker news | ticker catalyst estimator data |
| SEC filings | current issuer authority/audit; planned separately ablated issuer-specific estimator profile after causal collection and attachment |
| Finviz Elite | screening and current metadata only |
| Verified global/sector sources | separate context overlays only |
| Reddit | prohibited |
| Seeking Alpha | prohibited |

Global or sector context cannot become ticker catalyst through topic similarity.
Unknown coverage remains unknown and may cause abstention; it is never encoded as zero.

## Predictive Quality and AUC Direction

An ROC-AUC of 0.85 is not a credible promotion target for broad S&P 500 daily or
swing direction prediction from public market and news data. The current governed
validation result of approximately 0.55-0.57 is weak, but plausible. A sudden broad
result near 0.85 must be treated as a leakage incident until overlapping labels,
post-decision data, revised fundamentals, duplicate events, repeated validation
selection, universe survivorship, and cross-fold security overlap are disproved.

A high AUC may be valid for a narrow event specialist, such as a verified earnings
or material SEC event with abnormal-volume confirmation. Such a result must never be
reported as broad-market performance. Every specialist report must include eligible
event count, security count, calendar coverage, prediction coverage, abstention rate,
sector coverage, and confidence intervals alongside AUC.

Model promotion must optimize useful prediction rather than headline AUC alone:

- expected net excess return or cross-sectional return rank;
- rank information coefficient and top-quantile lift;
- calibrated probability and explicit abstention;
- return relative to SPY, QQQ, and the security's sector benchmark;
- net performance after spread, slippage, turnover, and doubled-cost stress;
- drawdown and stability across time, sectors, regimes, and unseen securities.
### Candidate v2 Failure Attribution

Analysis of the immutable `evaluation.json` reveals:
- **Rank IC and Top-Quantile Lift:** Not computed natively by the v2 binary classification pipeline, underscoring the misalignment between the binary objective and the cross-sectional ranking policy.
- **Regime Instability:** Both `technical_market` and `catalyst_full` profiles lost money in the `risk_off` and `risk_on` regimes across unseen security validation, only profiting during `neutral` regimes. This indicates the model fails to adapt to high-volatility directional shifts.
- **Sector Instability:** The model had severe drags in Consumer Staples (-0.02 to -0.04 average net return) and Industrials across most models, while only Financials showed consistent (but small sample) positive edges.
- **Catalyst Ablation (SHAP/Metrics):** The addition of Alpaca catalyst data slightly improved temporal AUC, but drastically reduced unseen-security net returns (e.g. from +0.027 down to +0.0006 on HGB.leaves_15). This indicates the model overfit to specific historical news events or specific tickers in the training set, rather than learning a generalizable catalyst reaction.

### Candidate v3 Failure Attribution (Experiment 2)

Analysis of the immutable `evaluation.json` for the Candidate v3 (Catalyst Confirmation Overlay):
- **Implementation:** Added a rigid catalyst confirmation overlay directly in `_evaluation_metrics` requiring a recent SEC filing OR Alpaca news/earnings, combined with abnormal volume and non-negative gap return, to filter candidates prior to portfolio construction.
- **Predictive Quality / Economic Gates (Rejection Reason):** The system returned `no_candidate`. Despite broadening the net to include all governed catalyst events, the model still failed.
  1. `threshold selects fewer than two independent sessions`: At higher probability thresholds, the overlap between model confidence and the catalyst overlay was too sparse.
  2. `one or more frozen validation scopes failed economic gates`: Where enough sessions existed, the filtered cohort still failed the rigorous lower-bound confidence gates under stress.
- **Conclusion:** A broad technical ranker cannot be successfully "salvaged" by a post-prediction catalyst overlay. The dataset's noise floor and constraints mandate predicting the post-catalyst magnitude directly (an Event-Driven Specialist).

### Candidate v4 Failure Attribution (Experiment 3: Event-Driven Specialist)

Analysis of the immutable `evaluation.json` for Candidate v4 (Event-Driven Specialist):
- **Implementation:** Pivot to an explicitly Event-Driven cohort by extracting only rows where `source_count_sec_3d > 0` or `source_count_alpaca_3d > 0`. A global cross-sectional rank (top 25% across the entire market) was assigned per session.
- **Predictive Quality / Economic Gates (Rejection Reason):** The system returned `no_candidate` because the final candidates failed the economic gate `portfolio_daily_return_ci_low_positive`. 
- **Conclusion:** While the model correctly ranked the catalysts in probability space, the lower bound of the 95% confidence interval for its net portfolio returns dipped below zero after applying our frozen 20 basis point round-trip cost constraint. The pipeline properly invoked its fail-closed logic. The baseline technical model remains statistically superior.

### Next Preregistered Experiments

Run these sequentially and preserve the locked test for final promotion only:

1. **Broad expected-return ranker (Completed / Rejected).**
2. **Verified catalyst specialists (Completed / Rejected).** 
3. **Modularity Refactoring Checkpoint (Completed).** Extracted swing_training.py into domain-specific modules (training.economics, training.walk_forward, training.evaluation, training.utils) without altering external behavior.

## Immediate Continuation

1. Preserve the replayed V11 and candidate-v2/v3/v4 `no_candidate` authorities.
2. Proceed to the next preregistered checkpoint.
5. Keep the locked test unopened.
6. Do not add an API success path before promotion.

## Do Not Do

- Do not train from V9.
- Do not attach SEC to an estimator until causal collection and exact issuer-specific
  decision-time attachment pass and the separate profile is preregistered.
- Do not reinterpret Finviz or global data as ticker estimator evidence.
- Do not restore Reddit, Seeking Alpha, legacy models, or compatibility commands.
- Do not use bars before `2019-07-09` as modeled decisions.
- Do not evaluate V3 on development-period data or fabricate a future holdout.
- Do not weaken economic or causal gates to obtain a passing model.
- Do not expose an unpromoted candidate through the prediction API.

## Completion Evidence Required

- V11 immutable replay, ranking-policy fields, and eligibility totals;
- swing candidate v2 or explicit no-candidate/rejection authority;
- chronological predictive and economic evaluation;
- repository-wide test, lint, type, compile, diff, and memory results;
- updated artifact inventory and final commit hash.


