# Active Edge Rebuild Handoff

Status: active

Last updated: 2026-08-02

Repository: `C:\project\market-predictor`

Branch: `main` (fast-forwarded from `r3-lineage`)

Completed implementation checkpoint: `be6156b`

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
- Swing candidate v2 is an immutable `no_candidate` result. Six governed models
  trained, but none passed both temporal and unseen-security economic gates. The
  best HGB AUC was approximately 0.55-0.57; all candidates retained a non-positive
  lower confidence bound in at least the calendar, portfolio-daily, doubled-cost,
  and holding-aligned benchmark gates. Locked-test access count is zero.
- Intraday V2 is published and replayable but economically rejected after costs. It
  is not a serving fallback.
- Intraday V3 is development code reserved for a genuinely future holdout. No V3
  future-holdout run has occurred.
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

### Next Preregistered Experiments

Run these sequentially and preserve the locked test for final promotion only:

1. **Broad expected-return ranker.** Replace the broad binary direction objective
   with managed future excess return or its cross-sectional rank. Use causal
   technical, momentum, volatility, liquidity, volume, market-relative,
   sector-relative, and regime features from the existing governed sources.
2. **Verified catalyst specialists.** Train separate earnings and SEC specialists
   only after each source has point-in-time authority, exact issuer binding, and
   exact decision-time attachment. Do not encode missing source coverage as no event.
3. **Confirmation and abstention.** Use specialist evidence, abnormal volume,
   premarket gap, market/sector agreement, calibration, and uncertainty to rank or
   suppress broad-model predictions. Keep catalyst as an overlay unless ablation
   demonstrates stable incremental validation value as a direct model feature.
4. **Optional point-in-time research.** Evaluate options implied volatility, skew,
   term structure, and opening put/call volume only if historically complete,
   timestamped data can be licensed and independently replayed.

Use chronological walk-forward folds with a purge and embargo at least as long as
the prediction horizon, point-in-time universes, unseen-security evaluation, and one
single-touch locked test. Record every attempted hypothesis so repeated validation
search cannot be mistaken for independent evidence. A broad validation AUC in the
0.60-0.65 range would be a material research improvement, not a minimum promotion
gate; economic and calibration gates remain authoritative.

Research basis:

- Gu, Kelly, and Xiu, *Empirical Asset Pricing via Machine Learning*:
  https://www.nber.org/papers/w25398
- Cawley and Talbot, *On Over-fitting in Model Selection and Subsequent Selection
  Bias in Performance Evaluation*: https://www.jmlr.org/papers/v11/cawley10a.html
- Bailey et al., *The Probability of Backtest Overfitting*:
  https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253
- Pan and Poteshman, *The Information in Option Volume for Future Stock Prices*:
  https://academic.oup.com/rfs/article-abstract/19/3/871/1646711

## Immediate Continuation

1. Preserve the replayed V11 and candidate-v2 `no_candidate` authorities.
2. Perform validation-only failure attribution for weak temporal economics,
   unseen-security instability, and catalyst ablation instability.
3. Preregister and run the broad expected-return ranker before changing estimator
   families; keep the locked test unopened.
4. Do not add an API success path before promotion.
5. Run full tests, Ruff, strict mypy, compileall, replay checks, `git diff --check`, and
   the peak-memory audit before committing.

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
