# Active Edge Rebuild Plan

Status: active

Last updated: 2026-09-09

Repository: `C:\project\market-predictor`

Branch: `er-intraday-refactoring`

This is the only active execution plan. Exact artifact state is recorded in
`docs/reviews/active_edge_rebuild_handoff.md`; statistical rules are defined in
`docs/model_training_validation_protocol.md`.

## Objective And Boundary

Current research priority, requested 2026-09-07: **long-only swing stock selection
with verifiable net outperformance of buy-and-hold SPY**. Intraday development and
unrelated structural cleanup are paused, not deleted or declared complete. This is
a research and implementation plan; no profitable candidate is asserted.

The repository's existing supported horizons remain:

- **Swing:** ten-session stock direction, managed return, and excess return against
  SPY, QQQ, and the point-in-time sector benchmark.
- **Intraday:** thirty-minute managed outcome from completed intraday evidence.

This repository produces predictions, abstentions, explanations, benchmark
comparisons, and matured outcomes. It does not produce alerts, orders, positions,
portfolio risk, or execution instructions. `trading_flow` may consume only a promoted,
versioned prediction API.

Offline portfolio accounting is necessary to evaluate predictions. It is not live
portfolio management: alerts, orders, final position sizing, and execution remain
outside this repository.

## Long-Only Swing Research And Implementation Plan

### Current User-Approved Dataset Direction

2026-09-08: the user permits excluding a small number of unusable whole securities
from a new research dataset instead of indefinitely repairing every corporate
action before feature engineering. This supersedes the requirement to repair all
70 old selected outcomes before any new cohort may be built. It does not permit
selectively deleting losing trades, modifying raw archives or certifying unknown
total returns. The original blocked control remains immutable historical evidence.

Immediate scope: freeze an evidence-bound whole-security research cohort, apply it
before peer features/relative labels/selection, and report cumulative exclusions and
sector/year effects. The parent modeled universe is 631 IDs: 27 are already excluded,
leaving 604. The additional eighteen failures would yield 45/631 (7.1315%), leaving
586; the 27 warm-up-only securities are not part of the denominator. The user
explicitly approved a 10% ceiling on 2026-09-08 for this research dataset. This
changes only the research restriction cap to 1,000 basis points, not the exclusion
list, live-inference policy, original universe or model/economic contract.
No reset to 18/604 or exclusion based on later performance is allowed.

This is an explicitly retrospective development restriction, not a historically
observable universe screen. Freeze all IDs and evidence before new fitting/results;
do not extend the list after seeing validation losses. Rebuild features rather than
filtering already computed ranks. Reject unknown IDs, altered lineage, cumulative
cap overflow, inconsistent split/profile cohorts and newly missing selected paths.
Benchmarks remain mandatory. Cohort acceptance is not price-basis acceptance, model
promotion or fresh prospective evidence. Full tests, independent review and a Git
checkpoint close this bounded change before dependent materialization/training.

Cohort implementation is now verified and pushed in `771c7bd`. The real audit at
`data/reports/swing_research_cohort/approved_research_population_audit.json` verifies 85
identity-only monthly projections: 853,417 parent rows, 10,971 proposed removals,
842,446 retained rows and 586 retained securities. Audit hash:
`41de559dd1c415dab60771e10fd489150853a6c2fff07e387d1024b33961efe0`.
The approved cohort hash is
`794bcf834501cbaa8ec54716e8b561a5aaf0137610fe67587f4db69f9189fca6`.
These are coverage counts, not a rebuilt feature authority. The new audit status is
`accepted_research_restriction`; accounting and promotion eligibility remain false.
The prior 5% blocked report is retained as historical evidence, not overwritten.

The original long-only SPY-relative implementation plan below remains the guide.
The retained-stock **holding-identity preflight** is verified and pushed in
`307cffe`. Its bounded scope: hash-bound cohort/parent/membership sources,
metadata-only monthly projections, exact ten-session XNYS windows, full membership
authority (including excluded competing owners), and separate initial-fit versus
full-history coverage. Exit: immutable coverage report, synthetic/tamper/lease tests,
independent code/ML review and full verification. It must not inspect numeric
outcomes, add exclusions or equate S&P removal with delisting. Missing ownership
produces an uncovered report, never a fill or a training pass. Bar availability,
stock/SPY/QQQ/sector numeric paths and total-return reconciliation remain distinct
dependent checks. Only then rebuild labels/features and the six sequential fits.

Real holding-identity preflight completed across all 85 months: 842,446 retained
decisions, 836,638 covered mature windows, 947 uncovered mature windows across
100 securities, and 4,861 terminal immature decisions. Initial fit contains
581,455 mature decisions: 580,889 covered and 566 uncovered across 60 securities.
The uncovered share is 0.1131% of mature decisions, not a whole-security exclusion
rate or missing-price count. Report: `data/reports/swing_research_cohort/holding_identity_preflight.json`,
audit SHA `8f8cdd60ca2f9c1372ceda20d04b7f90eefbfb2336a7f282f30aa97b8c7e723b`.
Peak process memory was 0.231568 GiB. No numeric outcomes/features were read.
The approved exclusion list remains fixed. Full verification: **2,502 passed,
three skipped, 132 warnings in 17m19s**; 133 focused tests pass, full tracked-Python
Ruff and strict mypy on 328 sources pass. Test peak: 351,141,888 bytes (0.327 GiB).
Independent code/ML and test reviews fixed partial-session competing ownership,
open-ended timestamp handling, continuous same-owner metadata changes and duplicate
decision IDs across partitions. All owned agents and workers are closed/exited.
This closes the preflight implementation, not the overall accounting/data step.

The next dependent implementation uses a separate **holding-observation authority**
under `swing/datasets`, bound to this cohort, original raw SIP artifacts and explicit
security/class ownership evidence. Reuse the retained raw-reader semantics in
`research/swing_transfer_sources.py`, not its frozen eleven-stock population.

September 9 bounded implementation, completed/pushed in `89d1aee`: the source-backed initial-fit holding
observation inventory for the 566 flagged decisions across 60 securities. Reproduce
the exact decision/session requirements from pinned parent identity columns before
opening numeric data. Filter raw Parquet at scan time to those required sessions;
validate using the shared outcome checker, retain missing/invalid observations,
and keep source presence, observation validity and ownership as separate fields.
No post-removal ownership authority currently exists: SEC relations inherit index
boundaries and cover-page stock facts are not validity intervals. Therefore these
observations remain diagnostic until independently effective-dated class ownership
is established; do not wire a partial repair inventory as a complete shard source.
Exit: immutable reproducible inventory, independent review, projection/future-poison,
duplicate/clock/missingness/tamper tests, full verification and real-data counts.
No new exclusions, raw modifications, model fitting or unknown-identity fills.

Observed September 9: the initial-fit inventory and independently pinned replay
reproduced 566 decisions / 60 IDs. The 1,106 unique security/ticker sessions contain
876 valid, 209 missing and 21 invalid observations; 574 sessions separately lack
ownership proof. Missing/invalid bars affect 23 IDs; 37 have valid raw bars for all
requirements. The report is `data/reports/swing_holding_observations/_manifest.json`,
audit SHA `b87e18fc2c706600c0063c606c2dd40dbba0422cfad6b6c8c1e4817242714a8a`.
Peak real-run memory 0.342045 GiB. No source download, raw change, extra exclusion,
admitted label or model fit occurred. Full suite: 2,625 passed, three skipped,
133 warnings in 22m09s; full Ruff and strict mypy on 331 sources pass. Independent
review findings were fixed and verified. All agents and worker PIDs are closed.
This inventory narrows the remaining work to effective-dated ownership and explicit
corporate-action/unavailable-trading treatment, followed by verified total-return
accounting. Valid observations alone do not close those gates.

Completed dependent scope (`58468fb`): collect and replay Alpaca corporate-action response evidence
for the initial-fit flagged ticker inventory, using the pinned holding-observation
report. Freeze process-date coverage to 2019-07-09 through 2024-05-28, all action
families, `data_quality=all`, exact symbols and bounded pagination. Keep process date,
effective/ex/payable dates and retrieval time distinct. A terminal empty response
means no provider records returned for that query, not proof of no corporate actions.
No announcement-time feature, ownership interval, cash availability, adjusted-price
conversion or model eligibility is inferred by collection. Retain encoded response
bytes, exact non-secret query, page tokens, hashes and independent failure receipts;
resume completed tickers without refetching, and verify offline. Exit gates: schema,
URL/body/token/tamper/failure-isolation tests, independent review, full verification,
real acquisition summary and pushed implementation/documentation checkpoints.
Then interpret supported events against exact required sessions through the existing
holding/outcome contracts; unsupported records remain explicit evidence gaps.

Real acquisition and pinned offline replay succeeded for all 60 tickers: 649
distinct records (621 cash dividends, 21 mergers, five name changes, two spin-offs).
Audit SHA `82dafea3055db20db9ee483800a7a22c508dbb325418b979a1cd23974a244c91`.
Sixteen mergers fall inside the affected holding windows; fourteen lack payable
dates. Collection is not ownership/accounting admission. Full suite: 2,799 passed,
three skipped, 133 warnings in 18m08s; 213 focused tests, Ruff and strict mypy on
333 source files pass. Peak collection/test memory: 0.238/0.327 GiB. All workers
exited and agents closed. No new model was fitted.

### Approved Event-Aware Accounting Change

On September 9 the user explicitly approved replacing price-only accounting with
separate tradable shares, available cash, unpaid proceeds and contingent rights.
This supersedes `verified_total_return_units_no_separate_distributions` for the
next implementation. It does not change the ten-session forecast horizon, costs,
funding order, 10% exclusion cap or fixed 45/631 list. The ABMD completion filing
establishes cash plus a nontradeable contingent right: its payout cap is neither a
valuation nor immediately reusable cash. No unsupported fact is assigned zero.

Completed code scope (`b7cdb4e`): strict typed holding/evidence contracts and one canonical lot
transition/valuation kernel consumed by labels and the existing funded ledger.
Replace terminal-mark liquidation with component reconciliation. Payment must
extinguish its matching claim exactly once; only evidenced available cash funds
purchases. Residual claims may survive horizon without being sold or extending the
forecast. Missing valuation makes NAV/return unavailable and blocks NAV-dependent
allocations. Report tradable, unpaid and contingent exposure separately; do not
invent risk weights. Unsupported event ordering, delivery, currency conversion or
fractional treatment remains a gap. Existing adjusted-price diagnostics cannot be
silently converted to admitted explicit-distribution accounting.

In scope: research/holding contracts, lot kernel, label target projection, canonical
ledger/accounting consumers and focused synthetic parity/poison tests. Out of scope:
new universe restrictions, intraday, fake real-data admission, feature materialization
or six-model fitting before the dependent evidence gates pass. Exit tests: ordinary
stock parity; cash plus unvalued CVR; payment after horizon; event after stop and
previously earned surviving claims; duplicate payments; unknown timing/valuation;
double-counted distributions; identical lot economics through label and portfolio
consumers. Consolidated code/ML review, full verification and Git closure are required.
Real source interpretation/materialization follows this shared implementation.

Verification closed September 9: 309 focused tests; full tracked-Python Ruff and
strict mypy on 336 source files; full suite 2,974 passed, three skipped, 133 warnings
in 18m16s, peak 0.329 GiB. Consolidated review fixes cover entry-tied event ambiguity,
prior-session sale cash availability, unearned post-exit events and premature
successor references. All owned agents/processes exited. Target projection and
funded accounting share the kernel; source admission remains false. No new fit ran.

Completed collector dependency (`46ab0f8`): make the existing exact-unit daily collector explicitly
price-basis aware, from transport through immutable request, receipts and replay.
Reuse its collection path; accept only supported adjustment choices and reject
cross-basis resumes. Existing adjusted archives remain unchanged. Bind a fresh
initial-fit raw-share SIP acquisition plan for the fixed cohort, SPY, QQQ and required
sector benchmarks, ending 2024-05-28. No held-out numeric observations may enter it.
This collection establishes raw price evidence, not ownership or distribution
admission. Then join independently interpreted actions/ownership into the actual
holding-specification and target materializer. Do not substitute another diagnostic
inventory for that materializer. Required tests: transport query, request identity,
wrong-basis response/resume rejection, bounded date/unit scope, offline tamper replay,
failure isolation and existing adjusted collection behavior. Fail closed on missing
evidence; do not expand the frozen exclusion list or fabricate cash/marks.

Collector verification: 53 source/collector tests plus 14 combination tests pass;
full tracked-Python Ruff and strict mypy on 336 sources pass. The complete suite
passed 2,999 tests, three skipped, 134 warnings in 19m41s, peak 0.329193 GiB.
Consolidated review found and fixed rejected transport receipt loss; the regression
replays the rejected query from retained bytes and verifies sibling resume. Retained
adjusted warm-up source replay passed for 549 units and 140,383 rows, peak 0.132084
GiB. Both agents and process PIDs25540/12624 exited. No new source collection or fit
ran. The initial-fit raw-plan and source-to-target materializer are not completed by
this transport checkpoint; they are the next bounded dependency below.

The raw-plan publication substep will bind the existing accepted cohort and full
membership provenance, project identity columns only from the parent partitions,
and reconstruct decision sessions plus the exact ten-session paths for mature
initial-fit decisions. Merge contiguous required sessions per security/ticker;
do not clip them at S&P removal. Include SPY, QQQ and the complete point-in-time
sector benchmark set across the full permitted initial-fit interval. Bind all 586
retained IDs, including the 41 with no in-window membership, without treating those
41 as exclusions. A metadata-only advisory found 545 in-window IDs, 551 stock runs
and 13 benchmark units (564 total) before independently evidenced successor needs.
These are expected reconstruction counts, not an already published authority.
Reuse the acquisition writer and collector, preserving their exact hash serialization.
Exit: pinned immutable plan, independent reconstruction and source-tamper tests,
scope/benchmark/session checks and no numeric held-out reads. Source presence after
collection still does not establish ownership or cash availability.

Read required sessions directly from raw artifacts and share
`validate_outcome_observations()` / `outcome_bar_lookup()`. Keep
`ownership_unresolved`, `observation_missing` and `observation_invalid` distinct.
For the initial-fit diagnostic, push the required-session/date filter into the
Parquet scan before returning numeric rows; do not reuse the transfer helper's
decode-all-then-filter behavior to inspect validation/test prices. Full-history
identity geometry remains separate from the authorized numeric diagnostic scope.
Pass admitted independent observations to `build_swing_feature_rows(outcome_bars=...)`;
do not alter decision membership, peer transforms or their feature-history loader.
The raw request spans the full modeled horizon and previously examined cases had
91 omitted combined-history sessions, but this does not prove these 100 securities'
coverage. Do not infer post-removal ownership from CIK equality, a retained ticker
or an SEC relation whose interval was copied from index membership. No broad
download, new exclusion or accounting acceptance is authorized by this preflight.

Source research checked September 8: Alpaca's
[corporate-actions endpoint](https://docs.alpaca.markets/us/reference/corporateactions-1)
provides split, dividend, spin-off, merger and name-change records, with pagination
and process-date interval semantics. Its creation/availability timing is explicitly
not guaranteed, so a historical response is not a causal announcement feature.
Validate required record fields even with `data_quality=complete`; the documented
response can still include incomplete records that have already been processed.
`sources/alpaca.py::fetch_security_transitions()` already uses that endpoint but
selects only name changes, mergers and reorganizations; it is not a distribution
or adjustment-factor authority. Reuse its transport ownership while preserving raw
response bytes and complete coverage receipts for the new accounting evidence.
The [bar reference](https://docs.alpaca.markets/us/reference/stockbars) distinguishes
split, cash-dividend and spin-off adjustments; `all` includes them, and `asof`
is documented as symbol/entity mapping, not an adjustment-vintage selector. These
documented semantics guide reconciliation; they are not per-stock price proof.

Previous committed verification: 2,422 tests passed, three skipped, 133 warnings; full Ruff and strict
mypy on 326 source files pass. Peak test-process memory was 348,155,904 bytes.
Consolidated review fixed explicit retrospective scope propagation and relative CLI
root handling. The first full run found a missing CLI inventory entry; that entry
was added and the final complete suite passed. All owned workers/agents are closed.
No raw sources, original controls, training labels or fitted models changed.

### Decision And Scope

The recommendation is a **benchmark-aware cross-sectional return model conditioned
on issuer news and the price/volume reaction already observable at decision time**.
Start with the existing ten-exchange-session horizon and point-in-time S&P universe.
Do not restart intraday, buy another data feed, change language, or launch another
repository refactor to do this research.

The economic question is: does a funded, unlevered, long-only stock portfolio end
with more money than SPY over the same calendar period, after costs? A high win rate,
positive average trade return, or AUC of 0.60 does not answer that question. A stock
that rises 1% while SPY rises 2% has positive direction but negative benchmark excess.
No paper, software design, or model provider can guarantee future outperformance.

This section supersedes the earlier four-model *work sequence* and its proposed
universal AUC gate for this new campaign. Existing artifacts and their rejection
decisions remain unchanged. Changes to labels, estimator sources, selection, or
promotion require new explicit contracts and tests before implementation; this
document does not silently modify a frozen model or authorize serving.

### What The Existing Evidence Establishes

- The retained technical authority has 853,417 rows, 604 securities, and 1,759
  sessions. This is substantial data, not 853,417 independent market histories.
  Stocks share market shocks and overlapping ten-session outcomes.
- The corrected broker-action comparison has 27,087 matched prediction rows from
  11,720 unique latest announcements per profile. The historical 113/19-event
  reductions were not evidence that the entire news archive was that small.
- Reported directional broker experiments reached worst-scope inner AUC of 0.524
  for upgrades and 0.552 for downgrades; they did not produce a candidate. This is
  evidence against those exact models, not against every news-conditioned strategy.
- Current technical labels emphasize a top sector-relative quantile of managed
  return. That is different from estimating the amount of future SPY excess.
  Subtracting the same SPY return from every stock on a date does **not** change
  their order; changing the target alone cannot create a predictive signal.
- The latest implementation checkpoint, `880f2a8`, improved outcome/drift correctness
  and passed its tests. It did not train a new model or produce economic evidence.
- The retained August 9 `data/models/swing/technical/evaluation.json` explicitly
  records `locked_test_outcomes_read=true` and `test_access_count=1`. Its reported
  July-2025 through June-2026 results are already exposed. Later specialist runs
  keeping their own locked reads at zero does not make that calendar globally fresh.
- The reviewer verified a historical inconsistency in that August 9 V11-lineage
  bundle: candidate eligibility is true despite failed selected economic gates.
  Current evaluation requires those gates; no current bypass was reproduced. Do not
  use that old bundle as evidence of a qualified current baseline or deployment.
- In that old run, classifier/regressor managed portfolios reported approximately
  +4.49%/+7.91% compounded return, yet approximate managed SPY excess per selected
  trade was -27.37/-9.27 bps. Unseen-security portfolios returned -1.58%/-0.23%.
  These are exposed historical diagnostics, not exact portfolio SPY excess and not
  performance of the current technical authority.
- A semantic conflict was reproduced and corrected in `6758671`: approximate
  managed-exit-session-close excess no longer authorizes or ranks swing candidates.
  Reporting now matches the required funded daily SPY comparison. Daily benchmark
  closes are not contemporaneous stock barrier quotes; the old numerical bias is
  unmeasured, and correcting accounting does not automatically create alpha.
- The May-2019 schedule/missing-warm-up language in the older validation protocol was
  stale and is corrected in `3b2bff5`. Current config begins decisions 2019-07-09, and the panel request records
  history beginning 2018-05-29. Do not download that history again based on stale prose.

Immediate diagnosis must distinguish six measurable causes: weak stock ranking,
return-magnitude mistakes, missing/misclassified catalyst information, exit-policy
losses, transaction costs, and benchmark exposure/cash/sector effects. These are
hypotheses until attributed from the actual ledger, not excuses for a rejected run.

### Research Basis And Limits

| Primary evidence | Relevant result | Application here and important limitation |
| --- | --- | --- |
| [Gu, Kelly and Xiu, Empirical Asset Pricing via Machine Learning](https://academic.oup.com/rfs/article/33/5/2223/5758276) | Nonlinear interactions improved return forecasts; momentum, liquidity and volatility were influential. | Use regularized and shallow tree return models as controlled comparisons. Their sample covers approximately 30,000 stocks over 1957-2016, largely monthly forecasts. It does not establish ten-day, long-only S&P alpha. |
| [AQR, Implementing Momentum](https://www.aqr.com/Insights/Research/Working-Paper/Implementing-Momentum-What-Have-We-Learned) | Examines seven years of live momentum implementation and returns after multiple real-world frictions. | Evidence that implementation matters, not a promise that a new swing model will work. This is manager-authored evidence. |
| [AQR momentum methodology](https://www.aqr.com/insights/datasets/momentum-indices-monthly) | Uses prior twelve-month performance excluding the latest month and quarterly reconstitution. | Separate slow momentum context from short-term entry/reaction features. Do not cite this as evidence for rapidly rotating ten-day positions. |
| [Acadian, Machine Learning in Quant Investing](https://www.acadian-asset.com/au/investment-insights/systematic-methods/machine-learning-in-quant-investing-revolution-or-evolution) | Demonstrates nonlinear enhancement of a financially motivated signal and emphasizes explicit research discipline. | Learn conditional relationships with a small, motivated feature set, not an unconstrained indicator search. Its case study is not our holding horizon or a replication of its live strategy. |
| [Novy-Marx and Velikov, Trading Costs](https://www.nber.org/papers/w20721) | Buy/hold separation reduces turnover; many high-turnover anomalies lose significance after costs. | Measure replacement benefit versus cost and inspect how many positions the policy churns. Evidence comes from anomaly portfolios, not a guaranteed threshold for this system. |
| [Frazzini, Israel and Moskowitz, Trading Costs of Anomalies](https://www.aqr.com/insights/research/working-paper/trading-costs-of-asset-pricing-anomalies) | Uses institutional live trades to study costs, capacity and cost-aware implementation. | Report size-dependent slippage and stress costs. Institutional execution estimates cannot simply be borrowed for a personal account. |
| [Daniel and Moskowitz, Momentum Crashes](https://www.nber.org/papers/w20439) | Momentum losses depend on panic/rebound conditions. | Test interactions with observable volatility and market rebound state. Much of this evidence concerns long-short momentum; do not transplant its crash statistics to our long-only portfolio. |
| [Savor, Stock Returns After Major Price Shocks](https://faculty.wharton.upenn.edu/wp-content/uploads/2012/10/Stock-Returns-After-Major-Price-Shocks---May-2012---Final.pdf) | Studies 1995-2009 shocks and subsequent 5/10/20/40-trading-day responses; analyst-report-associated shocks behave differently from other shocks. | Test information type multiplied by already-observed reaction. Its event classification can use day +1 information; we must not copy that into a day-0 prediction. Evidence is not a costed long-only SPY strategy. |
| [Tetlock, All the News That's Fit to Reprint, October 2010 manuscript](https://business.columbia.edu/sites/default/files-efs/pubfiles/3099/Tetlock%20Fit%20to%20Reprint%2010%2010.pdf) | Examines over 850,000 news firm-days in 1996-2008; repeated issuer text is associated with subsequent short-horizon reversal. | Measure novelty against prior available stories, not headline tone alone. The study's five-day reversal evidence is not ten-day long-only net performance. |
| [DellaVigna and Pollet, Investor Inattention, December 2006 working paper](https://eml.berkeley.edu/~sdellavi/wp/earnfr06-12-11NewTitle.pdf) | Studies 1995-2004 earnings; attention affects delayed reactions. | Motivation for earnings reaction, not a Friday trading rule. It relies on analyst expectations and much longer delayed-return windows than our ten-session forecast. |
| [Lerman and Livnat, The New Form 8-K Disclosures, March 2008 working paper](https://pages.stern.nyu.edu/~jlivnat/f8k%20current.pdf) | Studies item-specific filing reactions in 2005-2006, including later 30/60/90-calendar-day returns. | Form occurrence alone is not positive news. Verify content and first disclosure; do not imply ten-day SPY alpha from an 8-K flag. |
| [Bailey and Lopez de Prado, Deflated Sharpe Ratio](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf) | Selecting the best of many trials inflates apparent performance. | Record all model/policy trials and assess uncertainty using time blocks. A correction does not turn reused historical validation into fresh evidence. |

These sources motivate experiments, not advertised expected returns. Public material
from quantitative firms does not disclose a reproducible recipe for their proprietary
alpha. No claim is made that this design copies a successful fund.

### Proposed Prediction And Evaluation Contract

1. **Universe and clock.** Reuse eligible point-in-time S&P members, including former
   members. Finviz may inspect today's candidates but may not select a historical
   training population using today's winners. Score after the configured completed
   session cutoff, enter at the next exact session open, and count ten actual exchange
   sessions. News arriving after cutoff belongs to a later decision.
2. **Return target.** Add a separately named fixed-horizon total-return target:
   `stock_next_open_to_tenth_close - SPY_same_interval - stock_round_trip_cost`.
   Train magnitude as well as ordering. Preserve raw gross returns and cost columns;
   cost adjustment occurs once. QQQ and the sector ETF remain reported diagnostics,
   not three independent markets the model must simultaneously beat to satisfy a
   SPY-specific objective. Required benchmark data integrity remains strict.
3. **Managed outcomes.** Keep the existing 3 ATR target, 1.5 ATR stop, ten-session
   timeout as the control execution policy. Fixed-horizon forecasts are not falsely
   labeled managed-profit forecasts. Evaluate every candidate using the actual
   frozen exit policy. A second, preregistered research policy may remove only the
   profit cap while retaining the stop and ten-session maximum; it is an experiment,
   not an already approved live change. This tests whether early profit-taking loses
   continuation gains. Do not optimize a grid of stops or extend the holding period
   until a backtest looks good.
4. **Benchmark honesty.** A daily high/low reveals a possible barrier hit but not the
   simultaneous SPY quote. Keep exact fixed-horizon comparisons separate from
   managed-exit-time estimates. If exact intraday benchmark prices are unavailable,
   mark that trade-level comparison unavailable/approximate. Full-calendar daily NAV
   versus buy-and-hold SPY is still a valid portfolio comparison with documented fill
   assumptions; do not pretend an exit-session close is the intraday exit instant.
5. **One funded account.** Replay available cash, marked holdings, entries, exits,
   exposure, and total equity each session. Overlapping recommendations cannot each
   spend the entire account. Aggregate repeated positions, enforce funded capacity,
   include idle days and capital tied up in losing positions, and charge only actual
   transactions. Report realized and unrealized P&L. Unit-test reconciliation.
6. **Costs and distributions.** Start from the existing 20 bps round-trip estimate
   and 40 bps stress case; neither is claimed to be measured live slippage. Model gaps
   and conservative ambiguous fills. Verify split/dividend treatment for stocks and
   SPY, using either verified total-return series or explicit cash distributions,
   never both. Check adjusted prices are not mistaken for executable raw prices.
   Cash yield is explicitly zero unless a contemporaneous cash-rate source is bound.
   Results are pre-tax; account-specific taxes are not silently estimated.
7. **Attribution, not benchmark substitution.** Report full-account excess versus
   SPY, beta/exposure, cash drag, sector contribution, concentration, turnover, and
   costs. Also compare an exposure-matched SPY diagnostic and the same-universe
   deterministic control. Neither may replace buy-and-hold SPY as the headline
   benchmark. Do not add leverage, shorting, or a permanent SPY allocation to make the
   result pass. An index-core allocation would be a different approved experiment.

### Feature And Data Design

Use one global stock model, not a model per ticker or sector. Preserve the existing
technical source contract as the comparator. Inspect the actual ordered feature list
before adding anything; the following are groups, not permission to duplicate inputs.

| Group | Candidate information available before entry | Historical requirement |
| --- | --- | --- |
| Medium-term relative strength | Existing 20/60-session stock/sector/market returns; distinct six/twelve-month momentum excluding the latest month where supported | Complete price warm-up; insufficient windows excluded, not shortened silently |
| Short-term reaction/reversal | One/five-session residual move, overnight gap, close location, distance from trend, rebound after a pullback | Separate information already reflected in price from return after next-open entry |
| Volume and liquidity | Trailing dollar volume, abnormal volume against lagged history, volatility-normalized move, liquidity/size proxy | SIP feed known; current observation excluded from its baseline |
| Regime interactions | Existing SPY/QQQ/sector trend, realized volatility, market breadth, stock beta/residual return; interaction with short-term reaction | Causal rolling estimates and contemporaneous universe; no retrospective bull/bear labels |
| Issuer news | Event family/direction, age, novelty, deduplicated recurrence, issuer precision, and source coverage | Archived exact article/version, security identity, availability and cutoff; no latest edited text leaking backward |
| Event response | Observed stock-minus-market/sector reaction and abnormal volume in an explicitly defined post-release or announcement window, plus event-age interactions | Verify both start and end boundaries; no future reaction, high, closing volume, or pre-release move mislabeled post-release |
| SEC issuer events | Filing type and material content, earnings releases, financing/dilution, or business changes when verified | Accession and acceptance time, original exhibit/fact availability; no restated latest facts assigned to old dates |

Prioritize **earnings/guidance continuation** and **news-conditioned continuation
versus covered-source no-qualifying-event reversal** as event hypotheses. Broker upgrades/downgrades remain
comparators rather than the sole definition of catalyst. Negative issuer news can
help avoid a long purchase; a negative event is not an instruction to short.

An earnings "surprise" requires an estimate known before the announcement. Without
that evidence, use honestly named reported growth/guidance changes or observed price
reaction; never synthesize an analyst consensus from the later actual. SEC reporting
time is not necessarily the first earnings-announcement time. FinBERT tone alone is
not economic surprise. An LLM may assist extraction only with a frozen auditable
extractor, not use hindsight to decide whether an old headline was important.

For earnings-reaction continuation, admit reported issuer results, not previews or
calendar notices. For guidance, require an explicit issuer guidance raise/lower for
an identified fiscal period; reaffirmation and initiation are different events.
Supporting source spans must be retained. Waiting for a completed post-announcement
reaction session moves the decision and next-open entry forward. An after-close
announcement cannot already have tomorrow's reaction as an evening feature.

Freeze the reaction start as well as its end. With daily evidence, a strict
post-release reaction can use the first complete session whose open follows event
availability: same-day for a premarket release, next-session for an intraday or
after-close release. An event-day close-to-close return or gap may also be useful,
but call it **announcement-window context**, not a pure post-release move; it can
contain trading before the release. More precise intraday reaction requires exact
retained bars with starts after availability and independently accepted coverage.
Test that pre-release price moves cannot enter the post-release reaction feature.

The SEC manifest inspected in this review reports 689,467 filing events across 624
issuers and a companion 853,417-decision authority. These counts are manifest
inspection, not a fresh full replay. SEC is therefore not wholly absent. However,
current canonical form-level text does not prove original exhibit/content coverage,
EPS, guidance or 8-K item meaning; inspect retained document bytes before collection.
Both historical news and SEC timing remain retrospective research evidence where
first-observed history is unproven. Do not manufacture prospective availability.

Audit precision **and recall** on development-only article samples stratified by
event family, year, issuer and market session. Record actual source denominators,
unique announcements, duplicate reports, and attached stock decisions separately.
Include unclassified and rejected articles as well as detected events in the recall
sample; reviewing only admitted events cannot measure what the classifier missed.
Name the negative control **no qualifying event in the covered source window**, not
"no public information". Provider coverage cannot prove that no news existed
elsewhere; unclassified evidence must not automatically certify a clean control.
Ground truth must be independently source-checked; an extractor grading its own
answers is not an independent audit. Record whether labels were human-reviewed or
model-assisted and retain research-only status where admission evidence is insufficient.
Weak classification of one family must not make all other news disappear. Missing
coverage is not neutral sentiment and does not mean "no catalyst". New feature
profiles must explicitly define nulls, required-source abstention and optional-source
missingness; never silently switch the source set of a fitted model.

Global events are separate market/sector context. Start with observable market and
sector reactions already supported by the bars. GDELT/open-source flashpoint text
may enter only through a separate timestamped, full-history, preregistered ablation;
no fabricated war labels, current supply-chain maps applied backward, or ticker
attribution merely because the article mentions oil or technology. It is not a
dependency of the first bounded experiment.

Existing archives are the first source. Backfill accepted new transformations across
the complete permitted history. Download only an enumerated, genuinely missing
source/date range after checking raw archives and permissions. Missing SEC or news
coverage must be reported by ticker/year and source; it cannot be replaced with zero.
The first decision date remains 2019-07-09; source coverage before then is irrelevant
to modeled decisions. Retain the 5% whole-security exclusion rule and affected-window
handling, and show whether exclusions distort the universe.

### Bounded Experiment

Compare two existing-stack learner families: **regularized linear return regression**
and **shallow gradient-boosted return regression**. Forecast continuous return rather
than force every estimator through a top-quantile binary target. Rank predictions by
decision date. An independently calibrated binary view may still report the chance
of positive excess, but an uncalibrated score is not a probability.

Use at most six learned specifications: those two learners crossed with these three
feature profiles. Freeze exact columns and estimator settings before inspecting new
validation outcomes. Repeated folds/refits are logged but do not license new searches.

| Profile | Purpose |
| --- | --- |
| Existing technical features | Isolate the objective/learner change from new information |
| Technical features plus distinct price/volume/regime relationships | Test incremental relationships, not another collection of redundant indicators |
| Same relationships plus qualified issuer news and SEC reaction features | Measure catalyst value on exactly matched decisions as well as overall deployable coverage |

Source-blocked profiles are reported as not trained; train ready technical profiles
without pretending they include news. Paired comparisons use identical security/date
rows, weights, labels, splits, costs, and policies. Also report each profile's actual
full-universe coverage so a highly filtered event cohort cannot hide poor opportunity
coverage. Include known negative/non-event outcomes, not only stocks that later rose.
Neither a losing deterministic baseline nor a prior rejected specialist prevents a
valid new estimator from being trained.

Control models are SPY buy-and-hold, a same-universe simple momentum rank, and the
existing technical formula on the same policy. They are not extra learned families.
The two predefined exit policies give at most twelve learned model/policy comparisons;
they count toward the complete trial record. No post-result threshold grid, seed
shopping, sector winner-picking, or retrospective best holding-period selection.

A portfolio replacement must cover its incremental cost. Freeze entry/hold separation
and capital/sector constraints before evaluation, using current constraints as the
control. Do not reduce a sector peer threshold or a risk limit simply to get trades.
Measure losses caused by those constraints first. A change to sector-relative ranking
eligibility or hard sector allocation is a separately named contract decision.

### Validation And Success

- The reconciled split config and feature audit describe initial fit
  2019-07-09 through 2024-05-28, a ten-session embargo, 252 validation sessions, and
  historical test 2025-07-01 through 2026-06-30. The protocol's obsolete May-2019
  requirement was removed in `3b2bff5`. Do not move the approved cutoff or claim
  that resolving this prose discrepancy restores holdout freshness.
- Use session-grouped rolling/expanding development folds inside the permitted fit
  range, with purging based on actual label end time and at least the ten-session
  embargo. All fitting, normalization, clipping, feature selection and calibration
  precede scored dates. Give each date controlled weight; duplicated news decisions
  must not multiply one announcement into independent evidence.
- The July-2025 through June-2026 holdout has already been evaluated by the retained
  August 9 technical run. Its results are historical/exploratory for this new design,
  not an untouched final test. Preserve all genuinely unobserved outcomes by an
  access-audited manifest; use a newly accruing prospective holdout when none exists.
  Reviewed validation remains development evidence even when called out-of-sample.
  Do not relabel old dates "untouched" after three months of experiments.
- Primary outcome: positive net CAGR difference versus SPY on the complete funded
  equity curve, with a positive mean daily active return and uncertainty measured
  from **time blocks**, not a bootstrap of individual stock rows. Use the existing
  20-session block length as the primary setting; 40-session sensitivity must be
  reported rather than cherry-picked. Report confidence bounds and effective sample
  length, including the dependence from overlapping holdings.
- For promotion, require positive base-cost excess, survival under doubled costs,
  the frozen drawdown/capacity limits, and statistical evidence of positive active
  return after accounting for the complete trial search. Freeze the exact confidence
  procedure in the first checkpoint. A positive point estimate with an interval
  spanning zero is **inconclusive**, not proof of edge and not a fabricated failure
  to collect enough rows. No claim of 0.85 AUC or guaranteed CAGR.
- Report rank correlation, top-ranked return, calibration where applicable, turnover,
  drawdown, sector/year/regime attribution and unseen-security stress. AUC is a
  diagnostic for an explicitly named binary target, not an arbitrary universal veto
  on a useful continuous-return model. Beating QQQ every year is not the SPY mandate.
- If no model passes, publish the measured reason: weak ranking before costs, costs
  consuming gross edge, exit-policy damage, concentrated exposure, or uncertain
  evidence. Do not resume unbounded tuning on the same validation window.

### Ordered Checkpoints

The user initially approved two checkpoints, then explicitly authorized continuing
the remaining work through model training on 2026-09-07. The objective/evidence
checkpoint is complete. Accounting code is verified and pushed in `6758671`, but
retained-data acceptance remains blocked by 70 incomplete selected outcomes and
unverified price basis. The September 8 whole-security research restriction above
replaces the requirement to repair all eighteen affected securities before a new
cohort. Retained paths and price basis still require acceptance before dependent
training; the expanded authorization does not waive causal or accounting gates.
Names describe behavior rather than experiment serial numbers.

1. **Define the SPY objective and reconcile evidence (`complete`).**
   Freeze the new research objective, exact candidate/policy budget, chronological
   split, capital/cost assumptions, statistical procedure and target semantics.
   Reconcile conflicting current documents/configs without rewriting old artifacts.
   Produce a source/feature/experiment inventory and an access record for held-out
   data. Map each proposed relationship to an existing feature or a real gap.
   Owners: `configs/edge_rebuild_swing_training.toml`, swing/strategy contracts,
   temporal manifest, current feature audit, and the governing validation protocol.
   Exit: one reproducible contract and no contradictory current instructions; no
   training or locked-outcome access. Review: independent ML/economics reviewer.

2. **Reconcile returns, capital and SPY accounting (`in progress`).**
   Implementation is verified/pushed (`6758671`); real-data acceptance is blocked,
   not complete. The frozen control selected 30,525 stock-days before loading
   outcomes; 70 have incomplete fixed-horizon labels. No filtered replacement
   population or real-data accounting result was produced. Exact evidence follows
   under Research Checkpoint Status and in the current feature audit.
   Extend the existing label/evaluation owners with named fixed-horizon return
   evidence and a funded daily ledger. Replay retained development predictions only
   if immutable row-level prediction evidence actually exists. The inspected
   specialist rejection artifacts retain aggregates, not fitted models or prediction
   rows. Otherwise verify accounting with real-data deterministic controls and unit
   fixtures, then attribute learned-policy results from step 4's saved chronological
   out-of-fold predictions. Never score fitting data with a final fitted model and
   call it historical out-of-sample replay. Attribute gross ranking, managed exits,
   costs, cash and sector effects where row-level evidence supports it; do not call
   an accounting repair a new alpha result.
   Owners: `swing/evaluation/ledger.py`, `swing/evaluation/accounting.py`,
   `modeling/resampling.py`, `research/swing_accounting_control.py`, the existing
   label owners and retained trainer consumers. The ledger and bootstrap have one
   canonical implementation; no old definitions, compatibility re-exports or
   dependency-boundary exceptions remain.
   Exit tests: identical stock/SPY produces zero gross excess; one cost deduction;
   overlapping trades conserve cash; no negative cash/leverage; no free dividends;
   mark-to-market drawdowns; gap/collision cases; deterministic replay and tamper
   rejection. Exact managed benchmark unavailability remains visible.
   Reviewed admission boundary: accounting-code acceptance is separate from
   total-return evidence. Existing `adjustment=all` metadata does not independently
   reconcile distributions or prove executable raw fills. Until scope-matched
   source proof exists, emit price-ratio diagnostics and `price_basis_pending`,
   with economic eligibility false. Do not accept a caller's `passed=true` assertion
   as proof. No speculative bulk collection is required for this code checkpoint.

3. **Complete causal news and reaction features (`pending`).**
   Audit existing Alpaca/SEC artifacts, broaden eligible issuer categories only after
   development precision/recall evidence, and backfill the accepted feature profiles
   across the existing horizon. Expose the same transforms in batch and inference.
   Owners: `catalysts`, `swing/news_source_inventory.py`, `swing/datasets`,
   `swing/features/pipeline.py`, `catalyst_aggregates.py`, `technical_relationships.py`,
   and `serving/swing_features.py` where the current owner requires it.
   Exit: per-ticker/year coverage, first/last usable news and bars, exclusions, event
   counts, feature availability, and full vertical feature acceptance matrix.
   Tests: after-close news, revised text, wrong issuer, duplicate announcements,
   known-zero versus unknown, SEC acceptance timing, future-poison and batch/live
   parity. No new feature is "done" with batch-only implementation.

4. **Train the six bounded return-model comparisons (`pending`).**
   Use the existing Python stack and shared data IO. Fit models sequentially with a
   workspace lease, bounded projected batches and a 5 GiB process-memory limit.
   Store every development prediction with source/feature/split/model identity.
   Owners: `edge_rebuild/swing_training.py`, `edge_rebuild/training`, `modeling`,
   `swing/contracts/model_artifact.py`, and thin research CLI adapters.
   Exit: reproducible fits, chronological calibration, matched-profile comparisons,
   date-weighted metrics, complete failure records, and no held-out data reads.
   No more feature changes are permitted after this checkpoint's validation starts.

5. **Evaluate the frozen long-only policies (`pending`).**
   Replay each learned specification under the two preregistered exit policies and
   the same funding/cost constraints. Report net NAV versus SPY and attribution,
   source coverage, turnover/capacity, doubled costs and drawdowns. Rank candidates
   by the frozen economic criterion, not highest AUC or average winning-trade return.
   Exit: one selected candidate or `no_candidate`, all twelve or fewer comparisons
   accounted for, independent review of row-to-NAV reconciliation, and no manual
   policy adjustment. This is offline evaluation, not trading_flow execution code.

6. **Start frozen prospective validation (`pending`).**
   Freeze estimator, sources, feature order, selection/exit policy, costs and thresholds
   before producing research predictions for newly observed sessions. The exposed old
   test year is usable only as disclosed historical evaluation; it cannot substitute
   for this forward record. Start the forward prediction/outcome record immediately
   after candidate selection, not after a promotion that depends on that record.
   Preserve first-observed news, filing and bar times. Include the separately fitted
   unseen-security stress and inherited robustness diagnostics without treating them
   as additional independent calendar histories. Freeze the final assessment date,
   sample requirements and stopping rule before observing results; do not repeatedly
   test until a favorable confidence interval appears. Collection and maturation may
   progress while evidence is insufficient, without issuing actionable predictions.
   Owners: `swing/evaluation`, `governance/outcomes`, model/selection contracts and
   existing research/collection adapters. Exit: immutable forward predictions and
   matured outcomes, complete clock/source/policy identity, and a scheduled fixed
   evaluation boundary. Insufficient future data does not block earlier historical
   feature engineering or training.

7. **Evaluate once and publish verified swing serving (`pending`).**
   At the preregistered boundary, evaluate the protected forward predictions and
   compare the same economic policy. Publish a model card with net SPY difference,
   uncertainty, capacity and limitations, or an honest rejected/inconclusive result.
   Historical selection and this forward test must use the identical feature builder.
   Owners: `swing/evaluation`, `governance/outcomes`, `governance/promotion`, `serving`.
   Exit: sufficient preregistered evidence, reproducible realized/predicted differences
   and accepted promotion, with no retuning against final outcomes. No promotion from
   retrospective publication-time news alone. API returns named ten-session expected excess,
   calibrated uncertainty only where validated, benchmark, cutoff, catalyst evidence,
   source coverage, release identity and abstention reasons. No alerts or orders.

After the contract checkpoint, label/accounting work and source/feature work can run
in parallel on disjoint files with reviewer oversight. Training and heavy verification
remain sequential. Each implementation step gets a bounded design/diff review,
focused tests, the required repository verification, a Git checkpoint, and updates
to these same two continuity documents. Close reviewers and owned workers afterward.

Stop conditions are named: `source_blocked`, `contract_conflict`, `no_candidate`,
`statistically_inconclusive`, or `prospective_evidence_pending`. None is silently
converted to a pass. No new infrastructure, model architecture, or data purchase is
a substitute for diagnosing which of those states actually applies.

### Research Checkpoint Status

#### Selected Holding-Path Repair Scope

Code checkpoint **verified and pushed in `bc9dbd5`**. Full suite: **2,193 passed,
three skipped, 133 warnings, 15m46s**; Ruff and strict mypy on 320 source files pass.
The corrected source/path code, exact XNYS clocks, missing/zero-volume handling,
Timestamp parity and implementation-bound resume/complete rejection have focused
tests and independent review. A real metadata check rejected the old retained
authority; its bytes were not changed. Final report:
`.test-tmp/holding-path-verified.xml`. Agents and owned Python workers are closed.

The current panel schema is `market_predictor.swing_panel.independent_holding_paths`.
Old partitions cannot be resumed under this schema. The initial full attempt was
stopped for missing resume binding; a later full run caught an outdated label-policy
hash expectation after 1,839 passes. Both were corrected before final verification.

**Source admission remains blocked, so checkpoint two is not fully accepted.** No
repaired authority or model was produced. Forty price-complete paths are not yet
identity-reconciled; 30 need additional halt/merger/distribution evidence. Canonical
API support does not supply missing source history. Current materialization callers
still require verified post-membership source expansion. Preserve the 70-row blocked
receipt, frozen selected IDs and calendar until a new authority can be replayed.

**Official-source collection implementation complete (`e1e4803`, pushed):**
2,226 tests passed, three skipped, 133 warnings, 16m29s; full Ruff and strict mypy
on 321 source files pass. Report: `.test-tmp/official-documents-verified.xml`.
The first full attempt exposed an omitted collection-CLI inventory entry; corrected
before the final full run. The bounded review's stream-error isolation and external
attempt-path findings are fixed and tested, including relative-path offline replay.
Eight official bodies were retained: six from the ten-document holding inventory
and both AIV follow-up filings. Two initial requests failed and two were deferred;
see the feature audit for immutable report identities. No historical bar, selected
decision or model was changed. Acquisition is not financial/source admission.

**Frozen source-collection scope:** retain the official filing, exchange and
regulator documents behind the unresolved holding periods. Reuse retained evidence
before networking. A configuration names exact public URLs and document purposes;
collection stores bounded response bytes, retrieval metadata and content hashes.
Successful documents resume without another download. A failure on one source
must not erase or block unrelated documents; tampered retained evidence fails closed.
Request identity and immutable per-attempt receipts must be verifiable offline.
No collection receipt authorizes security mapping, distributions, settlement,
training or serving. Historical publication and today's retrieval remain distinct.
Exit checks: offline replay, request/body tamper, independent failure/resume,
redirect/error-body rejection, bounded network reads, dependency boundaries,
independent review, Ruff, strict typing and the complete test suite. Corporate-action
interpretation and the separately reviewed post-membership identity relation follow
only from retained evidence; the frozen stock selection and old panel stay unchanged.

**Next bounded identity work:** publish `retrospective_index_transfer_identity`
evidence for the 38 explicit transfer paths across eleven tickers. This is a
retrospective source-attribution interpretation, not an investment-rule change or
an assertion of lifetime continuity. Bind canonical membership/issuer/stock class,
the retained positive S&P transfer announcement, exact permitted holding sessions,
and an explicitly date-anchored Alpaca replay. Reuse existing filings and prices;
collect only the nine missing class-document candidates and eleven short provider
windows. A successful `asof` response alone is insufficient because provider lookup
may fall back to symbol-only resolution. Review concrete class/reuse/reorganization
contradictions, without demanding a certificate of no event for every day. Hash and
replay all sources; tamper, conflicting class, wrong interval, changed decisions or
unexplained price differences fail closed. Identity admission never clears the
separate total-return/fill gate. AIV and other action-dependent cases are outside
this 38-path interpretation. Synthetic merger redemption/distribution conventions
remain unapproved pending the asynchronous user decision; keep existing rules.

**Verified source subcheckpoint `0c5f1f7`:** the bounded replay/class-fact code is
complete and pushed; the combined retrospective identity relation is not yet
published. Source preparation projects only selected initial-fit identities and
required raw observations. The shared Alpaca decoder rejects duplicate JSON keys
and normalized-symbol collisions. Per-ticker attempts preserve exact bytes and
failures, resume verified successes and replay offline under the shared job lease.
SEC class extraction keeps issuer/context associations explicit without inventing
validity intervals. Full verification: 2,369 passed, three skipped, 133 warnings
in 16m19s; 167 focused tests, full Ruff and strict mypy on 324 source files pass.
All reviewers and owned workers are closed; measured test-process peak 0.324 GiB.

Real acquisition retained all eleven windows / 140 required ticker-sessions, with
no missing or zero-volume observation. Eight tickers match exactly (26 decisions,
101 ticker-sessions). ADS, GPS and PRGO have matching volume but 60/40/56 different
OHLC values (twelve decisions, 39 ticker-sessions). Retained sources, selected IDs,
labels and model files were not replaced. Report SHA-256:
`b8daebd9322da94bd10617d54a1a9cc4c9eb07baa4459f237d76300bc3797198`.
Nine class filings plus the LB rights announcement were acquired separately and
all ten replay offline; ADS/GPS filings were reused. All eleven classes extracted.
The handoff records their immutable archive/report paths and source diagnosis.

**Immediate bounded continuation:** independently compare those three same windows
using an explicit July 25, 2026 symbol-lookup control date, with all other query
parameters fixed. The old request did not record `asof`; the control is a hypothesis
test, not a reconstruction of July adjustment vintage. A separate immutable request
must preserve the completed replay. If the control explains ADS/GPS, compare raw
bars under both lookup dates before attributing an adjustment effect. Require exact
query/byte replay, no source mutation and explicit unresolved results; never tune
price tolerances or exclude the twelve selected decisions. Then bind the positive
S&P/class/identity evidence for the eligible paths. Identity admission cannot clear
the independent total-return/tradability/settlement gates or authorize training.

The bounded continuation of accounting repairs decision/outcome separation, not
the selected population. The source audit reproduced 40 recoverable paths across
12 tickers, requiring 91 retained sessions omitted from combined history. Thirty
other selected paths remain unavailable: nine contain zero-volume records and 21
contain absent sessions. These classifications do not establish corporate-action
proceeds, trading status or investor total returns.

- Decisions retain their original membership, sector, score and cutoff. Index
  removal does not terminate an existing holding or erase its expected label.
- Fixed and managed labels consume separate, explicitly security-identified daily
  outcome history. Align every future session to the benchmark calendar; never
  shift over a missing session or fabricate an executable zero-volume bar.
- Reuse canonical return/barrier evaluators. Preserve existing feature values and
  immutable source/panel authorities. Repaired outputs require new bound artifacts;
  no in-place patch of old labels and no compatibility implementation.
- Verify membership-removal continuation, exact missing-session behavior, duplicate
  identity rejection, zero-volume handling, future-outcome poison isolation, both
  fixed/managed paths, source reconciliation and unchanged ordinary complete paths.
- Missing source identity or corporate-action economics remains source-blocked.
  Independent distribution reconciliation remains required for admission; neither
  recovering 40 price paths nor passing fixtures authorizes training or promotion.


**Accounting implementation (`6758671`, pushed):** cash-conserving overlapping lots,
single prepaid costs, daily marked P&L, fixed idle/tail calendar, required SPY/QQQ/
sector curves, paired base/stress SPY comparisons and descriptive beta/exposure.
Exact intraday exposure-matched SPY and cash-drag decomposition remain unavailable;
closing cash weights are not presented as measured return drag. Approximate
managed-exit-close comparisons cannot admit or order candidates. Price-basis
eligibility remains false without independent total-return reconciliation.

**Retained-data result:** `data/reports/swing_accounting_control/_manifest.json`
is an immutable `blocked` receipt, not a backtest. It binds 30,525 selected IDs,
1,221 initial-fit decision sessions and 1,230 valuation sessions ending 2024-05-28.
Seventy rows have all eight fixed-return fields nonfinite (560 values). Eighteen
tickers are affected; 67 rows are at recorded membership boundaries and 17 explicitly
cross missing sessions, with overlap. Provider-level causes are not proven. The
receipt reads no validation/test outcomes and produces no funded result. Its file
SHA-256 is `f4411da442dcce28c904050045371c3e1ee20e4e1f6e9f041686e7c4f385ee9e`;
post-ownership replay produced identical bytes. Do not drop selected rows after
observing outcome availability or overwrite this evidence.

**Verification:** 340 focused tests and one skip; final complete suite **2,159
passed, three skipped, 133 warnings, 20m32s**; full Ruff and strict mypy on 319
source files; independent review; source-tamper, gap/collision, cash, cost and
dependency tests. The initial full run exposed a newly introduced dependency
violation at 1,149 passed; canonical ownership was corrected, not waived, before
the passing full rerun. Sampled test RSS stayed at or below 0.256 GiB, not a complete
peak measurement. Review agents and Python workers are closed. No real model was
trained, no provider collection started, and checkpoint three remains unstarted.

The following records the completed objective checkpoint:

Implementation `3b2bff5` completes the objective/evidence checkpoint and is pushed.
`configs/swing_research.toml` freezes six return specifications, two policies,
20/40-bps costs, funded one-tenth-NAV cohorts, delayed exit proceeds, one-sided
Bonferroni bounds over twelve comparisons, 20/40-session blocks, 20,000 resamples,
and one prospective assessment after 252 decision sessions plus ten maturation
sessions. Its SHA-256 is
`e37a72796bac9c08ad92a467d079d2a70b4d938f34e286f8dd1e0129a6e9007e`.

`audit-swing-research-evidence` verifies fifteen pinned metadata/config records,
120 ordered technical inputs and sixty recorded historical trial entries across
five manifests. Counts are not unique/lifetime experiments or raw-source replay.
The known exposed evaluation is hash-checked, never parsed for outcomes. The
retained trainer rejects its exposed final interval before panel loading or fitting.
Historical strategy/data hashes are unchanged; no source or model was rebuilt.

Two independent reviewers found no remaining supported checkpoint finding. The
final full suite passed 2,070 tests with three skips and 133 warnings in 27m08s.
Focused verification passed 39 tests; repository-wide `ruff check src tests`,
strict mypy over 315 source files, real metadata audit and diff checks passed.
Sampled test working sets were 0.21-0.25 GiB, not a measured training-memory claim.
No Python worker remained after verification.

The full static gate required mechanical import/string cleanup and deletion of
one proven unreferenced duplicate loader, `intraday/datasets/dataset_io.py`.
Its canonical implementation remains in `intraday/training/training.py`. This is
not an intraday redesign. Current A4.3 transformation identity remains unchanged;
import-only KS4 code-byte identities change, while rejected historical evidence
stays immutable. No accepted/current authority was invalidated or regenerated.
The historical sections below are retained evidence, not another active queue.

## Frozen Data Policy

1. Existing estimator artifacts retain their frozen source sets. A new model may use
   Alpaca direct ticker news, issuer-specific SEC events, or Finviz Elite ticker news
   only after that exact source set is causally backfilled across the model's complete
   decision horizon and independently ablated.
2. The SEC authority distinguishes known zero filings from unknown coverage. Finviz
   screening, global news, and sector news remain overlay/audit inputs unless a
   separately preregistered model contract proves causal estimator value.
3. Reddit and Seeking Alpha are removed and prohibited.
4. Swing decisions begin on `2019-07-09`. Earlier bars are warm-up only.
5. Features and source coverage must be available by the decision time. Unknown
   coverage is null, not zero.
6. Identity, membership, ticker changes, sectors, and benchmarks are point-in-time.
7. Alpaca market bars use SIP and `adjustment=all`; missing bars are not imputed.
8. Sparse gaps invalidate affected windows. Whole-security exclusion is capped at 5%
   of the filtered universe; benchmark failures are not waived.
9. Training and evaluation are chronological, purged, embargoed, and cost-aware.
10. One heavy process runs at a time. Swing candidate training has a 5 GiB
    hard process limit; intraday and serving workloads retain 4 GiB limits.

## Verified Artifact State

| Artifact | State | Evidence |
| --- | --- | --- |
| Swing V12 technical panel | published and replayed | 853,417 `technical_market` rows; 604 securities; 1,759 sessions |
| A3.4 broker-action comparison | corrected, published, and independently replayed | 27,087 prediction rows from 11,720 unique latest broker announcements per comparison dataset; research-only |
| Prior swing candidates | rejected evidence only | no promotion; some specialist runs left locked outcomes unopened, but August 9 technical evidence already exposed July 2025-June 2026 |
| A2 swing baseline trainer | implementation complete; no new candidate artifact | four nested technical ablations plus bounded full-feature ranker/regressor; a later governed run must publish separate statistical evidence |
| Intraday V2 | published and rejected | economically failed after costs; not serveable |
| Intraday V3 z-score lineage | invalid; prohibited | five declared cross-sectional inputs lacked a valid contemporaneous decision-cohort transformation |
| Promoted serving bundle | absent | API must fail closed |

## Model Semantics

### Swing

The active hypothesis is ten-session sector-residual momentum after a controlled
pullback and trend reclaim. Required technical evidence includes 20/60-session
relative strength, SMA50/SMA200 state and slope, pullback/reclaim state, volatility,
liquidity, capacity, SPY/QQQ context, and the point-in-time sector benchmark. The A2
baseline estimator is strictly `technical_market`; catalyst remains a confirmation
overlay. The separate A3 event-driven family may evaluate direct Alpaca ticker events.
SEC is planned as a separate issuer-specific event profile after causal
collection and attachment, preserving known zero versus unknown coverage. Finviz and
global context remain separate overlays.

Within-sector ranking has a preferred target of 50 peers and a hard floor of 30.
Every row persists the sector peer count, sector rank eligibility, whether the target
was met, and ranking reliability weight. Groups of 30-49 peers remain eligible with
weight `decision_time_sector_peer_count / 50`; groups below 30 are ineligible. Portfolio construction
targets a 20% maximum sector weight, adapts to 25% with four represented sectors and
33.3% with three, and skips sessions with fewer than three. Economic gates are
unchanged.

Entry is the next exact exchange-session open. Target, stop, timeout, costs, and all
benchmark returns use the same executable interval.

### Intraday

The active hypothesis is a thirty-minute VWAP exhaustion reversal. Evidence is built
from completed causal intraday bars with next-minute execution, exact one-minute path
labels, stock/market/sector context, and explicit abstention. Catalyst is a
confirmation or ranking overlay unless a preregistered causal ablation proves
estimator value.

Intraday V2 remains a valid rejected artifact. The later V3 z-score lineage is
invalidated and cannot be evaluated, trained, or served. A replacement intraday
cross-sectional feature contract must define one causal decision cohort, share the
same batch/live transformation, and be fully backfilled before candidate training.

## Historical Four-Model Improvement Program

The prior four governed model families are swing baseline, swing event-driven, intraday
baseline, and intraday event-driven. `ROC-AUC >= 0.60` is frozen as a validation and
later locked-test gate for their comparable binary outcome view; it is not permission
to optimize repeatedly on either split. Ranking quality, calibration, benchmark-relative net economics,
drawdown, turnover, capacity, and coverage remain co-equal promotion gates.

| Code | Descriptive checkpoint | State |
| --- | --- | --- |
| A0 | Restore research integrity | Completed |
| A1 | Verify labels and leakage controls | Completed |
| A2 | Build the technical swing baseline | Completed |
| A3 | Build catalyst-driven swing specialists | Completed; combined and directional specialists produced no candidate |
| A4 | Build the technical intraday baseline | Completed; both A4.4 hypotheses rejected |
| A5 | Build catalyst-driven intraday specialists | A5.1e completed with no candidate; prospective SIP horizon is 1/20 sessions |
| A6 | Run locked evaluation and promote qualified models | Not started |

### A0 - Restore Research Integrity (`completed`)

Problem: the current working tree contains unfinished experimental edits that bypass
economic validation, partition verification, source missingness, and governed sector
limits. Those edits invalidate any resulting metric.

In scope: restore fail-closed validation and immutable replay, preserve the 4 GiB
intraday/5 GiB swing limits, remove false zero catalyst inputs, verify the current
cross-sectional feature implementation, and checkpoint only supported code.

Out of scope: new provider collection, feature backfill, model training, locked-test
access, promotion, serving success, and trading alerts or execution.

Exit gates:

- no validation, partition, lineage, or economic-gate bypass remains;
- missing catalyst authority cannot become a numeric zero feature vector;
- training does not mutate loaded immutable dataset frames;
- focused poison tests, full tests, Ruff, strict mypy, compileall, and diff checks pass;
- active plan, feature audit, and handoff describe only verified behavior;
- implementation and documentation closure commits are pushed separately.

Rollback/failure behavior: training and serving remain blocked; no existing rejected
artifact is promoted or used as a fallback.

Completed evidence: implementation commit `e168482` restores monthly partition replay,
requires every validation scope to pass economic gates, preserves immutable input
frames during bounded sequential training, and removes five cross-sectional z-score
columns that had no valid decision-cohort implementation. Focused verification passed
145 tests with one skipped. The full suite passed 1,102 tests with two skipped; three
repository-lock permission failures passed independently with normal repository access.
Ruff, strict mypy, compileall, and staged diff checks passed.

### A1 - Verify Labels and Leakage Controls (`completed`)

- Freeze one comparable binary diagnostic for each model plus the economic training
  target: managed and exact-ten-session benchmark-relative swing return and
  30-minute managed intraday return after costs.
- Reproduce labels from immutable bars using the shared evaluator and identical stock,
  SPY, QQQ, and sector executable intervals.
- Add label shuffle, feature-time shift, future-poison, duplicate-event, overlapping
  label, and survivorship controls. Any abnormal control score blocks training.

Completed evidence: implementation commit `7b61873` removes training-time swing label
rewrites, hard-coded SEC attachment, missing-as-zero filing counts, and the event-only
sector-selection bypass. Technical and Alpaca ablations must now contain identical
published decisions and labels. Intraday label schema V2 adds QQQ on the exact stock
entry-to-managed-exit interval; missing QQQ evidence abstains. Both trainers publish
named estimator-target and after-cost stock/SPY/QQQ/sector binary diagnostics plus a
deterministic shuffled-label AUC control. Future-only label, return, and feature-time
poison tests cannot alter validation selection. The canonical suite passed 1,110 tests
with two skipped; tracked Ruff, changed-module strict mypy, compileall, and diff checks
passed.

### A2 - Build the Technical Swing Baseline (`completed`)

- Evaluate compact, evidence-backed groups sequentially: benchmark/sector residual
  momentum, volatility, liquidity, turnover, quality, profitability, investment,
  valuation, and estimate-revision data where a point-in-time authority exists.
- Use a regularized linear baseline followed by bounded tree ranker/regressor models.
- Backfill every accepted feature for every eligible decision from `2019-07-09` through
  the frozen end date before any model consumes it. Partial-period feature additions
  require an explicit specialist cohort or are rejected.

Completed evidence: implementation commit `cb2aba5` freezes four nested technical
groups: momentum/volatility, trend confirmation, pullback timing, and volume/liquidity.
It evaluates one regularized logistic candidate per group, then one full-feature
XGBoost ranker and regressor, all sequentially within the six-candidate budget. The
bundle records each estimator's exact ordered feature subset and a deterministic,
lineage-bound candidate ID. A promoted baseline selects `technical_market`; an event
specialist requires its own A3 feature contract and promotion and cannot reuse a broad
catalyst frame. Current Finviz snapshots cannot supply historical
point-in-time quality, profitability, investment, valuation, or estimate-revision
features, so those groups remain blocked rather than backfilled from present values.
No new real model was trained or promoted in A2. The canonical suite passed 1,110
tests with two skipped; tracked Ruff, changed-module strict mypy, compileall, and
governance hash replay passed.

### A3 - Build Catalyst-Driven Swing Specialists (`completed; rejected`)

1. **A3.1 - Verify the issuer-targeted event taxonomy (`completed`).** Build separate
   earnings/guidance, SEC material-event, analyst-revision, offering, M&A, regulatory,
   and product-event cohorts only when exact availability and issuer relevance verify.
2. **A3.2 - Backfill and replay historical event authorities (`completed`).** Publish immutable
   direct-issuer event, source-coverage, assignment, and cohort authorities for the
   complete development horizon. A source cannot imply coverage for a family it does
   not provide.
3. **A3.3 - Audit event precision and coverage (`completed`).** Use deterministic
   uniform samples of independent event clusters and preregistered one-sided lower
   confidence-bound gates. Report event, security, calendar, sector, source,
   abstention, unknown-coverage, issuer-error, and reviewer-agreement counts.
4. **A3.4 - Build identical-decision comparison datasets (`corrected and completed`).**
   Compare technical-only, broker-action-only, and combined inputs on the same
   predictions and outcomes. Exact ticker/time alignment replaces invalid direct hash
   matching; every excluded row carries a concrete reason.
5. **A3.5 - Define, train, and evaluate swing broker-action specialists (`completed`).**
   One rating-change specialist combines upgrades and downgrades with explicit action
   direction; a separate specialist handles new/resumed coverage. Price-target and
   generic actions remain report-only because only 55 independently aligned latest
   announcements exist. Each specialist compares technical-only, broker-action-only,
   and combined features with logistic and histogram-gradient-boosting estimators.

Completed A3.1/A3.2 evidence: commits `6ae703c`, `58ccc3d`, and `527e20f` publish the
V2 issuer-targeted classifier and immutable authority replay. Title-derived events
require a causal issuer anchor; bare ambiguous ticker words, preview/conditional deal
language, unsupported source/family pairs, and unknown coverage abstain. Two strict
historical authorities now cover the full development horizon without new network
collection:

- `2019-07-09` through `2021-07-08`: 9,018 classified events, 30,875 assignments,
  28,462 coverage rows, and 1.998 GiB observed peak memory;
- `2021-07-09` through `2026-07-08`: 26,370 classified and research-eligible events
  across 525 securities, 90,136 assignments, 18,333 coverage rows, and 2.157 GiB
  manifest-recorded peak memory.

Earnings, guidance, analyst revision, offering, merger/acquisition, regulatory
decision, and product event are admitted as Alpaca source families. SEC material
events remain `blocked_missing_source`; Alpaca coverage cannot imply SEC coverage.
Both authorities are research-only retrospective evidence and cannot authorize
production.

Completed A3.3 evidence: implementation commits `6ef0579` and `9c8aa5b` publish a
disk-backed, memory-guarded audit with uniform cryptographic cluster sampling, two
independent blind reviewers, separate adjudication, per-field agreement/kappa,
one-sided Wilson bounds, issuer-error vetoes, immutable ledger copies, strict replay,
and fail-fast ledger preflight. The older sample reviewed 1,788 inferential clusters
plus eight paired issuer diagnostics; the newer sample reviewed 1,830 inferential
clusters plus 29 diagnostics. Reviews were performed by independent Codex agents, not
human reviewers, and remain research evidence.

Both eras currently admit only broker rating actions (stored under the internal event
family code `analyst_revision`). Earnings, guidance, offering,
merger/acquisition, regulatory decision, and product event fail at least one frozen
precision, wrong-issuer, reviewer-agreement, or rule-variant gate. SEC material event
has no source-authorized population. Blocked families cannot enter A3.4 or training.

The catalyst-independent V12 base authority contains 853,417 `technical_market`
rows, 604 securities, and 1,759 sessions. The first A3.4 artifact was invalid: it joined
old event decision hashes directly to the rebuilt technical panel, so only 113 rows
from 50 announcements survived. The source data actually contains 17,401 broker
announcements and 37,372 causally covered prediction timestamps. The corrected A3.4
artifact aligns exact ticker and exact prediction timestamp, rejects conflicting CIKs,
and records every inclusion and exclusion. It publishes 27,087 prediction rows from
11,720 unique latest broker announcements in each of three exact datasets:
technical-only, broker-action-only, and technical-plus-broker-action. The internal
profile names retain `analyst_revision` for source lineage. Unknown three-day Alpaca
coverage abstains; blocked event families are absent, not zero. The artifact is
research-only and cannot serve or authorize production.

A3.5 development artifact
`data/models/swing_broker_action_specialists_dev_20260812_v4` records 12 sequential
experiments and a 0.414 GiB peak working set. Capacity passed: rating changes contain
5,841 development and 1,138 validation announcements; coverage initiations contain
2,344 and 502. Model/threshold selection uses a separate 2023-06-13 through
2024-05-28 inner window after a ten-session embargo. No experiment passed that inner
gate, so outer 2024-2025 validation and the locked test both remained unopened. Best
worst-scope inner AUC was 0.546 for rating changes and 0.506 for coverage. Broker-only
inputs were near or below chance; every candidate also failed the canonical portfolio
economic gate. No model artifact was emitted or promoted. The prior v2 report is
superseded because it selected and evaluated on one validation window and used a
simplified economic calculation.

6. **A3.6 - Evaluate upgrades and downgrades as separate swing specialists
   (`completed; no candidate`).** Reuse the corrected A3.4 identical-decision authority and existing
   governed analyst subtype classifier. Split the prior combined rating-change
   specialist into upgrade-only and downgrade-only cohorts, then compare technical-
   only, broker-action-only, and combined profiles on identical rows, folds, labels,
   costs, and security scopes. Capacity must be audited before training; no threshold,
   estimator, gate, or locked-test boundary may be changed based on A3.5 or A5.1e
   results. Coverage initiation and price-target/generic actions are out of scope.

   Implementation result:

   - Commit `82b4959` generalizes the single governed swing-specialist path to accept
     either the historical rating/coverage pair or the frozen upgrade/downgrade pair.
     The new policy changes only cohort membership; profiles, estimators, labels,
     chronological split, embargo, unseen-security assignment, costs, and gates are
     unchanged. Mixed specialist sets fail closed.
   - Upgrade capacity passes with 3,008 development, 561 chronological-validation,
     and 102 unseen-security announcements. Downgrade capacity passes with 2,833,
     577, and 111 respectively. Both include 359 development securities and all eight
     represented sectors.
   - Twelve experiments compare technical-only, broker-action-only, and combined
     profiles using logistic and histogram-gradient-boosting estimators. Upgrade best
     worst-scope inner ROC-AUC is 0.524 for combined logistic (0.533 chronological /
     0.524 unseen). Downgrade best is 0.552 for combined gradient boosting
     (0.552/0.560), only 0.002 above its technical-only control in the weaker scope.
   - No threshold passes canonical economics in both scopes and no experiment reaches
     the 0.60 AUC gate. The artifact is strict `no_development_candidate`; outer
     validation and the locked test remain unopened, no model file exists, and
     promotion remains prohibited.
   - Output
     `data/models/swing_directional_broker_action_specialists_dev_20260820_v1`
     strictly replays. Peak working set was 0.376 GiB under the 5 GiB limit. Final
     verification passed 1,463 tests with two skipped, tracked Ruff, strict mypy over
     231 source files, and tracked compilation.

### A4 - Build the Technical Intraday Baseline

1. **A4.1 - Build the SIP trade/quote source authority (`collector_complete`, full
   authority `environment_blocked`).** Add bounded,
   paginated Alpaca SIP trade and NBBO-quote clients plus resumable raw collection.
   Preserve provider timestamps, exchange/tape/condition identity, request bounds,
   page tokens, response rate-limit headers, and per-unit failures. Raw transport
   completion is not model readiness.
2. **A4.2 - Publish the one-minute microstructure authority.** Aggregate only completed
   regular-session minutes using the shared batch/live transformation. Publish
   time-weighted relative spread, time-weighted quote-size imbalance, quote-update
   count, trade count, share volume, dollar volume, and mean trade size with explicit
   availability and source coverage. Missing quotes or trades remain unavailable.
3. **A4.3 - Publish the bar-only causal technical dataset (`complete`).** This is a distinct
   governed model profile, not a degraded microstructure model. Its source set is
   limited to verified SIP/all one-minute bars, causal five-minute bars,
   point-in-time membership, SPY, QQQ, and sector ETFs. It may use volume clock,
   VWAP displacement, opening range, volatility, and exact market/sector residuals,
   but it must not contain trade-count, quote, spread, or imbalance fields. Prove
   batch/live parity, future-poison rejection, and ordered-feature hash identity.
   Cross-security ranks use explicit contemporaneous clock-time cohorts, never each
   stock's asynchronous volume-bar completion timestamp. ATR used by features,
   targets, and stops comes from the causal five-minute authority required by the
   strategy contract, not from volume bars.

   Frozen A4.3 contract:

   - Project the already verified canonical SIP/all regular-session five-minute store
     into one immutable selected-stock-session authority. This is a local projection,
     not a provider download. Retain incomplete sessions as explicit coverage metadata.
   - Continue to derive event-based volume bars from the verified selected-session
     one-minute stock collection, but decisions exist only on a pre-scheduled
     five-minute cohort clock after activation. At each fixed cohort, use the latest
     completed volume bar whose evidence was already available by the cutoff. Late
     evidence cannot move the cohort; it remains unavailable until a later scheduled
     decision. Cohorts follow exchange-session open plus the frozen 60-second
     finalization delay.
   - Set `source_feature_available_at_utc` to the latest source availability and
     `feature_available_at_utc` to the cohort cutoff. All one-minute stock, SPY, QQQ,
     and sector context is the latest exact completed minute available by that cutoff.
   - Compute `atr_14_5m` only from completed canonical five-minute bars in the same
     session. The model ATR fraction and the 2.0/1.5 ATR target/stop labels must use
     this value. Volume-bar ATR is prohibited.
   - The bar-only ordered estimator contract contains technical momentum/trend,
     volume/liquidity, session VWAP, 15-minute opening-range distance, exact
     stock/SPY/QQQ/sector returns and residuals, and timing fields. It contains no
     trade count, quote, spread, imbalance, catalyst, SEC, Finviz, or global-event
     input.
   - A later session gap cannot remove an earlier decision. Feature eligibility uses
     only evidence through the cohort cutoff; label eligibility uses only the exact
     next-minute entry and 30-minute managed outcome interval. Missing evidence
     abstains only the affected row.

   Exit gates: immutable selected five-minute projection and dataset replay; exact
   clock-cohort identity; no duplicate ticker/cohort; five-minute ATR lineage proof;
   identical batch/live ordered features; future-poison, missing-versus-zero,
   incomplete-later-session, benchmark-interval, artifact-tamper, and path-traversal
   tests; complete dataset publication under 4 GiB; no locked-test access and no model
   training in this checkpoint.

   Completion evidence: implementation commits `8a76ec1`, `e76bf8d`, and `1829fce`;
   immutable five-minute
   projection `data/canonical/edge_rebuild_selected_session_5m_bar_only_causal_20260814_v2`
   with 43,226 selected stock-sessions, 3,364,335 rows, 43,132 complete pairs, 94
   incomplete pairs retained as coverage, and no provider download; immutable dataset
   `data/features/intraday_causal_volume_bar_dataset_20260831_v2` with 794 sessions,
   501 tickers, 3,095,688 rows, and 1,365,015 eligible rows. Dataset manifest SHA-256 is
   `1f09a55489a2889b40899eff44ecb4205dba2162c3198f8def559b6e50951a58`, request SHA-256 is
   `5e8c508a4237320d6ea56205502f670244a49f713b22dbff10a336d4d2dc303a`, and
   transformation SHA-256 is
   `6fdfd0c8f07e4f7445b66d038cbd936e4459db68e087a5ddbcb30eac4795cb51`.
   Audit v2 at
   `data/reports/intraday_causal_volume_bar_dataset_20260831_v2_audit_v2.json`
   (SHA-256 `f1b21af3704317d070f479ac6a562fdb126c21d2b7da021ddf5c7b6108be97e8`)
   binds the dataset and exact projection path/authority/manifest/inventory. It reports
   zero duplicate decisions, normalized or raw source cutoff violations, incomplete
   five-minute-prefix eligibility, label-availability violations, eligible ATR defects,
   feature-hash defects, schema defects, and prohibited features. The resumed build
   predates mandatory per-invocation execution receipts, so
   `data/reports/intraday_causal_volume_bar_dataset_20260831_v2_execution_assessment.json`
   truthfully records `complete_run_memory_proven=false`; earlier invocation memory was
   not reconstructed. This operational limitation does not alter the independently
   replayed row or lineage evidence, but it cannot be cited as complete-run memory proof.
4. **A4.4 - Train separate bar-only continuation and reversion baselines (`complete; no candidate`).** Use purged,
   embargoed chronological selection and the canonical intraday portfolio evaluator.
   Do not open the future holdout unless a preregistered development candidate passes
   calibration, predictive, coverage, cost, drawdown, and benchmark-relative gates.
   Selecting the least-bad failed candidate does not authorize future-holdout access.

   Frozen A4.4 contract:

   - Train continuation and long reversion as independent hypotheses and immutable
     outputs. Continuation requires positive one-volume-bar return, positive
     twenty-minute stock return, and price at or above session VWAP. Reversion requires
     negative twenty-minute stock return, price at least 0.5 five-minute ATR below
     session VWAP, and volume-bar RSI at or below 45. These causal cohort rules are
     configuration identity, not fitted parameters.
   - Each hypothesis fits an expected-net-return opportunity estimator and a separate
     stop-hit downside estimator. Downside calibration uses only an earlier fit slice
     and a later purged calibration slice; validation outcomes cannot calibrate scores.
   - Fit excludes a stable 20% security holdout. Every candidate is evaluated on both
     chronological seen-security and contemporaneous unseen-security validation. The
     worse scope controls selection.
   - Candidate and policy grids are preregistered and bounded. Estimators run
     sequentially under the 4 GiB process limit; no GPU or parallel model fit is used.
   - Gates cover positive-net-return ROC-AUC, stop-risk ROC-AUC and calibration,
     prediction coverage, trade/session capacity, after-cost return, SPY/QQQ/sector
     excess return, rank gain, stress costs, drawdown, turnover, and fold stability.
     A failed run publishes reproducible rejection evidence but no model artifact.
   - The post-2026-07-08 future authority remains unopened unless one frozen
     development policy passes every gate. A4.4 itself cannot promote or serve.

   Result: both real-data runs completed sequentially and published strict immutable
   `no_candidate` evidence.
   - [x] Create `src/market_predictor/edge_rebuild/utils/` (`io.py`, `hashing.py`, `memory.py`, `validation.py`).
   - [x] Consolidate duplicated logic from `intraday_training.py` and `swing_training.py`.
   - [x] Restructure `intraday_training.py` into `intraday_dataset_io.py` and `intraday_types.py`.
   - [x] Refactor `swing_types.py` and `data_io.py` to import from `utils/` instead of local clones.
   - [x] Fix strict mypy and ruff linting errors.

   Continuation's best audited seen/unseen positive-return
   ROC-AUC was 0.510/0.516; long reversion's was 0.513/0.508. Stop-risk ROC-AUC was
   about 0.60 but calibration failed. Controlling scopes also failed after-cost return,
   benchmark excess, confidence-bound, trade-count, and fold-stability gates. Peak RSS
   was below 2.1 GiB. No candidate artifact was written and the future holdout remained
   unopened.

   Exit gates: exact A4.3 authority/hash replay; separate hypothesis identities;
   feature-order and source-contract binding; purged fit/calibration/validation and
   unseen-security overlap audits; deterministic rerun; future-poison and
   artifact-tamper tests; sequential real-data runs below 4 GiB; consolidated ML/code
   review; full tests, Ruff, strict mypy, compileall, implementation commit, and
   documentation closure commit.
5. **A4.5 - Add the microstructure-enhanced profile only after A4.1 and A4.2 are
   complete.** Join the independently verified one-minute microstructure authority at
   the exact completed-minute cutoff, add trade intensity, relative spread, quote-size
   imbalance, quote updates, trade count, and mean trade size, and repeat ablation and
   promotion gates under a new dataset and model identity. A partial trade/quote
   collection cannot be used by either the bar-only or enhanced profile.
6. **A4.6 - Compare profiles only on an immutable matched-ablation cohort.** The
   bar-only and microstructure-enhanced comparison must use identical decision IDs,
   labels, fold assignments, execution costs, and benchmark intervals. Report the
   broader bar-only coverage separately. Missing microstructure makes only the
   enhanced row unavailable; it cannot remove or relabel the corresponding bar-only
   decision.

Historical collection covers every selected stock-session in the source coverage
authority. End-of-session bar completeness is retained only as metadata and must not
decide whether an earlier historical decision exists. Feature eligibility is measured
only through each decision cutoff; label eligibility is measured only through its
managed outcome horizon. Later missing bars may make a label unavailable, but cannot
rewrite the live-equivalent decision cohort.

Existing evidence at A4 start: the selected-session SIP/all one-minute bar collection
contains 81,349,171 rows across 559 observed symbols, and the prior V3 dataset contains
4,173,230 rows. Neither artifact contains historical trades or NBBO quotes, so neither
can authorize microstructure features or A4 training. The invalid V3 cross-sectional
z-score lineage remains prohibited.

The A4.1 live capacity probe on 2026-07-08 found 2,425 AAPL trades and 4,172
AAPL quotes in one ordinary minute; SPY exceeded the 10,000-row quote page limit in
one minute. The local C: drive had approximately 52 GiB free. Therefore a replayable
43,226-stock-session raw tick backfill is storage-blocked locally. No quote/trade
feature may enter A4.5 until a complete immutable source authority exists. A4.3 and
A4.4 completed because its separately identified bar-only contract has no
trade/quote inputs and cannot silently acquire them.

A4.1 implementation commit `b03f4f1` publishes the bounded collector. The corrected
immutable v2 plan contains 43,226 selected stock-sessions and 86,452 jobs: 43,213 have
complete source-bar session status and 13 retain incomplete status as metadata. A real
two-invocation Alpaca SIP probe completed the first trade and quote jobs with zero
failures, 4.41 MiB on disk, and 0.350 GiB peak RSS. The probe remains
`transport_incomplete` and is not an authority. Completing all replayable raw tick jobs
still exceeds current local storage, so A4.2 and A4.5 remain blocked. The next phase is
A5 causal intraday event-cohort preflight; training remains blocked until that separate
authority passes attachment, timing, coverage, and replay checks.

### A5 - Build Catalyst-Driven Intraday Specialists

- Restrict training to verified event cohorts. Use publication regime, time since
  event, premarket gap, abnormal volume, initial 5/15-minute reaction, spread,
  liquidity, and sector concurrence.
- Catalyst remains outside the broad intraday estimator unless causal ablation passes.
  Unknown coverage causes abstention, never neutral sentiment or zero event counts.

1. **A5.1 - Publish the causal intraday event-cohort preflight (`complete; blocked`).** Bind
   the strict A4.3 decision authority to the two strict Alpaca direct-issuer event-family
   authorities. Do not use `intraday_catalysts.py`, ticker filenames, Finviz snapshots,
   prohibited sources, or publication time as a substitute for observed availability.

   Frozen A5.1 contract:

   - Source family is exactly Alpaca; relation channel is exactly `direct_issuer`; the
     only currently precision-approved event family is `analyst_revision`. Each parent
     authority and its exact hash must replay before attachment.
   - Preserve A4.3 `decision_id`, `security_id`, strategy, transformation, feature-order,
     and cost identities. Attach only events for the same `security_id` with
     `feature_available_at_utc <= decision_time_utc` in a half-open 24-hour lookback.
   - Retain known-zero coverage and unknown coverage as distinct states. Unknown,
     ambiguous, proxy-only, or post-decision evidence abstains and is never encoded as
     neutral sentiment or a zero count.
   - Reuse A4.4's stable 20% security holdout and four chronological development folds.
     Report unique event episodes separately from repeated attached decision rows.
   - Publish an immutable `eligible` or `blocked` authority containing decision
     eligibility, event attachments, coverage/split audit, request, manifest, and
     authority. Strict reload must reject changed, missing, extra, nested, overlapping,
     or non-deterministically rebuilt evidence.
   - Preflight eligibility requires observed first-seen/revision availability rather
     than `provider_publication_proxy`, zero issuer/time/hash violations, both seen and
     unseen event coverage, at least 1,000 unique episodes overall, 200 securities, 120
     fit sessions, and 1,000 rows/20 securities in each validation scope. These are
     authorization floors, not performance claims.
   - A blocked output sets `training_eligible=false`, `serving_eligible=false`, and
     `future_holdout_opened=false`, and contains no estimator. A passing preflight only
     authorizes A5.2 matched ablation; it does not authorize serving or locked-test use.

   Exit gates: deterministic strict replay; future-poison, issuer-mismatch,
   proxy-availability, unknown-coverage, duplicate, tamper, extra-file, and path-overlap
   tests; real-data preflight below 4 GiB; one consolidated ML/code review; full tests,
   tracked Ruff, strict mypy, compileall, implementation commit, and documentation
   closure commit.

   Result and evidence:

   - Implementation commit `98f7a48` publishes one canonical A5.1 path and removes
     any need to consume the retired ticker-file catalyst modules.
   - Authority:
     `data/research/edge_rebuild_intraday_event_preflight_20260815_v1`.
     Strict replay binds the exact A4.3 authority, both parent event authorities,
     every consumed artifact and child manifest, and the recursive parent inventory.
   - The parents contain 17,401 research broker-action episodes. Exact point-in-time
     `security_id` matching yields 19 unique attached episodes and 862 repeated
     event-decision pairs. No ticker-only fallback is permitted.
   - All 17,401 episodes use retrospective `provider_publication_proxy` timing and
     zero episodes have observed first-seen/revision-safe production availability.
     Retrospectively completed collection intervals remain unknown at historical
     decisions; they do not become known-zero coverage.
   - The authority is `blocked`, `training_eligible=false`,
     `serving_eligible=false`, and `future_holdout_opened=false`. It contains zero
     estimators and zero production-eligible decisions. A5.2 training and A6 locked
     evaluation are therefore prohibited.
   - Peak working set was 2.299 GiB. Final verification passed 1,348 tests with two
     skipped, tracked Ruff, strict mypy over 226 source files, compileall, strict
     real-authority replay, and consolidated independent review.

2. **Prospective broker-action observation authority (`implementation and weekday
   live validation complete; capacity collection active`).** Establish the
   only permitted path for future Alpaca broker-action evidence. Each polling run must
   archive the exact provider response and observation time, retain every distinct
   provider revision, and bind symbols to the same point-in-time S&P security identity
   namespace consumed by A4.3. Historical publication or provider-update timestamps may
   never be substituted for observation time.

   Frozen scope and invariants:

   - Source is Alpaca direct ticker news only. Reddit, Seeking Alpha, Finviz news,
     retrospective timestamp repair, model training, A5.2, serving, alerts, and order
     behavior are out of scope.
   - Every poll has one UTC observation timestamp, an immutable raw-page inventory,
     explicit successful/empty/failed coverage, and a hash-bound current membership
     parent. The current authority must reproduce A4.3's identity history through its
     cutoff and may then extend that namespace to the poll date.
   - Identity joins require one effective membership interval and one observed Alpaca
     asset identity for the symbol. Missing, ambiguous, changed, or authority-stale
     identity remains excluded with a persisted reason; ticker-only fallback is
     prohibited.
   - A provider item is versioned by provider event identity, provider update time, and
     raw-content hash. `first_seen_at_utc` is the first archived observation of that
     exact version. Repeated polls cannot rewrite it, and later revisions cannot alter
     earlier versions.
   - Canonical production eligibility requires `availability_policy=observed`, complete
     collection coverage, exact identity, and strict replay. Passing this checkpoint
     starts a prospective evidence horizon; it does not satisfy A5.1 capacity floors.
   - Collection is resumable, single-process, bounded below 4 GiB, and fail-closed on
     request, parent, raw-page, revision-chain, coverage, or identity tampering.

   Exit gates: deterministic strict replay; observed-time, revision-preservation,
   identity-change, known-zero-versus-unknown, resume, duplicate, future-poison,
   extra-file, path-traversal, and artifact-tamper tests; one real Alpaca poll when
   credentials and network are available; focused and full verification; consolidated
   review; implementation commit and documentation closure commit. Rollback is removal
   of the new command/module and its new prospective authority only; A5.1 remains
   blocked and no prior immutable authority is changed.

   Implementation result:

   - Commit `5530246` adds one poll collector and one generation publisher. Polls bind
     the strict A4.3 dataset, an extending current membership authority, exact Alpaca
     asset/news URLs and query parameters, response timing, raw bodies, every provider
     revision, immutable failed attempts, and an append-only cutoff claim/commit
     registry. Parent chains replay iteratively and must use the same authority and
     registry.
   - Wrong symbols/windows, redirects, responses before the scheduled cutoff, stale or
     changed identity, partial publication, duplicate cutoffs, lineage changes, and
     registry/raw/artifact tampering fail closed. Compaction is bounded by process
     memory and verified Parquet input size. No model or feature row is emitted.
   - Verification passed 1,373 tests with two skipped, 41 focused authority/source/CLI
     tests, tracked Ruff, strict mypy across 226 source files, compileall, and one
     consolidated two-reviewer correction pass.
   - Commit `27ab9b7` accepts only the exact live or paper Alpaca asset hosts, keeps
     news on the exact data host, and verifies the malformed 2018 Twitter/Monsanto
     source by complete semantic table grammar instead of transient page-shell hashes.
   - Poll `data/raw/prospective_broker_actions/poll_20260816T071230Z`, using stable
     registry `registry_v2`, strictly replays 503 eligible security identities, 76
     observed events, 46 observed-symbol collections, and 457 known-empty collections.
     Peak working memory was 0.328 GiB. The authority remains
     `production_ready=false`, `training_eligible=false`, and
     `serving_eligible=false`; one poll starts evidence collection but cannot train A5.2.
   - The first poll occurred on Sunday and used the latest fully closed New York
     publication date, Saturday `2026-08-15`. The weekday-safe observed membership
     authority is now complete under Step 4 below. Poll
     `data/raw/prospective_broker_actions/poll_20260817T202950Z` strictly replays 11 of
     11 batches, 503 constituents, 460 observations, and 454 exact production-identity
     events at cutoff `2026-08-17T20:30:00Z`; peak working memory was 0.379 GiB.
     This expands the prospective horizon but does not satisfy the frozen A5.1 capacity
     floors or authorize A5.2 training.
   - Final verification passed 74 focused tests and the exact tracked suite with 1,382
     passed and 2 skipped; tracked Ruff, strict mypy across 227 source files, bytecode
     compilation, strict real-authority replay, and consolidated independent review
     also passed.

3. **Current S&P membership extension (`complete`).** Extend the verified
   point-in-time S&P 500 membership authority from `2026-07-08` through
   `2026-08-15` so prospective Alpaca observations can resolve security identity.

   Frozen scope and invariants:

   - Parameterize the official S&P archive collection cutoff as an immutable request
     input; retain the frozen 83-release seed authority and the `2018-04-14` lower
     boundary.
   - Collect and archive exact official S&P release bytes through `2026-08-15`, then
     rebuild event, transition, and membership authorities through that cutoff.
   - The extension must reproduce every membership interval through `2026-07-08` from
     the existing authority. Only official events effective after that date may alter
     later membership. Announced but not-yet-effective changes remain future events.
   - The cutoff anchor must be independently observed and hash-bound. It may not be
     manufactured from the same change events it is intended to verify.
   - Model training, feature changes, serving, alerts, orders, and prospective Alpaca
     collection are out of scope. Missing or contradictory official evidence fails
     closed and leaves the existing stale-membership abstention unchanged.

   Exit gates: cutoff/resume/replay/tamper and historical-prefix tests; complete raw,
   event, transition, and membership authorities through `2026-08-15`; exact semantic
   equality with the existing authority through `2026-07-08`; focused tests, full
   suite, tracked Ruff, strict mypy, compileall, memory/process check, consolidated
   review, implementation commit/push, and documentation closure commit/push.
   Rollback removes only the new parameterization and extension artifacts; the
   `2026-07-08` authority remains immutable.

   Implementation commit `42ebe5f` is pushed. It hash-binds the configurable cutoff,
   requires the inclusive cutoff day to have closed in `America/New_York`, preserves
   the complete base membership contract before the extension boundary, rejects CIK
   conflicts and inconsistent base authorities, and shares namespace verification
   with prospective collection. Verification passed 111 focused tests, tracked Ruff,
   strict mypy across 227 source files, compileall, and the full tracked suite with
   1,374 passed and 2 skipped. Peak test-process memory stayed below 0.2 GiB.

   The early same-day `v1` artifacts remain invalid and must not be used. After the New
   York day closed, the immutable `v2` raw archive collected 109 releases; event
   authority `spglobal_events_20180414_20260815_v3` published 307 events with zero
   unresolved releases; transition authority `sp500_transitions_20180529_20260815_v2`
   published 13 transitions; and membership authority
   `sp500_memberships_20180529_20260815_v2` published 1,170 intervals across 659
   securities with five governed whole-security exclusions. Strict replay verifies the
   complete authority and exact July 8 base prefix.

4. **Weekday-safe observed S&P membership authority (`complete`).** Publish a
   causal identity authority for a prospective poll without describing an unfinished
   New York publication day as a complete historical archive.

   Frozen scope and invariants:

   - The latest fully closed official S&P archive, event authority, and membership
     authority remain immutable parents. Their historical completeness contract is not
     weakened or reused for an intraday cutoff.
   - A new observation archives exact HTTP response bytes and response times for a
     contiguous newest-to-closed-cutoff prefix of the official S&P release index and
     every newly discovered membership release. Redirects, malformed pagination,
     missing release bodies, parser failures, and responses outside the approved source
     hosts fail closed.
   - Membership state at the observation cutoff applies both already-announced
     future-effective changes from the closed event authority and newly observed
     official changes whose effective time is no later than the observation cutoff.
     Provider publication dates never replace first-observed response time for a new
     release.
   - The same run archives exact bytes for an independently observed current S&P 500
     constituent anchor. The reconstructed active ticker set and every inherited CIK
     identity must agree exactly with that anchor. This equality is also the quiet-day
     completeness gate: no new release plus an unequal anchor is an abstention, not an
     inferred no-change day.
   - The resulting membership table must preserve the complete closed-authority prefix
     and the A4.3 security namespace. It records separate closed archive, observation,
     and effective horizons; these timestamps may not be collapsed into one cutoff.
   - Prospective Alpaca polling may consume only a strictly replayed observed authority
     whose observation precedes the poll cutoff and whose effective horizon covers the
     poll. Model training, A5.2, serving, scheduling, alerts, and orders remain out of
     scope.

   Exit gates: offline strict replay from exact retained bytes; closed-prefix and A4.3
   namespace equality; already-announced future-effective, same-day observed change,
   quiet-day, stale anchor, race-window, parser failure, redirect, pagination,
   future-poison, extra-file, path-traversal, and tamper tests; prospective poll
   acceptance/rejection tests; focused and full verification; tracked Ruff, strict
   mypy, compileall, memory below 4 GiB, consolidated independent review,
   implementation commit/push, and documentation closure commit/push. A real weekday
   observation remains environment evidence and cannot be replaced by a weekend test.
   Rollback removes only the new authority and poll adapter; all closed authorities and
   the Sunday poll remain immutable.

   Implementation result:

   - Commit `a5aae9b` adds the collection-only
     `collect-edge-observed-sp500-memberships` command and a strict observed-time
     authority. It retains exact no-redirect S&P search/release bytes, a complete
     second search sweep, the current constituent anchor, SEC ticker/CIK identities,
     every release outcome, observed events, pending effective changes, and canonical
     membership intervals without altering the fully closed parent authorities.
   - Poll eligibility requires the authority observation to precede the poll, remain
     within the configured 60-300 second continuity bound, and precede the next known
     pending membership change. Authority rotation must move forward in observation
     time and retain every previously observed release outcome and event. The same
     chain rules replay during strict load. Closed authorities cannot authorize a
     weekday poll; the existing weekend replay remains valid.
   - Retained-byte publication/load tests verify exact anchor identity, future-change
     timing, quiet-day equality, full multi-page race confirmation, malformed-release
     rejection, URL/redirect policy, inventory/path/symlink rejection, tamper failure,
     per-ticker poll continuity, and strict authority-chain replay. The offline fixture
     used 500 members and 5,000 SEC identities; it is test evidence, not live market
     evidence.
   - Initial verification for commit `a5aae9b` passed 152 focused tests and the complete
     tracked suite with 1,404 passed and 2 skipped. Tracked Ruff, strict mypy across 228
     source files, compileall, and consolidated independent review passed.

   Reopened evidence on `2026-08-17`:

   - The first compliant live observation reached all configured sources but failed
     closed because SEC's 10,396-record `company_tickers.json` response omitted AEP.
     The exact SEC submissions endpoint for inherited CIK `0000004904` independently
     identifies American Electric Power and lists ticker AEP. No authority was
     published.
   - Scope is limited to a causal identity fallback for tickers absent from the SEC
     bulk map. The collector must retain the exact CIK-specific SEC submissions
     response, derive the candidate CIK from the closed membership identity when
     available, verify that the SEC response lists the expected ticker and CIK, and
     replay the fallback inventory exactly. Conflicts, missing ticker claims, extra
     fallback units, redirects, or response tampering fail closed.
   - S&P release parsing, membership transitions, Alpaca polling, features, training,
     serving, scheduling, alerts, and orders are outside this correction. Exit gates
     are focused fallback/replay/tamper tests, the existing focused authority suite,
     tracked Ruff, strict mypy, compileall, a real weekday collect/load replay, and a
     new implementation plus documentation checkpoint pushed before polling Alpaca.
   - The next compliant live attempt passed the AEP fallback and then failed closed on
     XOM: the closed authority inherited CIK `0000034088`, while both the current S&P
     constituent anchor and SEC bulk identity map identify ticker XOM with successor
     registrant CIK `0002115436`. SEC's July 1, 2026 Form 8-K confirms a one-for-one
     holding-company reorganization with the same ticker, so this is an issuer identity
     transition rather than an index addition or deletion.
   - Scope therefore also admits a same-ticker identity transition only when the
     retained current S&P anchor and retained SEC identity evidence agree on a new CIK.
     The old interval closes and the successor interval opens at the observation time,
     never at a retrospective inferred date. The complete closed-authority prefix and
     old security identity remain immutable. Duplicate active identities, ticker-set
     changes, ambiguous CIKs, or replay differences fail closed.
   - Implementation commit `a7fe60e` is pushed. It retains and strictly replays the
     exact SEC CIK-specific fallback, rejects inherited-versus-anchor CIK disagreement,
     records a causally observed same-ticker successor identity, validates every raw
     unit envelope and content-addressed body path, and binds the canonical membership
     manifest to the exact request, base authority, and observation-unit inventory.
   - Real authority
     `data/raw/index_membership/sp500_observed_20260817T203000Z_v3` strictly replays 503
     current constituents, 10,397 SEC identities including the AEP fallback, zero new
     membership releases/events, and the XOM successor CIK from observation time
     `2026-08-17T20:29:05.344531Z`. It remains a collection authority, not training or
     serving authorization.
   - The immediately following Alpaca poll
     `data/raw/prospective_broker_actions/poll_20260817T202950Z` strictly replays 11 of
     11 batches, 460 observations, and 454 exact production-identity events. Final
     verification passed 152 focused tests and the complete tracked suite with 1,417
     passed and 2 skipped; tracked Ruff, strict mypy across 228 source files,
     compileall, two-reviewer remediation, and final artifact replay passed. Peak poll
     working memory was 0.379 GiB. A5.2 remains prohibited until the prospective
     horizon satisfies the frozen capacity floors.

5. **A5.1b - Publish the prospective analyst-event horizon (`completed`).**
   Aggregate one or more strictly replayed prospective broker-action generations into
   one immutable, append-only source-capacity authority. Classify each observed
   revision with the existing frozen issuer-event policy, admit only exact-identity
   Alpaca `analyst_revision` events, preserve every revision, and count one episode per
   provider event and security. Revisions must never inflate episode capacity.

   Frozen scope and invariants:

   - Each parent generation, poll, membership authority, A4.3 namespace, and registry
     identity must strictly replay. Duplicate or overlapping polls, reversed cutoffs,
     broken parent chains, namespace changes, and cross-generation security-identity
     conflicts fail closed.
   - First-seen availability is the earliest retained provider-response observation.
     Publication or provider-update timestamps cannot replace or backdate it.
     Production availability is the later of first-seen and first exact-identity
     eligibility.
   - Issuer-company classification anchors come only from the strictly replayed
     membership authority observed before the corresponding poll. A revision without a
     causal issuer-company or explicit ticker anchor may remain retained but cannot be
     admitted as an analyst episode.
   - Publish classified revisions, admitted episodes, collection coverage, and a
     source-capacity audit with exact content hashes and strict replay. Processing must
     remain below 4 GiB and accept multiple non-overlapping generations so the existing
     60-poll per-generation bound remains intact.
   - The two-poll generation
     `data/research/prospective_broker_actions_generation_20260817_v1` is valid initial
     evidence: 536 revisions, 248 provider events, 530 exact-identity revisions, and
     184 exact-identity securities. These are raw observations, not yet classified
     analyst episodes and not training rows.
   - Training, serving, model fitting, historical timestamp repair, prospective bar
     collection, feature construction, label maturation, alerts, and orders are out of
     scope. The authority always records `training_eligible=false`,
     `serving_eligible=false`, and `future_holdout_opened=false`.

   Exit gates: deterministic strict replay; duplicate/overlap, parent-chain, identity,
   revision, timestamp-poison, issuer-attribution, artifact-tamper, extra-file,
   path-overlap, deterministic-order, and memory tests; one consolidated review; full
   tracked tests, Ruff, strict mypy, compileall, and strict replay of the real horizon.
   Rollback is deletion of only the new command/module/tests and newly published
   horizon; all parent polls and generations remain immutable.

   Implementation result:

   - Commit `fe2b4c9` adds one research-only publisher and strict loader for multiple
     chronological prospective generations. It binds every parent generation, poll,
     membership authority, security namespace, registry, preflight policy, and event
     classifier; preserves revisions; counts provider-event/security episodes; and
     keeps training, serving, and future-holdout access false.
   - Corrected real authority
     `data/research/prospective_analyst_revision_horizon_20260817_v2` strictly replays
     536 revisions from 248 provider events and two polls. Only three events qualify as
     causal exact-identity analyst episodes: AMCR, HBAN, and WDAY. Eligible-security
     count is three and source capacity remains `blocked`.
   - Every coverage row is bound to that poll's exact security identity. All persisted
     timestamps use one deterministic UTC dtype. The consolidated reviewer reproduced
     and closed a Parquet round-trip defect in mixed null/non-null previous-poll times;
     the chained-poll regression and real v2 replay now pass.
   - Peak publication memory was 0.478 GiB. Final verification passed 1,431 tests with
     two skipped, tracked Ruff, strict mypy across 229 source files, compileall, focused
     causal/tamper tests, and independent reviewer verification. No model was trained
     and A5.2 remains prohibited.

6. **A5.1c - Build prospective SIP sessions and mature outcomes
   (`implementation_complete_data_pending; 1/20 source sessions`).**
   After A5.1b closes, freeze a separate append-only authority for future SIP bars,
   A4.3-identical features, and exact 30-minute labels. It must collect the complete
   contemporaneous selection cohort plus SPY, QQQ, and sector ETFs, preserve the A4.3
   namespace, attach events only at or after observed availability, and define folds
   over the causally covered prospective cohort. A5.2 training remains prohibited
   until this authority and a rerun capacity audit pass all frozen floors.

   Ordered implementation checkpoints:

   1. **Exact SIP bar transport (`completed`).** Extend the canonical Alpaca bar-page
      response with exact bounded HTTP bytes, requested/final URL, status, retrieval
      time, safe headers, and redirect evidence. Requests must use SIP, `adjustment=all`,
      ascending order, explicit `asof`, bounded pages, and no redirects. Preserve
      backward compatibility only for test doubles; production collection must reject
      pages without transport evidence.
   2. **Immutable closed-session source authority
      (`implementation_complete_data_pending`).** Publish one
      append-only child authority per fully closed XNYS session. The first phase archives
      exact Alpaca HTTP response bytes and sidecars for SIP five-minute bars covering the
      complete membership cohort observed before that session, plus SIP one-minute bars
      for SPY, QQQ, and the sector ETFs. Strict replay must reconstruct canonical bars
      from those retained bytes, bind the exact membership parent and request identity,
      reject redirects, gaps, duplicate pages, and post-session mutation, and support
      hash-verified crash-safe resume below 4 GiB. The second phase may collect selected
      stock one-minute paths only when the target session has twenty contiguous prior
      five-minute sessions under the same causal namespace. A source-complete session is
      explicitly selection-ineligible while that warm-up is absent; stale July bars may
      not activate an August cohort. This checkpoint builds no features or labels and
      authorizes neither training nor serving.
      Benchmark one-minute grids must be complete. Full-cohort stock gaps are retained
      as explicit coverage evidence and may authorize the session only within the frozen
      whole-security exclusion ceiling of 5%; no bars are imputed, and coverage above
      that ceiling publishes status only without a parent source authority.
   3. **Causal feature authority (`not_started`).** Reproduce the twenty-session
      activation cohort and reuse the exact A4.3 volume-bar and feature transformation.
   4. **Mature outcome authority (`not_started`).** Publish exact stock/SPY/QQQ/sector
      30-minute paths only after availability; keep labels separate from features.
   5. **Matched prospective preflight (`not_started`).** Attach only observed analyst
      episodes available by decision time and evaluate the frozen capacity floors.

   The exact transport checkpoint changes only the Alpaca source contract and focused
   tests. It does not collect data, alter historical authorities, build features or
   labels, train models, or authorize serving. Exit gates are exact-byte and URL/query
   tests, redirect/status/body-integrity failures, existing collector compatibility,
   tracked tests, Ruff, strict mypy, and compileall. Rollback removes only the new bar
   transport fields and byte-fetch path.

   Exact transport result:

   - Commit `c56843e` replaces parsed-only bar fetches with a bounded exact-byte HTTP
     response. `AlpacaBarsPage` now carries raw bytes, safe headers, requested/final
     URL, status, retrieval time, and redirect evidence while retaining parsed bars for
     existing collectors.
   - Production bar requests require consolidated SIP, an explicit point-in-time
     `asof`, `adjustment=all`, ascending order, at most 50 symbols, and at most 10,000
     rows. Strict semantic URL/query validation rejects host/path/query changes,
     redirects, non-200 status, non-JSON content, naive retrieval times, and body
     hash/length/representation mismatches.
   - Focused source and collector compatibility verification passed 44 tests. The full
     repository suite passed 1,433 tests with two skipped; tracked Ruff, strict mypy
     across 229 source files, compileall, and one consolidated independent review also
     passed. No market data was downloaded and no model or authority changed.

   Closed-session source implementation result:

   - Commit `bf35a11` adds the immutable prospective session parent and the
     `collect-edge-prospective-sip-session` command. It collects exact point-in-time
     cohort five-minute SIP bars and exact SPY, QQQ, and sector-ETF one-minute SIP bars
     only after an XNYS session closes and before the next session opens.
   - Strict replay binds exact provider bytes, transport sidecars, request identity,
     membership lineage, exchange-calendar bounds, coverage, and child-authority
     fingerprints. Benchmark gaps fail the source gate; stock gaps remain explicit and
     may not exceed the frozen 5% whole-security exclusion ceiling. Resource policy is
     capped at 4 GiB and two workers.
   - Final verification passed 1,453 tests with two skipped, tracked Ruff, strict mypy
     across 230 source files, compileall, and independent review with no remaining high-
     or medium-severity finding.
   - No real source parent has been published. Collection requires a fresh observed
     membership authority after the preceding session close and before the target
     session open, followed by collection after target close plus 60 seconds. A5.1c
     remains open, and feature, outcome, preflight, training, and serving work remain
     prohibited until their preceding authorities pass.
   - The first post-close observation on `2026-08-19` failed closed because the
     independent anchor contained `VMRK` while the active lineage contained `EQR`.
     Retained official and SEC evidence identifies this as a ticker successor on the
     same CIK, not an addition/deletion. This reopens only observed-membership anchor
     reconciliation: one anchor-only ticker may replace one active-only ticker at the
     observation time only when their CIK is identical and uniquely matched. Different
     or ambiguous identities remain fatal. Exit evidence is a regression test, strict
     replay of the real observation, and the existing focused verification suite.
   - Commit `6386cb4` implements the bounded reconciliation. It activates a ticker
     successor only at observation time when one anchor-only ticker and one active-only
     ticker share one unique CIK; a pending future event, ambiguous match, or different
     CIK remains fatal. Public collection and strict replay regressions cover the path.
   - Failed immutable attempt
     `data/raw/index_membership/sp500_observed_20260819T200115Z_v4` remains failure
     evidence. Complete authority
     `data/raw/index_membership/sp500_observed_20260819T201500Z_v5` strictly replays 503
     constituents at `2026-08-19T20:06:40.779393Z`, closes `EQR`, opens `VMRK` on the
     same `cik:0000906107`, and publishes universe hash
     `5b6e68d4844f9b0baa00e517bf0b515ddc1648a730776bf9dba201cf9082b1b3`.
   - Final verification passed 1,457 tests with two skipped, tracked Ruff, strict mypy
     across 230 source files, tracked compilation, and independent re-review with no
     remaining high- or medium-severity finding.
   - Real authority
     `data/raw/prospective_sip_sessions/session_20260820_v1` now strictly replays the
     first eligible source session: 503 stocks, 39,181 five-minute stock rows, and
     5,070 one-minute benchmark rows. Twenty-two stocks are incomplete, or 4.374%,
     below the unchanged 5% ceiling; no benchmark is incomplete. Peak working memory
     was 0.289 GiB. Status is `source_complete_warmup_ineligible` because only one of
     twenty required prior five-minute sessions exists. Features, outcomes, preflight,
     training, and serving remain prohibited.
   - Pre-open authority
     `data/raw/index_membership/sp500_observed_20260821T071500Z_v7`, poll
     `data/raw/prospective_broker_actions/poll_20260821T071000Z`, generation
     `data/research/prospective_broker_actions_generation_20260821_v1`, and separate
     horizon `data/research/prospective_analyst_revision_horizon_20260821_v4` all
     strictly replay. The poll contains 451 identity-bound observations and the new
     causal chain contains 12 qualifying analyst episodes. It remains capacity-blocked.
     The August 17 and August 21 chains cannot be combined because polling was not
     contiguous; the publisher failed closed rather than manufacturing continuity.

7. **A5.1d - Correct historical intraday event identity and run matched development
   training (`completed`).** The A5.1 preflight attached events by literal
   `security_id`. Historical event authorities encode most SEC identities as
   `cik:<value>:ticker:<symbol>`, while A4.3 uses `cik:<value>`. A reproduced audit
   found 17,401 direct-issuer analyst episodes, 15,603 episodes whose ticker occurs in
   A4.3, but only 97 episodes with a literal security-ID overlap and 19 attached
   episodes. This is an identity-normalization defect, not an event-capacity result.

   Frozen correction and training contract:

   - Reconcile events and source-coverage rows to the A4.3 namespace by exact uppercase
     ticker. When both source and target identities contain CIKs, unequal CIKs fail the
     publication. Ambiguous ticker/security mappings abstain and are audited; they are
     never guessed.
   - Preserve the original event and coverage security IDs as lineage. Do not rewrite
     either parent authority. Publish one new immutable preflight authority and require
     strict replay, identity-tamper, ambiguity, future-evidence, and coverage tests.
   - Historical `provider_publication_proxy` timestamps may support research-only
     development experiments. They may not set production eligibility, open the future
     holdout, authorize serving, or satisfy prospective promotion floors.
   - Compare four explicit research families: swing technical, swing technical plus
     broker catalyst, intraday technical, and intraday technical plus broker catalyst.
     Catalyst comparisons use the same decision rows, labels, temporal folds, security
     holdout, costs, and estimator budget as their technical controls. Missing catalyst
     coverage causes abstention and is never encoded as zero.
   - Existing verified historical authorities are inputs; no provider download occurs.
     Swing and intraday jobs run sequentially under their 5 GiB and 4 GiB caps.
   - Selection uses development/validation data only. ROC-AUC is diagnostic and remains
     co-gated by calibration, after-cost benchmark-relative economics, drawdown, trade
     count, and fold stability. The post-2026-07-08 intraday holdout remains closed.
   - Exit evidence is the corrected attached-episode count, strict authority replay,
     matched technical-versus-catalyst evaluation, focused poison tests, repository
     tests, tracked Ruff, strict mypy, and compile verification. Rejected families must
     publish `no_candidate`; blocked families must publish the exact blocker.

   Implementation result:

   - Implementation commits `4cc5d4e` and `cda4c1f` correct identity attachment and add
     the research-only event-confirmed training path. Exact ticker plus CIK-compatible
     reconciliation corrects the namespace defect while
     preserving source identities. Conflicting CIKs fail publication and ambiguous
     ticker mappings abstain. Focused identity, future-evidence, immutable replay, and
     verified-parent reuse tests pass.
   - Corrected authority
     `data/research/edge_rebuild_intraday_event_preflight_20260820_v2` strictly replays
     17,401 research episodes. It attaches 1,912 unique episodes to 83,636 A4.3
     event/decision pairs, compared with 19 episodes and 862 pairs before correction.
     Production remains blocked because all historical availability is a retrospective
     provider-publication proxy.
   - Catalyst remains a confirmation/population filter, not a direct intraday model
     feature. The event-confirmed continuation population has 14,451 development rows;
     long reversion has 12,951. Both use the unchanged technical feature contract,
     four chronological folds, one-session embargo, stable 20% security holdout, and
     frozen cost/economic gates.
   - Continuation positive-return ROC-AUC is 0.513 seen / 0.509 unseen. Long-reversion
     ROC-AUC is 0.535 / 0.519, versus technical-only 0.513 / 0.508. Neither family
     passes: selected policies have negative average trade and daily returns after
     costs, profit factor below one, negative benchmark excess, weak fold stability,
     and failed stop-risk calibration. Both publish strict `no_candidate` authorities.
   - Outputs are
     `data/models/edge_rebuild_intraday_event_confirmed_continuation_dev_20260820_v1`
     and
     `data/models/edge_rebuild_intraday_event_confirmed_long_reversion_dev_20260820_v1`.
     Peak working set remained about 3.16 GiB after eliminating duplicate dataset loads;
     the 4 GiB hard cap and 3.25 GiB safety threshold were not weakened. The future
     holdout stayed closed and serving/promotion remain prohibited.
   - Final verification passed 1,456 tests with two skipped, tracked Ruff, strict mypy
     across 231 source files, and tracked compilation. One unrelated microstructure
     retry test failed on the first full-suite run and then passed in isolation, as a
     complete file, and in the clean full-suite rerun.

8. **A5.1e - Evaluate directional intraday broker-action cohorts (`completed; no candidate`).**
   Split the corrected A5.1d research cohort with the existing governed analyst-rule
   classifier. Test upgrades and downgrades as separate 30-minute research
   specialists; audit coverage initiations independently and block them when capacity
   is insufficient. Catalyst remains a cohort filter around the unchanged technical
   estimator and is not added to the model feature vector.

   Frozen contract and exit gates:

   - Reuse the immutable A4.3 dataset, corrected A5.1d preflight, exact event identity,
     four chronological folds, one-session embargo, stable unseen-security holdout,
     feature order, costs, estimators, and economic gates. No provider download occurs.
   - Admit a directional subtype only with at least 500 independent announcements,
     200 securities, 200 sessions, 100 announcements in every validation fold, and 100
     announcements represented in the unseen-security scope. These floors prevent one
     issuer, time period, or seen-security population from dominating the result.
   - Only `bare_upgrade`, `bare_downgrade`, and `coverage` are eligible subtype names.
     Price-target and generic actions remain excluded because their direction is not
     governed by this hypothesis.
   - Train eligible subtype/hypothesis combinations sequentially under the unchanged
     4 GiB process cap. Publish `no_candidate` when validation fails and an explicit
     capacity blocker when a subtype fails the frozen floors.
   - The historical publication-time proxy remains research-only. Future holdout,
     serving, promotion, and A6 stay closed regardless of development performance.
   - Exit evidence is deterministic subtype classification, capacity and future-poison
     tests, immutable parent binding, strict output replay, focused tests, full tests,
     Ruff, strict mypy, compilation, memory evidence, implementation commit, and
     documentation closure.

   Implementation result:

   - Commit `4821780` adds one strict directional-cohort path to the existing research
     trainer. Parent authorities are verified once; only classification fields are
     retained before large parent tables are released. Subtype, capacity, and parent
     hashes remain in model lineage. Unsupported or under-capacity subtypes fail before
     the A4.3 training matrix is loaded.
   - Upgrade capacity passed with 805 announcements, 32,970 attached decisions, 285
     securities, 461 sessions, 149-202 announcements per validation fold, and 180
     unseen-security announcements. Downgrade capacity passed with 860 announcements,
     31,243 decisions, 273 securities, 439 sessions, 124-200 per fold, and 173 unseen.
   - Coverage initiation was blocked without training: 245 announcements, 169
     securities, 168 sessions, 38-55 per fold, and 41 unseen-security announcements.
     Price-target and generic actions remained excluded.
   - Upgrade continuation trained on 6,709 rows and produced seen/unseen positive-net-
     return ROC-AUC of 0.492/0.531. Upgrade long reversion trained on 5,341 rows and
     produced 0.495/0.494. Downgrade continuation trained on 5,991 rows and produced
     0.510/0.485. Downgrade long reversion trained on 6,041 rows and produced
     0.506/0.529.
   - All four are strict `no_candidate` outputs. No family reached the 0.60 AUC gate.
     Positive economics in isolated scopes were based on only 14-35 unseen-security
     trades and failed in the paired seen-security scope. Other scopes had negative
     after-cost trade returns, profit factor at or below one, or failed benchmark,
     calibration, confidence-bound, and fold-stability gates.
   - Immutable outputs are the four
     `data/models/edge_rebuild_intraday_{upgrade|downgrade}_confirmed_{continuation|long_reversion}_dev_20260820_v1`
     directories. All strictly replay as `no_candidate`; future holdout and serving
     remain closed. Peak working set was 3.193 GiB under the unchanged 4 GiB hard cap
     and 3.25 GiB safety threshold.
   - Final verification passed 1,460 tests with two skipped, tracked Ruff, strict mypy
     across 231 source files, tracked compilation, real capacity replay, and strict
     replay of all four outputs.

### A6 - Run Locked Evaluation and Promote Qualified Models

- Use purged, embargoed walk-forward validation plus a stable held-out-security stress
  test within each validation period. The security test measures transfer to unseen
  symbols; it is not an additional independent time period. Swing uses the frozen
  approximately 5/1/1-year sequence; intraday uses
  the maximum causally complete 2-3-year history with frozen calendar boundaries.
- Open each locked test once for a preregistered candidate. Report ROC-AUC, PR-AUC,
  Brier/ECE, rank IC, top-quantile lift, net SPY/QQQ/sector excess return, costs,
  turnover, drawdown, capacity, regime stability, and coverage.
- A specialist passing on a narrow cohort remains a specialist. The API abstains
  outside its verified coverage.

## Promotion Gates

A model is not promoted because materialization or training succeeds. Promotion
requires:

- immutable source, feature, label, split, and model lineage;
- exact batch/live ordered-feature parity;
- chronological and unseen-security stability;
- calibration and ranking value;
- positive net economics after costs with acceptable drawdown and capacity;
- regime, sector, and market-cap stability;
- prospective shadow evidence;
- a hash-verified atomic serving bundle.

Until then, production scoring and prediction API paths fail closed. Rejected models
are audit evidence, never fallbacks.

## Completion Checklist

- [x] Remove Reddit and Seeking Alpha from the active system.
- [x] Freeze Alpaca as the only ticker catalyst estimator source.
- [x] Enforce `2019-07-09` as the first swing model decision date.
- [x] Publish and economically reject intraday V2.
- [x] Complete repository-wide verification and memory audit.
- [x] Publish and strictly replay the single-profile V12 technical swing authority.
- [x] Preserve prior failed candidates as rejection evidence, never serving fallbacks.
- [x] Record the historical Intraday V3 experiment as invalid because five declared
  cross-sectional z-score inputs lacked a causal decision-cohort implementation.
- [x] Preserve learning-to-rank as a future estimator family, not as evidence that the
  invalid V3 feature contract was repaired.
- [x] Complete A0 research-integrity recovery and push implementation commit
  `e168482` plus its documentation closure.
- [x] Complete A2 governed swing-baseline ablation and serving contracts in
  implementation commit `cb2aba5`; no performance or promotion claim was made.
- [x] Complete A3.1-A3.4 issuer-event authority, precision, and matched-ablation work.
- [x] Separate rating changes from coverage initiation, keep price-target changes
  report-only, run the chronological capacity audit, and complete all 12 frozen A3.5
  development experiments without opening the locked test. No candidate passed.
- [ ] Build and backfill the replacement A4 intraday feature authority before
  collecting a new locked holdout; the invalid V3 contract cannot be reused.
- [ ] Promote only a model that passes every gate.


## Paused Structural Repair Checkpoint

The August 24 review reopened the incomplete package refactor with reproducible
correctness and verification failures. This checkpoint changes code structure only;
model features, labels, thresholds, data authorities, and promotion state are out of
scope.

The canonical source layout is domain-based: `core`, `sources`, `evidence`,
`universe`, `catalysts`, `modeling`, `swing`, `intraday`, `governance`, `serving`,
`commands`, and `research`. Swing and intraday each own descriptive `contracts`,
`datasets`, `features`, `labels`, `training`, `evaluation`, and `live` packages.
Chronology and checkpoint labels are prohibited in active package, module, command,
test, and task names.

1. **Holdout access and shared contract repair (`completed`).**
   Use one `IntradayDevelopmentConfig`, restore constructible causal calibration
   results, repair the direct future-holdout validation path, and cover the unmocked
   path with regression tests. Future access must remain fail-closed and auditable.
   Implementation commit `99f635c` is pushed. The direct holdout path, atomic claim,
   reservation/failure evidence, registry isolation, temporary replay validation,
   shared config, and calibration construction are covered by 57 passing focused tests.
   The assigned senior reviewer accepted the bounded diff after all P1 findings were
   fixed; focused Ruff and strict mypy pass.
2. **Serialized artifact and namespace inventory (`completed`).**
   Inventory every command, manifest, model artifact, and import that depends on a
   chronology-named namespace. Retrain retained models under canonical modules or
   explicitly retire rejected artifacts before deleting their code dependencies.
   Implementation commit `3026450` is pushed. The machine-readable retention inventory
   classifies every artifact group needed by the research catalog, tracked hash-bound
   evidence, active replay claims, old serialized namespaces, and ungoverned outputs.
   The four research catalog models now use behavior-based IDs and local paths. Their
   authority, manifest, and candidate hashes are unchanged; only swing technical has a
   non-actionable research candidate, and all four remain promotion-ineligible. No
   model output was deleted because tracked evidence or incomplete regeneration
   provenance still blocks broader cleanup. Ten focused tests, touched Ruff, strict
   mypy, default-service smoke, retained-bundle hash checks, and specialist evidence
   replay checks passed. The assigned senior reviewer accepted the bounded diff.
3. **Market evidence and research package migration (`completed`).**
   Consolidate base contracts under `core`, immutable lineage under `evidence`, source
   transports under `sources`, membership and identity under `universe`, issuer and
   global events under `catalysts`, reusable estimators and validation under
   `modeling`, and non-production experiments under `research`.
   - **Cross-sectional research consolidation (`completed`).** Implementation commit
     `ade847c` removes the `market_predictor.v3` source package and replaces active
     chronology-named APIs and tests with behavior-based names. Generic contracts are
     split among `core`, `evidence`, `modeling`, and `universe`; raw S&P Global archive
     transport is under `sources/spglobal`; verified index changes and point-in-time
     membership are under `universe/sp500`; reusable validation, calibration, and
     ranking economics are under `modeling`; and candidate evaluation and development
     experiments remain under `research`. An AST guard prevents production imports of
     `research` or `commands`. The two audited rejected joblibs serialized against the
     removed namespace were deleted while their manifests were retained; no other
     model artifact was removed. Across the migrated modules and direct consumers, 330
     focused test cases passed. Ruff, strict mypy on 44 source files, CLI import smoke,
     diff checks, and post-review consumer reruns passed. The assigned senior reviewer
     accepted the final diff with no P0, P1, or P2 finding.
   - **Historical membership and security identity authority migration (`completed`).**
     Implementation commit `60cff69` moves corpus integrity to `evidence`, membership
     identity validation and SEC identity authority to `universe`, and historical S&P
     transition and membership authorities to `universe/sp500`. Every direct consumer
     now imports the semantic package; the five old modules are absent and guarded
     against reintroduction across source, tests, and scripts. A universe dependency
     allowlist enforces the current lower-layer boundary. Persisted schemas, hashes,
     locking, memory gates, and authority behavior are unchanged. Sixty authority tests
     and 106 consumer tests passed before review; four additional import-form poison
     tests passed after review remediation. Ruff, strict mypy on 15 source files,
     import smoke, diff checks, and process checks passed. The assigned senior reviewer
     accepted the final diff with no P0, P1, or P2 finding.
   - **Prospective observed-membership source and authority split (`completed`).**
     Implementation commit `5259bdb` moves provider URLs, HTTP collection, response
     validation, retained raw units, source parsing, and raw replay to
     `sources/spglobal/observed_membership_collection.py`. Observed membership lineage,
     identity reconciliation, effective-state construction, publication, and strict
     authority replay now belong to
     `universe/sp500/observed_membership_authority.py`. The universe orchestrator keeps
     the single operation lock and generates the unchanged request hash passed into
     every raw unit. Authority-root inventory and raw `objects`/`units` inventory are
     verified separately. The old combined module is absent; all consumers use the
     semantic authority package. Architecture tests prevent source-to-universe imports
     and all import forms of the removed path. Exact raw-envelope and lock-contention
     tests pass. Final checkpoint verification passed 113 affected tests, Ruff, strict
     mypy on six source files, compileall, import smoke, and diff checks. The assigned
     senior reviewer accepted the diff with no P0, P1, or P2 finding.
   - **Issuer and global catalyst authority migration (`completed`).** Move the
     remaining source-independent issuer-event, SEC-filing, global-event, and
     market-context authorities out of `edge_rebuild` into `catalysts`, `sources`, and
     `evidence` without changing source coverage, availability, or causal semantics.
     - **SEC filing evidence and decision authority (`completed`).** Implementation
       commit `9244893` creates `catalysts/sec_filings`, moves causal filing collection
       evidence to `collection.py`, and moves decision-time overlays to
       `decision_authority.py`. `sources/sec.py` remains the provider transport. The
       two old modules and edge-rebuild-prefixed tests are absent and guarded against
       reintroduction. Persisted schemas, availability, coverage missingness, raw
       replay, artifact hashes, memory limits, and command behavior are unchanged.
       Forty-four SEC, architecture, and CLI tests passed with Ruff, strict mypy on
       three source files, compileall, import smoke, zero-reference and diff checks.
       The assigned senior reviewer accepted the AST-equivalent move with no P0, P1,
       or P2 finding.
     - **GDELT collection and global-event authority (`completed`).** Implementation
       commit `5e1f65f` makes `sources/gdelt.py` the single strict provider transport,
       moves immutable canonical evidence into `catalysts/global_events/collection.py`,
       and moves the decision-time overlay into
       `catalysts/global_events/decision_authority.py`. Active APIs, commands, and
       tests use behavior names; the old modules and duplicate transport are absent.
       Provider URL identity, no-redirect behavior, request parameters, raw-response
       hashes, query/scorer policy hashes, canonical event hashes, availability,
       coverage, and persisted schema strings remain fail-closed and replay-compatible.
       Ninety-eight focused tests passed with Ruff, strict mypy on six source files,
       compileall, architecture guards, and diff checks. The assigned senior reviewer
       accepted the final diff with no P0, P1, or P2 finding.
     - **Issuer event family and precision authorities (`completed sequence`).** Move causal
       issuer evidence in four independently reviewed checkpoints without changing
       any classifier, coverage, audit, or artifact contract:
       1. **Alpaca issuer-news evidence collection and audit (`completed`).**
          Implementation commit `4f271ca` moves the immutable collector and strict
          audit from `swing` to `catalysts/issuer_events`; `sources/alpaca.py` remains
          the provider transport. Persisted schema strings are unchanged in
          `news_history_contracts.py`. Canonical symbols now belong to `core/symbols.py`
          and provider-specific mappings to `sources/provider_symbols.py`, avoiding a
          reverse catalyst dependency on the old mixed symbol module. Old modules,
          imports, and tests are absent and guarded against reintroduction. Verification
          passed 109 focused parity tests and the complete suite with 1,530 passed and
          2 skipped, plus affected-file Ruff, strict mypy on 14 source files,
          compileall, dependency/file-absence guards, and diff checks. The assigned
          senior reviewer found no P0, P1, or P2 issue.
       2. **Classification and attribution foundations (`completed`).**
          Implementation commit `2b9e195` moves reusable event-family classification,
          relevance, attribution, and attribution-history behavior to
          `catalysts/issuer_events`. The rule-variant helper has one semantic owner in
          `classification.py`; an AST guard rejects definitions, imports, or assignment
          aliases in its old precision-audit owner. Exact policy hashes, schema/version
          strings, every rule-variant branch, representative outputs, and strict replay
          remain fixed. Old modules, imports, and swing-prefixed foundation tests are
          absent and guarded. Verification passed 247 focused parity tests, 88 tests
          after reviewer fixes, 54 ownership/dependency tests, and the final complete
          suite with 1,551 passed and 2 skipped. Affected-file Ruff, strict mypy on 12
          source files, compileall, removed-module scans, diff checks, and process checks
          passed. The assigned senior reviewer accepted the final diff with no P0, P1,
          or P2 finding.
       3. **Swing catalyst decision authority (`completed`).** Implementation commit
          `9408515` moves the decision-time swing feature authority to
          `swing/features/catalyst_decision_authority.py`. All consumers use the new
          semantic path directly; the old module and test are absent and guarded.
          Persisted request, authority, manifest, lineage, decision-artifact, and
          coverage-artifact identities remain unchanged. A bidirectional architecture
          guard now prohibits imports between `swing` and `intraday`. Verification
          passed 171 focused tests with one skipped and the complete suite with 1,569
          passed and 2 skipped. Affected Ruff, strict mypy on six source files,
          compileall, removed-path scans, diff checks, and memory/process checks passed.
          The assigned senior reviewer accepted the final diff with no P0, P1, or P2
          finding.
       4. **Issuer-family evidence and horizon assignment split (`completed`).**
          Implementation commit `03f8233` preserves the two retained combined v2
          envelopes byte-for-byte while separating their runtime ownership. Strict
          structural verification and a swing-independent neutral projection identity
          belong to `evidence/issuer_family_combined_envelope.py`; neutral classified
          events, coverage, and unclassified semantic replay belong to
          `catalysts/issuer_events/family_evidence.py`; swing assignments and cohort
          replay belong to `swing/datasets/issuer_event_family_cohort.py`. Intraday now
          consumes only neutral evidence and cannot access swing assignments. A true
          persisted-authority split would change schemas and hashes, so it remains a
          separately approved data migration rather than part of this byte-preserving
          checkpoint. Verification passed 178 focused tests and the complete suite with
          1,584 passed and 2 skipped. Both retained v2 eras passed strict real-data
          replay below 2 GiB, affected-file Ruff and strict mypy passed, and the assigned
          reviewer accepted the final diff with no P0, P1, or P2 finding.
       5. **Issuer-event precision governance (`completed`).** Implementation commit
          `7ce23a0` moves deterministic sampling, blind review resolution, artifact
          integrity, and family/rule-variant admission into
          `governance/issuer_event_precision`. The old combined module is absent and
          guarded against reintroduction; command names and swing-ablation behavior
          are unchanged. Public loaders remain strict, staged publication validates
          fully rewritten final paths before atomic rename, and failure-injection tests
          prove invalid authorities are never made visible. Both retained periods
          replay with unchanged sample/audit authority hashes and 1,796/1,859 review
          rows. Verification passed 120 affected tests with one skipped, 25 final
          governance tests with one skipped, and the complete suite with 1,594 passed
          and three skipped. Affected Ruff, strict mypy, compileall, removed-path
          scans, diff checks, and process checks passed. The assigned senior reviewer
          accepted the final diff with no P0, P1, or P2 finding.
       The required dependency direction is `sources -> catalysts -> swing ->
       governance`; commands remain outer adapters.
4. **Swing and intraday package migration (`paused`).**
   Consolidate each horizon under descriptive `contracts`, `datasets`, `features`,
   `labels`, `training`, `evaluation`, and `live` packages and remove the intraday
   evaluation module/package collision. Compatibility aliases are prohibited because
   this repository is not deployed.
   - **Intraday module/package collision removal (`completed`).** Implementation
     commit `a176fbb` deletes the unreachable `intraday/contracts.py` and
     `intraday/evaluation.py` shadows. Runtime and pickle ownership remain in the
     existing canonical packages; no consumer import or artifact identity changed. A
     recursive architecture guard rejects future module/package collisions, with an
     explicit poison fixture and deleted-file assertions. Characterization freezes
     package origins, schema and feature-order hashes, label policy/hash, validators,
     pickle ownership, and deterministic evaluation outputs. Verification passed 143
     affected tests and the complete suite with 1,601 passed and three skipped. Touched
     Ruff, compileall, direct-path scans, diff checks, cleanup, and process checks
     passed. The assigned senior reviewer accepted the diff with no P0, P1, or P2
     finding.
   - **Shared strategy contract migration (`completed`).** Implementation commit
     `c408d58` moves the cross-horizon contract from `edge_rebuild` to `modeling` and
     updates every consumer directly without a compatibility alias. The schema string
     remains `edge_rebuild.strategy_contract.v2`, the active configuration SHA-256
     remains `39213ad6bd5c1f09f30065f737ffecadf05bbb0ae81b81f2ffda7a343967e972`,
     and retained artifact scans found no serialized Python owner at the removed path.
     Architecture guards reject every old import form and reintroduction of the removed
     file. Verification passed 452 affected tests with two skipped and the complete
     suite with 1,605 passed and three skipped. Ruff on the migrated authority and its
     boundary tests, strict mypy, compileall, import smoke, removed-path scans, diff
     checks, and process-memory checks passed. The assigned senior reviewer accepted
     the final diff with no P0, P1, or P2 finding.
   - **Shared mathematical primitive ownership (`completed`).** Implementation commit
     `8d42d26` moves the only horizon-neutral authority found in this pass,
     `FeatureStep` and `FeaturePipeline`, from `edge_rebuild` to
     `modeling/feature_pipeline.py`. The implementation is byte- and AST-identical;
     swing and intraday consumers import the new owner directly, no alias exists, and
     architecture tests reject every old import form and old-file reintroduction. The
     review explicitly keeps cross-sectional scaling and technical relationships out of
     `modeling`: their current policies and consumers are swing-specific, so they move
     later to `swing/features`. The mixed label module must be split during the horizon
     label migrations rather than moved wholesale. Verification passed 119 focused
     tests and the complete suite with 1,613 passed and three skipped. Touched Ruff,
     strict mypy, compileall, code-hash parity, removed-path scans, diff checks, and
     process-memory checks passed. The assigned senior reviewer accepted the final diff
     with no P0, P1, or P2 finding.
   - **Intraday history-collection contract migration (`completed`).** Implementation
     commit `09341dc` moves the complete intraday Alpaca/SIP history acquisition
     contract from `edge_rebuild/history_contracts.py` to
     `intraday/contracts/history_collection.py`. The implementation is byte- and
     AST-identical with source hash
     `6b5d3b42c73aeb40958ca01b5a35b2a821d1de46`; all collectors, intraday datasets,
     command adapters, and tests import the new owner directly. No compatibility alias
     exists. New characterization freezes all eight Pydantic owners and the six active
     configuration schema/hash pairs. Architecture guards reject every old import form
     and old-file reintroduction, while existing dependency guards prohibit provider
     sources from importing horizon code. Verification passed 171 focused tests, 56
     interrupted-refactor regression tests, and the complete suite with 1,631 passed
     and three skipped. Touched Ruff, strict mypy, compileall, CLI import/help, exact
     code-hash parity, artifact scans, diff checks, and the 4 GiB process-memory gate
     passed. The assigned senior reviewer accepted the final diff with no P0, P1, or
     P2 finding.
   - **Swing contract package and materialization ownership (`completed`).**
     Implementation commit `26c048d` converts `swing/contracts.py` byte-for-byte into
     `swing/contracts/__init__.py`, preserving the Python and pickle owner
     `market_predictor.swing.contracts`, and moves the swing materialization schema
     constants from `edge_rebuild` to `swing/contracts/materialization.py`. All
     consumers use the canonical materialization module directly; regression tests
     prohibit accidental constant aliases on the legacy materialization and training
     modules. Both moves are byte- and AST-identical, with source identities
     `36b698837a09a8cd0b23e9b48e4be291afa91727` and
     `c7add055ab12ab53d46988f89da862f0a631649a`. Characterization freezes config
     owners and pickle round trips, both materialization schemas, the 99-feature and
     53-feature profile hashes, three default config hashes, and the label-policy hash.
     Verification passed 208 affected tests with one skipped and the complete suite
     with 1,645 passed and three skipped. New-file Ruff, changed import-order Ruff,
     strict mypy, compileall, import/no-alias smoke, old-path and artifact scans, diff
     checks, and the 4 GiB process-memory gate passed. Known pre-existing unused
     re-export findings in `swing_training.py` remain assigned to Step 6. The reviewer
     found one accidental-alias P2, verified its fix, and accepted the final diff with
     no remaining P0, P1, or P2 finding.
   - **Swing technical-relationship feature ownership (`completed`).** Implementation
     commit `dd4dbcd` moves `technical_relationships.py` byte-for-byte from
     `edge_rebuild` to `swing/features`, updates both lazy pipeline consumers directly,
     and renames the characterization test for descriptive ownership. Source identity
     remains `391bac1540b6ef414dced0338b842cedc5e54bdb`; no alias exists. Tests freeze
     the new `TechnicalRelationshipSpec` owner and pickle round trip, nine-column
     output-order hash, strategy-derived specification hash, representative output
     hash, future-prefix causality, and session-boundary resets. Architecture guards
     reject all old import forms and old-file reintroduction. Verification passed 170
     affected tests with two skipped and the complete suite with 1,651 passed and three
     skipped. New-owner Ruff, changed import-order Ruff, strict mypy, compileall,
     import smoke, source parity, old-path and artifact scans, diff checks, and the 4
     GiB memory gate passed. The reviewer accepted the final diff with no P0, P1, or
     P2 finding.
   - **Swing cross-sectional feature ownership (`completed`).** Implementation commit
     `68d9893` moves `cross_sectional.py` byte-for-byte from `edge_rebuild` to
     `swing/features`, updates all three consumers with module-qualified imports, and
     renames its characterization test descriptively. Source identity remains
     `cfb54c43d06382235fd341d9a9713a5262715c4f`; no alias exists. Tests freeze the
     `CrossSectionSpec` owner and pickle round trip, suffixes, specification and emitted
     column hashes, representative output hash, future-session causality, session and
     sector isolation, winsorization, peer floors, collision handling, and empty-frame
     behavior. Verification passed 181 affected tests with two skipped and the resumed
     complete suite with 1,660 passed and three skipped. New-test and changed-import
     Ruff, strict mypy, compileall, import smoke, source parity, old-path and artifact
     scans, diff checks, and the 4 GiB memory gate passed. The moved byte-identical file
     retains its pre-existing import-spacing Ruff finding for Step 6. The reviewer
     accepted the final diff with no P0, P1, or P2 finding.
   - **Shared label outcomes and swing barrier/rank ownership (`completed`).**
     Implementation commit `61f6f4c` makes `modeling/label_outcomes.py` the only
     owner of the six integer outcome/rank constants, converts `swing/labels.py`
     byte-for-byte to `swing/labels/__init__.py`, and moves the daily barrier and
     session/sector ranking implementation to `swing/labels/barrier_and_rank.py`.
     Swing and intraday consumers use module-qualified canonical imports; no alias or
     old file remains. Tests freeze constant, specification, column, representative
     output, dtype, package, and pickle identity; append-only causality and group
     isolation remain explicit. Modeling is now guarded against absolute and relative
     imports from either horizon. Verification passed 155 direct label/boundary tests,
     133 broader regression tests with two skipped, and the complete suite with 1,681
     passed and three skipped in 21 minutes 51 seconds. Affected Ruff, strict source
     mypy, compileall, old-path scans, staged diff checks, temporary-output cleanup,
     and the 4 GiB process gate passed. The assigned reviewer verified all P2 fixes and
     approved the final diff with no remaining P0, P1, or P2 finding.
   - **Intraday causal volume-bar dataset ownership (`completed`).** Implementation
     commit `e76bf8d` moves the byte-identical implementation to
     `intraday/datasets/volume_bars.py`, renames its test descriptively, and updates all
     three dataset consumers directly. No alias or old file remains. Because the
     transformation identity hashes `bar_dataset.py` itself, the direct import change
     truthfully creates schema `market_predictor.intraday.bar_dataset_transformation.v2`
     with aggregate SHA-256
     `6fdfd0c8f07e4f7445b66d038cbd936e4459db68e087a5ddbcb30eac4795cb51`.
     Source hashing now canonicalizes only CRLF/LF so Windows publication and Linux
     replay share one identity. Tests freeze the canonical source, column,
     representative output, transformation, pickle, causality, isolation, threshold,
     remainder, eligibility, memory, publication, resume, and immutability contracts.
     The retained `edge_rebuild_intraday_bar_only_causal_20260814_v1` authority remains
     unchanged historical evidence under transformation `0da898cc...`; it is rejected
     by the current loader and cannot train or promote. Its audit, two event-preflight
     authorities, and eight retained development/rejection model bundles are likewise
     historical. Deterministic rematerialization completed at
     `intraday_causal_volume_bar_dataset_20260831_v2` with request `5e8c508a...303a` and
     the current transformation `6fdfd0c8...cb51`. Audit-remediation commit `1829fce`
     adds projection-lineage, raw-source-cutoff, and five-minute-prefix poison gates.
     Future CLI publications require separate hash-bound per-invocation execution
     evidence with aggregate process/worker memory validation and crash recovery. The
     present artifact's execution assessment remains incomplete because that telemetry
     was introduced after its resumable invocations. Verification passed 25 focused
     remediation tests and the complete suite with 1,714 passed and three skipped;
     affected Ruff, strict mypy, compileall, diff, CLI-help, and transformation checks
     passed. Independent code and ML-design reviewers reported no remaining P0, P1, or
     P2 finding. This registers a current data authority only: intraday training,
     promotion, serving, and locked-test access remain unauthorized.
   - **Canonical intraday ledger ownership (`completed`).** Supplemental commit
     `a4002ce` completes an interrupted test refactor by making
     `intraday/evaluation/ledger.py` the sole owner of position-ledger construction,
     position closing, and ledger metrics. The seven functions removed from
     `economics.py` were AST-identical duplicates; `economics.py` now owns only
     ranking diagnostics. Production gates and training coordination import the
     canonical ledger directly, and a runtime identity test prevents tests from
     exercising an unused implementation. Verification passed 173 affected tests,
     Ruff, strict mypy, compileall, and a one-owner scan. The code reviewer approved
     the final diff with no remaining P0, P1, or P2 finding. The complete isolated
     suite after both implementation commits passed 1,691 tests with three skipped in
     13 minutes 30 seconds.
   - **Intraday selected-session planning ownership (`completed`).** Implementation
     commit `d34ea25` moves the complete selected stock-session verification and
     one-minute/five-minute acquisition-plan publisher byte-for-byte from
     `edge_rebuild/selected_session_history.py` to
     `intraday/datasets/selected_session_history.py`. All five production consumers
     and both direct test consumers import the canonical owner; no compatibility alias
     or old file remains. The source Git object remains
     `2f43aa9ffd48a8a4f76f7b7bb448c204ccf7f2bd`. Tests freeze the new function and
     `SelectedSession` owners, pickle round trip, exchange-calendar and early-close
     behavior, selection lineage, one-minute alignment, and all old import forms.
     Persisted schemas, policy hashes, plan fingerprints, unit identities, and existing
     authorities are unchanged. The current volume-bar transformation remains
     `6fdfd0c8...cb51`, and its 794-session authority replays without regeneration.
     Verification passed 174 focused tests and the complete suite with 1,719 passed and
     three skipped. Affected Ruff, strict mypy, compileall, CLI help, import/old-path,
     source-parity, diff, process, and temporary-output checks passed. Repository-wide
     static results remain the recorded Step 6 baseline of 168 Ruff and 14 strict mypy
     findings in untouched files. Independent code and ML-design reviewers reported no
     remaining P0, P1, or P2 finding.
   - **Intraday canonical history materialization ownership (`completed`).**
     Implementation commit `7e96bc4` moves the complete two-pass bar shuffle,
     exchange-session segmentation, selected-session eligibility, overlapping-source
     resolution, and ticker-defect quarantine byte-for-byte from
     `edge_rebuild/history_materialization.py` to
     `intraday/datasets/history_materialization.py`. The command adapter and both test
     consumers import the canonical owner; no compatibility alias or old file remains.
     The source Git object remains `f5acc0e7de04dc00f0213a54beec1b8a5531f74a`.
     Tests freeze `SessionBounds` owner and pickle identity, all five public function
     owners, every old import form, early-close/session-segment behavior, source
     overlap, ticker quarantine, and selected-session eligibility. Persisted schemas
     remain `edge_rebuild.intraday_materialization.v1` and
     `edge_rebuild.intraday_materialization_authority.v1`; no authority regeneration
     occurred. The current volume-bar transformation remains `6fdfd0c8...cb51`, and
     its 794-session authority replays unchanged. Verification passed 200 focused tests
     and the complete suite with 1,724 passed and three skipped. Affected Ruff, strict
     mypy, compileall, CLI help, import/old-path, source-parity, diff, process, and
     temporary-output checks passed. Repository-wide static results remain the Step 6
     baseline of 168 Ruff and 14 strict mypy findings in untouched files. Independent
     code and ML-design reviews closed with no remaining P0, P1, or P2 finding.
   - **Intraday Alpaca/SIP history collection ownership (`completed`).**
     Implementation commit `01275d4` moves the complete bounded intraday collector
     byte-for-byte from `edge_rebuild/history_collection.py` to
     `intraday/datasets/history_collection.py`. All six production consumers and four
     direct test consumers import the canonical owner; no compatibility alias or old
     file remains. The source Git object remains
     `ce3c6f3132b1bf10df1f50b2a24a28aee0c6b2b4`, and tests freeze all three public
     function owners plus every old import form. All six persisted collection, unit,
     and authority schema strings remain unchanged. Retained stock collection evidence
     replays 2,116 units and 16,636,841 rows at authority `63d9d714...fd5c`; retained
     benchmark evidence replays 794 units and 4,005,350 rows at authority
     `889dcc46...dde6`. Their manifest hashes are unchanged. The derived 794-session
     volume-bar authority still replays 3,095,688 rows, including 1,365,015 eligible
     rows, under request `5e8c508a...303a` and transformation `6fdfd0c8...cb51`.
     Verification passed 242 focused tests and the complete suite with 1,729 passed
     and three skipped. Affected Ruff, strict mypy, compileall, collection CLI help,
     import/old-path, source-parity, real-authority replay, diff, process, and temporary
     output checks passed. Repository-wide static results remain the Step 6 baseline
     of 168 Ruff and 14 strict mypy findings in untouched files. Independent task,
     code, and ML/data-design reviews closed with no remaining P0, P1, or P2 finding.
   - **Selected-session one-minute coverage ownership (`completed`).**
     Implementation commit `268c62d` moves the complete one-minute coverage and
     canonical five-minute verification authority byte-for-byte from
     `edge_rebuild/one_minute_coverage.py` to
     `intraday/datasets/one_minute_coverage.py`. All five production consumers and two
     direct test consumers use the canonical owner; no compatibility alias or old file
     remains. The source Git object remains
     `c770f369f8f103d0895c4fb9d8363f29b0006ead`, and tests freeze all three public
     function owners plus every old import form. Persisted schemas remain
     `edge_rebuild.selected_session_one_minute_coverage.v2` and
     `edge_rebuild.selected_session_one_minute_coverage_authority.v2`. The retained
     authority replays ready at 43,226 stock-sessions across 502 securities, with 13
     incomplete sessions retained as metadata, zero excluded securities, a 95%
     continuity floor, and the unchanged 5% whole-security exclusion ceiling. Its
     authority remains `e18adef7...a75b` and manifest `d21c1733...c560`. The downstream
     794-session volume-bar authority still replays 3,095,688 rows, including 1,365,015
     eligible rows, under request `5e8c508a...303a` and transformation
     `6fdfd0c8...cb51`. Verification passed 249 focused tests and the complete suite
     with 1,734 passed and three skipped. Affected Ruff, strict mypy, compileall, CLI
     help, import/old-path, source-parity, retained-authority replay, diff, process, and
     temporary-output checks passed. Correcting import order in two touched files
     reduced repository-wide Ruff debt from 168 to 166; strict mypy debt remains 14
     findings in the same three untouched files. Independent task, code, and ML/data
     reviews closed with no remaining P0, P1, or P2 finding.
   - **Selected-session benchmark acquisition planning ownership (`completed`).**
     Implementation commit `9a59d6f` moves the benchmark acquisition-plan builder
     byte-for-byte from `edge_rebuild/benchmark_history.py` to
     `intraday/datasets/benchmark_history.py`. The command adapter and direct tests use
     the canonical owner; no compatibility alias or old file remains. The source Git
     object remains `1c106774cadf7fdf6c514406f72590cf0d782e62`, and tests freeze the
     function owner plus every old import form. The policy still requires SPY, QQQ,
     and all eleven sector ETFs, exact XNYS regular-session bounds, 390 normal-session
     and 210 early-close one-minute bars, SIP, `adjustment=all`, and a 4 GiB process
     limit with 0.75 GiB headroom. The retained 794-session plan replays unchanged at
     manifest `b855b250...e72`, authority `ffd0783d...0a0`, and fingerprint
     `af77d941...6616`; it contains 13 benchmarks and eight early closes. No provider
     request, artifact regeneration, training, promotion, serving, or locked-test
     access occurred. Verification passed 228 focused tests and the complete suite
     with 1,739 passed and three skipped. Compileall, CLI help, canonical/old-path
     imports, source parity, retained-plan replay, diff, process, and temporary-output
     checks passed. Repository-wide Step 6 debt remains 166 Ruff findings and 14
     strict-mypy findings in the same three untouched files. Independent task, code,
     and ML/data reviews closed with no remaining P0, P1, or P2 finding.
   - **Broad intraday five-minute acquisition planning ownership (`completed`).**
     Implementation commit `7affd05` moves the two broad-history plan functions
     byte-for-byte from `edge_rebuild/broad_intraday_history.py` to
     `intraday/datasets/broad_intraday_history.py`. The command adapter and direct
     tests use the canonical owner; no compatibility alias or old file remains. The
     source Git object stays `a014735fd60d6ba04d764804854649c752896cf8`, and tests
     freeze both function owners plus every old import form. Alpaca SIP five-minute
     bars with `adjustment=all`, exact XNYS regular-session bounds, existing-corpus
     subtraction, truncated-session replanning, explicit fund exclusion, and the
     current-snapshot limitation of the broad Finviz proxy are unchanged. The latest
     retained research plan strictly replays 814 sessions, 448,151 missing
     symbol-sessions across 579 symbols, 9,428 bounded acquisition units, and
     34,793,778 maximum expected rows at fingerprint `cc7ead87...3b8e`, manifest
     `ae454a08...5d88a`, and authority `f372997e...9e6f`. The earlier logically
     equivalent plan also replays and remains untouched. No provider request,
     artifact regeneration, training, promotion, serving, or locked-test access
     occurred. Verification passed 213 focused tests and the complete suite with
     1,744 passed and three skipped. Touched Ruff and strict mypy, compileall, CLI
     help, canonical/old-path imports, source parity, both retained-plan replays,
     diff, process, and temporary-output checks passed. Repository-wide Step 6 debt
     remains 166 Ruff findings and 14 strict-mypy findings in the same three untouched
     files. Independent task, code, and ML/data reviews closed with no remaining P0,
     P1, or P2 finding.
   - **Extended-session context acquisition planning ownership (`completed`).**
     Implementation commit `1702991` moves pre/post-market planning from
     `edge_rebuild/extended_session_context.py` to
     `intraday/datasets/extended_session_context.py` and removes the old owner without
     an alias. It also closes two independent-review findings: ER1B now binds the
     membership Parquet hash, membership-audit hash, and universe snapshot ID to the
     frozen ER1A plan; and an explicit suffix must be a non-empty canonical XNYS
     session rather than a weekend/holiday that the calendar could round forward.
     Tests freeze the function owner, all old import forms, empty-date rejection, DST
     conversion, and normal/early-close bar counts. Alpaca SIP five-minute
     `adjustment=all` acquisition, exact 04:00 ET-to-open and close-to-20:00 ET
     windows, atomic publication, and the hard separation from regular-session VWAP,
     EMA, ATR, and RVOL remain unchanged. Both retained plans and the completed
     1,925,863-row collection strictly replay without regeneration. The suffix plan
     remains 313 sessions and 6,886 units at fingerprint `e005505b...f7999`, manifest
     `8e753572...3415`, and authority `ce0842af...52d9`; collection manifest remains
     `f22147ee...5658`. Verification passed 217 focused tests and the complete suite
     with 1,755 passed and three skipped. Touched Ruff and strict mypy, compileall, CLI
     help, canonical/old-path imports, exact ER1A membership lineage, retained-plan
     and collection replay, diff, process, and temporary-output checks passed.
     Repository-wide Step 6 debt remains 166 Ruff findings and 14 strict-mypy findings
     in the same three untouched files. Independent task, code, and ML/data reviews
     closed with no remaining P0, P1, or P2 finding.
   - **Prospective closed-session SIP evidence ownership (`completed`).**
     Implementation commit `3791541` moves the closed-session collector from
     `edge_rebuild/prospective_sip_session.py` to
     `intraday/datasets/prospective_sip_session.py`; the collection command imports the
     canonical owner directly, every old import form is prohibited, and no alias
     remains. The original source Git object was
     `3ff60848019791e26ada7cb0eaee814d55e37932`. The migration also closes independent
     review findings: completed output is accepted only when the requested session,
     membership authority, policy bytes, effective configs, and benchmark set match;
     fresh configs must reconstruct exactly from their recorded policy files; the
     invocation unit limit is shared across stock and benchmark children; and the
     retained stock membership table must equal the causally active observed cohort.
     A crash after both children complete can publish the parent after the next open
     only when all immutable retrieval timestamps remain inside `[close + 60 seconds,
     next open)`; incomplete children cannot make late Alpaca requests.

     The retained `2026-08-20` authority strictly replays 503 securities, 39,181 SIP
     five-minute stock rows, all 13 required benchmark ETFs, and 5,070 SIP one-minute
     benchmark rows. Twenty-two sparse securities produce a 4.3738% exclusion rate,
     below the unchanged 5% ceiling. Status remains
     `source_complete_warmup_ineligible`; training, selection, and serving eligibility
     are false. No provider request, artifact regeneration, training, promotion,
     serving, or locked-test access occurred. Verification passed 249 focused tests
     and the complete suite with 1,772 passed and three skipped. Touched Ruff and
     strict mypy, compileall, collection CLI help, canonical and old-path imports,
     retained-authority replay, diff, process, and temporary-output checks passed.
     Repository-wide Ruff debt is now 165 findings; strict mypy debt remains 14
     findings in the same three untouched files. Independent task, Python-code, and
     ML/data-design reviews closed with no remaining P0, P1, or P2 finding.
   - **Prospective broker-action evidence ownership (`completed`).** Implementation
     commit `4a98f29` moves Alpaca polling and immutable generation publication from
     `edge_rebuild/prospective_broker_actions.py` to
     `intraday/datasets/prospective_broker_actions.py`; no alias remains and every old
     import form is prohibited. The intraday owner is intentional because collection
     binds the A4.3 security namespace and observed poll cadence. Downstream issuer
     classification remains under catalyst policy.

     Historical retained polls use a hash-only identity verifier and therefore remain
     auditable after the A4.3 transformation changed, but stale bar rows cannot enter
     training, selection, or serving. Fresh collection still requires the full current
     bar authority. Strict replay now covers exact JSON schemas, duplicate/non-finite
     values, numeric types, causal timestamps, derived counts, child eligibility,
     registry claims/commits, Windows reparse ancestry, and immutable artifact paths.
     Generation publication stages under a verified ownership marker and atomically
     renames only after complete replay; unowned staging is never deleted.

     Three retained polls replay 76, 460, and 451 observations; two retained generations
     replay 536 and 451 revisions. They remain explicitly ineligible for training and
     serving. No provider call, artifact regeneration, training, promotion, serving, or
     locked-test access occurred. Verification passed 283 focused tests and the final
     complete suite with 1,810 passed and three skipped. Touched Ruff, strict mypy,
     compileall, CLI help, import guards, retained evidence replay, diff, memory, and
     temporary-output checks passed. Independent Python and ML/data reviewers reported
     no remaining P0, P1, or P2 finding.
   - **Prospective analyst-revision horizon ownership (`completed`).** Implementation
     commit `7419df8` moves classification, episode construction, source coverage,
     capacity audit, publication, and strict replay from
     `edge_rebuild/prospective_analyst_revision_horizon.py` to
     `intraday/datasets/prospective_analyst_revision_horizon.py`. The command uses the
     canonical owner, all old import forms are prohibited, and no alias remains.

     The publisher now handles zero-news horizons, enforces pre-load and post-load
     memory checks plus bounded parent bytes and projected expansion, requires exact
     metadata and frame schemas, verifies cross-generation security identity, and
     recomputes provider-time collisions and event first-seen time across the complete
     horizon. Owned sibling staging and atomic rename prevent partial publication.
     Source-event capacity is kept separate from matched market-session capacity, and
     training and serving eligibility remain false.

     Both retained authorities replay without regeneration. The August 17 horizon has
     536 classified revisions, three episodes, and 1,006 coverage rows; the August 21
     horizon has 451 revisions, 12 episodes, and 503 coverage rows. Verification passed
     273 focused migration/integration tests, the restored 57-test intraday development
     file, and the exact full suite with 1,846 passed and three skipped. Touched Ruff,
     strict mypy, compileall, CLI help, retained replay, diff, and temporary-output
     checks passed. Independent Python and ML/data reviews closed with no remaining P0,
     P1, or P2 finding. Repository-wide debt is 166 Ruff findings and 14 strict-mypy
     findings in three untouched intraday dataset files.
   - **Prediction-data readiness governance (`completed`).** Implementation commit
     `e417960` moves readiness orchestration and contracts to
     `governance/readiness`, extracts immutable authority replay to
     `evidence/readiness_authority.py`, and moves promoted-bundle contracts and
     verification to `governance/promotion`. The research CLI now exposes the
     behavior-named `audit-prediction-data-readiness` command and uses
     `configs/prediction_data_readiness.toml`; no readiness compatibility alias
     remains.

     Current authorities bind exact swing and intraday strategy identities, XNYS
     calendar package/version, costs, folds, dimensions, exclusions, catalyst channel
     policy, availability, counts, and causal event/assignment replay. Historical-v1
     evidence remains strictly replayable but cannot authorize current planning.
     Readiness loads and reduces each large horizon independently under the 4 GiB
     process limit. Pytest discovery is fixed to `tests/`, so local data, model,
     scratch, and cache artifacts cannot change repository test collection.

     Verification passed 354 focused tests with one skip and the exact complete suite
     with 1,933 passed and three skipped in 18 minutes 41 seconds. Touched Ruff and
     strict mypy, compileall, CLI help, retained historical replay, and diff checks
     passed. Independent Python and ML/data reviewers reported no remaining P0, P1,
     or P2 finding. Repository-wide debt is 161 Ruff findings and 14 strict-mypy
     findings in the same three untouched intraday dataset files. No provider call,
     artifact regeneration, training, promotion, serving, or locked-test access
     occurred.
   - **Prediction-serving ownership (`completed`).** Implementation commit
     `fc2a32e` consolidates bundle loading, API-facing prediction contracts,
     prediction orchestration, snapshots, outcome-intent registration, investment
     replay, and swing inference under behavior-named `core`, `serving`, and `swing`
     packages. Every removed top-level or `edge_rebuild` import path is prohibited
     without an alias. Serving now rejects duplicate or non-finite JSON, reparse-point
     paths, manifest races, future model generations, stale drift evidence, policy
     identity mismatches, and incomplete requested model pairs. Swing outcomes use the
     bound ten-session triple-barrier policy; intraday outcomes retain their separate
     minute-horizon calibration and managed-outcome contract.

     Verification passed 442 focused tests with two skipped and the exact complete
     suite with 1,995 passed and three skipped. Changed-file Ruff, strict mypy over 46
     source files, compileall, CLI help, and staged diff checks passed. Independent
     architecture and ML/governance reviewers reported no P0, P1, or P2 finding.
     Repository-wide debt is 130 Ruff findings and 14 strict-mypy findings in
     `intraday/datasets/publisher.py`, `intraday/datasets/bar_dataset.py`, and
     `intraday/datasets/dataset_io.py`. No provider call, data regeneration, training,
     promotion, serving process, or locked-test access occurred.
5. **Governance, serving, and command package migration (`pending`).**
   Move readiness, promotion, drift, and outcomes to `governance`; bundle loading,
   prediction services, and API behavior to `serving`; and retain only thin CLI
   adapters in `commands`. Active modules, files, commands, and tests must use behavior
   names rather than chronological labels such as `v3` or checkpoint labels. Delete
   `v3` and `edge_rebuild` only after every implementation and consumer has migrated
   and a repository scan finds zero imports of either namespace.
   - **Outcome and drift governance ownership (`completed`).** Implementation
     commit `880f2a8` moves prediction selection to `modeling`, horizon-specific
     maturation to `swing/evaluation` and `intraday/evaluation`, and outcome,
     performance, feature-drift, and drift-policy ownership to `governance`.
     Removed top-level and duplicate swing policy modules have no aliases and are
     guarded against reintroduction. Outcome observations and matured economics are
     rebound to immutable intents on every read; swing and intraday maturation require
     exact daily and one-minute bars; execution economics remain separate from label
     economics; feature drift is bound to the promoted model's complete feature-name
     set and reference profile; and serving pins the approved drift-policy hash.
     Independent ML/governance and production-Python reviewers found seven P1 issues;
     all were fixed and both reviewers confirmed closure. Verification passed 339
     focused tests and the exact complete suite with 2,033 passed and three skipped.
     Changed-file Ruff, strict mypy over 30 source files, compileall, diff checks, and
     the final no-Python-process check passed. No provider call, data regeneration,
     training, promotion, serving process, or locked-test access occurred.
6. **Repository-wide static quality (`pending`).**
   Resolve all configured repository-wide Ruff and strict mypy findings, remove only
   reference-proven scratch or placeholder artifacts, and add architecture guards that
   prevent duplicate production namespaces from returning. The measured baseline after
   `1829fce` is 168 Ruff findings and 14 strict mypy findings across three files; none is
   in the files changed by `1829fce`. These remain repository debt, not accepted passes.
7. **Full verification and closure (`pending`).**
   Run focused tests after each task, then repository-wide Ruff, strict mypy, the full
   test suite under the configured writable runtime directory, `git diff --check`, and
   a process/memory check. Update the handoff with measured evidence only.

The paused next structural checkpoint is repository-wide static quality. Resolve the
current configured Ruff and strict-mypy debt without changing features, labels,
policies, datasets, model state, or artifact identities. The 2026-09-07 repository-wide
Ruff scan reports 106 errors; the last verified strict-mypy baseline remains 14 findings
in three intraday dataset files and must be remeasured before editing. Use independent
reviewers, one constrained Python process at a time, and close every reviewer and test
worker after the checkpoint.

The clean model-research checkpoint is implementation commit `880f2a8` plus its
documentation closure. The user's 2026-09-07 request supersedes the four-family next
step: follow the Long-Only Swing Research And Implementation Plan above. Keep causal
source authorities, chronological validation, costs and source integrity; do not
promote merely because AUC reaches 0.60. Intraday and unrelated cleanup remain paused.

Rollback is the last pushed task commit. A task is not accepted until the same senior
reviewer has inspected its bounded diff and all supported P0/P1 findings are fixed.

### Package Dependency Direction

- `core` is domain-neutral and cannot import another project package.
- `sources` and `evidence` may import `core`; immutable evidence does not import a
  provider transport.
- `universe` may import `core`, `evidence`, and `sources`.
- `catalysts` may import `core`, `evidence`, `sources`, and `universe`.
- `modeling` may import `core` and `evidence`; it cannot import a trading horizon.
- `swing` and `intraday` may import the lower layers above but cannot import each other.
- `governance` may import completed horizon contracts and lower layers.
- `serving` may import governance and completed horizon contracts; governance cannot
  import serving.
- `commands` and `research` are outer adapters. Production packages cannot import
  either, and production code cannot import tests.

An AST-based architecture test must enforce the allowed top-level production package
set, required swing/intraday subpackages, the dependency direction above, forbidden
chronology/checkpoint names, zero imports of removed namespaces, and no unexpected
top-level production modules. Import and CLI smoke tests must pass after final deletion.

## Historical Structural Refactor Evidence

- Original `swing_features.py` decoupled into `swing_pipeline_steps.py`, `swing_filters.py`, and `swing_catalyst_features.py`.
- Original `swing_training.py` orchestrated and pruned into `training/data_io.py`, `training/lgbm_models.py`, `training/swing_evaluation.py`, and `training/swing_types.py`.
- Maintained frozen contracts, mathematically exact baseline logic, strict memory budgets, and passing tests.
- Implementation commit `8e9cff6` ensures 100% strict `mypy` and `ruff` compliance with fully updated lineage tests.
