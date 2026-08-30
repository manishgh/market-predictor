# Active Edge Rebuild Plan

Status: active

Last updated: 2026-08-30

Repository: `C:\project\market-predictor`

Branch: `er-intraday-refactoring`

This is the only active execution plan. Exact artifact state is recorded in
`docs/reviews/active_edge_rebuild_handoff.md`; statistical rules are defined in
`docs/model_training_validation_protocol.md`.

## Objective And Boundary

Build causal prediction intelligence for:

- **Swing:** ten-session stock direction, managed return, and excess return against
  SPY, QQQ, and the point-in-time sector benchmark.
- **Intraday:** thirty-minute managed outcome from completed intraday evidence.

This repository produces predictions, abstentions, explanations, benchmark
comparisons, and matured outcomes. It does not produce alerts, orders, positions,
portfolio risk, or execution instructions. `trading_flow` may consume only a promoted,
versioned prediction API.

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
| Prior swing candidates | rejected evidence only | no candidate passed both temporal and unseen-security economic gates; locked test unopened |
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

## Active Four-Model Improvement Program

The four governed model families are swing baseline, swing event-driven, intraday
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

   Completion evidence: implementation commit `8a76ec1`; immutable five-minute
   projection `data/canonical/edge_rebuild_selected_session_5m_bar_only_causal_20260814_v2`
   with 43,226 selected stock-sessions, 3,364,335 rows, 43,132 complete pairs, 94
   incomplete pairs retained as coverage, and no provider download; immutable dataset
   `data/features/edge_rebuild_intraday_bar_only_causal_20260814_v1` with 794 sessions,
   501 tickers, 3,095,688 rows, and 1,365,015 eligible rows. Dataset request SHA-256 is
   `83820269d80019a46754aa451c1f1e13773995a889a51e605595511315af4bb2` and
   transformation SHA-256 is
   `0da898cc6fd3c1e933406ce07f24de197fc1fa34c4a909c4b9c4a28e2e96f3f6`.
   The reproducible audit report at
   `data/reports/edge_rebuild_intraday_bar_only_causal_20260814_v1_audit.json` is
   bound to the dataset manifest, authority, session inventory, projection manifest,
   projection authority, and projection inventory. It reports zero duplicate decisions,
   causal-cutoff violations, label-availability violations, eligible ATR violations,
   feature-hash violations, and prohibited features. Aggregate publication peak upper
   bound was 2.218 GiB. Verification closed with 1,293 tests passed, 2 skipped, tracked
   Ruff clean, strict mypy clean across 226 source files, compileall clean, and two
   independent re-reviewers reporting no remaining medium-or-higher findings.
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


## Active Structural Repair Checkpoint

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
4. **Swing and intraday package migration (`in progress`).**
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
   - **Intraday causal volume-bar dataset ownership (`next`).** Move
     `edge_rebuild/volume_bars.py` byte-for-byte to
     `intraday/datasets/volume_bars.py`, rename its test descriptively, and update the
     three intraday dataset consumers and transformation identity directly. Before
     changing `VolumeBarBuildResult` ownership, scan retained manifests and serialized
     artifacts for the old owner. Freeze source, column, representative output,
     transformation, pickle, causality, isolation, threshold, remainder, eligibility,
     and memory behavior. No aliases, schema changes, artifact rewrites, or adjacent
     history/feature/training migration. Rollback anchor: `61f6f4c`.
5. **Governance, serving, and command package migration (`pending`).**
   Move readiness, promotion, drift, and outcomes to `governance`; bundle loading,
   prediction services, and API behavior to `serving`; and retain only thin CLI
   adapters in `commands`. Active modules, files, commands, and tests must use behavior
   names rather than chronological labels such as `v3` or checkpoint labels. Delete
   `v3` and `edge_rebuild` only after every implementation and consumer has migrated
   and a repository scan finds zero imports of either namespace.
6. **Repository-wide static quality (`pending`).**
   Resolve all configured repository-wide Ruff and strict mypy findings, remove only
   reference-proven scratch or placeholder artifacts, and add architecture guards that
   prevent duplicate production namespaces from returning. The measured baseline after
   `8d42d26` is 198 Ruff findings (107 import-order, 61 import-placement, 20 unused
   imports, and 10 unused redefinitions) and 348 strict mypy findings across 61 files;
   these are existing repository debt, not accepted passes.
7. **Full verification and closure (`pending`).**
   Run focused tests after each task, then repository-wide Ruff, strict mypy, the full
   test suite under the configured writable runtime directory, `git diff --check`, and
   a process/memory check. Update the handoff with measured evidence only.

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
