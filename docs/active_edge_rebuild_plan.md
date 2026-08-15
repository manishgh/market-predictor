# Active Edge Rebuild Plan

Status: active

Last updated: 2026-08-13

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
| A3 | Build catalyst-driven swing specialists | Completed; no candidate passed |
| A4 | Build the technical intraday baseline | Completed; both A4.4 hypotheses rejected |
| A5 | Build catalyst-driven intraday specialists | A5.1 complete and blocked; prospective broker-action authority in progress |
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
   `no_candidate` evidence. Continuation's best audited seen/unseen positive-return
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

2. **Prospective broker-action observation authority (`in_progress`).** Establish the
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
     explicit successful/empty/failed coverage, and a hash-bound membership parent.
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
