# Active Edge Rebuild Execution Plan

Status: active
Date opened: 2026-07-28
Repository: `C:\project\market-predictor`
Branch: `r3-lineage`

This is the only active execution plan. The dated known-strategy, five-year-data,
primary-V2, and remediation plans remain historical design and evidence. This plan
supersedes their immediate execution queues but does not weaken their causality,
lineage, cost, validation, promotion, memory, or repository-boundary contracts.

The companion continuation document is
`docs/reviews/active_edge_rebuild_handoff.md`.

The dual-horizon training and evaluation method is governed by
`docs/model_training_validation_protocol.md`, aligned to the retained source PDF
at
`docs/references/comprehensive_quantitative_trading_model_implementation_plan_intraday_and_swing.pdf`.

## 1. Problem

KS0 through KS4 are complete. Their implementations are valid, but every evaluated
directional specialist was rejected or data-blocked. Primary V2 then proved that more
complex return, quantile, and competing-risk estimators do not rescue the unchanged
Cross-Sectional Momentum or VWAP Reversion setup populations.

The failure is economic:

- Intraday VWAP Reversion has negative average gross return before the exact 10 bps
  round-trip cost in both validation scopes.
- Swing Cross-Sectional Momentum has positive gross movement, but approximately 20 bps
  of stamped costs consume the edge; worst-phase net and SPY-relative evidence fail.
- No predeclared one-dimensional cohort passes the required phases in both temporal and
  unseen-ticker validation.
- The millions of raw bars collapse to hundreds of independent market sessions. Adding
  correlated rows or another estimator does not increase independent evidence.

KS5 remains blocked. Distributional models may consume only an independently passed
specialist. No rejected V1 or V2 artifact may be used as a KS5 input.

## 2. Objective

Create semantically honest candidate populations with exact labels and sufficient
independent evidence. Publish their deterministic economics as a baseline, then
train bounded specialists under the governed temporal protocol. Add causal
catalyst confirmation and resume KS5 only if at least one learned specialist
passes independently.

This plan does not promise a profitable model. Its outcome may be a reproducible
rejection. Gates may not be weakened to manufacture a pass.

## 3. Non-Negotiable Design Changes

1. **Population before estimator.** Eligibility, timing, labels, costs, and split
   manifests must pass causal readiness before model-family comparison. A failed
   deterministic score rejects that score; it does not veto a distinct
   preregistered learned relationship.
2. **Independent time matters.** Intraday readiness is measured in causally complete
   sessions and regimes, not bar count. Swing overlapping labels remain phase-separated.
3. **No forced trades.** Abstention is the default. A top-k cap limits qualifying
   opportunities; it never forces selection when expected net edge is insufficient.
4. **New semantics get new IDs.** Failed VWAP Reversion and five-session raw momentum
   are not silently redefined.
5. **Benchmark-relative targets are primary.** Net return, SPY excess, and sector excess
   use the identical executable interval.
6. **Catalysts are causal.** Ticker relevance, publication/first-observed availability,
   source coverage, duplicates, and candle assignment must verify. Global or sector
   events cannot be falsely attached to a ticker.
7. **One-use evaluation.** Design, model development, temporal validation,
   unseen-ticker validation, and prospective shadow evidence remain distinct.
8. **Sequential resource use.** One heavy process at a time and peak RSS below 4 GiB.

## 4. Primary Setup Hypotheses

These names define the planned behavior. ER2 must freeze exact thresholds, timing,
labels, and costs before outcome evaluation.

### Intraday

`INTRADAY.VWAP_EXHAUSTION_REVERSAL.30M.V1`

Long-only hypothesis: a liquidity-qualified downside extension away from session VWAP
may reverse only after causal exhaustion and price-reclaim confirmation. The setup must
distinguish opening, midday, and late-session behavior; broad unconditional VWAP
distance is prohibited.

Candidate decision-time evidence:

- VWAP distance in ATR units and standardized short-horizon shock;
- completed-bar reversal/reclaim confirmation;
- volume burst followed by exhaustion or failed continuation;
- spread/liquidity and observed one-minute-path readiness;
- SPY/QQQ/sector direction and relative strength;
- catalyst confirmation or contradiction as a separate overlay initially.

Entry remains the next observed one-minute open after the completed confirmation bar.
Exit is target first, stop first, or the exact thirtieth regular-session minute.

### Swing

`SWING.SECTOR_RESIDUAL_MOMENTUM.10D.V1`

Long-only hypothesis: stock-specific medium-term strength remaining after market and
sector removal may continue after a controlled pullback and causal trend-reclaim
confirmation. A ten-session horizon is tested to reduce the ratio of one round-trip
cost to expected movement; longer holding is not assumed to improve economics.

Candidate decision-time evidence:

- 20/60-session residual strength versus SPY and point-in-time sector;
- positive long-term trend and slope with complete daily warm-up;
- bounded pullback that preserves the trend;
- price/volume reclaim confirmation;
- liquidity, capacity, volatility, and exact cost evidence;
- causal catalyst confirmation or contradiction kept separately at first.

Entry is the next exact exchange-session open. Exit and any target/stop/timeout path
must be frozen in ER2 and compared with the same SPY and sector interval.

## 5. Checkpoint Sequence

Only one checkpoint may be `in_progress`.

| Step | Status | Purpose | Exit |
| --- | --- | --- | --- |
| ER0 | completed | Establish this active plan and companion handoff | Closed by implementation commit `8c28df9`; both documents and repository guidance are pushed |
| ER1 | completed | Audit effective independent history and causal data readiness | Closed by implementation commits `5ffa3d3`, `d9d93c8`, and `7b0ce6d`; immutable audit request `f80f70ae299bd5e5a6aeae6aeaa503ef4775573696b7d88856d539dbd1355080` reports one ER2 blocker |
| ER1A | in_progress | Complete targeted history and re-audit readiness | Intraday transport and canonical five-minute corpus are complete. Swing acquisition scope is frozen, but all 83 retained official membership sources fail their declared byte hashes; reacquire immutable official evidence before bars |
| ER2 | pending | Freeze new strategy contracts and bounded experiment budget | New IDs, setup eligibility, entry/exit/labels, design window, folds, costs, features, abstention, and retirement rules are immutable and tested |
| ER3 | pending | Build candidate populations, exact labels, and deterministic baselines | Each population replays from immutable bars; readiness, gross/net/benchmark baseline economics, and sample sufficiency are published before ML |
| ER4 | pending | Complete causal catalyst confirmation evidence | Direct/business/sector/global relations and event timing reconcile; technical-only, catalyst-only, and confirmation-overlay rows are identical and auditable; the deprecated V1 relevance path is retired per section 9A |
| ER5 | pending | Train bounded strategy specialists | Only ER3-ready populations are trained; deterministic/logistic/boosted-tree/ranker comparisons and ablations complete under frozen folds |
| ER6 | pending | Resume KS5 and KS6 conditionally | Quantiles, competing risks, and volatility sidecars run only for an independently passed ER5 specialist and must add out-of-sample economic value |
| ER7 | pending | Prospective shadow, promotion, API, and TradingFlow boundary | One-use untouched shadow passes; signed atomic serving bundle exposes predictions only; TradingFlow retains alerts and execution |

## 6. ER1: Independent Data Readiness Audit

ER1 is the immediate next implementation checkpoint. It performs no training.

### Deliverables

- Per-strategy session calendar and first/last usable decision time.
- Raw rows, eligible setup opportunities, unique decision groups, unique tickers, and
  effective/session-block sample size.
- Swing non-overlapping phase counts for the proposed ten-session horizon.
- Intraday session coverage by year, session segment, market regime, sector,
  market-cap, liquidity, and feed.
- Exact SIP/consolidated-feed and adjustment identity.
- Gross/net cost distribution and adverse-fill stress availability.
- Catalyst source-coverage, first-observed, relation, sentiment, and decision-join
  readiness reported separately from technical readiness.
- A list of reusable existing artifacts and an exact acquisition plan only for missing
  evidence.

### Minimum admission to ER2

- Target five years of US trading history.
- At least 750 causally complete intraday sessions before any learned intraday
  specialist is allowed; four purged folds must each retain at least 60 test sessions.
- Swing must retain at least 1,000 valid sessions and enough decisions in every
  non-overlapping ten-session phase for the later 60-session evidence gate.
- Every volume-dependent feature must verify consolidated SIP coverage.
- Costs, benchmark intervals, ticker identity, universe membership, and availability
  timestamps must be reproducible.

Failure does not trigger immediate bulk downloading. ER1 first proves which evidence is
actually missing and whether existing Alpaca data can be reused.

### ER1 Result

ER1 is complete. The immutable audit is
`data/research/edge_rebuild_readiness_er1_20260728` with request SHA-256
`f80f70ae299bd5e5a6aeae6aeaa503ef4775573696b7d88856d539dbd1355080`.

- Swing has 1,254 causally usable SIP sessions, 583 tickers, 610,818 usable
  technical rows, and 125 ten-session effective blocks. Every one of the ten
  non-overlapping phase-capacity checks has at least 125 sessions and passes.
- Intraday has 476 causally usable sessions, 546 tickers, and 69,301 usable
  VWAP-reversion proxy rows. Each provisional four-way chronological test chunk
  has 119 sessions, but the frozen total-history gate still fails by at least 274
  sessions.
- Point-in-time membership, SIP feed, `all` adjustment, exact stamped costs, and
  matching SPY/sector proxy intervals verify for both horizons.
- The existing one-minute coverage audit verifies 4,469,565 requirements at an
  88.36% exact-minute rate. Sparse provider bars remain causal only under the
  frozen no-imputation policy with observed trigger, entry, benchmark, and exit
  bars.
- Alpaca catalyst history has 564,916 source events and 489,088
  training-eligible direct-issuer events. Provider publication time is usable as
  a research proxy but is not prospective first-observed promotion evidence.
  Business-exposure, global-context, and intraday decision joins remain ER4 work.
- No data was downloaded, no model was fitted, and no model artifact was created.
- Peak audit RSS was 0.935 GiB under the 4 GiB hard limit.

## 6A. ER1A: Targeted Intraday History Completion

ER1A is the only active checkpoint. It does not train a model.

1. Reuse the verified five-year point-in-time S&P 500 universe and benchmark
   identity. A current static ticker list is prohibited because it would
   introduce survivor bias.
2. Acquire full-universe Alpaca SIP five-minute bars with `adjustment=all` for
   causal feature construction and setup discovery. Full-universe one-minute
   collection is prohibited because it adds storage and transport cost without
   improving the five-minute setup clock.
3. After causal setups are extracted, acquire only their exact SIP one-minute
   trigger, entry, benchmark, stop/target, and timeout paths. Missing provider
   trades remain no-trade observations and may not be imputed.
4. Acquire enough earlier sessions to produce at least 750 causally usable
   sessions; the research target remains approximately 1,250 sessions.
5. Rebuild the causal intraday technical/setup source from the expanded history.
6. Re-run the immutable ER1 audit. ER2 may start only when the audit reports
   `ready_for_ER2`, all four fold-capacity checks pass, and no membership,
   benchmark, feed, adjustment, or availability blocker remains.

The download scope must be derived from verified historical membership before
network collection begins. Memory remains below 4 GiB and collection/training jobs
remain sequential.

### ER1A Inventory And Frozen Plan

The local inventory found no reusable intraday history before July 2024. It did
verify the research-only five-year point-in-time universe at
`data/universe/sp500_point_in_time_20190709_20260708_v3.parquet`: 659 membership
intervals across 658 historical tickers, official S&P change evidence, reviewed
security transitions, and no overlapping intervals. Historical membership
availability remains a provider-publication proxy, so this source may support
retrospective research but not prospective promotion.

The immutable ER1A plan is
`data/research/edge_rebuild_intraday_history_plan_er1a_20260728`.

- Plan fingerprint:
  `96688b25df7abb24d0317649bde87bf29d9497f799d968801b8f5995bb7cb285`.
- Manifest SHA-256:
  `899a83a33032ab5878779d9feee67adfdabcf0f69f6d5cd0cb39ee768f7b159a`.
- 804 historical sessions from 2021-04-27 through 2024-07-08.
- 570 active historical tickers and 404,711 point-in-time ticker-sessions.
- 4,020 bounded, resumable Alpaca SIP/all five-minute request units.
- Maximum 32,289,798 five-minute rows; each request is capped at 10,000
  expected rows.
- One-minute requests are not present in this plan. They are generated only
  after causal setup extraction.
- Peak planning RSS was 0.663 GiB under the 4 GiB hard limit.
- No network request, model fitting, or model artifact was produced.

The ER1A collector is resumable at unit granularity. It permits two requests in
flight, caps each Alpaca request at the provider-enforced 50 symbols, stops
scheduling after five unit failures, stores compressed raw provider pages and
rate-limit headers, and publishes complete authority only after all units
verify. An operational batch limit may pause a run without changing collection
identity.

Live transport validation on 2026-07-28 proved:

- Alpaca premium SIP access is active.
- The endpoint reports a 10,000 request/hour limit.
- The initial 125-symbol assumption failed closed before transport expansion;
  the invalid non-authoritative smoke output was removed and the immutable plan
  was regenerated with the verified 50-symbol limit.
- Five corrected units returned 16,871 canonical five-minute bars with zero
  failures, 17-50 symbols per unit, one or two provider pages, and 0.294 GiB
  peak RSS.
- Sparse provider observations are retained as observed. No bar is filled or
  synthesized.

The complete transport is
`data/raw/edge_rebuild_intraday_history_er1a_20260728`.

- Collection request SHA-256:
  `5cf109183d4128310be7ad8d9353d3d7b568b3c531ad1458f27be939cdd6c377`.
- Collection manifest SHA-256:
  `a037802034724bfe54ed1dba0af8320514164c5002bcb35d451061e4daefcefb`.
- Complete authority SHA-256:
  `a474c622c10e7dd9ea23b9ac23b5847e248f5cde9305a1ca3566e949f5c69852`.
- 8,844/8,844 units completed with zero failures; five smoke units were
  resumed without another network request.
- 32,033,151 canonical SIP/all five-minute bars across 583 observed symbols.
- Peak collection RSS was 0.293 GiB.
- Full replay verified every registered Parquet hash and matching unit
  sidecar. The collection remains `model_data_ready=false` until
  materialization, setup extraction, selective one-minute labels, and ER1
  re-audit complete.

### ER1A Merge Scope And Measured Universe Gaps

The materialized corpus is the union of two sources with different schemas
and different universe semantics. Both facts were measured before design.

| | ER1A collection | Existing 730-day corpus |
| --- | --- | --- |
| Range | 2021-04-27 to 2024-07-08 | 2024-07-09 to 2026-07-08 |
| Sessions | 804 | 501 |
| Schema | `market_data.v1` | `ohlcv.v1` |
| Availability columns | `bar_end_utc`, `available_at_utc` present | neither present |
| Session scope | regular session only | 04:00-20:00 ET |
| Universe | point-in-time per session | current static S&P 500 list |

The ranges do not overlap. 804 + 501 - 30 warm-up sessions is 1,275 usable
sessions, above the 1,250 research target and the 750 gate.

Two consequences are frozen here:

1. **Availability is derived, not preserved, for the 730-day corpus.** That
   store has no `bar_end_utc` or `available_at_utc`. Both are derived through
   the shared `canonicalize_bars` path under the identical frozen
   `market_interval_close` policy and 60-second finalization delay, so one
   definition covers both eras. Evidence documents must say derived.
2. **Point-in-time membership is applied to both eras.** Measured over the
   501 later sessions: 21,809 ticker-sessions (7.97%) name a ticker that was
   not an index member on that session, and 271 ticker-sessions (0.11%, all
   `PARA`) are members with no bars. The first is look-ahead contamination and
   is filtered out. The second cannot be filtered away, only measured, and is
   published as a per-session coverage gap for the ER1 re-audit to judge. No
   new download closes it.

## 6B. ER1B: Extended-Session Context Layer

ER1B is a sub-step of ER1A, not a separate checkpoint; ER1A remains the only
`in_progress` step. It does not train a model and does not build features.

The ER1A corpus is regular-session only. Overnight gap, pre-market
return/range/volume, relative pre-market volume, and post-close price
reaction cannot be derived from it at all. ER1B acquires that evidence as a
**separate causal layer**, joined at each regular-session decision time.

Frozen rules:

1. Extended bars live in their own corpus and are never merged into
   regular-session indicator inputs. Measured pre-market density across the
   existing corpus ranges from 1.2 bars per session per symbol (`MCO`) to 31
   (`VZ`), so a shared VWAP, EMA, ATR, or relative-volume denominator would
   silently reweight the cross-section by liquidity.
2. Scope is the 804 ER1A sessions only. The 730-day corpus already carries
   verified SIP/`all` 04:00-20:00 ET bars for its own 501 sessions, so no
   verified row is re-fetched and the 32,033,151 regular-session bars are
   untouched.
3. Windows are exact exchange clock times: pre-market is 04:00 ET to the
   session open, post-market is the session close to 20:00 ET. Two separate
   request windows per symbol chunk mean no request can reach into the
   regular session, so no duplicate can conflict with the frozen collection.
4. The session set, cross-section, and chunking are inherited from the frozen
   ER1A plan rather than re-derived. ER1B binds to that plan fingerprint and
   to the completed ER1A transport, so the two layers cannot describe
   different sessions or a different universe.
5. One-minute bars remain reserved for selected executable entry and exit
   paths. Missing provider trades stay no-trade observations.
6. Feature construction is out of scope. The context features are ER2 freeze
   inputs, built in ER3.

### ER1B Completed Transport

At the confirmed 3.25-year intraday window the context layer covers
2023-04-10 through 2024-07-08.

- Plan: `data/research/edge_rebuild_extended_session_context_plan_er1b_20260730`
- Collection: `data/raw/edge_rebuild_extended_session_context_er1b_20260730`
- 313 sessions, 528 historical tickers, 157,443 point-in-time ticker-sessions.
- 3,443 pre-market and 3,443 post-market units; 6,886/6,886 completed with zero
  failures, against 17,688 for the full regular-session range.
- 1,925,863 canonical SIP/`all` five-minute bars across 541 observed symbols.
- Peak collection RSS 0.284 GiB. Zero zero-volume bars.
- Authority replay verified every unit hash and matching sidecar.
- Zero regular-session bars: layer isolation holds at the transport boundary.

**Segment classification must use exchange session bounds, never clock times.**
On 2024-07-03 the session closed at 13:00 ET, so 396 genuine post-market bars
fall inside a naive 09:30-16:00 band. Any code that separates regular from
extended bars by fixed clock time will misclassify every early-close session.
The planner uses `calendar.session_close`; materialization and feature code must
do the same.

### ER1B Frozen Plan

The immutable ER1B plan is
`data/research/edge_rebuild_extended_session_context_plan_er1b_20260729`.

- Plan fingerprint:
  `89c91d178ed1095cc33047508c270a817d716e1c594bfe0940064d2de770a250`.
- Manifest SHA-256:
  `a2a5675bed96aa312c94da4477aab1e5e7b9570935e8f4f2acda5b47be9edea0`.
- Policy SHA-256:
  `2fb6118c448438c5ffe59a1cb3319b39f4e80bf47bca5c77df55948e204700d6`.
- 804 sessions from 2021-04-27 through 2024-07-08, identical to ER1A.
- 570 historical tickers and 404,711 point-in-time ticker-sessions,
  identical to ER1A.
- 8,844 pre-market and 8,844 post-market units, exactly two per ER1A unit.
- Row budget ceiling 47,421,498; the expected yield is far lower because
  extended-hours bars are sparse and are never imputed.
- Peak planning RSS 0.303 GiB. No network request, model fitting, or model
  artifact was produced.

Live smoke validation on 2026-07-29 collected six units: 1,617 canonical
bars, 173 observed symbols, zero failures, 0.345 GiB peak RSS. Every row fell
in 04:00-09:25 ET with zero regular-session rows, confirming layer isolation
at the transport boundary; `available_at_utc` minus `bar_end_utc` was exactly
60 seconds on every row. Only 173 of roughly 300 requested symbols returned
any pre-market bar, which is why the no-imputation policy governs this layer.

### ER1A Published Canonical Five-Minute Corpus

The authoritative materialization is
`data/canonical/edge_rebuild_intraday_5m_20260731`.

- Authority state is `complete`; the manifest SHA-256 is
  `f71d25ec1a98d38b75a3175a1508f8529426623857a09f255aeacc7bd19db0e0`.
- 38,586,501 bars cover 814 sessions from 2023-04-10 through 2026-07-08:
  32,506,506 regular, 3,190,687 pre-market, and 2,889,308 post-market.
- 1,104 regular per-symbol files and 573 extended per-symbol files were
  published. The union includes point-in-time index members and the screened
  non-index intraday names.
- All 1,677 registered file hashes replayed exactly. A full Parquet scan proved
  that every regular file contains only regular-session rows and every extended
  file contains only pre-market or post-market rows.
- Corpus integrity reports zero blocking defects. Two isolated sparse
  ticker-sessions remain recorded and tolerated under the frozen
  `0.0001` maximum isolated-defect share.
- Zero symbols were excluded. The materializer would quarantine unprovable
  identities or fabricated observations and refuses when more than 5% of input
  symbols would be lost.
- Peak working-set memory recorded in the immutable manifest was 1.251 GiB,
  below the 4 GiB process ceiling.
- Implementation checkpoints are `39a50bd`, `93d073f`, and `12f5283`.
  Repository verification after publication passed 732 tests, Ruff, strict
  mypy across 190 source files, and compileall.

This closes five-minute transport and materialization, not ER1A. Setup
derivation, selective one-minute executable paths, and the ER1 readiness
re-audit remain before ER2 can start.

### ER1A Causal Technical Relationship Primitives

Implementation commit `3403866` added one shared implementation at
`src/market_predictor/edge_rebuild/technical_relationships.py`.

- RSI divergence compares two strictly confirmed five-bar pivots. A middle bar
  becomes observable as a pivot only after two later bars complete, and a poison
  test proves appending future bars cannot change any earlier output.
- Price/volume agreement uses normalized On-Balance Volume over 20 bars.
- Trend versus range state uses Kaufman's 20-bar Efficiency Ratio. RSI is
  decomposed into trend alignment and range position instead of adding an
  "overbought" threshold that means different things in different regimes.
- Every output is continuous. No buy, sell, overbought, or oversold flag exists.
- Intraday callers group by both ticker and exchange session, and tests prove
  all rolling state restarts at the session boundary.
- The frozen strategy contract hash is
  `f60666809a1c8c9df230b13fb875d224dd271a4393ae764dd49075ef3014dee8`.
- Verification passed 740 tests, Ruff, strict mypy across 191 source files,
  compileall, and `git diff --check`.

These are feature primitives, not a finished training table. The swing feature
builder is their first consumer; no model was trained in this step.

### ER1A Causal Swing Ranking Panel

Implementation commit `fccda19` added the two-stage builder at
`src/market_predictor/edge_rebuild/swing_features.py`.

- Stage one can replay complete security histories in memory-bounded batches.
  It consumes the canonical daily feature history, shared residual-strength
  components, and shared technical relationships. It adds conservative
  triple-barrier outcomes without computing any population-relative value.
- Stage two is deliberately population-wide. Same-session z-scores, centred
  ranks, sector-relative z-scores, and within-sector rank labels would be wrong
  if they were calculated inside security batches, so missing or extra
  securities and mixed feature profiles are refused.
- Feature families are named and stable: momentum, trend, pullback, volume,
  technical relationships, and ticker catalysts. Ticker catalyst aggregates
  retain the upstream canonical event-assignment boundary; global context uses
  the existing explicit canonical inputs.
- Raw news counts are never estimator columns. They may enter the
  same-session transformation, after which only normalized derivatives are
  returned by the estimator schema.
- Cold rows and unresolved barrier paths do not count toward scaling or rank
  label sample size. Appending and poisoning a later session leaves all earlier
  transforms and labels bit-identical.
- Cross-sectional outputs are emitted as a contiguous `float32` block to keep
  the million-row materialization within the 4 GiB process ceiling.

The contract now freezes 1% within-session winsorization and requires z-score,
centred-rank, and sector-relative outputs. Its SHA-256 is
`16709f3686ec737caa206dc1f45a80ea24f61d2dd1d18ded0b78cf978a433e38`.
Verification passed 748 tests with 85 existing warnings, Ruff, strict mypy
across 192 source files, compileall, and `git diff --check`. No Python worker
remained.

Implementation commits `c292b7d` and `b91c5ee` added the resumable materializer
and bounded its complete-population pass by disjoint session years. Stage two
is never split by security: every decision date is transformed against its full
tradable cross-section. The time partition reduced measured peak RSS from a
refused 8.06 GiB all-at-once attempt to 1.7724 GiB.

The authoritative panel is
`data/features/edge_rebuild_swing_panel_20190709_20260708_v1`:

- 883,604 rows, 627 securities, and 1,759 sessions from 2019-07-09 through
  2026-07-08;
- 733,515 feature-eligible rows, 869,242 resolved barrier outcomes, and
  450,074 rank-eligible rows;
- 20 verified stage-one shards and eight immutable yearly final partitions;
- 2,897 zero-volume provider placeholders removed before feature computation;
- request SHA-256
  `815ade0c661f6f35ecb1f9c49d7fd35c6f3e96cf2b02b0f6faaef0288b55252f`;
- final manifest SHA-256
  `a939de80ba7cc76821dde07ad73d0e7edab0af1cbaa7b938816ad20a9dd6b55b`;
- frozen strategy-contract SHA-256
  `16709f3686ec737caa206dc1f45a80ea24f61d2dd1d18ded0b78cf978a433e38`.

Immutable replay verified every partition hash. Verification passed 751 tests
with 85 existing warnings, Ruff, strict mypy across 193 source files,
compileall, and `git diff --check`. No Python worker remained.

Panel materialization is complete. The simple top-decile versus bottom-decile
technical ordering test is a deterministic benchmark. It does not have authority
to prohibit a preregistered learned model from testing a different functional
relationship under the governed temporal protocol.

### ER1A Deterministic Swing Ordering Gate

Implementation commit `de37067` froze the outcome-blind technical composite in
`configs/edge_rebuild_swing_ordering.toml` before reading the published panel's
outcomes. The audit is immutable at
`data/reports/edge_rebuild_swing_ordering_20190709_20260708_v1`.

The gate **failed** over 1,500 scored sessions from 2020-07-02 through
2026-06-23:

- top decile: 73,442 rows, +14.34 bps weighted mean managed return;
- bottom decile: 71,950 rows, +32.84 bps weighted mean managed return;
- session-neutral top-minus-bottom mean: -18.41 bps;
- median session spread: +17.28 bps;
- positive-session share: 53.53%;
- ten-session Newey-West t-statistic: -1.355.

The positive median and majority-positive session count do not rescue this
signal. Adverse tail sessions make the mean economic ordering negative, and the
bottom decile outperforms the top decile overall. The minimum spread and
significance gates failed. This rejects the frozen equal-weight technical
composite and any portfolio whose ranking is produced by that exact score. It
does not establish that every nonlinear or grouped relationship in the same
causal features is absent.

The failed score must not be tuned against its full-sample result. It remains a
comparison baseline. Subsequent learned experiments must be preregistered and
use session-grouped, purged walk-forward development, an independent
unseen-security scope, and one locked temporal test as specified in
`docs/model_training_validation_protocol.md`.

### ER1A PDF-Aligned Training And Validation Decision

The retained PDF calls for purged walk-forward validation, cross-sectional
normalization, triple-barrier labels, point-in-time S&P 500 membership, and
bounded tree-model comparisons. The repository resolves those requirements as
follows:

- a split groups the full point-in-time cross-section by decision session;
- the same security may appear in multiple chronological folds, while a separate
  deterministic holdout measures unseen-security transfer;
- swing uses repeated five-year fit and one-year validation windows followed by
  one locked test year, with at least a ten-session purge/embargo;
- intraday uses session-grouped folds, minute-horizon purging, and an overnight
  embargo;
- one global S&P 500 model is the first learned model, with sector-relative
  features and sector-stratified evaluation; sector specialists require later
  independent evidence;
- the bounded learned sequence includes a barrier classifier and a grouped
  cross-sectional ranker, compared with deterministic and logistic baselines.

The approved swing modeling horizon is seven usable years: five years fit, one
year validation, and one locked test year. The preceding 250 sessions are
warm-up only. The current seven-year panel remains authoritative for causal
feature and panel validation but lacks that pre-fit warm-up and the first 29 fit
sessions. Final training requires only the exact frozen 2018-05-29 through
2019-07-08 gap, not a 9-to-10-calendar-year model horizon.

The ER3 implementation now exposes separate `ready_for_modeling` and
`baseline_economics_passed` decisions. Readiness covers causal integrity,
independent phase/session sufficiency, and representation. Baseline economics
remain fully reported but cannot veto fitting. The prior `admitted` schema and
the unregistered retired attribution script were removed; there is no
compatibility alias that could preserve the superseded behavior.

Implementation commit `84afb07` also binds the unchanged four-page PDF in the
repository, verifies its SHA-256, and strengthens the canonical temporal-split
test so every row from a decision session remains in one fold role. The current
strategy-contract SHA-256 is
`a77ccb4d635cc604b49b9c8bd5277aa281ca36afaf25abbe14c1c44d45b6460a`;
the baseline-economics configuration SHA-256 is
`761c57dd7248560da36620295e86f466c6c3daae04b8389a291c63ab185a31df`.
The earlier contract hash attached to the published seven-year panel remains
that artifact's immutable provenance and is not rewritten.

Verification passed 756 tests with 85 existing warnings, repository-wide Ruff,
strict mypy across 194 source files, compileall, and `git diff --check`. No
Python worker remained.

### ER1C PDF-Aligned Temporal Manifest

Status: complete as a bounded ER1A sub-step. It opened no model outcomes and
performed no training. Implementation commits `7de78a0` and `d72d1c2` are
pushed; the latter freezes the approved seven-year horizon and 5% exclusion
rule.

The canonical manifest freezes one XNYS session-grouped validation year,
1,260 fit sessions per fold, 252 validation sessions per fold, a ten-session
embargo, 250 warm-up sessions, and the untouched 2025-07-01 through 2026-06-30
locked test. It also freezes a 20% `security_id` hash holdout independently of
time. The published seven-year panel is read only for authority, partition
hashes, `session_date_et`, and `decision_group_id`; outcome columns are
prohibited. Missing history produces an immutable acquisition gap rather than
a permissive shorter split.

The immutable audit is
`data/research/edge_rebuild_swing_temporal_manifest_20260731_v1`. Its manifest
SHA-256 is
`8d073839eb31e1baa734c9068c957798f1884e9bb2bbe871d80be0b82affe6cf`.
It proves the required raw target is 2,033 XNYS sessions from 2018-05-29 through
2026-06-30. The current panel supplies 1,754 of those sessions and is missing
exactly 279 contiguous sessions from 2018-05-29 through 2019-07-08. No shorter
split is permitted. Peak working set was 0.285 GiB.

The hash-bound request, fold file, session-assignment file, manifest, authority,
partition-tamper refusal, full-session ownership tests, exact coverage report,
and memory gate all pass. Repository verification passed 764 tests with 85
existing warnings, Ruff, strict mypy across 195 source files, compileall, and
`git diff --check`.

### ER1D Outcome-Blind Swing History Acquisition Plan

Status: implementation complete; acquisition remains blocked by invalid source
provenance. Implementation commit `18bb896` is pushed.

`plan-edge-rebuild-swing-history` verifies the frozen temporal authority,
point-in-time membership sidecar, universe audit, official source archive,
security-transition evidence, and current Alpaca daily collection without
reading outcomes. It publishes immutable request, manifest, and authority files.
It emits Alpaca SIP/all daily-bar units only after historical membership and
source identities are authoritative.

The real audit is
`data/research/edge_rebuild_swing_history_acquisition_20260731_v2`. Its manifest
SHA-256 is
`6a689951035fd8b236cc23ab5530a570158eda7d1c0d09eab8867da4eec6de48`.
The result is `official_source_reacquisition_required`: zero of 83 retained
official S&P release files match their declared SHA-256 identities. The audit
therefore publishes zero Alpaca request units and records
`blocked_until_source_reacquisition`. The complete expected/actual hash pairs
remain in the manifest. Peak working set was 0.230 GiB.

The next operation is not bar collection. Reacquire the official S&P
constituent-change releases into a new immutable byte-hashed archive, rebuild
and verify point-in-time membership through 2018-05-29, then rerun the planner.
Only a `ready_for_daily_bar_collection` result may authorize the exact
2018-05-29 through 2019-07-08 Alpaca units.

Repository verification passed 768 tests with 85 existing warnings, Ruff,
strict mypy across 196 source files, compileall, and `git diff --check`.

### ER1E Official Source Reacquisition And Membership Lineage

Status: `in_progress` within ER1A. Independent provenance and collector reviews
were completed before implementation. They found that the ER1D blocker artifact
is conservative and valid, but its future authorization path is insufficient:
source files, universe audit, and final memberships are not yet joined by one
verified parent-hash chain. No Alpaca bar request may rely on the current
`ready_for_daily_bar_collection` branch.

The frozen source scope is publication discovery from 2018-04-14 through the
2026-07-08 reconstruction cutoff. Reacquiring only the 2018-2019 gap is
insufficient because the backward membership replay depends on the complete
release chain through the cutoff. The 83 retained canonical release URLs are
seed identities, not valid payload evidence; discovery must independently cover
the full window and may add releases missed by the old narrow title matcher.

The source collector must:

- persist exact HTTP response bytes with `write_bytes()` and verify the stored
  SHA-256 after writing;
- record requested and final URL, redirect chain, retrieval time, status,
  content type and encoding, ETag, Last-Modified, byte length, and byte hash;
- retain and hash every archive discovery page, use one-result overlap between
  adjacent pages, and prove pagination crossed both frozen publication bounds;
- reject unexpected domains, generic landing pages, incomplete discovery, and
  response-identity failures. The live provider template itself contains a
  `saved from url` comment, so that marker is not provenance evidence;
- retain parser failures as explicitly unresolved raw evidence. Raw archive
  authority uses state `raw_complete`; event reconstruction must remain blocked
  until `parser_unresolved_releases` is zero and `event_extraction_ready` is true;
- publish an immutable request before collection, per-unit content-addressed
  payload and sidecar, hash-verified resume, partial status, and final authority
  only when all units pass;
- use at most two download workers and stay below the 4 GiB process limit.

Deterministic reconstruction remains offline and reuses the existing parser and
backward interval builder. Its authority must bind raw archive authority,
canonical sorted event hash, cutoff anchor, transition evidence, PIT universe,
and identity-filtered membership by parent hashes. Cutoff membership must replay
to the anchor exactly.

Transition evidence is an independent blocker. The retained provider artifact
begins on 2019-09-13 and the reviewed ledger begins on 2019-11-05. A new
immutable transition request and authority must cover 2018-05-29 through the
cutoff, or explicitly prove that no transition occurred in each uncovered
interval from primary evidence. Absence of a row is not proof of no transition.

Exit gates:

1. raw-byte HTTP transport and resumable source collector pass focused and full
   verification;
2. an independent code/provenance review finds no unresolved critical or high
   issue before the first authoritative network run;
3. the official archive authority covers the full frozen publication window;
4. transition authority covers the full membership interval;
5. offline replay publishes the complete parent-hash chain and exact anchor
   reconciliation;
6. only then may the swing acquisition planner emit exact Alpaca units.

### ER1A Repository And Data Convergence

The 2026-07-31 cleanup removed legacy compatibility narratives and 24.775 GiB
of rejected, reproducible, or superseded local data. Current raw provider
archives and authoritative edge-rebuild inputs were verified present after the
deletion. `data/` now occupies 14.828 GiB. Generated feature directories remain
absent until rebuilt through the current causal panel builder.

The cleanup does not change model evidence or checkpoint status. It prevents a
later command from accidentally selecting an old feature matrix or superseded
canonical corpus. Focused governance and architecture tests passed 17/17; the
documentation and data cleanup is isolated from model code.

## 7. ER2: Frozen Research Contract

ER2 creates a design-only window that is disjoint from evaluation. Thresholds may be
chosen once from domain rationale plus that design window, then become immutable.
Validation and unseen-ticker outcomes may not be inspected while choosing thresholds.

For each strategy freeze:

- eligibility and bounded abstention reasons;
- exact feature cutoff and session segment;
- entry, target, stop, timeout, and horizon;
- gross return, cost, net return, SPY excess, and sector excess labels;
- deterministic comparator;
- no-trade rule and maximum qualifying trades per period;
- the horizon-specific frozen fold count: one full swing validation year and
  four intraday folds, plus deterministic unseen-security assignment;
- whole-security exclusion for unavailable or unverifiable stock data through
  a hard 5% ceiling of the filtered universe; benchmarks and market-wide
  session gaps remain non-excludable;
- cost/adverse-fill stress;
- no more than six learned candidates, two feature profiles, and two selection policies;
- one retirement rule and no shadow retry.

ER2 must also freeze the extended-session context features that ER1B makes
available, before ER3 builds any of them:

- overnight gap from the prior regular-session close to the current open;
- pre-market return, range, and volume;
- relative pre-market volume against its own pre-market baseline, never
  against a regular-session denominator;
- post-close price reaction over the post-market window, named as a
  price/volume quantity and not as an earnings reaction.

Each must declare its exact decision cutoff, its abstention behavior when the
window is empty, and a minimum observed-bar requirement, because extended
windows are sparse and are never imputed. Attributing a post-close move to an
earnings event, and any news-since-prior-close feature, require the
first-observed evidence that ER1 reported as not ready; both remain ER4
catalyst-overlay work and may not enter the estimator before a preregistered
causal ablation.

## 8. ER3: Population Readiness And Deterministic Baseline

ER3 builds each candidate population and reproduces exact labels before fitting a
model. A population may proceed to ER5 only when both temporal-development and
unseen-security scopes satisfy causal readiness:

- required independent rows/sessions and every swing overlap phase;
- complete point-in-time identity, feature availability, benchmark, and label paths;
- exact gross, net, SPY-excess, and sector-excess return reproduction;
- frozen costs and adverse-fill stress;
- sufficient representation across years, sectors, regimes, and session segments;
- no threshold or cohort selected from validation or locked-test outcomes.

ER3 must publish the deterministic comparator's average return, confidence
interval, profit factor, drawdown, concentration, and stressed economics. Negative
baseline economics is evidence against that comparator, but is not by itself a
data-readiness failure and does not prohibit a preregistered classifier or ranker.
ER5 determines whether learned selection adds out-of-sample economic value over
the baseline. If the learned policies fail, retire the semantic version without
searching the locked test for a better threshold or cohort.

## 9. ER4-ER7 Rules

- Catalyst starts as confirmation, veto, explanation, and ranking overlay. It becomes a
  direct estimator feature only after a preregistered causal ablation improves both
  validation scopes.
- ER5 model selection optimizes calibrated cost-adjusted benchmark-relative economics,
  not AUC alone.
- ER6 is skipped when ER5 has no independently passed specialist.
- GARCH/HAR-RV remain volatility/risk sidecars and never become directional authority.
- ER7 requires untouched prospective evidence. Development or retrospective validation
  cannot authorize production actionability.
- Market Predictor emits prediction intelligence only. It does not create alerts,
  orders, positions, or final sizing.

## 8A. Frozen Universe And Eligibility Contracts

Decided 2026-07-29. These replace the earlier five-year intraday research target.

| | Swing | Intraday |
| --- | --- | --- |
| Universe | point-in-time S&P 500 membership | same, then eligibility-filtered |
| Eligibility | index membership is the filter | price > $8, dollar ADV >= $25M, bar continuity >= 95% |
| Approximate size | 503 per session | 554 of 570 historical securities |
| History | seven years from 2019-07 | approximately 3.25 years from 2023-04 |
| Catalysts | required; needs the 2019-07 to 2021-07 news gap filled | already covered |
| Bars | daily | five-minute regular plus five-minute extended context; one-minute only for exact entry and exit paths |

Every intraday filter is evaluated on a trailing window as of each decision
date, so eligibility changes over time and never uses future information.

### Why The Retail Share-Volume Standard Was Rejected

The common day-trading floor of one million shares per day keeps only 386 of 570
securities here and excludes the wrong ones. `CHTR`, `LMT`, `BLK`, `REGN`, and
`NOC` each trade over 250 million dollars per day with 100% bar continuity and
would be excluded solely because a high share price puts the share count under a
million. That threshold is calibrated for a retail universe of 20 to 50 dollar
stocks, not for the S&P 500.

Dollar volume alone is also insufficient. `AZO` shows 227 million dollars of
daily volume and prints in only 83.3% of five-minute buckets; `FICO`, `MTD`,
`NVR`, and `BIO` behave the same way. A high-priced security can carry large
dollar volume and low print frequency, and for a thirty-minute setup the print
frequency determines whether the modelled exit was achievable. Bar continuity is
the only filter that separates them.

The 95% continuity threshold sits in an empty region of the observed
distribution: the fifth percentile is 98.7%, the first percentile is 91.9%, and
the five problem securities fall between 74.4% and 89.7%. Any threshold from 92%
to 98% selects the same set, so the choice is robust rather than tuned.

### Catalyst Coverage And A Non-Stationarity It Creates

Swing catalyst coverage now spans the full seven-year window. The 2019-07-09 to
2021-07-08 gap was collected against the verified universe: 4,066 requested
chunks, 4,047 observed, 19 empty, zero failed, 149,140 rows.

Observed news density differs materially between the two eras:

| Era | Rows | Approximate rows per year |
| --- | ---: | ---: |
| 2019-07 to 2021-07 | 149,140 | 75,000 |
| 2021-07 to 2026-07 | 564,986 | 113,000 |

The earlier period carries roughly two thirds the density. This is provider
coverage growth, not a change in how much news issuers generated, and it makes
raw event counts non-stationary across the sample. A model given a raw
`news_count` would learn that later dates carry more news, which is an artifact
of collection rather than a property of the market.

Every news-volume feature must therefore be normalized within its decision
cross-section, as a cross-sectional rank or as a ratio against a trailing
per-ticker baseline. Raw counts are prohibited as estimator features. Sentiment
means are already relevance-weighted ratios and are unaffected.

### Corporate Actions

Renames are preserved, not excluded. Twenty-five of twenty-six observed alias
chains hand off cleanly, and excluding them would remove `META`, `ELV`, `BALL`,
`CTRA`, and `RTX` from the universe, which is survivorship bias.

For genuine splits the continuing entity retains its own identity and history.
The spun-off entity requires no special rule because a new listing cannot
satisfy the 250-session daily warm-up requirement in its first year and is
therefore already ineligible. The only residual artifact is the unadjusted price
step at the spin-off date, which is handled by excluding that single session
from label generation.

## 8B. Prescribed Methods And Their Sources

This program implements published methods. It does not invent alternatives, and
every method below is named so an implementation can be checked against its
definition rather than against someone's recollection.

| Concern | Method | Source |
| --- | --- | --- |
| Labels | Triple-barrier: whichever of target, stop, or timeout is touched first | López de Prado, *Advances in Financial Machine Learning*, 2018 |
| Validation | Purged k-fold with embargo | López de Prado, 2017 |
| Sampling | Event-based: sample when something happens, not on a fixed clock | López de Prado, 2018 |
| Direction versus size | Meta-labeling, deferred until one strategy passes admission alone | López de Prado, 2017 |
| Intraday selection | Average volume, relative volume, price band, spread | standard day-trading screen practice |
| RSI divergence pivots | Five-bar confirmed pivot plus RSI divergence | Bill Williams, *Trading Chaos*; J. Welles Wilder Jr., *New Concepts in Technical Trading Systems* |
| Price/volume confirmation | On-Balance Volume | Joseph Granville, *Granville's New Key to Stock Market Profits* |
| Trend versus range | Efficiency Ratio | Perry Kaufman, *Trading Systems and Methods* |

References:

- <https://en.wikipedia.org/wiki/Purged_cross-validation>
- <https://en.wikipedia.org/wiki/Meta-Labeling>
- <https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/>
- <https://www.tradingsim.com/blog/how-to-find-the-best-stocks-for-day-trading>
- <https://chartschool.stockcharts.com/table-of-contents/technical-indicators-and-overlays/technical-indicators/relative-volume-rvol>

Published work finds that event-based sampling, triple-barrier labeling, and
meta-labeling together improve strategy performance. The first two are in scope
now; meta-labeling waits, because it adds a second model to validate and nothing
has yet passed admission on its own.

## 8C. Intraday Universe: Two Layers, Not One

Standard practice selects intraday candidates in two stages, and conflating them
was a design error in this program.

**Layer one, could this be traded at all.** Average volume at least one million
shares over twenty sessions, price between eight and five hundred dollars,
prints in at least 95% of five-minute buckets, and operating companies only.

Exchange-traded products are excluded. The setup is a single-name reversal
conditioned on an issuer catalyst and a fund has no issuer; measured article
density is a median of 8 articles for exchange-traded products against 204 for
operating companies. Leveraged single-stock products would additionally
double-count exposure to an underlying already in the universe, and bond funds
are not equities at all.

**Layer two, is it moving today.** Relative volume at least 2.0 against its own
twenty-session average, measured from prior sessions only so selection never
uses information from the session being traded. At most thirty candidates per
session.

Measured on the S&P 500 corpus:

| Filter | Result |
| --- | ---: |
| Average volume at least 1M shares | 400 of 567 symbols |
| Plus 95% bar continuity | 399 of 567 symbols |
| Relative volume at least 1.5 | 10.6% of stock-days, median 29 per session |
| Relative volume at least 2.0 | 3.8% of stock-days, median 10 per session |
| Relative volume at least 3.0 | 1.0% of stock-days, median 3 per session |

An unconditional population is therefore roughly 96% symbols that were not
moving. Training on those teaches nothing and is the most likely reason the V2
intraday setup showed negative average gross return before costs: the failure
was in the population, not the estimator.

**The intraday universe is deliberately not the S&P 500.** The most
intraday-tradable names are frequently high-beta mid caps, recent listings, and
catalyst-driven movers that are not index constituents. Restricting intraday to
index membership excludes exactly the population the strategy needs. Swing
remains index-based, because sector rotation and medium-term relative strength
are index-shaped questions.

Acquisition follows from this. Daily bars are collected for a broad
point-in-time US universe, which is cheap at one bar per symbol per session and
sufficient to compute average volume, relative volume, price, and range.
Five-minute bars are then collected only for the stock-sessions that pass both
layers, which is roughly ten to thirty symbols per session rather than the whole
universe.

## 9A. ER4 Scope: Retire The Deprecated V1 Relevance Path

### Defect

`src/market_predictor/v3/catalysts.py` manufactures an event relevance value from
a text heuristic and then filters on it:

```
relevance = 1.0 + 0.75*title_match + 0.35*(~title_match & text_match)
relevance -= 0.60*generic_headline
relevance += 0.30*(source_family == "sec")
relevance = relevance.clip(lower=0.1)
kept if relevance >= minimum_relevance
```

`O1OverlayConfig` is a `FrozenContract` and binds `minimum_relevance` (0.1-scale
threshold, default 1.25). It does **not** bind the five constants that produce the
value being thresholded. Editing any of them leaves `config.sha256()` unchanged,
so the published O1 evidence is not reproducible from its own recorded identity.
`data/reports/v3_c8_o1_ablation_20260720.json` embeds the config but cannot embed
what the config does not cover. This is a section 3.6 no-deferred-correctness
defect: an identity that does not bind what it claims to bind.

This is a research-only path. Production never reaches it: the canonical path is
`canonical/joins.py:aggregate_event_features`, which consumes relevance and keeps
unknown values as `NaN` rather than inventing them, and accepted catalyst rows use
the point-in-time three-channel contract in `swing/event_attribution.py`.

### Preserved before deletion

These are the exact coefficients used to produce the rejected O1 evidence. They
are recorded here because deleting the code otherwise makes the published result
uninterpretable. Nothing may reuse them.

| Component | Value |
| --- | --- |
| Base relevance | 1.0 |
| Ticker in title | +0.75 |
| Ticker in body only | +0.35 |
| Generic headline | -0.60 |
| SEC source family | +0.30 |
| Floor | 0.1 |
| Accept threshold | `minimum_relevance`, default 1.25 |
| Windows | 2h and 1d |
| Overlay weight / veto penalty | 0.15 / 0.50 |

O1 result being preserved: 322,291 covered prediction rows, 503 covered tickers,
72,818 deduplicated event rows, zero future matches. Walk-forward top-10 excess
moved from -0.05744% to -0.04871%; ticker holdout worsened from -0.06423% to
-0.06690%. Both paired confidence intervals include zero. O1 is rejected.

### Exact deletion set

Verified self-contained. `v3/catalysts.py` symbols are referenced only by the
`audit-v3-o1-overlay` command and its test.

1. Delete `src/market_predictor/v3/catalysts.py`.
2. Delete `tests/test_v3_catalysts.py`.
3. In `src/market_predictor/commands/v3_evaluation.py`, remove the
   `market_predictor.v3.catalysts` import block and the `audit-v3-o1-overlay`
   command only.
4. Remove `audit-v3-o1-overlay` from `tests/fixtures/cli_command_inventory.json`.

Retained deliberately:

- `audit-v3-failure-attribution`, which depends on `v3/diagnostics.py`.
- `audit-v3-ranking`, which depends on `v3/evaluation.py`.
- `data/reports/v3_c8_o1_ablation_20260720.json` and
  `data/reports/v3_c8_o1_readiness_unscored_20260720.json` as immutable
  historical evidence.

### Why deletion rather than binding the coefficients

Section 1 authorizes removal: the system is not deployed and every consumer of
this module is a rejected, closed experiment. Binding the constants into the
contract would spend effort making a rejected experiment reproducible, and would
change the contract hash anyway, so it would not even preserve the identity it
set out to protect. Deletion also removes the live risk that ER4 silently reuses
the deprecated relevance while building the causal catalyst layer, which is the
only forward-looking harm this defect can still cause.

### Separately noted, not in scope here

`MATERIAL_EVENT_TYPES` is defined twice, in `v3/catalysts.py` and in
`catalyst_overlay.py`. Deleting the former leaves one definition. Confirm during
ER4 that the surviving definition is the intended one.

## 10. Per-Step Documentation Protocol

After every ER step:

1. Commit and push the verified implementation/evidence.
2. Update the status table so exactly one next step is `in_progress`.
3. Record actual evidence paths, hashes, memory, test counts, and implementation commit.
4. Rewrite `docs/reviews/active_edge_rebuild_handoff.md` with the exact next action.
5. Commit and push the documentation closure.

The next LLM must be able to continue from those two files without reading chat history.

## 11. Completed Evidence

### ER0

- Implementation commit: `8c28df9`
- Remote ref: `origin/r3-lineage`
- Deliverables: this plan, the companion handoff, mandatory repository guidance, and
  README links.
- Verification: 19 focused governance/dependency tests passed; repository-wide Ruff,
  strict mypy across 172 source files, compileall, and `git diff --check` passed.
- Next checkpoint: ER1, marked `in_progress`; implementation has not started.

### ER1

- Contract commit: `5ffa3d3`
- Audit implementation commits: `d9d93c8`, `7b0ce6d`
- Remote ref: `origin/r3-lineage`
- Evidence directory:
  `data/research/edge_rebuild_readiness_er1_20260728`
- Request SHA-256:
  `f80f70ae299bd5e5a6aeae6aeaa503ef4775573696b7d88856d539dbd1355080`
- Result: `blocked_pending_targeted_acquisition`; one ER2 blocker,
  `intraday_session_history_below_gate`.
- Verification: 585 tests passed with 85 existing warnings; repository-wide
  Ruff, strict mypy across 176 source files, compileall, immutable replay, and
  `git diff --check` passed.
- Peak RSS: 0.935 GiB.
- Next checkpoint: ER1A, marked `in_progress`; ER2 remains pending.
