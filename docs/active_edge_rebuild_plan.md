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

Create new, semantically honest setup populations that demonstrate repeatable gross,
net, and benchmark-relative economics before learned ranking. Then train bounded
specialists, add causal catalyst confirmation, and resume KS5 only if at least one
specialist passes independently.

This plan does not promise a profitable model. Its outcome may be a reproducible
rejection. Gates may not be weakened to manufacture a pass.

## 3. Non-Negotiable Design Changes

1. **Setup before estimator.** A deterministic setup must pass economic admission
   before model-family comparison.
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
| ER1A | in_progress | Complete targeted intraday history and re-audit readiness | PIT inventory and the two-tier acquisition plan are complete; collect the 5-minute discovery history, derive setups, collect selective 1-minute paths, and republish ER1 |
| ER2 | pending | Freeze new strategy contracts and bounded experiment budget | New IDs, setup eligibility, entry/exit/labels, design window, folds, costs, features, abstention, and retirement rules are immutable and tested |
| ER3 | pending | Build deterministic setup populations and exact labels | Each setup replays from immutable bars; gross/net/benchmark economics and sample sufficiency are published before ML |
| ER4 | pending | Complete causal catalyst confirmation evidence | Direct/business/sector/global relations and event timing reconcile; technical-only, catalyst-only, and confirmation-overlay rows are identical and auditable |
| ER5 | pending | Train bounded strategy specialists | Only ER3-admitted populations are trained; deterministic/logistic/HGB comparisons and ablations complete under frozen folds |
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
  `4c5e0c71db9337ab00f03bea47ad52b8d6745d96ed572aad1558035043023dc5`.
- Manifest SHA-256:
  `5cfb87aa7316aeaa5a635e7a8c3c4a78b8b12b9c7f6c63716118847daa95c885`.
- 804 historical sessions from 2021-04-27 through 2024-07-08.
- 570 active historical tickers and 404,711 point-in-time ticker-sessions.
- 4,020 bounded, resumable Alpaca SIP/all five-minute request units.
- Maximum 32,289,798 five-minute rows; each request is capped at 10,000
  expected rows.
- One-minute requests are not present in this plan. They are generated only
  after causal setup extraction.
- Peak planning RSS was 0.658 GiB under the 4 GiB hard limit.
- No network request, model fitting, or model artifact was produced.

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
- four purged folds and deterministic unseen-ticker assignment;
- cost/adverse-fill stress;
- no more than six learned candidates, two feature profiles, and two selection policies;
- one retirement rule and no shadow retry.

## 8. ER3: Setup Economic Admission

ER3 evaluates deterministic setups before fitting any model. A strategy may proceed to
ER5 only when both walk-forward and unseen-ticker evidence satisfy:

- required independent rows/sessions and every swing overlap phase;
- positive average gross, net, SPY-excess, and sector-excess return;
- positive session-block 95% lower bounds for net and SPY excess;
- profit factor at least 1.05;
- maximum drawdown no greater than 20%;
- positive economics under the frozen cost/adverse-fill stress;
- no result dependent on one ticker, one sector, one regime, or one session segment.

If a deterministic population fails, retire that semantic version. Do not search the
failed holdout for a better threshold or cohort.

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
