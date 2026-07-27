# KS4 Intraday Specialist Contract

Date: 2026-07-27

Status: frozen development contract; implementation and real-data replay follow
this checkpoint.

## Objective

KS4 determines whether seven named, long-only intraday behaviors have
repeatable cost-adjusted edge. It does not produce alerts, orders, portfolio
sizing, or a serving release. Each strategy is evaluated independently and may
be rejected without affecting the others.

## Strategies And Session Isolation

| Strategy | Horizon | Frozen session segment |
|---|---:|---|
| Opening Range Breakout | 60 minutes | 10:00-11:30 ET |
| Gap Continuation | 60 minutes | 09:35-11:00 ET |
| Gap Fade | 60 minutes | 09:35-11:00 ET |
| VWAP Continuation | 60 minutes | 11:30-14:00 ET |
| VWAP Reversion | 30 minutes | 11:30-14:30 ET |
| Intraday Momentum | 60 minutes | 13:30-15:00 ET |
| Short-Horizon Reversal | 30 minutes | 13:00-15:30 ET |

All strategies are long-only. Gap Fade means a confirmed long reversion after a
negative opening gap; it does not authorize short selling. Session segments are
part of setup identity. Evidence from one segment cannot justify output in
another.

The exact setup rules, estimator budgets, features, gates, and thresholds are
frozen in `configs/intraday_specialist_research.toml`.

## Data Contract

The immutable two-year S&P five-minute SIP corpus and its point-in-time
membership are the decision-feature source. KS4 consumes every regular-session
row from the retained monthly V3 technical shards, repairs the fixed six-bar
09:30-10:00 opening range, and recomputes exact market, sector, breadth, regime,
and cross-sectional features before applying setup rules. The old V3 monthly
label bundle is reference-only: its rotating hourly sampling and future-label
truncation make it ineligible as a KS4 setup population. Existing future labels
and outcome columns are prohibited inputs and are independently poison-tested.

The existing local one-minute archive is not eligible because its schema does
not declare `price_feed`. KS4 must collect a new selective Alpaca SIP archive
for:

1. every setup ticker from the required warm-up cutoff through its exact
   decision and label window;
2. SPY, QQQ, and the point-in-time sector ETF over identical intervals;
3. enough preceding one-minute bars to satisfy the 130-bar warm-up;
4. adjustment `all`, explicit feed `sip`, source timestamps, request identity,
   page identity, and terminal collection evidence.

Collection is driven only by causal five-minute setup rows. Postdecision
one-minute data cannot decide which windows are downloaded. Ticker/session
windows are merged before collection to avoid redundant requests. One ticker or
session failure is isolated and cannot corrupt completed peers.

## Decision And Label Semantics

- A five-minute feature is usable only after its completed bar plus the frozen
  30-second provider-finalization delay.
- The source V3 timestamp is a bar-start timestamp. KS4 corrects the signal
  cutoff to `bar_start + 5 minutes`, sets feature availability to another
  30 seconds later, and enters at the next whole-minute boundary. It never uses
  the legacy V3 `decision_time_utc` as an entry timestamp.
- Strategy session limits are evaluated against the actual decision minute,
  not the preceding five-minute signal minute.
- The fixed 09:30-10:00 ET opening range is first usable after the 09:55
  five-minute bar closes and finalizes. Opening-range fields are null on every
  earlier source row; ORB and Gap Fade cannot decide before 10:01 ET.
- Entry is the exact one-minute bar beginning one minute after the completed
  five-minute bar boundary and fills at that bar's open.
- A 60-minute strategy requires 60 consecutive in-session one-minute bars; a
  30-minute strategy requires 30.
- Label barriers use `atr_14_price_5m` from the causal five-minute setup row.
  One-minute ATR is a confirmation feature only and cannot move label
  barriers. Target is `1.0 * entry ATR`; stop is `0.75 * entry ATR`.
- Same-minute target/stop ambiguity is stop-first.
- A stop gap fills at the worse of the stop or trigger-bar open.
- A target fill receives the target price, never favorable gap improvement.
- Timeout fills at the final horizon-bar close.
- SPY, QQQ, and sector benchmark returns enter at the corresponding entry
  minute open and exit at the close of the stock's realized exit minute. This
  applies consistently to target, stop, and timeout outcomes.
- Costs are applied exactly once and never below 10 bps round trip.
- The content-addressed execution policy supplies conservative price,
  volatility, liquidity, and stress costs.

The 130-bar one-minute warm-up includes only bars whose interval and frozen
30-second finalization delay are complete by feature availability. The
one-minute bar immediately preceding entry is therefore excluded. Acquisition
requirements are split into regular-session segments before merging; an API
request may never span an overnight, premarket, or after-hours interval.

Historical quote-calibrated spread/impact coefficients are unavailable. This
does not permit a zero-cost fallback: development uses the conservative bound
policy and records calibration as unavailable. No KS4 candidate can be promoted
until prospective execution calibration is supplied under the later promotion
checkpoint.

## Feature And Catalyst Policy

Setup eligibility uses only completed five-minute technical, market, sector,
volume, and point-in-time universe evidence. Once a setup window has been
selected and collected, causal one-minute confirmation features may enter the
technical model if their full warm-up is exact.

Catalyst and sentiment fields are not estimator features in KS4 V1. They form a
separate confirmation/ranking overlay applied only after the technical score.
The overlay is retained only if it improves average net return and SPY excess
in both walk-forward and unseen-ticker scopes without increasing drawdown.
Missing catalyst coverage is missing, never neutral.

The overlay is data-blocked unless its immutable event bundle binds the raw
source artifact, security-resolution lineage, source coverage, provider event
time, first-observed time, ingestion time, and half-open lookback endpoints.
Historical publication time alone does not prove live availability. Events
without causal first-observed evidence cannot participate in overlay
evaluation.

## Candidate Budget

Every strategy compares:

- deterministic setup score;
- regularized logistic regression;
- histogram gradient boosting;
- direct ranking only for Intraday Momentum.

Every estimator is evaluated with:

- model score only;
- catalyst confirmation overlay.

This yields at most eight bounded evaluations per strategy and remains below the
repository experiment budget. No additional family, feature profile, threshold,
or selection policy may be introduced after inspecting validation outcomes.

## Validation

- Four purged, embargoed, chronological folds.
- At least one full session of embargo.
- Label availability strictly precedes each validation decision.
- Deterministic unseen-ticker holdout separate from temporal validation.
- Setup population and row identities identical across estimator comparisons.
- Raw score drives ranking; calibrated probability is interpretive.
- Opportunity and downside targets remain separate.
- Selection is capped at ten trades per session.
- Market-regime, liquidity, cost-stress, capacity, calibration, drawdown, and
  session-block confidence evidence is mandatory.
- Catalyst-overlay and score-only selections are compared on the same candidate
  decisions.

## Artifact And Resource Contract

Dataset, selective one-minute collection, strategy, candidate, and evidence
bundles are immutable and content-addressed. Authority is granted only by an
atomic pointer whose hash matches a complete manifest and exact file set.
Rejected candidates retain evidence but no loadable model.

Setup extraction and one-minute acquisition planning are separate immutable
phases. Setup shards are published monthly before any setup is expanded into
stock and benchmark requirements. The collection-plan bundle binds the setup
bundle fingerprint and carries an explicit requirement-to-window bridge.
Neither an incomplete setup directory nor an incomplete collection-plan
directory has authority.

Provider acquisition units consolidate every required ticker/session to that
session's full XNYS regular interval. This is a bounded causal superset of the
exact requirements, preserves session VWAP from the open, and excludes
premarket, after-hours, and overnight data. Symbols sharing an exact session
interval are batched deterministically with no more than 8,000 expected rows
and 50 symbols per unit. Every unit binds the setup and collection-plan
fingerprints, XNYS calendar version, provider-symbol mapping, `1Min`, SIP,
adjustment `all`, and the session-date `asof` policy. Completed units are
hash-verified against the exact request, symbol set, bounds, `asof`, feed,
adjustment, timeframe, and policy before they are skipped on resume. Failed
units alone are retried. The provider's inclusive `end` is translated from the
internal half-open interval by subtracting one microsecond.

Collection completion means every provider request reached a terminal,
integrity-checked transport result. It never means the corpus is eligible for
training. Every completed collection is explicitly `model_data_ready=false`
until the requirement-to-window bridge audit proves stock, SPY, QQQ, sector,
warm-up, entry, and label-path coverage. Missing or halted minutes are recorded,
never imputed. Any setup lacking its exact required path is excluded with a
machine-readable reason; aggregate-only silent removal is prohibited.

All heavy entry points acquire the shared workspace lease. Collection, dataset
construction, and training run sequentially. Each process fails before the
3.25 GiB safety threshold and never exceeds the 4 GiB hard limit. Full-grid
cross-sectional setup construction is processed in frozen five-session batches
and setup/request artifacts are published as monthly shards, so two years of
rows are never retained in memory together.

## Exit Gates

KS4 closes only when:

1. all seven causal setup populations are independently built or carry a
   precise data-blocked result;
2. all eligible labels replay from exact SIP one-minute stock and benchmark
   paths;
3. costs, adverse fills, benchmark intervals, session isolation, and catalyst
   overlay decisions reproduce;
4. every bounded candidate is independently accepted or rejected;
5. focused poison tests, the full test suite, Ruff, strict mypy, compilation,
   memory checks, and one consolidated ML/systems review pass;
6. evidence and the execution ledger are committed and pushed.

Passing KS4 development gates does not authorize production serving. Promotion
still requires untouched prospective shadow outcomes and execution calibration.
