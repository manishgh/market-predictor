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
membership are the decision-feature source. KS4 projects only predecision
columns from the monthly V3 feature artifacts. Existing future labels and
outcome columns are prohibited inputs and are independently poison-tested.

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

- A five-minute feature is usable only after its completed bar is available.
- Entry is the exact next tradable one-minute bar beginning at the synchronized
  decision cutoff.
- A 60-minute strategy requires 60 consecutive in-session one-minute bars; a
  30-minute strategy requires 30.
- Target is `1.0 * entry ATR`; stop is `0.75 * entry ATR`.
- Same-minute target/stop ambiguity is stop-first.
- A stop gap fills at the worse of the stop or trigger-bar open.
- A target fill receives the target price, never favorable gap improvement.
- Timeout fills at the final horizon-bar close.
- SPY, QQQ, and sector returns use the same entry and realized exit interval.
- Costs are applied exactly once and never below 10 bps round trip.
- The content-addressed execution policy supplies conservative price,
  volatility, liquidity, and stress costs.

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

All heavy entry points acquire the shared workspace lease. Collection, dataset
construction, and training run sequentially. Each process fails before the
3.25 GiB safety threshold and never exceeds the 4 GiB hard limit.

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
