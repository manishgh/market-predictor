# Primary V2 Strategy Research Plan

Status: frozen implementation contract  
Date: 2026-07-28

## Purpose

This amendment creates two new primary research strategies. It does not alter,
rescue, or reinterpret rejected KS3 or KS4 specialists, and it does not weaken
the frozen KS5 V1 rule that KS5 may consume only independently passed
specialists.

| Strategy ID | Human-readable name |
| --- | --- |
| `SWING.CROSS_SECTIONAL_MOMENTUM.5D.V2` | Swing Long Cross-Sectional Momentum - Next 5 Trading Sessions - Version 2 |
| `INTRADAY.VWAP_REVERSION.30M.V2` | Intraday Long VWAP Mean Reversion - Up to 30 Regular-Session Minutes - Version 2 |

`5D` means five exchange trading sessions, never five calendar days. `30M`
means a maximum holding path of thirty regular-session minutes, never a
thirty-minute candle.

## Exact Trading Semantics

### Swing

- Decision: after a completed exchange session.
- Entry: next exact exchange-session open.
- Timeout: close of the fifth future exchange session.
- Direction: long only.
- Net target: the existing cost-adjusted `strategy_net_return`.
- Comparators: exact contemporaneous SPY and point-in-time sector benchmark.

### Intraday

- Decision: a completed five-minute bar, usable after the frozen 30-second
  finalization delay.
- Entry: next one-minute bar open.
- Exit: first target touch, first stop touch, or the close of the thirtieth
  one-minute bar.
- Overnight positions: prohibited.
- Direction: long only.
- Net target: existing `path_realized_return_net_30m`, which includes the
  frozen round-trip cost.

## Frozen Hypotheses

The V1 classifiers primarily estimated whether a favorable event occurred.
V2 tests whether directly estimating return magnitude, downside distribution,
and path outcome produces better net-ranked selections without changing setup
generation or labels.

Swing candidates:

1. Exact deterministic V1 comparator.
2. Histogram gradient boosting expected-return regression.
3. Histogram gradient boosting 10th/50th/90th return quantiles.

Intraday candidates:

1. Multinomial target-first/stop-first/timeout baseline.
2. Histogram gradient boosting competing-risk probabilities.
3. Histogram gradient boosting 10th/50th/90th net-return quantiles.

No catalyst or news field is an estimator feature in this experiment.
Catalyst remains a separately reported confirmation and explanation overlay.
Missing catalyst history is not encoded as neutral sentiment.

## Selection Policies

Swing:

- `expected_net_top_10`: rank expected net return and select at most ten rows
  per session.
- `positive_lower_bound_then_median_top_10`: require predicted 10th-percentile
  net return above zero, rank predicted median return, and select at most ten
  rows per session.

Intraday:

- `no_veto_expected_net_top_10`: rank expected net return without the rejected
  V1 downside veto and select at most ten rows per session.
- `distributional_safety_top_10`: require positive predicted 10th-percentile
  return and positive target-minus-stop expected utility, then rank expected
  net return and select at most ten rows per session.

Thresholds and policies are frozen before evaluation. They may not be tuned
after seeing holdout results.

## Validation

Both strategies reuse the exact V1 source rows and split machinery:

- four purged walk-forward folds;
- causal training rows whose labels were available before each test decision;
- a deterministic 20% unseen-ticker holdout;
- identical rows for every paired V1/V2 comparison;
- sequential model fitting under a hard 4 GiB process-memory ceiling.

Promotion requires all of the following in both walk-forward and unseen-ticker
scopes:

- at least 100 selected rows;
- positive average net return;
- positive average SPY-relative and sector-relative return;
- positive lower confidence bounds for average net and SPY-relative return;
- profit factor at least 1.05;
- maximum drawdown no greater than 20%;
- no unsupported required market regime;
- positive regime-level net and SPY-relative return where the minimum regime
  sample is met;
- calibrated, non-crossing quantiles for quantile candidates;
- calibrated event probabilities for competing-risk candidates;
- incremental economic evidence against the exact V1 comparator.

AUC, accuracy, or calibration alone cannot promote a strategy.

## Data and Leakage Gates

Evaluation must fail closed when:

- source bundle identity or authority cannot be verified;
- required exact-label fields are absent;
- decision, entry, exit, or label-availability timestamps are invalid;
- a training label became available at or after a test decision;
- duplicate row identities exist;
- cost-adjusted and benchmark-relative labels are non-finite;
- an intraday path extends outside its regular session;
- ticker holdout identities overlap development identities;
- the source feed or adjustment mode is not the frozen SIP/all contract;
- process memory reaches the configured hard ceiling.

## Outputs

Each immutable run must publish:

- frozen request and implementation identity;
- source and split hashes;
- row-level out-of-sample predictions;
- fold and leakage audit;
- quantile or event calibration evidence;
- selected-trade economics by validation scope and regime;
- paired V1/V2 incremental evidence;
- final accepted/rejected authority;
- no retained serving model when the strategy is rejected.

