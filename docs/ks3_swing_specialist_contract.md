# KS3 Swing Specialist Contract

## Purpose

KS3 determines whether each registered swing behavior has repeatable,
cost-adjusted edge. It does not promote, serve, ensemble, alert, size, or
execute a strategy.

The frozen policy is
[`configs/swing_specialist_research.toml`](../configs/swing_specialist_research.toml).
The policy, source manifests, selected columns, folds, ticker holdout, candidate
artifacts, row-level predictions, and economic audits are content-addressed.

## Immutable Inputs

KS3 consumes:

- the five-year technical feature artifact produced from adjusted SIP bars;
- the completed KS1 direct-issuer catalyst lineage and source-coverage artifact;
- the KS2 strategy policy and exact-path evaluator;
- the immutable SPY, QQQ, and sector benchmark bars;
- the execution policy already bound by KS2.

No provider call or data download is part of KS3. The technical feature artifact
is projected to decision-time columns before any label build. Generic targets,
future values, and prior label columns are prohibited at the setup boundary.

## Dataset Construction

KS1 assignments are aggregated by canonical decision identity and frozen 2-hour,
1-day, and 3-day windows. Aggregates use the canonical rules:

- event count is the count of assigned direct-issuer event identities;
- sentiment is relevance-weighted;
- sentiment coverage is the fraction with sentiment;
- unknown relevance has zero sentiment weight and counts as low relevance;
- low relevance means missing relevance or relevance below 0.5;
- source-family count is the number of observed source families.

Source completeness is an interval fact, not an event count. A decision is
complete only when one contiguous `observed_complete` or `observed_empty` KS1
coverage interval with known missingness contains the full three-day feature
window ending at the decision timestamp. Blind or partial windows remain
unavailable. A complete row with zero assigned events is verified no-catalyst
evidence; an unavailable row is not converted to zero.

The KS2 evaluator is replayed after this join. This activates Catalyst Drift and
Short-Term Reversal only where their frozen setup rules pass. Exact stock,
SPY, QQQ, sector, entry, exit, cost, MFE, MAE, and breakout barrier semantics
remain those of KS2.

One immutable eligible-row dataset is published per strategy. The dataset
retains setup, label, catalyst, feature, universe, and execution identities.
PEAD remains data-blocked because point-in-time surprise and guidance history is
not available.

## Frozen Comparisons

Each strategy receives one deterministic comparator plus regularized logistic
and histogram gradient boosting candidates. Direct XGBoost ranking is allowed
only for cross-sectional and sector-residual strategies.

Candidate budget:

| Strategy | Candidate count |
| --- | ---: |
| Cross-Sectional Momentum | 4 |
| Time-Series Momentum | 3 |
| Catalyst Drift | 7 |
| Short-Term Reversal | 3 |
| Breakout Expansion | 3 |
| Sector-Residual Momentum | 4 |

Deterministic comparator formulas are fixed as follows:

- `xs_rank_rel_return_20d_vs_sector`: the identically named rank feature;
- `trend_strength`: `0.35 * return_20d + 0.25 * dist_sma_50 +
  0.25 * dist_sma_200 + 0.15 * sma_200_slope_20d`;
- `catalyst_confirmation`: `log1p(event_count_3d) *
  event_relevance_mean_3d * sentiment_coverage_3d +
  sentiment_mean_3d`;
- `reversal_extremity`: `-(return_5d / max(atr_pct_14, 1e-6)) +
  (35 - rsi_14) / 35`;
- `breakout_confirmation`: `log1p(volume_ratio_20) + close_location`.

These formulas are comparators, not promoted rules. Their raw scores receive
the same prior-fold-only isotonic calibration as learned candidates.

For Catalyst Drift, logistic and HGB are evaluated with technical-only,
catalyst-only, and combined features on the identical catalyst-confirmed
population. The deterministic comparator is evaluated once. Short-Term Reversal
uses technical features only because its verified zero-event setup makes every
catalyst aggregate constant and therefore unsuitable for a feature ablation.

## Validation

Each strategy freezes one deterministic unseen-ticker assignment from the
causally mature first training window. All candidates for that strategy use:

- the same eligible rows;
- the same expanding, time-ordered four-fold split;
- a purge equal to the strategy label horizon;
- labels available strictly before each test decision;
- the same unseen tickers and test sessions;
- the same top-10 selection policy.

The four requested folds are mandatory. A strategy is data-blocked if all four
cannot be constructed under the frozen minimum-history and sample-size rules.
Raw model score determines top-10 ranking; calibrated probability is retained
for probability interpretation and calibration audits, but isotonic plateaus
cannot decide selection.

The first fold seeds causal score calibration and is not scored. Calibration for
later folds uses only earlier-fold predictions whose labels are available before
the test cutoff. Deterministic and ranker scores are calibrated under the same
rule so candidate rows remain identical.

## Economic Evidence

Every candidate retains row-level temporal and unseen-ticker predictions plus:

- classification and ranking metrics;
- net return after the one KS2 execution cost;
- excess return versus SPY, QQQ, and sector on the identical interval;
- phase-separated returns to avoid overlapping-horizon inflation;
- selected trades, win rate, profit factor, drawdown, and negative-phase rate;
- regime results;
- liquidity and one-percent-ADV capacity estimates;
- feature-profile ablations;
- explicit accepted-development, rejected, failed, or data-blocked status.

Acceptance requires both validation scopes to pass every frozen economic gate,
including non-negative 95% confidence lower bounds for net and SPY-relative
returns. Risk-on, neutral, and risk-off evidence must each meet the frozen
minimum sample, net-return, and SPY-relative-return gates.
Passing KS3 permits later governance work only. It does not authorize
prospective shadow, promotion, serving, alerts, or trading.

## Fail-Closed Behavior

KS3 stops or isolates the affected strategy when:

- source, child-manifest, or request hashes do not reconcile;
- catalyst coverage is unavailable or stale by its historical interval;
- a feature is available after the decision;
- a label is available at or before the decision;
- stock or benchmark paths are missing or costs do not reconcile exactly once;
- fold or ticker assignments differ across candidate comparisons;
- the process exceeds the 3.25 GiB safety threshold under the 4 GiB budget.

One strategy failure is recorded independently and does not corrupt another
strategy's artifacts. Heavy builds and training remain sequential.
