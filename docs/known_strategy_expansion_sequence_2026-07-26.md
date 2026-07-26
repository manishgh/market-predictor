# Known Strategy Expansion Sequence

Date: 2026-07-26

Status: design and implementation plan; no checkpoint in this document is
implemented merely by being listed.

## 1. Objective

Replace generic "predict whether the stock rises" research with a bounded catalog
of known market strategies. Each strategy receives its own setup eligibility,
labels, features, estimator evidence, economic audit, model identity, and API
output. A strategy is a hypothesized market behavior; logistic regression,
gradient boosting, GARCH, survival analysis, and neural networks are model
families or risk components, not strategies.

The repository remains prediction-only. TradingFlow owns alerts, orders,
positions, final sizing, and execution.

## 2. Naming Contract

Stable strategy IDs use:

`<MODE>.<KNOWN_STRATEGY>.<HORIZON>.V<MAJOR>`

Examples:

- `SWING.CATALYST_DRIFT.5D.V1`
- `INTRADAY.OPENING_RANGE_BREAKOUT.60M.V1`

Rules:

- The display name uses the conventional market name.
- The ID describes behavior, not an estimator or vendor.
- Horizon and semantic major version are mandatory.
- A material label, execution, or eligibility change requires a new major version.
- Estimator family, feature profile, and hypothesis ID remain separate fields.
- Retired or failed strategy versions are never silently reused.

## 3. Canonical Strategy Catalog

### 3.1 Swing strategies

| Strategy ID | Conventional name | Hypothesis | Current state |
|---|---|---|---|
| `SWING.CROSS_SECTIONAL_MOMENTUM.5D.V1` | Cross-Sectional Momentum / Relative Strength | Recent leaders continue to outperform the same-session universe | Generic ranks exist; five-year technical candidate rejected |
| `SWING.TIME_SERIES_MOMENTUM.5D.V1` | Time-Series Momentum / Trend Following | A stock's own established trend continues | Features exist; no isolated strategy label/model |
| `SWING.CATALYST_DRIFT.5D.V1` | Catalyst Drift | Relevant new information is incorporated over several sessions | News pipeline mostly complete; no joined five-year model |
| `SWING.POST_EARNINGS_DRIFT.5D.V1` | Post-Earnings-Announcement Drift (PEAD) | Earnings surprise and guidance cause delayed repricing | Event subtype and surprise history are incomplete |
| `SWING.SHORT_TERM_REVERSAL.3D.V1` | Short-Term Reversal / Overreaction | An excessive reaction relative to evidence and peers partially reverses | MFE/MAE exist; no dedicated eligibility or label |
| `SWING.BREAKOUT_EXPANSION.5D.V1` | Breakout / Volatility Expansion | Compression followed by price and volume confirmation continues | Range, ATR, trend, and volume features exist; no specialist |
| `SWING.SECTOR_RESIDUAL_MOMENTUM.5D.V1` | Sector-Neutral Residual Momentum | Stock-specific strength persists after removing market and sector movement | Relative features and labels exist; no direct residual objective |
| `SWING.PAIRS_REVERSION.5D.V1` | Pairs Trading / Statistical Arbitrage | A validated peer spread converges after an abnormal divergence | Deferred; point-in-time peer and spread evidence required |

`CATALYST_DRIFT` is the parent strategy for contracts, approvals, analyst changes,
regulatory decisions, financing, M&A, product events, and geopolitical exposure.
`POST_EARNINGS_DRIFT` remains separate because earnings surprise, guidance, and
calendar timing require distinct evidence.

### 3.2 Intraday strategies

| Strategy ID | Conventional name | Hypothesis | Current state |
|---|---|---|---|
| `INTRADAY.OPENING_RANGE_BREAKOUT.60M.V1` | Opening Range Breakout (ORB) | A confirmed opening-range break continues | Causal opening-range features exist; no opening-only estimator |
| `INTRADAY.GAP_CONTINUATION.60M.V1` | Gap-and-Go / Gap Continuation | A catalyst-confirmed overnight gap continues | Gap and catalyst evidence exist; no dedicated label/model |
| `INTRADAY.GAP_FADE.60M.V1` | Gap Fade | An unsupported or exhausted gap mean-reverts | Gap exists; no fade eligibility or model |
| `INTRADAY.VWAP_CONTINUATION.60M.V1` | VWAP Reclaim / Continuation | A supported VWAP reclaim or hold continues | VWAP features exist; no specialist |
| `INTRADAY.VWAP_REVERSION.30M.V1` | VWAP Mean Reversion | Excessive VWAP distance without confirmation reverts | VWAP distance exists; no specialist |
| `INTRADAY.MOMENTUM_CONTINUATION.60M.V1` | Intraday Momentum | Relative-price and volume leadership persists | Generic ranker attempted and rejected; no setup-isolated model |
| `INTRADAY.SHORT_HORIZON_REVERSAL.30M.V1` | Short-Horizon Reversal | A liquidity-adjusted shock reverses over the next bars | Features partly exist; no dedicated model |
| `INTRADAY.ORDER_FLOW_IMBALANCE.15M.V1` | Order-Flow Imbalance (OFI) | Bid/ask event imbalance predicts the near-term move | Deferred; historical quotes/depth are absent |

Opening, midday, and power-hour observations must not share one specialist merely
because their nominal horizon matches. Session segment is part of setup
eligibility and validation strata.

## 4. Risk And Meta Components

These components may modify readiness, uncertainty, or ranking. They are not
standalone directional strategies.

| Component ID | Known method | Intended role |
|---|---|---|
| `RISK.REALIZED_VOLATILITY.60M.V1` | Realized volatility / ATR baseline | Current volatility reference |
| `RISK.GARCH.60M.V1` | GARCH(1,1) | Conditional intraday variance forecast |
| `RISK.EGARCH.60M.V1` | Exponential GARCH | Asymmetric response to positive/negative shocks |
| `RISK.GARCH.5D.V1` | Daily GARCH | Swing-horizon conditional variance |
| `RISK.HAR_RV.60M.V1` | Heterogeneous Autoregressive Realized Volatility | Multi-scale realized-volatility benchmark |
| `META.QUANTILE_RETURN.V1` | Quantile regression | Downside, median, and upside return estimates |
| `META.COMPETING_RISKS.V1` | Competing-risk / survival model | Target, stop, or timeout probability and timing |
| `META.META_LABEL.V1` | Meta-Labelling | Decide whether to accept a deterministic primary setup |
| `META.REGIME_MIXTURE.V1` | Regime-Switching / Mixture of Experts | Route a setup to a prevalidated regime specialist |

GARCH adoption requires improvement over realized volatility and ATR on both
forecast loss and downstream economic ablation. Intraday fitting must remove
time-of-day seasonality and separate overnight gaps. The implementation should
use the maintained `arch` library rather than a hand-written optimizer.

## 5. Shared Strategy Contract

Every strategy row must carry:

- `strategy_id` and `strategy_version`;
- `decision_time_utc` and exact feature cutoff;
- `setup_eligible` and bounded ineligibility reason;
- point-in-time `security_id`, ticker, sector, market-cap, and liquidity state;
- immutable source, feature, label, execution, and universe identities;
- exact entry, exit, target, stop, timeout, cost, MFE, and MAE evidence where applicable;
- SPY, QQQ, and sector returns over the identical executable interval;
- catalyst relation/channel and availability evidence when required;
- decision group and cross-sectional eligibility;
- strategy-specific outcome and economic return.

The API output for an eligible strategy should support:

- `opportunity_probability`;
- `downside_probability`;
- `expected_net_return`;
- `expected_excess_return_vs_spy`;
- `expected_excess_return_vs_sector`;
- downside, median, and upside return quantiles when available;
- expected MFE, MAE, and time to resolution when available;
- conditional-volatility forecast and volatility regime;
- catalyst confirmation and explanation;
- rank, uncertainty, readiness, model identity, and evidence cutoff.

The API must not average probabilities from different strategies. It returns
independent strategy assessments. A later deterministic selection policy may
rank only compatible, individually eligible assessments.

## 6. Sequential Checkpoints

Only one heavy build, inference, or training process may run at a time. Every
checkpoint is committed and pushed before the next begins.

### KS0: Freeze catalog and hypothesis budget

Deliver:

- strategy IDs and schemas;
- one bounded hypothesis per strategy version;
- shared baseline, folds, costs, and capacity assumptions;
- maximum experiment count and retirement rules;
- no shadow access.

Exit:

- strategy and component names are immutable;
- estimator family cannot be mistaken for a strategy;
- current generic models are reference baselines only.

### KS1: Complete catalyst lineage

Deliver:

- replay five-year ticker/business/sector event relations;
- verify identity, relevance, publication proxy, duplicates, and candle assignment;
- join relation and completed FinBERT artifacts causally to swing decisions;
- retain explicit missingness and exclude blind security/source windows;
- publish catalyst-only and technical-plus-catalyst feature inventories.

Exit:

- no unrelated ticker news;
- no profile data backdated before its evidence;
- no event later than a decision cutoff;
- exact event, relation, sentiment, and decision hashes reconcile.

### KS2: Build strategy-specific label primitives

Deliver:

- setup eligibility computed only from decision-time evidence;
- reusable exact-path outcome evaluator;
- separate continuation, reversal, breakout, and residual labels;
- strategy-specific decision groups and abstention reasons;
- poison tests for look-ahead setup selection.

Exit:

- one generic positive-return label is no longer reused as proof for every strategy;
- every strategy label replays from immutable bars and execution policy.

### KS3: Swing specialist sequence

Train sequentially on identical frozen folds:

1. `SWING.CROSS_SECTIONAL_MOMENTUM.5D.V1` as the reference.
2. `SWING.CATALYST_DRIFT.5D.V1`.
3. `SWING.POST_EARNINGS_DRIFT.5D.V1` after point-in-time surprise data passes.
4. `SWING.SHORT_TERM_REVERSAL.3D.V1`.
5. `SWING.BREAKOUT_EXPANSION.5D.V1`.
6. `SWING.TIME_SERIES_MOMENTUM.5D.V1`.
7. `SWING.SECTOR_RESIDUAL_MOMENTUM.5D.V1`.

For each strategy compare:

- deterministic strategy floor;
- regularized logistic baseline;
- histogram gradient boosting;
- direct ranking only when the target is cross-sectional.

Exit:

- each candidate is accepted or rejected independently;
- no failed specialist is hidden inside an ensemble;
- no prospective shadow is opened until development and holdout economics pass.

### KS4: Intraday specialist sequence

Train sequentially on exact one-minute paths:

1. `INTRADAY.OPENING_RANGE_BREAKOUT.60M.V1`.
2. `INTRADAY.GAP_CONTINUATION.60M.V1`.
3. `INTRADAY.GAP_FADE.60M.V1`.
4. `INTRADAY.VWAP_CONTINUATION.60M.V1`.
5. `INTRADAY.VWAP_REVERSION.30M.V1`.
6. `INTRADAY.MOMENTUM_CONTINUATION.60M.V1`.
7. `INTRADAY.SHORT_HORIZON_REVERSAL.30M.V1`.

Use catalyst as a confirmation/ranking overlay initially. It may enter an
estimator only after a preregistered ablation improves both walk-forward and
unseen-ticker evidence.

Exit:

- every specialist evaluates only its causal setup population;
- target, stop, timeout, spread/cost, and benchmark paths are exact;
- opening-only evidence cannot justify midday or power-hour output.

### KS5: Distributional and path models

Deliver:

- expected net and benchmark-relative return regression;
- calibrated return quantiles;
- MFE and MAE estimates;
- competing-risk target/stop/timeout probabilities;
- time-to-resolution estimates.

Exit:

- distributional outputs improve selection or abstention beyond binary probability;
- interval and event-probability calibration pass by regime and liquidity tier.

### KS6: Volatility sidecars

Evaluate in order:

1. realized volatility and ATR;
2. HAR-RV;
3. GARCH(1,1);
4. EGARCH only if asymmetry is supported.

Forecast horizons:

- intraday next 30 and 60 minutes;
- swing next one and five sessions.

Exit:

- QLIKE and forecast calibration improve out of sample;
- stop/risk or selection ablation improves net economics and drawdown;
- no sidecar becomes directional authority;
- fitting and inference remain below the 4 GiB limit.

### KS7: Regime routing and meta-labelling

Deliver:

- causal regime state using only information available at the decision;
- deterministic primary setup plus meta-label accept/reject experiment;
- mixture-of-experts comparison only among specialists that passed independently.

Exit:

- routing improves both temporal and ticker-holdout economics;
- no regime is defined from future returns;
- no ensemble rescues an individually failed strategy through model shopping.

### KS8: Data-dependent later strategies

Do not start until required evidence exists:

- `SWING.PAIRS_REVERSION.5D.V1`: point-in-time peer map and stable spread tests;
- `INTRADAY.ORDER_FLOW_IMBALANCE.15M.V1`: historical quotes, trades, spread,
  depth, cancellations, and execution simulation;
- sequence models such as TCN/LSTM/TFT: only after tabular specialists establish
  a positive baseline and the sample size supports the parameter count.

### KS9: API, promotion, and TradingFlow integration

Deliver:

- strategy-aware model registry and serving bundle;
- one coherent model identity per returned strategy assessment;
- unified response containing independent swing/intraday strategy results;
- TradingFlow contract that consumes predictions but owns action and execution;
- selected-strategy outcome maturation and monitoring.

Exit:

- only signed, promoted strategy versions are actionable;
- rejected candidates remain research-only;
- no fallback, probability averaging, predictor-owned alert, or order behavior.

## 7. Per-Strategy Research Gates

Before shadow evaluation, every strategy must pass:

- point-in-time feature and setup audit;
- exact label and benchmark-path replay;
- purged, embargoed chronological folds;
- deterministic unseen-ticker holdout;
- minimum independent decision groups;
- calibrated probability or distributional forecast;
- positive cost-adjusted selected return in both scopes;
- positive benchmark-relative economics in both scopes;
- declared drawdown, turnover, liquidity, and capacity limits;
- cost and adverse-fill stress;
- session-block confidence interval;
- feature and strategy ablation against its frozen baseline;
- memory peak below 4 GiB;
- immutable candidate and evidence manifests.

Promotion additionally requires untouched prospective shadow evidence, positive
paired confidence lower bounds, one-use shadow consumption, distinct authenticated
build/approval principals, and an atomic serving bundle.

## 8. Immediate Execution Order

The next implementation work is:

1. KS0 strategy contracts and tests.
2. KS1 five-year catalyst relation replay and causal join.
3. KS2 strategy-label framework.
4. KS3 Catalyst Drift, Short-Term Reversal, and Breakout Expansion.
5. KS4 Opening Range Breakout, Gap Continuation/Fade, and VWAP specialists.
6. KS5 distributional/path outputs.
7. KS6 GARCH/HAR-RV volatility sidecars.
8. KS7 regime routing and meta-labelling.
9. KS8 only after new quote/depth or peer evidence exists.
10. KS9 only for strategy versions that pass all existing promotion gates.

This order finishes the evidence already collected before adding new model
families. It also prevents GARCH, deep learning, or an ensemble from obscuring
the more fundamental question: which explicitly named market behavior has
repeatable, cost-adjusted predictive edge?
