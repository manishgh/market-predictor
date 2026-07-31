# Model Training And Validation Protocol

Status: governing research protocol
Aligned: 2026-07-31

This protocol makes the repository's implementation choices explicit where the
[quantitative trading plan](references/comprehensive_quantitative_trading_model_implementation_plan_intraday_and_swing.pdf)
states the methodology at a higher level. The source PDF is retained unchanged;
its SHA-256 is
`ea8df4ad6f3c1d17666cadff2672887b01ee46211ef11a040715f9890cb2b75b`.

## 1. Questions And Evaluation Scopes

The primary question is whether a model trained on prior market states ranks or
classifies future opportunities across the contemporaneous tradable universe. A
security may occur in training, validation, and test at different times. That is
not leakage: future rows are never available to earlier decisions.

Two independent scopes are required:

1. **Temporal generalization:** future sessions containing the full eligible
   point-in-time cross-section.
2. **Unseen-security generalization:** a deterministic security holdout nested
   inside development data. This measures transfer to names absent from fitting
   and must not replace the temporal test.

Every row from one decision session belongs to one split. Splitting securities
from the same session across train and validation is prohibited because it leaks
the market regime and cross-sectional normalization state.

## 2. Swing Dataset And Splits

- Universe: point-in-time S&P 500 membership, including historical members and
  delisted securities where identity and bars are provable.
- Modeling horizon: seven usable years: five years for fitting, one year for
  validation, and one untouched test year. The preceding 250 trading sessions
  are indicator warm-up only and are not training examples.
- Decision frequency: one cross-section per exchange session.
- Feature normalization: winsorization, z-scores, ranks, sector-relative values,
  imputation, and feature selection are fit or computed without future sessions.
- Labels: exact next-open entry and preregistered target, stop, and timeout paths;
  stock, SPY, and sector returns use the identical executable interval.
- Data-quality tolerance: exclude the complete security, with an audited reason
  and affected dates, when its data is unavailable or unverifiable. Continue
  through a maximum 5% loss of the filtered point-in-time universe; refuse above
  5%. SPY, sector-benchmark, and market-wide session gaps cannot use this rule.

The final evaluation uses one full validation year before one locked test year.
This is the maximum complete train/validation/test design within the approved
seven-year swing horizon. Dates are derived from XNYS sessions and frozen in the
temporal manifest:

| Fold | Fit window | Validation window |
| --- | --- | --- |
| 1 | May 2019-May 2024 | June 2024-June 2025 |
| Final refit | June 2020-June 2025 | none |
| Locked test | none | July 2025-June 2026 |

The split generator must use actual session boundaries, group by decision date,
and apply a purge plus embargo of at least the maximum ten-session swing label
horizon. Validation may select hyperparameters and stopping rounds. The locked
test is opened once after the model, feature profile, selection policy, costs,
and thresholds are frozen.

The current 2019-07-09 through 2026-07-08 panel is valid for causal panel and
feature diagnostics. It lacks the 250-session pre-fit warm-up and the first 29
fit sessions required by the frozen schedule. It therefore cannot supply final
promotion evidence until the exact 2018-05-29 through 2019-07-08 gap is filled.

## 3. Intraday Dataset And Splits

- History target: one to three years of causally complete sessions.
- Inputs: one-minute bars for executable paths; five-minute or volume bars for
  decision features as frozen by the strategy contract.
- Universe: the point-in-time eligible population for each session, not a current
  static ticker list.
- Session isolation: indicators and normalizers reset at session boundaries;
  overnight observations are explicit context and never extend an intraday
  rolling window.
- Split unit: full exchange session. Every ticker and bar belonging to a session
  remains in the same fold.
- Leakage control: purge by the maximum label duration in minutes and embargo
  the overnight boundary. No intraday position survives the frozen close cutoff.

Intraday and swing folds are separate artifacts. A daily split must not be
silently reused for intraday evidence or vice versa.

## 4. Model Sequence

The deterministic technical composite is a comparison baseline. It cannot veto
all estimators. Its failure rejects that exact formula only.

For each horizon, run this bounded sequence:

1. Deterministic comparator and logistic baseline.
2. Gradient-boosted barrier classifier estimating target-before-stop probability.
3. Grouped cross-sectional ranker, such as LightGBM LambdaMART, trained with
   decision session as the query group.
4. Random Forest or another preregistered nonlinear comparator only when it fits
   within the experiment and memory budget.

The classifier answers whether an opportunity clears its executable barrier
outcome. The ranker answers which securities are strongest relative to peers on
the same decision date. A selection policy may require both a high rank and an
acceptable calibrated barrier probability, then apply sector, liquidity,
turnover, and position-count constraints.

Feature diagnostics, redundancy removal, and hyperparameter selection occur
inside fitting data only. Catalyst features start as a causal confirmation,
veto, explanation, or ranking overlay. They enter an estimator only after a
preregistered ablation improves both temporal and unseen-security evidence.

## 5. Sector Treatment

Train one global S&P 500 model first. Include point-in-time sector identity,
sector ETF context, stock-versus-sector residual returns, and bounded
sector-feature interactions. Report every metric by sector and enforce portfolio
sector constraints.

Do not begin with separate sector models. A sector specialist is admissible only
when the sector has sufficient independent sessions and opportunities and beats
the frozen global model on untouched, sector-specific evidence after costs.

## 6. Metrics And Statistical Unit

Rows from one date are correlated; the independent unit is the decision session.
Report at minimum:

- daily Spearman rank information coefficient and its stability;
- NDCG at the frozen selection depth;
- top-minus-bottom quantile spread;
- selected top-k gross, net, SPY-excess, and sector-excess return;
- win rate, profit factor, turnover, capacity, and maximum drawdown;
- probability calibration and barrier-classification metrics;
- performance by year, market regime, sector, capitalization, and catalyst state.

Confidence intervals and significance use date-block bootstrap or Newey-West
corrections appropriate to the holding horizon. Hundreds of securities on one
date do not count as hundreds of independent observations.

## 7. Acceptance And Audit Evidence

Every run must bind immutable hashes for source data, point-in-time membership,
feature contract, label policy, split manifest, model configuration, costs, and
selection policy. The audit must prove:

- no feature, event, membership, normalizer, or label path exceeds its decision
  cutoff;
- all members of a decision date have one fold assignment;
- warm-up rows do not enter training counts or metrics;
- validation and locked-test results are distinguishable and cannot be
  overwritten;
- benchmark and sector comparisons use the stock's exact executable interval;
- the reported policy is reproduced from row-level predictions after costs.

Promotion still requires prospective shadow evidence. A retrospective locked
test is necessary evidence, not authorization for live trading.
