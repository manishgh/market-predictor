# Swing Technical 5D Histogram-Gradient Baseline

Date: 2026-07-25

## Intended Use

Nonlinear companion to the five-session logistic baseline, trained on the same
`technical_market` feature profile, rows, folds, labels, costs, and unseen
ticker holdout. It is baseline-only and cannot be promoted.

## Data And Validation

- Decision window: 2021-07-09 through 2026-07-08.
- Eligible rows: 607,909.
- Training tickers: 581.
- Features: 53.
- Exact outcome-path coverage: 99.9941%; 36 candidate windows excluded.
- Validation: four session-purged walk-forward folds plus deterministic unseen
  ticker holdout.
- Dataset SHA-256: `d97c8aaddb7357177c7bfcaab59fcb50b2c08b20c4673bd081cf4684e50c42f9`.
- Model SHA-256: `92153bc32763034c3c5a7512b55ee16a7600cfb213ad76d499d67ae6f20002b7`.

## Results

| Metric | Result |
|---|---:|
| Walk-forward ROC AUC | 0.5000 |
| Unseen-ticker ROC AUC | 0.4970 |
| Walk-forward top-decile lift | 1.0067 |
| Unseen-ticker top-decile lift | 1.0025 |
| Conservative average trade return | -0.0303% |
| Conservative profit factor | 0.9792 |
| Conservative maximum drawdown | 33.64% |
| Peak process RSS | 3.014 GiB |

## Decision

Rejected. The nonlinear estimator does not establish classification,
cross-sectional ranking, benchmark-relative, or cost-adjusted economic edge.
It remains the frozen technical comparison for catalyst/news ablations.
