# Swing Technical 5D Logistic Baseline

Date: 2026-07-25

## Intended Use

Five-session, post-close research baseline for measuring whether daily
technical, market-regime, SPY/QQQ/sector-relative, and eligible-cohort
cross-sectional features contain stable edge. This `technical_market` profile
is baseline-only and cannot be promoted or served as catalyst intelligence.

## Data And Validation

- Decision window: 2021-07-09 through 2026-07-08.
- Eligible rows: 607,909.
- Training tickers: 581.
- Features: 53.
- Exact outcome-path coverage: 99.9941%; 36 candidate windows excluded.
- Validation: four session-purged walk-forward folds plus deterministic unseen
  ticker holdout.
- Dataset SHA-256: `d97c8aaddb7357177c7bfcaab59fcb50b2c08b20c4673bd081cf4684e50c42f9`.
- Model SHA-256: `5fe299b06c721c141a5dd68519142655f4eb33b5afc6fa86bfb9cc37e825a8f8`.

## Results

| Metric | Result |
|---|---:|
| Walk-forward ROC AUC | 0.4962 |
| Unseen-ticker ROC AUC | 0.4993 |
| Walk-forward top-decile lift | 0.9910 |
| Unseen-ticker top-decile lift | 1.0051 |
| Conservative average trade return | -0.2343% |
| Conservative profit factor | 0.8595 |
| Conservative maximum drawdown | 36.93% |
| Peak process RSS | 3.082 GiB |

## Decision

Rejected. Classification and ranking are indistinguishable from random, and
selected-policy economics are negative after costs and versus SPY. Preserve
only as the frozen linear baseline for future catalyst ablation.
