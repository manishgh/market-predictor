# Edge-Rebuild Swing Candidate V2

Status: `reference_rejected` (`no_candidate`)

## Dataset

- Panel: `edge_rebuild.swing_panel_materialization.v11`
- Decisions: 2019-07-09 through 2026-07-08
- Rows per ablation: 853,417
- Modeled securities: 604
- Sessions: 1,759
- Rank-eligible rows: 640,107
- Profiles: technical/market and technical plus direct Alpaca ticker catalyst
- Panel manifest SHA-256: `c11d66ce404fcd91605000282446b29ff6e1551ad019f1a976beb3c6afad0abc`

## Evaluation

Six sequential candidates were evaluated: logistic regression and histogram gradient
boosting with 15 or 31 leaves for each profile. Selection used one exact 252-session
chronological validation window and a deterministic 20% unseen-security scope. The
ten-session embargo and 20 bps round-trip cost remained active.

The strongest diagnostic AUC values were approximately 0.55 on the full temporal
scope and 0.57 on unseen securities. Some selected trade subsets had positive mean
returns, but every candidate failed promotion because confidence intervals crossed
zero for calendar returns, portfolio daily returns, doubled-cost returns, or
holding-aligned excess returns against SPY, QQQ, and sector benchmarks. Catalyst
features improved one temporal HGB comparison but degraded its unseen-security
comparison, so the increment was not stable.

The locked test from 2025-07-01 through 2026-06-30 was not read. Test access count is
zero. No model file or serving bundle was published.

## Authority

- Candidate state: `no_candidate`
- Promotion permitted: false
- Evaluation SHA-256: `4bb2f753c2038adf417829f4fb08294c3f43b01fc1bc1ed55a218e5cf206d498`
- Candidate manifest SHA-256: `fd7ac448837f105137778f15790fc4ebb60728f833581a38c59a2423f299205e`
- Training configuration SHA-256: `d57d7ed55bfcfbdd2970845c9092a85a4e0d9f5932ae2663fa65e1c48fcc9be2`

This artifact is audit evidence only and is not eligible for prediction serving.
