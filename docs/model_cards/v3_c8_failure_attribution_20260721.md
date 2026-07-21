# V3 C8 R1 Failure Attribution

Status: completed development-only diagnostic. This is not a candidate or promotion artifact.

## Frozen evidence

- Development dataset fingerprint: `ae17ce380ce0765dbbfcb0e0e07a3dda2598c1bd3482a3bd6abaa5018183e098`
- R1 model run ID: `8a01f9913dcb777c0759f6e4`
- Joined OOF rows: 1,069,740
- Fixed top-10 rows: 37,240 across 1,862 decision groups
- Coverage: 542 tickers and 417 sessions
- Audit scopes: purged walk-forward and deterministic ticker holdout
- Shadow rows accessed: zero
- Unmatched frozen rows: zero
- Diagnostic strata: 107 total, 92 meeting minimum row/session evidence
- Selected evidence SHA-256: `d48628bea125cc828eb6c5f8471d64bed1f5d1db3fa2d25577d476720cf24654`
- Strata evidence SHA-256: `42cbd1260d4b2a2d03d5df3681b9f84f30e41ff9309311cce3b644ef5ba089c2`
- Observed process working set: below 2.7 GiB

## Findings

R1 cross-sectional rank correlation with 60-minute excess return is only 0.0064 walk-forward and 0.0147 on ticker holdout. The model ranks the highest score decile somewhat better than lower deciles, but the top-10 edge over the average eligible stock is only 0.0300 percentage points walk-forward and 0.0229 points on holdout. That does not cover the declared costs.

| Scope | Horizon | Selected excess return | Population group mean | Selection delta | 95% selected-return interval |
| --- | --- | ---: | ---: | ---: | ---: |
| Walk-forward | 30m | -0.0930% | -0.0999% | +0.0070 points | -0.1114% to -0.0744% |
| Walk-forward | 60m | -0.0715% | -0.1016% | +0.0300 points | -0.0980% to -0.0423% |
| Walk-forward | 120m | -0.0587% | -0.1027% | +0.0440 points | -0.0990% to -0.0145% |
| Walk-forward | To close | -0.0456% | -0.1217% | +0.0762 points | -0.1171% to +0.0238% |
| Ticker holdout | 30m | -0.0891% | -0.0985% | +0.0094 points | -0.1036% to -0.0762% |
| Ticker holdout | 60m | -0.0764% | -0.0993% | +0.0229 points | -0.0966% to -0.0570% |
| Ticker holdout | 120m | -0.0572% | -0.0991% | +0.0419 points | -0.0868% to -0.0248% |
| Ticker holdout | To close | -0.0295% | -0.1139% | +0.0844 points | -0.0822% to +0.0270% |

Only four adequately sized strata are positive: May/June 2026 in walk-forward, June 2026 on holdout, and walk-forward Real Estate. These are not stable across both scopes and must not become filters on inspected evidence. Time bucket, regime, liquidity, volatility, and sector do not produce a robust positive 60-minute stream in both scopes.

The oracle top-10 60-minute return is strongly positive, so cross-sectional opportunity exists in the label. The failure is signal extraction and cost coverage, not absence of dispersion.

## Next frozen hypothesis

V4-H1 changes one structural choice only:

- Primary ranking horizon: 120 minutes (`24` five-minute bars).
- Decision stride: 120 minutes (`24` bars), reducing overlapping decisions and turnover.
- Universe, feature schema, costs, cutoff, point-in-time membership, model family, and top-k remain unchanged.
- No month, sector, regime, catalyst, or score threshold is introduced.
- Build a new immutable dataset fingerprint and compare B0 versus R1 on identical new groups.

V4-H1 must produce positive cost-adjusted top-10 excess return in both development scopes before any shadow evaluation. Failure rejects the hypothesis; it does not authorize tuning on C9 data.
