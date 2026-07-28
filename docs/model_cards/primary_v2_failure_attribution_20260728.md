# Primary V2 Failure Attribution and Setup Viability

Date: 2026-07-28
Status: complete; no replicated viable cohort; no V3 hypothesis authorized

## Scope

This audit explains the rejection of the two primary V2 strategies. It uses all
101,918 swing and 50,471 intraday validation rows from the exact V2 split, not
training rows and not only model-selected trades. Cohorts were frozen before the
outcomes were calculated and are one-dimensional. Five-session swing outcomes
are evaluated in five non-overlapping calendar phases; adjacent overlapping
labels are never treated as independent.

Both source bundles and V2 runs passed recursive authority, implementation,
policy, artifact-hash, and exact row-identity verification.

## Swing Result

Strategy: `SWING.CROSS_SECTIONAL_MOMENTUM.5D.V2`, meaning a long
cross-sectional momentum position held for the next five exchange sessions.

| Scope | Minimum phase rows | Gross range | Cost range | Worst phase net | Worst net 95% lower bound | Worst SPY excess | Worst SPY 95% lower bound | Minimum profit factor | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Walk-forward | 16,457 | +14.03 to +16.70 bps | 20.83 to 20.89 bps | -6.82 bps | -38.30 bps | -29.90 bps | -44.03 bps | 0.959 | 35.20% |
| Unseen ticker | 3,740 | +17.01 to +26.33 bps | 19.96 to 20.02 bps | -3.02 bps | -34.81 bps | -24.14 bps | -38.77 bps | 0.981 | 33.33% |

The setup has some average gross movement, but it is not a stable tradable edge:

- Stamped execution costs consume or nearly consume the gross return.
- Some phases are net-positive, but the worst phase is negative in each scope
  and every phase has a negative net confidence lower bound.
- Every phase-qualified scope underperforms SPY.
- Profit factor and drawdown fail on the worst phase in both scopes.
- Zero of 180 scope/phase/cohort records passes. Zero of 18 predeclared cohort
  values passes all five phases in either scope.

The strongest conservative descriptive cohort is `risk_on`. Its worst phase
averages +1.79 bps net in walk-forward and +7.18 bps for unseen tickers, but the
net lower bounds are -30.57 and -25.65 bps. Its SPY-excess lower bounds are
-39.65 and -45.01 bps. It is not a viable or replicated hypothesis.

Immutable evidence:

- Directory:
  `data/research/primary_v2_failure_attribution_swing_phase_v3_20260728`
- Request SHA-256:
  `ba56fab32f87778fd85307bb58c30a2e2669f42641c62c3e80057ee58547524e`
- Peak process working set: 0.569 GiB.

## Intraday Result

Strategy: `INTRADAY.VWAP_REVERSION.30M.V2`, meaning a long VWAP mean-reversion
position held for at most thirty regular-session minutes.

| Scope | Rows | Gross | Cost | Net | Net 95% lower bound | SPY excess | SPY 95% lower bound | Profit factor | Maximum drawdown |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Walk-forward | 40,881 | -0.42 bps | 10.00 bps | -10.42 bps | -11.18 bps | -10.12 bps | -10.55 bps | 0.394 | 26.16% |
| Unseen ticker | 9,590 | -0.46 bps | 10.00 bps | -10.46 bps | -11.44 bps | -10.14 bps | -10.72 bps | 0.370 | 25.81% |

The intraday failure occurs before model selection and before transaction
costs. Average gross return is negative in both scopes. Adding the exact 10 bps
round-trip cost makes the loss larger. No broad estimator change can convert
this setup population into a valid edge.

Zero of 23 predeclared cohorts passes a complete validation scope. The only
positive-net descriptive row is a one-observation high-volatility walk-forward
cohort; it fails the 200-row, 60-session, and confidence requirements and has no
unseen-ticker replication.

Immutable evidence:

- Directory:
  `data/research/primary_v2_failure_attribution_intraday_phase_v3_20260728`
- Request SHA-256:
  `20f1c163d3b5aa6545c7e8d31960208d1aa84b76c0855eb8ac3badab016ab3a2`
- Peak process working set: 0.340 GiB.

## Decision

No V2 model may be promoted, retained, or used as a V3 input. No observed cohort
authorizes a narrower V3 hypothesis.

The next research checkpoint must change setup mechanics with a separately
frozen causal rationale:

- Intraday: replace the unconditional VWAP-reversion population. A new setup
  needs causal entry confirmation and must demonstrate positive gross economics
  before model ranking.
- Swing: reduce turnover/cost exposure or alter entry and holding mechanics,
  then require positive net and benchmark-relative economics. Selecting the
  best cohort from this report is prohibited.
- Both: retain the same purged walk-forward, unseen-ticker, cost, SPY-relative,
  confidence, profit-factor, drawdown, lineage, and memory gates.
