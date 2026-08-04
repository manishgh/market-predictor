# Active Edge Rebuild Plan

Status: active

Last updated: 2026-08-02

Repository: `C:\project\market-predictor`

Branch: `r3-lineage`

This is the only active execution plan. Exact artifact state is recorded in
`docs/reviews/active_edge_rebuild_handoff.md`; statistical rules are defined in
`docs/model_training_validation_protocol.md`.

## Objective And Boundary

Build causal prediction intelligence for:

- **Swing:** ten-session stock direction, managed return, and excess return against
  SPY, QQQ, and the point-in-time sector benchmark.
- **Intraday:** thirty-minute managed outcome from completed intraday evidence.

This repository produces predictions, abstentions, explanations, benchmark
comparisons, and matured outcomes. It does not produce alerts, orders, positions,
portfolio risk, or execution instructions. `trading_flow` may consume only a promoted,
versioned prediction API.

## Frozen Data Policy

1. Alpaca is the sole ticker catalyst source in the current estimator.
2. The SEC authority distinguishes known zero filings from unknown coverage. SEC is
   planned as a separately ablated issuer-specific estimator profile only after
   causal collection and exact decision-time attachment are verified. Finviz and
   verified global or sector sources remain overlay/audit inputs and cannot silently
   enter an estimator feature vector.
3. Reddit and Seeking Alpha are removed and prohibited.
4. Swing decisions begin on `2019-07-09`. Earlier bars are warm-up only.
5. Features and source coverage must be available by the decision time. Unknown
   coverage is null, not zero.
6. Identity, membership, ticker changes, sectors, and benchmarks are point-in-time.
7. Alpaca market bars use SIP and `adjustment=all`; missing bars are not imputed.
8. Sparse gaps invalidate affected windows. Whole-security exclusion is capped at 5%
   of the filtered universe; benchmark failures are not waived.
9. Training and evaluation are chronological, purged, embargoed, and cost-aware.
10. One heavy process runs at a time. Swing candidate training has a 5 GiB
    hard process limit; intraday and serving workloads retain 4 GiB limits.

## Verified Artifact State

| Artifact | State | Evidence |
| --- | --- | --- |
| Catalyst V5 identity rebind | published and replayed | 377,778 exact decision matches; 6,359 coverage rows; 604/604 target securities |
| Swing V9 | invalid; lineage only | managed-label index alignment corrupted labels; retained only because V5 binds it as target lineage |
| Swing V10 | published; candidate v1 structurally rejected | `no_candidate` occurred before economic gates because the 50-stock hard sector floor often yielded four sectors while the 20% cap required five |
| Swing V11 | published and replayed | 853,417 rows/profile; 604 securities; 1,759 sessions; 640,107 rank-eligible rows |
| Swing candidate v2 | published `no_candidate` | six governed candidates trained; none passed both temporal and unseen-security economic gates; locked test unopened |
| Swing candidate v3 | published `no_candidate` | overlay constraints yielded too few sessions and failed economic gates; locked test unopened |
| Intraday V2 | published and rejected | economically failed after costs; not serveable |
| Intraday V3 | future work only | development implementation reserved for a genuinely future holdout; no holdout run |
| Promoted serving bundle | absent | API must fail closed |

## Model Semantics

### Swing

The active hypothesis is ten-session sector-residual momentum after a controlled
pullback and trend reclaim. Required technical evidence includes 20/60-session
relative strength, SMA50/SMA200 state and slope, pullback/reclaim state, volatility,
liquidity, capacity, SPY/QQQ context, and the point-in-time sector benchmark. Direct
Alpaca ticker catalyst may be evaluated in the catalyst profile. The current estimator
is Alpaca-only. SEC is planned as a separate issuer-specific profile after causal
collection and attachment, preserving known zero versus unknown coverage. Finviz and
global context remain separate overlays.

Within-sector ranking has a preferred target of 50 peers and a hard floor of 30.
Every row persists the sector peer count, sector rank eligibility, whether the target
was met, and ranking reliability weight. Groups of 30-49 peers remain eligible with
weight `decision_time_sector_peer_count / 50`; groups below 30 are ineligible. Portfolio construction
targets a 20% maximum sector weight, adapts to 25% with four represented sectors and
33.3% with three, and skips sessions with fewer than three. Economic gates are
unchanged.

Entry is the next exact exchange-session open. Target, stop, timeout, costs, and all
benchmark returns use the same executable interval.

### Intraday

The active hypothesis is a thirty-minute VWAP exhaustion reversal. Evidence is built
from completed causal intraday bars with next-minute execution, exact one-minute path
labels, stock/market/sector context, and explicit abstention. Catalyst is a
confirmation or ranking overlay unless a preregistered causal ablation proves
estimator value.

Intraday V2 remains a valid rejected artifact. V3 cannot be evaluated until a new
holdout beginning on or after `2026-07-09` is collected and frozen without development
feedback.

## Current Sequence

1. Preserve V11 and candidate-v2 rejection evidence without opening the locked test.
2. Complete validation-only failure attribution for temporal versus unseen-security
   behavior and the unstable catalyst increment.
3. Preregister any next feature or estimator hypothesis before another validation
   run; do not tune gates to rescue candidate v2.
4. Run full tests, Ruff, strict mypy, compileall, replay, diff, and the 5 GiB
   swing-training memory gate before checkpointing.
5. Collect the future intraday holdout and evaluate V3 only after its observation
   window is complete.

## Promotion Gates

A model is not promoted because materialization or training succeeds. Promotion
requires:

- immutable source, feature, label, split, and model lineage;
- exact batch/live ordered-feature parity;
- chronological and unseen-security stability;
- calibration and ranking value;
- positive net economics after costs with acceptable drawdown and capacity;
- regime, sector, and market-cap stability;
- prospective shadow evidence;
- a hash-verified atomic serving bundle.

Until then, production scoring and prediction API paths fail closed. Rejected models
are audit evidence, never fallbacks.

## Completion Checklist

- [x] Remove Reddit and Seeking Alpha from the active system.
- [x] Freeze Alpaca as the only ticker catalyst estimator source.
- [x] Enforce `2019-07-09` as the first swing model decision date.
- [x] Publish and replay catalyst V5 identity rebind.
- [x] Publish and economically reject intraday V2.
- [x] Publish and replay corrected swing V10.
- [x] Record the V10 candidate v1 structural `no_candidate` result before economics.
- [x] Complete and replay swing V11 under the approved flexible ranking policy.
- [x] Train and evaluate swing candidate v2 with corrected holding-aligned benchmark,
  full-calendar portfolio-bootstrap, doubled-cost path, and active-sector gates.
- [x] Complete repository-wide verification and memory audit.
- [x] Train and evaluate swing candidate v3 with Catalyst (SEC + Alpaca) confirmation overlay. (Failed economic gates).
- [x] Train and evaluate swing candidate v4 (Event-Driven Specialist) with global catalyst peer group. (Failed economic gates).
- [x] Extract components from `swing_training.py` into cohesive modules without changing behavior.
- [x] Materialize Swing V12 dataset with advanced technical indicators and appropriate cross-sectional scaling.
- [x] Train and evaluate swing candidate v5 using V12 features (Failed economic gates / `no_candidate`).
- [x] Materialize Intraday V3 dataset with MACD, EMA distance, and SMA distance features.
- [x] Evaluate Intraday V3 Technical Features for target-hit predictive efficacy (AUC ~0.50-0.51).
- [x] Update final handoff, commit, and push the checkpoint.
- [ ] [environment-pending] Collect and freeze a genuinely future intraday V3 holdout. (Blocked: requires canonical membership and raw market data for 2026-07-09 onwards, which is unavailable in the current environment.)
- [ ] Promote only a model that passes every gate.
