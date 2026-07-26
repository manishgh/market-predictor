# Strategy Execution Traceability

Date: 2026-07-26

## Authority

The design authority is
[`known_strategy_expansion_sequence_2026-07-26.md`](known_strategy_expansion_sequence_2026-07-26.md).
The machine-readable execution authority is
[`strategy_execution_ledger.json`](strategy_execution_ledger.json).

This document explains the control process. It does not duplicate live checkpoint
status. Read and validate the JSON ledger for current state.

## Validation Command

From the repository root:

```powershell
market-predictor-research validate-strategy-execution-ledger `
  --ledger docs/strategy_execution_ledger.json `
  --repository-root .
```

An editable environment that has not regenerated console entry points can invoke
the same command through the module:

```powershell
.\.venv\Scripts\python.exe -m market_predictor.research_cli `
  validate-strategy-execution-ledger `
  --ledger docs/strategy_execution_ledger.json `
  --repository-root .
```

Validation fails when:

- the design file differs from its recorded SHA-256;
- the design commit is absent from the recorded remote-tracking branch;
- any design strategy, risk component, meta component, or `KS0`-`KS9`
  checkpoint is missing from the ledger;
- the ledger contains an item not present in the design;
- IDs are duplicated, misordered, malformed, or assigned to unknown checkpoints;
- blocked or deferred work lacks an explicit blocker;
- evaluated catalog state lacks an evidence path;
- an evidence path escapes the repository, is missing, or has the wrong hash;
- a completed checkpoint has an unpassed gate, unpassed verification, missing
  command, missing evidence, unhashed evidence, missing closure, or an unpushed
  closure commit.

Git verification uses local remote-tracking refs and makes no network call.
The checkpoint workflow must run `git push` before the ledger can validate a
completed closure.

## Checkpoint Traceability

| Checkpoint | Named work | Required evidence class |
|---|---|---|
| `KS0` | Catalog, hypotheses, experiment budget, shared assumptions | Contract tests, frozen registries, fold/cost/capacity/retirement policies |
| `KS1` | Catalyst lineage | Relation replay, event/sentiment/decision hashes, relevance and timing poison tests |
| `KS2` | Strategy-specific labels | Exact path replay, benchmark reconciliation, look-ahead and cost poison tests |
| `KS3` | Swing specialists | Frozen-fold candidates, temporal/ticker holdout, economic and ablation reports |
| `KS4` | Intraday specialists | Exact one-minute paths, session strata, spread/cost and catalyst-overlay ablations |
| `KS5` | Distributional/path models | Quantile and event calibration, incremental selection/abstention economics |
| `KS6` | Volatility sidecars | QLIKE, calibration, convergence, memory, and downstream risk/drawdown ablation |
| `KS7` | Regime routing/meta-labelling | Causal regime audit and routed-versus-unrouted economics |
| `KS8` | Data-dependent admission | Point-in-time peer, quote/depth, or sequence-model sample-size evidence |
| `KS9` | Promotion/API/TradingFlow | Attestation, atomic bundle, response identity, outcome maturation, boundary tests |

## KS0 Frozen Contracts

KS0 uses four authoritative inputs:

- `docs/strategy_hypothesis_registry.json`: exactly one bounded H1 claim for
  every named strategy, risk model, and meta model;
- `configs/strategy_research_governance.toml`: experiment budget, comparison
  dimensions, validation scopes, retirement triggers, memory ceiling, and
  hash-bound canonical configuration files;
- `docs/reference_model_inventory.json`: generic historical models that remain
  non-serving references and have no named-strategy identity;
- `market_predictor.execution_policy.v1`: the existing content-addressed cost,
  participation, stress, and capacity contract.

The research hypothesis registry is upstream of the immutable promotion
hypothesis registry. A research hypothesis defines the claim, eligible
population, outcome, comparator, and falsification rule before development. A
promotion hypothesis is created only after candidate and baseline artifacts
exist and before untouched shadow evidence is observed.

KS0 permits no more than 12 development experiments per strategy version, four
estimator families, three feature profiles, two selection policies, and one
shadow attempt. These are upper bounds, not targets. Inspecting additional
variants after the budget is exhausted requires retirement or a new semantic
strategy version with an independently justified hypothesis.

All five generic model families recorded in the reference inventory have
`strategy_id = null` and `serving_eligible = false`. A historical filename,
model card, or candidate metric cannot grant strategy identity or serving
eligibility.

## Named Strategy Coverage

The ledger includes all design strategies:

- Swing: Cross-Sectional Momentum, Time-Series Momentum, Catalyst Drift, PEAD,
  Short-Term Reversal, Breakout Expansion, Sector-Neutral Residual Momentum,
  and Pairs Reversion.
- Intraday: Opening Range Breakout, Gap Continuation, Gap Fade, VWAP
  Continuation, VWAP Reversion, Intraday Momentum, Short-Horizon Reversal,
  and Order-Flow Imbalance.

The ledger also includes the non-directional risk sequence:

1. `RISK.REALIZED_VOLATILITY.60M.V1`
2. `RISK.HAR_RV.60M.V1`
3. `RISK.GARCH.60M.V1`
4. `RISK.GARCH.5D.V1`
5. `RISK.EGARCH.60M.V1`

GARCH-family models forecast conditional variance and expected risk. They do not
become directional strategies. Adoption requires out-of-sample improvement over
realized volatility and ATR, plus downstream selection, abstention, or drawdown
improvement. Intraday fitting must remove time-of-day seasonality, separate
overnight gaps, and remain below the 4 GiB process limit.

The meta-model inventory includes Quantile Return, Competing Risks,
Meta-Labelling, and Regime-Switching/Mixture of Experts.

## Closure Procedure

For one checkpoint at a time:

1. Change its ledger status to `in_progress`.
2. Implement only its declared scope and contracts.
3. Add evidence artifacts under stable repository paths or record immutable
   generated report paths that are intentionally versioned.
4. Compute and record SHA-256 for every closure artifact.
5. Link every evidence ID to at least one gate or verification item. Orphan
   evidence is prohibited.
6. Mark gates and verification items `passed` only after their commands produce
   the recorded evidence.
7. Run focused tests, poison tests, Ruff, strict mypy, the full unit suite,
   `git diff --check`, memory checks, and process cleanup.
8. Commit and push the implementation and evidence.
9. Record the pushed commit, remote ref, UTC closure time, and factual summary.
10. Validate the ledger again. Start the next checkpoint only after validation
    reports the current checkpoint as completed.

Model rejection does not mean checkpoint failure. A reproducible rejected
candidate is valid evidence when the implementation and audit gates pass.
Promotion, profitability, and external deployment state remain independent facts.

## Change Control

Changing a strategy hypothesis, setup eligibility, label semantics, execution
policy, or horizon requires a new semantic strategy version. Do not edit evidence
to make an old ID mean something new.

Changing the design plan requires:

1. an explicit requirement or reproducible conflict;
2. a reviewed design commit;
3. a new plan hash and commit binding in the ledger;
4. exact catalog/checkpoint reconciliation;
5. a dedicated Git checkpoint.

The validator prevents accidental omission. It cannot authorize unsupported
market claims; those still require the model and economic evidence declared by
each checkpoint.
