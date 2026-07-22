# Senior ML Review Context - 2026-07-22

## Purpose

This handoff anchors two independent senior-engineer reviews of Market Predictor. The reviewers must inspect the current implementation and produce evidence-based findings without changing production code, model artifacts, or datasets.

## Review Base

- Repository: `C:\project\market-predictor`
- Branch: `main`
- Base commit: `48ac758` (`Resolve repository-wide lint and type debt`)
- Previous production-serving checkpoint: `b53ab2e` (`Complete production serving C6`)
- Previous model checkpoint: `feccb62` (`Record C5 checkpoint`)
- Worktree state before this handoff: clean

## Product Boundary

Market Predictor is prediction-only. It owns point-in-time data preparation, feature generation, model training and validation, promotion evidence, inference, prediction replay, and prediction telemetry for swing and intraday horizons.

TradingFlow owns watchlists, alerts, portfolio and risk policy, order decisions, execution, and the live trading UI. A prediction is decision support, not an instruction to trade. Reviewers must flag any code that weakens this boundary.

## Current Checkpoint

- C1 through C6 implementation is complete.
- C7 repository-wide lint and strict type cleanup is complete.
- The most recent verification passed 199 tests.
- Repository-wide Ruff checks passed.
- Strict mypy checks passed for 93 source files.
- Python compile and credential scans passed.
- No canonical swing or intraday model has yet been trained and promoted from the final canonical production datasets. Passing implementation tests is not evidence of market edge.
- A real Azure release rehearsal remains pending.
- Dependency locking was explicitly deferred.

## Mandatory Constraints

- Review only. Do not alter production source, tests, configuration, datasets, registry entries, or model artifacts.
- Each reviewer may create or update only the report assigned below.
- Do not commit, push, promote a model, call paid APIs, collect live data, or run external network jobs.
- Do not launch full training, large backtests, or memory-unbounded analysis.
- Keep total local memory use below 4 GiB. Prefer static inspection and narrowly scoped tests.
- Never print, copy, inspect, or include secret values. Credentials belong in local environment configuration only.
- Treat all market, news, filing, universe, and benchmark joins as point-in-time data unless the implementation proves otherwise.
- Do not infer correctness from documentation. Trace claims to executable code, tests, schemas, and persisted evidence.

## Reviewer Assignments

Review launch state:

- Statistical reviewer: `Sartre` (`019f891b-80ae-7290-b833-aa7d15f59d62`), running independently
- Production reviewer: `Halley` (`019f891b-9e2d-7af2-a12a-ef8d4be991be`), running independently
- Launch date: 2026-07-22
- Review base remains commit `48ac758`; later repository changes must not be treated as part of this review pass.

### Statistical and ML Validity

Report: `docs/reviews/ml_statistical_validity_review_2026-07-22.md`

Audit target and feature timing, news/candle alignment, leakage, splits and embargoes, survivorship bias, calibration, probability semantics, selection bias, trading-cost assumptions, benchmark-relative economics, drawdown, regime behavior, ablations, promotion gates, replay validity, drift, and evidence strength.

### Production Architecture and Serving

Report: `docs/reviews/ml_production_architecture_review_2026-07-22.md`

Audit data contracts, canonical schemas, batch/live feature parity, atomicity, model registry and release integrity, inference API semantics, readiness, telemetry, failure isolation, concurrency, memory bounds, configuration and secret handling, dependency boundaries, Azure deployability, and the TradingFlow integration boundary.

## Review Schedule

Each reviewer should work independently through three phases:

1. Inventory: map relevant modules, schemas, tests, model manifests, and documented contracts.
2. Deep review: trace high-risk paths and run only focused, bounded verification where needed.
3. Report: write a prioritized, actionable review with concrete evidence and an implementation sequence.

The reviewers run outside the main implementation loop. Their reports are inputs to a later triage and remediation checkpoint; they are not authorized to fix findings during this pass.

## Required Finding Format

Lead with findings ordered by severity:

- Severity: `P0`, `P1`, or `P2`
- Location: concrete file and line reference
- Defect or risk: what is wrong or unproven
- Impact: how it can bias predictions, invalidate promotion, break serving, or harm operations
- Evidence: code path, test, schema, or artifact inspected
- Remediation: specific code or design change
- Verification: test, audit, or acceptance criterion proving the remediation

The report must also include:

- Executive verdict
- Confirmed strengths
- Missing evidence and unresolved assumptions
- Promotion or deployment blockers
- Ordered remediation plan

Generic ML or architecture advice without repository evidence is out of scope.

## Starting Points

Reviewers should first read `AGENTS.md` if present, `README.md`, `docs/production_ml_rebuild_plan.md`, architecture and integration documents, model and data contracts, canonical feature builders, split and audit code, promotion code, serving code, and their associated tests.
