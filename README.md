# Market Predictor

Market Predictor produces evidence-backed prediction intelligence for swing and
intraday research. It owns market and catalyst data curation, causal feature
construction, model training, validation, prediction, outcome maturation, and
model governance.

It does **not** own alerts, broker execution, orders, positions, portfolio risk,
or final sizing. Those responsibilities belong to `trading_flow`.

This system is not deployed. There are no supported legacy models, schemas,
fallbacks, or compatibility paths. Only artifacts passing the current contracts
may become serving candidates.

## Current State

- Branch: `r3-lineage`.
- Active program: ER1A edge rebuild.
- Swing: the seven-year causal panel is published and immutable. The frozen
  equal-weight technical benchmark failed and is rejected, but it does not block
  preregistered model training. The frozen temporal audit requires 2,033 XNYS
  sessions and identifies one exact missing range: 279 sessions from 2018-05-29
  through 2019-07-08. The outcome-blind acquisition audit found that all 83
  retained official S&P release files fail their declared byte hashes, emitted
  zero Alpaca units, and requires immutable source reacquisition before the
  causal point-in-time history can be extended. Final model training remains
  blocked until that evidence is repaired and the panel is rebuilt.
- Intraday: verified canonical SIP five-minute corpus contains 38,586,501 bars,
  814 sessions, and 1,104 symbols. Volume-bar setup construction and selective
  one-minute executable paths remain pending.
- Promotion: no swing or intraday model is promoted or serveable.
- Runtime: no alerting or execution behavior exists in this repository.

Read these documents in order:

1. [Engineering covenant](AGENTS.md)
2. [Active edge-rebuild plan](docs/active_edge_rebuild_plan.md)
3. [Active handoff](docs/reviews/active_edge_rebuild_handoff.md)
4. [Prediction architecture](docs/catalyst_confirmation_architecture.md)
5. [Implementation guide](docs/implementation_guide.md)
6. [Model training and validation protocol](docs/model_training_validation_protocol.md)
7. [Source quantitative trading plan](docs/references/comprehensive_quantitative_trading_model_implementation_plan_intraday_and_swing.pdf)

Additional retained contracts and evidence:

- [Known strategy sequence](docs/known_strategy_expansion_sequence_2026-07-26.md)
- [Strategy traceability](docs/strategy_execution_traceability.md)
- [TradingFlow integration boundary](docs/trading_flow_integration_plan.md)
- [Azure deployment plan](docs/azure_deployment_plan.md)

## Data Sources

- **Alpaca premium:** SIP market bars, ticker universe, and primary news.
- **Reddit API:** community attention and ticker discussion, with strict symbol
  relevance checks.
- **Seeking Alpha through RapidAPI:** SA news, analysis, earnings, financials,
  and quant/rating snapshots. Credentials belong in environment variables.
- **SEC EDGAR:** filing events aligned by SEC acceptance time.
- **Finviz Elite:** candidate screening and current market metadata; it is not a
  substitute for point-in-time historical membership.
- **Market context:** SPY, QQQ, sector ETFs, and explicitly global events.

Historical publication-time news backfills are research-only when historical
first-observed timestamps are unavailable. They cannot be relabeled as live
observations.

## Authoritative Local Data

Current protected inputs include:

- `data/raw/swing_daily_sip_sp500_pit_20190709_20260708_v3`
- `data/raw/alpaca_news_20190709_20210708_v1`
- `data/raw/alpaca_news_20210709_20260708_v1`
- `data/raw/alpaca_news_intraday_candidates_20230410_20260708_v1`
- `data/canonical/swing_memberships_verified_20190709_20260708_v2.parquet`
- `data/canonical/edge_rebuild_intraday_5m_20260731`
- `data/raw/edge_rebuild_selected_session_5m_20260731`
- `data/research/intraday_universe_selection_20230410_20260708_v2`
- `data/universe/sp500_point_in_time_20190709_20260708_v3.parquet`
- `data/research/edge_rebuild_swing_temporal_manifest_20260731_v1`

Generated feature matrices are reproducible working data and are not retained
after rejection or supersession. Raw provider archives are retained only when
they have unique, expensive-to-recreate coverage and sufficient provenance to
pass the current canonical validators.

## Setup

Requires Python 3.11 or newer. The verified local environment currently uses
Python 3.14.

```powershell
Set-Location C:\project\market-predictor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Configure credentials only in `.env` or a managed secret store. Never place
keys or passwords in source, tests, reports, screenshots, or command arguments.

Use the command help as the authoritative CLI surface:

```powershell
market-predictor-collect --help
market-predictor-research --help
market-predictor-api --help
```

## Verification

Run one heavy process at a time and keep working-set memory below 4 GiB.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check --no-cache .
.\.venv\Scripts\mypy.exe --strict --python-version 3.14 src\market_predictor
.\.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --short --branch
```

## Core Invariants

- Features are usable only at or after their recorded availability timestamp.
- News relevance, publication time, first-observed time, and source coverage are
  separate evidence.
- Membership, ticker identity, corporate actions, and benchmarks are
  point-in-time.
- Security-specific missing or unverifiable data excludes the complete security
  and continues while at most 5% of the filtered universe is lost. Benchmark or
  market-wide session gaps are never waived.
- Swing and intraday labels use the shared target/stop/timeout evaluators.
- Costs are applied exactly once and benchmark comparisons use the same holding
  interval.
- Validation is time ordered, purged, and embargoed; random cross-validation is
  prohibited.
- Catalyst starts as confirmation and ranking context unless causal ablation
  proves it improves the estimator.
- A model is not actionable merely because tests pass. It must pass economic,
  calibration, drawdown, unseen-security, shadow, and promotion gates.

## Disclaimer

This repository is research and prediction tooling, not investment advice and
not an automated trading system.
