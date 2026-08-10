# Market Predictor

Market Predictor builds causal prediction intelligence for swing and intraday
research. It owns data curation, feature construction, model training, temporal
validation, outcome evaluation, and model governance.

It does **not** own alerts, orders, positions, portfolio risk, or execution. Those
responsibilities belong to `trading_flow`.

The system is not deployed. There are no supported legacy models or compatibility
paths. Serving fails closed until a model passes validation and is published in a
hash-verified promoted bundle.

## Verified State

- Active development branch: `er-intraday-refactoring`.
- **Current ticker catalyst estimator source:** Alpaca news only.
- **SEC filings:** the current estimator does not consume SEC features. The SEC
  authority distinguishes verified no-filing observations from unknown coverage.
  After causal collection and exact issuer attachment, SEC will be evaluated as a
  separate issuer-specific estimator profile rather than treated as a permanent
  overlay-only source.
- **Other overlay/audit sources:** Finviz and verified global or sector sources.
  They cannot be attributed to a ticker as direct issuer news.
- Reddit and Seeking Alpha are removed and prohibited from collection, features,
  training, and serving.
- Swing model decisions begin on `2019-07-09`. Earlier market bars are indicator
  warm-up only and cannot produce model features, labels, train rows, validation
  rows, or test rows.
- Catalyst V5 identity rebind is published and replayed: 377,778 exact decision
  matches and 6,359 source-coverage rows across all 604 target securities.
- Swing V9 is invalid because managed labels were corrupted by Pandas index
  alignment. It is retained only because V5 binds it as target-lineage evidence.
- Swing V10 materialization replayed, but candidate v1 returned `no_candidate`
  before economic evaluation. The cause was structural: the 50-stock hard sector
  floor frequently left only four eligible sectors while the 20% sector cap required
  at least five. This result does not establish that the model failed economic gates.
- The implemented replacement policy keeps a within-sector ranking target of 50 and
  uses a hard floor of 30. It persists sector peer count, rank eligibility, target
  status, and ranking reliability weight. Groups with 30-49 peers remain eligible
  with weight `decision_time_sector_peer_count / 50`.
- Sector allocation targets 20%; it adapts to 25% when only four sectors are
  represented and 33.3% when only three are represented. Sessions with fewer than
  three represented sectors are skipped.
- Economic acceptance uses managed holding-aligned benchmarks, includes cash days
  and overlapping positions in the portfolio bootstrap, applies doubled costs to
  the full daily path, and rejects active sector exposure above 33.3%.
- Live inference excludes individual missing, cold, or catalyst-incomplete securities
  through the governed 5% ceiling. Cached models are bound to the active contract,
  trust store, promotion policy, and model-size limit.
- Swing V11 is published and strictly replayed: 853,417 rows per ablation
  profile, 604 modeled securities, 1,759 sessions from `2019-07-09` through
  `2026-07-08`, and 640,107 rank-eligible rows.
- Swing candidate v2 trained six governed logistic/HGB ablations and returned
  `no_candidate`. AUC reached about 0.55-0.57, but no candidate had a positive
  lower confidence bound for calendar, portfolio-daily, doubled-cost, and
  holding-aligned benchmark economics in both validation scopes. The locked
  test was not read.
- Intraday V2 is published and replayable but economically rejected after costs.
  It is not serveable. The later V3 cross-sectional z-score lineage is invalid and
  prohibited because its declared inputs lacked a valid contemporaneous cohort
  transformation.
- Intraday label schema V2 requires exact stock, SPY, QQQ, and point-in-time sector
  returns over one executable entry-to-managed-exit interval. Missing benchmark
  evidence abstains. Swing and intraday evaluations report named after-cost binary
  outcomes and a shuffled-label AUC control; AUC remains diagnostic only.
- No swing or intraday model is promoted. The prediction API therefore returns no
  model prediction and must fail closed.

Read these documents in order:

1. [Engineering covenant](AGENTS.md)
2. [Active edge-rebuild plan](docs/active_edge_rebuild_plan.md)
3. [Active handoff](docs/reviews/active_edge_rebuild_handoff.md)
4. [Prediction architecture](docs/catalyst_confirmation_architecture.md)
5. [Implementation guide](docs/implementation_guide.md)
6. [Training protocol](docs/model_training_validation_protocol.md)

## Data Boundaries

- **Alpaca premium:** SIP market bars and direct ticker news used by estimators.
- **SEC EDGAR:** current issuer authority and causal audit source; planned as a
  separately ablated issuer-specific estimator profile after causal collection and
  attachment are verified.
- **Finviz Elite:** candidate screening and current metadata only; never historical
  membership authority or ticker-news estimator input.
- **Global and sector sources:** separately identified context overlays only.
- **Benchmarks:** SPY, QQQ, and point-in-time sector ETFs.

Unknown coverage is not converted to zero. Historical publication-time backfills are
research evidence and cannot be represented as prospectively observed events.

## Setup

Requires Python 3.11 or newer. The verified local environment currently uses Python
3.14.

```powershell
Set-Location C:\project\market-predictor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Credentials belong only in `.env` or a managed secret store.

```powershell
market-predictor-collect --help
market-predictor-research --help
market-predictor-prod --help
```

### Research Workbench

The local research workbench inventories four configured experiments: swing and
intraday, each with and without catalyst features. It reports the real training
state of every experiment. A `no_candidate` result remains unavailable rather than
being replaced by a fallback model.

```powershell
.\.venv\Scripts\python.exe -m uvicorn `
  market_predictor.research_api.server:app `
  --host 127.0.0.1 `
  --port 8123
```

Open `http://127.0.0.1:8123`. The workbench is non-actionable and does not alter the
production API. Scoring reads only integrity-checked snapshots registered through
the canonical live feature store. Missing, stale, non-finite, or schema-incomplete
features cause a readiness error; the workbench does not substitute zero values or
proxy ETF approximations for missing cross-sectional features.

## Verification

Run one heavy process at a time. Swing candidate training has a 5 GiB hard
process limit; intraday and serving workloads retain their 4 GiB limits.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check --no-cache .
.\.venv\Scripts\mypy.exe --strict --python-version 3.14 src\market_predictor
.\.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --short --branch
```

## Core Invariants

- Features are usable only after their recorded availability time.
- Membership, ticker identity, corporate actions, sectors, and benchmarks are
  point-in-time.
- Market bars are Alpaca SIP with `adjustment=all`; bars and timestamps are not
  imputed.
- Sparse gaps invalidate affected windows. Whole-security exclusions cannot exceed
  5% of the filtered universe; benchmark and market-wide failures are never waived.
- Costs are applied once and benchmarks use the same executable holding interval.
- Validation is chronological, purged, and embargoed. Random cross-validation is
  prohibited.
- Passing software tests does not promote a model. Economic, calibration, drawdown,
  stability, future-shadow, and bundle-verification gates must also pass.

This repository is prediction research tooling, not investment advice or an automated
trading system.
