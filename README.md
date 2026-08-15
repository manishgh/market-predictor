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
- **Swing baseline:** estimator inputs are `technical_market` only. Alpaca ticker
  news is a confirmation/explanation overlay and does not alter baseline probability.
- **Swing event-driven:** Alpaca direct ticker news is the only currently permitted
  ticker catalyst source, subject to complete causal authority and specialist ablation.
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
- Live inference excludes individual missing or cold securities through the governed
  5% ceiling. Cached models are bound to the active contract, trust store, promotion
  policy, and model-size limit.
- Swing V12 is published and strictly replayed as one catalyst-independent authority:
  853,417 `technical_market` rows, 604 modeled securities, and 1,759 sessions from
  `2019-07-09` through `2026-07-08`.
- Prior swing candidates are rejection evidence only; no rejected or obsolete model
  is retained as a supported compatibility path.
- A2 replaces that experiment contract with four nested technical ablations
  (momentum/volatility, trend confirmation, pullback timing, volume/liquidity), followed
  by one full-feature XGBoost ranker and regressor. The implementation is verified, but
  no new real candidate has been trained and no performance result is claimed.
- A3 historical issuer-event and precision authorities currently admit only direct-
  issuer broker rating actions. The internal source code calls this family
  `analyst_revision`, but the records are predominantly rating upgrades, rating
  downgrades, and new/resumed coverage; they are not EPS-estimate revisions. Earnings,
  guidance, offerings, M&A, regulatory, product, and SEC families remain unavailable
  because their reviewed precision or source completeness did not pass.
- An identity bug in the first A3.4 comparison joined old and rebuilt security hashes
  directly, reducing 17,401 broker announcements to 50. The corrected immutable
  comparison aligns exact ticker and prediction timestamp, rejects conflicting CIKs,
  and publishes 27,087 prediction rows from 11,720 unique latest broker announcements
  in each of three datasets: technical-only, broker-action-only, and combined.
- A3.5 trained separate rating-change and coverage-initiation specialists. Each used
  technical-only, broker-action-only, and combined profiles with logistic and
  histogram-gradient-boosting estimators. Capacity passed, but all 12 development
  experiments failed the frozen 0.60 AUC/generalization and benchmark-relative
  economic gates in a separate inner selection window. No experiment qualified to
  open outer validation; the locked test also remained unopened and no model was emitted.
- Intraday V2 is published and replayable but economically rejected after costs.
  It is not serveable. The later V3 cross-sectional z-score lineage is invalid and
  prohibited because its declared inputs lacked a valid contemporaneous cohort
  transformation.
- A4.1 now provides bounded, page-resumable Alpaca SIP trade and quote collection with
  immutable request/job/attempt/raw-page lineage, exact session bounds, failed-attempt
  hashing, path isolation, and a 4 GiB hard memory limit. The corrected collection
  plan contains 43,226 selected stock-sessions and 86,452 jobs; end-of-session bar
  coverage is metadata, not an earlier-decision selector. A two-job live Alpaca probe
  completed with zero failures at 0.35 GiB peak RSS. It is intentionally incomplete
  and cannot authorize microstructure features or training.
- A4.3's separately identified bar-only intraday authority is complete: 794 sessions,
  501 tickers, 3,095,688 rows, and 1,365,015 eligible rows from verified SIP/all one-
  and five-minute bars, SPY, QQQ, sector ETFs, and point-in-time membership. Its
  hash-bound row audit passes with zero causal, schema, duplicate, ATR, or prohibited-
  feature violations. A4.4 trained separate continuation and long-reversion bar-only
  baselines on this authority. Both published immutable `no_candidate` evidence: the
  best audited positive-return ROC-AUC was 0.510/0.516 for continuation and 0.513/0.508
  for reversion across seen/unseen securities, with negative after-cost economics in
  the controlling scopes. No candidate model was written and the future holdout stayed
  closed. Trade/quote features remain prohibited until a complete A4.1/A4.2 authority
  exists.
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

- **Alpaca premium:** SIP market bars are estimator data. Direct ticker news is an
  estimator input only for a separately governed event-driven model family.
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

The local research workbench inventories four configured experiments: baseline and
event-driven variants for swing and intraday. It reports the real training
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
.\.venv\Scripts\python.exe -m pytest tests -q
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
