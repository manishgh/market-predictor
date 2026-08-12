# Market Predictor Implementation Guide

Status: current edge-rebuild path

Last updated: 2026-08-10

Read `AGENTS.md`, `docs/active_edge_rebuild_plan.md`, and
`docs/reviews/active_edge_rebuild_handoff.md` first. Command `--help` output and code
contracts are authoritative.

## Runtime Surfaces

- `market-predictor-collect`: provider I/O and immutable raw/canonical collection.
- `market-predictor-research`: authority publication, feature materialization,
  training, validation, and promotion research.
- `market-predictor-prod`: verified production release, outcome, drift, and API
  infrastructure.

No edge model is active. Production scoring must fail closed until a compatible
promoted atomic bundle exists.

## Source Roles

- Alpaca SIP/all bars: estimator market data.
- Alpaca direct ticker news: the sole permitted ticker catalyst estimator source for
  the separate event-driven family. The swing baseline is technical-only.
- SEC filings: current issuer authority and causal audit. Known zero filings remain
  distinct from unknown coverage. A separately ablated issuer-specific estimator
  profile is planned after causal collection and exact attachment are verified.
- Finviz Elite: screening and current metadata only.
- Verified global and sector sources: separate overlays only.
- Reddit and Seeking Alpha: removed and prohibited.

These roles are model contracts, not suggestions. SEC cannot enter the current
estimator without verified causal collection and attachment, a preregistered separate
profile, causal ablation, and a new model version. Other overlay data cannot enter an
estimator without the same governance.

## Package Map

### Shared controls

- `config.py`: environment-backed settings.
- `canonical/contracts.py`, `canonical/store.py`: immutable data contracts and hashes.
- `resources.py`, `heavy_jobs.py`: memory enforcement and one-heavy-process lease.
- `edge_rebuild/strategy_contract.py`, `edge_rebuild/contracts.py`: frozen strategy
  and readiness contracts.

### Point-in-time evidence

- `sp500_transitions.py`, `sp500_memberships.py`, `universe_identity.py`: historical
  membership and security identity.
- `history_collection.py`, `history_materialization.py`: Alpaca intraday transport and
  canonical history.
- `swing_history_collection.py`, `swing_daily_combination.py`: swing daily history.
- `benchmark_history.py`, `corpus_integrity.py`, `readiness.py`: benchmark and corpus
  admission checks.

### Swing path

- `labeling.py`: next-open ten-session managed and rank outcomes.
- `technical_relationships.py`, `cross_sectional.py`: causal technical and
  cross-sectional relationships.
- `swing_features.py`: causal technical transformations and label finalization.
- `swing_materialization.py`: the single catalyst-independent `technical_market`
  population and strict physical replay.
- `issuer_event_family_authority.py`, `issuer_event_precision_audit.py`: direct-issuer
  event, assignment, coverage, and reviewed precision authorities.
- `swing_event_ablation.py`: matched broker-action technical-only, event-only, and
  combined research datasets. Blocked families are absent and unknown coverage
  abstains.
- `swing_broker_specialists.py`: development-only rating-change and coverage-initiation
  capacity audits, six-experiment matrices, nested chronological selection, canonical
  portfolio economics, strict deterministic replay, and immutable rejection evidence
  without locked-test access.
- `swing_training.py`: four nested technical baseline ablations, bounded tree
  candidates, validation-only selection, exact feature-subset persistence, and
  immutable candidate/no-candidate publication.
- `swing_live.py`: latest closed-session features using shared semantics.

Swing decisions begin on `2019-07-09`. Earlier bars initialize indicators only.
Obsolete materializations and rejected estimators are not supported compatibility
paths; only current strict authorities may enter a new run.

The implemented policy uses a within-sector target of 50 and a hard floor of 30.
Materialized rows persist sector peer count, sector rank eligibility, sector target
status, and ranking reliability weight. Groups with 30-49 peers remain eligible at
weight `decision_time_sector_peer_count / 50`; groups below 30 are ineligible. Portfolio selection targets
20% per sector, adapts to 25% with four represented sectors and 33.3% with three, and
skips sessions with fewer than three. Promotion gates use managed holding-aligned
benchmarks, full-calendar portfolio returns including cash days and overlapping
positions, doubled-cost portfolio stress, and the 33.3% active-sector ceiling. Live
processing excludes individual unavailable securities through the governed 5% limit.
V12 is published and replayed with 853,417 technical rows, 604 securities, and
1,759 sessions. Prior candidate results remain rejection evidence because no model
passed economic gates in both temporal and unseen-security validation; the locked test
remained unopened.

A2 now defines the replacement baseline contract: nested momentum/volatility, trend,
pullback, and volume/liquidity groups are evaluated with regularized logistic models;
the full group also receives one XGBoost ranker and regressor. Each fitted estimator
persists its own ordered feature subset. Quality, profitability, investment,
valuation, and estimate-revision groups are blocked because no complete historical
point-in-time authority exists. No new real A2 candidate has been trained.

### Intraday path

- `intraday_selection.py`: point-in-time in-play selection.
- `volume_bars.py`, `intraday_features.py`: causal completed volume bars and the shared
  V2 feature builder.
- `intraday_labels.py`, `one_minute_coverage.py`: exact next-minute entry,
  thirty-minute outcome path, and holding-aligned SPY, QQQ, and point-in-time sector
  returns. Missing exact benchmark evidence abstains.
- `intraday_dataset.py`, `intraday_live.py`: immutable publication and live parity.
- `intraday_development.py`: V3 expected-net-return development and future-holdout
  controls.
- `intraday_rejection.py`: immutable V2 rejection evidence.

Intraday V2 is published but economically rejected. The later V3 z-score lineage is
invalid and cannot train or serve. New candidate evidence uses explicit outcome
contracts and named after-cost stock/SPY/QQQ/sector binary diagnostics; the shared
shuffled-label control must remain at chance.

### Overlay and serving paths

- `sources/sec.py`, `edge_rebuild/sec_filing_authority.py`: current SEC issuer
  authority/audit and zero-versus-unknown coverage semantics; future separately
  ablated estimator input after causal collection and attachment.
- `edge_rebuild/global_event_collection.py`,
  `edge_rebuild/global_event_authority.py`: separate global context overlay.
- `edge_rebuild/serving.py`: promoted-bundle verification, model-family/profile
  binding, estimator-specific feature slicing, and strict prediction or abstention.
- `prediction_service.py`: serves only a hash-verified promoted model family; the new
  broker-action specialists remain unavailable because A3.5 produced no candidate.

## Current Workflow

1. Preserve the replayed V12 technical authority and prior rejection evidence.
2. Use only the separately replayed broker-action authority admitted by both historical
   precision audits. Its internal family code is `analyst_revision`, but its records
   are broker upgrades, downgrades, coverage actions, and a small number of price-target
   changes rather than EPS-estimate revisions.
3. A3.5 combines upgrades and downgrades in a directional rating-change specialist,
   models coverage initiation separately, and keeps price-target/generic actions
   report-only. Its 12 development experiments produced no candidate, so the locked
   test remains unopened and serving remains disabled.
4. Keep API scoring disabled unless a promoted bundle verifies at load time.
5. Build the replacement A4 intraday cohort authority before any new holdout run.

## Data And Training Rules

- Feature availability must be at or before the decision time; label availability is
  after the complete outcome path.
- Unknown, partial, late, or unreconciled source coverage is not zero.
- No bar, timestamp, benchmark path, label, or coverage value is imputed.
- Whole-security exclusions remain at or below 5% of the filtered universe.
- Trainers consume verified immutable datasets and frozen chronological splits.
- Costs are applied once; stock and benchmark returns use the same interval.
- Evaluation includes calibration, net return, benchmark excess, drawdown, turnover,
  capacity, cost stress, and temporal/unseen-security stability.
- Failed models are immutable audit evidence, never serving fallbacks.

## Verification

Run only one collection, materialization, or training process at a time and keep its
peak RSS below the workload limit: 5 GiB for swing candidate training and
4 GiB for intraday and serving workloads.

```powershell
Set-Location C:\project\market-predictor
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\ruff.exe check --no-cache .
.\.venv\Scripts\mypy.exe --strict --python-version 3.14 src\market_predictor
.\.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --short --branch
```
