# Market Predictor Implementation Guide

Status: current edge-rebuild path

Last updated: 2026-08-02

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
- Alpaca direct ticker news: the sole ticker catalyst source in the current estimator.
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
- `swing_features.py`: ordered technical and Alpaca-catalyst profiles.
- `catalyst_authority.py`: ticker catalyst decision and source-coverage authority.
- `catalyst_identity_rebind.py`: immutable V5 canonical identity rebind.
- `swing_materialization.py`: identical-population profile publication and replay.
- `swing_live.py`: latest closed-session features using shared semantics.

Swing decisions begin on `2019-07-09`. Earlier bars initialize indicators only. V9 is
invalid because managed labels were index-aligned incorrectly and is retained only as
V5 lineage. V10 is published and replayed; candidate v1 produced `no_candidate`
before economic evaluation because its 50-stock hard sector floor and 20% sector cap
were structurally incompatible on sessions with only four eligible sectors.

The implemented policy uses a within-sector target of 50 and a hard floor of 30.
Materialized rows persist sector peer count, sector rank eligibility, sector target
status, and ranking reliability weight. Groups with 30-49 peers remain eligible at
weight `decision_time_sector_peer_count / 50`; groups below 30 are ineligible. Portfolio selection targets
20% per sector, adapts to 25% with four represented sectors and 33.3% with three, and
skips sessions with fewer than three. Promotion gates use managed holding-aligned
benchmarks, full-calendar portfolio returns including cash days and overlapping
positions, doubled-cost portfolio stress, and the 33.3% active-sector ceiling. Live
processing excludes individual unavailable securities through the governed 5% limit.
V11 is published and replayed with 853,417 rows per profile, 604 securities, and
1,759 sessions. Candidate v2 trained six governed models and published an immutable
`no_candidate` result because no model passed economic gates in both temporal and
unseen-security validation; the locked test remained unopened.

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
- `edge_rebuild/serving.py`: promoted-bundle verification and strict prediction or
  abstention response.

## Current Workflow

1. Preserve the replayed V11 and candidate-v2 rejection authorities.
2. Attribute candidate-v2 failure using validation evidence only; do not open the
   locked test or weaken gates.
3. Preregister the next feature or estimator hypothesis before rerunning validation.
4. Keep API scoring disabled unless a promoted bundle verifies at load time.
5. Collect and freeze the future intraday holdout before any V3 holdout evaluation.

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
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\ruff.exe check --no-cache .
.\.venv\Scripts\mypy.exe --strict --python-version 3.14 src\market_predictor
.\.venv\Scripts\python.exe -m compileall -q src tests
git diff --check
git status --short --branch
```
