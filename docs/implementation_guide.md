# Market Predictor Implementation Guide

Status: current edge-rebuild path

Last updated: 2026-09-08

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

### Long-Only Swing Research

`configs/swing_research.toml` and `swing/contracts/research.py` define the new
ten-session SPY-excess research objective. Their separate identity preserves the
existing strategy/data hashes. AUC is diagnostic, not the economic objective.
The retained automatic-final-test trainer refuses the already exposed July 2025
through June 2026 interval before panel access. New development return-model
training is a later step in the active plan, not enabled by changing test dates.

Run `market-predictor-research audit-swing-research-evidence --root .` from the
repository root to verify the frozen metadata inventory and ordered technical
feature contract. The command prints JSON, reads no feature/outcome rows, performs
no provider requests, and does not fit a model. A metadata match is not a fresh
raw-source replay or proof of economic performance.

To verify funded accounting against retained initial-fit data:

```powershell
.\.venv\Scripts\market-predictor-research.exe audit-swing-accounting-control --root . --output-directory data/reports/swing_accounting_control
```

`configs/swing_accounting_audit.toml` freezes the source identities, initial-fit
window and `return_20d_xs_rank` control. The command acquires the workspace heavy-job
lease, reads projected monthly columns, freezes selection before outcomes, and
writes `_manifest.json` in a new immutable output directory. It does not download,
fit a model, select by outcome completeness, or inspect validation/test outcomes.

The report binds source/config hashes, selected decision IDs, calendars and the
base/stress ledgers. `swing/evaluation/accounting.py` supplies SPY comparisons around
the canonical ledger in `swing/evaluation/ledger.py`. The frozen control orchestrator
is `research/swing_accounting_control.py`; shared resampling belongs to
`modeling/resampling.py`. No compatibility imports from the old ledger owner remain. Missing selected
outcomes, calendar mismatches, changed inputs and invalid cash/P&L fail the audit.
Successful execution still reports `price_basis_pending`: source total-return
reconciliation is not established, so the report cannot authorize promotion.

The retained run at `data/reports/swing_accounting_control/_manifest.json` records
70 incomplete selected outcomes among 30,525 stock-days. It preserves the frozen
selection, input hashes and calendars, but contains no accounting/performance result.
The command returns nonzero for a blocked report. Do not overwrite that receipt or
drop incomplete outcomes and rerun the same control as if its population were unchanged.

## Source Roles

### Retained Holding-Identity Preflight

The user-approved research cohort excludes 45/631 security identities (7.13%)
under a 10% cap. Raw data and old controls are unchanged. Before rebuilding
features, inspect the retained 586 securities' membership-based holding windows:

```powershell
.\.venv\Scripts\market-predictor-research.exe audit-swing-holding-identity --root . --output data/reports/swing_research_cohort/holding_identity_preflight.json
```

`configs/swing_holding_identity_preflight.toml` pins the cohort, parent manifest,
membership, dates and ten-session horizon. The command reads identity/clock
columns only, one monthly partition at a time, under the shared workspace lease.
It verifies source hashes before publication and refuses conflicting output.
Exit code 2 reports uncovered ownership; terminal immature rows are separate.
Covered ownership does not prove that bars exist or that prices represent verified
total returns. No numeric held-out outcomes are inspected, and no model is fitted.

### Initial-Fit Holding Observations

```powershell
.\.venv\Scripts\market-predictor-research.exe audit-swing-holding-observations --root . --output-directory data/reports/swing_holding_observations
```

`configs/swing_holding_observations.toml` pins the preceding report and the expected
566 initial-fit decisions across 60 securities. All ten required future sessions
must end by the report's initial-fit cutoff; numeric validation/test rows are not
returned. Existing raw data is reused without downloads or edits. The output directory
contains `decisions.parquet`, one required-session observation file per security/ticker,
and `_manifest.json`. Sessions are deduplicated within each security/ticker, not across
decisions; the manifest names the counting unit explicitly.

`observation_valid` means OHLCV and exchange clocks passed, not ownership or verified
total-return accounting. `ownership_unresolved` keeps observed `security_id` null.
Malformed source data is `source_error`, never a missing or zero-price fill. The
command continues other cases and exits 2 when any source error exists. Replay requires
`--expected-audit-sha256 <original-run-audit-hash>` retained independently of the output
manifest, and verifies source, implementation and output hashes. Do not pass these partial diagnostics as
the materializer's complete outcome source.

### Holding Corporate-Action Evidence

```powershell
.\.venv\Scripts\market-predictor-collect.exe collect-swing-holding-corporate-actions --root . --out-dir data/raw/swing_holding_corporate_actions
```

`configs/swing_holding_corporate_actions.toml` binds the observation inventory and
freezes the provider process-date range, page limit, response byte cap and expected
security count. The collector requests all action families with `data_quality=all`;
incomplete fields remain incomplete. Credentials come from the existing Alpaca
configuration and are not written to evidence. Each ticker has immutable attempts
with raw `.bin` bodies, response metadata and a hashed `receipt.json`. Collection
summaries are retained under `reports/` using their audit hash as the filename.
The returned hash must be retained independently for both online resume and offline
verification; missing or altered prior attempts are rejected before new requests:

```powershell
.\.venv\Scripts\market-predictor-collect.exe collect-swing-holding-corporate-actions --root . --out-dir data/raw/swing_holding_corporate_actions --offline --expected-audit-sha256 <original-report-hash>
```

Complete pagination proves only what this provider returned for this query. It does
not prove the absence of other actions, historical announcement availability or
settlement eligibility. These response archives require separate ownership and
economic interpretation before they can support corrected training labels.

### Estimator And Overlay Sources

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
- `intraday/datasets/selected_session_history.py`: verified selected stock-session
  acquisition planning at exact exchange-session bounds.
- `intraday/datasets/history_materialization.py`: bounded canonical per-symbol history
  construction, exchange-session segmentation, and source-integrity quarantine.
- `intraday/datasets/history_collection.py`: bounded, resumable Alpaca/SIP intraday
  collection with immutable raw-page, canonical-bar, and authority replay.
- `intraday/datasets/benchmark_history.py`: selected-session SPY, QQQ, and sector-ETF
  one-minute acquisition planning at exact exchange-session bounds.
- `intraday/datasets/broad_intraday_history.py`: research-only broad-universe
  regular-session five-minute acquisition planning with explicit membership limits.
- `intraday/datasets/extended_session_context.py`: separately bound premarket and
  postmarket five-minute context planning with exact exchange-calendar windows.
- `intraday/datasets/prospective_sip_session.py`: one closed-session prospective
  authority combining the exact observed S&P cohort at SIP five-minute resolution with
  SPY, QQQ, and all sector ETFs at SIP one-minute resolution. It binds policy files to
  effective configs, shares one request budget across both children, supports
  hash-verified crash recovery, and permits post-open parent finalization only from
  children retrieved before that open. It does not create features, labels, or model
  eligibility.
- `intraday/datasets/prospective_broker_actions.py`: resumable observed-time Alpaca
  broker-action polls and immutable revision generations bound to the A4.3 security
  namespace. Historical replay exposes identity hashes only and cannot authorize stale
  bars. Fresh collection requires the current complete bar authority. Cutoff registry
  commits follow strict replay, generation publication is atomic, Windows reparse paths
  fail closed, and all poll/generation children remain research-only and ineligible for
  training or serving.
- `swing_history_collection.py`, `swing_daily_combination.py`: swing daily history.
- `corpus_integrity.py`, `readiness.py`: corpus admission checks.

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
- `volume_bars.py`, `intraday_bar_features.py`: causal completed volume-bar state sampled
  on fixed five-minute cohorts, canonical five-minute ATR, exact market/sector context,
  and the ordered bar-only feature contract.
- `intraday_bar_labels.py`, `intraday/datasets/one_minute_coverage.py`: exact
  next-minute entry evidence, whole-security coverage admission, thirty-minute
  target/stop/timeout paths, and holding-aligned SPY, QQQ, and point-in-time sector
  returns. Missing exact stock or benchmark evidence abstains only that row.
- `intraday_bar_only_five_minute.py`: local immutable selected-session projection from
  the verified SIP/all five-minute authority; it never downloads provider data.
- `intraday_bar_dataset.py`, `intraday_bar_live.py`: transformation-hash-bound resumable
  publication, fixed-cohort batch/live parity, bounded two-process execution, source
  hash replay, interruption recovery, and per-ticker live abstention.
- `intraday_bar_audit.py`: reproducible row-level causality audit bound to exact dataset
  and projection manifest, authority, and inventory hashes.
- `intraday_microstructure_history.py`: immutable A4.1 planning plus bounded,
  page-resumable Alpaca SIP trade/quote transport. Completion authorizes only later
  materialization; a partial collection cannot train or serve.
- `intraday_development.py`: V3 expected-net-return development and future-holdout
  controls.
- `intraday_development.py`: A4.4 paired opportunity/downside training, purged
  walk-forward evaluation, strict immutable evidence replay, and locked future access.

Intraday V2 is published but economically rejected. The later V3 z-score lineage is
invalid and cannot train or serve. New candidate evidence uses explicit outcome
contracts and named after-cost stock/SPY/QQQ/sector binary diagnostics; the shared
shuffled-label control must remain at chance.

### Overlay and serving paths

- `sources/official_documents.py`: bounded official-document acquisition and offline
  receipt/body replay. `configs/swing_holding_source_documents.toml` names exact
  source URLs; `commands/swing_collection.py` is its collection-CLI adapter. Uses
  existing HTTP, SEC pacing, hash, strict JSON and locking utilities. Response bytes
  and acquisition failures are immutable per attempt; successful documents resume
  without fetching again. This source layer does not interpret corporate actions
  or grant identity, accounting, model or serving readiness.

- `research/swing_transfer_sources.py`: projected, hash-bound preparation of the
  selected index-transfer replay population and exact daily holding windows.
- `research/swing_transfer_replay.py`: sequential Alpaca evidence acquisition,
  immutable failed/successful attempts, offline replay and exact OHLCV comparison.
  Configuration: `configs/swing_transfer_replay.toml`; research command adapter:
  `replay-swing-transfer-history` in `commands/swing_research.py`. The canonical
  response decoder remains in `sources/alpaca.py`; no second transport was added.
- `universe/security_class_evidence.py`: pure SEC stock-class fact extraction from
  pinned bytes, with explicit context association and ambiguity rejection. It does
  not infer lifetime mapping intervals or historical availability.

- `sources/sec.py`, `catalysts/sec_filings/collection.py`, and
  `catalysts/sec_filings/decision_authority.py`: SEC transport, immutable issuer
  evidence, and zero-versus-unknown decision-time coverage semantics.
- `sources/gdelt.py`, `catalysts/global_events/collection.py`, and
  `catalysts/global_events/decision_authority.py`: GDELT transport, immutable global
  event evidence, and the separate decision-time global context overlay.
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
5. Preserve the current A4.3 authority at
   `data/features/intraday_causal_volume_bar_dataset_20260831_v2` and its immutable audit
   v2. Audit v2 must bind the exact five-minute projection, raw source cutoff, and
   five-minute prefix state. Future publication commands also require a separate
   execution-evidence directory so every resumable invocation is memory-audited. The
   current post-hoc execution assessment is incomplete and must not be upgraded by
   inference. Keep A4.2/A4.5 trade/quote features blocked until the complete A4.1 raw
   authority can be stored and replayed.
6. Preserve A4.4 continuation and long-reversion outputs as rejection evidence. Both
   are `no_candidate`; neither may serve or open the future holdout. The best audited
   positive-return ROC-AUC values are near 0.51 and controlling after-cost scopes fail.
7. Proceed to A5 only with a separately verified causal intraday event cohort. Do not
   add catalyst columns to the rejected bar-only baseline or tune against the unopened
   future authority.

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
