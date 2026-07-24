# Implementation Guide

This document explains how `market-predictor` is designed, which files own each responsibility, and why the pipeline is split the way it is.

## Design Goals

- Predict swing behavior and intraday setups, not trade execution.
- Learn from event timing: pre-market, regular-hours, after-hours, and post-event price reaction.
- Keep API download, sentiment scoring, feature building, model training, and prediction as separate stages.
- Isolate failures by ticker and source. One failed ticker, Reddit request, or Seeking Alpha endpoint should not stop the rest of the run.
- Prevent obvious leakage by using event timestamps, next-session labels, and walk-forward validation.
- Keep Azure storage project-specific and avoid coupling it to any external repository layout.

## Main Runtime Flow

For operational prediction:

```text
audited upstream feature job
  -> atomically publish registered live feature snapshot
  -> readiness verifies feature and model manifests/hashes/freshness
  -> API filters by point-in-time cutoff
  -> score one server-registered promoted model per view
  -> persist immutable prediction snapshot
```

For training:

```text
historical events and bars
  -> canonicalize bar intervals, availability, source attempts, and universe membership
  -> verify event timestamps, first-seen time, scoring time, and ticker relevance
  -> score sentiment
  -> perform strict as-of joins to observed market/sector/event/fundamental context
  -> build event-level rows
  -> align prior price context and future labels
  -> train with purged walk-forward validation
  -> promote only clean models
```

## Important Commands

Setup and model download:

```powershell
python -m pip install --require-hashes --no-deps -r requirements/development.lock
python -m pip install --no-build-isolation --no-deps -e .
market-predictor-collect download-model
```

Research collection and sentiment scoring:

```powershell
market-predictor-collect collect-swing --tickers "LUNR,MXL,RGTI" --days 30 --out-dir data/raw/research --workers 4
market-predictor-research score-swing-events --tickers "LUNR,MXL,RGTI" --raw-dir data/raw/research --out-dir data/raw/research_scored
```

Prediction API requests are point-in-time contracts. `PredictionRequest.as_of`, when present, must be timezone-aware. Canonical daily and intraday inference require `feature_available_at_utc` and filter directly on that timestamp; neither reconstructs availability from a date or bar label.

`api_security.py` owns authentication and principal-scoped token buckets. Production startup permits only Entra mode and requires a local read-only JWKS, exact issuer, and exact audience. RS256 signature, `exp`, `iat`, optional `nbf`, issuer, and audience are validated before `scp` or `roles` authorization. The stable audit principal is tenant plus object id when available; the calling application id is retained separately. Static bearer auth is development-only. Liveness and minimal readiness are public; detailed operations, metrics, prediction, and replay routes require `operations.read`, `metrics.read`, `predictions.read`, and `replay.execute` respectively. Replay is disabled by default.

The API boundary rejects bodies above 64 KiB before Pydantic parsing, validates at most 100 canonical ticker symbols, rate-limits authenticated principals with bounded state, and emits opaque 401/403/413/429 envelopes. Structured audit events contain route templates, principal, scope, correlation id, release ids, and admission state; bearer tokens, request bodies, secrets, model paths, and exception details are never logged.

`PredictionRequest.horizon` defaults to `auto`. In that mode, the service resolves the horizon from the mode's server-owned route. Explicit horizons are validated against the registered model manifest. API clients cannot select a model, dataset, source mode, universe file, or promotion policy. `PredictionResponse.resolved_horizons` records the actual horizon used by each model, which is required for replay and downstream `trading_flow` audit records.

Daily and intraday readiness are separate gates. Daily models require daily-history depth; intraday models require intraday-bar warm-up. Feed provider and feed coverage are not interchangeable: `alpaca` alone does not prove consolidated coverage, while an explicit `sip`/consolidated value does. IEX invalidates volume-sensitive production readiness.

Top-level predictions are persisted by `prediction_snapshot.py` as content-addressed immutable records. The SHA-256 identifier covers the normalized request, response, model metadata, cutoff, and recording time. Loading a snapshot recomputes the hash and rejects modified content. Generated snapshot files belong under `data/predictions/snapshots/` and are runtime audit artifacts, not repository source files.

`outcome_intents.py` freezes identity-complete live prediction rows into semantic maturation intents. `outcome_repository.py` durably and idempotently stores intents, attempts, exact bar evidence, and outcomes while selecting one canonical occurrence for repeated semantic snapshots. `outcome_maturation.py` applies the model's full hash-bound label policy to hash-verified canonical bars: swing requires the exact next-session open, full session path, horizon close, and aligned SPY/QQQ/sector intervals; intraday requires the exact 1-minute bar starting at the decision timestamp, a consecutive path, canonical gap-through fills, and stop-first same-bar ambiguity. A missing entry bar is never shifted forward. `outcome_worker.py` isolates per-intent data failures and records deterministic pending or blocked attempts.

`performance_monitoring.py` builds content-addressed release/view/horizon cohorts from canonical semantic outcomes only. It reports calibration, net and SPY-relative economics, win rate, and drawdown for overall, regime, sector, market-cap, liquidity, and calibration-bin slices. `drift_policy.py` combines a validated performance report with feature drift under `configs/drift_policy.toml`, persists a release-specific assessment, and verifies its hash when read. `prediction_service.py` requires a current actionable assessment for each direct route. Unified output fails closed when either requested view is unavailable or non-actionable. Readiness exposes each route assessment; missing, stale, warming, severe, tampered, or wrong-release state returns not-ready.

`investment_replay.py` evaluates a stored prediction against subsequently available Alpaca bars. It enters the stock, SPY, and QQQ at an aligned next-bar open and exits them at the same completed-bar boundary. Slippage and commission assumptions apply on entry and exit. Replay validates model creation time, training-data end time, prediction readiness, and snapshot integrity before requesting price data. Historical requests made with a model that did not yet exist are invalid, even if the underlying feature row can be reconstructed.

`POST /v1/replays/investment` is snapshot-driven and does not accept filesystem paths. This prevents an API client from selecting arbitrary local artifacts and ensures that every replay can be traced to a served prediction. Non-actionable signals return `not_entered`; `force_entry` is only a research override and cannot bypass invalid readiness or future-model checks.

`live_features.py` owns the final inference-only selection gate. The swing and intraday dataset builders share feature-history construction with training, but live commands select one latest coherent decision cross-section and remove all labels, targets, and future paths. They reject future availability, under-warm history, stale or failed source state, schema mismatch, non-SIP volume, invalid cross sections, and any label-bearing output.

`feature_store.py` owns the live inference handoff. It accepts only canonical `swing_inference_features` or `intraday_inference_features` artifacts and publishes rolling Parquet files atomically with a sidecar manifest containing generation time, canonical source artifact type/hash, feature schema, source watermarks, feed tier, row/ticker/column identity, latest feature time, and artifact SHA-256. Production serving always reads these registered paths. Missing, modified, mixed-time, future-generated, stale, or label-bearing snapshots fail before model scoring. External API calls and FinBERT latency stay outside the request path.

`serving_context.py` resolves each route through a signed local active-release pointer, verifies the full immutable release, and caches one deserialized context per route. Pointer changes are serialized globally; the old route slot is released before the replacement is loaded, and projected/current RSS is checked against the 4 GiB policy. `admission.py` owns the single non-queueing inference lease and request memory reservation. `prediction_service.py` consumes cached payloads, enforces bounded ticker batches, and records model release ids in prediction evidence. API startup preloads all routes; readiness never deserializes a model or reads a full feature frame.

`telemetry.py` owns bounded in-process request, prediction-readiness, replay-outcome, model-hash, health, and memory metrics plus structured JSON events. `/v1/metrics` is intended for internal scraping, not public ingress. This is operational telemetry, not durable prediction-performance evidence; the immutable prediction/outcome repository and content-addressed performance reports are the validation source of truth.

`configs/default.toml` declares production routes under `[prediction_serving.routes.<mode>."<horizon>"]` using `release_repository`, `bar_timeframe`, and `estimated_resident_gib`. `[prediction_serving]` names the public attestation trust store. Direct model paths are rejected.

The previous `live-once` publisher was removed because it also scored four incompatible legacy model families and averaged their probabilities. Scheduled jobs now call `build-swing-live-features` or `build-intraday-live-features`, then `publish-live-features`. No generic Parquet publisher or legacy scoring fallback exists; a failed build leaves the prior snapshot untouched and readiness eventually fails on freshness.

`catalyst_overlay.py` is deliberately separate from estimator inference. It classifies recent evidence as confirmed, conflicting, veto, mixed, or absent. The original model probability is never modified. A separate `decision_score` adds a small confirmation bonus or conflict/veto penalty for ranking, and the API records the complete catalyst assessment so its incremental value can be ablated later.

Historical event collection and sentiment build:

```powershell
market-predictor-collect collect-swing --days 730 --out-dir data/raw/swing --workers 8
market-predictor-research verify-swing --raw-dir data/raw/swing --rewrite
market-predictor-research score-swing-events --raw-dir data/raw/swing --out-dir data/raw/swing_scored
```

Canonical data publication and decision build:

```powershell
market-predictor-research canonicalize-bars --input-path data/raw/bars.parquet --out data/canonical/bars.parquet --timeframe 5m --price-feed sip
market-predictor-research canonicalize-event-directory --input-dir data/raw/swing_scored --out data/canonical/events.parquet
market-predictor-research canonicalize-source-collections --input-path data/raw/swing/_source_collections.parquet --out data/canonical/source_collections.parquet
market-predictor-research canonicalize-memberships --input-path data/raw/universe_memberships.parquet --out data/canonical/memberships.parquet
market-predictor-research build-canonical-decisions --bars data/canonical/bars.parquet --events data/canonical/events.parquet --source-collections data/canonical/source_collections.parquet --memberships data/canonical/memberships.parquet --out data/canonical/decisions.parquet
```

Each canonical output has a sidecar manifest containing its SHA-256, input hashes, row/column identity, availability range, audit evidence, and production-readiness state. A consumer verifies the manifest and hash before reading the table. Production decisions fail when any required source was not successfully observed through a fresh request coverage end, when membership is unknown or ambiguous, when volume is not SIP, or when a joined feature is from the future.

Canonical swing build, training, and promotion:

```powershell
market-predictor-research build-swing-dataset --decisions data/canonical/decisions.parquet --benchmark-bars data/canonical/benchmark_daily_bars.parquet --global-events data/canonical/global_events.parquet --global-source-collections data/canonical/global_source_collections.parquet --config configs/swing_dataset.toml --out data/features/swing/swing_5d.parquet
market-predictor-research train-swing-model --dataset data/features/swing/swing_5d.parquet --config configs/swing_training.toml --model-out models/swing/candidates/swing_5d.joblib --evidence-dir data/reports/swing_5d_candidate
market-predictor-research promote-swing-model --model models/swing/candidates/swing_5d.joblib --evidence-dir data/reports/swing_5d_candidate --hypothesis-registry data/governance --hypothesis-id swing-5d-h001 --shadow-bundle data/governance/shadow/<shadow-fingerprint>.json --build-identity ci:<build-id> --approver-identity reviewer:<identity> --signing-private-key <secure-ed25519-private-key.pem> --attestation-trust-store configs/attestation_trust_store.json --signer-id promotion-ci-prod --config configs/swing_promotion.toml
```

`src/market_predictor/swing/contracts.py` freezes feature/model schemas and typed configs. `dataset.py` owns technical, benchmark-relative, catalyst/global, membership, cross-sectional, and exact future-path construction. `audits.py` owns fail-closed eligibility. `model.py` owns purged folds, unseen-ticker holdout, calibration, memory enforcement, immutable candidate registration, and scoring. `evaluation.py` owns classification, ranking-economics, regime, and catalyst validation. `promotion.py` owns hash-bound evidence bundles and fail-closed development gates, then delegates predeclared hypothesis, untouched-shadow, one-use ledger, and attestation enforcement to `promotion_workflow.py`. `commands/swing_model.py` is the only canonical research command entry point for those stages.

Canonical intraday build, training, and promotion:

```powershell
market-predictor-research build-intraday-dataset --decisions data/canonical/intraday_decisions_5m.parquet --one-minute-bars data/canonical/intraday_bars_1m.parquet --benchmark-bars data/canonical/intraday_benchmarks_5m.parquet --global-events data/canonical/global_events.parquet --global-source-collections data/canonical/global_source_collections.parquet --config configs/intraday_dataset.toml --out data/features/intraday/intraday_60m.parquet
market-predictor-research train-intraday-model --dataset data/features/intraday/intraday_60m.parquet --config configs/intraday_training.toml --model-out models/intraday/candidates/intraday_60m.joblib --evidence-dir data/reports/intraday_60m_candidate
market-predictor-research promote-intraday-model --model models/intraday/candidates/intraday_60m.joblib --evidence-dir data/reports/intraday_60m_candidate --hypothesis-registry data/governance --hypothesis-id intraday-60m-h001 --shadow-bundle data/governance/shadow/<shadow-fingerprint>.json --build-identity ci:<build-id> --approver-identity reviewer:<identity> --signing-private-key <secure-ed25519-private-key.pem> --attestation-trust-store configs/attestation_trust_store.json --signer-id promotion-ci-prod --config configs/intraday_promotion.toml
```

`src/market_predictor/intraday/contracts.py` freezes the 5-minute decision, 1-minute execution, feature, model, and typed configuration contracts. `dataset.py` owns completed-bar technical state, the latest fully available 1-minute confirmation state, and benchmark/global/membership/cross-sectional context. Intraday feature contract v2 preserves ticker-local availability, delays each nominal 5-minute cross-section to the latest required peer/benchmark cutoff, and joins 1-minute confirmation state only after that common cutoff is fixed. `labels.py` owns the exact decision-time 1-minute entry and consecutive target/stop path, benchmark returns over the same interval, overlap weights, and independent-event identities. `audits.py` rejects future features, peer availability beyond the common cross-section cutoff, under-warm rows, non-SIP/partially adjusted bars, stale source state, missing paths, and missing benchmark intervals. `model.py` trains opportunity and downside estimators atomically with session-purged walk-forward validation, deterministic unseen-ticker holdout, cross-fitted calibration, overlap weights, `float32` matrices, and a 4 GiB guard. It keeps seen/unseen classification diagnostics separate but combines their fold predictions into one full decision cross-section for economic selection, then attributes already-selected trades back to each cohort. Capacity consumes that same selected stream, retains the complete capital curve, and treats absent liquidity inputs separately from genuine participation or minimum-volume no-fills. `evaluation.py` owns classification, non-overlapping top-k economics, the conservative overlap-evidence count (the minimum of summed label uniqueness and independently non-overlapping event IDs), and deterministic session-block regime confidence intervals. `promotion.py` verifies persisted candidate/evidence hashes, reproduces capacity summaries from the signed curve, and gates effective evidence in both development and ticker-holdout scopes alongside independent-session, dual-model, economics, drawdown, no-fill, all required market regimes, confidence lower bounds, catalyst-coverage, alignment, memory, and provenance evidence before the shared hypothesis/shadow/attestation workflow can authorize the artifact. `commands/intraday_model.py` is the only canonical C5 CLI entry point.

The canonical intraday horizon is `60m`. Opportunity means target-before-stop; downside means stop-before-target. Catalyst/news is an external confirmation and ranking overlay and is not included in either estimator feature list. No real canonical intraday artifact is promoted yet.

Historical provider backfills normally know publication time but not when this system first observed the item. Such events are publication-time proxies and are research-only. They must not be relabeled as observed history. The same rule applies to SEC and Seeking Alpha current snapshots: only versioned facts with explicit availability can enter historical production features.

Trusted promotion and local release:

```powershell
market-predictor-prod publish-local-release --model models/swing/candidates/swing_5d.joblib --evidence-manifest data/reports/swing_5d_candidate/evidence.manifest.json --release-root data/local_release_repository --attestation-trust-store configs/attestation_trust_store.json
market-predictor-prod show-active-local-release --release-root data/local_release_repository --attestation-trust-store configs/attestation_trust_store.json
market-predictor-prod verify-local-release --release-id <64-character-release-id> --release-root data/local_release_repository --attestation-trust-store configs/attestation_trust_store.json
market-predictor-prod rollback-local-release --release-id <64-character-release-id> --release-root data/local_release_repository --attestation-trust-store configs/attestation_trust_store.json
```

`registry.py` can publish only immutable candidate manifests; effective `promoted` state is derived from a strict Ed25519-signed attestation whose issuer exists in the server-owned trust store. `hypothesis_registry.py` freezes the baseline and hypothesis before shadow generation. `shadow_ledger.py` content-addresses paired session evidence, recomputes the session-block interval, derives its source identity from the exact session records and frozen candidate/baseline/policies, consumes each fingerprint once in the governance root's canonical ledger, and retires a hypothesis family after failure. `promotion_attestation.py` binds the artifact, candidate manifest, training evidence, causal identity chain, baseline, ledger receipt, gates, build, distinct approver, signer, and timestamp. `release.py` stages and verifies the complete model/attestation/evidence unit, renames it into a versioned directory, and moves one hash-protected active pointer under an OS-released file lock. Mutation, partial publication, untrusted signatures, and rollback targets are reverified before activation. The private signing key belongs only to the promotion workload; set `MARKET_PREDICTOR_ATTESTATION_TRUST_STORE` to a read-only public trust-store path for serving.

Canonical live publication and external serving release:

```powershell
market-predictor-research build-swing-live-features --decisions data/canonical/decisions.parquet --benchmark-bars data/canonical/benchmark_daily_bars.parquet --global-events data/canonical/global_events.parquet --global-source-collections data/canonical/global_source_collections.parquet --config configs/swing_dataset.toml --out data/live/staging/swing_5d.parquet
market-predictor-prod publish-live-features --mode swing --input-path data/live/staging/swing_5d.parquet --live-dir data/live
```

The obsolete Azure serving-release transport was removed. Azure release
transport and disaster recovery remain `environment_pending`; the signed local
repository is the only activation authority. `azure_store.py` remains a
collection-side artifact utility and is not imported by production serving.

## File Responsibilities

### CLI and Orchestration

The installed commands are intentionally split:

- `src/market_predictor/production_cli.py` exposes only serving, live-feature
  publication, signed local-release management, and outcome/drift operations.
- `src/market_predictor/collection_cli.py` exposes provider collection, source
  quota/token utilities, model download, and artifact upload/export.
- `src/market_predictor/research_cli.py` exposes canonicalization, feature
  engineering, model training, evaluation, and promotion.
- `src/market_predictor/cli.py` is the internal research/collection command
  registry used to construct the two non-production surfaces. It is not an
  installed executable and is never imported by production.

Why the split exists: the serving process has a small auditable import graph and
cannot accidentally initialize provider clients, FinBERT, Torch, Transformers,
XGBoost, or research promotion modules. Collection remains restartable and
source-isolated; research retains all reproducible data/model workflows.

Command ownership:

- Collection: `collect`, `collect-swing`, `collect-seeking-alpha`, `alpaca-tickers`.
- Verification: `verify-events`, `verify-swing`, `audit-swing-alignment`.
- Sentiment: `download-model`, `score-swing-events`.
- Feature building: canonical training `build-swing-dataset` / `build-intraday-dataset`, canonical inference `build-swing-live-features` / `build-intraday-live-features`, research-only builders, and V3 research builders.
- Training/scoring: canonical swing and intraday train/promote commands, research-only entry-path commands, and V3 research evaluation.
- Production: `serve-api`, `publish-live-features`, signed local-release
  commands, and deterministic outcome/drift commands.
- Azure data utilities: `export-ohlcv-artifacts` and `azure-upload-artifacts`.

Canonical orchestration is registered by `src/market_predictor/commands/canonical_data.py`. `src/market_predictor/canonical/contracts.py` owns immutable schemas; `normalize.py` converts provider timestamps and provenance; `joins.py` performs strict as-of event, source, membership, and fundamental joins; `audits.py` implements fail-closed readiness; and `store.py` publishes hash-verified artifacts with manifests written last.

`build-swing-datasets` remains research-only. Production swing feature engineering is `build-swing-dataset` on the canonical decision table. The production path never fetches while building features, never forward-fills current Seeking Alpha/SEC snapshots, and never accepts publication-proxy history.

V3 orchestration is registered through focused modules under `src/market_predictor/commands/`: `v3_data.py`, `v3_features.py`, `v3_labels.py`, `v3_models.py`, and `v3_evaluation.py`. The corresponding implementation under `src/market_predictor/v3/` owns strict contracts, immutable development/shadow partitioning, exact labels, batch/live feature parity, session-purged validation, deterministic ticker holdout, B0/B1/B2/R1/D1 candidate training, disjoint calibration, and session-blocked ranking economics. Label schema `ml_v3.labels.v2` requires the maximum configured path to be contiguous at `bar_minutes`; decisions spanning a missing ticker candle are dropped rather than interpolated or shifted. These candidates and audit calibrators are research artifacts and are not connected to the promoted serving registry until later promotion checkpoints pass.

`train-v3-models` loads versioned datasets through a hash-verified column projection. The trainer compacts selected features to `float32`, uses single-worker XGBoost histograms for R1, releases each fold/holdout model before the next fit, and records current/peak working set against `--max-training-memory-gb`. A guard violation fails the family without publishing a model artifact.

`v3_readiness.py` scans large Parquet datasets in batches before C8. It rejects inadequate symbol/session coverage, post-cutoff rows, undeclared or non-SIP volume, current-only universe files, missing sector ETFs, and benchmark coverage gaps. `export-ohlcv-artifacts --end-date YYYY-MM-DD` creates reproducible frozen-cutoff exports and persists `price_feed` in every row and manifest.

`v3/catalysts.py` owns the O1 point-in-time overlay and paired ablation. It filters decisions to an explicit source interval, joins only events available by each decision timestamp, validates ticker-file and sentiment coverage, detects future matches, and compares R1/O1 on identical groups with a session-blocked paired bootstrap. Provider publication-time backfill is marked research-only. Optional global context must cover both declared interval boundaries or readiness fails.

`score-swing-events` keeps raw provider text unchanged and writes sentiment to a separate per-ticker directory. For catalyst research, `--text-mode title_summary --max-length 128` bounds inference to the immutable headline and provider summary. Every output row carries the FinBERT model, input mode, and token limit; an existing file is resumed only when all provenance fields match. Model inference loads the previously downloaded local cache and does not make hidden network requests.

`audit-v3-failure-attribution` is a development-only diagnostic for a rejected ranker. It loads only registered, hash-verified monthly shards; validates exact OOF-to-label identities; rejects shadow timestamps; and writes fixed top-k horizon, score-decile, and stratum evidence. Its session bootstrap is vectorized over session sums/counts, preserving block-resampling semantics without repeated DataFrame concatenation. The report is explicitly non-promotional and cannot justify filters on the inspected strata.

The V4-H1 audit sequence is deliberately fail-closed. The first 120-minute dataset was rejected before training after timing checks found non-contiguous 24-bar paths. The corrected immutable v2 dataset is separately fingerprinted and verifies exact wall-clock exits on every physical row, then verifies minimum cross-section and 120-minute cadence on rank-eligible rows. B0 and R1 are trained into separate directories so one family cannot overwrite or mask the other. Both were rejected on development economics; shadow was not read.

### Configuration

`src/market_predictor/config.py`

Loads `.env` secrets and exposes typed settings. This is where Alpaca, Reddit, RapidAPI, Seeking Alpha account-token, FinBERT, Azure, and runtime defaults become usable by code.

`src/market_predictor/app_config.py`

Loads non-secret TOML behavior from `configs/default.toml`.

`configs/default.toml`

Owns universe lists, sector groups, benchmark mapping, source behavior, Seeking Alpha endpoint templates, Reddit subreddit settings, and performance knobs. Environment settings also define the API memory budget/headroom and optional Azure release sync at container startup.

Why split config this way: secrets stay in `.env`; business/runtime behavior stays in TOML.

### Data Schemas and Quality

`src/market_predictor/schemas.py`

Defines the normalized `NewsEvent` record used across Alpaca, Reddit, Seeking Alpha, and SEC.

`src/market_predictor/data_quality.py`

Sanitizes and verifies event data. It handles timestamp parsing, required fields, duplicate rows, ticker normalization, and basic data validity.

Why it matters: the model is only useful if event timing and ticker relevance are clean. Bad timestamps create label leakage or false reactions.

### Source Adapters

`src/market_predictor/sources/alpaca.py`

Fetches Alpaca assets, news, daily bars, and hourly bars. Alpaca is the primary market-data and recent-news source.

`src/market_predictor/sources/reddit.py`

Collects Reddit ticker chatter from configured subreddits and comments. Reddit is treated as attention and sentiment, not normal news.

`src/market_predictor/sources/seeking_alpha.py`

Calls Seeking Alpha through RapidAPI, manages local quota tracking and caching, fetches analysis/news-style events, quant/rating snapshots, and account access tokens when configured.

It filters ticker-specific Seeking Alpha articles by ticker tags so unrelated market-wide articles are not blindly assigned to a ticker. Broad SA feeds are collected separately as market context and use ticker `MARKET`.

The expanded RapidAPI snapshot collection is intentionally stored as external data. Daily historical model rows only consume canonical compatibility columns, because current snapshots should not be backfilled across two years of labels.

`src/market_predictor/sources/sec.py`

Fetches SEC company facts and filings with no API key. Used for fundamentals and filing events.

`src/market_predictor/sources/http.py`

Small HTTP helper with retry/backoff behavior.

`src/market_predictor/sources/seeking_alpha_mcp.py`

MCP discovery helper for RapidAPI Seeking Alpha metadata when an MCP server is available. Runtime collection still uses direct REST calls.

### Price and Feature Engineering

`src/market_predictor/price.py`

Fetches daily and hourly bars through Alpaca wrappers.

`src/market_predictor/features.py`

Builds model rows from events and bars.

Major responsibilities:

- Daily direction datasets.
- Event swing datasets.
- Event timing buckets: pre-market, intraday, after-hours.
- Prior return and volume context.
- Open-gap, same-day, next-day, and next-week reaction labels.
- Hourly reaction features when 1h bars are available.
- Reddit, sentiment, Seeking Alpha, SEC, sector, and benchmark features.
- Global market-context features such as market news count, market sentiment, negative/positive sentiment fractions, and 30-day news-volume z-score.

Why this file is central: this is where market intuition becomes structured ML input.

### Sentiment

`src/market_predictor/sentiment.py`

Downloads and runs FinBERT. It uses GPU if PyTorch detects CUDA, otherwise CPU. Sentiment scoring is a separate stage so expensive NLP work can be retried independently from API collection.

### Modeling

`src/market_predictor/model.py`

Trains and scores the tabular models.

Important design choices:

- Uses scikit-learn pipelines.
- Uses purged walk-forward validation to reduce time leakage.
- Supports daily direction models and event-level swing models.
- Returns metrics that the guarded live trainer can use before promotion.

`DateGroupedPurgedWalkForwardSplit` keeps validation later in time than training and leaves an embargo gap between train and test sessions.

### Quota and Azure

`src/market_predictor/quota.py`

Tracks monthly RapidAPI usage locally so Seeking Alpha calls can be throttled before exhausting the plan.

`src/market_predictor/azure_store.py`

Uploads and downloads project artifacts and release bytes to Azure Blob Storage. It supports either a connection string or managed identity/default credentials through Azure Identity.

## Data Directories

Recommended local layout:

```text
data/raw/                 raw downloaded event files
data/raw/swing/           per-ticker swing event files
data/raw/swing_scored/    FinBERT-scored event files
data/features/            model-ready feature datasets
data/live/                managed live pipeline state
data/artifacts/           Azure-uploadable project artifacts
data/reports/             readable outputs and score reports
data/cache/               source cache files and SA account token cache
data/usage/               RapidAPI usage tracking
models/                   active and candidate model files
```

Do not treat `data/features` or `data/live` as public contracts. They are internal ML working sets and may evolve.

## Model Artifacts

The clean active model set lives in `models/`.

Every scoreable Joblib artifact has a `model_registry_manifest.v1` sidecar. Candidate and promoted research scorers verify the manifest and SHA-256 before deserialization; production routes additionally require `promoted`. The current evidence and route state are maintained in the README and model cards rather than inferred from filenames.

## Live Pipeline State

Canonical label-free live feature construction, atomic publication, immutable release activation, startup hydration, and rollback are implemented. Azure schedules remain deployment configuration rather than application-owned timers. Retraining is deliberately separate and is never triggered by prediction traffic. If no real canonical model has passed promotion, release publication fails and the API remains not ready; candidate or legacy artifacts are not substituted.

## Azure Deployment

Use Azure Blob Storage for project artifacts and Azure Container Apps Jobs for scheduled runs. Use Azure ML GPU compute or a stopped-when-idle GPU VM only for heavy FinBERT/retraining work.

Files:

- `Dockerfile`: digest-pinned, non-root, read-only-compatible API image with 4
  GiB defaults, hash-only production dependencies, and a liveness probe.
- `.dockerignore`: keeps local data, models, secrets, and venv out of the build context.
- `requirements/*.lock`: universal hash locks for production, collection,
  training, validation, and full development.
- `scripts/lock_dependencies.py`: pinned cross-platform lock regeneration.
- `docs/azure_deployment_plan.md`: deployment recommendation, schedule, and operational rules.

Azure model-release hydration is `environment_pending`; the current container
starts only from mounted signed local release repositories. Keep `/v1/metrics`
internal, configure readiness against `/v1/health/ready`, and enforce a 4 GiB
container memory limit in addition to the application guard.

## Why the Stages Are Separate

API collection, NLP scoring, feature building, and model training fail for different reasons and have different costs.

Keeping them separate gives:

- Restartability after partial API failures.
- Lower RapidAPI and Reddit waste.
- Easier data validation before training.
- Ability to run FinBERT on GPU later without changing collectors.
- Cleaner model audit trails.

## Adding a New Data Source

1. Add a source adapter under `src/market_predictor/sources/`.
2. Convert source records into `NewsEvent`.
3. Add config knobs to `configs/default.toml` if needed.
4. Wire the source into `collect_events_for_ticker` in `cli.py`.
5. Update `data_quality.py` only if the normalized schema needs new validation.
6. Add feature columns in `features.py` after the data is clean.

## Adding a New Model

1. Define a new immutable feature, target, model, and availability contract in the owning canonical package.
2. Build labels from the unsampled exact future path and audit every excluded row before estimator filtering.
3. Add purged time validation, unseen-ticker validation, calibration, benchmark-relative economics, drawdown, regime, alignment, provenance, and resource evidence.
4. Publish a hash-verified candidate and hash-bound evidence bundle through a focused command module.
5. Add a frozen promotion configuration; serving must reject the model type/schema until every gate passes and an operator registers the promoted route.
6. Keep research families under `v3/` or another explicitly research-only package; they cannot reuse the canonical registry type without satisfying its complete contract.

## Practical Debugging

Check source/API issues:

```powershell
market-predictor-collect seeking-alpha-limits
market-predictor-collect collect LUNR --days 3 --out data/raw/debug_lunr.parquet
```

Check event quality:

```powershell
market-predictor-research verify-events data/raw/debug_lunr.parquet --rewrite
```

Check Azure storage configuration:

```powershell
market-predictor-collect azure-upload-artifacts --root data/artifacts
```
