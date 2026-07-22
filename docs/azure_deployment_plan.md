# Azure Deployment Plan

## Recommendation

Start with:

```text
Azure Blob Storage + Azure Container Apps Jobs + Azure ML GPU compute on demand
```

Do not start with AKS unless the platform grows into multiple always-on services that need Kubernetes operations.

## Boundary

`market-predictor` owns its own Azure artifact layout. Other projects can keep independent repository layouts and data contracts. If another system needs model output later, expose a small explicit interface such as a prediction report export or API response.

## Why This Shape

The workload is mostly scheduled batch work:

- collect Alpaca, Seeking Alpha, SEC, Reddit, and screener data
- validate and deduplicate events
- align news timestamps with price bars for ML features
- score sentiment
- publish audited point-in-time feature snapshots
- evaluate candidate models only when enough matured labels exist
- publish complete, immutable model-plus-feature serving releases to Blob Storage

Container Apps Jobs fit this because they can run manual, scheduled, or event-triggered jobs. The nightly runs should stay small, observable, and isolated. One ticker or API failure should fail only that unit of work, not the whole platform.

Azure ML GPU compute or an on-demand GPU VM should be used only when FinBERT batch scoring or retraining is too slow on CPU. Keep GPU capacity off when it is not needed.

## Components

| Component | Azure Service | Purpose |
| --- | --- | --- |
| ML artifact storage | Azure Blob Storage | Stores this project's raw snapshots, curated datasets, reports, and active models. |
| Nightly collection | Azure Container Apps Job | Collects sources into isolated raw artifacts; it does not score predictions. |
| Feature publication | Azure Container Apps Job | Builds label-free canonical inference features, validates them, then atomically runs `publish-live-features`. |
| Prediction API | Azure Container App | Hydrates one active immutable release before startup and serves prediction, readiness, replay, and internal metrics routes. |
| Artifact export | Azure Container Apps Job | Runs `export-ohlcv-artifacts` and `azure-upload-artifacts` for this project only. |
| Guarded retraining | Azure ML job | Runs the registered training, shadow evaluation, and promotion workflow. Prediction traffic never triggers it. |
| Heavy sentiment/retraining | Azure ML compute cluster or GPU VM | Used on demand, not always running. |

## Artifact Layout

Default container:

```text
market-data
```

Default prefix:

```text
market-predictor
```

Recommended paths:

```text
market-predictor/raw/
market-predictor/live/
market-predictor/artifacts/ohlcv/
market-predictor/predictions/
market-predictor/serving/releases/<sha256>/
market-predictor/serving/_active_release.json
market-predictor/reports/
```

Each serving release contains configured promoted model artifacts/manifests and registered live feature artifacts/manifests. Release directories are immutable and content-addressed. `serving/_active_release.json` is the only mutable object in the serving protocol and is updated only after a complete release exists.

## Job Schedule

Use UTC in Azure schedules.

Berlin midnight is not always the same UTC time because of daylight saving time. Prefer one of these:

- Run at `22:30 UTC` during Central European Summer Time.
- Run at `23:30 UTC` during Central European Time.
- Or intentionally run after US market data has settled, for example `02:00 UTC`.

Recommended daily sequence:

```text
02:00 UTC  source-isolated collection
02:30 UTC  canonical feature build + validation + publication
03:00 UTC  immutable serving-release publication, when all routes are promotable
weekly/on demand  candidate training and shadow evaluation
```

Do not publish a new serving release merely because a nightly feature job succeeded. A release requires every configured route to resolve to a promoted, hash-verified model. Until a real canonical model passes promotion, the feature job may update local evidence but release publication must fail closed.

## Container Commands

Research collection:

```powershell
market-predictor collect-swing --days 3 --workers 8 --out-dir data/raw/nightly
market-predictor score-swing-events --raw-dir data/raw/nightly --out-dir data/raw/nightly_scored
```

Artifact export:

```powershell
market-predictor export-ohlcv-artifacts --days 730 --timeframes 1d,1h --workers 8
market-predictor azure-upload-artifacts --root data/artifacts
```

Swing feature publication after canonical source jobs finish:

```powershell
market-predictor build-swing-live-features `
  --decisions data/canonical/decisions.parquet `
  --benchmark-bars data/canonical/benchmark_daily_bars.parquet `
  --global-events data/canonical/global_events.parquet `
  --global-source-collections data/canonical/global_source_collections.parquet `
  --config configs/swing_dataset.toml `
  --out data/live/staging/swing_5d.parquet

market-predictor publish-live-features `
  --mode swing `
  --input-path data/live/staging/swing_5d.parquet `
  --live-dir data/live
```

Intraday uses the equivalent `build-intraday-live-features` command with canonical one-minute stock/benchmark bars and five-minute benchmark bars. Both builders write label-free canonical artifacts. `publish-live-features` derives feed and schema from the verified artifact; operators cannot override them.

Publish one complete serving release:

```powershell
market-predictor azure-publish-serving-release --root .
```

This command uploads all assets first, writes the immutable release manifest second, and moves the active pointer last. It accepts only promoted, integrity-checked models referenced by server-owned routes and fresh registered live snapshots with canonical source identity.

Hydrate an API revision manually or at startup:

```powershell
market-predictor azure-sync-serving-release --root .
```

The container entrypoint runs this before API import when `SYNC_AZURE_RELEASE_ON_STARTUP=true`. Downloads go to staging, every SHA-256 is verified, local manifests are installed after their artifacts, and the active-release marker is written last. Any failure stops startup rather than serving a partial release.

Rollback by moving the pointer to a complete prior release:

```powershell
market-predictor azure-rollback-serving-release --release-id <64-character-release-id>
```

Restart the API revision after rollback, or run sync in each instance. Rollback never mutates artifacts inside either release.

The removed `live-once`, `live-run`, and `live-train-event` commands must not be recreated as deployment wrappers. They blended incompatible models and promoted on insufficient gates. The production readiness endpoint returns 503 for missing/stale features, missing promoted models, hash failures, or process memory above the configured safety threshold.

## API Revision

Deploy the API and scheduled jobs independently. The API revision should use:

- target port `8000`
- `SYNC_AZURE_RELEASE_ON_STARTUP=true`
- `AZURE_STORAGE_ACCOUNT_URL` plus a managed identity with Blob Data Reader permission
- `AZURE_STORAGE_CONTAINER` and `AZURE_BLOB_PREFIX`
- 4 GiB memory limit, `RUNTIME_MEMORY_BUDGET_GIB=4.0`, and `RUNTIME_MEMORY_HEADROOM_GIB=0.25`
- liveness probe `/v1/health/live`
- readiness probe `/v1/health/ready`
- internal-only scrape access to `/v1/metrics`

Use a connection string only for controlled local work. Do not grant the serving identity Blob write access; publication and rollback jobs use a separate write-capable identity.

## VM vs AKS Decision

Use a VM first if you need a quick full-control box for experiments. Use Azure ML compute if you want GPU jobs that can shut down cleanly after runs. Use AKS later if you need Kubernetes-native operations, many services, custom autoscaling, service mesh, or multi-team deployment controls.

For this project today:

```text
Best first production path: Container Apps Jobs + Blob Storage + Azure ML GPU compute.
Not recommended initially: AKS.
Acceptable for experiments: one GPU VM, stopped when idle.
```

## Operational Rules

- Store secrets in Azure Container Apps secrets, Azure Key Vault, or managed identity. Do not bake API keys into images.
- Keep Alpaca, Seeking Alpha, Reddit, and SEC collectors independent so one source failure does not block other data.
- Write manifests for every exported dataset.
- Promote new models only after validation beats or matches the active baseline.
- Keep serving releases immutable; only the active release pointer may change.
- Give the API identity read-only Blob access and deployment jobs write access.
- Alert on readiness failure, source freshness, hash/audit failure, request errors/latency, drift status, replay outcomes, and memory headroom.
- Keep external integration explicit and narrow, for example exported prediction reports or API calls.
