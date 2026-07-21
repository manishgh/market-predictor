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
- publish this project's artifacts, reports, and models to Blob Storage

Container Apps Jobs fit this because they can run manual, scheduled, or event-triggered jobs. The nightly runs should stay small, observable, and isolated. One ticker or API failure should fail only that unit of work, not the whole platform.

Azure ML GPU compute or an on-demand GPU VM should be used only when FinBERT batch scoring or retraining is too slow on CPU. Keep GPU capacity off when it is not needed.

## Components

| Component | Azure Service | Purpose |
| --- | --- | --- |
| ML artifact storage | Azure Blob Storage | Stores this project's raw snapshots, curated datasets, reports, and active models. |
| Nightly collection | Azure Container Apps Job | Collects sources into isolated raw artifacts; it does not score predictions. |
| Feature publication | Azure Container Apps Job | Builds the canonical point-in-time table, validates it, then runs `publish-live-features`. Not enabled until the canonical builder is complete. |
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
market-predictor/models/active/
market-predictor/reports/
```

These paths are internal to `market-predictor` and may evolve with the ML pipeline.

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
03:00 UTC  artifact export + blob upload
weekly/on demand  candidate training and shadow evaluation
```

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

Feature publication, after canonical builder validation:

```powershell
market-predictor publish-live-features --mode swing --input-path <audited-parquet> --price-feed sip
```

Publish models:

```powershell
market-predictor azure-publish-models --models-dir models
```

This command publishes only promoted, integrity-checked artifacts referenced by the server-owned production routes. Each model and registry sidecar is written beneath its mode/horizon prefix, then `_production_routes_manifest.json` is uploaded last as the deployment commit point.

The removed `live-once` and `live-train-event` commands must not be recreated as deployment wrappers. They blended incompatible models and promoted on insufficient gates. Until the canonical feature and training jobs exist, the production readiness endpoint is expected to return 503 for missing live features.

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
- Keep external integration explicit and narrow, for example exported prediction reports or API calls.
