# Implementation Guide

This document explains how `market-predictor` is designed, which files own each responsibility, and why the pipeline is split the way it is.

## Design Goals

- Predict swing behavior, not intraday execution.
- Learn from event timing: pre-market, regular-hours, after-hours, and post-event price reaction.
- Keep API download, sentiment scoring, feature building, model training, and prediction as separate stages.
- Isolate failures by ticker and source. One failed ticker, Reddit request, or Seeking Alpha endpoint should not stop the rest of the run.
- Prevent obvious leakage by using event timestamps, next-session labels, and walk-forward validation.
- Keep Azure storage project-specific and avoid coupling it to any external repository layout.

## Main Runtime Flow

For daily use:

```text
input tickers
  -> collect recent Alpaca/Reddit/Seeking Alpha/SEC events
  -> sanitize and deduplicate events
  -> score event sentiment with FinBERT
  -> fetch latest daily/hourly bars and market-context features
  -> build event and watchlist features
  -> score active models
  -> write readable prediction reports
```

For training:

```text
historical events and bars
  -> verify event timestamps and ticker relevance
  -> score sentiment
  -> build market-wide context from SPY/sector/ETF proxy event streams
  -> build event-level rows
  -> align prior price context and future labels
  -> train with purged walk-forward validation
  -> promote only clean models
```

## Important Commands

Setup and model download:

```powershell
python -m pip install -e .
market-predictor download-model
```

Live prediction cycle:

```powershell
market-predictor live-once --tickers "LUNR,MXL,RGTI" --lookback-days 3 --workers 4
```

Watchlist prediction:

```powershell
market-predictor predict-watchlist --tickers "LUNR,MXL,RGTI" --out data/reports/watchlist_latest.csv
```

Prediction API requests are point-in-time contracts. `PredictionRequest.as_of`, when present, must be timezone-aware. The serving service filters daily feature rows by their 16:00 America/New_York availability time and intraday rows by inferred bar-close time. It does not interpret a bar's start timestamp as the moment its closing price and volume became known.

`PredictionRequest.horizon` defaults to `auto`. In that mode, the service resolves the horizon from an explicitly selected model target or from the mode's registered default. Explicit horizons are validated against the model manifest. `PredictionResponse.resolved_horizons` records the actual horizon used by each model, which is required for replay and downstream `trading_flow` audit records.

Daily and intraday readiness are separate gates. Daily models require daily-history depth; intraday models require intraday-bar warm-up. Feed provider and feed coverage are not interchangeable: `alpaca` alone does not prove consolidated coverage, while an explicit `sip`/consolidated value does. IEX invalidates volume-sensitive production readiness.

Top-level predictions are persisted by `prediction_snapshot.py` as content-addressed immutable records. The SHA-256 identifier covers the normalized request, response, model metadata, cutoff, and recording time. Loading a snapshot recomputes the hash and rejects modified content. Generated snapshot files belong under `data/predictions/snapshots/` and are runtime audit artifacts, not repository source files.

`investment_replay.py` evaluates a stored prediction against subsequently available Alpaca bars. It enters the stock, SPY, and QQQ at an aligned next-bar open and exits them at the same completed-bar boundary. Slippage and commission assumptions apply on entry and exit. Replay validates model creation time, training-data end time, prediction readiness, and snapshot integrity before requesting price data. Historical requests made with a model that did not yet exist are invalid, even if the underlying feature row can be reconstructed.

`POST /v1/replays/investment` is snapshot-driven and does not accept filesystem paths. This prevents an API client from selecting arbitrary local artifacts and ensures that every replay can be traced to a served prediction. Non-actionable signals return `not_entered`; `force_entry` is only a research override and cannot bypass invalid readiness or future-model checks.

`feature_store.py` owns the live inference handoff. Collection and feature jobs publish rolling swing or intraday Parquet files atomically with a sidecar manifest containing generation time, source watermarks, feed tier, row/ticker counts, and artifact SHA-256. `PredictionRequest.data_source="live"` can only read these registered paths. Missing, modified, future-generated, or stale snapshots fail before model scoring. This keeps external API calls and FinBERT latency outside the request path.

`live-once` now derives volatile-schema swing rows from each ticker's sanitized event store and daily feature history, then publishes their combined rolling snapshot to `data/live/features/swing.parquet`. The 5-minute pipeline publishes its final enriched table with `publish-live-features --mode intraday`. Both paths use the same manifest and integrity contract.

`catalyst_overlay.py` is deliberately separate from estimator inference. It classifies recent evidence as confirmed, conflicting, veto, mixed, or absent. The original model probability is never modified. A separate `decision_score` adds a small confirmation bonus or conflict/veto penalty for ranking, and the API records the complete catalyst assessment so its incremental value can be ablated later.

Historical event model build:

```powershell
market-predictor collect-swing --days 730 --out-dir data/raw/swing --workers 8
market-predictor verify-swing --raw-dir data/raw/swing --rewrite
market-predictor score-swing-events --raw-dir data/raw/swing --out-dir data/raw/swing_scored
market-predictor build-event-swing-datasets --raw-dir data/raw/swing_scored --out-dir data/features/event_swing --workers 8
market-predictor combine-event-swing-datasets --feature-dir data/features/event_swing --out data/features/event_swing_all.parquet
market-predictor train-event-swing --dataset data/features/event_swing_all.parquet --model-out models/event_swing_1d.joblib --target-col target_next_1d_up
```

Azure artifact publishing:

```powershell
market-predictor export-ohlcv-artifacts --days 730 --timeframes 1d,1h --workers 8
market-predictor azure-upload-artifacts --root data/artifacts
market-predictor azure-publish-models --models-dir models
```

## File Responsibilities

### CLI and Orchestration

`src/market_predictor/cli.py`

The command center. It wires sources, feature builders, model functions, file paths, and reports into Typer CLI commands.

Why it exists: keep operational workflows scriptable and restartable. Most commands isolate per-ticker failures and write intermediate files so a later stage can continue even if an earlier source partly failed.

Key command groups:

- Collection: `collect`, `collect-swing`, `collect-seeking-alpha`, `alpaca-tickers`.
- Verification: `verify-events`, `verify-swing`, `audit-swing-alignment`.
- Sentiment: `download-model`, `score-events`, `score-swing-events`, `score-swing`.
- Feature building: `build-dataset`, `build-swing-datasets`, `build-event-swing-datasets`.
- Training/scoring: `train`, `train-event-swing`, `score-event-swing`, `predict-watchlist`.
- Live operation: `live-once`, `live-run`, `live-train-event`.
- Azure: `export-ohlcv-artifacts`, `azure-upload-artifacts`, `azure-publish-models`.

### Configuration

`src/market_predictor/config.py`

Loads `.env` secrets and exposes typed settings. This is where Alpaca, Reddit, RapidAPI, Seeking Alpha account-token, FinBERT, Azure, and runtime defaults become usable by code.

`src/market_predictor/app_config.py`

Loads non-secret TOML behavior from `configs/default.toml`.

`configs/default.toml`

Owns universe lists, sector groups, benchmark mapping, source behavior, Seeking Alpha endpoint templates, Reddit subreddit settings, and performance knobs.

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

`PurgedWalkForwardSplit` keeps validation later in time than training and leaves an embargo gap between train and test windows.

### Quota and Azure

`src/market_predictor/quota.py`

Tracks monthly RapidAPI usage locally so Seeking Alpha calls can be throttled before exhausting the plan.

`src/market_predictor/azure_store.py`

Uploads and downloads project artifacts to Azure Blob Storage. It supports either a connection string or managed identity/default credentials through Azure Identity.

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

Current model types:

- `daily_swing_2y_market_context_1d_max.joblib`: daily ticker direction, next trading day.
- `daily_swing_2y_market_context_5d_max.joblib`: daily ticker direction, next 5 trading days.
- `event_swing_2y_market_context_1d_prereaction_max.joblib`: event swing, next trading day, excludes post-event reaction fields.
- `event_swing_2y_market_context_5d_prereaction_max.joblib`: event swing, next 5 trading days, excludes post-event reaction fields.

The watchlist predictor combines model probabilities with recent catalysts, sentiment, price movement, sector/cap profile, and source coverage.

Latest verified training shape:

- Daily combined datasets: 143,456 rows across 187 historical tickers.
- Event combined dataset: 161,369 event rows across 187 historical tickers.
- Seeking Alpha full pull: 188 configured symbols, 11,315 SA events, 188 snapshot rows.
- Market-context proxy events: 31,539 FinBERT-scored rows.
- Daily 1-day validation: accuracy 0.524, 96 features.
- Daily 5-day validation: accuracy 0.513, 96 features.
- Event 1-day pre-reaction validation: accuracy 0.519, 64 features.
- Event 5-day pre-reaction validation: accuracy 0.517, 64 features.

These are modest statistical edges, not high-confidence forecasts. Treat the model output as a ranking/watchlist signal and combine it with risk controls, liquidity checks, and catalyst review.

## Live Pipeline Design

`live-once` is the normal production-style unit of work.

It does:

1. Collect recent events per ticker and per source.
2. Sanitize and deduplicate.
3. Score missing sentiment.
4. Build event features.
5. Score available active market-context daily and event models.
6. Curate matured labeled rows for future retraining.
7. Write run state and predictions.

`live-train-event` is guarded. It does not blindly retrain every night. It checks whether enough matured live rows exist, trains candidates, compares validation metrics, and promotes only when gates pass.

## Scheduling

Local Windows scheduling:

```powershell
.\scripts\register_live_midnight_task.ps1
.\scripts\register_live_train_task.ps1
```

Scripts:

- `scripts/run_live_midnight.ps1`: runs the midnight live collection/scoring cycle.
- `scripts/run_live_train_event.ps1`: runs guarded event-model retraining.
- `scripts/azure_nightly.sh`: container-friendly nightly sequence for Azure.

## Azure Deployment

Use Azure Blob Storage for project artifacts and Azure Container Apps Jobs for scheduled runs. Use Azure ML GPU compute or a stopped-when-idle GPU VM only for heavy FinBERT/retraining work.

Files:

- `Dockerfile`: container image for the Python CLI.
- `.dockerignore`: keeps local data, models, secrets, and venv out of the build context.
- `docs/azure_deployment_plan.md`: deployment recommendation, schedule, and operational rules.

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

1. Build or extend a dataset in `features.py`.
2. Add a training function in `model.py` if the target or validation differs.
3. Add a CLI command or extend an existing one in `cli.py`.
4. Write outputs to `models/` and metrics to `data/reports/`.
5. Do not promote until walk-forward validation is acceptable.

## Practical Debugging

Check source/API issues:

```powershell
market-predictor seeking-alpha-limits
market-predictor collect LUNR --days 3 --out data/raw/debug_lunr.parquet
```

Check event quality:

```powershell
market-predictor verify-events data/raw/debug_lunr.parquet --rewrite
```

Check a watchlist:

```powershell
market-predictor predict-watchlist --tickers "LUNR,MXL" --out data/reports/debug_watchlist.csv
```

Check Azure storage configuration:

```powershell
market-predictor azure-upload-artifacts --root data/artifacts
```
