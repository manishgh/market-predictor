# Market Predictor

Python prediction-intelligence project for ticker-level swing and daily momentum scoring using:

- Alpaca premium news, ticker universe, and market bars as the primary live source.
- Reddit API crawling as the required attention/community signal.
- Seeking Alpha via RapidAPI for SA news/analysis, earnings, and quant/rating snapshots.
- SEC company facts for keyless EPS/fundamental snapshots.
- Market-wide context from SPY/QQQ/sector ETF/news proxy events so global news can affect predictions even when it is not ticker-specific.
- FinBERT sentiment features plus price movement labels for next-day and swing-horizon prediction targets.

This is research and prediction tooling, not investment advice and not an automated trading system.

The repository produces prediction intelligence: probabilities, catalyst summaries, feature/audit context, and watchlist rankings. It does not own broker execution, portfolio state, final sizing, stops, exits, or order lifecycle. Those responsibilities belong in a trading/runtime system such as `trading_flow`.

## Architecture Documents

- [Implementation guide](docs/implementation_guide.md)
- [Azure deployment plan](docs/azure_deployment_plan.md)
- [Swing prediction intelligence architecture](docs/catalyst_confirmation_architecture.md)

## Source Strategy

Primary:

- **Alpaca premium**: active/tradable ticker universe, latest news, and historical bars. Use this for reliable, timestamped market data and recent ticker news.
- **Reddit API**: subreddit search and ticker mentions. Use this for retail attention, sentiment, score, comments, and upvote-ratio signals.
- **Seeking Alpha via RapidAPI**: SA-owned news/analysis, earnings, and quant/rating snapshots. This is the only quant/rating API path.
- **SEC EDGAR APIs**: keyless company facts, filings, and EPS/fundamental history.

Seeking Alpha premium account access:

- Do not paste your Seeking Alpha password into chat.
- Your premium subscription can be used manually to export/check data until June 17, 2026.
- Automated collection should use RapidAPI key-based access through `.env`.

## Setup

```powershell
cd C:\project\market-predictor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -e .
Copy-Item .env.example .env
```

Fill `.env` with Alpaca keys and any optional service keys.

Minimum runtime:

- Python 3.11 or newer.
- Alpaca keys for bars, ticker universe, and news.
- Reddit credentials if Reddit chatter should be collected.
- RapidAPI key if Seeking Alpha feeds should be collected.
- SEC needs no key, but `SEC_USER_AGENT` should contain real contact information.

Verify the CLI:

```powershell
market-predictor --help
```

## Repository Artifact Policy

The repository contains source code, configuration examples, scripts, and documentation only. Runtime secrets, downloaded market data, feature tables, trained model binaries, cached API responses, and generated reports stay out of Git.

Ignored local paths include:

- `.env`
- `.venv/`
- `data/`
- `models/`
- Python caches and log files

Use Azure Blob Storage or another artifact store for durable datasets, reports, and active model files.

## Quick Start

Download FinBERT once:

```powershell
market-predictor download-model
```

Run one live collection/scoring cycle for a small watchlist:

```powershell
market-predictor live-once --tickers "LUNR,MXL,RGTI" --lookback-days 3 --workers 4
```

Predict a watchlist using the active clean models:

```powershell
market-predictor predict-watchlist --tickers "LUNR,MXL,RGTI" --out data/reports/watchlist_latest.csv
```

Export project-owned OHLCV artifacts for Azure upload:

```powershell
market-predictor export-ohlcv-artifacts --tickers "LUNR,MXL,RGTI" --days 730 --timeframes 1d,1h
```

Upload project artifacts to Azure Blob Storage after Azure env vars are configured:

```powershell
market-predictor azure-upload-artifacts --root data/artifacts
market-predictor azure-publish-models --models-dir models
```

For implementation details and file responsibilities, read:

```text
docs/implementation_guide.md
docs/azure_deployment_plan.md
docs/rapidapi_mcp.md
```

## Access Required

Alpaca:

- Already configured from your TradingFlow local config.
- Required fields in `.env`: `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`, `ALPACA_STOCK_FEED=sip`, `ALPACA_TRADING_BASE_URL`.
- Used for `/v1beta1/news`, daily bars, and `/v2/assets` ticker universe.

Configuration:

- Secrets live in `.env`.
- Non-secret behavior lives in `configs/default.toml`: universe filters, enabled sources, Reddit subreddits/chatter settings, Seeking Alpha endpoint templates, default horizons, and watch-score weights.

Reddit:

- Create a Reddit app at `https://old.reddit.com/prefs/apps/`.
- Choose app type `script`.
- Required fields in `.env`: `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`, `REDDIT_PASSWORD`, `REDDIT_USER_AGENT`.
- `REDDIT_USER_AGENT` should identify the project and your Reddit username, for example `market-predictor/0.1 by myusername`.
- Default subreddits and chatter settings are in `configs/default.toml`.
- Reddit search can provide past post chatter within Reddit API listing limits, but it is not a guaranteed full historical archive. The collector also fetches top comments from matched posts and keeps only comments that mention the ticker.

Seeking Alpha RapidAPI:

- Subscribe to the Seeking Alpha API on RapidAPI basic tier.
- Required field in `.env`: `RAPIDAPI_KEY`.
- Optional account-token fields in `.env`: `SEEKING_ALPHA_ACCOUNT_EMAIL`, `SEEKING_ALPHA_ACCOUNT_PASSWORD`.
- Optional MCP setup template: `mcp/rapidapi-seeking-alpha.mcp.json.example`.
- MCP notes: `docs/rapidapi_mcp.md`.
- Defaults assume host `seeking-alpha.p.rapidapi.com`.
- Event feeds and snapshot feeds are configurable in `configs/default.toml`.
- Default event feeds include ticker news, press releases, SA analysis, and broad market/context feeds.
- Default snapshot feeds include metadata, profile, financials, analyst recommendations, analyst price targets, earnings, SEC filings, sector metrics, metric grades, and historical-price attempts where the API supports them.
- If RapidAPI changes parameter names, update the template values in config instead of changing Python code.
- Your current basic plan screenshot shows 200 requests/month, 1000 requests/hour, and 10240 MB/month bandwidth.
- The project caches Seeking Alpha analysis and ratings for 24 hours by default and tracks local monthly usage in `data/usage/rapidapi_usage.json`.
- Check local usage and the last RapidAPI limit headers with `market-predictor seeking-alpha-limits`.
- If an endpoint requires a Seeking Alpha account token, cache it with `market-predictor seeking-alpha-token`. The token is stored locally and never printed.

SEC:

- No API key required.
- Keep `SEC_USER_AGENT` descriptive and include contact info.

## Download FinBERT

```powershell
market-predictor download-model
```

The default model is `ProsusAI/finbert`. Override with `FINBERT_MODEL`.

## Collect Data

```powershell
market-predictor alpaca-tickers --out data/universe/alpaca_tickers.csv
market-predictor collect AAPL --days 90 --out data/raw/aapl_events.parquet
market-predictor collect-seeking-alpha AAPL --out data/external/seeking_alpha_quant.csv
market-predictor build-dataset AAPL --events data/raw/aapl_events.parquet --out data/features/aapl_daily_1d.parquet --horizon-days 1 --seeking-alpha data/external/seeking_alpha_quant.csv
market-predictor train --dataset data/features/aapl_daily_1d.parquet --model-out models/aapl_direction_1d.joblib --horizon-days 1
market-predictor watch AAPL --model models/aapl_direction_1d.joblib --horizon-days 1
```

For a next-week model:

```powershell
market-predictor build-dataset AAPL --events data/raw/aapl_events.parquet --out data/features/aapl_daily_5d.parquet --horizon-days 5 --seeking-alpha data/external/seeking_alpha_quant.csv
market-predictor train --dataset data/features/aapl_daily_5d.parquet --model-out models/aapl_direction_5d.joblib --horizon-days 5
market-predictor predict AAPL --model models/aapl_direction_5d.joblib --days 30
```

The prediction label uses a tradable next-session open entry reference, not same-day close. Use `--horizon-days 1` for tomorrow/swing watch work and `--horizon-days 5` for a next-week view.

## Swing Engine Workflow

The configured swing universe starts with your seed symbols:

```text
POET, MXL, RDW, LASE, RGTI, MRVL
```

It also includes a broader high-beta US-listed and liquid ADR/ETF universe in `configs/default.toml`, currently about 188 configured symbols across technology, semiconductors, space/defense/quantum, biotech, fintech/crypto, consumer/EV/meme, plus benchmark ETFs.

Bulk workflow:

```powershell
market-predictor swing-universe --out data/universe/swing_candidates.csv
market-predictor collect-swing --days 180 --out-dir data/raw/swing
market-predictor score-swing --raw-dir data/raw/swing
market-predictor build-swing-datasets --horizon-days 1 --raw-dir data/raw/swing --out-dir data/features/swing
market-predictor rank-swing --horizon-days 1 --feature-dir data/features/swing --out data/reports/swing_watch_rank.csv
```

Current redesigned training artifacts use two years of data and market-context features:

```powershell
market-predictor build-market-context-from-proxies --raw-dir data/raw/uslisted_6sector_2y_clean --out data/external/market_context/market_context_events.parquet
market-predictor score-events --events data/external/market_context/market_context_events.parquet --out data/external/market_context/market_context_events_scored.parquet
market-predictor build-swing-datasets --horizon-days 1 --raw-dir data/raw/uslisted_6sector_2y_clean --out-dir data/features/daily_swing_2y_market_context --workers 8
market-predictor build-swing-datasets --horizon-days 5 --raw-dir data/raw/uslisted_6sector_2y_clean --out-dir data/features/daily_swing_2y_market_context --workers 8
market-predictor combine-swing-datasets --feature-dir data/features/daily_swing_2y_market_context --horizon-days 1 --out data/features/daily_swing_combined_2y_market_context_1d.parquet
market-predictor combine-swing-datasets --feature-dir data/features/daily_swing_2y_market_context --horizon-days 5 --out data/features/daily_swing_combined_2y_market_context_5d.parquet
market-predictor train --dataset data/features/daily_swing_combined_2y_market_context_1d.parquet --model-out models/daily_swing_2y_market_context_1d_candidate.joblib --horizon-days 1 --max-iter 320 --learning-rate 0.035
market-predictor train --dataset data/features/daily_swing_combined_2y_market_context_5d.parquet --model-out models/daily_swing_2y_market_context_5d_candidate.joblib --horizon-days 5 --max-iter 320 --learning-rate 0.035
```

Use a custom list:

```powershell
market-predictor collect-swing --tickers "POET,MXL,RDW,LASE,RGTI,MRVL,IONQ,QBTS" --days 180
```

Stage separation:

- `collect-swing`: API/data download only by default. Alpaca, Reddit, and Seeking Alpha failures are isolated per source and per ticker. Uses parallel workers for I/O.
- `score-swing`: FinBERT scoring only. Loads the model once, uses GPU if PyTorch detects CUDA, scores all ticker texts in batches, then writes per-ticker files.
- `build-swing-datasets`: daily/hourly price joins, event reaction features, technical features, and labels. Uses parallel workers per ticker.
- `rank-swing`: latest watch ranking across built datasets.

Performance knobs live in `configs/default.toml`:

```toml
[performance]
max_workers = 6
finbert_batch_size = 32
```

Override workers per command:

```powershell
market-predictor collect-swing --days 180 --workers 8
market-predictor score-swing --batch-size 64
market-predictor build-swing-datasets --horizon-days 1 --workers 8 --no-with-seeking-alpha
```

The engine separates event timing buckets:

- `pre_market`: news before 9:30 ET, with open gap and day return features.
- `intraday`: news during regular hours, with first-2-hour hourly-candle reaction and to-close reaction.
- `after_hours`: news after 16:00 ET, rolled to the next feature date with next-open gap and next-day return features.

Hourly reaction features require Alpaca bars. If hourly bars are unavailable, the daily gap/day-return features still build.

The active watchlist defaults now use:

```text
models/daily_swing_2y_market_context_1d_max.joblib
models/daily_swing_2y_market_context_5d_max.joblib
models/event_swing_2y_market_context_1d_prereaction_max.joblib
models/event_swing_2y_market_context_5d_prereaction_max.joblib
```

The latest `predict-watchlist` output writes three files: a readable CSV, a raw CSV, and a field-definition CSV explaining each probability column.

## Event Swing Model Workflow

This is the cleaner event-level workflow for learning how news/chatter/fundamental events map to next-day and next-week moves.

```powershell
market-predictor collect-swing --days 730 --out-dir data/raw/swing --workers 8
market-predictor verify-swing --raw-dir data/raw/swing --rewrite
market-predictor score-swing-events --raw-dir data/raw/swing --out-dir data/raw/swing_scored
market-predictor build-event-swing-datasets --raw-dir data/raw/swing_scored --out-dir data/features/event_swing --workers 8
market-predictor combine-event-swing-datasets --feature-dir data/features/event_swing --out data/features/event_swing_all.parquet
market-predictor train-event-swing --dataset data/features/event_swing_all.parquet --model-out models/event_swing_1d.joblib --target-col target_next_1d_up
market-predictor score-event-swing --dataset data/features/event_swing_all.parquet --model models/event_swing_1d.joblib --out data/reports/event_swing_scores.csv
```

The active production-style command for daily use is usually `predict-watchlist`, which collects the recent 2-3 day context, joins latest bars/profile/news/chatter features, and scores the clean active models.

## Live Nightly Pipeline

Run one managed live cycle:

```powershell
market-predictor live-once --lookback-days 3 --workers 8
```

Run continuously in the foreground:

```powershell
market-predictor live-run --poll-seconds 3600 --lookback-days 3 --workers 8
```

Monitor a watchlist for technical indicator alerts:

```powershell
market-predictor monitor-alerts --tickers "MSFT,NVDA,RGTI" --days 180 --poll-seconds 900
```

This writes the latest alert state to `data/live/alerts/latest_alerts.csv` and appends every run to `data/live/alerts/alert_history.csv`. Alert rules currently cover MACD signal-line crosses, EMA20 reclaim/loss, EMA20/EMA50 crosses, RSI oversold/overbought reversals, and volume-confirmed breakouts/breakdowns.

Backtest indicator alerts against existing labeled feature data:

```powershell
market-predictor backtest-alerts --horizon-days 1
market-predictor backtest-alerts --horizon-days 5
```

## Volatile Mover Research Pipeline

Build a news-aware daily/weekly volatile mover dataset from an audited universe:

```powershell
market-predictor build-volatile-dataset `
  --universe data/universe/volatile_mover_research_universe_20260704.csv `
  --out data/features/volatile_mover_daily_20260704.parquet `
  --audit-out data/reports/volatile_mover_dataset_audit_20260704.csv
```

Train a target-specific candidate model with purged walk-forward validation:

```powershell
market-predictor train-volatile-model `
  --target-col target_next_week_big_up `
  --model-out models/volatile_mover_next_week_big_up_20260704_candidate.joblib `
  --predictions-out data/reports/volatile_mover_next_week_big_up_oos_predictions_20260704.csv `
  --metrics-out data/reports/volatile_mover_next_week_big_up_metrics_20260704.csv
```

Score the latest row per ticker:

```powershell
market-predictor score-volatile-latest `
  --model models/volatile_mover_next_week_big_up_20260704_candidate.joblib `
  --out data/reports/volatile_mover_latest_scores_20260704.csv
```

The volatile mover schema is separate from the large-cap swing schema. It keeps news/catalyst features, source counts, sentiment, market context, price/volume pressure, and theme buckets, then labels next-day and next-week big-move outcomes.

Retrain only when enough matured live labels exist:

```powershell
market-predictor live-train-event --live-dir data/live
```

Register Windows scheduled tasks:

```powershell
.\scripts\register_live_midnight_task.ps1
.\scripts\register_live_train_task.ps1
```

The intended schedule is:

- `00:00` local time: collect, validate, score sentiment, build live features, and write predictions.
- `00:45` local time: guarded retraining if enough matured labels exist.

## Azure

Azure is project-specific. This project stores its own artifacts under `AZURE_BLOB_PREFIX`, defaulting to:

```text
market-predictor
```

Recommended deployment is:

```text
Azure Blob Storage + Azure Container Apps Jobs + Azure ML GPU compute on demand
```

The container entrypoint script is:

```text
scripts/azure_nightly.sh
```

Build context files:

```text
Dockerfile
.dockerignore
```

The deployment rationale and job schedule are in `docs/azure_deployment_plan.md`.

## Reddit Signals

Reddit is treated as an attention and sentiment source, not as normal news. The feature set includes:

- Reddit mention count by day.
- Reddit-only FinBERT sentiment.
- Sum of post scores.
- Sum of comment counts.
- Mean upvote ratio.

For swing trades, these features are most useful when they diverge from price/volume: high attention plus improving sentiment after a down move, or high attention plus negative sentiment into elevated volume.

## Seeking Alpha Quant Feed

Do not scrape account-gated Seeking Alpha pages unless your license explicitly permits it. RapidAPI snapshots and licensed exports are stored under `data/external/` and cached under `data/cache/`.

```text
data/external/seeking_alpha_quant.csv
```

The legacy daily-model quant compatibility columns are:

```text
timestamp,ticker,quant_rating,valuation,growth,profitability,momentum,eps_revision,eps_actual,eps_estimate
```

The RapidAPI adapter writes this CSV through `market-predictor collect-seeking-alpha`.

## Useful Official Docs

- Alpaca news endpoint: https://docs.alpaca.markets/us/reference/news-3
- SEC EDGAR data APIs: https://www.sec.gov/search-filings/edgar-application-programming-interfaces
- Reddit API docs: https://www.reddit.com/dev/api/
- Reddit app registration: https://old.reddit.com/prefs/apps/
- Seeking Alpha RapidAPI page: https://rapidapi.com/apidojo/api/seeking-alpha
