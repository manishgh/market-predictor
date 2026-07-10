# Market Predictor

Python prediction-intelligence project for ticker-level swing, daily momentum, and intraday setup scoring using:

- Alpaca premium news, ticker universe, and market bars as the primary live source.
- Reddit API crawling as the required attention/community signal.
- Seeking Alpha via RapidAPI for SA news/analysis, earnings, and quant/rating snapshots.
- SEC company facts for keyless EPS/fundamental snapshots.
- Market-wide context from SPY/QQQ/sector ETF/news proxy events so global news can affect predictions even when it is not ticker-specific.
- FinBERT sentiment features plus price movement labels for next-day and swing-horizon prediction targets.

This is research and prediction tooling, not investment advice and not an automated trading system.

The repository produces prediction intelligence: probabilities, catalyst summaries, feature/audit context, and watchlist rankings. It does not own broker execution, portfolio state, final sizing, stops, exits, or order lifecycle. Those responsibilities belong in a trading/runtime system such as `trading_flow`.

## Current Model State (2026-07-10)

Model lifecycle state comes from each artifact's `.manifest.json`; a filename such as `*_max.joblib` does not mean promoted.

| Serving view | Artifact / family | State | Current evidence |
| --- | --- | --- | --- |
| Swing 5D | `sp500_6m_next_week_big_up_v2_20260708_candidate.joblib` | **Promoted, conditional** | 499 tickers, 45,908 OOS rows, ROC AUC 0.7126, top-decile lift 2.5936. Promoted on 2026-07-08 under the earlier classification/alignment gates; must be re-audited under the newer profitability, drawdown, regime, and catalyst gates before real-capital use. |
| Swing 1D | `sp500_6m_next_day_big_up_v2_20260708_candidate.joblib` | Candidate | ROC AUC 0.6657 and top-decile lift 2.4850. Not promoted. |
| Intraday 12 bars, API default | 2026-07-09 technical ablation | Candidate | ROC AUC 0.6014 and lift 1.4719. Fails current AUC/lift gates. |
| Intraday opening V2 | 2026-07-10 non-overlapping, cost-aware experiment | Candidate; promotion rejected | Best exact-path AUC 0.5806, lift 1.1764, selected net return -0.184% per trade, profit factor 0.7076, max drawdown 30.28%. |
| Older daily/event `*_max.joblib` files | Legacy swing/event families | Baseline/research | These artifacts predate registry manifests and current promotion gates. They are not formally promoted. |

Production API implications:

- `require_promoted: true` permits the default 5D swing route because its manifest is promoted.
- The 1D swing and intraday routes are rejected as candidates; there is no silent fallback.
- Unified mode may return the promoted swing view plus an explicit intraday error until an intraday model passes promotion.
- Candidate scoring requires an explicit research override and must not be treated as a live trade instruction.

The next valid intraday promotion attempt requires new matured shadow data after 2026-07-08, predeclared model/threshold choices, and all current promotion audits. See [Intraday model promotion](docs/intraday_model_promotion.md).

## Architecture Documents

- [Implementation guide](docs/implementation_guide.md)
- [Azure deployment plan](docs/azure_deployment_plan.md)
- [Market prediction intelligence architecture](docs/catalyst_confirmation_architecture.md)
- [Intraday model promotion](docs/intraday_model_promotion.md)
- [TradingFlow integration plan](docs/trading_flow_integration_plan.md)

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

Start the prediction API:

```powershell
market-predictor serve-api --host 127.0.0.1 --port 8000
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

## Prediction API

The API is a serving layer over promoted model artifacts and curated feature datasets. It does not train models, collect live news, place trades, or bypass promotion gates.

Endpoints:

- `GET /v1/health`
- `POST /v1/predictions/swing`
- `POST /v1/predictions/intraday`
- `POST /v1/predictions/unified`
- `POST /v1/replays/investment`

Example request:

```json
{
  "tickers": ["MSFT", "NVDA"],
  "mode": "unified",
  "data_source": "live",
  "horizon": "auto",
  "as_of": "2026-07-09T20:15:00Z",
  "require_promoted": true
}
```

`as_of` is optional, but when supplied it must include a timezone or UTC offset. Daily close-derived rows are available only from 16:00 America/New_York on their trading date. Intraday rows are available only after the inferred bar closes. These rules also apply to historical replay, preventing a request from reading a feature row that was not tradable at the requested time.

Use `horizon: "auto"` for unified prediction so each model resolves its native horizon. Current built-in routes are `1d` and `5d` for swing and `12b` for the 5-minute intraday entry model. An explicit horizon is rejected when it conflicts with the selected model target. Responses include `resolved_horizons` and each model's `resolved_horizon` for auditability.

The unified response returns separate `swing` and `intraday` model views plus a final orchestration signal. It does not average unrelated model probabilities into one opaque number. Readiness reports daily and intraday history separately and treats unknown feed tiers as unproven; only explicit SIP/consolidated provenance satisfies the full-volume gate.

`data_source: "curated"` scores registered research datasets. `data_source: "live"` scores only the registered snapshots at `data/live/features/swing.parquet` and `data/live/features/intraday.parquet`. Live snapshots require a matching sidecar manifest, SHA-256 integrity, explicit feed provenance, and a generation time within the configured freshness window. Arbitrary filesystem paths are not accepted with live mode.

The nightly `live-once` cycle publishes the rolling swing snapshot after source-isolated collection, sanitization, FinBERT scoring, daily feature construction, and volatile-schema enrichment. Publish an audited intraday feature table from the 5-minute pipeline with:

```powershell
market-predictor publish-live-features `
  --mode intraday `
  --input-path data/features/intraday_latest_enriched.parquet `
  --live-dir data/live `
  --price-feed sip
```

Catalyst evidence is returned separately from the model probability. `probability` is the unmodified model output; `decision_score` applies only a transparent ranking adjustment. The `catalyst` object reports confirmation, conflict, veto, mixed, or absent evidence using relevance, sentiment, recency, source diversity, generic-headline rate, and material-event taxonomy. This preserves technical-model auditability while allowing strong negative material catalysts to block a long entry.

Every prediction served through the top-level API is written as an immutable, content-addressed JSON snapshot under `data/predictions/snapshots/`. The response returns `snapshot_id` and `snapshot_sha256`; both identify the exact request, response, model hashes, resolved horizons, feature cutoff, and generation time used for later outcome evaluation.

Replay an investment from a stored prediction:

```json
{
  "snapshot_id": "<64-character snapshot SHA-256>",
  "ticker": "MSFT",
  "model_view": "swing",
  "evaluation_as_of": "2026-07-17T21:00:00Z",
  "initial_capital": 10000,
  "slippage_bps": 5,
  "commission_bps": 0,
  "force_entry": false
}
```

The replay uses Alpaca adjusted bars, enters at the next tradable bar open, exits at the last completed bar at `evaluation_as_of`, applies configured slippage and commissions on both sides, and evaluates SPY and QQQ over the stock's exact entry/exit window. It returns ending value, P&L, return, and excess return versus both benchmarks.

Replay refuses to invest when the snapshot has invalid readiness, missing model identity, a model created after the decision time, or training data extending beyond the decision time. A neutral signal returns `not_entered`; `force_entry` is available for explicit what-if research but cannot override invalid readiness or temporal leakage gates.

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

- `collect-swing`: API/data download only by default. Alpaca, Finviz, Reddit, Seeking Alpha, and SEC failures are isolated per source and per ticker. Uses parallel workers for I/O. Seeking Alpha is enabled by default when RapidAPI credentials are configured; use `--no-seeking-alpha` only for an explicit quota outage or diagnostic run.
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

Legacy alert commands are deprecated and must not be scheduled in new deployments:

```powershell
market-predictor monitor-alerts --tickers "MSFT,NVDA,RGTI" --days 180 --poll-seconds 900
```

These commands remain temporarily for migration verification only. Runtime alerting, persistence, deduplication, acknowledgement, and web/mobile notification belong to `trading_flow`. The commands and `alerts.py` are scheduled for removal after rule parity is verified in TradingFlow.

Legacy research comparison commands:

```powershell
market-predictor backtest-alerts --horizon-days 1
market-predictor backtest-alerts --horizon-days 5
```

Do not build new alert behavior in this repository. See [TradingFlow integration plan](docs/trading_flow_integration_plan.md).

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

## Entry / Exit Path Models

Direction models answer whether a stock is likely to move. Entry/exit path models answer whether a long setup is tradable from the next bar's open: does price hit an ATR profit target before an ATR stop inside the configured horizon?

Build swing entry/exit labels from daily feature rows:

```powershell
market-predictor build-entry-exit-dataset `
  --input data/features/volatile_mover_daily_20260704.parquet `
  --horizon-bars 5 `
  --take-profit-atr 1.5 `
  --stop-loss-atr 1.0 `
  --bar-kind swing `
  --out data/features/entry_exit_swing_5b_20260704.parquet `
  --audit-out data/reports/entry_exit_swing_5b_audit_20260704.csv
```

Train separate entry and exit-risk models:

```powershell
market-predictor train-entry-exit-model `
  --dataset data/features/entry_exit_swing_5b_20260704.parquet `
  --target-col target_entry_success_5b `
  --model-out models/entry_exit_swing_entry_success_5b_20260704_candidate.joblib

market-predictor train-entry-exit-model `
  --dataset data/features/entry_exit_swing_5b_20260704.parquet `
  --target-col target_exit_risk_5b `
  --model-out models/entry_exit_swing_exit_risk_5b_20260704_candidate.joblib
```

Score the latest row per ticker:

```powershell
market-predictor score-entry-exit-latest `
  --dataset data/features/entry_exit_swing_5b_20260704.parquet `
  --model models/entry_exit_swing_entry_success_5b_20260704_candidate.joblib `
  --out data/reports/entry_exit_swing_entry_latest_20260704.csv
```

The same commands work for intraday datasets when the input rows are hourly or 5-minute OHLCV features. Labels always enter at the next bar open and evaluate only future high/low bars, which prevents same-bar leakage. Build labels from the complete consecutive bar stream, never from an already setup-filtered table.

Opening-session V2 example for a 60-minute horizon on 5-minute bars:

```powershell
market-predictor build-entry-exit-dataset `
  --input data/features/intraday_full_5m.parquet `
  --context data/features/intraday_point_in_time_context.parquet `
  --horizon-bars 12 `
  --bar-kind 5min `
  --session-scope opening `
  --min-setup-score 2 `
  --setup-cooldown-bars 13 `
  --round-trip-cost-bps 10 `
  --out data/features/entry_exit_intraday_opening_v2.parquet `
  --audit-out data/reports/entry_exit_intraday_opening_v2_audit.csv
```

`--session-scope opening` means 09:30 through 11:29 ET. Cooldown is measured in original bars, not filtered row positions, and is never shorter than `horizon_bars + 1`. The optional context join only adds approved missing model features; it cannot replace OHLCV or labels. V2 emits raw and cost-adjusted horizon returns, modeled target/stop/timeout realized returns, `target_entry_success_*`, `target_exit_risk_*`, and `target_net_positive_*`.

Controlled estimator comparisons use the same purged walk-forward folds:

```powershell
market-predictor train-entry-exit-model `
  --dataset data/features/entry_exit_intraday_opening_v2.parquet `
  --target-col target_entry_success_12b `
  --feature-set technical `
  --estimator hist_gradient_boosting `
  --model-out models/intraday_opening_candidate.joblib
```

Supported estimators are `hist_gradient_boosting`, `extra_trees`, and `logistic`. Selecting the best estimator on an inspected OOS interval does not make it promotable; it still requires an untouched shadow interval and all promotion gates.

Before promotion, build production-readiness audits from the feature table and out-of-sample predictions:

```powershell
market-predictor audit-promotion-readiness `
  --dataset data/features/entry_exit_swing_5b_20260704.parquet `
  --predictions data/reports/entry_exit_swing_entry_success_5b_oos_predictions_20260704.csv `
  --out-prefix data/reports/entry_exit_swing_entry_success_5b_promotion_20260704
```

The audit writes separate profitability, selected-trade, market-regime, and catalyst/news CSVs. `promote-model` can require those files so a model is not promoted on ROC AUC alone. The default gate checks out-of-sample selected-trade return, profit factor, drawdown, market-regime coverage, catalyst/news presence, and news/candle alignment.

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
