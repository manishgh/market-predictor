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

## Current Model State (2026-07-23)

Candidate identity comes from an immutable `.manifest.json`. Effective promoted state exists only when a content-addressed promotion attestation verifies the candidate, evidence manifest, causal identity chain, predeclared baseline/hypothesis, untouched-shadow confidence interval, gate configuration, and build/approver identities. Unregistered, unattested, or hash-mismatched artifacts cannot be served.

| Serving view | Artifact / family | State | Current evidence |
| --- | --- | --- | --- |
| Swing 5D | Canonical `swing.model.v1` | Implementation complete; no promoted artifact | Point-in-time dataset, exact five-session labels, purged walk-forward, unseen-ticker holdout, calibration, economics, drawdown, catalyst, alignment, provenance, and 4 GiB gates are implemented. Real-data training and promotion have not passed yet. |
| Legacy swing 1D/5D volatile models | Pre-C4 artifacts | Deprecated and not serveable | Their feature/target schemas do not satisfy the canonical C4 contract, regardless of an older manifest status. |
| Intraday 60m | Canonical `intraday.model.v1` | Implementation complete; no promoted artifact | Completed 5-minute decisions, exact next-available 1-minute entry/path labels, separate opportunity/downside estimators, purged walk-forward, unseen-ticker holdout, calibration, economics, drawdown, catalyst-overlay, alignment, provenance, and 4 GiB gates are implemented. Real-data training and promotion have not passed yet. |
| Legacy intraday 12 bars | 2026-07-09 technical ablation | Candidate; not serveable | ROC AUC 0.6014 and lift 1.4719. It predates the canonical C5 contract and fails its historical gates. |
| Intraday opening V2 | 2026-07-10 non-overlapping, cost-aware experiment | Candidate; promotion rejected | Best exact-path AUC 0.5806, lift 1.1764, selected net return -0.184% per trade, profit factor 0.7076, max drawdown 30.28%. |
| Intraday V3 R1 | 2026-07-20 grouped XGBoost ranker | Candidate; promotion rejected | Walk-forward/holdout NDCG@10 0.4930/0.5123, but top-10 cost-adjusted excess return is -0.0715%/-0.0764%. |
| Intraday V3 O1 | 2026-07-21 fixed ticker-catalyst overlay on R1 | Research ablation; rejected | Walk-forward top-10 excess return improves from -0.0574% to -0.0487%, but ticker holdout worsens from -0.0642% to -0.0669%; both paired confidence intervals include zero. |
| Intraday V4-H1 120m | 2026-07-21 exact-path B0/R1 experiment | Research candidates; rejected | R1 top-10 cost-adjusted excess return is -0.0802%/-0.0629% walk-forward/holdout. The longer horizon does not cover costs. |

Production API implications:

- Production routes are server-owned and always require a promoted, hash-verified artifact.
- The configured swing route is deliberately not ready until a real canonical candidate passes every C4 promotion gate; there is no legacy fallback.
- Unified mode may return explicit swing and intraday errors until each requested view has its own promoted canonical artifact.
- Candidate scoring is available only through research commands or an explicitly constructed test service, never through the HTTP request contract.
- R4 promotion and local release infrastructure is complete: immutable candidate manifests and attestations, predeclared hypotheses, one-use shadow evidence, paired session-block confidence gates, versioned local releases, atomic activation, and verified rollback are implemented. This does not change the model state above; no real canonical model has passed promotion.
- Azure publication, synchronization, rollback, and disaster-recovery rehearsal are `environment_pending` and are not evidence for R4 completion.
- Repository-wide Ruff and strict mypy pass, and the full 263-test suite is green at the R4 local-release checkpoint.

The next valid intraday promotion attempt requires a new predeclared development hypothesis that first passes both economic scopes, followed by matured shadow data after 2026-07-08 and all current promotion audits. See [Intraday model promotion](docs/intraday_model_promotion.md).

ML V3 checkpoints C1-C8 are complete with no selected candidate. B0/B1/B2/R1 and the fixed O1 catalyst overlay all have negative cost-adjusted top-10 excess return, while D1 is near-random as a downside gate. R2 is unavailable because every frozen C8 row lacks microstructure inputs; the system does not impute them. O1 is rejected on paired walk-forward and ticker-holdout evidence. C9 shadow evaluation remains closed because there is no candidate to promote or serve.

The post-C8 failure-attribution audit motivated V4-H1: a 120-minute primary target and decision stride with the same universe, features, costs, and R1 family. Its first dataset audit exposed 11,781 rank-eligible rows whose 24 observed bars spanned more than 120 wall-clock minutes. The labeler now requires a contiguous exact five-minute path and persists `ml_v3.labels.v2`; the invalid v1 dataset was never trained.

The corrected V4-H1 fingerprint contains 505,049 physical rows and 495,513 rank-eligible rows over 474 sessions. B0 and R1 remain negative after costs in both development scopes, so V4-H1 is rejected and shadow remains closed. See the [failure-attribution card](docs/model_cards/v3_c8_failure_attribution_20260721.md) and [V4-H1 card](docs/model_cards/v4_h1_120m_20260721.md).

## Architecture Documents

- [Implementation guide](docs/implementation_guide.md)
- [Production ML rebuild plan](docs/production_ml_rebuild_plan.md)
- [Azure deployment plan](docs/azure_deployment_plan.md)
- [Market prediction intelligence architecture](docs/catalyst_confirmation_architecture.md)
- [Intraday model promotion](docs/intraday_model_promotion.md)
- [ML model V3 improvement plan](docs/ml_model_v3_plan.md)
- [TradingFlow integration plan](docs/trading_flow_integration_plan.md)
- [Legacy alert rule parity](docs/legacy_alert_rule_parity.md)

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

Install the optional grouped-ranking dependency for R1 training:

```powershell
python -m pip install -e ".[ranking]"
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

V3 development workflow after the raw data audit passes:

```powershell
market-predictor build-v3-features --bars data/curated/v3_bars.parquet --benchmarks data/curated/v3_benchmarks.parquet --source-availability data/curated/v3_source_availability.parquet --out data/features/v3_features_latest.parquet
market-predictor build-v3-labels --bars data/features/v3_features_latest.parquet --benchmarks data/curated/v3_benchmarks.parquet --out data/features/v3_training_latest.parquet
market-predictor train-v3-models --dataset data/features/v3_training_latest.parquet --output-dir models/v3/candidates
market-predictor audit-v3-ranking --predictions data/reports/v3_oof_predictions_latest.parquet --opportunity-family R1 --downside-family D1
```

The label builder drops any decision whose maximum configured path is not contiguous at the declared bar interval. The training command refuses shadow rows, non-SIP volume provenance, future feature availability, cross-session labels, malformed ranking groups, and stale feature schemas. It writes per-family candidate manifests, walk-forward OOF predictions, deterministic ticker-holdout evidence, and a fold feature-coverage audit. One family failure is reported without discarding successful families; the command exits nonzero if any requested family failed.

`audit-v3-ranking` fits the chosen D1 calibrator on earlier OOF sessions and evaluates top-k economics only on later sessions. It requires calibrated downside probabilities and independent event IDs, and resamples whole sessions for confidence intervals. The calibration method, candidate family, risk threshold, and promotion thresholds must be frozen in C8 before the audit can be used for model selection or shadow evaluation.

Before C8, run the fail-closed development gate:

```powershell
market-predictor build-v3-sp500-point-in-time-universe --current-snapshot data/universe/sp500_current_20260708.csv --start-date 2024-07-09 --cutoff-date 2026-07-08 --out data/universe/sp500_point_in_time_20240709_20260708.parquet --raw-dir data/raw/index_membership/spglobal_20240709_20260708 --audit-out data/reports/sp500_point_in_time_20240709_20260708_audit.json
market-predictor audit-v3-development-readiness --bars data/artifacts/ohlcv/v3_sp500_current_730d_20260708/5m --universe data/universe/sp500_point_in_time_20240709_20260708.parquet --benchmark-dir data/artifacts/ohlcv/v3_development_benchmarks_730d_20260708/5m --out data/reports/v3_development_readiness_pit_20260711.json
```

The universe builder hashes official S&P Global add/drop announcements, joins Alpaca name-change events, and reverses those events from the frozen constituent anchor. As of 2026-07-11, the local development audit passes with 546 point-in-time symbols, 501 sessions, SIP provenance, non-overlapping membership windows, bars for every historical member, and all 13 market/sector benchmarks. This establishes data readiness only; no V3 candidate is selected or promoted by this audit.

Build the monthly development rows through the hash-verified, XNYS-calendar-aware path, then train only from that registered directory:

```powershell
market-predictor build-v3-development-dataset --bars-dir data/artifacts/ohlcv/v3_sp500_current_730d_20260708/5m --benchmark-dir data/artifacts/ohlcv/v3_development_benchmarks_730d_20260708/5m --memberships data/universe/sp500_point_in_time_20240709_20260708.parquet --technical-dir data/work/v3_c8_technical_20260711 --out-dir data/features/v3_c8_development_20260711_v9 --decision-start-date 2024-08-09 --minimum-cross-section 300 --decision-stride-bars 12 --reuse-technical
market-predictor train-v3-models --dataset data/features/v3_c8_development_20260711_v9 --families R1 --max-training-memory-gb 4
```

The loader rejects missing, modified, or unregistered monthly shards and carries the dataset fingerprint into training evidence. It projects only required training/audit columns; the trainer compacts features to `float32`, releases fold models, and enforces a configurable process-memory guard. The completed C8 dataset has 1,063,587 rows across 24 months. B0, B1, B2, R1, D1, and the external O1 catalyst overlay were evaluated and rejected; R2 could not be evaluated because the frozen rows contain no microstructure observations. See the [B0](docs/model_cards/v3_c8_b0_20260711.md), [B1](docs/model_cards/v3_c8_b1_20260711.md), [B2](docs/model_cards/v3_c8_b2_20260711.md), [R1](docs/model_cards/v3_c8_r1_20260720.md), [D1](docs/model_cards/v3_c8_d1_20260711.md), and [O1](docs/model_cards/v3_c8_o1_20260721.md) model cards.

V4-H1 was built with `--horizons 6,12,24 --primary-horizon-bars 24 --decision-stride-bars 24`. The corrected v2 dataset fingerprint is `c2906f10b543327cc265798ecd81e019c5365dc9ede3e432b33ba881970cc612`. Its audit verifies all 24 shard hashes, exact 120-minute exits on every row, exact 120-minute eligible decision cadence, SIP provenance, PIT groups, and the development cutoff. B0/R1 training stayed below 1.96 GiB; both candidates were rejected without opening shadow data.

O1 remains outside the estimator. Historical catalyst scoring is resumable and provenance-bound:

```powershell
market-predictor score-swing-events --tickers $tickers --raw-dir data/raw/sp500_6m_20260708 --out-dir data/raw/sp500_6m_20260708_scored --text-mode title_summary --max-length 128 --batch-size 64
market-predictor audit-v3-o1-overlay --predictions data/reports/v3_c8_r1_oof_20260720.parquet --event-dir data/raw/sp500_6m_20260708_scored --coverage-start 2026-01-09T00:00:00Z --coverage-end 2026-07-08T23:59:59Z --availability-policy provider_publication_backfill
```

Publication-time backfill is always research-only. A global-context file is accepted only when its first and last available events cover the declared interval within the configured boundary tolerance.

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

Collect and sentiment-score a small research universe:

```powershell
market-predictor collect-swing --tickers "LUNR,MXL,RGTI" --days 30 --out-dir data/raw/research --workers 4
market-predictor score-swing-events --tickers "LUNR,MXL,RGTI" --raw-dir data/raw/research --out-dir data/raw/research_scored
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
```

Azure serving publication, hydration, rollback, and disaster-recovery rehearsal are
`environment_pending` and are not exposed as production CLI commands. The verified
local release repository is the current serving authority.

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
```

Data collection is separate from training and serving. Use the volatile-mover or V3 workflows below for model research; use the prediction API only after an audited live feature snapshot has been published.

## Prediction API

The API is a serving layer over server-registered promoted model artifacts and live feature snapshots. It does not train models, collect live news, place trades, or bypass promotion gates.

Endpoints:

- `GET /v1/health/live`
- `GET /v1/health/ready`
- `POST /v1/predictions/swing`
- `POST /v1/predictions/intraday`
- `POST /v1/predictions/unified`
- `POST /v1/replays/investment`

Example request:

```json
{
  "tickers": ["MSFT", "NVDA"],
  "mode": "unified",
  "horizon": "auto",
  "as_of": "2026-07-09T20:15:00Z"
}
```

`as_of` is optional, but when supplied it must include a timezone or UTC offset. Daily close-derived rows are available only from 16:00 America/New_York on their trading date. Intraday rows are available only after the inferred bar closes. These rules also apply to historical replay, preventing a request from reading a feature row that was not tradable at the requested time.

Use `horizon: "auto"` for unified prediction so each server route resolves its native horizon. The only configured production route is `5d` swing, and it remains not-ready until a real artifact is promoted. Canonical intraday uses a `60m` horizon and remains unregistered until a real C5 candidate passes promotion. An explicit unsupported horizon is rejected; a registered model whose target conflicts with its route is also rejected. Responses include `resolved_horizons` and each model's `resolved_horizon` for auditability.

The unified response returns separate `swing` and `intraday` model views plus a final orchestration signal. It does not average unrelated model probabilities into one opaque number. Readiness reports daily and intraday history separately and treats unknown feed tiers as unproven; only explicit SIP/consolidated provenance satisfies the full-volume gate. A `warn` or `invalid` view is diagnostic only and returns `signal: "not_ready"` with no decision score, model decision, or rank.

The production service reads only the server-registered snapshots at `data/live/features/swing.parquet` and `data/live/features/intraday.parquet`. Live snapshots require a matching sidecar manifest, SHA-256 integrity, explicit feed provenance, a fresh generation time, and a fresh latest feature timestamp. Models are loaded only through each route's signed local active-release pointer. Startup verifies the complete release and public-key attestation, then deserializes one cached context per route. Requests never deserialize joblib artifacts. The request schema rejects filesystem paths, data-source overrides, and promotion overrides.

Production routes are declared under `[prediction_serving.routes]` with a
`release_repository`, `bar_timeframe`, and conservative
`estimated_resident_gib`; `[prediction_serving]` also names the read-only public
attestation trust store. Direct model paths are rejected.

The API preloads active contexts during lifespan startup. Readiness does not
deserialize models or load full feature frames. Inference has one process-wide,
non-queueing lease, a bounded ticker batch, an incremental-memory reservation,
and current/projected RSS guards under the 4 GiB budget. Capacity and memory
pressure return typed retryable HTTP 503 responses. Real-size soak evidence is
still required before deployment.

Build a label-free canonical inference artifact, then publish it atomically to the registered live path:

```powershell
market-predictor build-intraday-live-features `
  --decisions data/canonical/intraday_decisions_5m.parquet `
  --one-minute-bars data/canonical/intraday_bars_1m.parquet `
  --benchmark-bars data/canonical/intraday_benchmarks_5m.parquet `
  --global-events data/canonical/global_events.parquet `
  --global-source-collections data/canonical/global_source_collections.parquet `
  --config configs/intraday_dataset.toml `
  --out data/live/staging/intraday_60m.parquet

market-predictor publish-live-features `
  --mode intraday `
  --input-path data/live/staging/intraday_60m.parquet `
  --live-dir data/live
```

`publish-live-features` accepts only the matching `swing_inference_features` or `intraday_inference_features` canonical artifact. Arbitrary Parquet, label-bearing training rows, caller-supplied feed overrides, mixed decision timestamps, stale rows, and future feature availability are rejected.

Catalyst evidence is returned separately from model probabilities. Swing exposes its unmodified probability. Canonical intraday exposes independent `opportunity_probability` and `downside_probability`; its `decision_score` applies a transparent ranking adjustment to their risk-adjusted combination. The `catalyst` object reports confirmation, conflict, veto, mixed, or absent evidence using relevance, sentiment, recency, source diversity, generic-headline rate, and material-event taxonomy. Catalyst can confirm or veto a decision, but it does not modify either intraday estimator probability.

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
market-predictor score-swing-events --raw-dir data/raw/swing --out-dir data/raw/swing_scored
market-predictor build-swing-datasets --horizon-days 1 --raw-dir data/raw/swing --out-dir data/features/swing
```

Use a custom list:

```powershell
market-predictor collect-swing --tickers "POET,MXL,RDW,LASE,RGTI,MRVL,IONQ,QBTS" --days 180
```

Stage separation:

- `collect-swing`: API/data download only by default. Alpaca, Finviz, Reddit, Seeking Alpha, and SEC failures are isolated per source and per ticker. Uses parallel workers for I/O. Seeking Alpha is enabled by default when RapidAPI credentials are configured; use `--no-seeking-alpha` only for an explicit quota outage or diagnostic run.
- `score-swing-events`: FinBERT scoring only. Loads the model once, uses GPU if PyTorch detects CUDA, records inference provenance, then writes per-ticker files.
- `build-swing-datasets`: research-only daily/hourly joins, event reaction features, technical features, and labels. It never injects current Seeking Alpha or SEC snapshots into historical rows. Production training uses the canonical point-in-time path below.

Performance knobs live in `configs/default.toml`:

```toml
[performance]
max_workers = 6
finbert_batch_size = 32
```

Override workers per command:

```powershell
market-predictor collect-swing --days 180 --workers 8
market-predictor score-swing-events --batch-size 64
market-predictor build-swing-datasets --horizon-days 1 --workers 8
```

The engine separates event timing buckets:

- `pre_market`: news before 9:30 ET, with open gap and day return features.
- `intraday`: news during regular hours, with first-2-hour hourly-candle reaction and to-close reaction.
- `after_hours`: news after 16:00 ET, rolled to the next feature date with next-open gap and next-day return features.

Hourly reaction features require Alpaca bars. If hourly bars are unavailable, the daily gap/day-return features still build.

## Removed Legacy Paths

The former four-model watchlist average, heuristic watch/behavior commands, event/daily baseline trainers, `live-once`, `live-run`, and accuracy-only `live-train-event` path were removed. They mixed incompatible horizons, accepted unregistered artifacts, and could republish research features as live data. The API is the only operational prediction surface; research scorers require a registered, hash-matching candidate or promoted model.

The canonical point-in-time data boundary, C4 swing pipeline, C5 intraday pipeline, and C6 serving/deployment infrastructure are implemented. Scheduling remains an external Azure Container Apps responsibility. `/v1/health/ready` remains HTTP 503 while a promoted canonical model or audited live snapshot is absent.

## Canonical Point-In-Time Data

Production model inputs are immutable, hash-verified Parquet artifacts. The normal path is:

```powershell
market-predictor canonicalize-bars --input-path data/raw/bars.parquet --out data/canonical/bars.parquet --timeframe 5m --price-feed sip
market-predictor canonicalize-event-directory --input-dir data/raw/swing_scored --out data/canonical/events.parquet
market-predictor canonicalize-source-collections --input-path data/raw/swing/_source_collections.parquet --out data/canonical/source_collections.parquet
market-predictor canonicalize-memberships --input-path data/raw/universe_memberships.parquet --out data/canonical/memberships.parquet
market-predictor build-canonical-decisions --bars data/canonical/bars.parquet --events data/canonical/events.parquet --source-collections data/canonical/source_collections.parquet --memberships data/canonical/memberships.parquet --out data/canonical/decisions.parquet
```

Canonical guarantees:

- Alpaca bar timestamps are treated as interval starts. A five-minute bar is unavailable until its interval ends plus the configured finalization delay; daily bars use the actual XNYS close, including early closes.
- Event features use `feature_available_at_utc`, which includes provider updates, first observation, and FinBERT scoring time.
- Every required source is `observed` or `observed_empty` by each production decision, and its request coverage end must be within the configured freshness limit. A past successful pull cannot be carried forward indefinitely. `failed`, `partial`, `disabled`, `not_collected`, and stale coverage fail readiness.
- Universe sector, industry, market-cap bucket, liquidity bucket, and benchmark are joined only from an effective membership snapshot already available at the decision time.
- SEC/quant facts are joined by versioned availability. A current snapshot is never copied backward over historical rows.
- SIP is mandatory for production volume features. IEX and unknown feed provenance fail the production bar audit.

A historical news pull performed today does not recreate historical first-seen time. Publication-time proxy backfills must be marked `--research`; they cannot be loaded by a production decision build. This permits controlled research while preventing the same artifact from being presented as live-valid evidence.

Market Predictor has no runtime alert commands or alert persistence. Alert evaluation, deduplication, acknowledgement, and web/mobile delivery belong to `trading_flow`. The removed rule behavior is preserved in [Legacy alert rule parity](docs/legacy_alert_rule_parity.md). Do not build new alert behavior in this repository.

## Canonical Swing Model Pipeline

The production swing path consumes only hash-verified canonical artifacts. SPY, QQQ, and every sector ETF used by a membership row must be present in `benchmark_bars`.

```powershell
market-predictor build-swing-dataset `
  --decisions data/canonical/decisions.parquet `
  --benchmark-bars data/canonical/benchmark_daily_bars.parquet `
  --global-events data/canonical/global_events.parquet `
  --global-source-collections data/canonical/global_source_collections.parquet `
  --config configs/swing_dataset.toml `
  --out data/features/swing/swing_5d.parquet

market-predictor train-swing-model `
  --dataset data/features/swing/swing_5d.parquet `
  --config configs/swing_training.toml `
  --model-out models/swing/candidates/swing_5d.joblib `
  --evidence-dir data/reports/swing_5d_candidate

market-predictor promote-swing-model `
  --model models/swing/candidates/swing_5d.joblib `
  --evidence-dir data/reports/swing_5d_candidate `
  --hypothesis-registry data/governance `
  --hypothesis-id swing-5d-h001 `
  --shadow-bundle data/governance/shadow/<shadow-fingerprint>.json `
  --build-identity ci:<build-id> `
  --approver-identity reviewer:<identity> `
  --signing-private-key <secure-ed25519-private-key.pem> `
  --attestation-trust-store configs/attestation_trust_store.json `
  --signer-id promotion-ci-prod `
  --config configs/swing_promotion.toml
```

`build-swing-dataset` uses a post-close decision, next-session-open entry, and fifth-session-close exit. It writes exact entry/exit/label timestamps, costs, stock and benchmark returns, MFE/MAE, and eligibility evidence. `train-swing-model` publishes an immutable candidate plus a hash inventory for every promotion file. `promote-swing-model` verifies that inventory, applies frozen development gates, consumes one predeclared untouched-shadow bundle, requires a positive paired session-block confidence lower bound, and writes an immutable attestation. Editing a model, manifest, metric, audit, shadow bundle, or attestation invalidates authorization.

The removed `build-volatile-dataset`, `train-volatile-model`, and `score-volatile-latest` commands are not compatibility aliases. Old volatile artifacts cannot be loaded by the production swing API.

## Canonical Intraday Model Pipeline

The production intraday path consumes hash-verified canonical 5-minute decisions, 1-minute stock/benchmark bars, 5-minute benchmark bars, and global context. Dataset construction and training use column projection, `float32` matrices, sequential fold-model release, and fail before the configured 4 GiB process limit.

```powershell
market-predictor build-intraday-dataset `
  --decisions data/canonical/intraday_decisions_5m.parquet `
  --one-minute-bars data/canonical/intraday_bars_1m.parquet `
  --benchmark-bars data/canonical/intraday_benchmarks_5m.parquet `
  --global-events data/canonical/global_events.parquet `
  --global-source-collections data/canonical/global_source_collections.parquet `
  --config configs/intraday_dataset.toml `
  --out data/features/intraday/intraday_60m.parquet

market-predictor train-intraday-model `
  --dataset data/features/intraday/intraday_60m.parquet `
  --config configs/intraday_training.toml `
  --model-out models/intraday/candidates/intraday_60m.joblib `
  --evidence-dir data/reports/intraday_60m_candidate

market-predictor promote-intraday-model `
  --model models/intraday/candidates/intraday_60m.joblib `
  --evidence-dir data/reports/intraday_60m_candidate `
  --hypothesis-registry data/governance `
  --hypothesis-id intraday-60m-h001 `
  --shadow-bundle data/governance/shadow/<shadow-fingerprint>.json `
  --build-identity ci:<build-id> `
  --approver-identity reviewer:<identity> `
  --signing-private-key <secure-ed25519-private-key.pem> `
  --attestation-trust-store configs/attestation_trust_store.json `
  --signer-id promotion-ci-prod `
  --config configs/intraday_promotion.toml
```

Each decision is made only after a completed 5-minute bar. Entry is the first subsequent 1-minute open. The default 60-minute path uses exact consecutive 1-minute bars, a 1 ATR target, a 0.75 ATR stop, and stop-first resolution when both barriers occur in one bar. SPY, QQQ, and sector returns use the same actual entry/exit interval. Missing ticker or benchmark bars invalidate the row; they are never shifted or filled.

The candidate contains two estimators and is promoted atomically: opportunity estimates target-before-stop, while downside estimates stop-before-target. Catalyst/news features are audited and returned as a confirmation/ranking overlay, but are deliberately excluded from both estimators until fresh ablation evidence proves incremental value. No real C5 candidate has been promoted, so no canonical intraday route belongs in `configs/default.toml` yet.

## Trusted Local Releases

Only an attested model can enter the local release repository. Publication copies the model, immutable candidate manifest, promotion attestation, evidence manifest, and every evidence file into a versioned content-addressed directory. Every hash and the attestation are reverified before one locked active pointer is replaced.

```powershell
market-predictor publish-local-release `
  --model models/swing/candidates/swing_5d.joblib `
  --evidence-manifest data/reports/swing_5d_candidate/evidence.manifest.json `
  --release-root data/local_release_repository `
  --attestation-trust-store configs/attestation_trust_store.json

market-predictor show-active-local-release `
  --release-root data/local_release_repository `
  --attestation-trust-store configs/attestation_trust_store.json

market-predictor rollback-local-release `
  --release-id <64-character-release-id> `
  --release-root data/local_release_repository `
  --attestation-trust-store configs/attestation_trust_store.json
```

A partial, mutated, candidate-only, unsigned, untrusted, or unattested release cannot become active. The public trust store is server-owned and should be mounted read-only; the signing private key belongs only to the promotion workload and must not be stored in this repository. Set `MARKET_PREDICTOR_ATTESTATION_TRUST_STORE` for serving-time verification. The local pointer is the current R4 activation authority. Azure activation remains `environment_pending`.

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

Automated retraining remains disabled until the canonical dataset builder, shadow evaluator, and promotion workflow are connected end to end. Model retraining is never triggered from prediction traffic.

## Azure

Azure is project-specific. This project stores its own artifacts under `AZURE_BLOB_PREFIX`, defaulting to:

```text
market-predictor
```

Recommended deployment is:

```text
Azure Blob Storage + Azure Container Apps Jobs + Azure ML GPU compute on demand
```

Build context files:

```text
Dockerfile
.dockerignore
scripts/container-entrypoint.sh
```

The image runs the API as UID/GID 10001, exposes port 8000, and probes
`/v1/health/live`. Configure a 4 GiB container memory limit in addition to the
in-process guard. Azure model-release synchronization is not currently an
activation path.

The deployment rationale, identity/secret requirements, release ordering, and job schedule are in `docs/azure_deployment_plan.md`.

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
