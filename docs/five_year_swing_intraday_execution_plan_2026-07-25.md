# Five-Year Swing And Intraday Execution Plan

Date: 2026-07-25

## Objective

Build independently evaluated swing and intraday predictors using the longest
causal, auditable data already available. Download only evidence that is absent
locally. Keep model training sequential and peak process memory below 4 GiB.

Finviz Elite discovers and ranks the current inference workload. It is not a
historical training universe. Present-day Finviz score, change, volume, and
theme fields are prohibited estimator features unless archived point-in-time
snapshots later exist.

## Audited Reuse Matrix

| Evidence | Local state | Decision |
|---|---|---|
| Swing daily SIP bars | 2019-07-09 to 2026-07-08; 885,371 PIT stock rows; 656 tickers; 13 complete benchmarks | Reuse. No price download for the five-year swing baseline. |
| Intraday S&P 5-minute SIP bars | 2024-07-09 to 2026-07-08; 25,236,149 rows; 546 current symbols; 501 sessions | Reuse for the next two-year baseline. Do not rebuild. |
| Intraday benchmark 5-minute bars | 13 SPY/QQQ/sector ETFs; 768,307 rows; 501 sessions | Reuse. |
| Intraday 1-minute bars | 200 NASDAQ snapshot symbols; 41 sessions; 4,158,863 rows; feed not declared | Research-only confirmation evidence. Not promotion-grade volume evidence. |
| Canonical V4 120-minute panel | 505,049 rows; 474 sessions; 545 eligible tickers; peak training memory 1.95 GiB | Reuse as regression reference; rebuild only after new source identities are frozen. |
| Local ticker news | 510/656 PIT tickers have any event; earliest 2024-06-14; maximum 24.76 months; no historical first-observed timestamps | Reuse only for research comparison and parser regression. It does not meet five-year or promotion requirements. |
| Reddit | No verified events | Missing. Do not interpret existing zero columns as no chatter. |
| Global context | Two years of ETF-proxy events plus 20 recent GDELT events | Research-only and incomplete. Do not call ETF proxy rows direct global news. |
| Finviz candidates | July 2026 current snapshots; incomplete exchange/security/freshness identity | Parser fixtures only. Refresh before current prediction use. |

## Provider Facts

- Alpaca historical news is available back to 2015 and is supplied by Benzinga:
  <https://docs.alpaca.markets/us/docs/historical-news-data>.
- Alpaca news requests support symbol, start/end, and pagination:
  <https://docs.alpaca.markets/us/v1.1/reference/news-3>.
- Finviz Elite supports real-time screener data and export/API access:
  <https://finviz.com/elite>.
- Finviz defines relative volume as current volume divided by three-month
  average volume and exposes ATR, volatility, SMA, price, and volume fields:
  <https://finviz.com/help/screener>.

## Phase A: Current Candidate Profiles

Implement two immutable, inference-only profiles.

### `finviz.swing.current.v1`

- US-listed common stocks/ADRs on Nasdaq, NYSE, or NYSE American.
- Exclude ETFs, OTC securities, and unsupported instruments.
- Price at least $5.
- Average daily volume at least 500,000.
- Average daily dollar volume at least $20 million.
- Market capitalization at least $300 million.
- Lanes: trend/breakout, constructive pullback, and catalyst/high-relative-volume.
- Rank with Alpaca SIP-confirmed liquidity, 20/60-day relative strength versus
  SPY and sector ETF, SMA state, ATR%, relative volume, and catalyst freshness.

### `finviz.intraday.current.v1`

- Same listing/security restrictions.
- Price at least $2.
- Average volume at least 1 million.
- Average daily dollar volume at least $20 million.
- Relative volume at least 1.5.
- Require meaningful gap/change or elevated ATR/volatility.
- Alpaca SIP bars/quotes validate current volume, spread, and tradability.
- Freshness: 15 minutes premarket and 5 minutes during regular trading.

Every output binds profile/run IDs, observation and expiry timestamps, request,
response, and profile hashes, exchange/security identity, matched screen lanes,
rank inputs, final rank, and `usage_scope=current_inference_only`.
`training_eligible` is always false.

Current blocker: `FINVIZ_ELITE_AUTH` is not present in the process environment.
The token must remain an environment secret; it must not be placed in code,
command arguments, logs, or artifacts. Existing stale exports can test parsing
but cannot produce a current candidate result.

## Phase B: Five-Year Swing Baselines

Status on 2026-07-25: the technical dataset and both E0 baselines completed.
The dataset contains 630,978 rows, 607,909 exact-path eligible rows, 581
training tickers, and 99.9941% exact outcome-path coverage. Peak dataset-build
RSS was 3.115 GiB. Logistic and histogram-gradient walk-forward AUC were 0.4962
and 0.5000 respectively, with negative conservative return and SPY excess.
Both are rejected. Bars alone did not establish edge; proceed to causal
catalyst evidence without promoting either baseline.

1. Slice the existing canonical PIT daily panel to the frozen five-year window.
2. Build research decisions and exact five-session next-open labels using
   `security_id`, SPY, QQQ, and sector ETF paths.
3. Run technical/relative/regime E0 baselines first: logistic regression and
   histogram gradient boosting.
4. Use the existing two-year publication-proxy news only for a controlled
   research ablation. Missing pre-coverage news remains missing, never zero.
5. Collect five-year Alpaca/Benzinga news for full-window catalyst research.
6. Evaluate catalyst-only, direct catalyst, and catalyst-overlay variants on
   identical folds.

No swing price download is required.

## Phase C: Five-Year Historical News

A new immutable Alpaca collector is required because local event files cover at
most about two years, omit 146 PIT tickers, have no source manifests or
`security_id`, and lack historical first-observed evidence.

Collection rules:

- One ticker/source attempt is isolated and resumable.
- Freeze 2021-07-09 through 2026-07-08.
- Preserve provider `created_at` as publication and `updated_at` separately.
- Record collection/ingestion time honestly.
- Historical backfill uses `provider_publication_proxy` and is research-only.
- Map each event to the effective `security_id`; reject ticker-reuse ambiguity.
- Deduplicate by provider ID and content identity.
- Publish per-ticker artifacts plus a source-attempt ledger and final manifest.
- Collect raw events first; run FinBERT separately and sequentially.

Seeking Alpha, SEC, Reddit, and global events remain separate source families.
The 10,000-call Seeking Alpha allowance is reserved for current enrichment and
targeted ablation rather than duplicating the five-year Alpaca news corpus.

## Phase D: Intraday Corpus

### Immediate baseline without downloads

- Reuse the two-year S&P SIP 5-minute corpus and 13 benchmarks.
- Use the canonical 120-minute decision design.
- Use 1-minute data only where feed provenance and exact consecutive label paths
  are available; otherwise the row is ineligible.
- Re-run the opportunity/downside model sequentially against the frozen V4
  reference. Current Finviz fields stay outside both estimators.

### Five-year extension

- Backfill only the missing older three years of 5-minute SIP bars for PIT
  members and the 13 benchmarks.
- Do not download full five-year 1-minute history initially.
- Derive candidate sessions causally from information available before each
  decision, then fetch 1-minute bars only for exact entry/target/stop windows.
- Partition canonical bars/features by month and ticker.
- Project columns and use `float32` matrices.
- Train one model family and one walk-forward fold at a time.
- Use single-worker tree training and release each fold model before the next.

Estimated five-year scale is about 63 million 5-minute rows and 1.4-1.8 GiB
compressed. Full 1-minute history for roughly 500 names would approach 319
million rows and about 7 GiB compressed before features, so it is not the first
collection target.

## Sequential Model Order

1. Swing logistic technical baseline.
2. Swing histogram-gradient-boosted technical baseline.
3. Swing catalyst ablations after five-year news collection.
4. Intraday 120-minute opportunity model on existing two-year SIP data.
5. Intraday downside model on the identical folds.
6. Intraday five-year rerun after 5-minute backfill and selective 1-minute
   evidence are complete.

Only one heavy build or training process runs at a time. Each stage records peak
working set and fails before 4 GiB.

## Execution Concurrency

Read-only source, schema, leakage, and evidence audits may run concurrently in
independent agent processes. They must not load full training matrices or write
canonical artifacts.

Dataset assembly, sentiment inference, model training, backtesting, promotion
audits, and any command that materially loads a market panel are heavy jobs.
Heavy jobs run sequentially under one non-queueing workspace lease. A second
heavy command fails before loading data; it does not wait or compete for memory.
The lease records command, process, host, start time, and configuration identity,
and stale ownership is recovered only after proving the recorded process no
longer exists. Process RSS guards remain active inside every leased command.

## Exit Gates

- Point-in-time universe and security identity pass.
- No current Finviz snapshot appears in historical model features.
- Every feature availability is at or before its decision.
- News publication/update/availability and candle session assignment reconcile.
- Exact stock and benchmark label paths reproduce from immutable bars.
- Walk-forward folds are purged, embargoed, and causally ordered.
- Ticker holdout remains separate from temporal validation.
- Cost-adjusted top-k economics, calibration, drawdown, capacity, and regime
  evidence are reported.
- Candidate artifacts remain unpromoted until every existing promotion gate and
  prospective shadow requirement passes.
