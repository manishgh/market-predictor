# Production ML and Platform Architecture Review - 2026-07-22

Review base: `48ac758`  
Scope: production architecture, data reliability, serving correctness, deployment, and the TradingFlow boundary  
Method: static trace of executable paths, persisted-artifact inventory, and 81 bounded tests. No external APIs, full training, large backtests, model promotion, or credential inspection were performed.

## P0 Findings

### P0-1 - Every canonical live swing prediction is suppressed by an incorrect warm-up count

- **Location:** `src/market_predictor/live_features.py:62`, `src/market_predictor/live_features.py:67`, `src/market_predictor/live_features.py:113`, `src/market_predictor/live_features.py:178`, `src/market_predictor/prediction_service.py:394`, `src/market_predictor/prediction_service.py:406`, `src/market_predictor/readiness.py:99`
- **Defect or risk:** The live builder deliberately publishes one latest decision row per ticker and preserves `daily_bar_count`. Swing serving ignores that field and recomputes history depth by counting unique dates in the one-row live snapshot.
- **Impact:** A correctly warmed ticker with 250 or more daily bars is served with `daily_bar_count == 1`, readiness becomes `invalid`, and signal, decision score, prediction, and rank are suppressed. The configured swing production route cannot emit an actionable prediction.
- **Evidence:** A direct reproduction published a one-row live snapshot whose row contained `daily_bar_count=260`; serving returned `served_daily_bar_count=1`, `readiness=invalid`, and `signal=not_ready`. `tests/test_prediction_service.py:257` checks only probability for the live path and does not assert readiness or signal.
- **Remediation:** Make swing serving mirror the intraday implementation at `src/market_predictor/prediction_service.py:476`: use the row's audited `daily_bar_count`, with snapshot row count only as an explicitly research-only fallback. Require `daily_bar_count` in `LiveFeatureStore.publish` for swing.
- **Verification:** Add a production-path test that builds canonical swing inference features, publishes them, scores through `PredictionService(data_source="live")`, and proves a row with `daily_bar_count >= 250` remains `valid`. Add the inverse test for an under-warm row.

### P0-2 - The documented nightly collector cannot satisfy the swing decision-time source contract

- **Location:** `src/market_predictor/canonical/joins.py:27`, `src/market_predictor/canonical/joins.py:43`, `src/market_predictor/canonical/joins.py:207`, `src/market_predictor/canonical/joins.py:212`, `src/market_predictor/commands/canonical_data.py:268`, `src/market_predictor/commands/canonical_data.py:271`, `docs/azure_deployment_plan.md:83`
- **Defect or risk:** `decisions_from_completed_bars` fixes `decision_time_utc` to the bar's availability time. Source status is then joined only from collection attempts whose `completed_at_utc <= decision_time_utc`. The documented job collects at 02:00 UTC, hours after the US daily bar decision around 20:15/21:15 UTC, so that collection is necessarily invisible to the decision it is intended to enrich.
- **Impact:** Nightly live swing builds either fail required-source freshness or omit the newest post-close catalysts, including after-hours earnings. The architecture says the decision occurs after all required feature/source timestamps, but the executable path does not create such a decision cutoff.
- **Evidence:** A bounded reproduction joined a 02:01 UTC completed collection to the prior 20:15 UTC daily decision and returned `source_status_alpaca=not_collected`. The default production decision build also applies a 60-minute source-coverage age.
- **Remediation:** Introduce an explicit, calendar-aware `prediction_as_of_utc` or decision-cutoff artifact for each production run. Keep each feature's own availability timestamp, require all to be at or before that cutoff, and construct the next-session-open label from the cutoff. Do not move historical cutoffs after seeing outcomes. Align the Azure schedule and training decision schedule to the same frozen rule.
- **Verification:** Add normal-close, early-close, after-hours catalyst, and DST tests proving a source collected after the bar but before the declared nightly cutoff joins, while a source completed after the cutoff does not. Run the complete nightly command sequence from raw collection through live publication on a frozen fixture.

## P1 Findings

### P1-1 - Explicit `60m` intraday requests are normalized away from the configured route

- **Location:** `src/market_predictor/prediction_contracts.py:11`, `src/market_predictor/prediction_contracts.py:49`, `src/market_predictor/prediction_service.py:47`, `src/market_predictor/prediction_service.py:636`
- **Defect or risk:** The request validator maps `60m` to `1h`, while the canonical intraday route and default horizon are named `60m`.
- **Impact:** A TradingFlow client that explicitly requests the documented canonical horizon receives an unsupported-horizon failure. `auto` works, masking the contract defect.
- **Evidence:** Direct construction produced `requested=60m`, `normalized=1h`, while a parsed route retained `configured_horizons=['60m']`.
- **Remediation:** Pick one canonical wire value and use it unchanged across request validation, route keys, target parsing, response, and replay. Prefer `60m` because it describes the model horizon independently of bar timeframe.
- **Verification:** Contract-test `60m`, `1h`, and `auto` against an intraday `60m` route and assert the same documented resolution or an intentional rejection.

### P1-2 - Served evidence lacks the identity needed for TradingFlow validation and replay uses the wrong decision time

- **Location:** `src/market_predictor/prediction_contracts.py:153`, `src/market_predictor/prediction_contracts.py:166`, `src/market_predictor/prediction_service.py:734`, `src/market_predictor/prediction_snapshot.py:25`, `src/market_predictor/investment_replay.py:73`, `src/market_predictor/investment_replay.py:74`
- **Defect or risk:** The API has no independent contract version, correlation ID, actual prediction cutoff, per-row feature availability, feature artifact hash, release ID, or source watermarks. When request `as_of` is absent, replay substitutes response generation time for the feature decision time.
- **Impact:** TradingFlow cannot prove freshness or pin the exact evidence used by a strategy. Replay can enter later than the model decision, especially for a stale-but-allowed swing snapshot, invalidating self-evaluation and benchmark comparison.
- **Evidence:** The integration specification lists these as required/planned, but `PredictionResponse` exposes only model identity plus a local snapshot hash. `InvestmentReplayService` derives decision time from `request.as_of or response.generated_at_utc`.
- **Remediation:** Publish `PredictionEvidenceV1` as a separate stable wire contract containing contract version, caller correlation ID, prediction cutoff, feature availability, live feature hash, active release ID, model hash, source watermarks, and resolved horizon. Replay must use the persisted prediction cutoff, never generation time.
- **Verification:** Round-trip contract tests from Python JSON to the TradingFlow DTO; replay tests where generation time differs from feature cutoff; rejection tests for missing, future, stale, or mismatched evidence identity.

### P1-3 - Prediction snapshots and outcomes are local ephemeral files, not a durable audit ledger

- **Location:** `src/market_predictor/prediction_service.py:136`, `src/market_predictor/prediction_snapshot.py:40`, `src/market_predictor/prediction_snapshot.py:47`, `Dockerfile:26`
- **Defect or risk:** Snapshots are written under local `data/predictions/snapshots`; there is no Blob-backed implementation, write-through replication, retention policy, index, or startup recovery. The container creates local directories but does not mount durable storage.
- **Impact:** Container restart, revision replacement, or scale-out loses or fragments prediction evidence. Replay by snapshot ID becomes instance-dependent and audit continuity is broken.
- **Evidence:** No canonical promoted/live artifacts are currently present, and no persisted active release marker or prediction-store configuration exists. The Azure document names a `predictions/` path but executable code never writes snapshots there.
- **Remediation:** Define a `PredictionSnapshotRepository` with immutable Blob implementation and local bounded cache. Persist snapshot and outcome envelopes with conditional create, retention, partition/index metadata, and release/feature/model identities.
- **Verification:** Multi-replica create/load test, restart recovery test, duplicate-ID idempotency test, tamper test, retention test, and Azure failure-mode test proving local cache cannot acknowledge durability before Blob commit.

### P1-4 - A mutable manifest field can bypass all promotion gates

- **Location:** `src/market_predictor/registry.py:47`, `src/market_predictor/registry.py:57`, `src/market_predictor/registry.py:81`, `src/market_predictor/registry.py:95`, `src/market_predictor/swing/promotion.py:201`, `src/market_predictor/deployment.py:114`, `tests/test_deployment.py:242`
- **Defect or risk:** `write_model_manifest` accepts `status="promoted"` directly. Serving verifies only mutable manifest status and model artifact hash. A release ships model plus manifest, but not a hash-bound promotion report/evidence attestation. The deployment test itself creates a promoted model without running promotion gates.
- **Impact:** Any process with write access to the model directory can convert an arbitrary compatible joblib artifact into a serveable model without satisfying economics, alignment, calibration, or provenance gates. Artifact SHA-256 provides integrity, not authorization or authenticity.
- **Evidence:** Canonical promotion verifies a strong evidence bundle before mutating the same JSON status, but the resulting manifest records no evidence-manifest digest and release publication does not require one.
- **Remediation:** Make candidate creation incapable of setting promoted status. Store an immutable promotion attestation containing candidate hash, evidence-manifest hash, gate-config hash, approver/build identity, and timestamp in an append-only registry. Release publication must require and include that attestation. Use workload identity and separate write roles for training, promotion, and release.
- **Verification:** Prove direct manifest editing or direct `write_model_manifest(status="promoted")` cannot produce a serveable release; prove evidence tampering, threshold changes, or missing attestation fail publication.

### P1-5 - Serving is not pinned to one atomic local release and contains verification/load races

- **Location:** `src/market_predictor/feature_store.py:49`, `src/market_predictor/feature_store.py:51`, `src/market_predictor/swing/model.py:302`, `src/market_predictor/swing/model.py:305`, `src/market_predictor/intraday/model.py:361`, `src/market_predictor/intraday/model.py:364`, `src/market_predictor/deployment.py:233`, `src/market_predictor/deployment.py:252`
- **Defect or risk:** Feature files are hashed during `validate` and reopened later for Parquet read. Models are hashed, then reopened by `joblib.load`. Release sync replaces every artifact individually and health never verifies that all loaded assets belong to one active-release marker.
- **Impact:** Concurrent publication/sync can score a file different from the one that was verified, mix model and feature revisions, or deserialize an unverified replacement. More commonly it causes transient readiness failures during an otherwise valid deployment.
- **Evidence:** Manifests-last limits partial-state acceptance but does not pin open file handles or make the multi-asset local install atomic.
- **Remediation:** Install each release into a versioned immutable directory and atomically switch one local release pointer. Construct a release-scoped serving context, verify all assets once, load from immutable paths, and retain the prior context until in-flight requests finish.
- **Verification:** Concurrent scoring and release-switch tests with barriers at every verify/open/replace boundary; every response must contain one release ID and either succeed coherently or fail before scoring.

### P1-6 - The 4 GiB runtime limit is observed by health but not enforced on inference

- **Location:** `src/market_predictor/prediction_service.py:333`, `src/market_predictor/prediction_service.py:353`, `src/market_predictor/swing/model.py:295`, `src/market_predictor/swing/model.py:305`, `src/market_predictor/intraday/model.py:347`, `src/market_predictor/intraday/model.py:364`, `src/market_predictor/resources.py:28`
- **Defect or risk:** Every request deserializes its model from joblib and reads the feature snapshot again. There is no startup model cache, request semaphore, bounded queue, pre/post inference memory assertion, or worker-count policy. The runtime memory check is only reported by readiness after allocation pressure already exists.
- **Impact:** Concurrent FastAPI thread-pool requests can hold duplicate swing/intraday estimators and matrices, exceed 4 GiB, thrash, or be OOM-killed. A not-ready probe does not stop already admitted work.
- **Evidence:** Training calls `assert_memory_budget`; inference does not. The repository has no load/concurrency memory test.
- **Remediation:** Load and verify one immutable model set at startup, reuse it read-only, cap concurrent inference, bound ticker batch size, reject admission near the safety threshold, and set one API process per 4 GiB container unless measured evidence permits more.
- **Verification:** Soak and burst tests for swing, intraday, and unified requests with real-size artifacts; peak container RSS must remain below the declared threshold and overload must return a typed retryable response rather than OOM.

### P1-7 - The API has no implemented service authentication or abuse controls

- **Location:** `src/market_predictor/api.py:83`, `src/market_predictor/api.py:99`, `src/market_predictor/api.py:103`, `src/market_predictor/api.py:127`, `src/market_predictor/api.py:45`, `src/market_predictor/investment_replay.py:119`
- **Defect or risk:** All prediction, metrics, and replay routes are unauthenticated and have no authorization, rate limit, request-size limit, or caller identity. Replay makes Alpaca API calls from a request.
- **Impact:** Network misconfiguration exposes model metadata and operational metrics, permits resource exhaustion, and can consume paid market-data quota. TradingFlow calls cannot be attributed or revoked.
- **Evidence:** The Azure document requires private ingress and managed identity/service authentication, but there is no middleware or gateway contract enforcing either.
- **Remediation:** Require private ingress plus Entra workload identity or a validated service token, authorize metrics separately, rate-limit by caller, cap ticker batch size, and disable or separately authorize replay in production.
- **Verification:** Integration tests for missing/invalid/expired identity, TradingFlow identity, metrics role, replay role, rate limiting, and oversized requests; Azure rehearsal must prove public ingress is disabled.

### P1-8 - Parallel collection can corrupt or undercount Seeking Alpha quota/cache state

- **Location:** `src/market_predictor/cli.py:845`, `src/market_predictor/cli.py:894`, `src/market_predictor/quota.py:46`, `src/market_predictor/quota.py:47`, `src/market_predictor/quota.py:59`, `src/market_predictor/quota.py:70`, `src/market_predictor/sources/seeking_alpha.py:226`, `src/market_predictor/sources/seeking_alpha.py:260`
- **Defect or risk:** `collect-swing` uses multiple threads; each ticker creates a tracker against the same quota JSON. Quota update is an unlocked read-modify-write and both quota/cache writes are direct, non-atomic writes.
- **Impact:** Calls are lost from accounting, monthly limits can be exceeded, readers can observe partial JSON, and simultaneous identical requests can duplicate paid calls or corrupt cache files.
- **Evidence:** Default collection concurrency is six. There are only single-threaded quota tests and no concurrent cache tests.
- **Remediation:** Centralize provider scheduling and quota reservation, use an interprocess lock or transactional store, reserve before call and reconcile after response, and publish cache entries via temporary file plus atomic replace. Add single-flight by cache key.
- **Verification:** Thread/process race tests with hundreds of reservations must produce the exact call count, valid JSON at every read, one network call per cache key, and deterministic recovery after a killed writer.

### P1-9 - Reddit search results are accepted as ticker events without post-level relevance validation

- **Location:** `src/market_predictor/sources/reddit.py:47`, `src/market_predictor/sources/reddit.py:72`, `src/market_predictor/sources/reddit.py:105`, `src/market_predictor/sources/reddit.py:158`, `configs/default.toml:18`
- **Defect or risk:** Every post returned by Reddit search is assigned to the requested ticker. Explicit mention matching and the false-positive stoplist are applied only to comments.
- **Impact:** Ambiguous symbols such as `AI`, `ON`, `IT`, and other common words can inject unrelated chatter into stock catalyst features. This recreates the ticker-news relevance failure the canonical design is intended to prevent.
- **Evidence:** There is no Reddit source test. Seeking Alpha implements tag-first relevance filtering, but Reddit posts have no equivalent verification.
- **Remediation:** Require a verified cashtag or a symbol plus company/security alias in title/body; apply the stoplist to posts; record match method and relevance score; keep unmatched search results out of ticker features.
- **Verification:** Fixture corpus with true mentions, cashtags, company aliases, common-word false positives, cross-posts, and renamed symbols; require precision/recall thresholds and zero leakage of rejected posts into canonical features.

### P1-10 - Symbol identity has two incompatible canonical forms

- **Location:** `src/market_predictor/symbols.py:10`, `src/market_predictor/symbols.py:15`, `src/market_predictor/v3/contracts.py:17`, `src/market_predictor/prediction_contracts.py:42`
- **Defect or risk:** `canonical_symbol` converts class separators to hyphens, while canonical/V3 contracts accept and preserve dots or hyphens, and prediction requests only uppercase input. Most source adapters also uppercase directly instead of calling `provider_symbol`.
- **Impact:** `BRK-B`, `BRK.B`, and similar class symbols can miss feature rows, split history, or fail ticker relevance and provider calls. Model training and live inference may refer to the same security with different identities.
- **Evidence:** Symbol unit tests cover the isolated helper, not end-to-end canonicalization, requests, membership, events, bars, and serving.
- **Remediation:** Define one security master identity and enforce it in every contract and request. Convert only at provider boundaries; persist provider symbol and stable security ID/CUSIP mapping separately; handle rename intervals point in time.
- **Verification:** End-to-end class-share and rename tests across Alpaca bars/news, SEC, Seeking Alpha, memberships, canonical joins, feature rows, API request, and replay.

### P1-11 - The deployable environment is not reproducible because dependencies are unlocked

- **Location:** `pyproject.toml:7`, `pyproject.toml:22`, `pyproject.toml:23`, `Dockerfile:21`, `.github/workflows/ci.yml:19`
- **Defect or risk:** Runtime packages use broad lower bounds, Docker upgrades pip and resolves from the network at build time, and there is no hash-locked dependency set, SBOM, or vulnerability gate.
- **Impact:** The same commit can build different pandas, scikit-learn, FastAPI, Azure, Torch, or transitive versions, changing serialization compatibility and numerical/serving behavior. A rollback of code is not a rollback of the environment.
- **Evidence:** Dependency locking was explicitly deferred and no lock artifact exists.
- **Remediation:** Generate platform-appropriate hash-locked runtime/training sets, separate the lightweight serving image from training/NLP dependencies, pin the base image by digest, and produce an SBOM plus vulnerability report.
- **Verification:** Two clean builds must have identical lock and image digests; deserialize promoted fixture artifacts and pass API contract tests in the built image; fail CI on unreviewed lock drift or critical vulnerabilities.

## P2 Findings

### P2-1 - API failures use the wrong status class and expose internal exception text

- **Location:** `src/market_predictor/api.py:136`, `src/market_predictor/api.py:141`, `src/market_predictor/api.py:146`, `src/market_predictor/api.py:151`, `src/market_predictor/api.py:162`
- **Defect or risk:** Every prediction/replay exception becomes HTTP 422 with raw exception text.
- **Impact:** TradingFlow cannot distinguish bad input from stale features, unavailable models, provider failure, storage failure, or server defect; raw paths/details can leak.
- **Evidence:** No typed error envelope or exception taxonomy exists.
- **Remediation:** Map validation to 400/422, readiness/dependency failures to 503, missing snapshot to 404, conflict to 409, throttling to 429, and unexpected errors to opaque 500 with correlation ID.
- **Verification:** Contract tests for each error family and a test proving responses contain no filesystem path, provider body, or secret-like text.

### P2-2 - Raw collection publication is non-atomic and reusing an output directory can mix runs

- **Location:** `src/market_predictor/cli.py:879`, `src/market_predictor/cli.py:880`, `src/market_predictor/cli.py:909`, `src/market_predictor/commands/canonical_data.py:144`
- **Defect or risk:** Per-ticker Parquet and aggregate source files are overwritten directly; no run manifest, completion marker, run ID, or cleanup isolates one collection. Canonical event-directory import globs every matching file in the directory.
- **Impact:** Interrupted/rerun jobs can combine old tickers, new events, and mismatched source-collection state. A later canonical hash proves the mixed input bytes, not that they belong to one complete run.
- **Evidence:** Canonical outputs are atomic and hash-verified, but their raw input set has no completeness contract.
- **Remediation:** Write each collection into an immutable run directory, atomically publish a run manifest last, and require canonicalization to consume that manifest rather than globbing.
- **Verification:** Kill/restart and changed-universe tests must never expose a partial run or include stale ticker files.

### P2-3 - Azure release activation has no lease or compare-and-swap protection

- **Location:** `src/market_predictor/deployment.py:176`, `src/market_predictor/deployment.py:184`, `src/market_predictor/deployment.py:206`, `src/market_predictor/deployment.py:214`, `src/market_predictor/azure_store.py:46`
- **Defect or risk:** Publish and rollback read the current pointer and then overwrite it unconditionally. No Blob ETag condition, lease, deployment generation, or single-writer enforcement exists.
- **Impact:** Concurrent release/rollback jobs can lose updates and record an incorrect `previous_release_id`, weakening rollback audit and change control.
- **Evidence:** In-memory tests are single-threaded and do not model Azure conditional writes.
- **Remediation:** Use Blob leases or `If-Match` ETag writes, enforce a release-controller identity, and persist an append-only activation event.
- **Verification:** Competing publisher/rollback integration tests must yield one winner and one explicit conflict without pointer corruption.

### P2-4 - Telemetry is process-local and cannot support production SLO or drift history

- **Location:** `src/market_predictor/telemetry.py:16`, `src/market_predictor/telemetry.py:25`, `src/market_predictor/telemetry.py:90`, `src/market_predictor/drift.py:36`, `src/market_predictor/prediction_service.py:312`
- **Defect or risk:** Counters, last health, and drift summaries live only in memory and are exposed as JSON. There is no durable exporter, model/release labels on all metrics, alert rule definition, or drift time series.
- **Impact:** Restarts erase operational history, replicas cannot be aggregated reliably, and severe drift cannot be investigated against releases and prediction outcomes.
- **Evidence:** Documentation correctly says telemetry is not durable evidence, but no production metrics/tracing implementation replaces it.
- **Remediation:** Export OpenTelemetry/Prometheus metrics and structured traces with release/model/feature/correlation labels; persist drift and replay aggregates outside the API process.
- **Verification:** Multi-replica dashboard and restart tests, cardinality checks, and alert rehearsal for stale source, hash failure, memory headroom, latency, error rate, and severe drift.

### P2-5 - CI does not enforce the repository-wide quality state claimed by the checkpoint

- **Location:** `.github/workflows/ci.yml:24`, `.github/workflows/ci.yml:27`, `.github/workflows/ci.yml:30`, `.github/workflows/ci.yml:36`
- **Defect or risk:** CI runs Ruff only on V3/commands and mypy only on V3, despite local repository-wide Ruff and strict mypy being green. It does not build or smoke the Docker image.
- **Impact:** Serving, canonical, deployment, source, and API modules can regress lint/type quality without CI failure; container-only packaging or startup failures remain undiscovered.
- **Evidence:** The current local checkpoint is clean, but the enforcement scope is narrower than the codebase.
- **Remediation:** Run repository-wide Ruff/mypy, the full tests, Docker build/startup/readiness smoke, lock verification, and security scans in CI.
- **Verification:** Seeded violations in a serving module and a container startup failure must fail CI.

### P2-6 - Production and historical research commands remain mixed in one large operational CLI

- **Location:** `src/market_predictor/cli.py:82`, `src/market_predictor/cli.py:1406`, `src/market_predictor/cli.py:1471`, `src/market_predictor/cli.py:1582`, `src/market_predictor/cli.py:1689`, `src/market_predictor/cli.py:1791`
- **Defect or risk:** Canonical train/promote/serve commands share a 2,000-line CLI with generic legacy promotion, entry/exit research scorers, and historical experimental pipelines. Generic `promote-model` remains visible alongside canonical promotion commands.
- **Impact:** Operators can invoke a noncanonical path, pollute registry status, or misunderstand which reports are production evidence. The broad import graph also forces unnecessary dependencies into serving tooling.
- **Evidence:** Serving correctly rejects incompatible model types, but the command surface does not clearly isolate production from research.
- **Remediation:** Split `market-predictor-prod`, `market-predictor-research`, and collector entry points or explicit Typer subcommands with separate dependency extras. Remove the generic promotion path when canonical swing/intraday promotion is authoritative.
- **Verification:** Golden command inventory test and container package test proving the production image exposes only approved operational commands and cannot import research promotion modules.

### P2-7 - Secret values use plain strings and optional CLI arguments can expose them

- **Location:** `src/market_predictor/config.py:24`, `src/market_predictor/config.py:32`, `src/market_predictor/config.py:35`, `src/market_predictor/config.py:41`, `src/market_predictor/sources/seeking_alpha.py:272`, `src/market_predictor/cli.py:725`
- **Defect or risk:** Passwords, API keys, and connection strings are ordinary strings; the Seeking Alpha access token is cached as plaintext; Finviz auth can be passed on the command line and exposed through process history.
- **Impact:** Accidental object logging, process inspection, or weak local file permissions can disclose credentials.
- **Evidence:** `.env` is correctly ignored and status commands avoid printing tokens, but secret redaction is convention rather than type/storage enforcement.
- **Remediation:** Use `SecretStr` or a secret wrapper, prohibit secret CLI values in production, retrieve Azure secrets through managed identity/Key Vault, and set restrictive permissions or avoid long-lived token files.
- **Verification:** Automated log/exception snapshots must redact secrets; process arguments must contain none; file-permission and managed-identity deployment tests must pass.

## Executive Verdict

The repository has a substantially stronger canonical data and model-validation design than its earlier research paths, but it is **not production deployable** at this commit. The live swing route is functionally blocked by two independent time/readiness defects. The serving release, evidence contract, promotion trust, runtime memory control, authentication, and durable audit store also require implementation before TradingFlow can safely consume predictions, even in paper mode.

No canonical model has passed real-data promotion, so this review makes no claim of predictive edge. Code/test quality and model validity are separate gates.

## Confirmed Strengths

- Canonical Pydantic contracts are frozen, reject extra fields, require timezone-aware timestamps, and encode availability policy: `src/market_predictor/canonical/contracts.py:15`.
- Event availability includes provider update, first observation, and sentiment-scoring latency: `src/market_predictor/canonical/contracts.py:122`.
- Canonical joins and audits fail on future features, unknown membership, source failures/staleness, non-SIP volume, malformed OHLCV, and proxy history in production: `src/market_predictor/canonical/audits.py:46`, `src/market_predictor/canonical/audits.py:89`, `src/market_predictor/canonical/audits.py:271`.
- Swing and intraday training/live features share the same mode-specific builders; live selection strips target/future columns and enforces one coherent decision group: `src/market_predictor/swing/dataset.py:85`, `src/market_predictor/intraday/dataset.py:103`, `src/market_predictor/live_features.py:50`.
- Canonical promotion gates use hash-inventoried evidence bundles, model-run IDs, walk-forward/holdout metrics, economics, drawdown, regimes, catalyst coverage, alignment, and memory evidence: `src/market_predictor/swing/promotion.py:208`, `src/market_predictor/intraday/promotion.py:237`.
- Azure releases are content-addressed, path-constrained, staged, hash-checked, and publish the active pointer last: `src/market_predictor/deployment.py:75`, `src/market_predictor/deployment.py:218`, `src/market_predictor/deployment.py:398`.
- The container runs as non-root and separates liveness from fail-closed readiness: `Dockerfile:31`, `src/market_predictor/api.py:83`, `src/market_predictor/api.py:87`.
- The Market Predictor/TradingFlow ownership boundary is explicit and architecture-tested: `tests/test_architecture_boundaries.py:11`, `tests/test_architecture_boundaries.py:66`. No alert, broker, order, position, or portfolio route was found.

## Test and Artifact Evidence

- Bounded review suite: **81 tests passed** across canonical data/CLI, swing/intraday dataset construction, live feature store, prediction service/snapshots, registry, deployment, API, and architecture boundaries.
- Direct swing reproduction: a one-row live snapshot with audited `daily_bar_count=260` was served as count `1`, readiness `invalid`, signal `not_ready`.
- Direct timing reproduction: a nightly collection completed at 02:01 UTC joined to the prior 20:15 UTC daily decision as `not_collected`.
- Direct horizon reproduction: request `60m` normalized to `1h` while the route remained `60m`.
- No Python process remained after testing; the review launched no sustained worker and did not approach the 4 GiB constraint.

## Missing Evidence and Unresolved Assumptions

- No `swing.model.v1` or `intraday.model.v1` artifact is promoted under the configured canonical paths.
- No canonical swing or intraday live feature snapshot is registered locally.
- No local active-serving-release marker exists.
- No real Azure publish/sync/rollback, identity, network, restart, or disaster-recovery rehearsal evidence exists.
- No mature observed point-in-time news history has been demonstrated for canonical promotion. Historical publication-time backfill remains research-only, correctly so.
- No production load, concurrency, soak, OOM, multi-replica, or failover evidence exists.
- No end-to-end TradingFlow `PredictionEvidenceV1` consumer, durable cache, or decision-audit evidence exists.
- No dependency lock, image digest, SBOM, or vulnerability report exists.
- No Reddit relevance benchmark or test corpus exists.

## Deployment Blockers

1. Resolve both P0 swing-runtime defects and prove the full nightly path.
2. Train and promote real canonical swing/intraday models through current gates; do not use legacy candidates.
3. Implement the versioned evidence contract and durable snapshot/outcome repository.
4. Make promotion attestation and local release activation immutable and coherent.
5. Add model caching, inference admission control, and measured 4 GiB load evidence.
6. Implement private service authentication, authorization, rate limits, and production-safe replay policy.
7. Lock dependencies and complete a real Azure release/rollback/DR rehearsal.
8. Integrate with TradingFlow only in `observe` mode after the above; alerts, risk, and execution remain entirely in TradingFlow.

## Ordered Remediation Plan

1. **Restore swing correctness:** introduce the explicit scoring cutoff, fix live warm-up propagation, and add end-to-end nightly/live tests.
2. **Freeze the wire contract:** canonicalize `60m`, add `PredictionEvidenceV1`, actual cutoff/feature/release identities, correlation, and typed errors.
3. **Harden promotion and release:** immutable promotion attestation, versioned local release directories, atomic pointer switch, and Azure ETag/lease control.
4. **Bound serving resources:** startup model cache, request semaphore, ticker limit, memory admission, and real-size load/soak tests.
5. **Make audit durable:** Blob-backed immutable prediction/outcome repository and deterministic replay using frozen price artifacts.
6. **Secure the service:** private ingress, workload identity, route authorization, replay isolation, and secret types/storage.
7. **Make ingestion transactional:** provider quota reservation, atomic caches, immutable run manifests, Reddit relevance filters, and one symbol master.
8. **Make builds reproducible:** split serving/training dependencies, lock with hashes, pin image digest, add SBOM/security gates, and broaden CI.
9. **Rehearse before paper use:** publish/sync/rollback/DR on Azure, then let TradingFlow consume only valid promoted evidence in `observe` mode for a predeclared shadow period.
