# Research return API V1

The canonical V1 application provides six trained specifications: technical,
relationships and news profiles, each with ridge and XGBoost. All research
responses are non-actionable. A trained model is not a promoted release, and
these routes grant no trading admission.

Start the optional workbench using the factory command in [README](../README.md).
Use its bearer-token field, or send `Authorization: Bearer <token>`. Development
mode requires `API_ENVIRONMENT=development`, `API_AUTH_MODE=development` and a
private `API_DEVELOPMENT_BEARER_TOKEN` of at least 32 characters. Entra mode uses
its configured issuer, audience and local JWKS. The routes require
`predictions.read`; a missing token returns 401 and insufficient scope returns 403.

In development, open `/docs`, select **Authorize**, enter the token and use
**Try it out**. `/openapi.json` contains the same schemas, examples and error
models. Entra mode disables both documentation endpoints.

## Requests

For an application, start with a daily workflow: let the configured collectors
ingest stock prices and news, publish a verified feature generation after the
nightly decision cutoff, then request forecasts for the desired tickers. These
models use daily bars and a ten-session horizon; clients do not need to stream
prices or news into the forecast endpoint. Continuous collection can update the
source pipeline independently. Historical queries do not require new collection.

`GET /v1/research/models` lists the six model ids, pinned training status and
fresh-input publication status. Missing artifacts remain unavailable; the catalog
does not fit or promote anything.

`POST /v1/research/predict` reads exact retained temporal out-of-fold predictions:

```json
{"model_id":"news_ridge","session_date":"2021-07-21","tickers":["ABT"]}
```

Only exact saved scoring sessions are accepted. No nearby session or final model
substitutes for a missing historical prediction. These already inspected sessions
are development evidence, not an untouched final test.

`POST /v1/research/forecast` calculates a fresh estimate using the frozen final
model and verified current observed features:

```json
{"model_id":"technical_ridge","tickers":["MSFT","AAPL"]}
```

An optional `as_of` must carry an explicit timezone. Omitting it uses current UTC.
A current verified feature publication must cover that decision and follow the
final model's training cutoff. Missing, stale or future inputs refuse the request.
Current fresh-input publication is a separate data requirement; the HTTP request
does not collect prices/news, build features or refit the model.

The daily input job runs `market-predictor-research publish-observed-return-features
--request <registration.json> --output-directory <new-generation>`. Its typed
registration binds adjusted stock history, separate raw decision bars, benchmark
history, observed membership and news authorities, the original research cohort,
strategy contract, feature request and three frozen model runs. The command
uses the same technical, peer, relationship and news feature calculations as
training and publishes all three profiles together. Source clocks must include
actual ingestion; historical proxy sources cannot be declared observed.
The existing saved-configuration correspondence evidence is required where a
run retains the earlier configuration names; it verifies the unchanged numerical
contract while preserving those original hashes.

After a successful verified publication, configure `forecast.publication_directory`
and its exact `forecast.publication_manifest_sha256` in the server-owned model
catalog. The API verifies that registered generation, source bytes, feature order,
population and implementation before inference. The current catalog intentionally
has no registered observed generation: historical access is verified, while fresh
requests return 503 until suitable observed sources are published. No current
fresh-data or deployment proof has been completed.

The service loads this catalog when it starts. After registering a new immutable
generation, recreate or restart the service before requesting that generation;
editing the catalog alone does not reload a running instance. Keep source collection
ahead of the 18:00 America/New_York cutoff: information first observed later cannot
be backdated into that decision. SPY remains required benchmark evidence regardless
of any approved issuer exclusions from the research cohort.

Requests accept model id, canonical US ticker symbols and the applicable date or
cutoff only. They reject unknown fields, including raw prices/news, feature values
and artifact paths. Up to 100 tickers fit the public schema; a lower
`RUNTIME_MAX_TICKERS_PER_REQUEST` also applies. The shared configured body and
principal rate limits apply before processing.

## Results and errors

`predicted_excess_return` is the signed ten-session stock net return minus SPY
gross return as a fraction. `-0.012` means -1.2 percentage points relative to SPY,
not a probability. JSON `null` means unavailable and differs from a genuine
`0.0`. Responses identify their historical/fresh prediction basis and exact model
and input provenance. Every estimate remains research-only and non-actionable.

Failures use the shared API envelope:

```json
{"error":{"code":"prediction_not_ready","message":"Prediction inputs or models are not ready.","correlation_id":"request-correlation-id","retryable":true}}
```

| Status | Meaning |
| --- | --- |
| 401 / 403 | Missing/invalid bearer token or insufficient scope |
| 404 | Unknown model id |
| 409 | Pinned model or retained artifacts conflict with the expected bytes/contracts |
| 413 | Request exceeds the configured body limit |
| 422 | Invalid request or too many tickers |
| 429 | Principal rate limit exceeded |
| 503 | Required historical session, fresh inputs or forecast resources unavailable |

`x-correlation-id` accompanies responses. Rate-limited responses also provide
`retry-after`; use the error's `retryable` flag when deciding whether to retry.
