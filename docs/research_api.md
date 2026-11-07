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

## Investment fundamental features and evaluation

These research CLI operations are separate from the six existing swing models
and HTTP forecast routes. They preserve the investment contract: 252 XNYS
sessions primary, 63 diagnostic, next-open entry, marked terminal value without
liquidation, explicit corporate events/dividend cash, prepaid 20-bp stock cost,
and stock-net minus gross SPY return. QQQ and sector holding outcomes remain
required for complete target eligibility. Do not substitute reinvested adjusted
price ratios or reuse ten-session model weights/splits.

```powershell
market-predictor-collect collect-investment-company-facts --root C:/project/market-predictor --cik 0000789019 --security-id cik:0000789019 --output C:/project/market-predictor/data/raw/investment_company_facts/0000789019/new-collection
market-predictor-research build-investment-features --root C:/project/market-predictor --request data/research/investment-request.json --request-sha256 <sha256> --output <new-feature-directory>
market-predictor-research evaluate-investment-predictions --root C:/project/market-predictor --request data/research/investment-evaluation.json --request-sha256 <sha256> --output <new-evaluation-directory>
```

Collection uses the existing configured SEC identity, client and request governor.
Each issuer has an independent immutable directory; a failed issuer does not
invalidate another completed collection. No universe exclusions are selected by
these commands. Retain approved whole-security exclusion rules and mandatory SPY.
All operations acquire the shared lease under the supplied root's `data/runtime`
and enforce the existing host/process memory guard before input work/publication.
Use the same source root as other data jobs. Inputs are hash-pinned; outputs are
new immutable directories. Never reuse a completed output directory.

The feature request is `InvestmentFeatureBuildRequest` in
`research/investment_features.py`: `company_facts_manifest` and
`acceptance_clock_manifest` each contain root-relative `path` and `sha256`;
`decisions` contains `security_id`, ten-digit `cik` and aware `decision_time`;
`scope` is explicitly `nonfinancial`; `availability_basis` is `observed` or
`reconstructed_research`. Decisions must match the collected stable identity.
The corrected SEC clock manifest must bind the retained submission archive.
Missing accession clocks exclude facts with recorded reasons, never an invented
midnight or first-seen time. Historical reconstruction remains research evidence.
Observed mode cannot use a newly collected fact before its actual retrieval.

Ordered features: `revenue_growth_ttm`, `operating_margin_ttm`,
`operating_margin_change`, `fcf_margin_ttm`, `capex_to_revenue_ttm`,
`cash_conversion_ttm`. Revenue and cash-flow arithmetic use compatible annual,
YTD and standalone-quarter periods; ratios require matching aggregate intervals.
Negative profits/cash flow are retained; nonpositive ratio denominators and
ambiguous periods produce null plus a reason. No automatic imputation occurs.
Annual/YTD bridges assume consistent accounting across filings; ordinary 10-Q
comparative revisions do not necessarily carry an amendment suffix. This remains
an explicit research assumption, not admission of audited historical features.
Exact date matching can leave 52/53-week and leap-year comparisons unavailable.
Debt, market-cap yields, DCF, consensus and AI-specific features are deferred until
their required total-debt, shares, vintage and accounting evidence is qualified.

The evaluator request is `InvestmentEvaluationRequest`: canonical `requests` and
`targets`, complete `baseline` and `candidate` prediction arrays, and an aware
`evaluation_as_of_utc`. Each prediction binds request/policy hashes, stable identity,
decision/session/horizon, model artifact identity, fit time and latest training-label
availability. Null predictions require reasons. Exact paired cohorts, strict
training-label/fit/decision ordering and outcome maturity are checked. It reports
coverage, equal-date MAE/MSE, zero-SPY-excess errors and date-level rank correlations.
Overlapping annual outcomes are not independent observations: no IID confidence
intervals, trade-return annualization or profitability claim is produced.

The same pure feature kernel serves one decision or a historical batch. Publication
retains normalized source facts, source locators/hashes, consumed fact hashes,
ordered feature-contract hash, missingness and input/code identities. This is not
an HTTP model activation. Complete historical source/label coverage, an admitted
training pipeline and prospective evaluation are still required; every generated
artifact explicitly retains false training/promotion/production flags.

### Broad investment coverage and label qualification

`plan-investment-coverage` freezes monthly decisions, historical sector scope and
issuer identities before collection. Its `InvestmentCoverageRequest` pins the
approved cohort, membership, SEC identity relations and identity manifest,
corrected acceptance-clock manifest, and optional reused CompanyFacts manifests.
`collect-investment-coverage` consumes the resulting roster manifest and its hash.
It collects once per issuer and retains each security class separately. Resume
uses the same request, code and source identities; completed issuer checkpoints
are immutable. Missing issuers and features remain explicit coverage rows.

```powershell
market-predictor-research plan-investment-coverage --root C:/project/market-predictor --request <request.json> --request-sha256 <sha256> --output <new-roster-directory>
market-predictor-collect collect-investment-coverage --root C:/project/market-predictor --roster-manifest <roster-directory/_manifest.json> --roster-manifest-sha256 <sha256> --output <coverage-directory>
market-predictor-collect collect-investment-dividend-currencies --root C:/project/market-predictor --action-config configs/swing_corporate_action_sources.toml --action-config-sha256 <sha256> --action-archive <action-archive> --action-audit-sha256 <sha256> --output <currency-directory>
market-predictor-research materialize-investment-labels --root C:/project/market-predictor --request <label-request.json> --request-sha256 <sha256> --output <label-directory>
```

The currency collector requests existing dividend action IDs and preserves raw
responses. Qualification requires explicit USD and unchanged entitlement terms.
Missing currency, cancellations, revisions and unsupported distributions remain
unresolved. Neither an issuer's domicile nor its stock's trading currency proves
the currency of a distribution.

The label request is `InvestmentLabelMaterializationRequest` in
`research/investment_labels.py`. It pins `roster_manifest`, `target_config` and
optional `currency_manifest`, each with `path` and `sha256`. It also requires
`nominal_unpaid_claim_valuation: true` and an aware `evaluation_as_of_utc`.
Source qualification and availability by that frozen evaluation cutoff are
reported separately. Verification performed after the cutoff cannot be claimed
available before it. Per-security checkpoints retain all requested decisions and
horizons; the final artifact indexes them without loading the whole target corpus
into memory.

Investment labels use raw SIP prices and the shared holding evaluator over exact
63/252-session windows. Ordinary verified dividend entitlements may be valued as
unpaid, nontradable receivables under an explicit nominal-value research policy.
That valuation does not establish payment, spendable cash or reinvestment. Source
capture and membership verification clocks constrain label availability now;
reconstructed historical membership does not establish historical observation.
All required stock, SPY, QQQ and sector components must qualify. Incomplete paths
retain reasons; adjusted prices and ten-session swing labels cannot replace them.

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
