# Prediction API Contract

Owner: Market Predictor (producer). Consumer: TradingFlow (`MarketPredictorHttpClient`).
Contract name: `market_predictor.prediction`. It carries no version: nothing is in production,
and versions start only once the API is (a user rule, September 27).
Golden fixture: `tests/fixtures/contracts/swing_prediction_response.json`, a real served
response kept equal to the code by
`tests/test_swing_prediction_api.py::test_contract_fixture_matches_the_served_response`.

Predictions are advisory evidence. They are not alerts, orders, positions or execution
instructions, and a consumer must never treat them as trading authority.

## Endpoints

- `POST /v1/predictions/swing`: the only prediction endpoint.
- `GET /v1/health/live`: 200 while the process runs.
- `GET /v1/health/ready`: 200 only when a promoted model, live inputs and an actionable
  drift assessment are all available; otherwise 503. Nothing is promoted today, so a
  deployment answers 503.

Authentication follows the deployment (`Authorization: Bearer <token>`).

## Request

```json
{"tickers": ["MSFT"], "horizon": "auto", "as_of": "2026-07-08T22:05:00Z", "correlation_id": "optional"}
```

- `tickers`: 1 to 100 unique canonical US symbols (`[A-Z][A-Z0-9.-]{0,14}`, upper-cased).
- `horizon`: `auto` or `10b` (ten exchange sessions); anything else is 422.
- `mode`: optional; only `swing` is accepted.
- `as_of`: optional, must carry a timezone.
- Unknown fields are refused with 422.

## Response (200)

Every field below is always present. "non-null" means TradingFlow may rely on a value.

| Path | Type | Notes |
| --- | --- | --- |
| `contract` | string | `market_predictor.prediction` |
| `request_id` | string, non-null | |
| `generated_at_utc` | ISO-8601 UTC, non-null | |
| `mode` | `swing`, non-null | |
| `data_source` | `live` | |
| `horizon` | string, non-null | the resolved horizon, `10b` |
| `resolved_horizons` | object, non-null | always `{"swing": "10b"}` |
| `models.swing` | object, non-null | see Model |
| `predictions` | array, non-null | one entry per requested ticker, in request order |
| `errors` | array of strings, non-null | response-level problems |
| `evidence` | object | point-in-time identities; clients may ignore it |
| `snapshot_id`, `snapshot_sha256` | string or null | set when the server persists the response |

Model (`models.swing`): `status`, `model_type`, `schema_version`, `target`,
`artifact_sha256` are non-null. `training_data_end` is null today (promoted swing
artifacts do not yet record it; see Pending changes). `path` is server-internal and
must not be used.

Ticker prediction (`predictions[]`):

| Field | Type | Notes |
| --- | --- | --- |
| `ticker` | string, non-null | |
| `final_signal` | string, non-null | equals `swing.signal` |
| `readiness_status` | `valid`, `warn` or `invalid` | |
| `swing` | object, non-null | see Swing prediction |
| `errors` | array of strings, non-null | the abstention reasons, if any |

Swing prediction (`predictions[].swing`), the fields TradingFlow reads:

| Field | Type | Notes |
| --- | --- | --- |
| `probability` | number or null | null when the ticker abstains |
| `decision_score` | number or null | equals `probability` |
| `signal` | string, non-null | see Signals and actions |
| `action` | string, non-null | see Signals and actions |
| `rank` | integer or null | rank in the whole scored cross-section |
| `selection_eligible`, `selected_for_policy` | boolean | selected means `watch_for_entry` |
| `return_1d`, `volume_z20` | number or null | |
| `abstention_reasons` | array of strings | see Abstention reasons |
| `readiness` | object, non-null | `status`, `reasons`, `price_feed`, `benchmark_status`, `market_context_status`, `model_status`, `source_status` non-null; `latest_price_date` may be null |
| `catalyst` | object, non-null | `status` (`confirmed`, `conflicting`, `veto`, `mixed`, `absent`), `direction` (`positive`, `negative`, `mixed`, `none`), `score`, `event_count`, `relevance`, `reasons` non-null; `minutes_since_latest` may be null |
| `global_context` | object, non-null | `net_impact` number, `active_flashpoints` array |
| `expected_horizon` | `up to 10 trading sessions` | |
| `benchmark_context`, `managed_risk` | present when scored | managed risk gives distances as fractions, never price levels |

## Signals and actions

| `signal` (= `final_signal`) | `action` | Meaning |
| --- | --- | --- |
| `positive_setup` | `watch_for_entry` | selected by the served policy |
| `ranked_candidate` | `observe_ranked_candidate` | above the probability threshold, not selected |
| `neutral` | `hold_off` | below the threshold |
| `low_probability` | `avoid` | probability at or below 0.40 |
| `abstain` | `abstain` | not scored; see the reasons |

The retired names (`bullish_watch`, `strong_bullish_watch`, `bullish_watch_confirmed`,
`high_conviction_watch`, `watch_for_confirmation`) are never emitted.

## Abstention reasons

- `out_of_universe`: the ticker is not an effective point-in-time member.
- `live_inputs_incomplete`: a member whose market or catalyst inputs were incomplete at
  the decision; it was not scored.

## Errors

Non-200 responses carry `{"error": {"code", "message", "correlation_id", "retryable"}}`.

| Status | Codes |
| --- | --- |
| 422 | `prediction_validation_error` |
| 404 | `prediction_not_found` |
| 409 | `prediction_conflict` |
| 429 | `prediction_throttled` |
| 503 | `inference_capacity_exhausted`, `memory_pressure`, `prediction_not_ready`, `prediction_model_unavailable`, `prediction_drift_blocked`, `prediction_dependency_unavailable` |

## Change rules

- Market Predictor changes this contract; TradingFlow never assumes a change that is not
  written here.
- Every change adds a Change log entry (date, what changes, compatibility, the new
  fixture SHA-256) before the code lands, and the fixture is regenerated with
  `MARKET_PREDICTOR_WRITE_CONTRACT_FIXTURE=1`.
- Until production, every change keeps the unversioned name and is announced only in the
  Change log. Once in production, a field removal, a type or nullability change, a new
  enum value or a meaning change will introduce versions.
- TradingFlow acknowledges an entry in its notes file before relying on it.

## Pending changes (designed, not yet in code)

From the swing monitoring and replay correctness design, under review:
- `models.swing.training_data_end` will hold the last training decision session date
  (`YYYY-MM-DD`), and a new optional `models.swing.training_labels_available_through_utc`
  (ISO-8601 UTC) the instant the training labels became available. Both stay null until
  a promoted model records them.

## Change log

- 2026-09-27 (later): versions removed. The response field `contract_version` is renamed
  `contract` and holds `market_predictor.prediction`; evidence carries `contract`
  `market_predictor.prediction_evidence`. The fixture is renamed
  `swing_prediction_response.json`, SHA-256
  `b62f577dbd9ca1a09c9959d5da84df0db9040d8e6b3c17d26eabfcb4f369f1cf`.
- 2026-09-27: `market_predictor.prediction.v3` published as this document with its
  golden fixture. Relative to the v1 TradingFlow knew: intraday and unified views,
  `unified_score` and `readiness.intraday_bar_count` are removed; horizons are
  exchange-session counts; unknown fields are refused; signal names are those above;
  abstention reason `live_inputs_incomplete` is new. Fixture SHA-256
  `e445eaac20a42c21a5d83900d3ec3f63c9354745102d9633635143f862956c95`.
