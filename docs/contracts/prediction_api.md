# Prediction API Contract

Owner: Market Predictor (producer). Consumer: TradingFlow (`MarketPredictorHttpClient`).
Current version: `market_predictor.prediction.v1`. The API is the only versioned name:
models are not final, so internal records and nested policies carry no version.
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
| `contract_version` | string | `market_predictor.prediction.v1` |
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
`artifact_sha256` are non-null. `training_data_end` is the final-fit decision-end
session date, not the time when the model's information became available. The optional
`training_labels_available_through_utc` is the latest label availability across fit,
calibration, selection and promotion-test populations. The promoted serving path
requires both values to match the verified model artifact. `path` is server-internal
and must not be used.

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
- This is the initial pre-production V1 system. Implementation iterations do not
  increment API/data/model versions. Update the canonical schema and both consumers
  together, recording fixture changes; no compatibility aliases are required. A later
  major version requires explicit acceptance of an improved system and coordinated
  migration. V1 is not evidence of a trained, promoted or SPY-beating model.
- TradingFlow acknowledges an entry in its notes file before relying on it.

`sector_peer_floor` means the ticker is a verified member, but its sector naturally
has fewer eligible peers than the frozen ranking floor. An input failure which
causes a sector to fall below that floor remains `live_inputs_incomplete` and counts
toward the input-failure ceiling. Neither abstention carries a fabricated model score.

## Historical replay information boundary

`POST /v1/replays/investment` remains a historical simulation of a swing snapshot;
it does not forecast the 63- or 252-session investment horizons. Replay requires the
model's label-information boundary to be strictly earlier than the matching typed
prediction row's decision timestamp. Missing or ambiguous row evidence refuses replay.
The request's as-of and `training_data_end` cannot replace these timestamps.
The response exposes `model_training_labels_available_through_utc`. Entry selection
uses `selected_for_policy`, and daily bar execution uses actual XNYS opens and closes,
including early closes. Historical snapshots without the boundary cannot be replayed.

## Change log

- 2026-10-03 (current, explicit user reset): public contract is
  `market_predictor.prediction.v1` on unchanged `/v1/` routes. Internal schema and
  policy identifiers are descriptive and unversioned, with no legacy acceptance
  paths. Payload field structure remains current; model/policy/evidence hashes change
  where their canonical identities changed. Regenerated served fixture SHA-256:
  `59288471c249b0422443f0a68ce7623f4b03956826877ad1158dc79ebf1fea51`.
  This is a pre-production reset, not backwards compatibility with historical V1
  payloads and not a new model admission. TradingFlow's current working checkout
  expects `contract: market_predictor.prediction`; its owner must change to
  `contract_version`, accept only the current V1 payload and pin this fixture.
  No TradingFlow acceptance test or build was run by this task.

- 2026-10-03: API v4 adds optional `models.swing.training_labels_available_through_utc`
  (ISO-8601 UTC) and populates nullable `training_data_end` from verified promoted
  metadata. Existing field types, signal names and actions are unchanged. Replay adds
  the corresponding `model_training_labels_available_through_utc` field and refuses
  missing or future model information. TradingFlow remains untouched; its fixture
  acknowledgement is pending before it relies on the new optional field. Fixture
  SHA-256: `bc1f5109be83fc2985c28b0c0ded75e0487e34a68c237a1f147babc8e117a08b`.

- 2026-09-28: API v4 adds `sector_peer_floor` for effective members in naturally
  thin sectors and removes unused `evidence.row_feature_availability[].decision_atr`.
  Input failures and their cascaded peer exclusions stay `live_inputs_incomplete`;
  `out_of_universe` is reserved for nonmembers. The 100-ticker HTTP limit is unchanged.
  Internal full-cross-section scoring bypasses only drift and is not an HTTP route.
  TradingFlow must accept v4 and re-pin the regenerated golden fixture. Fixture
  SHA-256: `96cbcd133b8e96253b034fabba10624be79b0d80550f043f3869905251bec522`.

Entries are append-only history, newest first; the newest entry is the current contract.

- 2026-09-27 (historical correction): the user clarified that the public API stays
  versioned. The response field is `contract_version` again, value
  `market_predictor.prediction.v3`; routes stay under `/v1/`. This reverses the
  response-level change in the entry below. The nested records keep that entry's
  change: `evidence.contract` is `market_predictor.prediction_evidence` and
  `models.swing.prediction_policy.contract` is `market_predictor.swing_prediction_policy`,
  and `models.swing.label_policy.policy` becomes `market_predictor.swing_outcome_policy`.
  Renaming those nested fields under v3 is a recorded pre-production exception to the
  change rules: they are internal identities that TradingFlow does not read. Fixture
  `swing_prediction_response.json`, SHA-256
  `360206c2440e4fb39bac210a50b12c6d233e4f49a140fd356e5d283b034536cb`.
  TradingFlow action: read `contract_version`, accept only `market_predictor.prediction.v3`,
  and re-pin the fixture to this hash.
- 2026-09-27 (later, superseded by the correction above): versions removed. The response
  field `contract_version` was renamed `contract` holding `market_predictor.prediction`;
  evidence and prediction-policy records carried `contract` without a version. Fixture
  `swing_prediction_response.json`, SHA-256
  `b62f577dbd9ca1a09c9959d5da84df0db9040d8e6b3c17d26eabfcb4f369f1cf`.
- 2026-09-27: `market_predictor.prediction.v3` published as this document with its
  golden fixture. Relative to the v1 TradingFlow knew: intraday and unified views,
  `unified_score` and `readiness.intraday_bar_count` are removed; horizons are
  exchange-session counts; unknown fields are refused; signal names are those above;
  abstention reason `live_inputs_incomplete` is new. Fixture SHA-256
  `e445eaac20a42c21a5d83900d3ec3f63c9354745102d9633635143f862956c95`.
