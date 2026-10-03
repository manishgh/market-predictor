from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import market_predictor.serving.prediction_service as service_module
from market_predictor.api import create_app
from market_predictor.core.prediction_contracts import (
    PredictionDriftBlockedError,
    PredictionReadinessError,
    PredictionRequest,
    PredictionValidationError,
)
from market_predictor.serving.outcome_intents import maturation_intents_from_response
from tests.support.swing_serving import NOW, UnavailableInputs, swing_serving

ROOT = Path(__file__).resolve().parents[1]


def test_promoted_ten_session_swing_api_returns_human_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serving = swing_serving(tmp_path, monkeypatch)
    service, contract, bundle = serving.service, serving.contract, serving.bundle
    with pytest.raises(PredictionDriftBlockedError):
        service.predict_swing(
            PredictionRequest(tickers=["T000"], mode="swing", as_of=NOW)
        )
    drift_checks: list[dict[str, object]] = []
    monkeypatch.setattr(
        service,
        "_require_actionable_drift",
        lambda **kwargs: drift_checks.append(kwargs),
    )
    direct = service.predict_swing(
        PredictionRequest(
            tickers=["T000", "T059", "MISSING"],
            mode="swing",
            as_of=NOW,
        )
    )
    assert direct.predictions[0].swing is not None
    assert direct.evidence is not None
    assert direct.evidence.identity_status == "complete"
    model = direct.models["swing"]
    assert model.training_data_end == "2026-01-05"
    assert model.training_labels_available_through_utc is not None
    assert model.training_labels_available_through_utc.isoformat() == "2026-01-20T21:00:00+00:00"
    assert model.prediction_policy is not None
    assert model.prediction_policy_sha256 == direct.evidence.view_prediction_policy_sha256["swing"]
    assert model.prediction_policy["minimum_probability"] == 0.60
    assert (
        model.prediction_policy["maximum_predictions_per_decision"]
        == contract.swing.maximum_trades_per_decision
    )
    intents = maturation_intents_from_response(direct, snapshot_id="a" * 64)
    assert {intent.ticker for intent in intents} == {"T000", "T059"}

    with TestClient(create_app(service)) as client:
        response = client.post(
            "/v1/predictions/swing",
            json={
                "tickers": ["T000", "T059", "MISSING"],
                "as_of": NOW.isoformat(),
            },
        )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["horizon"] == "10b"
    scored = payload["predictions"][0]["swing"]
    assert scored["action"] == "watch_for_entry"
    assert scored["probability"] == 0.72
    assert scored["expected_horizon"] == "up to 10 trading sessions"
    assert len(scored["benchmark_context"]) == 3
    assert scored["managed_risk"]["entry_reference"] == "next_session_open"
    assert scored["managed_risk"]["price_levels_available"] is False
    assert scored["managed_risk"]["target_distance_fraction"] > 0
    assert "target_price" not in scored["managed_risk"]
    assert "stop_price" not in scored["managed_risk"]
    assert scored["catalyst"]["event_count"] == 2
    assert scored["lineage"]["model_artifact_sha256"] == bundle["model_artifact_sha256"]
    ranked = payload["predictions"][1]["swing"]
    assert ranked["selection_eligible"] is True
    assert ranked["selected_for_policy"] is False
    assert ranked["action"] == "observe_ranked_candidate"
    abstained = payload["predictions"][2]["swing"]
    assert abstained["action"] == "abstain"
    assert abstained["abstention_reasons"] == ["out_of_universe"]
    assert payload["evidence"]["identity_status"] == "complete"
    assert len(drift_checks) == 2
    assert all(check["checked_at"] == NOW for check in drift_checks)

    with pytest.raises(PredictionValidationError):
        service.predict_swing(
            PredictionRequest(
                tickers=["T000"],
                mode="swing",
                as_of=NOW,
                requested_models=["xgboost_regressor"],
            )
        )

    serving.generation_cache.current = False
    with pytest.raises(PredictionReadinessError) as rollover:
        service.predict_swing(
            PredictionRequest(tickers=["T000"], mode="swing", as_of=NOW)
        )
    assert rollover.value.__cause__ is not None
    assert "generation changed" in str(rollover.value.__cause__)

    _, changed_policy_sha256 = service_module._swing_prediction_policy(
        probability_threshold=0.61,
        contract=contract,
    )
    assert changed_policy_sha256 != model.prediction_policy_sha256
    serving.generation_cache.current = True

    service.swing_live_input_provider = UnavailableInputs()
    with TestClient(create_app(service)) as client:
        unavailable = client.post(
            "/v1/predictions/swing",
            json={"tickers": ["T000"], "as_of": NOW.isoformat()},
        )
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "prediction_not_ready"


# Fields TradingFlow's MarketPredictorHttpClient reads; C# non-nullable ones must never be null.
_REQUIRED_RESPONSE = ("request_id", "generated_at_utc", "mode", "resolved_horizons", "models", "predictions", "errors", "snapshot_id")
_REQUIRED_MODEL = ("status", "model_type", "schema_version", "target", "artifact_sha256")
_REQUIRED_TICKER = ("ticker", "final_signal", "readiness_status", "swing", "errors")
_REQUIRED_SWING = ("probability", "decision_score", "signal", "rank", "return_1d", "volume_z20", "global_context", "catalyst", "readiness")
_REQUIRED_READINESS = ("status", "reasons", "price_feed", "benchmark_status", "market_context_status", "model_status", "source_status")
_REQUIRED_CATALYST = ("status", "direction", "score", "event_count", "relevance", "reasons")
_REQUIRED_GLOBAL = ("net_impact", "active_flashpoints")
# Metadata stays nullable on the wire; the promoted serving path requires verified values.
_NULLABLE = {"readiness": ("latest_price_date",), "catalyst": ("minutes_since_latest",)}
_NULLABLE_MODEL = ("training_data_end", "training_labels_available_through_utc")


def _require(payload: dict[str, object], fields: tuple[str, ...], where: str) -> None:
    missing = [field for field in fields if payload.get(field) is None]
    assert not missing, f"{where} lacks non-null {missing}"


def test_serialized_swing_response_carries_every_tradingflow_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serving = swing_serving(tmp_path, monkeypatch, enforce_drift=False, persist_snapshots=True)

    with TestClient(create_app(serving.service)) as client:
        response = client.post("/v1/predictions/swing", json={"tickers": ["T000"], "as_of": NOW.isoformat()})

    assert response.status_code == 200, response.text
    payload = response.json()
    _require(payload, _REQUIRED_RESPONSE, "response")
    assert payload["mode"] == "swing"
    assert payload["resolved_horizons"] == {"swing": "10b"}
    _require(payload["models"]["swing"], _REQUIRED_MODEL, "models.swing")
    assert set(_NULLABLE_MODEL) <= set(payload["models"]["swing"])
    [ticker] = payload["predictions"]
    _require(ticker, _REQUIRED_TICKER, "prediction")
    swing = ticker["swing"]
    _require(swing, _REQUIRED_SWING, "swing")
    _require(swing["readiness"], _REQUIRED_READINESS, "swing.readiness")
    _require(swing["catalyst"], _REQUIRED_CATALYST, "swing.catalyst")
    _require(swing["global_context"], _REQUIRED_GLOBAL, "swing.global_context")
    for section, fields in _NULLABLE.items():
        assert set(fields) <= set(swing[section]), f"swing.{section} lacks {fields}"


CONTRACT_FIXTURE = ROOT / "tests" / "fixtures" / "contracts" / "swing_prediction_response.json"
_FIXED_ID = "00000000-0000-4000-8000-000000000000"


def _contract_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """A served response with its per-call values fixed, as TradingFlow receives it."""
    serving = swing_serving(tmp_path, monkeypatch, enforce_drift=False, excluded_tickers=("T060",), peer_floor_tickers=("T061",))
    with TestClient(create_app(serving.service)) as client:
        response = client.post(
            "/v1/predictions/swing",
            json={"tickers": ["T000", "T059", "T060", "T061", "MISSING"], "as_of": NOW.isoformat()},
        )
    assert response.status_code == 200, response.text
    payload = response.json()
    payload["request_id"] = _FIXED_ID
    payload["generated_at_utc"] = NOW.isoformat().replace("+00:00", "Z")
    payload["evidence"]["request_id"] = _FIXED_ID
    payload["evidence"]["correlation_id"] = _FIXED_ID
    model = payload["models"]["swing"]
    model["path"] = Path(model["path"]).relative_to(tmp_path).as_posix()
    return payload


def test_contract_fixture_matches_the_served_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared fixture TradingFlow parses; regenerate it deliberately when the contract changes."""
    payload = _contract_payload(tmp_path, monkeypatch)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if os.environ.get("MARKET_PREDICTOR_WRITE_CONTRACT_FIXTURE") == "1":
        CONTRACT_FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        CONTRACT_FIXTURE.write_bytes(text.encode("utf-8"))
    assert CONTRACT_FIXTURE.read_bytes().decode("utf-8") == text, (
        "the served swing response changed; update docs/contracts/prediction_api.md, regenerate the "
        "fixture with MARKET_PREDICTOR_WRITE_CONTRACT_FIXTURE=1 and tell TradingFlow"
    )


def test_monitoring_cross_section_exceeds_request_limit_and_preserves_client_drift_gate(tmp_path, monkeypatch):
    serving = swing_serving(tmp_path, monkeypatch, member_count=120, excluded_tickers=("INPUT",), peer_floor_tickers=("THIN",))
    result = serving.service.predict_swing_cross_section(NOW)
    assert len(result.members) == len(result.response.predictions) == 122
    assert len(result.response.evidence.row_feature_availability) == 120
    reasons = {row.ticker: row.swing.abstention_reasons for row in result.response.predictions}
    assert reasons["INPUT"] == ["live_inputs_incomplete"]
    assert reasons["THIN"] == ["sector_peer_floor"]
    with pytest.raises(PredictionDriftBlockedError):
        serving.service.predict(PredictionRequest(tickers=["T000"], mode="swing", as_of=NOW))
    serving.generation_cache.current = False
    with pytest.raises(PredictionReadinessError) as error:
        serving.service.predict_swing_cross_section(NOW)
    assert "generation changed" in str(error.value.__cause__)


def test_monitoring_cross_section_rejects_future_promotion_and_naive_time(tmp_path, monkeypatch):
    from dataclasses import replace
    from datetime import timedelta
    serving = swing_serving(tmp_path, monkeypatch)
    generation = serving.generation_cache.generation
    serving.generation_cache.generation = replace(generation, bundle=generation.bundle.model_copy(
        update={"promoted_at_utc": NOW + timedelta(seconds=1)},
    ))
    with pytest.raises(PredictionReadinessError) as error:
        serving.service.predict_swing_cross_section(NOW)
    assert "unavailable at the requested as_of" in str(error.value.__cause__)
    with pytest.raises(PredictionValidationError):
        serving.service.predict_swing_cross_section(NOW.replace(tzinfo=None))


def test_all_thin_members_abstain_without_estimator_and_keep_decision_cutoff(tmp_path, monkeypatch):
    from dataclasses import replace
    serving = swing_serving(tmp_path, monkeypatch)
    live = service_module.build_live_swing_features()
    empty = replace(live, technical_market=live.technical_market.iloc[:0], catalyst_full=live.catalyst_full.iloc[:0],
                    context=live.context.iloc[:0], members=tuple(member.model_copy(update={"abstention_reason": "sector_peer_floor"})
                                                              for member in live.members))
    monkeypatch.setattr(service_module, "build_live_swing_features", lambda *args, **kwargs: empty)
    def fail_predict(*args, **kwargs):
        pytest.fail("empty cross-section reached the estimator")
    monkeypatch.setattr(service_module.SwingInferenceEngine, "predict", fail_predict)
    result = serving.service.predict_swing_cross_section(NOW)
    assert len(result.response.predictions) == 60
    assert all(row.swing.abstention_reasons == ["sector_peer_floor"] for row in result.response.predictions)
    assert result.response.evidence.prediction_cutoff_utc == live.decision_time_utc
    assert result.response.evidence.row_feature_availability == []


def test_peer_floor_is_distinct_from_nonmembership_in_public_response(tmp_path, monkeypatch):
    serving = swing_serving(tmp_path, monkeypatch, enforce_drift=False, peer_floor_tickers=("THIN",))
    response = serving.service.predict(PredictionRequest(tickers=["THIN", "UNKNOWN"], mode="swing", as_of=NOW))
    assert response.contract_version == "market_predictor.prediction.v4"
    assert response.predictions[0].swing.abstention_reasons == ["sector_peer_floor"]
    assert response.predictions[1].swing.abstention_reasons == ["out_of_universe"]
