from __future__ import annotations

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
# Promoted swing artifacts do not yet record their training end, so the model field may be null.
_NULLABLE = {"readiness": ("latest_price_date",), "catalyst": ("minutes_since_latest",)}
_NULLABLE_MODEL = ("training_data_end",)


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
