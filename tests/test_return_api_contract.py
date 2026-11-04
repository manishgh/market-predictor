"""Synthetic unit checks for the shared HTTP boundary; no model/data job runs."""
from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from market_predictor import admission as admission_module
from market_predictor import api
from market_predictor.api_security import ApiSecurityConfig
from market_predictor.canonical.store import file_sha256
from market_predictor.config import Settings
from market_predictor.research_api import forecast as forecast_module
from market_predictor.research_api.catalog import PROFILES
from market_predictor.research_api.forecast import ReturnForecastService
from market_predictor.research_api.service import (
    ResearchFeatureUnavailableError,
    ResearchModelService,
    ResearchModelUnavailableError,
)
from market_predictor.serving.prediction_service import PredictionService, serving_routes_from_config
from tests.test_api import StubPredictionService


@pytest.fixture
def unit_models(tmp_path: Path) -> ResearchModelService:
    catalog = ['schema = "market_predictor.research_model_catalog"']
    for profile in PROFILES:
        catalog.extend(['[[profiles]]', f'id = "{profile}"', f'label = "{profile}"',
            f'artifact_directory = "data/{profile}"', f'manifest_sha256 = "{"0" * 64}"'])
    path = tmp_path / "catalog.toml"
    path.write_text("\n".join(catalog), encoding="utf-8")
    return ResearchModelService(tmp_path, catalog_path=path)


def _client(models: ResearchModelService) -> TestClient:
    return TestClient(api.create_app(cast(PredictionService, StubPredictionService()), return_model_service=models,
        security_config=ApiSecurityConfig(mode="disabled")))


def _payload(path: str) -> dict[str, Any]:
    result: dict[str, Any] = {"model_id": "technical_ridge", "tickers": ["MSFT", "AAPL"]}
    if path.endswith("predict"):
        result["session_date"] = "2021-07-21"
    return result


@pytest.mark.parametrize("path", ["/v1/research/predict", "/v1/research/forecast"])
@pytest.mark.parametrize(("error", "status", "code", "retryable"), [
    (KeyError("UNIT-only private artifact name"), 404, "prediction_not_found", False),
    (ResearchModelUnavailableError("UNIT-only private artifact name"), 409, "prediction_conflict", False),
    (ResearchFeatureUnavailableError("UNIT-only private artifact name"), 503, "prediction_not_ready", True),
])
def test_domain_failures_use_the_shared_sanitized_error_envelope(unit_models: ResearchModelService,
        monkeypatch: pytest.MonkeyPatch, path: str, error: Exception, status: int, code: str, retryable: bool) -> None:
    def fail(self: Any, **parameters: Any) -> dict[str, Any]:
        raise error

    monkeypatch.setattr(ResearchModelService, "predict", fail)
    monkeypatch.setattr(ReturnForecastService, "forecast", fail)
    response = _client(unit_models).post(path, json=_payload(path), headers={"x-correlation-id": "unit-error"})
    assert response.status_code == status
    body = response.json()
    assert set(body) == {"error"}
    assert set(body["error"]) == {"code", "message", "correlation_id", "retryable"}
    assert body["error"]["code"] == code
    assert body["error"]["retryable"] is retryable
    assert body["error"]["correlation_id"] == response.headers["x-correlation-id"] == "unit-error"
    assert "private artifact" not in response.text


@pytest.mark.parametrize("path", ["/v1/research/predict", "/v1/research/forecast"])
@pytest.mark.parametrize("limit", [1, 2])
def test_each_app_uses_its_configured_ticker_limit_before_artifact_access(unit_models: ResearchModelService,
        monkeypatch: pytest.MonkeyPatch, path: str, limit: int) -> None:
    calls: list[list[str]] = []

    def query(self: Any, **parameters: Any) -> dict[str, Any]:
        calls.append(parameters["tickers"])
        return {"unit_test_only": True}

    monkeypatch.setattr(ResearchModelService, "predict", query)
    monkeypatch.setattr(ReturnForecastService, "forecast", query)
    settings = Settings(RUNTIME_MAX_TICKERS_PER_REQUEST=limit)
    monkeypatch.setattr(api, "get_settings", lambda: settings)
    app = api.create_app(return_model_service=unit_models, security_config=ApiSecurityConfig(mode="disabled"))
    response = TestClient(app).post(path, json=_payload(path))
    if limit == 1:
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "prediction_validation_error"
        assert not calls
    else:
        assert response.status_code == 200
        assert calls == [["MSFT", "AAPL"]]


@pytest.mark.parametrize("path", ["/v1/research/predict", "/v1/research/forecast"])
@pytest.mark.parametrize("field", ["prices", "news", "features", "artifact_path"])
def test_public_requests_refuse_raw_inputs_and_artifact_control(unit_models: ResearchModelService,
        path: str, field: str) -> None:
    response = _client(unit_models).post(path, json={**_payload(path), field: "UNIT-only disallowed input"})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_error"


def test_swagger_documents_bearer_auth_signed_returns_and_canonical_errors(unit_models: ResearchModelService) -> None:
    schema = _client(unit_models).get("/openapi.json").json()
    assert schema["info"]["version"] == "1"
    assert schema["components"]["securitySchemes"]["ApiBearer"]["scheme"] == "bearer"
    for path in ("/v1/research/predict", "/v1/research/forecast"):
        operation = schema["paths"][path]["post"]
        assert operation["security"] == [{"ApiBearer": []}]
        assert "-1.2 percentage points" in operation["description"]
        assert "raw prices, news" in operation["description"]
        assert operation["responses"]["503"]["content"]["application/json"]["schema"]["$ref"].endswith(
            "/PredictionApiErrorEnvelope")
    assert _client(unit_models).get("/docs").status_code == 200

def test_fresh_forecast_shares_original_admission_and_recovers_after_release(unit_models: ResearchModelService,
        monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Synthetic UNIT inputs isolate admission; no real model or feature job runs."""
    service = PredictionService(tmp_path, routes=serving_routes_from_config(Settings().app_config),
        inference_memory_reservation_gib=0.375)
    monkeypatch.setattr(admission_module, "_current_rss_gib", lambda: 0.1)
    workspace_calls: list[str] = []
    artifact_reads: list[str] = []

    @contextmanager
    def workspace(name: str, **parameters: Any) -> Iterator[None]:
        snapshot = service.admission.snapshot()
        assert snapshot.active_requests == 1
        assert snapshot.reserved_gib == service.inference_memory_reservation_gib
        workspace_calls.append(name)
        yield

    def run(spec: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        artifact_reads.append(spec.model_id)
        return {"request_sha256": "0" * 64}, {"feature_names": [], "policy": {"target": "UNIT-only"}}

    def unit(*parameters: Any) -> tuple[Path, dict[str, Any]]:
        return tmp_path, {"unit": {"fit_cutoff_utc": "2020-01-01T22:00:00Z"}, "files": {"model.joblib": "0" * 64}}

    feature_path = tmp_path / "unit-feature-bytes"
    feature_path.write_bytes(b"UNIT-only unchanged feature bytes")
    frame = pd.DataFrame({"ticker": ["MSFT"], "security_id": ["unit-security"], "decision_id": ["unit-decision"],
        "decision_time_utc": [pd.Timestamp("2021-07-21T22:00:00Z")], "session_date_et": ["2021-07-21"],
        "forecast_eligible": [False], "forecast_abstention_reason": ["UNIT-only absent inputs"]})

    def features(self: Any, *parameters: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        return frame, {"feature_path": str(feature_path), "feature_sha256": file_sha256(feature_path)}

    monkeypatch.setattr(forecast_module, "heavy_job_lease", workspace)
    monkeypatch.setattr(forecast_module, "guard", lambda stage: None)
    monkeypatch.setattr(forecast_module, "assert_memory_budget", lambda **parameters: None)
    monkeypatch.setattr(forecast_module, "load_run", run)
    monkeypatch.setattr(forecast_module, "load_unit", unit)
    monkeypatch.setattr(ReturnForecastService, "_features", features)
    app = api.create_app(service, return_model_service=unit_models, security_config=ApiSecurityConfig(mode="disabled"))
    client = TestClient(app)
    with service.admission.lease(estimated_incremental_gib=service.inference_memory_reservation_gib):
        response = client.post("/v1/research/forecast", json=_payload("/v1/research/forecast"))
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "inference_capacity_exhausted"
        assert response.json()["error"]["retryable"] is True
        assert workspace_calls == artifact_reads == []
    response = client.post("/v1/research/forecast", json=_payload("/v1/research/forecast"))
    assert response.status_code == 200, response.text
    assert workspace_calls == ["return-model-forecast"]
    assert artifact_reads == ["technical_ridge"]
    assert service.admission.snapshot().active_requests == 0
    assert service.admission.snapshot().reserved_gib == 0
