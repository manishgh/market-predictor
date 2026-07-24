from __future__ import annotations

import unittest
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from market_predictor.api import create_app
from market_predictor.prediction_contracts import (
    InvestmentReplayRequest,
    InvestmentReplayResponse,
    PredictionCapacityError,
    PredictionConflictError,
    PredictionDependencyError,
    PredictionMemoryPressureError,
    PredictionNotFoundError,
    PredictionReadinessError,
    PredictionRequest,
    PredictionResponse,
    PredictionServiceError,
    PredictionThrottledError,
    PredictionValidationError,
)
from market_predictor.telemetry import RuntimeTelemetry


class StubPredictionService:
    def __init__(self) -> None:
        self.last_request: PredictionRequest | None = None

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        self.last_request = request
        return PredictionResponse(mode=request.mode, horizon=request.horizon, predictions=[])

    def health(self) -> dict[str, object]:
        return {"status": "ready", "components": {}}


class NotReadyPredictionService(StubPredictionService):
    def health(self) -> dict[str, object]:
        return {"status": "not_ready", "components": {"swing": {"status": "not_ready"}}}


class FailingPredictionService(StubPredictionService):
    def __init__(self, error: Exception) -> None:
        super().__init__()
        self.error = error

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        del request
        raise self.error


class StubReplayService:
    def replay(self, request: InvestmentReplayRequest) -> InvestmentReplayResponse:
        now = datetime.now(UTC)
        return InvestmentReplayResponse(
            snapshot_id=request.snapshot_id,
            ticker=request.ticker,
            model_view=request.model_view,
            decision_time=now,
            evaluation_time=now,
            prediction_signal="neutral",
            status="not_entered",
        )


class PredictionApiTests(unittest.TestCase):
    def test_swing_endpoint_forces_swing_mode(self) -> None:
        service = StubPredictionService()
        client = TestClient(create_app(service))  # type: ignore[arg-type]

        response = client.post("/v1/predictions/swing", json={"tickers": ["msft"], "mode": "unified"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["mode"], "swing")
        self.assertIsNotNone(service.last_request)
        assert service.last_request is not None
        self.assertEqual(service.last_request.mode, "swing")
        self.assertEqual(service.last_request.tickers, ["MSFT"])

    def test_prediction_request_rejects_timezone_free_as_of(self) -> None:
        client = TestClient(create_app(StubPredictionService()))  # type: ignore[arg-type]

        response = client.post(
            "/v1/predictions/swing",
            json={"tickers": ["MSFT"], "as_of": "2026-07-09T15:55:00"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "request_validation_error")
        self.assertNotIn("explicit UTC offset or timezone", response.text)

    def test_prediction_request_rejects_server_owned_artifact_fields(self) -> None:
        client = TestClient(create_app(StubPredictionService()))  # type: ignore[arg-type]

        response = client.post(
            "/v1/predictions/swing",
            json={"tickers": ["MSFT"], "swing_model": "models/research.joblib"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["error"]["code"], "request_validation_error")
        self.assertNotIn("extra_forbidden", response.text)

    def test_correlation_identity_is_propagated(self) -> None:
        service = StubPredictionService()
        client = TestClient(create_app(service))  # type: ignore[arg-type]

        response = client.post(
            "/v1/predictions/swing",
            json={"tickers": ["MSFT"]},
            headers={"x-correlation-id": "trading-flow:request-42"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-correlation-id"], "trading-flow:request-42")
        assert service.last_request is not None
        self.assertEqual(service.last_request.correlation_id, "trading-flow:request-42")

    def test_typed_prediction_errors_use_stable_status_and_opaque_envelope(self) -> None:
        cases: list[tuple[type[PredictionServiceError], int, str]] = [
            (PredictionValidationError, 422, "prediction_validation_error"),
            (PredictionNotFoundError, 404, "prediction_not_found"),
            (PredictionConflictError, 409, "prediction_conflict"),
            (PredictionThrottledError, 429, "prediction_throttled"),
            (PredictionCapacityError, 503, "inference_capacity_exhausted"),
            (PredictionMemoryPressureError, 503, "memory_pressure"),
            (PredictionReadinessError, 503, "prediction_not_ready"),
            (PredictionDependencyError, 503, "prediction_dependency_unavailable"),
        ]
        for error_type, status_code, code in cases:
            with self.subTest(error=error_type.__name__):
                client = TestClient(create_app(FailingPredictionService(error_type())))  # type: ignore[arg-type]
                response = client.post(
                    "/v1/predictions/swing",
                    json={"tickers": ["MSFT"]},
                    headers={"x-correlation-id": "test-correlation"},
                )
                self.assertEqual(response.status_code, status_code)
                self.assertEqual(response.json()["error"]["code"], code)
                self.assertEqual(response.json()["error"]["correlation_id"], "test-correlation")
                self.assertNotIn("detail", response.json())

    def test_unexpected_error_is_opaque(self) -> None:
        secret_text = r"C:\private\models\candidate.joblib api_key=should-not-leak"
        client = TestClient(  # type: ignore[arg-type]
            create_app(FailingPredictionService(RuntimeError(secret_text))),
            raise_server_exceptions=False,
        )

        response = client.post("/v1/predictions/swing", json={"tickers": ["MSFT"]})

        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["error"]["code"], "internal_server_error")
        self.assertNotIn("candidate.joblib", response.text)
        self.assertNotIn("should-not-leak", response.text)

    def test_health_separates_liveness_from_readiness(self) -> None:
        live_client = TestClient(create_app(NotReadyPredictionService()))  # type: ignore[arg-type]

        liveness = live_client.get("/v1/health/live")
        readiness = live_client.get("/v1/health/ready")

        self.assertEqual(liveness.status_code, 200)
        self.assertEqual(liveness.json(), {"status": "ok"})
        self.assertEqual(readiness.status_code, 503)
        self.assertEqual(readiness.json()["status"], "not_ready")

    def test_investment_replay_endpoint_uses_configured_service(self) -> None:
        client = TestClient(  # type: ignore[arg-type]
            create_app(StubPredictionService(), replay_service=StubReplayService())
        )

        response = client.post(
            "/v1/replays/investment",
            json={"snapshot_id": "a" * 64, "ticker": "msft"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["ticker"], "MSFT")
        self.assertEqual(response.json()["status"], "not_entered")

    def test_metrics_report_bounded_request_prediction_and_outcome_counters(self) -> None:
        telemetry = RuntimeTelemetry()
        client = TestClient(  # type: ignore[arg-type]
            create_app(
                StubPredictionService(),
                replay_service=StubReplayService(),
                telemetry=telemetry,
            )
        )

        client.post("/v1/predictions/swing", json={"tickers": ["MSFT"]})
        client.post(
            "/v1/replays/investment",
            json={"snapshot_id": "a" * 64, "ticker": "MSFT"},
        )
        response = client.get("/v1/metrics")

        self.assertEqual(response.status_code, 200)
        metrics = response.json()
        self.assertEqual(metrics["schema"], "market_predictor.runtime_metrics.v1")
        self.assertEqual(metrics["requests"]["POST /v1/predictions/swing"]["count"], 1)
        self.assertEqual(metrics["predictions"]["swing"]["requests"], 1)
        self.assertEqual(metrics["prediction_outcomes"]["not_entered"], 1)
        self.assertLess(float(metrics["memory"]["current_working_set_gib"]), 4.0)


if __name__ == "__main__":
    unittest.main()
