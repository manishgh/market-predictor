from __future__ import annotations

import unittest
from datetime import UTC, datetime

from fastapi.testclient import TestClient

from market_predictor.api import create_app
from market_predictor.prediction_contracts import (
    InvestmentReplayRequest,
    InvestmentReplayResponse,
    PredictionRequest,
    PredictionResponse,
)


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
        self.assertIn("explicit UTC offset or timezone", response.text)

    def test_prediction_request_rejects_server_owned_artifact_fields(self) -> None:
        client = TestClient(create_app(StubPredictionService()))  # type: ignore[arg-type]

        response = client.post(
            "/v1/predictions/swing",
            json={"tickers": ["MSFT"], "swing_model": "models/research.joblib"},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("extra_forbidden", response.text)

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


if __name__ == "__main__":
    unittest.main()
