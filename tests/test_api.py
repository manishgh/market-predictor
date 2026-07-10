from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from market_predictor.api import create_app
from market_predictor.prediction_contracts import PredictionRequest, PredictionResponse


class StubPredictionService:
    def __init__(self) -> None:
        self.last_request: PredictionRequest | None = None

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        self.last_request = request
        return PredictionResponse(mode=request.mode, horizon=request.horizon, predictions=[])


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


if __name__ == "__main__":
    unittest.main()
