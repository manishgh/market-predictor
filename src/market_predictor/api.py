from __future__ import annotations

from pathlib import Path

from market_predictor.prediction_contracts import PredictionRequest, PredictionResponse
from market_predictor.prediction_service import PredictionService

try:
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover - exercised only in minimal installs
    FastAPI = None  # type: ignore[assignment]
    HTTPException = None  # type: ignore[assignment]


def create_app(service: PredictionService | None = None) -> "FastAPI":
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Install the api extras/dependencies before serving.")
    prediction_service = service or PredictionService(Path("."))
    app = FastAPI(
        title="Market Predictor API",
        version="0.1.0",
        description="Production prediction API for swing and intraday market models.",
    )

    @app.get("/v1/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/predictions/swing", response_model=PredictionResponse)
    def predict_swing(request: PredictionRequest) -> PredictionResponse:
        return _run_prediction(prediction_service, request.model_copy(update={"mode": "swing"}))

    @app.post("/v1/predictions/intraday", response_model=PredictionResponse)
    def predict_intraday(request: PredictionRequest) -> PredictionResponse:
        return _run_prediction(prediction_service, request.model_copy(update={"mode": "intraday"}))

    @app.post("/v1/predictions/unified", response_model=PredictionResponse)
    def predict_unified(request: PredictionRequest) -> PredictionResponse:
        return _run_prediction(prediction_service, request.model_copy(update={"mode": "unified"}))

    return app


def _run_prediction(service: PredictionService, request: PredictionRequest) -> PredictionResponse:
    try:
        return service.predict(request)
    except Exception as exc:
        if HTTPException is None:
            raise
        raise HTTPException(status_code=422, detail=str(exc)) from exc


app = create_app() if FastAPI is not None else None
