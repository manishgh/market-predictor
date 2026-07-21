from __future__ import annotations

from pathlib import Path

from market_predictor.config import get_settings
from market_predictor.investment_replay import AlpacaReplayPriceProvider, InvestmentReplayService
from market_predictor.prediction_contracts import (
    InvestmentReplayRequest,
    InvestmentReplayResponse,
    PredictionRequest,
    PredictionResponse,
)
from market_predictor.prediction_service import PredictionService, serving_routes_from_config

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - exercised only in minimal installs
    FastAPI = None  # type: ignore[misc, assignment]
    HTTPException = None  # type: ignore[misc, assignment]
    JSONResponse = None  # type: ignore[misc, assignment]


def create_app(
    service: PredictionService | None = None,
    replay_service: InvestmentReplayService | None = None,
) -> FastAPI:
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Install the api extras/dependencies before serving.")
    prediction_service = service
    if prediction_service is None:
        settings = get_settings()
        prediction_service = PredictionService(
            Path("."),
            routes=serving_routes_from_config(settings.app_config),
        )
    configured_replay_service = replay_service
    if configured_replay_service is None and isinstance(prediction_service, PredictionService):
        configured_replay_service = InvestmentReplayService(
            snapshot_store=prediction_service.snapshot_store,
            price_provider=AlpacaReplayPriceProvider(get_settings()),
        )
    app = FastAPI(
        title="Market Predictor API",
        version="0.1.0",
        description="Production prediction API for swing and intraday market models.",
    )

    @app.get("/v1/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/health/ready")
    def readiness() -> JSONResponse:
        health_check = getattr(prediction_service, "health", None)
        result = (
            health_check()
            if callable(health_check)
            else {"status": "not_ready", "reason": "prediction readiness is not configured"}
        )
        status_code = 200 if result.get("status") == "ready" else 503
        return JSONResponse(status_code=status_code, content=result)

    @app.post("/v1/predictions/swing", response_model=PredictionResponse)
    def predict_swing(request: PredictionRequest) -> PredictionResponse:
        return _run_prediction(prediction_service, request.model_copy(update={"mode": "swing"}))

    @app.post("/v1/predictions/intraday", response_model=PredictionResponse)
    def predict_intraday(request: PredictionRequest) -> PredictionResponse:
        return _run_prediction(prediction_service, request.model_copy(update={"mode": "intraday"}))

    @app.post("/v1/predictions/unified", response_model=PredictionResponse)
    def predict_unified(request: PredictionRequest) -> PredictionResponse:
        return _run_prediction(prediction_service, request.model_copy(update={"mode": "unified"}))

    @app.post("/v1/replays/investment", response_model=InvestmentReplayResponse)
    def replay_investment(request: InvestmentReplayRequest) -> InvestmentReplayResponse:
        if configured_replay_service is None:
            raise HTTPException(status_code=503, detail="investment replay service is not configured")
        return _run_replay(configured_replay_service, request)

    return app


def _run_prediction(service: PredictionService, request: PredictionRequest) -> PredictionResponse:
    try:
        return service.predict(request)
    except Exception as exc:
        if HTTPException is None:
            raise
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _run_replay(
    service: InvestmentReplayService,
    request: InvestmentReplayRequest,
) -> InvestmentReplayResponse:
    try:
        return service.replay(request)
    except Exception as exc:
        if HTTPException is None:
            raise
        raise HTTPException(status_code=422, detail=str(exc)) from exc


app = create_app() if FastAPI is not None else None
