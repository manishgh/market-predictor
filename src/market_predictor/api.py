from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from market_predictor.config import get_settings
from market_predictor.investment_replay import AlpacaReplayPriceProvider, InvestmentReplayService
from market_predictor.prediction_contracts import (
    InvestmentReplayRequest,
    InvestmentReplayResponse,
    PredictionRequest,
    PredictionResponse,
)
from market_predictor.prediction_service import PredictionService, serving_routes_from_config
from market_predictor.telemetry import RuntimeTelemetry

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
    telemetry: RuntimeTelemetry | None = None,
) -> FastAPI:
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed. Install the api extras/dependencies before serving.")
    prediction_service = service
    settings = None
    if prediction_service is None:
        settings = get_settings()
        prediction_service = PredictionService(
            Path("."),
            routes=serving_routes_from_config(settings.app_config),
            memory_budget_gib=settings.runtime_memory_budget_gib,
            memory_headroom_gib=settings.runtime_memory_headroom_gib,
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
    if telemetry is not None:
        runtime_telemetry = telemetry
    elif settings is not None:
        runtime_telemetry = RuntimeTelemetry(
            memory_budget_gib=settings.runtime_memory_budget_gib,
            memory_headroom_gib=settings.runtime_memory_headroom_gib,
        )
    else:
        runtime_telemetry = RuntimeTelemetry()

    @app.middleware("http")
    async def observe_request(request: Any, call_next: Any) -> Any:
        started = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = int(response.status_code)
            return response
        finally:
            route = request.scope.get("route")
            route_path = getattr(route, "path", "__unmatched__")
            runtime_telemetry.record_request(
                method=str(request.method),
                path=str(route_path),
                status_code=status_code,
                elapsed_ms=(time.perf_counter() - started) * 1_000.0,
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
        runtime_telemetry.record_health(result)
        return JSONResponse(status_code=status_code, content=result)

    @app.get("/v1/metrics")
    def metrics() -> dict[str, object]:
        return runtime_telemetry.snapshot()

    @app.post("/v1/predictions/swing", response_model=PredictionResponse)
    def predict_swing(request: PredictionRequest) -> PredictionResponse:
        return _run_prediction(
            prediction_service,
            request.model_copy(update={"mode": "swing"}),
            runtime_telemetry,
        )

    @app.post("/v1/predictions/intraday", response_model=PredictionResponse)
    def predict_intraday(request: PredictionRequest) -> PredictionResponse:
        return _run_prediction(
            prediction_service,
            request.model_copy(update={"mode": "intraday"}),
            runtime_telemetry,
        )

    @app.post("/v1/predictions/unified", response_model=PredictionResponse)
    def predict_unified(request: PredictionRequest) -> PredictionResponse:
        return _run_prediction(
            prediction_service,
            request.model_copy(update={"mode": "unified"}),
            runtime_telemetry,
        )

    @app.post("/v1/replays/investment", response_model=InvestmentReplayResponse)
    def replay_investment(request: InvestmentReplayRequest) -> InvestmentReplayResponse:
        if configured_replay_service is None:
            raise HTTPException(status_code=503, detail="investment replay service is not configured")
        return _run_replay(configured_replay_service, request, runtime_telemetry)

    return app


def _run_prediction(
    service: PredictionService,
    request: PredictionRequest,
    telemetry: RuntimeTelemetry,
) -> PredictionResponse:
    try:
        response = service.predict(request)
        telemetry.record_prediction(response)
        return response
    except Exception as exc:
        if HTTPException is None:
            raise
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _run_replay(
    service: InvestmentReplayService,
    request: InvestmentReplayRequest,
    telemetry: RuntimeTelemetry,
) -> InvestmentReplayResponse:
    try:
        response = service.replay(request)
        telemetry.record_replay(response)
        return response
    except Exception as exc:
        if HTTPException is None:
            raise
        raise HTTPException(status_code=422, detail=str(exc)) from exc


app = create_app() if FastAPI is not None else None
