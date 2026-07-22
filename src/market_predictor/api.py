from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from uuid import uuid4

from market_predictor.config import get_settings
from market_predictor.investment_replay import AlpacaReplayPriceProvider, InvestmentReplayService
from market_predictor.prediction_contracts import (
    InvestmentReplayRequest,
    InvestmentReplayResponse,
    PredictionApiError,
    PredictionApiErrorEnvelope,
    PredictionDependencyError,
    PredictionNotFoundError,
    PredictionReadinessError,
    PredictionRequest,
    PredictionResponse,
    PredictionServiceError,
    PredictionValidationError,
)
from market_predictor.prediction_service import PredictionService, serving_routes_from_config
from market_predictor.telemetry import RuntimeTelemetry

try:
    from fastapi import FastAPI, Request
    from fastapi.exceptions import RequestValidationError
    from fastapi.responses import JSONResponse
except ImportError:  # pragma: no cover - exercised only in minimal installs
    FastAPI = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]
    RequestValidationError = None  # type: ignore[misc, assignment]
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
        request.state.correlation_id = _request_correlation_id(request.headers.get("x-correlation-id"))
        try:
            response = await call_next(request)
            status_code = int(response.status_code)
            response.headers["x-correlation-id"] = request.state.correlation_id
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

    @app.exception_handler(PredictionServiceError)
    async def handle_prediction_error(request: Request, exc: PredictionServiceError) -> JSONResponse:
        return _error_response(
            status_code=exc.status_code,
            code=exc.code,
            message=exc.public_message,
            correlation_id=_state_correlation_id(request),
            retryable=exc.retryable,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_request_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        del exc
        return _error_response(
            status_code=422,
            code="request_validation_error",
            message="The request body is invalid.",
            correlation_id=_state_correlation_id(request),
            retryable=False,
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        del exc
        return _error_response(
            status_code=500,
            code="internal_server_error",
            message="The request could not be completed.",
            correlation_id=_state_correlation_id(request),
            retryable=False,
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
    def predict_swing(request: PredictionRequest, http_request: Request) -> PredictionResponse:
        return _run_prediction(
            prediction_service,
            _with_request_context(request, http_request, mode="swing"),
            runtime_telemetry,
        )

    @app.post("/v1/predictions/intraday", response_model=PredictionResponse)
    def predict_intraday(request: PredictionRequest, http_request: Request) -> PredictionResponse:
        return _run_prediction(
            prediction_service,
            _with_request_context(request, http_request, mode="intraday"),
            runtime_telemetry,
        )

    @app.post("/v1/predictions/unified", response_model=PredictionResponse)
    def predict_unified(request: PredictionRequest, http_request: Request) -> PredictionResponse:
        return _run_prediction(
            prediction_service,
            _with_request_context(request, http_request, mode="unified"),
            runtime_telemetry,
        )

    @app.post("/v1/replays/investment", response_model=InvestmentReplayResponse)
    def replay_investment(request: InvestmentReplayRequest) -> InvestmentReplayResponse:
        if configured_replay_service is None:
            raise PredictionReadinessError
        return _run_replay(configured_replay_service, request, runtime_telemetry)

    return app


def _run_prediction(
    service: PredictionService,
    request: PredictionRequest,
    telemetry: RuntimeTelemetry,
) -> PredictionResponse:
    response = service.predict(request)
    telemetry.record_prediction(response)
    return response


def _run_replay(
    service: InvestmentReplayService,
    request: InvestmentReplayRequest,
    telemetry: RuntimeTelemetry,
) -> InvestmentReplayResponse:
    try:
        response = service.replay(request)
        telemetry.record_replay(response)
        return response
    except PredictionServiceError:
        raise
    except FileNotFoundError as exc:
        raise PredictionNotFoundError from exc
    except ValueError as exc:
        raise PredictionValidationError from exc
    except OSError as exc:
        raise PredictionDependencyError from exc


def _with_request_context(request: PredictionRequest, http_request: Request, *, mode: str) -> PredictionRequest:
    correlation_id = request.correlation_id or _state_correlation_id(http_request)
    return request.model_copy(update={"mode": mode, "correlation_id": correlation_id})


def _request_correlation_id(value: str | None) -> str:
    candidate = (value or "").strip()
    if candidate and len(candidate) <= 128 and all(character.isalnum() or character in "._:-" for character in candidate):
        return candidate
    return str(uuid4())


def _state_correlation_id(request: Request) -> str:
    return str(getattr(request.state, "correlation_id", str(uuid4())))


def _error_response(
    *,
    status_code: int,
    code: str,
    message: str,
    correlation_id: str,
    retryable: bool,
) -> JSONResponse:
    envelope = PredictionApiErrorEnvelope(
        error=PredictionApiError(
            code=code,
            message=message,
            correlation_id=correlation_id,
            retryable=retryable,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=envelope.model_dump(mode="json"),
        headers={"x-correlation-id": correlation_id},
    )


app = create_app() if FastAPI is not None else None
