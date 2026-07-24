from __future__ import annotations

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, cast
from uuid import uuid4

from market_predictor.api_security import (
    API_SCOPES,
    ApiAuthenticator,
    ApiPrincipal,
    ApiSecurityConfig,
    ApiSecurityError,
    PrincipalRateLimiter,
)
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
    *,
    security_config: ApiSecurityConfig | None = None,
    rate_limiter: PrincipalRateLimiter | None = None,
    maximum_body_bytes: int | None = None,
    replay_enabled: bool | None = None,
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
            max_concurrent_inference=settings.runtime_max_concurrent_inference,
            max_tickers_per_request=settings.runtime_max_tickers_per_request,
            inference_memory_reservation_gib=settings.runtime_inference_memory_reservation_gib,
            reject_unknown_memory=settings.runtime_reject_unknown_memory,
        )
    configured_security = security_config
    if configured_security is None:
        configured_security = (
            _security_config_from_settings(settings)
            if settings is not None
            else ApiSecurityConfig(mode="disabled")
        )
    authenticator = ApiAuthenticator(configured_security)
    body_limit = maximum_body_bytes or (
        settings.api_maximum_body_bytes if settings is not None else 65_536
    )
    if body_limit < 1_024:
        raise ValueError("API body limit must be at least 1024 bytes")
    limiter = rate_limiter or _rate_limiter_from_settings(settings)
    configured_replay_service = replay_service
    if configured_replay_service is None and isinstance(prediction_service, PredictionService):
        configured_replay_service = InvestmentReplayService(
            snapshot_store=prediction_service.snapshot_store,
            price_provider=AlpacaReplayPriceProvider(get_settings()),
        )
    allow_replay = (
        replay_enabled
        if replay_enabled is not None
        else (
            settings.api_replay_enabled
            if settings is not None
            else configured_replay_service is not None
        )
    )
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        preload = getattr(prediction_service, "preload", None)
        if callable(preload):
            preload()
        yield

    app = FastAPI(
        title="Market Predictor API",
        version="0.1.0",
        description="Production prediction API for swing and intraday market models.",
        lifespan=lifespan,
        docs_url=None if configured_security.mode == "entra" else "/docs",
        redoc_url=None,
        openapi_url=None if configured_security.mode == "entra" else "/openapi.json",
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
    async def enforce_boundary_and_observe(request: Any, call_next: Any) -> Any:
        started = time.perf_counter()
        status_code = 500
        correlation_id = _request_correlation_id(
            request.headers.get("x-correlation-id")
        )
        request.state.correlation_id = correlation_id
        principal: ApiPrincipal | None = None
        required_scope = _required_scope(str(request.method), str(request.url.path))
        try:
            if str(request.method).upper() in {"POST", "PUT", "PATCH"}:
                content_length = request.headers.get("content-length")
                if content_length is not None:
                    try:
                        declared_length = int(content_length)
                    except ValueError:
                        status_code = 400
                        return _boundary_error(
                            status_code=400,
                            code="invalid_content_length",
                            message="The request content length is invalid.",
                            correlation_id=correlation_id,
                        )
                    if declared_length < 0 or declared_length > body_limit:
                        status_code = 413
                        return _boundary_error(
                            status_code=413,
                            code="request_body_too_large",
                            message="The request body exceeds the configured limit.",
                            correlation_id=correlation_id,
                        )
                body = await request.body()
                if len(body) > body_limit:
                    status_code = 413
                    return _boundary_error(
                        status_code=413,
                        code="request_body_too_large",
                        message="The request body exceeds the configured limit.",
                        correlation_id=correlation_id,
                    )
            if (
                str(request.url.path) == "/v1/replays/investment"
                and not allow_replay
            ):
                status_code = 404
                return _boundary_error(
                    status_code=404,
                    code="resource_not_found",
                    message="The requested resource was not found.",
                    correlation_id=correlation_id,
                )
            if required_scope is not None:
                principal = authenticator.authenticate(
                    request.headers.get("authorization"),
                    required_scope=required_scope,
                )
                limiter.acquire(principal.principal_id, required_scope)
                request.state.api_principal = principal
            response = await call_next(request)
            status_code = int(response.status_code)
            response.headers["x-correlation-id"] = correlation_id
            return response
        except ApiSecurityError as exc:
            status_code = exc.status_code
            return _boundary_error(
                status_code=exc.status_code,
                code=exc.code,
                message=exc.message,
                correlation_id=correlation_id,
                retry_after_seconds=exc.retry_after_seconds,
            )
        finally:
            route = request.scope.get("route")
            route_path = getattr(route, "path", "__unmatched__")
            runtime_telemetry.record_request(
                method=str(request.method),
                path=str(route_path),
                status_code=status_code,
                elapsed_ms=(time.perf_counter() - started) * 1_000.0,
                principal_id=principal.principal_id if principal is not None else None,
                correlation_id=correlation_id,
                required_scope=required_scope,
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
        return JSONResponse(
            status_code=status_code,
            content={
                "status": result.get("status", "not_ready"),
                "checked_at_utc": result.get("checked_at_utc"),
            },
        )

    @app.get("/v1/operations/health")
    def operational_health() -> JSONResponse:
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
            http_request,
        )

    @app.post("/v1/predictions/intraday", response_model=PredictionResponse)
    def predict_intraday(request: PredictionRequest, http_request: Request) -> PredictionResponse:
        return _run_prediction(
            prediction_service,
            _with_request_context(request, http_request, mode="intraday"),
            runtime_telemetry,
            http_request,
        )

    @app.post("/v1/predictions/unified", response_model=PredictionResponse)
    def predict_unified(request: PredictionRequest, http_request: Request) -> PredictionResponse:
        return _run_prediction(
            prediction_service,
            _with_request_context(request, http_request, mode="unified"),
            runtime_telemetry,
            http_request,
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
    http_request: Request,
) -> PredictionResponse:
    response = service.predict(request)
    principal = _request_principal(http_request)
    admission = (
        service.admission.snapshot().to_record()
        if isinstance(service, PredictionService)
        else None
    )
    telemetry.record_prediction(
        response,
        principal_id=principal.principal_id if principal is not None else None,
        correlation_id=_state_correlation_id(http_request),
        admission=admission,
    )
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
    headers: dict[str, str] | None = None,
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
        headers={"x-correlation-id": correlation_id, **(headers or {})},
    )


def _boundary_error(
    *,
    status_code: int,
    code: str,
    message: str,
    correlation_id: str,
    retry_after_seconds: int | None = None,
) -> JSONResponse:
    headers: dict[str, str] = {}
    if status_code == 401:
        headers["www-authenticate"] = "Bearer"
    if retry_after_seconds is not None:
        headers["retry-after"] = str(retry_after_seconds)
    return _error_response(
        status_code=status_code,
        code=code,
        message=message,
        correlation_id=correlation_id,
        retryable=status_code in {429, 503},
        headers=headers,
    )


def _required_scope(method: str, path: str) -> str | None:
    return {
        ("POST", "/v1/predictions/swing"): "predictions.read",
        ("POST", "/v1/predictions/intraday"): "predictions.read",
        ("POST", "/v1/predictions/unified"): "predictions.read",
        ("GET", "/v1/operations/health"): "operations.read",
        ("GET", "/v1/metrics"): "metrics.read",
        ("POST", "/v1/replays/investment"): "replay.execute",
    }.get((method.upper(), path))


def _request_principal(request: Request) -> ApiPrincipal | None:
    principal = getattr(request.state, "api_principal", None)
    return principal if isinstance(principal, ApiPrincipal) else None


def _security_config_from_settings(settings: Any) -> ApiSecurityConfig:
    environment = str(settings.api_environment).strip().lower()
    mode = str(settings.api_auth_mode).strip().lower()
    if environment not in {"development", "production"}:
        raise RuntimeError("API_ENVIRONMENT must be development or production")
    if environment == "production" and mode != "entra":
        raise RuntimeError("production API requires Entra authentication")
    if environment == "development" and mode not in {"development", "entra"}:
        raise RuntimeError("development API auth mode must be development or entra")
    return ApiSecurityConfig(
        mode=cast(Literal["development", "entra"], mode),
        issuer=settings.api_jwt_issuer,
        audience=settings.api_jwt_audience,
        jwks_path=settings.api_jwks_path,
        development_token=settings.api_development_bearer_token,
    )


def _rate_limiter_from_settings(settings: Any | None) -> PrincipalRateLimiter:
    rates = {
        "predictions.read": (
            settings.api_prediction_requests_per_minute if settings is not None else 60
        ),
        "operations.read": (
            settings.api_operations_requests_per_minute if settings is not None else 30
        ),
        "metrics.read": (
            settings.api_metrics_requests_per_minute if settings is not None else 30
        ),
        "replay.execute": (
            settings.api_replay_requests_per_minute if settings is not None else 5
        ),
    }
    if set(rates) != API_SCOPES:
        raise RuntimeError("API rate-limit scope configuration is incomplete")
    return PrincipalRateLimiter(
        requests_per_minute=rates,
        maximum_principals=(
            settings.api_maximum_rate_limit_principals
            if settings is not None
            else 10_000
        ),
    )
