"""One V1 route adapter for frozen-model history and fresh research forecasts."""
from __future__ import annotations

import re
from datetime import UTC, date, datetime
from typing import Annotated, Any

from fastapi import Depends, FastAPI
from fastapi.security import HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator

from market_predictor.core.prediction_contracts import (
    PredictionApiErrorEnvelope,
    PredictionConflictError,
    PredictionNotFoundError,
    PredictionReadinessError,
    PredictionValidationError,
)
from market_predictor.research_api.forecast import ReturnForecastService
from market_predictor.research_api.service import (
    ResearchFeatureUnavailableError,
    ResearchModelService,
    ResearchModelUnavailableError,
)

RETURN_DESCRIPTION = (
    "Returns signed ten-session stock net return minus SPY gross return as a fraction: "
    "-0.012 means -1.2 percentage points relative to SPY. These are research estimates, "
    "not probabilities or trading instructions. No serving, promotion or trading admission is granted. "
    "Model artifacts and observed price/news inputs are owned by the server; raw prices, news, "
    "feature values and artifact paths are not accepted in requests."
)
ERROR_RESPONSES: dict[int | str, dict[str, Any]] = {
    code: {"model": PredictionApiErrorEnvelope}
    for code in (401, 403, 404, 409, 413, 422, 429, 503)
}


class ReturnModelQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=64, examples=["technical_ridge"],
        description="One of the six server-owned model ids listed by GET /v1/research/models.")
    tickers: Annotated[list[str], Field(min_length=1, max_length=100, examples=[["MSFT", "AAPL"]],
        description="Canonical US symbols; the configured per-request ticker limit also applies.")]

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, tickers: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(ticker.strip().upper() for ticker in tickers))
        if any(re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", ticker) is None for ticker in normalized):
            raise ValueError("tickers must use canonical US symbol syntax")
        return normalized


class HistoricalReturnRequest(ReturnModelQuery):
    session_date: date = Field(examples=["2021-07-21"],
        description="Exact historical exchange session with retained temporal out-of-fold predictions.")


class ReturnForecastRequest(ReturnModelQuery):
    as_of: datetime | None = Field(default=None,
        description="Optional timezone-aware cutoff; defaults to current UTC. Current observed inputs must cover this decision.")

    @field_validator("as_of")
    @classmethod
    def validate_as_of(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        return value


def register_return_model_routes(app: FastAPI, models: ResearchModelService,
                                 forecasts: ReturnForecastService, *, max_tickers_per_request: int = 100) -> None:
    if max_tickers_per_request < 1:
        raise ValueError("return-model request ticker limit must be positive")
    # Authentication remains in the shared API middleware; this exposes its bearer contract in Swagger.
    auth_docs = [Depends(HTTPBearer(auto_error=False, scheme_name="ApiBearer",
        description="Use the bearer token configured for the canonical V1 API (predictions.read scope)."))]

    def admit(request: ReturnModelQuery) -> None:
        if len(request.tickers) > max_tickers_per_request:
            raise PredictionValidationError

    @app.get("/v1/research/models", summary="List the six trained return specifications",
        description="Inventory and input availability only; no model is promoted or admitted for trading.",
        dependencies=auth_docs, responses=ERROR_RESPONSES)
    def model_catalog() -> dict[str, object]:
        states = models.model_states()
        for state in states:
            state["fresh_forecast_configured"] = forecasts.publication_configured()
            state["fresh_inputs_published"] = forecasts.inputs_published()
        return {"boundary": "research_only", "actionable": False, "models": states}

    @app.post("/v1/research/predict", summary="Read exact saved historical return predictions",
        description="Reads the requested session's saved temporal out-of-fold predictions without refitting. " + RETURN_DESCRIPTION,
        dependencies=auth_docs, responses=ERROR_RESPONSES)
    def history(request: HistoricalReturnRequest) -> dict[str, Any]:
        admit(request)
        return _query(models.predict, model_id=request.model_id, tickers=request.tickers,
                      session_date=request.session_date)

    @app.post("/v1/research/forecast", summary="Calculate a fresh return estimate",
        description="Uses the frozen final model and a current verified observed feature publication. " + RETURN_DESCRIPTION,
        dependencies=auth_docs, responses=ERROR_RESPONSES)
    def forecast(request: ReturnForecastRequest) -> dict[str, Any]:
        admit(request)
        return _query(forecasts.forecast, model_id=request.model_id, tickers=request.tickers,
                      as_of=request.as_of or datetime.now(UTC))


def _query(method: Any, **parameters: Any) -> dict[str, Any]:
    try:
        result: dict[str, Any] = method(**parameters)
    except KeyError as exc:
        raise PredictionNotFoundError from exc
    except ResearchModelUnavailableError as exc:
        raise PredictionConflictError from exc
    except ResearchFeatureUnavailableError as exc:
        raise PredictionReadinessError from exc
    return {"boundary": "research_only", "warning": "Research return estimates; release acceptance is pending.", **result}
