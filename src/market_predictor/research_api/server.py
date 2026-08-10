from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from market_predictor.research_api.service import (
    ResearchFeatureSource,
    ResearchFeatureUnavailableError,
    ResearchModelService,
    ResearchModelUnavailableError,
    model_state_record,
)


class ResearchPredictionRequest(BaseModel):
    model_id: str = Field(min_length=1, max_length=64)
    tickers: Annotated[list[str], Field(min_length=1, max_length=100)]
    as_of: datetime | None = None

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, tickers: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(ticker.strip().upper() for ticker in tickers))
        if any(re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", ticker) is None for ticker in normalized):
            raise ValueError("tickers must use canonical US symbol syntax")
        return normalized

    @field_validator("as_of")
    @classmethod
    def validate_as_of(cls, value: datetime | None) -> datetime | None:
        if value is not None and value.utcoffset() is None:
            raise ValueError("as_of must be timezone-aware")
        return value


def create_research_app(
    *,
    repository_root: Path | str = Path("."),
    catalog_path: Path | None = None,
    feature_source: ResearchFeatureSource | None = None,
) -> FastAPI:
    root = Path(repository_root).resolve()
    service = ResearchModelService(
        root,
        catalog_path=catalog_path,
        feature_source=feature_source,
    )
    static_root = Path(__file__).resolve().parent / "static"
    app = FastAPI(
        title="Market Predictor Research Workbench",
        version="0.1.0",
        description=(
            "Non-actionable model inspection and research scoring over registered, "
            "causal feature snapshots."
        ),
    )
    app.mount("/assets", StaticFiles(directory=static_root), name="research-assets")

    @app.get("/", include_in_schema=False)
    def research_ui() -> FileResponse:
        return FileResponse(static_root / "index.html")

    @app.get("/v1/health/live")
    def liveness() -> dict[str, str]:
        return {"status": "ok", "boundary": "research_only"}

    @app.get("/v1/research/models")
    def models() -> dict[str, object]:
        return {
            "boundary": "research_only",
            "actionable": False,
            "models": [model_state_record(state) for state in service.model_states()],
        }

    @app.post("/v1/research/predict")
    def predict(request: ResearchPredictionRequest) -> dict[str, object]:
        try:
            result = service.predict(
                model_id=request.model_id,
                tickers=request.tickers,
                as_of=request.as_of or datetime.now(UTC),
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown research model id.") from exc
        except ResearchModelUnavailableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ResearchFeatureUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "boundary": "research_only",
            "warning": "Scores are research evidence and are not trading instructions.",
            **result,
        }

    return app


app = create_research_app()
