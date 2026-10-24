from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, field_validator

from market_predictor.research_api.service import (
    ResearchFeatureUnavailableError,
    ResearchModelService,
    ResearchModelUnavailableError,
)


class ResearchPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1, max_length=64)
    tickers: Annotated[list[str], Field(min_length=1, max_length=100)]
    session_date: date

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, tickers: list[str]) -> list[str]:
        normalized = list(dict.fromkeys(ticker.strip().upper() for ticker in tickers))
        if any(re.fullmatch(r"[A-Z][A-Z0-9.-]{0,14}", ticker) is None for ticker in normalized):
            raise ValueError("tickers must use canonical US symbol syntax")
        return normalized

def create_research_app(
    *,
    repository_root: Path | str = Path("."),
    catalog_path: Path | None = None,
) -> FastAPI:
    root = Path(repository_root).resolve()
    service = ResearchModelService(
        root,
        catalog_path=catalog_path,
    )
    static_root = Path(__file__).resolve().parent / "static"
    app = FastAPI(
        title="Market Predictor Research Workbench",
        version="0.1.0",
        description=(
            "Inspect six trained return specifications and their exact saved historical "
            "validation predictions. No live scoring or release activation."
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
            "models": service.model_states(),
        }

    @app.post("/v1/research/predict")
    def predict(request: ResearchPredictionRequest) -> dict[str, object]:
        try:
            result = service.predict(
                model_id=request.model_id,
                tickers=request.tickers,
                session_date=request.session_date,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Unknown research model id.") from exc
        except ResearchModelUnavailableError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except ResearchFeatureUnavailableError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {
            "boundary": "research_only",
            "warning": "Saved historical validation predictions; not live forecasts or trading instructions.",
            **result,
        }

    return app


app = create_research_app()
