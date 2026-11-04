"""Optional workbench UI hosted by the canonical V1 API application."""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from market_predictor.api import create_app
from market_predictor.api_security import ApiSecurityConfig
from market_predictor.research_api.service import ResearchModelService


def create_research_app(*, repository_root: Path | str = Path("."), catalog_path: Path | None = None,
                        security_config: ApiSecurityConfig | None = None) -> FastAPI:
    models = ResearchModelService(Path(repository_root).resolve(), catalog_path=catalog_path)
    app = create_app(return_model_service=models, security_config=security_config, replay_enabled=False)
    static_root = Path(__file__).resolve().parent / "static"
    app.mount("/assets", StaticFiles(directory=static_root), name="research-assets")

    @app.get("/", include_in_schema=False)
    def research_ui() -> FileResponse:
        return FileResponse(static_root / "index.html")

    return app
