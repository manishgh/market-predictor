"""The CI container smoke release must start and fail closed, as the workflow asserts."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from market_predictor.api import create_app
from market_predictor.config import get_settings
from scripts.build_container_smoke_release import build_smoke_release


def test_smoke_release_config_starts_and_reports_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "container-smoke"
    build_smoke_release(output)
    # The same environment the production-container job passes to the image.
    for name, value in {
        "APP_CONFIG_PATH": str(output / "app_config.toml"),
        "API_ENVIRONMENT": "development",
        "API_AUTH_MODE": "development",
        "API_DEVELOPMENT_BEARER_TOKEN": "s" * 80,
        "API_REPLAY_ENABLED": "false",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.chdir(tmp_path)
    get_settings.cache_clear()
    try:
        with TestClient(create_app()) as client:
            live = client.get("/v1/health/live")
            ready = client.get("/v1/health/ready")
    finally:
        get_settings.cache_clear()

    assert live.status_code == 200, live.text
    assert ready.status_code == 503, ready.text
