"""Unit check: the real application refuses readiness without deployment artifacts."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from market_predictor.api import create_app
from market_predictor.config import get_settings


def test_unprovisioned_config_starts_and_reports_not_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = Path(__file__).resolve().parents[1]
    shutil.copytree(repository / "configs", tmp_path / "configs")
    # Use the same checked-in configuration as the container; create no model,
    # market inputs, fabricated metrics, signing keys or promotion evidence.
    for name, value in {
        "APP_CONFIG_PATH": str(tmp_path / "configs/default.toml"),
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
    assert not (tmp_path / "models").exists()
