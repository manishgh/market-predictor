from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import market_predictor.api
from market_predictor.research_api.server import create_research_app
from market_predictor.research_api.service import (
    ResearchFeatureUnavailableError,
    ResearchModelService,
)


class _LinearEstimator:
    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return frame["feature_a"].to_numpy(dtype="float64") * 2.0


class _FeatureSource:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def load(self, mode: str, *, as_of: datetime) -> pd.DataFrame:
        assert mode == "swing"
        assert as_of.utcoffset() is not None
        return self.frame.copy()


def test_canonical_api_module_is_not_shadowed() -> None:
    assert Path(market_predictor.api.__file__).name == "api.py"


def test_catalog_reports_all_four_models_and_real_artifact_states(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    _write_candidate(tmp_path / "data/models/swing_baseline")
    for relative in (
        "data/models/swing_event",
        "data/models/intraday_event",
        "data/models/intraday_baseline",
    ):
        _write_no_candidate(tmp_path / relative)

    service = ResearchModelService(
        tmp_path,
        catalog_path=catalog,
        feature_source=_FeatureSource(pd.DataFrame()),
    )

    states = {state.spec.model_id: state for state in service.model_states()}
    assert set(states) == {
        "swing_event_driven",
        "swing_baseline",
        "intraday_event_driven",
        "intraday_baseline",
    }
    assert states["swing_baseline"].research_scoring_available is True
    assert states["swing_baseline"].promotion_permitted is False
    assert states["intraday_baseline"].training_status == "no_candidate"
    assert states["intraday_baseline"].research_scoring_available is False


def test_scoring_preserves_real_feature_values_and_remains_non_actionable(
    tmp_path: Path,
) -> None:
    catalog = _write_catalog(tmp_path)
    _write_candidate(tmp_path / "data/models/swing_baseline")
    for relative in (
        "data/models/swing_event",
        "data/models/intraday_event",
        "data/models/intraday_baseline",
    ):
        _write_no_candidate(tmp_path / relative)
    features = pd.DataFrame(
        {
            "ticker": ["MSFT"],
            "feature_available_at_utc": ["2026-08-10T20:00:00Z"],
            "feature_a": [1.25],
        }
    )
    service = ResearchModelService(
        tmp_path,
        catalog_path=catalog,
        feature_source=_FeatureSource(features),
    )

    result = service.predict(
        model_id="swing_baseline",
        tickers=["MSFT", "NVDA"],
        as_of=datetime(2026, 8, 10, 21, tzinfo=UTC),
    )

    assert result["actionable"] is False
    assert result["predictions"][0]["score"] == pytest.approx(2.5)
    assert result["predictions"][0]["features"] == {"feature_a": 1.25}
    assert result["rejected"] == [
        {
            "ticker": "NVDA",
            "reason": "No current registered feature row is available.",
        }
    ]


def test_scoring_rejects_missing_features_instead_of_zero_filling(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    _write_candidate(tmp_path / "data/models/swing_baseline")
    for relative in (
        "data/models/swing_event",
        "data/models/intraday_event",
        "data/models/intraday_baseline",
    ):
        _write_no_candidate(tmp_path / relative)
    service = ResearchModelService(
        tmp_path,
        catalog_path=catalog,
        feature_source=_FeatureSource(pd.DataFrame({"ticker": ["MSFT"]})),
    )

    with pytest.raises(ResearchFeatureUnavailableError, match="missing columns"):
        service.predict(
            model_id="swing_baseline",
            tickers=["MSFT"],
            as_of=datetime(2026, 8, 10, 21, tzinfo=UTC),
        )


def test_http_catalog_and_unavailable_model_are_explicit(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    _write_candidate(tmp_path / "data/models/swing_baseline")
    for relative in (
        "data/models/swing_event",
        "data/models/intraday_event",
        "data/models/intraday_baseline",
    ):
        _write_no_candidate(tmp_path / relative)
    app = create_research_app(
        repository_root=tmp_path,
        catalog_path=catalog,
        feature_source=_FeatureSource(pd.DataFrame()),
    )
    client = TestClient(app)

    catalog_response = client.get("/v1/research/models")
    assert catalog_response.status_code == 200
    assert len(catalog_response.json()["models"]) == 4
    assert catalog_response.json()["actionable"] is False

    prediction_response = client.post(
        "/v1/research/predict",
        json={"model_id": "intraday_baseline", "tickers": ["MSFT"]},
    )
    assert prediction_response.status_code == 409
    assert "no candidate passed" in prediction_response.json()["detail"]


def test_ui_has_no_hardcoded_ticker_values_or_mock_feature_language() -> None:
    static_root = (
        Path(__file__).parents[1]
        / "src/market_predictor/research_api/static"
    )
    html = (static_root / "index.html").read_text(encoding="utf-8")
    script = (static_root / "app.js").read_text(encoding="utf-8")

    assert "AAPL" not in html
    assert "TSLA" not in html
    assert "value=\"AAPL" not in html
    assert "features_summary || 'N/A'" not in script
    assert "calculated features" in script


def _write_catalog(root: Path) -> Path:
    path = root / "catalog.toml"
    path.write_text(
        """
schema = "market_predictor.research_model_catalog.v1"

[[models]]
id = "swing_event_driven"
label = "Swing - Event Driven"
mode = "swing"
uses_catalyst = true
artifact_directory = "data/models/swing_event"

[[models]]
id = "swing_baseline"
label = "Swing - Technical Baseline"
mode = "swing"
uses_catalyst = false
artifact_directory = "data/models/swing_baseline"

[[models]]
id = "intraday_event_driven"
label = "Intraday - Event Driven"
mode = "intraday"
uses_catalyst = true
artifact_directory = "data/models/intraday_event"

[[models]]
id = "intraday_baseline"
label = "Intraday - Technical Baseline"
mode = "intraday"
uses_catalyst = false
artifact_directory = "data/models/intraday_baseline"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_candidate(directory: Path) -> None:
    directory.mkdir(parents=True)
    candidate = directory / "candidate.joblib"
    joblib.dump(
        {
            "feature_columns": ["feature_a"],
            "fitted_candidate": _LinearEstimator(),
        },
        candidate,
    )
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    (directory / "model_card.json").write_text(
        json.dumps(
            {
                "status": "candidate",
                "promotion_permitted": False,
                "candidate_id": "fixture-candidate",
            }
        ),
        encoding="utf-8",
    )
    (directory / "_manifest.json").write_text(
        json.dumps(
            {
                "state": "candidate",
                "promotion_permitted": False,
                "files": {"candidate.joblib": {"sha256": digest}},
            }
        ),
        encoding="utf-8",
    )


def _write_no_candidate(directory: Path) -> None:
    directory.mkdir(parents=True)
    (directory / "model_card.json").write_text(
        json.dumps(
            {
                "status": "no_candidate",
                "promotion_permitted": False,
                "candidate_id": None,
            }
        ),
        encoding="utf-8",
    )
    (directory / "_manifest.json").write_text(
        json.dumps({"state": "no_candidate", "promotion_permitted": False}),
        encoding="utf-8",
    )
