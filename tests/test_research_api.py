from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

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
        return cast(
            np.ndarray,
            frame["feature_a"].to_numpy(dtype="float64") * 2.0,
        )


class _FeatureBoundModel:
    def __init__(self, feature_columns: tuple[str, ...]) -> None:
        self.feature_columns = feature_columns
        self.estimator = _SubsetEstimator(feature_columns)


class _SubsetEstimator:
    def __init__(self, expected_columns: tuple[str, ...]) -> None:
        self.expected_columns = expected_columns

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        assert tuple(frame.columns) == self.expected_columns
        return cast(np.ndarray, frame.iloc[:, 0].to_numpy(dtype="float64"))


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
    _write_candidate(tmp_path / "data/models/swing/technical")
    for relative in (
        "data/models/swing/technical_with_catalyst",
        "data/models/intraday/technical_with_catalyst",
        "data/models/intraday/technical",
    ):
        _write_no_candidate(tmp_path / relative)

    service = ResearchModelService(
        tmp_path,
        catalog_path=catalog,
        feature_source=_FeatureSource(pd.DataFrame()),
    )

    states = {state.spec.model_id: state for state in service.model_states()}
    assert set(states) == {
        "swing_technical_with_catalyst",
        "swing_technical",
        "intraday_technical_with_catalyst",
        "intraday_technical",
    }
    assert states["swing_technical"].research_scoring_available is True
    assert states["swing_technical"].promotion_permitted is False
    assert states["intraday_technical"].training_status == "no_candidate"
    assert states["intraday_technical"].research_scoring_available is False


def test_scoring_preserves_real_feature_values_and_remains_non_actionable(
    tmp_path: Path,
) -> None:
    catalog = _write_catalog(tmp_path)
    _write_candidate(tmp_path / "data/models/swing/technical")
    for relative in (
        "data/models/swing/technical_with_catalyst",
        "data/models/intraday/technical_with_catalyst",
        "data/models/intraday/technical",
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
        model_id="swing_technical",
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
    _write_candidate(tmp_path / "data/models/swing/technical")
    for relative in (
        "data/models/swing/technical_with_catalyst",
        "data/models/intraday/technical_with_catalyst",
        "data/models/intraday/technical",
    ):
        _write_no_candidate(tmp_path / relative)
    service = ResearchModelService(
        tmp_path,
        catalog_path=catalog,
        feature_source=_FeatureSource(pd.DataFrame({"ticker": ["MSFT"]})),
    )

    with pytest.raises(ResearchFeatureUnavailableError, match="missing columns"):
        service.predict(
            model_id="swing_technical",
            tickers=["MSFT"],
            as_of=datetime(2026, 8, 10, 21, tzinfo=UTC),
        )


def test_scoring_uses_fitted_model_feature_subset_in_declared_order(
    tmp_path: Path,
) -> None:
    catalog = _write_catalog(tmp_path)
    _write_candidate(
        tmp_path / "data/models/swing/technical",
        feature_columns=("feature_a", "feature_b"),
        fitted_candidate=_FeatureBoundModel(("feature_b",)),
    )
    for relative in (
        "data/models/swing/technical_with_catalyst",
        "data/models/intraday/technical_with_catalyst",
        "data/models/intraday/technical",
    ):
        _write_no_candidate(tmp_path / relative)
    service = ResearchModelService(
        tmp_path,
        catalog_path=catalog,
        feature_source=_FeatureSource(
            pd.DataFrame(
                {
                    "ticker": ["MSFT"],
                    "feature_available_at_utc": ["2026-08-10T20:00:00Z"],
                    "feature_a": [1.25],
                    "feature_b": [3.5],
                }
            )
        ),
    )

    result = service.predict(
        model_id="swing_technical",
        tickers=["MSFT"],
        as_of=datetime(2026, 8, 10, 21, tzinfo=UTC),
    )

    assert result["predictions"][0]["score"] == pytest.approx(3.5)


def test_http_catalog_and_unavailable_model_are_explicit(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path)
    _write_candidate(tmp_path / "data/models/swing/technical")
    for relative in (
        "data/models/swing/technical_with_catalyst",
        "data/models/intraday/technical_with_catalyst",
        "data/models/intraday/technical",
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
        json={"model_id": "intraday_technical", "tickers": ["MSFT"]},
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
id = "swing_technical_with_catalyst"
label = "Swing - Technical and Catalyst"
mode = "swing"
uses_catalyst = true
artifact_directory = "data/models/swing/technical_with_catalyst"

[[models]]
id = "swing_technical"
label = "Swing - Technical"
mode = "swing"
uses_catalyst = false
artifact_directory = "data/models/swing/technical"

[[models]]
id = "intraday_technical_with_catalyst"
label = "Intraday - Technical and Catalyst"
mode = "intraday"
uses_catalyst = true
artifact_directory = "data/models/intraday/technical_with_catalyst"

[[models]]
id = "intraday_technical"
label = "Intraday - Technical"
mode = "intraday"
uses_catalyst = false
artifact_directory = "data/models/intraday/technical"
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def _write_candidate(
    directory: Path,
    *,
    feature_columns: tuple[str, ...] = ("feature_a",),
    fitted_candidate: object | None = None,
) -> None:
    directory.mkdir(parents=True)
    candidate = directory / "candidate.joblib"
    joblib.dump(
        {
            "feature_columns": list(feature_columns),
            "fitted_candidate": fitted_candidate or _LinearEstimator(),
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
