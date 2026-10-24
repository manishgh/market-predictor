from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from market_predictor.research_api.catalog import FAMILIES, FLAGS, PROFILES, TARGET
from market_predictor.research_api.server import create_research_app


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return _hash(path)


@pytest.fixture
def run(tmp_path: Path) -> tuple[Path, Path]:
    """Synthetic artifact bytes are confined to UNIT tests; no estimator is fit."""
    catalog = ['schema = "market_predictor.research_model_catalog"']
    for profile, profile_name in PROFILES.items():
        relative = f"data/research/{profile}"
        directory = tmp_path / relative
        request = {"schema": "market_predictor.swing_return_training_request", **dict.fromkeys(FLAGS, False),
            "policy": {"target": TARGET, "feature_profile": profile_name}, "feature_names": ["feature_a"],
            "folds": [{"number": i, "score": [f"2022-01-0{i + 2}"]} for i in range(1, 5)]}
        request_hash = _json(directory / "_request.json", request)
        units = {}
        for family in FAMILIES.values():
            for key in [f"{family}/{scope}/fold-{i}" for scope in ("temporal", "security_transfer") for i in range(1, 5)] + [
                    f"{family}/final_refit"]:
                unit_directory = directory / key
                unit_directory.mkdir(parents=True)
                model = unit_directory / "model.joblib"
                model.write_bytes(b"UNIT-only artifact; never deserialized")
                files = {"model.joblib": _hash(model)}
                scope = key.split("/")[1]
                if scope != "final_refit":
                    day = date(2022, 1, int(key[-1]) + 2)
                    frame = pd.DataFrame({"ticker": ["AAA", "BBB", "CCC"],
                        "decision_id": ["one", "two", "three"], "security_id": ["cik:1", "cik:2", "cik:3"],
                        "session_date_et": [day] * 3, "decision_time_utc": pd.to_datetime(["2022-01-03T22:00:00Z"] * 3),
                        "scope_eligible": [True, True, False], "predicted_excess_return": [-0.0123, 0.0, None],
                        "prediction_status": ["research_prediction", "research_prediction", "outside_eligible_scope"]})
                    frame.to_parquet(unit_directory / "predictions.parquet", index=False)
                    files["predictions.parquet"] = _hash(unit_directory / "predictions.parquet")
                payload = {"schema": "market_predictor.swing_return_research_model", "request_sha256": request_hash,
                    "unit": {"family": family, "scope": scope, "target": TARGET, "feature_names": ["feature_a"],
                        "fit_cutoff_utc": "2021-01-01T22:00:00Z", "maximum_training_label_maturity": "2020-12-31T22:00:00Z"},
                    "files": files, "serving_eligible": False, "promotion_eligible": False}
                digest = _json(unit_directory / "_manifest.json", payload)
                units[key] = {"path": f"{relative}/{key}", "manifest_sha256": digest}
        digest = _json(directory / "_manifest.json", {"schema": "market_predictor.swing_return_training",
            "status": "complete_research_only", "request_sha256": request_hash, "units": units, **dict.fromkeys(FLAGS, False)})
        catalog.extend(["[[profiles]]", f'id = "{profile}"', f'label = "{profile}"',
            f'artifact_directory = "{relative}"', f'manifest_sha256 = "{digest}"'])
    catalog_path = tmp_path / "catalog.toml"
    catalog_path.write_text("\n".join(catalog), encoding="utf-8")
    return tmp_path, catalog_path


def _client(run: tuple[Path, Path]) -> TestClient:
    return TestClient(create_research_app(repository_root=run[0], catalog_path=run[1]))


def _query(client: TestClient, **overrides: Any) -> Any:
    return client.post("/v1/research/predict", json={
        "model_id": "technical_ridge", "tickers": ["AAA", "BBB", "CCC", "DDD"],
        "session_date": "2022-01-03", **overrides})


def test_catalog_links_all_six_completed_specs_without_activating(run: tuple[Path, Path]) -> None:
    models = _client(run).get("/v1/research/models").json()["models"]
    assert {model["id"] for model in models} == {f"{profile}_{family}" for profile in PROFILES for family in FAMILIES}
    assert all(model["integrity_verified"] and model["training_status"] == "complete_research_only" for model in models)
    assert all(not model["live_scoring_available"] and not model["serving_eligible"] and not model["actionable"] for model in models)
    assert all(len(model["final_model_sha256"]) == 64 for model in models)


def test_signed_return_zero_and_unavailable_remain_distinct(run: tuple[Path, Path]) -> None:
    response = _query(_client(run))
    assert response.status_code == 200
    payload = response.json()
    assert [row["predicted_excess_return"] for row in payload["predictions"]] == [-0.0123, 0.0, None]
    assert payload["rejected"] == [{"ticker": "DDD", "reason": "No saved row for this exact session."}]
    assert payload["prediction_basis"] == "saved_temporal_out_of_fold"
    assert payload["return_unit"] == "fraction"
    assert payload["provenance"]["unit"] == "regularized_linear_return/temporal/fold-1"
    assert payload["assessment_status"] == "inspected_development_evidence"
    assert payload["actionable"] is False
    assert "actual_return" not in json.dumps(payload)


@pytest.mark.parametrize("name", ["model.joblib", "predictions.parquet", "_manifest.json"])
def test_tampered_fold_refuses_query(run: tuple[Path, Path], name: str) -> None:
    path = run[0] / "data/research/technical/regularized_linear_return/temporal/fold-1" / name
    path.write_bytes(path.read_bytes() + b"changed")
    assert _query(_client(run)).status_code == 409


def test_tampered_final_is_reported_unavailable(run: tuple[Path, Path]) -> None:
    (run[0] / "data/research/technical/regularized_linear_return/final_refit/model.joblib").write_bytes(b"changed")
    models = _client(run).get("/v1/research/models").json()["models"]
    assert models[0]["integrity_verified"] is False
    assert models[0]["training_status"] == "unavailable"
    # Historical fold verification is independent from the final-refit bytes.
    assert _query(_client(run)).status_code == 200


@pytest.mark.parametrize("day", ["2022-01-02", "2026-10-24"])
def test_exact_session_required_no_latest_or_final_refit_fallback(run: tuple[Path, Path], day: str) -> None:
    response = _query(_client(run), session_date=day)
    assert response.status_code == 503
    assert "exact session" in response.json()["detail"]


@pytest.mark.parametrize("overrides", [
    {"model_id": "swing_technical"}, {"model_id": "../../model.joblib"}])
def test_unknown_or_old_model_id_rejected(run: tuple[Path, Path], overrides: dict[str, Any]) -> None:
    assert _query(_client(run), **overrides).status_code == 404


@pytest.mark.parametrize("overrides", [
    {"tickers": []}, {"tickers": [""]}, {"tickers": ["../AAA"]}, {"session_date": "bad"},
    {"as_of": "2026-10-24T22:00:00Z"}, {"features": {"feature_a": 1}}])
def test_invalid_inputs_and_arbitrary_feature_scoring_refused(run: tuple[Path, Path], overrides: dict[str, Any]) -> None:
    assert _query(_client(run), **overrides).status_code == 422


def test_ui_represents_returns_and_requires_historical_date(run: tuple[Path, Path]) -> None:
    client = _client(run)
    html = client.get("/").text
    script = client.get("/assets/app.js").text
    assert 'id="session-date"' in html
    assert "not probabilities" in html
    assert "predicted_excess_return === null" in script
    assert "predicted_excess_return * 100" in script
    assert "final_refit" not in script


def test_production_module_not_shadowed() -> None:
    import market_predictor.api
    assert Path(market_predictor.api.__file__).name == "api.py"
