from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from market_predictor.api_security import ApiSecurityConfig
from market_predictor.research_api.catalog import FAMILIES, PROFILES, TARGET
from market_predictor.research_api.server import create_research_app
from market_predictor.swing.training.return_artifacts import FLAGS


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
            "policy": {"target": TARGET, "scope": "initial_fit_research_only", "feature_profile": profile_name},
            "feature_names": ["feature_a"],
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
    return TestClient(create_research_app(repository_root=run[0], catalog_path=run[1], security_config=ApiSecurityConfig(mode="disabled")))


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


def _fresh_client(run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
                  **changes: Any) -> tuple[TestClient, Path]:
    """UNIT adapter stub; canonical source construction is tested separately."""
    from datetime import UTC, datetime

    import numpy as np

    from market_predictor.core.errors import DataReadinessError
    from market_predictor.research_api import forecast, routes
    from market_predictor.serving.swing_features import _expected_swing_decision_time

    class Clock(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:
            return datetime(2026, 11, 3, 23, 10, tzinfo=UTC)

    monkeypatch.setattr(forecast, "datetime", Clock)
    monkeypatch.setattr(routes, "datetime", Clock)
    root, catalog = run
    catalog.write_text(catalog.read_text() + '\n[forecast]\npublication_directory = "live"\n'
                       + 'publication_manifest_sha256 = "' + 'a' * 64 + '"\n')
    source = root / "UNIT-source.txt"
    source.write_text("UNIT adapter source")
    source_sha = _hash(source)
    decision = _expected_swing_decision_time(pd.Timestamp(Clock.now(UTC)))
    frame = pd.DataFrame({"ticker": ["AAA", "BBB"], "security_id": ["cik:1", "cik:2"],
        "decision_id": ["fresh-one", "fresh-two"], "session_date_et": [decision.tz_convert("America/New_York").date()] * 2,
        "decision_time_utc": [decision] * 2, "feature_available_at_utc": [decision] * 2,
        "feature_profile": ["existing_technical"] * 2, "availability_semantics": ["observed"] * 2,
        "forecast_eligible": [True, True], "forecast_abstention_reason": [None, None], "feature_a": [-0.0123, 0.0]})
    for name, value in changes.items():
        frame[name] = value
    path = root / "live/technical.parquet"
    path.parent.mkdir()
    frame.to_parquet(path, index=False)

    def verified(**kwargs: Any) -> tuple[pd.DataFrame, dict[str, Any]]:
        if _hash(source) != source_sha or not path.is_file():
            raise DataReadinessError("UNIT changed source")
        return frame, {"feature_path": str(path), "feature_sha256": _hash(path),
                       "created_at_utc": decision.isoformat(), "publication_manifest_sha256": "a" * 64}

    monkeypatch.setattr(forecast, "verify_observed_return_publication", verified)
    monkeypatch.setattr(forecast, "load_return_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(forecast, "predict_return", lambda model, features: np.asarray(features.feature_a, dtype=float))
    monkeypatch.setattr(forecast, "guard", lambda stage: None)
    return _client(run), path


def test_fresh_forecast_uses_final_model_and_keeps_research_admission(run: tuple[Path, Path],
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _fresh_client(run, monkeypatch)
    response = client.post("/v1/research/forecast", json={"model_id": "technical_ridge", "tickers": ["BBB", "AAA", "ZZZ"]})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert [row["predicted_excess_return"] for row in payload["predictions"]] == [0.0, -0.0123]
    assert payload["provenance"]["unit"].endswith("/final_refit")
    assert payload["prediction_basis"] == "frozen_final_model_fresh_inputs"
    assert not payload["serving_eligible"] and not payload["promotion_eligible"] and not payload["actionable"]
    assert "feature_path" not in payload["provenance"]
    assert payload["rejected"][0]["ticker"] == "ZZZ"


@pytest.mark.parametrize("changes", [
    {"availability_semantics": "historical_proxy"},
    {"feature_profile": "technical_relationships"},
    {"feature_available_at_utc": pd.Timestamp("2099-01-01T00:00:00Z")},
    {"decision_time_utc": pd.Timestamp("2020-01-01T22:00:00Z")},
    {"ticker": ["AAA", "AAA"]},
    {"feature_a": [float("inf"), 0.0]},
])
def test_fresh_forecast_refuses_wrong_profile_clocks_or_values(run: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch,
                                                              changes: dict[str, Any]) -> None:
    client, _ = _fresh_client(run, monkeypatch, **changes)
    response = client.post("/v1/research/forecast", json={"model_id": "technical_ridge", "tickers": ["AAA"]})
    assert response.status_code == 503


def test_fresh_forecast_requires_current_inputs_and_rejects_changed_source(run: tuple[Path, Path],
                                                                         monkeypatch: pytest.MonkeyPatch) -> None:
    client, path = _fresh_client(run, monkeypatch)
    (run[0] / "UNIT-source.txt").write_text("changed")
    assert client.post("/v1/research/forecast", json={"model_id": "technical_ridge", "tickers": ["AAA"]}).status_code == 503
    path.unlink()
    assert client.post("/v1/research/forecast", json={"model_id": "technical_ridge", "tickers": ["AAA"]}).status_code == 503


def test_canonical_v1_app_authenticates_all_trained_model_routes(run: tuple[Path, Path]) -> None:
    from market_predictor.api import create_app
    from market_predictor.research_api.service import ResearchModelService
    from tests.test_api import StubPredictionService

    token = "UNIT-only-token-12345678901234567890"
    app = create_app(StubPredictionService(), return_model_service=ResearchModelService(run[0], catalog_path=run[1]),
        security_config=ApiSecurityConfig(mode="development", development_token=token,
            development_scopes=frozenset({"predictions.read"})))  # type: ignore[arg-type]
    client = TestClient(app)
    for method, path in (("GET", "/v1/research/models"), ("POST", "/v1/research/predict"), ("POST", "/v1/research/forecast")):
        assert client.request(method, path).status_code == 401
    assert client.get("/v1/research/models", headers={"Authorization": f"Bearer {token}"}).status_code == 200
    schema = client.get("/openapi.json").json()
    assert schema["info"]["version"] == "1"
    assert all(path.startswith("/v1/") for path in schema["paths"])
    assert client.post("/v2/research/forecast", json={}).status_code == 404


def test_forecast_has_no_arbitrary_partition_registration_api() -> None:
    from market_predictor.research_api import forecast
    assert not hasattr(forecast, "publish_return_forecast_inputs")


def test_fresh_abstention_is_null_and_does_not_score_ineligible_row(run: tuple[Path, Path],
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    client, _ = _fresh_client(run, monkeypatch, forecast_eligible=[False, True],
        forecast_abstention_reason=["insufficient_observed_history", None])
    result = client.post("/v1/research/forecast", json={"model_id": "technical_ridge", "tickers": ["AAA", "BBB"]})
    assert result.status_code == 200
    values = result.json()["predictions"]
    assert values[0]["predicted_excess_return"] is None
    assert values[0]["abstention_reason"] == "insufficient_observed_history"
    assert values[1]["predicted_excess_return"] == 0.0


def test_fresh_genuine_missing_operand_reaches_trained_preprocessor(run: tuple[Path, Path],
                                                                  monkeypatch: pytest.MonkeyPatch) -> None:
    from market_predictor.research_api import forecast

    client, _ = _fresh_client(run, monkeypatch, feature_a=[float("nan"), 0.0])

    def preprocessed(model: Any, features: pd.DataFrame) -> Any:
        import numpy as np
        assert features.feature_a.isna().tolist() == [True]
        return np.asarray([0.012])

    monkeypatch.setattr(forecast, "predict_return", preprocessed)
    result = client.post("/v1/research/forecast", json={"model_id": "technical_ridge", "tickers": ["AAA"]})
    assert result.status_code == 200
    assert result.json()["predictions"][0]["predicted_excess_return"] == 0.012


def test_fresh_service_refuses_not_yet_reached_decision(run: tuple[Path, Path],
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    from market_predictor.research_api import forecast
    from market_predictor.research_api.service import ResearchFeatureUnavailableError, ResearchModelService

    _fresh_client(run, monkeypatch)
    verified = forecast.verify_observed_return_publication

    def before_decision(**kwargs: Any) -> Any:
        frame, provenance = verified(**kwargs)
        frame = frame.copy()
        frame["feature_available_at_utc"] = pd.Timestamp("2026-11-03T22:30:00Z")
        provenance["created_at_utc"] = "2026-11-03T22:30:00Z"
        return frame, provenance

    monkeypatch.setattr(forecast, "verify_observed_return_publication", before_decision)
    service = forecast.ReturnForecastService(ResearchModelService(run[0], catalog_path=run[1]),
        memory_budget_gib=5.0, memory_headroom_gib=0.75)
    with pytest.raises(ResearchFeatureUnavailableError, match="future-dated"):
        service._features("technical", "a" * 64, ["feature_a"], pd.Timestamp("2026-11-03T22:45:00Z"))


def test_fresh_service_honors_configured_heavy_job_lease_before_loading(run: tuple[Path, Path],
                                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    from market_predictor.heavy_jobs import heavy_job_lease
    from market_predictor.research_api import forecast

    client, _ = _fresh_client(run, monkeypatch)
    runtime = run[0] / "configured-runtime"
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(runtime))
    monkeypatch.setattr(forecast, "load_run", lambda spec: pytest.fail("loaded input before lease"))
    with heavy_job_lease("UNIT-existing-job", runtime_dir=runtime):
        result = client.post("/v1/research/forecast", json={"model_id": "technical_ridge", "tickers": ["AAA"]})
    assert result.status_code == 503


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
    assert response.json()["error"]["code"] == "prediction_not_ready"


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
