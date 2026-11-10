"""Synthetic publication, reference integrity and exact CLI checks for investment ablation."""
import json
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from market_predictor.canonical.store import file_sha256
from market_predictor.commands import investment_research
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.io import write_json_object
from market_predictor.investment.diagnostics import CONTEXT, PREDICTION
from market_predictor.investment.feature_ablation import NEW_PROFILES, ablation_profiles
from market_predictor.investment.training import FAMILIES, investment_diagnostics, investment_fold
from market_predictor.research import investment_feature_ablation as module
from market_predictor.research.investment_training import FLAGS
from tests.test_investment_diagnostics_publication import fixture, write


def prepare(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], list[str]]:
    value = fixture(tmp_path, monkeypatch)
    request = {**value["request"], "schema_version": "market_predictor.investment_feature_ablation_request",
        "scope": "retrospective_research"}
    value["request_pin"] = write(tmp_path / "ablation-request.json", request)
    value["output"] = tmp_path / "ablation"
    dataset = value["dataset"]
    original_training_request = json.loads(Path(request["training_request"]["path"]).read_text())
    settings = json.loads(Path(original_training_request["estimator_settings"]["path"]).read_text())
    reference_settings = {FAMILIES[0]: settings["linear"], FAMILIES[1]: settings["boosted"]}

    def load_dataset(**kwargs: Any) -> Any:
        for field in ("manifest", "admission"):
            assert kwargs[field].model_dump(mode="json") == original_training_request[f"dataset_{field}"]
        return dataset

    monkeypatch.setattr(module, "load_investment_dataset", load_dataset)
    monkeypatch.setattr(module, "heavy_job_lease", lambda *_, **__: nullcontext())
    monkeypatch.setattr(module, "guard", lambda *_: None)
    monkeypatch.setattr(module, "_implementation", lambda: {})
    monkeypatch.setattr("market_predictor.research.investment_diagnostics.guard", lambda *_: None)
    columns = ablation_profiles(dataset.model_columns)
    calls: list[str] = []

    def publish_fit(output: Path, key: str, **kwargs: Any) -> tuple[dict[str, Any], pd.DataFrame]:
        calls.append(key)
        horizon, profile, family, scope, year = key.split("/")
        assert profile in NEW_PROFILES and kwargs["columns"] == columns[profile]
        assert kwargs["family"] == family and kwargs["parameters"] == reference_settings[family]
        assert kwargs["features"] is dataset.features
        assert kwargs["admitted_at"] == dataset.admitted_at_utc
        assert kwargs["request_sha256"] == value["request_pin"]["sha256"]
        fold = investment_fold(dataset.metadata, int(horizon), int(year), scope)
        for name in ("train_indices", "score_indices", "prediction_indices"):
            np.testing.assert_array_equal(kwargs[name], getattr(fold, name))
        assert kwargs["identity"]["experiment_sha256"] == file_sha256(output / "experiment.json")
        prediction = dataset.metadata.iloc[fold.score_indices].loc[:, list(CONTEXT)].copy()
        prediction[PREDICTION] = 0.0
        started = pd.Timestamp(datetime.now(UTC))
        prediction["actual_fit_started_at_utc"] = started
        prediction["actual_fitted_at_utc"] = started
        prediction["prediction_generated_at_utc"] = started
        prediction["evidence_scope"] = "retrospective_research"
        prediction["prediction_missing_reason"] = None
        target = output / key
        target.mkdir(parents=True)
        prediction.to_parquet(target / "predictions.parquet", index=False)
        (target / "model.joblib").write_bytes(b"synthetic newly published estimator")
        record = {"schema": "market_predictor.investment_research_model", "scope": "retrospective_research",
            "request_sha256": kwargs["request_sha256"], "unit": kwargs["identity"], "family": family,
            "parameters": kwargs["parameters"], "feature_names": list(kwargs["columns"]),
            "actual_fit_started_at_utc": started.isoformat(), "actual_fitted_at_utc": started.isoformat(),
            "dataset_admitted_at_utc": dataset.admitted_at_utc.isoformat(),
            "latest_training_label_available_at_utc": dataset.metadata.iloc[fold.train_indices].label_available_at_utc.max().isoformat(),
            "metrics": investment_diagnostics(prediction), "persistence_prediction_parity": True,
            "files": {name: file_sha256(target / name) for name in ("model.joblib", "predictions.parquet")}, **FLAGS}
        write_json_object(target / "_manifest.json", record)
        return {"path": str(target), "sha256": file_sha256(target / "_manifest.json"), **record}, prediction

    monkeypatch.setattr(module, "_publish_fit", publish_fit)
    return value, calls


def run(tmp_path: Path, value: dict[str, Any]) -> dict[str, Any]:
    pin = value["request_pin"]
    return module.run_investment_feature_ablation(root=tmp_path, request=Path(pin["path"]),
        request_sha256=pin["sha256"], output=value["output"])


def test_exact_reference_and_new_validation_inventory_publishes_atomically(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value, calls = prepare(tmp_path, monkeypatch)
    before = {path: file_sha256(path) for path in (tmp_path / "run").rglob("*") if path.is_file()}
    result = run(tmp_path, value)
    report = json.loads((value["output"] / "_manifest.json").read_text())
    experiment = json.loads((value["output"] / "experiment.json").read_text())
    assert result["status"] == "complete_research_only"
    assert len(calls) == len(set(calls)) == result["fold_scope_fits"] == result["reused_validation_fits"] == 48
    assert result["final_models"] == 0
    assert report["reused_units"] == {key: pin for key, pin in value["manifest"]["units"].items() if not key.endswith("final_refit")}
    assert set(report["new_units"]) == set(calls)
    assert len(report["unit_diagnostics"]) == 96 and len(report["aggregate_diagnostics"]) == 32
    assert len(report["factorial_diagnostics"]["units"]) == 24 and len(report["factorial_diagnostics"]["aggregates"]) == 8
    assert len(report["improvement_checks"]["cells"]) == 16 and len(report["improvement_checks"]["primary_both_scopes"]) == 4
    assert {name: len(columns) for name, columns in experiment["model_columns"].items()} == {
        "technical_relationships": 124, "technical_news": 209, "technical_financial": 130, "all_features": 215}
    assert all(file_sha256(path) == digest for path, digest in before.items())
    assert all(report["source_files"][str(path)] == digest for path, digest in before.items() if path.name == "model.joblib")
    assert not report["saved_models_deserialized"] and report["saved_model_bytes_rechecked"]
    assert not any(".investment-" in path for path in report["source_files"])
    assert report["experiment_sha256"] == report["files"]["experiment.json"]
    for key, pin in report["new_units"].items():
        path = value["output"] / key / "_manifest.json"
        assert Path(pin["path"]) == path and file_sha256(path) == pin["sha256"]
        record = json.loads(path.read_text())
        assert record["request_sha256"] == value["request_pin"]["sha256"]
        assert record["unit"]["experiment_sha256"] == report["experiment_sha256"]
    assert any(item["canonical_metrics"]["unknown_scored_outcomes"] > 0
        for item in report["aggregate_diagnostics"].values())
    assert all(not cell["passes"] for cell in report["improvement_checks"]["cells"].values())


@pytest.mark.parametrize("filename", ["model.joblib", "predictions.parquet"])
def test_reference_tamper_refuses_before_any_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str) -> None:
    value, calls = prepare(tmp_path, monkeypatch)
    key = f"63/all_features/{FAMILIES[-1]}/security_transfer/2024"
    path = Path(value["manifest"]["units"][key]["path"]).parent / filename
    path.write_bytes(b"changed after original training")
    with pytest.raises(DataReadinessError):
        run(tmp_path, value)
    assert not calls and not value["output"].exists()


def test_dependency_drift_refuses_before_any_fit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value, calls = prepare(tmp_path, monkeypatch)
    monkeypatch.setattr(module.importlib.metadata, "version", lambda _: "different version")
    with pytest.raises(DataReadinessError, match="dependency versions"):
        run(tmp_path, value)
    assert not calls and not value["output"].exists()


def test_publication_rechecks_reference_after_new_fits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    value, calls = prepare(tmp_path, monkeypatch)
    saved_model = next((tmp_path / "run").rglob("model.joblib"))
    publish = module._publish_fit

    def fit_and_mutate(*args: Any, **kwargs: Any) -> Any:
        result = publish(*args, **kwargs)
        if len(calls) == 1:
            saved_model.write_bytes(b"changed after reference readback")
        return result

    monkeypatch.setattr(module, "_publish_fit", fit_and_mutate)
    with pytest.raises(DataReadinessError, match="source changed"):
        run(tmp_path, value)
    assert len(calls) == 48 and not value["output"].exists()


def test_cli_forwards_exact_feature_ablation_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(investment_research, "run_investment_feature_ablation",
        lambda **kwargs: calls.append(kwargs) or {"status": "fixture"})
    app = typer.Typer()
    investment_research.register_investment_research_commands(app)
    request, output = tmp_path / "request.json", tmp_path / "output"
    result = CliRunner().invoke(app, ["ablate-investment-features", "--root", str(tmp_path), "--request",
        str(request), "--request-sha256", "a" * 64, "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert calls == [{"root": tmp_path, "request": request, "request_sha256": "a" * 64, "output": output}]
