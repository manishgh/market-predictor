"""Synthetic retained-source and publication checks for the fixed forecast experiment."""
import json
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import typer
from typer.testing import CliRunner

from market_predictor.canonical.store import file_sha256
from market_predictor.commands import investment_research
from market_predictor.core.errors import DataReadinessError
from market_predictor.investment.diagnostics import COVERAGE, PREDICTION
from market_predictor.investment.fundamentals import FEATURE_NAMES
from market_predictor.research import investment_feature_ablation, investment_training
from market_predictor.research import investment_forecast_adjustment as module
from market_predictor.research.investment_diagnostics import PREDICTION_COLUMNS
from tests.test_investment_diagnostics_publication import write
from tests.test_investment_feature_ablation_publication import prepare


def test_experiment_publishes_exact_raw_and_adjusted_inventory_and_rejects_reference_tamper(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Keep nested canonical fit paths within the Windows path length limit.
    root = tmp_path_factory.getbasetemp()
    tmp_path = root / "fixture"
    value, _ = prepare(tmp_path, monkeypatch)
    # The first experiment's closed synthetic publications use arbitrary forecasts
    # and invalid model bytes, so accidental historical deserialization fails.
    dataset = value["dataset"]
    # These finite and missing raw inputs exercise native branches in the new real fits.
    dataset.features[FEATURE_NAMES[0]] = pd.Series(range(len(dataset.features)), dtype=float)
    dataset.features[FEATURE_NAMES[1]] = float("nan")
    ablation_input = value["request_pin"]
    ablation_output = value["output"]
    investment_feature_ablation.run_investment_feature_ablation(root=tmp_path, request=Path(ablation_input["path"]),
        request_sha256=ablation_input["sha256"], output=ablation_output)
    ablation_manifest = ablation_output / "_manifest.json"
    reference_files = {path: file_sha256(path) for directory in (tmp_path / "run", ablation_output)
        for path in directory.rglob("*") if path.is_file()}
    request = write(tmp_path / "adjustment-request.json", {
        "schema_version": "market_predictor.investment_forecast_adjustment_request", "scope": "retrospective_research",
        "training_manifest": value["request"]["training_manifest"], "training_request": value["request"]["training_request"],
        "ablation_manifest": {"path": str(ablation_manifest), "sha256": file_sha256(ablation_manifest)},
        "ablation_request": ablation_input,
    })
    monkeypatch.setattr(module, "_implementation", lambda: {})
    monkeypatch.setattr(module, "guard", lambda *_: None)
    monkeypatch.setattr(investment_training, "guard", lambda *_: None)
    monkeypatch.setattr(module, "heavy_job_lease", lambda *_, **__: nullcontext())
    monkeypatch.setattr(module, "load_investment_dataset", lambda **_: dataset)
    calls = []
    publish = module._publish_fit

    def publish_fit(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[1])
        assert kwargs["family"] == module.FAMILY and kwargs["native_missing_columns"] == FEATURE_NAMES
        assert kwargs["columns"] == dataset.model_columns["all_features"]
        return publish(*args, **kwargs)

    monkeypatch.setattr(module, "_publish_fit", publish_fit)
    output = root / "adjustment"
    arguments = {"root": root, "request": Path(request["path"]), "request_sha256": request["sha256"]}
    result = module.run_investment_forecast_adjustment(**arguments, output=output)
    report = json.loads((output / "_manifest.json").read_text())
    assert result["fold_scope_fits"] == len(calls) == len(set(calls)) == 12 and result["final_models"] == 0
    assert len(report["unit_diagnostics"]) == 84 and len(report["aggregate_diagnostics"]) == 28
    assert len(report["reused_units"]) == len(report["adjusted_predictions"]) == 36
    assert all(file_sha256(path) == digest for path, digest in reference_files.items())
    assert not any(".investment-" in path for path in report["source_files"])
    for key, pin in report["new_units"].items():
        path = Path(pin["path"])
        assert path == output / key / "_manifest.json" and file_sha256(path) == pin["sha256"]
        raw_record = json.loads(path.read_text())
        state = raw_record["native_missing_features"]
        assert [item["name"] for item in state] == list(FEATURE_NAMES)
        assert state[1] == {"name": FEATURE_NAMES[1], "finite_training_rows": 0, "lower": None, "upper": None}
        assert state[0]["finite_training_rows"] == raw_record["training_rows"]
    for key, pin in report["adjusted_predictions"].items():
        path = Path(pin["path"])
        assert path == output / "adjusted" / key / "predictions.parquet" and file_sha256(path) == pin["sha256"]
        adjusted = pd.read_parquet(path)
        assert tuple(adjusted.columns) == (*PREDICTION_COLUMNS, *COVERAGE, "raw_predicted_excess_return", "adjusted_at_utc")
        assert adjusted[PREDICTION].notna().equals(adjusted.feature_eligible)
        assert adjusted.adjusted_at_utc.ge(adjusted.prediction_generated_at_utc).all()
        raw_pin = report["new_units"].get(key) or report["reused_units"][key]
        raw = pd.read_parquet(Path(raw_pin["path"]).parent / "predictions.parquet")
        pd.testing.assert_series_equal(adjusted.raw_predicted_excess_return, raw[PREDICTION], check_names=False)
        pd.testing.assert_series_equal(adjusted.research_eligible, raw.research_eligible)
        assert report["unit_diagnostics"][f"{key}/raw"]["canonical_metrics"]["top_decile_by_date"] == (
            report["unit_diagnostics"][f"{key}/centered_shrunk"]["canonical_metrics"]["top_decile_by_date"])
    assert len(report["improvement_checks"]["years"]) == 12
    primary = report["improvement_checks"]["primary_both_scopes"]
    assert primary["passes"] == all(primary["scopes"].values())
    old_model = next(Path(pin["path"]).parent / "model.joblib" for key, pin in report["reused_units"].items()
        if f"/{module.NEWS}/" in key)
    old_model.write_bytes(b"tampered historical model")
    refused = root / "refused"
    with pytest.raises(DataReadinessError, match="source bytes changed"):
        module.run_investment_forecast_adjustment(**arguments, output=refused)
    assert len(calls) == 12 and not refused.exists()


def test_cli_passes_exact_forecast_adjustment_request(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    monkeypatch.setattr(investment_research, "run_investment_forecast_adjustment",
        lambda **kwargs: calls.append(kwargs) or {"status": "fixture"})
    app = typer.Typer()
    investment_research.register_investment_research_commands(app)
    request, output = tmp_path / "request.json", tmp_path / "output"
    result = CliRunner().invoke(app, ["adjust-investment-forecasts", "--root", str(tmp_path),
        "--request", str(request), "--request-sha256", "a" * 64, "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert calls == [{"root": tmp_path, "request": request, "request_sha256": "a" * 64, "output": output}]
