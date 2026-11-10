"""Measure robust financial inputs and score-only shrinkage on the frozen investment folds."""
from __future__ import annotations

import importlib.metadata
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.investment.dataset import add_pins, load_investment_dataset, pin_source_file, read_object, require
from market_predictor.investment.diagnostics import PREDICTION, feature_coverage, prediction_diagnostics
from market_predictor.investment.feature_ablation import NEW_PROFILES, ablation_profiles
from market_predictor.investment.forecast_adjustment import centered_shrunk_predictions
from market_predictor.investment.fundamentals import FEATURE_NAMES
from market_predictor.investment.training import FAMILIES, HORIZONS, PROFILES, SCOPES, SCORE_YEARS
from market_predictor.research.investment_diagnostics import (
    _recheck,
    read_fit_predictions,
    read_reference_run,
    read_validation_predictions,
    reference_folds,
)
from market_predictor.research.investment_feature_ablation import InvestmentComparisonSources, read_ablation_reference
from market_predictor.research.investment_feature_ablation import _implementation as ablation_implementation
from market_predictor.research.investment_features import _publish
from market_predictor.research.investment_training import FLAGS, _fold_record, _publish_fit
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_training import BoostedReturnSettings

PROFILE = "clipped_financial_native_missing"
TECHNICAL, FULL = PROFILES
NEWS = NEW_PROFILES[0]
FAMILY = FAMILIES[1]
RAW_PROFILES = (TECHNICAL, NEWS, FULL, PROFILE)
ADJUSTED_PROFILES = (NEWS, FULL, PROFILE)


class InvestmentForecastAdjustmentRequest(InvestmentComparisonSources):
    schema_version: Literal["market_predictor.investment_forecast_adjustment_request"]
    scope: Literal["retrospective_research"]


def _implementation() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    names = ("research/investment_forecast_adjustment.py", "investment/forecast_adjustment.py")
    return {**ablation_implementation(), **{str(package / name): file_sha256(package / name) for name in names}}


def _benchmark_check(reports: dict[str, Any]) -> dict[str, Any]:
    candidate = reports[f"{PROFILE}/centered_shrunk"]
    metrics = candidate["canonical_metrics"]
    controls = {name: reports[name]["canonical_metrics"] for name in (
        f"{TECHNICAL}/raw", f"{NEWS}/raw", f"{NEWS}/centered_shrunk", f"{FULL}/centered_shrunk")}
    controls["zero_excess"] = {metric: metrics[f"zero_excess_baseline_{metric}"] for metric in ("mse", "mae")}
    comparisons: dict[str, Any] = {}
    for name, control in controls.items():
        losses = {metric: {"candidate": metrics[metric], "benchmark": control[metric],
            "passes": metrics[metric] < control[metric] if metrics[metric] is not None and control[metric] is not None else None}
            for metric in ("mse", "mae")}
        comparisons[name] = losses
    biases = {name: value["moments"]["mean_error"] if value["moments"] is not None else None
        for name, value in reports.items() if name in (f"{PROFILE}/raw", f"{PROFILE}/centered_shrunk", f"{FULL}/raw")}
    current, raw = biases[f"{PROFILE}/centered_shrunk"], biases[f"{PROFILE}/raw"]
    bias_pass = abs(current) < abs(raw) if current is not None and raw is not None else None
    gates = [bias_pass, *(loss["passes"] for comparison in comparisons.values() for loss in comparison.values())]
    return {"loss_comparisons": comparisons, "known_mean_errors": biases, "absolute_bias_improves": bias_pass,
        "passes": all(gates) if None not in gates else None}


def run_investment_forecast_adjustment(*, root: Path, request: Path, request_sha256: str, output: Path) -> dict[str, Any]:
    """Fit the twelve frozen boosted units and publish raw and derived forecast evidence."""
    root = root.resolve()
    output = inside(root, output)
    runtime = heavy_job_runtime_dir()
    with heavy_job_lease("adjust-investment-forecasts", runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("investment forecast adjustment preparation")
        if output.exists():
            raise FileExistsError(output)
        implementation = _implementation()
        pins = dict(implementation)
        parsed = InvestmentForecastAdjustmentRequest.model_validate_json(json.dumps(read_object(root,
            SourcePin(path=str(request), sha256=request_sha256), pins)))
        training = read_reference_run(root, parsed.training_manifest, parsed.training_request, pins)
        train_request = training.request
        settings = read_object(root, train_request.estimator_settings, pins)
        parameters = BoostedReturnSettings.model_validate_json(json.dumps(settings["boosted"])).model_dump()
        dependencies = {name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib", "exchange-calendars")}
        require(parameters == training.experiment["parameters"][FAMILY] and dependencies == training.experiment["dependencies"],
            "forecast adjustment estimator parameters or dependency versions differ")
        dataset = load_investment_dataset(root=root, manifest=train_request.dataset_manifest, admission=train_request.dataset_admission)
        add_pins(root, pins, dataset.source_files)
        folds = reference_folds(training, dataset)
        columns = ablation_profiles(dataset.model_columns)
        ablation = read_ablation_reference(root, parsed, training, columns, pins)
        metadata = dataset.metadata.reset_index(drop=True)
        coverage = feature_coverage(dataset.features, columns[FULL])
        columns[PROFILE] = columns[FULL]
        reused_units: dict[str, Any] = {}
        saved_frames: dict[str, pd.DataFrame] = {}
        unit_reports: dict[str, Any] = {}
        for horizon, scope, year in folds:
            for profile in RAW_PROFILES[:-1]:
                guard("investment forecast adjustment saved validation unit")
                key = f"{horizon}/{profile}/{FAMILY}/{scope}/{year}"
                reference = ablation if profile == NEWS else training
                inventory = "new_units" if profile == NEWS else "units"
                pin = SourcePin.model_validate_json(json.dumps(reference.manifest[inventory][key]))
                record, frame = read_validation_predictions(root, reference, pin, key, folds[(horizon, scope, year)],
                    dataset, metadata, coverage, pins, columns[profile])
                model = inside(root, pin.path).parent / "model.joblib"
                pin_source_file(root, SourcePin(path=str(model), sha256=record["files"]["model.joblib"]), pins)
                diagnostics = prediction_diagnostics(frame)
                require(diagnostics["canonical_metrics"] == record["metrics"], "forecast adjustment saved unit metrics differ")
                unit_reports[f"{key}/raw"] = diagnostics
                reused_units[key] = pin.model_dump(mode="json")
                saved_frames[key] = frame

        reference_pins = parsed.model_dump(mode="json", exclude={"schema_version", "scope"})
        report: dict[str, Any] = {
            "schema": "market_predictor.investment_forecast_adjustment", "status": "complete_research_only",
            "scope": "retrospective_research", "request_sha256": request_sha256, **reference_pins,
            "source_files": pins, "implementation_files": implementation, "reused_units": reused_units,
            "producer_source_provenance": {"training": training.manifest["source_files"], "ablation": ablation.manifest["source_files"]},
            "unit_diagnostics": unit_reports, "saved_model_bytes_rechecked": True, "saved_models_deserialized": False,
            "raw_sources_replayed": False, "calibrator_fitted": False, "model_selected": False, **FLAGS,
        }
        experiment = {
            "schema": "market_predictor.investment_forecast_adjustment_experiment", "scope": "retrospective_research",
            "request_sha256": request_sha256, **reference_pins,
            "training_experiment_sha256": training.manifest["experiment_sha256"],
            "ablation_experiment_sha256": ablation.manifest["experiment_sha256"],
            **{name: getattr(train_request, name).model_dump(mode="json")
                for name in ("dataset_manifest", "dataset_admission", "estimator_settings")},
            "feature_contract_sha256": dataset.feature_contract_sha256,
            "model_columns": {profile: columns[profile] for profile in RAW_PROFILES},
            "parameters": {FAMILY: parameters}, "dependencies": dependencies, "folds": training.experiment["folds"],
            "native_missing_columns": FEATURE_NAMES, "clip_quantiles": [0.01, 0.99], "quantile_method": "linear",
            "shrinkage": 0.5, "centering": "mean_all_feature_eligible_forecasts_within_horizon_date_scope",
            "created_at_utc": datetime.now(UTC).isoformat(), **FLAGS,
        }

        def publish_fits(staged: Path) -> None:
            experiment_path = staged / "experiment.json"
            write_json_object(experiment_path, experiment)
            experiment_sha256 = file_sha256(experiment_path)
            staged_pins = {str(experiment_path): experiment_sha256}
            new_units: dict[str, Any] = {}
            adjusted_predictions: dict[str, Any] = {}
            aggregate_reports: dict[str, Any] = {}
            cells: dict[str, Any] = {}
            years: dict[str, Any] = {}
            for horizon in HORIZONS:
                for scope in SCOPES:
                    collected: dict[str, list[pd.DataFrame]] = {}
                    for year in SCORE_YEARS:
                        fold = folds[(horizon, scope, year)]
                        year_reports = {}
                        for profile in RAW_PROFILES:
                            key = f"{horizon}/{profile}/{FAMILY}/{scope}/{year}"
                            raw_key = f"{key}/raw"
                            if profile == PROFILE:
                                identity = {**_fold_record(fold, metadata), "profile": profile, "experiment_sha256": experiment_sha256}
                                fitted, _ = _publish_fit(staged, key, features=dataset.features, metadata=metadata,
                                    columns=columns[profile], family=FAMILY, parameters=parameters, train_indices=fold.train_indices,
                                    score_indices=fold.score_indices, prediction_indices=fold.prediction_indices, identity=identity,
                                    request_sha256=request_sha256, admitted_at=dataset.admitted_at_utc,
                                    native_missing_columns=FEATURE_NAMES)
                                pin = SourcePin(path=str(staged / key / "_manifest.json"), sha256=fitted["sha256"])
                                record = read_object(root, pin, staged_pins)
                                require(record == {name: value for name, value in fitted.items() if name not in ("path", "sha256")}
                                    and tuple(feature["name"] for feature in record["native_missing_features"]) == FEATURE_NAMES,
                                    "forecast adjustment fitted native feature lineage differs")
                                pin_source_file(root, SourcePin(path=str(staged / key / "model.joblib"),
                                    sha256=record["files"]["model.joblib"]), staged_pins)
                                frame = read_fit_predictions(root, pin, record, metadata.iloc[fold.score_indices], coverage,
                                    staged_pins, pd.Timestamp(datetime.now(UTC)))
                                unit_reports[raw_key] = prediction_diagnostics(frame)
                                require(unit_reports[raw_key]["canonical_metrics"] == record["metrics"],
                                    "forecast adjustment saved new fit metrics differ")
                                new_units[key] = SourcePin(path=str(output / key / "_manifest.json"),
                                    sha256=pin.sha256).model_dump(mode="json")
                            else:
                                frame = saved_frames.pop(key)
                            raw_name = f"{profile}/raw"
                            collected.setdefault(raw_name, []).append(frame)
                            year_reports[raw_name] = unit_reports[raw_key]
                            if profile in ADJUSTED_PROFILES:
                                guard("investment forecast adjustment derived scores")
                                adjusted = centered_shrunk_predictions(frame)
                                diagnostics = prediction_diagnostics(adjusted)
                                require(diagnostics["canonical_metrics"]["top_decile_by_date"] ==
                                    unit_reports[raw_key]["canonical_metrics"]["top_decile_by_date"],
                                    "forecast centering changed within-date top-decile selection or coverage")
                                name = f"{profile}/centered_shrunk"
                                unit_reports[f"{key}/centered_shrunk"] = year_reports[name] = diagnostics
                                collected.setdefault(name, []).append(adjusted)
                                relative = Path("adjusted") / key / "predictions.parquet"
                                child = staged / relative
                                child.parent.mkdir(parents=True)
                                adjusted.assign(raw_predicted_excess_return=frame[PREDICTION],
                                    adjusted_at_utc=datetime.now(UTC)).to_parquet(child, index=False)
                                digest = file_sha256(child)
                                staged_pins[str(child)] = digest
                                adjusted_predictions[key] = SourcePin(path=str(output / relative), sha256=digest).model_dump(mode="json")
                        years[f"{horizon}/{scope}/{year}"] = _benchmark_check(year_reports)
                    guard("investment forecast adjustment aggregate diagnostics")
                    group_reports = {}
                    for name, frames in collected.items():
                        profile, view = name.split("/")
                        diagnostics = prediction_diagnostics(pd.concat(frames, ignore_index=True))
                        aggregate_reports[f"{horizon}/{profile}/{FAMILY}/{scope}/{view}"] = group_reports[name] = diagnostics
                        if profile != PROFILE and view == "raw":
                            key = f"{horizon}/{profile}/{FAMILY}/{scope}"
                            expected = (ablation.manifest["aggregate_diagnostics"][key]["canonical_metrics"] if profile == NEWS
                                else training.manifest["aggregate_metrics"][key])
                            require(diagnostics["canonical_metrics"] == expected, "forecast adjustment saved aggregate metrics differ")
                    cells[f"{horizon}/{scope}"] = _benchmark_check(group_reports)
            require(len(new_units) == 12 and len(adjusted_predictions) == len(reused_units) == 36
                and len(unit_reports) == 84 and len(aggregate_reports) == 28 and not saved_frames,
                "forecast adjustment output inventory differs")
            primary_scopes = {scope: cells[f"252/{scope}"]["passes"] for scope in SCOPES}
            _recheck(pins)
            _recheck(staged_pins)
            guard("investment forecast adjustment publication")
            report.update(experiment_sha256=experiment_sha256, new_units=new_units, adjusted_predictions=adjusted_predictions,
                aggregate_diagnostics=aggregate_reports, improvement_checks={"cells": cells, "years": years,
                    "primary_both_scopes": {"scopes": primary_scopes,
                        "passes": all(primary_scopes.values()) if None not in primary_scopes.values() else None}},
                fold_scope_fits=len(new_units), final_models=0, completed_at_utc=datetime.now(UTC).isoformat())

        result = _publish(output, report, pins, publish_fits)
        return {**result, "fold_scope_fits": report["fold_scope_fits"], "reused_validation_fits": len(reused_units),
            "final_models": report["final_models"]}
