"""Compare separate news and financial views of the admitted investment dataset."""
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
from market_predictor.investment.diagnostics import feature_coverage, prediction_diagnostics
from market_predictor.investment.feature_ablation import ABLATION_PROFILES, NEW_PROFILES, ablation_profiles, factorial_diagnostics
from market_predictor.investment.training import FAMILIES, HORIZONS, PROFILES, SCOPES, SCORE_YEARS
from market_predictor.research.investment_diagnostics import (
    _implementation as diagnostic_implementation,
)
from market_predictor.research.investment_diagnostics import (
    _recheck,
    read_fit_predictions,
    read_reference_predictions,
    read_reference_run,
    reference_folds,
)
from market_predictor.research.investment_features import _publish
from market_predictor.research.investment_training import FLAGS, _fold_record, _publish_fit
from market_predictor.research.investment_training import _implementation as training_implementation
from market_predictor.swing.contracts.holding_accounting import HoldingContract
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_training import BoostedReturnSettings, LinearReturnSettings


class InvestmentFeatureAblationRequest(HoldingContract):
    schema_version: Literal["market_predictor.investment_feature_ablation_request"]
    scope: Literal["retrospective_research"]
    training_manifest: SourcePin
    training_request: SourcePin


def _implementation() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    names = ("research/investment_feature_ablation.py", "investment/feature_ablation.py", "investment/fundamentals.py")
    return {**diagnostic_implementation(), **training_implementation(),
        **{str(package / name): file_sha256(package / name) for name in names}}


def _improvement_checks(aggregates: dict[str, Any]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    for horizon in HORIZONS:
        for family in FAMILIES:
            for scope in SCOPES:
                technical = aggregates[f"{horizon}/{PROFILES[0]}/{family}/{scope}"]["canonical_metrics"]
                for profile in NEW_PROFILES:
                    key = f"{horizon}/{profile}/{family}/{scope}"
                    candidate = aggregates[key]["canonical_metrics"]
                    require(all(candidate[f"zero_excess_baseline_{metric}"] == technical[f"zero_excess_baseline_{metric}"]
                        for metric in ("mse", "mae")), "feature comparison zero-excess benchmarks differ")
                    available = all(candidate[metric] is not None and technical[metric] is not None
                        and candidate[f"zero_excess_baseline_{metric}"] is not None for metric in ("mse", "mae"))
                    comparisons = {metric: {
                        "candidate": candidate[metric], "technical": technical[metric],
                        "zero_excess": candidate[f"zero_excess_baseline_{metric}"],
                        "beats_technical": candidate[metric] < technical[metric] if available else None,
                        "beats_zero_excess": candidate[metric] < candidate[f"zero_excess_baseline_{metric}"] if available else None,
                    } for metric in ("mse", "mae")}
                    cells[key] = {"status": "evaluated" if available else "unavailable", "metrics": comparisons,
                        "passes": all(value["beats_technical"] and value["beats_zero_excess"]
                            for value in comparisons.values()) if available else None}
    primary = {}
    for profile in NEW_PROFILES:
        for family in FAMILIES:
            scopes = {scope: cells[f"252/{profile}/{family}/{scope}"]["passes"] for scope in SCOPES}
            primary[f"{profile}/{family}"] = {"scopes": scopes,
                "passes": all(scopes.values()) if None not in scopes.values() else None}
    return {"cells": cells, "primary_both_scopes": primary}


def run_investment_feature_ablation(*, root: Path, request: Path, request_sha256: str, output: Path) -> dict[str, Any]:
    """Reuse the closed reference units and fit only the two newly defined validation profiles."""
    root = root.resolve()
    output = inside(root, output)
    runtime = heavy_job_runtime_dir()
    with heavy_job_lease("ablate-investment-features", runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("investment feature ablation preparation")
        if output.exists():
            raise FileExistsError(output)
        implementation = _implementation()
        pins = dict(implementation)
        parsed = InvestmentFeatureAblationRequest.model_validate_json(json.dumps(read_object(root,
            SourcePin(path=str(request), sha256=request_sha256), pins)))
        reference = read_reference_run(root, parsed.training_manifest, parsed.training_request, pins)
        train_request = reference.request
        settings = read_object(root, train_request.estimator_settings, pins)
        parameters = {
            FAMILIES[0]: LinearReturnSettings.model_validate_json(json.dumps(settings["linear"])).model_dump(),
            FAMILIES[1]: BoostedReturnSettings.model_validate_json(json.dumps(settings["boosted"])).model_dump(),
        }
        require(parameters == reference.experiment["parameters"], "investment ablation estimator settings differ from the saved run")
        dependencies = {name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib", "exchange-calendars")}
        require(dependencies == reference.experiment["dependencies"],
            "investment ablation dependency versions differ from the saved run")
        dataset = load_investment_dataset(root=root, manifest=train_request.dataset_manifest, admission=train_request.dataset_admission)
        add_pins(root, pins, dataset.source_files)
        folds = reference_folds(reference, dataset)
        columns = ablation_profiles(dataset.model_columns)
        metadata = dataset.metadata.reset_index(drop=True)
        coverage = feature_coverage(dataset.features, columns["all_features"])
        groups = [(horizon, scope, family) for horizon in HORIZONS for scope in SCOPES for family in FAMILIES]
        reused_units: dict[str, Any] = {}
        unit_reports: dict[str, Any] = {}
        aggregate_reports: dict[str, Any] = {}
        saved_frames: dict[str, pd.DataFrame] = {}
        for horizon, scope, family in groups:
            for profile in PROFILES:
                aggregate_key = f"{horizon}/{profile}/{family}/{scope}"
                for year in SCORE_YEARS:
                    guard("investment ablation saved validation unit")
                    fold = folds[(horizon, scope, year)]
                    key = f"{aggregate_key}/{year}"
                    pin, record, frame = read_reference_predictions(root, reference, key, fold, dataset, metadata, coverage, pins)
                    model_path = inside(root, pin.path).parent / "model.joblib"
                    pin_source_file(root, SourcePin(path=str(model_path), sha256=record["files"]["model.joblib"]), pins)
                    diagnostics = prediction_diagnostics(frame)
                    require(diagnostics["canonical_metrics"] == record["metrics"], "investment ablation saved unit metrics differ")
                    reused_units[key] = pin.model_dump(mode="json")
                    unit_reports[key] = diagnostics
                    saved_frames[key] = frame
                combined = pd.concat([saved_frames[f"{aggregate_key}/{year}"] for year in SCORE_YEARS], ignore_index=True)
                diagnostics = prediction_diagnostics(combined)
                require(diagnostics["canonical_metrics"] == reference.manifest["aggregate_metrics"][aggregate_key],
                    "investment ablation saved aggregate metrics differ")
                aggregate_reports[aggregate_key] = diagnostics

        report: dict[str, Any] = {
            "schema": "market_predictor.investment_feature_ablation", "status": "complete_research_only",
            "scope": "retrospective_research", "request_sha256": request_sha256,
            "training_manifest": parsed.training_manifest.model_dump(mode="json"),
            "training_request": parsed.training_request.model_dump(mode="json"),
            "source_files": pins, "implementation_files": implementation, "producer_source_provenance": reference.manifest["source_files"],
            "reused_units": reused_units, "unit_diagnostics": unit_reports, "aggregate_diagnostics": aggregate_reports,
            "saved_model_bytes_rechecked": True, "saved_models_deserialized": False,
            "raw_sources_replayed": False, "calibrator_fitted": False, "model_selected": False, **FLAGS,
        }
        experiment = {
            "schema": "market_predictor.investment_feature_ablation_experiment", "scope": "retrospective_research",
            "request_sha256": request_sha256, "reference_experiment_sha256": reference.manifest["experiment_sha256"],
            "dataset_manifest": train_request.dataset_manifest.model_dump(mode="json"),
            "dataset_admission": train_request.dataset_admission.model_dump(mode="json"),
            "estimator_settings": train_request.estimator_settings.model_dump(mode="json"),
            "feature_contract_sha256": dataset.feature_contract_sha256, "model_columns": columns,
            "parameters": parameters, "folds": reference.experiment["folds"], "dependencies": dependencies,
            "created_at_utc": datetime.now(UTC).isoformat(),
            **FLAGS,
        }

        def publish_fits(staged: Path) -> None:
            experiment_path = staged / "experiment.json"
            write_json_object(experiment_path, experiment)
            experiment_sha256 = file_sha256(experiment_path)
            staged_pins = {str(experiment_path): experiment_sha256}
            new_units: dict[str, Any] = {}
            factorial_units: dict[str, Any] = {}
            factorial_aggregates: dict[str, Any] = {}
            for horizon, scope, family in groups:
                collected: dict[str, list[pd.DataFrame]] = {profile: [] for profile in ABLATION_PROFILES}
                for year in SCORE_YEARS:
                    fold = folds[(horizon, scope, year)]
                    frames = {}
                    for profile in ABLATION_PROFILES:
                        key = f"{horizon}/{profile}/{family}/{scope}/{year}"
                        if profile in PROFILES:
                            frame = saved_frames.pop(key)
                        else:
                            identity = {**_fold_record(fold, metadata), "profile": profile, "experiment_sha256": experiment_sha256}
                            published, _ = _publish_fit(staged, key, features=dataset.features, metadata=metadata,
                                columns=columns[profile], family=family, parameters=parameters[family],
                                train_indices=fold.train_indices, score_indices=fold.score_indices,
                                prediction_indices=fold.prediction_indices, identity=identity, request_sha256=request_sha256,
                                admitted_at=dataset.admitted_at_utc)
                            pin = SourcePin(path=str(staged / key / "_manifest.json"), sha256=published["sha256"])
                            record = read_object(root, pin, staged_pins)
                            require(record == {name: value for name, value in published.items() if name not in ("path", "sha256")},
                                "investment ablation persisted unit differs from the fitted unit")
                            pin_source_file(root, SourcePin(path=str(staged / key / "model.joblib"),
                                sha256=record["files"]["model.joblib"]), staged_pins)
                            frame = read_fit_predictions(root, pin, record, metadata.iloc[fold.score_indices], coverage,
                                staged_pins, pd.Timestamp(datetime.now(UTC)))
                            diagnostics = prediction_diagnostics(frame)
                            require(diagnostics["canonical_metrics"] == record["metrics"],
                                "investment ablation new unit metrics differ on saved predictions")
                            unit_reports[key] = diagnostics
                            new_units[key] = SourcePin(path=str(output / key / "_manifest.json"),
                                sha256=pin.sha256).model_dump(mode="json")
                        frames[profile] = frame
                        collected[profile].append(frame)
                    guard("investment feature ablation fold diagnostics")
                    factorial_units[f"{horizon}/{family}/{scope}/{year}"] = factorial_diagnostics(frames)
                guard("investment feature ablation aggregate diagnostics")
                combined_frames = {profile: pd.concat(frames, ignore_index=True) for profile, frames in collected.items()}
                for profile in NEW_PROFILES:
                    aggregate_reports[f"{horizon}/{profile}/{family}/{scope}"] = prediction_diagnostics(combined_frames[profile])
                factorial_aggregates[f"{horizon}/{family}/{scope}"] = factorial_diagnostics(combined_frames)
            require(len(reused_units) == len(new_units) == 48 and len(unit_reports) == 96 and len(aggregate_reports) == 32
                and len(factorial_units) == 24 and len(factorial_aggregates) == 8 and not saved_frames,
                "investment feature ablation output inventory differs")
            _recheck(pins)
            _recheck(staged_pins)
            guard("investment feature ablation publication")
            report.update(experiment_sha256=experiment_sha256, new_units=new_units, fold_scope_fits=len(new_units),
                final_models=0, factorial_diagnostics={"units": factorial_units, "aggregates": factorial_aggregates},
                improvement_checks=_improvement_checks(aggregate_reports), completed_at_utc=datetime.now(UTC).isoformat())

        result = _publish(output, report, pins, publish_fits)
        return {**result, "fold_scope_fits": report["fold_scope_fits"], "reused_validation_fits": len(reused_units),
            "final_models": report["final_models"]}
