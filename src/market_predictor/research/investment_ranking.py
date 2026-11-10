"""Run the target-only within-date investment ranking experiment."""
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
from market_predictor.investment.diagnostics import PREDICTION, feature_coverage
from market_predictor.investment.feature_ablation import NEW_PROFILES, ablation_profiles
from market_predictor.investment.ranking import RANKING_SCORE, ranking_diagnostics
from market_predictor.investment.training import FAMILIES, HORIZONS, PROFILES, SCOPES, SCORE_YEARS, investment_diagnostics
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
from market_predictor.research.investment_training import FLAGS, RANKING_PROFILE, RANKING_TARGET, _fold_record, _publish_fit
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_training import BoostedReturnSettings

TECHNICAL, FULL = PROFILES
NEWS = NEW_PROFILES[0]
FAMILY = FAMILIES[1]
CONTROL_PROFILES = (TECHNICAL, NEWS, FULL)
RANKING_PROFILES = (*CONTROL_PROFILES, RANKING_PROFILE)


class InvestmentRankingRequest(InvestmentComparisonSources):
    schema_version: Literal["market_predictor.investment_ranking_request"]
    scope: Literal["retrospective_research"]


def _implementation() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    names = ("research/investment_ranking.py", "investment/ranking.py")
    return {**ablation_implementation(), **{str(package / name): file_sha256(package / name) for name in names}}


def _ranking_check(reports: dict[str, Any]) -> dict[str, Any]:
    candidate = reports[RANKING_PROFILE]
    common_support = ("declared_rows", "scored_rows", "known_scored_outcomes", "unknown_scored_outcomes",
        "rank_date_support", "auc_date_support")
    require(all(reports[profile][name] == candidate[name] for profile in CONTROL_PROFILES for name in common_support),
        "investment ranking controls differ in scoring or label-supported date population")
    metric_names = ("mean_date_rank_correlation", "mean_within_date_auc")
    metrics = {profile: {metric_names[0]: report[metric_names[0]], metric_names[1]: report["auc"][metric_names[1]]}
        for profile, report in reports.items()}
    controls = {profile: metrics[profile] for profile in CONTROL_PROFILES}
    controls["no_discrimination"] = dict(zip(metric_names, (0.0, 0.5), strict=True))
    comparisons = {}
    for profile, control in controls.items():
        comparisons[profile] = {
            metric: {"candidate": value, "benchmark": control[metric],
                "passes": value > control[metric] if value is not None and control[metric] is not None else None}
            for metric, value in metrics[RANKING_PROFILE].items()}
    passes = [value["passes"] for comparison in comparisons.values() for value in comparison.values()]
    return {"comparisons": comparisons, "passes": all(passes) if None not in passes else None,
        "rank_date_support": candidate["rank_date_support"], "auc_date_support": candidate["auc_date_support"]}


def train_investment_ranking(*, root: Path, request: Path, request_sha256: str, output: Path) -> dict[str, Any]:
    """Reuse exact saved return controls and fit twelve models with a rank target."""
    root = root.resolve()
    output = inside(root, output)
    runtime = heavy_job_runtime_dir()
    with heavy_job_lease("train-investment-ranking", runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("investment ranking preparation")
        if output.exists():
            raise FileExistsError(output)
        implementation = _implementation()
        pins = dict(implementation)
        parsed = InvestmentRankingRequest.model_validate_json(json.dumps(read_object(root,
            SourcePin(path=str(request), sha256=request_sha256), pins)))
        training = read_reference_run(root, parsed.training_manifest, parsed.training_request, pins)
        train_request = training.request
        settings = read_object(root, train_request.estimator_settings, pins)
        parameters = BoostedReturnSettings.model_validate_json(json.dumps(settings["boosted"])).model_dump()
        dependencies = {name: importlib.metadata.version(name)
            for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib", "exchange-calendars")}
        require(parameters == training.experiment["parameters"][FAMILY] and dependencies == training.experiment["dependencies"],
            "investment ranking estimator parameters or dependency versions differ")
        dataset = load_investment_dataset(root=root, manifest=train_request.dataset_manifest, admission=train_request.dataset_admission)
        add_pins(root, pins, dataset.source_files)
        folds = reference_folds(training, dataset)
        columns = ablation_profiles(dataset.model_columns)
        ablation = read_ablation_reference(root, parsed, training, columns, pins)
        metadata = dataset.metadata.reset_index(drop=True)
        coverage = feature_coverage(dataset.features, columns[FULL])
        reused_units: dict[str, Any] = {}
        control_frames: dict[str, pd.DataFrame] = {}
        unit_reports: dict[str, Any] = {}
        for horizon, scope, year in folds:
            for profile in CONTROL_PROFILES:
                guard("investment ranking saved control")
                key = f"{horizon}/{profile}/{FAMILY}/{scope}/{year}"
                reference = ablation if profile == NEWS else training
                inventory = "new_units" if profile == NEWS else "units"
                pin = SourcePin.model_validate_json(json.dumps(reference.manifest[inventory][key]))
                record, frame = read_validation_predictions(root, reference, pin, key, folds[(horizon, scope, year)],
                    dataset, metadata, coverage, pins, columns[profile])
                model_path = inside(root, pin.path).parent / "model.joblib"
                pin_source_file(root, SourcePin(path=str(model_path), sha256=record["files"]["model.joblib"]), pins)
                require(investment_diagnostics(frame) == record["metrics"], "investment ranking saved control metrics differ")
                unit_reports[key] = ranking_diagnostics(frame, score_column=PREDICTION)
                reused_units[key] = pin.model_dump(mode="json")
                control_frames[key] = frame

        reference_pins = parsed.model_dump(mode="json", exclude={"schema_version", "scope"})
        report: dict[str, Any] = {
            "schema": "market_predictor.investment_ranking", "status": "complete_research_only",
            "scope": "retrospective_research", "request_sha256": request_sha256, **reference_pins,
            "source_files": pins, "implementation_files": implementation, "reused_units": reused_units,
            "producer_source_provenance": {"training": training.manifest["source_files"], "ablation": ablation.manifest["source_files"]},
            "unit_diagnostics": unit_reports, "ranking_target": RANKING_TARGET, "score_column": RANKING_SCORE,
            "score_units": "dimensionless", "saved_model_bytes_rechecked": True, "saved_models_deserialized": False,
            "raw_sources_replayed": False, "calibrator_fitted": False, "model_selected": False, **FLAGS,
        }
        experiment = {
            "schema": "market_predictor.investment_ranking_experiment", "scope": "retrospective_research",
            "request_sha256": request_sha256, **reference_pins,
            "training_experiment_sha256": training.manifest["experiment_sha256"],
            "ablation_experiment_sha256": ablation.manifest["experiment_sha256"],
            **{name: getattr(train_request, name).model_dump(mode="json")
                for name in ("dataset_manifest", "dataset_admission", "estimator_settings")},
            "feature_contract_sha256": dataset.feature_contract_sha256, "model_columns": {RANKING_PROFILE: columns[NEWS]},
            "parameters": {FAMILY: parameters}, "dependencies": dependencies, "folds": training.experiment["folds"],
            "ranking_target": RANKING_TARGET, "target_formula": "2 * (average_rank - 0.5) / training_date_row_count - 1",
            "score_column": RANKING_SCORE, "score_units": "dimensionless",
            "created_at_utc": datetime.now(UTC).isoformat(), **FLAGS,
        }

        def publish_fits(staged: Path) -> None:
            experiment_path = staged / "experiment.json"
            write_json_object(experiment_path, experiment)
            experiment_sha256 = file_sha256(experiment_path)
            staged_pins = {str(experiment_path): experiment_sha256}
            new_units: dict[str, Any] = {}
            aggregate_reports: dict[str, Any] = {}
            cells: dict[str, Any] = {}
            years: dict[str, Any] = {}
            for horizon in HORIZONS:
                for scope in SCOPES:
                    collected: dict[str, list[pd.DataFrame]] = {profile: [] for profile in RANKING_PROFILES}
                    for year in SCORE_YEARS:
                        fold = folds[(horizon, scope, year)]
                        key = f"{horizon}/{RANKING_PROFILE}/{FAMILY}/{scope}/{year}"
                        identity = {**_fold_record(fold, metadata), "profile": RANKING_PROFILE, "experiment_sha256": experiment_sha256}
                        fitted, _ = _publish_fit(staged, key, features=dataset.features, metadata=metadata, columns=columns[NEWS],
                            family=FAMILY, parameters=parameters, train_indices=fold.train_indices, score_indices=fold.score_indices,
                            prediction_indices=fold.prediction_indices, identity=identity, request_sha256=request_sha256,
                            admitted_at=dataset.admitted_at_utc, ranking_target=True)
                        pin = SourcePin(path=str(staged / key / "_manifest.json"), sha256=fitted["sha256"])
                        record = read_object(root, pin, staged_pins)
                        require(record == {name: value for name, value in fitted.items() if name not in ("path", "sha256")},
                            "investment ranking persisted unit differs from the fitted unit")
                        pin_source_file(root, SourcePin(path=str(staged / key / "model.joblib"),
                            sha256=record["files"]["model.joblib"]), staged_pins)
                        frame = read_fit_predictions(root, pin, record, metadata.iloc[fold.score_indices], coverage,
                            staged_pins, pd.Timestamp(datetime.now(UTC)), score_column=RANKING_SCORE)
                        diagnostics = ranking_diagnostics(frame)
                        require(diagnostics == record["metrics"], "investment ranking new fit metrics do not replay on saved scores")
                        unit_reports[key] = diagnostics
                        new_units[key] = SourcePin(path=str(output / key / "_manifest.json"), sha256=pin.sha256).model_dump(mode="json")
                        collected[RANKING_PROFILE].append(frame)
                        year_reports = {RANKING_PROFILE: diagnostics}
                        for profile in CONTROL_PROFILES:
                            control_key = f"{horizon}/{profile}/{FAMILY}/{scope}/{year}"
                            collected[profile].append(control_frames.pop(control_key))
                            year_reports[profile] = unit_reports[control_key]
                        years[f"{horizon}/{scope}/{year}"] = _ranking_check(year_reports)
                    guard("investment ranking aggregate diagnostics")
                    group_reports = {}
                    for profile, frames in collected.items():
                        combined = pd.concat(frames, ignore_index=True)
                        key = f"{horizon}/{profile}/{FAMILY}/{scope}"
                        if profile in CONTROL_PROFILES:
                            expected = (ablation.manifest["aggregate_diagnostics"][key]["canonical_metrics"] if profile == NEWS
                                else training.manifest["aggregate_metrics"][key])
                            require(investment_diagnostics(combined) == expected,
                                "investment ranking saved control aggregate metrics differ")
                        score_column = RANKING_SCORE if profile == RANKING_PROFILE else PREDICTION
                        diagnostics = ranking_diagnostics(combined, score_column=score_column)
                        aggregate_reports[key] = group_reports[profile] = diagnostics
                    cells[f"{horizon}/{scope}"] = _ranking_check(group_reports)
            require(len(new_units) == 12 and len(reused_units) == 36 and len(unit_reports) == 48
                and len(aggregate_reports) == 16 and len(years) == 12 and len(cells) == 4 and not control_frames,
                "investment ranking output inventory differs")
            primary_scopes = {scope: cells[f"252/{scope}"]["passes"] for scope in SCOPES}
            _recheck(pins)
            _recheck(staged_pins)
            guard("investment ranking publication")
            report.update(experiment_sha256=experiment_sha256, new_units=new_units,
                aggregate_diagnostics=aggregate_reports, improvement_checks={"cells": cells, "years": years,
                    "primary_both_scopes": {"scopes": primary_scopes,
                        "passes": all(primary_scopes.values()) if None not in primary_scopes.values() else None}},
                fold_scope_fits=len(new_units), final_models=0, completed_at_utc=datetime.now(UTC).isoformat())

        result = _publish(output, report, pins, publish_fits)
        return {**result, "fold_scope_fits": report["fold_scope_fits"], "reused_validation_fits": len(reused_units),
            "final_models": report["final_models"]}
