"""Explain all saved investment validation units without refitting or raw replay."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.path_integrity import verify_no_reparse_ancestry
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.io import inside
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.investment.dataset import (
    InvestmentDataset,
    add_pins,
    load_investment_dataset,
    read_object,
    require,
)
from market_predictor.investment.diagnostics import (
    CONTEXT,
    COVERAGE,
    PREDICTION,
    feature_coverage,
    paired_diagnostics,
    prediction_diagnostics,
)
from market_predictor.investment.training import FAMILIES, HORIZONS, PROFILES, SCOPES, SCORE_YEARS, InvestmentFold, investment_fold
from market_predictor.research.investment_features import _publish
from market_predictor.research.investment_training import FLAGS, InvestmentTrainingRequest, _fold_record
from market_predictor.swing.contracts.holding_accounting import HoldingContract
from market_predictor.swing.contracts.holding_materialization import SourcePin

CLOCKS = ("actual_fit_started_at_utc", "actual_fitted_at_utc", "prediction_generated_at_utc")
PREDICTION_COLUMNS = (*CONTEXT, PREDICTION, *CLOCKS, "evidence_scope", "prediction_missing_reason")


class InvestmentDiagnosticRequest(HoldingContract):
    schema_version: Literal["market_predictor.investment_diagnostics_request"]
    scope: Literal["saved_retrospective_predictions"]
    training_manifest: SourcePin
    training_request: SourcePin


def _implementation() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {str(package / name): file_sha256(package / name) for name in (
        "investment/diagnostics.py", "research/investment_diagnostics.py", "investment/training.py", "investment/dataset.py",
        "research/investment_training.py", "swing/training/return_validation.py", "research/investment_features.py",
        "commands/investment_research.py", "canonical/store.py", "core/runtime_memory.py", "resources.py", "heavy_jobs.py")}


def _recheck(pins: dict[str, str], *, paths: EvidencePaths | None = None, implementation: frozenset[str] = frozenset()) -> None:
    for index, (name, digest) in enumerate(pins.items()):
        if index % 32 == 0:
            guard("investment diagnostic source recheck")
        source = paths.read(name) if paths is not None and name not in implementation else Path(name)
        if name in implementation:
            source = verify_no_reparse_ancestry(source, label="investment implementation")
        require(file_sha256(source) == digest, f"investment diagnostic source changed: {name}")


def read_fit_predictions(root: Path, pin: SourcePin, record: dict[str, Any], expected: pd.DataFrame,
    coverage: pd.DataFrame, pins: dict[str, str], completed_at: pd.Timestamp, *, score_column: str = PREDICTION,
    paths: EvidencePaths | None = None) -> pd.DataFrame:
    identity = (inside(root, pin.path) if paths is None else paths.identity(pin.path)).parent / "predictions.parquet"
    path = identity if paths is None else paths.read(identity)
    digest = record["files"]["predictions.parquet"]
    require(path.stat().st_size <= 64 * 1024**2 and file_sha256(path) == digest,
        "investment saved predictions exceed size bound or changed")
    add_pins(root, pins, {str(identity): digest}, paths=paths)
    frame = pd.read_parquet(path, columns=[score_column if name == PREDICTION else name for name in PREDICTION_COLUMNS])
    require(len(frame) == len(expected) and not frame.decision_id.duplicated().any(),
        "investment saved prediction population differs from its frozen fold")
    try:
        pd.testing.assert_frame_equal(frame.loc[:, list(CONTEXT)], expected.loc[:, list(CONTEXT)].reset_index(drop=True),
            check_exact=True, check_dtype=False)
    except AssertionError as error:
        from market_predictor.core.errors import DataReadinessError

        raise DataReadinessError("investment saved prediction identities/outcomes differ from the admitted dataset") from error
    require(frame[score_column].notna().equals(frame.feature_eligible)
        and frame.evidence_scope.eq("retrospective_research").all(), "investment saved prediction mask or scope differs")
    started, fitted = pd.Timestamp(record["actual_fit_started_at_utc"]), pd.Timestamp(record["actual_fitted_at_utc"])
    require(started.tzinfo is not None and fitted.tzinfo is not None and started <= fitted
        and pd.Timestamp(record["dataset_admitted_at_utc"]) < started
        and pd.Timestamp(record["latest_training_label_available_at_utc"]) < started
        and fitted <= completed_at
        and pd.to_datetime(frame[CLOCKS[0]], utc=True).eq(started).all()
        and pd.to_datetime(frame[CLOCKS[1]], utc=True).eq(fitted).all()
        and pd.to_datetime(frame[CLOCKS[2]], utc=True).ge(fitted).all()
        and pd.to_datetime(frame[CLOCKS[2]], utc=True).le(completed_at).all(), "investment actual fit/score clocks differ")
    require(frame.loc[frame.feature_eligible, "prediction_missing_reason"].isna().all()
        and frame.loc[~frame.feature_eligible, "prediction_missing_reason"].eq("source_features_unavailable").all(),
        "investment saved prediction reasons differ")
    for name in COVERAGE:
        frame[name] = coverage.loc[expected.index, name].to_numpy()
    return frame


@dataclass(frozen=True)
class InvestmentReferenceRun:
    manifest: dict[str, Any]
    experiment: dict[str, Any]
    request: InvestmentTrainingRequest
    request_sha256: str
    completed_at: pd.Timestamp


def read_reference_run(root: Path, manifest: SourcePin, request: SourcePin,
                       pins: dict[str, str], *, paths: EvidencePaths | None = None) -> InvestmentReferenceRun:
    run = read_object(root, manifest, pins, paths=paths)
    train_request = InvestmentTrainingRequest.model_validate_json(json.dumps(read_object(root, request, pins, paths=paths)))
    require(run.get("schema") == "market_predictor.investment_training" and run.get("status") == "complete_research_only"
        and run.get("scope") == "retrospective_research" and all(run.get(flag) is False for flag in FLAGS)
        and run["request_sha256"] == request.sha256 and run["fold_scope_fits"] == 48 and run["final_models"] == 8,
        "investment diagnostics require the complete closed 48-plus-8 research run")
    completed_at = pd.Timestamp(run.get("completed_at_utc"))
    require(not pd.isna(completed_at) and completed_at.tzinfo is not None
        and completed_at <= pd.Timestamp(datetime.now(UTC)), "investment closed run completion clock is absent or in the future")
    run_path = inside(root, manifest.path) if paths is None else paths.identity(manifest.path)
    experiment_pin = SourcePin(path=str(run_path.parent / "experiment.json"), sha256=run["experiment_sha256"])
    experiment = read_object(root, experiment_pin, pins, paths=paths)
    require(experiment["request_sha256"] == request.sha256 and all(experiment.get(flag) is False for flag in FLAGS),
        "investment diagnostic experiment identity or scope differs")
    return InvestmentReferenceRun(run, experiment, train_request, request.sha256, completed_at)


def reference_folds(reference: InvestmentReferenceRun, dataset: InvestmentDataset) -> dict[tuple[int, str, int], InvestmentFold]:
    run, experiment = reference.manifest, reference.experiment
    metadata = dataset.metadata.reset_index(drop=True)
    require(dataset.features.index.equals(metadata.index) and dataset.model_columns == {
        key: tuple(value) for key, value in experiment["model_columns"].items()}
        and dataset.feature_contract_sha256 == experiment["feature_contract_sha256"], "investment diagnostic feature order differs")
    fold_keys = {(h, scope, year) for h in HORIZONS for scope in SCOPES for year in SCORE_YEARS}
    folds = {key: investment_fold(metadata, key[0], key[2], cast(Any, key[1])) for key in fold_keys}
    require(all(not fold.missing_reasons for fold in folds.values()), "investment diagnostic original fold is unavailable")
    fold_records = [_fold_record(folds[(h, scope, year)], metadata) for h in HORIZONS for scope in SCOPES for year in SCORE_YEARS]
    require(run["folds"] == experiment["folds"] == fold_records, "investment diagnostic frozen split inventory differs")
    expected_aggregates = {f"{h}/{profile}/{family}/{scope}" for h in HORIZONS for scope in SCOPES
        for profile in PROFILES for family in FAMILIES}
    require(set(run["aggregate_metrics"]) == expected_aggregates, "investment diagnostic aggregate inventory differs")
    expected_keys = {f"{h}/{profile}/{family}/{scope}/{year}" for h, scope, year in fold_keys
        for profile in PROFILES for family in FAMILIES}
    expected_keys |= {f"{h}/{profile}/{family}/final_refit" for h in HORIZONS for profile in PROFILES for family in FAMILIES}
    require(set(run["units"]) == expected_keys, "investment diagnostic unit inventory is not the exact closed 56")
    return folds


def read_reference_predictions(root: Path, reference: InvestmentReferenceRun, key: str, fold: InvestmentFold,
                               dataset: InvestmentDataset, metadata: pd.DataFrame, coverage: pd.DataFrame,
                               pins: dict[str, str]) -> tuple[SourcePin, dict[str, Any], pd.DataFrame]:
    _, profile, _, _, _ = key.split("/")
    pin = SourcePin.model_validate_json(json.dumps(reference.manifest["units"][key]))
    record, frame = read_validation_predictions(root, reference, pin, key, fold, dataset, metadata,
        coverage, pins, dataset.model_columns[profile])
    return pin, record, frame


def read_validation_predictions(root: Path, reference: InvestmentReferenceRun, pin: SourcePin, key: str,
                                fold: InvestmentFold, dataset: InvestmentDataset, metadata: pd.DataFrame,
                                coverage: pd.DataFrame, pins: dict[str, str],
                                columns: tuple[str, ...], *, paths: EvidencePaths | None = None) -> tuple[dict[str, Any], pd.DataFrame]:
    run, experiment = reference.manifest, reference.experiment
    _, profile, family, _, _ = key.split("/")
    record = read_object(root, pin, pins, paths=paths)
    identity = {**_fold_record(fold, metadata), "profile": profile, "experiment_sha256": run["experiment_sha256"]}
    require(record.get("schema") == "market_predictor.investment_research_model"
        and record.get("scope") == "retrospective_research" and record["unit"] == identity
        and record["request_sha256"] == reference.request_sha256 and record["family"] == family
        and record["parameters"] == experiment["parameters"][family]
        and tuple(record["feature_names"]) == columns
        and set(record["files"]) == {"model.joblib", "predictions.parquet"}
        and all(record.get(flag) is False for flag in FLAGS), "investment saved model unit binding differs")
    train = metadata.iloc[fold.train_indices]
    require(record["training_rows"] == len(train)
        and record["training_decision_ids_sha256"] == identity["training_decision_ids_sha256"]
        and record["training_security_ids"] == identity["training_security_ids"]
        and pd.Timestamp(record["dataset_admitted_at_utc"]) == dataset.admitted_at_utc
        and pd.Timestamp(record["latest_training_label_available_at_utc"]) == pd.to_datetime(
            train.label_available_at_utc, utc=True).max()
        and pd.Timestamp(record["latest_training_economic_maturity_at_utc"]) == pd.to_datetime(
            train.economic_maturity_at_utc, utc=True).max(), "investment saved training lineage differs")
    frame = read_fit_predictions(root, pin, record, metadata.iloc[fold.score_indices], coverage, pins, reference.completed_at, paths=paths)
    return record, frame


def diagnose_investment_predictions(*, root: Path, request: Path, request_sha256: str, output: Path) -> dict[str, Any]:
    root = root.resolve()
    output = inside(root, output)
    runtime = heavy_job_runtime_dir()
    with heavy_job_lease("diagnose-investment-predictions", runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("investment prediction diagnostic preparation")
        pins = _implementation()
        parsed = InvestmentDiagnosticRequest.model_validate_json(json.dumps(read_object(root,
            SourcePin(path=str(request), sha256=request_sha256), pins)))
        reference = read_reference_run(root, parsed.training_manifest, parsed.training_request, pins)
        run, train_request = reference.manifest, reference.request
        dataset = load_investment_dataset(root=root, manifest=train_request.dataset_manifest, admission=train_request.dataset_admission)
        add_pins(root, pins, dataset.source_files)
        metadata = dataset.metadata.reset_index(drop=True)
        folds = reference_folds(reference, dataset)
        coverage = feature_coverage(dataset.features, dataset.model_columns["all_features"])
        unit_reports, aggregate_reports, paired_units, paired_aggregates = {}, {}, {}, {}
        for horizon in HORIZONS:
            for scope in SCOPES:
                for family in FAMILIES:
                    paired: dict[str, list[pd.DataFrame]] = {profile: [] for profile in PROFILES}
                    for year in SCORE_YEARS:
                        guard("investment saved prediction unit")
                        fold = folds[(horizon, scope, year)]
                        frames = {}
                        for profile in PROFILES:
                            key = f"{horizon}/{profile}/{family}/{scope}/{year}"
                            _, record, frame = read_reference_predictions(
                                root, reference, key, fold, dataset, metadata, coverage, pins)
                            diagnostics = prediction_diagnostics(frame)
                            require(diagnostics["canonical_metrics"] == record["metrics"],
                                "investment canonical saved unit metrics do not replay exactly")
                            unit_reports[key] = diagnostics
                            frames[profile] = frame
                            paired[profile].append(frame)
                        paired_units[f"{horizon}/{family}/{scope}/{year}"] = paired_diagnostics(
                            frames[PROFILES[0]], frames[PROFILES[1]])
                    for profile in PROFILES:
                        key = f"{horizon}/{profile}/{family}/{scope}"
                        combined = pd.concat(paired[profile], ignore_index=True)
                        report = prediction_diagnostics(combined)
                        require(report["canonical_metrics"] == run["aggregate_metrics"][key],
                            "investment canonical aggregate metrics do not replay exactly")
                        aggregate_reports[key] = report
                    paired_aggregates[f"{horizon}/{family}/{scope}"] = paired_diagnostics(
                        pd.concat(paired[PROFILES[0]], ignore_index=True), pd.concat(paired[PROFILES[1]], ignore_index=True))
        require(len(unit_reports) == 48 and len(aggregate_reports) == 16 and len(paired_units) == 24
            and len(paired_aggregates) == 8, "investment diagnostic output inventory differs")
        _recheck(pins)
        report = {"schema": "market_predictor.investment_diagnostics", "status": "saved_prediction_diagnostics_complete",
            "scope": "saved_retrospective_predictions", "request_sha256": request_sha256,
            "training_manifest": parsed.training_manifest.model_dump(mode="json"), "unit_diagnostics": unit_reports,
            "aggregate_diagnostics": aggregate_reports,
            "paired_diagnostics": {"units": paired_units, "aggregates": paired_aggregates},
            "source_files": pins, "producer_source_provenance": run["source_files"],
            "models_loaded": False, "models_refitted": False, "saved_model_bytes_rechecked": False,
            "predictions_modified": False, "raw_sources_replayed": False,
            "calibrator_fitted": False, "error_slope_is_descriptive_only": True,
            "feature_attribution_notice": "Joint news-plus-financial comparison cannot isolate the individual source effects.",
            **FLAGS}
        return _publish(output, report, pins)
