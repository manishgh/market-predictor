"""Fixed, sequential investment research fits with actual execution clocks."""
from __future__ import annotations

import importlib.metadata
import json
import os
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import joblib
import numpy as np
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, read_json_object, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.investment.dataset import load_investment_dataset
from market_predictor.investment.ranking import RANKING_SCORE, ranking_diagnostics, within_date_rank_targets
from market_predictor.investment.training import (
    FAMILIES,
    HORIZONS,
    PROFILES,
    SCOPES,
    SCORE_YEARS,
    TARGET,
    InvestmentFold,
    compare_feature_profiles,
    investment_diagnostics,
    investment_fold,
)
from market_predictor.swing.contracts.holding_accounting import HoldingContract
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_training import BoostedReturnSettings, LinearReturnSettings
from market_predictor.swing.training.return_estimators import fit_return_regressor, predict_return
from market_predictor.swing.training.return_validation import date_balanced_weights

FLAGS = {"promotion_eligible": False, "production_eligible": False, "serving_eligible": False,
         "portfolio_evaluated": False, "untouched_test_evidence": False}
RANKING_PROFILE = "within_date_return_rank"
RANKING_TARGET = "within_training_date_centered_average_rank_of_net_excess_vs_spy"


class InvestmentTrainingRequest(HoldingContract):
    schema_version: Literal["market_predictor.investment_training_request"]
    scope: Literal["retrospective_research"]
    dataset_manifest: SourcePin
    dataset_admission: SourcePin
    estimator_settings: SourcePin


def _read(root: Path, pin: SourcePin, sources: dict[str, str]) -> dict[str, Any]:
    path = inside(root, pin.path)
    if path.stat().st_size > 32 * 1024**2 or file_sha256(path) != pin.sha256:
        raise DataReadinessError("investment training input hash or bounded size differs")
    sources[str(path)] = pin.sha256
    return read_json_object(path)


def _implementation() -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    names = ("research/investment_training.py", "investment/training.py", "investment/dataset.py",
        "research/investment_dataset.py", "swing/training/return_estimators.py", "swing/training/return_validation.py",
        "swing/contracts/return_training.py", "core/runtime_memory.py", "core/system_memory.py", "resources.py",
        "heavy_jobs.py")
    return {str(package / name): file_sha256(package / name) for name in names}


def _recheck(sources: dict[str, str]) -> None:
    for index, (name, digest) in enumerate(sources.items()):
        if index % 32 == 0:
            guard("investment training source recheck")
        if file_sha256(Path(name)) != digest:
            raise DataReadinessError(f"investment training source changed: {name}")


def _fold_record(fold: InvestmentFold, metadata: pd.DataFrame) -> dict[str, Any]:
    train, score = metadata.iloc[fold.train_indices], metadata.iloc[fold.score_indices]
    return {"horizon_sessions": fold.horizon_sessions, "score_year": fold.score_year, "scope": fold.scope,
        "information_cutoff_utc": fold.information_cutoff_utc.isoformat(),
        "training_maturity_cutoff_utc": fold.training_maturity_cutoff_utc.isoformat(),
        "training_rows": len(train), "training_months": int(train.decision_session.nunique()),
        "training_security_ids": sorted(train.security_id.unique().tolist()),
        "training_decision_ids_sha256": json_sha256(sorted(train.decision_id.tolist())),
        "scoring_rows": len(score), "feature_eligible_scoring_rows": len(fold.prediction_indices),
        "scoring_decision_ids_sha256": json_sha256(sorted(score.decision_id.tolist())),
        "missing_reasons": list(fold.missing_reasons)}


def _publish_fit(output: Path, key: str, *, features: pd.DataFrame, metadata: pd.DataFrame,
                 columns: tuple[str, ...], family: str, parameters: dict[str, Any], train_indices: np.ndarray,
                 score_indices: np.ndarray, prediction_indices: np.ndarray, identity: dict[str, Any],
                 request_sha256: str, admitted_at: datetime, native_missing_columns: tuple[str, ...] = (), ranking_target: bool = False
                 ) -> tuple[dict[str, Any], pd.DataFrame | None]:
    guard("investment research fit " + key)
    started = datetime.now(UTC)
    train_meta = metadata.iloc[train_indices]
    available = pd.to_datetime(train_meta.label_available_at_utc, utc=True)
    if (not len(train_meta) or available.isna().any() or not available.lt(started).all()
            or pd.Timestamp(admitted_at) >= pd.Timestamp(started)):
        raise DataReadinessError("actual investment fit must follow admission and every training label verification")
    if ranking_target and (family != FAMILIES[1] or native_missing_columns or not len(score_indices)):
        raise DataReadinessError("ranking fits require the frozen boosted validation setup")
    target_values = within_date_rank_targets(train_meta) if ranking_target else train_meta[TARGET].to_numpy(dtype=np.float64)
    model = fit_return_regressor(family, features.iloc[train_indices].loc[:, list(columns)],
        target_values, date_balanced_weights(train_meta.decision_session), parameters,
        **({"native_missing_columns": native_missing_columns} if native_missing_columns else {}))
    fitted = datetime.now(UTC)
    guard("investment research prediction " + key)
    predictions: pd.DataFrame | None = None
    metrics: dict[str, object] = {}
    score_column = RANKING_SCORE if ranking_target else "predicted_excess_return"
    if len(score_indices):
        predictions = metadata.iloc[score_indices].copy()
        predictions[score_column] = np.nan
        if len(prediction_indices):
            values = predict_return(model, features.iloc[prediction_indices].loc[:, list(columns)])
            predictions.loc[prediction_indices, score_column] = values
        generated = datetime.now(UTC)
        predictions["actual_fit_started_at_utc"] = started
        predictions["actual_fitted_at_utc"] = fitted
        predictions["prediction_generated_at_utc"] = generated
        predictions["evidence_scope"] = "retrospective_research"
        predictions["prediction_missing_reason"] = pd.Series(None, index=predictions.index, dtype=object)
        predictions.loc[predictions[score_column].isna(), "prediction_missing_reason"] = "source_features_unavailable"
        metrics = ranking_diagnostics(predictions) if ranking_target else investment_diagnostics(predictions)
    target = output / key
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    schema = "market_predictor.investment_ranking_model" if ranking_target else "market_predictor.investment_research_model"
    record = {"schema": schema, "scope": "retrospective_research",
        "request_sha256": request_sha256, "unit": identity, "family": family, "parameters": parameters,
        "feature_names": list(columns), "actual_fit_started_at_utc": started.isoformat(),
        "actual_fitted_at_utc": fitted.isoformat(), "dataset_admitted_at_utc": admitted_at.isoformat(),
        "latest_training_label_available_at_utc": available.max().isoformat(),
        "latest_training_economic_maturity_at_utc": pd.to_datetime(train_meta.economic_maturity_at_utc, utc=True).max().isoformat(),
        "training_rows": len(train_meta), "training_security_ids": sorted(train_meta.security_id.unique().tolist()),
        "training_decision_ids_sha256": json_sha256(sorted(train_meta.decision_id.tolist())),
        "metrics": metrics, **FLAGS}
    if ranking_target:
        record.update(ranking_target=RANKING_TARGET, score_column=RANKING_SCORE, score_units="dimensionless",
            training_rank_targets_sha256=json_sha256(list(zip(train_meta.decision_id, target_values.tolist(), strict=True))))
    if native_missing_columns:
        record["native_missing_features"] = [asdict(feature) for feature in model.native_missing_features]
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".investment-fit-") as temporary:
        staged = Path(temporary) / "artifact"
        staged.mkdir()
        joblib.dump(model, staged / "model.joblib", compress=3)
        if predictions is not None:
            predictions.to_parquet(staged / "predictions.parquet", index=False)
        # Check persistence on actual consumed rows, without admitting a serving model.
        restored = joblib.load(staged / "model.joblib")
        probe = features.iloc[train_indices[:min(8, len(train_indices))]].loc[:, list(columns)]
        if not np.array_equal(predict_return(model, probe), predict_return(restored, probe)):
            raise DataReadinessError("investment model persistence changes predictions")
        record["persistence_prediction_parity"] = True
        record["files"] = {path.name: file_sha256(path) for path in staged.iterdir()}
        write_json_object(staged / "_manifest.json", record)
        guard("investment model publication")
        os.rename(staged, target)
    print(json.dumps({"event": "investment_fit_complete", "unit": key, "training_rows": len(train_meta),
        "scoring_rows": 0 if predictions is None else len(predictions), "actual_fitted_at_utc": fitted.isoformat()}), flush=True)
    return {"path": str(target), "sha256": file_sha256(target / "_manifest.json"), **record}, predictions


def train_investment_models(*, root: Path, request: Path, request_sha256: str, output: Path) -> dict[str, Any]:
    """Run only the frozen two-profile/two-learner experiment after dataset admission."""
    root = root.resolve()
    output = inside(root, output)
    runtime = heavy_job_runtime_dir()
    with heavy_job_lease("train-investment-models", runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("investment research training preparation")
        sources = _implementation()
        parsed = InvestmentTrainingRequest.model_validate_json(json.dumps(_read(root,
            SourcePin(path=str(request), sha256=request_sha256), sources)))
        settings = _read(root, parsed.estimator_settings, sources)
        parameters = {
            FAMILIES[0]: LinearReturnSettings.model_validate_json(json.dumps(settings["linear"])).model_dump(),
            FAMILIES[1]: BoostedReturnSettings.model_validate_json(json.dumps(settings["boosted"])).model_dump(),
        }
        dataset = load_investment_dataset(root=root, manifest=parsed.dataset_manifest, admission=parsed.dataset_admission)
        sources.update({str((root / name).resolve()): digest for name, digest in dataset.source_files.items()})
        metadata = dataset.metadata.reset_index(drop=True)
        features = dataset.features.reset_index(drop=True)
        if (len(metadata) != len(features) or metadata.duplicated(["decision_id", "horizon_sessions"]).any()
                or any(metadata[name].dtype.kind != "b" for name in ("feature_eligible", "research_eligible"))):
            raise DataReadinessError("investment dataset row alignment or eligibility types differ")
        if set(dataset.model_columns) != set(PROFILES):
            raise DataReadinessError("investment dataset feature profiles differ")
        folds = [investment_fold(metadata, horizon, year, cast(Any, scope))
            for horizon in HORIZONS for scope in SCOPES for year in SCORE_YEARS]
        fold_records = [_fold_record(fold, metadata) for fold in folds]
        horizon_allowed = {horizon: all(not fold.missing_reasons for fold in folds
            if fold.horizon_sessions == horizon and fold.scope == "temporal") for horizon in HORIZONS}
        experiment = {"schema": "market_predictor.investment_training_experiment", "request_sha256": request_sha256,
            "scope": "retrospective_research", "feature_contract_sha256": dataset.feature_contract_sha256,
            "model_columns": dataset.model_columns, "parameters": parameters, "folds": fold_records,
            "primary_horizon_sessions": 252, "diagnostic_horizon_sessions": 63,
            "embargo_exchange_sessions": 21, "minimum_training_months": 12, "minimum_training_rows": 100,
            "minimum_training_securities": 20, "minimum_scoring_months": 3, "minimum_known_scoring_rows": 30,
            "minimum_known_scoring_securities": 10, "holdout_fraction": 0.2,
            "holdout_assignment": "unseeded_sha256_security_id_first_64_bits",
            "selection": "ceil(scored_count/10)_descending_prediction_ascending_security_id_before_outcomes",
            "missingness": "train_partition_median_all_empty_zero_with_all_column_indicators",
            "weighting": "equal_total_weight_per_decision_date", "source_files": sources,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dependencies": {name: importlib.metadata.version(name)
                for name in ("numpy", "pandas", "scikit-learn", "xgboost", "joblib", "exchange-calendars")}, **FLAGS}
        if output.exists():
            raise FileExistsError(output)
        output.mkdir(parents=True)
        write_json_object(output / "experiment.json", experiment)
        experiment_sha = file_sha256(output / "experiment.json")
        units: dict[str, Any] = {}
        aggregates: dict[str, Any] = {}
        for horizon in HORIZONS:
            if not horizon_allowed[horizon]:
                continue
            for scope in SCOPES:
                for profile in PROFILES:
                    columns = tuple(dataset.model_columns[profile])
                    for family in FAMILIES:
                        collected: list[pd.DataFrame] = []
                        for fold in folds:
                            if fold.horizon_sessions != horizon or fold.scope != scope or fold.missing_reasons:
                                continue
                            key = f"{horizon}/{profile}/{family}/{scope}/{fold.score_year}"
                            identity = {**_fold_record(fold, metadata), "profile": profile,
                                "experiment_sha256": experiment_sha}
                            record, prediction = _publish_fit(output, key, features=features, metadata=metadata,
                                columns=columns, family=family, parameters=parameters[family], train_indices=fold.train_indices,
                                score_indices=fold.score_indices, prediction_indices=fold.prediction_indices,
                                identity=identity, request_sha256=request_sha256, admitted_at=dataset.admitted_at_utc)
                            units[key] = {"path": str(output / key / "_manifest.json"), "sha256": record["sha256"]}
                            if prediction is not None:
                                collected.append(prediction)
                        if collected:
                            combined = pd.concat(collected, ignore_index=True)
                            aggregates[f"{horizon}/{profile}/{family}/{scope}"] = investment_diagnostics(combined)
            train_indices = np.flatnonzero((metadata.horizon_sessions.eq(horizon) & metadata.research_eligible).to_numpy())
            for profile in PROFILES:
                for family in FAMILIES:
                    key = f"{horizon}/{profile}/{family}/final_refit"
                    record, _ = _publish_fit(output, key, features=features, metadata=metadata,
                        columns=tuple(dataset.model_columns[profile]), family=family, parameters=parameters[family],
                        train_indices=train_indices, score_indices=np.array([], dtype=int),
                        prediction_indices=np.array([], dtype=int),
                        identity={"horizon_sessions": horizon, "profile": profile, "scope": "final_refit",
                            "experiment_sha256": experiment_sha},
                        request_sha256=request_sha256, admitted_at=dataset.admitted_at_utc)
                    units[key] = {"path": str(output / key / "_manifest.json"), "sha256": record["sha256"]}
        _recheck(sources)
        report = {"schema": "market_predictor.investment_training", "scope": "retrospective_research",
            "status": "complete_research_only" if all(horizon_allowed.values()) else "insufficient_temporal_coverage",
            "request_sha256": request_sha256, "experiment_sha256": experiment_sha, "units": units,
            "aggregate_metrics": aggregates, "paired_feature_comparisons": compare_feature_profiles(aggregates),
            "folds": fold_records, "source_files": sources,
            "horizon_training_allowed": horizon_allowed, "completed_at_utc": datetime.now(UTC).isoformat(),
            "final_models": sum(key.endswith("final_refit") for key in units),
            "fold_scope_fits": sum(not key.endswith("final_refit") for key in units),
            "selection_notice": "Forecast selection precedes outcome matching; incomplete selected cohorts have no complete return.",
            "evidence_notice": "Reconstructed historical development; actual models fitted now. Overlapping outcomes are not independent.",
            **FLAGS}
        write_json_object(output / "_manifest.json", report)
        return {"status": report["status"], "output": str(output),
            "manifest_sha256": file_sha256(output / "_manifest.json"),
            "final_models": report["final_models"], "fold_scope_fits": report["fold_scope_fits"], **FLAGS}
