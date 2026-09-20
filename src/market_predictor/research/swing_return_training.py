"""Resumable sequential training on the verified initial-fit research interval."""
from __future__ import annotations

import importlib.metadata
import os
import platform
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research.swing_return_inputs import ReturnInputs, load_return_inputs
from market_predictor.research.swing_training_readiness import _guard, _verify
from market_predictor.resources import assert_peak_memory_budget, memory_audit, release_process_memory
from market_predictor.swing.contracts.return_training import ReturnTrainingPolicy
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.swing.training.return_artifacts import load_return_model, publish_unit, verify_unit
from market_predictor.swing.training.return_estimators import fit_return_regressor, predict_return
from market_predictor.swing.training.return_validation import date_balanced_weights, fit_indices, regression_diagnostics, return_folds

IMPLEMENTATION = (
    "research/swing_return_training.py", "research/swing_return_inputs.py", "swing/contracts/return_training.py",
    "swing/training/return_estimators.py", "swing/training/return_validation.py", "swing/training/return_artifacts.py",
    "modeling/validation.py", "core/system_memory.py", "resources.py", "process_memory.py", "heavy_jobs.py", "evidence/io.py",
)


def train_swing_returns(*, root: Path, config: Path, config_sha256: str, output: Path,
    resume_checkpoint_sha256: str | None = None) -> dict[str, Any]:
    root = root.resolve()
    config, output = inside(root, config), inside(root, output)
    if not output.is_relative_to(root / "data/research"):
        raise DataReadinessError("return research output must be below data/research, never the serving registry")
    runtime = heavy_job_runtime_dir()
    if not runtime.is_absolute():
        runtime = root / runtime
    with heavy_job_lease("train-swing-returns", runtime_dir=runtime, config_path=config):
        policy = ReturnTrainingPolicy.model_validate(pinned_object(config, config_sha256))
        data = load_return_inputs(root, policy)
        _training_guard(data)
        pins = {**data.pins, config.relative_to(root).as_posix(): config_sha256}
        for name in IMPLEMENTATION:
            path = Path(__file__).parents[1] / name
            pins[path.relative_to(root).as_posix()] = file_sha256(path)
        folds = return_folds(data.sessions, count=policy.folds, minimum_train=policy.minimum_train_sessions,
            embargo=policy.embargo_sessions)
        request = {"schema": "market_predictor.swing_return_training_request", "policy": policy.model_dump(mode="json"),
            "source_files": pins, "feature_names": list(data.features.columns),
            "input_rows": len(data.metadata), "input_decision_ids_sha256": json_sha256(sorted(data.metadata.decision_id)),
            "holdout_security_ids": sorted(data.metadata.loc[data.metadata.security_holdout, "security_id"].unique()),
            "folds": [{"number": fold.number, "train": [day.isoformat() for day in fold.train],
                "embargo": [day.isoformat() for day in fold.embargo], "score": [day.isoformat() for day in fold.score]} for fold in folds],
            "versions": {**{name: importlib.metadata.version(name) for name in
                ("numpy", "pandas", "scikit-learn", "scipy", "xgboost", "joblib", "pyarrow", "threadpoolctl", "exchange-calendars")},
                "python": platform.python_version(), "python_implementation": platform.python_implementation()},
            "memory_policy": {"maximum_system_used_percent": data.memory_policy.maximum_system_used_percent,
                "minimum_system_free_gib": None, "maximum_process_memory_gib": 5.0, "process_headroom_gib": 0.75},
            "serving_eligible": False, "promotion_eligible": False, "outer_validation_opened": False,
            "historical_test_opened": False, "portfolio_evaluated": False}
        _verify(root, pins)
        output.mkdir(parents=True, exist_ok=True)
        request_path = output / "_request.json"
        prior_units: dict[str, Any] = {}
        if request_path.exists():
            if resume_checkpoint_sha256 is None:
                raise DataReadinessError("resuming requires an independently pinned checkpoint SHA256")
            if pinned_object(request_path, file_sha256(request_path)) != request:
                raise DataReadinessError("immutable return research request differs; use a new output directory")
            checkpoint = pinned_object(output / "_checkpoint.json", resume_checkpoint_sha256)
            if checkpoint.get("request_sha256") != file_sha256(request_path):
                raise DataReadinessError("return checkpoint request differs")
            prior_units = checkpoint["units"]
        else:
            if resume_checkpoint_sha256 is not None:
                raise DataReadinessError("cannot resume absent return training request")
            if any(output.iterdir()):
                raise DataReadinessError("return output has files without a bound request")
            write_json_object(request_path, request)
        request_hash = file_sha256(request_path)
        if not (output / "_checkpoint.json").exists():
            _checkpoint(output, request_hash, {})
        records: dict[str, Any] = {}
        for family, parameters in (("regularized_linear_return", policy.linear.model_dump()),
                ("shallow_boosted_return", policy.boosted.model_dump())):
            for fold in folds:
                score_mask = data.metadata.session_date_et.isin(fold.score).to_numpy()
                cutoff = data.metadata.loc[score_mask, "decision_time_utc"].min()
                for scope in ("temporal", "security_transfer"):
                    train = fit_indices(data.metadata, fold.train, cutoff, transfer=scope == "security_transfer")
                    key = f"{family}/{scope}/fold-{fold.number}"
                    records[key] = _fit_unit(root, output / key, data, policy, family, parameters, scope, train,
                        np.flatnonzero(score_mask), request_hash, pins, cutoff.isoformat(), prior_units.pop(key, None))
                    _checkpoint(output, request_hash, {**prior_units, **records})
            train = fit_indices(data.metadata, data.sessions, data.fit_cutoff, transfer=False)
            key = f"{family}/final_refit"
            records[key] = _fit_unit(root, output / key, data, policy, family, parameters, "final_refit", train,
                np.array([], dtype=int), request_hash, pins, data.fit_cutoff.isoformat(), prior_units.pop(key, None))
            _checkpoint(output, request_hash, {**prior_units, **records})
        if prior_units:
            raise DataReadinessError("return checkpoint contains unexpected units")
        _verify(root, pins)
        for record in records.values():
            directory = inside(root, record["path"])
            manifest = pinned_object(directory / "_manifest.json", record["manifest_sha256"])
            verify_unit(directory, manifest_sha256=record["manifest_sha256"], request_sha256=request_hash, unit=manifest["unit"])
        _training_guard(data)
        result = {"schema": "market_predictor.swing_return_training", "status": "complete_research_only",
            "request_sha256": request_hash, "specifications": 2, "fold_scope_fits": 16, "final_models": 2,
            "units": records, "serving_eligible": False, "promotion_eligible": False,
            "portfolio_evaluated": False, "outer_validation_opened": False, "historical_test_opened": False}
        final = output / "_manifest.json"
        if final.exists():
            if pinned_object(final, file_sha256(final)) != result:
                raise DataReadinessError("completed return research manifest differs")
        else:
            _atomic_json(output, "_manifest.json", result)
        return {**result, "manifest_sha256": file_sha256(final)}


def _checkpoint(output: Path, request_hash: str, records: dict[str, Any]) -> None:
    _atomic_json(output, "_checkpoint.json", {"request_sha256": request_hash, "units": records})


def _atomic_json(output: Path, name: str, payload: dict[str, Any]) -> None:
    with tempfile.TemporaryDirectory(dir=output, prefix=".checkpoint-") as temporary:
        staged = Path(temporary) / name
        write_json_object(staged, payload)
        os.replace(staged, output / name)


def _training_guard(data: ReturnInputs) -> None:
    _guard(data.memory_policy)
    assert_peak_memory_budget(stage="swing return training", hard_budget_gib=5.0, headroom_gib=0.75)


def _fit_unit(root: Path, directory: Path, data: ReturnInputs, policy: ReturnTrainingPolicy,
    family: str, parameters: dict[str, Any], scope: str, train: np.ndarray, score: np.ndarray,
    request_hash: str, pins: dict[str, str], cutoff: str, prior: dict[str, Any] | None = None) -> dict[str, Any]:
    _training_guard(data)
    metadata = data.metadata
    scope_eligible = metadata.feature_eligible.to_numpy()[score].copy()
    if scope == "security_transfer":
        scope_eligible &= metadata.security_holdout.to_numpy()[score]
    if len(score) and not scope_eligible.any():
        raise DataReadinessError("research scoring scope has no eligible decisions")
    weights = date_balanced_weights(metadata.iloc[train].session_date_et)
    unit = {"family": family, "scope": scope, "parameters": parameters, "feature_names": list(data.features.columns),
        "fit_cutoff_utc": cutoff, "training_rows": len(train), "training_sessions": metadata.iloc[train].session_date_et.nunique(),
        "maximum_training_label_maturity": metadata.iloc[train].research_label_mature_at.max().isoformat(),
        "training_decision_ids_sha256": json_sha256(sorted(metadata.iloc[train].decision_id)),
        "training_security_ids": sorted(metadata.iloc[train].security_id.unique()),
        "training_weights_sha256": json_sha256(weights.tolist()),
        "scoring_decision_ids_sha256": json_sha256(sorted(metadata.iloc[score].decision_id)),
        "scoring_rows": len(score), "scope_eligible_rows": int(scope_eligible.sum()),
        "target": policy.target, "preprocessing": policy.missingness}
    if directory.exists():
        if (prior is None or prior["path"] != directory.relative_to(root).as_posix()
                or file_sha256(directory / "_manifest.json") != prior["manifest_sha256"]):
            raise DataReadinessError("completed return unit lacks matching pinned checkpoint evidence")
        digest = prior["manifest_sha256"]
        manifest = verify_unit(directory, manifest_sha256=digest, request_sha256=request_hash, unit=unit)
        print(f"Verified completed fit: {directory.relative_to(root)}", flush=True)
    else:
        if prior is not None:
            raise DataReadinessError("return checkpoint references a missing unit")
        print(f"Fitting {directory.relative_to(root)}: {len(train)} rows", flush=True)
        model = fit_return_regressor(family, data.features.iloc[train], metadata.iloc[train][policy.target].to_numpy(dtype=np.float64),
            weights, parameters)
        _training_guard(data)
        predictions = metadata.iloc[score].copy() if len(score) else None
        metrics: dict[str, object] = {}
        if predictions is not None:
            predictions["scope_eligible"] = scope_eligible
            predicted = np.full(len(score), np.nan)
            selected = np.flatnonzero(scope_eligible)
            for start in range(0, len(selected), 8192):
                _training_guard(data)
                positions = selected[start:start + 8192]
                predicted[positions] = predict_return(model, data.features.iloc[score[positions]])
            predictions["predicted_excess_return"] = predicted
            predictions["prediction_status"] = np.where(scope_eligible, "research_prediction", "outside_eligible_scope")
            metrics = regression_diagnostics(predictions, policy.target)
        _verify(root, pins)
        _training_guard(data)
        manifest = publish_unit(directory, request_sha256=request_hash, unit=unit, model=model,
            predictions=predictions, metrics=metrics, guard=lambda: _training_guard(data))
        digest = file_sha256(directory / "_manifest.json")
        loaded = load_return_model(directory, manifest_sha256=digest)
        sample = data.features.iloc[train[:min(32, len(train))]]
        np.testing.assert_array_equal(predict_return(model, sample), predict_return(loaded, sample))
        del model, loaded, predictions
        release_process_memory()
    _training_guard(data)
    print(f"Completed {family}/{scope}; memory {memory_audit(hard_budget_gib=5.0, headroom_gib=0.75).to_record()}", flush=True)
    return {"path": directory.relative_to(root).as_posix(), "manifest_sha256": digest,
        "metrics": manifest["metrics"]}
