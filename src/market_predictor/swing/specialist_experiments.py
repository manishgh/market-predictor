"""Resumable immutable experiment bundles for KS3 swing specialists."""

from __future__ import annotations

import gc
import json
import math
import os
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd

from market_predictor.canonical.store import (
    canonical_artifact_columns,
    file_sha256,
    load_canonical_artifact,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.swing.specialist_contracts import (
    SPECIALIST_DATASET_BUNDLE_SCHEMA,
    SPECIALIST_EVIDENCE_SCHEMA,
    SPECIALIST_MODEL_SCHEMA,
    SwingSpecialistResearchConfig,
)
from market_predictor.swing.specialist_model import (
    SPECIALIST_ACCEPTED_STATUS,
    SpecialistExperimentResult,
    SpecialistExperimentSpec,
    SpecialistSplitPlan,
    build_specialist_split_plan,
    evaluate_specialist_experiment,
    specialist_experiment_specs,
)
from market_predictor.swing.strategy_labels import STRATEGY_IDS
from market_predictor.v3.errors import DataReadinessError

SPECIALIST_EXPERIMENT_BUNDLE_SCHEMA = "swing.specialist_experiment_bundle.v1"
SPECIALIST_CANDIDATE_MANIFEST_SCHEMA = "swing.specialist_candidate.v1"
PEAD_STRATEGY_ID = "SWING.POST_EARNINGS_ANNOUNCEMENT_DRIFT.5D.V1"

_TRAINING_BASE_COLUMNS = {
    "strategy_id",
    "strategy_target",
    "strategy_label_eligible",
    "strategy_horizon_sessions",
    "strategy_dataset_row_id",
    "ticker",
    "security_id",
    "session_date_et",
    "decision_group_id",
    "decision_time_utc",
    "feature_available_at_utc",
    "label_available_at_utc",
    "entry_time_utc",
    "exit_time_utc",
    "universe_snapshot_id",
    "market_regime",
    "sector",
    "primary_benchmark",
    "close",
    "atr_pct_14",
    "dollar_volume",
    "strategy_gross_return",
    "strategy_execution_cost_fraction",
    "strategy_net_return",
    "strategy_spy_return",
    "strategy_qqq_return",
    "strategy_sector_return",
    "strategy_excess_return_vs_spy",
    "strategy_excess_return_vs_qqq",
    "strategy_excess_return_vs_sector",
    "strategy_mfe",
    "strategy_mae",
    "event_count_3d",
    "event_relevance_mean_3d",
    "low_relevance_event_fraction_3d",
}


def specialist_training_columns(
    path: Path,
    *,
    strategy_id: str,
    config: SwingSpecialistResearchConfig,
) -> tuple[str, ...]:
    available = canonical_artifact_columns(path)
    profile_names = config.strategies[strategy_id].feature_profiles
    configured_features = {
        feature
        for profile in profile_names
        for feature in config.feature_profiles[profile]
    }
    selected = {
        *_TRAINING_BASE_COLUMNS,
        *configured_features,
    }
    columns = tuple(
        column
        for column in available
        if column in selected
        or column.startswith(
            (
                "future_gross_return_",
                "future_net_return_",
                "future_spy_return_",
                "future_qqq_return_",
                "future_sector_return_",
                "future_excess_return_",
                "target_net_positive_",
            )
        )
    )
    missing = sorted(_TRAINING_BASE_COLUMNS.difference(columns))
    if missing:
        raise DataReadinessError(
            f"{strategy_id} training artifact is missing: "
            + ", ".join(missing)
        )
    return columns


def train_swing_specialist_experiments(
    *,
    dataset_dir: Path,
    out_dir: Path,
    config: SwingSpecialistResearchConfig,
    policy_path: Path,
    strategy_ids: Sequence[str] | None = None,
    progress: Callable[[object], None] | None = None,
) -> dict[str, object]:
    dataset_manifest_path = dataset_dir / "_manifest.json"
    dataset_manifest = _load_dataset_manifest(dataset_manifest_path)
    artifacts = {
        str(record["strategy_id"]): {
            str(key): value for key, value in record.items()
        }
        for record in _artifact_records(dataset_manifest)
    }
    if set(artifacts) != set(STRATEGY_IDS):
        raise DataReadinessError(
            "specialist dataset bundle does not contain the frozen catalog"
        )
    request = {
        "schema": SPECIALIST_EXPERIMENT_BUNDLE_SCHEMA,
        "dataset_manifest_path": str(dataset_manifest_path.resolve()),
        "dataset_manifest_sha256": file_sha256(dataset_manifest_path),
        "dataset_request_sha256": str(
            dataset_manifest["request_sha256"]
        ),
        "specialist_research_policy_sha256": config.sha256(),
        "specialist_research_policy_file_sha256": file_sha256(policy_path),
        "strategy_ids": list(STRATEGY_IDS),
    }
    request_sha256 = _json_sha256(request)
    request["request_sha256"] = request_sha256
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(out_dir / "_request.json", request)

    selected_strategy_ids = (
        tuple(STRATEGY_IDS)
        if strategy_ids is None
        else tuple(strategy_ids)
    )
    unknown = sorted(set(selected_strategy_ids).difference(STRATEGY_IDS))
    if not selected_strategy_ids or unknown:
        raise DataReadinessError(
            "invalid specialist strategy selection: "
            + ", ".join(unknown or ["empty"])
        )
    records: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    stop_for_memory = False
    for strategy_id in selected_strategy_ids:
        strategy_record = artifacts[strategy_id]
        path = Path(str(strategy_record["path"]))
        expected_sha256 = str(strategy_record["sha256"])
        dataset: pd.DataFrame | None = None
        try:
            dataset, manifest = load_canonical_artifact(
                path,
                expected_type="swing_specialist_dataset",
                allow_research=True,
                columns=specialist_training_columns(
                    path,
                    strategy_id=strategy_id,
                    config=config,
                ),
            )
            if str(manifest["artifact_sha256"]) != expected_sha256:
                raise DataReadinessError(
                    f"{strategy_id} dataset hash does not match bundle"
                )
            plan = build_specialist_split_plan(
                dataset,
                strategy_id=strategy_id,
                config=config,
            )
            strategy_result = _run_strategy_experiments(
                plan=plan,
                dataset_sha256=expected_sha256,
                out_dir=out_dir / "strategies" / _slug(strategy_id),
                config=config,
                request_sha256=request_sha256,
                progress=progress,
            )
            records.append(strategy_result)
        except (
            DataReadinessError,
            FileNotFoundError,
            ImportError,
            OSError,
            ValueError,
        ) as exc:
            failures[strategy_id] = f"{type(exc).__name__}: {exc}"
            if progress is not None:
                progress(
                    {
                        "strategy_id": strategy_id,
                        "status": "failed",
                        "error": failures[strategy_id],
                    }
                )
        finally:
            dataset = None
            gc.collect()
            release_process_memory()
            try:
                assert_memory_budget(
                    hard_budget_gib=config.maximum_process_memory_gib,
                    headroom_gib=config.memory_guard_headroom_gib,
                    stage=f"KS3 strategy experiments {strategy_id}",
                )
            except DataReadinessError as exc:
                failures[strategy_id] = str(exc)
                stop_for_memory = True
            _write_experiment_status(
                out_dir,
                request_sha256=request_sha256,
                records=records,
                failures=failures,
                config=config,
                complete=False,
                requested_strategies=selected_strategy_ids,
            )
        if stop_for_memory:
            break

    try:
        assert_peak_memory_budget(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
            stage="KS3 specialist experiments",
        )
    except DataReadinessError as exc:
        failures["memory_budget"] = str(exc)
    status = (
        "complete"
        if not failures and len(records) == len(STRATEGY_IDS)
        else "incomplete"
    )
    result = _write_experiment_status(
        out_dir,
        request_sha256=request_sha256,
        records=records,
        failures=failures,
        config=config,
        complete=status == "complete",
        requested_strategies=selected_strategy_ids,
    )
    if status == "complete":
        _atomic_json(out_dir / "_manifest.json", result)
    return result


def _run_strategy_experiments(
    *,
    plan: SpecialistSplitPlan,
    dataset_sha256: str,
    out_dir: Path,
    config: SwingSpecialistResearchConfig,
    request_sha256: str,
    progress: Callable[[object], None] | None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    candidate_records: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    stop_for_memory = False
    for spec in specialist_experiment_specs(plan.strategy_id, config):
        candidate_dir = out_dir / "candidates" / _slug(spec.candidate_id)
        candidate_request = _candidate_request(
            spec,
            dataset_sha256=dataset_sha256,
            split_sha256=plan.split_sha256,
            bundle_request_sha256=request_sha256,
            config=config,
        )
        try:
            existing = _load_existing_candidate(
                candidate_dir,
                expected_request_sha256=str(
                    candidate_request["request_sha256"]
                ),
            )
            if existing is not None:
                candidate_records.append(existing)
                if progress is not None:
                    progress(
                        {
                            "strategy_id": plan.strategy_id,
                            "candidate_id": spec.candidate_id,
                            "status": "resumed",
                        }
                    )
                continue
            result = evaluate_specialist_experiment(
                plan,
                spec,
                config=config,
            )
            candidate_record = _write_candidate_evidence(
                candidate_dir,
                result=result,
                request=candidate_request,
                dataset_sha256=dataset_sha256,
                config=config,
            )
            candidate_records.append(candidate_record)
            if progress is not None:
                progress(
                    {
                        "strategy_id": plan.strategy_id,
                        "candidate_id": spec.candidate_id,
                        "status": result.status,
                        "rejection_reasons": list(
                            result.rejection_reasons
                        ),
                    }
                )
            del result
        except (
            DataReadinessError,
            FileNotFoundError,
            ImportError,
            OSError,
            ValueError,
        ) as exc:
            failures[spec.candidate_id] = f"{type(exc).__name__}: {exc}"
            if progress is not None:
                progress(
                    {
                        "strategy_id": plan.strategy_id,
                        "candidate_id": spec.candidate_id,
                        "status": "failed",
                        "error": failures[spec.candidate_id],
                    }
                )
        finally:
            gc.collect()
            release_process_memory()
            try:
                assert_memory_budget(
                    hard_budget_gib=config.maximum_process_memory_gib,
                    headroom_gib=config.memory_guard_headroom_gib,
                    stage=(
                        f"KS3 candidate {plan.strategy_id} "
                        f"{spec.candidate_id}"
                    ),
                )
            except DataReadinessError as exc:
                failures[spec.candidate_id] = str(exc)
                stop_for_memory = True
        if stop_for_memory:
            break
    status = (
        "complete"
        if not failures
        and len(candidate_records)
        == len(specialist_experiment_specs(plan.strategy_id, config))
        else "incomplete"
    )
    record: dict[str, object] = {
        "strategy_id": plan.strategy_id,
        "status": status,
        "dataset_sha256": dataset_sha256,
        "split_sha256": plan.split_sha256,
        "candidate_count": len(candidate_records),
        "accepted_development_count": sum(
            str(candidate["status"]) == SPECIALIST_ACCEPTED_STATUS
            for candidate in candidate_records
        ),
        "rejected_count": sum(
            str(candidate["status"]) == "rejected"
            for candidate in candidate_records
        ),
        "failed_candidates": failures,
        "candidates": sorted(
            candidate_records,
            key=lambda candidate: str(candidate["candidate_id"]),
        ),
    }
    _atomic_json(out_dir / "_status.json", record)
    if status == "complete":
        _atomic_json(out_dir / "_manifest.json", record)
    if failures:
        raise DataReadinessError(
            f"{plan.strategy_id} candidate matrix is incomplete"
        )
    return record


def _candidate_request(
    spec: SpecialistExperimentSpec,
    *,
    dataset_sha256: str,
    split_sha256: str,
    bundle_request_sha256: str,
    config: SwingSpecialistResearchConfig,
) -> dict[str, object]:
    request: dict[str, object] = {
        "schema": SPECIALIST_CANDIDATE_MANIFEST_SCHEMA,
        "strategy_id": spec.strategy_id,
        "candidate_id": spec.candidate_id,
        "estimator_family": spec.estimator_family,
        "feature_profile": (
            spec.feature_profile or "deterministic_score"
        ),
        "deterministic_score": spec.deterministic_score,
        "dataset_sha256": dataset_sha256,
        "split_sha256": split_sha256,
        "bundle_request_sha256": bundle_request_sha256,
        "specialist_research_policy_sha256": config.sha256(),
    }
    request["request_sha256"] = _json_sha256(request)
    return request


def _write_candidate_evidence(
    out_dir: Path,
    *,
    result: SpecialistExperimentResult,
    request: Mapping[str, object],
    dataset_sha256: str,
    config: SwingSpecialistResearchConfig,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(out_dir / "_request.json", request)
    evidence_paths = {
        "predictions": out_dir / "predictions.parquet",
        "economics": out_dir / "economics.parquet",
        "regime_evidence": out_dir / "regime_evidence.parquet",
        "capacity_evidence": out_dir / "capacity_evidence.parquet",
        "fold_audit": out_dir / "fold_audit.parquet",
        "metrics": out_dir / "metrics.json",
    }
    _atomic_parquet(result.predictions, evidence_paths["predictions"])
    _atomic_parquet(result.economics, evidence_paths["economics"])
    _atomic_parquet(
        result.regime_evidence,
        evidence_paths["regime_evidence"],
    )
    _atomic_parquet(
        result.capacity_evidence,
        evidence_paths["capacity_evidence"],
    )
    _atomic_parquet(result.fold_audit, evidence_paths["fold_audit"])
    _atomic_json(evidence_paths["metrics"], result.metrics)
    model_path: Path | None = None
    if result.status == SPECIALIST_ACCEPTED_STATUS:
        model_path = out_dir / "model.joblib"
        payload = {
            "schema": SPECIALIST_MODEL_SCHEMA,
            "strategy_id": result.spec.strategy_id,
            "candidate_id": result.spec.candidate_id,
            "estimator_family": result.spec.estimator_family,
            "feature_profile": (
                result.spec.feature_profile or "deterministic_score"
            ),
            "deterministic_score": result.spec.deterministic_score,
            "model": result.final_estimator,
            "calibrator": result.final_calibrator,
            "features": result.metrics["features"],
            "split_sha256": result.metrics["split_sha256"],
            "dataset_sha256": dataset_sha256,
            "status": result.status,
        }
        _atomic_joblib(model_path, payload)
    files = {
        name: {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
            "bytes": path.stat().st_size,
        }
        for name, path in evidence_paths.items()
    }
    if model_path is not None:
        files["model"] = {
            "path": str(model_path.resolve()),
            "sha256": file_sha256(model_path),
            "bytes": model_path.stat().st_size,
        }
    manifest: dict[str, object] = {
        "schema": SPECIALIST_CANDIDATE_MANIFEST_SCHEMA,
        "evidence_schema": SPECIALIST_EVIDENCE_SCHEMA,
        "status": result.status,
        "strategy_id": result.spec.strategy_id,
        "candidate_id": result.spec.candidate_id,
        "request_sha256": str(request["request_sha256"]),
        "dataset_sha256": dataset_sha256,
        "split_sha256": str(result.metrics["split_sha256"]),
        "rejection_reasons": list(result.rejection_reasons),
        "prediction_rows": len(result.predictions),
        "files": files,
        "memory": memory_audit(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
        ).to_record(),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "production_ready": False,
    }
    _atomic_json(out_dir / "_manifest.json", manifest)
    return {
        "strategy_id": result.spec.strategy_id,
        "candidate_id": result.spec.candidate_id,
        "status": result.status,
        "request_sha256": str(request["request_sha256"]),
        "manifest_path": str((out_dir / "_manifest.json").resolve()),
        "manifest_sha256": file_sha256(out_dir / "_manifest.json"),
        "prediction_rows": len(result.predictions),
        "rejection_reasons": list(result.rejection_reasons),
        "resumed": False,
    }


def _load_existing_candidate(
    out_dir: Path,
    *,
    expected_request_sha256: str,
) -> dict[str, object] | None:
    manifest_path = out_dir / "_manifest.json"
    if not manifest_path.exists():
        return None
    loaded = _load_json(manifest_path)
    if (
        loaded.get("schema") != SPECIALIST_CANDIDATE_MANIFEST_SCHEMA
        or loaded.get("request_sha256") != expected_request_sha256
        or loaded.get("status")
        not in {SPECIALIST_ACCEPTED_STATUS, "rejected"}
    ):
        raise DataReadinessError(
            f"existing candidate manifest is incompatible: {manifest_path}"
        )
    files = loaded.get("files")
    if not isinstance(files, dict) or not files:
        raise DataReadinessError(
            f"existing candidate has no evidence files: {manifest_path}"
        )
    for raw in files.values():
        if not isinstance(raw, dict):
            raise DataReadinessError("invalid candidate file record")
        path = Path(str(raw.get("path", "")))
        if not path.is_file() or file_sha256(path) != str(
            raw.get("sha256", "")
        ):
            raise DataReadinessError(
                f"candidate evidence hash mismatch: {path}"
            )
    return {
        "strategy_id": str(loaded["strategy_id"]),
        "candidate_id": str(loaded["candidate_id"]),
        "status": str(loaded["status"]),
        "request_sha256": str(loaded["request_sha256"]),
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "prediction_rows": _object_int(loaded["prediction_rows"]),
        "rejection_reasons": _object_list(
            loaded.get("rejection_reasons", [])
        ),
        "resumed": True,
    }


def _write_experiment_status(
    out_dir: Path,
    *,
    request_sha256: str,
    records: Sequence[Mapping[str, object]],
    failures: Mapping[str, str],
    config: SwingSpecialistResearchConfig,
    complete: bool,
    requested_strategies: Sequence[str],
) -> dict[str, object]:
    result: dict[str, object] = {
        "schema": SPECIALIST_EXPERIMENT_BUNDLE_SCHEMA,
        "status": "complete" if complete else "incomplete",
        "request_sha256": request_sha256,
        "requested_strategies": len(requested_strategies),
        "requested_strategy_ids": list(requested_strategies),
        "observed_strategies": len(records),
        "accepted_development_candidates": sum(
            _object_int(record.get("accepted_development_count", 0))
            for record in records
        ),
        "rejected_candidates": sum(
            _object_int(record.get("rejected_count", 0))
            for record in records
        ),
        "failed_strategies": dict(failures),
        "strategies": sorted(
            [dict(record) for record in records],
            key=lambda record: str(record["strategy_id"]),
        ),
        "data_blocked_strategies": {
            PEAD_STRATEGY_ID: (
                "point-in-time earnings surprise and guidance history "
                "is unavailable"
            )
        },
        "memory": memory_audit(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
        ).to_record(),
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "production_ready": False,
    }
    _atomic_json(out_dir / "_status.json", result)
    return result


def _load_dataset_manifest(path: Path) -> dict[str, object]:
    loaded = _load_json(path)
    if (
        loaded.get("schema") != SPECIALIST_DATASET_BUNDLE_SCHEMA
        or loaded.get("status") != "complete"
        or loaded.get("failed_strategies") != {}
    ):
        raise DataReadinessError(
            f"specialist dataset manifest is incomplete: {path}"
        )
    return loaded


def _artifact_records(
    manifest: Mapping[str, object],
) -> list[dict[str, object]]:
    raw = manifest.get("artifacts")
    if not isinstance(raw, list):
        raise DataReadinessError("specialist dataset manifest has no artifacts")
    records: list[dict[str, object]] = []
    for record in raw:
        if not isinstance(record, dict):
            raise DataReadinessError(
                "specialist dataset artifact record is invalid"
            )
        records.append({str(key): value for key, value in record.items()})
    return records


def _load_json(path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"JSON artifact is unreadable: {path}") from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"JSON artifact must be an object: {path}")
    return {str(key): value for key, value in loaded.items()}


def _write_or_validate_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    if path.exists():
        if _load_json(path) != _json_safe(dict(payload)):
            raise DataReadinessError(
                f"immutable request mismatch: {path}"
            )
        return
    _atomic_json(path, payload)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                _json_safe(dict(payload)),
                indent=2,
                sort_keys=True,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_joblib(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        joblib.dump(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_safe(value: object) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _json_sha256(payload: Mapping[str, object]) -> str:
    import hashlib

    encoded = json.dumps(
        _json_safe(dict(payload)),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _slug(value: str) -> str:
    return value.lower().replace(".", "_").replace("-", "_")


def _object_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(
        value,
        (str, int, np.integer),
    ):
        raise DataReadinessError("expected integer experiment metadata")
    try:
        return int(value)
    except ValueError as exc:
        raise DataReadinessError(
            "expected integer experiment metadata"
        ) from exc


def _object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise DataReadinessError("expected list experiment metadata")
    return value
