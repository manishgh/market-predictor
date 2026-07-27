"""Resumable immutable orchestration for KS4 intraday specialists."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from market_predictor.canonical.store import file_sha256
from market_predictor.intraday import (
    specialist_contracts,
    specialist_model,
    specialist_training_data,
)
from market_predictor.intraday.specialist_contracts import (
    INTRADAY_SPECIALIST_IDS,
    IntradaySpecialistResearchConfig,
)
from market_predictor.intraday.specialist_model import (
    DETERMINISTIC_SCORE_FORMULA_SHA256,
    SPECIALIST_ACCEPTED_STATUS,
    SpecialistExperimentResult,
    SpecialistExperimentSpec,
    SpecialistSplitPlan,
    build_specialist_split_plan,
    evaluate_specialist_experiment,
    specialist_experiment_specs,
)
from market_predictor.intraday.specialist_training_data import (
    SPECIALIST_TRAINING_DATASET_SCHEMA,
    SPECIALIST_TRAINING_ROW_SCHEMA,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.v3.errors import DataReadinessError

SPECIALIST_EXPERIMENT_BUNDLE_SCHEMA = (
    "intraday.specialist_experiment_bundle.v1"
)
SPECIALIST_STRATEGY_MANIFEST_SCHEMA = "intraday.specialist_strategy.v1"
SPECIALIST_CANDIDATE_REQUEST_SCHEMA = (
    "intraday.specialist_candidate_request.v1"
)
SPECIALIST_CANDIDATE_MANIFEST_SCHEMA = "intraday.specialist_candidate.v1"
SPECIALIST_EVIDENCE_SCHEMA = "intraday.specialist_evidence.v1"
SPECIALIST_MODEL_SCHEMA = "intraday.specialist_model.v1"
SPECIALIST_AUTHORITY_SCHEMA = "intraday.specialist_authority.v1"
CATALYST_DATA_BLOCKED_REASON = (
    "causal catalyst history with immutable source, security-resolution, "
    "provider-event-time, first-observed-time, ingestion-time, and coverage "
    "lineage is unavailable"
)

_REQUIRED_CANDIDATE_EVIDENCE = frozenset(
    {
        "request",
        "predictions",
        "economics",
        "regime_evidence",
        "fold_audit",
        "metrics",
    }
)

_TRAINING_BASE_COLUMNS = {
    "training_schema_version",
    "strategy_id",
    "setup_id",
    "ticker",
    "session_date_et",
    "decision_time_utc",
    "feature_available_at_utc",
    "entry_time_utc",
    "exit_time_utc",
    "label_available_at_utc",
    "label_window_end_utc",
    "label_eligible",
    "horizon_minutes",
    "path_outcome",
    "entry_price",
    "entry_dollar_volume",
    "entry_atr_pct",
    "sector",
    "primary_benchmark",
    "market_cap_bucket",
    "liquidity_bucket",
    "regime_risk_on",
    "regime_risk_off",
    "regime_high_volatility",
}


@dataclass(frozen=True)
class VerifiedTrainingBundle:
    directory: Path
    manifest: dict[str, object]
    manifest_sha256: str
    dataset_fingerprint: str
    strategy_files: dict[str, tuple[dict[str, object], ...]]
    strategy_dataset_sha256: dict[str, str]


def specialist_implementation_identity() -> dict[str, object]:
    """Bind every KS4 implementation surface that affects an experiment."""

    modules = {
        "specialist_experiments": Path(__file__).resolve(),
        "specialist_model": Path(specialist_model.__file__).resolve(),
        "specialist_contracts": Path(
            specialist_contracts.__file__
        ).resolve(),
        "specialist_training_data": Path(
            specialist_training_data.__file__
        ).resolve(),
    }
    files = {
        name: {
            "file": path.name,
            "sha256": file_sha256(path),
        }
        for name, path in sorted(modules.items())
    }
    payload = {
        "files": files,
        "training_dataset_schema": SPECIALIST_TRAINING_DATASET_SCHEMA,
        "training_row_schema": SPECIALIST_TRAINING_ROW_SCHEMA,
        "formula_sha256": DETERMINISTIC_SCORE_FORMULA_SHA256,
    }
    return {
        **payload,
        "implementation_sha256": _json_sha256(payload),
    }


def train_intraday_specialist_experiments(
    *,
    dataset_dir: Path,
    out_dir: Path,
    config: IntradaySpecialistResearchConfig,
    policy_path: Path,
    strategy_ids: Sequence[str] | None = None,
    progress: Callable[[object], None] | None = None,
) -> dict[str, object]:
    """Run the frozen KS4 matrix sequentially with verified resumability."""

    training = verify_intraday_specialist_training_bundle(
        dataset_dir,
        config=config,
        policy_path=policy_path,
    )
    implementation = specialist_implementation_identity()
    request: dict[str, object] = {
        "schema": SPECIALIST_EXPERIMENT_BUNDLE_SCHEMA,
        "training_manifest_sha256": training.manifest_sha256,
        "training_dataset_fingerprint": training.dataset_fingerprint,
        "training_dataset_schema": SPECIALIST_TRAINING_DATASET_SCHEMA,
        "training_row_schema": SPECIALIST_TRAINING_ROW_SCHEMA,
        "specialist_research_policy_sha256": config.policy_sha256(),
        "specialist_research_policy_file_sha256": file_sha256(
            policy_path
        ),
        "specialist_research_policy_schema": config.schema_version,
        "deterministic_score_formula_sha256": (
            DETERMINISTIC_SCORE_FORMULA_SHA256
        ),
        "implementation": implementation,
        "strategy_ids": list(INTRADAY_SPECIALIST_IDS),
        "catalyst_overlay": {
            "status": "data_blocked",
            "reason": CATALYST_DATA_BLOCKED_REASON,
        },
    }
    request["request_sha256"] = _json_sha256(request)
    request_sha256 = str(request["request_sha256"])
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(out_dir / "_request.json", request)
    _write_authority(
        out_dir,
        state="building",
        request_sha256=request_sha256,
    )

    selected = (
        tuple(INTRADAY_SPECIALIST_IDS)
        if strategy_ids is None
        else tuple(strategy_ids)
    )
    unknown = sorted(set(selected).difference(INTRADAY_SPECIALIST_IDS))
    if not selected or unknown or len(set(selected)) != len(selected):
        raise DataReadinessError(
            "invalid KS4 strategy selection: "
            + ", ".join(unknown or ["empty or duplicate"])
        )

    records: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    stop_for_memory = False
    for strategy_id in selected:
        dataset: pd.DataFrame | None = None
        try:
            dataset = _load_strategy_dataset(
                training,
                strategy_id=strategy_id,
                config=config,
            )
            plan = build_specialist_split_plan(
                dataset,
                strategy_id=strategy_id,
                config=config,
            )
            record = _run_strategy_experiments(
                plan=plan,
                dataset_sha256=training.strategy_dataset_sha256[
                    strategy_id
                ],
                training_dataset_fingerprint=(
                    training.dataset_fingerprint
                ),
                out_dir=out_dir / "strategies" / _slug(strategy_id),
                config=config,
                bundle_request_sha256=request_sha256,
                implementation_sha256=str(
                    implementation["implementation_sha256"]
                ),
                progress=progress,
            )
            records.append(record)
        except Exception as exc:
            failures[strategy_id] = f"{type(exc).__name__}: {exc}"
            _write_failure_evidence(
                out_dir.with_name(f"{out_dir.name}.failures")
                / f"{_slug(strategy_id)}.failure.json",
                scope=strategy_id,
                request_sha256=request_sha256,
                exc=exc,
            )
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
                    stage=f"KS4 strategy {strategy_id}",
                )
            except DataReadinessError as exc:
                failures[strategy_id] = str(exc)
                stop_for_memory = True
            _write_experiment_status(
                out_dir,
                request_sha256=request_sha256,
                current=records,
                failures=failures,
                config=config,
                invocation_strategy_ids=selected,
            )
        if stop_for_memory:
            break

    try:
        assert_peak_memory_budget(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
            stage="KS4 specialist experiments",
        )
    except DataReadinessError as exc:
        failures["memory_budget"] = str(exc)
    result = _write_experiment_status(
        out_dir,
        request_sha256=request_sha256,
        current=records,
        failures=failures,
        config=config,
        invocation_strategy_ids=selected,
    )
    if result["status"] == "complete":
        _write_or_validate_json(out_dir / "_manifest.json", result)
        _validate_complete_bundle_file_set(out_dir)
        _write_authority(
            out_dir,
            state="complete",
            request_sha256=request_sha256,
            artifact_sha256=file_sha256(out_dir / "_manifest.json"),
        )
    else:
        _write_authority(
            out_dir,
            state="incomplete",
            request_sha256=request_sha256,
            artifact_sha256=file_sha256(out_dir / "_status.json"),
        )
    return result


def run_intraday_specialist_experiments(
    **kwargs: Any,
) -> dict[str, object]:
    """Compatibility-free descriptive alias for orchestration callers."""

    return train_intraday_specialist_experiments(**kwargs)


def verify_intraday_specialist_training_bundle(
    dataset_dir: Path,
    *,
    config: IntradaySpecialistResearchConfig,
    policy_path: Path,
) -> VerifiedTrainingBundle:
    """Verify KS4 lineage, every shard, and the dataset fingerprint."""

    directory = dataset_dir.resolve()
    manifest_path = directory / "_manifest.json"
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != SPECIALIST_TRAINING_DATASET_SCHEMA
        or manifest.get("row_schema") != SPECIALIST_TRAINING_ROW_SCHEMA
    ):
        raise DataReadinessError(
            f"invalid KS4 training manifest schema: {manifest_path}"
        )
    policy = _mapping(manifest.get("policy"))
    if (
        policy.get("policy_sha256") != config.policy_sha256()
        or policy.get("file_sha256") != file_sha256(policy_path)
        or policy.get("schema_version") != config.schema_version
    ):
        raise DataReadinessError(
            "KS4 training manifest policy identity does not match"
        )
    repo_root = policy_path.resolve().parent.parent
    setup = _mapping(manifest.get("setup_bundle"))
    plan = _mapping(manifest.get("collection_plan"))
    collection = _mapping(manifest.get("collection"))
    _verify_lineage_manifest(
        repo_root,
        setup,
        expected_sha256=str(setup.get("manifest_sha256", "")),
        expected_schema="intraday.specialist_setup_bundle.v1",
        identity_key="bundle_fingerprint",
        identity_value=str(setup.get("bundle_fingerprint", "")),
    )
    _verify_lineage_manifest(
        repo_root,
        plan,
        expected_sha256=str(plan.get("manifest_sha256", "")),
        expected_schema="intraday.specialist_collection_plan.v1",
        identity_key="plan_fingerprint",
        identity_value=str(plan.get("plan_fingerprint", "")),
    )
    _verify_lineage_manifest(
        repo_root,
        collection,
        expected_sha256=str(collection.get("manifest_sha256", "")),
        expected_schema=(
            "intraday.specialist_one_minute_collection.v1"
        ),
        identity_key="request_sha256",
        identity_value=str(collection.get("request_sha256", "")),
        require_complete_collection=True,
    )

    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise DataReadinessError("KS4 training manifest has no shard files")
    records: list[dict[str, object]] = []
    observed_paths: set[str] = set()
    strategy_files: dict[str, list[dict[str, object]]] = {
        strategy_id: [] for strategy_id in INTRADAY_SPECIALIST_IDS
    }
    slug_to_strategy = {
        _slug(strategy_id): strategy_id
        for strategy_id in INTRADAY_SPECIALIST_IDS
    }
    for raw in raw_files:
        record = _mapping(raw)
        relative = _safe_relative_path(
            str(record.get("path", "")),
            name="KS4 training shard",
        )
        if (
            len(relative.parts) != 3
            or relative.parts[0] != "strategies"
            or relative.suffix != ".parquet"
        ):
            raise DataReadinessError(
                f"invalid KS4 training shard path: {relative}"
            )
        strategy_id = slug_to_strategy.get(relative.parts[1])
        if strategy_id is None:
            raise DataReadinessError(
                f"unknown strategy shard directory: {relative.parts[1]}"
            )
        normalized = relative.as_posix()
        if normalized in observed_paths:
            raise DataReadinessError(
                f"duplicate KS4 training shard: {normalized}"
            )
        observed_paths.add(normalized)
        path = _resolve_inside(directory, relative)
        _verify_file_record(path, record)
        metadata_rows = cast(Any, pq).ParquetFile(path).metadata.num_rows
        if metadata_rows != _object_int(record.get("rows")):
            raise DataReadinessError(
                f"KS4 training shard row count mismatch: {path}"
            )
        normalized_record = {
            "path": normalized,
            "sha256": str(record.get("sha256", "")),
            "bytes": _object_int(record.get("bytes")),
            "rows": _object_int(record.get("rows")),
        }
        records.append(normalized_record)
        strategy_files[strategy_id].append(normalized_record)
    actual_paths = {
        path.relative_to(directory).as_posix()
        for path in (directory / "strategies").rglob("*.parquet")
    }
    if actual_paths != observed_paths:
        raise DataReadinessError(
            "KS4 training bundle parquet file set differs from manifest"
        )
    missing_strategies = [
        strategy_id
        for strategy_id, files in strategy_files.items()
        if not files
    ]
    if missing_strategies:
        raise DataReadinessError(
            "KS4 training bundle misses strategies: "
            + ", ".join(missing_strategies)
        )
    fingerprint = _training_dataset_fingerprint(
        files=records,
        setup_fingerprint=str(setup["bundle_fingerprint"]),
        plan_fingerprint=str(plan["plan_fingerprint"]),
        collection_manifest_sha256=str(
            collection["manifest_sha256"]
        ),
        policy_sha256=config.policy_sha256(),
    )
    if manifest.get("dataset_fingerprint") != fingerprint:
        raise DataReadinessError(
            "KS4 training dataset fingerprint does not verify"
        )
    summary = _mapping(manifest.get("summary"))
    expected_strategy_rows = _mapping(summary.get("strategy_rows"))
    total_rows = 0
    for strategy_id, files in strategy_files.items():
        rows = sum(_object_int(record["rows"]) for record in files)
        total_rows += rows
        if _object_int(expected_strategy_rows.get(strategy_id)) != rows:
            raise DataReadinessError(
                f"{strategy_id} manifest summary row count differs"
            )
    if _object_int(summary.get("rows")) != total_rows:
        raise DataReadinessError(
            "KS4 training manifest total row count differs"
        )
    eligible_by_strategy = _mapping(
        summary.get("strategy_eligible_rows")
    )
    eligible_rows = sum(
        _object_int(eligible_by_strategy.get(strategy_id))
        for strategy_id in INTRADAY_SPECIALIST_IDS
    )
    if (
        eligible_rows != _object_int(summary.get("eligible_rows"))
        or any(
            _object_int(eligible_by_strategy.get(strategy_id))
            > _object_int(expected_strategy_rows.get(strategy_id))
            for strategy_id in INTRADAY_SPECIALIST_IDS
        )
    ):
        raise DataReadinessError(
            "KS4 training manifest eligible row counts differ"
        )
    strategy_hashes = {
        strategy_id: _json_sha256(
            {
                "training_dataset_fingerprint": fingerprint,
                "strategy_id": strategy_id,
                "files": sorted(
                    files, key=lambda record: str(record["path"])
                ),
            }
        )
        for strategy_id, files in strategy_files.items()
    }
    return VerifiedTrainingBundle(
        directory=directory,
        manifest=manifest,
        manifest_sha256=file_sha256(manifest_path),
        dataset_fingerprint=fingerprint,
        strategy_files={
            strategy_id: tuple(
                sorted(files, key=lambda record: str(record["path"]))
            )
            for strategy_id, files in strategy_files.items()
        },
        strategy_dataset_sha256=strategy_hashes,
    )


def _load_strategy_dataset(
    training: VerifiedTrainingBundle,
    *,
    strategy_id: str,
    config: IntradaySpecialistResearchConfig,
) -> pd.DataFrame:
    records = training.strategy_files[strategy_id]
    horizon = config.strategies[strategy_id].horizon_minutes
    dynamic = {
        f"target_before_stop_{horizon}m",
        f"stop_before_target_{horizon}m",
        f"path_realized_return_gross_{horizon}m",
        f"path_realized_return_net_{horizon}m",
        f"path_spy_return_{horizon}m",
        f"path_qqq_return_{horizon}m",
        f"path_sector_return_{horizon}m",
        f"path_excess_return_{horizon}m_vs_spy",
        f"path_excess_return_{horizon}m_vs_qqq",
        f"path_excess_return_{horizon}m_vs_sector",
    }
    requested = {
        *_TRAINING_BASE_COLUMNS,
        *config.technical_features,
        *dynamic,
    }
    expected_schema: tuple[str, ...] | None = None
    parts: list[pd.DataFrame] = []
    for record in records:
        path = _resolve_inside(
            training.directory,
            Path(str(record["path"])),
        )
        _verify_file_record(path, record)
        schema = cast(Any, pq).read_schema(path)
        names = tuple(str(name) for name in schema.names)
        if expected_schema is None:
            expected_schema = names
        elif names != expected_schema:
            raise DataReadinessError(
                f"{strategy_id} shard schemas are inconsistent"
            )
        columns = [name for name in names if name in requested]
        frame = pd.read_parquet(path, columns=columns)
        if len(frame) != _object_int(record["rows"]):
            raise DataReadinessError(
                f"{strategy_id} shard changed after verification: {path}"
            )
        if (
            set(frame["strategy_id"].astype(str)) != {strategy_id}
            or set(frame["training_schema_version"].astype(str))
            != {SPECIALIST_TRAINING_ROW_SCHEMA}
        ):
            raise DataReadinessError(
                f"{strategy_id} shard row identity is invalid: {path}"
            )
        parts.append(frame)
    dataset = pd.concat(parts, ignore_index=True)
    del parts
    expected_rows = sum(
        _object_int(record["rows"]) for record in records
    )
    if len(dataset) != expected_rows:
        raise DataReadinessError(
            f"{strategy_id} loaded row count differs from its shards"
        )
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=f"KS4 load strategy {strategy_id}",
    )
    return dataset


def _run_strategy_experiments(
    *,
    plan: SpecialistSplitPlan,
    dataset_sha256: str,
    training_dataset_fingerprint: str,
    out_dir: Path,
    config: IntradaySpecialistResearchConfig,
    bundle_request_sha256: str,
    implementation_sha256: str,
    progress: Callable[[object], None] | None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = _load_existing_strategy(
        out_dir,
        plan=plan,
        dataset_sha256=dataset_sha256,
        training_dataset_fingerprint=training_dataset_fingerprint,
        bundle_request_sha256=bundle_request_sha256,
        implementation_sha256=implementation_sha256,
        config=config,
    )
    if existing is not None:
        return existing
    candidate_records: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    stop_for_memory = False
    specs = specialist_experiment_specs(plan.strategy_id, config)
    for spec in specs:
        candidate_dir = out_dir / "candidates" / _slug(
            spec.candidate_id
        )
        request = _candidate_request(
            spec,
            dataset_sha256=dataset_sha256,
            training_dataset_fingerprint=training_dataset_fingerprint,
            split_sha256=plan.split_sha256,
            bundle_request_sha256=bundle_request_sha256,
            implementation_sha256=implementation_sha256,
            config=config,
        )
        try:
            resumed = _load_existing_candidate(
                candidate_dir,
                expected_request_sha256=str(request["request_sha256"]),
            )
            if resumed is not None:
                candidate_records.append(resumed)
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
            candidate_records.append(
                _write_candidate_evidence(
                    candidate_dir,
                    result=result,
                    request=request,
                    dataset_sha256=dataset_sha256,
                    training_dataset_fingerprint=(
                        training_dataset_fingerprint
                    ),
                    config=config,
                )
            )
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
        except Exception as exc:
            failures[spec.candidate_id] = f"{type(exc).__name__}: {exc}"
            _write_failure_evidence(
                _candidate_failure_path(out_dir, spec.candidate_id),
                scope=f"{plan.strategy_id}:{spec.candidate_id}",
                request_sha256=str(request["request_sha256"]),
                exc=exc,
            )
        finally:
            gc.collect()
            release_process_memory()
            try:
                assert_memory_budget(
                    hard_budget_gib=config.maximum_process_memory_gib,
                    headroom_gib=config.memory_guard_headroom_gib,
                    stage=(
                        f"KS4 candidate {plan.strategy_id} "
                        f"{spec.candidate_id}"
                    ),
                )
            except DataReadinessError as exc:
                failures[spec.candidate_id] = str(exc)
                stop_for_memory = True
            _atomic_json(
                out_dir / "_status.json",
                _strategy_record(
                    plan=plan,
                    dataset_sha256=dataset_sha256,
                    training_dataset_fingerprint=(
                        training_dataset_fingerprint
                    ),
                    bundle_request_sha256=bundle_request_sha256,
                    implementation_sha256=implementation_sha256,
                    candidate_records=candidate_records,
                    failures=failures,
                    status="building",
                ),
            )
        if stop_for_memory:
            break
    status = (
        "complete"
        if not failures and len(candidate_records) == len(specs)
        else "incomplete"
    )
    record = _strategy_record(
        plan=plan,
        dataset_sha256=dataset_sha256,
        training_dataset_fingerprint=training_dataset_fingerprint,
        bundle_request_sha256=bundle_request_sha256,
        implementation_sha256=implementation_sha256,
        candidate_records=candidate_records,
        failures=failures,
        status=status,
    )
    _atomic_json(out_dir / "_status.json", record)
    if status == "complete":
        _write_or_validate_json(out_dir / "_manifest.json", record)
        _write_authority(
            out_dir,
            state="complete",
            request_sha256=bundle_request_sha256,
            artifact_sha256=file_sha256(out_dir / "_manifest.json"),
        )
        _validate_strategy_file_set(out_dir, record)
    if failures:
        raise DataReadinessError(
            f"{plan.strategy_id} candidate matrix is incomplete"
        )
    return record


def _candidate_request(
    spec: SpecialistExperimentSpec,
    *,
    dataset_sha256: str,
    training_dataset_fingerprint: str,
    split_sha256: str,
    bundle_request_sha256: str,
    implementation_sha256: str,
    config: IntradaySpecialistResearchConfig,
) -> dict[str, object]:
    request: dict[str, object] = {
        "schema": SPECIALIST_CANDIDATE_REQUEST_SCHEMA,
        "strategy_id": spec.strategy_id,
        "candidate_id": spec.candidate_id,
        "estimator_family": spec.estimator_family,
        "deterministic_score": spec.deterministic_score,
        "dataset_sha256": dataset_sha256,
        "training_dataset_fingerprint": training_dataset_fingerprint,
        "split_sha256": split_sha256,
        "bundle_request_sha256": bundle_request_sha256,
        "specialist_research_policy_sha256": config.policy_sha256(),
        "training_dataset_schema": SPECIALIST_TRAINING_DATASET_SCHEMA,
        "training_row_schema": SPECIALIST_TRAINING_ROW_SCHEMA,
        "deterministic_score_formula_sha256": (
            DETERMINISTIC_SCORE_FORMULA_SHA256
        ),
        "implementation_sha256": implementation_sha256,
        "catalyst_overlay": {
            "status": "data_blocked",
            "reason": CATALYST_DATA_BLOCKED_REASON,
        },
    }
    request["request_sha256"] = _json_sha256(request)
    return request


def _write_candidate_evidence(
    out_dir: Path,
    *,
    result: SpecialistExperimentResult,
    request: Mapping[str, object],
    dataset_sha256: str,
    training_dataset_fingerprint: str,
    config: IntradaySpecialistResearchConfig,
) -> dict[str, object]:
    if out_dir.exists():
        raise DataReadinessError(
            "candidate directory exists without verified authority: "
            f"{out_dir}"
        )
    temporary = out_dir.with_name(f".{out_dir.name}.{uuid4().hex}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        manifest = _write_candidate_evidence_directory(
            temporary,
            result=result,
            request=request,
            dataset_sha256=dataset_sha256,
            training_dataset_fingerprint=training_dataset_fingerprint,
            config=config,
        )
        os.replace(temporary, out_dir)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return _candidate_summary(out_dir, manifest)


def _write_candidate_evidence_directory(
    out_dir: Path,
    *,
    result: SpecialistExperimentResult,
    request: Mapping[str, object],
    dataset_sha256: str,
    training_dataset_fingerprint: str,
    config: IntradaySpecialistResearchConfig,
) -> dict[str, object]:
    _atomic_json(out_dir / "_request.json", request)
    evidence_paths = {
        "predictions": out_dir / "predictions.parquet",
        "economics": out_dir / "economics.parquet",
        "regime_evidence": out_dir / "regime_evidence.parquet",
        "fold_audit": out_dir / "fold_audit.parquet",
        "metrics": out_dir / "metrics.json",
    }
    _atomic_parquet(result.predictions, evidence_paths["predictions"])
    _atomic_parquet(result.economics, evidence_paths["economics"])
    _atomic_parquet(
        result.regime_evidence,
        evidence_paths["regime_evidence"],
    )
    _atomic_parquet(result.fold_audit, evidence_paths["fold_audit"])
    metrics = {
        **result.metrics,
        "catalyst_overlay": {
            "status": "data_blocked",
            "reason": CATALYST_DATA_BLOCKED_REASON,
        },
    }
    _atomic_json(evidence_paths["metrics"], metrics)
    model_path: Path | None = None
    if result.status == SPECIALIST_ACCEPTED_STATUS:
        if result.retained_model is None:
            raise DataReadinessError(
                "accepted KS4 candidate has no retained model"
            )
        model_path = out_dir / "model.joblib"
        retained = result.retained_model
        _atomic_joblib(
            model_path,
            {
                "schema": SPECIALIST_MODEL_SCHEMA,
                "status": result.status,
                "strategy_id": result.spec.strategy_id,
                "candidate_id": result.spec.candidate_id,
                "estimator_family": result.spec.estimator_family,
                "deterministic_score": result.spec.deterministic_score,
                "estimators": retained.estimators,
                "calibrators": retained.calibrators,
                "features": list(retained.features),
                "opportunity_target": retained.opportunity_target,
                "downside_target": retained.downside_target,
                "dataset_sha256": dataset_sha256,
                "training_dataset_fingerprint": (
                    training_dataset_fingerprint
                ),
                "split_sha256": result.metrics["split_sha256"],
                "policy_sha256": request[
                    "specialist_research_policy_sha256"
                ],
                "formula_sha256": (
                    DETERMINISTIC_SCORE_FORMULA_SHA256
                ),
                "implementation_sha256": request[
                    "implementation_sha256"
                ],
            },
        )
    elif result.retained_model is not None:
        raise DataReadinessError(
            "rejected KS4 candidate retained a loadable model"
        )
    files = {
        name: _candidate_file_record(path, root=out_dir)
        for name, path in evidence_paths.items()
    }
    files["request"] = _candidate_file_record(
        out_dir / "_request.json",
        root=out_dir,
    )
    if model_path is not None:
        files["model"] = _candidate_file_record(
            model_path,
            root=out_dir,
        )
    manifest: dict[str, object] = {
        "schema": SPECIALIST_CANDIDATE_MANIFEST_SCHEMA,
        "evidence_schema": SPECIALIST_EVIDENCE_SCHEMA,
        "status": result.status,
        "strategy_id": result.spec.strategy_id,
        "candidate_id": result.spec.candidate_id,
        "request_sha256": str(request["request_sha256"]),
        "dataset_sha256": dataset_sha256,
        "training_dataset_fingerprint": training_dataset_fingerprint,
        "split_sha256": str(result.metrics["split_sha256"]),
        "rejection_reasons": list(result.rejection_reasons),
        "prediction_rows": len(result.predictions),
        "files": files,
        "catalyst_overlay": {
            "status": "data_blocked",
            "reason": CATALYST_DATA_BLOCKED_REASON,
        },
        "memory": memory_audit(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
        ).to_record(),
        "production_ready": False,
    }
    _atomic_json(out_dir / "_manifest.json", manifest)
    _write_authority(
        out_dir,
        state="complete",
        request_sha256=str(request["request_sha256"]),
        artifact_sha256=file_sha256(out_dir / "_manifest.json"),
    )
    return manifest


def _load_existing_candidate(
    out_dir: Path,
    *,
    expected_request_sha256: str,
) -> dict[str, object] | None:
    manifest_path = out_dir / "_manifest.json"
    if not manifest_path.exists():
        if out_dir.exists():
            raise DataReadinessError(
                f"unverified partial candidate directory: {out_dir}"
            )
        return None
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != SPECIALIST_CANDIDATE_MANIFEST_SCHEMA
        or manifest.get("evidence_schema") != SPECIALIST_EVIDENCE_SCHEMA
        or manifest.get("request_sha256") != expected_request_sha256
        or manifest.get("status")
        not in {SPECIALIST_ACCEPTED_STATUS, "rejected"}
    ):
        raise DataReadinessError(
            f"existing KS4 candidate is incompatible: {manifest_path}"
        )
    authority = _load_json(out_dir / "_authority.json")
    if (
        authority.get("schema") != SPECIALIST_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("request_sha256") != expected_request_sha256
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
    ):
        raise DataReadinessError(
            f"candidate authority does not verify: {out_dir}"
        )
    files = _mapping(manifest.get("files"))
    expected = set(_REQUIRED_CANDIDATE_EVIDENCE)
    if manifest["status"] == SPECIALIST_ACCEPTED_STATUS:
        expected.add("model")
    if set(files) != expected:
        raise DataReadinessError(
            f"candidate evidence contract mismatch: {out_dir}"
        )
    request = _load_json(out_dir / "_request.json")
    if (
        request.get("schema") != SPECIALIST_CANDIDATE_REQUEST_SCHEMA
        or request.get("request_sha256") != expected_request_sha256
    ):
        raise DataReadinessError(
            f"candidate request does not verify: {out_dir}"
        )
    expected_names = {"_manifest.json", "_authority.json"}
    for raw in files.values():
        record = _mapping(raw)
        relative = _safe_relative_path(
            str(record.get("path", "")),
            name="candidate evidence",
        )
        path = _resolve_inside(out_dir.resolve(), relative)
        _verify_file_record(path, record)
        expected_columns = record.get("columns")
        if expected_columns is not None:
            schema = cast(Any, pq).read_schema(path)
            if (
                list(schema.names) != expected_columns
                or record.get("arrow_schema_sha256")
                != _json_sha256({"arrow_schema": str(schema)})
            ):
                raise DataReadinessError(
                    f"candidate parquet schema mismatch: {path}"
                )
        expected_names.add(relative.as_posix())
    observed_names = {
        path.relative_to(out_dir).as_posix()
        for path in out_dir.rglob("*")
        if path.is_file()
    }
    if observed_names != expected_names:
        raise DataReadinessError(
            f"candidate file set mismatch: {out_dir}"
        )
    return _candidate_summary(out_dir, manifest)


def _candidate_summary(
    out_dir: Path,
    manifest: Mapping[str, object],
) -> dict[str, object]:
    return {
        "strategy_id": str(manifest["strategy_id"]),
        "candidate_id": str(manifest["candidate_id"]),
        "status": str(manifest["status"]),
        "request_sha256": str(manifest["request_sha256"]),
        "manifest_path": (
            Path("candidates") / out_dir.name / "_manifest.json"
        ).as_posix(),
        "manifest_sha256": file_sha256(out_dir / "_manifest.json"),
        "prediction_rows": _object_int(manifest["prediction_rows"]),
        "rejection_reasons": _object_list(
            manifest.get("rejection_reasons", [])
        ),
        "peak_working_set_gib": _object_float(
            _mapping(manifest["memory"])["peak_working_set_gib"]
        ),
    }


def _load_existing_strategy(
    out_dir: Path,
    *,
    plan: SpecialistSplitPlan,
    dataset_sha256: str,
    training_dataset_fingerprint: str,
    bundle_request_sha256: str,
    implementation_sha256: str,
    config: IntradaySpecialistResearchConfig,
) -> dict[str, object] | None:
    manifest_path = out_dir / "_manifest.json"
    if not manifest_path.exists():
        return None
    candidates: list[dict[str, object]] = []
    for spec in specialist_experiment_specs(plan.strategy_id, config):
        request = _candidate_request(
            spec,
            dataset_sha256=dataset_sha256,
            training_dataset_fingerprint=training_dataset_fingerprint,
            split_sha256=plan.split_sha256,
            bundle_request_sha256=bundle_request_sha256,
            implementation_sha256=implementation_sha256,
            config=config,
        )
        candidate = _load_existing_candidate(
            out_dir / "candidates" / _slug(spec.candidate_id),
            expected_request_sha256=str(request["request_sha256"]),
        )
        if candidate is None:
            raise DataReadinessError(
                f"completed strategy misses {spec.candidate_id}"
            )
        candidates.append(candidate)
    expected = _strategy_record(
        plan=plan,
        dataset_sha256=dataset_sha256,
        training_dataset_fingerprint=training_dataset_fingerprint,
        bundle_request_sha256=bundle_request_sha256,
        implementation_sha256=implementation_sha256,
        candidate_records=candidates,
        failures={},
        status="complete",
    )
    loaded = _load_json(manifest_path)
    if loaded != _json_safe(expected):
        raise DataReadinessError(
            f"immutable KS4 strategy manifest mismatch: {manifest_path}"
        )
    _verify_authority(
        out_dir,
        request_sha256=bundle_request_sha256,
        artifact_sha256=file_sha256(manifest_path),
    )
    _validate_strategy_file_set(out_dir, expected)
    return expected


def _strategy_record(
    *,
    plan: SpecialistSplitPlan,
    dataset_sha256: str,
    training_dataset_fingerprint: str,
    bundle_request_sha256: str,
    implementation_sha256: str,
    candidate_records: Sequence[Mapping[str, object]],
    failures: Mapping[str, str],
    status: str,
) -> dict[str, object]:
    return {
        "schema": SPECIALIST_STRATEGY_MANIFEST_SCHEMA,
        "bundle_request_sha256": bundle_request_sha256,
        "strategy_id": plan.strategy_id,
        "status": status,
        "dataset_sha256": dataset_sha256,
        "training_dataset_fingerprint": training_dataset_fingerprint,
        "split_sha256": plan.split_sha256,
        "implementation_sha256": implementation_sha256,
        "candidate_count": len(candidate_records),
        "accepted_development_count": sum(
            str(record["status"]) == SPECIALIST_ACCEPTED_STATUS
            for record in candidate_records
        ),
        "rejected_count": sum(
            str(record["status"]) == "rejected"
            for record in candidate_records
        ),
        "failed_candidates": dict(failures),
        "catalyst_overlay": {
            "status": "data_blocked",
            "reason": CATALYST_DATA_BLOCKED_REASON,
        },
        "candidates": sorted(
            [dict(record) for record in candidate_records],
            key=lambda record: str(record["candidate_id"]),
        ),
    }


def _write_experiment_status(
    out_dir: Path,
    *,
    request_sha256: str,
    current: Sequence[Mapping[str, object]],
    failures: Mapping[str, str],
    config: IntradaySpecialistResearchConfig,
    invocation_strategy_ids: Sequence[str],
) -> dict[str, object]:
    records = _completed_strategy_records(
        out_dir,
        request_sha256=request_sha256,
        current=current,
    )
    complete = (
        not failures and len(records) == len(INTRADAY_SPECIALIST_IDS)
    )
    peaks = [
        _object_float(candidate.get("peak_working_set_gib", 0.0))
        for record in records
        for candidate in _mapping_list(record.get("candidates", []))
    ]
    result: dict[str, object] = {
        "schema": SPECIALIST_EXPERIMENT_BUNDLE_SCHEMA,
        "status": "complete" if complete else "incomplete",
        "request_sha256": request_sha256,
        "requested_strategies": len(INTRADAY_SPECIALIST_IDS),
        "requested_strategy_ids": list(INTRADAY_SPECIALIST_IDS),
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
        "catalyst_overlay": {
            "status": "data_blocked",
            "reason": CATALYST_DATA_BLOCKED_REASON,
        },
        "memory": {
            "hard_budget_gib": config.maximum_process_memory_gib,
            "safety_threshold_gib": (
                config.maximum_process_memory_gib
                - config.memory_guard_headroom_gib
            ),
            "peak_working_set_gib": max(peaks, default=0.0),
        },
        "production_ready": False,
    }
    status = {
        **result,
        "invocation_status": (
            "complete"
            if not failures
            and len(current) == len(invocation_strategy_ids)
            else "failed"
        ),
        "invocation_strategy_ids": list(invocation_strategy_ids),
        "updated_at_utc": datetime.now(UTC).isoformat(),
    }
    _atomic_json(out_dir / "_status.json", status)
    return result


def _completed_strategy_records(
    out_dir: Path,
    *,
    request_sha256: str,
    current: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    by_strategy = {
        str(record["strategy_id"]): dict(record) for record in current
    }
    directory = out_dir / "strategies"
    if directory.exists():
        for path in directory.glob("*/_manifest.json"):
            loaded = _load_json(path)
            if (
                loaded.get("schema")
                != SPECIALIST_STRATEGY_MANIFEST_SCHEMA
                or loaded.get("bundle_request_sha256")
                != request_sha256
            ):
                raise DataReadinessError(
                    f"incompatible completed strategy: {path}"
                )
            strategy_dir = path.parent
            _verify_authority(
                strategy_dir,
                request_sha256=request_sha256,
                artifact_sha256=file_sha256(path),
            )
            for candidate_record in _mapping_list(
                loaded.get("candidates", [])
            ):
                candidate_id = str(candidate_record["candidate_id"])
                verified = _load_existing_candidate(
                    strategy_dir
                    / "candidates"
                    / _slug(candidate_id),
                    expected_request_sha256=str(
                        candidate_record["request_sha256"]
                    ),
                )
                if verified != candidate_record:
                    raise DataReadinessError(
                        "completed strategy candidate summary differs: "
                        f"{strategy_dir / 'candidates' / _slug(candidate_id)}"
                    )
            _validate_strategy_file_set(strategy_dir, loaded)
            strategy_id = str(loaded.get("strategy_id", ""))
            existing = by_strategy.get(strategy_id)
            if existing is not None and existing != loaded:
                raise DataReadinessError(
                    f"strategy status disagrees with manifest: {path}"
                )
            by_strategy[strategy_id] = loaded
    unknown = sorted(
        set(by_strategy).difference(INTRADAY_SPECIALIST_IDS)
    )
    if unknown:
        raise DataReadinessError(
            "unexpected completed KS4 strategies: " + ", ".join(unknown)
        )
    return list(by_strategy.values())


def _validate_complete_bundle_file_set(out_dir: Path) -> None:
    files = {
        path.name for path in out_dir.iterdir() if path.is_file()
    }
    directories = {
        path.name for path in out_dir.iterdir() if path.is_dir()
    }
    if files != {
        "_authority.json",
        "_manifest.json",
        "_request.json",
        "_status.json",
    } or directories != {"strategies"}:
        raise DataReadinessError(
            f"KS4 experiment bundle file set mismatch: {out_dir}"
        )


def _validate_strategy_file_set(
    out_dir: Path,
    record: Mapping[str, object],
) -> None:
    files = {
        path.name for path in out_dir.iterdir() if path.is_file()
    }
    directories = {
        path.name for path in out_dir.iterdir() if path.is_dir()
    }
    if files != {
        "_authority.json",
        "_manifest.json",
        "_status.json",
    } or directories != {"candidates"}:
        raise DataReadinessError(
            f"KS4 strategy file set mismatch: {out_dir}"
        )
    expected = {
        _slug(str(candidate["candidate_id"]))
        for candidate in _mapping_list(record.get("candidates", []))
    }
    observed = {
        path.name
        for path in (out_dir / "candidates").iterdir()
        if path.is_dir()
    }
    stray_files = {
        path.name
        for path in (out_dir / "candidates").iterdir()
        if path.is_file()
    }
    if observed != expected or stray_files:
        raise DataReadinessError(
            f"KS4 strategy candidate set mismatch: {out_dir}"
        )


def _write_authority(
    out_dir: Path,
    *,
    state: str,
    request_sha256: str,
    artifact_sha256: str | None = None,
) -> None:
    if state not in {"building", "incomplete", "complete"}:
        raise ValueError(f"invalid KS4 authority state: {state}")
    _atomic_json(
        out_dir / "_authority.json",
        {
            "schema": SPECIALIST_AUTHORITY_SCHEMA,
            "state": state,
            "request_sha256": request_sha256,
            "artifact": (
                "_manifest.json"
                if state == "complete"
                else "_status.json"
            ),
            "artifact_sha256": artifact_sha256,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def _verify_authority(
    out_dir: Path,
    *,
    request_sha256: str,
    artifact_sha256: str,
) -> None:
    authority = _load_json(out_dir / "_authority.json")
    if (
        authority.get("schema") != SPECIALIST_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("request_sha256") != request_sha256
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != artifact_sha256
    ):
        raise DataReadinessError(
            f"KS4 authority does not verify: {out_dir}"
        )


def _candidate_file_record(
    path: Path,
    *,
    root: Path,
) -> dict[str, object]:
    record: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if path.suffix == ".parquet":
        schema = cast(Any, pq).read_schema(path)
        record["columns"] = list(schema.names)
        record["arrow_schema_sha256"] = _json_sha256(
            {"arrow_schema": str(schema)}
        )
    return record


def _verify_lineage_manifest(
    repo_root: Path,
    record: Mapping[str, object],
    *,
    expected_sha256: str,
    expected_schema: str,
    identity_key: str,
    identity_value: str,
    require_complete_collection: bool = False,
) -> None:
    raw_path = str(record.get("path", ""))
    path = Path(raw_path)
    directory = path if path.is_absolute() else repo_root / path
    manifest_path = directory.resolve() / "_manifest.json"
    try:
        manifest_path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise DataReadinessError(
            f"KS4 lineage path escapes repository: {raw_path}"
        ) from exc
    if (
        not manifest_path.is_file()
        or file_sha256(manifest_path) != expected_sha256
    ):
        raise DataReadinessError(
            f"KS4 lineage manifest hash mismatch: {manifest_path}"
        )
    manifest = _load_json(manifest_path)
    if (
        manifest.get("schema") != expected_schema
        or manifest.get(identity_key) != identity_value
    ):
        raise DataReadinessError(
            f"KS4 lineage identity mismatch: {manifest_path}"
        )
    if require_complete_collection and (
        manifest.get("status") != "transport_complete"
        or _mapping(manifest.get("failed_units")) != {}
        or _object_int(manifest.get("completed_units"))
        != _object_int(manifest.get("requested_units"))
    ):
        raise DataReadinessError(
            f"KS4 collection lineage is incomplete: {manifest_path}"
        )


def _training_dataset_fingerprint(
    *,
    files: Sequence[Mapping[str, object]],
    setup_fingerprint: str,
    plan_fingerprint: str,
    collection_manifest_sha256: str,
    policy_sha256: str,
) -> str:
    payload = {
        "schema": SPECIALIST_TRAINING_DATASET_SCHEMA,
        "setup_fingerprint": setup_fingerprint,
        "plan_fingerprint": plan_fingerprint,
        "collection_manifest_sha256": collection_manifest_sha256,
        "policy_sha256": policy_sha256,
        "files": [
            {
                "path": str(record["path"]),
                "sha256": str(record["sha256"]),
                "rows": _object_int(record["rows"]),
            }
            for record in sorted(
                files, key=lambda item: str(item["path"])
            )
        ],
    }
    return _json_sha256(payload)


def _verify_file_record(
    path: Path,
    record: Mapping[str, object],
) -> None:
    if (
        not path.is_file()
        or file_sha256(path) != str(record.get("sha256", ""))
        or path.stat().st_size != _object_int(record.get("bytes"))
    ):
        raise DataReadinessError(f"file record does not verify: {path}")


def _safe_relative_path(raw: str, *, name: str) -> Path:
    path = Path(raw)
    if (
        not raw
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() in {".", ""}
    ):
        raise DataReadinessError(f"{name} path is not bundle-relative")
    return path


def _resolve_inside(root: Path, relative: Path) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise DataReadinessError(
            f"artifact path escapes bundle: {relative}"
        ) from exc
    return path


def _write_failure_evidence(
    path: Path,
    *,
    scope: str,
    request_sha256: str,
    exc: Exception,
) -> None:
    try:
        _atomic_json(
            path,
            {
                "schema": "intraday.specialist_failure.v1",
                "scope": scope,
                "request_sha256": request_sha256,
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "failed_at_utc": datetime.now(UTC).isoformat(),
            },
        )
    except OSError:
        return


def _candidate_failure_path(
    strategy_out_dir: Path,
    candidate_id: str,
) -> Path:
    bundle_dir = strategy_out_dir.parents[1]
    return (
        bundle_dir.with_name(f"{bundle_dir.name}.failures")
        / strategy_out_dir.name
        / f"{_slug(candidate_id)}.failure.json"
    )


def _load_json(path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(
            f"JSON artifact is unreadable: {path}"
        ) from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"JSON artifact is not an object: {path}")
    return {str(key): value for key, value in loaded.items()}


def _write_or_validate_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    if path.exists():
        if _load_json(path) != _json_safe(dict(payload)):
            raise DataReadinessError(
                f"immutable KS4 request mismatch: {path}"
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


def _atomic_joblib(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        joblib.dump(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_safe(value: object) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _json_safe(item) for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _json_sha256(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(
            _json_safe(dict(payload)),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _slug(value: str) -> str:
    return value.lower().replace(".", "_").replace("-", "_")


def _object_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(
        value, (str, int, np.integer)
    ):
        raise DataReadinessError("expected integer KS4 metadata")
    try:
        return int(value)
    except ValueError as exc:
        raise DataReadinessError(
            "expected integer KS4 metadata"
        ) from exc


def _object_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (str, int, float, np.integer, np.floating)
    ):
        raise DataReadinessError("expected numeric KS4 metadata")
    try:
        return float(value)
    except ValueError as exc:
        raise DataReadinessError(
            "expected numeric KS4 metadata"
        ) from exc


def _object_list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise DataReadinessError("expected list KS4 metadata")
    return value


def _mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise DataReadinessError("expected object KS4 metadata")
    return {str(key): item for key, item in value.items()}


def _mapping_list(value: object) -> list[dict[str, object]]:
    if not isinstance(value, list):
        raise DataReadinessError("expected object-list KS4 metadata")
    return [_mapping(item) for item in value]
