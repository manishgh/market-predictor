"""Immutable orchestration for primary V2 research runs."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import joblib
import pandas as pd

from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
)
from market_predictor.intraday.specialist_contracts import (
    IntradaySpecialistResearchConfig,
)
from market_predictor.intraday.specialist_experiments import (
    verify_intraday_specialist_training_bundle,
)
from market_predictor.primary_v2 import contracts, model
from market_predictor.primary_v2.contracts import (
    INTRADAY_V2_ID,
    SWING_V2_ID,
    PrimaryV2ResearchConfig,
)
from market_predictor.primary_v2.model import (
    PrimaryV2ExperimentResult,
    build_intraday_v2_split_plan,
    build_swing_v2_split_plan,
    evaluate_primary_v2_experiment,
    primary_v2_experiment_specs,
)
from market_predictor.resources import (
    assert_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.swing.specialist_contracts import (
    SwingSpecialistResearchConfig,
)
from market_predictor.swing.specialist_experiments import (
    specialist_training_columns,
)
from market_predictor.v3.errors import DataReadinessError

PRIMARY_V2_RUN_SCHEMA = "primary_strategy_v2.run.v1"
PRIMARY_V2_CANDIDATE_SCHEMA = "primary_strategy_v2.candidate.v1"
PRIMARY_V2_AUTHORITY_SCHEMA = "primary_strategy_v2.authority.v1"


def run_primary_v2_experiments(
    *,
    strategy_id: str,
    source_dir: Path,
    out_dir: Path,
    config: PrimaryV2ResearchConfig,
    policy_path: Path,
    swing_v1_config: SwingSpecialistResearchConfig,
    swing_v1_policy_path: Path,
    intraday_v1_config: IntradaySpecialistResearchConfig,
    intraday_v1_policy_path: Path,
    candidate_ids: Sequence[str] | None = None,
    progress: Callable[[object], None] | None = None,
) -> dict[str, object]:
    """Verify source lineage, run candidates sequentially, and publish evidence."""

    if strategy_id not in config.strategies:
        raise DataReadinessError(f"unknown primary V2 strategy: {strategy_id}")
    implementation = primary_v2_implementation_identity()
    dataset, source_identity = _load_verified_source(
        strategy_id=strategy_id,
        source_dir=source_dir,
        swing_config=swing_v1_config,
        swing_policy_path=swing_v1_policy_path,
        intraday_config=intraday_v1_config,
        intraday_policy_path=intraday_v1_policy_path,
    )
    if strategy_id == SWING_V2_ID:
        plan = build_swing_v2_split_plan(
            dataset,
            v1_config=swing_v1_config,
            v2_config=config,
        )
    else:
        plan = build_intraday_v2_split_plan(
            dataset,
            v1_config=intraday_v1_config,
            v2_config=config,
        )
    del dataset
    release_process_memory()
    _assert_memory(config, f"{strategy_id} source load")

    specs = primary_v2_experiment_specs(strategy_id)
    if candidate_ids:
        requested = set(candidate_ids)
        specs = tuple(spec for spec in specs if spec.candidate_id in requested)
        missing = sorted(requested.difference(spec.candidate_id for spec in specs))
        if missing:
            raise DataReadinessError(
                "unknown primary V2 candidate IDs: " + ", ".join(missing)
            )
    request = {
        "schema": PRIMARY_V2_RUN_SCHEMA,
        "strategy_id": strategy_id,
        "policy_sha256": config.sha256(),
        "policy_file_sha256": file_sha256(policy_path),
        "source": source_identity,
        "split_sha256": plan.split_sha256,
        "implementation": implementation,
        "candidate_ids": [spec.candidate_id for spec in specs],
    }
    request_sha256 = _json_sha256(request)
    root = out_dir.resolve()
    if root.exists():
        try:
            return _load_complete_run(
                root,
                expected_request_sha256=request_sha256,
            )
        except DataReadinessError:
            existing_request = _load_json(root / "_request.json")
            if _json_sha256(existing_request) != request_sha256:
                raise DataReadinessError(
                    "primary V2 output exists for a different request"
                ) from None
    else:
        root.mkdir(parents=True)
        _write_json(root / "_request.json", request)
    _write_authority(root, state="running", request_sha256=request_sha256)

    records: list[dict[str, object]] = []
    baseline_selected: pd.DataFrame | None = None
    try:
        for index, spec in enumerate(specs, start=1):
            if progress is not None:
                progress(
                    {
                        "candidate": spec.candidate_id,
                        "position": index,
                        "total": len(specs),
                    }
                )
            candidate_dir = root / spec.candidate_id
            if candidate_dir.exists():
                records.append(
                    _load_complete_candidate(
                        candidate_dir,
                        expected_run_request_sha256=request_sha256,
                    )
                )
                if spec.candidate_family in {
                    "deterministic_v1_baseline",
                    "multinomial_v1_baseline",
                }:
                    baseline_selected = pd.read_parquet(
                        candidate_dir / "selected_predictions.parquet"
                    )
                continue
            result = evaluate_primary_v2_experiment(
                plan,
                spec,
                config=config,
                baseline_selected=baseline_selected,
            )
            record = _publish_candidate(
                candidate_dir,
                result=result,
                run_request_sha256=request_sha256,
                config=config,
            )
            records.append(record)
            if spec.candidate_family in {
                "deterministic_v1_baseline",
                "multinomial_v1_baseline",
            }:
                baseline_selected = result.selected_predictions.copy()
            del result
            release_process_memory()
            _assert_memory(config, f"{spec.candidate_id} publication")
        manifest: dict[str, object] = {
            "schema": PRIMARY_V2_RUN_SCHEMA,
            "strategy_id": strategy_id,
            "request_sha256": request_sha256,
            "source": source_identity,
            "split_sha256": plan.split_sha256,
            "candidates": records,
            "accepted_candidates": sorted(
                str(record["candidate_id"])
                for record in records
                if record["status"] == "accepted_development"
            ),
            "created_at_utc": datetime.now(UTC).isoformat(),
            "memory": memory_audit(
                hard_budget_gib=config.maximum_process_memory_gib,
                headroom_gib=config.memory_guard_headroom_gib,
            ).to_record(),
        }
        _write_json(root / "_manifest.json", manifest)
        _write_authority(
            root,
            state="complete",
            request_sha256=request_sha256,
            artifact="_manifest.json",
            artifact_sha256=file_sha256(root / "_manifest.json"),
        )
        return manifest
    except Exception:
        _write_authority(root, state="failed", request_sha256=request_sha256)
        raise


def primary_v2_implementation_identity() -> dict[str, object]:
    files = {
        "contracts": Path(contracts.__file__).resolve(),
        "model": Path(model.__file__).resolve(),
        "experiments": Path(__file__).resolve(),
    }
    return {
        name: {
            "path": path.name,
            "sha256": file_sha256(path),
        }
        for name, path in files.items()
    }


def _load_verified_source(
    *,
    strategy_id: str,
    source_dir: Path,
    swing_config: SwingSpecialistResearchConfig,
    swing_policy_path: Path,
    intraday_config: IntradaySpecialistResearchConfig,
    intraday_policy_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if strategy_id == SWING_V2_ID:
        return _load_swing_source(
            source_dir,
            config=swing_config,
            policy_path=swing_policy_path,
        )
    if strategy_id == INTRADAY_V2_ID:
        return _load_intraday_source(
            source_dir,
            config=intraday_config,
            policy_path=intraday_policy_path,
        )
    raise DataReadinessError(f"unknown primary V2 strategy: {strategy_id}")


def _load_swing_source(
    source_dir: Path,
    *,
    config: SwingSpecialistResearchConfig,
    policy_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    directory = source_dir.resolve()
    manifest_path = directory / "_manifest.json"
    authority_path = directory / "_authority.json"
    manifest = _load_json(manifest_path)
    authority = _load_json(authority_path)
    if (
        manifest.get("schema") != "swing.specialist_dataset_bundle.v4"
        or authority.get("schema") != "swing.specialist_dataset_authority.v1"
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("request_sha256") != manifest.get("request_sha256")
    ):
        raise DataReadinessError("swing V2 source bundle authority does not verify")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list):
        raise DataReadinessError("swing V2 source manifest has no artifacts")
    matches = [
        cast(dict[str, object], record)
        for record in artifacts
        if isinstance(record, dict)
        and record.get("strategy_id")
        == "SWING.CROSS_SECTIONAL_MOMENTUM.5D.V1"
    ]
    if len(matches) != 1:
        raise DataReadinessError("swing V2 source strategy artifact is ambiguous")
    record = matches[0]
    relative = Path(str(record["path"]))
    if relative.is_absolute() or ".." in relative.parts:
        raise DataReadinessError("swing V2 source artifact path is unsafe")
    path = (directory / relative).resolve()
    if directory not in path.parents:
        raise DataReadinessError("swing V2 source artifact escapes its bundle")
    if (
        file_sha256(path) != record.get("sha256")
        or file_sha256(Path(f"{path}.manifest.json"))
        != record.get("manifest_sha256")
    ):
        raise DataReadinessError("swing V2 source artifact integrity failed")
    columns = specialist_training_columns(
        path,
        strategy_id="SWING.CROSS_SECTIONAL_MOMENTUM.5D.V1",
        config=config,
    )
    frame, child_manifest = load_canonical_artifact(
        path,
        expected_type="swing_specialist_dataset",
        allow_research=True,
        columns=columns,
    )
    child_inputs = cast(dict[str, object], child_manifest.get("inputs", {}))
    policy_hashes = {
        str(value)
        for key, value in child_inputs.items()
        if str(key).replace("\\", "/").endswith(
            "/swing_specialist_research.toml"
        )
    }
    if policy_hashes != {file_sha256(policy_path)}:
        raise DataReadinessError(
            "swing V2 source policy-file identity differs"
        )
    return frame, {
        "type": "verified_swing_specialist_dataset",
        "bundle_manifest_sha256": file_sha256(manifest_path),
        "artifact_sha256": file_sha256(path),
        "rows": len(frame),
        "policy_file_sha256": file_sha256(policy_path),
    }


def _load_intraday_source(
    source_dir: Path,
    *,
    config: IntradaySpecialistResearchConfig,
    policy_path: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    verified = verify_intraday_specialist_training_bundle(
        source_dir,
        config=config,
        policy_path=policy_path,
    )
    strategy_id = "INTRADAY.VWAP_REVERSION.30M.V1"
    columns = {
        "strategy_id",
        "horizon_minutes",
        "direction",
        "session_segment",
        "setup_id",
        "ticker",
        "session_date_et",
        "primary_benchmark",
        "sector",
        "industry",
        "market_cap_bucket",
        "liquidity_bucket",
        "price_feed",
        "adjustment",
        "feature_available_at_utc",
        "decision_time_utc",
        "entry_time_utc",
        "exit_time_utc",
        "label_available_at_utc",
        "label_window_end_utc",
        "label_eligible",
        "path_outcome",
        "path_outcome_bar",
        "target_before_stop_30m",
        "stop_before_target_30m",
        "path_timeout_30m",
        "path_realized_return_gross_30m",
        "path_realized_return_net_30m",
        "path_mfe_30m",
        "path_mae_30m",
        "path_excess_return_30m_vs_spy",
        "path_excess_return_30m_vs_qqq",
        "path_excess_return_30m_vs_sector",
        *config.technical_features,
    }
    frames: list[pd.DataFrame] = []
    for record in verified.strategy_files[strategy_id]:
        path = verified.directory / str(record["path"])
        frames.append(pd.read_parquet(path, columns=sorted(columns)))
    frame = pd.concat(frames, ignore_index=True)
    del frames
    return frame, {
        "type": "verified_intraday_specialist_training_dataset",
        "bundle_manifest_sha256": verified.manifest_sha256,
        "dataset_fingerprint": verified.dataset_fingerprint,
        "strategy_dataset_sha256": verified.strategy_dataset_sha256[strategy_id],
        "rows": len(frame),
        "policy_file_sha256": file_sha256(policy_path),
    }


def _publish_candidate(
    candidate_dir: Path,
    *,
    result: PrimaryV2ExperimentResult,
    run_request_sha256: str,
    config: PrimaryV2ResearchConfig,
) -> dict[str, object]:
    temporary = candidate_dir.with_name(
        f".{candidate_dir.name}.{uuid4().hex}.tmp"
    )
    temporary.mkdir(parents=True)
    try:
        request = {
            "schema": PRIMARY_V2_CANDIDATE_SCHEMA,
            "run_request_sha256": run_request_sha256,
            "strategy_id": result.spec.strategy_id,
            "candidate_id": result.spec.candidate_id,
        }
        _write_json(temporary / "_request.json", request)
        result.predictions.to_parquet(temporary / "predictions.parquet", index=False)
        result.selected_predictions.to_parquet(
            temporary / "selected_predictions.parquet",
            index=False,
        )
        result.economics.to_csv(temporary / "economics.csv", index=False)
        result.regime_evidence.to_csv(
            temporary / "regime_evidence.csv",
            index=False,
        )
        result.calibration_evidence.to_csv(
            temporary / "calibration_evidence.csv",
            index=False,
        )
        result.incremental_evidence.to_csv(
            temporary / "incremental_evidence.csv",
            index=False,
        )
        result.fold_audit.to_csv(temporary / "fold_audit.csv", index=False)
        _write_json(temporary / "metrics.json", result.metrics)
        if result.status == "accepted_development":
            if result.final_candidate is None:
                raise DataReadinessError(
                    "accepted V2 candidate has no final fitted bundle"
                )
            joblib.dump(
                result.final_candidate,
                temporary / "model.joblib",
                compress=3,
            )
        elif result.final_candidate is not None:
            raise DataReadinessError("rejected V2 candidate retained a model")
        artifact_names = sorted(
            path.name
            for path in temporary.iterdir()
            if path.is_file()
        )
        artifacts = [
            {
                "path": name,
                "bytes": (temporary / name).stat().st_size,
                "sha256": file_sha256(temporary / name),
            }
            for name in artifact_names
        ]
        manifest = {
            "schema": PRIMARY_V2_CANDIDATE_SCHEMA,
            "candidate_id": result.spec.candidate_id,
            "status": result.status,
            "rejection_reasons": list(result.rejection_reasons),
            "request_sha256": _json_sha256(request),
            "artifacts": artifacts,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "memory": memory_audit(
                hard_budget_gib=config.maximum_process_memory_gib,
                headroom_gib=config.memory_guard_headroom_gib,
            ).to_record(),
        }
        _write_json(temporary / "_manifest.json", manifest)
        _write_json(
            temporary / "_authority.json",
            {
                "schema": PRIMARY_V2_AUTHORITY_SCHEMA,
                "state": "complete",
                "request_sha256": manifest["request_sha256"],
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(temporary / "_manifest.json"),
            },
        )
        temporary.replace(candidate_dir)
        return {
            "candidate_id": result.spec.candidate_id,
            "status": result.status,
            "rejection_reasons": list(result.rejection_reasons),
            "manifest_sha256": file_sha256(candidate_dir / "_manifest.json"),
        }
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_complete_run(
    root: Path,
    *,
    expected_request_sha256: str,
) -> dict[str, object]:
    authority = _load_json(root / "_authority.json")
    manifest = _load_json(root / "_manifest.json")
    if (
        authority.get("schema") != PRIMARY_V2_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("request_sha256") != expected_request_sha256
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256")
        != file_sha256(root / "_manifest.json")
        or manifest.get("request_sha256") != expected_request_sha256
    ):
        raise DataReadinessError(
            "primary V2 output exists without matching complete authority"
        )
    raw_candidates = manifest.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        raise DataReadinessError("primary V2 run manifest has no candidates")
    expected_directories: set[str] = set()
    for raw in raw_candidates:
        if not isinstance(raw, dict):
            raise DataReadinessError("primary V2 run candidate record is invalid")
        candidate_id = str(raw.get("candidate_id", ""))
        if (
            not candidate_id
            or candidate_id in expected_directories
            or "/" in candidate_id
            or "\\" in candidate_id
        ):
            raise DataReadinessError("primary V2 run candidate identity is invalid")
        expected_directories.add(candidate_id)
        verified = _load_complete_candidate(
            root / candidate_id,
            expected_run_request_sha256=expected_request_sha256,
        )
        if (
            verified["status"] != raw.get("status")
            or verified["manifest_sha256"] != raw.get("manifest_sha256")
            or verified["rejection_reasons"] != raw.get("rejection_reasons")
        ):
            raise DataReadinessError(
                f"primary V2 run candidate summary differs: {candidate_id}"
            )
    actual_directories = {
        path.name for path in root.iterdir() if path.is_dir()
    }
    if actual_directories != expected_directories:
        raise DataReadinessError(
            "primary V2 run candidate directory set differs from manifest"
        )
    return manifest


def _load_complete_candidate(
    candidate_dir: Path,
    *,
    expected_run_request_sha256: str,
) -> dict[str, object]:
    request = _load_json(candidate_dir / "_request.json")
    manifest = _load_json(candidate_dir / "_manifest.json")
    authority = _load_json(candidate_dir / "_authority.json")
    request_sha256 = _json_sha256(request)
    if (
        request.get("run_request_sha256") != expected_run_request_sha256
        or authority.get("schema") != PRIMARY_V2_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("request_sha256") != request_sha256
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256")
        != file_sha256(candidate_dir / "_manifest.json")
        or manifest.get("request_sha256") != request_sha256
    ):
        raise DataReadinessError(
            f"primary V2 candidate authority does not verify: {candidate_dir}"
        )
    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise DataReadinessError("primary V2 candidate manifest has no artifacts")
    for raw in raw_artifacts:
        if not isinstance(raw, dict):
            raise DataReadinessError("primary V2 candidate artifact is invalid")
        path = candidate_dir / str(raw.get("path", ""))
        if (
            not path.is_file()
            or path.stat().st_size != int(cast(Any, raw.get("bytes", -1)))
            or file_sha256(path) != raw.get("sha256")
        ):
            raise DataReadinessError(
                f"primary V2 candidate artifact does not verify: {path}"
            )
    return {
        "candidate_id": str(manifest["candidate_id"]),
        "status": str(manifest["status"]),
        "rejection_reasons": list(
            cast(list[object], manifest.get("rejection_reasons", []))
        ),
        "manifest_sha256": file_sha256(candidate_dir / "_manifest.json"),
    }


def _write_authority(
    root: Path,
    *,
    state: str,
    request_sha256: str,
    artifact: str | None = None,
    artifact_sha256: str | None = None,
) -> None:
    if state not in {"running", "complete", "failed"}:
        raise ValueError("invalid primary V2 authority state")
    _write_json(
        root / "_authority.json",
        {
            "schema": PRIMARY_V2_AUTHORITY_SCHEMA,
            "state": state,
            "request_sha256": request_sha256,
            "artifact": artifact,
            "artifact_sha256": artifact_sha256,
            "updated_at_utc": datetime.now(UTC).isoformat(),
        },
    )


def _load_json(path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"primary V2 JSON is unreadable: {path}") from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"primary V2 JSON must contain an object: {path}")
    return {str(key): value for key, value in loaded.items()}


def _write_json(path: Path, payload: object) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                default=_json_default,
            ),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_default(value: object) -> object:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return cast(Any, value).item()
    raise TypeError(f"object is not JSON serializable: {type(value).__name__}")


def _json_sha256(payload: object) -> str:
    import hashlib

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _assert_memory(config: PrimaryV2ResearchConfig, stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
