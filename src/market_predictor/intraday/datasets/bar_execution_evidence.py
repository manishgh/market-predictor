"""Operational evidence for resumable intraday dataset publication."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.contracts.lineage import (
    DEFAULT_INTRADAY_CONTRACT_LINEAGE_PATH,
)
from market_predictor.intraday.datasets.bar_dataset import (
    MEMORY_HARD_BUDGET_GIB,
    MEMORY_HEADROOM_GIB,
    load_complete_intraday_bar_dataset,
    publish_intraday_bar_dataset,
)
from market_predictor.intraday.datasets.history import json_sha256
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.resources import memory_audit

INTRADAY_BAR_EXECUTION_RECEIPT_SCHEMA: Final = "market_predictor.intraday.bar_dataset_execution_receipt.v1"
INTRADAY_BAR_EXECUTION_MANIFEST_SCHEMA: Final = "market_predictor.intraday.bar_dataset_execution_evidence.v1"
INTRADAY_BAR_EXECUTION_AUTHORITY_SCHEMA: Final = "market_predictor.intraday.bar_dataset_execution_authority.v1"
INTRADAY_BAR_EXECUTION_ASSESSMENT_SCHEMA: Final = "market_predictor.intraday.bar_dataset_execution_assessment.v1"
_METADATA_FILES: Final = frozenset({"_manifest.json", "_authority.json"})


def publish_intraday_bar_dataset_with_execution_evidence(
    *,
    selection_directory: Path,
    stock_collection_directory: Path,
    stock_coverage_directory: Path,
    benchmark_collection_directory: Path,
    membership_authority_directory: Path,
    five_minute_projection_directory: Path,
    strategy_contract: StrategyContract,
    strategy_contract_path: Path,
    output_directory: Path,
    execution_evidence_directory: Path,
    intraday_contract_lineage_path: Path = DEFAULT_INTRADAY_CONTRACT_LINEAGE_PATH,
    max_sessions_per_invocation: int | None = None,
    session_workers: int = 1,
) -> dict[str, Any]:
    """Publish data while retaining hash-bound evidence for every invocation."""

    inputs = (
        selection_directory,
        stock_collection_directory,
        stock_coverage_directory,
        benchmark_collection_directory,
        membership_authority_directory,
        five_minute_projection_directory,
        strategy_contract_path,
        intraday_contract_lineage_path,
        output_directory,
    )
    _require_evidence_isolation(execution_evidence_directory, inputs)

    def publish_dataset() -> dict[str, Any]:
        return publish_intraday_bar_dataset(
            selection_directory=selection_directory,
            stock_collection_directory=stock_collection_directory,
            stock_coverage_directory=stock_coverage_directory,
            benchmark_collection_directory=benchmark_collection_directory,
            membership_authority_directory=membership_authority_directory,
            five_minute_projection_directory=five_minute_projection_directory,
            strategy_contract=strategy_contract,
            strategy_contract_path=strategy_contract_path,
            output_directory=output_directory,
            intraday_contract_lineage_path=intraday_contract_lineage_path,
            max_sessions_per_invocation=max_sessions_per_invocation,
            session_workers=session_workers,
        )

    if execution_evidence_directory.exists():
        load_complete_intraday_bar_dataset_execution_evidence(
            execution_evidence_directory,
            dataset_directory=output_directory,
        )
        result = publish_dataset()
        load_complete_intraday_bar_dataset_execution_evidence(
            execution_evidence_directory,
            dataset_directory=output_directory,
        )
        return result

    work = execution_evidence_directory.with_name(f".{execution_evidence_directory.name}.work")
    if _recover_execution_work(
        work,
        output_directory=output_directory,
        execution_evidence_directory=execution_evidence_directory,
    ):
        result = publish_dataset()
        load_complete_intraday_bar_dataset_execution_evidence(
            execution_evidence_directory,
            dataset_directory=output_directory,
        )
        return result
    receipts = work / "invocations"
    receipts.mkdir(parents=True, exist_ok=True)
    invocation_id = uuid.uuid4().hex
    receipt_path = receipts / f"{invocation_id}.json"
    started_at = datetime.now(UTC).isoformat()
    before = _completed_sessions(output_directory)
    started: dict[str, Any] = {
        "schema": INTRADAY_BAR_EXECUTION_RECEIPT_SCHEMA,
        "invocation_id": invocation_id,
        "state": "started",
        "started_at_utc": started_at,
        "completed_at_utc": None,
        "output_directory": str(output_directory.resolve()),
        "input_directories": {
            "selection": str(selection_directory.resolve()),
            "stock_collection": str(stock_collection_directory.resolve()),
            "stock_coverage": str(stock_coverage_directory.resolve()),
            "benchmark_collection": str(benchmark_collection_directory.resolve()),
            "membership_authority": str(membership_authority_directory.resolve()),
            "five_minute_projection": str(five_minute_projection_directory.resolve()),
        },
        "strategy_contract_path": str(strategy_contract_path.resolve()),
        "intraday_contract_lineage_path": str(intraday_contract_lineage_path.resolve()),
        "max_sessions_per_invocation": max_sessions_per_invocation,
        "session_workers": session_workers,
        "request_sha256": None,
        "transformation_sha256": None,
        "processed_sessions": [],
        "completed_sessions_after_invocation": len(before),
        "memory": None,
        "exception": None,
    }
    _atomic_write_json(receipt_path, started)
    try:
        result = publish_dataset()
    except BaseException as exc:
        after = _completed_sessions(output_directory)
        identity = _discover_request_identity(output_directory)
        failed = {
            **started,
            "state": "failed",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            **identity,
            "processed_sessions": sorted(after - before),
            "completed_sessions_after_invocation": len(after),
            "memory": memory_audit(
                hard_budget_gib=MEMORY_HARD_BUDGET_GIB,
                headroom_gib=MEMORY_HEADROOM_GIB,
            ).to_record(),
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc)[:2000],
            },
        }
        _atomic_write_json(receipt_path, failed, replace=True)
        raise

    after = _completed_sessions(output_directory)
    identity = _discover_request_identity(output_directory, result=result)
    state = str(result.get("state", ""))
    if state not in {"work_incomplete", "complete"}:
        raise DataReadinessError(f"intraday dataset publisher returned an invalid state: {state}")
    summary = result.get("summary")
    if not isinstance(summary, Mapping) or not isinstance(summary.get("memory"), Mapping):
        raise DataReadinessError("intraday dataset publication omitted memory evidence")
    completed = {
        **started,
        "state": state,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        **identity,
        "processed_sessions": sorted(after - before),
        "completed_sessions_after_invocation": len(after),
        "memory": dict(cast(Mapping[str, Any], summary["memory"])),
    }
    _atomic_write_json(receipt_path, completed, replace=True)
    if state == "complete":
        _publish_execution_authority(
            work,
            output_directory=output_directory,
            execution_evidence_directory=execution_evidence_directory,
        )
    return result


def load_complete_intraday_bar_dataset_execution_evidence(
    directory: Path,
    *,
    dataset_directory: Path,
) -> dict[str, Any]:
    """Verify receipt inventory, memory gates, and exact dataset binding."""

    manifest = _read_json(directory / "_manifest.json")
    authority = _read_json(directory / "_authority.json")
    dataset = load_complete_intraday_bar_dataset(dataset_directory)
    inventory = manifest.get("invocations")
    if not isinstance(inventory, list) or not inventory:
        raise DataReadinessError("intraday execution evidence has no invocations")
    receipts = []
    for raw in cast(list[Mapping[str, Any]], inventory):
        relative = str(raw.get("path", ""))
        path = _resolve_inside(directory, relative)
        if file_sha256(path) != raw.get("sha256"):
            raise DataReadinessError("intraday execution receipt hash differs")
        receipts.append(_read_json(path))
    _validate_execution_receipts(
        receipts,
        dataset=dataset,
        dataset_directory=dataset_directory,
    )
    actual_files = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    expected_files = set(_METADATA_FILES) | {str(item["path"]) for item in cast(list[Mapping[str, Any]], inventory)}
    if (
        manifest.get("schema") != INTRADAY_BAR_EXECUTION_MANIFEST_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("dataset_directory") != str(dataset_directory.resolve())
        or manifest.get("dataset_manifest_sha256") != file_sha256(dataset_directory / "_manifest.json")
        or manifest.get("dataset_authority_sha256") != file_sha256(dataset_directory / "_authority.json")
        or manifest.get("dataset_request_sha256") != dataset.get("request_sha256")
        or manifest.get("dataset_transformation_sha256") != dataset.get("transformation_sha256")
        or manifest.get("invocation_inventory_sha256") != json_sha256(inventory)
        or authority.get("schema") != INTRADAY_BAR_EXECUTION_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(directory / "_manifest.json")
        or authority.get("invocation_inventory_sha256") != manifest.get("invocation_inventory_sha256")
        or actual_files != expected_files
    ):
        raise DataReadinessError("intraday execution evidence authority differs")
    return manifest


def publish_incomplete_intraday_bar_execution_assessment(
    *,
    dataset_directory: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Record that a pre-existing resumed build lacks full invocation receipts."""

    dataset = load_complete_intraday_bar_dataset(dataset_directory)
    summary = dataset.get("summary")
    recorded_memory = (
        dict(cast(Mapping[str, Any], summary["memory"]))
        if isinstance(summary, Mapping) and isinstance(summary.get("memory"), Mapping)
        else None
    )
    report = {
        "schema": INTRADAY_BAR_EXECUTION_ASSESSMENT_SCHEMA,
        "status": "incomplete",
        "recorded_scope": "final_invocation_only",
        "complete_run_memory_proven": False,
        "reason": (
            "publication completed before per-invocation execution receipts were "
            "required; earlier invocation memory cannot be reconstructed"
        ),
        "dataset_directory": str(dataset_directory.resolve()),
        "dataset_manifest_sha256": file_sha256(dataset_directory / "_manifest.json"),
        "dataset_authority_sha256": file_sha256(dataset_directory / "_authority.json"),
        "dataset_request_sha256": str(dataset["request_sha256"]),
        "dataset_transformation_sha256": str(dataset["transformation_sha256"]),
        "recorded_memory": recorded_memory,
    }
    _publish_immutable_json(output_path, report)
    return report


def _publish_execution_authority(
    work: Path,
    *,
    output_directory: Path,
    execution_evidence_directory: Path,
) -> None:
    dataset = load_complete_intraday_bar_dataset(output_directory)
    receipt_paths = sorted((work / "invocations").glob("*.json"))
    receipts = [_read_json(path) for path in receipt_paths]
    summary = _validate_execution_receipts(
        receipts,
        dataset=dataset,
        dataset_directory=output_directory,
    )
    inventory = [
        {
            "path": path.relative_to(work).as_posix(),
            "sha256": file_sha256(path),
        }
        for path in receipt_paths
    ]
    manifest = {
        "schema": INTRADAY_BAR_EXECUTION_MANIFEST_SCHEMA,
        "state": "complete",
        "dataset_directory": str(output_directory.resolve()),
        "dataset_manifest_sha256": file_sha256(output_directory / "_manifest.json"),
        "dataset_authority_sha256": file_sha256(output_directory / "_authority.json"),
        "dataset_request_sha256": str(dataset["request_sha256"]),
        "dataset_transformation_sha256": str(dataset["transformation_sha256"]),
        "invocations": inventory,
        "invocation_inventory_sha256": json_sha256(inventory),
        "summary": summary,
    }
    _atomic_write_json(work / "_manifest.json", manifest)
    _atomic_write_json(
        work / "_authority.json",
        {
            "schema": INTRADAY_BAR_EXECUTION_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(work / "_manifest.json"),
            "invocation_inventory_sha256": manifest["invocation_inventory_sha256"],
        },
    )
    load_complete_intraday_bar_dataset_execution_evidence(
        work,
        dataset_directory=output_directory,
    )
    execution_evidence_directory.parent.mkdir(parents=True, exist_ok=True)
    work.replace(execution_evidence_directory)


def _validate_execution_receipts(
    receipts: list[dict[str, Any]],
    *,
    dataset: Mapping[str, Any],
    dataset_directory: Path,
) -> dict[str, Any]:
    planned = set(str(value) for value in cast(list[Any], dataset["planned_sessions"]))
    request_sha = str(dataset["request_sha256"])
    transformation_sha = str(dataset["transformation_sha256"])
    accounted: set[str] = set()
    aggregate_peak = 0.0
    complete_receipts = 0
    for receipt in receipts:
        state = str(receipt.get("state", ""))
        processed = receipt.get("processed_sessions")
        memory = receipt.get("memory")
        if (
            receipt.get("schema") != INTRADAY_BAR_EXECUTION_RECEIPT_SCHEMA
            or state not in {"work_incomplete", "complete", "failed"}
            or not receipt.get("completed_at_utc")
            or receipt.get("output_directory") != str(dataset_directory.resolve())
            or not isinstance(processed, list)
            or not isinstance(memory, Mapping)
        ):
            raise DataReadinessError("intraday execution receipt is incomplete")
        processed_set = {str(value) for value in processed}
        if len(processed_set) != len(processed) or accounted & processed_set:
            raise DataReadinessError("intraday execution receipts duplicate sessions")
        if processed_set and (receipt.get("request_sha256") != request_sha or receipt.get("transformation_sha256") != transformation_sha):
            raise DataReadinessError("intraday execution receipt identity differs")
        recorded_peak = memory.get("peak_working_set_gib")
        if recorded_peak is None or not math.isfinite(float(recorded_peak)) or float(recorded_peak) < 0:
            raise DataReadinessError("intraday execution receipt omits peak memory")
        parent_peak = float(recorded_peak)
        hard_budget = float(memory.get("hard_budget_gib", -1.0))
        threshold = float(memory.get("safety_threshold_gib", -1.0))
        if hard_budget != MEMORY_HARD_BUDGET_GIB or threshold != MEMORY_HARD_BUDGET_GIB - MEMORY_HEADROOM_GIB:
            raise DataReadinessError("intraday execution receipt memory contract differs")
        raw_aggregate = memory.get("aggregate_peak_upper_bound_gib")
        workers = int(receipt.get("session_workers", -1))
        if raw_aggregate is None:
            if workers != 1:
                raise DataReadinessError("intraday execution receipt omits aggregate memory")
            raw_aggregate = recorded_peak
        recorded_aggregate = float(raw_aggregate)
        if not math.isfinite(recorded_aggregate) or parent_peak > recorded_aggregate or recorded_aggregate > threshold:
            raise DataReadinessError("intraday execution receipt breached memory budget")
        aggregate_peak = max(aggregate_peak, recorded_aggregate)
        accounted.update(processed_set)
        complete_receipts += int(state == "complete")
    if accounted != planned or complete_receipts != 1:
        raise DataReadinessError("intraday execution evidence does not cover the dataset")
    return {
        "invocations": len(receipts),
        "accounted_sessions": len(accounted),
        "maximum_recorded_aggregate_peak_working_set_gib": aggregate_peak,
        "hard_budget_gib": MEMORY_HARD_BUDGET_GIB,
        "complete_run_memory_proven": True,
    }


def _recover_execution_work(
    work: Path,
    *,
    output_directory: Path,
    execution_evidence_directory: Path,
) -> bool:
    if not work.is_dir() or not output_directory.is_dir():
        return False
    for temporary in work.glob(".*.tmp"):
        temporary.unlink(missing_ok=True)
    manifest = work / "_manifest.json"
    authority = work / "_authority.json"
    if manifest.is_file() and authority.is_file():
        load_complete_intraday_bar_dataset_execution_evidence(
            work,
            dataset_directory=output_directory,
        )
        execution_evidence_directory.parent.mkdir(parents=True, exist_ok=True)
        work.replace(execution_evidence_directory)
        return True
    if manifest.exists() or authority.exists() or (work / "invocations").is_dir():
        manifest.unlink(missing_ok=True)
        authority.unlink(missing_ok=True)
        _publish_execution_authority(
            work,
            output_directory=output_directory,
            execution_evidence_directory=execution_evidence_directory,
        )
        return True
    return False


def _completed_sessions(output_directory: Path) -> set[str]:
    roots = []
    if output_directory.is_dir():
        roots.append(output_directory)
    roots.extend(path for path in output_directory.parent.glob(f".{output_directory.name}.*.work") if path.is_dir())
    if len(roots) > 1:
        raise DataReadinessError("multiple intraday dataset work authorities exist")
    if not roots:
        return set()
    sessions = roots[0] / "sessions"
    if not sessions.is_dir():
        return set()
    return {
        path.name.removeprefix("session_date_et=")
        for path in sessions.iterdir()
        if path.is_dir() and path.name.startswith("session_date_et=") and (path / "_unit.json").is_file()
    }


def _discover_request_identity(
    output_directory: Path,
    *,
    result: Mapping[str, Any] | None = None,
) -> dict[str, str | None]:
    candidates = []
    if output_directory.is_dir():
        candidates.append(output_directory / "_request.json")
    if result is not None and result.get("work_directory"):
        candidates.append(Path(str(result["work_directory"])) / "_request.json")
    candidates.extend(path / "_request.json" for path in output_directory.parent.glob(f".{output_directory.name}.*.work") if path.is_dir())
    existing = sorted({path.resolve() for path in candidates if path.is_file()})
    if len(existing) > 1:
        raise DataReadinessError("multiple intraday dataset request identities exist")
    if not existing:
        return {"request_sha256": None, "transformation_sha256": None}
    request = _read_json(existing[0])
    return {
        "request_sha256": str(request.get("request_sha256") or "") or None,
        "transformation_sha256": (str(request.get("transformation_sha256") or "") or None),
    }


def _require_evidence_isolation(output: Path, inputs: tuple[Path, ...]) -> None:
    target = output.resolve()
    for source in inputs:
        resolved = source.resolve()
        if target == resolved or target in resolved.parents or resolved in target.parents:
            raise DataReadinessError("intraday execution evidence overlaps an authority")


def _resolve_inside(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise DataReadinessError("intraday execution receipt path is invalid")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise DataReadinessError("intraday execution receipt escapes its authority")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"intraday execution JSON is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError("intraday execution JSON must be an object")
    return {str(key): item for key, item in value.items()}


def _atomic_write_json(
    path: Path,
    value: Mapping[str, Any],
    *,
    replace: bool = False,
) -> None:
    if path.exists() and not replace:
        raise DataReadinessError(f"intraday execution artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_immutable_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != dict(value):
            raise DataReadinessError(f"intraday execution assessment is immutable and differs: {path}")
        return
    _atomic_write_json(path, value)
