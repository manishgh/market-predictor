"""Immutable rejection evidence for superseded intraday candidates."""

from __future__ import annotations

import json
import shutil
import tempfile
import tomllib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from market_predictor.canonical.store import file_sha256
from market_predictor.v3.errors import DataReadinessError

REJECTION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_candidate_rejection.v1"
REJECTION_AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_candidate_rejection_authority.v1"


def publish_intraday_candidate_rejection(
    candidate_directory: Path,
    rejection_policy_path: Path,
    output_directory: Path,
) -> Mapping[str, Any]:
    """Verify an immutable candidate and publish separately bound rejection evidence."""

    policy = _load_policy(rejection_policy_path)
    expected_directory = str(policy.get("candidate_directory"))
    if candidate_directory.as_posix().rstrip("/") != expected_directory.rstrip("/"):
        raise DataReadinessError("candidate directory differs from the frozen rejection policy")
    expected_files = _object(policy.get("candidate_files"), "candidate_files")
    actual_files: dict[str, dict[str, Any]] = {}
    for name, expected_raw in expected_files.items():
        expected = _object(expected_raw, f"candidate_files.{name}")
        path = candidate_directory / name
        if not path.is_file():
            raise DataReadinessError(f"candidate evidence file is missing: {name}")
        sha256 = file_sha256(path)
        size = path.stat().st_size
        if sha256 != expected.get("sha256") or size != int(expected.get("bytes", -1)):
            raise DataReadinessError(f"candidate evidence identity differs: {name}")
        actual_files[name] = {"sha256": sha256, "bytes": size}

    evaluation = _read_json(candidate_directory / "evaluation.json", "candidate evaluation")
    dataset = _object(evaluation.get("dataset"), "candidate evaluation dataset")
    expected_dataset = _object(policy.get("dataset"), "dataset")
    if dataset != expected_dataset:
        raise DataReadinessError("candidate dataset authority differs from rejection policy")
    reasons = policy.get("reasons")
    if not isinstance(reasons, list) or not reasons or not all(isinstance(reason, Mapping) for reason in reasons):
        raise DataReadinessError("rejection policy requires structured reasons")
    evidence = {
        "schema_version": REJECTION_SCHEMA_VERSION,
        "status": "rejected",
        "serving_eligible": False,
        "promotion_permitted": False,
        "candidate_reference_id": str(policy.get("candidate_reference_id")),
        "candidate_directory": expected_directory,
        "candidate_files": actual_files,
        "dataset": dataset,
        "reasons": reasons,
        "locked_test_status": "opened_and_retired_from_model_selection",
        "remediation_contract": {
            "development_data_latest_session": "2026-07-08",
            "future_holdout_earliest_session": "2026-07-09",
            "future_holdout_must_be_separate_authority": True,
        },
        "rejection_policy_sha256": file_sha256(rejection_policy_path),
    }
    _publish(output_directory, evidence)
    return evidence


def _publish(output: Path, evidence: Mapping[str, Any]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        evidence_path = temporary / "rejection.json"
        _write_json(evidence_path, evidence)
        manifest = {
            "schema_version": REJECTION_SCHEMA_VERSION,
            "state": "rejected",
            "serving_eligible": False,
            "promotion_permitted": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "files": {
                "rejection.json": {
                    "sha256": file_sha256(evidence_path),
                    "bytes": evidence_path.stat().st_size,
                }
            },
        }
        manifest_path = temporary / "_manifest.json"
        _write_json(manifest_path, manifest)
        _write_json(
            temporary / "_authority.json",
            {
                "schema_version": REJECTION_AUTHORITY_SCHEMA_VERSION,
                "state": "rejected",
                "serving_eligible": False,
                "promotion_permitted": False,
                "manifest_path": "_manifest.json",
                "manifest_sha256": file_sha256(manifest_path),
            },
        )
        try:
            temporary.rename(output)
        except FileExistsError:
            raise FileExistsError(f"immutable rejection already exists: {output}") from None
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_policy(path: Path) -> dict[str, Any]:
    try:
        value = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(f"rejection policy is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError("rejection policy must be an object")
    return value


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"{label} is invalid") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"{label} must be an object")
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DataReadinessError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}
