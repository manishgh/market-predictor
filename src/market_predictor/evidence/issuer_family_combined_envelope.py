"""Structural verification for retained issuer-family v2 evidence envelopes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from market_predictor.canonical.contracts import CANONICAL_SCHEMA_VERSION
from market_predictor.canonical.store import (
    CANONICAL_MANIFEST_SCHEMA,
    file_sha256,
    manifest_path_for,
)
from market_predictor.core.errors import DataReadinessError

AUTHORITY_SCHEMA: Final = "edge_rebuild.issuer_event_family_authority.v2"
MANIFEST_SCHEMA: Final = "edge_rebuild.issuer_event_family_manifest.v2"
FAMILY_EVENTS_ARTIFACT_TYPE: Final = "issuer_event_family_events"
FAMILY_ASSIGNMENTS_ARTIFACT_TYPE: Final = "issuer_event_family_assignments"
FAMILY_COVERAGE_ARTIFACT_TYPE: Final = "issuer_event_family_coverage"
COHORT_AUDIT_ARTIFACT_TYPE: Final = "issuer_event_family_cohort_audit"
UNCLASSIFIED_EVENTS_ARTIFACT_TYPE: Final = "issuer_event_family_unclassified_events"
NEUTRAL_PROJECTION_SCHEMA: Final = "market_predictor.issuer_family_neutral_projection.v1"

ARTIFACT_SPECIFICATIONS: Final = {
    "events": ("family_events.parquet", FAMILY_EVENTS_ARTIFACT_TYPE),
    "assignments": ("family_assignments.parquet", FAMILY_ASSIGNMENTS_ARTIFACT_TYPE),
    "coverage": ("family_coverage.parquet", FAMILY_COVERAGE_ARTIFACT_TYPE),
    "cohort_audit": ("cohort_audit.parquet", COHORT_AUDIT_ARTIFACT_TYPE),
}


@dataclass(frozen=True, slots=True)
class VerifiedIssuerFamilyEnvelopeArtifact:
    name: str
    path: Path
    artifact_type: str
    rows: int
    artifact_sha256: str
    manifest_sha256: str
    manifest: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class VerifiedIssuerFamilyCombinedEnvelope:
    directory: Path
    authority_sha256: str
    request_sha256: str
    full_inventory_sha256: str
    neutral_projection_sha256: str
    artifacts: Mapping[str, VerifiedIssuerFamilyEnvelopeArtifact]
    unclassified_artifact_records: tuple[Mapping[str, object], ...]
    manifest: Mapping[str, object]
    authority: Mapping[str, object]


def verify_issuer_family_combined_envelope(
    directory: Path,
    *,
    expected_authority_sha256: str | None = None,
    expected_policy_sha256: str | None = None,
) -> VerifiedIssuerFamilyCombinedEnvelope:
    """Verify retained bytes without assigning horizon-specific semantics."""

    if directory.is_symlink():
        raise DataReadinessError("issuer-family combined directory cannot be a symlink")
    root = directory.resolve()
    manifest_path = root / "_manifest.json"
    authority_path = root / "_authority.json"
    _require_regular_file(root, manifest_path)
    _require_regular_file(root, authority_path)
    manifest = _json_object(manifest_path)
    authority = _json_object(authority_path)
    authority_sha256 = file_sha256(authority_path)
    if expected_authority_sha256 is not None and authority_sha256 != expected_authority_sha256:
        raise DataReadinessError("issuer-family combined envelope identity differs")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("production_ready") is not False
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("production_ready") is not False
    ):
        raise DataReadinessError("issuer-family combined envelope does not verify")

    request = manifest.get("request")
    if (
        not isinstance(request, dict)
        or request.get("schema") != AUTHORITY_SCHEMA
        or request.get("classifier_policy_sha256")
        != manifest.get("event_family_policy_sha256")
        or request.get("production_ready") is not False
    ):
        raise DataReadinessError("issuer-family combined request contract differs")
    request_sha256 = _json_sha256(request)
    policy_sha256 = manifest.get("event_family_policy_sha256")
    if (
        request_sha256 != manifest.get("request_sha256")
        or authority.get("request_sha256") != request_sha256
        or authority.get("event_family_policy_sha256") != policy_sha256
        or (expected_policy_sha256 is not None and policy_sha256 != expected_policy_sha256)
    ):
        raise DataReadinessError("issuer-family combined request or policy identity differs")

    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, dict) or set(raw_artifacts) != set(ARTIFACT_SPECIFICATIONS):
        raise DataReadinessError("issuer-family combined artifact inventory is malformed")
    expected_inventory = {"_authority.json", "_manifest.json"}
    artifacts: dict[str, VerifiedIssuerFamilyEnvelopeArtifact] = {}
    for name, (filename, artifact_type) in ARTIFACT_SPECIFICATIONS.items():
        artifact = root / filename
        child_path = manifest_path_for(artifact)
        _require_regular_file(root, artifact)
        _require_regular_file(root, child_path)
        record = raw_artifacts.get(name)
        child = _json_object(child_path)
        rows = _nonnegative_int(child.get("rows"), f"{name} child rows")
        record_rows = _nonnegative_int(
            record.get("rows") if isinstance(record, dict) else None,
            f"{name} record rows",
        )
        child_inputs = child.get("inputs")
        artifact_sha256 = file_sha256(artifact)
        if (
            not isinstance(record, dict)
            or record.get("path") != filename
            or record_rows != rows
            or record.get("sha256") != artifact_sha256
            or child.get("artifact_sha256") != artifact_sha256
            or child.get("schema") != CANONICAL_MANIFEST_SCHEMA
            or child.get("canonical_schema_version") != CANONICAL_SCHEMA_VERSION
            or child.get("artifact_type") != artifact_type
            or child.get("artifact_path") != str(artifact.resolve())
            or child.get("production_ready") is not False
            or not _valid_columns(child.get("columns"))
            or not isinstance(child_inputs, dict)
            or child_inputs.get("request_sha256") != request_sha256
        ):
            raise DataReadinessError(f"issuer-family combined {name} lineage differs")
        artifacts[name] = VerifiedIssuerFamilyEnvelopeArtifact(
            name=name,
            path=artifact,
            artifact_type=artifact_type,
            rows=rows,
            artifact_sha256=artifact_sha256,
            manifest_sha256=file_sha256(child_path),
            manifest=child,
        )
        expected_inventory.update((filename, child_path.name))

    unclassified_records = _unclassified_records(manifest)
    unclassified_inventory = _verify_unclassified_inventory(
        root,
        unclassified_records,
        request_sha256=request_sha256,
        expected_rows=manifest.get("unclassified_event_rows"),
    )
    expected_inventory.update(unclassified_inventory)
    observed_paths = tuple(root.rglob("*"))
    if any(path.is_symlink() for path in observed_paths):
        raise DataReadinessError("issuer-family combined inventory cannot use symlinks")
    observed_inventory = {
        path.relative_to(root).as_posix()
        for path in observed_paths
        if path.is_file()
    }
    if observed_inventory != expected_inventory:
        raise DataReadinessError("issuer-family combined recursive inventory differs")
    full_inventory_sha256 = _json_sha256(
        [
            {"path": relative, "sha256": file_sha256(root / relative)}
            for relative in sorted(observed_inventory)
        ]
    )
    neutral_projection_sha256 = _json_sha256(
        {
            "schema": NEUTRAL_PROJECTION_SCHEMA,
            "event_family_policy_sha256": policy_sha256,
            "source_lineage": _neutral_source_lineage(request),
            "event_families": manifest.get("event_families"),
            "research_source_families": manifest.get("research_source_families"),
            "events": _projection_record(artifacts["events"]),
            "coverage": _projection_record(artifacts["coverage"]),
            "unclassified": [
                {
                    "path": _required_text(record, "path"),
                    "rows": _nonnegative_int(record.get("rows"), "unclassified rows"),
                    "sha256": _required_text(record, "sha256"),
                    "source_event_sha256": _required_text(record, "source_event_sha256"),
                    "relation_sha256": _required_text(record, "relation_sha256"),
                }
                for record in unclassified_records
            ],
        }
    )
    return VerifiedIssuerFamilyCombinedEnvelope(
        directory=root,
        authority_sha256=authority_sha256,
        request_sha256=request_sha256,
        full_inventory_sha256=full_inventory_sha256,
        neutral_projection_sha256=neutral_projection_sha256,
        artifacts=artifacts,
        unclassified_artifact_records=tuple(unclassified_records),
        manifest=manifest,
        authority=authority,
    )


def _verify_unclassified_inventory(
    root: Path,
    records: tuple[Mapping[str, object], ...],
    *,
    request_sha256: str,
    expected_rows: object,
) -> set[str]:
    child_directory = root / "unclassified"
    if child_directory.is_symlink() or not child_directory.is_dir():
        raise DataReadinessError("issuer-family unclassified directory is missing")
    expected_inventory: set[str] = set()
    total_rows = 0
    for record in records:
        chunk_id = _required_text(record, "chunk_id")
        relative = _required_text(record, "path")
        expected_relative = f"unclassified/{chunk_id}.parquet"
        if relative != expected_relative:
            raise DataReadinessError("issuer-family unclassified path differs")
        artifact = root / relative
        if root not in artifact.resolve().parents:
            raise DataReadinessError("issuer-family unclassified path escapes authority")
        child_path = manifest_path_for(artifact)
        _require_regular_file(root, artifact)
        _require_regular_file(root, child_path)
        child = _json_object(child_path)
        child_inputs = child.get("inputs")
        rows = _positive_int(record.get("rows"), "unclassified record rows")
        artifact_sha256 = file_sha256(artifact)
        if (
            child.get("artifact_type") != UNCLASSIFIED_EVENTS_ARTIFACT_TYPE
            or child.get("artifact_sha256") != artifact_sha256
            or child.get("schema") != CANONICAL_MANIFEST_SCHEMA
            or child.get("canonical_schema_version") != CANONICAL_SCHEMA_VERSION
            or child.get("artifact_path") != str(artifact.resolve())
            or child.get("rows") != rows
            or child.get("production_ready") is not False
            or not _valid_columns(child.get("columns"))
            or record.get("sha256") != artifact_sha256
            or not isinstance(child_inputs, dict)
            or child_inputs.get("request_sha256") != request_sha256
            or child_inputs.get("source_event_sha256") != record.get("source_event_sha256")
            or child_inputs.get("relation_sha256") != record.get("relation_sha256")
        ):
            raise DataReadinessError("issuer-family unclassified lineage differs")
        total_rows += rows
        expected_inventory.add(artifact.relative_to(root).as_posix())
        expected_inventory.add(child_path.relative_to(root).as_posix())
    if total_rows != _nonnegative_int(expected_rows, "unclassified row total"):
        raise DataReadinessError("issuer-family unclassified row total differs")
    return expected_inventory


def _unclassified_records(manifest: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw = manifest.get("unclassified_artifacts")
    if not isinstance(raw, list):
        raise DataReadinessError("issuer-family unclassified inventory is malformed")
    records: list[Mapping[str, object]] = []
    chunk_ids: set[str] = set()
    for value in raw:
        if not isinstance(value, dict):
            raise DataReadinessError("issuer-family unclassified record is malformed")
        chunk_id = _required_text(value, "chunk_id")
        if chunk_id in chunk_ids:
            raise DataReadinessError("issuer-family unclassified chunks are duplicated")
        chunk_ids.add(chunk_id)
        records.append(value)
    return tuple(records)


def _projection_record(artifact: VerifiedIssuerFamilyEnvelopeArtifact) -> dict[str, object]:
    child = artifact.manifest
    return {
        "path": artifact.path.name,
        "rows": artifact.rows,
        "sha256": artifact.artifact_sha256,
        "artifact_type": artifact.artifact_type,
        "canonical_manifest_schema": child.get("schema"),
        "canonical_schema_version": child.get("canonical_schema_version"),
        "columns": child.get("columns"),
    }


def _neutral_source_lineage(request: Mapping[str, object]) -> dict[str, str]:
    keys = (
        "collection_manifest_sha256",
        "collection_audit_sha256",
        "attribution_manifest_sha256",
        "security_identities_sha256",
        "source_coverage_sha256",
        "classifier_policy_sha256",
    )
    return {key: _required_text(request, key) for key in keys}


def _require_regular_file(root: Path, path: Path) -> None:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise DataReadinessError("issuer-family artifact path escapes authority") from exc
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise DataReadinessError("issuer-family artifacts cannot use symlinks")
    if not path.is_file() or root not in path.resolve().parents:
        raise DataReadinessError(f"issuer-family artifact is not a confined file: {path}")


def _valid_columns(value: object) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(column, str) and bool(column) for column in value)
        and len(value) == len(set(value))
    )


def _json_object(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataReadinessError(f"expected JSON object: {path}")
    return {str(key): item for key, item in value.items()}


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _required_text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DataReadinessError(f"issuer-family envelope has invalid {key}")
    return value.strip()


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DataReadinessError(f"issuer-family envelope has invalid {name}")
    return value


def _positive_int(value: object, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise DataReadinessError(f"issuer-family envelope has invalid {name}")
    return result
