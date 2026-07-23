from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditReport
from market_predictor.canonical.contracts import CANONICAL_SCHEMA_VERSION
from market_predictor.locking import file_lock
from market_predictor.v3.errors import DataReadinessError

CANONICAL_MANIFEST_SCHEMA = "market_data.artifact_manifest.v1"


def manifest_path_for(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def write_canonical_artifact(
    frame: pd.DataFrame,
    path: Path,
    *,
    artifact_type: str,
    audit: CanonicalAuditReport,
    inputs: Mapping[str, str] | None = None,
    production_ready: bool = True,
) -> dict[str, object]:
    """Atomically publish a canonical table and its integrity manifest.

    Concurrent publishers to the same path are serialized with a file lock, both
    files are staged before either is published, and the manifest (the reader's
    integrity gate) is renamed last so it never references a stale table.
    """

    if not audit.passed:
        audit.raise_for_failure()
    path.parent.mkdir(parents=True, exist_ok=True)
    availability_column = next(
        (
            column
            for column in ("feature_available_at_utc", "available_at_utc", "decision_time_utc")
            if column in frame.columns
        ),
        None,
    )
    availability = (
        pd.to_datetime(frame[availability_column], utc=True, errors="coerce")
        if availability_column is not None
        else pd.Series(dtype="datetime64[ns, UTC]")
    )
    manifest_path = manifest_path_for(path)
    with file_lock(path):
        data_temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        manifest_temporary = manifest_path.with_name(f".{manifest_path.name}.{uuid.uuid4().hex}.tmp")
        try:
            frame.to_parquet(data_temporary, index=False)
            manifest: dict[str, object] = {
                "schema": CANONICAL_MANIFEST_SCHEMA,
                "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
                "artifact_type": artifact_type.strip().lower(),
                "artifact_path": str(path),
                "artifact_sha256": file_sha256(data_temporary),
                "created_at_utc": datetime.now(UTC).isoformat(),
                "rows": len(frame),
                "columns": list(frame.columns),
                "first_available_at_utc": _iso(availability.min()) if not availability.empty else None,
                "last_available_at_utc": _iso(availability.max()) if not availability.empty else None,
                "inputs": dict(inputs or {}),
                "audit": [check.model_dump() for check in audit.checks],
                "production_ready": production_ready,
            }
            manifest_temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            data_temporary.replace(path)
            manifest_temporary.replace(manifest_path)
            return manifest
        finally:
            data_temporary.unlink(missing_ok=True)
            manifest_temporary.unlink(missing_ok=True)


def load_canonical_artifact(
    path: Path,
    *,
    expected_type: str | None = None,
    allow_research: bool = False,
) -> tuple[pd.DataFrame, dict[str, object]]:
    manifest_path = manifest_path_for(path)
    if not path.exists() or not manifest_path.exists():
        raise FileNotFoundError(f"canonical artifact or manifest is missing: {path}")
    loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"canonical manifest must contain an object: {manifest_path}")
    manifest = {str(key): value for key, value in loaded.items()}
    if manifest.get("schema") != CANONICAL_MANIFEST_SCHEMA:
        raise DataReadinessError(f"unsupported canonical manifest schema: {manifest_path}")
    if manifest.get("canonical_schema_version") != CANONICAL_SCHEMA_VERSION:
        raise DataReadinessError(f"canonical schema version mismatch: {manifest_path}")
    if expected_type is not None and manifest.get("artifact_type") != expected_type.strip().lower():
        raise DataReadinessError(f"canonical artifact type mismatch: expected {expected_type}")
    if not allow_research and not bool(manifest.get("production_ready")):
        raise DataReadinessError(f"canonical artifact is not production-ready: {path}")
    expected_hash = str(manifest.get("artifact_sha256", ""))
    if not expected_hash or file_sha256(path) != expected_hash:
        raise DataReadinessError(f"canonical artifact integrity check failed: {path}")
    frame = pd.read_parquet(path)
    if len(frame) != int(manifest.get("rows", -1)):
        raise DataReadinessError(f"canonical artifact row count does not match manifest: {path}")
    if list(frame.columns) != list(manifest.get("columns", [])):
        raise DataReadinessError(f"canonical artifact columns do not match manifest: {path}")
    return frame, manifest


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iso(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(pd.Timestamp(value).isoformat())
