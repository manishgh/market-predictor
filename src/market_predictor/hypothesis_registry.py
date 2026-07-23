from __future__ import annotations

import hashlib
import json
import os
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from market_predictor.locking import file_lock
from market_predictor.v3.errors import DataReadinessError

HYPOTHESIS_SCHEMA = "market_predictor.promotion_hypothesis.v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,127}$")


def declare_hypothesis(
    registry_root: Path,
    *,
    hypothesis_id: str,
    hypothesis_family: str,
    model_type: str,
    baseline_id: str,
    baseline_artifact_sha256: str,
    prediction_policy_sha256: str,
    objective: str,
    declared_at: datetime | None = None,
) -> dict[str, Any]:
    """Create an immutable hypothesis declaration before shadow evidence is opened."""

    for name, value in (("hypothesis_id", hypothesis_id), ("hypothesis_family", hypothesis_family), ("baseline_id", baseline_id)):
        if not _SAFE_ID.fullmatch(value):
            raise ValueError(f"{name} contains unsafe characters")
    if not model_type.strip() or not objective.strip():
        raise ValueError("model_type and objective are required")
    _require_sha256(baseline_artifact_sha256, "baseline_artifact_sha256")
    _require_sha256(prediction_policy_sha256, "prediction_policy_sha256")
    timestamp = _utc(declared_at or datetime.now(UTC))
    content: dict[str, Any] = {
        "schema": HYPOTHESIS_SCHEMA,
        "hypothesis_id": hypothesis_id,
        "hypothesis_family": hypothesis_family,
        "model_type": model_type.strip(),
        "baseline_id": baseline_id,
        "baseline_artifact_sha256": baseline_artifact_sha256,
        "prediction_policy_sha256": prediction_policy_sha256,
        "objective": objective.strip(),
        "declared_at_utc": timestamp.isoformat(),
    }
    payload = {**content, "record_sha256": _json_sha256(content)}
    path = hypothesis_path(registry_root, hypothesis_id)
    with file_lock(registry_root / ".hypothesis-registry"):
        if path.exists():
            existing = load_hypothesis(registry_root, hypothesis_id)
            if existing != payload:
                raise DataReadinessError(f"hypothesis declaration is immutable: {hypothesis_id}")
            return existing
        _write_json_atomic(path, payload)
    return payload


def load_hypothesis(registry_root: Path, hypothesis_id: str) -> dict[str, Any]:
    if not _SAFE_ID.fullmatch(hypothesis_id):
        raise ValueError("hypothesis_id contains unsafe characters")
    path = hypothesis_path(registry_root, hypothesis_id)
    if not path.is_file():
        raise DataReadinessError(f"hypothesis is not predeclared: {hypothesis_id}")
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DataReadinessError(f"hypothesis declaration is invalid: {hypothesis_id}") from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"hypothesis declaration is invalid: {hypothesis_id}")
    payload = {str(key): value for key, value in loaded.items()}
    record_sha = str(payload.pop("record_sha256", ""))
    if payload.get("schema") != HYPOTHESIS_SCHEMA or payload.get("hypothesis_id") != hypothesis_id:
        raise DataReadinessError(f"hypothesis declaration identity mismatch: {hypothesis_id}")
    if _json_sha256(payload) != record_sha:
        raise DataReadinessError(f"hypothesis declaration integrity check failed: {hypothesis_id}")
    return {**payload, "record_sha256": record_sha}


def hypothesis_path(registry_root: Path, hypothesis_id: str) -> Path:
    return registry_root / "hypotheses" / f"{hypothesis_id}.json"


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise ValueError(f"{name} must be a SHA-256 digest")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("hypothesis timestamps must be timezone-aware")
    return value.astimezone(UTC)
