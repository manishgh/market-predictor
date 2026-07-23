from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from market_predictor.locking import file_lock

MODEL_MANIFEST_SCHEMA = "model_registry_manifest.v2"
MODEL_STATUS_CANDIDATE = "candidate"
MODEL_STATUS_PROMOTED = "promoted"
MODEL_STATUS_DEPRECATED = "deprecated"


def feature_schema_hash(features: list[str]) -> str:
    return _json_hash({"features": list(features)})


def dataset_fingerprint(data: pd.DataFrame, *, target_col: str, features: list[str]) -> dict[str, Any]:
    date_column = next(
        (column for column in ("date", "decision_time_utc", "session_date_et") if column in data.columns),
        None,
    )
    dates = pd.to_datetime(data[date_column], errors="coerce", utc=True) if date_column else pd.Series(dtype="datetime64[ns]")
    target = pd.to_numeric(data[target_col], errors="coerce") if target_col in data.columns else pd.Series(dtype="float")
    target_values = set(target.dropna().unique())
    is_binary = bool(target_values) and target_values.issubset({0, 1})
    return {
        "rows": int(len(data)),
        "columns": int(len(data.columns)),
        "tickers": int(data["ticker"].nunique()) if "ticker" in data.columns else 0,
        "first_date": _date_value(dates.min()) if not dates.empty else None,
        "last_date": _date_value(dates.max()) if not dates.empty else None,
        "target_col": target_col,
        "target_non_null_rows": int(target.notna().sum()),
        "target_mean": float(target.mean()) if target.notna().any() else None,
        "positive_rate": float(target.mean()) if is_binary else None,
        "feature_count": int(len(features)),
        "feature_schema_hash": feature_schema_hash(features),
    }


def write_model_manifest(
    *,
    model_path: Path,
    model_type: str,
    schema_version: str,
    target_col: str,
    features: list[str],
    training_data: pd.DataFrame,
    metrics: dict[str, Any],
    validation_split: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish one immutable candidate manifest; promotion is an attestation, not a field."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact does not exist: {model_path}")
    manifest = {
        "schema": MODEL_MANIFEST_SCHEMA,
        "status": MODEL_STATUS_CANDIDATE,
        "model_type": model_type,
        "schema_version": schema_version,
        "target_col": target_col,
        "artifact_path": str(model_path),
        "artifact_sha256": file_sha256(model_path),
        "created_at_utc": datetime.now(UTC).isoformat(),
        "validation_split": validation_split,
        "dataset": dataset_fingerprint(training_data, target_col=target_col, features=features),
        "metrics": _json_safe(metrics),
    }
    if extra:
        manifest["extra"] = _json_safe(extra)
    path = manifest_path_for(model_path)
    with file_lock(path):
        if path.exists():
            raise FileExistsError(f"Model candidate manifest is immutable: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return manifest


def load_model_manifest(model_path: Path) -> dict[str, Any]:
    path = manifest_path_for(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing model manifest: {path}")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Model manifest must contain a JSON object: {path}")
    return {str(key): value for key, value in loaded.items()}


def verify_model_artifact(
    model_path: Path,
    *,
    allowed_statuses: set[str] | None = None,
) -> dict[str, Any]:
    """Verify immutable candidate identity and derive promoted status from attestation."""

    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact does not exist: {model_path}")
    manifest = load_model_manifest(model_path)
    if manifest.get("schema") != MODEL_MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported model manifest schema: {model_path}")
    declared_status = str(manifest.get("status", "unknown"))
    if declared_status != MODEL_STATUS_CANDIDATE:
        raise ValueError("Model manifest must remain candidate; promoted status requires an immutable attestation")
    expected_hash = str(manifest.get("artifact_sha256", ""))
    if not expected_hash or file_sha256(model_path) != expected_hash:
        raise ValueError(f"Model artifact integrity check failed: {model_path}")

    from market_predictor.promotion_attestation import (  # local import avoids a registry/attestation cycle
        promotion_attestation_path_for,
        verify_promotion_attestation,
    )

    attestation_path = promotion_attestation_path_for(model_path)
    effective_status = MODEL_STATUS_CANDIDATE
    result = dict(manifest)
    if attestation_path.exists():
        attestation = verify_promotion_attestation(model_path)
        effective_status = MODEL_STATUS_PROMOTED
        result["promotion_attestation"] = {
            "path": str(attestation_path),
            "attestation_id": attestation["attestation_id"],
            "promoted_at_utc": attestation["promoted_at_utc"],
        }
    accepted = allowed_statuses or {MODEL_STATUS_CANDIDATE, MODEL_STATUS_PROMOTED}
    if effective_status not in accepted:
        raise ValueError(f"Model status {effective_status} is not allowed; expected one of {sorted(accepted)}")
    result["status"] = effective_status
    return result


def manifest_path_for(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".manifest.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _date_value(value: Any) -> str | None:
    if pd.isna(value):
        return None
    return str(pd.Timestamp(value).isoformat())


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, float) and (pd.isna(value) or value in {float("inf"), float("-inf")}):
        return None
    return value
