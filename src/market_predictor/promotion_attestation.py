from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pandas as pd

from market_predictor.locking import file_lock
from market_predictor.v3.errors import DataReadinessError

PROMOTION_ATTESTATION_SCHEMA = "market_predictor.promotion_attestation.v1"
COMMON_IDENTITY_FIELDS = (
    "validation_split",
    "holdout_assignment_cutoff_utc",
    "holdout_ticker_summary_sha256",
    "feature_set_sha256",
    "reconciliation_sha256",
    "dataset_label_config_sha256",
    "calibration_method",
    "folds_causally_ordered",
    "prediction_policy_sha256",
    "execution_policy_sha256",
    "dataset_sha256",
)
SHA_IDENTITY_FIELDS = {
    "holdout_ticker_summary_sha256",
    "feature_set_sha256",
    "reconciliation_sha256",
    "dataset_label_config_sha256",
    "universe_identity_sha256",
    "prediction_policy_sha256",
    "execution_policy_sha256",
    "dataset_sha256",
}


def build_promotion_attestation(
    *,
    model_path: Path,
    evidence_manifest_path: Path,
    metrics: dict[str, Any],
    hypothesis: dict[str, Any],
    shadow_bundle: dict[str, Any],
    gate_config: dict[str, Any],
    build_identity: str,
    approver_identity: str,
    promoted_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a content-addressed authorization record after all gates pass."""

    manifest_path = candidate_manifest_path_for(model_path)
    manifest = _load_json_object(manifest_path, "candidate manifest")
    if manifest.get("schema") != "model_registry_manifest.v2" or manifest.get("status") != "candidate":
        raise DataReadinessError("promotion attestation requires an immutable candidate manifest")
    if not model_path.is_file():
        raise DataReadinessError(f"candidate model artifact is missing: {model_path}")
    artifact_sha = file_sha256(model_path)
    if manifest.get("artifact_sha256") != artifact_sha:
        raise DataReadinessError("candidate artifact does not match its manifest")
    manifest_metrics = manifest.get("metrics")
    if not isinstance(manifest_metrics, dict):
        raise DataReadinessError("candidate manifest is missing training metrics")
    identity = _validated_identity_chain(
        metrics,
        {str(key): value for key, value in manifest_metrics.items()},
        model_type=str(manifest.get("model_type") or ""),
    )
    model_run_id = str(metrics.get("model_run_id") or "")
    manifest_run_id = str(cast(dict[str, Any], manifest.get("extra") or {}).get("model_run_id") or "")
    if not model_run_id or model_run_id != manifest_run_id:
        raise DataReadinessError("candidate manifest and promotion metrics model_run_id do not match")

    evidence_manifest = _load_json_object(evidence_manifest_path, "training evidence manifest")
    if evidence_manifest.get("model_artifact_sha256") != artifact_sha:
        raise DataReadinessError("training evidence does not belong to the candidate artifact")
    if evidence_manifest.get("model_run_id") != model_run_id:
        raise DataReadinessError("training evidence model_run_id does not match the candidate")
    evidence_manifest_sha = file_sha256(evidence_manifest_path)

    hypothesis_sha = _required_sha(hypothesis, "record_sha256", "hypothesis")
    hypothesis_id = str(hypothesis.get("hypothesis_id") or "")
    hypothesis_family = str(hypothesis.get("hypothesis_family") or "")
    baseline_id = str(hypothesis.get("baseline_id") or "")
    if hypothesis.get("model_type") != manifest.get("model_type"):
        raise DataReadinessError("hypothesis model_type does not match the candidate")
    if hypothesis.get("prediction_policy_sha256") != identity["prediction_policy_sha256"]:
        raise DataReadinessError("hypothesis prediction policy does not match the candidate")
    shadow_fingerprint = _required_sha(shadow_bundle, "shadow_fingerprint", "shadow bundle")
    if shadow_bundle.get("hypothesis_record_sha256") != hypothesis_sha:
        raise DataReadinessError("shadow evidence does not match the hypothesis declaration")
    if shadow_bundle.get("hypothesis_id") != hypothesis_id or shadow_bundle.get("hypothesis_family") != hypothesis_family:
        raise DataReadinessError("shadow evidence hypothesis identity mismatch")
    if shadow_bundle.get("baseline_id") != baseline_id:
        raise DataReadinessError("shadow evidence baseline does not match the hypothesis")
    if shadow_bundle.get("candidate_artifact_sha256") != artifact_sha:
        raise DataReadinessError("shadow evidence does not belong to the candidate artifact")
    if shadow_bundle.get("prediction_policy_sha256") != identity["prediction_policy_sha256"]:
        raise DataReadinessError("shadow evidence policy does not match the candidate")
    interval = shadow_bundle.get("paired_improvement_interval")
    if not isinstance(interval, dict):
        raise DataReadinessError("shadow evidence is missing its paired confidence interval")
    if not build_identity.strip() or not approver_identity.strip():
        raise ValueError("build_identity and approver_identity are required")
    timestamp = _utc(promoted_at or datetime.now(UTC))
    content: dict[str, Any] = {
        "schema": PROMOTION_ATTESTATION_SCHEMA,
        "candidate": {
            "artifact_sha256": artifact_sha,
            "manifest_sha256": file_sha256(manifest_path),
            "model_run_id": model_run_id,
            "model_type": manifest.get("model_type"),
            "model_schema_version": manifest.get("schema_version"),
        },
        "evidence_manifest_sha256": evidence_manifest_sha,
        "identity_chain": identity,
        "hypothesis": {
            "hypothesis_id": hypothesis_id,
            "hypothesis_family": hypothesis_family,
            "hypothesis_record_sha256": hypothesis_sha,
            "baseline_id": baseline_id,
            "baseline_artifact_sha256": hypothesis.get("baseline_artifact_sha256"),
        },
        "shadow": {
            "shadow_fingerprint": shadow_fingerprint,
            "independent_sessions": shadow_bundle.get("independent_sessions"),
            "first_session_date_et": shadow_bundle.get("first_session_date_et"),
            "last_session_date_et": shadow_bundle.get("last_session_date_et"),
            "paired_improvement_interval": interval,
        },
        "gate_config_sha256": _json_sha256(gate_config),
        "build_identity": build_identity.strip(),
        "approver_identity": approver_identity.strip(),
        "promoted_at_utc": timestamp.isoformat(),
    }
    return {**content, "attestation_id": _json_sha256(content)}


def write_promotion_attestation(model_path: Path, attestation: dict[str, Any]) -> Path:
    """Publish one immutable attestation beside a candidate model."""

    _verify_attestation_content(attestation)
    path = promotion_attestation_path_for(model_path)
    with file_lock(path):
        if path.exists():
            existing = _load_json_object(path, "promotion attestation")
            if existing != attestation:
                raise DataReadinessError("promotion attestation is immutable")
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(json.dumps(attestation, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
    return path


def verify_promotion_attestation(
    model_path: Path,
    *,
    evidence_manifest_path: Path | None = None,
) -> dict[str, Any]:
    """Verify attestation content and every locally available bound artifact."""

    path = promotion_attestation_path_for(model_path)
    attestation = _load_json_object(path, "promotion attestation")
    _verify_attestation_content(attestation)
    candidate = attestation.get("candidate")
    if not isinstance(candidate, dict):
        raise DataReadinessError("promotion attestation candidate binding is invalid")
    manifest_path = candidate_manifest_path_for(model_path)
    if file_sha256(model_path) != candidate.get("artifact_sha256"):
        raise DataReadinessError("promotion attestation candidate artifact changed")
    if file_sha256(manifest_path) != candidate.get("manifest_sha256"):
        raise DataReadinessError("promotion attestation candidate manifest changed")
    manifest = _load_json_object(manifest_path, "candidate manifest")
    if manifest.get("status") != "candidate" or manifest.get("artifact_sha256") != candidate.get("artifact_sha256"):
        raise DataReadinessError("promotion attestation candidate manifest is not immutable")
    manifest_metrics = manifest.get("metrics")
    identity = attestation.get("identity_chain")
    if not isinstance(manifest_metrics, dict) or not isinstance(identity, dict):
        raise DataReadinessError("promotion attestation identity chain is incomplete")
    for field, value in identity.items():
        if manifest_metrics.get(field) != value:
            raise DataReadinessError(f"promotion attestation identity changed: {field}")
    if evidence_manifest_path is not None and file_sha256(evidence_manifest_path) != attestation.get("evidence_manifest_sha256"):
        raise DataReadinessError("promotion attestation evidence manifest changed")
    return attestation


def promotion_attestation_path_for(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".promotion.attestation.json")


def candidate_manifest_path_for(model_path: Path) -> Path:
    return model_path.with_suffix(model_path.suffix + ".manifest.json")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except FileNotFoundError as exc:
        raise DataReadinessError(f"bound promotion artifact is missing: {path}") from exc
    return digest.hexdigest()


def _validated_identity_chain(
    metrics: dict[str, Any],
    manifest_metrics: dict[str, Any],
    *,
    model_type: str,
) -> dict[str, Any]:
    fields = list(COMMON_IDENTITY_FIELDS)
    if model_type == "swing_classifier_v1":
        fields.append("universe_identity_sha256")
    identity: dict[str, Any] = {}
    for field in fields:
        value = metrics.get(field)
        if value is None or value == "":
            raise DataReadinessError(f"promotion identity is missing: {field}")
        if manifest_metrics.get(field) != value:
            raise DataReadinessError(f"candidate manifest identity does not match promotion evidence: {field}")
        if field in SHA_IDENTITY_FIELDS:
            _require_sha256(str(value), field)
        identity[field] = value
    if identity.get("folds_causally_ordered") is not True:
        raise DataReadinessError("promotion identity does not prove causal fold ordering")
    cutoff = pd.Timestamp(identity["holdout_assignment_cutoff_utc"])
    if cutoff.tzinfo is None:
        raise DataReadinessError("holdout assignment cutoff must be timezone-aware")
    return identity


def _verify_attestation_content(attestation: dict[str, Any]) -> None:
    payload = dict(attestation)
    attestation_id = str(payload.pop("attestation_id", ""))
    if payload.get("schema") != PROMOTION_ATTESTATION_SCHEMA or _json_sha256(payload) != attestation_id:
        raise DataReadinessError("promotion attestation content hash is invalid")
    _require_sha256(attestation_id, "attestation_id")


def _required_sha(payload: dict[str, Any], field: str, name: str) -> str:
    value = str(payload.get(field) or "")
    _require_sha256(value, f"{name}.{field}")
    return value


def _require_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise DataReadinessError(f"{name} must be a SHA-256 digest")


def _load_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"{name} is unavailable or invalid: {path}") from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"{name} must contain an object: {path}")
    return {str(key): value for key, value in loaded.items()}


def _json_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("promotion timestamps must be timezone-aware")
    return value.astimezone(UTC)
