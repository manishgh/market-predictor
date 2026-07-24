from __future__ import annotations

import hashlib
import json
import os
from base64 import b64decode, b64encode
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pandas as pd
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from market_predictor.locking import file_lock
from market_predictor.v3.errors import DataReadinessError

PROMOTION_ATTESTATION_SCHEMA = "market_predictor.promotion_attestation.v1"
ATTESTATION_TRUST_STORE_SCHEMA = "market_predictor.attestation_trust_store.v1"
ATTESTATION_TRUST_STORE_ENV = "MARKET_PREDICTOR_ATTESTATION_TRUST_STORE"
SIGNATURE_ALGORITHM = "ed25519"
COMMON_IDENTITY_FIELDS = (
    "validation_split",
    "holdout_assignment_cutoff_utc",
    "holdout_ticker_summary_sha256",
    "feature_set_sha256",
    "reconciliation_sha256",
    "event_assignment_sha256",
    "event_aggregate_sha256",
    "label_material_sha256",
    "label_source_reconciliation_sha256",
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
    "event_assignment_sha256",
    "event_aggregate_sha256",
    "label_material_sha256",
    "label_source_reconciliation_sha256",
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
    ledger_entry: dict[str, Any],
    gate_config: dict[str, Any],
    build_identity: str,
    approver_identity: str,
    signing_private_key_path: Path,
    signer_id: str,
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
    baseline_artifact_sha = _required_sha(
        hypothesis,
        "baseline_artifact_sha256",
        "hypothesis",
    )
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
    if shadow_bundle.get("baseline_artifact_sha256") != baseline_artifact_sha:
        raise DataReadinessError("shadow evidence baseline artifact does not match the hypothesis")
    if shadow_bundle.get("candidate_artifact_sha256") != artifact_sha:
        raise DataReadinessError("shadow evidence does not belong to the candidate artifact")
    if shadow_bundle.get("prediction_policy_sha256") != identity["prediction_policy_sha256"]:
        raise DataReadinessError("shadow evidence policy does not match the candidate")
    if shadow_bundle.get("execution_policy_sha256") != identity["execution_policy_sha256"]:
        raise DataReadinessError("shadow evidence execution policy does not match the candidate")
    _required_sha(shadow_bundle, "source_evidence_sha256", "shadow bundle")
    interval = shadow_bundle.get("paired_improvement_interval")
    if not isinstance(interval, dict):
        raise DataReadinessError("shadow evidence is missing its paired confidence interval")
    if not build_identity.strip() or not approver_identity.strip() or not signer_id.strip():
        raise ValueError("build_identity and approver_identity are required")
    if build_identity.strip() == approver_identity.strip():
        raise DataReadinessError("promotion build and approver identities must be distinct")
    ledger_receipt = _validated_ledger_receipt(
        ledger_entry,
        shadow_fingerprint=shadow_fingerprint,
        hypothesis_id=hypothesis_id,
        hypothesis_family=hypothesis_family,
    )
    timestamp = _utc(promoted_at or datetime.now(UTC))
    shadow_generated_at = _utc(datetime.fromisoformat(str(shadow_bundle.get("generated_at_utc") or "")))
    if timestamp < shadow_generated_at:
        raise DataReadinessError("promotion cannot predate shadow evidence generation")
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
            "baseline_artifact_sha256": baseline_artifact_sha,
        },
        "shadow": {
            "shadow_fingerprint": shadow_fingerprint,
            "independent_sessions": shadow_bundle.get("independent_sessions"),
            "first_session_date_et": shadow_bundle.get("first_session_date_et"),
            "last_session_date_et": shadow_bundle.get("last_session_date_et"),
            "generated_at_utc": shadow_generated_at.isoformat(),
            "paired_improvement_interval": interval,
            "source_evidence_sha256": shadow_bundle.get("source_evidence_sha256"),
        },
        "ledger_receipt": ledger_receipt,
        "gate_config_sha256": _json_sha256(gate_config),
        "build_identity": build_identity.strip(),
        "approver_identity": approver_identity.strip(),
        "promoted_at_utc": timestamp.isoformat(),
    }
    unsigned = {**content, "attestation_id": _json_sha256(content)}
    return _sign_attestation(
        unsigned,
        signing_private_key_path=signing_private_key_path,
        signer_id=signer_id.strip(),
    )


def write_promotion_attestation(
    model_path: Path,
    attestation: dict[str, Any],
    *,
    trust_store_path: Path | None = None,
) -> Path:
    """Publish one immutable attestation beside a candidate model."""

    _verify_attestation_content(
        attestation,
        trust_store_path=_resolve_trust_store_path(trust_store_path),
    )
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
    trust_store_path: Path | None = None,
) -> dict[str, Any]:
    """Verify attestation content and every locally available bound artifact."""

    path = promotion_attestation_path_for(model_path)
    attestation = _load_json_object(path, "promotion attestation")
    _verify_attestation_content(
        attestation,
        trust_store_path=_resolve_trust_store_path(trust_store_path),
    )
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
    if model_type == "canonical_swing":
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


def _verify_attestation_content(
    attestation: dict[str, Any],
    *,
    trust_store_path: Path,
) -> None:
    required_fields = {
        "schema",
        "candidate",
        "evidence_manifest_sha256",
        "identity_chain",
        "hypothesis",
        "shadow",
        "ledger_receipt",
        "gate_config_sha256",
        "build_identity",
        "approver_identity",
        "promoted_at_utc",
        "attestation_id",
        "signature",
    }
    if set(attestation) != required_fields:
        raise DataReadinessError("promotion attestation schema fields are invalid")
    signature = attestation.get("signature")
    if not isinstance(signature, dict) or set(signature) != {
        "algorithm",
        "signer_id",
        "signature_base64",
    }:
        raise DataReadinessError("promotion attestation signature fields are invalid")
    unsigned = dict(attestation)
    unsigned.pop("signature")
    payload = dict(unsigned)
    attestation_id = str(payload.pop("attestation_id", ""))
    if payload.get("schema") != PROMOTION_ATTESTATION_SCHEMA or _json_sha256(payload) != attestation_id:
        raise DataReadinessError("promotion attestation content hash is invalid")
    _require_sha256(attestation_id, "attestation_id")
    _validate_attestation_bindings(payload)
    _verify_signature(unsigned, signature, trust_store_path)


def _validate_attestation_bindings(payload: dict[str, Any]) -> None:
    candidate = _exact_object(
        payload.get("candidate"),
        {
            "artifact_sha256",
            "manifest_sha256",
            "model_run_id",
            "model_type",
            "model_schema_version",
        },
        "candidate",
    )
    _require_sha256(str(candidate["artifact_sha256"]), "candidate.artifact_sha256")
    _require_sha256(str(candidate["manifest_sha256"]), "candidate.manifest_sha256")
    if not all(str(candidate.get(field) or "").strip() for field in ("model_run_id", "model_type", "model_schema_version")):
        raise DataReadinessError("promotion attestation candidate identity is incomplete")
    _require_sha256(
        str(payload.get("evidence_manifest_sha256") or ""),
        "evidence_manifest_sha256",
    )
    _require_sha256(
        str(payload.get("gate_config_sha256") or ""),
        "gate_config_sha256",
    )
    identity = payload.get("identity_chain")
    if not isinstance(identity, dict):
        raise DataReadinessError("promotion attestation identity chain is invalid")
    expected_identity = set(COMMON_IDENTITY_FIELDS)
    if candidate["model_type"] == "canonical_swing":
        expected_identity.add("universe_identity_sha256")
    if set(identity) != expected_identity:
        raise DataReadinessError("promotion attestation identity chain fields are invalid")
    _validated_identity_chain(
        {str(key): value for key, value in identity.items()},
        {str(key): value for key, value in identity.items()},
        model_type=str(candidate["model_type"]),
    )
    hypothesis = _exact_object(
        payload.get("hypothesis"),
        {
            "hypothesis_id",
            "hypothesis_family",
            "hypothesis_record_sha256",
            "baseline_id",
            "baseline_artifact_sha256",
        },
        "hypothesis",
    )
    for field in ("hypothesis_id", "hypothesis_family", "baseline_id"):
        if not str(hypothesis.get(field) or "").strip():
            raise DataReadinessError(f"promotion attestation hypothesis is missing {field}")
    _require_sha256(
        str(hypothesis["hypothesis_record_sha256"]),
        "hypothesis.hypothesis_record_sha256",
    )
    _require_sha256(
        str(hypothesis["baseline_artifact_sha256"]),
        "hypothesis.baseline_artifact_sha256",
    )
    shadow = _exact_object(
        payload.get("shadow"),
        {
            "shadow_fingerprint",
            "independent_sessions",
            "first_session_date_et",
            "last_session_date_et",
            "generated_at_utc",
            "paired_improvement_interval",
            "source_evidence_sha256",
        },
        "shadow",
    )
    _require_sha256(str(shadow["shadow_fingerprint"]), "shadow.shadow_fingerprint")
    _require_sha256(
        str(shadow["source_evidence_sha256"]),
        "shadow.source_evidence_sha256",
    )
    sessions = shadow.get("independent_sessions")
    if not isinstance(sessions, int) or isinstance(sessions, bool) or sessions < 2:
        raise DataReadinessError("promotion attestation shadow sessions are invalid")
    interval = shadow.get("paired_improvement_interval")
    if not isinstance(interval, dict) or not {"point", "low", "high"}.issubset(interval):
        raise DataReadinessError("promotion attestation shadow interval is invalid")
    shadow_generated_at = _utc(datetime.fromisoformat(str(shadow.get("generated_at_utc") or "")))
    _validated_ledger_receipt(
        _exact_object(
            payload.get("ledger_receipt"),
            {
                "schema",
                "sequence",
                "previous_entry_sha256",
                "shadow_fingerprint",
                "hypothesis_id",
                "hypothesis_family",
                "result",
                "attestation_id",
                "transaction_id",
                "consumed_at_utc",
                "entry_sha256",
            },
            "ledger receipt",
        ),
        shadow_fingerprint=str(shadow["shadow_fingerprint"]),
        hypothesis_id=str(hypothesis["hypothesis_id"]),
        hypothesis_family=str(hypothesis["hypothesis_family"]),
    )
    build_identity = str(payload.get("build_identity") or "").strip()
    approver_identity = str(payload.get("approver_identity") or "").strip()
    if not build_identity or not approver_identity or build_identity == approver_identity:
        raise DataReadinessError("promotion attestation separation of duties is invalid")
    promoted_at = _utc(datetime.fromisoformat(str(payload.get("promoted_at_utc") or "")))
    if promoted_at < shadow_generated_at:
        raise DataReadinessError("promotion attestation predates shadow evidence")


def _sign_attestation(
    unsigned: dict[str, Any],
    *,
    signing_private_key_path: Path,
    signer_id: str,
) -> dict[str, Any]:
    try:
        key = serialization.load_pem_private_key(
            signing_private_key_path.read_bytes(),
            password=None,
        )
    except (FileNotFoundError, ValueError, TypeError) as exc:
        raise DataReadinessError("promotion signing key is unavailable or invalid") from exc
    if not isinstance(key, Ed25519PrivateKey):
        raise DataReadinessError("promotion signing key must be Ed25519")
    signature = key.sign(_canonical_bytes(unsigned))
    return {
        **unsigned,
        "signature": {
            "algorithm": SIGNATURE_ALGORITHM,
            "signer_id": signer_id,
            "signature_base64": b64encode(signature).decode("ascii"),
        },
    }


def _verify_signature(
    unsigned: dict[str, Any],
    signature: dict[str, Any],
    trust_store_path: Path,
) -> None:
    if signature.get("algorithm") != SIGNATURE_ALGORITHM:
        raise DataReadinessError("promotion attestation signature algorithm is unsupported")
    signer_id = str(signature.get("signer_id") or "")
    trust_store = _load_json_object(trust_store_path, "attestation trust store")
    if set(trust_store) != {"schema", "issuers"} or trust_store.get("schema") != ATTESTATION_TRUST_STORE_SCHEMA:
        raise DataReadinessError("attestation trust store schema is invalid")
    issuers = trust_store.get("issuers")
    issuer = issuers.get(signer_id) if isinstance(issuers, dict) else None
    if not isinstance(issuer, dict) or set(issuer) != {
        "algorithm",
        "public_key_base64",
    }:
        raise DataReadinessError("promotion attestation signer is not trusted")
    if issuer.get("algorithm") != SIGNATURE_ALGORITHM:
        raise DataReadinessError("trusted attestation signer algorithm is invalid")
    try:
        public_key = Ed25519PublicKey.from_public_bytes(b64decode(str(issuer["public_key_base64"]), validate=True))
        signature_bytes = b64decode(
            str(signature.get("signature_base64") or ""),
            validate=True,
        )
        public_key.verify(signature_bytes, _canonical_bytes(unsigned))
    except (ValueError, InvalidSignature) as exc:
        raise DataReadinessError("promotion attestation signature is invalid") from exc


def _validated_ledger_receipt(
    ledger_entry: dict[str, Any],
    *,
    shadow_fingerprint: str,
    hypothesis_id: str,
    hypothesis_family: str,
) -> dict[str, Any]:
    receipt = {str(key): value for key, value in ledger_entry.items()}
    entry_sha = str(receipt.pop("entry_sha256", ""))
    if (
        receipt.get("schema") != "market_predictor.shadow_ledger_entry.v2"
        or receipt.get("result") != "passed"
        or receipt.get("shadow_fingerprint") != shadow_fingerprint
        or receipt.get("hypothesis_id") != hypothesis_id
        or receipt.get("hypothesis_family") != hypothesis_family
        or _json_sha256(receipt) != entry_sha
    ):
        raise DataReadinessError("promotion shadow ledger receipt is invalid")
    _require_sha256(entry_sha, "ledger_receipt.entry_sha256")
    _require_sha256(
        str(receipt.get("transaction_id") or ""),
        "ledger_receipt.transaction_id",
    )
    sequence = receipt.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        raise DataReadinessError("promotion shadow ledger receipt sequence is invalid")
    return {**receipt, "entry_sha256": entry_sha}


def _exact_object(
    value: object,
    fields: set[str],
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        raise DataReadinessError(f"promotion attestation {name} fields are invalid")
    return {str(key): item for key, item in value.items()}


def _resolve_trust_store_path(path: Path | None) -> Path:
    selected = path or (Path(os.environ[ATTESTATION_TRUST_STORE_ENV]) if os.environ.get(ATTESTATION_TRUST_STORE_ENV) else None)
    if selected is None:
        raise DataReadinessError(f"attestation trust store is required via {ATTESTATION_TRUST_STORE_ENV}")
    return selected.resolve()


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
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _canonical_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("promotion timestamps must be timezone-aware")
    return value.astimezone(UTC)
