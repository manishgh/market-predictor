"""File-backed verification for governed promoted model bundles."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Literal, overload

from market_predictor.canonical.store import file_sha256
from market_predictor.core import path_integrity
from market_predictor.core.errors import (
    ArtifactIntegrityError,
    DataReadinessError,
    PromotionGateError,
)
from market_predictor.edge_rebuild.swing_training import (
    MODEL_SCHEMA as SWING_CANDIDATE_MODEL_SCHEMA,
)
from market_predictor.governance.promotion.bundle_contracts import (
    PromotedIntradayBundle,
    PromotedSwingBundle,
    validate_promoted_bundle,
)
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.promotion_attestation import (
    promotion_attestation_path_for,
    verify_promotion_attestation,
)


@overload
def validate_file_backed_promoted_bundle(
    payload: Mapping[str, object],
    *,
    bundle_root: Path,
    strategy_contract: StrategyContract,
    attestation_trust_store_path: Path | None = None,
    promotion_gate_policy_sha256: str | None = None,
    maximum_model_bytes: int | None = None,
    maximum_evidence_bytes: int = 1024 * 1024,
    expected_mode: Literal["swing"],
) -> PromotedSwingBundle: ...


@overload
def validate_file_backed_promoted_bundle(
    payload: Mapping[str, object],
    *,
    bundle_root: Path,
    strategy_contract: StrategyContract,
    attestation_trust_store_path: Path | None = None,
    promotion_gate_policy_sha256: str | None = None,
    maximum_model_bytes: int | None = None,
    maximum_evidence_bytes: int = 1024 * 1024,
    expected_mode: Literal["intraday"],
) -> PromotedIntradayBundle: ...


@overload
def validate_file_backed_promoted_bundle(
    payload: Mapping[str, object],
    *,
    bundle_root: Path,
    strategy_contract: StrategyContract,
    attestation_trust_store_path: Path | None = None,
    promotion_gate_policy_sha256: str | None = None,
    maximum_model_bytes: int | None = None,
    maximum_evidence_bytes: int = 1024 * 1024,
    expected_mode: None = None,
) -> PromotedSwingBundle | PromotedIntradayBundle: ...


def validate_file_backed_promoted_bundle(
    payload: Mapping[str, object],
    *,
    bundle_root: Path,
    strategy_contract: StrategyContract,
    attestation_trust_store_path: Path | None = None,
    promotion_gate_policy_sha256: str | None = None,
    maximum_model_bytes: int | None = None,
    maximum_evidence_bytes: int = 1024 * 1024,
    expected_mode: Literal["swing", "intraday"] | None = None,
) -> PromotedSwingBundle | PromotedIntradayBundle:
    """Validate bundle metadata, artifacts, and signed promotion authorization."""

    bundle = validate_promoted_bundle(
        payload,
        strategy_contract=strategy_contract,
        expected_mode=expected_mode,
    )
    root = resolve_verified_bundle_root(bundle_root)
    artifacts = (
        (
            "model",
            bundle.model_artifact_path,
            bundle.model_artifact_sha256,
            maximum_model_bytes,
        ),
        (
            "promotion evidence",
            bundle.promotion_evidence_path,
            bundle.promotion_evidence_sha256,
            maximum_evidence_bytes,
        ),
    )
    for label, relative_path, expected_sha256, maximum_bytes in artifacts:
        artifact_path = resolve_verified_bundle_artifact(
            root,
            relative_path,
            label=label,
        )
        before = artifact_path.stat()
        if maximum_bytes is not None and (maximum_bytes < 1 or before.st_size > maximum_bytes):
            raise DataReadinessError(f"{label} artifact byte limit exceeded")
        observed_sha256 = file_sha256(artifact_path)
        after = artifact_path.stat()
        identity_before = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        identity_after = (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if identity_before != identity_after:
            raise ArtifactIntegrityError(f"{label} changed while its serving hash was being verified")
        if observed_sha256 != expected_sha256:
            raise ArtifactIntegrityError(f"{label} artifact SHA256 does not verify")
    if bundle.mode != "swing":
        return bundle
    if attestation_trust_store_path is None:
        raise PromotionGateError("a configured promotion attestation trust store is required")
    if promotion_gate_policy_sha256 is None:
        raise PromotionGateError("a configured promotion gate-policy hash is required")
    if bundle.promotion_gate_policy_sha256 != promotion_gate_policy_sha256:
        raise PromotionGateError("serving bundle promotion gate policy differs from configuration")
    model_path = resolve_verified_bundle_artifact(
        root,
        bundle.model_artifact_path,
        label="model",
    )
    evidence_path = resolve_verified_bundle_artifact(
        root,
        bundle.promotion_evidence_path,
        label="promotion evidence",
    )
    if evidence_path != promotion_attestation_path_for(model_path):
        raise PromotionGateError("promotion evidence must be the immutable candidate attestation")
    try:
        attestation = verify_promotion_attestation(
            model_path,
            trust_store_path=attestation_trust_store_path,
        )
    except (DataReadinessError, OSError, TypeError, ValueError) as exc:
        raise PromotionGateError("promotion attestation did not verify") from exc
    candidate = attestation.get("candidate")
    approver = attestation.get("approver_principal")
    ledger = attestation.get("ledger_receipt")
    if not isinstance(candidate, Mapping) or not isinstance(approver, Mapping):
        raise PromotionGateError("promotion attestation identity is incomplete")
    if (
        candidate.get("artifact_sha256") != bundle.model_artifact_sha256
        or candidate.get("model_run_id") != bundle.model_id
        or candidate.get("model_schema_version") != SWING_CANDIDATE_MODEL_SCHEMA
        or attestation.get("attestation_id") != bundle.promotion_attestation_id
        or attestation.get("gate_config_sha256") != bundle.promotion_gate_policy_sha256
        or approver.get("principal_id") != bundle.approved_by_principal_id
        or attestation.get("promoted_at_utc") != bundle.promoted_at_utc.isoformat()
    ):
        raise PromotionGateError("promotion attestation does not bind the served candidate identity")
    if not isinstance(ledger, Mapping) or ledger.get("result") != "passed":
        raise PromotionGateError("promotion attestation does not prove passed gates")
    return bundle


def resolve_verified_bundle_root(bundle_root: Path) -> Path:
    """Resolve an immutable bundle tree without crossing reparse points."""

    try:
        return path_integrity.verify_tree_containment(
            bundle_root,
            label="immutable bundle root",
        )
    except DataReadinessError as exc:
        raise ArtifactIntegrityError(str(exc)) from exc


def resolve_verified_bundle_artifact(
    root: Path,
    value: str,
    *,
    label: str,
) -> Path:
    """Resolve one canonical regular-file path below a verified bundle root."""

    if "\\" in value:
        raise ArtifactIntegrityError(f"{label} path must use canonical bundle-relative POSIX syntax")
    relative = PurePosixPath(value)
    if (
        relative.as_posix() != value
        or relative.is_absolute()
        or not relative.parts
        or any(part in {"", ".", ".."} for part in relative.parts)
        or ":" in relative.parts[0]
    ):
        raise ArtifactIntegrityError(f"{label} path escapes the immutable bundle root")
    try:
        return path_integrity.resolve_existing_file_inside(
            root,
            value,
            label=f"{label} artifact",
        )
    except DataReadinessError as exc:
        raise ArtifactIntegrityError(str(exc)) from exc
