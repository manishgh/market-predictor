from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from market_predictor.causal_shadow import load_causal_shadow_bundle
from market_predictor.hypothesis_registry import load_hypothesis
from market_predictor.outcome_repository import OutcomeRepository
from market_predictor.promotion_attestation import (
    build_promotion_attestation,
    file_sha256,
    promotion_attestation_path_for,
    write_promotion_attestation,
)
from market_predictor.shadow_ledger import (
    consume_shadow_fingerprint,
    shadow_gate_failures,
)
from market_predictor.v3.errors import DataReadinessError


@dataclass(frozen=True)
class PromotionTrustContext:
    hypothesis_registry_root: Path
    hypothesis_id: str
    shadow_bundle_path: Path
    outcome_repository_root: Path
    baseline_artifact_path: Path
    build_identity: str
    approver_identity: str
    signing_private_key_path: Path
    attestation_trust_store_path: Path
    signer_id: str
    minimum_shadow_sessions: int = 60
    minimum_paired_improvement_ci_low: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_shadow_sessions < 2:
            raise ValueError("promotion requires at least two independent shadow sessions")
        if not all(
            value.strip()
            for value in (
                self.hypothesis_id,
                self.build_identity,
                self.approver_identity,
                self.signer_id,
            )
        ):
            raise ValueError(
                "hypothesis_id, build_identity, approver_identity, and signer_id are required"
            )
        if self.build_identity.strip() == self.approver_identity.strip():
            raise ValueError("build and approver identities must be distinct")
        registry_root = self.hypothesis_registry_root.resolve()
        shadow_root = registry_root / "shadow"
        if not self.shadow_bundle_path.resolve().is_relative_to(shadow_root):
            raise ValueError("shadow bundle must be stored inside the hypothesis registry")

    @property
    def shadow_ledger_path(self) -> Path:
        return self.hypothesis_registry_root.resolve() / "shadow-ledger.jsonl"


@dataclass(frozen=True)
class TrustedPromotionOutcome:
    failures: tuple[str, ...]
    shadow_evidence: dict[str, Any]
    ledger_entry: dict[str, Any] | None
    attestation: dict[str, Any] | None
    attestation_path: Path | None


def evaluate_shadow_and_attest(
    *,
    model_path: Path,
    evidence_manifest_path: Path,
    metrics: dict[str, Any],
    gate_config: dict[str, Any],
    context: PromotionTrustContext,
) -> TrustedPromotionOutcome:
    """Open shadow evidence once, retire failures, and attest passing candidates."""

    hypothesis = load_hypothesis(context.hypothesis_registry_root, context.hypothesis_id)
    shadow = load_causal_shadow_bundle(
        context.shadow_bundle_path,
        repository=OutcomeRepository(context.outcome_repository_root),
        hypothesis=hypothesis,
    )
    if shadow.get("hypothesis_id") != context.hypothesis_id:
        raise DataReadinessError("shadow bundle does not belong to the requested hypothesis")
    _validate_shadow_context(
        model_path,
        context.baseline_artifact_path,
        hypothesis,
        shadow,
        metrics,
    )
    transaction_id = _transaction_id(
        model_path=model_path,
        hypothesis=hypothesis,
        shadow=shadow,
        gate_config=gate_config,
    )
    failures = shadow_gate_failures(
        shadow,
        minimum_independent_sessions=context.minimum_shadow_sessions,
        minimum_paired_improvement_ci_low=context.minimum_paired_improvement_ci_low,
    )
    if failures:
        ledger_entry = consume_shadow_fingerprint(
            context.shadow_ledger_path,
            bundle=shadow,
            hypothesis=hypothesis,
            result="failed",
            attestation_id=None,
            transaction_id=transaction_id,
        )
        return TrustedPromotionOutcome(tuple(failures), shadow, ledger_entry, None, None)
    ledger_entry = consume_shadow_fingerprint(
        context.shadow_ledger_path,
        bundle=shadow,
        hypothesis=hypothesis,
        result="passed",
        attestation_id=None,
        transaction_id=transaction_id,
    )
    attestation = build_promotion_attestation(
        model_path=model_path,
        evidence_manifest_path=evidence_manifest_path,
        metrics=metrics,
        hypothesis=hypothesis,
        shadow_bundle=shadow,
        ledger_entry=ledger_entry,
        gate_config=gate_config,
        build_identity=context.build_identity,
        approver_identity=context.approver_identity,
        signing_private_key_path=context.signing_private_key_path,
        signer_id=context.signer_id,
    )
    path = write_promotion_attestation(
        model_path,
        attestation,
        trust_store_path=context.attestation_trust_store_path,
    )
    if path != promotion_attestation_path_for(model_path):
        raise DataReadinessError("promotion attestation was published to an unexpected path")
    return TrustedPromotionOutcome((), shadow, ledger_entry, attestation, path)


def _validate_shadow_context(
    model_path: Path,
    baseline_artifact_path: Path,
    hypothesis: dict[str, Any],
    shadow: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    if shadow.get("candidate_artifact_sha256") != file_sha256(model_path):
        raise DataReadinessError("shadow bundle does not belong to the candidate artifact")
    if hypothesis.get("candidate_artifact_sha256") != file_sha256(model_path):
        raise DataReadinessError(
            "hypothesis does not freeze the candidate artifact"
        )
    if (
        not baseline_artifact_path.is_file()
        or file_sha256(baseline_artifact_path)
        != hypothesis.get("baseline_artifact_sha256")
    ):
        raise DataReadinessError(
            "shadow baseline artifact is not frozen by the hypothesis"
        )
    if shadow.get("hypothesis_record_sha256") != hypothesis.get("record_sha256"):
        raise DataReadinessError("shadow bundle does not belong to the hypothesis record")
    if shadow.get("baseline_artifact_sha256") != hypothesis.get(
        "baseline_artifact_sha256"
    ):
        raise DataReadinessError("shadow bundle baseline artifact is invalid")
    for field in ("prediction_policy_sha256", "execution_policy_sha256"):
        if shadow.get(field) != metrics.get(field):
            raise DataReadinessError(f"shadow bundle {field} is invalid")
    try:
        declared_at = datetime.fromisoformat(
            str(hypothesis.get("declared_at_utc") or "")
        )
        generated_at = datetime.fromisoformat(
            str(shadow.get("generated_at_utc") or "")
        )
        first_session = date.fromisoformat(
            str(shadow.get("first_session_date_et") or "")
        )
        last_session = date.fromisoformat(
            str(shadow.get("last_session_date_et") or "")
        )
    except ValueError as exc:
        raise DataReadinessError("shadow timing identity is invalid") from exc
    if declared_at.tzinfo is None or generated_at.tzinfo is None:
        raise DataReadinessError("shadow timing identity must be timezone-aware")
    if (
        first_session <= declared_at.date()
        or last_session < first_session
        or generated_at.date() < last_session
        or generated_at.astimezone(UTC) > datetime.now(UTC) + timedelta(minutes=5)
    ):
        raise DataReadinessError("shadow timing does not prove prospective evaluation")


def _transaction_id(
    *,
    model_path: Path,
    hypothesis: dict[str, Any],
    shadow: dict[str, Any],
    gate_config: dict[str, Any],
) -> str:
    payload = {
        "candidate_artifact_sha256": file_sha256(model_path),
        "hypothesis_record_sha256": hypothesis.get("record_sha256"),
        "shadow_fingerprint": shadow.get("shadow_fingerprint"),
        "gate_config": gate_config,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
