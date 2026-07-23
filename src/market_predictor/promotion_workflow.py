from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from market_predictor.hypothesis_registry import load_hypothesis
from market_predictor.promotion_attestation import (
    build_promotion_attestation,
    promotion_attestation_path_for,
    write_promotion_attestation,
)
from market_predictor.shadow_ledger import (
    consume_shadow_fingerprint,
    load_shadow_bundle,
    shadow_gate_failures,
)
from market_predictor.v3.errors import DataReadinessError


@dataclass(frozen=True)
class PromotionTrustContext:
    hypothesis_registry_root: Path
    hypothesis_id: str
    shadow_bundle_path: Path
    shadow_ledger_path: Path
    build_identity: str
    approver_identity: str
    minimum_shadow_sessions: int = 60
    minimum_paired_improvement_ci_low: float = 0.0

    def __post_init__(self) -> None:
        if self.minimum_shadow_sessions < 2:
            raise ValueError("promotion requires at least two independent shadow sessions")
        if not self.hypothesis_id.strip() or not self.build_identity.strip() or not self.approver_identity.strip():
            raise ValueError("hypothesis_id, build_identity, and approver_identity are required")


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
    shadow = load_shadow_bundle(context.shadow_bundle_path)
    if shadow.get("hypothesis_id") != context.hypothesis_id:
        raise DataReadinessError("shadow bundle does not belong to the requested hypothesis")
    attestation = build_promotion_attestation(
        model_path=model_path,
        evidence_manifest_path=evidence_manifest_path,
        metrics=metrics,
        hypothesis=hypothesis,
        shadow_bundle=shadow,
        gate_config=gate_config,
        build_identity=context.build_identity,
        approver_identity=context.approver_identity,
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
        )
        return TrustedPromotionOutcome(tuple(failures), shadow, ledger_entry, None, None)
    ledger_entry = consume_shadow_fingerprint(
        context.shadow_ledger_path,
        bundle=shadow,
        hypothesis=hypothesis,
        result="passed",
        attestation_id=str(attestation["attestation_id"]),
    )
    path = write_promotion_attestation(model_path, attestation)
    if path != promotion_attestation_path_for(model_path):
        raise DataReadinessError("promotion attestation was published to an unexpected path")
    return TrustedPromotionOutcome((), shadow, ledger_entry, attestation, path)
