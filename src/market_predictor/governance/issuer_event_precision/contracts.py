"""Frozen contracts and policy for issuer-event precision governance."""
from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from market_predictor.catalysts.issuer_events.classification import (
    ALLOWED_SOURCE_FAMILIES_BY_FAMILY,
    EVENT_FAMILIES,
)
from market_predictor.catalysts.issuer_events.family_evidence import (
    IssuerFamilyEvidence,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.issuer_event_precision.artifact_integrity import (
    _bounded_float,
    _positive_int,
    _threshold,
)
from market_predictor.resources import (
    assert_memory_budget,
)

POLICY_SCHEMA: Final = "market_predictor.issuer_event_precision_audit.v2"
SAMPLE_AUTHORITY_SCHEMA: Final = "edge_rebuild.issuer_event_precision_sample_authority.v2"
SAMPLE_MANIFEST_SCHEMA: Final = "edge_rebuild.issuer_event_precision_sample_manifest.v2"
FINAL_AUTHORITY_SCHEMA: Final = "edge_rebuild.issuer_event_precision_audit_authority.v2"
FINAL_MANIFEST_SCHEMA: Final = "edge_rebuild.issuer_event_precision_audit_manifest.v2"
SAMPLE_ARTIFACT_TYPE: Final = "issuer_event_precision_sample"
REVIEWS_ARTIFACT_TYPE: Final = "issuer_event_precision_reviews"
METRICS_ARTIFACT_TYPE: Final = "issuer_event_precision_family_metrics"
RULE_VARIANT_METRICS_ARTIFACT_TYPE: Final = "issuer_event_precision_rule_variant_metrics"
SOURCE_AUTHORIZATION_SHA256: Final = hashlib.sha256(
    json.dumps(
        {family: list(ALLOWED_SOURCE_FAMILIES_BY_FAMILY[family]) for family in EVENT_FAMILIES},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
_YES_NO_UNCERTAIN: Final = frozenset({"yes", "no", "uncertain"})
_YES_NO: Final = frozenset({"yes", "no"})
_DECISION_FIELDS: Final = (
    "family_correct",
    "issuer_target_correct",
    "event_announced_or_completed",
)
_CORRECTION_FIELDS: Final = (
    "correct_family",
    "action_subject_text",
    "false_positive_reason",
)
_FORBIDDEN_EVIDENCE_TERMS: Final = (
    "return",
    "price",
    "outcome",
    "probability",
    "target_return",
)

SAMPLE_COLUMNS: Final = (
    "sample_id",
    "sample_role",
    "inference_cluster_id",
    "paired_inferential_sample_id",
    "family_event_id",
    "source_event_id",
    "relation_id",
    "source_security_id",
    "source_ticker",
    "security_id",
    "ticker",
    "source_family",
    "source",
    "proposed_event_family",
    "classification_rule_id",
    "classification_basis",
    "matched_text",
    "title",
    "summary",
    "text",
    "published_at_utc",
    "event_available_at_utc",
    "relation_available_at_utc",
    "feature_available_at_utc",
    "availability_policy",
    "relation_channel",
    "issuer_company",
    "identity_effective_from_utc",
    "identity_effective_to_utc",
    "identity_available_at_utc",
    "identity_status",
    "calendar_quarter",
    "rule_variant",
    "normalized_title_sha256",
    "multi_target_title",
    "multi_label_event",
    "stratum_id",
    "cluster_selection_sha256",
    "row_selection_sha256",
    "schema_version",
)
REVIEW_TEMPLATE_COLUMNS: Final = (
    "sample_id",
    "family_event_id",
    "event_family",
    "reviewer_slot",
    "reviewer_id",
    "family_correct",
    "issuer_target_correct",
    "event_announced_or_completed",
    "correct_family",
    "action_subject_text",
    "false_positive_reason",
    "comments",
)
ADJUDICATION_TEMPLATE_COLUMNS: Final = (
    "sample_id",
    "family_event_id",
    "event_family",
    "adjudicator_id",
    "family_correct",
    "issuer_target_correct",
    "event_announced_or_completed",
    "correct_family",
    "action_subject_text",
    "false_positive_reason",
    "comments",
)
REVIEW_COLUMNS: Final = (
    "sample_id",
    "sample_role",
    "inference_cluster_id",
    "family_event_id",
    "event_family",
    "reviewer_one_id",
    "reviewer_two_id",
    "adjudicator_id",
    "reviewer_one_family_correct",
    "reviewer_one_issuer_target_correct",
    "reviewer_one_event_announced_or_completed",
    "reviewer_two_family_correct",
    "reviewer_two_issuer_target_correct",
    "reviewer_two_event_announced_or_completed",
    "adjudication_required",
    "resolution_state",
    "family_correct",
    "issuer_target_correct",
    "event_announced_or_completed",
    "joint_correct",
    "wrong_issuer",
    "correct_family",
    "action_subject_text",
    "false_positive_reason",
    "comments",
    "schema_version",
)
METRIC_COLUMNS: Final = (
    "event_family",
    "population_eligible_events",
    "population_clusters",
    "population_issuers",
    "inferential_cluster_count",
    "diagnostic_count",
    "resolved_inferential_count",
    "minimum_population_clusters",
    "minimum_population_issuers",
    "family_successes",
    "family_lcb",
    "minimum_family_lcb",
    "issuer_successes",
    "issuer_lcb",
    "minimum_issuer_lcb",
    "event_successes",
    "event_lcb",
    "minimum_event_lcb",
    "joint_successes",
    "joint_lcb",
    "minimum_joint_lcb",
    "wrong_issuer_count",
    "diagnostic_wrong_issuer_count",
    "unresolved_count",
    "identity_unresolved_count",
    "family_reviewer_agreement",
    "family_reviewer_kappa",
    "family_kappa_estimable",
    "issuer_reviewer_agreement",
    "issuer_reviewer_kappa",
    "issuer_kappa_estimable",
    "event_reviewer_agreement",
    "event_reviewer_kappa",
    "event_kappa_estimable",
    "minimum_reviewer_agreement",
    "minimum_reviewer_kappa",
    "failed_rule_variant_count",
    "status",
    "blocker_reasons",
    "schema_version",
)
RULE_VARIANT_METRIC_COLUMNS: Final = (
    "event_family",
    "rule_variant",
    "population_clusters",
    "inferential_cluster_count",
    "joint_successes",
    "joint_lcb",
    "minimum_joint_lcb",
    "minimum_sample_clusters",
    "gate_applicable",
    "status",
    "blocker_reasons",
    "schema_version",
)


@dataclass(frozen=True, slots=True)
class FamilyPrecisionPolicy:
    sample_clusters: int
    minimum_population_clusters: int
    minimum_population_issuers: int
    minimum_family_lcb: float
    minimum_issuer_lcb: float
    minimum_event_lcb: float
    minimum_joint_lcb: float


@dataclass(frozen=True, slots=True)
class IssuerEventPrecisionPolicy:
    confidence_level: float
    reviewers_per_item: int
    unresolved_policy: str
    require_distinct_reviewers: bool
    no_wrong_issuer: bool
    maximum_process_memory_gib: float
    memory_guard_headroom_gib: float
    paired_wrong_issuer_diagnostics_per_cluster: int
    minimum_reviewer_agreement: float
    minimum_reviewer_kappa: float
    minimum_kappa_decisions: int
    rule_variant_gate_minimum_population_clusters: int
    minimum_rule_variant_sample_clusters: int
    minimum_rule_variant_joint_lcb: float
    families: Mapping[str, FamilyPrecisionPolicy]


@dataclass(frozen=True, slots=True)
class IssuerEventPrecisionSample:
    directory: Path
    sample: pd.DataFrame
    manifest: Mapping[str, object]
    authority: Mapping[str, object]
    source_authority: IssuerFamilyEvidence | None


@dataclass(frozen=True, slots=True)
class IssuerEventPrecisionAudit:
    directory: Path
    reviews: pd.DataFrame
    family_metrics: pd.DataFrame
    rule_variant_metrics: pd.DataFrame
    manifest: Mapping[str, object]
    authority: Mapping[str, object]
    source_authority: IssuerFamilyEvidence | None


def load_issuer_event_precision_policy(path: Path) -> IssuerEventPrecisionPolicy:
    """Load and strictly validate the frozen event precision policy."""

    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != POLICY_SCHEMA:
        raise DataReadinessError(f"unsupported issuer-event precision policy: {path}")
    if int(raw.get("reviewers_per_item", 0)) != 2:
        raise DataReadinessError("issuer-event precision audit requires two reviewers")
    if raw.get("unresolved_policy") != "failure":
        raise DataReadinessError("unresolved precision reviews must count as failures")
    if raw.get("require_distinct_reviewers") is not True:
        raise DataReadinessError("issuer-event precision reviewers must be distinct")
    if raw.get("no_wrong_issuer") is not True:
        raise DataReadinessError("issuer-event precision policy must reject wrong issuers")
    confidence = _bounded_float(raw, "confidence_level", lower=0.5, upper=1.0)
    maximum_memory = _bounded_float(raw, "maximum_process_memory_gib", lower=0.0, upper=5.0, inclusive=True)
    if maximum_memory <= 0:
        raise DataReadinessError("issuer-event precision memory budget must be positive and at most 5 GiB")
    memory_headroom = _bounded_float(raw, "memory_guard_headroom_gib", lower=0.0, upper=maximum_memory)
    reviewer_agreement = _threshold(raw, "minimum_reviewer_agreement")
    reviewer_kappa = _threshold(raw, "minimum_reviewer_kappa")
    raw_families = raw.get("family")
    if not isinstance(raw_families, dict) or set(raw_families) != set(EVENT_FAMILIES):
        raise DataReadinessError("issuer-event precision policy must define every frozen family exactly")
    families: dict[str, FamilyPrecisionPolicy] = {}
    expected_keys = {
        "sample_clusters",
        "minimum_population_clusters",
        "minimum_population_issuers",
        "minimum_family_lcb",
        "minimum_issuer_lcb",
        "minimum_event_lcb",
        "minimum_joint_lcb",
    }
    for family in EVENT_FAMILIES:
        value = raw_families.get(family)
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise DataReadinessError(f"issuer-event precision family policy is malformed: {family}")
        families[family] = FamilyPrecisionPolicy(
            sample_clusters=_positive_int(value, "sample_clusters"),
            minimum_population_clusters=_positive_int(value, "minimum_population_clusters"),
            minimum_population_issuers=_positive_int(value, "minimum_population_issuers"),
            minimum_family_lcb=_threshold(value, "minimum_family_lcb"),
            minimum_issuer_lcb=_threshold(value, "minimum_issuer_lcb"),
            minimum_event_lcb=_threshold(value, "minimum_event_lcb"),
            minimum_joint_lcb=_threshold(value, "minimum_joint_lcb"),
        )
    return IssuerEventPrecisionPolicy(
        confidence_level=confidence,
        reviewers_per_item=2,
        unresolved_policy="failure",
        require_distinct_reviewers=True,
        no_wrong_issuer=True,
        maximum_process_memory_gib=maximum_memory,
        memory_guard_headroom_gib=memory_headroom,
        paired_wrong_issuer_diagnostics_per_cluster=_positive_int(raw, "paired_wrong_issuer_diagnostics_per_cluster"),
        minimum_reviewer_agreement=reviewer_agreement,
        minimum_reviewer_kappa=reviewer_kappa,
        minimum_kappa_decisions=_positive_int(raw, "minimum_kappa_decisions"),
        rule_variant_gate_minimum_population_clusters=_positive_int(raw, "rule_variant_gate_minimum_population_clusters"),
        minimum_rule_variant_sample_clusters=_positive_int(raw, "minimum_rule_variant_sample_clusters"),
        minimum_rule_variant_joint_lcb=_threshold(raw, "minimum_rule_variant_joint_lcb"),
        families=families,
    )

def _guard_memory(policy: IssuerEventPrecisionPolicy, stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=policy.maximum_process_memory_gib,
        headroom_gib=policy.memory_guard_headroom_gib,
        stage=stage,
    )
