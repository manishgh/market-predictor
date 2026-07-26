from __future__ import annotations

import hashlib
import json
import re
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast

from market_predictor.v3.errors import DataReadinessError

LabelDimension = Literal["offering", "end_market", "driver"]
KnowledgeScope = Literal["historical_research_proxy", "current_inference_only"]
AssignmentDisposition = Literal["assigned", "insufficient_evidence"]

_DIMENSION_ORDER: dict[LabelDimension, int] = {
    "offering": 0,
    "end_market": 1,
    "driver": 2,
}
_LABEL_ID = re.compile(r"^[a-z0-9]+(?:[._][a-z0-9]+)+$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RULE_KEYS = {
    "id",
    "dimension",
    "membership_sector_exact",
    "membership_industry_exact",
    "profile_phrase_exact",
    "compatible_membership_label_ids",
}
_ROOT_KEYS = {
    "schema_version",
    "assignment_policy_version",
    "max_leaf_labels",
    "profile_validity_days",
    "allowed_membership_availability_policies",
    "industry_labels",
    "labels",
}


@dataclass(frozen=True, slots=True)
class LabelRule:
    label_id: str
    dimension: LabelDimension
    membership_sector_exact: frozenset[str]
    membership_industry_exact: frozenset[str]
    profile_phrase_exact: frozenset[str]
    compatible_membership_label_ids: frozenset[str]


@dataclass(frozen=True, slots=True)
class SecurityLabelPolicy:
    schema_version: str
    assignment_policy_version: str
    max_leaf_labels: int
    profile_validity_days: int
    allowed_membership_availability_policies: frozenset[str]
    labels: tuple[LabelRule, ...]
    taxonomy_sha256: str

    @property
    def label_ids(self) -> frozenset[str]:
        return frozenset(rule.label_id for rule in self.labels)


@dataclass(frozen=True, slots=True)
class MembershipEvidence:
    security_id: str
    sector: str
    industry: str
    effective_from_utc: datetime
    effective_to_utc: datetime | None
    available_at_utc: datetime
    availability_policy: str
    source_uri: str
    source_published_at_utc: datetime
    source_content_sha256: str


@dataclass(frozen=True, slots=True)
class CurrentProfileEvidence:
    security_id: str
    profile_terms: tuple[str, ...]
    membership_label_ids: tuple[str, ...]
    observed_at_utc: datetime
    source_uri: str
    source_published_at_utc: datetime
    source_content_sha256: str


@dataclass(frozen=True, slots=True)
class SecurityBusinessLabelAssignment:
    security_id: str
    label_id: str
    dimension: LabelDimension
    label_rank: int
    effective_from_utc: datetime
    effective_to_utc: datetime | None
    available_at_utc: datetime
    knowledge_scope: KnowledgeScope
    availability_policy: str
    source_uri: str
    source_content_sha256: str
    evidence_sha256: str
    taxonomy_version: str
    taxonomy_sha256: str
    assignment_policy_version: str
    assignment_sha256: str


@dataclass(frozen=True, slots=True)
class SecurityBusinessLabelSet:
    security_id: str
    effective_from_utc: datetime
    effective_to_utc: datetime | None
    available_at_utc: datetime
    knowledge_scope: KnowledgeScope
    availability_policy: str
    disposition: AssignmentDisposition
    assignments: tuple[SecurityBusinessLabelAssignment, ...]
    source_uri: str
    source_content_sha256: str
    evidence_sha256: str
    taxonomy_version: str
    taxonomy_sha256: str
    assignment_policy_version: str
    assignment_set_sha256: str

    def is_available_at(self, timestamp: datetime) -> bool:
        moment = _aware_utc(timestamp, field="timestamp")
        return (
            self.available_at_utc <= moment
            and self.effective_from_utc <= moment
            and (self.effective_to_utc is None or moment < self.effective_to_utc)
        )


def load_security_label_policy(path: Path) -> SecurityLabelPolicy:
    """Load and strictly validate the closed business-label vocabulary."""

    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    unknown_root = sorted(set(raw).difference(_ROOT_KEYS))
    if unknown_root:
        raise DataReadinessError(f"security label policy has unknown keys: {unknown_root}")

    schema_version = _required_string(raw, "schema_version")
    assignment_policy_version = _required_string(raw, "assignment_policy_version")
    max_leaf_labels = raw.get("max_leaf_labels")
    if not isinstance(max_leaf_labels, int) or isinstance(max_leaf_labels, bool) or max_leaf_labels != 3:
        raise DataReadinessError("security label policy max_leaf_labels must equal 3")
    profile_validity_days = raw.get("profile_validity_days")
    if (
        not isinstance(profile_validity_days, int)
        or isinstance(profile_validity_days, bool)
        or profile_validity_days < 1
        or profile_validity_days > 365
    ):
        raise DataReadinessError(
            "security label policy profile_validity_days must be "
            "between 1 and 365"
        )
    availability_policies = _string_collection(raw, "allowed_membership_availability_policies")
    if not availability_policies:
        raise DataReadinessError("security label policy requires membership availability policies")

    raw_labels = raw.get("labels")
    if not isinstance(raw_labels, list) or not raw_labels:
        raise DataReadinessError("security label policy requires at least one label")
    rules = (
        *(_parse_rule(item) for item in raw_labels),
        *_parse_industry_label_rules(raw.get("industry_labels")),
    )
    label_ids = [rule.label_id for rule in rules]
    if len(label_ids) != len(set(label_ids)):
        raise DataReadinessError("security label policy contains duplicate label ids")
    _reject_ambiguous_exact_rules(rules)
    _validate_profile_compatibility(rules)

    policy_payload = {
        "schema_version": schema_version,
        "assignment_policy_version": assignment_policy_version,
        "max_leaf_labels": max_leaf_labels,
        "profile_validity_days": profile_validity_days,
        "allowed_membership_availability_policies": sorted(availability_policies),
        "labels": [_rule_payload(rule) for rule in sorted(rules, key=lambda item: item.label_id)],
    }
    return SecurityLabelPolicy(
        schema_version=schema_version,
        assignment_policy_version=assignment_policy_version,
        max_leaf_labels=max_leaf_labels,
        profile_validity_days=profile_validity_days,
        allowed_membership_availability_policies=frozenset(availability_policies),
        labels=rules,
        taxonomy_sha256=_json_sha256(policy_payload),
    )


def assign_security_business_labels(
    policy: SecurityLabelPolicy,
    memberships: Sequence[MembershipEvidence],
    profiles: Sequence[CurrentProfileEvidence] = (),
    *,
    as_of_utc: datetime,
) -> tuple[SecurityBusinessLabelSet, ...]:
    """Assign exact evidence-backed labels without ticker or company-name inference."""

    as_of = _aware_utc(as_of_utc, field="as_of_utc")
    normalized_memberships = tuple(_normalize_membership(item, policy, as_of) for item in memberships)
    normalized_profiles = tuple(_normalize_profile(item, policy, as_of) for item in profiles)
    _reject_membership_overlaps(normalized_memberships)
    _reject_profile_overlaps(normalized_profiles)

    results = [_membership_label_set(policy, evidence) for evidence in normalized_memberships]
    results.extend(_profile_label_set(policy, evidence) for evidence in normalized_profiles)
    return tuple(
        sorted(
            results,
            key=lambda item: (
                item.security_id,
                item.knowledge_scope,
                item.effective_from_utc,
                item.assignment_set_sha256,
            ),
        )
    )


def validate_business_label_ids(
    policy: SecurityLabelPolicy,
    label_ids: Iterable[str],
) -> tuple[str, ...]:
    """Reject labels not present in the frozen controlled vocabulary."""

    values = tuple(label_ids)
    unknown = sorted(set(values).difference(policy.label_ids))
    if unknown:
        raise DataReadinessError(f"unknown security business labels: {unknown}")
    if len(values) != len(set(values)):
        raise DataReadinessError("security business label ids must be unique")
    if len(values) > policy.max_leaf_labels:
        raise DataReadinessError(f"security business label count {len(values)} exceeds maximum {policy.max_leaf_labels}")
    return values


def profile_terms_from_text(
    policy: SecurityLabelPolicy,
    text: str,
) -> tuple[str, ...]:
    """Extract only frozen exact evidence phrases from a profile description."""

    normalized = _normalize_exact(text)
    padded = f" {normalized} "
    phrases = {phrase for rule in policy.labels for phrase in rule.profile_phrase_exact if f" {phrase} " in padded}
    return tuple(sorted(phrases))


def _membership_label_set(
    policy: SecurityLabelPolicy,
    evidence: MembershipEvidence,
) -> SecurityBusinessLabelSet:
    sector = _normalize_exact(evidence.sector)
    industry = _normalize_exact(evidence.industry)
    matched = [rule for rule in policy.labels if sector in rule.membership_sector_exact or industry in rule.membership_industry_exact]
    evidence_payload: dict[str, object] = {
        "availability_policy": evidence.availability_policy,
        "industry": industry,
        "sector": sector,
    }
    return _build_label_set(
        policy=policy,
        security_id=evidence.security_id,
        effective_from_utc=evidence.effective_from_utc,
        effective_to_utc=evidence.effective_to_utc,
        available_at_utc=evidence.available_at_utc,
        knowledge_scope="historical_research_proxy",
        availability_policy=evidence.availability_policy,
        source_uri=evidence.source_uri,
        source_content_sha256=evidence.source_content_sha256,
        evidence_payload=evidence_payload,
        matched=matched,
    )


def _profile_label_set(
    policy: SecurityLabelPolicy,
    evidence: CurrentProfileEvidence,
) -> SecurityBusinessLabelSet:
    normalized_terms = tuple(sorted({_normalize_exact(term) for term in evidence.profile_terms if _normalize_exact(term)}))
    term_set = frozenset(normalized_terms)
    membership_label_ids = frozenset(evidence.membership_label_ids)
    matched = [
        rule
        for rule in policy.labels
        if rule.profile_phrase_exact.intersection(term_set)
        and (rule.dimension == "end_market" or bool(rule.compatible_membership_label_ids.intersection(membership_label_ids)))
    ]
    evidence_payload: dict[str, object] = {
        "membership_label_ids": sorted(membership_label_ids),
        "profile_terms": normalized_terms,
    }
    return _build_label_set(
        policy=policy,
        security_id=evidence.security_id,
        effective_from_utc=evidence.observed_at_utc,
        effective_to_utc=(
            evidence.observed_at_utc
            + timedelta(days=policy.profile_validity_days)
        ),
        available_at_utc=evidence.observed_at_utc,
        knowledge_scope="current_inference_only",
        availability_policy="profile_observation",
        source_uri=evidence.source_uri,
        source_content_sha256=evidence.source_content_sha256,
        evidence_payload=evidence_payload,
        matched=matched,
    )


def _build_label_set(
    *,
    policy: SecurityLabelPolicy,
    security_id: str,
    effective_from_utc: datetime,
    effective_to_utc: datetime | None,
    available_at_utc: datetime,
    knowledge_scope: KnowledgeScope,
    availability_policy: str,
    source_uri: str,
    source_content_sha256: str,
    evidence_payload: Mapping[str, object],
    matched: Sequence[LabelRule],
) -> SecurityBusinessLabelSet:
    ordered = tuple(
        sorted(
            {rule.label_id: rule for rule in matched}.values(),
            key=lambda rule: (_DIMENSION_ORDER[rule.dimension], rule.label_id),
        )
    )
    validate_business_label_ids(policy, (rule.label_id for rule in ordered))
    evidence_sha256 = _json_sha256(evidence_payload)
    assignments: list[SecurityBusinessLabelAssignment] = []
    for rank, rule in enumerate(ordered, start=1):
        payload = {
            "assignment_policy_version": policy.assignment_policy_version,
            "availability_policy": availability_policy,
            "available_at_utc": _utc_text(available_at_utc),
            "dimension": rule.dimension,
            "effective_from_utc": _utc_text(effective_from_utc),
            "effective_to_utc": _utc_text(effective_to_utc),
            "evidence_sha256": evidence_sha256,
            "knowledge_scope": knowledge_scope,
            "label_id": rule.label_id,
            "label_rank": rank,
            "security_id": security_id,
            "source_content_sha256": source_content_sha256,
            "source_uri": source_uri,
            "taxonomy_sha256": policy.taxonomy_sha256,
            "taxonomy_version": policy.schema_version,
        }
        assignments.append(
            SecurityBusinessLabelAssignment(
                security_id=security_id,
                label_id=rule.label_id,
                dimension=rule.dimension,
                label_rank=rank,
                effective_from_utc=effective_from_utc,
                effective_to_utc=effective_to_utc,
                available_at_utc=available_at_utc,
                knowledge_scope=knowledge_scope,
                availability_policy=availability_policy,
                source_uri=source_uri,
                source_content_sha256=source_content_sha256,
                evidence_sha256=evidence_sha256,
                taxonomy_version=policy.schema_version,
                taxonomy_sha256=policy.taxonomy_sha256,
                assignment_policy_version=policy.assignment_policy_version,
                assignment_sha256=_json_sha256(payload),
            )
        )
    disposition: AssignmentDisposition = "assigned" if assignments else "insufficient_evidence"
    set_payload = {
        "assignment_hashes": [item.assignment_sha256 for item in assignments],
        "assignment_policy_version": policy.assignment_policy_version,
        "availability_policy": availability_policy,
        "available_at_utc": _utc_text(available_at_utc),
        "disposition": disposition,
        "effective_from_utc": _utc_text(effective_from_utc),
        "effective_to_utc": _utc_text(effective_to_utc),
        "evidence_sha256": evidence_sha256,
        "knowledge_scope": knowledge_scope,
        "security_id": security_id,
        "source_content_sha256": source_content_sha256,
        "source_uri": source_uri,
        "taxonomy_sha256": policy.taxonomy_sha256,
        "taxonomy_version": policy.schema_version,
    }
    return SecurityBusinessLabelSet(
        security_id=security_id,
        effective_from_utc=effective_from_utc,
        effective_to_utc=effective_to_utc,
        available_at_utc=available_at_utc,
        knowledge_scope=knowledge_scope,
        availability_policy=availability_policy,
        disposition=disposition,
        assignments=tuple(assignments),
        source_uri=source_uri,
        source_content_sha256=source_content_sha256,
        evidence_sha256=evidence_sha256,
        taxonomy_version=policy.schema_version,
        taxonomy_sha256=policy.taxonomy_sha256,
        assignment_policy_version=policy.assignment_policy_version,
        assignment_set_sha256=_json_sha256(set_payload),
    )


def _normalize_membership(
    evidence: MembershipEvidence,
    policy: SecurityLabelPolicy,
    as_of_utc: datetime,
) -> MembershipEvidence:
    security_id = _nonempty(evidence.security_id, field="security_id")
    sector = _nonempty(evidence.sector, field="sector")
    industry = _nonempty(evidence.industry, field="industry")
    start = _aware_utc(evidence.effective_from_utc, field="effective_from_utc")
    end = _optional_aware_utc(evidence.effective_to_utc, field="effective_to_utc")
    available = _aware_utc(evidence.available_at_utc, field="available_at_utc")
    published = _aware_utc(evidence.source_published_at_utc, field="source_published_at_utc")
    if end is not None and end <= start:
        raise DataReadinessError("effective_to_utc must be later than effective_from_utc")
    if start > as_of_utc or (end is not None and end > as_of_utc):
        raise DataReadinessError("membership effective interval contains a future timestamp")
    if available > as_of_utc or published > as_of_utc:
        raise DataReadinessError("membership evidence contains a future availability timestamp")
    if published > available:
        raise DataReadinessError("membership source cannot be published after it becomes available")
    availability_policy = _nonempty(evidence.availability_policy, field="availability_policy")
    if availability_policy not in policy.allowed_membership_availability_policies:
        raise DataReadinessError(f"unsupported membership availability policy: {availability_policy}")
    return MembershipEvidence(
        security_id=security_id,
        sector=sector,
        industry=industry,
        effective_from_utc=start,
        effective_to_utc=end,
        available_at_utc=available,
        availability_policy=availability_policy,
        source_uri=_nonempty(evidence.source_uri, field="source_uri"),
        source_published_at_utc=published,
        source_content_sha256=_validated_sha256(evidence.source_content_sha256),
    )


def _normalize_profile(
    evidence: CurrentProfileEvidence,
    policy: SecurityLabelPolicy,
    as_of_utc: datetime,
) -> CurrentProfileEvidence:
    observed = _aware_utc(evidence.observed_at_utc, field="observed_at_utc")
    published = _aware_utc(evidence.source_published_at_utc, field="source_published_at_utc")
    if observed > as_of_utc or published > as_of_utc:
        raise DataReadinessError("current profile contains a future timestamp")
    if published > observed:
        raise DataReadinessError("profile source cannot be published after its observation")
    normalized_terms = tuple(sorted({_normalize_exact(term) for term in evidence.profile_terms if _normalize_exact(term)}))
    membership_label_ids = validate_business_label_ids(
        policy,
        evidence.membership_label_ids,
    )
    return CurrentProfileEvidence(
        security_id=_nonempty(evidence.security_id, field="security_id"),
        profile_terms=normalized_terms,
        membership_label_ids=membership_label_ids,
        observed_at_utc=observed,
        source_uri=_nonempty(evidence.source_uri, field="source_uri"),
        source_published_at_utc=published,
        source_content_sha256=_validated_sha256(evidence.source_content_sha256),
    )


def _reject_membership_overlaps(evidence: Sequence[MembershipEvidence]) -> None:
    grouped: dict[str, list[MembershipEvidence]] = {}
    for item in evidence:
        grouped.setdefault(item.security_id, []).append(item)
    for security_id, intervals in grouped.items():
        previous_end: datetime | None = None
        previous_open = False
        for item in sorted(intervals, key=lambda value: value.effective_from_utc):
            if previous_open or (previous_end is not None and item.effective_from_utc < previous_end):
                raise DataReadinessError(f"security label membership intervals overlap for {security_id}")
            previous_open = item.effective_to_utc is None
            previous_end = item.effective_to_utc


def _reject_profile_overlaps(evidence: Sequence[CurrentProfileEvidence]) -> None:
    seen: set[str] = set()
    for item in evidence:
        if item.security_id in seen:
            raise DataReadinessError(f"current profile intervals overlap for {item.security_id}")
        seen.add(item.security_id)


def _parse_rule(value: object) -> LabelRule:
    if not isinstance(value, dict):
        raise DataReadinessError("security label rules must be TOML tables")
    raw = cast(dict[str, object], value)
    unknown = sorted(set(raw).difference(_RULE_KEYS))
    if unknown:
        raise DataReadinessError(f"security label rule has unknown keys: {unknown}")
    label_id = _required_string(raw, "id")
    if _LABEL_ID.fullmatch(label_id) is None:
        raise DataReadinessError(f"invalid security label id: {label_id}")
    raw_dimension = _required_string(raw, "dimension")
    if raw_dimension not in _DIMENSION_ORDER:
        raise DataReadinessError(f"unknown security label dimension: {raw_dimension}")
    dimension: LabelDimension = raw_dimension
    if not label_id.startswith(f"{dimension}."):
        raise DataReadinessError(f"security label {label_id} must start with {dimension}.")
    rule = LabelRule(
        label_id=label_id,
        dimension=dimension,
        membership_sector_exact=frozenset(_normalized_collection(raw, "membership_sector_exact")),
        membership_industry_exact=frozenset(_normalized_collection(raw, "membership_industry_exact")),
        profile_phrase_exact=frozenset(_normalized_collection(raw, "profile_phrase_exact")),
        compatible_membership_label_ids=frozenset(
            _optional_string_collection(
                raw,
                "compatible_membership_label_ids",
            )
        ),
    )
    if not (rule.membership_sector_exact or rule.membership_industry_exact or rule.profile_phrase_exact):
        raise DataReadinessError(f"security label {label_id} has no exact evidence rules")
    return rule


def _parse_industry_label_rules(value: object) -> tuple[LabelRule, ...]:
    if not isinstance(value, dict) or not value:
        raise DataReadinessError("security label policy requires a non-empty industry_labels table")
    rules: list[LabelRule] = []
    normalized_industries: set[str] = set()
    for raw_industry, raw_label_id in sorted(value.items()):
        if not isinstance(raw_industry, str) or not raw_industry.strip():
            raise DataReadinessError("industry label names must be non-empty strings")
        if not isinstance(raw_label_id, str) or _LABEL_ID.fullmatch(raw_label_id) is None:
            raise DataReadinessError(f"invalid controlled industry label id: {raw_label_id}")
        if not raw_label_id.startswith("offering.industry."):
            raise DataReadinessError("controlled industry label ids must start with offering.industry.")
        industry = _normalize_exact(raw_industry)
        if industry in normalized_industries:
            raise DataReadinessError(f"duplicate normalized controlled industry: {raw_industry}")
        normalized_industries.add(industry)
        rules.append(
            LabelRule(
                label_id=raw_label_id,
                dimension="offering",
                membership_sector_exact=frozenset(),
                membership_industry_exact=frozenset({industry}),
                profile_phrase_exact=frozenset(),
                compatible_membership_label_ids=frozenset(),
            )
        )
    return tuple(rules)


def _validate_profile_compatibility(
    rules: Sequence[LabelRule],
) -> None:
    label_ids = {rule.label_id for rule in rules}
    for rule in rules:
        unknown = sorted(rule.compatible_membership_label_ids.difference(label_ids))
        if unknown:
            raise DataReadinessError(f"security label {rule.label_id} has unknown compatible membership labels: {unknown}")
        if rule.profile_phrase_exact and rule.dimension in {"offering", "driver"} and not rule.compatible_membership_label_ids:
            raise DataReadinessError(f"security label {rule.label_id} requires explicit membership compatibility for profile evidence")
        invalid = sorted(value for value in rule.compatible_membership_label_ids if not value.startswith("offering."))
        if invalid:
            raise DataReadinessError(f"security label {rule.label_id} has non-offering compatibility labels: {invalid}")


def _reject_ambiguous_exact_rules(rules: Sequence[LabelRule]) -> None:
    for attribute in (
        "membership_sector_exact",
        "membership_industry_exact",
        "profile_phrase_exact",
    ):
        owners: dict[tuple[LabelDimension, str], str] = {}
        for rule in rules:
            phrases = cast(frozenset[str], getattr(rule, attribute))
            for phrase in phrases:
                previous = owners.setdefault((rule.dimension, phrase), rule.label_id)
                if previous != rule.label_id:
                    raise DataReadinessError(
                        f"exact {rule.dimension} security label phrase {phrase!r} is ambiguous between {previous} and {rule.label_id}"
                    )


def _rule_payload(rule: LabelRule) -> dict[str, object]:
    return {
        "id": rule.label_id,
        "dimension": rule.dimension,
        "membership_sector_exact": sorted(rule.membership_sector_exact),
        "membership_industry_exact": sorted(rule.membership_industry_exact),
        "profile_phrase_exact": sorted(rule.profile_phrase_exact),
        "compatible_membership_label_ids": sorted(rule.compatible_membership_label_ids),
    }


def _required_string(raw: dict[str, object], key: str) -> str:
    return _nonempty(raw.get(key), field=key)


def _nonempty(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataReadinessError(f"{field} must be a non-empty string")
    return value.strip()


def _string_collection(raw: dict[str, object], key: str) -> tuple[str, ...]:
    value = raw.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise DataReadinessError(f"{key} must be a list of non-empty strings")
    values = tuple(item.strip() for item in value)
    if len(values) != len(set(values)):
        raise DataReadinessError(f"{key} contains duplicate values")
    return values


def _normalized_collection(raw: dict[str, object], key: str) -> tuple[str, ...]:
    values = _string_collection(raw, key)
    normalized = tuple(_normalize_exact(item) for item in values)
    if any(not item for item in normalized):
        raise DataReadinessError(f"{key} contains an empty normalized phrase")
    if len(normalized) != len(set(normalized)):
        raise DataReadinessError(f"{key} contains duplicate normalized phrases")
    return normalized


def _optional_string_collection(
    raw: dict[str, object],
    key: str,
) -> tuple[str, ...]:
    if key not in raw:
        return ()
    return _string_collection(raw, key)


def _normalize_exact(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _aware_utc(value: datetime, *, field: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DataReadinessError(f"{field} must be timezone-aware")
    return value.astimezone(UTC)


def _optional_aware_utc(value: datetime | None, *, field: str) -> datetime | None:
    return None if value is None else _aware_utc(value, field=field)


def _validated_sha256(value: str) -> str:
    normalized = value.strip().lower()
    if _SHA256.fullmatch(normalized) is None:
        raise DataReadinessError("source_content_sha256 must be a lowercase SHA-256 digest")
    return normalized


def _utc_text(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
