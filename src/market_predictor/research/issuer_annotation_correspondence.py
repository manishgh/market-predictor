"""Exact source correspondence; no qualification or source admission.

Only validated complete assessments and canonical candidate DTOs are consumed.
The eventual operational owner must establish file/source admission separately.
"""
from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from market_predictor.catalysts.issuer_events.content_review import EvidenceSpan, ReviewCandidate
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.research.issuer_review_packet_contracts import CORRESPONDENCE_POLICY, PacketVersion
from market_predictor.research.issuer_source_annotations import (
    AnnotationVerdicts,
    SourceEventAnnotation,
    SourceSpan,
    ValidatedPacketAssessments,
    VersionAssessment,
    VersionKey,
)

_ROLES = ("issuer", "action", "result", "fiscal_period")


@dataclass(frozen=True)
class AnnotationReference:
    reviewer_id: str
    annotation_id: str
    annotation_sha256: str
    version: VersionKey
    event_family: str
    anchor: SourceSpan
    roles: tuple[SourceSpan, ...]
    verdicts: AnnotationVerdicts


@dataclass(frozen=True)
class ReviewerCorrespondence:
    reviewer_id: str
    inspection_status: str
    assessment_status: str
    document_verdicts: AnnotationVerdicts
    structural_matches: tuple[AnnotationReference, ...]
    supported: bool
    reasons: tuple[str, ...]

    @property
    def structural_match_count(self) -> int:
        return len(self.structural_matches)


@dataclass(frozen=True)
class CandidateCorrespondence:
    sample_id: str
    version: VersionKey
    candidate: ReviewCandidate
    candidate_payload_sha256: str
    correspondence_policy_sha256: str
    reviewer_results: tuple[ReviewerCorrespondence, ReviewerCorrespondence]
    candidate_reasons: tuple[str, ...]
    supported: bool
    training_eligible: Literal[False] = False
    serving_eligible: Literal[False] = False
    promotion_eligible: Literal[False] = False
    qualification_established: Literal[False] = False
    economic_eligible: Literal[False] = False


@dataclass(frozen=True)
class CandidateOccurrence:
    version: PacketVersion
    candidate: ReviewCandidate
    reference: str


@dataclass(frozen=True)
class OccurrenceCorrespondence:
    reference: str
    correspondence: CandidateCorrespondence


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise DataReadinessError(reason)


def _exact(span: EvidenceSpan | SourceSpan, text: str) -> bool:
    quote = span.text if isinstance(span, EvidenceSpan) else span.quote
    return (type(span.start) is int and type(span.end) is int and isinstance(quote, str)
            and 0 <= span.start < span.end <= len(text) and text[span.start:span.end] == quote)


def _candidate_parts(version: PacketVersion, candidate: ReviewCandidate
                     ) -> tuple[dict[str, EvidenceSpan], tuple[str, ...]]:
    reasons = list(candidate.unresolved_reasons)
    if (version.display_disposition != "readable_source_text" or version.physical_unavailable_reasons
            or version.omission_reasons or version.security_id is None or version.text is None):
        reasons.append("source_version_unavailable")
    spans: dict[str, EvidenceSpan] = {span.role: span for span in candidate.spans}
    if len(candidate.spans) != 5 or len(spans) != 5 or set(spans) != {"statement", *_ROLES}:
        reasons.append("candidate_required_role_inventory_invalid")
        return spans, tuple(reasons)
    if version.text is None:
        return spans, tuple(reasons)
    if not all(_exact(span, version.text) for span in candidate.spans):
        reasons.append("candidate_span_or_quote_invalid")
    statement = spans["statement"]
    if not all(statement.start <= spans[role].start < spans[role].end <= statement.end for role in _ROLES):
        reasons.append("candidate_role_outside_statement")
    if candidate.action != spans["action"].text.casefold():
        reasons.append("candidate_action_differs_from_source_quote")
    if candidate.fiscal_period != spans["fiscal_period"].text:
        reasons.append("candidate_fiscal_period_differs_from_source_quote")
    return spans, tuple(reasons)


def _structural(annotation: SourceEventAnnotation, key: VersionKey, candidate: ReviewCandidate,
                spans: dict[str, EvidenceSpan], text: str) -> bool:
    if annotation.version != key or annotation.event_family != candidate.event_family:
        return False
    anchor, statement = annotation.anchor, spans["statement"]
    if not _exact(anchor, text) or not statement.start <= anchor.start < anchor.end <= statement.end:
        return False
    for role in _ROLES:
        actual: SourceSpan | None = getattr(annotation, role)
        expected = spans[role]
        if (actual is None or not _exact(actual, text)
                or (actual.start, actual.end, actual.quote) != (expected.start, expected.end, expected.text)
                or not anchor.start <= expected.start < expected.end <= anchor.end):
            return False
    return True


def _reviewer(value: ValidatedPacketAssessments, assessment: VersionAssessment, key: VersionKey,
              candidate: ReviewCandidate, spans: dict[str, EvidenceSpan], text: str,
              candidate_reasons: tuple[str, ...]) -> ReviewerCorrespondence:
    # Count every structural match before consulting any annotation verdict.
    complete_roles = set(spans) == {"statement", *_ROLES} and len(candidate.spans) == 5
    matches = tuple(annotation for annotation in assessment.annotations
                    if complete_roles and _structural(annotation, key, candidate, spans, text))
    references = tuple(AnnotationReference(
        reviewer_id=value.assessment.reviewer.reviewer_id,
        annotation_id=annotation.annotation_id,
        annotation_sha256=json_sha256(annotation.model_dump(mode="json")),
        version=key, event_family=annotation.event_family, anchor=annotation.anchor,
        roles=tuple(getattr(annotation, role) for role in _ROLES), verdicts=annotation.verdicts,
    ) for annotation in matches)
    reasons = list(candidate_reasons)
    if assessment.inspection_status != "complete" or assessment.assessment_status != "reviewed":
        reasons.append("reviewer_inspection_or_assessment_unresolved")
    if len(matches) != 1:
        reasons.append("no_structural_match" if not matches else "multiple_structural_matches")
    elif not matches[0].verdicts.all_true:
        reasons.append("matching_annotation_not_all_five_true")
    return ReviewerCorrespondence(value.assessment.reviewer.reviewer_id,
        assessment.inspection_status, assessment.assessment_status, assessment.verdicts,
        references, not reasons, tuple(reasons))


def correspond_candidate(
    *, version: PacketVersion, candidate: ReviewCandidate,
    reviewer_assessments: tuple[ValidatedPacketAssessments, ValidatedPacketAssessments],
) -> CandidateCorrespondence:
    """Evaluate one exact occurrence without reducing clusters or admitting data."""
    _require(len(reviewer_assessments) == 2, "correspondence requires exactly two reviewers")
    left, right = reviewer_assessments
    _require(left.assessment.reviewer.reviewer_id != right.assessment.reviewer.reviewer_id,
             "correspondence requires distinct reviewer identities")
    _require(left.packet == right.packet, "reviewers assessed different complete packets")
    _require(left.assessment.packet_publication == right.assessment.packet_publication,
             "reviewers bound different packet publications")
    _require(candidate.event_family == left.packet.event_family_to_assess, "candidate family differs from packet")
    _require(isinstance(candidate.candidate_id, str) and bool(candidate.candidate_id)
             and candidate.training_eligible is False and candidate.serving_eligible is False,
             "candidate identity or closed research boundary differs")
    key = VersionKey.from_version(version)
    packet_versions = [item for item in left.packet.versions if VersionKey.from_version(item) == key]
    _require(len(packet_versions) == 1 and packet_versions[0] == version, "candidate version differs from exact packet version")
    if version.text is not None:
        _require(hashlib.sha256(version.text.encode("utf-8")).hexdigest() == version.text_sha256,
                 "packet source text hash differs")
    selected: list[VersionAssessment] = []
    for reviewer in reviewer_assessments:
        assessments = [item for item in reviewer.assessment.versions if item.version == key]
        _require(len(assessments) == 1, "candidate version lacks exactly one reviewer assessment")
        selected.append(assessments[0])
    spans, reasons = _candidate_parts(version, candidate)
    results = (_reviewer(left, selected[0], key, candidate, spans, version.text or "", reasons),
               _reviewer(right, selected[1], key, candidate, spans, version.text or "", reasons))
    return CandidateCorrespondence(left.packet.sample_id, key, candidate, json_sha256(asdict(candidate)),
        json_sha256(CORRESPONDENCE_POLICY), results, reasons, all(item.supported for item in results))


def correspond_candidates(
    *, occurrences: Sequence[CandidateOccurrence],
    reviewer_assessments: tuple[ValidatedPacketAssessments, ValidatedPacketAssessments],
) -> tuple[OccurrenceCorrespondence, ...]:
    """Keep distinct references; only an identical occurrence reference deduplicates."""
    candidate_ids: dict[str, str] = {}
    references: dict[str, str] = {}
    results: list[OccurrenceCorrespondence] = []
    for occurrence in occurrences:
        _require(isinstance(occurrence.reference, str) and bool(occurrence.reference), "empty candidate occurrence reference")
        identity = json_sha256({"version": VersionKey.from_version(occurrence.version).model_dump(mode="json"),
                               "candidate": asdict(occurrence.candidate)})
        candidate_id = occurrence.candidate.candidate_id
        _require(candidate_ids.setdefault(candidate_id, identity) == identity, "candidate ID reused with conflicting version or payload")
        _require(references.setdefault(occurrence.reference, identity) == identity, "occurrence reference has conflicting payload")
        if any(item.reference == occurrence.reference for item in results):
            continue
        result = correspond_candidate(version=occurrence.version, candidate=occurrence.candidate,
                                      reviewer_assessments=reviewer_assessments)
        results.append(OccurrenceCorrespondence(occurrence.reference, result))
    return tuple(results)
