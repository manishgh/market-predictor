"""Pure complete-version annotation checks; these results are not source authorities."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.research.issuer_review_packet_contracts import PacketVersion, policy_identities
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin

Family = Literal["earnings", "guidance"]
Nonempty = Annotated[str, Field(min_length=1)]
Offset = Annotated[int, Field(ge=0)]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _encoded(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


class SourceSpan(HoldingContract):
    start: Offset
    end: Offset
    quote: Nonempty

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.end <= self.start:
            raise ValueError("source span must be nonempty")
        return self


class AnnotationVerdicts(HoldingContract):
    family_present: bool | None
    issuer_correct: bool | None
    announced_or_reported: bool | None
    explicit_fiscal_period: bool | None
    action_supported: bool | None

    @property
    def all_true(self) -> bool:
        return all(value is True for value in self.model_dump().values())

    @property
    def all_null(self) -> bool:
        return all(value is None for value in self.model_dump().values())


class VersionKey(HoldingContract):
    version_id: Sha256
    cluster_id: Sha256
    security_id: str | None
    source_family: Literal["sec", "alpaca"]
    source_id: str
    source_version_sha256: Sha256 | None
    content_json_sha256: Sha256
    metadata_sha256: Sha256
    text_sha256: Sha256 | None

    @classmethod
    def from_version(cls, version: PacketVersion) -> Self:
        values = version.model_dump()
        return cls.model_validate({key: values[key] for key in cls.model_fields})


class SourceEventAnnotation(HoldingContract):
    annotation_id: Nonempty
    version: VersionKey
    event_family: Family
    anchor: SourceSpan
    issuer: SourceSpan | None
    action: SourceSpan | None
    result: SourceSpan | None
    fiscal_period: SourceSpan | None
    verdicts: AnnotationVerdicts

    @model_validator(mode="after")
    def contained_roles(self) -> Self:
        roles = (self.issuer, self.action, self.result, self.fiscal_period)
        if self.verdicts.all_true and any(role is None for role in roles):
            raise ValueError("positive event requires all four source roles")
        if any(role is not None and not self.anchor.start <= role.start < role.end <= self.anchor.end for role in roles):
            raise ValueError("event role lies outside its statement anchor")
        return self


class VersionAssessment(HoldingContract):
    version: VersionKey
    inspection_status: Literal["complete", "partial", "unavailable"]
    assessment_status: Literal["reviewed", "unresolved", "unavailable"]
    verdicts: AnnotationVerdicts
    annotations: tuple[SourceEventAnnotation, ...]
    negative_spans: tuple[SourceSpan, ...]


class ReviewerIdentity(HoldingContract):
    reviewer_id: Nonempty
    reviewer_kind: Literal["human", "model_assisted"]
    reviewer_model: str | None
    evidence_scope: Literal["source_only"]
    independent_review: Literal[True]

    @field_validator("independent_review", mode="before")
    @classmethod
    def exact_independence(cls, value: object) -> bool:
        if value is not True:
            raise ValueError("independent source review requires literal true")
        return True

    @model_validator(mode="after")
    def assistance(self) -> Self:
        if not self.reviewer_id.strip():
            raise ValueError("reviewer identity is blank")
        if ((self.reviewer_kind == "human" and self.reviewer_model is not None)
                or (self.reviewer_kind == "model_assisted" and not (self.reviewer_model and self.reviewer_model.strip()))):
            raise ValueError("review assistance must be declared exactly")
        return self


class _PacketBindings(HoldingContract):
    sample_id: Sha256
    cluster_id: Sha256
    event_family_to_assess: Family
    request_sha256: Sha256
    frame_sha256: Sha256
    exclusions_sha256: Sha256
    sample_sha256: Sha256
    frame_policy_sha256: Sha256
    document_policy_sha256: Sha256
    correspondence_policy_sha256: Sha256
    version_inventory_sha256: Sha256
    version_count: Annotated[int, Field(gt=0)]
    occurrence_inventory_sha256: Sha256
    occurrence_count: Offset


class PacketOccurrence(HoldingContract):
    phase: Nonempty
    ordinal: Annotated[int, Field(gt=0)]
    version_id: str
    record_json_sha256: Sha256


class BlindReviewPacket(_PacketBindings):
    schema_version: Literal["market_predictor.issuer_review_packets_blind_packet"] = Field(alias="schema")
    instructions: str
    versions: tuple[PacketVersion, ...]
    occurrences: tuple[PacketOccurrence, ...]


class PacketAssessment(_PacketBindings):
    schema_version: Literal["market_predictor.issuer_source_packet_assessment"] = Field(alias="schema")
    packet_publication: SourcePin
    reviewer: ReviewerIdentity
    versions: tuple[VersionAssessment, ...]


@dataclass(frozen=True)
class ValidatedPacketAssessments:
    packet: BlindReviewPacket
    assessment: PacketAssessment


def parse_packet_assessment(data: bytes) -> PacketAssessment:
    parse_strict_json_object(data, label="source packet assessment")
    return PacketAssessment.model_validate_json(data)


def _span_matches(span: SourceSpan, text: str) -> None:
    _require(span.end <= len(text) and text[span.start:span.end] == span.quote,
             "annotation quote differs from exact Unicode source interval")


def _packet_inventory(packet: BlindReviewPacket) -> None:
    for name, policy_digest in policy_identities().items():
        _require(getattr(packet, name) == policy_digest, "packet annotation policy identity differs")
    ids = [version.version_id for version in packet.versions]
    _require(ids == sorted(set(ids)) and len(ids) == packet.version_count, "packet version inventory incomplete or reordered")
    digest = hashlib.sha256()
    for version in packet.versions:
        _require(version.cluster_id == packet.cluster_id, "packet version belongs to another cluster")
        digest.update(_encoded(version.model_dump(mode="json", exclude={"text"})) + b"\n")
        if version.display_disposition == "readable_source_text":
            _require(isinstance(version.text, str) and bool(version.text) and not version.omission_reasons
                     and not version.physical_unavailable_reasons and version.security_id is not None,
                     "readable packet version has missing source evidence")
            assert version.text is not None
            _require(hashlib.sha256(version.text.encode("utf-8")).hexdigest() == version.text_sha256,
                     "packet source text hash differs")
        else:
            _require(version.text is None and bool(version.omission_reasons), "metadata-only packet substituted source text")
    _require(digest.hexdigest() == packet.version_inventory_sha256, "packet version inventory hash differs")
    occurrences = [(item.phase, item.ordinal) for item in packet.occurrences]
    _require(occurrences == sorted(set(occurrences)) and len(occurrences) == packet.occurrence_count,
             "packet occurrence inventory incomplete or reordered")
    digest = hashlib.sha256()
    for item in packet.occurrences:
        _require(not item.version_id or item.version_id in ids, "packet occurrence has an unknown version")
        digest.update(_encoded(item.model_dump(mode="json")) + b"\n")
    _require(digest.hexdigest() == packet.occurrence_inventory_sha256, "packet occurrence inventory hash differs")


def validate_packet_assessments(packet: BlindReviewPacket, assessment: PacketAssessment) -> ValidatedPacketAssessments:
    """Validate one reviewer's entire sample packet; perform no file admission or metrics."""
    _packet_inventory(packet)
    _require(all(getattr(packet, name) == getattr(assessment, name) for name in _PacketBindings.model_fields),
             "review assessment packet/sample/inventory binding differs")
    expected = tuple(VersionKey.from_version(version) for version in packet.versions)
    actual = tuple(item.version for item in assessment.versions)
    _require(actual == expected, "review must assess every exact packet version once in inventory order")
    annotation_ids: set[str] = set()
    annotation_payloads: set[bytes] = set()
    for version, item in zip(packet.versions, assessment.versions, strict=True):
        if version.display_disposition == "metadata_only":
            _require(item.inspection_status == item.assessment_status == "unavailable"
                     and item.verdicts.all_null and not item.annotations and not item.negative_spans,
                     "metadata-only version cannot supply a positive or a measured negative")
            continue
        _require(item.inspection_status in ("complete", "partial") and item.assessment_status != "unavailable",
                 "readable source requires explicit inspection or unresolved assessment")
        if item.inspection_status == "partial":
            _require(item.assessment_status == "unresolved" and item.verdicts.all_null and not item.negative_spans,
                     "partial inspection cannot provide document verdicts or a measured negative")
        if item.assessment_status == "reviewed":
            _require(item.inspection_status == "complete" and all(value is not None for value in item.verdicts.model_dump().values()),
                     "resolved document review requires complete inspection and five explicit verdicts")
        if item.verdicts.all_true:
            _require(any(event.verdicts.all_true for event in item.annotations), "positive document requires one coherent positive event")
        if item.verdicts.family_present is False:
            _require(not any(event.verdicts.family_present is True for event in item.annotations),
                     "negative document contradicts a source event annotation")
        text = version.text
        assert text is not None
        for negative_span in item.negative_spans:
            _span_matches(negative_span, text)
        for event in item.annotations:
            _require(event.version == item.version and event.event_family == packet.event_family_to_assess,
                     "event annotation changes version, security or family")
            _require(event.annotation_id not in annotation_ids, "duplicate annotation ID")
            annotation_ids.add(event.annotation_id)
            payload = _encoded(event.model_dump(mode="json", exclude={"annotation_id"}))
            _require(payload not in annotation_payloads, "duplicate annotation payload")
            annotation_payloads.add(payload)
            for span in (event.anchor, event.issuer, event.action, event.result, event.fiscal_period):
                if span is not None:
                    _span_matches(span, text)
    return ValidatedPacketAssessments(packet, assessment)
