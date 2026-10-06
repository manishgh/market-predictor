"""Synthetic unit evidence only; never operational source-review judgments."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import pytest
from pydantic import ValidationError

from market_predictor.core.errors import DataReadinessError
from market_predictor.research import issuer_source_annotations as a


def _span(text: str, quote: str) -> dict[str, Any]:
    start = text.index(quote)
    return {"start": start, "end": start + len(quote), "quote": quote}


def _fixture() -> tuple[Any, Any]:
    text = "Acme 🧪 raises EPS to $7 for FY2021."
    versions: list[dict[str, Any]] = []
    assessments: list[dict[str, Any]] = []
    cluster = "b" * 64
    for index in range(1, 6):
        readable = index <= 2
        version = dict(version_id=str(index) * 64, cluster_id=cluster, security_id=None if index == 4 else "issuer",
            source_family="sec", source_id=f"filing/{index}", source_version_sha256="c" * 64,
            content_json_sha256="d" * 64, metadata_sha256="e" * 64,
            text_sha256=None if index == 5 else hashlib.sha256(text.encode()).hexdigest(), ticker="UNIT",
            published_at_utc="2021-01-01T00:00:00+00:00",
            version_available_at_utc="2025-01-01T00:00:00+00:00" if index == 3 else "2021-01-01T00:00:00+00:00",
            event_available_at_utc="2021-01-01T00:00:00+00:00", identity_available_at_utc="2021-01-01T00:00:00+00:00",
            first_seen_at_utc="2026-09-01T00:00:00+00:00", identity_authority_sha256="f" * 64,
            availability_semantics="historical_proxy", source_locator=f"unit://filing/{index}", content_kind="filing_document",
            encoding="utf-8", alias_proof={"aliases": ["Acme"]}, physical_unavailable_reasons=[] if readable else ["unavailable"],
            display_disposition="readable_source_text" if readable else "metadata_only",
            omission_reasons=[] if readable else ["future" if index == 3 else "unknown" if index == 4 else "unreadable"],
            text=text if readable else None)
        key = {name: version[name] for name in a.VersionKey.model_fields}
        verdicts = dict.fromkeys(a.AnnotationVerdicts.model_fields, True if readable else None)
        event = dict(annotation_id=f"annotation-{index}", version=key, event_family="guidance",
            anchor=_span(text, text), issuer=_span(text, "Acme"), action=_span(text, "raises"),
            result=_span(text, "EPS to $7"), fiscal_period=_span(text, "FY2021"), verdicts=verdicts)
        assessments.append(dict(version=key, inspection_status="complete" if readable else "unavailable",
            assessment_status="reviewed" if readable else "unavailable", verdicts=verdicts,
            annotations=[event] if readable else [], negative_spans=[]))
        versions.append(version)
    occurrences = [dict(phase="document", ordinal=index, version_id=version["version_id"], record_json_sha256="f" * 64)
                   for index, version in enumerate(versions, 1)]
    version_hash = hashlib.sha256(b"".join(a._encoded({key: value for key, value in row.items() if key != "text"}) + b"\n"
                                        for row in versions)).hexdigest()
    occurrence_hash = hashlib.sha256(b"".join(a._encoded(row) + b"\n" for row in occurrences)).hexdigest()
    bindings = dict(sample_id="a" * 64, cluster_id=cluster, event_family_to_assess="guidance", request_sha256="a" * 64,
        frame_sha256="b" * 64, exclusions_sha256="c" * 64, sample_sha256="d" * 64, **a.policy_identities(),
        version_inventory_sha256=version_hash, version_count=5, occurrence_inventory_sha256=occurrence_hash, occurrence_count=5)
    packet = a.BlindReviewPacket.model_validate_json(json.dumps(dict(bindings,
        schema="market_predictor.issuer_review_packets_blind_packet", instructions="Inspect complete source text.",
        versions=versions, occurrences=occurrences)))
    assessment = a.parse_packet_assessment(json.dumps(dict(bindings,
        schema="market_predictor.issuer_source_packet_assessment", packet_publication={"path": "unit/_manifest.json", "sha256": "a" * 64},
        reviewer=dict(reviewer_id="unit-reviewer", reviewer_kind="human", reviewer_model=None,
            evidence_scope="source_only", independent_review=True), versions=assessments)).encode())
    return packet, assessment


def _change(assessment: Any, change: Any) -> Any:
    payload = assessment.model_dump(mode="json", by_alias=True)
    change(payload)
    return a.parse_packet_assessment(json.dumps(payload).encode())


def test_complete_source_versions_and_unicode_are_preserved() -> None:
    packet, assessment = _fixture()
    result = a.validate_packet_assessments(packet, assessment)
    assert len(result.assessment.versions) == 5
    assert result.assessment.versions[2].verdicts.all_null
    assert result.packet.versions[2].text_sha256 is not None
    assert result.packet.versions[4].text_sha256 is None
    role = result.assessment.versions[0].annotations[0].action
    assert role.start == 7  # Non-BMP character counts once, not UTF-16 twice.


@pytest.mark.parametrize("mode", ["missing", "duplicate", "reordered", "security", "text", "sample", "inventory"])
def test_complete_keys_and_bindings_required(mode: str) -> None:
    packet, assessment = _fixture()
    def change(data: dict[str, Any]) -> None:
        if mode == "missing":
            data["versions"].pop()
        elif mode == "duplicate":
            data["versions"][1] = data["versions"][0]
        elif mode == "reordered":
            data["versions"].reverse()
        elif mode == "security":
            data["versions"][0]["version"]["security_id"] = "another-issuer"
        elif mode == "text":
            data["versions"][0]["version"]["text_sha256"] = "1" * 64
        elif mode == "sample":
            data["sample_id"] = "1" * 64
        else:
            data["occurrence_inventory_sha256"] = "1" * 64
    with pytest.raises(DataReadinessError):
        a.validate_packet_assessments(packet, _change(assessment, change))


@pytest.mark.parametrize("mode", ["positive", "negative", "inspection", "annotation"])
def test_metadata_only_never_becomes_a_label(mode: str) -> None:
    packet, assessment = _fixture()
    def change(data: dict[str, Any]) -> None:
        item = data["versions"][2]
        if mode in ("positive", "negative"):
            item["verdicts"]["family_present"] = mode == "positive"
        elif mode == "inspection":
            item["inspection_status"] = "complete"
        else:
            item["annotations"] = data["versions"][0]["annotations"]
    with pytest.raises(DataReadinessError):
        a.validate_packet_assessments(packet, _change(assessment, change))


def test_partial_empty_review_is_unresolved_not_negative() -> None:
    packet, assessment = _fixture()
    def partial(data: dict[str, Any]) -> None:
        item = data["versions"][0]
        item.update(inspection_status="partial", assessment_status="unresolved", annotations=[])
        item["verdicts"] = dict.fromkeys(item["verdicts"])
    changed = _change(assessment, partial)
    assert a.validate_packet_assessments(packet, changed).assessment.versions[0].verdicts.all_null
    def negative(data: dict[str, Any]) -> None:
        data["versions"][0]["verdicts"]["family_present"] = False
    with pytest.raises(DataReadinessError, match="partial inspection"):
        a.validate_packet_assessments(packet, _change(changed, negative))


@pytest.mark.parametrize("mode", ["quote", "offset", "version", "family", "duplicate_id", "duplicate_payload"])
def test_source_event_poison_rejected(mode: str) -> None:
    packet, assessment = _fixture()
    def change(data: dict[str, Any]) -> None:
        events = data["versions"][0]["annotations"]
        event = events[0]
        if mode == "quote":
            event["action"]["quote"] = "RAISES"
        elif mode == "offset":
            event["action"]["start"] += 1
            event["action"]["end"] += 1
        elif mode == "version":
            event["version"]["version_id"] = "9" * 64
        elif mode == "family":
            event["event_family"] = "earnings"
        else:
            copied = json.loads(json.dumps(event))
            if mode == "duplicate_payload":
                copied["annotation_id"] = "another-id"
            events.append(copied)
    with pytest.raises(DataReadinessError):
        a.validate_packet_assessments(packet, _change(assessment, change))


@pytest.mark.parametrize("mode", ["bool_offset", "int_verdict", "unknown", "missing_role", "outside_anchor", "int_independence"])
def test_strict_types_and_role_shape_rejected(mode: str) -> None:
    _, assessment = _fixture()
    def change(data: dict[str, Any]) -> None:
        event = data["versions"][0]["annotations"][0]
        if mode == "bool_offset":
            event["issuer"]["start"] = False
        elif mode == "int_verdict":
            event["verdicts"]["family_present"] = 1
        elif mode == "unknown":
            event["candidate_id"] = "forbidden-hint"
        elif mode == "missing_role":
            event["fiscal_period"] = None
        elif mode == "outside_anchor":
            event["anchor"]["start"] = 1
        else:
            data["reviewer"]["independent_review"] = 1
    with pytest.raises(ValidationError):
        _change(assessment, change)


def test_duplicate_json_keys_rejected() -> None:
    with pytest.raises((DataReadinessError, ValueError)):
        a.parse_packet_assessment(b'{"schema":"one","schema":"two"}')


def test_complete_empty_negative_is_explicit() -> None:
    packet, assessment = _fixture()
    def negative(data: dict[str, Any]) -> None:
        item = data["versions"][0]
        item["annotations"] = []
        item["verdicts"] = dict.fromkeys(item["verdicts"], False)
    result = a.validate_packet_assessments(packet, _change(assessment, negative))
    assert result.assessment.versions[0].inspection_status == "complete"


def test_rehashed_but_lost_occurrence_inventory_rejected() -> None:
    packet, assessment = _fixture()
    data = packet.model_dump(mode="json", by_alias=True)
    data["occurrences"].pop()
    changed = a.BlindReviewPacket.model_validate_json(json.dumps(data))
    with pytest.raises(DataReadinessError, match="occurrence inventory"):
        a.validate_packet_assessments(changed, assessment)
