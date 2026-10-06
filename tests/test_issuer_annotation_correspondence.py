"""Tiny synthetic UNIT cases; never operational source-review judgments."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any

import pytest

from market_predictor.catalysts.issuer_events.content_review import EvidenceSpan, ReviewCandidate
from market_predictor.core.errors import DataReadinessError
from market_predictor.research import issuer_annotation_correspondence as c
from market_predictor.research import issuer_source_annotations as a
from tests import test_issuer_source_annotations as fixtures


def _pair(packet: Any, assessment: Any) -> tuple[Any, Any]:
    second = assessment.model_copy(update={"reviewer": assessment.reviewer.model_copy(update={"reviewer_id": "unit-second"})})
    return a.validate_packet_assessments(packet, assessment), a.validate_packet_assessments(packet, second)


def _candidate(annotation: Any) -> ReviewCandidate:
    spans = tuple(EvidenceSpan(role, span.start, span.end, span.quote, ("unit://source",))
                  for role, span in (("statement", annotation.anchor), ("issuer", annotation.issuer),
                    ("action", annotation.action), ("result", annotation.result), ("fiscal_period", annotation.fiscal_period)))
    return ReviewCandidate("unit-candidate", "guidance", annotation.action.quote.casefold(), "synthetic-rule",
                           annotation.fiscal_period.quote, (), spans)


def _case() -> tuple[Any, Any, ReviewCandidate]:
    packet, assessment = fixtures._fixture()
    return packet, assessment, _candidate(assessment.versions[0].annotations[0])


def _run(packet: Any, assessment: Any, candidate: ReviewCandidate, index: int = 0) -> Any:
    return c.correspond_candidate(version=packet.versions[index], candidate=candidate,
                                  reviewer_assessments=_pair(packet, assessment))


def test_exact_unicode_roles_and_proof_references_support_only_this_version() -> None:
    packet, assessment, candidate = _case()
    result = _run(packet, assessment, candidate)
    assert result.supported and result.version == a.VersionKey.from_version(packet.versions[0])
    assert result.candidate == candidate and result.sample_id == packet.sample_id
    assert [item.structural_match_count for item in result.reviewer_results] == [1, 1]
    assert result.reviewer_results[0].structural_matches[0].roles[1].start == 7
    assert result.reviewer_results[0].structural_matches[0].verdicts.all_true
    assert not any((result.training_eligible, result.serving_eligible, result.promotion_eligible,
                    result.qualification_established, result.economic_eligible))


def test_independent_anchor_margins_may_differ_but_roles_are_exact() -> None:
    packet, assessment, candidate = _case()
    def narrower(data: dict[str, Any]) -> None:
        anchor = data["versions"][0]["annotations"][0]["anchor"]
        anchor["end"] -= 1
        anchor["quote"] = anchor["quote"][:-1]
    changed = fixtures._change(assessment, narrower)
    first, second = _pair(packet, changed)
    first = a.validate_packet_assessments(packet, assessment)
    result = c.correspond_candidate(version=packet.versions[0], candidate=candidate, reviewer_assessments=(first, second))
    assert result.supported


@pytest.mark.parametrize("poison", ["missing", "duplicate", "quote", "utf16_offset", "action", "fiscal", "unresolved"])
def test_candidate_role_or_value_uncertainty_cannot_be_rescued(poison: str) -> None:
    packet, assessment, candidate = _case()
    spans = list(candidate.spans)
    if poison == "missing":
        candidate = replace(candidate, spans=tuple(spans[:-1]))
    elif poison == "duplicate":
        candidate = replace(candidate, spans=(*candidate.spans, spans[1]))
    elif poison in ("quote", "utf16_offset"):
        spans[2] = replace(spans[2], text="RAISES") if poison == "quote" else replace(spans[2], start=spans[2].start + 1)
        candidate = replace(candidate, spans=tuple(spans))
    elif poison == "action":
        candidate = replace(candidate, action="RAISES")
    elif poison == "fiscal":
        candidate = replace(candidate, fiscal_period="FY 2021")
    else:
        candidate = replace(candidate, unresolved_reasons=("synthetic_unresolved_source",))
    result = _run(packet, assessment, candidate)
    assert not result.supported and result.candidate_reasons
    if poison == "unresolved":
        assert "synthetic_unresolved_source" in result.candidate_reasons


@pytest.mark.parametrize("field", ["family_present", "issuer_correct", "announced_or_reported",
                                   "explicit_fiscal_period", "action_supported"])
@pytest.mark.parametrize("value", [False, None])
def test_document_judgments_cannot_fill_a_matching_event_judgment(field: str, value: bool | None) -> None:
    packet, assessment, candidate = _case()
    def uncertainty(data: dict[str, Any]) -> None:
        item = data["versions"][0]
        item["annotations"][0]["verdicts"][field] = value
        item["verdicts"] = dict.fromkeys(item["verdicts"], None)
        item["assessment_status"] = "unresolved"
    changed = fixtures._change(assessment, uncertainty)
    result = _run(packet, changed, candidate)
    assert not result.supported
    assert all(item.structural_match_count == 1 for item in result.reviewer_results)
    assert getattr(result.reviewer_results[0].structural_matches[0].verdicts, field) is value


def test_two_structural_matches_count_before_favorable_verdict_selection() -> None:
    packet, assessment, candidate = _case()
    def ambiguous(data: dict[str, Any]) -> None:
        original = data["versions"][0]["annotations"][0]
        other = {**original, "annotation_id": "second-match", "verdicts": {**original["verdicts"], "action_supported": False}}
        data["versions"][0]["annotations"].append(other)
    result = _run(packet, fixtures._change(assessment, ambiguous), candidate)
    assert not result.supported
    assert all(item.structural_match_count == 2 and "multiple_structural_matches" in item.reasons
               for item in result.reviewer_results)


def test_partial_inspection_with_positive_source_annotation_cannot_rescue() -> None:
    packet, assessment, candidate = _case()
    def partial(data: dict[str, Any]) -> None:
        item = data["versions"][0]
        item.update(inspection_status="partial", assessment_status="unresolved")
        item["verdicts"] = dict.fromkeys(item["verdicts"], None)
    result = _run(packet, fixtures._change(assessment, partial), candidate)
    assert not result.supported and all(item.structural_match_count == 1 for item in result.reviewer_results)


@pytest.mark.parametrize("mode", ["null_role", "broad_role", "anchor_outside_candidate"])
def test_missing_or_nonexact_roles_and_anchor_do_not_correspond(mode: str) -> None:
    packet, assessment, candidate = _case()
    if mode == "anchor_outside_candidate":
        statement = candidate.spans[0]
        candidate = replace(candidate, spans=(replace(statement, end=statement.end - 1, text=statement.text[:-1]),
                                               *candidate.spans[1:]))
    else:
        def alter(data: dict[str, Any]) -> None:
            item = data["versions"][0]
            event = item["annotations"][0]
            if mode == "null_role":
                event["issuer"] = None
                event["verdicts"] = dict.fromkeys(event["verdicts"], None)
                item["verdicts"] = dict.fromkeys(item["verdicts"], None)
                item["assessment_status"] = "unresolved"
            else:
                event["action"] = fixtures._span(packet.versions[0].text, "raises EPS")
        assessment = fixtures._change(assessment, alter)
    result = _run(packet, assessment, candidate)
    assert not result.supported and all(item.structural_match_count == 0 for item in result.reviewer_results)


def test_one_positive_reviewer_cannot_replace_other_unresolved_verdicts() -> None:
    packet, assessment, candidate = _case()
    first, second = _pair(packet, assessment)
    def uncertainty(data: dict[str, Any]) -> None:
        item = data["versions"][0]
        item["verdicts"] = dict.fromkeys(item["verdicts"], None)
        item["assessment_status"] = "unresolved"
        item["annotations"][0]["verdicts"]["issuer_correct"] = None
    second = a.validate_packet_assessments(packet, fixtures._change(second.assessment, uncertainty))
    result = c.correspond_candidate(version=packet.versions[0], candidate=candidate, reviewer_assessments=(first, second))
    assert not result.supported and result.reviewer_results[0].supported
    assert not result.reviewer_results[1].supported
    assert result.reviewer_results[1].structural_matches[0].verdicts.issuer_correct is None


def test_true_other_exhibit_never_rescues_empty_cover_assessment() -> None:
    packet, assessment, candidate = _case()
    def negative_cover(data: dict[str, Any]) -> None:
        item = data["versions"][0]
        item["annotations"] = []
        item["verdicts"] = dict.fromkeys(item["verdicts"], False)
    result = _run(packet, fixtures._change(assessment, negative_cover), candidate)
    assert not result.supported and all(item.structural_match_count == 0 for item in result.reviewer_results)


def test_second_true_statement_in_same_document_does_not_rescue_first() -> None:
    packet, assessment, candidate = _case()
    first = packet.versions[0]
    original = first.text
    text = original + " " + original
    changed = first.model_copy(update={"text": text, "text_sha256": hashlib.sha256(text.encode()).hexdigest()})
    versions = (changed, *packet.versions[1:])
    digest = hashlib.sha256(b"".join(a._encoded(item.model_dump(mode="json", exclude={"text"})) + b"\n"
                                     for item in versions)).hexdigest()
    packet = packet.model_copy(update={"versions": versions, "version_inventory_sha256": digest})
    key = a.VersionKey.from_version(changed).model_dump(mode="json")
    def later_statement(data: dict[str, Any]) -> None:
        data["version_inventory_sha256"] = digest
        item = data["versions"][0]
        item["version"] = key
        event = item["annotations"][0]
        event["version"] = key
        for role in ("anchor", "issuer", "action", "result", "fiscal_period"):
            event[role] = {**event[role], "start": event[role]["start"] + len(original) + 1,
                           "end": event[role]["end"] + len(original) + 1}
    result = _run(packet, fixtures._change(assessment, later_statement), candidate)
    assert not result.supported and all(item.structural_match_count == 0 for item in result.reviewer_results)


@pytest.mark.parametrize("index", [2, 3, 4])
def test_metadata_only_future_unknown_unreadable_remain_unsupported(index: int) -> None:
    packet, assessment, candidate = _case()
    result = _run(packet, assessment, candidate, index)
    assert not result.supported and "source_version_unavailable" in result.candidate_reasons
    assert all(item.document_verdicts.all_null for item in result.reviewer_results)


@pytest.mark.parametrize("field", ["source_id", "source_version_sha256", "security_id", "text_sha256", "metadata_sha256"])
def test_candidate_version_substitution_rejected(field: str) -> None:
    packet, assessment, candidate = _case()
    value = "another-source" if field in ("source_id", "security_id") else "0" * 64
    version = packet.versions[0].model_copy(update={field: value})
    with pytest.raises(DataReadinessError, match="exact packet version"):
        c.correspond_candidate(version=version, candidate=candidate, reviewer_assessments=_pair(packet, assessment))


def test_exact_duplicate_refs_deduplicate_and_other_refs_remain_explicit() -> None:
    packet, assessment, candidate = _case()
    first = c.CandidateOccurrence(packet.versions[0], candidate, "unit-occurrence-1")
    other = c.CandidateOccurrence(packet.versions[0], candidate, "unit-occurrence-2")
    result = c.correspond_candidates(occurrences=(first, first, other), reviewer_assessments=_pair(packet, assessment))
    assert [item.reference for item in result] == ["unit-occurrence-1", "unit-occurrence-2"]
    assert all(item.correspondence.supported for item in result)


@pytest.mark.parametrize("mode", ["candidate_payload", "candidate_version", "reference_payload"])
def test_conflicting_reuse_rejected(mode: str) -> None:
    packet, assessment, candidate = _case()
    first = c.CandidateOccurrence(packet.versions[0], candidate, "unit-occurrence-1")
    if mode == "candidate_version":
        second = c.CandidateOccurrence(packet.versions[1], candidate, "unit-occurrence-2")
    else:
        altered = replace(candidate, rule_id="changed", candidate_id="another" if mode == "reference_payload" else candidate.candidate_id)
        second = c.CandidateOccurrence(packet.versions[0], altered,
                                       first.reference if mode == "reference_payload" else "unit-occurrence-2")
    with pytest.raises(DataReadinessError, match="conflicting"):
        c.correspond_candidates(occurrences=(first, second), reviewer_assessments=_pair(packet, assessment))
