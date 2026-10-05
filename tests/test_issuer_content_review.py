"""Synthetic source-bound content candidates; no archives, providers or admission."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from typing import Any

import pandas as pd
import pytest

from market_predictor.catalysts.issuer_events import content_review as owner
from market_predictor.core.errors import DataReadinessError


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def raw_hash(value: dict[str, Any]) -> str:
    return sha(json.dumps(value, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8"))


def test_alpaca_uses_original_saved_record_hash_not_compact_manifest_hash() -> None:
    value = record()
    compact = sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    with pytest.raises(DataReadinessError, match="record hash differs"):
        owner.alpaca_content_evidence(record=value, expected_record_sha256=compact, context=context())
    assert alpaca(value).source_version_sha256 != compact


def context(**changes: Any) -> owner.IssuerContentContext:
    values: dict[str, Any] = {
        "event_id": "event-acme", "security_id": "security-acme", "ticker": "ACME",
        "issuer_aliases": ("Acme", "Acme Corporation"), "identity_authority_sha256": "a" * 64,
        "identity_available_at_utc": pd.Timestamp("2024-07-01T12:00:00Z"),
        "first_seen_at_utc": pd.Timestamp("2024-07-19T16:05:00Z"),
        "availability_semantics": "observed", "availability_policy_sha256": "b" * 64,
    }
    return owner.IssuerContentContext(**{**values, **changes})


def record(**changes: Any) -> dict[str, Any]:
    return {"id": 10, "headline": "Acme reports Q2 2024 earnings.", "summary": "", "content": "",
        "created_at": "2024-07-19T16:00:00Z", "updated_at": "2024-07-19T16:01:00Z", **changes}


def alpaca(value: dict[str, Any], **kwargs: Any) -> owner.IssuerContentEvidence:
    return owner.alpaca_content_evidence(record=value, expected_record_sha256=raw_hash(value),
        context=kwargs.pop("context", context()), **kwargs)


def sec(body: bytes, **kwargs: Any) -> owner.IssuerContentEvidence:
    values: dict[str, Any] = {"body": body, "expected_body_sha256": sha(body),
        "accepted_at_utc": pd.Timestamp("2024-07-19T16:01:00Z"),
        "acceptance_clock_authority_sha256": "c" * 64, "context": context(),
        "document_locator": "accession/exhibit99.htm"}
    return owner.sec_content_evidence(**{**values, **kwargs})


@pytest.mark.parametrize("field", ["content", "summary", "headline"])
def test_selected_field_and_normalized_text_have_distinct_exact_hashes(field: str) -> None:
    value = record(content="", summary="", headline="")
    value[field] = "  Acme reports Q2 2024 earnings.  "
    evidence = alpaca(value)
    assert evidence.payload_sha256 == evidence.source_version_sha256 == raw_hash(value)
    assert evidence.source_locator == f"/{field}"
    assert evidence.content_kind == field
    assert evidence.chosen_field_sha256 == sha(value[field].encode())
    assert evidence.text == "Acme reports Q2 2024 earnings."
    assert evidence.text_sha256 == sha(evidence.text.encode())


def test_field_precedence_does_not_combine_content_with_unselected_headline() -> None:
    value = record(content="Beta reports Q2 2024 earnings.", summary="Acme raises FY2024 guidance.")
    evidence = alpaca(value)
    assert evidence.source_locator == "/content"
    assert owner.extract_review_candidates(evidence).candidates == ()


def test_changed_original_record_or_sec_body_rejected() -> None:
    value = record()
    digest = raw_hash(value)
    value["headline"] = "Acme raises FY2024 guidance."
    with pytest.raises(DataReadinessError, match="record hash"):
        owner.alpaca_content_evidence(record=value, expected_record_sha256=digest, context=context())
    with pytest.raises(DataReadinessError, match="document hash"):
        sec(b"<p>changed</p>", expected_body_sha256=sha(b"<p>original</p>"))


@pytest.mark.parametrize("text", [
    "Acme reports Q2 2024 financial results.",
    "Acme reported Q2 2024 revenue of $20 million.",
    "Acme posts second quarter 2024 earnings.",
])
def test_reported_earnings_has_reconstructable_source_spans_and_no_admission(text: str) -> None:
    evidence = alpaca(record(headline=text))
    result = owner.extract_review_candidates(evidence)
    assert result.disposition == "review_candidates"
    [candidate] = result.candidates
    assert candidate.event_family == "earnings"
    assert candidate.fiscal_period is not None and candidate.unresolved_reasons == ()
    assert {span.role for span in candidate.spans} == {"statement", "issuer", "action", "result", "fiscal_period"}
    for span in candidate.spans:
        assert evidence.text[span.start:span.end] == span.text
        assert span.source_locators == ("/headline",)
    for item in (result, candidate):
        assert item.training_eligible is False and item.serving_eligible is False


@pytest.mark.parametrize("text", [
    "Acme will report Q2 2024 earnings tomorrow.",
    "Preview: Acme reports Q2 2024 earnings tomorrow.",
    "Acme announces a Q2 2024 earnings conference call.",
    "Beta reports Q2 2024 earnings, beating rival Acme.",
    "Beta raises FY2024 guidance while Acme trades higher.",
    "Analysts raise Acme FY2024 earnings estimates.",
])
def test_preview_or_other_action_subject_does_not_propose_issuer_result(text: str) -> None:
    result = owner.extract_review_candidates(alpaca(record(headline=text)))
    assert result.disposition == "unclassified" and result.candidates == ()
    assert result.reasons


@pytest.mark.parametrize(("action", "period"), [("raises", "FY2025"), ("lowers", "Q3 2024")])
def test_direction_and_explicit_fiscal_period_are_retained(action: str, period: str) -> None:
    evidence = alpaca(record(headline=f"Acme {action} {period} revenue guidance."))
    [candidate] = owner.extract_review_candidates(evidence).candidates
    assert candidate.event_family == "guidance" and candidate.action == action
    assert candidate.fiscal_period == period and candidate.unresolved_reasons == ()
    [span] = [span for span in candidate.spans if span.role == "fiscal_period"]
    assert evidence.text[span.start:span.end] == period


def test_missing_fiscal_period_is_explicit_not_inferred_from_publication() -> None:
    [candidate] = owner.extract_review_candidates(alpaca(record(headline="Acme lowers revenue guidance."))).candidates
    assert candidate.fiscal_period is None
    assert candidate.unresolved_reasons == ("missing_explicit_fiscal_period",)
    assert not any(span.role == "fiscal_period" for span in candidate.spans)


def test_each_event_uses_its_own_explicit_fiscal_period_in_combined_statement() -> None:
    evidence = alpaca(record(headline="Acme reports Q2 2024 results and Acme raises FY2025 revenue guidance."))
    candidates = {item.event_family: item for item in owner.extract_review_candidates(evidence).candidates}
    assert set(candidates) == {"earnings", "guidance"}
    assert candidates["earnings"].fiscal_period == "Q2 2024"
    assert candidates["guidance"].fiscal_period == "FY2025"
    for item in candidates.values():
        [span] = [span for span in item.spans if span.role == "fiscal_period"]
        assert evidence.text[span.start:span.end] == item.fiscal_period


@pytest.mark.parametrize("text", [
    "Acme raises its dividend; Beta lowers FY2025 guidance.",
    "Acme reports a new product; Beta announces Q2 2024 results.",
])
def test_other_issuer_clause_cannot_supply_action_result_or_period(text: str) -> None:
    result = owner.extract_review_candidates(alpaca(record(headline=text)))
    assert result.candidates == () and result.disposition == "unclassified"


def test_sec_form_name_without_material_statement_never_becomes_candidate() -> None:
    evidence = sec(b"<html><body><h1>Acme Corporation Form 8-K</h1><p>Item 2.02</p></body></html>")
    assert owner.extract_review_candidates(evidence).candidates == ()


def test_html_entity_inline_spans_and_omitted_nodes_have_element_provenance() -> None:
    body = b"""<html><head><title>Acme raises FY2024 guidance.</title></head><body>
    <script>Acme lowers FY2024 guidance.</script><style>Acme lowers FY2024 guidance.</style>
    <p hidden>Acme lowers FY2024 guidance.</p><p style="display: none">Acme lowers FY2024 guidance.</p>
    <p><b>Acme</b> reports Q2 2024 revenue &amp; earnings.</p></body></html>"""
    evidence = sec(body)
    assert evidence.text == "Acme reports Q2 2024 revenue & earnings."
    assert evidence.payload_sha256 == sha(body)
    assert evidence.chosen_field_sha256 == sha(body)
    [candidate] = owner.extract_review_candidates(evidence).candidates
    assert candidate.event_family == "earnings"
    for span in candidate.spans:
        assert evidence.text[span.start:span.end] == span.text
        assert span.source_locators
        assert all("/html[1]/body[1]/p[" in locator and ":line=" in locator and ":column=" in locator
            for locator in span.source_locators)
    issuer = next(span for span in candidate.spans if span.role == "issuer")
    assert "/b[1]" in issuer.source_locators[0]


def test_table_cells_cannot_be_joined_into_fabricated_statement() -> None:
    evidence = sec(b"<table><tr><td>Acme</td><td>raises FY2024 guidance.</td></tr></table>")
    assert "Acme\nraises" in evidence.text
    assert owner.extract_review_candidates(evidence).candidates == ()


def test_complete_table_cell_statement_retains_cell_locator() -> None:
    evidence = sec(b"<table><tr><td>Acme raises FY2024 guidance.</td></tr></table>")
    [candidate] = owner.extract_review_candidates(evidence).candidates
    assert all("/td[1]" in locator for span in candidate.spans for locator in span.source_locators)


def test_duplicate_statements_and_record_key_order_are_stable_per_version() -> None:
    value = record(content="<p>Acme reports Q2 2024 earnings.</p><p>Acme reports Q2 2024 earnings.</p>")
    result = owner.extract_review_candidates(alpaca(value))
    reordered = dict(reversed(list(value.items())))
    other = owner.extract_review_candidates(alpaca(reordered, context=context(issuer_aliases=("Acme Corporation", "Acme"))))
    assert len(result.candidates) == 1
    assert result.candidates == other.candidates


def test_revision_changes_identity_and_cannot_backdate_availability() -> None:
    original = alpaca(record())
    revised = alpaca(record(updated_at="2024-07-19T16:04:00Z", headline="Acme reports Q2 2024 revenue."))
    assert revised.published_at_utc == original.published_at_utc
    assert revised.version_available_at_utc == pd.Timestamp("2024-07-19T16:04:00Z")
    assert revised.available_at_utc == context().first_seen_at_utc
    assert revised.source_version_sha256 != original.source_version_sha256
    original_id = owner.extract_review_candidates(original).candidates[0].candidate_id
    assert owner.extract_review_candidates(revised).candidates[0].candidate_id != original_id
    with pytest.raises(DataReadinessError, match="observation precedes"):
        alpaca(record(updated_at="2024-07-19T16:06:00Z"))


def test_identity_clock_can_delay_but_not_advance_content_availability() -> None:
    later = pd.Timestamp("2024-07-19T16:10:00Z")
    evidence = alpaca(record(), context=context(identity_available_at_utc=later))
    assert evidence.available_at_utc == later


@pytest.mark.parametrize("source", ["alpaca", "sec"])
@pytest.mark.parametrize("mutation", ["available_backdated", "version_before_publication", "publication_after_version",
    "first_seen_before_version", "identity_after_availability"])
def test_extraction_rejects_inconsistent_mutated_evidence_clocks(source: str, mutation: str) -> None:
    evidence = alpaca(record()) if source == "alpaca" else sec(b"<p>Acme reports Q2 2024 earnings.</p>")
    if mutation == "available_backdated":
        evidence = replace(evidence, available_at_utc=evidence.published_at_utc - pd.Timedelta(seconds=1))
    elif mutation == "version_before_publication":
        evidence = replace(evidence, version_available_at_utc=evidence.published_at_utc - pd.Timedelta(seconds=1))
    elif mutation == "publication_after_version":
        evidence = replace(evidence, published_at_utc=evidence.version_available_at_utc + pd.Timedelta(seconds=1))
    elif mutation == "first_seen_before_version":
        changed_context = replace(evidence.context, first_seen_at_utc=evidence.version_available_at_utc - pd.Timedelta(seconds=1))
        evidence = replace(evidence, context=changed_context)
    else:
        changed_context = replace(evidence.context, identity_available_at_utc=evidence.available_at_utc + pd.Timedelta(seconds=1))
        evidence = replace(evidence, context=changed_context)
    with pytest.raises(DataReadinessError):
        owner.extract_review_candidates(evidence)


@pytest.mark.parametrize("source", ["alpaca", "sec"])
def test_observed_and_historical_proxy_are_distinct_and_proxy_rejected_live(source: str) -> None:
    def adapter(**kwargs: Any) -> owner.IssuerContentEvidence:
        if source == "alpaca":
            return alpaca(record(), **kwargs)
        return sec(b"<p>Acme reports Q2 2024 earnings.</p>", **kwargs)

    observed = adapter(purpose="live_construction")
    historical_observed = adapter(purpose="historical_research")
    assert owner.extract_review_candidates(observed) == owner.extract_review_candidates(historical_observed)
    proxy = adapter(context=context(availability_semantics="historical_proxy"))
    assert observed.available_at_utc == context().first_seen_at_utc
    assert proxy.available_at_utc == proxy.version_available_at_utc < observed.available_at_utc
    with pytest.raises(DataReadinessError, match="live content cannot"):
        adapter(context=context(availability_semantics="historical_proxy"), purpose="live_construction")


def test_sec_acceptance_authority_and_encoding_are_bound_to_extraction_policy() -> None:
    body = b"<p>Acme reports Q2 2024 earnings.</p>"
    original = sec(body)
    different_clock_authority = sec(body, acceptance_clock_authority_sha256="d" * 64)
    different_encoding = sec(body, encoding="ascii")
    assert len({item.extraction_policy_sha256 for item in (original, different_clock_authority, different_encoding)}) == 3
    assert all(item.text == original.text for item in (different_clock_authority, different_encoding))


def test_tampered_normalized_text_and_out_of_bounds_spans_rejected() -> None:
    evidence = alpaca(record())
    with pytest.raises(DataReadinessError, match="text hash"):
        owner.extract_review_candidates(replace(evidence, text=evidence.text + " changed"))
    segment = replace(evidence.text_segments[0], end=len(evidence.text) + 1)
    with pytest.raises(DataReadinessError, match="segment"):
        owner.extract_review_candidates(replace(evidence, text_segments=(segment,)))


@pytest.mark.parametrize("body", [b"%PDF-1.7", b"\xff"])
def test_pdf_and_invalid_declared_encoding_are_not_silently_decoded(body: bytes) -> None:
    with pytest.raises(DataReadinessError):
        sec(body)
