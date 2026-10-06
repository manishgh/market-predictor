"""Frozen fresh-review frame and blinded document identities; no qualification."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Self

from pydantic import Field, model_validator

from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin

ORIGINAL_POPULATION = "3c59845761740dc1a07a2f994cceaba53354c6f9486f4baa634ebd422a6a84ce"
CANDIDATE_DERIVATIVE = "9c5f55807e9614c2e64d85902130e72c1dafdce73ba5f101f783daa9ac2f82d0"
ORIGINAL_SAMPLE = "f19f46917b1d629bfd55515556b5732b7b4f45cfb7669b1e0901f1a0b59d25c1"
ORIGINAL_REVIEWERS = frozenset({
    "17132bdcc38ca19738fd00b9e8a0f576f55e972fa20b4fd74a792918d9fc8922",
    "cfa9bea062ca443deed419593c1482a9b634bcb6f47797d770bc6219ade8342d",
})
DEVELOPMENT_SOURCES = (
    ("sec", "0001193125-21-011422/5"), ("sec", "0001193125-20-303343/2"),
    ("sec", "0001002910-21-000112/2"), ("sec", "0001104659-22-111755/2"),
    ("alpaca", "14199104"),
)
FRAME_POLICY = {
    "schema": "market_predictor.issuer_review_frame_policy",
    "population": "unchanged_readable_attributable_retained_announcement_clusters",
    "exclusions": "union_all_prior_sample_clusters_all_roles_families_and_all_development_source_clusters",
    "development_sources": DEVELOPMENT_SOURCES,
    "expansion": "original_cluster_all_documents_versions_revisions_query_copies_no_identity_merge",
    "sampling": "existing_policy_seed42_on_F_equals_U_minus_D_actual_F_N_over_n",
    "excluded_inclusion_probability": 0,
    "excluded_disposition": "development_inspected_not_newly_qualified",
    "unknowns": "complete_original_versions_records_retained_no_invented_negatives",
    "recall_scope": "fresh_frame_only_full_population_point_estimate_unmeasured",
}
DOCUMENT_POLICY = {
    "schema": "market_predictor.issuer_review_document_policy",
    "inventory": "all_original_cluster_versions_and_occurrences_including_metadata_only",
    "text": "exact_saved_normalized_unicode_no_reextraction_or_normalization",
    "offsets": "half_open_unicode_code_points_not_utf8_bytes_or_utf16_units",
    "omissions": "original_physical_unavailability_future_or_unknown_identity_text_null_keep_known_hash",
    "clocks": "original_published_version_event_identity_first_seen_separate_no_invention",
    "blindness": "no_candidate_ids_roles_spans_rules_reasons_dispositions_sample_role_old_labels_or_outcomes",
    "assessment": "one_explicit_assessment_per_exact_version_key_complete_text_inspection_required_for_negative",
}
CORRESPONDENCE_POLICY = {
    "schema": "market_predictor.issuer_review_correspondence_policy",
    "identity": "same_complete_version_key_text_hash_security_event_family",
    "roles": ("issuer", "action", "result", "fiscal_period"),
    "intervals": "four_exact_nonempty_half_open_role_intervals_and_exact_quotes",
    "anchor": "annotation_anchor_within_candidate_statement_and_contains_all_candidate_roles",
    "selection": "exactly_one_structural_match_among_all_annotations_before_verdicts",
    "verdicts": "all_five_true_in_each_independent_reviewer_unique_matching_annotation",
    "verdict_fields": ("family_present", "issuer_correct", "announced_or_reported", "explicit_fiscal_period", "action_supported"),
    "unresolved": "candidate_unresolved_missing_conflicting_or_unsupported_roles_cannot_be_rescued",
    "values": "action_quote_casefold_and_exact_fiscal_period_quote_no_inferred_values",
    "duplicates": "duplicate_annotation_ids_or_identical_payloads_malformed",
    "cluster_success": "every_distinct_candidate_supported_in_its_own_document_version_by_both_reviewers",
    "implemented_by_packet_slice": False,
}
CLOSED = dict(training_eligible=False, serving_eligible=False, promotion_eligible=False,
              qualification_established=False, economic_eligible=False)


def policy_identities() -> dict[str, str]:
    return {"frame_policy_sha256": json_sha256(FRAME_POLICY),
            "document_policy_sha256": json_sha256(DOCUMENT_POLICY),
            "correspondence_policy_sha256": json_sha256(CORRESPONDENCE_POLICY)}


class IssuerReviewPacketConfig(HoldingContract):
    schema_version: Literal["market_predictor.issuer_review_packet_config"] = Field(alias="schema")
    original_population: SourcePin
    candidate_derivative: SourcePin
    original_sample: SourcePin
    original_reviewers: tuple[SourcePin, SourcePin]

    @model_validator(mode="after")
    def exact_sources(self) -> Self:
        if (self.original_population.sha256 != ORIGINAL_POPULATION
                or self.candidate_derivative.sha256 != CANDIDATE_DERIVATIVE
                or self.original_sample.sha256 != ORIGINAL_SAMPLE
                or {pin.sha256 for pin in self.original_reviewers} != ORIGINAL_REVIEWERS
                or self.original_reviewers[0].path == self.original_reviewers[1].path):
            raise ValueError("packet config requires the frozen original population, derivative, sample and two reviews")
        return self


class PacketVersion(HoldingContract):
    version_id: Sha256
    cluster_id: Sha256
    security_id: str | None
    source_family: Literal["sec", "alpaca"]
    source_id: str
    source_version_sha256: Sha256 | None
    content_json_sha256: Sha256
    metadata_sha256: Sha256
    text_sha256: Sha256 | None
    ticker: str | None
    published_at_utc: str | None
    version_available_at_utc: str | None
    event_available_at_utc: str | None
    identity_available_at_utc: str | None
    first_seen_at_utc: str | None
    identity_authority_sha256: Sha256 | None
    availability_semantics: Literal["historical_proxy"]
    source_locator: str
    content_kind: str
    encoding: str | None
    alias_proof: dict[str, Any] | None
    physical_unavailable_reasons: tuple[str, ...]
    display_disposition: Literal["readable_source_text", "metadata_only"]
    omission_reasons: tuple[str, ...]
    text: str | None


@dataclass(frozen=True)
class VerifiedIssuerReviewPackets:
    publication: SourcePin
    request: dict[str, Any]
    counts: dict[str, Any]
    source_files: dict[str, str]
    frame_sha256: str
    exclusions_sha256: str
    sample_sha256: str
