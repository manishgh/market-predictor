"""Frozen decision-level reaction projection; source references are not admission."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.issuer_reaction import REACTION_COLUMNS, IssuerReactionSources, issuer_reaction_contract_sha256
from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_COLUMNS

ISSUER_REACTION_PROFILE: Final = "technical_relationships_issuer_reaction"
QUALIFIED_EVENT_COLUMNS: Final = (
    "security_id", "ticker", "source_family", "event_id", "event_version_sha256",
    "event_available_at_utc", "identity_available_at_utc", "event_family",
    "qualification_status", "qualification_authority_sha256", "duplicate_group_id",
)
REACTION_COVERAGE_COLUMNS: Final = ("decision_id", "coverage_status", "available_at_utc")
LOOKBACK: Final = "3D"


def issuer_reaction_profile_sha256(baseline_columns: Sequence[str], sources: IssuerReactionSources) -> str:
    names = tuple(baseline_columns)
    if (len(names) != 124 or len(set(names)) != 124 or names[-4:] != RETURN_RELATIONSHIP_COLUMNS
            or set(names).intersection(REACTION_COLUMNS)):
        raise ValueError("issuer reaction requires the unchanged ordered 124-column relationship parent")
    return json_sha256({
        "schema": "market_predictor.issuer_reaction_profile", "profile": ISSUER_REACTION_PROFILE,
        "baseline_columns": names, "additional_columns": REACTION_COLUMNS,
        "event_columns": QUALIFIED_EVENT_COLUMNS, "coverage_columns": REACTION_COVERAGE_COLUMNS,
        "lookback": "event_availability_in_(decision_minus_72_hours,decision]",
        "versions": "latest_available_source_event_version_before_qualification_filter_no_resurrection",
        "version_tie": "event_version_sha256_ascending_first",
        "duplicates": "explicit_group_earliest_qualified_event_availability_then_source_event_version_ascending",
        "selection": "latest_distinct_qualified_event_availability_then_source_event_version_ascending",
        "selection_precedes": "bar_completeness_and_all_measured_values",
        "missingness": "no_event_and_unknown_coverage_null_never_zero_no_fallback",
        "coverage": "diagnostic_only_known_requires_caller_verified_complete_query_and_qualification_denominator",
        "output": "all_parent_rows_and_columns_preserved_except_profile_identity_two_raw_float32_additions",
        "measurement_contract_sha256": issuer_reaction_contract_sha256(sources),
        "admission": "not_established_by_construction",
    })
