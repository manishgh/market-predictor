"""Frozen incremental return features; lineage references are not source admission."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Final, Literal

from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Identifier, Sha256

RETURN_RELATIONSHIP_PROFILE: Final = "technical_relationships"
RETURN_RELATIONSHIP_COLUMNS: Final = (
    "momentum_126_sessions_excluding_recent_21",
    "momentum_252_sessions_excluding_recent_21",
    "lagged_volume_weighted_stock_spy_response",
    "stock_spy_5_session_excess_times_spy_60_session_return",
)
RETURN_RELATIONSHIP_FORMULAS: Final = (
    "stock_close[t-21] / stock_close[t-126] - 1",
    "stock_close[t-21] / stock_close[t-252] - 1",
    "(stock_volume[t] / mean(stock_volume[t-20:t-1])) * "
    "((stock_close[t] / stock_open[t] - 1) - (spy_close[t] / spy_open[t] - 1))",
    "((stock_close[t] / stock_close[t-5] - 1) - (spy_close[t] / spy_close[t-5] - 1)) * "
    "(spy_close[t] / spy_close[t-60] - 1)",
)
RETURN_RELATIONSHIP_SESSION_POSITIONS: Final = (127, 253, 21, 61)


class ReturnRelationshipSources(HoldingContract):
    """Explicit caller-verified source bindings, never invented observation clocks.

    All three authorities must refer to a compatible price basis/vintage. The
    builder validates physical feed/adjustment fields, not authority-file replay.
    Historical proxy semantics may support separately admitted research fitting.
    """

    baseline_authority_sha256: Sha256
    stock_authority_sha256: Sha256
    spy_authority_sha256: Sha256
    availability_semantics: Literal["observed", "historical_proxy"]
    availability_policy_id: Identifier
    availability_policy_sha256: Sha256
    price_adjustment: Literal["raw", "split", "dividend", "all"]
    price_basis_and_vintage_id: Identifier


def return_relationship_profile_sha256(
    baseline_columns: Sequence[str], sources: ReturnRelationshipSources,
) -> str:
    """Bind ordering, formulas, missingness, clocks, and source semantics together."""
    baseline = tuple(baseline_columns)
    if len(baseline) != 120 or len(set(baseline)) != 120 or set(baseline).intersection(RETURN_RELATIONSHIP_COLUMNS):
        raise ValueError("return relationships require 120 distinct unchanged baseline columns")
    return json_sha256({
        "schema": "market_predictor.return_relationship_profile.v1",
        "profile": RETURN_RELATIONSHIP_PROFILE,
        "baseline_columns": baseline,
        "additional_columns": RETURN_RELATIONSHIP_COLUMNS,
        "formulas": RETURN_RELATIONSHIP_FORMULAS,
        "minimum_session_positions": RETURN_RELATIONSHIP_SESSION_POSITIONS,
        "calendar": "XNYS",
        "decision_start": "2019-07-09",
        "baseline_warmup_positions": 250,
        "long_momentum_warmup": "253_positions_required_never_shorten_to_250",
        "history": "complete_required_session_windows_no_fill_no_surviving_row_shifts",
        "volume_baseline": "previous_20_sessions_excluding_current",
        "availability": "maximum_consumed_bar_clock_at_or_before_decision",
        "missingness": "null_and_explicit_reason_no_population_filtering",
        "output": "baseline_unchanged_then_four_raw_columns_no_new_peer_transform",
        "additional_dtype": "float32_overflow_unavailable_never_clipped",
        "numeric_inputs": "real_numbers_only_no_boolean_or_string_coercion",
        "physical_bar_source": "alpaca_1d_sip_matching_adjustment",
        "physical_availability": "observed_requires_ingestion_bound_or_explicit_market_interval_close_research_proxy",
        "sources": sources.model_dump(mode="json"),
    })
