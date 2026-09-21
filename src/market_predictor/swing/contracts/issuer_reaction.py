"""Source-bound post-event measurements, not an admitted training profile."""
from __future__ import annotations

from typing import Final, Literal

from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Sha256
from market_predictor.swing.contracts.return_feature_profiles import ReturnRelationshipSources

REACTION_COLUMNS: Final = ("reaction_stock_minus_spy_oc", "reaction_volume_ratio_20")
REACTION_EVENT_COLUMNS: Final = (
    "decision_id", "security_id", "ticker", "event_id", "event_version_sha256",
    "event_available_at_utc", "identity_available_at_utc", "decision_time_utc",
)


class IssuerReactionSources(HoldingContract):
    """Caller-verified authorities; references alone never establish admission."""

    bars: ReturnRelationshipSources
    event_authority_sha256: Sha256
    identity_authority_sha256: Sha256
    event_availability_semantics: Literal["observed", "historical_proxy"]
    identity_availability_semantics: Literal["observed", "historical_proxy"]
    event_availability_policy_sha256: Sha256
    identity_availability_policy_sha256: Sha256


def issuer_reaction_contract_sha256(sources: IssuerReactionSources) -> str:
    return json_sha256({
        "schema": "market_predictor.issuer_reaction_measurements",
        "columns": REACTION_COLUMNS,
        "event_columns": REACTION_EVENT_COLUMNS,
        "calendar": "XNYS",
        "session_selection": "first_session_open_strictly_after_event_availability_before_inspecting_bars",
        "price_formula": "(stock_close/stock_open-1)-(spy_close/spy_open-1)",
        "volume_formula": "stock_volume/mean(previous_20_exchange_session_volumes)",
        "availability": "max(event_clock,identity_clock,consumed_bar_available_at_utc)<=decision",
        "bar_clock_validation": "all_physical_clocks_required_observed_ingestion_bounded_by_availability",
        "historical_proxy_ingestion": "retrospective_ingestion_not_a_historical_feature_clock",
        "missingness": "independent_null_values_and_reasons_never_substitute_sessions",
        "numeric_output": "float32_with_explicit_overflow_missingness",
        "clock_output": "datetime64[ns, UTC]",
        "source_admission": "caller_required_not_established_by_measurement",
        "content_qualification": "not_established",
        "causal_effect_identification": "not_claimed",
        "sources": sources.model_dump(mode="json"),
    })
