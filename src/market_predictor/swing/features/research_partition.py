"""Compose retained-population predictors and independent outcomes, without admission."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.features.panel import (
    CATALYST_RANKING_FEATURES,
    TECHNICAL_RANKING_FEATURES,
    swing_model_feature_columns,
)
from market_predictor.swing.features.research_join import (
    DECISION_KEYS,
    join_research_features_and_outcomes,
    rebuild_research_peer_features,
)

_CONTEXT = ("sector", "session_date_et", "primary_benchmark")
_FEATURE_STATE = ("feature_profile", "feature_eligible", "daily_bar_count")


@dataclass(frozen=True)
class ResearchFeaturePartition:
    """Calculated rows and ordered inputs; source and training admission are separate."""

    rows: pd.DataFrame
    model_columns: tuple[str, ...]
    availability_columns: Mapping[str, str]
    audit: Mapping[str, object]


def _context_matches(reference: pd.DataFrame, candidate: pd.DataFrame, name: str) -> None:
    required = (*DECISION_KEYS, *_CONTEXT)
    if (not reference.columns.is_unique or not candidate.columns.is_unique
            or not set(required).issubset(reference) or not set(required).issubset(candidate)):
        raise DataReadinessError(f"{name} lacks frozen decision context")
    if candidate.decision_id.duplicated().any() or reference.decision_id.duplicated().any():
        raise DataReadinessError(f"{name} duplicates a frozen decision")
    left = reference.set_index("decision_id")
    right = candidate.set_index("decision_id")
    if set(left.index) != set(right.index):
        raise DataReadinessError(f"{name} does not cover the complete frozen decision population")
    right = right.reindex(left.index)
    for column in (*DECISION_KEYS[1:], *_CONTEXT):
        expected, actual = left[column], right[column]
        if column == "decision_time_utc":
            expected = pd.to_datetime(expected, utc=True, errors="coerce")
            actual = pd.to_datetime(actual, utc=True, errors="coerce")
        elif column == "session_date_et":
            expected = pd.to_datetime(expected, errors="coerce").dt.date
            actual = pd.to_datetime(actual, errors="coerce").dt.date
        if expected.isna().any() or actual.isna().any() or not expected.eq(actual).all():
            raise DataReadinessError(f"{name} differs from frozen {column}")


def assemble_research_feature_partition(
    *,
    expected_decisions: pd.DataFrame,
    base_features: pd.DataFrame,
    outcomes: pd.DataFrame,
    contract: StrategyContract,
    retained_security_ids: frozenset[str],
    availability_columns: Mapping[str, str],
) -> ResearchFeaturePartition:
    """Validate the full population before peers, then join independently built labels.

    All three inputs must be independently source-verified by the publishing caller.
    This transformation does not certify provenance, completeness of news history,
    production eligibility or training readiness. It never selects rows by outcome.
    """
    if (expected_decisions.empty or not expected_decisions.columns.is_unique
            or not set(DECISION_KEYS).issubset(expected_decisions)):
        raise DataReadinessError("research partition requires frozen expected decisions")
    if not set(expected_decisions.security_id).issubset(retained_security_ids):
        raise DataReadinessError("frozen research population includes excluded identities")
    if not base_features.columns.is_unique or not set((*DECISION_KEYS, *_FEATURE_STATE)).issubset(base_features):
        raise DataReadinessError("research base features lack identity, profile or warm-up state")
    retained = base_features.loc[base_features.security_id.isin(retained_security_ids)].copy()
    _context_matches(expected_decisions, retained, "base features")
    _context_matches(expected_decisions, outcomes, "outcomes")
    if not set(_FEATURE_STATE).isdisjoint(outcomes.columns):
        raise DataReadinessError("outcomes cannot supply predictor eligibility, profile or warm-up")
    retained = retained.set_index("decision_id").loc[expected_decisions.decision_id].reset_index()
    profile = set(retained.feature_profile)
    if profile not in ({"technical_market"}, {"catalyst_full"}):
        raise DataReadinessError("research partition requires one supported feature profile")
    catalyst = profile == {"catalyst_full"}
    inputs = (*TECHNICAL_RANKING_FEATURES, *(CATALYST_RANKING_FEATURES if catalyst else ()))
    # Validate outcome identity before a subset could influence cross-sectional peers.
    join_research_features_and_outcomes(
        retained, outcomes, feature_columns=inputs, availability_columns=availability_columns,
        retained_security_ids=retained_security_ids,
    )
    peers, peer_clocks = rebuild_research_peer_features(
        retained, contract=contract, retained_security_ids=retained_security_ids,
        availability_columns=availability_columns,
    )
    model_columns = tuple(swing_model_feature_columns(contract=contract, catalyst=catalyst))
    all_clocks = {**availability_columns, **peer_clocks}
    columns = tuple(dict.fromkeys((*inputs, *model_columns)))
    joined = join_research_features_and_outcomes(
        peers, outcomes, feature_columns=columns, availability_columns=all_clocks,
        retained_security_ids=retained_security_ids,
    )
    for name in _FEATURE_STATE:
        joined[name] = peers[name].to_numpy()
    _context_matches(expected_decisions, joined, "joined features")
    predictor_complete = joined.loc[:, list(model_columns)].notna().all(axis=1)
    audit: dict[str, object] = {
        "rows": len(joined), "security_count": joined.security_id.nunique(),
        "feature_profile": next(iter(profile)), "model_feature_count": len(model_columns),
        "predictor_complete_rows": int(predictor_complete.sum()),
        "feature_eligible_rows": int(joined.feature_eligible.sum()),
        "population_preserved": True, "outcome_filtered_rows": 0,
        "source_admission": "required_from_publishing_caller",
        "training_eligible": False, "promotion_eligible": False,
    }
    return ResearchFeaturePartition(joined, model_columns, all_clocks, audit)
