"""Compose the frozen 120/124/126 transforms without granting admission."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from typing import Literal

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.contracts.issuer_reaction import IssuerReactionSources
from market_predictor.swing.contracts.return_feature_profiles import ReturnRelationshipSources
from market_predictor.swing.features.issuer_reaction_profile import build_issuer_reaction_profile
from market_predictor.swing.features.research_partition import ResearchFeaturePartition
from market_predictor.swing.features.return_relationships import build_return_relationship_profile


def build_return_candidate_features(
    *,
    baseline: ResearchFeaturePartition,
    baseline_availability_semantics: Literal["observed", "historical_proxy"],
    expected_decisions: pd.DataFrame,
    stock_bars: pd.DataFrame,
    spy_bars: pd.DataFrame,
    history_sessions: Sequence[date],
    qualified_events: pd.DataFrame,
    coverage: pd.DataFrame,
    contract: StrategyContract,
    relationship_sources: ReturnRelationshipSources,
    reaction_sources: IssuerReactionSources,
    purpose: Literal["historical_research", "live_construction"] = "historical_research",
) -> ResearchFeaturePartition:
    """Build candidate inputs through the same pure kernels for batch and live.

    The caller must independently verify baseline observation and every authority.
    A declaration, source hash or causal value clock alone does not establish that
    admission. The two baseline authorities may identify different parent stages;
    all shared physical bar, price and availability bindings must otherwise match.
    No imputation, model loading, selection or serving acceptance occurs here.
    """
    if baseline_availability_semantics not in ("observed", "historical_proxy"):
        raise DataReadinessError("candidate baseline requires explicit availability semantics")
    if purpose == "live_construction" and baseline_availability_semantics != "observed":
        raise DataReadinessError("live candidate baseline cannot consume historical proxy availability")
    relationship_record = relationship_sources.model_dump(mode="json")
    reaction_record = reaction_sources.model_dump(mode="json")
    reaction_bars = reaction_sources.bars.model_dump(mode="json")
    shared = {key: value for key, value in relationship_record.items() if key != "baseline_authority_sha256"}
    other = {key: value for key, value in reaction_bars.items() if key != "baseline_authority_sha256"}
    if shared != other:
        raise DataReadinessError("candidate components require identical stock/SPY price and availability authorities")

    relationships = build_return_relationship_profile(
        expected_decisions=expected_decisions,
        baseline=baseline,
        stock_bars=stock_bars,
        spy_bars=spy_bars,
        history_sessions=history_sessions,
        contract=contract,
        sources=relationship_sources,
        purpose=purpose,
    )
    result = build_issuer_reaction_profile(
        baseline=relationships,
        qualified_events=qualified_events,
        coverage=coverage,
        stock_bars=stock_bars,
        spy_bars=spy_bars,
        history_sessions=history_sessions,
        sources=reaction_sources,
        purpose=purpose,
    )
    lineage = {
        "relationship_sources_sha256": json_sha256(relationship_record),
        "reaction_sources_sha256": json_sha256(reaction_record),
        "relationship_profile_sha256": relationships.audit["profile_sha256"],
        "reaction_profile_sha256": result.audit["profile_sha256"],
        "relationship_baseline_authority_sha256": relationship_sources.baseline_authority_sha256,
        "reaction_baseline_authority_sha256": reaction_sources.bars.baseline_authority_sha256,
        "baseline_availability_semantics": baseline_availability_semantics,
    }
    audit = {
        **result.audit,
        "candidate_composition": lineage,
        "candidate_composition_sha256": json_sha256(lineage),
        "source_admission": "required_from_publishing_caller",
        "source_eligible": False,
        "training_eligible": False,
        "promotion_eligible": False,
        "serving_eligible": False,
        "economic_acceptance": False,
    }
    return replace(result, audit=audit)
