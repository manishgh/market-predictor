"""Matched research profiles with independently constructed, nullable outcomes."""
from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.features.catalyst_aggregates import build_swing_ablation_rows
from market_predictor.swing.features.catalyst_decision_authority import CatalystDecisionAuthority
from market_predictor.swing.features.panel import CATALYST_RANKING_FEATURES
from market_predictor.swing.features.research_join import DECISION_KEYS
from market_predictor.swing.features.research_partition import ResearchFeaturePartition, assemble_research_feature_partition


def assemble_research_ablations(*, expected_decisions: pd.DataFrame, technical: pd.DataFrame,
    outcomes: pd.DataFrame, catalyst: CatalystDecisionAuthority, contract: StrategyContract,
    retained_security_ids: frozenset[str], technical_availability: Mapping[str, str],
) -> dict[str, ResearchFeaturePartition]:
    """Retain every decision; unknown news is not zero or a cohort exclusion.

    These are matched-population ablations, not a training or promotion decision.
    Aggregate clocks are the validated authority's as-of cutoffs, never asserted
    provider reception times. Raw-event publication clocks remain in the lineage.
    """
    evidence = catalyst.decisions
    if (not set(DECISION_KEYS).issubset(evidence) or evidence.decision_id.duplicated().any()
            or not set(evidence.decision_id).issubset(expected_decisions.decision_id)):
        raise DataReadinessError("catalyst authority contains foreign or duplicate monthly decisions")
    if "label_eligible" in technical:
        raise DataReadinessError("technical inputs cannot supply outcome eligibility")
    profiles = build_swing_ablation_rows(technical.assign(label_eligible=False), catalyst)
    result: dict[str, ResearchFeaturePartition] = {}
    for profile, frame in profiles.items():
        availability = dict(technical_availability)
        if profile == "catalyst_full":
            frame["catalyst_aggregate_asof_utc"] = frame.decision_time_utc
            availability.update({name: "catalyst_aggregate_asof_utc" for name in CATALYST_RANKING_FEATURES})
        partition = assemble_research_feature_partition(expected_decisions=expected_decisions,
            base_features=frame, outcomes=outcomes, contract=contract,
            retained_security_ids=retained_security_ids, availability_columns=availability)
        # Preserve source and warmup diagnostics without making them model inputs.
        diagnostics = [name for name in frame if name.startswith(("source_coverage_known_", "catalyst_source_complete_"))]
        diagnostics += [name for name in ("catalyst_required_source_complete", "ablation_population_eligible",
            "pre_catalyst_feature_eligible", "technical_missing_reasons", "parent_decision_id") if name in frame]
        for name in diagnostics:
            if name not in partition.rows:
                partition.rows[name] = frame.set_index("decision_id").loc[partition.rows.decision_id, name].to_numpy()
        result[profile] = partition
    return result
