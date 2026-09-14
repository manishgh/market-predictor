"""Independent source pins for rebuilding initial-fit technical predictors."""
from __future__ import annotations

from typing import Literal

from market_predictor.swing.contracts.holding_accounting import HoldingContract
from market_predictor.swing.contracts.holding_materialization import SourcePin


class ResearchFeaturePolicy(HoldingContract):
    schema_version: Literal["market_predictor.corrected_research_features"]
    outcome_source_config: SourcePin
    strategy_contract: SourcePin
    parent_request: SourcePin
    parent_manifest: SourcePin
    parent_authority: SourcePin
    combined_manifest: SourcePin
    adjusted_plan_authority: SourcePin
    adjusted_archive_authority: SourcePin
