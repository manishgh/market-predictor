"""Pinned scope for auditing, not authorizing, fixed-horizon research fitting."""
from __future__ import annotations

from typing import Literal, Self

from pydantic import Field, model_validator

from market_predictor.swing.contracts.holding_accounting import HoldingContract
from market_predictor.swing.contracts.holding_materialization import SourcePin


class TrainingReadinessPolicy(HoldingContract):
    schema_version: Literal["market_predictor.swing_training_readiness"]
    scope: Literal["initial_fit_fixed_horizon_diagnostics"]
    published_profile: Literal[
        "technical_market", "technical_relationships", "technical_relationships_issuer_reaction", "technical_relationships_news"
    ] = "technical_market"
    maximum_system_used_percent: float = Field(gt=0, le=91)
    publication: SourcePin
    saved_row_verification: SourcePin
    research_contract: SourcePin
    strategy_contract: SourcePin
    temporal_contract: SourcePin

    @model_validator(mode="after")
    def profile_memory_policy(self) -> Self:
        if self.published_profile == "technical_relationships_news":
            if self.maximum_system_used_percent != 91:
                raise ValueError("news readiness requires the frozen 91 percent runtime policy")
        elif self.maximum_system_used_percent > 90:
            raise ValueError("existing readiness profiles retain their maximum 90 percent policy")
        return self
