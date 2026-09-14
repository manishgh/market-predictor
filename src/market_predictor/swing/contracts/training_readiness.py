"""Pinned scope for auditing, not authorizing, fixed-horizon research fitting."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from market_predictor.swing.contracts.holding_accounting import HoldingContract
from market_predictor.swing.contracts.holding_materialization import SourcePin


class TrainingReadinessPolicy(HoldingContract):
    schema_version: Literal["market_predictor.swing_training_readiness"]
    scope: Literal["initial_fit_fixed_horizon_diagnostics"]
    maximum_system_used_percent: float = Field(gt=0, le=90)
    publication: SourcePin
    saved_row_verification: SourcePin
    research_contract: SourcePin
    strategy_contract: SourcePin
    temporal_contract: SourcePin
