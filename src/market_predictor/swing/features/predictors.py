"""Feature-only composition for corrected history and live predictor construction."""
from __future__ import annotations

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.feature_pipeline import FeaturePipeline
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.dataset import build_swing_feature_history
from market_predictor.swing.features.panel import TECHNICAL_RANKING_FEATURES, swing_dataset_config
from market_predictor.swing.features.pipeline import SetupComponentsStep, TechnicalRelationshipsStep


def build_swing_predictor_history(
    decisions: pd.DataFrame, benchmark_bars: pd.DataFrame, *, contract: StrategyContract,
) -> pd.DataFrame:
    """Use the existing indicator pipeline without evaluating any future outcomes.

    Inputs must be canonical, identity-corrected, consistently adjusted histories
    including warm-up. The caller supplies their verified lineage and handles sparse
    source gaps. Peer transforms and catalyst joins happen afterward on the complete
    retained population. This function cannot certify source availability.
    """
    forbidden = [name for name in decisions if name.startswith(("future_", "managed_", "barrier_", "forward_"))]
    if forbidden:
        raise DataReadinessError("predictor history cannot consume outcome columns")
    config = swing_dataset_config(contract)
    features, benchmarks = build_swing_feature_history(
        decisions, benchmark_bars, config=config, defer_cross_sectional=True,
    )
    rows = FeaturePipeline([SetupComponentsStep(benchmarks), TechnicalRelationshipsStep(contract)]).transform(features)
    if not set(TECHNICAL_RANKING_FEATURES).issubset(rows):
        raise DataReadinessError("predictor history did not produce the declared technical inputs")
    if any(name.startswith(("future_", "managed_", "barrier_", "forward_")) for name in rows):
        raise DataReadinessError("predictor history unexpectedly generated outcome columns")
    return rows
