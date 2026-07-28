from pathlib import Path

import pytest
from pydantic import ValidationError

from market_predictor.primary_v2.contracts import INTRADAY_V2_ID, SWING_V2_ID
from market_predictor.primary_v2.failure_attribution_contracts import (
    INTRADAY_DIMENSIONS,
    SWING_DIMENSIONS,
    FailureAttributionConfig,
    load_failure_attribution_config,
)

POLICY = Path("configs/primary_v2_failure_attribution.toml")


def test_failure_attribution_policy_freezes_dimensions_and_scopes() -> None:
    config = load_failure_attribution_config(POLICY)

    assert config.validation_scopes == ("walk_forward", "ticker_holdout")
    assert config.swing.strategy_id == SWING_V2_ID
    assert config.swing.dimensions == SWING_DIMENSIONS
    assert config.swing.non_overlapping_phases == 5
    assert config.intraday.strategy_id == INTRADAY_V2_ID
    assert config.intraday.dimensions == INTRADAY_DIMENSIONS
    assert config.intraday.non_overlapping_phases == 1


def test_failure_attribution_policy_freezes_replicated_viability_gates() -> None:
    config = load_failure_attribution_config(POLICY)

    assert config.minimum_rows_per_scope >= 200
    assert config.minimum_sessions_per_scope >= 60
    assert config.minimum_average_net_return_ci_low == 0
    assert config.minimum_average_excess_return_vs_spy_ci_low == 0
    assert config.minimum_profit_factor >= 1.05
    assert config.maximum_drawdown <= 0.20
    assert config.minimum_stamped_round_trip_cost_bps >= 10
    assert config.maximum_process_memory_gib <= 4


def test_failure_attribution_rejects_dimension_search_expansion() -> None:
    config = load_failure_attribution_config(POLICY)
    raw = config.model_dump()
    raw["swing"]["dimensions"] = (*SWING_DIMENSIONS, "ticker")

    with pytest.raises(ValidationError, match="dimensions are not frozen"):
        FailureAttributionConfig.model_validate(raw)


def test_failure_attribution_rejects_weakened_evidence_gate() -> None:
    config = load_failure_attribution_config(POLICY)
    raw = config.model_dump()
    raw["minimum_rows_per_scope"] = 100

    with pytest.raises(ValidationError):
        FailureAttributionConfig.model_validate(raw)


def test_failure_attribution_rejects_overlapping_swing_evaluation() -> None:
    config = load_failure_attribution_config(POLICY)
    raw = config.model_dump()
    raw["swing"]["non_overlapping_phases"] = 1

    with pytest.raises(ValidationError, match="requires five phases"):
        FailureAttributionConfig.model_validate(raw)
