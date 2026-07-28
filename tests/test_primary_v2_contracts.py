from pathlib import Path

import pytest
from pydantic import ValidationError

from market_predictor.primary_v2.contracts import (
    INTRADAY_V2_ID,
    PRIMARY_V2_STRATEGY_IDS,
    SWING_V2_ID,
    PrimaryV2ResearchConfig,
    load_primary_v2_research_config,
)

POLICY = Path("configs/primary_strategy_v2.toml")


def test_primary_v2_policy_freezes_human_readable_horizons() -> None:
    config = load_primary_v2_research_config(POLICY)

    assert set(config.strategies) == PRIMARY_V2_STRATEGY_IDS
    swing = config.strategies[SWING_V2_ID]
    assert swing.horizon_value == 5
    assert swing.horizon_unit == "exchange_sessions"
    assert "5 Trading Sessions" in swing.display_name
    intraday = config.strategies[INTRADAY_V2_ID]
    assert intraday.horizon_value == 30
    assert intraday.horizon_unit == "regular_session_minutes"
    assert "30 Regular-Session Minutes" in intraday.display_name


def test_primary_v2_policy_freezes_candidates_costs_and_memory() -> None:
    config = load_primary_v2_research_config(POLICY)

    assert config.quantiles == (0.10, 0.50, 0.90)
    assert config.maximum_process_memory_gib == 4.0
    assert config.maximum_drawdown == 0.20
    assert config.minimum_incremental_net_return_ci_low == 0.0
    assert config.minimum_q10_q90_interval_coverage == 0.65
    assert config.maximum_event_brier_score == 0.70
    for strategy in config.strategies.values():
        assert strategy.minimum_round_trip_cost_bps >= 10.0
        assert len(strategy.candidate_families) * len(strategy.selection_policies) <= 12


def test_primary_v2_required_columns_include_causal_and_path_fields() -> None:
    config = load_primary_v2_research_config(POLICY)

    swing = config.strategies[SWING_V2_ID].required_source_columns
    assert {"decision_time_utc", "entry_time_utc", "exit_time_utc", "label_available_at_utc"} <= swing
    intraday = config.strategies[INTRADAY_V2_ID].required_source_columns
    assert {"target_before_stop_30m", "stop_before_target_30m", "path_timeout_30m"} <= intraday
    assert "path_excess_return_30m_vs_spy" in intraday


def test_primary_v2_rejects_weakened_memory_or_horizon() -> None:
    config = load_primary_v2_research_config(POLICY)
    payload = config.model_dump(mode="python")
    payload["maximum_process_memory_gib"] = 5.0
    with pytest.raises(ValidationError):
        PrimaryV2ResearchConfig.model_validate(payload)

    payload = config.model_dump(mode="python")
    payload["strategies"][SWING_V2_ID]["horizon_unit"] = "regular_session_minutes"
    with pytest.raises(ValidationError):
        PrimaryV2ResearchConfig.model_validate(payload)
