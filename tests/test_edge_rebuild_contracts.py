from pathlib import Path

import pytest
from pydantic import ValidationError

from market_predictor.edge_rebuild.contracts import (
    INTRADAY_STRATEGY_ID,
    READINESS_SCHEMA,
    SWING_STRATEGY_ID,
    EdgeRebuildReadinessConfig,
    load_edge_rebuild_readiness_config,
)
from market_predictor.v3.errors import DataReadinessError

POLICY_PATH = Path("configs/edge_rebuild_readiness.toml")


def test_edge_rebuild_readiness_contract_is_frozen() -> None:
    config = load_edge_rebuild_readiness_config(POLICY_PATH)

    assert config.schema_version == READINESS_SCHEMA
    assert config.swing.strategy_id == SWING_STRATEGY_ID
    assert config.swing.proposed_horizon_sessions == 10
    assert config.swing.non_overlapping_phases == 10
    assert config.swing.minimum_valid_sessions == 1_000
    assert config.intraday.strategy_id == INTRADAY_STRATEGY_ID
    assert config.intraday.minimum_causal_sessions == 750
    assert config.intraday.required_purged_folds == 4
    assert config.required_price_feed == "sip"
    assert config.required_adjustment == "all"
    assert config.maximum_process_memory_gib == 5.0
    assert len(config.sha256()) == 64


def test_readiness_contract_rejects_weakened_session_and_feed_gates() -> None:
    raw = load_edge_rebuild_readiness_config(POLICY_PATH).model_dump()
    raw["required_price_feed"] = "iex"
    raw["intraday"]["minimum_causal_sessions"] = 500

    with pytest.raises(ValidationError):
        EdgeRebuildReadinessConfig.model_validate(raw)


def test_readiness_contract_rejects_phase_horizon_mismatch() -> None:
    raw = load_edge_rebuild_readiness_config(POLICY_PATH).model_dump()
    raw["swing"]["non_overlapping_phases"] = 5

    with pytest.raises(ValidationError, match="phase count"):
        EdgeRebuildReadinessConfig.model_validate(raw)


def test_readiness_contract_requires_first_observed_catalyst_evidence() -> None:
    raw = load_edge_rebuild_readiness_config(POLICY_PATH).model_dump()
    raw["catalyst"]["required_fields"] = [
        value
        for value in raw["catalyst"]["required_fields"]
        if value != "first_observed_at_utc"
    ]

    with pytest.raises(ValidationError, match="first-observed"):
        EdgeRebuildReadinessConfig.model_validate(raw)


def test_unreadable_readiness_policy_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(DataReadinessError, match="unreadable"):
        load_edge_rebuild_readiness_config(tmp_path / "missing.toml")
