from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.readiness import (
    PredictionDataReadinessConfig,
    load_prediction_data_readiness_config,
)

POLICY_PATH = Path("configs/prediction_data_readiness.toml")


def _valid_policy() -> dict[str, Any]:
    return load_prediction_data_readiness_config(POLICY_PATH).model_dump()


def test_prediction_data_readiness_policy_preserves_frozen_contract() -> None:
    config = load_prediction_data_readiness_config(POLICY_PATH)

    assert config.schema_version == "edge_rebuild.readiness.v2"
    assert config.required_price_feed == "sip"
    assert config.required_adjustment == "all"
    assert config.target_history_sessions == 1_250
    assert config.maximum_process_memory_gib == 4.0
    assert config.memory_guard_headroom_gib == 0.75
    assert config.swing.strategy_id == "SWING.SECTOR_RESIDUAL_MOMENTUM.10D.V1"
    assert config.swing.proposed_horizon_sessions == 10
    assert config.swing.non_overlapping_phases == 10
    assert config.swing.minimum_valid_sessions == 1_000
    assert config.swing.minimum_sessions_per_phase == 60
    assert config.swing.minimum_daily_warmup_bars == 250
    assert config.intraday.strategy_id == "INTRADAY.VWAP_EXHAUSTION_REVERSAL.30M.V1"
    assert config.intraday.proxy_strategy_id == "INTRADAY.VWAP_REVERSION.30M.V1"
    assert config.intraday.proposed_horizon_minutes == 30
    assert config.intraday.minimum_causal_sessions == 750
    assert config.intraday.required_purged_folds == 4
    assert config.intraday.minimum_test_sessions_per_fold == 60
    assert config.intraday.required_timeframe == "1Min"
    assert "first_observed_at_utc" in config.catalyst.required_fields


def test_prediction_data_readiness_policy_is_frozen_and_forbids_extra_fields() -> None:
    config = load_prediction_data_readiness_config(POLICY_PATH)
    frozen_field = "required_price_feed"

    with pytest.raises(ValidationError, match="frozen"):
        setattr(config, frozen_field, "iex")

    raw = config.model_dump()
    raw["unexpected"] = True
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        PredictionDataReadinessConfig.model_validate(raw)


@pytest.mark.parametrize(
    ("section", "field", "value", "message"),
    [
        (None, "schema_version", "prediction.readiness.v3", "unsupported edge-rebuild readiness schema"),
        (None, "required_price_feed", "iex", "volume-dependent readiness requires SIP"),
        (None, "required_adjustment", "raw", "readiness adjustment identity must be all"),
        ("swing", "strategy_id", "OTHER", "swing readiness strategy ID is not frozen"),
        ("swing", "proposed_horizon_sessions", 9, "swing readiness horizon must be ten sessions"),
        ("swing", "non_overlapping_phases", 9, "swing phase count must equal its overlapping horizon"),
        ("intraday", "strategy_id", "OTHER", "intraday readiness strategy ID is not frozen"),
        ("intraday", "proxy_strategy_id", "OTHER", "intraday readiness proxy strategy ID is not frozen"),
        ("intraday", "proposed_horizon_minutes", 60, "intraday readiness horizon must be thirty minutes"),
        ("intraday", "required_timeframe", "5Min", "intraday exact-path source must use one-minute bars"),
    ],
)
def test_prediction_data_readiness_policy_rejects_non_frozen_values(
    section: str | None,
    field: str,
    value: object,
    message: str,
) -> None:
    raw = _valid_policy()
    target = raw if section is None else raw[section]
    target[field] = value

    with pytest.raises(ValidationError, match=message):
        PredictionDataReadinessConfig.model_validate(raw)


def test_prediction_data_readiness_policy_preserves_normalized_feed_acceptance() -> None:
    raw = _valid_policy()
    raw["required_price_feed"] = " SIP "
    raw["required_adjustment"] = " ALL "

    config = PredictionDataReadinessConfig.model_validate(raw)

    assert config.required_price_feed == " SIP "
    assert config.required_adjustment == " ALL "


def test_prediction_data_readiness_policy_rejects_intraday_fold_overcommitment() -> None:
    raw = _valid_policy()
    raw["intraday"]["required_purged_folds"] = 10
    raw["intraday"]["minimum_test_sessions_per_fold"] = 76

    with pytest.raises(ValidationError, match="intraday fold capacity exceeds minimum session history"):
        PredictionDataReadinessConfig.model_validate(raw)


def test_prediction_data_readiness_policy_rejects_memory_headroom_at_budget() -> None:
    raw = _valid_policy()
    raw["maximum_process_memory_gib"] = 1.0
    raw["memory_guard_headroom_gib"] = 1.0

    with pytest.raises(ValidationError, match="memory guard headroom must be below the hard budget"):
        PredictionDataReadinessConfig.model_validate(raw)


@pytest.mark.parametrize(
    "field",
    [
        "required_source_families",
        "required_relation_channels",
        "required_fields",
        "research_availability_policies",
        "promotion_availability_policies",
    ],
)
def test_prediction_data_readiness_policy_rejects_empty_catalyst_collections(field: str) -> None:
    raw = _valid_policy()
    raw["catalyst"][field] = []

    with pytest.raises(ValidationError, match=rf"{field} must contain non-empty values"):
        PredictionDataReadinessConfig.model_validate(raw)


@pytest.mark.parametrize(
    "values",
    [
        ["alpaca", "ALPACA"],
        ["alpaca", " alpaca "],
    ],
)
def test_prediction_data_readiness_policy_rejects_normalized_catalyst_duplicates(values: list[str]) -> None:
    raw = _valid_policy()
    raw["catalyst"]["required_source_families"] = values

    with pytest.raises(ValidationError, match="required_source_families must contain unique values"):
        PredictionDataReadinessConfig.model_validate(raw)


def test_prediction_data_readiness_policy_requires_first_observed_evidence() -> None:
    raw = _valid_policy()
    raw["catalyst"]["required_fields"] = [value for value in raw["catalyst"]["required_fields"] if value != "first_observed_at_utc"]

    with pytest.raises(ValidationError, match="catalyst readiness must require first-observed evidence"):
        PredictionDataReadinessConfig.model_validate(raw)


def test_prediction_data_readiness_policy_hash_is_deterministic_and_content_bound() -> None:
    config = load_prediction_data_readiness_config(POLICY_PATH)
    same_config = PredictionDataReadinessConfig.model_validate(config.model_dump())
    changed = config.model_dump()
    changed["target_history_sessions"] = 1_251

    assert len(config.sha256()) == 64
    assert config.sha256() == same_config.sha256()
    assert config.sha256() != PredictionDataReadinessConfig.model_validate(changed).sha256()


@pytest.mark.parametrize("contents", ["not = [valid", "schema_version = 'wrong'"])
def test_prediction_data_readiness_policy_loader_fails_closed(tmp_path: Path, contents: str) -> None:
    policy_path = tmp_path / "readiness.toml"
    policy_path.write_text(contents, encoding="utf-8")
    expected = "unreadable" if contents == "not = [valid" else "invalid"

    with pytest.raises(DataReadinessError, match=expected):
        load_prediction_data_readiness_config(policy_path)


def test_prediction_data_readiness_policy_loader_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(DataReadinessError, match="unreadable"):
        load_prediction_data_readiness_config(tmp_path / "missing.toml")
