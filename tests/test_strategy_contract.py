from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest

from market_predictor.edge_rebuild.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)
from market_predictor.v3.errors import DataReadinessError

CONTRACT_PATH = Path("configs/edge_rebuild_strategy_contract.toml")


def _raw() -> dict[str, Any]:
    return tomllib.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def test_frozen_contract_loads_and_is_hashable() -> None:
    contract = load_strategy_contract(CONTRACT_PATH)

    assert contract.swing.strategy_id == "SWING.SECTOR_RESIDUAL_MOMENTUM.10D.V1"
    assert contract.intraday.strategy_id == "INTRADAY.VWAP_EXHAUSTION_REVERSAL.30M.V1"
    assert len(contract.sha256()) == 64
    # The same content must always hash the same, or the contract cannot be bound
    # to the evidence produced under it.
    assert contract.sha256() == load_strategy_contract(CONTRACT_PATH).sha256()


def test_swing_exit_must_be_timeout_only() -> None:
    """Daily bars cannot show when an intraday stop was touched."""

    raw = _raw()
    raw["swing"]["exit_rule"] = "target_stop_timeout"

    with pytest.raises(ValueError, match="timeout-only"):
        StrategyContract.model_validate(raw)


def test_intraday_target_must_exceed_stop() -> None:
    raw = _raw()
    raw["intraday"]["target_atr_multiple"] = 1.2
    raw["intraday"]["stop_atr_multiple"] = 1.5

    with pytest.raises(ValueError, match="risks more than it seeks"):
        StrategyContract.model_validate(raw)


def test_stop_tighter_than_one_average_range_is_rejected() -> None:
    """A sub-1 ATR stop is hit by ordinary noise, not by the thesis failing."""

    raw = _raw()
    raw["intraday"]["stop_atr_multiple"] = 0.75

    with pytest.raises(ValueError, match="greater than or equal to 1"):
        StrategyContract.model_validate(raw)


def test_atr_must_be_measured_on_the_trading_timeframe() -> None:
    """A daily ATR on a 30-minute hold puts the stop out of reach."""

    raw = _raw()
    raw["intraday"]["atr_timeframe"] = "1Day"

    with pytest.raises(ValueError, match="trading timeframe"):
        StrategyContract.model_validate(raw)


def test_raw_news_counts_cannot_be_enabled() -> None:
    """Provider coverage grew across the sample; a raw count encodes that."""

    raw = _raw()
    raw["features"]["raw_news_counts_prohibited"] = False

    with pytest.raises(ValueError, match="raw news counts are prohibited"):
        StrategyContract.model_validate(raw)


def test_experiment_budget_cannot_be_widened() -> None:
    """Trying enough variants guarantees one passes by luck."""

    raw = _raw()
    raw["experiment_budget"]["maximum_learned_candidates"] = 20

    with pytest.raises(ValueError):
        StrategyContract.model_validate(raw)

    raw = _raw()
    raw["experiment_budget"]["shadow_retries"] = 1

    with pytest.raises(ValueError):
        StrategyContract.model_validate(raw)


def test_random_cross_validation_cannot_be_configured() -> None:
    raw = _raw()
    raw["validation"]["unseen_ticker_assignment"] = "random"

    with pytest.raises(ValueError, match="deterministic"):
        StrategyContract.model_validate(raw)


def test_strategy_identity_must_be_versioned() -> None:
    """A redefined setup reusing an old name invalidates every comparison."""

    raw = _raw()
    raw["swing"]["strategy_id"] = "SWING.SECTOR_RESIDUAL_MOMENTUM.10D"

    with pytest.raises(ValueError, match="must be versioned"):
        StrategyContract.model_validate(raw)


def test_labels_must_retain_benchmark_relative_returns() -> None:
    raw = _raw()
    raw["labels"]["retained"] = ["gross_return", "cost", "net_return"]

    with pytest.raises(ValueError, match="labels must retain"):
        StrategyContract.model_validate(raw)


def test_intraday_universe_cannot_be_index_restricted() -> None:
    """The most intraday-tradable names are often not index constituents."""

    raw = _raw()
    raw["intraday_universe"]["index_restricted"] = True

    with pytest.raises(ValueError, match="must not be index-restricted"):
        StrategyContract.model_validate(raw)


def test_relative_volume_must_exclude_the_session_being_traded() -> None:
    """Selecting on today's volume uses information the decision cannot have."""

    raw = _raw()
    raw["intraday_universe"]["relative_volume_excludes_current_session"] = False

    with pytest.raises(ValueError, match="prior sessions only"):
        StrategyContract.model_validate(raw)


def test_relative_volume_floor_must_actually_select() -> None:
    raw = _raw()
    raw["intraday_universe"]["minimum_relative_volume"] = 1.0

    with pytest.raises(ValueError, match="does not select stocks in play"):
        StrategyContract.model_validate(raw)


def test_fixed_clock_sampling_is_rejected() -> None:
    """Only 3.8% of eligible stock-days are in play; the rest teach nothing."""

    raw = _raw()
    raw["methodology"]["sampling"] = "fixed_interval"

    with pytest.raises(ValueError, match="stocks that were not moving"):
        StrategyContract.model_validate(raw)


def test_published_methods_cannot_be_swapped_for_ad_hoc_ones() -> None:
    raw = _raw()
    raw["methodology"]["cross_validation"] = "k_fold"

    with pytest.raises(ValueError, match="purged and embargoed"):
        StrategyContract.model_validate(raw)

    raw = _raw()
    raw["methodology"]["labeling"] = "fixed_horizon"

    with pytest.raises(ValueError, match="target, stop, or timeout"):
        StrategyContract.model_validate(raw)


def test_unreadable_contract_fails_closed() -> None:
    with pytest.raises(DataReadinessError, match="unreadable"):
        load_strategy_contract(Path("configs/does_not_exist.toml"))
