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

    assert contract.swing.strategy_id == "swing"
    assert contract.intraday.strategy_id == "intraday"
    assert len(contract.sha256()) == 64
    # The same content must always hash the same, or the contract cannot be bound
    # to the evidence produced under it.
    assert contract.sha256() == load_strategy_contract(CONTRACT_PATH).sha256()


def test_retirement_applies_to_learned_oos_failure_not_baseline_failure() -> None:
    contract = load_strategy_contract(CONTRACT_PATH)
    rule = contract.retirement.rule.lower()

    assert "learned strategy" in rule
    assert "out-of-sample economic acceptance" in rule
    assert "deterministic baseline failure alone does not block fitting" in rule
    assert "er3 admission" not in rule


def test_swing_exit_must_resolve_a_barrier() -> None:
    """Timeout-only holds through drawdowns a real stop would have closed."""

    raw = _raw()
    raw["swing"]["exit_rule"] = "horizon_close"

    with pytest.raises(ValueError, match="target, stop, or timeout"):
        StrategyContract.model_validate(raw)


def test_same_bar_ambiguity_must_resolve_conservatively() -> None:
    """A daily bar shows both barriers were touched, not which came first."""

    raw = _raw()
    raw["swing"]["same_bar_barrier_resolution"] = "target_first"

    with pytest.raises(ValueError, match="stop-first"):
        StrategyContract.model_validate(raw)


def test_swing_scoring_must_be_sector_neutral() -> None:
    raw = _raw()
    raw["swing"]["sector_neutral_scoring"] = False

    with pytest.raises(ValueError, match="sector-neutral"):
        StrategyContract.model_validate(raw)


def test_clock_bars_are_rejected_for_intraday_decisions() -> None:
    """The opening bar carries tens of times the volume of a midday bar."""

    raw = _raw()
    raw["intraday"]["decision_bar_structure"] = "time"

    with pytest.raises(ValueError, match="time of day rather than information"):
        StrategyContract.model_validate(raw)


def test_volume_bars_require_one_minute_input() -> None:
    raw = _raw()
    raw["intraday"]["volume_bar_source_timeframe"] = "5Min"

    with pytest.raises(ValueError, match="one-minute or finer"):
        StrategyContract.model_validate(raw)


def test_rolling_features_must_reset_overnight() -> None:
    raw = _raw()
    raw["intraday"]["reset_rolling_features_overnight"] = False

    with pytest.raises(ValueError, match="reset overnight"):
        StrategyContract.model_validate(raw)


def test_intraday_warmup_matches_longest_session_reset_feature() -> None:
    raw = _raw()
    raw["intraday"]["minimum_warmup_bars"] = 19

    with pytest.raises(ValueError, match="longest session-reset feature"):
        StrategyContract.model_validate(raw)


def test_intraday_warmup_leaves_decisions_in_a_normal_session() -> None:
    raw = _raw()
    raw["intraday"]["volume_bars_per_session_target"] = 20

    with pytest.raises(ValueError, match="leave decision bars"):
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


def test_cross_sectional_outputs_cannot_be_disabled() -> None:
    raw = _raw()
    raw["features"]["cross_sectional_emit_sector_relative"] = False

    with pytest.raises(ValueError, match="sector-relative"):
        StrategyContract.model_validate(raw)


def test_relationship_features_use_the_frozen_published_methods() -> None:
    raw = _raw()
    raw["features"]["technical_relationship_methods"] = [
        "ad_hoc_divergence",
        "granville_obv_confirmation",
        "kaufman_efficiency_ratio_regime",
    ]
    with pytest.raises(ValueError, match="frozen published methods"):
        StrategyContract.model_validate(raw)

    raw = _raw()
    raw["features"]["rsi_pivot_span_bars"] = 3
    with pytest.raises(ValueError, match="five-bar confirmed pivot"):
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


def test_swing_and_intraday_validation_counts_are_independent() -> None:
    raw = _raw()
    raw["validation"]["swing_walk_forward_folds"] = 2
    with pytest.raises(ValueError, match="seven-year swing"):
        StrategyContract.model_validate(raw)

    raw = _raw()
    raw["validation"]["intraday_purged_folds"] = 3
    with pytest.raises(ValueError, match="four purged folds"):
        StrategyContract.model_validate(raw)


def test_security_exclusion_rule_cannot_be_relaxed_or_applied_to_benchmarks() -> None:
    raw = _raw()
    raw["data_quality"]["maximum_security_exclusion_fraction"] = 0.06
    with pytest.raises(ValueError):
        StrategyContract.model_validate(raw)

    raw = _raw()
    raw["data_quality"]["benchmark_exclusions_allowed"] = True
    with pytest.raises(ValueError, match="cannot be excluded"):
        StrategyContract.model_validate(raw)


def test_both_label_schemes_are_required() -> None:
    """Either alone reproduces a failure already on record."""

    raw = _raw()
    raw["labels"]["barrier_labels_enabled"] = False
    with pytest.raises(ValueError, match="closed early"):
        StrategyContract.model_validate(raw)

    raw = _raw()
    raw["labels"]["rank_labels_enabled"] = False
    with pytest.raises(ValueError, match="unrelated to"):
        StrategyContract.model_validate(raw)


def test_rank_quantiles_cannot_swallow_the_middle_band() -> None:
    """A tail of half the cross-section is not a tail."""

    raw = _raw()
    raw["labels"]["rank_top_quantile"] = 0.5

    with pytest.raises(ValueError):
        StrategyContract.model_validate(raw)


def test_labels_must_retain_benchmark_relative_returns() -> None:
    raw = _raw()
    raw["labels"]["retained"] = ["gross_return", "cost", "net_return"]

    with pytest.raises(ValueError, match="labels must retain"):
        StrategyContract.model_validate(raw)


def test_intraday_universe_must_be_the_point_in_time_index() -> None:
    """`intraday_selection` loads index membership unconditionally.

    The field is asserted rather than declared so the contract cannot drift away
    from what the selection code actually does.
    """

    raw = _raw()
    raw["intraday_universe"]["index_restricted"] = False

    with pytest.raises(ValueError, match="point-in-time S&P 500"):
        StrategyContract.model_validate(raw)

    raw = _raw()
    raw["intraday_universe"]["scope"] = "broad_us_point_in_time"

    with pytest.raises(ValueError, match="point-in-time S&P 500"):
        StrategyContract.model_validate(raw)


def test_exchange_traded_products_must_be_excluded() -> None:
    """A fund has no issuer, so a catalyst setup has nothing to condition on.

    Measured density is a median of 8 articles for exchange-traded products
    against 204 for operating companies.
    """

    raw = _raw()
    raw["intraday_universe"]["exclude_exchange_traded_products"] = False

    with pytest.raises(ValueError, match="must be excluded"):
        StrategyContract.model_validate(raw)


def test_price_floor_matches_the_penny_stock_exclusion() -> None:
    raw = _raw()
    assert raw["intraday_universe"]["minimum_price"] == 8.0

    raw["intraday_universe"]["minimum_price"] = 1.0
    with pytest.raises(ValueError):
        StrategyContract.model_validate(raw)


def test_relative_volume_must_exclude_the_session_being_traded() -> None:
    """Today's cumulative numerator must never enter its historical baseline."""

    raw = _raw()
    raw["intraday_universe"]["relative_volume_excludes_current_session"] = False

    with pytest.raises(ValueError, match="prior sessions only"):
        StrategyContract.model_validate(raw)


def test_relative_volume_floor_must_actually_select() -> None:
    raw = _raw()
    raw["intraday_universe"]["minimum_relative_volume"] = 1.0

    with pytest.raises(ValueError, match="exactly 2.0"):
        StrategyContract.model_validate(raw)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("selection_timing", "end_of_session", "cumulative-to-decision"),
        ("activity_timeframe", "1Day", "five-minute"),
        ("activity_numerator", "full_session_volume", "cumulative observed"),
        ("activity_baseline", "mean_session_volume", "same-slot"),
        ("exact_slot_matching", False, "exact slots"),
        ("activity_resets_each_session", False, "reset"),
        ("imputation_allowed", True, "without imputation"),
        ("activation_delay_seconds", 0, "plus 60 seconds"),
        ("maximum_candidates_per_decision", 31, "capped at 30"),
    ),
)
def test_intraday_activity_screen_is_frozen_causally(
    field: str,
    value: object,
    message: str,
) -> None:
    raw = _raw()
    raw["intraday_universe"][field] = value

    with pytest.raises(ValueError, match=message):
        StrategyContract.model_validate(raw)


def test_rank_settings_are_split_by_horizon() -> None:
    contract = load_strategy_contract(CONTRACT_PATH)

    assert contract.labels.swing_rank_within_sector is True
    assert contract.labels.swing_target_cross_section_for_ranking == 50
    assert contract.labels.swing_minimum_cross_section_for_ranking == 30
    assert contract.labels.intraday_rank_within_sector is False
    assert contract.labels.intraday_minimum_cross_section_for_ranking == 10

    raw = _raw()
    raw["labels"]["intraday_rank_within_sector"] = True
    with pytest.raises(ValueError, match="contemporaneous group"):
        StrategyContract.model_validate(raw)


def test_swing_sector_policy_has_target_and_bounded_fallback() -> None:
    contract = load_strategy_contract(CONTRACT_PATH)

    assert contract.swing.target_maximum_sector_weight == pytest.approx(0.20)
    assert contract.swing.hard_maximum_sector_weight == pytest.approx(1.0 / 3.0)
    assert contract.swing.minimum_distinct_sectors_for_selection == 3

    raw = _raw()
    raw["swing"]["target_maximum_sector_weight"] = 0.40
    with pytest.raises(ValueError, match="target sector weight"):
        StrategyContract.model_validate(raw)

    raw = _raw()
    raw["swing"]["maximum_trades_per_decision"] = 2
    with pytest.raises(ValueError, match="required distinct sectors"):
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

    with pytest.raises(ValueError, match="barrier outcome and the"):
        StrategyContract.model_validate(raw)


def test_unreadable_contract_fails_closed() -> None:
    with pytest.raises(DataReadinessError, match="unreadable"):
        load_strategy_contract(Path("configs/does_not_exist.toml"))
