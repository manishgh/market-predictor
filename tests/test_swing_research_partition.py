from __future__ import annotations

import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.features.panel import CATALYST_RANKING_FEATURES, TECHNICAL_RANKING_FEATURES
from market_predictor.swing.features.research_join import DECISION_KEYS
from market_predictor.swing.features.research_partition import (
    ResearchFeaturePartition,
    assemble_research_feature_partition,
)
from tests.test_swing_features import contract as contract
from tests.test_swing_research_peer_features import _features


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = _features()
    features["primary_benchmark"] = "XLK"
    expected = features.iloc[:59].loc[:, [*DECISION_KEYS, "sector", "session_date_et", "primary_benchmark"]].copy()
    outcomes = expected.copy()
    outcomes["fixed_horizon_net_return"] = 0.02
    outcomes.loc[0, "fixed_horizon_net_return"] = float("nan")
    outcomes["stock_source_admitted"] = outcomes.fixed_horizon_net_return.notna()
    outcomes["training_eligible"] = False
    return expected, features, outcomes


def _assemble(expected: pd.DataFrame, features: pd.DataFrame, outcomes: pd.DataFrame,
    contract: StrategyContract,
) -> ResearchFeaturePartition:
    return assemble_research_feature_partition(expected_decisions=expected, base_features=features,
        outcomes=outcomes, contract=contract, retained_security_ids=frozenset(expected.security_id),
        availability_columns={name: "base_available_at" for name in TECHNICAL_RANKING_FEATURES})


def test_partition_preserves_population_context_missing_outcomes_and_order(contract: StrategyContract) -> None:
    expected, features, outcomes = _inputs()
    result = _assemble(expected, features.iloc[::-1], outcomes.iloc[::-1], contract)
    assert result.rows.decision_id.tolist() == expected.decision_id.tolist()
    assert len(result.rows) == 59
    assert pd.isna(result.rows.loc[0, "fixed_horizon_net_return"])
    assert result.rows.primary_benchmark.eq("XLK").all()
    assert result.rows.feature_eligible.all()
    assert not result.rows.training_eligible.any()
    assert len(result.model_columns) == 120
    assert not set(result.model_columns).intersection(outcomes.columns)
    assert result.audit["outcome_filtered_rows"] == 0
    assert result.audit["training_eligible"] is False


def test_jointly_missing_rows_cannot_change_peer_population(contract: StrategyContract) -> None:
    expected, features, outcomes = _inputs()
    with pytest.raises(DataReadinessError, match="complete frozen decision population"):
        _assemble(expected, features.iloc[1:], outcomes.iloc[1:], contract)


@pytest.mark.parametrize("column,value", [("sector", "wrong"), ("primary_benchmark", "XLE"),
    ("session_date_et", "2024-01-03"), ("ticker", "WRONG")])
@pytest.mark.parametrize("source", ["features", "outcomes"])
def test_context_poison_rejected(contract: StrategyContract, column: str, value: str, source: str) -> None:
    expected, features, outcomes = _inputs()
    selected = features if source == "features" else outcomes
    selected[column] = selected[column].astype(object)
    selected.loc[0, column] = value
    with pytest.raises(DataReadinessError, match="differs from frozen"):
        _assemble(expected, features, outcomes, contract)


def test_outcomes_cannot_supply_feature_eligibility(contract: StrategyContract) -> None:
    expected, features, outcomes = _inputs()
    outcomes["feature_eligible"] = outcomes.stock_source_admitted
    with pytest.raises(DataReadinessError, match="cannot supply predictor eligibility"):
        _assemble(expected, features, outcomes, contract)


def test_poisoned_targets_and_excluded_inputs_do_not_change_predictors(contract: StrategyContract) -> None:
    expected, features, outcomes = _inputs()
    original = _assemble(expected, features, outcomes, contract)
    outcomes["fixed_horizon_net_return"] = -1.0
    outcomes["stock_source_admitted"] = False
    features.loc[59, list(TECHNICAL_RANKING_FEATURES)] = 1e9
    result = _assemble(expected, features, outcomes, contract)
    pd.testing.assert_frame_equal(original.rows[list(original.model_columns)], result.rows[list(result.model_columns)])
    assert result.rows.feature_eligible.all()


def test_missing_predictors_remain_missing_without_losing_decision(contract: StrategyContract) -> None:
    expected, features, outcomes = _inputs()
    features.loc[0, list(TECHNICAL_RANKING_FEATURES)] = float("nan")
    features.loc[0, "feature_eligible"] = False
    features.loc[0, "daily_bar_count"] = 0
    features.loc[0, "base_available_at"] = pd.NaT
    result = _assemble(expected, features, outcomes, contract)
    assert len(result.rows) == 59
    assert result.rows.loc[0, list(TECHNICAL_RANKING_FEATURES)].isna().all()
    assert not result.rows.loc[0, "feature_eligible"]


def test_catalyst_profile_keeps_unknown_news_and_checks_publication_clock(contract: StrategyContract) -> None:
    expected, features, outcomes = _inputs()
    features["feature_profile"] = "catalyst_full"
    features["news_available_at"] = pd.Timestamp("2024-01-02T22:00:00Z")
    for column in CATALYST_RANKING_FEATURES:
        features[column] = 0.25
    features.loc[0, list(CATALYST_RANKING_FEATURES)] = float("nan")
    features.loc[0, "news_available_at"] = pd.NaT
    availability = {**{name: "base_available_at" for name in TECHNICAL_RANKING_FEATURES},
        **{name: "news_available_at" for name in CATALYST_RANKING_FEATURES}}
    arguments = dict(expected_decisions=expected, base_features=features, outcomes=outcomes,
        contract=contract, retained_security_ids=frozenset(expected.security_id), availability_columns=availability)
    result = assemble_research_feature_partition(**arguments)
    assert result.rows.loc[0, list(CATALYST_RANKING_FEATURES)].isna().all()
    assert len(result.model_columns) == 3 * (len(TECHNICAL_RANKING_FEATURES) + len(CATALYST_RANKING_FEATURES))
    features.loc[1, "news_available_at"] = pd.Timestamp("2024-01-02T23:01:00Z")
    with pytest.raises(DataReadinessError, match="unavailable at its decision"):
        assemble_research_feature_partition(**arguments)


def test_duplicate_columns_fail_explicitly(contract: StrategyContract) -> None:
    expected, features, outcomes = _inputs()
    features = pd.concat([features, features[["sector"]]], axis=1)
    with pytest.raises(DataReadinessError):
        _assemble(expected, features, outcomes, contract)
