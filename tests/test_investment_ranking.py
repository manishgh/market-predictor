"""Training-rank semantics and common date support with retained unknown outcomes."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.investment.diagnostics import PREDICTION
from market_predictor.investment.ranking import RANKING_SCORE, ranking_diagnostics, within_date_rank_targets
from market_predictor.investment.training import TARGET, investment_diagnostics, investment_selection_diagnostics


def frame() -> pd.DataFrame:
    actual = np.array([-0.2, 0.1, 0.1, 0.5, np.nan, -0.1, 0.1, 0.2, np.nan, 0.1, 0.1, np.nan])
    score = np.array([-0.5, 0.1, 0.1, 0.8, 0.99, 0.0, 0.0, 0.0, np.nan, 0.1, 0.2, 0.3])
    return pd.DataFrame({"decision_id": [f"decision-{i}" for i in range(12)],
        "security_id": [f"issuer-{i}" for i in range(12)],
        "decision_session": [date(2022, 1, 31)] * 5 + [date(2022, 2, 28)] * 4 + [date(2022, 3, 31)] * 3,
        "horizon_sessions": 252, "research_eligible": np.isfinite(actual), "feature_eligible": np.isfinite(score),
        TARGET: actual, "stock_net_return": actual + 0.05,
        "spy_gross_return": np.where(np.isfinite(actual), 0.05, np.nan),
        "qqq_gross_return": np.where(np.isfinite(actual), 0.04, np.nan),
        "sector_gross_return": np.where(np.isfinite(actual), 0.06, np.nan), RANKING_SCORE: score},
        index=np.arange(12) * 3 + 7)


def test_training_midranks_ties_alignment_and_singletons() -> None:
    source = frame()
    train = source.loc[source.research_eligible].iloc[:8].copy()
    before = train.copy(deep=True)
    ranks = within_date_rank_targets(train)
    np.testing.assert_allclose(ranks, [-0.75, 0.0, 0.0, 0.75, -2/3, 0.0, 2/3, 0.0], atol=1e-15)
    pd.testing.assert_frame_equal(train, before)
    permutation = [7, 2, 6, 0, 5, 1, 4, 3]
    np.testing.assert_array_equal(within_date_rank_targets(train.iloc[permutation]), ranks[permutation])
    train[TARGET] = 0.4
    np.testing.assert_array_equal(within_date_rank_targets(train), np.zeros(len(train)))


def test_nontraining_targets_do_not_enter_frozen_training_rank_vector() -> None:
    source = frame()
    train_index = source.index[[0, 1, 2, 3]]
    expected = within_date_rank_targets(source.loc[train_index])
    source.loc[~source.index.isin(train_index), TARGET] = 1e15
    source.loc[~source.index.isin(train_index), "research_eligible"] = True
    np.testing.assert_array_equal(within_date_rank_targets(source.loc[train_index]), expected)


def test_supported_constant_score_dates_contribute_zero_and_auc_half() -> None:
    result = ranking_diagnostics(frame())
    assert result["rank_date_support"] == ["2022-01-31", "2022-02-28"]
    assert result["auc_date_support"] == result["rank_date_support"]
    assert result["rank_correlation_dates"] == 2 and result["constant_score_rank_dates"] == 1
    assert result["mean_date_rank_correlation"] == pytest.approx(0.5)
    assert result["auc"]["mean_within_date_auc"] == pytest.approx(0.75)
    assert result["rank_by_date"][1]["correlation"] == 0.0
    assert result["rank_by_date"][2]["missing_reason"] == "fewer_than_three_known_outcomes"
    # The old diagnostic deliberately retains its original convention: omit constant-score dates.
    original = investment_diagnostics(frame().rename(columns={RANKING_SCORE: PREDICTION}))
    assert original["mean_date_rank_correlation"] == pytest.approx(1.0)
    assert original["rank_correlation_dates"] == 1
    assert not {"mse", "mae", "zero_excess_baseline_mse", "zero_excess_baseline_mae"}.intersection(result)


def test_metric_date_support_depends_on_labels_and_not_model_variation() -> None:
    source = frame()
    before = ranking_diagnostics(source)
    source.loc[source.feature_eligible, RANKING_SCORE] = 0.0
    constant = ranking_diagnostics(source)
    assert constant["rank_date_support"] == before["rank_date_support"]
    assert constant["auc_date_support"] == before["auc_date_support"]
    assert constant["mean_date_rank_correlation"] == 0.0
    assert constant["auc"]["mean_within_date_auc"] == 0.5
    assert constant["constant_score_rank_dates"] == 2


def test_selected_unknown_is_retained_and_common_selection_matches_original() -> None:
    source = frame()
    result = ranking_diagnostics(source)
    selection = investment_selection_diagnostics(source, RANKING_SCORE)
    assert all(result[key] == value for key, value in selection.items())
    original = investment_diagnostics(source.rename(columns={RANKING_SCORE: PREDICTION}))
    assert all(original[key] == value for key, value in selection.items())
    january = result["top_decile_by_date"][0]
    assert january["selected_security_ids"] == ["issuer-4"]
    assert january["known_selected_outcomes"] == 0
    assert january["mean_net_excess_vs_spy"] is None
    assert result["top_decile_mean_net_excess_vs_spy"] is None
    assert result["scored_rows"] == 11 and result["known_scored_outcomes"] == 9


def test_all_unknown_outcomes_have_no_ranking_metrics_but_keep_selected_stocks() -> None:
    source = frame()
    source["research_eligible"] = False
    for name in (TARGET, "stock_net_return", "spy_gross_return", "qqq_gross_return", "sector_gross_return"):
        source[name] = np.nan
    result = ranking_diagnostics(source)
    assert result["mean_date_rank_correlation"] is None and result["auc"]["mean_within_date_auc"] is None
    assert result["rank_date_support"] == result["auc_date_support"] == []
    assert result["unknown_scored_outcomes"] == 11
    assert len(result["top_decile_by_date"]) == 3 and result["top_decile_complete_dates"] == 0


@pytest.mark.parametrize("poison", ["target_nan", "target_inf", "ineligible", "horizon", "missing_horizon", "duplicate", "date"])
def test_invalid_training_partition_refuses(poison: str) -> None:
    source = frame().iloc[:4].copy()
    row = source.index[0]
    if poison == "target_nan":
        source.loc[row, TARGET] = np.nan
    elif poison == "target_inf":
        source.loc[row, TARGET] = np.inf
    elif poison == "ineligible":
        source.loc[row, "research_eligible"] = False
    elif poison == "horizon":
        source.loc[row, "horizon_sessions"] = 63
    elif poison == "missing_horizon":
        source["horizon_sessions"] = source.horizon_sessions.astype(float)
        source.loc[row, "horizon_sessions"] = np.nan
    elif poison == "duplicate":
        source.loc[row, "decision_id"] = source.decision_id.iloc[1]
    elif poison == "date":
        source.loc[row, "decision_session"] = None
    with pytest.raises(DataReadinessError):
        within_date_rank_targets(source)


@pytest.mark.parametrize("poison", ["score_nan", "score_inf", "unscored_value", "horizon", "date", "duplicate", "target_reconciliation"])
def test_invalid_scoring_contract_refuses(poison: str) -> None:
    source = frame()
    row = source.index[0]
    if poison == "score_nan":
        source.loc[row, RANKING_SCORE] = np.nan
    elif poison == "score_inf":
        source.loc[row, RANKING_SCORE] = np.inf
    elif poison == "unscored_value":
        source.loc[source.index[8], RANKING_SCORE] = 0.1
    elif poison == "horizon":
        source.loc[row, "horizon_sessions"] = 63
    elif poison == "date":
        source.loc[row, "decision_session"] = None
    elif poison == "duplicate":
        source.loc[row, "decision_id"] = source.decision_id.iloc[1]
    elif poison == "target_reconciliation":
        source.loc[row, TARGET] = 9.0
    with pytest.raises(DataReadinessError):
        ranking_diagnostics(source)
