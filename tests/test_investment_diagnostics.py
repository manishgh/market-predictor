"""Saved-error identities and denominator tests on synthetic paired forecasts."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.investment import diagnostics as module
from market_predictor.investment.training import TARGET


def predictions() -> pd.DataFrame:
    dates = [date(2022, 1, 31)] * 3 + [date(2022, 2, 28)] * 3 + [date(2022, 3, 31)] * 2
    actual = np.array([0.2, -0.1, np.nan, 0.3, 0.1, 0.05, np.nan, np.nan])
    forecasts = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.4, 0.5, np.nan])
    return pd.DataFrame({"decision_id": [f"decision-{i}" for i in range(8)], "security_id": [f"issuer-{i}" for i in range(8)],
        "decision_session": dates, "horizon_sessions": 252, "sector": ["A", "B", "A", "B", "A", "B", "Z", "Z"],
        "feature_eligible": np.isfinite(forecasts), "research_eligible": np.isfinite(actual), TARGET: actual,
        "stock_net_return": actual + 0.05, "spy_gross_return": np.where(np.isfinite(actual), 0.05, np.nan),
        "qqq_gross_return": np.where(np.isfinite(actual), 0.07, np.nan),
        "sector_gross_return": np.where(np.isfinite(actual), 0.06, np.nan), module.PREDICTION: forecasts,
        "news_finite_count": [0, 1, 85, 85, 1, 0, 0, 0], "financial_finite_count": [0, 1, 6, 6, 1, 0, 0, 0],
        "news_coverage_bin": ["0", "1-84", "85", "85", "1-84", "0", "0", "0"],
        "financial_coverage_bin": ["0", "1-5", "6", "6", "1-5", "0", "0", "0"]})


def test_mse_moment_and_date_decompositions_reconcile_without_forecast_changes() -> None:
    frame = predictions()
    before = frame.copy(deep=True)
    result = module.prediction_diagnostics(frame)
    moments = result["moments"]
    assert result["canonical_metrics"]["known_scored_outcomes"] == 5
    assert result["canonical_metrics"]["unknown_scored_outcomes"] == 2
    mse = result["canonical_metrics"]["mse"]
    assert moments["squared_bias"] + moments["error_variance"] == pytest.approx(mse)
    assert moments["date_mean_bias_mse"] + moments["within_date_centered_mse"] == pytest.approx(mse)
    assert moments["zero_baseline_identity_residual"] == pytest.approx(0, abs=1e-14)
    assert moments["date_mean_plus_within_residual"] == pytest.approx(0, abs=1e-14)
    assert result["diagnostic_slope_is_not_a_fitted_calibrator"] and not result["predictions_modified"]
    pd.testing.assert_frame_equal(frame, before)


def test_auc_signed_scores_ties_one_class_and_unknown_date_denominators() -> None:
    frame = predictions()
    frame.loc[frame.feature_eligible, module.PREDICTION] = -0.5  # signed returns, not clipped probabilities
    result = module.prediction_diagnostics(frame)["auc"]
    assert result["pooled_equal_date_weight_auc"] == 0.5
    assert result["mean_within_date_auc"] == 0.5
    assert (result["positive_rows"], result["negative_rows"]) == (4, 1)
    assert (result["eligible_dates"], result["one_class_dates"], result["known_dates"]) == (1, 1, 2)
    assert (result["declared_dates"], result["scored_dates"], result["scored_dates_without_known_outcomes"]) == (3, 3, 1)


def test_all_unknown_outcomes_are_retained_and_auc_is_unavailable() -> None:
    frame = predictions()
    frame["research_eligible"] = False
    for column in (TARGET, "stock_net_return", "spy_gross_return", "qqq_gross_return", "sector_gross_return"):
        frame[column] = np.nan
    result = module.prediction_diagnostics(frame)
    assert result["moments"] is None
    assert result["canonical_metrics"]["scored_rows"] == 7
    assert result["canonical_metrics"]["unknown_scored_outcomes"] == 7
    assert result["auc"]["pooled_equal_date_weight_auc"] is None
    assert result["auc"]["scored_dates_without_known_outcomes"] == 3


def test_pair_contributions_keep_global_date_weights_in_every_partition_and_tail() -> None:
    left, right = predictions(), predictions()
    right.loc[right.feature_eligible, module.PREDICTION] += np.array([0.05, -0.1, 0.2, 0.02, 0.2, 0.0, 0.3])
    result = module.paired_diagnostics(left, right)
    known = left.research_eligible
    # Independent date-normalized calculation: January has two known rows, February three.
    weights = np.array([1 / 4, 1 / 4, 1 / 6, 1 / 6, 1 / 6])
    y, p = left.loc[known, TARGET].to_numpy(), left.loc[known, module.PREDICTION].to_numpy()
    q = right.loc[known, module.PREDICTION].to_numpy()
    expected = float(np.dot(weights, (q - y) ** 2 - (p - y) ** 2))
    identity = result["paired_identity"]
    assert identity["full_minus_technical_mse"] == pytest.approx(expected)
    assert identity["squared_prediction_delta"] + identity["prediction_delta_error_interaction"] == pytest.approx(expected)
    assert identity["identity_residual"] == pytest.approx(0, abs=1e-14)
    for rows in result["strata"].values():
        assert sum(row["mse_delta_contribution"] for row in rows) == pytest.approx(expected)
        assert sum(row["global_weight_fraction"] for row in rows) == pytest.approx(1)
        assert sum(row["declared_rows"] for row in rows) == 8
        assert sum(row["unknown_scored_rows"] for row in rows) == 2
    unknown_sector = next(row for row in result["strata"]["sector"] if row["value"] == "Z")
    assert unknown_sector["known_rows"] == 0 and unknown_sector["unknown_scored_rows"] == 1
    assert unknown_sector["mse_delta_contribution"] == 0
    assert [row["fraction_by_known_row_count"] for row in result["tail_concentration"]] == [0.01, 0.05, 0.1]
    assert result["examples_removed"] == 0 and not result["model_selected"]


def test_finite_zero_coverage_and_exact_frozen_bins() -> None:
    names = tuple(f"column-{i}" for i in range(215))
    features = pd.DataFrame(0.0, index=range(3), columns=names)
    features.iloc[0, 124:] = np.nan
    features.iloc[1, 124:] = np.nan
    features.iloc[1, [124, 209]] = 0.0
    result = module.feature_coverage(features, names)
    assert result.news_finite_count.tolist() == [0, 1, 85]
    assert result.financial_finite_count.tolist() == [0, 1, 6]
    assert result.news_coverage_bin.tolist() == ["0", "1-84", "85"]
    assert result.financial_coverage_bin.tolist() == ["0", "1-5", "6"]


@pytest.mark.parametrize("poison", ["identity", "mask", "outcome", "coverage", "duplicate", "horizon", "nonfinite"])
def test_pair_refuses_unmatched_or_malformed_inputs(poison: str) -> None:
    left, right = predictions(), predictions()
    if poison == "identity":
        right.loc[0, "security_id"] = "wrong-issuer"
    elif poison == "mask":
        right.loc[0, module.PREDICTION] = np.nan
        right.loc[0, "feature_eligible"] = False
    elif poison == "outcome":
        right.loc[0, TARGET] += 0.01
    elif poison == "coverage":
        right.loc[0, "financial_finite_count"] = 2
    elif poison == "duplicate":
        right.loc[1, "decision_id"] = right.loc[0, "decision_id"]
    elif poison == "horizon":
        right.loc[0, "horizon_sessions"] = 63
    elif poison == "nonfinite":
        right.loc[0, module.PREDICTION] = np.inf
    with pytest.raises(DataReadinessError):
        module.paired_diagnostics(left, right)
