"""Score-only roster arithmetic, outcome poisoning and selection invariance."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.investment.diagnostics import PREDICTION
from market_predictor.investment.forecast_adjustment import centered_shrunk_predictions


def roster() -> pd.DataFrame:
    return pd.DataFrame({"decision_id": [f"decision-{i}" for i in range(8)],
        "security_id": ["c", "a", "b", "d", "a", "b", "c", "d"],
        "decision_session": [date(2022, 1, 31)] * 4 + [date(2022, 2, 28)] * 4,
        "horizon_sessions": 252, "scope": "temporal",
        "feature_eligible": [True, True, True, False, True, True, True, False],
        "research_eligible": [True, False, True, False, True, True, False, False],
        "target_net_excess_vs_spy": [0.1, np.nan, -0.1, np.nan, 0.2, 0.1, np.nan, np.nan],
        PREDICTION: [0.2, 0.8, -0.4, np.nan, 0.4, 0.4, -0.2, np.nan]},
        index=[13, 7, 22, 9, 100, 40, 60, 11])


def test_adjustment_includes_unknowns_preserves_values_and_does_not_mutate() -> None:
    frame = roster()
    before = frame.copy(deep=True)
    result = centered_shrunk_predictions(frame)
    # Each date mean is 0.2; the first date's highest forecast has no known label.
    np.testing.assert_allclose(result[PREDICTION], [0.0, 0.3, -0.3, np.nan, 0.1, 0.1, -0.2, np.nan],
        rtol=0, atol=1e-15, equal_nan=True)
    pd.testing.assert_frame_equal(frame, before)
    pd.testing.assert_frame_equal(result.drop(columns=PREDICTION), before.drop(columns=PREDICTION))
    assert result.index.equals(frame.index)
    assert result[PREDICTION].notna().equals(frame.feature_eligible)
    np.testing.assert_allclose(result.groupby("decision_session")[PREDICTION].mean(), 0, atol=1e-15)


def test_outcomes_and_future_labels_cannot_change_adjustment() -> None:
    frame = roster()
    expected = centered_shrunk_predictions(frame)[PREDICTION]
    frame["research_eligible"] = ~frame.research_eligible
    frame["target_net_excess_vs_spy"] = np.arange(len(frame), dtype=float) * 1e8
    frame["stock_net_return"] = -1e20
    pd.testing.assert_series_equal(centered_shrunk_predictions(frame)[PREDICTION], expected)
    no_labels = frame.drop(columns=["research_eligible", "target_net_excess_vs_spy", "stock_net_return"])
    pd.testing.assert_series_equal(centered_shrunk_predictions(no_labels)[PREDICTION], expected)


def test_float32_inputs_use_float64_date_mean() -> None:
    frame = roster().iloc[:2].copy()
    frame[PREDICTION] = np.array([1.0, 1.0 + 2.0**-23], dtype=np.float32)
    result = centered_shrunk_predictions(frame)
    np.testing.assert_array_equal(result[PREDICTION], [-2.0**-25, 2.0**-25])
    assert result[PREDICTION].mean() == 0.0


def test_date_isolation_row_permutation_ties_and_selection_are_preserved() -> None:
    frame = roster()
    adjusted = centered_shrunk_predictions(frame)
    permuted = centered_shrunk_predictions(frame.iloc[[6, 1, 5, 7, 4, 0, 3, 2]])
    pd.testing.assert_series_equal(permuted[PREDICTION].sort_index(), adjusted[PREDICTION].sort_index())
    for _, group in frame.loc[frame.feature_eligible].groupby("decision_session"):
        before = group.sort_values([PREDICTION, "security_id"], ascending=[False, True]).decision_id.tolist()
        after = adjusted.loc[group.index].sort_values([PREDICTION, "security_id"], ascending=[False, True]).decision_id.tolist()
        assert before == after
        np.testing.assert_array_equal(group[PREDICTION].rank(), adjusted.loc[group.index, PREDICTION].rank())
    frame.loc[frame.decision_session.eq(date(2022, 2, 28)), PREDICTION] *= 10
    pd.testing.assert_series_equal(centered_shrunk_predictions(frame)[PREDICTION].iloc[:4], adjusted[PREDICTION].iloc[:4])


def test_single_eligible_stock_centers_to_zero_and_empty_dates_stay_unscored() -> None:
    frame = roster()
    frame.loc[frame.index[1:], "feature_eligible"] = False
    frame.loc[frame.index[1:], PREDICTION] = np.nan
    result = centered_shrunk_predictions(frame)
    assert result[PREDICTION].iloc[0] == 0.0
    assert result[PREDICTION].iloc[1:].isna().all()
    frame.loc[:, "feature_eligible"] = False
    frame.loc[:, PREDICTION] = np.nan
    assert centered_shrunk_predictions(frame)[PREDICTION].isna().all()


@pytest.mark.parametrize("poison", ["horizon", "scope", "identity", "stock_date", "index", "infinite",
    "unscored_value", "missing_score", "eligibility", "missing_date", "text_forecast", "boolean_forecast"])
def test_invalid_roster_refuses(poison: str) -> None:
    frame = roster()
    row = frame.index[0]
    if poison == "horizon":
        frame.loc[row, "horizon_sessions"] = 63
    elif poison == "scope":
        frame.loc[row, "scope"] = "security_transfer"
    elif poison == "identity":
        frame.loc[row, "decision_id"] = frame.decision_id.iloc[1]
    elif poison == "stock_date":
        frame.loc[row, "security_id"] = frame.security_id.iloc[1]
    elif poison == "index":
        frame.index = [0] * len(frame)
    elif poison == "infinite":
        frame.loc[row, PREDICTION] = np.inf
    elif poison == "unscored_value":
        frame.loc[frame.index[3], PREDICTION] = 0.0
    elif poison == "missing_score":
        frame.loc[row, PREDICTION] = np.nan
    elif poison == "eligibility":
        frame["feature_eligible"] = frame.feature_eligible.astype(int)
    elif poison == "missing_date":
        frame.loc[row, "decision_session"] = None
    elif poison == "text_forecast":
        frame[PREDICTION] = frame[PREDICTION].astype(str)
    elif poison == "boolean_forecast":
        frame[PREDICTION] = True
    with pytest.raises(DataReadinessError):
        centered_shrunk_predictions(frame)
