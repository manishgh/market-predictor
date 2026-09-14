"""Bounded calendar, causal-mask, weighting and return-diagnostic unit tests."""
from __future__ import annotations

from datetime import date

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.training import return_validation as owner

TARGET = "spy_fixed_horizon_excess_return"
CUTOFF = pd.Timestamp("2024-01-17T14:30:00Z")


@pytest.fixture(scope="module")
def full_calendar() -> tuple[date, ...]:
    calendar = xcals.get_calendar("XNYS", start="2019-07-09", end="2024-05-28")
    return tuple(calendar.sessions_in_range("2019-07-09", "2024-05-28").date)


def test_exact_full_calendar_folds(full_calendar: tuple[date, ...]) -> None:
    assert len(full_calendar) == 1231
    assert full_calendar[0] == date(2019, 7, 9)
    assert full_calendar[-1] == date(2024, 5, 28)
    folds = owner.return_folds(full_calendar, count=4, minimum_train=503, embargo=10)
    assert [fold.number for fold in folds] == [1, 2, 3, 4]
    assert [len(fold.train) for fold in folds] == [503, 682, 861, 1040]
    assert [len(fold.score) for fold in folds] == [179, 179, 179, 181]
    assert [len(fold.embargo) for fold in folds] == [10] * 4
    boundaries = [
        ("2021-07-06", "2021-07-07", "2021-07-20", "2021-07-21", "2022-04-04"),
        ("2022-03-21", "2022-03-22", "2022-04-04", "2022-04-05", "2022-12-19"),
        ("2022-12-05", "2022-12-06", "2022-12-19", "2022-12-20", "2023-09-07"),
        ("2023-08-23", "2023-08-24", "2023-09-07", "2023-09-08", "2024-05-28"),
    ]
    for fold, expected in zip(folds, boundaries, strict=True):
        stop = len(fold.train)
        assert fold.train == full_calendar[:stop]
        assert fold.embargo == full_calendar[stop:stop + 10]
        assert fold.score == full_calendar[stop + 10:stop + 10 + len(fold.score)]
        actual = (fold.train[-1], fold.embargo[0], fold.embargo[-1], fold.score[0], fold.score[-1])
        assert tuple(day.isoformat() for day in actual) == expected
        assert not set(fold.train) & set(fold.score)
        assert not set(fold.embargo) & (set(fold.train) | set(fold.score))
    assert tuple(day for fold in folds for day in fold.score) == full_calendar[513:]
    assert date(2021, 7, 5) not in full_calendar
    assert date(2023, 9, 4) not in full_calendar


def test_all_missing_day_does_not_shrink_calendar(full_calendar: tuple[date, ...]) -> None:
    before = owner.return_folds(full_calendar, count=4, minimum_train=503, embargo=10)
    metadata = pd.DataFrame({
        "session_date_et": full_calendar, "feature_eligible": True,
        "fixed_horizon_supervision_available": True,
        "research_label_mature_at": pd.Timestamp("2019-07-10T20:00:00Z"),
        "security_holdout": False, "signal": 1.0,
    })
    # Include the first scoring day: missing supervision must not move its boundary.
    missing_positions = [200, 513]
    metadata.loc[missing_positions, "signal"] = np.nan
    metadata.loc[missing_positions, "fixed_horizon_supervision_available"] = False
    metadata.loc[missing_positions, "research_label_mature_at"] = pd.NaT
    indices = owner.fit_indices(metadata, before[0].train, pd.Timestamp("2021-07-21T13:30:00Z"), transfer=False)
    assert_array_equal(indices, np.delete(np.arange(503), 200))
    after = owner.return_folds(full_calendar, count=4, minimum_train=503, embargo=10)
    assert after == before
    assert full_calendar[200] in after[0].train
    assert full_calendar[513] == after[0].score[0]
    assert len(metadata) == 1231


@pytest.mark.parametrize("sessions", [
    (), (date(2024, 1, 3), date(2024, 1, 2)), (date(2024, 1, 2), date(2024, 1, 2)),
])
def test_calendar_must_be_nonempty_unique_and_ordered(sessions: tuple[date, ...]) -> None:
    with pytest.raises(DataReadinessError, match="unique ordered full calendar"):
        owner.return_folds(sessions, count=4, minimum_train=503, embargo=10)


def test_incomplete_fold_calendar_fails_closed(full_calendar: tuple[date, ...]) -> None:
    with pytest.raises(DataReadinessError, match="insufficient sessions"):
        owner.return_folds(full_calendar[:516], count=4, minimum_train=503, embargo=10)


@pytest.mark.parametrize("transfer", [False, True])
def test_actual_maturity_is_strict_and_feature_missingness_is_not_a_filter(transfer: bool) -> None:
    train_day = date(2024, 1, 2)
    metadata = pd.DataFrame({
        "case": ["before", "equal", "after", "null", "unavailable", "ineligible", "outside", "holdout"],
        "session_date_et": [train_day] * 6 + [date(2024, 1, 3), train_day],
        "feature_eligible": [True] * 5 + [False, True, True],
        "fixed_horizon_supervision_available": [True] * 4 + [False, True, True, True],
        "research_label_mature_at": [
            CUTOFF - pd.Timedelta(1, unit="ns"), CUTOFF, CUTOFF + pd.Timedelta(1, unit="ns"),
            pd.NaT, CUTOFF - pd.Timedelta(days=1), CUTOFF - pd.Timedelta(days=1),
            CUTOFF - pd.Timedelta(days=1), CUTOFF - pd.Timedelta(days=1),
        ],
        "security_holdout": [False] * 7 + [True],
        "missing_feature": [np.nan] * 8,
    }, index=[90, 80, 70, 60, 50, 40, 30, 20])
    original = metadata.copy(deep=True)
    indices = owner.fit_indices(metadata, (train_day,), CUTOFF, transfer=transfer)
    assert_array_equal(indices, [0] if transfer else [0, 7])
    assert metadata.iloc[indices].missing_feature.isna().all()
    assert metadata.iloc[indices].research_label_mature_at.lt(CUTOFF).all()
    observed_features = metadata.assign(missing_feature=123.0)
    assert_array_equal(owner.fit_indices(observed_features, (train_day,), CUTOFF, transfer=transfer), indices)
    pd.testing.assert_frame_equal(metadata, original)


def test_independent_transfer_mask_is_identity_stable_under_reordering() -> None:
    names = [f"security-{number}" for number in range(6)] * 2
    metadata = pd.DataFrame({
        "decision_id": [f"decision-{number}" for number in range(12)],
        "security_id": names,
        "session_date_et": [date(2024, 1, 2)] * 6 + [date(2024, 1, 3)] * 6,
        "feature_eligible": True, "fixed_horizon_supervision_available": True,
        "research_label_mature_at": CUTOFF - pd.Timedelta(days=1),
        "signal": np.nan,
    })
    # Frozen unseeded SHA256 first-64-bit threshold assignments, not a ranked 20% quota.
    expected = [False, True, False, False, True, False] * 2
    metadata["security_holdout"] = owner.security_transfer_mask(metadata.security_id)
    assert_array_equal(metadata.security_holdout, expected)
    original = metadata.copy(deep=True)
    sessions = (date(2024, 1, 2), date(2024, 1, 3))
    temporal = owner.fit_indices(metadata, sessions, CUTOFF, transfer=False)
    transfer = owner.fit_indices(metadata, sessions, CUTOFF, transfer=True)
    assert_array_equal(temporal, np.arange(12))
    assert_array_equal(transfer, [0, 2, 3, 5, 6, 8, 9, 11])
    assert not metadata.iloc[transfer].security_holdout.any()
    assert_array_equal(owner.fit_indices(metadata, sessions, CUTOFF, transfer=False), temporal)
    shuffled = metadata.iloc[[7, 2, 11, 4, 0, 9, 3, 6, 10, 5, 8, 1]].copy()
    shuffled["security_holdout"] = owner.security_transfer_mask(shuffled.security_id)
    assert_array_equal(shuffled.security_holdout, shuffled.security_id.isin(["security-1", "security-4"]))
    selected = owner.fit_indices(shuffled, sessions, CUTOFF, transfer=True)
    assert set(shuffled.iloc[selected].decision_id) == set(metadata.iloc[transfer].decision_id)
    augmented = pd.concat([metadata.security_id, pd.Series(["new-security"] * 20)], ignore_index=True)
    assert_array_equal(owner.security_transfer_mask(augmented)[:12], expected)
    pd.testing.assert_frame_equal(metadata, original)


@pytest.mark.parametrize("transfer", [False, True])
def test_empty_causally_matured_fit_scope_fails_closed(transfer: bool) -> None:
    metadata = pd.DataFrame({
        "session_date_et": [date(2024, 1, 2)], "feature_eligible": [True],
        "fixed_horizon_supervision_available": [True], "research_label_mature_at": [CUTOFF],
        "security_holdout": [False],
    })
    with pytest.raises(DataReadinessError, match="no causally matured"):
        owner.fit_indices(metadata, (date(2024, 1, 2),), CUTOFF, transfer=transfer)


@pytest.mark.parametrize("identity", [None, "", " security-1", "security-1 ", 123, True])
def test_noncanonical_security_ids_rejected(identity: object) -> None:
    with pytest.raises(DataReadinessError, match="canonical identities"):
        owner.security_transfer_mask(pd.Series([identity]))


@pytest.mark.parametrize("fraction", [0.0, 1.0, -0.1, float("nan")])
def test_invalid_holdout_fraction_rejected(fraction: float) -> None:
    with pytest.raises(DataReadinessError, match="fraction"):
        owner.security_transfer_mask(pd.Series(["security-1"]), fraction=fraction)


def test_date_weights_have_equal_totals_mean_one_and_preserve_order() -> None:
    sessions = pd.Series([date(2024, 1, 2)] + [date(2024, 1, 3)] * 2 + [date(2024, 1, 4)] * 3,
        index=[9, 2, 7, 4, 3, 8])
    original = sessions.copy(deep=True)
    weights = owner.date_balanced_weights(sessions)
    assert weights.dtype == np.float64
    assert (weights > 0).all()
    assert_allclose(weights, [2, 1, 1, 2 / 3, 2 / 3, 2 / 3])
    assert weights.mean() == pytest.approx(1.0)
    totals = pd.Series(weights, index=sessions.index).groupby(sessions).sum()
    assert_allclose(totals, [2, 2, 2])
    order = [5, 1, 3, 0, 4, 2]
    assert_allclose(owner.date_balanced_weights(sessions.iloc[order]), weights[order])
    assert_allclose(owner.date_balanced_weights(sessions.iloc[:1]), [1])
    pd.testing.assert_series_equal(sessions, original)


@pytest.mark.parametrize("sessions", [pd.Series([], dtype=object), pd.Series([date(2024, 1, 2), None])])
def test_unobserved_weight_sessions_rejected(sessions: pd.Series) -> None:
    with pytest.raises(DataReadinessError, match="observed sessions"):
        owner.date_balanced_weights(sessions)


def test_diagnostics_retain_unknown_scored_rows_without_portfolio_claim() -> None:
    predictions = pd.DataFrame({
        "session_date_et": [date(2024, 1, 2)] * 3 + [date(2024, 1, 3)] * 5,
        "predicted_excess_return": [1.0, 4.0, 8.0, 2.0, 10.0, 20.0, np.nan, np.nan],
        TARGET: [2.0, 3.0, 6.0, 0.0, np.nan, 999.0, 7.0, np.nan],
        "fixed_horizon_supervision_available": [True, True, True, True, True, False, True, False],
    })
    original = predictions.copy(deep=True)
    result = owner.regression_diagnostics(predictions, TARGET)
    assert result == {
        "rows": 8, "scored_rows": 6, "known_scored_outcomes": 4, "unknown_scored_outcomes": 2,
        "portfolio_evaluated": False, "mse": pytest.approx(3.0), "mae": pytest.approx(5 / 3),
        "zero_excess_baseline_mse": pytest.approx(49 / 6),
        "mean_session_rank_correlation": pytest.approx(1.0), "rank_correlation_sessions": 1,
    }
    poisoned = predictions.copy(deep=True)
    poisoned.loc[5, TARGET] = -1e15
    poisoned.loc[[4, 5], "predicted_excess_return"] = [1e15, -1e15]
    poisoned.loc[6, TARGET] = 1e15
    assert owner.regression_diagnostics(poisoned, TARGET) == result
    pd.testing.assert_frame_equal(predictions, original)


@pytest.mark.parametrize("rows", [0, 3])
def test_diagnostics_without_known_outcomes_remain_explicit(rows: int) -> None:
    predictions = pd.DataFrame({
        "session_date_et": [date(2024, 1, 2)] * rows,
        "predicted_excess_return": pd.Series([1.0] * rows, dtype=float),
        TARGET: pd.Series([np.nan] * rows, dtype=float),
        "fixed_horizon_supervision_available": pd.Series([False] * rows, dtype=bool),
    })
    assert owner.regression_diagnostics(predictions, TARGET) == {
        "rows": rows, "scored_rows": rows, "known_scored_outcomes": 0, "unknown_scored_outcomes": rows,
        "portfolio_evaluated": False, "mse": None, "mae": None, "zero_excess_baseline_mse": None,
        "mean_session_rank_correlation": None, "rank_correlation_sessions": 0,
    }


def test_constant_session_predictions_do_not_claim_rank_correlation() -> None:
    predictions = pd.DataFrame({
        "session_date_et": [date(2024, 1, 2)] * 2,
        "predicted_excess_return": [1.0, 1.0], TARGET: [0.0, 2.0],
        "fixed_horizon_supervision_available": [True, True],
    })
    result = owner.regression_diagnostics(predictions, TARGET)
    assert result["mse"] == pytest.approx(1.0)
    assert result["mean_session_rank_correlation"] is None
    assert result["rank_correlation_sessions"] == 0
    assert result["portfolio_evaluated"] is False
