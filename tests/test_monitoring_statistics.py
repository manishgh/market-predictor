from __future__ import annotations

import math
from datetime import date

import pytest

from market_predictor.governance.outcomes.monitoring_statistics import (
    exchange_session_ordinals,
    mean_statistics,
    ratio_statistics,
)


def test_hansen_hodrick_mean_matches_hand_covariance() -> None:
    # Centered [-2,-1,1,2]: squared sum 10, lag-one product sum 3.
    estimate = mean_statistics([0, 1, 2, 3], [1., 2., 4., 5.], horizon_sessions=2)
    assert estimate.mean == 3.
    assert estimate.standard_error == 1.  # sqrt((10 + 2*3) / 4**2)
    assert estimate.t_statistic == 3.
    assert estimate.method == "hansen_hodrick"
    assert estimate.sample_count == 4


def test_missing_sessions_are_not_compressed_into_adjacent_pairs() -> None:
    estimate = mean_statistics([0, 2, 4, 6], [1., 2., 4., 5.], horizon_sessions=2)
    assert estimate.standard_error == pytest.approx(math.sqrt(10 / 16))
    assert estimate.method == "hansen_hodrick"
    # Order does not alter the session lag or associated value.
    reordered = mean_statistics([6, 2, 0, 4], [5., 2., 1., 4.], horizon_sessions=2)
    assert reordered == estimate


def test_negative_hh_uses_bartlett_newey_west_at_twice_horizon() -> None:
    # HH: (6 + 2*(-4))/9 < 0. NW at lag 4:
    # (6 + 2*(4/5)*(-4) + 2*(3/5)*1)/9 = 4/45.
    estimate = mean_statistics([0, 1, 2], [2., -1., 2.], horizon_sessions=2)
    assert estimate.mean == 1.
    assert estimate.method == "newey_west"
    assert estimate.standard_error == pytest.approx(math.sqrt(4 / 45))
    assert estimate.t_statistic == pytest.approx(math.sqrt(45 / 4))


@pytest.mark.parametrize("values", [[0., 0., 0.], [2., 2., 2.], [-2.]])
def test_zero_variance_has_no_infinite_t_statistic(values: list[float]) -> None:
    estimate = mean_statistics(list(range(len(values))), values, horizon_sessions=10)
    assert estimate.standard_error == 0.
    assert estimate.t_statistic is None


def test_ratio_uses_actual_durations_and_ratio_residual_covariance() -> None:
    estimate = ratio_statistics([0, 1], [.1, .1], [1., 9.], horizon_sessions=1)
    assert estimate.mean == pytest.approx(.02)  # .2 / 10, not mean(.1, .1/9)
    assert estimate.standard_error == pytest.approx(math.sqrt(2 * .08**2 / 10**2))
    assert estimate.t_statistic == pytest.approx(.02 / math.sqrt(.000128))
    assert estimate.sample_count == 2


def test_ratio_preserves_zero_exposure_and_session_gaps() -> None:
    estimate = ratio_statistics([0, 1, 2], [.1, 0., .1], [1., 0., 9.], horizon_sessions=2)
    omitted = ratio_statistics([0, 2], [.1, .1], [1., 9.], horizon_sessions=2)
    assert estimate.mean == omitted.mean
    assert estimate.standard_error == omitted.standard_error
    assert estimate.sample_count == 3
    assert omitted.sample_count == 2


def test_ratio_constant_rate_has_no_t_statistic() -> None:
    estimate = ratio_statistics([0, 1], [.125, .25], [1., 2.], horizon_sessions=10)
    assert estimate.mean == .125
    assert estimate.standard_error == 0.
    assert estimate.t_statistic is None


def test_session_ordinals_use_exchange_sessions_not_calendar_days() -> None:
    thursday, monday, wednesday = exchange_session_ordinals(
        [date(2026, 7, 2), "2026-07-06", "2026-07-08"])
    assert monday - thursday == 1  # Friday July 3 is a holiday.
    assert wednesday - monday == 2  # Absent Tuesday remains a gap.
    with pytest.raises(ValueError, match="XNYS"):
        exchange_session_ordinals(["2026-07-03"])
    with pytest.raises(ValueError, match="duplicate"):
        exchange_session_ordinals(["2026-07-02", "2026-07-02"])


@pytest.mark.parametrize(("ordinals", "values", "horizon"), [
    ([], [], 10), ([0], [1., 2.], 10), ([0, 0], [1., 2.], 10),
    ([0], [math.nan], 10), ([0], [math.inf], 10), ([0], [True], 10),
    ([False], [1.], 10), ([0.5], [1.], 10), ([0], [1.], 0),
    ([0], [1.], True), ([0], [1.], 1.5),
])
def test_mean_rejects_invalid_inputs(ordinals: list[int], values: list[float], horizon: int) -> None:
    with pytest.raises(ValueError):
        mean_statistics(ordinals, values, horizon_sessions=horizon)


@pytest.mark.parametrize(("numerators", "denominators"), [
    ([1.], [0.]), ([0.], [0.]), ([1.], [-1.]), ([1.], [math.inf]),
    ([1.], [True]), ([1.], []),
])
def test_ratio_rejects_invalid_exposure(numerators: list[float], denominators: list[float]) -> None:
    with pytest.raises(ValueError):
        ratio_statistics([0], numerators, denominators, horizon_sessions=10)


def test_unrepresentable_uncertainty_is_rejected_not_returned_as_infinity() -> None:
    with pytest.raises(ValueError):
        mean_statistics([0, 1], [1e308, -1e308], horizon_sessions=1)
