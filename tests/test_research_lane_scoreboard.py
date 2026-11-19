"""Unit tests on synthetic arrays: rank correlation, Newey-West, top-tenth spread and recovery."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import spearmanr

from market_predictor.research_lane.labels import HorizonLabels
from market_predictor.research_lane.scoreboard import (
    build_scoreboard,
    daily_rank_correlation,
    newey_west,
    rebalance_rows,
    scored_cells,
    sector_neutral_rank_correlation,
    simple_t,
    top_tenth_spread,
)


def test_daily_rank_correlation_matches_scipy_spearman() -> None:
    rng = np.random.default_rng(3)
    predictor = rng.normal(size=(6, 80))
    predictor[:, :5] = np.round(predictor[:, :5])  # ties
    forward = 0.3 * predictor + rng.normal(size=predictor.shape)
    forward[rng.random(forward.shape) < 0.1] = np.nan
    eligible = rng.random(predictor.shape) > 0.1
    cells = scored_cells(predictor, forward, eligible, np.ones(6, dtype=bool), minimum=50)
    result = daily_rank_correlation(predictor, forward, cells)
    for row in range(6):
        chosen = cells[row]
        assert chosen.sum() >= 50
        expected = spearmanr(predictor[row, chosen], forward[row, chosen]).statistic
        assert result[row] == pytest.approx(expected, rel=1e-12)
    few = scored_cells(predictor, forward, eligible, np.ones(6, dtype=bool), minimum=79)
    assert not few.any()


def test_newey_west_matches_hand_calculation() -> None:
    result = newey_west([1.0, 2.0, 3.0, 4.0, 10.0], lag=1)
    # centered: -3,-2,-1,0,6; gamma0 = 50/5 = 10; gamma1 = 8/5 = 1.6; S = 10 + 2*0.5*1.6 = 11.6
    assert result["mean"] == pytest.approx(4.0)
    assert result["standard_error"] == pytest.approx(np.sqrt(11.6 / 5.0))
    assert result["t"] == pytest.approx(4.0 / np.sqrt(11.6 / 5.0))
    lag0 = newey_west([1.0, 2.0, 3.0, 4.0, 10.0], lag=0)
    assert lag0["standard_error"] == pytest.approx(np.sqrt(10.0 / 5.0))
    assert newey_west([np.nan, 1.0], lag=2)["count"] == 1
    plain = simple_t([1.0, 2.0, 3.0])
    assert plain["t"] == pytest.approx(2.0 / (1.0 / np.sqrt(3.0)))


def test_top_tenth_spread_and_turnover_on_tiny_example() -> None:
    forward = np.tile(np.arange(20, dtype=float) / 100.0, (3, 1))
    predictor = np.tile(np.arange(20, dtype=float), (3, 1))
    predictor[2, 19] = -1.0  # on the last rebalance column 19 drops out; 17 joins
    cells = np.ones((3, 20), dtype=bool)
    cells[1] = False  # row 1 is not a rebalance row anyway
    top = top_tenth_spread(predictor, forward, cells, rebalance_rows(0, 3, 2))
    np.testing.assert_array_equal(top.rows, [0, 2])
    np.testing.assert_array_equal(top.top_counts, [2, 2])
    # Row 0: top = columns 19, 18 -> mean 0.185; all-mean 0.095
    assert top.spreads[0] == pytest.approx(0.185 - 0.095)
    # Row 2: top = columns 18, 17 -> mean 0.175
    assert top.spreads[1] == pytest.approx(0.175 - 0.095)
    assert np.isnan(top.turnovers[0])
    assert top.turnovers[1] == pytest.approx(0.5)


def test_recovery_signal_is_found_and_noise_is_not() -> None:
    rng = np.random.default_rng(11)
    dates, stocks, horizon = 300, 200, 5
    predictor = rng.normal(size=(dates, stocks))
    noise = rng.normal(size=(dates, stocks))
    eligible = np.ones((dates, stocks), dtype=bool)
    sector = np.tile(np.arange(stocks, dtype=np.int16) % 5, (dates, 1))
    sessions = np.datetime64("2020-01-01", "D") + np.arange(dates)
    zeros = np.zeros(dates)
    for strength, check in ((0.5, "signal"), (0.0, "noise")):
        forward = strength * predictor + noise
        labels = HorizonLabels(horizon=horizon, stock=forward, reason=np.zeros(forward.shape, dtype=np.int8),
                               spy=zeros, qqq=zeros)
        board = build_scoreboard({"x": predictor}, [labels], eligible=eligible, sessions=sessions, sector=sector,
                                 first_row=0)
        entry = board["predictors"]["x"]["5"]
        mean, t = entry["rank_correlation"]["mean"], entry["rank_correlation"]["t"]
        if check == "signal":
            assert mean > 0.3 and t > 20
            assert entry["sector_neutral_rank_correlation"]["mean"] > 0.3
            assert entry["top_tenth"]["mean_spread_per_period_fraction"] > 0.5
            assert entry["share_of_years_positive"] == 1.0
        else:
            assert abs(mean) < 0.02 and abs(t) < 3
            assert abs(entry["top_tenth"]["mean_spread_per_period_fraction"]) < 0.1
        assert entry["rank_correlation"]["count"] == dates
        assert entry["rank_correlation"]["non_overlapping"]["count"] == dates // horizon
        assert entry["top_tenth"]["rebalance_periods"] == dates // horizon


def test_sector_neutral_removes_pure_sector_effect() -> None:
    rng = np.random.default_rng(5)
    dates, stocks = 40, 120
    sector = np.tile(np.arange(stocks, dtype=np.int16) % 4, (dates, 1))
    predictor = sector.astype(float) + 0.001 * rng.normal(size=(dates, stocks))
    forward = sector.astype(float) * 0.05 + rng.normal(size=(dates, stocks)) * 0.01
    cells = np.ones((dates, stocks), dtype=bool)
    raw = daily_rank_correlation(predictor, forward, cells)
    neutral = sector_neutral_rank_correlation(predictor, forward, sector, cells)
    assert np.nanmean(raw) > 0.8
    assert abs(np.nanmean(neutral)) < 0.1


def test_universe_facts_count_missing_reasons_and_benchmark_gap() -> None:
    dates, stocks = 10, 60
    stock = np.full((dates, stocks), 0.02)
    reason = np.zeros((dates, stocks), dtype=np.int8)
    stock[0, :3] = np.nan
    reason[0, :3] = 2
    stock[8:] = np.nan
    reason[8:] = 1
    labels = HorizonLabels(horizon=2, stock=stock, reason=reason, spy=np.full(dates, 0.01), qqq=np.full(dates, 0.03))
    sessions = np.arange("2021-01-04", "2021-01-14", dtype="datetime64[D]")
    board = build_scoreboard({}, [labels], eligible=np.ones((dates, stocks), dtype=bool), sessions=sessions,
                             sector=np.zeros((dates, stocks), dtype=np.int16), first_row=1)
    facts = board["universe"]["2"]
    assert facts["eligible_decisions_stock_sessions"] == 9 * stocks  # row 0 precedes the first decision row
    assert facts["missing_label_stock_sessions_by_reason"] == {
        "window_runs_past_development_end": 2 * stocks, "entry_bar_missing": 0, "exit_bar_missing": 0}
    assert facts["known_label_stock_sessions"] == 7 * stocks
    assert facts["average_stock_minus_spy"]["mean"] == pytest.approx(0.01)
    assert facts["average_stock_minus_qqq"]["mean"] == pytest.approx(-0.01)
    assert facts["decision_dates_with_average"] == 7
