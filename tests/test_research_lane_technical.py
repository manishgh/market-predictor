"""Unit tests on synthetic panels: technical indicators, feature sets and walk-forward fits (not real-data evidence)."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from market_predictor.research_lane.fitted import (
    date_weights,
    scored_folds,
    session_years,
    target_ranks,
    training_rows,
    walk_forward,
)
from market_predictor.research_lane.indicators import (
    INDICATOR_NAMES,
    build_feature_sets,
    compute_indicators,
    feature_set_members,
    sector_centered_rank,
    smooth,
)
from market_predictor.research_lane.labels import HORIZONS, HorizonLabels, first_decision_row, research_labels
from market_predictor.research_lane.panel import ResearchPanel
from market_predictor.research_lane.run import fitted_summary
from market_predictor.research_lane.scoreboard import score_predictor
from tests.support.research_lane_panel import synthetic_panel
from tests.test_research_lane_predictors import _poisoned

CUTS = (260, 330, 451, 520, 619)


@pytest.fixture(scope="module")
def panel() -> ResearchPanel:
    return synthetic_panel()


@pytest.fixture(scope="module")
def full(panel: ResearchPanel) -> dict[str, np.ndarray]:
    return compute_indicators(panel)


@pytest.fixture(scope="module")
def full_features(panel: ResearchPanel) -> dict[str, np.ndarray]:
    return build_feature_sets(panel).features


def _day(panel: ResearchPanel, row: int) -> object:
    return pd.Timestamp(panel.sessions[row]).date()


def test_indicator_list_and_feature_set_membership() -> None:
    assert len(INDICATOR_NAMES) == len(set(INDICATOR_NAMES)) == 51
    sets = feature_set_members()
    assert [len(sets[name]) for name in ("set_12", "set_all", "set_all_sector_neutral")] == [12, 63, 126]


@pytest.mark.parametrize("cut", CUTS)
def test_truncated_history_gives_identical_values_for_every_indicator(
        panel: ResearchPanel, full: dict[str, np.ndarray], cut: int) -> None:
    truncated = compute_indicators(panel.truncated(_day(panel, cut)))
    assert tuple(truncated) == INDICATOR_NAMES
    for name, values in truncated.items():
        assert values.shape[0] == cut + 1
        np.testing.assert_allclose(values, full[name][:cut + 1], rtol=1e-12, atol=0, equal_nan=True, err_msg=name)
        assert np.isfinite(values[-1]).any(), name


@pytest.mark.parametrize("cut", (260, 451, 619))
def test_truncated_history_gives_identical_feature_sets(
        panel: ResearchPanel, full_features: dict[str, np.ndarray], cut: int) -> None:
    truncated = build_feature_sets(panel.truncated(_day(panel, cut))).features
    assert set(truncated) == set(full_features)
    for name, values in truncated.items():
        np.testing.assert_array_equal(values, full_features[name][:cut + 1], err_msg=name)
        assert np.isfinite(values[-1]).any(), name


@pytest.mark.parametrize("cut", (300, 480))
def test_future_poison_changes_no_indicator_or_feature_at_or_before_cut(
        panel: ResearchPanel, full: dict[str, np.ndarray], full_features: dict[str, np.ndarray], cut: int) -> None:
    poisoned_panel = _poisoned(panel, cut, seed=cut)
    poisoned = compute_indicators(poisoned_panel)
    for name, values in poisoned.items():
        np.testing.assert_array_equal(values[:cut + 1], full[name][:cut + 1], err_msg=name)
        assert not np.array_equal(values[cut + 1:], full[name][cut + 1:], equal_nan=True), name
    features = build_feature_sets(poisoned_panel).features
    for name, values in features.items():
        np.testing.assert_array_equal(values[:cut + 1], full_features[name][:cut + 1], err_msg=name)


def _wilder(values: np.ndarray, window: int) -> np.ndarray:
    out = np.full(values.shape, np.nan)
    start = int(np.flatnonzero(np.isfinite(values))[0])
    average = values[start:start + window].mean()
    out[start + window - 1] = average
    for row in range(start + window, len(values)):
        average = (average * (window - 1) + values[row]) / window
        out[row] = average
    return out


def test_spot_arithmetic_against_reference_values(panel: ResearchPanel, full: dict[str, np.ndarray]) -> None:
    column, t = 0, 400
    close = panel.close[:, column].astype(np.float64)
    high = panel.high[:, column].astype(np.float64)
    low = panel.low[:, column].astype(np.float64)
    volume = panel.volume[:, column].astype(np.float64)
    change = np.concatenate([[np.nan], np.diff(close)])
    gain, loss = np.where(np.isfinite(change), np.maximum(change, 0), np.nan), np.where(
        np.isfinite(change), np.maximum(-change, 0), np.nan)
    average_gain, average_loss = _wilder(gain, 14), _wilder(loss, 14)
    assert np.isnan(full["rsi_14"][13, column]) and np.isfinite(full["rsi_14"][14, column])
    assert full["rsi_14"][t, column] == pytest.approx(100 - 100 / (1 + average_gain[t] / average_loss[t]), rel=1e-9)
    lowest = pd.Series(low).rolling(14).min().to_numpy()
    highest = pd.Series(high).rolling(14).max().to_numpy()
    assert full["stochastic_k_14"][t, column] == pytest.approx((close[t] - lowest[t]) / (highest[t] - lowest[t]))
    window = close[t - 19:t + 1]
    middle, deviation = window.mean(), window.std(ddof=0)
    assert full["bollinger_pct_b_20"][t, column] == pytest.approx(
        (close[t] - (middle - 2 * deviation)) / (4 * deviation), rel=1e-8)
    previous = np.concatenate([[np.nan], close[:-1]])
    true_range = np.fmax(high - low, np.fmax(np.abs(high - previous), np.abs(low - previous)))
    true_range[0] = np.nan
    assert full["atr_pct_14"][t, column] == pytest.approx(_wilder(true_range, 14)[t] / close[t], rel=1e-9)
    rows = slice(t - 19, t + 1)
    multiplier = ((close[rows] - low[rows]) - (high[rows] - close[rows])) / (high[rows] - low[rows])
    assert full["cmf_20"][t, column] == pytest.approx((multiplier * volume[rows]).sum() / volume[rows].sum(), rel=1e-9)
    # Streak: count consecutive same-sign closes back from t (cap 10), needs 10 finite returns.
    returns = close[1:] / close[:-1] - 1.0
    for row in (15, 200, 400, 555):
        recent = returns[:row][::-1][:10]
        sign = np.sign(recent[0])
        expected = 0 if sign == 0 else sign * next((i for i, value in enumerate(recent) if np.sign(value) != sign), 10)
        assert full["streak_length"][row, column] == expected
    assert np.isnan(full["streak_length"][9, column]) and np.isfinite(full["streak_length"][10, column])
    # Column 3 lacks bars on rows 100..103: the streak needs 10 finite returns again (rows 105..113 missing).
    assert np.isnan(full["streak_length"][100:114, 3]).all() and np.isfinite(full["streak_length"][114, 3])
    rising = panel.close.copy()
    rising[:, 0] = np.linspace(10.0, 20.0, rising.shape[0], dtype=np.float32)
    assert compute_indicators(replace(panel, close=rising))["streak_length"][50, 0] == 10


def test_smoother_starts_after_full_window_and_restarts_after_gap() -> None:
    values = np.arange(1.0, 31.0).reshape(-1, 1)
    values[15, 0] = np.nan
    result = smooth(values, 5, 0.5)
    assert np.isnan(result[:4, 0]).all() and result[4, 0] == pytest.approx(3.0)
    assert result[5, 0] == pytest.approx(3.0 + 0.5 * (6.0 - 3.0))
    assert np.isnan(result[15:20, 0]).all()
    assert result[20, 0] == pytest.approx(values[16:21, 0].mean())


def test_sector_centered_rank_needs_five_stocks() -> None:
    values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 20.0, 30.0, 40.0]])
    sector = np.array([[0, 0, 0, 0, 0, 1, 1, 1, -1]], dtype=np.int16)
    result = sector_centered_rank(values, np.ones(values.shape, dtype=bool), sector)
    np.testing.assert_allclose(result, [[-0.5, -0.25, 0.0, 0.25, 0.5, np.nan, np.nan, np.nan, np.nan]], equal_nan=True)


def test_training_rows_are_purged_for_every_horizon_and_fold() -> None:
    sessions = pd.bdate_range("2018-05-29", "2024-05-28").to_numpy().astype("datetime64[D]")
    first_row = int(np.searchsorted(sessions, np.datetime64("2019-07-09")))
    folds = scored_folds(sessions)
    assert [fold.year for fold in folds] == [2021, 2022, 2023, 2024]
    for fold in folds:
        assert session_years(sessions[fold.test_start:fold.test_stop]).tolist() == [fold.year] * (
            fold.test_stop - fold.test_start)
        for horizon in HORIZONS:
            rows = training_rows(first_row, fold.test_start, horizon)
            assert rows.size and rows[0] == first_row and np.all(np.diff(rows) == 5)
            assert np.all(rows + horizon < fold.test_start)
            assert rows[-1] + 5 + horizon >= fold.test_start


def test_date_weights_give_each_date_equal_total_with_mean_one() -> None:
    cells = np.array([[True, True, False, False], [True, True, True, True], [False] * 4])
    weights = date_weights(cells)
    np.testing.assert_allclose(weights, [1.5, 1.5, 0.75, 0.75, 0.75, 0.75])
    assert weights.mean() == pytest.approx(1.0)


def test_ridge_fold_is_unchanged_when_prices_from_the_first_test_session_are_garbage() -> None:
    panel = synthetic_panel(sessions=900, securities=40, seed=11)
    first_row = first_decision_row(panel)
    fold = scored_folds(panel.sessions, (2021,))[0]
    poisoned = _poisoned(panel, fold.test_start - 1, seed=5)
    clean_sets, dirty_sets = build_feature_sets(panel), build_feature_sets(poisoned)
    names = clean_sets.sets["set_all"]
    for horizon in HORIZONS:
        fits = []
        for source, sets in ((panel, clean_sets), (poisoned, dirty_sets)):
            target = target_ranks(research_labels(source, horizon).stock, source.eligible)
            fits.append(walk_forward(sets.features, names, target, eligible=source.eligible, sessions=source.sessions,
                                     first_row=first_row, horizon=horizon, learner="ridge", folds=[fold],
                                     keep_models=True))
        clean, dirty = fits[0].models[0], fits[1].models[0]
        assert clean.coefficients is not None and dirty.coefficients is not None
        assert np.abs(clean.coefficients).max() > 0
        np.testing.assert_array_equal(clean.coefficients, dirty.coefficients)
        assert clean.intercept == dirty.intercept
        assert fits[0].folds[0]["training_rows_stock_sessions"] > 0
        assert not np.array_equal(fits[0].scores, fits[1].scores, equal_nan=True)


def _planted(signal: bool, seed: int = 3) -> tuple[dict[str, np.ndarray], HorizonLabels, np.ndarray]:
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range("2018-05-29", "2024-05-28").to_numpy().astype("datetime64[D]")
    shape = (len(sessions), 120)
    features = {f"x{index}": rng.uniform(-0.5, 0.5, shape).astype(np.float32) for index in range(3)}
    noise = rng.normal(0.0, 0.01, shape)
    stock = noise + (0.03 * (np.abs(features["x0"].astype(np.float64)) - 0.25) if signal else 0.0)
    zeros = np.zeros(len(sessions))
    labels = HorizonLabels(horizon=5, stock=stock, reason=np.zeros(shape, dtype=np.int8), spy=zeros, qqq=zeros)
    return features, labels, sessions


def _oos_rank_correlation(features: dict[str, np.ndarray], labels: HorizonLabels, sessions: np.ndarray,
                          learner: str) -> tuple[float, np.ndarray]:
    eligible = np.ones(labels.stock.shape, dtype=bool)
    first_row = int(np.searchsorted(sessions, np.datetime64("2019-07-09")))
    folds = scored_folds(sessions)
    result = walk_forward(features, tuple(features), target_ranks(labels.stock, eligible), eligible=eligible,
                          sessions=sessions, first_row=first_row, horizon=labels.horizon, learner=learner, folds=folds)
    decision_rows = np.arange(len(sessions)) >= folds[0].test_start
    entry = score_predictor(result.scores, labels, eligible=eligible, decision_rows=decision_rows,
                            sector=np.zeros(labels.stock.shape, dtype=np.int16), years=session_years(sessions),
                            first_row=folds[0].test_start)
    assert np.isnan(result.scores[:folds[0].test_start]).all()
    return float(entry["rank_correlation"]["mean"]), result.scores


def test_trees_recover_planted_nonlinear_signal_out_of_sample_and_noise_gives_zero() -> None:
    features, labels, sessions = _planted(signal=True)
    trees, scores = _oos_rank_correlation(features, labels, sessions, "xgboost")
    ridge, _ = _oos_rank_correlation(features, labels, sessions, "ridge")
    assert trees > 0.2
    assert abs(ridge) < 0.05  # a symmetric signal has no linear component
    _, repeat = _oos_rank_correlation(features, labels, sessions, "xgboost")
    np.testing.assert_array_equal(scores, repeat)  # deterministic fitted run
    noise_features, noise_labels, _ = _planted(signal=False, seed=4)
    noise, _ = _oos_rank_correlation(noise_features, noise_labels, sessions, "xgboost")
    assert abs(noise) < 0.03


def test_fitted_summary_reading_rule() -> None:
    specs = {f"s{index}": {"rank_correlation_mean": value, "positive_years": years, "learner": "ridge",
                           "feature_set": "set_12", "horizon_sessions": 5}
             for index, (value, years) in enumerate([(0.04, 4), (0.03, 3), (0.01, 2)])}
    summary = fitted_summary(specs)
    assert summary["median_mean_rank_correlation"] == pytest.approx(0.03)
    assert summary["specifications_positive_in_at_least_3_of_4_years"] == 2
    assert summary["rule_met"] is True
    assert "median is at or above 0.03" in summary["reading_rule"]
