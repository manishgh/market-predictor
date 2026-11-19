"""Unit tests on synthetic panels: predictor causality, windows and the frozen sign table."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from market_predictor.research_lane.labels import HORIZONS, research_labels
from market_predictor.research_lane.panel import ResearchPanel
from market_predictor.research_lane.predictors import (
    COMPOSITE_GROUPS,
    EXPECTED_SIGNS,
    PREDICTOR_NAMES,
    centered_rank,
    compute_composites,
    compute_predictors,
    minimum_present,
)
from tests.support.research_lane_panel import synthetic_panel

CUTS = (260, 330, 451, 520, 619)


@pytest.fixture(scope="module")
def panel() -> ResearchPanel:
    return synthetic_panel()


@pytest.fixture(scope="module")
def full(panel: ResearchPanel) -> dict[str, np.ndarray]:
    raw = compute_predictors(panel)
    return {**raw, **compute_composites(raw, panel.eligible)}


def _all_outputs(panel: ResearchPanel) -> dict[str, np.ndarray]:
    raw = compute_predictors(panel)
    return {**raw, **compute_composites(raw, panel.eligible)}


def _day(panel: ResearchPanel, row: int) -> object:
    return pd.Timestamp(panel.sessions[row]).date()


def test_expected_sign_table_is_frozen() -> None:
    assert dict(EXPECTED_SIGNS) == {
        "mom_12_1": 1, "mom_6_1": 1, "residual_mom_12_1": 1, "sector_mom_6m": 1, "high_52w_proximity": 1,
        "reversal_5d": -1, "reversal_21d": -1, "idio_vol_60d": -1, "beta_252d": -1, "max_daily_return_21d": -1,
        "amihud_illiquidity_60d": 1, "volume_ratio_5_60": 1,
    }
    assert COMPOSITE_GROUPS["composite_trend"] == PREDICTOR_NAMES[:5]
    assert COMPOSITE_GROUPS["composite_reversal"] == ("reversal_5d", "reversal_21d")
    assert COMPOSITE_GROUPS["composite_risk_liquidity"] == PREDICTOR_NAMES[7:]
    assert [minimum_present(size) for size in (12, 5, 2)] == [8, 4, 2]


@pytest.mark.parametrize("cut", CUTS)
def test_truncated_history_gives_identical_values_for_every_predictor_and_composite(
        panel: ResearchPanel, full: dict[str, np.ndarray], cut: int) -> None:
    truncated = _all_outputs(panel.truncated(_day(panel, cut)))
    assert set(truncated) == set(PREDICTOR_NAMES) | set(COMPOSITE_GROUPS)
    for name, values in truncated.items():
        assert values.shape[0] == cut + 1
        np.testing.assert_allclose(values, full[name][:cut + 1], rtol=1e-12, atol=0, equal_nan=True, err_msg=name)
        assert np.isfinite(values[-1]).any(), name


def _poisoned(panel: ResearchPanel, cut: int, seed: int) -> ResearchPanel:
    rng = np.random.default_rng(seed)
    later = slice(cut + 1, None)

    def garbage(values: np.ndarray) -> np.ndarray:
        result = values.copy()
        noise = rng.uniform(-1e6, 1e9, result[later].shape).astype(result.dtype)
        noise[rng.random(noise.shape) < 0.1] = np.nan
        result[later] = noise
        return result

    eligible = panel.eligible.copy()
    eligible[later] = rng.random(eligible[later].shape) > 0.5
    sector = panel.sector.copy()
    sector[later] = rng.integers(0, 4, sector[later].shape)
    return replace(panel, open=garbage(panel.open), high=garbage(panel.high), low=garbage(panel.low),
                   close=garbage(panel.close), volume=garbage(panel.volume), spy_open=garbage(panel.spy_open),
                   spy_close=garbage(panel.spy_close), qqq_open=garbage(panel.qqq_open),
                   qqq_close=garbage(panel.qqq_close), eligible=eligible, sector=sector)


@pytest.mark.parametrize("cut", (300, 480))
def test_future_poison_changes_no_predictor_at_or_before_cut(panel: ResearchPanel, full: dict[str, np.ndarray],
                                                             cut: int) -> None:
    poisoned = _all_outputs(_poisoned(panel, cut, seed=cut))
    for name, values in poisoned.items():
        np.testing.assert_array_equal(values[:cut + 1], full[name][:cut + 1], err_msg=name)
        assert not np.array_equal(values[cut + 1:], full[name][cut + 1:], equal_nan=True), name


@pytest.mark.parametrize("cut", (300, 480))
def test_future_poison_changes_labels_only_when_window_passes_cut(panel: ResearchPanel, cut: int) -> None:
    poisoned = _poisoned(panel, cut, seed=cut + 1)
    for horizon in HORIZONS:
        clean, dirty = research_labels(panel, horizon), research_labels(poisoned, horizon)
        safe = cut - horizon + 1  # decisions t with t + h <= cut
        for name in ("stock", "spy", "qqq"):
            np.testing.assert_array_equal(getattr(dirty, name)[:safe], getattr(clean, name)[:safe])
        np.testing.assert_array_equal(dirty.reason[:safe], clean.reason[:safe])
        assert not np.array_equal(dirty.stock[safe:cut + 1], clean.stock[safe:cut + 1], equal_nan=True)


def test_windows_require_complete_history(panel: ResearchPanel, full: dict[str, np.ndarray]) -> None:
    close = panel.close.astype(np.float64)
    # mom_12_1 needs close[t-252]; first defined row is 252.
    assert np.isnan(full["mom_12_1"][251, 0]) and np.isfinite(full["mom_12_1"][252, 0])
    assert full["mom_12_1"][252, 0] == pytest.approx(close[231, 0] / close[0, 0] - 1.0, rel=1e-12)
    assert full["reversal_5d"][10, 0] == pytest.approx(close[10, 0] / close[5, 0] - 1.0, rel=1e-12)
    # Column 3 lacks bars on rows 100..103: values needing those bars are NaN, never filled.
    assert np.isnan(full["reversal_21d"][100:104, 3]).all() and np.isnan(full["reversal_21d"][121:125, 3]).all()
    assert np.isfinite(full["reversal_21d"][104, 3]) and np.isfinite(full["reversal_21d"][125, 3])
    assert np.isnan(full["max_daily_return_21d"][100:125, 3]).all()
    assert np.isnan(full["idio_vol_60d"][100:164, 3]).all() and np.isfinite(full["idio_vol_60d"][164, 3])
    # Residual momentum uses returns t-252..t-21, so it first exists at row 253.
    assert np.isnan(full["residual_mom_12_1"][252, 0]) and np.isfinite(full["residual_mom_12_1"][253, 0])
    assert np.isnan(full["beta_252d"][251, 0]) and np.isfinite(full["beta_252d"][252, 0])


def test_regression_predictors_match_direct_least_squares(panel: ResearchPanel,
                                                          full: dict[str, np.ndarray]) -> None:
    close = panel.close.astype(np.float64)
    spy = panel.spy_close.astype(np.float64)
    returns = close[1:] / close[:-1] - 1.0
    market = spy[1:] / spy[:-1] - 1.0
    t, column = 400, 2
    r, m = returns[t - 252:t, column], market[t - 252:t]  # daily returns of sessions t-251..t
    slope = np.polyfit(m, r, 1)[0]
    assert full["beta_252d"][t, column] == pytest.approx(slope, rel=1e-8)
    r60, m60 = returns[t - 60:t, column], market[t - 60:t]
    slope60, intercept60 = np.polyfit(m60, r60, 1)
    residual = r60 - slope60 * m60 - intercept60
    assert full["idio_vol_60d"][t, column] == pytest.approx(np.sqrt(residual @ residual / 58.0), rel=1e-7)
    rl, ml = returns[t - 253:t - 21, column], market[t - 253:t - 21]  # sessions t-252..t-21
    beta_long = np.polyfit(ml, rl, 1)[0]
    assert full["residual_mom_12_1"][t, column] == pytest.approx(np.sum(rl - beta_long * ml), rel=1e-8)
    high = panel.high.astype(np.float64)
    assert full["high_52w_proximity"][t, column] == pytest.approx(close[t, column] / high[t - 251:t + 1, column].max())
    volume = panel.volume.astype(np.float64)
    assert full["volume_ratio_5_60"][t, column] == pytest.approx(
        volume[t - 4:t + 1, column].mean() / volume[t - 59:t + 1, column].mean(), rel=1e-10)
    amihud = np.abs(returns[t - 60:t, column]) / (close[t - 59:t + 1, column] * volume[t - 59:t + 1, column])
    assert full["amihud_illiquidity_60d"][t, column] == pytest.approx(amihud.mean(), rel=1e-8)


def test_sector_momentum_averages_eligible_same_sector_members(panel: ResearchPanel,
                                                              full: dict[str, np.ndarray]) -> None:
    close = panel.close.astype(np.float64)
    t = 410
    ret = close[t] / close[t - 126] - 1.0
    for column in (0, 5):
        members = (panel.sector[t] == panel.sector[t, column]) & panel.eligible[t] & np.isfinite(ret)
        assert full["sector_mom_6m"][t, column] == pytest.approx(ret[members].mean(), rel=1e-12)


def test_centered_rank_and_composite_minimum() -> None:
    values = np.array([[3.0, 1.0, 2.0, np.nan, 5.0]])
    include = np.array([[True, True, True, True, False]])
    np.testing.assert_allclose(centered_rank(values, include), [[0.5, -0.5, 0.0, np.nan, np.nan]], equal_nan=True)
    predictors = {name: np.array([[1.0, 2.0, 3.0]]) for name in PREDICTOR_NAMES}
    for name in PREDICTOR_NAMES[:5]:
        predictors[name] = np.array([[np.nan, np.nan, np.nan]])
    composites = compute_composites(predictors, np.ones((1, 3), dtype=bool))
    assert np.isnan(composites["composite_all"]).all()  # 7 of 12 present
    assert np.isnan(composites["composite_trend"]).all()
    expected_reversal = np.array([[0.5, 0.0, -0.5]])  # both reversal signs are -1
    np.testing.assert_allclose(composites["composite_reversal"], expected_reversal)
