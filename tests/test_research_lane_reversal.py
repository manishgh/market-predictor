"""Unit tests on synthetic data for the frozen step 2c very-short-term reversal test."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from market_predictor.research_lane import reversal
from market_predictor.research_lane.indicators import compute_indicators
from market_predictor.research_lane.panel import PANEL_DIRECTORY
from market_predictor.research_lane.predictors import centered_rank
from market_predictor.research_lane.run import score_short_reversal
from tests.support.research_lane_panel import synthetic_panel, write_panel_files

END = date(2024, 5, 28)


def test_frozen_definition() -> None:
    assert reversal.REVERSAL_INPUTS == ("return_1d", "return_3d", "intraday_today", "streak_length")
    assert reversal.REVERSAL_SIGN == -1
    assert reversal.REVERSAL_HORIZONS == (1, 2, 3, 5)
    assert reversal.REVERSAL_MINIMUM_PRESENT == 3
    assert (reversal.REQUIRED_T, reversal.REQUIRED_POSITIVE_YEARS, reversal.FROZEN_COST) == (3.0, 5, 0.002)


def test_composite_is_mean_signed_rank_with_three_of_four_present() -> None:
    rng = np.random.default_rng(3)
    shape = (4, 30)
    indicators = {name: rng.normal(size=shape) for name in reversal.REVERSAL_INPUTS}
    indicators["streak_length"][1, :] = np.nan  # three of four present on row 1
    indicators["return_3d"][2, :] = np.nan
    indicators["streak_length"][2, :] = np.nan  # two of four present on row 2
    eligible = np.ones(shape, dtype=bool)
    signals = reversal.reversal_signals(indicators, eligible)
    for name in reversal.REVERSAL_INPUTS:
        assert np.array_equal(signals[name], -indicators[name], equal_nan=True)
    ranks = np.stack([centered_rank(-indicators[name], eligible) for name in reversal.REVERSAL_INPUTS])
    expected_full = ranks[:, 0, :].mean(axis=0)
    assert np.allclose(signals[reversal.REVERSAL_COMPOSITE][0], expected_full)
    assert np.allclose(signals[reversal.REVERSAL_COMPOSITE][1], ranks[:3, 1, :].mean(axis=0))
    assert np.isnan(signals[reversal.REVERSAL_COMPOSITE][2]).all()
    # Rising stocks rank low on the composite: sign -1 is applied before ranking.
    row = signals[reversal.REVERSAL_COMPOSITE][0]
    assert np.corrcoef(row, indicators["return_1d"][0])[0, 1] < 0


def test_composite_uses_no_later_rows() -> None:
    panel = synthetic_panel(sessions=420, securities=30)
    full = reversal.reversal_signals(compute_indicators(panel), panel.eligible)
    for cut in (260, 333, 400):
        last = date.fromisoformat(str(panel.sessions[cut]))
        short = panel.truncated(last)
        part = reversal.reversal_signals(compute_indicators(short), short.eligible)
        rows = len(short.sessions)
        for name, values in part.items():
            assert np.allclose(values, full[name][:rows], equal_nan=True, rtol=1e-12, atol=0.0), (name, cut)


def _entry(t: float, year_means: list[float], spread: float, turnover: float) -> dict[str, Any]:
    return {"rank_correlation": {"mean": 0.02, "t": t},
            "rank_correlation_by_year": {str(2019 + i): {"mean": m, "dates": 100} for i, m in enumerate(year_means)},
            "top_tenth": {"mean_spread_per_period_fraction": spread, "mean_turnover_fraction": turnover}}


def test_reading_rule_requires_all_three_conditions() -> None:
    good_years = [0.01, 0.02, 0.01, -0.01, 0.02, 0.03]
    board = {"predictors": {reversal.REVERSAL_COMPOSITE: {
        "1": _entry(3.2, good_years, 0.0030, 0.8),    # net 0.0030 - 0.0016 > 0: passes
        "2": _entry(2.9, good_years, 0.0030, 0.8),    # t below 3
        "3": _entry(3.5, [0.01, -0.02, 0.01, -0.01, 0.02, 0.03], 0.0030, 0.8),  # only 4 of 6 years
        "5": _entry(4.0, good_years, 0.0015, 0.8),    # net 0.0015 - 0.0016 < 0
    }}}
    result = reversal.reading(board)
    assert [result["horizons"][h]["passes"] for h in ("1", "2", "3", "5")] == [True, False, False, False]
    assert result["earns_sealed_period_test"] is True
    assert result["horizons"]["5"]["net_spread_per_period_at_0_20pct_fraction"] == pytest.approx(-0.0001)
    assert result["horizons"]["5"]["net_spread_per_period_at_0_10pct_fraction"] == pytest.approx(0.0007)
    board["predictors"][reversal.REVERSAL_COMPOSITE]["1"] = _entry(3.2, good_years, 0.0010, 0.8)
    assert reversal.reading(board)["earns_sealed_period_test"] is False
    assert "lead closed" in "\n".join(reversal.reading_markdown(reversal.reading(board)))


def test_run_scores_the_frozen_horizons_and_records_the_reading(tmp_path: Path,
                                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(tmp_path / "runtime"))  # isolated lease
    write_panel_files(tmp_path / PANEL_DIRECTORY, end=END)
    output = Path("data/research/research_lane/short_reversal_test")
    with patch("market_predictor.research_lane.run.guard"):
        score_short_reversal(root=tmp_path, output=output, development_end=END)
    board = json.loads((tmp_path / output / "scoreboard.json").read_text(encoding="utf-8"))
    assert board["settings"]["horizons_sessions"] == [1, 2, 3, 5]
    assert set(board["predictors"]) == {*reversal.REVERSAL_INPUTS, reversal.REVERSAL_COMPOSITE}
    assert set(board["reversal_reading"]["horizons"]) == {"1", "2", "3", "5"}
    assert "Pre-registered reading rule" in (tmp_path / output / "scoreboard.md").read_text(encoding="utf-8")
