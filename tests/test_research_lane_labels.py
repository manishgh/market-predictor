"""Unit tests: research-label arithmetic, missing bars and the development-end cutoff."""
from __future__ import annotations

import numpy as np
import pytest

from market_predictor.research_lane.labels import (
    ENTRY_BAR_MISSING,
    EXIT_BAR_MISSING,
    KNOWN,
    WINDOW_PAST_DEVELOPMENT_END,
    first_decision_row,
    interval_returns,
)
from tests.support.research_lane_panel import synthetic_panel


def test_label_is_next_open_to_close_h_sessions_after_decision() -> None:
    nan = np.nan
    open_ = np.array([10.0, 11.0, 12.0, nan, 14.0, 15.0, 16.0, 17.0])
    close = np.array([10.5, 11.5, 12.5, 13.5, nan, 15.5, 16.5, 17.5])
    values, reason = interval_returns(open_, close, 2)
    # t=0: close[2]/open[1]-1 = 12.5/11 - 1
    assert values[0] == pytest.approx(12.5 / 11.0 - 1.0)
    assert reason[0] == KNOWN
    # t=1: close[3]/open[2]-1
    assert values[1] == pytest.approx(13.5 / 12.0 - 1.0)
    # t=2: entry open[3] missing
    assert np.isnan(values[2]) and reason[2] == ENTRY_BAR_MISSING
    # t=3: entry open[4]=14, exit close[5]=15.5
    assert values[3] == pytest.approx(15.5 / 14.0 - 1.0)
    # t=2 also has a missing exit (close[4]); the entry reason takes precedence. t=... exit missing case:
    values_h1, reason_h1 = interval_returns(open_, close, 1)
    assert np.isnan(values_h1[3]) and reason_h1[3] == EXIT_BAR_MISSING  # open[4] ok, close[4] missing
    assert values_h1[0] == pytest.approx(11.5 / 11.0 - 1.0)  # h=1 is next-session open-to-close
    # The last h decisions have exits beyond the last loaded session (development end).
    assert np.isnan(values[6:]).all() and (reason[6:] == WINDOW_PAST_DEVELOPMENT_END).all()
    assert reason[5] == KNOWN and values[5] == pytest.approx(17.5 / 16.0 - 1.0)


def test_labels_for_matrices_and_short_histories() -> None:
    open_ = np.array([[1.0, 2.0], [1.1, 2.2], [1.2, 0.0]])
    close = np.array([[1.0, 2.0], [1.15, 2.1], [1.3, 2.4]])
    values, reason = interval_returns(open_, close, 1)
    np.testing.assert_allclose(values[0], [1.15 / 1.1 - 1.0, 2.1 / 2.2 - 1.0])
    assert reason[1, 1] == ENTRY_BAR_MISSING  # non-positive entry open is treated as missing
    assert (reason[2] == WINDOW_PAST_DEVELOPMENT_END).all()
    values_long, reason_long = interval_returns(open_, close, 5)
    assert np.isnan(values_long).all() and (reason_long == WINDOW_PAST_DEVELOPMENT_END).all()
    with pytest.raises(ValueError):
        interval_returns(open_, close, 0)


def test_first_decision_row_is_first_session_on_or_after_2019_07_09() -> None:
    panel = synthetic_panel(sessions=400)
    row = first_decision_row(panel)
    assert str(panel.sessions[row]) == "2019-07-09"
    assert str(panel.sessions[row - 1]) < "2019-07-09"
