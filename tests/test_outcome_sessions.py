from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.drift.policy import DriftPolicy
from market_predictor.governance.outcomes.sessions import (
    fewest_sessions_in_window,
    horizon_last_close,
    most_sessions_in_window,
    outcome_overdue,
)
from market_predictor.swing.labels.holding_paths import holding_calendar

GRACE = timedelta(days=7)


def test_last_close_counts_sessions_after_the_decision_across_a_holiday() -> None:
    # 3 July 2026 is a holiday: the tenth session after 2 July is 17 July.
    close = datetime(2026, 7, 17, 20, 0, tzinfo=UTC)

    assert horizon_last_close(date(2026, 7, 2), 10, through=close) == close
    assert horizon_last_close(date(2026, 7, 2), 10, through=close - timedelta(seconds=1)) is None
    assert horizon_last_close(date(2026, 7, 2), 10, through=datetime(2026, 7, 10, tzinfo=UTC)) is None


@pytest.mark.parametrize(
    "decision",
    (
        date(2026, 7, 2),  # holiday inside the path
        date(2026, 3, 6),  # daylight-saving change inside the path
        date(2025, 11, 21),  # early close on 28 November inside the path
        date(2025, 12, 19),  # two holidays inside the path
    ),
)
def test_last_session_matches_the_pinned_holding_calendar(decision: date) -> None:
    sessions = holding_calendar(decision, date(2027, 1, 29))
    tenth = sessions[sessions.index(decision) + 10]
    close = horizon_last_close(decision, 10, through=datetime(2027, 1, 30, tzinfo=UTC))

    assert close is not None
    assert close.astimezone(UTC).date() == tenth


def test_early_close_uses_the_actual_close() -> None:
    # 28 November 2025 closed at 13:00 New York (18:00 UTC).
    close = horizon_last_close(date(2025, 11, 26), 1, through=datetime(2025, 11, 29, tzinfo=UTC))

    assert close == datetime(2025, 11, 28, 18, 0, tzinfo=UTC)


def test_overdue_after_last_close_plus_grace_only() -> None:
    deadline = datetime(2026, 7, 17, 20, 0, tzinfo=UTC) + GRACE

    assert not outcome_overdue(date(2026, 7, 2), 10, now=deadline, grace=GRACE)
    assert outcome_overdue(date(2026, 7, 2), 10, now=deadline + timedelta(microseconds=1), grace=GRACE)
    assert not outcome_overdue(date(2026, 7, 2), 10, now=datetime(2026, 7, 10, tzinfo=UTC), grace=GRACE)


def test_refuses_naive_times_non_sessions_and_dates_past_the_calendar() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        horizon_last_close(date(2026, 7, 2), 10, through=datetime(2026, 8, 1))
    with pytest.raises(DataReadinessError, match="not an XNYS session"):
        horizon_last_close(date(2026, 7, 3), 10, through=datetime(2026, 8, 1, tzinfo=UTC))
    with pytest.raises(DataReadinessError, match="calendar ends"):
        horizon_last_close(date(2026, 7, 2), 10, through=datetime(2099, 1, 2, tzinfo=UTC))
    with pytest.raises(ValueError, match="at least one session"):
        horizon_last_close(date(2026, 7, 2), 0, through=datetime(2026, 8, 1, tzinfo=UTC))


def test_window_session_counts_match_the_calendar() -> None:
    # Measured over the calendar's range; they only relax as that range rolls forward.
    assert fewest_sessions_in_window(150) == 99
    assert fewest_sessions_in_window(180) == 120
    assert most_sessions_in_window(7) == 5


def test_lookback_check_allows_pending_outcomes_and_tolerated_failures() -> None:
    policy = DriftPolicy()

    # 170 days: 112 sessions - 5 still within the grace - 5 tolerated failures = 102 >= 100.
    assert policy.lookback_supports_minimum(170, "10b")
    # 160 days: 106 - 5 - 5 = 96 < 100; the old 150-day default was further short.
    assert not policy.lookback_supports_minimum(160, "10b")
    assert not policy.lookback_supports_minimum(150, "10b")
    assert policy.lookback_supports_minimum(180, "10b")


def test_tolerated_failures_use_exact_arithmetic() -> None:
    # A 94-day window holds at least 60 sessions. Exactly, 10% of 60 is 6 tolerated failures,
    # leaving 60 - 5 - 6 = 49 < 50. In floats (1 - 0.9) * 60 is 5.999..., which floors to 5
    # and would wrongly leave 50.
    assert fewest_sessions_in_window(94) == 60
    policy = DriftPolicy(minimum_registered_session_share=0.9, minimum_independent_decision_groups=5)
    assert not policy.lookback_supports_minimum(94, "10b")
