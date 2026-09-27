from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.drift.policy import DriftPolicy
from market_predictor.governance.outcomes.sessions import (
    horizon_last_close,
    outcome_overdue,
    session_index,
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


def test_session_index_skips_holidays() -> None:
    assert session_index(date(2026, 7, 6)) - session_index(date(2026, 7, 2)) == 1
    assert session_index(date(2026, 7, 17)) - session_index(date(2026, 7, 2)) == 10


def test_default_lookback_holds_the_minimum_and_the_old_one_did_not() -> None:
    policy = DriftPolicy()

    # Every 180-day window holds at least 120 sessions; a 150-day window can hold 99.
    assert policy.lookback_supports_minimum(180, "10b")
    assert not policy.lookback_supports_minimum(150, "10b")
