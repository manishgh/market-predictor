"""XNYS session arithmetic shared by outcome maturation, performance and drift monitoring."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import cache

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_predictor.canonical.cutoffs import NEW_YORK
from market_predictor.core.errors import DataReadinessError


def _closes() -> pd.Series:
    return xcals.get_calendar("XNYS").closes


def _utc_date(moment: datetime) -> pd.Timestamp:
    if moment.utcoffset() is None:
        raise ValueError("session arithmetic requires a timezone-aware time")
    return pd.Timestamp(moment).tz_convert("UTC").tz_localize(None).normalize()


def _session(decision_session: date, closes: pd.Series) -> pd.Timestamp:
    label = pd.Timestamp(decision_session)
    if label not in closes.index:
        raise DataReadinessError(f"decision session is not an XNYS session: {decision_session}")
    return label


def session_after(session: date, count: int) -> date:
    """The XNYS session `count` sessions after `session`, itself an XNYS session."""
    if count < 0:
        raise ValueError("a session offset cannot be negative")
    closes = _closes()
    position = int(closes.index.get_loc(_session(session, closes))) + count
    if position >= len(closes):
        raise DataReadinessError(f"the XNYS calendar ends {closes.index[-1].date()}, before {count} sessions after {session}")
    later: date = closes.index[position].date()
    return later


def horizon_last_close(decision_session: date, sessions: int, *, through: datetime) -> datetime | None:
    """The close of the Nth XNYS session after the decision session, or None if not closed by `through`."""
    if sessions < 1:
        raise ValueError("a horizon has at least one session")
    closes = _closes()
    decision = _session(decision_session, closes)
    # A session closes on its own UTC date, so labels after `through`'s UTC date have not closed.
    last_label = _utc_date(through)
    if last_label > closes.index[-1]:
        raise DataReadinessError(f"the XNYS calendar ends {closes.index[-1].date()}, before {last_label.date()}")
    later = closes.loc[decision + pd.Timedelta(days=1) : last_label]
    if len(later) < sessions:
        return None
    close: datetime = later.iloc[sessions - 1].to_pydatetime()
    return close if close <= through else None


def decision_group_session(decision_group_id: str) -> date:
    """The decision session of a decision group, which is identified by its decision time.

    The swing decision cutoff falls on the session's own New York date, so that date is the
    session. A group id that is not a timezone-aware time raises ValueError.
    """
    decision_time = datetime.fromisoformat(decision_group_id)
    if decision_time.utcoffset() is None:
        raise ValueError(f"decision group is not a timezone-aware decision time: {decision_group_id}")
    return decision_time.astimezone(NEW_YORK).date()


def outcome_overdue(decision_session: date, sessions: int, *, now: datetime, grace: timedelta) -> bool:
    """Whether the horizon's last close plus the grace period has passed."""
    close = horizon_last_close(decision_session, sessions, through=now)
    return close is not None and now > close + grace


def _window_session_counts(lookback_days: int) -> np.ndarray:
    """Sessions in every `lookback_days` window lying inside the calendar.

    The default XNYS calendar spans about twenty years back to one year ahead of process
    start, so these counts can change slowly as that range rolls; they only relax unless a
    new market closure occurs.
    """
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    labels = _closes().index.to_numpy(dtype="datetime64[D]")
    # A window ending on day d spans (d - lookback_days, d]; only windows inside the calendar count.
    days = np.arange(labels[0] + np.timedelta64(lookback_days, "D"), labels[-1] + np.timedelta64(1, "D"))
    ends = np.searchsorted(labels, days, side="right")
    starts = np.searchsorted(labels, days - np.timedelta64(lookback_days, "D"), side="right")
    return ends - starts


@cache
def fewest_sessions_in_window(lookback_days: int) -> int:
    """The fewest XNYS sessions any `lookback_days` calendar-day window holds."""
    return int(_window_session_counts(lookback_days).min())


@cache
def most_sessions_in_window(lookback_days: int) -> int:
    """The most XNYS sessions any `lookback_days` calendar-day window holds."""
    return int(_window_session_counts(lookback_days).max())
