"""XNYS session arithmetic shared by outcome maturation, performance and drift monitoring."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from functools import cache

import exchange_calendars as xcals
import numpy as np
import pandas as pd

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


def outcome_overdue(decision_session: date, sessions: int, *, now: datetime, grace: timedelta) -> bool:
    """Whether the horizon's last close plus the grace period has passed."""
    close = horizon_last_close(decision_session, sessions, through=now)
    return close is not None and now > close + grace


def session_index(session: date) -> int:
    """The position of an XNYS session in the calendar, for spacing sessions exactly."""
    closes = _closes()
    return int(closes.index.get_loc(_session(session, closes)))


@cache
def fewest_sessions_in_window(lookback_days: int) -> int:
    """The fewest XNYS sessions any `lookback_days` calendar-day window of the calendar holds."""
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    labels = _closes().index.to_numpy(dtype="datetime64[D]")
    # A window ending on day d spans (d - lookback_days, d]; only windows inside the calendar count.
    days = np.arange(labels[0] + np.timedelta64(lookback_days, "D"), labels[-1] + np.timedelta64(1, "D"))
    ends = np.searchsorted(labels, days, side="right")
    starts = np.searchsorted(labels, days - np.timedelta64(lookback_days, "D"), side="right")
    return int((ends - starts).min())
