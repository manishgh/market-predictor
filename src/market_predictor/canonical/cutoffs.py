from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from market_predictor.v3.errors import DataReadinessError

NEW_YORK = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class SwingPredictionCutoffPolicy:
    policy_id: str
    calendar_name: str
    local_cutoff: time


SWING_NIGHTLY_CUTOFF = SwingPredictionCutoffPolicy(
    policy_id="xnys_1800_america_new_york_v1",
    calendar_name="XNYS",
    local_cutoff=time(hour=18),
)


def swing_prediction_cutoffs(
    session_dates: pd.Series,
    *,
    policy: SwingPredictionCutoffPolicy = SWING_NIGHTLY_CUTOFF,
) -> pd.Series:
    """Return the immutable post-close cutoff for each XNYS session."""

    parsed = session_dates.map(_session_date)
    if bool(parsed.isna().any()):
        raise DataReadinessError("swing cutoff requires valid New York session dates")
    unique_sessions = sorted(set(parsed))
    if not unique_sessions:
        return pd.Series(pd.NaT, index=session_dates.index, dtype="datetime64[ns, UTC]")

    calendar = xcals.get_calendar(policy.calendar_name)
    cutoffs: dict[date, pd.Timestamp] = {}
    for session_date in unique_sessions:
        session = pd.Timestamp(session_date)
        if not calendar.is_session(session):
            raise DataReadinessError(f"swing cutoff received non-session date: {session_date}")
        market_close = pd.Timestamp(calendar.session_close(session)).tz_convert("UTC")
        next_session = calendar.next_session(session)
        next_open = pd.Timestamp(calendar.session_open(next_session)).tz_convert("UTC")
        local_cutoff = datetime.combine(session_date, policy.local_cutoff, tzinfo=NEW_YORK)
        cutoff = pd.Timestamp(local_cutoff).tz_convert("UTC")
        if not market_close < cutoff < next_open:
            raise DataReadinessError(
                "swing cutoff policy is outside the post-close/pre-open interval: "
                f"session={session_date}, close={market_close.isoformat()}, "
                f"cutoff={cutoff.isoformat()}, next_open={next_open.isoformat()}"
            )
        cutoffs[session_date] = cutoff
    return pd.to_datetime(parsed.map(cutoffs), utc=True)


def _session_date(value: object) -> date | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return date(timestamp.year, timestamp.month, timestamp.day)
