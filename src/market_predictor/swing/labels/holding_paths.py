"""Exact holding calendars and independently identified outcome observations."""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from datetime import date

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError

OUTCOME_BAR_COLUMNS = (
    "security_id", "session_date_et", "bar_start_utc", "bar_end_utc",
    "available_at_utc", "open", "high", "low", "close", "volume",
)


def holding_calendar(start: date, end: date) -> tuple[date, ...]:
    """Derive sessions from XNYS, never from whichever provider rows survived."""
    if start > end:
        raise DataReadinessError("holding calendar start follows end")
    calendar = xcals.get_calendar("XNYS")
    return tuple(session.date() for session in calendar.sessions_in_range(start, end))


def outcome_bar_lookup(bars: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(OUTCOME_BAR_COLUMNS).difference(bars.columns))
    if missing or not bars.columns.is_unique:
        raise DataReadinessError(f"outcome bars require unique columns: missing={missing}")
    data = bars.loc[:, list(OUTCOME_BAR_COLUMNS)].copy()
    identities = data["security_id"]
    if not identities.map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise DataReadinessError("outcome bars require explicit security identity")
    dates = pd.to_datetime(data["session_date_et"], errors="coerce")
    if dates.isna().any():
        raise DataReadinessError("outcome bars contain invalid session dates")
    data["session_date_et"] = dates.dt.date
    if data.duplicated(["security_id", "session_date_et"]).any():
        raise DataReadinessError("duplicate security/session outcome observations")
    data = validate_outcome_observations(data)
    return data.set_index(["security_id", "session_date_et"])


def validate_outcome_observations(bars: pd.DataFrame) -> pd.DataFrame:
    """Validate observation clocks and mark unusable prices without identity inference."""
    data = bars.copy()
    required = set(OUTCOME_BAR_COLUMNS).difference({"security_id"})
    if required.difference(data.columns) or not data.columns.is_unique:
        raise DataReadinessError("outcome observations lack required daily fields")
    dates = pd.to_datetime(data["session_date_et"], errors="coerce")
    if dates.isna().any():
        raise DataReadinessError("outcome observations contain invalid session dates")
    data["session_date_et"] = dates.dt.date
    for name in ("bar_start_utc", "bar_end_utc", "available_at_utc"):
        values = data[name]
        try:
            aware = values.map(lambda value: pd.notna(value) and pd.Timestamp(value).tzinfo is not None).all()
        except (TypeError, ValueError) as exc:
            raise DataReadinessError(f"invalid outcome observation {name}") from exc
        if not aware:
            raise DataReadinessError(f"outcome observations require timezone-aware {name}")
        data[name] = pd.to_datetime(values, utc=True)
    schedule = xcals.get_calendar("XNYS").schedule
    calendar_rows = schedule.reindex(pd.DatetimeIndex(data["session_date_et"]))
    expected_open = pd.Series(calendar_rows["open"].array, index=data.index)
    expected_close = pd.Series(calendar_rows["close"].array, index=data.index)
    if (data["bar_start_utc"].ne(expected_open).any()
            or data["bar_end_utc"].ne(expected_close).any()
            or data["available_at_utc"].lt(data["bar_end_utc"]).any()):
        raise DataReadinessError("outcome observation session/availability mismatch")
    prices = data[["open", "high", "low", "close"]].apply(pd.to_numeric, errors="coerce")
    volume = pd.to_numeric(data["volume"], errors="coerce")
    valid = (
        pd.Series(np.isfinite(prices.to_numpy(float)).all(axis=1), index=data.index)
        & prices.gt(0).all(axis=1)
        & prices["high"].ge(prices[["open", "low", "close"]].max(axis=1))
        & prices["low"].le(prices[["open", "high", "close"]].min(axis=1))
        & volume.gt(0) & np.isfinite(volume)
    )
    # An unusable observation is retained as missing evidence, not a synthetic fill.
    data.loc[~valid, ["open", "high", "low", "close"]] = np.nan
    data["outcome_observation_valid"] = valid
    return data


def future_outcome_rows(
    decisions: pd.DataFrame,
    lookup: pd.DataFrame,
    sessions: Sequence[date],
    horizon: int,
) -> Iterator[pd.DataFrame]:
    """Yield exact calendar-offset observations in the unchanged decision order."""
    if horizon < 1 or len(set(sessions)) != len(sessions) or tuple(sorted(sessions)) != tuple(sessions):
        raise DataReadinessError("outcome horizon/calendar is invalid")
    if not decisions.index.is_unique or decisions.duplicated(["security_id", "session_date_et"]).any():
        raise DataReadinessError("outcome decisions require unique security/session identities")
    if not decisions["security_id"].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
        raise DataReadinessError("outcome decisions require explicit security identity")
    ordinal = {day: index for index, day in enumerate(sessions)}
    starts = decisions["session_date_et"].map(ordinal)
    if starts.isna().any():
        raise DataReadinessError("outcome decision is not in the exchange calendar")
    for offset in range(1, horizon + 1):
        expected = (starts + offset).map(lambda index: sessions[int(index)] if index < len(sessions) else None)
        keys = pd.MultiIndex.from_arrays([decisions["security_id"], expected], names=lookup.index.names)
        observed = lookup.reindex(keys).reset_index(drop=True)
        observed.index = decisions.index
        observed["session_date_et"] = expected
        yield observed
