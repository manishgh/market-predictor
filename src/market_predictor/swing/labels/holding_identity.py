"""Check holding-window identity coverage, without asserting bar or fill availability."""
from __future__ import annotations

from datetime import date
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.joins import join_universe_membership
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.labels.holding_paths import future_outcome_rows, holding_calendar


def _continuous_owner_intervals(records: pd.DataFrame) -> list[tuple[str, pd.Timestamp, pd.Timestamp | None]]:
    """Metadata changes must not split an otherwise continuous security owner."""
    merged: list[tuple[str, pd.Timestamp, pd.Timestamp | None]] = []
    for identity, group in records.groupby("security_id", sort=True):
        owner: list[tuple[pd.Timestamp, pd.Timestamp | None]] = []
        for record in group.sort_values("effective_from_utc").to_dict("records"):
            start = pd.Timestamp(record["effective_from_utc"])
            raw_end: Any = record["effective_to_utc"]
            end = pd.Timestamp(raw_end) if pd.notna(raw_end) else None
            if owner and (owner[-1][1] is None or start <= owner[-1][1]):
                previous_start, previous_end = owner[-1]
                owner[-1] = (previous_start, None if previous_end is None or end is None else max(previous_end, end))
            else:
                owner.append((start, end))
        merged.extend((str(identity), start, end) for start, end in owner)
    return merged


def membership_session_coverage(
    memberships: pd.DataFrame, *, sessions: tuple[date, ...], security_ids: tuple[str, ...],
) -> pd.DataFrame:
    """Shared full-session ownership coverage; callers retain all competing owners."""
    required = {"ticker", "security_id", "effective_from_utc", "effective_to_utc"}
    if (not sessions or tuple(sorted(set(sessions))) != sessions
            or sessions != holding_calendar(sessions[0], sessions[-1]) or not security_ids
            or len(set(security_ids)) != len(security_ids) or not required.issubset(memberships.columns)
            or not memberships.columns.is_unique
            or any(not isinstance(value, str) or not value.strip() for value in security_ids)
            or any(not memberships[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
                   for name in ("ticker", "security_id"))):
        raise DataReadinessError("holding membership coverage requires explicit identities and an exact calendar")
    schedule = xcals.get_calendar("XNYS").schedule.reindex(pd.DatetimeIndex(sessions))
    opens, closes = schedule["open"], schedule["close"]
    intervals = memberships.copy()
    for name in ("effective_from_utc", "effective_to_utc"):
        non_null = intervals[name].dropna()
        if not all(pd.Timestamp(value).tzinfo is not None for value in non_null):
            raise DataReadinessError("holding identity intervals require timezone-aware timestamps")
        intervals[name] = pd.to_datetime(intervals[name], utc=True)
    ended = intervals.loc[intervals["effective_to_utc"].notna()]
    if intervals["effective_from_utc"].isna().any() or (not ended.empty and
            ended["effective_to_utc"].le(ended["effective_from_utc"]).any()):
        raise DataReadinessError("holding identity interval bounds are invalid")

    # Compute calendar coverage per ticker once. Competing identities remain visible.
    tickers = set(intervals.loc[intervals.security_id.isin(security_ids), "ticker"])
    coverage: dict[str, NDArray[np.bool_]] = {
        identity: np.zeros(len(sessions), dtype=bool) for identity in security_ids
    }
    for ticker in sorted(tickers):
        records = intervals.loc[intervals.ticker.eq(ticker)]
        masks = []
        overlaps = np.zeros(len(sessions), dtype=np.int64)
        for identity, start, end in _continuous_owner_intervals(records):
            mask = opens.ge(start)
            overlap = closes.ge(start)
            if end is not None:
                mask &= closes.lt(end)
                overlap &= opens.lt(end)
            overlaps += overlap.to_numpy(dtype=np.int64)
            masks.append((identity, mask.to_numpy(dtype=bool)))
        for identity, mask in masks:
            if identity in coverage:
                coverage[identity] |= mask & (overlaps == 1)
    return pd.concat([
        pd.DataFrame({"security_id": identity, "session_date_et": sessions, "membership_covered": mask})
        for identity, mask in coverage.items()
    ], ignore_index=True).set_index(["security_id", "session_date_et"])


def inspect_holding_membership_windows(
    decisions: pd.DataFrame, memberships: pd.DataFrame, *, sessions: tuple[date, ...],
    horizon_sessions: int, initial_fit_end: date,
) -> pd.DataFrame:
    """Inspect exact future sessions using bound membership intervals, not ticker guesses.

    A covered window establishes only membership-based identity evidence. Missing
    post-removal coverage is not proof of delisting, missing prices or a losing trade.
    Full membership input is required, including excluded identities, so competing
    ownership cannot disappear when the research population is restricted.
    """
    required = {"decision_id", "security_id", "ticker", "sector", "primary_benchmark", "session_date_et", "decision_time_utc"}
    if (decisions.empty or not sessions or tuple(sorted(set(sessions))) != sessions
            or sessions != holding_calendar(sessions[0], sessions[-1]) or horizon_sessions < 1
            or not required.issubset(decisions.columns) or not decisions.columns.is_unique
            or decisions[list(required)].isna().any().any() or decisions["decision_id"].duplicated().any()
            or decisions.duplicated(["security_id", "session_date_et"]).any()):
        raise DataReadinessError("holding identity preflight requires complete unique decision identities and an exact calendar")
    rows = decisions.loc[:, sorted(required)].copy().reset_index(drop=True)
    dates = pd.to_datetime(rows["session_date_et"], errors="coerce")
    if dates.isna().any():
        raise DataReadinessError("holding identity decision dates are invalid")
    rows["session_date_et"] = dates.dt.date
    cutoffs = swing_prediction_cutoffs(rows["session_date_et"])
    if not rows["decision_time_utc"].eq(cutoffs).all():
        raise DataReadinessError("holding identity decision cutoff differs from the canonical swing cutoff")
    joined = join_universe_membership(rows, memberships).set_index("decision_id")
    expected = rows.set_index("decision_id")
    for column in ("security_id", "sector", "primary_benchmark"):
        if not joined[column].reindex(expected.index).eq(expected[column]).all():
            raise DataReadinessError(f"holding identity parent decision {column} differs from membership")
    lookup = membership_session_coverage(memberships, sessions=sessions, security_ids=tuple(rows.security_id.unique()))
    ordinal = {day: index for index, day in enumerate(sessions)}
    starts = rows.session_date_et.map(ordinal)
    if starts.isna().any():
        raise DataReadinessError("holding identity decision falls outside the snapshot calendar")
    exit_days = (starts + horizon_sessions).map(lambda value: sessions[int(value)] if value < len(sessions) else None)
    mature = exit_days.notna()
    missing = pd.Series(0, index=rows.index, dtype="int64")
    first_missing = pd.Series(None, index=rows.index, dtype=object)
    for future in future_outcome_rows(rows, lookup, sessions, horizon_sessions):
        absent = mature & ~future["membership_covered"].eq(True)
        first = absent & missing.eq(0)
        first_missing.loc[first] = future.loc[first, "session_date_et"]
        missing += absent.astype(int)
    result = rows.assign(
        exit_session_date_et=exit_days,
        terminal_immature=~mature,
        development_matured=mature & exit_days.map(lambda value: value is not None and value <= initial_fit_end),
        uncovered_holding_sessions=missing,
        first_uncovered_session_date_et=first_missing,
        membership_holding_window_covered=mature & missing.eq(0),
    )
    return result
