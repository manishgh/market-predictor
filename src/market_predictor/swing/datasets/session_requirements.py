"""Membership-session requirements for the frozen combined daily archive.

These bounds describe that archive, not a new collection or training cutoff.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from datetime import date, timedelta
from typing import Any, Final, cast
from zoneinfo import ZoneInfo

import pandas as pd

from market_predictor.core.errors import DataReadinessError

START_DATE: Final = date(2018, 5, 29)
CUTOFF_DATE: Final = date(2026, 7, 8)
EASTERN: Final = ZoneInfo("America/New_York")


def eastern_date(value: object) -> date:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise DataReadinessError("membership timestamp is not timezone-aware")
    return cast(date, timestamp.tz_convert(EASTERN).date())


def expected_ticker_sessions(
    ticker: str,
    *,
    memberships: pd.DataFrame,
    benchmark_tickers: tuple[str, ...],
    benchmark_start_sessions: Mapping[str, date],
    all_sessions: tuple[date, ...],
    session_abstentions: set[date] | None = None,
) -> set[date]:
    abstentions = session_abstentions or set()
    if ticker in benchmark_tickers:
        if abstentions:
            raise DataReadinessError(
                f"benchmark cannot have stock session abstentions: {ticker}"
            )
        start = benchmark_start_sessions.get(ticker)
        if start is None:
            raise DataReadinessError(
                f"benchmark coverage audit is absent for {ticker}"
            )
        return {session for session in all_sessions if session >= start}
    rows = memberships.loc[memberships["ticker"].astype(str).str.upper().eq(ticker)]
    expected: dict[date, str] = {}
    for row in rows.itertuples(index=False):
        start = max(START_DATE, eastern_date(row.effective_from_utc))
        end = (
            CUTOFF_DATE
            if pd.isna(row.effective_to_utc)
            else min(CUTOFF_DATE, eastern_date(row.effective_to_utc) - timedelta(days=1))
        )
        for session in all_sessions:
            if start <= session <= end:
                prior = expected.setdefault(session, str(row.security_id))
                if prior != str(row.security_id):
                    raise DataReadinessError(
                        f"ticker {ticker} maps to multiple securities on {session}"
                    )
    unknown = abstentions.difference(expected)
    if unknown:
        raise DataReadinessError(
            f"session abstention is outside ticker membership: {ticker}; "
            f"first={min(unknown)}"
        )
    return set(expected).difference(abstentions)


def session_abstentions_by_ticker(
    session_gap_audit: Mapping[str, Any],
) -> dict[str, set[date]]:
    result: dict[str, set[date]] = defaultdict(set)
    gaps = session_gap_audit.get("gaps")
    if not isinstance(gaps, list):
        raise DataReadinessError("combined daily session-gap records are absent")
    for item in gaps:
        if not isinstance(item, Mapping):
            raise DataReadinessError("combined daily session-gap record is invalid")
        ticker = str(item["ticker"]).strip().upper()
        result[ticker].update(
            date.fromisoformat(str(value)) for value in item["missing_sessions"]
        )
    return dict(result)
