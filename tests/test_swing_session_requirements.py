from datetime import date

import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.datasets.session_requirements import (
    CUTOFF_DATE,
    START_DATE,
    eastern_date,
    expected_ticker_sessions,
    session_abstentions_by_ticker,
)


def membership(end: str | None = None) -> pd.DataFrame:
    return pd.DataFrame([dict(security_id="issuer:one", ticker="AAA",
        effective_from_utc=pd.Timestamp("2024-01-02T05:00:00Z"),
        effective_to_utc=pd.Timestamp(end) if end else pd.NaT)])


def test_exclusive_end_and_eastern_date() -> None:
    days = tuple(date(2024, 1, d) for d in (2, 3, 4, 5))
    result = expected_ticker_sessions("AAA", memberships=membership("2024-01-04T05:00:00Z"),
        benchmark_tickers=(), benchmark_start_sessions={}, all_sessions=days)
    assert result == set(days[:2])
    assert eastern_date("2024-01-04T02:00:00Z") == date(2024, 1, 3)
    with pytest.raises(DataReadinessError, match="timezone-aware"):
        eastern_date("2024-01-04")


def test_sparse_gap_is_absent_not_imputed() -> None:
    days = (date(2024, 1, 2), date(2024, 1, 3))
    gaps = session_abstentions_by_ticker({"gaps": [{"ticker": " aaa ", "missing_sessions": [str(days[0])]}]})
    assert expected_ticker_sessions("AAA", memberships=membership(), benchmark_tickers=(),
        benchmark_start_sessions={}, all_sessions=days, session_abstentions=gaps["AAA"]) == {days[1]}
    with pytest.raises(DataReadinessError, match="outside ticker membership"):
        expected_ticker_sessions("AAA", memberships=membership(), benchmark_tickers=(),
            benchmark_start_sessions={}, all_sessions=days, session_abstentions={date(2023, 12, 29)})


def test_competing_security_ownership_rejected() -> None:
    rows = pd.concat([membership(), membership().assign(security_id="issuer:other")], ignore_index=True)
    with pytest.raises(DataReadinessError, match="multiple securities"):
        expected_ticker_sessions("AAA", memberships=rows, benchmark_tickers=(),
            benchmark_start_sessions={}, all_sessions=(date(2024, 1, 2),))


def test_benchmark_inception_needs_evidence_and_cannot_abstain() -> None:
    days = (date(2018, 6, 18), date(2018, 6, 19), date(2018, 6, 20))
    args = dict(memberships=pd.DataFrame(), benchmark_tickers=("XLC",), all_sessions=days)
    assert expected_ticker_sessions("XLC", benchmark_start_sessions={"XLC": days[1]}, **args) == set(days[1:])
    with pytest.raises(DataReadinessError, match="coverage audit is absent"):
        expected_ticker_sessions("XLC", benchmark_start_sessions={}, **args)
    with pytest.raises(DataReadinessError, match="benchmark cannot"):
        expected_ticker_sessions("XLC", benchmark_start_sessions={"XLC": days[1]}, session_abstentions={days[2]}, **args)


def test_frozen_archive_bounds_do_not_expand_with_current_date() -> None:
    rows = membership().assign(effective_from_utc=pd.Timestamp("2010-01-01T05:00:00Z"))
    assert START_DATE == date(2018, 5, 29)
    assert CUTOFF_DATE == date(2026, 7, 8)
    assert expected_ticker_sessions("AAA", memberships=rows, benchmark_tickers=(), benchmark_start_sessions={},
        all_sessions=(date(2010, 1, 4), START_DATE, CUTOFF_DATE, date(2026, 9, 11))) == {START_DATE, CUTOFF_DATE}
