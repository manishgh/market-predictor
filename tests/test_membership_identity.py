from __future__ import annotations

import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.universe.membership_identity_validation import (
    validate_security_exclusion_share,
    verify_membership_identity,
)


def _membership(
    ticker: str,
    security_id: str,
    start: str,
    end: str | None,
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "security_id": security_id,
        "effective_from_utc": pd.Timestamp(start, tz="UTC"),
        "effective_to_utc": pd.NaT if end is None else pd.Timestamp(end, tz="UTC"),
    }


def _evidence(rows: list[tuple[str, str, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"session": session, "ticker": ticker, "bars": 78, "last_close": close}
            for session, ticker, close in rows
        ]
    )


def test_continuous_series_verifies_the_interval() -> None:
    memberships = pd.DataFrame([_membership("AAPL", "cusip:037833100", "2024-01-01", None)])
    evidence = _evidence(
        [(f"2024-01-{day:02d}", "AAPL", 180.0 + day) for day in range(2, 12)]
    )

    result = verify_membership_identity(memberships, evidence)

    assert len(result.verified) == 1
    assert not result.excluded
    assert result.verified[0]["evidence_sessions"] == 10


def test_symbol_that_changed_hands_is_excluded_not_corrected() -> None:
    """The observed defect: a 3.15 series resuming at 115.88 under one claim.

    Excluded rather than repaired, because repairing means substituting an
    inferred rename date for evidence we do not have.
    """

    memberships = pd.DataFrame(
        [_membership("FI", "cusip:337738108", "2019-07-09", "2025-11-11")]
    )
    evidence = _evidence(
        [("2021-09-28", "FI", 3.15), ("2021-09-29", "FI", 3.20)]
        + [("2023-06-07", "FI", 115.88), ("2023-06-08", "FI", 116.40)]
    )

    result = verify_membership_identity(memberships, evidence)

    assert not result.verified
    assert len(result.excluded) == 1
    breach = result.excluded[0]
    assert breach["reason"] == "symbol_changed_hands"
    assert breach["ratio"] == pytest.approx(36.212, rel=1e-3)
    assert result.excluded_securities == {"cusip:337738108"}


def test_long_unexplained_gap_marks_the_interval_unproven() -> None:
    memberships = pd.DataFrame([_membership("GAPPY", "cusip:000000000", "2024-01-01", None)])
    evidence = _evidence(
        [("2024-01-02", "GAPPY", 50.0), ("2024-01-03", "GAPPY", 50.5)]
        + [("2024-03-01", "GAPPY", 51.0), ("2024-03-04", "GAPPY", 51.2)]
    )

    result = verify_membership_identity(memberships, evidence)

    assert len(result.excluded) == 1
    assert result.excluded[0]["reason"] == "symbol_unproven_for_interval"


def test_rename_chain_verifies_both_sides_independently() -> None:
    """Renames must survive. Excluding them would remove META, ELV, BALL, RTX."""

    memberships = pd.DataFrame(
        [
            _membership("FB", "cusip:30303M102", "2019-07-09", "2022-06-09"),
            _membership("META", "cusip:30303M102", "2022-06-09", None),
        ]
    )
    evidence = _evidence(
        [(f"2022-06-0{day}", "FB", 190.0 + day) for day in range(1, 9)]
        + [(f"2022-06-{day:02d}", "META", 183.0 + day) for day in range(9, 18)]
    )

    result = verify_membership_identity(memberships, evidence)

    assert len(result.verified) == 2
    assert not result.excluded


def test_interval_outside_the_bar_corpus_is_unevaluated_not_excluded() -> None:
    """Absence of bars in a window the corpus never covered proves nothing."""

    memberships = pd.DataFrame(
        [_membership("OLD", "cusip:111111111", "2016-01-01", "2018-01-01")]
    )
    evidence = _evidence([(f"2024-01-{day:02d}", "NEW", 10.0) for day in range(2, 6)])

    result = verify_membership_identity(memberships, evidence)

    assert not result.excluded
    assert len(result.unevaluated) == 1
    assert result.unevaluated[0]["reason"] == "interval_precedes_corpus"


def test_symbol_reused_outside_its_interval_is_excluded() -> None:
    """The SunTrust shape: no bars inside the claimed window, bars years later.

    A within-interval continuity check cannot see this, because there is no
    series to check. Without this branch the reuse hides in an unevaluable
    bucket instead of being named.
    """

    memberships = pd.DataFrame(
        [_membership("STI", "sp500-historical:60c6fe", "2019-07-09", "2019-12-09")]
    )
    evidence = _evidence(
        [("2019-08-01", "OTHER", 10.0), ("2026-01-05", "OTHER", 11.0)]
        + [(f"2022-05-0{day}", "STI", 40.0 + day) for day in range(2, 7)]
    )

    result = verify_membership_identity(memberships, evidence)

    assert len(result.excluded) == 1
    assert result.excluded[0]["reason"] == "symbol_reused_outside_interval"
    assert not result.unevaluated


def test_security_with_no_bars_anywhere_is_excluded_as_unverifiable() -> None:
    """The Red Hat shape: acquired days into the window, no bars at all."""

    memberships = pd.DataFrame(
        [_membership("RHT", "sp500-historical:acf5a2", "2019-07-09", "2019-07-15")]
    )
    evidence = _evidence(
        [("2019-07-08", "OTHER", 10.0), ("2019-08-01", "OTHER", 11.0)]
    )

    result = verify_membership_identity(memberships, evidence)

    assert len(result.excluded) == 1
    assert result.excluded[0]["reason"] == "no_bar_evidence"


def test_missing_columns_fail_closed() -> None:
    with pytest.raises(DataReadinessError, match="membership input lacks"):
        verify_membership_identity(
            pd.DataFrame({"ticker": ["AAPL"]}),
            _evidence([("2024-01-02", "AAPL", 1.0)]),
        )


def test_security_exclusions_continue_through_five_percent_then_refuse() -> None:
    assert validate_security_exclusion_share(
        source_securities=20,
        excluded_securities=1,
    ) == pytest.approx(0.05)

    with pytest.raises(DataReadinessError, match="above the frozen 5.00% ceiling"):
        validate_security_exclusion_share(
            source_securities=20,
            excluded_securities=2,
        )
