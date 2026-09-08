"""Synthetic identity evidence checks; no prices or real outcomes are consumed."""
from __future__ import annotations

from datetime import date

import exchange_calendars as xcals
import pandas as pd
import pytest

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.labels.holding_identity import inspect_holding_membership_windows
from market_predictor.swing.labels.holding_paths import holding_calendar

SESSIONS = holding_calendar(date(2024, 1, 2), date(2024, 2, 2))


def _membership(**changes: object) -> dict[str, object]:
    values: dict[str, object] = {
        "security_id": "issuer:original", "ticker": "OLD", "sector": "technology",
        "industry": "software", "market_cap_bucket": "large", "liquidity_bucket": "high",
        "primary_benchmark": "XLK", "universe_snapshot_id": "synthetic-membership",
        "source": "synthetic-test-evidence",
        "effective_from_utc": pd.Timestamp("2023-12-01T00:00:00Z"),
        "effective_to_utc": pd.NaT,
        "available_at_utc": pd.Timestamp("2023-11-30T00:00:00Z"),
    }
    values.update(changes)
    return values


def _decisions(*offsets: int) -> pd.DataFrame:
    rows = pd.DataFrame({
        "decision_id": [f"decision:{offset}" for offset in offsets],
        "security_id": "issuer:original", "ticker": "OLD", "sector": "technology",
        "primary_benchmark": "XLK", "session_date_et": [SESSIONS[offset] for offset in offsets],
    })
    rows["decision_time_utc"] = swing_prediction_cutoffs(rows["session_date_et"])
    return rows


def _clock(offset: int, boundary: str) -> pd.Timestamp:
    calendar = xcals.get_calendar("XNYS")
    method = calendar.session_open if boundary == "open" else calendar.session_close
    return method(pd.Timestamp(SESSIONS[offset]))


def _inspect(
    decisions: pd.DataFrame | None = None, memberships: pd.DataFrame | None = None,
    *, sessions: tuple[date, ...] = SESSIONS, horizon: int = 10,
    initial_fit_end: date = SESSIONS[-1],
) -> pd.DataFrame:
    return inspect_holding_membership_windows(
        _decisions(0) if decisions is None else decisions,
        pd.DataFrame([_membership()]) if memberships is None else memberships,
        sessions=sessions, horizon_sessions=horizon, initial_fit_end=initial_fit_end,
    )


def test_complete_ten_session_window_preserves_inputs_and_decision_order() -> None:
    decisions = _decisions(2, 0)
    memberships = pd.DataFrame([_membership()])
    original_decisions, original_memberships = decisions.copy(deep=True), memberships.copy(deep=True)
    result = _inspect(decisions, memberships)
    assert result.decision_id.tolist() == decisions.decision_id.tolist()
    assert result.exit_session_date_et.tolist() == [SESSIONS[12], SESSIONS[10]]
    assert result.membership_holding_window_covered.all()
    assert result.development_matured.all()
    assert not result.terminal_immature.any()
    assert result.uncovered_holding_sessions.eq(0).all()
    assert result.first_uncovered_session_date_et.isna().all()
    pd.testing.assert_frame_equal(decisions, original_decisions)
    pd.testing.assert_frame_equal(memberships, original_memberships)


def test_removal_is_uncovered_identity_evidence_not_a_delisting_or_price_claim() -> None:
    memberships = pd.DataFrame([_membership(effective_to_utc=_clock(5, "open"))])
    result = _inspect(memberships=memberships).iloc[0]
    assert result.uncovered_holding_sessions == 6
    assert result.first_uncovered_session_date_et == SESSIONS[5]
    assert result.exit_session_date_et == SESSIONS[10]
    assert not result.membership_holding_window_covered
    assert result.development_matured
    assert not any("delist" in name or "return" in name or "fill" in name for name in result.index)


def test_one_missing_session_never_compresses_horizon_and_holiday_is_skipped() -> None:
    memberships = pd.DataFrame([
        _membership(effective_to_utc=_clock(5, "open")),
        _membership(effective_from_utc=_clock(6, "open")),
    ])
    result = _inspect(memberships=memberships).iloc[0]
    assert SESSIONS[10] == date(2024, 1, 17)
    assert date(2024, 1, 15) not in SESSIONS
    assert result.exit_session_date_et == SESSIONS[10]
    assert result.uncovered_holding_sessions == 1
    assert result.first_uncovered_session_date_et == SESSIONS[5]


def test_terminal_immature_rows_do_not_count_partial_missing_windows() -> None:
    memberships = pd.DataFrame([_membership(effective_to_utc=_clock(len(SESSIONS) - 2, "open"))])
    result = _inspect(_decisions(len(SESSIONS) - 4), memberships).iloc[0]
    assert result.terminal_immature
    assert pd.isna(result.exit_session_date_et)
    assert result.uncovered_holding_sessions == 0
    assert pd.isna(result.first_uncovered_session_date_et)
    assert not result.development_matured
    assert not result.membership_holding_window_covered


def test_initial_fit_end_uses_exit_not_decision_and_is_inclusive() -> None:
    result = _inspect(_decisions(0, 1), initial_fit_end=SESSIONS[10])
    assert result.development_matured.tolist() == [True, False]
    assert result.membership_holding_window_covered.all()
    assert not result.terminal_immature.any()


def test_bound_identity_survives_a_rename_without_inventing_another_security() -> None:
    boundary = _clock(5, "open")
    memberships = pd.DataFrame([
        _membership(effective_to_utc=boundary),
        _membership(ticker="NEW", effective_from_utc=boundary),
    ])
    decisions = _decisions(0, 6)
    decisions.loc[1, "ticker"] = "NEW"
    result = _inspect(decisions, memberships)
    assert result.membership_holding_window_covered.all()
    assert result.uncovered_holding_sessions.eq(0).all()
    assert result.security_id.eq("issuer:original").all()


def test_reused_ticker_does_not_cover_old_security_after_removal() -> None:
    boundary = _clock(5, "open")
    memberships = pd.DataFrame([
        _membership(effective_to_utc=boundary),
        _membership(security_id="issuer:replacement", effective_from_utc=boundary),
    ])
    result = _inspect(memberships=memberships).iloc[0]
    assert result.uncovered_holding_sessions == 6
    assert result.first_uncovered_session_date_et == SESSIONS[5]
    assert not result.membership_holding_window_covered


@pytest.mark.parametrize("boundary", ["midday", "close"])
@pytest.mark.parametrize("overlap_seconds", [0, 1])
def test_same_owner_metadata_change_preserves_continuous_ownership(boundary: str, overlap_seconds: int) -> None:
    time = _clock(5, "close") if boundary == "close" else _clock(5, "open") + pd.Timedelta(hours=2)
    memberships = pd.DataFrame([
        _membership(effective_to_utc=time),
        _membership(effective_from_utc=time - pd.Timedelta(seconds=overlap_seconds), sector="healthcare"),
    ])
    result = _inspect(memberships=memberships).iloc[0]
    assert result.membership_holding_window_covered
    assert result.uncovered_holding_sessions == 0


def test_same_owner_intraday_gap_is_not_coalesced() -> None:
    time = _clock(5, "open") + pd.Timedelta(hours=2)
    memberships = pd.DataFrame([
        _membership(effective_to_utc=time),
        _membership(effective_from_utc=time + pd.Timedelta(seconds=1)),
    ])
    result = _inspect(memberships=memberships).iloc[0]
    assert not result.membership_holding_window_covered
    assert result.uncovered_holding_sessions == 1


@pytest.mark.parametrize("boundary", ["full_session", "open_only", "close_only"])
def test_excluded_competing_security_still_makes_endpoint_ownership_ambiguous(boundary: str) -> None:
    start, end = _clock(5, "open"), _clock(5, "close") + pd.Timedelta(seconds=1)
    if boundary == "open_only":
        end = start + pd.Timedelta(hours=1)
    elif boundary == "close_only":
        start = end - pd.Timedelta(hours=1)
    memberships = pd.DataFrame([
        _membership(),
        _membership(security_id="issuer:excluded-competitor", effective_from_utc=start, effective_to_utc=end),
    ])
    # Only the retained issuer has a decision; the complete authority keeps competitors.
    result = _inspect(memberships=memberships).iloc[0]
    assert result.uncovered_holding_sessions == 1
    assert result.first_uncovered_session_date_et == SESSIONS[5]
    assert not result.membership_holding_window_covered


@pytest.mark.parametrize("end_offset,missing", [(-1, 1), (0, 1), (1, 0)])
def test_close_membership_end_is_exclusive(end_offset: int, missing: int) -> None:
    end = _clock(10, "close") + pd.Timedelta(seconds=end_offset)
    result = _inspect(memberships=pd.DataFrame([_membership(effective_to_utc=end)])).iloc[0]
    assert result.uncovered_holding_sessions == missing
    assert bool(result.membership_holding_window_covered) is (missing == 0)


@pytest.mark.parametrize("start_offset,missing", [(-1, 0), (0, 0), (1, 1)])
def test_renamed_security_must_cover_the_open_inclusively(start_offset: int, missing: int) -> None:
    boundary = _clock(5, "open")
    memberships = pd.DataFrame([
        _membership(effective_to_utc=boundary),
        _membership(ticker="NEW", effective_from_utc=boundary + pd.Timedelta(seconds=start_offset)),
    ])
    result = _inspect(memberships=memberships).iloc[0]
    assert result.uncovered_holding_sessions == missing
    assert bool(result.membership_holding_window_covered) is (missing == 0)


@pytest.mark.parametrize("column,value", [
    ("security_id", "issuer:wrong"), ("sector", "healthcare"),
    ("primary_benchmark", "XLV"), ("ticker", "UNKNOWN"),
])
def test_parent_decision_must_match_membership(column: str, value: str) -> None:
    decisions = _decisions(0)
    decisions[column] = value
    with pytest.raises(DataReadinessError):
        _inspect(decisions)


@pytest.mark.parametrize("defect", ["late", "naive"])
def test_noncanonical_decision_cutoff_is_rejected(defect: str) -> None:
    decisions = _decisions(0)
    if defect == "late":
        decisions["decision_time_utc"] += pd.Timedelta(seconds=1)
    else:
        decisions["decision_time_utc"] = decisions.decision_time_utc.dt.tz_localize(None)
    with pytest.raises(DataReadinessError):
        _inspect(decisions)


@pytest.mark.parametrize("column", ["effective_from_utc", "effective_to_utc", "available_at_utc"])
def test_timezone_naive_membership_is_rejected(column: str) -> None:
    memberships = pd.DataFrame([_membership(**{column: pd.Timestamp("2024-01-01")})])
    with pytest.raises(DataReadinessError):
        _inspect(memberships=memberships)


@pytest.mark.parametrize("defect", ["duplicate_id", "duplicate_security_session", "null_id"])
def test_repeated_or_missing_decision_identity_is_rejected(defect: str) -> None:
    decisions = _decisions(0, 1)
    if defect == "duplicate_id":
        decisions["decision_id"] = "same-id"
    elif defect == "null_id":
        decisions.loc[1, "decision_id"] = None
    else:
        decisions.loc[1, ["session_date_et", "decision_time_utc"]] = decisions.loc[
            0, ["session_date_et", "decision_time_utc"]
        ].to_numpy()
    with pytest.raises(DataReadinessError):
        _inspect(decisions)


@pytest.mark.parametrize("sessions", [
    (), SESSIONS[:4] + SESSIONS[5:], tuple(reversed(SESSIONS)),
    SESSIONS[:4] + (SESSIONS[3],) + SESSIONS[4:],
    SESSIONS[:9] + (date(2024, 1, 15),) + SESSIONS[9:],
])
def test_invalid_or_compressed_calendar_is_rejected(sessions: tuple[date, ...]) -> None:
    with pytest.raises(DataReadinessError):
        _inspect(sessions=sessions)


@pytest.mark.parametrize("horizon", [0, -1])
def test_nonpositive_horizon_is_rejected(horizon: int) -> None:
    with pytest.raises(DataReadinessError):
        _inspect(horizon=horizon)


def test_numeric_outcomes_and_features_cannot_change_identity_report() -> None:
    decisions, memberships = _decisions(0, 1), pd.DataFrame([_membership()])
    expected = _inspect(decisions, memberships)
    for column in ("open", "close", "volume", "future_net_return_10d", "prediction_score"):
        decisions[column] = [float("nan"), float("inf")]
        memberships[column] = "must-not-be-parsed-as-a-number"
    pd.testing.assert_frame_equal(_inspect(decisions, memberships), expected)
