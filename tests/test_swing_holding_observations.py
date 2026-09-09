"""Synthetic required-session reads: numeric evidence never establishes ownership."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pyarrow.dataset as ds
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.datasets import holding_observations as observations
from market_predictor.swing.labels.holding_identity import membership_session_coverage
from market_predictor.swing.labels.holding_paths import holding_calendar

SESSIONS = holding_calendar(date(2024, 1, 2), date(2024, 1, 19))
SECURITY_ID = "issuer:original"
RAW_COLUMNS = (
    "ticker", "timeframe", "bar_start_utc", "bar_end_utc", "available_at_utc",
    "open", "high", "low", "close", "volume", "price_feed", "adjustment",
)


def _clock(day: date, boundary: str) -> pd.Timestamp:
    calendar = xcals.get_calendar("XNYS")
    return getattr(calendar, f"session_{boundary}")(pd.Timestamp(day))


def _membership(**changes: object) -> dict[str, object]:
    row: dict[str, object] = {
        "ticker": "OLD", "security_id": SECURITY_ID,
        "effective_from_utc": pd.Timestamp("2023-12-01T00:00:00Z"),
        "effective_to_utc": pd.NaT,
    }
    row.update(changes)
    return row


def _bars(sessions: tuple[date, ...] = SESSIONS[:3], *, ticker: str = "OLD") -> pd.DataFrame:
    return pd.DataFrame({
        "ticker": ticker, "timeframe": "1d",
        "bar_start_utc": [_clock(day, "open") for day in sessions],
        "bar_end_utc": [_clock(day, "close") for day in sessions],
        "available_at_utc": [_clock(day, "close") + pd.Timedelta(minutes=15) for day in sessions],
        "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0,
        "volume": 1000.0, "price_feed": "sip", "adjustment": "all",
    })


def _write(tmp_path: Path, bars: pd.DataFrame, name: str = "synthetic.parquet") -> Path:
    path = tmp_path / name
    bars.to_parquet(path, index=False)
    return path


def _read(
    path: Path | None, *, sessions: tuple[date, ...] = SESSIONS[:3],
    numeric_end: date | None = None, memberships: pd.DataFrame | None = None,
    ticker: str = "OLD",
) -> pd.DataFrame:
    return observations.read_required_holding_observations(
        path, ticker=ticker, security_id=SECURITY_ID, sessions=sessions,
        numeric_end=SESSIONS[-1] if numeric_end is None else numeric_end,
        memberships=pd.DataFrame([_membership()]) if memberships is None else memberships,
    )


def test_arrow_filters_exact_dates_and_projects_before_to_pandas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    required = (SESSIONS[1], SESSIONS[3])
    bars = _bars(SESSIONS[:6])
    outside = ~bars.bar_start_utc.dt.date.isin(required)
    bars.loc[outside, ["ticker", "price_feed", "adjustment"]] = "outside-scope-poison"
    for name in ("open", "high", "low", "close", "volume", "bar_end_utc", "available_at_utc"):
        bars[name] = bars[name].astype(str)
        bars.loc[outside, name] = "must-not-be-decoded-or-validated"
    bars["heldout_only_payload"] = [[{"poison": "unrelated numeric schema"}]] * len(bars)
    path = _write(tmp_path, pd.concat([bars, bars.loc[outside]], ignore_index=True))
    original_dataset = ds.dataset
    decoded_dates: list[date] = []

    class GuardedTable:
        def __init__(self, table: Any) -> None:
            self.table = table

        def to_pandas(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
            dates = [value.date() for value in self.table.column("bar_start_utc").to_pylist()]
            assert dates == list(required), "nonrequired numeric rows reached pandas decoding"
            assert self.table.column_names == list(RAW_COLUMNS), "unrelated source columns reached pandas"
            decoded_dates.extend(dates)
            return self.table.to_pandas(*args, **kwargs)

    class GuardedDataset:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.dataset = original_dataset(*args, **kwargs)
            self.schema = self.dataset.schema

        def count_rows(self, **kwargs: Any) -> int:
            assert kwargs.get("filter") is not None
            return int(self.dataset.count_rows(**kwargs))

        def to_table(self, *args: Any, **kwargs: Any) -> GuardedTable:
            assert kwargs.get("filter") is not None, "required dates must be pushed into the Arrow scan"
            assert kwargs.get("columns") == list(RAW_COLUMNS)
            return GuardedTable(self.dataset.to_table(*args, **kwargs))

    monkeypatch.setattr(observations.ds, "dataset", GuardedDataset)
    result = _read(path, sessions=required, numeric_end=required[-1])
    assert decoded_dates == list(required)
    assert result.session_date_et.tolist() == list(required)
    assert result.observation_status.eq("observation_valid").all()
    assert "heldout_only_payload" not in result


def test_null_start_is_source_corruption_not_missing_bar(tmp_path: Path) -> None:
    bars = _bars()
    bars.loc[1, "bar_start_utc"] = pd.NaT
    with pytest.raises(DataReadinessError, match="unassignable null start"):
        _read(_write(tmp_path, bars))


@pytest.mark.parametrize("owned", [True, False])
@pytest.mark.parametrize("state", ["missing", "valid", "invalid"])
def test_source_presence_validity_and_ownership_are_independent(tmp_path: Path, owned: bool, state: str) -> None:
    memberships = pd.DataFrame([_membership(
        effective_to_utc=pd.NaT if owned else _clock(SESSIONS[0], "open"),
    )])
    bars = _bars()
    if state == "invalid":
        bars["close"] = 0.0
    path = None if state == "missing" else _write(tmp_path, bars)
    result = _read(path, memberships=memberships)
    assert result.source_present.tolist() == [state != "missing"] * 3
    assert result.observation_status.tolist() == [f"observation_{state}"] * 3
    assert result.ownership_status.tolist() == ["membership_supported" if owned else "ownership_unresolved"] * 3
    assert result.requested_security_id.eq(SECURITY_ID).all()
    if owned:
        assert result.security_id.eq(SECURITY_ID).all()
    else:
        assert result.security_id.isna().all()
    if state != "missing":
        assert result.outcome_observation_valid.tolist() == [state == "valid"] * 3


@pytest.mark.parametrize("boundary", ["full_session", "open_only", "close_only"])
def test_full_membership_keeps_excluded_competing_owner(tmp_path: Path, boundary: str) -> None:
    start, end = _clock(SESSIONS[1], "open"), _clock(SESSIONS[1], "close") + pd.Timedelta(seconds=1)
    if boundary == "open_only":
        end = start + pd.Timedelta(seconds=1)
    elif boundary == "close_only":
        start = end - pd.Timedelta(seconds=1)
    memberships = pd.DataFrame([
        _membership(),
        _membership(security_id="issuer:excluded", effective_from_utc=start, effective_to_utc=end),
    ])
    coverage = membership_session_coverage(memberships, sessions=SESSIONS[:3], security_ids=(SECURITY_ID,))
    assert coverage.membership_covered.tolist() == [True, False, True]
    result = _read(_write(tmp_path, _bars()), memberships=memberships)
    assert result.ownership_status.tolist() == ["membership_supported", "ownership_unresolved", "membership_supported"]
    assert result.security_id.isna().tolist() == [False, True, False]
    assert result.source_present.all()
    assert result.observation_status.eq("observation_valid").all()


@pytest.mark.parametrize("reused", [False, True])
@pytest.mark.parametrize("ticker", ["OLD", "NEW"])
def test_alias_rename_cannot_grant_wrong_raw_ticker_ownership(tmp_path: Path, reused: bool, ticker: str) -> None:
    boundary = _clock(SESSIONS[1], "open")
    records = [
        _membership(effective_to_utc=boundary),
        _membership(ticker="NEW", effective_from_utc=boundary),
    ]
    if reused:
        records.append(_membership(security_id="issuer:replacement", effective_from_utc=boundary))
    memberships = pd.DataFrame(records)
    coverage = membership_session_coverage(memberships, sessions=SESSIONS[:3], security_ids=(SECURITY_ID,))
    assert coverage.membership_covered.all()
    result = _read(_write(tmp_path, _bars(ticker=ticker)), memberships=memberships, ticker=ticker)
    expected = [True, False, False] if ticker == "OLD" else [False, True, True]
    assert result.security_id.notna().tolist() == expected
    assert result.ownership_status.eq("membership_supported").tolist() == expected
    assert result.source_present.all()
    assert result.observation_status.eq("observation_valid").all()


@pytest.mark.parametrize("source", ["none", "empty", "outside_only"])
def test_missing_source_returns_every_exact_required_session(tmp_path: Path, source: str) -> None:
    path = None
    if source != "none":
        bars = _bars(SESSIONS[4:6]) if source == "outside_only" else _bars().iloc[:0]
        path = _write(tmp_path, bars)
    result = _read(path)
    assert result.session_date_et.tolist() == list(SESSIONS[:3])
    assert not result.source_present.any()
    assert result.observation_status.eq("observation_missing").all()
    assert result.ownership_status.eq("membership_supported").all()
    assert result.security_id.eq(SECURITY_ID).all()


def test_missing_middle_row_is_not_filled_or_compressed(tmp_path: Path) -> None:
    result = _read(_write(tmp_path, _bars().drop(index=1)))
    assert result.session_date_et.tolist() == list(SESSIONS[:3])
    assert result.source_present.tolist() == [True, False, True]
    assert result.observation_status.tolist() == ["observation_valid", "observation_missing", "observation_valid"]
    assert pd.isna(result.loc[1, "close"])
    assert result.loc[2, "close"] == 101.0


@pytest.mark.parametrize("column", ["open", "high", "low", "close", "volume"])
@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf"), -float("inf")])
def test_zero_negative_and_nonfinite_values_are_present_but_invalid(tmp_path: Path, column: str, value: float) -> None:
    bars = _bars()
    bars.loc[1, column] = value
    result = _read(_write(tmp_path, bars))
    assert result.source_present.all()
    assert result.observation_status.tolist() == ["observation_valid", "observation_invalid", "observation_valid"]
    assert result.outcome_observation_valid.tolist() == [True, False, True]
    assert result.security_id.eq(SECURITY_ID).all()
    assert result.loc[1, ["open", "high", "low", "close"]].isna().all()


@pytest.mark.parametrize("column,value", [("high", 100.0), ("low", 102.0)])
def test_inconsistent_ohlc_is_invalid_not_missing(tmp_path: Path, column: str, value: float) -> None:
    bars = _bars()
    bars.loc[1, column] = value
    result = _read(_write(tmp_path, bars))
    assert result.loc[1, "source_present"]
    assert result.loc[1, "observation_status"] == "observation_invalid"


@pytest.mark.parametrize("defect", [
    "duplicate", "duplicate_replacing_missing", "wrong_ticker", "wrong_feed", "wrong_adjustment", "wrong_timeframe",
    "late_open", "early_close", "early_available", "null_clock", "naive_open", "naive_available", "malformed_available",
])
def test_corrupt_required_rows_raise_instead_of_becoming_missing(tmp_path: Path, defect: str) -> None:
    bars = _bars()
    if defect == "duplicate":
        bars = pd.concat([bars, bars.iloc[[1]]], ignore_index=True)
    elif defect == "duplicate_replacing_missing":
        bars = pd.concat([bars.iloc[:2], bars.iloc[[1]]], ignore_index=True)
    elif defect.startswith("wrong_"):
        column = {"ticker": "ticker", "feed": "price_feed", "adjustment": "adjustment", "timeframe": "timeframe"}[
            defect.removeprefix("wrong_")
        ]
        bars.loc[1, column] = "not-the-required-identity"
    elif defect == "late_open":
        bars.loc[1, "bar_start_utc"] += pd.Timedelta(seconds=1)
    elif defect == "early_close":
        bars.loc[1, "bar_end_utc"] -= pd.Timedelta(seconds=1)
    elif defect == "early_available":
        bars.loc[1, "available_at_utc"] = bars.loc[1, "bar_end_utc"] - pd.Timedelta(seconds=1)
    elif defect == "null_clock":
        bars.loc[1, "bar_end_utc"] = pd.NaT
    elif defect.startswith("naive_"):
        column = "bar_start_utc" if defect == "naive_open" else "available_at_utc"
        bars[column] = bars[column].dt.tz_localize(None)
    else:
        bars["available_at_utc"] = bars.available_at_utc.astype(str)
        bars.loc[1, "available_at_utc"] = "not-a-clock"
    with pytest.raises(DataReadinessError):
        _read(_write(tmp_path, bars))


def test_numeric_end_is_inclusive_required_exit_not_decision_date(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    required = SESSIONS[1:11]
    assert required[-1] == date(2024, 1, 17)
    path = _write(tmp_path, _bars(required))
    result = _read(path, sessions=required, numeric_end=required[-1])
    assert result.session_date_et.tolist() == list(required)
    assert result.observation_status.eq("observation_valid").all()

    def forbidden_scan(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("requirements outside numeric_end must fail before opening raw data")

    monkeypatch.setattr(observations.ds, "dataset", forbidden_scan)
    for end in (SESSIONS[0], required[-2]):
        with pytest.raises(DataReadinessError):
            _read(path, sessions=required, numeric_end=end)


@pytest.mark.parametrize("required", [(), (SESSIONS[1], SESSIONS[0]), (SESSIONS[0],) * 2, (date(2024, 1, 15),)])
def test_invalid_requirements_fail_even_without_source(required: tuple[date, ...]) -> None:
    with pytest.raises(DataReadinessError):
        _read(None, sessions=required)


def test_inputs_and_raw_artifact_are_unchanged(tmp_path: Path) -> None:
    bars = _bars().iloc[::-1].copy()
    memberships = pd.DataFrame([_membership()])
    before_bars, before_memberships = bars.copy(deep=True), memberships.copy(deep=True)
    path = _write(tmp_path, bars)
    before_bytes = path.read_bytes()
    required = SESSIONS[:3]
    result = _read(path, sessions=required, memberships=memberships)
    assert result.session_date_et.tolist() == list(required)
    pd.testing.assert_frame_equal(bars, before_bars)
    pd.testing.assert_frame_equal(memberships, before_memberships)
    assert path.read_bytes() == before_bytes
    assert required == SESSIONS[:3]


def test_future_poison_cannot_change_required_observations(tmp_path: Path) -> None:
    bars = _bars(SESSIONS[:5])
    path = _write(tmp_path, bars, "baseline.parquet")
    expected = _read(path, numeric_end=SESSIONS[2])
    future = bars.bar_start_utc.dt.date.gt(SESSIONS[2])
    bars.loc[future, ["open", "high", "low", "close", "volume"]] = float("inf")
    bars.loc[future, ["ticker", "price_feed", "adjustment", "timeframe"]] = "heldout-poison"
    bars.loc[future, "available_at_utc"] = pd.NaT
    bars.loc[future, "bar_end_utc"] = pd.NaT
    bars = pd.concat([bars, bars.loc[future]], ignore_index=True)
    bars["future_return"] = "must-not-be-inspected"
    poisoned = _write(tmp_path, bars, "future-poison.parquet")
    pd.testing.assert_frame_equal(_read(poisoned, numeric_end=SESSIONS[2]), expected)
