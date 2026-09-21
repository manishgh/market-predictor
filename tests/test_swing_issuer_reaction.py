from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Literal

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pytest
from pydantic import ValidationError

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.issuer_reaction import (
    REACTION_COLUMNS,
    REACTION_EVENT_COLUMNS,
    IssuerReactionSources,
    issuer_reaction_contract_sha256,
)
from market_predictor.swing.contracts.return_feature_profiles import ReturnRelationshipSources
from market_predictor.swing.features.issuer_reaction import build_issuer_reactions
from tests.test_swing_return_feature_profiles import _bars

PRICE, VOLUME = REACTION_COLUMNS
UTC_NS = pd.DatetimeTZDtype(unit="ns", tz="UTC")


@dataclass
class Inputs:
    events: pd.DataFrame
    stocks: pd.DataFrame
    spy: pd.DataFrame
    sessions: tuple[date, ...]
    sources: IssuerReactionSources
    position: int


def _inputs(event: str = "2023-11-24T12:00:00Z", session: str = "2023-11-24") -> Inputs:
    calendar = xcals.get_calendar("XNYS")
    day = pd.Timestamp(session)
    sessions = tuple(stamp.date() for stamp in calendar.sessions_in_range(
        day - pd.Timedelta(days=60), day + pd.Timedelta(days=10),
    ))
    position = sessions.index(day.date())
    stocks, spy = _bars(sessions, "AAA"), _bars(sessions, "SPY")
    for frame in (stocks, spy):
        for column in ("bar_start_utc", "bar_end_utc", "available_at_utc", "ingested_at_utc"):
            frame[column] = frame[column].dt.as_unit("ns")
    stocks.loc[:, "volume"] = 100.0
    stocks.loc[position - 20:position - 1, "volume"] = np.arange(1, 21) * 10.0
    stocks.loc[position, ["open", "close", "volume"]] = [100.0, 110.0, 420.0]
    spy.loc[position, ["open", "close"]] = [200.0, 204.0]
    events = pd.DataFrame({
        "decision_id": ["decision-a"], "security_id": ["security-a"], "ticker": ["AAA"],
        "event_id": ["event-a"], "event_version_sha256": ["a" * 64],
        "event_available_at_utc": pd.to_datetime([event], utc=True).as_unit("ns"),
        "identity_available_at_utc": pd.to_datetime(["2020-01-02T12:00:00Z"], utc=True).as_unit("ns"),
        "decision_time_utc": pd.DatetimeIndex([calendar.session_close(day) + pd.Timedelta(hours=1)]).as_unit("ns"),
    })
    sources = IssuerReactionSources(
        bars=ReturnRelationshipSources(
            baseline_authority_sha256="1" * 64, stock_authority_sha256="2" * 64,
            spy_authority_sha256="3" * 64, availability_semantics="historical_proxy",
            availability_policy_id="test-approved-historical-proxy", availability_policy_sha256="4" * 64,
            price_adjustment="all", price_basis_and_vintage_id="test-frozen-vintage",
        ),
        event_authority_sha256="5" * 64, identity_authority_sha256="6" * 64,
        event_availability_semantics="historical_proxy", identity_availability_semantics="historical_proxy",
        event_availability_policy_sha256="7" * 64, identity_availability_policy_sha256="8" * 64,
    )
    return Inputs(events, stocks, spy, sessions, sources, position)


@pytest.fixture
def inputs() -> Inputs:
    return _inputs()


def _build(
    inputs: Inputs, *, purpose: Literal["historical_research", "live_construction"] = "historical_research",
) -> pd.DataFrame:
    return build_issuer_reactions(
        events=inputs.events, stock_bars=inputs.stocks, spy_bars=inputs.spy,
        history_sessions=inputs.sessions, sources=inputs.sources, purpose=purpose,
    )


def _missing(result: pd.DataFrame, *names: str) -> None:
    for name in names:
        assert result[name].isna().all()
        assert result[f"available_at_{name}"].isna().all()
        assert result[f"available_at_{name}"].dtype == UTC_NS
        assert result[f"missing_reason_{name}"].map(
            lambda value: isinstance(value, str) and bool(value.strip()),
        ).all()


def _present(result: pd.DataFrame, *names: str) -> None:
    for name in names:
        assert result[name].notna().all()
        assert result[name].dtype == np.dtype("float32")
        assert result[f"available_at_{name}"].notna().all()
        assert result[f"available_at_{name}"].dtype == UTC_NS
        assert result[f"missing_reason_{name}"].fillna("").eq("").all()


def test_hand_computed_formulas_and_output_schema(inputs: Inputs) -> None:
    result = _build(inputs)
    assert REACTION_COLUMNS == ("reaction_stock_minus_spy_oc", "reaction_volume_ratio_20")
    assert REACTION_EVENT_COLUMNS == tuple(inputs.events.columns)
    assert result.loc[0, PRICE] == pytest.approx(0.08)
    assert result.loc[0, VOLUME] == pytest.approx(4.0)
    _present(result, *REACTION_COLUMNS)
    assert result.loc[0, "reaction_session_date_et"] == date(2023, 11, 24)
    assert result.loc[0, "reaction_start_utc"] == pd.Timestamp("2023-11-24T14:30:00Z")
    assert result.loc[0, "reaction_end_utc"] == pd.Timestamp("2023-11-24T18:00:00Z")
    for name in ("reaction_start_utc", "reaction_end_utc"):
        assert result[name].dtype == UTC_NS
    for name in REACTION_COLUMNS:
        assert result.loc[0, f"available_at_{name}"] == pd.Timestamp("2023-11-24T18:15:00Z")


@pytest.mark.parametrize("event,session,start,end", [
    ("2023-11-21T12:00:00Z", "2023-11-21", "2023-11-21T14:30:00Z", "2023-11-21T21:00:00Z"),
    ("2023-11-21T16:00:00Z", "2023-11-22", "2023-11-22T14:30:00Z", "2023-11-22T21:00:00Z"),
    ("2023-11-21T22:00:00Z", "2023-11-22", "2023-11-22T14:30:00Z", "2023-11-22T21:00:00Z"),
    ("2023-11-18T12:00:00Z", "2023-11-20", "2023-11-20T14:30:00Z", "2023-11-20T21:00:00Z"),
    ("2023-11-23T12:00:00Z", "2023-11-24", "2023-11-24T14:30:00Z", "2023-11-24T18:00:00Z"),
    ("2023-11-24T18:00:00Z", "2023-11-27", "2023-11-27T14:30:00Z", "2023-11-27T21:00:00Z"),
    ("2023-03-10T22:00:00Z", "2023-03-13", "2023-03-13T13:30:00Z", "2023-03-13T20:00:00Z"),
    ("2023-11-03T21:00:00Z", "2023-11-06", "2023-11-06T14:30:00Z", "2023-11-06T21:00:00Z"),
])
def test_official_calendar_selection(event: str, session: str, start: str, end: str) -> None:
    result = _build(_inputs(event, session))
    assert result.loc[0, "reaction_session_date_et"] == date.fromisoformat(session)
    assert result.loc[0, "reaction_start_utc"] == pd.Timestamp(start)
    assert result.loc[0, "reaction_end_utc"] == pd.Timestamp(end)
    _present(result, *REACTION_COLUMNS)


@pytest.mark.parametrize("offset,session", [(-1, "2023-11-21"), (0, "2023-11-22"), (1, "2023-11-22")])
def test_open_boundary_is_strict_to_one_nanosecond(offset: int, session: str) -> None:
    event = pd.Timestamp("2023-11-21T14:30:00Z") + pd.Timedelta(nanoseconds=offset)
    result = _build(_inputs(event.isoformat(), session))
    assert result.loc[0, "reaction_session_date_et"] == date.fromisoformat(session)
    assert result.loc[0, "reaction_start_utc"] > event
    _present(result, *REACTION_COLUMNS)


def test_missing_chosen_stock_bar_never_skips_to_later_available_bar(inputs: Inputs) -> None:
    inputs.stocks = inputs.stocks.drop(index=inputs.position)
    inputs.events["decision_time_utc"] = inputs.spy.available_at_utc.iloc[-1]
    result = _build(inputs)
    _missing(result, *REACTION_COLUMNS)
    assert result.loc[0, "reaction_session_date_et"] == date(2023, 11, 24)
    pd.testing.assert_frame_equal(result[inputs.events.columns], inputs.events)


@pytest.mark.parametrize("later_history", [False, True])
def test_selected_official_session_does_not_depend_on_history_bounds(inputs: Inputs, later_history: bool) -> None:
    selection = slice(inputs.position + 1, None) if later_history else slice(None, inputs.position)
    inputs.stocks = inputs.stocks.iloc[selection]
    inputs.spy = inputs.spy.iloc[selection]
    inputs.sessions = inputs.sessions[selection]
    inputs.events["decision_time_utc"] = pd.Timestamp("2023-12-06T22:00:00Z")
    result = _build(inputs)
    _missing(result, *REACTION_COLUMNS)
    assert result.loc[0, "reaction_session_date_et"] == date(2023, 11, 24)
    assert result.loc[0, "reaction_start_utc"] == pd.Timestamp("2023-11-24T14:30:00Z")


def test_only_nineteen_prior_sessions_does_not_shorten_volume_baseline(inputs: Inputs) -> None:
    first = inputs.position - 19
    inputs.stocks = inputs.stocks.iloc[first:]
    inputs.spy = inputs.spy.iloc[first:]
    inputs.sessions = inputs.sessions[first:]
    result = _build(inputs)
    _missing(result, VOLUME)
    _present(result, PRICE)


def test_missing_spy_only_suppresses_price(inputs: Inputs) -> None:
    inputs.spy = inputs.spy.drop(index=inputs.position)
    result = _build(inputs)
    _missing(result, PRICE)
    _present(result, VOLUME)
    assert result.loc[0, VOLUME] == pytest.approx(4.0)


@pytest.mark.parametrize("lag", [1, 10, 20])
@pytest.mark.parametrize("missing_row", [False, True])
def test_missing_lagged_volume_does_not_shorten_or_shift_window(inputs: Inputs, lag: int, missing_row: bool) -> None:
    index = inputs.position - lag
    if missing_row:
        inputs.stocks = inputs.stocks.drop(index=index)
    else:
        inputs.stocks.loc[index, "volume"] = np.nan
    result = _build(inputs)
    _missing(result, VOLUME)
    _present(result, PRICE)


@pytest.mark.parametrize("source,column,missing", [
    ("stocks", "open", PRICE), ("stocks", "close", PRICE), ("spy", "open", PRICE),
    ("spy", "close", PRICE), ("stocks", "volume", VOLUME),
])
def test_feature_values_have_independent_dependencies(inputs: Inputs, source: str, column: str, missing: str) -> None:
    getattr(inputs, source).loc[inputs.position, column] = np.nan
    result = _build(inputs)
    _missing(result, missing)
    _present(result, VOLUME if missing == PRICE else PRICE)


def test_zero_current_volume_is_valid_but_zero_lagged_denominator_is_missing(inputs: Inputs) -> None:
    inputs.stocks.loc[inputs.position, "volume"] = 0.0
    result = _build(inputs)
    _present(result, *REACTION_COLUMNS)
    assert result.loc[0, VOLUME] == 0.0
    inputs.stocks.loc[inputs.position - 20:inputs.position - 1, "volume"] = 0.0
    result = _build(inputs)
    _missing(result, VOLUME)
    _present(result, PRICE)


def test_current_volume_is_not_in_its_own_denominator(inputs: Inputs) -> None:
    inputs.stocks.loc[inputs.position, "volume"] *= 2
    assert _build(inputs).loc[0, VOLUME] == pytest.approx(8.0)


def test_isolated_zero_lagged_volume_is_valid_when_full_window_mean_is_positive(inputs: Inputs) -> None:
    inputs.stocks.loc[inputs.position - 20, "volume"] = 0.0
    result = _build(inputs)
    _present(result, *REACTION_COLUMNS)
    assert result.loc[0, VOLUME] == pytest.approx(420.0 / 104.5)


def test_incomplete_chosen_session_preserves_decision_and_window(inputs: Inputs) -> None:
    inputs.events["decision_time_utc"] = pd.Timestamp("2023-11-24T17:59:59.999999999Z")
    result = _build(inputs)
    _missing(result, *REACTION_COLUMNS)
    assert result.loc[0, "reaction_end_utc"] == pd.Timestamp("2023-11-24T18:00:00Z")
    pd.testing.assert_frame_equal(result[inputs.events.columns], inputs.events)


def test_exact_session_close_is_completed_when_bar_clocks_are_available(inputs: Inputs) -> None:
    close = inputs.stocks.loc[inputs.position, "bar_end_utc"]
    inputs.events["decision_time_utc"] = close
    for frame in (inputs.stocks, inputs.spy):
        frame.loc[inputs.position, ["available_at_utc", "ingested_at_utc"]] = close
    _present(_build(inputs), *REACTION_COLUMNS)


@pytest.mark.parametrize("source,lag,affected", [
    ("stocks", 0, REACTION_COLUMNS), ("stocks", 1, (VOLUME,)),
    ("stocks", 20, (VOLUME,)), ("spy", 0, (PRICE,)),
])
@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_each_consumed_bar_clock_uses_exact_nanosecond_cutoff(
    inputs: Inputs, source: str, lag: int, affected: tuple[str, ...], offset: int,
) -> None:
    cutoff = inputs.events.loc[0, "decision_time_utc"]
    clock = cutoff + pd.Timedelta(nanoseconds=offset)
    getattr(inputs, source).loc[inputs.position - lag, "available_at_utc"] = clock
    result = _build(inputs)
    if offset > 0:
        _missing(result, *affected)
    else:
        _present(result, *affected)
        for name in affected:
            assert result.loc[0, f"available_at_{name}"] == clock
    _present(result, *(name for name in REACTION_COLUMNS if name not in affected))


@pytest.mark.parametrize("source,lag,affected", [
    ("stocks", 0, REACTION_COLUMNS), ("stocks", 10, (VOLUME,)), ("spy", 0, (PRICE,)),
])
@pytest.mark.parametrize("column", ["available_at_utc", "bar_start_utc", "bar_end_utc", "ingested_at_utc"])
def test_missing_consumed_bar_clock_is_not_fabricated(
    inputs: Inputs, source: str, lag: int, affected: tuple[str, ...], column: str,
) -> None:
    getattr(inputs, source).loc[inputs.position - lag, column] = pd.NaT
    result = _build(inputs)
    _missing(result, *affected)
    _present(result, *(name for name in REACTION_COLUMNS if name not in affected))


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_identity_availability_is_consumed_without_changing_reaction_session(inputs: Inputs, offset: int) -> None:
    clock = inputs.events.loc[0, "decision_time_utc"] + pd.Timedelta(nanoseconds=offset)
    inputs.events["identity_available_at_utc"] = clock
    result = _build(inputs)
    assert result.loc[0, "reaction_session_date_et"] == date(2023, 11, 24)
    if offset > 0:
        _missing(result, *REACTION_COLUMNS)
    else:
        _present(result, *REACTION_COLUMNS)
        for name in REACTION_COLUMNS:
            assert result.loc[0, f"available_at_{name}"] == clock


def test_feature_specific_maximum_clock_includes_lagged_volume(inputs: Inputs) -> None:
    inputs.events["identity_available_at_utc"] = pd.Timestamp("2023-11-24T18:20:00.000000001Z")
    inputs.spy.loc[inputs.position, "available_at_utc"] = pd.Timestamp("2023-11-24T18:21:00.000000002Z")
    inputs.stocks.loc[inputs.position - 20, "available_at_utc"] = pd.Timestamp("2023-11-24T18:22:00.000000003Z")
    result = _build(inputs)
    assert result.loc[0, f"available_at_{PRICE}"] == pd.Timestamp("2023-11-24T18:21:00.000000002Z")
    assert result.loc[0, f"available_at_{VOLUME}"] == pd.Timestamp("2023-11-24T18:22:00.000000003Z")


def test_pre_release_prices_and_unconsumed_spy_history_cannot_change_reaction(inputs: Inputs) -> None:
    original = _build(inputs)
    inputs.stocks.loc[:inputs.position - 1, ["open", "close"]] *= 1000
    inputs.spy.loc[:inputs.position - 1, ["open", "close", "volume"]] *= 1000
    inputs.spy.loc[:inputs.position - 1, "available_at_utc"] = pd.Timestamp("2030-01-01T00:00:00Z")
    inputs.stocks.loc[inputs.position - 21, "volume"] = 1e20
    pd.testing.assert_frame_equal(_build(inputs), original, check_exact=True)


def test_future_poison_and_truncated_history_do_not_change_past_measurements(inputs: Inputs) -> None:
    original = _build(inputs)
    for frame in (inputs.stocks, inputs.spy):
        frame.loc[inputs.position + 1:, ["open", "close", "volume"]] = 1e20
        frame.loc[inputs.position + 1:, "available_at_utc"] = pd.Timestamp("2030-01-01T00:00:00Z")
    pd.testing.assert_frame_equal(_build(inputs), original, check_exact=True)
    inputs.stocks = inputs.stocks.iloc[:inputs.position + 1]
    inputs.spy = inputs.spy.iloc[:inputs.position + 1]
    inputs.sessions = inputs.sessions[:inputs.position + 1]
    pd.testing.assert_frame_equal(_build(inputs), original, check_exact=True)


def test_input_values_population_order_and_frames_are_preserved(inputs: Inputs) -> None:
    second = inputs.events.assign(decision_id="decision-b", event_id="event-b", fixed_horizon_net_return=-999.0)
    inputs.events["fixed_horizon_net_return"] = np.nan
    inputs.events = pd.concat([second, inputs.events], ignore_index=True)
    inputs.events.index = pd.Index([9, 3], name="source_row")
    original = [frame.copy(deep=True) for frame in (inputs.events, inputs.stocks, inputs.spy)]
    result = _build(inputs)
    assert len(result) == 2
    pd.testing.assert_frame_equal(result[inputs.events.columns], inputs.events)
    for actual, expected in zip((inputs.events, inputs.stocks, inputs.spy), original, strict=True):
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_same_event_at_distinct_decisions_uses_each_original_cutoff(inputs: Inputs) -> None:
    early = inputs.events.assign(
        decision_id="decision-early", decision_time_utc=pd.Timestamp("2023-11-24T17:00:00Z"),
    )
    late = inputs.events.copy(deep=True)
    inputs.events = pd.concat([late, early], ignore_index=True)
    result = _build(inputs)
    _present(result.iloc[[0]], *REACTION_COLUMNS)
    _missing(result.iloc[[1]], *REACTION_COLUMNS)
    assert result.reaction_session_date_et.eq(date(2023, 11, 24)).all()
    for index in range(2):
        single = _build(replace(inputs, events=inputs.events.iloc[[index]].reset_index(drop=True)))
        pd.testing.assert_frame_equal(single, result.iloc[[index]].reset_index(drop=True), check_exact=True)


def test_multiple_events_for_same_decision_are_not_dropped(inputs: Inputs) -> None:
    other = inputs.events.assign(event_id="event-b", event_version_sha256="b" * 64)
    inputs.events = pd.concat([inputs.events, other], ignore_index=True)
    result = _build(inputs)
    assert len(result) == 2
    pd.testing.assert_frame_equal(result[inputs.events.columns], inputs.events)
    _present(result, *REACTION_COLUMNS)


def test_two_security_batch_matches_single_calls_in_input_order(inputs: Inputs) -> None:
    first = _build(inputs)
    other = replace(inputs, events=inputs.events.assign(
        decision_id="decision-b", security_id="security-b", ticker="BBB", event_id="event-b",
    ), stocks=inputs.stocks.assign(security_id="security-b", ticker="BBB"))
    other.stocks.loc[inputs.position, "close"] = 90.0
    other.stocks.loc[inputs.position, "volume"] = 210.0
    second = _build(other)
    inputs.events = pd.concat([other.events, inputs.events], ignore_index=True)
    inputs.stocks = pd.concat([inputs.stocks, other.stocks], ignore_index=True).iloc[::-1]
    inputs.spy = inputs.spy.iloc[::-1]
    pd.testing.assert_frame_equal(_build(inputs), pd.concat([second, first], ignore_index=True), check_exact=True)


def test_historical_proxy_cannot_be_used_for_live_construction(inputs: Inputs) -> None:
    _present(_build(inputs), *REACTION_COLUMNS)
    with pytest.raises(DataReadinessError):
        _build(inputs, purpose="live_construction")


def test_observed_batch_and_single_live_calls_share_transform(inputs: Inputs) -> None:
    inputs.sources = inputs.sources.model_copy(update={
        "bars": inputs.sources.bars.model_copy(update={"availability_semantics": "observed"}),
        "event_availability_semantics": "observed", "identity_availability_semantics": "observed",
    })
    for frame in (inputs.stocks, inputs.spy):
        frame["availability_policy"] = "observed"
    pd.testing.assert_frame_equal(_build(inputs), _build(inputs, purpose="live_construction"), check_exact=True)


@pytest.mark.parametrize("family", ["bars", "event", "identity"])
def test_live_rejects_proxy_in_any_single_source_family(inputs: Inputs, family: str) -> None:
    inputs.sources = inputs.sources.model_copy(update={
        "bars": inputs.sources.bars.model_copy(update={
            "availability_semantics": "historical_proxy" if family == "bars" else "observed",
        }),
        "event_availability_semantics": "historical_proxy" if family == "event" else "observed",
        "identity_availability_semantics": "historical_proxy" if family == "identity" else "observed",
    })
    for frame in (inputs.stocks, inputs.spy):
        frame["availability_policy"] = "market_interval_close" if family == "bars" else "observed"
    _present(_build(inputs), *REACTION_COLUMNS)
    with pytest.raises(DataReadinessError):
        _build(inputs, purpose="live_construction")


def test_historical_proxy_backfill_ingestion_years_after_decision_is_not_feature_clock(inputs: Inputs) -> None:
    original = _build(inputs)
    for frame in (inputs.stocks, inputs.spy):
        frame["ingested_at_utc"] = pd.Timestamp("2030-01-01T00:00:00.000000001Z")
    pd.testing.assert_frame_equal(_build(inputs), original, check_exact=True)


@pytest.mark.parametrize("source", ["stocks", "spy"])
def test_observed_availability_cannot_precede_ingestion(inputs: Inputs, source: str) -> None:
    inputs.sources = inputs.sources.model_copy(update={
        "bars": inputs.sources.bars.model_copy(update={"availability_semantics": "observed"}),
        "event_availability_semantics": "observed", "identity_availability_semantics": "observed",
    })
    for frame in (inputs.stocks, inputs.spy):
        frame["availability_policy"] = "observed"
    getattr(inputs, source).loc[inputs.position, "ingested_at_utc"] = pd.Timestamp("2030-01-01T00:00:00Z")
    with pytest.raises(DataReadinessError):
        _build(inputs, purpose="live_construction")


@pytest.mark.parametrize("source", ["stocks", "spy"])
def test_claimed_observed_semantics_cannot_hide_physical_proxy(inputs: Inputs, source: str) -> None:
    inputs.sources = inputs.sources.model_copy(update={
        "bars": inputs.sources.bars.model_copy(update={"availability_semantics": "observed"}),
        "event_availability_semantics": "observed", "identity_availability_semantics": "observed",
    })
    for frame in (inputs.stocks, inputs.spy):
        frame["availability_policy"] = "observed"
    getattr(inputs, source).loc[inputs.position, "availability_policy"] = "market_interval_close"
    with pytest.raises(DataReadinessError):
        _build(inputs, purpose="live_construction")


@pytest.mark.parametrize("source", ["stocks", "spy"])
@pytest.mark.parametrize("column,value", [
    ("price_feed", "iex"), ("adjustment", "raw"), ("source", "yahoo"),
    ("timeframe", "1h"), ("availability_policy", "observed"),
])
def test_forged_physical_source_rejected(inputs: Inputs, source: str, column: str, value: str) -> None:
    getattr(inputs, source).loc[inputs.position, column] = value
    with pytest.raises(DataReadinessError):
        _build(inputs)


@pytest.mark.parametrize("source", ["stocks", "spy"])
@pytest.mark.parametrize("column", ["bar_start_utc", "bar_end_utc"])
def test_forged_interval_rejected(inputs: Inputs, source: str, column: str) -> None:
    getattr(inputs, source).loc[inputs.position, column] += pd.Timedelta(nanoseconds=1)
    with pytest.raises(DataReadinessError):
        _build(inputs)


@pytest.mark.parametrize("source,column,value", [
    ("stocks", "security_id", "foreign"), ("stocks", "ticker", "OTHER"), ("spy", "ticker", "QQQ"),
])
def test_identity_substitution_rejected(inputs: Inputs, source: str, column: str, value: str) -> None:
    getattr(inputs, source).loc[inputs.position, column] = value
    with pytest.raises(DataReadinessError):
        _build(inputs)


@pytest.mark.parametrize("source", ["events", "stocks", "spy"])
def test_duplicate_rows_rejected(inputs: Inputs, source: str) -> None:
    frame = getattr(inputs, source)
    setattr(inputs, source, pd.concat([frame, frame.iloc[[0]]], ignore_index=True))
    with pytest.raises(DataReadinessError):
        _build(inputs)


@pytest.mark.parametrize("column", REACTION_EVENT_COLUMNS)
def test_missing_required_event_column_rejected(inputs: Inputs, column: str) -> None:
    inputs.events = inputs.events.drop(columns=column)
    with pytest.raises(DataReadinessError):
        _build(inputs)


@pytest.mark.parametrize("source,column", [
    ("stocks", "security_id"), ("stocks", "volume"), ("spy", "available_at_utc"), ("spy", "price_feed"),
])
def test_missing_bar_schema_rejected(inputs: Inputs, source: str, column: str) -> None:
    setattr(inputs, source, getattr(inputs, source).drop(columns=column))
    with pytest.raises(DataReadinessError):
        _build(inputs)


@pytest.mark.parametrize("column,value", [
    ("decision_id", ""), ("security_id", " security-a"), ("ticker", None),
    ("event_id", ""), ("event_version_sha256", "unverified"),
    ("decision_time_utc", "not-a-clock"), ("event_available_at_utc", "2023-11-24T12:00:00"),
    ("identity_available_at_utc", "2020-01-02T12:00:00"),
])
def test_invalid_event_inputs_rejected(inputs: Inputs, column: str, value: object) -> None:
    inputs.events[column] = inputs.events[column].astype(object)
    inputs.events.loc[0, column] = value
    with pytest.raises(DataReadinessError):
        _build(inputs)


@pytest.mark.parametrize("column", ["event_available_at_utc", "identity_available_at_utc", "decision_time_utc"])
def test_missing_event_or_identity_clock_is_not_invented(inputs: Inputs, column: str) -> None:
    inputs.events[column] = pd.NaT
    with pytest.raises(DataReadinessError):
        _build(inputs)


def test_event_after_decision_keeps_original_row_as_unavailable(inputs: Inputs) -> None:
    inputs.events["decision_time_utc"] = inputs.events.event_available_at_utc - pd.Timedelta(nanoseconds=1)
    result = _build(inputs)
    _missing(result, *REACTION_COLUMNS)
    pd.testing.assert_frame_equal(result[inputs.events.columns], inputs.events)


@pytest.mark.parametrize("source", ["events", "stocks", "spy"])
def test_duplicate_column_names_rejected(inputs: Inputs, source: str) -> None:
    frame = getattr(inputs, source)
    setattr(inputs, source, pd.concat([frame, frame.iloc[:, [0]]], axis=1))
    with pytest.raises(DataReadinessError):
        _build(inputs)


@pytest.mark.parametrize("source", ["stocks", "spy"])
@pytest.mark.parametrize("column", ["open", "close", "volume"])
@pytest.mark.parametrize("value", [True, "123.0", float("inf"), -1.0])
def test_invalid_bar_numbers_rejected(inputs: Inputs, source: str, column: str, value: object) -> None:
    frame = getattr(inputs, source)
    frame[column] = frame[column].astype(object)
    frame.loc[inputs.position, column] = value
    with pytest.raises(DataReadinessError):
        _build(inputs)


@pytest.mark.parametrize("source", ["stocks", "spy"])
def test_zero_price_denominator_rejected(inputs: Inputs, source: str) -> None:
    getattr(inputs, source).loc[inputs.position, "open"] = 0.0
    with pytest.raises(DataReadinessError):
        _build(inputs)


@pytest.mark.parametrize("column", ["close", "volume"])
def test_float32_overflow_remains_explicit_null(inputs: Inputs, column: str) -> None:
    inputs.stocks.loc[inputs.position, column] = 1e50
    result = _build(inputs)
    _missing(result, PRICE if column == "close" else VOLUME)
    _present(result, VOLUME if column == "close" else PRICE)
    assert not np.isinf(result[list(REACTION_COLUMNS)].to_numpy()).any()


@pytest.mark.parametrize("kind", ["gap", "duplicate", "reverse", "empty"])
def test_invalid_history_calendar_rejected(inputs: Inputs, kind: str) -> None:
    if kind == "gap":
        inputs.sessions = inputs.sessions[:inputs.position - 1] + inputs.sessions[inputs.position:]
    elif kind == "duplicate":
        inputs.sessions = (*inputs.sessions, inputs.sessions[-1])
    elif kind == "reverse":
        inputs.sessions = inputs.sessions[::-1]
    else:
        inputs.sessions = ()
    with pytest.raises(DataReadinessError):
        _build(inputs)


@pytest.mark.parametrize("column", [
    *REACTION_COLUMNS, "reaction_session_date_et", "reaction_start_utc", "reaction_end_utc",
    *(f"available_at_{name}" for name in REACTION_COLUMNS),
    *(f"missing_reason_{name}" for name in REACTION_COLUMNS),
])
def test_output_collision_rejected(inputs: Inputs, column: str) -> None:
    inputs.events[column] = None
    with pytest.raises(DataReadinessError):
        _build(inputs)


def test_unsupported_purpose_rejected(inputs: Inputs) -> None:
    with pytest.raises(DataReadinessError):
        _build(inputs, purpose="training")  # type: ignore[arg-type]


@pytest.mark.parametrize("source", ["stocks", "spy"])
@pytest.mark.parametrize("null_first", [False, True])
def test_all_null_utc_ns_clocks_survive_parquet_roundtrip(
    inputs: Inputs, tmp_path: Path, source: str, null_first: bool,
) -> None:
    inputs.events["decision_time_utc"] = inputs.events.decision_time_utc.dt.as_unit("us")
    populated = _build(inputs)
    setattr(inputs, source, getattr(inputs, source).iloc[:0])
    missing = _build(inputs)
    _missing(missing, *(REACTION_COLUMNS if source == "stocks" else (PRICE,)))
    assert len(missing) == len(inputs.events)
    pd.testing.assert_frame_equal(missing[inputs.events.columns], inputs.events, check_exact=True)
    groups = [missing, populated] if null_first else [populated, missing]
    paths = []
    for index, frame in enumerate(groups):
        path = tmp_path / f"group-{index}.parquet"
        frame.to_parquet(path, index=False)
        pd.testing.assert_frame_equal(pd.read_parquet(path), frame, check_exact=True)
        paths.append(str(path))
    actual = ds.dataset(paths, format="parquet").to_table().to_pandas()
    pd.testing.assert_frame_equal(actual, pd.concat(groups, ignore_index=True), check_exact=True)
    for name in ("reaction_start_utc", "reaction_end_utc", *(f"available_at_{name}" for name in REACTION_COLUMNS)):
        assert actual[name].dtype == UTC_NS


def test_contract_hash_freezes_formulas_clocks_calendar_and_claim_boundaries(inputs: Inputs) -> None:
    expected = json_sha256({
        "schema": "market_predictor.issuer_reaction_measurements",
        "columns": REACTION_COLUMNS, "event_columns": REACTION_EVENT_COLUMNS, "calendar": "XNYS",
        "session_selection": "first_session_open_strictly_after_event_availability_before_inspecting_bars",
        "price_formula": "(stock_close/stock_open-1)-(spy_close/spy_open-1)",
        "volume_formula": "stock_volume/mean(previous_20_exchange_session_volumes)",
        "availability": "max(event_clock,identity_clock,consumed_bar_available_at_utc)<=decision",
        "bar_clock_validation": "all_physical_clocks_required_observed_ingestion_bounded_by_availability",
        "historical_proxy_ingestion": "retrospective_ingestion_not_a_historical_feature_clock",
        "missingness": "independent_null_values_and_reasons_never_substitute_sessions",
        "numeric_output": "float32_with_explicit_overflow_missingness", "clock_output": "datetime64[ns, UTC]",
        "source_admission": "caller_required_not_established_by_measurement",
        "content_qualification": "not_established", "causal_effect_identification": "not_claimed",
        "sources": inputs.sources.model_dump(mode="json"),
    })
    assert issuer_reaction_contract_sha256(inputs.sources) == expected


@pytest.mark.parametrize("field", [
    "event_authority_sha256", "identity_authority_sha256",
    "event_availability_policy_sha256", "identity_availability_policy_sha256",
])
def test_contract_hash_binds_event_and_identity_authorities(inputs: Inputs, field: str) -> None:
    original = issuer_reaction_contract_sha256(inputs.sources)
    changed = inputs.sources.model_copy(update={field: "b" * 64})
    assert issuer_reaction_contract_sha256(changed) != original
    with pytest.raises(ValidationError):
        IssuerReactionSources.model_validate({**inputs.sources.model_dump(), field: "unverified"})


@pytest.mark.parametrize("field", ["event_availability_semantics", "identity_availability_semantics"])
def test_contract_hash_binds_event_and_identity_semantics(inputs: Inputs, field: str) -> None:
    changed = inputs.sources.model_copy(update={field: "observed"})
    assert issuer_reaction_contract_sha256(changed) != issuer_reaction_contract_sha256(inputs.sources)
    with pytest.raises(ValidationError):
        IssuerReactionSources.model_validate({**inputs.sources.model_dump(), field: "inferred"})


@pytest.mark.parametrize("field,value", [
    ("baseline_authority_sha256", "b" * 64), ("stock_authority_sha256", "b" * 64),
    ("spy_authority_sha256", "b" * 64), ("availability_policy_sha256", "b" * 64),
    ("availability_policy_id", "different-policy"), ("availability_semantics", "observed"),
    ("price_adjustment", "raw"), ("price_basis_and_vintage_id", "different-vintage"),
])
def test_contract_hash_binds_each_bar_source_field(inputs: Inputs, field: str, value: str) -> None:
    changed = inputs.sources.model_copy(update={"bars": inputs.sources.bars.model_copy(update={field: value})})
    assert issuer_reaction_contract_sha256(changed) != issuer_reaction_contract_sha256(inputs.sources)
