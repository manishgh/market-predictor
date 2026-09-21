"""Bounded, source-bound issuer measurements shared by research and live callers.

Source/content qualification and authority replay remain the caller's responsibility.
These measurements neither identify causal effects nor admit a model profile.
"""
from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Literal

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.contracts.issuer_reaction import (
    REACTION_COLUMNS,
    REACTION_EVENT_COLUMNS,
    IssuerReactionSources,
)
from market_predictor.swing.features.return_relationships import (
    _prepare_bars,
    _require,
    _sessions,
    _utc,
)

_BAR_CLOCKS = ("bar_start_utc", "bar_end_utc", "available_at_utc", "ingested_at_utc")
_EVENT_CLOCKS = ("event_available_at_utc", "identity_available_at_utc", "decision_time_utc")
_INTERVAL_COLUMNS = ("reaction_session_date_et", "reaction_start_utc", "reaction_end_utc")
_NAT = int(np.iinfo(np.int64).min)
_Measurement = tuple[float, int, str]


def _validated_events(events: pd.DataFrame) -> pd.DataFrame:
    _require(events, REACTION_EVENT_COLUMNS, "issuer reaction events")
    data = events.loc[:, list(REACTION_EVENT_COLUMNS)].reset_index(drop=True).copy()
    for column in ("decision_id", "security_id", "ticker", "event_id", "event_version_sha256"):
        if not data[column].map(lambda value: isinstance(value, str) and bool(value) and value.strip() == value).all():
            raise DataReadinessError(f"issuer reaction events contain invalid {column}")
    if not data.event_version_sha256.str.fullmatch(r"[0-9a-f]{64}").all():
        raise DataReadinessError("issuer reaction event versions require lowercase SHA256 hashes")
    for column in _EVENT_CLOCKS:
        data[column] = _utc(data[column], f"issuer reaction events.{column}").astype("datetime64[ns, UTC]")
        if data[column].isna().any():
            raise DataReadinessError(f"issuer reaction events require nonnull {column}")
    if data.duplicated(["decision_id", "event_id", "event_version_sha256", "security_id"]).any():
        raise DataReadinessError("issuer reaction events contain duplicate decision/event/version/security pairs")
    identities = (
        (["decision_id"], ["security_id", "ticker", "decision_time_utc"]),
        (["event_id", "event_version_sha256"], ["event_available_at_utc"]),
        (["event_id", "event_version_sha256", "security_id"], ["ticker", "identity_available_at_utc"]),
    )
    for keys, columns in identities:
        if data.groupby(keys, sort=False)[columns].nunique().gt(1).any(axis=None):
            raise DataReadinessError(f"issuer reaction events contain contradictory identities for {keys}")
    return data


def _bar_state(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[int, str]:
    if frame.ticker.isna().any() or frame.loc[:, list(columns)].isna().any(axis=None):
        return _NAT, "missing_required_session_or_value"
    if frame.loc[:, list(_BAR_CLOCKS)].isna().any(axis=None):
        return _NAT, "missing_required_bar_clock"
    # Physical clocks are mandatory, but only the bound availability sets the cutoff.
    clock = int(pd.DatetimeIndex(frame.available_at_utc).as_unit("ns").asi8.max())
    return clock, ""


def _finite_measurement(value: float, clock: int) -> _Measurement:
    if not np.isfinite(value):
        return np.nan, _NAT, "nonfinite_reaction_result"
    if abs(value) > float(np.finfo(np.float32).max):
        return np.nan, _NAT, "reaction_exceeds_float32_range"
    return value, clock, ""


def _measure_session(stock: pd.DataFrame, spy: pd.DataFrame, position: int) -> tuple[_Measurement, _Measurement]:
    current = stock.iloc[position:position + 1]
    market = spy.iloc[position:position + 1]
    stock_clock, stock_reason = _bar_state(current, ("open", "close"))
    spy_clock, spy_reason = _bar_state(market, ("open", "close"))
    price: _Measurement = (np.nan, _NAT, stock_reason or spy_reason)
    if not price[2]:
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            value = (current.close.iloc[0] / current.open.iloc[0] - 1.0) - (
                market.close.iloc[0] / market.open.iloc[0] - 1.0
            )
        price = _finite_measurement(float(value), max(stock_clock, spy_clock))

    if position < 20:
        return price, (np.nan, _NAT, "insufficient_history_sessions_requires_20_prior")
    window = stock.iloc[position - 20:position + 1]
    volume_clock, volume_reason = _bar_state(window, ("volume",))
    if volume_reason:
        return price, (np.nan, _NAT, volume_reason)
    baseline = window.volume.iloc[:-1].to_numpy(dtype=np.float64)
    # Scaling avoids overflowing a finite positive baseline's intermediate sum.
    scale = float(baseline.max())
    mean = scale * float(np.mean(baseline / scale)) if scale > 0 else 0.0
    if mean <= 0:
        return price, (np.nan, _NAT, "nonpositive_lagged_volume_baseline")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ratio = current.volume.iloc[0] / mean
    return price, _finite_measurement(float(ratio), volume_clock)


def build_issuer_reactions(
    *, events: pd.DataFrame, stock_bars: pd.DataFrame, spy_bars: pd.DataFrame,
    history_sessions: Sequence[date], sources: IssuerReactionSources,
    purpose: Literal["historical_research", "live_construction"] = "historical_research",
) -> pd.DataFrame:
    """Append two independently nullable measurements without filtering input pairs.

    The first official XNYS open strictly after event availability fixes the
    reaction session, even when that session is absent from supplied history.
    All consumed physical clocks must be present. Bound bar availability, event
    and identity clocks, and the selected session close must be at or before
    each original decision cutoff; archival ingestion is not a proxy cutoff.
    """
    if purpose not in ("historical_research", "live_construction"):
        raise DataReadinessError("unsupported issuer reaction construction purpose")
    semantics = (sources.bars.availability_semantics, sources.event_availability_semantics,
        sources.identity_availability_semantics)
    if purpose == "live_construction" and any(value != "observed" for value in semantics):
        raise DataReadinessError("live construction cannot consume historical proxy availability")
    data = _validated_events(events)
    additions = (*_INTERVAL_COLUMNS, *REACTION_COLUMNS,
        *(f"available_at_{name}" for name in REACTION_COLUMNS),
        *(f"missing_reason_{name}" for name in REACTION_COLUMNS))
    if set(additions).intersection(events.columns):
        raise DataReadinessError("issuer reaction output columns already exist in events")
    sessions = _sessions(history_sessions)
    calendar = xcals.get_calendar("XNYS")
    opens = pd.DatetimeIndex(calendar.opens).as_unit("ns").asi8
    event_clocks = pd.DatetimeIndex(data.event_available_at_utc).as_unit("ns").asi8
    identity_clocks = pd.DatetimeIndex(data.identity_available_at_utc).as_unit("ns").asi8
    cutoffs = pd.DatetimeIndex(data.decision_time_utc).as_unit("ns").asi8
    # Search the official calendar, never the supplied history or surviving bars.
    selected = np.searchsorted(opens, event_clocks, side="right")
    days: list[date | None] = [None] * len(data)
    starts = np.full(len(data), _NAT, dtype=np.int64)
    ends = np.full(len(data), _NAT, dtype=np.int64)
    positions = np.full(len(data), -1, dtype=np.int64)
    common_reasons = np.full(len(data), "", dtype=object)
    for row, session_position in enumerate(selected):
        local_day = data.event_available_at_utc.iloc[row].tz_convert("America/New_York").date()
        if local_day < calendar.first_session.date() or session_position >= len(opens):
            common_reasons[row] = "event_outside_supported_calendar_window"
            continue
        session = calendar.sessions[session_position]
        day = session.date()
        days[row] = day
        starts[row] = opens[session_position]
        ends[row] = calendar.session_close(session).value
        positions[row] = sessions.get_indexer([day])[0]
        if positions[row] < 0:
            common_reasons[row] = "reaction_session_outside_history_calendar"
        elif ends[row] > cutoffs[row]:
            common_reasons[row] = "reaction_session_not_complete_at_decision"
    common_reasons[identity_clocks > cutoffs] = "identity_available_after_decision"
    common_reasons[event_clocks > cutoffs] = "event_available_after_decision"

    stocks = _prepare_bars(stock_bars, stock=True, sessions=sessions, sources=sources.bars)
    spy = _prepare_bars(spy_bars, stock=False, sessions=sessions, sources=sources.bars)
    if not stocks.security_id.isin(data.security_id).all():
        raise DataReadinessError("stock history contains foreign security identities")
    spy = spy.set_index("session_date_et").reindex(sessions)
    values = np.full((len(data), len(REACTION_COLUMNS)), np.nan, dtype=np.float32)
    clocks = np.full((len(data), len(REACTION_COLUMNS)), _NAT, dtype=np.int64)
    reasons = np.repeat(common_reasons[:, None], len(REACTION_COLUMNS), axis=1)
    stock_groups = {security: frame for security, frame in stocks.groupby("security_id", sort=False)}
    for security, decisions in data.groupby("security_id", sort=False):
        stock = stock_groups.get(security, stocks.iloc[:0]).set_index("session_date_et").reindex(sessions)
        cache: dict[int, tuple[_Measurement, _Measurement]] = {}
        for row in decisions.index:
            position = int(positions[row])
            # Check the selected physical identity even when its values are unavailable.
            if position >= 0:
                ticker = stock.ticker.iloc[position]
                if pd.notna(ticker) and ticker != data.ticker.iloc[row]:
                    raise DataReadinessError("selected stock bar differs from event security/ticker identity")
            if common_reasons[row]:
                continue
            if position not in cache:
                cache[position] = _measure_session(stock, spy, position)
            for feature, (value, bar_clock, reason) in enumerate(cache[position]):
                clock = max(int(event_clocks[row]), int(identity_clocks[row]), bar_clock)
                if not reason and clock > cutoffs[row]:
                    reason = "required_bar_available_after_decision"
                reasons[row, feature] = reason
                if not reason:
                    values[row, feature] = value
                    clocks[row, feature] = clock

    result = events.copy()
    result["reaction_session_date_et"] = pd.array(days, dtype=object)
    result["reaction_start_utc"] = pd.to_datetime(starts, utc=True).as_unit("ns").array
    result["reaction_end_utc"] = pd.to_datetime(ends, utc=True).as_unit("ns").array
    for feature, name in enumerate(REACTION_COLUMNS):
        result[name] = values[:, feature]
        result[f"available_at_{name}"] = pd.to_datetime(clocks[:, feature], utc=True).as_unit("ns").array
        result[f"missing_reason_{name}"] = reasons[:, feature]
    return result
