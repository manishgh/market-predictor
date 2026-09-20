"""One session-aligned relationship transform and baseline assembly for batch/live.

Callers supply independently verified bounded histories and baseline partitions.
This builder computes features, not authority replay, fitting, or serving admission.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from numbers import Real
from typing import Literal

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.contracts.return_feature_profiles import (
    RETURN_RELATIONSHIP_COLUMNS,
    RETURN_RELATIONSHIP_PROFILE,
    RETURN_RELATIONSHIP_SESSION_POSITIONS,
    ReturnRelationshipSources,
    return_relationship_profile_sha256,
)
from market_predictor.swing.features.panel import swing_model_feature_columns
from market_predictor.swing.features.research_join import DECISION_KEYS
from market_predictor.swing.features.research_partition import ResearchFeaturePartition

_BAR_COLUMNS = (
    "ticker", "session_date_et", "bar_start_utc", "bar_end_utc", "available_at_utc",
    "open", "close", "volume", "price_feed", "adjustment", "source", "timeframe",
    "availability_policy", "ingested_at_utc",
)
_CLOCK_COLUMNS = ("bar_start_utc", "bar_end_utc", "available_at_utc", "ingested_at_utc")


def _require(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    if not frame.columns.is_unique or not set(columns).issubset(frame):
        raise DataReadinessError(f"{label} requires unique columns and {sorted(set(columns).difference(frame))}")


def _utc(values: pd.Series, label: str) -> pd.Series:
    for value in values.dropna():
        try:
            if not isinstance(value, (str, pd.Timestamp)) and not hasattr(value, "tzinfo"):
                raise ValueError("not a timestamp")
            if pd.Timestamp(value).tzinfo is None:
                raise ValueError("naive timestamp")
        except (ValueError, TypeError, OverflowError) as exc:
            raise DataReadinessError(f"{label} requires explicit timezone-aware clocks") from exc
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if (values.notna() & parsed.isna()).any():
        raise DataReadinessError(f"{label} contains invalid clocks")
    return parsed


def _session_dates(values: pd.Series, label: str) -> pd.Series:
    if not values.map(lambda value: type(value) is date).all():
        raise DataReadinessError(f"{label} requires explicit exchange-session dates")
    return values


def _numeric(values: pd.Series, label: str) -> pd.Series:
    if not values.dropna().map(lambda value: isinstance(value, Real) and not isinstance(value, (bool, np.bool_))).all():
        raise DataReadinessError(f"{label} requires real numeric values, never boolean or string coercion")
    numeric = pd.to_numeric(values, errors="coerce").astype("float64")
    if (values.notna() & numeric.isna()).any() or np.isinf(numeric.to_numpy(dtype=float)).any():
        raise DataReadinessError(f"{label} must be finite or explicitly null")
    return numeric


def _sessions(sessions: Sequence[date]) -> pd.Index:
    values = tuple(sessions)
    if not values or any(type(value) is not date for value in values) or tuple(sorted(set(values))) != values:
        raise DataReadinessError("history sessions must be nonempty, ordered and unique exchange dates")
    calendar = xcals.get_calendar("XNYS")
    expected = tuple(stamp.date() for stamp in calendar.sessions_in_range(values[0], values[-1]))
    if expected != values:
        raise DataReadinessError("history sessions must include every XNYS session without gaps")
    return pd.Index(values, name="session_date_et")


def _prepare_bars(
    frame: pd.DataFrame, *, stock: bool, sessions: pd.Index, sources: ReturnRelationshipSources,
) -> pd.DataFrame:
    label = "stock bars" if stock else "SPY bars"
    columns = (*_BAR_COLUMNS, *(("security_id",) if stock else ()))
    _require(frame, columns, label)
    data = frame.loc[:, list(columns)].copy()
    _session_dates(data.session_date_et, label)
    for column in ("ticker", *(("security_id",) if stock else ())):
        if not data[column].map(lambda value: isinstance(value, str) and bool(value) and value.strip() == value).all():
            raise DataReadinessError(f"{label} contains invalid {column}")
    keys = ["security_id", "session_date_et"] if stock else ["session_date_et"]
    if data.duplicated(keys).any() or not data.session_date_et.isin(sessions).all():
        raise DataReadinessError(f"{label} contains duplicate or off-calendar sessions")
    if not stock and not data.ticker.eq("SPY").all():
        raise DataReadinessError("SPY bars contain a substituted benchmark")
    if not data.price_feed.eq("sip").all() or not data.adjustment.eq(sources.price_adjustment).all():
        raise DataReadinessError(f"{label} require SIP and the bound price adjustment")
    if not data.source.eq("alpaca").all() or not data.timeframe.eq("1d").all():
        raise DataReadinessError(f"{label} require physical Alpaca daily source rows")
    policy = "observed" if sources.availability_semantics == "observed" else "market_interval_close"
    if not data.availability_policy.eq(policy).all():
        raise DataReadinessError(f"{label} physical availability policy differs from bound source semantics")
    for column in _CLOCK_COLUMNS:
        data[column] = _utc(data[column], f"{label}.{column}")
    calendar = xcals.get_calendar("XNYS")
    opens = {day: calendar.session_open(pd.Timestamp(day)) for day in set(data.session_date_et)}
    closes = {day: calendar.session_close(pd.Timestamp(day)) for day in set(data.session_date_et)}
    # canonicalize_bars already maps provider daily timestamps to these intervals.
    # Validate the supplied canonical bounds; do not normalize midnight rows here.
    for column, expected in (("bar_start_utc", opens), ("bar_end_utc", closes)):
        actual = data[column]
        if (actual.notna() & actual.ne(pd.to_datetime(data.session_date_et.map(expected), utc=True))).any():
            raise DataReadinessError(f"{label} interval does not match the complete exchange session")
    if (data.available_at_utc.notna() & data.bar_end_utc.notna() & data.available_at_utc.lt(data.bar_end_utc)).any():
        raise DataReadinessError(f"{label} availability precedes the completed session")
    if policy == "observed" and (
        data.available_at_utc.notna() & data.ingested_at_utc.notna() & data.available_at_utc.lt(data.ingested_at_utc)
    ).any():
        raise DataReadinessError(f"{label} observed availability precedes ingestion")
    for column in ("open", "close", "volume"):
        values = _numeric(data[column], f"{label}.{column}")
        invalid = values.lt(0) if column == "volume" else values.le(0)
        if invalid.any():
            raise DataReadinessError(f"{label} contains invalid {column}")
        data[column] = values
    return data


def _full_window(present: pd.Series, positions: int) -> pd.Series:
    return present.astype("int64").rolling(positions, min_periods=positions).sum().eq(positions)


def _window_clocks(frame: pd.DataFrame, positions: int) -> pd.Series:
    # Integer reduction preserves nanoseconds; pandas rolling max casts to float.
    values = pd.DatetimeIndex(frame.available_at_utc).as_unit("ns").asi8
    maximum = np.full(len(frame), np.iinfo(np.int64).min, dtype=np.int64)
    if len(frame) >= positions:
        windows = np.lib.stride_tricks.sliding_window_view(values, positions)
        maximum[positions - 1:] = windows.max(axis=1)
    return pd.Series(pd.to_datetime(maximum, utc=True), index=frame.index)


def _window_state(frame: pd.DataFrame, columns: Sequence[str], positions: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    complete = _full_window(frame.loc[:, list(columns)].notna().all(axis=1), positions)
    clocks_present = _full_window(frame.loc[:, list(_CLOCK_COLUMNS)].notna().all(axis=1), positions)
    return complete, clocks_present, _window_clocks(frame, positions)


def _combine_clocks(left: pd.Series, right: pd.Series) -> pd.Series:
    a = pd.DatetimeIndex(left).as_unit("ns").asi8
    b = pd.DatetimeIndex(right).as_unit("ns").asi8
    return pd.Series(pd.to_datetime(np.maximum(a, b), utc=True), index=left.index)


def _transform_security(stock: pd.DataFrame, spy: pd.DataFrame, decisions: pd.DataFrame) -> pd.DataFrame:
    close, market_close, volume = stock.close, spy.close, stock.volume
    lagged_volume = volume.shift(1).rolling(20, min_periods=20).mean()
    formulas = (
        close.shift(21) / close.shift(126) - 1.0,
        close.shift(21) / close.shift(252) - 1.0,
        volume / lagged_volume.where(lagged_volume.gt(0)) * ((close / stock.open - 1.0) - (market_close / spy.open - 1.0)),
        ((close / close.shift(5) - 1.0) - (market_close / market_close.shift(5) - 1.0))
        * (market_close / market_close.shift(60) - 1.0),
    )
    stocks = (
        _window_state(stock, ("close",), 127),
        _window_state(stock, ("close",), 253),
        _window_state(stock, ("volume",), 21),
        _window_state(stock, ("close",), 6),
    )
    spy_response = _window_state(spy, ("open", "close"), 1)
    spy_regime = _window_state(spy, ("close",), 61)
    result = decisions.loc[:, ["decision_id"]].copy()
    days = decisions.session_date_et.tolist()
    cutoffs = pd.DatetimeIndex(decisions.decision_time_utc).as_unit("ns").asi8
    for index, name in enumerate(RETURN_RELATIONSHIP_COLUMNS):
        complete, clocks_present, clock = stocks[index]
        if index == 2:
            complete &= stock[["open", "close"]].notna().all(axis=1) & spy_response[0]
            clocks_present &= spy_response[1]
            clock = _combine_clocks(clock, spy_response[2])
        elif index == 3:
            complete &= spy_regime[0]
            clocks_present &= spy_regime[1]
            clock = _combine_clocks(clock, spy_regime[2])
        values = formulas[index].reindex(days).to_numpy(dtype=float, copy=True)
        ready_values = complete.reindex(days).to_numpy(dtype=bool)
        ready_clocks = clocks_present.reindex(days).to_numpy(dtype=bool)
        clocks = pd.DatetimeIndex(clock.reindex(days)).as_unit("ns").asi8.copy()
        reasons = np.full(len(days), "", dtype=object)
        reasons[~ready_values] = "missing_required_session_or_value"
        reasons[ready_values & ~ready_clocks] = "missing_required_bar_clock"
        reasons[ready_values & ready_clocks & (clocks > cutoffs)] = "required_bar_available_after_decision"
        if index == 2:
            zero_baseline = lagged_volume.reindex(days).eq(0).to_numpy(dtype=bool)
            reasons[(reasons == "") & zero_baseline] = "zero_lagged_volume_baseline"
        # A 250-session baseline warm-up cannot shorten a 253-position formula.
        first = stock.ticker.first_valid_index()
        position = stock.index.get_indexer(days)
        first_position = len(stock) if first is None else int(stock.index.get_indexer([first])[0])
        stock_positions = position - first_position + 1
        required_stock = (127, 253, 21, 6)[index]
        too_short = stock_positions < required_stock
        reasons[too_short] = f"insufficient_stock_history_positions_requires_{required_stock}"
        if index == 1:
            reasons[too_short & (stock_positions >= 250)] = "baseline_warmup_250_does_not_supply_253_positions"
        if index >= 2:
            first_spy = spy.ticker.first_valid_index()
            first_spy_position = len(spy) if first_spy is None else int(spy.index.get_indexer([first_spy])[0])
            required_spy = 1 if index == 2 else 61
            spy_short = position - first_spy_position + 1 < required_spy
            reasons[~too_short & spy_short] = f"insufficient_spy_history_positions_requires_{required_spy}"
        reasons[(reasons == "") & ~np.isfinite(values)] = "nonfinite_relationship_result"
        reasons[(reasons == "") & (np.abs(values) > np.finfo(np.float32).max)] = "relationship_exceeds_float32_range"
        unavailable = reasons != ""
        values[unavailable] = np.nan
        clocks[unavailable] = np.iinfo(np.int64).min
        result[name] = values.astype(np.float32)
        result[f"available_at_{name}"] = pd.to_datetime(clocks, utc=True)
        result[f"missing_reason_{name}"] = reasons
    return result


def _baseline_rows(
    expected: pd.DataFrame, baseline: ResearchFeaturePartition, contract: StrategyContract,
) -> pd.DataFrame:
    identity = (*DECISION_KEYS, "session_date_et")
    _require(expected, identity, "expected decisions")
    _require(baseline.rows, identity, "baseline")
    if expected.empty or expected.decision_id.duplicated().any() or baseline.rows.decision_id.duplicated().any():
        raise DataReadinessError("profile requires nonempty unique decisions")
    for column in DECISION_KEYS[:-1]:
        if not expected[column].map(lambda value: isinstance(value, str) and bool(value) and value.strip() == value).all():
            raise DataReadinessError(f"expected decisions contain invalid {column}")
    _session_dates(expected.session_date_et, "expected decisions")
    cutoffs = _utc(expected.decision_time_utc, "expected decisions")
    if cutoffs.isna().any() or not cutoffs.eq(swing_prediction_cutoffs(expected.session_date_et)).all():
        raise DataReadinessError("expected decisions require exact canonical swing cutoffs")
    if expected.session_date_et.lt(date(2019, 7, 9)).any() or expected.duplicated(["security_id", "session_date_et"]).any():
        raise DataReadinessError("profile contains forbidden warm-up decisions or duplicate security sessions")
    if set(expected.decision_id) != set(baseline.rows.decision_id):
        raise DataReadinessError("baseline must preserve the complete expected population")
    rows = baseline.rows.set_index("decision_id").loc[expected.decision_id].reset_index()
    for column in identity:
        actual = rows[column].reset_index(drop=True)
        wanted = expected[column].reset_index(drop=True)
        if column == "decision_time_utc":
            actual, wanted = _utc(actual, "baseline"), _utc(wanted, "expected decisions")
        if actual.isna().any() or not actual.eq(wanted).all():
            raise DataReadinessError(f"baseline differs from frozen {column}")
    names = tuple(swing_model_feature_columns(contract=contract, catalyst=False))
    if len(names) != 120 or baseline.model_columns != names or not set(names).issubset(baseline.availability_columns):
        raise DataReadinessError("baseline requires exactly the existing 120 ordered model inputs and clocks")
    _require(rows, (*names, *baseline.availability_columns.values()), "baseline model inputs")
    forbidden = ("future_", "managed_", "label_", "exit_", "entry_", "barrier_", "forward_", "target_")
    for name in names:
        clock_column = baseline.availability_columns[name]
        if clock_column.startswith(forbidden):
            raise DataReadinessError("baseline availability cannot come from outcomes")
        numeric = _numeric(rows[name], f"baseline.{name}")
        if numeric.abs().gt(np.finfo(np.float32).max).any():
            raise DataReadinessError(f"baseline feature {name} exceeds float32 range")
        clock = _utc(rows[clock_column], f"baseline.{clock_column}")
        if (numeric.notna() & (clock.isna() | clock.gt(_utc(rows.decision_time_utc, "baseline cutoff")))).any():
            raise DataReadinessError(f"baseline feature {name} is unavailable at its cutoff")
    if "feature_profile" in rows and not rows.feature_profile.eq("technical_market").all():
        raise DataReadinessError("only the unchanged technical_market baseline can be extended")
    return rows


def build_return_relationship_profile(
    *, expected_decisions: pd.DataFrame, baseline: ResearchFeaturePartition,
    stock_bars: pd.DataFrame, spy_bars: pd.DataFrame, history_sessions: Sequence[date],
    contract: StrategyContract, sources: ReturnRelationshipSources,
    purpose: Literal["historical_research", "live_construction"] = "historical_research",
) -> ResearchFeaturePartition:
    """Append four raw inputs using identical computation for batch and live rows.

    A single live decision still requires its full trailing history. The caller
    bounds each security/month batch and independently verifies source authorities,
    security mappings, adjustment vintages and baseline lineage before this call.
    Proxy clocks are accepted for explicit historical research, never live use.
    No feature, source, training, promotion or serving acceptance is implied.
    """
    if purpose not in ("historical_research", "live_construction"):
        raise DataReadinessError("unsupported return feature construction purpose")
    if purpose == "live_construction" and sources.availability_semantics != "observed":
        raise DataReadinessError("live construction cannot consume historical proxy availability")
    rows = _baseline_rows(expected_decisions, baseline, contract)
    sessions = _sessions(history_sessions)
    if not rows.session_date_et.isin(sessions).all():
        raise DataReadinessError("history calendar does not cover every decision session")
    stocks = _prepare_bars(stock_bars, stock=True, sessions=sessions, sources=sources)
    spy = _prepare_bars(spy_bars, stock=False, sessions=sessions, sources=sources).set_index("session_date_et").reindex(sessions)
    if not set(stocks.security_id).issubset(set(rows.security_id)):
        raise DataReadinessError("stock history contains foreign security identities")
    additions = (*RETURN_RELATIONSHIP_COLUMNS, *(f"available_at_{name}" for name in RETURN_RELATIONSHIP_COLUMNS),
        *(f"missing_reason_{name}" for name in RETURN_RELATIONSHIP_COLUMNS))
    if set(additions).intersection(rows.columns):
        raise DataReadinessError("relationship output columns already exist in baseline")
    pieces = []
    for security_id, decisions in rows.groupby("security_id", sort=False):
        stock = stocks.loc[stocks.security_id.eq(security_id)].set_index("session_date_et").reindex(sessions)
        tickers = stock.ticker.reindex(decisions.session_date_et).reset_index(drop=True)
        if (tickers.notna() & tickers.ne(decisions.ticker.reset_index(drop=True))).any():
            raise DataReadinessError("stock history differs from decision security/ticker identity")
        pieces.append(_transform_security(stock, spy, decisions))
    derived = pd.concat(pieces, ignore_index=True).set_index("decision_id").loc[rows.decision_id]
    for name in additions:
        rows[name] = derived[name].to_numpy()
    # Preserve baseline values and clocks; only the profile identity changes.
    rows["feature_profile"] = RETURN_RELATIONSHIP_PROFILE
    names = (*baseline.model_columns, *RETURN_RELATIONSHIP_COLUMNS)
    clocks: Mapping[str, str] = {**baseline.availability_columns,
        **{name: f"available_at_{name}" for name in RETURN_RELATIONSHIP_COLUMNS}}
    coverage = {
        name: {
            "available_rows": int(rows[name].notna().sum()),
            "missing_rows": int(rows[name].isna().sum()),
            "missing_reasons": {str(reason): int(count) for reason, count in
                rows.loc[rows[name].isna(), f"missing_reason_{name}"].value_counts().items()},
        }
        for name in RETURN_RELATIONSHIP_COLUMNS
    }
    audit: dict[str, object] = {
        "feature_profile": RETURN_RELATIONSHIP_PROFILE,
        "profile_sha256": return_relationship_profile_sha256(baseline.model_columns, sources),
        "sources": sources.model_dump(mode="json"),
        "construction_purpose": purpose,
        "rows": len(rows), "model_feature_count": len(names),
        "population_preserved": True, "outcome_filtered_rows": 0,
        "baseline_model_inputs_unchanged": True,
        "minimum_session_positions": dict(zip(RETURN_RELATIONSHIP_COLUMNS, RETURN_RELATIONSHIP_SESSION_POSITIONS, strict=True)),
        "long_momentum_warmup": "253_positions_required_never_shorten_to_250",
        "coverage": coverage,
        "complete_model_input_rows": int(rows.loc[:, list(names)].notna().all(axis=1).sum()),
        "baseline_audit": dict(baseline.audit),
        "source_admission": "required_from_publishing_caller",
        "feature_acceptance": "not_established_by_construction",
        "training_eligible": False, "promotion_eligible": False, "serving_eligible": False,
    }
    return ResearchFeaturePartition(rows, names, clocks, audit)
