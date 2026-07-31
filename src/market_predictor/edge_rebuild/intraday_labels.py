"""Exact, causal intraday labels from executable one-minute paths."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import time
from typing import Final, TypedDict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from market_predictor.edge_rebuild.intraday_features import FEATURE_SCHEMA_VERSION
from market_predictor.edge_rebuild.labeling import (
    RANK_BOTTOM,
    RANK_MIDDLE,
    RANK_TOP,
    STOP_HIT,
    TARGET_HIT,
    TIMEOUT,
)
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.execution_policy import executable_fill_price
from market_predictor.v3.errors import DataReadinessError

LABEL_SCHEMA_VERSION: Final = "edge_rebuild.intraday_labels.v1"
EXCHANGE_TIMEZONE: Final = ZoneInfo("America/New_York")
_REGULAR_OPEN: Final = time(9, 30)
_REGULAR_CLOSE: Final = time(16, 0)
_ONE_MINUTE: Final = pd.Timedelta(minutes=1)

_FEATURE_REQUIRED: Final = frozenset(
    {
        "ticker",
        "security_id",
        "session_date_et",
        "session_open_utc",
        "session_close_utc",
        "volume_bar_number",
        "available_at_utc",
        "feature_available_at_utc",
        "feature_eligible",
        "feature_ineligible_reason",
        "atr_14",
        "primary_benchmark",
        "universe_snapshot_id",
        "source",
        "price_feed",
        "adjustment",
        "source_timeframe",
        "feature_schema_version",
        "strategy_contract_sha256",
    }
)
_MINUTE_REQUIRED: Final = frozenset(
    {
        "ticker",
        "timeframe",
        "bar_start_utc",
        "bar_end_utc",
        "available_at_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "price_feed",
        "adjustment",
    }
)
_MATERIAL_LABEL_COLUMNS: Final = (
    "label_schema_version",
    "label_eligible",
    "label_ineligible_reason",
    "decision_group_id",
    "entry_time_utc",
    "entry_bar_end_utc",
    "entry_price",
    "target_price",
    "stop_price",
    "exit_time_utc",
    "exit_bar_end_utc",
    "label_available_at_utc",
    "exit_price",
    "holding_minutes",
    "barrier_label",
    "label_outcome",
    "label_outcome_reason",
    "target_hit",
    "stop_hit",
    "timeout",
    "gross_return",
    "cost",
    "net_return",
    "spy_return",
    "sector_return",
    "spy_excess_return",
    "sector_excess_return",
    "rank_label",
    "rank_percentile",
    "ranking_group_size",
)


class _PathResult(TypedDict):
    entry_price: float
    target_price: float
    stop_price: float
    outcome: str
    barrier_label: int
    outcome_offset: int
    reason: str
    exit_price: float
    gross_return: float
    cost: float
    net_return: float


def build_exact_causal_intraday_labels(
    feature_rows: pd.DataFrame,
    stock_one_minute_bars: pd.DataFrame,
    benchmark_one_minute_bars: pd.DataFrame,
    *,
    contract: StrategyContract,
    strategy_contract_sha256: str,
) -> pd.DataFrame:
    """Label each decision from the next exact executable one-minute open.

    Missing execution evidence abstains only the affected decision. Invalid
    schema, source identity, timestamps, or OHLCV rejects the complete input.
    Barriers are checked on the entry minute and the following 29 exact minutes;
    paths never continue into another exchange session.
    """

    _validate_contract(contract, strategy_contract_sha256)
    features = _validate_features(
        feature_rows,
        strategy_contract_sha256=strategy_contract_sha256,
    )
    stocks = _validate_minute_bars(stock_one_minute_bars, label="stock")
    benchmarks = _validate_minute_bars(
        benchmark_one_minute_bars,
        label="benchmark",
    )

    stock_groups = {
        key: group.set_index("bar_start_utc", drop=False).sort_index()
        for key, group in stocks.groupby(["ticker", "session_date_et"], sort=False, observed=True)
    }
    benchmark_lookup = benchmarks.set_index(["ticker", "bar_start_utc"], drop=False).sort_index()
    output = _empty_label_columns(features.copy())

    for row_index, row in output.iterrows():
        if not bool(row["feature_eligible"]):
            source_reason = row["feature_ineligible_reason"]
            reason = "feature_ineligible"
            if pd.notna(source_reason) and str(source_reason).strip():
                reason = f"feature_ineligible:{str(source_reason).strip()}"
            output.at[row_index, "label_ineligible_reason"] = reason
            continue

        session_key = (str(row["ticker"]), row["session_date_et"])
        session_bars = stock_groups.get(session_key)
        if session_bars is None:
            output.at[row_index, "label_ineligible_reason"] = "missing_exact_next_minute"
            continue

        available_at = pd.Timestamp(row["feature_available_at_utc"])
        entry_time = available_at.floor("min") + _ONE_MINUTE
        session_close = pd.Timestamp(row["session_close_utc"])
        horizon_end = entry_time + contract.intraday.horizon_minutes * _ONE_MINUTE
        if horizon_end > session_close:
            output.at[row_index, "label_ineligible_reason"] = "horizon_crosses_session_close"
            continue
        if entry_time not in session_bars.index:
            output.at[row_index, "label_ineligible_reason"] = "missing_exact_next_minute"
            continue

        expected_starts = pd.date_range(
            entry_time,
            periods=contract.intraday.horizon_minutes,
            freq="1min",
            tz="UTC",
        )
        path = session_bars.reindex(expected_starts)
        if bool(path["bar_start_utc"].isna().any()):
            output.at[row_index, "label_ineligible_reason"] = "missing_exact_one_minute_path"
            continue

        atr = float(row["atr_14"])
        path_result = _evaluate_path(
            path,
            atr=atr,
            target_atr_multiple=contract.intraday.target_atr_multiple,
            stop_atr_multiple=contract.intraday.stop_atr_multiple,
            round_trip_cost_bps=contract.intraday.round_trip_cost_bps,
        )
        exit_time = pd.Timestamp(expected_starts[path_result["outcome_offset"]])
        spy_return = _benchmark_return(
            benchmark_lookup,
            ticker=contract.labels.benchmark_market,
            entry_time=entry_time,
            exit_time=exit_time,
        )
        sector_return = _benchmark_return(
            benchmark_lookup,
            ticker=str(row["primary_benchmark"]),
            entry_time=entry_time,
            exit_time=exit_time,
        )
        if spy_return is None:
            output.at[row_index, "label_ineligible_reason"] = "missing_exact_spy_interval"
            continue
        if sector_return is None:
            output.at[row_index, "label_ineligible_reason"] = "missing_exact_sector_interval"
            continue

        output.at[row_index, "label_eligible"] = True
        output.at[row_index, "label_ineligible_reason"] = pd.NA
        output.at[row_index, "decision_group_id"] = available_at.isoformat()
        output.at[row_index, "entry_time_utc"] = entry_time
        output.at[row_index, "entry_bar_end_utc"] = entry_time + _ONE_MINUTE
        output.at[row_index, "entry_price"] = path_result["entry_price"]
        output.at[row_index, "target_price"] = path_result["target_price"]
        output.at[row_index, "stop_price"] = path_result["stop_price"]
        output.at[row_index, "exit_time_utc"] = exit_time
        output.at[row_index, "exit_bar_end_utc"] = exit_time + _ONE_MINUTE
        output.at[row_index, "label_available_at_utc"] = path.loc[exit_time, "available_at_utc"]
        output.at[row_index, "exit_price"] = path_result["exit_price"]
        output.at[row_index, "holding_minutes"] = path_result["outcome_offset"] + 1
        output.at[row_index, "barrier_label"] = path_result["barrier_label"]
        output.at[row_index, "label_outcome"] = path_result["outcome"]
        output.at[row_index, "label_outcome_reason"] = path_result["reason"]
        output.at[row_index, "target_hit"] = path_result["outcome"] == "target_first"
        output.at[row_index, "stop_hit"] = path_result["outcome"] == "stop_first"
        output.at[row_index, "timeout"] = path_result["outcome"] == "timeout"
        output.at[row_index, "gross_return"] = path_result["gross_return"]
        output.at[row_index, "cost"] = path_result["cost"]
        output.at[row_index, "net_return"] = path_result["net_return"]
        output.at[row_index, "spy_return"] = spy_return
        output.at[row_index, "sector_return"] = sector_return
        output.at[row_index, "spy_excess_return"] = path_result["net_return"] - spy_return
        output.at[row_index, "sector_excess_return"] = path_result["net_return"] - sector_return

    output = _add_contemporaneous_rank(output, contract)
    return output.sort_values(
        ["session_date_et", "feature_available_at_utc", "ticker"],
        kind="stable",
    ).reset_index(drop=True)


def _evaluate_path(
    path: pd.DataFrame,
    *,
    atr: float,
    target_atr_multiple: float,
    stop_atr_multiple: float,
    round_trip_cost_bps: float,
) -> _PathResult:
    opens = path["open"].to_numpy(dtype="float64")
    highs = path["high"].to_numpy(dtype="float64")
    lows = path["low"].to_numpy(dtype="float64")
    closes = path["close"].to_numpy(dtype="float64")
    entry = float(opens[0])
    target = entry + target_atr_multiple * atr
    stop = entry - stop_atr_multiple * atr
    if stop <= 0:
        raise DataReadinessError("intraday label stop price must remain positive")

    target_hits = highs >= target
    stop_hits = lows <= stop
    missing = len(path) + 1
    first_target = int(np.argmax(target_hits)) if target_hits.any() else missing
    first_stop = int(np.argmax(stop_hits)) if stop_hits.any() else missing
    collision = first_target == first_stop and first_stop < missing
    if first_stop <= first_target and first_stop < missing:
        outcome = "stop_first"
        barrier_label = STOP_HIT
        outcome_offset = first_stop
        reason = "same_minute_collision_stop_first" if collision else "stop_touched_first"
    elif first_target < first_stop:
        outcome = "target_first"
        barrier_label = TARGET_HIT
        outcome_offset = first_target
        reason = "target_touched_first"
    else:
        outcome = "timeout"
        barrier_label = TIMEOUT
        outcome_offset = len(path) - 1
        reason = "timeout_at_horizon"

    exit_price = executable_fill_price(
        outcome=outcome,
        target_price=target,
        stop_price=stop,
        trigger_open=float(opens[outcome_offset]),
        final_price=float(closes[-1]),
    )
    gross = exit_price / entry - 1.0
    cost = round_trip_cost_bps / 10_000.0
    return {
        "entry_price": entry,
        "target_price": target,
        "stop_price": stop,
        "outcome": outcome,
        "barrier_label": barrier_label,
        "outcome_offset": outcome_offset,
        "reason": reason,
        "exit_price": exit_price,
        "gross_return": gross,
        "cost": cost,
        "net_return": gross - cost,
    }


def _benchmark_return(
    lookup: pd.DataFrame,
    *,
    ticker: str,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
) -> float | None:
    normalized = ticker.upper().strip()
    entry_key = (normalized, entry_time)
    exit_key = (normalized, exit_time)
    if entry_key not in lookup.index or exit_key not in lookup.index:
        return None
    entry_row = lookup.loc[entry_key]
    exit_row = lookup.loc[exit_key]
    if isinstance(entry_row, pd.DataFrame) or isinstance(exit_row, pd.DataFrame):
        raise DataReadinessError("benchmark minute identity is not unique")
    return float(exit_row["close"]) / float(entry_row["open"]) - 1.0


def _add_contemporaneous_rank(
    frame: pd.DataFrame,
    contract: StrategyContract,
) -> pd.DataFrame:
    output = frame.copy()
    eligible = output["label_eligible"].astype(bool)
    output.loc[eligible, "ranking_group_size"] = output.loc[eligible].groupby("decision_group_id", sort=False)["ticker"].transform("size")
    for _, indices in output.loc[eligible].groupby("decision_group_id", sort=False).groups.items():
        if len(indices) < contract.labels.intraday_minimum_cross_section_for_ranking:
            continue
        returns = output.loc[indices, "net_return"].astype(float)
        percentile = returns.rank(pct=True, method="average")
        labels = pd.Series(RANK_MIDDLE, index=indices, dtype="Int64")
        labels.loc[percentile > 1.0 - contract.labels.rank_top_quantile] = RANK_TOP
        labels.loc[percentile <= contract.labels.rank_bottom_quantile] = RANK_BOTTOM
        output.loc[indices, "rank_percentile"] = percentile
        output.loc[indices, "rank_label"] = labels
    return output


def _empty_label_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame["label_schema_version"] = LABEL_SCHEMA_VERSION
    frame["label_eligible"] = False
    frame["label_ineligible_reason"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    frame["decision_group_id"] = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in (
        "entry_time_utc",
        "entry_bar_end_utc",
        "exit_time_utc",
        "exit_bar_end_utc",
        "label_available_at_utc",
    ):
        frame[column] = pd.Series(
            pd.NaT,
            index=frame.index,
            dtype="datetime64[ns, UTC]",
        )
    for column in (
        "entry_price",
        "target_price",
        "stop_price",
        "exit_price",
        "gross_return",
        "cost",
        "net_return",
        "spy_return",
        "sector_return",
        "spy_excess_return",
        "sector_excess_return",
        "rank_percentile",
    ):
        frame[column] = np.nan
    for column in ("holding_minutes", "barrier_label", "rank_label", "ranking_group_size"):
        frame[column] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    for column in ("label_outcome", "label_outcome_reason"):
        frame[column] = pd.Series(pd.NA, index=frame.index, dtype="string")
    for column in ("target_hit", "stop_hit", "timeout"):
        frame[column] = pd.Series(pd.NA, index=frame.index, dtype="boolean")
    return frame


def _validate_contract(
    contract: StrategyContract,
    strategy_contract_sha256: str,
) -> None:
    if not strategy_contract_sha256 or strategy_contract_sha256 != contract.sha256():
        raise DataReadinessError("intraday label strategy contract hash does not match")
    intraday = contract.intraday
    if (
        intraday.entry_reference != "next_one_minute_open"
        or intraday.exit_rule != "target_stop_timeout"
        or intraday.horizon_minutes != 30
        or intraday.target_atr_multiple != 2.0
        or intraday.stop_atr_multiple != 1.5
        or intraday.round_trip_cost_bps != 10.0
        or intraday.atr_timeframe != "5Min"
        or contract.labels.benchmark_market != "SPY"
        or contract.labels.intraday_rank_within_sector
    ):
        raise DataReadinessError("intraday label contract is not the frozen causal design")


def _validate_features(
    frame: pd.DataFrame,
    *,
    strategy_contract_sha256: str,
) -> pd.DataFrame:
    _require_columns(frame, _FEATURE_REQUIRED, "intraday features")
    conflict = sorted(set(_MATERIAL_LABEL_COLUMNS).intersection(frame.columns))
    if conflict:
        raise DataReadinessError(f"intraday features already contain label columns: {conflict}")
    if frame.empty:
        raise DataReadinessError("intraday features are empty")
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["primary_benchmark"] = data["primary_benchmark"].astype(str).str.upper().str.strip()
    data["session_date_et"] = pd.to_datetime(data["session_date_et"], errors="raise").dt.date
    for column in (
        "available_at_utc",
        "feature_available_at_utc",
        "session_open_utc",
        "session_close_utc",
    ):
        data[column] = pd.to_datetime(data[column], utc=True, errors="raise")
    data["atr_14"] = pd.to_numeric(data["atr_14"], errors="coerce")
    data["volume_bar_number"] = pd.to_numeric(data["volume_bar_number"], errors="coerce")
    identity_ok = (
        data["source"].astype(str).str.lower().str.strip().eq("alpaca").all()
        and data["price_feed"].astype(str).str.lower().str.strip().eq("sip").all()
        and data["adjustment"].astype(str).str.lower().str.strip().eq("all").all()
        and data["source_timeframe"].astype(str).str.lower().str.strip().eq("1m").all()
        and data["feature_schema_version"].astype(str).eq(FEATURE_SCHEMA_VERSION).all()
        and data["strategy_contract_sha256"].astype(str).eq(strategy_contract_sha256).all()
    )
    text_columns = [
        "ticker",
        "security_id",
        "primary_benchmark",
        "universe_snapshot_id",
    ]
    text_invalid = data[text_columns].apply(lambda values: values.isna() | values.astype(str).str.strip().eq(""))
    local_open = data["session_open_utc"].dt.tz_convert(EXCHANGE_TIMEZONE)
    local_close = data["session_close_utc"].dt.tz_convert(EXCHANGE_TIMEZONE)
    if not is_bool_dtype(data["feature_eligible"]):
        raise DataReadinessError("intraday feature eligibility must be boolean")
    eligible = data["feature_eligible"]
    invalid = (
        not bool(identity_ok)
        or bool(text_invalid.any().any())
        or bool(data.duplicated(["ticker", "session_date_et", "volume_bar_number"]).any())
        or bool(data["session_open_utc"].ge(data["session_close_utc"]).any())
        or bool(local_open.dt.date.ne(data["session_date_et"]).any())
        or bool(local_close.dt.date.ne(data["session_date_et"]).any())
        or bool(local_open.dt.time.ne(_REGULAR_OPEN).any())
        or bool(local_close.dt.time.gt(_REGULAR_CLOSE).any())
        or bool(data["feature_available_at_utc"].lt(data["available_at_utc"]).any())
        or bool(data["feature_available_at_utc"].ge(data["session_close_utc"]).any())
        or bool((data["volume_bar_number"] % 1).ne(0).any())
        or bool(data["volume_bar_number"].le(0).any())
        or bool((~np.isfinite(data.loc[eligible, "atr_14"])).any())
        or bool(data.loc[eligible, "atr_14"].le(0).any())
    )
    if invalid:
        raise DataReadinessError("intraday features contain invalid identity, timestamps, ATR, or lineage")
    return data


def _validate_minute_bars(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    _require_columns(frame, _MINUTE_REQUIRED, f"{label} one-minute bars")
    if frame.empty:
        raise DataReadinessError(f"{label} one-minute bars are empty")
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    for column in ("bar_start_utc", "bar_end_utc", "available_at_utc"):
        data[column] = pd.to_datetime(data[column], utc=True, errors="raise")
    for column in ("open", "high", "low", "close", "volume"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    identity_ok = (
        data["timeframe"].astype(str).str.lower().str.strip().eq("1m").all()
        and data["source"].astype(str).str.lower().str.strip().eq("alpaca").all()
        and data["price_feed"].astype(str).str.lower().str.strip().eq("sip").all()
        and data["adjustment"].astype(str).str.lower().str.strip().eq("all").all()
    )
    prices = data[["open", "high", "low", "close", "volume"]]
    local_start = data["bar_start_utc"].dt.tz_convert(EXCHANGE_TIMEZONE)
    local_end = data["bar_end_utc"].dt.tz_convert(EXCHANGE_TIMEZONE)
    invalid = (
        not bool(identity_ok)
        or bool(data["ticker"].eq("").any())
        or not bool(np.isfinite(prices.to_numpy(dtype="float64")).all())
        or bool(prices.le(0.0).any().any())
        or bool(data["high"].lt(data[["open", "close"]].max(axis=1)).any())
        or bool(data["low"].gt(data[["open", "close"]].min(axis=1)).any())
        or bool(data["high"].lt(data["low"]).any())
        or bool(data["bar_start_utc"].dt.second.ne(0).any())
        or bool(data["bar_start_utc"].dt.microsecond.ne(0).any())
        or bool((data["bar_end_utc"] - data["bar_start_utc"]).ne(_ONE_MINUTE).any())
        or bool(data["available_at_utc"].lt(data["bar_end_utc"]).any())
        or bool(local_start.dt.date.ne(local_end.dt.date).any())
        or bool(local_start.dt.time.lt(_REGULAR_OPEN).any())
        or bool(local_end.dt.time.gt(_REGULAR_CLOSE).any())
        or bool(data.duplicated(["ticker", "bar_start_utc"]).any())
    )
    if invalid:
        raise DataReadinessError(f"{label} labels require unique, completed, regular-session Alpaca SIP/all 1m OHLCV bars")
    data["session_date_et"] = local_start.dt.date
    return data.sort_values(["ticker", "bar_start_utc"], kind="stable").reset_index(drop=True)


def _require_columns(
    frame: pd.DataFrame,
    required: Iterable[str],
    label: str,
) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise DataReadinessError(f"{label} are missing columns: {missing}")
