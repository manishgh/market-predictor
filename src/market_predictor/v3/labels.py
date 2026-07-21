from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError
from market_predictor.v3.partitions import DevelopmentShadowPolicy, assert_development_only
from market_predictor.v3.schema import ML_V3_SCHEMA_VERSION, FrozenContract

V3_LABEL_SCHEMA_VERSION = "ml_v3.labels.v2"


class V3LabelConfig(FrozenContract):
    horizons_bars: tuple[int, ...] = (6, 12, 24)
    primary_horizon_bars: int = 12
    bar_minutes: int = Field(default=5, ge=1)
    round_trip_cost_bps: float = Field(default=10.0, ge=0)
    target_atr: float = Field(default=1.5, gt=0)
    stop_atr: float = Field(default=1.0, gt=0)
    minimum_ranking_group: int = Field(default=20, ge=2)
    ranking_grades: int = Field(default=5, ge=2, le=10)
    evaluation_cooldown_bars: int = Field(default=0, ge=0)
    decision_stride_bars: int = Field(default=1, ge=1)
    rotate_decision_offset_by_session: bool = False
    decision_start_minute_et: int = Field(default=9 * 60 + 30, ge=0, le=1_439)
    decision_end_minute_et: int = Field(default=16 * 60, ge=1, le=1_440)
    ambiguous_barrier_policy: Literal["stop", "target"] = "stop"
    schema_version: str = V3_LABEL_SCHEMA_VERSION

    @field_validator("horizons_bars")
    @classmethod
    def validate_horizons(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        normalized = tuple(sorted(set(value)))
        if not normalized or any(horizon < 1 for horizon in normalized):
            raise ValueError("horizons_bars must contain positive integers")
        return normalized

    @model_validator(mode="after")
    def validate_primary_horizon(self) -> Self:
        if self.primary_horizon_bars not in self.horizons_bars:
            raise ValueError("primary_horizon_bars must be present in horizons_bars")
        if self.decision_end_minute_et <= self.decision_start_minute_et:
            raise ValueError("decision session end must be later than its start")
        return self


def build_v3_labels(
    bars: pd.DataFrame,
    benchmarks: pd.DataFrame,
    *,
    config: V3LabelConfig = V3LabelConfig(),
    partition: Literal["development", "shadow"] = "development",
    shadow_policy: DevelopmentShadowPolicy = DevelopmentShadowPolicy(timestamp_column="timestamp"),
) -> pd.DataFrame:
    """Build cost-adjusted, point-in-time V3 labels from next-open entries."""
    data = _prepare_bars(bars, name="bars", require_context=True)
    benchmark_data = _prepare_bars(benchmarks, name="benchmarks", require_context=False)
    if partition == "development":
        assert_development_only(data, policy=shadow_policy)
    elif bool((data["timestamp"] <= shadow_policy.development_cutoff_utc).any()):
        raise DataReadinessError("shadow label input contains development rows")
    benchmark_lookup = benchmark_data.set_index(["ticker", "timestamp"])
    sessions: list[pd.DataFrame] = []
    for (_, _), session in data.groupby(["ticker", "_session_date_et"], sort=False):
        labeled_session = _label_session(session.reset_index(drop=True), config)
        if not labeled_session.empty:
            sessions.append(labeled_session)
    if not sessions:
        return pd.DataFrame()
    labeled = pd.concat(sessions, ignore_index=True).sort_values(["decision_time_utc", "ticker"]).reset_index(drop=True)
    labeled = _add_benchmark_targets(labeled, benchmark_lookup, config)
    labeled = _add_overlap_metadata(labeled, config)
    return _add_ranking_grades(labeled, config)


def _prepare_bars(frame: pd.DataFrame, *, name: str, require_context: bool) -> pd.DataFrame:
    required = {"ticker", "timestamp", "open", "high", "low", "close", "volume"}
    if require_context:
        required.update({"atr_14", "primary_benchmark", "universe_snapshot_id", "price_feed"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SchemaMismatchError(f"{name} missing columns: {', '.join(missing)}")
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    if isinstance(data["timestamp"].dtype, pd.DatetimeTZDtype):
        data["timestamp"] = data["timestamp"].dt.tz_convert("UTC")
    else:
        data["timestamp"] = data["timestamp"].map(_aware_timestamp)
    if bool(data["timestamp"].isna().any()):
        raise DataReadinessError(f"{name} contains invalid or timezone-naive timestamps")
    numeric_columns = ["open", "high", "low", "close", "volume"]
    if require_context:
        numeric_columns.append("atr_14")
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")
    if bool(data[numeric_columns].isna().any(axis=None)):
        raise DataReadinessError(f"{name} contains non-numeric prices, volume, or ATR")
    if bool(data.duplicated(["ticker", "timestamp"]).any()):
        raise DataReadinessError(f"{name} contains duplicate ticker/timestamp bars")
    data["_session_date_et"] = data["timestamp"].dt.tz_convert("America/New_York").dt.date
    return data.sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def _label_session(
    session: pd.DataFrame,
    config: V3LabelConfig,
) -> pd.DataFrame:
    maximum_horizon = max(config.horizons_bars)
    if len(session) <= maximum_horizon:
        return pd.DataFrame()
    cost = config.round_trip_cost_bps / 10_000.0
    label_config_json = json.dumps(config.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)
    label_config_hash = hashlib.sha256(label_config_json.encode()).hexdigest()
    eastern = session["timestamp"].dt.tz_convert("America/New_York")
    minute_et = (eastern.dt.hour * 60 + eastern.dt.minute).to_numpy(dtype=int)
    regular_indices = np.flatnonzero(
        (minute_et >= config.decision_start_minute_et) & (minute_et < config.decision_end_minute_et)
    )
    if len(regular_indices) == 0:
        return pd.DataFrame()
    regular_close_index = int(regular_indices[-1])
    offset = 0
    if config.rotate_decision_offset_by_session and config.decision_stride_bars > 1:
        session_date = pd.Timestamp(session.iloc[0]["_session_date_et"]).date()
        offset = int(session_date.toordinal()) % config.decision_stride_bars
    decision_indices = regular_indices[offset :: config.decision_stride_bars]
    decision_indices = decision_indices[decision_indices < len(session) - maximum_horizon]
    if len(decision_indices) == 0:
        return pd.DataFrame()
    maximum_exit_indices = decision_indices + maximum_horizon
    decision_indices = decision_indices[minute_et[maximum_exit_indices] < config.decision_end_minute_et]
    if len(decision_indices) == 0:
        return pd.DataFrame()
    timestamp_ns = pd.DatetimeIndex(session["timestamp"]).as_unit("ns").asi8
    maximum_exit_indices = decision_indices + maximum_horizon
    expected_horizon_ns = maximum_horizon * config.bar_minutes * 60 * 1_000_000_000
    exact_interval = timestamp_ns[maximum_exit_indices] - timestamp_ns[decision_indices] == expected_horizon_ns
    decision_indices = decision_indices[exact_interval]
    if len(decision_indices) == 0:
        return pd.DataFrame()

    opens = session["open"].to_numpy(dtype=float)
    highs = session["high"].to_numpy(dtype=float)
    lows = session["low"].to_numpy(dtype=float)
    closes = session["close"].to_numpy(dtype=float)
    timestamps = session["timestamp"].to_numpy()
    atr = session["atr_14"].to_numpy(dtype=float)[decision_indices]
    entry_indices = decision_indices + 1
    entry_prices = opens[entry_indices]
    valid = (entry_prices > 0) & (atr > 0)
    decision_indices = decision_indices[valid]
    entry_indices = entry_indices[valid]
    entry_prices = entry_prices[valid]
    atr = atr[valid]
    if len(decision_indices) == 0:
        return pd.DataFrame()

    output = session.iloc[decision_indices].drop(columns="_session_date_et").reset_index(drop=True).copy()
    decision_times = timestamps[decision_indices]
    entry_times = timestamps[entry_indices]
    primary_benchmarks = output["primary_benchmark"].astype(str).str.upper().str.strip().to_numpy()
    output["decision_time_utc"] = decision_times
    output["feature_available_at_utc"] = decision_times
    output["entry_time_utc"] = entry_times
    output["primary_exit_time_utc"] = timestamps[decision_indices + config.primary_horizon_bars]
    output["session_date_et"] = session.iloc[decision_indices]["_session_date_et"].to_numpy()
    output["decision_group_id"] = [pd.Timestamp(value).isoformat() for value in decision_times]
    output["price_feed"] = output["price_feed"].astype(str).str.lower().str.strip()
    if "feature_schema_version" not in output.columns:
        output["feature_schema_version"] = ML_V3_SCHEMA_VERSION
    output["label_schema_version"] = config.schema_version
    output["label_config_json"] = label_config_json
    output["label_config_hash"] = label_config_hash
    output["entry_price"] = entry_prices
    output["primary_benchmark"] = primary_benchmarks
    output["_source_decision_index"] = decision_indices
    output["_source_entry_index"] = entry_indices
    output["_source_exit_index"] = decision_indices + config.primary_horizon_bars

    for horizon in config.horizons_bars:
        exit_indices = decision_indices + horizon
        net_return = closes[exit_indices] / entry_prices - 1.0 - cost
        path_indices = entry_indices[:, None] + np.arange(horizon)
        favorable = highs[path_indices] / entry_prices[:, None] - 1.0
        adverse = lows[path_indices] / entry_prices[:, None] - 1.0
        suffix = f"{horizon * config.bar_minutes}m"
        output[f"_exit_time_{suffix}"] = timestamps[exit_indices]
        output[f"net_return_{suffix}"] = net_return
        output[f"mfe_{suffix}"] = favorable.max(axis=1)
        output[f"mae_{suffix}"] = adverse.min(axis=1)
        output[f"bars_to_mfe_{suffix}"] = favorable.argmax(axis=1) + 1
        output[f"bars_to_mae_{suffix}"] = adverse.argmin(axis=1) + 1

    session_close_time = timestamps[regular_close_index]
    net_return_to_close = closes[regular_close_index] / entry_prices - 1.0 - cost
    output["_session_close_time"] = np.full(len(output), session_close_time, dtype=object)
    output["net_return_to_close"] = net_return_to_close
    for column, values in _path_targets(
        highs=highs,
        lows=lows,
        closes=closes,
        decision_indices=decision_indices,
        entry_prices=entry_prices,
        atr=atr,
        cost=cost,
        config=config,
    ).items():
        output[column] = values
    return output


def _add_benchmark_targets(
    frame: pd.DataFrame,
    benchmark_lookup: pd.DataFrame,
    config: V3LabelConfig,
) -> pd.DataFrame:
    output = frame.copy()
    entry_times = output["entry_time_utc"].to_numpy()
    primary_benchmarks = output["primary_benchmark"].astype(str).str.upper().str.strip().to_numpy()
    qqq = np.full(len(output), "QQQ", dtype=object)
    for horizon in config.horizons_bars:
        suffix = f"{horizon * config.bar_minutes}m"
        exit_times = output.pop(f"_exit_time_{suffix}").to_numpy()
        qqq_return = _benchmark_returns(benchmark_lookup, qqq, entry_times, exit_times)
        sector_return = _benchmark_returns(benchmark_lookup, primary_benchmarks, entry_times, exit_times)
        missing = np.isnan(qqq_return) | np.isnan(sector_return)
        if bool(missing.any()):
            row = output.iloc[int(np.flatnonzero(missing)[0])]
            raise DataReadinessError(f"missing exact benchmark interval for {row['ticker']} at {row['timestamp']}")
        net_return = output[f"net_return_{suffix}"].to_numpy(dtype=float)
        output[f"qqq_return_{suffix}"] = qqq_return
        output[f"sector_return_{suffix}"] = sector_return
        output[f"net_excess_qqq_{suffix}"] = net_return - qqq_return
        output[f"net_excess_sector_{suffix}"] = net_return - sector_return

    close_times = output.pop("_session_close_time").to_numpy()
    qqq_to_close = _benchmark_returns(benchmark_lookup, qqq, entry_times, close_times)
    sector_to_close = _benchmark_returns(benchmark_lookup, primary_benchmarks, entry_times, close_times)
    missing_close = np.isnan(qqq_to_close) | np.isnan(sector_to_close)
    if bool(missing_close.any()):
        row = output.iloc[int(np.flatnonzero(missing_close)[0])]
        raise DataReadinessError(f"missing session-close benchmark interval for {row['ticker']} at {row['timestamp']}")
    net_return_to_close = output["net_return_to_close"].to_numpy(dtype=float)
    output["qqq_return_to_close"] = qqq_to_close
    output["sector_return_to_close"] = sector_to_close
    output["net_excess_qqq_to_close"] = net_return_to_close - qqq_to_close
    output["net_excess_sector_to_close"] = net_return_to_close - sector_to_close
    return output


def _benchmark_returns(
    lookup: pd.DataFrame,
    symbols: np.ndarray,
    entry_times: np.ndarray,
    exit_times: np.ndarray,
) -> np.ndarray:
    entry_keys = pd.MultiIndex.from_arrays([symbols, pd.DatetimeIndex(entry_times)], names=["ticker", "timestamp"])
    exit_keys = pd.MultiIndex.from_arrays([symbols, pd.DatetimeIndex(exit_times)], names=["ticker", "timestamp"])
    entry_price = pd.to_numeric(lookup["open"].reindex(entry_keys), errors="coerce").to_numpy(dtype=float)
    exit_price = pd.to_numeric(lookup["close"].reindex(exit_keys), errors="coerce").to_numpy(dtype=float)
    result = exit_price / entry_price - 1.0
    result[(entry_price <= 0) | ~np.isfinite(entry_price) | ~np.isfinite(exit_price)] = np.nan
    return np.asarray(result, dtype=float)


def _path_targets(
    *,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    decision_indices: np.ndarray,
    entry_prices: np.ndarray,
    atr: np.ndarray,
    cost: float,
    config: V3LabelConfig,
) -> dict[str, np.ndarray]:
    horizon = config.primary_horizon_bars
    target_price = entry_prices + config.target_atr * atr
    stop_price = entry_prices - config.stop_atr * atr
    path_indices = decision_indices[:, None] + 1 + np.arange(horizon)
    hit_target = highs[path_indices] >= target_price[:, None]
    hit_stop = lows[path_indices] <= stop_price[:, None]
    no_hit = horizon + 1
    first_target = np.where(hit_target.any(axis=1), hit_target.argmax(axis=1) + 1, no_hit)
    first_stop = np.where(hit_stop.any(axis=1), hit_stop.argmax(axis=1) + 1, no_hit)
    target_first = first_target < first_stop
    stop_first = first_stop < first_target
    simultaneous = (first_target == first_stop) & (first_target <= horizon)
    if config.ambiguous_barrier_policy == "target":
        target_first |= simultaneous
    else:
        stop_first |= simultaneous
    outcome = np.full(len(decision_indices), "timeout", dtype=object)
    outcome[target_first] = "target_first"
    outcome[stop_first] = "stop_first"
    realized_price = closes[decision_indices + horizon].copy()
    realized_price[target_first] = target_price[target_first]
    realized_price[stop_first] = stop_price[stop_first]
    outcome_bar = np.minimum(np.minimum(first_target, first_stop), horizon)
    return {
        "path_outcome": outcome,
        "target_before_stop": target_first.astype(int),
        "stop_before_target": stop_first.astype(int),
        "path_timeout": (~(target_first | stop_first)).astype(int),
        "path_outcome_bar": outcome_bar,
        "path_realized_return_net": realized_price / entry_prices - 1.0 - cost,
        "target_price": target_price,
        "stop_price": stop_price,
    }


def _add_overlap_metadata(frame: pd.DataFrame, config: V3LabelConfig) -> pd.DataFrame:
    output = frame.copy()
    concurrent_label_count = np.ones(len(output), dtype=int)
    overlap_weight = np.ones(len(output), dtype=float)
    independent_event_id = np.full(len(output), None, dtype=object)
    output["cooldown_bars"] = config.evaluation_cooldown_bars
    entry = output["_source_entry_index"].to_numpy(dtype=int)
    exit_ = output["_source_exit_index"].to_numpy(dtype=int)
    ticker = output["ticker"].astype(str).to_numpy()
    session_date = output["session_date_et"].astype(str).to_numpy()
    for (_, _), indices in output.groupby(["ticker", "session_date_et"], sort=False).groups.items():
        positions = np.asarray(indices, dtype=int)
        positions = positions[np.argsort(entry[positions], kind="stable")]
        maximum_position = int(exit_[positions].max()) + 1
        concurrency = np.zeros(maximum_position, dtype=int)
        for position in positions:
            concurrency[entry[position] : exit_[position] + 1] += 1
        last_exit = -1
        event_number = 0
        for position in positions:
            start = entry[position]
            stop = exit_[position]
            active = concurrency[start : stop + 1]
            concurrent_label_count[position] = int(active.max())
            overlap_weight[position] = float(np.mean(1.0 / active))
            if start > last_exit + config.evaluation_cooldown_bars:
                event_number += 1
                independent_event_id[position] = f"{ticker[position]}:{session_date[position]}:{event_number}"
                last_exit = stop
    output["concurrent_label_count"] = concurrent_label_count
    output["overlap_weight"] = overlap_weight
    output["independent_event_id"] = independent_event_id
    return output


def _add_ranking_grades(frame: pd.DataFrame, config: V3LabelConfig) -> pd.DataFrame:
    output = frame.copy()
    target = f"net_excess_qqq_{config.primary_horizon_bars * config.bar_minutes}m"
    output["ranking_target"] = output[target]
    output["ranking_grade"] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    output["ranking_group_size"] = output.groupby("decision_group_id")["ticker"].transform("size")
    for _, indices in output.groupby("decision_group_id", sort=False).groups.items():
        if len(indices) < config.minimum_ranking_group:
            continue
        values = output.loc[indices, "ranking_target"]
        quality = values.rank(method="first", ascending=True) - 1
        grades = np.floor(quality / (len(values) - 1) * (config.ranking_grades - 1) + 1e-12).astype(int)
        output.loc[indices, "ranking_grade"] = grades.to_numpy()
    return output.drop(columns=["_source_decision_index", "_source_entry_index", "_source_exit_index"])


def _aware_timestamp(value: object) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        return pd.NaT
    return timestamp.tz_convert("UTC")
