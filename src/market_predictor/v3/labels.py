from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from typing import Literal, Self

import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator

from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError
from market_predictor.v3.partitions import DevelopmentShadowPolicy, assert_development_only
from market_predictor.v3.schema import ML_V3_SCHEMA_VERSION, FrozenContract


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
    ambiguous_barrier_policy: Literal["stop", "target"] = "stop"
    schema_version: str = ML_V3_SCHEMA_VERSION

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
    records: list[dict[str, object]] = []
    for (_, _), session in data.groupby(["ticker", "_session_date_et"], sort=False):
        records.extend(_label_session(session.reset_index(drop=True), benchmark_lookup, config))
    if not records:
        return pd.DataFrame()
    labeled = pd.DataFrame(records).sort_values(["decision_time_utc", "ticker"]).reset_index(drop=True)
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
    benchmark_lookup: pd.DataFrame,
    config: V3LabelConfig,
) -> Iterable[dict[str, object]]:
    maximum_horizon = max(config.horizons_bars)
    if len(session) <= maximum_horizon:
        return []
    output: list[dict[str, object]] = []
    cost = config.round_trip_cost_bps / 10_000.0
    qqq = "QQQ"
    label_config_json = json.dumps(config.model_dump(mode="json"), separators=(",", ":"), sort_keys=True)
    label_config_hash = hashlib.sha256(label_config_json.encode()).hexdigest()
    for decision_index in range(len(session) - maximum_horizon):
        decision = session.iloc[decision_index]
        entry_index = decision_index + 1
        entry = session.iloc[entry_index]
        entry_price = float(entry["open"])
        if entry_price <= 0 or float(decision["atr_14"]) <= 0:
            continue
        record: dict[str, object] = {
            key: value for key, value in decision.to_dict().items() if key != "_session_date_et"
        }
        record.update({
            "ticker": decision["ticker"],
            "decision_time_utc": decision["timestamp"],
            "feature_available_at_utc": decision["timestamp"],
            "entry_time_utc": entry["timestamp"],
            "session_date_et": decision["_session_date_et"],
            "decision_group_id": pd.Timestamp(decision["timestamp"]).isoformat(),
            "universe_snapshot_id": decision["universe_snapshot_id"],
            "price_feed": str(decision["price_feed"]).lower().strip(),
            "feature_schema_version": decision.get("feature_schema_version", ML_V3_SCHEMA_VERSION),
            "label_schema_version": config.schema_version,
            "label_config_json": label_config_json,
            "label_config_hash": label_config_hash,
            "entry_price": entry_price,
            "primary_benchmark": str(decision["primary_benchmark"]).upper().strip(),
            "_source_decision_index": decision_index,
            "_source_entry_index": entry_index,
            "_source_exit_index": decision_index + config.primary_horizon_bars,
        })
        for horizon in config.horizons_bars:
            exit_index = decision_index + horizon
            future = session.iloc[entry_index : exit_index + 1]
            exit_bar = session.iloc[exit_index]
            suffix = f"{horizon * config.bar_minutes}m"
            net_return = float(exit_bar["close"]) / entry_price - 1.0 - cost
            qqq_return = _benchmark_return(benchmark_lookup, qqq, entry["timestamp"], exit_bar["timestamp"])
            sector_return = _benchmark_return(
                benchmark_lookup,
                str(decision["primary_benchmark"]).upper().strip(),
                entry["timestamp"],
                exit_bar["timestamp"],
            )
            if qqq_return is None or sector_return is None:
                raise DataReadinessError(
                    f"missing exact benchmark interval for {decision['ticker']} at {decision['timestamp']}"
                )
            favorable = future["high"].astype(float) / entry_price - 1.0
            adverse = future["low"].astype(float) / entry_price - 1.0
            record[f"net_return_{suffix}"] = net_return
            record[f"qqq_return_{suffix}"] = qqq_return
            record[f"sector_return_{suffix}"] = sector_return
            record[f"net_excess_qqq_{suffix}"] = net_return - qqq_return
            record[f"net_excess_sector_{suffix}"] = net_return - sector_return
            record[f"mfe_{suffix}"] = float(favorable.max())
            record[f"mae_{suffix}"] = float(adverse.min())
            record[f"bars_to_mfe_{suffix}"] = int(np.argmax(favorable.to_numpy())) + 1
            record[f"bars_to_mae_{suffix}"] = int(np.argmin(adverse.to_numpy())) + 1
        session_close_bar = session.iloc[-1]
        session_close = float(session_close_bar["close"])
        net_return_to_close = session_close / entry_price - 1.0 - cost
        record["net_return_to_close"] = net_return_to_close
        qqq_to_close = _benchmark_return(benchmark_lookup, qqq, entry["timestamp"], session_close_bar["timestamp"])
        sector_to_close = _benchmark_return(
            benchmark_lookup,
            str(decision["primary_benchmark"]).upper().strip(),
            entry["timestamp"],
            session_close_bar["timestamp"],
        )
        if qqq_to_close is None or sector_to_close is None:
            raise DataReadinessError(
                f"missing session-close benchmark interval for {decision['ticker']} at {decision['timestamp']}"
            )
        record["qqq_return_to_close"] = qqq_to_close
        record["sector_return_to_close"] = sector_to_close
        record["net_excess_qqq_to_close"] = net_return_to_close - qqq_to_close
        record["net_excess_sector_to_close"] = net_return_to_close - sector_to_close
        record.update(_path_target(session, decision_index, entry_price, float(decision["atr_14"]), cost, config))
        output.append(record)
    return output


def _benchmark_return(
    lookup: pd.DataFrame,
    symbol: str,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
) -> float | None:
    try:
        entry = lookup.loc[(symbol, entry_time)]
        exit_bar = lookup.loc[(symbol, exit_time)]
    except KeyError:
        return None
    if isinstance(entry, pd.DataFrame) or isinstance(exit_bar, pd.DataFrame):
        raise DataReadinessError(f"duplicate benchmark rows for {symbol}")
    entry_price = float(entry["open"])
    return None if entry_price <= 0 else float(exit_bar["close"]) / entry_price - 1.0


def _path_target(
    session: pd.DataFrame,
    decision_index: int,
    entry_price: float,
    atr: float,
    cost: float,
    config: V3LabelConfig,
) -> dict[str, object]:
    entry_index = decision_index + 1
    exit_index = decision_index + config.primary_horizon_bars
    target_price = entry_price + config.target_atr * atr
    stop_price = entry_price - config.stop_atr * atr
    outcome = "timeout"
    outcome_bar = config.primary_horizon_bars
    realized_price = float(session.iloc[exit_index]["close"])
    for offset, (_, bar) in enumerate(session.iloc[entry_index : exit_index + 1].iterrows(), start=1):
        hit_target = float(bar["high"]) >= target_price
        hit_stop = float(bar["low"]) <= stop_price
        if hit_target and hit_stop:
            outcome = f"{config.ambiguous_barrier_policy}_first"
            realized_price = stop_price if config.ambiguous_barrier_policy == "stop" else target_price
        elif hit_target:
            outcome = "target_first"
            realized_price = target_price
        elif hit_stop:
            outcome = "stop_first"
            realized_price = stop_price
        else:
            continue
        outcome_bar = offset
        break
    return {
        "path_outcome": outcome,
        "target_before_stop": int(outcome == "target_first"),
        "stop_before_target": int(outcome == "stop_first"),
        "path_timeout": int(outcome == "timeout"),
        "path_outcome_bar": outcome_bar,
        "path_realized_return_net": realized_price / entry_price - 1.0 - cost,
        "target_price": target_price,
        "stop_price": stop_price,
    }


def _add_overlap_metadata(frame: pd.DataFrame, config: V3LabelConfig) -> pd.DataFrame:
    output = frame.copy()
    output["concurrent_label_count"] = 1
    output["overlap_weight"] = 1.0
    output["independent_event_id"] = pd.NA
    output["cooldown_bars"] = config.evaluation_cooldown_bars
    for (_, _), indices in output.groupby(["ticker", "session_date_et"], sort=False).groups.items():
        group = output.loc[indices].sort_values("_source_entry_index")
        maximum_position = int(group["_source_exit_index"].max()) + 1
        concurrency = np.zeros(maximum_position, dtype=int)
        for _, row in group.iterrows():
            concurrency[int(row["_source_entry_index"]) : int(row["_source_exit_index"]) + 1] += 1
        last_exit = -1
        event_number = 0
        for index, row in group.iterrows():
            start = int(row["_source_entry_index"])
            stop = int(row["_source_exit_index"])
            active = concurrency[start : stop + 1]
            output.at[index, "concurrent_label_count"] = int(active.max())
            output.at[index, "overlap_weight"] = float(np.mean(1.0 / active))
            if start > last_exit + config.evaluation_cooldown_bars:
                event_number += 1
                output.at[index, "independent_event_id"] = f"{row['ticker']}:{row['session_date_et']}:{event_number}"
                last_exit = stop
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
