"""Exact managed-path labels for fixed-clock intraday decision cohorts."""
from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Final

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

import market_predictor.modeling.label_outcomes as label_outcomes
from market_predictor.core.errors import DataReadinessError
from market_predictor.label_paths import (
    IntradayBarrierBatch,
    evaluate_intraday_barrier_paths,
    open_close_return,
)
from market_predictor.modeling.strategy_contract import StrategyContract

INTRADAY_BAR_LABEL_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_labels.v1"
_ONE_MINUTE: Final = pd.Timedelta(minutes=1)
_FEATURE_REQUIRED: Final = frozenset({"ticker", "decision_time_utc", "atr_14_5m", "primary_benchmark"})
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
        "source",
        "price_feed",
        "adjustment",
    }
)
_LABEL_COLUMNS: Final = (
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
    "qqq_return",
    "sector_return",
    "spy_excess_return",
    "qqq_excess_return",
    "sector_excess_return",
    "rank_label",
    "rank_percentile",
    "ranking_group_size",
)


def build_exact_intraday_bar_labels(
    feature_rows: pd.DataFrame,
    stock_one_minute_bars: pd.DataFrame,
    benchmark_one_minute_bars: pd.DataFrame,
    *,
    contract: StrategyContract,
    strategy_contract_sha256: str,
) -> pd.DataFrame:
    """Label fixed-clock decisions from the next exact thirty one-minute bars."""

    _validate_contract(contract, strategy_contract_sha256)
    features = _validate_features(feature_rows)
    stocks = _validate_minute_bars(stock_one_minute_bars, label="stock")
    benchmarks = _validate_minute_bars(
        benchmark_one_minute_bars,
        label="benchmark",
    )
    output = _empty_label_columns(features)
    stock_groups = {
        str(ticker): group.set_index("bar_start_utc", drop=False).sort_index()
        for ticker, group in stocks.groupby("ticker", sort=False, observed=True)
    }
    benchmark_groups = {
        str(ticker): group.set_index("bar_start_utc", drop=False).sort_index()
        for ticker, group in benchmarks.groupby("ticker", sort=False, observed=True)
    }

    feature_eligible = pd.Series(True, index=output.index, dtype=bool)
    if "feature_eligible" in output.columns:
        if not is_bool_dtype(output["feature_eligible"]) or bool(output["feature_eligible"].isna().any()):
            raise DataReadinessError("intraday bar feature eligibility must be non-null boolean")
        feature_eligible = output["feature_eligible"].astype(bool)
        feature_ineligible = ~feature_eligible
        output.loc[feature_ineligible, "label_ineligible_reason"] = "feature_ineligible"
        if "feature_ineligible_reason" in output.columns:
            source_reasons = output["feature_ineligible_reason"].astype("string")
            has_source_reason = feature_ineligible & source_reasons.notna() & source_reasons.str.strip().ne("")
            output.loc[has_source_reason, "label_ineligible_reason"] = (
                "feature_ineligible:" + source_reasons.loc[has_source_reason].str.strip()
            )

    candidates: list[dict[str, Any]] = []
    for row in output.loc[feature_eligible].itertuples():
        index = row.Index
        atr = row.atr_14_5m
        if pd.isna(atr):
            output.at[index, "label_ineligible_reason"] = "missing_atr_14_5m"
            continue
        if not np.isfinite(float(atr)) or float(atr) <= 0:
            output.at[index, "label_ineligible_reason"] = "invalid_atr_14_5m"
            continue
        benchmark = row.primary_benchmark
        if pd.isna(benchmark) or not str(benchmark).strip():
            output.at[index, "label_ineligible_reason"] = "missing_primary_benchmark"
            continue

        decision_time = pd.Timestamp(row.decision_time_utc)
        entry_time = decision_time + _ONE_MINUTE
        expected = pd.date_range(
            entry_time,
            periods=contract.intraday.horizon_minutes,
            freq="1min",
            tz="UTC",
        )
        ticker = str(row.ticker)
        stock = stock_groups.get(ticker)
        if stock is None:
            output.at[index, "label_ineligible_reason"] = "missing_exact_stock_one_minute_path"
            continue
        path = stock.reindex(expected)
        if bool(path["bar_start_utc"].isna().any()):
            output.at[index, "label_ineligible_reason"] = "missing_exact_stock_one_minute_path"
            continue
        candidates.append(
            {
                "row_index": index,
                "ticker": ticker,
                "sector_ticker": str(benchmark).upper().strip(),
                "decision_time": decision_time,
                "entry_time": entry_time,
                "path": path,
                "atr": float(atr),
            }
        )

    if candidates:
        paths = evaluate_intraday_barrier_paths(
            path_open=np.stack([item["path"]["open"].to_numpy(dtype="float64") for item in candidates]),
            path_high=np.stack([item["path"]["high"].to_numpy(dtype="float64") for item in candidates]),
            path_low=np.stack([item["path"]["low"].to_numpy(dtype="float64") for item in candidates]),
            path_close=np.stack([item["path"]["close"].to_numpy(dtype="float64") for item in candidates]),
            entry_atr=np.asarray([item["atr"] for item in candidates], dtype="float64"),
            target_atr=contract.intraday.target_atr_multiple,
            stop_atr=contract.intraday.stop_atr_multiple,
            round_trip_cost_bps=contract.intraday.round_trip_cost_bps,
        )
        for position, candidate in enumerate(candidates):
            _apply_candidate(
                output,
                candidate,
                position=position,
                paths=paths,
                benchmark_groups=benchmark_groups,
                round_trip_cost_bps=contract.intraday.round_trip_cost_bps,
            )

    output = _add_fixed_cohort_ranks(output, contract)
    return output.sort_values(["decision_time_utc", "ticker"], kind="stable").reset_index(drop=True)


def _apply_candidate(
    output: pd.DataFrame,
    candidate: dict[str, Any],
    *,
    position: int,
    paths: IntradayBarrierBatch,
    benchmark_groups: dict[str, pd.DataFrame],
    round_trip_cost_bps: float,
) -> None:
    index = candidate["row_index"]
    offset = int(paths.outcome_offset[position])
    entry_time = pd.Timestamp(candidate["entry_time"])
    exit_time = entry_time + offset * _ONE_MINUTE
    returns: dict[str, float] = {}
    evidence: list[pd.Timestamp] = list(
        pd.to_datetime(
            candidate["path"].iloc[: offset + 1]["available_at_utc"],
            utc=True,
            errors="raise",
        )
    )
    benchmark_specs = (
        ("spy", "SPY"),
        ("qqq", "QQQ"),
        ("sector", candidate["sector_ticker"]),
    )
    for name, ticker in benchmark_specs:
        group = benchmark_groups.get(str(ticker))
        if group is None:
            output.at[index, "label_ineligible_reason"] = f"missing_exact_{name}_interval"
            return
        expected = pd.date_range(entry_time, exit_time, freq="1min", tz="UTC")
        interval = group.reindex(expected)
        if bool(interval["bar_start_utc"].isna().any()):
            output.at[index, "label_ineligible_reason"] = f"missing_exact_{name}_interval"
            return
        entry_row = interval.iloc[0]
        exit_row = interval.iloc[-1]
        value = open_close_return(
            np.asarray([entry_row["open"]], dtype="float64"),
            np.asarray([exit_row["close"]], dtype="float64"),
        )[0]
        if not np.isfinite(value):
            output.at[index, "label_ineligible_reason"] = f"invalid_exact_{name}_interval"
            return
        returns[name] = float(value)
        evidence.extend(
            pd.to_datetime(
                interval["available_at_utc"],
                utc=True,
                errors="raise",
            )
        )

    outcome = str(paths.outcome[position])
    target = float(paths.target_price[position])
    stop = float(paths.stop_price[position])
    path = candidate["path"]
    collision = bool(outcome == "stop_first" and float(path.iloc[offset]["high"]) >= target and float(path.iloc[offset]["low"]) <= stop)
    reason = {
        "target_first": "target_touched_first",
        "stop_first": ("same_minute_collision_stop_first" if collision else "stop_touched_first"),
        "timeout": "timeout_at_horizon",
    }[outcome]
    barrier = {
        "target_first": label_outcomes.TARGET_HIT,
        "stop_first": label_outcomes.STOP_HIT,
        "timeout": label_outcomes.TIMEOUT,
    }[outcome]
    net = float(paths.net_return[position])
    output.loc[index, list(_LABEL_UPDATE_COLUMNS)] = [
        True,
        pd.NA,
        pd.Timestamp(candidate["decision_time"]).isoformat(),
        entry_time,
        entry_time + _ONE_MINUTE,
        float(path.iloc[0]["open"]),
        target,
        stop,
        exit_time,
        exit_time + _ONE_MINUTE,
        max(evidence),
        float(paths.realized_price[position]),
        offset + 1,
        barrier,
        outcome,
        reason,
        bool(paths.target_first[position]),
        bool(paths.stop_first[position]),
        bool(paths.timeout[position]),
        float(paths.gross_return[position]),
        float(round_trip_cost_bps) / 10_000.0,
        net,
        returns["spy"],
        returns["qqq"],
        returns["sector"],
        net - returns["spy"],
        net - returns["qqq"],
        net - returns["sector"],
    ]


_LABEL_UPDATE_COLUMNS: Final = (
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
    "qqq_return",
    "sector_return",
    "spy_excess_return",
    "qqq_excess_return",
    "sector_excess_return",
)


def _add_fixed_cohort_ranks(
    frame: pd.DataFrame,
    contract: StrategyContract,
) -> pd.DataFrame:
    output = frame.copy()
    eligible = output["label_eligible"].astype(bool)
    grouped = output.loc[eligible].groupby("decision_time_utc", sort=False)
    output.loc[eligible, "ranking_group_size"] = grouped["ticker"].transform("size")
    minimum = contract.labels.intraday_minimum_cross_section_for_ranking
    for decision_time, indices in grouped.groups.items():
        if len(indices) < minimum:
            continue
        if not output.loc[indices, "decision_time_utc"].eq(decision_time).all():
            raise DataReadinessError("intraday rank cohort mixed decision timestamps")
        percentile = output.loc[indices, "net_return"].astype(float).rank(pct=True, method="average")
        labels = pd.Series(
            label_outcomes.RANK_MIDDLE,
            index=indices,
            dtype="Int64",
        )
        labels.loc[
            percentile > 1.0 - contract.labels.rank_top_quantile
        ] = label_outcomes.RANK_TOP
        labels.loc[
            percentile <= contract.labels.rank_bottom_quantile
        ] = label_outcomes.RANK_BOTTOM
        output.loc[indices, "rank_percentile"] = percentile
        output.loc[indices, "rank_label"] = labels
    return output


def _empty_label_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    collisions = sorted(set(_LABEL_COLUMNS).intersection(output.columns))
    if collisions:
        raise DataReadinessError(f"feature rows already contain intraday bar label columns: {collisions}")
    output["label_schema_version"] = INTRADAY_BAR_LABEL_SCHEMA_VERSION
    output["label_eligible"] = False
    output["label_ineligible_reason"] = pd.Series(pd.NA, index=output.index, dtype="string")
    output["decision_group_id"] = pd.Series(pd.NA, index=output.index, dtype="string")
    for column in (
        "entry_time_utc",
        "entry_bar_end_utc",
        "exit_time_utc",
        "exit_bar_end_utc",
        "label_available_at_utc",
    ):
        output[column] = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
    for column in (
        "entry_price",
        "target_price",
        "stop_price",
        "exit_price",
        "gross_return",
        "cost",
        "net_return",
        "spy_return",
        "qqq_return",
        "sector_return",
        "spy_excess_return",
        "qqq_excess_return",
        "sector_excess_return",
        "rank_percentile",
    ):
        output[column] = np.nan
    for column in ("holding_minutes", "barrier_label", "rank_label", "ranking_group_size"):
        output[column] = pd.Series(pd.NA, index=output.index, dtype="Int64")
    for column in ("label_outcome", "label_outcome_reason"):
        output[column] = pd.Series(pd.NA, index=output.index, dtype="string")
    for column in ("target_hit", "stop_hit", "timeout"):
        output[column] = pd.Series(pd.NA, index=output.index, dtype="boolean")
    return output


def _validate_contract(contract: StrategyContract, expected_sha256: str) -> None:
    intraday = contract.intraday
    if not expected_sha256 or expected_sha256 != contract.sha256():
        raise DataReadinessError("intraday bar label strategy contract hash differs")
    if (
        intraday.horizon_minutes != 30
        or intraday.entry_reference != "next_one_minute_open"
        or intraday.exit_rule != "target_stop_timeout"
        or intraday.atr_timeframe != "5Min"
        or intraday.atr_lookback_bars != 14
        or intraday.target_atr_multiple != 2.0
        or intraday.stop_atr_multiple != 1.5
        or intraday.round_trip_cost_bps != 10.0
        or not contract.labels.rank_labels_enabled
        or contract.labels.intraday_rank_within_sector
    ):
        raise DataReadinessError("intraday bar label contract is not the frozen design")


def _validate_features(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _FEATURE_REQUIRED, "fixed-cohort feature rows")
    if frame.empty:
        raise DataReadinessError("fixed-cohort feature rows are empty")
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["decision_time_utc"] = pd.to_datetime(data["decision_time_utc"], utc=True, errors="raise")
    data["atr_14_5m"] = pd.to_numeric(data["atr_14_5m"], errors="coerce")
    if (
        bool(data["ticker"].eq("").any())
        or bool(data["decision_time_utc"].isna().any())
        or bool(data["decision_time_utc"].dt.second.ne(0).any())
        or bool(data["decision_time_utc"].dt.microsecond.ne(0).any())
        or bool(data.duplicated(["ticker", "decision_time_utc"]).any())
    ):
        raise DataReadinessError("fixed-cohort feature identity is invalid")
    if "feature_available_at_utc" in data.columns:
        available = pd.to_datetime(data["feature_available_at_utc"], utc=True, errors="raise")
        if bool(available.isna().any()) or bool(available.gt(data["decision_time_utc"]).any()):
            raise DataReadinessError("fixed-cohort features contain evidence after decision_time_utc")
        data["feature_available_at_utc"] = available
    return data


def _validate_minute_bars(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    _require_columns(frame, _MINUTE_REQUIRED, f"{label} one-minute bars")
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    for column in ("bar_start_utc", "bar_end_utc", "available_at_utc"):
        data[column] = pd.to_datetime(data[column], utc=True, errors="raise")
    if bool(
        data[["bar_start_utc", "bar_end_utc", "available_at_utc"]]
        .isna()
        .any()
        .any()
    ):
        raise DataReadinessError(
            f"{label} one-minute bar timestamps or availability are incomplete"
        )
    for column in ("open", "high", "low", "close"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    prices = data[["open", "high", "low", "close"]]
    if (
        bool(data["ticker"].eq("").any())
        or bool(data.duplicated(["ticker", "bar_start_utc"]).any())
        or bool(data["timeframe"].astype(str).str.lower().ne("1m").any())
        or bool(data["source"].astype(str).str.lower().ne("alpaca").any())
        or bool(data["price_feed"].astype(str).str.lower().ne("sip").any())
        or bool(data["adjustment"].astype(str).str.lower().ne("all").any())
        or not np.isfinite(prices.to_numpy(dtype="float64")).all()
        or bool(prices.le(0).any().any())
        or bool(data["bar_end_utc"].ne(data["bar_start_utc"] + _ONE_MINUTE).any())
        or bool(data["available_at_utc"].lt(data["bar_end_utc"]).any())
        or bool(data["high"].lt(data[["open", "close"]].max(axis=1)).any())
        or bool(data["low"].gt(data[["open", "close"]].min(axis=1)).any())
        or bool(data["high"].lt(data["low"]).any())
    ):
        raise DataReadinessError(f"{label} one-minute bar identity is invalid")
    return data.sort_values(["ticker", "bar_start_utc"], kind="stable").reset_index(drop=True)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise DataReadinessError(f"{label} omit required columns: {missing}")
