from __future__ import annotations

import numpy as np
import pandas as pd

from market_predictor.execution_policy import executable_fill_prices
from market_predictor.intraday.contracts import (
    IntradayDatasetConfig,
    downside_target_column,
    excess_return_column,
    net_return_column,
    opportunity_target_column,
)


def add_exact_one_minute_labels(
    frame: pd.DataFrame,
    one_minute_bars: pd.DataFrame,
    config: IntradayDatasetConfig,
) -> pd.DataFrame:
    output = frame.copy()
    horizon = config.horizon_minutes // config.execution_bar_minutes
    result_parts: list[pd.DataFrame] = []
    for ticker, decisions in output.groupby("ticker", sort=False):
        bars = one_minute_bars[one_minute_bars["ticker"].eq(ticker)]
        result_parts.append(_label_ticker(decisions.copy(), bars, config, horizon))
    output = pd.concat(result_parts, ignore_index=True)
    output = _add_benchmark_label_returns(output, one_minute_bars, config)
    net_col = net_return_column(config.horizon_minutes)
    invalid = (
        ~output["label_path_exact"].fillna(False).astype(bool)
        | output[net_col].isna()
        | output[excess_return_column(config.horizon_minutes, "sector")].isna()
    )
    label_columns = [
        opportunity_target_column(config.horizon_minutes),
        downside_target_column(config.horizon_minutes),
        f"path_timeout_{config.horizon_minutes}m",
        f"path_realized_return_gross_{config.horizon_minutes}m",
        net_col,
        f"path_mfe_{config.horizon_minutes}m",
        f"path_mae_{config.horizon_minutes}m",
        f"path_spy_return_{config.horizon_minutes}m",
        f"path_qqq_return_{config.horizon_minutes}m",
        f"path_sector_return_{config.horizon_minutes}m",
        excess_return_column(config.horizon_minutes, "spy"),
        excess_return_column(config.horizon_minutes, "qqq"),
        excess_return_column(config.horizon_minutes, "sector"),
    ]
    output.loc[invalid, label_columns] = np.nan
    output["label_eligible"] = (
        output["feature_eligible"].fillna(False).astype(bool)
        & output["label_path_exact"].fillna(False).astype(bool)
        & output[opportunity_target_column(config.horizon_minutes)].notna()
        & output[downside_target_column(config.horizon_minutes)].notna()
    )
    return output


def add_overlap_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach concurrency, average uniqueness, and independent-event identity.

    ``overlap_weight`` is the exact average uniqueness of a label: the mean of
    ``1 / concurrency`` over each one-minute bar the label spans, where
    ``concurrency`` is the number of same-ticker/session labels active at that
    bar (half-open ``[entry, label_window_end)`` intervals). This replaces the
    prior single-reciprocal ``1 / concurrent_count`` proxy so a staggered
    horizon is weighted by how uniquely it observes each bar. ``concurrent_label_count``
    is the peak concurrency over the span; ``independent_event_id`` numbers a
    greedy non-overlapping subset for strictly non-overlapping evaluation.
    """

    data = frame.copy()
    data["concurrent_label_count"] = 0
    data["overlap_weight"] = 0.0
    data["independent_event_id"] = pd.Series(pd.NA, index=data.index, dtype="string")
    bar_ns = 60 * 1_000_000_000  # one-minute execution bar
    for (ticker, session), indices in data.groupby(["ticker", "session_date_et"], sort=False).groups.items():
        part = data.loc[indices]
        part = part[part["label_path_exact"].fillna(False).astype(bool)].sort_values("entry_time_utc", kind="stable")
        if part.empty:
            continue
        start_bar = pd.DatetimeIndex(part["entry_time_utc"]).as_unit("ns").asi8 // bar_ns
        end_bar = pd.DatetimeIndex(part["label_window_end_utc"]).as_unit("ns").asi8 // bar_ns
        origin = int(start_bar.min())
        starts = (start_bar - origin).astype(int)
        ends = np.maximum((end_bar - origin).astype(int), starts + 1)
        concurrency = np.zeros(int(ends.max()), dtype=int)
        for start, end in zip(starts, ends, strict=True):
            concurrency[start:end] += 1
        counts = np.empty(len(part), dtype=int)
        weights = np.empty(len(part), dtype=float)
        event_ids: list[object] = [pd.NA] * len(part)
        last_end = np.iinfo(np.int64).min
        number = 0
        for position, (start, end) in enumerate(zip(starts, ends, strict=True)):
            active = concurrency[start:end]
            counts[position] = int(active.max())
            weights[position] = float(np.mean(1.0 / active))
            if start >= last_end:
                number += 1
                event_ids[position] = f"{ticker}:{session}:{number}"
                last_end = end
        data.loc[part.index, "concurrent_label_count"] = counts
        data.loc[part.index, "overlap_weight"] = weights
        data.loc[part.index, "independent_event_id"] = pd.array(event_ids, dtype="string")
    return data


def _label_ticker(
    decisions: pd.DataFrame,
    bars: pd.DataFrame,
    config: IntradayDatasetConfig,
    horizon: int,
) -> pd.DataFrame:
    data = decisions.sort_values("decision_time_utc", kind="stable").reset_index(drop=True).copy()
    _initialize_label_columns(data, config)
    if bars.empty or data.empty:
        return data
    bars = bars.sort_values("bar_start_utc", kind="stable").reset_index(drop=True)
    starts = pd.DatetimeIndex(bars["bar_start_utc"]).as_unit("ns").asi8
    decisions_ns = pd.DatetimeIndex(data["decision_time_utc"]).as_unit("ns").asi8
    entry_index = np.searchsorted(starts, decisions_ns, side="left")
    in_bounds = entry_index + horizon <= len(bars)
    expected = data["session_minute_et"].le(16 * 60 - config.horizon_minutes)
    data["label_window_expected"] = expected.to_numpy(bool)
    candidate_positions = np.flatnonzero(in_bounds & expected.to_numpy(bool))
    if len(candidate_positions) == 0:
        return data
    offsets = np.arange(horizon, dtype=np.int64)
    path_indices = entry_index[candidate_positions, None] + offsets[None, :]
    path_starts = starts[path_indices]
    expected_starts = starts[entry_index[candidate_positions], None] + offsets[None, :] * 60_000_000_000
    path_sessions = bars["session_date_et"].to_numpy(object)[path_indices]
    decision_sessions = data["session_date_et"].to_numpy(object)[candidate_positions, None]
    exact = (path_starts == expected_starts).all(axis=1) & (path_sessions == decision_sessions).all(axis=1)
    positions = candidate_positions[exact]
    if len(positions) == 0:
        return data
    path_indices = path_indices[exact]
    entry_indices = entry_index[positions]
    open_price = bars["open"].to_numpy(float)[entry_indices]
    atr = pd.to_numeric(data["atr_14_price_5m"], errors="coerce").to_numpy(float)[positions]
    target_price = open_price + config.target_atr * atr
    stop_price = open_price - config.stop_atr * atr
    highs = bars["high"].to_numpy(float)[path_indices]
    lows = bars["low"].to_numpy(float)[path_indices]
    closes = bars["close"].to_numpy(float)[path_indices]
    hit_target = highs >= target_price[:, None]
    hit_stop = lows <= stop_price[:, None]
    missing = horizon + 1
    first_target = np.where(hit_target.any(axis=1), hit_target.argmax(axis=1), missing)
    first_stop = np.where(hit_stop.any(axis=1), hit_stop.argmax(axis=1), missing)
    target_first = first_target < first_stop
    stop_first = (first_stop <= first_target) & (first_stop < missing)
    timeout = ~(target_first | stop_first)
    outcome_offset = np.minimum(np.minimum(first_target, first_stop), horizon - 1)
    outcome_labels = np.full(len(positions), "timeout", dtype=object)
    outcome_labels[target_first] = "target_first"
    outcome_labels[stop_first] = "stop_first"
    opens = bars["open"].to_numpy(float)[path_indices]
    trigger_open = opens[np.arange(len(positions)), outcome_offset]
    realized = executable_fill_prices(
        outcome=outcome_labels,
        target_price=target_price,
        stop_price=stop_price,
        trigger_open=trigger_open,
        final_price=closes[:, -1],
    )
    active = offsets[None, :] <= outcome_offset[:, None]
    mfe_high = np.where(active, highs, -np.inf).max(axis=1)
    mae_low = np.where(active, lows, np.inf).min(axis=1)
    exit_indices = entry_indices + outcome_offset
    entry_volume = bars["volume"].to_numpy(float)[entry_indices]
    entry_dollar_volume = open_price * entry_volume
    entry_atr_pct = np.divide(atr, open_price, out=np.full_like(atr, np.nan), where=open_price > 0)
    gross = realized / open_price - 1.0
    net = gross - config.round_trip_cost_bps / 10_000.0
    valid_price = np.isfinite(open_price) & (open_price > 0) & np.isfinite(atr) & (atr > 0)
    positions = positions[valid_price]
    if len(positions) == 0:
        return data
    select = valid_price
    data.loc[positions, "label_path_exact"] = True
    data.loc[positions, "entry_time_utc"] = bars["bar_start_utc"].to_numpy()[entry_indices[select]]
    data.loc[positions, "entry_price"] = open_price[select]
    data.loc[positions, "target_price"] = target_price[select]
    data.loc[positions, "stop_price"] = stop_price[select]
    data.loc[positions, "entry_target_pct"] = target_price[select] / open_price[select] - 1.0
    data.loc[positions, "entry_stop_pct"] = open_price[select] / stop_price[select] - 1.0
    data.loc[positions, "exit_time_utc"] = bars["bar_end_utc"].to_numpy()[exit_indices[select]]
    data.loc[positions, "label_available_at_utc"] = bars["available_at_utc"].to_numpy()[exit_indices[select]]
    data.loc[positions, "label_window_end_utc"] = bars["bar_end_utc"].to_numpy()[entry_indices[select] + horizon - 1]
    data.loc[positions, "path_outcome"] = outcome_labels[select]
    data.loc[positions, "path_outcome_bar"] = outcome_offset[select] + 1
    data.loc[positions, "entry_dollar_volume"] = entry_dollar_volume[select]
    data.loc[positions, "entry_atr_pct"] = entry_atr_pct[select]
    data.loc[positions, opportunity_target_column(config.horizon_minutes)] = target_first[select].astype(int)
    data.loc[positions, downside_target_column(config.horizon_minutes)] = stop_first[select].astype(int)
    data.loc[positions, f"path_timeout_{config.horizon_minutes}m"] = timeout[select].astype(int)
    data.loc[positions, f"path_realized_return_gross_{config.horizon_minutes}m"] = gross[select]
    data.loc[positions, net_return_column(config.horizon_minutes)] = net[select]
    data.loc[positions, f"path_mfe_{config.horizon_minutes}m"] = mfe_high[select] / open_price[select] - 1.0
    data.loc[positions, f"path_mae_{config.horizon_minutes}m"] = mae_low[select] / open_price[select] - 1.0
    return data


def _initialize_label_columns(data: pd.DataFrame, config: IntradayDatasetConfig) -> None:
    data["label_window_expected"] = False
    data["label_path_exact"] = False
    for column in (
        "entry_time_utc",
        "exit_time_utc",
        "label_available_at_utc",
        "label_window_end_utc",
    ):
        data[column] = pd.Series(pd.NaT, index=data.index, dtype="datetime64[ns, UTC]")
    for column in (
        "entry_price",
        "entry_dollar_volume",
        "entry_atr_pct",
        "target_price",
        "stop_price",
        "entry_target_pct",
        "entry_stop_pct",
        "path_outcome_bar",
        opportunity_target_column(config.horizon_minutes),
        downside_target_column(config.horizon_minutes),
        f"path_timeout_{config.horizon_minutes}m",
        f"path_realized_return_gross_{config.horizon_minutes}m",
        net_return_column(config.horizon_minutes),
        f"path_mfe_{config.horizon_minutes}m",
        f"path_mae_{config.horizon_minutes}m",
    ):
        data[column] = np.nan
    data["path_outcome"] = pd.Series(pd.NA, index=data.index, dtype="string")


def _add_benchmark_label_returns(
    frame: pd.DataFrame,
    one_minute_bars: pd.DataFrame,
    config: IntradayDatasetConfig,
) -> pd.DataFrame:
    data = frame.copy()
    lookup = one_minute_bars.set_index(["ticker", "bar_start_utc"])
    entry_time = pd.to_datetime(data["entry_time_utc"], utc=True, errors="coerce")
    exit_start = pd.to_datetime(data["exit_time_utc"], utc=True, errors="coerce") - pd.Timedelta(minutes=1)
    for name, tickers in (
        ("spy", pd.Series(config.broad_benchmark.upper(), index=data.index)),
        ("qqq", pd.Series(config.growth_benchmark.upper(), index=data.index)),
        ("sector", data["primary_benchmark"].astype(str).str.upper()),
    ):
        entry_keys = pd.MultiIndex.from_arrays([tickers, entry_time])
        exit_keys = pd.MultiIndex.from_arrays([tickers, exit_start])
        entry = _lookup_values(lookup, entry_keys, "open")
        exit_price = _lookup_values(lookup, exit_keys, "close")
        returns = exit_price / entry - 1.0
        returns[(entry <= 0) | ~np.isfinite(entry) | ~np.isfinite(exit_price)] = np.nan
        data[f"path_{name}_return_{config.horizon_minutes}m"] = returns
        data[excess_return_column(config.horizon_minutes, name)] = data[net_return_column(config.horizon_minutes)] - returns
    return data


def _lookup_values(lookup: pd.DataFrame, keys: pd.MultiIndex, column: str) -> np.ndarray:
    values = lookup[column].reindex(keys)
    return np.asarray(pd.to_numeric(values, errors="coerce").to_numpy(float), dtype=float)
