from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from market_predictor.v3.contracts import SourceAvailability
from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError

V3_FEATURE_SCHEMA_VERSION = "ml_v3.features.v1"
RETURN_WINDOWS = (1, 3, 6, 12)
SOURCE_FAMILIES = ("alpaca", "seeking_alpha", "sec", "finviz", "reddit", "global_context")
CROSS_SECTIONAL_BASE_FEATURES = (
    "return_1bar",
    "return_3bar",
    "return_6bar",
    "relative_volume_same_minute_20d",
    "dollar_volume",
    "atr_pct",
    "dist_session_vwap",
    "dist_opening_range_high",
    "rel_return_3bar_vs_qqq",
    "rel_return_3bar_vs_sector",
    "overnight_gap",
)
V3_MICROSTRUCTURE_FEATURES = (
    "microstructure_available",
    "spread_pct",
    "quote_imbalance",
    "average_trade_size",
)


def build_v3_features(
    bars: pd.DataFrame,
    benchmarks: pd.DataFrame,
    *,
    source_availability: pd.DataFrame | None = None,
    minimum_cross_section: int = 3,
) -> pd.DataFrame:
    """Build the same completed-bar feature schema for batch and live callers."""
    if minimum_cross_section < 2:
        raise ValueError("minimum_cross_section must be at least 2")
    technical = build_v3_ticker_features(bars, source_availability=source_availability)
    return finalize_v3_cross_sectional_features(
        technical,
        benchmarks,
        minimum_cross_section=minimum_cross_section,
    )


def build_v3_ticker_features(
    bars: pd.DataFrame,
    *,
    source_availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build ticker-local state so large universes can be processed in bounded shards."""
    data = _prepare_bars(bars, require_context=True)
    data = _add_technical_features(data)
    data = _add_source_availability(data, source_availability)
    data = _add_microstructure_features(data)
    return data.sort_values(["decision_time_utc", "ticker"]).reset_index(drop=True)


def finalize_v3_cross_sectional_features(
    technical_features: pd.DataFrame,
    benchmarks: pd.DataFrame,
    *,
    minimum_cross_section: int = 3,
) -> pd.DataFrame:
    """Add exact benchmark, breadth, regime, and point-in-time cross-sectional features."""
    if minimum_cross_section < 2:
        raise ValueError("minimum_cross_section must be at least 2")
    required = {
        "ticker",
        "timestamp",
        "decision_time_utc",
        "decision_group_id",
        "primary_benchmark",
        "universe_snapshot_id",
        "price_feed",
        "_session_date_et",
    }
    missing = sorted(required.difference(technical_features.columns))
    if missing:
        raise SchemaMismatchError(f"technical feature shards missing columns: {', '.join(missing)}")
    data = technical_features.copy()
    benchmark_data = _prepare_bars(benchmarks, require_context=False)
    data = _add_benchmark_features(data, benchmark_data)
    data = _add_market_and_breadth_features(data)
    data = _add_cross_sectional_features(data, minimum_group=minimum_cross_section)
    data["feature_schema_version"] = V3_FEATURE_SCHEMA_VERSION
    return data.sort_values(["decision_time_utc", "ticker"]).reset_index(drop=True)


def core_feature_columns() -> tuple[str, ...]:
    technical = [
        *(f"return_{window}bar" for window in RETURN_WINDOWS),
        *(f"volatility_{window}bar" for window in (3, 6, 12)),
        "ema_10",
        "ema_20",
        "ema_50",
        "ema_10_slope_3bar",
        "ema_20_slope_3bar",
        "macd",
        "macd_signal",
        "macd_histogram",
        "rsi_14",
        "atr_14",
        "atr_pct",
        "session_vwap",
        "dist_session_vwap",
        "session_vwap_slope_3bar",
        "opening_range_width_pct",
        "dist_opening_range_high",
        "dist_opening_range_low",
        "premarket_range_pct",
        "overnight_gap",
        "relative_volume_same_minute_20d",
        "volume_burst_20bar",
        "dollar_volume",
        "dist_recent_20bar_high_atr",
        "dist_recent_20bar_low_atr",
        "session_progress",
        "session_minute_sin",
        "session_minute_cos",
    ]
    benchmark = [
        *(f"qqq_return_{window}bar" for window in RETURN_WINDOWS),
        *(f"spy_return_{window}bar" for window in RETURN_WINDOWS),
        *(f"sector_return_{window}bar" for window in RETURN_WINDOWS),
        *(f"rel_return_{window}bar_vs_qqq" for window in RETURN_WINDOWS),
        *(f"rel_return_{window}bar_vs_sector" for window in RETURN_WINDOWS),
        "eligible_breadth_positive_1bar",
        "eligible_breadth_above_vwap",
        "regime_risk_on",
        "regime_risk_off",
        "regime_high_volatility",
    ]
    sources = ["source_availability_declared"]
    for family in SOURCE_FAMILIES:
        sources.extend((f"source_available_{family}", f"source_rows_{family}"))
    cross_sectional = [
        name
        for feature in CROSS_SECTIONAL_BASE_FEATURES
        for name in (f"xs_rank_{feature}", f"xs_robust_z_{feature}")
    ]
    return tuple(dict.fromkeys([*technical, *benchmark, *sources, "cross_section_eligible", *cross_sectional]))


def _prepare_bars(frame: pd.DataFrame, *, require_context: bool) -> pd.DataFrame:
    required = {"ticker", "timestamp", "open", "high", "low", "close", "volume"}
    if require_context:
        required.update({"primary_benchmark", "universe_snapshot_id", "price_feed"})
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SchemaMismatchError(f"feature bars missing columns: {', '.join(missing)}")
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    if isinstance(data["timestamp"].dtype, pd.DatetimeTZDtype):
        data["timestamp"] = data["timestamp"].dt.tz_convert("UTC")
    else:
        data["timestamp"] = data["timestamp"].map(_aware_timestamp)
    if bool(data["timestamp"].isna().any()):
        raise DataReadinessError("feature bars contain invalid or timezone-naive timestamps")
    numeric = ["open", "high", "low", "close", "volume"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    if bool(data[numeric].isna().any(axis=None)):
        raise DataReadinessError("feature bars contain non-numeric OHLCV")
    if bool(data.duplicated(["ticker", "timestamp"]).any()):
        raise DataReadinessError("feature bars contain duplicate ticker/timestamp rows")
    data["_session_date_et"] = data["timestamp"].dt.tz_convert("America/New_York").dt.date
    if require_context:
        data["decision_time_utc"] = data["timestamp"]
        data["feature_available_at_utc"] = data["timestamp"]
        data["session_date_et"] = data["_session_date_et"]
        if "decision_group_id" not in data.columns:
            data["decision_group_id"] = data["timestamp"].map(lambda value: value.isoformat())
        group_times = data.groupby("decision_group_id")["timestamp"].nunique()
        if bool(group_times.gt(1).any()):
            raise DataReadinessError("decision_group_id spans multiple timestamps")
        if bool(data.duplicated(["decision_group_id", "ticker"]).any()):
            raise DataReadinessError("decision group contains duplicate tickers")
    return data.sort_values(["ticker", "timestamp"]).reset_index(drop=True)


def _add_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    session_keys = [data["ticker"], data["_session_date_et"]]
    session_group = data.groupby(["ticker", "_session_date_et"], sort=False)
    ticker_group = data.groupby("ticker", sort=False)
    close = data["close"].astype(float)
    volume = data["volume"].astype(float)
    one_bar_return = session_group["close"].pct_change()
    for window in RETURN_WINDOWS:
        data[f"return_{window}bar"] = session_group["close"].pct_change(window)
    for window in (3, 6, 12):
        data[f"volatility_{window}bar"] = one_bar_return.groupby(session_keys, sort=False).transform(
            lambda values, size=window: values.rolling(size, min_periods=max(2, size // 2)).std()
        )
    for span in (10, 20, 50):
        data[f"ema_{span}"] = ticker_group["close"].transform(
            lambda values, size=span: values.ewm(span=size, adjust=False, min_periods=max(3, size // 2)).mean()
        )
    data["ema_10_slope_3bar"] = ticker_group["ema_10"].pct_change(3)
    data["ema_20_slope_3bar"] = ticker_group["ema_20"].pct_change(3)
    ema_12 = ticker_group["close"].transform(lambda values: values.ewm(span=12, adjust=False, min_periods=6).mean())
    ema_26 = ticker_group["close"].transform(lambda values: values.ewm(span=26, adjust=False, min_periods=13).mean())
    data["macd"] = ema_12 - ema_26
    data["macd_signal"] = data.groupby("ticker", sort=False)["macd"].transform(
        lambda values: values.ewm(span=9, adjust=False, min_periods=5).mean()
    )
    data["macd_histogram"] = data["macd"] - data["macd_signal"]
    delta = ticker_group["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.groupby(data["ticker"], sort=False).transform(lambda values: values.ewm(alpha=1 / 14, adjust=False).mean())
    average_loss = loss.groupby(data["ticker"], sort=False).transform(lambda values: values.ewm(alpha=1 / 14, adjust=False).mean())
    relative_strength = average_gain / average_loss.replace(0, np.nan)
    data["rsi_14"] = 100.0 - 100.0 / (1.0 + relative_strength)
    data.loc[average_loss.eq(0) & average_gain.gt(0), "rsi_14"] = 100.0
    data.loc[average_loss.eq(0) & average_gain.eq(0), "rsi_14"] = 50.0
    prior_close = ticker_group["close"].shift(1)
    true_range = pd.concat(
        [(data["high"] - data["low"]), (data["high"] - prior_close).abs(), (data["low"] - prior_close).abs()],
        axis=1,
    ).max(axis=1)
    data["atr_14"] = true_range.groupby(data["ticker"], sort=False).transform(
        lambda values: values.ewm(alpha=1 / 14, adjust=False).mean()
    )
    data["atr_pct"] = data["atr_14"] / close.replace(0, np.nan)
    _add_session_state(data)
    prior_volume_median = ticker_group["volume"].transform(lambda values: values.shift(1).rolling(20, min_periods=5).median())
    data["volume_burst_20bar"] = volume / prior_volume_median.replace(0, np.nan)
    minute_slot = _eastern_minute(data["timestamp"])
    same_minute_baseline = volume.groupby([data["ticker"], minute_slot], sort=False).transform(
        lambda values: values.shift(1).rolling(20, min_periods=3).median()
    )
    data["relative_volume_same_minute_20d"] = volume / same_minute_baseline.replace(0, np.nan)
    data["dollar_volume"] = close * volume
    recent_high = ticker_group["high"].transform(lambda values: values.rolling(20, min_periods=5).max())
    recent_low = ticker_group["low"].transform(lambda values: values.rolling(20, min_periods=5).min())
    data["dist_recent_20bar_high_atr"] = (close - recent_high) / data["atr_14"].replace(0, np.nan)
    data["dist_recent_20bar_low_atr"] = (close - recent_low) / data["atr_14"].replace(0, np.nan)
    return data


def _add_session_state(data: pd.DataFrame) -> None:
    session_group = data.groupby(["ticker", "_session_date_et"], sort=False)
    minute = _eastern_minute(data["timestamp"])
    from_open = minute - (9 * 60 + 30)
    data["session_progress"] = (from_open / 390).clip(0, 1)
    radians = 2 * np.pi * data["session_progress"]
    data["session_minute_sin"] = np.sin(radians)
    data["session_minute_cos"] = np.cos(radians)
    typical = data[["high", "low", "close"]].mean(axis=1)
    cumulative_dollar = (typical * data["volume"]).groupby([data["ticker"], data["_session_date_et"]], sort=False).cumsum()
    cumulative_volume = data["volume"].groupby([data["ticker"], data["_session_date_et"]], sort=False).cumsum()
    data["session_vwap"] = cumulative_dollar / cumulative_volume.replace(0, np.nan)
    data["dist_session_vwap"] = data["close"] / data["session_vwap"] - 1.0
    data["session_vwap_slope_3bar"] = session_group["session_vwap"].pct_change(3)
    opening = from_open.between(0, 30)
    opening_high = data["high"].where(opening).groupby([data["ticker"], data["_session_date_et"]], sort=False).cummax()
    opening_low = data["low"].where(opening).groupby([data["ticker"], data["_session_date_et"]], sort=False).cummin()
    data["opening_range_high"] = opening_high.groupby([data["ticker"], data["_session_date_et"]], sort=False).ffill()
    data["opening_range_low"] = opening_low.groupby([data["ticker"], data["_session_date_et"]], sort=False).ffill()
    data["opening_range_width_pct"] = (data["opening_range_high"] - data["opening_range_low"]) / data["close"]
    data["dist_opening_range_high"] = data["close"] / data["opening_range_high"] - 1.0
    data["dist_opening_range_low"] = data["close"] / data["opening_range_low"] - 1.0
    premarket = minute.between(4 * 60, 9 * 60 + 29)
    premarket_high = data["high"].where(premarket).groupby([data["ticker"], data["_session_date_et"]], sort=False).cummax()
    premarket_low = data["low"].where(premarket).groupby([data["ticker"], data["_session_date_et"]], sort=False).cummin()
    premarket_high = premarket_high.groupby([data["ticker"], data["_session_date_et"]], sort=False).ffill()
    premarket_low = premarket_low.groupby([data["ticker"], data["_session_date_et"]], sort=False).ffill()
    data["premarket_range_pct"] = (premarket_high - premarket_low) / data["close"]
    data["overnight_gap"] = _overnight_gap(data, regular_mask=from_open.ge(0))


def _overnight_gap(data: pd.DataFrame, *, regular_mask: pd.Series) -> pd.Series:
    regular = data.loc[regular_mask, ["ticker", "_session_date_et", "open", "close"]]
    sessions = regular.groupby(["ticker", "_session_date_et"], sort=False).agg(
        session_open=("open", "first"),
        session_close=("close", "last"),
    )
    sessions["prior_session_close"] = sessions.groupby(level="ticker")["session_close"].shift(1)
    sessions["gap"] = sessions["session_open"] / sessions["prior_session_close"] - 1.0
    keys = pd.MultiIndex.from_arrays([data["ticker"], data["_session_date_et"]])
    mapped = pd.Series(sessions["gap"].reindex(keys).to_numpy(), index=data.index, dtype=float)
    return mapped.where(regular_mask)


def _add_benchmark_features(data: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    benchmark = _add_benchmark_returns(benchmarks)
    output = data.copy()
    for symbol in ("QQQ", "SPY"):
        selected_columns = ["timestamp", *_return_columns()]
        if symbol == "QQQ":
            selected_columns.append("volatility_12bar")
        selected = benchmark[benchmark["ticker"] == symbol][selected_columns].copy()
        selected[f"_{symbol.lower()}_present"] = 1
        selected = selected.rename(columns={column: f"{symbol.lower()}_{column}" for column in _return_columns()})
        if symbol == "QQQ":
            selected = selected.rename(columns={"volatility_12bar": "_qqq_volatility_12bar"})
        output = output.merge(selected, on="timestamp", how="left", validate="many_to_one")
        if bool(output[f"_{symbol.lower()}_present"].isna().any()):
            raise DataReadinessError(f"missing exact {symbol} benchmark timestamps")
        output = output.drop(columns=f"_{symbol.lower()}_present")
    sector = benchmark[["ticker", "timestamp", *_return_columns()]].rename(
        columns={"ticker": "primary_benchmark", **{column: f"sector_{column}" for column in _return_columns()}}
    )
    sector["_sector_present"] = 1
    output["primary_benchmark"] = output["primary_benchmark"].astype(str).str.upper().str.strip()
    output = output.merge(sector, on=["primary_benchmark", "timestamp"], how="left", validate="many_to_one")
    if bool(output["_sector_present"].isna().any()):
        raise DataReadinessError("missing exact sector benchmark timestamps")
    output = output.drop(columns="_sector_present")
    for window in RETURN_WINDOWS:
        output[f"rel_return_{window}bar_vs_qqq"] = output[f"return_{window}bar"] - output[f"qqq_return_{window}bar"]
        output[f"rel_return_{window}bar_vs_sector"] = output[f"return_{window}bar"] - output[f"sector_return_{window}bar"]
    return output


def _add_benchmark_returns(benchmarks: pd.DataFrame) -> pd.DataFrame:
    output = benchmarks.copy()
    group = output.groupby(["ticker", "_session_date_et"], sort=False)
    for window in RETURN_WINDOWS:
        output[f"return_{window}bar"] = group["close"].pct_change(window)
    one_bar = group["close"].pct_change()
    output["volatility_12bar"] = one_bar.groupby([output["ticker"], output["_session_date_et"]], sort=False).transform(
        lambda values: values.rolling(12, min_periods=4).std()
    )
    return output


def _add_market_and_breadth_features(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    group = output.groupby("decision_group_id", sort=False)
    output["eligible_breadth_positive_1bar"] = group["return_1bar"].transform(lambda values: values.gt(0).mean())
    output["eligible_breadth_above_vwap"] = group["dist_session_vwap"].transform(lambda values: values.gt(0).mean())
    qqq_momentum = output["qqq_return_6bar"]
    qqq_volatility = output["_qqq_volatility_12bar"]
    output["regime_risk_on"] = (qqq_momentum > 0.002).astype(int)
    output["regime_risk_off"] = (qqq_momentum < -0.002).astype(int)
    output["regime_high_volatility"] = (qqq_volatility > 0.005).astype(int)
    return output.drop(columns="_qqq_volatility_12bar")


def _add_source_availability(data: pd.DataFrame, availability: pd.DataFrame | None) -> pd.DataFrame:
    output = data.copy()
    output["source_availability_declared"] = int(availability is not None)
    for family in SOURCE_FAMILIES:
        output[f"source_available_{family}"] = 0
        output[f"source_rows_{family}"] = 0
    if availability is None or availability.empty:
        return output
    records: list[SourceAvailability] = []
    for record in availability.to_dict(orient="records"):
        if pd.isna(record.get("first_available_at_utc")):
            record["first_available_at_utc"] = None
        if pd.isna(record.get("last_available_at_utc")):
            record["last_available_at_utc"] = None
        records.append(SourceAvailability.model_validate(record))
    for family in SOURCE_FAMILIES:
        family_records = [record for record in records if record.source_family.strip().lower() == family]
        if not family_records:
            continue
        for ticker, indices in output.groupby("ticker", sort=False).groups.items():
            ticker_records = sorted(
                (record for record in family_records if record.ticker == ticker),
                key=lambda record: record.collected_at_utc,
            )
            if not ticker_records:
                continue
            record_times = np.array([record.collected_at_utc.timestamp() for record in ticker_records])
            decision_times = output.loc[indices, "decision_time_utc"].map(lambda value: value.timestamp()).to_numpy()
            positions = np.searchsorted(record_times, decision_times, side="right") - 1
            for row_index, position in zip(indices, positions, strict=True):
                if position < 0:
                    continue
                record = ticker_records[int(position)]
                output.at[row_index, f"source_available_{family}"] = int(record.available)
                output.at[row_index, f"source_rows_{family}"] = record.row_count
    return output


def _add_microstructure_features(data: pd.DataFrame) -> pd.DataFrame:
    output = data.copy()
    required = {"bid", "ask", "bid_size", "ask_size", "trade_count"}
    if not required.issubset(output.columns):
        output["microstructure_available"] = 0
        output["spread_pct"] = np.nan
        output["quote_imbalance"] = np.nan
        output["average_trade_size"] = np.nan
        return output
    bid = pd.to_numeric(output["bid"], errors="coerce")
    ask = pd.to_numeric(output["ask"], errors="coerce")
    bid_size = pd.to_numeric(output["bid_size"], errors="coerce")
    ask_size = pd.to_numeric(output["ask_size"], errors="coerce")
    trade_count = pd.to_numeric(output["trade_count"], errors="coerce")
    valid = bid.gt(0) & ask.ge(bid) & bid_size.ge(0) & ask_size.ge(0) & trade_count.gt(0)
    midpoint = (bid + ask) / 2
    output["microstructure_available"] = valid.astype(int)
    output["spread_pct"] = ((ask - bid) / midpoint.replace(0, np.nan)).where(valid)
    output["quote_imbalance"] = ((bid_size - ask_size) / (bid_size + ask_size).replace(0, np.nan)).where(valid)
    output["average_trade_size"] = (output["volume"] / trade_count).where(valid)
    return output


def _add_cross_sectional_features(data: pd.DataFrame, *, minimum_group: int) -> pd.DataFrame:
    output = data.copy()
    group_sizes = output.groupby("decision_group_id")["ticker"].transform("size")
    output["cross_section_eligible"] = group_sizes.ge(minimum_group).astype(int)
    for feature in CROSS_SECTIONAL_BASE_FEATURES:
        values = pd.to_numeric(output[feature], errors="coerce")
        grouped = values.groupby(output["decision_group_id"], sort=False)
        count = grouped.transform("count")
        rank = grouped.rank(method="average")
        output[f"xs_rank_{feature}"] = ((rank - 1) / (count - 1).replace(0, np.nan)).where(count.ge(minimum_group))
        median = grouped.transform("median")
        q75 = grouped.transform("quantile", q=0.75)
        q25 = grouped.transform("quantile", q=0.25)
        robust_scale = ((q75 - q25) / 1.349).replace(0, np.nan)
        robust_z = (values - median) / robust_scale
        output[f"xs_robust_z_{feature}"] = robust_z.replace([np.inf, -np.inf], np.nan).where(count.ge(minimum_group))
    return output


def _return_columns() -> Iterable[str]:
    return (f"return_{window}bar" for window in RETURN_WINDOWS)


def _eastern_minute(timestamp: pd.Series) -> pd.Series:
    eastern = timestamp.dt.tz_convert("America/New_York")
    return eastern.dt.hour * 60 + eastern.dt.minute


def _aware_timestamp(value: object) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        return pd.NaT
    return timestamp.tz_convert("UTC")
