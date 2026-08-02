"""Decision-time-causal intraday features built from completed volume bars."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import time
from typing import Final
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype

from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.v3.errors import DataReadinessError

FEATURE_SCHEMA_VERSION: Final = "edge_rebuild.intraday_features.v2"
EXCHANGE_TIMEZONE: Final = ZoneInfo("America/New_York")
_REGULAR_OPEN: Final = time(9, 30)
_REGULAR_CLOSE: Final = time(16, 0)
_HORIZONS: Final = (1, 5, 20)
CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS: Final = (
    "return_1_bar",
    "return_3_bars",
    "return_5_bars",
    "rsi_14",
    "atr_fraction_of_close",
    "ema_10_distance",
    "ema_20_distance",
    "realized_volatility_5",
    "realized_volatility_20",
    "granville_obv_confirmation",
    "kaufman_efficiency_ratio",
    "session_vwap_distance_atr",
    "volume_bar_progress",
    "normalized_volume_overshoot",
    "volume_bar_duration_minutes",
    "relative_volume_at_activation",
    "minutes_since_causal_activation",
    "regular_session_progress",
    "stock_clock_return_1m",
    "stock_clock_return_5m",
    "stock_clock_return_20m",
    "spy_return_1m",
    "spy_return_5m",
    "spy_return_20m",
    "qqq_return_1m",
    "qqq_return_5m",
    "qqq_return_20m",
    "sector_return_1m",
    "sector_return_5m",
    "sector_return_20m",
    "spy_relative_strength_1m",
    "spy_relative_strength_5m",
    "spy_relative_strength_20m",
    "qqq_relative_strength_1m",
    "qqq_relative_strength_5m",
    "qqq_relative_strength_20m",
    "sector_relative_strength_1m",
    "sector_relative_strength_5m",
    "sector_relative_strength_20m",
)
_SECTOR_BENCHMARKS: Final = frozenset({"XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"})
_VOLUME_BAR_REQUIRED: Final = frozenset(
    {
        "ticker",
        "session_date_et",
        "volume_bar_number",
        "bar_start_utc",
        "bar_end_utc",
        "available_at_utc",
        "first_source_minute_utc",
        "last_source_minute_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source_row_count",
        "volume_threshold",
        "volume_overshoot",
        "relative_volume_at_activation",
        "activation_time_utc",
        "model_eligible",
        "source",
        "price_feed",
        "adjustment",
        "source_timeframe",
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
_MEMBERSHIP_REQUIRED: Final = frozenset(
    {
        "ticker",
        "session_date_et",
        "session_open_utc",
        "session_close_utc",
        "security_id",
        "sector",
        "primary_benchmark",
        "universe_snapshot_id",
        "effective_from_utc",
        "effective_to_utc",
    }
)


def build_causal_intraday_features(
    volume_bars: pd.DataFrame,
    stock_one_minute_bars: pd.DataFrame,
    benchmark_one_minute_bars: pd.DataFrame,
    point_in_time_memberships: pd.DataFrame,
    *,
    contract: StrategyContract,
    strategy_contract_sha256: str,
) -> pd.DataFrame:
    """Build session-reset features from information known at each decision.

    Every market context is joined to the exact last source minute of the
    completed volume bar. Missing context remains NaN and makes the row
    ineligible; no previous, later, or cross-session row is substituted.
    """

    _validate_contract(contract, strategy_contract_sha256)
    bars = _validate_volume_bars(
        volume_bars,
        contract=contract,
        strategy_contract_sha256=strategy_contract_sha256,
    )
    stocks = _validate_minute_bars(stock_one_minute_bars, label="stock")
    benchmarks = _validate_minute_bars(benchmark_one_minute_bars, label="benchmark")
    memberships = _validate_memberships(point_in_time_memberships)

    data = bars.sort_values(["ticker", "session_date_et", "volume_bar_number"], kind="stable").reset_index(drop=True)
    data = _add_session_technical_features(data, contract)
    data["feature_context_minute_utc"] = data["last_source_minute_utc"]
    data = data.merge(
        memberships,
        on=["ticker", "session_date_et"],
        how="left",
        validate="many_to_one",
    )

    stock_context = _minute_context(stocks)
    benchmark_context = _minute_context(benchmarks)
    data = _attach_minute_context(
        data,
        stock_context,
        requested_ticker_column="ticker",
        prefix="stock_clock",
    )
    data = _attach_static_benchmark(data, benchmark_context, "SPY", "spy")
    data = _attach_static_benchmark(data, benchmark_context, "QQQ", "qqq")
    data = _attach_minute_context(
        data,
        benchmark_context,
        requested_ticker_column="primary_benchmark",
        prefix="sector",
    )
    _validate_stock_context_identity(data)

    context_availability_columns = [f"{prefix}_context_available_at_utc" for prefix in ("stock_clock", "spy", "qqq", "sector")]
    data["feature_available_at_utc"] = pd.concat(
        [data["available_at_utc"], *[data[column] for column in context_availability_columns]],
        axis=1,
    ).max(axis=1)

    for horizon in _HORIZONS:
        stock_return = f"stock_clock_return_{horizon}m"
        for benchmark in ("spy", "qqq", "sector"):
            data[f"{benchmark}_relative_strength_{horizon}m"] = (data[stock_return] - data[f"{benchmark}_return_{horizon}m"]).astype(
                "float32"
            )

    data["session_vwap_distance_atr"] = ((data["close"] - data["stock_clock_session_vwap"]) / data["atr_14"].replace(0.0, np.nan)).astype(
        "float32"
    )
    data["atr_fraction_of_close"] = (data["atr_14"] / data["close"]).astype("float32")
    data["normalized_volume_overshoot"] = (data["volume_overshoot"] / data["volume_threshold"]).astype("float32")
    data["volume_bar_duration_minutes"] = (
        (data["bar_end_utc"] - data["bar_start_utc"]) / pd.Timedelta(minutes=1)
    ).astype("float32")
    data["minutes_since_causal_activation"] = (
        (data["feature_available_at_utc"] - data["activation_time_utc"]) / pd.Timedelta(minutes=1)
    ).astype("float32")

    local_timestamp = data["feature_available_at_utc"].dt.tz_convert(EXCHANGE_TIMEZONE)
    regular_minutes = (
        local_timestamp.dt.hour * 60.0
        + local_timestamp.dt.minute
        + local_timestamp.dt.second / 60.0
        - (_REGULAR_OPEN.hour * 60 + _REGULAR_OPEN.minute)
    ).clip(lower=0.0, upper=390.0)
    data["regular_session_progress"] = (regular_minutes / 390.0).astype(
        "float32"
    )

    local_clock = local_timestamp.dt.strftime("%H:%M")
    data["session_segment"] = np.select(
        [
            local_clock.lt(contract.intraday.opening_end_et),
            local_clock.lt(contract.intraday.midday_end_et),
        ],
        ["opening", "midday"],
        default="late",
    )
    data["feature_schema_version"] = FEATURE_SCHEMA_VERSION
    data["feature_eligible"] = True
    data["feature_ineligible_reason"] = pd.Series(pd.NA, index=data.index, dtype="string")

    _reject(
        data,
        data["volume_bar_number"].lt(contract.intraday.minimum_warmup_bars),
        "insufficient_completed_volume_bars",
    )
    _reject(data, ~data["model_eligible"], "volume_bar_not_model_eligible")
    _reject(data, data["primary_benchmark"].isna(), "missing_point_in_time_membership")
    for prefix in ("stock_clock", "spy", "qqq", "sector"):
        _reject(
            data,
            data[f"{prefix}_context_available_at_utc"].isna(),
            f"missing_exact_{prefix}_minute_context",
        )

    finite_required = np.isfinite(
        data[list(CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS)].to_numpy(dtype="float64")
    ).all(axis=1)
    _reject(
        data,
        pd.Series(~finite_required, index=data.index),
        "incomplete_required_feature_history",
    )

    float_columns = [
        column
        for column in data.columns
        if column.startswith(
            (
                "return_",
                "rsi_",
                "atr_",
                "ema_",
                "realized_",
                "williams_",
                "granville_",
                "kaufman_",
            )
        )
        or column.endswith(
            (
                "_return_1m",
                "_return_5m",
                "_return_20m",
                "_relative_strength_1m",
                "_relative_strength_5m",
                "_relative_strength_20m",
            )
        )
        or column in {
            *CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS,
            "atr_14",
        }
    ]
    for column in float_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("float32")
    return data.sort_values(["session_date_et", "feature_available_at_utc", "ticker"], kind="stable").reset_index(drop=True)


def _add_session_technical_features(data: pd.DataFrame, contract: StrategyContract) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in data.groupby(["ticker", "session_date_et"], sort=False, observed=True):
        part = group.sort_values("volume_bar_number", kind="stable").copy()
        open_ = part["open"].to_numpy(dtype="float64")
        high = part["high"].to_numpy(dtype="float64")
        low = part["low"].to_numpy(dtype="float64")
        close = part["close"].to_numpy(dtype="float64")
        volume = part["volume"].to_numpy(dtype="float64")
        close_series = pd.Series(close, dtype="float64")

        for horizon in (1, 3, 5):
            suffix = "bar" if horizon == 1 else "bars"
            part[f"return_{horizon}_{suffix}"] = close_series.pct_change(periods=horizon, fill_method=None).to_numpy(dtype="float32")

        movement = close_series.diff()
        gains = movement.clip(lower=0.0)
        losses = -movement.clip(upper=0.0)
        average_gain = gains.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
        average_loss = losses.ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
        relative_strength = average_gain / average_loss.replace(0.0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + relative_strength))
        rsi = rsi.mask((average_loss == 0.0) & (average_gain > 0.0), 100.0)
        rsi = rsi.mask((average_loss == 0.0) & (average_gain == 0.0), 50.0)
        part["rsi_14"] = rsi.to_numpy(dtype="float32")

        previous_close = np.concatenate(([np.nan], close[:-1]))
        true_range = np.maximum.reduce(
            [
                high - low,
                np.abs(high - previous_close),
                np.abs(low - previous_close),
            ]
        )
        true_range[0] = high[0] - low[0]
        part["atr_14"] = (
            pd.Series(true_range, dtype="float64").ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean().to_numpy(dtype="float32")
        )
        ema_10 = close_series.ewm(span=10, adjust=False, min_periods=10).mean()
        ema_20 = close_series.ewm(span=20, adjust=False, min_periods=20).mean()
        part["ema_10_distance"] = (close_series / ema_10 - 1.0).to_numpy(dtype="float32")
        part["ema_20_distance"] = (close_series / ema_20 - 1.0).to_numpy(dtype="float32")

        interval_log_return = np.log(close_series / close_series.shift(1))
        interval_log_return.iloc[0] = np.log(close[0] / open_[0])
        squared = interval_log_return.pow(2.0)
        for horizon in (5, 20):
            part[f"realized_volatility_{horizon}"] = np.sqrt(squared.rolling(horizon, min_periods=horizon).sum()).to_numpy(dtype="float32")

        part["williams_five_bar_rsi_divergence"] = _williams_divergence(high, low, part["rsi_14"].to_numpy(dtype="float64"))
        obv, efficiency = _granville_and_kaufman(close, volume, lookback=20)
        part["granville_obv_confirmation"] = obv
        part["kaufman_efficiency_ratio"] = efficiency
        part["volume_bar_progress"] = (
            part["volume_bar_number"].to_numpy(dtype="float64") / float(contract.intraday.volume_bars_per_session_target)
        ).astype("float32")
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else data.copy()


def _williams_divergence(high: np.ndarray, low: np.ndarray, rsi: np.ndarray) -> np.ndarray:
    output = np.full(len(high), np.nan, dtype="float32")
    previous_high: tuple[float, float] | None = None
    previous_low: tuple[float, float] | None = None
    current = 0.0
    for observed_at in range(4, len(high)):
        candidate = observed_at - 2
        window = slice(candidate - 2, candidate + 3)
        if np.isfinite(rsi[candidate]):
            high_peers = np.delete(high[window], 2)
            if bool((high[candidate] > high_peers).all()):
                if previous_high is not None:
                    price_change = np.log(high[candidate] / previous_high[0])
                    oscillator_change = (rsi[candidate] - previous_high[1]) / 100.0
                    current = -max(price_change, 0.0) * max(-oscillator_change, 0.0)
                previous_high = (high[candidate], rsi[candidate])
            low_peers = np.delete(low[window], 2)
            if bool((low[candidate] < low_peers).all()):
                if previous_low is not None:
                    price_change = np.log(low[candidate] / previous_low[0])
                    oscillator_change = (rsi[candidate] - previous_low[1]) / 100.0
                    current = max(-price_change, 0.0) * max(oscillator_change, 0.0)
                previous_low = (low[candidate], rsi[candidate])
        output[observed_at] = np.float32(current)
    return output


def _granville_and_kaufman(
    close: np.ndarray,
    volume: np.ndarray,
    *,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    obv = np.full(len(close), np.nan, dtype="float32")
    efficiency = np.full(len(close), np.nan, dtype="float32")
    movement = np.concatenate(([0.0], np.diff(close)))
    signed_volume = np.sign(movement) * volume
    path_change = np.abs(movement)
    for index in range(lookback - 1, len(close)):
        start = index - lookback + 1
        observed_volume = float(volume[start : index + 1].sum())
        obv_change = float(signed_volume[start : index + 1].sum())
        direction = float(np.sign(close[index] - close[start]))
        if observed_volume > 0.0:
            obv[index] = np.float32(direction * obv_change / observed_volume)
        path = float(path_change[start + 1 : index + 1].sum())
        efficiency[index] = np.float32(abs(close[index] - close[start]) / path if path > 0.0 else 0.0)
    return obv, efficiency


def _minute_context(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby(["ticker", "session_date_et"], sort=False, observed=True):
        part = group.sort_values("bar_start_utc", kind="stable").copy()
        open_ = part["open"].to_numpy(dtype="float64")
        close = part["close"].to_numpy(dtype="float64")
        typical = (part["high"].to_numpy(dtype="float64") + part["low"].to_numpy(dtype="float64") + close) / 3.0
        volume = part["volume"].to_numpy(dtype="float64")
        cumulative_volume = np.cumsum(volume)
        part["session_vwap"] = (np.cumsum(typical * volume) / cumulative_volume).astype("float32")
        for horizon in _HORIZONS:
            values = np.full(len(part), np.nan, dtype="float32")
            if len(part) >= horizon:
                values[horizon - 1 :] = (close[horizon - 1 :] / open_[: len(part) - horizon + 1] - 1.0).astype("float32")
            part[f"return_{horizon}m"] = values
        parts.append(
            part[
                [
                    "ticker",
                    "session_date_et",
                    "bar_start_utc",
                    "available_at_utc",
                    "close",
                    "session_vwap",
                    *[f"return_{horizon}m" for horizon in _HORIZONS],
                ]
            ]
        )
    if not parts:
        raise DataReadinessError("one-minute context cannot be empty")
    return pd.concat(parts, ignore_index=True)


def _attach_static_benchmark(
    data: pd.DataFrame,
    context: pd.DataFrame,
    ticker: str,
    prefix: str,
) -> pd.DataFrame:
    result = data.copy()
    ticker_column = f"__{prefix}_ticker"
    result[ticker_column] = ticker
    return _attach_minute_context(
        result,
        context,
        requested_ticker_column=ticker_column,
        prefix=prefix,
    ).drop(columns=ticker_column)


def _attach_minute_context(
    data: pd.DataFrame,
    context: pd.DataFrame,
    *,
    requested_ticker_column: str,
    prefix: str,
) -> pd.DataFrame:
    result = data.copy()
    result["__context_ticker"] = result[requested_ticker_column].astype("string")
    renamed = context.rename(
        columns={
            "ticker": "__context_ticker",
            "bar_start_utc": "feature_context_minute_utc",
            "available_at_utc": f"{prefix}_context_available_at_utc",
            "close": f"{prefix}_context_close",
            "session_vwap": f"{prefix}_session_vwap",
            **{f"return_{horizon}m": f"{prefix}_return_{horizon}m" for horizon in _HORIZONS},
        }
    )
    result = result.merge(
        renamed,
        on=["__context_ticker", "session_date_et", "feature_context_minute_utc"],
        how="left",
        validate="many_to_one",
    )
    return result.drop(columns="__context_ticker")


def _validate_contract(contract: StrategyContract, strategy_contract_sha256: str) -> None:
    if not strategy_contract_sha256 or strategy_contract_sha256 != contract.sha256():
        raise DataReadinessError("intraday feature strategy contract hash does not match")
    if (
        contract.intraday.decision_bar_structure != "volume"
        or contract.intraday.volume_bar_source_timeframe != "1Min"
        or contract.intraday.volume_bars_per_session_target != 78
        or contract.intraday.minimum_warmup_bars != 20
        or contract.intraday.atr_timeframe != "5Min"
        or contract.intraday.atr_lookback_bars != 14
        or not contract.intraday.reset_rolling_features_overnight
    ):
        raise DataReadinessError("intraday feature contract is not the frozen causal design")


def _validate_volume_bars(
    frame: pd.DataFrame,
    *,
    contract: StrategyContract,
    strategy_contract_sha256: str,
) -> pd.DataFrame:
    _require_columns(frame, _VOLUME_BAR_REQUIRED, "volume bars")
    if frame.empty:
        raise DataReadinessError("volume bars are empty")
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["session_date_et"] = pd.to_datetime(data["session_date_et"], errors="raise").dt.date
    time_columns = (
        "bar_start_utc",
        "bar_end_utc",
        "available_at_utc",
        "first_source_minute_utc",
        "last_source_minute_utc",
        "activation_time_utc",
    )
    for column in time_columns:
        data[column] = pd.to_datetime(data[column], utc=True, errors="raise")
    numeric_columns = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source_row_count",
        "volume_threshold",
        "volume_overshoot",
        "relative_volume_at_activation",
        "volume_bar_number",
    )
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    numeric = data[list(numeric_columns)].to_numpy(dtype="float64")
    identity_ok = (
        data["source"].astype(str).str.lower().str.strip().eq("alpaca").all()
        and data["price_feed"].astype(str).str.lower().str.strip().eq("sip").all()
        and data["adjustment"].astype(str).str.lower().str.strip().eq("all").all()
        and data["source_timeframe"].astype(str).str.lower().str.strip().eq("1m").all()
        and data["strategy_contract_sha256"].astype(str).eq(strategy_contract_sha256).all()
    )
    if not bool(identity_ok):
        raise DataReadinessError("features require hash-bound Alpaca SIP/all 1m volume bars")
    if not is_bool_dtype(data["model_eligible"]):
        raise DataReadinessError("volume-bar model_eligible must be boolean")
    invalid_values = (
        bool(data["ticker"].eq("").any())
        or not bool(np.isfinite(numeric).all())
        or bool(data[["open", "high", "low", "close", "volume", "source_row_count", "volume_threshold"]].le(0.0).any().any())
        or bool(data["volume_overshoot"].lt(0.0).any())
        or bool(
            data["relative_volume_at_activation"]
            .lt(contract.intraday_universe.minimum_relative_volume)
            .any()
        )
        or bool(data["high"].lt(data[["open", "close"]].max(axis=1)).any())
        or bool(data["low"].gt(data[["open", "close"]].min(axis=1)).any())
        or bool(data["high"].lt(data["low"]).any())
        or bool(data.duplicated(["ticker", "session_date_et", "volume_bar_number"]).any())
        or bool((data["volume_bar_number"] % 1).ne(0).any())
        or bool((data["source_row_count"] % 1).ne(0).any())
        or not bool(np.allclose(data["volume"] - data["volume_threshold"], data["volume_overshoot"]))
    )
    if invalid_values:
        raise DataReadinessError("volume bars contain invalid completed-bar values or identity")

    for _, group in data.groupby(["ticker", "session_date_et"], sort=False, observed=True):
        ordered = group.sort_values("volume_bar_number", kind="stable")
        expected_numbers = np.arange(1, len(ordered) + 1)
        if not np.array_equal(ordered["volume_bar_number"].to_numpy(dtype="int64"), expected_numbers):
            raise DataReadinessError("volume-bar numbering must be contiguous and start at one per session")
        if bool(ordered["bar_start_utc"].iloc[1:].reset_index(drop=True).lt(ordered["bar_end_utc"].iloc[:-1].reset_index(drop=True)).any()):
            raise DataReadinessError("completed volume bars overlap within a stock-session")
        if ordered["relative_volume_at_activation"].nunique(dropna=False) != 1:
            raise DataReadinessError(
                "activation relative volume must be constant within a stock-session"
            )

    expected_eligible = data["volume_bar_number"].ge(contract.intraday.minimum_warmup_bars) & data["available_at_utc"].ge(
        data["activation_time_utc"]
    )
    local_start = data["bar_start_utc"].dt.tz_convert(EXCHANGE_TIMEZONE)
    local_end = data["bar_end_utc"].dt.tz_convert(EXCHANGE_TIMEZONE)
    local_activation = data["activation_time_utc"].dt.tz_convert(EXCHANGE_TIMEZONE)
    elapsed_source_minutes = (data["bar_end_utc"] - data["bar_start_utc"]) / pd.Timedelta(minutes=1)
    invalid_time = (
        bool(data["bar_start_utc"].ge(data["bar_end_utc"]).any())
        or bool(data["bar_end_utc"].gt(data["available_at_utc"]).any())
        or bool(data["first_source_minute_utc"].ne(data["bar_start_utc"]).any())
        or bool((data["last_source_minute_utc"] + pd.Timedelta(minutes=1)).ne(data["bar_end_utc"]).any())
        or bool(data["last_source_minute_utc"].gt(data["available_at_utc"]).any())
        or bool(local_start.dt.date.ne(data["session_date_et"]).any())
        or bool(local_end.dt.date.ne(data["session_date_et"]).any())
        or bool(local_activation.dt.date.ne(data["session_date_et"]).any())
        or bool(local_start.dt.time.lt(_REGULAR_OPEN).any())
        or bool(local_end.dt.time.gt(_REGULAR_CLOSE).any())
        or bool(data["source_row_count"].gt(elapsed_source_minutes).any())
        or any(bool(data[column].dt.second.ne(0).any()) or bool(data[column].dt.microsecond.ne(0).any()) for column in time_columns)
        or bool(data["model_eligible"].ne(expected_eligible).any())
    )
    if invalid_time:
        raise DataReadinessError("volume bars violate completion, session, availability, or warm-up lineage")
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
    numeric = data[["open", "high", "low", "close", "volume"]].to_numpy(dtype="float64")
    starts = data["bar_start_utc"]
    local_start = starts.dt.tz_convert(EXCHANGE_TIMEZONE)
    local_end = data["bar_end_utc"].dt.tz_convert(EXCHANGE_TIMEZONE)
    invalid = (
        not bool(identity_ok)
        or bool(data["ticker"].eq("").any())
        or not bool(np.isfinite(numeric).all())
        or bool(data[["open", "high", "low", "close", "volume"]].le(0.0).any().any())
        or bool(data["high"].lt(data[["open", "close"]].max(axis=1)).any())
        or bool(data["low"].gt(data[["open", "close"]].min(axis=1)).any())
        or bool(data["high"].lt(data["low"]).any())
        or bool(starts.dt.second.ne(0).any())
        or bool(starts.dt.microsecond.ne(0).any())
        or bool((data["bar_end_utc"] - starts).ne(pd.Timedelta(minutes=1)).any())
        or bool(data["available_at_utc"].lt(data["bar_end_utc"]).any())
        or bool(local_start.dt.date.ne(local_end.dt.date).any())
        or bool(local_start.dt.time.lt(_REGULAR_OPEN).any())
        or bool(local_end.dt.time.gt(_REGULAR_CLOSE).any())
        or bool(data.duplicated(["ticker", "bar_start_utc"]).any())
    )
    if invalid:
        raise DataReadinessError(f"{label} paths require unique, completed, regular-session Alpaca SIP/all 1m OHLCV bars")
    data["session_date_et"] = local_start.dt.date
    return data.sort_values(["ticker", "bar_start_utc"], kind="stable").reset_index(drop=True)


def _validate_memberships(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _MEMBERSHIP_REQUIRED, "point-in-time memberships")
    if frame.empty:
        raise DataReadinessError("point-in-time memberships are empty")
    data = frame.loc[:, sorted(_MEMBERSHIP_REQUIRED)].copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["primary_benchmark"] = data["primary_benchmark"].astype(str).str.upper().str.strip()
    data["session_date_et"] = pd.to_datetime(data["session_date_et"], errors="raise").dt.date
    for column in (
        "session_open_utc",
        "session_close_utc",
        "effective_from_utc",
        "effective_to_utc",
    ):
        data[column] = pd.to_datetime(data[column], utc=True, errors="raise")
    local_open = data["session_open_utc"].dt.tz_convert(EXCHANGE_TIMEZONE)
    identity_columns = [
        "ticker",
        "security_id",
        "sector",
        "primary_benchmark",
        "universe_snapshot_id",
    ]
    text_invalid = data[identity_columns].apply(lambda values: values.isna() | values.astype(str).str.strip().eq(""))
    active = data["effective_from_utc"].le(data["session_open_utc"]) & (
        data["effective_to_utc"].isna() | data["effective_to_utc"].gt(data["session_open_utc"])
    )
    invalid = (
        bool(text_invalid.any().any())
        or bool(data["session_open_utc"].isna().any())
        or bool(data["session_close_utc"].isna().any())
        or bool(data["effective_from_utc"].isna().any())
        or bool(data["session_open_utc"].ge(data["session_close_utc"]).any())
        or bool(local_open.dt.date.ne(data["session_date_et"]).any())
        or bool(data["primary_benchmark"].isin(_SECTOR_BENCHMARKS).eq(False).any())
        or bool(active.eq(False).any())
        or bool(data.duplicated(["ticker", "session_date_et"]).any())
    )
    if invalid:
        raise DataReadinessError("point-in-time memberships contain invalid, inactive, duplicate, or non-sector lineage")
    return data


def _validate_stock_context_identity(data: pd.DataFrame) -> None:
    matched = data["stock_clock_context_close"].notna()
    if matched.any() and not bool(
        np.isclose(
            data.loc[matched, "close"].to_numpy(dtype="float64"),
            data.loc[matched, "stock_clock_context_close"].to_numpy(dtype="float64"),
            rtol=1e-7,
            atol=1e-8,
        ).all()
    ):
        raise DataReadinessError("volume-bar close does not match its exact source one-minute close")


def _reject(data: pd.DataFrame, mask: pd.Series, reason: str) -> None:
    active = data["feature_eligible"] & mask.fillna(True).astype(bool)
    data.loc[active, "feature_eligible"] = False
    data.loc[active, "feature_ineligible_reason"] = reason


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise DataReadinessError(f"{label} are missing columns: {missing}")
