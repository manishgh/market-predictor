import numpy as np
import pandas as pd
from datetime import time
from zoneinfo import ZoneInfo
from market_predictor.edge_rebuild.strategy_contract import StrategyContract

EXCHANGE_TIMEZONE = ZoneInfo("America/New_York")
_REGULAR_OPEN = time(9, 30)
_REGULAR_CLOSE = time(16, 0)
_HORIZONS = (1, 5, 20)

class IntradayAdvancedIndicatorsStep:
    def __init__(self, contract: StrategyContract):
        self.contract = contract

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
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

            sma_10 = close_series.rolling(window=10, min_periods=10).mean()
            sma_20 = close_series.rolling(window=20, min_periods=20).mean()
            part["sma_10_distance"] = (close_series / sma_10 - 1.0).to_numpy(dtype="float32")
            part["sma_20_distance"] = (close_series / sma_20 - 1.0).to_numpy(dtype="float32")

            ema_12 = close_series.ewm(span=12, adjust=False, min_periods=12).mean()
            ema_26 = close_series.ewm(span=26, adjust=False, min_periods=20).mean()
            macd = (ema_12 - ema_26) / close_series
            macd_signal = macd.ewm(span=9, adjust=False, min_periods=1).mean()
            part["macd"] = macd.to_numpy(dtype="float32")
            part["macd_signal"] = macd_signal.to_numpy(dtype="float32")
            part["macd_hist"] = (macd - macd_signal).to_numpy(dtype="float32")

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
                part["volume_bar_number"].to_numpy(dtype="float64") / float(self.contract.intraday.volume_bars_per_session_target)
            ).astype("float32")
            parts.append(part)
        return pd.concat(parts, ignore_index=True) if parts else data.copy()


class IntradaySessionContextStep:
    def __init__(self, contract: StrategyContract, feature_schema_version: str):
        self.contract = contract
        self.feature_schema_version = feature_schema_version

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data
            
        data = data.copy()
        
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
                local_clock.lt(self.contract.intraday.opening_end_et),
                local_clock.lt(self.contract.intraday.midday_end_et),
            ],
            ["opening", "midday"],
            default="late",
        )
        data["feature_schema_version"] = self.feature_schema_version
        data["feature_eligible"] = True
        data["feature_ineligible_reason"] = pd.Series(pd.NA, index=data.index, dtype="string")
        return data


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
