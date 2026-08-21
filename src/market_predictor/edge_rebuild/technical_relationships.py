"""Causal relationships between price, RSI, volume, and market state.

Single-row indicator levels discard the path that produced them. RSI at 70 can
describe persistent momentum in a directional trend or an extended price in a
noisy range. A tree can discover a threshold on RSI itself; it cannot reconstruct
two earlier pivots or an entire trailing price-volume path from one row.

This module adds only those missing relationships:

* RSI divergence uses Bill Williams' five-bar pivot definition. The middle bar
  is a pivot only after two following bars complete, so the implementation emits
  no value before that confirmation time.
* Price-volume confirmation uses Joseph Granville's On-Balance Volume (OBV),
  normalized by observed volume over the same trailing window.
* Trend versus range state uses Perry Kaufman's Efficiency Ratio, the absolute
  net price change divided by the sum of absolute bar-to-bar changes.

All outputs are continuous. There are no overbought, oversold, buy, or sell
flags. Intraday callers include the exchange session in ``group_columns`` so no
rolling path can cross the overnight gap.
"""
from __future__ import annotations



from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.core.errors import DataReadinessError

_BASE_FEATURE_NAMES: Final = (
    "rsi_bearish_divergence_strength",
    "rsi_bearish_divergence_confirmation_age_bars",
    "rsi_bullish_divergence_strength",
    "rsi_bullish_divergence_confirmation_age_bars",
    "obv_directional_change_ratio",
    "price_obv_confirmation",
    "kaufman_efficiency_ratio",
    "rsi_trend_alignment",
    "rsi_range_position",
)
_ROW_ORDER = "_edge_relationship_original_row_order"


@dataclass(frozen=True, slots=True)
class TechnicalRelationshipSpec:
    """Columns and frozen structural lookbacks for one bar stream."""

    group_columns: tuple[str, ...]
    time_column: str
    pivot_span_bars: int
    obv_lookback_bars: int
    efficiency_lookback_bars: int
    high_column: str = "high"
    low_column: str = "low"
    close_column: str = "close"
    volume_column: str = "volume"
    rsi_column: str = "rsi_14"
    suffix: str = ""

    def __post_init__(self) -> None:
        if not self.group_columns or len(set(self.group_columns)) != len(
            self.group_columns
        ):
            raise ValueError("relationship groups must be non-empty and unique")
        if self.pivot_span_bars != 2:
            raise ValueError("RSI divergence requires a five-bar confirmed pivot")
        if self.obv_lookback_bars < 5:
            raise ValueError("OBV confirmation needs at least five bars")
        if self.efficiency_lookback_bars < 5:
            raise ValueError("efficiency ratio needs at least five bars")
        if self.suffix and (
            not self.suffix.startswith("_")
            or not self.suffix[1:].replace("_", "").isalnum()
        ):
            raise ValueError("relationship suffix must be empty or underscore-prefixed")


def relationship_spec_from_contract(
    contract: StrategyContract,
    *,
    group_columns: tuple[str, ...],
    time_column: str,
    rsi_column: str = "rsi_14",
    suffix: str = "",
) -> TechnicalRelationshipSpec:
    """Bind one bar stream to the immutable relationship feature contract."""

    features = contract.features
    return TechnicalRelationshipSpec(
        group_columns=group_columns,
        time_column=time_column,
        pivot_span_bars=features.rsi_pivot_span_bars,
        obv_lookback_bars=features.obv_confirmation_lookback_bars,
        efficiency_lookback_bars=features.efficiency_ratio_lookback_bars,
        rsi_column=rsi_column,
        suffix=suffix,
    )


def technical_relationship_feature_names(suffix: str = "") -> tuple[str, ...]:
    """Return the complete, stable output schema."""

    return tuple(f"{name}{suffix}" for name in _BASE_FEATURE_NAMES)


def add_technical_relationship_features(
    frame: pd.DataFrame,
    *,
    spec: TechnicalRelationshipSpec,
) -> pd.DataFrame:
    """Add causal relationship features and preserve the caller's row order."""

    required = {
        *spec.group_columns,
        spec.time_column,
        spec.high_column,
        spec.low_column,
        spec.close_column,
        spec.volume_column,
        spec.rsi_column,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataReadinessError(
            f"technical relationships lack required columns: {missing}"
        )
    if _ROW_ORDER in frame.columns:
        raise DataReadinessError(f"reserved relationship column already exists: {_ROW_ORDER}")
    names = technical_relationship_feature_names(spec.suffix)
    if any(name in frame.columns for name in names):
        collisions = sorted(name for name in names if name in frame.columns)
        raise DataReadinessError(
            f"technical relationship output columns already exist: {collisions}"
        )
    if frame.empty:
        output = frame.copy()
        for name in names:
            output[name] = pd.Series(index=output.index, dtype="float32")
        return output

    _validate_identity_and_time(frame, spec)
    data = frame.copy()
    data[_ROW_ORDER] = np.arange(len(data), dtype=np.int64)
    data = data.sort_values(
        [*spec.group_columns, spec.time_column],
        kind="stable",
        ignore_index=True,
    )
    numeric = _validated_numeric_input(data, spec)
    outputs = {
        name: np.full(len(data), np.nan, dtype=np.float32)
        for name in names
    }
    grouped = data.groupby(
        list(spec.group_columns),
        sort=False,
        observed=True,
        dropna=False,
    )
    for positions in grouped.indices.values():
        at = np.asarray(positions, dtype=np.int64)
        group_features = _group_relationships(numeric.iloc[at], spec)
        for name, values in zip(names, group_features, strict=True):
            outputs[name][at] = values
    for name in names:
        data[name] = outputs[name]
    return (
        data.sort_values(_ROW_ORDER, kind="stable")
        .drop(columns=_ROW_ORDER)
        .set_axis(frame.index)
    )


def _validate_identity_and_time(
    frame: pd.DataFrame,
    spec: TechnicalRelationshipSpec,
) -> None:
    identity = [*spec.group_columns, spec.time_column]
    if bool(frame.loc[:, identity].isna().any(axis=None)):
        raise DataReadinessError(
            "technical relationship identity and time columns cannot be missing"
        )
    if bool(frame.duplicated(identity).any()):
        raise DataReadinessError(
            "technical relationships require one bar per group and timestamp"
        )


def _validated_numeric_input(
    data: pd.DataFrame,
    spec: TechnicalRelationshipSpec,
) -> pd.DataFrame:
    columns = [
        spec.high_column,
        spec.low_column,
        spec.close_column,
        spec.volume_column,
        spec.rsi_column,
    ]
    numeric = data.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    prices = numeric[
        [spec.high_column, spec.low_column, spec.close_column]
    ]
    invalid_price = (
        prices.isna().any(axis=1)
        | prices.le(0).any(axis=1)
        | numeric[spec.high_column].lt(numeric[spec.low_column])
        | numeric[spec.high_column].lt(numeric[spec.close_column])
        | numeric[spec.low_column].gt(numeric[spec.close_column])
    )
    volume = numeric[spec.volume_column]
    invalid_volume = volume.isna() | volume.lt(0)
    rsi = numeric[spec.rsi_column]
    invalid_rsi = rsi.notna() & ~rsi.between(0.0, 100.0)
    if bool((invalid_price | invalid_volume | invalid_rsi).any()):
        raise DataReadinessError(
            "technical relationships contain invalid price, volume, or RSI values"
        )
    return numeric


def _group_relationships(
    numeric: pd.DataFrame,
    spec: TechnicalRelationshipSpec,
) -> tuple[np.ndarray, ...]:
    high = numeric[spec.high_column].to_numpy(dtype=np.float64)
    low = numeric[spec.low_column].to_numpy(dtype=np.float64)
    close = pd.Series(
        numeric[spec.close_column].to_numpy(dtype=np.float64),
        dtype="float64",
    )
    volume = pd.Series(
        numeric[spec.volume_column].to_numpy(dtype=np.float64),
        dtype="float64",
    )
    rsi = pd.Series(
        numeric[spec.rsi_column].to_numpy(dtype=np.float64),
        dtype="float64",
    )

    bearish, bearish_age, bullish, bullish_age = _confirmed_rsi_divergence(
        high,
        low,
        rsi.to_numpy(dtype=np.float64),
        span=spec.pivot_span_bars,
    )

    price_direction = np.sign(
        np.log(close / close.shift(spec.obv_lookback_bars))
    )
    signed_volume = np.sign(close.diff()) * volume
    obv_change = signed_volume.rolling(
        spec.obv_lookback_bars,
        min_periods=spec.obv_lookback_bars,
    ).sum()
    observed_volume = volume.rolling(
        spec.obv_lookback_bars,
        min_periods=spec.obv_lookback_bars,
    ).sum()
    obv_ratio = obv_change / observed_volume.replace(0.0, np.nan)
    price_obv_confirmation = price_direction * obv_ratio

    efficiency_change = close.diff(spec.efficiency_lookback_bars)
    efficiency_path = close.diff().abs().rolling(
        spec.efficiency_lookback_bars,
        min_periods=spec.efficiency_lookback_bars,
    ).sum()
    efficiency = (
        efficiency_change.abs() / efficiency_path.replace(0.0, np.nan)
    ).clip(0.0, 1.0)
    centered_rsi = (rsi - 50.0) / 50.0
    trend_alignment = (
        centered_rsi * np.sign(efficiency_change) * efficiency
    )
    range_position = centered_rsi * (1.0 - efficiency)

    return (
        bearish,
        bearish_age,
        bullish,
        bullish_age,
        obv_ratio.to_numpy(dtype=np.float32),
        price_obv_confirmation.to_numpy(dtype=np.float32),
        efficiency.to_numpy(dtype=np.float32),
        trend_alignment.to_numpy(dtype=np.float32),
        range_position.to_numpy(dtype=np.float32),
    )


def _confirmed_rsi_divergence(
    high: np.ndarray,
    low: np.ndarray,
    rsi: np.ndarray,
    *,
    span: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compare the latest two pivots only when the newer pivot is confirmed."""

    size = len(high)
    bearish = np.full(size, np.nan, dtype=np.float32)
    bearish_age = np.full(size, np.nan, dtype=np.float32)
    bullish = np.full(size, np.nan, dtype=np.float32)
    bullish_age = np.full(size, np.nan, dtype=np.float32)
    previous_high: tuple[float, float] | None = None
    previous_low: tuple[float, float] | None = None
    current_bearish = np.nan
    current_bullish = np.nan
    current_bearish_age = np.nan
    current_bullish_age = np.nan

    for observed_at in range(size):
        if np.isfinite(current_bearish_age):
            current_bearish_age += 1.0
        if np.isfinite(current_bullish_age):
            current_bullish_age += 1.0
        candidate = observed_at - span
        if candidate >= span:
            start = candidate - span
            stop = candidate + span + 1
            candidate_rsi = rsi[candidate]
            if np.isfinite(candidate_rsi) and _strict_high_pivot(
                high[start:stop], span
            ):
                if previous_high is not None:
                    price_change = np.log(high[candidate] / previous_high[0])
                    oscillator_change = (candidate_rsi - previous_high[1]) / 100.0
                    current_bearish = max(price_change, 0.0) * max(
                        -oscillator_change, 0.0
                    )
                    current_bearish_age = 0.0
                previous_high = (high[candidate], candidate_rsi)
            if np.isfinite(candidate_rsi) and _strict_low_pivot(
                low[start:stop], span
            ):
                if previous_low is not None:
                    price_change = np.log(low[candidate] / previous_low[0])
                    oscillator_change = (candidate_rsi - previous_low[1]) / 100.0
                    current_bullish = max(-price_change, 0.0) * max(
                        oscillator_change, 0.0
                    )
                    current_bullish_age = 0.0
                previous_low = (low[candidate], candidate_rsi)
        bearish[observed_at] = current_bearish
        bearish_age[observed_at] = current_bearish_age
        bullish[observed_at] = current_bullish
        bullish_age[observed_at] = current_bullish_age
    return bearish, bearish_age, bullish, bullish_age


def _strict_high_pivot(window: np.ndarray, center: int) -> bool:
    if not bool(np.isfinite(window).all()):
        return False
    candidate = window[center]
    peers = np.delete(window, center)
    return bool((candidate > peers).all())


def _strict_low_pivot(window: np.ndarray, center: int) -> bool:
    if not bool(np.isfinite(window).all()):
        return False
    candidate = window[center]
    peers = np.delete(window, center)
    return bool((candidate < peers).all())
