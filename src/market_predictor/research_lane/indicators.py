"""Additional textbook technical indicators and the cross-sectional feature sets built from them.

Every indicator value at session t uses adjusted open/high/low/close/volume through the
close of t only. Rolling statistics require a complete window (no partial windows, no
fills). Recursive smoothers (EMA, Wilder) start only after a full initial window, seeded
with that window's simple mean, depend only on earlier rows, and restart after a missing
value once a new full window has accumulated. Features are centered cross-sectional ranks
among eligible stocks on the date; no indicator carries an expected sign.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from market_predictor.research_lane.panel import ResearchPanel
from market_predictor.research_lane.predictors import (
    PREDICTOR_NAMES,
    centered_rank,
    compute_predictors,
    daily_returns,
    sector_mean,
)

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]

INDICATOR_NAMES: Final = (
    "rsi_14", "rsi_5", "stochastic_k_14", "cci_20", "macd_hist_pct", "bollinger_pct_b_20", "bollinger_bandwidth_20",
    "dist_sma_20", "dist_sma_50", "dist_sma_200", "sma_50_200_ratio", "adx_14", "di_diff_14", "aroon_osc_25",
    "donchian_pos_55", "atr_pct_14", "keltner_pos_20", "mfi_14", "cmf_20", "obv_change_20", "up_down_volume_ratio_20",
    "volume_z_20", "dollar_volume_log_20", "overnight_return_21", "intraday_return_21", "gap_today", "intraday_today",
    "close_location_today", "range_pct_5", "realized_vol_10", "realized_vol_60", "vol_ratio_10_60", "skew_60",
    "kurt_60", "downside_vol_60", "days_since_252_high", "up_days_fraction_21", "streak_length", "return_1d",
    "return_3d", "return_10d", "return_63d", "return_126d", "return_252d", "efficiency_ratio_20",
    "sector_relative_return_21", "sector_relative_return_5", "residual_return_21", "price_volume_corr_20",
    "range_position_252", "momentum_change_21",
)
SECTOR_SUFFIX: Final = "__sector"
SECTOR_MINIMUM_STOCKS: Final = 5
STREAK_CAP: Final = 10
FEATURE_SET_NAMES: Final = ("set_12", "set_all", "set_all_sector_neutral")
_WINDOW_CHUNK_ROWS: Final = 64

Statistic = Literal["mean", "max", "min", "std0", "std1", "skew", "kurt"]


def _lag(values: FloatArray, sessions: int) -> FloatArray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if sessions < values.shape[0]:
        out[sessions:] = values[:values.shape[0] - sessions]
    return out


def _rolling(values: FloatArray, window: int, statistic: Statistic) -> FloatArray:
    """Rolling statistic over rows t-window+1..t; NaN unless all ``window`` cells are finite."""
    rolling = pd.DataFrame(values).rolling(window, min_periods=window)
    if statistic == "mean":
        frame = rolling.mean()
    elif statistic == "max":
        frame = rolling.max()
    elif statistic == "min":
        frame = rolling.min()
    elif statistic == "std0":
        frame = rolling.std(ddof=0)
    elif statistic == "std1":
        frame = rolling.std(ddof=1)
    elif statistic == "skew":
        frame = rolling.skew()
    else:
        frame = rolling.kurt()
    result: FloatArray = frame.to_numpy(dtype=np.float64)
    return result


def _sum(values: FloatArray, window: int) -> FloatArray:
    return _rolling(values, window, "mean") * window


def _divide(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    """numerator / denominator where the denominator is finite and non-zero, else NaN."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(denominator) & (denominator != 0), numerator / denominator, np.nan)


def _positive_ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(denominator) & (denominator > 0), numerator / denominator, np.nan)


def smooth(values: FloatArray, window: int, alpha: float) -> FloatArray:
    """Recursive exponential smoother per column (EMA: alpha=2/(n+1); Wilder: alpha=1/n).

    The first value is the simple mean of the first ``window`` consecutive finite inputs;
    afterwards s_t = s_{t-1} + alpha (x_t - s_{t-1}). A missing input sets the output to NaN
    and restarts the smoother, which resumes only after a new full window. The loop runs over
    sessions with vector operations across securities; row t depends on rows <= t only.
    """
    seed = _rolling(values, window, "mean")
    out = np.full(values.shape, np.nan, dtype=np.float64)
    state = np.full(values.shape[1], np.nan, dtype=np.float64)
    run = np.zeros(values.shape[1], dtype=np.int64)
    for row in range(values.shape[0]):
        current = values[row]
        finite = np.isfinite(current)
        run = np.where(finite, run + 1, 0)
        updated = state + alpha * (current - state)
        state = np.where(run == window, seed[row], np.where(run > window, updated, np.nan))
        out[row] = state
    return out


def _complete(values: FloatArray, window: int) -> BoolArray:
    count = _rolling(np.isfinite(values).astype(np.float64), window, "mean")
    return np.asarray(count == 1.0)


def _window_blocks(values: FloatArray, window: int) -> list[tuple[int, npt.NDArray[np.float64]]]:
    """(first output row, sliding block rows x securities x window) chunks to bound memory."""
    if values.shape[0] < window:
        return []
    view = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
    return [(start + window - 1, view[start:start + _WINDOW_CHUNK_ROWS])
            for start in range(0, view.shape[0], _WINDOW_CHUNK_ROWS)]


def sessions_since_extreme(values: FloatArray, window: int) -> FloatArray:
    """Sessions since the most recent maximum within rows t-window+1..t (0 = today); complete windows only."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    filled = np.where(np.isfinite(values), values, -np.inf)
    for row, block in _window_blocks(filled, window):
        age = np.argmax(block[..., ::-1], axis=-1)
        out[row:row + block.shape[0]] = age
    return np.where(_complete(values, window), out, np.nan)


def mean_absolute_deviation(values: FloatArray, window: int) -> FloatArray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for row, block in _window_blocks(values, window):
        center = block.mean(axis=-1, keepdims=True)
        out[row:row + block.shape[0]] = np.abs(block - center).mean(axis=-1)
    return out


def _moments(x: FloatArray, y: FloatArray, window: int) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Rolling population covariance and variances over cells where both inputs are finite (complete windows)."""
    valid = np.isfinite(x) & np.isfinite(y)
    xs, ys = np.where(valid, x, np.nan), np.where(valid, y, np.nan)
    mean_x, mean_y = _rolling(xs, window, "mean"), _rolling(ys, window, "mean")
    covariance = _rolling(xs * ys, window, "mean") - mean_x * mean_y
    variance_x = _rolling(xs * xs, window, "mean") - mean_x * mean_x
    variance_y = _rolling(ys * ys, window, "mean") - mean_y * mean_y
    return covariance, variance_x, variance_y


def _rsi(change: FloatArray, window: int) -> FloatArray:
    finite = np.isfinite(change)
    gain = np.where(finite, np.maximum(change, 0.0), np.nan)
    loss = np.where(finite, np.maximum(-change, 0.0), np.nan)
    average_gain, average_loss = smooth(gain, window, 1.0 / window), smooth(loss, window, 1.0 / window)
    total = average_gain + average_loss
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(total > 0, 100.0 * average_gain / total, np.where(np.isfinite(total), 50.0, np.nan))


def _streak(returns: FloatArray) -> FloatArray:
    """Signed count of consecutive up (+) or down (-) closes ending at t, capped at +-10; needs 10 finite returns."""
    up = np.zeros(returns.shape, dtype=np.float64)
    down = np.zeros(returns.shape, dtype=np.float64)
    alive_up = np.ones(returns.shape, dtype=bool)
    alive_down = np.ones(returns.shape, dtype=bool)
    for distance in range(STREAK_CAP):
        lagged = _lag(returns, distance)
        alive_up &= lagged > 0
        alive_down &= lagged < 0
        up += alive_up
        down += alive_down
    return np.where(_complete(returns, STREAK_CAP), up - down, np.nan)


def compute_indicators(panel: ResearchPanel) -> dict[str, FloatArray]:
    """All additional raw indicators as float64 (sessions x securities); row t uses rows <= t only."""
    open_ = panel.open.astype(np.float64)
    high = panel.high.astype(np.float64)
    low = panel.low.astype(np.float64)
    close = panel.close.astype(np.float64)
    volume = panel.volume.astype(np.float64)
    previous_close = _lag(close, 1)
    change = close - previous_close
    returns = daily_returns(close)
    market = np.broadcast_to(daily_returns(panel.spy_close).reshape(-1, 1), close.shape)
    out: dict[str, FloatArray] = {}

    out["rsi_14"] = _rsi(change, 14)
    out["rsi_5"] = _rsi(change, 5)
    low_14, high_14 = _rolling(low, 14, "min"), _rolling(high, 14, "max")
    out["stochastic_k_14"] = _positive_ratio(close - low_14, high_14 - low_14)
    typical = (high + low + close) / 3.0
    out["cci_20"] = _positive_ratio(typical - _rolling(typical, 20, "mean"), 0.015 * mean_absolute_deviation(typical, 20))
    macd = smooth(close, 12, 2.0 / 13.0) - smooth(close, 26, 2.0 / 27.0)
    out["macd_hist_pct"] = _positive_ratio(macd - smooth(macd, 9, 0.2), close)
    sma_20, deviation_20 = _rolling(close, 20, "mean"), _rolling(close, 20, "std0")
    out["bollinger_pct_b_20"] = _positive_ratio(close - (sma_20 - 2.0 * deviation_20), 4.0 * deviation_20)
    out["bollinger_bandwidth_20"] = _positive_ratio(4.0 * deviation_20, sma_20)
    sma_50, sma_200 = _rolling(close, 50, "mean"), _rolling(close, 200, "mean")
    out["dist_sma_20"] = _positive_ratio(close, sma_20) - 1.0
    out["dist_sma_50"] = _positive_ratio(close, sma_50) - 1.0
    out["dist_sma_200"] = _positive_ratio(close, sma_200) - 1.0
    out["sma_50_200_ratio"] = _positive_ratio(sma_50, sma_200) - 1.0

    true_range = np.maximum(high - low, np.maximum(np.abs(high - previous_close), np.abs(low - previous_close)))
    atr_14 = smooth(true_range, 14, 1.0 / 14.0)
    up_move, down_move = high - _lag(high, 1), _lag(low, 1) - low
    moves_known = np.isfinite(up_move) & np.isfinite(down_move)
    plus_dm = np.where(moves_known, np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), np.nan)
    minus_dm = np.where(moves_known, np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), np.nan)
    plus_di = 100.0 * _positive_ratio(smooth(plus_dm, 14, 1.0 / 14.0), atr_14)
    minus_di = 100.0 * _positive_ratio(smooth(minus_dm, 14, 1.0 / 14.0), atr_14)
    di_total = plus_di + minus_di
    with np.errstate(invalid="ignore", divide="ignore"):
        dx = np.where(di_total > 0, 100.0 * np.abs(plus_di - minus_di) / di_total,
                      np.where(np.isfinite(di_total), 0.0, np.nan))
    out["adx_14"] = smooth(dx, 14, 1.0 / 14.0)
    out["di_diff_14"] = plus_di - minus_di
    aroon_up = 100.0 * (25.0 - sessions_since_extreme(high, 26)) / 25.0
    aroon_down = 100.0 * (25.0 - sessions_since_extreme(-low, 26)) / 25.0
    out["aroon_osc_25"] = aroon_up - aroon_down
    low_55, high_55 = _rolling(low, 55, "min"), _rolling(high, 55, "max")
    out["donchian_pos_55"] = _positive_ratio(close - low_55, high_55 - low_55)
    out["atr_pct_14"] = _positive_ratio(atr_14, close)
    out["keltner_pos_20"] = _positive_ratio(close - smooth(close, 20, 2.0 / 21.0), smooth(true_range, 20, 1.0 / 20.0))

    typical_change = typical - _lag(typical, 1)
    flow = typical * volume
    flow_known = np.isfinite(typical_change) & np.isfinite(flow)
    positive_flow = _sum(np.where(flow_known, np.where(typical_change > 0, flow, 0.0), np.nan), 14)
    negative_flow = _sum(np.where(flow_known, np.where(typical_change < 0, flow, 0.0), np.nan), 14)
    total_flow = positive_flow + negative_flow
    with np.errstate(invalid="ignore", divide="ignore"):
        out["mfi_14"] = np.where(total_flow > 0, 100.0 * positive_flow / total_flow,
                                 np.where(np.isfinite(total_flow), 50.0, np.nan))
    bar_range = high - low
    with np.errstate(invalid="ignore", divide="ignore"):
        multiplier = np.where(bar_range > 0, ((close - low) - (high - close)) / bar_range,
                              np.where(np.isfinite(bar_range), 0.0, np.nan))
    volume_20 = _sum(volume, 20)
    out["cmf_20"] = _positive_ratio(_sum(multiplier * volume, 20), volume_20)
    out["obv_change_20"] = _positive_ratio(_sum(np.sign(change) * volume, 20), volume_20)
    change_known = np.isfinite(change) & np.isfinite(volume)
    up_volume = _sum(np.where(change_known, np.where(change > 0, volume, 0.0), np.nan), 20)
    down_volume = _sum(np.where(change_known, np.where(change < 0, volume, 0.0), np.nan), 20)
    out["up_down_volume_ratio_20"] = _positive_ratio(up_volume, down_volume)
    out["volume_z_20"] = _positive_ratio(volume - _rolling(volume, 20, "mean"), _rolling(volume, 20, "std1"))
    mean_dollar_volume = _rolling(close * volume, 20, "mean")
    with np.errstate(invalid="ignore", divide="ignore"):
        out["dollar_volume_log_20"] = np.where(mean_dollar_volume > 0, np.log(mean_dollar_volume), np.nan)

    gap = _positive_ratio(open_, previous_close) - 1.0
    intraday = _positive_ratio(close, open_) - 1.0
    out["overnight_return_21"] = _sum(gap, 21)
    out["intraday_return_21"] = _sum(intraday, 21)
    out["gap_today"] = gap
    out["intraday_today"] = intraday
    out["close_location_today"] = _positive_ratio(close - low, bar_range)
    out["range_pct_5"] = _rolling(_positive_ratio(bar_range, close), 5, "mean")
    volatility_10, volatility_60 = _rolling(returns, 10, "std1"), _rolling(returns, 60, "std1")
    out["realized_vol_10"] = volatility_10
    out["realized_vol_60"] = volatility_60
    out["vol_ratio_10_60"] = _positive_ratio(volatility_10, volatility_60)
    out["skew_60"] = _rolling(returns, 60, "skew")
    out["kurt_60"] = _rolling(returns, 60, "kurt")
    out["downside_vol_60"] = np.sqrt(_rolling(np.minimum(returns, 0.0) ** 2, 60, "mean"))
    out["days_since_252_high"] = sessions_since_extreme(high, 252)
    out["up_days_fraction_21"] = _rolling(np.where(np.isfinite(returns), (returns > 0).astype(np.float64), np.nan),
                                          21, "mean")
    out["streak_length"] = _streak(returns)
    for sessions in (1, 3, 10, 63, 126, 252):
        out[f"return_{sessions}d"] = _positive_ratio(close, _lag(close, sessions)) - 1.0
    out["efficiency_ratio_20"] = _positive_ratio(np.abs(close - _lag(close, 20)), _sum(np.abs(change), 20))
    return_21 = _positive_ratio(close, _lag(close, 21)) - 1.0
    return_5 = _positive_ratio(close, _lag(close, 5)) - 1.0
    out["sector_relative_return_21"] = return_21 - sector_mean(return_21, panel.sector, panel.eligible)
    out["sector_relative_return_5"] = return_5 - sector_mean(return_5, panel.sector, panel.eligible)
    covariance, _, market_variance = _moments(returns, market, 252)
    beta = _positive_ratio(covariance, market_variance)
    pair_known = np.isfinite(returns) & np.isfinite(market)
    stock_sum = _sum(np.where(pair_known, returns, np.nan), 21)
    market_sum = _sum(np.where(pair_known, market, np.nan), 21)
    out["residual_return_21"] = stock_sum - beta * market_sum
    volume_change = _positive_ratio(volume, _lag(volume, 1)) - 1.0
    covariance_20, variance_r, variance_v = _moments(returns, volume_change, 20)
    out["price_volume_corr_20"] = _positive_ratio(covariance_20, np.sqrt(np.maximum(variance_r, 0.0)
                                                                        * np.maximum(variance_v, 0.0)))
    low_252, high_252 = _rolling(low, 252, "min"), _rolling(high, 252, "max")
    out["range_position_252"] = _positive_ratio(close - low_252, high_252 - low_252)
    out["momentum_change_21"] = return_21 - _lag(return_21, 21)
    if tuple(out) != INDICATOR_NAMES:
        raise RuntimeError("indicator order differs from INDICATOR_NAMES")
    return out


def sector_centered_rank(values: FloatArray, include: BoolArray, sector: npt.NDArray[np.integer],
                         minimum: int = SECTOR_MINIMUM_STOCKS) -> FloatArray:
    """Centered rank (-0.5..+0.5) within the stock's sector among included finite cells on the date.

    Cells without a sector, or in a sector with fewer than ``minimum`` included finite cells on the
    date, are NaN. One row rank handles all sectors: key = 2 x sector code + overall centered rank.
    """
    base = centered_rank(values, include)
    valid = np.isfinite(base) & (sector >= 0)
    key = np.where(valid, 2.0 * sector.astype(np.float64) + base, np.nan)
    ranks = pd.DataFrame(key).rank(axis=1, method="average").to_numpy(dtype=np.float64)
    offset = np.zeros(values.shape, dtype=np.float64)
    count = np.zeros(values.shape, dtype=np.float64)
    below = np.zeros(values.shape[0], dtype=np.float64)
    for code in np.unique(sector[valid]):
        members = valid & (sector == code)
        members_per_date = members.sum(axis=1).astype(np.float64)
        count = np.where(members, members_per_date[:, None], count)
        offset = np.where(members, below[:, None], offset)
        below = below + members_per_date
    with np.errstate(invalid="ignore", divide="ignore"):
        centered = np.where(count > 1, (ranks - offset - 1.0) / (count - 1.0) - 0.5, 0.0)
    return np.where(valid & (count >= minimum), centered, np.nan)


@dataclass(frozen=True)
class FeatureSets:
    """Centered-rank features (float32, sessions x securities) and the frozen membership of each feature set."""

    features: dict[str, Float32Array]
    sets: dict[str, tuple[str, ...]]


def feature_set_members() -> dict[str, tuple[str, ...]]:
    all_names = PREDICTOR_NAMES + INDICATOR_NAMES
    return {"set_12": PREDICTOR_NAMES, "set_all": all_names,
            "set_all_sector_neutral": all_names + tuple(f"{name}{SECTOR_SUFFIX}" for name in all_names)}


def build_feature_sets(panel: ResearchPanel) -> FeatureSets:
    """Rank the 12 step-1 predictors and every new indicator among eligible stocks, plus within-sector versions."""
    eligible = panel.eligible
    features: dict[str, Float32Array] = {}
    raw = compute_predictors(panel)
    for name in PREDICTOR_NAMES:
        features[name] = centered_rank(raw.pop(name), eligible).astype(np.float32)
    raw = compute_indicators(panel)
    for name in INDICATOR_NAMES:
        features[name] = centered_rank(raw.pop(name), eligible).astype(np.float32)
    del raw
    for name in PREDICTOR_NAMES + INDICATOR_NAMES:
        features[f"{name}{SECTOR_SUFFIX}"] = sector_centered_rank(
            features[name].astype(np.float64), eligible, panel.sector).astype(np.float32)
    return FeatureSets(features=features, sets=feature_set_members())
