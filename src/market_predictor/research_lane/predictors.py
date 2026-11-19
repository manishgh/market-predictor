"""Textbook price-based return predictors computed from data through the close of session t.

Every predictor requires a complete window (no partial windows, no fills); an
incomplete window yields NaN. Daily returns are close-to-close on adjusted prices.
Expected directions are frozen in ``EXPECTED_SIGNS`` before any result is seen:
+1 means a higher value is expected to precede a higher forward return.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from market_predictor.research_lane.panel import ResearchPanel

FloatArray = npt.NDArray[np.float64]

EXPECTED_SIGNS: Final[Mapping[str, int]] = MappingProxyType({
    "mom_12_1": 1,
    "mom_6_1": 1,
    "residual_mom_12_1": 1,
    "sector_mom_6m": 1,
    "high_52w_proximity": 1,
    "reversal_5d": -1,
    "reversal_21d": -1,
    "idio_vol_60d": -1,
    "beta_252d": -1,
    "max_daily_return_21d": -1,
    "amihud_illiquidity_60d": 1,
    "volume_ratio_5_60": 1,
})
PREDICTOR_NAMES: Final = tuple(EXPECTED_SIGNS)
COMPOSITE_GROUPS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    "composite_all": PREDICTOR_NAMES,
    "composite_trend": PREDICTOR_NAMES[0:5],
    "composite_reversal": PREDICTOR_NAMES[5:7],
    "composite_risk_liquidity": PREDICTOR_NAMES[7:12],
})


def minimum_present(group_size: int) -> int:
    """At least two thirds of a group's predictors must be present (8 of 12, 4 of 5, 2 of 2)."""
    return math.ceil(2 * group_size / 3)


def _lag(values: FloatArray, sessions: int) -> FloatArray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if sessions < values.shape[0]:
        out[sessions:] = values[:values.shape[0] - sessions]
    return out


def _rolling(values: FloatArray, window: int, statistic: Literal["mean", "max"]) -> FloatArray:
    matrix = values if values.ndim == 2 else values.reshape(-1, 1)
    rolling = pd.DataFrame(matrix).rolling(window, min_periods=window)
    result: FloatArray = (rolling.mean() if statistic == "mean" else rolling.max()).to_numpy(dtype=np.float64)
    return result if values.ndim == 2 else result[:, 0]


def daily_returns(close: npt.NDArray[np.floating]) -> FloatArray:
    prices = np.asarray(close, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        return prices / _lag(prices, 1) - 1.0


@dataclass(frozen=True)
class _Regression:
    beta: FloatArray
    mean_stock: FloatArray
    mean_market: FloatArray
    stock_variance: FloatArray
    covariance: FloatArray
    market_variance: FloatArray


def _regression(stock: FloatArray, market: FloatArray, window: int) -> _Regression:
    """Rolling population moments of stock vs market daily returns over complete windows only."""
    market_matrix = np.broadcast_to(market.reshape(-1, 1), stock.shape)
    valid = np.isfinite(stock) & np.isfinite(market_matrix)
    r = np.where(valid, stock, np.nan)
    m = np.where(valid, market_matrix, np.nan)
    mean_r, mean_m = _rolling(r, window, "mean"), _rolling(m, window, "mean")
    covariance = _rolling(r * m, window, "mean") - mean_r * mean_m
    market_variance = _rolling(m * m, window, "mean") - mean_m * mean_m
    stock_variance = _rolling(r * r, window, "mean") - mean_r * mean_r
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(market_variance > 0, covariance / market_variance, np.nan)
    return _Regression(beta, mean_r, mean_m, stock_variance, covariance, market_variance)


def _ratio(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(denominator) & (denominator > 0), numerator / denominator, np.nan)


def sector_mean(values: FloatArray, sector: npt.NDArray[np.integer], include: npt.NDArray[np.bool_]) -> FloatArray:
    """Per-date mean of ``values`` over included finite cells of the same sector, broadcast to sector members."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    usable = include & np.isfinite(values)
    for code in np.unique(sector[sector >= 0]):
        members = sector == code
        selected = usable & members
        count = selected.sum(axis=1)
        total = np.where(selected, values, 0.0).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(count > 0, total / np.maximum(count, 1), np.nan)
        out = np.where(members, mean.reshape(-1, 1), out)
    return out


def compute_predictors(panel: ResearchPanel) -> dict[str, FloatArray]:
    """All twelve raw predictors as float64 (sessions x securities); value at row t uses rows <= t only."""
    close = panel.close.astype(np.float64)
    high = panel.high.astype(np.float64)
    volume = panel.volume.astype(np.float64)
    returns = daily_returns(close)
    market = daily_returns(panel.spy_close)

    # Residual momentum: daily returns of sessions t-252..t-21 inclusive (232 returns), beta from that window.
    residual_window = 232
    long = _regression(_lag(returns, 21), _lag(market, 21), residual_window)
    residual_momentum = residual_window * (long.mean_stock - long.beta * long.mean_market)

    # Idiosyncratic volatility: residual standard deviation of a 60-session regression on SPY,
    # sum of squared OLS residuals / (n - 2).
    short = _regression(returns, market, 60)
    with np.errstate(invalid="ignore", divide="ignore"):
        residual_variance = np.where(short.market_variance > 0, short.stock_variance
                                     - short.covariance * short.covariance / short.market_variance, np.nan)
    idio_vol = np.where(np.isfinite(residual_variance), np.sqrt(np.maximum(residual_variance, 0.0) * 60.0 / 58.0), np.nan)

    beta_252 = _regression(returns, market, 252).beta
    return_126 = _ratio(close, _lag(close, 126)) - 1.0
    dollar_volume = close * volume
    amihud_daily = _ratio(np.abs(returns), dollar_volume)
    eligible = panel.eligible
    return {
        "mom_12_1": _ratio(_lag(close, 21), _lag(close, 252)) - 1.0,
        "mom_6_1": _ratio(_lag(close, 21), _lag(close, 126)) - 1.0,
        "residual_mom_12_1": residual_momentum,
        "sector_mom_6m": sector_mean(return_126, panel.sector, eligible),
        "high_52w_proximity": _ratio(close, _rolling(high, 252, "max")),
        "reversal_5d": _ratio(close, _lag(close, 5)) - 1.0,
        "reversal_21d": _ratio(close, _lag(close, 21)) - 1.0,
        "idio_vol_60d": idio_vol,
        "beta_252d": beta_252,
        "max_daily_return_21d": _rolling(returns, 21, "max"),
        "amihud_illiquidity_60d": _rolling(amihud_daily, 60, "mean"),
        "volume_ratio_5_60": _ratio(_rolling(volume, 5, "mean"), _rolling(volume, 60, "mean")),
    }


def centered_rank(values: FloatArray, include: npt.NDArray[np.bool_]) -> FloatArray:
    """Per-date percentile rank among included finite cells, centered to -0.5..+0.5 (ties share the average rank)."""
    masked = np.where(include & np.isfinite(values), values, np.nan)
    ranks = pd.DataFrame(masked).rank(axis=1, method="average").to_numpy(dtype=np.float64)
    count = np.isfinite(masked).sum(axis=1, keepdims=True).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        centered = np.where(count > 1, (ranks - 1.0) / (count - 1.0) - 0.5, 0.0)
    return np.where(np.isfinite(masked), centered, np.nan)


def compute_composites(predictors: Mapping[str, FloatArray], eligible: npt.NDArray[np.bool_]) -> dict[str, FloatArray]:
    """Average signed centered ranks among eligible stocks; already oriented so +1 is the expected direction."""
    signed_ranks = {name: EXPECTED_SIGNS[name] * centered_rank(predictors[name], eligible) for name in PREDICTOR_NAMES}
    composites: dict[str, FloatArray] = {}
    for composite, members in COMPOSITE_GROUPS.items():
        stack = np.stack([signed_ranks[name] for name in members])
        present = np.isfinite(stack).sum(axis=0)
        total = np.where(np.isfinite(stack), stack, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / present
        composites[composite] = np.where(present >= minimum_present(len(members)), mean, np.nan)
    return composites
