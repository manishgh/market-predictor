"""Deterministic Stage-1 swing setup scanners and stationary ATR-normalized event features.

Every feature and setup flag at session ``t`` uses adjusted OHLCV and catalyst state through
the close of session ``t`` only (verified by truncated-history unit tests). No raw dollar price,
raw moving average or raw volume enters ``SETUP_FEATURE_NAMES``.

Volatility ``V_t`` defaults to Wilder's 14-session Average True Range (``wilder_atr_14``) and
also supports 14-session Median True Range (``mtr_14``) for single-variable ceteris paribus
testing under Hypothesis V2-01.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd

from market_predictor.research_lane.indicators import smooth
from market_predictor.research_lane.news_catalysts import CATEGORY_CODES, CatalystMatrix
from market_predictor.research_lane.panel import ResearchPanel
from market_predictor.research_lane.predictors import daily_returns

VolatilityMode = Literal["wilder_atr_14", "mtr_14"]

PRIMARY_MIN_ADR_PCT: Final = 0.015   # 1.5% 20-day average daily range floor for S&P 500
STRICT_MIN_ADR_PCT: Final = 0.025    # 2.5% Gemini reference screen (reported in diagnostics)
SHOCK_RATIO_THRESHOLD: Final = 3.0   # max(TR_14) > 3 * MTR(14) defines post-shock Bucket B
_WINDOW_CHUNK_ROWS: Final = 64

SETUP_FEATURE_NAMES: Final = (
    "dist_ema20_atr",
    "dist_ema21_atr",
    "dist_sma50_atr",
    "dist_sma200_atr",
    "slope_ema20_5d_atr",
    "slope_sma50_20d_atr",
    "bb_width_pct60",
    "candle_close_range_pct",
    "rvol_20",
    "adr_pct_20",
    "opening_gap_atr",
    "rel_strength_spy_5d_atr",
    "rel_strength_spy_20d_atr",
    "spy_dist_sma50_vol",
    "spy_slope_sma50_20d_vol",
    "spy_rv10_60_ratio",
    "has_tier1_catalyst_2d",
    "sessions_since_catalyst",
    "pre_catalyst_10d_run_atr",
    "catalyst_reaction_atr",
    "has_bearish_catalyst_3d",
    "cat_earnings",
    "cat_guidance",
    "cat_analyst",
    "cat_corporate_other",
    "setup_a_catalyst",
    "setup_b_pullback",
    "setup_c_squeeze",
)

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]


def _lag(values: FloatArray, sessions: int) -> FloatArray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if 0 < sessions < values.shape[0]:
        out[sessions:] = values[: values.shape[0] - sessions]
    elif sessions == 0:
        out[:] = values
    return out


def _rolling_1d_or_2d(values: FloatArray, window: int, stat: Literal["mean", "max", "min", "std0"]) -> FloatArray:
    matrix = values if values.ndim == 2 else values.reshape(-1, 1)
    roll = pd.DataFrame(matrix).rolling(window, min_periods=window)
    if stat == "mean":
        res = roll.mean()
    elif stat == "max":
        res = roll.max()
    elif stat == "min":
        res = roll.min()
    else:
        res = roll.std(ddof=0)
    arr: FloatArray = res.to_numpy(dtype=np.float64)
    return arr if values.ndim == 2 else arr[:, 0]


def _complete(values: FloatArray, window: int) -> BoolArray:
    finite_share = _rolling_1d_or_2d(np.isfinite(values).astype(np.float64), window, "mean")
    return np.asarray(finite_share == 1.0, dtype=bool)


def _pos_div(numerator: FloatArray, denominator: FloatArray) -> FloatArray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(np.isfinite(denominator) & (denominator > 0.0), numerator / denominator, np.nan)


def _rolling_median_2d(values: FloatArray, window: int) -> FloatArray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if values.shape[0] < window:
        return out
    view = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
    for start in range(0, view.shape[0], _WINDOW_CHUNK_ROWS):
        block = view[start : start + _WINDOW_CHUNK_ROWS]
        with np.errstate(invalid="ignore"):
            med = np.median(block, axis=-1)
        out[start + window - 1 : start + window - 1 + block.shape[0]] = med
    return np.where(_complete(values, window), out, np.nan)


def _rolling_percentile_rank_2d(values: FloatArray, window: int) -> FloatArray:
    """Causal percentile rank in [0, 1] of ``values[t]`` within ``values[t-window+1 .. t]``."""
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if values.shape[0] < window or window < 2:
        return out
    view = np.lib.stride_tricks.sliding_window_view(values, window, axis=0)
    denom = float(window - 1)
    for start in range(0, view.shape[0], _WINDOW_CHUNK_ROWS):
        block = view[start : start + _WINDOW_CHUNK_ROWS]
        last = block[..., -1:]
        prior = block[..., :-1]
        less = (prior < last).sum(axis=-1)
        equal = (prior == last).sum(axis=-1)
        out[start + window - 1 : start + window - 1 + block.shape[0]] = (less + 0.5 * equal) / denom
    return np.where(_complete(values, window), out, np.nan)


def compute_true_range_and_volatility(
    panel: ResearchPanel,
    mode: VolatilityMode = "wilder_atr_14",
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Return ``(volatility_14, mtr_14, shock_ratio_14)`` as float64 matrices (sessions x securities)."""
    high = panel.high.astype(np.float64)
    low = panel.low.astype(np.float64)
    close = panel.close.astype(np.float64)
    prev_close = _lag(close, 1)
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    # Row 0 has no prior close; true range requires prev_close to capture overnight gaps.
    tr = np.where(np.isfinite(prev_close) & np.isfinite(high) & np.isfinite(low) & (high >= low), tr, np.nan)
    wilder_14 = smooth(tr, 14, 1.0 / 14.0)
    mtr_14 = _rolling_median_2d(tr, 14)
    max_tr_14 = _rolling_1d_or_2d(tr, 14, "max")
    shock_ratio = _pos_div(max_tr_14, mtr_14)
    chosen = wilder_14 if mode == "wilder_atr_14" else mtr_14
    return chosen, mtr_14, shock_ratio


@dataclass(frozen=True)
class SetupDetection:
    """Detected Stage-1 candidate masks, stationary features, volatility and diagnostics."""

    volatility_mode: VolatilityMode
    volatility: FloatArray
    mtr_14: FloatArray
    shock_ratio_14: FloatArray
    adr_pct_20: FloatArray
    spy_bull_regime: BoolArray  # 1D length = sessions
    setup_a: BoolArray          # sessions x securities (on eligible rows >= first_row, before ADR screen)
    setup_b: BoolArray
    setup_c: BoolArray
    any_setup_unfiltered: BoolArray
    any_setup_primary: BoolArray     # adr_pct_20 >= 1.5%
    any_setup_strict_adr: BoolArray  # adr_pct_20 >= 2.5%
    features: dict[str, Float32Array]
    report: dict[str, Any]


def build_setups_and_features(
    panel: ResearchPanel,
    catalysts: CatalystMatrix,
    *,
    first_row: int,
    volatility_mode: VolatilityMode = "wilder_atr_14",
) -> SetupDetection:
    """Compute stationary ATR-normalized features and Stage-1 setups A, B and C."""
    open_ = panel.open.astype(np.float64)
    high = panel.high.astype(np.float64)
    low = panel.low.astype(np.float64)
    close = panel.close.astype(np.float64)
    volume = panel.volume.astype(np.float64)
    spy_close = panel.spy_close.astype(np.float64)
    spy_open = panel.spy_open.astype(np.float64)

    vol, mtr_14, shock_ratio_14 = compute_true_range_and_volatility(panel, volatility_mode)
    vol_pct = _pos_div(vol, close)

    ema_20 = smooth(close, 20, 2.0 / 21.0)
    ema_21 = smooth(close, 21, 2.0 / 22.0)
    sma_20 = _rolling_1d_or_2d(close, 20, "mean")
    sma_50 = _rolling_1d_or_2d(close, 50, "mean")
    sma_200 = _rolling_1d_or_2d(close, 200, "mean")
    std_20 = _rolling_1d_or_2d(close, 20, "std0")

    dist_ema20_atr = _pos_div(close - ema_20, vol)
    dist_ema21_atr = _pos_div(close - ema_21, vol)
    dist_sma50_atr = _pos_div(close - sma_50, vol)
    dist_sma200_atr = _pos_div(close - sma_200, vol)
    slope_ema20_5d_atr = _pos_div(ema_20 - _lag(ema_20, 5), vol)
    slope_sma50_20d_atr = _pos_div(sma_50 - _lag(sma_50, 20), vol)

    bb_width = _pos_div(4.0 * std_20, sma_20)
    bb_width_pct60 = _rolling_percentile_rank_2d(bb_width, 60)

    bar_range = high - low
    with np.errstate(invalid="ignore", divide="ignore"):
        close_range_pct = np.where(
            np.isfinite(bar_range) & (bar_range > 0.0),
            (close - low) / bar_range,
            np.where(np.isfinite(bar_range) & (bar_range == 0.0), 0.0, np.nan),
        )

    # Prior-20-session volume baseline excludes session t so a catalyst spike never inflates its own denominator.
    prior_vol_sma20 = _rolling_1d_or_2d(_lag(volume, 1), 20, "mean")
    rvol_20 = _pos_div(volume, prior_vol_sma20)
    adr_pct_20 = _rolling_1d_or_2d(_pos_div(high - low, close), 20, "mean")
    opening_gap_atr = _pos_div(open_ - _lag(close, 1), _lag(vol, 1))

    ret_1d = daily_returns(close)
    ret_5d = _pos_div(close, _lag(close, 5)) - 1.0
    ret_20d = _pos_div(close, _lag(close, 20)) - 1.0

    spy_ret_1d = daily_returns(spy_close)
    spy_ret_5d = _pos_div(spy_close, _lag(spy_close, 5)) - 1.0
    spy_ret_20d = _pos_div(spy_close, _lag(spy_close, 20)) - 1.0
    spy_vol_pct_14 = smooth(np.abs(spy_ret_1d).reshape(-1, 1), 14, 1.0 / 14.0)[:, 0]
    spy_sma50 = _rolling_1d_or_2d(spy_close, 50, "mean")
    spy_dist_sma50_1d = _pos_div((spy_close / spy_sma50) - 1.0, spy_vol_pct_14)
    spy_slope_sma50_1d = _pos_div((spy_sma50 - _lag(spy_sma50, 20)) / _lag(spy_sma50, 20), spy_vol_pct_14)
    spy_rv10 = _rolling_1d_or_2d(spy_ret_1d, 10, "std0")
    spy_rv60 = _rolling_1d_or_2d(spy_ret_1d, 60, "std0")
    spy_rv10_60_1d = _pos_div(spy_rv10, spy_rv60)
    spy_bull_1d = np.asarray(
        np.isfinite(spy_close) & np.isfinite(spy_sma50) & np.isfinite(_lag(spy_sma50, 20))
        & (spy_close > spy_sma50) & (spy_sma50 > _lag(spy_sma50, 20)),
        dtype=bool,
    )

    rel_strength_5d = _pos_div(ret_5d, vol_pct) - _pos_div(spy_ret_5d, spy_vol_pct_14)[:, None]
    rel_strength_20d = _pos_div(ret_20d, vol_pct) - _pos_div(spy_ret_20d, spy_vol_pct_14)[:, None]

    # Catalyst interaction features (causal at row t: e = latest_tier1_row[t, c] <= t).
    tier1_t = catalysts.tier1_on_session
    tier1_prev = np.vstack([np.zeros((1, close.shape[1]), dtype=bool), tier1_t[:-1]])
    bear_t = catalysts.bearish_on_session
    bear_prev = np.vstack([np.zeros((1, close.shape[1]), dtype=bool), bear_t[:-1]])
    has_tier1_2d = (tier1_t | tier1_prev).astype(np.float64)

    pre_cat_run = np.full(close.shape, np.nan, dtype=np.float64)
    erow = catalysts.latest_tier1_row
    valid_e = erow >= 11
    if bool(valid_e.any()):
        r_idx, c_idx = np.nonzero(valid_e)
        e_vals = erow[r_idx, c_idx]
        c_before = close[e_vals - 1, c_idx]
        c_early = close[e_vals - 11, c_idx]
        v_before = vol[e_vals - 1, c_idx]
        with np.errstate(invalid="ignore", divide="ignore"):
            pre_cat_run[r_idx, c_idx] = np.where(
                np.isfinite(v_before) & (v_before > 0.0),
                (c_before - c_early) / v_before,
                np.nan,
            )

    stock_oc_or_1d = ret_1d - spy_ret_1d[:, None]
    spy_oc = _pos_div(spy_close - spy_open, spy_open)
    stock_oc = _pos_div(close - open_, open_)
    # Market-adjusted daily reaction normalized by daily volatility% (combines gap + intraday move).
    catalyst_reaction_atr = _pos_div(stock_oc_or_1d, vol_pct)

    cat_latest = catalysts.latest_tier1_category
    cat_earnings = (cat_latest == CATEGORY_CODES["earnings"]).astype(np.float64)
    cat_guidance = (cat_latest == CATEGORY_CODES["guidance"]).astype(np.float64)
    cat_analyst = (cat_latest == CATEGORY_CODES["analyst_revision"]).astype(np.float64)
    cat_corp_other = np.isin(
        cat_latest,
        (
            CATEGORY_CODES["merger_acquisition"],
            CATEGORY_CODES["regulatory_decision"],
            CATEGORY_CODES["product_event"],
        ),
    ).astype(np.float64)

    # --- Stage 1 Deterministic Scanners ---
    decision_mask = np.zeros(close.shape, dtype=bool)
    decision_mask[first_row:] = True
    base_ok = panel.eligible & decision_mask & np.isfinite(vol) & (vol > 0.0) & np.isfinite(close) & (close > 0.0)

    # Setup A: Confirmed Post-Catalyst Breakout / Drift
    # Tier-1 positive/structural catalyst in {t, t-1}, no bearish catalyst in {t, t-1},
    # RVOL_20 > 2.0, positive market-adjusted return AND non-negative regular-session candle vs SPY,
    # close in top 30% of daily range.
    setup_a = (
        base_ok
        & (tier1_t | tier1_prev)
        & ~(bear_t | bear_prev)
        & np.isfinite(rvol_20) & (rvol_20 > 2.0)
        & np.isfinite(stock_oc_or_1d) & (stock_oc_or_1d > 0.0)
        & np.isfinite(stock_oc) & ((stock_oc - spy_oc[:, None]) > -0.005)
        & np.isfinite(close_range_pct) & (close_range_pct >= 0.70)
    )

    # Setup B: Uptrend 21-EMA Controlled Pullback & Reversal
    # Above rising SMA_50 > SMA_200, prior 5-day impulse >= 2.0 ATR within last 15 sessions (ending <= t-2),
    # 2-day pullback mean RVOL < 0.85, low/close tests within 0.5 ATR of EMA_21, bullish reversal close > High[t-1].
    gain_5d_atr = _pos_div(close - _lag(close, 5), _lag(vol, 5))
    max_prior_impulse = _rolling_1d_or_2d(_lag(gain_5d_atr, 2), 14, "max")
    mean_pullback_rvol = 0.5 * (_lag(rvol_20, 1) + _lag(rvol_20, 2))
    dist_low_ema21 = np.minimum(
        np.abs(low - ema_21),
        np.minimum(np.abs(_lag(low, 1) - _lag(ema_21, 1)), np.abs(_lag(close, 1) - _lag(ema_21, 1))),
    )
    setup_b = (
        base_ok
        & np.isfinite(sma_50) & np.isfinite(sma_200) & np.isfinite(_lag(sma_50, 20))
        & (close > sma_50) & (sma_50 > sma_200) & (sma_50 > _lag(sma_50, 20))
        & np.isfinite(max_prior_impulse) & (max_prior_impulse >= 2.0)
        & np.isfinite(mean_pullback_rvol) & (mean_pullback_rvol < 0.85)
        & np.isfinite(dist_low_ema21) & (dist_low_ema21 <= 0.5 * vol)
        & np.isfinite(_lag(high, 1)) & (close > _lag(high, 1)) & (close > open_)
    )

    # Setup C: Volatility Contraction / Bollinger Squeeze Breakout
    # Prior session 60-day Bollinger Bandwidth percentile < 0.15, close breaks above prior 10-day high,
    # RVOL_20 > 1.5, close in top 25% of daily range.
    prior_10d_high = _rolling_1d_or_2d(_lag(high, 1), 10, "max")
    setup_c = (
        base_ok
        & np.isfinite(_lag(bb_width_pct60, 1)) & (_lag(bb_width_pct60, 1) < 0.15)
        & np.isfinite(prior_10d_high) & (close > prior_10d_high)
        & np.isfinite(rvol_20) & (rvol_20 > 1.5)
        & np.isfinite(close_range_pct) & (close_range_pct >= 0.75)
    )

    any_unfiltered = setup_a | setup_b | setup_c
    adr_primary = np.isfinite(adr_pct_20) & (adr_pct_20 >= PRIMARY_MIN_ADR_PCT)
    adr_strict = np.isfinite(adr_pct_20) & (adr_pct_20 >= STRICT_MIN_ADR_PCT)
    any_primary = any_unfiltered & adr_primary
    any_strict = any_unfiltered & adr_strict

    n_rows, n_cols = close.shape
    spy_dist_2d = np.broadcast_to(spy_dist_sma50_1d.reshape(n_rows, 1), (n_rows, n_cols))
    spy_slope_2d = np.broadcast_to(spy_slope_sma50_1d.reshape(n_rows, 1), (n_rows, n_cols))
    spy_rv_2d = np.broadcast_to(spy_rv10_60_1d.reshape(n_rows, 1), (n_rows, n_cols))

    features: dict[str, Float32Array] = {
        "dist_ema20_atr": dist_ema20_atr.astype(np.float32),
        "dist_ema21_atr": dist_ema21_atr.astype(np.float32),
        "dist_sma50_atr": dist_sma50_atr.astype(np.float32),
        "dist_sma200_atr": dist_sma200_atr.astype(np.float32),
        "slope_ema20_5d_atr": slope_ema20_5d_atr.astype(np.float32),
        "slope_sma50_20d_atr": slope_sma50_20d_atr.astype(np.float32),
        "bb_width_pct60": bb_width_pct60.astype(np.float32),
        "candle_close_range_pct": close_range_pct.astype(np.float32),
        "rvol_20": rvol_20.astype(np.float32),
        "adr_pct_20": adr_pct_20.astype(np.float32),
        "opening_gap_atr": opening_gap_atr.astype(np.float32),
        "rel_strength_spy_5d_atr": rel_strength_5d.astype(np.float32),
        "rel_strength_spy_20d_atr": rel_strength_20d.astype(np.float32),
        "spy_dist_sma50_vol": spy_dist_2d.astype(np.float32),
        "spy_slope_sma50_20d_vol": spy_slope_2d.astype(np.float32),
        "spy_rv10_60_ratio": spy_rv_2d.astype(np.float32),
        "has_tier1_catalyst_2d": has_tier1_2d.astype(np.float32),
        "sessions_since_catalyst": catalysts.sessions_since_tier1.copy(),
        "pre_catalyst_10d_run_atr": pre_cat_run.astype(np.float32),
        "catalyst_reaction_atr": catalyst_reaction_atr.astype(np.float32),
        "has_bearish_catalyst_3d": catalysts.has_bearish_3d.copy(),
        "cat_earnings": cat_earnings.astype(np.float32),
        "cat_guidance": cat_guidance.astype(np.float32),
        "cat_analyst": cat_analyst.astype(np.float32),
        "cat_corporate_other": cat_corp_other.astype(np.float32),
        "setup_a_catalyst": setup_a.astype(np.float32),
        "setup_b_pullback": setup_b.astype(np.float32),
        "setup_c_squeeze": setup_c.astype(np.float32),
    }
    if tuple(features) != SETUP_FEATURE_NAMES:
        raise RuntimeError("setup feature dictionary order differs from SETUP_FEATURE_NAMES")

    eligible_decisions = int((panel.eligible & decision_mask).sum())
    report: dict[str, Any] = {
        "volatility_mode": volatility_mode,
        "eligible_decision_stock_sessions": eligible_decisions,
        "eligible_share_passing_adr_1_5pct": (
            float((panel.eligible & decision_mask & adr_primary).sum() / eligible_decisions)
            if eligible_decisions else None
        ),
        "eligible_share_passing_adr_2_5pct": (
            float((panel.eligible & decision_mask & adr_strict).sum() / eligible_decisions)
            if eligible_decisions else None
        ),
        "spy_bull_regime_session_share": (
            float(spy_bull_1d[first_row:].mean()) if first_row < len(spy_bull_1d) else None
        ),
        "candidate_counts_unfiltered_sp500": {
            "setup_a_catalyst": int(setup_a.sum()),
            "setup_b_pullback": int(setup_b.sum()),
            "setup_c_squeeze": int(setup_c.sum()),
            "any_setup": int(any_unfiltered.sum()),
        },
        "candidate_counts_primary_adr_1_5pct": {
            "setup_a_catalyst": int((setup_a & adr_primary).sum()),
            "setup_b_pullback": int((setup_b & adr_primary).sum()),
            "setup_c_squeeze": int((setup_c & adr_primary).sum()),
            "any_setup": int(any_primary.sum()),
        },
        "candidate_counts_strict_adr_2_5pct": {
            "setup_a_catalyst": int((setup_a & adr_strict).sum()),
            "setup_b_pullback": int((setup_b & adr_strict).sum()),
            "setup_c_squeeze": int((setup_c & adr_strict).sum()),
            "any_setup": int(any_strict.sum()),
        },
    }
    return SetupDetection(
        volatility_mode=volatility_mode,
        volatility=vol,
        mtr_14=mtr_14,
        shock_ratio_14=shock_ratio_14,
        adr_pct_20=adr_pct_20,
        spy_bull_regime=spy_bull_1d,
        setup_a=setup_a,
        setup_b=setup_b,
        setup_c=setup_c,
        any_setup_unfiltered=any_unfiltered,
        any_setup_primary=any_primary,
        any_setup_strict_adr=any_strict,
        features=features,
        report=report,
    )
