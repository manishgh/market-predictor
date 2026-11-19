"""Path-dependent Triple-Barrier and matched-interval SPY research labels for candidate setups.

Decision after the close of session ``t``; entry at the adjusted open of ``t + 1``; path
checked chronologically across ``t + 1 .. t + horizon``. To avoid right-censoring bias near
``development_end``, a decision row ``t`` is resolved only when ``t + horizon < len(sessions)``
and every stock and SPY bar in ``t + 1 .. t + horizon`` is finite and positive.

Same-bar ambiguity resolves conservatively to ``stop_first``, and fills follow
``market_predictor.execution_policy.executable_fill_prices``:
- stop hit fills at ``min(open[exit_row], stop_price)`` (penalizing adverse overnight gaps);
- target hit fills at ``target_price`` (resting limit order, never crediting upside gaps);
- timeout fills at ``close[t + horizon]``.

Every outcome built on ``ResearchPanel`` uses provider split- and dividend-adjusted prices
and is therefore a research label, not document-proven holding accounting.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt

from market_predictor.research_lane.panel import ResearchPanel

DEFAULT_TARGET_MULTIPLE: Final = 2.5
DEFAULT_STOP_MULTIPLE: Final = 1.0
GOVERNED_TARGET_MULTIPLE: Final = 3.0
GOVERNED_STOP_MULTIPLE: Final = 1.5
BARRIER_HORIZON_SESSIONS: Final = 10
FROZEN_ROUND_TRIP_COST: Final = 0.0020  # 20 bps
INFO_ROUND_TRIP_COST: Final = 0.0010    # 10 bps

OUTCOME_TARGET: Final = np.int8(1)
OUTCOME_TIMEOUT: Final = np.int8(0)
OUTCOME_STOP: Final = np.int8(-1)

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Int8Array = npt.NDArray[np.int8]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class CandidateOutcomes:
    """Resolved barrier and fixed-horizon economics for a list of (row, col) candidate events."""

    valid: BoolArray
    y_barrier: Int8Array              # 1 if target hit strictly before stop within horizon, else 0
    outcome_code: Int8Array           # +1 target, 0 timeout, -1 stop
    exit_row: IntArray                # session index of barrier/timeout exit (-1 when invalid)
    holding_sessions: IntArray        # 1..horizon (0 when invalid)
    gross_return: FloatArray          # barrier path gross return
    net_return_20bps: FloatArray      # gross_return - 0.0020
    net_return_10bps: FloatArray      # gross_return - 0.0010
    net_r_20bps: FloatArray           # net profit after 20 bps divided by initial 1R stop distance
    spy_matched_return: FloatArray    # spy_close[exit_row] / spy_open[t+1] - 1 over exact trade interval
    net_excess_spy_20bps: FloatArray  # net_return_20bps - spy_matched_return
    net_excess_spy_10bps: FloatArray  # net_return_10bps - spy_matched_return
    fixed_gross_return: FloatArray    # close[t+h] / open[t+1] - 1
    fixed_net_20bps: FloatArray       # fixed_gross_return - 0.0020
    fixed_spy_return: FloatArray      # spy_close[t+h] / spy_open[t+1] - 1
    fixed_net_excess_spy_20bps: FloatArray
    vol_scaled_fixed_excess_spy: FloatArray  # (fixed_gross - fixed_spy) / (vol[t, c] / close[t, c])


def evaluate_candidate_barriers(
    panel: ResearchPanel,
    volatility: FloatArray,
    rows: IntArray,
    cols: IntArray,
    *,
    target_multiple: float = DEFAULT_TARGET_MULTIPLE,
    stop_multiple: float = DEFAULT_STOP_MULTIPLE,
    horizon: int = BARRIER_HORIZON_SESSIONS,
) -> CandidateOutcomes:
    """Evaluate path-dependent Triple-Barrier and fixed-horizon returns for candidate events."""
    if target_multiple <= stop_multiple or stop_multiple <= 0.0 or horizon < 1:
        raise ValueError("require target_multiple > stop_multiple > 0 and horizon >= 1")
    if rows.shape != cols.shape or rows.ndim != 1:
        raise ValueError("rows and cols must be matching 1D index arrays")

    n = int(rows.size)
    n_sessions = len(panel.sessions)
    open_ = panel.open.astype(np.float64)
    high = panel.high.astype(np.float64)
    low = panel.low.astype(np.float64)
    close = panel.close.astype(np.float64)
    spy_open = panel.spy_open.astype(np.float64)
    spy_close = panel.spy_close.astype(np.float64)

    valid = np.zeros(n, dtype=bool)
    y_barrier = np.zeros(n, dtype=np.int8)
    outcome_code = np.zeros(n, dtype=np.int8)
    exit_row = np.full(n, -1, dtype=np.int64)
    holding = np.zeros(n, dtype=np.int64)
    gross = np.full(n, np.nan, dtype=np.float64)
    net20 = np.full(n, np.nan, dtype=np.float64)
    net10 = np.full(n, np.nan, dtype=np.float64)
    net_r = np.full(n, np.nan, dtype=np.float64)
    spy_matched = np.full(n, np.nan, dtype=np.float64)
    excess20 = np.full(n, np.nan, dtype=np.float64)
    excess10 = np.full(n, np.nan, dtype=np.float64)
    fixed_gross = np.full(n, np.nan, dtype=np.float64)
    fixed_net20 = np.full(n, np.nan, dtype=np.float64)
    fixed_spy = np.full(n, np.nan, dtype=np.float64)
    fixed_excess20 = np.full(n, np.nan, dtype=np.float64)
    vol_scaled_excess = np.full(n, np.nan, dtype=np.float64)

    bar_valid = (
        np.isfinite(open_) & (open_ > 0)
        & np.isfinite(high) & (high > 0)
        & np.isfinite(low) & (low > 0)
        & np.isfinite(close) & (close > 0)
        & (high >= np.maximum(open_, close))
        & (low <= np.minimum(open_, close))
    )
    spy_valid = np.isfinite(spy_open) & (spy_open > 0) & np.isfinite(spy_close) & (spy_close > 0)

    for idx in range(n):
        t = int(rows[idx])
        c = int(cols[idx])
        if t < 0 or t + horizon >= n_sessions:
            continue
        vol_t = float(volatility[t, c])
        close_t = float(close[t, c])
        if not (np.isfinite(vol_t) and vol_t > 0.0 and np.isfinite(close_t) and close_t > 0.0):
            continue
        start = t + 1
        stop_idx = t + horizon + 1
        if not (bar_valid[start:stop_idx, c].all() and spy_valid[start:stop_idx].all()):
            continue

        entry = float(open_[start, c])
        spy_entry = float(spy_open[start])
        stop_dist = stop_multiple * vol_t
        target_price = entry + target_multiple * vol_t
        stop_price = entry - stop_dist
        if stop_price <= 0.0:
            continue

        path_high = high[start:stop_idx, c]
        path_low = low[start:stop_idx, c]
        hit_stop = path_low <= stop_price
        hit_target = path_high >= target_price
        touches = np.flatnonzero(hit_stop | hit_target)

        if touches.size > 0:
            offset = int(touches[0])
            ex_row = start + offset
            # Conservative stop_first tie-break when a daily bar spans both barriers.
            if bool(hit_stop[offset]):
                code = OUTCOME_STOP
                yb = np.int8(0)
                exit_price = min(float(open_[ex_row, c]), stop_price)
            else:
                code = OUTCOME_TARGET
                yb = np.int8(1)
                exit_price = target_price
        else:
            offset = horizon - 1
            ex_row = t + horizon
            code = OUTCOME_TIMEOUT
            yb = np.int8(0)
            exit_price = float(close[ex_row, c])

        g_ret = exit_price / entry - 1.0
        n20 = g_ret - FROZEN_ROUND_TRIP_COST
        n10 = g_ret - INFO_ROUND_TRIP_COST
        r20 = (exit_price - entry - FROZEN_ROUND_TRIP_COST * entry) / stop_dist
        spy_m = float(spy_close[ex_row]) / spy_entry - 1.0

        fg = float(close[t + horizon, c]) / entry - 1.0
        fs = float(spy_close[t + horizon]) / spy_entry - 1.0
        vol_pct = vol_t / close_t

        valid[idx] = True
        y_barrier[idx] = yb
        outcome_code[idx] = code
        exit_row[idx] = ex_row
        holding[idx] = offset + 1
        gross[idx] = g_ret
        net20[idx] = n20
        net10[idx] = n10
        net_r[idx] = r20
        spy_matched[idx] = spy_m
        excess20[idx] = n20 - spy_m
        excess10[idx] = n10 - spy_m
        fixed_gross[idx] = fg
        fixed_net20[idx] = fg - FROZEN_ROUND_TRIP_COST
        fixed_spy[idx] = fs
        fixed_excess20[idx] = (fg - FROZEN_ROUND_TRIP_COST) - fs
        vol_scaled_excess[idx] = (fg - fs) / vol_pct

    return CandidateOutcomes(
        valid=valid,
        y_barrier=y_barrier,
        outcome_code=outcome_code,
        exit_row=exit_row,
        holding_sessions=holding,
        gross_return=gross,
        net_return_20bps=net20,
        net_return_10bps=net10,
        net_r_20bps=net_r,
        spy_matched_return=spy_matched,
        net_excess_spy_20bps=excess20,
        net_excess_spy_10bps=excess10,
        fixed_gross_return=fixed_gross,
        fixed_net_20bps=fixed_net20,
        fixed_spy_return=fixed_spy,
        fixed_net_excess_spy_20bps=fixed_excess20,
        vol_scaled_fixed_excess_spy=vol_scaled_excess,
    )
