from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from market_predictor.execution_policy import executable_fill_prices


@dataclass(frozen=True)
class SwingPathBatch:
    gross_return: np.ndarray
    net_return: np.ndarray
    mfe: np.ndarray
    mae: np.ndarray


@dataclass(frozen=True)
class IntradayBarrierBatch:
    outcome: np.ndarray
    outcome_offset: np.ndarray
    target_first: np.ndarray
    stop_first: np.ndarray
    timeout: np.ndarray
    target_price: np.ndarray
    stop_price: np.ndarray
    realized_price: np.ndarray
    gross_return: np.ndarray
    net_return: np.ndarray
    mfe: np.ndarray
    mae: np.ndarray


def evaluate_swing_paths(
    *,
    entry_price: np.ndarray,
    exit_price: np.ndarray,
    path_high: np.ndarray,
    path_low: np.ndarray,
    round_trip_cost_bps: float,
) -> SwingPathBatch:
    """Evaluate exact next-open to horizon-close swing paths."""

    entries = np.asarray(entry_price, dtype=float)
    exits = np.asarray(exit_price, dtype=float)
    highs = np.asarray(path_high, dtype=float)
    lows = np.asarray(path_low, dtype=float)
    if highs.ndim != 2 or lows.shape != highs.shape:
        raise ValueError("swing path high/low matrices must have equal 2D shape")
    if entries.shape != (highs.shape[0],) or exits.shape != entries.shape:
        raise ValueError("swing entry/exit vectors do not match path rows")
    valid = (
        np.isfinite(entries)
        & (entries > 0)
        & np.isfinite(exits)
        & (exits > 0)
        & np.isfinite(highs).all(axis=1)
        & np.isfinite(lows).all(axis=1)
    )
    gross = np.full(entries.shape, np.nan)
    mfe = np.full(entries.shape, np.nan)
    mae = np.full(entries.shape, np.nan)
    gross[valid] = exits[valid] / entries[valid] - 1.0
    mfe[valid] = highs[valid].max(axis=1) / entries[valid] - 1.0
    mae[valid] = lows[valid].min(axis=1) / entries[valid] - 1.0
    net = gross - float(round_trip_cost_bps) / 10_000.0
    return SwingPathBatch(
        gross_return=gross,
        net_return=net,
        mfe=mfe,
        mae=mae,
    )


def evaluate_intraday_barrier_paths(
    *,
    path_open: np.ndarray,
    path_high: np.ndarray,
    path_low: np.ndarray,
    path_close: np.ndarray,
    entry_atr: np.ndarray,
    target_atr: float,
    stop_atr: float,
    round_trip_cost_bps: float,
) -> IntradayBarrierBatch:
    """Evaluate target/stop/timeout outcomes for exact execution-bar paths."""

    opens = np.asarray(path_open, dtype=float)
    highs = np.asarray(path_high, dtype=float)
    lows = np.asarray(path_low, dtype=float)
    closes = np.asarray(path_close, dtype=float)
    atr = np.asarray(entry_atr, dtype=float)
    if opens.ndim != 2 or highs.shape != opens.shape or lows.shape != opens.shape or closes.shape != opens.shape:
        raise ValueError("intraday OHLC path matrices must have equal 2D shape")
    if atr.shape != (opens.shape[0],):
        raise ValueError("intraday ATR vector does not match path rows")
    if opens.shape[1] < 1:
        raise ValueError("intraday path must contain at least one execution bar")
    entry = opens[:, 0]
    valid = (
        np.isfinite(opens).all(axis=1)
        & np.isfinite(highs).all(axis=1)
        & np.isfinite(lows).all(axis=1)
        & np.isfinite(closes).all(axis=1)
        & np.isfinite(entry)
        & (entry > 0)
        & np.isfinite(atr)
        & (atr > 0)
    )
    if not bool(valid.all()):
        raise ValueError("intraday path contains invalid price or ATR evidence")

    target_price = entry + float(target_atr) * atr
    stop_price = entry - float(stop_atr) * atr
    if bool((stop_price <= 0).any()):
        raise ValueError("intraday stop price must remain positive")
    hit_target = highs >= target_price[:, None]
    hit_stop = lows <= stop_price[:, None]
    missing = opens.shape[1] + 1
    first_target = np.where(
        hit_target.any(axis=1),
        hit_target.argmax(axis=1),
        missing,
    )
    first_stop = np.where(
        hit_stop.any(axis=1),
        hit_stop.argmax(axis=1),
        missing,
    )
    target_first = first_target < first_stop
    # A same-bar target/stop collision is conservatively stop-first.
    stop_first = (first_stop <= first_target) & (first_stop < missing)
    timeout = ~(target_first | stop_first)
    outcome_offset = np.minimum(
        np.minimum(first_target, first_stop),
        opens.shape[1] - 1,
    )
    outcome = np.full(len(entry), "timeout", dtype=object)
    outcome[target_first] = "target_first"
    outcome[stop_first] = "stop_first"
    row = np.arange(len(entry))
    realized = executable_fill_prices(
        outcome=outcome,
        target_price=target_price,
        stop_price=stop_price,
        trigger_open=opens[row, outcome_offset],
        final_price=closes[:, -1],
    )
    active = np.arange(opens.shape[1])[None, :] <= outcome_offset[:, None]
    mfe_high = np.where(active, highs, -np.inf).max(axis=1)
    mae_low = np.where(active, lows, np.inf).min(axis=1)
    gross = realized / entry - 1.0
    net = gross - float(round_trip_cost_bps) / 10_000.0
    return IntradayBarrierBatch(
        outcome=outcome,
        outcome_offset=outcome_offset,
        target_first=target_first,
        stop_first=stop_first,
        timeout=timeout,
        target_price=target_price,
        stop_price=stop_price,
        realized_price=realized,
        gross_return=gross,
        net_return=net,
        mfe=mfe_high / entry - 1.0,
        mae=mae_low / entry - 1.0,
    )


def open_close_return(
    entry_price: np.ndarray,
    exit_price: np.ndarray,
) -> np.ndarray:
    entries = np.asarray(entry_price, dtype=float)
    exits = np.asarray(exit_price, dtype=float)
    if entries.shape != exits.shape:
        raise ValueError("benchmark entry/exit vectors must have equal shape")
    output = np.full(entries.shape, np.nan)
    valid = np.isfinite(entries) & (entries > 0) & np.isfinite(exits) & (exits > 0)
    output[valid] = exits[valid] / entries[valid] - 1.0
    return output
