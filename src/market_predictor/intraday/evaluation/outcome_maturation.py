"""Causal thirty-minute intraday outcome evaluation."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.label_paths import evaluate_intraday_barrier_paths
from market_predictor.modeling.maturation import (
    MaturationIntent,
    MaturedPath,
    PathEvaluation,
    PendingPath,
    evidence_rows,
    intraday_path,
    max_available,
    one_intraday_row,
    pair_return,
    policy_float,
    policy_int,
    require_policy,
    timestamp,
)


def evaluate_intraday_maturation(
    intent: MaturationIntent,
    bars: pd.DataFrame,
    *,
    observed_at: datetime,
) -> PathEvaluation:
    policy = intent.label_policy
    require_policy(policy, "policy", "intraday_label.v2")
    if intent.decision_atr is None:
        raise DataReadinessError("intraday intent has no decision ATR")
    horizon_minutes = policy_int(policy, "horizon_minutes")
    execution_minutes = policy_int(policy, "execution_bar_minutes")
    horizon_bars = horizon_minutes // execution_minutes
    expected_starts = [
        intent.decision_time_utc + timedelta(minutes=execution_minutes * offset)
        for offset in range(horizon_bars)
    ]
    stock_path, missing = intraday_path(
        bars,
        ticker=intent.ticker,
        expected_starts=expected_starts,
        session=intent.decision_session_et,
        bar_minutes=execution_minutes,
    )
    if missing:
        reason = (
            "horizon_not_complete"
            if expected_starts[-1] >= observed_at
            else "required_bar_path_incomplete"
        )
        return PendingPath(
            reasons=(reason,),
            missing_intervals=tuple(missing),
        )

    entry_price = float(stock_path.iloc[0]["open"])
    round_trip_cost_bps = policy_float(policy, "round_trip_cost_bps")
    evaluated = evaluate_intraday_barrier_paths(
        path_open=stock_path["open"].to_numpy(float)[None, :],
        path_high=stock_path["high"].to_numpy(float)[None, :],
        path_low=stock_path["low"].to_numpy(float)[None, :],
        path_close=stock_path["close"].to_numpy(float)[None, :],
        entry_atr=np.asarray([intent.decision_atr]),
        target_atr=policy_float(policy, "target_atr"),
        stop_atr=policy_float(policy, "stop_atr"),
        round_trip_cost_bps=round_trip_cost_bps,
    )
    outcome_index = int(evaluated.outcome_offset[0])
    active_path = stock_path.iloc[: outcome_index + 1]
    entry_start = expected_starts[0]
    exit_start = expected_starts[outcome_index]
    benchmark_tickers = (
        str(policy["broad_benchmark"]).upper(),
        str(policy["growth_benchmark"]).upper(),
        intent.primary_benchmark,
    )
    benchmark_pairs: dict[str, tuple[pd.Series, pd.Series]] = {}
    benchmark_missing: list[str] = []
    for ticker in benchmark_tickers:
        entry = one_intraday_row(
            bars,
            ticker=ticker,
            start=entry_start,
            bar_minutes=execution_minutes,
        )
        exit_row = one_intraday_row(
            bars,
            ticker=ticker,
            start=exit_start,
            bar_minutes=execution_minutes,
        )
        if entry is None:
            benchmark_missing.append(f"{ticker}:{entry_start.isoformat()}:entry")
        if exit_row is None:
            benchmark_missing.append(f"{ticker}:{exit_start.isoformat()}:exit")
        if entry is not None and exit_row is not None:
            benchmark_pairs[ticker] = (entry, exit_row)
    if benchmark_missing:
        return PendingPath(
            reasons=("required_benchmark_path_incomplete",),
            missing_intervals=tuple(sorted(benchmark_missing)),
        )
    spy_ticker, qqq_ticker, sector_ticker = benchmark_tickers
    evidence_frames = [active_path]
    evidence_frames.extend(
        pd.DataFrame([entry, exit_row])
        for entry, exit_row in benchmark_pairs.values()
    )
    rows = evidence_rows(evidence_frames)
    return MaturedPath(
        entry_time=timestamp(active_path.iloc[0]["bar_start_utc"]),
        exit_time=timestamp(active_path.iloc[-1]["bar_end_utc"]),
        label_available=max_available(evidence_frames),
        entry_price=entry_price,
        exit_price=float(evaluated.realized_price[0]),
        gross_return=float(evaluated.gross_return[0]),
        label_round_trip_cost_bps=round_trip_cost_bps,
        label_net_return=float(evaluated.net_return[0]),
        mfe=float(evaluated.mfe[0]),
        mae=float(evaluated.mae[0]),
        path_outcome=str(evaluated.outcome[0]),
        opportunity_target=int(evaluated.target_first[0]),
        downside_target=int(evaluated.stop_first[0]),
        spy_return=pair_return(benchmark_pairs[spy_ticker]),
        qqq_return=pair_return(benchmark_pairs[qqq_ticker]),
        sector_return=pair_return(benchmark_pairs[sector_ticker]),
        evidence_rows=rows,
    )
