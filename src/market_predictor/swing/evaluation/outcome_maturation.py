"""Causal ten-session swing outcome evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from itertools import takewhile

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.label_paths import evaluate_swing_paths
from market_predictor.modeling.label_outcomes import STOP_HIT, TARGET_HIT
from market_predictor.modeling.maturation import (
    MaturationIntent,
    MaturedPath,
    PathEvaluation,
    PendingPath,
    evidence_rows,
    max_available,
    pair_return,
    policy_float,
    policy_int,
    require_policy,
    require_positive_prices,
    timestamp,
)
from market_predictor.swing.labels.barrier_and_rank import (
    BarrierSpec,
    apply_triple_barrier,
)
from market_predictor.swing.labels.holding_paths import holding_calendar, validate_outcome_observations


def evaluate_swing_maturation(
    intent: MaturationIntent,
    bars: pd.DataFrame,
    *,
    proven_stock_gaps: frozenset[date],
) -> PathEvaluation:
    """Evaluate the managed exit and the fixed-horizon return on the stock's usable path.

    The stock's path runs over its consecutive sessions from entry that each hold exactly
    one valid bar. A target or stop reached on that path matures once the path is complete,
    or once its first unusable session is in `proven_stock_gaps`; until then the outcome
    waits, because the missing sessions may still arrive. Benchmarks are checked only on
    the rows the returns use.
    """
    policy = intent.label_policy
    require_policy(policy, "policy", "market_predictor.swing_outcome_policy")
    horizon = policy_int(policy, "horizon_sessions")
    spy_ticker = str(policy["broad_benchmark"]).upper()
    qqq_ticker = str(policy["growth_benchmark"]).upper()
    observed_sessions = (
        bars.loc[bars["ticker"].eq(spy_ticker), "session_date_et"]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )
    if intent.decision_session_et not in observed_sessions:
        return PendingPath(reasons=("decision_session_not_observed",))
    sessions: list[date] = list(holding_calendar(intent.decision_session_et, max(observed_sessions)))
    decision_index = sessions.index(intent.decision_session_et)
    if decision_index + horizon >= len(sessions):
        return PendingPath(reasons=("horizon_not_complete",))
    path_sessions = sessions[decision_index + 1 : decision_index + horizon + 1]
    decision_row = _usable_rows(bars, ticker=intent.ticker, sessions=[intent.decision_session_et]).get(
        intent.decision_session_et
    )
    if decision_row is None:
        return PendingPath(
            reasons=("required_bar_path_incomplete",),
            missing_intervals=(f"{intent.ticker}:{intent.decision_session_et}:decision",),
        )
    usable = _usable_rows(bars, ticker=intent.ticker, sessions=path_sessions)
    prefix_sessions = list(takewhile(usable.__contains__, path_sessions))
    gap_sessions = path_sessions[len(prefix_sessions) :]
    missing_stock = tuple(f"{intent.ticker}:{session}" for session in gap_sessions)
    if not prefix_sessions:
        return PendingPath(reasons=("required_bar_path_incomplete",), missing_intervals=missing_stock)
    stock_path = pd.DataFrame([usable[session] for session in prefix_sessions]).reset_index(drop=True)
    decision_close = float(decision_row["close"])
    barrier_bars = pd.concat(
        [pd.DataFrame([decision_row]), stock_path],
        ignore_index=True,
    ).loc[:, ["session_date_et", "open", "high", "low", "close"]]
    barrier_bars = barrier_bars.rename(columns={"session_date_et": "session"})
    resolved = apply_triple_barrier(
        barrier_bars,
        pd.DataFrame(
            {
                "session": [intent.decision_session_et],
                # The ATR in the price basis of these bars.
                "atr": [intent.decision_atr_fraction * decision_close],
            }
        ),
        spec=BarrierSpec(
            target_atr_multiple=policy_float(policy, "target_atr_multiple"),
            stop_atr_multiple=policy_float(policy, "stop_atr_multiple"),
            horizon_sessions=horizon,
            same_bar_resolution=str(policy["same_bar_barrier_resolution"]),
        ),
    ).iloc[0]
    if pd.isna(resolved["exit_session"]) or pd.isna(resolved["exit_price"]):
        if gap_sessions:
            # No target or stop was reached before the gap, so the outcome needs the missing sessions.
            return PendingPath(reasons=("required_bar_path_incomplete",), missing_intervals=missing_stock)
        return PendingPath(reasons=("managed_path_unresolved",))
    if gap_sessions and gap_sessions[0] not in proven_stock_gaps:
        # The exit is known, but the fixed-horizon return may still become observable.
        return PendingPath(reasons=("required_bar_path_incomplete",), missing_intervals=missing_stock)
    exit_session = pd.Timestamp(resolved["exit_session"]).date()
    holding_sessions = int(resolved["holding_sessions"])
    if holding_sessions < 1 or holding_sessions > len(prefix_sessions):
        raise DataReadinessError("managed swing holding period is invalid")
    realized_path = stock_path.iloc[:holding_sessions].copy()
    entry_session = path_sessions[0]
    benchmark_tickers = (spy_ticker, qqq_ticker, intent.primary_benchmark)
    benchmark_pairs: dict[str, tuple[pd.Series, pd.Series]] = {}
    missing: list[str] = []
    for ticker in benchmark_tickers:
        benchmark_rows = _usable_rows(bars, ticker=ticker, sessions=[entry_session, exit_session])
        if entry_session not in benchmark_rows:
            missing.append(f"{ticker}:{entry_session}:entry")
        if exit_session not in benchmark_rows:
            missing.append(f"{ticker}:{exit_session}:exit")
        if entry_session in benchmark_rows and exit_session in benchmark_rows:
            benchmark_pairs[ticker] = (benchmark_rows[entry_session], benchmark_rows[exit_session])
    if missing:
        return PendingPath(
            reasons=("required_bar_path_incomplete",),
            missing_intervals=tuple(sorted(missing)),
        )
    entry_price = float(stock_path.iloc[0]["open"])
    exit_price = float(resolved["exit_price"])
    require_positive_prices(entry_price, exit_price)
    round_trip_cost_bps = policy_float(policy, "round_trip_cost_bps")
    evaluated = evaluate_swing_paths(
        entry_price=pd.array([entry_price], dtype="float64").to_numpy(),
        exit_price=pd.array([exit_price], dtype="float64").to_numpy(),
        path_high=realized_path["high"].to_numpy(float)[None, :],
        path_low=realized_path["low"].to_numpy(float)[None, :],
        round_trip_cost_bps=round_trip_cost_bps,
    )
    evidence_frames = [pd.DataFrame([decision_row]), realized_path]
    evidence_frames.extend(
        pd.DataFrame([entry, exit_row])
        for entry, exit_row in benchmark_pairs.values()
    )
    fixed_net: float | None = None
    fixed_sector: float | None = None
    if not gap_sessions:
        last_session = path_sessions[-1]
        sector_last = _usable_rows(bars, ticker=intent.primary_benchmark, sessions=[last_session]).get(last_session)
        if sector_last is None:
            return PendingPath(
                reasons=("required_bar_path_incomplete",),
                missing_intervals=(f"{intent.primary_benchmark}:{last_session}:fixed_horizon",),
            )
        fixed = evaluate_swing_paths(
            entry_price=pd.array([entry_price], dtype="float64").to_numpy(),
            exit_price=pd.array([float(stock_path.iloc[-1]["close"])], dtype="float64").to_numpy(),
            path_high=stock_path["high"].to_numpy(float)[None, :],
            path_low=stock_path["low"].to_numpy(float)[None, :],
            round_trip_cost_bps=round_trip_cost_bps,
        )
        fixed_net = float(fixed.net_return[0])
        fixed_sector = pair_return((benchmark_pairs[intent.primary_benchmark][0], sector_last))
        evidence_frames.extend([stock_path.iloc[holding_sessions:], pd.DataFrame([sector_last])])
    rows = evidence_rows(evidence_frames)
    barrier_label = int(resolved["barrier_label"])
    return MaturedPath(
        entry_time=timestamp(stock_path.iloc[0]["bar_start_utc"]),
        exit_time=timestamp(realized_path.iloc[-1]["bar_end_utc"]),
        label_available=max_available(evidence_frames),
        decision_close=decision_close,
        entry_price=entry_price,
        exit_price=exit_price,
        gross_return=float(evaluated.gross_return[0]),
        label_round_trip_cost_bps=round_trip_cost_bps,
        label_net_return=float(evaluated.net_return[0]),
        mfe=float(evaluated.mfe[0]),
        mae=float(evaluated.mae[0]),
        path_outcome=(
            "target_first"
            if barrier_label == TARGET_HIT
            else "stop_first"
            if barrier_label == STOP_HIT
            else "timeout"
        ),
        spy_return=pair_return(benchmark_pairs[spy_ticker]),
        qqq_return=pair_return(benchmark_pairs[qqq_ticker]),
        sector_return=pair_return(benchmark_pairs[intent.primary_benchmark]),
        evidence_rows=rows,
        holding_sessions=holding_sessions,
        fixed_horizon_net_return=fixed_net,
        fixed_horizon_sector_return=fixed_sector,
    )


def _usable_rows(bars: pd.DataFrame, *, ticker: str, sessions: Sequence[date]) -> dict[date, pd.Series]:
    """Each session's bar for the ticker, when the session holds exactly one bar and it is valid."""
    rows = bars.loc[bars["ticker"].eq(ticker) & bars["session_date_et"].isin(list(sessions))]
    if rows.empty:
        return {}
    valid = validate_outcome_observations(rows)["outcome_observation_valid"]
    single = rows.groupby("session_date_et")["ticker"].transform("size").eq(1)
    return {row["session_date_et"]: row for _, row in rows.loc[valid & single].iterrows()}
