"""Causal ten-session swing outcome evaluation."""

from __future__ import annotations

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.label_paths import evaluate_swing_paths
from market_predictor.modeling.label_outcomes import STOP_HIT, TARGET_HIT
from market_predictor.modeling.maturation import (
    MaturationIntent,
    MaturedPath,
    PathEvaluation,
    PendingPath,
    daily_path,
    evidence_rows,
    max_available,
    one_daily_row,
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
) -> PathEvaluation:
    policy = intent.label_policy
    require_policy(policy, "policy", "market_predictor.swing_outcome_policy.v1")
    if intent.decision_atr is None:
        raise DataReadinessError("swing intent has no decision ATR")
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
    sessions: list[object] = list(holding_calendar(intent.decision_session_et, max(observed_sessions)))
    decision_index = sessions.index(intent.decision_session_et)
    if decision_index + horizon >= len(sessions):
        return PendingPath(reasons=("horizon_not_complete",))
    path_sessions = sessions[decision_index + 1 : decision_index + horizon + 1]
    required_tickers = {intent.ticker, spy_ticker, qqq_ticker, intent.primary_benchmark}
    observations = validate_outcome_observations(bars.loc[
        bars["session_date_et"].isin([intent.decision_session_et, *path_sessions])
        & bars["ticker"].isin(required_tickers)
    ])
    invalid = observations.loc[~observations["outcome_observation_valid"]]
    if not invalid.empty:
        return PendingPath(
            reasons=("required_bar_path_incomplete",),
            missing_intervals=tuple(sorted(f"{row.ticker}:{row.session_date_et}:invalid_observation" for row in invalid.itertuples())),
        )
    entry_session = path_sessions[0]
    stock_path, missing = daily_path(
        bars,
        ticker=intent.ticker,
        sessions=path_sessions,
    )
    decision_row = one_daily_row(
        bars,
        ticker=intent.ticker,
        session=intent.decision_session_et,
    )
    if decision_row is None:
        missing.append(f"{intent.ticker}:{intent.decision_session_et}:decision")
    if missing:
        return PendingPath(
            reasons=("required_bar_path_incomplete",),
            missing_intervals=tuple(sorted(missing)),
        )
    assert decision_row is not None
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
                "atr": [intent.decision_atr],
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
        return PendingPath(reasons=("managed_path_unresolved",))
    exit_session = pd.Timestamp(resolved["exit_session"]).date()
    holding_sessions = int(resolved["holding_sessions"])
    if holding_sessions < 1 or holding_sessions > horizon:
        raise DataReadinessError("managed swing holding period is invalid")
    realized_path = stock_path.iloc[:holding_sessions].copy()
    benchmark_tickers = (spy_ticker, qqq_ticker, intent.primary_benchmark)
    benchmark_pairs: dict[str, tuple[pd.Series, pd.Series]] = {}
    for ticker in benchmark_tickers:
        entry = one_daily_row(bars, ticker=ticker, session=entry_session)
        exit_row = one_daily_row(bars, ticker=ticker, session=exit_session)
        if entry is None:
            missing.append(f"{ticker}:{entry_session}:entry")
        if exit_row is None:
            missing.append(f"{ticker}:{exit_session}:exit")
        if entry is not None and exit_row is not None:
            benchmark_pairs[ticker] = (entry, exit_row)
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
    rows = evidence_rows(evidence_frames)
    barrier_label = int(resolved["barrier_label"])
    return MaturedPath(
        entry_time=timestamp(stock_path.iloc[0]["bar_start_utc"]),
        exit_time=timestamp(realized_path.iloc[-1]["bar_end_utc"]),
        label_available=max_available(evidence_frames),
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
        opportunity_target=None,
        downside_target=None,
        spy_return=pair_return(benchmark_pairs[spy_ticker]),
        qqq_return=pair_return(benchmark_pairs[qqq_ticker]),
        sector_return=pair_return(benchmark_pairs[intent.primary_benchmark]),
        evidence_rows=rows,
    )
