from __future__ import annotations

import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.label_paths import evaluate_swing_paths
from market_predictor.swing.contracts import (
    SwingDatasetConfig,
    swing_excess_column,
    swing_net_return_column,
    swing_target_column,
)
from market_predictor.swing.labels.holding_paths import future_outcome_rows, holding_calendar, outcome_bar_lookup


def add_exact_swing_labels(
    frame: pd.DataFrame,
    benchmarks: pd.DataFrame,
    config: SwingDatasetConfig,
    *,
    outcome_bars: pd.DataFrame,
    inplace: bool = False,
) -> pd.DataFrame:
    if "security_id" not in frame.columns:
        raise DataReadinessError("swing labels require security_id")
    horizon = config.horizon_sessions
    if inplace:
        frame.sort_values(
            ["security_id", "session_date_et"],
            kind="stable",
            inplace=True,
        )
        data = frame
    else:
        data = frame.sort_values(
            ["security_id", "session_date_et"],
            kind="stable",
        ).copy()
    spy = benchmarks[benchmarks["ticker"].eq(config.broad_benchmark.upper())].sort_values("session_date_et")
    if spy.empty:
        raise DataReadinessError(f"benchmark bars do not contain {config.broad_benchmark}")
    ordered_sessions = holding_calendar(min(spy["session_date_et"]), max(spy["session_date_et"]))
    session_ordinal = {session: index for index, session in enumerate(ordered_sessions)}
    decision_ordinals = data["session_date_et"].map(session_ordinal)
    if decision_ordinals.isna().any():
        raise DataReadinessError("equity decisions contain non-exchange sessions")
    lookup = outcome_bar_lookup(outcome_bars)
    paths = list(future_outcome_rows(data, lookup, ordered_sessions, horizon))
    first, last = paths[0], paths[-1]
    data["entry_time_utc"] = first["bar_start_utc"]
    data["exit_time_utc"] = last["bar_end_utc"]
    data["label_available_at_utc"] = pd.concat(
        [path["available_at_utc"] for path in paths], axis=1,
    ).max(axis=1)
    data["entry_session_date_et"] = first["session_date_et"]
    data["exit_session_date_et"] = last["session_date_et"]
    data["entry_price"] = first["open"]
    data["exit_price"] = last["close"]
    # Membership admits a decision; it never cancels an already-open holding.
    data["label_window_expected"] = (decision_ordinals + horizon).lt(len(ordered_sessions))
    data["label_path_exact"] = pd.concat(
        [path["outcome_observation_valid"].eq(True) for path in paths], axis=1,
    ).all(axis=1)
    future_highs = pd.concat([path["high"] for path in paths], axis=1)
    future_lows = pd.concat([path["low"] for path in paths], axis=1)
    evaluated = evaluate_swing_paths(
        entry_price=pd.to_numeric(
            data["entry_price"],
            errors="coerce",
        ).to_numpy(float),
        exit_price=pd.to_numeric(
            data["exit_price"],
            errors="coerce",
        ).to_numpy(float),
        path_high=future_highs.to_numpy(float),
        path_low=future_lows.to_numpy(float),
        round_trip_cost_bps=config.round_trip_cost_bps,
    )
    data[f"future_mfe_{horizon}d"] = evaluated.mfe
    data[f"future_mae_{horizon}d"] = evaluated.mae
    gross = pd.Series(evaluated.gross_return, index=data.index)
    net = pd.Series(evaluated.net_return, index=data.index)
    data[f"future_gross_return_{horizon}d"] = gross
    data[swing_net_return_column(horizon)] = net

    benchmark_lookup = benchmarks.set_index(["ticker", "session_date_et"])
    for benchmark_name, benchmark_ticker in (
        ("spy", config.broad_benchmark.upper()),
        ("qqq", config.growth_benchmark.upper()),
    ):
        benchmark_return = _benchmark_label_return(
            data,
            benchmark_lookup,
            pd.Series(benchmark_ticker, index=data.index),
        )
        data[f"future_{benchmark_name}_return_{horizon}d"] = benchmark_return
        data[swing_excess_column(horizon, benchmark_name)] = net - benchmark_return
    sector_return = _benchmark_label_return(
        data,
        benchmark_lookup,
        data["primary_benchmark"],
    )
    data[f"future_sector_return_{horizon}d"] = sector_return
    data[swing_excess_column(horizon, "sector")] = net - sector_return
    data[swing_target_column(horizon)] = (net > 0).astype("Int64")
    invalid_label = ~data["label_path_exact"] | net.isna() | sector_return.isna()
    label_columns = [
        f"future_gross_return_{horizon}d",
        swing_net_return_column(horizon),
        f"future_spy_return_{horizon}d",
        f"future_qqq_return_{horizon}d",
        f"future_sector_return_{horizon}d",
        swing_excess_column(horizon, "spy"),
        swing_excess_column(horizon, "qqq"),
        swing_excess_column(horizon, "sector"),
        f"future_mfe_{horizon}d",
        f"future_mae_{horizon}d",
    ]
    data.loc[invalid_label, label_columns] = np.nan
    data.loc[invalid_label, swing_target_column(horizon)] = pd.NA
    data["target_excess_rank"] = data.groupby("decision_group_id")[swing_excess_column(horizon, "spy")].rank(method="average", pct=True)
    data["label_eligible"] = (
        data["feature_eligible"]
        & data["label_window_expected"]
        & data["label_path_exact"]
        & data[swing_target_column(horizon)].notna()
    )
    return data


def _benchmark_label_return(
    decisions: pd.DataFrame,
    lookup: pd.DataFrame,
    benchmark_tickers: pd.Series,
) -> pd.Series:
    tickers = benchmark_tickers.astype(str).str.upper()
    entry_index = pd.MultiIndex.from_arrays(
        [tickers, decisions["entry_session_date_et"]],
        names=lookup.index.names,
    )
    exit_index = pd.MultiIndex.from_arrays(
        [tickers, decisions["exit_session_date_et"]],
        names=lookup.index.names,
    )
    entry_open = pd.to_numeric(
        lookup["open"].reindex(entry_index),
        errors="coerce",
    ).to_numpy(dtype=float)
    exit_close = pd.to_numeric(
        lookup["close"].reindex(exit_index),
        errors="coerce",
    ).to_numpy(dtype=float)
    values = np.divide(
        exit_close,
        entry_open,
        out=np.full(len(decisions), np.nan, dtype=float),
        where=np.isfinite(entry_open) & np.isfinite(exit_close) & (entry_open != 0),
    )
    values -= 1.0
    return pd.Series(values, index=decisions.index, dtype="float64")
