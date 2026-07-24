from __future__ import annotations

import numpy as np
import pandas as pd

from market_predictor.label_paths import evaluate_swing_paths
from market_predictor.swing.contracts import (
    SwingDatasetConfig,
    swing_excess_column,
    swing_net_return_column,
    swing_target_column,
)
from market_predictor.v3.errors import DataReadinessError


def add_exact_swing_labels(
    frame: pd.DataFrame,
    benchmarks: pd.DataFrame,
    config: SwingDatasetConfig,
) -> pd.DataFrame:
    horizon = config.horizon_sessions
    data = frame.sort_values(
        ["ticker", "session_date_et"],
        kind="stable",
    ).copy()
    spy = benchmarks[benchmarks["ticker"].eq(config.broad_benchmark.upper())].sort_values("session_date_et")
    if spy.empty:
        raise DataReadinessError(f"benchmark bars do not contain {config.broad_benchmark}")
    ordered_sessions = list(spy["session_date_et"])
    session_ordinal = {session: index for index, session in enumerate(ordered_sessions)}
    data["_session_ordinal"] = data["session_date_et"].map(session_ordinal)
    if bool(data["_session_ordinal"].isna().any()):
        raise DataReadinessError("equity decisions contain sessions absent from SPY")

    grouped = data.groupby("ticker", sort=False)
    data["entry_time_utc"] = grouped["bar_start_utc"].shift(-1)
    data["exit_time_utc"] = grouped["bar_end_utc"].shift(-horizon)
    data["label_available_at_utc"] = grouped["available_at_utc"].shift(-horizon)
    data["entry_session_date_et"] = grouped["session_date_et"].shift(-1)
    data["exit_session_date_et"] = grouped["session_date_et"].shift(-horizon)
    data["entry_price"] = grouped["open"].shift(-1)
    data["exit_price"] = grouped["close"].shift(-horizon)
    expected_entry = data["_session_ordinal"] + 1
    expected_exit = data["_session_ordinal"] + horizon
    actual_entry = data["entry_session_date_et"].map(session_ordinal)
    actual_exit = data["exit_session_date_et"].map(session_ordinal)
    data["label_window_expected"] = expected_exit.lt(len(ordered_sessions))
    data["label_path_exact"] = actual_entry.eq(expected_entry) & actual_exit.eq(expected_exit)

    future_highs = pd.concat(
        [grouped["high"].shift(-offset) for offset in range(1, horizon + 1)],
        axis=1,
    )
    future_lows = pd.concat(
        [grouped["low"].shift(-offset) for offset in range(1, horizon + 1)],
        axis=1,
    )
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
    data["label_eligible"] = data["feature_eligible"] & data["label_path_exact"] & data[swing_target_column(horizon)].notna()
    return data.drop(columns="_session_ordinal")


def _benchmark_label_return(
    decisions: pd.DataFrame,
    lookup: pd.DataFrame,
    benchmark_tickers: pd.Series,
) -> pd.Series:
    values = np.full(len(decisions), np.nan, dtype=float)
    for position, (ticker, entry_date, exit_date) in enumerate(
        zip(
            benchmark_tickers.astype(str).str.upper(),
            decisions["entry_session_date_et"],
            decisions["exit_session_date_et"],
            strict=True,
        )
    ):
        if pd.isna(entry_date) or pd.isna(exit_date):
            continue
        try:
            entry_open = float(lookup.loc[(ticker, entry_date), "open"])
            exit_close = float(lookup.loc[(ticker, exit_date), "close"])
        except (KeyError, TypeError, ValueError):
            continue
        values[position] = exit_close / entry_open - 1.0
    return pd.Series(values, index=decisions.index, dtype="float64")
