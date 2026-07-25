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
    *,
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
    ordered_sessions = list(spy["session_date_et"])
    session_ordinal = {session: index for index, session in enumerate(ordered_sessions)}
    data["_session_ordinal"] = data["session_date_et"].map(session_ordinal)
    if bool(data["_session_ordinal"].isna().any()):
        raise DataReadinessError("equity decisions contain sessions absent from SPY")

    grouped = data.groupby("security_id", sort=False)
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
    market_window_expected = expected_exit.lt(len(ordered_sessions))
    membership_window_expected = pd.Series(True, index=data.index)
    if "membership_effective_to_utc" in data.columns:
        membership_end = pd.to_datetime(
            data["membership_effective_to_utc"],
            utc=True,
            errors="coerce",
        )
        membership_end_date = membership_end.dt.tz_convert(
            "America/New_York"
        ).dt.date
        expected_exit_date = expected_exit.map(
            lambda ordinal: (
                ordered_sessions[int(ordinal)]
                if pd.notna(ordinal)
                and int(ordinal) >= 0
                and int(ordinal) < len(ordered_sessions)
                else pd.NaT
            )
        )
        membership_window_expected = (
            membership_end_date.isna()
            | (
                pd.Series(expected_exit_date, index=data.index)
                < membership_end_date
            )
        )
    data["label_window_expected"] = (
        market_window_expected & membership_window_expected
    )
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
    data["label_eligible"] = (
        data["feature_eligible"]
        & data["label_window_expected"]
        & data["label_path_exact"]
        & data[swing_target_column(horizon)].notna()
    )
    return data.drop(columns="_session_ordinal")


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
