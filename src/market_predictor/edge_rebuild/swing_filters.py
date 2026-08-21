from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Final

import numpy as np
import pandas as pd

from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.core.errors import DataReadinessError

MANAGED_PATH_SESSION_ORDINAL_COLUMNS: Final = tuple(
    f"managed_path_session_ordinal_d{offset}" for offset in range(1, 11)
)
MANAGED_PATH_NET_RETURN_COLUMNS: Final = tuple(
    f"managed_cumulative_net_return_d{offset}" for offset in range(1, 11)
)
MANAGED_BENCHMARK_RETURN_COLUMNS: Final = (
    "approx_managed_exit_session_close_spy_return",
    "approx_managed_exit_session_close_qqq_return",
    "approx_managed_exit_session_close_sector_return",
)
MANAGED_EXCESS_RETURN_COLUMNS: Final = (
    "approx_managed_exit_session_close_excess_vs_spy",
    "approx_managed_exit_session_close_excess_vs_qqq",
    "approx_managed_exit_session_close_excess_vs_sector",
)
MANAGED_PATH_COST_POLICY: Final = "full_round_trip_cost_applied_from_first_daily_mark"


def apply_sparse_session_gap_abstentions(
    rows: pd.DataFrame,
    *,
    benchmark_bars: pd.DataFrame,
    sparse_missing_sessions_by_ticker: Mapping[str, Sequence[date]],
    contract: StrategyContract,
) -> pd.DataFrame:
    """Invalidate feature and label windows that cross retained sparse gaps."""

    data = rows.copy()
    data["sparse_gap_feature_eligible"] = True
    data["sparse_gap_label_eligible"] = True
    data["sparse_gap_abstention_reason"] = ""
    if not sparse_missing_sessions_by_ticker:
        return data

    required = {
        "ticker",
        "session_date_et",
        "feature_eligible",
        "label_eligible",
    }
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise DataReadinessError(
            f"sparse-gap abstention rows are missing columns: {missing_columns}"
        )
    market_ticker = contract.labels.benchmark_market.upper()
    market_session_values = (
        benchmark_bars.loc[
            benchmark_bars["ticker"].astype(str).str.upper().eq(market_ticker),
            "session_date_et",
        ]
        .drop_duplicates()
        .sort_values()
    )
    market_sessions = pd.to_datetime(
        market_session_values,
        errors="coerce",
    )
    if market_sessions.empty or bool(market_sessions.isna().any()):
        raise DataReadinessError(
            f"sparse-gap abstention requires {market_ticker} sessions"
        )
    ordinal = {
        value: index
        for index, value in enumerate(market_sessions.dt.date.tolist())
    }
    row_sessions = pd.to_datetime(data["session_date_et"], errors="coerce")
    row_ordinal = row_sessions.dt.date.map(ordinal)
    if bool(row_sessions.isna().any() or row_ordinal.isna().any()):
        raise DataReadinessError(
            "sparse-gap abstention found decisions outside benchmark sessions"
        )

    feature_cross = pd.Series(False, index=data.index)
    label_cross = pd.Series(False, index=data.index)
    normalized = {
        str(ticker).strip().upper(): tuple(sorted(set(sessions)))
        for ticker, sessions in sparse_missing_sessions_by_ticker.items()
    }
    for ticker, sessions in normalized.items():
        selected = data["ticker"].astype(str).str.upper().eq(ticker)
        if not bool(selected.any()):
            continue
        unknown = sorted(set(sessions).difference(ordinal))
        if unknown:
            raise DataReadinessError(
                f"sparse-gap sessions are absent from {market_ticker}: "
                f"{ticker} {unknown[:5]}"
            )
        positions = row_ordinal.loc[selected].astype(int)
        for missing_session in sessions:
            gap_ordinal = ordinal[missing_session]
            distance = positions - gap_ordinal
            feature_cross.loc[selected] |= (
                distance.ge(0)
                & distance.lt(contract.swing.minimum_warmup_sessions)
            ).to_numpy()
            label_cross.loc[selected] |= (
                distance.lt(0)
                & distance.ge(-contract.swing.horizon_sessions)
            ).to_numpy()

    data["sparse_gap_feature_eligible"] = ~feature_cross
    data["sparse_gap_label_eligible"] = ~label_cross
    data["sparse_gap_abstention_reason"] = np.select(
        [feature_cross & label_cross, feature_cross, label_cross],
        [
            "feature_and_label_window_crosses_missing_session",
            "feature_warmup_crosses_missing_session",
            "label_window_crosses_missing_session",
        ],
        default="",
    )
    data["feature_eligible"] = (
        data["feature_eligible"].fillna(False).astype(bool) & ~feature_cross
    )
    data["label_eligible"] = (
        data["label_eligible"].fillna(False).astype(bool)
        & ~feature_cross
        & ~label_cross
    )
    label_columns = [
        column
        for column in data.columns
        if column.startswith("future_")
        or column.startswith("target_net_positive_")
        or column.startswith("barrier_")
        or column in {"forward_return", "target_excess_rank"}
    ]
    if label_columns:
        data.loc[label_cross, label_columns] = pd.NA
    return data



def _apply_sector_benchmark_eligibility(
    rows: pd.DataFrame,
    *,
    horizon_sessions: int,
) -> pd.DataFrame:
    required = {
        "feature_eligible",
        "label_eligible",
        "sector_available_at_utc",
        "sector_return_5d",
        "sector_return_20d",
        "sector_return_60d",
        f"future_sector_return_{horizon_sessions}d",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise DataReadinessError(
            f"sector benchmark eligibility inputs are missing: {missing}"
        )
    data = rows.copy()
    feature_ready = data[
        [
            "sector_available_at_utc",
            "sector_return_5d",
            "sector_return_20d",
            "sector_return_60d",
        ]
    ].notna().all(axis=1)
    label_ready = pd.to_numeric(
        data[f"future_sector_return_{horizon_sessions}d"],
        errors="coerce",
    ).notna()
    data["sector_benchmark_feature_eligible"] = feature_ready
    data["sector_benchmark_label_eligible"] = feature_ready & label_ready
    data["sector_benchmark_abstention_reason"] = np.select(
        [~feature_ready, feature_ready & ~label_ready],
        [
            "sector_benchmark_feature_unavailable",
            "sector_benchmark_label_window_unavailable",
        ],
        default="",
    )
    data["feature_eligible"] = (
        data["feature_eligible"].fillna(False).astype(bool) & feature_ready
    )
    data["label_eligible"] = (
        data["label_eligible"].fillna(False).astype(bool)
        & data["sector_benchmark_label_eligible"]
    )
    return data



def _mask_sector_benchmark_ineligible_outcomes(rows: pd.DataFrame) -> pd.DataFrame:
    if "sector_benchmark_label_eligible" not in rows.columns:
        raise DataReadinessError(
            "managed swing outcomes require sector benchmark label eligibility"
        )
    data = rows.copy()
    abstain = ~data["sector_benchmark_label_eligible"].fillna(False).astype(bool)
    outcome_columns = [
        "barrier_label",
        "barrier_exit_session_date_et",
        "barrier_exit_price",
        "barrier_holding_sessions",
        "barrier_target_price",
        "barrier_stop_price",
        "barrier_label_available_at_utc",
        "barrier_gross_return",
        "barrier_cost",
        "barrier_net_return",
        "forward_return",
        *MANAGED_BENCHMARK_RETURN_COLUMNS,
        *MANAGED_EXCESS_RETURN_COLUMNS,
        *MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
        *MANAGED_PATH_NET_RETURN_COLUMNS,
    ]
    missing = sorted(set(outcome_columns).difference(data.columns))
    if missing:
        raise DataReadinessError(
            f"managed swing outcome columns are missing: {missing}"
        )
    data.loc[abstain, outcome_columns] = pd.NA
    if "managed_path_eligible" not in data.columns:
        raise DataReadinessError("managed swing outcome column is missing: managed_path_eligible")
    data.loc[abstain, "managed_path_eligible"] = False
    return data



