from __future__ import annotations

from typing import Final

from market_predictor.edge_rebuild.intraday_bar_features import (
    INTRADAY_BAR_MODEL_FEATURE_COLUMNS,
)

MODEL_FEATURE_COLUMNS: Final = INTRADAY_BAR_MODEL_FEATURE_COLUMNS

IDENTITY_COLUMNS: Final = (
    "decision_id",
    "decision_cohort_id",
    "decision_group_id",
    "ticker",
    "security_id",
    "session_date_et",
    "sector",
    "primary_benchmark",
    "universe_snapshot_id",
    "strategy_contract_sha256",
)

TIMESTAMP_COLUMNS: Final = (
    "decision_time_utc",
    "feature_available_at_utc",
    "entry_time_utc",
    "entry_bar_end_utc",
    "exit_time_utc",
    "exit_bar_end_utc",
    "label_available_at_utc",
)

BOOLEAN_COLUMNS: Final = (
    "dataset_eligible",
    "feature_eligible",
    "label_eligible",
    "target_hit",
    "stop_hit",
)

RETURN_COLUMNS: Final = (
    "gross_return",
    "cost",
    "net_return",
    "spy_return",
    "qqq_return",
    "sector_return",
    "spy_excess_return",
    "qqq_excess_return",
    "sector_excess_return",
)

PRICE_COLUMNS: Final = (
    "entry_price",
    "stop_price",
)

ROW_CONTRACT_COLUMNS: Final = (
    "feature_schema_version",
    "label_schema_version",
    "ordered_feature_sha256",
)

PROJECTED_COLUMNS: Final = tuple(
    dict.fromkeys(
        (
            *IDENTITY_COLUMNS,
            *TIMESTAMP_COLUMNS,
            *BOOLEAN_COLUMNS,
            *RETURN_COLUMNS,
            *PRICE_COLUMNS,
            *ROW_CONTRACT_COLUMNS,
            *MODEL_FEATURE_COLUMNS,
        )
    )
)
