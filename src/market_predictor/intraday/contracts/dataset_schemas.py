"""Atomic, lineage-bound publisher for the causal intraday training dataset."""
from __future__ import annotations



import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd
import pyarrow as pa

from market_predictor.edge_rebuild.strategy_contract import (
    StrategyContract,
)

INTRADAY_DATASET_SCHEMA: Final = "edge_rebuild.intraday_dataset.v2"

INTRADAY_DATASET_AUTHORITY_SCHEMA: Final = "edge_rebuild.intraday_dataset_authority.v2"

MEMORY_HARD_BUDGET_GIB: Final = 4.0

MEMORY_HEADROOM_GIB: Final = 0.75

MAXIMUM_SECURITY_EXCLUSION_FRACTION: Final = 0.05

MAX_SESSION_WORKERS: Final = 4

WORKING_SET_RELEASE_INTERVAL_SESSIONS: Final = 25

_SAFE_TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,31}$")

_REQUIRED_BENCHMARKS: Final = frozenset(
    {
        "SPY",
        "QQQ",
        "XLB",
        "XLC",
        "XLE",
        "XLF",
        "XLI",
        "XLK",
        "XLP",
        "XLRE",
        "XLU",
        "XLV",
        "XLY",
    }
)

_LABEL_COLUMNS: Final = frozenset(
    {
        "label_schema_version",
        "label_eligible",
        "label_ineligible_reason",
        "decision_group_id",
        "entry_time_utc",
        "entry_bar_end_utc",
        "entry_price",
        "target_price",
        "stop_price",
        "exit_time_utc",
        "exit_bar_end_utc",
        "label_available_at_utc",
        "exit_price",
        "holding_minutes",
        "barrier_label",
        "label_outcome",
        "label_outcome_reason",
        "target_hit",
        "stop_hit",
        "timeout",
        "gross_return",
        "cost",
        "net_return",
        "spy_return",
        "qqq_return",
        "sector_return",
        "spy_excess_return",
        "qqq_excess_return",
        "sector_excess_return",
        "rank_label",
        "rank_percentile",
        "ranking_group_size",
    }
)

_ABSTENTION_COLUMNS: Final = (
    "dataset_row_id",
    "ticker",
    "session_date_et",
    "volume_bar_number",
    "feature_available_at_utc",
    "stage",
    "reason",
)

_PAIR_AUDIT_COLUMNS: Final = (
    "ticker",
    "session_date_et",
    "status",
    "reason",
    "source_rows",
    "completed_volume_bars",
    "feature_rows",
    "feature_eligible_rows",
    "label_eligible_rows",
    "dataset_eligible_rows",
    "abstention_rows",
)

_PAIR_AUDIT_SCHEMA: Final = pa.schema(
    [
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("session_date_et", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("reason", pa.string()),
        pa.field("source_rows", pa.int64(), nullable=False),
        pa.field("completed_volume_bars", pa.int64(), nullable=False),
        pa.field("feature_rows", pa.int64(), nullable=False),
        pa.field("feature_eligible_rows", pa.int64(), nullable=False),
        pa.field("label_eligible_rows", pa.int64(), nullable=False),
        pa.field("dataset_eligible_rows", pa.int64(), nullable=False),
        pa.field("abstention_rows", pa.int64(), nullable=False),
    ]
)

_ABSTENTION_SCHEMA: Final = pa.schema(
    [
        pa.field("dataset_row_id", pa.string()),
        pa.field("ticker", pa.string(), nullable=False),
        pa.field("session_date_et", pa.string(), nullable=False),
        pa.field("volume_bar_number", pa.int64()),
        pa.field("feature_available_at_utc", pa.timestamp("ns", tz="UTC")),
        pa.field("stage", pa.string(), nullable=False),
        pa.field("reason", pa.string(), nullable=False),
    ]
)

@dataclass(frozen=True, slots=True)
class _Artifact:
    path: Path
    session_date_et: str
    symbol_rows: dict[str, int]
    sha256: str

@dataclass(frozen=True, slots=True)
class _VerifiedInputs:
    selection: pd.DataFrame
    coverage: pd.DataFrame
    excluded_tickers: frozenset[str]
    membership_sector_excluded_tickers: frozenset[str]
    incomplete_pairs: frozenset[tuple[str, str]]
    memberships: pd.DataFrame
    stock_artifacts: tuple[_Artifact, ...]
    benchmark_artifacts: tuple[_Artifact, ...]
    benchmark_tickers: frozenset[str]
    parent_lineage: dict[str, str]
    contract: StrategyContract
    contract_sha256: str

@dataclass(frozen=True, slots=True)
class _SessionResult:
    rows: pd.DataFrame | None
    pair_audits: tuple[dict[str, Any], ...]
    abstentions: tuple[dict[str, Any], ...]