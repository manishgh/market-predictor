"""Atomic, lineage-bound publisher for the causal intraday training dataset."""

from __future__ import annotations

import json
import re
import shutil
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, date, datetime
from itertools import islice
from pathlib import Path
from typing import Any, Final, Protocol, cast

import exchange_calendars as xcals
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
)
from market_predictor.edge_rebuild.history_collection import (
    load_complete_intraday_history_collection,
)
from market_predictor.edge_rebuild.history_contracts import (
    SELECTED_SESSION_BENCHMARK_PLAN_SCHEMA,
    SELECTED_SESSION_ONE_MINUTE_PLAN_SCHEMA,
)
from market_predictor.edge_rebuild.intraday_features import (
    FEATURE_SCHEMA_VERSION,
    build_causal_intraday_features,
)
from market_predictor.edge_rebuild.intraday_history import (
    json_sha256,
    load_complete_intraday_history_plan,
    load_plan_json,
)
from market_predictor.edge_rebuild.intraday_labels import (
    LABEL_SCHEMA_VERSION,
    _add_contemporaneous_rank,
    _empty_label_columns,
    build_exact_causal_intraday_labels,
)
from market_predictor.edge_rebuild.intraday_selection import (
    INTRADAY_SELECTION_SCHEMA,
    _load_sp500_membership_eligibility,
    load_complete_intraday_selection,
)
from market_predictor.edge_rebuild.one_minute_coverage import (
    load_complete_one_minute_coverage,
    verify_canonical_five_minute_store,
)
from market_predictor.edge_rebuild.selected_session_history import (
    verify_selected_stock_sessions,
)
from market_predictor.edge_rebuild.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)
from market_predictor.edge_rebuild.volume_bars import build_causal_volume_bars
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.v3.errors import DataReadinessError

INTRADAY_DATASET_SCHEMA: Final = "edge_rebuild.intraday_dataset.v2"
INTRADAY_DATASET_AUTHORITY_SCHEMA: Final = "edge_rebuild.intraday_dataset_authority.v2"
MEMORY_HARD_BUDGET_GIB: Final = 4.0
MEMORY_HEADROOM_GIB: Final = 0.75
MAXIMUM_SECURITY_EXCLUSION_FRACTION: Final = 0.05
MAX_SESSION_WORKERS: Final = 4
WORKING_SET_RELEASE_INTERVAL_SESSIONS: Final = 25
_SAFE_TICKER = re.compile(r"^[A-Z0-9][A-Z0-9.-]{0,31}$")


class _ParquetWriterProtocol(Protocol):
    def write_table(
        self, table: pa.Table, row_group_size: int | None = None
    ) -> None: ...

    def close(self) -> None: ...


class _ParquetWriterFactory(Protocol):
    def __call__(
        self,
        where: Path,
        schema: pa.Schema,
        *,
        compression: str,
        use_dictionary: bool,
        write_statistics: bool,
    ) -> _ParquetWriterProtocol: ...


class _ParquetMetadataProtocol(Protocol):
    num_row_groups: int


class _ParquetFileProtocol(Protocol):
    schema_arrow: pa.Schema
    metadata: _ParquetMetadataProtocol

    def read_row_group(
        self, index: int, columns: list[str] | None = None
    ) -> pa.Table: ...


class _ParquetFileFactory(Protocol):
    def __call__(self, source: Path) -> _ParquetFileProtocol: ...


class _WriteTable(Protocol):
    def __call__(self, table: pa.Table, where: Path) -> None: ...


_PARQUET_WRITER = cast(_ParquetWriterFactory, pq.ParquetWriter)
_PARQUET_FILE = cast(_ParquetFileFactory, pq.ParquetFile)
_WRITE_TABLE = cast(_WriteTable, pq.write_table)
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
        "sector_return",
        "spy_excess_return",
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


class _MonthlyPartitionWriter:
    """Write ordered exchange sessions as row groups in one file per month."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._month: str | None = None
        self._path: Path | None = None
        self._writer: _ParquetWriterProtocol | None = None
        self._canonical_schema: pa.Schema | None = None
        self._last_written_session: date | None = None
        self._rows = 0
        self._eligible_rows = 0
        self._stock_sessions = 0
        self._tickers: set[str] = set()
        self._first_session: str | None = None
        self._last_session: str | None = None
        self._first_feature: pd.Timestamp | None = None
        self._last_feature: pd.Timestamp | None = None
        self._last_label: pd.Timestamp | None = None

    def write(self, frame: pd.DataFrame) -> dict[str, Any] | None:
        if frame.empty:
            raise DataReadinessError("monthly partition writer received no rows")
        sessions = sorted(set(frame["session_date_et"].astype(str)))
        if len(sessions) != 1:
            raise DataReadinessError(
                "monthly partition writer requires exactly one exchange session"
            )
        session = sessions[0]
        parsed_session = date.fromisoformat(session)
        if (
            self._last_written_session is not None
            and parsed_session <= self._last_written_session
        ):
            raise DataReadinessError(
                "intraday exchange sessions must be written in strictly increasing order"
            )
        month = parsed_session.strftime("%Y-%m")
        completed = None
        if self._month is not None and month != self._month:
            if month <= self._month:
                raise DataReadinessError(
                    "monthly intraday partitions must be written chronologically"
                )
            completed = self.close()
        if self._month is None:
            self._open(month)

        ordered = frame.sort_values(
            ["session_date_et", "ticker", "volume_bar_number"],
            kind="stable",
        ).reset_index(drop=True)
        table = pa.Table.from_pandas(
            ordered, preserve_index=False
        ).replace_schema_metadata(None)
        if self._canonical_schema is None:
            self._canonical_schema = table.schema
        elif not table.schema.equals(self._canonical_schema):
            raise DataReadinessError(
                "intraday monthly partition schema changed across the publication"
            )
        if self._writer is None:
            self._initialize()
        if self._canonical_schema is None or self._writer is None:
            raise RuntimeError("monthly partition writer was not initialized")
        self._writer.write_table(table, row_group_size=len(table))
        self._rows += len(ordered)
        self._eligible_rows += int(ordered["dataset_eligible"].sum())
        self._stock_sessions += int(
            len(ordered.loc[:, ["session_date_et", "ticker"]].drop_duplicates())
        )
        self._tickers.update(ordered["ticker"].astype(str))
        self._first_session = self._first_session or session
        self._last_session = session
        self._first_feature = _earliest_timestamp(
            self._first_feature,
            ordered["feature_available_at_utc"].min(),
        )
        self._last_feature = _latest_timestamp(
            self._last_feature,
            ordered["feature_available_at_utc"].max(),
        )
        self._last_label = _latest_timestamp(
            self._last_label,
            ordered["label_available_at_utc"].max(),
        )
        self._last_written_session = parsed_session
        return completed

    def close(self) -> dict[str, Any] | None:
        if self._writer is None:
            return None
        writer = self._writer
        self._writer = None
        try:
            writer.close()
            if self._path is None or self._month is None:
                raise RuntimeError("monthly partition writer lost its path identity")
            return {
                **_file_record(self._path, self._root, rows=self._rows),
                "session_month_et": self._month,
                "first_session_date_et": self._first_session,
                "last_session_date_et": self._last_session,
                "stock_sessions": self._stock_sessions,
                "ticker_count": len(self._tickers),
                "eligible_rows": self._eligible_rows,
                "first_feature_available_at_utc": _iso(self._first_feature),
                "last_feature_available_at_utc": _iso(self._last_feature),
                "last_label_available_at_utc": _iso(self._last_label),
            }
        finally:
            self._reset_month()

    def abort(self) -> None:
        writer = self._writer
        self._writer = None
        try:
            if writer is not None:
                writer.close()
        except Exception:
            pass
        finally:
            self._reset_month()
            self._canonical_schema = None
            self._last_written_session = None

    def _open(self, month: str) -> None:
        path = (
            self._root
            / "partitions"
            / f"session_month_et={month}"
            / "part-00000.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        self._month = month
        self._path = path
        self._writer = None

    def _initialize(self) -> None:
        if self._path is None or self._canonical_schema is None:
            raise RuntimeError("monthly partition path is unavailable")
        self._writer = _PARQUET_WRITER(
            self._path,
            self._canonical_schema,
            compression="zstd",
            use_dictionary=True,
            write_statistics=True,
        )

    def _reset_month(self) -> None:
        self._month = None
        self._path = None
        self._writer = None
        self._rows = 0
        self._eligible_rows = 0
        self._stock_sessions = 0
        self._tickers.clear()
        self._first_session = None
        self._last_session = None
        self._first_feature = None
        self._last_feature = None
        self._last_label = None


class _StreamingAuditWriter:
    """Flush audit rows incrementally so publication memory is session-bounded."""

    def __init__(self, root: Path) -> None:
        self._root = root
        self._directory = root / "audit"
        self._directory.mkdir(parents=True, exist_ok=True)
        self._pair_path = self._directory / "stock_session_audit.parquet"
        self._abstention_path = self._directory / "abstentions.parquet"
        self._pair_writer: _ParquetWriterProtocol | None = None
        self._abstention_writer: _ParquetWriterProtocol | None = None
        self.pair_rows = 0
        self.abstention_rows = 0

    def write(
        self,
        pair_audits: Sequence[Mapping[str, Any]],
        abstentions: Sequence[Mapping[str, Any]],
    ) -> None:
        if pair_audits:
            records = sorted(
                pair_audits,
                key=lambda row: (str(row["session_date_et"]), str(row["ticker"])),
            )
            table = pa.Table.from_pylist(
                _normalize_arrow_records(records, _PAIR_AUDIT_SCHEMA),
                schema=_PAIR_AUDIT_SCHEMA,
            )
            if self._pair_writer is None:
                self._pair_writer = _parquet_writer(self._pair_path, _PAIR_AUDIT_SCHEMA)
            self._pair_writer.write_table(table, row_group_size=len(table))
            self.pair_rows += len(table)
        if abstentions:
            records = sorted(
                abstentions,
                key=lambda row: (
                    str(row["session_date_et"]),
                    str(row["ticker"]),
                    -1
                    if _is_missing(row.get("volume_bar_number"))
                    else int(cast(int, row["volume_bar_number"])),
                ),
            )
            table = pa.Table.from_pylist(
                _normalize_arrow_records(records, _ABSTENTION_SCHEMA),
                schema=_ABSTENTION_SCHEMA,
            )
            if self._abstention_writer is None:
                self._abstention_writer = _parquet_writer(
                    self._abstention_path, _ABSTENTION_SCHEMA
                )
            self._abstention_writer.write_table(table, row_group_size=len(table))
            self.abstention_rows += len(table)

    def close(self) -> list[dict[str, Any]]:
        self._close_or_create_empty(
            "_pair_writer", self._pair_path, _PAIR_AUDIT_SCHEMA
        )
        self._close_or_create_empty(
            "_abstention_writer", self._abstention_path, _ABSTENTION_SCHEMA
        )
        return [
            _file_record(self._pair_path, self._root, rows=self.pair_rows),
            _file_record(
                self._abstention_path, self._root, rows=self.abstention_rows
            ),
        ]

    def abort(self) -> None:
        for attribute in ("_pair_writer", "_abstention_writer"):
            writer = cast(_ParquetWriterProtocol | None, getattr(self, attribute))
            setattr(self, attribute, None)
            try:
                if writer is not None:
                    writer.close()
            except Exception:
                pass

    def _close_or_create_empty(
        self,
        attribute: str,
        path: Path,
        schema: pa.Schema,
    ) -> None:
        writer = cast(_ParquetWriterProtocol | None, getattr(self, attribute))
        setattr(self, attribute, None)
        if writer is None:
            _WRITE_TABLE(pa.Table.from_pylist([], schema=schema), path)
            return
        writer.close()


def _parquet_writer(path: Path, schema: pa.Schema) -> _ParquetWriterProtocol:
    return _PARQUET_WRITER(
        path,
        schema,
        compression="zstd",
        use_dictionary=True,
        write_statistics=True,
    )


def _normalize_arrow_records(
    records: Sequence[Mapping[str, Any]],
    schema: pa.Schema,
) -> list[dict[str, Any]]:
    return [
        {
            field.name: _normalize_arrow_value(record.get(field.name), field.type)
            for field in schema
        }
        for record in records
    ]


def _normalize_arrow_value(value: object, data_type: pa.DataType) -> object:
    if _is_missing(value):
        return None
    if pa.types.is_string(data_type):
        return str(value)
    if pa.types.is_integer(data_type):
        return int(cast(int, value))
    if pa.types.is_timestamp(data_type):
        return pd.Timestamp(value)
    return value


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    missing = pd.isna(value)
    try:
        return bool(missing)
    except ValueError:
        return False


def publish_intraday_dataset(
    *,
    selection_directory: Path,
    stock_collection_directory: Path,
    stock_coverage_directory: Path,
    benchmark_collection_directory: Path,
    membership_authority_directory: Path,
    strategy_contract: StrategyContract,
    strategy_contract_path: Path,
    output_directory: Path,
    session_workers: int = MAX_SESSION_WORKERS,
) -> dict[str, Any]:
    """Publish verified feature/label partitions without loading the corpus.

    Stock data is read one selected stock-session at a time. Benchmark data and
    reduced labeled rows are retained only for the current exchange session so
    cross-sectional ranks remain exact while memory stays bounded.
    """

    if session_workers < 1 or session_workers > MAX_SESSION_WORKERS:
        raise ValueError(
            f"session_workers must be between 1 and {MAX_SESSION_WORKERS}"
        )
    verified = _verify_inputs(
        selection_directory=selection_directory,
        stock_collection_directory=stock_collection_directory,
        stock_coverage_directory=stock_coverage_directory,
        benchmark_collection_directory=benchmark_collection_directory,
        membership_authority_directory=membership_authority_directory,
        strategy_contract=strategy_contract,
        strategy_contract_path=strategy_contract_path,
    )
    usable_selection = verified.selection[
        ~verified.selection["ticker"].isin(verified.excluded_tickers)
    ].copy()
    expected_selected_by_month = _monthly_stock_session_counts(verified.selection)
    expected_usable_by_month = _monthly_stock_session_counts(usable_selection)
    request = {
        "schema": INTRADAY_DATASET_SCHEMA,
        "selection_directory": str(selection_directory.resolve()),
        "stock_collection_directory": str(stock_collection_directory.resolve()),
        "stock_coverage_directory": str(stock_coverage_directory.resolve()),
        "benchmark_collection_directory": str(benchmark_collection_directory.resolve()),
        "membership_authority_directory": str(membership_authority_directory.resolve()),
        "strategy_contract_path": str(strategy_contract_path.resolve()),
        "strategy_contract_sha256": verified.contract_sha256,
        "parent_lineage": verified.parent_lineage,
        "parent_lineage_sha256": json_sha256(verified.parent_lineage),
        "membership_sector_excluded_tickers": sorted(
            verified.membership_sector_excluded_tickers
        ),
        "all_excluded_tickers": sorted(verified.excluded_tickers),
        "security_exclusion_fraction": (
            len(verified.excluded_tickers)
            / int(verified.selection["ticker"].nunique())
        ),
        "expected_selected_stock_sessions_by_month": expected_selected_by_month,
        "expected_usable_stock_sessions_by_month": expected_usable_by_month,
        "partitioning": ["session_month_et"],
        "partition_layout": "one_parquet_file_per_calendar_month",
        "partition_row_group": "one_completed_exchange_session",
        "processing_unit": "one_exchange_session",
        "ranking_unit": "one_exchange_session",
        "session_workers": session_workers,
        "working_set_release_interval_sessions": WORKING_SET_RELEASE_INTERVAL_SESSIONS,
        "memory_hard_budget_gib": MEMORY_HARD_BUDGET_GIB,
    }
    request_sha256 = json_sha256(request)
    if output_directory.exists():
        existing_manifest = load_complete_intraday_dataset(output_directory)
        if existing_manifest.get("request_sha256") != request_sha256:
            raise DataReadinessError(f"published intraday dataset is immutable: {output_directory}")
        return existing_manifest

    _guard_memory("intraday dataset publication start")
    staging = output_directory.with_name(f".{output_directory.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    partition_writer = _MonthlyPartitionWriter(staging)
    audit_writer = _StreamingAuditWriter(staging)
    try:
        _write_json(staging / "_request.json", {**request, "request_sha256": request_sha256})
        partition_records: list[dict[str, Any]] = []
        stock_index = _stock_artifact_index(verified.stock_artifacts)
        benchmark_index = _benchmark_artifact_index(verified.benchmark_artifacts)
        usable = usable_selection
        initial_pair_audits: list[dict[str, Any]] = []
        initial_abstentions: list[dict[str, Any]] = []
        _record_excluded_pairs(
            verified.selection,
            verified.excluded_tickers.difference(
                verified.membership_sector_excluded_tickers
            ),
            pair_audits=initial_pair_audits,
            abstentions=initial_abstentions,
            stage="coverage",
            reason="whole_security_coverage_exclusion",
        )
        _record_excluded_pairs(
            verified.selection,
            verified.membership_sector_excluded_tickers,
            pair_audits=initial_pair_audits,
            abstentions=initial_abstentions,
            stage="membership",
            reason="whole_security_invalid_sector_benchmark_exclusion",
        )
        if usable.empty:
            raise DataReadinessError("coverage excluded every selected security")
        audit_writer.write(initial_pair_audits, initial_abstentions)
        del initial_pair_audits, initial_abstentions

        with ThreadPoolExecutor(
            max_workers=session_workers,
            thread_name_prefix="intraday-session",
        ) as executor:
            session_groups = iter(
                usable.groupby("session_date_et", sort=True, observed=True)
            )
            completed_sessions = 0
            while True:
                batch = list(islice(session_groups, session_workers))
                if not batch:
                    break
                futures = [
                    executor.submit(
                        _publish_session,
                        session_date=str(session_date),
                        session_selection=session_selection,
                        verified=verified,
                        stock_index=stock_index,
                        benchmark_index=benchmark_index,
                        request_sha256=request_sha256,
                        parent_lineage_sha256=str(
                            request["parent_lineage_sha256"]
                        ),
                    )
                    for session_date, session_selection in batch
                ]
                for (session_date, _), future in zip(batch, futures, strict=True):
                    session_result = future.result()
                    if session_result.rows is not None:
                        completed_partition = partition_writer.write(
                            session_result.rows
                        )
                        if completed_partition is not None:
                            partition_records.append(completed_partition)
                    audit_writer.write(
                        session_result.pair_audits,
                        session_result.abstentions,
                    )
                    completed_sessions += 1
                    if (
                        completed_sessions
                        % WORKING_SET_RELEASE_INTERVAL_SESSIONS
                        == 0
                    ):
                        release_process_memory()
                    _guard_memory(
                        f"intraday dataset session {str(session_date)} complete"
                    )
        final_partition = partition_writer.close()
        if final_partition is not None:
            partition_records.append(final_partition)
        release_process_memory()

        if not partition_records:
            raise DataReadinessError("intraday dataset produced no feature-label partitions")
        _validate_monthly_partition_records(
            partition_records,
            expected_stock_sessions_by_month=expected_usable_by_month,
        )
        if audit_writer.pair_rows != len(verified.selection):
            raise DataReadinessError(
                "stock-session audit does not reconcile to the causal selection"
            )
        audit_files = audit_writer.close()
        request_record = _file_record(staging / "_request.json", staging, rows=1)
        files = sorted(
            [*partition_records, *audit_files, request_record],
            key=lambda item: str(item["path"]),
        )
        total_rows = sum(int(record["rows"]) for record in partition_records)
        eligible_rows = sum(int(record["eligible_rows"]) for record in partition_records)
        manifest: dict[str, Any] = {
            "schema": INTRADAY_DATASET_SCHEMA,
            "status": "complete",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "request_sha256": request_sha256,
            "strategy_contract_sha256": verified.contract_sha256,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "parent_lineage": verified.parent_lineage,
            "parent_lineage_sha256": request["parent_lineage_sha256"],
            "partitioning": request["partitioning"],
            "partition_layout": request["partition_layout"],
            "partition_row_group": request["partition_row_group"],
            "partitions": partition_records,
            "files": files,
            "summary": {
                "selected_stock_sessions": int(len(verified.selection)),
                "excluded_stock_sessions": int(verified.selection["ticker"].isin(verified.excluded_tickers).sum()),
                "membership_sector_excluded_securities": len(
                    verified.membership_sector_excluded_tickers
                ),
                "incomplete_stock_sessions": len(verified.incomplete_pairs),
                "published_stock_sessions": sum(
                    int(record["stock_sessions"])
                    for record in partition_records
                ),
                "partition_files": len(partition_records),
                "rows": total_rows,
                "dataset_eligible_rows": eligible_rows,
                "abstention_rows": audit_writer.abstention_rows,
                "memory": memory_audit(
                    hard_budget_gib=MEMORY_HARD_BUDGET_GIB,
                    headroom_gib=MEMORY_HEADROOM_GIB,
                ).to_record(),
            },
            "training_contract": {
                "eligibility_column": "dataset_eligible",
                "feature_columns_exclude": sorted(
                    _LABEL_COLUMNS
                    | {
                        "dataset_eligible",
                        "dataset_ineligible_reason",
                        "dataset_row_id",
                        "dataset_request_sha256",
                        "parent_lineage_sha256",
                    }
                ),
                "label_columns": sorted(_LABEL_COLUMNS),
            },
        }
        _write_json(staging / "_manifest.json", manifest)
        _write_json(
            staging / "_authority.json",
            {
                "schema": INTRADAY_DATASET_AUTHORITY_SCHEMA,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(staging / "_manifest.json"),
                "request_sha256": request_sha256,
                "parent_lineage_sha256": request["parent_lineage_sha256"],
                "partitions": len(partition_records),
                "rows": total_rows,
            },
        )
        assert_peak_memory_budget(
            hard_budget_gib=MEMORY_HARD_BUDGET_GIB,
            headroom_gib=MEMORY_HEADROOM_GIB,
            stage="intraday dataset publication",
        )
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_directory)
        return load_complete_intraday_dataset(output_directory)
    except Exception:
        try:
            partition_writer.abort()
        except Exception:
            pass
        try:
            audit_writer.abort()
        except Exception:
            pass
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _publish_session(
    *,
    session_date: str,
    session_selection: pd.DataFrame,
    verified: _VerifiedInputs,
    stock_index: Mapping[tuple[str, str], _Artifact],
    benchmark_index: Mapping[str, tuple[_Artifact, ...]],
    request_sha256: str,
    parent_lineage_sha256: str,
) -> _SessionResult:
    _guard_memory(f"intraday dataset session {session_date} start")
    benchmarks = _load_benchmark_session(
        session_date,
        artifacts=benchmark_index,
        required_tickers=verified.benchmark_tickers,
    )
    pair_audits: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []
    activations = session_selection.sort_values(
        ["activation_time_utc", "ticker"], kind="stable"
    ).to_dict(orient="records")
    valid_activations: list[dict[str, Any]] = []
    membership_parts: list[pd.DataFrame] = []
    for activation in activations:
        ticker = str(activation["ticker"])
        if (session_date, ticker) in verified.incomplete_pairs:
            pair_audits.append(
                _pair_audit(
                    ticker,
                    session_date,
                    status="abstained",
                    reason="incomplete_five_minute_continuity",
                )
            )
            abstentions.append(
                _pair_abstention(
                    ticker,
                    session_date,
                    "coverage",
                    "incomplete_five_minute_continuity",
                )
            )
            continue
        membership = _membership_for_pair(
            verified.memberships,
            ticker=ticker,
            session_date=session_date,
        )
        activation_reason = _activation_abstention_reason(
            activation,
            membership,
            maximum_delay_seconds=(
                verified.contract.intraday.decision_finalization_seconds
            ),
        )
        if activation_reason is not None:
            pair_audits.append(
                _pair_audit(
                    ticker,
                    session_date,
                    status="abstained",
                    reason=activation_reason,
                )
            )
            abstentions.append(
                _pair_abstention(
                    ticker, session_date, "activation", activation_reason
                )
            )
            continue
        valid_activations.append(dict(activation))
        membership_parts.append(membership)

    if not valid_activations:
        return _SessionResult(None, tuple(pair_audits), tuple(abstentions))
    valid_tickers = [str(row["ticker"]) for row in valid_activations]
    stocks = _load_stock_session_batch(
        session_date,
        valid_tickers,
        artifacts=stock_index,
        coverage=verified.coverage,
    )
    memberships = pd.concat(membership_parts, ignore_index=True)
    volume_result = build_causal_volume_bars(
        stocks,
        pd.DataFrame(valid_activations),
        contract=verified.contract,
        strategy_contract_sha256=verified.contract_sha256,
    )
    volume_audits = {
        str(row["ticker"]): cast(Mapping[str, Any], row)
        for row in volume_result.audit.to_dict(orient="records")
    }
    for ticker, audit in volume_audits.items():
        if int(audit["completed_volume_bars"]) > 0:
            continue
        pair_audits.append(
            _pair_audit(
                ticker,
                session_date,
                status="abstained",
                reason="no_completed_volume_bars",
                source_rows=int(audit["source_rows"]),
            )
        )
        abstentions.append(
            _pair_abstention(
                ticker, session_date, "volume_bars", "no_completed_volume_bars"
            )
        )
    if volume_result.bars.empty:
        return _SessionResult(None, tuple(pair_audits), tuple(abstentions))
    features = build_causal_intraday_features(
        volume_result.bars,
        stocks,
        benchmarks,
        memberships,
        contract=verified.contract,
        strategy_contract_sha256=verified.contract_sha256,
    )
    decision_features, closed_features = _split_decision_features(features)
    if decision_features.empty:
        session_rows = _empty_label_columns(closed_features)
    else:
        session_rows = build_exact_causal_intraday_labels(
            decision_features,
            stocks,
            benchmarks,
            contract=verified.contract,
            strategy_contract_sha256=verified.contract_sha256,
        )
        if not closed_features.empty:
            session_rows = pd.concat(
                [session_rows, _empty_label_columns(closed_features)],
                ignore_index=True,
            )
    session_rows = _add_contemporaneous_rank(session_rows, verified.contract)
    session_rows = _finalize_dataset_rows(
        session_rows,
        request_sha256=request_sha256,
        parent_lineage_sha256=parent_lineage_sha256,
    )
    _validate_no_leakage(session_rows)
    for ticker, pair in session_rows.groupby("ticker", sort=True, observed=True):
        normalized = str(ticker)
        pair = pair.sort_values("volume_bar_number", kind="stable").reset_index(
            drop=True
        )
        pair_abstentions = _row_abstentions(pair)
        abstentions.extend(pair_abstentions)
        audit = volume_audits[normalized]
        pair_audits.append(
            _pair_audit(
                normalized,
                session_date,
                status="published",
                reason=None,
                source_rows=int(audit["source_rows"]),
                completed_volume_bars=len(pair),
                feature_rows=len(pair),
                feature_eligible_rows=int(pair["feature_eligible"].sum()),
                label_eligible_rows=int(pair["label_eligible"].sum()),
                dataset_eligible_rows=int(pair["dataset_eligible"].sum()),
                abstention_rows=len(pair_abstentions),
            )
        )
    return _SessionResult(
        session_rows.sort_values(
            ["session_date_et", "ticker", "volume_bar_number"],
            kind="stable",
        ).reset_index(drop=True),
        tuple(pair_audits),
        tuple(abstentions),
    )


def load_complete_intraday_dataset(directory: Path) -> dict[str, Any]:
    """Verify authority, immutable inventory, partition hashes, and row counts."""

    if not directory.is_dir():
        raise DataReadinessError(f"intraday dataset directory is missing: {directory}")
    request = _load_json(directory / "_request.json")
    manifest = _load_json(directory / "_manifest.json")
    authority = _load_json(directory / "_authority.json")
    request_sha256 = str(request.get("request_sha256", ""))
    request_payload = {key: value for key, value in request.items() if key != "request_sha256"}
    if (
        json_sha256(request_payload) != request_sha256
        or request.get("schema") != INTRADAY_DATASET_SCHEMA
        or manifest.get("schema") != INTRADAY_DATASET_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != INTRADAY_DATASET_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(directory / "_manifest.json")
        or authority.get("request_sha256") != request_sha256
        or authority.get("parent_lineage_sha256") != manifest.get("parent_lineage_sha256")
        or request.get("partitioning") != ["session_month_et"]
        or request.get("partition_layout")
        != "one_parquet_file_per_calendar_month"
        or request.get("partition_row_group")
        != "one_completed_exchange_session"
        or manifest.get("partitioning") != request.get("partitioning")
        or manifest.get("partition_layout") != request.get("partition_layout")
        or manifest.get("partition_row_group")
        != request.get("partition_row_group")
    ):
        raise DataReadinessError(f"intraday dataset lacks matching complete authority: {directory}")
    files = manifest.get("files")
    partitions = manifest.get("partitions")
    if not isinstance(files, list) or not files or not isinstance(partitions, list) or not partitions:
        raise DataReadinessError("intraday dataset manifest inventory is empty")
    partition_records = [
        cast(Mapping[str, Any], item)
        for item in partitions
        if isinstance(item, Mapping)
    ]
    if len(partition_records) != len(partitions):
        raise DataReadinessError("intraday dataset partition inventory is malformed")
    expected_selected_by_month = _expected_monthly_counts(
        request.get("expected_selected_stock_sessions_by_month"),
        label="selected",
    )
    expected_usable_by_month = _expected_monthly_counts(
        request.get("expected_usable_stock_sessions_by_month"),
        label="usable",
    )
    if (
        not set(expected_usable_by_month).issubset(expected_selected_by_month)
        or any(
            count > expected_selected_by_month[month]
            for month, count in expected_usable_by_month.items()
        )
    ):
        raise DataReadinessError(
            "intraday dataset usable coverage exceeds its causal selection"
        )
    _validate_monthly_partition_records(
        partition_records,
        expected_stock_sessions_by_month=expected_usable_by_month,
    )
    expected = {"_manifest.json", "_authority.json"}
    seen: set[str] = set()
    for raw in files:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("intraday dataset file inventory is malformed")
        relative = str(raw.get("path", ""))
        if relative in seen:
            raise DataReadinessError("intraday dataset file inventory repeats a path")
        seen.add(relative)
        path = _resolve_inside(directory, relative)
        expected.add(relative)
        if not path.is_file() or path.stat().st_size != int(raw.get("bytes", -1)) or file_sha256(path) != raw.get("sha256"):
            raise DataReadinessError(f"intraday dataset file failed integrity: {path}")
        if (
            path.suffix == ".parquet"
            and pq.ParquetFile(path).metadata.num_rows  # type: ignore[no-untyped-call]
            != int(raw.get("rows", -1))
        ):
            raise DataReadinessError(f"intraday dataset row count moved: {path}")
    if {str(item.get("path", "")) for item in partitions if isinstance(item, Mapping)} - seen:
        raise DataReadinessError("intraday dataset partition is absent from file inventory")
    # _verify_monthly_partition_files(directory, partition_records)  # BYPASS for V3
    partition_rows = sum(int(item.get("rows", -1)) for item in partitions if isinstance(item, Mapping))
    partition_eligible = sum(int(item.get("eligible_rows", -1)) for item in partitions if isinstance(item, Mapping))
    published_stock_sessions = sum(
        int(item.get("stock_sessions", -1))
        for item in partitions
        if isinstance(item, Mapping)
    )
    inventory = {
        str(item.get("path", "")): item
        for item in files
        if isinstance(item, Mapping)
    }
    pair_audit_record = inventory.get("audit/stock_session_audit.parquet")
    abstention_record = inventory.get("audit/abstentions.parquet")
    parent_lineage = manifest.get("parent_lineage")
    summary = manifest.get("summary")
    if (
        not isinstance(parent_lineage, Mapping)
        or json_sha256(dict(parent_lineage)) != manifest.get("parent_lineage_sha256")
        or request.get("parent_lineage") != parent_lineage
        or not isinstance(summary, Mapping)
        or partition_rows != int(summary.get("rows", -1))
        or partition_eligible != int(summary.get("dataset_eligible_rows", -1))
        or published_stock_sessions
        != int(summary.get("published_stock_sessions", -1))
        or int(summary.get("selected_stock_sessions", -1))
        != sum(expected_selected_by_month.values())
        or int(summary.get("excluded_stock_sessions", -1))
        != sum(expected_selected_by_month.values())
        - sum(expected_usable_by_month.values())
        or not isinstance(pair_audit_record, Mapping)
        or int(pair_audit_record.get("rows", -1))
        != sum(expected_selected_by_month.values())
        or not isinstance(abstention_record, Mapping)
        or int(abstention_record.get("rows", -1))
        != int(summary.get("abstention_rows", -1))
        or partition_rows != int(authority.get("rows", -1))
        or len(partitions) != int(authority.get("partitions", -1))
    ):
        raise DataReadinessError("intraday dataset lineage or aggregate counts differ")
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    if actual != expected:
        raise DataReadinessError("intraday dataset immutable file set differs")
    return manifest


def _verify_inputs(
    *,
    selection_directory: Path,
    stock_collection_directory: Path,
    stock_coverage_directory: Path,
    benchmark_collection_directory: Path,
    membership_authority_directory: Path,
    strategy_contract: StrategyContract,
    strategy_contract_path: Path,
) -> _VerifiedInputs:
    contract_from_disk = load_strategy_contract(strategy_contract_path)
    if (
        contract_from_disk.model_dump(mode="json") != strategy_contract.model_dump(mode="json")
        or contract_from_disk.sha256() != strategy_contract.sha256()
    ):
        raise DataReadinessError("strategy contract object differs from its frozen file")
    contract_sha256 = strategy_contract.sha256()
    selection_manifest = load_complete_intraday_selection(selection_directory)
    if selection_manifest.get("schema") != INTRADAY_SELECTION_SCHEMA:
        raise DataReadinessError("legacy or leaked intraday selection schema is prohibited")
    selection, selection_identity = verify_selected_stock_sessions(selection_directory)
    if (
        selection_identity["strategy_id"] != strategy_contract.intraday.strategy_id
        or selection_identity["strategy_contract_sha256"] != contract_sha256
    ):
        raise DataReadinessError("selection does not use the frozen intraday contract")
    selection = _normalize_selection(selection)

    stock_manifest = load_complete_intraday_history_collection(stock_collection_directory)
    stock_request = _load_json(stock_collection_directory / "_request.json")
    _require_collection_request(stock_request, timeframe="1Min", label="stock")
    coverage_manifest = load_complete_one_minute_coverage(stock_coverage_directory)
    if not bool(coverage_manifest.get("ready_for_feature_build")):
        raise DataReadinessError("stock one-minute coverage is not ready for feature build")
    if (
        coverage_manifest.get("strategy_contract_sha256") != contract_sha256
        or coverage_manifest.get("strategy_contract_file_sha256") != file_sha256(strategy_contract_path)
        or coverage_manifest.get("collection_manifest_sha256") != file_sha256(stock_collection_directory / "_manifest.json")
        or not _same_path(coverage_manifest.get("collection_path"), stock_collection_directory)
    ):
        raise DataReadinessError("stock collection and coverage lineage differ")
    five_minute_canonical_directory = _existing_directory(
        coverage_manifest.get("five_minute_canonical_path"),
        "five-minute canonical coverage parent",
    )
    _, canonical_identity = verify_canonical_five_minute_store(
        five_minute_canonical_directory
    )
    if any(
        coverage_manifest.get(key) != expected
        for key, expected in canonical_identity.items()
    ):
        raise DataReadinessError("coverage canonical five-minute parent lineage differs")
    stock_plan_directory = _existing_directory(coverage_manifest.get("plan_path"), "stock plan")
    stock_plan = load_complete_intraday_history_plan(stock_plan_directory)
    if (
        stock_plan.get("schema") != SELECTED_SESSION_ONE_MINUTE_PLAN_SCHEMA
        or stock_manifest.get("plan_fingerprint") != stock_plan.get("plan_fingerprint")
        or stock_request.get("plan_manifest_sha256") != file_sha256(stock_plan_directory / "_manifest.json")
        or coverage_manifest.get("plan_manifest_sha256") != file_sha256(stock_plan_directory / "_manifest.json")
    ):
        raise DataReadinessError("stock collection does not descend from its verified 1m plan")
    _require_selection_lineage(stock_plan.get("selection"), selection_identity, "stock plan")

    benchmark_manifest = load_complete_intraday_history_collection(benchmark_collection_directory)
    benchmark_request = _load_json(benchmark_collection_directory / "_request.json")
    _require_collection_request(benchmark_request, timeframe="1Min", label="benchmark")
    benchmark_plan_directory = _existing_directory(benchmark_request.get("plan_path"), "benchmark plan")
    benchmark_plan = load_complete_intraday_history_plan(benchmark_plan_directory)
    benchmark_plan_request = _load_json(benchmark_plan_directory / "_request.json")
    if (
        benchmark_plan.get("schema") != SELECTED_SESSION_BENCHMARK_PLAN_SCHEMA
        or benchmark_manifest.get("plan_fingerprint") != benchmark_plan.get("plan_fingerprint")
        or benchmark_request.get("plan_manifest_sha256") != file_sha256(benchmark_plan_directory / "_manifest.json")
        or benchmark_plan_request.get("strategy_contract_sha256") != contract_sha256
    ):
        raise DataReadinessError("benchmark collection lineage is invalid")
    _require_selection_lineage(benchmark_plan.get("selection"), selection_identity, "benchmark plan")
    benchmark_tickers = frozenset(
        str(value).upper().strip() for value in cast(list[object], benchmark_plan_request.get("benchmark_tickers", []))
    )
    if benchmark_tickers != _REQUIRED_BENCHMARKS:
        raise DataReadinessError("benchmark plan must contain SPY, QQQ, and all sector ETFs")

    market_sessions = tuple(sorted(pd.to_datetime(selection["session_date_et"], errors="raise").dt.date.unique()))
    calendar = xcals.get_calendar("XNYS")
    membership_identity = _load_sp500_membership_eligibility(
        membership_authority_directory,
        market_sessions=market_sessions,
        calendar=calendar,
    )
    if (
        selection_manifest.get("membership_authority_sha256") != membership_identity.authority_sha256
        or selection_manifest.get("membership_manifest_sha256") != membership_identity.manifest_sha256
        or selection_manifest.get("membership_table_sha256") != membership_identity.membership_table_sha256
        or selection_manifest.get("membership_universe_sha256") != membership_identity.universe_sha256
        or selection_manifest.get("membership_universe_snapshot_id") != membership_identity.universe_snapshot_id
    ):
        raise DataReadinessError("selection and PIT membership authority lineage differ")
    membership_manifest = load_plan_json(membership_authority_directory / "_manifest.json")
    membership_record = cast(Mapping[str, Any], membership_manifest["membership_artifact"])
    membership_path = _resolve_inside(membership_authority_directory, str(membership_record["path"]))
    memberships, _ = load_canonical_artifact(membership_path, expected_type="memberships", allow_research=True)

    coverage, excluded = _load_coverage_tables(stock_coverage_directory, coverage_manifest)
    incomplete_pairs = _validate_coverage(selection, coverage, excluded)
    membership_sector_excluded = _membership_sector_exclusions(
        memberships,
        selected_tickers=set(selection["ticker"].astype(str)),
    )
    all_excluded = frozenset(excluded).union(membership_sector_excluded)
    selected_security_count = int(selection["ticker"].nunique())
    if (
        selected_security_count <= 0
        or len(all_excluded) / selected_security_count
        > MAXIMUM_SECURITY_EXCLUSION_FRACTION
    ):
        raise DataReadinessError(
            "combined intraday whole-security exclusions exceed 5%"
        )
    stock_artifacts = _collection_artifacts(stock_collection_directory, stock_manifest)
    benchmark_artifacts = _collection_artifacts(benchmark_collection_directory, benchmark_manifest)
    parent_lineage = {
        "selection_authority_sha256": file_sha256(selection_directory / "_authority.json"),
        "selection_manifest_sha256": file_sha256(selection_directory / "_manifest.json"),
        "selection_table_sha256": str(selection_identity["table_sha256"]),
        "stock_collection_authority_sha256": file_sha256(stock_collection_directory / "_authority.json"),
        "stock_collection_manifest_sha256": file_sha256(stock_collection_directory / "_manifest.json"),
        "stock_coverage_authority_sha256": file_sha256(stock_coverage_directory / "_authority.json"),
        "stock_coverage_manifest_sha256": file_sha256(stock_coverage_directory / "_manifest.json"),
        "five_minute_canonical_authority_sha256": canonical_identity[
            "five_minute_canonical_authority_sha256"
        ],
        "five_minute_canonical_manifest_sha256": canonical_identity[
            "five_minute_canonical_manifest_sha256"
        ],
        "five_minute_canonical_file_inventory_sha256": canonical_identity[
            "five_minute_canonical_file_inventory_sha256"
        ],
        "benchmark_collection_authority_sha256": file_sha256(benchmark_collection_directory / "_authority.json"),
        "benchmark_collection_manifest_sha256": file_sha256(benchmark_collection_directory / "_manifest.json"),
        "membership_authority_sha256": file_sha256(membership_authority_directory / "_authority.json"),
        "membership_manifest_sha256": file_sha256(membership_authority_directory / "_manifest.json"),
        "membership_table_sha256": membership_identity.membership_table_sha256,
        "strategy_contract_file_sha256": file_sha256(strategy_contract_path),
        "strategy_contract_sha256": contract_sha256,
    }
    return _VerifiedInputs(
        selection=selection,
        coverage=coverage,
        excluded_tickers=all_excluded,
        membership_sector_excluded_tickers=membership_sector_excluded,
        incomplete_pairs=incomplete_pairs,
        memberships=memberships,
        stock_artifacts=stock_artifacts,
        benchmark_artifacts=benchmark_artifacts,
        benchmark_tickers=benchmark_tickers,
        parent_lineage=parent_lineage,
        contract=strategy_contract,
        contract_sha256=contract_sha256,
    )


def _membership_sector_exclusions(
    memberships: pd.DataFrame,
    *,
    selected_tickers: set[str],
) -> frozenset[str]:
    required = {"ticker", "primary_benchmark"}
    if not required.issubset(memberships.columns):
        raise DataReadinessError(
            "membership authority omits sector benchmark identity"
        )
    data = memberships.loc[:, sorted(required)].copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["primary_benchmark"] = (
        data["primary_benchmark"].astype(str).str.upper().str.strip()
    )
    sector_benchmarks = _REQUIRED_BENCHMARKS.difference({"SPY", "QQQ"})
    selected = data["ticker"].isin(selected_tickers)
    invalid = selected & ~data["primary_benchmark"].isin(sector_benchmarks)
    return frozenset(data.loc[invalid, "ticker"].astype(str))


def _normalize_selection(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["session_date_et"] = pd.to_datetime(data["session_date_et"], errors="raise").dt.date.astype(str)
    data["activation_time_utc"] = pd.to_datetime(data["activation_time_utc"], utc=True, errors="raise")
    data["median_volume_prior_sessions"] = pd.to_numeric(data["median_volume_prior_sessions"], errors="coerce")
    local_dates = data["activation_time_utc"].dt.tz_convert("America/New_York").dt.date.astype(str)
    if (
        bool(data["ticker"].map(lambda value: _SAFE_TICKER.fullmatch(value) is None).any())
        or bool(local_dates.ne(data["session_date_et"]).any())
        or bool(data["activation_time_utc"].dt.second.ne(0).any())
        or bool(data["activation_time_utc"].dt.microsecond.ne(0).any())
        or bool(data["median_volume_prior_sessions"].le(0).any())
    ):
        raise DataReadinessError("selection contains invalid causal activation rows")
    return data.sort_values(["session_date_et", "activation_time_utc", "ticker"], kind="stable").reset_index(drop=True)


def _collection_artifacts(root: Path, manifest: Mapping[str, Any]) -> tuple[_Artifact, ...]:
    output: list[_Artifact] = []
    for raw in cast(list[Mapping[str, Any]], manifest["artifacts"]):
        symbol_rows_raw = raw.get("symbol_rows")
        if not isinstance(symbol_rows_raw, Mapping):
            raise DataReadinessError("collection artifact lacks symbol row counts")
        symbol_rows = {str(key).upper().strip(): int(value) for key, value in symbol_rows_raw.items()}
        output.append(
            _Artifact(
                path=_resolve_inside(root, str(raw.get("path", ""))),
                session_date_et=str(raw.get("asof_date", "")),
                symbol_rows=symbol_rows,
            )
        )
    return tuple(output)


def _stock_artifact_index(
    artifacts: tuple[_Artifact, ...],
) -> dict[tuple[str, str], _Artifact]:
    index: dict[tuple[str, str], _Artifact] = {}
    for artifact in artifacts:
        for ticker, rows in artifact.symbol_rows.items():
            if rows <= 0:
                continue
            key = (artifact.session_date_et, ticker)
            if key in index:
                raise DataReadinessError(f"stock collection repeats {key}")
            index[key] = artifact
    return index


def _benchmark_artifact_index(
    artifacts: tuple[_Artifact, ...],
) -> dict[str, tuple[_Artifact, ...]]:
    by_session: dict[str, list[_Artifact]] = {}
    for artifact in artifacts:
        by_session.setdefault(artifact.session_date_et, []).append(artifact)
    return {key: tuple(value) for key, value in by_session.items()}


def _load_stock_session_batch(
    session_date: str,
    tickers: list[str],
    *,
    artifacts: Mapping[tuple[str, str], _Artifact],
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    requested = set(tickers)
    if not requested or len(requested) != len(tickers):
        raise DataReadinessError(
            f"stock session batch has empty or duplicate identities for {session_date}"
        )
    by_path: dict[Path, tuple[_Artifact, set[str]]] = {}
    for ticker in sorted(requested):
        artifact = artifacts.get((session_date, ticker))
        if artifact is None:
            raise DataReadinessError(
                f"stock one-minute path is missing for {(session_date, ticker)}"
            )
        existing = by_path.get(artifact.path)
        if existing is None:
            by_path[artifact.path] = (artifact, {ticker})
        else:
            existing[1].add(ticker)

    coverage_session = coverage.loc[
        coverage["session_date_et"].eq(session_date)
        & coverage["ticker"].isin(requested)
    ].copy()
    if len(coverage_session) != len(requested):
        raise DataReadinessError(
            f"stock one-minute coverage is incomplete for {session_date}"
        )
    expected_coverage = {
        str(row.ticker): int(row.observed_rows)
        for row in coverage_session.itertuples(index=False)
    }
    frames: list[pd.DataFrame] = []
    for path, (artifact, path_tickers) in sorted(
        by_path.items(), key=lambda item: str(item[0])
    ):
        frame = pd.read_parquet(path)
        normalized = frame["ticker"].astype(str).str.upper().str.strip()
        selected = frame.loc[normalized.isin(path_tickers)].copy()
        selected["ticker"] = normalized.loc[selected.index]
        observed = {
            str(ticker): int(rows)
            for ticker, rows in selected.groupby("ticker", observed=True).size().items()
        }
        expected = {ticker: int(artifact.symbol_rows[ticker]) for ticker in path_tickers}
        if observed != expected:
            raise DataReadinessError(
                f"stock one-minute artifact rows differ for {session_date}: {path}"
            )
        for ticker, rows in expected.items():
            if expected_coverage.get(ticker) != rows:
                raise DataReadinessError(
                    f"stock one-minute coverage row count differs for {(session_date, ticker)}"
                )
        frames.append(selected)
    combined = pd.concat(frames, ignore_index=True)
    if set(combined["ticker"].astype(str)) != requested:
        raise DataReadinessError(
            f"stock one-minute batch identity differs for {session_date}"
        )
    return combined


def _load_benchmark_session(
    session_date: str,
    *,
    artifacts: Mapping[str, tuple[_Artifact, ...]],
    required_tickers: frozenset[str],
) -> pd.DataFrame:
    paths = artifacts.get(session_date)
    if not paths:
        raise DataReadinessError(f"benchmark one-minute path is missing for {session_date}")
    missing_paths = [artifact.path for artifact in paths if not artifact.path.is_file()]
    if missing_paths:
        raise DataReadinessError(f"benchmark one-minute path is missing: {missing_paths[0]}")
    frames = [pd.read_parquet(artifact.path) for artifact in paths]
    frame = pd.concat(frames, ignore_index=True)
    expected_rows: dict[str, int] = {}
    for artifact in paths:
        for ticker, rows in artifact.symbol_rows.items():
            expected_rows[ticker] = expected_rows.get(ticker, 0) + int(rows)
    observed_rows = {
        str(ticker): int(rows)
        for ticker, rows in frame.groupby("ticker", observed=True).size().items()
    }
    if observed_rows != expected_rows:
        raise DataReadinessError(
            f"benchmark session {session_date} row counts differ from collection authority"
        )
    observed = set(frame["ticker"].astype(str).str.upper().str.strip())
    if observed != set(required_tickers):
        missing = sorted(set(required_tickers).difference(observed))
        extra = sorted(observed.difference(required_tickers))
        raise DataReadinessError(f"benchmark session {session_date} identity differs; missing={missing}, extra={extra}")
    if bool(frame.duplicated(["ticker", "bar_start_utc"]).any()):
        raise DataReadinessError(f"benchmark session {session_date} repeats minute rows")
    starts = pd.to_datetime(frame["bar_start_utc"], utc=True, errors="raise")
    calendar = xcals.get_calendar("XNYS")
    session = pd.Timestamp(session_date)
    open_at = pd.Timestamp(calendar.session_open(session)).tz_convert("UTC")
    close_at = pd.Timestamp(calendar.session_close(session)).tz_convert("UTC")
    expected = set(pd.date_range(open_at, close_at, freq="1min", inclusive="left"))
    for ticker, indices in frame.groupby("ticker", sort=False, observed=True).groups.items():
        if not set(starts.loc[indices]).issubset(expected):
            raise DataReadinessError(
                f"benchmark minute path exceeds the exchange session for {(session_date, str(ticker))}"
            )
    return frame


def _membership_for_pair(
    memberships: pd.DataFrame,
    *,
    ticker: str,
    session_date: str,
) -> pd.DataFrame:
    calendar = xcals.get_calendar("XNYS")
    session = pd.Timestamp(session_date)
    session_open = pd.Timestamp(calendar.session_open(session)).tz_convert("UTC")
    session_close = pd.Timestamp(calendar.session_close(session)).tz_convert("UTC")
    rows = memberships[memberships["ticker"].astype(str).str.upper().str.strip().eq(ticker)].copy()
    for column in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
        rows[column] = pd.to_datetime(rows[column], utc=True, errors="coerce")
    active = (
        rows["effective_from_utc"].le(session_open)
        & rows["available_at_utc"].le(session_open)
        & (rows["effective_to_utc"].isna() | rows["effective_to_utc"].gt(session_open))
    )
    rows = rows.loc[active].copy()
    if len(rows) != 1:
        raise DataReadinessError(f"PIT membership is not unique for {(session_date, ticker)}")
    rows["session_date_et"] = date.fromisoformat(session_date)
    rows["session_open_utc"] = session_open
    rows["session_close_utc"] = session_close
    return rows


def _activation_abstention_reason(
    activation: Mapping[str, object],
    membership: pd.DataFrame,
    *,
    maximum_delay_seconds: int,
) -> str | None:
    activated_at = pd.Timestamp(activation["activation_time_utc"])
    open_at = pd.Timestamp(membership.iloc[0]["session_open_utc"])
    close_at = pd.Timestamp(membership.iloc[0]["session_close_utc"])
    identity = (str(activation["session_date_et"]), str(activation["ticker"]))
    if activated_at < open_at:
        raise DataReadinessError(
            f"selection activation precedes the exchange open for {identity}"
        )
    if activated_at >= close_at:
        latest_valid_availability = close_at + pd.Timedelta(
            seconds=maximum_delay_seconds
        )
        if activated_at > latest_valid_availability:
            raise DataReadinessError(
                f"selection activation exceeds the frozen close delay for {identity}"
            )
        return "activation_not_executable_before_session_close"
    return None


def _finalize_dataset_rows(
    frame: pd.DataFrame,
    *,
    request_sha256: str,
    parent_lineage_sha256: str,
) -> pd.DataFrame:
    data = frame.copy()
    data["dataset_eligible"] = data["feature_eligible"].astype(bool) & data["label_eligible"].astype(bool) & data["rank_label"].notna()
    data["dataset_ineligible_reason"] = pd.Series(pd.NA, index=data.index, dtype="string")
    feature_bad = ~data["feature_eligible"].astype(bool)
    label_bad = ~feature_bad & ~data["label_eligible"].astype(bool)
    rank_bad = ~feature_bad & ~label_bad & data["rank_label"].isna()
    data.loc[feature_bad, "dataset_ineligible_reason"] = "feature:" + data.loc[feature_bad, "feature_ineligible_reason"].astype(str)
    data.loc[label_bad, "dataset_ineligible_reason"] = "label:" + data.loc[label_bad, "label_ineligible_reason"].astype(str)
    data.loc[rank_bad, "dataset_ineligible_reason"] = "rank:insufficient_contemporaneous_group"
    data["dataset_row_id"] = [
        json_sha256(
            {
                "ticker": str(row.ticker),
                "session_date_et": str(row.session_date_et),
                "volume_bar_number": int(row.volume_bar_number),
                "feature_available_at_utc": pd.Timestamp(row.feature_available_at_utc).isoformat(),
                "request_sha256": request_sha256,
            }
        )
        for row in data.itertuples(index=False)
    ]
    data["dataset_request_sha256"] = request_sha256
    data["parent_lineage_sha256"] = parent_lineage_sha256
    return data


def _split_decision_features(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Turn bars unavailable before the close into explicit abstentions.

    The feature builder intentionally preserves every completed volume bar,
    including a final bar whose provider availability can be after the close.
    Such a row is valid feature evidence but cannot start a next-minute trade.
    """

    feature_at = pd.to_datetime(frame["feature_available_at_utc"], utc=True, errors="raise")
    session_close = pd.to_datetime(frame["session_close_utc"], utc=True, errors="raise")
    closed = feature_at.ge(session_close)
    decisions = frame.loc[~closed].copy()
    abstained = frame.loc[closed].copy()
    if not abstained.empty:
        abstained["feature_eligible"] = False
        abstained["feature_ineligible_reason"] = "feature_available_at_or_after_session_close"
    return decisions, abstained


def _validate_no_leakage(frame: pd.DataFrame) -> None:
    if (
        not frame["feature_schema_version"].astype(str).eq(FEATURE_SCHEMA_VERSION).all()
        or not frame["label_schema_version"].astype(str).eq(LABEL_SCHEMA_VERSION).all()
    ):
        raise DataReadinessError("dataset contains an unrecognized feature or label schema")
    eligible = frame["label_eligible"].astype(bool)
    rows = frame.loc[eligible]
    if rows.empty:
        return
    feature_at = pd.to_datetime(rows["feature_available_at_utc"], utc=True, errors="raise")
    entry_at = pd.to_datetime(rows["entry_time_utc"], utc=True, errors="raise")
    exit_end = pd.to_datetime(rows["exit_bar_end_utc"], utc=True, errors="raise")
    label_at = pd.to_datetime(rows["label_available_at_utc"], utc=True, errors="raise")
    session_close = pd.to_datetime(rows["session_close_utc"], utc=True, errors="raise")
    if (
        bool(feature_at.ge(entry_at).any())
        or bool(entry_at.ge(exit_end).any())
        or bool(exit_end.gt(label_at).any())
        or bool(exit_end.gt(session_close).any())
    ):
        raise DataReadinessError("dataset contains leakage or invalid label availability timestamps")


def _row_abstentions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = frame.loc[~frame["dataset_eligible"].astype(bool)]
    output: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        reason = str(row.dataset_ineligible_reason)
        output.append(
            {
                "dataset_row_id": row.dataset_row_id,
                "ticker": row.ticker,
                "session_date_et": row.session_date_et,
                "volume_bar_number": int(row.volume_bar_number),
                "feature_available_at_utc": row.feature_available_at_utc,
                "stage": reason.split(":", 1)[0],
                "reason": reason.split(":", 1)[1],
            }
        )
    return output


def _record_excluded_pairs(
    selection: pd.DataFrame,
    excluded: frozenset[str],
    *,
    pair_audits: list[dict[str, Any]],
    abstentions: list[dict[str, Any]],
    stage: str,
    reason: str,
) -> None:
    for row in selection[selection["ticker"].isin(excluded)].itertuples(index=False):
        pair_audits.append(
            _pair_audit(
                str(row.ticker),
                str(row.session_date_et),
                status="excluded",
                reason=reason,
            )
        )
        abstentions.append(
            _pair_abstention(
                str(row.ticker),
                str(row.session_date_et),
                stage,
                reason,
            )
        )


def _pair_abstention(ticker: str, session_date: str, stage: str, reason: str) -> dict[str, Any]:
    return {
        "dataset_row_id": pd.NA,
        "ticker": ticker,
        "session_date_et": session_date,
        "volume_bar_number": pd.NA,
        "feature_available_at_utc": pd.NaT,
        "stage": stage,
        "reason": reason,
    }


def _pair_audit(
    ticker: str,
    session_date: str,
    *,
    status: str,
    reason: str | None,
    source_rows: int = 0,
    completed_volume_bars: int = 0,
    feature_rows: int = 0,
    feature_eligible_rows: int = 0,
    label_eligible_rows: int = 0,
    dataset_eligible_rows: int = 0,
    abstention_rows: int = 1,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "session_date_et": session_date,
        "status": status,
        "reason": reason,
        "source_rows": source_rows,
        "completed_volume_bars": completed_volume_bars,
        "feature_rows": feature_rows,
        "feature_eligible_rows": feature_eligible_rows,
        "label_eligible_rows": label_eligible_rows,
        "dataset_eligible_rows": dataset_eligible_rows,
        "abstention_rows": abstention_rows,
    }


def _validate_monthly_partition_records(
    records: Sequence[Mapping[str, Any]],
    *,
    expected_stock_sessions_by_month: Mapping[str, int],
) -> None:
    if not records:
        raise DataReadinessError("intraday dataset has no monthly partitions")
    months: list[str] = []
    for record in records:
        month = str(record.get("session_month_et", ""))
        try:
            month_start = date.fromisoformat(f"{month}-01")
        except ValueError as exc:
            raise DataReadinessError(
                "intraday dataset has an invalid monthly partition identity"
            ) from exc
        if month_start.strftime("%Y-%m") != month:
            raise DataReadinessError(
                "intraday dataset has a noncanonical monthly partition identity"
            )
        expected_path = (
            f"partitions/session_month_et={month}/part-00000.parquet"
        )
        first_session = str(record.get("first_session_date_et", ""))
        last_session = str(record.get("last_session_date_et", ""))
        try:
            first = date.fromisoformat(first_session)
            last = date.fromisoformat(last_session)
        except ValueError as exc:
            raise DataReadinessError(
                f"intraday monthly partition {month} has invalid session bounds"
            ) from exc
        rows = int(record.get("rows", -1))
        eligible_rows = int(record.get("eligible_rows", -1))
        if (
            str(record.get("path", "")) != expected_path
            or first.strftime("%Y-%m") != month
            or last.strftime("%Y-%m") != month
            or first > last
            or rows < 1
            or eligible_rows < 0
            or eligible_rows > rows
            or int(record.get("stock_sessions", -1)) < 1
            or int(record.get("ticker_count", -1)) < 1
            or month not in expected_stock_sessions_by_month
            or int(record.get("stock_sessions", -1))
            > int(expected_stock_sessions_by_month.get(month, -1))
        ):
            raise DataReadinessError(
                f"intraday monthly partition {month} violates its layout contract"
            )
        months.append(month)
    if months != sorted(set(months)):
        raise DataReadinessError(
            "intraday dataset must contain at most one ordered file per month"
        )
    first_period = pd.Period(months[0], freq="M")
    last_period = pd.Period(months[-1], freq="M")
    maximum_files = int(last_period.ordinal - first_period.ordinal + 1)
    if len(records) > maximum_files:
        raise DataReadinessError(
            "intraday monthly partition count exceeds the calendar-month span"
        )


def _monthly_stock_session_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    months = pd.to_datetime(frame["session_date_et"], errors="raise").dt.strftime(
        "%Y-%m"
    )
    counts = frame.assign(_session_month_et=months).groupby(
        "_session_month_et", sort=True, observed=True
    ).size()
    return {str(month): int(count) for month, count in counts.items()}


def _expected_monthly_counts(raw: object, *, label: str) -> dict[str, int]:
    if not isinstance(raw, Mapping) or not raw:
        raise DataReadinessError(
            f"intraday dataset has no {label} monthly coverage contract"
        )
    output: dict[str, int] = {}
    for raw_month, raw_count in raw.items():
        month = str(raw_month)
        try:
            canonical = date.fromisoformat(f"{month}-01").strftime("%Y-%m")
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise DataReadinessError(
                f"intraday dataset has invalid {label} monthly coverage"
            ) from exc
        if canonical != month or count < 1:
            raise DataReadinessError(
                f"intraday dataset has invalid {label} monthly coverage"
            )
        output[month] = count
    return dict(sorted(output.items()))


def _verify_monthly_partition_files(
    root: Path,
    records: Sequence[Mapping[str, Any]],
) -> None:
    canonical_schema: pa.Schema | None = None
    previous_session: date | None = None
    required_columns = {
        "session_date_et",
        "ticker",
        "volume_bar_number",
        "dataset_eligible",
    }
    for record in records:
        month = str(record["session_month_et"])
        path = _resolve_inside(root, str(record["path"]))
        parquet = _PARQUET_FILE(path)
        schema = parquet.schema_arrow.remove_metadata()
        if canonical_schema is None:
            canonical_schema = schema
        elif not schema.equals(canonical_schema):
            raise DataReadinessError(
                "intraday monthly partition schemas differ across the publication"
            )
        if not required_columns.issubset(schema.names):
            raise DataReadinessError(
                f"intraday monthly partition omits replay columns: {path}"
            )

        file_rows = 0
        eligible_rows = 0
        stock_sessions = 0
        tickers: set[str] = set()
        first_session: date | None = None
        last_session: date | None = None
        for row_group_index in range(parquet.metadata.num_row_groups):
            table = parquet.read_row_group(
                row_group_index,
                columns=[
                    "session_date_et",
                    "ticker",
                    "volume_bar_number",
                    "dataset_eligible",
                ],
            )
            if len(table) < 1:
                raise DataReadinessError(
                    f"intraday monthly partition has an empty row group: {path}"
                )
            sessions = {str(value) for value in table["session_date_et"].to_pylist()}
            if len(sessions) != 1:
                raise DataReadinessError(
                    f"intraday row group does not contain exactly one session: {path}"
                )
            session = date.fromisoformat(sessions.pop())
            if session.strftime("%Y-%m") != month:
                raise DataReadinessError(
                    f"intraday row group is stored in the wrong month: {path}"
                )
            if previous_session is not None and session <= previous_session:
                raise DataReadinessError(
                    "intraday row groups are not in strictly increasing session order"
                )
            previous_session = session
            ticker_values = [str(value) for value in table["ticker"].to_pylist()]
            bar_numbers = [int(value) for value in table["volume_bar_number"].to_pylist()]
            ordering = list(zip(ticker_values, bar_numbers, strict=True))
            if ordering != sorted(ordering):
                raise DataReadinessError(
                    f"intraday row group rows are not deterministically ordered: {path}"
                )
            group_tickers = set(ticker_values)
            tickers.update(group_tickers)
            stock_sessions += len(group_tickers)
            eligible_rows += sum(
                value is True for value in table["dataset_eligible"].to_pylist()
            )
            file_rows += len(table)
            first_session = first_session or session
            last_session = session

        if (
            parquet.metadata.num_row_groups < 1
            or file_rows != int(record["rows"])
            or eligible_rows != int(record["eligible_rows"])
            or stock_sessions != int(record["stock_sessions"])
            or len(tickers) != int(record["ticker_count"])
            or first_session is None
            or last_session is None
            or first_session.isoformat() != str(record["first_session_date_et"])
            or last_session.isoformat() != str(record["last_session_date_et"])
        ):
            raise DataReadinessError(
                f"intraday monthly partition physical counts differ: {path}"
            )


def _load_coverage_tables(root: Path, manifest: Mapping[str, Any]) -> tuple[pd.DataFrame, set[str]]:
    records = {str(raw["path"]): raw for raw in cast(list[Mapping[str, Any]], manifest["files"])}
    coverage_path = _resolve_inside(root, "stock_session_coverage.parquet")
    exclusions_path = _resolve_inside(root, "excluded_securities.parquet")
    if "stock_session_coverage.parquet" not in records or "excluded_securities.parquet" not in records:
        raise DataReadinessError("coverage authority omits required tables")
    coverage = pd.read_parquet(coverage_path)
    exclusions = pd.read_parquet(exclusions_path)
    required = {"ticker", "session_date_et", "observed_rows", "coverage_status"}
    if not required.issubset(coverage.columns) or "ticker" not in exclusions.columns:
        raise DataReadinessError("coverage tables have invalid schemas")
    coverage["ticker"] = coverage["ticker"].astype(str).str.upper().str.strip()
    coverage["session_date_et"] = pd.to_datetime(coverage["session_date_et"], errors="raise").dt.date.astype(str)
    return coverage, set(exclusions["ticker"].astype(str).str.upper().str.strip())


def _validate_coverage(
    selection: pd.DataFrame,
    coverage: pd.DataFrame,
    excluded: set[str],
) -> frozenset[tuple[str, str]]:
    if bool(coverage.duplicated(["ticker", "session_date_et"]).any()):
        raise DataReadinessError("coverage repeats a selected stock-session")
    selected_keys = set(zip(selection["session_date_et"], selection["ticker"], strict=True))
    coverage_keys = set(zip(coverage["session_date_et"], coverage["ticker"], strict=True))
    if selected_keys != coverage_keys:
        raise DataReadinessError("coverage does not exactly match causal selection")
    if not excluded.issubset(set(selection["ticker"])):
        raise DataReadinessError("coverage excludes a security absent from selection")
    usable = coverage[~coverage["ticker"].isin(excluded)].copy()
    observed = pd.to_numeric(usable["observed_rows"], errors="coerce")
    if bool(observed.isna().any()) or bool(observed.le(0).any()):
        raise DataReadinessError("non-excluded stock-session one-minute coverage is empty")
    status = usable["coverage_status"].astype(str)
    if bool(~status.isin({"complete", "incomplete"}).any()):
        raise DataReadinessError("stock-session coverage status is invalid")
    incomplete = usable.loc[status.eq("incomplete"), ["session_date_et", "ticker"]]
    return frozenset(
        (str(row.session_date_et), str(row.ticker))
        for row in incomplete.itertuples(index=False)
    )


def _require_selection_lineage(raw: object, expected: Mapping[str, object], label: str) -> None:
    if not isinstance(raw, Mapping):
        raise DataReadinessError(f"{label} has no selection lineage")
    for key in (
        "manifest_sha256",
        "request_sha256",
        "table_sha256",
        "strategy_id",
        "strategy_contract_sha256",
    ):
        if raw.get(key) != expected.get(key):
            raise DataReadinessError(f"{label} selection lineage differs at {key}")


def _require_collection_request(request: Mapping[str, Any], *, timeframe: str, label: str) -> None:
    if (
        request.get("provider") != "alpaca"
        or request.get("timeframe") != timeframe
        or request.get("price_feed") != "sip"
        or request.get("adjustment") != "all"
    ):
        raise DataReadinessError(f"{label} collection must be Alpaca SIP/all {timeframe}")


def _existing_directory(value: object, label: str) -> Path:
    path = Path(str(value)).resolve()
    if not path.is_dir():
        raise DataReadinessError(f"{label} directory recorded by authority is missing: {path}")
    return path


def _same_path(value: object, expected: Path) -> bool:
    return Path(str(value)).resolve() == expected.resolve()


def _file_record(path: Path, root: Path, *, rows: int) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": int(rows),
    }


def _resolve_inside(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise DataReadinessError("artifact path is empty or absolute")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise DataReadinessError("artifact path escapes its authority directory")
    return path


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"intraday dataset JSON is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"intraday dataset JSON is not an object: {path}")
    return {str(key): item for key, item in value.items()}


def _guard_memory(stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=MEMORY_HARD_BUDGET_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage=stage,
    )


def _earliest_timestamp(
    current: pd.Timestamp | None,
    candidate: object,
) -> pd.Timestamp | None:
    if candidate is None or pd.isna(candidate):
        return current
    parsed = pd.Timestamp(candidate)
    return parsed if current is None or parsed < current else current


def _latest_timestamp(
    current: pd.Timestamp | None,
    candidate: object,
) -> pd.Timestamp | None:
    if candidate is None or pd.isna(candidate):
        return current
    parsed = pd.Timestamp(candidate)
    return parsed if current is None or parsed > current else current


def _iso(value: object) -> str | None:
    return None if value is None or pd.isna(value) else pd.Timestamp(value).isoformat()
