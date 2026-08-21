"""Atomic, lineage-bound publisher for the causal intraday training dataset."""
from __future__ import annotations



import json
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from market_predictor.canonical.store import (
    file_sha256,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.contracts.dataset_schemas import (
    _ABSTENTION_SCHEMA,
    _PAIR_AUDIT_SCHEMA,
    MEMORY_HARD_BUDGET_GIB,
    MEMORY_HEADROOM_GIB,
)
from market_predictor.resources import (
    assert_memory_budget,
)


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
        from market_predictor.intraday.datasets.transformations import _is_missing, _normalize_arrow_records
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
        from market_predictor.intraday.datasets.transformations import _is_missing, _normalize_arrow_records
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