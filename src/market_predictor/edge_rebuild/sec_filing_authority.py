"""Bounded, hash-bound decision-time SEC filing overlay for swing research."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final, Protocol, SupportsInt, cast
from uuid import uuid4
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.edge_rebuild.sec_filing_collection import (
    SEC_SOURCE_FAMILY,
    SecFilingCollection,
    load_sec_filing_collection,
    load_sec_identity_relations,
    normalize_sec_identity_relations,
)
from market_predictor.resources import assert_memory_budget, assert_peak_memory_budget
from market_predictor.v3.errors import DataReadinessError

SEC_AUTHORITY_SCHEMA: Final = "edge_rebuild.sec_filing_decision_authority.v2"
SEC_AUTHORITY_MANIFEST_SCHEMA: Final = "edge_rebuild.sec_filing_decision_manifest.v2"
SEC_AUTHORITY_COVERAGE_ARTIFACT_TYPE: Final = "sec_filing_authority_coverage"
WINDOWS: Final[Mapping[str, pd.Timedelta]] = {
    "1d": pd.Timedelta(days=1),
    "3d": pd.Timedelta(days=3),
}
MATERIAL_FORMS: Final = frozenset({"8-K", "8-K/A", "10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A", "6-K", "6-K/A"})
FORM_FEATURES: Final[Mapping[str, frozenset[str]]] = {
    "8k": frozenset({"8-K", "8-K/A", "6-K", "6-K/A"}),
    "periodic": frozenset({"10-K", "10-K/A", "10-Q", "10-Q/A", "20-F", "20-F/A"}),
    "offering": frozenset({"S-1", "S-1/A", "S-3", "S-3/A", "424B1", "424B2", "424B3", "424B4", "424B5"}),
    "insider": frozenset({"3", "3/A", "4", "4/A", "5", "5/A"}),
}
_ET = ZoneInfo("America/New_York")
_IDENTITY_COLUMNS = ("decision_id", "security_id", "ticker", "decision_time_utc")
_MAXIMUM_MEMORY_GIB = 6.0
_MEMORY_HEADROOM_GIB = 0.75


class _DatasetExpression(Protocol):
    def __and__(self, other: _DatasetExpression, /) -> _DatasetExpression: ...


class _DatasetField(Protocol):
    def __ge__(self, other: datetime, /) -> _DatasetExpression: ...

    def __lt__(self, other: datetime, /) -> _DatasetExpression: ...


class _DatasetScanner(Protocol):
    def to_batches(self) -> Iterable[pa.RecordBatch]: ...


class _DatasetProtocol(Protocol):
    schema: pa.Schema

    def scanner(
        self, *, columns: list[str], batch_size: int
    ) -> _DatasetScanner: ...

    def to_table(
        self, *, columns: list[str], filter: _DatasetExpression
    ) -> pa.Table: ...


class _DatasetFactory(Protocol):
    def __call__(self, source: str, *, format: str) -> _DatasetProtocol: ...


class _DatasetFieldFactory(Protocol):
    def __call__(self, name: str) -> _DatasetField: ...


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


class _ParquetFileProtocol(Protocol):
    schema_arrow: pa.Schema
    num_row_groups: int

    def read_row_group(self, index: int) -> pa.Table: ...


class _ParquetFileFactory(Protocol):
    def __call__(self, source: Path) -> _ParquetFileProtocol: ...


_DATASET = cast(_DatasetFactory, vars(pads)["dataset"])
_DATASET_FIELD = cast(_DatasetFieldFactory, vars(pads)["field"])
_PARQUET_WRITER = cast(_ParquetWriterFactory, pq.ParquetWriter)
_PARQUET_FILE = cast(_ParquetFileFactory, pq.ParquetFile)


def _as_int(value: object) -> int:
    return int(cast(str | bytes | bytearray | SupportsInt, value))


def _decision_columns() -> tuple[str, ...]:
    columns = [*_IDENTITY_COLUMNS, "sec_identity_proven", "sec_cik"]
    for window in WINDOWS:
        columns.extend(
            [
                f"sec_source_complete_{window}",
                f"sec_source_coverage_start_utc_{window}",
                f"sec_source_coverage_end_utc_{window}",
                f"sec_source_coverage_collected_at_utc_{window}",
                f"sec_filing_count_{window}",
                f"sec_material_filing_count_{window}",
                f"sec_amendment_filing_count_{window}",
                *(f"sec_{name}_filing_count_{window}" for name in FORM_FEATURES),
                f"sec_latest_filing_accepted_at_utc_{window}",
                f"sec_latest_filing_available_at_utc_{window}",
            ]
        )
    return tuple(columns)


_DECISION_COLUMNS = _decision_columns()


def _arrow_schema() -> pa.Schema:
    fields: list[pa.Field] = []
    timestamp_columns = {"decision_time_utc"}
    bool_columns = {"sec_identity_proven", *(f"sec_source_complete_{window}" for window in WINDOWS)}
    for window in WINDOWS:
        timestamp_columns.update(
            {
                f"sec_source_coverage_start_utc_{window}",
                f"sec_source_coverage_end_utc_{window}",
                f"sec_source_coverage_collected_at_utc_{window}",
                f"sec_latest_filing_accepted_at_utc_{window}",
                f"sec_latest_filing_available_at_utc_{window}",
            }
        )
    for column in _DECISION_COLUMNS:
        if column in timestamp_columns:
            fields.append(pa.field(column, pa.timestamp("ns", tz="UTC")))
        elif column in bool_columns:
            fields.append(pa.field(column, pa.bool_()))
        elif column.startswith("sec_") and column not in {"sec_cik"}:
            fields.append(pa.field(column, pa.float64()))
        else:
            fields.append(pa.field(column, pa.string()))
    return pa.schema(fields)


_AUTHORITY_ARROW_SCHEMA = _arrow_schema()


@dataclass(frozen=True, slots=True)
class SecFilingDecisionAuthority:
    directory: Path
    partition_records: tuple[Mapping[str, object], ...]
    coverage: pd.DataFrame
    manifest: Mapping[str, object]
    authority: Mapping[str, object]

    @property
    def decision_rows(self) -> int:
        return _as_int(self.manifest["decision_rows"])

    def read_decisions(self, months: set[str] | None = None) -> pd.DataFrame:
        frames = []
        for record in self.partition_records:
            if months is not None and str(record["month_et"]) not in months:
                continue
            frames.append(pd.read_parquet(self.directory / str(record["path"])))
        if not frames:
            return pd.DataFrame(columns=_DECISION_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    @property
    def decisions(self) -> pd.DataFrame:
        return self.read_decisions()


class _DecisionSource:
    def __init__(self, value: pd.DataFrame | Path) -> None:
        self.path = Path(value).resolve() if isinstance(value, Path) else None
        self.frame = None if isinstance(value, Path) else _normalize_decisions(value)
        self.dataset: _DatasetProtocol | None = None
        if self.path is not None:
            if not self.path.exists():
                raise DataReadinessError(f"SEC decisions source is missing: {self.path}")
            dataset = _DATASET(str(self.path), format="parquet")
            missing = sorted(set(_IDENTITY_COLUMNS).difference(dataset.schema.names))
            if missing:
                raise DataReadinessError("SEC decisions source missing columns: " + ", ".join(missing))
            self.dataset = dataset

    def months(self) -> tuple[str, ...]:
        if self.frame is not None:
            values = self.frame["decision_time_utc"]
            return tuple(sorted(set(values.dt.tz_convert(_ET).dt.strftime("%Y-%m"))))
        months: set[str] = set()
        dataset = cast(_DatasetProtocol, self.dataset)
        for batch in dataset.scanner(columns=["decision_time_utc"], batch_size=65_536).to_batches():
            values = _strict_utc(batch.column(0).to_pandas(), "decision_time_utc")
            months.update(values.dt.tz_convert(_ET).dt.strftime("%Y-%m"))
        return tuple(sorted(months))

    def load_month(self, month: str) -> pd.DataFrame:
        start_local = pd.Timestamp(f"{month}-01", tz=_ET)
        end_local = start_local + pd.offsets.MonthBegin(1)
        if self.frame is not None:
            local = self.frame["decision_time_utc"].dt.tz_convert(_ET)
            frame = self.frame.loc[(local >= start_local) & (local < end_local)].copy()
        else:
            dataset = cast(_DatasetProtocol, self.dataset)
            expression = (_DATASET_FIELD("decision_time_utc") >= start_local.tz_convert("UTC").to_pydatetime()) & (
                _DATASET_FIELD("decision_time_utc") < end_local.tz_convert("UTC").to_pydatetime()
            )
            frame = _normalize_decisions(dataset.to_table(columns=list(_IDENTITY_COLUMNS), filter=expression).to_pandas())
            local = frame["decision_time_utc"].dt.tz_convert(_ET)
            frame = frame.loc[(local >= start_local) & (local < end_local)].copy()
        return frame.sort_values(["decision_time_utc", "decision_id"], kind="stable").reset_index(drop=True)

    def lineage(self) -> dict[str, object]:
        if self.path is None:
            return {"kind": "in_memory", "semantic_sha256": _identity_frame_sha256(self.frame)}
        if self.path.is_file():
            return {"kind": "parquet_file", "path": str(self.path), "sha256": file_sha256(self.path)}
        files = sorted(path for path in self.path.rglob("*.parquet") if path.is_file())
        return {
            "kind": "parquet_directory",
            "path": str(self.path),
            "files": [{"path": str(path.relative_to(self.path)), "sha256": file_sha256(path)} for path in files],
        }


def publish_sec_filing_decision_authority(
    decisions: pd.DataFrame | Path,
    collection_directories: Sequence[Path],
    identity_relations: pd.DataFrame | Path,
    output_directory: Path,
    *,
    production_ready: bool = False,
) -> SecFilingDecisionAuthority:
    """Publish monthly SEC overlay partitions without materializing the full output."""

    if production_ready:
        raise DataReadinessError("retrospective SEC collections cannot produce a production-ready authority")
    source = _DecisionSource(decisions)
    months = source.months()
    if not months:
        raise DataReadinessError("SEC filing authority requires decision rows")
    relations = (
        load_sec_identity_relations(identity_relations)
        if isinstance(identity_relations, Path)
        else normalize_sec_identity_relations(identity_relations)
    )
    relation_lineage = _relation_lineage(identity_relations, relations)
    roots = tuple(sorted({Path(value).resolve() for value in collection_directories}, key=str))
    if not roots:
        raise DataReadinessError("SEC filing authority requires collection inputs")
    collections = tuple(load_sec_filing_collection(root) for root in roots)
    events = _merge_events(collections)
    coverage = _merge_coverage(collections)
    _validate_input_identity(events, coverage)
    relation_index = _relation_index(relations)
    event_index = _event_index(events)
    coverage_index = _coverage_index(coverage)
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise DataReadinessError(f"SEC filing authority is immutable: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = output_directory.with_name(f".{output_directory.name}.{uuid4().hex}.tmp")
    (staging / "partitions").mkdir(parents=True)
    source_artifacts = [_collection_lineage(value) for value in collections]
    request = {
        "schema": SEC_AUTHORITY_SCHEMA,
        "decision_source": source.lineage(),
        "decision_months_et": list(months),
        "identity_relation": relation_lineage,
        "windows": {name: int(value.total_seconds()) for name, value in WINDOWS.items()},
        "source_family": SEC_SOURCE_FAMILY,
        "source_artifacts": source_artifacts,
        "availability_policy": "sec_daily_swing_conservative_proxy",
        "historical_availability_proven": False,
        "coverage_completion_policy": "retrospective collection may complete after decision time",
        "missing_value_policy": "unproven relation or incomplete issuer coverage is null",
        "partitioning": "one file per ET calendar month; one row group per ET decision session",
        "production_ready": False,
    }
    request_sha256 = _json_sha256(request)
    source_lineage_sha256 = _json_sha256(source_artifacts)
    identity_hasher = hashlib.sha256()
    partition_records: list[dict[str, object]] = []
    decision_rows = 0
    try:
        assert_memory_budget(hard_budget_gib=_MAXIMUM_MEMORY_GIB, headroom_gib=_MEMORY_HEADROOM_GIB, stage="SEC authority start")
        for month in months:
            month_input = source.load_month(month)
            if month_input.empty:
                raise DataReadinessError(f"SEC decision month is empty after projection: {month}")
            relative = Path("partitions") / f"decision_month_et={month}.parquet"
            path = staging / relative
            writer = _PARQUET_WRITER(
                path,
                _AUTHORITY_ARROW_SCHEMA,
                compression="zstd",
                use_dictionary=True,
                write_statistics=True,
            )
            month_rows = 0
            sessions = 0
            first_session: str | None = None
            last_session: str | None = None
            try:
                session_keys = month_input["decision_time_utc"].dt.tz_convert(_ET).dt.date.astype(str)
                for session, positions in month_input.groupby(session_keys, sort=True).indices.items():
                    chunk_input = (
                        month_input.iloc[positions].sort_values(["decision_time_utc", "decision_id"], kind="stable").reset_index(drop=True)
                    )
                    _update_identity_hash(identity_hasher, chunk_input)
                    chunk = _aggregate_session(chunk_input, relation_index, event_index, coverage_index)
                    _decision_audit(chunk).raise_for_failure()
                    table = pa.Table.from_pandas(
                        chunk.loc[:, list(_DECISION_COLUMNS)], schema=_AUTHORITY_ARROW_SCHEMA, preserve_index=False, safe=True
                    )
                    writer.write_table(table, row_group_size=len(table))
                    month_rows += len(chunk)
                    decision_rows += len(chunk)
                    sessions += 1
                    first_session = first_session or str(session)
                    last_session = str(session)
                    del chunk, table, chunk_input
            finally:
                writer.close()
            partition_records.append(
                {
                    "path": relative.as_posix(),
                    "month_et": month,
                    "rows": month_rows,
                    "sessions": sessions,
                    "first_session_et": first_session,
                    "last_session_et": last_session,
                    "sha256": file_sha256(path),
                }
            )
            del month_input
            assert_memory_budget(
                hard_budget_gib=_MAXIMUM_MEMORY_GIB, headroom_gib=_MEMORY_HEADROOM_GIB, stage=f"SEC authority month {month}"
            )
        coverage_path = staging / "source_coverage.parquet"
        coverage_manifest = write_canonical_artifact(
            coverage,
            coverage_path,
            artifact_type=SEC_AUTHORITY_COVERAGE_ARTIFACT_TYPE,
            audit=_coverage_audit(coverage),
            inputs={"request_sha256": request_sha256, "source_lineage_sha256": source_lineage_sha256},
            production_ready=False,
        )
        coverage_path.with_suffix(".parquet.lock").unlink(missing_ok=True)
        _rewrite_artifact_path(coverage_path, output_directory)
        coverage_manifest = _json_object(manifest_path_for(coverage_path))
        manifest: dict[str, object] = {
            "schema": SEC_AUTHORITY_MANIFEST_SCHEMA,
            "state": "complete",
            "request": request,
            "request_sha256": request_sha256,
            "source_lineage_sha256": source_lineage_sha256,
            "decision_identity_sha256": identity_hasher.hexdigest(),
            "decision_rows": decision_rows,
            "coverage_rows": len(coverage),
            "event_rows": len(events),
            "partition_count": len(partition_records),
            "partitions": partition_records,
            "production_ready": False,
            "research_only_reason": "historical SEC first-seen availability is not proven",
            "coverage_artifact": _artifact_record(coverage_path, coverage_manifest),
        }
        _atomic_json(staging / "_manifest.json", manifest)
        authority = {
            "schema": SEC_AUTHORITY_SCHEMA,
            "state": "complete",
            "manifest": "_manifest.json",
            "manifest_sha256": file_sha256(staging / "_manifest.json"),
            "request_sha256": request_sha256,
            "source_lineage_sha256": source_lineage_sha256,
            "decision_identity_sha256": identity_hasher.hexdigest(),
            "coverage_artifact_sha256": coverage_manifest["artifact_sha256"],
            "production_ready": False,
        }
        _atomic_json(staging / "_authority.json", authority)
        load_sec_filing_decision_authority(staging, require_production_ready=False)
        os.replace(staging, output_directory)
        assert_peak_memory_budget(hard_budget_gib=_MAXIMUM_MEMORY_GIB, headroom_gib=_MEMORY_HEADROOM_GIB, stage="SEC authority publication")
        return load_sec_filing_decision_authority(output_directory, require_production_ready=False)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_sec_filing_decision_authority(
    directory: Path,
    *,
    require_production_ready: bool = True,
) -> SecFilingDecisionAuthority:
    directory = directory.resolve()
    manifest_path = directory / "_manifest.json"
    manifest = _json_object(manifest_path)
    authority = _json_object(directory / "_authority.json")
    if manifest.get("schema") != SEC_AUTHORITY_MANIFEST_SCHEMA or manifest.get("state") != "complete":
        raise DataReadinessError("SEC filing authority manifest is incomplete")
    if require_production_ready:
        raise DataReadinessError("retrospective SEC filing authority is not production ready")
    if bool(manifest.get("production_ready")) or bool(authority.get("production_ready")):
        raise DataReadinessError("retrospective SEC authority cannot claim production readiness")
    if authority.get("schema") != SEC_AUTHORITY_SCHEMA or authority.get("manifest_sha256") != file_sha256(manifest_path):
        raise DataReadinessError("SEC filing authority does not verify")
    request = manifest.get("request")
    if not isinstance(request, dict) or _json_sha256(request) != manifest.get("request_sha256"):
        raise DataReadinessError("SEC authority request hash does not verify")
    if (
        request.get("availability_policy") != "sec_daily_swing_conservative_proxy"
        or request.get("historical_availability_proven") is not False
        or request.get("production_ready") is not False
    ):
        raise DataReadinessError("SEC authority research contract does not verify")
    _verify_source_lineage(request.get("source_artifacts"), manifest.get("source_lineage_sha256"))
    _verify_relation_lineage(request.get("identity_relation"))
    _verify_decision_source_lineage(request.get("decision_source"))
    records = manifest.get("partitions")
    if not isinstance(records, list) or not records or len(records) != _as_int(manifest.get("partition_count", -1)):
        raise DataReadinessError("SEC authority partition inventory is invalid")
    expected_files = {"_authority.json", "_manifest.json", "source_coverage.parquet", "source_coverage.parquet.manifest.json", "partitions"}
    if {path.name for path in directory.iterdir()} != expected_files:
        raise DataReadinessError("SEC authority root inventory does not verify")
    partition_records: list[Mapping[str, object]] = []
    hasher = hashlib.sha256()
    total_rows = 0
    prior_month = ""
    for raw in records:
        if not isinstance(raw, dict):
            raise DataReadinessError("SEC authority partition record is invalid")
        month = str(raw.get("month_et", ""))
        if month <= prior_month:
            raise DataReadinessError("SEC authority months are duplicated or unordered")
        prior_month = month
        path = _resolve_inside(directory, raw.get("path"))
        if raw.get("sha256") != file_sha256(path):
            raise DataReadinessError("SEC authority partition hash does not verify")
        rows, sessions = _verify_partition(path, month, hasher)
        if rows != int(raw.get("rows", -1)) or sessions != int(raw.get("sessions", -1)):
            raise DataReadinessError("SEC authority partition counts do not verify")
        total_rows += rows
        partition_records.append(raw)
    if (
        total_rows != _as_int(manifest.get("decision_rows", -1))
        or hasher.hexdigest() != manifest.get("decision_identity_sha256")
        or authority.get("decision_identity_sha256") != hasher.hexdigest()
    ):
        raise DataReadinessError("SEC authority decision identity does not replay")
    coverage_path = directory / "source_coverage.parquet"
    coverage, coverage_manifest = load_canonical_artifact(
        coverage_path, expected_type=SEC_AUTHORITY_COVERAGE_ARTIFACT_TYPE, allow_research=True
    )
    _verify_artifact(manifest.get("coverage_artifact"), coverage_path, coverage_manifest, len(coverage))
    if authority.get("coverage_artifact_sha256") != coverage_manifest.get("artifact_sha256") or len(coverage) != _as_int(
        manifest.get("coverage_rows", -1)
    ):
        raise DataReadinessError("SEC authority coverage lineage does not verify")
    _coverage_audit(coverage).raise_for_failure()
    return SecFilingDecisionAuthority(directory, tuple(partition_records), coverage, manifest, authority)


def attach_sec_filing_features(
    decisions: pd.DataFrame,
    authority: SecFilingDecisionAuthority | Path,
    *,
    require_production_ready: bool = True,
) -> pd.DataFrame:
    directory = authority if isinstance(authority, Path) else authority.directory
    loaded = load_sec_filing_decision_authority(directory, require_production_ready=require_production_ready)
    input_frame = _normalize_decisions(decisions)
    months = set(input_frame["decision_time_utc"].dt.tz_convert(_ET).dt.strftime("%Y-%m"))
    evidence = loaded.read_decisions(months)
    feature_columns = [column for column in evidence.columns if column not in _IDENTITY_COLUMNS]
    collisions = sorted(set(feature_columns).intersection(decisions.columns))
    if collisions:
        raise DataReadinessError("SEC attachment would overwrite columns: " + ", ".join(collisions))
    identity = input_frame.merge(
        evidence[list(_IDENTITY_COLUMNS)], on="decision_id", how="left", suffixes=("", "_authority"), validate="one_to_one", indicator=True
    )
    if bool(identity["_merge"].ne("both").any()):
        raise DataReadinessError("SEC authority has no exact decision row")
    for column in ("security_id", "ticker"):
        if bool(identity[column].astype(str).ne(identity[f"{column}_authority"].astype(str)).any()):
            raise DataReadinessError(f"SEC authority {column} conflicts with decision identity")
    if bool(
        pd.to_datetime(identity["decision_time_utc"], utc=True).ne(pd.to_datetime(identity["decision_time_utc_authority"], utc=True)).any()
    ):
        raise DataReadinessError("SEC authority decision time conflicts with decision identity")
    return decisions.copy().merge(
        evidence[["decision_id", *feature_columns]], on="decision_id", how="left", validate="one_to_one", sort=False
    )


def _normalize_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(_IDENTITY_COLUMNS).difference(frame.columns))
    if missing:
        raise DataReadinessError("SEC authority decisions missing columns: " + ", ".join(missing))
    output = frame.loc[:, list(_IDENTITY_COLUMNS)].copy()
    output["decision_id"] = output["decision_id"].astype(str).str.strip()
    output["security_id"] = output["security_id"].astype(str).str.strip()
    output["ticker"] = output["ticker"].astype(str).str.strip().str.upper().str.replace("/", ".", regex=False)
    output["decision_time_utc"] = _strict_utc(output["decision_time_utc"], "decision_time_utc")
    if bool(
        output["decision_id"].eq("").any()
        or output["decision_id"].duplicated().any()
        or output["security_id"].eq("").any()
        or output["ticker"].isin({"", "MARKET"}).any()
    ):
        raise DataReadinessError("SEC authority decision identity is invalid")
    return output.sort_values(["decision_time_utc", "decision_id"], kind="stable").reset_index(drop=True)


def _relation_index(frame: pd.DataFrame) -> dict[tuple[str, str], tuple[tuple[pd.Timestamp, pd.Timestamp | None, pd.Timestamp, str], ...]]:
    result = {}
    for key, group in frame.groupby(["security_id", "ticker"], sort=False):
        result[(str(key[0]), str(key[1]))] = tuple(
            (
                pd.Timestamp(row.effective_from_utc),
                None if pd.isna(row.effective_to_utc) else pd.Timestamp(row.effective_to_utc),
                pd.Timestamp(row.available_at_utc),
                str(row.sec_cik),
            )
            for row in group.itertuples(index=False)
        )
    return result


def _event_index(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(cik): group.sort_values(["feature_available_at_utc", "event_id"], kind="stable").reset_index(drop=True)
        for cik, group in frame.groupby("sec_cik", sort=False)
    }


def _coverage_index(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {str(cik): group.copy() for cik, group in frame.groupby("sec_cik", sort=False)}


def _aggregate_session(
    decisions: pd.DataFrame,
    relations: Mapping[tuple[str, str], tuple[tuple[pd.Timestamp, pd.Timestamp | None, pd.Timestamp, str], ...]],
    events: Mapping[str, pd.DataFrame],
    coverage: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in decisions.itertuples(index=False):
        decision_time = pd.Timestamp(row.decision_time_utc)
        candidates = [
            item
            for item in relations.get((str(row.security_id), str(row.ticker)), ())
            if item[0] <= decision_time and (item[1] is None or decision_time < item[1]) and item[2] <= decision_time
        ]
        if len(candidates) > 1:
            raise DataReadinessError("SEC identity relation is ambiguous at decision time")
        cik = candidates[0][3] if candidates else None
        output: dict[str, object] = {
            "decision_id": str(row.decision_id),
            "security_id": str(row.security_id),
            "ticker": str(row.ticker),
            "decision_time_utc": decision_time,
            "sec_identity_proven": cik is not None,
            "sec_cik": cik,
        }
        for window_name, duration in WINDOWS.items():
            if cik is None:
                _set_unknown_window(output, window_name)
                continue
            start = decision_time - duration
            known, coverage_start, coverage_end, coverage_completed = _coverage_state(coverage.get(cik), start, decision_time)
            output[f"sec_source_complete_{window_name}"] = known
            output[f"sec_source_coverage_start_utc_{window_name}"] = coverage_start
            output[f"sec_source_coverage_end_utc_{window_name}"] = coverage_end
            output[f"sec_source_coverage_collected_at_utc_{window_name}"] = coverage_completed
            if not known:
                _set_unknown_metrics(output, window_name)
                continue
            selected = _window_events(events.get(cik), start, decision_time)
            forms = selected["sec_form"].astype(str).str.upper() if not selected.empty else pd.Series(dtype=str)
            output[f"sec_filing_count_{window_name}"] = float(len(selected))
            output[f"sec_material_filing_count_{window_name}"] = float(forms.isin(MATERIAL_FORMS).sum())
            output[f"sec_amendment_filing_count_{window_name}"] = (
                float(selected["is_amendment"].fillna(False).astype(bool).sum()) if not selected.empty else 0.0
            )
            for name, form_set in FORM_FEATURES.items():
                output[f"sec_{name}_filing_count_{window_name}"] = float(forms.isin(form_set).sum())
            output[f"sec_latest_filing_accepted_at_utc_{window_name}"] = (
                pd.to_datetime(selected["accepted_at_utc"], utc=True).max() if not selected.empty else pd.NaT
            )
            output[f"sec_latest_filing_available_at_utc_{window_name}"] = (
                pd.to_datetime(selected["feature_available_at_utc"], utc=True).max() if not selected.empty else pd.NaT
            )
        records.append(output)
    return _coerce_decision_frame(pd.DataFrame.from_records(records, columns=_DECISION_COLUMNS))


def _coverage_state(frame: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> tuple[bool, object, object, object]:
    if frame is None or frame.empty:
        return False, pd.NaT, pd.NaT, pd.NaT
    successful = frame.loc[frame["status"].isin(["observed", "observed_empty"])].copy()
    if successful.empty:
        return False, pd.NaT, pd.NaT, pd.NaT
    successful["coverage_start"] = pd.to_datetime(successful["requested_start_utc"], utc=True)
    successful["coverage_end"] = pd.to_datetime(successful["requested_end_utc"], utc=True)
    successful["coverage_completed"] = pd.to_datetime(successful["completed_at_utc"], utc=True)
    successful = successful.loc[successful["coverage_end"].ge(start) & successful["coverage_start"].le(end)].sort_values("coverage_start")
    cursor = start
    used: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    for row in successful.loc[:, ["coverage_start", "coverage_end", "coverage_completed"]].itertuples(index=False):
        row_start = pd.Timestamp(row.coverage_start)
        row_end = pd.Timestamp(row.coverage_end)
        if row_end < cursor:
            continue
        if row_start > cursor:
            break
        cursor = max(cursor, row_end)
        used.append((row_start, row_end, pd.Timestamp(row.coverage_completed)))
        if cursor >= end:
            return (
                True,
                min(value[0] for value in used),
                max(value[1] for value in used),
                max(value[2] for value in used),
            )
    return False, pd.NaT, pd.NaT, pd.NaT


def _window_events(frame: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=["sec_form", "is_amendment", "accepted_at_utc", "feature_available_at_utc"])
    available = pd.to_datetime(frame["feature_available_at_utc"], utc=True)
    return frame.loc[available.gt(start) & available.le(end)]


def _set_unknown_window(output: dict[str, object], window: str) -> None:
    output[f"sec_source_complete_{window}"] = False
    output[f"sec_source_coverage_start_utc_{window}"] = pd.NaT
    output[f"sec_source_coverage_end_utc_{window}"] = pd.NaT
    output[f"sec_source_coverage_collected_at_utc_{window}"] = pd.NaT
    _set_unknown_metrics(output, window)


def _set_unknown_metrics(output: dict[str, object], window: str) -> None:
    output[f"sec_filing_count_{window}"] = np.nan
    output[f"sec_material_filing_count_{window}"] = np.nan
    output[f"sec_amendment_filing_count_{window}"] = np.nan
    for name in FORM_FEATURES:
        output[f"sec_{name}_filing_count_{window}"] = np.nan
    output[f"sec_latest_filing_accepted_at_utc_{window}"] = pd.NaT
    output[f"sec_latest_filing_available_at_utc_{window}"] = pd.NaT


def _coerce_decision_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for field in _AUTHORITY_ARROW_SCHEMA:
        if pa.types.is_timestamp(field.type):
            output[field.name] = pd.to_datetime(output[field.name], utc=True, errors="coerce")
        elif pa.types.is_boolean(field.type):
            output[field.name] = output[field.name].fillna(False).astype(bool)
        elif pa.types.is_floating(field.type):
            output[field.name] = pd.to_numeric(output[field.name], errors="coerce")
        else:
            output[field.name] = output[field.name].astype("string")
    return output.loc[:, list(_DECISION_COLUMNS)]


def _merge_events(collections: Sequence[SecFilingCollection]) -> pd.DataFrame:
    frames = [value.events for value in collections if not value.events.empty]
    if not frames:
        return collections[0].events.iloc[0:0].copy()
    merged = pd.concat(frames, ignore_index=True).sort_values(["first_seen_at_utc", "event_id"], kind="stable")
    content_columns = [
        "event_id",
        "security_id",
        "source_family",
        "source",
        "published_at_utc",
        "available_at_utc",
        "feature_available_at_utc",
        "sec_cik",
        "sec_form",
        "accession_number",
        "filing_date",
        "report_date",
        "primary_document",
        "file_number",
        "accepted_at_utc",
        "is_amendment",
        "availability_rule",
    ]
    retained = []
    for event_id, group in merged.groupby("event_id", sort=False):
        signatures = {_json_sha256(_json_compatible(row)) for row in group.loc[:, content_columns].to_dict(orient="records")}
        if len(signatures) != 1:
            raise DataReadinessError(f"SEC collections contain conflicting accession event: {event_id}")
        parent_values = {str(value) for value in group["amends_accession_number"].dropna().astype(str) if str(value).strip()}
        if len(parent_values) > 1:
            raise DataReadinessError(f"SEC collections disagree on amendment parent: {event_id}")
        retained_row = group.iloc[0].copy()
        if parent_values:
            retained_row["amends_accession_number"] = next(iter(parent_values))
        retained.append(retained_row)
    return pd.DataFrame(retained).sort_values(["sec_cik", "feature_available_at_utc", "event_id"], kind="stable").reset_index(drop=True)


def _merge_coverage(collections: Sequence[SecFilingCollection]) -> pd.DataFrame:
    merged = pd.concat([value.source_collections for value in collections], ignore_index=True)
    signatures = merged.apply(lambda row: _json_sha256(_json_compatible(row.to_dict())), axis=1)
    merged["source_collection_id"] = merged["collection_id"].astype(str)
    merged["_signature"] = signatures
    merged = merged.loc[~merged["_signature"].duplicated()].copy()
    duplicate_ids = merged["collection_id"].astype(str).duplicated(keep=False)
    merged.loc[duplicate_ids, "collection_id"] = merged.loc[duplicate_ids].apply(
        lambda row: _json_sha256(
            {
                "source_collection_id": str(row["source_collection_id"]),
                "generation_signature": str(row["_signature"]),
            }
        ),
        axis=1,
    )
    if merged["collection_id"].astype(str).duplicated().any():
        raise DataReadinessError("SEC coverage generation identity is not unique")
    return (
        merged.drop(columns="_signature")
        .sort_values(["sec_cik", "requested_start_utc", "requested_end_utc"], kind="stable")
        .reset_index(drop=True)
    )


def _validate_input_identity(events: pd.DataFrame, coverage: pd.DataFrame) -> None:
    if bool(coverage["sec_cik"].astype(str).str.fullmatch(r"\d{10}").eq(False).any()):
        raise DataReadinessError("SEC coverage contains invalid CIK identity")
    if not events.empty:
        if bool(events["source_family"].astype(str).ne(SEC_SOURCE_FAMILY).any()):
            raise DataReadinessError("SEC authority rejects non-SEC events")
        if bool(events["security_id"].astype(str).ne("cik:" + events["sec_cik"].astype(str)).any()):
            raise DataReadinessError("SEC authority input issuer identity is invalid")


def _decision_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    failures = int(frame.empty) + len(set(_DECISION_COLUMNS).difference(frame.columns))
    if failures == 0:
        failures += int(frame["decision_id"].astype(str).duplicated().sum())
        decisions = pd.to_datetime(frame["decision_time_utc"], utc=True, errors="coerce")
        failures += int(decisions.isna().sum())
        for window in WINDOWS:
            complete = frame[f"sec_source_complete_{window}"].astype(bool)
            count = pd.to_numeric(frame[f"sec_filing_count_{window}"], errors="coerce")
            latest = pd.to_datetime(frame[f"sec_latest_filing_available_at_utc_{window}"], utc=True, errors="coerce")
            failures += int((complete & count.isna()).sum()) + int((~complete & count.notna()).sum())
            failures += int((latest > decisions).fillna(False).sum())
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name="sec_decision_authority",
                status="pass" if failures == 0 else "fail",
                failures=failures,
                rows_checked=len(frame),
                detail="CIK relation, conservative availability, and explicit coverage verify",
            ),
        )
    )


def _coverage_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    required = {
        "collection_id",
        "sec_cik",
        "source_family",
        "requested_start_utc",
        "requested_end_utc",
        "completed_at_utc",
        "status",
        "row_count",
        "historical_availability_proven",
        "production_eligible",
    }
    failures = len(required.difference(frame.columns)) + int(frame.empty)
    if failures == 0:
        failures += int(frame["collection_id"].astype(str).duplicated().sum())
        failures += int(frame["source_family"].astype(str).ne(SEC_SOURCE_FAMILY).sum())
        failures += int(frame["historical_availability_proven"].astype(bool).sum()) + int(frame["production_eligible"].astype(bool).sum())
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name="sec_coverage_authority",
                status="pass" if failures == 0 else "fail",
                failures=failures,
                rows_checked=len(frame),
                detail="per-CIK SEC coverage preserves retrospective evidence",
            ),
        )
    )


def _verify_partition(path: Path, month: str, hasher: Any) -> tuple[int, int]:
    parquet = _PARQUET_FILE(path)
    if parquet.schema_arrow != _AUTHORITY_ARROW_SCHEMA:
        raise DataReadinessError("SEC authority partition schema does not verify")
    rows = 0
    prior_session = ""
    for index in range(parquet.num_row_groups):
        frame = parquet.read_row_group(index).to_pandas()
        _decision_audit(frame).raise_for_failure()
        sessions = set(pd.to_datetime(frame["decision_time_utc"], utc=True).dt.tz_convert(_ET).dt.date.astype(str))
        if len(sessions) != 1:
            raise DataReadinessError("SEC authority row group spans multiple sessions")
        session = next(iter(sessions))
        if not session.startswith(month) or session <= prior_session:
            raise DataReadinessError("SEC authority row groups are not chronological within month")
        prior_session = session
        _update_identity_hash(hasher, frame)
        rows += len(frame)
    return rows, parquet.num_row_groups


def _update_identity_hash(hasher: Any, frame: pd.DataFrame) -> None:
    ordered = frame.loc[:, list(_IDENTITY_COLUMNS)].sort_values(["decision_time_utc", "decision_id"], kind="stable")
    for row in ordered.itertuples(index=False):
        payload = {
            "decision_id": str(row.decision_id),
            "security_id": str(row.security_id),
            "ticker": str(row.ticker),
            "decision_time_utc": pd.Timestamp(row.decision_time_utc).isoformat(),
        }
        hasher.update(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
        hasher.update(b"\n")


def _identity_frame_sha256(frame: pd.DataFrame | None) -> str:
    if frame is None:
        raise DataReadinessError("SEC in-memory decision source is missing")
    hasher = hashlib.sha256()
    _update_identity_hash(hasher, frame)
    return hasher.hexdigest()


def _relation_lineage(value: pd.DataFrame | Path, frame: pd.DataFrame) -> dict[str, object]:
    semantic = _relation_semantic_sha256(frame)
    if isinstance(value, Path):
        path = value.resolve()
        return {"path": str(path), "sha256": file_sha256(path), "semantic_sha256": semantic}
    return {"path": None, "sha256": None, "semantic_sha256": semantic}


def _relation_semantic_sha256(frame: pd.DataFrame) -> str:
    return _json_sha256([_json_compatible(row) for row in frame.to_dict(orient="records")])


def _collection_lineage(collection: SecFilingCollection) -> dict[str, object]:
    return {
        "directory": str(collection.directory),
        "manifest_sha256": file_sha256(collection.directory / "_manifest.json"),
        "request_sha256": collection.manifest["request_sha256"],
        "event_artifact_sha256": collection.authority["event_artifact_sha256"],
        "coverage_artifact_sha256": collection.authority["coverage_artifact_sha256"],
        "raw_archive_sha256": collection.authority["raw_archive_sha256"],
        "production_ready": False,
    }


def _verify_source_lineage(value: object, expected_sha256: object) -> None:
    if not isinstance(value, list) or _json_sha256(value) != expected_sha256:
        raise DataReadinessError("SEC source lineage set does not verify")
    for record in value:
        if not isinstance(record, dict):
            raise DataReadinessError("SEC source lineage record is invalid")
        collection = load_sec_filing_collection(Path(str(record.get("directory"))))
        if _collection_lineage(collection) != record:
            raise DataReadinessError("SEC source collection no longer replays")


def _verify_relation_lineage(value: object) -> None:
    if not isinstance(value, dict):
        raise DataReadinessError("SEC relation lineage is invalid")
    raw_path = value.get("path")
    if raw_path is None:
        return
    path = Path(str(raw_path))
    if not path.is_file() or file_sha256(path) != value.get("sha256"):
        raise DataReadinessError("SEC relation source hash does not verify")
    if _relation_semantic_sha256(load_sec_identity_relations(path)) != value.get("semantic_sha256"):
        raise DataReadinessError("SEC relation semantic identity does not verify")


def _verify_decision_source_lineage(value: object) -> None:
    if not isinstance(value, dict):
        raise DataReadinessError("SEC decision source lineage is invalid")
    kind = value.get("kind")
    if kind == "in_memory":
        return
    path = Path(str(value.get("path")))
    if kind == "parquet_file":
        if not path.is_file() or file_sha256(path) != value.get("sha256"):
            raise DataReadinessError("SEC decision source file no longer verifies")
        return
    if kind == "parquet_directory":
        expected = value.get("files")
        actual = [{"path": str(item.relative_to(path)), "sha256": file_sha256(item)} for item in sorted(path.rglob("*.parquet"))]
        if expected != actual:
            raise DataReadinessError("SEC decision source directory no longer verifies")
        return
    raise DataReadinessError("SEC decision source kind is invalid")


def _artifact_record(path: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": manifest["artifact_sha256"],
        "manifest_sha256": file_sha256(manifest_path_for(path)),
        "rows": manifest["rows"],
    }


def _verify_artifact(record: object, path: Path, manifest: Mapping[str, object], rows: int) -> None:
    if not isinstance(record, dict) or record != _artifact_record(path, manifest) or int(record.get("rows", -1)) != rows:
        raise DataReadinessError("SEC authority child artifact does not verify")


def _rewrite_artifact_path(path: Path, output_directory: Path) -> None:
    child_path = manifest_path_for(path)
    child = _json_object(child_path)
    child["artifact_path"] = str((output_directory / path.name).resolve())
    _atomic_json(child_path, child)


def _resolve_inside(root: Path, raw: object) -> Path:
    path = (root / str(raw)).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise DataReadinessError("SEC authority partition escapes its root") from exc
    if not path.is_file():
        raise DataReadinessError(f"SEC authority partition is missing: {path}")
    return path


def _strict_utc(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise DataReadinessError(f"SEC {name} contains invalid timestamps")
    return parsed


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"SEC authority JSON is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"SEC authority JSON must be an object: {path}")
    return {str(key): item for key, item in value.items()}


def _json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(value)
        return None if pd.isna(timestamp) else timestamp.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or pd.isna(value):
        return None
    return value


def _json_sha256(value: object) -> str:
    encoded = json.dumps(_json_compatible(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
