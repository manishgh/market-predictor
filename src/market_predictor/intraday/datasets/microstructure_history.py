"""Immutable planning and resumable Alpaca SIP microstructure collection."""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import shutil
import threading
import tomllib
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Literal, cast

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.datasets.one_minute_coverage import (
    load_complete_one_minute_coverage,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.sources.alpaca import (
    AlpacaQuotesPage,
    AlpacaSource,
    AlpacaTradesPage,
)

PLAN_SCHEMA: Final = "edge_rebuild.intraday_microstructure_plan.v1"
PLAN_AUTHORITY_SCHEMA: Final = (
    "edge_rebuild.intraday_microstructure_plan_authority.v1"
)
COLLECTION_SCHEMA: Final = "edge_rebuild.intraday_microstructure_collection.v1"
COLLECTION_AUTHORITY_SCHEMA: Final = (
    "edge_rebuild.intraday_microstructure_collection_authority.v1"
)
JOB_SCHEMA: Final = "edge_rebuild.intraday_microstructure_job.v1"
ATTEMPT_SCHEMA: Final = "edge_rebuild.intraday_microstructure_attempt.v1"

EventType = Literal["trades", "quotes"]
SourceFactory = Callable[[], AlpacaSource]
_EVENT_TYPES: Final[tuple[EventType, ...]] = ("trades", "quotes")
_SAFE_RATE_HEADERS: Final = {
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
}
_COVERAGE_COLUMNS: Final = (
    "session_date_et",
    "session_open_utc",
    "session_close_utc",
    "ticker",
    "coverage_status",
)


@dataclass(frozen=True, slots=True)
class MicrostructureCollectionConfig:
    """Transport limits; the hard process-memory ceiling cannot exceed 4 GiB."""

    workers: int = 2
    retries: int = 5
    request_timeout_seconds: float = 60.0
    page_size: int = 10_000
    maximum_pages_per_job: int = 2_000
    maximum_process_memory_gib: float = 4.0
    memory_guard_headroom_gib: float = 0.75

    def __post_init__(self) -> None:
        if not 1 <= self.workers <= 4:
            raise ValueError("microstructure workers must be in [1, 4]")
        if not 1 <= self.retries <= 10:
            raise ValueError("microstructure retries must be in [1, 10]")
        if not 10 <= self.request_timeout_seconds <= 300:
            raise ValueError("microstructure timeout must be in [10, 300] seconds")
        if not 1 <= self.page_size <= 10_000:
            raise ValueError("microstructure page size must be in [1, 10000]")
        if not 1 <= self.maximum_pages_per_job <= 10_000:
            raise ValueError("microstructure page budget must be in [1, 10000]")
        if not 1 <= self.maximum_process_memory_gib <= 4:
            raise ValueError("microstructure process memory must be in [1, 4] GiB")
        if not 0 < self.memory_guard_headroom_gib < self.maximum_process_memory_gib:
            raise ValueError("microstructure memory headroom is invalid")


def load_microstructure_collection_config(
    path: Path,
) -> MicrostructureCollectionConfig:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(
            f"microstructure collection config is unreadable: {path}"
        ) from exc
    if raw.get("schema_version") != COLLECTION_SCHEMA:
        raise DataReadinessError("microstructure collection config schema differs")
    collection = raw.get("collection")
    expected = {field.name for field in fields(MicrostructureCollectionConfig)}
    if not isinstance(collection, dict) or set(collection) != expected:
        raise DataReadinessError("microstructure collection config fields differ")
    try:
        return MicrostructureCollectionConfig(**collection)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("microstructure collection config is invalid") from exc


def build_intraday_microstructure_plan(
    *,
    one_minute_coverage_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Publish one immutable unit per selected symbol/session coverage row."""

    _require_disjoint_trees(one_minute_coverage_directory, output_directory)
    if output_directory.exists():
        raise DataReadinessError(
            f"microstructure plan output must be new: {output_directory}"
        )
    coverage = load_complete_one_minute_coverage(one_minute_coverage_directory)
    if (
        coverage.get("status") != "ready"
        or coverage.get("ready_for_feature_build") is not True
    ):
        raise DataReadinessError("one-minute coverage is not ready")
    coverage_record = _exact_file_record(
        coverage.get("files"), "stock_session_coverage.parquet"
    )
    coverage_path = _resolve_inside(
        one_minute_coverage_directory, str(coverage_record["path"])
    )
    frame = pd.read_parquet(coverage_path, columns=list(_COVERAGE_COLUMNS))
    units = _plan_units(frame)
    coverage_counts = {
        str(key): int(value)
        for key, value in frame["coverage_status"].astype(str).value_counts().items()
    }
    unit_records = units.to_dict(orient="records")
    plan_fingerprint = _json_sha256({"units": unit_records})
    request_payload = {
        "schema": PLAN_SCHEMA,
        "one_minute_coverage_directory": str(
            one_minute_coverage_directory.resolve()
        ),
        "one_minute_coverage_authority_sha256": file_sha256(
            one_minute_coverage_directory / "_authority.json"
        ),
        "one_minute_coverage_manifest_sha256": file_sha256(
            one_minute_coverage_directory / "_manifest.json"
        ),
        "stock_session_coverage_sha256": str(coverage_record["sha256"]),
        "coverage_inclusion": "all_selected_stock_sessions_status_is_metadata",
        "event_types": list(_EVENT_TYPES),
        "price_feed": "sip",
        "plan_fingerprint": plan_fingerprint,
    }
    request = {**request_payload, "request_sha256": _json_sha256(request_payload)}
    staging = output_directory.with_name(
        f".{output_directory.name}.{uuid.uuid4().hex}.staging"
    )
    staging.mkdir(parents=True)
    try:
        units_path = staging / "units.parquet"
        _atomic_parquet(units, units_path)
        _atomic_json(staging / "_request.json", request)
        manifest = {
            **request,
            "status": "complete",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "units": len(units),
            "symbols": int(units["ticker"].nunique()),
            "first_session": str(units["session_date_et"].min()),
            "last_session": str(units["session_date_et"].max()),
            "jobs": len(units) * len(_EVENT_TYPES),
            "source_coverage_rows": len(frame),
            "included_stock_sessions_by_status": coverage_counts,
            "files": [
                _file_record(staging / "_request.json", staging, rows=1),
                _file_record(units_path, staging, rows=len(units)),
            ],
        }
        _atomic_json(staging / "_manifest.json", manifest)
        _atomic_json(
            staging / "_authority.json",
            {
                "schema": PLAN_AUTHORITY_SCHEMA,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(staging / "_manifest.json"),
                "request_sha256": request["request_sha256"],
                "plan_fingerprint": plan_fingerprint,
            },
        )
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_directory)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def load_complete_intraday_microstructure_plan(
    directory: Path,
) -> dict[str, Any]:
    """Verify plan authority, source authority, inventory, and deterministic units."""

    request = _read_json(directory / "_request.json")
    manifest_path = directory / "_manifest.json"
    manifest = _read_json(manifest_path)
    authority = _read_json(directory / "_authority.json")
    request_sha256 = _embedded_sha256(request)
    coverage_directory = Path(str(request.get("one_minute_coverage_directory", "")))
    coverage = load_complete_one_minute_coverage(coverage_directory)
    coverage_record = _exact_file_record(
        coverage.get("files"), "stock_session_coverage.parquet"
    )
    if (
        manifest.get("schema") != PLAN_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != PLAN_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("request_sha256") != request_sha256
        or authority.get("plan_fingerprint") != request.get("plan_fingerprint")
        or request.get("price_feed") != "sip"
        or request.get("event_types") != list(_EVENT_TYPES)
        or request.get("one_minute_coverage_authority_sha256")
        != file_sha256(coverage_directory / "_authority.json")
        or request.get("one_minute_coverage_manifest_sha256")
        != file_sha256(coverage_directory / "_manifest.json")
        or request.get("stock_session_coverage_sha256")
        != coverage_record.get("sha256")
    ):
        raise DataReadinessError("microstructure plan authority is invalid")
    records = manifest.get("files")
    request_record = _exact_file_record(records, "_request.json")
    units_record = _exact_file_record(records, "units.parquet")
    _verify_file_record(directory, request_record)
    units_path = _verify_file_record(directory, units_record)
    units = pd.read_parquet(units_path)
    _validate_plan_units(units)
    coverage_path = _resolve_inside(
        coverage_directory, str(coverage_record["path"])
    )
    coverage_frame = pd.read_parquet(
        coverage_path, columns=list(_COVERAGE_COLUMNS)
    )
    expected_units = _plan_units(coverage_frame)
    if not expected_units.astype(str).equals(units.astype(str)):
        raise DataReadinessError("microstructure plan differs from source coverage")
    expected_status_counts = {
        str(key): int(value)
        for key, value in coverage_frame["coverage_status"]
        .astype(str)
        .value_counts()
        .items()
    }
    fingerprint = _json_sha256({"units": units.to_dict(orient="records")})
    if (
        units_record.get("rows") != len(units)
        or manifest.get("units") != len(units)
        or manifest.get("jobs") != len(units) * len(_EVENT_TYPES)
        or manifest.get("symbols") != int(units["ticker"].nunique())
        or manifest.get("first_session") != str(units["session_date_et"].min())
        or manifest.get("last_session") != str(units["session_date_et"].max())
        or manifest.get("plan_fingerprint") != fingerprint
        or manifest.get("source_coverage_rows") != len(coverage_frame)
        or manifest.get("included_stock_sessions_by_status")
        != expected_status_counts
    ):
        raise DataReadinessError("microstructure plan content does not replay")
    _require_exact_inventory(
        directory,
        {"_request.json", "_manifest.json", "_authority.json", "units.parquet"},
    )
    return manifest


def collect_intraday_microstructure_history(
    *,
    plan_directory: Path,
    output_directory: Path,
    source_factory: SourceFactory,
    config: MicrostructureCollectionConfig | None = None,
    maximum_jobs_this_run: int,
) -> dict[str, Any]:
    """Collect bounded trade/quote jobs and publish only after complete replay."""

    active = config or MicrostructureCollectionConfig()
    if maximum_jobs_this_run < 1:
        raise ValueError("maximum_jobs_this_run must be positive")
    _require_disjoint_trees(plan_directory, output_directory)
    plan = load_complete_intraday_microstructure_plan(plan_directory)
    if (output_directory / "_authority.json").exists():
        raise DataReadinessError("completed microstructure collection is immutable")
    units = pd.read_parquet(plan_directory / "units.parquet")
    _validate_plan_units(units)
    request_payload = {
        "schema": COLLECTION_SCHEMA,
        "plan_directory": str(plan_directory.resolve()),
        "plan_authority_sha256": file_sha256(plan_directory / "_authority.json"),
        "plan_manifest_sha256": file_sha256(plan_directory / "_manifest.json"),
        "plan_fingerprint": plan["plan_fingerprint"],
        "provider": "alpaca",
        "price_feed": "sip",
        "event_types": list(_EVENT_TYPES),
        "transport": asdict(active),
    }
    request = {**request_payload, "request_sha256": _json_sha256(request_payload)}
    request_sha256 = str(request["request_sha256"])
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(output_directory / "_request.json", request)
    jobs = _job_specs(units)
    completed: dict[str, dict[str, Any]] = {}
    pending: list[dict[str, Any]] = []
    for job in jobs:
        existing = _load_existing_job(
            output_directory,
            job,
            plan_fingerprint=str(plan["plan_fingerprint"]),
            request_sha256=request_sha256,
        )
        if existing is None:
            pending.append(job)
        else:
            completed[str(job["job_id"])] = existing
    scheduled = pending[:maximum_jobs_this_run]
    _guard_memory(active, "microstructure collection start")
    local = threading.local()

    def get_source() -> AlpacaSource:
        source = getattr(local, "alpaca_source", None)
        if source is None:
            source = source_factory()
            settings = getattr(source, "settings", None)
            if str(getattr(settings, "alpaca_stock_feed", "")).lower() != "sip":
                raise DataReadinessError("microstructure collection requires Alpaca SIP")
            client = getattr(source, "client", None)
            if client is not None:
                client.timeout = int(active.request_timeout_seconds)
            local.alpaca_source = source
        return cast(AlpacaSource, source)

    def collect_job(job: Mapping[str, Any]) -> dict[str, Any]:
        return _collect_job(
            job,
            source=get_source(),
            root=output_directory,
            plan_fingerprint=str(plan["plan_fingerprint"]),
            request_sha256=request_sha256,
            config=active,
        )

    failures = _run_jobs(
        scheduled=scheduled,
        collect_job=collect_job,
        completed=completed,
        config=active,
    )
    unattempted = len(jobs) - len(completed) - len(failures)
    complete = not failures and unattempted == 0 and len(completed) == len(jobs)
    status = {
        "schema": COLLECTION_SCHEMA,
        "status": "transport_complete" if complete else "transport_incomplete",
        "ready_for_materialization": complete,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "request_sha256": request_sha256,
        "plan_fingerprint": plan["plan_fingerprint"],
        "requested_jobs": len(jobs),
        "completed_jobs": len(completed),
        "failed_jobs": failures,
        "unattempted_jobs": unattempted,
        "resumed_jobs": len(jobs) - len(pending),
        "memory": memory_audit(
            hard_budget_gib=active.maximum_process_memory_gib,
            headroom_gib=active.memory_guard_headroom_gib,
        ).to_record(),
    }
    _atomic_json(output_directory / "_status.json", status)
    if not complete:
        return status
    ordered_jobs = [completed[str(job["job_id"])] for job in jobs]
    rows_by_event = {event: 0 for event in _EVENT_TYPES}
    for expected, wrapper in zip(jobs, ordered_jobs, strict=True):
        _, payload = _verify_job_wrapper(
            output_directory,
            wrapper,
            expected,
            plan_fingerprint=str(plan["plan_fingerprint"]),
            request_sha256=request_sha256,
        )
        rows_by_event[cast(EventType, payload["event_type"])] += int(payload["rows"])
    attempt_count, raw_page_count = _attempt_summary(output_directory)
    data_file_count, data_inventory_sha256 = _collection_data_inventory(
        output_directory
    )
    manifest = {
        **status,
        "jobs": ordered_jobs,
        "attempt_count": attempt_count,
        "raw_page_count": raw_page_count,
        "data_file_count": data_file_count,
        "data_inventory_sha256": data_inventory_sha256,
        "rows_by_event": rows_by_event,
        "files": [
            _file_record(output_directory / "_request.json", output_directory, rows=1),
            _file_record(output_directory / "_status.json", output_directory, rows=1),
        ],
    }
    _atomic_json(output_directory / "_manifest.json", manifest)
    _verify_collection_content(
        output_directory,
        request=request,
        manifest=manifest,
        plan=plan,
        authority_file_count=0,
    )
    _atomic_json(
        output_directory / "_authority.json",
        {
            "schema": COLLECTION_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(output_directory / "_manifest.json"),
            "request_sha256": request_sha256,
            "plan_fingerprint": plan["plan_fingerprint"],
            "ready_for_materialization": True,
        },
    )
    load_complete_intraday_microstructure_collection(output_directory)
    return manifest


def load_complete_intraday_microstructure_collection(
    directory: Path,
) -> dict[str, Any]:
    """Strictly replay a complete collection and every raw provider page."""

    request = _read_json(directory / "_request.json")
    manifest_path = directory / "_manifest.json"
    manifest = _read_json(manifest_path)
    authority = _read_json(directory / "_authority.json")
    request_sha256 = _embedded_sha256(request)
    plan_directory = Path(str(request.get("plan_directory", "")))
    plan = load_complete_intraday_microstructure_plan(plan_directory)
    if (
        manifest.get("schema") != COLLECTION_SCHEMA
        or manifest.get("status") != "transport_complete"
        or manifest.get("ready_for_materialization") is not True
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != COLLECTION_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("request_sha256") != request_sha256
        or authority.get("plan_fingerprint") != plan.get("plan_fingerprint")
        or authority.get("ready_for_materialization") is not True
        or request.get("price_feed") != "sip"
        or request.get("event_types") != list(_EVENT_TYPES)
        or request.get("plan_authority_sha256")
        != file_sha256(plan_directory / "_authority.json")
        or request.get("plan_manifest_sha256")
        != file_sha256(plan_directory / "_manifest.json")
        or request.get("plan_fingerprint") != plan.get("plan_fingerprint")
    ):
        raise DataReadinessError("microstructure collection authority is invalid")
    _verify_collection_content(
        directory,
        request=request,
        manifest=manifest,
        plan=plan,
        authority_file_count=1,
    )
    return manifest


def _verify_collection_content(
    directory: Path,
    *,
    request: Mapping[str, Any],
    manifest: Mapping[str, Any],
    plan: Mapping[str, Any],
    authority_file_count: int,
) -> None:
    request_sha256 = _embedded_sha256(request)
    plan_directory = Path(str(request["plan_directory"]))
    if (
        manifest.get("schema") != COLLECTION_SCHEMA
        or manifest.get("status") != "transport_complete"
        or manifest.get("ready_for_materialization") is not True
        or manifest.get("request_sha256") != request_sha256
    ):
        raise DataReadinessError("microstructure collection content is invalid")
    units = pd.read_parquet(plan_directory / "units.parquet")
    expected_jobs = _job_specs(units)
    raw_jobs = manifest.get("jobs")
    if not isinstance(raw_jobs, list) or len(raw_jobs) != len(expected_jobs):
        raise DataReadinessError("microstructure collection job inventory differs")
    rows_by_event = {event: 0 for event in _EVENT_TYPES}
    for expected, raw in zip(expected_jobs, raw_jobs, strict=True):
        if not isinstance(raw, Mapping):
            raise DataReadinessError("microstructure job record is malformed")
        verified, job_payload = _verify_job_wrapper(
            directory,
            raw,
            expected,
            plan_fingerprint=str(plan["plan_fingerprint"]),
            request_sha256=request_sha256,
        )
        del verified
        rows_by_event[cast(EventType, job_payload["event_type"])] += int(
            job_payload["rows"]
        )
    attempt_count, raw_page_count = _attempt_summary(directory)
    data_file_count, data_inventory_sha256 = _collection_data_inventory(directory)
    files = manifest.get("files")
    _verify_file_record(directory, _exact_file_record(files, "_request.json"))
    _verify_file_record(directory, _exact_file_record(files, "_status.json"))
    if (
        manifest.get("requested_jobs") != len(expected_jobs)
        or manifest.get("completed_jobs") != len(expected_jobs)
        or manifest.get("failed_jobs") != {}
        or manifest.get("unattempted_jobs") != 0
        or manifest.get("rows_by_event") != rows_by_event
        or manifest.get("attempt_count") != attempt_count
        or manifest.get("raw_page_count") != raw_page_count
        or manifest.get("data_file_count") != data_file_count
        or manifest.get("data_inventory_sha256") != data_inventory_sha256
    ):
        raise DataReadinessError("microstructure collection summary differs")
    expected_file_count = (
        3
        + authority_file_count
        + data_file_count
    )
    observed_file_count = sum(1 for path in directory.rglob("*") if path.is_file())
    if observed_file_count != expected_file_count:
        raise DataReadinessError("microstructure collection file inventory differs")


def _plan_units(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(_COVERAGE_COLUMNS).difference(frame.columns)
    if missing:
        raise DataReadinessError(
            f"one-minute coverage lacks microstructure fields: {sorted(missing)}"
        )
    work = frame.loc[:, list(_COVERAGE_COLUMNS)].copy()
    work["session_date_et"] = pd.to_datetime(
        work["session_date_et"], errors="raise"
    ).dt.date.astype(str)
    work["ticker"] = work["ticker"].astype(str).str.upper().str.strip()
    work["session_open_utc"] = pd.to_datetime(
        work["session_open_utc"], utc=True, errors="raise"
    )
    work["session_close_utc"] = pd.to_datetime(
        work["session_close_utc"], utc=True, errors="raise"
    )
    work["source_bar_coverage_status"] = (
        work["coverage_status"].astype(str).str.strip()
    )
    work = work.drop(columns=["coverage_status"])
    work = work.sort_values(["session_date_et", "ticker"], kind="stable").reset_index(drop=True)
    if (
        work.empty
        or bool(work["ticker"].eq("").any())
        or bool(work["session_open_utc"].ge(work["session_close_utc"]).any())
        or bool(work.duplicated(["session_date_et", "ticker"]).any())
    ):
        raise DataReadinessError("one-minute coverage cannot form microstructure units")
    work["requested_start_utc"] = work["session_open_utc"].map(
        lambda value: pd.Timestamp(value).isoformat()
    )
    work["requested_end_utc"] = work["session_close_utc"].map(
        lambda value: pd.Timestamp(value).isoformat()
    )
    work = work.drop(columns=["session_open_utc", "session_close_utc"])
    work["unit_id"] = [
        _json_sha256(
            {
                "session_date_et": row.session_date_et,
                "ticker": row.ticker,
                "requested_start_utc": row.requested_start_utc,
                "requested_end_utc": row.requested_end_utc,
                "source_bar_coverage_status": row.source_bar_coverage_status,
            }
        )
        for row in work.itertuples(index=False)
    ]
    return work[
        [
            "unit_id",
            "session_date_et",
            "ticker",
            "requested_start_utc",
            "requested_end_utc",
            "source_bar_coverage_status",
        ]
    ]


def _validate_plan_units(units: pd.DataFrame) -> None:
    expected = {
        "unit_id",
        "session_date_et",
        "ticker",
        "requested_start_utc",
        "requested_end_utc",
        "source_bar_coverage_status",
    }
    if set(units.columns) != expected:
        raise DataReadinessError("microstructure plan unit schema differs")
    replayed = _plan_units(
        units.assign(
            session_open_utc=units["requested_start_utc"],
            session_close_utc=units["requested_end_utc"],
            coverage_status=units["source_bar_coverage_status"],
        ).drop(columns=[
            "unit_id",
            "requested_start_utc",
            "requested_end_utc",
            "source_bar_coverage_status",
        ])
    )
    observed = units.reset_index(drop=True).astype(str)
    if not replayed.astype(str).equals(observed):
        raise DataReadinessError("microstructure plan units do not replay")


def _job_specs(units: pd.DataFrame) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for row in units.itertuples(index=False):
        for event_type in _EVENT_TYPES:
            job_id = _json_sha256(
                {"unit_id": str(row.unit_id), "event_type": event_type}
            )
            jobs.append(
                {
                    "job_id": job_id,
                    "unit_id": str(row.unit_id),
                    "event_type": event_type,
                    "session_date_et": str(row.session_date_et),
                    "ticker": str(row.ticker),
                    "requested_start_utc": str(row.requested_start_utc),
                    "requested_end_utc": str(row.requested_end_utc),
                    "source_bar_coverage_status": str(
                        row.source_bar_coverage_status
                    ),
                }
            )
    return jobs


def _run_jobs(
    *,
    scheduled: Sequence[dict[str, Any]],
    collect_job: Callable[[Mapping[str, Any]], dict[str, Any]],
    completed: dict[str, dict[str, Any]],
    config: MicrostructureCollectionConfig,
) -> dict[str, str]:
    failures: dict[str, str] = {}
    rows = iter(scheduled)
    futures: dict[Future[dict[str, Any]], str] = {}
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        for _ in range(config.workers):
            row = next(rows, None)
            if row is not None:
                futures[executor.submit(collect_job, row)] = str(row["job_id"])
        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                job_id = futures.pop(future)
                try:
                    completed[job_id] = future.result()
                except Exception as exc:
                    failures[job_id] = f"{type(exc).__name__}: {str(exc)[:500]}"
                release_process_memory()
                _guard_memory(config, f"microstructure persist {job_id}")
                row = next(rows, None)
                if row is not None:
                    futures[executor.submit(collect_job, row)] = str(row["job_id"])
    return failures


def _collect_job(
    job: Mapping[str, Any],
    *,
    source: AlpacaSource,
    root: Path,
    plan_fingerprint: str,
    request_sha256: str,
    config: MicrostructureCollectionConfig,
) -> dict[str, Any]:
    event_type = cast(EventType, job["event_type"])
    job_id = str(job["job_id"])
    ticker = str(job["ticker"])
    start = _aware_datetime(job["requested_start_utc"])
    end = _aware_datetime(job["requested_end_utc"])
    asof = pd.Timestamp(job["session_date_et"]).date()
    attempt_id = uuid.uuid4().hex
    attempt_path = root / "attempts" / event_type / job_id / f"{attempt_id}.json"
    pages, resumed_from_attempt = _resume_job_pages(
        root,
        job,
        event_type=event_type,
        ticker=ticker,
        requested_start=start,
        requested_end=end,
    )
    observed_fields, first_timestamp, last_timestamp = _summarize_saved_pages(
        root,
        pages,
        event_type=event_type,
        ticker=ticker,
        requested_start=start,
        requested_end=end,
    )
    page_token = cast(
        str | None,
        pages[-1].get("next_page_token") if pages else None,
    )
    seen_tokens = {
        str(page["request_page_token"])
        for page in pages
        if page.get("request_page_token") is not None
    }
    started_at = datetime.now(UTC)
    attempt_base = {
        "schema": ATTEMPT_SCHEMA,
        "attempt_id": attempt_id,
        "job_id": job_id,
        "event_type": event_type,
        "ticker": ticker,
        "started_at_utc": started_at.isoformat(),
        "resumed_from_attempt_path": resumed_from_attempt,
    }
    _atomic_json(
        attempt_path,
        {**attempt_base, "status": "in_progress", "pages": pages},
    )
    try:
        collection_complete = bool(pages) and page_token is None
        while not collection_complete:
            if len(pages) >= config.maximum_pages_per_job:
                raise DataReadinessError(
                    "microstructure job exceeded its page budget"
                )
            page = _fetch_event_page(
                source,
                event_type,
                ticker=ticker,
                start=start,
                end=end,
                page_token=page_token,
                asof=asof,
                config=config,
            )
            if page.request_page_token != page_token:
                raise DataReadinessError(
                    "Alpaca microstructure request page token differs"
                )
            rows = _event_rows(event_type, page)
            if set(rows).difference({ticker}):
                raise DataReadinessError("Alpaca returned an unexpected microstructure symbol")
            page_rows = 0
            regular_session_rows = 0
            for value in rows.get(ticker, ()):
                _validate_event_value(event_type, value)
                timestamp = value.get("t")
                if not isinstance(timestamp, str) or not timestamp:
                    raise DataReadinessError("Alpaca microstructure row lacks timestamp")
                observed = _aware_datetime(timestamp)
                if observed < start or observed > end:
                    raise DataReadinessError("Alpaca microstructure row is outside its session")
                if observed < end:
                    regular_session_rows += 1
                observed_fields.update(str(key) for key in value)
                normalized_timestamp = observed.isoformat()
                first_timestamp = (
                    min(first_timestamp, normalized_timestamp)
                    if first_timestamp
                    else normalized_timestamp
                )
                last_timestamp = (
                    max(last_timestamp, normalized_timestamp)
                    if last_timestamp
                    else normalized_timestamp
                )
                page_rows += 1
            raw_payload = page.raw_payload
            if not isinstance(raw_payload, Mapping):
                raise DataReadinessError("Alpaca microstructure raw page is not an object")
            page_number = len(pages) + 1
            raw_page_path = (
                root
                / "raw_pages"
                / event_type
                / job_id
                / attempt_id
                / f"page-{page_number:05d}.json.gz"
            )
            _atomic_gzip_json_stream(raw_page_path, raw_payload)
            page_record = {
                "page_number": page_number,
                "request_page_token": page.request_page_token,
                "next_page_token": page.next_page_token,
                "rows": page_rows,
                "regular_session_rows": regular_session_rows,
                "response_sha256": _event_page_sha256(event_type, page),
                "raw_page_path": _relative(root, raw_page_path),
                "raw_page_sha256": file_sha256(raw_page_path),
                "raw_page_bytes": raw_page_path.stat().st_size,
                "rate_headers": {
                    key.lower(): value
                    for key, value in page.response_headers.items()
                    if key.lower() in _SAFE_RATE_HEADERS
                },
            }
            pages.append(page_record)
            _atomic_json(
                attempt_path,
                {**attempt_base, "status": "in_progress", "pages": pages},
            )
            next_token = page.next_page_token
            if next_token is None:
                collection_complete = True
                continue
            if next_token in seen_tokens:
                raise DataReadinessError("Alpaca repeated a microstructure page token")
            seen_tokens.add(next_token)
            page_token = next_token
        attempt = {
            **attempt_base,
            "status": "complete",
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "pages": pages,
        }
        _atomic_json(attempt_path, attempt)
        payload = {
            "schema": JOB_SCHEMA,
            **dict(job),
            "plan_fingerprint": plan_fingerprint,
            "request_sha256": request_sha256,
            "provider": "alpaca",
            "price_feed": "sip",
            "rows": sum(int(page["rows"]) for page in pages),
            "regular_session_rows": sum(
                int(page["regular_session_rows"]) for page in pages
            ),
            "pages": pages,
            "observed_fields": sorted(observed_fields),
            "first_timestamp": first_timestamp,
            "last_timestamp": last_timestamp,
            "successful_attempt_path": _relative(root, attempt_path),
            "successful_attempt_sha256": file_sha256(attempt_path),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        job_path = _job_path(root, job)
        _atomic_json(job_path, payload)
        return _job_wrapper(root, job_path, payload)
    except Exception as exc:
        _atomic_json(
            attempt_path,
            {
                **attempt_base,
                "status": "failed",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "pages": pages,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            },
        )
        raise


def _fetch_event_page(
    source: AlpacaSource,
    event_type: EventType,
    *,
    ticker: str,
    start: datetime,
    end: datetime,
    page_token: str | None,
    asof: Any,
    config: MicrostructureCollectionConfig,
) -> AlpacaTradesPage | AlpacaQuotesPage:
    method = (
        source.fetch_trades_page
        if event_type == "trades"
        else source.fetch_quotes_page
    )
    return method(
        (ticker,),
        start,
        end,
        page_token=page_token,
        asof=asof,
        limit=config.page_size,
        retries=config.retries,
    )


def _resume_job_pages(
    root: Path,
    job: Mapping[str, Any],
    *,
    event_type: EventType,
    ticker: str,
    requested_start: datetime,
    requested_end: datetime,
) -> tuple[list[dict[str, Any]], str | None]:
    attempt_root = root / "attempts" / event_type / str(job["job_id"])
    candidates: list[tuple[int, str, Path, list[dict[str, Any]]]] = []
    for path in sorted(attempt_root.glob("*.json")):
        attempt = _read_json(path)
        if attempt.get("status") == "in_progress":
            attempt = {
                **attempt,
                "status": "failed",
                "completed_at_utc": datetime.now(UTC).isoformat(),
                "error": "InterruptedError: prior process ended before completion",
            }
            _atomic_json(path, attempt)
        pages = attempt.get("pages")
        if (
            attempt.get("schema") != ATTEMPT_SCHEMA
            or attempt.get("job_id") != job.get("job_id")
            or attempt.get("event_type") != event_type
            or attempt.get("ticker") != ticker
            or attempt.get("status") not in {"failed", "in_progress", "complete"}
            or not isinstance(pages, list)
        ):
            raise DataReadinessError("microstructure resume attempt differs")
        normalized = [
            dict(cast(Mapping[str, Any], page))
            for page in pages
            if isinstance(page, Mapping)
        ]
        if len(normalized) != len(pages):
            raise DataReadinessError("microstructure resume pages are malformed")
        _verify_saved_page_sequence(
            root,
            normalized,
            event_type=event_type,
            ticker=ticker,
            requested_start=requested_start,
            requested_end=requested_end,
        )
        candidates.append(
            (
                len(normalized),
                str(attempt.get("completed_at_utc", attempt.get("started_at_utc", ""))),
                path,
                normalized,
            )
        )
    if not candidates:
        return [], None
    _, _, path, pages = max(candidates, key=lambda item: (item[0], item[1]))
    return pages, _relative(root, path)


def _verify_saved_page_sequence(
    root: Path,
    pages: Sequence[Mapping[str, Any]],
    *,
    event_type: EventType,
    ticker: str,
    requested_start: datetime,
    requested_end: datetime,
) -> None:
    expected_token: str | None = None
    for index, page in enumerate(pages, start=1):
        if (
            page.get("page_number") != index
            or page.get("request_page_token") != expected_token
        ):
            raise DataReadinessError("microstructure resume page sequence differs")
        _verify_raw_page(
            root,
            page,
            event_type=event_type,
            ticker=ticker,
            requested_start=requested_start,
            requested_end=requested_end,
        )
        expected_token = cast(str | None, page.get("next_page_token"))


def _summarize_saved_pages(
    root: Path,
    pages: Sequence[Mapping[str, Any]],
    *,
    event_type: EventType,
    ticker: str,
    requested_start: datetime,
    requested_end: datetime,
) -> tuple[set[str], str | None, str | None]:
    fields: set[str] = set()
    first: datetime | None = None
    last: datetime | None = None
    for page in pages:
        _, _, page_first, page_last = _verify_raw_page(
            root,
            page,
            event_type=event_type,
            ticker=ticker,
            requested_start=requested_start,
            requested_end=requested_end,
        )
        raw = _read_gzip_json(
            _resolve_inside(root, str(page["raw_page_path"]))
        )
        raw_rows = cast(Mapping[str, Any], raw[event_type])
        for value in cast(Sequence[Mapping[str, Any]], raw_rows.get(ticker, [])):
            fields.update(str(key) for key in value)
        if page_first is not None:
            first = min(first, page_first) if first else page_first
        if page_last is not None:
            last = max(last, page_last) if last else page_last
    return (
        fields,
        first.isoformat() if first else None,
        last.isoformat() if last else None,
    )


def _load_existing_job(
    root: Path,
    job: Mapping[str, Any],
    *,
    plan_fingerprint: str,
    request_sha256: str,
) -> dict[str, Any] | None:
    path = _job_path(root, job)
    if not path.exists():
        return None
    payload = _read_json(path)
    wrapper = _job_wrapper(root, path, payload)
    identity = {key: job[key] for key in job}
    attempt_path = _resolve_inside(
        root, str(payload.get("successful_attempt_path", ""))
    )
    attempt = _read_json(attempt_path) if attempt_path.is_file() else {}
    if (
        payload.get("schema") != JOB_SCHEMA
        or any(payload.get(key) != value for key, value in identity.items())
        or payload.get("plan_fingerprint") != plan_fingerprint
        or payload.get("request_sha256") != request_sha256
        or payload.get("provider") != "alpaca"
        or payload.get("price_feed") != "sip"
        or not attempt_path.is_file()
        or file_sha256(attempt_path) != payload.get("successful_attempt_sha256")
        or attempt.get("schema") != ATTEMPT_SCHEMA
        or attempt.get("status") != "complete"
        or attempt.get("job_id") != job.get("job_id")
    ):
        raise DataReadinessError("microstructure completed checkpoint differs")
    return wrapper


def _verify_job_wrapper(
    root: Path,
    raw: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    plan_fingerprint: str,
    request_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_path = _relative(root, _job_path(root, expected))
    if (
        raw.get("path") != expected_path
        or raw.get("job_id") != expected.get("job_id")
        or raw.get("event_type") != expected.get("event_type")
    ):
        raise DataReadinessError("microstructure job path differs")
    path = _verify_file_record(root, raw)
    payload = _read_json(path)
    identity = {key: expected[key] for key in expected}
    if (
        payload.get("schema") != JOB_SCHEMA
        or any(payload.get(key) != value for key, value in identity.items())
        or payload.get("plan_fingerprint") != plan_fingerprint
        or payload.get("request_sha256") != request_sha256
        or payload.get("provider") != "alpaca"
        or payload.get("price_feed") != "sip"
    ):
        raise DataReadinessError("microstructure job identity differs")
    pages = payload.get("pages")
    if not isinstance(pages, list) or not pages:
        raise DataReadinessError("microstructure job has no provider pages")
    total_rows = 0
    total_regular_rows = 0
    expected_token: str | None = None
    first_observed: datetime | None = None
    last_observed: datetime | None = None
    event_type = cast(EventType, payload["event_type"])
    ticker = str(payload["ticker"])
    requested_start = _aware_datetime(payload["requested_start_utc"])
    requested_end = _aware_datetime(payload["requested_end_utc"])
    for index, page in enumerate(pages, start=1):
        if (
            not isinstance(page, Mapping)
            or page.get("page_number") != index
            or page.get("request_page_token") != expected_token
        ):
            raise DataReadinessError("microstructure page sequence differs")
        page_count, regular_count, page_first, page_last = _verify_raw_page(
            root,
            page,
            event_type=event_type,
            ticker=ticker,
            requested_start=requested_start,
            requested_end=requested_end,
        )
        if page_count != int(page.get("rows", -1)):
            raise DataReadinessError("microstructure page row count differs")
        if regular_count != int(page.get("regular_session_rows", -1)):
            raise DataReadinessError("microstructure regular-session count differs")
        if page_first is not None:
            first_observed = min(first_observed, page_first) if first_observed else page_first
        if page_last is not None:
            last_observed = max(last_observed, page_last) if last_observed else page_last
        total_rows += int(page.get("rows", -1))
        total_regular_rows += int(page.get("regular_session_rows", -1))
        expected_token = cast(str | None, page.get("next_page_token"))
    attempt_path = _resolve_inside(
        root, str(payload.get("successful_attempt_path", ""))
    )
    attempt = _read_json(attempt_path) if attempt_path.is_file() else {}
    if (
        expected_token is not None
        or total_rows != payload.get("rows")
        or total_regular_rows != payload.get("regular_session_rows")
        or payload.get("first_timestamp")
        != (first_observed.isoformat() if first_observed else None)
        or payload.get("last_timestamp")
        != (last_observed.isoformat() if last_observed else None)
        or not attempt_path.is_file()
        or file_sha256(attempt_path) != payload.get("successful_attempt_sha256")
        or attempt.get("schema") != ATTEMPT_SCHEMA
        or attempt.get("status") != "complete"
        or attempt.get("job_id") != payload.get("job_id")
        or attempt.get("event_type") != event_type
        or attempt.get("ticker") != ticker
        or attempt.get("pages") != pages
    ):
        raise DataReadinessError("microstructure job completion differs")
    return dict(raw), payload


def _verify_raw_page(
    root: Path,
    page: Mapping[str, Any],
    *,
    event_type: EventType | None = None,
    ticker: str | None = None,
    requested_start: datetime | None = None,
    requested_end: datetime | None = None,
) -> tuple[int, int, datetime | None, datetime | None]:
    path = _resolve_inside(root, str(page.get("raw_page_path", "")))
    if (
        not path.is_file()
        or path.stat().st_size != int(page.get("raw_page_bytes", -1))
        or file_sha256(path) != page.get("raw_page_sha256")
    ):
        raise DataReadinessError("microstructure raw page failed integrity")
    if event_type is None or ticker is None:
        return 0, 0, None, None
    raw = _read_gzip_json(path)
    raw_rows = raw.get(event_type)
    if not isinstance(raw_rows, Mapping) or set(raw_rows).difference({ticker}):
        raise DataReadinessError("microstructure raw page symbol identity differs")
    values = raw_rows.get(ticker, [])
    if not isinstance(values, list) or any(not isinstance(value, Mapping) for value in values):
        raise DataReadinessError("microstructure raw page rows are malformed")
    next_value = raw.get("next_page_token")
    next_token = str(next_value).strip() if next_value is not None and str(next_value).strip() else None
    normalized = {
        str(symbol).upper(): tuple(
            {
                str(key): item
                for key, item in value.items()
            }
            for value in cast(Sequence[Mapping[str, Any]], raw_values)
        )
        for symbol, raw_values in raw_rows.items()
        if isinstance(raw_values, list)
    }
    expected_response = _json_sha256(
        {
            "event_type": event_type,
            "request_page_token": page.get("request_page_token"),
            "next_page_token": next_token,
            "rows": normalized,
        }
    )
    if (
        len(values) != int(page.get("rows", -1))
        or next_token != page.get("next_page_token")
        or expected_response != page.get("response_sha256")
    ):
        raise DataReadinessError("microstructure raw page content differs")
    first_timestamp: datetime | None = None
    last_timestamp: datetime | None = None
    regular_session_rows = 0
    for raw_value in values:
        value = cast(Mapping[str, Any], raw_value)
        _validate_event_value(event_type, value)
        observed = _aware_datetime(value["t"])
        if (
            requested_start is not None
            and requested_end is not None
            and (observed < requested_start or observed > requested_end)
        ):
            raise DataReadinessError("microstructure raw page timestamp is outside job")
        if requested_end is None or observed < requested_end:
            regular_session_rows += 1
        first_timestamp = min(first_timestamp, observed) if first_timestamp else observed
        last_timestamp = max(last_timestamp, observed) if last_timestamp else observed
    return len(values), regular_session_rows, first_timestamp, last_timestamp


def _attempt_summary(root: Path) -> tuple[int, int]:
    attempt_count = 0
    raw_page_paths: set[str] = set()
    for path in sorted((root / "attempts").rglob("*.json")):
        attempt = _read_json(path)
        event_type = str(attempt.get("event_type", ""))
        job_id = str(attempt.get("job_id", ""))
        attempt_id = str(attempt.get("attempt_id", ""))
        expected_path = (
            root / "attempts" / event_type / job_id / f"{attempt_id}.json"
        )
        if (
            attempt.get("schema") != ATTEMPT_SCHEMA
            or attempt.get("status") not in {"failed", "complete"}
            or path.resolve() != expected_path.resolve()
        ):
            raise DataReadinessError("microstructure attempt identity differs")
        pages = attempt.get("pages")
        if not isinstance(pages, list):
            raise DataReadinessError("microstructure attempt pages are malformed")
        for page in pages:
            if not isinstance(page, Mapping):
                raise DataReadinessError("microstructure attempt page is malformed")
            _verify_raw_page(root, page)
            raw_page_paths.add(str(page.get("raw_page_path", "")))
        attempt_count += 1
    observed_raw_pages = {
        _relative(root, path)
        for path in (root / "raw_pages").rglob("*.json.gz")
    }
    if raw_page_paths != observed_raw_pages:
        raise DataReadinessError("microstructure raw page inventory differs")
    return attempt_count, len(raw_page_paths)


def _collection_data_inventory(root: Path) -> tuple[int, str]:
    records: list[dict[str, Any]] = []
    for directory_name in ("jobs", "attempts", "raw_pages"):
        directory = root / directory_name
        if not directory.exists():
            continue
        for path in sorted(candidate for candidate in directory.rglob("*") if candidate.is_file()):
            records.append(
                {
                    "path": _relative(root, path),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return len(records), _json_sha256(records)


def _job_path(root: Path, job: Mapping[str, Any]) -> Path:
    event_type = str(job["event_type"])
    if event_type not in _EVENT_TYPES:
        raise DataReadinessError("microstructure event type is invalid")
    month = str(pd.Timestamp(job["session_date_et"]).strftime("%Y-%m"))
    job_id = str(job["job_id"])
    if not job_id or any(character not in "0123456789abcdef" for character in job_id):
        raise DataReadinessError("microstructure job id is invalid")
    return root / "jobs" / event_type / month / f"{job_id}.manifest.json"


def _job_wrapper(root: Path, path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "path": _relative(root, path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "job_id": payload["job_id"],
        "event_type": payload["event_type"],
    }


def _event_page_sha256(
    event_type: EventType,
    page: AlpacaTradesPage | AlpacaQuotesPage,
) -> str:
    rows = _event_rows(event_type, page)
    return _json_sha256(
        {
            "event_type": event_type,
            "request_page_token": page.request_page_token,
            "next_page_token": page.next_page_token,
            "rows": rows,
        }
    )


def _event_rows(
    event_type: EventType,
    page: AlpacaTradesPage | AlpacaQuotesPage,
) -> Mapping[str, Sequence[Mapping[str, Any]]]:
    if event_type == "trades":
        if not isinstance(page, AlpacaTradesPage):
            raise DataReadinessError("trade request returned a quote page")
        return page.trades
    if not isinstance(page, AlpacaQuotesPage):
        raise DataReadinessError("quote request returned a trade page")
    return page.quotes


def _exact_file_record(raw: object, expected: str) -> Mapping[str, Any]:
    if not isinstance(raw, list):
        raise DataReadinessError("artifact file inventory is malformed")
    matches = [
        item
        for item in raw
        if isinstance(item, Mapping) and item.get("path") == expected
    ]
    if len(matches) != 1:
        raise DataReadinessError(f"artifact requires exactly one {expected}")
    return matches[0]


def _file_record(path: Path, root: Path, *, rows: int) -> dict[str, Any]:
    return {
        "path": _relative(root, path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def _verify_file_record(root: Path, raw: Mapping[str, Any]) -> Path:
    path = _resolve_inside(root, str(raw.get("path", "")))
    if (
        not path.is_file()
        or path.stat().st_size != int(raw.get("bytes", -1))
        or file_sha256(path) != raw.get("sha256")
    ):
        raise DataReadinessError(f"artifact file failed integrity: {path}")
    return path


def _require_exact_inventory(root: Path, expected: set[str]) -> None:
    observed = {
        _relative(root, path)
        for path in root.rglob("*")
        if path.is_file()
    }
    if observed != expected:
        raise DataReadinessError("artifact file inventory differs")


def _embedded_sha256(record: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in record.items() if key != "request_sha256"}
    observed = record.get("request_sha256")
    expected = _json_sha256(payload)
    if observed != expected:
        raise DataReadinessError("artifact request identity differs")
    return expected


def _validate_event_value(
    event_type: EventType,
    value: Mapping[str, Any],
) -> None:
    required = (
        {"t", "p", "s", "x", "z"}
        if event_type == "trades"
        else {"t", "ap", "as", "ax", "bp", "bs", "bx", "z"}
    )
    if not required.issubset(value):
        raise DataReadinessError(
            f"Alpaca {event_type} row lacks required market identity"
        )
    try:
        numbers: tuple[Any, ...]
        if event_type == "trades":
            numbers = (value["p"], value["s"])
            invalid = float(value["p"]) <= 0.0 or float(value["s"]) <= 0.0
        else:
            numbers = (value["ap"], value["as"], value["bp"], value["bs"])
            invalid = (
                float(value["ap"]) < 0.0
                or float(value["bp"]) < 0.0
                or float(value["as"]) < 0.0
                or float(value["bs"]) < 0.0
            )
        finite = all(math.isfinite(float(number)) for number in numbers)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError(
            f"Alpaca {event_type} row has nonnumeric market values"
        ) from exc
    if invalid or not finite:
        raise DataReadinessError(
            f"Alpaca {event_type} row has invalid market values"
        )


def _read_gzip_json(path: Path) -> dict[str, Any]:
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(
            f"microstructure raw page is unreadable: {path}"
        ) from exc
    if not isinstance(value, dict):
        raise DataReadinessError("microstructure raw page is not an object")
    return {str(key): item for key, item in value.items()}


def _aware_datetime(value: object) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise DataReadinessError("microstructure timestamp is timezone-naive")
    return cast(
        datetime,
        timestamp.tz_convert("UTC").to_pydatetime(warn=False),
    )


def _relative(root: Path, path: Path) -> str:
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved_root not in resolved.parents:
        raise DataReadinessError("microstructure artifact escapes its root")
    return str(resolved.relative_to(resolved_root)).replace("\\", "/")


def _resolve_inside(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise DataReadinessError("microstructure artifact path is not relative")
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if resolved_root not in candidate.parents:
        raise DataReadinessError("microstructure artifact escapes its root")
    return candidate


def _require_disjoint_trees(left: Path, right: Path) -> None:
    resolved_left = left.resolve()
    resolved_right = right.resolve()
    if (
        resolved_left == resolved_right
        or resolved_left in resolved_right.parents
        or resolved_right in resolved_left.parents
    ):
        raise DataReadinessError("microstructure input and output trees overlap")


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"artifact JSON is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"artifact JSON is not an object: {path}")
    return {str(key): item for key, item in value.items()}


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if _read_json(path) != dict(payload):
            raise DataReadinessError(f"microstructure resume identity differs: {path}")
        return
    _atomic_json(path, payload)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_gzip_json_stream(path: Path, payload: Mapping[str, Any]) -> None:
    """Serialize one provider page incrementally; never build a unit-sized buffer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), default=str)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=6,
                fileobj=raw,
                mtime=0,
            ) as compressed:
                with io.TextIOWrapper(compressed, encoding="utf-8") as text:
                    for chunk in encoder.iterencode(dict(payload)):
                        text.write(chunk)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _guard_memory(config: MicrostructureCollectionConfig, stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
    assert_peak_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
