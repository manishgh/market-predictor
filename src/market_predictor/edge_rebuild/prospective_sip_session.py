"""Immutable post-close SIP source authority for one prospective XNYS session."""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any, Final, cast

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.history_collection import (
    collect_intraday_history,
    load_complete_intraday_history_collection,
)
from market_predictor.intraday.contracts.history_collection import (
    INTRADAY_HISTORY_PLAN_SCHEMA,
    SELECTED_SESSION_BENCHMARK_PLAN_SCHEMA,
    IntradayHistoryConfig,
    IntradayTransportConfig,
    SelectedSessionBenchmarkConfig,
)
from market_predictor.intraday.datasets.history import (
    PLAN_AUTHORITY_SCHEMA,
    SELECTED_SESSION_BENCHMARK_PLAN_AUTHORITY_SCHEMA,
    chunk_request_symbols,
    file_record,
    load_complete_intraday_history_plan,
    request_unit_record,
    stable_identity_hash,
    write_plan_json,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.universe.sp500.observed_membership_authority import (
    ObservedMembershipAuthority,
    load_observed_sp500_membership_authority,
)

REQUEST_SCHEMA: Final = "edge_rebuild.prospective_sip_session_request.v1"
MANIFEST_SCHEMA: Final = "edge_rebuild.prospective_sip_session_manifest.v1"
AUTHORITY_SCHEMA: Final = "edge_rebuild.prospective_sip_session_authority.v1"
REQUIRED_BENCHMARKS: Final = {
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
SourceFactory = Callable[[], AlpacaSource]


def collect_prospective_sip_session(
    *,
    session_date: date,
    membership_authority_directory: Path,
    five_minute_policy_path: Path,
    benchmark_policy_path: Path,
    output_directory: Path,
    five_minute_config: IntradayHistoryConfig,
    benchmark_config: SelectedSessionBenchmarkConfig,
    source_factory: SourceFactory,
    maximum_units_this_run: int | None = None,
    now_utc: datetime | None = None,
) -> dict[str, Any]:
    """Collect exact post-close bars and publish one non-model source authority."""

    if (output_directory / "_authority.json").exists():
        return load_complete_prospective_sip_session(output_directory)
    if maximum_units_this_run is not None and maximum_units_this_run < 1:
        raise ValueError("maximum_units_this_run must be positive")
    _validate_prospective_resource_policy(five_minute_config)
    _validate_prospective_resource_policy(benchmark_config)
    _guard_memory(five_minute_config, "prospective SIP session start")
    observed = load_observed_sp500_membership_authority(
        membership_authority_directory
    )
    session, open_at, close_at, next_open = _closed_session_bounds(
        session_date,
        calendar_name=five_minute_config.calendar,
        finalization_delay_seconds=(
            five_minute_config.intraday_finalization_delay_seconds
        ),
        now_utc=now_utc,
    )
    if benchmark_config.calendar != five_minute_config.calendar:
        raise DataReadinessError("prospective SIP calendars differ")
    if (
        benchmark_config.intraday_finalization_delay_seconds
        != five_minute_config.intraday_finalization_delay_seconds
    ):
        raise DataReadinessError("prospective SIP finalization delays differ")
    benchmarks = benchmark_config.normalized_benchmarks()
    if set(benchmarks) != REQUIRED_BENCHMARKS:
        raise DataReadinessError("prospective SIP benchmark set is incomplete")
    _validate_membership_observation(
        observed,
        session=session,
        open_at=open_at,
        calendar_name=five_minute_config.calendar,
    )
    active = _session_membership(
        observed,
        open_at=open_at,
        minimum_cross_section=five_minute_config.minimum_session_cross_section,
    )
    membership_parent = _membership_parent(
        membership_authority_directory,
        observed,
    )
    request_payload: dict[str, Any] = {
        "schema": REQUEST_SCHEMA,
        "session_date_et": session.date().isoformat(),
        "session_open_utc": open_at.isoformat(),
        "session_close_utc": close_at.isoformat(),
        "next_session_open_utc": next_open.isoformat(),
        "finalization_delay_seconds": (
            five_minute_config.intraday_finalization_delay_seconds
        ),
        "membership_parent": membership_parent,
        "five_minute_policy_path": str(five_minute_policy_path),
        "five_minute_policy_file_sha256": file_sha256(
            five_minute_policy_path
        ),
        "five_minute_policy_sha256": five_minute_config.sha256(),
        "benchmark_policy_path": str(benchmark_policy_path),
        "benchmark_policy_file_sha256": file_sha256(benchmark_policy_path),
        "benchmark_policy_sha256": benchmark_config.sha256(),
        "price_feed": "sip",
        "adjustment": "all",
        "sort": "asc",
        "asof_date": session.date().isoformat(),
        "full_cohort_symbols": active["ticker"].astype(str).tolist(),
        "full_cohort_security_ids": {
            str(row.ticker): str(row.security_id)
            for row in active.itertuples(index=False)
        },
        "benchmark_symbols": list(benchmarks),
        "minimum_session_cross_section": (
            five_minute_config.minimum_session_cross_section
        ),
        "maximum_full_cohort_incomplete_fraction": 0.05,
        "resource_policy": {
            "maximum_process_memory_gib": max(
                five_minute_config.maximum_process_memory_gib,
                benchmark_config.maximum_process_memory_gib,
            ),
            "maximum_collection_workers": max(
                five_minute_config.collection_workers,
                benchmark_config.collection_workers,
            ),
        },
        "training_eligible": False,
        "serving_eligible": False,
    }
    request_sha256 = _json_sha256(request_payload)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(
        output_directory / "_request.json",
        {**request_payload, "request_sha256": request_sha256},
    )
    plans = output_directory / "plans"
    five_plan = plans / "full_cohort_5m"
    benchmark_plan = plans / "benchmarks_1m"
    _publish_or_verify_plan(
        output_directory=five_plan,
        schema=INTRADAY_HISTORY_PLAN_SCHEMA,
        authority_schema=PLAN_AUTHORITY_SCHEMA,
        policy_path=five_minute_policy_path,
        policy_sha256=five_minute_config.sha256(),
        parent_request_sha256=request_sha256,
        membership_parent=membership_parent,
        session=session,
        open_at=open_at,
        close_at=close_at,
        symbols=active["ticker"].astype(str).tolist(),
        timeframe="5Min",
        maximum_symbols_per_unit=five_minute_config.maximum_symbols_per_unit,
        maximum_expected_rows_per_unit=(
            five_minute_config.maximum_expected_rows_per_unit
        ),
        session_membership=active,
    )
    _publish_or_verify_plan(
        output_directory=benchmark_plan,
        schema=SELECTED_SESSION_BENCHMARK_PLAN_SCHEMA,
        authority_schema=SELECTED_SESSION_BENCHMARK_PLAN_AUTHORITY_SCHEMA,
        policy_path=benchmark_policy_path,
        policy_sha256=benchmark_config.sha256(),
        parent_request_sha256=request_sha256,
        membership_parent=membership_parent,
        session=session,
        open_at=open_at,
        close_at=close_at,
        symbols=list(benchmarks),
        timeframe="1Min",
        maximum_symbols_per_unit=benchmark_config.maximum_symbols_per_unit,
        maximum_expected_rows_per_unit=(
            benchmark_config.maximum_expected_rows_per_unit
        ),
        session_membership=None,
    )
    collections = output_directory / "collections"
    five_result = _collect_or_load_child(
        plan_directory=five_plan,
        policy_path=five_minute_policy_path,
        output_directory=collections / "full_cohort_5m",
        config=five_minute_config,
        source_factory=source_factory,
        maximum_units_this_run=maximum_units_this_run,
    )
    if five_result["status"] != "transport_complete":
        return _publish_status(
            output_directory,
            request_sha256=request_sha256,
            status="source_incomplete",
            full_cohort=five_result,
            benchmarks=None,
            five_minute_config=five_minute_config,
        )
    release_process_memory()
    _guard_memory(five_minute_config, "prospective SIP five-minute complete")
    benchmark_result = _collect_or_load_child(
        plan_directory=benchmark_plan,
        policy_path=benchmark_policy_path,
        output_directory=collections / "benchmarks_1m",
        config=benchmark_config,
        source_factory=source_factory,
        maximum_units_this_run=maximum_units_this_run,
    )
    if benchmark_result["status"] != "transport_complete":
        return _publish_status(
            output_directory,
            request_sha256=request_sha256,
            status="source_incomplete",
            full_cohort=five_result,
            benchmarks=benchmark_result,
            five_minute_config=five_minute_config,
        )
    _closed_session_bounds(
        session_date,
        calendar_name=five_minute_config.calendar,
        finalization_delay_seconds=(
            five_minute_config.intraday_finalization_delay_seconds
        ),
        now_utc=now_utc,
    )
    ready_at = close_at + pd.Timedelta(
        seconds=five_minute_config.intraday_finalization_delay_seconds
    )
    _validate_child_retrieval_window(
        five_result,
        ready_at=ready_at,
        next_open=next_open,
    )
    _validate_child_retrieval_window(
        benchmark_result,
        ready_at=ready_at,
        next_open=next_open,
    )
    coverage = _source_coverage_summary(five_result, benchmark_result)
    if set(_symbol_coverage(five_result)) != set(
        request_payload["full_cohort_symbols"]
    ):
        raise DataReadinessError(
            "prospective SIP full-cohort coverage identity changed"
        )
    if coverage["status"] not in {"complete", "acceptable_with_exclusions"}:
        return _publish_status(
            output_directory,
            request_sha256=request_sha256,
            status="source_incomplete_coverage",
            full_cohort=five_result,
            benchmarks=benchmark_result,
            five_minute_config=five_minute_config,
        )
    manifest = _publish_status(
        output_directory,
        request_sha256=request_sha256,
        status="source_complete_warmup_ineligible",
        full_cohort=five_result,
        benchmarks=benchmark_result,
        five_minute_config=five_minute_config,
    )
    _atomic_json(output_directory / "_manifest.json", manifest)
    _atomic_json(
        output_directory / "_authority.json",
        {
            "schema": AUTHORITY_SCHEMA,
            "state": "source_complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(
                output_directory / "_manifest.json"
            ),
            "request_sha256": request_sha256,
        },
    )
    return load_complete_prospective_sip_session(output_directory)


def load_complete_prospective_sip_session(
    directory: Path,
) -> dict[str, Any]:
    """Strictly replay both exact child collections and their parent lineage."""

    request = _load_json(directory / "_request.json")
    manifest = _load_json(directory / "_manifest.json")
    status = _load_json(directory / "_status.json")
    authority = _load_json(directory / "_authority.json")
    payload = {key: value for key, value in request.items() if key != "request_sha256"}
    request_sha256 = _json_sha256(payload)
    if (
        request.get("schema") != REQUEST_SCHEMA
        or request.get("request_sha256") != request_sha256
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != "source_complete_warmup_ineligible"
        or status != manifest
        or manifest.get("request_sha256") != request_sha256
        or manifest.get("training_eligible") is not False
        or manifest.get("serving_eligible") is not False
        or manifest.get("selection_eligible") is not False
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("state") != "source_complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256")
        != file_sha256(directory / "_manifest.json")
        or authority.get("request_sha256") != request_sha256
    ):
        raise DataReadinessError("prospective SIP session envelope is invalid")
    membership_directory = Path(
        str(cast(Mapping[str, Any], request["membership_parent"])["directory"])
    )
    observed = load_observed_sp500_membership_authority(membership_directory)
    if _membership_parent(membership_directory, observed) != request["membership_parent"]:
        raise DataReadinessError("prospective SIP membership parent changed")
    session, open_at, close_at, next_open = _calendar_session_bounds(
        date.fromisoformat(str(request["session_date_et"])),
        calendar_name="XNYS",
    )
    if (
        request.get("session_open_utc") != open_at.isoformat()
        or request.get("session_close_utc") != close_at.isoformat()
        or request.get("next_session_open_utc") != next_open.isoformat()
    ):
        raise DataReadinessError("prospective SIP calendar bounds changed")
    _validate_membership_observation(
        observed,
        session=session,
        open_at=open_at,
        calendar_name="XNYS",
    )
    active = _session_membership(
        observed,
        open_at=open_at,
        minimum_cross_section=int(request["minimum_session_cross_section"]),
    )
    resource_policy = cast(Mapping[str, Any], request["resource_policy"])
    if (
        float(resource_policy.get("maximum_process_memory_gib", 5.0)) > 4.0
        or int(resource_policy.get("maximum_collection_workers", 3)) > 2
    ):
        raise DataReadinessError("prospective SIP resource policy changed")
    replayed_symbols = active["ticker"].astype(str).tolist()
    replayed_security_ids = {
        str(row.ticker): str(row.security_id)
        for row in active.itertuples(index=False)
    }
    if (
        request.get("full_cohort_symbols") != replayed_symbols
        or request.get("full_cohort_security_ids") != replayed_security_ids
    ):
        raise DataReadinessError("prospective SIP membership cohort changed")
    five_plan = directory / "plans" / "full_cohort_5m"
    benchmark_plan = directory / "plans" / "benchmarks_1m"
    _verify_child_plan_identity(
        five_plan,
        parent_request=request,
        expected_schema=INTRADAY_HISTORY_PLAN_SCHEMA,
        expected_timeframe="5Min",
        expected_symbols=replayed_symbols,
    )
    _verify_child_plan_identity(
        benchmark_plan,
        parent_request=request,
        expected_schema=SELECTED_SESSION_BENCHMARK_PLAN_SCHEMA,
        expected_timeframe="1Min",
        expected_symbols=cast(list[str], request["benchmark_symbols"]),
    )
    load_complete_intraday_history_plan(five_plan)
    load_complete_intraday_history_plan(benchmark_plan)
    full = load_complete_intraday_history_collection(
        directory / "collections" / "full_cohort_5m"
    )
    benchmarks = load_complete_intraday_history_collection(
        directory / "collections" / "benchmarks_1m"
    )
    _verify_child_collection_lineage(
        plan_directory=five_plan,
        collection_directory=directory / "collections" / "full_cohort_5m",
    )
    _verify_child_collection_lineage(
        plan_directory=benchmark_plan,
        collection_directory=directory / "collections" / "benchmarks_1m",
    )
    ready_at = close_at + pd.Timedelta(
        seconds=int(request["finalization_delay_seconds"])
    )
    _validate_child_retrieval_window(
        full,
        ready_at=ready_at,
        next_open=next_open,
    )
    _validate_child_retrieval_window(
        benchmarks,
        ready_at=ready_at,
        next_open=next_open,
    )
    expected_coverage = _source_coverage_summary(full, benchmarks)
    if set(_symbol_coverage(full)) != set(replayed_symbols):
        raise DataReadinessError(
            "prospective SIP full-cohort coverage identity changed"
        )
    if (
        float(request.get("maximum_full_cohort_incomplete_fraction", -1.0))
        != 0.05
        or expected_coverage["status"]
        not in {"complete", "acceptable_with_exclusions"}
        or manifest.get("coverage") != expected_coverage
        or manifest.get("coverage_status") != expected_coverage["status"]
    ):
        raise DataReadinessError("prospective SIP coverage evidence changed")
    if (
        manifest.get("full_cohort_manifest_sha256")
        != file_sha256(directory / "collections" / "full_cohort_5m" / "_manifest.json")
        or manifest.get("benchmark_manifest_sha256")
        != file_sha256(directory / "collections" / "benchmarks_1m" / "_manifest.json")
        or int(manifest.get("full_cohort_rows", -1)) != int(full["total_rows"])
        or int(manifest.get("benchmark_rows", -1)) != int(benchmarks["total_rows"])
    ):
        raise DataReadinessError("prospective SIP child authority changed")
    expected_top_level = {
        "_request.json",
        "_status.json",
        "_manifest.json",
        "_authority.json",
        "plans",
        "collections",
    }
    if {path.name for path in directory.iterdir()} != expected_top_level:
        raise DataReadinessError("prospective SIP top-level inventory changed")
    return manifest


def _publish_or_verify_plan(
    *,
    output_directory: Path,
    schema: str,
    authority_schema: str,
    policy_path: Path,
    policy_sha256: str,
    parent_request_sha256: str,
    membership_parent: Mapping[str, Any],
    session: pd.Timestamp,
    open_at: pd.Timestamp,
    close_at: pd.Timestamp,
    symbols: list[str],
    timeframe: str,
    maximum_symbols_per_unit: int,
    maximum_expected_rows_per_unit: int,
    session_membership: pd.DataFrame | None,
) -> None:
    expected_rows = int((close_at - open_at).total_seconds() // 60)
    if timeframe == "5Min":
        expected_rows //= 5
    request: dict[str, Any] = {
        "schema": schema,
        "policy_path": str(policy_path),
        "policy_file_sha256": file_sha256(policy_path),
        "policy_sha256": policy_sha256,
        "parent_request_sha256": parent_request_sha256,
        "membership_parent": dict(membership_parent),
        "session_date_et": session.date().isoformat(),
        "session_open_utc": open_at.isoformat(),
        "session_close_utc": close_at.isoformat(),
        "timeframe": timeframe,
        "symbols": symbols,
        "price_feed": "sip",
        "adjustment": "all",
        "sort": "asc",
        "training_performed": False,
        "download_performed": False,
    }
    fingerprint = _json_sha256(request)
    expected_request = {**request, "plan_fingerprint": fingerprint}
    if output_directory.exists():
        load_complete_intraday_history_plan(output_directory)
        if _load_json(output_directory / "_request.json") != expected_request:
            raise DataReadinessError(
                "prospective SIP child plan differs from the requested session"
            )
        return
    units: list[dict[str, object]] = []
    for chunk, mapping in chunk_request_symbols(
        symbols,
        expected_bars_per_symbol=expected_rows,
        maximum_symbols_per_unit=maximum_symbols_per_unit,
        maximum_expected_rows_per_unit=maximum_expected_rows_per_unit,
        label=f"{session.date()} {timeframe}",
    ):
        unit_id = stable_identity_hash(
            fingerprint,
            session.date().isoformat(),
            open_at.isoformat(),
            close_at.isoformat(),
            *sorted(mapping),
            timeframe,
            "sip",
            "all",
        )
        units.append(
            request_unit_record(
                unit_id=unit_id,
                session_date=session.date(),
                start=open_at,
                end=close_at,
                chunk=chunk,
                mapping=mapping,
                expected_bars_per_symbol=expected_rows,
                plan_fingerprint=fingerprint,
                timeframe=timeframe,
            )
        )
    temporary = output_directory.with_name(
        f".{output_directory.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir(parents=True)
    try:
        files: list[dict[str, Any]] = []
        units_path = temporary / "units" / timeframe / f"{session.strftime('%Y-%m')}.parquet"
        units_path.parent.mkdir(parents=True)
        pd.DataFrame(units).to_parquet(units_path, index=False)
        files.append(file_record(units_path, temporary, len(units)))
        if session_membership is not None:
            membership_path = temporary / "session_memberships" / f"{session.strftime('%Y-%m')}.parquet"
            membership_path.parent.mkdir(parents=True)
            session_membership.to_parquet(membership_path, index=False)
            files.append(
                file_record(membership_path, temporary, len(session_membership))
            )
        request["plan_fingerprint"] = fingerprint
        write_plan_json(temporary / "_request.json", request)
        files.append(file_record(temporary / "_request.json", temporary, 1))
        manifest = {
            "schema": schema,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "plan_fingerprint": fingerprint,
            "policy_sha256": policy_sha256,
            "research_only": True,
            "promotion_eligible": False,
            "acquisition": {
                "provider": "alpaca",
                "calendar": "XNYS",
                "calendar_version": version("exchange-calendars"),
                "price_feed": "sip",
                "adjustment": "all",
                "timeframe": timeframe,
            },
            "summary": {
                "session_date_et": session.date().isoformat(),
                "symbols": len(symbols),
                "acquisition_units": len(units),
                "expected_rows_per_symbol": expected_rows,
            },
            "files": sorted(files, key=lambda item: str(item["path"])),
        }
        write_plan_json(temporary / "_manifest.json", manifest)
        write_plan_json(
            temporary / "_authority.json",
            {
                "schema": authority_schema,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(temporary / "_manifest.json"),
                "plan_fingerprint": fingerprint,
            },
        )
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_directory)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _collect_or_load_child(
    *,
    plan_directory: Path,
    policy_path: Path,
    output_directory: Path,
    config: IntradayTransportConfig,
    source_factory: SourceFactory,
    maximum_units_this_run: int | None,
) -> dict[str, Any]:
    if (output_directory / "_authority.json").exists():
        return load_complete_intraday_history_collection(output_directory)
    resolved_plan = plan_directory.resolve()
    return collect_intraday_history(
        plan_directory=resolved_plan,
        policy_path=policy_path,
        output_directory=output_directory,
        config=config,
        source_factory=source_factory,
        maximum_units_this_run=maximum_units_this_run,
    )


def _verify_child_plan_identity(
    directory: Path,
    *,
    parent_request: Mapping[str, Any],
    expected_schema: str,
    expected_timeframe: str,
    expected_symbols: list[str],
) -> None:
    request = _load_json(directory / "_request.json")
    fingerprint = str(request.get("plan_fingerprint", ""))
    payload = {
        key: value for key, value in request.items() if key != "plan_fingerprint"
    }
    policy_prefix = (
        "five_minute" if expected_timeframe == "5Min" else "benchmark"
    )
    expected = {
        "schema": expected_schema,
        "policy_path": parent_request[f"{policy_prefix}_policy_path"],
        "policy_file_sha256": parent_request[
            f"{policy_prefix}_policy_file_sha256"
        ],
        "policy_sha256": parent_request[f"{policy_prefix}_policy_sha256"],
        "parent_request_sha256": parent_request["request_sha256"],
        "membership_parent": parent_request["membership_parent"],
        "session_date_et": parent_request["session_date_et"],
        "session_open_utc": parent_request["session_open_utc"],
        "session_close_utc": parent_request["session_close_utc"],
        "timeframe": expected_timeframe,
        "symbols": expected_symbols,
        "price_feed": "sip",
        "adjustment": "all",
        "sort": "asc",
        "training_performed": False,
        "download_performed": False,
    }
    if payload != expected or fingerprint != _json_sha256(expected):
        raise DataReadinessError("prospective SIP child plan identity changed")


def _verify_child_collection_lineage(
    *,
    plan_directory: Path,
    collection_directory: Path,
) -> None:
    plan_request = _load_json(plan_directory / "_request.json")
    collection_request = _load_json(collection_directory / "_request.json")
    plan_unit_ids = {
        str(unit_id)
        for path in (plan_directory / "units").rglob("*.parquet")
        for unit_id in pd.read_parquet(path, columns=["unit_id"])["unit_id"]
    }
    collection_manifest = _load_json(collection_directory / "_manifest.json")
    artifacts = collection_manifest.get("artifacts")
    if not isinstance(artifacts, list) or any(
        not isinstance(value, Mapping) for value in artifacts
    ):
        raise DataReadinessError(
            "prospective SIP child collection inventory is malformed"
        )
    collection_unit_ids = {
        str(cast(Mapping[str, Any], value).get("unit_id", ""))
        for value in artifacts
    }
    if (
        collection_request.get("plan_fingerprint")
        != plan_request.get("plan_fingerprint")
        or collection_request.get("plan_manifest_sha256")
        != file_sha256(plan_directory / "_manifest.json")
        or Path(str(collection_request.get("plan_path", ""))).resolve()
        != plan_directory.resolve()
        or collection_unit_ids != plan_unit_ids
    ):
        raise DataReadinessError(
            "prospective SIP child collection does not belong to its plan"
        )


def _session_membership(
    observed: ObservedMembershipAuthority,
    *,
    open_at: pd.Timestamp,
    minimum_cross_section: int,
) -> pd.DataFrame:
    frame = observed.memberships.copy()
    for column in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    active = frame[
        frame["effective_from_utc"].le(open_at)
        & (frame["effective_to_utc"].isna() | frame["effective_to_utc"].gt(open_at))
        & frame["available_at_utc"].le(open_at)
    ].sort_values("ticker", kind="stable")
    if (
        len(active) < minimum_cross_section
        or bool(active["ticker"].astype(str).str.strip().eq("").any())
        or bool(active["ticker"].duplicated().any())
        or bool(active["security_id"].duplicated().any())
        or bool(active["security_id"].astype(str).str.strip().eq("").any())
    ):
        raise DataReadinessError("prospective SIP membership cohort is invalid")
    return active.reset_index(drop=True)


def _validate_membership_observation(
    observed: ObservedMembershipAuthority,
    *,
    session: pd.Timestamp,
    open_at: pd.Timestamp,
    calendar_name: str,
) -> None:
    calendar = xcals.get_calendar(calendar_name)
    previous_session = calendar.previous_session(session)
    previous_close = pd.Timestamp(
        calendar.session_close(previous_session)
    ).tz_convert("UTC")
    observed_at = pd.Timestamp(observed.manifest["observed_at_utc"])
    if observed_at.tzinfo is None:
        raise DataReadinessError("membership observation time is timezone-naive")
    observed_at = observed_at.tz_convert("UTC")
    if not previous_close < observed_at <= open_at:
        raise DataReadinessError(
            "membership authority was not freshly observed before session open"
        )


def _closed_session_bounds(
    session_date: date,
    *,
    calendar_name: str,
    finalization_delay_seconds: int,
    now_utc: datetime | None,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    session, open_at, close_at, next_open = _calendar_session_bounds(
        session_date,
        calendar_name=calendar_name,
    )
    observed_now = pd.Timestamp(now_utc or datetime.now(UTC)).tz_convert("UTC")
    ready_at = close_at + pd.Timedelta(seconds=finalization_delay_seconds)
    if observed_now < ready_at:
        raise DataReadinessError("XNYS session bars have not finalized")
    if observed_now >= next_open:
        raise DataReadinessError(
            "prospective SIP collection window closed at the next XNYS open"
        )
    return session, open_at, close_at, next_open


def _calendar_session_bounds(
    session_date: date,
    *,
    calendar_name: str,
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    calendar = xcals.get_calendar(calendar_name)
    try:
        session = calendar.date_to_session(pd.Timestamp(session_date), direction="none")
    except ValueError as exc:
        raise DataReadinessError("requested date is not an XNYS session") from exc
    open_at = pd.Timestamp(calendar.session_open(session)).tz_convert("UTC")
    close_at = pd.Timestamp(calendar.session_close(session)).tz_convert("UTC")
    next_session = calendar.next_session(session)
    next_open = pd.Timestamp(calendar.session_open(next_session)).tz_convert("UTC")
    return session, open_at, close_at, next_open


def _membership_parent(
    directory: Path,
    observed: ObservedMembershipAuthority,
) -> dict[str, Any]:
    manifest = cast(Mapping[str, Any], observed.manifest)
    artifact = cast(Mapping[str, Any], manifest["membership_artifact"])
    membership_path = directory / str(artifact["path"])
    return {
        "directory": str(directory.resolve()),
        "authority_sha256": file_sha256(directory / "_authority.json"),
        "manifest_sha256": file_sha256(directory / "_manifest.json"),
        "membership_sha256": file_sha256(membership_path),
        "observed_at_utc": manifest["observed_at_utc"],
        "effective_horizon_date": manifest["effective_horizon_date"],
        "universe_snapshot_id": str(
            observed.memberships["universe_snapshot_id"].iloc[-1]
        ),
    }


def _publish_status(
    output_directory: Path,
    *,
    request_sha256: str,
    status: str,
    full_cohort: Mapping[str, Any],
    benchmarks: Mapping[str, Any] | None,
    five_minute_config: IntradayHistoryConfig,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "request_sha256": request_sha256,
        "status": status,
        "training_eligible": False,
        "serving_eligible": False,
        "selection_eligible": False,
        "selection_status": "warmup_incomplete",
        "required_prior_five_minute_sessions": 20,
        "full_cohort_status": full_cohort["status"],
        "benchmark_status": benchmarks["status"] if benchmarks else "not_started",
        "memory": memory_audit(
            hard_budget_gib=five_minute_config.maximum_process_memory_gib,
            headroom_gib=five_minute_config.memory_guard_headroom_gib,
        ).to_record(),
    }
    if full_cohort.get("status") == "transport_complete":
        payload.update(
            {
                "full_cohort_rows": int(full_cohort["total_rows"]),
                "full_cohort_manifest_sha256": file_sha256(
                    output_directory
                    / "collections"
                    / "full_cohort_5m"
                    / "_manifest.json"
                ),
            }
        )
    if benchmarks and benchmarks.get("status") == "transport_complete":
        payload.update(
            {
                "benchmark_rows": int(benchmarks["total_rows"]),
                "benchmark_manifest_sha256": file_sha256(
                    output_directory
                    / "collections"
                    / "benchmarks_1m"
                    / "_manifest.json"
                ),
            }
        )
    if (
        full_cohort.get("status") == "transport_complete"
        and benchmarks
        and benchmarks.get("status") == "transport_complete"
    ):
        coverage = _source_coverage_summary(full_cohort, benchmarks)
        payload["coverage"] = coverage
        payload["coverage_status"] = coverage["status"]
    else:
        payload["coverage_status"] = "not_evaluated"
    _atomic_json(output_directory / "_status.json", payload)
    return payload


def _validate_child_retrieval_window(
    child: Mapping[str, Any],
    *,
    ready_at: pd.Timestamp,
    next_open: pd.Timestamp,
) -> None:
    artifacts = child.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise DataReadinessError("prospective SIP child has no artifacts")
    retrieved: list[pd.Timestamp] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise DataReadinessError("prospective SIP child artifact is malformed")
        pages = artifact.get("pages")
        if not isinstance(pages, list) or not pages:
            raise DataReadinessError("prospective SIP child has no provider pages")
        for page in pages:
            if not isinstance(page, Mapping):
                raise DataReadinessError("prospective SIP child page is malformed")
            timestamp = pd.Timestamp(page.get("retrieved_at_utc"))
            if timestamp.tzinfo is None:
                raise DataReadinessError(
                    "prospective SIP child retrieval time is timezone-naive"
                )
            retrieved.append(timestamp.tz_convert("UTC"))
    if min(retrieved) < ready_at or max(retrieved) >= next_open:
        raise DataReadinessError(
            "provider pages were not retrieved inside the prospective post-close window"
        )


def _source_coverage_summary(
    full_cohort: Mapping[str, Any],
    benchmarks: Mapping[str, Any],
) -> dict[str, Any]:
    full = _symbol_coverage(full_cohort)
    benchmark = _symbol_coverage(benchmarks)
    if set(benchmark) != REQUIRED_BENCHMARKS:
        raise DataReadinessError("prospective SIP benchmark coverage identity changed")
    incomplete_full = sorted(
        ticker
        for ticker, value in full.items()
        if value.get("status") != "complete"
    )
    incomplete_benchmarks = sorted(
        ticker
        for ticker, value in benchmark.items()
        if value.get("status") != "complete"
    )
    incomplete_fraction = len(incomplete_full) / len(full) if full else 1.0
    status = "incomplete"
    if not incomplete_benchmarks and not incomplete_full:
        status = "complete"
    elif not incomplete_benchmarks and incomplete_fraction <= 0.05:
        status = "acceptable_with_exclusions"
    return {
        "status": status,
        "full_cohort_symbols": len(full),
        "full_cohort_incomplete_symbols": incomplete_full,
        "full_cohort_incomplete_fraction": incomplete_fraction,
        "benchmark_symbols": len(benchmark),
        "benchmark_incomplete_symbols": incomplete_benchmarks,
        "maximum_full_cohort_incomplete_fraction": 0.05,
        "missing_bar_policy": "preserve_observed_coverage_never_impute",
    }


def _symbol_coverage(
    child: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    artifacts = child.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise DataReadinessError("prospective SIP child has no coverage artifacts")
    combined: dict[str, Mapping[str, Any]] = {}
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise DataReadinessError("prospective SIP coverage artifact is malformed")
        coverage = artifact.get("symbol_coverage")
        if not isinstance(coverage, Mapping):
            raise DataReadinessError("prospective SIP symbol coverage is missing")
        for ticker, raw in coverage.items():
            if not isinstance(raw, Mapping) or str(ticker) in combined:
                raise DataReadinessError("prospective SIP symbol coverage is ambiguous")
            combined[str(ticker)] = cast(Mapping[str, Any], raw)
    return combined


def _guard_memory(config: IntradayHistoryConfig, stage: str) -> None:
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


def _validate_prospective_resource_policy(
    config: IntradayTransportConfig,
) -> None:
    if config.maximum_process_memory_gib > 4.0:
        raise DataReadinessError("prospective SIP memory limit exceeds 4 GiB")
    if config.collection_workers > 2:
        raise DataReadinessError("prospective SIP transport workers exceed two")


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if _load_json(path) != dict(payload):
            raise DataReadinessError("prospective SIP resume identity changed")
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


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"prospective SIP JSON is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"prospective SIP JSON is not an object: {path}")
    return {str(key): item for key, item in value.items()}


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode()
    ).hexdigest()
