"""Strict replay and atomic publication of readiness authority evidence."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import shutil
from collections.abc import Mapping
from dataclasses import dataclass, replace
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Final, Literal, cast

import exchange_calendars as xcals
import pandas as pd

from market_predictor.core import path_integrity
from market_predictor.core.errors import DataReadinessError

HISTORICAL_RUN_SCHEMA: Final = "edge_rebuild.readiness.run.v1"
CURRENT_RUN_SCHEMA: Final = "edge_rebuild.readiness.run.v3"
HISTORICAL_AUTHORITY_SCHEMA: Final = "edge_rebuild.readiness.authority.v1"
CURRENT_AUTHORITY_SCHEMA: Final = "edge_rebuild.readiness.authority.v3"
HISTORICAL_SUMMARY_SCHEMA: Final = "edge_rebuild.readiness.v1"
CURRENT_SUMMARY_SCHEMA: Final = "edge_rebuild.readiness.v2"
STAGING_OWNER_SCHEMA: Final = "market_predictor.readiness_authority_staging_owner.v1"

_CSV_COLUMNS: Final[dict[str, tuple[str, ...]]] = {
    "blockers.csv": (
        "blocker_code",
        "scope",
        "blocks_er2",
        "required_action",
        "detail",
    ),
    "catalyst_readiness.csv": (
        "evidence_type",
        "evidence_value",
        "observed_count",
        "research_status",
        "promotion_status",
        "detail",
    ),
    "cost_readiness.csv": (
        "strategy_id",
        "rows",
        "minimum_cost_bps",
        "median_cost_bps",
        "mean_cost_bps",
        "maximum_cost_bps",
        "exact_cost_available",
        "adverse_fill_stress_available",
    ),
    "dimension_coverage.csv": (
        "strategy_id",
        "dimension",
        "value",
        "source_rows",
        "sessions",
        "tickers",
        "proxy_eligible_opportunities",
    ),
    "exclusion_reasons.csv": (
        "strategy_id",
        "reason",
        "excluded_rows",
        "total_rows",
    ),
    "fold_capacity.csv": (
        "strategy_id",
        "fold",
        "capacity_type",
        "first_test_session",
        "last_test_session",
        "test_sessions",
        "minimum_test_sessions_required",
        "status",
    ),
    "phase_capacity.csv": (
        "strategy_id",
        "phase",
        "sessions",
        "source_rows",
        "unique_decision_groups",
        "unique_tickers",
        "minimum_sessions_required",
        "status",
    ),
    "session_calendar.csv": (
        "strategy_id",
        "proxy_strategy_id",
        "session_date_et",
        "source_rows",
        "unique_decision_groups",
        "unique_tickers",
        "proxy_eligible_opportunities",
        "year",
        "er_setup_opportunities",
        "er_setup_status",
    ),
    "source_inventory.csv": (
        "strategy_id",
        "proxy_strategy_id",
        "source_role",
        "raw_rows",
        "source_usable_rows",
        "proxy_setup_rows",
        "proxy_label_eligible_rows",
        "er_setup_opportunities",
        "unique_decision_groups",
        "unique_tickers",
        "valid_sessions",
        "effective_session_blocks",
        "first_usable_decision_time_utc",
        "last_usable_decision_time_utc",
        "price_feed",
        "adjustment",
        "session_gate",
        "exact_new_horizon_labels",
        "source_authority",
        "coverage_exact_rate",
        "collection_model_data_ready",
        "point_in_time_membership_status",
        "benchmark_interval_status",
    ),
}
_MANIFEST_ARTIFACTS: Final = frozenset({"_request.json", "summary.json", *_CSV_COLUMNS})
_DIRECTORY_FILES: Final = _MANIFEST_ARTIFACTS | {
    "_manifest.json",
    "_authority.json",
}
_MANIFEST_KEYS: Final = frozenset({"schema", "request_sha256", "status", "artifacts", "created_at_utc"})
_AUTHORITY_KEYS: Final = frozenset({"schema", "state", "request_sha256", "artifact", "artifact_sha256"})
_ARTIFACT_KEYS: Final = frozenset({"path", "bytes", "sha256"})
_V1_REQUEST_KEYS: Final = frozenset(
    {
        "schema",
        "policy_sha256",
        "policy_file_sha256",
        "swing_policy_file_sha256",
        "intraday_policy_file_sha256",
        "sources",
        "implementation",
        "training_performed",
        "download_performed",
    }
)
_V3_REQUEST_KEYS: Final = frozenset(
    {
        "schema",
        "policy_sha256",
        "policy_file_sha256",
        "swing_training_policy_file_sha256",
        "strategy_contract_file_sha256",
        "intraday_policy_file_sha256",
        "sources",
        "implementation",
        "training_performed",
        "download_performed",
        "exchange_calendar",
    }
)
_SUMMARY_KEYS: Final = frozenset(
    {
        "schema",
        "request_sha256",
        "status",
        "er2_authorized",
        "training_performed",
        "download_performed",
        "models_created",
        "blocking_findings",
        "nonblocking_required_work",
        "acquisition_plan",
        "memory",
    }
)
_ACQUISITION_KEYS: Final = frozenset(
    {
        "authorized_by_audit",
        "scope",
        "provider",
        "feed",
        "timeframe",
        "adjustment",
        "minimum_additional_sessions",
        "target_history_sessions",
        "end_before",
        "reuse_existing_daily_and_catalyst_sources",
    }
)
_MEMORY_KEYS: Final = frozenset(
    {
        "current_working_set_gib",
        "hard_budget_gib",
        "peak_working_set_gib",
        "safety_threshold_gib",
    }
)
_SOURCE_KEYS: Final = frozenset({"swing", "intraday", "catalyst"})
_HISTORICAL_SWING_SOURCE_KEYS: Final = frozenset(
    {
        "type",
        "bundle_manifest_sha256",
        "bundle_request_sha256",
        "technical_manifest_sha256",
        "technical_artifact_sha256",
        "proxy_artifact_sha256",
        "technical_rows",
        "proxy_rows",
    }
)
_CURRENT_SWING_SOURCE_KEYS: Final = frozenset(
    {
        "type",
        "strategy_id",
        "horizon_sessions",
        "panel_manifest_sha256",
        "panel_authority_sha256",
        "panel_request_sha256",
        "candidate_status",
        "candidate_id",
        "candidate_authority_sha256",
        "promoted_bundle_status",
        "promoted_bundle_sha256",
        "technical_rows",
        "proxy_rows",
    }
)
_HISTORICAL_INTRADAY_SOURCE_KEYS: Final = frozenset(
    {
        "type",
        "training_manifest_sha256",
        "collection_manifest_sha256",
        "collection_request_sha256",
        "coverage_manifest_sha256",
        "coverage_fingerprint",
        "dataset_fingerprint",
        "proxy_dataset_sha256",
        "collection_rows",
        "proxy_rows",
    }
)
_CURRENT_INTRADAY_SOURCE_KEYS: Final = _HISTORICAL_INTRADAY_SOURCE_KEYS | {
    "strategy_id",
    "proxy_strategy_id",
}
_CATALYST_SOURCE_KEYS: Final = frozenset(
    {
        "type",
        "lineage_manifest_sha256",
        "lineage_request_sha256",
        "news_manifest_sha256",
        "news_request_sha256",
        "coverage_sha256",
        "source_event_rows",
    }
)
_HISTORICAL_IMPLEMENTATION_KEYS: Final = frozenset({"contracts", "readiness"})
_CURRENT_IMPLEMENTATION_KEYS: Final = frozenset(
    {
        "authority",
        "catalyst_source_verification",
        "contracts",
        "intraday_source_verification",
        "promotion_verification",
        "readiness",
        "swing_source_verification",
    }
)
_IMPLEMENTATION_RECORD_KEYS: Final = frozenset({"path", "sha256"})
_EXCHANGE_CALENDAR_KEYS: Final = frozenset({"name", "package", "version"})
_HEX = frozenset("0123456789abcdef")
_VALID_STATUSES: Final = frozenset({"blocked_pending_targeted_acquisition", "ready_for_ER2"})


@dataclass(frozen=True, slots=True)
class VerifiedReadinessAuthority:
    """One hash-verified readiness snapshot consumed without reopening files."""

    directory: Path
    format_version: Literal["historical_v1", "current_v3"]
    request: Mapping[str, object]
    manifest: Mapping[str, object]
    authority: Mapping[str, object]
    summary: Mapping[str, object]
    session_calendar: pd.DataFrame
    request_sha256: str
    manifest_sha256: str
    authority_sha256: str
    artifact_sha256: Mapping[str, str]


def current_exchange_calendar_identity() -> dict[str, str]:
    """Return the calendar implementation bound into current readiness evidence."""

    return {
        "name": "XNYS",
        "package": "exchange-calendars",
        "version": package_version("exchange-calendars"),
    }


def load_readiness_authority(
    directory: Path,
    *,
    expected_request_sha256: str | None = None,
    require_current: bool = False,
) -> VerifiedReadinessAuthority:
    """Strictly replay a historical v1 or current v3 readiness authority."""

    root = path_integrity.verify_tree_containment(
        directory,
        label="readiness authority",
    )
    actual_entries = {path.name for path in root.iterdir()}
    if actual_entries != _DIRECTORY_FILES:
        raise DataReadinessError("readiness authority file inventory differs")

    request_bytes = _read_confined(root, "_request.json")
    manifest_bytes = _read_confined(root, "_manifest.json")
    authority_bytes = _read_confined(root, "_authority.json")
    request = _json_object(request_bytes, "readiness request")
    manifest = _json_object(manifest_bytes, "readiness manifest")
    authority = _json_object(authority_bytes, "readiness authority")
    version = _validate_envelope_schemas(request, manifest, authority, require_current)
    _require_exact_keys(manifest, _MANIFEST_KEYS, "readiness manifest")
    _require_exact_keys(authority, _AUTHORITY_KEYS, "readiness authority")

    request_hash = _json_sha256(request)
    if expected_request_sha256 is not None and request_hash != expected_request_sha256:
        raise DataReadinessError("readiness request identity differs")
    if manifest.get("request_sha256") != request_hash:
        raise DataReadinessError("readiness manifest request identity differs")
    if authority.get("request_sha256") != request_hash:
        raise DataReadinessError("readiness authority request identity differs")
    if authority.get("state") != "complete" or authority.get("artifact") != "_manifest.json":
        raise DataReadinessError("readiness authority is not complete")
    manifest_hash = _bytes_sha256(manifest_bytes)
    if authority.get("artifact_sha256") != manifest_hash:
        raise DataReadinessError("readiness manifest identity differs")

    _validate_request(request, version)
    records = _artifact_records(manifest)
    verified_bytes: dict[str, bytes] = {}
    artifact_hashes: dict[str, str] = {}
    for name, record in records.items():
        payload = _read_confined(root, name)
        size = _strict_nonnegative_int(record.get("bytes"), f"{name} bytes")
        digest = _required_sha256(record.get("sha256"), f"{name} sha256")
        if len(payload) != size or _bytes_sha256(payload) != digest:
            raise DataReadinessError(f"readiness artifact does not verify: {name}")
        verified_bytes[name] = payload
        artifact_hashes[name] = digest

    summary = _json_object(verified_bytes["summary.json"], "readiness summary")
    _validate_summary(summary, version, request_hash)
    status = _required_text(summary.get("status"), "summary status")
    if manifest.get("status") != status:
        raise DataReadinessError("readiness manifest and summary status differ")
    if request.get("training_performed") != summary.get("training_performed"):
        raise DataReadinessError("readiness training status differs")
    if request.get("download_performed") != summary.get("download_performed"):
        raise DataReadinessError("readiness download status differs")

    evidence = {
        name: _read_csv_evidence(verified_bytes[name], columns, name)
        for name, columns in _CSV_COLUMNS.items()
    }
    _validate_evidence(evidence, summary=summary, request=request)
    session_calendar = evidence["session_calendar.csv"]

    return VerifiedReadinessAuthority(
        directory=root,
        format_version=version,
        request=request,
        manifest=manifest,
        authority=authority,
        summary=summary,
        session_calendar=session_calendar,
        request_sha256=request_hash,
        manifest_sha256=manifest_hash,
        authority_sha256=_bytes_sha256(authority_bytes),
        artifact_sha256=artifact_hashes,
    )


def publish_readiness_authority(
    *,
    output_directory: Path,
    request: Mapping[str, object],
    request_sha256: str,
    summary: Mapping[str, object],
    evidence: Mapping[str, pd.DataFrame],
) -> VerifiedReadinessAuthority:
    """Atomically publish the current readiness envelope or replay an existing one."""

    output = path_integrity.verify_no_reparse_ancestry(
        output_directory,
        label="readiness authority output",
    )
    if output.exists():
        return load_readiness_authority(
            output,
            expected_request_sha256=request_sha256,
            require_current=True,
        )
    if _json_sha256(dict(request)) != request_sha256:
        raise DataReadinessError("readiness publication request hash differs")
    _validate_request(request, "current_v3")
    _validate_summary(summary, "current_v3", request_sha256)
    _validate_evidence(evidence, summary=summary, request=request)

    output.parent.mkdir(parents=True, exist_ok=True)
    path_integrity.verify_no_reparse_ancestry(
        output,
        label="readiness authority output",
    )
    staging = output.with_name(f".{output.name}.staging")
    owner = output.with_name(f".{output.name}.staging.owner.json")
    _remove_owned_staging(staging, output=output, owner=owner)
    _write_json_new(
        owner,
        {
            "schema": STAGING_OWNER_SCHEMA,
            "staging_directory": str(staging),
            "output_directory": str(output),
        },
    )
    staging.mkdir()
    try:
        _write_json_new(staging / "_request.json", dict(request))
        for name in _CSV_COLUMNS:
            evidence[name].to_csv(staging / name, index=False)
        _write_json_new(staging / "summary.json", dict(summary))
        artifacts = [
            {
                "path": name,
                "bytes": (staging / name).stat().st_size,
                "sha256": _file_sha256(staging / name),
            }
            for name in sorted(_MANIFEST_ARTIFACTS)
        ]
        manifest = {
            "schema": CURRENT_RUN_SCHEMA,
            "request_sha256": request_sha256,
            "status": summary["status"],
            "artifacts": artifacts,
            "created_at_utc": _created_at_utc(),
        }
        _write_json_new(staging / "_manifest.json", manifest)
        _write_json_new(
            staging / "_authority.json",
            {
                "schema": CURRENT_AUTHORITY_SCHEMA,
                "state": "complete",
                "request_sha256": request_sha256,
                "artifact": "_manifest.json",
                "artifact_sha256": _file_sha256(staging / "_manifest.json"),
            },
        )
        verified = load_readiness_authority(
            staging,
            expected_request_sha256=request_sha256,
            require_current=True,
        )
        if output.exists():
            raise DataReadinessError("readiness output appeared during publication")
        staging.replace(output)
    except Exception:
        _remove_owned_staging(staging, output=output, owner=owner)
        raise
    try:
        owner.unlink(missing_ok=True)
    except OSError:
        pass
    return replace(verified, directory=output.resolve(strict=True))


def _validate_envelope_schemas(
    request: Mapping[str, object],
    manifest: Mapping[str, object],
    authority: Mapping[str, object],
    require_current: bool,
) -> Literal["historical_v1", "current_v3"]:
    triplet = (request.get("schema"), manifest.get("schema"), authority.get("schema"))
    if triplet == (HISTORICAL_RUN_SCHEMA, HISTORICAL_RUN_SCHEMA, HISTORICAL_AUTHORITY_SCHEMA):
        if require_current:
            raise DataReadinessError("historical readiness authority cannot authorize current planning")
        return "historical_v1"
    if triplet == (CURRENT_RUN_SCHEMA, CURRENT_RUN_SCHEMA, CURRENT_AUTHORITY_SCHEMA):
        return "current_v3"
    raise DataReadinessError("readiness authority schema versions differ or are unsupported")


def _validate_request(
    request: Mapping[str, object],
    version: Literal["historical_v1", "current_v3"],
) -> None:
    expected = _V1_REQUEST_KEYS if version == "historical_v1" else _V3_REQUEST_KEYS
    _require_exact_keys(request, expected, "readiness request")
    for key in expected:
        if key.endswith("sha256"):
            _required_sha256(request.get(key), f"request {key}")
    _require_exact_bool(request.get("training_performed"), "training_performed", False)
    _require_exact_bool(request.get("download_performed"), "download_performed", False)
    sources = _required_mapping(request.get("sources"), "request sources")
    _require_exact_keys(sources, _SOURCE_KEYS, "request sources")
    if version == "historical_v1":
        _validate_source(
            sources.get("swing"),
            _HISTORICAL_SWING_SOURCE_KEYS,
            {"technical_rows", "proxy_rows"},
            "swing",
        )
    else:
        _validate_current_swing_source(sources.get("swing"))
        calendar = _required_mapping(request.get("exchange_calendar"), "exchange calendar")
        _require_exact_keys(calendar, _EXCHANGE_CALENDAR_KEYS, "exchange calendar")
        if (
            calendar.get("name") != "XNYS"
            or calendar.get("package") != "exchange-calendars"
            or calendar.get("version") != package_version("exchange-calendars")
        ):
            raise DataReadinessError("readiness exchange-calendar identity differs")
    if version == "historical_v1":
        _validate_source(
            sources.get("intraday"),
            _HISTORICAL_INTRADAY_SOURCE_KEYS,
            {"collection_rows", "proxy_rows"},
            "intraday",
        )
    else:
        _validate_current_intraday_source(sources.get("intraday"))
    _validate_source(sources.get("catalyst"), _CATALYST_SOURCE_KEYS, {"source_event_rows"}, "catalyst")
    implementation = _required_mapping(request.get("implementation"), "request implementation")
    implementation_keys = _HISTORICAL_IMPLEMENTATION_KEYS if version == "historical_v1" else _CURRENT_IMPLEMENTATION_KEYS
    _require_exact_keys(implementation, implementation_keys, "request implementation")
    for role in implementation_keys:
        record = _required_mapping(implementation.get(role), f"implementation {role}")
        _require_exact_keys(record, _IMPLEMENTATION_RECORD_KEYS, f"implementation {role}")
        path = _required_text(record.get("path"), f"implementation {role} path")
        if Path(path).name != path:
            raise DataReadinessError(f"implementation {role} path must be a basename")
        _required_sha256(record.get("sha256"), f"implementation {role} sha256")


def _validate_source(
    raw: object,
    keys: frozenset[str],
    integer_keys: set[str],
    label: str,
) -> None:
    source = _required_mapping(raw, f"{label} source")
    _require_exact_keys(source, keys, f"{label} source")
    _required_text(source.get("type"), f"{label} source type")
    for key in keys - integer_keys - {"type"}:
        _required_sha256(source.get(key), f"{label} source {key}")
    for key in integer_keys:
        _strict_nonnegative_int(source.get(key), f"{label} source {key}")


def _validate_current_swing_source(raw: object) -> None:
    source = _required_mapping(raw, "swing source")
    _require_exact_keys(source, _CURRENT_SWING_SOURCE_KEYS, "swing source")
    if source.get("type") != "verified_edge_rebuild_swing_sources":
        raise DataReadinessError("swing source type differs")
    _required_text(source.get("strategy_id"), "swing source strategy_id")
    if _strict_nonnegative_int(source.get("horizon_sessions"), "swing horizon") != 10:
        raise DataReadinessError("swing readiness horizon differs")
    for key in (
        "panel_manifest_sha256",
        "panel_authority_sha256",
        "panel_request_sha256",
        "candidate_authority_sha256",
    ):
        _required_sha256(source.get(key), f"swing source {key}")
    for key in ("technical_rows", "proxy_rows"):
        _strict_nonnegative_int(source.get(key), f"swing source {key}")
    candidate_status = _required_text(source.get("candidate_status"), "swing candidate status")
    candidate_id = source.get("candidate_id")
    if candidate_status == "candidate":
        _required_text(candidate_id, "swing candidate identity")
    elif candidate_id is not None:
        raise DataReadinessError("non-candidate swing source has a candidate identity")
    promoted_status = _required_text(source.get("promoted_bundle_status"), "promoted bundle status")
    promoted_sha = source.get("promoted_bundle_sha256")
    if promoted_status == "verified":
        _required_sha256(promoted_sha, "promoted bundle sha256")
    elif promoted_status == "unavailable":
        if promoted_sha is not None:
            raise DataReadinessError("unavailable promoted bundle has an identity")
    else:
        raise DataReadinessError("promoted bundle status is unsupported")


def _validate_current_intraday_source(raw: object) -> None:
    source = _required_mapping(raw, "intraday source")
    _require_exact_keys(source, _CURRENT_INTRADAY_SOURCE_KEYS, "intraday source")
    if source.get("type") != "verified_intraday_specialist_training_sources":
        raise DataReadinessError("intraday source type differs")
    _required_text(source.get("strategy_id"), "intraday source strategy_id")
    _required_text(source.get("proxy_strategy_id"), "intraday source proxy_strategy_id")
    for key in _HISTORICAL_INTRADAY_SOURCE_KEYS - {"type", "collection_rows", "proxy_rows"}:
        _required_sha256(source.get(key), f"intraday source {key}")
    for key in ("collection_rows", "proxy_rows"):
        _strict_nonnegative_int(source.get(key), f"intraday source {key}")


def _validate_summary(
    summary: Mapping[str, object],
    version: Literal["historical_v1", "current_v3"],
    request_sha256: str,
) -> None:
    _require_exact_keys(summary, _SUMMARY_KEYS, "readiness summary")
    expected_schema = HISTORICAL_SUMMARY_SCHEMA if version == "historical_v1" else CURRENT_SUMMARY_SCHEMA
    if summary.get("schema") != expected_schema:
        raise DataReadinessError("readiness summary schema differs")
    if summary.get("request_sha256") != request_sha256:
        raise DataReadinessError("readiness summary request identity differs")
    status = _required_text(summary.get("status"), "summary status")
    if status not in _VALID_STATUSES:
        raise DataReadinessError("readiness summary status is unsupported")
    authorized = _require_exact_bool(summary.get("er2_authorized"), "er2_authorized")
    if authorized != (status == "ready_for_ER2"):
        raise DataReadinessError("readiness summary authorization and status differ")
    _require_exact_bool(summary.get("training_performed"), "training_performed", False)
    _require_exact_bool(summary.get("download_performed"), "download_performed", False)
    models = _strict_nonnegative_int(summary.get("models_created"), "models_created")
    if models != 0:
        raise DataReadinessError("readiness audit cannot claim created models")
    blocking = _strict_nonnegative_int(summary.get("blocking_findings"), "blocking_findings")
    _strict_nonnegative_int(summary.get("nonblocking_required_work"), "nonblocking_required_work")
    if (blocking > 0) != (status == "blocked_pending_targeted_acquisition"):
        raise DataReadinessError("readiness blocker count and status differ")
    acquisition = _required_mapping(summary.get("acquisition_plan"), "acquisition plan")
    _require_exact_keys(acquisition, _ACQUISITION_KEYS, "acquisition plan")
    _require_exact_bool(acquisition.get("authorized_by_audit"), "acquisition authorized")
    _require_exact_bool(
        acquisition.get("reuse_existing_daily_and_catalyst_sources"),
        "reuse existing sources",
        True,
    )
    for key in ("scope", "provider", "feed", "timeframe", "adjustment", "end_before"):
        _required_text(acquisition.get(key), f"acquisition {key}")
    _strict_nonnegative_int(acquisition.get("minimum_additional_sessions"), "minimum additional sessions")
    _strict_nonnegative_int(acquisition.get("target_history_sessions"), "target history sessions")
    memory = _required_mapping(summary.get("memory"), "readiness memory")
    _require_exact_keys(memory, _MEMORY_KEYS, "readiness memory")
    values = {key: _nonnegative_finite_number(memory.get(key), f"memory {key}") for key in _MEMORY_KEYS}
    if values["hard_budget_gib"] <= 0 or values["safety_threshold_gib"] > values["hard_budget_gib"]:
        raise DataReadinessError("readiness memory budget is invalid")
    if values["current_working_set_gib"] > values["hard_budget_gib"] or values["peak_working_set_gib"] > values["hard_budget_gib"]:
        raise DataReadinessError("readiness memory observation exceeds its hard budget")


def _artifact_records(manifest: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    raw = manifest.get("artifacts")
    if not isinstance(raw, list):
        raise DataReadinessError("readiness manifest artifacts must be a list")
    records: dict[str, Mapping[str, object]] = {}
    for index, value in enumerate(raw):
        record = _required_mapping(value, f"artifact record {index}")
        _require_exact_keys(record, _ARTIFACT_KEYS, f"artifact record {index}")
        name = _required_text(record.get("path"), f"artifact record {index} path")
        candidate = Path(name)
        if candidate.name != name or len(candidate.parts) != 1 or name in {".", ".."}:
            raise DataReadinessError("readiness artifact paths must be basenames")
        if name in records:
            raise DataReadinessError("readiness manifest has duplicate artifact paths")
        records[name] = record
    if set(records) != _MANIFEST_ARTIFACTS:
        raise DataReadinessError("readiness manifest artifact inventory differs")
    return records


def _validate_evidence(
    evidence: Mapping[str, pd.DataFrame],
    *,
    summary: Mapping[str, object],
    request: Mapping[str, object],
) -> None:
    if set(evidence) != set(_CSV_COLUMNS):
        raise DataReadinessError("readiness publication evidence inventory differs")
    for name, columns in _CSV_COLUMNS.items():
        frame = evidence.get(name)
        if not isinstance(frame, pd.DataFrame) or tuple(frame.columns) != columns:
            raise DataReadinessError(f"readiness publication {name} schema differs")
        if frame.duplicated().any():
            raise DataReadinessError(f"readiness publication {name} contains duplicate rows")

    integer_columns = {
        "catalyst_readiness.csv": ("observed_count",),
        "cost_readiness.csv": ("rows",),
        "dimension_coverage.csv": (
            "source_rows",
            "sessions",
            "tickers",
            "proxy_eligible_opportunities",
        ),
        "exclusion_reasons.csv": ("excluded_rows", "total_rows"),
        "fold_capacity.csv": (
            "fold",
            "test_sessions",
            "minimum_test_sessions_required",
        ),
        "phase_capacity.csv": (
            "phase",
            "sessions",
            "source_rows",
            "unique_decision_groups",
            "unique_tickers",
            "minimum_sessions_required",
        ),
        "session_calendar.csv": (
            "source_rows",
            "unique_decision_groups",
            "unique_tickers",
            "proxy_eligible_opportunities",
            "year",
        ),
        "source_inventory.csv": (
            "raw_rows",
            "source_usable_rows",
            "proxy_setup_rows",
            "proxy_label_eligible_rows",
            "unique_decision_groups",
            "unique_tickers",
            "valid_sessions",
            "effective_session_blocks",
        ),
    }
    optional_integer_columns = {
        "session_calendar.csv": ("er_setup_opportunities",),
        "source_inventory.csv": ("er_setup_opportunities",),
    }
    boolean_columns = {
        "blockers.csv": ("blocks_er2",),
        "cost_readiness.csv": (
            "exact_cost_available",
            "adverse_fill_stress_available",
        ),
        "source_inventory.csv": (
            "exact_new_horizon_labels",
            "collection_model_data_ready",
        ),
    }
    for name, columns in integer_columns.items():
        for column in columns:
            _validate_nonnegative_integer_series(evidence[name][column], f"{name} {column}")
    for name, columns in optional_integer_columns.items():
        for column in columns:
            _validate_nonnegative_integer_series(
                evidence[name][column],
                f"{name} {column}",
                allow_missing=True,
            )
    for name, columns in boolean_columns.items():
        for column in columns:
            _strict_bool_series(evidence[name][column], f"{name} {column}")

    required_text_columns = {
        "blockers.csv": ("blocker_code", "scope", "required_action", "detail"),
        "catalyst_readiness.csv": (
            "evidence_type",
            "evidence_value",
            "research_status",
            "promotion_status",
            "detail",
        ),
        "cost_readiness.csv": ("strategy_id",),
        "dimension_coverage.csv": ("strategy_id", "dimension", "value"),
        "exclusion_reasons.csv": ("strategy_id", "reason"),
        "fold_capacity.csv": ("strategy_id", "capacity_type", "status"),
        "phase_capacity.csv": ("strategy_id", "status"),
        "session_calendar.csv": (
            "strategy_id",
            "proxy_strategy_id",
            "session_date_et",
            "er_setup_status",
        ),
        "source_inventory.csv": (
            "strategy_id",
            "source_role",
            "price_feed",
            "adjustment",
            "session_gate",
            "source_authority",
            "point_in_time_membership_status",
            "benchmark_interval_status",
        ),
    }
    for name, columns in required_text_columns.items():
        for column in columns:
            _validate_required_text_series(evidence[name][column], f"{name} {column}")

    costs = evidence["cost_readiness.csv"]
    for column in (
        "minimum_cost_bps",
        "median_cost_bps",
        "mean_cost_bps",
        "maximum_cost_bps",
    ):
        _validate_nonnegative_number_series(costs[column], f"cost_readiness.csv {column}")
    if (
        costs["minimum_cost_bps"].gt(costs["median_cost_bps"]).any()
        or costs["median_cost_bps"].gt(costs["maximum_cost_bps"]).any()
        or costs["mean_cost_bps"].lt(costs["minimum_cost_bps"]).any()
        or costs["mean_cost_bps"].gt(costs["maximum_cost_bps"]).any()
    ):
        raise DataReadinessError("readiness cost statistics are inconsistent")
    inventory = evidence["source_inventory.csv"]
    _validate_fraction_series(inventory["coverage_exact_rate"], "source_inventory.csv coverage_exact_rate")
    if len(inventory) != 3:
        raise DataReadinessError("readiness source inventory must contain swing, intraday, and catalyst authorities")

    calendar = evidence["session_calendar.csv"]
    _validate_canonical_dates(calendar["session_date_et"], "session_calendar.csv session_date_et")
    _validate_exchange_sessions(calendar["session_date_et"], request=request)
    if calendar.duplicated(["strategy_id", "session_date_et"]).any():
        raise DataReadinessError("readiness session calendar contains duplicate strategy sessions")
    dates = pd.to_datetime(calendar["session_date_et"], format="%Y-%m-%d", errors="coerce")
    years = pd.to_numeric(calendar["year"], errors="coerce")
    if not calendar.empty and not dates.dt.year.eq(years).all():
        raise DataReadinessError("readiness session calendar year differs from its date")
    if (
        pd.to_numeric(calendar["unique_decision_groups"]).gt(calendar["source_rows"]).any()
        or pd.to_numeric(calendar["unique_tickers"]).gt(calendar["source_rows"]).any()
        or pd.to_numeric(calendar["proxy_eligible_opportunities"]).gt(calendar["source_rows"]).any()
    ):
        raise DataReadinessError("readiness session calendar counts are inconsistent")
    _validate_status_series(
        calendar["er_setup_status"],
        {"not_built_until_ER3"},
        "session calendar ER setup status",
    )

    if inventory["strategy_id"].astype(str).str.strip().eq("").any():
        raise DataReadinessError("readiness source inventory has an empty strategy identity")
    if inventory["strategy_id"].duplicated().any():
        raise DataReadinessError("readiness source inventory has duplicate strategy identities")
    if (
        pd.to_numeric(inventory["source_usable_rows"]).gt(inventory["raw_rows"]).any()
        or pd.to_numeric(inventory["proxy_label_eligible_rows"]).gt(inventory["proxy_setup_rows"]).any()
    ):
        raise DataReadinessError("readiness source inventory counts are inconsistent")
    known_strategies = set(inventory["strategy_id"].astype(str))
    swing_inventory = inventory.loc[inventory["strategy_id"].astype(str).str.startswith("SWING.")]
    intraday_inventory = inventory.loc[
        inventory["source_role"].eq("exact_one_minute_proxy_training")
        & inventory["strategy_id"].astype(str).str.startswith("INTRADAY.")
    ]
    catalyst_inventory = inventory.loc[inventory["strategy_id"].eq("CATALYST.OVERLAY")]
    if len(swing_inventory) != 1 or len(intraday_inventory) != 1 or len(catalyst_inventory) != 1:
        raise DataReadinessError("readiness source inventory authority roles differ")
    raw_swing_source = _required_mapping(
        _required_mapping(request.get("sources"), "request sources").get("swing"),
        "swing source",
    )
    requested_swing_strategy = raw_swing_source.get("strategy_id")
    if requested_swing_strategy is not None and requested_swing_strategy != swing_inventory.iloc[0]["strategy_id"]:
        raise DataReadinessError("readiness swing evidence identity differs from its request")
    if not set(calendar["strategy_id"].astype(str)).issubset(known_strategies):
        raise DataReadinessError("readiness calendar references an unknown strategy")
    swing_strategy_id = str(swing_inventory.iloc[0]["strategy_id"])
    intraday_strategy_id = str(intraday_inventory.iloc[0]["strategy_id"])
    raw_intraday_source = _required_mapping(
        _required_mapping(request.get("sources"), "request sources").get("intraday"),
        "intraday source",
    )
    requested_intraday_strategy = raw_intraday_source.get("strategy_id")
    requested_intraday_proxy = raw_intraday_source.get("proxy_strategy_id")
    if requested_intraday_strategy is not None and (
        requested_intraday_strategy != intraday_strategy_id
        or requested_intraday_proxy != intraday_inventory.iloc[0]["proxy_strategy_id"]
    ):
        raise DataReadinessError("readiness intraday evidence identity differs from its request")
    modeled_strategies = {swing_strategy_id, intraday_strategy_id}
    if set(calendar["strategy_id"].astype(str)) != modeled_strategies:
        raise DataReadinessError("readiness calendar does not cover both modeled strategies")
    observed_sessions = calendar.groupby("strategy_id", sort=False)["session_date_et"].nunique()
    for _, row in inventory.iterrows():
        strategy_id = str(row["strategy_id"])
        expected_sessions = int(row["valid_sessions"])
        if expected_sessions != int(observed_sessions.get(strategy_id, 0)):
            raise DataReadinessError("readiness source inventory session count differs from calendar")

    _validate_status_series(
        evidence["fold_capacity.csv"]["status"],
        {"pass", "blocked"},
        "fold capacity status",
    )
    _validate_status_series(
        evidence["phase_capacity.csv"]["status"],
        {"pass", "blocked"},
        "phase capacity status",
    )
    _validate_status_series(
        inventory["session_gate"],
        {"pass", "blocked", "not_applicable"},
        "source inventory session gate",
    )
    for column in ("research_status", "promotion_status"):
        _validate_status_series(
            evidence["catalyst_readiness.csv"][column],
            {"pass", "blocked"},
            f"catalyst {column}",
        )
    if evidence["catalyst_readiness.csv"].empty:
        raise DataReadinessError("readiness catalyst evidence is empty")

    required_strategy_coverage = {
        "cost_readiness.csv": modeled_strategies,
        "dimension_coverage.csv": modeled_strategies,
        "exclusion_reasons.csv": modeled_strategies,
        "phase_capacity.csv": {swing_strategy_id},
        "fold_capacity.csv": {intraday_strategy_id},
    }
    for name, expected_strategies in required_strategy_coverage.items():
        if set(evidence[name]["strategy_id"].astype(str)) != expected_strategies:
            raise DataReadinessError(f"readiness publication {name} strategy coverage differs")
    if not _strict_bool_series(costs["exact_cost_available"], "exact cost availability").all():
        raise DataReadinessError("readiness costs lack exact interval evidence")

    uniqueness = {
        "blockers.csv": ["blocker_code", "scope"],
        "catalyst_readiness.csv": ["evidence_type", "evidence_value"],
        "cost_readiness.csv": ["strategy_id"],
        "dimension_coverage.csv": ["strategy_id", "dimension", "value"],
        "exclusion_reasons.csv": ["strategy_id", "reason"],
        "fold_capacity.csv": ["strategy_id", "fold"],
        "phase_capacity.csv": ["strategy_id", "phase"],
    }
    for name, identity_columns in uniqueness.items():
        if evidence[name].duplicated(identity_columns).any():
            raise DataReadinessError(f"readiness publication {name} has duplicate identities")
    exclusions = evidence["exclusion_reasons.csv"]
    if pd.to_numeric(exclusions["excluded_rows"]).gt(exclusions["total_rows"]).any():
        raise DataReadinessError("readiness exclusion counts are inconsistent")
    _validate_capacity_dates(evidence["fold_capacity.csv"])

    blockers = evidence["blockers.csv"]
    blocker_flags = _strict_bool_series(blockers["blocks_er2"], "blockers.csv blocks_er2")
    blocking = blockers.loc[blocker_flags]
    blocking_count = _strict_nonnegative_int(summary.get("blocking_findings"), "blocking_findings")
    nonblocking_count = _strict_nonnegative_int(
        summary.get("nonblocking_required_work"),
        "nonblocking_required_work",
    )
    if blocking_count != len(blocking) or nonblocking_count != len(blockers) - len(blocking):
        raise DataReadinessError("readiness summary blocker counts differ from blocker evidence")
    expected_status = "blocked_pending_targeted_acquisition" if not blocking.empty else "ready_for_ER2"
    if summary.get("status") != expected_status:
        raise DataReadinessError("readiness summary status differs from blocker evidence")
    acquisition = _required_mapping(summary.get("acquisition_plan"), "acquisition plan")
    authorized = _require_exact_bool(acquisition.get("authorized_by_audit"), "acquisition authorized")
    intraday_shortage = blocking.loc[
        blocking["blocker_code"].eq("intraday_session_history_below_gate")
        & blocking["scope"].eq(intraday_strategy_id)
    ]
    expected_authorized = len(blocking) == 1 and len(intraday_shortage) == 1
    if authorized != expected_authorized:
        raise DataReadinessError("readiness acquisition authorization differs from eligible blockers")
    if authorized and (
        acquisition.get("scope") != "intraday_only"
        or _strict_nonnegative_int(
            acquisition.get("minimum_additional_sessions"),
            "minimum additional sessions",
        )
        <= 0
    ):
        raise DataReadinessError("readiness intraday acquisition plan is invalid")


def _validate_csv_header(payload: bytes, expected: tuple[str, ...], name: str) -> None:
    try:
        text = payload.decode("utf-8")
        header = next(csv.reader(io.StringIO(text, newline="")))
    except (UnicodeError, csv.Error, StopIteration) as exc:
        raise DataReadinessError(f"readiness artifact is not valid CSV: {name}") from exc
    if tuple(header) != expected:
        raise DataReadinessError(f"readiness artifact schema differs: {name}")


def _read_csv_evidence(
    payload: bytes,
    expected: tuple[str, ...],
    name: str,
) -> pd.DataFrame:
    _validate_csv_header(payload, expected, name)
    try:
        frame = pd.read_csv(io.BytesIO(payload))
    except (OSError, UnicodeError, pd.errors.ParserError) as exc:
        raise DataReadinessError(f"readiness artifact is unreadable: {name}") from exc
    if tuple(frame.columns) != expected:
        raise DataReadinessError(f"readiness artifact schema differs: {name}")
    return frame


def _validate_nonnegative_integer_series(
    values: pd.Series,
    label: str,
    *,
    allow_missing: bool = False,
) -> None:
    if pd.api.types.is_bool_dtype(values.dtype) or values.map(lambda value: isinstance(value, bool)).any():
        raise DataReadinessError(f"{label} must not contain Booleans")
    numeric = pd.to_numeric(values, errors="coerce")
    missing = values.isna() | values.astype(str).str.strip().eq("")
    if (missing.any() and not allow_missing) or ((~missing) & numeric.isna()).any():
        raise DataReadinessError(f"{label} must contain integers")
    observed = numeric.loc[~missing].astype(float)
    if not observed.map(math.isfinite).all() or observed.lt(0).any() or observed.mod(1).ne(0).any():
        raise DataReadinessError(f"{label} must contain nonnegative integers")


def _validate_required_text_series(values: pd.Series, label: str) -> None:
    if values.isna().any() or values.astype(str).str.strip().eq("").any():
        raise DataReadinessError(f"{label} contains an empty value")


def _validate_nonnegative_number_series(values: pd.Series, label: str) -> None:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any() or not numeric.astype(float).map(math.isfinite).all() or numeric.lt(0).any():
        raise DataReadinessError(f"{label} must contain finite nonnegative numbers")


def _validate_fraction_series(values: pd.Series, label: str) -> None:
    _validate_nonnegative_number_series(values, label)
    if pd.to_numeric(values, errors="coerce").gt(1).any():
        raise DataReadinessError(f"{label} must be between zero and one")


def _strict_bool_series(values: pd.Series, label: str) -> pd.Series:
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin(("true", "false", "1", "0")).all():
        raise DataReadinessError(f"{label} contains a non-canonical Boolean")
    return normalized.isin(("true", "1"))


def _validate_canonical_dates(values: pd.Series, label: str) -> None:
    normalized = values.astype(str).str.strip()
    parsed = pd.to_datetime(normalized, format="%Y-%m-%d", errors="coerce")
    if parsed.isna().any() or not parsed.dt.strftime("%Y-%m-%d").eq(normalized).all():
        raise DataReadinessError(f"{label} contains an invalid date")


def _validate_exchange_sessions(
    values: pd.Series,
    *,
    request: Mapping[str, object],
) -> None:
    if values.empty:
        return
    raw_calendar = request.get("exchange_calendar")
    if raw_calendar is None:
        return
    calendar_identity = _required_mapping(raw_calendar, "exchange calendar")
    name = _required_text(calendar_identity.get("name"), "exchange calendar name")
    calendar = xcals.get_calendar(name)
    dates = pd.DatetimeIndex(pd.to_datetime(values, format="%Y-%m-%d"))
    sessions = calendar.sessions_in_range(dates.min(), dates.max()).tz_localize(None)
    if not dates.isin(sessions).all():
        raise DataReadinessError("readiness session calendar contains a non-trading session")


def _validate_status_series(values: pd.Series, allowed: set[str], label: str) -> None:
    observed = set(values.astype(str).str.strip())
    if not observed.issubset(allowed):
        raise DataReadinessError(f"{label} contains an unsupported value")


def _validate_capacity_dates(frame: pd.DataFrame) -> None:
    if frame.empty:
        return
    first = frame["first_test_session"].fillna("").astype(str).str.strip()
    last = frame["last_test_session"].fillna("").astype(str).str.strip()
    sessions = pd.to_numeric(frame["test_sessions"], errors="coerce")
    populated = sessions.gt(0)
    if first.loc[populated].eq("").any() or last.loc[populated].eq("").any():
        raise DataReadinessError("readiness fold with sessions has an empty date boundary")
    if first.loc[~populated].ne("").any() or last.loc[~populated].ne("").any():
        raise DataReadinessError("readiness empty fold has a date boundary")
    _validate_canonical_dates(first.loc[populated], "fold first test session")
    _validate_canonical_dates(last.loc[populated], "fold last test session")
    first_dates = pd.to_datetime(first.loc[populated], format="%Y-%m-%d")
    last_dates = pd.to_datetime(last.loc[populated], format="%Y-%m-%d")
    if first_dates.gt(last_dates).any():
        raise DataReadinessError("readiness fold date interval is inverted")


def _remove_owned_staging(staging: Path, *, output: Path, owner: Path) -> None:
    staging = path_integrity.verify_no_reparse_ancestry(staging, label="readiness staging")
    owner = path_integrity.verify_no_reparse_ancestry(owner, label="readiness staging owner")
    if not staging.exists() and not owner.exists():
        return
    if not owner.is_file():
        raise DataReadinessError("readiness staging is not owned by this publisher")
    owner_record = _json_object(owner.read_bytes(), "readiness staging owner")
    if owner_record != {
        "schema": STAGING_OWNER_SCHEMA,
        "staging_directory": str(staging),
        "output_directory": str(output),
    }:
        raise DataReadinessError("readiness staging owner does not verify")
    if staging.exists():
        verified = path_integrity.verify_tree_containment(staging, label="readiness staging")
        shutil.rmtree(verified)
    owner.unlink(missing_ok=True)


def _read_confined(root: Path, name: str) -> bytes:
    path = path_integrity.resolve_existing_file_inside(root, name, label=f"readiness artifact {name}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise DataReadinessError(f"readiness artifact is unreadable: {name}") from exc


def _json_object(payload: bytes, label: str) -> dict[str, object]:
    def reject_constant(value: str) -> object:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DataReadinessError(f"{label} is invalid") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _write_json_new(path: Path, payload: Mapping[str, object]) -> None:
    try:
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        raise DataReadinessError(f"cannot publish readiness JSON: {path.name}") from exc


def _json_sha256(value: Mapping[str, object]) -> str:
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("readiness request is not canonical JSON") from exc
    return _bytes_sha256(encoded)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_exact_keys(record: Mapping[str, object], expected: frozenset[str], label: str) -> None:
    if set(record) != expected:
        raise DataReadinessError(f"{label} schema differs")


def _required_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, dict):
        raise DataReadinessError(f"{label} must be an object")
    return cast(dict[str, object], value)


def _required_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataReadinessError(f"{label} is invalid")
    return value


def _required_sha256(value: object, label: str) -> str:
    text = _required_text(value, label)
    if len(text) != 64 or any(character not in _HEX for character in text):
        raise DataReadinessError(f"{label} is invalid")
    return text


def _strict_nonnegative_int(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DataReadinessError(f"{label} must be a nonnegative integer")
    return value


def _require_exact_bool(value: object, label: str, expected: bool | None = None) -> bool:
    if not isinstance(value, bool):
        raise DataReadinessError(f"{label} must be a boolean")
    if expected is not None and value is not expected:
        raise DataReadinessError(f"{label} has an invalid value")
    return value


def _nonnegative_finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataReadinessError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise DataReadinessError(f"{label} must be finite and nonnegative")
    return number


def _created_at_utc() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()
