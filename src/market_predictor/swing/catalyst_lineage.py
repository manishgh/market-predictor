"""Hash-bound historical catalyst lineage and decision assignment replay."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.reconciliation import (
    ASSIGNMENT_COLUMNS,
    ASSIGNMENT_SCHEMA_VERSION,
    ASSIGNMENT_STATUSES,
    assignment_integrity_summary,
    build_event_assignments,
    reconciliation_sha256,
    reconciliation_summary,
)
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.core import path_integrity
from market_predictor.core.errors import DataReadinessError
from market_predictor.resources import (
    assert_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.swing.news_source_inventory import (
    build_source_news_shard_inventory,
)

CATALYST_LINEAGE_REQUEST_SCHEMA = "swing.catalyst_lineage_request.v2"
CATALYST_LINEAGE_MANIFEST_SCHEMA = "swing.catalyst_lineage_manifest.v2"
CATALYST_EVENT_SCHEMA = "swing.catalyst_event.v1"
CATALYST_COVERAGE_SCHEMA = "swing.catalyst_source_coverage.v1"
FEATURE_INVENTORY_SCHEMA = "swing.catalyst_feature_inventory.v1"
_EXPECTED_POLICY_SCHEMA = "market_predictor.catalyst_lineage.v1"
_SUPPORTED_CHANNELS = frozenset({"direct_issuer", "business_exposure", "sector_context"})
_TRAINING_ELIGIBLE_CHANNELS = frozenset({"direct_issuer"})
_RESEARCH_ONLY_CHANNELS = _SUPPORTED_CHANNELS - _TRAINING_ELIGIBLE_CHANNELS
_FEATURE_INVENTORY_KEYS = frozenset(
    {
        "availability_policy",
        "channel_counts",
        "coverage_states",
        "event_artifact_count",
        "production_ready",
        "profiles",
        "request_sha256",
        "research_only_channels",
        "schema",
        "training_contract",
        "training_eligible_channels",
    }
)
_DECISION_COLUMNS = (
    "ticker",
    "security_id",
    "decision_time_utc",
    "prediction_cutoff_policy_id",
    "timeframe",
    "bar_start_utc",
)
CATALYST_EVENT_COLUMNS = (
    "event_id",
    "source_event_id",
    "relation_id",
    "source_security_id",
    "source_ticker",
    "security_id",
    "ticker",
    "source_family",
    "published_at_utc",
    "event_available_at_utc",
    "relation_feature_available_at_utc",
    "sentiment_feature_available_at_utc",
    "feature_available_at_utc",
    "availability_policy",
    "relation_channel",
    "relation_score",
    "relation_basis",
    "sentiment_label",
    "sentiment_confidence",
    "sentiment_numeric",
    "relevance",
    "source_relevance",
    "source_relevance_basis",
    "sentiment_input_sha256",
    "sentiment_model",
    "sentiment_model_revision",
    "attribution_policy_version",
    "attribution_policy_sha256",
    "training_eligible",
    "training_exclusion_reason",
    "schema_version",
)
CATALYST_COVERAGE_COLUMNS = (
    "collection_id",
    "chunk_id",
    "security_id",
    "ticker",
    "source_family",
    "requested_start_utc",
    "requested_end_utc",
    "status",
    "row_count",
    "coverage_state",
    "missingness_known",
    "zero_event_semantics",
    "training_eligible",
    "schema_version",
)

_REQUEST_KEYS = frozenset(
    {
        "schema",
        "collection_manifest_sha256",
        "collection_audit_sha256",
        "attribution_manifest_sha256",
        "sentiment_manifest_sha256",
        "decisions_sha256",
        "source_collections_sha256",
        "policy_sha256",
        "excluded_security_ids",
        "production_ready",
        "request_sha256",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "request_sha256",
        "status",
        "requested_chunks",
        "observed_chunks",
        "skipped_chunks",
        "failed_chunks",
        "excluded_security_ids",
        "source_event_rows",
        "related_source_events",
        "relation_rows",
        "training_eligible_rows",
        "channel_counts",
        "assignment_rows",
        "assignment_status_counts",
        "coverage",
        "feature_inventory",
        "artifacts",
        "lineage_sha256",
        "memory",
        "completed_at_utc",
        "production_ready",
    }
)
_ARTIFACT_RECORD_KEYS = frozenset(
    {
        "chunk_id",
        "source_event_sha256",
        "relation_sha256",
        "sentiment_sha256",
        "event_path",
        "event_sha256",
        "event_rows",
        "training_eligible_rows",
        "assignment_path",
        "assignment_sha256",
        "assignment_material_sha256",
        "assignment_rows",
    }
)
_CANONICAL_MANIFEST_KEYS = frozenset(
    {
        "schema",
        "canonical_schema_version",
        "artifact_type",
        "artifact_path",
        "artifact_sha256",
        "created_at_utc",
        "rows",
        "columns",
        "first_available_at_utc",
        "last_available_at_utc",
        "inputs",
        "audit",
        "production_ready",
    }
)


@dataclass(frozen=True, slots=True)
class CatalystLineagePolicy:
    availability_policy: str
    training_eligible_channels: tuple[str, ...]
    research_only_channels: tuple[str, ...]
    assignment_windows: Mapping[str, pd.Timedelta]
    feature_profiles: Mapping[str, tuple[str, ...]]
    maximum_process_memory_gib: float
    memory_guard_headroom_gib: float
    raw: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class VerifiedCatalystLineage:
    """Verified metadata for one immutable completed catalyst-lineage bundle."""

    directory: Path
    manifest: Mapping[str, object]
    request: Mapping[str, object]
    feature_inventory: Mapping[str, object]
    coverage: pd.DataFrame
    manifest_sha256: str
    request_sha256: str
    coverage_sha256: str
    coverage_manifest_sha256: str
    feature_inventory_sha256: str
    lineage_sha256: str


def load_catalyst_lineage_policy(path: Path) -> CatalystLineagePolicy:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != _EXPECTED_POLICY_SCHEMA:
        raise DataReadinessError(f"unsupported catalyst lineage policy: {path}")
    eligible = _string_tuple(raw.get("training_eligible_channels"), "training_eligible_channels")
    research_only = _string_tuple(raw.get("research_only_channels"), "research_only_channels")
    if set(eligible).intersection(research_only):
        raise DataReadinessError("catalyst channels cannot be both eligible and research-only")
    if set(eligible).union(research_only) != _SUPPORTED_CHANNELS:
        raise DataReadinessError("catalyst lineage policy must classify every supported relation channel")
    raw_windows = raw.get("assignment_windows")
    if not isinstance(raw_windows, dict) or not raw_windows:
        raise DataReadinessError("catalyst lineage policy has no assignment windows")
    windows = {str(name): pd.Timedelta(str(duration)) for name, duration in raw_windows.items()}
    if any(duration <= pd.Timedelta(0) for duration in windows.values()):
        raise DataReadinessError("catalyst assignment windows must be positive")
    raw_profiles = raw.get("feature_profiles")
    if not isinstance(raw_profiles, dict):
        raise DataReadinessError("catalyst lineage policy has no feature profiles")
    profiles: dict[str, tuple[str, ...]] = {}
    for name in ("catalyst_only", "technical_plus_catalyst"):
        record = raw_profiles.get(name)
        if not isinstance(record, dict):
            raise DataReadinessError(f"catalyst feature profile is missing: {name}")
        profiles[name] = _string_tuple(record.get("features"), f"feature_profiles.{name}.features")
    maximum_memory = float(raw.get("maximum_process_memory_gib", 0))
    headroom = float(raw.get("memory_guard_headroom_gib", 0))
    if maximum_memory <= 0 or headroom <= 0 or headroom >= maximum_memory:
        raise DataReadinessError("catalyst lineage memory policy is invalid")
    availability_policy = str(raw.get("availability_policy", "")).strip()
    if not availability_policy:
        raise DataReadinessError("catalyst lineage availability policy is empty")
    return CatalystLineagePolicy(
        availability_policy=availability_policy,
        training_eligible_channels=eligible,
        research_only_channels=research_only,
        assignment_windows=windows,
        feature_profiles=profiles,
        maximum_process_memory_gib=maximum_memory,
        memory_guard_headroom_gib=headroom,
        raw=raw,
    )


def verify_completed_catalyst_lineage(directory: Path) -> VerifiedCatalystLineage:
    """Verify a completed lineage bundle without loading any artifact columns."""

    root = path_integrity.verify_tree_containment(
        directory,
        label="catalyst lineage bundle",
    )
    manifest_path = path_integrity.resolve_existing_file_inside(
        root,
        "_manifest.json",
        label="catalyst lineage manifest",
    )
    request_path = path_integrity.resolve_existing_file_inside(
        root,
        "_request.json",
        label="catalyst lineage request",
    )
    status_path = path_integrity.resolve_existing_file_inside(
        root,
        "_status.json",
        label="catalyst lineage status",
    )
    inventory_path = path_integrity.resolve_existing_file_inside(
        root,
        "feature_inventory.json",
        label="catalyst feature inventory",
    )
    coverage_path = path_integrity.resolve_existing_file_inside(
        root,
        "source_coverage.parquet",
        label="catalyst source coverage",
    )

    manifest = _json_object(manifest_path)
    if set(manifest) != _MANIFEST_KEYS:
        raise DataReadinessError("catalyst lineage manifest fields do not match the contract")
    if (
        manifest.get("schema") != CATALYST_LINEAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("failed_chunks") != {}
        or manifest.get("production_ready") is not False
    ):
        raise DataReadinessError(f"catalyst lineage is not a completed research bundle: {root}")
    status = _json_object(status_path)
    if status != manifest:
        raise DataReadinessError("catalyst lineage status and manifest do not match")

    request = _json_object(request_path)
    if set(request) != _REQUEST_KEYS:
        raise DataReadinessError("catalyst lineage request fields do not match the contract")
    request_material = dict(request)
    embedded_request_sha256 = _verified_sha256(
        request_material.pop("request_sha256", None),
        "request_sha256",
    )
    if (
        request_material.get("schema") != CATALYST_LINEAGE_REQUEST_SCHEMA
        or request_material.get("production_ready") is not False
        or _json_sha256(request_material) != embedded_request_sha256
        or manifest.get("request_sha256") != embedded_request_sha256
    ):
        raise DataReadinessError("catalyst lineage request hash binding does not verify")
    for key in (
        "collection_manifest_sha256",
        "collection_audit_sha256",
        "attribution_manifest_sha256",
        "sentiment_manifest_sha256",
        "decisions_sha256",
        "source_collections_sha256",
        "policy_sha256",
    ):
        _verified_sha256(request_material.get(key), key)

    inventory_record = _verified_mapping(manifest.get("feature_inventory"), "feature_inventory")
    if set(inventory_record) != {"path", "sha256"}:
        raise DataReadinessError("catalyst feature inventory record is malformed")
    if Path(_verified_text(inventory_record.get("path"), "feature_inventory.path")).resolve() != inventory_path:
        raise DataReadinessError("catalyst feature inventory path does not verify")
    feature_inventory_sha256 = _verified_sha256(
        inventory_record.get("sha256"),
        "feature_inventory.sha256",
    )
    inventory = _json_object(inventory_path)
    if (
        file_sha256(inventory_path) != feature_inventory_sha256
        or inventory.get("schema") != FEATURE_INVENTORY_SCHEMA
        or inventory.get("request_sha256") != embedded_request_sha256
        or inventory.get("production_ready") is not False
    ):
        raise DataReadinessError("catalyst feature inventory identity does not verify")
    _verify_feature_inventory(inventory, manifest=manifest)

    coverage_record = _verified_mapping(manifest.get("coverage"), "coverage")
    if set(coverage_record) != {"path", "sha256", "rows", "states"}:
        raise DataReadinessError("catalyst coverage record is malformed")
    if Path(_verified_text(coverage_record.get("path"), "coverage.path")).resolve() != coverage_path:
        raise DataReadinessError("catalyst coverage path does not verify")
    coverage_sha256 = _verified_sha256(coverage_record.get("sha256"), "coverage.sha256")
    coverage_frame, coverage_manifest = _load_canonical_identity(
        coverage_path,
        expected_type="catalyst_source_coverage",
    )
    del coverage_frame
    _verify_canonical_manifest(
        coverage_manifest,
        path=coverage_path,
        artifact_sha256=coverage_sha256,
        rows=_verified_nonnegative_int(coverage_record.get("rows"), "coverage.rows"),
        expected_inputs={
            "catalyst_lineage_request_sha256": embedded_request_sha256,
            "source_collections_sha256": _verified_sha256(
                request_material.get("source_collections_sha256"),
                "source_collections_sha256",
            ),
        },
    )
    records = _verified_lineage_records(manifest)
    requested_chunks = _verified_nonnegative_int(manifest.get("requested_chunks"), "requested_chunks")
    observed_chunks = _verified_nonnegative_int(manifest.get("observed_chunks"), "observed_chunks")
    if requested_chunks != observed_chunks or observed_chunks != len(records):
        raise DataReadinessError("catalyst lineage chunk counts do not reconcile")

    expected_files = {
        "_manifest.json",
        "_request.json",
        "_status.json",
        "feature_inventory.json",
        "source_coverage.parquet",
        "source_coverage.parquet.manifest.json",
    }
    for record in records:
        chunk_id = _verified_chunk_id(record.get("chunk_id"))
        expected_files.update(
            {
                f"events/{chunk_id}.parquet",
                f"events/{chunk_id}.parquet.manifest.json",
                f"assignments/{chunk_id}.parquet",
                f"assignments/{chunk_id}.parquet.manifest.json",
            }
        )
    _verify_exact_lineage_inventory(root, expected_files)
    initial_file_hashes = {relative: file_sha256(root / Path(relative)) for relative in sorted(expected_files)}
    coverage, projected_coverage_manifest = load_canonical_artifact(
        coverage_path,
        expected_type="catalyst_source_coverage",
        allow_research=True,
        columns=CATALYST_COVERAGE_COLUMNS,
    )
    if projected_coverage_manifest != coverage_manifest:
        raise DataReadinessError("catalyst coverage sidecar changed during projected load")
    _verify_coverage_semantics(coverage, manifest=manifest)
    event_rows = 0
    eligible_rows = 0
    assignment_rows = 0
    observed_channel_counts = {channel: 0 for channel in sorted(_SUPPORTED_CHANNELS)}
    observed_assignment_status_counts: dict[str, int] = {}
    eligible_channels = tuple(str(value) for value in cast(list[object], inventory["training_eligible_channels"]))
    decisions_sha256 = _verified_sha256(request_material.get("decisions_sha256"), "decisions_sha256")
    for record in records:
        chunk_id = _verified_chunk_id(record.get("chunk_id"))
        event_relative = f"events/{chunk_id}.parquet"
        assignment_relative = f"assignments/{chunk_id}.parquet"
        event_path = path_integrity.resolve_existing_file_inside(
            root,
            event_relative,
            label=f"catalyst event artifact {chunk_id}",
        )
        assignment_path = path_integrity.resolve_existing_file_inside(
            root,
            assignment_relative,
            label=f"catalyst assignment artifact {chunk_id}",
        )
        if Path(_verified_text(record.get("event_path"), "event_path")).resolve() != event_path:
            raise DataReadinessError(f"catalyst event path mismatch: {chunk_id}")
        if Path(_verified_text(record.get("assignment_path"), "assignment_path")).resolve() != assignment_path:
            raise DataReadinessError(f"catalyst assignment path mismatch: {chunk_id}")

        source_event_sha256 = _verified_sha256(record.get("source_event_sha256"), "source_event_sha256")
        relation_sha256 = _verified_sha256(record.get("relation_sha256"), "relation_sha256")
        sentiment_sha256 = _verified_sha256(record.get("sentiment_sha256"), "sentiment_sha256")
        event_sha256 = _verified_sha256(record.get("event_sha256"), "event_sha256")
        assignment_sha256 = _verified_sha256(record.get("assignment_sha256"), "assignment_sha256")
        assignment_material_sha256 = _verified_sha256(
            record.get("assignment_material_sha256"),
            "assignment_material_sha256",
        )
        chunk_event_rows = _verified_nonnegative_int(record.get("event_rows"), "event_rows")
        chunk_eligible_rows = _verified_nonnegative_int(
            record.get("training_eligible_rows"),
            "training_eligible_rows",
        )
        chunk_assignment_rows = _verified_nonnegative_int(
            record.get("assignment_rows"),
            "assignment_rows",
        )
        if chunk_eligible_rows > chunk_event_rows:
            raise DataReadinessError(f"catalyst eligible rows exceed event rows: {chunk_id}")
        common_inputs = {
            "catalyst_lineage_request_sha256": embedded_request_sha256,
            "source_event_sha256": source_event_sha256,
            "relation_sha256": relation_sha256,
            "sentiment_sha256": sentiment_sha256,
            "decisions_sha256": decisions_sha256,
        }
        event_frame, event_manifest = _load_canonical_identity(
            event_path,
            expected_type="catalyst_events",
            columns=CATALYST_EVENT_COLUMNS,
        )
        _verify_canonical_manifest(
            event_manifest,
            path=event_path,
            artifact_sha256=event_sha256,
            rows=chunk_event_rows,
            expected_inputs=common_inputs,
        )
        chunk_channel_counts = _verify_event_semantics(
            event_frame,
            eligible_channels=eligible_channels,
        )
        if int(_strict_bool_values(event_frame["training_eligible"], "event training eligibility").sum()) != chunk_eligible_rows:
            raise DataReadinessError(f"catalyst eligible event count differs: {chunk_id}")
        assignment_frame, assignment_manifest = _load_canonical_identity(
            assignment_path,
            expected_type="catalyst_event_assignments",
            columns=ASSIGNMENT_COLUMNS,
        )
        _verify_canonical_manifest(
            assignment_manifest,
            path=assignment_path,
            artifact_sha256=assignment_sha256,
            rows=chunk_assignment_rows,
            expected_inputs={
                **common_inputs,
                "catalyst_events_sha256": event_sha256,
                "assignment_sha256": assignment_material_sha256,
            },
        )
        chunk_assignment_counts = _verify_assignment_semantics(
            assignment_frame,
            event_ids=set(event_frame["event_id"].astype(str)),
            expected_material_sha256=assignment_material_sha256,
        )
        for channel, count in chunk_channel_counts.items():
            observed_channel_counts[channel] += count
        for status_name, count in chunk_assignment_counts.items():
            observed_assignment_status_counts[status_name] = observed_assignment_status_counts.get(status_name, 0) + count
        del event_frame, assignment_frame
        event_rows += chunk_event_rows
        eligible_rows += chunk_eligible_rows
        assignment_rows += chunk_assignment_rows

    if (
        event_rows != _verified_nonnegative_int(manifest.get("relation_rows"), "relation_rows")
        or eligible_rows != _verified_nonnegative_int(manifest.get("training_eligible_rows"), "training_eligible_rows")
        or assignment_rows != _verified_nonnegative_int(manifest.get("assignment_rows"), "assignment_rows")
        or inventory.get("event_artifact_count") != len(records)
    ):
        raise DataReadinessError("catalyst lineage aggregate counts do not reconcile")
    if observed_channel_counts != _verified_count_mapping(manifest.get("channel_counts"), "channel_counts"):
        raise DataReadinessError("catalyst lineage channel counts do not reconcile")
    if observed_assignment_status_counts != _verified_count_mapping(
        manifest.get("assignment_status_counts"),
        "assignment_status_counts",
    ):
        raise DataReadinessError("catalyst lineage assignment status counts do not reconcile")

    lineage_sha256 = _verified_sha256(manifest.get("lineage_sha256"), "lineage_sha256")
    lineage_material = {
        "request": request_material,
        "coverage_sha256": coverage_sha256,
        "artifacts": records,
        "feature_inventory": inventory,
    }
    if _json_sha256(lineage_material) != lineage_sha256:
        raise DataReadinessError("catalyst lineage hash does not verify")

    if _json_object(manifest_path) != manifest or _json_object(request_path) != request:
        raise DataReadinessError("catalyst lineage metadata changed during verification")
    if _json_object(status_path) != status or _json_object(inventory_path) != inventory:
        raise DataReadinessError("catalyst lineage metadata changed during verification")
    for relative, observed_sha256 in initial_file_hashes.items():
        if file_sha256(root / Path(relative)) != observed_sha256:
            raise DataReadinessError("catalyst lineage artifact changed during verification")

    coverage_manifest_path = manifest_path_for(coverage_path)
    return VerifiedCatalystLineage(
        directory=root,
        manifest=manifest,
        request=request,
        feature_inventory=inventory,
        coverage=coverage,
        manifest_sha256=file_sha256(manifest_path),
        request_sha256=embedded_request_sha256,
        coverage_sha256=coverage_sha256,
        coverage_manifest_sha256=file_sha256(coverage_manifest_path),
        feature_inventory_sha256=feature_inventory_sha256,
        lineage_sha256=lineage_sha256,
    )


def build_catalyst_lineage(
    *,
    collection_dir: Path,
    collection_audit_path: Path,
    attribution_dir: Path,
    sentiment_dir: Path,
    decisions_path: Path,
    policy_path: Path,
    out_dir: Path,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Join relation and sentiment evidence, then assign eligible rows to decisions."""

    policy = load_catalyst_lineage_policy(policy_path)
    collection_manifest_path = collection_dir / "_manifest.json"
    attribution_manifest_path = attribution_dir / "_manifest.json"
    sentiment_manifest_path = sentiment_dir / "_manifest.json"
    collection = _complete_manifest(collection_manifest_path, "news collection")
    attribution = _complete_manifest(attribution_manifest_path, "event attribution")
    sentiment = _complete_manifest(sentiment_manifest_path, "event sentiment")
    collection_audit = _json_object(collection_audit_path)
    if not bool(collection_audit.get("passed")) or collection_audit.get("request_sha256") != collection.get("request_sha256"):
        raise DataReadinessError("catalyst lineage requires a passed collection audit")
    excluded = _validated_exclusions(collection_audit, attribution, sentiment)
    source_inventory = {str(record["chunk_id"]): record for record in build_source_news_shard_inventory(collection_dir, collection)}
    source_records = _records_by_chunk(collection, "news collection")
    relation_records = _records_by_chunk(attribution, "event attribution")
    sentiment_records = _records_by_chunk(sentiment, "event sentiment")
    source_collections_path = Path(str(collection["source_collections_path"]))
    source_collections, source_collection_manifest = load_canonical_artifact(
        source_collections_path,
        expected_type="source_collections",
        allow_research=True,
    )
    if str(source_collection_manifest.get("artifact_sha256", "")) != str(collection.get("source_collections_sha256", "")) or bool(
        source_collection_manifest.get("production_ready")
    ):
        raise DataReadinessError("collection source-ledger identity is invalid")
    eligible_chunk_ids = sorted(
        chunk_id for chunk_id, record in source_records.items() if str(record.get("security_id", "")) not in excluded
    )
    if set(relation_records) != set(eligible_chunk_ids):
        raise DataReadinessError("event attribution chunk inventory does not match eligible news chunks")
    sentiment_records = _reconcile_sentiment_inventory(
        sentiment_records,
        eligible_chunk_ids=set(eligible_chunk_ids),
        source_collections=source_collections,
        excluded_security_ids=excluded,
        source_inventory=source_inventory,
        sentiment_dir=sentiment_dir,
        sentiment_request_sha256=_required_text(sentiment, "request_sha256"),
    )

    decisions, decision_manifest = load_canonical_artifact(
        decisions_path,
        expected_type="decisions",
        allow_research=True,
        columns=_DECISION_COLUMNS,
    )
    decisions["security_id"] = decisions["security_id"].astype(str).str.strip()
    decision_indices = {str(security_id): indices for security_id, indices in decisions.groupby("security_id", sort=False).indices.items()}
    request = {
        "schema": CATALYST_LINEAGE_REQUEST_SCHEMA,
        "collection_manifest_sha256": file_sha256(collection_manifest_path),
        "collection_audit_sha256": file_sha256(collection_audit_path),
        "attribution_manifest_sha256": file_sha256(attribution_manifest_path),
        "sentiment_manifest_sha256": file_sha256(sentiment_manifest_path),
        "decisions_sha256": str(decision_manifest["artifact_sha256"]),
        "source_collections_sha256": str(source_collection_manifest["artifact_sha256"]),
        "policy_sha256": file_sha256(policy_path),
        "excluded_security_ids": sorted(excluded),
        "production_ready": False,
    }
    request_sha256 = _json_sha256(request)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "_manifest.json"
    if final_path.exists():
        raise DataReadinessError(f"completed catalyst lineage is immutable: {final_path}")
    _write_or_validate_request(out_dir / "_request.json", request, request_sha256)
    event_dir = out_dir / "events"
    assignment_dir = out_dir / "assignments"
    event_dir.mkdir(parents=True, exist_ok=True)
    assignment_dir.mkdir(parents=True, exist_ok=True)

    coverage = _coverage_frame(
        source_collections,
        excluded_security_ids=excluded,
        relation_chunk_ids=set(relation_records),
        sentiment_chunk_ids=set(sentiment_records),
    )
    coverage_path = out_dir / "source_coverage.parquet"
    coverage_manifest = write_canonical_artifact(
        coverage,
        coverage_path,
        artifact_type="catalyst_source_coverage",
        audit=_coverage_audit(coverage),
        inputs={
            "catalyst_lineage_request_sha256": request_sha256,
            "source_collections_sha256": str(source_collection_manifest["artifact_sha256"]),
        },
        production_ready=False,
    )

    observed: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    relation_ids: set[str] = set()
    source_event_ids: set[str] = set()
    channel_counts = {channel: 0 for channel in sorted(_SUPPORTED_CHANNELS)}
    assignment_status_counts: dict[str, int] = {}
    skipped = 0
    for index, chunk_id in enumerate(eligible_chunk_ids, start=1):
        source_events: pd.DataFrame | None = None
        relations: pd.DataFrame | None = None
        sentiments: pd.DataFrame | None = None
        event_frame: pd.DataFrame | None = None
        direct: pd.DataFrame | None = None
        decision_part: pd.DataFrame | None = None
        assignments: pd.DataFrame | None = None
        source_record = source_records[chunk_id]
        relation_record = relation_records[chunk_id]
        sentiment_record = sentiment_records[chunk_id]
        event_target = event_dir / f"{chunk_id}.parquet"
        assignment_target = assignment_dir / f"{chunk_id}.parquet"
        try:
            existing = _load_existing_chunk(
                event_target=event_target,
                assignment_target=assignment_target,
                request_sha256=request_sha256,
                source_record=source_record,
                relation_record=relation_record,
                sentiment_record=sentiment_record,
                decisions=decisions,
                decision_indices=decision_indices,
                policy=policy,
            )
            if existing is not None:
                record, event_frame, assignments = existing
                skipped += 1
            else:
                source_events, source_manifest = load_canonical_artifact(
                    Path(_required_text(source_record, "path")),
                    expected_type="events",
                    allow_research=True,
                )
                relations, relation_manifest = load_canonical_artifact(
                    Path(_required_text(relation_record, "path")),
                    expected_type="event_security_relations",
                    allow_research=True,
                )
                sentiments, sentiment_artifact_manifest = load_canonical_artifact(
                    Path(_required_text(sentiment_record, "path")),
                    expected_type="event_sentiment_research",
                    allow_research=True,
                )
                _verify_chunk_lineage(
                    chunk_id=chunk_id,
                    source_record=source_record,
                    relation_record=relation_record,
                    sentiment_record=sentiment_record,
                    source_manifest=source_manifest,
                    relation_manifest=relation_manifest,
                    sentiment_manifest=sentiment_artifact_manifest,
                    source_events=source_events,
                    relations=relations,
                    sentiments=sentiments,
                )
                event_frame = _join_catalyst_events(relations, sentiments, policy=policy)
                direct = event_frame.loc[event_frame["training_eligible"].astype(bool)].copy()
                target_security_ids = direct["security_id"].astype(str).unique()
                parts = [
                    decisions.iloc[decision_indices[security_id]] for security_id in target_security_ids if security_id in decision_indices
                ]
                decision_part = pd.concat(parts, ignore_index=True) if parts else decisions.iloc[0:0].copy()
                assignments = build_event_assignments(
                    decision_part,
                    direct,
                    windows=policy.assignment_windows,
                )
                assignment_integrity = assignment_integrity_summary(
                    decision_part,
                    direct,
                    assignments,
                    windows=policy.assignment_windows,
                )
                if assignment_integrity["assignment_integrity_errors"]:
                    raise DataReadinessError(f"assignment replay mismatch for {chunk_id}")
                event_manifest = write_canonical_artifact(
                    event_frame,
                    event_target,
                    artifact_type="catalyst_events",
                    audit=_catalyst_event_audit(event_frame, policy),
                    inputs=_chunk_inputs(
                        request_sha256,
                        source_record,
                        relation_record,
                        sentiment_record,
                        decision_manifest,
                    ),
                    production_ready=False,
                )
                assignment_manifest = write_canonical_artifact(
                    assignments,
                    assignment_target,
                    artifact_type="catalyst_event_assignments",
                    audit=_assignment_audit(assignments, assignment_integrity),
                    inputs={
                        **_chunk_inputs(
                            request_sha256,
                            source_record,
                            relation_record,
                            sentiment_record,
                            decision_manifest,
                        ),
                        "catalyst_events_sha256": str(event_manifest["artifact_sha256"]),
                        "assignment_sha256": reconciliation_sha256(assignments),
                    },
                    production_ready=False,
                )
                record = _chunk_record(
                    chunk_id=chunk_id,
                    event_path=event_target,
                    assignment_path=assignment_target,
                    event_frame=event_frame,
                    assignments=assignments,
                    event_manifest=event_manifest,
                    assignment_manifest=assignment_manifest,
                    source_record=source_record,
                    relation_record=relation_record,
                    sentiment_record=sentiment_record,
                )
            _accumulate_unique_ids(
                relation_ids,
                event_frame["relation_id"],
                "relation",
                chunk_id,
            )
            _accumulate_unique_ids(
                source_event_ids,
                event_frame["source_event_id"],
                "related source event",
                chunk_id,
                allow_repeated=True,
            )
            for channel, count in event_frame["relation_channel"].value_counts().items():
                channel_counts[str(channel)] += int(count)
            for status, count in assignments["status"].value_counts().items():
                assignment_status_counts[str(status)] = assignment_status_counts.get(str(status), 0) + int(count)
            observed.append(record)
            _progress(
                progress,
                index=index,
                total=len(eligible_chunk_ids),
                chunk_id=chunk_id,
                status="skipped" if existing is not None else "observed",
                rows=len(event_frame),
            )
        except Exception as exc:
            failures[chunk_id] = f"{type(exc).__name__}: {str(exc)[:500]}"
            _progress(
                progress,
                index=index,
                total=len(eligible_chunk_ids),
                chunk_id=chunk_id,
                status="failed",
                rows=0,
            )
        finally:
            source_events = None
            relations = None
            sentiments = None
            event_frame = None
            direct = None
            decision_part = None
            assignments = None
            gc.collect()
            release_process_memory()
            assert_memory_budget(
                hard_budget_gib=policy.maximum_process_memory_gib,
                headroom_gib=policy.memory_guard_headroom_gib,
                stage=f"catalyst lineage chunk {chunk_id}",
            )

    inventory = _feature_inventory(
        policy,
        request_sha256=request_sha256,
        channel_counts=channel_counts,
        coverage=coverage,
        event_records=observed,
    )
    inventory_path = out_dir / "feature_inventory.json"
    _atomic_json(inventory_path, inventory)
    status = "complete" if not failures and len(observed) == len(eligible_chunk_ids) else "incomplete"
    result: dict[str, object] = {
        "schema": CATALYST_LINEAGE_MANIFEST_SCHEMA,
        "request_sha256": request_sha256,
        "status": status,
        "requested_chunks": len(eligible_chunk_ids),
        "observed_chunks": len(observed),
        "skipped_chunks": skipped,
        "failed_chunks": failures,
        "excluded_security_ids": sorted(excluded),
        "source_event_rows": _required_int(sentiment, "total_rows"),
        "related_source_events": len(source_event_ids),
        "relation_rows": sum(_required_int(record, "event_rows") for record in observed),
        "training_eligible_rows": sum(_required_int(record, "training_eligible_rows") for record in observed),
        "channel_counts": channel_counts,
        "assignment_rows": sum(_required_int(record, "assignment_rows") for record in observed),
        "assignment_status_counts": dict(sorted(assignment_status_counts.items())),
        "coverage": {
            "path": str(coverage_path.resolve()),
            "sha256": str(coverage_manifest["artifact_sha256"]),
            "rows": len(coverage),
            "states": {str(key): int(value) for key, value in coverage["coverage_state"].value_counts().sort_index().items()},
        },
        "feature_inventory": {
            "path": str(inventory_path.resolve()),
            "sha256": file_sha256(inventory_path),
        },
        "artifacts": sorted(observed, key=lambda item: str(item["chunk_id"])),
        "lineage_sha256": _lineage_sha256(request, coverage_manifest, observed, inventory),
        "memory": memory_audit(
            hard_budget_gib=policy.maximum_process_memory_gib,
            headroom_gib=policy.memory_guard_headroom_gib,
        ).to_record(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "production_ready": False,
    }
    _atomic_json(out_dir / "_status.json", result)
    if status == "complete":
        _atomic_json(final_path, result)
    return result


def _verify_chunk_lineage(
    *,
    chunk_id: str,
    source_record: Mapping[str, object],
    relation_record: Mapping[str, object],
    sentiment_record: Mapping[str, object],
    source_manifest: Mapping[str, object],
    relation_manifest: Mapping[str, object],
    sentiment_manifest: Mapping[str, object],
    source_events: pd.DataFrame,
    relations: pd.DataFrame,
    sentiments: pd.DataFrame,
) -> None:
    source_sha256 = _required_text(source_record, "sha256")
    if (
        str(source_manifest.get("artifact_sha256")) != source_sha256
        or _required_text(relation_record, "source_event_sha256") != source_sha256
        or _required_text(sentiment_record, "source_event_artifact_sha256") != source_sha256
    ):
        raise DataReadinessError(f"source-event lineage hash mismatch for {chunk_id}")
    relation_inputs = relation_manifest.get("inputs")
    sentiment_inputs = sentiment_manifest.get("inputs")
    if (
        not isinstance(relation_inputs, dict)
        or relation_inputs.get("source_event_artifact_sha256") != source_sha256
        or not isinstance(sentiment_inputs, dict)
        or sentiment_inputs.get("source_event_artifact_sha256") != source_sha256
    ):
        raise DataReadinessError(f"child artifact lineage mismatch for {chunk_id}")
    source_ids = source_events["event_id"].astype(str)
    sentiment_ids = sentiments["event_id"].astype(str)
    relation_event_ids = relations["event_id"].astype(str)
    if bool(source_ids.duplicated().any() or sentiment_ids.duplicated().any()):
        raise DataReadinessError(f"duplicate source or sentiment event IDs for {chunk_id}")
    if set(source_ids) != set(sentiment_ids):
        raise DataReadinessError(f"sentiment event inventory mismatch for {chunk_id}")
    if not set(relation_event_ids).issubset(set(source_ids)):
        raise DataReadinessError(f"relation references an unrelated event for {chunk_id}")
    if bool(relations["relation_id"].astype(str).duplicated().any()):
        raise DataReadinessError(f"duplicate relation IDs for {chunk_id}")
    source_identity = source_events.set_index("event_id")[["security_id", "ticker"]]
    if not relations.empty:
        expected = source_identity.loc[relation_event_ids].reset_index(drop=True)
        source_security = relations["source_security_id"].astype(str).reset_index(drop=True)
        source_ticker = relations["source_ticker"].astype(str).str.upper().reset_index(drop=True)
        if bool(
            source_security.ne(expected["security_id"].astype(str)).any()
            or source_ticker.ne(expected["ticker"].astype(str).str.upper()).any()
        ):
            raise DataReadinessError(f"relation source identity mismatch for {chunk_id}")


def _join_catalyst_events(
    relations: pd.DataFrame,
    sentiments: pd.DataFrame,
    *,
    policy: CatalystLineagePolicy,
) -> pd.DataFrame:
    if relations.empty:
        return pd.DataFrame(columns=CATALYST_EVENT_COLUMNS)
    joined = relations.merge(
        sentiments,
        on="event_id",
        how="left",
        validate="many_to_one",
        suffixes=("_relation", "_sentiment"),
    )
    if bool(joined["sentiment_numeric"].isna().any()):
        raise DataReadinessError("relation rows have missing sentiment")
    channels = joined["relation_channel"].astype(str)
    invalid_channels = sorted(set(channels).difference(_SUPPORTED_CHANNELS))
    if invalid_channels:
        raise DataReadinessError(f"unsupported relation channels: {invalid_channels}")
    relation_available = _strict_utc(joined["feature_available_at_utc"], "relation feature availability")
    sentiment_available = _strict_utc(
        joined["research_feature_available_at_utc"],
        "sentiment feature availability",
    )
    event_available = _strict_utc(joined["event_available_at_utc"], "event availability")
    published = _strict_utc(joined["published_at_utc"], "event publication")
    event_relation_available = _strict_utc(
        joined["event_feature_available_at_utc"],
        "relation event availability",
    )
    identity_available = pd.to_datetime(joined["identity_available_at_utc"], utc=True, errors="coerce")
    label_available = pd.to_datetime(joined["label_available_at_utc"], utc=True, errors="coerce")
    if bool(
        published.gt(event_available).any()
        or event_relation_available.gt(relation_available).any()
        or (identity_available.notna() & identity_available.gt(relation_available)).any()
        or (label_available.notna() & label_available.gt(relation_available)).any()
        or sentiment_available.lt(event_available).any()
    ):
        raise DataReadinessError("catalyst lineage contains backdated availability")
    feature_available = pd.concat(
        [relation_available.rename("relation"), sentiment_available.rename("sentiment")],
        axis=1,
    ).max(axis=1)
    eligible = channels.isin(policy.training_eligible_channels)
    exclusion = channels.map(
        {
            "direct_issuer": "",
            "business_exposure": "historical_business_evidence_not_training_ready",
            "sector_context": "sector_context_not_direct_issuer_evidence",
        }
    )
    output = pd.DataFrame(
        {
            "event_id": joined["relation_id"].astype(str),
            "source_event_id": joined["event_id"].astype(str),
            "relation_id": joined["relation_id"].astype(str),
            "source_security_id": joined["source_security_id"].astype(str),
            "source_ticker": joined["source_ticker"].astype(str).str.upper(),
            "security_id": joined["target_security_id"].astype(str),
            "ticker": joined["target_ticker"].astype(str).str.upper(),
            "source_family": joined["source_family"].astype(str).str.lower(),
            "published_at_utc": published,
            "event_available_at_utc": event_available,
            "relation_feature_available_at_utc": relation_available,
            "sentiment_feature_available_at_utc": sentiment_available,
            "feature_available_at_utc": feature_available,
            "availability_policy": policy.availability_policy,
            "relation_channel": channels,
            "relation_score": pd.to_numeric(joined["relation_score"], errors="coerce"),
            "relation_basis": joined["relation_basis"].astype(str),
            "sentiment_label": joined["sentiment_label"].astype(str),
            "sentiment_confidence": pd.to_numeric(joined["sentiment_confidence"], errors="coerce"),
            "sentiment_numeric": pd.to_numeric(joined["sentiment_numeric"], errors="coerce"),
            "relevance": pd.to_numeric(joined["relation_score"], errors="coerce"),
            "source_relevance": pd.to_numeric(joined["relevance"], errors="coerce"),
            "source_relevance_basis": joined["relevance_basis"].astype(str),
            "sentiment_input_sha256": joined["sentiment_input_sha256"].astype(str),
            "sentiment_model": joined["sentiment_model"].astype(str),
            "sentiment_model_revision": joined["sentiment_model_revision"].astype(str),
            "attribution_policy_version": joined["attribution_policy_version"].astype(str),
            "attribution_policy_sha256": joined["attribution_policy_sha256"].astype(str),
            "training_eligible": eligible,
            "training_exclusion_reason": exclusion,
            "schema_version": CATALYST_EVENT_SCHEMA,
        },
        columns=CATALYST_EVENT_COLUMNS,
    )
    return output.sort_values(
        ["source_event_id", "security_id", "relation_channel"],
        kind="stable",
    ).reset_index(drop=True)


def _coverage_frame(
    source_collections: pd.DataFrame,
    *,
    excluded_security_ids: set[str],
    relation_chunk_ids: set[str],
    sentiment_chunk_ids: set[str],
) -> pd.DataFrame:
    output = source_collections[
        [
            "collection_id",
            "chunk_id",
            "security_id",
            "ticker",
            "source_family",
            "requested_start_utc",
            "requested_end_utc",
            "status",
            "row_count",
        ]
    ].copy()
    blind = output["security_id"].astype(str).isin(excluded_security_ids)
    empty = output["status"].astype(str).eq("observed_empty")
    complete = output["status"].astype(str).eq("observed")
    has_relation = output["chunk_id"].astype(str).isin(relation_chunk_ids)
    has_sentiment = output["chunk_id"].astype(str).isin(sentiment_chunk_ids)
    output["coverage_state"] = "failed_or_unobserved"
    output.loc[complete & has_relation & has_sentiment, "coverage_state"] = "observed_complete"
    output.loc[empty, "coverage_state"] = "observed_empty"
    output.loc[blind, "coverage_state"] = "coverage_blindspot"
    output["missingness_known"] = output["coverage_state"].isin({"observed_complete", "observed_empty"})
    output["zero_event_semantics"] = output["coverage_state"].map(
        {
            "observed_complete": "observed_history",
            "observed_empty": "known_zero_events",
            "coverage_blindspot": "unknown_excluded",
            "failed_or_unobserved": "unknown_failed",
        }
    )
    output["training_eligible"] = output["missingness_known"]
    output["schema_version"] = CATALYST_COVERAGE_SCHEMA
    return output


def _catalyst_event_audit(
    frame: pd.DataFrame,
    policy: CatalystLineagePolicy,
) -> CanonicalAuditReport:
    failures = 0
    if list(frame.columns) != list(CATALYST_EVENT_COLUMNS):
        failures += 1
    if not frame.empty:
        available = _strict_utc(frame["feature_available_at_utc"], "catalyst feature availability")
        relation_available = _strict_utc(
            frame["relation_feature_available_at_utc"],
            "relation feature availability",
        )
        sentiment_available = _strict_utc(
            frame["sentiment_feature_available_at_utc"],
            "sentiment feature availability",
        )
        failures += int(frame["event_id"].astype(str).duplicated().sum())
        failures += int((available < relation_available).sum())
        failures += int((available < sentiment_available).sum())
        failures += int(
            (frame["training_eligible"].astype(bool) != frame["relation_channel"].astype(str).isin(policy.training_eligible_channels)).sum()
        )
    return _audit_report(
        "catalyst_events",
        failures,
        len(frame),
        "identity, relevance, sentiment, availability, and eligibility reconcile",
    )


def _assignment_audit(
    assignments: pd.DataFrame,
    integrity: Mapping[str, int],
) -> CanonicalAuditReport:
    summary = reconciliation_summary(assignments)
    failures = int(integrity["assignment_integrity_errors"]) + int(summary["lineage_error_events"])
    assigned = assignments.loc[assignments["status"].astype(str).eq("assigned")]
    if not assigned.empty:
        event_available = _strict_utc(assigned["feature_available_at_utc"], "assigned event availability")
        decision_time = _strict_utc(assigned["decision_time_utc"], "assigned decision time")
        failures += int((event_available > decision_time).sum())
        age = decision_time - event_available
        failures += int((age.dt.total_seconds() > pd.to_numeric(assigned["window_seconds"], errors="coerce")).sum())
    return _audit_report(
        "catalyst_event_assignments",
        failures,
        len(assignments),
        "deterministic assignment replay, cutoff, and window checks pass",
    )


def _coverage_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    failures = int(frame["coverage_state"].eq("failed_or_unobserved").sum())
    failures += int((frame["training_eligible"].astype(bool) & ~frame["missingness_known"].astype(bool)).sum())
    return _audit_report(
        "catalyst_source_coverage",
        failures,
        len(frame),
        "observed-empty and unavailable source windows remain distinct",
    )


def _audit_report(
    name: str,
    failures: int,
    rows: int,
    detail: str,
) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name=name,
                status="pass" if failures == 0 else "fail",
                failures=failures,
                rows_checked=rows,
                detail=detail,
            ),
        )
    )


def _feature_inventory(
    policy: CatalystLineagePolicy,
    *,
    request_sha256: str,
    channel_counts: Mapping[str, int],
    coverage: pd.DataFrame,
    event_records: list[dict[str, object]],
) -> dict[str, object]:
    windows = list(policy.assignment_windows)
    profiles = {
        name: sorted({template.format(window=window) for template in templates for window in windows})
        for name, templates in policy.feature_profiles.items()
    }
    return {
        "schema": FEATURE_INVENTORY_SCHEMA,
        "request_sha256": request_sha256,
        "profiles": profiles,
        "training_eligible_channels": list(policy.training_eligible_channels),
        "research_only_channels": list(policy.research_only_channels),
        "availability_policy": policy.availability_policy,
        "channel_counts": dict(channel_counts),
        "coverage_states": {str(key): int(value) for key, value in coverage["coverage_state"].value_counts().sort_index().items()},
        "event_artifact_count": len(event_records),
        "training_contract": (
            "Only direct-issuer rows from source-complete windows may produce catalyst-only or technical-plus-catalyst features."
        ),
        "production_ready": False,
    }


def _load_existing_chunk(
    *,
    event_target: Path,
    assignment_target: Path,
    request_sha256: str,
    source_record: Mapping[str, object],
    relation_record: Mapping[str, object],
    sentiment_record: Mapping[str, object],
    decisions: pd.DataFrame,
    decision_indices: Mapping[str, object],
    policy: CatalystLineagePolicy,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame] | None:
    event_exists = event_target.exists() or manifest_path_for(event_target).exists()
    assignment_exists = assignment_target.exists() or manifest_path_for(assignment_target).exists()
    if event_exists != assignment_exists:
        raise DataReadinessError(f"partial catalyst lineage chunk exists: {event_target.stem}")
    if not event_exists:
        return None
    events, event_manifest = load_canonical_artifact(
        event_target,
        expected_type="catalyst_events",
        allow_research=True,
    )
    assignments, assignment_manifest = load_canonical_artifact(
        assignment_target,
        expected_type="catalyst_event_assignments",
        allow_research=True,
    )
    expected_inputs = _chunk_inputs(
        request_sha256,
        source_record,
        relation_record,
        sentiment_record,
        {"artifact_sha256": _required_text_from_manifest_input(assignment_manifest, "decisions_sha256")},
    )
    inputs = event_manifest.get("inputs")
    if not isinstance(inputs, dict) or any(inputs.get(key) != value for key, value in expected_inputs.items()):
        raise DataReadinessError(f"existing catalyst event lineage mismatch: {event_target}")
    assignment_inputs = assignment_manifest.get("inputs")
    if (
        not isinstance(assignment_inputs, dict)
        or assignment_inputs.get("catalyst_events_sha256") != event_manifest.get("artifact_sha256")
        or assignment_inputs.get("assignment_sha256") != reconciliation_sha256(assignments)
    ):
        raise DataReadinessError(f"existing catalyst assignment lineage mismatch: {assignment_target}")
    event_audit = _catalyst_event_audit(events, policy)
    event_audit.raise_for_failure()
    direct = events.loc[events["training_eligible"].astype(bool)]
    target_security_ids = direct["security_id"].astype(str).unique()
    parts = [decisions.iloc[decision_indices[security_id]] for security_id in target_security_ids if security_id in decision_indices]
    decision_part = pd.concat(parts, ignore_index=True) if parts else decisions.iloc[0:0].copy()
    integrity = assignment_integrity_summary(
        decision_part,
        direct,
        assignments,
        windows=policy.assignment_windows,
    )
    assignment_audit = _assignment_audit(assignments, integrity)
    assignment_audit.raise_for_failure()
    return (
        _chunk_record(
            chunk_id=event_target.stem,
            event_path=event_target,
            assignment_path=assignment_target,
            event_frame=events,
            assignments=assignments,
            event_manifest=event_manifest,
            assignment_manifest=assignment_manifest,
            source_record=source_record,
            relation_record=relation_record,
            sentiment_record=sentiment_record,
        ),
        events,
        assignments,
    )


def _chunk_inputs(
    request_sha256: str,
    source_record: Mapping[str, object],
    relation_record: Mapping[str, object],
    sentiment_record: Mapping[str, object],
    decision_manifest: Mapping[str, object],
) -> dict[str, str]:
    return {
        "catalyst_lineage_request_sha256": request_sha256,
        "source_event_sha256": _required_text(source_record, "sha256"),
        "relation_sha256": _required_text(relation_record, "sha256"),
        "sentiment_sha256": _required_text(sentiment_record, "sha256"),
        "decisions_sha256": str(decision_manifest["artifact_sha256"]),
    }


def _chunk_record(
    *,
    chunk_id: str,
    event_path: Path,
    assignment_path: Path,
    event_frame: pd.DataFrame,
    assignments: pd.DataFrame,
    event_manifest: Mapping[str, object],
    assignment_manifest: Mapping[str, object],
    source_record: Mapping[str, object],
    relation_record: Mapping[str, object],
    sentiment_record: Mapping[str, object],
) -> dict[str, object]:
    return {
        "chunk_id": chunk_id,
        "source_event_sha256": _required_text(source_record, "sha256"),
        "relation_sha256": _required_text(relation_record, "sha256"),
        "sentiment_sha256": _required_text(sentiment_record, "sha256"),
        "event_path": str(event_path.resolve()),
        "event_sha256": str(event_manifest["artifact_sha256"]),
        "event_rows": len(event_frame),
        "training_eligible_rows": int(event_frame["training_eligible"].astype(bool).sum()),
        "assignment_path": str(assignment_path.resolve()),
        "assignment_sha256": str(assignment_manifest["artifact_sha256"]),
        "assignment_material_sha256": reconciliation_sha256(assignments),
        "assignment_rows": len(assignments),
    }


def _lineage_sha256(
    request: Mapping[str, object],
    coverage_manifest: Mapping[str, object],
    records: list[dict[str, object]],
    inventory: Mapping[str, object],
) -> str:
    material = {
        "request": request,
        "coverage_sha256": coverage_manifest["artifact_sha256"],
        "artifacts": sorted(records, key=lambda item: str(item["chunk_id"])),
        "feature_inventory": inventory,
    }
    return _json_sha256(material)


def _validated_exclusions(
    collection_audit: Mapping[str, object],
    attribution: Mapping[str, object],
    sentiment: Mapping[str, object],
) -> set[str]:
    values: list[set[str]] = []
    for payload, key in (
        (collection_audit, "coverage_blindspot_security_ids"),
        (attribution, "excluded_security_ids"),
        (sentiment, "excluded_security_ids"),
    ):
        raw = payload.get(key)
        if not isinstance(raw, list):
            raise DataReadinessError(f"malformed catalyst exclusion inventory: {key}")
        values.append({str(value) for value in raw})
    if values[0] != values[1] or values[0] != values[2]:
        raise DataReadinessError("catalyst exclusion inventories do not reconcile")
    return values[0]


def _records_by_chunk(
    manifest: Mapping[str, object],
    name: str,
) -> dict[str, dict[str, object]]:
    raw = manifest.get("artifacts")
    if not isinstance(raw, list):
        raise DataReadinessError(f"{name} has no artifact inventory")
    records: dict[str, dict[str, object]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise DataReadinessError(f"{name} artifact inventory is malformed")
        record = {str(key): value for key, value in item.items()}
        chunk_id = _required_text(record, "chunk_id")
        if chunk_id in records:
            raise DataReadinessError(f"{name} has duplicate chunk IDs")
        records[chunk_id] = record
    return records


def _reconcile_sentiment_inventory(
    records: Mapping[str, dict[str, object]],
    *,
    eligible_chunk_ids: set[str],
    source_collections: pd.DataFrame,
    excluded_security_ids: set[str],
    source_inventory: Mapping[str, Mapping[str, object]],
    sentiment_dir: Path,
    sentiment_request_sha256: str,
) -> dict[str, dict[str, object]]:
    missing = eligible_chunk_ids.difference(records)
    if missing:
        raise DataReadinessError("sentiment chunk inventory does not match eligible news chunks")
    extras = set(records).difference(eligible_chunk_ids)
    if not extras:
        return dict(records)

    required = {"chunk_id", "security_id", "ticker", "status", "row_count"}
    if not required.issubset(source_collections.columns):
        raise DataReadinessError("source collection inventory cannot reconcile empty sentiment chunks")
    source_rows = source_collections.loc[
        source_collections["chunk_id"].astype(str).isin(extras),
        list(required),
    ].copy()
    if bool(source_rows["chunk_id"].astype(str).duplicated().any()):
        raise DataReadinessError("source collection inventory has duplicate chunk IDs")
    source_by_chunk = {str(row["chunk_id"]): row for row in source_rows.to_dict(orient="records")}
    for chunk_id in extras:
        source = source_by_chunk.get(chunk_id)
        sentiment = records[chunk_id]
        source_evidence = source_inventory.get(chunk_id)
        if (
            source is None
            or source_evidence is None
            or not bool(source_evidence.get("source_empty"))
            or str(source["security_id"]) in excluded_security_ids
            or str(source["status"]) != "observed_empty"
            or _required_int(source, "row_count") != 0
            or _required_int(sentiment, "rows") != 0
            or str(sentiment.get("security_id", "")) != str(source["security_id"])
            or str(sentiment.get("ticker", "")).upper() != str(source.get("ticker", "")).upper()
            or _required_text(sentiment, "source_event_artifact_sha256") != _required_text(source_evidence, "sha256")
        ):
            raise DataReadinessError("sentiment chunk inventory does not match eligible news chunks")
        _validate_empty_sentiment_artifact(
            chunk_id=chunk_id,
            record=sentiment,
            sentiment_dir=sentiment_dir,
            sentiment_request_sha256=sentiment_request_sha256,
            source_evidence_sha256=_required_text(source_evidence, "sha256"),
        )
    return {chunk_id: records[chunk_id] for chunk_id in eligible_chunk_ids}


def _validate_empty_sentiment_artifact(
    *,
    chunk_id: str,
    record: Mapping[str, object],
    sentiment_dir: Path,
    sentiment_request_sha256: str,
    source_evidence_sha256: str,
) -> None:
    artifact_path = Path(_required_text(record, "path"))
    expected_parent = (sentiment_dir / "sentiment").resolve()
    resolved = artifact_path.resolve()
    if resolved.parent != expected_parent or resolved.name != f"{chunk_id}.parquet":
        raise DataReadinessError(f"empty sentiment artifact path mismatch for {chunk_id}")
    frame, manifest = load_canonical_artifact(
        resolved,
        expected_type="event_sentiment_research",
        allow_research=True,
    )
    inputs = manifest.get("inputs")
    if (
        not frame.empty
        or str(manifest.get("artifact_sha256", "")) != _required_text(record, "sha256")
        or not isinstance(inputs, dict)
        or inputs.get("chunk_id") != chunk_id
        or inputs.get("sentiment_request_sha256") != sentiment_request_sha256
        or inputs.get("source_event_artifact_sha256") != source_evidence_sha256
    ):
        raise DataReadinessError(f"empty sentiment artifact integrity mismatch for {chunk_id}")


def _complete_manifest(path: Path, name: str) -> dict[str, object]:
    manifest = _json_object(path)
    if manifest.get("status") != "complete" or bool(manifest.get("production_ready")):
        raise DataReadinessError(f"{name} must be a completed research-only artifact")
    return manifest


def _accumulate_unique_ids(
    observed: set[str],
    values: pd.Series,
    name: str,
    chunk_id: str,
    *,
    allow_repeated: bool = False,
) -> None:
    current = set(values.astype(str))
    if len(current) != len(values) and not allow_repeated:
        raise DataReadinessError(f"duplicate {name} IDs within {chunk_id}")
    overlap = observed.intersection(current)
    if overlap and not allow_repeated:
        raise DataReadinessError(f"duplicate {name} IDs across chunks: {chunk_id}")
    observed.update(current)


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise DataReadinessError(f"{name} must be a non-empty list")
    output = tuple(str(item).strip() for item in value)
    if any(not item for item in output) or len(set(output)) != len(output):
        raise DataReadinessError(f"{name} contains empty or duplicate values")
    return output


def _strict_utc(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if bool(parsed.isna().any()):
        raise DataReadinessError(f"{name} contains invalid timestamps")
    return parsed


def _required_text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DataReadinessError(f"artifact record has no {key}")
    return value


def _required_int(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataReadinessError(f"artifact record has no integer {key}")
    return value


def _required_text_from_manifest_input(
    manifest: Mapping[str, object],
    key: str,
) -> str:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise DataReadinessError("canonical artifact has no input lineage")
    value = inputs.get(key)
    if not isinstance(value, str) or not value:
        raise DataReadinessError(f"canonical artifact has no {key}")
    return value


def _verified_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise DataReadinessError(f"catalyst lineage {name} must be an object")
    return {str(key): item for key, item in value.items()}


def _verified_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataReadinessError(f"catalyst lineage {name} must be non-empty text")
    return value


def _verified_sha256(value: object, name: str) -> str:
    text = _verified_text(value, name)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise DataReadinessError(f"catalyst lineage {name} must be a lowercase SHA-256")
    return text


def _verified_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise DataReadinessError(f"catalyst lineage {name} must be a nonnegative integer")
    return value


def _verified_chunk_id(value: object) -> str:
    chunk_id = _verified_text(value, "chunk_id")
    if Path(chunk_id).name != chunk_id or chunk_id in {".", ".."}:
        raise DataReadinessError("catalyst lineage chunk_id is unsafe")
    return chunk_id


def _verified_lineage_records(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    raw_records = manifest.get("artifacts")
    if not isinstance(raw_records, list):
        raise DataReadinessError("catalyst lineage artifacts must be a list")
    records: list[dict[str, object]] = []
    chunk_ids: set[str] = set()
    for raw_record in raw_records:
        record = _verified_mapping(raw_record, "artifact record")
        if set(record) != _ARTIFACT_RECORD_KEYS:
            raise DataReadinessError("catalyst lineage artifact record fields do not match the contract")
        chunk_id = _verified_chunk_id(record.get("chunk_id"))
        if chunk_id in chunk_ids:
            raise DataReadinessError("catalyst lineage contains duplicate chunk IDs")
        chunk_ids.add(chunk_id)
        records.append(record)
    return sorted(records, key=lambda record: str(record["chunk_id"]))


def _load_canonical_identity(
    path: Path,
    *,
    expected_type: str,
    columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, dict[str, object]]:
    return load_canonical_artifact(
        path,
        expected_type=expected_type,
        allow_research=True,
        columns=columns,
    )


def _verify_feature_inventory(
    inventory: Mapping[str, object],
    *,
    manifest: Mapping[str, object],
) -> None:
    if set(inventory) != _FEATURE_INVENTORY_KEYS:
        raise DataReadinessError("catalyst feature inventory fields differ")
    eligible = inventory.get("training_eligible_channels")
    research_only = inventory.get("research_only_channels")
    if not isinstance(eligible, list) or not isinstance(research_only, list):
        raise DataReadinessError("catalyst feature inventory channel policy is malformed")
    eligible_set = {str(value) for value in eligible}
    research_set = {str(value) for value in research_only}
    if (
        eligible_set != _TRAINING_ELIGIBLE_CHANNELS
        or research_set != _RESEARCH_ONLY_CHANNELS
        or eligible_set.intersection(research_set)
        or eligible_set.union(research_set) != _SUPPORTED_CHANNELS
    ):
        raise DataReadinessError("catalyst feature inventory channel policy differs")
    availability = inventory.get("availability_policy")
    if not isinstance(availability, str) or not availability.strip():
        raise DataReadinessError("catalyst feature inventory availability policy is invalid")
    if _verified_count_mapping(inventory.get("channel_counts"), "inventory channel_counts") != _verified_count_mapping(
        manifest.get("channel_counts"),
        "channel_counts",
    ):
        raise DataReadinessError("catalyst feature inventory channel counts differ")
    coverage_record = _verified_mapping(manifest.get("coverage"), "coverage")
    if _verified_count_mapping(inventory.get("coverage_states"), "inventory coverage_states") != _verified_count_mapping(
        coverage_record.get("states"),
        "coverage states",
    ):
        raise DataReadinessError("catalyst feature inventory coverage states differ")
    profiles = inventory.get("profiles")
    if not isinstance(profiles, dict) or set(profiles) != {"catalyst_only", "technical_plus_catalyst"}:
        raise DataReadinessError("catalyst feature inventory profiles differ")


def _verify_coverage_semantics(
    coverage: pd.DataFrame,
    *,
    manifest: Mapping[str, object],
) -> None:
    if tuple(coverage.columns) != CATALYST_COVERAGE_COLUMNS:
        raise DataReadinessError("catalyst coverage columns differ")
    if coverage["chunk_id"].astype(str).duplicated().any():
        raise DataReadinessError("catalyst coverage contains duplicate chunk identities")
    states = coverage["coverage_state"].astype(str)
    allowed_states = {
        "observed_complete",
        "observed_empty",
        "coverage_blindspot",
        "failed_or_unobserved",
    }
    if not states.isin(allowed_states).all():
        raise DataReadinessError("catalyst coverage contains an unsupported state")
    known = _strict_bool_values(coverage["missingness_known"], "coverage missingness")
    eligible = _strict_bool_values(coverage["training_eligible"], "coverage training eligibility")
    if not eligible.eq(known).all():
        raise DataReadinessError("catalyst coverage eligibility contradicts missingness")
    status = coverage["status"].astype(str)
    row_count = pd.to_numeric(coverage["row_count"], errors="coerce")
    if row_count.isna().any() or row_count.lt(0).any() or row_count.mod(1).ne(0).any():
        raise DataReadinessError("catalyst coverage row counts are invalid")
    complete = states.eq("observed_complete")
    empty = states.eq("observed_empty")
    unknown = ~(complete | empty)
    if (
        (~status.loc[complete].eq("observed")).any()
        or (~status.loc[empty].eq("observed_empty")).any()
        or row_count.loc[empty].ne(0).any()
        or (~known.loc[complete | empty]).any()
        or known.loc[unknown].any()
    ):
        raise DataReadinessError("catalyst coverage state semantics do not reconcile")
    if not coverage["schema_version"].astype(str).eq(CATALYST_COVERAGE_SCHEMA).all():
        raise DataReadinessError("catalyst coverage schema differs")
    coverage_record = _verified_mapping(manifest.get("coverage"), "coverage")
    expected_states = _verified_count_mapping(coverage_record.get("states"), "coverage states")
    observed_states = {
        str(key): int(value)
        for key, value in states.value_counts().sort_index().items()
    }
    if observed_states != expected_states:
        raise DataReadinessError("catalyst coverage state counts do not reconcile")


def _verify_event_semantics(
    events: pd.DataFrame,
    *,
    eligible_channels: tuple[str, ...],
) -> dict[str, int]:
    if tuple(events.columns) != CATALYST_EVENT_COLUMNS:
        raise DataReadinessError("catalyst event columns differ")
    event_ids = events["event_id"].astype(str)
    if event_ids.str.strip().eq("").any() or event_ids.duplicated().any():
        raise DataReadinessError("catalyst events have invalid identities")
    channels = events["relation_channel"].astype(str)
    if not channels.isin(_SUPPORTED_CHANNELS).all():
        raise DataReadinessError("catalyst events contain unsupported relation channels")
    eligible = _strict_bool_values(events["training_eligible"], "event training eligibility")
    if not eligible.eq(channels.isin(eligible_channels)).all():
        raise DataReadinessError("catalyst event eligibility contradicts its relation channel")
    published = _strict_utc(events["published_at_utc"], "event publication")
    event_available = _strict_utc(events["event_available_at_utc"], "event availability")
    relation_available = _strict_utc(events["relation_feature_available_at_utc"], "relation availability")
    sentiment_available = _strict_utc(events["sentiment_feature_available_at_utc"], "sentiment availability")
    feature_available = _strict_utc(events["feature_available_at_utc"], "feature availability")
    expected_feature_available = pd.concat(
        [relation_available, sentiment_available],
        axis=1,
    ).max(axis=1)
    if (
        published.gt(event_available).any()
        or relation_available.lt(event_available).any()
        or sentiment_available.lt(event_available).any()
        or not feature_available.eq(expected_feature_available).all()
    ):
        raise DataReadinessError("catalyst event availability is not causal")
    if not events["schema_version"].astype(str).eq(CATALYST_EVENT_SCHEMA).all():
        raise DataReadinessError("catalyst event schema differs")
    return {
        str(channel): int(count)
        for channel, count in channels.value_counts().sort_index().items()
    }


def _verify_assignment_semantics(
    assignments: pd.DataFrame,
    *,
    event_ids: set[str],
    expected_material_sha256: str,
) -> dict[str, int]:
    if tuple(assignments.columns) != ASSIGNMENT_COLUMNS:
        raise DataReadinessError("catalyst assignment columns differ")
    assignment_ids = assignments["assignment_id"].astype(str)
    if assignment_ids.str.strip().eq("").any() or assignment_ids.duplicated().any():
        raise DataReadinessError("catalyst assignments have invalid identities")
    statuses = assignments["status"].astype(str)
    if not statuses.isin(ASSIGNMENT_STATUSES).all():
        raise DataReadinessError("catalyst assignments contain an unsupported status")
    if not set(assignments["event_id"].astype(str)).issubset(event_ids):
        raise DataReadinessError("catalyst assignment references an unrelated event")
    if not assignments["schema_version"].astype(str).eq(ASSIGNMENT_SCHEMA_VERSION).all():
        raise DataReadinessError("catalyst assignment schema differs")
    assigned = assignments.loc[statuses.eq("assigned")]
    if not assigned.empty:
        available = _strict_utc(assigned["feature_available_at_utc"], "assigned event availability")
        decision = _strict_utc(assigned["decision_time_utc"], "assigned decision time")
        window_seconds = pd.to_numeric(assigned["window_seconds"], errors="coerce")
        age_seconds = (decision - available).dt.total_seconds()
        if (
            assigned["decision_id"].fillna("").astype(str).str.strip().eq("").any()
            or assigned["window_name"].fillna("").astype(str).str.strip().eq("").any()
            or window_seconds.isna().any()
            or window_seconds.le(0).any()
            or age_seconds.lt(0).any()
            or age_seconds.gt(window_seconds).any()
        ):
            raise DataReadinessError("catalyst assignment violates its causal window")
    if reconciliation_sha256(assignments) != expected_material_sha256:
        raise DataReadinessError("catalyst assignment material hash does not reconcile")
    return {
        str(status): int(count)
        for status, count in statuses.value_counts().sort_index().items()
    }


def _strict_bool_values(values: pd.Series, label: str) -> pd.Series:
    if values.dtype == bool:
        return values
    normalized = values.astype(str).str.strip().str.lower()
    if not normalized.isin(("true", "false", "1", "0")).all():
        raise DataReadinessError(f"catalyst {label} contains a non-canonical Boolean")
    return normalized.isin(("true", "1"))


def _verified_count_mapping(value: object, name: str) -> dict[str, int]:
    raw = _verified_mapping(value, name)
    return {
        str(key): _verified_nonnegative_int(count, f"{name}.{key}")
        for key, count in raw.items()
    }


def _verify_canonical_manifest(
    manifest: Mapping[str, object],
    *,
    path: Path,
    artifact_sha256: str,
    rows: int,
    expected_inputs: Mapping[str, str],
) -> None:
    if set(manifest) != _CANONICAL_MANIFEST_KEYS:
        raise DataReadinessError(f"canonical sidecar fields do not match the contract: {path}")
    inputs = _verified_mapping(manifest.get("inputs"), "canonical inputs")
    if (
        Path(_verified_text(manifest.get("artifact_path"), "artifact_path")).resolve() != path
        or manifest.get("artifact_sha256") != artifact_sha256
        or manifest.get("rows") != rows
        or manifest.get("production_ready") is not False
        or inputs != dict(expected_inputs)
        or file_sha256(path) != artifact_sha256
    ):
        raise DataReadinessError(f"canonical artifact or sidecar identity does not verify: {path}")


def _verify_exact_lineage_inventory(root: Path, expected_files: set[str]) -> None:
    expected_with_locks = {
        *expected_files,
        *(f"{relative}.lock" for relative in expected_files if relative.endswith(".parquet")),
    }
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    actual_directories = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_dir()}
    if actual_files != expected_with_locks or actual_directories != {"events", "assignments"}:
        raise DataReadinessError("catalyst lineage bundle inventory does not match its manifest")


def _write_or_validate_request(
    path: Path,
    request: Mapping[str, object],
    request_sha256: str,
) -> None:
    payload = {**request, "request_sha256": request_sha256}
    if path.exists():
        if _json_object(path) != payload:
            raise DataReadinessError(f"catalyst lineage resume request mismatch: {path}")
        return
    _atomic_json(path, payload)


def _progress(
    callback: Callable[[dict[str, object]], None] | None,
    **payload: object,
) -> None:
    if callback is not None:
        callback(payload)


def _json_object(path: Path) -> dict[str, object]:
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
        loaded = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise DataReadinessError(f"JSON artifact is invalid: {path}") from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"JSON artifact must contain an object: {path}")
    return {str(key): value for key, value in loaded.items()}


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
