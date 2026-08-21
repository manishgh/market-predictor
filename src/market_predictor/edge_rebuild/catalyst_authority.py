"""Immutable decision-level catalyst evidence aggregated across lineage generations."""
from __future__ import annotations



import gc
import hashlib
import json
import os
import shutil
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from uuid import uuid4

import numpy as np
import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.reconciliation import (
    ASSIGNMENT_COLUMNS,
    aggregate_event_assignments,
    event_feature_columns,
    reconciliation_sha256,
)
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.resources import assert_memory_budget, memory_audit, release_process_memory
from market_predictor.swing.contracts import MINIMUM_SWING_DECISION_DATE
from market_predictor.core.errors import DataReadinessError

LINEAGE_MANIFEST_SCHEMA: Final = "swing.catalyst_lineage_manifest.v2"
DECISION_AUTHORITY_SCHEMA: Final = "edge_rebuild.catalyst_decision_authority.v5"
DECISION_MANIFEST_SCHEMA: Final = "edge_rebuild.catalyst_decision_manifest.v5"
DECISION_ARTIFACT_TYPE: Final = "edge_rebuild_catalyst_decisions"
COVERAGE_ARTIFACT_TYPE: Final = "edge_rebuild_catalyst_coverage"
WINDOWS: Final[Mapping[str, pd.Timedelta]] = {
    "1d": pd.Timedelta(days=1),
    "3d": pd.Timedelta(days=3),
}
TRACKED_SOURCE_FAMILIES: Final = (
    "alpaca",
    "sec",
    "finviz",
)
REQUIRED_MODEL_SOURCE_FAMILIES: Final = ("alpaca",)
# Retained as the public feature-inventory name used by existing callers.
RANKING_SOURCE_FAMILIES: Final = TRACKED_SOURCE_FAMILIES
COVERAGE_FLAG_COLUMNS: Final = tuple(f"source_coverage_known_{family}_{window}" for family in RANKING_SOURCE_FAMILIES for window in WINDOWS)
MAXIMUM_PROCESS_MEMORY_GIB: Final = 4.0
MEMORY_GUARD_HEADROOM_GIB: Final = 0.5

_EVENT_PROJECTION: Final = (
    "event_id",
    "source_event_id",
    "source_security_id",
    "security_id",
    "ticker",
    "source_family",
    "feature_available_at_utc",
    "sentiment_input_sha256",
    "relation_channel",
    "training_eligible",
    "sentiment_model",
    "sentiment_model_revision",
)
_COVERAGE_COLUMNS: Final = (
    "collection_id",
    "chunk_id",
    "security_id",
    "ticker",
    "source_family",
    "requested_start_utc",
    "requested_end_utc",
    "completed_at_utc",
    "status",
    "row_count",
    "coverage_state",
    "missingness_known",
    "zero_event_semantics",
    "training_eligible",
    "schema_version",
)
_KNOWN_COVERAGE_STATES: Final = frozenset({"observed_complete", "observed_empty"})
_ZERO_SEMANTICS: Final = {
    "observed_complete": "observed_history",
    "observed_empty": "known_zero_events",
    "coverage_blindspot": "unknown_excluded",
    "failed_or_unobserved": "unknown_failed",
}


@dataclass(frozen=True, slots=True)
class CatalystDecisionAuthority:
    directory: Path
    decisions: pd.DataFrame
    coverage: pd.DataFrame
    manifest: Mapping[str, object]
    authority: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _VerifiedLineage:
    directory: Path
    manifest: Mapping[str, object]
    manifest_sha256: str
    request_sha256: str
    lineage_sha256: str
    coverage: pd.DataFrame


def _validate_decision_window(decisions: pd.DataFrame) -> None:
    if "decision_time_utc" not in decisions:
        raise DataReadinessError(
            "catalyst decision authority is missing decision_time_utc"
        )
    decision_dates = pd.to_datetime(
        decisions["decision_time_utc"],
        utc=True,
        errors="coerce",
    ).dt.date
    if decision_dates.isna().any() or bool(
        decision_dates.lt(MINIMUM_SWING_DECISION_DATE).any()
    ):
        raise DataReadinessError(
            "catalyst authority contains a missing or pre-"
            f"{MINIMUM_SWING_DECISION_DATE.isoformat()} decision"
        )


def attach_catalyst_decision_features(
    decisions: pd.DataFrame,
    authority: CatalystDecisionAuthority | Path,
) -> pd.DataFrame:
    """Attach verified catalyst features without treating unknown coverage as zero."""

    loaded = load_catalyst_decision_authority(authority) if isinstance(authority, Path) else authority
    required = {"decision_id", "security_id", "ticker", "decision_time_utc"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        raise DataReadinessError("catalyst attachment decisions missing columns: " + ", ".join(missing))
    output = decisions.copy()
    output["decision_id"] = output["decision_id"].astype(str)
    if bool(output["decision_id"].eq("").any() or output["decision_id"].duplicated().any()):
        raise DataReadinessError("catalyst attachment requires unique non-empty decision_id values")
    decision_time = pd.to_datetime(output["decision_time_utc"], utc=True, errors="coerce")
    if bool(decision_time.isna().any()):
        raise DataReadinessError("catalyst attachment decisions contain invalid decision_time_utc")
    _validate_decision_window(output)
    feature_columns = event_feature_columns(
        WINDOWS,
        source_families=RANKING_SOURCE_FAMILIES,
    )
    reserved = {
        *feature_columns,
        *COVERAGE_FLAG_COLUMNS,
        "evidence_lineage_count",
        "evidence_lineage_sha256s",
        "catalyst_source_complete_1d",
        "catalyst_source_complete_3d",
    }
    collisions = sorted(reserved.intersection(output.columns))
    if collisions:
        raise DataReadinessError("catalyst attachment would overwrite columns: " + ", ".join(collisions))
    evidence = loaded.decisions.copy()
    evidence_ids = set(evidence["decision_id"].astype(str))
    matched = output["decision_id"].isin(evidence_ids)
    if bool(matched.any()):
        indexed = evidence.set_index("decision_id")
        matched_rows = output.loc[matched]
        for identity in ("security_id", "ticker"):
            expected = matched_rows["decision_id"].map(indexed[identity]).astype(str)
            actual = matched_rows[identity].astype(str)
            if identity == "ticker":
                expected = expected.str.upper()
                actual = actual.str.upper()
            if bool(actual.ne(expected).any()):
                raise DataReadinessError(f"catalyst authority {identity} conflicts with canonical decision_id")
        expected_time = pd.to_datetime(
            matched_rows["decision_id"].map(indexed["decision_time_utc"]),
            utc=True,
            errors="coerce",
        )
        actual_time = decision_time.loc[matched]
        expected_time.index = actual_time.index
        if bool(expected_time.ne(actual_time).any()):
            raise DataReadinessError("catalyst authority decision_time_utc conflicts with canonical decision_id")
    join_columns = [
        "decision_id",
        "evidence_lineage_count",
        "evidence_lineage_sha256s",
        *feature_columns,
    ]
    output = output.merge(
        evidence.loc[:, join_columns],
        on="decision_id",
        how="left",
        validate="one_to_one",
        sort=False,
    )
    decision_time = pd.to_datetime(output["decision_time_utc"], utc=True, errors="raise")
    latest = pd.to_datetime(
        output["latest_event_feature_available_at_utc"],
        utc=True,
        errors="coerce",
    )
    if bool((latest > decision_time).fillna(False).any()):
        raise DataReadinessError("catalyst feature availability is after decision_time_utc")
    output = _apply_coverage_semantics(
        output,
        loaded.coverage,
        require_completion_by_decision=bool(loaded.manifest.get("production_ready")),
    )
    output["latest_event_feature_available_at_utc"] = latest
    return output


def publish_catalyst_decision_authority(
    lineage_directories: Sequence[Path],
    output_directory: Path,
    *,
    maximum_process_memory_gib: float = MAXIMUM_PROCESS_MEMORY_GIB,
    memory_guard_headroom_gib: float = MEMORY_GUARD_HEADROOM_GIB,
    production_ready: bool = False,
    progress: Callable[[Mapping[str, object]], None] | None = None,
) -> CatalystDecisionAuthority:
    """Verify, merge, aggregate, and atomically publish catalyst lineage evidence."""

    _validate_memory_policy(maximum_process_memory_gib, memory_guard_headroom_gib)
    roots = _normalized_lineage_directories(lineage_directories)
    if output_directory.exists():
        raise DataReadinessError(f"catalyst decision authority is immutable: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = output_directory.with_name(f".{output_directory.name}.{uuid4().hex}.tmp")
    staging.mkdir()
    database_path = staging / ".assignment_merge.sqlite3"
    verified: list[_VerifiedLineage] = []
    coverage_frames: list[pd.DataFrame] = []
    source_families: set[str] = set()
    assignment_rows_read = 0
    retained_rows_read = 0
    unique_assignment_rows = 0
    scorer_identities: set[tuple[str, str]] = set()
    database: sqlite3.Connection | None = None
    try:
        with sqlite3.connect(database_path) as database:
            _initialize_database(database)
            for generation, root in enumerate(roots):
                lineage = _verify_lineage(
                    root,
                    require_production_ready=production_ready,
                )
                verified.append(lineage)
                coverage = lineage.coverage.copy()
                coverage["source_lineage_sha256"] = lineage.lineage_sha256
                coverage_frames.append(coverage)
                records = _artifact_records(lineage.manifest)
                for index, record in enumerate(records, start=1):
                    read_count, retained_count, families, identities = _merge_artifact(
                        database,
                        lineage=lineage,
                        record=record,
                    )
                    assignment_rows_read += read_count
                    retained_rows_read += retained_count
                    source_families.update(families)
                    scorer_identities.update(identities)
                    if len(scorer_identities) > 1:
                        raise DataReadinessError("catalyst generations contain mixed sentiment scorer identity")
                    _memory_guard(
                        maximum_process_memory_gib,
                        memory_guard_headroom_gib,
                        f"catalyst authority generation {generation + 1} artifact {index}",
                    )
                    if progress is not None:
                        progress(
                            {
                                "stage": "assignment_merge",
                                "generation": generation + 1,
                                "generations": len(roots),
                                "artifact": index,
                                "artifacts": len(records),
                            }
                        )
            database.commit()
            source_families.update(RANKING_SOURCE_FAMILIES)
            decisions = _aggregate_database(database, source_families=tuple(sorted(source_families)))
            unique_assignment_rows = int(database.execute("SELECT COUNT(*) FROM assignments").fetchone()[0])
        database.close()
        database_path.unlink(missing_ok=True)
        coverage = _merge_coverage(coverage_frames)
        decisions = _apply_coverage_semantics(
            decisions,
            coverage,
            require_completion_by_decision=production_ready,
        )
        _validate_decision_window(decisions)
        scorer_identity = _scorer_identity_record(scorer_identities)
        request = _request_payload(
            verified,
            scorer_identity=scorer_identity,
            production_ready=production_ready,
        )
        request_sha256 = _json_sha256(request)
        lineage_set_sha256 = _json_sha256(request["source_lineages"])
        scorer_identity_sha256 = _json_sha256(scorer_identity)
        decision_manifest = write_canonical_artifact(
            decisions,
            staging / "decision_catalysts.parquet",
            artifact_type=DECISION_ARTIFACT_TYPE,
            audit=_decision_audit(decisions),
            inputs={
                "request_sha256": request_sha256,
                "source_lineage_set_sha256": lineage_set_sha256,
                "sentiment_scorer_identity_sha256": scorer_identity_sha256,
            },
            production_ready=production_ready,
        )
        coverage_manifest = write_canonical_artifact(
            coverage,
            staging / "source_coverage.parquet",
            artifact_type=COVERAGE_ARTIFACT_TYPE,
            audit=_coverage_audit(coverage),
            inputs={
                "request_sha256": request_sha256,
                "source_lineage_set_sha256": lineage_set_sha256,
                "sentiment_scorer_identity_sha256": scorer_identity_sha256,
            },
            production_ready=production_ready,
        )
        (staging / "decision_catalysts.parquet.lock").unlink(missing_ok=True)
        (staging / "source_coverage.parquet.lock").unlink(missing_ok=True)
        _rewrite_artifact_path(staging / "decision_catalysts.parquet", output_directory)
        _rewrite_artifact_path(staging / "source_coverage.parquet", output_directory)
        manifest: dict[str, object] = {
            "schema": DECISION_MANIFEST_SCHEMA,
            "state": "complete",
            "request": request,
            "request_sha256": request_sha256,
            "source_lineage_set_sha256": lineage_set_sha256,
            "sentiment_scorer_identity": scorer_identity,
            "sentiment_scorer_identity_sha256": scorer_identity_sha256,
            "windows": {name: int(value.total_seconds()) for name, value in WINDOWS.items()},
            "source_families": sorted(source_families),
            "tracked_source_families": list(TRACKED_SOURCE_FAMILIES),
            "required_model_source_families": list(REQUIRED_MODEL_SOURCE_FAMILIES),
            "minimum_decision_date": MINIMUM_SWING_DECISION_DATE.isoformat(),
            "assignment_rows_read": assignment_rows_read,
            "eligible_assignment_rows_read": retained_rows_read,
            "unique_assignment_rows": unique_assignment_rows,
            "duplicate_assignment_rows_merged": retained_rows_read - unique_assignment_rows,
            "decision_rows": len(decisions),
            "coverage_rows": len(coverage),
            "artifacts": {
                "decisions": _artifact_record(staging / "decision_catalysts.parquet", decision_manifest),
                "coverage": _artifact_record(staging / "source_coverage.parquet", coverage_manifest),
            },
            "memory": memory_audit(
                hard_budget_gib=maximum_process_memory_gib,
                headroom_gib=memory_guard_headroom_gib,
            ).to_record(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "production_ready": production_ready,
            "missing_value_policy": "absent decision evidence is unknown unless source coverage independently proves the lookback observed",
        }
        _atomic_json(staging / "_manifest.json", manifest)
        authority = {
            "schema": DECISION_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(staging / "_manifest.json"),
            "request_sha256": request_sha256,
            "source_lineage_set_sha256": lineage_set_sha256,
            "decision_artifact_sha256": decision_manifest["artifact_sha256"],
            "coverage_artifact_sha256": coverage_manifest["artifact_sha256"],
            "sentiment_scorer_identity_sha256": scorer_identity_sha256,
            "production_ready": production_ready,
            "minimum_decision_date": MINIMUM_SWING_DECISION_DATE.isoformat(),
        }
        _atomic_json(staging / "_authority.json", authority)
        load_catalyst_decision_authority(
            staging,
            require_production_ready=production_ready,
        )
        os.replace(staging, output_directory)
        return load_catalyst_decision_authority(
            output_directory,
            require_production_ready=production_ready,
        )
    except Exception:
        if database is not None:
            database.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_catalyst_decision_authority(
    directory: Path,
    *,
    require_production_ready: bool | None = None,
    expected_authority_sha256: str | None = None,
) -> CatalystDecisionAuthority:
    """Strictly verify and load an immutable catalyst decision authority."""

    manifest_path = directory / "_manifest.json"
    authority_path = directory / "_authority.json"
    manifest = _json_object(manifest_path)
    if expected_authority_sha256 is not None:
        if (
            len(expected_authority_sha256) != 64
            or file_sha256(authority_path) != expected_authority_sha256
        ):
            raise DataReadinessError(
                "catalyst decision authority does not match its expected identity"
            )
    authority = _json_object(authority_path)
    production_ready = manifest.get("production_ready")
    if (
        manifest.get("schema") != DECISION_MANIFEST_SCHEMA
        or manifest.get("state") != "complete"
        or not isinstance(production_ready, bool)
        or (require_production_ready is not None and production_ready is not require_production_ready)
    ):
        raise DataReadinessError("catalyst decision manifest does not satisfy the required authority mode")
    if (
        authority.get("schema") != DECISION_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("request_sha256") != manifest.get("request_sha256")
        or authority.get("source_lineage_set_sha256") != manifest.get("source_lineage_set_sha256")
        or authority.get("production_ready") is not production_ready
        or manifest.get("minimum_decision_date")
        != MINIMUM_SWING_DECISION_DATE.isoformat()
        or authority.get("minimum_decision_date")
        != MINIMUM_SWING_DECISION_DATE.isoformat()
    ):
        raise DataReadinessError("catalyst decision authority does not verify")
    request = manifest.get("request")
    if not isinstance(request, dict) or _json_sha256(request) != manifest.get("request_sha256"):
        raise DataReadinessError("catalyst decision request hash does not verify")
    if _json_sha256(request.get("source_lineages")) != manifest.get("source_lineage_set_sha256"):
        raise DataReadinessError("catalyst decision source lineage set does not verify")
    scorer_identity = request.get("sentiment_scorer_identity")
    scorer_identity_sha256 = _json_sha256(scorer_identity)
    if (
        manifest.get("sentiment_scorer_identity") != scorer_identity
        or manifest.get("sentiment_scorer_identity_sha256") != scorer_identity_sha256
        or authority.get("sentiment_scorer_identity_sha256") != scorer_identity_sha256
    ):
        raise DataReadinessError("catalyst sentiment scorer identity does not verify")
    expected_files = {
        "_authority.json",
        "_manifest.json",
        "decision_catalysts.parquet",
        "decision_catalysts.parquet.manifest.json",
        "source_coverage.parquet",
        "source_coverage.parquet.manifest.json",
    }
    observed_files = {path.name for path in directory.iterdir() if path.is_file()}
    if observed_files != expected_files or any(path.is_dir() for path in directory.iterdir()):
        raise DataReadinessError("catalyst decision authority inventory does not verify")
    decisions, decision_manifest = load_canonical_artifact(
        directory / "decision_catalysts.parquet",
        expected_type=DECISION_ARTIFACT_TYPE,
        allow_research=not production_ready,
    )
    coverage, coverage_manifest = load_canonical_artifact(
        directory / "source_coverage.parquet",
        expected_type=COVERAGE_ARTIFACT_TYPE,
        allow_research=not production_ready,
    )
    _validate_decision_window(decisions)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise DataReadinessError("catalyst decision artifact inventory is malformed")
    _verify_published_artifact(
        artifacts.get("decisions"),
        decision_manifest,
        directory / "decision_catalysts.parquet",
        len(decisions),
    )
    _verify_published_artifact(
        artifacts.get("coverage"),
        coverage_manifest,
        directory / "source_coverage.parquet",
        len(coverage),
    )
    for child_manifest in (decision_manifest, coverage_manifest):
        inputs = _required_mapping(child_manifest, "inputs")
        if (
            inputs.get("request_sha256") != manifest.get("request_sha256")
            or inputs.get("source_lineage_set_sha256") != manifest.get("source_lineage_set_sha256")
            or inputs.get("sentiment_scorer_identity_sha256") != scorer_identity_sha256
        ):
            raise DataReadinessError("catalyst decision child artifact input lineage does not verify")
    if (
        authority.get("decision_artifact_sha256") != decision_manifest.get("artifact_sha256")
        or authority.get("coverage_artifact_sha256") != coverage_manifest.get("artifact_sha256")
        or len(decisions) != _integer(manifest.get("decision_rows"), "decision_rows")
        or len(coverage) != _integer(manifest.get("coverage_rows"), "coverage_rows")
    ):
        raise DataReadinessError("catalyst decision authority row or artifact lineage mismatch")
    _decision_audit(decisions).raise_for_failure()
    _coverage_audit(coverage).raise_for_failure()
    if (
        expected_authority_sha256 is not None
        and file_sha256(authority_path) != expected_authority_sha256
    ):
        raise DataReadinessError(
            "catalyst decision authority changed while it was loaded"
        )
    return CatalystDecisionAuthority(
        directory=directory.resolve(),
        decisions=decisions,
        coverage=coverage,
        manifest=manifest,
        authority=authority,
    )


def _verify_lineage(
    directory: Path,
    *,
    require_production_ready: bool,
) -> _VerifiedLineage:
    manifest_path = directory / "_manifest.json"
    manifest = _json_object(manifest_path)
    if (
        manifest.get("schema") != LINEAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("failed_chunks") != {}
        or manifest.get("observed_chunks") != manifest.get("requested_chunks")
        or manifest.get("production_ready") is not require_production_ready
    ):
        raise DataReadinessError(f"catalyst lineage is incomplete: {directory}")
    request_file = _json_object(directory / "_request.json")
    declared_request_sha256 = _required_sha256(manifest, "request_sha256")
    request_sha256 = str(request_file.pop("request_sha256", ""))
    if (
        request_sha256 != declared_request_sha256
        or _json_sha256(request_file) != declared_request_sha256
        or request_file.get("production_ready") is not require_production_ready
    ):
        raise DataReadinessError(f"catalyst lineage request hash mismatch: {directory}")
    inventory_record = _required_mapping(manifest, "feature_inventory")
    inventory_path = directory / "feature_inventory.json"
    inventory = _json_object(inventory_path)
    if (
        file_sha256(inventory_path) != inventory_record.get("sha256")
        or inventory.get("request_sha256") != declared_request_sha256
        or inventory.get("training_eligible_channels") != ["direct_issuer"]
        or inventory.get("production_ready") is not require_production_ready
    ):
        raise DataReadinessError(f"catalyst feature inventory does not verify: {directory}")
    coverage_record = _required_mapping(manifest, "coverage")
    coverage_path = directory / "source_coverage.parquet"
    coverage, coverage_manifest = load_canonical_artifact(
        coverage_path,
        expected_type="catalyst_source_coverage",
        allow_research=not require_production_ready,
    )
    if "completed_at_utc" not in coverage.columns:
        coverage["completed_at_utc"] = _timestamp_text(
            manifest.get("completed_at_utc"),
            "catalyst lineage completion",
        )
    if (
        coverage_manifest.get("artifact_sha256") != coverage_record.get("sha256")
        or len(coverage) != _integer(coverage_record.get("rows"), "coverage.rows")
        or _required_mapping(coverage_manifest, "inputs").get("catalyst_lineage_request_sha256") != declared_request_sha256
    ):
        raise DataReadinessError(f"catalyst source coverage lineage mismatch: {directory}")
    _validate_source_coverage(coverage)
    records = _artifact_records(manifest)
    if (
        sum(_integer(record.get("event_rows"), "event_rows") for record in records)
        != _integer(manifest.get("relation_rows"), "relation_rows")
        or sum(_integer(record.get("training_eligible_rows"), "training_eligible_rows") for record in records)
        != _integer(manifest.get("training_eligible_rows"), "training_eligible_rows")
        or sum(_integer(record.get("assignment_rows"), "assignment_rows") for record in records)
        != _integer(manifest.get("assignment_rows"), "assignment_rows")
    ):
        raise DataReadinessError(f"catalyst lineage manifest row totals do not reconcile: {directory}")
    lineage_material = {
        "request": request_file,
        "coverage_sha256": coverage_manifest["artifact_sha256"],
        "artifacts": sorted(records, key=lambda item: str(item["chunk_id"])),
        "feature_inventory": inventory,
    }
    lineage_sha256 = _required_sha256(manifest, "lineage_sha256")
    if _json_sha256(lineage_material) != lineage_sha256:
        raise DataReadinessError(f"catalyst lineage material hash mismatch: {directory}")
    return _VerifiedLineage(
        directory=directory.resolve(),
        manifest=manifest,
        manifest_sha256=file_sha256(manifest_path),
        request_sha256=declared_request_sha256,
        lineage_sha256=lineage_sha256,
        coverage=coverage,
    )


def _merge_artifact(
    database: sqlite3.Connection,
    *,
    lineage: _VerifiedLineage,
    record: Mapping[str, object],
) -> tuple[int, int, set[str], set[tuple[str, str]]]:
    chunk_id = _required_text(record, "chunk_id")
    event_path = _child_path(lineage.directory, "events", chunk_id)
    assignment_path = _child_path(lineage.directory, "assignments", chunk_id)
    if Path(_required_text(record, "event_path")).resolve() != event_path:
        raise DataReadinessError(f"catalyst event path mismatch: {chunk_id}")
    if Path(_required_text(record, "assignment_path")).resolve() != assignment_path:
        raise DataReadinessError(f"catalyst assignment path mismatch: {chunk_id}")
    events, event_manifest = load_canonical_artifact(
        event_path,
        expected_type="catalyst_events",
        allow_research=not bool(lineage.manifest.get("production_ready")),
        columns=_EVENT_PROJECTION,
    )
    assignments, assignment_manifest = load_canonical_artifact(
        assignment_path,
        expected_type="catalyst_event_assignments",
        allow_research=not bool(lineage.manifest.get("production_ready")),
        columns=ASSIGNMENT_COLUMNS,
    )
    if (
        event_manifest.get("artifact_sha256") != record.get("event_sha256")
        or len(events) != _integer(record.get("event_rows"), "event_rows")
        or assignment_manifest.get("artifact_sha256") != record.get("assignment_sha256")
        or len(assignments) != _integer(record.get("assignment_rows"), "assignment_rows")
        or reconciliation_sha256(assignments) != record.get("assignment_material_sha256")
    ):
        raise DataReadinessError(f"catalyst child artifact hash or row mismatch: {chunk_id}")
    event_inputs = _required_mapping(event_manifest, "inputs")
    assignment_inputs = _required_mapping(assignment_manifest, "inputs")
    if (
        event_inputs.get("catalyst_lineage_request_sha256") != lineage.request_sha256
        or assignment_inputs.get("catalyst_lineage_request_sha256") != lineage.request_sha256
        or assignment_inputs.get("catalyst_events_sha256") != event_manifest.get("artifact_sha256")
        or assignment_inputs.get("assignment_sha256") != record.get("assignment_material_sha256")
    ):
        raise DataReadinessError(f"catalyst child artifact lineage mismatch: {chunk_id}")
    direct = events.loc[events["training_eligible"].fillna(False).astype(bool) & events["relation_channel"].astype(str).eq("direct_issuer")]
    if len(direct) != _integer(record.get("training_eligible_rows"), "training_eligible_rows"):
        raise DataReadinessError(f"catalyst eligible event row mismatch: {chunk_id}")
    if bool(direct["event_id"].astype(str).duplicated().any()):
        raise DataReadinessError(f"duplicate eligible event identity: {chunk_id}")
    scorer_identities = _event_scorer_identities(direct, chunk_id=chunk_id)
    eligible_ids = set(direct["event_id"].astype(str))
    retained = assignments.loc[
        assignments["status"].astype(str).eq("assigned")
        & assignments["event_id"].astype(str).isin(eligible_ids)
        & assignments["window_name"].astype(str).isin(WINDOWS)
        & assignments["source_family"].fillna("").astype(str).str.lower().str.strip().isin(REQUIRED_MODEL_SOURCE_FAMILIES)
    ].copy()
    if not retained.empty:
        retained = _attach_verified_event_identity(retained, direct, chunk_id=chunk_id)
        _insert_assignments(database, retained, lineage.lineage_sha256)
    families = set(retained["source_family"].fillna("").astype(str).str.lower().str.strip())
    families.discard("")
    rows = len(assignments)
    kept = len(retained)
    del events, assignments, direct, retained
    gc.collect()
    release_process_memory()
    return rows, kept, families, scorer_identities


def _initialize_database(database: sqlite3.Connection) -> None:
    database.execute("PRAGMA journal_mode=OFF")
    database.execute("PRAGMA synchronous=OFF")
    database.execute(
        """
        CREATE TABLE assignments (
            evidence_id TEXT PRIMARY KEY,
            payload_sha256 TEXT NOT NULL,
            event_id TEXT NOT NULL,
            ticker TEXT NOT NULL,
            security_id TEXT NOT NULL,
            source_family TEXT NOT NULL,
            feature_available_at_utc TEXT NOT NULL,
            decision_id TEXT NOT NULL,
            decision_time_utc TEXT NOT NULL,
            window_name TEXT NOT NULL,
            window_seconds INTEGER NOT NULL,
            status TEXT NOT NULL,
            sentiment_numeric REAL,
            relevance REAL,
            source_event_id TEXT NOT NULL,
            source_security_id TEXT NOT NULL,
            content_identity_sha256 TEXT NOT NULL,
            source_lineages TEXT NOT NULL
        )
        """
    )
    database.execute("CREATE INDEX assignments_decision_idx ON assignments(decision_id)")


def _insert_assignments(database: sqlite3.Connection, frame: pd.DataFrame, lineage_sha256: str) -> None:
    for record in frame.to_dict(orient="records"):
        normalized = _normalized_assignment(record)
        evidence_id = _json_sha256(
            {
                "event_id": normalized["event_id"],
                "decision_id": normalized["decision_id"],
                "window_name": normalized["window_name"],
            }
        )
        payload_sha256 = _json_sha256(normalized)
        existing = database.execute(
            "SELECT payload_sha256, source_lineages FROM assignments WHERE evidence_id = ?",
            (evidence_id,),
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != payload_sha256:
                raise DataReadinessError("conflicting duplicate catalyst assignment evidence")
            lineages = sorted({*json.loads(str(existing[1])), lineage_sha256})
            database.execute(
                "UPDATE assignments SET source_lineages = ? WHERE evidence_id = ?",
                (_compact_json(lineages), evidence_id),
            )
            continue
        database.execute(
            """
            INSERT INTO assignments VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                evidence_id,
                payload_sha256,
                normalized["event_id"],
                normalized["ticker"],
                normalized["security_id"],
                normalized["source_family"],
                normalized["feature_available_at_utc"],
                normalized["decision_id"],
                normalized["decision_time_utc"],
                normalized["window_name"],
                normalized["window_seconds"],
                normalized["status"],
                normalized["sentiment_numeric"],
                normalized["relevance"],
                normalized["source_event_id"],
                normalized["source_security_id"],
                normalized["content_identity_sha256"],
                _compact_json([lineage_sha256]),
            ),
        )


def _aggregate_database(database: sqlite3.Connection, *, source_families: tuple[str, ...]) -> pd.DataFrame:
    columns = (
        "event_id",
        "ticker",
        "security_id",
        "source_family",
        "feature_available_at_utc",
        "decision_id",
        "decision_time_utc",
        "window_name",
        "window_seconds",
        "status",
        "sentiment_numeric",
        "relevance",
        "source_event_id",
        "source_security_id",
        "content_identity_sha256",
        "source_lineages",
    )
    cursor = database.execute(f"SELECT {', '.join(columns)} FROM assignments ORDER BY decision_id, event_id, window_name")
    aggregates: list[pd.DataFrame] = []
    pending: list[tuple[object, ...]] = []
    while True:
        rows = cursor.fetchmany(100_000)
        if not rows:
            if pending:
                aggregates.append(_aggregate_rows(pending, columns, source_families))
            break
        pending.extend(rows)
        final_decision = str(pending[-1][5])
        split = len(pending)
        while split and str(pending[split - 1][5]) == final_decision:
            split -= 1
        ready, pending = pending[:split], pending[split:]
        if ready:
            aggregates.append(_aggregate_rows(ready, columns, source_families))
    if not aggregates:
        return _empty_decisions(source_families)
    output = pd.concat(aggregates, ignore_index=True)
    aggregates.clear()
    return output.sort_values("decision_id", kind="stable").reset_index(drop=True)


def _aggregate_rows(
    rows: list[tuple[object, ...]],
    columns: tuple[str, ...],
    source_families: tuple[str, ...],
) -> pd.DataFrame:
    frame = pd.DataFrame.from_records(rows, columns=columns)
    frame["feature_available_at_utc"] = pd.to_datetime(frame["feature_available_at_utc"], utc=True)
    frame["decision_time_utc"] = pd.to_datetime(frame["decision_time_utc"], utc=True)
    identities = frame.groupby("decision_id", sort=False).agg(
        ticker=("ticker", "first"),
        ticker_count=("ticker", "nunique"),
        security_id=("security_id", "first"),
        security_count=("security_id", "nunique"),
        decision_time_utc=("decision_time_utc", "first"),
        decision_time_count=("decision_time_utc", "nunique"),
        evidence_lineage_count=("source_lineages", lambda values: len(_lineage_union(values))),
        evidence_lineage_sha256s=("source_lineages", lambda values: _compact_json(_lineage_union(values))),
    )
    if bool(identities[["ticker_count", "security_count", "decision_time_count"]].ne(1).any(axis=None)):
        raise DataReadinessError("decision identity conflicts across catalyst lineage generations")
    deduped_frame = _deduplicate_verified_events(frame)
    aggregates = aggregate_event_assignments(
        deduped_frame,
        windows=WINDOWS,
        source_families=source_families,
    ).set_index("decision_id")
    output = identities.drop(columns=["ticker_count", "security_count", "decision_time_count"]).join(
        aggregates,
        how="left",
        validate="one_to_one",
    )
    return output.reset_index()


def _deduplicate_verified_events(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate only lineage-bound event or issuer/content identities."""

    if frame.empty:
        return frame
    durable_values = (
        "security_id",
        "ticker",
        "source_family",
        "feature_available_at_utc",
        "source_event_id",
        "source_security_id",
        "content_identity_sha256",
        "sentiment_numeric",
        "relevance",
    )
    _reject_identity_conflicts(
        frame,
        keys=("event_id",),
        values=durable_values,
        description="durable catalyst event identity",
    )
    _reject_identity_conflicts(
        frame,
        keys=("source_family", "source_event_id", "security_id"),
        values=(
            "ticker",
            "feature_available_at_utc",
            "source_security_id",
            "content_identity_sha256",
            "sentiment_numeric",
            "relevance",
        ),
        description="durable catalyst source-event identity",
    )
    _reject_identity_conflicts(
        frame,
        keys=("security_id", "content_identity_sha256"),
        values=("ticker", "source_security_id", "sentiment_numeric", "relevance"),
        description="catalyst issuer/content identity",
    )

    priority = {"sec": 0, "alpaca": 1, "finviz": 2}
    output = frame.assign(
        _source_priority=frame["source_family"].map(priority).fillna(99),
    ).sort_values(
        [
            "decision_id",
            "window_name",
            "feature_available_at_utc",
            "_source_priority",
            "event_id",
        ],
        kind="stable",
    )
    output = output.drop_duplicates(
        ["decision_id", "window_name", "event_id"],
        keep="first",
    )
    output = output.drop_duplicates(
        ["decision_id", "window_name", "security_id", "source_family", "source_event_id"],
        keep="first",
    )
    output = output.drop_duplicates(
        ["decision_id", "window_name", "security_id", "content_identity_sha256"],
        keep="first",
    )
    return output.drop(columns="_source_priority").reset_index(drop=True)


def _reject_identity_conflicts(
    frame: pd.DataFrame,
    *,
    keys: tuple[str, ...],
    values: tuple[str, ...],
    description: str,
) -> None:
    grouped = frame.groupby(list(keys), sort=False, dropna=False)
    for value in values:
        if bool(grouped[value].nunique(dropna=False).gt(1).any()):
            raise DataReadinessError(f"conflicting {description}")


def _merge_coverage(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=["coverage_evidence_id", *_COVERAGE_COLUMNS, "source_lineage_sha256s"])
    combined = pd.concat(frames, ignore_index=True)
    frames.clear()
    _validate_source_coverage(combined)
    records: dict[str, dict[str, object]] = {}
    lineages: dict[str, set[str]] = {}
    for raw in combined.to_dict(orient="records"):
        normalized = _normalized_coverage(raw)
        evidence_id = _json_sha256(normalized)
        records[evidence_id] = normalized
        lineages.setdefault(evidence_id, set()).add(str(raw["source_lineage_sha256"]))
    output_records = [
        {
            "coverage_evidence_id": evidence_id,
            **records[evidence_id],
            "source_lineage_sha256s": _compact_json(sorted(lineages[evidence_id])),
        }
        for evidence_id in sorted(records)
    ]
    return pd.DataFrame.from_records(
        output_records,
        columns=["coverage_evidence_id", *_COVERAGE_COLUMNS, "source_lineage_sha256s"],
    )


def _coverage_eligibility(
    decisions: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    source_family: str,
    lookback: pd.Timedelta,
    require_completion_by_decision: bool,
) -> pd.Series:
    intervals = coverage.loc[
        coverage["source_family"].astype(str).str.lower().eq(source_family)
        & coverage["coverage_state"].astype(str).isin(_KNOWN_COVERAGE_STATES)
        & coverage["missingness_known"].fillna(False).astype(bool)
        & coverage["training_eligible"].fillna(False).astype(bool)
    ].copy()
    intervals["requested_start_utc"] = pd.to_datetime(
        intervals["requested_start_utc"],
        utc=True,
        errors="raise",
    )
    intervals["requested_end_utc"] = pd.to_datetime(
        intervals["requested_end_utc"],
        utc=True,
        errors="raise",
    )
    intervals["completed_at_utc"] = pd.to_datetime(
        intervals["completed_at_utc"],
        utc=True,
        errors="raise",
    )
    result = pd.Series(False, index=decisions.index, dtype=bool)
    decision_time = pd.to_datetime(decisions["decision_time_utc"], utc=True, errors="raise")
    grouping = pd.DataFrame(
        {
            "security_id": decisions["security_id"].astype(str),
            "ticker": decisions["ticker"].astype(str).str.upper(),
        }
    )
    interval_security_ids = intervals["security_id"].astype(str)
    interval_tickers = intervals["ticker"].astype(str).str.upper()
    for (security_id, ticker), positions in grouping.groupby(
        ["security_id", "ticker"], sort=False
    ).indices.items():
        rows = intervals.loc[
            interval_security_ids.eq(str(security_id))
            & interval_tickers.eq(str(ticker))
        ].sort_values(
            ["requested_start_utc", "requested_end_utc"], kind="stable"
        )
        if rows.empty:
            continue
        for position in np.asarray(positions, dtype=np.int64):
            cutoff = pd.Timestamp(decision_time.iloc[position])
            known_at_cutoff = rows.loc[rows["completed_at_utc"].le(cutoff)] if require_completion_by_decision else rows
            if known_at_cutoff.empty:
                continue
            merged: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            for interval in known_at_cutoff.itertuples(index=False):
                start = pd.Timestamp(interval.requested_start_utc)
                end = pd.Timestamp(interval.requested_end_utc)
                if not merged or start > merged[-1][1]:
                    merged.append((start, end))
                else:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            lookback_start = cutoff - lookback
            result.iloc[position] = any(start <= lookback_start and cutoff <= end for start, end in merged)
    return result


def _apply_coverage_semantics(
    decisions: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    require_completion_by_decision: bool,
) -> pd.DataFrame:
    output = decisions.copy()
    feature_columns = event_feature_columns(
        WINDOWS,
        source_families=RANKING_SOURCE_FAMILIES,
    )
    missing = sorted(set(feature_columns).difference(output.columns))
    if missing:
        raise DataReadinessError("catalyst aggregate is missing required ranking features: " + ", ".join(missing))
    eligibility: dict[tuple[str, str], pd.Series] = {}
    for family in RANKING_SOURCE_FAMILIES:
        for window, duration in WINDOWS.items():
            known = _coverage_eligibility(
                output,
                coverage,
                source_family=family,
                lookback=duration,
                require_completion_by_decision=require_completion_by_decision,
            )
            eligibility[(family, window)] = known
            output[f"source_coverage_known_{family}_{window}"] = known
            column = f"source_count_{family}_{window}"
            numeric = pd.to_numeric(output[column], errors="coerce")
            output[column] = numeric.where(~known | numeric.notna(), 0.0).where(known)
    for window in WINDOWS:
        all_known = pd.concat(
            [eligibility[(family, window)] for family in REQUIRED_MODEL_SOURCE_FAMILIES],
            axis=1,
        ).all(axis=1)
        output[f"catalyst_source_complete_{window}"] = all_known
        for column in feature_columns:
            if column.endswith(f"_{window}") and not column.startswith("source_count_"):
                numeric = pd.to_numeric(output[column], errors="coerce")
                output[column] = numeric.where(
                    ~all_known | numeric.notna(),
                    0.0,
                ).where(all_known)
    return output


def _validate_source_coverage(frame: pd.DataFrame) -> None:
    missing = sorted(set(_COVERAGE_COLUMNS).difference(frame.columns))
    if missing:
        raise DataReadinessError("catalyst source coverage missing columns: " + ", ".join(missing))
    starts = pd.to_datetime(frame["requested_start_utc"], utc=True, errors="coerce")
    ends = pd.to_datetime(frame["requested_end_utc"], utc=True, errors="coerce")
    completed = pd.to_datetime(
        frame["completed_at_utc"],
        utc=True,
        errors="coerce",
    )
    states = frame["coverage_state"].astype(str)
    known = frame["missingness_known"].fillna(False).astype(bool)
    eligible = frame["training_eligible"].fillna(False).astype(bool)
    expected_known = states.isin(_KNOWN_COVERAGE_STATES)
    expected_semantics = states.map(_ZERO_SEMANTICS)
    failures = (
        starts.isna()
        | ends.isna()
        | ends.le(starts)
        | completed.isna()
        | completed.lt(ends)
        | expected_semantics.isna()
        | frame["zero_event_semantics"].astype(str).ne(expected_semantics)
        | known.ne(expected_known)
        | eligible.ne(expected_known)
    )
    empty = states.eq("observed_empty")
    failures |= empty & pd.to_numeric(frame["row_count"], errors="coerce").ne(0)
    if bool(failures.any()):
        raise DataReadinessError("catalyst source coverage violates known-zero/unknown semantics")


def _decision_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    source_families = sorted(
        column.removeprefix("source_count_").removesuffix("_3d")
        for column in frame
        if column.startswith("source_count_") and column.endswith("_3d")
    )
    required = {
        "decision_id",
        "ticker",
        "security_id",
        "decision_time_utc",
        "evidence_lineage_count",
        "evidence_lineage_sha256s",
        *COVERAGE_FLAG_COLUMNS,
        "catalyst_source_complete_1d",
        "catalyst_source_complete_3d",
        *event_feature_columns(WINDOWS, source_families=source_families),
    }
    failures = len(required.difference(frame.columns))
    if failures == 0 and not frame.empty:
        decision_time = pd.to_datetime(frame["decision_time_utc"], utc=True, errors="coerce")
        latest = pd.to_datetime(frame["latest_event_feature_available_at_utc"], utc=True, errors="coerce")
        failures += int(frame["decision_id"].astype(str).duplicated().sum())
        failures += int(frame["decision_id"].astype(str).eq("").sum())
        failures += int(frame["security_id"].astype(str).eq("").sum())
        failures += int(decision_time.isna().sum() + latest.isna().sum())
        failures += int((latest > decision_time).fillna(False).sum())
        complete_3d = frame["catalyst_source_complete_3d"].fillna(False).astype(bool)
        event_count_3d = pd.to_numeric(frame["event_count_3d"], errors="coerce")
        failures += int((complete_3d & event_count_3d.le(0)).sum())
        failures += int((~complete_3d & event_count_3d.notna()).sum())
        for family in RANKING_SOURCE_FAMILIES:
            known = frame[f"source_coverage_known_{family}_3d"].fillna(False).astype(bool)
            count = pd.to_numeric(frame[f"source_count_{family}_3d"], errors="coerce")
            failures += int((known & count.isna()).sum())
            failures += int((~known & count.notna()).sum())
    return _audit_report("catalyst_decision_authority", failures, len(frame))


def _coverage_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    try:
        _validate_source_coverage(frame)
        failures = int(frame.get("coverage_evidence_id", pd.Series(dtype=str)).astype(str).duplicated().sum())
    except DataReadinessError:
        failures = max(len(frame), 1)
    return _audit_report("catalyst_coverage_authority", failures, len(frame))


def _audit_report(name: str, failures: int, rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name=name,
                status="pass" if failures == 0 else "fail",
                failures=failures,
                rows_checked=rows,
                detail="hash-bound direct-issuer evidence and source missingness semantics verify",
            ),
        )
    )


def _request_payload(
    lineages: Sequence[_VerifiedLineage],
    *,
    scorer_identity: Mapping[str, str] | None,
    production_ready: bool,
) -> dict[str, object]:
    return {
        "schema": "edge_rebuild.catalyst_decision_request.v1",
        "windows": {name: int(value.total_seconds()) for name, value in WINDOWS.items()},
        "eligibility": {
            "training_eligible": True,
            "relation_channel": "direct_issuer",
            "status": "assigned",
            "source_family": list(REQUIRED_MODEL_SOURCE_FAMILIES),
        },
        "tracked_source_families": list(TRACKED_SOURCE_FAMILIES),
        "required_model_source_families": list(REQUIRED_MODEL_SOURCE_FAMILIES),
        "sentiment_scorer_identity": scorer_identity,
        "source_lineages": [
            {
                "manifest_sha256": item.manifest_sha256,
                "request_sha256": item.request_sha256,
                "lineage_sha256": item.lineage_sha256,
            }
            for item in sorted(lineages, key=lambda value: value.lineage_sha256)
        ],
        "missing_value_policy": "missing decision evidence is not zero; source coverage must prove known zero",
        "production_ready": production_ready,
    }


def _artifact_record(path: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": manifest["artifact_sha256"],
        "manifest_sha256": file_sha256(manifest_path_for(path)),
        "rows": manifest["rows"],
    }


def _verify_published_artifact(
    raw_record: object,
    child_manifest: Mapping[str, object],
    path: Path,
    rows: int,
) -> None:
    if not isinstance(raw_record, dict):
        raise DataReadinessError("catalyst decision child artifact record is malformed")
    if (
        raw_record.get("path") != path.name
        or raw_record.get("sha256") != child_manifest.get("artifact_sha256")
        or raw_record.get("manifest_sha256") != file_sha256(manifest_path_for(path))
        or int(raw_record.get("rows", -1)) != rows
    ):
        raise DataReadinessError("catalyst decision child artifact does not verify")


def _rewrite_artifact_path(path: Path, output_directory: Path) -> None:
    manifest_path = manifest_path_for(path)
    manifest = _json_object(manifest_path)
    manifest["artifact_path"] = str((output_directory / path.name).resolve())
    _atomic_json(manifest_path, manifest)


def _artifact_records(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    raw = manifest.get("artifacts")
    if not isinstance(raw, list) or not raw:
        raise DataReadinessError("catalyst lineage has no artifact inventory")
    records: list[dict[str, object]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise DataReadinessError("catalyst lineage artifact inventory is malformed")
        record = {str(key): value for key, value in item.items()}
        chunk_id = _required_text(record, "chunk_id")
        if chunk_id in seen:
            raise DataReadinessError("catalyst lineage has duplicate chunk IDs")
        seen.add(chunk_id)
        records.append(record)
    return sorted(records, key=lambda item: str(item["chunk_id"]))


def _child_path(root: Path, child: str, chunk_id: str) -> Path:
    directory = (root / child).resolve()
    path = (directory / f"{chunk_id}.parquet").resolve()
    if path.parent != directory:
        raise DataReadinessError("catalyst child artifact path traversal")
    return path


def _normalized_assignment(record: Mapping[str, object]) -> dict[str, object]:
    available = _timestamp_text(record.get("feature_available_at_utc"), "assignment feature availability")
    decision = _timestamp_text(record.get("decision_time_utc"), "assignment decision time")
    if pd.Timestamp(available) > pd.Timestamp(decision):
        raise DataReadinessError("catalyst assignment contains post-decision evidence")
    window_name = str(record.get("window_name", ""))
    expected_seconds = int(WINDOWS[window_name].total_seconds())
    window_seconds = _integer(record.get("window_seconds"), "window_seconds")
    if window_seconds != expected_seconds:
        raise DataReadinessError("catalyst assignment window duration mismatch")
    return {
        "event_id": _value_text(record.get("event_id"), "event_id"),
        "ticker": _value_text(record.get("ticker"), "ticker").upper(),
        "security_id": _value_text(record.get("security_id"), "security_id"),
        "source_family": _value_text(record.get("source_family"), "source_family").lower(),
        "feature_available_at_utc": available,
        "decision_id": _value_text(record.get("decision_id"), "decision_id"),
        "decision_time_utc": decision,
        "window_name": window_name,
        "window_seconds": window_seconds,
        "status": "assigned",
        "sentiment_numeric": _nullable_float(record.get("sentiment_numeric")),
        "relevance": _nullable_float(record.get("relevance")),
        "source_event_id": _value_text(record.get("source_event_id"), "source_event_id"),
        "source_security_id": _value_text(record.get("source_security_id"), "source_security_id"),
        "content_identity_sha256": _sha256_text(
            record.get("content_identity_sha256"),
            "content_identity_sha256",
        ),
    }


def _attach_verified_event_identity(
    assignments: pd.DataFrame,
    direct_events: pd.DataFrame,
    *,
    chunk_id: str,
) -> pd.DataFrame:
    identity_columns = (
        "event_id",
        "source_event_id",
        "source_security_id",
        "security_id",
        "ticker",
        "source_family",
        "feature_available_at_utc",
        "sentiment_input_sha256",
    )
    identities = direct_events.loc[:, identity_columns].rename(
        columns={
            "security_id": "event_security_id",
            "ticker": "event_ticker",
            "source_family": "event_source_family",
            "feature_available_at_utc": "event_feature_available_at_utc",
            "sentiment_input_sha256": "content_identity_sha256",
        }
    )
    output = assignments.merge(
        identities,
        on="event_id",
        how="left",
        validate="many_to_one",
    )
    required_identity = (
        "source_event_id",
        "source_security_id",
        "event_security_id",
        "event_ticker",
        "event_source_family",
        "event_feature_available_at_utc",
        "content_identity_sha256",
    )
    if bool(output.loc[:, required_identity].isna().any(axis=None)):
        raise DataReadinessError(f"catalyst event identity is missing: {chunk_id}")
    event_available = pd.to_datetime(output["event_feature_available_at_utc"], utc=True)
    assignment_available = pd.to_datetime(output["feature_available_at_utc"], utc=True)
    if bool(
        output["security_id"].astype(str).ne(output["event_security_id"].astype(str)).any()
        or output["ticker"].astype(str).str.upper().ne(output["event_ticker"].astype(str).str.upper()).any()
        or output["source_family"].astype(str).str.lower().ne(output["event_source_family"].astype(str).str.lower()).any()
        or assignment_available.ne(event_available).any()
    ):
        raise DataReadinessError(f"catalyst assignment conflicts with event identity: {chunk_id}")
    return output.drop(
        columns=[
            "event_security_id",
            "event_ticker",
            "event_source_family",
            "event_feature_available_at_utc",
        ]
    )


def _normalized_coverage(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "collection_id": str(record.get("collection_id", "")),
        "chunk_id": str(record.get("chunk_id", "")),
        "security_id": str(record.get("security_id", "")),
        "ticker": str(record.get("ticker", "")).upper(),
        "source_family": str(record.get("source_family", "")).lower(),
        "requested_start_utc": _timestamp_text(record.get("requested_start_utc"), "coverage start"),
        "requested_end_utc": _timestamp_text(record.get("requested_end_utc"), "coverage end"),
        "completed_at_utc": _timestamp_text(
            record.get("completed_at_utc"),
            "coverage collection completion",
        ),
        "status": str(record.get("status", "")),
        "row_count": _integer(record.get("row_count"), "coverage.row_count"),
        "coverage_state": str(record.get("coverage_state", "")),
        "missingness_known": bool(record.get("missingness_known")),
        "zero_event_semantics": str(record.get("zero_event_semantics", "")),
        "training_eligible": bool(record.get("training_eligible")),
        "schema_version": str(record.get("schema_version", "")),
    }


def _event_scorer_identities(
    direct_events: pd.DataFrame,
    *,
    chunk_id: str,
) -> set[tuple[str, str]]:
    if direct_events.empty:
        return set()
    models = direct_events["sentiment_model"].astype(str).str.strip()
    revisions = direct_events["sentiment_model_revision"].astype(str).str.strip()
    if bool(models.eq("").any() or revisions.eq("").any()):
        raise DataReadinessError(f"catalyst scorer identity is missing for eligible events: {chunk_id}")
    identities = set(zip(models, revisions, strict=True))
    if len(identities) != 1:
        raise DataReadinessError(f"catalyst artifact contains mixed sentiment scorer identity: {chunk_id}")
    return identities


def _scorer_identity_record(
    identities: set[tuple[str, str]],
) -> dict[str, str] | None:
    if not identities:
        return None
    if len(identities) != 1:
        raise DataReadinessError("catalyst generations contain mixed sentiment scorer identity")
    model, revision = next(iter(identities))
    return {"model": model, "revision": revision}


def _normalized_lineage_directories(values: Sequence[Path]) -> tuple[Path, ...]:
    roots = tuple(sorted({Path(value).resolve() for value in values}, key=str))
    if not roots:
        raise DataReadinessError("at least one catalyst lineage directory is required")
    return roots


def _empty_decisions(source_families: tuple[str, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "decision_id",
            "ticker",
            "security_id",
            "decision_time_utc",
            "evidence_lineage_count",
            "evidence_lineage_sha256s",
            *event_feature_columns(WINDOWS, source_families=source_families),
        ]
    )


def _lineage_union(values: pd.Series) -> list[str]:
    return sorted({item for value in values for item in json.loads(str(value))})


def _nullable_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise DataReadinessError("catalyst assignment has invalid numeric evidence")
    return float(str(value))


def _timestamp_text(value: object, name: str) -> str:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise DataReadinessError(f"{name} is invalid")
    return str(pd.Timestamp(parsed).isoformat())


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DataReadinessError(f"catalyst artifact has invalid integer {name}")
    return int(str(value))


def _value_text(value: object, name: str) -> str:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        raise DataReadinessError(f"catalyst assignment has empty {name}")
    return text


def _sha256_text(value: object, name: str) -> str:
    text = _value_text(value, name).lower()
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise DataReadinessError(f"catalyst assignment has invalid {name}")
    return text


def _required_mapping(record: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = record.get(key)
    if not isinstance(value, dict):
        raise DataReadinessError(f"catalyst artifact has no {key}")
    return value


def _required_text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DataReadinessError(f"catalyst artifact has no {key}")
    return value


def _required_sha256(record: Mapping[str, object], key: str) -> str:
    value = _required_text(record, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise DataReadinessError(f"catalyst artifact has invalid {key}")
    return value.lower()


def _validate_memory_policy(hard_budget_gib: float, headroom_gib: float) -> None:
    if hard_budget_gib > MAXIMUM_PROCESS_MEMORY_GIB:
        raise DataReadinessError("catalyst authority hard memory budget cannot exceed 4 GiB")
    if hard_budget_gib <= 0 or headroom_gib <= 0 or headroom_gib >= hard_budget_gib:
        raise DataReadinessError("catalyst authority memory policy is invalid")


def _memory_guard(hard_budget_gib: float, headroom_gib: float, stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=hard_budget_gib,
        headroom_gib=headroom_gib,
        stage=stage,
    )


def _json_object(path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"JSON artifact is unreadable: {path}") from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"JSON artifact must contain an object: {path}")
    return {str(key): value for key, value in loaded.items()}


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_compact_json(value).encode("utf-8")).hexdigest()


def _compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
