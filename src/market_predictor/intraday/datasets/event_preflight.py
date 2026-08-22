"""Causal eligibility authority for A5 intraday event specialists."""
from __future__ import annotations



import hashlib
import json
import os
import shutil
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final
from uuid import uuid4

import numpy as np
import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    write_canonical_artifact,
)
from market_predictor.intraday.datasets.history import json_sha256
from market_predictor.intraday.training.training import (
    PublishedIntradayDataset,
    load_published_intraday_dataset,
)
from market_predictor.edge_rebuild.issuer_event_family_authority import (
    AUTHORITY_SCHEMA as EVENT_AUTHORITY_SCHEMA,
)
from market_predictor.edge_rebuild.issuer_event_family_authority import (
    FAMILY_ASSIGNMENTS_ARTIFACT_TYPE,
    FAMILY_COVERAGE_ARTIFACT_TYPE,
    FAMILY_EVENTS_ARTIFACT_TYPE,
)
from market_predictor.edge_rebuild.issuer_event_family_authority import (
    MANIFEST_SCHEMA as EVENT_MANIFEST_SCHEMA,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.swing.event_families import EVENT_FAMILY_POLICY_SHA256
from market_predictor.core.errors import DataReadinessError

POLICY_SCHEMA: Final = "edge_rebuild.intraday_event_preflight_policy.v1"
MANIFEST_SCHEMA: Final = "edge_rebuild.intraday_event_preflight_manifest.v1"
AUTHORITY_SCHEMA: Final = "edge_rebuild.intraday_event_preflight_authority.v1"
DECISION_ARTIFACT_TYPE: Final = "intraday_event_preflight_decisions"
ATTACHMENT_ARTIFACT_TYPE: Final = "intraday_event_preflight_attachments"
COVERAGE_ARTIFACT_TYPE: Final = "intraday_event_preflight_coverage_audit"
_METADATA_FILES: Final = frozenset({"_request.json", "_manifest.json", "_authority.json"})
_ARTIFACTS: Final = {
    "decisions": ("decision_eligibility.parquet", DECISION_ARTIFACT_TYPE),
    "attachments": ("event_attachments.parquet", ATTACHMENT_ARTIFACT_TYPE),
    "coverage_audit": ("coverage_audit.parquet", COVERAGE_ARTIFACT_TYPE),
}
_FROZEN_POLICY: Final = {
    "source_family": "alpaca",
    "relation_channel": "direct_issuer",
    "event_family": "analyst_revision",
    "lookback_hours": 24,
    "security_holdout_fraction": 0.20,
    "validation_folds": 4,
    "minimum_unique_event_episodes": 1000,
    "minimum_securities": 200,
    "minimum_fit_sessions": 120,
    "minimum_scope_rows": 1000,
    "minimum_scope_securities": 20,
    "maximum_process_memory_gib": 4.0,
    "memory_guard_headroom_gib": 0.75,
}
_IDENTITY_ALIGNMENT_POLICY: Final = "exact_uppercase_ticker_with_cik_conflict_rejection_v1"


@dataclass(frozen=True, slots=True)
class IntradayEventPreflightConfig:
    source_family: str
    relation_channel: str
    event_family: str
    lookback_hours: int
    security_holdout_fraction: float
    validation_folds: int
    minimum_unique_event_episodes: int
    minimum_securities: int
    minimum_fit_sessions: int
    minimum_scope_rows: int
    minimum_scope_securities: int
    maximum_process_memory_gib: float
    memory_guard_headroom_gib: float


@dataclass(frozen=True, slots=True)
class IntradayEventPreflightAuthority:
    directory: Path
    decisions: pd.DataFrame
    attachments: pd.DataFrame
    coverage_audit: pd.DataFrame
    manifest: Mapping[str, Any]
    authority: Mapping[str, Any]
    verified_parent_events: pd.DataFrame | None = None


@dataclass(frozen=True, slots=True)
class _EventAuthoritySlice:
    events: pd.DataFrame
    assignments: pd.DataFrame
    coverage: pd.DataFrame
    projected_inventory_sha256: str


def load_intraday_event_preflight_config(path: Path) -> IntradayEventPreflightConfig:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != POLICY_SCHEMA:
        raise DataReadinessError(f"unsupported intraday event preflight policy: {path}")
    if raw.get("unknown_coverage_policy") != "abstain":
        raise DataReadinessError("unknown event coverage must cause abstention")
    if raw.get("historical_proxy_policy") != "research_only":
        raise DataReadinessError("historical proxy evidence must remain research-only")
    config = IntradayEventPreflightConfig(
        source_family=_required_text(raw, "source_family"),
        relation_channel=_required_text(raw, "relation_channel"),
        event_family=_required_text(raw, "event_family"),
        lookback_hours=int(raw.get("lookback_hours", 0)),
        security_holdout_fraction=float(raw.get("security_holdout_fraction", 0.0)),
        validation_folds=int(raw.get("validation_folds", 0)),
        minimum_unique_event_episodes=int(raw.get("minimum_unique_event_episodes", 0)),
        minimum_securities=int(raw.get("minimum_securities", 0)),
        minimum_fit_sessions=int(raw.get("minimum_fit_sessions", 0)),
        minimum_scope_rows=int(raw.get("minimum_scope_rows", 0)),
        minimum_scope_securities=int(raw.get("minimum_scope_securities", 0)),
        maximum_process_memory_gib=float(raw.get("maximum_process_memory_gib", 0.0)),
        memory_guard_headroom_gib=float(raw.get("memory_guard_headroom_gib", 0.0)),
    )
    _validate_config(config)
    return config


def load_issuer_event_family_authority(
    directory: Path,
    *,
    expected_authority_sha256: str | None = None,
) -> _EventAuthoritySlice:
    """Strictly load only the parent tables consumed by A5.1.

    The parent contains thousands of unclassified research shards and a swing cohort
    audit. Their identities remain bound by the parent manifest, but A5.1 neither
    reads nor semantically rebuilds them.
    """

    manifest_path = directory / "_manifest.json"
    authority_path = directory / "_authority.json"
    manifest = _read_json(manifest_path)
    authority = _read_json(authority_path)
    authority_sha256 = file_sha256(authority_path)
    if expected_authority_sha256 is not None and authority_sha256 != expected_authority_sha256:
        raise DataReadinessError("A5.1 parent event authority identity differs")
    request = manifest.get("request")
    if (
        manifest.get("schema") != EVENT_MANIFEST_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("production_ready") is not False
        or authority.get("schema") != EVENT_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("production_ready") is not False
        or manifest.get("event_family_policy_sha256") != EVENT_FAMILY_POLICY_SHA256
        or authority.get("event_family_policy_sha256") != EVENT_FAMILY_POLICY_SHA256
        or not isinstance(request, dict)
        or json_sha256(request) != manifest.get("request_sha256")
        or authority.get("request_sha256") != manifest.get("request_sha256")
    ):
        raise DataReadinessError("A5.1 parent event authority does not verify")
    records = manifest.get("artifacts")
    if not isinstance(records, dict):
        raise DataReadinessError("A5.1 parent event artifact inventory is malformed")
    specifications = {
        "events": ("family_events.parquet", FAMILY_EVENTS_ARTIFACT_TYPE),
        "assignments": ("family_assignments.parquet", FAMILY_ASSIGNMENTS_ARTIFACT_TYPE),
        "coverage": ("family_coverage.parquet", FAMILY_COVERAGE_ARTIFACT_TYPE),
    }
    expected_inventory = {
        "_authority.json",
        "_manifest.json",
        "family_events.parquet",
        "family_events.parquet.manifest.json",
        "family_assignments.parquet",
        "family_assignments.parquet.manifest.json",
        "family_coverage.parquet",
        "family_coverage.parquet.manifest.json",
        "cohort_audit.parquet",
        "cohort_audit.parquet.manifest.json",
    }
    _verify_unclassified_parent_inventory(
        directory,
        manifest,
        expected_inventory=expected_inventory,
    )
    observed_inventory = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if observed_inventory != expected_inventory:
        raise DataReadinessError("A5.1 parent event recursive inventory differs")
    projected_inventory_sha256 = json_sha256(
        [
            {"path": relative, "sha256": file_sha256(directory / relative)}
            for relative in sorted(observed_inventory)
        ]
    )
    frames: dict[str, pd.DataFrame] = {}
    for name, (filename, artifact_type) in specifications.items():
        frame, child = load_canonical_artifact(
            directory / filename,
            expected_type=artifact_type,
            allow_research=True,
        )
        record = records.get(name)
        child_inputs = child.get("inputs")
        if (
            not isinstance(record, dict)
            or record.get("path") != filename
            or record.get("rows") != len(frame)
            or record.get("sha256") != child.get("artifact_sha256")
            or not isinstance(child_inputs, dict)
            or child_inputs.get("request_sha256") != manifest.get("request_sha256")
        ):
            raise DataReadinessError(f"A5.1 parent event {name} lineage differs")
        frames[name] = frame
    cohort, cohort_child = load_canonical_artifact(
        directory / "cohort_audit.parquet",
        expected_type="issuer_event_family_cohort_audit",
        allow_research=True,
    )
    cohort_record = records.get("cohort_audit")
    if (
        not isinstance(cohort_record, dict)
        or cohort_record.get("path") != "cohort_audit.parquet"
        or cohort_record.get("rows") != len(cohort)
        or cohort_record.get("sha256") != cohort_child.get("artifact_sha256")
    ):
        raise DataReadinessError("A5.1 parent event cohort-audit lineage differs")
    return _EventAuthoritySlice(
        events=frames["events"],
        assignments=frames["assignments"],
        coverage=frames["coverage"],
        projected_inventory_sha256=projected_inventory_sha256,
    )


def _verify_unclassified_parent_inventory(
    directory: Path,
    manifest: Mapping[str, Any],
    *,
    expected_inventory: set[str],
) -> None:
    raw_records = manifest.get("unclassified_artifacts", [])
    if not isinstance(raw_records, list):
        raise DataReadinessError("A5.1 parent unclassified inventory is malformed")
    request_sha256 = str(manifest.get("request_sha256", ""))
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise DataReadinessError("A5.1 parent unclassified record is malformed")
        relative = str(raw.get("path", ""))
        if not relative or Path(relative).is_absolute():
            raise DataReadinessError("A5.1 parent unclassified path is invalid")
        artifact = (directory / relative).resolve()
        if directory.resolve() not in artifact.parents:
            raise DataReadinessError("A5.1 parent unclassified path escapes authority")
        child_manifest_path = artifact.with_suffix(artifact.suffix + ".manifest.json")
        child = _read_json(child_manifest_path)
        child_inputs = child.get("inputs")
        if (
            file_sha256(artifact) != raw.get("sha256")
            or child.get("artifact_sha256") != raw.get("sha256")
            or child.get("rows") != raw.get("rows")
            or child.get("artifact_type") != "issuer_event_family_unclassified_events"
            or not isinstance(child_inputs, dict)
            or child_inputs.get("request_sha256") != request_sha256
        ):
            raise DataReadinessError("A5.1 parent unclassified artifact lineage differs")
        expected_inventory.add(artifact.relative_to(directory).as_posix())
        expected_inventory.add(child_manifest_path.relative_to(directory).as_posix())


def publish_intraday_event_preflight(
    *,
    dataset_authority_directory: Path,
    event_authority_directories: Sequence[Path],
    output_directory: Path,
    config: IntradayEventPreflightConfig,
    policy_path: Path,
) -> IntradayEventPreflightAuthority:
    """Publish a fail-closed A5.1 capacity and causality decision."""

    if len(event_authority_directories) < 1:
        raise DataReadinessError("A5.1 requires at least one event authority")
    _validate_config(config)
    inputs = (dataset_authority_directory, *event_authority_directories, policy_path)
    _require_path_isolation(output_directory, inputs)
    policy_sha256 = file_sha256(policy_path)
    dataset_authority_sha256 = file_sha256(dataset_authority_directory / "_authority.json")
    event_parents = sorted(
        (
            path.resolve(),
            file_sha256(path / "_authority.json"),
        )
        for path in event_authority_directories
    )
    event_authorities = [
        load_issuer_event_family_authority(
            path,
            expected_authority_sha256=authority_sha256,
        )
        for path, authority_sha256 in event_parents
    ]
    event_identities = [
        {
            "directory": str(path),
            "authority_sha256": authority_sha256,
            "projected_inventory_sha256": authority.projected_inventory_sha256,
        }
        for (path, authority_sha256), authority in zip(
            event_parents, event_authorities, strict=True
        )
    ]
    request_payload = {
        "schema": MANIFEST_SCHEMA,
        "identity_alignment_policy": _IDENTITY_ALIGNMENT_POLICY,
        "dataset_authority_directory": str(dataset_authority_directory.resolve()),
        "dataset_authority_sha256": dataset_authority_sha256,
        "event_authorities": event_identities,
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": policy_sha256,
        "policy": _config_record(config),
    }
    request = {**request_payload, "request_sha256": json_sha256(request_payload)}
    if output_directory.exists():
        existing = load_intraday_event_preflight(output_directory)
        if _read_json(output_directory / "_request.json") != request:
            raise DataReadinessError(f"published A5.1 authority is immutable: {output_directory}")
        return existing

    dataset = load_published_intraday_dataset(dataset_authority_directory)
    if dataset.authority_sha256 != dataset_authority_sha256:
        raise DataReadinessError("A4.3 authority changed while A5.1 was loading")
    events, coverage = _combine_event_authorities(event_authorities, config=config)
    events, coverage = _reconcile_event_namespace(dataset.frame, events, coverage)
    decisions = _build_decision_eligibility(dataset, events, coverage, config=config)
    attachments = _build_event_attachments(decisions, events, config=config)
    coverage_audit, blockers = _build_coverage_audit(
        decisions,
        attachments,
        events,
        config=config,
    )
    status = "eligible" if not blockers else "blocked"
    training_eligible = status == "eligible"
    audit = _publication_audit(decisions, attachments, events)
    audit.raise_for_failure()

    staging = output_directory.with_name(f".{output_directory.name}.{uuid4().hex}.staging")
    staging.mkdir(parents=True)
    try:
        _atomic_json(staging / "_request.json", request)
        inputs_record = {
            "request_sha256": str(request["request_sha256"]),
            "dataset_authority_sha256": dataset.authority_sha256,
            "dataset_manifest_sha256": dataset.manifest_sha256,
            "dataset_request_sha256": dataset.request_sha256,
            "dataset_transformation_sha256": dataset.transformation_sha256,
            "dataset_session_inventory_sha256": dataset.session_unit_inventory_sha256,
            "ordered_feature_sha256": dataset.ordered_feature_sha256,
            "strategy_contract_sha256": dataset.strategy_contract_sha256,
        }
        frames = {
            "decisions": decisions,
            "attachments": attachments,
            "coverage_audit": coverage_audit,
        }
        artifact_records: dict[str, dict[str, Any]] = {}
        for name, frame in frames.items():
            filename, artifact_type = _ARTIFACTS[name]
            child = write_canonical_artifact(
                frame,
                staging / filename,
                artifact_type=artifact_type,
                audit=audit,
                inputs=inputs_record,
                production_ready=False,
            )
            (staging / f"{filename}.lock").unlink(missing_ok=True)
            child["artifact_path"] = filename
            child_manifest_path = staging / f"{filename}.manifest.json"
            _atomic_json(child_manifest_path, child)
            artifact_records[name] = {
                "path": filename,
                "rows": len(frame),
                "sha256": child["artifact_sha256"],
                "manifest_sha256": file_sha256(child_manifest_path),
            }
        manifest = {
            **request,
            "state": "complete",
            "status": status,
            "training_eligible": training_eligible,
            "serving_eligible": False,
            "future_holdout_opened": False,
            "blockers": blockers,
            "artifacts": artifact_records,
            "dataset_identity": inputs_record,
            "event_authority_identities": event_identities,
            "summary": _summary(decisions, attachments, events),
            "memory": memory_audit(
                hard_budget_gib=config.maximum_process_memory_gib,
                headroom_gib=config.memory_guard_headroom_gib,
            ).to_record(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
        }
        _atomic_json(staging / "_manifest.json", manifest)
        authority = {
            "schema": AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(staging / "_manifest.json"),
            "request_sha256": request["request_sha256"],
            "status": status,
            "training_eligible": training_eligible,
            "serving_eligible": False,
            "future_holdout_opened": False,
        }
        _atomic_json(staging / "_authority.json", authority)
        verified = _load_intraday_event_preflight(staging, verify_parents=False)
        assert_memory_budget(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
            stage="A5.1 intraday event preflight publication",
        )
        assert_peak_memory_budget(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
            stage="A5.1 intraday event preflight publication",
        )
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, output_directory)
        return IntradayEventPreflightAuthority(
            directory=output_directory,
            decisions=verified.decisions,
            attachments=verified.attachments,
            coverage_audit=verified.coverage_audit,
            manifest=verified.manifest,
            authority=verified.authority,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_intraday_event_preflight(
    directory: Path,
    *,
    verified_dataset: PublishedIntradayDataset | None = None,
    retain_verified_parent_events: bool = False,
) -> IntradayEventPreflightAuthority:
    """Strictly replay the A5.1 authority and reject inventory or lineage drift."""

    return _load_intraday_event_preflight(
        directory,
        verify_parents=True,
        verified_dataset=verified_dataset,
        retain_verified_parent_events=retain_verified_parent_events,
    )


def _load_intraday_event_preflight(
    directory: Path,
    *,
    verify_parents: bool,
    verified_dataset: PublishedIntradayDataset | None = None,
    retain_verified_parent_events: bool = False,
) -> IntradayEventPreflightAuthority:
    """Load one authority; publishers may reuse parents verified in the same process."""

    request = _read_json(directory / "_request.json")
    manifest = _read_json(directory / "_manifest.json")
    authority = _read_json(directory / "_authority.json")
    request_payload = {key: value for key, value in request.items() if key != "request_sha256"}
    request_sha256 = json_sha256(request_payload)
    status = str(manifest.get("status", ""))
    training_eligible = status == "eligible"
    if (
        request.get("schema") != MANIFEST_SCHEMA
        or request.get("identity_alignment_policy") != _IDENTITY_ALIGNMENT_POLICY
        or request.get("request_sha256") != request_sha256
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(directory / "_manifest.json")
        or authority.get("request_sha256") != request_sha256
        or status not in {"eligible", "blocked"}
        or bool(manifest.get("training_eligible")) != training_eligible
        or bool(authority.get("training_eligible")) != training_eligible
        or bool(manifest.get("serving_eligible"))
        or bool(authority.get("serving_eligible"))
        or bool(manifest.get("future_holdout_opened"))
        or bool(authority.get("future_holdout_opened"))
        or authority.get("status") != status
    ):
        raise DataReadinessError("A5.1 authority identity does not verify")
    expected_files = set(_METADATA_FILES)
    for filename, _artifact_type in _ARTIFACTS.values():
        expected_files.add(filename)
        expected_files.add(f"{filename}.manifest.json")
    observed = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    if observed != expected_files:
        raise DataReadinessError("A5.1 authority recursive file inventory differs")
    policy_path = Path(_required_text(request, "policy_path"))
    if file_sha256(policy_path) != request.get("policy_sha256"):
        raise DataReadinessError("A5.1 policy lineage differs")
    config = load_intraday_event_preflight_config(policy_path)
    if _config_record(config) != request.get("policy"):
        raise DataReadinessError("A5.1 embedded policy differs")
    dataset_directory = Path(_required_text(request, "dataset_authority_directory"))
    if file_sha256(dataset_directory / "_authority.json") != request.get("dataset_authority_sha256"):
        raise DataReadinessError("A5.1 A4.3 parent authority differs")
    if verify_parents:
        parent_dataset = verified_dataset or load_published_intraday_dataset(
            dataset_directory
        )
        if (
            parent_dataset.root.resolve() != dataset_directory.resolve()
            or parent_dataset.authority_sha256
            != request.get("dataset_authority_sha256")
        ):
            raise DataReadinessError("A5.1 strict A4.3 parent replay differs")
        if verified_dataset is None:
            del parent_dataset
            release_process_memory()
    raw_event_identities = request.get("event_authorities")
    if not isinstance(raw_event_identities, list) or not raw_event_identities:
        raise DataReadinessError("A5.1 event authority inventory is malformed")
    parent_event_parts: list[pd.DataFrame] = []
    for raw in raw_event_identities:
        if not isinstance(raw, dict):
            raise DataReadinessError("A5.1 event authority identity is malformed")
        parent = Path(_required_text(raw, "directory"))
        expected = _required_text(raw, "authority_sha256")
        if file_sha256(parent / "_authority.json") != expected:
            raise DataReadinessError("A5.1 parent event authority identity differs")
        if verify_parents:
            projected = load_issuer_event_family_authority(
                parent, expected_authority_sha256=expected
            )
            if projected.projected_inventory_sha256 != _required_text(
                raw, "projected_inventory_sha256"
            ):
                raise DataReadinessError(
                    "A5.1 parent event projected inventory differs"
                )
            if retain_verified_parent_events:
                parent_event_parts.append(
                    projected.events.loc[
                        projected.events["event_family"].astype(str).eq(
                            config.event_family
                        ),
                        [
                            "family_event_id",
                            "event_family",
                            "classification_rule_id",
                            "matched_text",
                        ],
                    ].copy()
                )
            del projected
            release_process_memory()
    frames: dict[str, pd.DataFrame] = {}
    records = manifest.get("artifacts")
    if not isinstance(records, dict):
        raise DataReadinessError("A5.1 artifact inventory is malformed")
    for name, (filename, artifact_type) in _ARTIFACTS.items():
        frame, child = load_canonical_artifact(
            directory / filename,
            expected_type=artifact_type,
            allow_research=True,
        )
        record = records.get(name)
        child_inputs = child.get("inputs")
        if (
            not isinstance(record, dict)
            or record.get("path") != filename
            or record.get("rows") != len(frame)
            or record.get("sha256") != child.get("artifact_sha256")
            or not isinstance(child_inputs, dict)
            or child_inputs.get("request_sha256") != request_sha256
            or record.get("manifest_sha256")
            != file_sha256(directory / f"{filename}.manifest.json")
            or child.get("artifact_path") != filename
            or bool(child.get("production_ready"))
        ):
            raise DataReadinessError(f"A5.1 {name} artifact lineage differs")
        frames[name] = frame
    blockers = manifest.get("blockers")
    if not isinstance(blockers, list) or (training_eligible == bool(blockers)):
        raise DataReadinessError("A5.1 blocker state differs")
    expected_summary = _summary_from_published(
        frames["decisions"], frames["attachments"], frames["coverage_audit"]
    )
    summary = manifest.get("summary")
    if not isinstance(summary, dict):
        raise DataReadinessError("A5.1 summary is malformed")
    for key, value in expected_summary.items():
        if summary.get(key) != value:
            raise DataReadinessError(f"A5.1 summary differs for {key}")
    _validate_published_frames(frames["decisions"], frames["attachments"])
    return IntradayEventPreflightAuthority(
        directory=directory,
        decisions=frames["decisions"],
        attachments=frames["attachments"],
        coverage_audit=frames["coverage_audit"],
        manifest=manifest,
        authority=authority,
        verified_parent_events=(
            pd.concat(parent_event_parts, ignore_index=True)
            if parent_event_parts
            else None
        ),
    )


def _combine_event_authorities(
    authorities: Sequence[_EventAuthoritySlice],
    *,
    config: IntradayEventPreflightConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    event_parts: list[pd.DataFrame] = []
    coverage_parts: list[pd.DataFrame] = []
    for authority in authorities:
        _validate_parent_event_authority(authority)
        event_parts.append(
            authority.events.loc[
                authority.events["source_family"].astype(str).eq(config.source_family)
                & authority.events["relation_channel"].astype(str).eq(config.relation_channel)
                & authority.events["event_family"].astype(str).eq(config.event_family)
            ].copy()
        )
        coverage_parts.append(
            authority.coverage.loc[
                authority.coverage["source_family"].astype(str).eq(config.source_family)
                & authority.coverage["event_family"].astype(str).eq(config.event_family)
            ].copy()
        )
    events = pd.concat(event_parts, ignore_index=True)
    coverage = pd.concat(coverage_parts, ignore_index=True)
    if events["family_event_id"].astype(str).duplicated().any():
        raise DataReadinessError("A5.1 parent authorities repeat an event episode")
    events["feature_available_at_utc"] = _utc(events["feature_available_at_utc"], "event availability")
    events["published_at_utc"] = _utc(events["published_at_utc"], "event publication")
    if events["feature_available_at_utc"].lt(events["published_at_utc"]).any():
        raise DataReadinessError("A5.1 event availability precedes publication")
    direct = events["relation_channel"].astype(str).eq("direct_issuer")
    if (
        events.loc[direct, "source_security_id"].astype(str)
        .ne(events.loc[direct, "security_id"].astype(str))
        .any()
    ):
        raise DataReadinessError("A5.1 direct-issuer source and target security identity differ")
    for column in ("requested_start_utc", "requested_end_utc", "completed_at_utc"):
        coverage[column] = _utc(coverage[column], column)
    if coverage["requested_end_utc"].le(coverage["requested_start_utc"]).any():
        raise DataReadinessError("A5.1 coverage interval is invalid")
    events = events.sort_values(
        ["family_event_id", "security_id", "feature_available_at_utc"],
        kind="stable",
    ).reset_index(drop=True)
    coverage = coverage.sort_values(
        ["security_id", "requested_start_utc", "requested_end_utc", "chunk_id"],
        kind="stable",
    ).reset_index(drop=True)
    _validate_production_availability(events)
    return events, coverage


def _reconcile_event_namespace(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map historical event identities onto A4.3 without weakening issuer checks."""

    required_decision_columns = {"ticker", "security_id"}
    if not required_decision_columns.issubset(decisions.columns):
        raise DataReadinessError("A5.1 decision identity spine is incomplete")
    target = decisions.loc[:, ["ticker", "security_id"]].drop_duplicates().copy()
    target["ticker"] = _normalized_ticker(target["ticker"])
    target["security_id"] = target["security_id"].astype(str).str.strip()
    if target["ticker"].eq("").any() or target["security_id"].eq("").any():
        raise DataReadinessError("A5.1 decision identity spine contains blanks")
    target_counts = target.groupby("ticker", sort=False)["security_id"].nunique()
    ambiguous_tickers = set(target_counts.loc[target_counts.ne(1)].index.astype(str))
    unique_target = (
        target.loc[~target["ticker"].isin(ambiguous_tickers)]
        .drop_duplicates("ticker", keep="first")
        .set_index("ticker")["security_id"]
    )

    aligned_events = _align_identity_frame(
        events,
        unique_target=unique_target,
        ambiguous_tickers=ambiguous_tickers,
        label="event",
    )
    aligned_coverage = _align_identity_frame(
        coverage,
        unique_target=unique_target,
        ambiguous_tickers=ambiguous_tickers,
        label="coverage",
    )
    return aligned_events, aligned_coverage


def _align_identity_frame(
    frame: pd.DataFrame,
    *,
    unique_target: pd.Series,
    ambiguous_tickers: set[str],
    label: str,
) -> pd.DataFrame:
    required = {"ticker", "security_id"}
    if not required.issubset(frame.columns):
        raise DataReadinessError(f"A5.1 {label} identity fields are incomplete")
    output = frame.copy()
    output["ticker"] = _normalized_ticker(output["ticker"])
    output["source_namespace_security_id"] = output["security_id"].astype(str).str.strip()
    output["target_security_id"] = output["ticker"].map(unique_target)
    output["identity_alignment"] = "no_exact_ticker_in_intraday_dataset"
    ambiguous = output["ticker"].isin(ambiguous_tickers)
    output.loc[ambiguous, "identity_alignment"] = "ambiguous_intraday_ticker_identity"
    matched = output["target_security_id"].notna() & ~ambiguous
    source_cik = output["source_namespace_security_id"].map(_embedded_cik)
    target_cik = output["target_security_id"].map(_embedded_cik)
    conflicting_cik = matched & source_cik.notna() & target_cik.notna() & source_cik.ne(target_cik)
    if bool(conflicting_cik.any()):
        conflicts = output.loc[
            conflicting_cik,
            ["ticker", "source_namespace_security_id", "target_security_id"],
        ].drop_duplicates()
        raise DataReadinessError(
            "A5.1 exact-ticker identity alignment found conflicting CIKs: "
            f"{conflicts.head(5).to_dict(orient='records')}"
        )
    output.loc[matched, "security_id"] = output.loc[matched, "target_security_id"].astype(str)
    output.loc[matched, "identity_alignment"] = "exact_ticker_cik_compatible"
    return output.drop(columns="target_security_id")


def _normalized_ticker(values: pd.Series) -> pd.Series:
    return values.astype(str).str.upper().str.strip()


def _embedded_cik(value: object) -> str | None:
    text = str(value).strip()
    if not text.startswith("cik:"):
        return None
    return text.split(":ticker:", maxsplit=1)[0]


def _build_decision_eligibility(
    dataset: PublishedIntradayDataset,
    events: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    config: IntradayEventPreflightConfig,
) -> pd.DataFrame:
    columns = [
        "decision_id",
        "security_id",
        "ticker",
        "session_date_et",
        "decision_time_utc",
        "feature_available_at_utc",
    ]
    output = dataset.frame.loc[:, columns].copy()
    holdout = _stable_security_holdout(output["security_id"], config.security_holdout_fraction)
    output["validation_scope"] = np.where(
        output["security_id"].astype(str).isin(holdout),
        "unseen_security",
        "seen_security",
    )
    output["development_fold"] = _development_fold(output, config=config)
    research_counts = np.zeros(len(output), dtype="int32")
    production_counts = np.zeros(len(output), dtype="int32")
    known_coverage = np.zeros(len(output), dtype=bool)
    production_coverage = np.zeros(len(output), dtype=bool)
    lookback = pd.Timedelta(hours=config.lookback_hours)
    indexed_events = {key: group for key, group in events.groupby("security_id", sort=False)}
    indexed_coverage = {key: group for key, group in coverage.groupby("security_id", sort=False)}
    for security, index in output.groupby("security_id", sort=False).groups.items():
        positions = np.asarray(index, dtype="int64")
        decision_times = output.loc[positions, "decision_time_utc"]
        decision_ns = _timestamp_ns(decision_times)
        event_rows = indexed_events.get(security)
        if event_rows is not None:
            event_ns = np.sort(_timestamp_ns(event_rows["feature_available_at_utc"]))
            starts = np.searchsorted(event_ns, decision_ns - lookback.value, side="right")
            ends = np.searchsorted(event_ns, decision_ns, side="right")
            research_counts[positions] = (ends - starts).astype("int32")
            production_times = event_rows.loc[
                event_rows["production_eligible"].astype(bool)
                & event_rows["availability_policy"].astype(str).eq("observed"),
                "feature_available_at_utc",
            ]
            production_ns = np.sort(_timestamp_ns(production_times))
            if len(production_ns):
                pstarts = np.searchsorted(production_ns, decision_ns - lookback.value, side="right")
                pends = np.searchsorted(production_ns, decision_ns, side="right")
                production_counts[positions] = (pends - pstarts).astype("int32")
        coverage_rows = indexed_coverage.get(security)
        if coverage_rows is not None:
            known_coverage[positions] = _interval_coverage(
                decision_ns,
                coverage_rows.loc[coverage_rows["missingness_known"].astype(bool)],
                lookback=lookback,
            )
            production_coverage[positions] = _interval_coverage(
                decision_ns,
                coverage_rows.loc[
                    coverage_rows["missingness_known"].astype(bool)
                    & coverage_rows["production_eligible"].astype(bool)
                ],
                lookback=lookback,
            )
    output["research_event_count_24h"] = pd.Series(research_counts, dtype="Int32")
    output.loc[~known_coverage, "research_event_count_24h"] = pd.NA
    output["production_event_count_24h"] = pd.Series(production_counts, dtype="Int32")
    output.loc[~production_coverage, "production_event_count_24h"] = pd.NA
    output["research_coverage_state"] = np.select(
        [known_coverage & (research_counts > 0), known_coverage],
        ["known_events", "known_zero_events"],
        default="unknown",
    )
    output["production_coverage_state"] = np.select(
        [production_coverage & (production_counts > 0), production_coverage],
        ["known_events", "known_zero_events"],
        default="unknown_or_proxy_only",
    )
    output["training_eligible"] = production_coverage & (production_counts > 0)
    output["ineligibility_reason"] = np.select(
        [output["training_eligible"], ~production_coverage],
        ["", "unknown_or_proxy_only_coverage"],
        default="no_observed_event_in_lookback",
    )
    return output.sort_values(["decision_time_utc", "decision_id"], kind="stable").reset_index(
        drop=True
    )


def _build_event_attachments(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: IntradayEventPreflightConfig,
) -> pd.DataFrame:
    columns = [
        "family_event_id",
        "security_id",
        "source_namespace_security_id",
        "ticker",
        "identity_alignment",
        "feature_available_at_utc",
        "decision_id",
        "decision_time_utc",
        "publication_regime",
        "availability_policy",
        "research_eligible",
        "production_eligible",
        "attachment_eligible",
        "attachment_status",
    ]
    rows: list[dict[str, Any]] = []
    decisions_by_security = {
        key: group.sort_values("decision_time_utc", kind="stable")
        for key, group in decisions.groupby("security_id", sort=False)
    }
    lookback = pd.Timedelta(hours=config.lookback_hours)
    for event in events.itertuples(index=False):
        security = str(event.security_id)
        candidates = decisions_by_security.get(security)
        available = pd.Timestamp(event.feature_available_at_utc)
        attached_count = 0
        if candidates is not None:
            times = _timestamp_ns(candidates["decision_time_utc"])
            start = int(np.searchsorted(times, available.value, side="left"))
            end = int(
                np.searchsorted(times, (available + lookback).value, side="left")
            )
        production = bool(event.production_eligible) and str(event.availability_policy) == "observed"
        if candidates is not None:
            for position in range(start, end):
                candidate = candidates.iloc[position]
                candidate_time = pd.Timestamp(candidate["decision_time_utc"])
                decision_eligible = bool(candidate["training_eligible"])
                attachment_eligible = production and decision_eligible
                status = "attached_research_only"
                if production and not decision_eligible:
                    status = "attached_decision_ineligible"
                elif attachment_eligible:
                    status = "attached_production_eligible"
                rows.append(
                    _attachment_record(
                        event=event,
                        available=available,
                        decision_id=str(candidate["decision_id"]),
                        decision_time=candidate_time,
                        production=production,
                        attachment_eligible=attachment_eligible,
                        status=status,
                    )
                )
                attached_count += 1
        if attached_count == 0:
            rows.append(
                _attachment_record(
                    event=event,
                    available=available,
                    decision_id="",
                    decision_time=pd.NaT,
                    production=production,
                    attachment_eligible=False,
                    status="no_decision_within_lookback",
                )
            )
    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values(["family_event_id", "decision_time_utc", "decision_id"], kind="stable")
        .reset_index(drop=True)
    )


def _build_coverage_audit(
    decisions: pd.DataFrame,
    attachments: pd.DataFrame,
    events: pd.DataFrame,
    *,
    config: IntradayEventPreflightConfig,
) -> tuple[pd.DataFrame, list[str]]:
    proxy_events = int(events["availability_policy"].astype(str).ne("observed").sum())
    production_events = int(
        (
            events["production_eligible"].astype(bool)
            & events["availability_policy"].astype(str).eq("observed")
        ).sum()
    )
    attached = attachments.loc[attachments["decision_id"].astype(str).ne("")]
    eligible_attachments = attached.loc[attached["attachment_eligible"].astype(bool)]
    eligible_decisions = decisions.loc[decisions["training_eligible"].astype(bool)]
    research_episodes = int(events["family_event_id"].nunique())
    attached_episodes = int(attached["family_event_id"].nunique())
    production_attached_episodes = int(
        eligible_attachments["family_event_id"].nunique()
    )
    security_count = int(eligible_decisions["security_id"].nunique())
    fit_sessions = int(
        eligible_decisions.loc[
            eligible_decisions["development_fold"].eq(-1), "session_date_et"
        ].nunique()
    )
    metrics: list[tuple[str, str, int, int, bool]] = [
        (
            "all",
            "unique_research_event_episodes",
            research_episodes,
            config.minimum_unique_event_episodes,
            research_episodes >= config.minimum_unique_event_episodes,
        ),
        (
            "all",
            "unique_production_event_episodes",
            production_attached_episodes,
            config.minimum_unique_event_episodes,
            production_attached_episodes >= config.minimum_unique_event_episodes,
        ),
        (
            "all",
            "attached_research_event_episodes",
            attached_episodes,
            config.minimum_unique_event_episodes,
            attached_episodes >= config.minimum_unique_event_episodes,
        ),
        (
            "all",
            "securities",
            security_count,
            config.minimum_securities,
            security_count >= config.minimum_securities,
        ),
        (
            "all",
            "fit_sessions",
            fit_sessions,
            config.minimum_fit_sessions,
            fit_sessions >= config.minimum_fit_sessions,
        ),
        ("all", "proxy_event_episodes", proxy_events, 0, proxy_events == 0),
    ]
    for scope in ("seen_security", "unseen_security"):
        scoped = eligible_decisions.loc[
            eligible_decisions["validation_scope"].eq(scope)
            & eligible_decisions["development_fold"].ge(0)
        ]
        scoped_securities = int(scoped["security_id"].nunique())
        eligible_rows = len(scoped)
        metrics.extend(
            [
                (scope, "rows", len(scoped), config.minimum_scope_rows, len(scoped) >= config.minimum_scope_rows),
                (
                    scope,
                    "securities",
                    scoped_securities,
                    config.minimum_scope_securities,
                    scoped_securities >= config.minimum_scope_securities,
                ),
                (
                    scope,
                    "production_eligible_rows",
                    eligible_rows,
                    config.minimum_scope_rows,
                    eligible_rows >= config.minimum_scope_rows,
                ),
            ]
        )
        for fold in range(config.validation_folds):
            folded = scoped.loc[scoped["development_fold"].eq(fold)]
            fold_securities = int(folded["security_id"].nunique())
            metrics.extend(
                [
                    (
                        f"{scope}/fold_{fold}",
                        "production_eligible_rows",
                        len(folded),
                        config.minimum_scope_rows,
                        len(folded) >= config.minimum_scope_rows,
                    ),
                    (
                        f"{scope}/fold_{fold}",
                        "securities",
                        fold_securities,
                        config.minimum_scope_securities,
                        fold_securities >= config.minimum_scope_securities,
                    ),
                ]
            )
    audit = pd.DataFrame(
        metrics,
        columns=["scope", "metric", "observed", "required", "passed"],
    )
    blockers: list[str] = []
    if proxy_events:
        blockers.append("historical_availability_proxy_only")
    if production_events == 0:
        blockers.append("no_production_eligible_events")
    if not decisions["training_eligible"].any():
        blockers.append("no_production_eligible_decisions")
    if not bool(audit["passed"].all()):
        blockers.append("minimum_capacity_gate_failed")
    return audit, blockers


def _publication_audit(
    decisions: pd.DataFrame,
    attachments: pd.DataFrame,
    events: pd.DataFrame,
) -> CanonicalAuditReport:
    duplicate_decisions = int(decisions["decision_id"].astype(str).duplicated().sum())
    duplicate_events = int(events["family_event_id"].astype(str).duplicated().sum())
    attached = attachments.loc[attachments["decision_id"].astype(str).ne("")]
    future = int(
        (
            pd.to_datetime(attached["feature_available_at_utc"], utc=True)
            > pd.to_datetime(attached["decision_time_utc"], utc=True)
        ).sum()
    )
    joined = attached.merge(
        decisions[["decision_id", "security_id"]],
        on="decision_id",
        suffixes=("_event", "_decision"),
        how="left",
        validate="many_to_one",
    )
    issuer_mismatch = (
        int(
            joined["security_id_event"].astype(str).ne(
                joined["security_id_decision"].astype(str)
            ).sum()
        )
        if not attached.empty
        else 0
    )
    identity_mismatch = int(
        attached["identity_alignment"].astype(str).ne("exact_ticker_cik_compatible").sum()
    )
    return CanonicalAuditReport(
        checks=(
            _check("decision_identity", duplicate_decisions, len(decisions), "decision_id is unique"),
            _check("event_identity", duplicate_events, len(events), "family_event_id is unique"),
            _check("causal_attachment", future, len(attached), "event availability is not after decision"),
            _check("issuer_attachment", issuer_mismatch, len(attached), "event and decision security_id match"),
            _check(
                "identity_alignment",
                identity_mismatch,
                len(attached),
                "attached events use exact ticker and CIK-compatible identity",
            ),
        )
    )


def _validate_published_frames(decisions: pd.DataFrame, attachments: pd.DataFrame) -> None:
    if decisions.empty or decisions["decision_id"].astype(str).duplicated().any():
        raise DataReadinessError("A5.1 published decisions are empty or duplicated")
    allowed_coverage = {"known_events", "known_zero_events", "unknown", "unknown_or_proxy_only"}
    if not set(decisions["research_coverage_state"].astype(str)).issubset(allowed_coverage):
        raise DataReadinessError("A5.1 research coverage state is invalid")
    if not set(decisions["production_coverage_state"].astype(str)).issubset(allowed_coverage):
        raise DataReadinessError("A5.1 production coverage state is invalid")
    attached = attachments.loc[attachments["decision_id"].astype(str).ne("")]
    if not attached.empty:
        if not attached["identity_alignment"].astype(str).eq(
            "exact_ticker_cik_compatible"
        ).all():
            raise DataReadinessError("A5.1 published attachment identity alignment differs")
        if pd.to_datetime(attached["feature_available_at_utc"], utc=True).gt(
            pd.to_datetime(attached["decision_time_utc"], utc=True)
        ).any():
            raise DataReadinessError("A5.1 published attachment uses future evidence")
        identities = decisions.set_index("decision_id")["security_id"].astype(str)
        expected = attached["decision_id"].astype(str).map(identities)
        if expected.isna().any() or not expected.eq(attached["security_id"].astype(str)).all():
            raise DataReadinessError("A5.1 published attachment issuer differs")
        decision_eligibility = attached["decision_id"].astype(str).map(
            decisions.set_index("decision_id")["training_eligible"].astype(bool)
        )
        expected_attachment_eligibility = (
            attached["production_eligible"].astype(bool)
            & decision_eligibility.astype(bool)
        )
        if not expected_attachment_eligibility.eq(
            attached["attachment_eligible"].astype(bool)
        ).all():
            raise DataReadinessError(
                "A5.1 attachment eligibility differs from event and decision eligibility"
            )


def _summary(
    decisions: pd.DataFrame,
    attachments: pd.DataFrame,
    events: pd.DataFrame,
) -> dict[str, int]:
    return {
        "decision_rows": len(decisions),
        "security_count": int(decisions["security_id"].nunique()),
        "session_count": int(decisions["session_date_et"].nunique()),
        "research_event_episodes": int(events["family_event_id"].nunique()),
        "production_event_episodes": int(
            attachments.loc[
                attachments["attachment_eligible"].astype(bool), "family_event_id"
            ].nunique()
        ),
        "attached_event_episodes": int(
            attachments.loc[
                attachments["decision_id"].astype(str).ne(""), "family_event_id"
            ].nunique()
        ),
        "event_decision_attachment_rows": int(
            attachments["decision_id"].astype(str).ne("").sum()
        ),
        "production_eligible_decision_rows": int(decisions["training_eligible"].astype(bool).sum()),
    }


def _summary_from_published(
    decisions: pd.DataFrame,
    attachments: pd.DataFrame,
    coverage_audit: pd.DataFrame,
) -> dict[str, int]:
    research = coverage_audit.loc[
        coverage_audit["scope"].astype(str).eq("all")
        & coverage_audit["metric"].astype(str).eq("unique_research_event_episodes"),
        "observed",
    ]
    return {
        "decision_rows": len(decisions),
        "security_count": int(decisions["security_id"].nunique()),
        "session_count": int(decisions["session_date_et"].nunique()),
        "research_event_episodes": int(research.iloc[0]) if len(research) == 1 else -1,
        "production_event_episodes": int(
            attachments.loc[
                attachments["attachment_eligible"].astype(bool), "family_event_id"
            ].nunique()
        ),
        "attached_event_episodes": int(
            attachments.loc[
                attachments["decision_id"].astype(str).ne(""), "family_event_id"
            ].nunique()
        ),
        "event_decision_attachment_rows": int(
            attachments["decision_id"].astype(str).ne("").sum()
        ),
        "production_eligible_decision_rows": int(decisions["training_eligible"].astype(bool).sum()),
    }


def _stable_security_holdout(values: pd.Series, fraction: float) -> frozenset[str]:
    securities = sorted(set(values.astype(str)))
    threshold = int(fraction * 2**64)
    selected = frozenset(
        security
        for security in securities
        if int(hashlib.sha256(security.encode("utf-8")).hexdigest()[:16], 16) < threshold
    )
    if not selected or len(selected) == len(securities):
        raise DataReadinessError("A5.1 stable security holdout produced an empty partition")
    return selected


def _development_fold(
    decisions: pd.DataFrame,
    *,
    config: IntradayEventPreflightConfig,
) -> np.ndarray:
    sessions = sorted(set(decisions["session_date_et"].astype(str)))
    remaining = len(sessions) - config.minimum_fit_sessions
    fold_size = remaining // config.validation_folds
    if fold_size < 1:
        raise DataReadinessError("A5.1 history is too short for frozen development folds")
    mapping = {session: -1 for session in sessions[: config.minimum_fit_sessions]}
    for fold in range(config.validation_folds):
        start = config.minimum_fit_sessions + fold * fold_size
        end = len(sessions) if fold == config.validation_folds - 1 else start + fold_size
        mapping.update({session: fold for session in sessions[start:end]})
    result: np.ndarray[Any, np.dtype[np.int16]] = (
        decisions["session_date_et"].astype(str).map(mapping).to_numpy(dtype="int16")
    )
    return result


def _interval_coverage(
    decision_ns: np.ndarray,
    coverage: pd.DataFrame,
    *,
    lookback: pd.Timedelta,
) -> np.ndarray:
    result = np.zeros(len(decision_ns), dtype=bool)
    for row in coverage.itertuples(index=False):
        start = pd.Timestamp(row.requested_start_utc).value + lookback.value
        end = pd.Timestamp(row.requested_end_utc).value
        completed = pd.Timestamp(row.completed_at_utc).value
        result |= (
            (decision_ns >= start)
            & (decision_ns <= end)
            & (decision_ns >= completed)
        )
    return result


def _attachment_record(
    *,
    event: Any,
    available: pd.Timestamp,
    decision_id: str,
    decision_time: object,
    production: bool,
    attachment_eligible: bool,
    status: str,
) -> dict[str, Any]:
    return {
        "family_event_id": str(event.family_event_id),
        "security_id": str(event.security_id),
        "source_namespace_security_id": str(event.source_namespace_security_id),
        "ticker": str(event.ticker),
        "identity_alignment": str(event.identity_alignment),
        "feature_available_at_utc": available,
        "decision_id": decision_id,
        "decision_time_utc": decision_time,
        "publication_regime": _publication_regime(pd.Timestamp(event.published_at_utc)),
        "availability_policy": str(event.availability_policy),
        "research_eligible": bool(event.research_eligible),
        "production_eligible": production,
        "attachment_eligible": attachment_eligible,
        "attachment_status": status,
    }


def _validate_production_availability(events: pd.DataFrame) -> None:
    production = events.loc[events["production_eligible"].astype(bool)]
    if production.empty:
        return
    required = {
        "first_seen_at_utc",
        "revision_id",
        "revision_available_at_utc",
        "event_available_at_utc",
        "relation_available_at_utc",
    }
    missing = sorted(required.difference(production.columns))
    if missing:
        raise DataReadinessError(
            f"A5.1 production events lack observed revision lineage: {missing}"
        )
    policies = production["availability_policy"].astype(str)
    revision_ids = production["revision_id"].fillna("").astype(str).str.strip()
    timestamps = pd.DataFrame(
        {
            column: _utc(production[column], f"production {column}")
            for column in (
                "published_at_utc",
                "event_available_at_utc",
                "relation_available_at_utc",
                "first_seen_at_utc",
                "revision_available_at_utc",
            )
        }
    )
    expected = timestamps.max(axis=1)
    feature_available = _utc(
        production["feature_available_at_utc"], "production feature availability"
    )
    if (
        policies.ne("observed").any()
        or revision_ids.eq("").any()
        or feature_available.lt(expected).any()
    ):
        raise DataReadinessError(
            "A5.1 production event does not have observed first-seen/revision-safe availability"
        )


def _validate_parent_event_authority(authority: _EventAuthoritySlice) -> None:
    assignments = authority.assignments
    if assignments.empty:
        return
    assigned = assignments.loc[assignments["status"].astype(str).eq("assigned")]
    if assigned.empty:
        return
    available = _utc(assigned["feature_available_at_utc"], "parent assignment availability")
    decisions = _utc(assigned["decision_time_utc"], "parent assignment decision time")
    if available.gt(decisions).any():
        raise DataReadinessError("A5.1 parent authority assigns future event evidence")
    if (
        "source_security_id" in assigned.columns
        and assigned["source_security_id"].astype(str)
        .ne(assigned["security_id"].astype(str))
        .any()
    ):
        raise DataReadinessError("A5.1 parent assignment issuer identity differs")


def _timestamp_ns(values: pd.Series) -> np.ndarray[Any, np.dtype[np.int64]]:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise DataReadinessError("A5.1 timestamp conversion contains invalid UTC values")
    result: np.ndarray[Any, np.dtype[np.int64]] = (
        parsed.to_numpy(dtype="datetime64[ns]").astype("int64", copy=False)
    )
    return result


def _publication_regime(timestamp: pd.Timestamp) -> str:
    local = timestamp.tz_convert("America/New_York")
    if local.dayofweek >= 5:
        return "non_session"
    minutes = local.hour * 60 + local.minute
    if minutes < 9 * 60 + 30:
        return "premarket"
    if minutes < 16 * 60:
        return "regular_session"
    return "after_close"


def _config_record(config: IntradayEventPreflightConfig) -> dict[str, Any]:
    return asdict(config)


def _validate_config(config: IntradayEventPreflightConfig) -> None:
    observed = _config_record(config)
    if observed != _FROZEN_POLICY:
        differences = sorted(
            key
            for key, expected in _FROZEN_POLICY.items()
            if observed.get(key) != expected
        )
        raise DataReadinessError(
            "intraday event preflight policy differs from the frozen contract: "
            + ", ".join(differences)
        )


def _check(name: str, failures: int, rows: int, detail: str) -> CanonicalAuditCheck:
    return CanonicalAuditCheck(
        name=name,
        status="pass" if failures == 0 else "fail",
        failures=failures,
        rows_checked=rows,
        detail=detail,
    )


def _require_path_isolation(output: Path, inputs: Sequence[Path]) -> None:
    target = output.resolve()
    for source in inputs:
        resolved = source.resolve()
        if target == resolved or target in resolved.parents or resolved in target.parents:
            raise DataReadinessError("A5.1 output overlaps an input")


def _utc(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise DataReadinessError(f"A5.1 {name} contains invalid UTC timestamps")
    return parsed


def _required_text(value: Mapping[str, Any], key: str) -> str:
    result = str(value.get(key, "")).strip()
    if not result:
        raise DataReadinessError(f"A5.1 requires {key}")
    return result


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataReadinessError(f"A5.1 expected a JSON object: {path}")
    return {str(key): item for key, item in value.items()}


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
