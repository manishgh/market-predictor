"""Immutable source-side horizon for prospectively observed analyst revisions."""
from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, cast
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.catalysts.issuer_events.classification import (
    EVENT_FAMILY_POLICY_SHA256,
    EVENT_FAMILY_POLICY_VERSION,
    classify_event_families,
)
from market_predictor.core import path_integrity
from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.datasets.event_preflight import (
    load_intraday_event_preflight_config,
)
from market_predictor.intraday.datasets.history import json_sha256
from market_predictor.intraday.datasets.prospective_broker_actions import (
    ProspectiveGeneration,
    ProspectivePoll,
    load_prospective_broker_action_generation,
    load_prospective_broker_action_poll,
)
from market_predictor.locking import LockTimeout, file_lock
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
)
from market_predictor.universe.sp500.observed_membership_authority import (
    AUTHORITY_SCHEMA as OBSERVED_MEMBERSHIP_AUTHORITY_SCHEMA,
)
from market_predictor.universe.sp500.observed_membership_authority import (
    load_observed_sp500_membership_authority,
)

REQUEST_SCHEMA: Final = "edge_rebuild.prospective_analyst_revision_horizon_request.v1"
MANIFEST_SCHEMA: Final = "edge_rebuild.prospective_analyst_revision_horizon_manifest.v1"
AUTHORITY_SCHEMA: Final = "edge_rebuild.prospective_analyst_revision_horizon_authority.v1"
STAGING_OWNER_SCHEMA: Final = (
    "edge_rebuild.prospective_analyst_revision_horizon_staging_owner.v1"
)
CLASSIFIED_ARTIFACT_TYPE: Final = "prospective_analyst_revision_classified_revisions"
EPISODE_ARTIFACT_TYPE: Final = "prospective_analyst_revision_episodes"
COVERAGE_ARTIFACT_TYPE: Final = "prospective_analyst_revision_collection_coverage"
CAPACITY_ARTIFACT_TYPE: Final = "prospective_analyst_revision_source_capacity"
_ARTIFACTS: Final = {
    "classified_revisions": ("classified_revisions.parquet", CLASSIFIED_ARTIFACT_TYPE),
    "episodes": ("episodes.parquet", EPISODE_ARTIFACT_TYPE),
    "coverage": ("coverage.parquet", COVERAGE_ARTIFACT_TYPE),
    "capacity_audit": ("capacity_audit.parquet", CAPACITY_ARTIFACT_TYPE),
}
_METADATA_FILES: Final = frozenset({"_request.json", "_manifest.json", "_authority.json"})
MAX_PARENT_INPUT_BYTES: Final = 1024 * 1024 * 1024
DERIVATION_EXPANSION_FACTOR: Final = 16
_REQUEST_KEYS: Final = frozenset(
    {
        "schema",
        "generations",
        "generation_inventory_sha256",
        "flattened_poll_inventory_sha256",
        "security_identity_namespace_sha256",
        "registry_directory",
        "preflight_policy_path",
        "preflight_policy_sha256",
        "preflight_policy",
        "event_family_policy_version",
        "event_family_policy_sha256",
        "episode_identity",
        "availability_policy",
        "memory_hard_budget_gib",
        "memory_headroom_gib",
        "request_sha256",
    }
)
_GENERATION_RECORD_KEYS: Final = frozenset(
    {
        "directory",
        "request_sha256",
        "manifest_sha256",
        "authority_sha256",
        "poll_inventory_sha256",
    }
)
_MANIFEST_KEYS: Final = frozenset(
    {
        "schema",
        "status",
        "request_sha256",
        "generation_count",
        "poll_count",
        "classified_revision_count",
        "analyst_episode_count",
        "eligible_security_count",
        "source_capacity_status",
        "artifacts",
        "artifact_manifest_hashes",
        "training_eligible",
        "serving_eligible",
        "future_holdout_opened",
        "memory",
    }
)
_AUTHORITY_KEYS: Final = frozenset(
    {
        "schema",
        "state",
        "artifact",
        "artifact_sha256",
        "request_sha256",
        "source_capacity_status",
        "training_eligible",
        "serving_eligible",
        "future_holdout_opened",
    }
)
_ARTIFACT_RECORD_KEYS: Final = frozenset({"path", "sha256", "bytes"})
_MEMORY_KEYS: Final = frozenset(
    {
        "current_working_set_gib",
        "hard_budget_gib",
        "peak_working_set_gib",
        "safety_threshold_gib",
    }
)
_CANONICAL_MANIFEST_KEYS: Final = frozenset(
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
_CANONICAL_AUDIT_KEYS: Final = frozenset(
    {"name", "status", "failures", "rows_checked", "detail"}
)
_CLASSIFIED_COLUMNS: Final = (
    "revision_event_id",
    "revision_id",
    "provider_event_id",
    "ticker",
    "security_id",
    "source_family",
    "relation_channel",
    "title",
    "published_at_utc",
    "provider_updated_at_utc",
    "revision_first_seen_at_utc",
    "event_first_seen_at_utc",
    "production_available_at_utc",
    "identity_eligible",
    "issuer_company",
    "issuer_company_available_at_utc",
    "classified_analyst_revision",
    "classification_rule_id",
    "classification_basis",
    "matched_text",
    "eligibility_reason",
)
_EPISODE_COLUMNS: Final = (
    "source_episode_id",
    "family_event_id",
    "provider_event_id",
    "ticker",
    "security_id",
    "first_qualifying_revision_event_id",
    "feature_available_at_utc",
    "title",
    "classification_rule_id",
    "classification_basis",
    "revision_count",
    "event_family_policy_sha256",
)
_COVERAGE_COLUMNS: Final = (
    "collection_id",
    "ticker",
    "source_family",
    "requested_start_utc",
    "requested_end_utc",
    "started_at_utc",
    "completed_at_utc",
    "status",
    "row_count",
    "error_type",
    "schema_version",
    "scheduled_poll_at_utc",
    "continuous_from_previous_poll",
    "previous_poll_at_utc",
    "batch_id",
    "security_id",
    "identity_eligible",
    "identity_ineligible_reason",
    "poll_observed_at_utc",
    "poll_authority_sha256",
    "generation_directory",
)
_CAPACITY_COLUMNS: Final = (
    "generation_count",
    "poll_count",
    "observation_date_count",
    "retained_revision_count",
    "classified_analyst_revision_count",
    "unique_analyst_episode_count",
    "eligible_security_count",
    "minimum_unique_event_episodes",
    "minimum_securities",
    "minimum_fit_sessions",
    "source_capacity_status",
    "matched_decision_capacity_evaluated",
    "training_eligible",
    "serving_eligible",
    "future_holdout_opened",
)
_OUTPUT_COLUMNS: Final = {
    "classified_revisions": _CLASSIFIED_COLUMNS,
    "episodes": _EPISODE_COLUMNS,
    "coverage": _COVERAGE_COLUMNS,
    "capacity_audit": _CAPACITY_COLUMNS,
}


@dataclass(frozen=True, slots=True)
class ProspectiveAnalystRevisionHorizon:
    directory: Path
    classified_revisions: pd.DataFrame
    episodes: pd.DataFrame
    coverage: pd.DataFrame
    capacity_audit: pd.DataFrame
    manifest: Mapping[str, object]
    authority: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _Inputs:
    generations: tuple[ProspectiveGeneration, ...]
    polls: tuple[ProspectivePoll, ...]
    generation_inventory: tuple[Mapping[str, object], ...]
    poll_inventory_sha256: str
    namespace_sha256: str
    registry_directory: str


def publish_prospective_analyst_revision_horizon(
    *,
    generation_directories: Sequence[Path],
    output_directory: Path,
    preflight_policy_path: Path,
    memory_hard_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.75,
) -> ProspectiveAnalystRevisionHorizon:
    """Publish classified prospective source evidence without authorizing training."""

    _validate_memory_policy(memory_hard_budget_gib, memory_headroom_gib)
    output = path_integrity.verify_no_reparse_ancestry(
        output_directory,
        label="prospective analyst horizon output",
    ).resolve(strict=False)
    policy_absolute = path_integrity.verify_no_reparse_ancestry(
        preflight_policy_path,
        label="prospective analyst horizon policy",
    )
    try:
        policy_path = policy_absolute.resolve(strict=True)
    except OSError as exc:
        raise DataReadinessError(
            "prospective analyst horizon policy is unavailable"
        ) from exc
    if not policy_path.is_file():
        raise DataReadinessError("prospective analyst horizon policy is unavailable")
    output.parent.mkdir(parents=True, exist_ok=True)
    path_integrity.verify_no_reparse_ancestry(
        output,
        label="prospective analyst horizon output",
    )
    staging = output.with_name(f".{output.name}.staging")
    staging_owner = output.with_name(f".{output.name}.staging.owner.json")
    try:
        with file_lock(output.with_name(f".{output.name}.publisher"), timeout=0.0):
            if output.exists():
                raise DataReadinessError(
                    f"prospective horizon output must be new: {output}"
                )
            _remove_owned_staging(
                staging,
                output=output,
                owner_path=staging_owner,
            )
            _write_new_json(
                staging_owner,
                {
                    "schema": STAGING_OWNER_SCHEMA,
                    "staging_directory": str(staging),
                    "output_directory": str(output),
                },
            )
            staging.mkdir()
            try:
                staged = _publish(
                    generation_directories=generation_directories,
                    output=staging,
                    published_output=output,
                    preflight_policy_path=policy_path,
                    memory_hard_budget_gib=memory_hard_budget_gib,
                    memory_headroom_gib=memory_headroom_gib,
                )
                if output.exists():
                    raise DataReadinessError(
                        "prospective horizon output appeared during publication"
                )
                staging.replace(output)
            except Exception:
                _remove_owned_staging(
                    staging,
                    output=output,
                    owner_path=staging_owner,
                )
                raise
            try:
                staging_owner.unlink(missing_ok=True)
            except OSError:
                pass
            return ProspectiveAnalystRevisionHorizon(
                directory=output,
                classified_revisions=staged.classified_revisions,
                episodes=staged.episodes,
                coverage=staged.coverage,
                capacity_audit=staged.capacity_audit,
                manifest=staged.manifest,
                authority=staged.authority,
            )
    except LockTimeout as exc:
        raise DataReadinessError(f"another process owns prospective horizon {output}") from exc


def _publish(
    *,
    generation_directories: Sequence[Path],
    output: Path,
    published_output: Path,
    preflight_policy_path: Path,
    memory_hard_budget_gib: float,
    memory_headroom_gib: float,
) -> ProspectiveAnalystRevisionHorizon:
    policy = load_intraday_event_preflight_config(preflight_policy_path)
    if policy.source_family != "alpaca" or policy.event_family != "analyst_revision":
        raise DataReadinessError("prospective horizon requires the frozen Alpaca analyst policy")
    assert_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective analyst horizon before parent replay",
    )
    inputs = _load_inputs(
        generation_directories,
        output=output,
        memory_hard_budget_gib=memory_hard_budget_gib,
        memory_headroom_gib=memory_headroom_gib,
    )
    assert_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective analyst horizon parent replay",
    )
    frames = _build_frames(
        inputs,
        policy=policy,
        memory_hard_budget_gib=memory_hard_budget_gib,
        memory_headroom_gib=memory_headroom_gib,
    )
    request_payload: dict[str, object] = {
        "schema": REQUEST_SCHEMA,
        "generations": list(inputs.generation_inventory),
        "generation_inventory_sha256": json_sha256(list(inputs.generation_inventory)),
        "flattened_poll_inventory_sha256": inputs.poll_inventory_sha256,
        "security_identity_namespace_sha256": inputs.namespace_sha256,
        "registry_directory": inputs.registry_directory,
        "preflight_policy_path": str(preflight_policy_path),
        "preflight_policy_sha256": file_sha256(preflight_policy_path),
        "preflight_policy": asdict(policy),
        "event_family_policy_version": EVENT_FAMILY_POLICY_VERSION,
        "event_family_policy_sha256": EVENT_FAMILY_POLICY_SHA256,
        "episode_identity": "sha256(alpaca|provider_event_id|security_id)",
        "availability_policy": "earliest_observed_provider_response",
        "memory_hard_budget_gib": memory_hard_budget_gib,
        "memory_headroom_gib": memory_headroom_gib,
    }
    request_sha256 = json_sha256(request_payload)
    request = {**request_payload, "request_sha256": request_sha256}
    _atomic_json(output / "_request.json", request)
    child_inputs: dict[str, str] = {
        "request_sha256": request_sha256,
        "generation_inventory_sha256": str(request_payload["generation_inventory_sha256"]),
        "flattened_poll_inventory_sha256": inputs.poll_inventory_sha256,
        "event_family_policy_sha256": EVENT_FAMILY_POLICY_SHA256,
        "preflight_policy_sha256": str(request_payload["preflight_policy_sha256"]),
    }
    artifacts: dict[str, dict[str, object]] = {}
    for role, (filename, artifact_type) in _ARTIFACTS.items():
        path = output / filename
        frame = frames[role]
        write_canonical_artifact(
            frame,
            path,
            artifact_type=artifact_type,
            audit=_passing_audit(role, len(frame)),
            inputs=child_inputs,
            production_ready=False,
        )
        _retarget_canonical_manifest(
            path,
            published_path=published_output / filename,
        )
        artifacts[role] = _artifact_record(path)
    capacity = frames["capacity_audit"].iloc[0]
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "request_sha256": request_sha256,
        "generation_count": len(inputs.generations),
        "poll_count": len(inputs.polls),
        "classified_revision_count": len(frames["classified_revisions"]),
        "analyst_episode_count": len(frames["episodes"]),
        "eligible_security_count": int(capacity["eligible_security_count"]),
        "source_capacity_status": str(capacity["source_capacity_status"]),
        "artifacts": artifacts,
        "artifact_manifest_hashes": {
            role: file_sha256(manifest_path_for(output / filename))
            for role, (filename, _) in _ARTIFACTS.items()
        },
        "training_eligible": False,
        "serving_eligible": False,
        "future_holdout_opened": False,
        "memory": memory_audit(
            hard_budget_gib=memory_hard_budget_gib,
            headroom_gib=memory_headroom_gib,
        ).to_record(),
    }
    assert_peak_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective analyst horizon publication",
    )
    _atomic_json(output / "_manifest.json", manifest)
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "state": "complete",
        "artifact": "_manifest.json",
        "artifact_sha256": file_sha256(output / "_manifest.json"),
        "request_sha256": request_sha256,
        "source_capacity_status": manifest["source_capacity_status"],
        "training_eligible": False,
        "serving_eligible": False,
        "future_holdout_opened": False,
    }
    _atomic_json(output / "_authority.json", authority)
    return _load_prospective_analyst_revision_horizon(
        output,
        expected_artifact_directory=published_output,
    )


def load_prospective_analyst_revision_horizon(
    directory: Path,
) -> ProspectiveAnalystRevisionHorizon:
    """Strictly replay a prospective analyst-event source horizon."""

    return _load_prospective_analyst_revision_horizon(
        directory,
        expected_artifact_directory=directory,
    )


def _load_prospective_analyst_revision_horizon(
    directory: Path,
    *,
    expected_artifact_directory: Path,
) -> ProspectiveAnalystRevisionHorizon:
    """Replay a horizon staged for one immutable publication directory."""

    root = path_integrity.verify_tree_containment(
        directory,
        label="prospective analyst horizon",
    )
    expected_artifact_root = path_integrity.verify_no_reparse_ancestry(
        expected_artifact_directory,
        label="prospective analyst horizon artifact root",
    ).resolve(strict=False)
    expected_files = set(_METADATA_FILES)
    for filename, _ in _ARTIFACTS.values():
        expected_files.update({filename, f"{filename}.manifest.json", f"{filename}.lock"})
    if {path.name for path in root.iterdir()} != expected_files:
        raise DataReadinessError("prospective analyst horizon root inventory does not verify")
    request = _json_object(root / "_request.json")
    generation_records, hard_budget_gib, headroom_gib = _validate_request(request)
    request_payload = {str(key): value for key, value in request.items() if key != "request_sha256"}
    request_sha256 = request.get("request_sha256")
    policy_path = _required_existing_file(
        request_payload,
        "preflight_policy_path",
        label="prospective analyst horizon policy",
    )
    if file_sha256(policy_path) != request_payload.get("preflight_policy_sha256"):
        raise DataReadinessError("prospective analyst horizon preflight policy changed")
    policy = load_intraday_event_preflight_config(policy_path)
    if asdict(policy) != request_payload.get("preflight_policy"):
        raise DataReadinessError("prospective analyst horizon preflight values changed")
    directories = [
        Path(_required_string(record, "directory"))
        for record in generation_records
    ]
    inputs = _load_inputs(
        directories,
        output=root,
        memory_hard_budget_gib=hard_budget_gib,
        memory_headroom_gib=headroom_gib,
    )
    if (
        list(inputs.generation_inventory) != generation_records
        or inputs.poll_inventory_sha256 != request_payload.get("flattened_poll_inventory_sha256")
        or inputs.namespace_sha256 != request_payload.get("security_identity_namespace_sha256")
        or inputs.registry_directory != request_payload.get("registry_directory")
    ):
        raise DataReadinessError("prospective analyst horizon parent lineage changed")
    manifest = _json_object(root / "_manifest.json")
    authority = _json_object(root / "_authority.json")
    if (
        set(manifest) != _MANIFEST_KEYS
        or set(authority) != _AUTHORITY_KEYS
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(root / "_manifest.json")
        or authority.get("request_sha256") != request_sha256
        or manifest.get("training_eligible") is not False
        or manifest.get("serving_eligible") is not False
        or manifest.get("future_holdout_opened") is not False
        or authority.get("training_eligible") is not False
        or authority.get("serving_eligible") is not False
        or authority.get("future_holdout_opened") is not False
    ):
        raise DataReadinessError("prospective analyst horizon authority does not verify")
    _validate_memory_record(
        manifest.get("memory"),
        hard_budget_gib=hard_budget_gib,
        headroom_gib=headroom_gib,
    )
    records = manifest.get("artifacts")
    sidecars = manifest.get("artifact_manifest_hashes")
    if (
        not isinstance(records, Mapping)
        or not isinstance(sidecars, Mapping)
        or set(records) != set(_ARTIFACTS)
        or set(sidecars) != set(_ARTIFACTS)
    ):
        raise DataReadinessError("prospective analyst horizon artifact inventory is malformed")
    child_inputs: dict[str, str] = {
        "request_sha256": str(request_sha256),
        "generation_inventory_sha256": str(request_payload["generation_inventory_sha256"]),
        "flattened_poll_inventory_sha256": inputs.poll_inventory_sha256,
        "event_family_policy_sha256": EVENT_FAMILY_POLICY_SHA256,
        "preflight_policy_sha256": str(request_payload["preflight_policy_sha256"]),
    }
    loaded: dict[str, pd.DataFrame] = {}
    for role, (filename, artifact_type) in _ARTIFACTS.items():
        record = records.get(role)
        if not isinstance(record, Mapping) or set(record) != _ARTIFACT_RECORD_KEYS:
            raise DataReadinessError(f"prospective analyst horizon {role} is missing")
        path = root / filename
        strict_child = _json_object(manifest_path_for(path))
        _validate_child_manifest(
            strict_child,
            role=role,
            artifact_type=artifact_type,
            artifact_path=expected_artifact_root / filename,
            child_inputs=child_inputs,
        )
        if (
            record.get("path") != filename
            or file_sha256(path) != record.get("sha256")
            or _required_int(record, "bytes", minimum=0) != path.stat().st_size
            or file_sha256(manifest_path_for(path)) != sidecars.get(role)
        ):
            raise DataReadinessError(f"prospective analyst horizon {role} hash changed")
        frame, child = load_canonical_artifact(path, expected_type=artifact_type, allow_research=True)
        if (
            child != strict_child
            or child.get("inputs") != child_inputs
            or child.get("production_ready") is not False
            or child.get("artifact_path")
            != str(expected_artifact_root / filename)
        ):
            raise DataReadinessError(f"prospective analyst horizon {role} lineage changed")
        loaded[role] = frame
    expected = _build_frames(
        inputs,
        policy=policy,
        memory_hard_budget_gib=hard_budget_gib,
        memory_headroom_gib=headroom_gib,
    )
    _require_output_schemas(loaded)
    if any(not _frames_equal(loaded[role], expected[role]) for role in _ARTIFACTS):
        raise DataReadinessError("prospective analyst horizon does not replay from parents")
    capacity = loaded["capacity_audit"].iloc[0]
    if (
        _required_int(manifest, "generation_count", minimum=0)
        != len(inputs.generations)
        or _required_int(manifest, "poll_count", minimum=0) != len(inputs.polls)
        or _required_int(manifest, "classified_revision_count", minimum=0)
        != len(loaded["classified_revisions"])
        or _required_int(manifest, "analyst_episode_count", minimum=0)
        != len(loaded["episodes"])
        or _required_int(manifest, "eligible_security_count", minimum=0)
        != int(capacity["eligible_security_count"])
        or manifest.get("source_capacity_status") != capacity["source_capacity_status"]
        or authority.get("source_capacity_status") != capacity["source_capacity_status"]
    ):
        raise DataReadinessError("prospective analyst horizon counts do not verify")
    return ProspectiveAnalystRevisionHorizon(
        directory=root,
        classified_revisions=loaded["classified_revisions"],
        episodes=loaded["episodes"],
        coverage=loaded["coverage"],
        capacity_audit=loaded["capacity_audit"],
        manifest=manifest,
        authority=authority,
    )


def _load_inputs(
    generation_directories: Sequence[Path],
    *,
    output: Path,
    memory_hard_budget_gib: float,
    memory_headroom_gib: float,
) -> _Inputs:
    if not generation_directories:
        raise ValueError("generation_directories must not be empty")
    roots = tuple(
        path_integrity.verify_tree_containment(
            path,
            label="prospective analyst horizon generation",
        )
        for path in generation_directories
    )
    if len(roots) != len(set(roots)):
        raise DataReadinessError("prospective horizon contains duplicate generations")
    if any(output == root or output in root.parents or root in output.parents for root in roots):
        raise DataReadinessError("prospective horizon output and parents must be disjoint")
    input_bytes = 0
    generation_items: list[ProspectiveGeneration] = []
    for root in roots:
        input_bytes += _directory_file_bytes(root)
        if input_bytes > MAX_PARENT_INPUT_BYTES:
            raise DataReadinessError(
                "prospective horizon parents exceed the bounded input limit"
            )
        assert_memory_budget(
            hard_budget_gib=memory_hard_budget_gib,
            headroom_gib=memory_headroom_gib,
            stage="prospective analyst horizon before generation load",
        )
        generation_items.append(load_prospective_broker_action_generation(root))
        assert_memory_budget(
            hard_budget_gib=memory_hard_budget_gib,
            headroom_gib=memory_headroom_gib,
            stage="prospective analyst horizon after generation load",
        )
    generations = tuple(generation_items)
    first_cutoffs = [pd.Timestamp(item.manifest["first_poll_at_utc"]) for item in generations]
    if first_cutoffs != sorted(first_cutoffs):
        raise DataReadinessError("prospective horizon generations must be chronological")
    generation_inventory = tuple(
        {
            "directory": str(item.directory),
            "request_sha256": item.request["request_sha256"],
            "manifest_sha256": file_sha256(item.directory / "_manifest.json"),
            "authority_sha256": file_sha256(item.directory / "_authority.json"),
            "poll_inventory_sha256": item.request["poll_inventory_sha256"],
        }
        for item in generations
    )
    poll_records: list[Mapping[str, object]] = []
    polls: list[ProspectivePoll] = []
    seen_poll_roots: set[Path] = set()
    for generation in generations:
        raw_records = generation.request.get("polls")
        if not isinstance(raw_records, list):
            raise DataReadinessError("prospective generation poll inventory is malformed")
        for raw in raw_records:
            if not isinstance(raw, Mapping):
                raise DataReadinessError("prospective generation poll record is malformed")
            poll_root = path_integrity.verify_tree_containment(
                Path(_required_string(raw, "directory")),
                label="prospective analyst horizon poll",
            )
            if poll_root in seen_poll_roots:
                raise DataReadinessError("prospective horizon contains overlapping polls")
            seen_poll_roots.add(poll_root)
            input_bytes += _directory_file_bytes(poll_root)
            if input_bytes > MAX_PARENT_INPUT_BYTES:
                raise DataReadinessError(
                    "prospective horizon parents exceed the bounded input limit"
                )
            assert_memory_budget(
                hard_budget_gib=memory_hard_budget_gib,
                headroom_gib=memory_headroom_gib,
                stage="prospective analyst horizon before poll load",
            )
            poll = load_prospective_broker_action_poll(poll_root)
            assert_memory_budget(
                hard_budget_gib=memory_hard_budget_gib,
                headroom_gib=memory_headroom_gib,
                stage="prospective analyst horizon after poll load",
            )
            polls.append(poll)
            poll_records.append(dict(raw))
    cutoffs = [pd.Timestamp(item.manifest["observed_at_utc"]) for item in polls]
    if cutoffs != sorted(cutoffs) or len(cutoffs) != len(set(cutoffs)):
        raise DataReadinessError("prospective horizon poll cutoffs are not unique and chronological")
    for previous, current in zip(polls, polls[1:], strict=False):
        parent = current.request.get("previous_poll")
        if (
            not isinstance(parent, Mapping)
            or path_integrity.verify_no_reparse_ancestry(
                Path(_required_string(parent, "directory")),
                label="prospective analyst horizon previous poll",
            ).resolve()
            != previous.directory
        ):
            raise DataReadinessError("prospective horizon poll chain is not contiguous")
        if parent.get("authority_sha256") != file_sha256(previous.directory / "_authority.json"):
            raise DataReadinessError("prospective horizon previous-poll authority changed")
    namespace_values = {str(item.request["security_identity_namespace_sha256"]) for item in polls}
    registry_values = {str(item.request["registry_directory"]) for item in polls}
    if len(namespace_values) != 1 or len(registry_values) != 1:
        raise DataReadinessError("prospective horizon namespace or registry changed")
    return _Inputs(
        generations=generations,
        polls=tuple(polls),
        generation_inventory=generation_inventory,
        poll_inventory_sha256=json_sha256(poll_records),
        namespace_sha256=next(iter(namespace_values)),
        registry_directory=next(iter(registry_values)),
    )


def _build_frames(
    inputs: _Inputs,
    *,
    policy: Any,
    memory_hard_budget_gib: float,
    memory_headroom_gib: float,
) -> dict[str, pd.DataFrame]:
    _assert_derivation_capacity(
        inputs,
        memory_hard_budget_gib=memory_hard_budget_gib,
        memory_headroom_gib=memory_headroom_gib,
    )
    revisions = _merge_revisions(inputs.generations)
    assert_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective analyst horizon revision merge",
    )
    first_poll_by_revision: dict[tuple[str, str], ProspectivePoll] = {}
    for poll in inputs.polls:
        for row in poll.observations.to_dict("records"):
            key = (str(row["revision_id"]), str(row["ticker"]))
            first_poll_by_revision.setdefault(key, poll)
    assert_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective analyst horizon first-observation index",
    )
    company_maps = {poll.directory: _causal_company_map(poll) for poll in inputs.polls}
    assert_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective analyst horizon issuer index",
    )
    security_conflicts = (
        revisions.loc[revisions["candidate_security_id"].astype(str).ne("")]
        .groupby(["provider_event_id", "ticker"], sort=False)["candidate_security_id"]
        .nunique()
        .gt(1)
    )
    asset_conflicts = (
        revisions.loc[revisions["alpaca_asset_id"].astype(str).ne("")]
        .groupby(["provider_event_id", "ticker"], sort=False)["alpaca_asset_id"]
        .nunique()
        .gt(1)
    )
    rows: list[dict[str, object]] = []
    classifier_inputs: list[dict[str, object]] = []
    for row in revisions.to_dict("records"):
        key = (str(row["revision_id"]), str(row["ticker"]))
        first_seen_poll = first_poll_by_revision.get(key)
        security_id = str(row["candidate_security_id"])
        event_key = (str(row["provider_event_id"]), str(row["ticker"]))
        conflict = bool(security_conflicts.get(event_key, False)) or bool(
            asset_conflicts.get(event_key, False)
        )
        eligible = bool(row["identity_eligible"]) and not bool(row["provider_timestamp_anomaly"]) and not conflict
        revision_event_id = hashlib.sha256(
            f"{row['revision_id']}|{row['ticker']}|{security_id}".encode()
        ).hexdigest()
        company = ""
        company_available: object = pd.NaT
        if first_seen_poll is not None:
            company, company_available = company_maps[first_seen_poll.directory].get(
                (str(row["ticker"]), security_id),
                ("", pd.NaT),
            )
        reason = ""
        if conflict:
            reason = "security_identity_changed_across_horizon"
        elif not bool(row["identity_eligible"]):
            reason = str(row["identity_ineligible_reason"] or "identity_ineligible")
        elif bool(row["provider_timestamp_anomaly"]):
            reason = "provider_timestamp_anomaly"
        record = {
            "revision_event_id": revision_event_id,
            "revision_id": str(row["revision_id"]),
            "provider_event_id": str(row["provider_event_id"]),
            "ticker": str(row["ticker"]),
            "security_id": security_id,
            "source_family": "alpaca",
            "relation_channel": "direct_issuer",
            "title": str(row["title"]),
            "published_at_utc": row["published_at_utc"],
            "provider_updated_at_utc": row["provider_updated_at_utc"],
            "revision_first_seen_at_utc": row["revision_first_seen_at_utc"],
            "event_first_seen_at_utc": row["event_first_seen_at_utc"],
            "production_available_at_utc": (
                row["production_available_at_utc"] if eligible else pd.NaT
            ),
            "identity_eligible": eligible,
            "issuer_company": company,
            "issuer_company_available_at_utc": company_available,
            "classified_analyst_revision": False,
            "classification_rule_id": "",
            "classification_basis": "",
            "matched_text": "",
            "eligibility_reason": reason,
        }
        rows.append(record)
        if eligible:
            classifier_inputs.append(
                {
                    "event_id": revision_event_id,
                    "security_id": security_id,
                    "ticker": str(row["ticker"]),
                    "source_family": "alpaca",
                    "feature_available_at_utc": row["production_available_at_utc"],
                    "title": str(row["title"]),
                    "issuer_company": company,
                    "issuer_company_available_at_utc": company_available,
                }
            )
    assert_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective analyst horizon classification records",
    )
    classified = classify_event_families(pd.DataFrame(classifier_inputs)) if classifier_inputs else pd.DataFrame()
    analyst = classified[classified["event_family"].eq("analyst_revision")] if not classified.empty else classified
    by_event = {str(item["event_id"]): item for item in analyst.to_dict("records")}
    for record in rows:
        match = by_event.get(str(record["revision_event_id"]))
        if match is not None:
            record["classified_analyst_revision"] = True
            record["classification_rule_id"] = str(match["classification_rule_id"])
            record["classification_basis"] = str(match["classification_basis"])
            record["matched_text"] = str(match["matched_text"])
    classified_revisions = pd.DataFrame(rows, columns=_CLASSIFIED_COLUMNS).sort_values(
        ["revision_first_seen_at_utc", "provider_event_id", "ticker", "revision_id"], kind="stable"
    ).reset_index(drop=True)
    classified_revisions = _normalize_classified_revisions(classified_revisions)
    episodes = _episodes(classified_revisions)
    assert_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective analyst horizon classification",
    )
    coverage_parts: list[pd.DataFrame] = []
    generation_by_poll = {
        path_integrity.verify_no_reparse_ancestry(
            Path(_required_string(record, "directory")),
            label="prospective analyst horizon generation poll",
        ).resolve(): generation.directory
        for generation in inputs.generations
        for record in cast(list[Mapping[str, object]], generation.request["polls"])
    }
    for poll in inputs.polls:
        part = poll.source_collections.copy()
        identity = poll.identity_audit.loc[
            :,
            [
                "ticker",
                "candidate_security_id",
                "identity_eligible",
                "identity_ineligible_reason",
            ],
        ].rename(columns={"candidate_security_id": "security_id"})
        if bool(identity["ticker"].duplicated().any()):
            raise DataReadinessError("prospective poll identity contains duplicate tickers")
        part = part.merge(identity, on="ticker", how="left", validate="one_to_one")
        if bool(part["security_id"].isna().any()):
            raise DataReadinessError("prospective source coverage lacks exact poll identity")
        part["poll_observed_at_utc"] = pd.Timestamp(poll.manifest["observed_at_utc"])
        part["poll_authority_sha256"] = file_sha256(poll.directory / "_authority.json")
        part["generation_directory"] = str(generation_by_poll[poll.directory])
        coverage_parts.append(part)
        assert_memory_budget(
            hard_budget_gib=memory_hard_budget_gib,
            headroom_gib=memory_headroom_gib,
            stage="prospective analyst horizon coverage assembly",
        )
    coverage = pd.concat(coverage_parts, ignore_index=True).sort_values(
        ["scheduled_poll_at_utc", "ticker"], kind="stable"
    ).reset_index(drop=True)
    coverage = coverage.loc[:, list(_COVERAGE_COLUMNS)]
    for column in (
        "requested_start_utc",
        "requested_end_utc",
        "started_at_utc",
        "completed_at_utc",
        "scheduled_poll_at_utc",
        "previous_poll_at_utc",
        "poll_observed_at_utc",
    ):
        coverage[column] = pd.Series(
            pd.to_datetime(coverage[column], utc=True),
            dtype="datetime64[us, UTC]",
        )
    observation_dates = pd.to_datetime(coverage["scheduled_poll_at_utc"], utc=True).dt.date.nunique()
    securities = int(episodes["security_id"].nunique()) if not episodes.empty else 0
    episode_count = len(episodes)
    source_ready = (
        episode_count >= int(policy.minimum_unique_event_episodes)
        and securities >= int(policy.minimum_securities)
    )
    capacity = pd.DataFrame(
        [
            {
                "generation_count": len(inputs.generations),
                "poll_count": len(inputs.polls),
                "observation_date_count": observation_dates,
                "retained_revision_count": len(classified_revisions),
                "classified_analyst_revision_count": int(classified_revisions["classified_analyst_revision"].sum()),
                "unique_analyst_episode_count": episode_count,
                "eligible_security_count": securities,
                "minimum_unique_event_episodes": int(policy.minimum_unique_event_episodes),
                "minimum_securities": int(policy.minimum_securities),
                "minimum_fit_sessions": int(policy.minimum_fit_sessions),
                "source_capacity_status": "ready_for_matched_preflight" if source_ready else "blocked",
                "matched_decision_capacity_evaluated": False,
                "training_eligible": False,
                "serving_eligible": False,
                "future_holdout_opened": False,
            }
        ],
        columns=_CAPACITY_COLUMNS,
    )
    frames = {
        "classified_revisions": classified_revisions,
        "episodes": episodes,
        "coverage": coverage,
        "capacity_audit": capacity,
    }
    _require_output_schemas(frames)
    assert_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective analyst horizon derived frames",
    )
    return frames


def _merge_revisions(generations: Sequence[ProspectiveGeneration]) -> pd.DataFrame:
    data = pd.concat([item.revisions for item in generations], ignore_index=True)
    if data.empty:
        return data.reset_index(drop=True)
    update_key = data["provider_updated_at_utc"].fillna("").astype(str)
    timestamp_collision = (
        data.assign(_provider_update_key=update_key)
        .groupby(
            ["provider_event_id", "ticker", "_provider_update_key"],
            sort=False,
        )["raw_sha256"]
        .transform("nunique")
        .gt(1)
    )
    data["provider_timestamp_anomaly"] = (
        data["provider_timestamp_anomaly"].astype(bool) | timestamp_collision
    )
    invariant = [
        "provider_event_id",
        "ticker",
        "published_at_utc",
        "provider_updated_at_utc",
        "source",
        "title",
        "url",
        "summary",
        "text",
        "raw_sha256",
    ]
    event_first = data.groupby(
        ["provider_event_id", "ticker", "candidate_security_id"],
        sort=False,
    )["event_first_seen_at_utc"].min()
    rows: list[dict[str, object]] = []
    for (_, _), group in data.groupby(["revision_id", "ticker"], sort=True):
        for column in invariant:
            if group[column].fillna("").astype(str).nunique(dropna=False) != 1:
                raise DataReadinessError(f"prospective revision changed invariant field {column}")
        first = group.sort_values("revision_first_seen_at_utc", kind="stable").iloc[0].to_dict()
        first_security_id = str(first["candidate_security_id"])
        security_ids = sorted(value for value in set(group["candidate_security_id"].astype(str)) if value)
        asset_ids = sorted(value for value in set(group["alpaca_asset_id"].astype(str)) if value)
        eligible = group[group["identity_eligible"].astype(bool)]
        identity_conflict = len(security_ids) > 1 or len(asset_ids) > 1
        identity_complete = len(security_ids) == 1 and len(asset_ids) == 1
        first["candidate_security_id"] = security_ids[0] if len(security_ids) == 1 else ""
        first["alpaca_asset_id"] = asset_ids[0] if len(asset_ids) == 1 else ""
        first["identity_eligible"] = (
            not eligible.empty and identity_complete and not identity_conflict
        )
        first["identity_ineligible_reason"] = (
            ""
            if first["identity_eligible"]
            else "identity_never_eligible"
            if eligible.empty
            else "identity_changed_across_generations"
            if identity_conflict
            else "identity_incomplete_across_generations"
        )
        first_seen = pd.to_datetime(group["revision_first_seen_at_utc"], utc=True).min()
        identity_first = pd.to_datetime(eligible["identity_first_eligible_at_utc"], utc=True, errors="coerce").min()
        first["revision_first_seen_at_utc"] = first_seen
        first["event_first_seen_at_utc"] = event_first.loc[
            (
                str(first["provider_event_id"]),
                str(first["ticker"]),
                first_security_id,
            )
        ]
        first["last_seen_at_utc"] = pd.to_datetime(group["last_seen_at_utc"], utc=True).max()
        first["identity_first_eligible_at_utc"] = identity_first
        first["production_available_at_utc"] = max(first_seen, identity_first) if first["identity_eligible"] else pd.NaT
        first["observation_count"] = int(group["observation_count"].sum())
        first["provider_timestamp_anomaly"] = bool(group["provider_timestamp_anomaly"].astype(bool).any())
        rows.append(first)
    return pd.DataFrame(rows, columns=data.columns).sort_values(
        ["revision_first_seen_at_utc", "provider_event_id", "ticker"], kind="stable"
    ).reset_index(drop=True)


def _episodes(classified: pd.DataFrame) -> pd.DataFrame:
    admitted = classified[
        classified["identity_eligible"].astype(bool)
        & classified["classified_analyst_revision"].astype(bool)
    ]
    rows: list[dict[str, object]] = []
    for (provider_event_id, security_id), group in admitted.groupby(
        ["provider_event_id", "security_id"], sort=True
    ):
        first = group.sort_values("production_available_at_utc", kind="stable").iloc[0]
        source_episode_id = hashlib.sha256(
            f"alpaca|{provider_event_id}|{security_id}".encode()
        ).hexdigest()
        rows.append(
            {
                "source_episode_id": source_episode_id,
                "family_event_id": hashlib.sha256(
                    f"{source_episode_id}|analyst_revision|{EVENT_FAMILY_POLICY_SHA256}".encode()
                ).hexdigest(),
                "provider_event_id": str(provider_event_id),
                "ticker": str(first["ticker"]),
                "security_id": str(security_id),
                "first_qualifying_revision_event_id": str(first["revision_event_id"]),
                "feature_available_at_utc": first["production_available_at_utc"],
                "title": str(first["title"]),
                "classification_rule_id": str(first["classification_rule_id"]),
                "classification_basis": str(first["classification_basis"]),
                "revision_count": len(group),
                "event_family_policy_sha256": EVENT_FAMILY_POLICY_SHA256,
            }
        )
    result = pd.DataFrame(rows, columns=_EPISODE_COLUMNS).sort_values(
        ["feature_available_at_utc", "provider_event_id", "security_id"], kind="stable"
    ).reset_index(drop=True)
    for column in (
        "source_episode_id",
        "family_event_id",
        "provider_event_id",
        "ticker",
        "security_id",
        "first_qualifying_revision_event_id",
        "title",
        "classification_rule_id",
        "classification_basis",
        "event_family_policy_sha256",
    ):
        result[column] = result[column].astype("string")
    result["feature_available_at_utc"] = pd.Series(
        pd.to_datetime(result["feature_available_at_utc"], utc=True),
        dtype="datetime64[us, UTC]",
    )
    result["revision_count"] = result["revision_count"].astype("int64")
    return result


def _normalize_classified_revisions(frame: pd.DataFrame) -> pd.DataFrame:
    timestamp_columns = (
        "published_at_utc",
        "provider_updated_at_utc",
        "revision_first_seen_at_utc",
        "event_first_seen_at_utc",
        "production_available_at_utc",
        "issuer_company_available_at_utc",
    )
    for column in timestamp_columns:
        frame[column] = pd.Series(
            pd.to_datetime(frame[column], utc=True),
            dtype="datetime64[us, UTC]",
        )
    string_columns = set(frame.columns).difference(
        {*timestamp_columns, "identity_eligible", "classified_analyst_revision"}
    )
    for column in sorted(string_columns):
        frame[column] = frame[column].astype("string")
    frame["identity_eligible"] = frame["identity_eligible"].astype(bool)
    frame["classified_analyst_revision"] = frame[
        "classified_analyst_revision"
    ].astype(bool)
    return frame


def _causal_company_map(
    poll: ProspectivePoll,
) -> dict[tuple[str, str], tuple[str, object]]:
    membership_root = path_integrity.verify_tree_containment(
        Path(_required_string(poll.request, "membership_authority_directory")),
        label="prospective analyst horizon membership authority",
    )
    authority = _json_object(membership_root / "_authority.json")
    if authority.get("schema") != OBSERVED_MEMBERSHIP_AUTHORITY_SCHEMA:
        return {}
    observed = load_observed_sp500_membership_authority(membership_root)
    observed_at = pd.Timestamp(observed.manifest["observed_at_utc"])
    if observed_at > pd.Timestamp(poll.manifest["observed_at_utc"]):
        raise DataReadinessError("issuer-company anchor was observed after the poll")
    anchor = pd.read_csv(membership_root / "current_anchor.csv", dtype=str, keep_default_na=False)
    if not {"ticker", "company", "cik"}.issubset(anchor.columns):
        raise DataReadinessError("observed membership issuer anchor is incomplete")
    return {
        (str(row["ticker"]).upper(), f"cik:{str(row['cik']).zfill(10)}"): (
            str(row["company"]),
            observed_at,
        )
        for row in anchor.to_dict("records")
        if str(row["company"]).strip()
    }


def _validate_memory_policy(hard_budget_gib: float, headroom_gib: float) -> None:
    if (
        isinstance(hard_budget_gib, bool)
        or isinstance(headroom_gib, bool)
        or not isinstance(hard_budget_gib, (int, float))
        or not isinstance(headroom_gib, (int, float))
        or not math.isfinite(float(hard_budget_gib))
        or not math.isfinite(float(headroom_gib))
        or float(hard_budget_gib) > 4.0
        or float(headroom_gib) <= 0.0
        or float(hard_budget_gib) <= float(headroom_gib)
    ):
        raise ValueError(
            "prospective horizon memory policy must be finite and fit the 4 GiB hard limit"
        )


def _required_string(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DataReadinessError(f"{key} is missing or invalid")
    return value


def _required_sha256(record: Mapping[str, object], key: str) -> str:
    value = _required_string(record, key)
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise DataReadinessError(f"{key} is not a lowercase SHA-256 digest")
    return value


def _required_int(
    record: Mapping[str, object],
    key: str,
    *,
    minimum: int | None = None,
) -> int:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataReadinessError(f"{key} is missing or invalid")
    if minimum is not None and value < minimum:
        raise DataReadinessError(f"{key} is below its permitted minimum")
    return value


def _required_number(record: Mapping[str, object], key: str) -> float:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataReadinessError(f"{key} is missing or invalid")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise DataReadinessError(f"{key} is not finite")
    return numeric


def _required_existing_file(
    record: Mapping[str, object],
    key: str,
    *,
    label: str,
) -> Path:
    path = Path(_required_string(record, key))
    absolute = path_integrity.verify_no_reparse_ancestry(path, label=label)
    try:
        resolved = absolute.resolve(strict=True)
    except OSError as exc:
        raise DataReadinessError(f"{label} is unavailable") from exc
    if not resolved.is_file():
        raise DataReadinessError(f"{label} is not a file")
    return resolved


def _validate_request(
    request: Mapping[str, object],
) -> tuple[list[Mapping[str, object]], float, float]:
    if set(request) != _REQUEST_KEYS:
        raise DataReadinessError("prospective analyst horizon request schema differs")
    payload = {
        str(key): value
        for key, value in request.items()
        if key != "request_sha256"
    }
    records = request.get("generations")
    if (
        request.get("schema") != REQUEST_SCHEMA
        or _required_sha256(request, "request_sha256") != json_sha256(payload)
        or not isinstance(records, list)
        or not records
        or request.get("generation_inventory_sha256") != json_sha256(records)
        or request.get("event_family_policy_version") != EVENT_FAMILY_POLICY_VERSION
        or request.get("event_family_policy_sha256") != EVENT_FAMILY_POLICY_SHA256
        or request.get("episode_identity")
        != "sha256(alpaca|provider_event_id|security_id)"
        or request.get("availability_policy")
        != "earliest_observed_provider_response"
    ):
        raise DataReadinessError("prospective analyst horizon request does not verify")
    _required_sha256(request, "generation_inventory_sha256")
    _required_sha256(request, "flattened_poll_inventory_sha256")
    _required_sha256(request, "security_identity_namespace_sha256")
    _required_sha256(request, "preflight_policy_sha256")
    _required_string(request, "registry_directory")
    _required_string(request, "preflight_policy_path")
    if not isinstance(request.get("preflight_policy"), Mapping):
        raise DataReadinessError("prospective analyst horizon policy record is malformed")
    validated: list[Mapping[str, object]] = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != _GENERATION_RECORD_KEYS:
            raise DataReadinessError(
                "prospective analyst horizon generation record is malformed"
            )
        _required_string(record, "directory")
        for key in (
            "request_sha256",
            "manifest_sha256",
            "authority_sha256",
            "poll_inventory_sha256",
        ):
            _required_sha256(record, key)
        validated.append(record)
    hard_budget_gib = _required_number(request, "memory_hard_budget_gib")
    headroom_gib = _required_number(request, "memory_headroom_gib")
    try:
        _validate_memory_policy(hard_budget_gib, headroom_gib)
    except ValueError as exc:
        raise DataReadinessError("prospective horizon memory policy is invalid") from exc
    return validated, hard_budget_gib, headroom_gib


def _validate_memory_record(
    raw: object,
    *,
    hard_budget_gib: float,
    headroom_gib: float,
) -> None:
    if not isinstance(raw, Mapping) or set(raw) != _MEMORY_KEYS:
        raise DataReadinessError("prospective analyst horizon memory record is malformed")
    values = {key: _required_number(raw, key) for key in _MEMORY_KEYS}
    if (
        any(value < 0.0 for value in values.values())
        or not math.isclose(values["hard_budget_gib"], hard_budget_gib)
        or not math.isclose(
            values["safety_threshold_gib"],
            hard_budget_gib - headroom_gib,
        )
        or values["peak_working_set_gib"] < values["current_working_set_gib"]
        or values["current_working_set_gib"] > values["safety_threshold_gib"]
        or values["peak_working_set_gib"] > values["safety_threshold_gib"]
    ):
        raise DataReadinessError("prospective analyst horizon memory record does not verify")


def _validate_child_manifest(
    manifest: Mapping[str, object],
    *,
    role: str,
    artifact_type: str,
    artifact_path: Path,
    child_inputs: Mapping[str, str],
) -> None:
    if set(manifest) != _CANONICAL_MANIFEST_KEYS:
        raise DataReadinessError(
            f"prospective analyst horizon {role} child schema differs"
        )
    rows = _required_int(manifest, "rows", minimum=0)
    columns = manifest.get("columns")
    audits = manifest.get("audit")
    if (
        not isinstance(columns, list)
        or tuple(columns) != _OUTPUT_COLUMNS[role]
        or manifest.get("artifact_type") != artifact_type
        or manifest.get("artifact_path") != str(artifact_path)
        or manifest.get("inputs") != child_inputs
        or manifest.get("production_ready") is not False
    ):
        raise DataReadinessError(
            f"prospective analyst horizon {role} child values differ"
        )
    _required_string(manifest, "schema")
    _required_string(manifest, "canonical_schema_version")
    _required_sha256(manifest, "artifact_sha256")
    created = _required_utc_timestamp(manifest, "created_at_utc", allow_none=False)
    if created is None:
        raise DataReadinessError(
            f"prospective analyst horizon {role} creation time is missing"
        )
    first = _required_utc_timestamp(
        manifest,
        "first_available_at_utc",
        allow_none=True,
    )
    last = _required_utc_timestamp(
        manifest,
        "last_available_at_utc",
        allow_none=True,
    )
    if (first is None) != (last is None) or (
        first is not None and last is not None and first > last
    ):
        raise DataReadinessError(
            f"prospective analyst horizon {role} availability range is invalid"
        )
    if not isinstance(audits, list) or len(audits) != 1:
        raise DataReadinessError(
            f"prospective analyst horizon {role} child audit is malformed"
        )
    audit = audits[0]
    if (
        not isinstance(audit, Mapping)
        or set(audit) != _CANONICAL_AUDIT_KEYS
        or _required_string(audit, "name") != role
        or _required_string(audit, "status") != "pass"
        or _required_int(audit, "failures", minimum=0) != 0
        or _required_int(audit, "rows_checked", minimum=0) != rows
    ):
        raise DataReadinessError(
            f"prospective analyst horizon {role} child audit does not verify"
        )
    _required_string(audit, "detail")


def _required_utc_timestamp(
    record: Mapping[str, object],
    key: str,
    *,
    allow_none: bool,
) -> pd.Timestamp | None:
    value = record.get(key)
    if value is None and allow_none:
        return None
    if not isinstance(value, str) or not value.strip():
        raise DataReadinessError(f"{key} is missing or invalid")
    try:
        parsed = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError(f"{key} is not a valid timestamp") from exc
    if parsed.tzinfo is None:
        raise DataReadinessError(f"{key} must be timezone-aware")
    return parsed.tz_convert("UTC")


def _assert_derivation_capacity(
    inputs: _Inputs,
    *,
    memory_hard_budget_gib: float,
    memory_headroom_gib: float,
) -> None:
    frames = [generation.revisions for generation in inputs.generations]
    for poll in inputs.polls:
        frames.extend(
            (poll.observations, poll.source_collections, poll.identity_audit)
        )
    expanded_input_bytes = sum(
        int(frame.memory_usage(index=True, deep=True).sum()) for frame in frames
    )
    audit = memory_audit(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
    )
    if audit.current_working_set_gib is None:
        raise DataReadinessError(
            "prospective analyst horizon cannot account for derivation memory"
        )
    safety_threshold_bytes = int(audit.safety_threshold_gib * 1024**3)
    current_bytes = int(audit.current_working_set_gib * 1024**3)
    projected_bytes = current_bytes + expanded_input_bytes * DERIVATION_EXPANSION_FACTOR
    if projected_bytes > safety_threshold_bytes:
        raise DataReadinessError(
            "prospective analyst horizon projected derivation memory exceeds "
            "the configured safety threshold"
        )


def _directory_file_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += path.stat().st_size
            if total > MAX_PARENT_INPUT_BYTES:
                break
    return total


def _require_output_schemas(frames: Mapping[str, pd.DataFrame]) -> None:
    for role, columns in _OUTPUT_COLUMNS.items():
        frame = frames.get(role)
        if frame is None or tuple(frame.columns) != columns:
            raise DataReadinessError(
                f"prospective analyst horizon {role} schema differs"
            )


def _remove_owned_staging(
    staging: Path,
    *,
    output: Path,
    owner_path: Path,
) -> None:
    absolute = path_integrity.verify_no_reparse_ancestry(
        staging,
        label="prospective analyst horizon staging",
    )
    owner_absolute = path_integrity.verify_no_reparse_ancestry(
        owner_path,
        label="prospective analyst horizon staging owner",
    )
    if not absolute.exists() and not owner_absolute.exists():
        return
    if not owner_absolute.is_file():
        raise DataReadinessError(
            "prospective analyst horizon staging is not owned by this publisher"
        )
    owner = _json_object(owner_absolute)
    if owner != {
        "schema": STAGING_OWNER_SCHEMA,
        "staging_directory": str(staging),
        "output_directory": str(output),
    }:
        raise DataReadinessError(
            "prospective analyst horizon staging owner does not verify"
        )
    if not absolute.exists():
        owner_absolute.unlink()
        return
    verified = path_integrity.verify_tree_containment(
        absolute,
        label="prospective analyst horizon staging",
    )
    shutil.rmtree(verified)
    owner_absolute.unlink()


def _retarget_canonical_manifest(path: Path, *, published_path: Path) -> None:
    sidecar = manifest_path_for(path)
    manifest = _json_object(sidecar)
    if manifest.get("artifact_path") != str(path):
        raise DataReadinessError(
            "prospective analyst horizon canonical artifact path does not verify"
        )
    _atomic_json(sidecar, {**manifest, "artifact_path": str(published_path)})


def _artifact_record(path: Path) -> dict[str, object]:
    return {"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}


def _passing_audit(name: str, rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name=name,
                status="pass",
                failures=0,
                rows_checked=rows,
                detail=f"{rows} rows replayed",
            ),
        )
    )


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=True,
            check_like=False,
        )
    except AssertionError:
        return False
    return True


def _json_object(path: Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_non_finite(value: str) -> object:
        raise ValueError(f"non-finite JSON value: {value}")

    def parse_finite_float(value: str) -> float:
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError(f"non-finite JSON float: {value}")
        return parsed

    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=reject_duplicates,
            parse_constant=reject_non_finite,
            parse_float=parse_finite_float,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DataReadinessError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"expected JSON object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise DataReadinessError(f"immutable JSON artifact already exists: {path}")
    _atomic_json(path, payload)
