"""Anchor-bound point-in-time S&P 500 membership reconstruction."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, cast
from zoneinfo import ZoneInfo

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditReport, audit_universe_memberships
from market_predictor.canonical.normalize import canonicalize_universe_memberships
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.symbols import normalized_ticker
from market_predictor.locking import LockTimeout, file_lock
from market_predictor.resources import assert_memory_budget, assert_peak_memory_budget
from market_predictor.sources.spglobal.archive import MAXIMUM_MEMORY_GIB, MEMORY_HEADROOM_GIB
from market_predictor.universe.membership_identity_validation import (
    validate_security_exclusion_share,
)
from market_predictor.universe.sp500.index_change_events import (
    require_spglobal_event_reconstruction_ready,
)
from market_predictor.universe.sp500.membership_history import (
    SECTOR_BENCHMARKS,
    IndexChange,
)
from market_predictor.universe.sp500.transition_authority import (
    require_sp500_transition_authority,
)

MEMBERSHIP_REQUEST_SCHEMA: Final = "edge_rebuild.sp500_membership_request.v1"
MEMBERSHIP_MANIFEST_SCHEMA: Final = "edge_rebuild.sp500_membership_manifest.v1"
MEMBERSHIP_AUTHORITY_SCHEMA: Final = "edge_rebuild.sp500_membership_authority.v1"
MEMBERSHIP_RECONSTRUCTION_SCHEMA: Final = "edge_rebuild.sp500_membership_reconstruction.v1"
MEMBERSHIP_FILE: Final = "memberships.parquet"
EXCLUSION_FILE: Final = "security_exclusions.json"
MAXIMUM_SECURITY_EXCLUSION_FRACTION: Final = 0.05
_BENCHMARK_TICKERS: Final = frozenset({"SPY", "QQQ", *SECTOR_BENCHMARKS.values()})


@dataclass(frozen=True)
class _State:
    security_id: str
    company: str
    sector: str
    industry: str
    effective_to_utc: datetime | None
    evidence_urls: tuple[str, ...]


def publish_sp500_membership_authority(
    *,
    archive_directory: Path,
    event_directory: Path,
    transition_directory: Path,
    reviewed_transitions_path: Path,
    anchor_path: Path,
    start_date: date,
    cutoff_date: date,
    output_directory: Path,
    base_membership_directory: Path | None = None,
    security_exclusions_path: Path | None = None,
    maximum_security_exclusion_fraction: float = MAXIMUM_SECURITY_EXCLUSION_FRACTION,
) -> dict[str, Any]:
    """Publish canonical memberships whose complete lineage replays offline."""

    _validate_parameters(
        start_date=start_date,
        cutoff_date=cutoff_date,
        maximum_security_exclusion_fraction=maximum_security_exclusion_fraction,
    )
    output = output_directory.resolve()
    for parent in (
        archive_directory.resolve(),
        event_directory.resolve(),
        transition_directory.resolve(),
        *(() if base_membership_directory is None else (base_membership_directory.resolve(),)),
    ):
        if output == parent or output in parent.parents or parent in output.parents:
            raise DataReadinessError("membership output and parent directories must be disjoint")
    output_directory.mkdir(parents=True, exist_ok=True)
    try:
        with file_lock(output_directory / "_publisher", timeout=0.0):
            return _publish_locked(
                archive_directory=archive_directory,
                event_directory=event_directory,
                transition_directory=transition_directory,
                reviewed_transitions_path=reviewed_transitions_path,
                anchor_path=anchor_path,
                start_date=start_date,
                cutoff_date=cutoff_date,
                output_directory=output_directory,
                base_membership_directory=base_membership_directory,
                security_exclusions_path=security_exclusions_path,
                maximum_security_exclusion_fraction=maximum_security_exclusion_fraction,
            )
    except LockTimeout as exc:
        raise DataReadinessError(f"another process is publishing S&P memberships {output_directory}") from exc


def require_sp500_membership_authority(
    membership_directory: Path,
    *,
    archive_directory: Path,
    event_directory: Path,
    transition_directory: Path,
    reviewed_transitions_path: Path,
    anchor_path: Path,
    start_date: date,
    cutoff_date: date,
    base_membership_directory: Path | None = None,
    security_exclusions_path: Path | None = None,
    maximum_security_exclusion_fraction: float = MAXIMUM_SECURITY_EXCLUSION_FRACTION,
) -> pd.DataFrame:
    """Verify all parents, artifact hashes, exclusions, and semantic replay."""

    _validate_parameters(
        start_date=start_date,
        cutoff_date=cutoff_date,
        maximum_security_exclusion_fraction=maximum_security_exclusion_fraction,
    )
    base_memberships, extension_parent = _load_extension_parent(
        base_membership_directory,
        start_date=start_date,
        cutoff_date=cutoff_date,
    )
    transitions = require_sp500_transition_authority(
        transition_directory,
        archive_directory=archive_directory,
        event_directory=event_directory,
        reviewed_transitions_path=reviewed_transitions_path,
        start_date=start_date,
        cutoff_date=cutoff_date,
    )
    verified_events = require_spglobal_event_reconstruction_ready(
        event_directory,
        archive_directory=archive_directory,
    )
    anchor, anchor_semantic_sha256 = _load_anchor(anchor_path)
    parent = _parent_lineage(
        archive_directory=archive_directory,
        event_directory=event_directory,
        transition_directory=transition_directory,
        anchor_path=anchor_path,
        anchor_semantic_sha256=anchor_semantic_sha256,
    )
    request_payload = _request_payload(
        parent=parent,
        start_date=start_date,
        cutoff_date=cutoff_date,
        security_exclusions_path=security_exclusions_path,
        maximum_security_exclusion_fraction=maximum_security_exclusion_fraction,
        extension_parent=extension_parent,
    )
    request = _load_object(membership_directory / "_request.json")
    request_sha256 = _json_sha256(request_payload)
    if request != {**request_payload, "request_sha256": request_sha256}:
        raise DataReadinessError("S&P membership request identity is invalid")
    authority = _load_object(membership_directory / "_authority.json")
    manifest_path = _resolve_inside(
        membership_directory,
        str(authority.get("artifact", "")),
    )
    if (
        authority.get("schema") != MEMBERSHIP_AUTHORITY_SCHEMA
        or authority.get("state") != "membership_complete"
        or not manifest_path.is_file()
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
    ):
        raise DataReadinessError("S&P membership authority is invalid")
    manifest = _load_object(manifest_path)
    if (
        manifest.get("schema") != MEMBERSHIP_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or authority.get("request_sha256") != request_sha256
        or manifest.get("parent_lineage") != parent
        or authority.get("parent_lineage") != parent
        or int(manifest.get("benchmark_session_exclusions", -1)) != 0
    ):
        raise DataReadinessError("S&P membership lineage or readiness is invalid")
    membership_record = manifest.get("membership_artifact")
    exclusion_record = manifest.get("exclusion_artifact")
    if not isinstance(membership_record, dict) or not isinstance(exclusion_record, dict):
        raise DataReadinessError("S&P membership artifact inventory is invalid")
    membership_path = _verified_artifact(membership_directory, membership_record)
    exclusion_path = _verified_artifact(membership_directory, exclusion_record)
    membership_sidecar = manifest_path_for(membership_path)
    if not membership_sidecar.is_file() or manifest.get("membership_manifest_sha256") != file_sha256(membership_sidecar):
        raise DataReadinessError("canonical S&P membership manifest hash is invalid")
    actual, _ = load_canonical_artifact(
        membership_path,
        expected_type="memberships",
        allow_research=True,
    )
    expected, exclusions, reconstruction = _build_memberships(
        anchor=anchor,
        changes=list(verified_events.changes),
        transitions=transitions,
        start_date=start_date,
        cutoff_date=cutoff_date,
        security_exclusions_path=security_exclusions_path,
        maximum_security_exclusion_fraction=maximum_security_exclusion_fraction,
        snapshot_id=str(manifest.get("universe_snapshot_id", "")),
        base_memberships=base_memberships,
        base_cutoff_date=(None if extension_parent is None else date.fromisoformat(str(extension_parent["cutoff_date"]))),
    )
    persisted_exclusions = _load_array(exclusion_path)
    if persisted_exclusions != exclusions:
        raise DataReadinessError("S&P security exclusions do not replay")
    if _membership_records(actual) != _membership_records(expected):
        raise DataReadinessError("S&P membership artifact does not replay from its parents")
    universe_sha256 = _membership_sha256(actual)
    if (
        manifest.get("universe_sha256") != universe_sha256
        or authority.get("universe_sha256") != universe_sha256
        or int(manifest.get("membership_intervals", -1)) != len(actual)
        or int(manifest.get("security_count", -1)) != actual["security_id"].nunique()
        or manifest.get("reconstruction") != reconstruction
        or manifest.get("extension_parent") != extension_parent
        or authority.get("extension_parent") != extension_parent
    ):
        raise DataReadinessError("S&P membership counts or semantic identity are invalid")
    return actual


def load_sp500_membership_authority_envelope(
    membership_directory: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and hash-verify a closed S&P membership authority without its source paths."""

    root = membership_directory.resolve()
    request_path = root / "_request.json"
    manifest_path = root / "_manifest.json"
    authority_path = root / "_authority.json"
    request = _load_object(request_path)
    manifest = _load_object(manifest_path)
    authority = _load_object(authority_path)
    request_payload = {str(key): value for key, value in request.items() if key != "request_sha256"}
    request_sha256 = _json_sha256(request_payload)
    parent = request.get("parent_lineage")
    extension_parent = request.get("extension_parent")
    if (
        request.get("schema") != MEMBERSHIP_REQUEST_SCHEMA
        or request.get("request_sha256") != request_sha256
        or not isinstance(parent, dict)
        or authority.get("schema") != MEMBERSHIP_AUTHORITY_SCHEMA
        or authority.get("state") != "membership_complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("request_sha256") != request_sha256
        or manifest.get("schema") != MEMBERSHIP_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or manifest.get("parent_lineage") != parent
        or authority.get("parent_lineage") != parent
        or manifest.get("extension_parent") != extension_parent
        or authority.get("extension_parent") != extension_parent
        or manifest.get("start_date") != request.get("start_date")
        or manifest.get("cutoff_date") != request.get("cutoff_date")
    ):
        raise DataReadinessError("S&P membership authority envelope is invalid")
    record = manifest.get("membership_artifact")
    if not isinstance(record, dict):
        raise DataReadinessError("S&P membership artifact inventory is missing")
    membership_path = _verified_artifact(root, record)
    sidecar = manifest_path_for(membership_path)
    if not sidecar.is_file() or manifest.get("membership_manifest_sha256") != file_sha256(sidecar):
        raise DataReadinessError("S&P membership canonical manifest is invalid")
    memberships, _ = load_canonical_artifact(
        membership_path,
        expected_type="memberships",
        allow_research=True,
    )
    universe_sha256 = _membership_sha256(memberships)
    if (
        manifest.get("universe_sha256") != universe_sha256
        or authority.get("universe_sha256") != universe_sha256
        or int(manifest.get("membership_intervals", -1)) != len(memberships)
        or int(authority.get("membership_intervals", -1)) != len(memberships)
        or int(manifest.get("security_count", -1)) != memberships["security_id"].nunique()
        or int(authority.get("security_count", -1)) != memberships["security_id"].nunique()
        or int(manifest.get("ticker_count", -1)) != memberships["ticker"].nunique()
    ):
        raise DataReadinessError("S&P membership authority semantics are invalid")
    return memberships, {
        "authority_type": "closed_archive",
        "authority_sha256": file_sha256(authority_path),
        "manifest_sha256": file_sha256(manifest_path),
        "membership_table_sha256": file_sha256(membership_path),
        "universe_sha256": universe_sha256,
        "cutoff_date": str(manifest.get("cutoff_date", "")),
        "observed_at_utc": None,
    }


def _publish_locked(
    *,
    archive_directory: Path,
    event_directory: Path,
    transition_directory: Path,
    reviewed_transitions_path: Path,
    anchor_path: Path,
    start_date: date,
    cutoff_date: date,
    output_directory: Path,
    base_membership_directory: Path | None,
    security_exclusions_path: Path | None,
    maximum_security_exclusion_fraction: float,
) -> dict[str, Any]:
    assert_memory_budget(
        hard_budget_gib=MAXIMUM_MEMORY_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage="S&P membership publication start",
    )
    if (output_directory / "_authority.json").exists():
        raise DataReadinessError("completed S&P membership authority is immutable")
    base_memberships, extension_parent = _load_extension_parent(
        base_membership_directory,
        start_date=start_date,
        cutoff_date=cutoff_date,
    )
    transitions = require_sp500_transition_authority(
        transition_directory,
        archive_directory=archive_directory,
        event_directory=event_directory,
        reviewed_transitions_path=reviewed_transitions_path,
        start_date=start_date,
        cutoff_date=cutoff_date,
    )
    verified_events = require_spglobal_event_reconstruction_ready(
        event_directory,
        archive_directory=archive_directory,
    )
    anchor, anchor_semantic_sha256 = _load_anchor(anchor_path)
    parent = _parent_lineage(
        archive_directory=archive_directory,
        event_directory=event_directory,
        transition_directory=transition_directory,
        anchor_path=anchor_path,
        anchor_semantic_sha256=anchor_semantic_sha256,
    )
    request_payload = _request_payload(
        parent=parent,
        start_date=start_date,
        cutoff_date=cutoff_date,
        security_exclusions_path=security_exclusions_path,
        maximum_security_exclusion_fraction=maximum_security_exclusion_fraction,
        extension_parent=extension_parent,
    )
    request_sha256 = _json_sha256(request_payload)
    _write_json_atomic(
        output_directory / "_request.json",
        {**request_payload, "request_sha256": request_sha256},
    )
    snapshot_id = f"sp500-pit-{request_sha256[:20]}"
    memberships, exclusions, reconstruction = _build_memberships(
        anchor=anchor,
        changes=list(verified_events.changes),
        transitions=transitions,
        start_date=start_date,
        cutoff_date=cutoff_date,
        security_exclusions_path=security_exclusions_path,
        maximum_security_exclusion_fraction=maximum_security_exclusion_fraction,
        snapshot_id=snapshot_id,
        base_memberships=base_memberships,
        base_cutoff_date=(None if extension_parent is None else date.fromisoformat(str(extension_parent["cutoff_date"]))),
    )
    membership_path = output_directory / MEMBERSHIP_FILE
    checks = audit_universe_memberships(memberships, require_observed=False)
    audit = CanonicalAuditReport(checks=checks)
    if not audit.passed:
        failures = [check.name for check in checks if check.status != "pass"]
        raise DataReadinessError(f"canonical S&P membership audit failed: {failures}")
    artifact_inputs: dict[str, Any] = {
        **parent,
        "request_sha256": request_sha256,
        "reconstruction_schema": MEMBERSHIP_RECONSTRUCTION_SCHEMA,
        "extension_parent": extension_parent,
    }
    write_canonical_artifact(
        memberships,
        membership_path,
        artifact_type="memberships",
        audit=audit,
        inputs=artifact_inputs,
        production_ready=False,
    )
    exclusion_path = output_directory / EXCLUSION_FILE
    _write_json_atomic(exclusion_path, exclusions)
    universe_sha256 = _membership_sha256(memberships)
    manifest: dict[str, Any] = {
        "schema": MEMBERSHIP_MANIFEST_SCHEMA,
        "status": "complete",
        "request_sha256": request_sha256,
        "parent_lineage": parent,
        "extension_parent": extension_parent,
        "start_date": start_date.isoformat(),
        "cutoff_date": cutoff_date.isoformat(),
        "universe_snapshot_id": snapshot_id,
        "membership_intervals": len(memberships),
        "security_count": int(memberships["security_id"].nunique()),
        "ticker_count": int(memberships["ticker"].nunique()),
        "excluded_security_count": len(exclusions),
        "excluded_security_fraction": reconstruction["excluded_security_fraction"],
        "maximum_security_exclusion_fraction": maximum_security_exclusion_fraction,
        "benchmark_session_exclusions": 0,
        "security_exclusion_scope": "whole_security_only",
        "universe_sha256": universe_sha256,
        "membership_artifact": _artifact_record(membership_path),
        "membership_manifest_sha256": file_sha256(manifest_path_for(membership_path)),
        "exclusion_artifact": _artifact_record(exclusion_path),
        "reconstruction": reconstruction,
    }
    manifest_path = output_directory / "_manifest.json"
    _write_json_atomic(manifest_path, manifest)
    _write_json_atomic(output_directory / "_status.json", manifest)
    authority = {
        "schema": MEMBERSHIP_AUTHORITY_SCHEMA,
        "state": "membership_complete",
        "artifact": "_manifest.json",
        "artifact_sha256": file_sha256(manifest_path),
        "request_sha256": request_sha256,
        "parent_lineage": parent,
        "extension_parent": extension_parent,
        "universe_sha256": universe_sha256,
        "membership_intervals": len(memberships),
        "security_count": int(memberships["security_id"].nunique()),
    }
    _write_json_atomic(output_directory / "_authority.json", authority)
    assert_peak_memory_budget(
        hard_budget_gib=MAXIMUM_MEMORY_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage="S&P membership publication",
    )
    require_sp500_membership_authority(
        output_directory,
        archive_directory=archive_directory,
        event_directory=event_directory,
        transition_directory=transition_directory,
        reviewed_transitions_path=reviewed_transitions_path,
        anchor_path=anchor_path,
        start_date=start_date,
        cutoff_date=cutoff_date,
        base_membership_directory=base_membership_directory,
        security_exclusions_path=security_exclusions_path,
        maximum_security_exclusion_fraction=maximum_security_exclusion_fraction,
    )
    return manifest


def _build_memberships(
    *,
    anchor: pd.DataFrame,
    changes: list[IndexChange],
    transitions: pd.DataFrame,
    start_date: date,
    cutoff_date: date,
    security_exclusions_path: Path | None,
    maximum_security_exclusion_fraction: float,
    snapshot_id: str,
    base_memberships: pd.DataFrame | None = None,
    base_cutoff_date: date | None = None,
) -> tuple[pd.DataFrame, list[dict[str, str]], dict[str, Any]]:
    if (base_memberships is None) != (base_cutoff_date is None):
        raise DataReadinessError("membership extension requires both base memberships and cutoff")
    relevant_changes = [change for change in changes if start_date <= _ny_date(change.effective_at_utc) <= cutoff_date]
    resolved = [_resolved_change(change, transitions) for change in relevant_changes]
    states = _anchor_states(
        anchor,
        base_memberships=base_memberships,
        base_cutoff_date=base_cutoff_date,
    )
    intervals: list[dict[str, Any]] = []
    events_by_time: dict[datetime, list[IndexChange]] = {}
    for change in resolved:
        events_by_time.setdefault(change.effective_at_utc, []).append(change)
    membership_continuity = transitions["membership_continuity"].astype(bool)
    transitions_by_time = {
        pd.Timestamp(moment).to_pydatetime(): group.reset_index(drop=True)
        for moment, group in transitions.loc[membership_continuity].groupby(
            "effective_at_utc",
            sort=True,
        )
    }
    automatic_exclusions: list[dict[str, str]] = []
    boundary_counts: list[int] = [len(states)]
    for effective_at in sorted(set(events_by_time).union(transitions_by_time), reverse=True):
        events = events_by_time.get(effective_at, [])
        _reverse_events(
            states=states,
            intervals=intervals,
            changes=events,
            effective_at=effective_at,
            snapshot_id=snapshot_id,
            automatic_exclusions=automatic_exclusions,
            base_memberships=base_memberships,
        )
        transition_group = transitions_by_time.get(effective_at)
        if transition_group is not None:
            _reverse_transition_batch(
                states=states,
                intervals=intervals,
                transitions=transition_group,
                effective_at=effective_at,
                snapshot_id=snapshot_id,
            )
        boundary_counts.append(len(states))
    start_at = _session_midnight(start_date)
    for ticker, state in states.items():
        _append_interval(
            intervals,
            ticker=ticker,
            state=state,
            effective_from=start_at,
            snapshot_id=snapshot_id,
        )
    raw = pd.DataFrame(intervals)
    if raw.empty:
        raise DataReadinessError("S&P reconstruction produced no membership intervals")
    canonical = canonicalize_universe_memberships(
        raw,
        source="spglobal_official_point_in_time",
        availability_policy="provider_publication_proxy",
    )
    if base_memberships is not None and base_cutoff_date is not None:
        canonical = _combine_base_prefix_with_extension(
            base_memberships,
            canonical,
            base_cutoff_date=base_cutoff_date,
            snapshot_id=snapshot_id,
        )
    canonical, exclusions = _apply_security_exclusions(
        canonical,
        automatic_exclusions=automatic_exclusions,
        security_exclusions_path=security_exclusions_path,
        maximum_security_exclusion_fraction=maximum_security_exclusion_fraction,
    )
    if base_memberships is not None and base_cutoff_date is not None:
        verify_membership_namespace_extension(
            base_memberships,
            canonical,
            base_cutoff_date=base_cutoff_date.isoformat(),
            current_cutoff_date=cutoff_date.isoformat(),
        )
    if int(min(boundary_counts)) < 450 or int(max(boundary_counts)) > 550:
        raise DataReadinessError(
            f"S&P reconstruction boundary counts fall outside 450..550: {min(boundary_counts)}..{max(boundary_counts)}"
        )
    raw_security_ids = set(raw["security_id"].astype(str))
    automatic_exclusion_ids = {item["security_id"] for item in automatic_exclusions}
    excluded_only_security_count = len(automatic_exclusion_ids.difference(raw_security_ids))
    source_security_count = int(raw["security_id"].nunique()) + excluded_only_security_count
    reconstruction = {
        "schema": MEMBERSHIP_RECONSTRUCTION_SCHEMA,
        "event_count": len(relevant_changes),
        "transition_count": len(transitions),
        "anchor_ticker_count": len(anchor),
        "minimum_boundary_members": int(min(boundary_counts)),
        "maximum_boundary_members": int(max(boundary_counts)),
        "membership_intervals_before_exclusions": len(raw),
        "security_count_before_exclusions": source_security_count,
        "excluded_security_count": len(exclusions),
        "excluded_security_fraction": round(
            len(exclusions) / source_security_count,
            8,
        ),
        "benchmark_session_exclusions": 0,
    }
    return canonical.reset_index(drop=True), exclusions, reconstruction


def _combine_base_prefix_with_extension(
    base: pd.DataFrame,
    current: pd.DataFrame,
    *,
    base_cutoff_date: date,
    snapshot_id: str,
) -> pd.DataFrame:
    if set(base.columns) != set(current.columns):
        raise DataReadinessError("S&P membership extension contract differs from its base authority")
    boundary = pd.Timestamp(base_cutoff_date, tz="UTC") + pd.Timedelta(days=1)
    prefix = base[base["effective_from_utc"].lt(boundary)].copy()
    crossing = prefix["effective_to_utc"].isna() | prefix["effective_to_utc"].gt(boundary)
    prefix.loc[crossing, "effective_to_utc"] = boundary

    suffix = current[current["effective_to_utc"].isna() | current["effective_to_utc"].gt(boundary)].copy()
    starts_before_boundary = suffix["effective_from_utc"].lt(boundary)
    suffix.loc[starts_before_boundary, "effective_from_utc"] = boundary
    suffix.loc[starts_before_boundary, "available_at_utc"] = boundary
    suffix = suffix[suffix["effective_to_utc"].isna() | suffix["effective_to_utc"].gt(suffix["effective_from_utc"])]

    combined = pd.concat([prefix, suffix], ignore_index=True)
    combined["universe_snapshot_id"] = snapshot_id
    return combined.sort_values(
        ["ticker", "effective_from_utc", "security_id"],
        kind="stable",
    ).reset_index(drop=True)


def _reverse_events(
    *,
    states: dict[str, _State],
    intervals: list[dict[str, Any]],
    changes: list[IndexChange],
    effective_at: datetime,
    snapshot_id: str,
    automatic_exclusions: list[dict[str, str]],
    base_memberships: pd.DataFrame | None,
) -> None:
    additions = [change for change in changes if change.action == "addition"]
    deletions = [change for change in changes if change.action == "deletion"]
    if len({change.ticker for change in additions}) != len(additions):
        raise DataReadinessError(f"duplicate S&P additions at {effective_at.isoformat()}")
    if len({change.ticker for change in deletions}) != len(deletions):
        raise DataReadinessError(f"duplicate S&P deletions at {effective_at.isoformat()}")
    for change in additions:
        state = states.get(change.ticker)
        if state is None:
            automatic_exclusions.append(
                {
                    "security_id": _historical_security_id(change),
                    "ticker": change.ticker,
                    "reason": "addition_absent_from_cutoff_anchor_replay",
                    "effective_at_utc": effective_at.isoformat(),
                }
            )
            continue
        _append_interval(
            intervals,
            ticker=change.ticker,
            state=state,
            effective_from=effective_at,
            snapshot_id=snapshot_id,
        )
        del states[change.ticker]
    for change in deletions:
        if change.ticker in states:
            state = states.pop(change.ticker)
            automatic_exclusions.append(
                {
                    "security_id": state.security_id,
                    "ticker": change.ticker,
                    "reason": "deletion_remains_in_cutoff_anchor_replay",
                    "effective_at_utc": effective_at.isoformat(),
                }
            )
            continue
        states[change.ticker] = _State(
            security_id=(
                _base_security_id_at(
                    base_memberships,
                    ticker=change.ticker,
                    effective_at=effective_at,
                )
                or _historical_security_id(change)
            ),
            company=change.company.strip() or change.ticker,
            sector=change.sector.strip() or "Unknown",
            industry="Unknown",
            effective_to_utc=effective_at,
            evidence_urls=tuple(sorted(source.source_url for source in change.source_evidence())),
        )


def _reverse_transition_batch(
    *,
    states: dict[str, _State],
    intervals: list[dict[str, Any]],
    transitions: pd.DataFrame,
    effective_at: datetime,
    snapshot_id: str,
) -> None:
    snapshot = dict(states)
    removals: set[str] = set()
    activations: dict[str, _State] = {}
    for record in transitions.to_dict(orient="records"):
        old_ticker = str(record["old_ticker"])
        new_ticker = str(record["new_ticker"])
        new_state = snapshot.get(new_ticker)
        old_state = snapshot.get(old_ticker)
        if new_state is None:
            if old_state is not None:
                raise DataReadinessError(f"transition post-state retains old ticker {old_ticker} at {effective_at.isoformat()}")
            continue
        if old_state is not None:
            raise DataReadinessError(f"transition has both {old_ticker} and {new_ticker} active at {effective_at.isoformat()}")
        _append_interval(
            intervals,
            ticker=new_ticker,
            state=new_state,
            effective_from=effective_at,
            snapshot_id=snapshot_id,
        )
        removals.add(new_ticker)
        identity_continuity = bool(record["identity_continuity"])
        explicit_new = str(record.get("new_security_id", "")).strip()
        explicit_old = str(record.get("old_security_id", "")).strip()
        if identity_continuity:
            if (
                explicit_new
                and new_state.security_id.startswith("cik:")
                and not _equivalent_security_id(new_state.security_id, explicit_new)
            ):
                raise DataReadinessError(f"transition {record['transition_id']} new security identity disagrees with anchor")
            security_id = explicit_old or explicit_new or new_state.security_id
        else:
            security_id = explicit_old or _transition_security_id(record, side="old")
        if old_ticker in activations:
            raise DataReadinessError(f"multiple transition activations target {old_ticker} at {effective_at.isoformat()}")
        activations[old_ticker] = _State(
            security_id=security_id,
            company=new_state.company,
            sector=new_state.sector,
            industry=new_state.industry,
            effective_to_utc=effective_at,
            evidence_urls=tuple(sorted({*new_state.evidence_urls, str(record["source_url"])})),
        )
    for ticker in removals:
        states.pop(ticker, None)
    for ticker, state in activations.items():
        if ticker in states:
            raise DataReadinessError(f"transition activation collides with active ticker {ticker} at {effective_at.isoformat()}")
        states[ticker] = state


def _append_interval(
    intervals: list[dict[str, Any]],
    *,
    ticker: str,
    state: _State,
    effective_from: datetime,
    snapshot_id: str,
) -> None:
    if state.effective_to_utc is not None and state.effective_to_utc <= effective_from:
        if state.effective_to_utc == effective_from:
            return
        raise DataReadinessError(f"invalid reconstructed interval for {ticker}")
    sector = state.sector or "Unknown"
    intervals.append(
        {
            "ticker": ticker,
            "security_id": state.security_id,
            "effective_from_utc": effective_from,
            "effective_to_utc": state.effective_to_utc,
            "available_at_utc": effective_from,
            "sector": sector,
            "industry": state.industry or "Unknown",
            "market_cap_bucket": "large_cap_sp500",
            "liquidity_bucket": "sp500_constituent",
            "primary_benchmark": SECTOR_BENCHMARKS.get(sector, "SPY"),
            "universe_snapshot_id": snapshot_id,
        }
    )


def _resolved_change(change: IndexChange, transitions: pd.DataFrame) -> IndexChange:
    current = change.ticker
    published = change.source_published_date
    eligible = transitions[
        transitions["identity_continuity"]
        & transitions["membership_continuity"]
        & transitions["effective_at_utc"].le(pd.Timestamp(change.effective_at_utc))
    ]
    for moment, group in eligible.groupby("effective_at_utc", sort=True):
        effective_date = _ny_date(pd.Timestamp(moment).to_pydatetime())
        if published >= effective_date:
            continue
        matches = group[group["old_ticker"].eq(current)]
        if len(matches) > 1:
            raise DataReadinessError(f"ambiguous event ticker transition for {current} at {moment}")
        if len(matches) == 1:
            # Exactly one hop per timestamp. Same-time chains are simultaneous,
            # not sequential; this is essential for the Fox temporary symbols.
            current = str(matches.iloc[0]["new_ticker"])
    if current == change.ticker:
        return change
    return IndexChange(
        effective_at_utc=change.effective_at_utc,
        action=change.action,
        ticker=current,
        company=change.company,
        sector=change.sector,
        source_url=change.source_url,
        source_published_date=change.source_published_date,
        source_sha256=change.source_sha256,
        supporting_sources=change.supporting_sources,
    )


def _anchor_states(
    anchor: pd.DataFrame,
    *,
    base_memberships: pd.DataFrame | None,
    base_cutoff_date: date | None,
) -> dict[str, _State]:
    cik_counts = anchor["cik"].value_counts()
    base_by_ticker, base_by_cik = _base_active_identity_maps(
        base_memberships,
        base_cutoff_date=base_cutoff_date,
    )
    states: dict[str, _State] = {}
    for record in anchor.to_dict(orient="records"):
        ticker = str(record["ticker"])
        cik = str(record["cik"])
        ticker_identity = base_by_ticker.get(ticker)
        if ticker_identity is not None and _security_cik(ticker_identity[1]) != cik:
            raise DataReadinessError(f"S&P cutoff anchor CIK conflicts with the base identity for {ticker}")
        inherited = ticker_identity or base_by_cik.get(cik)
        if inherited is None:
            security_id = f"cik:{cik}" if int(cik_counts[cik]) == 1 else f"cik:{cik}:ticker:{ticker}"
        else:
            inherited_ticker, security_id = inherited
            if _punctuation_alias(ticker, inherited_ticker):
                ticker = inherited_ticker
        if ticker in states:
            raise DataReadinessError(f"S&P cutoff anchor aliases collide after base identity inheritance: {ticker}")
        states[ticker] = _State(
            security_id=security_id,
            company=str(record["company"]),
            sector=str(record["sector"]),
            industry=str(record["industry"]),
            effective_to_utc=None,
            evidence_urls=(),
        )
    return states


def _base_active_identity_maps(
    memberships: pd.DataFrame | None,
    *,
    base_cutoff_date: date | None,
) -> tuple[dict[str, tuple[str, str]], dict[str, tuple[str, str]]]:
    if memberships is None or base_cutoff_date is None:
        return {}, {}
    boundary = pd.Timestamp(base_cutoff_date, tz="UTC") + pd.Timedelta(days=1)
    active = memberships[
        memberships["effective_from_utc"].lt(boundary)
        & (memberships["effective_to_utc"].isna() | memberships["effective_to_utc"].ge(boundary))
    ]
    if bool(active["ticker"].duplicated().any()):
        raise DataReadinessError("base membership has ambiguous active tickers")
    by_ticker = {str(row.ticker): (str(row.ticker), str(row.security_id)) for row in active.itertuples(index=False)}
    by_cik: dict[str, tuple[str, str]] = {}
    ambiguous_ciks: set[str] = set()
    for ticker, identity in by_ticker.values():
        security_id = identity.split(":ticker:", maxsplit=1)[0]
        if not security_id.startswith("cik:"):
            continue
        cik = security_id.removeprefix("cik:")
        if cik in by_cik and by_cik[cik] != (ticker, identity):
            ambiguous_ciks.add(cik)
            continue
        by_cik[cik] = (ticker, identity)
    for cik in ambiguous_ciks:
        by_cik.pop(cik, None)
    return by_ticker, by_cik


def _security_cik(security_id: str) -> str | None:
    cik_identity = security_id.split(":ticker:", maxsplit=1)[0]
    if not cik_identity.startswith("cik:"):
        return None
    return cik_identity.removeprefix("cik:")


def _punctuation_alias(left: str, right: str) -> bool:
    return left.replace(".", "-") == right.replace(".", "-")


def _base_security_id_at(
    memberships: pd.DataFrame | None,
    *,
    ticker: str,
    effective_at: datetime,
) -> str | None:
    if memberships is None:
        return None
    moment = pd.Timestamp(effective_at)
    matches = memberships[
        memberships["ticker"].eq(ticker)
        & memberships["effective_from_utc"].le(moment)
        & (memberships["effective_to_utc"].isna() | memberships["effective_to_utc"].ge(moment))
    ]
    identities = sorted(set(matches["security_id"].astype(str)))
    if len(identities) > 1:
        raise DataReadinessError(f"base membership has ambiguous security identity for {ticker}")
    return identities[0] if identities else None


def _load_anchor(path: Path) -> tuple[pd.DataFrame, str]:
    if not path.is_file():
        raise DataReadinessError(f"S&P cutoff anchor is missing: {path}")
    data = pd.read_csv(path, dtype=str, keep_default_na=False)
    aliases = {column.strip().lower(): column for column in data.columns}
    required = {"ticker", "company", "sector", "industry", "cik"}
    missing = sorted(required.difference(aliases))
    if missing:
        raise DataReadinessError(f"S&P cutoff anchor is missing columns: {missing}")
    anchor = pd.DataFrame(
        {
            "ticker": data[aliases["ticker"]].map(normalized_ticker),
            "company": data[aliases["company"]].astype(str).str.strip(),
            "sector": data[aliases["sector"]].astype(str).str.strip(),
            "industry": data[aliases["industry"]].astype(str).str.strip(),
            "cik": data[aliases["cik"]].astype(str).str.strip().str.removesuffix(".0").str.zfill(10),
        }
    )
    if anchor.empty or len(anchor) < 450 or len(anchor) > 550:
        raise DataReadinessError("S&P cutoff anchor must contain 450..550 securities")
    if bool(anchor["ticker"].duplicated().any()):
        duplicates = sorted(anchor.loc[anchor["ticker"].duplicated(False), "ticker"].unique())
        raise DataReadinessError(f"S&P cutoff anchor has duplicate tickers: {duplicates[:20]}")
    for column in ("ticker", "company", "sector", "industry", "cik"):
        if bool(anchor[column].eq("").any()):
            raise DataReadinessError(f"S&P cutoff anchor has empty {column}")
    records = anchor.sort_values("ticker").to_dict(orient="records")
    return anchor.sort_values("ticker").reset_index(drop=True), _json_sha256(records)


def _apply_security_exclusions(
    memberships: pd.DataFrame,
    *,
    automatic_exclusions: list[dict[str, str]],
    security_exclusions_path: Path | None,
    maximum_security_exclusion_fraction: float,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    exclusions_by_id = {item["security_id"]: dict(item) for item in automatic_exclusions}
    if security_exclusions_path is not None:
        if not security_exclusions_path.is_file():
            raise DataReadinessError(f"S&P security exclusion ledger is missing: {security_exclusions_path}")
        frame = pd.read_csv(
            security_exclusions_path,
            dtype=str,
            keep_default_na=False,
        )
        missing = sorted({"security_id", "reason"}.difference(frame.columns))
        if missing:
            raise DataReadinessError(f"S&P security exclusions are missing columns: {missing}")
        if bool(frame["security_id"].duplicated().any()):
            raise DataReadinessError("S&P security exclusion identities are duplicated")
        if "ticker" in frame.columns:
            forbidden = sorted(set(frame["ticker"].map(normalized_ticker)).intersection(_BENCHMARK_TICKERS))
            if forbidden:
                raise DataReadinessError(f"benchmark sessions cannot be excluded: {forbidden}")
        known = {
            *memberships["security_id"].astype(str),
            *exclusions_by_id,
        }
        requested_user = set(frame["security_id"].astype(str).str.strip())
        unknown = sorted(requested_user.difference(known))
        if unknown:
            raise DataReadinessError(f"S&P security exclusions contain unknown identities: {unknown[:20]}")
        if bool(frame["reason"].astype(str).str.strip().eq("").any()):
            raise DataReadinessError("S&P security exclusion reason must not be empty")
        for record in frame.to_dict(orient="records"):
            security_id = str(record["security_id"]).strip()
            exclusions_by_id.setdefault(
                security_id,
                {
                    "security_id": security_id,
                    "ticker": str(record.get("ticker", "")).strip(),
                    "reason": str(record["reason"]).strip(),
                    "effective_at_utc": "",
                },
            )
    requested = set(exclusions_by_id)
    source_count = len({*memberships["security_id"].astype(str), *requested})
    excluded_fraction = validate_security_exclusion_share(
        source_securities=source_count,
        excluded_securities=len(requested),
    )
    if excluded_fraction > maximum_security_exclusion_fraction:
        raise DataReadinessError(f"S&P security exclusions {excluded_fraction:.2%} exceed {maximum_security_exclusion_fraction:.2%}")
    exclusions = sorted(
        exclusions_by_id.values(),
        key=lambda item: item["security_id"],
    )
    kept = memberships[~memberships["security_id"].astype(str).isin(requested)].copy()
    if kept.empty:
        raise DataReadinessError("S&P security exclusions removed the complete universe")
    return kept.reset_index(drop=True), exclusions


def _parent_lineage(
    *,
    archive_directory: Path,
    event_directory: Path,
    transition_directory: Path,
    anchor_path: Path,
    anchor_semantic_sha256: str,
) -> dict[str, str]:
    raw_authority = _load_object(archive_directory / "_authority.json")
    event_authority = _load_object(event_directory / "_authority.json")
    transition_authority = _load_object(transition_directory / "_authority.json")
    return {
        "raw_authority_sha256": file_sha256(archive_directory / "_authority.json"),
        "raw_manifest_sha256": str(raw_authority.get("artifact_sha256", "")),
        "event_authority_sha256": file_sha256(event_directory / "_authority.json"),
        "event_set_sha256": str(event_authority.get("event_set_sha256", "")),
        "transition_authority_sha256": file_sha256(transition_directory / "_authority.json"),
        "transition_set_sha256": str(transition_authority.get("transition_set_sha256", "")),
        "anchor_file_sha256": file_sha256(anchor_path),
        "anchor_semantic_sha256": anchor_semantic_sha256,
    }


def _load_extension_parent(
    membership_directory: Path | None,
    *,
    start_date: date,
    cutoff_date: date,
) -> tuple[pd.DataFrame | None, dict[str, Any] | None]:
    if membership_directory is None:
        return None, None
    root = membership_directory.resolve()
    request_path = root / "_request.json"
    manifest_path = root / "_manifest.json"
    authority_path = root / "_authority.json"
    request = _load_object(request_path)
    manifest = _load_object(manifest_path)
    authority = _load_object(authority_path)
    request_payload = {str(key): value for key, value in request.items() if key != "request_sha256"}
    request_sha256 = _json_sha256(request_payload)
    request_parent = request.get("parent_lineage")
    request_extension_parent = request.get("extension_parent")
    if (
        request.get("schema") != MEMBERSHIP_REQUEST_SCHEMA
        or request.get("request_sha256") != request_sha256
        or authority.get("schema") != MEMBERSHIP_AUTHORITY_SCHEMA
        or authority.get("state") != "membership_complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("request_sha256") != request_sha256
        or manifest.get("schema") != MEMBERSHIP_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or manifest.get("parent_lineage") != request_parent
        or authority.get("parent_lineage") != request_parent
        or manifest.get("extension_parent") != request_extension_parent
        or authority.get("extension_parent") != request_extension_parent
        or manifest.get("start_date") != request.get("start_date")
        or manifest.get("cutoff_date") != request.get("cutoff_date")
    ):
        raise DataReadinessError("base S&P membership authority is invalid")
    if str(request.get("start_date")) != start_date.isoformat():
        raise DataReadinessError("base S&P membership start date differs from extension")
    try:
        base_cutoff = date.fromisoformat(str(request.get("cutoff_date", "")))
    except ValueError as exc:
        raise DataReadinessError("base S&P membership cutoff is invalid") from exc
    if base_cutoff >= cutoff_date:
        raise DataReadinessError("base S&P membership cutoff must precede extension cutoff")
    record = manifest.get("membership_artifact")
    exclusion_record = manifest.get("exclusion_artifact")
    if not isinstance(record, dict) or not isinstance(exclusion_record, dict):
        raise DataReadinessError("base S&P membership artifact inventory is invalid")
    membership_path = _verified_artifact(root, record)
    exclusion_path = _verified_artifact(root, exclusion_record)
    sidecar = manifest_path_for(membership_path)
    if not sidecar.is_file() or manifest.get("membership_manifest_sha256") != file_sha256(sidecar):
        raise DataReadinessError("base canonical S&P membership manifest is invalid")
    memberships, _ = load_canonical_artifact(
        membership_path,
        expected_type="memberships",
        allow_research=True,
    )
    exclusions = _load_array(exclusion_path)
    universe_sha256 = _membership_sha256(memberships)
    if (
        manifest.get("universe_sha256") != universe_sha256
        or authority.get("universe_sha256") != universe_sha256
        or int(manifest.get("membership_intervals", -1)) != len(memberships)
        or int(authority.get("membership_intervals", -1)) != len(memberships)
        or int(manifest.get("security_count", -1)) != memberships["security_id"].nunique()
        or int(authority.get("security_count", -1)) != memberships["security_id"].nunique()
        or int(manifest.get("ticker_count", -1)) != memberships["ticker"].nunique()
        or int(manifest.get("excluded_security_count", -1)) != len(exclusions)
    ):
        raise DataReadinessError("base S&P membership semantic identity is invalid")
    return memberships, {
        "schema": "edge_rebuild.sp500_membership_extension_parent.v1",
        "start_date": start_date.isoformat(),
        "cutoff_date": base_cutoff.isoformat(),
        "authority_sha256": file_sha256(authority_path),
        "manifest_sha256": file_sha256(manifest_path),
        "membership_table_sha256": file_sha256(membership_path),
        "universe_sha256": universe_sha256,
    }


def _request_payload(
    *,
    parent: dict[str, str],
    start_date: date,
    cutoff_date: date,
    security_exclusions_path: Path | None,
    maximum_security_exclusion_fraction: float,
    extension_parent: dict[str, Any] | None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": MEMBERSHIP_REQUEST_SCHEMA,
        "reconstruction_schema": MEMBERSHIP_RECONSTRUCTION_SCHEMA,
        "start_date": start_date.isoformat(),
        "cutoff_date": cutoff_date.isoformat(),
        "maximum_security_exclusion_fraction": maximum_security_exclusion_fraction,
        "security_exclusions_sha256": (None if security_exclusions_path is None else file_sha256(security_exclusions_path)),
        "parent_lineage": parent,
    }
    if extension_parent is not None:
        payload["extension_parent"] = extension_parent
    return payload


def verify_membership_namespace_extension(
    base: pd.DataFrame,
    current: pd.DataFrame,
    *,
    base_cutoff_date: str,
    current_cutoff_date: str,
) -> None:
    """Require an extension to preserve every pre-cutoff identity interval."""

    base_cutoff = pd.Timestamp(base_cutoff_date).date()
    current_cutoff = pd.Timestamp(current_cutoff_date).date()
    if current_cutoff < base_cutoff:
        raise DataReadinessError("S&P membership extension predates its base A4.3 namespace authority")
    boundary = pd.Timestamp(base_cutoff, tz="UTC") + pd.Timedelta(days=1)
    excluded_columns = {"universe_snapshot_id"}
    base_columns = set(base.columns).difference(excluded_columns)
    current_columns = set(current.columns).difference(excluded_columns)
    if base_columns != current_columns:
        raise DataReadinessError("S&P membership extension contract differs from its base authority")
    columns = sorted(base_columns)
    timestamp_columns = (
        "effective_from_utc",
        "effective_to_utc",
        "available_at_utc",
    )
    if not set(timestamp_columns).issubset(base_columns):
        raise DataReadinessError("S&P membership extension is missing causal timestamp columns")

    def prefix(frame: pd.DataFrame) -> list[dict[str, Any]]:
        data = frame.loc[:, columns].copy()
        for column in timestamp_columns:
            data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")
        data = data[data["effective_from_utc"].lt(boundary)].copy()
        data.loc[
            data["effective_to_utc"].isna() | data["effective_to_utc"].ge(boundary),
            "effective_to_utc",
        ] = pd.NaT
        data = data.sort_values(
            ["security_id", "effective_from_utc", "ticker"],
            kind="stable",
        ).reset_index(drop=True)
        records: list[dict[str, Any]] = []
        for record in data.to_dict(orient="records"):
            for column in timestamp_columns:
                value = record[column]
                record[column] = None if pd.isna(value) else pd.Timestamp(value).tz_convert("UTC").isoformat()
            records.append({str(key): value for key, value in record.items()})
        return records

    if prefix(base) != prefix(current):
        raise DataReadinessError("S&P membership authority does not preserve its base identity namespace")


def _historical_security_id(change: IndexChange) -> str:
    payload = {
        "company": change.company.strip().lower(),
        "ticker": change.ticker,
    }
    return f"sp500-historical:{_json_sha256(payload)[:24]}"


def _transition_security_id(record: dict[str, Any], *, side: str) -> str:
    return f"sp500-transition:{_json_sha256({'id': record['transition_id'], 'side': side})[:24]}"


def _equivalent_security_id(left: str, right: str) -> bool:
    if left == right:
        return True
    left_cik = left.split(":ticker:", maxsplit=1)[0]
    right_cik = right.split(":ticker:", maxsplit=1)[0]
    return left_cik.startswith("cik:") and left_cik == right_cik


def _membership_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    ordered = frame.sort_values(["ticker", "effective_from_utc", "security_id"], kind="stable")
    records: list[dict[str, Any]] = []
    for record in ordered.to_dict(orient="records"):
        for field in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
            value = record[field]
            record[field] = None if pd.isna(value) else pd.Timestamp(value).tz_convert("UTC").isoformat()
        records.append({str(key): value for key, value in record.items()})
    return records


def _membership_sha256(frame: pd.DataFrame) -> str:
    return _json_sha256(_membership_records(frame))


def _artifact_record(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": file_sha256(path)}


def _verified_artifact(root: Path, record: dict[str, Any]) -> Path:
    path = _resolve_inside(root, str(record.get("path", "")))
    if not path.is_file() or record.get("sha256") != file_sha256(path) or int(record.get("bytes", -1)) != path.stat().st_size:
        raise DataReadinessError(f"S&P membership artifact hash is invalid: {path.name}")
    return path


def _validate_parameters(
    *,
    start_date: date,
    cutoff_date: date,
    maximum_security_exclusion_fraction: float,
) -> None:
    if start_date > cutoff_date:
        raise ValueError("start_date must not be after cutoff_date")
    if not 0 <= maximum_security_exclusion_fraction <= MAXIMUM_SECURITY_EXCLUSION_FRACTION:
        raise ValueError("maximum_security_exclusion_fraction must be between 0 and 0.05")


def _ny_date(value: datetime) -> date:
    return value.astimezone(ZoneInfo("America/New_York")).date()


def _session_midnight(value: date) -> datetime:
    return datetime.combine(
        value,
        datetime.min.time(),
        tzinfo=ZoneInfo("America/New_York"),
    ).astimezone(UTC)


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise DataReadinessError("S&P membership artifact escapes its authority directory")
    return candidate


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise DataReadinessError(f"JSON artifact is not an object: {path}")
    return cast(dict[str, Any], value)


def _load_array(path: Path) -> list[dict[str, str]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"invalid JSON artifact: {path}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise DataReadinessError(f"JSON artifact is not an array of objects: {path}")
    return [{str(key): str(item[key]) for key in sorted(item)} for item in value]


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_json_atomic(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)
