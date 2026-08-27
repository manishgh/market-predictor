"""Prospective Alpaca news observations with real first-seen timestamps.

This module deliberately does not repair retrospective news.  One invocation
publishes one immutable poll.  Repeated polls are compacted separately so every
provider revision and every source-coverage interval remains auditable.
"""
from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final, cast
from urllib.parse import parse_qsl, urlsplit
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd
import pyarrow.parquet as pq

from market_predictor.canonical.audits import (
    CanonicalAuditCheck,
    CanonicalAuditReport,
    audit_source_collections,
)
from market_predictor.canonical.contracts import SourceCollection
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.symbols import canonical_symbol
from market_predictor.intraday.datasets.bar_dataset import (
    load_complete_intraday_bar_dataset,
)
from market_predictor.intraday.datasets.history import json_sha256
from market_predictor.locking import LockTimeout, file_lock
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
)
from market_predictor.sources.alpaca import AlpacaAssetSnapshot, AlpacaNewsPage
from market_predictor.universe.sp500.membership_authority import (
    load_sp500_membership_authority_envelope,
    verify_membership_namespace_extension,
)
from market_predictor.universe.sp500.observed_membership_authority import (
    AUTHORITY_SCHEMA as OBSERVED_MEMBERSHIP_AUTHORITY_SCHEMA,
)
from market_predictor.universe.sp500.observed_membership_authority import load_observed_sp500_membership_authority

POLL_REQUEST_SCHEMA: Final = "edge_rebuild.prospective_broker_action_poll_request.v1"
POLL_MANIFEST_SCHEMA: Final = "edge_rebuild.prospective_broker_action_poll_manifest.v1"
POLL_AUTHORITY_SCHEMA: Final = "edge_rebuild.prospective_broker_action_poll_authority.v1"
OBSERVATION_SCHEMA: Final = "edge_rebuild.prospective_broker_action_observation.v1"
IDENTITY_AUDIT_SCHEMA: Final = "edge_rebuild.prospective_security_identity_audit.v1"
GENERATION_REQUEST_SCHEMA: Final = "edge_rebuild.prospective_broker_action_generation_request.v1"
GENERATION_MANIFEST_SCHEMA: Final = "edge_rebuild.prospective_broker_action_generation_manifest.v1"
GENERATION_AUTHORITY_SCHEMA: Final = "edge_rebuild.prospective_broker_action_generation_authority.v1"
REVISION_SCHEMA: Final = "edge_rebuild.prospective_broker_action_revision.v1"
SECURITY_NAMESPACE_SCHEMA: Final = "edge_rebuild.a43_security_identity_namespace.v1"
ATTEMPT_SCHEMA: Final = "edge_rebuild.prospective_broker_action_attempt.v1"
MAX_PAGES_PER_BATCH: Final = 200
MAX_BYTES_PER_BATCH: Final = 32 * 1024 * 1024
MAX_BYTES_PER_POLL: Final = 64 * 1024 * 1024
MAX_GENERATION_INPUT_BYTES: Final = 64 * 1024 * 1024
ALPACA_ASSET_HOSTNAMES: Final = frozenset({"api.alpaca.markets", "paper-api.alpaca.markets"})
NEW_YORK: Final = ZoneInfo("America/New_York")
_ROOT_FILES: Final = frozenset(
    {
        "_request.json",
        "_status.json",
        "_manifest.json",
        "_authority.json",
        "_collector.lock",
        "assets.parquet",
        "assets.parquet.manifest.json",
        "assets.parquet.lock",
        "event_observations.parquet",
        "event_observations.parquet.manifest.json",
        "event_observations.parquet.lock",
        "identity_audit.parquet",
        "identity_audit.parquet.manifest.json",
        "identity_audit.parquet.lock",
        "source_collections.parquet",
        "source_collections.parquet.manifest.json",
        "source_collections.parquet.lock",
        "raw_assets.json",
        "raw_assets.body",
        "attempts",
        "raw_pages",
    }
)
_MEMBERSHIP_PARENT_LINEAGE_KEYS: Final = frozenset(
    {
        "anchor_file_sha256",
        "anchor_semantic_sha256",
        "event_authority_sha256",
        "event_set_sha256",
        "raw_authority_sha256",
        "raw_manifest_sha256",
        "transition_authority_sha256",
        "transition_set_sha256",
    }
)
_GENERATION_ROOT_FILES: Final = frozenset(
    {
        "_request.json",
        "_manifest.json",
        "_authority.json",
        "event_revisions.parquet",
        "event_revisions.parquet.manifest.json",
        "event_revisions.parquet.lock",
        "identity_observations.parquet",
        "identity_observations.parquet.manifest.json",
        "identity_observations.parquet.lock",
        "source_collections.parquet",
        "source_collections.parquet.manifest.json",
        "source_collections.parquet.lock",
    }
)
_OBSERVATION_COLUMNS: Final = (
    "observation_id",
    "revision_id",
    "provider_event_id",
    "ticker",
    "candidate_security_id",
    "alpaca_asset_id",
    "identity_available_at_utc",
    "identity_eligible",
    "identity_ineligible_reason",
    "published_at_utc",
    "provider_updated_at_utc",
    "revision_first_seen_at_utc",
    "source",
    "title",
    "url",
    "summary",
    "text",
    "raw_sha256",
    "provider_timestamp_anomaly",
    "batch_id",
    "page_index",
    "schema_version",
)

AssetsFetcher = Callable[[], AlpacaAssetSnapshot]
NewsPageFetcher = Callable[[str, datetime, datetime, str | None], AlpacaNewsPage]
Clock = Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class ProspectivePoll:
    directory: Path
    request: Mapping[str, object]
    manifest: Mapping[str, object]
    observations: pd.DataFrame
    source_collections: pd.DataFrame
    identity_audit: pd.DataFrame


@dataclass(frozen=True, slots=True)
class ProspectiveGeneration:
    directory: Path
    request: Mapping[str, object]
    manifest: Mapping[str, object]
    revisions: pd.DataFrame
    source_collections: pd.DataFrame
    identity_observations: pd.DataFrame


def collect_prospective_broker_action_poll(
    *,
    membership_authority_directory: Path,
    intraday_bar_dataset_directory: Path,
    registry_directory: Path,
    output_directory: Path,
    fetch_assets: AssetsFetcher,
    fetch_page: NewsPageFetcher,
    observed_at_utc: datetime | None = None,
    previous_poll_directory: Path | None = None,
    lookback_hours: int = 25,
    batch_size: int = 50,
    maximum_continuous_gap_seconds: int = 120,
    memory_hard_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.75,
    clock: Clock | None = None,
) -> dict[str, object]:
    """Collect one resumable, immutable prospective Alpaca news poll."""

    output = output_directory.resolve()
    _validate_poll_paths(
        output=output,
        membership=membership_authority_directory.resolve(),
        intraday_dataset=intraday_bar_dataset_directory.resolve(),
        registry=registry_directory.resolve(),
        previous=(previous_poll_directory.resolve() if previous_poll_directory is not None else None),
    )
    try:
        with file_lock(output / "_collector", timeout=0.0):
            return _collect_prospective_broker_action_poll(
                membership_authority_directory=membership_authority_directory,
                intraday_bar_dataset_directory=intraday_bar_dataset_directory,
                registry_directory=registry_directory,
                output_directory=output_directory,
                fetch_assets=fetch_assets,
                fetch_page=fetch_page,
                observed_at_utc=observed_at_utc,
                previous_poll_directory=previous_poll_directory,
                lookback_hours=lookback_hours,
                batch_size=batch_size,
                maximum_continuous_gap_seconds=maximum_continuous_gap_seconds,
                memory_hard_budget_gib=memory_hard_budget_gib,
                memory_headroom_gib=memory_headroom_gib,
                clock=clock,
            )
    except LockTimeout as exc:
        raise DataReadinessError(f"another process owns prospective poll {output}") from exc


def _collect_prospective_broker_action_poll(
    *,
    membership_authority_directory: Path,
    intraday_bar_dataset_directory: Path,
    registry_directory: Path,
    output_directory: Path,
    fetch_assets: AssetsFetcher,
    fetch_page: NewsPageFetcher,
    observed_at_utc: datetime | None = None,
    previous_poll_directory: Path | None = None,
    lookback_hours: int = 25,
    batch_size: int = 50,
    maximum_continuous_gap_seconds: int = 120,
    memory_hard_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.75,
    clock: Clock | None = None,
) -> dict[str, object]:
    """Implementation executed under the poll-specific process lock."""

    if not 24 <= lookback_hours <= 48:
        raise ValueError("lookback_hours must be between 24 and 48")
    if not 1 <= batch_size <= 50:
        raise ValueError("batch_size must be between 1 and 50")
    if not 60 <= maximum_continuous_gap_seconds <= 300:
        raise ValueError("maximum_continuous_gap_seconds must be between 60 and 300")
    now = clock or (lambda: datetime.now(UTC))
    output = output_directory.resolve()
    membership_root = membership_authority_directory.resolve()
    intraday_dataset_root = intraday_bar_dataset_directory.resolve()
    registry_root = registry_directory.resolve()
    output.mkdir(parents=True, exist_ok=True)
    registry_root.mkdir(parents=True, exist_ok=True)
    if (output / "_authority.json").exists():
        return _resume_completed_poll(
            output,
            registry_root=registry_root,
            membership_root=membership_root,
            intraday_dataset_root=intraday_dataset_root,
        )

    membership, parent = _load_membership_authority(membership_root)
    namespace = _load_a43_security_namespace(
        intraday_dataset_root,
        membership_root=membership_root,
        membership_parent=parent,
    )
    request_path = output / "_request.json"
    existing_request = _json_object(request_path) if request_path.exists() else None
    if existing_request is not None:
        observed = _required_utc(existing_request, "observed_at_utc")
    else:
        observed = _utc(observed_at_utc or now()).replace(second=0, microsecond=0)
    query_start = observed - timedelta(hours=lookback_hours)
    previous = load_prospective_broker_action_poll(previous_poll_directory) if previous_poll_directory is not None else None
    if previous is not None and _required_utc(previous.manifest, "observed_at_utc") >= observed:
        raise DataReadinessError("prospective previous poll must precede the scheduled cutoff")
    if previous is not None and (
        previous.request.get("security_identity_namespace_sha256") != namespace["security_identity_namespace_sha256"]
        or Path(str(previous.request.get("registry_directory", ""))).resolve() != registry_root
    ):
        raise DataReadinessError("prospective previous poll uses a different authority or registry")
    if previous is not None:
        previous_membership_root = Path(str(previous.request.get("membership_authority_directory", ""))).resolve()
        previous_memberships, previous_parent = _load_membership_authority(previous_membership_root)
        _require_membership_authority_chain(
            previous_memberships,
            previous_parent,
            membership,
            parent,
        )
    _require_membership_observed_before_poll(
        parent,
        poll_cutoff=observed,
        maximum_age_seconds=maximum_continuous_gap_seconds,
    )
    previous_identity = _previous_identity(previous)
    request_payload: dict[str, object] = {
        "schema": POLL_REQUEST_SCHEMA,
        "observed_at_utc": observed.isoformat(),
        "query_start_utc": query_start.isoformat(),
        "query_end_utc": observed.isoformat(),
        "lookback_hours": lookback_hours,
        "batch_size": batch_size,
        "maximum_continuous_gap_seconds": maximum_continuous_gap_seconds,
        "source": "alpaca:benzinga",
        "availability_policy": "observed",
        "membership_authority_directory": str(membership_root),
        "membership_authority_sha256": parent["authority_sha256"],
        "membership_manifest_sha256": parent["manifest_sha256"],
        "membership_table_sha256": parent["membership_table_sha256"],
        "membership_universe_sha256": parent["universe_sha256"],
        "membership_cutoff_date": parent["cutoff_date"],
        "intraday_bar_dataset_directory": str(intraday_dataset_root),
        "intraday_bar_authority_sha256": namespace["intraday_bar_authority_sha256"],
        "intraday_bar_manifest_sha256": namespace["intraday_bar_manifest_sha256"],
        "intraday_bar_request_sha256": namespace["intraday_bar_request_sha256"],
        "intraday_bar_parent_lineage_sha256": namespace["intraday_bar_parent_lineage_sha256"],
        "security_identity_namespace_sha256": namespace["security_identity_namespace_sha256"],
        "registry_directory": str(registry_root),
        "previous_poll": previous_identity,
    }
    request_sha256 = json_sha256(request_payload)
    request = {**request_payload, "request_sha256": request_sha256}
    _write_or_validate_json(request_path, request)
    claim = _claim_poll_cutoff(
        registry_root,
        output=output,
        request=request,
        namespace=namespace,
        membership_parent=parent,
        previous=previous,
        claimed_at=_utc(now()),
    )
    assert_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective broker-action poll start",
    )

    attempts_root = output / "attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    raw_assets_path = output / "raw_assets.json"
    raw_assets_body_path = output / "raw_assets.body"
    assets_path = output / "assets.parquet"
    asset_artifact_complete = assets_path.exists() and manifest_path_for(assets_path).exists()
    asset_artifact_partial = assets_path.exists() != manifest_path_for(assets_path).exists()
    raw_asset_complete = raw_assets_path.exists() and raw_assets_body_path.exists()
    raw_asset_partial = raw_assets_path.exists() != raw_assets_body_path.exists()
    if raw_asset_partial:
        raw_assets_path.unlink(missing_ok=True)
        raw_assets_body_path.unlink(missing_ok=True)
        raw_asset_complete = False
    if asset_artifact_partial or (asset_artifact_complete and not raw_asset_complete):
        assets_path.unlink(missing_ok=True)
        manifest_path_for(assets_path).unlink(missing_ok=True)
        asset_artifact_complete = False
    if raw_asset_complete:
        raw_assets = _json_object(raw_assets_path)
        if (
            raw_assets.get("request_sha256") != request_sha256
            or raw_assets.get("body_sha256") != file_sha256(raw_assets_body_path)
            or raw_assets.get("status_code") != 200
        ):
            raise DataReadinessError("prospective asset raw evidence does not verify")
        _verify_asset_request_url(
            str(raw_assets.get("requested_url", "")),
            final_url=str(raw_assets.get("final_url", "")),
            redirect_chain=raw_assets.get("redirect_chain"),
        )
        assets_received = _required_utc(raw_assets, "response_received_at_utc")
        reconstructed = _normalize_assets(_asset_frame_from_body(raw_assets_body_path.read_bytes()))
    else:
        try:
            snapshot = fetch_assets()
            assets_received = _utc(snapshot.retrieved_at_utc)
            reconstructed = _normalize_assets(_asset_frame_from_body(snapshot.raw_body))
            supplied = _normalize_assets(snapshot.assets)
            if not _frames_equal(reconstructed, supplied):
                raise DataReadinessError("parsed Alpaca assets differ from the archived HTTP body")
            if snapshot.final_url is None:
                raise DataReadinessError("prospective Alpaca asset snapshot lacks final URL evidence")
            _verify_asset_request_url(
                snapshot.requested_url,
                final_url=snapshot.final_url,
                redirect_chain=snapshot.redirect_chain,
            )
            if not (observed <= assets_received <= observed + timedelta(seconds=maximum_continuous_gap_seconds)):
                raise DataReadinessError("prospective asset snapshot exceeded the continuous-coverage limit")
            _atomic_bytes(raw_assets_body_path, snapshot.raw_body)
            raw_assets = {
                "request_sha256": request_sha256,
                "response_received_at_utc": assets_received.isoformat(),
                "requested_url": snapshot.requested_url,
                "final_url": snapshot.final_url,
                "redirect_chain": list(snapshot.redirect_chain),
                "status_code": snapshot.status_code,
                "response_headers": snapshot.response_headers,
                "body_sha256": file_sha256(raw_assets_body_path),
                "body_bytes": raw_assets_body_path.stat().st_size,
            }
            _atomic_json(raw_assets_path, raw_assets)
        except Exception as exc:
            _record_asset_failure(
                output=output,
                attempts_root=attempts_root,
                request_sha256=request_sha256,
                error=f"{type(exc).__name__}: {str(exc)[:500]}",
                recorded_at=_utc(now()),
            )
            raise
    if asset_artifact_complete:
        assets, _ = load_canonical_artifact(
            assets_path,
            expected_type="prospective_alpaca_assets",
            allow_research=True,
        )
        if not _frames_equal(assets, reconstructed):
            raise DataReadinessError("prospective asset artifact does not replay from raw HTTP evidence")
    else:
        assets = reconstructed
        write_canonical_artifact(
            assets,
            assets_path,
            artifact_type="prospective_alpaca_assets",
            audit=_passing_audit("asset_snapshot", len(assets), "asset symbols and IDs are unique"),
            inputs={
                "request_sha256": request_sha256,
                "raw_assets_envelope_sha256": file_sha256(raw_assets_path),
                "raw_assets_body_sha256": file_sha256(raw_assets_body_path),
            },
            production_ready=False,
        )
    if not (observed <= assets_received <= observed + timedelta(seconds=maximum_continuous_gap_seconds)):
        raise DataReadinessError("prospective asset snapshot exceeded the continuous-coverage limit")

    identity = _build_identity_audit(
        membership,
        assets,
        observed_at=observed,
        identity_observed_at=assets_received,
        membership_cutoff_date=str(parent["cutoff_date"]),
        previous_identity=(previous.identity_audit if previous is not None else None),
    )
    symbols = tuple(sorted(identity["ticker"].astype(str)))
    if not symbols:
        raise DataReadinessError("prospective poll has no membership symbols")
    batches = [symbols[index : index + batch_size] for index in range(0, len(symbols), batch_size)]
    raw_pages_root = output / "raw_pages"
    raw_pages_root.mkdir(parents=True, exist_ok=True)
    page_rows: list[tuple[str, int, datetime, dict[str, Any]]] = []
    failures: dict[str, str] = {}
    batch_times: dict[str, tuple[datetime, datetime]] = {}
    poll_body_bytes = 0
    for index, batch in enumerate(batches):
        batch_id = f"batch-{index:04d}"
        try:
            pages, started_at, completed_at = _collect_batch_pages(
                batch_id=batch_id,
                symbols=batch,
                query_start=query_start,
                query_end=observed,
                request_sha256=request_sha256,
                raw_pages_root=raw_pages_root,
                fetch_page=fetch_page,
                clock=now,
                memory_hard_budget_gib=memory_hard_budget_gib,
                memory_headroom_gib=memory_headroom_gib,
                maximum_continuous_gap_seconds=maximum_continuous_gap_seconds,
            )
            batch_times[batch_id] = (started_at, completed_at)
            poll_body_bytes += sum(int(payload["body_bytes"]) for _, _, payload in pages)
            if poll_body_bytes > MAX_BYTES_PER_POLL:
                raise DataReadinessError("Alpaca news poll exceeded the 64 MiB raw evidence limit")
            page_rows.extend((batch_id, page_index, received, payload) for page_index, received, payload in pages)
        except Exception as exc:
            failures[batch_id] = f"{type(exc).__name__}: {str(exc)[:500]}"
            _write_attempt(
                attempts_root,
                batch_id=batch_id,
                request_sha256=request_sha256,
                error=failures[batch_id],
                recorded_at=_utc(now()),
            )

    for batch_id, (_, completed_at) in batch_times.items():
        if (completed_at - observed).total_seconds() > maximum_continuous_gap_seconds:
            failures[batch_id] = "poll_completion_exceeded_continuous_coverage_limit"
            _write_attempt(
                attempts_root,
                batch_id=batch_id,
                request_sha256=request_sha256,
                error=failures[batch_id],
                recorded_at=_utc(now()),
            )

    if failures:
        status = {
            "schema": POLL_MANIFEST_SCHEMA,
            "status": "incomplete",
            "request_sha256": request_sha256,
            "failed_batches": failures,
            "completed_batches": len(batch_times),
            "requested_batches": len(batches),
            "updated_at_utc": _utc(now()).isoformat(),
        }
        _atomic_json(output / "_status.json", status)
        return status

    page_rows, replayed_batch_times = _replay_raw_pages(
        raw_pages_root,
        batches=batches,
        request_sha256=request_sha256,
        query_start=query_start,
        query_end=observed,
        maximum_continuous_gap_seconds=maximum_continuous_gap_seconds,
    )
    if replayed_batch_times != batch_times:
        raise DataReadinessError("prospective batch timing does not replay")
    _verify_attempt_inventory(attempts_root, request_sha256=request_sha256)
    observations = _build_observations(page_rows, identity, observed_at=observed)
    collections = _build_source_collections(
        batches=batches,
        batch_times=batch_times,
        observations=observations,
        identity=identity,
        observed_at=observed,
        previous=previous,
        maximum_gap_seconds=maximum_continuous_gap_seconds,
    )
    identity_path = output / "identity_audit.parquet"
    observations_path = output / "event_observations.parquet"
    collections_path = output / "source_collections.parquet"
    inputs = {
        "request_sha256": request_sha256,
        "registry_claim_sha256": str(claim["claim_file_sha256"]),
        "intraday_bar_authority_sha256": str(namespace["intraday_bar_authority_sha256"]),
        "security_identity_namespace_sha256": str(namespace["security_identity_namespace_sha256"]),
        "membership_authority_sha256": str(parent["authority_sha256"]),
        "membership_table_sha256": str(parent["membership_table_sha256"]),
        "raw_assets_sha256": file_sha256(raw_assets_path),
        "raw_assets_body_sha256": file_sha256(raw_assets_body_path),
        "raw_pages_inventory_sha256": _directory_inventory_sha256(raw_pages_root),
        "attempt_inventory_sha256": _directory_inventory_sha256(attempts_root),
    }
    identity_manifest = write_canonical_artifact(
        identity,
        identity_path,
        artifact_type="prospective_security_identity_audit",
        audit=_passing_audit("identity_audit", len(identity), "every requested symbol has one persisted identity status"),
        inputs=inputs,
        production_ready=False,
    )
    observations_manifest = write_canonical_artifact(
        observations,
        observations_path,
        artifact_type="prospective_broker_action_observations",
        audit=_passing_audit("event_observations", len(observations), "revision observations are unique and timestamped"),
        inputs=inputs,
        production_ready=False,
    )
    collection_audit = CanonicalAuditReport(checks=audit_source_collections(collections, require_success=True))
    collection_manifest = write_canonical_artifact(
        collections,
        collections_path,
        artifact_type="source_collections",
        audit=collection_audit,
        inputs=inputs,
        production_ready=False,
    )
    manifest: dict[str, object] = {
        "schema": POLL_MANIFEST_SCHEMA,
        "status": "complete",
        "request_sha256": request_sha256,
        "observed_at_utc": observed.isoformat(),
        "requested_batches": len(batches),
        "completed_batches": len(batch_times),
        "failed_batches": {},
        "security_count": len(identity),
        "identity_eligible_security_count": int(identity["identity_eligible"].sum()),
        "event_observation_count": len(observations),
        "production_identity_event_count": int(observations["identity_eligible"].sum()) if not observations.empty else 0,
        "artifacts": {
            "assets": _artifact_record(assets_path),
            "identity_audit": _artifact_record(identity_path),
            "event_observations": _artifact_record(observations_path),
            "source_collections": _artifact_record(collections_path),
        },
        "artifact_manifest_hashes": {
            "assets": file_sha256(manifest_path_for(assets_path)),
            "identity_audit": file_sha256(manifest_path_for(identity_path)),
            "event_observations": file_sha256(manifest_path_for(observations_path)),
            "source_collections": file_sha256(manifest_path_for(collections_path)),
        },
        "raw_assets_sha256": file_sha256(raw_assets_path),
        "raw_assets_body_sha256": file_sha256(raw_assets_body_path),
        "raw_pages_inventory_sha256": _directory_inventory_sha256(raw_pages_root),
        "attempt_inventory_sha256": _directory_inventory_sha256(attempts_root),
        "membership_authority_sha256": parent["authority_sha256"],
        "membership_manifest_sha256": parent["manifest_sha256"],
        "membership_table_sha256": parent["membership_table_sha256"],
        "membership_cutoff_date": parent["cutoff_date"],
        "intraday_bar_authority_sha256": namespace["intraday_bar_authority_sha256"],
        "intraday_bar_manifest_sha256": namespace["intraday_bar_manifest_sha256"],
        "intraday_bar_request_sha256": namespace["intraday_bar_request_sha256"],
        "intraday_bar_parent_lineage_sha256": namespace["intraday_bar_parent_lineage_sha256"],
        "security_identity_namespace_sha256": namespace["security_identity_namespace_sha256"],
        "registry_claim_sha256": claim["claim_file_sha256"],
        "availability_policy": "observed",
        "production_ready": False,
        "completed_at_utc": _utc(now()).isoformat(),
        "memory": memory_audit(
            hard_budget_gib=memory_hard_budget_gib,
            headroom_gib=memory_headroom_gib,
        ).to_record(),
    }
    del identity_manifest, observations_manifest, collection_manifest
    assert_peak_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective broker-action poll publication",
    )
    _atomic_json(output / "_status.json", manifest)
    _atomic_json(output / "_manifest.json", manifest)
    authority = {
        "schema": POLL_AUTHORITY_SCHEMA,
        "state": "complete",
        "artifact": "_manifest.json",
        "artifact_sha256": file_sha256(output / "_manifest.json"),
        "request_sha256": request_sha256,
        "observed_at_utc": observed.isoformat(),
        "security_identity_namespace_sha256": namespace["security_identity_namespace_sha256"],
        "registry_claim_sha256": claim["claim_file_sha256"],
        "production_ready": False,
    }
    _atomic_json(output / "_authority.json", authority)
    _commit_poll_cutoff(registry_root, poll_root=output)
    load_prospective_broker_action_poll(output)
    return manifest


def load_prospective_broker_action_poll(
    directory: Path,
) -> ProspectivePoll:
    """Strictly replay a completed poll and its parent chain without recursion."""

    chain: list[Path] = []
    visited: set[Path] = set()
    current = directory.resolve()
    child_cutoff: datetime | None = None
    while True:
        if current in visited:
            raise DataReadinessError("prospective previous-poll lineage contains a cycle")
        visited.add(current)
        request = _json_object(current / "_request.json")
        payload = {str(key): value for key, value in request.items() if key != "request_sha256"}
        if request.get("schema") != POLL_REQUEST_SCHEMA or request.get("request_sha256") != json_sha256(payload):
            raise DataReadinessError("prospective poll request identity does not verify")
        cutoff = _required_utc(request, "observed_at_utc")
        if child_cutoff is not None and cutoff >= child_cutoff:
            raise DataReadinessError("prospective previous-poll cutoffs are not strictly increasing")
        chain.append(current)
        previous_record = request.get("previous_poll")
        if previous_record is None:
            break
        if not isinstance(previous_record, Mapping):
            raise DataReadinessError("prospective previous-poll identity is malformed")
        child_cutoff = cutoff
        current = Path(str(previous_record.get("directory", ""))).resolve()

    previous: ProspectivePoll | None = None
    for root in reversed(chain):
        previous = _load_prospective_broker_action_poll_once(root, preloaded_previous=previous)
    if previous is None:
        raise DataReadinessError("prospective poll chain is empty")
    return previous


def _load_prospective_broker_action_poll_once(
    root: Path,
    *,
    preloaded_previous: ProspectivePoll | None,
) -> ProspectivePoll:
    """Replay exactly one poll using its already verified parent."""

    if not root.is_dir() or {path.name for path in root.iterdir()} != _ROOT_FILES:
        raise DataReadinessError("prospective poll root inventory does not verify")
    request = _json_object(root / "_request.json")
    payload = {str(key): value for key, value in request.items() if key != "request_sha256"}
    request_sha256 = request.get("request_sha256")
    if request.get("schema") != POLL_REQUEST_SCHEMA or request_sha256 != json_sha256(payload):
        raise DataReadinessError("prospective poll request identity does not verify")
    manifest_path = root / "_manifest.json"
    manifest = _json_object(manifest_path)
    status = _json_object(root / "_status.json")
    authority = _json_object(root / "_authority.json")
    if (
        manifest != status
        or manifest.get("schema") != POLL_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or manifest.get("failed_batches") != {}
        or authority.get("schema") != POLL_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("request_sha256") != request_sha256
        or set(authority)
        != {
            "schema",
            "state",
            "artifact",
            "artifact_sha256",
            "request_sha256",
            "observed_at_utc",
            "security_identity_namespace_sha256",
            "registry_claim_sha256",
            "production_ready",
        }
        or authority.get("artifact") != "_manifest.json"
        or authority.get("observed_at_utc") != manifest.get("observed_at_utc")
        or authority.get("security_identity_namespace_sha256") != manifest.get("security_identity_namespace_sha256")
        or authority.get("registry_claim_sha256") != manifest.get("registry_claim_sha256")
        or authority.get("production_ready") is not False
        or manifest.get("production_ready") is not False
    ):
        raise DataReadinessError("prospective poll manifest or authority does not verify")
    membership_root = Path(str(request.get("membership_authority_directory", ""))).resolve()
    if root == membership_root or root in membership_root.parents or membership_root in root.parents:
        raise DataReadinessError("prospective poll parent path overlaps its output")
    memberships, membership_parent = _load_membership_authority(membership_root)
    _require_membership_observed_before_poll(
        membership_parent,
        poll_cutoff=_required_utc(request, "observed_at_utc"),
        maximum_age_seconds=int(request.get("maximum_continuous_gap_seconds", -1)),
    )
    expected_membership = {
        "membership_authority_sha256": membership_parent["authority_sha256"],
        "membership_manifest_sha256": membership_parent["manifest_sha256"],
        "membership_table_sha256": membership_parent["membership_table_sha256"],
        "membership_universe_sha256": membership_parent["universe_sha256"],
        "membership_cutoff_date": membership_parent["cutoff_date"],
    }
    if any(request.get(key) != value for key, value in expected_membership.items()):
        raise DataReadinessError("prospective poll membership parent changed")
    intraday_root = Path(str(request.get("intraday_bar_dataset_directory", ""))).resolve()
    namespace = _load_a43_security_namespace(
        intraday_root,
        membership_root=membership_root,
        membership_parent=membership_parent,
    )
    expected_namespace = {
        "intraday_bar_authority_sha256": namespace["intraday_bar_authority_sha256"],
        "intraday_bar_manifest_sha256": namespace["intraday_bar_manifest_sha256"],
        "intraday_bar_request_sha256": namespace["intraday_bar_request_sha256"],
        "intraday_bar_parent_lineage_sha256": namespace["intraday_bar_parent_lineage_sha256"],
        "security_identity_namespace_sha256": namespace["security_identity_namespace_sha256"],
    }
    if any(request.get(key) != value for key, value in expected_namespace.items()):
        raise DataReadinessError("prospective poll A4.3 namespace parent changed")
    if any(manifest.get(key) != value for key, value in expected_namespace.items()):
        raise DataReadinessError("prospective poll manifest namespace changed")
    registry_root = Path(str(request.get("registry_directory", ""))).resolve()
    claim = _verify_poll_registry(
        registry_root,
        poll_root=root,
        request=request,
        manifest=manifest,
        authority=authority,
    )
    if manifest.get("registry_claim_sha256") != claim["claim_file_sha256"]:
        raise DataReadinessError("prospective poll registry claim changed")
    previous_record = request.get("previous_poll")
    previous = preloaded_previous
    if previous_record is not None:
        if not isinstance(previous_record, Mapping):
            raise DataReadinessError("prospective previous-poll identity is malformed")
        previous_root = Path(str(previous_record.get("directory", ""))).resolve()
        if root == previous_root or root in previous_root.parents or previous_root in root.parents:
            raise DataReadinessError("prospective previous-poll path overlaps child output")
        if previous is None or previous.directory != previous_root:
            raise DataReadinessError("prospective previous-poll chain does not match its child")
        if (
            file_sha256(previous_root / "_authority.json") != previous_record.get("authority_sha256")
            or file_sha256(previous_root / "_manifest.json") != previous_record.get("manifest_sha256")
            or previous.manifest.get("observed_at_utc") != previous_record.get("observed_at_utc")
        ):
            raise DataReadinessError("prospective previous-poll lineage changed")
        previous_membership_root = Path(
            str(previous.request.get("membership_authority_directory", ""))
        ).resolve()
        previous_memberships, previous_membership_parent = (
            _load_membership_authority(previous_membership_root)
        )
        _require_membership_authority_chain(
            previous_memberships,
            previous_membership_parent,
            memberships,
            membership_parent,
        )
    elif previous is not None:
        raise DataReadinessError("prospective root poll has an unexpected parent")
    raw_assets_path = root / "raw_assets.json"
    raw_assets_body_path = root / "raw_assets.body"
    raw_pages_root = root / "raw_pages"
    attempts_root = root / "attempts"
    if (
        manifest.get("raw_assets_sha256") != file_sha256(raw_assets_path)
        or manifest.get("raw_assets_body_sha256") != file_sha256(raw_assets_body_path)
        or manifest.get("raw_pages_inventory_sha256") != _directory_inventory_sha256(raw_pages_root)
        or manifest.get("attempt_inventory_sha256") != _directory_inventory_sha256(attempts_root)
    ):
        raise DataReadinessError("prospective poll raw inventory does not verify")
    raw_assets = _json_object(raw_assets_path)
    if raw_assets.get("body_sha256") != file_sha256(raw_assets_body_path):
        raise DataReadinessError("prospective raw asset body does not verify")
    _verify_asset_request_url(
        str(raw_assets.get("requested_url", "")),
        final_url=str(raw_assets.get("final_url", "")),
        redirect_chain=raw_assets.get("redirect_chain"),
    )
    records = manifest.get("artifacts")
    sidecars = manifest.get("artifact_manifest_hashes")
    if not isinstance(records, Mapping) or not isinstance(sidecars, Mapping):
        raise DataReadinessError("prospective poll artifact inventory is malformed")
    loaded: dict[str, pd.DataFrame] = {}
    expected = {
        "assets": "prospective_alpaca_assets",
        "identity_audit": "prospective_security_identity_audit",
        "event_observations": "prospective_broker_action_observations",
        "source_collections": "source_collections",
    }
    for role, artifact_type in expected.items():
        record = records.get(role)
        if not isinstance(record, Mapping):
            raise DataReadinessError(f"prospective poll {role} inventory is missing")
        path = _resolve_inside(root, str(record.get("path", "")))
        if file_sha256(path) != record.get("sha256") or file_sha256(manifest_path_for(path)) != sidecars.get(role):
            raise DataReadinessError(f"prospective poll {role} hash does not verify")
        frame, child = load_canonical_artifact(path, expected_type=artifact_type, allow_research=True)
        child_inputs = child.get("inputs")
        if not isinstance(child_inputs, Mapping) or child_inputs.get("request_sha256") != request_sha256:
            raise DataReadinessError(f"prospective poll {role} lineage does not verify")
        loaded[role] = frame
    common_inputs = {
        "request_sha256": request_sha256,
        "registry_claim_sha256": str(claim["claim_file_sha256"]),
        "intraday_bar_authority_sha256": str(namespace["intraday_bar_authority_sha256"]),
        "security_identity_namespace_sha256": str(namespace["security_identity_namespace_sha256"]),
        "membership_authority_sha256": str(membership_parent["authority_sha256"]),
        "membership_table_sha256": str(membership_parent["membership_table_sha256"]),
        "raw_assets_sha256": file_sha256(raw_assets_path),
        "raw_assets_body_sha256": file_sha256(raw_assets_body_path),
        "raw_pages_inventory_sha256": _directory_inventory_sha256(raw_pages_root),
        "attempt_inventory_sha256": _directory_inventory_sha256(attempts_root),
    }
    for role in ("identity_audit", "event_observations", "source_collections"):
        path = _resolve_inside(root, str(cast(Mapping[str, object], records[role])["path"]))
        child = _json_object(manifest_path_for(path))
        if child.get("inputs") != common_inputs:
            raise DataReadinessError(f"prospective poll {role} input hashes do not verify")
    assets_path = _resolve_inside(root, str(cast(Mapping[str, object], records["assets"])["path"]))
    assets_child = _json_object(manifest_path_for(assets_path))
    if assets_child.get("inputs") != {
        "request_sha256": request_sha256,
        "raw_assets_envelope_sha256": file_sha256(raw_assets_path),
        "raw_assets_body_sha256": file_sha256(raw_assets_body_path),
    }:
        raise DataReadinessError("prospective asset input hashes do not verify")
    observations = loaded["event_observations"]
    if list(observations.columns) != list(_OBSERVATION_COLUMNS) or bool(observations["observation_id"].duplicated().any()):
        raise DataReadinessError("prospective event observation identity does not verify")
    checks = audit_source_collections(loaded["source_collections"], require_success=True)
    if any(check.status != "pass" for check in checks):
        raise DataReadinessError("prospective source collection coverage does not verify")
    assets = loaded["assets"]
    expected_assets = _normalize_assets(_asset_frame_from_body(raw_assets_body_path.read_bytes()))
    if not _frames_equal(assets, expected_assets):
        raise DataReadinessError("prospective assets do not replay from raw evidence")
    identity_received = _required_utc(raw_assets, "response_received_at_utc")
    poll_cutoff = _required_utc(request, "observed_at_utc")
    if not (poll_cutoff <= identity_received <= poll_cutoff + timedelta(seconds=int(request["maximum_continuous_gap_seconds"]))):
        raise DataReadinessError("prospective asset snapshot exceeded the continuous-coverage limit")
    expected_identity = _build_identity_audit(
        memberships,
        assets,
        observed_at=_required_utc(request, "observed_at_utc"),
        identity_observed_at=identity_received,
        membership_cutoff_date=str(membership_parent["cutoff_date"]),
        previous_identity=(previous.identity_audit if previous is not None else None),
    )
    if not _frames_equal(loaded["identity_audit"], expected_identity):
        raise DataReadinessError("prospective identity audit does not replay")
    symbols = tuple(sorted(expected_identity["ticker"].astype(str)))
    batches = [symbols[index : index + int(request["batch_size"])] for index in range(0, len(symbols), int(request["batch_size"]))]
    page_rows, batch_times = _replay_raw_pages(
        raw_pages_root,
        batches=batches,
        request_sha256=str(request_sha256),
        query_start=_required_utc(request, "query_start_utc"),
        query_end=_required_utc(request, "query_end_utc"),
        maximum_continuous_gap_seconds=int(request["maximum_continuous_gap_seconds"]),
    )
    _verify_attempt_inventory(attempts_root, request_sha256=str(request_sha256))
    expected_observations = _build_observations(
        page_rows,
        expected_identity,
        observed_at=_required_utc(request, "observed_at_utc"),
    )
    if not _frames_equal(observations, expected_observations):
        raise DataReadinessError("prospective event observations do not replay")
    expected_collections = _build_source_collections(
        batches=batches,
        batch_times=batch_times,
        observations=expected_observations,
        identity=expected_identity,
        observed_at=_required_utc(request, "observed_at_utc"),
        previous=previous,
        maximum_gap_seconds=int(request["maximum_continuous_gap_seconds"]),
    )
    if not _frames_equal(loaded["source_collections"], expected_collections):
        raise DataReadinessError("prospective source coverage does not replay")
    return ProspectivePoll(
        directory=root,
        request=request,
        manifest=manifest,
        observations=observations,
        source_collections=loaded["source_collections"],
        identity_audit=loaded["identity_audit"],
    )


def publish_prospective_broker_action_generation(
    *,
    poll_directories: Sequence[Path],
    output_directory: Path,
    memory_hard_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.75,
) -> dict[str, object]:
    """Compact immutable polls without losing first-seen or revision history."""

    output = output_directory.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with file_lock(output.with_name(f".{output.name}.publisher"), timeout=0.0):
            return _publish_prospective_broker_action_generation(
                poll_directories=poll_directories,
                output_directory=output,
                memory_hard_budget_gib=memory_hard_budget_gib,
                memory_headroom_gib=memory_headroom_gib,
            )
    except LockTimeout as exc:
        raise DataReadinessError(f"another process owns prospective generation {output}") from exc


def _publish_prospective_broker_action_generation(
    *,
    poll_directories: Sequence[Path],
    output_directory: Path,
    memory_hard_budget_gib: float,
    memory_headroom_gib: float,
) -> dict[str, object]:
    """Implementation executed under the generation-specific process lock."""

    if not poll_directories:
        raise ValueError("poll_directories must not be empty")
    if len(poll_directories) > 60:
        raise ValueError("one prospective generation may contain at most 60 polls")
    output = output_directory.resolve()
    polls: list[ProspectivePoll] = []
    input_bytes = 0
    for path in poll_directories:
        input_bytes += _poll_artifact_bytes(path.resolve())
        if input_bytes > MAX_GENERATION_INPUT_BYTES:
            raise DataReadinessError("prospective generation inputs exceed the bounded compaction limit")
        assert_memory_budget(
            hard_budget_gib=memory_hard_budget_gib,
            headroom_gib=memory_headroom_gib,
            stage="prospective generation before parent load",
        )
        polls.append(load_prospective_broker_action_poll(path))
        assert_memory_budget(
            hard_budget_gib=memory_hard_budget_gib,
            headroom_gib=memory_headroom_gib,
            stage="prospective generation after parent load",
        )
    polls.sort(key=lambda item: _required_utc(item.manifest, "observed_at_utc"))
    poll_roots = [poll.directory for poll in polls]
    if len(poll_roots) != len(set(poll_roots)):
        raise DataReadinessError("prospective generation contains duplicate poll directories")
    poll_cutoffs = [str(poll.manifest["observed_at_utc"]) for poll in polls]
    if len(poll_cutoffs) != len(set(poll_cutoffs)):
        raise DataReadinessError("prospective generation contains duplicate poll cutoffs")
    if any(output == root or output in root.parents or root in output.parents for root in poll_roots):
        raise DataReadinessError("prospective generation output and poll inputs must be disjoint")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise DataReadinessError(f"prospective generation output must be new and empty: {output}")
    assert_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective broker-action generation start",
    )
    poll_inventory = [
        {
            "directory": str(poll.directory),
            "observed_at_utc": poll.manifest["observed_at_utc"],
            "request_sha256": poll.request["request_sha256"],
            "manifest_sha256": file_sha256(poll.directory / "_manifest.json"),
            "authority_sha256": file_sha256(poll.directory / "_authority.json"),
        }
        for poll in polls
    ]
    request_payload: dict[str, object] = {
        "schema": GENERATION_REQUEST_SCHEMA,
        "polls": poll_inventory,
        "poll_inventory_sha256": json_sha256(poll_inventory),
        "availability_policy": "observed",
        "revision_identity": "provider_event_id|provider_updated_at_utc|raw_sha256",
        "identity_conflict_policy": "abstain",
    }
    request_sha256 = json_sha256(request_payload)
    request = {**request_payload, "request_sha256": request_sha256}
    _atomic_json(output / "_request.json", request)

    revisions = _compact_revision_observations(polls)
    collections = pd.concat([poll.source_collections for poll in polls], ignore_index=True).sort_values(
        ["scheduled_poll_at_utc", "ticker"], kind="stable"
    )
    identity_parts: list[pd.DataFrame] = []
    for poll in polls:
        part = poll.identity_audit.copy()
        part["poll_observed_at_utc"] = _required_utc(poll.manifest, "observed_at_utc")
        part["poll_authority_sha256"] = file_sha256(poll.directory / "_authority.json")
        identity_parts.append(part)
    identity = pd.concat(identity_parts, ignore_index=True).sort_values(["poll_observed_at_utc", "ticker"], kind="stable")
    assert_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective broker-action generation compaction",
    )
    revisions_path = output / "event_revisions.parquet"
    collections_path = output / "source_collections.parquet"
    identity_path = output / "identity_observations.parquet"
    child_inputs = {
        "request_sha256": request_sha256,
        "poll_inventory_sha256": str(request_payload["poll_inventory_sha256"]),
    }
    write_canonical_artifact(
        revisions,
        revisions_path,
        artifact_type="prospective_broker_action_revisions",
        audit=_passing_audit(
            "revision_history",
            len(revisions),
            "every provider content revision is retained with earliest observation",
        ),
        inputs=child_inputs,
        production_ready=False,
    )
    collection_audit = CanonicalAuditReport(checks=audit_source_collections(collections, require_success=True))
    write_canonical_artifact(
        collections,
        collections_path,
        artifact_type="source_collections",
        audit=collection_audit,
        inputs=child_inputs,
        production_ready=False,
    )
    write_canonical_artifact(
        identity,
        identity_path,
        artifact_type="prospective_security_identity_observations",
        audit=_passing_audit(
            "identity_observations",
            len(identity),
            "each poll preserves its independent identity eligibility result",
        ),
        inputs=child_inputs,
        production_ready=False,
    )
    artifacts = {
        "event_revisions": _artifact_record(revisions_path),
        "source_collections": _artifact_record(collections_path),
        "identity_observations": _artifact_record(identity_path),
    }
    artifact_manifests = {role: file_sha256(manifest_path_for(output / str(record["path"]))) for role, record in artifacts.items()}
    manifest: dict[str, object] = {
        "schema": GENERATION_MANIFEST_SCHEMA,
        "status": "complete",
        "request_sha256": request_sha256,
        "poll_inventory_sha256": request_payload["poll_inventory_sha256"],
        "first_poll_at_utc": polls[0].manifest["observed_at_utc"],
        "last_poll_at_utc": polls[-1].manifest["observed_at_utc"],
        "poll_count": len(polls),
        "revision_count": len(revisions),
        "provider_event_count": int(revisions["provider_event_id"].nunique()) if not revisions.empty else 0,
        "production_identity_revision_count": int(revisions["identity_eligible"].sum()) if not revisions.empty else 0,
        "source_collection_count": len(collections),
        "artifacts": artifacts,
        "artifact_manifest_hashes": artifact_manifests,
        "availability_policy": "observed",
        "training_eligible": False,
        "serving_eligible": False,
        "memory": memory_audit(
            hard_budget_gib=memory_hard_budget_gib,
            headroom_gib=memory_headroom_gib,
        ).to_record(),
    }
    assert_peak_memory_budget(
        hard_budget_gib=memory_hard_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="prospective broker-action generation publication",
    )
    _atomic_json(output / "_manifest.json", manifest)
    authority = {
        "schema": GENERATION_AUTHORITY_SCHEMA,
        "state": "complete",
        "artifact": "_manifest.json",
        "artifact_sha256": file_sha256(output / "_manifest.json"),
        "request_sha256": request_sha256,
        "training_eligible": False,
        "serving_eligible": False,
    }
    _atomic_json(output / "_authority.json", authority)
    load_prospective_broker_action_generation(output)
    return manifest


def load_prospective_broker_action_generation(
    directory: Path,
) -> ProspectiveGeneration:
    """Strictly replay one compacted prospective poll generation."""

    root = directory.resolve()
    if not root.is_dir() or {path.name for path in root.iterdir()} != _GENERATION_ROOT_FILES:
        raise DataReadinessError("prospective generation root inventory does not verify")
    request = _json_object(root / "_request.json")
    request_payload = {str(key): value for key, value in request.items() if key != "request_sha256"}
    request_sha256 = request.get("request_sha256")
    poll_records = request_payload.get("polls")
    if (
        request.get("schema") != GENERATION_REQUEST_SCHEMA
        or request_sha256 != json_sha256(request_payload)
        or not isinstance(poll_records, list)
        or not poll_records
        or request_payload.get("poll_inventory_sha256") != json_sha256(poll_records)
    ):
        raise DataReadinessError("prospective generation request does not verify")
    polls: list[ProspectivePoll] = []
    input_bytes = 0
    for raw_record in poll_records:
        if not isinstance(raw_record, Mapping):
            raise DataReadinessError("prospective generation poll record is malformed")
        poll_root = Path(str(raw_record.get("directory", ""))).resolve()
        input_bytes += _poll_artifact_bytes(poll_root)
        if input_bytes > MAX_GENERATION_INPUT_BYTES:
            raise DataReadinessError("prospective generation inputs exceed the bounded replay limit")
        assert_memory_budget(
            hard_budget_gib=4.0,
            headroom_gib=0.75,
            stage="prospective generation replay before parent load",
        )
        poll = load_prospective_broker_action_poll(poll_root)
        assert_memory_budget(
            hard_budget_gib=4.0,
            headroom_gib=0.75,
            stage="prospective generation replay after parent load",
        )
        polls.append(poll)
        if (
            file_sha256(poll.directory / "_manifest.json") != raw_record.get("manifest_sha256")
            or file_sha256(poll.directory / "_authority.json") != raw_record.get("authority_sha256")
            or poll.request.get("request_sha256") != raw_record.get("request_sha256")
        ):
            raise DataReadinessError("prospective generation poll lineage changed")
    manifest_path = root / "_manifest.json"
    manifest = _json_object(manifest_path)
    authority = _json_object(root / "_authority.json")
    if (
        manifest.get("schema") != GENERATION_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != GENERATION_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("request_sha256") != request_sha256
        or set(authority)
        != {
            "schema",
            "state",
            "artifact",
            "artifact_sha256",
            "request_sha256",
            "training_eligible",
            "serving_eligible",
        }
        or authority.get("artifact") != "_manifest.json"
        or authority.get("training_eligible") is not False
        or authority.get("serving_eligible") is not False
        or manifest.get("training_eligible") is not False
        or manifest.get("serving_eligible") is not False
    ):
        raise DataReadinessError("prospective generation authority does not verify")
    records = manifest.get("artifacts")
    sidecars = manifest.get("artifact_manifest_hashes")
    if not isinstance(records, Mapping) or not isinstance(sidecars, Mapping):
        raise DataReadinessError("prospective generation artifact inventory is malformed")
    expected = {
        "event_revisions": "prospective_broker_action_revisions",
        "source_collections": "source_collections",
        "identity_observations": "prospective_security_identity_observations",
    }
    loaded: dict[str, pd.DataFrame] = {}
    for role, artifact_type in expected.items():
        record = records.get(role)
        if not isinstance(record, Mapping):
            raise DataReadinessError(f"prospective generation {role} is missing")
        path = _resolve_inside(root, str(record.get("path", "")))
        if file_sha256(path) != record.get("sha256") or file_sha256(manifest_path_for(path)) != sidecars.get(role):
            raise DataReadinessError(f"prospective generation {role} hash changed")
        frame, child = load_canonical_artifact(path, expected_type=artifact_type, allow_research=True)
        inputs = child.get("inputs")
        if inputs != {
            "request_sha256": request_sha256,
            "poll_inventory_sha256": request_payload["poll_inventory_sha256"],
        }:
            raise DataReadinessError(f"prospective generation {role} lineage changed")
        loaded[role] = frame
    revisions = loaded["event_revisions"]
    if bool(revisions.duplicated(["revision_id", "ticker"]).any()):
        raise DataReadinessError("prospective generation contains duplicate revisions")
    if (
        int(manifest.get("poll_count", -1)) != len(polls)
        or int(manifest.get("revision_count", -1)) != len(revisions)
        or int(manifest.get("source_collection_count", -1)) != len(loaded["source_collections"])
    ):
        raise DataReadinessError("prospective generation counts do not verify")
    expected_revisions = _compact_revision_observations(polls)
    expected_collections = pd.concat([poll.source_collections for poll in polls], ignore_index=True).sort_values(
        ["scheduled_poll_at_utc", "ticker"], kind="stable"
    )
    expected_identity_parts: list[pd.DataFrame] = []
    for poll in polls:
        part = poll.identity_audit.copy()
        part["poll_observed_at_utc"] = _required_utc(poll.manifest, "observed_at_utc")
        part["poll_authority_sha256"] = file_sha256(poll.directory / "_authority.json")
        expected_identity_parts.append(part)
    expected_identity = pd.concat(expected_identity_parts, ignore_index=True).sort_values(["poll_observed_at_utc", "ticker"], kind="stable")
    if (
        not _frames_equal(revisions, expected_revisions)
        or not _frames_equal(loaded["source_collections"], expected_collections)
        or not _frames_equal(loaded["identity_observations"], expected_identity)
    ):
        raise DataReadinessError("prospective generation does not replay from its polls")
    return ProspectiveGeneration(
        directory=root,
        request=request,
        manifest=manifest,
        revisions=revisions,
        source_collections=loaded["source_collections"],
        identity_observations=loaded["identity_observations"],
    )


def _compact_revision_observations(
    polls: Sequence[ProspectivePoll],
) -> pd.DataFrame:
    frames = [poll.observations for poll in polls if not poll.observations.empty]
    extra_columns = (
        "event_first_seen_at_utc",
        "last_seen_at_utc",
        "identity_first_eligible_at_utc",
        "production_available_at_utc",
        "observation_count",
    )
    if not frames:
        return pd.DataFrame(columns=[*list(_OBSERVATION_COLUMNS), *extra_columns])
    data = pd.concat(frames, ignore_index=True)
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
    data["provider_timestamp_anomaly"] = data["provider_timestamp_anomaly"].astype(bool) | timestamp_collision
    invariant_columns = [
        "provider_event_id",
        "published_at_utc",
        "provider_updated_at_utc",
        "source",
        "title",
        "url",
        "summary",
        "text",
        "raw_sha256",
        "provider_timestamp_anomaly",
    ]
    event_first = data.groupby(["provider_event_id", "ticker"], sort=False)["revision_first_seen_at_utc"].min()
    rows: list[dict[str, object]] = []
    for (revision_id, ticker), group in data.groupby(["revision_id", "ticker"], sort=True):
        for column in invariant_columns:
            normalized = group[column].fillna("").astype(str)
            if normalized.nunique(dropna=False) != 1:
                raise DataReadinessError(f"provider revision changed invariant field {column}")
        security_ids = sorted(value for value in set(group["candidate_security_id"].astype(str)) if value)
        asset_ids = sorted(value for value in set(group["alpaca_asset_id"].astype(str)) if value)
        eligible_rows = group[group["identity_eligible"].astype(bool)]
        identity_conflict = len(security_ids) != 1 or len(asset_ids) != 1
        eligible = not eligible_rows.empty and not identity_conflict
        first_seen = pd.to_datetime(group["revision_first_seen_at_utc"], utc=True).min()
        last_seen = pd.to_datetime(group["revision_first_seen_at_utc"], utc=True).max()
        identity_first = pd.to_datetime(eligible_rows["identity_available_at_utc"], utc=True).min() if eligible else pd.NaT
        first = group.sort_values("revision_first_seen_at_utc", kind="stable").iloc[0]
        row = first.to_dict()
        row["observation_id"] = hashlib.sha256(f"{revision_id}|{ticker}".encode()).hexdigest()
        row["candidate_security_id"] = security_ids[0] if len(security_ids) == 1 else ""
        row["alpaca_asset_id"] = asset_ids[0] if len(asset_ids) == 1 else ""
        row["identity_eligible"] = eligible
        row["identity_ineligible_reason"] = (
            "" if eligible else "identity_changed_across_polls" if identity_conflict else "identity_never_eligible"
        )
        row["revision_first_seen_at_utc"] = first_seen
        row["event_first_seen_at_utc"] = event_first.loc[(str(first["provider_event_id"]), str(ticker))]
        row["last_seen_at_utc"] = last_seen
        row["identity_first_eligible_at_utc"] = identity_first
        row["production_available_at_utc"] = max(first_seen, identity_first) if eligible else pd.NaT
        row["observation_count"] = len(group)
        row["batch_id"] = "compacted"
        row["page_index"] = -1
        row["schema_version"] = REVISION_SCHEMA
        rows.append(row)
    columns = [*list(_OBSERVATION_COLUMNS), *extra_columns]
    return (
        pd.DataFrame(rows)
        .loc[:, columns]
        .sort_values(
            ["revision_first_seen_at_utc", "provider_event_id", "ticker"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _load_membership_authority(root: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    authority_path = root / "_authority.json"
    authority = _json_object(authority_path)
    if authority.get("schema") == OBSERVED_MEMBERSHIP_AUTHORITY_SCHEMA:
        loaded = load_observed_sp500_membership_authority(root)
        return loaded.memberships, dict(loaded.parent)
    memberships, parent = load_sp500_membership_authority_envelope(root)
    request = _json_object(root / "_request.json")
    lineage = request.get("parent_lineage")
    if not isinstance(lineage, Mapping) or set(lineage) != _MEMBERSHIP_PARENT_LINEAGE_KEYS:
        raise DataReadinessError("prospective poll closed membership lineage is invalid")
    return memberships, parent


def _require_membership_observed_before_poll(
    parent: Mapping[str, object],
    *,
    poll_cutoff: datetime,
    maximum_age_seconds: int,
) -> None:
    observed_value = parent.get("observed_at_utc")
    if observed_value is None:
        if _utc(poll_cutoff).astimezone(NEW_YORK).weekday() < 5:
            raise DataReadinessError(
                "closed S&P membership authority cannot authorize a weekday poll"
            )
        return
    authority_observed = _required_utc(parent, "observed_at_utc")
    age_seconds = (_utc(poll_cutoff) - authority_observed).total_seconds()
    if age_seconds < 0:
        raise DataReadinessError("observed membership authority is later than the poll cutoff")
    if age_seconds > maximum_age_seconds:
        raise DataReadinessError("observed membership authority is stale for the poll cutoff")
    next_effective_value = parent.get("next_pending_effective_at_utc")
    if next_effective_value is not None and _utc(
        poll_cutoff
    ) >= _required_utc(parent, "next_pending_effective_at_utc"):
        raise DataReadinessError(
            "observed membership authority does not cover the poll cutoff after a known change"
        )


def _require_membership_authority_progression(
    previous: Mapping[str, object],
    current: Mapping[str, object],
) -> None:
    previous_type = previous.get("authority_type")
    current_type = current.get("authority_type")
    if previous_type == "observed_time" and current_type != "observed_time":
        raise DataReadinessError(
            "prospective membership authority moved backward from observed to closed"
        )
    if previous_type != "observed_time" or current_type != "observed_time":
        return
    if (
        previous.get("base_membership_authority_sha256")
        != current.get("base_membership_authority_sha256")
    ):
        raise DataReadinessError(
            "observed membership authorities use different closed parents"
        )
    if _required_utc(current, "observed_at_utc") < _required_utc(
        previous, "observed_at_utc"
    ):
        raise DataReadinessError("observed membership authority moved backward in time")
    for field, label in (
        ("observed_release_outcomes", "release outcomes"),
        ("observed_events", "events"),
    ):
        previous_values = previous.get(field)
        current_values = current.get(field)
        if not isinstance(previous_values, list) or not isinstance(
            current_values, list
        ):
            raise DataReadinessError(
                f"observed membership {label} inventory is invalid"
            )
        previous_set = {json_sha256(value) for value in previous_values}
        current_set = {json_sha256(value) for value in current_values}
        if not previous_set.issubset(current_set):
            raise DataReadinessError(
                f"observed membership authority lost previously observed {label}"
            )


def _require_membership_authority_chain(
    previous_memberships: pd.DataFrame,
    previous_parent: Mapping[str, object],
    current_memberships: pd.DataFrame,
    current_parent: Mapping[str, object],
) -> None:
    _require_membership_authority_progression(previous_parent, current_parent)
    previous_cutoff = pd.Timestamp(str(previous_parent["cutoff_date"])).date()
    current_cutoff = pd.Timestamp(str(current_parent["cutoff_date"])).date()
    if current_cutoff < previous_cutoff:
        raise DataReadinessError("prospective membership authority moved backward")
    if current_cutoff > previous_cutoff:
        verify_membership_namespace_extension(
            previous_memberships,
            current_memberships,
            base_cutoff_date=previous_cutoff.isoformat(),
            current_cutoff_date=current_cutoff.isoformat(),
        )


def _load_a43_security_namespace(
    root: Path,
    *,
    membership_root: Path,
    membership_parent: Mapping[str, object],
) -> dict[str, str]:
    """Verify the A4.3 metadata that fixes the security identity namespace."""

    load_complete_intraday_bar_dataset(root)
    request_path = root / "_request.json"
    manifest_path = root / "_manifest.json"
    authority_path = root / "_authority.json"
    request = _json_object(request_path)
    manifest = _json_object(manifest_path)
    authority = _json_object(authority_path)
    request_payload = {str(key): value for key, value in request.items() if key != "request_sha256"}
    request_sha256 = json_sha256(request_payload)
    parent_lineage = request.get("parent_lineage")
    if (
        request.get("schema") != "edge_rebuild.intraday_bar_dataset.v1"
        or request.get("request_sha256") != request_sha256
        or manifest.get("schema") != "edge_rebuild.intraday_bar_dataset.v1"
        or manifest.get("state") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != "edge_rebuild.intraday_bar_dataset_authority.v1"
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("request_sha256") != request_sha256
        or not isinstance(parent_lineage, Mapping)
        or manifest.get("parent_lineage") != parent_lineage
        or request.get("parent_lineage_sha256") != json_sha256(parent_lineage)
        or manifest.get("parent_lineage_sha256") != json_sha256(parent_lineage)
    ):
        raise DataReadinessError("prospective poll A4.3 intraday dataset authority is invalid")
    base_membership_root = Path(str(request.get("membership_authority_directory", ""))).resolve()
    base_memberships, base_parent = _load_membership_authority(base_membership_root)
    expected_base_membership = {
        "membership_authority_sha256": base_parent["authority_sha256"],
        "membership_manifest_sha256": base_parent["manifest_sha256"],
        "membership_table_sha256": base_parent["membership_table_sha256"],
    }
    if any(parent_lineage.get(key) != value for key, value in expected_base_membership.items()):
        raise DataReadinessError("prospective poll membership authority is not the A4.3 identity namespace")
    current_memberships, _ = _load_membership_authority(membership_root)
    verify_membership_namespace_extension(
        base_memberships,
        current_memberships,
        base_cutoff_date=str(base_parent["cutoff_date"]),
        current_cutoff_date=str(membership_parent["cutoff_date"]),
    )
    namespace_payload = {
        "schema": SECURITY_NAMESPACE_SCHEMA,
        **expected_base_membership,
        "membership_universe_sha256": base_parent["universe_sha256"],
    }
    return {
        "intraday_bar_authority_sha256": file_sha256(authority_path),
        "intraday_bar_manifest_sha256": file_sha256(manifest_path),
        "intraday_bar_request_sha256": request_sha256,
        "intraday_bar_parent_lineage_sha256": json_sha256(parent_lineage),
        "security_identity_namespace_sha256": json_sha256(namespace_payload),
    }


def _normalize_assets(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"id", "symbol", "status", "exchange", "tradable"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataReadinessError(f"Alpaca asset snapshot is missing columns: {missing}")
    data = frame.loc[:, sorted(required)].copy()
    data["id"] = data["id"].fillna("").astype(str).str.strip()
    data["symbol"] = data["symbol"].map(lambda value: canonical_symbol(str(value)))
    data["status"] = data["status"].fillna("").astype(str).str.lower().str.strip()
    data["exchange"] = data["exchange"].fillna("").astype(str).str.upper().str.strip()
    data["tradable"] = data["tradable"].astype(bool)
    if bool(data["id"].eq("").any()) or bool(data.duplicated("symbol", keep=False).any()) or bool(data.duplicated("id", keep=False).any()):
        raise DataReadinessError("Alpaca asset snapshot has blank or ambiguous identity")
    return data.sort_values("symbol", kind="stable").reset_index(drop=True)


def _build_identity_audit(
    memberships: pd.DataFrame,
    assets: pd.DataFrame,
    *,
    observed_at: datetime,
    identity_observed_at: datetime,
    membership_cutoff_date: str,
    previous_identity: pd.DataFrame | None,
) -> pd.DataFrame:
    required = {
        "ticker",
        "security_id",
        "effective_from_utc",
        "effective_to_utc",
        "available_at_utc",
    }
    missing = sorted(required.difference(memberships.columns))
    if missing:
        raise DataReadinessError(f"membership identity table is missing columns: {missing}")
    data = memberships.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["effective_from_utc"] = pd.to_datetime(data["effective_from_utc"], utc=True, errors="coerce")
    data["effective_to_utc"] = pd.to_datetime(data["effective_to_utc"], utc=True, errors="coerce")
    data["available_at_utc"] = pd.to_datetime(data["available_at_utc"], utc=True, errors="coerce")
    if bool(data[["effective_from_utc", "available_at_utc"]].isna().any().any()):
        raise DataReadinessError("membership identity timestamps are invalid")
    point = pd.Timestamp(observed_at)
    active = data[data["effective_from_utc"].le(point) & (data["effective_to_utc"].isna() | data["effective_to_utc"].gt(point))].copy()
    if active.empty:
        raise DataReadinessError("membership authority has no intervals active at poll time")
    asset_index = assets.set_index("symbol", drop=False).to_dict("index")
    cutoff = pd.Timestamp(membership_cutoff_date).date()
    new_york_date = _utc(observed_at).astimezone(NEW_YORK).date()
    required_cutoff = new_york_date - timedelta(days=1) if new_york_date.weekday() >= 5 else new_york_date
    stale = cutoff < required_cutoff
    previous_index = previous_identity.set_index("ticker", drop=False).to_dict("index") if previous_identity is not None else {}
    rows: list[dict[str, object]] = []
    for ticker, group in active.groupby("ticker", sort=True):
        candidate_ids = sorted(set(group["security_id"].astype(str)))
        asset = asset_index.get(str(ticker))
        reason = ""
        if stale:
            reason = "membership_authority_stale"
        elif bool(group["available_at_utc"].gt(point).any()):
            reason = "membership_identity_not_available_at_poll"
        elif len(candidate_ids) != 1:
            reason = "ambiguous_membership_identity"
        elif len(group) != 1:
            reason = "multiple_active_membership_intervals"
        elif asset is None:
            reason = "alpaca_asset_missing"
        elif str(asset["status"]) != "active" or not bool(asset["tradable"]):
            reason = "alpaca_asset_inactive_or_not_tradable"
        previous = previous_index.get(str(ticker))
        previous_quarantined = bool(previous is not None and previous.get("identity_quarantined", False))
        last_security_id = str(previous.get("last_accepted_security_id", "")) if previous is not None else ""
        last_asset_id = str(previous.get("last_accepted_alpaca_asset_id", "")) if previous is not None else ""
        if previous is not None and bool(previous.get("identity_eligible")):
            last_security_id = str(previous.get("candidate_security_id", ""))
            last_asset_id = str(previous.get("alpaca_asset_id", ""))
        if (
            reason == ""
            and previous is not None
            and (
                previous_quarantined
                or (bool(last_security_id) and (last_security_id != candidate_ids[0] or last_asset_id != str(asset["id"])))
            )
        ):
            reason = "unresolved_identity_change_from_previous_poll"
        eligible = reason == ""
        if eligible:
            last_security_id = candidate_ids[0]
            last_asset_id = str(asset["id"])
        quarantined = previous_quarantined or reason.startswith("unresolved_identity_change")
        rows.append(
            {
                "ticker": str(ticker),
                "candidate_security_id": candidate_ids[0] if len(candidate_ids) == 1 else "",
                "alpaca_asset_id": "" if asset is None else str(asset["id"]),
                "membership_interval_count": len(group),
                "membership_cutoff_date": membership_cutoff_date,
                "identity_observed_at_utc": identity_observed_at,
                "identity_eligible": eligible,
                "identity_ineligible_reason": reason,
                "last_accepted_security_id": last_security_id,
                "last_accepted_alpaca_asset_id": last_asset_id,
                "identity_quarantined": quarantined,
                "schema_version": IDENTITY_AUDIT_SCHEMA,
            }
        )
    return pd.DataFrame(rows).sort_values("ticker", kind="stable").reset_index(drop=True)


def _collect_batch_pages(
    *,
    batch_id: str,
    symbols: Sequence[str],
    query_start: datetime,
    query_end: datetime,
    request_sha256: str,
    raw_pages_root: Path,
    fetch_page: NewsPageFetcher,
    clock: Clock,
    memory_hard_budget_gib: float,
    memory_headroom_gib: float,
    maximum_continuous_gap_seconds: int,
) -> tuple[list[tuple[int, datetime, dict[str, Any]]], datetime, datetime]:
    directory = raw_pages_root / batch_id
    directory.mkdir(parents=True, exist_ok=True)
    pages: list[tuple[int, datetime, dict[str, Any]]] = []
    token: str | None = None
    seen_tokens: set[str] = set()
    started_at: datetime | None = None
    page_index = 0
    total_body_bytes = 0
    while True:
        if page_index >= MAX_PAGES_PER_BATCH:
            raise DataReadinessError(f"Alpaca news pagination exceeded {MAX_PAGES_PER_BATCH} pages for one batch")
        assert_memory_budget(
            hard_budget_gib=memory_hard_budget_gib,
            headroom_gib=memory_headroom_gib,
            stage=f"prospective news pagination {batch_id}",
        )
        path = directory / f"page_{page_index:06d}.json"
        body_path = directory / f"page_{page_index:06d}.body"
        if path.exists():
            if not body_path.exists():
                raise DataReadinessError(f"archived prospective page body is missing: {body_path}")
            payload = _json_object(path)
            if (
                payload.get("request_sha256") != request_sha256
                or payload.get("batch_id") != batch_id
                or payload.get("symbols") != list(symbols)
                or payload.get("request_page_token") != token
                or payload.get("body_sha256") != file_sha256(body_path)
            ):
                raise DataReadinessError(f"archived prospective page identity mismatch: {path}")
            parsed = _news_payload_from_body(body_path.read_bytes())
            if parsed.get("news") != payload.get("news") or parsed.get("next_page_token") != payload.get("next_page_token"):
                raise DataReadinessError(f"archived prospective page does not replay: {path}")
            _verify_news_request_url(
                str(payload.get("requested_url", "")),
                final_url=str(payload.get("final_url", "")),
                redirect_chain=payload.get("redirect_chain"),
                symbols=symbols,
                query_start=query_start,
                query_end=query_end,
                page_token=token,
            )
            received = _required_utc(payload, "response_received_at_utc")
            page_started = _required_utc(payload, "request_started_at_utc")
        else:
            if body_path.exists():
                body_path.unlink()
            page_started = _utc(clock())
            page = fetch_page(",".join(symbols), query_start, query_end, token)
            if (
                page.raw_body is None
                or page.requested_url is None
                or page.final_url is None
                or page.status_code != 200
                or page.retrieved_at_utc is None
            ):
                raise DataReadinessError("prospective Alpaca page lacks raw HTTP evidence")
            _verify_news_request_url(
                page.requested_url,
                final_url=page.final_url,
                redirect_chain=page.redirect_chain,
                symbols=symbols,
                query_start=query_start,
                query_end=query_end,
                page_token=token,
            )
            received = _utc(page.retrieved_at_utc)
            if page.request_page_token != token:
                raise DataReadinessError("Alpaca news response page token does not match request")
            parsed = _news_payload_from_body(page.raw_body)
            parsed_news = parsed.get("news")
            if (
                not isinstance(parsed_news, list)
                or tuple(parsed_news) != page.news
                or parsed.get("next_page_token") != page.next_page_token
            ):
                raise DataReadinessError("parsed Alpaca news page differs from raw HTTP body")
            _atomic_bytes(body_path, page.raw_body)
            payload = {
                "request_sha256": request_sha256,
                "batch_id": batch_id,
                "symbols": list(symbols),
                "request_page_token": token,
                "next_page_token": page.next_page_token,
                "request_started_at_utc": page_started.isoformat(),
                "response_received_at_utc": received.isoformat(),
                "news": [dict(item) for item in page.news],
                "response_headers": dict(page.response_headers),
                "requested_url": page.requested_url,
                "final_url": page.final_url,
                "redirect_chain": list(page.redirect_chain),
                "status_code": page.status_code,
                "body_sha256": file_sha256(body_path),
                "body_bytes": body_path.stat().st_size,
            }
            _atomic_json(path, payload)
        if not (query_end <= page_started <= received <= query_end + timedelta(seconds=maximum_continuous_gap_seconds)):
            raise DataReadinessError("prospective Alpaca response timing is outside its poll window")
        total_body_bytes += body_path.stat().st_size
        if total_body_bytes > MAX_BYTES_PER_BATCH:
            raise DataReadinessError("Alpaca news pagination exceeded 32 MiB for one batch")
        if started_at is None or page_started < started_at:
            started_at = page_started
        pages.append((page_index, received, payload))
        next_token_raw = payload.get("next_page_token")
        next_token = str(next_token_raw).strip() if next_token_raw is not None else ""
        if not next_token:
            if started_at is None:
                raise DataReadinessError("prospective batch has no request start time")
            return pages, started_at, received
        if next_token in seen_tokens:
            raise DataReadinessError("Alpaca news pagination repeated a page token")
        seen_tokens.add(next_token)
        token = next_token
        page_index += 1


def _build_observations(
    pages: Sequence[tuple[str, int, datetime, dict[str, Any]]],
    identity: pd.DataFrame,
    *,
    observed_at: datetime,
) -> pd.DataFrame:
    identity_index = identity.set_index("ticker", drop=False).to_dict("index")
    rows: list[dict[str, object]] = []
    for batch_id, page_index, received, payload in pages:
        news = payload.get("news")
        if not isinstance(news, list) or any(not isinstance(item, Mapping) for item in news):
            raise DataReadinessError("prospective Alpaca page has malformed news rows")
        for raw_item in news:
            item = {str(key): value for key, value in raw_item.items()}
            provider_id = str(item.get("id", "")).strip()
            published = pd.to_datetime(item.get("created_at"), utc=True, errors="coerce")
            updated = pd.to_datetime(item.get("updated_at"), utc=True, errors="coerce")
            title = str(item.get("headline") or "").strip()
            if not provider_id or pd.isna(published) or not title:
                raise DataReadinessError("prospective Alpaca event lacks provider identity, publication time, or title")
            if pd.Timestamp(published) > pd.Timestamp(observed_at):
                raise DataReadinessError("prospective Alpaca event is published after the poll cutoff")
            raw_json = json.dumps(item, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            raw_sha = hashlib.sha256(raw_json.encode("utf-8")).hexdigest()
            update_text = "" if pd.isna(updated) else pd.Timestamp(updated).isoformat()
            revision_id = hashlib.sha256(f"{provider_id}|{update_text}|{raw_sha}".encode()).hexdigest()
            symbols_raw = item.get("symbols")
            symbols = sorted({canonical_symbol(str(value)) for value in symbols_raw}) if isinstance(symbols_raw, list) else []
            requested_raw = payload.get("symbols")
            requested = {canonical_symbol(str(value)) for value in requested_raw} if isinstance(requested_raw, list) else set()
            for ticker in sorted(set(symbols).intersection(requested)):
                audit = identity_index.get(ticker)
                if audit is None:
                    continue
                observation_id = hashlib.sha256(f"{revision_id}|{ticker}|{received.isoformat()}".encode()).hexdigest()
                rows.append(
                    {
                        "observation_id": observation_id,
                        "revision_id": revision_id,
                        "provider_event_id": provider_id,
                        "ticker": ticker,
                        "candidate_security_id": str(audit["candidate_security_id"]),
                        "alpaca_asset_id": str(audit["alpaca_asset_id"]),
                        "identity_available_at_utc": audit["identity_observed_at_utc"],
                        "identity_eligible": bool(audit["identity_eligible"]),
                        "identity_ineligible_reason": str(audit["identity_ineligible_reason"]),
                        "published_at_utc": pd.Timestamp(published),
                        "provider_updated_at_utc": pd.NaT if pd.isna(updated) else pd.Timestamp(updated),
                        "revision_first_seen_at_utc": received,
                        "source": f"alpaca:{item.get('source') or 'unknown'}",
                        "title": title,
                        "url": str(item.get("url") or ""),
                        "summary": str(item.get("summary") or ""),
                        "text": str(item.get("content") or item.get("summary") or ""),
                        "raw_sha256": raw_sha,
                        "provider_timestamp_anomaly": bool(pd.notna(updated) and pd.Timestamp(updated) < pd.Timestamp(published)),
                        "batch_id": batch_id,
                        "page_index": page_index,
                        "schema_version": OBSERVATION_SCHEMA,
                    }
                )
    if not rows:
        return pd.DataFrame(columns=list(_OBSERVATION_COLUMNS))
    frame = pd.DataFrame(rows).sort_values(["revision_first_seen_at_utc", "provider_event_id", "ticker"], kind="stable")
    if bool(frame["observation_id"].duplicated().any()):
        raise DataReadinessError("prospective poll contains duplicate event observations")
    return frame.loc[:, list(_OBSERVATION_COLUMNS)].reset_index(drop=True)


def _build_source_collections(
    *,
    batches: Sequence[Sequence[str]],
    batch_times: Mapping[str, tuple[datetime, datetime]],
    observations: pd.DataFrame,
    identity: pd.DataFrame,
    observed_at: datetime,
    previous: ProspectivePoll | None,
    maximum_gap_seconds: int,
) -> pd.DataFrame:
    previous_at = _required_utc(previous.manifest, "observed_at_utc") if previous is not None else observed_at
    gap = (observed_at - previous_at).total_seconds()
    gap_is_continuous = previous is not None and 0 < gap <= maximum_gap_seconds
    previous_collections = (
        previous.source_collections.set_index("ticker", drop=False).to_dict("index")
        if previous is not None
        else {}
    )
    previous_identities = (
        previous.identity_audit.set_index("ticker", drop=False).to_dict("index")
        if previous is not None
        else {}
    )
    current_identities = identity.set_index("ticker", drop=False).to_dict("index")
    event_counts = observations.groupby("ticker").size().to_dict() if not observations.empty else {}
    rows: list[dict[str, object]] = []
    for index, symbols in enumerate(batches):
        batch_id = f"batch-{index:04d}"
        started, completed = batch_times[batch_id]
        for ticker in symbols:
            previous_collection = previous_collections.get(str(ticker))
            previous_identity = previous_identities.get(str(ticker))
            current_identity = current_identities.get(str(ticker))
            continuous = bool(
                gap_is_continuous
                and previous_collection is not None
                and str(previous_collection.get("status"))
                in {"observed", "observed_empty"}
                and previous_identity is not None
                and current_identity is not None
                and bool(previous_identity.get("identity_eligible"))
                and bool(current_identity.get("identity_eligible"))
                and previous_identity.get("candidate_security_id")
                == current_identity.get("candidate_security_id")
                and previous_identity.get("alpaca_asset_id")
                == current_identity.get("alpaca_asset_id")
            )
            coverage_start = previous_at if continuous else observed_at
            count = int(event_counts.get(ticker, 0))
            collection = SourceCollection(
                collection_id=f"alpaca-prospective-{hashlib.sha256(f'{batch_id}|{ticker}|{observed_at.isoformat()}'.encode()).hexdigest()}",
                ticker=str(ticker),
                source_family="alpaca",
                requested_start_utc=coverage_start,
                requested_end_utc=observed_at,
                started_at_utc=started,
                completed_at_utc=completed,
                status="observed" if count else "observed_empty",
                row_count=count,
            )
            rows.append(
                {
                    **collection.model_dump(),
                    "scheduled_poll_at_utc": observed_at,
                    "continuous_from_previous_poll": continuous,
                    "previous_poll_at_utc": previous_at if previous is not None else pd.NaT,
                    "batch_id": batch_id,
                }
            )
    return pd.DataFrame(rows).sort_values("ticker", kind="stable").reset_index(drop=True)


def _previous_identity(previous: ProspectivePoll | None) -> dict[str, object] | None:
    if previous is None:
        return None
    return {
        "directory": str(previous.directory),
        "authority_sha256": file_sha256(previous.directory / "_authority.json"),
        "manifest_sha256": file_sha256(previous.directory / "_manifest.json"),
        "observed_at_utc": previous.manifest["observed_at_utc"],
    }


def _registry_paths(root: Path, observed_at: datetime) -> tuple[Path, Path]:
    cutoff = _utc(observed_at)
    relative = Path(f"{cutoff:%Y}") / f"{cutoff:%m}" / f"{cutoff:%d}" / f"{cutoff:%Y%m%dT%H%M%SZ}.json"
    return root / "claims" / relative, root / "commits" / relative


def _validate_poll_paths(
    *,
    output: Path,
    membership: Path,
    intraday_dataset: Path,
    registry: Path,
    previous: Path | None,
) -> None:
    named = {
        "output": output,
        "membership": membership,
        "intraday_dataset": intraday_dataset,
        "registry": registry,
    }
    if previous is not None:
        named["previous"] = previous
    items = list(named.items())
    for index, (left_name, left) in enumerate(items):
        for right_name, right in items[index + 1 :]:
            if left == right or left in right.parents or right in left.parents:
                raise DataReadinessError(f"prospective poll paths overlap: {left_name}={left} {right_name}={right}")


def _claim_poll_cutoff(
    root: Path,
    *,
    output: Path,
    request: Mapping[str, object],
    namespace: Mapping[str, str],
    membership_parent: Mapping[str, object],
    previous: ProspectivePoll | None,
    claimed_at: datetime,
) -> dict[str, object]:
    observed = _required_utc(request, "observed_at_utc")
    claim_path, _ = _registry_paths(root, observed)
    payload: dict[str, object] = {
        "schema": "edge_rebuild.prospective_broker_action_cutoff_claim.v1",
        "cutoff_utc": observed.isoformat(),
        "poll_output_directory": str(output),
        "poll_request_sha256": request["request_sha256"],
        "intraday_bar_authority_sha256": namespace["intraday_bar_authority_sha256"],
        "security_identity_namespace_sha256": namespace["security_identity_namespace_sha256"],
        "membership_authority_sha256": membership_parent["authority_sha256"],
        "membership_table_sha256": membership_parent["membership_table_sha256"],
        "previous_poll_authority_sha256": (file_sha256(previous.directory / "_authority.json") if previous is not None else None),
    }
    with file_lock(root / "_registry"):
        if claim_path.exists():
            claim = _json_object(claim_path)
            stable = {str(key): value for key, value in claim.items() if key not in {"claimed_at_utc", "record_sha256"}}
            if stable != payload or claim.get("record_sha256") != json_sha256(
                {str(key): value for key, value in claim.items() if key != "record_sha256"}
            ):
                raise DataReadinessError("prospective cutoff is already claimed by a different poll")
        else:
            claim_payload = {**payload, "claimed_at_utc": claimed_at.isoformat()}
            claim = {
                **claim_payload,
                "record_sha256": json_sha256(claim_payload),
            }
            _write_new_json(claim_path, claim)
    return {**claim, "claim_file_sha256": file_sha256(claim_path)}


def _commit_poll_cutoff(root: Path, *, poll_root: Path) -> None:
    request = _json_object(poll_root / "_request.json")
    manifest = _json_object(poll_root / "_manifest.json")
    authority = _json_object(poll_root / "_authority.json")
    claim_path, commit_path = _registry_paths(root, _required_utc(request, "observed_at_utc"))
    claim = _json_object(claim_path)
    claim_payload = {str(key): value for key, value in claim.items() if key != "record_sha256"}
    if (
        claim.get("record_sha256") != json_sha256(claim_payload)
        or claim.get("poll_output_directory") != str(poll_root)
        or claim.get("poll_request_sha256") != request.get("request_sha256")
        or authority.get("registry_claim_sha256") != file_sha256(claim_path)
    ):
        raise DataReadinessError("prospective cutoff claim does not match the published poll")
    payload = {
        "schema": "edge_rebuild.prospective_broker_action_cutoff_commit.v1",
        "cutoff_utc": request["observed_at_utc"],
        "poll_output_directory": str(poll_root),
        "poll_request_sha256": request["request_sha256"],
        "claim_file_sha256": file_sha256(claim_path),
        "poll_manifest_sha256": file_sha256(poll_root / "_manifest.json"),
        "poll_authority_sha256": file_sha256(poll_root / "_authority.json"),
        "raw_pages_inventory_sha256": manifest["raw_pages_inventory_sha256"],
        "committed_at_utc": manifest["completed_at_utc"],
    }
    record = {**payload, "record_sha256": json_sha256(payload)}
    with file_lock(root / "_registry"):
        if commit_path.exists():
            if _json_object(commit_path) != record:
                raise DataReadinessError("prospective cutoff commit differs from the published poll")
        else:
            _write_new_json(commit_path, record)


def _verify_poll_registry(
    root: Path,
    *,
    poll_root: Path,
    request: Mapping[str, object],
    manifest: Mapping[str, object],
    authority: Mapping[str, object],
) -> dict[str, object]:
    if not root.is_dir() or root.is_symlink():
        raise DataReadinessError("prospective poll registry is unavailable")
    claim_path, commit_path = _registry_paths(root, _required_utc(request, "observed_at_utc"))
    claim = _json_object(claim_path)
    claim_payload = {str(key): value for key, value in claim.items() if key != "record_sha256"}
    commit = _json_object(commit_path)
    commit_payload = {str(key): value for key, value in commit.items() if key != "record_sha256"}
    previous_record = request.get("previous_poll")
    expected_previous_authority = previous_record.get("authority_sha256") if isinstance(previous_record, Mapping) else None
    if (
        claim.get("schema") != "edge_rebuild.prospective_broker_action_cutoff_claim.v1"
        or claim.get("record_sha256") != json_sha256(claim_payload)
        or claim.get("cutoff_utc") != request.get("observed_at_utc")
        or claim.get("poll_output_directory") != str(poll_root)
        or claim.get("poll_request_sha256") != request.get("request_sha256")
        or claim.get("intraday_bar_authority_sha256") != request.get("intraday_bar_authority_sha256")
        or claim.get("security_identity_namespace_sha256") != request.get("security_identity_namespace_sha256")
        or claim.get("membership_authority_sha256") != request.get("membership_authority_sha256")
        or claim.get("membership_table_sha256") != request.get("membership_table_sha256")
        or claim.get("previous_poll_authority_sha256") != expected_previous_authority
        or commit.get("schema") != "edge_rebuild.prospective_broker_action_cutoff_commit.v1"
        or commit.get("record_sha256") != json_sha256(commit_payload)
        or commit.get("claim_file_sha256") != file_sha256(claim_path)
        or commit.get("poll_output_directory") != str(poll_root)
        or commit.get("poll_request_sha256") != request.get("request_sha256")
        or commit.get("poll_manifest_sha256") != file_sha256(poll_root / "_manifest.json")
        or commit.get("poll_authority_sha256") != file_sha256(poll_root / "_authority.json")
        or commit.get("raw_pages_inventory_sha256") != manifest.get("raw_pages_inventory_sha256")
        or authority.get("registry_claim_sha256") != file_sha256(claim_path)
    ):
        raise DataReadinessError("prospective poll cutoff registry does not verify")
    _required_utc(claim, "claimed_at_utc")
    _required_utc(commit, "committed_at_utc")
    return {**claim, "claim_file_sha256": file_sha256(claim_path)}


def _resume_completed_poll(
    output: Path,
    *,
    registry_root: Path,
    membership_root: Path,
    intraday_dataset_root: Path,
) -> dict[str, object]:
    request = _json_object(output / "_request.json")
    expected_directories = {
        "registry_directory": registry_root,
        "membership_authority_directory": membership_root,
        "intraday_bar_dataset_directory": intraday_dataset_root,
    }
    if any(Path(str(request.get(key, ""))).resolve() != expected for key, expected in expected_directories.items()):
        raise DataReadinessError("completed prospective poll parent changed")
    _commit_poll_cutoff(registry_root, poll_root=output)
    return dict(load_prospective_broker_action_poll(output).manifest)


def _artifact_record(path: Path) -> dict[str, object]:
    return {"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}


def _poll_artifact_bytes(root: Path) -> int:
    manifest = _json_object(root / "_manifest.json")
    records = manifest.get("artifacts")
    if not isinstance(records, Mapping):
        raise DataReadinessError("prospective poll artifact inventory is malformed")
    total = 0
    for record in records.values():
        if not isinstance(record, Mapping):
            raise DataReadinessError("prospective poll artifact record is malformed")
        path = _resolve_inside(root, str(record.get("path", "")))
        actual = path.stat().st_size
        if int(record.get("bytes", -1)) != actual:
            raise DataReadinessError("prospective poll artifact byte count changed")
        parquet = pq.ParquetFile(path)  # type: ignore[no-untyped-call]
        total += sum(parquet.metadata.row_group(index).total_byte_size for index in range(parquet.metadata.num_row_groups))
    return total


def _passing_audit(name: str, rows: int, detail: str) -> CanonicalAuditReport:
    return CanonicalAuditReport(checks=(CanonicalAuditCheck(name=name, status="pass", failures=0, rows_checked=rows, detail=detail),))


def _asset_frame_from_body(body: bytes) -> pd.DataFrame:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataReadinessError("Alpaca asset HTTP body is not valid UTF-8 JSON") from exc
    if not isinstance(payload, list) or any(not isinstance(item, dict) for item in payload):
        raise DataReadinessError("Alpaca asset HTTP body must be an array of objects")
    return pd.DataFrame(payload)


def _news_payload_from_body(body: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataReadinessError("Alpaca news HTTP body is not valid UTF-8 JSON") from exc
    if not isinstance(payload, dict):
        raise DataReadinessError("Alpaca news HTTP body must be an object")
    news = payload.get("news", [])
    if not isinstance(news, list) or any(not isinstance(item, dict) for item in news):
        raise DataReadinessError("Alpaca news HTTP body has malformed news rows")
    next_raw = payload.get("next_page_token")
    next_token = str(next_raw).strip() if next_raw is not None and str(next_raw).strip() else None
    return {
        "news": [{str(key): value for key, value in item.items()} for item in news],
        "next_page_token": next_token,
    }


def _replay_raw_pages(
    root: Path,
    *,
    batches: Sequence[Sequence[str]],
    request_sha256: str,
    query_start: datetime,
    query_end: datetime,
    maximum_continuous_gap_seconds: int,
) -> tuple[
    list[tuple[str, int, datetime, dict[str, Any]]],
    dict[str, tuple[datetime, datetime]],
]:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise DataReadinessError("prospective raw-page inventory contains a symlink")
    expected_directories = {f"batch-{index:04d}" for index in range(len(batches))}
    actual_directories = {path.name for path in root.iterdir() if path.is_dir()}
    if actual_directories != expected_directories or any(path.is_file() for path in root.iterdir()):
        raise DataReadinessError("prospective raw-page batch inventory is invalid")
    rows: list[tuple[str, int, datetime, dict[str, Any]]] = []
    times: dict[str, tuple[datetime, datetime]] = {}
    poll_body_bytes = 0
    for index, symbols in enumerate(batches):
        batch_id = f"batch-{index:04d}"
        directory = root / batch_id
        files = {path.name for path in directory.iterdir() if path.is_file()}
        json_files = sorted(name for name in files if name.endswith(".json"))
        body_files = sorted(name for name in files if name.endswith(".body"))
        if len(json_files) != len(body_files) or not json_files or len(json_files) > MAX_PAGES_PER_BATCH:
            raise DataReadinessError(f"prospective raw-page pair inventory is invalid for {batch_id}")
        expected_json = [f"page_{page:06d}.json" for page in range(len(json_files))]
        expected_body = [f"page_{page:06d}.body" for page in range(len(body_files))]
        if json_files != expected_json or body_files != expected_body or files != set(expected_json + expected_body):
            raise DataReadinessError(f"prospective raw pages are not contiguous for {batch_id}")
        token: str | None = None
        total_body_bytes = 0
        started_values: list[datetime] = []
        received_values: list[datetime] = []
        for page_index, name in enumerate(json_files):
            envelope = _json_object(directory / name)
            body_path = directory / expected_body[page_index]
            total_body_bytes += body_path.stat().st_size
            poll_body_bytes += body_path.stat().st_size
            if total_body_bytes > MAX_BYTES_PER_BATCH:
                raise DataReadinessError(f"prospective raw pages exceed the byte limit for {batch_id}")
            if poll_body_bytes > MAX_BYTES_PER_POLL:
                raise DataReadinessError("prospective raw pages exceed the poll byte limit")
            if (
                envelope.get("request_sha256") != request_sha256
                or envelope.get("batch_id") != batch_id
                or envelope.get("symbols") != list(symbols)
                or envelope.get("request_page_token") != token
                or envelope.get("body_sha256") != file_sha256(body_path)
                or int(envelope.get("body_bytes", -1)) != body_path.stat().st_size
                or envelope.get("status_code") != 200
            ):
                raise DataReadinessError(f"prospective raw-page envelope is invalid for {name}")
            _verify_news_request_url(
                str(envelope.get("requested_url", "")),
                final_url=str(envelope.get("final_url", "")),
                redirect_chain=envelope.get("redirect_chain"),
                symbols=symbols,
                query_start=query_start,
                query_end=query_end,
                page_token=token,
            )
            parsed = _news_payload_from_body(body_path.read_bytes())
            if parsed["news"] != envelope.get("news") or parsed["next_page_token"] != envelope.get("next_page_token"):
                raise DataReadinessError(f"prospective raw-page body mismatch for {name}")
            started = _required_utc(envelope, "request_started_at_utc")
            received = _required_utc(envelope, "response_received_at_utc")
            if not (query_end <= started <= received <= query_end + timedelta(seconds=maximum_continuous_gap_seconds)):
                raise DataReadinessError("prospective response timing is outside its poll window")
            started_values.append(started)
            received_values.append(received)
            rows.append((batch_id, page_index, received, envelope))
            next_raw = envelope.get("next_page_token")
            token = str(next_raw).strip() if next_raw is not None else ""
            if not token and page_index != len(json_files) - 1:
                raise DataReadinessError("prospective raw page exists after terminal token")
            token = token or None
        if token is not None:
            raise DataReadinessError("prospective raw-page chain has no terminal page")
        if (max(received_values) - query_end).total_seconds() > maximum_continuous_gap_seconds:
            raise DataReadinessError(f"prospective raw pages exceed the completion limit for {batch_id}")
        times[batch_id] = (min(started_values), max(received_values))
    return rows, times


def _write_attempt(
    root: Path,
    *,
    batch_id: str,
    request_sha256: str,
    error: str,
    recorded_at: datetime,
) -> None:
    directory = root / batch_id
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": ATTEMPT_SCHEMA,
        "request_sha256": request_sha256,
        "batch_id": batch_id,
        "recorded_at_utc": recorded_at.isoformat(),
        "error": error,
    }
    name = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    path = directory / f"attempt_{name}.json"
    if not path.exists():
        _atomic_json(path, payload)


def _record_asset_failure(
    *,
    output: Path,
    attempts_root: Path,
    request_sha256: str,
    error: str,
    recorded_at: datetime,
) -> None:
    _write_attempt(
        attempts_root,
        batch_id="asset-snapshot",
        request_sha256=request_sha256,
        error=error,
        recorded_at=recorded_at,
    )
    _atomic_json(
        output / "_status.json",
        {
            "schema": POLL_MANIFEST_SCHEMA,
            "status": "incomplete",
            "request_sha256": request_sha256,
            "failed_stage": "asset_snapshot",
            "error": error,
            "updated_at_utc": recorded_at.isoformat(),
        },
    )


def _verify_attempt_inventory(root: Path, *, request_sha256: str) -> None:
    if root.is_symlink() or any(path.is_symlink() for path in root.rglob("*")):
        raise DataReadinessError("prospective attempt inventory contains a symlink")
    for path in root.rglob("*"):
        if path.is_dir():
            if path.parent != root or (not path.name.startswith("batch-") and path.name != "asset-snapshot"):
                raise DataReadinessError("prospective attempt directory grammar is invalid")
            continue
        if path.parent.parent != root or not path.name.startswith("attempt_") or path.suffix != ".json":
            raise DataReadinessError("prospective attempt file grammar is invalid")
        payload = _json_object(path)
        if (
            payload.get("schema") != ATTEMPT_SCHEMA
            or payload.get("request_sha256") != request_sha256
            or payload.get("batch_id") != path.parent.name
            or not str(payload.get("error", ""))
        ):
            raise DataReadinessError("prospective attempt record is invalid")
        _required_utc(payload, "recorded_at_utc")


def _frames_equal(left: pd.DataFrame, right: pd.DataFrame) -> bool:
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
            check_exact=True,
        )
    except AssertionError:
        return False
    return True


def _verify_asset_request_url(
    requested_url: str,
    *,
    final_url: str,
    redirect_chain: object,
) -> None:
    hostname = (urlsplit(requested_url).hostname or "").lower()
    if hostname not in ALPACA_ASSET_HOSTNAMES:
        raise DataReadinessError("prospective raw response is not bound to an approved Alpaca asset host")
    _verify_exact_alpaca_request(
        requested_url,
        final_url=final_url,
        redirect_chain=redirect_chain,
        hostname=hostname,
        path="/v2/assets",
        expected_query={"status": "active", "asset_class": "us_equity"},
    )


def _verify_news_request_url(
    requested_url: str,
    *,
    final_url: str,
    redirect_chain: object,
    symbols: Sequence[str],
    query_start: datetime,
    query_end: datetime,
    page_token: str | None,
) -> None:
    expected_query = {
        "symbols": ",".join(symbols),
        "start": _utc(query_start).isoformat(),
        "end": _utc(query_end).isoformat(),
        "sort": "asc",
        "limit": "50",
        "include_content": "true",
    }
    if page_token is not None:
        expected_query["page_token"] = page_token
    _verify_exact_alpaca_request(
        requested_url,
        final_url=final_url,
        redirect_chain=redirect_chain,
        hostname="data.alpaca.markets",
        path="/v1beta1/news",
        expected_query=expected_query,
    )


def _verify_exact_alpaca_request(
    requested_url: str,
    *,
    final_url: str,
    redirect_chain: object,
    hostname: str,
    path: str,
    expected_query: Mapping[str, str],
) -> None:
    parsed = urlsplit(requested_url)
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    query = {key: value for key, value in pairs}
    if (
        parsed.scheme != "https"
        or (parsed.hostname or "").lower() != hostname
        or parsed.path != path
        or parsed.fragment
        or len(query) != len(pairs)
        or query != dict(expected_query)
        or final_url != requested_url
        or not isinstance(redirect_chain, (list, tuple))
        or len(redirect_chain) != 0
    ):
        raise DataReadinessError(f"prospective raw response is not bound to the expected {path} request")


def _directory_inventory_sha256(directory: Path) -> str:
    if directory.is_symlink() or any(path.is_symlink() for path in directory.rglob("*")):
        raise DataReadinessError("authority inventory contains a symlink")
    records = [
        {"path": path.relative_to(directory).as_posix(), "sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    return json_sha256(records)


def _write_or_validate_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        if _json_object(path) != dict(payload):
            raise DataReadinessError(f"existing prospective request differs: {path}")
        return
    _atomic_json(path, payload)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_new_json(path: Path, payload: Mapping[str, object]) -> None:
    if path.exists():
        raise DataReadinessError(f"immutable registry record already exists: {path}")
    _atomic_json(path, payload)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"JSON artifact is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise DataReadinessError(f"JSON artifact must be an object: {path}")
    return {str(key): value for key, value in payload.items()}


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate == root or root not in candidate.parents or not candidate.is_file():
        raise DataReadinessError("artifact path escapes authority root or is missing")
    return candidate


def _required_utc(record: Mapping[str, object], key: str) -> datetime:
    value = pd.to_datetime(record.get(key), utc=True, errors="coerce")
    if pd.isna(value):
        raise DataReadinessError(f"{key} is missing or invalid")
    return cast(datetime, cast(pd.Timestamp, value).to_pydatetime())


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


def _json_default(value: object) -> object:
    if isinstance(value, (datetime, pd.Timestamp)):
        return pd.Timestamp(value).isoformat()
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"cannot serialize {type(value).__name__}")
