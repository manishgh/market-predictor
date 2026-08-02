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
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.reconciliation import (
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
from market_predictor.resources import (
    assert_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.swing.news_source_inventory import (
    build_source_news_shard_inventory,
)
from market_predictor.v3.errors import DataReadinessError

CATALYST_LINEAGE_REQUEST_SCHEMA = "swing.catalyst_lineage_request.v2"
CATALYST_LINEAGE_MANIFEST_SCHEMA = "swing.catalyst_lineage_manifest.v2"
CATALYST_EVENT_SCHEMA = "swing.catalyst_event.v1"
CATALYST_COVERAGE_SCHEMA = "swing.catalyst_source_coverage.v1"
FEATURE_INVENTORY_SCHEMA = "swing.catalyst_feature_inventory.v1"
_EXPECTED_POLICY_SCHEMA = "market_predictor.catalyst_lineage.v1"
_SUPPORTED_CHANNELS = frozenset(
    {"direct_issuer", "business_exposure", "sector_context"}
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
    windows = {
        str(name): pd.Timedelta(str(duration))
        for name, duration in raw_windows.items()
    }
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
    if (
        not bool(collection_audit.get("passed"))
        or collection_audit.get("request_sha256") != collection.get("request_sha256")
    ):
        raise DataReadinessError("catalyst lineage requires a passed collection audit")
    excluded = _validated_exclusions(collection_audit, attribution, sentiment)
    source_inventory = {
        str(record["chunk_id"]): record
        for record in build_source_news_shard_inventory(collection_dir, collection)
    }
    source_records = _records_by_chunk(collection, "news collection")
    relation_records = _records_by_chunk(attribution, "event attribution")
    sentiment_records = _records_by_chunk(sentiment, "event sentiment")
    source_collections_path = Path(str(collection["source_collections_path"]))
    source_collections, source_collection_manifest = load_canonical_artifact(
        source_collections_path,
        expected_type="source_collections",
        allow_research=True,
    )
    if (
        str(source_collection_manifest.get("artifact_sha256", ""))
        != str(collection.get("source_collections_sha256", ""))
        or bool(source_collection_manifest.get("production_ready"))
    ):
        raise DataReadinessError("collection source-ledger identity is invalid")
    eligible_chunk_ids = sorted(
        chunk_id
        for chunk_id, record in source_records.items()
        if str(record.get("security_id", "")) not in excluded
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
    decision_indices = {
        str(security_id): indices
        for security_id, indices in decisions.groupby("security_id", sort=False).indices.items()
    }
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
                    decisions.iloc[decision_indices[security_id]]
                    for security_id in target_security_ids
                    if security_id in decision_indices
                ]
                decision_part = (
                    pd.concat(parts, ignore_index=True)
                    if parts
                    else decisions.iloc[0:0].copy()
                )
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
                assignment_status_counts[str(status)] = (
                    assignment_status_counts.get(str(status), 0) + int(count)
                )
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
    status = (
        "complete"
        if not failures and len(observed) == len(eligible_chunk_ids)
        else "incomplete"
    )
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
        "training_eligible_rows": sum(
            _required_int(record, "training_eligible_rows")
            for record in observed
        ),
        "channel_counts": channel_counts,
        "assignment_rows": sum(
            _required_int(record, "assignment_rows")
            for record in observed
        ),
        "assignment_status_counts": dict(sorted(assignment_status_counts.items())),
        "coverage": {
            "path": str(coverage_path.resolve()),
            "sha256": str(coverage_manifest["artifact_sha256"]),
            "rows": len(coverage),
            "states": {
                str(key): int(value)
                for key, value in coverage["coverage_state"].value_counts().sort_index().items()
            },
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
    output["missingness_known"] = output["coverage_state"].isin(
        {"observed_complete", "observed_empty"}
    )
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
            (
                frame["training_eligible"].astype(bool)
                != frame["relation_channel"].astype(str).isin(policy.training_eligible_channels)
            ).sum()
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
        failures += int(
            (
                age.dt.total_seconds()
                > pd.to_numeric(assigned["window_seconds"], errors="coerce")
            ).sum()
        )
    return _audit_report(
        "catalyst_event_assignments",
        failures,
        len(assignments),
        "deterministic assignment replay, cutoff, and window checks pass",
    )


def _coverage_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    failures = int(frame["coverage_state"].eq("failed_or_unobserved").sum())
    failures += int(
        (
            frame["training_eligible"].astype(bool)
            & ~frame["missingness_known"].astype(bool)
        ).sum()
    )
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
        name: sorted(
            {
                template.format(window=window)
                for template in templates
                for window in windows
            }
        )
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
        "coverage_states": {
            str(key): int(value)
            for key, value in coverage["coverage_state"].value_counts().sort_index().items()
        },
        "event_artifact_count": len(event_records),
        "training_contract": (
            "Only direct-issuer rows from source-complete windows may produce "
            "catalyst-only or technical-plus-catalyst features."
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
    parts = [
        decisions.iloc[decision_indices[security_id]]
        for security_id in target_security_ids
        if security_id in decision_indices
    ]
    decision_part = (
        pd.concat(parts, ignore_index=True)
        if parts
        else decisions.iloc[0:0].copy()
    )
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
    source_by_chunk = {
        str(row["chunk_id"]): row
        for row in source_rows.to_dict(orient="records")
    }
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
            or str(sentiment.get("ticker", "")).upper()
            != str(source.get("ticker", "")).upper()
            or _required_text(sentiment, "source_event_artifact_sha256")
            != _required_text(source_evidence, "sha256")
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
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"JSON artifact must contain an object: {path}")
    return {str(key): value for key, value in loaded.items()}


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
