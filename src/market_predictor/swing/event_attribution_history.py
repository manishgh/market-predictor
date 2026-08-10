from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditCheck,
    CanonicalAuditReport,
)
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.swing.event_attribution import (
    ATTRIBUTION_POLICY_SHA256,
    ATTRIBUTION_POLICY_VERSION,
    RELATION_COLUMNS,
    build_event_security_relations,
)
from market_predictor.swing.news_history import NEWS_HISTORY_MANIFEST_SCHEMA
from market_predictor.v3.errors import DataReadinessError

ATTRIBUTION_REQUEST_SCHEMA = "swing.event_attribution_request.v1"
ATTRIBUTION_MANIFEST_SCHEMA = "swing.event_attribution_manifest.v1"
_RELATION_CHANNELS = (
    "direct_issuer",
    "business_exposure",
    "sector_context",
)
_ROOT_INVENTORY = {
    "_manifest.json",
    "_request.json",
    "_status.json",
    "relations",
}


@dataclass(frozen=True, slots=True)
class EventAttributionHistory:
    directory: Path
    request: Mapping[str, object]
    manifest: Mapping[str, object]
    artifact_records: tuple[Mapping[str, object], ...]


def attribute_alpaca_news_history(
    *,
    collection_dir: Path,
    collection_audit_path: Path,
    business_labels_path: Path,
    security_identities_path: Path,
    out_dir: Path,
    progress: Callable[[dict[str, object]], None] | None = None,
) -> dict[str, object]:
    """Build resumable point-in-time event-security relation artifacts."""

    collection_manifest_path = collection_dir / "_manifest.json"
    collection = _json_object(collection_manifest_path)
    collection_audit = _json_object(collection_audit_path)
    if (
        collection.get("status") != "complete"
        or bool(collection.get("production_ready"))
        or not bool(collection_audit.get("passed"))
        or collection_audit.get("request_sha256") != collection.get("request_sha256")
    ):
        raise DataReadinessError("event attribution requires a passed research-only collection audit")
    labels, label_manifest = load_canonical_artifact(
        business_labels_path,
        expected_type="security_business_labels",
        allow_research=True,
    )
    identities, identity_manifest = load_canonical_artifact(
        security_identities_path,
        expected_type="security_business_label_coverage",
        allow_research=True,
    )
    artifacts_raw = collection.get("artifacts")
    if not isinstance(artifacts_raw, list):
        raise DataReadinessError("news collection manifest has no artifact inventory")
    artifacts = [{str(key): value for key, value in item.items()} for item in artifacts_raw if isinstance(item, dict)]
    if len(artifacts) != len(artifacts_raw):
        raise DataReadinessError("news collection artifact inventory is malformed")
    excluded_raw = collection_audit.get(
        "coverage_blindspot_security_ids",
        [],
    )
    if not isinstance(excluded_raw, list):
        raise DataReadinessError("collection audit blindspot identities are malformed")
    excluded_security_ids = tuple(sorted(str(value) for value in excluded_raw))
    eligible = [artifact for artifact in artifacts if str(artifact.get("security_id", "")) not in excluded_security_ids]
    request = {
        "schema": ATTRIBUTION_REQUEST_SCHEMA,
        "collection_manifest_path": str(collection_manifest_path.resolve()),
        "collection_manifest_sha256": file_sha256(collection_manifest_path),
        "collection_request_sha256": str(collection["request_sha256"]),
        "collection_audit_path": str(collection_audit_path.resolve()),
        "collection_audit_sha256": file_sha256(collection_audit_path),
        "business_labels_path": str(business_labels_path.resolve()),
        "business_labels_sha256": str(label_manifest["artifact_sha256"]),
        "security_identities_path": str(
            security_identities_path.resolve()
        ),
        "security_identities_sha256": str(
            identity_manifest["artifact_sha256"]
        ),
        "attribution_policy_version": ATTRIBUTION_POLICY_VERSION,
        "attribution_policy_sha256": ATTRIBUTION_POLICY_SHA256,
        "excluded_security_ids": list(excluded_security_ids),
        "production_ready": False,
    }
    request_sha256 = _json_sha256(request)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "_manifest.json"
    if final_path.exists():
        raise DataReadinessError(f"completed event attribution is immutable: {final_path}")
    _write_or_validate_request(
        out_dir / "_request.json",
        request,
        request_sha256,
    )
    relation_dir = out_dir / "relations"
    relation_dir.mkdir(parents=True, exist_ok=True)

    observed: dict[str, dict[str, object]] = {}
    failures: dict[str, str] = {}
    skipped = 0
    for index, artifact in enumerate(eligible, start=1):
        chunk_id = _required_artifact_text(artifact, "chunk_id")
        source_sha256 = _required_artifact_text(
            artifact,
            "sha256",
        )
        target = relation_dir / f"{chunk_id}.parquet"
        existing = _load_existing_relation(
            target,
            request_sha256=request_sha256,
            chunk_id=chunk_id,
            source_sha256=source_sha256,
        )
        if existing is not None:
            observed[chunk_id] = existing
            _lock_path_for(target).unlink(missing_ok=True)
            skipped += 1
            _progress(
                progress,
                index=index,
                total=len(eligible),
                chunk_id=chunk_id,
                status="skipped",
                rows=_required_record_int(existing, "rows"),
            )
            continue
        try:
            source_path = Path(_required_artifact_text(artifact, "path"))
            events, event_manifest = load_canonical_artifact(
                source_path,
                expected_type="events",
                allow_research=True,
            )
            if str(event_manifest["artifact_sha256"]) != source_sha256:
                raise DataReadinessError(f"source event hash mismatch for {chunk_id}")
            relations = build_event_security_relations(
                events,
                labels,
                identities,
            )
            relation_manifest = write_canonical_artifact(
                relations,
                target,
                artifact_type="event_security_relations",
                audit=_passing_audit(len(relations)),
                inputs={
                    "event_attribution_request_sha256": (request_sha256),
                    "source_event_artifact_sha256": source_sha256,
                    "business_labels_sha256": str(label_manifest["artifact_sha256"]),
                    "security_identities_sha256": str(
                        identity_manifest["artifact_sha256"]
                    ),
                    "attribution_policy_sha256": (ATTRIBUTION_POLICY_SHA256),
                    "chunk_id": chunk_id,
                },
                production_ready=False,
            )
            record = _relation_record(
                artifact=artifact,
                target=target,
                relations=relations,
                manifest=relation_manifest,
            )
            _lock_path_for(target).unlink(missing_ok=True)
            observed[chunk_id] = record
            _progress(
                progress,
                index=index,
                total=len(eligible),
                chunk_id=chunk_id,
                status="observed",
                rows=len(relations),
            )
        except Exception as exc:
            failures[chunk_id] = f"{type(exc).__name__}: {str(exc)[:500]}"
            _progress(
                progress,
                index=index,
                total=len(eligible),
                chunk_id=chunk_id,
                status="failed",
                rows=0,
            )

    records = [observed[key] for key in sorted(observed)]
    channel_counts = {
        "direct_issuer": 0,
        "business_exposure": 0,
        "sector_context": 0,
    }
    for record in records:
        counts = record.get("channel_counts")
        if not isinstance(counts, dict):
            raise DataReadinessError("relation artifact channel counts are malformed")
        for channel in channel_counts:
            channel_counts[channel] += int(counts.get(channel, 0))
    status = "complete" if not failures and len(records) == len(eligible) else "incomplete"
    result: dict[str, object] = {
        "schema": ATTRIBUTION_MANIFEST_SCHEMA,
        "request_sha256": request_sha256,
        "status": status,
        "requested_chunks": len(eligible),
        "observed_chunks": len(records),
        "skipped_chunks": skipped,
        "failed_chunks": failures,
        "excluded_security_ids": list(excluded_security_ids),
        "relation_rows": sum(_required_record_int(item, "rows") for item in records),
        "channel_counts": channel_counts,
        "artifacts": records,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "production_ready": False,
    }
    _atomic_json(out_dir / "_status.json", result)
    if status == "complete":
        _atomic_json(final_path, result)
        load_event_attribution_history(out_dir)
    return result


def load_event_attribution_history(
    directory: Path,
    *,
    require_production_ready: bool = False,
    expected_manifest_sha256: str | None = None,
) -> EventAttributionHistory:
    """Strictly replay a completed, research-only event-attribution authority."""

    root = directory.resolve()
    if not root.is_dir() or {path.name for path in root.iterdir()} != _ROOT_INVENTORY:
        raise DataReadinessError("event attribution root inventory does not verify")
    manifest_path = root / "_manifest.json"
    if expected_manifest_sha256 is not None and (
        not _is_sha256(expected_manifest_sha256)
        or file_sha256(manifest_path) != expected_manifest_sha256
    ):
        raise DataReadinessError("event attribution manifest identity does not verify")

    manifest = _json_object(manifest_path)
    status = _json_object(root / "_status.json")
    production_ready = manifest.get("production_ready")
    if (
        manifest.get("schema") != ATTRIBUTION_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("failed_chunks") != {}
        or production_ready is not False
    ):
        raise DataReadinessError("event attribution manifest is not a completed research-only authority")
    if require_production_ready:
        raise DataReadinessError("event attribution history is research-only and not production ready")
    if status != manifest:
        raise DataReadinessError("event attribution status does not match the completed manifest")

    request_payload = _json_object(root / "_request.json")
    request_sha256 = request_payload.pop("request_sha256", None)
    if (
        not isinstance(request_sha256, str)
        or not _is_sha256(request_sha256)
        or _json_sha256(request_payload) != request_sha256
        or manifest.get("request_sha256") != request_sha256
        or request_payload.get("schema") != ATTRIBUTION_REQUEST_SCHEMA
        or request_payload.get("production_ready") is not False
        or request_payload.get("attribution_policy_version") != ATTRIBUTION_POLICY_VERSION
        or request_payload.get("attribution_policy_sha256") != ATTRIBUTION_POLICY_SHA256
    ):
        raise DataReadinessError("event attribution request hash or policy does not verify")

    source_records = _verify_source_lineage(request_payload)
    excluded = _string_list(
        request_payload.get("excluded_security_ids"),
        "request excluded_security_ids",
    )
    if excluded != sorted(excluded) or len(excluded) != len(set(excluded)):
        raise DataReadinessError("event attribution excluded security identities do not verify")
    if manifest.get("excluded_security_ids") != excluded:
        raise DataReadinessError("event attribution exclusion lineage does not verify")
    eligible_sources = {
        chunk_id: record
        for chunk_id, record in source_records.items()
        if str(record.get("security_id", "")) not in set(excluded)
    }

    artifact_records = _artifact_records(manifest)
    records_by_chunk = {
        _required_chunk_id(record): record for record in artifact_records
    }
    if len(records_by_chunk) != len(artifact_records):
        raise DataReadinessError("event attribution contains duplicate relation chunks")
    if set(records_by_chunk) != set(eligible_sources):
        raise DataReadinessError("event attribution relation inventory is incomplete")
    requested_chunks = _required_int(manifest, "requested_chunks")
    observed_chunks = _required_int(manifest, "observed_chunks")
    skipped_chunks = _required_int(manifest, "skipped_chunks")
    if (
        requested_chunks != len(eligible_sources)
        or observed_chunks != len(artifact_records)
        or requested_chunks != observed_chunks
        or skipped_chunks > observed_chunks
    ):
        raise DataReadinessError("event attribution root chunk counts do not verify")

    relations_directory = root / "relations"
    if not relations_directory.is_dir():
        raise DataReadinessError("event attribution relations directory is missing")
    expected_relation_files = {
        name
        for chunk_id in records_by_chunk
        for name in (
            f"{chunk_id}.parquet",
            f"{chunk_id}.parquet.manifest.json",
        )
    }
    if (
        {path.name for path in relations_directory.iterdir()} != expected_relation_files
        or any(path.is_dir() for path in relations_directory.iterdir())
    ):
        raise DataReadinessError("event attribution child inventory does not verify")

    total_rows = 0
    channel_totals = {channel: 0 for channel in _RELATION_CHANNELS}
    for chunk_id in sorted(records_by_chunk):
        record = records_by_chunk[chunk_id]
        source_record = eligible_sources[chunk_id]
        expected_path = (relations_directory / f"{chunk_id}.parquet").resolve()
        declared_path = _resolved_path(record.get("path"), "relation artifact path")
        if declared_path != expected_path or not _is_inside(root, declared_path):
            raise DataReadinessError("event attribution relation path escapes its authority")
        relations, child_manifest = load_canonical_artifact(
            expected_path,
            expected_type="event_security_relations",
            allow_research=True,
        )
        source_sha256 = _required_sha256(
            source_record,
            "sha256",
            "source event artifact",
        )
        if list(relations.columns) != list(RELATION_COLUMNS):
            raise DataReadinessError("event attribution relation schema does not verify")
        if child_manifest.get("production_ready") is not False:
            raise DataReadinessError("event attribution child must remain research-only")
        child_inputs = child_manifest.get("inputs")
        if not isinstance(child_inputs, dict) or child_inputs != {
            "event_attribution_request_sha256": request_sha256,
            "source_event_artifact_sha256": source_sha256,
            "business_labels_sha256": _required_sha256(
                request_payload, "business_labels_sha256", "request"
            ),
            "security_identities_sha256": _required_sha256(
                request_payload, "security_identities_sha256", "request"
            ),
            "attribution_policy_sha256": ATTRIBUTION_POLICY_SHA256,
            "chunk_id": chunk_id,
        }:
            raise DataReadinessError("event attribution child lineage does not verify")
        if (
            _required_sha256(record, "source_event_sha256", "relation artifact")
            != source_sha256
            or child_manifest.get("artifact_sha256")
            != _required_sha256(record, "sha256", "relation artifact")
            or len(relations) != _required_int(record, "rows")
        ):
            raise DataReadinessError("event attribution child hash or row count does not verify")
        observed_channels = _channel_counts(relations)
        declared_channels = _declared_channel_counts(record.get("channel_counts"))
        if observed_channels != declared_channels:
            raise DataReadinessError("event attribution child channel counts do not verify")
        total_rows += len(relations)
        for channel, count in observed_channels.items():
            channel_totals[channel] += count

    if (
        total_rows != _required_int(manifest, "relation_rows")
        or channel_totals != _declared_channel_counts(manifest.get("channel_counts"))
    ):
        raise DataReadinessError("event attribution root row or channel counts do not verify")
    if expected_manifest_sha256 is not None and file_sha256(manifest_path) != expected_manifest_sha256:
        raise DataReadinessError("event attribution manifest changed while loading")
    return EventAttributionHistory(
        directory=root,
        request=request_payload,
        manifest=manifest,
        artifact_records=tuple(artifact_records),
    )


def _load_existing_relation(
    path: Path,
    *,
    request_sha256: str,
    chunk_id: str,
    source_sha256: str,
) -> dict[str, object] | None:
    if not path.exists() and not manifest_path_for(path).exists():
        return None
    relations, manifest = load_canonical_artifact(
        path,
        expected_type="event_security_relations",
        allow_research=True,
    )
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or (
        inputs.get("event_attribution_request_sha256") != request_sha256
        or inputs.get("source_event_artifact_sha256") != source_sha256
        or inputs.get("chunk_id") != chunk_id
        or inputs.get("attribution_policy_sha256") != ATTRIBUTION_POLICY_SHA256
    ):
        raise DataReadinessError(f"existing relation artifact lineage mismatch: {path}")
    return _relation_record(
        artifact={
            "chunk_id": chunk_id,
            "sha256": source_sha256,
        },
        target=path,
        relations=relations,
        manifest=manifest,
    )


def _relation_record(
    *,
    artifact: dict[str, object],
    target: Path,
    relations: pd.DataFrame,
    manifest: dict[str, object],
) -> dict[str, object]:
    counts = {channel: int(value) for channel, value in relations["relation_channel"].value_counts().items()}
    return {
        "chunk_id": _required_artifact_text(
            artifact,
            "chunk_id",
        ),
        "source_event_sha256": _required_artifact_text(
            artifact,
            "sha256",
        ),
        "path": str(target.resolve()),
        "sha256": str(manifest["artifact_sha256"]),
        "rows": len(relations),
        "channel_counts": {
            "direct_issuer": counts.get("direct_issuer", 0),
            "business_exposure": counts.get(
                "business_exposure",
                0,
            ),
            "sector_context": counts.get("sector_context", 0),
        },
    }


def _verify_source_lineage(
    request: Mapping[str, object],
) -> dict[str, dict[str, object]]:
    collection_manifest_path = _resolved_path(
        request.get("collection_manifest_path"),
        "collection manifest path",
    )
    if file_sha256(collection_manifest_path) != _required_sha256(
        request,
        "collection_manifest_sha256",
        "request",
    ):
        raise DataReadinessError("event attribution collection manifest hash does not verify")
    collection = _json_object(collection_manifest_path)
    collection_request_sha256 = _required_sha256(
        request,
        "collection_request_sha256",
        "request",
    )
    if (
        collection.get("schema") != NEWS_HISTORY_MANIFEST_SCHEMA
        or collection.get("status") != "complete"
        or collection.get("production_ready") is not False
        or collection.get("request_sha256") != collection_request_sha256
        or collection.get("failed_chunks") != {}
    ):
        raise DataReadinessError("event attribution source collection does not verify")

    audit_path = _resolved_path(
        request.get("collection_audit_path"),
        "collection audit path",
    )
    if file_sha256(audit_path) != _required_sha256(
        request,
        "collection_audit_sha256",
        "request",
    ):
        raise DataReadinessError("event attribution collection audit hash does not verify")
    audit = _json_object(audit_path)
    excluded = _string_list(
        audit.get("coverage_blindspot_security_ids"),
        "collection audit coverage_blindspot_security_ids",
    )
    if (
        audit.get("passed") is not True
        or audit.get("request_sha256") != collection_request_sha256
        or excluded != request.get("excluded_security_ids")
    ):
        raise DataReadinessError("event attribution collection audit lineage does not verify")

    labels_path = _resolved_path(
        request.get("business_labels_path"),
        "business labels path",
    )
    _, labels_manifest = load_canonical_artifact(
        labels_path,
        expected_type="security_business_labels",
        allow_research=True,
    )
    if labels_manifest.get("artifact_sha256") != _required_sha256(
        request,
        "business_labels_sha256",
        "request",
    ):
        raise DataReadinessError("event attribution business-label hash does not verify")

    identities_path = _resolved_path(
        request.get("security_identities_path"),
        "security identities path",
    )
    _, identities_manifest = load_canonical_artifact(
        identities_path,
        expected_type="security_business_label_coverage",
        allow_research=True,
    )
    if identities_manifest.get("artifact_sha256") != _required_sha256(
        request,
        "security_identities_sha256",
        "request",
    ):
        raise DataReadinessError("event attribution security-identity hash does not verify")

    raw_records = collection.get("artifacts")
    if not isinstance(raw_records, list):
        raise DataReadinessError("event attribution source artifact inventory is malformed")
    records: dict[str, dict[str, object]] = {}
    total_rows = 0
    collection_root = collection_manifest_path.parent.resolve()
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise DataReadinessError("event attribution source artifact record is malformed")
        record = {str(key): value for key, value in raw.items()}
        chunk_id = _required_chunk_id(record)
        if chunk_id in records:
            raise DataReadinessError("event attribution source chunks are duplicated")
        source_path = _resolved_path(record.get("path"), "source event path")
        if not _is_inside(collection_root, source_path):
            raise DataReadinessError("event attribution source event path escapes its collection")
        declared_manifest_path = _resolved_path(
            record.get("manifest_path"),
            "source event manifest path",
        )
        if declared_manifest_path != manifest_path_for(source_path).resolve():
            raise DataReadinessError("event attribution source event manifest path does not verify")
        events, source_manifest = load_canonical_artifact(
            source_path,
            expected_type="events",
            allow_research=True,
        )
        inputs = source_manifest.get("inputs")
        if (
            source_manifest.get("production_ready") is not False
            or source_manifest.get("artifact_sha256")
            != _required_sha256(record, "sha256", "source event artifact")
            or not isinstance(inputs, dict)
            or inputs.get("collection_request_sha256") != collection_request_sha256
            or inputs.get("chunk_id") != chunk_id
            or len(events) != _required_int(record, "rows")
        ):
            raise DataReadinessError("event attribution source event lineage does not verify")
        total_rows += len(events)
        records[chunk_id] = record
    if (
        len(records) != _required_int(collection, "artifact_count")
        or total_rows != _required_int(collection, "total_rows")
        or len(records) != _required_int(collection, "observed_chunks")
        or _required_int(collection, "requested_chunks")
        != len(records) + _required_int(collection, "empty_chunks")
    ):
        raise DataReadinessError("event attribution source inventory counts do not verify")
    return records


def _artifact_records(manifest: Mapping[str, object]) -> list[dict[str, object]]:
    raw_records = manifest.get("artifacts")
    if not isinstance(raw_records, list):
        raise DataReadinessError("event attribution relation inventory is malformed")
    records: list[dict[str, object]] = []
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise DataReadinessError("event attribution relation record is malformed")
        records.append({str(key): value for key, value in raw.items()})
    return records


def _required_chunk_id(record: Mapping[str, object]) -> str:
    value = record.get("chunk_id")
    if not isinstance(value, str) or not value or Path(value).name != value or value in {".", ".."}:
        raise DataReadinessError("event attribution chunk identity is invalid")
    return value


def _required_sha256(
    record: Mapping[str, object],
    key: str,
    label: str,
) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not _is_sha256(value):
        raise DataReadinessError(f"{label} has invalid {key}")
    return value


def _required_int(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DataReadinessError(f"event attribution record has invalid {key}")
    return value


def _declared_channel_counts(value: object) -> dict[str, int]:
    if not isinstance(value, dict) or set(value) != set(_RELATION_CHANNELS):
        raise DataReadinessError("event attribution channel-count schema does not verify")
    output: dict[str, int] = {}
    for channel in _RELATION_CHANNELS:
        count = value.get(channel)
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise DataReadinessError("event attribution channel count is invalid")
        output[channel] = count
    return output


def _channel_counts(relations: pd.DataFrame) -> dict[str, int]:
    if "relation_channel" not in relations.columns:
        raise DataReadinessError("event attribution relation channel is missing")
    observed = {
        str(channel): int(count)
        for channel, count in relations["relation_channel"].value_counts().items()
    }
    if not set(observed).issubset(_RELATION_CHANNELS):
        raise DataReadinessError("event attribution relation channel is unsupported")
    return {channel: observed.get(channel, 0) for channel in _RELATION_CHANNELS}


def _string_list(value: object, label: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise DataReadinessError(f"{label} is invalid")
    return list(value)


def _resolved_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise DataReadinessError(f"{label} is invalid")
    return Path(value).resolve()


def _is_inside(root: Path, path: Path) -> bool:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    return resolved_path == resolved_root or resolved_root in resolved_path.parents


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _lock_path_for(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


def _passing_audit(rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name="event_security_relations",
                status="pass",
                failures=0,
                rows_checked=rows,
                detail=("point-in-time identity, relation channel, availability, and lineage validated"),
            ),
        )
    )


def _write_or_validate_request(
    path: Path,
    request: dict[str, object],
    request_sha256: str,
) -> None:
    payload = {
        **request,
        "request_sha256": request_sha256,
    }
    if path.exists():
        if _json_object(path) != payload:
            raise DataReadinessError(f"event attribution resume request mismatch: {path}")
        return
    _atomic_json(path, payload)


def _required_artifact_text(
    artifact: dict[str, object],
    key: str,
) -> str:
    value = artifact.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DataReadinessError(f"news artifact has invalid {key}")
    return value.strip()


def _required_record_int(
    record: dict[str, object],
    key: str,
) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DataReadinessError(f"relation artifact has invalid {key}")
    return value


def _progress(
    callback: Callable[[dict[str, object]], None] | None,
    *,
    index: int,
    total: int,
    chunk_id: str,
    status: str,
    rows: int,
) -> None:
    if callback is not None:
        callback(
            {
                "index": index,
                "total": total,
                "chunk_id": chunk_id,
                "status": status,
                "rows": rows,
            }
        )


def _json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataReadinessError(f"expected JSON object: {path}")
    return {str(key): item for key, item in value.items()}


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(
    path: Path,
    value: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
