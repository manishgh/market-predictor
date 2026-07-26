from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
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
    build_event_security_relations,
)
from market_predictor.v3.errors import DataReadinessError

ATTRIBUTION_REQUEST_SCHEMA = "swing.event_attribution_request.v1"
ATTRIBUTION_MANIFEST_SCHEMA = "swing.event_attribution_manifest.v1"


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
    return result


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
