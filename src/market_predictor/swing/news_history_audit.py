from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditReport,
    audit_canonical_events,
    audit_source_collections,
)
from market_predictor.canonical.store import (
    load_canonical_artifact,
    manifest_path_for,
)
from market_predictor.resources import (
    assert_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.swing.news_history import (
    NEWS_HISTORY_MANIFEST_SCHEMA,
    NEWS_HISTORY_REQUEST_SCHEMA,
    NEWS_PAGE_SCHEMA,
)
from market_predictor.v3.errors import DataReadinessError


def audit_alpaca_news_history(
    collection_dir: Path,
    *,
    memory_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.75,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replay a completed news collection without loading the corpus at once."""

    request_path = collection_dir / "_request.json"
    final_path = collection_dir / "_manifest.json"
    status_path = collection_dir / "_status.json"
    ledger_path = collection_dir / "_source_collections.parquet"
    required_paths = (request_path, final_path, status_path, ledger_path)
    missing = [str(path) for path in required_paths if not path.exists()]
    if missing:
        raise DataReadinessError(
            f"Alpaca news collection audit is missing files: {missing}"
        )

    request = _json_object(request_path)
    final = _json_object(final_path)
    status = _json_object(status_path)
    request_identity = str(request.get("request_sha256", ""))
    request_payload = {
        key: value for key, value in request.items() if key != "request_sha256"
    }
    errors: list[str] = []
    if (
        request.get("schema") != NEWS_HISTORY_REQUEST_SCHEMA
        or request_identity != _sha256_json(request_payload)
    ):
        errors.append("request identity is invalid")
    if (
        final.get("schema") != NEWS_HISTORY_MANIFEST_SCHEMA
        or final.get("status") != "complete"
        or final.get("request_sha256") != request_identity
        or bool(final.get("production_ready"))
        or final.get("availability_policy") != "provider_publication_proxy"
    ):
        errors.append("final manifest identity or research-only policy is invalid")
    if (
        status.get("request_sha256") != request_identity
        or status.get("status") != "complete"
    ):
        errors.append("status does not match the final request")

    work_units_raw = request.get("work_units")
    artifacts_raw = final.get("artifacts")
    if not isinstance(work_units_raw, list) or not isinstance(artifacts_raw, list):
        raise DataReadinessError(
            "Alpaca news request/final manifest has invalid work-unit inventory"
        )
    work_units = {
        str(record["chunk_id"]): record
        for record in work_units_raw
        if isinstance(record, dict) and record.get("chunk_id")
    }
    artifacts = {
        str(record["chunk_id"]): record
        for record in artifacts_raw
        if isinstance(record, dict) and record.get("chunk_id")
    }
    if len(work_units) != len(work_units_raw):
        errors.append("work-unit chunk identities are missing or duplicated")
    if len(artifacts) != len(artifacts_raw):
        errors.append("artifact chunk identities are missing or duplicated")

    ledger, ledger_manifest = load_canonical_artifact(
        ledger_path,
        expected_type="source_collections",
        allow_research=True,
    )
    CanonicalAuditReport(
        checks=audit_source_collections(
            ledger,
            required_tickers=(
                str(unit.get("ticker", ""))
                for unit in work_units.values()
            ),
            required_sources=("alpaca",),
            require_success=True,
        )
    ).raise_for_failure()
    if (
        str(final.get("source_collections_sha256", ""))
        != str(ledger_manifest.get("artifact_sha256", ""))
        or bool(ledger_manifest.get("production_ready"))
    ):
        errors.append("source-collection ledger identity is invalid")
    if "chunk_id" not in ledger or bool(ledger["chunk_id"].astype(str).duplicated().any()):
        errors.append("source-collection ledger chunk identities are invalid")
    ledger_by_chunk = {
        str(row["chunk_id"]): row for row in ledger.to_dict(orient="records")
    }
    if set(ledger_by_chunk) != set(work_units):
        errors.append("source-collection ledger does not cover every work unit")

    event_ids: set[str] = set()
    duplicate_event_ids = 0
    report_rows: list[dict[str, Any]] = []
    for index, (chunk_id, unit) in enumerate(work_units.items()):
        assert_memory_budget(
            hard_budget_gib=memory_budget_gib,
            headroom_gib=memory_headroom_gib,
            stage=f"Alpaca news audit chunk {index + 1}/{len(work_units)}",
        )
        chunk_errors: list[str] = []
        ledger_row = ledger_by_chunk.get(chunk_id, {})
        expected_status = str(ledger_row.get("status", "missing"))
        page_dir = collection_dir / "raw_pages" / chunk_id
        try:
            page_count, page_hashes = _audit_pages(
                page_dir,
                request_sha256=request_identity,
                chunk_id=chunk_id,
            )
        except Exception as exc:
            page_count = 0
            page_hashes = {}
            chunk_errors.append(f"pages:{type(exc).__name__}:{exc}")

        artifact = artifacts.get(chunk_id)
        event_rows = 0
        first_published: str | None = None
        last_published: str | None = None
        if expected_status == "observed":
            if artifact is None:
                chunk_errors.append("observed chunk has no event artifact")
            else:
                try:
                    event_rows, first_published, last_published, duplicate_count = (
                        _audit_event_artifact(
                            unit=unit,
                            artifact=artifact,
                            request_sha256=request_identity,
                            page_hashes=page_hashes,
                            global_event_ids=event_ids,
                        )
                    )
                    duplicate_event_ids += duplicate_count
                except Exception as exc:
                    chunk_errors.append(
                        f"events:{type(exc).__name__}:{exc}"
                    )
        elif expected_status == "observed_empty":
            if artifact is not None:
                chunk_errors.append("observed-empty chunk has an event artifact")
            if int(ledger_row.get("row_count", -1)) != 0:
                chunk_errors.append("observed-empty chunk has nonzero row count")
        else:
            chunk_errors.append(f"nonterminal source status {expected_status}")

        if int(ledger_row.get("pages", -1)) != page_count:
            chunk_errors.append("ledger page count does not match archive")
        if int(ledger_row.get("row_count", -1)) != event_rows:
            chunk_errors.append("ledger row count does not match events")
        if chunk_errors:
            errors.extend(f"{chunk_id}:{detail}" for detail in chunk_errors)
        report_rows.append(
            {
                "chunk_id": chunk_id,
                "ticker": str(unit.get("ticker", "")),
                "security_id": str(unit.get("security_id", "")),
                "start_utc": str(unit.get("start_utc", "")),
                "end_exclusive_utc": str(unit.get("end_exclusive_utc", "")),
                "status": expected_status,
                "pages": page_count,
                "provider_rows": int(ledger_row.get("provider_rows", 0)),
                "accepted_rows": int(ledger_row.get("accepted_rows", 0)),
                "event_rows": event_rows,
                "first_published_at_utc": first_published,
                "last_published_at_utc": last_published,
                "audit_errors": " | ".join(chunk_errors),
            }
        )
        release_process_memory()

    report = pd.DataFrame.from_records(report_rows)
    starts = pd.to_datetime(report["start_utc"], utc=True, errors="coerce")
    ends = pd.to_datetime(
        report["end_exclusive_utc"],
        utc=True,
        errors="coerce",
    )
    long_empty = report["status"].eq("observed_empty") & (
        ends.sub(starts).ge(pd.Timedelta(days=30))
    )
    blindspot_security_ids = set(
        report.loc[long_empty, "security_id"].astype(str)
    )
    report["catalyst_source_complete"] = ~report["security_id"].astype(
        str
    ).isin(blindspot_security_ids)
    total_rows = int(report["event_rows"].sum()) if not report.empty else 0
    if int(final.get("requested_chunks", -1)) != len(work_units):
        errors.append("final requested chunk count is wrong")
    if int(final.get("artifact_count", -1)) != len(artifacts):
        errors.append("final artifact count is wrong")
    if int(final.get("total_rows", -1)) != total_rows:
        errors.append("final total row count is wrong")
    if duplicate_event_ids:
        errors.append(
            f"canonical event IDs repeat across chunks: {duplicate_event_ids}"
        )
    memory = memory_audit(
        hard_budget_gib=memory_budget_gib,
        headroom_gib=memory_headroom_gib,
    ).to_record()
    summary: dict[str, Any] = {
        "schema": "swing.alpaca_news_history_audit.v1",
        "collection_dir": str(collection_dir),
        "request_sha256": request_identity,
        "passed": not errors,
        "errors": errors,
        "requested_chunks": len(work_units),
        "observed_chunks": int(report["status"].eq("observed").sum()),
        "empty_chunks": int(report["status"].eq("observed_empty").sum()),
        "event_rows": total_rows,
        "unique_event_ids": len(event_ids),
        "duplicate_event_ids": duplicate_event_ids,
        "page_count": int(report["pages"].sum()),
        "tickers": int(report["ticker"].nunique()),
        "security_ids": int(report["security_id"].nunique()),
        "coverage_blindspot_chunks": int(long_empty.sum()),
        "coverage_blindspot_tickers": sorted(
            report.loc[long_empty, "ticker"].astype(str).unique()
        ),
        "coverage_blindspot_security_ids": sorted(blindspot_security_ids),
        "catalyst_training_policy": (
            "exclude coverage_blindspot_security_ids; never impute missing "
            "source history as zero events"
        ),
        "catalyst_training_security_ids": int(
            report.loc[
                report["catalyst_source_complete"],
                "security_id",
            ].nunique()
        ),
        "first_published_at_utc": _optional_min(
            report["first_published_at_utc"]
        ),
        "last_published_at_utc": _optional_max(
            report["last_published_at_utc"]
        ),
        "memory": memory,
    }
    return report, summary


def _audit_pages(
    page_dir: Path,
    *,
    request_sha256: str,
    chunk_id: str,
) -> tuple[int, dict[Path, str]]:
    paths = sorted(page_dir.glob("page_*.json"))
    if not paths:
        raise DataReadinessError("chunk has no archived provider page")
    expected_token: str | None = None
    hashes: dict[Path, str] = {}
    for expected_index, path in enumerate(paths):
        raw = path.read_bytes()
        file_hash = hashlib.sha256(raw).hexdigest()
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise DataReadinessError(f"page is not an object: {path}")
        content_hash = payload.get("content_sha256")
        content = {
            key: value
            for key, value in payload.items()
            if key != "content_sha256"
        }
        if (
            payload.get("schema") != NEWS_PAGE_SCHEMA
            or payload.get("collection_request_sha256") != request_sha256
            or payload.get("chunk_id") != chunk_id
            or payload.get("page_index") != expected_index
            or payload.get("request_page_token") != expected_token
            or content_hash != _sha256_json(content)
            or not isinstance(payload.get("news"), list)
        ):
            raise DataReadinessError(f"page integrity failed: {path}")
        expected_token = payload.get("next_page_token")
        if expected_token is None and expected_index != len(paths) - 1:
            raise DataReadinessError(f"page follows a terminal token: {path}")
        hashes[path.resolve()] = file_hash
    if expected_token is not None:
        raise DataReadinessError("last archived page is not terminal")
    return len(paths), hashes


def _audit_event_artifact(
    *,
    unit: dict[str, Any],
    artifact: dict[str, Any],
    request_sha256: str,
    page_hashes: dict[Path, str],
    global_event_ids: set[str],
) -> tuple[int, str, str, int]:
    path = Path(str(artifact["path"]))
    events, manifest = load_canonical_artifact(
        path,
        expected_type="events",
        allow_research=True,
    )
    if bool(manifest.get("production_ready")):
        raise DataReadinessError("historical event artifact is production-ready")
    inputs = manifest.get("inputs")
    if (
        not isinstance(inputs, dict)
        or inputs.get("collection_request_sha256") != request_sha256
        or inputs.get("chunk_id") != unit.get("chunk_id")
    ):
        raise DataReadinessError("event artifact request identity is invalid")
    declared_pages = {
        Path(str(key)).resolve(): str(value)
        for key, value in inputs.items()
        if str(key).lower().endswith(".json")
    }
    if declared_pages != page_hashes:
        raise DataReadinessError("event artifact raw-page inventory is invalid")
    if (
        str(artifact.get("sha256", "")) != str(manifest["artifact_sha256"])
        or str(artifact.get("manifest_path", ""))
        != str(manifest_path_for(path))
    ):
        raise DataReadinessError("event artifact manifest record is invalid")
    CanonicalAuditReport(
        checks=audit_canonical_events(events, require_observed=False)
    ).raise_for_failure()
    if (
        set(events["ticker"].astype(str)) != {str(unit["ticker"])}
        or set(events["security_id"].astype(str)) != {str(unit["security_id"])}
        or set(events["availability_policy"].astype(str))
        != {"provider_publication_proxy"}
    ):
        raise DataReadinessError("event artifact has invalid stock identity or policy")
    published = pd.to_datetime(events["published_at_utc"], utc=True)
    start = pd.Timestamp(unit["start_utc"])
    end = pd.Timestamp(unit["end_exclusive_utc"])
    if bool((published.lt(start) | published.ge(end)).any()):
        raise DataReadinessError("event publication is outside its half-open chunk")
    duplicate_count = 0
    for event_id in events["event_id"].astype(str):
        if event_id in global_event_ids:
            duplicate_count += 1
        global_event_ids.add(event_id)
    return (
        len(events),
        published.min().isoformat(),
        published.max().isoformat(),
        duplicate_count,
    )


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataReadinessError(f"expected JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _optional_min(values: pd.Series) -> str | None:
    present = values.dropna().astype(str)
    return present.min() if not present.empty else None


def _optional_max(values: pd.Series) -> str | None:
    present = values.dropna().astype(str)
    return present.max() if not present.empty else None
