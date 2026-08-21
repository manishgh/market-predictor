"""Live GDELT collection with observed availability and immutable lineage."""
from __future__ import annotations



import hashlib
import json
import os
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Protocol, cast
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import numpy as np
import pandas as pd
import requests

from market_predictor.canonical.audits import (
    CanonicalAuditReport,
    audit_canonical_events,
    audit_source_collections,
)
from market_predictor.canonical.contracts import CanonicalEvent, SourceCollection
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.resources import assert_memory_budget, memory_audit
from market_predictor.core.errors import DataReadinessError

GDELT_COLLECTION_SCHEMA: Final = "edge_rebuild.gdelt_global_collection.v2"
GDELT_COLLECTION_MANIFEST_SCHEMA: Final = "edge_rebuild.gdelt_global_collection_manifest.v2"
GDELT_COLLECTION_REQUEST_SCHEMA: Final = "edge_rebuild.gdelt_global_collection_request.v2"
GDELT_DOC_ENDPOINT: Final = "https://api.gdeltproject.org/api/v2/doc/doc"
GLOBAL_TICKER: Final = "MARKET"
GLOBAL_SECURITY_ID: Final = "market:global"
SOURCE_FAMILY: Final = "gdelt"
MAXIMUM_PROCESS_MEMORY_GIB: Final = 4.0
MEMORY_GUARD_HEADROOM_GIB: Final = 0.5
MAX_GDELT_RECORDS: Final = 250
GDELT_MAX_ATTEMPTS: Final = 4
GDELT_INITIAL_BACKOFF_SECONDS: Final = 0.5
GDELT_MAX_BACKOFF_SECONDS: Final = 8.0
GDELT_MAX_RETRY_AFTER_SECONDS: Final = 30.0
GDELT_RETRYABLE_STATUS_CODES: Final = frozenset({429, 500, 502, 503, 504})
GLOBAL_EVENT_QUERY_POLICY_V1: Final = (
    '("strait of hormuz" OR hormuz OR "red sea" OR "suez canal") '
    '(oil OR tanker OR shipping OR blockade OR attack OR disruption)',
    '("taiwan strait" OR taiwan OR tsmc OR "south china sea") '
    '(military OR blockade OR invasion OR missile OR "export control" OR sanction)',
    '(russia OR ukraine OR "black sea" OR nato) '
    '(missile OR drone OR sanction OR pipeline OR lng OR grain OR wheat)',
    '("rare earth" OR gallium OR germanium OR lithium OR cobalt OR graphite) '
    '("export control" OR ban OR restriction OR quota OR sanction OR tariff)',
    '(cyberattack OR ransomware OR "critical infrastructure" OR "power grid") '
    '(outage OR shutdown OR breach OR malware)',
)


@dataclass(frozen=True, slots=True)
class GdeltCollectionRequest:
    queries: tuple[str, ...]
    requested_start_utc: datetime
    requested_end_utc: datetime
    max_records: int = MAX_GDELT_RECORDS
    timeout_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class GdeltFetchResult:
    records: Sequence[Mapping[str, object]]
    complete: bool
    completed_queries: tuple[str, ...]
    errors: tuple[str, ...] = ()
    raw_response_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class GdeltGlobalEventCollection:
    directory: Path
    events: pd.DataFrame
    source_collections: pd.DataFrame
    manifest: Mapping[str, object]
    authority: Mapping[str, object]

    @property
    def events_path(self) -> Path:
        return self.directory / "events.parquet"

    @property
    def source_collections_path(self) -> Path:
        return self.directory / "source_collections.parquet"


class GdeltFetcher(Protocol):
    def __call__(self, request: GdeltCollectionRequest) -> GdeltFetchResult: ...


class EventScorer(Protocol):
    def score_texts(self, texts: list[str], batch_size: int = 16) -> pd.DataFrame: ...


Clock = Callable[[], datetime]


def fetch_gdelt_doc_api(
    request: GdeltCollectionRequest,
    *,
    max_attempts: int = GDELT_MAX_ATTEMPTS,
    initial_backoff_seconds: float = GDELT_INITIAL_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> GdeltFetchResult:
    """Fetch one bounded live window from the GDELT DOC 2.0 article-list API."""

    normalized = validate_gdelt_collection_request(request)
    if max_attempts <= 0 or max_attempts > 10:
        raise ValueError("max_attempts must be between 1 and 10")
    if initial_backoff_seconds < 0 or initial_backoff_seconds > GDELT_MAX_BACKOFF_SECONDS:
        raise ValueError(
            f"initial_backoff_seconds must be between 0 and {GDELT_MAX_BACKOFF_SECONDS}"
        )
    records: list[dict[str, object]] = []
    response_hashes: list[str] = []
    complete = True
    for query in normalized.queries:
        parameters: dict[str, str | int] = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "maxrecords": normalized.max_records,
            "startdatetime": normalized.requested_start_utc.strftime("%Y%m%d%H%M%S"),
            "enddatetime": normalized.requested_end_utc.strftime("%Y%m%d%H%M%S"),
        }
        response = _get_gdelt_response(
            parameters,
            timeout_seconds=normalized.timeout_seconds,
            max_attempts=max_attempts,
            initial_backoff_seconds=initial_backoff_seconds,
            sleep=sleep,
        )
        response_hashes.append(hashlib.sha256(response.content).hexdigest())
        try:
            payload = response.json()
        except (requests.JSONDecodeError, json.JSONDecodeError) as exc:
            raise DataReadinessError("GDELT returned invalid JSON") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("articles"), list):
            raise DataReadinessError("GDELT response is missing the articles list")
        query_records = payload["articles"]
        if any(not isinstance(record, dict) for record in query_records):
            raise DataReadinessError("GDELT response contains a non-object article")
        complete &= len(query_records) < normalized.max_records
        records.extend(
            {**cast(dict[str, object], record), "collection_query": query}
            for record in query_records
        )
    raw_sha256 = _json_sha256(response_hashes)
    return GdeltFetchResult(
        records=records,
        complete=complete,
        completed_queries=normalized.queries,
        raw_response_sha256=raw_sha256,
    )


def collect_live_gdelt_global_events(
    request: GdeltCollectionRequest,
    output_directory: Path,
    *,
    scorer: EventScorer,
    fetch: GdeltFetcher = fetch_gdelt_doc_api,
    clock: Clock | None = None,
    scorer_batch_size: int = 16,
    scorer_identity: str | None = None,
    maximum_process_memory_gib: float = MAXIMUM_PROCESS_MEMORY_GIB,
    memory_guard_headroom_gib: float = MEMORY_GUARD_HEADROOM_GIB,
) -> GdeltGlobalEventCollection:
    """Collect, score, and atomically publish one observed GDELT window."""

    normalized = validate_gdelt_collection_request(request)
    _validate_memory_policy(maximum_process_memory_gib, memory_guard_headroom_gib)
    if scorer_batch_size <= 0 or scorer_batch_size > MAX_GDELT_RECORDS:
        raise ValueError(f"scorer_batch_size must be between 1 and {MAX_GDELT_RECORDS}")
    now = clock or _utc_now
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise DataReadinessError(f"GDELT collection is immutable: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = output_directory.with_name(f".{output_directory.name}.{uuid4().hex}.tmp")
    staging.mkdir()
    try:
        started_at = _observed_now(now, "collection start")
        if normalized.requested_end_utc > started_at:
            raise DataReadinessError("live GDELT collection cannot claim a future coverage end")
        request_payload = _request_payload(
            normalized,
            scorer,
            scorer_batch_size,
            scorer_identity,
        )
        request_sha256 = _json_sha256(request_payload)
        _guard_memory(maximum_process_memory_gib, memory_guard_headroom_gib, "before GDELT fetch")
        fetched = fetch(normalized)
        fetched_at = _observed_now(now, "GDELT fetch completion")
        if fetched.errors:
            raise DataReadinessError("GDELT fetch reported errors: " + "; ".join(fetched.errors))
        if not fetched.complete:
            raise DataReadinessError("GDELT fetch was partial or truncated; coverage was not published")
        if fetched.completed_queries != normalized.queries:
            raise DataReadinessError(
                "GDELT fetch did not complete the exact immutable query policy"
            )
        if len(fetched.records) > normalized.max_records * len(normalized.queries):
            raise DataReadinessError("GDELT fetch exceeded the bounded per-query request size")
        raw_response_sha256 = _raw_response_sha256(fetched)
        raw_records = _normalize_raw_records(fetched.records, normalized, fetched_at)
        deduplicated = _deduplicate(raw_records)
        _guard_memory(maximum_process_memory_gib, memory_guard_headroom_gib, "after GDELT normalization")
        scored_at: datetime | None = None
        if deduplicated:
            texts = [_score_text(record) for record in deduplicated]
            score_frame = scorer.score_texts(texts, batch_size=scorer_batch_size)
            scored_at = _observed_now(now, "GDELT scoring completion")
            scores = _normalize_scores(score_frame, len(deduplicated))
            event_rows = [
                _canonical_event(record, score, fetched_at, scored_at)
                for record, score in zip(deduplicated, scores, strict=True)
            ]
        else:
            event_rows = []
        completed_at = _observed_now(now, "collection completion")
        _validate_clock_order(started_at, fetched_at, scored_at, completed_at)
        events = _canonical_event_frame(event_rows)
        collection_id = _json_sha256(
            {
                "collection_request_sha256": request_sha256,
                "started_at_utc": started_at.isoformat(),
                "completed_at_utc": completed_at.isoformat(),
                "raw_response_sha256": raw_response_sha256,
            }
        )
        source_collections = pd.DataFrame.from_records(
            [
                SourceCollection(
                    collection_id=collection_id,
                    ticker=GLOBAL_TICKER,
                    source_family=SOURCE_FAMILY,
                    requested_start_utc=normalized.requested_start_utc,
                    requested_end_utc=normalized.requested_end_utc,
                    started_at_utc=started_at,
                    completed_at_utc=completed_at,
                    status="observed" if event_rows else "observed_empty",
                    row_count=len(event_rows),
                ).model_dump()
            ],
            columns=list(SourceCollection.model_fields),
        )
        event_audit = CanonicalAuditReport(checks=audit_canonical_events(events, require_observed=True))
        coverage_audit = CanonicalAuditReport(
            checks=audit_source_collections(
                source_collections,
                required_tickers=(GLOBAL_TICKER,),
                required_sources=(SOURCE_FAMILY,),
                require_success=True,
            )
        )
        event_audit.raise_for_failure()
        coverage_audit.raise_for_failure()
        inputs: dict[str, str] = {
            "collection_request_sha256": request_sha256,
            "source_policy_sha256": str(request_payload["source_policy_sha256"]),
            "sentiment_scorer_identity": str(request_payload["scorer_identity"]),
            "raw_response_sha256": raw_response_sha256,
            "collector_schema": GDELT_COLLECTION_SCHEMA,
        }
        event_path = staging / "events.parquet"
        coverage_path = staging / "source_collections.parquet"
        event_manifest = write_canonical_artifact(
            events,
            event_path,
            artifact_type="events",
            audit=event_audit,
            inputs=inputs,
            production_ready=True,
        )
        coverage_manifest = write_canonical_artifact(
            source_collections,
            coverage_path,
            artifact_type="source_collections",
            audit=coverage_audit,
            inputs=inputs,
            production_ready=True,
        )
        event_path.with_suffix(".parquet.lock").unlink(missing_ok=True)
        coverage_path.with_suffix(".parquet.lock").unlink(missing_ok=True)
        _rewrite_artifact_path(event_path, output_directory)
        _rewrite_artifact_path(coverage_path, output_directory)
        event_manifest = _json_object(manifest_path_for(event_path))
        coverage_manifest = _json_object(manifest_path_for(coverage_path))
        manifest: dict[str, object] = {
            "schema": GDELT_COLLECTION_MANIFEST_SCHEMA,
            "state": "complete",
            "request": request_payload,
            "collection_request_sha256": request_sha256,
            "raw_response_sha256": raw_response_sha256,
            "started_at_utc": started_at.isoformat(),
            "fetched_at_utc": fetched_at.isoformat(),
            "scored_at_utc": scored_at.isoformat() if scored_at is not None else None,
            "completed_at_utc": completed_at.isoformat(),
            "raw_rows": len(fetched.records),
            "unique_rows": len(events),
            "duplicate_rows": len(fetched.records) - len(events),
            "coverage_status": "observed" if len(events) else "observed_empty",
            "artifacts": {
                "events": _artifact_record(event_path, event_manifest),
                "source_collections": _artifact_record(coverage_path, coverage_manifest),
            },
            "memory": memory_audit(
                hard_budget_gib=maximum_process_memory_gib,
                headroom_gib=memory_guard_headroom_gib,
            ).to_record(),
            "production_ready": True,
        }
        _atomic_json(staging / "_manifest.json", manifest)
        authority = {
            "schema": GDELT_COLLECTION_SCHEMA,
            "state": "complete",
            "manifest": "_manifest.json",
            "manifest_sha256": file_sha256(staging / "_manifest.json"),
            "collection_request_sha256": request_sha256,
            "event_artifact_sha256": event_manifest["artifact_sha256"],
            "source_collection_artifact_sha256": coverage_manifest["artifact_sha256"],
        }
        _atomic_json(staging / "_authority.json", authority)
        load_gdelt_global_event_collection(
            staging,
            expected_collection_request_sha256=request_sha256,
        )
        os.replace(staging, output_directory)
        return load_gdelt_global_event_collection(
            output_directory,
            expected_collection_request_sha256=request_sha256,
        )
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_gdelt_global_event_collection(
    directory: Path,
    *,
    expected_collection_request_sha256: str | None = None,
) -> GdeltGlobalEventCollection:
    """Load a collection only after complete replay and artifact verification."""

    directory = directory.resolve()
    expected_files = {
        "_authority.json",
        "_manifest.json",
        "events.parquet",
        "events.parquet.manifest.json",
        "source_collections.parquet",
        "source_collections.parquet.manifest.json",
    }
    if not directory.is_dir():
        raise DataReadinessError(f"GDELT collection directory is missing: {directory}")
    if {path.name for path in directory.iterdir()} != expected_files:
        raise DataReadinessError("GDELT collection inventory does not verify")
    manifest_path = directory / "_manifest.json"
    authority = _json_object(directory / "_authority.json")
    manifest = _json_object(manifest_path)
    if manifest.get("schema") != GDELT_COLLECTION_MANIFEST_SCHEMA or manifest.get("state") != "complete":
        raise DataReadinessError("GDELT collection manifest is not complete")
    if (
        authority.get("schema") != GDELT_COLLECTION_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("manifest") != "_manifest.json"
        or authority.get("manifest_sha256") != file_sha256(manifest_path)
    ):
        raise DataReadinessError("GDELT collection authority does not verify")
    request = manifest.get("request")
    if not isinstance(request, dict):
        raise DataReadinessError("GDELT collection request is malformed")
    request_sha256 = _json_sha256(request)
    if request_sha256 != manifest.get("collection_request_sha256") or request_sha256 != authority.get(
        "collection_request_sha256"
    ):
        raise DataReadinessError("GDELT collection request hash does not verify")
    if expected_collection_request_sha256 is not None and request_sha256 != expected_collection_request_sha256:
        raise DataReadinessError("GDELT collection belongs to another request")
    event_path = directory / "events.parquet"
    coverage_path = directory / "source_collections.parquet"
    events, event_manifest = load_canonical_artifact(event_path, expected_type="events")
    source_collections, coverage_manifest = load_canonical_artifact(
        coverage_path,
        expected_type="source_collections",
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise DataReadinessError("GDELT artifact inventory is malformed")
    _verify_artifact(artifacts.get("events"), event_path, event_manifest, len(events))
    _verify_artifact(
        artifacts.get("source_collections"),
        coverage_path,
        coverage_manifest,
        len(source_collections),
    )
    for child in (event_manifest, coverage_manifest):
        inputs = child.get("inputs")
        if not isinstance(inputs, dict) or (
            inputs.get("collection_request_sha256") != request_sha256
            or inputs.get("source_policy_sha256")
            != request.get("source_policy_sha256")
            or inputs.get("sentiment_scorer_identity")
            != request.get("scorer_identity")
            or inputs.get("raw_response_sha256") != manifest.get("raw_response_sha256")
            or inputs.get("collector_schema") != GDELT_COLLECTION_SCHEMA
        ):
            raise DataReadinessError("GDELT child artifact lineage does not verify")
    if (
        authority.get("event_artifact_sha256") != event_manifest.get("artifact_sha256")
        or authority.get("source_collection_artifact_sha256") != coverage_manifest.get("artifact_sha256")
    ):
        raise DataReadinessError("GDELT artifact authority hashes do not verify")
    _validate_loaded_collection(events, source_collections, manifest)
    return GdeltGlobalEventCollection(
        directory=directory,
        events=events,
        source_collections=source_collections,
        manifest=manifest,
        authority=authority,
    )


def validate_gdelt_collection_request(request: GdeltCollectionRequest) -> GdeltCollectionRequest:
    """Validate and normalize a request before allocating scorer resources."""

    queries = tuple(query.strip() for query in request.queries)
    if not queries or any(not query for query in queries):
        raise ValueError("GDELT queries must contain non-empty values")
    if len(queries) != len(set(queries)):
        raise ValueError("GDELT queries must be unique")
    start = _strict_utc(request.requested_start_utc, "requested_start_utc")
    end = _strict_utc(request.requested_end_utc, "requested_end_utc")
    if end < start:
        raise ValueError("GDELT request window is reversed")
    if request.max_records <= 0 or request.max_records > MAX_GDELT_RECORDS:
        raise ValueError(f"max_records must be between 1 and {MAX_GDELT_RECORDS}")
    if request.timeout_seconds <= 0 or request.timeout_seconds > 120:
        raise ValueError("timeout_seconds must be in (0, 120]")
    return GdeltCollectionRequest(
        queries=queries,
        requested_start_utc=start,
        requested_end_utc=end,
        max_records=request.max_records,
        timeout_seconds=float(request.timeout_seconds),
    )


def _get_gdelt_response(
    parameters: Mapping[str, str | int],
    *,
    timeout_seconds: float,
    max_attempts: int,
    initial_backoff_seconds: float,
    sleep: Callable[[float], None],
) -> requests.Response:
    last_transport_error: requests.RequestException | None = None
    for attempt in range(max_attempts):
        try:
            response = requests.get(
                GDELT_DOC_ENDPOINT,
                params=dict(parameters),
                timeout=timeout_seconds,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_transport_error = exc
            if attempt + 1 == max_attempts:
                break
            sleep(_retry_delay(None, attempt, initial_backoff_seconds))
            continue

        if response.status_code in GDELT_RETRYABLE_STATUS_CODES:
            if attempt + 1 == max_attempts:
                raise DataReadinessError(
                    f"GDELT request failed after {max_attempts} attempts "
                    f"with HTTP {response.status_code}"
                )
            sleep(_retry_delay(response, attempt, initial_backoff_seconds))
            continue
        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise DataReadinessError(
                f"GDELT request failed permanently with HTTP {response.status_code}"
            ) from exc
        return response

    raise DataReadinessError(
        f"GDELT request failed after {max_attempts} attempts due to a transport error"
    ) from last_transport_error


def _retry_delay(
    response: requests.Response | None,
    attempt: int,
    initial_backoff_seconds: float,
) -> float:
    if response is not None:
        retry_after = response.headers.get("Retry-After")
        if retry_after is not None:
            try:
                parsed = float(retry_after)
            except ValueError:
                parsed = -1.0
            if parsed >= 0:
                return min(parsed, GDELT_MAX_RETRY_AFTER_SECONDS)
    return min(
        initial_backoff_seconds * float(2**attempt),
        GDELT_MAX_BACKOFF_SECONDS,
    )


def _request_payload(
    request: GdeltCollectionRequest,
    scorer: EventScorer,
    scorer_batch_size: int,
    scorer_identity: str | None,
) -> dict[str, object]:
    identity = scorer_identity.strip() if scorer_identity is not None else _default_scorer_identity(scorer)
    if not identity:
        raise ValueError("scorer_identity must not be empty")
    query_policy_sha256 = _json_sha256(list(request.queries))
    source_policy_sha256 = _json_sha256(
        {
            "query_policy_sha256": query_policy_sha256,
            "scorer_identity": identity,
        }
    )
    return {
        "schema": GDELT_COLLECTION_REQUEST_SCHEMA,
        "source_family": SOURCE_FAMILY,
        "global_identity": {"ticker": GLOBAL_TICKER, "security_id": GLOBAL_SECURITY_ID},
        "queries": list(request.queries),
        "query_policy_sha256": query_policy_sha256,
        "source_policy_sha256": source_policy_sha256,
        "requested_start_utc": request.requested_start_utc.isoformat(),
        "requested_end_utc": request.requested_end_utc.isoformat(),
        "max_records": request.max_records,
        "endpoint": GDELT_DOC_ENDPOINT,
        "availability_policy": "observed collection and scoring timestamps only",
        "scorer_identity": identity,
        "scorer_batch_size": scorer_batch_size,
    }


def _default_scorer_identity(scorer: EventScorer) -> str:
    parts = [f"{type(scorer).__module__}.{type(scorer).__qualname__}"]
    for attribute in ("model_name", "model_revision", "max_length"):
        value = getattr(scorer, attribute, None)
        if value is not None:
            parts.append(f"{attribute}={value}")
    return "|".join(parts)


def _normalize_raw_records(
    records: Sequence[Mapping[str, object]],
    request: GdeltCollectionRequest,
    observed_at: datetime,
) -> list[dict[str, object]]:
    normalized: list[dict[str, object]] = []
    for index, item in enumerate(records):
        if not isinstance(item, Mapping):
            raise DataReadinessError(f"GDELT record {index} is not an object")
        raw = {str(key): value for key, value in item.items()}
        collection_query = str(raw.get("collection_query") or "").strip()
        if collection_query not in request.queries:
            raise DataReadinessError(
                f"GDELT record {index} has no valid frozen query-family identity"
            )
        query_family = _query_family_token(collection_query)
        raw_sha256 = _json_sha256(raw)
        title = str(raw.get("title") or "").strip()
        if not title:
            raise DataReadinessError(f"GDELT record {index} has no title")
        published = _provider_timestamp(
            raw.get("published_at_utc") or raw.get("published_at") or raw.get("seendate"),
            f"GDELT record {index} publication",
        )
        updated_value = raw.get("provider_updated_at_utc") or raw.get("updated_at")
        updated = _provider_timestamp(updated_value, f"GDELT record {index} update") if updated_value else None
        if published > observed_at or (updated is not None and updated > observed_at):
            raise DataReadinessError(f"GDELT record {index} claims provider content from the future")
        if published < request.requested_start_utc or published > request.requested_end_utc:
            raise DataReadinessError(f"GDELT record {index} is outside the requested publication window")
        url = _canonical_url(str(raw.get("url") or raw.get("url_mobile") or ""))
        identity = {
            "source_family": SOURCE_FAMILY,
            "query_family": query_family,
            "url": url,
            "published_at_utc": published.isoformat(),
            "title": "" if url else title,
        }
        normalized.append(
            {
                "event_id": _json_sha256(identity),
                "published_at_utc": published,
                "provider_updated_at_utc": updated,
                "title": title,
                "url": url,
                "summary": str(raw.get("summary") or "").strip(),
                "text": str(raw.get("text") or raw.get("content") or "").strip(),
                "source": (
                    f"gdelt:{query_family}:"
                    f"{str(raw.get('domain') or raw.get('source') or 'unknown').strip().lower()}"
                ),
                "raw_sha256": raw_sha256,
            }
        )
    return normalized


def _query_family_token(query: str) -> str:
    return f"flashpoint-{_json_sha256(query)[:16]}"


def _deduplicate(records: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    ordered = sorted(records, key=lambda row: (str(row["event_id"]), str(row["raw_sha256"])))
    unique: dict[str, dict[str, object]] = {}
    for record in ordered:
        unique.setdefault(str(record["event_id"]), record)
    return [unique[event_id] for event_id in sorted(unique)]


def _normalize_scores(frame: pd.DataFrame, expected_rows: int) -> list[dict[str, float | None]]:
    if not isinstance(frame, pd.DataFrame) or len(frame) != expected_rows:
        raise DataReadinessError("GDELT scorer row count does not match collected events")
    if "sentiment_numeric" not in frame.columns:
        raise DataReadinessError("GDELT scorer must return sentiment_numeric")
    sentiment = pd.to_numeric(frame["sentiment_numeric"], errors="coerce")
    if bool(sentiment.isna().any() or np.isinf(sentiment.to_numpy(dtype=float)).any() or sentiment.abs().gt(1).any()):
        raise DataReadinessError("GDELT scorer returned invalid sentiment_numeric")
    if "relevance" in frame.columns:
        relevance = pd.to_numeric(frame["relevance"], errors="coerce")
        invalid_relevance = relevance.notna() & (np.isinf(relevance.to_numpy(dtype=float)) | relevance.lt(0))
        if bool(invalid_relevance.any()):
            raise DataReadinessError("GDELT scorer returned invalid relevance")
    else:
        relevance = pd.Series(np.nan, index=frame.index, dtype=float)
    return [
        {
            "sentiment_numeric": float(sentiment.iloc[index]),
            "relevance": float(relevance.iloc[index]) if pd.notna(relevance.iloc[index]) else None,
        }
        for index in range(expected_rows)
    ]


def _canonical_event(
    record: Mapping[str, object],
    score: Mapping[str, float | None],
    fetched_at: datetime,
    scored_at: datetime,
) -> dict[str, object]:
    return CanonicalEvent(
        event_id=str(record["event_id"]),
        ticker=GLOBAL_TICKER,
        security_id=GLOBAL_SECURITY_ID,
        source_family=SOURCE_FAMILY,
        source=str(record["source"]),
        published_at_utc=cast(datetime, record["published_at_utc"]),
        provider_updated_at_utc=cast(datetime | None, record["provider_updated_at_utc"]),
        first_seen_at_utc=fetched_at,
        available_at_utc=fetched_at,
        sentiment_scored_at_utc=scored_at,
        feature_available_at_utc=scored_at,
        title=str(record["title"]),
        url=str(record["url"]),
        summary=str(record["summary"]),
        text=str(record["text"]),
        sentiment_numeric=cast(float, score["sentiment_numeric"]),
        relevance=score["relevance"],
        availability_policy="observed",
        raw_sha256=str(record["raw_sha256"]),
    ).model_dump()


def _canonical_event_frame(rows: Sequence[Mapping[str, object]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows, columns=list(CanonicalEvent.model_fields))


def _validate_loaded_collection(
    events: pd.DataFrame,
    coverage: pd.DataFrame,
    manifest: Mapping[str, object],
) -> None:
    CanonicalAuditReport(checks=audit_canonical_events(events, require_observed=True)).raise_for_failure()
    CanonicalAuditReport(
        checks=audit_source_collections(
            coverage,
            required_tickers=(GLOBAL_TICKER,),
            required_sources=(SOURCE_FAMILY,),
            require_success=True,
        )
    ).raise_for_failure()
    if len(coverage) != 1:
        raise DataReadinessError("GDELT collection must contain exactly one coverage row")
    row = coverage.iloc[0]
    if str(row["ticker"]) != GLOBAL_TICKER or str(row["source_family"]) != SOURCE_FAMILY:
        raise DataReadinessError("GDELT source coverage identity does not verify")
    expected_status = "observed" if len(events) else "observed_empty"
    if str(row["status"]) != expected_status or int(row["row_count"]) != len(events):
        raise DataReadinessError("GDELT source coverage does not reconcile with events")
    unique_rows = manifest.get("unique_rows")
    if (
        manifest.get("coverage_status") != expected_status
        or not isinstance(unique_rows, int)
        or isinstance(unique_rows, bool)
        or unique_rows != len(events)
    ):
        raise DataReadinessError("GDELT collection manifest does not reconcile with events")
    raw_rows = manifest.get("raw_rows")
    duplicate_rows = manifest.get("duplicate_rows")
    if (
        not isinstance(raw_rows, int)
        or isinstance(raw_rows, bool)
        or not isinstance(duplicate_rows, int)
        or isinstance(duplicate_rows, bool)
        or raw_rows < unique_rows
        or duplicate_rows != raw_rows - unique_rows
    ):
        raise DataReadinessError("GDELT raw-row reconciliation does not verify")
    manifest_started = _strict_utc(manifest.get("started_at_utc"), "manifest started_at_utc")
    manifest_fetched = _strict_utc(manifest.get("fetched_at_utc"), "manifest fetched_at_utc")
    manifest_completed = _strict_utc(manifest.get("completed_at_utc"), "manifest completed_at_utc")
    manifest_scored_value = manifest.get("scored_at_utc")
    manifest_scored = (
        None
        if manifest_scored_value is None
        else _strict_utc(manifest_scored_value, "manifest scored_at_utc")
    )
    _validate_clock_order(manifest_started, manifest_fetched, manifest_scored, manifest_completed)
    coverage_started = _strict_utc(row["started_at_utc"], "started_at_utc")
    coverage_completed = _strict_utc(row["completed_at_utc"], "completed_at_utc")
    if coverage_started != manifest_started or coverage_completed != manifest_completed:
        raise DataReadinessError("GDELT observed collection timestamps do not verify")
    if not events.empty:
        if bool(
            events["ticker"].astype(str).ne(GLOBAL_TICKER).any()
            or events["security_id"].astype(str).ne(GLOBAL_SECURITY_ID).any()
            or events["source_family"].astype(str).ne(SOURCE_FAMILY).any()
            or events["availability_policy"].astype(str).ne("observed").any()
        ):
            raise DataReadinessError("GDELT event identity or observed policy does not verify")
        if manifest_scored is None:
            raise DataReadinessError("non-empty GDELT collection requires a scoring time")
        first_seen = pd.to_datetime(events["first_seen_at_utc"], utc=True)
        available = pd.to_datetime(events["available_at_utc"], utc=True)
        scored = pd.to_datetime(events["sentiment_scored_at_utc"], utc=True)
        feature = pd.to_datetime(events["feature_available_at_utc"], utc=True)
        if bool(
            first_seen.ne(manifest_fetched).any()
            or available.ne(first_seen).any()
            or scored.ne(manifest_scored).any()
            or feature.ne(scored).any()
            or feature.gt(manifest_completed).any()
        ):
            raise DataReadinessError("GDELT observed availability timestamps do not verify")
    elif manifest_scored is not None:
        raise DataReadinessError("empty GDELT collection cannot claim a scoring time")


def _raw_response_sha256(result: GdeltFetchResult) -> str:
    calculated = _json_sha256([dict(record) for record in result.records])
    if result.raw_response_sha256 is None:
        return calculated
    supplied = result.raw_response_sha256.strip().lower()
    if len(supplied) != 64 or any(character not in "0123456789abcdef" for character in supplied):
        raise DataReadinessError("GDELT raw response hash is malformed")
    return supplied


def _provider_timestamp(value: object, label: str) -> datetime:
    if isinstance(value, str):
        stripped = value.strip()
        if len(stripped) == 16 and stripped.endswith("Z") and "T" in stripped:
            try:
                return datetime.strptime(stripped, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
            except ValueError:
                pass
    return _strict_utc(value, label)


def _strict_utc(value: object, label: str) -> datetime:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError(f"{label} is invalid") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise DataReadinessError(f"{label} must be timezone-aware")
    return cast(datetime, timestamp.tz_convert("UTC").to_pydatetime())


def _observed_now(clock: Clock, label: str) -> datetime:
    return _strict_utc(clock(), label)


def _validate_clock_order(
    started: datetime,
    fetched: datetime,
    scored: datetime | None,
    completed: datetime,
) -> None:
    ordered = [started, fetched, *(tuple() if scored is None else (scored,)), completed]
    if any(later < earlier for earlier, later in zip(ordered[:-1], ordered[1:], strict=True)):
        raise DataReadinessError("observed collection clock moved backwards")


def _canonical_url(value: str) -> str:
    stripped = value.strip()
    if not stripped:
        return ""
    split = urlsplit(stripped)
    if split.scheme.lower() not in {"http", "https"} or not split.netloc:
        raise DataReadinessError("GDELT article URL is invalid")
    return urlunsplit((split.scheme.lower(), split.netloc.lower(), split.path, split.query, ""))


def _score_text(record: Mapping[str, object]) -> str:
    title = str(record["title"]).strip()
    summary = str(record["summary"]).strip()
    return f"{title}. {summary}".strip(". ") if summary and summary != title else title


def _artifact_record(path: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": path.name,
        "artifact_sha256": manifest["artifact_sha256"],
        "manifest_sha256": file_sha256(manifest_path_for(path)),
        "rows": manifest["rows"],
    }


def _verify_artifact(
    record: object,
    path: Path,
    child_manifest: Mapping[str, object],
    rows: int,
) -> None:
    if not isinstance(record, dict) or (
        record.get("path") != path.name
        or record.get("artifact_sha256") != child_manifest.get("artifact_sha256")
        or record.get("manifest_sha256") != file_sha256(manifest_path_for(path))
        or int(record.get("rows", -1)) != rows
    ):
        raise DataReadinessError(f"GDELT artifact record does not verify: {path.name}")


def _rewrite_artifact_path(path: Path, final_directory: Path) -> None:
    manifest_path = manifest_path_for(path)
    manifest = _json_object(manifest_path)
    manifest["artifact_path"] = str(final_directory / path.name)
    _atomic_json(manifest_path, manifest)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _json_object(path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"invalid GDELT JSON artifact: {path}") from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"GDELT JSON artifact must contain an object: {path}")
    return {str(key): value for key, value in loaded.items()}


def _json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    ).hexdigest()


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        return _strict_utc(value, "JSON datetime").isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _validate_memory_policy(hard_budget_gib: float, headroom_gib: float) -> None:
    if hard_budget_gib > MAXIMUM_PROCESS_MEMORY_GIB:
        raise ValueError(f"maximum_process_memory_gib cannot exceed {MAXIMUM_PROCESS_MEMORY_GIB}")
    if hard_budget_gib <= 0 or headroom_gib <= 0 or headroom_gib >= hard_budget_gib:
        raise ValueError("memory budget and headroom are invalid")


def _guard_memory(hard_budget_gib: float, headroom_gib: float, stage: str) -> None:
    assert_memory_budget(hard_budget_gib=hard_budget_gib, headroom_gib=headroom_gib, stage=stage)


def _utc_now() -> datetime:
    return datetime.now(UTC)
