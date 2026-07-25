from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditReport,
    audit_canonical_events,
    audit_source_collections,
)
from market_predictor.canonical.contracts import SourceCollection
from market_predictor.canonical.normalize import canonicalize_events
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
from market_predictor.sources.alpaca import AlpacaNewsPage
from market_predictor.symbols import canonical_symbol
from market_predictor.v3.errors import DataReadinessError

NEWS_HISTORY_REQUEST_SCHEMA = "swing.alpaca_news_history_request.v1"
NEWS_HISTORY_MANIFEST_SCHEMA = "swing.alpaca_news_history_manifest.v1"
NEWS_PAGE_SCHEMA = "swing.alpaca_news_page.v1"
NewsPageFetcher = Callable[
    [str, datetime, datetime, str | None],
    AlpacaNewsPage,
]


@dataclass(frozen=True, slots=True)
class NewsHistoryCollectionResult:
    status: str
    requested_chunks: int
    observed_chunks: int
    empty_chunks: int
    failed_chunks: tuple[str, ...]
    skipped_chunks: int
    manifest_path: Path | None
    status_path: Path


@dataclass(frozen=True, slots=True)
class _WorkUnit:
    chunk_id: str
    ticker: str
    provider_symbol: str
    security_id: str
    start_utc: datetime
    end_exclusive_utc: datetime


def collect_alpaca_news_history(
    *,
    memberships_path: Path,
    start_date: date,
    end_date: date,
    out_dir: Path,
    fetch_page: NewsPageFetcher,
    provider_symbol_for: Callable[[str], str],
    workers: int = 2,
    chunk_days: int = 92,
    memory_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.75,
) -> NewsHistoryCollectionResult:
    """Collect immutable, publication-time-proxy Alpaca news by security interval."""

    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    if workers < 1 or workers > 4:
        raise ValueError("workers must be between 1 and 4")
    if chunk_days < 7 or chunk_days > 366:
        raise ValueError("chunk_days must be between 7 and 366")
    memberships, membership_manifest = load_canonical_artifact(
        memberships_path,
        expected_type="memberships",
        allow_research=True,
    )
    required = {
        "ticker",
        "security_id",
        "effective_from_utc",
        "effective_to_utc",
    }
    missing = sorted(required.difference(memberships.columns))
    if missing:
        raise DataReadinessError(
            f"news history memberships are missing columns: {missing}"
        )

    start_utc = datetime.combine(start_date, time.min, tzinfo=UTC)
    end_exclusive_utc = datetime.combine(
        end_date + timedelta(days=1),
        time.min,
        tzinfo=UTC,
    )
    work_units = _build_work_units(
        memberships,
        start_utc=start_utc,
        end_exclusive_utc=end_exclusive_utc,
        chunk_days=chunk_days,
        provider_symbol_for=provider_symbol_for,
    )
    if not work_units:
        raise DataReadinessError("news history request has no effective membership chunks")

    request = {
        "schema": NEWS_HISTORY_REQUEST_SCHEMA,
        "memberships_path": str(memberships_path.resolve()),
        "memberships_sha256": str(membership_manifest["artifact_sha256"]),
        "start_utc": start_utc.isoformat(),
        "end_exclusive_utc": end_exclusive_utc.isoformat(),
        "source": "alpaca:benzinga",
        "availability_policy": "provider_publication_proxy",
        "production_ready": False,
        "include_content": True,
        "page_limit": 50,
        "chunk_days": chunk_days,
        "work_units": [_work_unit_record(unit) for unit in work_units],
    }
    request_hash = _sha256_json(request)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_manifest_path = out_dir / "_manifest.json"
    if final_manifest_path.exists():
        raise DataReadinessError(
            f"completed Alpaca news collection is immutable: {final_manifest_path}"
        )
    _write_or_validate_request(out_dir / "_request.json", request, request_hash)

    events_dir = out_dir / "events"
    pages_dir = out_dir / "raw_pages"
    attempts_dir = out_dir / "attempts"
    events_dir.mkdir(parents=True, exist_ok=True)
    pages_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(parents=True, exist_ok=True)
    assert_memory_budget(
        hard_budget_gib=memory_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="Alpaca news history collection start",
    )

    latest_attempts = _latest_attempts(attempts_dir)
    observed: dict[str, dict[str, Any]] = {}
    empty: set[str] = set()
    pending: list[_WorkUnit] = []
    skipped = 0
    for unit in work_units:
        previous = latest_attempts.get(unit.chunk_id)
        if previous is not None and previous["collection"].status == "observed_empty":
            empty.add(unit.chunk_id)
            skipped += 1
            continue
        existing = _load_existing_chunk(
            unit=unit,
            path=events_dir / f"{unit.chunk_id}.parquet",
            request_hash=request_hash,
        )
        if existing is not None:
            observed[unit.chunk_id] = existing
            skipped += 1
        else:
            pending.append(unit)

    failures: dict[str, str] = {}

    def collect_unit(
        unit: _WorkUnit,
    ) -> tuple[_WorkUnit, dict[str, Any] | None, SourceCollection, dict[str, int]]:
        started_at = datetime.now(UTC)
        collection_id = f"alpaca-news-{unit.chunk_id}-{uuid4().hex}"
        stats = {
            "provider_rows": 0,
            "accepted_rows": 0,
            "duplicate_rows": 0,
            "symbol_mismatch_rows": 0,
            "invalid_timestamp_rows": 0,
            "outside_window_rows": 0,
            "pages": 0,
        }
        try:
            page_payloads = _collect_pages(
                unit=unit,
                page_dir=pages_dir / unit.chunk_id,
                request_hash=request_hash,
                fetch_page=fetch_page,
                memory_budget_gib=memory_budget_gib,
                memory_headroom_gib=memory_headroom_gib,
            )
            stats["pages"] = len(page_payloads)
            raw, page_inputs, normalized_stats = _normalize_pages(
                unit,
                page_payloads,
            )
            stats.update(normalized_stats)
            completed_at = datetime.now(UTC)
            if raw.empty:
                collection = SourceCollection(
                    collection_id=collection_id,
                    ticker=unit.ticker,
                    source_family="alpaca",
                    requested_start_utc=unit.start_utc,
                    requested_end_utc=unit.end_exclusive_utc,
                    started_at_utc=started_at,
                    completed_at_utc=completed_at,
                    status="observed_empty",
                    row_count=0,
                )
                return unit, None, collection, stats

            events = canonicalize_events(
                raw,
                availability_policy="provider_publication_proxy",
            )
            audit = CanonicalAuditReport(
                checks=audit_canonical_events(events, require_observed=False)
            )
            path = events_dir / f"{unit.chunk_id}.parquet"
            manifest = write_canonical_artifact(
                events,
                path,
                artifact_type="events",
                audit=audit,
                inputs={
                    "collection_request_sha256": request_hash,
                    "chunk_id": unit.chunk_id,
                    **page_inputs,
                },
                production_ready=False,
            )
            completed_at = datetime.now(UTC)
            collection = SourceCollection(
                collection_id=collection_id,
                ticker=unit.ticker,
                source_family="alpaca",
                requested_start_utc=unit.start_utc,
                requested_end_utc=unit.end_exclusive_utc,
                started_at_utc=started_at,
                completed_at_utc=completed_at,
                status="observed",
                row_count=len(events),
            )
            return (
                unit,
                _artifact_record(unit, path, events, manifest),
                collection,
                stats,
            )
        except Exception as exc:
            collection = SourceCollection(
                collection_id=collection_id,
                ticker=unit.ticker,
                source_family="alpaca",
                requested_start_utc=unit.start_utc,
                requested_end_utc=unit.end_exclusive_utc,
                started_at_utc=started_at,
                completed_at_utc=datetime.now(UTC),
                status="failed",
                row_count=0,
                error_type=type(exc).__name__,
            )
            return (
                unit,
                {"error": f"{type(exc).__name__}: {str(exc)[:500]}"},
                collection,
                stats,
            )
        finally:
            release_process_memory()

    with ThreadPoolExecutor(max_workers=min(workers, len(pending) or 1)) as executor:
        futures = {
            executor.submit(collect_unit, unit): unit.chunk_id for unit in pending
        }
        for future in as_completed(futures):
            unit, artifact, collection, stats = future.result()
            _write_attempt(
                attempts_dir,
                request_hash=request_hash,
                unit=unit,
                collection=collection,
                artifact=artifact,
                stats=stats,
            )
            if artifact is not None and "error" not in artifact:
                observed[unit.chunk_id] = artifact
            elif collection.status == "observed_empty":
                empty.add(unit.chunk_id)
            else:
                failures[unit.chunk_id] = (
                    str(artifact["error"])
                    if artifact is not None
                    else "DataReadinessError: Alpaca news chunk failed"
                )
            assert_memory_budget(
                hard_budget_gib=memory_budget_gib,
                headroom_gib=memory_headroom_gib,
                stage=f"Alpaca news persist {unit.chunk_id}",
            )

    ledger = _latest_collection_ledger(
        attempts_dir,
        work_units,
        observed=observed,
    )
    ledger_path = out_dir / "_source_collections.parquet"
    ledger_audit = CanonicalAuditReport(
        checks=audit_source_collections(ledger, require_success=False)
    )
    ledger_manifest = write_canonical_artifact(
        ledger,
        ledger_path,
        artifact_type="source_collections",
        audit=ledger_audit,
        inputs={"collection_request_sha256": request_hash},
        production_ready=False,
    )
    memory = memory_audit(
        hard_budget_gib=memory_budget_gib,
        headroom_gib=memory_headroom_gib,
    ).to_record()
    terminal = len(observed) + len(empty)
    status = "complete" if not failures and terminal == len(work_units) else "incomplete"
    status_payload: dict[str, Any] = {
        "schema": NEWS_HISTORY_MANIFEST_SCHEMA,
        "request_sha256": request_hash,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "status": status,
        "requested_chunks": len(work_units),
        "observed_chunks": len(observed),
        "empty_chunks": len(empty),
        "failed_chunks": failures,
        "skipped_chunks": skipped,
        "source_collections_path": str(ledger_path),
        "source_collections_sha256": str(ledger_manifest["artifact_sha256"]),
        "memory": memory,
        "production_ready": False,
        "availability_policy": "provider_publication_proxy",
    }
    status_path = out_dir / "_status.json"
    _atomic_json(status_path, status_payload)
    if status == "incomplete":
        return NewsHistoryCollectionResult(
            status=status,
            requested_chunks=len(work_units),
            observed_chunks=len(observed),
            empty_chunks=len(empty),
            failed_chunks=tuple(sorted(failures)),
            skipped_chunks=skipped,
            manifest_path=None,
            status_path=status_path,
        )

    artifacts = [
        observed[unit.chunk_id]
        for unit in work_units
        if unit.chunk_id in observed
    ]
    final_payload = {
        **status_payload,
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
        "total_rows": sum(int(record["rows"]) for record in artifacts),
    }
    _atomic_json(final_manifest_path, final_payload)
    return NewsHistoryCollectionResult(
        status=status,
        requested_chunks=len(work_units),
        observed_chunks=len(observed),
        empty_chunks=len(empty),
        failed_chunks=(),
        skipped_chunks=skipped,
        manifest_path=final_manifest_path,
        status_path=status_path,
    )


def _build_work_units(
    memberships: pd.DataFrame,
    *,
    start_utc: datetime,
    end_exclusive_utc: datetime,
    chunk_days: int,
    provider_symbol_for: Callable[[str], str],
) -> list[_WorkUnit]:
    units: list[_WorkUnit] = []
    ordered = memberships.sort_values(
        ["security_id", "effective_from_utc", "ticker"],
        kind="stable",
    )
    for row in ordered.to_dict(orient="records"):
        ticker = canonical_symbol(str(row["ticker"]))
        security_id = str(row["security_id"]).strip()
        effective_from = pd.Timestamp(row["effective_from_utc"])
        if effective_from.tzinfo is None:
            raise DataReadinessError("membership effective_from_utc must be timezone-aware")
        effective_from_utc = effective_from.tz_convert("UTC").to_pydatetime()
        raw_end = row.get("effective_to_utc")
        if raw_end is None or pd.isna(raw_end):
            effective_to_utc = end_exclusive_utc
        else:
            effective_to = pd.Timestamp(raw_end)
            if effective_to.tzinfo is None:
                raise DataReadinessError(
                    "membership effective_to_utc must be timezone-aware"
                )
            effective_to_utc = effective_to.tz_convert("UTC").to_pydatetime()
        segment_start = max(start_utc, effective_from_utc)
        segment_end = min(end_exclusive_utc, effective_to_utc)
        cursor = segment_start
        while cursor < segment_end:
            chunk_end = min(segment_end, cursor + timedelta(days=chunk_days))
            identity = "|".join(
                (
                    security_id,
                    ticker,
                    cursor.isoformat(),
                    chunk_end.isoformat(),
                )
            )
            chunk_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24]
            units.append(
                _WorkUnit(
                    chunk_id=chunk_id,
                    ticker=ticker,
                    provider_symbol=provider_symbol_for(ticker),
                    security_id=security_id,
                    start_utc=cursor,
                    end_exclusive_utc=chunk_end,
                )
            )
            cursor = chunk_end
    identities = [unit.chunk_id for unit in units]
    if len(identities) != len(set(identities)):
        raise DataReadinessError("news history membership chunks are duplicated")
    return units


def _collect_pages(
    *,
    unit: _WorkUnit,
    page_dir: Path,
    request_hash: str,
    fetch_page: NewsPageFetcher,
    memory_budget_gib: float,
    memory_headroom_gib: float,
) -> list[dict[str, Any]]:
    page_dir.mkdir(parents=True, exist_ok=True)
    pages = _load_pages(page_dir, unit=unit, request_hash=request_hash)
    if pages and pages[-1]["next_page_token"] is None:
        return pages
    token = str(pages[-1]["next_page_token"]) if pages else None
    seen_tokens = {
        str(page["request_page_token"])
        for page in pages
        if page["request_page_token"] is not None
    }
    index = len(pages)
    while True:
        assert_memory_budget(
            hard_budget_gib=memory_budget_gib,
            headroom_gib=memory_headroom_gib,
            stage=f"Alpaca news page {unit.chunk_id}:{index}",
        )
        page = fetch_page(
            unit.provider_symbol,
            unit.start_utc,
            unit.end_exclusive_utc,
            token,
        )
        if page.request_page_token != token:
            raise DataReadinessError(
                f"Alpaca page request token mismatch for {unit.chunk_id}"
            )
        if token is not None:
            seen_tokens.add(token)
        if (
            page.next_page_token is not None
            and page.next_page_token in seen_tokens
        ):
            raise DataReadinessError(
                f"Alpaca page token repeated for {unit.chunk_id}"
            )
        payload = {
            "schema": NEWS_PAGE_SCHEMA,
            "collection_request_sha256": request_hash,
            "chunk_id": unit.chunk_id,
            "page_index": index,
            "request_page_token": page.request_page_token,
            "next_page_token": page.next_page_token,
            "collected_at_utc": datetime.now(UTC).isoformat(),
            "news": list(page.news),
        }
        payload["content_sha256"] = _sha256_json(payload)
        page_path = page_dir / f"page_{index:06d}.json"
        _atomic_json(page_path, payload)
        pages.append(
            {
                **payload,
                "_page_path": str(page_path),
                "_page_sha256": file_sha256(page_path),
            }
        )
        index += 1
        token = page.next_page_token
        if token is None:
            return pages


def _load_pages(
    page_dir: Path,
    *,
    unit: _WorkUnit,
    request_hash: str,
) -> list[dict[str, Any]]:
    paths = sorted(page_dir.glob("page_*.json"))
    pages: list[dict[str, Any]] = []
    expected_token: str | None = None
    for expected_index, path in enumerate(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        content_hash = payload.get("content_sha256")
        content_payload = {
            key: value
            for key, value in payload.items()
            if key != "content_sha256"
        }
        if (
            payload.get("schema") != NEWS_PAGE_SCHEMA
            or content_hash != _sha256_json(content_payload)
            or payload.get("collection_request_sha256") != request_hash
            or payload.get("chunk_id") != unit.chunk_id
            or payload.get("page_index") != expected_index
            or payload.get("request_page_token") != expected_token
            or not isinstance(payload.get("news"), list)
        ):
            raise DataReadinessError(f"invalid resumable Alpaca news page: {path}")
        pages.append(
            {
                **payload,
                "_page_path": str(path),
                "_page_sha256": file_sha256(path),
            }
        )
        expected_token = payload.get("next_page_token")
        if expected_token is None and expected_index != len(paths) - 1:
            raise DataReadinessError(
                f"Alpaca page exists after terminal token: {path}"
            )
    return pages


def _normalize_pages(
    unit: _WorkUnit,
    pages: list[dict[str, Any]],
) -> tuple[pd.DataFrame, dict[str, str], dict[str, int]]:
    records: list[dict[str, Any]] = []
    stats = {
        "provider_rows": 0,
        "accepted_rows": 0,
        "duplicate_rows": 0,
        "symbol_mismatch_rows": 0,
        "invalid_timestamp_rows": 0,
        "outside_window_rows": 0,
    }
    page_inputs: dict[str, str] = {}
    for page in pages:
        page_inputs[str(page["_page_path"])] = str(page["_page_sha256"])
        first_seen = pd.Timestamp(page["collected_at_utc"])
        for item in page["news"]:
            stats["provider_rows"] += 1
            created = pd.to_datetime(item.get("created_at"), utc=True, errors="coerce")
            updated = pd.to_datetime(item.get("updated_at"), utc=True, errors="coerce")
            if pd.isna(created) or (pd.notna(updated) and updated < created):
                stats["invalid_timestamp_rows"] += 1
                continue
            created_at = pd.Timestamp(created)
            if not (
                created_at >= pd.Timestamp(unit.start_utc)
                and created_at < pd.Timestamp(unit.end_exclusive_utc)
            ):
                stats["outside_window_rows"] += 1
                continue
            symbols = item.get("symbols")
            attached = (
                {canonical_symbol(str(symbol)) for symbol in symbols}
                if isinstance(symbols, list)
                else set()
            )
            if unit.ticker not in attached:
                stats["symbol_mismatch_rows"] += 1
                continue
            title = str(item.get("headline") or "").strip()
            if not title:
                stats["invalid_timestamp_rows"] += 1
                continue
            provider_id = str(item.get("id") or "").strip()
            fallback = "|".join(
                (
                    created_at.isoformat(),
                    title,
                    str(item.get("url") or ""),
                )
            )
            identity = provider_id or hashlib.sha256(
                fallback.encode("utf-8")
            ).hexdigest()
            records.append(
                {
                    "_provider_identity": identity,
                    "_revision_time": pd.Timestamp(updated)
                    if pd.notna(updated)
                    else created_at,
                    "ticker": unit.ticker,
                    "security_id": unit.security_id,
                    "timestamp": created_at,
                    "published_at_utc": created_at,
                    "provider_updated_at_utc": (
                        pd.Timestamp(updated) if pd.notna(updated) else pd.NaT
                    ),
                    "first_seen_at_utc": first_seen,
                    "source": f"alpaca:{item.get('source') or 'unknown'}",
                    "title": title,
                    "url": str(item.get("url") or ""),
                    "summary": str(item.get("summary") or ""),
                    "text": str(item.get("content") or item.get("summary") or ""),
                    "raw": item,
                }
            )
    if not records:
        return pd.DataFrame(), page_inputs, stats
    raw = pd.DataFrame.from_records(records)
    raw = raw.sort_values(
        ["_provider_identity", "_revision_time"],
        kind="stable",
    )
    before = len(raw)
    raw = raw.drop_duplicates("_provider_identity", keep="last")
    stats["duplicate_rows"] = before - len(raw)
    stats["accepted_rows"] = len(raw)
    return (
        raw.drop(columns=["_provider_identity", "_revision_time"]).reset_index(
            drop=True
        ),
        page_inputs,
        stats,
    )


def _load_existing_chunk(
    *,
    unit: _WorkUnit,
    path: Path,
    request_hash: str,
) -> dict[str, Any] | None:
    if not path.exists() and not manifest_path_for(path).exists():
        return None
    events, manifest = load_canonical_artifact(
        path,
        expected_type="events",
        allow_research=True,
    )
    inputs = manifest.get("inputs")
    if (
        not isinstance(inputs, dict)
        or inputs.get("collection_request_sha256") != request_hash
        or inputs.get("chunk_id") != unit.chunk_id
    ):
        raise DataReadinessError(
            f"existing Alpaca news artifact has another request identity: {path}"
        )
    if bool(events["security_id"].astype(str).ne(unit.security_id).any()):
        raise DataReadinessError(
            f"existing Alpaca news artifact has another security identity: {path}"
        )
    audit = CanonicalAuditReport(
        checks=audit_canonical_events(events, require_observed=False)
    )
    audit.raise_for_failure()
    return _artifact_record(unit, path, events, manifest)


def _artifact_record(
    unit: _WorkUnit,
    path: Path,
    events: pd.DataFrame,
    manifest: dict[str, object],
) -> dict[str, Any]:
    published = pd.to_datetime(events["published_at_utc"], utc=True)
    return {
        **_work_unit_record(unit),
        "path": str(path),
        "manifest_path": str(manifest_path_for(path)),
        "sha256": str(manifest["artifact_sha256"]),
        "rows": len(events),
        "first_published_at_utc": published.min().isoformat(),
        "last_published_at_utc": published.max().isoformat(),
        "availability_policy": "provider_publication_proxy",
        "production_ready": False,
    }


def _write_attempt(
    attempts_dir: Path,
    *,
    request_hash: str,
    unit: _WorkUnit,
    collection: SourceCollection,
    artifact: dict[str, Any] | None,
    stats: dict[str, int],
) -> None:
    payload = {
        "schema": NEWS_HISTORY_REQUEST_SCHEMA,
        "collection_request_sha256": request_hash,
        "work_unit": _work_unit_record(unit),
        "collection": collection.model_dump(mode="json"),
        "artifact": artifact,
        "stats": stats,
    }
    path = attempts_dir / (
        f"{unit.chunk_id}_{collection.collection_id}.json"
    )
    _atomic_json(path, payload)


def _latest_attempts(
    attempts_dir: Path,
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in attempts_dir.glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        unit = payload.get("work_unit")
        if not isinstance(unit, dict):
            raise DataReadinessError(f"invalid Alpaca news attempt: {path}")
        chunk_id = str(unit.get("chunk_id", ""))
        collection = SourceCollection.model_validate(payload["collection"])
        previous = latest.get(chunk_id)
        if (
            previous is None
            or collection.completed_at_utc
            > previous["collection"].completed_at_utc
        ):
            latest[chunk_id] = {
                "collection": collection,
                "payload": payload,
            }
    return latest


def _latest_collection_ledger(
    attempts_dir: Path,
    work_units: list[_WorkUnit],
    *,
    observed: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    latest = _latest_attempts(attempts_dir)
    rows: list[dict[str, Any]] = []
    now = datetime.now(UTC)
    for unit in work_units:
        attempt = latest.get(unit.chunk_id)
        if attempt is None and unit.chunk_id in observed:
            collection = SourceCollection(
                collection_id=f"resumed-{unit.chunk_id}-{uuid4().hex}",
                ticker=unit.ticker,
                source_family="alpaca",
                requested_start_utc=unit.start_utc,
                requested_end_utc=unit.end_exclusive_utc,
                started_at_utc=now,
                completed_at_utc=now,
                status="observed",
                row_count=int(observed[unit.chunk_id]["rows"]),
            )
            stats: dict[str, int] = {}
        elif attempt is not None:
            collection = attempt["collection"]
            raw_stats = attempt["payload"].get("stats", {})
            stats = (
                {str(key): int(value) for key, value in raw_stats.items()}
                if isinstance(raw_stats, dict)
                else {}
            )
        else:
            continue
        rows.append(
            {
                **collection.model_dump(),
                "security_id": unit.security_id,
                "chunk_id": unit.chunk_id,
                "provider_symbol": unit.provider_symbol,
                **stats,
            }
        )
    if not rows:
        return pd.DataFrame(columns=list(SourceCollection.model_fields))
    return pd.DataFrame(rows).sort_values(
        ["ticker", "requested_start_utc", "chunk_id"],
        kind="stable",
    ).reset_index(drop=True)


def _work_unit_record(unit: _WorkUnit) -> dict[str, str]:
    return {
        "chunk_id": unit.chunk_id,
        "ticker": unit.ticker,
        "provider_symbol": unit.provider_symbol,
        "security_id": unit.security_id,
        "start_utc": unit.start_utc.isoformat(),
        "end_exclusive_utc": unit.end_exclusive_utc.isoformat(),
    }


def _write_or_validate_request(
    path: Path,
    request: dict[str, Any],
    request_hash: str,
) -> None:
    payload = {**request, "request_sha256": request_hash}
    if path.exists():
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if loaded != payload:
            raise DataReadinessError(
                f"Alpaca news resume request does not match {path}"
            )
        return
    _atomic_json(path, payload)


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(
                payload,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
