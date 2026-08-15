"""Immutable, resumable collection of official S&P 500 change releases.

Unit hashes detect accidental corruption during collection. They are not digital
signatures and do not defend against an operator deliberately rewriting the raw
object, sidecar, and checksum together. The final authority anchors the fully
replayed unit set for downstream lineage checks.
"""

from __future__ import annotations

import hashlib
import json
import re
import threading
import uuid
import zlib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol, cast
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse
from zoneinfo import ZoneInfo

from bs4 import BeautifulSoup

from market_predictor.locking import LockTimeout, file_lock
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
)
from market_predictor.sources.http import HttpClient
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.universe import ARCHIVE_QUERY, SP_GLOBAL_ARCHIVE_URL

_SOURCE_TIMEZONE = ZoneInfo("America/New_York")

ARCHIVE_REQUEST_SCHEMA = "ml_v3.spglobal_official_archive_request.v1"
ARCHIVE_STATUS_SCHEMA = "ml_v3.spglobal_official_archive_status.v1"
ARCHIVE_MANIFEST_SCHEMA = "ml_v3.spglobal_official_archive_manifest.v2"
ARCHIVE_AUTHORITY_SCHEMA = "ml_v3.spglobal_official_archive_authority.v2"
ARCHIVE_UNIT_SCHEMA = "ml_v3.spglobal_official_archive_unit.v1"
DISCOVERY_SCHEMA = "ml_v3.spglobal_official_archive_discovery.v1"
DISCOVERY_START = date(2018, 4, 14)
DISCOVERY_END = date(2026, 7, 8)
EXPECTED_SEED_URLS = 83
SEARCH_PAGE_SIZE = int(ARCHIVE_QUERY["l"])
SEARCH_PAGE_STRIDE = SEARCH_PAGE_SIZE - 1
MAXIMUM_WORKERS = 2
MAXIMUM_MEMORY_GIB = 4.0
MEMORY_HEADROOM_GIB = 0.75
MAXIMUM_RESPONSE_BYTES = 16 * 1024 * 1024
MAXIMUM_DECODED_BYTES = 4 * 1024 * 1024
SOURCE_AUDIT_SCHEMA = "ml_v3.sp500_point_in_time_universe.v1"
SOURCE_MANIFEST_SCHEMA = "ml_v3.sp500_change_sources.v1"

_PUBLISHED_DATE = re.compile(r"/(20\d{2}-\d{2}-\d{2})-")
_MEMBERSHIP_TITLE = re.compile(
    r"(?:\bS&P\s*500\b.{0,180}\b(?:join|joins|joining|replace|replaces|replacing|change|changes|changed|addition|deletion|added|deleted)\b"
    r"|\b(?:join|joins|joining|replace|replaces|replacing|change|changes|changed|addition|deletion|added|deleted)\b.{0,180}\bS&P\s*500\b)",
    re.IGNORECASE,
)
_OBJECT_WRITE_LOCK = threading.Lock()


class BytesResponse(Protocol):
    body: bytes
    requested_url: str
    final_url: str
    redirect_chain: Sequence[str]
    status_code: int
    retrieved_at_utc: datetime | str
    content_type: str | None
    content_encoding: str | None
    etag: str | None
    last_modified: str | None
    body_length: int
    sha256: str
    body_representation: str


class BytesHttpClient(Protocol):
    def get_bytes_with_metadata(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        retries: int = 3,
        pause: float = 1.0,
        maximum_body_bytes: int = MAXIMUM_RESPONSE_BYTES,
    ) -> BytesResponse: ...


ClientFactory = Callable[[], BytesHttpClient]


@dataclass(frozen=True)
class ArchiveCollectionConfig:
    discovery_end: date = DISCOVERY_END
    maximum_pages: int = 20
    workers: int = 1
    retries: int = 3
    retry_pause_seconds: float = 1.0
    maximum_units_this_run: int | None = None

    def validate(self) -> None:
        if self.discovery_end < DISCOVERY_START:
            raise ValueError(
                f"discovery_end must be on or after {DISCOVERY_START.isoformat()}"
            )
        if self.maximum_pages < 1:
            raise ValueError("maximum_pages must be positive")
        if not 1 <= self.workers <= MAXIMUM_WORKERS:
            raise ValueError(f"workers must be between 1 and {MAXIMUM_WORKERS}")
        if self.retries < 1:
            raise ValueError("retries must be positive")
        if self.retry_pause_seconds < 0:
            raise ValueError("retry_pause_seconds must not be negative")
        if self.maximum_units_this_run is not None and self.maximum_units_this_run < 1:
            raise ValueError("maximum_units_this_run must be positive")


@dataclass(frozen=True)
class VerifiedSpGlobalRawArchive:
    root: Path
    authority: dict[str, Any]
    manifest: dict[str, Any]
    releases: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _Announcement:
    published_date: date
    title: str
    url: str
    origin: str

    def to_record(self) -> dict[str, str]:
        return {
            "published_date": self.published_date.isoformat(),
            "title": self.title,
            "url": self.url,
            "origin": self.origin,
        }


def collect_spglobal_archive(
    *,
    source_audit_path: Path,
    expected_source_audit_sha256: str,
    output_directory: Path,
    client_factory: ClientFactory | None = None,
    config: ArchiveCollectionConfig | None = None,
) -> dict[str, Any]:
    """Collect and authorize exact official release bytes with verified resume."""

    policy = config or ArchiveCollectionConfig()
    policy.validate()
    output_directory.mkdir(parents=True, exist_ok=True)
    try:
        with file_lock(output_directory / "_collector", timeout=0.0):
            return _collect_spglobal_archive_locked(
                source_audit_path=source_audit_path,
                expected_source_audit_sha256=expected_source_audit_sha256,
                output_directory=output_directory,
                client_factory=client_factory,
                config=policy,
            )
    except LockTimeout as exc:
        raise DataReadinessError(
            f"another collector is already writing S&P archive {output_directory}"
        ) from exc


def _collect_spglobal_archive_locked(
    *,
    source_audit_path: Path,
    expected_source_audit_sha256: str,
    output_directory: Path,
    client_factory: ClientFactory | None,
    config: ArchiveCollectionConfig,
) -> dict[str, Any]:
    policy = config
    assert_memory_budget(
        hard_budget_gib=MAXIMUM_MEMORY_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage="S&P official archive collection start",
    )
    if (output_directory / "_authority.json").exists():
        raise DataReadinessError("completed S&P official archive is immutable")
    audit_sha256 = _file_sha256(source_audit_path)
    if audit_sha256 != expected_source_audit_sha256:
        raise DataReadinessError("source audit SHA-256 does not match the frozen request")
    seeds = _load_seed_announcements(source_audit_path)
    if any(seed.published_date > policy.discovery_end for seed in seeds):
        raise DataReadinessError(
            "source audit contains a frozen seed published after discovery_end"
        )
    request_payload: dict[str, Any] = {
        "schema": ARCHIVE_REQUEST_SCHEMA,
        "source_audit_path": str(source_audit_path),
        "source_audit_sha256": audit_sha256,
        "seed_url_count": len(seeds),
        "seed_urls": [item.url for item in seeds],
        "discovery_start": DISCOVERY_START.isoformat(),
        "discovery_end": policy.discovery_end.isoformat(),
        "archive_url": SP_GLOBAL_ARCHIVE_URL,
        "archive_query": dict(sorted(ARCHIVE_QUERY.items())),
    }
    request_sha256 = _json_sha256(request_payload)
    request = {**request_payload, "request_sha256": request_sha256}
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(output_directory / "_request.json", request)

    factory = client_factory or _default_client_factory
    budget = policy.maximum_units_this_run
    discovery, discovery_network_units, discovery_error = _discover_or_resume(
        output_directory=output_directory,
        seeds=seeds,
        request_sha256=request_sha256,
        client_factory=factory,
        config=policy,
        discovery_end=policy.discovery_end,
        network_budget=budget,
    )
    used_network_units = discovery_network_units
    if discovery is None:
        return _publish_status(
            output_directory,
            request_sha256=request_sha256,
            status="incomplete",
            stop_reason=discovery_error or "operational_batch_limit",
            discovery_complete=False,
            requested_releases=0,
            completed_releases=0,
            resumed_releases=0,
            failed_releases={},
            network_units=used_network_units,
        )

    announcements = [_announcement_from_record(item) for item in _records(discovery, "announcements")]
    remaining_budget = None if budget is None else max(0, budget - used_network_units)
    completed, resumed, failures, release_network_units = _collect_releases(
        output_directory=output_directory,
        announcements=announcements,
        request_sha256=request_sha256,
        client_factory=factory,
        config=policy,
        network_budget=remaining_budget,
    )
    used_network_units += release_network_units
    all_complete = len(completed) == len(announcements) and not failures
    if not all_complete:
        return _publish_status(
            output_directory,
            request_sha256=request_sha256,
            status="incomplete",
            stop_reason=(
                "release_failures" if failures else "operational_batch_limit"
            ),
            discovery_complete=True,
            requested_releases=len(announcements),
            completed_releases=len(completed),
            resumed_releases=resumed,
            failed_releases=failures,
            network_units=used_network_units,
        )

    discovery, release_records = _replay_complete_archive(
        output_directory,
        seeds=seeds,
        request_sha256=request_sha256,
        discovery_end=policy.discovery_end,
    )
    assert_peak_memory_budget(
        hard_budget_gib=MAXIMUM_MEMORY_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage="S&P official archive publication",
    )
    status = _status_payload(
        request_sha256=request_sha256,
        status="complete",
        stop_reason="complete",
        discovery_complete=True,
        requested_releases=len(announcements),
        completed_releases=len(release_records),
        resumed_releases=resumed,
        failed_releases={},
        network_units=used_network_units,
    )
    status["archive_scope"] = "raw_official_responses"
    discovery_path = output_directory / "_discovery.json"
    raw_release_records = [_raw_release_record(record) for record in release_records]
    manifest: dict[str, Any] = {
        **status,
        "schema": ARCHIVE_MANIFEST_SCHEMA,
        "source_audit_sha256": audit_sha256,
        "seed_url_count": len(seeds),
        "discovered_url_count": int(discovery["discovered_url_count"]),
        "release_url_count": len(announcements),
        "discovery_sha256": _file_sha256(discovery_path),
        "search_pages": _records(discovery, "search_pages"),
        "releases": raw_release_records,
        "release_set_sha256": _json_sha256(
            [_raw_release_identity(record) for record in raw_release_records]
        ),
    }
    _atomic_json(output_directory / "_manifest.json", manifest)
    authority = {
        "schema": ARCHIVE_AUTHORITY_SCHEMA,
        "state": "raw_complete",
        "archive_scope": "raw_official_responses",
        "artifact": "_manifest.json",
        "artifact_sha256": _file_sha256(output_directory / "_manifest.json"),
        "request_sha256": request_sha256,
        "source_audit_sha256": audit_sha256,
        "release_set_sha256": manifest["release_set_sha256"],
    }
    _atomic_json(output_directory / "_authority.json", authority)
    _atomic_json(output_directory / "_status.json", status)
    return manifest


def require_spglobal_raw_archive_complete(
    archive_directory: Path,
) -> VerifiedSpGlobalRawArchive:
    """Replay all retained units and verify immutable raw-source authority."""

    authority = _load_json(archive_directory / "_authority.json")
    artifact = _resolve_inside(
        archive_directory,
        str(authority.get("artifact", "")),
    )
    if (
        authority.get("schema") != ARCHIVE_AUTHORITY_SCHEMA
        or authority.get("state") != "raw_complete"
        or authority.get("archive_scope") != "raw_official_responses"
        or not artifact.is_file()
        or authority.get("artifact_sha256") != _file_sha256(artifact)
    ):
        raise DataReadinessError("S&P raw archive authority is invalid")
    manifest = _load_json(artifact)
    releases = _records(manifest, "releases")
    release_count = len(releases)
    release_set_sha256 = _json_sha256(
        [_raw_release_identity(record) for record in releases]
    )
    if (
        manifest.get("schema") != ARCHIVE_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("archive_scope") != "raw_official_responses"
        or manifest.get("request_sha256") != authority.get("request_sha256")
        or manifest.get("source_audit_sha256")
        != authority.get("source_audit_sha256")
        or manifest.get("release_set_sha256") != release_set_sha256
        or authority.get("release_set_sha256") != release_set_sha256
        or int(manifest.get("requested_releases", -1)) != release_count
        or int(manifest.get("completed_releases", -1)) != release_count
        or int(manifest.get("release_url_count", -1)) != release_count
    ):
        raise DataReadinessError(
            "S&P raw archive manifest lineage or release counts are invalid"
        )
    request = _load_json(archive_directory / "_request.json")
    request_payload = {
        key: value for key, value in request.items() if key != "request_sha256"
    }
    if (
        request.get("schema") != ARCHIVE_REQUEST_SCHEMA
        or request.get("request_sha256") != _json_sha256(request_payload)
        or request.get("request_sha256") != authority.get("request_sha256")
    ):
        raise DataReadinessError(
            "S&P raw archive request identity is invalid"
        )
    try:
        discovery_start = date.fromisoformat(str(request.get("discovery_start", "")))
        discovery_end = date.fromisoformat(str(request.get("discovery_end", "")))
    except ValueError as exc:
        raise DataReadinessError(
            "S&P raw archive request discovery boundary is invalid"
        ) from exc
    if discovery_start != DISCOVERY_START or discovery_end < discovery_start:
        raise DataReadinessError(
            "S&P raw archive request discovery boundary is invalid"
        )
    discovery = _load_json(archive_directory / "_discovery.json")
    request_seed_urls = request.get("seed_urls")
    if not isinstance(request_seed_urls, list) or not all(
        isinstance(url, str) for url in request_seed_urls
    ):
        raise DataReadinessError("S&P raw archive request seed set is invalid")
    seeds = [
        _announcement_from_record(record)
        for record in _records(discovery, "announcements")
        if record.get("origin") == "seed"
    ]
    if (
        len(seeds) != EXPECTED_SEED_URLS
        or [seed.url for seed in seeds] != request_seed_urls
    ):
        raise DataReadinessError(
            "S&P raw archive discovery does not retain the frozen seed set"
        )
    replayed_discovery, replayed_releases = _replay_complete_archive(
        archive_directory,
        seeds=seeds,
        request_sha256=str(authority["request_sha256"]),
        discovery_end=discovery_end,
    )
    if (
        replayed_discovery != discovery
        or [_raw_release_record(record) for record in replayed_releases]
        != releases
        or manifest.get("discovery_sha256")
        != _file_sha256(archive_directory / "_discovery.json")
    ):
        raise DataReadinessError(
            "S&P raw archive manifest does not equal the retained unit replay"
        )
    return VerifiedSpGlobalRawArchive(
        root=archive_directory.resolve(),
        authority=authority,
        manifest=manifest,
        releases=tuple(releases),
    )


def read_verified_spglobal_release_html(
    archive: VerifiedSpGlobalRawArchive,
    record: Mapping[str, Any],
) -> str:
    """Decode one release that belongs to a fully replayed raw archive."""

    identity = (record.get("url"), record.get("sha256"), record.get("unit_id"))
    if not any(
        (item.get("url"), item.get("sha256"), item.get("unit_id")) == identity
        for item in archive.releases
    ):
        raise DataReadinessError(
            "S&P release is not a member of the verified raw archive"
        )
    return _decode_html(
        _decode_http_entity(
            _read_unit_body(archive.root, record),
            str(record.get("content_encoding") or ""),
        ),
        str(record.get("content_type") or ""),
    )


def _replay_complete_archive(
    root: Path,
    *,
    seeds: list[_Announcement],
    request_sha256: str,
    discovery_end: date,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reload every retained unit before publishing final authority."""

    discovery = _load_json(root / "_discovery.json")
    _validate_discovery(
        discovery,
        root=root,
        seeds=seeds,
        request_sha256=request_sha256,
        discovery_end=discovery_end,
    )
    announcements = [
        _announcement_from_record(item)
        for item in _records(discovery, "announcements")
    ]
    expected_release_units = {
        f"{_release_unit_id(item.url)}.json" for item in announcements
    }
    release_directory = root / "units" / "releases"
    observed_release_units = (
        {path.name for path in release_directory.glob("*.json")}
        if release_directory.is_dir()
        else set()
    )
    if observed_release_units != expected_release_units:
        raise DataReadinessError(
            "S&P official archive release sidecars do not match discovery"
        )
    release_records: list[dict[str, Any]] = []
    for announcement in sorted(announcements, key=lambda item: item.url):
        record = _load_unit(
            root,
            kind="release",
            unit_id=_release_unit_id(announcement.url),
            identity=announcement.url,
            request_sha256=request_sha256,
        )
        if record is None:
            raise DataReadinessError(
                f"S&P official archive release is missing: {announcement.url}"
            )
        _validate_release_record(record, announcement)
        release_records.append(record)
    return discovery, release_records


def _discover_or_resume(
    *,
    output_directory: Path,
    seeds: list[_Announcement],
    request_sha256: str,
    client_factory: ClientFactory,
    config: ArchiveCollectionConfig,
    discovery_end: date,
    network_budget: int | None,
) -> tuple[dict[str, Any] | None, int, str | None]:
    discovery_path = output_directory / "_discovery.json"
    if discovery_path.exists():
        discovery = _load_json(discovery_path)
        _validate_discovery(
            discovery,
            root=output_directory,
            seeds=seeds,
            request_sha256=request_sha256,
            discovery_end=discovery_end,
        )
        return discovery, 0, None

    discovered: dict[str, _Announcement] = {}
    page_records: list[dict[str, Any]] = []
    network_units = 0
    boundary_reached = False
    previous_last_url: str | None = None
    upper_boundary_crossed = False
    seen_search_urls: set[str] = set()
    client = client_factory()
    for page_number in range(config.maximum_pages):
        offset = page_number * SEARCH_PAGE_STRIDE
        params = {**ARCHIVE_QUERY, "o": str(offset)}
        identity = _search_identity(params)
        unit_id = f"search-{page_number:04d}-{hashlib.sha256(identity.encode()).hexdigest()[:12]}"
        existing = _load_unit(
            output_directory,
            kind="search_page",
            unit_id=unit_id,
            identity=identity,
            request_sha256=request_sha256,
        )
        if existing is None:
            if network_budget is not None and network_units >= network_budget:
                return None, network_units, "operational_batch_limit"
            response = client.get_bytes_with_metadata(
                SP_GLOBAL_ARCHIVE_URL,
                params=params,
                retries=config.retries,
                pause=config.retry_pause_seconds,
                maximum_body_bytes=MAXIMUM_RESPONSE_BYTES,
            )
            assert_memory_budget(
                hard_budget_gib=MAXIMUM_MEMORY_GIB,
                headroom_gib=MEMORY_HEADROOM_GIB,
                stage="S&P archive search response",
            )
            record = _persist_search_page(
                output_directory,
                unit_id=unit_id,
                identity=identity,
                request_sha256=request_sha256,
                response=response,
                page_number=page_number,
                offset=offset,
            )
            network_units += 1
        else:
            record = existing
        page_records.append(record)
        body = _read_unit_body(output_directory, record)
        page_announcements, dated_urls = _parse_search_page(
            _decode_http_entity(body, str(record.get("content_encoding") or "")),
            base_url=str(record["final_url"]),
            content_type=str(record.get("content_type") or ""),
        )
        previous_last_url, upper_boundary_crossed = _validate_search_page_coverage(
            page_number=page_number,
            dated_urls=dated_urls,
            previous_last_url=previous_last_url,
            seen_urls=seen_search_urls,
            upper_boundary_crossed=upper_boundary_crossed,
            discovery_end=discovery_end,
            retrieved_at_utc=str(record["retrieved_at_utc"]),
        )
        for announcement in page_announcements:
            if DISCOVERY_START <= announcement.published_date <= discovery_end:
                discovered[announcement.url] = announcement
        if max(published for published, _ in dated_urls) < DISCOVERY_START:
            boundary_reached = True
            break

    if not boundary_reached or not upper_boundary_crossed:
        _publish_status(
            output_directory,
            request_sha256=request_sha256,
            status="incomplete",
            stop_reason=(
                "discovery_boundary_not_reached"
                if not boundary_reached
                else "discovery_upper_boundary_not_crossed"
            ),
            discovery_complete=False,
            requested_releases=0,
            completed_releases=0,
            resumed_releases=0,
            failed_releases={},
            network_units=network_units,
        )
        return (
            None,
            network_units,
            "discovery_boundary_not_reached"
            if not boundary_reached
            else "discovery_upper_boundary_not_crossed",
        )

    announcements = _union_announcements(seeds, discovered)
    discovery = {
        "schema": DISCOVERY_SCHEMA,
        "request_sha256": request_sha256,
        "lower_boundary": DISCOVERY_START.isoformat(),
        "upper_boundary": discovery_end.isoformat(),
        "lower_boundary_reached": True,
        "seed_url_count": len(seeds),
        "discovered_url_count": len(discovered),
        "release_url_count": len(announcements),
        "search_pages": page_records,
        "announcements": [item.to_record() for item in announcements],
    }
    _atomic_json(discovery_path, discovery)
    _validate_discovery(
        discovery,
        root=output_directory,
        seeds=seeds,
        request_sha256=request_sha256,
        discovery_end=discovery_end,
    )
    return discovery, network_units, None


def _collect_releases(
    *,
    output_directory: Path,
    announcements: list[_Announcement],
    request_sha256: str,
    client_factory: ClientFactory,
    config: ArchiveCollectionConfig,
    network_budget: int | None,
) -> tuple[dict[str, dict[str, Any]], int, dict[str, str], int]:
    completed: dict[str, dict[str, Any]] = {}
    pending: list[_Announcement] = []
    resumed = 0
    for announcement in announcements:
        unit_id = _release_unit_id(announcement.url)
        existing = _load_unit(
            output_directory,
            kind="release",
            unit_id=unit_id,
            identity=announcement.url,
            request_sha256=request_sha256,
        )
        if existing is None:
            pending.append(announcement)
        else:
            _validate_release_record(existing, announcement)
            completed[announcement.url] = existing
            resumed += 1
    scheduled = pending if network_budget is None else pending[:network_budget]
    failures: dict[str, str] = {}
    local = threading.local()

    def collect_one(announcement: _Announcement) -> dict[str, Any]:
        client = getattr(local, "client", None)
        if client is None:
            client = client_factory()
            local.client = client
        response = cast(BytesHttpClient, client).get_bytes_with_metadata(
            announcement.url,
            retries=config.retries,
            pause=config.retry_pause_seconds,
            maximum_body_bytes=MAXIMUM_RESPONSE_BYTES,
        )
        assert_memory_budget(
            hard_budget_gib=MAXIMUM_MEMORY_GIB,
            headroom_gib=MEMORY_HEADROOM_GIB,
            stage="S&P archive release response",
        )
        return _persist_release(
            output_directory,
            announcement=announcement,
            request_sha256=request_sha256,
            response=response,
        )

    futures: dict[Future[dict[str, Any]], _Announcement] = {}
    with ThreadPoolExecutor(max_workers=config.workers) as executor:
        for announcement in scheduled:
            futures[executor.submit(collect_one, announcement)] = announcement
        for future in as_completed(futures):
            announcement = futures[future]
            try:
                completed[announcement.url] = future.result()
            except Exception as exc:  # noqa: BLE001 - each source must have a durable terminal status
                failures[announcement.url] = f"{type(exc).__name__}: {exc}"
    return completed, resumed, failures, len(scheduled)


def _persist_search_page(
    root: Path,
    *,
    unit_id: str,
    identity: str,
    request_sha256: str,
    response: BytesResponse,
    page_number: int,
    offset: int,
) -> dict[str, Any]:
    metadata = _validate_response(response, expected_release=False)
    for observed_url in (
        response.requested_url,
        *response.redirect_chain,
        response.final_url,
    ):
        if not _same_search_identity(str(observed_url), identity):
            raise DataReadinessError(
                "official archive search redirect changed the query identity: "
                f"{identity} -> {observed_url}"
            )
    body_path = _write_content_addressed(root, response.body)
    record = {
        "schema": ARCHIVE_UNIT_SCHEMA,
        "kind": "search_page",
        "unit_id": unit_id,
        "identity": identity,
        "request_sha256": request_sha256,
        "page_number": page_number,
        "offset": offset,
        "path": str(body_path.relative_to(root)),
        **metadata,
    }
    sealed = _with_unit_integrity_hash(record)
    _write_new_json(_sidecar_path(root, "search_page", unit_id), sealed)
    return sealed


def _persist_release(
    root: Path,
    *,
    announcement: _Announcement,
    request_sha256: str,
    response: BytesResponse,
) -> dict[str, Any]:
    metadata = _validate_response(response, expected_release=True)
    if _canonical_release_path(response.final_url) != _canonical_release_path(
        announcement.url
    ):
        raise DataReadinessError(
            "official release redirect changed the release identity: "
            f"{announcement.url} -> {response.final_url}"
        )
    body = response.body
    body_path = _write_content_addressed(root, body)
    unit_id = _release_unit_id(announcement.url)
    record = {
        "schema": ARCHIVE_UNIT_SCHEMA,
        "kind": "release",
        "unit_id": unit_id,
        "identity": announcement.url,
        "request_sha256": request_sha256,
        "url": announcement.url,
        "title": announcement.title,
        "published_date": announcement.published_date.isoformat(),
        "origin": announcement.origin,
        "path": str(body_path.relative_to(root)),
        **metadata,
    }
    sealed = _with_unit_integrity_hash(record)
    _write_new_json(_sidecar_path(root, "release", unit_id), sealed)
    return sealed


def _validate_response(response: BytesResponse, *, expected_release: bool) -> dict[str, Any]:
    content = bytes(response.body)
    digest = hashlib.sha256(content).hexdigest()
    if response.status_code != 200:
        raise DataReadinessError(f"official source returned HTTP {response.status_code}: {response.requested_url}")
    if response.body_length != len(content) or response.sha256 != digest:
        raise DataReadinessError(f"HTTP response metadata does not match exact body bytes: {response.requested_url}")
    if response.body_representation != "http_entity_encoded":
        raise DataReadinessError(
            "official response body representation must be http_entity_encoded"
        )
    urls = [response.requested_url, *response.redirect_chain, response.final_url]
    for url in urls:
        _require_spglobal_url(str(url))
    if expected_release and _is_generic_landing_url(response.final_url):
        raise DataReadinessError(f"release redirected to generic S&P Global landing page: {response.final_url}")
    retrieved = response.retrieved_at_utc
    if isinstance(retrieved, datetime) and retrieved.tzinfo is None:
        raise DataReadinessError(
            "official response retrieval timestamp must be timezone-aware"
        )
    retrieved_text = retrieved.astimezone(UTC).isoformat() if isinstance(retrieved, datetime) else str(retrieved)
    return {
        "requested_url": response.requested_url,
        "final_url": response.final_url,
        "redirect_chain": [str(item) for item in response.redirect_chain],
        "status_code": response.status_code,
        "retrieved_at_utc": retrieved_text,
        "content_type": response.content_type,
        "content_encoding": response.content_encoding,
        "etag": response.etag,
        "last_modified": response.last_modified,
        "body_length": len(content),
        "sha256": digest,
        "body_representation": response.body_representation,
    }


def _parse_search_page(
    body: bytes,
    *,
    base_url: str,
    content_type: str,
) -> tuple[list[_Announcement], list[tuple[date, str]]]:
    html = _decode_html(body, content_type or "text/html; charset=utf-8")
    soup = BeautifulSoup(html, "html.parser")
    announcements: dict[str, _Announcement] = {}
    dated_by_url: dict[str, date] = {}
    dated_urls: list[tuple[date, str]] = []
    for anchor in soup.find_all("a", href=True):
        url = urljoin(base_url, str(anchor["href"]))
        match = _PUBLISHED_DATE.search(urlparse(url).path)
        if match is None:
            continue
        published = date.fromisoformat(match.group(1))
        previous = dated_by_url.get(url)
        if previous is not None and previous != published:
            raise DataReadinessError(
                f"official archive URL has conflicting publication dates: {url}"
            )
        if previous is None:
            dated_by_url[url] = published
            dated_urls.append((published, url))
        title = " ".join(anchor.get_text(" ", strip=True).replace("&", "&").split())
        if _is_membership_title(title):
            _require_spglobal_url(url)
            announcements[url] = _Announcement(published, title, url, "discovery")
    return (
        sorted(
            announcements.values(),
            key=lambda item: (item.published_date, item.url),
        ),
        dated_urls,
    )


def _validate_search_page_coverage(
    *,
    page_number: int,
    dated_urls: list[tuple[date, str]],
    previous_last_url: str | None,
    seen_urls: set[str],
    upper_boundary_crossed: bool,
    discovery_end: date,
    retrieved_at_utc: str,
) -> tuple[str, bool]:
    if len(dated_urls) != SEARCH_PAGE_SIZE:
        raise DataReadinessError(
            "official archive search page is truncated or structurally changed: "
            f"page={page_number} dated_urls={len(dated_urls)} expected={SEARCH_PAGE_SIZE}"
        )
    dates = [published for published, _ in dated_urls]
    urls = {url for _, url in dated_urls}
    if any(left < right for left, right in zip(dates, dates[1:], strict=False)):
        raise DataReadinessError(
            "official archive search page is not newest-to-oldest"
        )
    overlap = seen_urls.intersection(urls)
    if page_number == 0:
        if previous_last_url is not None or overlap:
            raise DataReadinessError(
                "official archive first page has invalid pagination state"
            )
    elif (
        previous_last_url is None
        or dated_urls[0][1] != previous_last_url
        or overlap != {previous_last_url}
    ):
        raise DataReadinessError(
            "official archive pagination lacks the required one-result overlap"
        )
    newest = max(dates)
    oldest = min(dates)
    try:
        observed_at = datetime.fromisoformat(retrieved_at_utc)
    except ValueError as exc:
        raise DataReadinessError(
            "official archive search retrieval timestamp is invalid"
        ) from exc
    if observed_at.tzinfo is None:
        raise DataReadinessError(
            "official archive search retrieval timestamp is not timezone-aware"
        )
    observed_through_cutoff = (
        observed_at.astimezone(_SOURCE_TIMEZONE).date() > discovery_end
    )
    if page_number == 0 and newest < discovery_end and not observed_through_cutoff:
        raise DataReadinessError(
            "official archive first page was observed before the frozen upper boundary"
        )
    seen_urls.update(urls)
    return dated_urls[-1][1], (
        upper_boundary_crossed
        or newest >= discovery_end >= oldest
        or (
            page_number == 0
            and newest < discovery_end
            and observed_through_cutoff
        )
    )


def _load_seed_announcements(path: Path) -> list[_Announcement]:
    audit = _load_json(path)
    source_manifest = audit.get("source_manifest")
    if audit.get("schema") != SOURCE_AUDIT_SCHEMA:
        raise DataReadinessError("source audit schema is unsupported")
    if (
        not isinstance(source_manifest, Mapping)
        or source_manifest.get("schema") != SOURCE_MANIFEST_SCHEMA
    ):
        raise DataReadinessError("source audit has no source_manifest object")
    raw_sources = source_manifest.get("sources")
    if not isinstance(raw_sources, list):
        raise DataReadinessError("source audit has no source_manifest.sources list")
    by_url: dict[str, _Announcement] = {}
    for raw in raw_sources:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("source audit contains an invalid source record")
        url = str(raw.get("url", "")).strip()
        title = str(raw.get("title", "")).strip()
        published_text = str(raw.get("published_date", "")).strip()
        _require_spglobal_url(url)
        try:
            published = date.fromisoformat(published_text)
        except ValueError as exc:
            raise DataReadinessError(f"invalid seed publication date for {url}") from exc
        by_url[url] = _Announcement(published, title, url, "seed")
    if len(by_url) != EXPECTED_SEED_URLS:
        raise DataReadinessError(
            f"frozen source audit must contain exactly {EXPECTED_SEED_URLS} distinct URLs, got {len(by_url)}"
        )
    audit_urls = audit.get("source_urls")
    if isinstance(audit_urls, list) and not {
        str(item) for item in audit_urls
    }.issubset(by_url):
        raise DataReadinessError(
            "source audit URL inventory contains URLs outside source_manifest.sources"
        )
    return sorted(by_url.values(), key=lambda item: (item.published_date, item.url))


def _load_unit(
    root: Path,
    *,
    kind: str,
    unit_id: str,
    identity: str,
    request_sha256: str,
) -> dict[str, Any] | None:
    sidecar = _sidecar_path(root, kind, unit_id)
    if not sidecar.exists():
        return None
    record = _load_json(sidecar)
    record_without_hash = {
        key: value for key, value in record.items() if key != "unit_sha256"
    }
    relative = str(record.get("path", ""))
    body_path = _resolve_inside(root, relative)
    if (
        record.get("unit_sha256") != _json_sha256(record_without_hash)
        or record.get("schema") != ARCHIVE_UNIT_SCHEMA
        or record.get("kind") != kind
        or record.get("unit_id") != unit_id
        or record.get("identity") != identity
        or record.get("request_sha256") != request_sha256
        or not body_path.is_file()
        or record.get("sha256") != _file_sha256(body_path)
        or record.get("body_length") != body_path.stat().st_size
    ):
        raise DataReadinessError(f"S&P official archive resume integrity failed: {sidecar}")
    _validate_stored_unit_metadata(record, expected_release=kind == "release")
    return record


def _with_unit_integrity_hash(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(record)
    payload["unit_sha256"] = _json_sha256(payload)
    return payload


def _validate_stored_unit_metadata(
    record: Mapping[str, Any],
    *,
    expected_release: bool,
) -> None:
    required = {
        "requested_url",
        "final_url",
        "redirect_chain",
        "status_code",
        "retrieved_at_utc",
        "content_type",
        "content_encoding",
        "etag",
        "last_modified",
        "body_length",
        "sha256",
        "body_representation",
    }
    if not required.issubset(record) or int(record["status_code"]) != 200:
        raise DataReadinessError("S&P official archive unit metadata is incomplete")
    redirects = record["redirect_chain"]
    if not isinstance(redirects, list):
        raise DataReadinessError("S&P official archive redirect chain is invalid")
    urls = [
        str(record["requested_url"]),
        *(str(value) for value in redirects),
        str(record["final_url"]),
    ]
    for url in urls:
        _require_spglobal_url(url)
    try:
        retrieved = datetime.fromisoformat(str(record["retrieved_at_utc"]))
    except ValueError as exc:
        raise DataReadinessError(
            "S&P official archive retrieval timestamp is invalid"
        ) from exc
    if retrieved.tzinfo is None:
        raise DataReadinessError(
            "S&P official archive retrieval timestamp is not timezone-aware"
        )
    if not re.fullmatch(r"[0-9a-f]{64}", str(record["sha256"])):
        raise DataReadinessError("S&P official archive unit SHA-256 is invalid")
    if expected_release:
        if _is_generic_landing_url(str(record["final_url"])):
            raise DataReadinessError(
                "S&P official release resume points to a generic landing page"
            )
    elif any(
        not _same_search_identity(url, str(record["identity"]))
        for url in urls
    ):
        raise DataReadinessError(
            "S&P official archive search resume changed the query identity"
        )
    if record["body_representation"] != "http_entity_encoded":
        raise DataReadinessError(
            "S&P official archive body representation is invalid"
        )


def _validate_release_record(
    record: Mapping[str, Any],
    announcement: _Announcement,
) -> None:
    if (
        record.get("url") != announcement.url
        or record.get("published_date") != announcement.published_date.isoformat()
        or _canonical_release_path(str(record.get("requested_url", "")))
        != _canonical_release_path(announcement.url)
        or _canonical_release_path(str(record.get("final_url", "")))
        != _canonical_release_path(announcement.url)
    ):
        raise DataReadinessError(f"S&P official release resume metadata failed: {announcement.url}")


def _raw_release_record(record: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "schema",
        "kind",
        "unit_id",
        "identity",
        "request_sha256",
        "url",
        "title",
        "published_date",
        "origin",
        "path",
        "requested_url",
        "final_url",
        "redirect_chain",
        "status_code",
        "retrieved_at_utc",
        "content_type",
        "content_encoding",
        "etag",
        "last_modified",
        "body_length",
        "sha256",
        "body_representation",
    )
    missing = [field for field in fields if field not in record]
    if missing:
        raise DataReadinessError(
            f"S&P raw release metadata is incomplete: {missing}"
        )
    return {field: record[field] for field in fields}


def _raw_release_identity(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: record[key]
        for key in (
            "url",
            "requested_url",
            "final_url",
            "redirect_chain",
            "status_code",
            "retrieved_at_utc",
            "content_type",
            "content_encoding",
            "etag",
            "last_modified",
            "body_length",
            "sha256",
        )
    }


def _validate_discovery(
    discovery: Mapping[str, Any],
    *,
    root: Path,
    seeds: list[_Announcement],
    request_sha256: str,
    discovery_end: date,
) -> None:
    if (
        discovery.get("schema") != DISCOVERY_SCHEMA
        or discovery.get("request_sha256") != request_sha256
        or discovery.get("lower_boundary") != DISCOVERY_START.isoformat()
        or discovery.get("upper_boundary") != discovery_end.isoformat()
        or discovery.get("lower_boundary_reached") is not True
        or int(discovery.get("seed_url_count", -1)) != EXPECTED_SEED_URLS
    ):
        raise DataReadinessError("S&P official archive discovery authority is invalid")
    announcements = _records(discovery, "announcements")
    urls = [str(item.get("url", "")) for item in announcements]
    if len(urls) != len(set(urls)) or len(urls) < EXPECTED_SEED_URLS:
        raise DataReadinessError("S&P official archive discovery URL set is invalid")
    page_records = _records(discovery, "search_pages")
    if not page_records:
        raise DataReadinessError(
            "S&P official archive discovery has no search pages"
        )
    discovered: dict[str, _Announcement] = {}
    boundary_reached = False
    previous_last_url: str | None = None
    upper_boundary_crossed = False
    seen_search_urls: set[str] = set()
    for page_number, page_record in enumerate(page_records):
        offset = page_number * SEARCH_PAGE_STRIDE
        expected_identity = _search_identity({**ARCHIVE_QUERY, "o": str(offset)})
        expected_unit_id = (
            f"search-{page_number:04d}-"
            f"{hashlib.sha256(expected_identity.encode()).hexdigest()[:12]}"
        )
        if (
            page_record.get("page_number") != page_number
            or page_record.get("offset") != offset
            or page_record.get("identity") != expected_identity
            or page_record.get("unit_id") != expected_unit_id
        ):
            raise DataReadinessError(
                "S&P official archive discovery pages are not contiguous from page zero"
            )
        loaded = _load_unit(
            root,
            kind="search_page",
            unit_id=expected_unit_id,
            identity=expected_identity,
            request_sha256=request_sha256,
        )
        if loaded != page_record:
            raise DataReadinessError(
                "S&P official archive discovery differs from search-page sidecar"
            )
        page_announcements, dated_urls = _parse_search_page(
            _decode_http_entity(
                _read_unit_body(root, loaded),
                str(loaded.get("content_encoding") or ""),
            ),
            base_url=str(loaded["final_url"]),
            content_type=str(loaded.get("content_type") or ""),
        )
        previous_last_url, upper_boundary_crossed = _validate_search_page_coverage(
            page_number=page_number,
            dated_urls=dated_urls,
            previous_last_url=previous_last_url,
            seen_urls=seen_search_urls,
            upper_boundary_crossed=upper_boundary_crossed,
            discovery_end=discovery_end,
            retrieved_at_utc=str(loaded["retrieved_at_utc"]),
        )
        for announcement in page_announcements:
            if DISCOVERY_START <= announcement.published_date <= discovery_end:
                discovered[announcement.url] = announcement
        if max(published for published, _ in dated_urls) < DISCOVERY_START:
            if page_number != len(page_records) - 1:
                raise DataReadinessError(
                    "S&P official archive discovery continued after its lower boundary"
                )
            boundary_reached = True
    expected = [
        item.to_record() for item in _union_announcements(seeds, discovered)
    ]
    if (
        not boundary_reached
        or not upper_boundary_crossed
        or announcements != expected
    ):
        raise DataReadinessError(
            "S&P official archive discovery does not replay from retained search pages"
        )
    if (
        int(discovery.get("discovered_url_count", -1)) != len(discovered)
        or int(discovery.get("release_url_count", -1)) != len(expected)
    ):
        raise DataReadinessError("S&P official archive discovery counts are invalid")


def _publish_status(
    root: Path,
    *,
    request_sha256: str,
    status: str,
    stop_reason: str,
    discovery_complete: bool,
    requested_releases: int,
    completed_releases: int,
    resumed_releases: int,
    failed_releases: Mapping[str, str],
    network_units: int,
) -> dict[str, Any]:
    payload = _status_payload(
        request_sha256=request_sha256,
        status=status,
        stop_reason=stop_reason,
        discovery_complete=discovery_complete,
        requested_releases=requested_releases,
        completed_releases=completed_releases,
        resumed_releases=resumed_releases,
        failed_releases=failed_releases,
        network_units=network_units,
    )
    _atomic_json(root / "_status.json", payload)
    return payload


def _status_payload(
    *,
    request_sha256: str,
    status: str,
    stop_reason: str,
    discovery_complete: bool,
    requested_releases: int,
    completed_releases: int,
    resumed_releases: int,
    failed_releases: Mapping[str, str],
    network_units: int,
) -> dict[str, Any]:
    return {
        "schema": ARCHIVE_STATUS_SCHEMA,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "request_sha256": request_sha256,
        "status": status,
        "stop_reason": stop_reason,
        "discovery_complete": discovery_complete,
        "requested_releases": requested_releases,
        "completed_releases": completed_releases,
        "resumed_releases": resumed_releases,
        "failed_releases": dict(sorted(failed_releases.items())),
        "network_units_this_run": network_units,
        "resources": memory_audit(
            hard_budget_gib=MAXIMUM_MEMORY_GIB,
            headroom_gib=MEMORY_HEADROOM_GIB,
        ).to_record(),
    }


def _write_content_addressed(root: Path, body: bytes) -> Path:
    with _OBJECT_WRITE_LOCK:
        digest = hashlib.sha256(body).hexdigest()
        path = root / "objects" / digest[:2] / f"{digest}.html"
        if path.exists():
            if _file_sha256(path) != digest or path.read_bytes() != body:
                raise DataReadinessError(
                    f"content-addressed object is corrupt: {path}"
                )
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(body)
            if _file_sha256(temporary) != digest:
                raise DataReadinessError(
                    f"stored response bytes failed post-write verification: {path}"
                )
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return path


def _write_or_validate_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        if _load_json(path) != dict(payload):
            raise DataReadinessError(f"S&P official archive resume request differs: {path}")
        return
    _atomic_json(path, payload)


def _write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise DataReadinessError(f"refusing to overwrite immutable S&P archive unit: {path}")
    _atomic_json(path, payload)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=str), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"invalid S&P official archive JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"S&P official archive JSON must be an object: {path}")
    return {str(key): item for key, item in value.items()}


def _records(value: Mapping[str, Any], field: str) -> list[dict[str, Any]]:
    raw = value.get(field)
    if not isinstance(raw, list) or any(not isinstance(item, Mapping) for item in raw):
        raise DataReadinessError(f"S&P official archive {field} must be a list of objects")
    return [{str(key): item for key, item in cast(Mapping[str, Any], record).items()} for record in raw]


def _announcement_from_record(record: Mapping[str, Any]) -> _Announcement:
    try:
        published = date.fromisoformat(str(record["published_date"]))
        url = str(record["url"])
        _require_spglobal_url(url)
        return _Announcement(published, str(record.get("title", "")), url, str(record.get("origin", "")))
    except (KeyError, ValueError) as exc:
        raise DataReadinessError("invalid S&P official archive announcement record") from exc


def _union_announcements(
    seeds: list[_Announcement],
    discovered: Mapping[str, _Announcement],
) -> list[_Announcement]:
    union = {item.url: item for item in seeds}
    for url, announcement in discovered.items():
        seed_item = union.get(url)
        if seed_item is None:
            union[url] = announcement
        elif seed_item.published_date != announcement.published_date:
            raise DataReadinessError(
                f"conflicting publication dates for official release: {url}"
            )
        elif not seed_item.title and announcement.title:
            union[url] = _Announcement(
                published_date=seed_item.published_date,
                title=announcement.title,
                url=url,
                origin="seed_and_discovery",
            )
    return sorted(
        union.values(), key=lambda item: (item.published_date, item.url)
    )


def _read_unit_body(root: Path, record: Mapping[str, Any]) -> bytes:
    path = _resolve_inside(root, str(record["path"]))
    body = path.read_bytes()
    if hashlib.sha256(body).hexdigest() != record.get("sha256"):
        raise DataReadinessError(f"S&P official archive object failed integrity verification: {path}")
    return body


def _sidecar_path(root: Path, kind: str, unit_id: str) -> Path:
    directory = "search_pages" if kind == "search_page" else "releases"
    return root / "units" / directory / f"{unit_id}.json"


def _release_unit_id(url: str) -> str:
    return f"release-{hashlib.sha256(url.encode()).hexdigest()}"


def _search_identity(params: Mapping[str, str]) -> str:
    return f"{SP_GLOBAL_ARCHIVE_URL}?{urlencode(sorted(params.items()))}"


def _same_search_identity(actual_url: str, expected_url: str) -> bool:
    actual = urlparse(actual_url)
    expected = urlparse(expected_url)
    return (
        actual.path.rstrip("/") == expected.path.rstrip("/")
        and dict(parse_qsl(actual.query, keep_blank_values=True))
        == dict(parse_qsl(expected.query, keep_blank_values=True))
    )


def _is_membership_title(title: str) -> bool:
    normalized = title.replace("&amp;", "&").replace("S & P", "S&P").replace("S &P", "S&P").replace("S& P", "S&P")
    return bool(_MEMBERSHIP_TITLE.search(normalized))


def _require_spglobal_url(url: str) -> None:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (host == "spglobal.com" or host.endswith(".spglobal.com")):
        raise DataReadinessError(f"official archive URL is not an HTTPS S&P Global domain: {url}")


def _is_generic_landing_url(url: str) -> bool:
    parsed = urlparse(url)
    return parsed.path.rstrip("/").lower() in {
        "",
        "/index.php",
        "/press/press-releases",
    }


def _canonical_release_path(url: str) -> str:
    return urlparse(url).path.rstrip("/")


def _decode_html(body: bytes, content_type: str | None) -> str:
    charset = "utf-8"
    match = re.search(r"charset\s*=\s*[\"']?([^;\s\"']+)", content_type or "", re.IGNORECASE)
    if match is not None:
        charset = match.group(1)
    try:
        return body.decode(charset)
    except (LookupError, UnicodeDecodeError) as exc:
        raise DataReadinessError(f"official HTML cannot be decoded as {charset}") from exc


def _decode_http_entity(
    body: bytes,
    content_encoding: str,
    *,
    maximum_decoded_bytes: int = MAXIMUM_DECODED_BYTES,
) -> bytes:
    if maximum_decoded_bytes < 1:
        raise ValueError("maximum_decoded_bytes must be positive")
    decoded = body
    encodings = [
        item.strip().lower()
        for item in content_encoding.split(",")
        if item.strip() and item.strip().lower() != "identity"
    ]
    if not encodings and len(body) > maximum_decoded_bytes:
        raise DataReadinessError(
            "official response decoded body exceeds the configured limit"
        )
    try:
        for encoding in reversed(encodings):
            if encoding in {"gzip", "x-gzip"}:
                decoded = _decompress_bounded(
                    decoded,
                    wbits=16 + zlib.MAX_WBITS,
                    maximum_decoded_bytes=maximum_decoded_bytes,
                )
            elif encoding == "deflate":
                try:
                    decoded = _decompress_bounded(
                        decoded,
                        wbits=zlib.MAX_WBITS,
                        maximum_decoded_bytes=maximum_decoded_bytes,
                    )
                except zlib.error:
                    decoded = _decompress_bounded(
                        decoded,
                        wbits=-zlib.MAX_WBITS,
                        maximum_decoded_bytes=maximum_decoded_bytes,
                    )
            else:
                raise DataReadinessError(
                    f"unsupported official response Content-Encoding: {encoding}"
                )
    except zlib.error as exc:
        raise DataReadinessError(
            "official response Content-Encoding cannot be decoded"
        ) from exc
    return decoded


def _decompress_bounded(
    body: bytes,
    *,
    wbits: int,
    maximum_decoded_bytes: int,
) -> bytes:
    decompressor = zlib.decompressobj(wbits)
    output: list[bytes] = []
    total = 0
    cursor = 0
    while cursor < len(body):
        chunk = body[cursor : cursor + 64 * 1024]
        cursor += len(chunk)
        while chunk:
            decoded = decompressor.decompress(
                chunk,
                maximum_decoded_bytes - total + 1,
            )
            total += len(decoded)
            if total > maximum_decoded_bytes:
                raise DataReadinessError(
                    "official response decoded body exceeds the configured limit"
                )
            output.append(decoded)
            chunk = decompressor.unconsumed_tail
    tail = decompressor.flush(maximum_decoded_bytes - total + 1)
    total += len(tail)
    if total > maximum_decoded_bytes:
        raise DataReadinessError(
            "official response decoded body exceeds the configured limit"
        )
    output.append(tail)
    if not decompressor.eof:
        raise DataReadinessError(
            "official response Content-Encoding is truncated"
        )
    return b"".join(output)


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise DataReadinessError(f"S&P official archive path escapes collection root: {relative}")
    return candidate


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _default_client_factory() -> BytesHttpClient:
    return cast(BytesHttpClient, HttpClient(user_agent="market-predictor/0.1 spglobal-archive"))
