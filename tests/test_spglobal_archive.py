from __future__ import annotations

import hashlib
import json
import threading
import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pytest
from typer.testing import CliRunner

import market_predictor.commands.v3_readiness as command_module
import market_predictor.v3.spglobal_archive as archive_module
from market_predictor.collection_cli import app as collection_app
from market_predictor.locking import file_lock
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.spglobal_archive import (
    ARCHIVE_AUTHORITY_SCHEMA,
    ArchiveCollectionConfig,
    collect_spglobal_archive,
)


@dataclass
class _Response:
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


class _Transport:
    def __init__(
        self,
        bodies: dict[str, bytes],
        *,
        final_urls: dict[str, str] | None = None,
        redirect_chains: dict[str, Sequence[str]] | None = None,
    ) -> None:
        self.bodies = bodies
        self.final_urls = final_urls or {}
        self.redirect_chains = redirect_chains or {}
        self.calls: list[str] = []
        self.lock = threading.Lock()

    def factory(self) -> _Client:
        return _Client(self)


class _Client:
    def __init__(self, transport: _Transport) -> None:
        self.transport = transport

    def get_bytes_with_metadata(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        retries: int = 3,
        pause: float = 1.0,
        maximum_body_bytes: int = 16 * 1024 * 1024,
    ) -> _Response:
        del retries, pause
        key = f"search:{params['o']}" if params is not None else url
        with self.transport.lock:
            self.transport.calls.append(key)
        body = self.transport.bodies[key]
        if len(body) > maximum_body_bytes:
            raise RuntimeError("fixture body exceeds maximum_body_bytes")
        requested_url = (
            f"{url}?{urlencode(sorted((str(k), str(v)) for k, v in params.items()))}"
            if params is not None
            else url
        )
        final_url = self.transport.final_urls.get(key, requested_url)
        return _Response(
            body=body,
            requested_url=requested_url,
            final_url=final_url,
            redirect_chain=self.transport.redirect_chains.get(key, ()),
            status_code=200,
            retrieved_at_utc=datetime(2026, 7, 31, 12, tzinfo=UTC),
            content_type="text/html; charset=utf-8",
            content_encoding=None,
            etag='"fixture"',
            last_modified="Fri, 31 Jul 2026 12:00:00 GMT",
            body_length=len(body),
            sha256=hashlib.sha256(body).hexdigest(),
            body_representation="http_entity_encoded",
        )


def test_discovery_refuses_authority_when_page_limit_does_not_reach_boundary(tmp_path: Path) -> None:
    audit_path, audit_hash, _ = _write_source_audit(tmp_path)
    transport = _Transport(
        {
            "search:0": _search_page(
                [
                    (
                        "2026-07-09",
                        "Recent release",
                        "https://press.spglobal.com/2026-07-09-recent",
                    )
                ],
                overlap_out=(
                    "2026-07-09",
                    "https://press.spglobal.com/2026-07-09-overlap-0-99",
                ),
            ),
            "search:99": _search_page(
                [
                    (
                        "2026-07-09",
                        "Pagination overlap",
                        "https://press.spglobal.com/2026-07-09-overlap-0-99",
                    ),
                    (
                        "2025-01-01",
                        "Older release",
                        "https://press.spglobal.com/2025-01-01-older",
                    )
                ]
            ),
        }
    )

    status = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=tmp_path / "out",
        client_factory=transport.factory,
        config=ArchiveCollectionConfig(maximum_pages=2),
    )

    assert status["stop_reason"] == "discovery_boundary_not_reached"
    assert status["discovery_complete"] is False
    assert not (tmp_path / "out" / "_authority.json").exists()
    assert transport.calls == ["search:0", "search:99"]


def test_period_used_urls_may_be_subset_of_complete_source_manifest(
    tmp_path: Path,
) -> None:
    audit_path, _, _ = _write_source_audit(tmp_path)
    audit = _read_json(audit_path)
    audit["source_urls"] = audit["source_urls"][:-5]
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    assert len(archive_module._load_seed_announcements(audit_path)) == 83


def test_period_used_urls_cannot_reference_unknown_source(tmp_path: Path) -> None:
    audit_path, _, _ = _write_source_audit(tmp_path)
    audit = _read_json(audit_path)
    audit["source_urls"].append(
        "https://press.spglobal.com/2026-07-01-unknown-source"
    )
    audit_path.write_text(json.dumps(audit), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="outside source_manifest"):
        archive_module._load_seed_announcements(audit_path)


def test_discovery_unions_seed_urls_and_broad_membership_title(tmp_path: Path) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    extra = "https://press.spglobal.com/2018-05-01-Netflix-and-Twitter-to-Join-S-P-500"
    transport = _Transport(
        {
            "search:0": _search_page(
                [
                    (
                        "2026-07-09",
                        "Recent release",
                        "https://press.spglobal.com/2026-07-09-recent",
                    ),
                    ("2019-01-02", "Seed Company Set to Join S&P 500", seeds[0]),
                    ("2018-05-01", "Netflix and Twitter to Join S&P 500", extra),
                    ("2018-04-14", "Unrelated archive boundary", "https://press.spglobal.com/2018-04-14-boundary"),
                ],
                page_tag="boundary-zero",
                overlap_out=(
                    "2018-04-13",
                    "https://press.spglobal.com/2018-04-13-overlap-0-99",
                ),
            ),
            "search:99": _search_page(
                [
                    (
                        "2018-04-13",
                        "Pagination overlap",
                        "https://press.spglobal.com/2018-04-13-overlap-0-99",
                    ),
                    (
                        "2018-04-13",
                        "Archive boundary",
                        "https://press.spglobal.com/2018-04-13-boundary",
                    )
                ],
                page_tag="boundary-one",
            ),
        }
    )

    status = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=tmp_path / "out",
        client_factory=transport.factory,
        config=ArchiveCollectionConfig(maximum_units_this_run=2),
    )

    discovery = _read_json(tmp_path / "out" / "_discovery.json")
    urls = {str(record["url"]) for record in discovery["announcements"]}
    assert status["stop_reason"] == "operational_batch_limit"
    assert discovery["seed_url_count"] == 83
    assert discovery["release_url_count"] == 84
    assert urls == {*seeds, extra}


def test_unrelated_old_link_cannot_end_discovery_early(tmp_path: Path) -> None:
    audit_path, audit_hash, _ = _write_source_audit(tmp_path)
    transport = _Transport(
        {
            "search:0": _search_page(
                [
                    (
                        "2026-07-09",
                        "Recent release",
                        "https://press.spglobal.com/2026-07-09-recent",
                    ),
                    (
                        "2018-04-14",
                        "Unrelated archive boundary",
                        "https://press.spglobal.com/2018-04-14-boundary",
                    ),
                ]
            )
        }
    )

    status = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=tmp_path / "out",
        client_factory=transport.factory,
        config=ArchiveCollectionConfig(maximum_pages=1),
    )

    assert status["stop_reason"] == "discovery_boundary_not_reached"
    assert not (tmp_path / "out" / "_discovery.json").exists()


def test_truncated_first_search_page_cannot_publish_authority(tmp_path: Path) -> None:
    audit_path, audit_hash, _ = _write_source_audit(tmp_path)
    transport = _Transport(
        {
            "search:0": _search_page(
                [
                    (
                        "2026-07-09",
                        "Recent release",
                        "https://press.spglobal.com/2026-07-09-recent",
                    )
                ],
                total_dated_urls=1,
            )
        }
    )

    with pytest.raises(DataReadinessError, match="truncated or structurally changed"):
        collect_spglobal_archive(
            source_audit_path=audit_path,
            expected_source_audit_sha256=audit_hash,
            output_directory=tmp_path / "out",
            client_factory=transport.factory,
        )

    assert not (tmp_path / "out" / "_authority.json").exists()


def test_boundary_date_split_across_pages_is_fully_collected(tmp_path: Path) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    boundary_release = (
        "https://press.spglobal.com/2018-04-14-Boundary-Company-Set-to-Join-S-P-500"
    )
    transport = _Transport(
        {
            "search:0": _search_page(
                [
                    (
                        "2026-07-09",
                        "Recent release",
                        "https://press.spglobal.com/2026-07-09-recent",
                    ),
                    (
                        "2018-04-14",
                        "Boundary archive item",
                        "https://press.spglobal.com/2018-04-14-page-zero",
                    ),
                ],
                page_tag="split-zero",
                overlap_out=(
                    "2018-04-14",
                    "https://press.spglobal.com/2018-04-14-overlap-0-99",
                ),
            ),
            "search:99": _search_page(
                [
                    (
                        "2018-04-14",
                        "Pagination overlap",
                        "https://press.spglobal.com/2018-04-14-overlap-0-99",
                    ),
                    (
                        "2018-04-14",
                        "Boundary Company Set to Join S&P 500",
                        boundary_release,
                    )
                ],
                page_tag="split-one",
                overlap_out=(
                    "2018-04-13",
                    "https://press.spglobal.com/2018-04-13-overlap-99-198",
                ),
            ),
            "search:198": _search_page(
                [
                    (
                        "2018-04-13",
                        "Pagination overlap",
                        "https://press.spglobal.com/2018-04-13-overlap-99-198",
                    ),
                    (
                        "2018-04-13",
                        "Older archive item",
                        "https://press.spglobal.com/2018-04-13-older",
                    )
                ]
            ),
        }
    )

    status = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=tmp_path / "out",
        client_factory=transport.factory,
        config=ArchiveCollectionConfig(maximum_units_this_run=3),
    )

    discovery = _read_json(tmp_path / "out" / "_discovery.json")
    assert status["discovery_complete"] is True
    assert boundary_release in {
        str(record["url"]) for record in discovery["announcements"]
    }
    assert transport.calls == ["search:0", "search:99", "search:198"]


def test_discovery_includes_generic_sp500_changes_title(tmp_path: Path) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    extra = "https://press.spglobal.com/2018-05-02-S-P-500-Changes"
    transport = _Transport(
        {
            "search:0": _search_page(
                [
                    (
                        "2026-07-09",
                        "Recent release",
                        "https://press.spglobal.com/2026-07-09-recent",
                    ),
                    ("2018-05-02", "S&P 500 Changes", extra),
                ],
                overlap_out=(
                    "2018-04-13",
                    "https://press.spglobal.com/2018-04-13-generic-overlap",
                ),
                page_tag="generic-zero",
            ),
            "search:99": _search_page(
                [
                    (
                        "2018-04-13",
                        "Pagination overlap",
                        "https://press.spglobal.com/2018-04-13-generic-overlap",
                    ),
                    (
                        "2018-04-13",
                        "Archive boundary",
                        "https://press.spglobal.com/2018-04-13-boundary",
                    ),
                ],
                page_tag="generic-one",
            )
        }
    )

    collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=tmp_path / "out",
        client_factory=transport.factory,
        config=ArchiveCollectionConfig(maximum_units_this_run=2),
    )

    discovery = _read_json(tmp_path / "out" / "_discovery.json")
    assert extra in {str(record["url"]) for record in discovery["announcements"]}


def test_final_archive_preserves_exact_bytes_and_parser_source_hash(tmp_path: Path) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    release_body = _release_body().replace(b"><", b">\r\n<")
    transport = _complete_transport(seeds, release_body=release_body)

    manifest = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=tmp_path / "out",
        client_factory=transport.factory,
        config=ArchiveCollectionConfig(workers=2),
    )

    digest = hashlib.sha256(release_body).hexdigest()
    object_path = tmp_path / "out" / "objects" / digest[:2] / f"{digest}.html"
    assert object_path.read_bytes() == release_body
    assert manifest["status"] == "complete"
    assert manifest["release_url_count"] == 84
    assert all(record["sha256"] == digest for record in manifest["releases"])
    assert all(record["parser_source_sha256"] == record["sha256"] for record in manifest["releases"])
    authority = _read_json(tmp_path / "out" / "_authority.json")
    assert authority["schema"] == ARCHIVE_AUTHORITY_SCHEMA
    assert authority["state"] == "raw_complete"
    assert authority["event_extraction_ready"] is True
    archive_module.require_spglobal_event_reconstruction_ready(tmp_path / "out")


def test_partial_run_resumes_search_page_without_network(tmp_path: Path) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    transport = _complete_transport(seeds)
    output = tmp_path / "out"

    first = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=output,
        client_factory=transport.factory,
        config=ArchiveCollectionConfig(maximum_units_this_run=2),
    )
    second = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=output,
        client_factory=transport.factory,
    )

    assert first["status"] == "incomplete"
    assert second["status"] == "complete"
    assert transport.calls.count("search:0") == 1


def test_incomplete_discovery_resumes_with_larger_page_limit(tmp_path: Path) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    transport = _complete_transport(seeds)
    output = tmp_path / "out"

    first = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=output,
        client_factory=transport.factory,
        config=ArchiveCollectionConfig(maximum_pages=1),
    )
    second = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=output,
        client_factory=transport.factory,
        config=ArchiveCollectionConfig(
            maximum_pages=2,
            maximum_units_this_run=1,
        ),
    )

    assert first["stop_reason"] == "discovery_boundary_not_reached"
    assert second["discovery_complete"] is True
    assert transport.calls.count("search:0") == 1
    assert transport.calls.count("search:99") == 1


def test_resume_rejects_mutated_content_addressed_object(tmp_path: Path) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    transport = _complete_transport(seeds)
    output = tmp_path / "out"
    collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=output,
        client_factory=transport.factory,
        config=ArchiveCollectionConfig(maximum_units_this_run=2),
    )
    discovery = _read_json(output / "_discovery.json")
    page = discovery["search_pages"][0]
    object_path = output / str(page["path"])
    object_path.write_bytes(b"mutated")
    calls_before_resume = len(transport.calls)

    with pytest.raises(DataReadinessError, match="resume integrity failed"):
        collect_spglobal_archive(
            source_audit_path=audit_path,
            expected_source_audit_sha256=audit_hash,
            output_directory=output,
            client_factory=transport.factory,
        )

    assert len(transport.calls) == calls_before_resume
    assert not (output / "_authority.json").exists()


def test_resume_rejects_modified_discovery_announcements(tmp_path: Path) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    transport = _complete_transport(seeds)
    output = tmp_path / "out"
    collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=output,
        client_factory=transport.factory,
        config=ArchiveCollectionConfig(maximum_units_this_run=2),
    )
    discovery_path = output / "_discovery.json"
    discovery = _read_json(discovery_path)
    discovery["announcements"] = discovery["announcements"][:-1]
    discovery_path.write_text(json.dumps(discovery), encoding="utf-8")
    calls_before_resume = len(transport.calls)

    with pytest.raises(DataReadinessError, match="does not replay"):
        collect_spglobal_archive(
            source_audit_path=audit_path,
            expected_source_audit_sha256=audit_hash,
            output_directory=output,
            client_factory=transport.factory,
        )

    assert len(transport.calls) == calls_before_resume


def test_resume_rejects_discovery_with_missing_first_page(tmp_path: Path) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    transport = _complete_transport(seeds)
    output = tmp_path / "out"
    collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=output,
        client_factory=transport.factory,
        config=ArchiveCollectionConfig(maximum_units_this_run=2),
    )
    discovery_path = output / "_discovery.json"
    discovery = _read_json(discovery_path)
    discovery["search_pages"] = discovery["search_pages"][1:]
    discovery_path.write_text(json.dumps(discovery), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="not contiguous"):
        collect_spglobal_archive(
            source_audit_path=audit_path,
            expected_source_audit_sha256=audit_hash,
            output_directory=output,
            client_factory=transport.factory,
        )


def test_resume_rejects_modified_sidecar_metadata(tmp_path: Path) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    transport = _complete_transport(seeds)
    output = tmp_path / "out"
    collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=output,
        client_factory=transport.factory,
        config=ArchiveCollectionConfig(maximum_units_this_run=2),
    )
    discovery = _read_json(output / "_discovery.json")
    page = discovery["search_pages"][0]
    sidecar = output / "units" / "search_pages" / f"{page['unit_id']}.json"
    record = _read_json(sidecar)
    record["final_url"] = "https://press.spglobal.com/index.php?wrong=1"
    sidecar.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="resume integrity failed"):
        collect_spglobal_archive(
            source_audit_path=audit_path,
            expected_source_audit_sha256=audit_hash,
            output_directory=output,
            client_factory=transport.factory,
        )


def test_search_redirect_must_preserve_query_identity(tmp_path: Path) -> None:
    audit_path, audit_hash, _ = _write_source_audit(tmp_path)
    transport = _Transport(
        {"search:0": _search_page([])},
        final_urls={"search:0": "https://press.spglobal.com/index.php?wrong=1"},
    )

    with pytest.raises(DataReadinessError, match="changed the query identity"):
        collect_spglobal_archive(
            source_audit_path=audit_path,
            expected_source_audit_sha256=audit_hash,
            output_directory=tmp_path / "out",
            client_factory=transport.factory,
        )

    assert not (tmp_path / "out" / "_authority.json").exists()


def test_each_search_redirect_hop_must_preserve_query_identity(tmp_path: Path) -> None:
    audit_path, audit_hash, _ = _write_source_audit(tmp_path)
    transport = _Transport(
        {"search:0": _search_page([])},
        redirect_chains={
            "search:0": ("https://press.spglobal.com/index.php?wrong=1",)
        },
    )

    with pytest.raises(DataReadinessError, match="changed the query identity"):
        collect_spglobal_archive(
            source_audit_path=audit_path,
            expected_source_audit_sha256=audit_hash,
            output_directory=tmp_path / "out",
            client_factory=transport.factory,
        )


def test_finalization_replays_newly_written_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    transport = _complete_transport(seeds)
    output = tmp_path / "out"
    original = archive_module._collect_releases

    def collect_then_corrupt(**kwargs: Any) -> tuple[dict[str, dict[str, Any]], int, dict[str, str], int]:
        result = original(**kwargs)
        first = next(iter(result[0].values()))
        (output / str(first["path"])).write_bytes(b"corrupt-after-write")
        return result

    monkeypatch.setattr(archive_module, "_collect_releases", collect_then_corrupt)

    with pytest.raises(DataReadinessError, match="resume integrity failed"):
        collect_spglobal_archive(
            source_audit_path=audit_path,
            expected_source_audit_sha256=audit_hash,
            output_directory=output,
            client_factory=transport.factory,
        )

    assert not (output / "_authority.json").exists()
    status_path = output / "_status.json"
    assert not status_path.exists() or _read_json(status_path)["status"] != "complete"


def test_collector_refuses_concurrent_writer_for_same_output(tmp_path: Path) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    output = tmp_path / "out"
    output.mkdir()

    with file_lock(output / "_collector", timeout=0.0):
        with pytest.raises(DataReadinessError, match="another collector"):
            collect_spglobal_archive(
                source_audit_path=audit_path,
                expected_source_audit_sha256=audit_hash,
                output_directory=output,
                client_factory=_complete_transport(seeds).factory,
            )


def test_decoded_response_limit_blocks_compression_expansion() -> None:
    encoded = zlib.compress(b"12345")

    with pytest.raises(DataReadinessError, match="decoded body exceeds"):
        archive_module._decode_http_entity(
            encoded,
            "deflate",
            maximum_decoded_bytes=4,
        )


def test_collection_cli_returns_nonzero_for_every_partial_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(tmp_path / "runtime"))
    monkeypatch.setattr(
        command_module,
        "collect_spglobal_archive",
        lambda **_: {
            "status": "incomplete",
            "stop_reason": "operational_batch_limit",
            "discovery_complete": True,
            "requested_releases": 84,
            "completed_releases": 1,
            "resumed_releases": 0,
            "network_units_this_run": 1,
        },
    )

    result = CliRunner().invoke(
        collection_app,
        [
            "collect-sp500-official-source-archive",
            "--source-audit",
            str(tmp_path / "audit.json"),
            "--source-audit-sha256",
            "0" * 64,
            "--out-dir",
            str(tmp_path / "out"),
        ],
    )

    assert result.exit_code == 2


def test_generic_landing_page_is_failed_and_blocks_authority(tmp_path: Path) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    transport = _complete_transport(seeds)
    transport.final_urls[seeds[0]] = "https://press.spglobal.com/index.php"

    status = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=tmp_path / "out",
        client_factory=transport.factory,
        config=ArchiveCollectionConfig(workers=2),
    )

    assert status["status"] == "incomplete"
    assert status["stop_reason"] == "release_failures"
    assert seeds[0] in status["failed_releases"]
    assert "generic" in status["failed_releases"][seeds[0]]
    assert not (tmp_path / "out" / "_manifest.json").exists()
    assert not (tmp_path / "out" / "_authority.json").exists()


def test_redirect_to_different_release_identity_is_rejected(tmp_path: Path) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    transport = _complete_transport(seeds)
    transport.final_urls[seeds[0]] = seeds[1]

    status = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=tmp_path / "out",
        client_factory=transport.factory,
    )

    assert "changed the release identity" in status["failed_releases"][seeds[0]]
    assert not (tmp_path / "out" / "_authority.json").exists()


def test_authoritative_archive_cannot_be_overwritten(tmp_path: Path) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    transport = _complete_transport(seeds)
    output = tmp_path / "out"
    collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=output,
        client_factory=transport.factory,
    )
    calls = len(transport.calls)

    with pytest.raises(DataReadinessError, match="immutable"):
        collect_spglobal_archive(
            source_audit_path=audit_path,
            expected_source_audit_sha256=audit_hash,
            output_directory=output,
            client_factory=transport.factory,
        )

    assert len(transport.calls) == calls


def test_provider_template_marker_is_accepted_with_verified_http_lineage(
    tmp_path: Path,
) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    transport = _complete_transport(seeds)
    transport.bodies[seeds[0]] = (
        b"<!-- saved from url=(0051)https://www.spglobal.com/press/press-releases -->"
        + _release_body()
    )

    manifest = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=tmp_path / "out",
        client_factory=transport.factory,
    )

    assert manifest["status"] == "complete"
    assert (tmp_path / "out" / "_authority.json").exists()


def test_raw_archive_retains_release_with_unresolved_parser(
    tmp_path: Path,
) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    transport = _complete_transport(seeds)
    transport.bodies[seeds[0]] = (
        b"<html><body><h1>Official S&amp;P 500 announcement</h1></body></html>"
    )

    manifest = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=tmp_path / "out",
        client_factory=transport.factory,
    )

    record = next(item for item in manifest["releases"] if item["url"] == seeds[0])
    assert manifest["status"] == "complete"
    assert manifest["parser_unresolved_releases"] == 1
    assert record["parser_status"] == "parser_unresolved"
    assert record["change_rows"] == 0
    assert record["parser_source_sha256"] is None
    assert record["parser_error"]
    authority = _read_json(tmp_path / "out" / "_authority.json")
    assert authority["state"] == "raw_complete"
    assert authority["event_extraction_ready"] is False
    with pytest.raises(DataReadinessError, match="blocked by 1 unresolved"):
        archive_module.require_spglobal_event_reconstruction_ready(
            tmp_path / "out"
        )


def test_no_effective_rows_are_not_reported_as_parser_unresolved(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    monkeypatch.setattr(archive_module, "_parse_release_changes", lambda *_, **__: [])

    manifest = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=tmp_path / "out",
        client_factory=_complete_transport(seeds).factory,
    )

    assert manifest["parser_unresolved_releases"] == 0
    assert manifest["event_extraction_ready"] is True
    assert all(
        record["parser_status"] == "no_effective_rows"
        for record in manifest["releases"]
    )
    archive_module.require_spglobal_event_reconstruction_ready(tmp_path / "out")


def test_unexpected_parser_exception_is_retained_as_unresolved_raw_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    transport = _complete_transport(seeds)

    def fail_parser(*_: Any, **__: Any) -> list[Any]:
        raise ValueError("unexpected historical date format")

    monkeypatch.setattr(archive_module, "_parse_release_changes", fail_parser)
    manifest = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=tmp_path / "out",
        client_factory=transport.factory,
    )

    assert manifest["status"] == "complete"
    assert manifest["parser_unresolved_releases"] == 84
    assert all(
        record["parser_status"] == "parser_unresolved"
        for record in manifest["releases"]
    )


def test_decode_failure_is_retained_as_unresolved_raw_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    transport = _complete_transport(seeds)
    original = archive_module._decode_http_entity

    def fail_release_decode(
        body: bytes,
        content_encoding: str,
        *,
        maximum_decoded_bytes: int = archive_module.MAXIMUM_DECODED_BYTES,
    ) -> bytes:
        if b"Effective Date" in body:
            raise UnicodeError("historical release encoding failure")
        return original(
            body,
            content_encoding,
            maximum_decoded_bytes=maximum_decoded_bytes,
        )

    monkeypatch.setattr(archive_module, "_decode_http_entity", fail_release_decode)
    manifest = collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=tmp_path / "out",
        client_factory=transport.factory,
    )

    assert manifest["status"] == "complete"
    assert manifest["parser_unresolved_releases"] == 84
    assert all(
        (tmp_path / "out" / str(record["path"])).is_file()
        for record in manifest["releases"]
    )


def test_event_readiness_recomputes_manifest_counts_and_release_set(
    tmp_path: Path,
) -> None:
    audit_path, audit_hash, seeds = _write_source_audit(tmp_path)
    output = tmp_path / "out"
    collect_spglobal_archive(
        source_audit_path=audit_path,
        expected_source_audit_sha256=audit_hash,
        output_directory=output,
        client_factory=_complete_transport(seeds).factory,
    )
    manifest_path = output / "_manifest.json"
    authority_path = output / "_authority.json"
    manifest = _read_json(manifest_path)
    manifest["completed_releases"] = int(manifest["completed_releases"]) - 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    authority = _read_json(authority_path)
    authority["artifact_sha256"] = hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest()
    authority_path.write_text(json.dumps(authority), encoding="utf-8")

    with pytest.raises(DataReadinessError, match="lineage or release counts"):
        archive_module.require_spglobal_event_reconstruction_ready(output)


def _write_source_audit(tmp_path: Path) -> tuple[Path, str, list[str]]:
    first = date(2019, 1, 2)
    sources: list[dict[str, object]] = []
    urls: list[str] = []
    for number in range(83):
        published = first + timedelta(days=number)
        url = f"https://press.spglobal.com/{published.isoformat()}-Seed-Company-{number}-Set-to-Join-S-P-500"
        urls.append(url)
        sources.append(
            {
                "published_date": published.isoformat(),
                "title": f"Seed Company {number} Set to Join S&P 500",
                "url": url,
            }
        )
    path = tmp_path / "source_audit.json"
    path.write_text(
        json.dumps(
            {
                "schema": "ml_v3.sp500_point_in_time_universe.v1",
                "source_manifest": {
                    "schema": "ml_v3.sp500_change_sources.v1",
                    "sources": sources,
                },
                "source_urls": urls,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path, hashlib.sha256(path.read_bytes()).hexdigest(), urls


def _complete_transport(seeds: list[str], *, release_body: bytes | None = None) -> _Transport:
    extra = "https://press.spglobal.com/2018-05-01-Netflix-and-Twitter-to-Join-S-P-500"
    search = _search_page(
        [
            (
                "2026-07-09",
                "Recent release",
                "https://press.spglobal.com/2026-07-09-recent",
            ),
            ("2019-01-02", "Seed Company Set to Join S&P 500", seeds[0]),
            ("2018-05-01", "Netflix and Twitter to Join S&P 500", extra),
        ],
        overlap_out=(
            "2018-04-13",
            "https://press.spglobal.com/2018-04-13-complete-overlap",
        ),
        page_tag="complete-zero",
    )
    boundary = _search_page(
        [
            (
                "2018-04-13",
                "Pagination overlap",
                "https://press.spglobal.com/2018-04-13-complete-overlap",
            ),
            (
                "2018-04-13",
                "Archive boundary",
                "https://press.spglobal.com/2018-04-13-boundary",
            )
        ],
        page_tag="complete-one",
    )
    body = release_body or _release_body()
    return _Transport(
        {
            "search:0": search,
            "search:99": boundary,
            **{url: body for url in [*seeds, extra]},
        }
    )


def _search_page(
    rows: list[tuple[str, str, str]],
    *,
    total_dated_urls: int = 100,
    page_tag: str = "page",
    overlap_out: tuple[str, str] | None = None,
) -> bytes:
    padded = list(rows)
    filler_date = min(
        [item[0] for item in rows]
        + ([overlap_out[0]] if overlap_out is not None else []),
        default="2026-07-09",
    )
    used = {url for _, _, url in padded}
    target_before_overlap = total_dated_urls - (1 if overlap_out is not None else 0)
    index = 0
    while len(used) < target_before_overlap:
        url = (
            f"https://press.spglobal.com/"
            f"{filler_date}-fixture-{page_tag}-{index}"
        )
        index += 1
        if url in used:
            continue
        used.add(url)
        padded.append((filler_date, "Unrelated archive item", url))
    if overlap_out is not None:
        overlap_date, overlap_url = overlap_out
        if overlap_url in used:
            raise AssertionError("fixture overlap_out URL must be unique on its source page")
        used.add(overlap_url)
        padded.append((overlap_date, "Pagination overlap", overlap_url))
    if len(used) != total_dated_urls:
        raise AssertionError("fixture search page has the wrong unique URL count")
    links = "".join(f'<a href="{url}">{title}</a>' for _, title, url in padded)
    return f"<html><body>{links}</body></html>".encode()


def _release_body() -> bytes:
    return b"""<html><body><table>
<tr><th>Effective Date</th><th>Index Name</th><th>Action</th><th>Company Name</th><th>Ticker</th><th>GICS Sector</th></tr>
<tr><td>May 7, 2018</td><td>S&amp;P 500</td><td>Addition</td><td>New Company</td><td>NEW</td><td>Industrials</td></tr>
<tr><td>May 7, 2018</td><td>S&amp;P 500</td><td>Deletion</td><td>Old Company</td><td>OLD</td><td>Industrials</td></tr>
</table></body></html>"""


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value
