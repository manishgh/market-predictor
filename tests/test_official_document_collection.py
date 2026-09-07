from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest
import requests
from pydantic import ValidationError
from typer.testing import CliRunner
from urllib3.exceptions import ProtocolError, ReadTimeoutError

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.locking import LockTimeout, file_lock
from market_predictor.sources import official_documents as module
from market_predictor.sources.http import HttpByteResponse, HttpClient
from market_predictor.sources.official_documents import (
    OfficialDocument,
    OfficialDocumentInventory,
    collect_official_documents,
    load_official_document_inventory,
    verify_official_document_collection,
)


def _inventory(count: int = 2) -> OfficialDocumentInventory:
    return OfficialDocumentInventory.model_validate({
        "schema_version": "market_predictor.official_document_inventory",
        "maximum_response_bytes": 1024,
        "attempts_per_document_per_run": 1,
        "documents": [{
            "document_id": f"filing_{index}", "url": f"https://www.sec.gov/Archives/filing{index}.htm",
            "purpose": "Test-only source acquisition fixture", "expected_media": "html",
        } for index in range(count)],
    })


def _response(document: OfficialDocument, body: bytes = b"<html>Test-only filing</html>") -> HttpByteResponse:
    return HttpByteResponse(
        body=body, requested_url=document.url, final_url=document.url, redirect_chain=(), status_code=200,
        retrieved_at_utc=datetime.now(UTC), content_type="text/html", content_encoding=None, etag=None,
        last_modified=None, body_length=len(body), sha256=hashlib.sha256(body).hexdigest(),
        body_representation="http_entity_encoded", safe_headers=(("content-type", "text/html"),),
    )


def _fetch(document: OfficialDocument, maximum: int) -> HttpByteResponse:
    assert maximum == 1024
    return _response(document)


def _no_fetch(*args: Any) -> HttpByteResponse:
    pytest.fail("resume or offline verification must not make a network request")


def test_collection_and_offline_replay_never_admit_economic_evidence(tmp_path: Path) -> None:
    inventory = _inventory()
    output = tmp_path / "collection"
    result = collect_official_documents(inventory=inventory, output_directory=output, fetch=_fetch)
    assert result["status"] == "collected_unreviewed"
    assert result["archived_documents"] == 2
    assert result["accounting_eligible"] is False
    assert result["historical_availability_proven"] is False
    assert result["interpretation_status"] == "not_reviewed"
    first_receipts = list(output.glob("documents/*/*/_receipt.json"))
    before = {path: path.read_bytes() for path in first_receipts}
    resumed = collect_official_documents(inventory=inventory, output_directory=output, fetch=_no_fetch)
    assert resumed["attempted_this_run"] == 0
    assert before == {path: path.read_bytes() for path in first_receipts}
    assert verify_official_document_collection(output, inventory)["archived_documents"] == 2
    report = Path(result["report_path"])
    assert report.stem == hashlib.sha256(report.read_bytes()).hexdigest()


def test_failed_document_does_not_block_others_and_retry_retains_failure(tmp_path: Path) -> None:
    output = tmp_path / "collection"
    inventory = _inventory()
    calls: list[str] = []

    def fetch(document: OfficialDocument, maximum: int) -> HttpByteResponse:
        calls.append(document.document_id)
        if document.document_id == "filing_0":
            raise RuntimeError("test-only request failure")
        return _fetch(document, maximum)

    first = collect_official_documents(inventory=inventory, output_directory=output, fetch=fetch)
    assert first["status"] == "incomplete"
    assert first["archived_documents"] == 1
    assert calls == ["filing_0", "filing_1"]
    calls.clear()

    def retry(document: OfficialDocument, maximum: int) -> HttpByteResponse:
        calls.append(document.document_id)
        return _fetch(document, maximum)

    second = collect_official_documents(inventory=inventory, output_directory=output, fetch=retry)
    assert second["status"] == "collected_unreviewed"
    assert calls == ["filing_0"]
    assert len(second["documents"][0]["attempts"]) == 2


@pytest.mark.parametrize("status", [403, 429])
def test_provider_cooldown_defers_host_without_sleeping_or_blocking_other_hosts(tmp_path: Path, status: int) -> None:
    payload = _inventory(3).model_dump()
    payload["documents"][2]["url"] = "https://ir.amd.com/filing.htm"
    inventory = OfficialDocumentInventory.model_validate(payload)
    called: list[str] = []

    def fetch(document: OfficialDocument, maximum: int) -> HttpByteResponse:
        called.append(document.document_id)
        if document.document_id == "filing_0":
            response = requests.Response()
            response.status_code = status
            error = requests.HTTPError(response=response)
            raise RuntimeError("failed request") from error
        return _fetch(document, maximum)

    result = collect_official_documents(inventory=inventory, output_directory=tmp_path / "collection", fetch=fetch)
    assert result["archived_documents"] == 1
    assert called == ["filing_0", "filing_2"]
    assert result["deferred_hosts_this_run"] == ["www.sec.gov"]
    assert result["documents"][1]["attempts"] == []


@pytest.mark.parametrize("kind", ["redirect", "status", "empty", "wrong_media"])
def test_non_document_responses_are_retained_but_rejected(tmp_path: Path, kind: str) -> None:
    def fetch(document: OfficialDocument, maximum: int) -> HttpByteResponse:
        response = _fetch(document, maximum)
        if kind == "redirect":
            return replace(response, status_code=302, final_url="https://www.sec.gov/other", redirect_chain=(document.url,))
        if kind == "status":
            return replace(response, status_code=503)
        if kind == "empty":
            return _response(document, b"")
        return replace(response, content_type="application/json")

    output = tmp_path / "collection"
    result = collect_official_documents(inventory=_inventory(1), output_directory=output, fetch=fetch)
    assert result["archived_documents"] == 0
    assert result["documents"][0]["attempts"][0]["state"] == "response_rejected"
    assert len(list(output.glob("documents/*/*/body-*.bin"))) == 1


def test_encoded_non_utf8_entity_is_retained_without_claiming_parsed_content(tmp_path: Path) -> None:
    encoded = gzip.compress(b"<html>\xff\xfe Test-only encoded bytes</html>")

    def fetch(document: OfficialDocument, maximum: int) -> HttpByteResponse:
        return replace(_response(document, encoded), content_encoding="gzip")

    output = tmp_path / "collection"
    result = collect_official_documents(inventory=_inventory(1), output_directory=output, fetch=fetch)
    assert result["status"] == "collected_unreviewed"
    assert next(output.glob("documents/*/*/body-*.bin")).read_bytes() == encoded


@pytest.mark.parametrize("kind", ["body", "receipt", "request", "rehashed_state", "extra_file"])
def test_retained_tampering_fails_before_any_network_request(tmp_path: Path, kind: str) -> None:
    output = tmp_path / "collection"
    inventory = _inventory(1)
    collect_official_documents(inventory=inventory, output_directory=output, fetch=_fetch)
    receipt_path = next(output.glob("documents/*/*/_receipt.json"))
    if kind == "body":
        next(output.glob("documents/*/*/body-*.bin")).write_bytes(b"corrupt")
    elif kind == "request":
        payload = json.loads((output / "_request.json").read_text())
        payload["inventory_sha256"] = "0" * 64
        (output / "_request.json").write_text(json.dumps(payload))
    elif kind == "extra_file":
        (receipt_path.parent / "extra.bin").write_bytes(b"unexpected")
    else:
        payload = json.loads(receipt_path.read_text())
        payload["state"] = "request_failed"
        if kind == "rehashed_state":
            payload.pop("receipt_sha256")
            payload["receipt_sha256"] = json_sha256(payload)
        receipt_path.write_text(json.dumps(payload))
    with pytest.raises(DataReadinessError):
        collect_official_documents(inventory=inventory, output_directory=output, fetch=_no_fetch)


def test_inventory_change_requires_new_directory(tmp_path: Path) -> None:
    output = tmp_path / "collection"
    collect_official_documents(inventory=_inventory(), output_directory=output, fetch=_fetch)
    with pytest.raises(DataReadinessError, match="inventory changed"):
        collect_official_documents(inventory=_inventory(1), output_directory=output, fetch=_no_fetch)


@pytest.mark.parametrize("rehashed", [False, True])
def test_report_hash_and_receipt_anchors_are_verified(tmp_path: Path, rehashed: bool) -> None:
    output = tmp_path / "collection"
    inventory = _inventory(1)
    report = collect_official_documents(inventory=inventory, output_directory=output, fetch=_fetch)
    path = Path(report["report_path"])
    payload = json.loads(path.read_text())
    payload["documents"][0]["attempts"][0]["receipt_sha256"] = "0" * 64
    path.write_text(json.dumps(payload))
    if rehashed:
        path.rename(path.with_name(f"{hashlib.sha256(path.read_bytes()).hexdigest()}.json"))
    with pytest.raises(DataReadinessError, match="report"):
        collect_official_documents(inventory=inventory, output_directory=output, fetch=_no_fetch)


@pytest.mark.parametrize("kind", ["oversize", "hash"])
def test_byte_limit_and_transport_metadata_are_checked_before_publication(tmp_path: Path, kind: str) -> None:
    def fetch(document: OfficialDocument, maximum: int) -> HttpByteResponse:
        if kind == "oversize":
            return _response(document, b"x" * (maximum + 1))
        return replace(_response(document), sha256="0" * 64)

    output = tmp_path / kind
    with pytest.raises(DataReadinessError):
        collect_official_documents(inventory=_inventory(1), output_directory=output, fetch=fetch)
    assert not list(output.glob("documents/*/*/_receipt.json"))


def test_interrupted_publication_is_not_resumed_as_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "collection"
    original = module.os.rename

    def interrupted(source: Path, destination: Path) -> None:
        if "documents" in destination.parts:
            raise OSError("test-only interruption before commit")
        original(source, destination)

    monkeypatch.setattr(module.os, "rename", interrupted)
    with pytest.raises(OSError, match="interruption"):
        collect_official_documents(inventory=_inventory(1), output_directory=output, fetch=_fetch)
    assert verify_official_document_collection(output, _inventory(1))["archived_documents"] == 0
    monkeypatch.setattr(module.os, "rename", original)
    assert collect_official_documents(inventory=_inventory(1), output_directory=output, fetch=_fetch)["archived_documents"] == 1


def test_concurrent_collector_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "collection"
    with file_lock(tmp_path / ".collection.collection"):
        with pytest.raises(LockTimeout):
            collect_official_documents(inventory=_inventory(1), output_directory=output, fetch=_no_fetch)


@pytest.mark.parametrize("error", [ProtocolError("test broken stream"), ReadTimeoutError(None, "/filing", "test timeout")])
def test_stream_failures_are_normalized_and_other_documents_continue(tmp_path: Path, error: Exception) -> None:
    client = HttpClient()
    response = requests.Response()
    response.status_code = 200
    response.raw = Mock()
    response.raw.read.side_effect = error
    client.session = Mock()
    client.session.get.return_value = response
    calls: list[str] = []

    def fetch(document: OfficialDocument, maximum: int) -> HttpByteResponse:
        calls.append(document.document_id)
        if document.document_id == "filing_0":
            return client.get_bytes_with_metadata(document.url, maximum_body_bytes=maximum, retries=1, allow_redirects=False)
        return _fetch(document, maximum)

    output = tmp_path / "collection"
    first = collect_official_documents(inventory=_inventory(), output_directory=output, fetch=fetch)
    assert calls == ["filing_0", "filing_1"]
    assert first["archived_documents"] == 1
    assert first["documents"][0]["attempts"][0]["state"] == "request_failed"
    second = collect_official_documents(inventory=_inventory(), output_directory=output, fetch=_fetch)
    assert second["archived_documents"] == 2
    assert second["attempted_this_run"] == 1


def test_external_attempt_reference_is_rejected_before_body_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output = tmp_path / "collection"
    inventory = _inventory(1)
    collect_official_documents(inventory=inventory, output_directory=output, fetch=_fetch)
    receipt = next(output.glob("documents/*/*/_receipt.json"))
    external = tmp_path / "external"
    external.mkdir()
    external_receipt = external / "_receipt.json"
    external_receipt.write_bytes(receipt.read_bytes())
    original = Path.resolve

    # Windows symlink creation needs privileges; emulate the resolved target of
    # a directory junction to exercise the same containment invariant everywhere.
    def resolve(path: Path, *args: Any, **kwargs: Any) -> Path:
        if path == receipt:
            return external_receipt
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(DataReadinessError, match="escapes authority"):
        collect_official_documents(inventory=inventory, output_directory=output, fetch=_no_fetch)


@pytest.mark.parametrize("url", [
    "http://www.sec.gov/file", "https://www.sec.gov.evil.test/file", "https://user:secret@www.sec.gov/file",
    "https://www.sec.gov/file?token=secret", "https://www.sec.gov/file#fragment",
])
def test_inventory_rejects_unapproved_or_credential_urls(url: str) -> None:
    with pytest.raises(ValidationError):
        OfficialDocument(document_id="test_filing", url=url, purpose="Test-only evidence reference", expected_media="html")


def test_inventory_rejects_duplicates_and_unbounded_limits() -> None:
    payload = _inventory().model_dump()
    payload["documents"][1] = payload["documents"][0]
    with pytest.raises(ValidationError):
        OfficialDocumentInventory.model_validate(payload)
    payload = _inventory().model_dump()
    payload["maximum_response_bytes"] = 17 * 1024 * 1024
    with pytest.raises(ValidationError):
        OfficialDocumentInventory.model_validate(payload)


def test_source_uses_sec_client_and_bounded_non_redirecting_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, dict[str, Any]]] = []

    class Client:
        def get_bytes_with_metadata(self, url: str, **kwargs: Any) -> HttpByteResponse:
            calls.append((url, kwargs))
            return _response(_inventory(1).documents[0])

    source = object.__new__(module.OfficialDocumentSource)
    monkeypatch.setattr(source, "sec_client", Client(), raising=False)
    monkeypatch.setattr(source, "other_client", None, raising=False)
    source.fetch(_inventory(1).documents[0], 321)
    assert calls[0][1] == {"maximum_body_bytes": 321, "retries": 1, "allow_redirects": False}


def test_offline_cli_does_not_construct_credentials_or_network_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import market_predictor.commands.swing_collection as commands
    from market_predictor.collection_cli import app

    inventory = _inventory(1)
    output = tmp_path / "collection"
    collect_official_documents(inventory=inventory, output_directory=output, fetch=_fetch)
    monkeypatch.setattr(commands, "load_official_document_inventory", lambda path: inventory)
    monkeypatch.setattr(commands, "OfficialDocumentSource", _no_fetch)
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["collect-swing-holding-source-documents", "--out-dir", "collection", "--offline"])
    assert result.exit_code == 0, result.output


def test_real_inventory_is_bounded_and_unique() -> None:
    inventory = load_official_document_inventory(Path("configs/swing_holding_source_documents.toml"))
    assert len(inventory.documents) == 10
