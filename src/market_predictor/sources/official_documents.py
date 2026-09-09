"""Retain official response evidence; never infer identities, entitlements or fills.

Hashes detect corruption, not an operator rewriting both bytes and receipts.
Downstream interpretation must pin a collection report hash and review the bodies.
"""
from __future__ import annotations

import hashlib
import os
import re
import tomllib
from collections.abc import Callable
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import uuid4

import requests
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator

from market_predictor.config import Settings
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority, write_json_object
from market_predictor.locking import file_lock
from market_predictor.sources.http import HttpByteResponse, HttpClient
from market_predictor.sources.sec import SecSource

_HOSTS = frozenset({
    "www.sec.gov", "data.sec.gov", "ir.amd.com", "investors.bbwinc.com", "investors.fiserv.com",
    "www.nasdaqtrader.com", "infomemo.theocc.com", "www.fdic.gov",
})
_SHA = r"^[0-9a-f]{64}$"
_MAX_METADATA_BYTES = 1024 * 1024


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OfficialDocument(_Contract):
    document_id: str = Field(pattern=r"^[a-z][a-z0-9_-]{2,95}$")
    url: str
    purpose: str = Field(min_length=12, max_length=1000)
    expected_media: Literal["html", "pdf"]

    @field_validator("url")
    @classmethod
    def official_url(cls, value: str) -> str:
        parts = urlsplit(value)
        if (parts.scheme != "https" or parts.hostname not in _HOSTS or parts.username
                or parts.password or parts.port not in {None, 443} or parts.fragment
                or value != value.strip() or any(ord(char) < 33 for char in value)):
            raise ValueError("document requires an exact public official HTTPS URL without credentials or fragments")
        # Only the two official bulletin endpoints in this inventory use queries.
        if parts.query and not (
            (parts.hostname == "infomemo.theocc.com" and re.fullmatch(r"number=\d+", parts.query))
            or (parts.hostname == "www.nasdaqtrader.com" and re.fullmatch(r"id=eca\d{4}-\d+", parts.query))
        ):
            raise ValueError("document URL has an unsupported query")
        return value


class OfficialDocumentInventory(_Contract):
    schema_version: Literal["market_predictor.official_document_inventory"]
    maximum_response_bytes: int = Field(strict=True, ge=1, le=16 * 1024 * 1024)
    attempts_per_document_per_run: Literal[1]
    documents: list[OfficialDocument] = Field(min_length=1, max_length=256)

    @field_validator("documents")
    @classmethod
    def unique_documents(cls, value: list[OfficialDocument]) -> list[OfficialDocument]:
        if len({item.document_id for item in value}) != len(value) or len({item.url for item in value}) != len(value):
            raise ValueError("official document IDs and URLs must be unique")
        return value


class _Response(_Contract):
    requested_url: str
    final_url: str
    redirect_chain: list[str]
    status_code: int = Field(strict=True)
    retrieved_at_utc: AwareDatetime
    content_type: str | None
    content_encoding: str | None
    etag: str | None
    last_modified: str | None
    body_length: int = Field(strict=True, ge=0)
    sha256: str = Field(pattern=_SHA)
    body_representation: Literal["http_entity_encoded"]
    safe_headers: list[tuple[str, str]]


class _Receipt(_Contract):
    schema_version: Literal["market_predictor.official_document_attempt"]
    inventory_sha256: str = Field(pattern=_SHA)
    document_id: str
    attempt_id: str = Field(pattern=r"^[0-9a-f]{32}$")
    started_at_utc: AwareDatetime
    completed_at_utc: AwareDatetime
    state: Literal["archived_unreviewed", "response_rejected", "request_failed"]
    response: _Response | None
    body_path: str | None
    error_type: str | None
    failure_status_code: int | None
    interpretation_status: Literal["not_reviewed"] = "not_reviewed"
    historical_availability_proven: Literal[False] = False
    accounting_eligible: Literal[False] = False


DocumentFetcher = Callable[[OfficialDocument, int], HttpByteResponse]


class OfficialDocumentSource:
    """Sequential bounded requests; SEC uses its existing process-wide governor."""

    def __init__(self, settings: Settings) -> None:
        self.sec_client = SecSource(settings).client
        self.other_client = HttpClient(user_agent=requests.utils.default_user_agent())

    def fetch(self, document: OfficialDocument, maximum_bytes: int) -> HttpByteResponse:
        client = self.sec_client if urlsplit(document.url).hostname in {"www.sec.gov", "data.sec.gov"} else self.other_client
        return client.get_bytes_with_metadata(
            document.url, maximum_body_bytes=maximum_bytes, retries=1, allow_redirects=False,
        )

    def close(self) -> None:
        self.sec_client.session.close()
        self.other_client.session.close()


def load_official_document_inventory(path: Path) -> OfficialDocumentInventory:
    if path.stat().st_size > _MAX_METADATA_BYTES:
        raise DataReadinessError("official document inventory exceeds size limit")
    return OfficialDocumentInventory.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def _read_object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > _MAX_METADATA_BYTES:
        raise DataReadinessError("official document metadata exceeds size limit")
    return parse_strict_json_object(path.read_bytes(), label=str(path))


def _request(inventory: OfficialDocumentInventory) -> dict[str, Any]:
    payload = inventory.model_dump(mode="json")
    return {"inventory": payload, "inventory_sha256": json_sha256(payload)}


def _response_state(document: OfficialDocument, response: _Response) -> Literal["archived_unreviewed", "response_rejected"]:
    media = (response.content_type or "").split(";", 1)[0].strip().lower()
    expected = {"text/html", "application/xhtml+xml"} if document.expected_media == "html" else {"application/pdf"}
    if (response.status_code != 200 or response.requested_url != document.url
            or response.final_url != document.url or response.redirect_chain
            or response.body_length == 0 or media not in expected):
        return "response_rejected"
    return "archived_unreviewed"


def _verify_attempt(
    path: Path, document: OfficialDocument, request: dict[str, Any], maximum_bytes: int, collection_root: Path,
) -> _Receipt:
    receipt_path = resolve_inside_authority(collection_root, path / "_receipt.json")
    payload = _read_object(receipt_path)
    digest = payload.pop("receipt_sha256", None)
    if digest != json_sha256(payload):
        raise DataReadinessError("official document receipt hash mismatch")
    receipt = _Receipt.model_validate(payload)
    if (receipt.inventory_sha256 != request["inventory_sha256"] or receipt.document_id != document.document_id
            or receipt.attempt_id != path.name or receipt.completed_at_utc < receipt.started_at_utc):
        raise DataReadinessError("official document attempt identity or clock mismatch")
    expected_files = {"_receipt.json"}
    if receipt.response is None:
        if (receipt.state != "request_failed" or receipt.body_path is not None or not receipt.error_type
                or (receipt.failure_status_code is not None and not 400 <= receipt.failure_status_code <= 599)):
            raise DataReadinessError("official document failure receipt is inconsistent")
    else:
        response = receipt.response
        if (receipt.error_type is not None or receipt.failure_status_code is not None
                or not receipt.started_at_utc <= response.retrieved_at_utc <= receipt.completed_at_utc
                or receipt.body_path != f"body-{response.sha256}.bin"
                or response.body_length > maximum_bytes or receipt.state != _response_state(document, response)):
            raise DataReadinessError("official document response receipt is inconsistent")
        body_path = resolve_inside_authority(path, receipt.body_path)
        if body_path.stat().st_size != response.body_length:
            raise DataReadinessError("official document body length mismatch")
        if hashlib.sha256(body_path.read_bytes()).hexdigest() != response.sha256:
            raise DataReadinessError("official document body hash mismatch")
        expected_files.add(str(receipt.body_path))
    if {entry.name for entry in path.iterdir()} != expected_files:
        raise DataReadinessError("official document attempt has unexpected files")
    return receipt


def verify_official_document_collection(
    output_directory: Path, inventory: OfficialDocumentInventory,
) -> dict[str, Any]:
    """Replay all committed attempts offline; HTTP success is not content approval."""
    output_directory = output_directory.resolve()
    request = _request(inventory)
    if _read_object(resolve_inside_authority(output_directory, "_request.json")) != request:
        raise DataReadinessError("official document inventory changed; use a new collection directory")
    units_root = output_directory / "documents"
    documents = {item.document_id: item for item in inventory.documents}
    if units_root.exists() and any(path.name not in documents or not path.is_dir() for path in units_root.iterdir()):
        raise DataReadinessError("official document collection has an unknown document")
    results: list[dict[str, Any]] = []
    for document in inventory.documents:
        parent = units_root / document.document_id
        attempts: list[dict[str, Any]] = []
        archived = 0
        for path in sorted(parent.iterdir()) if parent.exists() else []:
            receipt = _verify_attempt(path, document, request, inventory.maximum_response_bytes, output_directory)
            archived += int(receipt.state == "archived_unreviewed")
            attempts.append({
                "receipt_path": (path / "_receipt.json").relative_to(output_directory).as_posix(),
                "receipt_sha256": hashlib.sha256((path / "_receipt.json").read_bytes()).hexdigest(),
                "state": receipt.state,
            })
        if archived > 1:
            raise DataReadinessError("official document collection has multiple successful acquisitions")
        results.append({"document_id": document.document_id, "archived": bool(archived), "attempts": attempts})
    count = sum(int(row["archived"]) for row in results)
    report = {
        "schema_version": "market_predictor.official_document_collection",
        "inventory_sha256": request["inventory_sha256"],
        "status": "collected_unreviewed" if count == len(documents) else "incomplete",
        "requested_documents": len(documents), "archived_documents": count,
        "documents": results, "interpretation_status": "not_reviewed",
        "historical_availability_proven": False, "accounting_eligible": False,
    }
    _verify_previous_reports(output_directory, report)
    return report


def _verify_previous_reports(output: Path, current: dict[str, Any]) -> None:
    reports = output / "reports"
    if not reports.exists():
        return
    for path in reports.iterdir():
        path = resolve_inside_authority(output, path)
        payload = _read_object(path)
        if path.name != f"{hashlib.sha256(path.read_bytes()).hexdigest()}.json":
            raise DataReadinessError("official document report hash mismatch")
        if (set(payload) != set(current) | {"attempted_this_run", "deferred_hosts_this_run"}
                or not isinstance(payload["documents"], list)
                or len(payload["documents"]) != len(current["documents"])):
            raise DataReadinessError("official document report contract differs")
        count = 0
        for prior, present in zip(payload["documents"], current["documents"], strict=True):
            if (not isinstance(prior, dict) or set(prior) != {"document_id", "archived", "attempts"}
                    or prior["document_id"] != present["document_id"] or not isinstance(prior["attempts"], list)):
                raise DataReadinessError("official document report identity differs")
            seen: set[str] = set()
            for attempt in prior["attempts"]:
                if not isinstance(attempt, dict) or attempt not in present["attempts"] or attempt["receipt_path"] in seen:
                    raise DataReadinessError("official document report receipt anchor differs")
                seen.add(attempt["receipt_path"])
            archived = any(row["state"] == "archived_unreviewed" for row in prior["attempts"])
            if prior["archived"] is not archived:
                raise DataReadinessError("official document report acquisition status differs")
            count += int(archived)
        status = "collected_unreviewed" if count == current["requested_documents"] else "incomplete"
        expected = {**current, "documents": payload["documents"], "archived_documents": count, "status": status}
        if any(payload[key] != value for key, value in expected.items()):
            raise DataReadinessError("official document report summary differs")


def _failure_status(error: Exception) -> int | None:
    cause: BaseException | None = error
    while cause is not None:
        if isinstance(cause, requests.HTTPError) and cause.response is not None:
            return int(cause.response.status_code)
        cause = cause.__cause__
    return None


def collect_official_documents(
    *, inventory: OfficialDocumentInventory, output_directory: Path, fetch: DocumentFetcher,
) -> dict[str, Any]:
    """Retain one attempt per missing document, preserving failures and prior bytes."""
    output_directory = output_directory.resolve()
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(output_directory.parent / f".{output_directory.name}.collection", timeout=0):
        request = _request(inventory)
        if not output_directory.exists():
            staging = output_directory.parent / f".{output_directory.name}.{uuid4().hex}.pending"
            staging.mkdir()
            write_json_object(staging / "_request.json", request)
            os.rename(staging, output_directory)
        report = verify_official_document_collection(output_directory, inventory)
        complete = {str(row["document_id"]) for row in report["documents"] if row["archived"]}
        blocked_hosts: set[str | None] = set()
        attempted = 0
        for document in inventory.documents:
            host = urlsplit(document.url).hostname
            if document.document_id in complete or host in blocked_hosts:
                continue
            attempt_id = uuid4().hex
            staging = output_directory / ".pending" / attempt_id
            staging.mkdir(parents=True)
            started = datetime.now(UTC)
            attempted += 1
            response: HttpByteResponse | None = None
            error_type: str | None = None
            failure_status: int | None = None
            try:
                response = fetch(document, inventory.maximum_response_bytes)
            except (RuntimeError, requests.RequestException, OSError) as exc:
                error_type = type(exc).__name__
                failure_status = _failure_status(exc)
                if failure_status in {403, 429}:
                    # Respect provider cooldown without sleeping for ten minutes or
                    # preventing acquisition from unrelated hosts in this run.
                    blocked_hosts.add(host)
            response_metadata = None
            body_path = None
            state: Literal["archived_unreviewed", "response_rejected", "request_failed"] = "request_failed"
            if response is not None:
                if len(response.body) > inventory.maximum_response_bytes:
                    raise DataReadinessError("official document response exceeds bounded size")
                metadata = asdict(response)
                metadata.pop("body")
                response_metadata = _Response.model_validate(metadata)
                if response.body_length != len(response.body) or response.sha256 != hashlib.sha256(response.body).hexdigest():
                    raise DataReadinessError("official document transport body metadata differs")
                state = _response_state(document, response_metadata)
                body_path = f"body-{response.sha256}.bin"
                (staging / body_path).write_bytes(response.body)
            receipt = _Receipt(
                schema_version="market_predictor.official_document_attempt", inventory_sha256=request["inventory_sha256"],
                document_id=document.document_id, attempt_id=attempt_id, started_at_utc=started,
                completed_at_utc=datetime.now(UTC), state=state, response=response_metadata, body_path=body_path,
                error_type=error_type, failure_status_code=failure_status,
            ).model_dump(mode="json")
            write_json_object(staging / "_receipt.json", {**receipt, "receipt_sha256": json_sha256(receipt)})
            _verify_attempt(staging, document, request, inventory.maximum_response_bytes, output_directory)
            destination = output_directory / "documents" / document.document_id / attempt_id
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.rename(staging, destination)
        report = verify_official_document_collection(output_directory, inventory)
        report["attempted_this_run"] = attempted
        report["deferred_hosts_this_run"] = sorted(str(host) for host in blocked_hosts)
        reports = output_directory / "reports"
        reports.mkdir(exist_ok=True)
        temporary = output_directory / ".pending" / f"{uuid4().hex}.json"
        temporary.parent.mkdir(exist_ok=True)
        write_json_object(temporary, report)
        report_path = reports / f"{hashlib.sha256(temporary.read_bytes()).hexdigest()}.json"
        if report_path.exists():
            if report_path.read_bytes() != temporary.read_bytes():
                raise DataReadinessError("official document report hash collision")
            temporary.unlink()
        else:
            os.rename(temporary, report_path)
        return {**report, "report_path": str(report_path)}
