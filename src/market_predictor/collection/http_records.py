"""Complete records of provider HTTP responses, so that receipts can be decoded again on load."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from hashlib import sha256

from market_predictor.core.errors import DataReadinessError
from market_predictor.sources.http import HttpByteResponse


def http_response_record(response: HttpByteResponse) -> dict[str, object]:
    """Every response field except the body, which is stored by its SHA-256."""
    return {
        "requested_url": response.requested_url,
        "final_url": response.final_url,
        "redirect_chain": list(response.redirect_chain),
        "status_code": response.status_code,
        "retrieved_at_utc": response.retrieved_at_utc.isoformat(),
        "content_type": response.content_type,
        "content_encoding": response.content_encoding,
        "etag": response.etag,
        "last_modified": response.last_modified,
        "body_length": response.body_length,
        "body_sha256": response.sha256,
        "body_representation": response.body_representation,
        "safe_headers": [list(pair) for pair in response.safe_headers],
    }


def http_response_from_record(record: Mapping[str, object], body: bytes) -> HttpByteResponse:
    """Rebuild a recorded response around its stored body, refusing a body that does not match."""
    if sha256(body).hexdigest() != record.get("body_sha256") or len(body) != record.get("body_length"):
        raise DataReadinessError("a stored response body does not match its record")
    try:
        return HttpByteResponse(
            body=body,
            requested_url=str(record["requested_url"]),
            final_url=str(record["final_url"]),
            redirect_chain=tuple(str(url) for url in _list(record["redirect_chain"])),
            status_code=int(str(record["status_code"])),
            retrieved_at_utc=datetime.fromisoformat(str(record["retrieved_at_utc"])),
            content_type=_optional_text(record["content_type"]),
            content_encoding=_optional_text(record["content_encoding"]),
            etag=_optional_text(record["etag"]),
            last_modified=_optional_text(record["last_modified"]),
            body_length=len(body),
            sha256=sha256(body).hexdigest(),
            body_representation=str(record["body_representation"]),
            safe_headers=tuple((str(pair[0]), str(pair[1])) for pair in map(_list, _list(record["safe_headers"]))),
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise DataReadinessError("a stored response record is malformed") from exc


def _list(value: object) -> list[object]:
    if not isinstance(value, list):
        raise TypeError("expected a list")
    return value


def _optional_text(value: object) -> str | None:
    return None if value is None else str(value)
