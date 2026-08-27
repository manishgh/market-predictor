"""Strict GDELT DOC 2.0 provider transport."""
from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final, Protocol, cast
from urllib.parse import urlsplit

import requests

from market_predictor.core.errors import DataReadinessError

GDELT_DOCUMENT_ENDPOINT: Final = "https://api.gdeltproject.org/api/v2/doc/doc"
MAX_GDELT_RECORDS: Final = 250
GDELT_MAX_ATTEMPTS: Final = 4
GDELT_INITIAL_BACKOFF_SECONDS: Final = 0.5
GDELT_MAX_BACKOFF_SECONDS: Final = 8.0
GDELT_MAX_RETRY_AFTER_SECONDS: Final = 30.0
GDELT_RETRYABLE_STATUS_CODES: Final = frozenset({429, 500, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class GdeltDocumentRequest:
    queries: tuple[str, ...]
    requested_start_utc: datetime
    requested_end_utc: datetime
    max_records: int = MAX_GDELT_RECORDS
    timeout_seconds: float = 30.0


@dataclass(frozen=True, slots=True)
class GdeltDocumentResult:
    records: Sequence[Mapping[str, object]]
    complete: bool
    completed_queries: tuple[str, ...]
    errors: tuple[str, ...] = ()
    raw_response_sha256: str | None = None


class GdeltDocumentFetcher(Protocol):
    def __call__(self, request: GdeltDocumentRequest) -> GdeltDocumentResult: ...


def validate_gdelt_document_request(request: GdeltDocumentRequest) -> GdeltDocumentRequest:
    """Validate and normalize a request before network or scorer allocation."""

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
    return GdeltDocumentRequest(
        queries=queries,
        requested_start_utc=start,
        requested_end_utc=end,
        max_records=request.max_records,
        timeout_seconds=float(request.timeout_seconds),
    )


def fetch_gdelt_documents(
    request: GdeltDocumentRequest,
    *,
    max_attempts: int = GDELT_MAX_ATTEMPTS,
    initial_backoff_seconds: float = GDELT_INITIAL_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> GdeltDocumentResult:
    """Fetch one bounded window from the GDELT article-list API."""

    normalized = validate_gdelt_document_request(request)
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
    return GdeltDocumentResult(
        records=records,
        complete=complete,
        completed_queries=normalized.queries,
        raw_response_sha256=_json_sha256(response_hashes),
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
                GDELT_DOCUMENT_ENDPOINT,
                params=dict(parameters),
                timeout=timeout_seconds,
                allow_redirects=False,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_transport_error = exc
            if attempt + 1 == max_attempts:
                break
            sleep(_retry_delay(None, attempt, initial_backoff_seconds))
            continue

        _validate_gdelt_response_identity(response)
        if 300 <= response.status_code < 400:
            raise DataReadinessError("GDELT response attempted an HTTP redirect")
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


def _validate_gdelt_response_identity(response: requests.Response) -> None:
    if response.history:
        raise DataReadinessError("GDELT response contains an unexpected redirect history")
    if not isinstance(response.url, str) or not response.url:
        raise DataReadinessError("GDELT response has no provider URL")
    expected = urlsplit(GDELT_DOCUMENT_ENDPOINT)
    actual = urlsplit(response.url)
    expected_port = expected.port or 443
    actual_port = actual.port or (443 if actual.scheme.lower() == "https" else None)
    if (
        actual.scheme.lower() != expected.scheme
        or actual.hostname != expected.hostname
        or actual_port != expected_port
        or actual.path != expected.path
    ):
        raise DataReadinessError("GDELT response provider identity does not match the configured endpoint")


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


def _strict_utc(value: datetime, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise DataReadinessError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
