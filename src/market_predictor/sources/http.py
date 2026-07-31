from __future__ import annotations

import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from hashlib import sha256
from typing import Any, cast

import requests

_SAFE_RESPONSE_HEADERS = (
    "cache-control",
    "content-encoding",
    "content-language",
    "content-type",
    "date",
    "etag",
    "expires",
    "last-modified",
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
    "x-ratelimit-reset",
)
_MAX_SAFE_HEADER_VALUE_LENGTH = 2_048
_DEFAULT_MAXIMUM_BODY_BYTES = 16 * 1024 * 1024
_READ_CHUNK_BYTES = 64 * 1024


@dataclass(frozen=True, slots=True)
class HttpByteResponse:
    body: bytes
    requested_url: str
    final_url: str
    redirect_chain: tuple[str, ...]
    status_code: int
    retrieved_at_utc: datetime
    content_type: str | None
    content_encoding: str | None
    etag: str | None
    last_modified: str | None
    body_length: int
    sha256: str
    body_representation: str
    safe_headers: tuple[tuple[str, str], ...]


def _http_error_message(method: str, url: str, error: Exception | None) -> str:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        response = error.response
        return f"{method} failed: {url} status={response.status_code}"
    if error is not None:
        return f"{method} failed: {url} error={error}"
    return f"{method} failed: {url}"


class HttpClient:
    def __init__(self, user_agent: str = "market-predictor/0.1", timeout: int = 30) -> None:
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})
        self.timeout = timeout

    def get_json(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retries: int = 3,
        pause: float = 1.0,
    ) -> Any:
        payload, _ = self.get_json_with_headers(
            url,
            params=params,
            headers=headers,
            retries=retries,
            pause=pause,
        )
        return payload

    def get_json_with_headers(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retries: int = 3,
        pause: float = 1.0,
    ) -> tuple[Any, dict[str, str]]:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = self.session.get(
                    url,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                if (
                    _is_retriable_status(response.status_code)
                    and attempt < retries - 1
                ):
                    time.sleep(
                        _retry_delay(
                            response,
                            attempt=attempt,
                            pause=pause,
                        )
                    )
                    continue
                response.raise_for_status()
                return response.json(), dict(response.headers)
            except requests.RequestException as exc:
                last_error = exc
                status = (
                    exc.response.status_code
                    if isinstance(exc, requests.HTTPError)
                    and exc.response is not None
                    else None
                )
                if status is not None and not _is_retriable_status(status):
                    break
                if attempt < retries - 1:
                    time.sleep(
                        pause * (2**attempt)
                        + random.uniform(0.0, pause)
                    )
        raise RuntimeError(_http_error_message("GET", url, last_error)) from last_error

    def get_bytes_with_metadata(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retries: int = 3,
        pause: float = 1.0,
        maximum_body_bytes: int = _DEFAULT_MAXIMUM_BODY_BYTES,
    ) -> HttpByteResponse:
        if maximum_body_bytes < 1:
            raise ValueError("maximum_body_bytes must be positive")
        last_error: Exception | None = None
        for attempt in range(retries):
            response: requests.Response | None = None
            try:
                request_headers = {"Accept-Encoding": "identity", **(headers or {})}
                response = self.session.get(
                    url,
                    params=params,
                    headers=request_headers,
                    timeout=self.timeout,
                    stream=True,
                )
                if (
                    _is_retriable_status(response.status_code)
                    and attempt < retries - 1
                ):
                    time.sleep(
                        _retry_delay(
                            response,
                            attempt=attempt,
                            pause=pause,
                        )
                    )
                    continue
                response.raise_for_status()
                body = _read_bounded_http_entity(
                    response,
                    maximum_body_bytes=maximum_body_bytes,
                )
                safe_headers = _bounded_safe_headers(response.headers)
                safe_header_map = dict(safe_headers)
                final_url = response.url or url
                requested_url = (
                    response.request.url
                    if response.request is not None and response.request.url
                    else url
                )
                redirect_chain = tuple(
                    [*(hop.url for hop in response.history), final_url]
                    if response.history
                    else []
                )
                return HttpByteResponse(
                    body=body,
                    requested_url=requested_url,
                    final_url=final_url,
                    redirect_chain=redirect_chain,
                    status_code=response.status_code,
                    retrieved_at_utc=datetime.now(UTC),
                    content_type=safe_header_map.get("content-type"),
                    content_encoding=safe_header_map.get("content-encoding"),
                    etag=safe_header_map.get("etag"),
                    last_modified=safe_header_map.get("last-modified"),
                    body_length=len(body),
                    sha256=sha256(body).hexdigest(),
                    body_representation="http_entity_encoded",
                    safe_headers=safe_headers,
                )
            except requests.RequestException as exc:
                last_error = exc
                status = (
                    exc.response.status_code
                    if isinstance(exc, requests.HTTPError)
                    and exc.response is not None
                    else None
                )
                if status is not None and not _is_retriable_status(status):
                    break
                if attempt < retries - 1:
                    time.sleep(
                        pause * (2**attempt)
                        + random.uniform(0.0, pause)
                    )
            finally:
                if response is not None:
                    response.close()
        raise RuntimeError(_http_error_message("GET", url, last_error)) from last_error

    def post_json_with_headers(
        self,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        retries: int = 3,
        pause: float = 1.0,
    ) -> tuple[Any, dict[str, str]]:
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                response = self.session.post(
                    url,
                    json=payload or {},
                    headers=headers,
                    timeout=self.timeout,
                )
                if (
                    _is_retriable_status(response.status_code)
                    and attempt < retries - 1
                ):
                    time.sleep(
                        _retry_delay(
                            response,
                            attempt=attempt,
                            pause=pause,
                        )
                    )
                    continue
                response.raise_for_status()
                return response.json(), dict(response.headers)
            except requests.RequestException as exc:
                last_error = exc
                status = (
                    exc.response.status_code
                    if isinstance(exc, requests.HTTPError)
                    and exc.response is not None
                    else None
                )
                if status is not None and not _is_retriable_status(status):
                    break
                if attempt < retries - 1:
                    time.sleep(
                        pause * (2**attempt)
                        + random.uniform(0.0, pause)
                    )
        raise RuntimeError(_http_error_message("POST", url, last_error)) from last_error


def _retry_delay(
    response: requests.Response,
    *,
    attempt: int,
    pause: float,
) -> float:
    retry_after = response.headers.get("Retry-After", "").strip()
    if retry_after:
        try:
            parsed = float(retry_after)
            return min(max(parsed, 0.0), 120.0)
        except ValueError:
            try:
                target = parsedate_to_datetime(retry_after)
                if target.tzinfo is None:
                    target = target.replace(tzinfo=UTC)
                delay = (
                    target.astimezone(UTC) - datetime.now(UTC)
                ).total_seconds()
                return min(max(delay, 0.0), 120.0)
            except (TypeError, ValueError, OverflowError):
                pass
    reset = response.headers.get("X-RateLimit-Reset", "").strip()
    if reset:
        try:
            delay = float(reset) - time.time()
            return min(max(delay, 0.0), 120.0)
        except ValueError:
            pass
    return min(
        cast(
            float,
            pause * (2**attempt) + random.uniform(0.0, pause),
        ),
        120.0,
    )


def _read_bounded_http_entity(
    response: requests.Response,
    *,
    maximum_body_bytes: int,
) -> bytes:
    """Read transfer-decoded but content-encoded entity bytes within a hard limit."""

    content_length = response.headers.get("Content-Length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError as exc:
            raise RuntimeError("HTTP response Content-Length is invalid") from exc
        if declared_length < 0 or declared_length > maximum_body_bytes:
            raise RuntimeError(
                "HTTP response exceeds maximum_body_bytes: "
                f"declared={declared_length} limit={maximum_body_bytes}"
            )
    response.raw.decode_content = False
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = response.raw.read(
            min(_READ_CHUNK_BYTES, maximum_body_bytes - total + 1)
        )
        if not chunk:
            break
        total += len(chunk)
        if total > maximum_body_bytes:
            raise RuntimeError(
                "HTTP response exceeds maximum_body_bytes: "
                f"observed>{maximum_body_bytes} limit={maximum_body_bytes}"
            )
        chunks.append(bytes(chunk))
    return b"".join(chunks)


def _bounded_safe_headers(
    headers: requests.structures.CaseInsensitiveDict[str] | dict[str, str],
) -> tuple[tuple[str, str], ...]:
    normalized = {str(name).lower(): str(value) for name, value in headers.items()}
    return tuple(
        (name, normalized[name][:_MAX_SAFE_HEADER_VALUE_LENGTH])
        for name in _SAFE_RESPONSE_HEADERS
        if name in normalized
    )


def _is_retriable_status(status: int) -> bool:
    return status in {408, 429} or 500 <= status <= 599
