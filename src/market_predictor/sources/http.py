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
    safe_headers: tuple[tuple[str, str], ...]


def _http_error_message(method: str, url: str, error: Exception | None) -> str:
    if isinstance(error, requests.HTTPError) and error.response is not None:
        response = error.response
        body = response.text[:300].replace("\n", " ").replace("\r", " ")
        return f"{method} failed: {url} status={response.status_code} body={body}"
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
    ) -> HttpByteResponse:
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
                body = response.content
                safe_headers = _bounded_safe_headers(response.headers)
                safe_header_map = dict(safe_headers)
                final_url = response.url or url
                redirect_chain = tuple(
                    [*(hop.url for hop in response.history), final_url]
                    if response.history
                    else []
                )
                return HttpByteResponse(
                    body=body,
                    requested_url=url,
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
