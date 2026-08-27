from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest
import requests

from market_predictor.core.errors import DataReadinessError
from market_predictor.sources.gdelt import (
    GDELT_DOCUMENT_ENDPOINT,
    GdeltDocumentRequest,
    fetch_gdelt_documents,
    validate_gdelt_document_request,
)

START = datetime(2025, 1, 7, 19, 0, tzinfo=UTC)
END = datetime(2025, 1, 10, 20, 0, tzinfo=UTC)
QUERIES = (
    "(war OR sanctions OR shipping OR energy)",
    "(semiconductor OR export controls)",
)


def test_document_source_uses_exact_request_and_preserves_query_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bodies = [
        (
            b'{"articles":[{"title":"Shipping disruption raises energy risk",'
            b'"url":"https://example.com/story#fragment",'
            b'"seendate":"2025-01-10T19:30:00Z",'
            b'"domain":"example.com","summary":"Global macro event"}]}'
        ),
        b'{"articles":[]}',
    ]
    responses = [_response(200, raw=body) for body in bodies]
    calls: list[tuple[str, Mapping[str, object], float, bool]] = []

    def fake_get(
        url: str,
        *,
        params: Mapping[str, object],
        timeout: float,
        allow_redirects: bool,
    ) -> requests.Response:
        calls.append((url, params, timeout, allow_redirects))
        return responses.pop(0)

    monkeypatch.setattr(requests, "get", fake_get)
    result = fetch_gdelt_documents(_request())

    assert result.complete is True
    assert result.completed_queries == QUERIES
    assert [record["collection_query"] for record in result.records] == [QUERIES[0]]
    assert result.raw_response_sha256 == (
        "fffda4358a7061552d96d59b62d5c90684cef4863e3c718dd53527e4d7efb9da"
    )
    assert calls == [
        (
            GDELT_DOCUMENT_ENDPOINT,
            {
                "query": query,
                "mode": "artlist",
                "format": "json",
                "maxrecords": 100,
                "startdatetime": "20250107190000",
                "enddatetime": "20250110200000",
            },
            30.0,
            False,
        )
        for query in QUERIES
    ]


@pytest.mark.parametrize("transport_error", [requests.Timeout(), requests.ConnectionError()])
def test_document_source_retries_transient_failures(
    monkeypatch: pytest.MonkeyPatch,
    transport_error: requests.RequestException,
) -> None:
    outcomes: list[requests.Response | requests.RequestException] = [
        transport_error,
        _response(429, {"error": "rate limited"}, headers={"Retry-After": "0.25"}),
        _response(200, {"articles": []}),
    ]
    sleeps: list[float] = []

    def fake_get(
        _url: str,
        **_kwargs: object,
    ) -> requests.Response:
        outcome = outcomes.pop(0)
        if isinstance(outcome, requests.RequestException):
            raise outcome
        return outcome

    monkeypatch.setattr(requests, "get", fake_get)
    result = fetch_gdelt_documents(
        _request(queries=(QUERIES[0],)),
        max_attempts=4,
        initial_backoff_seconds=0.1,
        sleep=sleeps.append,
    )

    assert result.complete is True
    assert sleeps == [0.1, 0.25]


def test_document_source_rejects_invalid_json_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_get(_url: str, **_kwargs: object) -> requests.Response:
        nonlocal calls
        calls += 1
        return _response(200, raw=b"not-json")

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(DataReadinessError, match="invalid JSON"):
        fetch_gdelt_documents(_request(queries=(QUERIES[0],)))
    assert calls == 1


def test_document_source_marks_maximum_size_response_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    articles = [
        {
            "title": f"Article {index}",
            "url": f"https://example.com/{index}",
            "seendate": "2025-01-10T19:30:00Z",
        }
        for index in range(2)
    ]
    monkeypatch.setattr(
        requests,
        "get",
        lambda *_args, **_kwargs: _response(200, {"articles": articles}),
    )

    result = fetch_gdelt_documents(_request(queries=(QUERIES[0],), max_records=2))
    assert result.complete is False
    assert len(result.records) == 2


def test_document_source_rejects_permanent_http_failure_without_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def fake_get(_url: str, **_kwargs: object) -> requests.Response:
        nonlocal calls
        calls += 1
        return _response(400, {"error": "bad request"})

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(DataReadinessError, match="permanently.*400"):
        fetch_gdelt_documents(_request(queries=(QUERIES[0],)))
    assert calls == 1


@pytest.mark.parametrize(
    ("response_url", "status_code", "headers", "match"),
    (
        (GDELT_DOCUMENT_ENDPOINT, 302, {"Location": "https://foreign.example/doc"}, "redirect"),
        ("https://foreign.example/api/v2/doc/doc", 200, {}, "provider identity"),
        ("https://api.gdeltproject.org/api/v2/doc/other", 200, {}, "provider identity"),
    ),
)
def test_document_source_rejects_redirects_and_wrong_provider_identity(
    monkeypatch: pytest.MonkeyPatch,
    response_url: str,
    status_code: int,
    headers: Mapping[str, str],
    match: str,
) -> None:
    def fake_get(_url: str, **kwargs: object) -> requests.Response:
        assert kwargs["allow_redirects"] is False
        response = _response(status_code, {"articles": []}, headers=headers)
        response.url = response_url
        return response

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(DataReadinessError, match=match):
        fetch_gdelt_documents(_request(queries=(QUERIES[0],)))


def test_document_source_stops_after_bounded_retryable_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    sleeps: list[float] = []

    def fake_get(_url: str, **_kwargs: object) -> requests.Response:
        nonlocal calls
        calls += 1
        return _response(503, {"error": "temporarily unavailable"})

    monkeypatch.setattr(requests, "get", fake_get)
    with pytest.raises(DataReadinessError, match="after 3 attempts.*503"):
        fetch_gdelt_documents(
            _request(queries=(QUERIES[0],)),
            max_attempts=3,
            initial_backoff_seconds=0.25,
            sleep=sleeps.append,
        )
    assert calls == 3
    assert sleeps == [0.25, 0.5]


def test_document_request_validation_is_independent_of_collection_scoring() -> None:
    normalized = validate_gdelt_document_request(_request())
    assert normalized.queries == QUERIES
    with pytest.raises(ValueError, match="reversed"):
        validate_gdelt_document_request(
            GdeltDocumentRequest(
                queries=QUERIES,
                requested_start_utc=END,
                requested_end_utc=START,
            )
        )


def _request(
    *,
    queries: tuple[str, ...] = QUERIES,
    max_records: int = 100,
) -> GdeltDocumentRequest:
    return GdeltDocumentRequest(
        queries=queries,
        requested_start_utc=START,
        requested_end_utc=END,
        max_records=max_records,
    )


def _response(
    status_code: int,
    payload: object | None = None,
    *,
    raw: bytes | None = None,
    headers: Mapping[str, str] | None = None,
) -> requests.Response:
    response = requests.Response()
    response.status_code = status_code
    response.url = GDELT_DOCUMENT_ENDPOINT
    response.headers.update(headers or {})
    response.headers.setdefault("Content-Type", "application/json")
    response._content = raw if raw is not None else json.dumps(payload).encode("utf-8")
    return response
