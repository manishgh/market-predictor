from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta, timezone
from hashlib import sha256
from io import BytesIO
from typing import Any
from unittest.mock import Mock
from urllib.parse import urlencode

import pytest
import requests

from market_predictor.config import Settings
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.sources.alpaca_corporate_actions import (
    corporate_action_parameters,
    decode_corporate_actions_page,
    fetch_corporate_actions_page,
)
from market_predictor.sources.http import HttpByteResponse

ENDPOINT = "https://data.alpaca.markets/v1/corporate-actions"
START = date(2020, 1, 1)
END = date(2020, 2, 1)
PARAMS: dict[str, Any] = {
    "symbols": "TEST", "start": START.isoformat(), "end": END.isoformat(),
    "region": "us", "data_quality": "all", "limit": 1000, "sort": "asc",
}
CASH_MERGER = {
    "id": "synthetic-cash-merger",
    "acquiree_cusip": "000000001",
    "acquiree_symbol": "TEST",
    "process_date": "2020-01-10",
    "effective_date": "2020-01-09",
    "payable_date": "2020-01-13",
    "rate": "42.50",
}


def _response(body: bytes, *, params: dict[str, Any] | None = None) -> HttpByteResponse:
    url = f"{ENDPOINT}?{urlencode(PARAMS if params is None else params)}"
    return HttpByteResponse(
        body=body, requested_url=url, final_url=url, redirect_chain=(), status_code=200,
        retrieved_at_utc=datetime(2026, 9, 9, tzinfo=UTC),
        content_type="application/json; charset=utf-8", content_encoding=None,
        etag=None, last_modified=None, body_length=len(body), sha256=sha256(body).hexdigest(),
        body_representation="http_entity_encoded",
        safe_headers=(("content-type", "application/json; charset=utf-8"),),
    )


def _page(actions: Any = None, token: Any = None) -> HttpByteResponse:
    return _response(json.dumps({
        "corporate_actions": {} if actions is None else actions, "next_page_token": token,
    }).encode())


def _source() -> AlpacaSource:
    return AlpacaSource(Settings(ALPACA_API_KEY_ID="synthetic-key", ALPACA_API_SECRET_KEY="synthetic-secret"))


def test_shared_parameters_defaults_and_same_day_window() -> None:
    assert corporate_action_parameters(ticker=" test ", start=START, end=END) == PARAMS
    assert corporate_action_parameters(ticker="TEST", start=START, end=START, limit=2, page_token="opaque+/=") == {
        **PARAMS, "end": START.isoformat(), "limit": 2, "page_token": "opaque+/=",
    }


def test_preserves_cash_merger_unknown_families_and_incomplete_records() -> None:
    actions = {
        "cash_mergers": [CASH_MERGER],
        "future_provider_family": [{"id": "future-1", "opaque": {"items": [1, None, "x"]}}],
        "name_changes": [{"process_date": "2020-01-12"}, {"id": None}, {"id": ""}, {}],
        "empty_future_family": [],
    }
    records, token = decode_corporate_actions_page(_page(actions, "opaque+/=token"), expected_params=PARAMS)
    assert records == actions
    assert records["cash_mergers"][0]["effective_date"] != records["cash_mergers"][0]["process_date"]
    assert "effective_date" not in records["name_changes"][0]
    assert "acquirer_symbol" not in records["cash_mergers"][0]
    assert token == "opaque+/=token"


@pytest.mark.parametrize("encoding", [None, "identity", "IDENTITY"])
def test_empty_page_and_supported_encoding(encoding: str | None) -> None:
    response = replace(_page(), content_encoding=encoding)
    assert decode_corporate_actions_page(response, expected_params=PARAMS) == ({}, None)


@pytest.mark.parametrize("updates", [
    {"status_code": 201}, {"status_code": 301}, {"status_code": 429},
    {"redirect_chain": (ENDPOINT,)}, {"final_url": "https://example.test/actions"},
    {"retrieved_at_utc": None}, {"retrieved_at_utc": datetime(2026, 9, 9)},
    {"retrieved_at_utc": datetime(2026, 9, 9, tzinfo=timezone(timedelta(hours=1)))},
    {"content_type": None}, {"content_type": "text/html"},
    {"content_encoding": "gzip"}, {"content_encoding": "br"}, {"content_encoding": ""},
    {"content_encoding": "identity, gzip"},
    {"sha256": "0" * 64}, {"body_length": 1}, {"body_length": True},
    {"body_representation": "decoded"},
    {"safe_headers": (("content-encoding", "gzip"),)},
    {"safe_headers": (("content-type", "text/html"),)},
    {"safe_headers": (("date", "x"), ("Date", "y"))},
])
def test_rejects_invalid_transport_metadata(updates: dict[str, Any]) -> None:
    with pytest.raises(RuntimeError):
        decode_corporate_actions_page(replace(_page(), **updates), expected_params=PARAMS)


@pytest.mark.parametrize("endpoint", [
    "http://data.alpaca.markets/v1/corporate-actions",
    "https://data.alpaca.markets.evil.test/v1/corporate-actions",
    "https://data.alpaca.markets/v1/corporate-actions/",
    "https://data.alpaca.markets/v2/corporate-actions",
    "https://user@data.alpaca.markets/v1/corporate-actions",
    "https://data.alpaca.markets:444/v1/corporate-actions",
    " https://data.alpaca.markets/v1/corporate-actions",
])
def test_rejects_wrong_endpoint(endpoint: str) -> None:
    url = f"{endpoint}?{urlencode(PARAMS)}"
    with pytest.raises(RuntimeError, match="URL"):
        decode_corporate_actions_page(replace(_page(), requested_url=url, final_url=url), expected_params=PARAMS)


@pytest.mark.parametrize("suffix", ["&symbols=TEST", "&extra=1", "#fragment", "#", "&malformed"])
def test_rejects_ambiguous_or_extra_query(suffix: str) -> None:
    url = _page().requested_url + suffix
    with pytest.raises(RuntimeError, match="URL"):
        decode_corporate_actions_page(replace(_page(), requested_url=url, final_url=url), expected_params=PARAMS)


@pytest.mark.parametrize("key,value", [
    ("symbols", "OTHER"), ("symbols", "TEST,OTHER"), ("start", "2020-01-02"),
    ("end", "2020-02-02"), ("region", "eu"), ("data_quality", "confirmed"),
    ("limit", 10), ("sort", "desc"), ("page_token", "unexpected"),
])
def test_rejects_wrong_query_parameters(key: str, value: Any) -> None:
    response = _response(_page().body, params={**PARAMS, key: value})
    with pytest.raises(RuntimeError, match="URL"):
        decode_corporate_actions_page(response, expected_params=PARAMS)


def test_page_token_must_match_exactly_and_is_not_normalized() -> None:
    params = {**PARAMS, "page_token": "opaque+/=previous"}
    response = _response(_page(token="opaque+/=next").body, params=params)
    assert decode_corporate_actions_page(response, expected_params=params)[1] == "opaque+/=next"
    with pytest.raises(RuntimeError, match="URL"):
        decode_corporate_actions_page(response, expected_params={**params, "page_token": "other"})


@pytest.mark.parametrize("updates", [
    {"symbols": "TEST,OTHER"}, {"symbols": "test"}, {"symbols": ""},
    {"region": "eu"}, {"data_quality": "confirmed"}, {"sort": "desc"},
    {"start": "20200101"}, {"start": "2021-01-01"}, {"end": None},
    {"limit": 0}, {"limit": -1}, {"limit": 1001}, {"limit": True}, {"limit": "1000"},
    {"page_token": None}, {"page_token": ""}, {"extra": "x"},
])
def test_rejects_invalid_expected_contract(updates: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        decode_corporate_actions_page(_page(), expected_params={**PARAMS, **updates})


@pytest.mark.parametrize("body", [
    b"", b"not-json", b"{", b"[]", b"null", b"{} trailing", b"\xff",
    b'{"corporate_actions":{},"corporate_actions":{},"next_page_token":null}',
    b'{"corporate_actions":{"x":[{"id":"a","id":"b"}]},"next_page_token":null}',
    b'{"corporate_actions":{"x":[],"x":[]},"next_page_token":null}',
    b'{"corporate_actions":{},"next_page_token":null,"next_page_token":"x"}',
    b'{"corporate_actions":{"x":[{"rate":NaN}]},"next_page_token":null}',
    b'{"corporate_actions":{"x":[{"rate":Infinity}]},"next_page_token":null}',
    b'{"corporate_actions":{"x":[{"rate":-Infinity}]},"next_page_token":null}',
    b'{"corporate_actions":{"x":[{"rate":1e999}]},"next_page_token":null}',
])
def test_rejects_malformed_or_non_strict_json(body: bytes) -> None:
    with pytest.raises(RuntimeError, match="strict UTF-8 JSON"):
        decode_corporate_actions_page(_response(body), expected_params=PARAMS)


@pytest.mark.parametrize("body", [
    b'{"next_page_token":null}', b'{"corporate_actions":null,"next_page_token":null}',
    b'{"corporate_actions":[],"next_page_token":null}', b'{"corporate_actions":{}}',
])
def test_rejects_missing_or_invalid_envelope(body: bytes) -> None:
    with pytest.raises(RuntimeError):
        decode_corporate_actions_page(_response(body), expected_params=PARAMS)


@pytest.mark.parametrize("rows", [None, {}, "bad", [None], [1], ["record"], [[]]])
def test_rejects_null_or_non_record_family_entries(rows: Any) -> None:
    with pytest.raises(RuntimeError, match="lists of records"):
        decode_corporate_actions_page(_page({"unknown": rows}), expected_params=PARAMS)


@pytest.mark.parametrize("token", [0, 1, True, [], {}, "", "  "])
def test_rejects_invalid_next_token(token: Any) -> None:
    with pytest.raises(RuntimeError, match="token"):
        decode_corporate_actions_page(_page(token=token), expected_params=PARAMS)


@pytest.mark.parametrize("actions", [
    {"cash_mergers": [{"id": "same"}, {"id": "same"}]},
    {"cash_mergers": [{"id": "same"}], "future": [{"id": "same"}]},
])
def test_rejects_duplicate_ids_across_entire_page(actions: dict[str, Any]) -> None:
    with pytest.raises(RuntimeError, match="duplicate action IDs"):
        decode_corporate_actions_page(_page(actions), expected_params=PARAMS)


@pytest.mark.parametrize("action_id", [1, True, [], {}])
def test_rejects_malformed_present_id(action_id: Any) -> None:
    with pytest.raises(RuntimeError, match="ID must be"):
        decode_corporate_actions_page(_page({"future": [{"id": action_id}]}), expected_params=PARAMS)


def test_limit_counts_all_families_including_incomplete_records() -> None:
    params = {**PARAMS, "limit": 2}
    actions: dict[str, list[dict[str, Any]]] = {"cash_mergers": [CASH_MERGER], "unknown": [{}]}
    response = _response(_page(actions).body, params=params)
    assert decode_corporate_actions_page(response, expected_params=params)[0] == actions
    actions["unknown"].append({})
    with pytest.raises(RuntimeError, match="exceeds page limit"):
        decode_corporate_actions_page(_response(_page(actions).body, params=params), expected_params=params)


@pytest.mark.parametrize("page_token", [None, "opaque+/=previous"])
def test_fetch_uses_source_client_headers_and_one_bounded_attempt(page_token: str | None) -> None:
    source = _source()
    client = Mock()
    client.get_bytes_with_metadata.return_value = _page()
    source.client = client
    result = fetch_corporate_actions_page(source, ticker="test", start=START, end=END, page_token=page_token)
    params = dict(PARAMS)
    if page_token is not None:
        params["page_token"] = page_token
    client.get_bytes_with_metadata.assert_called_once_with(
        ENDPOINT, params=params, headers={**source.headers, "Accept-Encoding": "identity"},
        retries=1, maximum_body_bytes=4 * 1024**2, allow_redirects=False, raise_for_status=False,
    )
    assert result is client.get_bytes_with_metadata.return_value


@pytest.mark.parametrize("updates", [
    {"ticker": ""}, {"ticker": "TEST,OTHER"}, {"ticker": "TEST OTHER"},
    {"ticker": "TEST\nOTHER"}, {"start": END, "end": START},
    {"start": datetime(2020, 1, 1, tzinfo=UTC)}, {"limit": 0}, {"limit": True},
    {"limit": 1.5}, {"maximum_body_bytes": 0}, {"maximum_body_bytes": True},
    {"page_token": ""}, {"page_token": 1},
])
def test_fetch_rejects_invalid_arguments_before_transport(updates: dict[str, Any]) -> None:
    source = _source()
    client = Mock()
    source.client = client
    kwargs: dict[str, Any] = {"ticker": "TEST", "start": START, "end": END, **updates}
    with pytest.raises(ValueError):
        fetch_corporate_actions_page(source, **kwargs)
    client.get_bytes_with_metadata.assert_not_called()


def _http_response(status: int, body: bytes) -> requests.Response:
    response = requests.Response()
    response.status_code = status
    response.url = _page().requested_url
    response.request = requests.Request("GET", ENDPOINT, params=PARAMS).prepare()
    response.headers["Content-Type"] = "application/json"
    response.raw = BytesIO(body)
    return response


@pytest.mark.parametrize("status", [429, 500, 503])
def test_real_client_does_not_retry_synthetic_http_errors(status: int) -> None:
    source = _source()
    session = Mock()
    source.client.session = session
    session.get.return_value = _http_response(status, b"failure")
    response = fetch_corporate_actions_page(source, ticker="TEST", start=START, end=END)
    assert response.status_code == status
    assert response.body == b"failure"
    assert response.content_type == "application/json"
    with pytest.raises(RuntimeError, match="direct HTTP 200"):
        decode_corporate_actions_page(response, expected_params=PARAMS)
    session.get.assert_called_once()
    assert session.get.call_args.kwargs["allow_redirects"] is False


def test_real_client_enforces_stream_limit_without_network() -> None:
    source = _source()
    session = Mock()
    source.client.session = session
    session.get.return_value = _http_response(200, b"12345")
    with pytest.raises(RuntimeError, match="maximum_body_bytes"):
        fetch_corporate_actions_page(source, ticker="TEST", start=START, end=END, maximum_body_bytes=4)
    session.get.assert_called_once()


def test_real_client_round_trip_synthetic_entity_metadata() -> None:
    source = _source()
    session = Mock()
    source.client.session = session
    session.get.return_value = _http_response(200, _page({"cash_mergers": [CASH_MERGER]}).body)
    response = fetch_corporate_actions_page(source, ticker="TEST", start=START, end=END)
    assert decode_corporate_actions_page(response, expected_params=PARAMS) == ({"cash_mergers": [CASH_MERGER]}, None)
    session.get.assert_called_once()
    assert session.get.call_args.kwargs["headers"]["Accept-Encoding"] == "identity"
