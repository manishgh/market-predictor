"""One-page corporate-action transport and structural validation, not admission."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from hashlib import sha256
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.sources.http import HttpByteResponse

_ENDPOINT = "https://data.alpaca.markets/v1/corporate-actions"


def _validate_token(token: object) -> None:
    if token is not None and (not isinstance(token, str) or not token.strip()):
        raise ValueError("Alpaca corporate-actions page token must be a nonblank string or null")


def corporate_action_parameters(
    *, ticker: str, start: date, end: date, page_token: str | None = None, limit: int = 1000,
) -> dict[str, Any]:
    """Build the shared single-ticker query, retaining opaque page tokens exactly."""
    if not isinstance(ticker, str) or not ticker.strip():
        raise ValueError("Alpaca corporate-actions page requires one ticker")
    symbol = ticker.strip().upper()
    if "," in symbol or any(character.isspace() or ord(character) < 32 for character in symbol):
        raise ValueError("Alpaca corporate-actions page requires one ticker")
    if (
        not isinstance(start, date) or isinstance(start, datetime)
        or not isinstance(end, date) or isinstance(end, datetime)
        or start > end
    ):
        raise ValueError("Alpaca corporate-actions bounds must be dates with start <= end")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise ValueError("Alpaca corporate-actions page limit must be an integer from 1 to 1000")
    _validate_token(page_token)
    params: dict[str, Any] = {
        "symbols": symbol,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "region": "us",
        "data_quality": "all",
        "limit": limit,
        "sort": "asc",
    }
    if page_token is not None:
        params["page_token"] = page_token
    return params


def fetch_corporate_actions_page(
    source: AlpacaSource,
    *,
    ticker: str,
    start: date,
    end: date,
    page_token: str | None = None,
    limit: int = 1000,
    maximum_body_bytes: int = 4 * 1024**2,
) -> HttpByteResponse:
    """Fetch one bounded entity, without pagination, redirects, or retry attempts.

    The caller must decode/validate the returned evidence before consuming records.
    """
    params = corporate_action_parameters(ticker=ticker, start=start, end=end, page_token=page_token, limit=limit)
    if type(maximum_body_bytes) is not int or maximum_body_bytes < 1:
        raise ValueError("maximum_body_bytes must be a positive integer")
    response = source.client.get_bytes_with_metadata(
        _ENDPOINT,
        params=params,
        headers={**source.headers, "Accept-Encoding": "identity"},
        retries=1,
        maximum_body_bytes=maximum_body_bytes,
        allow_redirects=False,
        raise_for_status=False,
    )
    if len(response.body) > maximum_body_bytes:
        raise RuntimeError("Alpaca corporate-actions response exceeds maximum_body_bytes")
    return response


def _verify_expected_params(params: dict[str, Any]) -> None:
    try:
        canonical = corporate_action_parameters(
            ticker=params["symbols"],
            start=date.fromisoformat(params["start"]),
            end=date.fromisoformat(params["end"]),
            page_token=params.get("page_token"),
            limit=params["limit"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Invalid expected Alpaca corporate-actions query") from exc
    if params != canonical:
        raise ValueError("Expected Alpaca corporate-actions query must match the frozen contract")


def _verify_url(url: str, params: dict[str, Any]) -> None:
    try:
        parsed = urlsplit(url)
        pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True, errors="strict")
    except ValueError as exc:
        raise RuntimeError("Alpaca corporate-actions URL is malformed") from exc
    expected = {key: str(value) for key, value in params.items()}
    if (
        url.partition("?")[0] != _ENDPOINT
        or "#" in url
        or any(character.isspace() or ord(character) < 32 for character in url)
        or len(pairs) != len(expected)
        or dict(pairs) != expected
    ):
        raise RuntimeError("Alpaca corporate-actions URL does not match the frozen query")


def decode_corporate_actions_page(
    response: HttpByteResponse,
    *,
    expected_params: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], str | None]:
    """Validate transport and page structure without interpreting provider fields.

    Unknown families and incomplete records (including missing/null/blank IDs or
    missing CUSIPs) remain unchanged. They are not admitted identity, effective-date,
    ownership, or economic evidence. In particular, process_date is never substituted
    for effective_date. Pagination and semantic validation belong to the caller.
    """
    _verify_expected_params(expected_params)
    if (
        response.status_code != 200
        or response.redirect_chain
        or response.final_url != response.requested_url
    ):
        raise RuntimeError("Alpaca corporate-actions transport must be a direct HTTP 200 response")
    _verify_url(response.requested_url, expected_params)
    _verify_url(response.final_url, expected_params)
    clock = response.retrieved_at_utc
    if not isinstance(clock, datetime) or clock.tzinfo is None or clock.utcoffset() != timedelta(0):
        raise RuntimeError("Alpaca corporate-actions retrieval time must be UTC-aware")
    if (response.content_type or "").split(";", maxsplit=1)[0].strip().lower() != "application/json":
        raise RuntimeError("Alpaca corporate-actions response must use application/json")
    if response.content_encoding is not None and response.content_encoding.strip().lower() != "identity":
        raise RuntimeError("Alpaca corporate-actions response has unsupported content encoding")
    seen_headers: set[str] = set()
    for name, value in response.safe_headers:
        name = name.lower()
        if name in seen_headers:
            raise RuntimeError("Alpaca corporate-actions response has duplicate header metadata")
        seen_headers.add(name)
        if (
            (name == "content-type" and value != response.content_type)
            or (name == "content-encoding" and value != response.content_encoding)
        ):
            raise RuntimeError("Alpaca corporate-actions response header metadata is inconsistent")
    if (
        type(response.body_length) is not int
        or response.body_length != len(response.body)
        or response.sha256 != sha256(response.body).hexdigest()
        or response.body_representation != "http_entity_encoded"
    ):
        raise RuntimeError("Alpaca corporate-actions response body metadata is inconsistent")
    try:
        payload = parse_strict_json_object(response.body, label="Alpaca corporate-actions response")
    except (ValueError, RecursionError) as exc:
        raise RuntimeError("Alpaca corporate-actions response is not valid strict UTF-8 JSON") from exc
    actions = payload.get("corporate_actions")
    if not isinstance(actions, dict):
        raise RuntimeError("Alpaca corporate_actions must be an object")
    records: dict[str, list[dict[str, Any]]] = {}
    action_ids: set[str] = set()
    count = 0
    for family, rows in actions.items():
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise RuntimeError("Alpaca corporate-actions families must contain lists of records")
        count += len(rows)
        if count > expected_params["limit"]:
            raise RuntimeError("Alpaca corporate-actions record count exceeds page limit")
        for row in rows:
            action_id = row.get("id")
            if action_id is None:
                continue
            if not isinstance(action_id, str):
                raise RuntimeError("Alpaca corporate-action ID must be a string or null")
            if not action_id.strip():
                continue
            if action_id in action_ids:
                raise RuntimeError("Alpaca corporate-actions page contains duplicate action IDs")
            action_ids.add(action_id)
        records[family] = rows
    if "next_page_token" not in payload:
        raise RuntimeError("Alpaca corporate-actions page is missing next_page_token")
    next_token = payload["next_page_token"]
    try:
        _validate_token(next_token)
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    assert next_token is None or isinstance(next_token, str)
    return records, next_token
