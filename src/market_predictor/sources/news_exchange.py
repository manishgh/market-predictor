"""Exchange adapter over the existing observed Alpaca transport."""

from __future__ import annotations

import hashlib
import json
from urllib.parse import parse_qsl, urlsplit

from market_predictor.evidence.news_exchange import (
    MAX_PAYLOAD_BYTES,
    NewsHttpReceipt,
    NewsPageRequest,
    ValidatedNewsReceipt,
    validate_news_receipt,
)
from market_predictor.sources.alpaca import AlpacaSource


def fetch_news_receipt(source: AlpacaSource, request: NewsPageRequest, *, producer_revision: str) -> ValidatedNewsReceipt:
    page = source.fetch_news_page_observed(
        request.symbol, request.start_utc, request.end_utc,
        page_token=request.page_token, include_content=request.include_content, limit=request.limit,
        maximum_body_bytes=MAX_PAYLOAD_BYTES,
        allow_redirects=False,
    )
    if page.raw_body is None or page.retrieved_at_utc is None or page.status_code != 200:
        raise ValueError("news exchange requires original successful HTTP receipt evidence")
    if page.redirect_chain or page.request_page_token != request.page_token:
        raise ValueError("news receipt endpoint or page identity does not match the request")
    expected_query = {
        "symbols": request.symbol, "start": request.start_utc.isoformat(),
        "end": request.end_utc.isoformat(), "sort": "asc", "limit": str(request.limit),
        "include_content": str(request.include_content).lower(),
    }
    if request.page_token is not None:
        expected_query["page_token"] = request.page_token
    for recorded_url in (page.requested_url, page.final_url or page.requested_url):
        parsed = urlsplit(recorded_url or "")
        pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=7)
        if (
            parsed.scheme != "https" or parsed.netloc != "data.alpaca.markets"
            or parsed.path != "/v1beta1/news" or parsed.fragment
            or len(pairs) != len(expected_query) or dict(pairs) != expected_query
        ):
            raise ValueError("news receipt recorded request does not match the supplied request")
    receipt = NewsHttpReceipt(
        schema_version="alpaca.news_http_receipt.v1",
        producer="market_predictor",
        producer_revision=producer_revision,
        endpoint="alpaca.news",
        transport="http", status_code=200,
        byte_representation="decoded_http_body_utf8",
        availability_basis="observed_receipt",
        request=request, received_at_utc=page.retrieved_at_utc,
        payload_sha256=hashlib.sha256(page.raw_body).hexdigest(),
        payload_bytes=len(page.raw_body),
    )
    values = receipt.model_dump(mode="json")
    values["received_at_utc"] = receipt.received_at_utc.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    for key in ("start_utc", "end_utc"):
        values["request"][key] = getattr(request, key).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    manifest = json.dumps(values, ensure_ascii=True, separators=(",", ":")).encode("utf-8")
    return validate_news_receipt(manifest, page.raw_body)
