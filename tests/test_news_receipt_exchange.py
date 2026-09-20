from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from urllib.parse import urlencode

import pytest

from market_predictor.evidence.news_exchange import MAX_MANIFEST_BYTES, MAX_PAYLOAD_BYTES, read_news_receipt, validate_news_receipt
from market_predictor.sources.alpaca import AlpacaNewsPage, AlpacaSource
from market_predictor.sources.news_exchange import fetch_news_receipt

FIXTURE_PATH = Path(__file__).parent / "fixtures/news_receipt_exchange.json"
CASES = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[row["name"] for row in CASES])
def test_shared_receipt_vectors(case: dict[str, object]) -> None:
    manifest = str(case["manifest_utf8"]).encode("utf-8")
    payload = base64.b64decode(str(case["payload_base64"])) if "payload_base64" in case else str(case["payload_utf8"]).encode("utf-8")
    if not case["valid"]:
        with pytest.raises(ValueError):
            validate_news_receipt(manifest, payload)
        return
    validated = validate_news_receipt(manifest, payload)
    assert validated.manifest_bytes == manifest
    assert validated.payload_bytes == payload
    assert validated.receipt_id == hashlib.sha256(manifest).hexdigest()
    assert not validated.observable_at(datetime(2019, 7, 10, tzinfo=UTC))
    assert validated.observable_at(validated.manifest.received_at_utc)
    assert not validated.observable_at(validated.manifest.received_at_utc - timedelta(microseconds=1))
    assert "content" not in json.loads(validated.payload_bytes)["news"][0]


def test_file_import_preserves_identity_and_original_clock(tmp_path: Path) -> None:
    case = CASES[0]
    manifest = tmp_path / "receipt.json"
    payload = tmp_path / "response.json"
    manifest.write_bytes(case["manifest_utf8"].encode())
    payload.write_bytes(case["payload_utf8"].encode())
    pin = hashlib.sha256(case["manifest_utf8"].encode()).hexdigest()
    first = read_news_receipt(manifest, payload, expected_receipt_sha256=pin)
    again = read_news_receipt(manifest, payload, expected_receipt_sha256=pin)
    assert first == again
    payload.write_bytes(payload.read_bytes() + b" ")
    with pytest.raises(ValueError, match="integrity"):
        read_news_receipt(manifest, payload, expected_receipt_sha256=pin)
    manifest.write_bytes(case["manifest_utf8"].replace("2026-09-17T10:00:00", "2019-07-10T10:00:00").encode())
    with pytest.raises(ValueError, match="independent integrity pin"):
        read_news_receipt(manifest, payload, expected_receipt_sha256=pin)


def test_receipt_size_limits_precede_json_parsing() -> None:
    with pytest.raises(ValueError, match="byte limit"):
        validate_news_receipt(b" " * (MAX_MANIFEST_BYTES + 1), b"{}")
    with pytest.raises(ValueError, match="byte limit"):
        validate_news_receipt(b"{}", b" " * (MAX_PAYLOAD_BYTES + 1))


@pytest.mark.parametrize("body", [b'{"news":[],"news":[]}', b'{"news":[{"value":NaN}]}', b'{"news":[{"value":1e400}]}', b'{"news":[1]}'])
def test_integrity_does_not_replace_payload_validation(body: bytes) -> None:
    manifest = json.loads(CASES[0]["manifest_utf8"])
    manifest.update(payload_sha256=hashlib.sha256(body).hexdigest(), payload_bytes=len(body))
    with pytest.raises(ValueError):
        validate_news_receipt(json.dumps(manifest).encode(), body)


def test_exchange_calls_observed_transport_and_preserves_exact_bytes() -> None:
    saved = validate_news_receipt(CASES[0]["manifest_utf8"].encode(), CASES[0]["payload_utf8"].encode())
    source = Mock(spec=AlpacaSource)
    source.fetch_news_page_observed.return_value = AlpacaNewsPage(
        request_page_token=None, next_page_token=None, news=(), raw_body=saved.payload_bytes,
        retrieved_at_utc=saved.manifest.received_at_utc, status_code=200,
        requested_url="https://data.alpaca.markets/v1beta1/news?" + urlencode({
            "symbols": "MSFT", "start": saved.manifest.request.start_utc.isoformat(),
            "end": saved.manifest.request.end_utc.isoformat(), "sort": "asc", "limit": "50", "include_content": "true",
        }),
    )
    receipt = fetch_news_receipt(source, saved.manifest.request, producer_revision="a" * 40)
    assert receipt.payload_bytes == saved.payload_bytes
    assert receipt.manifest.received_at_utc == saved.manifest.received_at_utc
    source.fetch_news_page_observed.assert_called_once_with(
        "MSFT", saved.manifest.request.start_utc, saved.manifest.request.end_utc,
        page_token=None, include_content=True, limit=50,
        maximum_body_bytes=MAX_PAYLOAD_BYTES,
        allow_redirects=False,
    )


@pytest.mark.parametrize("change", ["symbol", "duplicate", "dates", "sort", "limit", "content", "redirect"])
def test_transport_request_identity_cannot_be_relabelled(change: str) -> None:
    saved = validate_news_receipt(CASES[0]["manifest_utf8"].encode(), CASES[0]["payload_utf8"].encode())
    query = {"symbols": "MSFT", "start": saved.manifest.request.start_utc.isoformat(),
        "end": saved.manifest.request.end_utc.isoformat(), "sort": "asc", "limit": "50", "include_content": "true"}
    updates = {"symbol": ("symbols", "AAPL"), "dates": ("start", "2020-01-01T00:00:00+00:00"),
        "sort": ("sort", "desc"), "limit": ("limit", "1"), "content": ("include_content", "false")}
    if change in updates:
        key, value = updates[change]
        query[key] = value
    url = "https://data.alpaca.markets/v1beta1/news?" + urlencode(query)
    if change == "duplicate":
        url += "&symbols=MSFT"
    source = Mock(spec=AlpacaSource)
    source.fetch_news_page_observed.return_value = AlpacaNewsPage(
        None, None, (), raw_body=saved.payload_bytes, status_code=200, retrieved_at_utc=saved.manifest.received_at_utc,
        requested_url=url, final_url="https://example.com/news" if change == "redirect" else url,
    )
    with pytest.raises(ValueError):
        fetch_news_receipt(source, saved.manifest.request, producer_revision="a" * 40)


def test_exchange_rejects_reconstructed_body_or_missing_receipt_clock() -> None:
    saved = validate_news_receipt(CASES[0]["manifest_utf8"].encode(), CASES[0]["payload_utf8"].encode())
    source = Mock(spec=AlpacaSource)
    source.fetch_news_page_observed.return_value = AlpacaNewsPage(None, None, (), raw_payload={"news": []})
    with pytest.raises(ValueError, match="original"):
        fetch_news_receipt(source, saved.manifest.request, producer_revision="a" * 40)
