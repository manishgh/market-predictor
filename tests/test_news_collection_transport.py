"""Exercise the real requests redirect/retry machinery without network access."""

from __future__ import annotations

import io
import json
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest
import requests
from requests.adapters import BaseAdapter

from market_predictor.config import Settings
from market_predictor.evidence.news_collection import (
    NewsCollectionPlan,
    NewsCollectionWindow,
    collect_news_receipts,
    news_collection_plan_sha256,
)
from market_predictor.evidence.news_exchange import validate_news_receipt
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.sources.news_collection import alpaca_news_receipt_fetcher


class _SyntheticHttpAdapter(BaseAdapter):
    def __init__(self, *, status: int = 200, body: bytes = b'{"news":[]}', fail_symbol: str | None = None) -> None:
        self.status = status
        self.body = body
        self.fail_symbol = fail_symbol
        self.urls: list[str] = []

    def send(self, request: requests.PreparedRequest, **kwargs: object) -> requests.Response:
        url = str(request.url)
        self.urls.append(url)
        if parse_qs(urlsplit(url).query).get("symbols") == [self.fail_symbol]:
            raise requests.ConnectionError("synthetic outage")
        response = requests.Response()
        response.status_code = self.status
        response.url = url
        response.request = request
        response.raw = io.BytesIO(self.body)
        if self.status == 302:
            response.headers["Location"] = "https://untrusted.example/news"
        return response

    def close(self) -> None:
        pass


def _plan() -> NewsCollectionPlan:
    case = json.loads((Path(__file__).parent / "fixtures/news_receipt_exchange.json").read_text(encoding="utf-8"))["cases"][0]
    saved = validate_news_receipt(case["manifest_utf8"].encode(), case["payload_utf8"].encode())
    return NewsCollectionPlan(producer_revision="a" * 40, max_pages_per_window=3, windows=tuple(
        NewsCollectionWindow(window_id=symbol, request=saved.manifest.request.model_copy(update={"symbol": symbol}))
        for symbol in ("MSFT", "NVDA")
    ))


def _source(transport: _SyntheticHttpAdapter) -> AlpacaSource:
    source = AlpacaSource(Settings(ALPACA_API_KEY_ID="synthetic-test-key", ALPACA_API_SECRET_KEY="synthetic-test-secret"))
    source.client.session.mount("https://", transport)
    return source


def test_cross_host_redirect_never_sends_a_destination_request(tmp_path: Path) -> None:
    transport = _SyntheticHttpAdapter(status=302)
    source = _source(transport)
    plan = _plan()
    try:
        with pytest.raises((ValueError, RuntimeError)):
            collect_news_receipts(root=tmp_path, plan=plan, expected_plan_sha256=news_collection_plan_sha256(plan),
                                  fetch_page=alpaca_news_receipt_fetcher(source, producer_revision=plan.producer_revision))
        assert len(transport.urls) == 1
        assert all(urlsplit(url).hostname == "data.alpaca.markets" for url in transport.urls)
        assert not list(tmp_path.rglob("result.json"))
    finally:
        source.client.session.close()


@pytest.mark.parametrize("body", [b'{"news":[1]}', b'[]', b'not-json', b'\xff'])
def test_malformed_provider_response_aborts_before_next_window(tmp_path: Path, body: bytes) -> None:
    transport = _SyntheticHttpAdapter(body=body)
    source = _source(transport)
    plan = _plan()
    try:
        with pytest.raises(RuntimeError):
            collect_news_receipts(root=tmp_path, plan=plan, expected_plan_sha256=news_collection_plan_sha256(plan),
                                  fetch_page=alpaca_news_receipt_fetcher(source, producer_revision=plan.producer_revision))
        assert len(transport.urls) == 1
        assert not list(tmp_path.rglob("result.json"))
    finally:
        source.client.session.close()


def test_real_transport_retries_are_one_logical_failure_and_other_window_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("market_predictor.sources.http.time.sleep", lambda _: None)
    transport = _SyntheticHttpAdapter(fail_symbol="MSFT")
    source = _source(transport)
    plan = _plan()
    try:
        report = collect_news_receipts(root=tmp_path, plan=plan, expected_plan_sha256=news_collection_plan_sha256(plan),
                                      fetch_page=alpaca_news_receipt_fetcher(source, producer_revision=plan.producer_revision))
        assert [window.status for window in report.windows] == ["transport_failure", "complete"]
        assert report.windows[0].attempts == 1
        assert len(transport.urls) == 4
    finally:
        source.client.session.close()
