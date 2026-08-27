from __future__ import annotations

from datetime import UTC, datetime

from market_predictor.catalysts.global_events.collection import (
    GLOBAL_MARKET_EVENT_QUERIES,
)
from market_predictor.commands.market_context import (
    _market_context_events,
    collect_gdelt_market_context_events,
)
from market_predictor.sources.gdelt import (
    GdeltDocumentRequest,
    GdeltDocumentResult,
)

START = datetime(2026, 7, 8, tzinfo=UTC)
END = datetime(2026, 7, 9, tzinfo=UTC)


def test_command_adapter_maps_documents_and_preserves_query_metadata() -> None:
    captured: list[GdeltDocumentRequest] = []

    def fetch(request: GdeltDocumentRequest) -> GdeltDocumentResult:
        captured.append(request)
        return GdeltDocumentResult(
            records=[
                {
                    "title": "Hormuz blockade threat raises oil tanker risk",
                    "url": "https://example.com/hormuz",
                    "seendate": "20260708T081500Z",
                    "domain": "example.com",
                    "language": "English",
                    "sourceCountry": "US",
                    "socialimage": "https://example.com/image.jpg",
                    "collection_query": GLOBAL_MARKET_EVENT_QUERIES[0],
                }
            ],
            complete=True,
            completed_queries=GLOBAL_MARKET_EVENT_QUERIES,
        )

    events, errors = collect_gdelt_market_context_events(
        START,
        end=END,
        max_records_per_query=25,
        fetch=fetch,
    )

    assert not errors
    assert captured == [
        GdeltDocumentRequest(
            queries=GLOBAL_MARKET_EVENT_QUERIES,
            requested_start_utc=START,
            requested_end_utc=END,
            max_records=25,
            timeout_seconds=30.0,
        )
    ]
    assert len(events) == 1
    event = events[0]
    assert event.ticker == "MARKET"
    assert event.source == "gdelt:doc"
    assert event.raw["query"] == GLOBAL_MARKET_EVENT_QUERIES[0]
    assert event.raw["source_country"] == "US"


def test_command_adapter_skips_invalid_documents_and_deduplicates() -> None:
    valid = {
        "title": "Taiwan Strait blockade risk hits semiconductors",
        "url": "https://example.com/taiwan",
        "seendate": "20260708T081500Z",
        "collection_query": GLOBAL_MARKET_EVENT_QUERIES[1],
    }
    events = _market_context_events(
        [
            valid,
            dict(valid),
            {**valid, "title": ""},
            {**valid, "url": ""},
            {**valid, "seendate": ""},
        ]
    )

    assert len(events) == 1
    assert events[0].title == valid["title"]


def test_command_adapter_reports_failed_or_incomplete_collection() -> None:
    def failed_fetch(_request: GdeltDocumentRequest) -> GdeltDocumentResult:
        raise RuntimeError("rate limited")

    events, errors = collect_gdelt_market_context_events(
        START,
        end=END,
        fetch=failed_fetch,
    )
    assert not events
    assert errors == ["rate limited"]

    events, errors = collect_gdelt_market_context_events(
        START,
        end=END,
        fetch=lambda _request: GdeltDocumentResult(
            records=[],
            complete=False,
            completed_queries=GLOBAL_MARKET_EVENT_QUERIES,
        ),
    )
    assert not events
    assert errors == ["GDELT fetch was partial or truncated"]
