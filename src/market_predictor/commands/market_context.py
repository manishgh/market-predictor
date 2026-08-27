"""Command adapter for collecting GDELT market-context events."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

import pandas as pd

from market_predictor.catalysts.global_events.collection import (
    GLOBAL_MARKET_EVENT_QUERIES,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.schemas import NewsEvent
from market_predictor.sources.gdelt import (
    GdeltDocumentFetcher,
    GdeltDocumentRequest,
    fetch_gdelt_documents,
    validate_gdelt_document_request,
)


def collect_gdelt_market_context_events(
    start: datetime,
    *,
    end: datetime | None = None,
    max_records_per_query: int = 75,
    fetch: GdeltDocumentFetcher = fetch_gdelt_documents,
) -> tuple[list[NewsEvent], list[str]]:
    """Collect legacy command output through the canonical GDELT transport."""

    request = validate_gdelt_document_request(
        GdeltDocumentRequest(
            queries=GLOBAL_MARKET_EVENT_QUERIES,
            requested_start_utc=start,
            requested_end_utc=end or datetime.now(UTC),
            max_records=max_records_per_query,
        )
    )
    try:
        result = fetch(request)
        if result.errors:
            raise DataReadinessError("GDELT fetch reported errors: " + "; ".join(result.errors))
        if not result.complete:
            raise DataReadinessError("GDELT fetch was partial or truncated")
        if result.completed_queries != request.queries:
            raise DataReadinessError("GDELT fetch did not complete the requested queries")
        return _market_context_events(result.records), []
    except Exception as exc:
        return [], [str(exc)]


def _market_context_events(records: Sequence[Mapping[str, object]]) -> list[NewsEvent]:
    events: list[NewsEvent] = []
    for record in records:
        title = str(record.get("title") or "").strip()
        url = str(record.get("url") or "").strip()
        timestamp = _parse_gdelt_timestamp(
            record.get("seendate") or record.get("datetime")
        )
        if not title or not url or timestamp is None:
            continue
        events.append(
            NewsEvent(
                ticker="MARKET",
                timestamp=timestamp,
                source="gdelt:doc",
                title=title,
                url=url,
                summary=str(record.get("snippet") or record.get("seendate") or ""),
                text=title,
                raw={
                    "query": str(record.get("collection_query") or ""),
                    "domain": str(record.get("domain") or "").strip(),
                    "language": str(record.get("language") or "").strip(),
                    "source_country": str(record.get("sourceCountry") or "").strip(),
                    "image": record.get("socialimage"),
                },
            )
        )
    deduplicated = {
        (event.title.strip().lower(), str(event.url or "").strip().lower(), event.timestamp.isoformat()): event
        for event in events
    }
    return sorted(deduplicated.values(), key=lambda event: event.timestamp)


def _parse_gdelt_timestamp(value: object) -> datetime | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if not isinstance(parsed, pd.Timestamp) or pd.isna(parsed):
        return None
    converted = parsed.to_pydatetime()
    return converted if isinstance(converted, datetime) else None
