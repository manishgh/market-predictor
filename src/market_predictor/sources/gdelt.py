from __future__ import annotations

from datetime import datetime, timezone
import time
from typing import Any

import pandas as pd

from market_predictor.schemas import NewsEvent
from market_predictor.sources.http import HttpClient


GDELT_DOC_ENDPOINT = "https://api.gdeltproject.org/api/v2/doc/doc"

DEFAULT_GDELT_CONTEXT_QUERIES = (
    '("strait of hormuz" OR hormuz OR "persian gulf" OR "red sea" OR "suez canal" OR "bab el-mandeb") '
    '(oil OR tanker OR shipping OR blockade OR missile OR attack OR disruption)',
    '("taiwan strait" OR taiwan OR tsmc OR "south china sea") '
    '("military drill" OR blockade OR invasion OR missile OR "export control" OR sanction)',
    '("russia" OR "ukraine" OR "black sea" OR nato) '
    '(missile OR drone OR sanction OR pipeline OR lng OR grain OR wheat)',
    '("rare earth" OR gallium OR germanium OR lithium OR cobalt OR graphite OR "critical minerals") '
    '("export control" OR ban OR restriction OR quota OR sanction OR tariff)',
    '(cyberattack OR ransomware OR "critical infrastructure" OR "power grid" OR "pipeline hack") '
    '(outage OR shutdown OR breach OR malware)',
)


class GdeltSource:
    """GDELT DOC 2.0 source adapter for global market-context news."""

    def __init__(
        self,
        client: HttpClient | None = None,
        endpoint: str = GDELT_DOC_ENDPOINT,
        request_pause_seconds: float = 5.5,
        request_retries: int = 2,
    ) -> None:
        self.client = client or HttpClient(user_agent="market-predictor/0.1 gdelt")
        self.endpoint = endpoint
        self.request_pause_seconds = request_pause_seconds
        self.request_retries = request_retries

    def fetch_context_events(
        self,
        start: datetime,
        *,
        end: datetime | None = None,
        queries: tuple[str, ...] = DEFAULT_GDELT_CONTEXT_QUERIES,
        max_records_per_query: int = 75,
    ) -> list[NewsEvent]:
        events, errors = self.fetch_context_events_with_errors(
            start,
            end=end,
            queries=queries,
            max_records_per_query=max_records_per_query,
        )
        if errors and not events:
            raise RuntimeError(" | ".join(errors))
        return events

    def fetch_context_events_with_errors(
        self,
        start: datetime,
        *,
        end: datetime | None = None,
        queries: tuple[str, ...] = DEFAULT_GDELT_CONTEXT_QUERIES,
        max_records_per_query: int = 75,
    ) -> tuple[list[NewsEvent], list[str]]:
        end = end or datetime.now(timezone.utc)
        events: list[NewsEvent] = []
        errors: list[str] = []
        for index, query in enumerate(queries):
            try:
                payload = self.client.get_json(
                    self.endpoint,
                    params={
                        "query": query,
                        "mode": "ArtList",
                        "format": "json",
                        "maxrecords": str(max_records_per_query),
                        "sort": "HybridRel",
                        "startdatetime": _gdelt_datetime(start),
                        "enddatetime": _gdelt_datetime(end),
                    },
                    retries=self.request_retries,
                    pause=self.request_pause_seconds,
                )
                events.extend(self.events_from_payload(payload, query=query))
            except Exception as exc:
                errors.append(f"query={query[:80]} error={exc}")
            if index < len(queries) - 1:
                time.sleep(self.request_pause_seconds)
        return _dedupe_events(events), errors

    @staticmethod
    def events_from_payload(payload: dict[str, Any], *, query: str) -> list[NewsEvent]:
        rows = payload.get("articles", []) if isinstance(payload, dict) else []
        events: list[NewsEvent] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            url = str(row.get("url") or "").strip()
            if not title or not url:
                continue
            timestamp = _parse_gdelt_timestamp(row.get("seendate") or row.get("datetime"))
            if timestamp is None:
                continue
            domain = str(row.get("domain") or "").strip()
            language = str(row.get("language") or "").strip()
            source_country = str(row.get("sourceCountry") or "").strip()
            events.append(
                NewsEvent(
                    ticker="MARKET",
                    timestamp=timestamp,
                    source="gdelt:doc",
                    title=title,
                    url=url,
                    summary=str(row.get("snippet") or row.get("seendate") or ""),
                    text=title,
                    raw={
                        "query": query,
                        "domain": domain,
                        "language": language,
                        "source_country": source_country,
                        "image": row.get("socialimage"),
                    },
                )
            )
        return events


def _gdelt_datetime(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


def _parse_gdelt_timestamp(value: Any) -> datetime | None:
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _dedupe_events(events: list[NewsEvent]) -> list[NewsEvent]:
    deduped: dict[tuple[str, str, str], NewsEvent] = {}
    for event in events:
        key = (event.title.strip().lower(), str(event.url or "").strip().lower(), event.timestamp.isoformat())
        deduped[key] = event
    return sorted(deduped.values(), key=lambda event: event.timestamp)
