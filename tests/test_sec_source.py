from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256

import pandas as pd
import pytest

from market_predictor.config import Settings
from market_predictor.sources.http import HttpByteResponse
from market_predictor.sources.sec import (
    SecRequestGovernor,
    SecSource,
    SecSourceResponseError,
    validate_sec_user_agent,
)


def _recent(*, form: str = "8-K", accession: str = "0000789019-26-000001") -> dict[str, list[str]]:
    return {
        "form": [form],
        "acceptanceDateTime": ["2026-07-21T16:05:00"],
        "filingDate": ["2026-07-21"],
        "reportDate": ["2026-07-21"],
        "accessionNumber": [accession],
        "primaryDocument": ["msft-8k.htm"],
        "fileNumber": ["001-37845"],
    }


class _FakeSecClient:
    def __init__(self, *, malformed_historical_count: bool = False, amendment: bool = False) -> None:
        self.urls: list[str] = []
        self.malformed_historical_count = malformed_historical_count
        self.amendment = amendment

    def get_json(self, url: str, **_: object) -> dict[str, object]:
        self.urls.append(url)
        return {"0": {"ticker": "MSFT", "cik_str": 789019, "title": "MICROSOFT CORP"}}

    def get_bytes_with_metadata(self, url: str, **_: object) -> HttpByteResponse:
        self.urls.append(url)
        if url.endswith("CIK0000789019-submissions-001.json"):
            payload: dict[str, object] = {
                "form": ["10-Q"],
                "acceptanceDateTime": ["20200721160500"],
                "filingDate": ["2020-07-21"],
                "reportDate": ["2020-06-30"],
                "accessionNumber": ["0000789019-20-000001"],
                "primaryDocument": ["msft-10q.htm"],
                "fileNumber": ["001-37845"],
            }
            if self.malformed_historical_count:
                payload["form"] = []
        else:
            recent = _recent()
            if self.amendment:
                recent = {key: [*values, values[0]] for key, values in recent.items()}
                recent["form"][1] = "8-K/A"
                recent["acceptanceDateTime"][1] = "2026-07-21T16:10:00"
                recent["accessionNumber"][1] = "0000789019-26-000002"
            payload = {
                "cik": "789019",
                "name": "MICROSOFT CORP",
                "filings": {
                    "recent": recent,
                    "files": [
                        {
                            "name": "CIK0000789019-submissions-001.json",
                            "filingCount": 1,
                            "filingFrom": "2020-01-01",
                            "filingTo": "2020-12-31",
                        },
                        {
                            "name": "CIK0000789019-submissions-002.json",
                            "filingCount": 1,
                            "filingFrom": "2010-01-01",
                            "filingTo": "2010-12-31",
                        },
                    ],
                },
            }
        body = json.dumps(payload, sort_keys=True).encode("utf-8")
        return HttpByteResponse(
            body=body,
            requested_url=url,
            final_url=url,
            redirect_chain=(),
            status_code=200,
            retrieved_at_utc=datetime(2026, 8, 2, 10, len(self.urls), tzinfo=UTC),
            content_type="application/json",
            content_encoding="identity",
            etag='"fixture"',
            last_modified="Sun, 02 Aug 2026 10:00:00 GMT",
            body_length=len(body),
            sha256=sha256(body).hexdigest(),
            body_representation="http_entity_encoded",
            safe_headers=(("content-type", "application/json"),),
        )


def _source(client: _FakeSecClient) -> SecSource:
    return SecSource(
        Settings(SEC_USER_AGENT="Market Predictor Tests sec-tests@marketpredictor.local"),
        client=client,  # type: ignore[arg-type]
    )


def test_acceptance_datetime_is_interpreted_as_eastern_clock_time() -> None:
    source = _source(_FakeSecClient())
    events = source.fetch_filings(
        "MSFT",
        datetime(2026, 7, 21, 0, 0, tzinfo=UTC),
        end=datetime(2026, 7, 22, 0, 0, tzinfo=UTC),
    )
    assert len(events) == 1
    assert pd.Timestamp(events[0].timestamp) == pd.Timestamp("2026-07-21T20:05:00Z")


def test_historical_files_are_followed_once_only_when_overlapping() -> None:
    client = _FakeSecClient()
    history = _source(client).fetch_filing_history(
        "MSFT",
        datetime(2020, 1, 1, tzinfo=UTC),
        datetime(2020, 12, 31, 23, 59, tzinfo=UTC),
    )
    assert [filing.form for filing in history.filings] == ["10-Q"]
    assert pd.Timestamp(history.filings[0].accepted_at_utc) == pd.Timestamp("2020-07-21T20:05:00Z")
    assert len(history.raw_responses) == 2
    assert history.source_row_count == 2
    assert sum(url.endswith("CIK0000789019-submissions-001.json") for url in client.urls) == 1
    assert not any(url.endswith("CIK0000789019-submissions-002.json") for url in client.urls)


def test_malformed_historical_count_is_unknown_and_preserves_raw_responses() -> None:
    with pytest.raises(SecSourceResponseError) as caught:
        _source(_FakeSecClient(malformed_historical_count=True)).fetch_cik_filing_history(
            "0000789019",
            datetime(2020, 1, 1, tzinfo=UTC),
            datetime(2020, 12, 31, 23, 59, tzinfo=UTC),
            forms={"10-Q"},
            ticker_hint="MSFT",
        )
    assert len(caught.value.raw_responses) == 2
    assert "inconsistent lengths" in str(caught.value)


def test_amendment_parent_is_linked_when_unique() -> None:
    history = _source(_FakeSecClient(amendment=True)).fetch_cik_filing_history(
        "0000789019",
        datetime(2026, 7, 21, tzinfo=UTC),
        datetime(2026, 7, 22, tzinfo=UTC),
        ticker_hint="MSFT",
    )
    assert history.filings[1].is_amendment
    assert history.filings[1].amends_accession_number == history.filings[0].accession_number


@pytest.mark.parametrize(
    "value",
    ["market-predictor/0.1 contact@example.com", "your-email@company.test", "no-email-identity"],
)
def test_sec_user_agent_rejects_placeholders(value: str) -> None:
    with pytest.raises(ValueError, match="SEC_USER_AGENT"):
        validate_sec_user_agent(value)


def test_process_governor_serializes_requests_and_centralizes_cooldown() -> None:
    now = [0.0]
    sleeps: list[float] = []

    def monotonic() -> float:
        return now[0]

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    governor = SecRequestGovernor(
        requests_per_second=5,
        forbidden_cooldown_seconds=10,
        rate_limit_cooldown_seconds=2,
        monotonic=monotonic,
        sleeper=sleep,
    )
    governor.acquire()
    governor.acquire()
    governor.observe_response(429, {"Retry-After": "3"})
    governor.acquire()

    assert sleeps == pytest.approx([0.2, 3.0])


def test_governor_rejects_sec_limit_or_higher() -> None:
    with pytest.raises(ValueError, match="below 10"):
        SecRequestGovernor(requests_per_second=10)
