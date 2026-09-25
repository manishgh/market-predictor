"""EDGAR-shaped SEC archives written by the real collector through the real `SecSource` parser."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.catalysts.sec_filings.collection import (
    SecFilingCollection,
    SecFilingCollectionConfig,
    collect_historical_sec_filings,
)
from market_predictor.config import Settings
from market_predictor.sources.http import HttpByteResponse, HttpClient
from market_predictor.sources.sec import SecSource

USER_AGENT = "Market Predictor Tests sec-tests@marketpredictor.local"
_COLUMNS = ("accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "act", "form", "fileNumber", "filmNumber",
            "items", "core_type", "size", "isXBRL", "isInlineXBRL", "primaryDocument", "primaryDocDescription")


def settings() -> Settings:
    return Settings(SEC_USER_AGENT=USER_AGENT)


def filing(accession: str, form: str, accepted: str, *, items: str = "", report: str = "", primary: str = "doc.htm",
           size: int = 1000, description: str = "") -> dict[str, Any]:
    """One EDGAR submissions row; `accepted` is an ISO UTC instant."""
    moment = pd.Timestamp(accepted).tz_convert("UTC")
    return {"accessionNumber": accession, "filingDate": moment.tz_convert("America/New_York").strftime("%Y-%m-%d"),
            "reportDate": report, "acceptanceDateTime": moment.strftime("%Y-%m-%dT%H:%M:%S.000Z"), "act": "34",
            "form": form, "fileNumber": "001-00001", "filmNumber": "1", "items": items, "core_type": form, "size": size,
            "isXBRL": 0, "isInlineXBRL": 0, "primaryDocument": primary, "primaryDocDescription": description or form}


def _columns(rows: list[dict[str, Any]]) -> dict[str, list[Any]]:
    return {column: [row[column] for row in rows] for column in _COLUMNS}


def submissions(cik: str, name: str, recent: list[dict[str, Any]], *,
                older: tuple[tuple[str, list[dict[str, Any]]], ...] = ()) -> dict[str, dict[str, Any]]:
    """URL to payload for one issuer: the root listing and each older page named in its `files`."""
    files = [{"name": page, "filingCount": len(rows), "filingFrom": min(row["filingDate"] for row in rows),
              "filingTo": max(row["filingDate"] for row in rows)} for page, rows in older]
    pages: dict[str, dict[str, Any]] = {f"https://data.sec.gov/submissions/CIK{cik}.json":
             {"cik": str(int(cik)), "name": name, "filings": {"recent": _columns(recent), "files": files}}}
    pages.update({f"https://data.sec.gov/submissions/{page}": _columns(rows) for page, rows in older})
    return pages


class EdgarFake(HttpClient):
    """Serves fixed EDGAR JSON bodies with deterministic retrieval clocks; no network."""

    def __init__(self, pages: dict[str, dict[str, Any]]) -> None:
        super().__init__(user_agent=USER_AGENT)
        self.pages = pages
        self.urls: list[str] = []
        self.clock = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)

    def get_bytes_with_metadata(
        self, url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, retries: int = 3,
        pause: float = 1.0, maximum_body_bytes: int = 0, allow_redirects: bool = True, raise_for_status: bool = True,
    ) -> HttpByteResponse:
        self.urls.append(url)
        if url not in self.pages:
            raise RuntimeError(f"test EDGAR page is not defined: {url}")
        body = json.dumps(self.pages[url], sort_keys=True).encode()
        self.clock += timedelta(seconds=1)
        return HttpByteResponse(body=body, requested_url=url, final_url=url, redirect_chain=(), status_code=200,
                                retrieved_at_utc=self.clock, content_type="application/json", content_encoding=None,
                                etag=None, last_modified=None, body_length=len(body), sha256=hashlib.sha256(body).hexdigest(),
                                body_representation="http_entity_encoded",
                                safe_headers=(("content-type", "application/json"),))


class _Clock:
    def __init__(self) -> None:
        self.value = datetime(2026, 8, 2, 12, 0, tzinfo=UTC)

    def __call__(self) -> datetime:
        self.value += timedelta(seconds=1)
        return self.value


def write_collection(directory: Path, pages: dict[str, dict[str, Any]], relations: pd.DataFrame,
                     forms: tuple[str, ...]) -> SecFilingCollection:
    """Collect every relation CIK with the real collector over the fake EDGAR pages."""
    fake = EdgarFake(pages)
    return collect_historical_sec_filings(
        relations, directory, source_factory=lambda: SecSource(settings(), client=fake),
        config=SecFilingCollectionConfig(start_date=date(2019, 7, 9), end_date=date(2026, 7, 8), forms=forms, max_workers=1),
        clock=_Clock())
