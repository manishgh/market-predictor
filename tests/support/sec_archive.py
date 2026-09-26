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
from market_predictor.research import sec_acceptance_clock as clock
from market_predictor.research import sec_page_collection as runner
from market_predictor.sources.http import HttpByteResponse, HttpClient
from market_predictor.sources.sec import SecSource

USER_AGENT = "Market Predictor Tests sec-tests@marketpredictor.local"
_COLUMNS = ("accessionNumber", "filingDate", "reportDate", "acceptanceDateTime", "act", "form", "fileNumber", "filmNumber",
            "items", "core_type", "size", "isXBRL", "isInlineXBRL", "primaryDocument", "primaryDocDescription")


def settings() -> Settings:
    return Settings(SEC_USER_AGENT=USER_AGENT)


def filing(accession: str, form: str, accepted: str, *, items: str = "", report: str = "", primary: str = "doc.htm",
           size: int = 1000, description: str = "", filing_date: str | None = None) -> dict[str, Any]:
    """One EDGAR submissions row; `accepted` is the API's label as an ISO UTC instant (for an issuer that labels New
    York wall-clock time as UTC, that wall-clock time)."""
    moment = pd.Timestamp(accepted).tz_convert("UTC")
    date_ = filing_date or moment.tz_convert("America/New_York").strftime("%Y-%m-%d")
    return {"accessionNumber": accession, "filingDate": date_,
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


def header_page(accession: str, filing_date: str, accepted_new_york: str) -> bytes:
    """An EDGAR detail page reduced to the header fields the acceptance clock reads."""
    return (f'<html><div id="secNum"><strong>SEC Accession No.</strong> {accession}</div>'
            f'<div class="infoHead">Filing Date</div><div class="info">{filing_date}</div>'
            f'<div class="infoHead">Accepted</div><div class="info">{accepted_new_york}</div></html>').encode()


def page_response(url: str, status: int, body: bytes = b"", retry_after: str | None = None) -> HttpByteResponse:
    headers = (("content-type", "text/html"), *((("retry-after", retry_after),) if retry_after is not None else ()))
    return HttpByteResponse(body=body, requested_url=url, final_url=url, redirect_chain=(), status_code=status,
                            retrieved_at_utc=datetime.now(UTC), content_type="text/html", content_encoding=None, etag=None,
                            last_modified=None, body_length=len(body), sha256=hashlib.sha256(body).hexdigest(),
                            body_representation="http_entity_encoded", safe_headers=headers)


class PageFake:
    """Detail pages per accession: EDGAR's New York acceptance, a status code or a whole body served with 200;
    thread-safe for reads."""

    def __init__(self, pages: dict[str, tuple[str, str] | int | bytes]) -> None:
        self.pages = pages
        self.urls: list[str] = []

    def __call__(self, url: str) -> HttpByteResponse:
        self.urls.append(url)
        accession = url.rsplit("/", 1)[1].removesuffix("-index.htm")
        page = self.pages[accession]
        if isinstance(page, int):
            return page_response(url, page)
        if isinstance(page, bytes):
            return page_response(url, 200, page)
        return page_response(url, 200, header_page(accession, *page))

    def close(self) -> None:
        """No session to release."""


def true_pages(pages: dict[str, dict[str, Any]], new_york_labelled: frozenset[str] = frozenset(),
               statuses: dict[str, int] | None = None) -> dict[str, tuple[str, str] | int | bytes]:
    """EDGAR's own page for every submissions row: labels read as UTC, or as New York wall clock for listed CIKs."""
    rows: dict[str, tuple[str, str] | int | bytes] = {}
    for url, payload in pages.items():
        owner = url.split("CIK")[-1][:10]
        section = payload["filings"]["recent"] if "filings" in payload else payload
        for accession, label, filed in zip(section["accessionNumber"], section["acceptanceDateTime"], section["filingDate"],
                                           strict=True):
            wall = pd.Timestamp(label[:19])
            local = wall if owner in new_york_labelled else wall.tz_localize("UTC").tz_convert("America/New_York")
            rows[accession] = (filed, local.strftime("%Y-%m-%d %H:%M:%S"))
    rows.update(statuses or {})
    return rows


def collect_clock_pages(root: Path, collection: Path, fake: PageFake, monkeypatch: Any, name: str,
                        resume_checkpoint_sha256: str | None = None) -> dict[str, Any]:
    """Collect clock pages through the real runner over `fake`."""
    monkeypatch.setattr(runner, "sec_fetch", lambda _settings, _stop: fake)
    monkeypatch.setattr(runner, "_guard", lambda: None)
    monkeypatch.setattr(runner.documents, "RETRY_WAITS_SECONDS", (0.0, 0.0))
    monkeypatch.setattr(clock, "_guard", lambda: None)
    monkeypatch.setattr(clock, "Settings", settings)
    return clock.collect_sec_clock_pages(root=root, collection=collection_pin(root, collection),
                                         output=root / "data/raw" / f"{name}_pages",
                                         resume_checkpoint_sha256=resume_checkpoint_sha256)


def collection_pin(root: Path, collection: Path) -> dict[str, str]:
    authority = collection / "_authority.json"
    return {"path": authority.relative_to(root).as_posix(), "sha256": _sha256(authority)}


def write_clock(root: Path, collection: Path, fake: PageFake, monkeypatch: Any, name: str = "sec_clock") -> dict[str, str]:
    """Collect clock pages over `fake`, publish the clock and return its pin."""
    pages = collect_clock_pages(root, collection, fake, monkeypatch, name)
    if pages["status"] != "complete":
        raise RuntimeError(f"test clock pages did not complete: {pages}")
    published = clock.publish_sec_acceptance_clock(
        root=root, collection=collection_pin(root, collection),
        pages={"path": f"data/raw/{name}_pages/_manifest.json", "sha256": pages["manifest_sha256"]},
        output=root / "data/research" / name)
    return {"path": f"data/research/{name}/_manifest.json", "sha256": published["manifest_sha256"]}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
