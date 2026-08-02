from __future__ import annotations

import gzip
import hashlib
import json
import re
import threading
import time
import zlib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from email.utils import parsedate_to_datetime
from typing import Any, cast
from zoneinfo import ZoneInfo

import pandas as pd

from market_predictor.config import Settings
from market_predictor.schemas import NewsEvent
from market_predictor.sources.http import HttpByteResponse, HttpClient

_MAX_SEC_RESPONSE_BYTES = 64 * 1024 * 1024
_SEC_USER_AGENT_EMAIL = re.compile(r"\b[^\s@]+@[^\s@]+\.[^\s@]+\b")
_SEC_PLACEHOLDERS = ("example.com", "example.org", "your-email", "contact@", "changeme")


@dataclass(frozen=True, slots=True)
class SecRawResponse:
    response_id: str
    requested_url: str
    final_url: str
    status_code: int
    retrieved_at_utc: datetime
    content_type: str | None
    content_encoding: str | None
    etag: str | None
    last_modified: str | None
    body: bytes
    body_sha256: str
    body_length: int
    safe_headers: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class SecFilingRecord:
    ticker: str
    cik: str
    company_name: str
    form: str
    accepted_at_utc: datetime
    filing_date: str
    report_date: str
    accession_number: str
    primary_document: str
    document_url: str
    submission_file: str
    file_number: str
    is_amendment: bool
    amends_accession_number: str | None
    raw_sha256: str


@dataclass(frozen=True, slots=True)
class SecFilingHistory:
    ticker: str
    cik: str
    company_name: str
    filings: tuple[SecFilingRecord, ...]
    submission_files: tuple[str, ...]
    response_sha256: str
    raw_responses: tuple[SecRawResponse, ...]
    source_row_count: int


class SecSourceResponseError(RuntimeError):
    """SEC response or schema failure retaining every response already observed."""

    def __init__(self, message: str, raw_responses: tuple[SecRawResponse, ...]) -> None:
        super().__init__(message)
        self.raw_responses = raw_responses


class SecRequestGovernor:
    """Process-wide SEC request pacing and fair-access cooldown state."""

    def __init__(
        self,
        *,
        requests_per_second: float = 6.0,
        forbidden_cooldown_seconds: float = 600.0,
        rate_limit_cooldown_seconds: float = 60.0,
        monotonic: Any = time.monotonic,
        sleeper: Any = time.sleep,
    ) -> None:
        if requests_per_second <= 0 or requests_per_second >= 10:
            raise ValueError("SEC requests_per_second must be greater than zero and below 10")
        if forbidden_cooldown_seconds <= 0 or rate_limit_cooldown_seconds <= 0:
            raise ValueError("SEC cooldowns must be positive")
        self._interval = 1.0 / requests_per_second
        self._forbidden_cooldown = forbidden_cooldown_seconds
        self._rate_limit_cooldown = rate_limit_cooldown_seconds
        self._monotonic = monotonic
        self._sleep = sleeper
        self._lock = threading.Lock()
        self._next_request_at = 0.0
        self._blocked_until = 0.0

    def acquire(self) -> None:
        while True:
            with self._lock:
                now = float(self._monotonic())
                target = max(self._next_request_at, self._blocked_until)
                if now >= target:
                    self._next_request_at = now + self._interval
                    return
                delay = target - now
            self._sleep(delay)

    def observe_response(self, status_code: int, headers: Mapping[str, str]) -> None:
        if status_code not in {403, 429}:
            return
        fallback = self._forbidden_cooldown if status_code == 403 else self._rate_limit_cooldown
        delay = _retry_after_seconds(headers.get("Retry-After"), fallback=fallback)
        with self._lock:
            self._blocked_until = max(
                self._blocked_until,
                float(self._monotonic()) + delay,
            )


_PROCESS_SEC_GOVERNOR = SecRequestGovernor()


def validate_sec_user_agent(value: str) -> str:
    normalized = " ".join(value.strip().split())
    lowered = normalized.lower()
    if len(normalized) < 12 or _SEC_USER_AGENT_EMAIL.search(normalized) is None or any(token in lowered for token in _SEC_PLACEHOLDERS):
        raise ValueError("SEC_USER_AGENT must identify a real organization and monitored email address")
    return normalized


class SecSource:
    ticker_map_url = "https://www.sec.gov/files/company_tickers.json"
    submissions_url = "https://data.sec.gov/submissions/CIK{cik}.json"
    historical_submission_url = "https://data.sec.gov/submissions/{name}"

    def __init__(
        self,
        settings: Settings,
        *,
        governor: SecRequestGovernor | None = None,
        client: HttpClient | None = None,
    ) -> None:
        user_agent = validate_sec_user_agent(settings.sec_user_agent)
        self.governor = governor or _PROCESS_SEC_GOVERNOR
        self.client = client or HttpClient(
            user_agent=user_agent,
            before_request=self.governor.acquire,
            after_response=self.governor.observe_response,
            additional_retriable_statuses=frozenset({403}),
        )
        self._ticker_map: dict[str, tuple[str, str]] | None = None
        self._raw_response_sink: list[SecRawResponse] | None = None

    def cik_for_ticker(self, ticker: str) -> str:
        return self.identity_for_ticker(ticker)[0]

    def identity_for_ticker(self, ticker: str) -> tuple[str, str]:
        ticker_upper = _normalized_symbol(ticker)
        if self._ticker_map is None:
            payload = self.client.get_json(self.ticker_map_url)
            if not isinstance(payload, Mapping):
                raise RuntimeError("SEC company ticker map is malformed")
            mapping: dict[str, tuple[str, str]] = {}
            for item in payload.values():
                if not isinstance(item, Mapping):
                    continue
                symbol = _normalized_symbol(str(item.get("ticker", "")))
                cik = str(item.get("cik_str", "")).strip()
                if symbol and cik.isdigit():
                    mapping[symbol] = (cik.zfill(10), str(item.get("title", "")).strip())
            if not mapping:
                raise RuntimeError("SEC company ticker map contains no valid identities")
            self._ticker_map = mapping
        try:
            return self._ticker_map[ticker_upper]
        except KeyError as exc:
            raise ValueError(f"CIK not found for ticker {ticker_upper}") from exc

    def fetch_cik_filing_history(
        self,
        cik: str,
        start: datetime,
        end: datetime,
        *,
        forms: set[str] | None = None,
        ticker_hint: str = "SEC",
    ) -> SecFilingHistory:
        if self._raw_response_sink is not None:
            raise RuntimeError("SEC source does not support concurrent history calls")
        self._raw_response_sink = []
        try:
            return self._fetch_cik_filing_history(
                cik,
                start,
                end,
                forms=forms,
                ticker_hint=ticker_hint,
            )
        except SecSourceResponseError:
            raise
        except Exception as exc:
            raise SecSourceResponseError(str(exc), tuple(self._raw_response_sink)) from exc
        finally:
            self._raw_response_sink = None

    def _fetch_cik_filing_history(
        self,
        cik: str,
        start: datetime,
        end: datetime,
        *,
        forms: set[str] | None = None,
        ticker_hint: str = "SEC",
    ) -> SecFilingHistory:
        start_utc = _require_utc(start, "start")
        end_utc = _require_utc(end, "end")
        if end_utc < start_utc:
            raise ValueError("SEC filing window is reversed")
        normalized_cik = _normalized_cik(cik)
        normalized_ticker = _normalized_symbol(ticker_hint)
        root_name = f"CIK{normalized_cik}.json"
        root_payload, root_response = self._get_archived_json(self.submissions_url.format(cik=normalized_cik))
        payload_cik = str(root_payload.get("cik", "")).strip()
        if payload_cik and _normalized_cik(payload_cik) != normalized_cik:
            raise RuntimeError("SEC submissions payload CIK conflicts with request")
        company_name = str(root_payload.get("name", "")).strip()
        if not company_name:
            raise RuntimeError("SEC submissions payload has no issuer name")
        filings = root_payload.get("filings")
        if not isinstance(filings, Mapping):
            raise RuntimeError("SEC submissions payload has no filings object")
        recent = filings.get("recent")
        if not isinstance(recent, Mapping):
            raise RuntimeError("SEC submissions payload has no filings.recent object")
        descriptors = filings.get("files")
        if not isinstance(descriptors, list):
            raise RuntimeError("SEC submissions payload has no filings.files array")

        payloads: list[tuple[str, Mapping[str, object], int | None]] = [(root_name, cast(Mapping[str, object], recent), None)]
        raw_responses = [root_response]
        for descriptor in descriptors:
            validated = _validated_descriptor(descriptor)
            if not _descriptor_overlaps(validated, start_utc.date(), end_utc.date()):
                continue
            name = str(validated["name"])
            payload, response = self._get_archived_json(self.historical_submission_url.format(name=name))
            section: object = payload.get("filings", payload)
            if isinstance(section, Mapping) and "recent" in section:
                section = section.get("recent")
            if not isinstance(section, Mapping):
                raise RuntimeError(f"SEC historical submissions payload is malformed: {name}")
            payloads.append(
                (
                    name,
                    cast(Mapping[str, object], section),
                    int(cast(int, validated["filingCount"])),
                )
            )
            raw_responses.append(response)

        wanted = {value.strip().upper() for value in forms or set() if value.strip()}
        records: dict[str, SecFilingRecord] = {}
        source_row_count = 0
        for source_name, section, expected_count in payloads:
            rows = _rows(section, expected_count=expected_count, source_name=source_name)
            source_row_count += len(rows)
            for row in rows:
                record = _filing_record(
                    row,
                    ticker=normalized_ticker,
                    cik=normalized_cik,
                    company_name=company_name,
                    submission_file=source_name,
                )
                if record.accepted_at_utc < start_utc or record.accepted_at_utc > end_utc:
                    continue
                if wanted and record.form not in wanted:
                    continue
                existing = records.get(record.accession_number)
                if existing is not None and _filing_identity(existing) != _filing_identity(record):
                    raise RuntimeError(f"SEC accession has conflicting metadata: {record.accession_number}")
                records[record.accession_number] = record
        ordered = _link_amendments(tuple(sorted(records.values(), key=lambda item: (item.accepted_at_utc, item.accession_number))))
        return SecFilingHistory(
            ticker=normalized_ticker,
            cik=normalized_cik,
            company_name=company_name,
            filings=ordered,
            submission_files=tuple(name for name, _, _ in payloads),
            response_sha256=_json_sha256(
                [
                    {
                        "response_id": response.response_id,
                        "url": response.final_url,
                        "sha256": response.body_sha256,
                    }
                    for response in raw_responses
                ]
            ),
            raw_responses=tuple(raw_responses),
            source_row_count=source_row_count,
        )

    def fetch_filing_history(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        *,
        forms: set[str] | None = None,
    ) -> SecFilingHistory:
        ticker_upper = _normalized_symbol(ticker)
        cik, _ = self.identity_for_ticker(ticker_upper)
        return self.fetch_cik_filing_history(
            cik,
            start,
            end,
            forms=forms,
            ticker_hint=ticker_upper,
        )

    def fetch_filings(
        self,
        ticker: str,
        start: datetime,
        end: datetime | None = None,
        *,
        forms: set[str] | None = None,
        limit: int = 100,
    ) -> list[NewsEvent]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        history = self.fetch_filing_history(ticker, start, end or datetime.now(UTC), forms=forms)
        events: list[NewsEvent] = []
        for filing in reversed(history.filings[-limit:]):
            title = f"{filing.ticker} SEC {filing.form}"
            summary = f"SEC filing {filing.form}, filed {filing.filing_date}"
            events.append(
                NewsEvent(
                    ticker=filing.ticker,
                    timestamp=filing.accepted_at_utc,
                    source=f"sec:{filing.form.lower()}",
                    title=title,
                    url=filing.document_url,
                    summary=summary,
                    text=f"{title}. {summary}.",
                    raw={
                        "cik": filing.cik,
                        "form": filing.form,
                        "accession_number": filing.accession_number,
                        "amends_accession_number": filing.amends_accession_number,
                        "raw_sha256": filing.raw_sha256,
                    },
                )
            )
        return events

    def _get_archived_json(self, url: str) -> tuple[Mapping[str, object], SecRawResponse]:
        response = self.client.get_bytes_with_metadata(
            url,
            headers={"Accept": "application/json"},
            maximum_body_bytes=_MAX_SEC_RESPONSE_BYTES,
        )
        raw = SecRawResponse(
            response_id=_json_sha256(
                {
                    "requested_url": response.requested_url,
                    "final_url": response.final_url,
                    "retrieved_at_utc": response.retrieved_at_utc.isoformat(),
                    "sha256": response.sha256,
                }
            ),
            requested_url=response.requested_url,
            final_url=response.final_url,
            status_code=response.status_code,
            retrieved_at_utc=response.retrieved_at_utc,
            content_type=response.content_type,
            content_encoding=response.content_encoding,
            etag=response.etag,
            last_modified=response.last_modified,
            body=response.body,
            body_sha256=response.sha256,
            body_length=response.body_length,
            safe_headers=response.safe_headers,
        )
        if self._raw_response_sink is not None:
            self._raw_response_sink.append(raw)
        decoded = _decoded_body(response)
        try:
            payload = json.loads(decoded.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"SEC response is not valid UTF-8 JSON: {url}") from exc
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"SEC response must contain a JSON object: {url}")
        return cast(Mapping[str, object], payload), raw

    @staticmethod
    def _acceptance_time_utc(value: object) -> pd.Timestamp:
        return _acceptance_time_utc(value)


def _rows(
    section: Mapping[str, object],
    *,
    expected_count: int | None,
    source_name: str,
) -> list[dict[str, object]]:
    required = {
        "accessionNumber",
        "filingDate",
        "reportDate",
        "acceptanceDateTime",
        "form",
        "primaryDocument",
    }
    missing = sorted(required.difference(section))
    if missing:
        raise RuntimeError(f"SEC submission arrays missing required columns in {source_name}: {missing}")
    columns = {str(key): value for key, value in section.items() if isinstance(value, list)}
    if set(required).difference(columns):
        raise RuntimeError(f"SEC required submission columns are not arrays in {source_name}")
    lengths = {len(value) for value in columns.values()}
    if len(lengths) != 1:
        raise RuntimeError(f"SEC submission arrays have inconsistent lengths in {source_name}")
    observed_count = next(iter(lengths), 0)
    if expected_count is not None and observed_count != expected_count:
        raise RuntimeError(f"SEC historical filingCount mismatch in {source_name}: expected={expected_count} observed={observed_count}")
    return cast(list[dict[str, object]], pd.DataFrame(columns).to_dict(orient="records"))


def _filing_record(
    row: Mapping[str, object],
    *,
    ticker: str,
    cik: str,
    company_name: str,
    submission_file: str,
) -> SecFilingRecord:
    accepted = _acceptance_time_utc(row.get("acceptanceDateTime"))
    if pd.isna(accepted):
        raise RuntimeError(f"SEC filing has invalid acceptanceDateTime in {submission_file}")
    accession = str(row.get("accessionNumber", "")).strip()
    form = str(row.get("form", "")).strip().upper()
    filing_date = str(row.get("filingDate", "")).strip()
    if not accession or not form or not filing_date:
        raise RuntimeError(f"SEC filing identity is incomplete in {submission_file}")
    primary_document = str(row.get("primaryDocument", "")).strip()
    accession_path = accession.replace("-", "")
    document_url = (
        f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession_path}/{primary_document}"
        if accession_path and primary_document
        else ""
    )
    normalized_raw = {str(key): _json_value(value) for key, value in row.items()}
    return SecFilingRecord(
        ticker=ticker,
        cik=cik,
        company_name=company_name,
        form=form,
        accepted_at_utc=accepted.to_pydatetime(),
        filing_date=filing_date,
        report_date=str(row.get("reportDate", "") or "").strip(),
        accession_number=accession,
        primary_document=primary_document,
        document_url=document_url,
        submission_file=submission_file,
        file_number=str(row.get("fileNumber", "") or "").strip(),
        is_amendment=form.endswith("/A"),
        amends_accession_number=None,
        raw_sha256=_json_sha256(normalized_raw),
    )


def _link_amendments(records: tuple[SecFilingRecord, ...]) -> tuple[SecFilingRecord, ...]:
    originals: dict[tuple[str, str, str], list[SecFilingRecord]] = {}
    output: list[SecFilingRecord] = []
    for record in records:
        base_form = record.form.removesuffix("/A")
        key = (base_form, record.report_date, record.file_number)
        if record.is_amendment:
            candidates = originals.get(key, [])
            parent = candidates[-1].accession_number if len(candidates) == 1 else None
            output.append(replace(record, amends_accession_number=parent))
        else:
            originals.setdefault(key, []).append(record)
            output.append(record)
    return tuple(output)


def _validated_descriptor(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError("SEC historical submission descriptor is malformed")
    name = str(value.get("name", "")).strip()
    if not name or "/" in name or "\\" in name or not name.endswith(".json"):
        raise RuntimeError("SEC historical submission filename is unsafe")
    try:
        filing_count = int(value.get("filingCount", -1))
        filing_from = date.fromisoformat(str(value.get("filingFrom", "")))
        filing_to = date.fromisoformat(str(value.get("filingTo", "")))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("SEC historical submission descriptor is invalid") from exc
    if filing_count < 0 or filing_to < filing_from:
        raise RuntimeError("SEC historical submission descriptor is incoherent")
    return {
        "name": name,
        "filingCount": filing_count,
        "filingFrom": filing_from,
        "filingTo": filing_to,
    }


def _descriptor_overlaps(descriptor: Mapping[str, object], start: date, end: date) -> bool:
    filing_from = cast(date, descriptor["filingFrom"])
    filing_to = cast(date, descriptor["filingTo"])
    return filing_from <= end and filing_to >= start


def _acceptance_time_utc(value: object) -> pd.Timestamp:
    text = str(value or "").strip()
    if not text:
        return pd.NaT
    if len(text) == 14 and text.isdigit():
        timestamp = pd.Timestamp(datetime.strptime(text, "%Y%m%d%H%M%S"))
    else:
        try:
            timestamp = pd.Timestamp(text)
        except (TypeError, ValueError):
            return pd.NaT
    if pd.isna(timestamp):
        return pd.NaT
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(ZoneInfo("America/New_York"))
    return timestamp.tz_convert("UTC")


def _filing_identity(record: SecFilingRecord) -> tuple[object, ...]:
    return (
        record.cik,
        record.form,
        record.accepted_at_utc,
        record.accession_number,
        record.primary_document,
        record.raw_sha256,
    )


def _decoded_body(response: HttpByteResponse) -> bytes:
    encoding = (response.content_encoding or "").lower().strip()
    if not encoding or encoding == "identity":
        return response.body
    if encoding == "gzip":
        return gzip.decompress(response.body)
    if encoding == "deflate":
        return zlib.decompress(response.body)
    raise RuntimeError(f"unsupported SEC response content encoding: {encoding}")


def _retry_after_seconds(value: str | None, *, fallback: float) -> float:
    text = str(value or "").strip()
    if not text:
        return fallback
    try:
        return max(float(text), fallback)
    except ValueError:
        try:
            target = parsedate_to_datetime(text)
            if target.tzinfo is None:
                target = target.replace(tzinfo=UTC)
            return max((target.astimezone(UTC) - datetime.now(UTC)).total_seconds(), fallback)
        except (TypeError, ValueError, OverflowError):
            return fallback


def _normalized_cik(value: object) -> str:
    text = str(value).strip()
    if text.lower().startswith("cik:"):
        text = text[4:]
    if not text.isdigit() or len(text) > 10:
        raise ValueError(f"invalid SEC CIK: {value}")
    return text.zfill(10)


def _normalized_symbol(value: str) -> str:
    symbol = value.strip().upper().replace("/", ".").replace("-", ".")
    if not symbol:
        raise ValueError("ticker is empty")
    return symbol


def _require_utc(value: datetime, name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"SEC {name} must be timezone-aware")
    return value.astimezone(UTC)


def _json_value(value: object) -> object:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
