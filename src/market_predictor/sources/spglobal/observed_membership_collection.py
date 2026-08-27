"""Collect and replay raw observations used by the S&P membership authority."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, Protocol, cast
from urllib.parse import parse_qsl, urlencode, urlsplit

import pandas as pd
from bs4 import BeautifulSoup

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.symbols import normalized_ticker
from market_predictor.sources.http import HttpByteResponse, HttpClient
from market_predictor.sources.spglobal.archive import (
    ARCHIVE_QUERY,
    SEARCH_PAGE_SIZE,
    SEARCH_PAGE_STRIDE,
    SP_GLOBAL_ARCHIVE_URL,
    SpGlobalAnnouncement,
    decode_spglobal_html,
    decode_spglobal_http_entity,
    parse_spglobal_archive_search_inventory,
)

RAW_UNIT_SCHEMA: Final = "edge_rebuild.sp500_observed_http_unit.v1"
ANCHOR_URL: Final = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
SEC_IDENTITY_URL: Final = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL: Final = "https://data.sec.gov/submissions/CIK{cik}.json"
MAXIMUM_RESPONSE_BYTES: Final = 16 * 1024 * 1024


class BytesHttpClient(Protocol):
    def get_bytes_with_metadata(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        retries: int = 3,
        pause: float = 1.0,
        maximum_body_bytes: int = MAXIMUM_RESPONSE_BYTES,
        allow_redirects: bool = False,
    ) -> HttpByteResponse: ...


class ObservedMembershipCollectionPolicy(Protocol):
    @property
    def maximum_pages(self) -> int: ...

    @property
    def retries(self) -> int: ...

    @property
    def retry_pause_seconds(self) -> float: ...


ClientFactory = Callable[[], BytesHttpClient]


def collect_official_release_prefix(
    client: BytesHttpClient,
    *,
    output_directory: Path,
    base_cutoff: date,
    config: ObservedMembershipCollectionPolicy,
    request_sha256: str,
) -> tuple[
    list[dict[str, object]],
    list[SpGlobalAnnouncement],
    list[tuple[object, ...]],
]:
    units: list[dict[str, object]] = []
    announcements: dict[str, SpGlobalAnnouncement] = {}
    previous_last_url: str | None = None
    seen_urls: set[str] = set()
    page_identities: list[tuple[object, ...]] = []
    for page_number in range(config.maximum_pages):
        unit = fetch_search_page(
            client,
            page_number=page_number,
            output_directory=output_directory,
            request_sha256=request_sha256,
            config=config,
            role="search_page",
        )
        parsed, dated_urls = parse_search_unit(output_directory, unit)
        validate_search_page_overlap(
            page_number,
            dated_urls,
            previous_last_url,
            seen_urls=seen_urls,
        )
        previous_last_url = dated_urls[-1][1]
        page_identities.append(search_page_semantics(parsed, dated_urls))
        units.append(unit)
        for item in parsed:
            if item.published_date > base_cutoff:
                announcements[item.url] = item
        if min(published for published, _ in dated_urls) <= base_cutoff:
            return (
                units,
                sorted(announcements.values(), key=lambda item: item.url),
                page_identities,
            )
    raise DataReadinessError("observed official prefix did not reach the closed cutoff")


def fetch_search_page(
    client: BytesHttpClient,
    *,
    page_number: int,
    output_directory: Path,
    request_sha256: str,
    config: ObservedMembershipCollectionPolicy,
    role: str,
) -> dict[str, object]:
    params = {**ARCHIVE_QUERY, "o": str(page_number * SEARCH_PAGE_STRIDE)}
    unit = fetch_observation_unit(
        client,
        url=SP_GLOBAL_ARCHIVE_URL,
        params=params,
        role=role,
        unit_id=f"{role}-{page_number:04d}",
        output_directory=output_directory,
        request_sha256=request_sha256,
        config=config,
        expected_host="press.spglobal.com",
        exact_path=urlsplit(SP_GLOBAL_ARCHIVE_URL).path,
    )
    unit["page_number"] = page_number
    write_observation_unit(output_directory, unit)
    return unit


def collect_release_documents(
    client: BytesHttpClient,
    *,
    output_directory: Path,
    announcements: list[SpGlobalAnnouncement],
    config: ObservedMembershipCollectionPolicy,
    request_sha256: str,
    role: str,
) -> list[dict[str, object]]:
    units: list[dict[str, object]] = []
    for item in announcements:
        unit_id = f"{role}-{hashlib.sha256(item.url.encode()).hexdigest()}"
        unit = fetch_observation_unit(
            client,
            url=item.url,
            params=None,
            role=role,
            unit_id=unit_id,
            output_directory=output_directory,
            request_sha256=request_sha256,
            config=config,
            expected_host="press.spglobal.com",
            exact_path=urlsplit(item.url).path,
        )
        unit["source_url"] = item.url
        unit["published_date"] = item.published_date.isoformat()
        unit["title"] = item.title
        write_observation_unit(output_directory, unit)
        units.append(unit)
    return units


def collect_identity_fallback_documents(
    client: BytesHttpClient,
    *,
    output_directory: Path,
    requests: Sequence[tuple[str, str]],
    config: ObservedMembershipCollectionPolicy,
    request_sha256: str,
) -> list[dict[str, object]]:
    units: list[dict[str, object]] = []
    for ticker, cik in requests:
        url = SEC_SUBMISSIONS_URL.format(cik=cik)
        unit = fetch_observation_unit(
            client,
            url=url,
            params=None,
            role="identity_fallback",
            unit_id=f"sec-submission-{ticker}-{cik}",
            output_directory=output_directory,
            request_sha256=request_sha256,
            config=config,
            expected_host="data.sec.gov",
            exact_path=f"/submissions/CIK{cik}.json",
        )
        unit["ticker"] = ticker
        unit["cik"] = cik
        write_observation_unit(output_directory, unit)
        units.append(unit)
    return units


def fetch_observation_unit(
    client: BytesHttpClient,
    *,
    url: str,
    params: dict[str, Any] | None,
    role: str,
    unit_id: str,
    output_directory: Path,
    request_sha256: str,
    config: ObservedMembershipCollectionPolicy,
    expected_host: str,
    exact_path: str,
) -> dict[str, object]:
    response = client.get_bytes_with_metadata(
        url,
        params=params,
        retries=config.retries,
        pause=config.retry_pause_seconds,
        maximum_body_bytes=MAXIMUM_RESPONSE_BYTES,
        allow_redirects=False,
    )
    expected_url = url if params is None else f"{url}?{urlencode(params)}"
    verify_observation_response(
        response,
        expected_url=expected_url,
        expected_host=expected_host,
        exact_path=exact_path,
    )
    body_sha256 = hashlib.sha256(response.body).hexdigest()
    body_path = output_directory / "objects" / f"{body_sha256}.bin"
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_bytes(response.body)
    unit: dict[str, object] = {
        "schema": RAW_UNIT_SCHEMA,
        "request_sha256": request_sha256,
        "role": role,
        "unit_id": unit_id,
        "requested_url": response.requested_url,
        "final_url": response.final_url,
        "redirect_chain": list(response.redirect_chain),
        "status_code": response.status_code,
        "retrieved_at_utc": parse_utc_timestamp(response.retrieved_at_utc).isoformat(),
        "content_type": response.content_type,
        "content_encoding": response.content_encoding,
        "etag": response.etag,
        "last_modified": response.last_modified,
        "body_length": len(response.body),
        "body_sha256": body_sha256,
        "body_path": body_path.relative_to(output_directory).as_posix(),
        "body_representation": response.body_representation,
    }
    write_observation_unit(output_directory, unit)
    return unit


def verify_observation_response(
    response: HttpByteResponse,
    *,
    expected_url: str,
    expected_host: str,
    exact_path: str,
) -> None:
    requested = urlsplit(response.requested_url)
    final = urlsplit(response.final_url)
    expected = urlsplit(expected_url)
    if (
        requested.scheme != "https"
        or requested.hostname != expected_host
        or requested.username is not None
        or requested.password is not None
        or requested.port not in {None, 443}
        or requested.path != exact_path
        or requested.fragment
        or final != requested
        or response.redirect_chain
        or response.status_code != 200
        or response.body_length != len(response.body)
        or response.sha256 != hashlib.sha256(response.body).hexdigest()
        or response.body_representation != "http_entity_encoded"
        or sorted(parse_qsl(requested.query)) != sorted(parse_qsl(expected.query))
    ):
        raise DataReadinessError("observed S&P HTTP response identity is invalid")
    parse_utc_timestamp(response.retrieved_at_utc)


def parse_search_unit(
    root: Path,
    unit: Mapping[str, object],
) -> tuple[list[SpGlobalAnnouncement], list[tuple[date, str]]]:
    body = load_observation_body(root, unit)
    html = decode_spglobal_http_entity(body, str(unit.get("content_encoding") or ""))
    return parse_spglobal_archive_search_inventory(
        html,
        base_url=str(unit["final_url"]),
        content_type=str(unit.get("content_type") or ""),
    )


def load_release_html(root: Path, unit: Mapping[str, object]) -> str:
    decoded = decode_spglobal_http_entity(
        load_observation_body(root, unit),
        str(unit.get("content_encoding") or ""),
    )
    return decode_spglobal_html(decoded, str(unit.get("content_type") or ""))


def validate_search_page_overlap(
    page_number: int,
    dated_urls: list[tuple[date, str]],
    previous_last_url: str | None,
    *,
    seen_urls: set[str] | None = None,
) -> None:
    if len(dated_urls) != SEARCH_PAGE_SIZE:
        raise DataReadinessError("observed official search page is truncated")
    dates = [published for published, _ in dated_urls]
    if any(left < right for left, right in zip(dates, dates[1:], strict=False)):
        raise DataReadinessError("observed official search page is not newest-to-oldest")
    urls = {url for _, url in dated_urls}
    overlap = set() if seen_urls is None else seen_urls.intersection(urls)
    if page_number == 0:
        if previous_last_url is not None:
            raise DataReadinessError("observed official first page has prior state")
        if overlap:
            raise DataReadinessError("observed official first page repeats URLs")
    elif previous_last_url is None or dated_urls[0][1] != previous_last_url or (
        seen_urls is not None and overlap != {previous_last_url}
    ):
        raise DataReadinessError("observed official pagination overlap is invalid")
    if seen_urls is not None:
        seen_urls.update(urls)


def search_page_semantics(
    announcements: list[SpGlobalAnnouncement],
    dated_urls: list[tuple[date, str]],
) -> tuple[object, ...]:
    return (
        tuple((published.isoformat(), url) for published, url in dated_urls),
        tuple((item.published_date.isoformat(), item.title, item.url) for item in announcements),
    )


def parse_constituent_anchor_source(
    root: Path,
    unit: Mapping[str, object],
) -> pd.DataFrame:
    body = load_observation_body(root, unit)
    html = decode_spglobal_http_entity(body, str(unit.get("content_encoding") or ""))
    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table", id="constituents")
    if table is None:
        raise DataReadinessError("independent S&P anchor table is missing")
    rows = table.find_all("tr")
    if not rows:
        raise DataReadinessError("independent S&P anchor has no rows")
    header = [cell.get_text(" ", strip=True) for cell in rows[0].find_all(["th", "td"])]
    aliases = {value.lower(): index for index, value in enumerate(header)}
    required = {
        "symbol": "ticker",
        "security": "company",
        "gics sector": "sector",
        "gics sub-industry": "industry",
        "cik": "cik",
    }
    missing = sorted(set(required).difference(aliases))
    if missing:
        raise DataReadinessError(f"independent S&P anchor columns are missing: {missing}")
    records: list[dict[str, str]] = []
    for row in rows[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])]
        if not cells:
            continue
        if len(cells) != len(header):
            raise DataReadinessError("independent S&P anchor row width changed")
        record = {target: cells[aliases[source]].strip() for source, target in required.items()}
        record["ticker"] = normalized_ticker(record["ticker"].replace("-", "."))
        record["cik"] = record["cik"].removesuffix(".0").zfill(10)
        records.append(record)
    anchor = pd.DataFrame(records, columns=["ticker", "company", "sector", "industry", "cik"])
    if not 450 <= len(anchor) <= 550:
        raise DataReadinessError("independent S&P anchor must contain 450..550 constituents")
    if bool(anchor["ticker"].duplicated().any()) or bool(anchor.eq("").any().any()):
        raise DataReadinessError("independent S&P anchor has duplicate or empty identity")
    if bool((~anchor["cik"].str.fullmatch(r"\d{10}")).any()):
        raise DataReadinessError("independent S&P anchor CIK is invalid")
    return anchor.sort_values("ticker", kind="stable").reset_index(drop=True)


def parse_sec_company_identities(
    root: Path,
    unit: Mapping[str, object],
) -> dict[str, str]:
    body = decode_spglobal_http_entity(
        load_observation_body(root, unit),
        str(unit.get("content_encoding") or ""),
        maximum_decoded_bytes=MAXIMUM_RESPONSE_BYTES,
    )
    try:
        value = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataReadinessError("SEC company-ticker identity response is invalid") from exc
    if not isinstance(value, dict):
        raise DataReadinessError("SEC company-ticker identity response is not an object")
    identities: dict[str, str] = {}
    for record in value.values():
        if not isinstance(record, dict):
            raise DataReadinessError("SEC company-ticker identity record is invalid")
        try:
            ticker = normalized_ticker(str(record["ticker"]).replace("-", "."))
            cik = str(int(record["cik_str"])).zfill(10)
        except (KeyError, TypeError, ValueError) as exc:
            raise DataReadinessError("SEC company-ticker identity record is incomplete") from exc
        existing = identities.get(ticker)
        if existing is not None and existing != cik:
            raise DataReadinessError(f"SEC company-ticker identity is ambiguous: {ticker}")
        identities[ticker] = cik
    if len(identities) < 5_000:
        raise DataReadinessError("SEC company-ticker identity response is truncated")
    return identities


def parse_sec_submission_identities(
    root: Path,
    units: Sequence[Mapping[str, object]],
) -> dict[str, str]:
    identities: dict[str, str] = {}
    for unit in units:
        ticker = normalized_ticker(str(unit.get("ticker", "")).replace("-", "."))
        cik = str(unit.get("cik", ""))
        expected_url = SEC_SUBMISSIONS_URL.format(cik=cik)
        if (
            unit.get("role") != "identity_fallback"
            or unit.get("unit_id") != f"sec-submission-{ticker}-{cik}"
            or unit.get("requested_url") != expected_url
            or unit.get("final_url") != expected_url
            or unit.get("redirect_chain") != []
            or int(str(unit.get("status_code", -1))) != 200
            or not cik.isdigit()
            or len(cik) != 10
        ):
            raise DataReadinessError("SEC identity fallback response identity changed")
        body = decode_spglobal_http_entity(
            load_observation_body(root, unit),
            str(unit.get("content_encoding") or ""),
            maximum_decoded_bytes=MAXIMUM_RESPONSE_BYTES,
        )
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DataReadinessError("SEC identity fallback response is invalid") from exc
        if not isinstance(payload, dict):
            raise DataReadinessError("SEC identity fallback response is not an object")
        response_cik = str(payload.get("cik", "")).strip().zfill(10)
        response_tickers = payload.get("tickers")
        if not isinstance(response_tickers, list) or not all(isinstance(value, str) for value in response_tickers):
            raise DataReadinessError("SEC identity fallback ticker inventory is invalid")
        normalized_response_tickers = {normalized_ticker(value.replace("-", ".")) for value in response_tickers}
        if response_cik != cik or ticker not in normalized_response_tickers:
            raise DataReadinessError(f"SEC identity fallback does not verify ticker and CIK: {ticker}")
        existing = identities.get(ticker)
        if existing is not None and existing != cik:
            raise DataReadinessError(f"SEC identity fallback is ambiguous: {ticker}")
        identities[ticker] = cik
    return identities


def verify_observed_membership_raw_evidence(
    root: Path,
    units: Sequence[Mapping[str, object]],
    *,
    request_sha256: str,
) -> None:
    units_directory = root / "units"
    objects_directory = root / "objects"
    if (
        not units_directory.is_dir()
        or not objects_directory.is_dir()
        or units_directory.is_symlink()
        or objects_directory.is_symlink()
    ):
        raise DataReadinessError("observed membership raw directories differ from inventory")
    expected_units = {f"{unit['unit_id']}.json" for unit in units}
    unit_entries = list(units_directory.iterdir())
    observed_units = {path.name for path in unit_entries}
    expected_objects = {Path(str(unit["body_path"])).name for unit in units}
    object_entries = list(objects_directory.iterdir())
    observed_objects = {path.name for path in object_entries}
    if (
        observed_units != expected_units
        or observed_objects != expected_objects
        or any(path.is_symlink() or not path.is_file() for path in unit_entries)
        or any(path.is_symlink() or not path.is_file() for path in object_entries)
    ):
        raise DataReadinessError("observed membership raw files differ from inventory")
    for expected in units:
        body_sha256 = str(expected.get("body_sha256", ""))
        if (
            expected.get("schema") != RAW_UNIT_SCHEMA
            or expected.get("request_sha256") != request_sha256
            or expected.get("body_representation") != "http_entity_encoded"
            or len(body_sha256) != 64
            or any(character not in "0123456789abcdef" for character in body_sha256)
            or expected.get("body_path") != f"objects/{body_sha256}.bin"
        ):
            raise DataReadinessError("observed membership raw envelope changed")
        actual = _json_object(units_directory / f"{expected['unit_id']}.json")
        if actual != expected:
            raise DataReadinessError("observed membership raw sidecar changed")
        load_observation_body(root, actual)


def load_observation_body(root: Path, unit: Mapping[str, object]) -> bytes:
    path = _resolve_inside(root, str(unit.get("body_path", "")))
    body = path.read_bytes()
    if len(body) != int(str(unit.get("body_length", -1))) or hashlib.sha256(body).hexdigest() != unit.get(
        "body_sha256"
    ):
        raise DataReadinessError("observed membership response body changed")
    return body


def write_observation_unit(root: Path, unit: Mapping[str, object]) -> None:
    path = root / "units" / f"{unit['unit_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, unit)


def observation_time(unit: Mapping[str, object]) -> datetime:
    return parse_utc_timestamp(str(unit.get("retrieved_at_utc", "")))


def parse_utc_timestamp(value: datetime | str) -> datetime:
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value)
    except ValueError as exc:
        raise DataReadinessError("observed membership timestamp is invalid") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise DataReadinessError("observed membership timestamp is not timezone-aware")
    return parsed.astimezone(UTC)


def default_client_factory() -> BytesHttpClient:
    return cast(
        BytesHttpClient,
        HttpClient(user_agent="market-predictor/0.1 observed-sp500-membership"),
    )


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise DataReadinessError("observed membership path escapes authority root")
    return candidate


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"observed membership JSON is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"observed membership JSON is not an object: {path}")
    return {str(key): item for key, item in value.items()}


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
