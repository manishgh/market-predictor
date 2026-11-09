"""Resumable original-response archive of public ICTax distribution sources.

Collection establishes request/response provenance only. Currency and entitlement
admission remain the responsibility of the independent dividend evidence adapter.
Anonymous session bodies and CSRF headers are never persisted.
"""
from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from pydantic import Field, model_validator

from market_predictor.canonical.store import file_sha256
from market_predictor.collection.http_records import http_response_from_record, http_response_record
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, read_json_object, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease
from market_predictor.research.investment_features import FLAGS, _publish
from market_predictor.sources.http import HttpClient
from market_predictor.swing.contracts.holding_accounting import HoldingContract
from market_predictor.swing.contracts.holding_materialization import SourcePin

BASE_URL = "https://www.ictax.admin.ch"
SESSION_URL = BASE_URL + "/extern/api/authentication/session.json"
SEARCH_URL = BASE_URL + "/lsi/api/security"
STRUCTURED_URL = BASE_URL + "/extern/api/security/security.json"
HTML_URL = BASE_URL + "/extern/api/security/securityHtml.json"
SCHEMA = "market_predictor.ictax_archive"
MAXIMUM_RESPONSE_BYTES = 2 * 1024**2
MAXIMUM_REQUEST_BYTES = 4 * 1024**2


class CapturedTaxSource(HoldingContract):
    body: SourcePin
    receipt: SourcePin


class TaxFund(HoldingContract):
    symbol: Annotated[str, Field(pattern=r"^[A-Z][A-Z0-9.\-]{0,15}$")]
    isin: Annotated[str, Field(pattern=r"^US[A-Z0-9]{9}[0-9]$")]
    years: Annotated[tuple[Annotated[int, Field(ge=2000, le=2100)], ...], Field(min_length=1, max_length=32)]

    @model_validator(mode="after")
    def ordered_years(self) -> Self:
        if tuple(sorted(set(self.years))) != self.years:
            raise ValueError("ICTax payment years must be sorted and unique")
        return self


class RetainedTaxPair(HoldingContract):
    symbol: str
    year: int
    structured: CapturedTaxSource
    gross_context: CapturedTaxSource


class ICTaxArchiveRequest(HoldingContract):
    schema_name: Literal["market_predictor.ictax_archive_request"] = Field(alias="schema")
    funds: Annotated[tuple[TaxFund, ...], Field(min_length=1, max_length=32)]
    retained: Annotated[tuple[RetainedTaxPair, ...], Field(max_length=1024)] = ()
    retry_attempts: Annotated[int, Field(ge=1, le=5)] = 3
    timeout_seconds: Annotated[int, Field(ge=1, le=120)] = 30
    pause_seconds: Annotated[float, Field(ge=0, le=10)] = 1.0

    @model_validator(mode="after")
    def unique_population(self) -> Self:
        if len({f.symbol for f in self.funds}) != len(self.funds) or len({f.isin for f in self.funds}) != len(self.funds):
            raise ValueError("ICTax funds must have unique symbols and ISINs")
        requested = {(f.symbol, year) for f in self.funds for year in f.years}
        retained = [(item.symbol, item.year) for item in self.retained]
        if len(set(retained)) != len(retained) or not set(retained).issubset(requested):
            raise ValueError("ICTax retained sources must identify distinct requested fund-years")
        return self


def reference_date(year: int) -> int:
    """The observed public client uses UTC December 31 milliseconds."""
    return int(datetime(year, 12, 31, tzinfo=UTC).timestamp()) * 1000


def _detail_request(fund: TaxFund, year: int, valor: int) -> dict[str, Any]:
    return {"isin": fund.isin, "valorNumber": valor, "referenceDate": reference_date(year),
            "language": "en", "isCantonBL": False}


def _read_pin(root: Path, pin: SourcePin, pins: dict[str, str], maximum: int = MAXIMUM_RESPONSE_BYTES) -> bytes:
    path = inside(root, pin.path)
    if path.stat().st_size > maximum:
        raise DataReadinessError("ICTax pinned source hash or byte bound differs")
    raw = path.read_bytes()
    if len(raw) > maximum or sha256(raw).hexdigest() != pin.sha256:
        raise DataReadinessError("ICTax pinned source hash or byte bound differs")
    pins[str(path)] = pin.sha256
    return raw


def _json(raw: bytes) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON field")
            result[key] = value
        return result

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda value: _invalid_constant())
    if not isinstance(value, dict):
        raise ValueError("ICTax response is not an object")
    return value


def _invalid_constant() -> None:
    raise ValueError("nonfinite JSON field")


def _contains_token(value: object, token: str) -> bool:
    if isinstance(value, str):
        return token in value
    if isinstance(value, dict):
        return any(_contains_token(key, token) or _contains_token(item, token) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(_contains_token(item, token) for item in value)
    return False


def _unsafe_response(body: bytes, metadata: dict[str, object], token: str) -> bool:
    if token.encode("utf-8") in body or _contains_token(metadata, token):
        return True
    try:
        # Preserve duplicate object members during screening, even though the
        # source parser subsequently refuses them as malformed source evidence.
        return _contains_token(json.loads(body, object_pairs_hook=lambda pairs: pairs), token)
    except (ValueError, UnicodeError):
        return False


def _clock(value: object) -> None:
    if not isinstance(value, str):
        raise DataReadinessError("ICTax capture clock missing")
    try:
        clock = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise DataReadinessError("ICTax capture clock invalid") from exc
    if clock.utcoffset() is None or clock > datetime.now(UTC):
        raise DataReadinessError("ICTax capture clock must be aware and not in the future")


def _pin(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}


def _load_directory(root: Path, directory: Path, pins: dict[str, str]) -> dict[str, Any]:
    manifest = directory / "_manifest.json"
    payload = _json(_read_pin(root, SourcePin(**_pin(root, manifest)), pins))
    files = payload.get("files")
    if not isinstance(files, dict):
        raise DataReadinessError("ICTax archive file inventory missing")
    for name, digest in files.items():
        if Path(name).name != name:
            raise DataReadinessError("ICTax checkpoint file path differs")
        _read_pin(root, SourcePin(path=(directory / name).relative_to(root).as_posix(), sha256=digest), pins)
    return payload


def _read_capture(root: Path, directory: Path, url: str, request: dict[str, Any],
                  job_sha256: str, pins: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    unit = _load_directory(root, directory, pins)
    if (unit.get("schema") != SCHEMA + ".capture" or unit.get("job_sha256") != job_sha256
        or unit.get("url") != url or unit.get("request") != request):
        raise DataReadinessError("ICTax capture request identity differs")
    if unit.get("status") == "transport_error":
        if unit["files"] or not isinstance(unit.get("error_type"), str):
            raise DataReadinessError("ICTax transport failure checkpoint differs")
        return unit, None
    if unit.get("status") != "response_archived" or set(unit["files"]) != {"response.bin", "receipt.json"}:
        raise DataReadinessError("ICTax response checkpoint differs")
    body = (directory / "response.bin").read_bytes()
    receipt = read_json_object(directory / "receipt.json")
    response = http_response_from_record(receipt, body)
    if (receipt.get("url") != url or receipt.get("method") != "POST" or receipt.get("request") != request
        or receipt.get("sha256") != response.sha256 or receipt.get("bytes") != len(body)
        or response.requested_url != url or response.final_url != url or response.redirect_chain
        or response.content_encoding not in (None, "", "identity")):
        raise DataReadinessError("ICTax receipt/request/transport binding differs")
    _clock(receipt.get("retrieved_at"))
    if receipt.get("retrieved_at") != receipt.get("retrieved_at_utc"):
        raise DataReadinessError("ICTax response capture clocks differ")
    unit["source"] = {"body": _pin(root, directory / "response.bin"), "receipt": _pin(root, directory / "receipt.json")}
    if response.status_code != 200:
        unit["gap_reason"] = f"http_status_{response.status_code}"
        return unit, None
    try:
        return unit, _json(body)
    except (ValueError, UnicodeError):
        unit["gap_reason"] = "invalid_response_json"
        return unit, None


class _Session:
    def __init__(self, client: HttpClient, policy: ICTaxArchiveRequest) -> None:
        self.client, self.policy = client, policy
        self.token: str | None = None

    def headers(self) -> dict[str, str]:
        if self.token is None:
            guard("ICTax anonymous session")
            response = self.client.get_bytes_with_metadata(
                SESSION_URL, retries=self.policy.retry_attempts, pause=self.policy.pause_seconds,
                maximum_body_bytes=MAXIMUM_RESPONSE_BYTES, allow_redirects=False,
            )
            if (response.status_code != 200 or response.final_url != SESSION_URL
                or response.content_encoding not in (None, "", "identity")):
                raise RuntimeError("ICTax session unavailable")
            payload = _json(response.body)
            data = payload.get("data")
            token = data.get("csrfToken") if isinstance(data, dict) else None
            if payload.get("status") != "SUCCESS" or not isinstance(token, str) or not token or len(token) > 4096:
                raise RuntimeError("ICTax anonymous session missing token")
            self.token = token
        return {"X-CSRF-TOKEN": self.token, "Accept": "application/json"}


def _capture(root: Path, directory: Path, url: str, request: dict[str, Any],
             job_sha256: str, session: _Session, pins: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not directory.exists():
        guard("ICTax source request")
        report: dict[str, Any] = {"schema": SCHEMA + ".capture", "job_sha256": job_sha256,
                                  "url": url, "request": request, **FLAGS}
        try:
            response = session.client.post_bytes_with_metadata(
                url, payload=request, headers=session.headers(), retries=session.policy.retry_attempts,
                pause=session.policy.pause_seconds, maximum_body_bytes=MAXIMUM_RESPONSE_BYTES,
                allow_redirects=False, raise_for_status=False,
            )
        except MemoryBudgetError:
            raise
        except (OSError, ValueError, RuntimeError) as exc:
            # Exception text can contain headers/session payloads. Persist only its type.
            report.update(status="transport_error", error_type=type(exc).__name__)
            _publish(directory, report, pins)
        else:
            token = session.token
            if (response.content_encoding not in (None, "", "identity") or
                token is not None and _unsafe_response(response.body, http_response_record(response), token)):
                report.update(status="transport_error", error_type="UnsafeSessionResponse")
                _publish(directory, report, pins)
                return _read_capture(root, directory, url, request, job_sha256, pins)
            report["status"] = "response_archived"
            receipt = {**http_response_record(response), "url": url, "method": "POST", "request": request,
                       "sha256": response.sha256, "bytes": len(response.body),
                       "retrieved_at": response.retrieved_at_utc.isoformat()}

            def save(path: Path) -> None:
                (path / "response.bin").write_bytes(response.body)
                write_json_object(path / "receipt.json", receipt)

            _publish(directory, report, pins, save)
    return _read_capture(root, directory, url, request, job_sha256, pins)


def _valor(payload: dict[str, Any] | None, isin: str) -> int | None:
    if payload is None:
        return None
    records = payload.get("security")
    if records is None:
        return None
    if records == [] and payload.get("total") == 0:
        return None
    if (not isinstance(records, list) or len(records) != 1 or payload.get("total") != 1
        or payload.get("count") != 1 or not isinstance(records[0], dict) or records[0].get("isin") != isin):
        raise DataReadinessError("ICTax lookup does not uniquely identify requested ISIN")
    valor = records[0].get("vn")
    if type(valor) is not int or valor <= 0:
        raise DataReadinessError("ICTax lookup valor identity differs")
    return valor


def _detail_status(payload: dict[str, Any] | None, request: dict[str, Any], *, html: bool) -> str | None:
    if payload is None:
        return "response_unavailable"
    if payload.get("status") != "SUCCESS" or payload.get("error") is not None:
        return "provider_response_unsuccessful"
    data = payload.get("data")
    if not isinstance(data, dict):
        return "response_data_missing"
    if html:
        return None if isinstance(data.get("html"), str) and data["html"] else "gross_context_missing"
    calculation = data.get("calculation")
    if not isinstance(calculation, dict):
        return "calculation_missing"
    if any(calculation.get(name) != request[name] for name in ("isin", "valorNumber", "referenceDate", "language")):
        raise DataReadinessError("ICTax response security/year identity differs")
    return None


def _retained(root: Path, pair: RetainedTaxPair, fund: TaxFund, pins: dict[str, str]) -> dict[str, Any]:
    captured: dict[str, dict[str, Any]] = {}
    requests: list[dict[str, Any]] = []
    has_http_status: list[bool] = []
    for name, source, url in (("structured", pair.structured, STRUCTURED_URL), ("gross_context", pair.gross_context, HTML_URL)):
        raw = _read_pin(root, source.body, pins)
        receipt = _json(_read_pin(root, source.receipt, pins))
        if (receipt.get("url") != url or receipt.get("method") != "POST" or receipt.get("sha256") != source.body.sha256
            or receipt.get("bytes") != len(raw) or receipt.get("status_code", 200) != 200):
            raise DataReadinessError("retained ICTax receipt binding differs")
        _clock(receipt.get("retrieved_at"))
        request = receipt.get("request")
        if not isinstance(request, dict) or type(request.get("valorNumber")) is not int or request["valorNumber"] <= 0:
            raise DataReadinessError("retained ICTax request identity missing")
        if request != _detail_request(fund, pair.year, request["valorNumber"]):
            raise DataReadinessError("retained ICTax request year/security differs")
        if _detail_status(_json(raw), request, html=name == "gross_context") is not None:
            raise DataReadinessError("retained ICTax source was not successful")
        requests.append(request)
        has_http_status.append("status_code" in receipt)
        captured[name] = source.model_dump()
    if requests[0] != requests[1]:
        raise DataReadinessError("retained ICTax pair requests differ")
    return {"symbol": fund.symbol, "isin": fund.isin, "year": pair.year, "status": "retained_sources_verified",
            "request": requests[0], "sources": captured,
            "transport_status_basis": ("original_receipts_http_status" if all(has_http_status)
                                       else "original_legacy_receipts_no_http_status")}


def _request(root: Path, path: Path, digest: str, pins: dict[str, str]) -> ICTaxArchiveRequest:
    raw = _read_pin(root, SourcePin(path=path.relative_to(root).as_posix(), sha256=digest), pins, MAXIMUM_REQUEST_BYTES)
    return ICTaxArchiveRequest.model_validate_json(raw)


def _records(root: Path, output: Path, policy: ICTaxArchiveRequest, job_sha256: str,
             pins: dict[str, str], session: _Session | None = None) -> list[dict[str, Any]]:
    def capture(directory: Path, url: str, request: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
        if session is None:
            return _read_capture(root, directory, url, request, job_sha256, pins)
        return _capture(root, directory, url, request, job_sha256, session, pins)

    retained = {(p.symbol, p.year): p for p in policy.retained}
    rows: list[dict[str, Any]] = []
    for fund in policy.funds:
        lookup: tuple[Path, int | None] | None = None
        for year in fund.years:
            guard("ICTax fund-year checkpoint")
            if (fund.symbol, year) in retained:
                rows.append(_retained(root, retained[fund.symbol, year], fund, pins))
                continue
            if lookup is None or lookup[1] is None:
                search = {"max": 100, "fetch": 10, "offset": 0, "isin": fund.isin, "year": year, "lang": "en"}
                lookup_directory = output / "lookups" / fund.isin / str(year)
                _, payload = capture(lookup_directory, SEARCH_URL, search)
                lookup = lookup_directory, _valor(payload, fund.isin)
            lookup_directory, valor = lookup
            row: dict[str, Any] = {"symbol": fund.symbol, "isin": fund.isin, "year": year,
                                   "lookup": _pin(root, lookup_directory / "_manifest.json")}
            if valor is None:
                rows.append({**row, "status": "source_gap", "reasons": ["lookup_unavailable"]})
                continue
            detail = _detail_request(fund, year, valor)
            sources: dict[str, Any] = {}
            gaps: list[str] = []
            for name, url in (("structured", STRUCTURED_URL), ("gross_context", HTML_URL)):
                archived, payload = capture(output / "units" / f"{fund.symbol}-{year}" / name, url, detail)
                sources[name] = archived.get("source")
                reason = _detail_status(payload, detail, html=name == "gross_context")
                if reason:
                    gaps.append(name + ":" + str(archived.get("gap_reason", archived.get("error_type", reason))))
            rows.append({**row, "status": "source_gap" if gaps else "responses_archived", "request": detail,
                         "sources": sources, "reasons": gaps})
    return rows


def collect_ictax_archive(*, root: Path, request: Path, request_sha256: str, output: Path,
                         client: HttpClient | None = None) -> dict[str, Any]:
    """Archive requested fund-years serially; failures never admit currency facts."""
    root = root.resolve()
    request, output = inside(root, request), inside(root, output)
    with heavy_job_lease("collect-ictax-distribution-sources", runtime_dir=root / "data/runtime"):
        guard("ICTax archive inputs")
        if (output / "final").exists():
            raise FileExistsError("completed ICTax archive is immutable")
        pins: dict[str, str] = {}
        policy = _request(root, request, request_sha256, pins)
        for path in (Path(__file__), Path(__file__).with_name("http.py")):
            pins[str(path.resolve())] = file_sha256(path)
        job = {"schema": SCHEMA + ".job", "request": _pin(root, request),
               "policy": policy.model_dump(mode="json", by_alias=True), "implementation": dict(pins)}
        job_sha256 = json_sha256(job)
        output.mkdir(parents=True, exist_ok=True)
        job_path = output / "_job.json"
        if job_path.exists():
            if read_json_object(job_path) != job:
                raise DataReadinessError("ICTax resume request or implementation differs")
        else:
            write_json_object(job_path, job)
        pins[str(job_path)] = file_sha256(job_path)
        owned_client = client is None
        transport = client or HttpClient(timeout=policy.timeout_seconds, before_request=lambda: guard("ICTax HTTP attempt"))
        session = _Session(transport, policy)
        try:
            rows = _records(root, output, policy, job_sha256, pins, session)
        finally:
            session.token = None
            if owned_client:
                transport.session.close()
        report = {"schema": SCHEMA, "status": "archive_complete_with_gaps" if any(r["status"] == "source_gap" for r in rows)
                  else "archive_complete", "job_sha256": job_sha256, "job": _pin(root, job_path),
                  "requested_fund_years": len(rows), "source_gap_fund_years": sum(r["status"] == "source_gap" for r in rows),
                  "retained_fund_years": len(policy.retained), "source_files": pins,
                  "availability_basis": "actual_capture_only_no_historical_backdating", **FLAGS}
        return _publish(output / "final", report, pins, lambda path: write_json_object(path / "records.json", {"records": rows}))


def verify_ictax_archive(*, root: Path, manifest: Path, manifest_sha256: str) -> dict[str, Any]:
    """Verify immutable request, response bytes and receipts without admitting events."""
    root = root.resolve()
    manifest = inside(root, manifest)
    with heavy_job_lease("verify-ictax-distribution-sources", runtime_dir=root / "data/runtime"):
        guard("ICTax archive verification")
        pins: dict[str, str] = {}
        report = _json(_read_pin(root, SourcePin(path=manifest.relative_to(root).as_posix(), sha256=manifest_sha256), pins))
        if report.get("schema") != SCHEMA or report.get("status") not in {"archive_complete", "archive_complete_with_gaps"}:
            raise DataReadinessError("not a completed ICTax archive")
        source_files = report.get("source_files")
        if not isinstance(source_files, dict):
            raise DataReadinessError("ICTax source inventory missing")
        for name, digest in source_files.items():
            guard("ICTax source verification")
            _read_pin(root, SourcePin(path=name, sha256=digest), pins, MAXIMUM_REQUEST_BYTES)
        _load_directory(root, manifest.parent, pins)
        records = read_json_object(manifest.parent / "records.json").get("records")
        if not isinstance(records, list) or len(records) != report.get("requested_fund_years"):
            raise DataReadinessError("ICTax fund-year accounting differs")
        output = manifest.parent.parent
        job = read_json_object(output / "_job.json")
        if (json_sha256(job) != report.get("job_sha256") or job.get("schema") != SCHEMA + ".job"
            or report.get("job") != _pin(root, output / "_job.json")):
            raise DataReadinessError("ICTax archive job hash differs")
        request_pin = SourcePin.model_validate(job["request"])
        policy = _request(root, inside(root, request_pin.path), request_pin.sha256, pins)
        if policy.model_dump(mode="json", by_alias=True) != job.get("policy"):
            raise DataReadinessError("ICTax archive policy differs from pinned request")
        expected = _records(root, output, policy, report["job_sha256"], pins)
        if (records != expected or report.get("retained_fund_years") != len(policy.retained)
            or report.get("source_gap_fund_years") != sum(r["status"] == "source_gap" for r in expected)
            or any(report.get(key) is not False for key in FLAGS)
            or report.get("status") != ("archive_complete_with_gaps" if any(r["status"] == "source_gap" for r in expected)
                                       else "archive_complete")):
            raise DataReadinessError("ICTax replayed fund-year records differ")
        return {**report, "records": records}
