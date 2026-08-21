"""Immutable issuer-level SEC filing collection with raw-response replay."""
from __future__ import annotations



import gzip
import hashlib
import json
import os
import shutil
import threading
import tomllib
import zipfile
import zlib
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Final, Protocol
from urllib.parse import unquote, urlparse
from uuid import uuid4
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditCheck,
    CanonicalAuditReport,
    audit_canonical_events,
    audit_source_collections,
)
from market_predictor.canonical.contracts import CanonicalEvent, SourceCollection
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.sources.sec import SecFilingHistory, SecRawResponse, SecSourceResponseError
from market_predictor.core.errors import DataReadinessError

SEC_COLLECTION_SCHEMA: Final = "edge_rebuild.sec_filing_collection.v2"
SEC_COLLECTION_MANIFEST_SCHEMA: Final = "edge_rebuild.sec_filing_collection_manifest.v2"
SEC_SOURCE_FAMILY: Final = "sec"
SEC_EVENT_ARTIFACT_TYPE: Final = "sec_issuer_filing_events"
SEC_COVERAGE_ARTIFACT_TYPE: Final = "sec_issuer_source_collections"
SEC_RAW_INVENTORY_ARTIFACT_TYPE: Final = "sec_raw_response_inventory"
DEFAULT_START_DATE: Final = date(2019, 7, 9)
DEFAULT_END_DATE: Final = date(2026, 7, 8)
DEFAULT_FORMS: Final = (
    "8-K",
    "8-K/A",
    "10-K",
    "10-K/A",
    "10-Q",
    "10-Q/A",
    "20-F",
    "20-F/A",
    "6-K",
    "6-K/A",
    "S-1",
    "S-1/A",
    "S-3",
    "S-3/A",
    "424B1",
    "424B2",
    "424B3",
    "424B4",
    "424B5",
    "DEF 14A",
    "3",
    "3/A",
    "4",
    "4/A",
    "5",
    "5/A",
)
_ET = ZoneInfo("America/New_York")
_XNYS = xcals.get_calendar("XNYS")
_EVENT_EXTRA_COLUMNS: Final = (
    "sec_cik",
    "sec_form",
    "accession_number",
    "filing_date",
    "report_date",
    "primary_document",
    "submission_file",
    "file_number",
    "accepted_at_utc",
    "is_amendment",
    "amends_accession_number",
    "availability_rule",
)
_COVERAGE_EXTRA_COLUMNS: Final = (
    "sec_cik",
    "issuer_security_id",
    "company_name",
    "submission_files_json",
    "response_sha256",
    "source_row_count",
    "availability_evidence",
    "historical_availability_proven",
    "production_eligible",
)
_RELATION_COLUMNS: Final = (
    "sec_cik",
    "security_id",
    "ticker",
    "effective_from_utc",
    "effective_to_utc",
    "available_at_utc",
)


@dataclass(frozen=True, slots=True)
class SecFilingCollectionConfig:
    start_date: date = DEFAULT_START_DATE
    end_date: date = DEFAULT_END_DATE
    forms: tuple[str, ...] = DEFAULT_FORMS
    max_workers: int = 2
    requests_per_second: float = 6.0
    forbidden_cooldown_seconds: float = 600.0
    rate_limit_cooldown_seconds: float = 60.0
    dissemination_lag_minutes: int = 5


@dataclass(frozen=True, slots=True)
class SecFilingCollection:
    directory: Path
    events: pd.DataFrame
    source_collections: pd.DataFrame
    raw_inventory: pd.DataFrame
    manifest: Mapping[str, object]
    authority: Mapping[str, object]


class SecHistorySource(Protocol):
    def fetch_cik_filing_history(
        self,
        cik: str,
        start: datetime,
        end: datetime,
        *,
        forms: set[str] | None = None,
        ticker_hint: str = "SEC",
    ) -> SecFilingHistory: ...


Clock = Callable[[], datetime]


def load_sec_filing_collection_config(path: Path) -> SecFilingCollectionConfig:
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(f"SEC filing config is unreadable: {path}") from exc
    section = payload.get("sec_filings")
    if not isinstance(section, dict):
        raise DataReadinessError("SEC filing config requires [sec_filings]")
    try:
        config = SecFilingCollectionConfig(
            start_date=date.fromisoformat(str(section.get("start_date", DEFAULT_START_DATE))),
            end_date=date.fromisoformat(str(section.get("end_date", DEFAULT_END_DATE))),
            forms=tuple(str(value) for value in section.get("forms", DEFAULT_FORMS)),
            max_workers=int(section.get("max_workers", 2)),
            requests_per_second=float(section.get("requests_per_second", 6.0)),
            forbidden_cooldown_seconds=float(section.get("forbidden_cooldown_seconds", 600.0)),
            rate_limit_cooldown_seconds=float(section.get("rate_limit_cooldown_seconds", 60.0)),
            dissemination_lag_minutes=int(section.get("dissemination_lag_minutes", 5)),
        )
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("SEC filing config contains invalid values") from exc
    return validate_sec_filing_collection_config(config)


def validate_sec_filing_collection_config(config: SecFilingCollectionConfig) -> SecFilingCollectionConfig:
    if config.end_date < config.start_date:
        raise DataReadinessError("SEC filing collection window is reversed")
    if config.max_workers < 1 or config.max_workers > 4:
        raise DataReadinessError("SEC max_workers must be between 1 and 4")
    if config.requests_per_second <= 0 or config.requests_per_second >= 10:
        raise DataReadinessError("SEC requests_per_second must be below the SEC 10 requests/second limit")
    if config.forbidden_cooldown_seconds < 600 or config.rate_limit_cooldown_seconds < 1:
        raise DataReadinessError("SEC cooldown policy is too short")
    if config.dissemination_lag_minutes < 3:
        raise DataReadinessError("SEC retrospective dissemination lag must be at least three minutes")
    forms = tuple(sorted({value.strip().upper() for value in config.forms if value.strip()}))
    if not forms:
        raise DataReadinessError("SEC filing forms must be explicit and non-empty")
    return SecFilingCollectionConfig(
        start_date=config.start_date,
        end_date=config.end_date,
        forms=forms,
        max_workers=config.max_workers,
        requests_per_second=config.requests_per_second,
        forbidden_cooldown_seconds=config.forbidden_cooldown_seconds,
        rate_limit_cooldown_seconds=config.rate_limit_cooldown_seconds,
        dissemination_lag_minutes=config.dissemination_lag_minutes,
    )


def load_sec_identity_relations(path: Path) -> pd.DataFrame:
    try:
        frame = pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)
    except (OSError, ValueError) as exc:
        raise DataReadinessError(f"SEC identity relation is unreadable: {path}") from exc
    return normalize_sec_identity_relations(frame)


def normalize_sec_identity_relations(frame: pd.DataFrame) -> pd.DataFrame:
    required_membership = {"security_id", "ticker", "effective_from_utc", "effective_to_utc", "available_at_utc"}
    missing = sorted(required_membership.difference(frame.columns))
    if missing:
        raise DataReadinessError("SEC identity relation missing columns: " + ", ".join(missing))
    output = frame.loc[:, list(required_membership | ({"sec_cik"} & set(frame.columns)))].copy()
    if "sec_cik" not in output:
        security = output["security_id"].astype(str).str.strip()
        output["sec_cik"] = security.where(security.str.match(r"^cik:\d{1,10}$", case=False)).str[4:]
    output["sec_cik"] = output["sec_cik"].astype("string").str.strip().str.replace(r"(?i)^cik:", "", regex=True)
    output = output.loc[output["sec_cik"].str.fullmatch(r"\d{1,10}", na=False)].copy()
    if output.empty:
        raise DataReadinessError("SEC identity relation contains no proven CIK mappings")
    output["sec_cik"] = output["sec_cik"].str.zfill(10)
    output["security_id"] = output["security_id"].astype(str).str.strip()
    output["ticker"] = output["ticker"].astype(str).str.strip().str.upper().str.replace("/", ".", regex=False)
    for column in ("effective_from_utc", "available_at_utc"):
        output[column] = _strict_utc(output[column], column)
    output["effective_to_utc"] = pd.to_datetime(output["effective_to_utc"], utc=True, errors="coerce")
    if bool(
        output["security_id"].eq("").any()
        or output["ticker"].eq("").any()
        or output.duplicated(["security_id", "ticker", "effective_from_utc"]).any()
        or (output["effective_to_utc"].notna() & output["effective_to_utc"].le(output["effective_from_utc"])).any()
        or output["available_at_utc"].gt(output["effective_from_utc"]).any()
    ):
        raise DataReadinessError("SEC identity relation contains invalid or unavailable intervals")
    ordered = (
        output.loc[:, list(_RELATION_COLUMNS)]
        .sort_values(["security_id", "ticker", "effective_from_utc"], kind="stable")
        .reset_index(drop=True)
    )
    for _, group in ordered.groupby(["security_id", "ticker"], sort=False):
        prior_end: pd.Timestamp | None = None
        open_ended_seen = False
        for index, row in enumerate(group.itertuples(index=False)):
            start = pd.Timestamp(row.effective_from_utc)
            if index > 0 and open_ended_seen:
                raise DataReadinessError("SEC identity relation follows an open-ended interval")
            if prior_end is not None and start < prior_end:
                raise DataReadinessError("SEC identity relation intervals overlap")
            open_ended_seen = pd.isna(row.effective_to_utc)
            prior_end = None if open_ended_seen else pd.Timestamp(row.effective_to_utc)
    return ordered


def collect_historical_sec_filings(
    identity_relations: pd.DataFrame,
    output_directory: Path,
    *,
    source_factory: Callable[[], SecHistorySource],
    config: SecFilingCollectionConfig,
    clock: Clock | None = None,
) -> SecFilingCollection:
    """Collect each unique issuer CIK once and publish research-only evidence."""

    normalized = validate_sec_filing_collection_config(config)
    relations = normalize_sec_identity_relations(identity_relations)
    issuer_hints = relations.groupby("sec_cik", sort=True)["ticker"].agg(lambda values: sorted(set(values.astype(str)))[0]).to_dict()
    ciks = tuple(sorted(issuer_hints))
    now = clock or (lambda: datetime.now(UTC))
    requested_start = datetime.combine(normalized.start_date, time.min, tzinfo=UTC)
    requested_end = datetime.combine(normalized.end_date, time.max, tzinfo=UTC)
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise DataReadinessError(f"SEC filing collection is immutable: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = output_directory.with_name(f".{output_directory.name}.{uuid4().hex}.tmp")
    staging.mkdir()
    relation_sha256 = _relation_sha256(relations)
    request = {
        "schema": SEC_COLLECTION_SCHEMA,
        "issuer_ciks": list(ciks),
        "identity_relation_sha256": relation_sha256,
        "requested_start_utc": requested_start.isoformat(),
        "requested_end_utc": requested_end.isoformat(),
        "forms": list(normalized.forms),
        "max_workers": normalized.max_workers,
        "requests_per_second": normalized.requests_per_second,
        "forbidden_cooldown_seconds": normalized.forbidden_cooldown_seconds,
        "rate_limit_cooldown_seconds": normalized.rate_limit_cooldown_seconds,
        "dissemination_lag_minutes": normalized.dissemination_lag_minutes,
        "availability_policy": "sec_daily_swing_conservative_proxy",
        "historical_availability_proven": False,
        "production_ready": False,
    }
    request_sha256 = _json_sha256(request)
    source_policy = {
        "source_family": SEC_SOURCE_FAMILY,
        "identity": "one collection per unique issuer CIK",
        "event_time": "SEC acceptanceDateTime",
        "first_seen_time": "actual raw-response retrieval/collector completion time",
        "research_feature_time": "acceptance plus conservative dissemination/late-filing policy",
        "production_policy": "retrospective proxy is never production eligible",
        "zero_policy": "zero is known only after strict schema and filingCount reconciliation",
    }
    source_policy_sha256 = _json_sha256(source_policy)
    try:
        results: list[_IssuerResult] = []
        thread_state = threading.local()

        def worker_source() -> SecHistorySource:
            source = getattr(thread_state, "source", None)
            if source is None:
                source = source_factory()
                thread_state.source = source
            return source

        raw_zip_path = staging / "raw_responses.zip"
        raw_records: list[dict[str, object]] = []
        raw_response_count = 0
        with zipfile.ZipFile(raw_zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as raw_archive:
            with ThreadPoolExecutor(max_workers=normalized.max_workers) as executor:
                futures = {
                    executor.submit(
                        _collect_issuer,
                        worker_source,
                        cik,
                        issuer_hints[cik],
                        requested_start,
                        requested_end,
                        set(normalized.forms),
                        now,
                    ): cik
                    for cik in ciks
                }
                for future in as_completed(futures):
                    futures.pop(future, None)
                    result = future.result()
                    raw_response_count += _archive_raw_responses(raw_archive, raw_records, result)
                    results.append(_release_raw_bodies(result))
                    del result
        results.sort(key=lambda item: item.cik)
        events = _event_frame(results, lag_minutes=normalized.dissemination_lag_minutes)
        coverage = _coverage_frame(results, requested_start, requested_end, request_sha256)
        _validate_collection_content(events, coverage, ciks)
        event_audit = CanonicalAuditReport(checks=audit_canonical_events(events, require_observed=False))
        coverage_audit = CanonicalAuditReport(
            checks=(*audit_source_collections(coverage, require_success=False), _collection_identity_check(coverage, ciks))
        )
        event_audit.raise_for_failure()
        coverage_audit.raise_for_failure()
        raw_inventory = _raw_inventory_frame(raw_records, expected_count=raw_response_count)
        raw_audit = CanonicalAuditReport(
            checks=(
                CanonicalAuditCheck(
                    name="sec_raw_response_inventory",
                    status="pass",
                    failures=0,
                    rows_checked=len(raw_inventory),
                    detail="every successful issuer response is archived and hash-addressed",
                ),
            )
        )
        response_sha256 = _json_sha256(
            [
                {
                    "sec_cik": result.cik,
                    "response_sha256": result.response_sha256,
                    "error_type": result.error_type,
                }
                for result in results
            ]
        )
        inputs = {
            "collection_request_sha256": request_sha256,
            "source_policy_sha256": source_policy_sha256,
            "source_response_sha256": response_sha256,
            "collector_schema": SEC_COLLECTION_SCHEMA,
        }
        event_path = staging / "filing_events.parquet"
        coverage_path = staging / "source_collections.parquet"
        inventory_path = staging / "raw_response_inventory.parquet"
        event_manifest = write_canonical_artifact(
            events, event_path, artifact_type=SEC_EVENT_ARTIFACT_TYPE, audit=event_audit, inputs=inputs, production_ready=False
        )
        coverage_manifest = write_canonical_artifact(
            coverage, coverage_path, artifact_type=SEC_COVERAGE_ARTIFACT_TYPE, audit=coverage_audit, inputs=inputs, production_ready=False
        )
        inventory_manifest = write_canonical_artifact(
            raw_inventory,
            inventory_path,
            artifact_type=SEC_RAW_INVENTORY_ARTIFACT_TYPE,
            audit=raw_audit,
            inputs=inputs,
            production_ready=False,
        )
        for path in (event_path, coverage_path, inventory_path):
            path.with_suffix(".parquet.lock").unlink(missing_ok=True)
            _rewrite_artifact_path(path, output_directory)
        event_manifest = _json_object(manifest_path_for(event_path))
        coverage_manifest = _json_object(manifest_path_for(coverage_path))
        inventory_manifest = _json_object(manifest_path_for(inventory_path))
        failures = sum(result.error_type is not None for result in results)
        manifest: dict[str, object] = {
            "schema": SEC_COLLECTION_MANIFEST_SCHEMA,
            "state": "complete",
            "request": request,
            "request_sha256": request_sha256,
            "source_policy": source_policy,
            "source_policy_sha256": source_policy_sha256,
            "source_response_sha256": response_sha256,
            "issuer_count": len(ciks),
            "successful_issuers": len(ciks) - failures,
            "failed_issuers": failures,
            "event_rows": len(events),
            "production_ready": False,
            "research_only_reason": "historical SEC first-seen availability is not proven",
            "artifacts": {
                "events": _artifact_record(event_path, event_manifest),
                "source_collections": _artifact_record(coverage_path, coverage_manifest),
                "raw_inventory": _artifact_record(inventory_path, inventory_manifest),
                "raw_archive": {"path": raw_zip_path.name, "sha256": file_sha256(raw_zip_path), "bytes": raw_zip_path.stat().st_size},
            },
        }
        _atomic_json(staging / "_manifest.json", manifest)
        authority = {
            "schema": SEC_COLLECTION_SCHEMA,
            "state": "complete",
            "manifest": "_manifest.json",
            "manifest_sha256": file_sha256(staging / "_manifest.json"),
            "request_sha256": request_sha256,
            "event_artifact_sha256": event_manifest["artifact_sha256"],
            "coverage_artifact_sha256": coverage_manifest["artifact_sha256"],
            "raw_inventory_sha256": inventory_manifest["artifact_sha256"],
            "raw_archive_sha256": file_sha256(raw_zip_path),
            "production_ready": False,
        }
        _atomic_json(staging / "_authority.json", authority)
        load_sec_filing_collection(staging)
        os.replace(staging, output_directory)
        return load_sec_filing_collection(output_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_sec_filing_collection(directory: Path) -> SecFilingCollection:
    directory = directory.resolve()
    expected = {
        "_authority.json",
        "_manifest.json",
        "filing_events.parquet",
        "filing_events.parquet.manifest.json",
        "source_collections.parquet",
        "source_collections.parquet.manifest.json",
        "raw_response_inventory.parquet",
        "raw_response_inventory.parquet.manifest.json",
        "raw_responses.zip",
    }
    if not directory.is_dir() or {path.name for path in directory.iterdir()} != expected:
        raise DataReadinessError("SEC filing collection inventory does not verify")
    manifest_path = directory / "_manifest.json"
    manifest = _json_object(manifest_path)
    authority = _json_object(directory / "_authority.json")
    if manifest.get("schema") != SEC_COLLECTION_MANIFEST_SCHEMA or manifest.get("state") != "complete":
        raise DataReadinessError("SEC filing collection manifest is incomplete")
    if bool(manifest.get("production_ready")) or bool(authority.get("production_ready")):
        raise DataReadinessError("retrospective SEC collection cannot claim production readiness")
    if authority.get("schema") != SEC_COLLECTION_SCHEMA or authority.get("manifest_sha256") != file_sha256(manifest_path):
        raise DataReadinessError("SEC filing collection authority does not verify")
    request = manifest.get("request")
    if not isinstance(request, dict) or _json_sha256(request) != manifest.get("request_sha256"):
        raise DataReadinessError("SEC filing collection request hash does not verify")
    if (
        request.get("availability_policy") != "sec_daily_swing_conservative_proxy"
        or request.get("historical_availability_proven") is not False
    ):
        raise DataReadinessError("SEC retrospective availability contract does not verify")
    if _json_sha256(manifest.get("source_policy")) != manifest.get("source_policy_sha256"):
        raise DataReadinessError("SEC source policy hash does not verify")
    event_path = directory / "filing_events.parquet"
    coverage_path = directory / "source_collections.parquet"
    inventory_path = directory / "raw_response_inventory.parquet"
    events, event_manifest = load_canonical_artifact(event_path, expected_type=SEC_EVENT_ARTIFACT_TYPE, allow_research=True)
    coverage, coverage_manifest = load_canonical_artifact(coverage_path, expected_type=SEC_COVERAGE_ARTIFACT_TYPE, allow_research=True)
    inventory, inventory_manifest = load_canonical_artifact(
        inventory_path, expected_type=SEC_RAW_INVENTORY_ARTIFACT_TYPE, allow_research=True
    )
    coverage["error_type"] = coverage["error_type"].astype(object).where(coverage["error_type"].notna(), None)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise DataReadinessError("SEC filing artifact inventory is malformed")
    _verify_artifact(artifacts.get("events"), event_path, event_manifest, len(events))
    _verify_artifact(artifacts.get("source_collections"), coverage_path, coverage_manifest, len(coverage))
    _verify_artifact(artifacts.get("raw_inventory"), inventory_path, inventory_manifest, len(inventory))
    raw_record = artifacts.get("raw_archive")
    raw_path = directory / "raw_responses.zip"
    if (
        not isinstance(raw_record, dict)
        or raw_record.get("path") != raw_path.name
        or raw_record.get("sha256") != file_sha256(raw_path)
        or int(raw_record.get("bytes", -1)) != raw_path.stat().st_size
    ):
        raise DataReadinessError("SEC raw response archive does not verify")
    _verify_raw_archive(raw_path, inventory)
    for child in (event_manifest, coverage_manifest, inventory_manifest):
        inputs = child.get("inputs")
        if (
            not isinstance(inputs, dict)
            or inputs.get("collection_request_sha256") != manifest.get("request_sha256")
            or inputs.get("source_policy_sha256") != manifest.get("source_policy_sha256")
            or inputs.get("source_response_sha256") != manifest.get("source_response_sha256")
            or inputs.get("collector_schema") != SEC_COLLECTION_SCHEMA
        ):
            raise DataReadinessError("SEC child artifact lineage does not verify")
    ciks = tuple(str(value) for value in request.get("issuer_ciks", []))
    _validate_collection_content(events, coverage, ciks)
    if len(events) != _integer(manifest.get("event_rows"), "event_rows") or len(coverage) != _integer(
        manifest.get("issuer_count"), "issuer_count"
    ):
        raise DataReadinessError("SEC collection row lineage does not verify")
    return SecFilingCollection(directory, events, coverage, inventory, manifest, authority)


@dataclass(frozen=True, slots=True)
class _IssuerResult:
    cik: str
    ticker_hint: str
    started_at_utc: datetime
    completed_at_utc: datetime
    history: SecFilingHistory | None
    raw_responses: tuple[SecRawResponse, ...]
    submission_first_seen: tuple[tuple[str, datetime], ...]
    response_sha256: str | None
    error_type: str | None


def _collect_issuer(
    source_factory: Callable[[], SecHistorySource],
    cik: str,
    ticker_hint: str,
    start: datetime,
    end: datetime,
    forms: set[str],
    clock: Clock,
) -> _IssuerResult:
    started = _observed_now(clock, f"SEC collection start for {cik}")
    try:
        history = source_factory().fetch_cik_filing_history(cik, start, end, forms=forms, ticker_hint=ticker_hint)
    except SecSourceResponseError as exc:
        completed = _observed_now(clock, f"failed SEC collection for {cik}")
        if completed < started or completed < end:
            raise DataReadinessError("SEC collection clock ordering is invalid") from exc
        cause = exc.__cause__
        error_type = type(cause).__name__ if cause is not None else type(exc).__name__
        return _IssuerResult(
            cik,
            ticker_hint,
            started,
            completed,
            None,
            exc.raw_responses,
            (),
            _response_set_sha256(exc.raw_responses),
            error_type,
        )
    except Exception as exc:
        completed = _observed_now(clock, f"failed SEC collection for {cik}")
        if completed < started or completed < end:
            raise DataReadinessError("SEC collection clock ordering is invalid") from exc
        return _IssuerResult(cik, ticker_hint, started, completed, None, (), (), None, type(exc).__name__)
    completed = _observed_now(clock, f"SEC collection completion for {cik}")
    if completed < started or completed < end or history.cik != cik:
        raise DataReadinessError("SEC issuer collection identity or clock ordering is invalid")
    if not history.raw_responses:
        raise DataReadinessError("successful SEC issuer collection has no archived response")
    first_seen = tuple((name, _filing_first_seen(history, name)) for name in history.submission_files)
    return _IssuerResult(
        cik,
        ticker_hint,
        started,
        completed,
        history,
        history.raw_responses,
        first_seen,
        _response_set_sha256(history.raw_responses),
        None,
    )


def _event_frame(results: Sequence[_IssuerResult], *, lag_minutes: int) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for result in results:
        if result.history is None:
            continue
        first_seen_by_submission = dict(result.submission_first_seen)
        for filing in result.history.filings:
            accepted = filing.accepted_at_utc.astimezone(UTC)
            available, rule = conservative_sec_daily_swing_availability(accepted, filing.form, lag_minutes=lag_minutes)
            event = CanonicalEvent(
                event_id=_json_sha256({"source": "sec", "cik": filing.cik, "accession_number": filing.accession_number}),
                ticker=result.ticker_hint,
                security_id=f"cik:{filing.cik}",
                source_family=SEC_SOURCE_FAMILY,
                source=f"sec:{filing.form.lower()}",
                published_at_utc=accepted,
                first_seen_at_utc=first_seen_by_submission[filing.submission_file],
                available_at_utc=available,
                feature_available_at_utc=available,
                title=f"{result.ticker_hint} SEC {filing.form}",
                url=filing.document_url,
                summary=f"SEC filing {filing.form}, filed {filing.filing_date}",
                text=f"SEC issuer filing {filing.form}",
                availability_policy="provider_publication_proxy",
                raw_sha256=filing.raw_sha256,
            ).model_dump()
            event.update(
                {
                    "sec_cik": filing.cik,
                    "sec_form": filing.form,
                    "accession_number": filing.accession_number,
                    "filing_date": filing.filing_date,
                    "report_date": filing.report_date,
                    "primary_document": filing.primary_document,
                    "submission_file": filing.submission_file,
                    "file_number": filing.file_number,
                    "accepted_at_utc": accepted,
                    "is_amendment": filing.is_amendment,
                    "amends_accession_number": filing.amends_accession_number,
                    "availability_rule": rule,
                }
            )
            records.append(event)
    columns = [*CanonicalEvent.model_fields, *_EVENT_EXTRA_COLUMNS]
    return (
        pd.DataFrame.from_records(records, columns=columns)
        .sort_values(["sec_cik", "feature_available_at_utc", "event_id"], kind="stable")
        .reset_index(drop=True)
    )


def conservative_sec_daily_swing_availability(
    accepted_at_utc: datetime,
    form: str,
    *,
    lag_minutes: int = 5,
) -> tuple[datetime, str]:
    accepted = pd.Timestamp(accepted_at_utc).tz_convert("UTC")
    local = accepted.tz_convert(_ET)
    ownership = form.upper() in {"3", "3/A", "4", "4/A", "5", "5/A"}
    cutoff = time(22, 0) if ownership else time(17, 30)
    candidate = local + pd.Timedelta(minutes=lag_minutes)
    is_session = bool(_XNYS.is_session(local.date()))
    if is_session and candidate.time() <= cutoff:
        return candidate.tz_convert("UTC").to_pydatetime(), "acceptance_plus_safety_lag"
    session = _XNYS.date_to_session(local.date(), direction="next")
    if is_session:
        session = _XNYS.next_session(session)
    return _XNYS.session_open(session).to_pydatetime(), "late_submission_next_xnys_open"


def _filing_first_seen(history: SecFilingHistory, submission_file: str) -> datetime:
    observed = [
        response.retrieved_at_utc
        for response in history.raw_responses
        if Path(unquote(urlparse(response.final_url).path)).name == submission_file
    ]
    if not observed:
        raise DataReadinessError(f"SEC filing response evidence is missing for {submission_file}")
    return min(observed)


def _coverage_frame(results: Sequence[_IssuerResult], start: datetime, end: datetime, request_sha256: str) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for result in results:
        history = result.history
        rows = len(history.filings) if history is not None else 0
        status = "failed" if result.error_type else ("observed" if rows else "observed_empty")
        record = SourceCollection(
            collection_id=_json_sha256({"request_sha256": request_sha256, "sec_cik": result.cik}),
            ticker=result.ticker_hint,
            source_family=SEC_SOURCE_FAMILY,
            requested_start_utc=start,
            requested_end_utc=end,
            started_at_utc=result.started_at_utc,
            completed_at_utc=result.completed_at_utc,
            status=status,
            row_count=rows,
            error_type=result.error_type,
        ).model_dump()
        record.update(
            {
                "sec_cik": result.cik,
                "issuer_security_id": f"cik:{result.cik}",
                "company_name": history.company_name if history else None,
                "submission_files_json": json.dumps(history.submission_files) if history else None,
                "response_sha256": result.response_sha256,
                "source_row_count": history.source_row_count if history else None,
                "availability_evidence": "retrospective_raw_response_archive",
                "historical_availability_proven": False,
                "production_eligible": False,
            }
        )
        records.append(record)
    frame = (
        pd.DataFrame.from_records(records, columns=[*SourceCollection.model_fields, *_COVERAGE_EXTRA_COLUMNS])
        .sort_values("sec_cik", kind="stable")
        .reset_index(drop=True)
    )
    frame["error_type"] = frame["error_type"].astype(object).where(frame["error_type"].notna(), None)
    return frame


def _archive_raw_responses(
    archive: zipfile.ZipFile,
    records: list[dict[str, object]],
    result: _IssuerResult,
) -> int:
    for ordinal, response in enumerate(result.raw_responses):
        member = f"raw/{result.cik}/{ordinal:03d}-{response.body_sha256}.json"
        archive.writestr(member, response.body)
        records.append(_raw_inventory_record(result.cik, member, response, result.error_type))
    return len(result.raw_responses)


def _release_raw_bodies(result: _IssuerResult) -> _IssuerResult:
    history = replace(result.history, raw_responses=()) if result.history is not None else None
    return replace(result, history=history, raw_responses=())


def _raw_inventory_frame(records: list[dict[str, object]], *, expected_count: int) -> pd.DataFrame:
    inventory = (
        pd.DataFrame.from_records(
            records,
            columns=[
                "response_id",
                "sec_cik",
                "archive_member",
                "requested_url",
                "final_url",
                "status_code",
                "retrieved_at_utc",
                "content_type",
                "content_encoding",
                "etag",
                "last_modified",
                "body_sha256",
                "body_length",
                "safe_headers_json",
                "issuer_error_type",
            ],
        )
        .sort_values(["sec_cik", "retrieved_at_utc", "response_id"], kind="stable")
        .reset_index(drop=True)
    )
    if len(inventory) != expected_count:
        raise DataReadinessError("SEC raw response inventory count does not reconcile")
    return inventory


def _response_set_sha256(responses: Sequence[SecRawResponse]) -> str | None:
    if not responses:
        return None
    return _json_sha256(
        [
            {
                "response_id": response.response_id,
                "final_url": response.final_url,
                "retrieved_at_utc": response.retrieved_at_utc.isoformat(),
                "body_sha256": response.body_sha256,
            }
            for response in responses
        ]
    )


def _raw_inventory_record(
    cik: str,
    member: str,
    response: SecRawResponse,
    issuer_error_type: str | None,
) -> dict[str, object]:
    return {
        "response_id": response.response_id,
        "sec_cik": cik,
        "archive_member": member,
        "requested_url": response.requested_url,
        "final_url": response.final_url,
        "status_code": response.status_code,
        "retrieved_at_utc": response.retrieved_at_utc,
        "content_type": response.content_type,
        "content_encoding": response.content_encoding,
        "etag": response.etag,
        "last_modified": response.last_modified,
        "body_sha256": response.body_sha256,
        "body_length": response.body_length,
        "safe_headers_json": json.dumps(dict(response.safe_headers), sort_keys=True),
        "issuer_error_type": issuer_error_type,
    }


def _verify_raw_archive(path: Path, inventory: pd.DataFrame) -> None:
    expected = set(inventory["archive_member"].astype(str))
    with zipfile.ZipFile(path, "r") as archive:
        names = set(archive.namelist())
        if names != expected:
            raise DataReadinessError("SEC raw archive member inventory does not verify")
        for row in inventory.itertuples(index=False):
            name = str(row.archive_member)
            if name.startswith("/") or ".." in Path(name).parts:
                raise DataReadinessError("SEC raw archive contains an unsafe member")
            retrieved = pd.Timestamp(row.retrieved_at_utc)
            if retrieved.tzinfo is None:
                raise DataReadinessError("SEC raw response retrieval time is not timezone-aware")
            for raw_url in (row.requested_url, row.final_url):
                parsed_url = urlparse(str(raw_url))
                if parsed_url.scheme != "https" or parsed_url.hostname not in {"data.sec.gov", "www.sec.gov"}:
                    raise DataReadinessError("SEC raw response URL is not an approved SEC endpoint")
            try:
                headers = json.loads(str(row.safe_headers_json))
            except json.JSONDecodeError as exc:
                raise DataReadinessError("SEC raw response headers are not replayable") from exc
            if not isinstance(headers, dict) or any(
                not isinstance(key, str) or not isinstance(value, str) for key, value in headers.items()
            ):
                raise DataReadinessError("SEC raw response headers are malformed")
            expected_response_id = _json_sha256(
                {
                    "requested_url": str(row.requested_url),
                    "final_url": str(row.final_url),
                    "retrieved_at_utc": retrieved.isoformat(),
                    "sha256": str(row.body_sha256),
                }
            )
            if str(row.response_id) != expected_response_id or int(row.status_code) != 200:
                raise DataReadinessError("SEC raw response identity does not replay")
            body = archive.read(name)
            if len(body) != int(row.body_length) or hashlib.sha256(body).hexdigest() != str(row.body_sha256):
                raise DataReadinessError("SEC raw response body does not verify")
            if pd.notna(row.issuer_error_type):
                continue
            try:
                decoded = body
                encoding = str(row.content_encoding or "").lower()
                if encoding == "gzip":
                    decoded = gzip.decompress(body)
                elif encoding == "deflate":
                    decoded = zlib.decompress(body)
                if not isinstance(json.loads(decoded.decode("utf-8")), dict):
                    raise ValueError
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
                raise DataReadinessError("SEC archived response is not replayable JSON") from exc


def _validate_collection_content(events: pd.DataFrame, coverage: pd.DataFrame, ciks: Sequence[str]) -> None:
    expected = set(ciks)
    if set(coverage["sec_cik"].astype(str)) != expected or coverage["sec_cik"].duplicated().any():
        raise DataReadinessError("SEC coverage must contain exactly one row per requested issuer CIK")
    if not events.empty:
        if not set(events["sec_cik"].astype(str)).issubset(expected) or events["event_id"].astype(str).duplicated().any():
            raise DataReadinessError("SEC issuer event identity is invalid")
        if bool(events["security_id"].astype(str).ne("cik:" + events["sec_cik"].astype(str)).any()):
            raise DataReadinessError("SEC event CIK security identity is invalid")
        accepted = pd.to_datetime(events["accepted_at_utc"], utc=True)
        available = pd.to_datetime(events["feature_available_at_utc"], utc=True)
        if bool(available.lt(accepted).any()):
            raise DataReadinessError("SEC conservative availability precedes acceptance")
    counts = events.groupby("sec_cik").size().to_dict() if not events.empty else {}
    for row in coverage.to_dict(orient="records"):
        status = str(row["status"])
        if status in {"observed", "observed_empty"} and int(row["row_count"]) != int(counts.get(str(row["sec_cik"]), 0)):
            raise DataReadinessError("SEC coverage row_count does not reconcile with issuer events")
        if bool(row["historical_availability_proven"]) or bool(row["production_eligible"]):
            raise DataReadinessError("SEC retrospective coverage cannot claim production eligibility")


def _collection_identity_check(frame: pd.DataFrame, ciks: Sequence[str]) -> CanonicalAuditCheck:
    failures = int(set(frame["sec_cik"].astype(str)) != set(ciks)) + int(frame["sec_cik"].duplicated().sum())
    return CanonicalAuditCheck(
        name="sec_issuer_collection_identity",
        status="pass" if failures == 0 else "fail",
        failures=failures,
        rows_checked=len(frame),
        detail="each issuer CIK has exactly one isolated collection result",
    )


def _relation_sha256(frame: pd.DataFrame) -> str:
    records = []
    for row in frame.loc[:, list(_RELATION_COLUMNS)].to_dict(orient="records"):
        records.append({key: _json_compatible(value) for key, value in row.items()})
    return _json_sha256(records)


def _strict_utc(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if parsed.isna().any():
        raise DataReadinessError(f"SEC {name} contains invalid timestamps")
    return parsed


def _observed_now(clock: Clock, stage: str) -> datetime:
    value = clock()
    if value.tzinfo is None or value.utcoffset() is None:
        raise DataReadinessError(f"{stage} clock must be timezone-aware")
    return value.astimezone(UTC)


def _artifact_record(path: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": manifest["artifact_sha256"],
        "manifest_sha256": file_sha256(manifest_path_for(path)),
        "rows": manifest["rows"],
    }


def _verify_artifact(record: object, path: Path, manifest: Mapping[str, object], rows: int) -> None:
    if not isinstance(record, dict) or record != _artifact_record(path, manifest) or int(record.get("rows", -1)) != rows:
        raise DataReadinessError("SEC child artifact does not verify")


def _rewrite_artifact_path(path: Path, output_directory: Path) -> None:
    child_path = manifest_path_for(path)
    child = _json_object(child_path)
    child["artifact_path"] = str((output_directory / path.name).resolve())
    _atomic_json(child_path, child)


def _json_object(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"SEC JSON artifact is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"SEC JSON artifact must be an object: {path}")
    return {str(key): item for key, item in value.items()}


def _json_compatible(value: object) -> object:
    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(value)
        return None if pd.isna(timestamp) else timestamp.isoformat()
    if value is None or pd.isna(value):
        return None
    return value


def _json_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataReadinessError(f"SEC {name} is not an integer")
    return value


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
