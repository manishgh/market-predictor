"""Sequential swing SEC extension of a pinned archive; query hints are not ownership.

TOML [sec_incremental] requires archive_directory, archive_manifest_sha256,
identity_relations, identity_relations_sha256, output_root, extension_start.
Paths are repository-relative (not config-relative). Historical archive reuse is
byte verification only, never a load of its full filing-event table. Each new
issuer attempt has its own immutable request, collection and content-pinned result.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
import zipfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.sec_filings.collection import (
    SEC_COLLECTION_MANIFEST_SCHEMA,
    SEC_COLLECTION_SCHEMA,
    SecFilingCollectionConfig,
    _json_sha256,
    _relation_sha256,
    collect_historical_sec_filings,
    load_sec_filing_collection,
    load_sec_identity_relations,
    validate_sec_filing_collection_config,
)
from market_predictor.config import Settings
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.heavy_jobs import HEAVY_JOB_BUSY_EXIT_CODE, HeavyJobBusyError, heavy_job_lease, heavy_job_runtime_dir
from market_predictor.resources import assert_memory_budget
from market_predictor.sources.http import HttpClient
from market_predictor.sources.sec import SecRawResponse, SecRequestGovernor, SecSource, SecSourceResponseError, validate_sec_user_agent

SCHEMA = "market_predictor.sec_incremental.v1"
_DAY = timedelta(days=1)
_FILES = {
    "events": "filing_events.parquet",
    "source_collections": "source_collections.parquet",
    "raw_inventory": "raw_response_inventory.parquet",
    "raw_archive": "raw_responses.zip",
}


@dataclass(frozen=True)
class SecIncrementalConfig:
    archive_directory: Path
    archive_manifest_sha256: str
    identity_relations: Path
    identity_relations_sha256: str
    output_root: Path
    extension_start: date


def _object(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataReadinessError("SEC metadata must be an object")
    return cast(dict[str, Any], value)


def _read(path: Path) -> dict[str, Any]:
    try:
        return _object(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, ValueError) as exc:
        raise DataReadinessError(f"SEC metadata unreadable: {path}") from exc


def _pin(path: Path, expected: object) -> None:
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise DataReadinessError(f"SEC pin invalid: {path}")
    if not path.is_file() or file_sha256(path) != expected:
        raise DataReadinessError(f"SEC pin mismatch: {path}")


def _write(path: Path, payload: Mapping[str, object]) -> None:
    _publish_bytes(path, json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode())


def _publish_bytes(path: Path, encoded: bytes, *, replace: bool = False) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_pinned(directory: Path, name: str, payload: Mapping[str, object]) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False).encode()
    _publish_bytes(directory / f"{name}.{sha256(encoded).hexdigest()}.json", encoded)


def _read_pinned(path: Path) -> dict[str, Any]:
    _pin(path, path.name.split(".")[-2])
    return _read(path)


def load_config(path: Path, *, root: Path | None = None) -> SecIncrementalConfig:
    root = (root or Path.cwd()).resolve()
    try:
        section = _object(tomllib.loads(path.read_text(encoding="utf-8"))["sec_incremental"])
        expected = {
            "archive_directory", "archive_manifest_sha256", "identity_relations",
            "identity_relations_sha256", "output_root", "extension_start",
        }
        if set(section) != expected:
            raise DataReadinessError(f"SEC config requires exactly {sorted(expected)}")
        for key in ("archive_manifest_sha256", "identity_relations_sha256"):
            if re.fullmatch(r"[0-9a-f]{64}", str(section[key])) is None:
                raise DataReadinessError(f"SEC config invalid {key}")
        paths = {}
        for key in ("archive_directory", "identity_relations", "output_root"):
            if not isinstance(section[key], str) or not section[key].strip():
                raise DataReadinessError(f"SEC config invalid {key}")
            paths[key] = (root / section[key]).resolve()
        output = paths["output_root"]
        archive = paths["archive_directory"]
        if output == archive or output.is_relative_to(archive) or archive.is_relative_to(output):
            raise DataReadinessError("SEC extension output must be separate from immutable archive")
        return SecIncrementalConfig(
            archive, section["archive_manifest_sha256"], paths["identity_relations"],
            section["identity_relations_sha256"], output,
            date.fromisoformat(str(section["extension_start"])),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise DataReadinessError(f"SEC incremental config invalid: {path}") from exc


def guard_memory() -> None:
    assert_memory_budget(stage="SEC incremental", hard_budget_gib=4.0, headroom_gib=0.75)
    assert_system_memory_available(minimum_available_gib=2.0, maximum_used_percent=85.0)


class _RequestGate:
    def __init__(self, governor: SecRequestGovernor, guard: Callable[[], None]) -> None:
        self.governor = governor
        self.guard = guard
        self.pressure = False

    def __call__(self) -> None:
        try:
            self.guard()
            self.governor.acquire()
            self.guard()
        except MemoryBudgetError:
            self.pressure = True
            raise


def verify_archive(config: SecIncrementalConfig, *, guard: Callable[[], None] = guard_memory) -> dict[str, Any]:
    """Verify pinned manifest and child bytes in bounded chunks, not historical rows."""
    directory = config.archive_directory
    manifest_path = directory / "_manifest.json"
    _pin(manifest_path, config.archive_manifest_sha256)
    manifest = _read(manifest_path)
    authority = _read(directory / "_authority.json")
    request = _object(manifest.get("request"))
    if (
        manifest.get("schema") != SEC_COLLECTION_MANIFEST_SCHEMA or manifest.get("state") != "complete"
        or manifest.get("failed_issuers") != 0 or manifest.get("production_ready") is not False
        or authority.get("schema") != SEC_COLLECTION_SCHEMA
        or authority.get("manifest_sha256") != config.archive_manifest_sha256
        or authority.get("production_ready") is not False
        or _json_sha256(request) != manifest.get("request_sha256")
        or _json_sha256(manifest.get("source_policy")) != manifest.get("source_policy_sha256")
        or request.get("historical_availability_proven") is not False
        or request.get("availability_policy") != "sec_daily_swing_conservative_proxy"
    ):
        raise DataReadinessError("SEC archive request/authority/success contract does not verify")
    ciks = request.get("issuer_ciks")
    if (not isinstance(ciks, list) or not ciks or len(set(ciks)) != len(ciks)
            or any(not isinstance(cik, str) or re.fullmatch(r"[0-9]{10}", cik) is None for cik in ciks)
            or manifest.get("issuer_count") != len(ciks) or manifest.get("successful_issuers") != len(ciks)):
        raise DataReadinessError("SEC archive issuer inventory does not verify")
    if (
        request.get("requested_start_utc") != datetime.combine(date(2019, 7, 9), time.min, UTC).isoformat()
        or request.get("requested_end_utc") != datetime.combine(config.extension_start - _DAY, time.max, UTC).isoformat()
    ):
        raise DataReadinessError("SEC archive must cover 2019-07-09 through the day before extension_start")
    artifacts = _object(manifest.get("artifacts"))
    if set(artifacts) != set(_FILES):
        raise DataReadinessError("SEC archive artifact inventory does not verify")
    expected_files = {"_manifest.json", "_authority.json"}
    for key, filename in _FILES.items():
        guard()
        record = _object(artifacts[key])
        path = directory / filename
        expected_files.add(filename)
        if record.get("path") != filename:
            raise DataReadinessError("SEC archive artifact path does not verify")
        _pin(path, record.get("sha256"))
        if key == "raw_archive":
            if record.get("bytes") != path.stat().st_size:
                raise DataReadinessError("SEC archive byte count does not verify")
        else:
            sidecar = path.with_suffix(".parquet.manifest.json")
            expected_files.add(sidecar.name)
            _pin(sidecar, record.get("manifest_sha256"))
            child = _read(sidecar)
            if child.get("artifact_sha256") != record.get("sha256") or child.get("rows") != record.get("rows"):
                raise DataReadinessError("SEC archive child lineage does not verify")
    if {path.name for path in directory.iterdir()} != expected_files:
        raise DataReadinessError("SEC archive file inventory does not verify")
    return manifest


def _collection_config(start: date, end: date, archive: Mapping[str, Any]) -> SecFilingCollectionConfig:
    request = _object(archive["request"])
    return validate_sec_filing_collection_config(SecFilingCollectionConfig(
        start_date=start, end_date=end, forms=tuple(request["forms"]), max_workers=1,
        requests_per_second=2.0, forbidden_cooldown_seconds=600.0,
        rate_limit_cooldown_seconds=60.0, dissemination_lag_minutes=int(request["dissemination_lag_minutes"]),
    ))


def _expected_request(relations: pd.DataFrame, config: SecFilingCollectionConfig) -> dict[str, object]:
    return {
        "schema": SEC_COLLECTION_SCHEMA,
        "issuer_ciks": sorted(set(relations["sec_cik"].astype(str))),
        "identity_relation_sha256": _relation_sha256(relations),
        "requested_start_utc": datetime.combine(config.start_date, time.min, UTC).isoformat(),
        "requested_end_utc": datetime.combine(config.end_date, time.max, UTC).isoformat(),
        "forms": list(config.forms), "max_workers": 1, "requests_per_second": 2.0,
        "forbidden_cooldown_seconds": 600.0, "rate_limit_cooldown_seconds": 60.0,
        "dissemination_lag_minutes": config.dissemination_lag_minutes,
        "availability_policy": "sec_daily_swing_conservative_proxy",
        "historical_availability_proven": False, "production_ready": False,
    }


def _binding(config: SecIncrementalConfig) -> dict[str, object]:
    return {
        "schema": SCHEMA, "archive_manifest_sha256": config.archive_manifest_sha256,
        "identity_relations_sha256": config.identity_relations_sha256,
        "extension_start": config.extension_start.isoformat(), "identity_role": "CIK_query_hints_only",
        "current_ownership_claimed": False, "historical_availability_proven": False, "production_ready": False,
    }


def _replay(
    attempt: Path, request: Mapping[str, Any], expected: Mapping[str, object],
) -> bool:
    if request.get("collection_request") != expected:
        raise DataReadinessError("SEC completed request differs from current pinned request")
    receipts = list(attempt.glob("result.*.json"))
    if len(receipts) > 1:
        raise DataReadinessError("SEC attempt has conflicting results")
    receipt = _read_pinned(receipts[0]) if receipts else None
    if receipt is not None:
        if receipt.get("request_sha256") != file_sha256(attempt / "request.json"):
            raise DataReadinessError("SEC attempt request pin mismatch")
        manifest_pin = receipt.get("collection_manifest_sha256")
        if manifest_pin is None:
            if receipt.get("status") != "failed":
                raise DataReadinessError("SEC uncollected attempt claims success")
            return False
        _pin(attempt / "collection" / "_manifest.json", manifest_pin)
    elif not (attempt / "collection").exists():
        return False
    collection = load_sec_filing_collection(attempt / "collection")
    if collection.manifest.get("request") != expected:
        raise DataReadinessError("SEC replay collection request mismatch")
    success = (
        collection.manifest.get("failed_issuers") == 0
        and collection.manifest.get("successful_issuers") == 1
        and len(collection.source_collections) == 1
        and bool(collection.source_collections["error_type"].isna().all())
    )
    if receipt is None:
        _write_pinned(attempt, "result", {
            "request_sha256": file_sha256(attempt / "request.json"),
            "collection_manifest_sha256": file_sha256(attempt / "collection" / "_manifest.json"),
            "status": "complete" if success else "failed", "recovered_by_offline_replay": True,
        })
    elif receipt.get("status") != ("complete" if success else "failed"):
        raise DataReadinessError("SEC replay success accounting mismatch")
    return success


def _checkpoint(
    config: SecIncrementalConfig, relations: pd.DataFrame, archive: Mapping[str, Any],
    through: date, guard: Callable[[], None],
) -> date:
    cik = str(relations.iloc[0]["sec_cik"])
    completed = config.extension_start - _DAY
    intervals: list[tuple[date, date, Path, dict[str, Any]]] = []
    issuer_root = config.output_root / cik
    if not issuer_root.exists():
        return completed
    for attempt in issuer_root.iterdir():
        if not attempt.is_dir() or attempt.is_symlink():
            raise DataReadinessError("SEC attempt directory inventory invalid")
        if not (attempt / "request.json").exists():
            if any(not (child.name.startswith(".") and child.name.endswith(".tmp")) for child in attempt.iterdir()):
                raise DataReadinessError("SEC attempt missing request")
            continue
        request = _read(attempt / "request.json")
        if request.get("binding") != _binding(config):
            raise DataReadinessError("SEC attempt belongs to different pinned inputs")
        canonical = _object(request.get("collection_request"))
        start = datetime.fromisoformat(str(canonical.get("requested_start_utc"))).date()
        end = datetime.fromisoformat(str(canonical.get("requested_end_utc"))).date()
        if start < config.extension_start or end < start:
            raise DataReadinessError("SEC attempt dates invalid")
        intervals.append((start, end, attempt, request))
    for start, end, attempt, request in sorted(intervals, key=lambda item: (item[0], item[1], str(item[2]))):
        guard()
        expected = _expected_request(relations, _collection_config(start, end, archive))
        success = _replay(attempt, request, expected)
        if success and start <= completed + _DAY:
            completed = max(completed, end)
        elif success:
            raise DataReadinessError("SEC completed issuer history has a gap")
    return min(completed, through)


def _attempt(
    config: SecIncrementalConfig, relations: pd.DataFrame, collection_config: SecFilingCollectionConfig,
    requested_through: date, source: SecSource, gate: _RequestGate,
) -> bool:
    cik = str(relations.iloc[0]["sec_cik"])
    attempt = config.output_root / cik / uuid4().hex
    attempt.mkdir(parents=True, exist_ok=False)
    expected = _expected_request(relations, collection_config)
    request: dict[str, object] = {
        "binding": _binding(config), "collection_request": expected,
        "requested_through": requested_through.isoformat(),
        "partial_current_day": requested_through > collection_config.end_date,
    }
    _write(attempt / "request.json", request)
    receipt: dict[str, object] = {
        "request_sha256": file_sha256(attempt / "request.json"),
        "status": "failed", "collection_manifest_sha256": None,
    }
    try:
        collection = collect_historical_sec_filings(
            relations, attempt / "collection", source_factory=lambda: source, config=collection_config,
        )
        receipt["collection_manifest_sha256"] = file_sha256(attempt / "collection" / "_manifest.json")
        if collection.manifest.get("request") != expected:
            raise DataReadinessError("SEC collected request mismatch")
        if (collection.manifest.get("failed_issuers") == 0 and collection.manifest.get("successful_issuers") == 1
                and bool(collection.source_collections["error_type"].isna().all())):
            receipt["status"] = "complete"
        del collection
    except Exception as exc:
        # Do not persist exception text: upstream errors may contain credential-bearing URLs.
        receipt["error_type"] = type(exc).__name__
        if isinstance(exc, MemoryBudgetError):
            gate.pressure = True
    receipt["observed_at_utc"] = datetime.now(UTC).isoformat()
    receipt["paused_memory_pressure"] = gate.pressure
    _write_pinned(attempt, "result", receipt)
    return _replay(attempt, request, expected)


def _snapshot_request(config: SecIncrementalConfig, now: datetime, *, offline: bool) -> dict[str, Any] | None:
    # One immutable observation cutoff per UTC day, shared across bounded resumptions.
    directory = config.output_root / "snapshots" / now.date().isoformat()
    requests = list(directory.glob("request.*.json"))
    if len(requests) > 1:
        raise DataReadinessError("SEC current-day snapshot has conflicting cutoffs")
    if requests:
        request = _read_pinned(requests[0])
        cutoff = datetime.fromisoformat(str(request.get("capture_cutoff_utc")))
        if (request.get("binding") != _binding(config) or cutoff.tzinfo is None
                or cutoff.date() != now.date() or cutoff > now
                or request.get("partial_current_day") is not True
                or request.get("coverage_complete") is not False
                or request.get("advances_closed_day_checkpoint") is not False
                or request.get("requested_start_utc") != datetime.combine(now.date(), time.min, UTC).isoformat()):
            raise DataReadinessError("SEC current-day snapshot request does not verify")
        return request
    if offline:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    request = {
        "binding": _binding(config), "requested_start_utc": datetime.combine(now.date(), time.min, UTC).isoformat(),
        "capture_cutoff_utc": now.isoformat(), "partial_current_day": True,
        "coverage_complete": False, "advances_closed_day_checkpoint": False,
    }
    _write_pinned(directory, "request", request)
    return request


def _snapshot_replay(directory: Path, expected: Mapping[str, object]) -> bool:
    observed = False
    for attempt in directory.iterdir() if directory.exists() else ():
        if not attempt.is_dir() or attempt.is_symlink():
            raise DataReadinessError("SEC snapshot attempt inventory invalid")
        manifests = list(attempt.glob("manifest.*.json"))
        if not manifests:
            continue
        if len(manifests) != 1:
            raise DataReadinessError("SEC snapshot has conflicting manifests")
        manifest = _read_pinned(manifests[0])
        if manifest.get("request") != expected:
            raise DataReadinessError("SEC snapshot differs from current pinned request")
        _pin(attempt / "raw_responses.zip", manifest.get("raw_archive_sha256"))
        inventory = manifest.get("responses")
        if not isinstance(inventory, list):
            raise DataReadinessError("SEC snapshot response inventory invalid")
        with zipfile.ZipFile(attempt / "raw_responses.zip") as archive:
            if sorted(archive.namelist()) != sorted(row["member"] for row in inventory):
                raise DataReadinessError("SEC snapshot raw inventory mismatch")
            for row in inventory:
                guard_memory()
                digest = sha256()
                size = 0
                with archive.open(row["member"]) as body:
                    for chunk in iter(lambda: body.read(1024 * 1024), b""):
                        digest.update(chunk)
                        size += len(chunk)
                if digest.hexdigest() != row["body_sha256"] or size != row["body_length"]:
                    raise DataReadinessError("SEC snapshot raw response does not verify")
        success = manifest.get("status") == "partial_observed"
        if (manifest.get("status") not in {"partial_observed", "failed"}
                or manifest.get("coverage_complete") is not False
                or manifest.get("advances_closed_day_checkpoint") is not False
                or (success and (not inventory or manifest.get("error_type") is not None))):
            raise DataReadinessError("SEC snapshot observation status invalid")
        observed = observed or success
    return observed


def _snapshot(
    directory: Path, request: Mapping[str, Any], source: SecSource, gate: _RequestGate,
) -> bool:
    attempt = directory / uuid4().hex
    attempt.mkdir(parents=True, exist_ok=False)
    raw_responses: tuple[SecRawResponse, ...] = ()
    filings: list[dict[str, Any]] = []
    error_type: str | None = None
    source_rows: int | None = None
    try:
        history = source.fetch_cik_filing_history(
            request["cik"], datetime.fromisoformat(request["requested_start_utc"]),
            datetime.fromisoformat(request["capture_cutoff_utc"]),
            forms=set(request["forms"]), ticker_hint=request["ticker_hint"],
        )
        raw_responses = history.raw_responses
        if history.cik != request["cik"] or not raw_responses:
            raise DataReadinessError("SEC snapshot identity or archived response missing")
        for filing in history.filings:
            if not (datetime.fromisoformat(request["requested_start_utc"]) <= filing.accepted_at_utc
                    <= datetime.fromisoformat(request["capture_cutoff_utc"])):
                raise DataReadinessError("SEC snapshot filing outside observed window")
            filings.append({**asdict(filing), "accepted_at_utc": filing.accepted_at_utc.isoformat()})
        source_rows = history.source_row_count
        del history
    except Exception as exc:
        error_type = type(exc).__name__
        if isinstance(exc, SecSourceResponseError):
            raw_responses = exc.raw_responses
        if isinstance(exc, MemoryBudgetError):
            gate.pressure = True
        filings = []
    inventory = []
    with zipfile.ZipFile(attempt / "raw_responses.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for index, response in enumerate(raw_responses):
            member = f"{index:04d}-{response.body_sha256}.json"
            archive.writestr(member, response.body)
            metadata = {key: value for key, value in asdict(response).items() if key != "body"}
            metadata["retrieved_at_utc"] = response.retrieved_at_utc.isoformat()
            inventory.append({**metadata, "member": member})
    _write_pinned(attempt, "manifest", {
        "request": dict(request), "status": "partial_observed" if error_type is None else "failed",
        "error_type": error_type, "observed_at_utc": datetime.now(UTC).isoformat(),
        "raw_archive_sha256": file_sha256(attempt / "raw_responses.zip"), "responses": inventory,
        "filings": filings, "source_row_count": source_rows, "coverage_complete": False,
        "advances_closed_day_checkpoint": False, "historical_availability_proven": False,
        "source_content": "SEC submissions metadata, not filing document or exhibit text",
    })
    return _snapshot_replay(directory, request)


def run(
    config_path: Path, *, through: date | None = None, max_issuers: int | None = None, offline: bool = False,
    root: Path | None = None,
) -> dict[str, object]:
    """Hold the workspace lease before config, relation or archive input loading."""
    root = (root or Path.cwd()).resolve()
    config_path = root / config_path
    now = datetime.now(UTC)
    today = now.date()
    requested = through if through is not None else today - _DAY
    if requested > today or (max_issuers is not None and max_issuers < 1):
        raise DataReadinessError("SEC through must not be future; max_issuers must be positive")
    runtime = heavy_job_runtime_dir() if os.environ.get("MARKET_PREDICTOR_RUNTIME_DIR", "").strip() else root / "data/runtime"
    with heavy_job_lease("sec-incremental", runtime_dir=runtime):
        guard_memory()
        config = load_config(config_path, root=root)
        archive = verify_archive(config, guard=guard_memory)
        _pin(config.identity_relations, config.identity_relations_sha256)
        relations = load_sec_identity_relations(config.identity_relations)
        _pin(config.identity_relations, config.identity_relations_sha256)
        ciks = sorted(set(relations["sec_cik"].astype(str)))
        if ciks != sorted(archive["request"]["issuer_ciks"]):
            raise DataReadinessError("SEC relations must retain exactly the archived CIK query universe")
        if requested < config.extension_start - _DAY:
            raise DataReadinessError("SEC through precedes archive end")
        closed_through = min(requested, today - _DAY)
        snapshot_request = _snapshot_request(config, now, offline=offline) if requested == today else None
        governor = SecRequestGovernor(requests_per_second=2.0)
        gate = _RequestGate(governor, guard_memory)
        client: HttpClient | None = None
        source: SecSource | None = None
        attempted = 0
        results: list[dict[str, object]] = []
        try:
            for cik in ciks:
                guard_memory()
                subset = relations.loc[relations["sec_cik"].eq(cik)].reset_index(drop=True)
                completed = _checkpoint(config, subset, archive, closed_through, guard_memory)
                status = "verified"
                snapshot_expected = {
                    **(snapshot_request or {}), "cik": cik, "ticker_hint": str(subset["ticker"].min()),
                    "identity_relation_sha256": _relation_sha256(subset), "forms": archive["request"]["forms"],
                }
                snapshot_directory = config.output_root / "snapshots" / requested.isoformat() / cik
                observed = _snapshot_replay(snapshot_directory, snapshot_expected) if snapshot_request else False
                needs_snapshot = requested == today and not observed
                if completed < closed_through or needs_snapshot:
                    status = "pending" if completed < closed_through else "verified"
                    if not offline and (max_issuers is None or attempted < max_issuers):
                        if source is None:
                            settings = Settings()
                            client = HttpClient(
                                user_agent=validate_sec_user_agent(settings.sec_user_agent), before_request=gate,
                                after_response=governor.observe_response, additional_retriable_statuses=frozenset({403}),
                            )
                            source = SecSource(settings, governor=governor, client=client)
                        attempted += 1
                        if completed < closed_through:
                            success = _attempt(
                                config, subset, _collection_config(completed + _DAY, closed_through, archive),
                                requested, source, gate,
                            )
                            status = "collected" if success else "failed"
                            if success:
                                completed = closed_through
                        if needs_snapshot and not gate.pressure:
                            observed = _snapshot(snapshot_directory, snapshot_expected, source, gate)
                results.append({
                    "cik": cik, "status": status, "fully_successful_through": completed.isoformat(),
                    "current_day_snapshot_observed": observed,
                })
                if gate.pressure:
                    break
        finally:
            if client is not None:
                client.session.close()
        all_closed = len(results) == len(ciks) and all(
            row["fully_successful_through"] == closed_through.isoformat() for row in results
        )
        partial = requested == today
        all_observed = partial and len(results) == len(ciks) and all(
            row["current_day_snapshot_observed"] for row in results
        )
        status = "partial"
        if gate.pressure:
            status = "paused_memory_pressure"
        elif all_closed:
            if not partial:
                status = "complete"
            elif all_observed:
                status = "snapshot_collected"
        report = {
            **_binding(config), "requested_through": requested.isoformat(), "offline": offline,
            "status": status,
            "partial_current_day": partial,
            "current_day_collected": all_observed,
            "capture_cutoff_utc": snapshot_request["capture_cutoff_utc"] if snapshot_request else None,
            "requested_window_fully_collected": all_closed and not partial,
            "coverage_complete": False, "archive_reuse_verification": "pinned_metadata_and_file_bytes_only",
            "source_content": "SEC submissions filing metadata; not filing document or exhibit text",
            "issuer_count": len(ciks), "attempted_issuers": attempted, "issuers": results,
        }
        runs = config.output_root / "_runs"
        runs.mkdir(parents=True, exist_ok=True)
        report_path = runs / f"{uuid4().hex}.json"
        report["report_path"] = str(report_path)
        _write(report_path, report)
        _publish_bytes(config.output_root / "latest_status.json", json.dumps(
            report, sort_keys=True, indent=2, allow_nan=False,
        ).encode(), replace=True)
        return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=None, help="Repository root; defaults to the current working directory")
    parser.add_argument("--through", type=date.fromisoformat, help="Inclusive UTC date; default yesterday UTC")
    parser.add_argument("--max-issuers", type=int, help="Maximum new issuer attempts, excluding verified skips")
    parser.add_argument("--offline", action="store_true", help="Verify archive pins and replay attempts without HTTP")
    args = parser.parse_args(argv)
    try:
        result = run(args.config, root=args.root, through=args.through, max_issuers=args.max_issuers, offline=args.offline)
        print(json.dumps({key: value for key, value in result.items() if key != "issuers"}, sort_keys=True, indent=2))
        return 0 if result["status"] in {"complete", "snapshot_collected"} else 75 if result["status"] == "paused_memory_pressure" else 2
    except HeavyJobBusyError as exc:
        print(str(exc), file=sys.stderr)
        return HEAVY_JOB_BUSY_EXIT_CODE
    except MemoryBudgetError as exc:
        print(str(exc), file=sys.stderr)
        return 75
    except (DataReadinessError, OSError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
