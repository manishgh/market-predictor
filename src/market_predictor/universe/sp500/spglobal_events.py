from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any, cast

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256 as _json_sha256
from market_predictor.locking import LockTimeout, file_lock
from market_predictor.resources import assert_memory_budget, assert_peak_memory_budget
from market_predictor.universe.sp500.spglobal_archive import (
    MAXIMUM_MEMORY_GIB,
    MEMORY_HEADROOM_GIB,
    VerifiedSpGlobalRawArchive,
    read_verified_spglobal_release_html,
    require_spglobal_raw_archive_complete,
)
from market_predictor.universe.sp500.universe import (
    IndexChange,
    IndexChangeSource,
    VerifiedIndexChanges,
    parse_sp500_changes,
)

EVENT_REQUEST_SCHEMA = "ml_v3.spglobal_event_extraction_request.v1"

EVENT_MANIFEST_SCHEMA = "ml_v3.spglobal_event_extraction_manifest.v1"

EVENT_AUTHORITY_SCHEMA = "ml_v3.spglobal_event_extraction_authority.v1"

EVENT_PARSER_SCHEMA = "ml_v3.spglobal_membership_event_parser.v1"

def extract_spglobal_events(
    *,
    archive_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Parse a verified raw archive offline and publish event authority."""

    archive_root = archive_directory.resolve()
    output_root = output_directory.resolve()
    if archive_root == output_root or archive_root in output_root.parents or output_root in archive_root.parents:
        raise DataReadinessError("S&P raw archive and event output directories must be disjoint")
    output_directory.mkdir(parents=True, exist_ok=True)
    try:
        with file_lock(output_directory / "_extractor", timeout=0.0):
            return _extract_spglobal_events_locked(
                archive_directory=archive_directory,
                output_directory=output_directory,
            )
    except LockTimeout as exc:
        raise DataReadinessError(f"another process is writing S&P events {output_directory}") from exc

def _extract_spglobal_events_locked(
    *,
    archive_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    assert_memory_budget(
        hard_budget_gib=MAXIMUM_MEMORY_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage="S&P event extraction start",
    )
    if (output_directory / "_authority.json").exists():
        raise DataReadinessError("completed S&P event extraction is immutable")
    archive = require_spglobal_raw_archive_complete(archive_directory)
    raw_authority_sha256 = _file_sha256(archive.root / "_authority.json")
    raw_manifest_sha256 = str(archive.authority["artifact_sha256"])
    request_payload = {
        "schema": EVENT_REQUEST_SCHEMA,
        "parser_schema": EVENT_PARSER_SCHEMA,
        "raw_authority_sha256": raw_authority_sha256,
        "raw_manifest_sha256": raw_manifest_sha256,
        "raw_release_set_sha256": archive.manifest["release_set_sha256"],
    }
    request_sha256 = _json_sha256(request_payload)
    request = {**request_payload, "request_sha256": request_sha256}
    _write_or_validate_json(output_directory / "_request.json", request)

    assertions, outcomes, unresolved = _parse_verified_archive(archive)
    events, conflicts = _canonical_events(assertions)
    assertions_path = output_directory / "assertions.json"
    outcomes_path = output_directory / "release_outcomes.json"
    events_path = output_directory / "events.json"
    _atomic_json(assertions_path, assertions)
    _atomic_json(outcomes_path, outcomes)
    _atomic_json(events_path, events)
    status = "complete" if not unresolved and not conflicts else "blocked"
    manifest: dict[str, Any] = {
        "schema": EVENT_MANIFEST_SCHEMA,
        "status": status,
        "request_sha256": request_sha256,
        "parser_schema": EVENT_PARSER_SCHEMA,
        "raw_authority_sha256": raw_authority_sha256,
        "raw_manifest_sha256": raw_manifest_sha256,
        "raw_release_set_sha256": archive.manifest["release_set_sha256"],
        "release_count": len(archive.releases),
        "parsed_release_count": sum(item["disposition"] == "parsed" for item in outcomes),
        "no_effective_event_release_count": sum(item["disposition"] == "no_effective_event" for item in outcomes),
        "unresolved_release_count": len(unresolved),
        "unresolved_releases": unresolved,
        "assertion_count": len(assertions),
        "event_count": len(events),
        "duplicate_support_count": _duplicate_support_count(assertions),
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "artifacts": {
            "assertions": _artifact_record(assertions_path),
            "release_outcomes": _artifact_record(outcomes_path),
            "events": _artifact_record(events_path),
        },
    }
    manifest_path = output_directory / "_manifest.json"
    _atomic_json(manifest_path, manifest)
    _atomic_json(output_directory / "_status.json", manifest)
    assert_peak_memory_budget(
        hard_budget_gib=MAXIMUM_MEMORY_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage="S&P event extraction publication",
    )
    if status == "complete":
        authority = {
            "schema": EVENT_AUTHORITY_SCHEMA,
            "state": "event_complete",
            "artifact": "_manifest.json",
            "artifact_sha256": _file_sha256(manifest_path),
            "request_sha256": request_sha256,
            "parser_schema": EVENT_PARSER_SCHEMA,
            "raw_authority_sha256": raw_authority_sha256,
            "raw_manifest_sha256": raw_manifest_sha256,
            "event_count": len(events),
            "event_set_sha256": _json_sha256(events),
        }
        _atomic_json(output_directory / "_authority.json", authority)
    return manifest

def require_spglobal_event_reconstruction_ready(
    event_directory: Path,
    *,
    archive_directory: Path,
) -> VerifiedIndexChanges:
    """Verify event authority, all artifacts, and its raw-source parent."""

    archive = require_spglobal_raw_archive_complete(archive_directory)
    authority_value = _load_json(event_directory / "_authority.json")
    if not isinstance(authority_value, dict) or not all(isinstance(key, str) for key in authority_value):
        raise DataReadinessError("S&P event authority is not a JSON object")
    authority = cast(dict[str, Any], authority_value)
    artifact = _resolve_inside(
        event_directory,
        str(authority.get("artifact", "")),
    )
    if (
        authority.get("schema") != EVENT_AUTHORITY_SCHEMA
        or authority.get("state") != "event_complete"
        or not artifact.is_file()
        or authority.get("artifact_sha256") != _file_sha256(artifact)
    ):
        raise DataReadinessError("S&P event authority is invalid")
    manifest = _load_json(artifact)
    raw_authority_sha256 = _file_sha256(archive.root / "_authority.json")
    request = _load_json(event_directory / "_request.json")
    expected_request_payload = {
        "schema": EVENT_REQUEST_SCHEMA,
        "parser_schema": EVENT_PARSER_SCHEMA,
        "raw_authority_sha256": raw_authority_sha256,
        "raw_manifest_sha256": archive.authority["artifact_sha256"],
        "raw_release_set_sha256": archive.manifest["release_set_sha256"],
    }
    expected_request_sha256 = _json_sha256(expected_request_payload)
    if (
        request
        != {
            **expected_request_payload,
            "request_sha256": expected_request_sha256,
        }
        or manifest.get("schema") != EVENT_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("parser_schema") != EVENT_PARSER_SCHEMA
        or manifest.get("request_sha256") != expected_request_sha256
        or authority.get("request_sha256") != expected_request_sha256
        or manifest.get("raw_authority_sha256") != raw_authority_sha256
        or authority.get("raw_authority_sha256") != raw_authority_sha256
        or manifest.get("raw_manifest_sha256") != archive.authority.get("artifact_sha256")
        or authority.get("raw_manifest_sha256") != archive.authority.get("artifact_sha256")
        or manifest.get("raw_release_set_sha256") != archive.manifest.get("release_set_sha256")
        or int(manifest.get("release_count", -1)) != len(archive.releases)
        or int(manifest.get("unresolved_release_count", -1)) != 0
        or int(manifest.get("conflict_count", -1)) != 0
    ):
        raise DataReadinessError("S&P event manifest lineage or readiness is invalid")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise DataReadinessError("S&P event artifact inventory is invalid")
    loaded: dict[str, list[Any]] = {}
    for name in ("assertions", "release_outcomes", "events"):
        record = artifacts.get(name)
        if not isinstance(record, dict):
            raise DataReadinessError("S&P event artifact record is invalid")
        path = _resolve_inside(event_directory, str(record.get("path", "")))
        if not path.is_file() or record.get("sha256") != _file_sha256(path) or int(record.get("bytes", -1)) != path.stat().st_size:
            raise DataReadinessError(f"S&P event artifact failed: {name}")
        value = _load_json(path)
        if not isinstance(value, list):
            raise DataReadinessError(f"S&P event artifact is not a list: {name}")
        loaded[name] = value
    events = loaded["events"]
    replayed_assertions, replayed_outcomes, unresolved = _parse_verified_archive(archive)
    replayed_events, conflicts = _canonical_events(replayed_assertions)
    if (
        unresolved
        or conflicts
        or loaded["assertions"] != replayed_assertions
        or loaded["release_outcomes"] != replayed_outcomes
        or events != replayed_events
        or int(manifest.get("assertion_count", -1)) != len(loaded["assertions"])
        or int(manifest.get("event_count", -1)) != len(events)
        or int(manifest.get("parsed_release_count", -1)) != sum(item["disposition"] == "parsed" for item in replayed_outcomes)
        or int(manifest.get("no_effective_event_release_count", -1))
        != sum(item["disposition"] == "no_effective_event" for item in replayed_outcomes)
        or int(manifest.get("duplicate_support_count", -1)) != _duplicate_support_count(replayed_assertions)
        or int(authority.get("event_count", -1)) != len(events)
        or authority.get("event_set_sha256") != _json_sha256(events)
    ):
        raise DataReadinessError("S&P event artifact counts or identity are invalid")
    verified_changes: list[IndexChange] = []
    for event in events:
        if not isinstance(event, dict):
            raise DataReadinessError("S&P canonical event record is invalid")
        sources = event.get("supporting_sources")
        companies = event.get("companies")
        if (
            not isinstance(sources, list)
            or not sources
            or not isinstance(sources[0], dict)
            or not isinstance(companies, list)
            or not companies
        ):
            raise DataReadinessError("S&P canonical event evidence is invalid")
        if not all(isinstance(source, dict) for source in sources):
            raise DataReadinessError("S&P canonical event source evidence is invalid")
        supporting_sources = tuple(
            IndexChangeSource(
                source_url=str(source["source_url"]),
                source_published_date=date.fromisoformat(str(source["source_published_date"])),
                source_sha256=str(source["source_sha256"]),
            )
            for source in sources
        )
        source = supporting_sources[0]
        verified_changes.append(
            IndexChange(
                effective_at_utc=datetime.fromisoformat(str(event["effective_at_utc"])),
                action=str(event["action"]),
                ticker=str(event["ticker"]),
                company=str(companies[0]),
                sector=str(event["sector"]),
                source_url=source.source_url,
                source_published_date=source.source_published_date,
                source_sha256=source.source_sha256,
                supporting_sources=supporting_sources,
            )
        )
    return VerifiedIndexChanges(
        changes=tuple(verified_changes),
        authority_sha256=_file_sha256(event_directory / "_authority.json"),
        event_set_sha256=str(authority["event_set_sha256"]),
    )

def _parse_verified_archive(
    archive: VerifiedSpGlobalRawArchive,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, str]],
]:
    assertions: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    unresolved: list[dict[str, str]] = []
    for record in sorted(archive.releases, key=lambda item: str(item["url"])):
        try:
            changes = parse_sp500_changes(
                read_verified_spglobal_release_html(archive, record),
                source_url=str(record["url"]),
                published_date=date.fromisoformat(str(record["published_date"])),
                source_sha256=str(record["sha256"]),
            )
        except Exception as exc:  # noqa: BLE001 - unresolved evidence must be published
            error = f"{type(exc).__name__}: {exc}"
            unresolved.append({"source_url": str(record["url"]), "error": error})
            outcomes.append(
                _release_outcome(
                    record,
                    disposition="parser_unresolved",
                    count=0,
                    error=error,
                )
            )
            continue
        if any(change.source_published_date > change.effective_at_utc.date() for change in changes):
            error = "DataReadinessError: S&P event effective date precedes its official publication date"
            unresolved.append({"source_url": str(record["url"]), "error": error})
            outcomes.append(
                _release_outcome(
                    record,
                    disposition="parser_unresolved",
                    count=0,
                    error=error,
                )
            )
            continue
        assertions.extend(change.to_record() for change in changes)
        outcomes.append(
            _release_outcome(
                record,
                disposition="parsed" if changes else "no_effective_event",
                count=len(changes),
                error=None,
            )
        )
    assertions.sort(key=_assertion_sort_key)
    return assertions, outcomes, unresolved

def _release_outcome(
    record: Mapping[str, Any],
    *,
    disposition: str,
    count: int,
    error: str | None,
) -> dict[str, Any]:
    return {
        "source_url": record["url"],
        "published_date": record["published_date"],
        "source_sha256": record["sha256"],
        "disposition": disposition,
        "event_assertion_count": count,
        "error": error,
    }

def _canonical_events(
    assertions: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    actions_by_identity: dict[tuple[str, str], set[str]] = defaultdict(set)
    for assertion in assertions:
        key = (
            str(assertion["effective_at_utc"]),
            str(assertion["action"]),
            str(assertion["ticker"]),
        )
        grouped[key].append(assertion)
        actions_by_identity[(key[0], key[2])].add(key[1])
    conflicts: list[dict[str, Any]] = []
    for (effective_at, ticker), actions in sorted(actions_by_identity.items()):
        if len(actions) > 1:
            conflicts.append(
                {
                    "type": "opposite_actions_same_effective_time",
                    "effective_at_utc": effective_at,
                    "ticker": ticker,
                    "actions": sorted(actions),
                }
            )
    timeline_by_ticker: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for effective_at, action, ticker in grouped:
        timeline_by_ticker[ticker].append((effective_at, action))
    for ticker, timeline in sorted(timeline_by_ticker.items()):
        ordered = sorted(timeline)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous[1] == current[1]:
                conflicts.append(
                    {
                        "type": "non_alternating_membership_actions",
                        "ticker": ticker,
                        "action": current[1],
                        "previous_effective_at_utc": previous[0],
                        "effective_at_utc": current[0],
                    }
                )
    events: list[dict[str, Any]] = []
    for key, support in sorted(grouped.items()):
        sectors = {str(item["sector"]).strip() for item in support if str(item["sector"]).strip()}
        if len(sectors) > 1:
            conflicts.append(
                {
                    "type": "sector_disagreement",
                    "effective_at_utc": key[0],
                    "action": key[1],
                    "ticker": key[2],
                    "sectors": sorted(sectors),
                }
            )
            continue
        evidence = sorted(
            (
                {
                    "source_url": item["source_url"],
                    "source_published_date": item["source_published_date"],
                    "source_sha256": item["source_sha256"],
                    "company": item["company"],
                    "sector": item["sector"],
                }
                for item in support
            ),
            key=lambda item: (item["source_url"], item["source_sha256"]),
        )
        event_identity = {
            "effective_at_utc": key[0],
            "action": key[1],
            "ticker": key[2],
        }
        events.append(
            {
                "event_id": _json_sha256(event_identity),
                **event_identity,
                "companies": sorted({str(item["company"]) for item in support}),
                "sector": next(iter(sectors), ""),
                "support_count": len(evidence),
                "supporting_sources": evidence,
            }
        )
    return events, conflicts

def _duplicate_support_count(assertions: list[dict[str, Any]]) -> int:
    identities = {
        (
            str(assertion["effective_at_utc"]),
            str(assertion["action"]),
            str(assertion["ticker"]),
        )
        for assertion in assertions
    }
    return len(assertions) - len(identities)

def _assertion_sort_key(value: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(value["effective_at_utc"]),
        str(value["action"]),
        str(value["ticker"]),
        str(value["source_url"]),
    )

def _artifact_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _file_sha256(path),
        "bytes": path.stat().st_size,
    }

def _write_or_validate_json(path: Path, value: object) -> None:
    if path.exists():
        if _load_json(path) != value:
            raise DataReadinessError(f"S&P event request conflicts with {path}")
        return
    _atomic_json(path, value)

def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)

def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    resolved_root = root.resolve()
    if candidate != resolved_root and resolved_root not in candidate.parents:
        raise DataReadinessError(f"S&P event path escapes output directory: {relative}")
    return candidate

def _load_json(path: Path) -> Any:
    if not path.is_file():
        raise DataReadinessError(f"S&P event JSON is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"S&P event JSON is invalid: {path}") from exc

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
