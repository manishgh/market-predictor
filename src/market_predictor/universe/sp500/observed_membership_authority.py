"""Observed-time S&P membership authority for prospective weekday collection."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, cast
from zoneinfo import ZoneInfo

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditReport, audit_universe_memberships
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.locking import LockTimeout, file_lock
from market_predictor.resources import assert_memory_budget, assert_peak_memory_budget
from market_predictor.sources.spglobal import observed_membership_collection as observed_source
from market_predictor.sources.spglobal.archive import (
    ARCHIVE_QUERY,
    MAXIMUM_MEMORY_GIB,
    MEMORY_HEADROOM_GIB,
    SP_GLOBAL_ARCHIVE_URL,
    SpGlobalAnnouncement,
)
from market_predictor.universe.sp500.index_change_events import (
    require_spglobal_event_reconstruction_ready,
)
from market_predictor.universe.sp500.membership_authority import (
    load_sp500_membership_authority_envelope,
    verify_membership_namespace_extension,
)
from market_predictor.universe.sp500.membership_history import (
    SECTOR_BENCHMARKS,
    IndexChange,
    parse_sp500_changes,
)

REQUEST_SCHEMA: Final = "edge_rebuild.sp500_observed_membership_request.v2"
MANIFEST_SCHEMA: Final = "edge_rebuild.sp500_observed_membership_manifest.v2"
AUTHORITY_SCHEMA: Final = "edge_rebuild.sp500_observed_membership_authority.v2"
MEMBERSHIP_FILE: Final = "memberships.parquet"
ANCHOR_FILE: Final = "current_anchor.csv"
EVENT_FILE: Final = "observed_events.json"
OUTCOME_FILE: Final = "release_outcomes.json"
PENDING_FILE: Final = "pending_changes.json"
NEW_YORK: Final = ZoneInfo("America/New_York")
MEMBERSHIP_SCHEMA_VERSION: Final = "market_data.v1"
OBSERVED_AVAILABILITY_POLICY: Final = "observed"
OFFICIAL_MEMBERSHIP_SOURCE: Final = "spglobal_official_point_in_time"
OBSERVED_IDENTITY_SOURCE: Final = "spglobal_current_anchor_sec_identity"

@dataclass(frozen=True, slots=True)
class ObservedMembershipConfig:
    maximum_pages: int = 5
    retries: int = 3
    retry_pause_seconds: float = 1.0

    def validate(self) -> None:
        if not 1 <= self.maximum_pages <= 20:
            raise ValueError("maximum_pages must be between 1 and 20")
        if not 1 <= self.retries <= 10:
            raise ValueError("retries must be between 1 and 10")
        if not 0 <= self.retry_pause_seconds <= 120:
            raise ValueError("retry_pause_seconds must be between 0 and 120")


@dataclass(frozen=True, slots=True)
class ObservedMembershipAuthority:
    directory: Path
    memberships: pd.DataFrame
    manifest: Mapping[str, object]
    parent: Mapping[str, object]


def collect_observed_sp500_membership_authority(
    *,
    base_membership_directory: Path,
    closed_archive_directory: Path,
    closed_event_directory: Path,
    output_directory: Path,
    client_factory: observed_source.ClientFactory | None = None,
    config: ObservedMembershipConfig | None = None,
) -> dict[str, object]:
    """Collect exact observations and publish one immutable effective-state authority."""

    policy = config or ObservedMembershipConfig()
    policy.validate()
    output = output_directory.resolve()
    parents = (
        base_membership_directory.resolve(),
        closed_archive_directory.resolve(),
        closed_event_directory.resolve(),
    )
    for parent in parents:
        if output == parent or output in parent.parents or parent in output.parents:
            raise DataReadinessError("observed membership output and parents must be disjoint")
    output.mkdir(parents=True, exist_ok=True)
    try:
        with file_lock(output / "_collector", timeout=0.0):
            return _collect_locked(
                base_membership_directory=parents[0],
                closed_archive_directory=parents[1],
                closed_event_directory=parents[2],
                output_directory=output,
                client_factory=client_factory,
                config=policy,
            )
    except LockTimeout as exc:
        raise DataReadinessError(f"another process owns observed membership {output}") from exc


def _collect_locked(
    *,
    base_membership_directory: Path,
    closed_archive_directory: Path,
    closed_event_directory: Path,
    output_directory: Path,
    client_factory: observed_source.ClientFactory | None,
    config: ObservedMembershipConfig,
) -> dict[str, object]:
    assert_memory_budget(
        hard_budget_gib=MAXIMUM_MEMORY_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage="observed S&P membership collection start",
    )
    if (output_directory / "_authority.json").exists():
        return dict(load_observed_sp500_membership_authority(output_directory).manifest)
    base, base_parent = load_sp500_membership_authority_envelope(base_membership_directory)
    closed_events = require_spglobal_event_reconstruction_ready(
        closed_event_directory,
        archive_directory=closed_archive_directory,
    )
    base_request = _json_object(base_membership_directory / "_request.json")
    base_lineage = base_request.get("parent_lineage")
    if (
        not isinstance(base_lineage, Mapping)
        or base_lineage.get("event_authority_sha256") != closed_events.authority_sha256
        or base_lineage.get("event_set_sha256") != closed_events.event_set_sha256
    ):
        raise DataReadinessError("closed event authority is not the base membership parent")
    base_cutoff = date.fromisoformat(str(base_parent["cutoff_date"]))
    request_payload: dict[str, object] = {
        "schema": REQUEST_SCHEMA,
        "base_membership_directory": str(base_membership_directory),
        "base_membership_authority_sha256": base_parent["authority_sha256"],
        "base_membership_manifest_sha256": base_parent["manifest_sha256"],
        "base_membership_table_sha256": base_parent["membership_table_sha256"],
        "base_membership_universe_sha256": base_parent["universe_sha256"],
        "closed_archive_directory": str(closed_archive_directory),
        "closed_event_directory": str(closed_event_directory),
        "closed_event_authority_sha256": closed_events.authority_sha256,
        "closed_event_set_sha256": closed_events.event_set_sha256,
        "closed_cutoff_date": base_cutoff.isoformat(),
        "official_archive_url": SP_GLOBAL_ARCHIVE_URL,
        "official_archive_query": dict(sorted(ARCHIVE_QUERY.items())),
        "independent_anchor_url": observed_source.ANCHOR_URL,
        "independent_identity_url": observed_source.SEC_IDENTITY_URL,
        "maximum_pages": config.maximum_pages,
    }
    request_sha256 = _json_sha256(request_payload)
    _write_new_json(
        output_directory / "_request.json",
        {**request_payload, "request_sha256": request_sha256},
    )
    client = (client_factory or observed_source.default_client_factory)()
    first_pages, announcements, page_identities = observed_source.collect_official_release_prefix(
        client,
        output_directory=output_directory,
        base_cutoff=base_cutoff,
        config=config,
        request_sha256=request_sha256,
    )
    release_units = observed_source.collect_release_documents(
        client,
        output_directory=output_directory,
        announcements=announcements,
        config=config,
        request_sha256=request_sha256,
        role="release",
    )
    anchor_unit = observed_source.fetch_observation_unit(
        client,
        url=observed_source.ANCHOR_URL,
        params=None,
        role="anchor",
        unit_id="independent-anchor",
        output_directory=output_directory,
        request_sha256=request_sha256,
        config=config,
        expected_host="en.wikipedia.org",
        exact_path="/wiki/List_of_S%26P_500_companies",
    )
    identity_unit = observed_source.fetch_observation_unit(
        client,
        url=observed_source.SEC_IDENTITY_URL,
        params=None,
        role="identity_anchor",
        unit_id="independent-identity-anchor",
        output_directory=output_directory,
        request_sha256=request_sha256,
        config=config,
        expected_host="www.sec.gov",
        exact_path="/files/company_tickers.json",
    )
    anchor_source = observed_source.parse_constituent_anchor_source(output_directory, anchor_unit)
    sec_identities = observed_source.parse_sec_company_identities(output_directory, identity_unit)
    identity_fallback_units = observed_source.collect_identity_fallback_documents(
        client,
        output_directory=output_directory,
        requests=_required_identity_fallbacks(
            base,
            base_cutoff=base_cutoff,
            anchor=anchor_source,
            sec_identities=sec_identities,
        ),
        config=config,
        request_sha256=request_sha256,
    )
    release_confirmation_units = observed_source.collect_release_documents(
        client,
        output_directory=output_directory,
        announcements=announcements,
        config=config,
        request_sha256=request_sha256,
        role="release_confirmation",
    )
    original_release_hashes = {str(unit["source_url"]): str(unit["body_sha256"]) for unit in release_units}
    confirmation_release_hashes = {str(unit["source_url"]): str(unit["body_sha256"]) for unit in release_confirmation_units}
    if confirmation_release_hashes != original_release_hashes:
        raise DataReadinessError("official release body changed during anchor observation")
    confirmation_pages: list[dict[str, object]] = []
    confirmation_previous_url: str | None = None
    confirmation_seen_urls: set[str] = set()
    for page_number, expected_identity in enumerate(page_identities):
        confirmation = observed_source.fetch_search_page(
            client,
            page_number=page_number,
            output_directory=output_directory,
            request_sha256=request_sha256,
            config=config,
            role="search_confirmation",
        )
        confirmation_announcements, confirmation_urls = observed_source.parse_search_unit(
            output_directory,
            confirmation,
        )
        observed_source.validate_search_page_overlap(
            page_number,
            confirmation_urls,
            confirmation_previous_url,
            seen_urls=confirmation_seen_urls,
        )
        confirmation_previous_url = confirmation_urls[-1][1]
        if observed_source.search_page_semantics(confirmation_announcements, confirmation_urls) != expected_identity:
            raise DataReadinessError("official release index changed during anchor observation")
        confirmation_pages.append(confirmation)
    units = [
        *first_pages,
        *release_units,
        anchor_unit,
        identity_unit,
        *identity_fallback_units,
        *release_confirmation_units,
        *confirmation_pages,
    ]
    observed_at = max(observed_source.observation_time(unit) for unit in units)
    if any(observed_source.observation_time(unit) > observed_at for unit in units):
        raise DataReadinessError("observed membership response time is inconsistent")
    anchor = _parse_anchor(
        output_directory,
        anchor_unit,
        identity_unit,
        identity_fallback_units,
    )
    observed_changes, release_outcomes = _parse_observed_changes(
        output_directory,
        release_units,
    )
    memberships = _extend_memberships(
        base,
        base_cutoff=base_cutoff,
        observed_at=observed_at,
        closed_changes=closed_events.changes,
        observed_changes=observed_changes,
        anchor=anchor,
    )
    horizon_date = observed_at.astimezone(NEW_YORK).date()
    verify_membership_namespace_extension(
        base,
        memberships,
        base_cutoff_date=base_cutoff.isoformat(),
        current_cutoff_date=horizon_date.isoformat(),
    )
    anchor_path = output_directory / ANCHOR_FILE
    anchor.to_csv(anchor_path, index=False, lineterminator="\n")
    events_path = output_directory / EVENT_FILE
    _atomic_json(events_path, [_change_record(item) for item in observed_changes])
    outcomes_path = output_directory / OUTCOME_FILE
    _atomic_json(outcomes_path, release_outcomes)
    pending_path = output_directory / PENDING_FILE
    pending_changes = _pending_changes(
        closed_events.changes,
        observed_changes,
        observed_at=observed_at,
    )
    next_pending_effective_at = pending_changes[0].effective_at_utc.isoformat() if pending_changes else None
    _atomic_json(pending_path, [_change_record(item) for item in pending_changes])
    membership_path = output_directory / MEMBERSHIP_FILE
    checks = audit_universe_memberships(memberships, require_observed=False)
    audit = CanonicalAuditReport(checks=checks)
    if not audit.passed:
        failures = [check.name for check in checks if check.status != "pass"]
        raise DataReadinessError(f"observed S&P membership audit failed: {failures}")
    write_canonical_artifact(
        memberships,
        membership_path,
        artifact_type="memberships",
        audit=audit,
        inputs={
            "request_sha256": request_sha256,
            "base_membership_authority_sha256": str(base_parent["authority_sha256"]),
            "observation_unit_set_sha256": _json_sha256(units),
        },
        production_ready=False,
    )
    universe_sha256 = _membership_sha256(memberships)
    manifest: dict[str, object] = {
        "schema": MANIFEST_SCHEMA,
        "status": "complete",
        "request_sha256": request_sha256,
        "closed_cutoff_date": base_cutoff.isoformat(),
        "observed_at_utc": observed_at.isoformat(),
        "effective_horizon_date": horizon_date.isoformat(),
        "base_parent": dict(base_parent),
        "official_search_page_count": len(first_pages),
        "new_release_count": len(release_units),
        "observed_event_count": len(observed_changes),
        "no_membership_release_count": sum(outcome["disposition"] == "no_membership_event" for outcome in release_outcomes),
        "pending_change_count": len(pending_changes),
        "next_pending_effective_at_utc": next_pending_effective_at,
        "anchor_constituent_count": len(anchor),
        "sec_identity_count": len(sec_identities) + len(identity_fallback_units),
        "membership_intervals": len(memberships),
        "security_count": int(memberships["security_id"].nunique()),
        "ticker_count": int(memberships["ticker"].nunique()),
        "universe_sha256": universe_sha256,
        "raw_units": units,
        "raw_unit_set_sha256": _json_sha256(units),
        "anchor_artifact": _artifact_record(anchor_path),
        "event_artifact": _artifact_record(events_path),
        "outcome_artifact": _artifact_record(outcomes_path),
        "pending_artifact": _artifact_record(pending_path),
        "membership_artifact": _artifact_record(membership_path),
        "membership_manifest_sha256": file_sha256(manifest_path_for(membership_path)),
        "availability_policy": "observed",
        "production_ready": False,
        "training_eligible": False,
        "serving_eligible": False,
    }
    manifest_path = output_directory / "_manifest.json"
    _atomic_json(manifest_path, manifest)
    _atomic_json(output_directory / "_status.json", manifest)
    authority = {
        "schema": AUTHORITY_SCHEMA,
        "state": "observed_membership_complete",
        "artifact": "_manifest.json",
        "artifact_sha256": file_sha256(manifest_path),
        "request_sha256": request_sha256,
        "observed_at_utc": observed_at.isoformat(),
        "effective_horizon_date": horizon_date.isoformat(),
        "next_pending_effective_at_utc": next_pending_effective_at,
        "universe_sha256": universe_sha256,
    }
    _atomic_json(output_directory / "_authority.json", authority)
    assert_peak_memory_budget(
        hard_budget_gib=MAXIMUM_MEMORY_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage="observed S&P membership publication",
    )
    load_observed_sp500_membership_authority(output_directory)
    return manifest


def load_observed_sp500_membership_authority(
    directory: Path,
) -> ObservedMembershipAuthority:
    """Strictly replay an observed membership authority from retained response bytes."""

    root = directory.resolve()
    request = _json_object(root / "_request.json")
    manifest = _json_object(root / "_manifest.json")
    status = _json_object(root / "_status.json")
    authority = _json_object(root / "_authority.json")
    request_payload = {str(key): value for key, value in request.items() if key != "request_sha256"}
    request_sha256 = _json_sha256(request_payload)
    if (
        request.get("schema") != REQUEST_SCHEMA
        or request.get("request_sha256") != request_sha256
        or manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or status != manifest
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("state") != "observed_membership_complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(root / "_manifest.json")
        or authority.get("request_sha256") != request_sha256
        or manifest.get("production_ready") is not False
        or manifest.get("training_eligible") is not False
        or manifest.get("serving_eligible") is not False
    ):
        raise DataReadinessError("observed S&P membership envelope is invalid")
    base_root = Path(str(request.get("base_membership_directory", ""))).resolve()
    archive_root = Path(str(request.get("closed_archive_directory", ""))).resolve()
    event_root = Path(str(request.get("closed_event_directory", ""))).resolve()
    for parent in (base_root, archive_root, event_root):
        if root == parent or root in parent.parents or parent in root.parents:
            raise DataReadinessError("observed membership parent paths overlap")
    base, base_parent = load_sp500_membership_authority_envelope(base_root)
    expected_base = {
        "base_membership_authority_sha256": base_parent["authority_sha256"],
        "base_membership_manifest_sha256": base_parent["manifest_sha256"],
        "base_membership_table_sha256": base_parent["membership_table_sha256"],
        "base_membership_universe_sha256": base_parent["universe_sha256"],
    }
    if any(request.get(key) != value for key, value in expected_base.items()):
        raise DataReadinessError("observed membership base parent changed")
    closed_events = require_spglobal_event_reconstruction_ready(
        event_root,
        archive_directory=archive_root,
    )
    if (
        request.get("closed_event_authority_sha256") != closed_events.authority_sha256
        or request.get("closed_event_set_sha256") != closed_events.event_set_sha256
    ):
        raise DataReadinessError("observed membership closed events changed")
    units_value = manifest.get("raw_units")
    if not isinstance(units_value, list) or not all(isinstance(item, dict) for item in units_value):
        raise DataReadinessError("observed membership raw inventory is invalid")
    units = cast(list[dict[str, object]], units_value)
    if manifest.get("raw_unit_set_sha256") != _json_sha256(units):
        raise DataReadinessError("observed membership raw inventory hash changed")
    _verify_authority_root_inventory(root)
    observed_source.verify_observed_membership_raw_evidence(
        root,
        units,
        request_sha256=request_sha256,
    )
    search_units = [unit for unit in units if unit.get("role") == "search_page"]
    release_units = [unit for unit in units if unit.get("role") == "release"]
    release_confirmation_units = [unit for unit in units if unit.get("role") == "release_confirmation"]
    anchor_units = [unit for unit in units if unit.get("role") == "anchor"]
    identity_units = [unit for unit in units if unit.get("role") == "identity_anchor"]
    identity_fallback_units = [unit for unit in units if unit.get("role") == "identity_fallback"]
    confirmation_units = [unit for unit in units if unit.get("role") == "search_confirmation"]
    recognized_units = (
        len(search_units)
        + len(release_units)
        + len(release_confirmation_units)
        + len(anchor_units)
        + len(identity_units)
        + len(identity_fallback_units)
        + len(confirmation_units)
    )
    if (
        len(anchor_units) != 1
        or len(identity_units) != 1
        or len(confirmation_units) != len(search_units)
        or not search_units
        or recognized_units != len(units)
    ):
        raise DataReadinessError("observed membership raw roles are incomplete")
    announcements: list[SpGlobalAnnouncement] = []
    previous_last_url: str | None = None
    seen_urls: set[str] = set()
    for page_number, unit in enumerate(search_units):
        if int(str(unit.get("page_number", -1))) != page_number:
            raise DataReadinessError("observed official search pages are not contiguous")
        page_announcements, dated_urls = observed_source.parse_search_unit(root, unit)
        observed_source.validate_search_page_overlap(
            page_number,
            dated_urls,
            previous_last_url,
            seen_urls=seen_urls,
        )
        previous_last_url = dated_urls[-1][1]
        announcements.extend(page_announcements)
    if min(published for published, _ in dated_urls) > date.fromisoformat(str(request["closed_cutoff_date"])):
        raise DataReadinessError("observed official prefix does not reach the closed cutoff")
    confirmation_previous_url: str | None = None
    confirmation_seen_urls: set[str] = set()
    for page_number, (search_unit, confirmation_unit) in enumerate(zip(search_units, confirmation_units, strict=True)):
        page_announcements, page_urls = observed_source.parse_search_unit(root, search_unit)
        confirmation_announcements, confirmation_urls = observed_source.parse_search_unit(
            root,
            confirmation_unit,
        )
        if int(str(confirmation_unit.get("page_number", -1))) != page_number:
            raise DataReadinessError("observed confirmation pages are not contiguous")
        observed_source.validate_search_page_overlap(
            page_number,
            confirmation_urls,
            confirmation_previous_url,
            seen_urls=confirmation_seen_urls,
        )
        confirmation_previous_url = confirmation_urls[-1][1]
        if observed_source.search_page_semantics(page_announcements, page_urls) != observed_source.search_page_semantics(
            confirmation_announcements,
            confirmation_urls,
        ):
            raise DataReadinessError("observed official race confirmation does not replay")
    expected_release_urls = sorted(
        {item.url for item in announcements if item.published_date > date.fromisoformat(str(request["closed_cutoff_date"]))}
    )
    if sorted(str(unit.get("source_url", "")) for unit in release_units) != expected_release_urls:
        raise DataReadinessError("observed release inventory differs from search discovery")
    if sorted(str(unit.get("source_url", "")) for unit in release_confirmation_units) != expected_release_urls or {
        str(unit["source_url"]): str(unit["body_sha256"]) for unit in release_confirmation_units
    } != {str(unit["source_url"]): str(unit["body_sha256"]) for unit in release_units}:
        raise DataReadinessError("observed release confirmation does not replay")
    anchor_source = observed_source.parse_constituent_anchor_source(root, anchor_units[0])
    sec_identities = observed_source.parse_sec_company_identities(root, identity_units[0])
    expected_fallbacks = _required_identity_fallbacks(
        base,
        base_cutoff=date.fromisoformat(str(request["closed_cutoff_date"])),
        anchor=anchor_source,
        sec_identities=sec_identities,
    )
    observed_fallbacks = sorted((str(unit.get("ticker", "")), str(unit.get("cik", ""))) for unit in identity_fallback_units)
    if observed_fallbacks != expected_fallbacks:
        raise DataReadinessError("observed SEC identity fallback inventory changed")
    anchor = _parse_anchor(
        root,
        anchor_units[0],
        identity_units[0],
        identity_fallback_units,
    )
    observed_changes, release_outcomes = _parse_observed_changes(root, release_units)
    observed_at = max(observed_source.observation_time(unit) for unit in units)
    if manifest.get("observed_at_utc") != observed_at.isoformat():
        raise DataReadinessError("observed membership cutoff does not replay")
    expected = _extend_memberships(
        base,
        base_cutoff=date.fromisoformat(str(request["closed_cutoff_date"])),
        observed_at=observed_at,
        closed_changes=closed_events.changes,
        observed_changes=observed_changes,
        anchor=anchor,
    )
    anchor_path = _verified_artifact(root, manifest.get("anchor_artifact"))
    event_path = _verified_artifact(root, manifest.get("event_artifact"))
    outcome_path = _verified_artifact(root, manifest.get("outcome_artifact"))
    pending_path = _verified_artifact(root, manifest.get("pending_artifact"))
    membership_path = _verified_artifact(root, manifest.get("membership_artifact"))
    if pd.read_csv(anchor_path, dtype=str, keep_default_na=False).to_dict("records") != anchor.astype(str).to_dict("records"):
        raise DataReadinessError("observed membership anchor artifact does not replay")
    if _load_json(event_path) != [_change_record(item) for item in observed_changes]:
        raise DataReadinessError("observed membership event artifact does not replay")
    if _load_json(outcome_path) != release_outcomes:
        raise DataReadinessError("observed membership release outcomes do not replay")
    pending_changes = _pending_changes(
        closed_events.changes,
        observed_changes,
        observed_at=observed_at,
    )
    next_pending_effective_at = pending_changes[0].effective_at_utc.isoformat() if pending_changes else None
    if _load_json(pending_path) != [_change_record(item) for item in pending_changes]:
        raise DataReadinessError("observed membership pending changes do not replay")
    if not manifest_path_for(membership_path).is_file() or manifest.get("membership_manifest_sha256") != file_sha256(
        manifest_path_for(membership_path)
    ):
        raise DataReadinessError("observed membership canonical manifest changed")
    actual, canonical_manifest = load_canonical_artifact(
        membership_path,
        expected_type="memberships",
        allow_research=True,
    )
    expected_canonical_inputs = {
        "request_sha256": request_sha256,
        "base_membership_authority_sha256": str(base_parent["authority_sha256"]),
        "observation_unit_set_sha256": _json_sha256(units),
    }
    if canonical_manifest.get("inputs") != expected_canonical_inputs or canonical_manifest.get("production_ready") is not False:
        raise DataReadinessError("observed membership canonical lineage changed")
    if _membership_records(actual) != _membership_records(expected):
        raise DataReadinessError("observed membership table does not replay")
    universe_sha256 = _membership_sha256(actual)
    horizon_date = observed_at.astimezone(NEW_YORK).date().isoformat()
    expected_authority = {
        "schema": AUTHORITY_SCHEMA,
        "state": "observed_membership_complete",
        "artifact": "_manifest.json",
        "artifact_sha256": file_sha256(root / "_manifest.json"),
        "request_sha256": request_sha256,
        "observed_at_utc": observed_at.isoformat(),
        "effective_horizon_date": horizon_date,
        "next_pending_effective_at_utc": next_pending_effective_at,
        "universe_sha256": universe_sha256,
    }
    if (
        authority != expected_authority
        or manifest.get("observed_at_utc") != observed_at.isoformat()
        or manifest.get("effective_horizon_date") != horizon_date
        or authority.get("effective_horizon_date") != horizon_date
        or manifest.get("next_pending_effective_at_utc") != next_pending_effective_at
        or authority.get("next_pending_effective_at_utc") != next_pending_effective_at
        or manifest.get("universe_sha256") != universe_sha256
        or authority.get("universe_sha256") != universe_sha256
        or int(manifest.get("membership_intervals", -1)) != len(actual)
        or int(manifest.get("security_count", -1)) != actual["security_id"].nunique()
        or int(manifest.get("ticker_count", -1)) != actual["ticker"].nunique()
        or int(manifest.get("pending_change_count", -1)) != len(pending_changes)
        or int(manifest.get("anchor_constituent_count", -1)) != len(anchor)
        or int(manifest.get("sec_identity_count", -1)) != len(sec_identities) + len(identity_fallback_units)
    ):
        raise DataReadinessError("observed membership semantic counts are invalid")
    verify_membership_namespace_extension(
        base,
        actual,
        base_cutoff_date=str(request["closed_cutoff_date"]),
        current_cutoff_date=horizon_date,
    )
    parent_descriptor: dict[str, object] = {
        "authority_type": "observed_time",
        "authority_sha256": file_sha256(root / "_authority.json"),
        "manifest_sha256": file_sha256(root / "_manifest.json"),
        "membership_table_sha256": file_sha256(membership_path),
        "universe_sha256": universe_sha256,
        "cutoff_date": horizon_date,
        "observed_at_utc": observed_at.isoformat(),
        "next_pending_effective_at_utc": next_pending_effective_at,
        "base_membership_authority_sha256": base_parent["authority_sha256"],
        "observed_release_outcomes": release_outcomes,
        "observed_events": [_change_record(item) for item in observed_changes],
    }
    return ObservedMembershipAuthority(root, actual, manifest, parent_descriptor)


def _parse_anchor(
    root: Path,
    unit: Mapping[str, object],
    identity_unit: Mapping[str, object],
    identity_fallback_units: Sequence[Mapping[str, object]] = (),
) -> pd.DataFrame:
    anchor = observed_source.parse_constituent_anchor_source(root, unit)
    sec_identities = observed_source.parse_sec_company_identities(root, identity_unit)
    fallback_identities = observed_source.parse_sec_submission_identities(
        root,
        identity_fallback_units,
    )
    overlap = sorted(set(sec_identities).intersection(fallback_identities))
    if overlap:
        raise DataReadinessError(f"SEC identity fallback duplicates bulk-map tickers: {overlap[:20]}")
    sec_identities.update(fallback_identities)
    missing_identities = sorted(set(anchor["ticker"]).difference(sec_identities))
    if missing_identities:
        raise DataReadinessError(f"SEC identity anchor is missing S&P tickers: {missing_identities[:20]}")
    anchor["cik"] = anchor["ticker"].map(sec_identities)
    return anchor.sort_values("ticker", kind="stable").reset_index(drop=True)


def _required_identity_fallbacks(
    base: pd.DataFrame,
    *,
    base_cutoff: date,
    anchor: pd.DataFrame,
    sec_identities: Mapping[str, str],
) -> list[tuple[str, str]]:
    missing = sorted(set(anchor["ticker"].astype(str)).difference(sec_identities))
    if not missing:
        return []
    required = {
        "ticker",
        "security_id",
        "effective_from_utc",
        "effective_to_utc",
    }
    if not required.issubset(base.columns):
        raise DataReadinessError("base membership identity columns are incomplete")
    data = base.copy()
    data["effective_from_utc"] = pd.to_datetime(
        data["effective_from_utc"],
        utc=True,
        errors="coerce",
    )
    data["effective_to_utc"] = pd.to_datetime(
        data["effective_to_utc"],
        utc=True,
        errors="coerce",
    )
    if bool(data["effective_from_utc"].isna().any()):
        raise DataReadinessError("base membership identity timestamps are invalid")
    boundary = pd.Timestamp(base_cutoff, tz="UTC") + pd.Timedelta(days=1)
    active = data[data["effective_from_utc"].lt(boundary) & (data["effective_to_utc"].isna() | data["effective_to_utc"].ge(boundary))]
    anchor_index = anchor.set_index("ticker", drop=False).to_dict("index")
    requests: list[tuple[str, str]] = []
    for ticker in missing:
        inherited = active[active["ticker"].astype(str).eq(ticker)]
        inherited_ciks = {cik for cik in (_security_cik(str(value)) for value in inherited["security_id"]) if cik is not None}
        if len(inherited_ciks) > 1:
            raise DataReadinessError(f"SEC identity fallback has ambiguous inherited CIK: {ticker}")
        anchor_cik = str(anchor_index[ticker]["cik"])
        if inherited_ciks:
            candidate = next(iter(inherited_ciks))
            if candidate != anchor_cik:
                raise DataReadinessError(f"SEC identity fallback anchor CIK conflicts for {ticker}")
        else:
            candidate = anchor_cik
        if not candidate.isdigit() or len(candidate) != 10:
            raise DataReadinessError(f"SEC identity fallback CIK is invalid: {ticker}")
        requests.append((ticker, candidate))
    return requests


def _parse_observed_changes(
    root: Path,
    units: Sequence[Mapping[str, object]],
) -> tuple[tuple[IndexChange, ...], list[dict[str, object]]]:
    changes: list[IndexChange] = []
    outcomes: list[dict[str, object]] = []
    identities: dict[tuple[datetime, str, str], IndexChange] = {}
    for unit in units:
        html = observed_source.load_release_html(root, unit)
        parsed = parse_sp500_changes(
            html,
            source_url=str(unit["source_url"]),
            published_date=date.fromisoformat(str(unit["published_date"])),
            source_sha256=str(unit["body_sha256"]),
            allow_verified_no_membership_event=True,
            source_title=str(unit.get("title", "")),
        )
        for item in parsed:
            key = (item.effective_at_utc, item.action, item.ticker)
            existing = identities.get(key)
            if existing is not None and _change_record(existing) != _change_record(item):
                raise DataReadinessError("observed S&P releases conflict")
            identities[key] = item
        outcomes.append(
            {
                "source_url": str(unit["source_url"]),
                "published_date": str(unit["published_date"]),
                "source_sha256": str(unit["body_sha256"]),
                "first_observed_at_utc": str(unit["retrieved_at_utc"]),
                "disposition": "membership_event" if parsed else "no_membership_event",
                "event_count": len(parsed),
            }
        )
    changes.extend(identities.values())
    return (
        tuple(
            sorted(
                changes,
                key=lambda item: (item.effective_at_utc, item.action, item.ticker),
            )
        ),
        sorted(outcomes, key=lambda item: str(item["source_url"])),
    )


def _pending_changes(
    closed_changes: Sequence[IndexChange],
    observed_changes: Sequence[IndexChange],
    *,
    observed_at: datetime,
) -> tuple[IndexChange, ...]:
    pending: dict[tuple[datetime, str, str], IndexChange] = {}
    for item in (*closed_changes, *observed_changes):
        if item.effective_at_utc > observed_at:
            key = (item.effective_at_utc, item.action, item.ticker)
            existing = pending.get(key)
            if existing is not None and _change_record(existing) != _change_record(item):
                raise DataReadinessError("pending S&P changes conflict")
            pending[key] = item
    return tuple(
        sorted(
            pending.values(),
            key=lambda item: (item.effective_at_utc, item.action, item.ticker),
        )
    )


def _extend_memberships(
    base: pd.DataFrame,
    *,
    base_cutoff: date,
    observed_at: datetime,
    closed_changes: Sequence[IndexChange],
    observed_changes: Sequence[IndexChange],
    anchor: pd.DataFrame,
) -> pd.DataFrame:
    horizon_date = observed_at.astimezone(NEW_YORK).date()
    if horizon_date < base_cutoff:
        raise DataReadinessError("observed membership horizon predates closed authority")
    data = base.copy()
    for column in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
        data[column] = pd.to_datetime(data[column], utc=True, errors="coerce")
    if bool(data[["effective_from_utc", "available_at_utc"]].isna().any().any()):
        raise DataReadinessError("base membership timestamps are invalid")
    boundary = pd.Timestamp(base_cutoff, tz="UTC") + pd.Timedelta(days=1)
    active = data[data["effective_from_utc"].lt(boundary) & (data["effective_to_utc"].isna() | data["effective_to_utc"].ge(boundary))]
    if bool(active["ticker"].duplicated().any()):
        raise DataReadinessError("base membership has ambiguous active tickers")
    active_index = {str(row.ticker): int(row.Index) for row in active.itertuples()}
    active_by_cik: dict[str, str] = {}
    ambiguous_active_ciks: set[str] = set()
    for ticker, row_index in active_index.items():
        cik = _security_cik(str(data.loc[row_index, "security_id"]))
        if cik is None:
            continue
        if cik in active_by_cik and active_by_cik[cik] != ticker:
            ambiguous_active_ciks.add(cik)
        else:
            active_by_cik[cik] = ticker
    for cik in ambiguous_active_ciks:
        active_by_cik.pop(cik, None)
    normalized_anchor = anchor.copy()
    for row_index, record in normalized_anchor.iterrows():
        inherited_ticker = active_by_cik.get(str(record["cik"]))
        observed_ticker = str(record["ticker"])
        if inherited_ticker is not None and _punctuation_alias(inherited_ticker, observed_ticker):
            normalized_anchor.loc[row_index, "ticker"] = inherited_ticker
    if bool(normalized_anchor["ticker"].duplicated().any()):
        raise DataReadinessError("independent S&P anchor aliases collide")
    anchor_index = normalized_anchor.set_index("ticker", drop=False).to_dict("index")
    cik_identities: dict[str, set[str]] = {}
    for identity in data["security_id"].astype(str).unique():
        cik = _security_cik(identity)
        if cik is not None:
            cik_identities.setdefault(cik, set()).add(identity)
    event_sources: dict[tuple[datetime, str, str], tuple[IndexChange, datetime]] = {}
    for item in closed_changes:
        effective_date = item.effective_at_utc.astimezone(NEW_YORK).date()
        if base_cutoff < effective_date and item.effective_at_utc <= observed_at:
            event_sources[(item.effective_at_utc, item.action, item.ticker)] = (
                item,
                item.effective_at_utc,
            )
    for item in observed_changes:
        effective_date = item.effective_at_utc.astimezone(NEW_YORK).date()
        if effective_date <= base_cutoff:
            raise DataReadinessError("newly observed S&P event predates closed authority")
        if item.effective_at_utc <= observed_at:
            key = (item.effective_at_utc, item.action, item.ticker)
            existing = event_sources.get(key)
            if existing is not None and _change_record(existing[0]) != _change_record(item):
                raise DataReadinessError("closed and observed S&P events conflict")
            event_sources[key] = (item, observed_at)
    events_by_time: dict[datetime, list[tuple[IndexChange, datetime]]] = {}
    for value in event_sources.values():
        events_by_time.setdefault(value[0].effective_at_utc, []).append(value)
    snapshot_id = f"sp500-observed-{hashlib.sha256(observed_at.isoformat().encode()).hexdigest()[:20]}"
    for effective_at in sorted(events_by_time):
        group = events_by_time[effective_at]
        deletions = [value for value in group if value[0].action == "deletion"]
        additions = [value for value in group if value[0].action == "addition"]
        if len(deletions) != len(additions):
            raise DataReadinessError(f"observed S&P change batch is unbalanced at {effective_at.isoformat()}")
        for item, _ in deletions:
            deletion_row_index = active_index.get(item.ticker)
            if deletion_row_index is None:
                raise DataReadinessError(f"observed S&P deletion is not active: {item.ticker}")
            del active_index[item.ticker]
            data.loc[deletion_row_index, "effective_to_utc"] = pd.Timestamp(effective_at)
        for item, availability in additions:
            if item.ticker in active_index:
                raise DataReadinessError(f"observed S&P addition is already active: {item.ticker}")
            anchor_row = anchor_index.get(item.ticker)
            if anchor_row is None:
                raise DataReadinessError(f"observed S&P addition is absent from independent anchor: {item.ticker}")
            cik = str(anchor_row["cik"])
            identities = cik_identities.get(cik, set())
            if len(identities) > 1:
                raise DataReadinessError(f"observed S&P addition has ambiguous CIK identity: {item.ticker}")
            security_id = next(iter(identities), f"cik:{cik}")
            sector = str(anchor_row["sector"])
            row = {
                "ticker": item.ticker,
                "security_id": security_id,
                "effective_from_utc": pd.Timestamp(effective_at),
                "effective_to_utc": pd.NaT,
                "available_at_utc": pd.Timestamp(max(effective_at, availability)),
                "sector": sector,
                "industry": str(anchor_row["industry"]),
                "market_cap_bucket": "large_cap_sp500",
                "liquidity_bucket": "sp500_constituent",
                "primary_benchmark": SECTOR_BENCHMARKS.get(sector, "SPY"),
                "universe_snapshot_id": snapshot_id,
                "source": OFFICIAL_MEMBERSHIP_SOURCE,
                "availability_policy": OBSERVED_AVAILABILITY_POLICY,
                "schema_version": MEMBERSHIP_SCHEMA_VERSION,
            }
            data = pd.concat([data, pd.DataFrame([row])], ignore_index=True)
            active_index[item.ticker] = len(data) - 1
            cik_identities.setdefault(cik, set()).add(security_id)
    active_tickers = set(active_index)
    anchor_tickers = set(anchor_index)
    missing_tickers = anchor_tickers.difference(active_tickers)
    extra_tickers = active_tickers.difference(anchor_tickers)
    pending_event_tickers = {item.ticker for item in (*closed_changes, *observed_changes) if item.effective_at_utc > observed_at}
    missing_by_cik: dict[str, list[str]] = {}
    extra_by_cik: dict[str, list[str]] = {}
    for ticker in missing_tickers:
        missing_by_cik.setdefault(str(anchor_index[ticker]["cik"]), []).append(ticker)
    for ticker in extra_tickers:
        cik = _security_cik(str(data.loc[active_index[ticker], "security_id"]))
        if cik is not None:
            extra_by_cik.setdefault(cik, []).append(ticker)
    for cik in sorted(set(missing_by_cik).intersection(extra_by_cik)):
        successors = missing_by_cik[cik]
        predecessors = extra_by_cik[cik]
        if len(successors) != 1 or len(predecessors) != 1:
            continue
        successor_ticker = successors[0]
        predecessor_ticker = predecessors[0]
        if {successor_ticker, predecessor_ticker}.intersection(pending_event_tickers):
            continue
        predecessor_index = active_index.pop(predecessor_ticker)
        security_id = str(data.loc[predecessor_index, "security_id"])
        data.loc[predecessor_index, "effective_to_utc"] = pd.Timestamp(observed_at)
        anchor_row = anchor_index[successor_ticker]
        successor = {
            "ticker": successor_ticker,
            "security_id": security_id,
            "effective_from_utc": pd.Timestamp(observed_at),
            "effective_to_utc": pd.NaT,
            "available_at_utc": pd.Timestamp(observed_at),
            "sector": str(anchor_row["sector"]),
            "industry": str(anchor_row["industry"]),
            "market_cap_bucket": "large_cap_sp500",
            "liquidity_bucket": "sp500_constituent",
            "primary_benchmark": SECTOR_BENCHMARKS.get(
                str(anchor_row["sector"]),
                "SPY",
            ),
            "universe_snapshot_id": snapshot_id,
            "source": OBSERVED_IDENTITY_SOURCE,
            "availability_policy": OBSERVED_AVAILABILITY_POLICY,
            "schema_version": MEMBERSHIP_SCHEMA_VERSION,
        }
        data = pd.concat([data, pd.DataFrame([successor])], ignore_index=True)
        active_index[successor_ticker] = len(data) - 1
    active_tickers = set(active_index)
    if active_tickers != anchor_tickers:
        missing = sorted(anchor_tickers.difference(active_tickers))
        extra = sorted(active_tickers.difference(anchor_tickers))
        raise DataReadinessError(f"observed S&P state differs from independent anchor: missing={missing[:20]} extra={extra[:20]}")
    active_cik_owners: dict[str, set[str]] = {}
    for ticker, row_index in active_index.items():
        cik = _security_cik(str(data.loc[row_index, "security_id"]))
        if cik is not None:
            active_cik_owners.setdefault(cik, set()).add(ticker)
    for ticker, row_index in list(active_index.items()):
        expected_cik = str(anchor_index[ticker]["cik"])
        actual_cik = _security_cik(str(data.loc[row_index, "security_id"]))
        if actual_cik == expected_cik:
            continue
        if actual_cik is None:
            raise DataReadinessError(f"observed S&P inherited identity has no CIK for {ticker}")
        conflicting_owners = active_cik_owners.get(expected_cik, set()).difference({ticker})
        if conflicting_owners:
            raise DataReadinessError(f"observed S&P successor CIK is already active for {ticker}")
        identities = cik_identities.get(expected_cik, set())
        if len(identities) > 1:
            raise DataReadinessError(f"observed S&P successor CIK identity is ambiguous for {ticker}")
        anchor_row = anchor_index[ticker]
        security_id = next(iter(identities), f"cik:{expected_cik}")
        data.loc[row_index, "effective_to_utc"] = pd.Timestamp(observed_at)
        successor = {
            "ticker": ticker,
            "security_id": security_id,
            "effective_from_utc": pd.Timestamp(observed_at),
            "effective_to_utc": pd.NaT,
            "available_at_utc": pd.Timestamp(observed_at),
            "sector": str(anchor_row["sector"]),
            "industry": str(anchor_row["industry"]),
            "market_cap_bucket": "large_cap_sp500",
            "liquidity_bucket": "sp500_constituent",
            "primary_benchmark": SECTOR_BENCHMARKS.get(
                str(anchor_row["sector"]),
                "SPY",
            ),
            "universe_snapshot_id": snapshot_id,
            "source": OBSERVED_IDENTITY_SOURCE,
            "availability_policy": OBSERVED_AVAILABILITY_POLICY,
            "schema_version": MEMBERSHIP_SCHEMA_VERSION,
        }
        data = pd.concat([data, pd.DataFrame([successor])], ignore_index=True)
        active_index[ticker] = len(data) - 1
        active_cik_owners.get(actual_cik, set()).discard(ticker)
        active_cik_owners.setdefault(expected_cik, set()).add(ticker)
        cik_identities.setdefault(expected_cik, set()).add(security_id)
    data["universe_snapshot_id"] = snapshot_id
    return data.sort_values(
        ["ticker", "effective_from_utc", "security_id"],
        kind="stable",
    ).reset_index(drop=True)


def _verify_authority_root_inventory(root: Path) -> None:
    allowed_root = {
        "_request.json",
        "_status.json",
        "_manifest.json",
        "_authority.json",
        "_collector.lock",
        "objects",
        "units",
        ANCHOR_FILE,
        EVENT_FILE,
        OUTCOME_FILE,
        PENDING_FILE,
        MEMBERSHIP_FILE,
        manifest_path_for(root / MEMBERSHIP_FILE).name,
        f"{MEMBERSHIP_FILE}.lock",
    }
    root_entries = list(root.iterdir())
    observed_root = {path.name for path in root_entries}
    if (
        observed_root != allowed_root
        or any(path.is_symlink() for path in root_entries)
        or not (root / "objects").is_dir()
        or not (root / "units").is_dir()
        or any(path.is_dir() for path in root_entries if path.name not in {"objects", "units"})
    ):
        raise DataReadinessError("observed membership root files differ from inventory")


def _security_cik(security_id: str) -> str | None:
    prefix = security_id.split(":ticker:", maxsplit=1)[0]
    return prefix.removeprefix("cik:") if prefix.startswith("cik:") else None


def _punctuation_alias(left: str, right: str) -> bool:
    return left.replace(".", "-") == right.replace(".", "-")


def _change_record(item: IndexChange) -> dict[str, object]:
    return item.to_record()


def _membership_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    data = frame.sort_values(["ticker", "effective_from_utc", "security_id"], kind="stable")
    records: list[dict[str, object]] = []
    for record in data.to_dict("records"):
        for field in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
            value = record[field]
            record[field] = None if pd.isna(value) else pd.Timestamp(value).tz_convert("UTC").isoformat()
        records.append({str(key): value for key, value in record.items()})
    return records


def _membership_sha256(frame: pd.DataFrame) -> str:
    return _json_sha256(_membership_records(frame))


def _artifact_record(path: Path) -> dict[str, object]:
    return {"path": path.name, "sha256": file_sha256(path), "bytes": path.stat().st_size}


def _verified_artifact(root: Path, value: object) -> Path:
    if not isinstance(value, Mapping):
        raise DataReadinessError("observed membership artifact inventory is invalid")
    path = _resolve_inside(root, str(value.get("path", "")))
    if not path.is_file() or value.get("sha256") != file_sha256(path) or int(value.get("bytes", -1)) != path.stat().st_size:
        raise DataReadinessError("observed membership artifact changed")
    return path


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise DataReadinessError("observed membership path escapes authority root")
    return candidate


def _json_sha256(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"observed membership JSON is invalid: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"observed membership JSON is not an object: {path}")
    return {str(key): item for key, item in value.items()}


def _load_json(path: Path) -> object:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"observed membership JSON is invalid: {path}") from exc


def _write_new_json(path: Path, value: object) -> None:
    if path.exists():
        raise DataReadinessError(f"observed membership attempt is immutable: {path}")
    _atomic_json(path, value)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
