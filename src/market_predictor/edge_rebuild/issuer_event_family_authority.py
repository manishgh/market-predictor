"""Immutable research authority for catalyst-driven swing event cohorts."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final
from uuid import uuid4

import numpy as np
import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.reconciliation import (
    assignment_integrity_summary,
    build_event_assignments,
    reconciliation_sha256,
    stamp_canonical_decision_ids,
)
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    write_canonical_artifact,
)
from market_predictor.resources import (
    assert_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.swing.event_attribution_history import (
    load_event_attribution_history,
)
from market_predictor.swing.event_families import (
    ALLOWED_SOURCE_FAMILIES_BY_FAMILY,
    EVENT_FAMILIES,
    EVENT_FAMILY_POLICY_SHA256,
    EVENT_FAMILY_POLICY_VERSION,
    classify_event_families,
)
from market_predictor.v3.errors import DataReadinessError

AUTHORITY_SCHEMA: Final = "edge_rebuild.issuer_event_family_authority.v2"
MANIFEST_SCHEMA: Final = "edge_rebuild.issuer_event_family_manifest.v2"
POLICY_SCHEMA: Final = "market_predictor.swing_event_family_authority.v2"
FAMILY_EVENTS_ARTIFACT_TYPE: Final = "issuer_event_family_events"
FAMILY_ASSIGNMENTS_ARTIFACT_TYPE: Final = "issuer_event_family_assignments"
FAMILY_COVERAGE_ARTIFACT_TYPE: Final = "issuer_event_family_coverage"
COHORT_AUDIT_ARTIFACT_TYPE: Final = "issuer_event_family_cohort_audit"
UNCLASSIFIED_EVENTS_ARTIFACT_TYPE: Final = "issuer_event_family_unclassified_events"
FAMILY_STATUSES: Final = frozenset(
    {"admitted", "blocked_missing_source", "absent"}
)

FAMILY_EVENT_COLUMNS: Final = (
    "family_event_id",
    "source_event_id",
    "relation_id",
    "source_security_id",
    "source_ticker",
    "security_id",
    "ticker",
    "source_family",
    "event_family",
    "classification_state",
    "classification_rule_id",
    "classification_basis",
    "matched_text",
    "published_at_utc",
    "event_available_at_utc",
    "relation_available_at_utc",
    "feature_available_at_utc",
    "availability_policy",
    "relation_channel",
    "relation_score",
    "research_eligible",
    "production_eligible",
    "exclusion_reason",
    "event_family_policy_version",
    "event_family_policy_sha256",
    "schema_version",
)
FAMILY_ASSIGNMENT_EXTRA_COLUMNS: Final = (
    "event_family",
    "original_source_family",
    "source_event_id",
    "relation_id",
    "event_family_policy_sha256",
)
FAMILY_COVERAGE_COLUMNS: Final = (
    "collection_id",
    "chunk_id",
    "security_id",
    "ticker",
    "source_family",
    "event_family",
    "requested_start_utc",
    "requested_end_utc",
    "completed_at_utc",
    "coverage_state",
    "missingness_known",
    "zero_event_semantics",
    "research_eligible",
    "production_eligible",
    "schema_version",
)
COHORT_AUDIT_COLUMNS: Final = (
    "event_family",
    "dimension_type",
    "dimension_value",
    "event_count",
    "security_count",
    "assigned_decision_count",
    "known_coverage_decision_count",
    "abstention_count",
    "abstention_rate",
    "first_event_available_at_utc",
    "last_event_available_at_utc",
    "schema_version",
)


@dataclass(frozen=True, slots=True)
class SwingEventFamilyPolicy:
    minimum_decision_date: date
    eligible_channels: tuple[str, ...]
    source_families: tuple[str, ...]
    event_families: tuple[str, ...]
    windows: Mapping[str, pd.Timedelta]
    maximum_process_memory_gib: float
    memory_guard_headroom_gib: float


@dataclass(frozen=True, slots=True)
class IssuerEventFamilyAuthority:
    directory: Path
    events: pd.DataFrame
    assignments: pd.DataFrame
    coverage: pd.DataFrame
    cohort_audit: pd.DataFrame
    unclassified_artifact_records: tuple[Mapping[str, object], ...]
    manifest: Mapping[str, object]
    authority: Mapping[str, object]


def load_swing_event_family_policy(path: Path) -> SwingEventFamilyPolicy:
    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != POLICY_SCHEMA:
        raise DataReadinessError(f"unsupported swing event-family policy: {path}")
    if raw.get("event_family_policy_version") != EVENT_FAMILY_POLICY_VERSION:
        raise DataReadinessError("event-family classifier and authority policy versions differ")
    families = _text_tuple(raw.get("event_families"), "event_families")
    if families != EVENT_FAMILIES:
        raise DataReadinessError("event-family authority policy must list the frozen families in order")
    channels = _text_tuple(
        raw.get("training_eligible_relation_channels"),
        "training_eligible_relation_channels",
    )
    if channels != ("direct_issuer",):
        raise DataReadinessError("only direct-issuer relations may train swing event specialists")
    if raw.get("unknown_coverage_policy") != "abstain":
        raise DataReadinessError("unknown event coverage must cause abstention")
    if raw.get("historical_proxy_policy") != "research_only":
        raise DataReadinessError("historical proxy evidence must remain research-only")
    raw_windows = raw.get("assignment_windows")
    if not isinstance(raw_windows, dict) or not raw_windows:
        raise DataReadinessError("event-family policy has no assignment windows")
    windows = {str(name): pd.Timedelta(str(value)) for name, value in raw_windows.items()}
    if any(value <= pd.Timedelta(0) for value in windows.values()):
        raise DataReadinessError("event-family assignment windows must be positive")
    maximum = float(raw.get("maximum_process_memory_gib", 0))
    headroom = float(raw.get("memory_guard_headroom_gib", 0))
    if maximum <= 0 or headroom <= 0 or headroom >= maximum:
        raise DataReadinessError("event-family memory policy is invalid")
    source_families = _text_tuple(
        raw.get("research_source_families"), "research_source_families"
    )
    if any(source != source.lower() for source in source_families):
        raise DataReadinessError("event-family source allowlist must be lowercase")
    required_sources = {
        source
        for family in families
        for source in ALLOWED_SOURCE_FAMILIES_BY_FAMILY[family]
    }
    if not required_sources.issubset(source_families):
        raise DataReadinessError(
            "event-family source allowlist omits classifier-authoritative sources"
        )
    raw_family_sources = raw.get("allowed_source_families")
    expected_family_sources = {
        family: list(ALLOWED_SOURCE_FAMILIES_BY_FAMILY[family])
        for family in families
    }
    if raw_family_sources != expected_family_sources:
        raise DataReadinessError(
            "event-family policy source mapping differs from classifier policy"
        )
    return SwingEventFamilyPolicy(
        minimum_decision_date=date.fromisoformat(str(raw["minimum_decision_date"])),
        eligible_channels=channels,
        source_families=source_families,
        event_families=families,
        windows=windows,
        maximum_process_memory_gib=maximum,
        memory_guard_headroom_gib=headroom,
    )


def _validate_source_family_allowlist(
    frame: pd.DataFrame,
    *,
    policy: SwingEventFamilyPolicy,
    context: str,
) -> None:
    _validate_observed_source_families(
        frame,
        allowed_sources=policy.source_families,
        context=context,
    )


def _validate_observed_source_families(
    frame: pd.DataFrame,
    *,
    allowed_sources: tuple[str, ...],
    context: str,
    source_column: str = "source_family",
) -> None:
    if source_column not in frame.columns:
        raise DataReadinessError(f"{context} is missing {source_column}")
    raw_sources = frame[source_column]
    normalized = raw_sources.fillna("").astype(str).str.lower().str.strip()
    observed = set(normalized.tolist())
    unsupported = sorted(observed.difference(allowed_sources))
    noncanonical = raw_sources.fillna("").astype(str).ne(normalized)
    if "" in observed or unsupported or bool(noncanonical.any()):
        detail = ", ".join(unsupported or ["<empty-or-noncanonical>"])
        raise DataReadinessError(
            f"{context} contains source families outside policy: {detail}"
        )


def _validate_family_source_pairs(
    frame: pd.DataFrame,
    *,
    context: str,
    source_column: str = "source_family",
) -> None:
    required = {"event_family", source_column}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataReadinessError(
            f"{context} is missing family/source columns: {', '.join(missing)}"
        )
    invalid: set[str] = set()
    for family, source in frame.loc[:, ["event_family", source_column]].itertuples(
        index=False,
        name=None,
    ):
        family_name = str(family)
        source_name = str(source).lower().strip()
        allowed = ALLOWED_SOURCE_FAMILIES_BY_FAMILY.get(family_name)
        if allowed is None or source_name not in allowed:
            invalid.add(f"{family_name}:{source_name}")
    if invalid:
        raise DataReadinessError(
            f"{context} contains non-authoritative family/source pairs: "
            + ", ".join(sorted(invalid))
        )


def _source_supports_family(
    *,
    source_family: str,
    event_family: str,
    policy: SwingEventFamilyPolicy,
) -> bool:
    source = source_family.lower().strip()
    if source not in policy.source_families:
        return False
    allowed = ALLOWED_SOURCE_FAMILIES_BY_FAMILY.get(event_family)
    return allowed is not None and source in allowed


def _family_statuses(
    families: tuple[str, ...],
    events: pd.DataFrame,
    coverage: pd.DataFrame,
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for family in families:
        family_coverage = coverage.loc[
            coverage["event_family"].astype(str).eq(family)
        ]
        usable_source = bool(
            not family_coverage.empty
            and family_coverage["research_eligible"].astype(bool).any()
        )
        family_events = events.loc[
            events["event_family"].astype(str).eq(family)
            & events["research_eligible"].astype(bool)
        ]
        if not usable_source:
            status = "blocked_missing_source"
        elif family_events.empty:
            status = "absent"
        else:
            status = "admitted"
        statuses[family] = status
    return statuses


def _identity_intervals_by_security(
    identities: pd.DataFrame,
) -> dict[str, tuple[dict[str, object], ...]]:
    required = {
        "security_id",
        "company",
        "effective_from_utc",
        "effective_to_utc",
        "available_at_utc",
    }
    missing = sorted(required.difference(identities.columns))
    if missing:
        raise DataReadinessError(
            "event-family security identities are missing: " + ", ".join(missing)
        )
    data = identities.loc[:, sorted(required)].copy()
    data["effective_from_utc"] = _utc(
        data["effective_from_utc"], "identity effective from"
    )
    data["effective_to_utc"] = pd.to_datetime(
        data["effective_to_utc"], utc=True, errors="coerce"
    )
    data["available_at_utc"] = _utc(
        data["available_at_utc"], "identity availability"
    )
    data["security_id"] = data["security_id"].fillna("").astype(str).str.strip()
    data["company"] = data["company"].fillna("").astype(str).str.strip()
    if bool(data["security_id"].eq("").any() or data["company"].eq("").any()):
        raise DataReadinessError("event-family security identities contain blanks")
    grouped: dict[str, tuple[dict[str, object], ...]] = {}
    for security_id, part in data.groupby("security_id", sort=True):
        grouped[str(security_id)] = tuple(
            part.sort_values(
                ["effective_from_utc", "available_at_utc", "company"],
                kind="stable",
            ).to_dict(orient="records")
        )
    return grouped


def _attach_causal_issuer_companies(
    events: pd.DataFrame,
    identity_intervals: Mapping[str, tuple[Mapping[str, object], ...]],
) -> pd.DataFrame:
    required = {"security_id", "feature_available_at_utc"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise DataReadinessError(
            "event-family source events are missing: " + ", ".join(missing)
        )
    output = events.copy()
    event_times = _utc(output["feature_available_at_utc"], "event availability")
    companies: list[str] = []
    company_available: list[object] = []
    for security_id, event_time in zip(
        output["security_id"].astype(str),
        event_times,
        strict=True,
    ):
        matches: list[Mapping[str, object]] = []
        for identity in identity_intervals.get(security_id, ()):
            available_at = pd.Timestamp(identity["available_at_utc"])
            effective_from = pd.Timestamp(identity["effective_from_utc"])
            effective_to_raw = identity["effective_to_utc"]
            effective_to = (
                None
                if pd.isna(effective_to_raw)
                else pd.Timestamp(effective_to_raw)
            )
            if (
                available_at <= event_time
                and effective_from <= event_time
                and (effective_to is None or event_time < effective_to)
            ):
                matches.append(identity)
        if len(matches) > 1:
            raise DataReadinessError(
                "event-family issuer identity intervals overlap at event time"
            )
        if not matches:
            companies.append("")
            company_available.append(pd.NaT)
            continue
        companies.append(str(matches[0]["company"]))
        company_available.append(pd.Timestamp(matches[0]["available_at_utc"]))
    output["issuer_company"] = companies
    output["issuer_company_available_at_utc"] = pd.Series(
        company_available,
        index=output.index,
        dtype="datetime64[us, UTC]",
    )
    return output


def publish_issuer_event_family_authority(
    *,
    collection_dir: Path,
    collection_audit_path: Path,
    attribution_dir: Path,
    decisions_path: Path,
    policy_path: Path,
    output_directory: Path,
) -> IssuerEventFamilyAuthority:
    """Publish normalized family cohorts from original events and issuer relations."""

    policy = load_swing_event_family_policy(policy_path)
    if output_directory.exists():
        raise DataReadinessError(f"issuer event-family authority is immutable: {output_directory}")
    collection_manifest_path = collection_dir / "_manifest.json"
    attribution_manifest_path = attribution_dir / "_manifest.json"
    collection = _complete_research_manifest(collection_manifest_path, "news collection")
    attribution_history = load_event_attribution_history(attribution_dir)
    identity_path = Path(
        _required_text(
            attribution_history.request,
            "security_identities_path",
        )
    )
    identities, identity_manifest = load_canonical_artifact(
        identity_path,
        expected_type="security_business_label_coverage",
        allow_research=True,
    )
    if identity_manifest.get("artifact_sha256") != attribution_history.request.get(
        "security_identities_sha256"
    ):
        raise DataReadinessError("event-family security identity hash does not verify")
    identity_intervals = _identity_intervals_by_security(identities)
    collection_audit = _json_object(collection_audit_path)
    if (
        not bool(collection_audit.get("passed"))
        or collection_audit.get("request_sha256") != collection.get("request_sha256")
    ):
        raise DataReadinessError("event-family authority requires a passed collection audit")

    decisions, decision_manifest = load_canonical_artifact(
        decisions_path,
        expected_type="decisions",
        allow_research=True,
    )
    decisions = stamp_canonical_decision_ids(decisions)
    decision_times = _utc(decisions["decision_time_utc"], "decision time")
    if bool(decision_times.dt.date.lt(policy.minimum_decision_date).any()):
        raise DataReadinessError("event-family decisions precede the frozen swing start date")
    decisions["decision_time_utc"] = decision_times

    coverage_path = Path(_required_text(collection, "source_collections_path"))
    source_coverage, coverage_manifest = load_canonical_artifact(
        coverage_path,
        expected_type="source_collections",
        allow_research=True,
    )
    if coverage_manifest.get("artifact_sha256") != collection.get("source_collections_sha256"):
        raise DataReadinessError("event-family source coverage hash does not verify")
    _validate_source_family_allowlist(
        source_coverage,
        policy=policy,
        context="source coverage",
    )

    source_records = _records_by_chunk(collection, "news collection")
    relation_records = {
        _required_text(record, "chunk_id"): record
        for record in attribution_history.artifact_records
    }
    excluded = _text_set(collection_audit.get("coverage_blindspot_security_ids", []))
    expected_relations = {
        chunk_id
        for chunk_id, record in source_records.items()
        if str(record.get("security_id", "")) not in excluded
    }
    if set(relation_records) != expected_relations:
        raise DataReadinessError("event attribution inventory does not match eligible news chunks")

    request = {
        "schema": AUTHORITY_SCHEMA,
        "collection_manifest_path": str(collection_manifest_path.resolve()),
        "collection_manifest_sha256": file_sha256(collection_manifest_path),
        "collection_audit_path": str(collection_audit_path.resolve()),
        "collection_audit_sha256": file_sha256(collection_audit_path),
        "attribution_manifest_path": str(attribution_manifest_path.resolve()),
        "attribution_manifest_sha256": file_sha256(attribution_manifest_path),
        "security_identities_path": str(identity_path.resolve()),
        "security_identities_sha256": str(identity_manifest["artifact_sha256"]),
        "decisions_path": str(decisions_path.resolve()),
        "decisions_sha256": str(decision_manifest["artifact_sha256"]),
        "source_coverage_sha256": str(coverage_manifest["artifact_sha256"]),
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": file_sha256(policy_path),
        "classifier_policy_sha256": EVENT_FAMILY_POLICY_SHA256,
        "production_ready": False,
    }
    request_sha256 = _json_sha256(request)
    staging = output_directory.with_name(f".{output_directory.name}.{uuid4().hex}.tmp")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    try:
        unclassified_directory = staging / "unclassified"
        unclassified_directory.mkdir()
        event_parts: list[pd.DataFrame] = []
        unclassified_records: list[dict[str, object]] = []
        relation_channel_counts = {
            "direct_issuer": 0,
            "business_exposure": 0,
            "sector_context": 0,
        }
        for chunk_index, chunk_id in enumerate(
            sorted(relation_records), start=1
        ):
            source_record = source_records[chunk_id]
            relation_record = relation_records[chunk_id]
            source_events, source_manifest = load_canonical_artifact(
                Path(_required_text(source_record, "path")),
                expected_type="events",
                allow_research=True,
            )
            _validate_source_family_allowlist(
                source_events,
                policy=policy,
                context=f"source events for {chunk_id}",
            )
            relations, relation_manifest = load_canonical_artifact(
                Path(_required_text(relation_record, "path")),
                expected_type="event_security_relations",
                allow_research=True,
            )
            if (
                source_manifest.get("artifact_sha256") != source_record.get("sha256")
                or relation_manifest.get("artifact_sha256") != relation_record.get("sha256")
                or _manifest_input(relation_manifest, "source_event_artifact_sha256")
                != source_manifest.get("artifact_sha256")
            ):
                raise DataReadinessError(f"event-family child lineage mismatch: {chunk_id}")
            observed_channels = relations["relation_channel"].astype(str).value_counts()
            unsupported_channels = sorted(
                set(observed_channels.index).difference(relation_channel_counts)
            )
            if unsupported_channels:
                raise DataReadinessError(
                    "event-family authority found unsupported relation channels: "
                    + ", ".join(unsupported_channels)
                )
            for channel, count in observed_channels.items():
                relation_channel_counts[str(channel)] += int(count)
            source_events = _attach_causal_issuer_companies(
                source_events,
                identity_intervals,
            )
            direct_relations = relations.loc[
                relations["relation_channel"].astype(str).eq("direct_issuer")
            ].copy()
            family_chunk = _build_family_events(
                source_events,
                direct_relations,
                policy=policy,
                coverage_known=True,
            )
            classified = family_chunk.loc[
                family_chunk["classification_state"].astype(str).eq("classified")
            ].copy()
            unclassified = family_chunk.loc[
                family_chunk["classification_state"].astype(str).eq("unclassified")
            ].copy()
            if not classified.empty:
                event_parts.append(classified)
            if not unclassified.empty:
                unclassified_path = unclassified_directory / f"{chunk_id}.parquet"
                unclassified_manifest = write_canonical_artifact(
                    unclassified,
                    unclassified_path,
                    artifact_type=UNCLASSIFIED_EVENTS_ARTIFACT_TYPE,
                    audit=_unclassified_event_audit(unclassified),
                    inputs={
                        "request_sha256": request_sha256,
                        "source_event_sha256": str(
                            source_manifest["artifact_sha256"]
                        ),
                        "relation_sha256": str(
                            relation_manifest["artifact_sha256"]
                        ),
                    },
                    production_ready=False,
                )
                unclassified_path.with_name(
                    f"{unclassified_path.name}.lock"
                ).unlink(missing_ok=True)
                _rewrite_artifact_path(
                    unclassified_path,
                    output_directory / "unclassified" / unclassified_path.name,
                )
                unclassified_records.append(
                    {
                        "chunk_id": chunk_id,
                        "path": f"unclassified/{unclassified_path.name}",
                        "sha256": str(
                            unclassified_manifest["artifact_sha256"]
                        ),
                        "rows": len(unclassified),
                        "source_event_sha256": str(
                            source_manifest["artifact_sha256"]
                        ),
                        "relation_sha256": str(
                            relation_manifest["artifact_sha256"]
                        ),
                    }
                )
            del source_events, relations, direct_relations, family_chunk, unclassified
            if chunk_index % 32 == 0:
                gc.collect()
                release_process_memory()
            assert_memory_budget(
                hard_budget_gib=policy.maximum_process_memory_gib,
                headroom_gib=policy.memory_guard_headroom_gib,
                stage=f"event-family chunk {chunk_id}",
            )
        family_events = (
            pd.concat(event_parts, ignore_index=True)
            if event_parts
            else pd.DataFrame(columns=FAMILY_EVENT_COLUMNS)
        )
        family_events = _validate_family_events(family_events)
        eligible_events = family_events.loc[family_events["research_eligible"].astype(bool)].copy()
        assignment_input = eligible_events.rename(
            columns={"family_event_id": "event_id", "source_family": "original_source_family"}
        )
        assignment_input["source_family"] = assignment_input["event_family"]
        assignment_input["sentiment_numeric"] = pd.NA
        assignment_input["relevance"] = assignment_input["relation_score"]
        assignments = build_event_assignments(
            decisions,
            assignment_input,
            windows=policy.windows,
        )
        integrity = assignment_integrity_summary(
            decisions,
            assignment_input,
            assignments,
            windows=policy.windows,
        )
        if integrity["assignment_integrity_errors"]:
            raise DataReadinessError("event-family decision assignment replay failed")
        assignments = _augment_assignments(assignments, eligible_events)
        family_coverage = _build_family_coverage(
            source_coverage,
            relation_chunk_ids=set(relation_records),
            blind_security_ids=excluded,
            policy=policy,
            collection_completed_at=collection.get("completed_at_utc"),
        )
        cohort_audit = _build_cohort_audit(
            family_events,
            assignments,
            family_coverage,
            decisions,
            policy=policy,
        )
        family_status = _family_statuses(
            policy.event_families,
            family_events,
            family_coverage,
        )

        artifacts: dict[str, Mapping[str, object]] = {}
        artifacts["events"] = write_canonical_artifact(
            family_events,
            staging / "family_events.parquet",
            artifact_type=FAMILY_EVENTS_ARTIFACT_TYPE,
            audit=_event_audit(family_events),
            inputs={"request_sha256": request_sha256},
            production_ready=False,
        )
        artifacts["assignments"] = write_canonical_artifact(
            assignments,
            staging / "family_assignments.parquet",
            artifact_type=FAMILY_ASSIGNMENTS_ARTIFACT_TYPE,
            audit=_assignment_audit(assignments),
            inputs={
                "request_sha256": request_sha256,
                "family_events_sha256": str(artifacts["events"]["artifact_sha256"]),
                "assignment_sha256": reconciliation_sha256(assignments),
            },
            production_ready=False,
        )
        artifacts["coverage"] = write_canonical_artifact(
            family_coverage,
            staging / "family_coverage.parquet",
            artifact_type=FAMILY_COVERAGE_ARTIFACT_TYPE,
            audit=_coverage_audit(family_coverage),
            inputs={"request_sha256": request_sha256},
            production_ready=False,
        )
        artifacts["cohort_audit"] = write_canonical_artifact(
            cohort_audit,
            staging / "cohort_audit.parquet",
            artifact_type=COHORT_AUDIT_ARTIFACT_TYPE,
            audit=_cohort_audit(cohort_audit),
            inputs={
                "request_sha256": request_sha256,
                "family_events_sha256": str(artifacts["events"]["artifact_sha256"]),
                "family_assignments_sha256": str(artifacts["assignments"]["artifact_sha256"]),
                "family_coverage_sha256": str(artifacts["coverage"]["artifact_sha256"]),
            },
            production_ready=False,
        )
        for key in artifacts:
            _rewrite_artifact_path(
                staging / _artifact_filename(key),
                output_directory / _artifact_filename(key),
            )
        for path in staging.glob("*.lock"):
            path.unlink(missing_ok=True)
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "state": "complete",
            "request": request,
            "request_sha256": request_sha256,
            "event_family_policy_version": EVENT_FAMILY_POLICY_VERSION,
            "event_family_policy_sha256": EVENT_FAMILY_POLICY_SHA256,
            "event_families": list(policy.event_families),
            "research_source_families": list(policy.source_families),
            "family_status": family_status,
            "assignment_windows_seconds": {
                name: int(window.total_seconds()) for name, window in policy.windows.items()
            },
            "event_rows": len(family_events),
            "unclassified_event_rows": sum(
                _record_rows(record) for record in unclassified_records
            ),
            "unclassified_artifacts": unclassified_records,
            "source_relation_channel_counts": relation_channel_counts,
            "excluded_relation_channel_counts": {
                channel: count
                for channel, count in relation_channel_counts.items()
                if channel != "direct_issuer"
            },
            "classified_event_rows": int(family_events["classification_state"].eq("classified").sum()),
            "research_eligible_event_rows": int(family_events["research_eligible"].astype(bool).sum()),
            "production_eligible_event_rows": 0,
            "assignment_rows": len(assignments),
            "coverage_rows": len(family_coverage),
            "cohort_audit_rows": len(cohort_audit),
            "artifacts": {
                key: _artifact_record(staging / _artifact_filename(key), value)
                for key, value in artifacts.items()
            },
            "memory": memory_audit(
                hard_budget_gib=policy.maximum_process_memory_gib,
                headroom_gib=policy.memory_guard_headroom_gib,
            ).to_record(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "production_ready": False,
            "promotion_blocker": "historical source availability is retrospective/proxy evidence",
        }
        _atomic_json(staging / "_manifest.json", manifest)
        authority = {
            "schema": AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(staging / "_manifest.json"),
            "request_sha256": request_sha256,
            "event_family_policy_sha256": EVENT_FAMILY_POLICY_SHA256,
            "production_ready": False,
        }
        _atomic_json(staging / "_authority.json", authority)
        load_issuer_event_family_authority(staging)
        os.replace(staging, output_directory)
        return load_issuer_event_family_authority(output_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_issuer_event_family_authority(
    directory: Path,
    *,
    expected_authority_sha256: str | None = None,
) -> IssuerEventFamilyAuthority:
    manifest_path = directory / "_manifest.json"
    authority_path = directory / "_authority.json"
    manifest = _json_object(manifest_path)
    authority = _json_object(authority_path)
    if expected_authority_sha256 is not None and file_sha256(authority_path) != expected_authority_sha256:
        raise DataReadinessError("issuer event-family authority identity does not match")
    if (
        manifest.get("schema") != MANIFEST_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("production_ready") is not False
        or authority.get("schema") != AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("production_ready") is not False
    ):
        raise DataReadinessError("issuer event-family authority does not verify")
    request = manifest.get("request")
    if (
        not isinstance(request, dict)
        or _json_sha256(request) != manifest.get("request_sha256")
        or authority.get("request_sha256") != manifest.get("request_sha256")
        or manifest.get("event_family_policy_sha256") != EVENT_FAMILY_POLICY_SHA256
        or authority.get("event_family_policy_sha256") != EVENT_FAMILY_POLICY_SHA256
    ):
        raise DataReadinessError("issuer event-family authority request or policy hash fails")
    expected_files = {
        "_authority.json",
        "_manifest.json",
        "family_events.parquet",
        "family_events.parquet.manifest.json",
        "family_assignments.parquet",
        "family_assignments.parquet.manifest.json",
        "family_coverage.parquet",
        "family_coverage.parquet.manifest.json",
        "cohort_audit.parquet",
        "cohort_audit.parquet.manifest.json",
    }
    observed_files = {path.name for path in directory.iterdir() if path.is_file()}
    if observed_files != expected_files:
        raise DataReadinessError("issuer event-family authority file inventory differs")
    unclassified_records = _unclassified_records(manifest)
    _verify_unclassified_artifacts(
        directory,
        unclassified_records,
        request_sha256=str(manifest["request_sha256"]),
        expected_rows=manifest.get("unclassified_event_rows"),
    )
    frames: dict[str, pd.DataFrame] = {}
    artifact_types = {
        "events": FAMILY_EVENTS_ARTIFACT_TYPE,
        "assignments": FAMILY_ASSIGNMENTS_ARTIFACT_TYPE,
        "coverage": FAMILY_COVERAGE_ARTIFACT_TYPE,
        "cohort_audit": COHORT_AUDIT_ARTIFACT_TYPE,
    }
    manifest_artifacts = manifest.get("artifacts")
    if not isinstance(manifest_artifacts, dict):
        raise DataReadinessError("issuer event-family artifact inventory is malformed")
    for key, artifact_type in artifact_types.items():
        path = directory / _artifact_filename(key)
        frame, child = load_canonical_artifact(path, expected_type=artifact_type, allow_research=True)
        record = manifest_artifacts.get(key)
        if (
            not isinstance(record, dict)
            or record.get("path") != path.name
            or record.get("sha256") != child.get("artifact_sha256")
            or record.get("rows") != len(frame)
            or _manifest_input(child, "request_sha256") != manifest.get("request_sha256")
        ):
            raise DataReadinessError(f"issuer event-family {key} lineage does not verify")
        frames[key] = frame
    assignment_inputs = _artifact_inputs(
        directory / "family_assignments.parquet"
    )
    cohort_inputs = _artifact_inputs(directory / "cohort_audit.parquet")
    if (
        assignment_inputs.get("family_events_sha256")
        != _manifest_artifact_sha256(manifest_artifacts, "events")
        or assignment_inputs.get("assignment_sha256")
        != reconciliation_sha256(frames["assignments"])
        or cohort_inputs.get("family_events_sha256")
        != _manifest_artifact_sha256(manifest_artifacts, "events")
        or cohort_inputs.get("family_assignments_sha256")
        != _manifest_artifact_sha256(manifest_artifacts, "assignments")
        or cohort_inputs.get("family_coverage_sha256")
        != _manifest_artifact_sha256(manifest_artifacts, "coverage")
    ):
        raise DataReadinessError(
            "issuer event-family cross-artifact lineage does not verify"
        )
    _event_audit(frames["events"]).raise_for_failure()
    _assignment_audit(frames["assignments"]).raise_for_failure()
    _coverage_audit(frames["coverage"]).raise_for_failure()
    _cohort_audit(frames["cohort_audit"]).raise_for_failure()
    decisions_path = Path(_required_text(request, "decisions_path"))
    decisions, decisions_manifest = load_canonical_artifact(
        decisions_path,
        expected_type="decisions",
        allow_research=True,
    )
    if decisions_manifest.get("artifact_sha256") != request.get("decisions_sha256"):
        raise DataReadinessError("issuer event-family decision lineage fails")
    decisions = stamp_canonical_decision_ids(decisions)
    decisions["decision_time_utc"] = _utc(
        decisions["decision_time_utc"],
        "decision time",
    )
    policy_path = Path(_required_text(request, "policy_path"))
    if file_sha256(policy_path) != request.get("policy_sha256"):
        raise DataReadinessError("issuer event-family policy file lineage fails")
    policy = load_swing_event_family_policy(policy_path)
    expected_cohort = _build_cohort_audit(
        frames["events"],
        frames["assignments"],
        frames["coverage"],
        decisions,
        policy=policy,
    )
    expected_cohort["abstention_rate"] = pd.to_numeric(
        expected_cohort["abstention_rate"],
        errors="coerce",
    )
    try:
        pd.testing.assert_frame_equal(
            frames["cohort_audit"].reset_index(drop=True),
            expected_cohort.reset_index(drop=True),
            check_exact=True,
            check_dtype=False,
        )
    except AssertionError as exc:
        raise DataReadinessError(
            "issuer event-family cohort semantic replay fails"
        ) from exc
    manifest_sources = _text_tuple(
        manifest.get("research_source_families"),
        "research_source_families",
    )
    if (
        _text_tuple(manifest.get("event_families"), "event_families")
        != EVENT_FAMILIES
        or any(source != source.lower() for source in manifest_sources)
        or not {
            source
            for family in EVENT_FAMILIES
            for source in ALLOWED_SOURCE_FAMILIES_BY_FAMILY[family]
        }.issubset(manifest_sources)
    ):
        raise DataReadinessError("issuer event-family manifest policy inventory fails")
    _validate_observed_source_families(
        frames["events"],
        allowed_sources=manifest_sources,
        context="authority events",
    )
    _validate_observed_source_families(
        frames["coverage"],
        allowed_sources=manifest_sources,
        context="authority coverage",
    )
    _validate_observed_source_families(
        frames["assignments"],
        allowed_sources=manifest_sources,
        context="authority assignments",
        source_column="original_source_family",
    )
    _validate_family_source_pairs(
        frames["events"],
        context="authority events",
    )
    _validate_family_source_pairs(
        frames["coverage"],
        context="authority coverage",
    )
    _validate_family_source_pairs(
        frames["assignments"],
        context="authority assignments",
        source_column="original_source_family",
    )
    expected_family_status = _family_statuses(
        EVENT_FAMILIES,
        frames["events"],
        frames["coverage"],
    )
    manifest_family_status = manifest.get("family_status")
    if (
        not isinstance(manifest_family_status, dict)
        or set(manifest_family_status) != set(EVENT_FAMILIES)
        or not set(str(value) for value in manifest_family_status.values()).issubset(
            FAMILY_STATUSES
        )
        or manifest_family_status != expected_family_status
    ):
        raise DataReadinessError("issuer event-family manifest family status fails")
    if (
        len(frames["events"]) != manifest.get("event_rows")
        or len(frames["assignments"]) != manifest.get("assignment_rows")
        or len(frames["coverage"]) != manifest.get("coverage_rows")
        or len(frames["cohort_audit"]) != manifest.get("cohort_audit_rows")
        or bool(frames["events"]["production_eligible"].astype(bool).any())
        or bool(frames["coverage"]["production_eligible"].astype(bool).any())
    ):
        raise DataReadinessError("issuer event-family authority row totals or research mode fail")
    return IssuerEventFamilyAuthority(
        directory=directory.resolve(),
        events=frames["events"],
        assignments=frames["assignments"],
        coverage=frames["coverage"],
        cohort_audit=frames["cohort_audit"],
        unclassified_artifact_records=tuple(unclassified_records),
        manifest=manifest,
        authority=authority,
    )


def _build_family_events(
    events: pd.DataFrame,
    relations: pd.DataFrame,
    *,
    policy: SwingEventFamilyPolicy,
    coverage_known: bool,
) -> pd.DataFrame:
    _validate_source_family_allowlist(
        events,
        policy=policy,
        context="event-family source events",
    )
    required_relation = {
        "relation_id",
        "event_id",
        "source_security_id",
        "source_ticker",
        "target_security_id",
        "target_ticker",
        "relation_channel",
        "relation_score",
        "feature_available_at_utc",
    }
    missing = sorted(required_relation.difference(relations.columns))
    if missing:
        raise DataReadinessError("event-family relations are missing columns: " + ", ".join(missing))
    if bool(relations["relation_id"].astype(str).duplicated().any()):
        raise DataReadinessError("event-family relations contain duplicate identities")
    classifications = classify_event_families(events)
    event_index = events.set_index("event_id")
    if not event_index.index.is_unique:
        raise DataReadinessError("event-family source events contain duplicate identities")
    grouped = {
        str(event_id): group.to_dict(orient="records")
        for event_id, group in classifications.groupby("event_id", sort=True)
    }
    rows: list[dict[str, object]] = []
    for relation in relations.sort_values("relation_id", kind="stable").to_dict(orient="records"):
        source_event_id = str(relation["event_id"])
        if source_event_id not in event_index.index:
            raise DataReadinessError("event-family relation references a missing source event")
        event = event_index.loc[source_event_id]
        source_security_id = str(event["security_id"])
        source_ticker = str(event["ticker"]).upper()
        if (
            source_security_id != str(relation["source_security_id"])
            or source_ticker != str(relation["source_ticker"]).upper()
        ):
            raise DataReadinessError("event-family source issuer identity conflicts with relation")
        family_rows = grouped.get(source_event_id) or [None]
        for classification in family_rows:
            family = "" if classification is None else str(classification["event_family"])
            classified = bool(family)
            channel = str(relation["relation_channel"])
            source_supported = (
                classified
                and _source_supports_family(
                    source_family=str(event["source_family"]),
                    event_family=family,
                    policy=policy,
                )
            )
            research_eligible = (
                coverage_known
                and classified
                and source_supported
                and channel in policy.eligible_channels
            )
            if not coverage_known:
                exclusion = "unknown_source_coverage"
            elif channel not in policy.eligible_channels:
                exclusion = "not_direct_issuer"
            elif not classified:
                exclusion = "unclassified_event_family"
            elif not source_supported:
                exclusion = "source_not_authoritative_for_family"
            else:
                exclusion = ""
            published_at = pd.Timestamp(event["published_at_utc"])
            event_available = pd.Timestamp(event["feature_available_at_utc"])
            relation_available = pd.Timestamp(relation["feature_available_at_utc"])
            if (
                published_at > event_available
                or event_available > relation_available
            ):
                raise DataReadinessError(
                    "event-family publication or relation availability is backdated"
                )
            feature_available = max(event_available, relation_available)
            family_event_id = _json_sha256(
                {
                    "relation_id": str(relation["relation_id"]),
                    "event_family": family or "unclassified",
                    "policy_sha256": EVENT_FAMILY_POLICY_SHA256,
                }
            )
            rows.append(
                {
                    "family_event_id": family_event_id,
                    "source_event_id": source_event_id,
                    "relation_id": str(relation["relation_id"]),
                    "source_security_id": source_security_id,
                    "source_ticker": source_ticker,
                    "security_id": str(relation["target_security_id"]),
                    "ticker": str(relation["target_ticker"]).upper(),
                    "source_family": str(event["source_family"]).lower(),
                    "event_family": family,
                    "classification_state": "classified" if classified else "unclassified",
                    "classification_rule_id": "" if classification is None else str(classification["classification_rule_id"]),
                    "classification_basis": "" if classification is None else str(classification["classification_basis"]),
                    "matched_text": "" if classification is None else str(classification["matched_text"]),
                    "published_at_utc": published_at,
                    "event_available_at_utc": event_available,
                    "relation_available_at_utc": relation_available,
                    "feature_available_at_utc": feature_available,
                    "availability_policy": str(event["availability_policy"]),
                    "relation_channel": channel,
                    "relation_score": float(relation["relation_score"]),
                    "research_eligible": research_eligible,
                    "production_eligible": False,
                    "exclusion_reason": exclusion,
                    "event_family_policy_version": EVENT_FAMILY_POLICY_VERSION,
                    "event_family_policy_sha256": EVENT_FAMILY_POLICY_SHA256,
                    "schema_version": AUTHORITY_SCHEMA,
                }
            )
    return pd.DataFrame.from_records(rows, columns=FAMILY_EVENT_COLUMNS)


def _validate_family_events(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.loc[:, list(FAMILY_EVENT_COLUMNS)].copy()
    for column in (
        "published_at_utc",
        "event_available_at_utc",
        "relation_available_at_utc",
        "feature_available_at_utc",
    ):
        output[column] = _utc(output[column], column)
    if bool(output["family_event_id"].astype(str).duplicated().any()):
        raise DataReadinessError("event-family authority produced duplicate family events")
    if not output.empty:
        expected_available = pd.concat(
            [output["event_available_at_utc"], output["relation_available_at_utc"]], axis=1
        ).max(axis=1)
        if bool(output["feature_available_at_utc"].ne(expected_available).any()):
            raise DataReadinessError("event-family feature availability is backdated")
    return output.sort_values(
        ["feature_available_at_utc", "family_event_id"], kind="stable"
    ).reset_index(drop=True)


def _augment_assignments(assignments: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    if assignments.empty:
        for column in FAMILY_ASSIGNMENT_EXTRA_COLUMNS:
            assignments[column] = pd.Series(dtype="object")
        return assignments
    metadata = events.rename(columns={"family_event_id": "event_id"})[
        [
            "event_id",
            "event_family",
            "source_family",
            "source_event_id",
            "relation_id",
            "event_family_policy_sha256",
        ]
    ].rename(columns={"source_family": "original_source_family"})
    return assignments.merge(metadata, on="event_id", how="left", validate="many_to_one")


def _build_family_coverage(
    source_coverage: pd.DataFrame,
    *,
    relation_chunk_ids: set[str],
    blind_security_ids: set[str],
    policy: SwingEventFamilyPolicy,
    collection_completed_at: object,
) -> pd.DataFrame:
    _validate_source_family_allowlist(
        source_coverage,
        policy=policy,
        context="event-family source coverage",
    )
    required = {
        "collection_id",
        "chunk_id",
        "security_id",
        "ticker",
        "source_family",
        "requested_start_utc",
        "requested_end_utc",
        "status",
    }
    missing = sorted(required.difference(source_coverage.columns))
    if missing:
        raise DataReadinessError("event-family source coverage is missing: " + ", ".join(missing))
    rows: list[dict[str, object]] = []
    completed_default = pd.to_datetime(collection_completed_at, utc=True, errors="coerce")
    for source in source_coverage.to_dict(orient="records"):
        status = str(source["status"])
        chunk_id = str(source["chunk_id"])
        security_id = str(source["security_id"])
        if security_id in blind_security_ids:
            state = "coverage_blindspot"
        elif status == "observed_empty":
            state = "observed_empty"
        elif status == "observed" and chunk_id in relation_chunk_ids:
            state = "observed_complete"
        else:
            state = "failed_or_unobserved"
        known = state in {"observed_complete", "observed_empty"}
        completed = pd.to_datetime(source.get("completed_at_utc"), utc=True, errors="coerce")
        if pd.isna(completed):
            completed = completed_default
        source_family = str(source["source_family"]).lower().strip()
        for family in policy.event_families:
            if not _source_supports_family(
                source_family=source_family,
                event_family=family,
                policy=policy,
            ):
                continue
            rows.append(
                {
                    "collection_id": str(source["collection_id"]),
                    "chunk_id": chunk_id,
                    "security_id": security_id,
                    "ticker": str(source["ticker"]).upper(),
                    "source_family": source_family,
                    "event_family": family,
                    "requested_start_utc": pd.Timestamp(source["requested_start_utc"]),
                    "requested_end_utc": pd.Timestamp(source["requested_end_utc"]),
                    "completed_at_utc": completed,
                    "coverage_state": state,
                    "missingness_known": known,
                    "zero_event_semantics": {
                        "observed_complete": "observed_history",
                        "observed_empty": "known_zero_events",
                        "coverage_blindspot": "unknown_excluded",
                        "failed_or_unobserved": "unknown_failed",
                    }[state],
                    "research_eligible": known,
                    "production_eligible": False,
                    "schema_version": AUTHORITY_SCHEMA,
                }
            )
    return pd.DataFrame.from_records(rows, columns=FAMILY_COVERAGE_COLUMNS).sort_values(
        ["security_id", "source_family", "event_family", "requested_start_utc"], kind="stable"
    ).reset_index(drop=True)


def _build_cohort_audit(
    events: pd.DataFrame,
    assignments: pd.DataFrame,
    coverage: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    policy: SwingEventFamilyPolicy,
) -> pd.DataFrame:
    eligible = events.loc[events["research_eligible"].astype(bool)].copy()
    assigned = assignments.loc[assignments["status"].astype(str).eq("assigned")].copy()
    decision_sector = decisions.set_index("decision_id").get("sector")
    if decision_sector is None:
        assigned["sector"] = "unknown"
    else:
        assigned["sector"] = assigned["decision_id"].map(decision_sector).fillna("unknown")
    eligible["calendar_month"] = eligible["feature_available_at_utc"].dt.strftime("%Y-%m")
    decisions = decisions.copy()
    decisions["calendar_month"] = decisions["decision_time_utc"].dt.strftime("%Y-%m")
    decisions["sector"] = decisions.get("sector", "unknown")
    max_window = max(policy.windows.values())
    _validate_replicated_family_coverage(coverage)
    coverage_sources = sorted(set(coverage["source_family"].astype(str)))
    known_source_cache: dict[str, set[str]] = {}
    for source_family in coverage_sources:
        source_rows = coverage.loc[
            coverage["source_family"].astype(str).eq(source_family)
        ]
        source_families = sorted(set(source_rows["event_family"].astype(str)))
        if not source_families:
            continue
        known_source_cache[source_family] = _known_coverage_decision_ids(
            coverage,
            decisions,
            family=source_families[0],
            max_window=max_window,
            source_family=source_family,
        )
    records: list[dict[str, object]] = []
    for family in policy.event_families:
        part = eligible.loc[eligible["event_family"].eq(family)]
        assigned_part = assigned.loc[assigned.get("event_family", pd.Series(dtype=str)).eq(family)]
        coverage_sources = sorted(
            set(
                coverage.loc[
                    coverage["event_family"].astype(str).eq(family),
                    "source_family",
                ].astype(str)
            )
        )
        known_by_source = {
            source_family: known_source_cache[source_family]
            for source_family in coverage_sources
            if source_family in known_source_cache
        }
        known_ids = (
            set().union(*known_by_source.values()) if known_by_source else set()
        )
        covered_assignment_parts = [
            assigned_part.loc[
                assigned_part["original_source_family"].astype(str).eq(source_family)
                & assigned_part["decision_id"].astype(str).isin(source_ids)
            ]
            for source_family, source_ids in known_by_source.items()
        ]
        covered_assigned_part = (
            pd.concat(covered_assignment_parts, ignore_index=False)
            if covered_assignment_parts
            else assigned_part.iloc[0:0].copy()
        )
        records.append(
            _cohort_record(
                family,
                "overall",
                "all",
                part,
                covered_assigned_part,
                len(known_ids),
            )
        )
        months = sorted(
            set(decisions.loc[decisions["decision_id"].isin(known_ids), "calendar_month"].astype(str))
            | set(part["calendar_month"].astype(str))
        )
        for month in months:
            month_part = part.loc[part["calendar_month"].eq(month)]
            month_decisions = set(
                decisions.loc[
                    decisions["decision_id"].isin(known_ids)
                    & decisions["calendar_month"].eq(month),
                    "decision_id",
                ].astype(str)
            )
            month_assigned = covered_assigned_part.loc[
                covered_assigned_part["decision_id"].astype(str).isin(
                    month_decisions
                )
            ]
            records.append(
                _cohort_record(
                    family,
                    "calendar_month",
                    str(month),
                    month_part,
                    month_assigned,
                    len(month_decisions),
                )
            )
        sectors = sorted(
            set(decisions.loc[decisions["decision_id"].isin(known_ids), "sector"].astype(str))
            | set(assigned_part["sector"].astype(str))
        )
        for sector in sectors:
            sector_decisions = set(
                decisions.loc[
                    decisions["decision_id"].isin(known_ids)
                    & decisions["sector"].astype(str).eq(sector),
                    "decision_id",
                ].astype(str)
            )
            sector_assignments = covered_assigned_part.loc[
                covered_assigned_part["decision_id"].astype(str).isin(
                    sector_decisions
                )
            ]
            event_ids = set(sector_assignments["event_id"].astype(str))
            sector_events = part.loc[part["family_event_id"].astype(str).isin(event_ids)]
            records.append(
                _cohort_record(
                    family,
                    "sector",
                    str(sector),
                    sector_events,
                    sector_assignments,
                    len(sector_decisions),
                )
            )
        source_families = sorted(
            set(
                coverage.loc[
                    coverage["event_family"].astype(str).eq(family),
                    "source_family",
                ].astype(str)
            )
            | set(part["source_family"].astype(str))
        )
        for source_family in source_families:
            source_events = part.loc[
                part["source_family"].astype(str).eq(source_family)
            ]
            source_ids = known_by_source.get(str(source_family), set())
            source_assignments = assigned_part.loc[
                assigned_part["original_source_family"].astype(str).eq(str(source_family))
                & assigned_part["decision_id"].astype(str).isin(source_ids)
            ]
            records.append(
                _cohort_record(
                    family,
                    "source_family",
                    str(source_family),
                    source_events,
                    source_assignments,
                    len(source_ids),
                )
            )
    return pd.DataFrame.from_records(records, columns=COHORT_AUDIT_COLUMNS)


def _validate_replicated_family_coverage(coverage: pd.DataFrame) -> None:
    comparison_columns = [
        column for column in FAMILY_COVERAGE_COLUMNS if column != "event_family"
    ]
    for source_family, source_rows in coverage.groupby("source_family", sort=True):
        families = sorted(set(source_rows["event_family"].astype(str)))
        if not families:
            continue
        reference = (
            source_rows.loc[
                source_rows["event_family"].astype(str).eq(families[0]),
                comparison_columns,
            ]
            .sort_values(
                ["security_id", "chunk_id", "requested_start_utc"],
                kind="stable",
            )
            .reset_index(drop=True)
        )
        for family in families[1:]:
            candidate = (
                source_rows.loc[
                    source_rows["event_family"].astype(str).eq(family),
                    comparison_columns,
                ]
                .sort_values(
                    ["security_id", "chunk_id", "requested_start_utc"],
                    kind="stable",
                )
                .reset_index(drop=True)
            )
            try:
                pd.testing.assert_frame_equal(
                    candidate,
                    reference,
                    check_exact=True,
                    check_dtype=False,
                )
            except AssertionError as exc:
                raise DataReadinessError(
                    "event-family coverage differs across replicated families for "
                    f"source {source_family}"
                ) from exc


def _known_coverage_decision_ids(
    coverage: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    family: str,
    max_window: pd.Timedelta,
    source_family: str | None = None,
) -> set[str]:
    eligible = coverage.loc[
        coverage["event_family"].astype(str).eq(family)
        & coverage["missingness_known"].astype(bool)
    ].copy()
    if source_family is not None:
        eligible = eligible.loc[
            eligible["source_family"].astype(str).eq(source_family)
        ]
    if eligible.empty:
        return set()
    eligible["requested_start_utc"] = _utc(
        eligible["requested_start_utc"], "coverage start"
    )
    eligible["requested_end_utc"] = _utc(
        eligible["requested_end_utc"], "coverage end"
    )
    decisions_by_security = {
        str(security_id): part
        for security_id, part in decisions.groupby("security_id", sort=False)
    }
    known: set[str] = set()
    window_ns = int(max_window.value)
    for security_id, intervals in eligible.groupby("security_id", sort=False):
        decision_part = decisions_by_security.get(str(security_id))
        if decision_part is None or decision_part.empty:
            continue
        ordered_intervals = intervals.sort_values(
            ["requested_start_utc", "requested_end_utc"], kind="stable"
        )
        # Pandas may store parsed UTC timestamps at microsecond resolution. Convert
        # explicitly because Timedelta.value is always expressed in nanoseconds.
        starts = ordered_intervals["requested_start_utc"].to_numpy(
            dtype="datetime64[ns]"
        ).astype(np.int64)
        ends = ordered_intervals["requested_end_utc"].to_numpy(
            dtype="datetime64[ns]"
        ).astype(np.int64)
        decision_ns = decision_part["decision_time_utc"].to_numpy(
            dtype="datetime64[ns]"
        ).astype(np.int64)
        positions = np.searchsorted(
            starts,
            decision_ns - window_ns,
            side="right",
        ) - 1
        valid_position = positions >= 0
        safe_positions = positions.clip(min=0)
        covered = valid_position & (ends[safe_positions] >= decision_ns)
        known.update(
            decision_part.loc[covered, "decision_id"].astype(str)
        )
    return known


def _cohort_record(
    family: str,
    dimension_type: str,
    dimension_value: str,
    events: pd.DataFrame,
    assignments: pd.DataFrame,
    known_decisions: int,
) -> dict[str, object]:
    assigned_count = assignments["decision_id"].nunique() if "decision_id" in assignments else 0
    if assigned_count > known_decisions:
        raise DataReadinessError(
            "event-family cohort assigned decisions exceed known coverage"
        )
    abstention = known_decisions - assigned_count
    return {
        "event_family": family,
        "dimension_type": dimension_type,
        "dimension_value": dimension_value,
        "event_count": len(events),
        "security_count": events["security_id"].nunique() if "security_id" in events else 0,
        "assigned_decision_count": assigned_count,
        "known_coverage_decision_count": known_decisions,
        "abstention_count": abstention,
        "abstention_rate": float(abstention / known_decisions) if known_decisions else pd.NA,
        "first_event_available_at_utc": events["feature_available_at_utc"].min() if not events.empty else pd.NaT,
        "last_event_available_at_utc": events["feature_available_at_utc"].max() if not events.empty else pd.NaT,
        "schema_version": AUTHORITY_SCHEMA,
    }


def _event_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    failures = len(set(FAMILY_EVENT_COLUMNS).difference(frame.columns))
    if not failures and not frame.empty:
        failures += int(frame["family_event_id"].astype(str).duplicated().sum())
        failures += int(frame["production_eligible"].astype(bool).sum())
        failures += int(
            (
                frame["research_eligible"].astype(bool)
                & (
                    frame["classification_state"].astype(str).ne("classified")
                    | frame["relation_channel"].astype(str).ne("direct_issuer")
                )
            ).sum()
        )
    return _audit("issuer_event_family_events", failures, len(frame), "family, issuer, relation, and availability verify")


def _unclassified_event_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    failures = len(set(FAMILY_EVENT_COLUMNS).difference(frame.columns))
    if not failures and not frame.empty:
        failures += int(frame["family_event_id"].astype(str).duplicated().sum())
        failures += int(frame["classification_state"].astype(str).ne("unclassified").sum())
        failures += int(frame["event_family"].astype(str).ne("").sum())
        failures += int(frame["research_eligible"].astype(bool).sum())
        failures += int(frame["production_eligible"].astype(bool).sum())
    return _audit(
        "issuer_event_family_unclassified_events",
        failures,
        len(frame),
        "unclassified issuer events remain explicit audit evidence",
    )


def _assignment_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    failures = len(set(FAMILY_ASSIGNMENT_EXTRA_COLUMNS).difference(frame.columns))
    if not failures and not frame.empty:
        available = _utc(frame["feature_available_at_utc"], "assignment availability")
        decision = pd.to_datetime(frame["decision_time_utc"], utc=True, errors="coerce")
        assigned = frame["status"].astype(str).eq("assigned")
        failures += int((assigned & (decision.isna() | available.gt(decision))).sum())
    return _audit("issuer_event_family_assignments", failures, len(frame), "exact decision assignment cutoff verifies")


def _coverage_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    failures = len(set(FAMILY_COVERAGE_COLUMNS).difference(frame.columns))
    if not failures and not frame.empty:
        failures += int(frame["production_eligible"].astype(bool).sum())
        failures += int(
            (
                frame["research_eligible"].astype(bool)
                != frame["missingness_known"].astype(bool)
            ).sum()
        )
    return _audit("issuer_event_family_coverage", failures, len(frame), "known zero and unknown coverage remain distinct")


def _cohort_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    failures = len(set(COHORT_AUDIT_COLUMNS).difference(frame.columns))
    if not failures and not frame.empty:
        assigned = pd.to_numeric(
            frame["assigned_decision_count"], errors="coerce"
        )
        known = pd.to_numeric(
            frame["known_coverage_decision_count"], errors="coerce"
        )
        abstention = pd.to_numeric(frame["abstention_count"], errors="coerce")
        rate = pd.to_numeric(frame["abstention_rate"], errors="coerce")
        invalid_counts = (
            assigned.isna()
            | known.isna()
            | abstention.isna()
            | assigned.lt(0)
            | known.lt(0)
            | abstention.lt(0)
            | assigned.gt(known)
            | abstention.ne(known - assigned)
        )
        expected_rate = abstention.div(known.where(known.ne(0)))
        invalid_rate = (
            (known.eq(0) & rate.notna())
            | (
                known.gt(0)
                & (
                    rate.isna()
                    | ~np.isclose(
                        rate.fillna(0).to_numpy(dtype=float),
                        expected_rate.fillna(0).to_numpy(dtype=float),
                        rtol=0.0,
                        atol=1e-12,
                    )
                )
            )
            | (rate.notna() & ((rate < 0) | (rate > 1)))
        )
        failures += int(invalid_counts.sum())
        failures += int(invalid_rate.sum())
    return _audit("issuer_event_family_cohort_audit", failures, len(frame), "cohort counts and abstention bounds verify")


def _audit(name: str, failures: int, rows: int, detail: str) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name=name,
                status="pass" if failures == 0 else "fail",
                failures=failures,
                rows_checked=rows,
                detail=detail,
            ),
        )
    )


def _complete_research_manifest(path: Path, name: str) -> dict[str, object]:
    manifest = _json_object(path)
    if manifest.get("status") != "complete" or bool(manifest.get("production_ready")):
        raise DataReadinessError(f"{name} must be a complete research-only artifact")
    return manifest


def _records_by_chunk(manifest: Mapping[str, object], name: str) -> dict[str, Mapping[str, object]]:
    raw = manifest.get("artifacts")
    if not isinstance(raw, list):
        raise DataReadinessError(f"{name} artifact inventory is malformed")
    result: dict[str, Mapping[str, object]] = {}
    for value in raw:
        if not isinstance(value, dict):
            raise DataReadinessError(f"{name} artifact record is malformed")
        chunk_id = _required_text(value, "chunk_id")
        if chunk_id in result:
            raise DataReadinessError(f"{name} contains duplicate chunk IDs")
        result[chunk_id] = value
    return result


def _unclassified_records(
    manifest: Mapping[str, object],
) -> list[Mapping[str, object]]:
    raw = manifest.get("unclassified_artifacts")
    if not isinstance(raw, list):
        raise DataReadinessError(
            "issuer event-family unclassified inventory is malformed"
        )
    records: list[Mapping[str, object]] = []
    chunk_ids: set[str] = set()
    for value in raw:
        if not isinstance(value, dict):
            raise DataReadinessError(
                "issuer event-family unclassified record is malformed"
            )
        chunk_id = _required_text(value, "chunk_id")
        if chunk_id in chunk_ids:
            raise DataReadinessError(
                "issuer event-family unclassified chunks are duplicated"
            )
        chunk_ids.add(chunk_id)
        records.append(value)
    return records


def _verify_unclassified_artifacts(
    directory: Path,
    records: list[Mapping[str, object]],
    *,
    request_sha256: str,
    expected_rows: object,
) -> None:
    root = directory.resolve()
    child_directory = root / "unclassified"
    if not child_directory.is_dir():
        raise DataReadinessError(
            "issuer event-family unclassified directory is missing"
        )
    expected_files = {
        name
        for record in records
        for name in (
            f"{_required_text(record, 'chunk_id')}.parquet",
            f"{_required_text(record, 'chunk_id')}.parquet.manifest.json",
        )
    }
    if (
        {path.name for path in child_directory.iterdir()} != expected_files
        or any(path.is_dir() for path in child_directory.iterdir())
    ):
        raise DataReadinessError(
            "issuer event-family unclassified child inventory differs"
        )
    total_rows = 0
    for record in records:
        chunk_id = _required_text(record, "chunk_id")
        expected_relative = f"unclassified/{chunk_id}.parquet"
        if record.get("path") != expected_relative:
            raise DataReadinessError(
                "issuer event-family unclassified path does not verify"
            )
        path = child_directory / f"{chunk_id}.parquet"
        frame, child = load_canonical_artifact(
            path,
            expected_type=UNCLASSIFIED_EVENTS_ARTIFACT_TYPE,
            allow_research=True,
        )
        child_inputs = child.get("inputs")
        rows = record.get("rows")
        if (
            not isinstance(rows, int)
            or isinstance(rows, bool)
            or rows <= 0
            or len(frame) != rows
            or child.get("artifact_sha256") != record.get("sha256")
            or not isinstance(child_inputs, dict)
            or child_inputs.get("request_sha256") != request_sha256
            or child_inputs.get("source_event_sha256")
            != record.get("source_event_sha256")
            or child_inputs.get("relation_sha256")
            != record.get("relation_sha256")
        ):
            raise DataReadinessError(
                "issuer event-family unclassified lineage does not verify"
            )
        _unclassified_event_audit(frame).raise_for_failure()
        total_rows += len(frame)
    if (
        not isinstance(expected_rows, int)
        or isinstance(expected_rows, bool)
        or expected_rows < 0
        or total_rows != expected_rows
    ):
        raise DataReadinessError(
            "issuer event-family unclassified row total does not verify"
        )


def _artifact_filename(key: str) -> str:
    return {
        "events": "family_events.parquet",
        "assignments": "family_assignments.parquet",
        "coverage": "family_coverage.parquet",
        "cohort_audit": "cohort_audit.parquet",
    }[key]


def _record_rows(record: Mapping[str, object]) -> int:
    value = record.get("rows")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DataReadinessError("event-family artifact has invalid row count")
    return value


def _artifact_record(path: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    rows = manifest.get("rows")
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 0:
        raise DataReadinessError("canonical event-family artifact has invalid row count")
    return {
        "path": path.name,
        "sha256": str(manifest["artifact_sha256"]),
        "rows": rows,
    }


def _manifest_input(manifest: Mapping[str, object], key: str) -> object:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise DataReadinessError("canonical artifact inputs are malformed")
    return inputs.get(key)


def _artifact_inputs(path: Path) -> Mapping[str, object]:
    manifest = _json_object(path.with_suffix(path.suffix + ".manifest.json"))
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise DataReadinessError("event-family child inputs are malformed")
    return inputs


def _manifest_artifact_sha256(
    artifacts: Mapping[str, object], key: str
) -> object:
    record = artifacts.get(key)
    if not isinstance(record, dict):
        raise DataReadinessError("event-family root artifact record is malformed")
    return record.get("sha256")


def _rewrite_artifact_path(path: Path, final_path: Path) -> None:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = _json_object(manifest_path)
    manifest["artifact_path"] = str(final_path.resolve())
    _atomic_json(manifest_path, manifest)


def _required_text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DataReadinessError(f"event-family lineage has invalid {key}")
    return value.strip()


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise DataReadinessError(f"event-family policy has invalid {name}")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise DataReadinessError(f"event-family policy has invalid {name}")
    return result


def _text_set(value: object) -> set[str]:
    if not isinstance(value, list):
        raise DataReadinessError("event-family exclusion inventory is malformed")
    return {str(item) for item in value}


def _utc(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if bool(parsed.isna().any()):
        raise DataReadinessError(f"{name} contains invalid UTC timestamps")
    return parsed


def _json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataReadinessError(f"expected JSON object: {path}")
    return {str(key): item for key, item in value.items()}


def _json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
