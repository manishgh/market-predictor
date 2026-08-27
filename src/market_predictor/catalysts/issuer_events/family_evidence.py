"""Horizon-neutral issuer-family event and source-coverage evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import load_canonical_artifact
from market_predictor.catalysts.issuer_events.classification import (
    ALLOWED_SOURCE_FAMILIES_BY_FAMILY,
    EVENT_FAMILIES,
    EVENT_FAMILY_POLICY_SHA256,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.issuer_family_combined_envelope import (
    FAMILY_COVERAGE_ARTIFACT_TYPE,
    FAMILY_EVENTS_ARTIFACT_TYPE,
    UNCLASSIFIED_EVENTS_ARTIFACT_TYPE,
    verify_issuer_family_combined_envelope,
)

FAMILY_STATUSES: Final = frozenset({"admitted", "blocked_missing_source", "absent"})
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


@dataclass(frozen=True, slots=True)
class IssuerFamilyEvidence:
    directory: Path
    events: pd.DataFrame
    coverage: pd.DataFrame
    unclassified_artifact_records: tuple[Mapping[str, object], ...]
    combined_envelope_sha256: str
    full_inventory_sha256: str
    neutral_projection_sha256: str
    manifest: Mapping[str, object]
    authority: Mapping[str, object]


def load_issuer_family_evidence(
    directory: Path,
    *,
    expected_authority_sha256: str | None = None,
    verify_unclassified_semantics: bool = True,
) -> IssuerFamilyEvidence:
    """Strictly project neutral evidence from a retained combined v2 envelope."""

    envelope = verify_issuer_family_combined_envelope(
        directory,
        expected_authority_sha256=expected_authority_sha256,
        expected_policy_sha256=EVENT_FAMILY_POLICY_SHA256,
    )
    events, _ = load_canonical_artifact(
        envelope.artifacts["events"].path,
        expected_type=FAMILY_EVENTS_ARTIFACT_TYPE,
        allow_research=True,
    )
    coverage, _ = load_canonical_artifact(
        envelope.artifacts["coverage"].path,
        expected_type=FAMILY_COVERAGE_ARTIFACT_TYPE,
        allow_research=True,
    )
    issuer_family_event_audit(events).raise_for_failure()
    issuer_family_coverage_audit(coverage).raise_for_failure()
    if verify_unclassified_semantics:
        _verify_unclassified_semantics(
            envelope.directory,
            envelope.unclassified_artifact_records,
        )

    manifest_sources = _text_tuple(
        envelope.manifest.get("research_source_families"),
        "research_source_families",
    )
    if (
        _text_tuple(envelope.manifest.get("event_families"), "event_families")
        != EVENT_FAMILIES
        or any(source != source.lower() for source in manifest_sources)
        or not {
            source
            for family in EVENT_FAMILIES
            for source in ALLOWED_SOURCE_FAMILIES_BY_FAMILY[family]
        }.issubset(manifest_sources)
    ):
        raise DataReadinessError("issuer-family evidence policy inventory differs")
    validate_observed_source_families(
        events,
        allowed_sources=manifest_sources,
        context="issuer-family events",
    )
    validate_observed_source_families(
        coverage,
        allowed_sources=manifest_sources,
        context="issuer-family coverage",
    )
    validate_family_source_pairs(events, context="issuer-family events")
    validate_family_source_pairs(coverage, context="issuer-family coverage")
    validate_replicated_family_coverage(coverage)

    expected_status = family_statuses(EVENT_FAMILIES, events, coverage)
    manifest_status = envelope.manifest.get("family_status")
    if (
        not isinstance(manifest_status, dict)
        or set(manifest_status) != set(EVENT_FAMILIES)
        or not set(str(value) for value in manifest_status.values()).issubset(FAMILY_STATUSES)
        or manifest_status != expected_status
    ):
        raise DataReadinessError("issuer-family evidence family status inventory differs")
    if (
        len(events) != envelope.manifest.get("event_rows")
        or len(coverage) != envelope.manifest.get("coverage_rows")
        or bool(events["production_eligible"].astype(bool).any())
        or bool(coverage["production_eligible"].astype(bool).any())
    ):
        raise DataReadinessError("issuer-family evidence row totals or research mode differ")
    return IssuerFamilyEvidence(
        directory=envelope.directory,
        events=events,
        coverage=coverage,
        unclassified_artifact_records=envelope.unclassified_artifact_records,
        combined_envelope_sha256=envelope.authority_sha256,
        full_inventory_sha256=envelope.full_inventory_sha256,
        neutral_projection_sha256=envelope.neutral_projection_sha256,
        manifest=envelope.manifest,
        authority=envelope.authority,
    )


def validate_observed_source_families(
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
        raise DataReadinessError(f"{context} contains source families outside policy: {detail}")


def validate_family_source_pairs(
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


def family_statuses(
    families: tuple[str, ...],
    events: pd.DataFrame,
    coverage: pd.DataFrame,
) -> dict[str, str]:
    statuses: dict[str, str] = {}
    for family in families:
        family_coverage = coverage.loc[coverage["event_family"].astype(str).eq(family)]
        usable_source = bool(
            not family_coverage.empty
            and family_coverage["research_eligible"].astype(bool).any()
        )
        family_events = events.loc[
            events["event_family"].astype(str).eq(family)
            & events["research_eligible"].astype(bool)
        ]
        statuses[family] = (
            "blocked_missing_source"
            if not usable_source
            else "absent"
            if family_events.empty
            else "admitted"
        )
    return statuses


def validate_replicated_family_coverage(coverage: pd.DataFrame) -> None:
    comparison_columns = [
        column for column in FAMILY_COVERAGE_COLUMNS if column != "event_family"
    ]
    for source_family, source_rows in coverage.groupby("source_family", sort=True):
        families = sorted(set(source_rows["event_family"].astype(str)))
        if not families:
            continue
        reference = _ordered_coverage(source_rows, families[0], comparison_columns)
        for family in families[1:]:
            candidate = _ordered_coverage(source_rows, family, comparison_columns)
            try:
                pd.testing.assert_frame_equal(
                    candidate,
                    reference,
                    check_exact=True,
                    check_dtype=False,
                )
            except AssertionError as exc:
                raise DataReadinessError(
                    "issuer-family coverage differs across replicated families for "
                    f"source {source_family}"
                ) from exc


def _ordered_coverage(
    rows: pd.DataFrame,
    family: str,
    columns: list[str],
) -> pd.DataFrame:
    return (
        rows.loc[rows["event_family"].astype(str).eq(family), columns]
        .sort_values(["security_id", "chunk_id", "requested_start_utc"], kind="stable")
        .reset_index(drop=True)
    )


def _verify_unclassified_semantics(
    directory: Path,
    records: tuple[Mapping[str, object], ...],
) -> None:
    for record in records:
        relative = str(record["path"])
        frame, _ = load_canonical_artifact(
            directory / relative,
            expected_type=UNCLASSIFIED_EVENTS_ARTIFACT_TYPE,
            allow_research=True,
        )
        issuer_family_unclassified_event_audit(frame).raise_for_failure()


def issuer_family_event_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
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
    return _audit(
        "issuer_event_family_events",
        failures,
        len(frame),
        "family, issuer, relation, and availability verify",
    )


def issuer_family_unclassified_event_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
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


def issuer_family_coverage_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    failures = len(set(FAMILY_COVERAGE_COLUMNS).difference(frame.columns))
    if not failures and not frame.empty:
        failures += int(frame["production_eligible"].astype(bool).sum())
        failures += int(
            (
                frame["research_eligible"].astype(bool)
                != frame["missingness_known"].astype(bool)
            ).sum()
        )
    return _audit(
        "issuer_event_family_coverage",
        failures,
        len(frame),
        "known zero and unknown coverage remain distinct",
    )


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


def _text_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise DataReadinessError(f"issuer-family policy has invalid {name}")
    result = tuple(str(item).strip() for item in value)
    if any(not item for item in result) or len(set(result)) != len(result):
        raise DataReadinessError(f"issuer-family policy has invalid {name}")
    return result
