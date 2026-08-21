from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditCheck,
    CanonicalAuditReport,
)
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
)
from market_predictor.security_labels import (
    CurrentProfileEvidence,
    MembershipEvidence,
    SecurityBusinessLabelAssignment,
    SecurityBusinessLabelSet,
    SecurityLabelPolicy,
    assign_security_business_labels,
    load_security_label_policy,
    profile_terms_from_text,
)
from market_predictor.core.errors import DataReadinessError

SECURITY_LABEL_ARTIFACT_SCHEMA = "security.business_label_artifact.v1"
RelationUse = Literal["exposure", "context"]

ASSIGNMENT_COLUMNS = (
    "security_id",
    "ticker",
    "company",
    "business_tag",
    "label_type",
    "match_terms",
    "tag_rank",
    "confidence",
    "relation_use",
    "effective_from_utc",
    "effective_to_utc",
    "available_at_utc",
    "knowledge_scope",
    "availability_policy",
    "source_uri",
    "source_content_sha256",
    "evidence_sha256",
    "taxonomy_version",
    "taxonomy_sha256",
    "assignment_policy_version",
    "assignment_sha256",
    "assignment_set_sha256",
)


@dataclass(frozen=True, slots=True)
class SecurityLabelArtifact:
    assignments: pd.DataFrame
    coverage: pd.DataFrame
    audit: CanonicalAuditReport
    summary: dict[str, object]
    inputs: dict[str, str]


@dataclass(frozen=True, slots=True)
class _Identity:
    security_id: str
    ticker: str
    company: str
    industry: str
    effective_from_utc: pd.Timestamp
    effective_to_utc: pd.Timestamp | None


@dataclass(frozen=True, slots=True)
class _SelectedAssignment:
    assignment: SecurityBusinessLabelAssignment
    evidence_kind: Literal["membership", "profile"]
    match_terms: tuple[str, ...]


def build_security_label_artifact(
    *,
    memberships_path: Path,
    universe_path: Path,
    profiles_path: Path,
    training_dataset_path: Path,
    policy_path: Path,
) -> SecurityLabelArtifact:
    """Build point-in-time business tags and explicit training coverage.

    Historical membership industry is a context-only research proxy. Current
    profile evidence can establish exposure relations only from its observed
    timestamp forward. It is never backdated into the training horizon.
    """

    memberships, membership_manifest = load_canonical_artifact(
        memberships_path,
        expected_type="memberships",
        allow_research=True,
    )
    profiles, profile_manifest = load_canonical_artifact(
        profiles_path,
        expected_type="security_profiles_current",
        allow_research=True,
    )
    training, training_manifest = load_canonical_artifact(
        training_dataset_path,
        expected_type="swing_dataset",
        allow_research=True,
        columns=("security_id", "ticker", "label_eligible"),
    )
    universe = pd.read_parquet(universe_path)
    policy = load_security_label_policy(policy_path)

    _require_columns(
        memberships,
        {
            "security_id",
            "ticker",
            "sector",
            "industry",
            "effective_from_utc",
            "effective_to_utc",
            "available_at_utc",
            "availability_policy",
        },
        "memberships",
    )
    _require_columns(
        universe,
        {
            "security_id",
            "ticker",
            "company",
            "effective_from_utc",
            "effective_to_utc",
        },
        "point-in-time universe",
    )
    _require_columns(
        profiles,
        {
            "security_id",
            "ticker",
            "long_description",
            "source_document_id",
            "source_content_sha256",
            "observed_at_utc",
            "available_at_utc",
            "knowledge_scope",
        },
        "current profiles",
    )

    eligible = training.loc[_strict_bool(training["label_eligible"])].copy()
    if eligible.empty:
        raise DataReadinessError("training dataset has no label-eligible rows")
    eligible["security_id"] = eligible["security_id"].astype(str)
    eligible["ticker"] = eligible["ticker"].astype(str).str.upper()
    training_counts = eligible.groupby(["security_id", "ticker"], observed=True).size().rename("training_rows").reset_index()
    training_security_ids = frozenset(eligible["security_id"].unique())

    identities = _prepare_identities(
        memberships,
        universe,
        training_security_ids,
    )
    missing_memberships = sorted(training_security_ids.difference(identity.security_id for identity in identities))
    if missing_memberships:
        raise DataReadinessError(f"training securities missing point-in-time memberships: {missing_memberships[:10]}")

    profile_artifact_available_at = _timestamp(profile_manifest["created_at_utc"])
    as_of = max(
        _artifact_as_of(memberships, profiles),
        profile_artifact_available_at.to_pydatetime(),
    )
    membership_hash = str(membership_manifest["artifact_sha256"])
    membership_evidence = tuple(
        MembershipEvidence(
            security_id=identity.security_id,
            sector=str(
                memberships.loc[
                    _identity_mask(memberships, identity),
                    "sector",
                ].iloc[0]
            ),
            industry=identity.industry or "unknown",
            effective_from_utc=identity.effective_from_utc.to_pydatetime(),
            effective_to_utc=_optional_datetime(identity.effective_to_utc),
            available_at_utc=_timestamp(
                memberships.loc[
                    _identity_mask(memberships, identity),
                    "available_at_utc",
                ].iloc[0]
            ).to_pydatetime(),
            availability_policy=str(
                memberships.loc[
                    _identity_mask(memberships, identity),
                    "availability_policy",
                ].iloc[0]
            ),
            source_uri=str(memberships_path.resolve()),
            source_published_at_utc=_timestamp(
                memberships.loc[
                    _identity_mask(memberships, identity),
                    "available_at_utc",
                ].iloc[0]
            ).to_pydatetime(),
            source_content_sha256=membership_hash,
        )
        for identity in identities
    )
    membership_sets = assign_security_business_labels(
        policy,
        membership_evidence,
        as_of_utc=as_of,
    )
    membership_sets_by_security: dict[
        str,
        list[SecurityBusinessLabelSet],
    ] = {}
    for label_set in membership_sets:
        membership_sets_by_security.setdefault(
            label_set.security_id,
            [],
        ).append(label_set)
    eligible_profiles = profiles.loc[profiles["security_id"].astype(str).isin(training_security_ids)].copy()
    _validate_profile_scope(eligible_profiles)
    profile_evidence_list: list[CurrentProfileEvidence] = []
    for row in eligible_profiles.itertuples(index=False):
        observed_at = max(
            _timestamp(row.observed_at_utc),
            profile_artifact_available_at,
        )
        active_membership = _active_membership_set(
            membership_sets_by_security.get(
                str(row.security_id),
                (),
            ),
            observed_at,
        )
        if active_membership is None:
            raise DataReadinessError(f"profile has no active membership evidence: {row.security_id}")
        profile_evidence_list.append(
            CurrentProfileEvidence(
                security_id=str(row.security_id),
                profile_terms=profile_terms_from_text(
                    policy,
                    str(row.long_description),
                ),
                membership_label_ids=tuple(assignment.label_id for assignment in active_membership.assignments),
                observed_at_utc=observed_at.to_pydatetime(),
                source_uri=str(row.source_document_id),
                source_published_at_utc=observed_at.to_pydatetime(),
                source_content_sha256=str(row.source_content_sha256),
            )
        )
    profile_sets = assign_security_business_labels(
        policy,
        (),
        tuple(profile_evidence_list),
        as_of_utc=as_of,
    )
    assignments = _flatten_assignments(
        policy=policy,
        identities=identities,
        membership_sets=membership_sets,
        profile_sets=profile_sets,
        profiles=eligible_profiles,
    )
    coverage = _build_coverage(
        identities=identities,
        membership_sets=membership_sets,
        profile_sets=profile_sets,
        profiles=eligible_profiles,
        training_counts=training_counts,
    )
    checks = _audit_artifact(
        assignments,
        coverage,
        expected_security_ids=training_security_ids,
        training_rows=len(eligible),
    )
    audit = CanonicalAuditReport(checks=checks)
    audit.raise_for_failure()
    inputs = {
        "memberships_sha256": membership_hash,
        "profiles_sha256": str(profile_manifest["artifact_sha256"]),
        "training_dataset_sha256": str(training_manifest["artifact_sha256"]),
        "universe_sha256": file_sha256(universe_path),
        "taxonomy_sha256": policy.taxonomy_sha256,
    }
    summary: dict[str, object] = {
        "schema": SECURITY_LABEL_ARTIFACT_SCHEMA,
        "training_rows": len(eligible),
        "training_security_ids": len(training_security_ids),
        "training_ticker_histories": int(training_counts[["security_id", "ticker"]].drop_duplicates().shape[0]),
        "assignment_rows": len(assignments),
        "coverage_rows": len(coverage),
        "historical_assigned_security_ids": int(
            coverage.loc[
                coverage["historical_disposition"].eq("assigned_historical_context_proxy"),
                "security_id",
            ].nunique()
        ),
        "historical_insufficient_security_ids": int(
            coverage.loc[
                coverage["historical_disposition"].eq("insufficient_historical_evidence"),
                "security_id",
            ].nunique()
        ),
        "current_profile_security_ids": int(eligible_profiles["security_id"].astype(str).nunique()),
        "current_profile_assigned_security_ids": len({item.security_id for item in profile_sets if item.disposition == "assigned"}),
        "historical_exposure_training_ready": False,
        "historical_exposure_blocker": (
            "historical tags are membership context proxies; SEC Item 1 or equivalent point-in-time business evidence is still required"
        ),
        "inputs": inputs,
    }
    return SecurityLabelArtifact(
        assignments=assignments,
        coverage=coverage,
        audit=audit,
        summary=summary,
        inputs=inputs,
    )


def _prepare_identities(
    memberships: pd.DataFrame,
    universe: pd.DataFrame,
    training_security_ids: frozenset[str],
) -> tuple[_Identity, ...]:
    keys = [
        "security_id",
        "ticker",
        "effective_from_utc",
        "effective_to_utc",
    ]
    left = memberships.loc[memberships["security_id"].astype(str).isin(training_security_ids)].copy()
    left["ticker"] = left["ticker"].astype(str).str.upper()
    right = universe.loc[:, [*keys, "company"]].copy()
    right["ticker"] = right["ticker"].astype(str).str.upper()
    merged = left.merge(
        right,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    if bool(merged["_merge"].ne("both").any()):
        raise DataReadinessError("membership intervals do not map one-to-one to company identities")
    if bool(merged["company"].fillna("").astype(str).str.strip().eq("").any()):
        raise DataReadinessError("membership identity has an empty company")
    values = []
    for row in merged.itertuples(index=False):
        values.append(
            _Identity(
                security_id=str(row.security_id),
                ticker=str(row.ticker),
                company=str(row.company).strip(),
                industry=str(row.industry or "").strip(),
                effective_from_utc=_timestamp(row.effective_from_utc),
                effective_to_utc=_optional_timestamp(row.effective_to_utc),
            )
        )
    return tuple(
        sorted(
            values,
            key=lambda item: (
                item.security_id,
                item.effective_from_utc,
                item.ticker,
            ),
        )
    )


def _flatten_assignments(
    *,
    policy: SecurityLabelPolicy,
    identities: Sequence[_Identity],
    membership_sets: Sequence[SecurityBusinessLabelSet],
    profile_sets: Sequence[SecurityBusinessLabelSet],
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    identity_by_key = {
        (
            item.security_id,
            item.effective_from_utc,
        ): item
        for item in identities
    }
    profile_rows = {str(row.security_id): row for row in profiles.itertuples(index=False)}
    profile_by_security = {item.security_id: item for item in profile_sets}
    records: list[dict[str, object]] = []
    membership_by_security: dict[
        str,
        list[SecurityBusinessLabelSet],
    ] = {}
    for label_set in membership_sets:
        membership_by_security.setdefault(
            label_set.security_id,
            [],
        ).append(label_set)
        identity = identity_by_key[
            (
                label_set.security_id,
                pd.Timestamp(label_set.effective_from_utc),
            )
        ]
        profile_set = profile_by_security.get(label_set.security_id)
        end = _optional_timestamp(label_set.effective_to_utc)
        if (
            profile_set is not None
            and profile_set.disposition == "assigned"
            and _contains_moment(
                label_set,
                pd.Timestamp(profile_set.effective_from_utc),
            )
        ):
            end = pd.Timestamp(profile_set.effective_from_utc)
        selected = tuple(
            _SelectedAssignment(
                assignment=assignment,
                evidence_kind="membership",
                match_terms=_membership_match_terms(
                    policy,
                    assignment,
                    identity.industry,
                ),
            )
            for assignment in label_set.assignments
        )
        records.extend(
            _assignment_records(
                identity=identity,
                selected=selected,
                effective_from=pd.Timestamp(label_set.effective_from_utc),
                effective_to=end,
            )
        )

    for profile_set in profile_sets:
        if profile_set.disposition != "assigned":
            continue
        observed = pd.Timestamp(profile_set.effective_from_utc)
        active_membership = _active_membership_set(
            membership_by_security.get(profile_set.security_id, ()),
            observed,
        )
        if active_membership is None:
            raise DataReadinessError(f"current profile has no active point-in-time membership: {profile_set.security_id}")
        identity = identity_by_key[
            (
                active_membership.security_id,
                pd.Timestamp(active_membership.effective_from_utc),
            )
        ]
        raw_profile = profile_rows[profile_set.security_id]
        profile_terms = frozenset(
            profile_terms_from_text(
                policy,
                str(raw_profile.long_description),
            )
        )
        selected_by_label: dict[str, _SelectedAssignment] = {}
        for assignment in profile_set.assignments:
            selected_by_label[assignment.label_id] = _SelectedAssignment(
                assignment=assignment,
                evidence_kind="profile",
                match_terms=_profile_match_terms(
                    policy,
                    assignment,
                    profile_terms,
                ),
            )
        for assignment in active_membership.assignments:
            if assignment.label_id not in selected_by_label and len(selected_by_label) < policy.max_leaf_labels:
                selected_by_label[assignment.label_id] = _SelectedAssignment(
                    assignment=assignment,
                    evidence_kind="membership",
                    match_terms=_membership_match_terms(
                        policy,
                        assignment,
                        identity.industry,
                    ),
                )
        selected = tuple(
            sorted(
                selected_by_label.values(),
                key=lambda item: (
                    item.assignment.dimension,
                    item.assignment.label_id,
                ),
            )
        )
        if len(selected) > policy.max_leaf_labels:
            raise DataReadinessError(f"{profile_set.security_id} exceeds the active tag limit")
        membership_end = _optional_timestamp(
            active_membership.effective_to_utc
        )
        profile_end = _optional_timestamp(
            profile_set.effective_to_utc
        )
        effective_end = _earliest_timestamp(
            membership_end,
            profile_end,
        )
        records.extend(
            _assignment_records(
                identity=identity,
                selected=selected,
                effective_from=observed,
                effective_to=effective_end,
            )
        )
        if (
            profile_end is not None
            and effective_end == profile_end
            and (
                membership_end is None
                or profile_end < membership_end
            )
        ):
            membership_selected = tuple(
                _SelectedAssignment(
                    assignment=assignment,
                    evidence_kind="membership",
                    match_terms=_membership_match_terms(
                        policy,
                        assignment,
                        identity.industry,
                    ),
                )
                for assignment in active_membership.assignments
            )
            records.extend(
                _assignment_records(
                    identity=identity,
                    selected=membership_selected,
                    effective_from=profile_end,
                    effective_to=membership_end,
                )
            )
    output = pd.DataFrame.from_records(
        records,
        columns=ASSIGNMENT_COLUMNS,
    )
    if output.empty:
        raise DataReadinessError("security label artifact produced no assignments")
    return output.sort_values(
        [
            "security_id",
            "effective_from_utc",
            "tag_rank",
            "business_tag",
        ],
        kind="stable",
    ).reset_index(drop=True)


def _assignment_records(
    *,
    identity: _Identity,
    selected: Sequence[_SelectedAssignment],
    effective_from: pd.Timestamp,
    effective_to: pd.Timestamp | None,
) -> list[dict[str, object]]:
    if effective_to is not None and effective_to <= effective_from:
        return []
    ordered = tuple(
        sorted(
            selected,
            key=lambda item: (
                _dimension_rank(item.assignment.dimension),
                item.assignment.label_id,
            ),
        )
    )
    set_material = {
        "security_id": identity.security_id,
        "ticker": identity.ticker,
        "effective_from_utc": effective_from.isoformat(),
        "effective_to_utc": (effective_to.isoformat() if effective_to is not None else None),
        "labels": [
            {
                "label_id": item.assignment.label_id,
                "evidence_sha256": item.assignment.evidence_sha256,
                "evidence_kind": item.evidence_kind,
            }
            for item in ordered
        ],
    }
    assignment_set_sha256 = _json_sha256(set_material)
    rows: list[dict[str, object]] = []
    for rank, selected_assignment in enumerate(ordered, start=1):
        assignment = selected_assignment.assignment
        relation_use: RelationUse = (
            "exposure" if selected_assignment.evidence_kind == "profile" and assignment.dimension in {"offering", "driver"} else "context"
        )
        row: dict[str, object] = {
            "security_id": identity.security_id,
            "ticker": identity.ticker,
            "company": identity.company,
            "business_tag": assignment.label_id,
            "label_type": assignment.dimension,
            "match_terms": json.dumps(
                list(selected_assignment.match_terms),
                separators=(",", ":"),
            ),
            "tag_rank": rank,
            "confidence": (0.85 if selected_assignment.evidence_kind == "profile" else 0.60),
            "relation_use": relation_use,
            "effective_from_utc": effective_from,
            "effective_to_utc": (effective_to if effective_to is not None else pd.NaT),
            "available_at_utc": pd.Timestamp(assignment.available_at_utc),
            "knowledge_scope": assignment.knowledge_scope,
            "availability_policy": assignment.availability_policy,
            "source_uri": assignment.source_uri,
            "source_content_sha256": (assignment.source_content_sha256),
            "evidence_sha256": assignment.evidence_sha256,
            "taxonomy_version": assignment.taxonomy_version,
            "taxonomy_sha256": assignment.taxonomy_sha256,
            "assignment_policy_version": (assignment.assignment_policy_version),
            "assignment_set_sha256": assignment_set_sha256,
        }
        row["assignment_sha256"] = _json_sha256(row)
        rows.append(row)
    return rows


def _build_coverage(
    *,
    identities: Sequence[_Identity],
    membership_sets: Sequence[SecurityBusinessLabelSet],
    profile_sets: Sequence[SecurityBusinessLabelSet],
    profiles: pd.DataFrame,
    training_counts: pd.DataFrame,
) -> pd.DataFrame:
    membership_by_key = {
        (
            item.security_id,
            pd.Timestamp(item.effective_from_utc),
        ): item
        for item in membership_sets
    }
    profile_by_security = {item.security_id: item for item in profile_sets}
    profile_ids = frozenset(profiles["security_id"].astype(str))
    count_map = {(str(row.security_id), str(row.ticker)): int(row.training_rows) for row in training_counts.itertuples(index=False)}
    rows: list[dict[str, object]] = []
    for identity in identities:
        label_set = membership_by_key[(identity.security_id, identity.effective_from_utc)]
        profile_set = profile_by_security.get(identity.security_id)
        current_interval = identity.effective_to_utc is None
        profile_disposition = (
            (
                "observed_profile_assigned"
                if profile_set is not None and profile_set.disposition == "assigned"
                else "observed_profile_no_controlled_terms"
            )
            if identity.security_id in profile_ids and current_interval
            else ("provider_missing_current_profile" if current_interval else "not_current_at_collection")
        )
        rows.append(
            {
                "security_id": identity.security_id,
                "ticker": identity.ticker,
                "company": identity.company,
                "effective_from_utc": identity.effective_from_utc,
                "effective_to_utc": (identity.effective_to_utc if identity.effective_to_utc is not None else pd.NaT),
                "available_at_utc": pd.Timestamp(
                    label_set.available_at_utc
                ),
                "training_rows": count_map.get(
                    (identity.security_id, identity.ticker),
                    0,
                ),
                "historical_label_count": len(label_set.assignments),
                "historical_business_tags": json.dumps(
                    [item.label_id for item in label_set.assignments],
                    separators=(",", ":"),
                ),
                "historical_disposition": (
                    "assigned_historical_context_proxy" if label_set.disposition == "assigned" else "insufficient_historical_evidence"
                ),
                "historical_exposure_training_eligible": False,
                "current_profile_disposition": profile_disposition,
                "current_profile_label_count": (len(profile_set.assignments) if profile_set is not None and current_interval else 0),
                "current_profile_effective_to_utc": (
                    pd.Timestamp(profile_set.effective_to_utc)
                    if profile_set is not None
                    and current_interval
                    and profile_set.effective_to_utc is not None
                    else pd.NaT
                ),
                "current_profile_business_tags": json.dumps(
                    ([item.label_id for item in profile_set.assignments] if profile_set is not None and current_interval else []),
                    separators=(",", ":"),
                ),
                "current_profile_exposure_eligible": bool(
                    profile_set is not None
                    and current_interval
                    and any(item.dimension in {"offering", "driver"} for item in profile_set.assignments)
                ),
            }
        )
    return (
        pd.DataFrame.from_records(rows)
        .sort_values(
            ["security_id", "effective_from_utc", "ticker"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def _audit_artifact(
    assignments: pd.DataFrame,
    coverage: pd.DataFrame,
    *,
    expected_security_ids: frozenset[str],
    training_rows: int,
) -> tuple[CanonicalAuditCheck, ...]:
    coverage_ids = frozenset(coverage["security_id"].astype(str))
    missing = expected_security_ids.difference(coverage_ids)
    active_failures = _active_limit_failures(assignments)
    leaked_profiles = int(
        (
            assignments["knowledge_scope"].eq("current_inference_only")
            & (
                pd.to_datetime(
                    assignments["effective_from_utc"],
                    utc=True,
                )
                < pd.to_datetime(
                    assignments["available_at_utc"],
                    utc=True,
                )
            )
        ).sum()
    )
    return (
        CanonicalAuditCheck(
            name="training_security_coverage",
            status="pass" if not missing else "fail",
            failures=len(missing),
            rows_checked=len(expected_security_ids),
            detail=("every label-eligible training security has an explicit assignment or insufficient-evidence disposition"),
        ),
        CanonicalAuditCheck(
            name="training_row_reconciliation",
            status=("pass" if int(coverage["training_rows"].sum()) == training_rows else "fail"),
            failures=abs(int(coverage["training_rows"].sum()) - training_rows),
            rows_checked=training_rows,
            detail="coverage training rows reconcile to the frozen dataset",
        ),
        CanonicalAuditCheck(
            name="maximum_three_active_tags",
            status="pass" if active_failures == 0 else "fail",
            failures=active_failures,
            rows_checked=len(assignments),
            detail="no security has more than three active business tags",
        ),
        CanonicalAuditCheck(
            name="current_profile_availability",
            status="pass" if leaked_profiles == 0 else "fail",
            failures=leaked_profiles,
            rows_checked=len(assignments),
            detail="current profile tags never precede first observation",
        ),
    )


def _active_limit_failures(assignments: pd.DataFrame) -> int:
    failures = 0
    for _, group in assignments.groupby("security_id", observed=True):
        starts = pd.to_datetime(group["effective_from_utc"], utc=True)
        ends = pd.to_datetime(group["effective_to_utc"], utc=True)
        points = sorted(set(starts) | {value for value in ends if pd.notna(value)})
        for point in points:
            active = starts.le(point) & (ends.isna() | ends.gt(point))
            if int(active.sum()) > 3:
                failures += 1
    return failures


def _membership_match_terms(
    policy: SecurityLabelPolicy,
    assignment: SecurityBusinessLabelAssignment,
    industry: str,
) -> tuple[str, ...]:
    normalized_industry = _normalize(industry)
    if not normalized_industry:
        return ()
    matching_rule = next(
        (rule for rule in policy.labels if rule.label_id == assignment.label_id),
        None,
    )
    if matching_rule is not None and normalized_industry not in matching_rule.membership_industry_exact:
        return ()
    return (normalized_industry,)


def _profile_match_terms(
    policy: SecurityLabelPolicy,
    assignment: SecurityBusinessLabelAssignment,
    observed_terms: frozenset[str],
) -> tuple[str, ...]:
    rule = next(
        (item for item in policy.labels if item.label_id == assignment.label_id),
        None,
    )
    if rule is None:
        return ()
    return tuple(sorted(rule.profile_phrase_exact.intersection(observed_terms)))


def _active_membership_set(
    sets: Iterable[SecurityBusinessLabelSet],
    timestamp: pd.Timestamp,
) -> SecurityBusinessLabelSet | None:
    active = [item for item in sets if _contains_moment(item, timestamp)]
    if len(active) > 1:
        raise DataReadinessError("profile maps to overlapping membership label sets")
    return active[0] if active else None


def _contains_moment(
    label_set: SecurityBusinessLabelSet,
    timestamp: pd.Timestamp,
) -> bool:
    start = pd.Timestamp(label_set.effective_from_utc)
    end = _optional_timestamp(label_set.effective_to_utc)
    return start <= timestamp and (end is None or timestamp < end)


def _identity_mask(
    frame: pd.DataFrame,
    identity: _Identity,
) -> pd.Series:
    starts = pd.to_datetime(frame["effective_from_utc"], utc=True)
    ends = pd.to_datetime(frame["effective_to_utc"], utc=True)
    return (
        frame["security_id"].astype(str).eq(identity.security_id)
        & frame["ticker"].astype(str).str.upper().eq(identity.ticker)
        & starts.eq(identity.effective_from_utc)
        & (ends.eq(identity.effective_to_utc) if identity.effective_to_utc is not None else ends.isna())
    )


def _artifact_as_of(
    memberships: pd.DataFrame,
    profiles: pd.DataFrame,
) -> datetime:
    candidates = [
        pd.to_datetime(
            memberships["effective_from_utc"],
            utc=True,
        ).max(),
        pd.to_datetime(
            memberships["effective_to_utc"],
            utc=True,
        ).max(),
        pd.to_datetime(
            memberships["available_at_utc"],
            utc=True,
        ).max(),
    ]
    if not profiles.empty:
        candidates.extend(
            [
                pd.to_datetime(
                    profiles["observed_at_utc"],
                    utc=True,
                ).max(),
                pd.to_datetime(
                    profiles["available_at_utc"],
                    utc=True,
                ).max(),
            ]
        )
    valid = [item for item in candidates if pd.notna(item)]
    if not valid:
        raise DataReadinessError("security label inputs contain no valid timestamps")
    return cast(datetime, max(valid).to_pydatetime())


def _validate_profile_scope(profiles: pd.DataFrame) -> None:
    invalid = set(profiles["knowledge_scope"].fillna("").astype(str)).difference({"current_inference_only"})
    if invalid:
        raise DataReadinessError(f"profile artifact contains invalid knowledge scopes: {invalid}")
    observed = pd.to_datetime(profiles["observed_at_utc"], utc=True)
    available = pd.to_datetime(profiles["available_at_utc"], utc=True)
    if bool(observed.isna().any() or available.ne(observed).any()):
        raise DataReadinessError("profile observation and availability timestamps must match")


def _strict_bool(values: pd.Series) -> pd.Series:
    if str(values.dtype) in {"bool", "boolean"}:
        return values.fillna(False).astype(bool)
    normalized = values.fillna("").astype(str).str.lower()
    invalid = sorted(set(normalized).difference({"true", "false"}))
    if invalid:
        raise DataReadinessError(f"label_eligible contains non-boolean values: {invalid[:10]}")
    return normalized.eq("true")


def _timestamp(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise DataReadinessError("security label timestamp is timezone-naive")
    return timestamp.tz_convert("UTC")


def _optional_timestamp(value: object) -> pd.Timestamp | None:
    return None if value is None or pd.isna(value) else _timestamp(value)


def _optional_datetime(value: object) -> datetime | None:
    timestamp = _optional_timestamp(value)
    return timestamp.to_pydatetime() if timestamp is not None else None


def _earliest_timestamp(
    first: pd.Timestamp | None,
    second: pd.Timestamp | None,
) -> pd.Timestamp | None:
    values = [
        value
        for value in (first, second)
        if value is not None
    ]
    return min(values) if values else None


def _dimension_rank(value: str) -> int:
    return {"offering": 0, "end_market": 1, "driver": 2}[value]


def _normalize(value: object) -> str:
    return " ".join("".join(character if character.isalnum() else " " for character in str(value).lower()).split())


def _json_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataReadinessError(f"{name} is missing columns: {missing}")
