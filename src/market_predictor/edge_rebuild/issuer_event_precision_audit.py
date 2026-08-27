"""Immutable, causal precision audit for issuer event-family authorities."""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import re
import shutil
import sqlite3
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Final
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import (
    canonical_artifact_columns,
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.catalysts.issuer_events import classification as issuer_event_classification
from market_predictor.catalysts.issuer_events.attribution_history import (
    EventAttributionHistory,
    load_event_attribution_history,
)
from market_predictor.catalysts.issuer_events.classification import (
    ALLOWED_SOURCE_FAMILIES_BY_FAMILY,
    EVENT_FAMILIES,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.issuer_event_family_authority import (
    IssuerEventFamilyAuthority,
    load_issuer_event_family_authority,
)
from market_predictor.resources import (
    assert_memory_budget,
    memory_audit,
    release_process_memory,
)

POLICY_SCHEMA: Final = "market_predictor.issuer_event_precision_audit.v2"
SAMPLE_AUTHORITY_SCHEMA: Final = "edge_rebuild.issuer_event_precision_sample_authority.v2"
SAMPLE_MANIFEST_SCHEMA: Final = "edge_rebuild.issuer_event_precision_sample_manifest.v2"
FINAL_AUTHORITY_SCHEMA: Final = "edge_rebuild.issuer_event_precision_audit_authority.v2"
FINAL_MANIFEST_SCHEMA: Final = "edge_rebuild.issuer_event_precision_audit_manifest.v2"
SAMPLE_ARTIFACT_TYPE: Final = "issuer_event_precision_sample"
REVIEWS_ARTIFACT_TYPE: Final = "issuer_event_precision_reviews"
METRICS_ARTIFACT_TYPE: Final = "issuer_event_precision_family_metrics"
RULE_VARIANT_METRICS_ARTIFACT_TYPE: Final = "issuer_event_precision_rule_variant_metrics"
SOURCE_AUTHORIZATION_SHA256: Final = hashlib.sha256(
    json.dumps(
        {family: list(ALLOWED_SOURCE_FAMILIES_BY_FAMILY[family]) for family in EVENT_FAMILIES},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
_YES_NO_UNCERTAIN: Final = frozenset({"yes", "no", "uncertain"})
_YES_NO: Final = frozenset({"yes", "no"})
_DECISION_FIELDS: Final = (
    "family_correct",
    "issuer_target_correct",
    "event_announced_or_completed",
)
_CORRECTION_FIELDS: Final = (
    "correct_family",
    "action_subject_text",
    "false_positive_reason",
)
_NORMALIZE_PATTERN: Final = re.compile(r"[^a-z0-9]+")
_FORBIDDEN_EVIDENCE_TERMS: Final = (
    "return",
    "price",
    "outcome",
    "probability",
    "target_return",
)

SAMPLE_COLUMNS: Final = (
    "sample_id",
    "sample_role",
    "inference_cluster_id",
    "paired_inferential_sample_id",
    "family_event_id",
    "source_event_id",
    "relation_id",
    "source_security_id",
    "source_ticker",
    "security_id",
    "ticker",
    "source_family",
    "source",
    "proposed_event_family",
    "classification_rule_id",
    "classification_basis",
    "matched_text",
    "title",
    "summary",
    "text",
    "published_at_utc",
    "event_available_at_utc",
    "relation_available_at_utc",
    "feature_available_at_utc",
    "availability_policy",
    "relation_channel",
    "issuer_company",
    "identity_effective_from_utc",
    "identity_effective_to_utc",
    "identity_available_at_utc",
    "identity_status",
    "calendar_quarter",
    "rule_variant",
    "normalized_title_sha256",
    "multi_target_title",
    "multi_label_event",
    "stratum_id",
    "cluster_selection_sha256",
    "row_selection_sha256",
    "schema_version",
)
REVIEW_TEMPLATE_COLUMNS: Final = (
    "sample_id",
    "family_event_id",
    "event_family",
    "reviewer_slot",
    "reviewer_id",
    "family_correct",
    "issuer_target_correct",
    "event_announced_or_completed",
    "correct_family",
    "action_subject_text",
    "false_positive_reason",
    "comments",
)
ADJUDICATION_TEMPLATE_COLUMNS: Final = (
    "sample_id",
    "family_event_id",
    "event_family",
    "adjudicator_id",
    "family_correct",
    "issuer_target_correct",
    "event_announced_or_completed",
    "correct_family",
    "action_subject_text",
    "false_positive_reason",
    "comments",
)
REVIEW_COLUMNS: Final = (
    "sample_id",
    "sample_role",
    "inference_cluster_id",
    "family_event_id",
    "event_family",
    "reviewer_one_id",
    "reviewer_two_id",
    "adjudicator_id",
    "reviewer_one_family_correct",
    "reviewer_one_issuer_target_correct",
    "reviewer_one_event_announced_or_completed",
    "reviewer_two_family_correct",
    "reviewer_two_issuer_target_correct",
    "reviewer_two_event_announced_or_completed",
    "adjudication_required",
    "resolution_state",
    "family_correct",
    "issuer_target_correct",
    "event_announced_or_completed",
    "joint_correct",
    "wrong_issuer",
    "correct_family",
    "action_subject_text",
    "false_positive_reason",
    "comments",
    "schema_version",
)
METRIC_COLUMNS: Final = (
    "event_family",
    "population_eligible_events",
    "population_clusters",
    "population_issuers",
    "inferential_cluster_count",
    "diagnostic_count",
    "resolved_inferential_count",
    "minimum_population_clusters",
    "minimum_population_issuers",
    "family_successes",
    "family_lcb",
    "minimum_family_lcb",
    "issuer_successes",
    "issuer_lcb",
    "minimum_issuer_lcb",
    "event_successes",
    "event_lcb",
    "minimum_event_lcb",
    "joint_successes",
    "joint_lcb",
    "minimum_joint_lcb",
    "wrong_issuer_count",
    "diagnostic_wrong_issuer_count",
    "unresolved_count",
    "identity_unresolved_count",
    "family_reviewer_agreement",
    "family_reviewer_kappa",
    "family_kappa_estimable",
    "issuer_reviewer_agreement",
    "issuer_reviewer_kappa",
    "issuer_kappa_estimable",
    "event_reviewer_agreement",
    "event_reviewer_kappa",
    "event_kappa_estimable",
    "minimum_reviewer_agreement",
    "minimum_reviewer_kappa",
    "failed_rule_variant_count",
    "status",
    "blocker_reasons",
    "schema_version",
)
RULE_VARIANT_METRIC_COLUMNS: Final = (
    "event_family",
    "rule_variant",
    "population_clusters",
    "inferential_cluster_count",
    "joint_successes",
    "joint_lcb",
    "minimum_joint_lcb",
    "minimum_sample_clusters",
    "gate_applicable",
    "status",
    "blocker_reasons",
    "schema_version",
)


@dataclass(frozen=True, slots=True)
class FamilyPrecisionPolicy:
    sample_clusters: int
    minimum_population_clusters: int
    minimum_population_issuers: int
    minimum_family_lcb: float
    minimum_issuer_lcb: float
    minimum_event_lcb: float
    minimum_joint_lcb: float


@dataclass(frozen=True, slots=True)
class IssuerEventPrecisionPolicy:
    confidence_level: float
    reviewers_per_item: int
    unresolved_policy: str
    require_distinct_reviewers: bool
    no_wrong_issuer: bool
    maximum_process_memory_gib: float
    memory_guard_headroom_gib: float
    paired_wrong_issuer_diagnostics_per_cluster: int
    minimum_reviewer_agreement: float
    minimum_reviewer_kappa: float
    minimum_kappa_decisions: int
    rule_variant_gate_minimum_population_clusters: int
    minimum_rule_variant_sample_clusters: int
    minimum_rule_variant_joint_lcb: float
    families: Mapping[str, FamilyPrecisionPolicy]


@dataclass(frozen=True, slots=True)
class IssuerEventPrecisionSample:
    directory: Path
    sample: pd.DataFrame
    manifest: Mapping[str, object]
    authority: Mapping[str, object]
    source_authority: IssuerEventFamilyAuthority | None


@dataclass(frozen=True, slots=True)
class IssuerEventPrecisionAudit:
    directory: Path
    reviews: pd.DataFrame
    family_metrics: pd.DataFrame
    rule_variant_metrics: pd.DataFrame
    manifest: Mapping[str, object]
    authority: Mapping[str, object]
    source_authority: IssuerEventFamilyAuthority | None


def load_issuer_event_precision_policy(path: Path) -> IssuerEventPrecisionPolicy:
    """Load and strictly validate the frozen event precision policy."""

    with path.open("rb") as handle:
        raw = tomllib.load(handle)
    if raw.get("schema_version") != POLICY_SCHEMA:
        raise DataReadinessError(f"unsupported issuer-event precision policy: {path}")
    if int(raw.get("reviewers_per_item", 0)) != 2:
        raise DataReadinessError("issuer-event precision audit requires two reviewers")
    if raw.get("unresolved_policy") != "failure":
        raise DataReadinessError("unresolved precision reviews must count as failures")
    if raw.get("require_distinct_reviewers") is not True:
        raise DataReadinessError("issuer-event precision reviewers must be distinct")
    if raw.get("no_wrong_issuer") is not True:
        raise DataReadinessError("issuer-event precision policy must reject wrong issuers")
    confidence = _bounded_float(raw, "confidence_level", lower=0.5, upper=1.0)
    maximum_memory = _bounded_float(raw, "maximum_process_memory_gib", lower=0.0, upper=5.0, inclusive=True)
    if maximum_memory <= 0:
        raise DataReadinessError("issuer-event precision memory budget must be positive and at most 5 GiB")
    memory_headroom = _bounded_float(raw, "memory_guard_headroom_gib", lower=0.0, upper=maximum_memory)
    reviewer_agreement = _threshold(raw, "minimum_reviewer_agreement")
    reviewer_kappa = _threshold(raw, "minimum_reviewer_kappa")
    raw_families = raw.get("family")
    if not isinstance(raw_families, dict) or set(raw_families) != set(EVENT_FAMILIES):
        raise DataReadinessError("issuer-event precision policy must define every frozen family exactly")
    families: dict[str, FamilyPrecisionPolicy] = {}
    expected_keys = {
        "sample_clusters",
        "minimum_population_clusters",
        "minimum_population_issuers",
        "minimum_family_lcb",
        "minimum_issuer_lcb",
        "minimum_event_lcb",
        "minimum_joint_lcb",
    }
    for family in EVENT_FAMILIES:
        value = raw_families.get(family)
        if not isinstance(value, dict) or set(value) != expected_keys:
            raise DataReadinessError(f"issuer-event precision family policy is malformed: {family}")
        families[family] = FamilyPrecisionPolicy(
            sample_clusters=_positive_int(value, "sample_clusters"),
            minimum_population_clusters=_positive_int(value, "minimum_population_clusters"),
            minimum_population_issuers=_positive_int(value, "minimum_population_issuers"),
            minimum_family_lcb=_threshold(value, "minimum_family_lcb"),
            minimum_issuer_lcb=_threshold(value, "minimum_issuer_lcb"),
            minimum_event_lcb=_threshold(value, "minimum_event_lcb"),
            minimum_joint_lcb=_threshold(value, "minimum_joint_lcb"),
        )
    return IssuerEventPrecisionPolicy(
        confidence_level=confidence,
        reviewers_per_item=2,
        unresolved_policy="failure",
        require_distinct_reviewers=True,
        no_wrong_issuer=True,
        maximum_process_memory_gib=maximum_memory,
        memory_guard_headroom_gib=memory_headroom,
        paired_wrong_issuer_diagnostics_per_cluster=_positive_int(raw, "paired_wrong_issuer_diagnostics_per_cluster"),
        minimum_reviewer_agreement=reviewer_agreement,
        minimum_reviewer_kappa=reviewer_kappa,
        minimum_kappa_decisions=_positive_int(raw, "minimum_kappa_decisions"),
        rule_variant_gate_minimum_population_clusters=_positive_int(raw, "rule_variant_gate_minimum_population_clusters"),
        minimum_rule_variant_sample_clusters=_positive_int(raw, "minimum_rule_variant_sample_clusters"),
        minimum_rule_variant_joint_lcb=_threshold(raw, "minimum_rule_variant_joint_lcb"),
        families=families,
    )


def publish_issuer_event_precision_sample(
    *,
    authority_directory: Path,
    policy_path: Path,
    output_directory: Path,
) -> IssuerEventPrecisionSample:
    """Publish a deterministic causal evidence sample and blank review ledgers."""

    if output_directory.exists():
        raise DataReadinessError(f"issuer-event precision sample is immutable: {output_directory}")
    policy = load_issuer_event_precision_policy(policy_path)
    issuer_authority = load_issuer_event_family_authority(authority_directory)
    _guard_memory(policy, "issuer-event authority publication replay")
    authority_path = authority_directory / "_authority.json"
    authority_sha256 = file_sha256(authority_path)
    sample, population, rule_variant_population = _build_deterministic_sample(
        issuer_authority,
        policy=policy,
        policy_sha256=file_sha256(policy_path),
        authority_sha256=authority_sha256,
    )
    reviewer_one = _review_template(sample, reviewer_slot=1)
    reviewer_two = _review_template(sample, reviewer_slot=2)
    adjudication = _adjudication_template(sample)
    request = {
        "schema": SAMPLE_AUTHORITY_SCHEMA,
        "issuer_event_authority_directory": str(authority_directory.resolve()),
        "issuer_event_authority_path": str(authority_path.resolve()),
        "issuer_event_authority_sha256": authority_sha256,
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": file_sha256(policy_path),
        "source_authorization_sha256": SOURCE_AUTHORIZATION_SHA256,
        "production_ready": False,
    }
    request_sha256 = _json_sha256(request)
    staging = _new_staging(output_directory)
    try:
        sample_path = staging / "sample.parquet"
        sample_manifest = write_canonical_artifact(
            sample,
            sample_path,
            artifact_type=SAMPLE_ARTIFACT_TYPE,
            audit=_sample_audit(sample),
            inputs={"request_sha256": request_sha256},
            production_ready=False,
        )
        _remove_lock(sample_path)
        _rewrite_artifact_path(sample_path, output_directory / sample_path.name)
        _write_csv(staging / "reviewer_one_template.csv", reviewer_one)
        _write_csv(staging / "reviewer_two_template.csv", reviewer_two)
        _write_csv(staging / "adjudication_template.csv", adjudication)
        manifest = {
            "schema": SAMPLE_MANIFEST_SCHEMA,
            "state": "complete",
            "request": request,
            "request_sha256": request_sha256,
            "event_families": list(EVENT_FAMILIES),
            "population": population,
            "rule_variant_population": rule_variant_population,
            "inferential_sample_counts": {
                family: int((sample["proposed_event_family"].eq(family) & sample["sample_role"].eq("inferential")).sum())
                for family in EVENT_FAMILIES
            },
            "diagnostic_sample_counts": {
                family: int((sample["proposed_event_family"].eq(family) & sample["sample_role"].eq("paired_wrong_issuer_diagnostic")).sum())
                for family in EVENT_FAMILIES
            },
            "artifacts": {
                "sample": _artifact_record(sample_path, sample_manifest),
                "reviewer_one_template": _file_record(staging / "reviewer_one_template.csv", len(reviewer_one)),
                "reviewer_two_template": _file_record(staging / "reviewer_two_template.csv", len(reviewer_two)),
                "adjudication_template": _file_record(staging / "adjudication_template.csv", len(adjudication)),
            },
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "memory": memory_audit(
                hard_budget_gib=policy.maximum_process_memory_gib,
                headroom_gib=policy.memory_guard_headroom_gib,
            ).to_record(),
            "production_ready": False,
            "training_eligible": False,
            "alerts_eligible": False,
        }
        _atomic_json(staging / "_manifest.json", manifest)
        root = {
            "schema": SAMPLE_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(staging / "_manifest.json"),
            "request_sha256": request_sha256,
            "production_ready": False,
        }
        _atomic_json(staging / "_authority.json", root)
        load_issuer_event_precision_sample(staging)
        os.replace(staging, output_directory)
        return load_issuer_event_precision_sample(output_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_issuer_event_precision_sample(
    directory: Path,
    *,
    expected_authority_sha256: str | None = None,
    retain_source_authority: bool = False,
) -> IssuerEventPrecisionSample:
    """Strictly load and causally replay an immutable precision sample."""

    manifest_path = directory / "_manifest.json"
    authority_path = directory / "_authority.json"
    manifest = _json_object(manifest_path)
    authority = _json_object(authority_path)
    if expected_authority_sha256 is not None and (file_sha256(authority_path) != expected_authority_sha256):
        raise DataReadinessError("issuer-event precision sample identity changed")
    if (
        manifest.get("schema") != SAMPLE_MANIFEST_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("production_ready") is not False
        or authority.get("schema") != SAMPLE_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("production_ready") is not False
    ):
        raise DataReadinessError("issuer-event precision sample root does not verify")
    _verify_inventory(
        directory,
        {
            "_authority.json",
            "_manifest.json",
            "sample.parquet",
            "sample.parquet.manifest.json",
            "reviewer_one_template.csv",
            "reviewer_two_template.csv",
            "adjudication_template.csv",
        },
    )
    request = _request(manifest, authority)
    policy_path = _required_path(request, "policy_path")
    if file_sha256(policy_path) != _required_hash(request, "policy_sha256"):
        raise DataReadinessError("issuer-event precision sample policy changed")
    policy = load_issuer_event_precision_policy(policy_path)
    if request.get("source_authorization_sha256") != SOURCE_AUTHORIZATION_SHA256:
        raise DataReadinessError("issuer-event source authorization policy changed")
    source_directory = _required_path(request, "issuer_event_authority_directory")
    source_authority_path = _required_path(request, "issuer_event_authority_path")
    source_sha256 = _required_hash(request, "issuer_event_authority_sha256")
    if source_authority_path != (source_directory / "_authority.json").resolve():
        raise DataReadinessError("issuer-event precision source authority path differs")
    source = load_issuer_event_family_authority(
        source_directory,
        expected_authority_sha256=source_sha256,
    )
    _guard_memory(policy, "issuer-event authority replay")
    sample_path = directory / "sample.parquet"
    sample, child = load_canonical_artifact(sample_path, expected_type=SAMPLE_ARTIFACT_TYPE, allow_research=True)
    artifact_records = _artifact_records(manifest)
    _verify_canonical_record(
        sample_path,
        sample,
        child,
        artifact_records,
        "sample",
        request_sha256=str(manifest["request_sha256"]),
    )
    _sample_audit(sample).raise_for_failure()
    expected, expected_population, expected_variant_population = _build_deterministic_sample(
        source,
        policy=policy,
        policy_sha256=_required_hash(request, "policy_sha256"),
        authority_sha256=source_sha256,
    )
    _assert_frame_equal(sample, expected, "precision sample replay")
    templates = (
        ("reviewer_one_template", _review_template(sample, reviewer_slot=1)),
        ("reviewer_two_template", _review_template(sample, reviewer_slot=2)),
        ("adjudication_template", _adjudication_template(sample)),
    )
    for name, expected_template in templates:
        path = directory / f"{name}.csv"
        _verify_file_record(path, artifact_records, name, len(expected_template))
        observed = _read_csv(path, tuple(expected_template.columns))
        _assert_frame_equal(observed, expected_template, f"{name} replay")
    expected_inferential_counts = {
        family: int((sample["proposed_event_family"].eq(family) & sample["sample_role"].eq("inferential")).sum())
        for family in EVENT_FAMILIES
    }
    expected_diagnostic_counts = {
        family: int((sample["proposed_event_family"].eq(family) & sample["sample_role"].eq("paired_wrong_issuer_diagnostic")).sum())
        for family in EVENT_FAMILIES
    }
    if (
        manifest.get("event_families") != list(EVENT_FAMILIES)
        or manifest.get("population") != expected_population
        or manifest.get("rule_variant_population") != expected_variant_population
        or manifest.get("inferential_sample_counts") != expected_inferential_counts
        or manifest.get("diagnostic_sample_counts") != expected_diagnostic_counts
        or manifest.get("training_eligible") is not False
        or manifest.get("alerts_eligible") is not False
    ):
        raise DataReadinessError("issuer-event precision sample summary does not verify")
    return IssuerEventPrecisionSample(
        directory=directory.resolve(),
        sample=sample,
        manifest=manifest,
        authority=authority,
        source_authority=source if retain_source_authority else None,
    )


def finalize_issuer_event_precision_audit(
    *,
    sample_directory: Path,
    reviewer_one_path: Path,
    reviewer_two_path: Path,
    adjudication_path: Path,
    output_directory: Path,
) -> IssuerEventPrecisionAudit:
    """Finalize two blind reviews and publish per-family precision gates."""

    if output_directory.exists():
        raise DataReadinessError(f"issuer-event precision audit is immutable: {output_directory}")
    _preflight_review_ledgers(
        reviewer_one_path=reviewer_one_path,
        reviewer_two_path=reviewer_two_path,
        adjudication_path=adjudication_path,
    )
    sample_root = load_issuer_event_precision_sample(sample_directory)
    policy_path = _required_path(_manifest_request(sample_root.manifest), "policy_path")
    policy = load_issuer_event_precision_policy(policy_path)
    reviewer_one = _load_review_ledger(reviewer_one_path, sample_root.sample, reviewer_slot=1)
    reviewer_two = _load_review_ledger(reviewer_two_path, sample_root.sample, reviewer_slot=2)
    adjudication = _load_adjudication_ledger(adjudication_path, sample_root.sample)
    reviews = _resolve_reviews(sample_root.sample, reviewer_one, reviewer_two, adjudication)
    population = _population_from_manifest(sample_root.manifest)
    rule_variant_metrics = _build_rule_variant_metrics(
        sample_root.sample,
        reviews,
        population=_rule_variant_population_from_manifest(sample_root.manifest),
        policy=policy,
    )
    metrics = _build_family_metrics(
        sample_root.sample,
        reviews,
        population=population,
        rule_variant_metrics=rule_variant_metrics,
        policy=policy,
    )
    _guard_memory(policy, "precision audit finalization")
    sample_authority_path = sample_directory / "_authority.json"
    request = {
        "schema": FINAL_AUTHORITY_SCHEMA,
        "sample_directory": str(sample_directory.resolve()),
        "sample_authority_path": str(sample_authority_path.resolve()),
        "sample_authority_sha256": file_sha256(sample_authority_path),
        "ingested_reviewer_one_path": str(reviewer_one_path.resolve()),
        "ingested_reviewer_one_sha256": file_sha256(reviewer_one_path),
        "ingested_reviewer_two_path": str(reviewer_two_path.resolve()),
        "ingested_reviewer_two_sha256": file_sha256(reviewer_two_path),
        "ingested_adjudication_path": str(adjudication_path.resolve()),
        "ingested_adjudication_sha256": file_sha256(adjudication_path),
        "policy_sha256": file_sha256(policy_path),
        "production_ready": False,
    }
    request_sha256 = _json_sha256(request)
    staging = _new_staging(output_directory)
    try:
        local_reviewer_one = staging / "reviewer_one.csv"
        local_reviewer_two = staging / "reviewer_two.csv"
        local_adjudication = staging / "adjudication.csv"
        shutil.copyfile(reviewer_one_path, local_reviewer_one)
        shutil.copyfile(reviewer_two_path, local_reviewer_two)
        shutil.copyfile(adjudication_path, local_adjudication)
        if (
            file_sha256(local_reviewer_one) != request["ingested_reviewer_one_sha256"]
            or file_sha256(local_reviewer_two) != request["ingested_reviewer_two_sha256"]
            or file_sha256(local_adjudication) != request["ingested_adjudication_sha256"]
        ):
            raise DataReadinessError("precision ledger copy does not verify")
        review_path = staging / "reviews.parquet"
        metric_path = staging / "family_metrics.parquet"
        rule_variant_metric_path = staging / "rule_variant_metrics.parquet"
        review_manifest = write_canonical_artifact(
            reviews,
            review_path,
            artifact_type=REVIEWS_ARTIFACT_TYPE,
            audit=_review_audit(reviews, sample_root.sample),
            inputs={"request_sha256": request_sha256},
            production_ready=False,
        )
        rule_variant_metric_manifest = write_canonical_artifact(
            rule_variant_metrics,
            rule_variant_metric_path,
            artifact_type=RULE_VARIANT_METRICS_ARTIFACT_TYPE,
            audit=_rule_variant_metric_audit(rule_variant_metrics),
            inputs={
                "request_sha256": request_sha256,
                "reviews_sha256": str(review_manifest["artifact_sha256"]),
            },
            production_ready=False,
        )
        metric_manifest = write_canonical_artifact(
            metrics,
            metric_path,
            artifact_type=METRICS_ARTIFACT_TYPE,
            audit=_metric_audit(metrics),
            inputs={
                "request_sha256": request_sha256,
                "reviews_sha256": str(review_manifest["artifact_sha256"]),
                "rule_variant_metrics_sha256": str(rule_variant_metric_manifest["artifact_sha256"]),
            },
            production_ready=False,
        )
        for path in (review_path, metric_path, rule_variant_metric_path):
            _remove_lock(path)
            _rewrite_artifact_path(path, output_directory / path.name)
        admitted = metrics.loc[metrics["status"].eq("admitted"), "event_family"].tolist()
        blocked = metrics.loc[metrics["status"].eq("blocked"), "event_family"].tolist()
        audit_status = _overall_audit_status(admitted, blocked)
        manifest = {
            "schema": FINAL_MANIFEST_SCHEMA,
            "state": "complete",
            "audit_status": audit_status,
            "request": request,
            "request_sha256": request_sha256,
            "event_families": list(EVENT_FAMILIES),
            "admitted_families": admitted,
            "blocked_families": blocked,
            "artifacts": {
                "reviews": _artifact_record(review_path, review_manifest),
                "family_metrics": _artifact_record(metric_path, metric_manifest),
                "rule_variant_metrics": _artifact_record(rule_variant_metric_path, rule_variant_metric_manifest),
                "reviewer_one": _file_record(local_reviewer_one, len(reviewer_one)),
                "reviewer_two": _file_record(local_reviewer_two, len(reviewer_two)),
                "adjudication": _file_record(local_adjudication, len(adjudication)),
            },
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "memory": memory_audit(
                hard_budget_gib=policy.maximum_process_memory_gib,
                headroom_gib=policy.memory_guard_headroom_gib,
            ).to_record(),
            "production_ready": False,
            "training_eligible": False,
            "alerts_eligible": False,
        }
        _atomic_json(staging / "_manifest.json", manifest)
        root = {
            "schema": FINAL_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(staging / "_manifest.json"),
            "request_sha256": request_sha256,
            "audit_status": audit_status,
            "production_ready": False,
        }
        _atomic_json(staging / "_authority.json", root)
        load_issuer_event_precision_audit(staging)
        os.replace(staging, output_directory)
        return load_issuer_event_precision_audit(output_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_issuer_event_precision_audit(
    directory: Path,
    *,
    expected_authority_sha256: str | None = None,
    retain_source_authority: bool = False,
) -> IssuerEventPrecisionAudit:
    """Strictly load and semantically replay a finalized precision audit."""

    manifest_path = directory / "_manifest.json"
    authority_path = directory / "_authority.json"
    manifest = _json_object(manifest_path)
    authority = _json_object(authority_path)
    if expected_authority_sha256 is not None and (file_sha256(authority_path) != expected_authority_sha256):
        raise DataReadinessError("issuer-event precision audit identity changed")
    if (
        manifest.get("schema") != FINAL_MANIFEST_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("production_ready") is not False
        or authority.get("schema") != FINAL_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("audit_status") != manifest.get("audit_status")
        or authority.get("production_ready") is not False
    ):
        raise DataReadinessError("issuer-event precision audit root does not verify")
    _verify_inventory(
        directory,
        {
            "_authority.json",
            "_manifest.json",
            "reviews.parquet",
            "reviews.parquet.manifest.json",
            "family_metrics.parquet",
            "family_metrics.parquet.manifest.json",
            "rule_variant_metrics.parquet",
            "rule_variant_metrics.parquet.manifest.json",
            "reviewer_one.csv",
            "reviewer_two.csv",
            "adjudication.csv",
        },
    )
    request = _request(manifest, authority)
    sample_directory = _required_path(request, "sample_directory")
    sample_authority_path = _required_path(request, "sample_authority_path")
    sample_hash = _required_hash(request, "sample_authority_sha256")
    if sample_authority_path != (sample_directory / "_authority.json").resolve():
        raise DataReadinessError("precision audit sample authority path differs")
    sample_root = load_issuer_event_precision_sample(
        sample_directory,
        expected_authority_sha256=sample_hash,
        retain_source_authority=retain_source_authority,
    )
    policy_path = _required_path(_manifest_request(sample_root.manifest), "policy_path")
    if file_sha256(policy_path) != _required_hash(request, "policy_sha256"):
        raise DataReadinessError("precision audit policy lineage changed")
    policy = load_issuer_event_precision_policy(policy_path)
    artifacts = _artifact_records(manifest)
    reviewer_one_path = directory / "reviewer_one.csv"
    reviewer_two_path = directory / "reviewer_two.csv"
    adjudication_path = directory / "adjudication.csv"
    reviewer_one = _load_review_ledger(reviewer_one_path, sample_root.sample, reviewer_slot=1)
    reviewer_two = _load_review_ledger(reviewer_two_path, sample_root.sample, reviewer_slot=2)
    adjudication = _load_adjudication_ledger(adjudication_path, sample_root.sample)
    for key, path, rows in (
        ("reviewer_one", reviewer_one_path, len(reviewer_one)),
        ("reviewer_two", reviewer_two_path, len(reviewer_two)),
        ("adjudication", adjudication_path, len(adjudication)),
    ):
        _verify_file_record(path, artifacts, key, rows)
        if file_sha256(path) != _required_hash(request, f"ingested_{key}_sha256"):
            raise DataReadinessError(f"precision {key} ingestion lineage differs")
    expected_reviews = _resolve_reviews(sample_root.sample, reviewer_one, reviewer_two, adjudication)
    expected_rule_variant_metrics = _build_rule_variant_metrics(
        sample_root.sample,
        expected_reviews,
        population=_rule_variant_population_from_manifest(sample_root.manifest),
        policy=policy,
    )
    expected_metrics = _build_family_metrics(
        sample_root.sample,
        expected_reviews,
        population=_population_from_manifest(sample_root.manifest),
        rule_variant_metrics=expected_rule_variant_metrics,
        policy=policy,
    )
    reviews_path = directory / "reviews.parquet"
    metrics_path = directory / "family_metrics.parquet"
    rule_variant_metrics_path = directory / "rule_variant_metrics.parquet"
    reviews, review_child = load_canonical_artifact(reviews_path, expected_type=REVIEWS_ARTIFACT_TYPE, allow_research=True)
    metrics, metric_child = load_canonical_artifact(metrics_path, expected_type=METRICS_ARTIFACT_TYPE, allow_research=True)
    rule_variant_metrics, rule_variant_metric_child = load_canonical_artifact(
        rule_variant_metrics_path,
        expected_type=RULE_VARIANT_METRICS_ARTIFACT_TYPE,
        allow_research=True,
    )
    _verify_canonical_record(
        reviews_path,
        reviews,
        review_child,
        artifacts,
        "reviews",
        request_sha256=str(manifest["request_sha256"]),
    )
    _verify_canonical_record(
        rule_variant_metrics_path,
        rule_variant_metrics,
        rule_variant_metric_child,
        artifacts,
        "rule_variant_metrics",
        request_sha256=str(manifest["request_sha256"]),
    )
    _verify_canonical_record(
        metrics_path,
        metrics,
        metric_child,
        artifacts,
        "family_metrics",
        request_sha256=str(manifest["request_sha256"]),
    )
    metric_inputs = metric_child.get("inputs")
    rule_variant_inputs = rule_variant_metric_child.get("inputs")
    if (
        not isinstance(metric_inputs, dict)
        or not isinstance(rule_variant_inputs, dict)
        or metric_inputs.get("reviews_sha256") != review_child.get("artifact_sha256")
        or rule_variant_inputs.get("reviews_sha256") != review_child.get("artifact_sha256")
        or metric_inputs.get("rule_variant_metrics_sha256") != rule_variant_metric_child.get("artifact_sha256")
    ):
        raise DataReadinessError("precision audit review-to-metric lineage fails")
    _assert_frame_equal(reviews, expected_reviews, "precision review replay")
    _assert_frame_equal(
        rule_variant_metrics,
        expected_rule_variant_metrics,
        "precision rule-variant metric replay",
    )
    _assert_frame_equal(metrics, expected_metrics, "precision metric replay")
    _guard_memory(policy, "precision audit replay")
    admitted = metrics.loc[metrics["status"].eq("admitted"), "event_family"].tolist()
    blocked = metrics.loc[metrics["status"].eq("blocked"), "event_family"].tolist()
    expected_status = _overall_audit_status(admitted, blocked)
    if (
        manifest.get("event_families") != list(EVENT_FAMILIES)
        or manifest.get("admitted_families") != admitted
        or manifest.get("blocked_families") != blocked
        or manifest.get("audit_status") != expected_status
        or manifest.get("training_eligible") is not False
        or manifest.get("alerts_eligible") is not False
    ):
        raise DataReadinessError("issuer-event precision audit status does not verify")
    return IssuerEventPrecisionAudit(
        directory=directory.resolve(),
        reviews=reviews,
        family_metrics=metrics,
        rule_variant_metrics=rule_variant_metrics,
        manifest=manifest,
        authority=authority,
        source_authority=sample_root.source_authority,
    )


def _preflight_review_ledgers(
    *,
    reviewer_one_path: Path,
    reviewer_two_path: Path,
    adjudication_path: Path,
) -> None:
    """Reject malformed review logic before the expensive authority replay."""

    reviewer_one = _read_csv(reviewer_one_path, REVIEW_TEMPLATE_COLUMNS)
    reviewer_two = _read_csv(reviewer_two_path, REVIEW_TEMPLATE_COLUMNS)
    adjudication = _read_csv(adjudication_path, ADJUDICATION_TEMPLATE_COLUMNS)
    identity_columns = ("sample_id", "family_event_id", "event_family")
    _assert_frame_equal(
        reviewer_one.loc[:, identity_columns],
        reviewer_two.loc[:, identity_columns],
        "reviewer preflight identity",
    )
    _assert_frame_equal(
        reviewer_one.loc[:, identity_columns],
        adjudication.loc[:, identity_columns],
        "adjudication preflight identity",
    )
    sample = pd.DataFrame(
        {
            "sample_id": reviewer_one["sample_id"].astype(str),
            "family_event_id": reviewer_one["family_event_id"].astype(str),
            "proposed_event_family": reviewer_one["event_family"].astype(str),
            "sample_role": "preflight",
            "inference_cluster_id": reviewer_one["sample_id"].astype(str),
        }
    )
    normalized_one = _load_review_ledger(
        reviewer_one_path,
        sample,
        reviewer_slot=1,
    )
    normalized_two = _load_review_ledger(
        reviewer_two_path,
        sample,
        reviewer_slot=2,
    )
    normalized_adjudication = _load_adjudication_ledger(adjudication_path, sample)
    _resolve_reviews(
        sample,
        normalized_one,
        normalized_two,
        normalized_adjudication,
    )


def wilson_lower_bound(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> float:
    """Return the one-sided Wilson score lower confidence bound."""

    if total <= 0 or successes < 0 or successes > total:
        raise ValueError("Wilson counts must satisfy 0 <= successes <= total and total > 0")
    if not 0.5 < confidence_level < 1.0:
        raise ValueError("one-sided Wilson confidence must be between 0.5 and 1")
    z = NormalDist().inv_cdf(confidence_level)
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    center = proportion + z_squared / (2.0 * total)
    margin = z * math.sqrt(proportion * (1.0 - proportion) / total + z_squared / (4.0 * total * total))
    return (center - margin) / denominator


def _build_deterministic_sample(
    authority: IssuerEventFamilyAuthority,
    *,
    policy: IssuerEventPrecisionPolicy,
    policy_sha256: str,
    authority_sha256: str,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
]:
    events = authority.events
    observed_families = set(events.get("event_family", pd.Series(dtype=str)).astype(str))
    unknown = sorted(observed_families.difference(EVENT_FAMILIES))
    if unknown:
        raise DataReadinessError("issuer-event authority contains unknown families: " + ", ".join(unknown))
    with tempfile.TemporaryDirectory(prefix="market-predictor-precision-") as temporary:
        database_path = Path(temporary) / "candidate_index.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            _create_candidate_index(connection)
            _index_eligible_authority_events(connection, events, policy=policy)
            attribution, collection_manifest_path = _strict_attribution_context(authority)
            _stream_candidate_population(
                connection,
                collection_manifest_path=collection_manifest_path,
                policy=policy,
                policy_sha256=policy_sha256,
            )
            population = _population_from_candidate_index(connection)
            variant_population = _variant_population_from_candidate_index(connection)
            selected = _select_uniform_cluster_rows(
                connection,
                policy=policy,
                policy_sha256=policy_sha256,
                authority_sha256=authority_sha256,
            )
            sample = _selected_causal_evidence(
                attribution=attribution,
                collection_manifest_path=collection_manifest_path,
                selected=selected,
                policy=policy,
            )
        finally:
            connection.close()
    if sample.empty:
        sample = pd.DataFrame(columns=SAMPLE_COLUMNS)
    else:
        sample["schema_version"] = SAMPLE_MANIFEST_SCHEMA
        sample["_role_order"] = sample["sample_role"].map({"inferential": 0, "paired_wrong_issuer_diagnostic": 1})
        sample = (
            sample.sort_values(
                [
                    "proposed_event_family",
                    "cluster_selection_sha256",
                    "row_selection_sha256",
                    "_role_order",
                    "family_event_id",
                ],
                kind="stable",
            )
            .loc[:, SAMPLE_COLUMNS]
            .reset_index(drop=True)
        )
    _sample_audit(sample).raise_for_failure()
    _guard_memory(policy, "precision sample construction")
    return sample, population, variant_population


def _source_authorized_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_family",
        "source_family",
        "research_eligible",
        "family_event_id",
        "source_event_id",
        "relation_id",
        "security_id",
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise DataReadinessError("issuer-event precision source is missing columns: " + ", ".join(missing))
    research_eligible = events["research_eligible"].astype(bool)
    source_authorized = pd.Series(False, index=events.index)
    for family in EVENT_FAMILIES:
        source_authorized |= events["event_family"].eq(family) & events["source_family"].astype(str).isin(
            ALLOWED_SOURCE_FAMILIES_BY_FAMILY[family]
        )
    eligible = events.loc[research_eligible & source_authorized]
    if bool(eligible["family_event_id"].astype(str).duplicated().any()):
        raise DataReadinessError("eligible issuer-event identities are duplicated")
    return eligible


def _create_candidate_index(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = FILE;
        CREATE TABLE eligible_events (
            family_event_id TEXT PRIMARY KEY,
            source_event_id TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX eligible_source_event_idx
            ON eligible_events(source_event_id);
        CREATE TABLE candidates (
            family_event_id TEXT PRIMARY KEY,
            source_event_id TEXT NOT NULL,
            relation_id TEXT NOT NULL,
            security_id TEXT NOT NULL,
            event_family TEXT NOT NULL,
            source_family TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            inference_cluster_id TEXT NOT NULL,
            cluster_selection_sha256 TEXT NOT NULL,
            row_selection_sha256 TEXT NOT NULL,
            normalized_title_sha256 TEXT NOT NULL,
            calendar_quarter TEXT NOT NULL,
            rule_variant TEXT NOT NULL,
            stratum_id TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX candidate_family_cluster_idx
            ON candidates(event_family, inference_cluster_id);
        CREATE INDEX candidate_cluster_row_idx
            ON candidates(inference_cluster_id, row_selection_sha256);
        CREATE INDEX candidate_source_event_idx
            ON candidates(source_event_id);
        """
    )


def _index_eligible_authority_events(
    connection: sqlite3.Connection,
    events: pd.DataFrame,
    *,
    policy: IssuerEventPrecisionPolicy,
) -> None:
    eligible = _source_authorized_events(events)
    columns = tuple(str(column) for column in eligible.columns)
    cursor = connection.cursor()
    try:
        for index, values in enumerate(eligible.itertuples(index=False, name=None), start=1):
            row = dict(zip(columns, values, strict=True))
            try:
                cursor.execute(
                    "INSERT INTO eligible_events VALUES (?, ?, ?)",
                    (
                        str(row["family_event_id"]),
                        str(row["source_event_id"]),
                        json.dumps(
                            row,
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        ),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DataReadinessError("eligible issuer-event identities are duplicated") from exc
            if index % 25_000 == 0:
                _guard_memory(policy, f"authority candidate index row {index}")
    finally:
        cursor.close()
    connection.commit()


def _strict_attribution_context(
    authority: IssuerEventFamilyAuthority,
) -> tuple[EventAttributionHistory, Path]:
    authority_request = authority.manifest.get("request")
    if not isinstance(authority_request, dict):
        raise DataReadinessError("issuer-event authority request is malformed")
    attribution_manifest_path = _required_path(authority_request, "attribution_manifest_path")
    if file_sha256(attribution_manifest_path) != _required_hash(authority_request, "attribution_manifest_sha256"):
        raise DataReadinessError("issuer-event attribution manifest changed")
    attribution = load_event_attribution_history(attribution_manifest_path.parent)
    collection_manifest_path = _required_path(attribution.request, "collection_manifest_path")
    if collection_manifest_path != _required_path(authority_request, "collection_manifest_path"):
        raise DataReadinessError("issuer-event collection lineage differs")
    return attribution, collection_manifest_path


def _stream_candidate_population(
    connection: sqlite3.Connection,
    *,
    collection_manifest_path: Path,
    policy: IssuerEventPrecisionPolicy,
    policy_sha256: str,
) -> None:
    collection = _json_object(collection_manifest_path)
    records = collection.get("artifacts")
    if not isinstance(records, list):
        raise DataReadinessError("precision source event inventory is malformed")
    desired_columns = {
        "event_id",
        "source_family",
        "title",
        "published_at_utc",
    }
    for index, raw in enumerate(records, start=1):
        if not isinstance(raw, dict):
            raise DataReadinessError("precision source event record is malformed")
        path = _required_path(raw, "path")
        declared = canonical_artifact_columns(path)
        columns = tuple(column for column in declared if column in desired_columns)
        if set(columns) != desired_columns:
            raise DataReadinessError("precision source event artifact lacks cluster fields")
        events, manifest = load_canonical_artifact(
            path,
            expected_type="events",
            allow_research=True,
            columns=columns,
        )
        if manifest.get("artifact_sha256") != _required_hash(raw, "sha256") or len(events) != _nonnegative_int(raw, "rows"):
            raise DataReadinessError("precision source event artifact differs")
        chunk_id = _required_text(raw, "chunk_id")
        columns_in_frame = tuple(str(column) for column in events.columns)
        insert_cursor = connection.cursor()
        for values in events.itertuples(index=False, name=None):
            source = dict(zip(columns_in_frame, values, strict=True))
            source_event_id = str(source["event_id"])
            matches = connection.execute(
                "SELECT payload_json FROM eligible_events WHERE source_event_id = ?",
                (source_event_id,),
            )
            for (payload_json,) in matches:
                payload = _json_text_object(str(payload_json), "eligible event payload")
                try:
                    insert_cursor.execute(
                        "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        _candidate_index_row(
                            payload,
                            source,
                            chunk_id=chunk_id,
                            policy_sha256=policy_sha256,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise DataReadinessError("precision candidate identities are duplicated") from exc
        insert_cursor.close()
        connection.commit()
        del events
        if index % 32 == 0:
            gc.collect()
            release_process_memory()
        _guard_memory(policy, f"candidate source chunk {index}")
    eligible_count = int(connection.execute("SELECT COUNT(*) FROM eligible_events").fetchone()[0])
    candidate_count = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
    if candidate_count != eligible_count:
        raise DataReadinessError("precision candidate population is incomplete")


def _candidate_index_row(
    family: Mapping[str, object],
    source: Mapping[str, object],
    *,
    chunk_id: str,
    policy_sha256: str,
) -> tuple[str, ...]:
    if str(family["source_event_id"]) != str(source["event_id"]) or str(family["source_family"]) != str(source["source_family"]):
        raise DataReadinessError("precision source event identity differs")
    title = _clean_text(source.get("title"))
    if not title:
        raise DataReadinessError("precision source event title is empty")
    published = _timestamp(source["published_at_utc"], "source publication time")
    if published != _timestamp(family["published_at_utc"], "family publication time"):
        raise DataReadinessError("precision source publication timing differs")
    family_name = str(family["event_family"])
    normalized_title_sha256 = _sha256(_normalize_title(title))
    publication_day = published.strftime("%Y-%m-%d")
    cluster_id = _sha256(f"{family_name}|{normalized_title_sha256}|{publication_day}")
    cluster_hash = _sha256(f"{policy_sha256}|cluster|{cluster_id}")
    family_event_id = str(family["family_event_id"])
    row_hash = _sha256(f"{policy_sha256}|row|{family_event_id}")
    feature_time = _timestamp(family["feature_available_at_utc"], "feature time")
    quarter = f"{feature_time.year}Q{feature_time.quarter}"
    variant = issuer_event_classification.issuer_event_rule_variant(family)
    stratum = "|".join(
        (
            str(family["source_family"]),
            str(family["classification_rule_id"]),
            variant,
            quarter,
        )
    )
    return (
        family_event_id,
        str(family["source_event_id"]),
        str(family["relation_id"]),
        str(family["security_id"]),
        family_name,
        str(family["source_family"]),
        chunk_id,
        cluster_id,
        cluster_hash,
        row_hash,
        normalized_title_sha256,
        quarter,
        variant,
        stratum,
        json.dumps(family, sort_keys=True, separators=(",", ":"), default=str),
    )


def _population_from_candidate_index(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for family in EVENT_FAMILIES:
        row = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT inference_cluster_id), COUNT(DISTINCT security_id) FROM candidates WHERE event_family = ?",
            (family,),
        ).fetchone()
        output[family] = {
            "eligible_events": int(row[0]),
            "clusters": int(row[1]),
            "issuers": int(row[2]),
        }
    return output


def _variant_population_from_candidate_index(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {family: {} for family in EVENT_FAMILIES}
    rows = connection.execute(
        """
        WITH representatives AS (
            SELECT event_family, inference_cluster_id, rule_variant,
                   ROW_NUMBER() OVER (
                       PARTITION BY inference_cluster_id
                       ORDER BY row_selection_sha256, family_event_id
                   ) AS row_number
            FROM candidates
        )
        SELECT event_family, rule_variant, COUNT(*)
        FROM representatives
        WHERE row_number = 1
        GROUP BY event_family, rule_variant
        ORDER BY event_family, rule_variant
        """
    ).fetchall()
    for family, variant, count in rows:
        output[str(family)][str(variant)] = int(count)
    return output


def _select_uniform_cluster_rows(
    connection: sqlite3.Connection,
    *,
    policy: IssuerEventPrecisionPolicy,
    policy_sha256: str,
    authority_sha256: str,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for family in EVENT_FAMILIES:
        clusters = connection.execute(
            "SELECT inference_cluster_id, cluster_selection_sha256 "
            "FROM candidates WHERE event_family = ? "
            "GROUP BY inference_cluster_id, cluster_selection_sha256 "
            "ORDER BY cluster_selection_sha256, inference_cluster_id LIMIT ?",
            (family, policy.families[family].sample_clusters),
        ).fetchall()
        for cluster_id, cluster_hash in clusters:
            representative_row = connection.execute(
                "SELECT payload_json, chunk_id, row_selection_sha256, "
                "normalized_title_sha256, calendar_quarter, rule_variant, stratum_id "
                "FROM candidates WHERE inference_cluster_id = ? "
                "ORDER BY row_selection_sha256, family_event_id LIMIT 1",
                (cluster_id,),
            ).fetchone()
            if representative_row is None:
                raise DataReadinessError("selected precision cluster is empty")
            representative = _selected_candidate_record(
                representative_row,
                cluster_id=str(cluster_id),
                cluster_hash=str(cluster_hash),
                sample_role="inferential",
                paired_sample_id="",
                role_index=0,
                policy_sha256=policy_sha256,
                authority_sha256=authority_sha256,
            )
            _add_candidate_population_flags(connection, representative)
            selected.append(representative)
            representative_security = str(representative["security_id"])
            diagnostic_rows = connection.execute(
                "SELECT payload_json, chunk_id, row_selection_sha256, "
                "normalized_title_sha256, calendar_quarter, rule_variant, stratum_id "
                "FROM candidates WHERE inference_cluster_id = ? AND security_id != ? "
                "ORDER BY row_selection_sha256, family_event_id LIMIT ?",
                (
                    cluster_id,
                    representative_security,
                    policy.paired_wrong_issuer_diagnostics_per_cluster,
                ),
            )
            for diagnostic_count, row in enumerate(diagnostic_rows, start=1):
                diagnostic = _selected_candidate_record(
                    row,
                    cluster_id=str(cluster_id),
                    cluster_hash=str(cluster_hash),
                    sample_role="paired_wrong_issuer_diagnostic",
                    paired_sample_id=str(representative["sample_id"]),
                    role_index=diagnostic_count,
                    policy_sha256=policy_sha256,
                    authority_sha256=authority_sha256,
                )
                _add_candidate_population_flags(connection, diagnostic)
                selected.append(diagnostic)
    return selected


def _selected_candidate_record(
    row: Sequence[object],
    *,
    cluster_id: str,
    cluster_hash: str,
    sample_role: str,
    paired_sample_id: str,
    role_index: int,
    policy_sha256: str,
    authority_sha256: str,
) -> dict[str, object]:
    payload = _json_text_object(str(row[0]), "candidate payload")
    row_hash = str(row[2])
    sample_hash = _sha256(f"{policy_sha256}|{authority_sha256}|sample|{sample_role}|{cluster_id}|{row_hash}|{role_index}")
    payload.update(
        {
            "sample_id": f"sample:{sample_hash[:32]}",
            "sample_role": sample_role,
            "inference_cluster_id": cluster_id,
            "paired_inferential_sample_id": paired_sample_id,
            "chunk_id": str(row[1]),
            "cluster_selection_sha256": cluster_hash,
            "row_selection_sha256": row_hash,
            "normalized_title_sha256": str(row[3]),
            "calendar_quarter": str(row[4]),
            "rule_variant": str(row[5]),
            "stratum_id": str(row[6]),
        }
    )
    return payload


def _add_candidate_population_flags(
    connection: sqlite3.Connection,
    candidate: dict[str, object],
) -> None:
    cluster_targets = connection.execute(
        "SELECT COUNT(DISTINCT security_id) FROM candidates WHERE inference_cluster_id = ?",
        (str(candidate["inference_cluster_id"]),),
    ).fetchone()
    source_families = connection.execute(
        "SELECT COUNT(DISTINCT event_family) FROM candidates WHERE source_event_id = ?",
        (str(candidate["source_event_id"]),),
    ).fetchone()
    candidate["multi_target_title"] = int(cluster_targets[0]) > 1
    candidate["multi_label_event"] = int(source_families[0]) > 1


def _selected_causal_evidence(
    *,
    attribution: EventAttributionHistory,
    collection_manifest_path: Path,
    selected: Sequence[Mapping[str, object]],
    policy: IssuerEventPrecisionPolicy,
) -> pd.DataFrame:
    if not selected:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    source_by_id = _load_selected_source_events(
        collection_manifest_path,
        selected,
        policy=policy,
    )
    relation_by_id = _load_selected_relations(attribution, selected, policy=policy)
    identity_groups = _load_selected_identities(attribution, selected, policy=policy)
    rows: list[dict[str, object]] = []
    for family_row in selected:
        source_event_id = str(family_row["source_event_id"])
        relation_id = str(family_row["relation_id"])
        source = source_by_id.get(source_event_id)
        relation = relation_by_id.get(relation_id)
        if source is None or relation is None:
            raise DataReadinessError("issuer-event evidence lineage is incomplete")
        _verify_event_relation_lineage(family_row, source, relation)
        feature_time = _timestamp(family_row["feature_available_at_utc"], "feature time")
        identity = _causal_identity(identity_groups.get(str(family_row["security_id"])), feature_time)
        source_family = _clean_text(source.get("source_family"))
        source_name = _clean_text(source.get("source")) or source_family
        title = _clean_text(source.get("title"))
        published_day = _timestamp(family_row["published_at_utc"], "publication time").strftime("%Y-%m-%d")
        expected_cluster = _sha256(f"{family_row['event_family']}|{_sha256(_normalize_title(title))}|{published_day}")
        if expected_cluster != str(family_row["inference_cluster_id"]):
            raise DataReadinessError("selected precision cluster identity differs")
        rows.append(
            {
                "sample_id": str(family_row["sample_id"]),
                "sample_role": str(family_row["sample_role"]),
                "inference_cluster_id": str(family_row["inference_cluster_id"]),
                "paired_inferential_sample_id": str(family_row["paired_inferential_sample_id"]),
                "family_event_id": str(family_row["family_event_id"]),
                "source_event_id": source_event_id,
                "relation_id": relation_id,
                "source_security_id": str(family_row["source_security_id"]),
                "source_ticker": str(family_row["source_ticker"]),
                "security_id": str(family_row["security_id"]),
                "ticker": str(family_row["ticker"]),
                "source_family": str(family_row["source_family"]),
                "source": source_name,
                "proposed_event_family": str(family_row["event_family"]),
                "classification_rule_id": str(family_row["classification_rule_id"]),
                "classification_basis": str(family_row["classification_basis"]),
                "matched_text": str(family_row["matched_text"]),
                "title": title,
                "summary": _clean_text(source.get("summary")),
                "text": _clean_text(source.get("text")),
                "published_at_utc": _timestamp(family_row["published_at_utc"], "publication time"),
                "event_available_at_utc": _timestamp(family_row["event_available_at_utc"], "event availability"),
                "relation_available_at_utc": _timestamp(family_row["relation_available_at_utc"], "relation availability"),
                "feature_available_at_utc": feature_time,
                "availability_policy": str(family_row["availability_policy"]),
                "relation_channel": str(family_row["relation_channel"]),
                "issuer_company": identity["company"],
                "identity_effective_from_utc": identity["effective_from_utc"],
                "identity_effective_to_utc": identity["effective_to_utc"],
                "identity_available_at_utc": identity["available_at_utc"],
                "identity_status": identity["status"],
                "calendar_quarter": str(family_row["calendar_quarter"]),
                "rule_variant": str(family_row["rule_variant"]),
                "normalized_title_sha256": str(family_row["normalized_title_sha256"]),
                "multi_target_title": bool(family_row["multi_target_title"]),
                "multi_label_event": bool(family_row["multi_label_event"]),
                "stratum_id": str(family_row["stratum_id"]),
                "cluster_selection_sha256": str(family_row["cluster_selection_sha256"]),
                "row_selection_sha256": str(family_row["row_selection_sha256"]),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if bool(frame["title"].str.strip().eq("").any()) or bool(frame["source"].str.strip().eq("").any()):
        raise DataReadinessError("precision evidence requires causal title and source")
    if any(term in column.lower() for column in frame.columns for term in _FORBIDDEN_EVIDENCE_TERMS):
        raise DataReadinessError("precision evidence contains future outcome fields")
    _guard_memory(policy, "causal precision evidence")
    return frame


def _load_selected_source_events(
    collection_manifest_path: Path,
    selected: Sequence[Mapping[str, object]],
    *,
    policy: IssuerEventPrecisionPolicy,
) -> dict[str, dict[str, object]]:
    collection = _json_object(collection_manifest_path)
    records = collection.get("artifacts")
    if not isinstance(records, list):
        raise DataReadinessError("precision source event inventory is malformed")
    by_chunk: dict[str, set[str]] = {}
    for row in selected:
        by_chunk.setdefault(str(row["chunk_id"]), set()).add(str(row["source_event_id"]))
    output: dict[str, dict[str, object]] = {}
    desired_columns = (
        "event_id",
        "security_id",
        "ticker",
        "source_family",
        "source",
        "title",
        "summary",
        "text",
        "published_at_utc",
        "feature_available_at_utc",
    )
    records_by_chunk = {_required_text(raw, "chunk_id"): raw for raw in records if isinstance(raw, dict)}
    if set(by_chunk).difference(records_by_chunk):
        raise DataReadinessError("selected precision source chunks are missing")
    for index, chunk_id in enumerate(sorted(by_chunk), start=1):
        raw = records_by_chunk[chunk_id]
        if not isinstance(raw, dict):
            raise DataReadinessError("precision source event record is malformed")
        path = _required_path(raw, "path")
        events = _verified_filtered_artifact(
            path,
            expected_type="events",
            expected_sha256=_required_hash(raw, "sha256"),
            columns=desired_columns,
            filter_column="event_id",
            values=by_chunk[chunk_id],
        )
        for row in events.to_dict(orient="records"):
            event_id = str(row["event_id"])
            if event_id in output:
                raise DataReadinessError("precision source event identities are duplicated")
            output[event_id] = {str(key): value for key, value in row.items()}
        del events
        if index % 32 == 0:
            gc.collect()
            release_process_memory()
        _guard_memory(policy, f"source evidence chunk {index}")
    required_event_ids = {str(row["source_event_id"]) for row in selected}
    if set(output) != required_event_ids:
        raise DataReadinessError("precision source events are incomplete")
    return output


def _load_selected_relations(
    attribution: EventAttributionHistory,
    selected: Sequence[Mapping[str, object]],
    *,
    policy: IssuerEventPrecisionPolicy,
) -> dict[str, dict[str, object]]:
    by_chunk: dict[str, set[str]] = {}
    for row in selected:
        by_chunk.setdefault(str(row["chunk_id"]), set()).add(str(row["relation_id"]))
    records_by_chunk = {_required_text(record, "chunk_id"): record for record in attribution.artifact_records}
    if set(by_chunk).difference(records_by_chunk):
        raise DataReadinessError("selected precision relation chunks are missing")
    output: dict[str, dict[str, object]] = {}
    relation_columns = (
        "relation_id",
        "event_id",
        "source_security_id",
        "source_ticker",
        "target_security_id",
        "target_ticker",
        "relation_channel",
        "feature_available_at_utc",
    )
    for index, chunk_id in enumerate(sorted(by_chunk), start=1):
        record = records_by_chunk[chunk_id]
        path = _required_path(record, "path")
        relations = _verified_filtered_artifact(
            path,
            expected_type="event_security_relations",
            expected_sha256=_required_hash(record, "sha256"),
            columns=relation_columns,
            filter_column="relation_id",
            values=by_chunk[chunk_id],
        )
        for row in relations.to_dict(orient="records"):
            relation_id = str(row["relation_id"])
            if relation_id in output:
                raise DataReadinessError("precision relation identities are duplicated")
            output[relation_id] = {str(key): value for key, value in row.items()}
        del relations
        if index % 32 == 0:
            gc.collect()
            release_process_memory()
        _guard_memory(policy, f"relation evidence chunk {index}")
    required_relation_ids = {str(row["relation_id"]) for row in selected}
    if set(output) != required_relation_ids:
        raise DataReadinessError("precision event relations are incomplete")
    return output


def _load_selected_identities(
    attribution: EventAttributionHistory,
    selected: Sequence[Mapping[str, object]],
    *,
    policy: IssuerEventPrecisionPolicy,
) -> dict[str, list[dict[str, object]]]:
    identity_path = _required_path(attribution.request, "security_identities_path")
    security_ids = {str(row["security_id"]) for row in selected}
    identities = _verified_filtered_artifact(
        identity_path,
        expected_type="security_business_label_coverage",
        expected_sha256=_required_hash(attribution.request, "security_identities_sha256"),
        columns=(
            "security_id",
            "ticker",
            "company",
            "effective_from_utc",
            "effective_to_utc",
            "available_at_utc",
        ),
        filter_column="security_id",
        values=security_ids,
    )
    output = _identity_records_by_security(identities)
    del identities
    gc.collect()
    release_process_memory()
    _guard_memory(policy, "selected identity evidence")
    return output


def _verified_filtered_artifact(
    path: Path,
    *,
    expected_type: str,
    expected_sha256: str,
    columns: Sequence[str],
    filter_column: str,
    values: set[str],
) -> pd.DataFrame:
    manifest = _json_object(manifest_path_for(path))
    declared = canonical_artifact_columns(path)
    if (
        manifest.get("artifact_type") != expected_type
        or manifest.get("artifact_sha256") != expected_sha256
        or file_sha256(path) != expected_sha256
        or not set(columns).issubset(declared)
        or filter_column not in declared
    ):
        raise DataReadinessError(f"selected canonical artifact does not verify: {path}")
    if not values:
        return pd.DataFrame(columns=columns)
    frame = pd.read_parquet(
        path,
        columns=list(columns),
        filters=[(filter_column, "in", sorted(values))],
    )
    return frame.loc[frame[filter_column].astype(str).isin(values)].reset_index(drop=True)


def _verify_event_relation_lineage(
    family: Mapping[str, object],
    source: Mapping[str, object],
    relation: Mapping[str, object],
) -> None:
    comparisons = (
        (family["source_event_id"], source["event_id"]),
        (family["source_event_id"], relation["event_id"]),
        (family["source_security_id"], relation["source_security_id"]),
        (family["source_ticker"], relation["source_ticker"]),
        (family["security_id"], relation["target_security_id"]),
        (family["ticker"], relation["target_ticker"]),
        (family["relation_channel"], relation["relation_channel"]),
        (family["source_family"], source["source_family"]),
    )
    if any(str(left) != str(right) for left, right in comparisons):
        raise DataReadinessError("precision source event or relation identity differs")
    timestamp_pairs = (
        (family["published_at_utc"], source["published_at_utc"]),
        (family["event_available_at_utc"], source["feature_available_at_utc"]),
        (family["relation_available_at_utc"], relation["feature_available_at_utc"]),
    )
    if any(_timestamp(left, "lineage") != _timestamp(right, "lineage") for left, right in timestamp_pairs):
        raise DataReadinessError("precision source event or relation timing differs")
    if str(family["relation_channel"]) != "direct_issuer":
        raise DataReadinessError("precision sample cannot admit indirect issuer relations")


def _causal_identity(
    identities: Sequence[Mapping[str, object]] | None,
    feature_time: pd.Timestamp,
) -> dict[str, object]:
    empty: dict[str, object] = {
        "company": "",
        "effective_from_utc": pd.NaT,
        "effective_to_utc": pd.NaT,
        "available_at_utc": pd.NaT,
        "status": "unresolved",
    }
    if not identities:
        return empty
    eligible = [
        row
        for row in identities
        if isinstance(row["effective_from_utc"], pd.Timestamp)
        and isinstance(row["available_at_utc"], pd.Timestamp)
        and row["effective_from_utc"] <= feature_time
        and row["available_at_utc"] <= feature_time
        and (not isinstance(row["effective_to_utc"], pd.Timestamp) or row["effective_to_utc"] > feature_time)
    ]
    if len(eligible) != 1:
        empty["status"] = "ambiguous" if len(eligible) > 1 else "unresolved"
        return empty
    row = eligible[0]
    company = _clean_text(row["company"])
    if not company:
        return empty
    return {
        "company": company,
        "effective_from_utc": row["effective_from_utc"],
        "effective_to_utc": row["effective_to_utc"],
        "available_at_utc": row["available_at_utc"],
        "status": "resolved",
    }


def _identity_records_by_security(
    identities: pd.DataFrame,
) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    for raw in identities.to_dict(orient="records"):
        row = {str(key): value for key, value in raw.items()}
        row["effective_from_utc"] = _optional_timestamp(row.get("effective_from_utc"))
        row["effective_to_utc"] = _optional_timestamp(row.get("effective_to_utc"))
        row["available_at_utc"] = _optional_timestamp(row.get("available_at_utc"))
        output.setdefault(str(row["security_id"]), []).append(row)
    return output


def _review_template(sample: pd.DataFrame, *, reviewer_slot: int) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_id": sample["sample_id"].astype(str),
            "family_event_id": sample["family_event_id"].astype(str),
            "event_family": sample["proposed_event_family"].astype(str),
            "reviewer_slot": str(reviewer_slot),
            "reviewer_id": "",
            "family_correct": "",
            "issuer_target_correct": "",
            "event_announced_or_completed": "",
            "correct_family": "",
            "action_subject_text": "",
            "false_positive_reason": "",
            "comments": "",
        }
    )
    return frame.loc[:, REVIEW_TEMPLATE_COLUMNS]


def _adjudication_template(sample: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "sample_id": sample["sample_id"].astype(str),
            "family_event_id": sample["family_event_id"].astype(str),
            "event_family": sample["proposed_event_family"].astype(str),
            "adjudicator_id": "",
            "family_correct": "",
            "issuer_target_correct": "",
            "event_announced_or_completed": "",
            "correct_family": "",
            "action_subject_text": "",
            "false_positive_reason": "",
            "comments": "",
        }
    )
    return frame.loc[:, ADJUDICATION_TEMPLATE_COLUMNS]


def _load_review_ledger(
    path: Path,
    sample: pd.DataFrame,
    *,
    reviewer_slot: int,
) -> pd.DataFrame:
    frame = _read_csv(path, REVIEW_TEMPLATE_COLUMNS)
    expected = _review_template(sample, reviewer_slot=reviewer_slot)
    _validate_ledger_identity(frame, expected, reviewer_slot=reviewer_slot)
    frame["reviewer_id"] = frame["reviewer_id"].map(_normalized_review_text)
    if bool(frame["reviewer_id"].eq("").any()):
        raise DataReadinessError("precision reviewer identity is incomplete")
    if frame["reviewer_id"].nunique(dropna=False) != 1:
        raise DataReadinessError(f"precision reviewer slot {reviewer_slot} must use exactly one identity")
    for column in _DECISION_FIELDS:
        normalized = frame[column].str.strip().str.lower()
        if not set(normalized).issubset(_YES_NO_UNCERTAIN):
            raise DataReadinessError(f"precision review has invalid {column}")
        frame[column] = normalized
    for column in (*_CORRECTION_FIELDS, "comments"):
        frame[column] = frame[column].str.strip()
    frame["correct_family"] = frame["correct_family"].str.lower()
    _validate_correction_fields(frame, context=f"reviewer {reviewer_slot}")
    return frame


def _load_adjudication_ledger(path: Path, sample: pd.DataFrame) -> pd.DataFrame:
    frame = _read_csv(path, ADJUDICATION_TEMPLATE_COLUMNS)
    expected = _adjudication_template(sample)
    identity_columns = ("sample_id", "family_event_id", "event_family")
    _assert_frame_equal(
        frame.loc[:, identity_columns],
        expected.loc[:, identity_columns],
        "adjudication identity",
    )
    if bool(frame["sample_id"].duplicated().any()):
        raise DataReadinessError("precision adjudication contains duplicate samples")
    for column in ("adjudicator_id", *_DECISION_FIELDS, *_CORRECTION_FIELDS, "comments"):
        frame[column] = frame[column].str.strip()
    frame["adjudicator_id"] = frame["adjudicator_id"].map(_normalized_review_text)
    adjudicators = frame.loc[frame["adjudicator_id"].ne(""), "adjudicator_id"]
    if adjudicators.nunique(dropna=False) > 1:
        raise DataReadinessError("precision adjudication slot must use at most one identity")
    frame["correct_family"] = frame["correct_family"].str.lower()
    return frame


def _validate_ledger_identity(
    frame: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    reviewer_slot: int,
) -> None:
    identity_columns = (
        "sample_id",
        "family_event_id",
        "event_family",
        "reviewer_slot",
    )
    if not frame["reviewer_slot"].eq(str(reviewer_slot)).all():
        raise DataReadinessError("precision reviewer slot differs from its template")
    _assert_frame_equal(
        frame.loc[:, identity_columns].astype(str),
        expected.loc[:, identity_columns].astype(str),
        f"reviewer {reviewer_slot} identity",
    )
    if bool(frame["sample_id"].duplicated().any()):
        raise DataReadinessError("precision review contains duplicate samples")


def _resolve_reviews(
    sample: pd.DataFrame,
    reviewer_one: pd.DataFrame,
    reviewer_two: pd.DataFrame,
    adjudication: pd.DataFrame,
) -> pd.DataFrame:
    one = reviewer_one.set_index("sample_id", drop=False)
    two = reviewer_two.set_index("sample_id", drop=False)
    adjudicated = adjudication.set_index("sample_id", drop=False)
    rows: list[dict[str, object]] = []
    for sample_row in sample.to_dict(orient="records"):
        sample_id = str(sample_row["sample_id"])
        first = one.loc[sample_id]
        second = two.loc[sample_id]
        reviewer_one_id = str(first["reviewer_id"]).strip()
        reviewer_two_id = str(second["reviewer_id"]).strip()
        if reviewer_one_id.casefold() == reviewer_two_id.casefold():
            raise DataReadinessError("precision audit cannot use the same reviewer twice")
        disagreement = any(
            str(first[column]) != str(second[column]) or str(first[column]) == "uncertain" or str(second[column]) == "uncertain"
            for column in _DECISION_FIELDS
        ) or any(_normalized_review_text(first[column]) != _normalized_review_text(second[column]) for column in _CORRECTION_FIELDS)
        adjudication_row = adjudicated.loc[sample_id]
        adjudicator_id = str(adjudication_row["adjudicator_id"]).strip()
        final_values: dict[str, bool]
        resolution_state: str
        if disagreement:
            adjudication_values = {column: str(adjudication_row[column]).strip().lower() for column in _DECISION_FIELDS}
            complete = bool(adjudicator_id) and set(adjudication_values.values()).issubset(_YES_NO)
            if complete:
                if adjudicator_id.casefold() in {
                    reviewer_one_id.casefold(),
                    reviewer_two_id.casefold(),
                }:
                    raise DataReadinessError("precision adjudicator must differ from both reviewers")
                final_values = {column: value == "yes" for column, value in adjudication_values.items()}
                _validate_correction_row(
                    adjudication_row,
                    decisions=adjudication_values,
                    context="adjudication",
                )
                resolution_state = "adjudicated"
            else:
                if (
                    adjudicator_id
                    or any(adjudication_values.values())
                    or any(str(adjudication_row[column]).strip() for column in _CORRECTION_FIELDS)
                ):
                    raise DataReadinessError("precision adjudication is partially completed")
                final_values = {column: False for column in _DECISION_FIELDS}
                resolution_state = "unresolved_failure"
        else:
            if adjudicator_id or any(str(adjudication_row[column]).strip() for column in (*_DECISION_FIELDS, *_CORRECTION_FIELDS)):
                raise DataReadinessError("precision adjudication cannot override agreeing reviewers")
            final_values = {column: str(first[column]) == "yes" for column in _DECISION_FIELDS}
            resolution_state = "reviewer_agreement"
        notes_source = adjudication_row if resolution_state == "adjudicated" else first
        rows.append(
            {
                "sample_id": sample_id,
                "sample_role": str(sample_row["sample_role"]),
                "inference_cluster_id": str(sample_row["inference_cluster_id"]),
                "family_event_id": str(sample_row["family_event_id"]),
                "event_family": str(sample_row["proposed_event_family"]),
                "reviewer_one_id": reviewer_one_id,
                "reviewer_two_id": reviewer_two_id,
                "adjudicator_id": adjudicator_id,
                "reviewer_one_family_correct": str(first["family_correct"]),
                "reviewer_one_issuer_target_correct": str(first["issuer_target_correct"]),
                "reviewer_one_event_announced_or_completed": str(first["event_announced_or_completed"]),
                "reviewer_two_family_correct": str(second["family_correct"]),
                "reviewer_two_issuer_target_correct": str(second["issuer_target_correct"]),
                "reviewer_two_event_announced_or_completed": str(second["event_announced_or_completed"]),
                "adjudication_required": disagreement,
                "resolution_state": resolution_state,
                **final_values,
                "joint_correct": all(final_values.values()),
                "wrong_issuer": (resolution_state != "unresolved_failure" and not final_values["issuer_target_correct"]),
                "correct_family": str(notes_source["correct_family"]).strip(),
                "action_subject_text": str(notes_source["action_subject_text"]).strip(),
                "false_positive_reason": str(notes_source["false_positive_reason"]).strip(),
                "comments": str(notes_source["comments"]).strip(),
                "schema_version": FINAL_MANIFEST_SCHEMA,
            }
        )
    output = pd.DataFrame.from_records(rows, columns=REVIEW_COLUMNS)
    _review_audit(output, sample).raise_for_failure()
    return output


def _validate_correction_fields(frame: pd.DataFrame, *, context: str) -> None:
    for row in frame.to_dict(orient="records"):
        _validate_correction_row(
            row,
            decisions={field: str(row[field]) for field in _DECISION_FIELDS},
            context=context,
        )


def _validate_correction_row(
    row: Mapping[str, object] | pd.Series,
    *,
    decisions: Mapping[str, str],
    context: str,
) -> None:
    correct_family = _clean_text(row["correct_family"])
    action_subject = _clean_text(row["action_subject_text"])
    false_positive_reason = _clean_text(row["false_positive_reason"])
    if decisions["family_correct"] == "no" and correct_family not in {
        *EVENT_FAMILIES,
        "none",
    }:
        raise DataReadinessError(f"{context} must supply a valid correct_family for an incorrect family")
    if decisions["family_correct"] == "yes" and correct_family:
        raise DataReadinessError(f"{context} cannot supply correct_family when the family is correct")
    if decisions["issuer_target_correct"] == "no" and not action_subject:
        raise DataReadinessError(f"{context} must identify the action subject for a wrong issuer")
    if decisions["issuer_target_correct"] == "yes" and action_subject:
        raise DataReadinessError(f"{context} cannot supply an action subject when the issuer is correct")
    if decisions["event_announced_or_completed"] == "no" and not false_positive_reason:
        raise DataReadinessError(f"{context} must explain a non-event false positive")
    if decisions["event_announced_or_completed"] == "yes" and false_positive_reason:
        raise DataReadinessError(f"{context} cannot supply a false-positive reason for a confirmed event")


def _normalized_review_text(value: object) -> str:
    return " ".join(_clean_text(value).casefold().split())


def _build_family_metrics(
    sample: pd.DataFrame,
    reviews: pd.DataFrame,
    *,
    population: Mapping[str, Mapping[str, int]],
    rule_variant_metrics: pd.DataFrame,
    policy: IssuerEventPrecisionPolicy,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for family in EVENT_FAMILIES:
        family_policy = policy.families[family]
        family_sample = sample.loc[sample["proposed_event_family"].eq(family)]
        family_reviews = reviews.loc[reviews["event_family"].eq(family)]
        inferential_sample = family_sample.loc[family_sample["sample_role"].eq("inferential")]
        inferential_reviews = family_reviews.loc[family_reviews["sample_role"].eq("inferential")]
        diagnostic_reviews = family_reviews.loc[family_reviews["sample_role"].eq("paired_wrong_issuer_diagnostic")]
        counts = population.get(family, {"eligible_events": 0, "clusters": 0, "issuers": 0})
        event_count = int(counts.get("eligible_events", 0))
        cluster_count = int(counts.get("clusters", 0))
        issuer_count = int(counts.get("issuers", 0))
        inferential_count = len(inferential_sample)
        diagnostic_count = len(family_sample) - inferential_count
        resolved_count = int(inferential_reviews["resolution_state"].ne("unresolved_failure").sum())
        family_successes = _boolean_sum(inferential_reviews, "family_correct")
        issuer_successes = _boolean_sum(inferential_reviews, "issuer_target_correct")
        event_successes = _boolean_sum(inferential_reviews, "event_announced_or_completed")
        joint_successes = _boolean_sum(inferential_reviews, "joint_correct")
        family_lcb = _lcb_or_nan(family_successes, inferential_count, policy.confidence_level)
        issuer_lcb = _lcb_or_nan(issuer_successes, inferential_count, policy.confidence_level)
        event_lcb = _lcb_or_nan(event_successes, inferential_count, policy.confidence_level)
        joint_lcb = _lcb_or_nan(joint_successes, inferential_count, policy.confidence_level)
        wrong_issuer_count = _boolean_sum(inferential_reviews, "wrong_issuer")
        diagnostic_wrong_issuer_count = _boolean_sum(diagnostic_reviews, "wrong_issuer")
        unresolved_count = int(family_reviews["resolution_state"].eq("unresolved_failure").sum())
        identity_unresolved_count = int(family_sample["identity_status"].ne("resolved").sum())
        agreement = _reviewer_agreement_by_field(
            inferential_reviews,
            minimum_decisions=policy.minimum_kappa_decisions,
        )
        family_variant_metrics = rule_variant_metrics.loc[rule_variant_metrics["event_family"].eq(family)]
        failed_rule_variants = int(family_variant_metrics["status"].eq("blocked").sum())
        blockers: list[str] = []
        if cluster_count == 0:
            blockers.append("missing_family_population")
        if cluster_count < family_policy.minimum_population_clusters:
            blockers.append("below_minimum_population_clusters")
        if issuer_count < family_policy.minimum_population_issuers:
            blockers.append("below_minimum_population_issuers")
        expected_cluster_count = min(family_policy.sample_clusters, cluster_count)
        if (
            inferential_count != expected_cluster_count
            or len(inferential_reviews) != inferential_count
            or bool(inferential_sample["inference_cluster_id"].astype(str).duplicated().any())
        ):
            blockers.append("incomplete_cluster_sample")
        if identity_unresolved_count:
            blockers.append("unresolved_causal_identity")
        if policy.no_wrong_issuer and wrong_issuer_count:
            blockers.append("wrong_issuer_found")
        if policy.no_wrong_issuer and diagnostic_wrong_issuer_count:
            blockers.append("paired_wrong_issuer_diagnostic_failed")
        if unresolved_count:
            blockers.append("unresolved_review")
        for field, (field_agreement, field_kappa, estimable) in agreement.items():
            if not math.isnan(field_agreement) and field_agreement < policy.minimum_reviewer_agreement:
                blockers.append(f"{field}_reviewer_agreement_below_threshold")
            if estimable and not math.isnan(field_kappa) and field_kappa < policy.minimum_reviewer_kappa:
                blockers.append(f"{field}_reviewer_kappa_below_threshold")
        if failed_rule_variants:
            blockers.append("rule_variant_gate_failed")
        for value, threshold, reason in (
            (family_lcb, family_policy.minimum_family_lcb, "family_lcb_below_threshold"),
            (issuer_lcb, family_policy.minimum_issuer_lcb, "issuer_lcb_below_threshold"),
            (event_lcb, family_policy.minimum_event_lcb, "event_lcb_below_threshold"),
            (joint_lcb, family_policy.minimum_joint_lcb, "joint_lcb_below_threshold"),
        ):
            if math.isnan(value) or value < threshold:
                blockers.append(reason)
        rows.append(
            {
                "event_family": family,
                "population_eligible_events": event_count,
                "population_clusters": cluster_count,
                "population_issuers": issuer_count,
                "inferential_cluster_count": inferential_count,
                "diagnostic_count": diagnostic_count,
                "resolved_inferential_count": resolved_count,
                "minimum_population_clusters": (family_policy.minimum_population_clusters),
                "minimum_population_issuers": (family_policy.minimum_population_issuers),
                "family_successes": family_successes,
                "family_lcb": family_lcb,
                "minimum_family_lcb": family_policy.minimum_family_lcb,
                "issuer_successes": issuer_successes,
                "issuer_lcb": issuer_lcb,
                "minimum_issuer_lcb": family_policy.minimum_issuer_lcb,
                "event_successes": event_successes,
                "event_lcb": event_lcb,
                "minimum_event_lcb": family_policy.minimum_event_lcb,
                "joint_successes": joint_successes,
                "joint_lcb": joint_lcb,
                "minimum_joint_lcb": family_policy.minimum_joint_lcb,
                "wrong_issuer_count": wrong_issuer_count,
                "diagnostic_wrong_issuer_count": diagnostic_wrong_issuer_count,
                "unresolved_count": unresolved_count,
                "identity_unresolved_count": identity_unresolved_count,
                "family_reviewer_agreement": agreement["family_correct"][0],
                "family_reviewer_kappa": agreement["family_correct"][1],
                "family_kappa_estimable": agreement["family_correct"][2],
                "issuer_reviewer_agreement": agreement["issuer_target_correct"][0],
                "issuer_reviewer_kappa": agreement["issuer_target_correct"][1],
                "issuer_kappa_estimable": agreement["issuer_target_correct"][2],
                "event_reviewer_agreement": agreement["event_announced_or_completed"][0],
                "event_reviewer_kappa": agreement["event_announced_or_completed"][1],
                "event_kappa_estimable": agreement["event_announced_or_completed"][2],
                "minimum_reviewer_agreement": policy.minimum_reviewer_agreement,
                "minimum_reviewer_kappa": policy.minimum_reviewer_kappa,
                "failed_rule_variant_count": failed_rule_variants,
                "status": "blocked" if blockers else "admitted",
                "blocker_reasons": json.dumps(blockers, separators=(",", ":")),
                "schema_version": FINAL_MANIFEST_SCHEMA,
            }
        )
    output = pd.DataFrame.from_records(rows, columns=METRIC_COLUMNS)
    _metric_audit(output).raise_for_failure()
    return output


def _build_rule_variant_metrics(
    sample: pd.DataFrame,
    reviews: pd.DataFrame,
    *,
    population: Mapping[str, Mapping[str, int]],
    policy: IssuerEventPrecisionPolicy,
) -> pd.DataFrame:
    inferential = sample.loc[sample["sample_role"].eq("inferential")]
    joined = inferential.loc[:, ["sample_id", "proposed_event_family", "rule_variant"]].merge(
        reviews.loc[:, ["sample_id", "joint_correct"]],
        on="sample_id",
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, object]] = []
    for family in EVENT_FAMILIES:
        for variant, population_clusters in sorted(population.get(family, {}).items()):
            selected = joined.loc[joined["proposed_event_family"].eq(family) & joined["rule_variant"].eq(variant)]
            sample_clusters = len(selected)
            successes = _boolean_sum(selected, "joint_correct")
            lcb = _lcb_or_nan(successes, sample_clusters, policy.confidence_level)
            applicable = int(population_clusters) >= policy.rule_variant_gate_minimum_population_clusters
            blockers: list[str] = []
            if applicable and sample_clusters < policy.minimum_rule_variant_sample_clusters:
                blockers.append("insufficient_rule_variant_sample")
            if applicable and (math.isnan(lcb) or lcb < policy.minimum_rule_variant_joint_lcb):
                blockers.append("rule_variant_joint_lcb_below_threshold")
            rows.append(
                {
                    "event_family": family,
                    "rule_variant": variant,
                    "population_clusters": int(population_clusters),
                    "inferential_cluster_count": sample_clusters,
                    "joint_successes": successes,
                    "joint_lcb": lcb,
                    "minimum_joint_lcb": policy.minimum_rule_variant_joint_lcb,
                    "minimum_sample_clusters": (policy.minimum_rule_variant_sample_clusters),
                    "gate_applicable": applicable,
                    "status": ("blocked" if blockers else "admitted" if applicable else "diagnostic_only"),
                    "blocker_reasons": json.dumps(blockers, separators=(",", ":")),
                    "schema_version": FINAL_MANIFEST_SCHEMA,
                }
            )
    output = pd.DataFrame.from_records(rows, columns=RULE_VARIANT_METRIC_COLUMNS)
    _rule_variant_metric_audit(output).raise_for_failure()
    return output


def _reviewer_agreement_by_field(
    reviews: pd.DataFrame,
    *,
    minimum_decisions: int,
) -> dict[str, tuple[float, float, bool]]:
    output: dict[str, tuple[float, float, bool]] = {}
    for field in _DECISION_FIELDS:
        first = reviews[f"reviewer_one_{field}"].astype(str).tolist()
        second = reviews[f"reviewer_two_{field}"].astype(str).tolist()
        if not first:
            output[field] = (math.nan, math.nan, False)
            continue
        observed = sum(left == right for left, right in zip(first, second, strict=True)) / len(first)
        categories = sorted(set(first).union(second))
        first_counts = {value: first.count(value) / len(first) for value in categories}
        second_counts = {value: second.count(value) / len(second) for value in categories}
        expected = sum(first_counts[value] * second_counts[value] for value in categories)
        estimable = len(first) >= minimum_decisions and len(categories) > 1 and expected < 1.0
        kappa = (observed - expected) / (1.0 - expected) if estimable else math.nan
        output[field] = (observed, kappa, estimable)
    return output


def _sample_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    failures = 0
    if list(frame.columns) != list(SAMPLE_COLUMNS):
        failures += 1
    if not frame.empty:
        failures += int(frame["sample_id"].astype(str).duplicated().sum())
        failures += int(frame["family_event_id"].astype(str).duplicated().sum())
        failures += int((~frame["proposed_event_family"].astype(str).isin(EVENT_FAMILIES)).sum())
        failures += int(frame["title"].astype(str).str.strip().eq("").sum())
        failures += int(frame["source"].astype(str).str.strip().eq("").sum())
        failures += int((~frame["sample_role"].isin(("inferential", "paired_wrong_issuer_diagnostic"))).sum())
        inferential = frame.loc[frame["sample_role"].eq("inferential")]
        diagnostics = frame.loc[frame["sample_role"].eq("paired_wrong_issuer_diagnostic")]
        failures += int(inferential["inference_cluster_id"].astype(str).duplicated().sum())
        failures += int(inferential["paired_inferential_sample_id"].astype(str).ne("").sum())
        inferential_ids = set(inferential["sample_id"].astype(str))
        failures += int((~diagnostics["paired_inferential_sample_id"].astype(str).isin(inferential_ids)).sum())
        parent_clusters = inferential.set_index("sample_id")["inference_cluster_id"].astype(str).to_dict()
        failures += sum(
            parent_clusters.get(str(row["paired_inferential_sample_id"])) != str(row["inference_cluster_id"])
            for row in diagnostics.to_dict(orient="records")
        )
        availability = pd.to_datetime(frame["feature_available_at_utc"], utc=True, errors="coerce")
        identity_available = pd.to_datetime(frame["identity_available_at_utc"], utc=True, errors="coerce")
        resolved = frame["identity_status"].eq("resolved")
        failures += int((resolved & identity_available.gt(availability)).sum())
    return _audit_report("issuer_event_precision_sample", len(frame), failures)


def _review_audit(frame: pd.DataFrame, sample: pd.DataFrame) -> CanonicalAuditReport:
    failures = int(list(frame.columns) != list(REVIEW_COLUMNS))
    failures += abs(len(frame) - len(sample))
    if not frame.empty:
        failures += int(frame["sample_id"].astype(str).duplicated().sum())
        failures += len(set(frame["sample_id"].astype(str)).symmetric_difference(set(sample["sample_id"].astype(str))))
    return _audit_report("issuer_event_precision_reviews", len(frame), failures)


def _metric_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    failures = int(list(frame.columns) != list(METRIC_COLUMNS))
    failures += abs(len(frame) - len(EVENT_FAMILIES))
    if not frame.empty:
        failures += len(set(frame["event_family"].astype(str)).symmetric_difference(EVENT_FAMILIES))
        failures += int((~frame["status"].isin(("admitted", "blocked"))).sum())
    return _audit_report("issuer_event_precision_metrics", len(frame), failures)


def _rule_variant_metric_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    failures = int(list(frame.columns) != list(RULE_VARIANT_METRIC_COLUMNS))
    if not frame.empty:
        failures += int(frame.duplicated(["event_family", "rule_variant"]).sum())
        failures += int((~frame["event_family"].isin(EVENT_FAMILIES)).sum())
        failures += int((~frame["status"].isin(("admitted", "blocked", "diagnostic_only"))).sum())
    return _audit_report("issuer_event_precision_rule_variant_metrics", len(frame), failures)


def _audit_report(name: str, rows: int, failures: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name=name,
                status="pass" if failures == 0 else "fail",
                failures=failures,
                rows_checked=rows,
                detail="deterministic issuer-event precision authority validation",
            ),
        )
    )


def _population_from_manifest(
    manifest: Mapping[str, object],
) -> dict[str, dict[str, int]]:
    raw = manifest.get("population")
    if not isinstance(raw, dict) or set(raw) != set(EVENT_FAMILIES):
        raise DataReadinessError("precision sample population summary is malformed")
    output: dict[str, dict[str, int]] = {}
    for family in EVENT_FAMILIES:
        value = raw.get(family)
        if not isinstance(value, dict):
            raise DataReadinessError("precision sample family population is malformed")
        if set(value) != {"eligible_events", "clusters", "issuers"}:
            raise DataReadinessError("precision sample family population fields differ")
        output[family] = {
            "eligible_events": _nonnegative_int(value, "eligible_events"),
            "clusters": _nonnegative_int(value, "clusters"),
            "issuers": _nonnegative_int(value, "issuers"),
        }
    return output


def _rule_variant_population_from_manifest(
    manifest: Mapping[str, object],
) -> dict[str, dict[str, int]]:
    raw = manifest.get("rule_variant_population")
    if not isinstance(raw, dict) or set(raw) != set(EVENT_FAMILIES):
        raise DataReadinessError("precision rule-variant population is malformed")
    output: dict[str, dict[str, int]] = {}
    for family in EVENT_FAMILIES:
        values = raw.get(family)
        if not isinstance(values, dict):
            raise DataReadinessError("precision family rule-variant population is malformed")
        output[family] = {str(variant): _nonnegative_int(values, str(variant)) for variant in sorted(values)}
    return output


def _request(manifest: Mapping[str, object], authority: Mapping[str, object]) -> dict[str, object]:
    value = manifest.get("request")
    if (
        not isinstance(value, dict)
        or _json_sha256(value) != manifest.get("request_sha256")
        or authority.get("request_sha256") != manifest.get("request_sha256")
    ):
        raise DataReadinessError("precision authority request does not verify")
    return {str(key): item for key, item in value.items()}


def _artifact_records(manifest: Mapping[str, object]) -> dict[str, dict[str, object]]:
    raw = manifest.get("artifacts")
    if not isinstance(raw, dict):
        raise DataReadinessError("precision authority artifact inventory is malformed")
    output: dict[str, dict[str, object]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            raise DataReadinessError("precision authority artifact record is malformed")
        output[str(key)] = {str(name): item for name, item in value.items()}
    return output


def _artifact_record(path: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": str(manifest["artifact_sha256"]),
        "rows": _nonnegative_int(manifest, "rows"),
    }


def _file_record(path: Path, rows: int) -> dict[str, object]:
    return {"path": path.name, "sha256": file_sha256(path), "rows": rows}


def _verify_canonical_record(
    path: Path,
    frame: pd.DataFrame,
    child: Mapping[str, object],
    records: Mapping[str, Mapping[str, object]],
    key: str,
    *,
    request_sha256: str,
) -> None:
    record = records.get(key)
    inputs = child.get("inputs")
    if (
        record is None
        or record.get("path") != path.name
        or record.get("sha256") != child.get("artifact_sha256")
        or record.get("rows") != len(frame)
        or not isinstance(inputs, dict)
        or inputs.get("request_sha256") != request_sha256
    ):
        raise DataReadinessError(f"precision {key} artifact lineage does not verify")


def _verify_file_record(
    path: Path,
    records: Mapping[str, Mapping[str, object]],
    key: str,
    rows: int,
) -> None:
    record = records.get(key)
    if record is None or record.get("path") != path.name or record.get("sha256") != file_sha256(path) or record.get("rows") != rows:
        raise DataReadinessError(f"precision {key} file lineage does not verify")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, lineterminator="\n")


def _read_csv(path: Path, expected_columns: Sequence[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != list(expected_columns):
        raise DataReadinessError(f"precision ledger schema differs: {path}")
    return frame


def _verify_inventory(directory: Path, expected: set[str]) -> None:
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    if observed != expected or any(path.is_dir() for path in directory.iterdir()):
        raise DataReadinessError("precision authority file inventory differs")


def _new_staging(output_directory: Path) -> Path:
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = output_directory.with_name(f".{output_directory.name}.{uuid4().hex}.tmp")
    staging.mkdir()
    return staging


def _rewrite_artifact_path(path: Path, final_path: Path) -> None:
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest = _json_object(manifest_path)
    manifest["artifact_path"] = str(final_path.resolve())
    _atomic_json(manifest_path, manifest)


def _remove_lock(path: Path) -> None:
    path.with_name(f"{path.name}.lock").unlink(missing_ok=True)


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


def _json_object(path: Path) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"expected JSON object: {path}")
    return {str(key): value for key, value in loaded.items()}


def _json_text_object(value: str, context: str) -> dict[str, object]:
    try:
        loaded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise DataReadinessError(f"{context} is not valid JSON") from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"{context} is malformed")
    return {str(key): item for key, item in loaded.items()}


def _json_sha256(value: object) -> str:
    return _sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            default=str,
        )
    )


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _normalize_title(value: str) -> str:
    return " ".join(_NORMALIZE_PATTERN.sub(" ", value.lower()).split())


def _clean_text(value: object) -> str:
    if value is None or (not isinstance(value, (list, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _required_path(record: Mapping[str, object], key: str) -> Path:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise DataReadinessError(f"precision lineage has invalid {key}")
    return Path(value).resolve()


def _manifest_request(manifest: Mapping[str, object]) -> Mapping[str, object]:
    value = manifest.get("request")
    if not isinstance(value, dict):
        raise DataReadinessError("precision manifest request is malformed")
    return value


def _required_hash(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or len(value) != 64 or any(character not in "0123456789abcdef" for character in value.lower()):
        raise DataReadinessError(f"precision lineage has invalid {key}")
    return value


def _required_text(record: Mapping[str, object], key: str) -> str:
    value = _clean_text(record.get(key))
    if not value:
        raise DataReadinessError(f"precision authority requires {key}")
    return value


def _positive_int(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise DataReadinessError(f"precision policy has invalid {key}")
    return value


def _nonnegative_int(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise DataReadinessError(f"precision lineage has invalid {key}")
    return value


def _bounded_float(
    record: Mapping[str, object],
    key: str,
    *,
    lower: float,
    upper: float,
    inclusive: bool = False,
) -> float:
    value = record.get(key)
    if not isinstance(value, (float, int)) or isinstance(value, bool):
        raise DataReadinessError(f"precision policy has invalid {key}")
    parsed = float(value)
    valid = lower <= parsed <= upper if inclusive else lower < parsed < upper
    if not valid:
        raise DataReadinessError(f"precision policy has invalid {key}")
    return parsed


def _threshold(record: Mapping[str, object], key: str) -> float:
    return _bounded_float(record, key, lower=0.0, upper=1.0, inclusive=True)


def _timestamp(value: object, label: str) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise DataReadinessError(f"precision {label} is invalid")
    return pd.Timestamp(parsed)


def _optional_timestamp(value: object) -> pd.Timestamp | None:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(parsed) else pd.Timestamp(parsed)


def _boolean_sum(frame: pd.DataFrame, column: str) -> int:
    return int(frame[column].astype(bool).sum()) if not frame.empty else 0


def _lcb_or_nan(successes: int, total: int, confidence: float) -> float:
    return wilson_lower_bound(successes, total, confidence) if total else math.nan


def _overall_audit_status(admitted: Sequence[object], blocked: Sequence[object]) -> str:
    if admitted and blocked:
        return "partial"
    if admitted:
        return "admitted"
    return "blocked"


def _guard_memory(policy: IssuerEventPrecisionPolicy, stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=policy.maximum_process_memory_gib,
        headroom_gib=policy.memory_guard_headroom_gib,
        stage=stage,
    )


def _assert_frame_equal(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    label: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            observed.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_exact=True,
            check_dtype=False,
        )
    except AssertionError as exc:
        raise DataReadinessError(f"{label} failed") from exc
