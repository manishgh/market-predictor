"""Final review admission decisions and replay for issuer-event precision evidence."""
from __future__ import annotations

import json
import math
import os
import shutil
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditReport
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    write_canonical_artifact,
)
from market_predictor.catalysts.issuer_events.classification import (
    EVENT_FAMILIES,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.issuer_event_precision.artifact_integrity import (
    _artifact_record,
    _artifact_records,
    _assert_frame_equal,
    _atomic_json,
    _audit_report,
    _file_record,
    _json_object,
    _json_sha256,
    _manifest_request,
    _new_staging,
    _remove_lock,
    _request,
    _required_hash,
    _required_path,
    _rewrite_artifact_path,
    _verify_canonical_record,
    _verify_file_record,
    _verify_inventory,
)
from market_predictor.governance.issuer_event_precision.contracts import (
    _DECISION_FIELDS,
    FINAL_AUTHORITY_SCHEMA,
    FINAL_MANIFEST_SCHEMA,
    METRIC_COLUMNS,
    METRICS_ARTIFACT_TYPE,
    REVIEWS_ARTIFACT_TYPE,
    RULE_VARIANT_METRIC_COLUMNS,
    RULE_VARIANT_METRICS_ARTIFACT_TYPE,
    IssuerEventPrecisionAudit,
    IssuerEventPrecisionPolicy,
    _guard_memory,
    load_issuer_event_precision_policy,
)
from market_predictor.governance.issuer_event_precision.review_resolution import (
    _load_adjudication_ledger,
    _load_review_ledger,
    _preflight_review_ledgers,
    _resolve_reviews,
    _review_audit,
)
from market_predictor.governance.issuer_event_precision.sample_authority import (
    _population_from_manifest,
    _rule_variant_population_from_manifest,
    load_issuer_event_precision_sample,
)
from market_predictor.resources import (
    memory_audit,
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
        for path in (review_path, metric_path, rule_variant_metric_path):
            _rewrite_artifact_path(path, output_directory / path.name)
        _load_issuer_event_precision_audit(
            staging,
            _expected_artifact_directory=output_directory,
        )
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

    return _load_issuer_event_precision_audit(
        directory,
        expected_authority_sha256=expected_authority_sha256,
        retain_source_authority=retain_source_authority,
    )


def _load_issuer_event_precision_audit(
    directory: Path,
    *,
    expected_authority_sha256: str | None = None,
    retain_source_authority: bool = False,
    _expected_artifact_directory: Path | None = None,
) -> IssuerEventPrecisionAudit:
    """Load final or staged audit evidence with an internal path binding."""

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
        expected_artifact_path=(
            _expected_artifact_directory / reviews_path.name
            if _expected_artifact_directory is not None
            else None
        ),
    )
    _verify_canonical_record(
        rule_variant_metrics_path,
        rule_variant_metrics,
        rule_variant_metric_child,
        artifacts,
        "rule_variant_metrics",
        request_sha256=str(manifest["request_sha256"]),
        expected_artifact_path=(
            _expected_artifact_directory / rule_variant_metrics_path.name
            if _expected_artifact_directory is not None
            else None
        ),
    )
    _verify_canonical_record(
        metrics_path,
        metrics,
        metric_child,
        artifacts,
        "family_metrics",
        request_sha256=str(manifest["request_sha256"]),
        expected_artifact_path=(
            _expected_artifact_directory / metrics_path.name
            if _expected_artifact_directory is not None
            else None
        ),
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
