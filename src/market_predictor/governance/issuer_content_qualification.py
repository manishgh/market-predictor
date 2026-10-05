"""Frozen, source-clustered review sampling and measured content admission.

This module does not infer labels. Distinct reviewers supply source-only labels;
the publishing caller binds their files, source text, spans and implementation.
Recall describes the retained readable source population, not provider completeness.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.governance.issuer_event_precision.admission_authority import wilson_lower_bound

Family = Literal["earnings", "guidance"]
Role = Literal["candidate", "noncandidate"]
FAMILIES: tuple[Family, ...] = ("earnings", "guidance")
POSITIVE_SAMPLES = {"earnings": 300, "guidance": 250}
MINIMUM_ISSUERS = {"earnings": 75, "guidance": 60}
NEGATIVE_SAMPLES = 600
POLICY = {
    "schema": "market_predictor.issuer_content_qualification",
    "population": "retained_initial_fit_readable_source_announcement_clusters_not_provider_completeness",
    "clustering": "source_event_identity_with_all_retained_versions_one_inferential_unit_per_announcement",
    "candidate_sampling": "uniform_sha256_fixed_seed_42_per_family",
    "noncandidate_sampling": "source_year_stratified_uniform_sha256_seed_42_recorded_inclusion_probability",
    "samples": POSITIVE_SAMPLES, "noncandidate_samples_per_family": NEGATIVE_SAMPLES,
    "minimum_population_clusters": 100, "minimum_issuers": MINIMUM_ISSUERS,
    "confidence": 0.95, "minimum_joint_precision_lcb": 0.95, "minimum_issuer_precision_lcb": 0.98,
    "minimum_family_precision_lcb": 0.95, "minimum_event_precision_lcb": 0.95,
    "rule_variant_gate_minimum_population_clusters": 50,
    "minimum_rule_variant_sample_clusters": 10, "minimum_rule_variant_joint_lcb": 0.80,
    "reviewers": 2, "minimum_agreement": 0.90, "minimum_kappa": 0.80, "minimum_kappa_decisions": 30,
    "unresolved": "precision_failure_and_worst_case_recall_miss",
    "wrong_issuer": "any_wrong_issuer_in_candidate_review_rejects_family",
    "recall": "inverse_probability_weighted_counts_and_simultaneous_stratum_bounds_reported_no_new_threshold",
    "missingness": "unknown_unreadable_or_unqualified_event_never_a_zero_reaction",
    "annotation": "source_only_two_independent_reviewers_human_or_explicitly_model_assisted",
}
POLICY_SHA256 = json_sha256(POLICY)


def _require(value: bool, message: str) -> None:
    if not value:
        raise DataReadinessError(message)


@dataclass(frozen=True)
class ReviewCluster:
    cluster_id: str
    security_id: str
    source_family: Literal["alpaca", "sec"]
    year: int
    candidate_families: tuple[Family, ...]
    rule_ids: tuple[tuple[Family, str], ...]


@dataclass(frozen=True)
class SampleItem:
    sample_id: str
    cluster_id: str
    event_family: Family
    role: Role
    stratum: str
    population_clusters: int
    sampled_clusters: int
    inclusion_probability: float


def _rank(cluster: ReviewCluster, family: Family, role: Role) -> str:
    return json_sha256([POLICY_SHA256, 42, family, role, cluster.cluster_id])


def select_content_review_sample(clusters: Sequence[ReviewCluster]) -> tuple[SampleItem, ...]:
    """One inferential unit per announcement cluster, never one per copied article."""
    _require(len({row.cluster_id for row in clusters}) == len(clusters), "content review clusters are duplicated")
    for row in clusters:
        _require(bool(row.cluster_id.strip()) and bool(row.security_id.strip())
                 and row.source_family in ("alpaca", "sec") and 2019 <= row.year <= 2024,
                 "content review cluster lies outside the frozen source/window")
        _require(len(set(row.candidate_families)) == len(row.candidate_families)
                 and set(row.candidate_families).issubset(FAMILIES), "unknown or duplicate proposed event family")
        _require(len(set(row.rule_ids)) == len(row.rule_ids)
                 and all(family in row.candidate_families and bool(rule.strip()) for family, rule in row.rule_ids)
                 and {family for family, _ in row.rule_ids} == set(row.candidate_families),
                 "candidate families require exact nonempty rule variant identities")
    result: list[SampleItem] = []
    for family in FAMILIES:
        positives = [row for row in clusters if family in row.candidate_families]
        selected = sorted(positives, key=lambda row: _rank(row, family, "candidate"))[:POSITIVE_SAMPLES[family]]
        for row in selected:
            result.append(SampleItem(json_sha256([POLICY_SHA256, family, "candidate", row.cluster_id]),
                row.cluster_id, family, "candidate", "candidate_uniform", len(positives), len(selected), len(selected) / len(positives)))
        strata: dict[str, list[ReviewCluster]] = defaultdict(list)
        for row in clusters:
            if family not in row.candidate_families:
                strata[f"{row.source_family}/{row.year}"].append(row)
        # Allocate one per nonempty stratum, then approximately proportionally.
        # The exact resulting N/n weights, rather than equal stratum averages,
        # determine the recall estimate and interval.
        counts = dict.fromkeys(sorted(strata), 0)
        for _ in range(min(NEGATIVE_SAMPLES, sum(len(rows) for rows in strata.values()))):
            eligible = [key for key in counts if counts[key] < len(strata[key])]
            key = min(eligible, key=lambda value: (counts[value] / len(strata[value]), value))
            counts[key] += 1
        for key, rows in sorted(strata.items()):
            chosen = sorted(rows, key=lambda row: _rank(row, family, "noncandidate"))[:counts[key]]
            for row in chosen:
                result.append(SampleItem(json_sha256([POLICY_SHA256, family, "noncandidate", row.cluster_id]),
                    row.cluster_id, family, "noncandidate", key, len(rows), len(chosen), len(chosen) / len(rows)))
    return tuple(sorted(result, key=lambda item: item.sample_id))


@dataclass(frozen=True)
class ContentReview:
    sample_id: str
    reviewer_id: str
    reviewer_kind: Literal["human", "model_assisted"]
    reviewer_model: str | None
    family_present: bool | None
    issuer_correct: bool | None
    announced_or_reported: bool | None
    explicit_fiscal_period: bool | None
    action_supported: bool | None
    supporting_text_sha256: str
    supporting_spans: tuple[tuple[int, int], ...]
    evidence_scope: Literal["source_only"] = "source_only"
    independent_review: bool = True

    @property
    def resolved(self) -> bool:
        return all(type(value) is bool for value in (
            self.family_present, self.issuer_correct, self.announced_or_reported, self.explicit_fiscal_period, self.action_supported))

    @property
    def joint(self) -> bool:
        return self.resolved and all((self.family_present, self.issuer_correct, self.announced_or_reported,
                                     self.explicit_fiscal_period, self.action_supported))


_REVIEW_FIELDS = (
    "family_present", "issuer_correct", "announced_or_reported",
    "explicit_fiscal_period", "action_supported",
)


def _wilson(successes: int, count: int, confidence: float) -> tuple[float, float]:
    if count == 0:
        return 0.0, 1.0
    return (
        wilson_lower_bound(successes, count, confidence),
        1.0 - wilson_lower_bound(count - successes, count, confidence),
    )


def _validate_reviews(
    samples: Sequence[SampleItem], reviews: Sequence[ContentReview],
) -> dict[str, list[ContentReview]]:
    by_sample: dict[str, list[ContentReview]] = defaultdict(list)
    sample_ids = {item.sample_id for item in samples}
    identities: dict[str, tuple[str, str | None]] = {}
    for row in reviews:
        _require(row.sample_id in sample_ids and bool(row.reviewer_id.strip()), "review has an unknown sample/reviewer")
        _require(row.reviewer_kind in ("human", "model_assisted")
                 and ((row.reviewer_kind == "human" and row.reviewer_model is None)
                      or (row.reviewer_kind == "model_assisted" and bool(row.reviewer_model and row.reviewer_model.strip()))),
                 "review assistance is not declared")
        _require(row.evidence_scope == "source_only" and row.independent_review is True,
                 "content reviews require independent source-only evidence")
        _require(all(getattr(row, field) is None or type(getattr(row, field)) is bool for field in _REVIEW_FIELDS),
                 "review decisions must be booleans or explicitly unresolved")
        identity = (row.reviewer_kind, row.reviewer_model)
        _require(identities.setdefault(row.reviewer_id, identity) == identity, "reviewer identity declaration changed")
        _require(len(row.supporting_text_sha256) == 64
                 and all(ch in "0123456789abcdef" for ch in row.supporting_text_sha256), "review text hash is missing")
        _require(not any(getattr(row, field) is True for field in _REVIEW_FIELDS) or bool(row.supporting_spans),
                 "positive review decisions require original text evidence spans")
        _require(all(type(a) is int and type(b) is int and 0 <= a < b for a, b in row.supporting_spans),
                 "invalid review text spans")
        by_sample[row.sample_id].append(row)
    for rows in by_sample.values():
        _require(len(rows) <= 2 and len({row.reviewer_id for row in rows}) == len(rows),
                 "reviewers are repeated or exceed two")
        _require(len({row.supporting_text_sha256 for row in rows}) == 1, "reviewers did not review the same source text")
        rows.sort(key=lambda row: row.reviewer_id)
    return by_sample


def _agrees(rows: Sequence[ContentReview]) -> bool:
    return (len(rows) == 2 and all(row.resolved for row in rows)
            and all(getattr(rows[0], field) == getattr(rows[1], field) for field in _REVIEW_FIELDS))


def _agreement(pairs: Sequence[tuple[bool, bool]]) -> dict[str, object]:
    count = len(pairs)
    observed = sum(left == right for left, right in pairs) / count if count else None
    first = sum(left for left, _ in pairs) / count if count else 0.0
    second = sum(right for _, right in pairs) / count if count else 0.0
    chance = first * second + (1 - first) * (1 - second)
    estimable = count >= 30 and chance < 1.0
    kappa = (observed - chance) / (1 - chance) if estimable and observed is not None else None
    return {"paired_decisions": count, "agreement": observed, "kappa": kappa, "kappa_estimable": estimable}


def _rule_metrics(
    family: Family, population: Sequence[ReviewCluster], positive: Sequence[SampleItem],
    successes: dict[str, bool],
) -> list[dict[str, object]]:
    populations: dict[str, set[str]] = defaultdict(set)
    for cluster in population:
        for event_family, rule in cluster.rule_ids:
            if event_family == family:
                populations[rule].add(cluster.cluster_id)
    result: list[dict[str, object]] = []
    for rule, cluster_ids in sorted(populations.items()):
        selected = [item for item in positive if item.cluster_id in cluster_ids]
        count = sum(successes[item.sample_id] for item in selected)
        lower = _wilson(count, len(selected), 0.95)[0]
        applicable = len(cluster_ids) >= 50
        reasons = []
        if applicable and len(selected) < 10:
            reasons.append("insufficient_rule_variant_sample")
        if applicable and lower < 0.80:
            reasons.append("rule_variant_joint_lcb_below_threshold")
        result.append({
            "rule_id": rule, "population_clusters": len(cluster_ids), "sample_clusters": len(selected),
            "joint_successes": count, "joint_precision_lower_bound": lower, "gate_applicable": applicable,
            "status": "blocked" if reasons else "admitted" if applicable else "diagnostic_only", "reasons": reasons,
        })
    return result


def evaluate_content_reviews(
    *, clusters: Sequence[ReviewCluster], samples: Sequence[SampleItem], reviews: Sequence[ContentReview],
) -> dict[str, object]:
    """Measure source-only labels; this does not verify source bytes or authorize a model.

    The publication owner must verify each text hash/span against the pinned source.
    Unresolved/disagreed candidate annotations fail precision; corresponding negative
    annotations count as possible false negatives. Recall is measured, never a veto.
    """
    _require(tuple(samples) == select_content_review_sample(clusters),
             "content sample differs from the frozen population/policy")
    by_sample = _validate_reviews(samples, reviews)
    metrics: dict[str, object] = {}
    for family in FAMILIES:
        selected = [item for item in samples if item.event_family == family]
        positive = [item for item in selected if item.role == "candidate"]
        population = [row for row in clusters if family in row.candidate_families]
        counts: Counter[str] = Counter()
        field_pairs: dict[str, list[tuple[bool, bool]]] = {field: [] for field in _REVIEW_FIELDS}
        successes: dict[str, bool] = {}
        negative_counts: Counter[str] = Counter()
        negative_sizes: dict[str, tuple[int, int]] = {}
        negative_unresolved: Counter[str] = Counter()
        incomplete_negative_reviews = 0
        for item in selected:
            rows = by_sample.get(item.sample_id, [])
            agreed = _agrees(rows)
            if item.role == "candidate":
                for field in _REVIEW_FIELDS:
                    if len(rows) == 2:
                        left, right = getattr(rows[0], field), getattr(rows[1], field)
                        if type(left) is bool and type(right) is bool:
                            field_pairs[field].append((left, right))
                ok = agreed and all(row.joint for row in rows)
                successes[item.sample_id] = ok
                counts["joint"] += int(ok)
                # Every unresolved or disputed cluster is a failure for each
                # precision measure, including a dispute outside that field.
                for metric, field in (("family", "family_present"), ("issuer", "issuer_correct"),
                                      ("event", "announced_or_reported")):
                    counts[metric] += int(agreed and all(getattr(row, field) is True for row in rows))
                counts["wrong_issuer"] += int(any(row.issuer_correct is False for row in rows))
                counts["unresolved"] += int(not agreed)
            else:
                # Validation already enforces distinct independent source-only
                # reviewers. Uncertain labels are allowed; missing work is not.
                incomplete_negative_reviews += int(len(rows) != 2)
                negative_counts[item.stratum] += int(not agreed or any(row.joint for row in rows))
                negative_unresolved[item.stratum] += int(not agreed)
                negative_sizes[item.stratum] = (item.population_clusters, item.sampled_clusters)
        n = len(positive)
        reasons = []
        if len(population) < 100:
            reasons.append("fewer_than_100_candidate_announcement_clusters")
        if len({row.security_id for row in population}) < MINIMUM_ISSUERS[family]:
            reasons.append("too_few_distinct_candidate_issuers")
        if n != min(POSITIVE_SAMPLES[family], len(population)):
            reasons.append("candidate_precision_sample_incomplete")
        measured: dict[str, object] = {}
        for name, threshold in (("family", 0.95), ("issuer", 0.98), ("event", 0.95), ("joint", 0.95)):
            lower = _wilson(counts[name], n, 0.95)[0]
            measured[f"{name}_successes"] = counts[name]
            measured[f"{name}_precision"] = counts[name] / n if n else None
            measured[f"{name}_precision_lower_bound"] = lower
            if lower < threshold:
                reasons.append(f"{name}_precision_lower_bound_below_threshold")
        if counts["wrong_issuer"]:
            reasons.append("wrong_issuer_in_candidate_review")
        if counts["unresolved"]:
            reasons.append("unresolved_or_disagreed_candidate_annotations")
        if incomplete_negative_reviews:
            reasons.append("noncandidate_review_sample_incomplete")
        agreement = {field: _agreement(pairs) for field, pairs in field_pairs.items()}
        for field, pairs in field_pairs.items():
            assessment = agreement[field]
            observed, kappa = assessment["agreement"], assessment["kappa"]
            if not pairs or (isinstance(observed, float) and observed < 0.90):
                reasons.append(f"{field}_review_agreement_requirement_not_met")
            if isinstance(kappa, float) and kappa < 0.80:
                reasons.append(f"{field}_review_kappa_requirement_not_met")
        variants = _rule_metrics(family, population, positive, successes)
        if any(item["status"] == "blocked" for item in variants):
            reasons.append("rule_variant_gate_failed")
        estimated_tp = len(population) * counts["joint"] / n if n else 0.0
        estimated_fn = sum(negative_counts[key] * total / sample for key, (total, sample) in negative_sizes.items())
        # A union bound over the sampled candidate and negative strata. Census
        # totals are exact; uncertainty from unresolved labels remains worst-case.
        level = 1 - 0.05 / (len(negative_sizes) + 1)
        tp_lower = float(counts["joint"]) if n == len(population) else len(population) * _wilson(counts["joint"], n, level)[0]
        fn_upper = sum(float(negative_counts[key]) if total == sample else
                       total * _wilson(negative_counts[key], sample, level)[1]
                       for key, (total, sample) in negative_sizes.items())
        recall = estimated_tp / (estimated_tp + estimated_fn) if estimated_tp + estimated_fn else None
        recall_lower = tp_lower / (tp_lower + fn_upper) if tp_lower + fn_upper else None
        metrics[family] = {
            "candidate_population_clusters": len(population),
            "candidate_population_issuers": len({row.security_id for row in population}),
            "candidate_review_clusters": n, **measured,
            "wrong_issuer_clusters": counts["wrong_issuer"], "unresolved_candidate_clusters": counts["unresolved"],
            "incomplete_noncandidate_review_clusters": incomplete_negative_reviews,
            "reviewer_agreement_by_field": agreement, "rule_variants": variants,
            "estimated_true_positive_clusters": estimated_tp, "estimated_false_negative_clusters": estimated_fn,
            "true_positive_simultaneous_lower_bound": tp_lower, "false_negative_simultaneous_upper_bound": fn_upper,
            "recall_estimate": recall, "recall_simultaneous_lower_bound": recall_lower,
            "recall_scope": POLICY["population"], "recall_threshold": None,
            "recall_strata": {
                key: {"population": total, "sample": sample, "inclusion_probability": sample / total,
                      "possible_misses": negative_counts[key], "unresolved_clusters": negative_unresolved[key],
                      "census": total == sample}
                for key, (total, sample) in sorted(negative_sizes.items())
            },
            "qualified_for_historical_feature_use": not reasons, "reasons": reasons,
        }
    return {
        "schema": POLICY["schema"], "policy_sha256": POLICY_SHA256, "families": metrics,
        "reviewer_kinds": sorted({row.reviewer_kind for row in reviews}),
        "serving_eligible": False, "promotion_eligible": False, "provider_completeness_established": False,
    }
