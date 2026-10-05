"""Source-only synthetic precision, sampling and conservative recall contracts."""

from collections import Counter
from dataclasses import replace
from typing import Any

import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.issuer_content_qualification import (
    ContentReview,
    Family,
    ReviewCluster,
    SampleItem,
    evaluate_content_reviews,
    select_content_review_sample,
)
from market_predictor.governance.issuer_event_precision.admission_authority import wilson_lower_bound


def _population(earnings: int = 300, guidance: int = 250, negatives: int = 700) -> list[ReviewCluster]:
    result = []
    populations: tuple[tuple[Family | None, int], ...] = (("earnings", earnings), ("guidance", guidance), (None, negatives))
    for family, count in populations:
        for number in range(count):
            result.append(ReviewCluster(
                cluster_id=f"{family}/{number}", security_id=f"issuer/{number}",
                source_family="alpaca" if number % 2 else "sec", year=2020 + number % 3,
                candidate_families=(family,) if family else (),
                rule_ids=((family, f"{family}_statement"),) if family else (),
            ))
    return result


def _reviews(samples: tuple[SampleItem, ...], *, negative_present: bool = False) -> list[ContentReview]:
    result = []
    for item in samples:
        present = item.role == "candidate" or negative_present
        for reviewer in ("first", "second"):
            result.append(ContentReview(
                sample_id=item.sample_id, reviewer_id=reviewer, reviewer_kind="model_assisted",
                reviewer_model="synthetic-test-reviewer", family_present=present, issuer_correct=present,
                announced_or_reported=present, explicit_fiscal_period=present, action_supported=present,
                supporting_text_sha256="a" * 64, supporting_spans=((0, 15),) if present else (),
            ))
    return result


def _evaluate(clusters: list[ReviewCluster], reviews: list[ContentReview] | None = None) -> dict[str, Any]:
    samples = select_content_review_sample(clusters)
    return evaluate_content_reviews(clusters=clusters, samples=samples, reviews=_reviews(samples) if reviews is None else reviews)


def test_fixed_samples_reproducible_with_recorded_stratum_probabilities() -> None:
    clusters = _population(450, 350, 1300)
    samples = select_content_review_sample(clusters)
    assert samples == select_content_review_sample(list(reversed(clusters)))
    assert Counter((item.event_family, item.role) for item in samples) == {
        ("earnings", "candidate"): 300, ("guidance", "candidate"): 250,
        ("earnings", "noncandidate"): 600, ("guidance", "noncandidate"): 600,
    }
    for item in samples:
        assert item.inclusion_probability == item.sampled_clusters / item.population_clusters
    assert len({(item.event_family, item.cluster_id) for item in samples}) == len(samples)


def test_all_correct_candidates_qualify_without_estimable_kappa_or_serving_claim() -> None:
    result = _evaluate(_population())
    for family in ("earnings", "guidance"):
        measured = result["families"][family]
        assert measured["qualified_for_historical_feature_use"]
        assert measured["reasons"] == []
        for name in ("family", "issuer", "event", "joint"):
            assert measured[f"{name}_precision"] == 1.0
            assert measured[f"{name}_precision_lower_bound"] == wilson_lower_bound(
                measured["candidate_review_clusters"], measured["candidate_review_clusters"],
            )
        for agreement in measured["reviewer_agreement_by_field"].values():
            assert agreement["agreement"] == 1.0
            assert agreement["kappa"] is None
            assert not agreement["kappa_estimable"]
    assert not result["serving_eligible"]
    assert not result["promotion_eligible"]
    assert not result["provider_completeness_established"]


@pytest.mark.parametrize("field,metric", [
    ("family_present", "family"), ("issuer_correct", "issuer"), ("announced_or_reported", "event"),
    ("explicit_fiscal_period", "joint"), ("action_supported", "joint"),
])
def test_actual_separate_precision_counts(field: str, metric: str) -> None:
    clusters = _population()
    samples = select_content_review_sample(clusters)
    candidate = next(item for item in samples if item.event_family == "earnings" and item.role == "candidate")
    change: dict[str, Any] = {field: False}
    reviews = [replace(row, **change) if row.sample_id == candidate.sample_id else row for row in _reviews(samples)]
    result = _evaluate(clusters, reviews)["families"]["earnings"]
    assert result[f"{metric}_successes"] == 299
    assert result["joint_successes"] == 299
    for other in {"family", "issuer", "event"} - {metric}:
        assert result[f"{other}_successes"] == 300
    if field == "issuer_correct":
        assert "wrong_issuer_in_candidate_review" in result["reasons"]


def test_field_disagreement_cannot_hide_behind_equal_negative_joint_labels() -> None:
    clusters = _population()
    samples = select_content_review_sample(clusters)
    candidate = next(item for item in samples if item.event_family == "earnings" and item.role == "candidate")
    reviews = _reviews(samples)
    reviews = [replace(row, family_present=False) if row.sample_id == candidate.sample_id and row.reviewer_id == "first"
               else replace(row, action_supported=False) if row.sample_id == candidate.sample_id else row for row in reviews]
    result = _evaluate(clusters, reviews)["families"]["earnings"]
    assert result["unresolved_candidate_clusters"] == 1
    assert result["issuer_successes"] == 299
    assert result["reviewer_agreement_by_field"]["family_present"]["kappa_estimable"]
    assert not result["qualified_for_historical_feature_use"]


def test_census_uses_exact_recall_totals_and_accepts_population_below_nominal_sample() -> None:
    clusters = _population(200, 0, 50)
    samples = select_content_review_sample(clusters)
    result = _evaluate(clusters, _reviews(samples, negative_present=True))["families"]["earnings"]
    assert result["qualified_for_historical_feature_use"]
    assert result["recall_estimate"] == result["recall_simultaneous_lower_bound"] == 0.8
    assert result["estimated_true_positive_clusters"] == result["true_positive_simultaneous_lower_bound"] == 200
    assert result["estimated_false_negative_clusters"] == result["false_negative_simultaneous_upper_bound"] == 50
    assert all(item["census"] for item in result["recall_strata"].values())


def test_low_measured_recall_does_not_create_new_admission_threshold() -> None:
    clusters = _population(300, 0, 1800)
    result = _evaluate(clusters, _reviews(select_content_review_sample(clusters), negative_present=True))["families"]["earnings"]
    assert result["recall_estimate"] == pytest.approx(1 / 7)
    assert result["recall_threshold"] is None
    assert result["qualified_for_historical_feature_use"]


@pytest.mark.parametrize("negative_reviewers", [0, 1])
def test_incomplete_noncandidate_reviews_block_qualification(negative_reviewers: int) -> None:
    clusters = _population()
    samples = select_content_review_sample(clusters)
    candidates = {item.sample_id for item in samples if item.role == "candidate"}
    reviews = [row for row in _reviews(samples)
               if row.sample_id in candidates or (negative_reviewers == 1 and row.reviewer_id == "first")]
    result = _evaluate(clusters, reviews)
    for family in ("earnings", "guidance"):
        measured = result["families"][family]
        assert measured["joint_precision"] == 1.0
        assert measured["incomplete_noncandidate_review_clusters"] == 600
        assert "noncandidate_review_sample_incomplete" in measured["reasons"]
        assert not measured["qualified_for_historical_feature_use"]


def test_stratified_recall_uses_population_weights_and_nonzero_zero_miss_uncertainty() -> None:
    clusters = _population(300, 0, 2000)
    clusters = [replace(row, source_family="sec", year=2020) if row.cluster_id.startswith("None/")
                else row for row in clusters]
    clusters.append(ReviewCluster("rare", "rare", "alpaca", 2023, (), ()))
    samples = select_content_review_sample(clusters)
    reviews = _reviews(samples)
    rare = next(item for item in samples if item.event_family == "earnings" and item.cluster_id == "rare")
    reviews = [replace(row, family_present=True, issuer_correct=True, announced_or_reported=True,
                       explicit_fiscal_period=True, action_supported=True, supporting_spans=((0, 15),))
               if row.sample_id == rare.sample_id else row for row in reviews]
    result = _evaluate(clusters, reviews)["families"]["earnings"]
    assert result["estimated_false_negative_clusters"] == 1.0
    assert result["recall_estimate"] == pytest.approx(300 / 301)
    assert result["recall_strata"]["alpaca/2023"]["census"]
    assert result["recall_strata"]["sec/2020"]["possible_misses"] == 0
    assert result["false_negative_simultaneous_upper_bound"] > 1.0
    assert result["recall_simultaneous_lower_bound"] < result["recall_estimate"]


def test_unresolved_negative_census_is_possible_miss_and_missing_candidate_fails() -> None:
    clusters = _population(200, 0, 10)
    samples = select_content_review_sample(clusters)
    reviews = [row for row in _reviews(samples) if row.reviewer_id == "first"]
    result = _evaluate(clusters, reviews)["families"]["earnings"]
    assert result["estimated_false_negative_clusters"] == 10
    assert result["false_negative_simultaneous_upper_bound"] == 10
    assert result["unresolved_candidate_clusters"] == 200
    assert result["recall_estimate"] == 0.0
    assert not result["qualified_for_historical_feature_use"]


def test_unrepresented_rule_variant_with_population_fifty_fails_even_when_family_passes() -> None:
    clusters = _population(1000, 0, 50)
    sampled = {item.cluster_id for item in select_content_review_sample(clusters)
               if item.event_family == "earnings" and item.role == "candidate"}
    omitted = {row.cluster_id for row in clusters if row.candidate_families and row.cluster_id not in sampled}
    rule_ids = set(sorted(omitted)[:50])
    clusters = [replace(row, rule_ids=row.rule_ids + (("earnings", "rare_rule"),))
                if row.cluster_id in rule_ids else row for row in clusters]
    result = _evaluate(clusters)["families"]["earnings"]
    rare = next(row for row in result["rule_variants"] if row["rule_id"] == "rare_rule")
    assert rare["population_clusters"] == 50
    assert rare["sample_clusters"] == 0
    assert rare["status"] == "blocked"
    assert "rule_variant_gate_failed" in result["reasons"]
    assert result["joint_precision_lower_bound"] >= 0.95


def test_rule_variant_below_fifty_is_diagnostic_only() -> None:
    clusters = _population()
    clusters[0] = replace(clusters[0], rule_ids=clusters[0].rule_ids + (("earnings", "rare"),))
    result = _evaluate(clusters)["families"]["earnings"]
    rare = next(row for row in result["rule_variants"] if row["rule_id"] == "rare")
    assert rare["status"] == "diagnostic_only"
    assert result["qualified_for_historical_feature_use"]


@pytest.mark.parametrize("reviewed,expected_status", [(9, "blocked"), (10, "blocked"), (11, "admitted")])
def test_rule_variant_minimum_sample_and_joint_lower_bound_are_both_required(reviewed: int, expected_status: str) -> None:
    clusters = _population(1000, 0, 0)
    sampled = {item.cluster_id for item in select_content_review_sample(clusters)
               if item.event_family == "earnings" and item.role == "candidate"}
    omitted = {row.cluster_id for row in clusters} - sampled
    variant = set(sorted(sampled)[:reviewed]) | set(sorted(omitted)[:50 - reviewed])
    clusters = [replace(row, rule_ids=row.rule_ids + (("earnings", "bounded_variant"),))
                if row.cluster_id in variant else row for row in clusters]
    result = _evaluate(clusters)["families"]["earnings"]
    measured = next(row for row in result["rule_variants"] if row["rule_id"] == "bounded_variant")
    assert measured["sample_clusters"] == reviewed
    assert measured["status"] == expected_status
    assert ("insufficient_rule_variant_sample" in measured["reasons"]) == (reviewed < 10)
    assert ("rule_variant_joint_lcb_below_threshold" in measured["reasons"]) == (reviewed < 11)


def test_unknown_and_disagreed_negative_labels_are_not_counted_as_true_negatives() -> None:
    clusters = _population(200, 0, 1)
    samples = select_content_review_sample(clusters)
    negative = next(item for item in samples if item.event_family == "earnings" and item.role == "noncandidate")
    reviews = [replace(row, family_present=None) if row.sample_id == negative.sample_id else row for row in _reviews(samples)]
    result = _evaluate(clusters, reviews)["families"]["earnings"]
    assert result["estimated_false_negative_clusters"] == 1
    reviews = [replace(row, family_present=row.reviewer_id == "first", supporting_spans=((0, 15),))
               if row.sample_id == negative.sample_id else row for row in _reviews(samples)]
    result = _evaluate(clusters, reviews)["families"]["earnings"]
    assert result["estimated_false_negative_clusters"] == 1
    assert result["qualified_for_historical_feature_use"]


@pytest.mark.parametrize("change,message", [
    ({"independent_review": False}, "independent source-only"),
    ({"evidence_scope": "market_outcomes"}, "independent source-only"),
    ({"family_present": 1}, "booleans"),
    ({"supporting_text_sha256": "bad"}, "text hash"),
    ({"reviewer_model": None}, "assistance"),
    ({"supporting_spans": ()}, "text evidence spans"),
    ({"supporting_spans": ((-1, 2),)}, "invalid review text spans"),
])
def test_review_evidence_poison_is_rejected(change: dict[str, Any], message: str) -> None:
    clusters = _population(1, 0, 0)
    samples = select_content_review_sample(clusters)
    reviews = _reviews(samples)
    index = next(index for index, row in enumerate(reviews) if row.joint)
    reviews[index] = replace(reviews[index], **change)
    with pytest.raises(DataReadinessError, match=message):
        _evaluate(clusters, reviews)


@pytest.mark.parametrize("poison", ["duplicate_reviewer", "different_text", "identity_change"])
def test_independent_review_pair_binds_stable_identity_and_same_text(poison: str) -> None:
    clusters = _population(2, 0, 0)
    samples = select_content_review_sample(clusters)
    reviews = _reviews(samples)
    if poison == "duplicate_reviewer":
        reviews[1] = replace(reviews[1], reviewer_id=reviews[0].reviewer_id)
    elif poison == "different_text":
        reviews[1] = replace(reviews[1], supporting_text_sha256="b" * 64)
    else:
        reviews[-1] = replace(reviews[-1], reviewer_model="changed-model")
    with pytest.raises(DataReadinessError):
        _evaluate(clusters, reviews)


def test_sampling_and_candidate_rule_declaration_poison_rejected() -> None:
    clusters = _population(1, 0, 0)
    samples = select_content_review_sample(clusters)
    poisoned = (replace(samples[0], inclusion_probability=0.5), *samples[1:])
    with pytest.raises(DataReadinessError, match="frozen population"):
        evaluate_content_reviews(clusters=clusters, samples=poisoned, reviews=[])
    with pytest.raises(DataReadinessError, match="rule variant"):
        select_content_review_sample([replace(clusters[0], rule_ids=())])
    with pytest.raises(DataReadinessError, match="duplicated"):
        select_content_review_sample(clusters + clusters)
