"""Synthetic UNIT inputs only; these tests do not sample or admit actual articles."""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import FrozenInstanceError, fields, replace
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

import pytest

from market_predictor.research.news_sample_buckets import (
    EnrichedNewsReference,
    NewsSampleBuckets,
    SampleCapacityError,
    SamplingLimits,
)


def _reference(index: int, **changes: Any) -> EnrichedNewsReference:
    return replace(
        EnrichedNewsReference(
            document_ref=f"unit://article/{index}",
            version_ref=f"unit://version/{index}/immutable",
            source="unit_news",
            published_at_utc=datetime(2023, 5, 2, 12, tzinfo=UTC),
            categories=("contract",),
            sentiment="neutral",
            confidence="high",
        ),
        **changes,
    )


def test_membership_and_counts_ignore_order_and_batch_boundaries() -> None:
    records = [_reference(index) for index in range(200)]
    expected = NewsSampleBuckets(SamplingLimits(samples_per_bucket=7))
    expected.add_many(records)
    shuffled = records.copy()
    random.Random(41).shuffle(shuffled)
    for ordered in (list(reversed(records)), shuffled):
        actual = NewsSampleBuckets(SamplingLimits(samples_per_bucket=7))
        for start in range(0, len(ordered), 13):
            actual.add_many(iter(ordered[start : start + 13]))
        assert actual.snapshot() == expected.snapshot()
    snapshot = expected.snapshot()
    assert snapshot.records_seen == 200
    assert snapshot.buckets[0].records_seen == 200
    assert len(snapshot.buckets[0].references) == 7


def test_rare_labels_mixed_sentiment_and_multiple_categories_are_retained() -> None:
    sampler = NewsSampleBuckets(SamplingLimits(samples_per_bucket=1))
    sampler.add_many(_reference(index) for index in range(50))
    rare = _reference(100, categories=("guidance", "legal"), sentiment="mixed", confidence="low")
    sampler.add(rare)
    buckets = sampler.snapshot().buckets
    assert len(buckets) == 4
    assert {item.key.category for item in buckets if item.key.kind == "category"} == {"contract", "guidance", "legal"}
    for item in buckets:
        if item.key.category != "contract":
            assert item.references == (rare,)
            assert item.key.sentiment == "mixed"
            assert item.key.confidence == "low"


@pytest.mark.parametrize(
    "categories, expected_multi_category",
    [
        (("guidance", "legal"), True),
        (("guidance",), False),
    ],
)
def test_multi_category_bucket_describes_labels_not_event_count(
    categories: tuple[str, ...],
    expected_multi_category: bool,
) -> None:
    sampler = NewsSampleBuckets()
    sampler.add(_reference(1, categories=categories))
    kinds = {bucket.key.kind for bucket in sampler.snapshot().buckets}
    assert ("multi_category" in kinds) is expected_multi_category
    assert kinds <= {"category", "multi_category"}
    assert "multi_event" not in kinds


def test_source_publication_year_and_confidence_diversity() -> None:
    sampler = NewsSampleBuckets(SamplingLimits(samples_per_bucket=1))
    for source in ("unit_news", "unit_sec"):
        for year in (2021, 2022, 2023):
            for confidence in ("low", "high"):
                sampler.add(
                    _reference(1, source=source, published_at_utc=datetime(year, 12, 31, 23, 59, tzinfo=UTC), confidence=confidence)
                )
    assert len(sampler.snapshot().buckets) == 12
    assert all(bucket.records_seen == 1 for bucket in sampler.snapshot().buckets)
    assert {bucket.key.publication_year for bucket in sampler.snapshot().buckets} == {2021, 2022, 2023}


def test_empty_categories_and_unavailable_states_are_explicit() -> None:
    sampler = NewsSampleBuckets()
    reference = _reference(1, categories=(), sentiment="unavailable", confidence="unavailable")
    sampler.add(reference)
    (bucket,) = sampler.snapshot().buckets
    assert bucket.key.kind == "other_unresolved"
    assert bucket.key.sentiment == bucket.key.confidence == "unavailable"
    assert bucket.references == (reference,)


def test_repeated_records_count_occurrences_not_unique_articles() -> None:
    sampler = NewsSampleBuckets()
    sampler.add_many([_reference(1)] * 8)
    snapshot = sampler.snapshot()
    assert snapshot.records_seen == snapshot.buckets[0].records_seen == 8
    assert len(snapshot.buckets[0].references) == 1
    # The same document with a different immutable version is a distinct record.
    sampler.add(_reference(1, version_ref="unit://version/1/revised"))
    assert sampler.snapshot().records_seen == 9
    assert len(sampler.snapshot().buckets[0].references) == 2


def test_category_order_is_not_sampling_evidence() -> None:
    left, right = NewsSampleBuckets(), NewsSampleBuckets()
    left.add(_reference(1, categories=("legal", "guidance")))
    right.add(_reference(1, categories=("guidance", "legal")))
    assert left.snapshot() == right.snapshot()


def test_bucket_limit_rejects_whole_multilabel_record_atomically() -> None:
    sampler = NewsSampleBuckets(SamplingLimits(max_buckets=3))
    sampler.add(_reference(1))
    before = sampler.snapshot()
    with pytest.raises(SampleCapacityError, match="max_buckets"):
        sampler.add(_reference(2, categories=("legal", "guidance")))
    assert sampler.snapshot() == before
    sampler.add(_reference(3))
    assert sampler.snapshot().records_seen == 2


def test_storage_budget_rejects_atomically_and_counts_all_bucket_copies() -> None:
    probe = NewsSampleBuckets()
    probe.add(_reference(1))
    exact_bytes = probe.snapshot().retained_metadata_bytes
    sampler = NewsSampleBuckets(SamplingLimits(max_retained_bytes=exact_bytes))
    sampler.add(_reference(1))
    assert sampler.snapshot() == probe.snapshot()
    with pytest.raises(SampleCapacityError, match="max_retained_bytes"):
        sampler.add(_reference(2, categories=("legal", "guidance")))
    assert sampler.snapshot() == probe.snapshot()
    too_small = NewsSampleBuckets(SamplingLimits(max_retained_bytes=exact_bytes - 1))
    with pytest.raises(SampleCapacityError, match="max_retained_bytes"):
        too_small.add(_reference(1))
    assert too_small.snapshot().records_seen == 0


@pytest.mark.parametrize(
    "timestamp",
    [
        datetime(2023, 1, 1),
        datetime(2023, 1, 1, tzinfo=timezone(timedelta(hours=1))),
        "2023-01-01T00:00:00Z",
        None,
    ],
)
def test_invalid_timestamps_are_not_guessed(timestamp: Any) -> None:
    with pytest.raises(ValueError, match="aware UTC"):
        _reference(1, published_at_utc=timestamp)


@pytest.mark.parametrize(
    "changes",
    [
        {"document_ref": ""},
        {"version_ref": " "},
        {"source": "x" * 65},
        {"categories": ["legal"]},
        {"categories": ("legal", "legal")},
        {"categories": tuple(str(index) for index in range(33))},
        {"sentiment": "bullish"},
        {"confidence": 0.99},
    ],
)
def test_invalid_or_unbounded_metadata_is_rejected(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        _reference(1, **changes)


def test_unicode_payload_byte_limit_is_checked_before_state_changes() -> None:
    sampler = NewsSampleBuckets()
    categories = tuple(f"{index}" + "\U0001f9ea" * 90 for index in range(32))
    with pytest.raises(ValueError, match="encoded bytes"):
        sampler.add(_reference(1, categories=categories))
    assert sampler.snapshot().records_seen == 0
    assert sampler.snapshot().buckets == ()


@pytest.mark.parametrize(
    "changes",
    [
        {"samples_per_bucket": 0},
        {"max_buckets": -1},
        {"max_retained_bytes": True},
    ],
)
def test_limits_must_be_explicit_positive_integers(changes: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        SamplingLimits(**changes)


def test_snapshot_is_immutable_and_does_not_follow_later_additions() -> None:
    sampler = NewsSampleBuckets()
    sampler.add(_reference(1))
    before = sampler.snapshot()
    with pytest.raises(FrozenInstanceError):
        before.buckets[0].references[0].source = "changed"  # type: ignore[misc]
    sampler.add(_reference(2))
    assert before.records_seen == 1
    assert sampler.snapshot().records_seen == 2


def test_rank_is_metadata_hash_without_future_outcome_or_model_fields() -> None:
    assert {field.name for field in fields(EnrichedNewsReference)} == {
        "document_ref",
        "version_ref",
        "source",
        "published_at_utc",
        "categories",
        "sentiment",
        "confidence",
    }
    records = [_reference(index) for index in range(30)]
    key = json.dumps(["category", "contract", "unit_news", 2023, "neutral", "high"], ensure_ascii=False, separators=(",", ":")).encode()
    ranks = []
    for record in records:
        payload = json.dumps(
            {
                "document_ref": record.document_ref,
                "version_ref": record.version_ref,
                "source": record.source,
                "published_at_utc": record.published_at_utc.isoformat(),
                "categories": list(record.categories),
                "sentiment": record.sentiment,
                "confidence": record.confidence,
            },
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode()
        ranks.append((hashlib.sha256(key + b"\x00" + payload).digest(), payload, record.version_ref))
    sampler = NewsSampleBuckets(SamplingLimits(samples_per_bucket=4))
    sampler.add_many(records)
    assert [record.version_ref for record in sampler.snapshot().buckets[0].references] == [item[2] for item in sorted(ranks)[:4]]
    with pytest.raises(TypeError, match="unexpected keyword"):
        EnrichedNewsReference(**{field.name: getattr(records[0], field.name) for field in fields(EnrichedNewsReference)}, future_return=9.0)  # type: ignore[call-arg]
