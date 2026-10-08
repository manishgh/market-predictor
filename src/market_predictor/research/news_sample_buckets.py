"""Bounded QA reference samples from an already-enriched news stream.

This helper neither enriches articles nor filters the caller's full corpus. Publication
year is a QA stratum, not proof of historical feature availability or source coverage.
References must resolve immutable document versions downstream. No text, embeddings,
prices, outcomes, models, or admission decisions are accepted.

Bottom-K membership is over distinct complete metadata records. Identical repeated
records consume one sample slot but increment records_seen on every occurrence.
Counts are therefore input records, never a claim about unique articles or versions.
The multi_category bucket means multiple upstream labels, not multiple events;
one event can have multiple labels and multiple events can share one label.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

Sentiment = Literal["positive", "negative", "neutral", "mixed", "unavailable"]
Confidence = Literal["high", "medium", "low", "unavailable"]
BucketKind = Literal["category", "multi_category", "other_unresolved"]

_MAX_CATEGORIES = 32
_MAX_RECORD_BYTES = 8192
_SENTIMENTS = frozenset(("positive", "negative", "neutral", "mixed", "unavailable"))
_CONFIDENCES = frozenset(("high", "medium", "low", "unavailable"))


def _bounded_text(value: str, field: str, limit: int) -> None:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > limit:
        raise ValueError(f"{field} must be nonempty, trimmed text of at most {limit} characters")


@dataclass(frozen=True, slots=True)
class EnrichedNewsReference:
    """Small metadata only; categories are upstream labels, not inferred here."""

    document_ref: str
    version_ref: str
    source: str
    published_at_utc: datetime
    categories: tuple[str, ...]
    sentiment: Sentiment
    confidence: Confidence

    def __post_init__(self) -> None:
        _bounded_text(self.document_ref, "document_ref", 512)
        _bounded_text(self.version_ref, "version_ref", 512)
        _bounded_text(self.source, "source", 64)
        if (
            not isinstance(self.published_at_utc, datetime)
            or self.published_at_utc.tzinfo is None
            or self.published_at_utc.utcoffset() != timedelta(0)
        ):
            raise ValueError("published_at_utc must be an aware UTC datetime")
        if not isinstance(self.categories, tuple) or len(self.categories) > _MAX_CATEGORIES:
            raise ValueError(f"categories must be a tuple with at most {_MAX_CATEGORIES} labels")
        for category in self.categories:
            _bounded_text(category, "category", 96)
        if len(set(self.categories)) != len(self.categories):
            raise ValueError("categories must not repeat labels")
        if self.sentiment not in _SENTIMENTS or self.confidence not in _CONFIDENCES:
            raise ValueError("unsupported sentiment or confidence state")


@dataclass(frozen=True, slots=True)
class SamplingLimits:
    """Hard capacity limits; overflow rejects a record without changing state.

    max_retained_bytes bounds serialized keys, record metadata and priority digests,
    counting a reference again for every bucket that retains it. Python container
    overhead is separately bounded by max_buckets * samples_per_bucket; the byte
    budget is not an operating-system RSS limit. Each input payload is capped at
    8 KiB and 32 category labels, bounding temporary work independently of corpus size.
    """

    samples_per_bucket: int = 3
    max_buckets: int = 16384
    max_retained_bytes: int = 32 * 1024 * 1024

    def __post_init__(self) -> None:
        for name, value in (
            ("samples_per_bucket", self.samples_per_bucket),
            ("max_buckets", self.max_buckets),
            ("max_retained_bytes", self.max_retained_bytes),
        ):
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True, order=True)
class BucketKey:
    kind: BucketKind
    category: str
    source: str
    publication_year: int
    sentiment: Sentiment
    confidence: Confidence


@dataclass(frozen=True, slots=True)
class BucketSample:
    key: BucketKey
    records_seen: int
    references: tuple[EnrichedNewsReference, ...]


@dataclass(frozen=True, slots=True)
class NewsSampleSnapshot:
    records_seen: int
    retained_metadata_bytes: int
    buckets: tuple[BucketSample, ...]


class SampleCapacityError(ValueError):
    """A declared storage limit would be exceeded; no partial record was added."""


@dataclass(frozen=True, slots=True)
class _Bucket:
    records_seen: int
    # Lexicographic digest/payload tie-breaking makes collisions deterministic.
    entries: tuple[tuple[bytes, bytes], ...]
    storage_bytes: int


def _encode(reference: EnrichedNewsReference) -> bytes:
    payload = json.dumps(
        {
            "document_ref": reference.document_ref,
            "version_ref": reference.version_ref,
            "source": reference.source,
            "published_at_utc": reference.published_at_utc.isoformat(),
            "categories": sorted(reference.categories),
            "sentiment": reference.sentiment,
            "confidence": reference.confidence,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(payload) > _MAX_RECORD_BYTES:
        raise ValueError(f"reference metadata exceeds {_MAX_RECORD_BYTES} encoded bytes")
    return payload


def _decode(payload: bytes) -> EnrichedNewsReference:
    record = json.loads(payload)
    return EnrichedNewsReference(
        document_ref=record["document_ref"],
        version_ref=record["version_ref"],
        source=record["source"],
        published_at_utc=datetime.fromisoformat(record["published_at_utc"]),
        categories=tuple(record["categories"]),
        sentiment=record["sentiment"],
        confidence=record["confidence"],
    )


def _key_bytes(key: BucketKey) -> bytes:
    return json.dumps(
        [key.kind, key.category, key.source, key.publication_year, key.sentiment, key.confidence],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")


def _keys(reference: EnrichedNewsReference) -> tuple[BucketKey, ...]:
    groups: list[tuple[BucketKind, str]] = [("category", label) for label in sorted(reference.categories)]
    if not groups:
        groups.append(("other_unresolved", ""))
    elif len(groups) > 1:
        groups.append(("multi_category", ""))
    return tuple(
        BucketKey(kind, label, reference.source, reference.published_at_utc.year, reference.sentiment, reference.confidence)
        for kind, label in groups
    )


class NewsSampleBuckets:
    """Streaming deterministic bottom-K samples, for QA only.

    add_many can consume a generator; no batch or corpus is retained. The same input
    multiset produces the same snapshot regardless of arrival order or batch splits
    when it fits the declared limits. Failures propagate to the caller; no records
    or new strata are silently discarded to keep a job running.

    Sample rank is a SHA-256 hash of the complete immutable metadata and bucket key,
    never an economic return, sentiment score, confidence score or model prediction.
    Mixed and unavailable states have their own strata, with no positivity threshold.
    """

    def __init__(self, limits: SamplingLimits = SamplingLimits()) -> None:
        self._limits = limits
        self._buckets: dict[BucketKey, _Bucket] = {}
        self._records_seen = 0
        self._retained_bytes = 0

    def add(self, reference: EnrichedNewsReference) -> None:
        payload = _encode(reference)
        keys = _keys(reference)
        new_buckets = sum(key not in self._buckets for key in keys)
        if len(self._buckets) + new_buckets > self._limits.max_buckets:
            raise SampleCapacityError("max_buckets exceeded")
        updates: dict[BucketKey, _Bucket] = {}
        total_bytes = self._retained_bytes
        for key in keys:
            encoded_key = _key_bytes(key)
            priority = hashlib.sha256(encoded_key + b"\x00" + payload).digest()
            old = self._buckets.get(key, _Bucket(0, (), 0))
            candidate = (priority, payload)
            entries = old.entries
            if candidate not in entries:
                entries = tuple(sorted((*entries, candidate))[: self._limits.samples_per_bucket])
            storage_bytes = len(encoded_key) + sum(len(digest) + len(record) for digest, record in entries)
            updates[key] = _Bucket(old.records_seen + 1, entries, storage_bytes)
            total_bytes += storage_bytes - old.storage_bytes
        if total_bytes > self._limits.max_retained_bytes:
            raise SampleCapacityError("max_retained_bytes exceeded")
        self._buckets.update(updates)
        self._records_seen += 1
        self._retained_bytes = total_bytes

    def add_many(self, references: Iterable[EnrichedNewsReference]) -> None:
        """Consume references sequentially; records before a failing record remain."""
        for reference in references:
            self.add(reference)

    def snapshot(self) -> NewsSampleSnapshot:
        """Return an immutable, sorted copy; snapshots also remain bounded by limits."""
        return NewsSampleSnapshot(
            records_seen=self._records_seen,
            retained_metadata_bytes=self._retained_bytes,
            buckets=tuple(
                BucketSample(key, bucket.records_seen, tuple(_decode(payload) for _, payload in bucket.entries))
                for key, bucket in sorted(self._buckets.items())
            ),
        )
