"""Streaming feature-only enrichment of every retained source version.

Checkpoint counts refer to stored version rows, not distinct articles or events.
Cue and sentiment clocks are separate. No source, training, or model admission is
granted. Only uncommitted .partial files in the owned private stage are replaced.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import stat
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.io import inside
from market_predictor.heavy_jobs import heavy_job_lease
from market_predictor.research.news_feature_enrichment import EnrichmentLimits, enrich_news_features
from market_predictor.research.news_sample_buckets import EnrichedNewsReference, NewsSampleBuckets, SamplingLimits, Sentiment
from market_predictor.resources import assert_memory_budget

_IMPLEMENTATION_PATHS = (
    "research/news_corpus_enrichment.py",
    "research/news_feature_enrichment.py",
    "research/news_sample_buckets.py",
    "catalysts/issuer_events/news_query_scope.py",
    "core/errors.py",
    "core/system_memory.py",
    "evidence/io.py",
    "heavy_jobs.py",
    "locking.py",
    "resources.py",
    "process_memory.py",
)
_MAX_JSON_LINE = 2 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class CorpusOptions:
    batch_size: int = 1024
    max_text_bytes: int = 16 * 1024 * 1024
    max_metadata_bytes: int = 1024 * 1024
    max_shards: int = 65536
    max_shard_bytes: int = 128 * 1024 * 1024
    enrichment_limits: EnrichmentLimits = EnrichmentLimits()
    sampling_limits: SamplingLimits = SamplingLimits()

    def __post_init__(self) -> None:
        for value in (self.batch_size, self.max_text_bytes, self.max_metadata_bytes, self.max_shards, self.max_shard_bytes):
            if type(value) is not int or value < 1:
                raise ValueError("corpus limits must be positive integers")
        if self.batch_size > 8192 or self.max_text_bytes > 64 * 1024 * 1024 or self.max_metadata_bytes > 4 * 1024 * 1024:
            raise ValueError("corpus per-record or batch limits exceed bounded capacity")


@dataclass(frozen=True, slots=True)
class CorpusVersion:
    version_id: str
    cluster_id: str
    security_id: str | None
    source_family: str
    year: int
    metadata: dict[str, Any]
    text: str | None


@dataclass(frozen=True, slots=True)
class SentimentResolution:
    status: str
    score: float | None = None
    label: Sentiment | None = None
    available_at_utc: datetime | None = None
    binding: dict[str, Any] = field(default_factory=dict)


class SentimentResolver(Protocol):
    @property
    def source_files(self) -> Mapping[str, str]: ...

    @property
    def implementation_files(self) -> Mapping[str, str]: ...

    @property
    def identity(self) -> Mapping[str, Any]: ...

    def resolve(self, record: CorpusVersion) -> SentimentResolution: ...


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _guard() -> None:
    assert_memory_budget(hard_budget_gib=5.0, headroom_gib=0.75, stage="news corpus enrichment")
    assert_system_memory_available(minimum_available_gib=2.0, maximum_used_percent=85.0)


def _hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def _encode(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=_json_default) + "\n"
    ).encode("utf-8")


def _no_links(path: Path) -> None:
    for part in (path, *path.parents):
        _require(not part.is_symlink(), "corpus path contains a symlink")
        if part.exists():
            attributes = getattr(part.lstat(), "st_file_attributes", 0)
            _require(not attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024), "corpus path contains a Windows reparse point")


def _check_files(root: Path, files: Mapping[str, str]) -> None:
    for relative, expected in files.items():
        path = inside(root, relative)
        _no_links(path)
        _require(path.is_file() and _hash(path) == expected, f"corpus pinned file changed: {relative}")


def _merge_files(*groups: Mapping[str, str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for group in groups:
        for path, digest in group.items():
            _require(path not in result or result[path] == digest, f"conflicting corpus pin: {path}")
            result[path] = digest
    return result


def _write_equal(path: Path, data: bytes) -> None:
    _no_links(path)
    if path.exists():
        _require(path.read_bytes() == data, f"immutable corpus artifact differs: {path.name}")
    else:
        temporary = path.with_name("." + path.name + ".partial")
        _no_links(temporary)
        with temporary.open("wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.rename(path)


def _clock(value: object, name: str) -> datetime:
    if not isinstance(value, str):
        raise ValueError(f"missing_{name}")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ValueError(f"invalid_{name}") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"naive_{name}")
    return parsed.astimezone(UTC)


def _sentiment(record: CorpusVersion, resolver: SentimentResolver | None, cue_available: datetime) -> dict[str, Any]:
    resolution = SentimentResolution("not_supplied") if resolver is None else resolver.resolve(record)
    _require(isinstance(resolution.status, str) and 0 < len(resolution.status) <= 96, "invalid sentiment resolution status")
    if resolution.status != "matched":
        _require(
            resolution.score is None and resolution.label is None and resolution.available_at_utc is None,
            "unmatched sentiment cannot expose a score, label, or clock",
        )
        return asdict(resolution)
    _require(
        resolution.score is not None and type(resolution.score) in (int, float) and math.isfinite(resolution.score),
        "matched sentiment requires a finite original score",
    )
    _require(
        resolution.label in ("positive", "negative", "neutral", "mixed", "unavailable"),
        "matched sentiment requires an explicit original label",
    )
    _require(
        isinstance(resolution.available_at_utc, datetime) and resolution.available_at_utc.tzinfo is not None,
        "matched sentiment requires its own availability clock",
    )
    _require(bool(resolution.binding), "matched sentiment requires explicit source/model binding evidence")
    assert resolution.available_at_utc is not None
    result = asdict(resolution)
    result["original_available_at_utc"] = resolution.available_at_utc
    result["available_at_utc"] = max(cue_available, resolution.available_at_utc.astimezone(UTC))
    return result


def _record(row: sqlite3.Row, options: CorpusOptions, resolver: SentimentResolver | None) -> dict[str, Any]:
    base: dict[str, Any] = {
        "version_id": row["version_id"],
        "cluster_id": row["cluster_id"],
        "retained_security_id": row["security_id"],
        "source_family": row["source_family"],
        "input_source_year": row["year"],
    }
    _require(isinstance(base["version_id"], str) and bool(base["version_id"]), "version_id must be nonempty text")
    if row["metadata_oversize"] or row["text_oversize"]:
        return {**base, "disposition": "unavailable_size_limit"}
    try:
        metadata = json.loads(row["content_json"])
        if not isinstance(metadata, dict):
            raise ValueError("content_json is not an object")
    except (ValueError, TypeError):
        return {**base, "disposition": "unavailable_metadata"}
    base["content_json_sha256"] = hashlib.sha256(row["content_json"].encode("utf-8")).hexdigest()
    base["availability_semantics"] = metadata.get("availability_semantics")
    base["source_clocks"] = {
        name: metadata.get(name)
        for name in (
            "published_at_utc",
            "version_available_at_utc",
            "event_available_at_utc",
            "identity_available_at_utc",
            "first_seen_at_utc",
        )
    }
    text = row["text"]
    if not isinstance(text, str) or not text.strip():
        return {**base, "disposition": "unavailable_text"}
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != metadata.get("text_sha256"):
        return {**base, "disposition": "unavailable_text_identity"}
    if metadata.get("source_family") != row["source_family"] or metadata.get("security_id") != row["security_id"]:
        return {**base, "disposition": "unavailable_record_identity"}
    try:
        published = _clock(metadata.get("published_at_utc"), "published_at_utc")
        clocks = [
            published,
            *[
                _clock(metadata.get(key), key)
                for key in (
                    "version_available_at_utc",
                    "event_available_at_utc",
                    "identity_available_at_utc",
                )
            ],
        ]
        if metadata.get("availability_semantics") == "observed":
            clocks.append(_clock(metadata.get("first_seen_at_utc"), "first_seen_at_utc"))
        elif metadata.get("availability_semantics") != "historical_proxy":
            raise ValueError("unknown_availability_semantics")
    except ValueError as error:
        return {**base, "disposition": "unavailable_clock", "reason": str(error)}
    source_id = metadata.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        return {**base, "disposition": "unavailable_reference"}
    available = max(clocks)
    try:
        enriched = enrich_news_features(
            document_ref=f"{row['source_family']}:{source_id}",
            version_ref=row["version_id"],
            source=row["source_family"],
            published_at_utc=published,
            available_at_utc=available,
            text=text,
            attribution_scope="unresolved",
            limits=options.enrichment_limits,
        )
    except ValueError as error:
        return {**base, "disposition": "unavailable_enrichment_input", "reason": str(error)}
    version = CorpusVersion(row["version_id"], row["cluster_id"], row["security_id"], row["source_family"], row["year"], metadata, text)
    sentiment = _sentiment(version, resolver, available)
    # Cue availability is never delayed by FinBERT processing. Its own result remains
    # separate and must be masked at its own available_at_utc by downstream features.
    return {
        **base,
        "disposition": "enriched",
        "enrichment": asdict(enriched),
        "sentiment": sentiment,
        "qa_reference": asdict(enriched.to_qa_reference()),
    }


def _account(record: dict[str, Any], counts: Counter[str], sampler: NewsSampleBuckets) -> None:
    counts["input_versions"] += 1
    counts[f"disposition:{record['disposition']}"] += 1
    if record["disposition"] == "enriched":
        enriched = record["enrichment"]
        counts["cue_occurrences"] += len(enriched["cues"])
        for category in enriched["categories"]:
            counts[f"category_versions:{category}"] += 1
        counts[f"sentiment_status:{record['sentiment']['status']}"] += 1
        counts[f"availability_semantics:{record['availability_semantics']}"] += 1
        raw = record["qa_reference"]
        sampler.add(
            EnrichedNewsReference(
                raw["document_ref"],
                raw["version_ref"],
                raw["source"],
                _clock(raw["published_at_utc"], "qa_publication"),
                tuple(raw["categories"]),
                raw["sentiment"],
                raw["confidence"],
            )
        )
    _require(len(counts) <= 4096, "corpus counter-key bound exceeded")


def _read_lines(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("rb") as stream:
        while line := stream.readline(_MAX_JSON_LINE + 1):
            _require(len(line) <= _MAX_JSON_LINE and line.endswith(b"\n"), "unbounded or partial corpus JSONL record")
            value = json.loads(line)
            _require(isinstance(value, dict), "corpus JSONL record is not an object")
            yield value


def _query_tail(last: str | None, batch_size: int) -> tuple[str, tuple[str | int, ...]]:
    if last is None:
        return "FROM versions ORDER BY version_id COLLATE BINARY LIMIT ?", (batch_size,)
    return "FROM versions WHERE version_id COLLATE BINARY>? ORDER BY version_id COLLATE BINARY LIMIT ?", (last, batch_size)


def _query(connection: sqlite3.Connection, last: str | None, options: CorpusOptions) -> sqlite3.Cursor:
    suffix, parameters = _query_tail(last, options.batch_size)
    return connection.execute(
        "SELECT version_id,cluster_id,security_id,source_family,year,"
        "CASE WHEN length(CAST(content_json AS BLOB))<=? THEN content_json END AS content_json,"
        "CASE WHEN length(CAST(text AS BLOB))<=? THEN text END AS text,"
        "length(CAST(content_json AS BLOB))>? AS metadata_oversize,"
        "length(CAST(text AS BLOB))>? AS text_oversize " + suffix,
        (options.max_metadata_bytes, options.max_text_bytes, options.max_metadata_bytes, options.max_text_bytes, *parameters),
    )


def enrich_news_corpus(
    *,
    root: Path,
    input_database: SourcePin,
    output: Path,
    options: CorpusOptions = CorpusOptions(),
    sentiment_resolver: SentimentResolver | None = None,
    additional_source_files: Mapping[str, str] | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Run under the sole heavy-job lease; resume only verified contiguous shards.

    A resolver must bind original query identity and exact source-event/raw-content
    hashes, publication, model revision and model input mode. Its source/index/code
    pins and declared identity form the immutable request; no score-only ID join is
    a valid adapter. A None resolver leaves sentiment explicitly not_supplied.
    """
    root = root.resolve()
    destination = inside(root, output)
    stage = destination.with_name("." + destination.name + ".private_stage")
    _no_links(destination)
    _no_links(stage)
    database = inside(root, input_database.path)
    with heavy_job_lease("news-corpus-enrichment", runtime_dir=root / "data/runtime"):
        _guard()
        _require(
            not Path(str(database) + "-wal").exists() and not Path(str(database) + "-journal").exists(),
            "source database must be a closed immutable SQLite file",
        )
        implementations = {"src/market_predictor/" + path: _hash(root / "src/market_predictor" / path) for path in _IMPLEMENTATION_PATHS}
        if sentiment_resolver is not None:
            _require(
                bool(sentiment_resolver.source_files) and bool(sentiment_resolver.implementation_files),
                "sentiment resolver must declare pinned source and implementation files",
            )
        files = _merge_files(
            {input_database.path: input_database.sha256},
            additional_source_files or {},
            implementations,
            {} if sentiment_resolver is None else sentiment_resolver.source_files,
            {} if sentiment_resolver is None else sentiment_resolver.implementation_files,
        )
        for relative in files:
            _require(
                not inside(root, relative).is_relative_to(destination) and not inside(root, relative).is_relative_to(stage),
                "corpus output cannot contain an input pin",
            )
        _check_files(root, files)
        request = {
            "purpose": "source_text_rule_cues_only",
            "input_database": input_database.model_dump(mode="json"),
            "options": asdict(options),
            "source_files": files,
            "implementation_files": implementations,
            "sentiment_identity": None if sentiment_resolver is None else dict(sentiment_resolver.identity),
            "training_eligible": False,
            "serving_eligible": False,
            "promotion_eligible": False,
        }
        request_bytes = _encode(request)
        request_sha = hashlib.sha256(request_bytes).hexdigest()
        _require(not (destination.exists() and stage.exists()), "both published output and private stage exist")
        if destination.exists():
            _require(
                (destination / "_manifest.json").is_file() and (destination / "_request.json").is_file(),
                "existing published output is incomplete",
            )
        work = destination if destination.exists() else stage
        work.mkdir(parents=True, exist_ok=True)
        _write_equal(work / "_request.json", request_bytes)
        sampler = NewsSampleBuckets(options.sampling_limits)
        counts: Counter[str] = Counter()
        artifacts: dict[str, str] = {"_request.json": request_sha}
        last: str | None = None
        sequence = 0
        with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as connection:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA cache_size=-2048")
            connection.execute("PRAGMA temp_store=FILE")
            connection.execute("BEGIN")
            columns = {row["name"]: row for row in connection.execute("PRAGMA table_info(versions)")}
            _require(
                {"version_id", "cluster_id", "security_id", "source_family", "year", "content_json", "text"} <= columns.keys(),
                "source versions schema is incomplete",
            )
            _require(columns["version_id"]["pk"] == 1, "version_id must be the primary key")
            total = connection.execute("SELECT count(*) FROM versions").fetchone()[0]
            while True:
                _guard()
                checkpoint = work / f"checkpoint-{sequence:06d}.json"
                shard = work / f"part-{sequence:06d}.jsonl"
                if checkpoint.exists():
                    cursor = _query(connection, last, options)
                    _no_links(checkpoint)
                    _no_links(shard)
                    _require(checkpoint.stat().st_size <= _MAX_JSON_LINE, "checkpoint exceeds metadata bound")
                    saved = json.loads(checkpoint.read_bytes())
                    _require(
                        saved["request_sha256"] == request_sha and saved["sequence"] == sequence, "checkpoint request or sequence differs"
                    )
                    _require(shard.is_file() and _hash(shard) == saved["sha256"], "completed shard hash differs")
                    shard_counts: Counter[str] = Counter()
                    start, previous, rows = last, last, 0
                    for record in _read_lines(shard):
                        source_row = cursor.fetchone()
                        _require(
                            source_row is not None and source_row["version_id"] == record["version_id"],
                            "completed shard does not match contiguous source versions",
                        )
                        assert source_row is not None
                        if rows % 32 == 0:
                            _guard()
                        replayed = json.loads(_encode(_record(source_row, options, sentiment_resolver)))
                        _require(_encode(record) == _encode(replayed), "completed shard content differs from source replay")
                        previous = record["version_id"]
                        _account(record, shard_counts, sampler)
                        rows += 1
                    _require(cursor.fetchone() is None and rows > 0, "completed shard has missing source versions")
                    _require(
                        saved
                        == {
                            "request_sha256": request_sha,
                            "sequence": sequence,
                            "after_version_id": start,
                            "last_version_id": previous,
                            "rows": rows,
                            "sha256": _hash(shard),
                            "counts": dict(shard_counts),
                        },
                        "checkpoint counters or boundaries differ",
                    )
                    counts.update(shard_counts)
                    last = previous
                else:
                    cursor = _query(connection, last, options)
                    first = cursor.fetchone()
                    if first is None:
                        break
                    _require(work == stage, "completed publication is missing a checkpoint")
                    _require(sequence < options.max_shards, "corpus shard limit exceeded")
                    partial = work / f".part-{sequence:06d}.partial"
                    _no_links(partial)
                    shard_counts = Counter()
                    start = last
                    rows = 0
                    shard_bytes = 0
                    with partial.open("wb") as stream:
                        row = first
                        while row is not None:
                            if rows % 32 == 0:
                                _guard()
                            encoded = _encode(_record(row, options, sentiment_resolver))
                            _require(len(encoded) <= _MAX_JSON_LINE, "enriched record exceeds bounded JSONL size")
                            # Account from exact serialized output: fresh and resumed paths
                            # use identical clock representations and QA inputs.
                            record = json.loads(encoded)
                            _account(record, shard_counts, sampler)
                            shard_bytes += len(encoded)
                            _require(shard_bytes <= options.max_shard_bytes, "corpus shard byte limit exceeded")
                            stream.write(encoded)
                            last = row["version_id"]
                            rows += 1
                            row = cursor.fetchone()
                        stream.flush()
                        os.fsync(stream.fileno())
                    digest = _hash(partial)
                    if shard.exists():
                        _require(_hash(shard) == digest, "uncheckpointed shard differs from deterministic replay")
                        partial.unlink()
                    else:
                        partial.rename(shard)
                    saved = {
                        "request_sha256": request_sha,
                        "sequence": sequence,
                        "after_version_id": start,
                        "last_version_id": last,
                        "rows": rows,
                        "sha256": digest,
                        "counts": dict(shard_counts),
                    }
                    _write_equal(checkpoint, _encode(saved))
                    counts.update(shard_counts)
                artifacts[shard.name] = _hash(shard)
                artifacts[checkpoint.name] = _hash(checkpoint)
                if progress is not None:
                    progress(
                        {
                            "sequence": sequence,
                            "processed_versions": counts["input_versions"],
                            "total_versions": total,
                            "counts": dict(counts),
                        }
                    )
                sequence += 1
            _require(counts["input_versions"] == total, "not every source version is accounted for")
        allowed = set(artifacts) | {"qa_samples.json", "_manifest.json", ".qa_samples.json.partial", "._manifest.json.partial"}
        _require({path.name for path in work.iterdir()} <= allowed, "unverified or partial corpus artifacts remain")
        qa = _encode(asdict(sampler.snapshot()))
        _write_equal(work / "qa_samples.json", qa)
        artifacts["qa_samples.json"] = hashlib.sha256(qa).hexdigest()
        _guard()
        _check_files(root, files)
        for name, expected in artifacts.items():
            artifact = work / name
            _no_links(artifact)
            _require(artifact.is_file() and _hash(artifact) == expected, "output artifact changed before publication: " + name)
        manifest = {
            "status": "complete_feature_cues_only",
            "request_sha256": request_sha,
            "source_versions": total,
            "shards": sequence,
            "counts": dict(counts),
            "qa_scope": "enriched_version_records_only",
            "artifacts": artifacts,
            "classification_precision_measured": False,
            "stock_attribution_verified": False,
            "finbert_executed": False,
            "embeddings_executed": False,
            "training_eligible": False,
            "serving_eligible": False,
            "promotion_eligible": False,
        }
        _write_equal(work / "_manifest.json", _encode(manifest))
        _require({path.name for path in work.iterdir()} == set(artifacts) | {"_manifest.json"}, "corpus final inventory differs")
        if work == stage:
            stage.rename(destination)
        return manifest
