"""Exact saved FinBERT-to-source-version binding; research proxy evidence only.

Historical schema names identify immutable input snapshots. They are not accepted
by current canonical writers. No provider fetch, text inference or attribution is
performed here.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow.parquet as pq

from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research.news_corpus_enrichment import CorpusVersion, SentimentResolution
from market_predictor.research.news_sample_buckets import Sentiment
from market_predictor.resources import assert_memory_budget
from market_predictor.sentiment import build_sentiment_inputs

SCHEMA = "market_predictor.news_sentiment_reuse"
END = pd.Timestamp("2024-05-28T22:00:00Z")
BATCH_ROWS = 256
REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
CLOSED = {"training_eligible": False, "serving_eligible": False, "promotion_eligible": False,
          "company_attribution_established": False}
IMPLEMENTATION_PATHS = (
    "research/news_sentiment_reuse.py", "research/news_corpus_enrichment.py", "sentiment.py",
    "research/news_sample_buckets.py", "research/news_feature_enrichment.py", "catalysts/issuer_events/news_query_scope.py",
    "core/errors.py", "core/json_integrity.py", "core/system_memory.py", "evidence/io.py",
    "heavy_jobs.py", "locking.py", "resources.py", "process_memory.py",
)
EVENT_COLUMNS = (
    "event_id", "security_id", "ticker", "source_family", "raw_sha256", "title", "summary", "text",
    "published_at_utc", "available_at_utc", "provider_updated_at_utc", "availability_policy",
)
SCORE_COLUMNS = (
    "event_id", "security_id", "ticker", "source_family", "published_at_utc", "event_available_at_utc",
    "research_feature_available_at_utc", "inference_computed_at_utc", "sentiment_label",
    "sentiment_confidence", "sentiment_numeric", "sentiment_input_sha256", "sentiment_model",
    "sentiment_model_revision", "sentiment_input_mode", "sentiment_max_length", "sentiment_availability_policy",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _guard() -> None:
    assert_memory_budget(stage="saved sentiment binding", hard_budget_gib=5.0, headroom_gib=0.75)
    assert_system_memory_available(minimum_available_gib=2.0, maximum_used_percent=85.0)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for number, block in enumerate(iter(lambda: stream.read(1024**2), b"")):
            if number % 64 == 0:
                _guard()
            digest.update(block)
    return digest.hexdigest()


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _object(path: Path) -> dict[str, Any]:
    _require(path.stat().st_size <= 64 * 1024**2, "sentiment metadata exceeds bounded JSON size")
    return dict(parse_strict_json_object(path.read_bytes(), label=str(path)))


def _clock(value: Any) -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("sentiment clock malformed") from exc
    _require(not pd.isna(stamp) and stamp.tzinfo is not None, "sentiment clock missing or naive")
    return stamp.tz_convert("UTC")


def _version_clock(event: dict[str, Any]) -> pd.Timestamp:
    updated = event["provider_updated_at_utc"]
    # The original provider adapter leaves this nullable for never-updated news.
    # Only actual nulls use publication; malformed non-null values are rejected.
    return _clock(event["published_at_utc"] if updated is None or pd.isna(updated) else updated)


def _pin(root: Path, name: str, digest: str, files: dict[str, str]) -> Path:
    path = inside(root, name)
    key = path.relative_to(root).as_posix()
    _require(key not in files or files[key] == digest, "conflicting sentiment source pin")
    _require(_hash(path) == digest, f"sentiment input changed: {key}")
    files[key] = digest
    return path


def _capture(root: Path, path: Path, files: dict[str, str]) -> str:
    digest = _hash(path)
    key = path.relative_to(root).as_posix()
    _require(key not in files or files[key] == digest, "conflicting sentiment source capture")
    files[key] = digest
    return digest


def _recheck(root: Path, files: Mapping[str, str]) -> None:
    for name, digest in files.items():
        _require(_hash(inside(root, name)) == digest, f"sentiment bound bytes changed: {name}")


def _implementation(root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {(package / name).relative_to(root).as_posix(): _hash(package / name) for name in IMPLEMENTATION_PATHS}


@dataclass(frozen=True)
class SentimentArchive:
    name: str
    collection_manifest: SourcePin
    sentiment_manifest: SourcePin


def _request(root: Path, manifest_path: Path, manifest: dict[str, Any], files: dict[str, str]) -> dict[str, Any]:
    path = manifest_path.parent / "_request.json"
    _capture(root, path, files)
    request = _object(path)
    digest = request.pop("request_sha256", None)
    _require(digest == manifest["request_sha256"] and
             hashlib.sha256(_json(request).encode()).hexdigest() == digest, "original sentiment request hash differs")
    return request


def _artifacts(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in manifest["artifacts"]:
        key = str(item["chunk_id"])
        _require(key not in result, "duplicate source chunk")
        result[key] = item
    return result


def _artifact(root: Path, item: dict[str, Any], files: dict[str, str], inputs: dict[str, str]) -> Path:
    path = _pin(root, str(item["path"]), str(item["sha256"]), files)
    sidecar_path = inside(root, str(item["manifest_path"]))
    _require(sidecar_path == Path(str(path) + ".manifest.json"), "unexpected original sidecar location")
    _capture(root, sidecar_path, files)
    sidecar = _object(sidecar_path)
    _require(sidecar.get("schema") == "market_data.artifact_manifest.v1" and
             sidecar.get("artifact_sha256") == item["sha256"] and sidecar.get("rows") == item["rows"],
             "original sentiment child sidecar differs")
    _require(all(sidecar.get("inputs", {}).get(key) == value for key, value in inputs.items()),
             "original sentiment child input binding differs")
    _require(cast(Any, pq).ParquetFile(path).metadata.num_rows == item["rows"], "original sentiment child row count differs")
    return path


def _rows(path: Path, columns: Sequence[str]) -> Iterator[dict[str, Any]]:
    for batch in cast(Any, pq).ParquetFile(path).iter_batches(batch_size=BATCH_ROWS, columns=list(columns)):
        _guard()
        yield from batch.to_pylist()


def _binding(event: dict[str, Any], score: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    _require(all(event[key] == score[key] for key in ("event_id", "security_id", "ticker", "source_family")),
             "sentiment query identity differs from source event")
    _require(event["source_family"] == "alpaca" and event["availability_policy"] == "provider_publication_proxy",
             "unsupported original sentiment source semantics")
    for score_key, request_key in (("sentiment_model", "model_name"), ("sentiment_model_revision", "model_revision"),
                                   ("sentiment_input_mode", "text_mode"), ("sentiment_max_length", "max_length"),
                                   ("sentiment_availability_policy", "sentiment_availability_policy")):
        _require(score[score_key] == request[request_key], f"sentiment method differs: {score_key}")
    text = str(build_sentiment_inputs(pd.DataFrame([event]), mode=request["text_mode"]).iloc[0])
    _require(hashlib.sha256(text.encode("utf-8")).hexdigest() == score["sentiment_input_sha256"],
             "scored input hash differs from original source text")
    published, available = _clock(event["published_at_utc"]), _clock(event["available_at_utc"])
    _require(published == _clock(score["published_at_utc"]) and available == _clock(score["event_available_at_utc"]),
             "sentiment source clocks differ")
    score_time = _clock(score["research_feature_available_at_utc"])
    _require(published <= available and score_time == available + pd.Timedelta(minutes=request["fixed_latency_minutes"]),
             "sentiment processing availability differs")
    _require(_clock(score["inference_computed_at_utc"]) >= available, "sentiment computation precedes source")
    _require(type(score["sentiment_confidence"]) in {int, float} and type(score["sentiment_numeric"]) in {int, float},
             "missing or invalid saved sentiment confidence/numeric value")
    label, confidence, numeric = score["sentiment_label"], float(score["sentiment_confidence"]), float(score["sentiment_numeric"])
    _require(label in {"positive", "negative", "neutral"} and math.isfinite(confidence) and 0 <= confidence <= 1,
             "invalid saved sentiment confidence")
    expected = confidence if label == "positive" else -confidence if label == "negative" else 0.0
    _require(math.isfinite(numeric) and numeric == expected, "saved signed sentiment differs from label/confidence")
    result = {key: score[key] for key in SCORE_COLUMNS}
    for key in ("published_at_utc", "event_available_at_utc", "research_feature_available_at_utc", "inference_computed_at_utc"):
        result[key] = _clock(result[key]).isoformat()
    result["raw_sha256"] = event["raw_sha256"]
    updated = event["provider_updated_at_utc"]
    result["provider_updated_at_utc"] = None if updated is None or pd.isna(updated) else _clock(updated).isoformat()
    result["effective_version_available_at_utc"] = _version_clock(event).isoformat()
    return result


def _load_archive(root: Path, archive: SentimentArchive, db: sqlite3.Connection, files: dict[str, str],
                  progress: Callable[[dict[str, Any]], None] | None) -> int:
    collection_path = _pin(root, archive.collection_manifest.path, archive.collection_manifest.sha256, files)
    sentiment_path = _pin(root, archive.sentiment_manifest.path, archive.sentiment_manifest.sha256, files)
    collection, sentiment = _object(collection_path), _object(sentiment_path)
    _require(collection.get("schema") == "swing.alpaca_news_history_manifest.v1" and
             sentiment.get("schema") == "swing.event_sentiment_manifest.v1", "unexpected original snapshot schema")
    _request(root, collection_path, collection, files)
    request = _request(root, sentiment_path, sentiment, files)
    _require(request.get("collection_manifest_sha256") == archive.collection_manifest.sha256 and
             request.get("collection_request_sha256") == collection["request_sha256"], "sentiment collection binding differs")
    _require(request.get("model_name") == "ProsusAI/finbert" and request.get("model_revision") == REVISION and
             request.get("text_mode") == "title_summary" and request.get("max_length") == 128 and
             request.get("fixed_latency_minutes") == 5 and request.get("sentiment_availability_policy") ==
             "provider_publication_proxy_plus_fixed_inference_latency", "unexpected saved FinBERT method")
    events, scores = _artifacts(collection), _artifacts(sentiment)
    rows_so_far = int(db.execute("SELECT COUNT(*) FROM scores").fetchone()[0])
    empty_without_events = 0
    for position, (chunk, item) in enumerate(scores.items(), 1):
        if chunk not in events:
            _require(type(item.get("rows")) is int and item["rows"] == 0,
                     "nonempty sentiment chunk has no original source events")
            # Original collection manifests omit zero-event artifacts. Verify the
            # independently pinned empty score child; never fabricate an event.
            _artifact(root, item, files, {"chunk_id": chunk, "sentiment_request_sha256": sentiment["request_sha256"],
                                         "source_event_artifact_sha256": item["source_event_artifact_sha256"]})
            empty_without_events += 1
            if progress is not None:
                progress({"archive": archive.name, "chunk_id": chunk, "completed_chunks": position,
                          "total_chunks": len(scores), "rows_so_far": rows_so_far,
                          "disposition": "verified_empty_score_artifact_without_source_events"})
            continue
        original = events[chunk]
        if _clock(original["start_utc"]) > END:
            continue
        _require(item["source_event_artifact_sha256"] == original["sha256"], "sentiment source event hash differs")
        event_path = _artifact(root, original, files, {"chunk_id": chunk, "collection_request_sha256": collection["request_sha256"]})
        score_path = _artifact(root, item, files, {"chunk_id": chunk, "sentiment_request_sha256": sentiment["request_sha256"],
                                                "source_event_artifact_sha256": original["sha256"]})
        db.execute("DELETE FROM events")
        for event in _rows(event_path, EVENT_COLUMNS):
            # No future event text enters the retained index. Full child bytes remain pinned.
            if _clock(event["available_at_utc"]) > END or _version_clock(event) > END:
                continue
            updated = event["provider_updated_at_utc"]
            event["provider_updated_at_utc"] = None if updated is None or pd.isna(updated) else _clock(updated).isoformat()
            for key in ("published_at_utc", "available_at_utc"):
                event[key] = _clock(event[key]).isoformat()
            db.execute("INSERT INTO events VALUES (?, ?, 0)", (event["event_id"], _json(event)))
        for score in _rows(score_path, SCORE_COLUMNS):
            found = db.execute("SELECT payload, seen FROM events WHERE event_id=?", (score["event_id"],)).fetchone()
            if found is None:
                continue
            _require(found[1] == 0, "duplicate saved sentiment event")
            event = json.loads(found[0])
            bound = _binding(event, score, request)
            provenance = {"archive": archive.name, "chunk_id": chunk,
                          "source_event_artifact": {"path": event_path.relative_to(root).as_posix(), "sha256": original["sha256"]},
                          "sentiment_artifact": {"path": score_path.relative_to(root).as_posix(), "sha256": item["sha256"]}}
            db.execute("INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (archive.name, event["event_id"], event["security_id"], event["ticker"], event["raw_sha256"],
                        _json(bound), _json(provenance)))
            db.execute("UPDATE events SET seen=1 WHERE event_id=?", (score["event_id"],))
            rows_so_far += 1
        db.commit()
        if progress is not None:
            progress({"archive": archive.name, "chunk_id": chunk, "completed_chunks": position,
                      "total_chunks": len(scores), "rows_so_far": rows_so_far})
    return empty_without_events


def build_sentiment_reuse_index(*, root: Path, output: Path, sources: Sequence[SentimentArchive],
                                progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Publish a bounded, immutable initial-fit index; caller supplies original manifest pins."""
    root = root.resolve()
    output = inside(root, output)
    _require(bool(sources) and len({source.name for source in sources}) == len(sources), "invalid sentiment archives")
    stage = output.with_name(f".{output.name}.building")
    _require(not output.exists() and not stage.exists(), "sentiment index output or private stage already exists")
    runtime = heavy_job_runtime_dir()
    with heavy_job_lease("saved-news-sentiment-index", runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        _guard()
        implementation = _implementation(root)
        files: dict[str, str] = {}
        empty_by_archive: dict[str, int] = {}
        stage.mkdir(parents=True)
        with closing(sqlite3.connect(stage / "index.sqlite")) as db:
            db.execute("PRAGMA cache_size=-8192")
            db.execute("PRAGMA temp_store=FILE")
            db.execute("PRAGMA mmap_size=0")
            db.execute("CREATE TABLE events(event_id TEXT PRIMARY KEY,payload TEXT NOT NULL,seen INTEGER NOT NULL)")
            db.execute("CREATE TABLE scores(archive TEXT,event_id TEXT,security_id TEXT,ticker TEXT,"
                       "raw_sha256 TEXT,payload TEXT,origin TEXT)")
            db.execute("CREATE INDEX lookup ON scores(archive,event_id,security_id,ticker,raw_sha256)")
            for archive in sources:
                empty_by_archive[archive.name] = _load_archive(root, archive, db, files, progress)
            rows = int(db.execute("SELECT COUNT(*) FROM scores").fetchone()[0])
            db.execute("DROP TABLE events")
            db.commit()
        manifest = {"schema": SCHEMA, "rows": rows, "cutoff_utc": END.isoformat(),
                    "availability_semantics": "historical_proxy", "source_files": files,
                    "empty_sentiment_artifacts_without_source_events": {
                        "files": sum(empty_by_archive.values()), "scores": 0, "by_archive": empty_by_archive},
                    "implementation_files": implementation, "database_sha256": _hash(stage / "index.sqlite"),
                    "archives": [{"name": item.name, "collection_manifest": item.collection_manifest.model_dump(mode="json"),
                                  "sentiment_manifest": item.sentiment_manifest.model_dump(mode="json")} for item in sources], **CLOSED}
        _recheck(root, files)
        _recheck(root, implementation)
        _guard()
        write_json_object(stage / "_manifest.json", manifest)
        stage.rename(output)
        return {**manifest, "manifest_sha256": _hash(output / "_manifest.json")}


@dataclass
class SavedSentimentResolver:
    source_files: Mapping[str, str]
    implementation_files: Mapping[str, str]
    identity: Mapping[str, Any]
    _db: sqlite3.Connection

    def close(self) -> None:
        self._db.close()

    def resolve(self, record: CorpusVersion) -> SentimentResolution:
        if record.source_family != "alpaca":
            return SentimentResolution(status="missing_source_family")
        metadata = record.metadata
        source = metadata.get("source_metadata", {})
        key = (source.get("archive"), source.get("event_id"), source.get("query_security_id"),
               source.get("query_ticker"), metadata.get("source_version_sha256"))
        if any(not isinstance(value, str) or not value for value in key) or source.get("raw_sha256") != key[-1]:
            return SentimentResolution(status="missing_or_mismatched_source_identity")
        found = self._db.execute("SELECT payload,origin FROM scores WHERE archive=? AND event_id=? "
                                 "AND security_id=? AND ticker=? AND raw_sha256=?", key).fetchall()
        if not found:
            return SentimentResolution(status="missing_exact_source_version")
        values = [json.loads(row[0]) for row in found]
        if len({_json(value) for value in values}) != 1:
            return SentimentResolution(status="conflicting_saved_scores")
        value = values[0]
        try:
            if (_clock(metadata["published_at_utc"]) != _clock(value["published_at_utc"]) or
                    _clock(metadata["version_available_at_utc"]) != _clock(value["effective_version_available_at_utc"])):
                return SentimentResolution(status="mismatched_source_clock")
        except (KeyError, ValueError, TypeError):
            return SentimentResolution(status="missing_source_clock")
        if metadata.get("availability_semantics") != "historical_proxy":
            return SentimentResolution(status="unsupported_availability_semantics")
        return SentimentResolution(status="matched", score=float(value["sentiment_numeric"]),
                                   label=cast(Sentiment, value["sentiment_label"]),
                                   available_at_utc=cast(datetime, _clock(value["research_feature_available_at_utc"]).to_pydatetime()),
                                   binding={**value, "version_id": record.version_id,
                                            "origins": [json.loads(row[1]) for row in found],
                                            "availability_semantics": "historical_proxy", **CLOSED})


def open_sentiment_reuse_index(*, root: Path, index: SourcePin) -> SavedSentimentResolver:
    """Open read-only under the caller's lease; all index/source/code bytes are rechecked."""
    root = root.resolve()
    files: dict[str, str] = {}
    path = _pin(root, index.path, index.sha256, files)
    manifest = _object(path)
    _require(manifest.get("schema") == SCHEMA and manifest.get("cutoff_utc") == END.isoformat() and
             all(manifest.get(key) is value for key, value in CLOSED.items()), "invalid sentiment reuse manifest")
    _recheck(root, manifest["source_files"])
    files.update(manifest["source_files"])
    _require(manifest["implementation_files"] == _implementation(root), "sentiment executed implementation differs")
    database = path.parent / "index.sqlite"
    _pin(root, database.relative_to(root).as_posix(), manifest["database_sha256"], files)
    db = sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)
    db.execute("PRAGMA query_only=ON")
    db.execute("PRAGMA cache_size=-8192")
    db.execute("PRAGMA mmap_size=0")
    return SavedSentimentResolver(files, manifest["implementation_files"],
                                  {"index": index.model_dump(mode="json"), "availability_semantics": "historical_proxy", **CLOSED}, db)
