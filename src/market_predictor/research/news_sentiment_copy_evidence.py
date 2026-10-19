"""Retain exact saved scoring identities for conflicting source-document copies.

This joins two completed publications, never provider archives or model inference.
The evidence is provenance for a later policy; it does not resolve disagreements.
Consumers recheck the compact evidence and executed code, not original raw inputs.
"""
from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research import news_runtime_memory as runtime
from market_predictor.research import news_source_index_inputs as source_inputs
from market_predictor.research import news_source_links as source_owner
from market_predictor.research.news_corpus_enrichment import _no_links
from market_predictor.research.news_decision_features import NewsVersion
from market_predictor.swing.datasets.return_relationship_integrity import pins, read_object

SCHEMA = "market_predictor.news_sentiment_copy_evidence"
CLOSED = {"training_eligible": False, "serving_eligible": False, "promotion_eligible": False,
          "source_coverage_admitted": False, "world_actor_verified": False, "inference_executed": False}
MODEL = "ProsusAI/finbert"
REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
AVAILABILITY_POLICY = "provider_publication_proxy_plus_fixed_inference_latency"
MAX_RECORD_BYTES = 2 * 1024**2
IMPLEMENTATION_PATHS = (
    "research/news_sentiment_copy_evidence.py", "research/news_runtime_memory.py",
    "research/news_decision_features.py", "research/news_corpus_enrichment.py",
    "research/news_source_index_inputs.py",
    "catalysts/issuer_events/news_query_scope.py", "swing/datasets/return_relationship_integrity.py",
    "core/errors.py", "core/json_integrity.py", "core/system_memory.py", "evidence/io.py", "evidence/hashing.py",
    "resources.py", "process_memory.py", "heavy_jobs.py", "locking.py",
)


@dataclass(frozen=True, slots=True)
class SentimentCopyIdentity:
    source_copy_id: str
    source_family: str
    source_document_id: str
    score: float
    score_available_at_utc: datetime
    source_version_sha256: str
    input_sha256: str
    model: str
    model_revision: str
    input_mode: str
    max_length: int
    availability_policy: str
    label: str
    binding_sha256: str
    evidence_ref: str


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _json(value: Any) -> str:
    def encode(item: Any) -> str:
        if isinstance(item, datetime):
            return item.isoformat()
        raise TypeError(f"unsupported evidence JSON type: {type(item).__name__}")

    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False,
                      default=encode)


def _clock(value: Any) -> datetime:
    _require(isinstance(value, str), "copy evidence requires an explicit timestamp")
    stamp = datetime.fromisoformat(value)
    _require(stamp.tzinfo is not None and stamp.utcoffset() is not None, "copy evidence clock is naive")
    return stamp.astimezone(UTC)


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _implementation(root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {(package / name).relative_to(root).as_posix(): runtime.file_sha256(package / name) for name in IMPLEMENTATION_PATHS}


def _connect(path: Path, *, readonly: bool = False) -> sqlite3.Connection:
    db = sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) if readonly else sqlite3.connect(path)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA cache_size=-2048")
    db.execute("PRAGMA mmap_size=0")
    db.execute("PRAGMA temp_store=FILE")
    if readonly:
        db.execute("PRAGMA query_only=ON")
    return db


def _lines(path: Path) -> Iterator[tuple[int, dict[str, Any], str]]:
    with path.open("rb") as stream:
        number = 0
        while raw := stream.readline(MAX_RECORD_BYTES + 1):
            _require(len(raw) <= MAX_RECORD_BYTES and raw.endswith(b"\n"), "corpus record exceeds bound or lacks terminator")
            number += 1
            if number % 128 == 1:
                runtime.guard("sentiment copy corpus stream")
            record = json.loads(raw)
            _require(isinstance(record, dict), "corpus record is not an object")
            yield number, record, hashlib.sha256(raw).hexdigest()


def _corpus(root: Path, pin: SourcePin, files: dict[str, str]) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path = runtime.pin_file(root, pin.path, pin.sha256, files)
    manifest = read_object(path, pin.sha256)
    _require(path.name == "_manifest.json" and manifest.get("status") == "complete_feature_cues_only"
             and all(manifest.get(key) is False for key in ("training_eligible", "serving_eligible", "promotion_eligible",
                 "classification_precision_measured", "stock_attribution_verified", "finbert_executed", "embeddings_executed")),
             "copy evidence requires the completed research-only corpus")
    artifacts = manifest["artifacts"]
    _require(isinstance(artifacts, dict) and artifacts.get("_request.json") == manifest["request_sha256"],
             "corpus artifact/request binding differs")
    _require({item.name for item in path.parent.iterdir()} == set(artifacts) | {"_manifest.json"}, "corpus inventory differs")
    for name, digest in artifacts.items():
        _require(Path(name).name == name, "corpus artifact must be a direct child")
        artifact = inside(path.parent, name)
        _no_links(artifact)
        runtime.pin_file(root, artifact.relative_to(root).as_posix(), digest, files)
    request = read_object(path.parent / "_request.json", manifest["request_sha256"])
    _require(all(request.get(key) is False for key in ("training_eligible", "serving_eligible", "promotion_eligible")),
             "corpus request grants admission")
    return path.parent, manifest, request


def _candidates(source: sqlite3.Connection, output: sqlite3.Connection) -> Counter[str]:
    output.execute("CREATE TABLE copies(copy_id TEXT PRIMARY KEY,version_id TEXT,version_json TEXT,status TEXT,payload TEXT)")
    output.execute("CREATE INDEX copy_version ON copies(version_id)")
    output.execute("CREATE TABLE seen_versions(version_id TEXT PRIMARY KEY)")
    counts: Counter[str] = Counter()
    def check_memory() -> int:
        runtime.guard("sentiment copy conflict scan")
        return 0

    source.set_progress_handler(check_memory, 100000)
    try:
        groups = source.execute("""SELECT source,document,json_extract(payload,'$.source_version_sha256') AS digest
            FROM versions WHERE payload IS NOT NULL GROUP BY source,document,digest
            HAVING COUNT(DISTINCT json_extract(payload,'$.sentiment_score'))>1 ORDER BY source,document,digest""")
        for group in groups:
            runtime.guard("sentiment copy group")
            counts["conflicting_exact_version_groups"] += 1
            for row in source.execute("SELECT copy_id,version_id,payload FROM versions WHERE source=? AND document=?",
                                      (group["source"], group["document"])):
                if row["payload"] is None:
                    continue
                version = source_owner._decode_version(row["payload"])
                if version.source_version_sha256 != group["digest"]:
                    continue
                _require(version.version_id == row["copy_id"] and version.source_family == group["source"]
                         and version.source_document_id == group["document"], "source copy ownership differs")
                output.execute("INSERT INTO copies VALUES(?,?,?,'unseen_corpus_version',NULL)",
                               (row["copy_id"], row["version_id"], row["payload"]))
                counts["candidate_copy_records"] += 1
    finally:
        source.set_progress_handler(None, 0)
    return counts


def _bound_identity(version: NewsVersion, population_id: str, record: dict[str, Any],
                    proof: dict[str, Any]) -> tuple[str, dict[str, Any] | None]:
    if version.sentiment_score is None:
        return "missing_saved_copy_score", None
    if record.get("disposition") != "enriched" or record.get("sentiment", {}).get("status") != "matched":
        return "missing_corpus_sentiment", None
    sentiment, enrichment = record["sentiment"], record["enrichment"]
    _require(record["version_id"] == population_id and record.get("source_family") == version.source_family
             and enrichment.get("version_ref") == population_id
             and enrichment.get("document_ref") == f"{version.source_family}:{version.source_document_id}",
             "corpus source/document/copy identity differs")
    binding = sentiment.get("binding")
    required = {"version_id", "raw_sha256", "source_family", "sentiment_input_sha256", "sentiment_model",
        "sentiment_model_revision", "sentiment_input_mode", "sentiment_max_length", "sentiment_availability_policy",
        "sentiment_label", "sentiment_numeric", "sentiment_confidence", "research_feature_available_at_utc", "origins"}
    if not isinstance(binding, dict) or not required.issubset(binding):
        return "missing_scoring_identity", None
    _require(binding["version_id"] == population_id and binding["raw_sha256"] == version.source_version_sha256
             and binding["source_family"] == version.source_family, "saved scoring source version differs")
    _require(type(sentiment.get("score")) in (int, float) and math.isfinite(sentiment["score"])
             and type(binding["sentiment_numeric"]) in (int, float)
             and sentiment["score"] == binding["sentiment_numeric"] == version.sentiment_score,
             "saved copy score differs from corpus binding")
    available = max(version.known_at_utc, _clock(sentiment["available_at_utc"]))
    original_available = _clock(binding["research_feature_available_at_utc"])
    _require(available == version.sentiment_available_at_utc
             and _clock(sentiment["original_available_at_utc"]) == original_available
             and original_available <= available, "saved copy score clock differs")
    _require(_sha(binding["sentiment_input_sha256"]) and isinstance(binding["origins"], list) and bool(binding["origins"]),
             "scoring input or origin proof missing")
    for origin in binding["origins"]:
        for name in ("source_event_artifact", "sentiment_artifact"):
            SourcePin.model_validate(origin[name])
    _require(all(binding.get(key) is False for key in ("training_eligible", "serving_eligible", "promotion_eligible")),
             "saved scoring binding grants admission")
    if (binding["sentiment_model"] != MODEL or binding["sentiment_model_revision"] != REVISION
            or binding["sentiment_input_mode"] != "title_summary" or type(binding["sentiment_max_length"]) is not int
            or binding["sentiment_max_length"] != 128 or binding["sentiment_availability_policy"] != AVAILABILITY_POLICY
            or binding["sentiment_label"] not in ("positive", "negative", "neutral")
            or version.source_family != "alpaca"):
        return "unsupported_scoring_identity", None
    confidence, label = binding["sentiment_confidence"], binding["sentiment_label"]
    _require(type(confidence) in (int, float) and math.isfinite(confidence) and 0 <= confidence <= 1
             and sentiment["label"] == label, "saved sentiment label/confidence differs")
    expected = confidence if label == "positive" else -confidence if label == "negative" else 0.0
    _require(expected == version.sentiment_score, "saved sentiment score is not original signed confidence")
    identity = SentimentCopyIdentity(version.version_id, version.source_family, version.source_document_id,
        float(version.sentiment_score), available, version.source_version_sha256, binding["sentiment_input_sha256"],
        binding["sentiment_model"], binding["sentiment_model_revision"], binding["sentiment_input_mode"],
        binding["sentiment_max_length"], binding["sentiment_availability_policy"], label,
        hashlib.sha256(_json(binding).encode()).hexdigest(), "")
    return "verified_scoring_identity", {"identity": asdict(identity), "original_binding": binding,
        "original_saved_score_available_at_utc": original_available.isoformat(), "corpus_proof": proof}


def build_news_sentiment_copy_evidence(*, root: Path, source_index: SourcePin, output: Path,
                                      progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Create a fresh immutable evidence index under one resource lease; never resume."""
    root = root.resolve()
    output = inside(root, output)
    _require(output.is_relative_to(root / "data/research"), "copy evidence output must be below data/research")
    stage = output.with_name(output.name + ".partial")
    _no_links(output)
    _no_links(stage)
    runtime_dir = heavy_job_runtime_dir()
    runtime_dir = runtime_dir if runtime_dir.is_absolute() else root / runtime_dir
    with heavy_job_lease("news-sentiment-copy-evidence", runtime_dir=runtime_dir):
        runtime.guard("sentiment copy evidence")
        _require(not output.exists() and not stage.exists(), "copy evidence output or stage already exists")
        with source_inputs.open_news_source_link_index(root=root, index=source_index) as source:
            source_manifest = read_object(inside(root, source_index.path), source_index.sha256)
            source_request = read_object(inside(root, source_index.path).parent / "_request.json", source_manifest["request_sha256"])
            corpus_pin = SourcePin.model_validate(source_request["policy"]["corpus_manifest"])
            files = dict(source.source_files)
            directory, corpus, corpus_request = _corpus(root, corpus_pin, files)
            implementations = pins(root, source.implementation_files, _implementation(root))
            runtime.recheck_files(root, implementations)
            request = {"schema": SCHEMA + "_request", "source_index": source_index.model_dump(mode="json"),
                "corpus": corpus_pin.model_dump(mode="json"), "source_files": files,
                "implementation_files": implementations, "purpose": "saved_conflicting_copy_scoring_identity_only", **CLOSED}
            stage.mkdir(parents=True)
            write_json_object(stage / "_request.json", request)
            request_sha = runtime.file_sha256(stage / "_request.json")
            with closing(_connect(stage / "index.sqlite")) as db:
                counts = _candidates(source._db, db)
                processed = 0
                shards = sorted(name for name in corpus["artifacts"] if name.startswith("part-") and name.endswith(".jsonl"))
                _require(len(shards) == corpus["shards"], "corpus shard count differs")
                for sequence, name in enumerate(shards, 1):
                    for line, record, record_sha in _lines(directory / name):
                        version_id = record["version_id"]
                        db.execute("INSERT INTO seen_versions VALUES(?)", (version_id,))
                        for copy in db.execute("SELECT copy_id,version_json FROM copies WHERE version_id=?", (version_id,)):
                            version = source_owner._decode_version(copy["version_json"])
                            status, payload = _bound_identity(version, version_id, record, {
                                "path": (directory / name).relative_to(root).as_posix(),
                                "sha256": corpus["artifacts"][name], "line": line, "record_sha256": record_sha})
                            db.execute("UPDATE copies SET status=?,payload=? WHERE copy_id=?",
                                       (status, None if payload is None else _json(payload), copy["copy_id"]))
                            counts[status] += 1
                        processed += 1
                    if progress is not None:
                        progress({"completed_shards": sequence, "total_shards": len(shards),
                                  "processed_versions": processed, "total_versions": corpus["source_versions"], "counts": dict(counts)})
                _require(processed == corpus["source_versions"], "corpus version count differs")
                _require(db.execute("SELECT COUNT(*) FROM copies WHERE status='unseen_corpus_version'").fetchone()[0] == 0,
                         "source copy population version is absent from corpus")
                counts["corpus_versions_checked"] = processed
                db.execute("DROP TABLE seen_versions")
                db.commit()
                runtime.guard("sentiment copy evidence compact output")
                db.execute("VACUUM")
            artifacts = {name: runtime.file_sha256(stage / name) for name in ("_request.json", "index.sqlite")}
            runtime.recheck_files(root, files)
            runtime.recheck_files(root, implementations)
        # The completed source context must exit successfully before any manifest exists.
        runtime.recheck_files(root, files)
        runtime.recheck_files(root, implementations)
        for name, digest in artifacts.items():
            _require(runtime.file_sha256(stage / name) == digest, "copy evidence stage changed")
        manifest = {"schema": SCHEMA, "status": "complete_copy_identity_evidence_only", "request_sha256": request_sha,
            "source_index": source_index.model_dump(mode="json"), "corpus": corpus_pin.model_dump(mode="json"),
            "artifacts": artifacts, "counts": dict(counts), "source_files": files, "implementation_files": implementations,
            "producer_source_provenance": {"source_index": dict(source.producer_source_provenance),
                "corpus": corpus_request.get("source_files", {})}, **CLOSED}
        runtime.guard("sentiment copy evidence publication")
        write_json_object(stage / "_manifest.json", manifest)
        _require({item.name for item in stage.iterdir()} == {"_manifest.json", "_request.json", "index.sqlite"},
                 "copy evidence stage inventory differs")
        stage.rename(output)
        return manifest


@dataclass
class SentimentCopyEvidenceReader:
    source_files: Mapping[str, str]
    implementation_files: Mapping[str, str]
    producer_source_provenance: Mapping[str, Any]
    evidence: SourcePin
    _db: sqlite3.Connection
    raw_sources_rechecked: bool = False

    def identity(self, version: NewsVersion) -> SentimentCopyIdentity | None:
        row = self._db.execute("SELECT version_json,status,payload FROM copies WHERE copy_id=?", (version.version_id,)).fetchone()
        if row is None:
            return None
        original = source_owner._decode_version(row["version_json"])
        for name in ("source_family", "source_document_id", "source_version_sha256", "source_proof_sha256",
                     "known_at_utc", "cue_available_at_utc", "copy_proof_sha256", "copy_proof_available_at_utc",
                     "sentiment_score", "sentiment_available_at_utc"):
            _require(getattr(version, name) == getattr(original, name), "requested copy differs from pinned score/source/clock")
        if row["status"] != "verified_scoring_identity":
            return None
        payload = json.loads(row["payload"])
        value = payload["identity"]
        _require(hashlib.sha256(_json(payload["original_binding"]).encode()).hexdigest() == value["binding_sha256"],
                 "copy scoring binding hash differs")
        value["score_available_at_utc"] = _clock(value["score_available_at_utc"])
        value["evidence_ref"] = f"{self.evidence.path}#{self.evidence.sha256}:{version.version_id}"
        return SentimentCopyIdentity(**value)


@contextmanager
def open_news_sentiment_copy_evidence(*, root: Path, evidence: SourcePin,
                                     source_index: SourcePin) -> Iterator[SentimentCopyEvidenceReader]:
    """Consume compact evidence under caller's lease, retaining raw provenance only."""
    root = root.resolve()
    runtime.guard("sentiment copy evidence input")
    files: dict[str, str] = {}
    path = runtime.pin_file(root, evidence.path, evidence.sha256, files)
    _require(path.name == "_manifest.json" and not path.parent.name.endswith(".partial"), "copy evidence must be completed")
    _no_links(path)
    manifest = read_object(path, evidence.sha256)
    _require(manifest.get("schema") == SCHEMA and manifest.get("status") == "complete_copy_identity_evidence_only"
             and manifest.get("source_index") == source_index.model_dump(mode="json")
             and all(manifest.get(key) is value for key, value in CLOSED.items()), "copy evidence scope or source-index differs")
    artifacts = manifest["artifacts"]
    _require(set(artifacts) == {"_request.json", "index.sqlite"} and artifacts["_request.json"] == manifest["request_sha256"]
             and {item.name for item in path.parent.iterdir()} == {"_manifest.json", *artifacts}, "copy evidence inventory differs")
    for name, digest in artifacts.items():
        artifact = path.parent / name
        _no_links(artifact)
        runtime.pin_file(root, artifact.relative_to(root).as_posix(), digest, files)
    request = read_object(path.parent / "_request.json", manifest["request_sha256"])
    _require(request.get("schema") == SCHEMA + "_request" and all(request.get(key) is value for key, value in CLOSED.items())
             and all(request.get(key) == manifest[key] for key in ("source_index", "corpus", "source_files", "implementation_files")),
             "copy evidence request differs")
    implementations = _implementation(root)
    # Reader uses canonical source-index payload decoding, but does not reopen that index.
    implementations = pins(root, implementations, source_owner._implementation(root))
    _require(all(manifest["implementation_files"].get(name) == digest for name, digest in implementations.items()),
             "copy evidence executed implementation differs")
    provenance = {"build_inputs": manifest["source_files"], "upstream": manifest["producer_source_provenance"]}
    with closing(_connect(path.parent / "index.sqlite", readonly=True)) as db:
        try:
            yield SentimentCopyEvidenceReader(files, implementations, provenance, evidence, db)
        finally:
            runtime.recheck_files(root, files)
            runtime.recheck_files(root, implementations)
            _require({item.name for item in path.parent.iterdir()} == {"_manifest.json", "_request.json", "index.sqlite"},
                     "copy evidence final inventory differs")
