"""Offline candidate replay over one immutable saved source-text population.

Original extraction/producer bytes are provenance, never executed or represented
as current code. The derivative owns candidates only; text, aliases, identities,
clocks and original exclusions remain in the read-only parent database.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sqlite3
from collections import Counter
from collections.abc import Iterator, Mapping
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import pandas as pd

from market_predictor.catalysts.issuer_events.content_review import (
    EXTRACTION_POLICY_SHA256,
    IssuerContentContext,
    extract_review_candidates,
    saved_text_content_evidence,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.resources import assert_memory_budget
from market_predictor.swing.contracts.holding_materialization import SourcePin

SCHEMA = "market_predictor.issuer_candidate_derivative"
ORIGINAL_MANIFEST_SHA256 = "3c59845761740dc1a07a2f994cceaba53354c6f9486f4baa634ebd422a6a84ce"
ORIGINAL_VERSION_COUNT = 516679
ORIGINAL_PRODUCER_COUNT = 20
PARENT_SCHEMA = "market_predictor.issuer_content_review_population"
END = pd.Timestamp("2024-05-28T22:00:00Z")
BATCH_ROWS = 256
MAX_JSON_BYTES = 64 * 1024**2
CLOSED = {"training_eligible": False, "serving_eligible": False, "promotion_eligible": False,
          "qualification_established": False, "economic_eligible": False}
IMPLEMENTATION_PATHS = (
    "research/issuer_candidate_derivative.py", "catalysts/issuer_events/content_review.py",
    "swing/contracts/holding_materialization.py", "swing/contracts/holding_accounting.py",
    "evidence/io.py", "evidence/hashing.py", "core/json_integrity.py", "core/errors.py",
    "core/system_memory.py", "heavy_jobs.py", "resources.py", "locking.py",
)
COLUMNS = (
    "version_id", "cluster_id", "security_id", "source_family", "year", "source_id", "source_version_sha256",
    "content_json_sha256", "metadata_sha256", "text_sha256", "candidate_policy_sha256",
    "candidates_json", "reasons_json", "disposition",
)
PARENT_COLUMNS = "version_id,cluster_id,security_id,source_family,year,content_json,text,candidates_json"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _guard() -> None:
    assert_memory_budget(stage="saved issuer candidate derivative", hard_budget_gib=5.0, headroom_gib=0.75)
    assert_system_memory_available(minimum_available_gib=0.75, maximum_used_percent=90.0)


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        count = 0
        while block := stream.read(1024**2):
            digest.update(block)
            count += 1
            if count % 64 == 0:
                _guard()
    return digest.hexdigest()


def _digest(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def _object(path: Path) -> dict[str, Any]:
    _require(path.stat().st_size <= MAX_JSON_BYTES, "derivative JSON exceeds bounded reader")
    with path.open("rb") as stream:
        content = stream.read(MAX_JSON_BYTES + 1)
    _require(len(content) <= MAX_JSON_BYTES, "derivative JSON grew beyond byte limit")
    return parse_strict_json_object(content, label=str(path))


def _pin(root: Path, pin: SourcePin, files: dict[str, str]) -> Path:
    path = inside(root, pin.path)
    name = path.relative_to(root).as_posix()
    _require(name not in files or files[name] == pin.sha256, "conflicting derivative source pins")
    _require(_hash(path) == pin.sha256, f"derivative input bytes changed: {name}")
    files[name] = pin.sha256
    return path


def _recheck(root: Path, files: Mapping[str, str]) -> None:
    for position, (name, digest) in enumerate(files.items()):
        if position % BATCH_ROWS == 0:
            _guard()
        _require(_digest(digest) and _hash(inside(root, name)) == digest, f"derivative source changed: {name}")
    _guard()


def _implementations(root: Path) -> dict[str, str]:
    # This prototype is not executed from scratch. Installed source must be beneath root.
    package = Path(__file__).resolve().parents[1]
    return {(package / name).relative_to(root).as_posix(): _hash(package / name) for name in IMPLEMENTATION_PATHS}


@dataclass(frozen=True)
class _Parent:
    pin: SourcePin
    folder: Path
    manifest: dict[str, Any]
    request: dict[str, Any]
    files: dict[str, str]


def _parent(root: Path, pin: SourcePin) -> _Parent:
    _require(pin.sha256 == ORIGINAL_MANIFEST_SHA256, "derivative requires the exact original population authority")
    files: dict[str, str] = {}
    path = _pin(root, pin, files)
    _require(path.name == "_manifest.json", "parent is not a completed manifest")
    manifest = _object(path)
    _require(manifest.get("schema") == PARENT_SCHEMA and manifest.get("status") == "complete_review_population_only"
             and all(manifest.get(key) is False for key in ("training_eligible", "serving_eligible", "promotion_eligible")),
             "original review population claims or schema differ")
    producer = manifest.get("implementation_files")
    _require(isinstance(producer, dict) and len(producer) == ORIGINAL_PRODUCER_COUNT
             and all(isinstance(name, str) and name.startswith("src/market_predictor/") and name.endswith(".py")
                     and _digest(digest) for name, digest in producer.items()), "original producer inventory differs")
    assert isinstance(producer, dict)
    artifacts, sources = manifest.get("artifacts"), manifest.get("source_files")
    _require(isinstance(artifacts, dict) and {"population.sqlite", "_request.json", "sample.json"}.issubset(artifacts)
             and isinstance(sources, dict) and bool(sources), "original source/artifact inventory is absent")
    assert isinstance(artifacts, dict) and isinstance(sources, dict)
    _require(not set(sources).intersection(producer), "original provenance is mixed into current source evidence")
    for position, (name, digest) in enumerate(artifacts.items()):
        if position % BATCH_ROWS == 0:
            _guard()
        child = inside(path.parent, name)
        _pin(root, SourcePin(path=child.relative_to(root).as_posix(), sha256=digest), files)
    for position, (name, digest) in enumerate(sources.items()):
        if position % BATCH_ROWS == 0:
            _guard()
        _pin(root, SourcePin(path=name, sha256=digest), files)
    request = _object(path.parent / "_request.json")
    _require(manifest.get("request_sha256") == artifacts["_request.json"]
             and request.get("schema") == PARENT_SCHEMA + "_request"
             and request.get("implementation_files") == producer
             and _digest(request.get("extraction_policy_sha256"))
             and manifest.get("totals", {}).get("source_versions") == ORIGINAL_VERSION_COUNT,
             "original request, producer, policy or population differs")
    return _Parent(pin, path.parent, manifest, request, files)


def _readonly_uri(path: Path) -> str:
    return path.resolve().as_uri() + "?mode=ro&immutable=1"


def _settings(connection: sqlite3.Connection) -> None:
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-8192")
    connection.execute("PRAGMA mmap_size=0")
    connection.execute("PRAGMA trusted_schema=OFF")


@contextmanager
def _original_connection(parent: _Parent) -> Iterator[sqlite3.Connection]:
    with closing(sqlite3.connect(_readonly_uri(parent.folder / "population.sqlite"), uri=True)) as connection:
        _settings(connection)
        connection.execute("PRAGMA query_only=ON")
        yield connection


def _clock(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    _require(not pd.isna(stamp) and stamp.tzinfo is not None and stamp.utcoffset() == pd.Timedelta(0),
             "parent source clock is not explicit UTC")
    return stamp


def _project(row: tuple[Any, ...]) -> tuple[tuple[Any, ...], bool]:
    version, cluster, security, family, year, encoded, text, old_candidates = row
    _require(isinstance(encoded, str) and len(encoded.encode("utf-8")) <= MAX_JSON_BYTES,
             "original version metadata exceeds bound")
    metadata = parse_strict_json_object(encoded.encode("utf-8"), label="original content_json")
    _require(metadata.get("source_family") == family and metadata.get("security_id") == security
             and family in ("alpaca", "sec") and type(year) is int, "parent version identity columns contradict metadata")
    source_id, source_version = metadata.get("source_id"), metadata.get("source_version_sha256")
    _require(isinstance(source_id, str) and bool(source_id) and _digest(cluster) and _digest(version)
             and version == json_sha256([cluster, source_id, source_version]), "parent source/version identity differs")
    assert isinstance(source_id, str)
    text_hash = None if text is None else hashlib.sha256(text.encode("utf-8")).hexdigest()
    _require(text_hash == metadata.get("text_sha256"), "saved source text changed")
    original_reasons = metadata.get("unavailable_reasons")
    _require(isinstance(original_reasons, list) and all(isinstance(reason, str) and reason for reason in original_reasons),
             "parent unavailable reasons are malformed")
    reasons = list(cast(list[str], original_reasons))
    if _clock(metadata.get("version_available_at_utc")) > END:
        reasons.append("version_after_initial_fit_cutoff")
    if text is None or not text:
        reasons.append("saved_text_unavailable")
    if not security:
        reasons.append("saved_issuer_identity_unavailable")
    candidates: list[dict[str, Any]] = []
    disposition = "preserved_unavailable"
    if not reasons:
        alias = metadata.get("alias_proof")
        _require(isinstance(alias, dict) and isinstance(alias.get("aliases"), list), "parent causal aliases are absent")
        assert isinstance(alias, dict)
        source = metadata.get("source_metadata")
        _require(isinstance(source, dict), "parent source clock metadata absent")
        assert isinstance(source, dict)
        event_clock = _clock(source["filing"]["available_at_utc"] if family == "sec" else source["available_at_utc"])
        context = IssuerContentContext(event_id=source_id, security_id=security, ticker=cast(str, metadata["ticker"]),
            issuer_aliases=tuple(alias["aliases"]), identity_authority_sha256=cast(str, metadata["identity_authority_sha256"]),
            identity_available_at_utc=_clock(metadata["identity_available_at_utc"]),
            first_seen_at_utc=_clock(metadata["first_seen_at_utc"]), availability_semantics="historical_proxy",
            availability_policy_sha256=json_sha256(["retained_initial_fit_source_publication_proxy", event_clock.isoformat()]))
        _require(isinstance(text, str) and isinstance(text_hash, str), "readable saved text is absent")
        assert isinstance(text, str) and isinstance(text_hash, str)
        evidence = saved_text_content_evidence(text=text, expected_text_sha256=text_hash, metadata=metadata,
            expected_metadata_sha256=json_sha256(metadata), context=context,
            expected_extraction_policy_sha256=EXTRACTION_POLICY_SHA256)
        result = extract_review_candidates(evidence)
        candidates = [asdict(candidate) for candidate in result.candidates]
        reasons = list(result.reasons)
        disposition = result.disposition
    previous = json.loads(old_candidates)
    _require(isinstance(previous, list), "parent candidate record is not a list")
    fields = (version, cluster, security, family, year, source_id, source_version,
              hashlib.sha256(encoded.encode("utf-8")).hexdigest(), json_sha256(metadata), text_hash,
              EXTRACTION_POLICY_SHA256, _json(candidates), _json(sorted(set(reasons))), disposition)
    return fields, _json(previous) != fields[11]


def _request(parent: _Parent, implementations: dict[str, str]) -> dict[str, Any]:
    return {"schema": SCHEMA + "_request", "parent_population": parent.pin.model_dump(mode="json"),
            "original_request_sha256": parent.manifest["request_sha256"],
            "original_extraction_policy_sha256": parent.request["extraction_policy_sha256"],
            "original_producer_implementation_files": parent.manifest["implementation_files"],
            "candidate_policy_sha256": EXTRACTION_POLICY_SHA256,
            "implementation_files": implementations, "source_files": parent.files,
            "scope": "all_original_versions_candidates_only_no_source_text_reconstruction",
            "runtime": {"python": platform.python_version(), "sqlite": sqlite3.sqlite_version, "pandas": pd.__version__},
            **CLOSED}


def _create_sidecar(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    _settings(connection)
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA synchronous=FULL")
    definitions = [f"{name} {'INTEGER' if name == 'year' else 'TEXT'}"
                   + (" PRIMARY KEY NOT NULL" if name == "version_id" else "") for name in COLUMNS]
    connection.execute("CREATE TABLE candidate_versions (" + ",".join(definitions) + ") WITHOUT ROWID")
    connection.commit()
    return connection


@dataclass(frozen=True)
class VerifiedCandidateDerivative:
    publication: SourcePin
    parent_population: SourcePin
    counts: dict[str, int]
    source_files: dict[str, str]
    original_producer_implementation_files: dict[str, str]


def _verify(root: Path, publication: SourcePin, parent: _Parent | None = None) -> VerifiedCandidateDerivative:
    """Private, actual replay for an already leased owner; never accept a passed flag."""
    files: dict[str, str] = {}
    path = _pin(root, publication, files)
    manifest = _object(path)
    _require(path.name == "_manifest.json" and manifest.get("schema") == SCHEMA
             and manifest.get("status") == "complete_candidate_derivative_only"
             and all(manifest.get(key) is value for key, value in CLOSED.items()), "derivative completion contract differs")
    artifacts = manifest.get("artifacts")
    _require(isinstance(artifacts, dict) and set(artifacts) == {"_request.json", "candidates.sqlite"},
             "derivative artifact inventory differs")
    assert isinstance(artifacts, dict)
    for name, digest in artifacts.items():
        _pin(root, SourcePin(path=(path.parent / name).relative_to(root).as_posix(), sha256=digest), files)
    request = _object(path.parent / "_request.json")
    parent_pin = SourcePin.model_validate(request["parent_population"])
    if parent is None:
        parent = _parent(root, parent_pin)
    _require(parent.pin == parent_pin and request == _request(parent, _implementations(root))
             and manifest.get("request_sha256") == artifacts["_request.json"], "derivative request or source identity differs")
    all_files = {**parent.files, **request["implementation_files"], **files}
    counts: Counter[str] = Counter()
    with _original_connection(parent) as original, closing(sqlite3.connect(
            _readonly_uri(path.parent / "candidates.sqlite"), uri=True)) as derivative:
        _settings(derivative)
        derivative.execute("PRAGMA query_only=ON")
        _require(tuple(item[1] for item in derivative.execute("PRAGMA table_info(candidate_versions)")) == COLUMNS,
                 "derivative columns differ")
        _require(derivative.execute("SELECT name,type FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'").fetchall()
                 == [("candidate_versions", "table")], "derivative contains undeclared database objects")
        cursor = derivative.execute("SELECT " + ",".join(COLUMNS) + " FROM candidate_versions ORDER BY version_id")
        for position, row in enumerate(original.execute("SELECT " + PARENT_COLUMNS + " FROM versions ORDER BY version_id")):
            if position % BATCH_ROWS == 0:
                _guard()
            expected, changed = _project(row)
            _require(cursor.fetchone() == expected, "derivative version missing, extra, reordered or candidate replay differs")
            counts["versions"] += 1
            counts["changed_candidate_versions"] += int(changed)
            counts[str(expected[-1])] += 1
        _require(cursor.fetchone() is None and counts["versions"] == ORIGINAL_VERSION_COUNT,
                 "derivative does not cover exactly the original version population")
    _require(dict(counts) == manifest.get("counts"), "derivative completion counts differ")
    _recheck(root, all_files)
    return VerifiedCandidateDerivative(publication, parent_pin, dict(counts), all_files,
                                       dict(parent.manifest["implementation_files"]))


def _runtime(root: Path) -> Path:
    runtime = heavy_job_runtime_dir()
    return runtime if runtime.is_absolute() else root / runtime


def publish_issuer_candidate_derivative(*, root: Path, parent_population: SourcePin, output: Path) -> dict[str, Any]:
    """Publish a new small candidate sidecar; failed private stages are retained."""
    root = root.resolve()
    destination = inside(root, output)
    _require(destination.parent == root / "data/research", "derivative output must be a new research directory")
    stage = destination.with_name("." + destination.name + ".deriving")
    with heavy_job_lease("derive-saved-issuer-candidates", runtime_dir=_runtime(root)):
        _guard()
        _require(not destination.exists() and not stage.exists(), "derivative output/private stage already exists")
        parent = _parent(root, parent_population)
        implementations = _implementations(root)
        for name in parent.files:
            source = inside(root, name)
            _require(not source.is_relative_to(destination) and not source.is_relative_to(stage), "derivative overlaps original inputs")
        stage.mkdir()
        request = _request(parent, implementations)
        write_json_object(stage / "_request.json", request)
        counts: Counter[str] = Counter()
        with _original_connection(parent) as original, closing(_create_sidecar(stage / "candidates.sqlite")) as derivative:
            insert = "INSERT INTO candidate_versions VALUES (" + ",".join("?" for _ in COLUMNS) + ")"
            for position, row in enumerate(original.execute("SELECT " + PARENT_COLUMNS + " FROM versions ORDER BY version_id")):
                if position % BATCH_ROWS == 0:
                    _guard()
                values, changed = _project(row)
                derivative.execute(insert, values)
                counts["versions"] += 1
                counts["changed_candidate_versions"] += int(changed)
                counts[str(values[-1])] += 1
                if counts["versions"] % BATCH_ROWS == 0:
                    derivative.commit()
            derivative.commit()
        _require(counts["versions"] == ORIGINAL_VERSION_COUNT, "derivative source inventory changed")
        artifacts = {name: _hash(stage / name) for name in ("_request.json", "candidates.sqlite")}
        manifest = {"schema": SCHEMA, "status": "complete_candidate_derivative_only",
                    "request_sha256": artifacts["_request.json"], "artifacts": artifacts, "counts": dict(counts), **CLOSED}
        _recheck(root, {**parent.files, **implementations})
        write_json_object(stage / "_manifest.json", manifest)
        pin = SourcePin(path=(stage / "_manifest.json").relative_to(root).as_posix(), sha256=_hash(stage / "_manifest.json"))
        _verify(root, pin, parent)
        _guard()
        _require(not destination.exists(), "derivative destination appeared before publication")
        stage.rename(destination)
        return {**manifest, "manifest_sha256": pin.sha256}


def verify_issuer_candidate_derivative(*, root: Path, publication: SourcePin) -> VerifiedCandidateDerivative:
    root = root.resolve()
    with heavy_job_lease("verify-saved-issuer-candidates", runtime_dir=_runtime(root)):
        return _verify(root, publication)


@contextmanager
def _verified_candidate_connection(*, root: Path, publication: SourcePin) -> Iterator[sqlite3.Connection]:
    """Already-leased canonical owner: replay fully, then expose read-only views.

    Never accepts caller-provided verification objects or skip-verification flags.
    """
    verified = _verify(root, publication)
    parent = inside(root, verified.parent_population.path).parent / "population.sqlite"
    derivative = inside(root, publication.path).parent / "candidates.sqlite"
    with closing(sqlite3.connect(":memory:", uri=True)) as connection:
        _settings(connection)
        connection.execute("ATTACH DATABASE ? AS original", (_readonly_uri(parent),))
        connection.execute("ATTACH DATABASE ? AS derivative", (_readonly_uri(derivative),))
        for schema in ("original", "derivative"):
            connection.execute(f"PRAGMA {schema}.cache_size=-8192")
            connection.execute(f"PRAGMA {schema}.mmap_size=0")
        connection.execute("CREATE TEMP VIEW records AS SELECT * FROM original.records")
        connection.execute("CREATE TEMP VIEW aliases AS SELECT * FROM original.aliases")
        connection.execute("CREATE TEMP VIEW versions AS SELECT p.version_id,p.cluster_id,p.security_id,p.source_family,"
                           "p.year,p.content_json,p.text,d.candidates_json FROM original.versions p "
                           "JOIN derivative.candidate_versions d ON d.version_id=p.version_id")
        connection.execute("PRAGMA query_only=ON")
        try:
            yield connection
        finally:
            _recheck(root, verified.source_files)


@contextmanager
def verified_candidate_connection(*, root: Path, publication: SourcePin) -> Iterator[sqlite3.Connection]:
    root = root.resolve()
    with heavy_job_lease("read-saved-issuer-candidates", runtime_dir=_runtime(root)):
        with _verified_candidate_connection(root=root, publication=publication) as connection:
            yield connection
