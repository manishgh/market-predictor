"""Stream blinded review packets from the immutable verified candidate sidecar."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, BinaryIO

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.governance.issuer_content_qualification import (
    FAMILIES,
    NEGATIVE_SAMPLES,
    POLICY,
    POLICY_SHA256,
    POSITIVE_SAMPLES,
    ReviewCluster,
    SampleItem,
    select_content_review_sample,
)
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research import issuer_candidate_derivative as derivative
from market_predictor.research import issuer_content_review_population as population
from market_predictor.research import issuer_review_packet_contracts as contract
from market_predictor.research.issuer_review_packet_contracts import (
    CLOSED,
    CORRESPONDENCE_POLICY,
    DOCUMENT_POLICY,
    FRAME_POLICY,
    IssuerReviewPacketConfig,
    PacketVersion,
    VerifiedIssuerReviewPackets,
    policy_identities,
)
from market_predictor.swing.contracts.holding_materialization import SourcePin

SCHEMA = "market_predictor.issuer_review_packets"
ORIGINAL_SAMPLE_ENTRIES = 1750
ORIGINAL_SAMPLE_CLUSTERS = 1748
BATCH_ROWS = 256
MAX_JSON_BYTES = 64 * 1024**2


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _guard(position: int = 0) -> None:
    if position % BATCH_ROWS == 0:
        derivative._guard()


def _hash(path: Path) -> str:
    return derivative._hash(path)


def _object(path: Path) -> dict[str, Any]:
    _require(path.is_file() and path.stat().st_size <= MAX_JSON_BYTES, "packet metadata missing or exceeds bound")
    with path.open("rb") as stream:
        data = stream.read(MAX_JSON_BYTES + 1)
    _require(len(data) <= MAX_JSON_BYTES, "packet metadata grew beyond bound")
    return parse_strict_json_object(data, label=str(path))


def _pin(root: Path, pin: SourcePin, files: dict[str, str]) -> Path:
    path = inside(root, pin.path)
    name = path.relative_to(root).as_posix()
    _require(name not in files or files[name] == pin.sha256, "packet source pins conflict")
    _require(_hash(path) == pin.sha256, f"packet source bytes changed: {name}")
    files[name] = pin.sha256
    return path


def _recheck(root: Path, files: dict[str, str]) -> None:
    for position, (name, digest) in enumerate(sorted(files.items())):
        _guard(position)
        _require(_hash(inside(root, name)) == digest, f"packet source changed: {name}")


def _implementations(root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    _require(package == root / "src/market_predictor", "packet owner must execute from bound repository")
    names = {*derivative.IMPLEMENTATION_PATHS, *population.IMPLEMENTATION_PATHS,
        "research/issuer_review_packet_contracts.py", "research/issuer_review_packets.py",
        "governance/issuer_content_qualification.py", "governance/issuer_event_precision/admission_authority.py"}
    return {(package / name).relative_to(root).as_posix(): _hash(package / name) for name in sorted(names)}


@dataclass(frozen=True)
class _Inputs:
    config: IssuerReviewPacketConfig
    config_pin: SourcePin
    request: dict[str, Any]
    old_samples: list[dict[str, Any]]
    files: dict[str, str]


def _inputs(root: Path, config_pin: SourcePin) -> _Inputs:
    files: dict[str, str] = {}
    config_path = _pin(root, config_pin, files)
    config = IssuerReviewPacketConfig.model_validate_json(_json(_object(config_path)))
    parent_path = _pin(root, config.original_population, files)
    parent = _object(parent_path)
    candidate_path = _pin(root, config.candidate_derivative, files)
    candidate = _object(candidate_path)
    candidate_request_pin = SourcePin(path=(candidate_path.parent / "_request.json").relative_to(root).as_posix(),
        sha256=candidate["request_sha256"])
    candidate_request = _object(_pin(root, candidate_request_pin, files))
    _require(candidate_request["parent_population"] == config.original_population.model_dump(mode="json"),
             "packet derivative belongs to another original population")
    for name, digest in candidate_request["source_files"].items():
        # The canonical connection verifies every original file before yielding.
        pin = SourcePin(path=name, sha256=digest)
        normalized = inside(root, pin.path).relative_to(root).as_posix()
        _require(normalized not in files or files[normalized] == digest, "packet source pins conflict")
        files[normalized] = digest
    for name, digest in candidate["artifacts"].items():
        _pin(root, SourcePin(path=(candidate_path.parent / name).relative_to(root).as_posix(), sha256=digest), files)
    original_sample_path = _pin(root, config.original_sample, files)
    _require(original_sample_path == parent_path.parent / "sample.json"
             and parent["artifacts"]["sample.json"] == config.original_sample.sha256,
             "old sample is not owned by the original population")
    original_sample = _object(original_sample_path)
    _require(original_sample.get("schema") == population.SCHEMA + "_sample"
             and original_sample.get("policy_sha256") == POLICY_SHA256, "original sample policy changed")
    samples = original_sample.get("samples")
    _require(isinstance(samples, list) and len(samples) == ORIGINAL_SAMPLE_ENTRIES, "old sample entry count differs")
    assert isinstance(samples, list)
    _require(all(isinstance(item, dict) for item in samples), "old sample entry malformed")
    ids = [item["sample_id"] for item in samples]
    _require(len(set(ids)) == len(ids) and len({item["cluster_id"] for item in samples}) == ORIGINAL_SAMPLE_CLUSTERS,
             "old sample IDs or distinct cluster count differ")
    headers: list[dict[str, Any]] = []
    for reviewer in config.original_reviewers:
        review = _object(_pin(root, reviewer, files))
        _require(set(review) == {"schema", "population_authority", "sample_sha256", "reviewer_id", "reviewer_kind",
                 "reviewer_model", "evidence_scope", "independent_review", "reviews"}
                 and review["schema"] == "market_predictor.issuer_content_reviews"
                 and review["population_authority"] == config.original_population.model_dump(mode="json")
                 and review["sample_sha256"] == config.original_sample.sha256
                 and review["evidence_scope"] == "source_only" and review["independent_review"] is True,
                 "old reviewer header does not bind independent original source review")
        _require(isinstance(review["reviewer_id"], str) and bool(review["reviewer_id"])
                 and ((review["reviewer_kind"] == "human" and review["reviewer_model"] is None)
                      or (review["reviewer_kind"] == "model_assisted" and isinstance(review["reviewer_model"], str)
                          and bool(review["reviewer_model"]))), "old reviewer identity declaration differs")
        entries = review["reviews"]
        _require(isinstance(entries, list) and all(isinstance(entry, dict) for entry in entries), "old review coverage malformed")
        reviewed_ids = [entry["sample_id"] for entry in entries]
        _require(len(reviewed_ids) == len(ids) and len(set(reviewed_ids)) == len(ids) and set(reviewed_ids) == set(ids),
                 "old reviewer does not cover the complete original sample")
        headers.append({key: value for key, value in review.items() if key != "reviews"}
                       | {"file": reviewer.model_dump(mode="json"), "sample_count": len(reviewed_ids),
                          "sample_ids_sha256": json_sha256(sorted(reviewed_ids))})
    _require(len({header["reviewer_id"] for header in headers}) == 2, "old reviewers are not distinct")
    implementations = _implementations(root)
    for name, digest in candidate_request["implementation_files"].items():
        _require(implementations.get(name) == digest, "packet execution differs from verified derivative implementation")
    _require(all(name not in files or files[name] == digest for name, digest in implementations.items()),
             "packet source and executed implementation pins conflict")
    files.update(implementations)
    request = {"schema": SCHEMA + "_request", "config": config_pin.model_dump(mode="json"),
        "settings": config.model_dump(mode="json", by_alias=True), "source_files": dict(sorted(files.items())),
        "implementation_files": implementations, "original_producer_implementation_files": parent["implementation_files"],
        "candidate_policy_sha256": candidate_request["candidate_policy_sha256"], "numeric_policy": POLICY,
        "numeric_policy_sha256": POLICY_SHA256, "frame_policy": FRAME_POLICY, "document_policy": DOCUMENT_POLICY,
        "correspondence_policy": CORRESPONDENCE_POLICY, **policy_identities(), "old_review_coverage": headers,
        "old_labels_used_as_truth": False, "source_text_reextracted": False, **CLOSED}
    return _Inputs(config, config_pin, request, samples, files)


class _Artifacts:
    """Write new bytes or independently compare the complete expected byte stream."""

    def __init__(self, folder: Path, verify: bool) -> None:
        self.folder, self.verify = folder, verify
        self.hashes: dict[str, str] = {}

    def emit(self, name: str, chunks: Iterator[bytes]) -> str:
        path = inside(self.folder, name)
        _require(not path.is_symlink(), "packet artifact cannot be a symlink")
        if not self.verify:
            path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        stream: BinaryIO
        with path.open("rb" if self.verify else "xb") as stream:
            for chunk in chunks:
                digest.update(chunk)
                if self.verify:
                    _require(stream.read(len(chunk)) == chunk, f"packet artifact replay differs: {name}")
                else:
                    stream.write(chunk)
            if self.verify:
                _require(stream.read(1) == b"", f"packet artifact has extra bytes: {name}")
        self.hashes[name] = digest.hexdigest()
        return digest.hexdigest()

    def document(self, name: str, value: Any) -> str:
        return self.emit(name, iter((_json(value) + b"\n",)))


def _inventories(connection: sqlite3.Connection) -> tuple[dict[str, Any], set[str], dict[str, set[str]]]:
    versions_hash, records_hash, aliases_hash = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    version_ids: dict[str, str] = {}
    cluster_ids: set[str] = set()
    hits: dict[str, set[str]] = {f"{family}:{source}": set() for family, source in contract.DEVELOPMENT_SOURCES}
    counts: Counter[str] = Counter({name: 0 for name in (
        "versions", "unknown_identity_versions", "unavailable_text_versions", "records", "records_without_version", "aliases")})
    for position, row in enumerate(connection.execute(
        "SELECT version_id,cluster_id,security_id,source_family,year,content_json,candidates_json FROM versions ORDER BY version_id")):
        _guard(position)
        version, cluster, security, family, year, encoded, candidates = row
        metadata = parse_strict_json_object(encoded.encode("utf-8"), label="original version metadata")
        identity = f"{family}:{metadata['source_id']}"
        if identity in hits:
            hits[identity].add(cluster)
        cluster_ids.add(cluster)
        _require(version not in version_ids, "original version inventory repeats an ID")
        version_ids[version] = cluster
        versions_hash.update(_json([version, cluster, security, family, year,
            hashlib.sha256(encoded.encode()).hexdigest(), json_sha256(metadata), metadata.get("text_sha256"),
            hashlib.sha256(candidates.encode()).hexdigest()]) + b"\n")
        counts["versions"] += 1
        counts["unknown_identity_versions"] += int(security is None)
        counts["unavailable_text_versions"] += int(metadata.get("text_sha256") is None)
    for position, (phase, ordinal, cluster, version, encoded) in enumerate(connection.execute(
        "SELECT phase,ordinal,cluster_id,version_id,record_json FROM records ORDER BY phase,ordinal")):
        _guard(position)
        _require(not version or version_ids.get(version) == cluster, "original occurrence loses its version/cluster owner")
        records_hash.update(_json([phase, ordinal, cluster, version, hashlib.sha256(encoded.encode()).hexdigest()]) + b"\n")
        counts["records"] += 1
        counts["records_without_version"] += int(not version)
    for position, row in enumerate(connection.execute(
        "SELECT proof_id,security_id,cik,available_ns,proof_json FROM aliases ORDER BY proof_id")):
        _guard(position)
        aliases_hash.update(_json([*row[:4], hashlib.sha256(row[4].encode()).hexdigest()]) + b"\n")
        counts["aliases"] += 1
    _require(counts["versions"] == derivative.ORIGINAL_VERSION_COUNT, "packet inventory does not retain every original version")
    _require(all(hits.values()), "a frozen development source ID is absent from retained versions")
    return {**dict(counts), "versions_sha256": versions_hash.hexdigest(), "records_sha256": records_hash.hexdigest(),
            "aliases_sha256": aliases_hash.hexdigest()}, cluster_ids, hits


def _scope_counts(clusters: list[ReviewCluster]) -> dict[str, Any]:
    result: dict[str, Any] = {"clusters": len(clusters), "families": {}}
    for family in FAMILIES:
        strata: dict[str, Counter[str]] = {}
        roles: Counter[str] = Counter()
        for item in clusters:
            role = "candidate" if family in item.candidate_families else "noncandidate"
            roles[role] += 1
            strata.setdefault(f"{item.source_family}/{item.year}", Counter())[role] += 1
        result["families"][family] = {"candidate": roles["candidate"], "noncandidate": roles["noncandidate"],
            "strata": {key: {"candidate": value["candidate"], "noncandidate": value["noncandidate"]}
                       for key, value in sorted(strata.items())}}
    return result


def _packet_version(connection: sqlite3.Connection, row: tuple[Any, ...]) -> dict[str, Any]:
    version, cluster, security, family, encoded = row
    metadata = parse_strict_json_object(encoded.encode("utf-8"), label="packet original metadata")
    reasons = metadata.get("unavailable_reasons")
    _require(isinstance(reasons, list) and all(isinstance(value, str) for value in reasons), "physical unavailable reasons malformed")
    assert isinstance(reasons, list)
    # These are original content_json physical reasons, never sidecar extraction reasons.
    _require(not any(token in reason for reason in reasons for token in
                     ("unclassified", "no_supported_event", "guidance_direction", "candidate", "rule_id")),
             "extractor status cannot enter a blind physical omission reason")
    omitted = list(reasons)
    if security is None:
        omitted.append("original_identity_unavailable")
    if metadata.get("text_sha256") is None:
        omitted.append("original_text_unavailable")
    elif metadata["text_sha256"] == hashlib.sha256(b"").hexdigest():
        omitted.append("original_text_empty")
    first_seen = metadata.get("first_seen_at_utc")
    if first_seen is None:
        omitted.append("original_first_seen_at_utc_unavailable")
    else:
        stamp = pd.Timestamp(first_seen)
        _require(not pd.isna(stamp) and stamp.tzinfo is not None, "original first-seen clock is invalid or naive")
    for field in ("published_at_utc", "version_available_at_utc", "event_available_at_utc", "identity_available_at_utc"):
        value = metadata.get(field)
        if value is None:
            omitted.append(f"original_{field}_unavailable")
        else:
            stamp = pd.Timestamp(value)
            _require(not pd.isna(stamp) and stamp.tzinfo is not None, "packet original clock is invalid or naive")
            if stamp > derivative.END:
                omitted.append(f"original_{field}_after_initial_fit_cutoff")
    text: str | None = None
    if not omitted:
        found = connection.execute("SELECT text FROM versions WHERE version_id=?", (version,)).fetchone()
        _require(found is not None and isinstance(found[0], str) and bool(found[0]), "reviewable original text missing")
        text = found[0]
        assert isinstance(text, str)
        _require(hashlib.sha256(text.encode("utf-8")).hexdigest() == metadata["text_sha256"], "blind source text hash differs")
    payload = {name: metadata.get(name) for name in (
        "source_id", "source_version_sha256", "text_sha256", "ticker", "published_at_utc", "version_available_at_utc",
        "event_available_at_utc", "identity_available_at_utc", "first_seen_at_utc", "identity_authority_sha256",
        "availability_semantics", "source_locator", "content_kind", "encoding", "alias_proof")}
    payload.update(version_id=version, cluster_id=cluster, security_id=security, source_family=family,
        content_json_sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(), metadata_sha256=json_sha256(metadata),
        physical_unavailable_reasons=tuple(reasons), omission_reasons=tuple(sorted(set(omitted))),
        display_disposition="metadata_only" if omitted else "readable_source_text", text=text)
    return PacketVersion.model_validate(payload).model_dump(mode="json")


def _sample_occurrences(connection: sqlite3.Connection, samples: Sequence[SampleItem]) -> dict[str, list[dict[str, Any]]]:
    """Read original occurrences once, retaining only sampled references and hashes."""
    result: dict[str, list[dict[str, Any]]] = {item.cluster_id: [] for item in samples}
    for position, (phase, ordinal, cluster, version, encoded) in enumerate(connection.execute(
        "SELECT phase,ordinal,cluster_id,version_id,record_json FROM records ORDER BY phase,ordinal")):
        _guard(position)
        if cluster in result:
            result[cluster].append({"phase": phase, "ordinal": ordinal, "version_id": version,
                "record_json_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()})
    return result


def _packet(connection: sqlite3.Connection, sample: SampleItem, bindings: dict[str, str],
            counts: Counter[str], occurrences: list[dict[str, Any]]) -> Iterator[bytes]:
    header = {"schema": SCHEMA + "_blind_packet", "sample_id": sample.sample_id, "cluster_id": sample.cluster_id,
        "event_family_to_assess": sample.event_family, **bindings,
        "instructions": "Inspect every version independently using only original source text and identity evidence. "
            "Record uncertainty for unavailable versions. Offsets are half-open Unicode code points in this exact text. "
            "Do not inspect extractor output, old labels, other reviewers, predictions or outcomes."}
    yield _json(header)[:-1] + b',"versions":['
    version_hash, occurrence_hash = hashlib.sha256(), hashlib.sha256()
    version_count = occurrence_count = 0
    cursor = connection.execute("SELECT version_id,cluster_id,security_id,source_family,content_json FROM versions "
                                "WHERE cluster_id=? ORDER BY version_id", (sample.cluster_id,))
    for position, row in enumerate(cursor):
        _guard(position)
        document = _packet_version(connection, row)
        version_hash.update(_json({key: value for key, value in document.items() if key != "text"}) + b"\n")
        yield (b"," if version_count else b"") + _json(document)
        version_count += 1
        counts["packet_version_entries"] += 1
        counts["metadata_only_entries"] += int(document["display_disposition"] == "metadata_only")
    _require(version_count > 0, "sample packet has no retained versions")
    yield b'],"occurrences":['
    for position, occurrence in enumerate(occurrences):
        _guard(position)
        occurrence_hash.update(_json(occurrence) + b"\n")
        yield (b"," if occurrence_count else b"") + _json(occurrence)
        occurrence_count += 1
    counts["packet_occurrence_entries"] += occurrence_count
    tail = {"version_inventory_sha256": version_hash.hexdigest(), "version_count": version_count,
            "occurrence_inventory_sha256": occurrence_hash.hexdigest(), "occurrence_count": occurrence_count}
    yield b"]," + _json(tail)[1:] + b"\n"


def _render(connection: sqlite3.Connection, inputs: _Inputs, artifacts: _Artifacts) -> dict[str, Any]:
    request_hash = artifacts.document("_request.json", inputs.request)
    inventory, all_clusters, hits = _inventories(connection)
    old_clusters = {item["cluster_id"] for item in inputs.old_samples}
    _require(old_clusters.issubset(all_clusters), "prior sample cluster disappeared from original versions")
    excluded = old_clusters.union(*(values for values in hits.values()))
    clusters = population._clusters(connection)
    universe = {item.cluster_id for item in clusters}
    _require(len(universe) == len(clusters), "full review cluster inventory has duplicates")
    fresh = [item for item in clusters if item.cluster_id not in excluded]
    excluded_u = [item for item in clusters if item.cluster_id in excluded]
    reasons: dict[str, list[str]] = {key: ["prior_sample_any_family_or_role"] for key in old_clusters}
    for source, owners in sorted(hits.items()):
        for key in owners:
            reasons.setdefault(key, []).append("development_source:" + source)
    exclusions = {"schema": SCHEMA + "_exclusions", "request_sha256": request_hash, **policy_identities(),
        "original_sample_entries": len(inputs.old_samples), "original_sample_distinct_clusters": len(old_clusters),
        "development_resolutions": {key: sorted(value) for key, value in sorted(hits.items())},
        "clusters": [{"cluster_id": key, "reasons": sorted(reasons[key]), "in_U": key in universe,
                      "fresh_inclusion_probability": 0, "disposition": "development_inspected_not_newly_qualified"}
                     for key in sorted(excluded)], "outside_U": sorted(excluded - universe)}
    exclusion_hash = artifacts.document("exclusions.json", exclusions)
    def frame_rows() -> Iterator[bytes]:
        for position, item in enumerate(clusters):
            _guard(position)
            yield _json({**asdict(item), "fresh_review_eligible": item.cluster_id not in excluded}) + b"\n"
    frame_hash = artifacts.emit("frame.jsonl", frame_rows())
    samples = select_content_review_sample(fresh)
    sample_hash = artifacts.document("sample.json", {"schema": SCHEMA + "_sample", "request_sha256": request_hash,
        "numeric_policy_sha256": POLICY_SHA256, **policy_identities(), "frame_sha256": frame_hash,
        "exclusions_sha256": exclusion_hash, "samples": [asdict(item) for item in samples]})
    bindings = {"request_sha256": request_hash, "frame_sha256": frame_hash, "exclusions_sha256": exclusion_hash,
                "sample_sha256": sample_hash, **policy_identities()}
    packet_counts: Counter[str] = Counter()
    occurrences = _sample_occurrences(connection, samples)
    for sample in samples:
        _guard()
        _require(sample.cluster_id not in excluded, "development cluster leaked into the fresh sample")
        artifacts.emit(f"blind/{sample.sample_id}.json",
                       _packet(connection, sample, bindings, packet_counts, occurrences[sample.cluster_id]))
    sample_counts = Counter(f"{item.event_family}/{item.role}" for item in samples)
    requested = {f"{family}/{role}": POSITIVE_SAMPLES[family] if role == "candidate" else NEGATIVE_SAMPLES
                 for family in FAMILIES for role in ("candidate", "noncandidate")}
    return {"schema": SCHEMA, "status": "complete_blind_review_packets_only", "request_sha256": request_hash,
        "artifacts": dict(sorted(artifacts.hashes.items())), "source_files": inputs.files,
        "counts": {"inventory": inventory, "U": _scope_counts(clusters), "D_in_U": _scope_counts(excluded_u),
                   "F": _scope_counts(fresh), "excluded_clusters": len(excluded), "excluded_outside_U": len(excluded - universe),
                   "sample_entries": len(samples), "sample_distinct_clusters": len({item.cluster_id for item in samples}),
                   "sample_by_family_role": dict(sorted(sample_counts.items())), "requested_sample_by_family_role": requested,
                   "sample_shortages": {key: max(0, value - sample_counts[key]) for key, value in requested.items()},
                   **dict(packet_counts)},
        "frame_sha256": frame_hash, "exclusions_sha256": exclusion_hash, "sample_sha256": sample_hash,
        "complete_original_version_inventory_preserved": True, "full_population_quality_established": False, **CLOSED}


def _verify_artifacts(folder: Path, manifest: dict[str, Any], *, include_manifest: bool) -> None:
    expected = set(manifest["artifacts"]) | ({"_manifest.json"} if include_manifest else set())
    actual: set[str] = set()
    for path in folder.rglob("*"):
        _require(not path.is_symlink(), "packet output contains a symlink")
        if path.is_file():
            actual.add(path.relative_to(folder).as_posix())
    _require(actual == expected, "packet artifact inventory contains extra or missing files")
    for position, (name, digest) in enumerate(manifest["artifacts"].items()):
        _guard(position)
        _require(_hash(inside(folder, name)) == digest, "packet output artifact changed")


def _runtime(root: Path) -> Path:
    runtime = heavy_job_runtime_dir()
    return runtime if runtime.is_absolute() else root / runtime


def publish_issuer_review_packets(*, root: Path, config: Path, expected_config_sha256: str,
                                 output: Path) -> dict[str, Any]:
    root = root.resolve()
    destination = inside(root, output)
    _require(destination.parent == root / "data/research", "packet output must be a new research directory")
    stage = destination.with_name("." + destination.name + ".private_stage")
    with heavy_job_lease("publish-issuer-review-packets", runtime_dir=_runtime(root)):
        _guard()
        _require(not destination.exists() and not stage.exists(), "packet output or private stage already exists")
        inputs = _inputs(root, SourcePin(path=inside(root, config).relative_to(root).as_posix(), sha256=expected_config_sha256))
        for name in inputs.files:
            path = inside(root, name)
            _require(not path.is_relative_to(destination) and not path.is_relative_to(stage), "packet output overlaps source evidence")
        stage.mkdir()
        with derivative._verified_candidate_connection(root=root, publication=inputs.config.candidate_derivative) as connection:
            manifest = _render(connection, inputs, _Artifacts(stage, verify=False))
        # All source-owner exit checks have completed before a public authority exists.
        _recheck(root, inputs.files)
        _require(_implementations(root) == inputs.request["implementation_files"], "packet implementation changed")
        _verify_artifacts(stage, manifest, include_manifest=False)
        _Artifacts(stage, verify=False).document("_manifest.json", manifest)
        digest = _hash(stage / "_manifest.json")
        _recheck(root, inputs.files)
        _verify_artifacts(stage, manifest, include_manifest=True)
        _require(_hash(stage / "_manifest.json") == digest, "packet manifest changed before publication")
        _guard()
        _require(not destination.exists(), "packet destination appeared before publication")
        stage.rename(destination)
    return {"status": manifest["status"], "manifest_sha256": digest, "counts": manifest["counts"], **CLOSED}


def verify_issuer_review_packets(*, root: Path, publication: SourcePin) -> VerifiedIssuerReviewPackets:
    root = root.resolve()
    with heavy_job_lease("verify-issuer-review-packets", runtime_dir=_runtime(root)):
        _guard()
        files: dict[str, str] = {}
        path = _pin(root, publication, files)
        _require(path.name == "_manifest.json", "packet verification requires completed manifest pin")
        saved = _object(path)
        request = _object(path.parent / "_request.json")
        _require(_hash(path.parent / "_request.json") == saved.get("request_sha256"), "packet request hash differs")
        inputs = _inputs(root, SourcePin.model_validate(request["config"]))
        with derivative._verified_candidate_connection(root=root, publication=inputs.config.candidate_derivative) as connection:
            expected = _render(connection, inputs, _Artifacts(path.parent, verify=True))
            _Artifacts(path.parent, verify=True).document("_manifest.json", expected)
        _recheck(root, inputs.files | files)
        _verify_artifacts(path.parent, expected, include_manifest=True)
    return VerifiedIssuerReviewPackets(publication, inputs.request, expected["counts"], inputs.files | files,
        expected["frame_sha256"], expected["exclusions_sha256"], expected["sample_sha256"])
