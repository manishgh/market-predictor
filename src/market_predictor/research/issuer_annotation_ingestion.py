"""Pinned complete-review ingestion; correspondence evidence grants no admission."""
from __future__ import annotations

import hashlib
import os
import sqlite3
import stat
from collections.abc import Iterator
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from market_predictor.catalysts.issuer_events.content_review import EvidenceSpan, ReviewCandidate
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.heavy_jobs import heavy_job_lease
from market_predictor.research import issuer_candidate_derivative as derivative
from market_predictor.research import issuer_review_packets as packets
from market_predictor.research.issuer_annotation_correspondence import (
    CandidateOccurrence,
    OccurrenceCorrespondence,
    correspond_candidates,
)
from market_predictor.research.issuer_review_packet_contracts import CLOSED, VerifiedIssuerReviewPackets, policy_identities
from market_predictor.research.issuer_source_annotations import (
    BlindReviewPacket,
    ReviewerIdentity,
    ValidatedPacketAssessments,
    parse_packet_assessment,
    validate_packet_assessments,
)
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin

SCHEMA = "market_predictor.issuer_annotation_correspondence"
FROZEN_PACKET_SHA256 = "2b4b933565704a06fae86e815f3edab56412c2d2717254f4eb39e5ad5f50b3a9"
_CANDIDATE = TypeAdapter(ReviewCandidate)
_RESULT = TypeAdapter(OccurrenceCorrespondence)
_require = packets._require


class AnnotationIngestionConfig(HoldingContract):
    schema_version: Literal["market_predictor.issuer_annotation_ingestion_config"] = Field(alias="schema")
    packet_publication: SourcePin
    reviewer_files: tuple[SourcePin, SourcePin]
    frame_policy_sha256: Sha256
    document_policy_sha256: Sha256
    correspondence_policy_sha256: Sha256

    @model_validator(mode="after")
    def frozen_sources(self) -> Self:
        _require(self.packet_publication.sha256 == FROZEN_PACKET_SHA256, "annotation config requires frozen actual packet")
        _require(self.reviewer_files[0].path != self.reviewer_files[1].path, "reviewer paths must differ")
        _require(all(getattr(self, name) == digest for name, digest in policy_identities().items()),
                 "annotation policy differs from frozen packet policy")
        return self


@dataclass(frozen=True)
class VerifiedAnnotationCorrespondence:
    publication: SourcePin
    source_files: dict[str, str]
    counts: dict[str, int]
    qualification_established: Literal[False] = False
    training_eligible: Literal[False] = False
    serving_eligible: Literal[False] = False
    promotion_eligible: Literal[False] = False
    economic_eligible: Literal[False] = False


def _path(root: Path, name: str | Path) -> Path:
    path = inside(root, name)
    original = root / name
    for part in (original, *original.parents):
        _require(not part.is_symlink(), "annotation path contains a symlink")
        if part.exists():
            attributes = getattr(part.lstat(), "st_file_attributes", 0)
            _require(not attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT, "annotation path contains a Windows reparse point")
        if part == root:
            break
    return path


def _implementations(root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    _require(package == root / "src/market_predictor", "annotation owner must execute from its bound repository")
    result = packets._implementations(root)
    for name in ("research/issuer_annotation_ingestion.py", "research/issuer_source_annotations.py",
                 "research/issuer_annotation_correspondence.py", "core/json_integrity.py",
                 "core/errors.py", "evidence/hashing.py", "evidence/io.py",
                 "swing/contracts/holding_accounting.py", "swing/contracts/holding_materialization.py"):
        path = package / name
        result[path.relative_to(root).as_posix()] = packets._hash(path)
    return dict(sorted(result.items()))


def _recheck(root: Path, files: dict[str, str]) -> None:
    for position, name in enumerate(sorted(files)):
        packets._guard(position)
        _path(root, name)
    packets._recheck(root, files)


def _settings(root: Path, pin: SourcePin) -> AnnotationIngestionConfig:
    _path(root, pin.path)
    path = packets._pin(root, pin, {})
    return AnnotationIngestionConfig.model_validate_json(packets._json(packets._object(path)))


def _reviewer_preflight(root: Path, config: AnnotationIngestionConfig) -> None:
    """Cheap existence/distinctness only; processing still rechecks physical pins."""
    paths = [_path(root, pin.path) for pin in config.reviewer_files]
    _require(all(path.is_file() for path in paths), "both physical reviewer files are required before packet replay")
    _require(paths[0] != paths[1] and not os.path.samefile(*paths), "reviewer artifacts resolve to the same physical file")


def _inputs(root: Path, config_pin: SourcePin, config: AnnotationIngestionConfig,
            verified: VerifiedIssuerReviewPackets) -> tuple[dict[str, Any], dict[str, str], Path, list[str]]:
    _require(verified.publication == config.packet_publication, "canonical packet verifier returned another publication")
    files = dict(verified.source_files)
    folder = _path(root, config.packet_publication.path).parent
    packets._pin(root, config.packet_publication, files)
    manifest = packets._object(folder / "_manifest.json")
    packets._verify_artifacts(folder, manifest, include_manifest=True)
    for name, digest in manifest["artifacts"].items():
        path = _path(root, folder / name)
        packets._pin(root, SourcePin(path=path.relative_to(root).as_posix(), sha256=digest), files)
    packet_request = packets._object(folder / "_request.json")
    # Replayed policies contain tuples; immutable JSON represents them as arrays.
    # Compare their canonical wire bytes, preserving value types and array order.
    _require(packets._json(packet_request) == packets._json(verified.request), "verified packet request differs")
    reviewer_paths = [_path(root, pin.path) for pin in config.reviewer_files]
    _require(reviewer_paths[0] != reviewer_paths[1] and not os.path.samefile(*reviewer_paths),
             "reviewer artifacts resolve to the same physical file")
    for pin in (config_pin, *config.reviewer_files):
        _path(root, pin.path)
        packets._pin(root, pin, files)
    executed = _implementations(root)
    for name, digest in executed.items():
        _require(name not in files or files[name] == digest, "executed implementation conflicts with source provenance")
        files[name] = digest
    sample = packets._object(folder / "sample.json")
    samples = sorted(item["sample_id"] for item in sample["samples"])
    _require(bool(samples) and len(set(samples)) == len(samples), "packet sample inventory empty or repeated")
    _require({f"blind/{sample_id}.json" for sample_id in samples}
             == {name for name in manifest["artifacts"] if name.startswith("blind/")}, "packet sample artifact inventory differs")
    request = {"schema": SCHEMA + "_request", "config": config_pin.model_dump(mode="json"),
        "settings": config.model_dump(mode="json", by_alias=True), "source_files": dict(sorted(files.items())),
        "implementation_files": executed, "packet_implementation_files": packet_request["implementation_files"],
        "frame_sha256": verified.frame_sha256, "sample_sha256": verified.sample_sha256,
        "exclusions_sha256": verified.exclusions_sha256, **CLOSED}
    _recheck(root, files)
    return request, files, folder, samples


def _reviews(root: Path, config: AnnotationIngestionConfig, folder: Path, samples: list[str]
             ) -> Iterator[tuple[BlindReviewPacket, tuple[ValidatedPacketAssessments, ValidatedPacketAssessments]]]:
    headers: list[ReviewerIdentity | None] = [None, None]
    ids: list[set[str]] = [set(), set()]
    with _path(root, config.reviewer_files[0].path).open("rb") as first, _path(root, config.reviewer_files[1].path).open("rb") as second:
        for position, sample_id in enumerate(samples):
            packets._guard(position)
            packet = BlindReviewPacket.model_validate_json(packets._json(packets._object(folder / "blind" / f"{sample_id}.json")))
            _require(packet.sample_id == sample_id, "blind packet sample identity differs")
            validated = []
            for index, stream in enumerate((first, second)):
                line = stream.readline(packets.MAX_JSON_BYTES + 1)
                _require(bool(line) and len(line) <= packets.MAX_JSON_BYTES, "review record missing or exceeds bound")
                assessment = parse_packet_assessment(line)
                _require(assessment.sample_id == sample_id and assessment.packet_publication == config.packet_publication,
                         "review sample order, coverage or packet publication differs")
                if headers[index] is None:
                    headers[index] = assessment.reviewer
                _require(headers[index] == assessment.reviewer, "reviewer declaration changed within file")
                for version in assessment.versions:
                    for annotation in version.annotations:
                        _require(annotation.annotation_id not in ids[index], "reviewer annotation ID repeated across packets")
                        ids[index].add(annotation.annotation_id)
                validated.append(validate_packet_assessments(packet, assessment))
            _require(headers[0] is not None and headers[1] is not None
                     and headers[0].reviewer_id != headers[1].reviewer_id, "reviewer identities must differ")
            yield packet, (validated[0], validated[1])
        _require(first.read(1) == b"" and second.read(1) == b"", "review file has extra assessments or bytes")


def _decode_candidates(encoded: str) -> tuple[ReviewCandidate, ...]:
    raw = parse_strict_json_object(b'{"candidates":' + encoded.encode("utf-8") + b'}', label="canonical candidate payload")["candidates"]
    _require(isinstance(raw, list), "candidate payload must be a list")
    assert isinstance(raw, list)
    result = []
    for position, value in enumerate(raw):
        packets._guard(position)
        _require(isinstance(value, dict) and set(value) == {field.name for field in fields(ReviewCandidate)},
                 "candidate fields differ from original complete shape")
        _require(isinstance(value["spans"], list) and all(isinstance(span, dict)
                 and set(span) == {field.name for field in fields(EvidenceSpan)} for span in value["spans"]),
                 "candidate span fields differ from original complete shape")
        candidate = _CANDIDATE.validate_json(packets._json(value), strict=True)
        _require(packets._json(asdict(candidate)) == packets._json(value), "candidate strict JSON roundtrip changed payload")
        result.append(candidate)
    return tuple(result)


def _occurrences(connection: sqlite3.Connection, packet: BlindReviewPacket) -> tuple[CandidateOccurrence, ...]:
    results = []
    for position, version in enumerate(packet.versions):
        packets._guard(position)
        sizes = connection.execute("SELECT length(CAST(content_json AS BLOB)),length(CAST(candidates_json AS BLOB)) "
                                   "FROM versions WHERE version_id=?", (version.version_id,)).fetchall()
        _require(len(sizes) == 1 and all(type(size) is int and 0 < size <= packets.MAX_JSON_BYTES for size in sizes[0]),
                 "canonical candidate version missing, duplicated or exceeds metadata bound")
        row = connection.execute("SELECT cluster_id,security_id,source_family,content_json,candidates_json "
                                 "FROM versions WHERE version_id=?", (version.version_id,)).fetchone()
        assert row is not None
        cluster, security, family, encoded, candidates = row
        metadata = parse_strict_json_object(encoded.encode("utf-8"), label="candidate original metadata")
        _require((cluster, security, family) == (version.cluster_id, version.security_id, version.source_family)
                 and hashlib.sha256(encoded.encode("utf-8")).hexdigest() == version.content_json_sha256
                 and json_sha256(metadata) == version.metadata_sha256
                 and all(metadata.get(name) == getattr(version, name)
                         for name in ("source_id", "source_version_sha256", "text_sha256")),
                 "candidate source version differs from complete packet key")
        payload_sha256 = hashlib.sha256(candidates.encode("utf-8")).hexdigest()
        for index, candidate in enumerate(_decode_candidates(candidates)):
            if candidate.event_family == packet.event_family_to_assess:
                reference = json_sha256({"version_id": version.version_id, "candidate_payload_sha256": payload_sha256, "index": index})
                results.append(CandidateOccurrence(version, candidate, reference))
    return tuple(results)


def _render(root: Path, config: AnnotationIngestionConfig, request: dict[str, Any], folder: Path, samples: list[str],
            connection: sqlite3.Connection, artifacts: packets._Artifacts) -> dict[str, Any]:
    request_hash = artifacts.document("_request.json", request)
    counts = {"samples": 0, "version_assessments": 0, "candidate_occurrences": 0, "supported_occurrences": 0}

    def assessments() -> Iterator[bytes]:
        for _, reviewers in _reviews(root, config, folder, samples):
            counts["samples"] += 1
            for reviewer in reviewers:
                counts["version_assessments"] += len(reviewer.assessment.versions)
                yield packets._json(reviewer.assessment.model_dump(mode="json", by_alias=True)) + b"\n"

    artifacts.emit("assessments.jsonl", assessments())
    # A second bounded pass avoids retaining a corpus or introducing a second
    # artifact writer. The canonical candidate owner remains entered only once.
    def correspondences() -> Iterator[bytes]:
        candidate_ids: dict[str, str] = {}
        for packet, reviewers in _reviews(root, config, folder, samples):
            occurrences = _occurrences(connection, packet)
            for occurrence in occurrences:
                digest = json_sha256({"version_id": occurrence.version.version_id, "candidate": asdict(occurrence.candidate)})
                _require(candidate_ids.setdefault(occurrence.candidate.candidate_id, digest) == digest,
                         "candidate ID has conflicting payload across packets")
            results = correspond_candidates(occurrences=occurrences, reviewer_assessments=reviewers)
            counts["candidate_occurrences"] += len(results)
            counts["supported_occurrences"] += sum(item.correspondence.supported for item in results)
            yield packets._json({"sample_id": packet.sample_id, "event_family": packet.event_family_to_assess,
                "results": [_RESULT.dump_python(item, mode="json") for item in results], **CLOSED}) + b"\n"

    artifacts.emit("correspondence.jsonl", correspondences())
    return {"schema": SCHEMA, "status": "complete_correspondence_evidence_only", "request_sha256": request_hash,
            "artifacts": dict(sorted(artifacts.hashes.items())), "source_files": request["source_files"], "counts": counts, **CLOSED}


def _finish_checks(root: Path, files: dict[str, str], request: dict[str, Any], packet_folder: Path) -> None:
    _recheck(root, files)
    packets._verify_artifacts(packet_folder, packets._object(packet_folder / "_manifest.json"), include_manifest=True)
    _require(_implementations(root) == request["implementation_files"], "annotation executed implementation changed")


def publish_issuer_annotation_correspondence(*, root: Path, config: Path, expected_config_sha256: str,
                                            output: Path) -> dict[str, Any]:
    root = root.resolve()
    config_pin = SourcePin(path=_path(root, config).relative_to(root).as_posix(), sha256=expected_config_sha256)
    settings = _settings(root, config_pin)
    _reviewer_preflight(root, settings)
    verified = packets.verify_issuer_review_packets(root=root, publication=settings.packet_publication)
    destination = _path(root, output)
    _require(destination.parent == root / "data/research", "annotation output must be a new research directory")
    stage = destination.with_name("." + destination.name + ".private_stage")
    with heavy_job_lease("publish-issuer-annotations", runtime_dir=packets._runtime(root)):
        packets._guard()
        _require(not destination.exists() and not stage.exists(), "annotation output or private stage exists")
        request, files, folder, samples = _inputs(root, config_pin, settings, verified)
        for name in files:
            path = _path(root, name)
            _require(not any(path.is_relative_to(target) or target.is_relative_to(path) for target in (destination, stage)),
                     "annotation output overlaps source evidence")
        stage.mkdir()
        derivative_pin = SourcePin.model_validate(verified.request["settings"]["candidate_derivative"])
        with derivative._verified_candidate_connection(root=root, publication=derivative_pin) as connection:
            manifest = _render(root, settings, request, folder, samples, connection, packets._Artifacts(stage, verify=False))
        _finish_checks(root, files, request, folder)
        packets._verify_artifacts(stage, manifest, include_manifest=False)
        packets._Artifacts(stage, verify=False).document("_manifest.json", manifest)
        digest = packets._hash(stage / "_manifest.json")
        _finish_checks(root, files, request, folder)
        packets._verify_artifacts(stage, manifest, include_manifest=True)
        _require(packets._hash(stage / "_manifest.json") == digest, "annotation manifest changed before publication")
        packets._guard()
        _require(not destination.exists(), "annotation destination appeared before publication")
        stage.rename(destination)
    return {"status": manifest["status"], "manifest_sha256": digest, "counts": manifest["counts"], **CLOSED}


def verify_issuer_annotation_correspondence(*, root: Path, publication: SourcePin) -> VerifiedAnnotationCorrespondence:
    root = root.resolve()
    path = _path(root, publication.path)
    _require(path.is_file() and path.stat().st_size <= packets.MAX_JSON_BYTES,
             "annotation manifest missing or exceeds routing metadata bound")
    packets._pin(root, publication, {})
    _require(path.name == "_manifest.json", "annotation verification requires complete manifest")
    manifest = packets._object(path)
    request_path = path.parent / "_request.json"
    saved_request = packets._object(request_path)
    _require(packets._hash(request_path) == manifest["request_sha256"], "annotation request hash differs")
    config_pin = SourcePin.model_validate(saved_request["config"])
    settings = _settings(root, config_pin)
    _reviewer_preflight(root, settings)
    verified = packets.verify_issuer_review_packets(root=root, publication=settings.packet_publication)
    with heavy_job_lease("verify-issuer-annotations", runtime_dir=packets._runtime(root)):
        packets._guard()
        packets._pin(root, publication, {})
        packets._verify_artifacts(path.parent, manifest, include_manifest=True)
        request, files, folder, samples = _inputs(root, config_pin, settings, verified)
        _require(request == saved_request, "annotation request replay differs")
        derivative_pin = SourcePin.model_validate(verified.request["settings"]["candidate_derivative"])
        artifacts = packets._Artifacts(path.parent, verify=True)
        with derivative._verified_candidate_connection(root=root, publication=derivative_pin) as connection:
            expected = _render(root, settings, request, folder, samples, connection, artifacts)
            artifacts.document("_manifest.json", expected)
        _finish_checks(root, files, request, folder)
        packets._verify_artifacts(path.parent, expected, include_manifest=True)
        packets._pin(root, publication, files)
        for name, digest in expected["artifacts"].items():
            packets._pin(root, SourcePin(path=(path.parent / name).relative_to(root).as_posix(), sha256=digest), files)
    return VerifiedAnnotationCorrespondence(publication, files, expected["counts"])
