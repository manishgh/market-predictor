"""Tiny UNIT doubles for physical admission; no actual review labels or source data."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytest

from market_predictor.catalysts.issuer_events.content_review import EvidenceSpan, ReviewCandidate
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import HeavyJobBusyError
from market_predictor.research import issuer_annotation_ingestion as owner
from market_predictor.research.issuer_review_packet_contracts import CLOSED, VerifiedIssuerReviewPackets, policy_identities
from market_predictor.research.issuer_source_annotations import VersionKey, _encoded
from market_predictor.swing.contracts.holding_materialization import SourcePin
from tests.test_issuer_source_annotations import _fixture


def _write(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(owner.packets._json(value) + b"\n")
    return owner.packets._hash(path)


def _pin(root: Path, path: Path) -> SourcePin:
    return SourcePin(path=path.relative_to(root).as_posix(), sha256=owner.packets._hash(path))


@pytest.fixture
def unit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path
    packet, assessment = _fixture()
    folder = root / "data/research/synthetic_packets"
    folder.mkdir(parents=True)
    database = root / "synthetic_candidates.sqlite"
    versions = []
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE versions(version_id TEXT,cluster_id TEXT,security_id TEXT,source_family TEXT,"
                           "content_json TEXT,candidates_json TEXT)")
        for index, version in enumerate(packet.versions):
            metadata = {name: getattr(version, name) for name in ("source_id", "source_version_sha256", "text_sha256")}
            encoded = json.dumps(metadata, sort_keys=True)
            changed = version.model_copy(update={"content_json_sha256": hashlib.sha256(encoded.encode()).hexdigest(),
                                                  "metadata_sha256": json_sha256(metadata)})
            versions.append(changed)
            candidates = []
            if index == 0:
                event = assessment.versions[0].annotations[0]
                spans = tuple(EvidenceSpan(role, span.start, span.end, span.quote, ("unit://physical",))
                    for role, span in (("statement", event.anchor), ("issuer", event.issuer), ("action", event.action),
                                      ("result", event.result), ("fiscal_period", event.fiscal_period)))
                candidates = [asdict(ReviewCandidate("unit-candidate", "guidance", "raises", "unit-rule", "FY2021", (), spans))]
            connection.execute("INSERT INTO versions VALUES(?,?,?,?,?,?)", (version.version_id, version.cluster_id,
                version.security_id, version.source_family, encoded, owner.packets._json(candidates).decode()))
    derivative_pin = _pin(root, database)
    request = {"implementation_files": {}, "settings": {"candidate_derivative": derivative_pin.model_dump(mode="json")},
               "frame_policy": deepcopy(owner.packets.FRAME_POLICY),
               "correspondence_policy": deepcopy(owner.packets.CORRESPONDENCE_POLICY), **CLOSED}
    artifacts = {"_request.json": _write(folder / "_request.json", request),
                 "sample.json": _write(folder / "sample.json", {"samples": [{"sample_id": packet.sample_id}]}),
                 "frame.jsonl": _write(folder / "frame.jsonl", {"synthetic_unit_only": True}),
                 "exclusions.json": _write(folder / "exclusions.json", {"synthetic_unit_only": True})}
    inventory = hashlib.sha256(b"".join(_encoded(version.model_dump(mode="json", exclude={"text"})) + b"\n"
                                       for version in versions)).hexdigest()
    changes = dict(request_sha256=artifacts["_request.json"], sample_sha256=artifacts["sample.json"],
                   frame_sha256=artifacts["frame.jsonl"], exclusions_sha256=artifacts["exclusions.json"],
                   version_inventory_sha256=inventory)
    packet = packet.model_copy(update={**changes, "versions": tuple(versions)})
    name = f"blind/{packet.sample_id}.json"
    artifacts[name] = _write(folder / name, packet.model_dump(mode="json", by_alias=True))
    manifest = {"artifacts": artifacts, "request_sha256": artifacts["_request.json"], **CLOSED}
    _write(folder / "_manifest.json", manifest)
    publication = _pin(root, folder / "_manifest.json")
    monkeypatch.setattr(owner, "FROZEN_PACKET_SHA256", publication.sha256)
    reviewers = []
    for index in range(2):
        rows = []
        for version, old in zip(versions, assessment.versions, strict=True):
            key = VersionKey.from_version(version)
            annotations = tuple(event.model_copy(update={"version": key}) for event in old.annotations)
            rows.append(old.model_copy(update={"version": key, "annotations": annotations}))
        review = assessment.model_copy(update={**changes, "packet_publication": publication, "versions": tuple(rows),
            "reviewer": assessment.reviewer.model_copy(update={"reviewer_id": f"synthetic-reviewer-{index}"})})
        path = root / f"synthetic_reviewer_{index}.jsonl"
        _write(path, review.model_dump(mode="json", by_alias=True))
        reviewers.append(_pin(root, path))
    config = root / "configs/synthetic_ingestion.json"
    _write(config, {"schema": "market_predictor.issuer_annotation_ingestion_config",
        "packet_publication": publication.model_dump(mode="json"),
        "reviewer_files": [pin.model_dump(mode="json") for pin in reviewers], **policy_identities()})
    verified = VerifiedIssuerReviewPackets(publication, request, {}, {derivative_pin.path: derivative_pin.sha256},
        packet.frame_sha256, packet.exclusions_sha256, packet.sample_sha256)
    events: list[str] = []
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(root / "runtime"))
    monkeypatch.setattr(owner.packets, "_guard", lambda position=0: None)
    monkeypatch.setattr(owner, "_implementations", lambda root: {})

    def packet_verifier(**kwargs: Any) -> Any:
        with owner.heavy_job_lease("synthetic-public-verifier", runtime_dir=root / "runtime"):
            events.append("packet_verified")
        return verified

    @contextmanager
    def canonical_candidates(**kwargs: Any) -> Any:
        assert kwargs == {"root": root, "publication": derivative_pin}
        events.append("candidate_enter")
        with sqlite3.connect(database) as connection:
            yield connection
        events.append("candidate_exit")

    monkeypatch.setattr(owner.packets, "verify_issuer_review_packets", packet_verifier)
    monkeypatch.setattr(owner.derivative, "_verified_candidate_connection", canonical_candidates)
    return dict(root=root, config=config, packet=packet, publication=publication, folder=folder,
                reviewers=reviewers, database=database, events=events, verified=verified,
                output=root / "data/research/synthetic_correspondence")


def _publish(unit: dict[str, Any]) -> dict[str, Any]:
    return owner.publish_issuer_annotation_correspondence(root=unit["root"], config=unit["config"],
        expected_config_sha256=owner.packets._hash(unit["config"]), output=unit["output"])


def test_synthetic_publish_independent_byte_replay_and_closed_flags(unit: dict[str, Any]) -> None:
    result = _publish(unit)
    assert result["counts"] == {"samples": 1, "version_assessments": 10, "candidate_occurrences": 1, "supported_occurrences": 1}
    assert unit["events"] == ["packet_verified", "candidate_enter", "candidate_exit"]
    verified = owner.verify_issuer_annotation_correspondence(
        root=unit["root"], publication=_pin(unit["root"], unit["output"] / "_manifest.json"))
    assert verified.counts == result["counts"] and verified.qualification_established is False
    assert all(result[name] is False for name in CLOSED)
    assert len((unit["output"] / "assessments.jsonl").read_bytes().splitlines()) == 2


@pytest.mark.parametrize("mode", ["missing", "extra", "wrong_sample", "same_reviewer", "unknown_field", "forged_packet"])
def test_review_physical_coverage_and_identity_reject(unit: dict[str, Any], mode: str) -> None:
    path = unit["root"] / unit["reviewers"][1].path
    value = json.loads(path.read_text())
    if mode == "wrong_sample":
        value["sample_id"] = "0" * 64
    elif mode == "same_reviewer":
        value["reviewer"]["reviewer_id"] = "synthetic-reviewer-0"
    elif mode == "unknown_field":
        value["invented"] = True
    elif mode == "forged_packet":
        value["packet_publication"]["sha256"] = "0" * 64
    data = owner.packets._json(value) + b"\n"
    path.write_bytes(b"" if mode == "missing" else data * (2 if mode == "extra" else 1))
    config = json.loads(unit["config"].read_text())
    config["reviewer_files"][1] = _pin(unit["root"], path).model_dump(mode="json")
    _write(unit["config"], config)
    with pytest.raises((DataReadinessError, ValueError)):
        _publish(unit)
    assert not unit["output"].exists()


@pytest.mark.parametrize("field", ["training_eligible", "spans", "unexpected"])
def test_candidate_shape_never_drops_fields_or_fills_defaults(unit: dict[str, Any], field: str) -> None:
    with sqlite3.connect(unit["database"]) as connection:
        encoded = connection.execute("SELECT candidates_json FROM versions ORDER BY version_id LIMIT 1").fetchone()[0]
    raw = json.loads(encoded)
    if field == "unexpected":
        raw[0][field] = False
    else:
        del raw[0][field]
    with pytest.raises(DataReadinessError, match="fields differ"):
        owner._decode_candidates(json.dumps(raw))


@pytest.mark.parametrize("target", ["config", "database", "blind"])
def test_mutation_between_sequential_leases_rejects(unit: dict[str, Any], monkeypatch: pytest.MonkeyPatch, target: str) -> None:
    original = owner.packets.verify_issuer_review_packets
    def mutate(**kwargs: Any) -> Any:
        verified = original(**kwargs)
        path = unit[target] if target != "blind" else unit["folder"] / "blind" / f"{unit['packet'].sample_id}.json"
        path.write_bytes(path.read_bytes() + b" ")
        return verified
    monkeypatch.setattr(owner.packets, "verify_issuer_review_packets", mutate)
    with pytest.raises(DataReadinessError):
        _publish(unit)
    assert not unit["output"].exists()


def test_source_mutation_at_candidate_context_exit_never_publishes(unit: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = owner.derivative._verified_candidate_connection
    @contextmanager
    def mutate(**kwargs: Any) -> Any:
        with original(**kwargs) as connection:
            yield connection
        unit["database"].write_bytes(unit["database"].read_bytes() + b"mutation")
    monkeypatch.setattr(owner.derivative, "_verified_candidate_connection", mutate)
    with pytest.raises(DataReadinessError, match="changed"):
        _publish(unit)
    assert not unit["output"].exists()
    assert not (unit["output"].with_name("." + unit["output"].name + ".private_stage") / "_manifest.json").exists()


def test_same_physical_reviewer_identity_rejects_before_replay(
    unit: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Explicit UNIT filesystem-identity double: this does not claim the sandbox
    # permits creating hardlinks. Both input paths are ordinary existing files.
    paths = tuple(unit["root"] / pin.path for pin in unit["reviewers"])
    calls = []
    def same_file(first: Path, second: Path) -> bool:
        calls.append((first, second))
        return (first, second) == paths
    monkeypatch.setattr(owner.os.path, "samefile", same_file)
    with pytest.raises(DataReadinessError, match="same physical"):
        _publish(unit)
    assert calls == [paths] and unit["events"] == []


def test_missing_physical_review_fails_before_public_packet_replay(unit: dict[str, Any]) -> None:
    config = json.loads(unit["config"].read_text())
    config["reviewer_files"][1]["path"] = "not_a_real_reviewer.jsonl"
    _write(unit["config"], config)
    with pytest.raises(DataReadinessError, match="physical reviewer files"):
        _publish(unit)
    assert unit["events"] == [] and not unit["output"].exists()


def test_busy_processing_lease_prevents_candidate_reads(unit: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(owner.packets, "verify_issuer_review_packets", lambda **kwargs: unit["verified"])
    with owner.heavy_job_lease("synthetic-competing-job", runtime_dir=unit["root"] / "runtime"):
        with pytest.raises(HeavyJobBusyError):
            _publish(unit)
    assert unit["events"] == [] and not unit["output"].exists()


def test_busy_verifier_never_reads_bulk_evidence_before_processing_lease(
    unit: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    _publish(unit)
    publication = _pin(unit["root"], unit["output"] / "_manifest.json")
    monkeypatch.setattr(owner.packets, "verify_issuer_review_packets", lambda **kwargs: unit["verified"])
    reads = []
    original_hash = owner.packets._hash

    def record_hash(path: Path) -> str:
        reads.append(path.name)
        return original_hash(path)

    monkeypatch.setattr(owner.packets, "_hash", record_hash)
    with owner.heavy_job_lease("synthetic-competing-job", runtime_dir=unit["root"] / "runtime"):
        with pytest.raises(HeavyJobBusyError):
            owner.verify_issuer_annotation_correspondence(root=unit["root"], publication=publication)
    assert "assessments.jsonl" not in reads and "correspondence.jsonl" not in reads


@pytest.mark.parametrize("mode", ["extra_file", "rehash_correspondence", "forged_counts"])
def test_verifier_replays_complete_inventory_and_bytes(unit: dict[str, Any], mode: str) -> None:
    _publish(unit)
    path = unit["output"] / "_manifest.json"
    manifest = json.loads(path.read_text())
    if mode == "extra_file":
        (unit["output"] / "extra.json").write_text("{}")
    elif mode == "rehash_correspondence":
        target = unit["output"] / "correspondence.jsonl"
        target.write_bytes(target.read_bytes().replace(b'"supported":true', b'"supported":false'))
        manifest["artifacts"]["correspondence.jsonl"] = owner.packets._hash(target)
    else:
        manifest["counts"]["supported_occurrences"] = 0
    _write(path, manifest)
    with pytest.raises(DataReadinessError):
        owner.verify_issuer_annotation_correspondence(root=unit["root"], publication=_pin(unit["root"], path))


@pytest.mark.parametrize("change", ["boolean_to_number", "array_order"])
def test_packet_request_rejects_wire_value_changes(unit: dict[str, Any], change: str) -> None:
    # UNIT only: emulate a canonical verifier returning changed policy values.
    request = unit["verified"].request
    if change == "boolean_to_number":
        request["training_eligible"] = 0
    else:
        request["correspondence_policy"]["roles"] = tuple(reversed(request["correspondence_policy"]["roles"]))
    with pytest.raises(DataReadinessError, match="verified packet request differs"):
        _publish(unit)
    assert not unit["output"].exists()
    assert not unit["output"].with_name("." + unit["output"].name + ".private_stage").exists()
    assert unit["events"] == ["packet_verified"]
