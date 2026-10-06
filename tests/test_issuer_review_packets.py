"""Tiny source fixtures; admission doubles exist only in this unit module."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.research import issuer_review_packet_contracts as contract
from market_predictor.research import issuer_review_packets as owner
from market_predictor.swing.contracts.holding_materialization import SourcePin


def _write(root: Path, name: str, value: Any) -> SourcePin:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(owner._json(value) + b"\n")
    return SourcePin(path=name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@pytest.fixture
def state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    database = tmp_path / "original/population.sqlite"
    database.parent.mkdir()
    connection = sqlite3.connect(database)
    connection.executescript("""
        CREATE TABLE versions(version_id TEXT PRIMARY KEY,cluster_id TEXT,security_id TEXT,
            source_family TEXT,year INTEGER,content_json TEXT,text TEXT,candidates_json TEXT);
        CREATE TABLE records(phase TEXT,ordinal INTEGER,cluster_id TEXT,version_id TEXT,record_json TEXT,
            PRIMARY KEY(phase,ordinal));
        CREATE TABLE aliases(proof_id TEXT PRIMARY KEY,security_id TEXT,cik TEXT,available_ns INTEGER,proof_json TEXT);
    """)
    clusters = {key: json_sha256(key) for key in
                ("old-a", "old-b", "d1", "d2", "d3", "d4", "fresh-a", "fresh-b", "fresh-c", "unknown")}
    version_count = 0

    def add(cluster: str, family: str, source: str, candidates: tuple[str, ...] = (),
            *, unavailable: bool = False, future: bool = False, unknown: bool = False) -> None:
        nonlocal version_count
        version_count += 1
        text = None if unavailable else "Example announces € guidance 🧪. Full exhibit retained."
        source_hash = json_sha256([source, version_count])
        version = json_sha256([clusters[cluster], source, source_hash])
        security = None if unknown else "issuer-" + cluster
        metadata = dict(source_id=source, source_version_sha256=source_hash, security_id=security,
            source_family=family, ticker="UNIT", published_at_utc="2021-01-05T20:00:00+00:00",
            version_available_at_utc="2025-01-05T20:00:00+00:00" if future else "2021-01-05T20:00:00+00:00",
            event_available_at_utc="2021-01-05T20:00:00+00:00", identity_available_at_utc="2021-01-04T20:00:00+00:00",
            first_seen_at_utc="2026-09-01T20:00:00+00:00", identity_authority_sha256="a" * 64,
            availability_semantics="historical_proxy", source_locator="unit://" + source,
            content_kind="html", encoding="utf-8", alias_proof={"aliases": ["Example"]},
            text_sha256=None if text is None else hashlib.sha256(text.encode()).hexdigest(),
            unavailable_reasons=["source_text_unavailable"] if unavailable else [])
        proposed = [{"event_family": item, "rule_id": "unit-rule", "candidate_id": "hidden-candidate"}
                    for item in candidates]
        connection.execute("INSERT INTO versions VALUES(?,?,?,?,?,?,?,?)",
            (version, clusters[cluster], security, family, 2021, json.dumps(metadata), text, json.dumps(proposed)))
        connection.execute("INSERT INTO records VALUES(?,?,?,?,?)",
            ("sources", version_count, clusters[cluster], version, json.dumps({"source": source})))

    add("old-a", *contract.DEVELOPMENT_SOURCES[0], ("earnings",))
    add("old-b", "sec", "old-source", ("guidance",))
    for index, (family, source) in enumerate(contract.DEVELOPMENT_SOURCES[1:], 1):
        add("d" + str(index), family, source, ("earnings", "guidance"))
    add("fresh-a", "sec", "cover", ("earnings",))
    add("fresh-a", "sec", "exhibit", ("guidance",))
    add("fresh-a", "sec", "future", future=True)
    add("fresh-a", "sec", "unreadable", unavailable=True)
    add("fresh-a", "sec", "unknown-copy", unknown=True)
    add("fresh-b", "alpaca", "new-guidance", ("guidance",))
    add("fresh-c", "sec", "negative")
    add("unknown", "alpaca", "unknown", unknown=True)
    connection.execute("INSERT INTO records VALUES(?,?,?,?,?)", ("index", 1, "", "", "{}"))
    connection.commit()
    connection.close()
    samples = [{"sample_id": json_sha256(index), "cluster_id": clusters[key], "event_family": family, "role": role}
               for index, (key, family, role) in enumerate((("old-a", "earnings", "candidate"),
                   ("old-a", "guidance", "noncandidate"), ("old-b", "earnings", "noncandidate")))]
    sample = _write(tmp_path, "original/sample.json", {"schema": owner.population.SCHEMA + "_sample",
        "policy_sha256": owner.POLICY_SHA256, "samples": samples})
    parent = _write(tmp_path, "original/_manifest.json", {"artifacts": {"sample.json": sample.sha256},
        "implementation_files": {"original/producer.py": "b" * 64}})
    reviewers = [_write(tmp_path, f"original/reviewer-{name}.json", {
        "schema": "market_predictor.issuer_content_reviews", "population_authority": parent.model_dump(mode="json"),
        "sample_sha256": sample.sha256, "reviewer_id": name, "reviewer_kind": "human", "reviewer_model": None,
        "evidence_scope": "source_only", "independent_review": True,
        "reviews": [{"sample_id": item["sample_id"], "old_label_not_new_truth": "NEVER_DISPLAY"} for item in samples]})
        for name in ("a", "b")]
    request = _write(tmp_path, "candidate/_request.json", {"parent_population": parent.model_dump(mode="json"),
        "source_files": {parent.path: parent.sha256, sample.path: sample.sha256,
            "original/population.sqlite": hashlib.sha256(database.read_bytes()).hexdigest()},
        "implementation_files": {}, "candidate_policy_sha256": "c" * 64})
    derivative = _write(tmp_path, "candidate/_manifest.json", {"request_sha256": request.sha256,
        "artifacts": {"_request.json": request.sha256}})
    monkeypatch.setattr(contract, "ORIGINAL_POPULATION", parent.sha256)
    monkeypatch.setattr(contract, "CANDIDATE_DERIVATIVE", derivative.sha256)
    monkeypatch.setattr(contract, "ORIGINAL_SAMPLE", sample.sha256)
    monkeypatch.setattr(contract, "ORIGINAL_REVIEWERS", frozenset(pin.sha256 for pin in reviewers))
    config = _write(tmp_path, "config.json", {"schema": "market_predictor.issuer_review_packet_config",
        "original_population": parent.model_dump(mode="json"), "candidate_derivative": derivative.model_dump(mode="json"),
        "original_sample": sample.model_dump(mode="json"), "original_reviewers": [pin.model_dump(mode="json") for pin in reviewers]})
    (tmp_path / "data/research").mkdir(parents=True)
    monkeypatch.setattr(owner, "ORIGINAL_SAMPLE_ENTRIES", 3)
    monkeypatch.setattr(owner, "ORIGINAL_SAMPLE_CLUSTERS", 2)
    monkeypatch.setattr(owner.derivative, "ORIGINAL_VERSION_COUNT", version_count)
    monkeypatch.setattr(owner, "_guard", lambda position=0: None)
    monkeypatch.setattr(owner.derivative, "_guard", lambda: None)
    monkeypatch.setattr(owner, "_implementations", lambda root: {})
    calls = {"lease": 0, "connection": 0, "exited": 0}

    @contextmanager
    def lease(*args: Any, **kwargs: Any) -> Any:
        calls["lease"] += 1
        yield

    @contextmanager
    def admitted_unit_connection(**kwargs: Any) -> Any:
        # Only this tiny unit fixture doubles the full retained-data admission.
        calls["connection"] += 1
        original = sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)
        original.execute("PRAGMA query_only=ON")
        try:
            yield original
        finally:
            original.close()
            calls["exited"] += 1

    monkeypatch.setattr(owner, "heavy_job_lease", lease)
    monkeypatch.setattr(owner.derivative, "_verified_candidate_connection", admitted_unit_connection)
    return {"root": tmp_path, "config": config, "output": tmp_path / "data/research/packets", "database": database,
        "clusters": clusters, "calls": calls, "reviewers": reviewers, "versions": version_count,
        "connection": admitted_unit_connection}


def _publish(state: dict[str, Any]) -> tuple[dict[str, Any], SourcePin]:
    result = owner.publish_issuer_review_packets(root=state["root"], config=Path(state["config"].path),
        expected_config_sha256=state["config"].sha256, output=state["output"])
    return result, SourcePin(path="data/research/packets/_manifest.json", sha256=result["manifest_sha256"])


def test_complete_inventory_fresh_frame_and_independent_reproduction(state: dict[str, Any]) -> None:
    result, pin = _publish(state)
    verified = owner.verify_issuer_review_packets(root=state["root"], publication=pin)
    counts = verified.counts
    assert counts["inventory"]["versions"] == state["versions"] == 14
    assert counts["inventory"]["records"] == 15
    assert counts["inventory"]["records_without_version"] == 1
    assert counts["excluded_clusters"] == 6
    assert counts["U"]["clusters"] == 9
    assert counts["F"]["clusters"] == 3
    assert counts["sample_entries"] == 6
    assert counts["sample_shortages"]["earnings/candidate"] == 299
    assert state["calls"] == {"lease": 2, "connection": 2, "exited": 2}
    assert all(result[key] is False for key in contract.CLOSED)
    samples = json.loads((state["output"] / "sample.json").read_text())["samples"]
    for sample in samples:
        assert sample["cluster_id"] in {state["clusters"][key] for key in ("fresh-a", "fresh-b", "fresh-c")}
        assert sample["inclusion_probability"] == sample["sampled_clusters"] / sample["population_clusters"]


def test_blind_packet_retains_all_versions_exact_unicode_and_only_physical_status(state: dict[str, Any]) -> None:
    _publish(state)
    packets = [json.loads(path.read_text(encoding="utf-8")) for path in (state["output"] / "blind").glob("*.json")]
    packet = next(item for item in packets if item["cluster_id"] == state["clusters"]["fresh-a"])
    assert packet["version_count"] == packet["occurrence_count"] == 5
    versions = {item["source_id"]: item for item in packet["versions"]}
    assert versions["cover"]["text"] == "Example announces € guidance 🧪. Full exhibit retained."
    assert versions["cover"]["first_seen_at_utc"].startswith("2026")
    for source in ("future", "unknown-copy", "unreadable"):
        assert versions[source]["text"] is None
        assert versions[source]["omission_reasons"]
    assert versions["future"]["text_sha256"] is not None
    assert versions["unreadable"]["text_sha256"] is None
    encoded = json.dumps(packet)
    for secret in ("hidden-candidate", "unit-rule", "NEVER_DISPLAY", '"role"', '"candidates_json"', '"disposition"'):
        assert secret not in encoded


def test_development_union_deduplicates_old_entries_and_expands_sources(state: dict[str, Any]) -> None:
    _publish(state)
    exclusions = json.loads((state["output"] / "exclusions.json").read_text())
    assert exclusions["original_sample_entries"] == 3
    assert exclusions["original_sample_distinct_clusters"] == 2
    assert len(exclusions["development_resolutions"]) == 5
    prior = next(item for item in exclusions["clusters"] if item["cluster_id"] == state["clusters"]["old-a"])
    assert len(prior["reasons"]) == 2
    assert prior["fresh_inclusion_probability"] == 0


@pytest.mark.parametrize("artifact", ["frame.jsonl", "sample.json", "exclusions.json", "_manifest.json", "blind"])
def test_rehashed_output_poison_still_fails_exact_reproduction(state: dict[str, Any], artifact: str) -> None:
    _publish(state)
    folder = state["output"]
    manifest = json.loads((folder / "_manifest.json").read_text())
    if artifact == "_manifest.json":
        manifest["counts"]["F"]["clusters"] += 1
    else:
        if artifact == "blind":
            artifact = next((folder / "blind").glob("*.json")).relative_to(folder).as_posix()
        path = folder / artifact
        path.write_bytes(path.read_bytes() + b" ")
        manifest["artifacts"][artifact] = hashlib.sha256(path.read_bytes()).hexdigest()
    pin = _write(state["root"], "data/research/packets/_manifest.json", manifest)
    with pytest.raises(DataReadinessError):
        owner.verify_issuer_review_packets(root=state["root"], publication=pin)


def test_unlisted_output_rejected(state: dict[str, Any]) -> None:
    _, pin = _publish(state)
    (state["output"] / "unlisted.json").write_text("{}")
    with pytest.raises(DataReadinessError, match="extra or missing"):
        owner.verify_issuer_review_packets(root=state["root"], publication=pin)


@pytest.mark.parametrize("mutation", ["context_exit", "source", "destination"])
def test_exit_failures_never_publish_or_overwrite(state: dict[str, Any], monkeypatch: pytest.MonkeyPatch, mutation: str) -> None:
    @contextmanager
    def poisoned(**kwargs: Any) -> Any:
        with state["connection"](**kwargs) as connection:
            yield connection
        if mutation == "context_exit":
            raise DataReadinessError("source context exit failed")
        if mutation == "source":
            (state["root"] / state["reviewers"][0].path).write_text("changed")
        else:
            state["output"].mkdir()
            (state["output"] / "sentinel").write_text("preserve")

    monkeypatch.setattr(owner.derivative, "_verified_candidate_connection", poisoned)
    with pytest.raises(DataReadinessError):
        _publish(state)
    assert not (state["output"] / "_manifest.json").exists()
    stage = state["output"].with_name(".packets.private_stage")
    assert stage.exists()
    if mutation != "destination":
        assert not (stage / "_manifest.json").exists()
    else:
        assert (state["output"] / "sentinel").read_text() == "preserve"


def test_missing_development_identity_rejected(state: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract, "DEVELOPMENT_SOURCES", (*contract.DEVELOPMENT_SOURCES, ("sec", "missing")))
    with pytest.raises(DataReadinessError, match="development source ID"):
        _publish(state)
    assert not state["output"].exists()


def test_config_forbids_extra_fields_and_wrong_frozen_pins(state: dict[str, Any]) -> None:
    payload = json.loads((state["root"] / state["config"].path).read_text())
    with pytest.raises(ValidationError):
        contract.IssuerReviewPacketConfig.model_validate(payload | {"skip_verification": True})
    payload["original_population"]["sha256"] = "f" * 64
    with pytest.raises(ValidationError):
        contract.IssuerReviewPacketConfig.model_validate(payload)


def test_existing_private_stage_is_preserved(state: dict[str, Any]) -> None:
    stage = state["output"].with_name(".packets.private_stage")
    stage.mkdir()
    (stage / "sentinel").write_text("preserve")
    with pytest.raises(DataReadinessError, match="already exists"):
        _publish(state)
    assert (stage / "sentinel").read_text() == "preserve"
    assert state["calls"]["connection"] == 0


def test_unknown_development_cluster_stays_outside_u(state: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(contract, "DEVELOPMENT_SOURCES", (*contract.DEVELOPMENT_SOURCES, ("alpaca", "unknown")))
    result, _ = _publish(state)
    assert result["counts"]["excluded_outside_U"] == 1
    assert result["counts"]["F"]["clusters"] == 3
    exclusions = json.loads((state["output"] / "exclusions.json").read_text())
    assert exclusions["outside_U"] == [state["clusters"]["unknown"]]


def test_original_occurrence_cannot_reassign_version_to_other_cluster(state: dict[str, Any]) -> None:
    with sqlite3.connect(state["database"]) as connection:
        connection.execute("UPDATE records SET cluster_id=? WHERE phase='sources' AND ordinal=1",
                           (state["clusters"]["fresh-c"],))
        with pytest.raises(DataReadinessError, match="version/cluster owner"):
            owner._inventories(connection)


def test_single_occurrence_pass_preserves_original_query_bytes_for_both_families(state: dict[str, Any]) -> None:
    with sqlite3.connect(state["database"]) as connection:
        cluster = state["clusters"]["fresh-a"]
        version, = connection.execute("SELECT version_id FROM versions WHERE cluster_id=? LIMIT 1", (cluster,)).fetchone()
        for phase, ordinal, key, identity in (("before", 2, cluster, version),
            ("before", 1, state["clusters"]["old-a"], ""), ("zz", 1, cluster, "")):
            connection.execute("INSERT INTO records VALUES(?,?,?,?,?)",
                (phase, ordinal, key, identity, '{"source":"€ query copy 🧪"}'))
        clusters = owner.population._clusters(connection)
        samples = [item for item in owner.select_content_review_sample(clusters) if item.cluster_id == cluster]
        assert {item.event_family for item in samples} == {"earnings", "guidance"}
        expected = [{"phase": phase, "ordinal": ordinal, "version_id": identity,
            "record_json_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest()}
            for phase, ordinal, identity, encoded in connection.execute(
                "SELECT phase,ordinal,version_id,record_json FROM records WHERE cluster_id=? ORDER BY phase,ordinal", (cluster,))]
        queries: list[str] = []
        connection.set_trace_callback(queries.append)
        retained = owner._sample_occurrences(connection, samples)
        assert set(retained) == {cluster}
        assert retained[cluster] == expected
        expected_bytes = b"".join(owner._json(item) + b"\n" for item in expected)
        for sample in samples:
            packet = json.loads(b"".join(owner._packet(connection, sample, {}, Counter(), retained[cluster])))
            assert packet["occurrences"] == expected
            assert packet["occurrence_count"] == len(expected) == 7
            assert packet["occurrence_inventory_sha256"] == hashlib.sha256(expected_bytes).hexdigest()
        record_queries = [query for query in queries if "FROM records" in query]
        assert record_queries == [
            "SELECT phase,ordinal,cluster_id,version_id,record_json FROM records ORDER BY phase,ordinal"]


def test_blind_reader_rejects_extraction_status_as_physical_reason(state: dict[str, Any]) -> None:
    with sqlite3.connect(state["database"]) as connection:
        row = connection.execute("SELECT version_id,cluster_id,security_id,source_family,content_json FROM versions "
                                 "WHERE cluster_id=? LIMIT 1", (state["clusters"]["fresh-b"],)).fetchone()
        assert row is not None
        metadata = json.loads(row[-1])
        metadata["unavailable_reasons"] = ["guidance_direction_unresolved"]
        with pytest.raises(DataReadinessError, match="extractor status"):
            owner._packet_version(connection, (*row[:-1], json.dumps(metadata)))
