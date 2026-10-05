"""Synthetic unit cases only; no retained population or operational job is opened."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.research import issuer_candidate_derivative as derivative
from market_predictor.swing.contracts.holding_materialization import SourcePin


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return digest(path)


def version(number: int, *, unavailable: bool = False, future: bool = False) -> tuple[Any, ...]:
    security = None if unavailable else "cik:0000000001"
    source = f"story-{number}"
    cluster = json_sha256(["alpaca", source, security])
    source_hash = hashlib.sha256(source.encode()).hexdigest()
    identity = json_sha256([cluster, source, source_hash])
    text = None if future else "Acme Corporation reports Q1 2020 financial results with revenue of $2 million."
    clock = "2024-06-01T10:00:00+00:00" if future else "2020-01-30T10:00:00+00:00"
    metadata = {
        "source_family": "alpaca", "source_id": source, "source_version_sha256": source_hash,
        "security_id": security, "ticker": "ACME", "published_at_utc": clock,
        "version_available_at_utc": clock, "event_available_at_utc": clock,
        "first_seen_at_utc": "2026-10-01T10:00:00+00:00",
        "identity_available_at_utc": "2019-01-01T00:00:00+00:00", "identity_authority_sha256": "c" * 64,
        "availability_semantics": "historical_proxy", "source_locator": "unit-source",
        "content_kind": "provider_body", "encoding": "utf-8",
        "alias_proof": {"aliases": ["Acme Corporation"], "available_at_utc": "2019-01-01T00:00:00+00:00"},
        "source_metadata": {"available_at_utc": clock},
        "text_sha256": None if text is None else hashlib.sha256(text.encode()).hexdigest(),
        "unavailable_reasons": ["version_after_initial_fit_cutoff"] if future else ["missing_cohort_identity"] if unavailable else [],
    }
    return identity, cluster, security, "alpaca", 2024 if future else 2020, json.dumps(metadata, sort_keys=True), text, "[]"


@pytest.fixture
def parent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, SourcePin]:
    """Only this unit fixture replaces the production original-authority identity."""
    folder = tmp_path / "data/research/original"
    folder.mkdir(parents=True)
    with sqlite3.connect(folder / "population.sqlite") as db:
        db.execute("CREATE TABLE versions (version_id TEXT PRIMARY KEY,cluster_id TEXT,security_id TEXT,"
                   "source_family TEXT,year INTEGER,content_json TEXT,text TEXT,candidates_json TEXT)")
        db.execute("CREATE TABLE records (ordinal INTEGER,phase TEXT,cluster_id TEXT,version_id TEXT,record_json TEXT)")
        db.execute("CREATE TABLE aliases (proof_id TEXT PRIMARY KEY,security_id TEXT,cik TEXT,available_ns INTEGER,proof_json TEXT)")
        rows = [version(1), version(2, unavailable=True), version(3, future=True)]
        db.executemany("INSERT INTO versions VALUES (?,?,?,?,?,?,?,?)", rows)
        db.execute("INSERT INTO records VALUES (0,'alpaca',?,?, '{}')", (rows[0][1], rows[0][0]))
        db.execute("INSERT INTO aliases VALUES ('proof','cik:0000000001','0000000001',0,'{}')")
    source = tmp_path / "data/source.json"
    source_hash = write(source, {"synthetic_unit_source": True})
    producer = {f"src/market_predictor/original_{i}.py": f"{i:064x}" for i in range(20)}
    request_hash = write(folder / "_request.json", {
        "schema": derivative.PARENT_SCHEMA + "_request", "extraction_policy_sha256": "b" * 64,
        "implementation_files": producer,
    })
    sample_hash = write(folder / "sample.json", {"unit_fixture": True})
    manifest_hash = write(folder / "_manifest.json", {
        "schema": derivative.PARENT_SCHEMA, "status": "complete_review_population_only",
        "training_eligible": False, "serving_eligible": False, "promotion_eligible": False,
        "implementation_files": producer, "source_files": {source.relative_to(tmp_path).as_posix(): source_hash},
        "request_sha256": request_hash, "totals": {"source_versions": 3},
        "artifacts": {"population.sqlite": digest(folder / "population.sqlite"), "_request.json": request_hash,
                      "sample.json": sample_hash},
    })
    code = tmp_path / "src/current.py"
    code.parent.mkdir()
    code.write_text("# synthetic unit implementation pin\n", encoding="utf-8")
    monkeypatch.setattr(derivative, "ORIGINAL_MANIFEST_SHA256", manifest_hash)
    monkeypatch.setattr(derivative, "ORIGINAL_VERSION_COUNT", 3)
    monkeypatch.setattr(derivative, "_guard", lambda: None)
    monkeypatch.setattr(derivative, "_runtime", lambda root: root / "runtime")
    monkeypatch.setattr(derivative, "_implementations", lambda root: {"src/current.py": digest(root / "src/current.py")})
    return tmp_path, SourcePin(path="data/research/original/_manifest.json", sha256=manifest_hash)


def publish(parent: tuple[Path, SourcePin]) -> tuple[Path, SourcePin]:
    root, original = parent
    output = Path("data/research/derivative")
    result = derivative.publish_issuer_candidate_derivative(root=root, parent_population=original, output=output)
    return root, SourcePin(path=(output / "_manifest.json").as_posix(), sha256=result["manifest_sha256"])


def rehash_sidecar(root: Path, publication: SourcePin) -> SourcePin:
    path = root / publication.path
    manifest = json.loads(path.read_text())
    manifest["artifacts"]["candidates.sqlite"] = digest(path.parent / "candidates.sqlite")
    return SourcePin(path=publication.path, sha256=write(path, manifest))


def test_unit_roundtrip_replays_all_versions_and_keeps_original_readonly(parent: tuple[Path, SourcePin]) -> None:
    root, original = parent
    original_db = root / Path(original.path).parent / "population.sqlite"
    before = digest(original_db)
    root, publication = publish(parent)
    checked = derivative.verify_issuer_candidate_derivative(root=root, publication=publication)
    assert checked.counts["versions"] == 3
    assert checked.counts["preserved_unavailable"] == 2
    assert len(checked.original_producer_implementation_files) == 20
    with derivative.verified_candidate_connection(root=root, publication=publication) as connection:
        assert connection.execute("SELECT count(*) FROM versions").fetchone() == (3,)
        assert connection.execute("SELECT count(*) FROM records").fetchone() == (1,)
        assert connection.execute("SELECT count(*) FROM aliases").fetchone() == (1,)
        with pytest.raises(sqlite3.OperationalError):
            connection.execute("DELETE FROM original.versions")
        columns = {row[1] for row in connection.execute("PRAGMA derivative.table_info(candidate_versions)")}
        assert not columns.intersection({"text", "content_json", "aliases"})
    assert digest(original_db) == before
    request = json.loads((root / publication.path).with_name("_request.json").read_text())
    assert request["original_extraction_policy_sha256"] == "b" * 64
    assert request["candidate_policy_sha256"] == derivative.EXTRACTION_POLICY_SHA256


def test_unit_rejects_other_original_authority(parent: tuple[Path, SourcePin]) -> None:
    root, original = parent
    with pytest.raises(DataReadinessError, match="exact original"):
        derivative.publish_issuer_candidate_derivative(root=root,
            parent_population=SourcePin(path=original.path, sha256="f" * 64), output=Path("data/research/derivative"))
    assert not (root / "data/research/derivative").exists()


@pytest.mark.parametrize("change", ["delete", "extra", "candidate", "metadata", "clock_identity"])
def test_unit_independent_replay_rejects_rehashed_sidecar(parent: tuple[Path, SourcePin], change: str) -> None:
    root, publication = publish(parent)
    with sqlite3.connect((root / publication.path).with_name("candidates.sqlite")) as db:
        identity = db.execute("SELECT version_id FROM candidate_versions ORDER BY version_id LIMIT 1").fetchone()[0]
        if change == "delete":
            db.execute("DELETE FROM candidate_versions WHERE version_id=?", (identity,))
        elif change == "extra":
            row = list(db.execute("SELECT * FROM candidate_versions WHERE version_id=?", (identity,)).fetchone())
            row[0] = "f" * 64
            db.execute("INSERT INTO candidate_versions VALUES (" + ",".join("?" for _ in row) + ")", row)
        elif change == "candidate":
            db.execute("UPDATE candidate_versions SET candidates_json='[]',disposition='unclassified'")
        elif change == "metadata":
            db.execute("UPDATE candidate_versions SET content_json_sha256=?", ("e" * 64,))
        else:
            db.execute("UPDATE candidate_versions SET metadata_sha256=?", ("e" * 64,))
    with pytest.raises(DataReadinessError):
        derivative.verify_issuer_candidate_derivative(root=root, publication=rehash_sidecar(root, publication))


def test_unit_unavailable_versions_never_reach_candidate_adapter(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(**kwargs: Any) -> None:
        raise AssertionError("unavailable version entered candidate extraction")
    monkeypatch.setattr(derivative, "saved_text_content_evidence", forbidden)
    for row in (version(1, unavailable=True), version(2, future=True)):
        projected, _ = derivative._project(row)
        assert projected[-1] == "preserved_unavailable"
        assert projected[11] == "[]"


def test_unit_metadata_or_text_mismatch_fails_without_new_exclusion() -> None:
    row = list(version(1))
    row[6] += " Altered source."
    with pytest.raises(DataReadinessError, match="text changed"):
        derivative._project(tuple(row))
    row = list(version(1))
    content = json.loads(row[5])
    content["identity_available_at_utc"] = "2021-01-01T00:00:00+00:00"
    row[5] = json.dumps(content)
    with pytest.raises(DataReadinessError, match="availability clock"):
        derivative._project(tuple(row))


@pytest.mark.parametrize("mutate", ["source", "implementation"])
def test_unit_final_mutation_retains_private_failure_no_public_completion(
    parent: tuple[Path, SourcePin], monkeypatch: pytest.MonkeyPatch, mutate: str,
) -> None:
    root, original = parent
    project = derivative._project
    count = 0
    def injected(row: tuple[Any, ...]) -> tuple[tuple[Any, ...], bool]:
        nonlocal count
        result = project(row)
        count += 1
        if count == 3:
            path = root / ("data/source.json" if mutate == "source" else "src/current.py")
            path.write_text("changed during unit publication", encoding="utf-8")
        return result
    monkeypatch.setattr(derivative, "_project", injected)
    with pytest.raises(DataReadinessError, match="source changed"):
        derivative.publish_issuer_candidate_derivative(root=root, parent_population=original,
                                                       output=Path("data/research/derivative"))
    assert not (root / "data/research/derivative").exists()
    stage = root / "data/research/.derivative.deriving"
    assert stage.is_dir()
    assert not (stage / "_manifest.json").exists()


def test_unit_joined_reader_rechecks_parent_sources_on_exit(parent: tuple[Path, SourcePin]) -> None:
    root, publication = publish(parent)
    with pytest.raises(DataReadinessError, match="source changed"):
        with derivative.verified_candidate_connection(root=root, publication=publication):
            (root / "data/source.json").write_text("changed during unit read", encoding="utf-8")


def test_unit_no_overwrite_or_silent_failed_stage_resume(parent: tuple[Path, SourcePin]) -> None:
    root, original = parent
    stage = root / "data/research/.derivative.deriving"
    stage.mkdir()
    with pytest.raises(DataReadinessError, match="stage already exists"):
        derivative.publish_issuer_candidate_derivative(root=root, parent_population=original,
                                                       output=Path("data/research/derivative"))
    assert stage.exists()
