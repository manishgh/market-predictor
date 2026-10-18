"""Synthetic SQLite UNIT fixtures only; no real-source or admission evidence."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.research import news_source_index_inputs as owner
from market_predictor.research import news_source_links as original
from tests.test_news_source_links import CUT, STAMP, _insert_version, _metadata, _record, _sha, _write


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(owner.runtime, "guard", lambda *args, **kwargs: None)
    old_code = tmp_path / "original_reader.py"
    old_code.write_text("# Synthetic UNIT original implementation identity", encoding="utf-8")
    new_code = tmp_path / "adapter.py"
    new_code.write_text("# Synthetic UNIT adapter implementation identity", encoding="utf-8")
    old_inventory = {old_code.name: file_sha256(old_code)}
    monkeypatch.setattr(original, "_implementation", lambda root: {old_code.name: file_sha256(old_code)})
    monkeypatch.setattr(owner, "_implementation", lambda root: {new_code.name: file_sha256(new_code)})
    # Deliberately absent old raw payload: only its immutable hash is consumed.
    opaque = {"path": "raw/never_opened.json", "sha256": "a" * 64}
    policy = {"schema": original.SCHEMA + "_config", "corpus_manifest": opaque, "population_manifest": opaque,
              "population_database": opaque, "identity_manifest": opaque,
              "archives": [{"name": name, "collection_manifest": opaque, "attribution_manifest": opaque}
                           for name in ("early", "later", "corrected")]}
    config = _write(tmp_path, "config.json", policy)
    directory = tmp_path / "published"
    directory.mkdir()
    database = directory / "index.sqlite"
    proof = {"source": "Synthetic UNIT proof", "ordinal": 1}
    proof_sha = original.json_sha256(proof)
    with closing(original._connect(database)) as db:
        original._schema(db)
        _insert_version(db, "old", _metadata(), _record(), link=True)
        newer = _metadata()
        newer.update(version_available_at_utc=CUT.isoformat(), source_version_sha256=_sha("new raw"))
        _insert_version(db, "new", newer, {"disposition": "unavailable_text"}, link=False)
        db.execute("INSERT INTO versions VALUES (?,?,?,?,?,?,?)",
                   ("unknown", "unknown", "alpaca", "story", None, None, "unavailable_clock"))
        db.execute("INSERT INTO proofs VALUES (?,?)", (proof_sha, original._json(proof)))
        for table in ("events", "relations", "sec_receipts", "enriched"):
            db.execute(f"DROP TABLE {table}")
        db.commit()
    request = {"schema": original.SCHEMA + "_request", "config": config.model_dump(mode="json"), "policy": policy,
               "implementation_files": old_inventory, **original.CLOSED}
    request_pin = _write(tmp_path, "published/_request.json", request)
    provenance = {opaque["path"]: opaque["sha256"], config.path: config.sha256}
    manifest = {"schema": original.SCHEMA, "status": "complete_source_links_only", "coverage": "unknown",
                "request_sha256": request_pin.sha256, "source_files": provenance, "implementation_files": old_inventory,
                "artifacts": {"index.sqlite": file_sha256(database), "_request.json": request_pin.sha256},
                "counts": {"query_copy_records": 3}, **original.CLOSED}
    index = _write(tmp_path, "published/_manifest.json", manifest)
    return dict(root=tmp_path, index=index, manifest=manifest, request=request, config=config, database=database,
                directory=directory, proof=proof, proof_sha=proof_sha, original_code=old_code, adapter_code=new_code)


def _reseal(case: dict[str, Any]) -> None:
    case["index"] = _write(case["root"], "published/_manifest.json", case["manifest"])


def test_original_reader_methods_full_revision_history_and_provenance(case: dict[str, Any]) -> None:
    assert owner.ConsumedNewsSourceLinkReader.for_period is original.NewsSourceLinkReader.for_period
    assert owner.ConsumedNewsSourceLinkReader.for_decision is original.NewsSourceLinkReader.for_decision
    assert owner.ConsumedNewsSourceLinkReader.proof is original.NewsSourceLinkReader.proof
    with owner.open_news_source_link_index(root=case["root"], index=case["index"]) as reader:
        result = reader.for_period("A", CUT, CUT)
        assert [version.version_id for version in result.versions] == ["old", "new"]
        assert [link.version_id for link in result.links] == ["old"]
        assert len(result.uncertainties) == 1 and result.uncertainties[0].known_at_utc is None
        assert result.coverage == () and result.source_coverage_admitted is False
        assert [version.version_id for version in reader.for_decision("A", STAMP).versions] == ["old"]
        assert reader.proof(case["proof_sha"]) == case["proof"]
        assert reader.producer_source_provenance == case["manifest"]["source_files"]
        assert "raw/never_opened.json" not in reader.source_files
        assert reader.raw_sources_rechecked is False and reader.consumption_scope == "saved_source_link_index"
        with pytest.raises(TypeError):
            reader.producer_source_provenance["raw/never_opened.json"] = "b" * 64
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            reader._db.execute("DELETE FROM versions")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        reader.for_decision("A", STAMP)


@pytest.mark.parametrize("field,value", [("status", "partial"), ("coverage", "known"),
                                        *[(name, True) for name in original.CLOSED]])
def test_rehashed_scope_or_admission_poison_rejected(case: dict[str, Any], field: str, value: Any) -> None:
    case["manifest"][field] = value
    _reseal(case)
    with pytest.raises(DataReadinessError, match="scope|admission"):
        with owner.open_news_source_link_index(root=case["root"], index=case["index"]):
            pytest.fail("poison must reject before yield")


@pytest.mark.parametrize("name", ["published/_manifest.json", "published/_request.json", "published/index.sqlite", "config.json"])
def test_changed_consumed_bytes_fail_before_yield(case: dict[str, Any], name: str) -> None:
    with (case["root"] / name).open("ab") as stream:
        stream.write(b"UNIT mutation")
    with pytest.raises(DataReadinessError):
        with owner.open_news_source_link_index(root=case["root"], index=case["index"]):
            pytest.fail("changed input must reject before yield")


@pytest.mark.parametrize("name", ["published/_manifest.json", "published/_request.json", "config.json",
                                 "original_reader.py", "adapter.py"])
def test_changed_consumed_bytes_fail_on_exit(case: dict[str, Any], name: str) -> None:
    with pytest.raises(DataReadinessError):
        with owner.open_news_source_link_index(root=case["root"], index=case["index"]) as reader:
            assert reader.for_decision("A", STAMP).versions
            with (case["root"] / name).open("ab") as stream:
                stream.write(b"UNIT mutation")


def test_sqlite_changed_during_read_rejected_at_exit(case: dict[str, Any]) -> None:
    with pytest.raises(DataReadinessError, match="changed"):
        with owner.open_news_source_link_index(root=case["root"], index=case["index"]):
            with sqlite3.connect(case["database"]) as db:
                db.execute("DELETE FROM links")


@pytest.mark.parametrize("fault", ["partial_directory", "extra_file", "extra_artifact", "missing_provenance", "request_schema",
                                  "request_flags", "request_hash", "implementation", "policy"])
def test_resealed_incomplete_or_inconsistent_authority_rejects(case: dict[str, Any], fault: str) -> None:
    if fault == "partial_directory":
        case["directory"].rename(case["root"] / "published.partial")
        case["index"] = case["index"].model_copy(update={"path": "published.partial/_manifest.json"})
    elif fault == "extra_file":
        (case["directory"] / "index.sqlite-wal").write_bytes(b"UNIT unfinished journal")
    elif fault == "extra_artifact":
        case["manifest"]["artifacts"]["other.json"] = "b" * 64
        _reseal(case)
    elif fault == "missing_provenance":
        case["manifest"]["source_files"].pop("raw/never_opened.json")
        _reseal(case)
    elif fault == "request_hash":
        case["manifest"]["request_sha256"] = "b" * 64
        _reseal(case)
    elif fault == "implementation":
        case["original_code"].write_text("UNIT changed original reader", encoding="utf-8")
    else:
        if fault == "request_schema":
            case["request"]["schema"] = "unrelated"
        elif fault == "request_flags":
            case["request"]["training_eligible"] = True
        else:
            case["request"]["policy"] = {**case["request"]["policy"], "identity_manifest": {"path": "other", "sha256": "b" * 64}}
        request_pin = _write(case["root"], "published/_request.json", case["request"])
        case["manifest"]["request_sha256"] = case["manifest"]["artifacts"]["_request.json"] = request_pin.sha256
        _reseal(case)
    with pytest.raises(DataReadinessError):
        with owner.open_news_source_link_index(root=case["root"], index=case["index"]):
            pytest.fail("inconsistent authority must reject")


def test_original_proof_hash_validation_is_not_reimplemented_or_bypassed(case: dict[str, Any]) -> None:
    with sqlite3.connect(case["database"]) as db:
        db.execute("UPDATE proofs SET payload=?", (json.dumps({"changed": True}),))
    case["manifest"]["artifacts"]["index.sqlite"] = file_sha256(case["database"])
    _reseal(case)
    with owner.open_news_source_link_index(root=case["root"], index=case["index"]) as reader:
        with pytest.raises(DataReadinessError, match="proof payload differs"):
            reader.proof(case["proof_sha"])
