"""Tiny synthetic UNIT publications; original source-index admission is mocked."""
from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.core.errors import DataReadinessError
from market_predictor.research import news_sentiment_copy_evidence as owner
from market_predictor.research.news_decision_features import DecisionCue, NewsVersion

REPO = Path(__file__).resolve().parents[1]
NOW = datetime(2020, 1, 2, 14, tzinfo=UTC)


def _write(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(owner._json(payload) + "\n", encoding="utf-8")
    return _hash(path)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pin(root: Path, path: Path) -> SourcePin:
    return SourcePin(path=path.relative_to(root).as_posix(), sha256=_hash(path))


def _corpus(state: dict[str, Any]) -> None:
    directory = state["root"] / "data/research/synthetic-corpus"
    shard = directory / "part-000000.jsonl"
    shard.write_text("".join(owner._json(record) + "\n" for record in state["records"]), encoding="utf-8")
    manifest = {"status": "complete_feature_cues_only", "request_sha256": _hash(directory / "_request.json"),
        "source_versions": len(state["records"]), "shards": 1,
        "artifacts": {name: _hash(directory / name) for name in ("_request.json", shard.name)},
        **{name: False for name in ("training_eligible", "serving_eligible", "promotion_eligible",
            "classification_precision_measured", "stock_attribution_verified", "finbert_executed", "embeddings_executed")}}
    _write(directory / "_manifest.json", manifest)
    source = state["root"] / "data/research/synthetic-source"
    request_sha = _write(source / "_request.json", {
        "policy": {"corpus_manifest": _pin(state["root"], directory / "_manifest.json").model_dump()}})
    _write(source / "_manifest.json", {"request_sha256": request_sha})
    state["source_pin"] = _pin(state["root"], source / "_manifest.json")
    state["source_files"] = {(source / name).relative_to(state["root"]).as_posix(): _hash(source / name)
                             for name in ("_request.json", "_manifest.json", "index.sqlite")}
    state["shard"] = shard


@pytest.fixture
def synthetic(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(root / "data/runtime"))
    monkeypatch.setattr(owner.runtime, "guard", lambda *args, **kwargs: None)
    package = root / "src/market_predictor"
    for name in owner.IMPLEMENTATION_PATHS:
        destination = package / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "src/market_predictor" / name, destination)
    monkeypatch.setattr(owner, "__file__", str(package / "research/news_sentiment_copy_evidence.py"))
    original_impl = {"src/market_predictor/research/news_decision_features.py": _hash(package / "research/news_decision_features.py")}
    monkeypatch.setattr(owner.source_owner, "_implementation", lambda root: original_impl)
    directory = root / "data/research/synthetic-source"
    directory.mkdir(parents=True)
    versions = tuple(NewsVersion(f"copy-{number}", "alpaca", "document", "a" * 64, "b" * 64,
        NOW, NOW, NOW, True, ("earnings",), (DecisionCue("earnings", "up", "announced"),), False,
        score, NOW + timedelta(minutes=5), "historical_proxy", "b" * 64, NOW)
        for number, score in enumerate((0.5, 0.500001)))
    with sqlite3.connect(directory / "index.sqlite") as db:
        db.execute("CREATE TABLE versions(copy_id TEXT PRIMARY KEY,version_id TEXT,source TEXT,document TEXT,payload TEXT)")
        for number, version in enumerate(versions):
            db.execute("INSERT INTO versions VALUES(?,?,?,?,?)", (version.version_id, f"population-{number}", "alpaca", "document",
                                                                  owner._json(asdict(version))))
        db.commit()
    records = []
    for number, version in enumerate(versions):
        population_id = f"population-{number}"
        binding = {"version_id": population_id, "raw_sha256": version.source_version_sha256, "source_family": "alpaca",
            "sentiment_input_sha256": "c" * 64, "sentiment_model": owner.MODEL, "sentiment_model_revision": owner.REVISION,
            "sentiment_input_mode": "title_summary", "sentiment_max_length": 128,
            "sentiment_availability_policy": owner.AVAILABILITY_POLICY, "sentiment_label": "positive",
            "sentiment_numeric": version.sentiment_score, "sentiment_confidence": version.sentiment_score,
            "research_feature_available_at_utc": version.sentiment_available_at_utc.isoformat(),
            "origins": [{key: {"path": f"synthetic_unopened/{key}.parquet", "sha256": "d" * 64}
                         for key in ("source_event_artifact", "sentiment_artifact")}],
            "training_eligible": False, "serving_eligible": False, "promotion_eligible": False}
        records.append({"version_id": population_id, "source_family": "alpaca", "disposition": "enriched",
            "enrichment": {"version_ref": population_id, "document_ref": "alpaca:document"},
            "sentiment": {"status": "matched", "score": version.sentiment_score, "label": "positive", "binding": binding,
                "available_at_utc": version.sentiment_available_at_utc.isoformat(),
                "original_available_at_utc": version.sentiment_available_at_utc.isoformat()}})
    records.append({"version_id": "unrelated", "disposition": "unavailable_text"})
    corpus = root / "data/research/synthetic-corpus"
    _write(corpus / "_request.json", {"source_files": {"unopened_provider_archive": "e" * 64},
        "training_eligible": False, "serving_eligible": False, "promotion_eligible": False})
    state = {"root": root, "versions": versions, "records": records, "output": root / "data/research/copy-evidence"}
    _corpus(state)

    @contextmanager
    def source_reader(*, root: Path, index: SourcePin) -> Any:
        assert index == state["source_pin"]
        owner.runtime.recheck_files(root, state["source_files"])
        with owner._connect(directory / "index.sqlite", readonly=True) as db:
            yield SimpleNamespace(_db=db, source_files=state["source_files"], implementation_files=original_impl,
                                  producer_source_provenance={"unopened_source_archive": "e" * 64})
        owner.runtime.recheck_files(root, state["source_files"])

    monkeypatch.setattr(owner.source_inputs, "open_news_source_link_index", source_reader)
    return state


def _build(state: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
    return owner.build_news_sentiment_copy_evidence(root=state["root"], source_index=state["source_pin"],
                                                   output=state["output"], **kwargs)


def _open(state: dict[str, Any]) -> Any:
    return owner.open_news_sentiment_copy_evidence(root=state["root"], source_index=state["source_pin"],
                                                  evidence=_pin(state["root"], state["output"] / "_manifest.json"))


def test_real_tiny_join_preserves_exact_identities_proofs_and_consumed_scope(synthetic: dict[str, Any]) -> None:
    progress = []
    manifest = _build(synthetic, progress=progress.append)
    assert manifest["counts"] == {"conflicting_exact_version_groups": 1, "candidate_copy_records": 2,
        "verified_scoring_identity": 2, "corpus_versions_checked": 3}
    assert progress[-1]["processed_versions"] == 3
    assert all(manifest[key] is False for key in owner.CLOSED)
    # Original corpus/provider bytes are provenance; consumption must not reopen them.
    synthetic["shard"].write_bytes(b"no longer opened by evidence consumer")
    with _open(synthetic) as reader:
        first = reader.identity(synthetic["versions"][0])
        assert first is not None and first.score == 0.5 and first.input_sha256 == "c" * 64
        assert first.model_revision == owner.REVISION and first.source_copy_id == "copy-0"
        assert first.score_available_at_utc == NOW + timedelta(minutes=5)
        assert first.binding_sha256 == hashlib.sha256(owner._json(synthetic["records"][0]["sentiment"]["binding"]).encode()).hexdigest()
        assert first.evidence_ref.endswith(":copy-0")
        assert not any("synthetic-corpus" in name or "synthetic-source" in name for name in reader.source_files)
        assert reader.raw_sources_rechecked is False
        payload = json.loads(reader._db.execute("SELECT payload FROM copies WHERE copy_id='copy-0'").fetchone()[0])
        assert payload["corpus_proof"]["line"] == 1 and payload["original_binding"] == synthetic["records"][0]["sentiment"]["binding"]
        assert reader.identity(replace(synthetic["versions"][0], version_id="absent")) is None


@pytest.mark.parametrize("field,value", [("sentiment_score", 0.7), ("sentiment_available_at_utc", NOW + timedelta(minutes=6)),
    ("source_version_sha256", "f" * 64), ("source_document_id", "wrong"), ("source_family", "sec")])
def test_reader_rejects_mismatched_copy_score_clock_or_source(synthetic: dict[str, Any], field: str, value: Any) -> None:
    _build(synthetic)
    with _open(synthetic) as reader, pytest.raises(DataReadinessError, match="requested copy differs"):
        reader.identity(replace(synthetic["versions"][0], **{field: value}))


@pytest.mark.parametrize("field,value", [("sentiment_model", "different-model"), ("sentiment_model_revision", "f" * 40),
    ("sentiment_input_mode", "text"), ("sentiment_max_length", 512)])
def test_unsupported_model_identity_is_explicit_missing(synthetic: dict[str, Any], field: str, value: Any) -> None:
    synthetic["records"][0]["sentiment"]["binding"][field] = value
    _corpus(synthetic)
    manifest = _build(synthetic)
    assert manifest["counts"]["unsupported_scoring_identity"] == 1
    with _open(synthetic) as reader:
        assert reader.identity(synthetic["versions"][0]) is None


def test_missing_binding_is_not_invented(synthetic: dict[str, Any]) -> None:
    del synthetic["records"][0]["sentiment"]["binding"]["sentiment_input_sha256"]
    _corpus(synthetic)
    manifest = _build(synthetic)
    assert manifest["counts"]["missing_scoring_identity"] == 1
    with _open(synthetic) as reader:
        assert reader.identity(synthetic["versions"][0]) is None


@pytest.mark.parametrize("field,value", [("sentiment_numeric", 0.7), ("raw_sha256", "f" * 64),
    ("research_feature_available_at_utc", "2020-01-02T14:06:00+00:00"), ("sentiment_input_sha256", "bad")])
def test_rehashed_corpus_binding_mismatch_rejects(synthetic: dict[str, Any], field: str, value: Any) -> None:
    synthetic["records"][0]["sentiment"]["binding"][field] = value
    _corpus(synthetic)
    with pytest.raises(DataReadinessError):
        _build(synthetic)
    assert not (synthetic["output"] / "_manifest.json").exists()


def test_input_hash_differences_are_preserved_for_resolution_policy(synthetic: dict[str, Any]) -> None:
    synthetic["records"][1]["sentiment"]["binding"]["sentiment_input_sha256"] = "f" * 64
    _corpus(synthetic)
    _build(synthetic)
    with _open(synthetic) as reader:
        first, second = (reader.identity(version) for version in synthetic["versions"])
        assert first is not None and second is not None and first.input_sha256 != second.input_sha256


def test_missing_corpus_version_prevents_publication(synthetic: dict[str, Any]) -> None:
    synthetic["records"].pop(0)
    _corpus(synthetic)
    with pytest.raises(DataReadinessError, match="absent from corpus"):
        _build(synthetic)
    assert not synthetic["output"].exists()


def test_shard_hash_tamper_is_rejected_before_join(synthetic: dict[str, Any]) -> None:
    synthetic["shard"].write_bytes(b"changed")
    with pytest.raises(DataReadinessError, match="input changed"):
        _build(synthetic)


def test_source_mutation_during_progress_prevents_completed_output(synthetic: dict[str, Any]) -> None:
    def progress(_: dict[str, Any]) -> None:
        synthetic["shard"].write_bytes(b"changed during stream")

    with pytest.raises(DataReadinessError, match="bound bytes changed"):
        _build(synthetic, progress=progress)
    assert not synthetic["output"].exists()


@pytest.mark.parametrize("target", ["index.sqlite", "_request.json"])
def test_evidence_artifact_tamper_rejects(synthetic: dict[str, Any], target: str) -> None:
    _build(synthetic)
    with (synthetic["output"] / target).open("ab") as stream:
        stream.write(b"tamper")
    with pytest.raises(DataReadinessError, match="input changed"), _open(synthetic):
        pass


def test_wrong_external_source_index_pin_rejects(synthetic: dict[str, Any]) -> None:
    _build(synthetic)
    with pytest.raises(DataReadinessError, match="source-index differs"), owner.open_news_sentiment_copy_evidence(
        root=synthetic["root"], evidence=_pin(synthetic["root"], synthetic["output"] / "_manifest.json"),
        source_index=synthetic["source_pin"].model_copy(update={"sha256": "f" * 64}),
    ):
        pass


def test_completed_output_is_immutable(synthetic: dict[str, Any]) -> None:
    _build(synthetic)
    with pytest.raises(DataReadinessError, match="already exists"):
        _build(synthetic)
