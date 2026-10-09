"""Synthetic UNIT corpus fixtures; not actual retained-source or model evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.core.errors import DataReadinessError
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.research import news_corpus_enrichment as c

_PUBLISHED = datetime(2023, 1, 1, 12, tzinfo=UTC)
_REPO = Path(__file__).resolve().parents[1]


def _metadata(index: int, text: str | None, **changes: Any) -> dict[str, Any]:
    result = {
        "source_id": f"unit-story-{index}",
        "source_family": "unit",
        "security_id": "unit-security",
        "published_at_utc": _PUBLISHED.isoformat(),
        "version_available_at_utc": (_PUBLISHED + timedelta(minutes=1)).isoformat(),
        "event_available_at_utc": (_PUBLISHED + timedelta(minutes=2)).isoformat(),
        "identity_available_at_utc": (_PUBLISHED + timedelta(minutes=3)).isoformat(),
        "first_seen_at_utc": (_PUBLISHED + timedelta(days=100)).isoformat(),
        "availability_semantics": "historical_proxy",
        "text_sha256": None if text is None else hashlib.sha256(text.encode()).hexdigest(),
    }
    result.update(changes)
    return result


@pytest.fixture
def corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, SourcePin]:
    # Only resource measurements are bypassed for tiny synthetic UNIT fixtures.
    monkeypatch.setattr(c, "_guard", lambda: None)
    for relative in c._IMPLEMENTATION_PATHS:
        path = tmp_path / "src/market_predictor" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_REPO / "src/market_predictor" / relative, path)
    path = tmp_path / "versions.sqlite"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE versions(version_id TEXT PRIMARY KEY,cluster_id TEXT,security_id TEXT,"
            "source_family TEXT,year INTEGER,content_json TEXT,text TEXT)"
        )
        for index in reversed(range(7)):
            text: str | None = "Acme earnings grew but cuts guidance."
            metadata = _metadata(index, text)
            if index == 1:
                text = None
                metadata = _metadata(index, text)
            if index == 2:
                metadata.pop("event_available_at_utc")
            if index == 3:
                metadata["availability_semantics"] = "observed"
            connection.execute(
                "INSERT INTO versions VALUES(?,?,?,?,?,?,?)",
                (f"v{index:03}", f"cluster-{index}", "unit-security", "unit", 2023, json.dumps(metadata), text),
            )
    return tmp_path, SourcePin(path=path.name, sha256=c._hash(path))


def _run(corpus: tuple[Path, SourcePin], output: str = "result", **kwargs: Any) -> dict[str, Any]:
    root, pin = corpus
    return c.enrich_news_corpus(root=root, input_database=pin, output=Path(output), options=c.CorpusOptions(batch_size=2), **kwargs)


def _records(root: Path, output: str = "result") -> list[dict[str, Any]]:
    return [record for path in sorted((root / output).glob("part-*.jsonl")) for record in c._read_lines(path)]


def test_every_version_is_accounted_with_separate_historical_and_observed_clocks(corpus: tuple[Path, SourcePin]) -> None:
    manifest = _run(corpus)
    root, _ = corpus
    records = _records(root)
    assert [record["version_id"] for record in records] == [f"v{index:03}" for index in range(7)]
    assert manifest["source_versions"] == manifest["counts"]["input_versions"] == 7
    assert manifest["counts"]["disposition:enriched"] == 5
    assert manifest["counts"]["disposition:unavailable_text"] == 1
    assert manifest["counts"]["disposition:unavailable_clock"] == 1
    assert records[0]["enrichment"]["available_at_utc"] == (_PUBLISHED + timedelta(minutes=3)).isoformat()
    assert records[3]["enrichment"]["available_at_utc"] == (_PUBLISHED + timedelta(days=100)).isoformat()
    assert records[0]["source_clocks"]["first_seen_at_utc"] == (_PUBLISHED + timedelta(days=100)).isoformat()
    assert records[0]["availability_semantics"] == "historical_proxy"
    assert all("text" not in record for record in records)
    assert not manifest["training_eligible"]
    assert records[0]["enrichment"]["attribution_scope"] == "unresolved"
    assert _run(corpus) == manifest


def test_resume_rebuilds_exact_counts_and_qa_after_interrupted_partial_shard(
    corpus: tuple[Path, SourcePin],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = corpus
    original = c._record
    calls = 0

    def interrupted(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RuntimeError("unit interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(c, "_record", interrupted)
    with pytest.raises(RuntimeError, match="unit interruption"):
        _run(corpus)
    stage = root / ".result.private_stage"
    assert (stage / "checkpoint-000000.json").exists()
    assert (stage / ".part-000001.partial").exists()
    assert not (stage / "_manifest.json").exists()
    monkeypatch.setattr(c, "_record", original)
    resumed = _run(corpus)
    fresh = _run(corpus, "fresh")
    assert resumed == fresh
    assert (root / "result/qa_samples.json").read_bytes() == (root / "fresh/qa_samples.json").read_bytes()
    assert not list((root / "result").glob("*.partial"))


def test_uncheckpointed_finished_shard_is_recomputed_before_adoption(
    corpus: tuple[Path, SourcePin],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = corpus
    original = c._write_equal

    def interrupt_checkpoint(path: Path, data: bytes) -> None:
        if path.name == "checkpoint-000000.json":
            raise RuntimeError("unit checkpoint interruption")
        original(path, data)

    monkeypatch.setattr(c, "_write_equal", interrupt_checkpoint)
    with pytest.raises(RuntimeError, match="unit checkpoint"):
        _run(corpus)
    assert (root / ".result.private_stage/part-000000.jsonl").exists()
    monkeypatch.setattr(c, "_write_equal", original)
    assert _run(corpus)["counts"]["input_versions"] == 7


@pytest.mark.parametrize("mutation", ["shard", "counter", "options", "code", "source"])
def test_resume_rejects_changed_completed_evidence_or_request(
    corpus: tuple[Path, SourcePin],
    mutation: str,
) -> None:
    root, pin = corpus
    _run(corpus)
    if mutation == "shard":
        with (root / "result/part-000000.jsonl").open("ab") as stream:
            stream.write(b"{}\n")
    elif mutation == "counter":
        path = root / "result/checkpoint-000000.json"
        record = json.loads(path.read_bytes())
        record["counts"]["input_versions"] += 1
        path.write_bytes(c._encode(record))
    elif mutation == "code":
        (root / "src/market_predictor/research/news_feature_enrichment.py").write_text("# changed UNIT fixture")
    elif mutation == "source":
        with (root / pin.path).open("ab") as stream:
            stream.write(b"unit change")
    with pytest.raises(DataReadinessError):
        if mutation == "options":
            c.enrich_news_corpus(root=root, input_database=pin, output=Path("result"), options=c.CorpusOptions(batch_size=3))
        else:
            _run(corpus)


class _UnitResolver:
    """Explicit UNIT double: proves pipeline clock handling, not score binding."""

    def __init__(self, root: Path) -> None:
        (root / "unit-index.json").write_text("{}")
        (root / "unit-resolver.py").write_text("# UNIT adapter")
        self.source_files = {"unit-index.json": c._hash(root / "unit-index.json")}
        self.implementation_files = {"unit-resolver.py": c._hash(root / "unit-resolver.py")}
        self.identity = {"model": "unit-only", "revision": "unit", "input_mode": "title_summary"}
        self.result = c.SentimentResolution(
            "matched",
            -0.5,
            "negative",
            _PUBLISHED + timedelta(days=2),
            {"unit_only": True, "content_identity": "fixture", "query_identity": "fixture"},
        )

    def resolve(self, record: c.CorpusVersion) -> c.SentimentResolution:
        assert record.version_id
        return self.result


def test_sentiment_latency_never_delays_cues_or_backdates_scores(corpus: tuple[Path, SourcePin]) -> None:
    root, _ = corpus
    resolver = _UnitResolver(root)
    _run(corpus, sentiment_resolver=resolver)
    records = _records(root)
    assert records[0]["enrichment"]["available_at_utc"] == (_PUBLISHED + timedelta(minutes=3)).isoformat()
    assert records[0]["enrichment"]["existing_sentiment_score"] is None
    assert records[0]["sentiment"]["available_at_utc"] == (_PUBLISHED + timedelta(days=2)).isoformat()
    assert records[3]["sentiment"]["available_at_utc"] == (_PUBLISHED + timedelta(days=100)).isoformat()
    assert records[3]["sentiment"]["original_available_at_utc"] == (_PUBLISHED + timedelta(days=2)).isoformat()


def test_mismatched_sentiment_is_explicit_and_cannot_expose_a_score(corpus: tuple[Path, SourcePin]) -> None:
    root, _ = corpus
    resolver = _UnitResolver(root)
    resolver.result = c.SentimentResolution("raw_version_mismatch")
    _run(corpus, sentiment_resolver=resolver)
    assert _records(root)[0]["sentiment"]["status"] == "raw_version_mismatch"
    resolver.result = replace(resolver.result, score=0.8)
    with pytest.raises(DataReadinessError, match="unmatched sentiment"):
        _run(corpus, "invalid", sentiment_resolver=resolver)


def test_code_mutation_during_work_prevents_manifest(
    corpus: tuple[Path, SourcePin],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = corpus
    original = c._record

    def mutate(row: sqlite3.Row, *args: Any) -> dict[str, Any]:
        result = original(row, *args)
        if row["version_id"] == "v006":
            (root / "src/market_predictor/research/news_sample_buckets.py").write_text("# changed UNIT fixture")
        return result

    monkeypatch.setattr(c, "_record", mutate)
    with pytest.raises(DataReadinessError, match="pinned file changed"):
        _run(corpus)
    assert not (root / ".result.private_stage/_manifest.json").exists()
    assert not (root / "result").exists()


def test_source_size_bounds_keep_row_dispositions(corpus: tuple[Path, SourcePin]) -> None:
    root, pin = corpus
    result = c.enrich_news_corpus(
        root=root, input_database=pin, output=Path("tiny"), options=c.CorpusOptions(batch_size=2, max_text_bytes=5)
    )
    assert result["counts"]["input_versions"] == 7
    assert result["counts"]["disposition:unavailable_size_limit"] == 6
    assert result["counts"]["disposition:unavailable_text"] == 1


def test_shard_byte_limit_does_not_commit_partial_work(corpus: tuple[Path, SourcePin]) -> None:
    root, pin = corpus
    with pytest.raises(DataReadinessError, match="shard byte limit"):
        c.enrich_news_corpus(root=root, input_database=pin, output=Path("tiny"), options=c.CorpusOptions(batch_size=2, max_shard_bytes=32))
    assert not (root / ".tiny.private_stage/checkpoint-000000.json").exists()
    assert not (root / ".tiny.private_stage/_manifest.json").exists()


def test_busy_lease_refuses_before_source_reads(corpus: tuple[Path, SourcePin], monkeypatch: pytest.MonkeyPatch) -> None:
    root, _ = corpus

    def forbidden(*args: Any) -> str:
        raise AssertionError("source read before lease")

    monkeypatch.setattr(c, "_hash", forbidden)
    with heavy_job_lease("unit lease holder", runtime_dir=root / "data/runtime"):
        with pytest.raises(HeavyJobBusyError):
            _run(corpus)


def test_subsequent_pk_batches_use_index_seek(corpus: tuple[Path, SourcePin]) -> None:
    root, pin = corpus
    suffix, parameters = c._query_tail("v001", 2)
    with sqlite3.connect(root / pin.path) as connection:
        plan = connection.execute("EXPLAIN QUERY PLAN SELECT version_id " + suffix, parameters).fetchall()
        assert any("SEARCH" in row[3] and "version_id>?" in row[3] for row in plan)
        assert [row[0] for row in connection.execute("SELECT version_id " + suffix, parameters)] == ["v002", "v003"]


def test_full_job_memory_guard_uses_canonical_thresholds(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(c, "assert_memory_budget", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(c, "assert_system_memory_available", lambda **kwargs: calls.append(kwargs))
    c._guard()
    assert calls[0]["hard_budget_gib"] == 5.0
    assert calls[1] == {"minimum_available_gib": 2.0, "maximum_used_percent": 85.0}


def test_progress_fires_only_after_durable_checkpoint_and_resume_reconstructs(
    corpus: tuple[Path, SourcePin],
) -> None:
    root, _ = corpus
    observed: list[dict[str, Any]] = []

    def observer(value: dict[str, Any]) -> None:
        observed.append(value)
        checkpoint = root / ".result.private_stage" / f"checkpoint-{value['sequence']:06d}.json"
        assert checkpoint.is_file()
        assert value["processed_versions"] == value["counts"]["input_versions"] == 2
        assert value["total_versions"] == 7
        raise RuntimeError("UNIT observer interrupted")

    with pytest.raises(RuntimeError, match="observer interrupted"):
        _run(corpus, progress=observer)
    assert len(observed) == 1
    assert not (root / ".result.private_stage/_manifest.json").exists()
    resumed = _run(corpus, progress=observed.append)
    assert resumed == _run(corpus, "fresh")
    assert observed[-1]["processed_versions"] == 7


@pytest.mark.parametrize(
    "change, expected",
    [
        ({"availability_semantics": "observed", "first_seen_at_utc": None}, "unavailable_clock"),
        ({"text_sha256": "0" * 64}, "unavailable_text_identity"),
        ({"source_family": "different"}, "unavailable_record_identity"),
        ({"source_id": None}, "unavailable_reference"),
    ],
)
def test_unavailable_source_facts_remain_explicit_dispositions(
    corpus: tuple[Path, SourcePin],
    change: dict[str, Any],
    expected: str,
) -> None:
    root, pin = corpus
    with sqlite3.connect(root / pin.path) as connection:
        connection.row_factory = sqlite3.Row
        row = dict(c._query(connection, None, c.CorpusOptions(batch_size=1)).fetchone())
    metadata = json.loads(row["content_json"])
    metadata.update(change)
    row["content_json"] = json.dumps(metadata)
    record = c._record(row, c.CorpusOptions(), None)  # type: ignore[arg-type]
    assert record["disposition"] == expected
    assert record["version_id"] == "v000"
    assert "enrichment" not in record


@pytest.mark.parametrize("field", ["quote", "sentiment_score"])
def test_resume_rejects_content_forgery_even_with_rehashed_checkpoint(
    corpus: tuple[Path, SourcePin],
    field: str,
) -> None:
    root, _ = corpus
    resolver = _UnitResolver(root)

    def interrupt(value: dict[str, Any]) -> None:
        raise RuntimeError("UNIT stopped after durable checkpoint")

    with pytest.raises(RuntimeError, match="durable checkpoint"):
        _run(corpus, sentiment_resolver=resolver, progress=interrupt)
    stage = root / ".result.private_stage"
    shard = stage / "part-000000.jsonl"
    records = list(c._read_lines(shard))
    if field == "quote":
        records[0]["enrichment"]["cues"][0]["quote"] = "invented UNIT quotation"
    else:
        records[0]["sentiment"]["score"] = 0.99
    shard.write_bytes(b"".join(c._encode(record) for record in records))
    checkpoint = stage / "checkpoint-000000.json"
    saved = json.loads(checkpoint.read_bytes())
    saved["sha256"] = c._hash(shard)
    checkpoint.write_bytes(c._encode(saved))
    with pytest.raises(DataReadinessError, match="content differs from source replay"):
        _run(corpus, sentiment_resolver=resolver)
    assert not (stage / "_manifest.json").exists()
    assert not (root / "result").exists()


def test_progress_mutation_of_durable_shard_cannot_publish_stale_manifest(
    corpus: tuple[Path, SourcePin],
) -> None:
    root, _ = corpus

    def mutate(value: dict[str, Any]) -> None:
        if value["sequence"] == 0:
            shard = root / ".result.private_stage/part-000000.jsonl"
            records = list(c._read_lines(shard))
            records[0]["enrichment"]["cues"][0]["quote"] = "changed after checkpoint"
            shard.write_bytes(b"".join(c._encode(record) for record in records))

    with pytest.raises(DataReadinessError, match="output artifact changed before publication"):
        _run(corpus, progress=mutate)
    assert not (root / ".result.private_stage/_manifest.json").exists()
    assert not (root / "result").exists()
