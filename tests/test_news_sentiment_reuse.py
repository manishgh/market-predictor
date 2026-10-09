"""Synthetic UNIT fixtures, never operational source/qualification evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.core.errors import DataReadinessError
from market_predictor.research import news_sentiment_reuse as reuse
from market_predictor.research.news_corpus_enrichment import CorpusVersion
from market_predictor.sentiment import build_sentiment_inputs


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _request(path: Path, value: dict[str, Any]) -> str:
    digest = hashlib.sha256(reuse._json(value).encode()).hexdigest()
    _write(path, {**value, "request_sha256": digest})
    return digest


def _child(root: Path, path: str, frame: pd.DataFrame, inputs: dict[str, Any]) -> dict[str, Any]:
    target = root / path
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(target, index=False)
    item = {"chunk_id": "chunk", "path": path, "sha256": _hash(target), "rows": len(frame),
            "manifest_path": path + ".manifest.json", "start_utc": "2023-01-01T00:00:00Z"}
    _write(root / item["manifest_path"], {"schema": "market_data.artifact_manifest.v1", "artifact_sha256": item["sha256"],
                                        "rows": len(frame), "inputs": inputs})
    return item


def _fixture(root: Path, monkeypatch: pytest.MonkeyPatch, *, event_change: dict[str, Any] | None = None,
             score_change: dict[str, Any] | None = None) -> tuple[reuse.SentimentArchive, CorpusVersion]:
    monkeypatch.setattr(reuse, "_guard", lambda: None)
    (root / "unit_code.py").write_text("# unit fixture\n", encoding="utf-8")
    monkeypatch.setattr(reuse, "_implementation", lambda root: {"unit_code.py": _hash(root / "unit_code.py")})
    event = {"event_id": "event-query-A", "security_id": "legacy-query-A", "ticker": "A", "source_family": "alpaca",
             "raw_sha256": "a" * 64, "title": "Company wins contract", "summary": "Delivery next year.",
             "text": "Long body differs from scored input", "published_at_utc": "2023-12-03T12:00:00Z",
             "available_at_utc": "2023-12-03T12:01:00Z", "provider_updated_at_utc": "2023-12-03T12:01:00Z",
             "availability_policy": "provider_publication_proxy"}
    method = {"model_name": "ProsusAI/finbert", "model_revision": reuse.REVISION, "text_mode": "title_summary",
              "max_length": 128, "fixed_latency_minutes": 5,
              "sentiment_availability_policy": "provider_publication_proxy_plus_fixed_inference_latency"}
    score = {"event_id": event["event_id"], "security_id": event["security_id"], "ticker": "A", "source_family": "alpaca",
             "published_at_utc": event["published_at_utc"], "event_available_at_utc": event["available_at_utc"],
             "research_feature_available_at_utc": "2023-12-03T12:06:00Z", "inference_computed_at_utc": "2026-01-01T00:00:00Z",
             "sentiment_label": "positive", "sentiment_confidence": 0.8, "sentiment_numeric": 0.8,
             "sentiment_input_sha256": hashlib.sha256(str(build_sentiment_inputs(pd.DataFrame([event])).iloc[0]).encode()).hexdigest(),
             "sentiment_model": method["model_name"], "sentiment_model_revision": method["model_revision"],
             "sentiment_input_mode": "title_summary", "sentiment_max_length": 128,
             "sentiment_availability_policy": method["sentiment_availability_policy"]}
    event.update(event_change or {})
    score.update(score_change or {})
    collection_request = _request(root / "collection/_request.json", {"schema": "original-unit-collection"})
    events = _child(root, "collection/events.parquet", pd.DataFrame([event]),
                    {"chunk_id": "chunk", "collection_request_sha256": collection_request})
    _write(root / "collection/_manifest.json", {"schema": "swing.alpaca_news_history_manifest.v1",
                                               "request_sha256": collection_request, "artifacts": [events]})
    collection_pin = SourcePin(path="collection/_manifest.json", sha256=_hash(root / "collection/_manifest.json"))
    request = _request(root / "sentiment/_request.json", {**method, "collection_manifest_sha256": collection_pin.sha256,
                                                         "collection_request_sha256": collection_request})
    scores = _child(root, "sentiment/scores.parquet", pd.DataFrame([score]),
                    {"chunk_id": "chunk", "sentiment_request_sha256": request, "source_event_artifact_sha256": events["sha256"]})
    scores["source_event_artifact_sha256"] = events["sha256"]
    _write(root / "sentiment/_manifest.json", {"schema": "swing.event_sentiment_manifest.v1", "request_sha256": request,
                                              "artifacts": [scores]})
    sentiment_pin = SourcePin(path="sentiment/_manifest.json", sha256=_hash(root / "sentiment/_manifest.json"))
    version = CorpusVersion("version", "cluster", "cohort-company-A", "alpaca", 2023,
                            {"source_version_sha256": "a" * 64, "source_metadata": {
                                "archive": "early", "event_id": "event-query-A", "query_security_id": "legacy-query-A",
                                "query_ticker": "A", "raw_sha256": "a" * 64},
                             "published_at_utc": "2023-12-03T12:00:00Z", "version_available_at_utc": "2023-12-03T12:01:00Z",
                             "availability_semantics": "historical_proxy"}, "Body text is not FinBERT input")
    return reuse.SentimentArchive("early", collection_pin, sentiment_pin), version


def _build(root: Path, archive: reuse.SentimentArchive) -> SourcePin:
    result = reuse.build_sentiment_reuse_index(root=root, output=Path("index"), sources=[archive])
    return SourcePin(path="index/_manifest.json", sha256=result["manifest_sha256"])


def test_exact_bridge_and_separate_score_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive, record = _fixture(tmp_path, monkeypatch)
    pin = _build(tmp_path, archive)
    resolver = reuse.open_sentiment_reuse_index(root=tmp_path, index=pin)
    try:
        result = resolver.resolve(record)
        assert result.status == "matched" and result.score == 0.8 and result.label == "positive"
        assert result.available_at_utc == pd.Timestamp("2023-12-03T12:06:00Z")
        assert result.binding["inference_computed_at_utc"].startswith("2026-")
        assert result.binding["security_id"] == "legacy-query-A"
        assert result.binding["company_attribution_established"] is False
        assert result.binding["sentiment_model_revision"] == reuse.REVISION
        assert "collection/events.parquet" in resolver.source_files
        assert "sentiment/scores.parquet.manifest.json" in resolver.source_files
    finally:
        resolver.close()


@pytest.mark.parametrize("change,match", [
    ({"security_id": "wrong-company"}, "identity"),
    ({"ticker": "B"}, "identity"),
    ({"sentiment_input_sha256": "b" * 64}, "input hash"),
    ({"sentiment_model_revision": "b" * 40}, "method"),
    ({"sentiment_input_mode": "text"}, "method"),
    ({"sentiment_max_length": 512}, "method"),
    ({"research_feature_available_at_utc": "2023-12-03T12:01:00Z"}, "availability"),
    ({"event_available_at_utc": "2023-12-03T12:02:00Z"}, "clocks"),
    ({"sentiment_numeric": -0.8}, "signed"),
    ({"sentiment_confidence": float("nan")}, "confidence"),
])
def test_poisoned_original_score_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                        change: dict[str, Any], match: str) -> None:
    archive, _ = _fixture(tmp_path, monkeypatch, score_change=change)
    with pytest.raises(DataReadinessError, match=match):
        _build(tmp_path, archive)
    assert not (tmp_path / "index").exists()


@pytest.mark.parametrize("field,value", [("archive", "later"), ("event_id", "other"),
                                         ("query_security_id", "cohort-company-A"), ("query_ticker", "B")])
def test_no_story_or_company_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: str) -> None:
    archive, record = _fixture(tmp_path, monkeypatch)
    record.metadata["source_metadata"][field] = value
    resolver = reuse.open_sentiment_reuse_index(root=tmp_path, index=_build(tmp_path, archive))
    try:
        result = resolver.resolve(record)
        assert result.status == "missing_exact_source_version"
        assert result.score is result.label is result.available_at_utc is None
    finally:
        resolver.close()


def test_revision_no_carry_forward_and_sec_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive, record = _fixture(tmp_path, monkeypatch)
    resolver = reuse.open_sentiment_reuse_index(root=tmp_path, index=_build(tmp_path, archive))
    try:
        record.metadata["source_version_sha256"] = "b" * 64
        record.metadata["source_metadata"]["raw_sha256"] = "b" * 64
        assert resolver.resolve(record).status == "missing_exact_source_version"
        assert resolver.resolve(replace(record, source_family="sec")).status == "missing_source_family"
    finally:
        resolver.close()


def test_future_source_not_indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive, record = _fixture(tmp_path, monkeypatch, event_change={"provider_updated_at_utc": "2025-01-01T00:00:00Z"})
    resolver = reuse.open_sentiment_reuse_index(root=tmp_path, index=_build(tmp_path, archive))
    try:
        assert resolver.resolve(record).status == "missing_exact_source_version"
    finally:
        resolver.close()


@pytest.mark.parametrize("updated", [None, pd.NaT])
def test_never_updated_article_uses_publication_version(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                       updated: Any) -> None:
    published = "2023-12-03T12:00:00Z"
    archive, record = _fixture(tmp_path, monkeypatch,
                               event_change={"provider_updated_at_utc": updated, "available_at_utc": published},
                               score_change={"event_available_at_utc": published,
                                             "research_feature_available_at_utc": "2023-12-03T12:05:00Z"})
    record.metadata["version_available_at_utc"] = published
    resolver = reuse.open_sentiment_reuse_index(root=tmp_path, index=_build(tmp_path, archive))
    try:
        result = resolver.resolve(record)
        assert result.status == "matched" and result.score == 0.8
        assert result.binding["provider_updated_at_utc"] is None
        assert result.binding["effective_version_available_at_utc"] == "2023-12-03T12:00:00+00:00"
        assert result.available_at_utc == pd.Timestamp("2023-12-03T12:05:00Z")
    finally:
        resolver.close()


@pytest.mark.parametrize("updated", ["not-a-clock", "NaT", ""])
def test_malformed_nonnull_update_is_not_publication_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                             updated: str) -> None:
    archive, _ = _fixture(tmp_path, monkeypatch, event_change={"provider_updated_at_utc": updated})
    with pytest.raises(DataReadinessError, match="clock"):
        _build(tmp_path, archive)
    assert not (tmp_path / "index").exists()


@pytest.mark.parametrize("path", ["collection/events.parquet", "sentiment/scores.parquet.manifest.json",
                                  "unit_code.py", "index/index.sqlite"])
def test_published_index_rechecks_bound_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    archive, _ = _fixture(tmp_path, monkeypatch)
    pin = _build(tmp_path, archive)
    with (tmp_path / path).open("ab") as stream:
        stream.write(b"poison")
    with pytest.raises(DataReadinessError):
        reuse.open_sentiment_reuse_index(root=tmp_path, index=pin)


def test_reject_observed_claim_and_changed_version_clock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive, record = _fixture(tmp_path, monkeypatch)
    resolver = reuse.open_sentiment_reuse_index(root=tmp_path, index=_build(tmp_path, archive))
    try:
        record.metadata["availability_semantics"] = "observed"
        assert resolver.resolve(record).status == "unsupported_availability_semantics"
        record.metadata["availability_semantics"] = "historical_proxy"
        record.metadata["version_available_at_utc"] = "2023-12-04T00:00:00Z"
        assert resolver.resolve(record).status == "mismatched_source_clock"
    finally:
        resolver.close()
