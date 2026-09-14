"""Tiny synthetic saved histories; no model, network or real archive reads."""
from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.catalysts.issuer_events.attribution import ATTRIBUTION_POLICY_SHA256, ATTRIBUTION_POLICY_VERSION
from market_predictor.catalysts.issuer_events.relevance import SecurityMetadata, add_event_relevance
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.catalyst_lineage import _coverage_frame, verify_completed_catalyst_lineage
from market_predictor.swing.datasets.initial_fit_issuer_news import (
    FINBERT_REVISION,
    SavedIssuerAuthority,
    _bounded_rows,
    _derive,
    _metadata,
    _validate_selected_scores,
    _verify_decision_bounds,
    _write_frame,
    build_initial_fit_catalyst_lineage,
    derive_initial_fit_issuer_inputs,
    load_initial_fit_issuer_inputs,
    publish_initial_fit_issuer_family,
)
from market_predictor.swing.features.catalyst_decision_authority import publish_catalyst_decision_authority
from tests.test_swing_catalyst_lineage import _policy_text, _relations, _sentiments


def _json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


@pytest.fixture
def saved(tmp_path: Path) -> SavedIssuerAuthority:
    return _saved_authority(tmp_path)


def _saved_authority(tmp_path: Path, *, repository_relative_records: bool = False) -> SavedIssuerAuthority:
    names = ("data/raw/alpaca_news_20190709_20210708_v1",
        "data/research/alpaca_news_20190709_20210708_v1_attribution_v2_strict",
        "data/research/alpaca_news_20190709_20210708_v1_sentiment_finbert_cuda_v1") if repository_relative_records else (
            "raw", "old-attribution", "old-sentiment")
    collection, attribution, sentiment = (tmp_path / name for name in names)
    labels = _write_frame(pd.DataFrame({"company": ["Synthetic Only"]}), tmp_path / "labels.parquet", "security_business_labels", {})
    identities = _write_frame(pd.DataFrame({"company": ["Synthetic Only"], "security_id": ["security:a"], "ticker": ["AAA"],
        "effective_from_utc": pd.to_datetime(["2019-07-09T00:00:00Z"], utc=True),
        "effective_to_utc": pd.to_datetime(["2026-07-09T00:00:00Z"], utc=True),
        "available_at_utc": pd.to_datetime(["2019-01-01T00:00:00Z"], utc=True)}),
        tmp_path / "identities.parquet", "security_business_label_coverage", {})
    records = []
    for chunk in ("crossing", "missing", "heldout"):
        times = ["2024-05-28T20:00:00Z", "2024-05-29T20:00:00Z"] if chunk != "heldout" else ["2025-01-02T20:00:00Z"]
        frame = pd.DataFrame({"event_id": [f"{chunk}-{i}" for i in range(len(times))],
                              "feature_available_at_utc": pd.to_datetime(times, utc=True),
                              "published_at_utc": pd.to_datetime(times, utc=True), "source_family": "alpaca",
                              "security_id": "security:b" if chunk == "missing" else "security:a", "ticker": "AAA",
                              "title": "Synthetic Only raises full-year earnings guidance", "summary": "", "text": "",
                              "availability_policy": "provider_publication_proxy"})
        record = _write_frame(frame, collection / "events" / f"{chunk}.parquet", "events", {})
        records.append({**record, "chunk_id": chunk, "security_id": "security:b" if chunk == "missing" else "security:a", "ticker": "AAA"})
    ledger = pd.DataFrame([{"chunk_id": r["chunk_id"], "collection_id": r["chunk_id"],
        "security_id": r["security_id"], "ticker": "AAA", "source_family": "alpaca", "status": "observed", "row_count": r["rows"],
        "requested_start_utc": pd.Timestamp("2025-01-01T00:00:00Z" if r["chunk_id"] == "heldout" else "2024-05-01T00:00:00Z"),
        "requested_end_utc": pd.Timestamp("2025-02-01T00:00:00Z" if r["chunk_id"] == "heldout" else "2024-06-01T00:00:00Z")}
        for r in records])
    ledger_record = _write_frame(ledger, collection / "ledger.parquet", "source_collections", {})
    cr = {"schema": "synthetic_collection_request", "work_units": []}
    ch = json_sha256(cr)
    _json(collection / "_request.json", {**cr, "request_sha256": ch})
    cm = {"status": "complete", "production_ready": False, "request_sha256": ch, "artifacts": records,
          "source_collections_path": ledger_record["path"], "source_collections_sha256": ledger_record["sha256"]}
    _json(collection / "_manifest.json", cm)
    audit = tmp_path / "audit.json"
    _json(audit, {"passed": True, "request_sha256": ch, "coverage_blindspot_security_ids": ["security:b"]})
    common = {"collection_manifest_path": str(collection / "_manifest.json"),
              "collection_manifest_sha256": file_sha256(collection / "_manifest.json"),
              "collection_audit_path": str(audit), "collection_audit_sha256": file_sha256(audit),
              "collection_request_sha256": ch, "excluded_security_ids": ["security:b"], "production_ready": False}
    ar = {**common, "schema": "swing.event_attribution_request.v1", "attribution_policy_version": ATTRIBUTION_POLICY_VERSION,
          "attribution_policy_sha256": ATTRIBUTION_POLICY_SHA256, "business_labels_path": labels["path"],
          "business_labels_sha256": labels["sha256"], "security_identities_path": identities["path"],
          "security_identities_sha256": identities["sha256"]}
    sr = {**common, "schema": "swing.event_sentiment_request.v1", "model_name": "ProsusAI/finbert", "model_revision": FINBERT_REVISION}
    ah, sh = json_sha256(ar), json_sha256(sr)
    _json(attribution / "_request.json", {**ar, "request_sha256": ah})
    _json(sentiment / "_request.json", {**sr, "request_sha256": sh})
    rr, ss = [], []
    for r in records:
        if r["chunk_id"] == "missing":
            continue
        frame, _ = load_canonical_artifact(Path(r["path"]), allow_research=True)
        relation = pd.concat([_relations(time) for time in frame.feature_available_at_utc], ignore_index=True)
        relation["event_id"] = frame.event_id
        relation["relation_id"] = frame.event_id + "-relation"
        relation["event_feature_available_at_utc"] = frame.feature_available_at_utc
        relation["source_security_id"] = relation["target_security_id"] = "security:a"
        relation["source_ticker"] = relation["target_ticker"] = "AAA"
        relation_record = _write_frame(relation, attribution / "relations" / f"{r['chunk_id']}.parquet", "event_security_relations",
            {"event_attribution_request_sha256": ah, "source_event_artifact_sha256": r["sha256"], "chunk_id": r["chunk_id"]})
        score = _sentiments().iloc[:len(frame)].copy()
        score["event_id"] = frame.event_id
        score["published_at_utc"] = score["event_available_at_utc"] = frame.feature_available_at_utc
        score["research_feature_available_at_utc"] = frame.feature_available_at_utc + pd.Timedelta(minutes=5)
        score["security_id"], score["ticker"] = "security:a", "AAA"
        score["sentiment_model_revision"] = FINBERT_REVISION
        score["sentiment_input_sha256"] = [json_sha256({"synthetic_event": value}) for value in frame.event_id]
        score["sentiment_numeric"] = [0.4] + [float("inf")] * (len(frame) - 1)
        score_record = _write_frame(score, sentiment / "sentiment" / f"{r['chunk_id']}.parquet", "event_sentiment_research",
            {"sentiment_request_sha256": sh, "source_event_artifact_sha256": r["sha256"], "chunk_id": r["chunk_id"]})
        rr.append({**relation_record, "chunk_id": r["chunk_id"]})
        ss.append({**score_record, "chunk_id": r["chunk_id"]})
    for root, schema, request_hash, rows in ((attribution, "swing.event_attribution_manifest.v1", ah, rr),
                                            (sentiment, "swing.event_sentiment_manifest.v1", sh, ss)):
        _json(root / "_manifest.json", {"schema": schema, "status": "complete", "production_ready": False,
              "failed_chunks": {}, "request_sha256": request_hash, "excluded_security_ids": ["security:b"], "artifacts": rows})
    if repository_relative_records:
        for directory in (collection, sentiment):
            parent = json.loads((directory / "_manifest.json").read_text())
            if directory == collection:
                parent["schema"] = "swing.alpaca_news_history_manifest.v1"
                parent["source_collections_path"] = str(Path(parent["source_collections_path"]).relative_to(tmp_path))
            for child in parent["artifacts"]:
                path = Path(child["path"])
                child["path"] = str(path.relative_to(tmp_path))
                child["manifest_path"] = str(manifest_path_for(path).relative_to(tmp_path))
                canonical = json.loads(manifest_path_for(path).read_text())
                canonical["artifact_path"] = child["path"]
                _json(manifest_path_for(path), canonical)
            _json(directory / "_manifest.json", parent)
        for directory, payload, key in ((attribution, ar, "event_attribution_request_sha256"),
            (sentiment, sr, "sentiment_request_sha256")):
            payload["collection_manifest_sha256"] = file_sha256(collection / "_manifest.json")
            digest = json_sha256(payload)
            _json(directory / "_request.json", {**payload, "request_sha256": digest})
            parent = json.loads((directory / "_manifest.json").read_text())
            parent["request_sha256"] = digest
            for child in parent["artifacts"]:
                path = Path(child["path"])
                if not path.is_absolute():
                    path = tmp_path / path
                canonical = json.loads(manifest_path_for(path).read_text())
                canonical["inputs"][key] = digest
                _json(manifest_path_for(path), canonical)
            _json(directory / "_manifest.json", parent)
    return SavedIssuerAuthority(attribution, file_sha256(attribution / "_manifest.json"), sentiment,
        file_sha256(sentiment / "_manifest.json"), file_sha256(manifest_path_for(Path(labels["path"]))),
        file_sha256(manifest_path_for(Path(identities["path"]))))


def _run(saved: SavedIssuerAuthority, output: Path):  # type: ignore[no-untyped-def]
    with patch("market_predictor.swing.datasets.initial_fit_issuer_news.assert_system_memory_available"):
        return _derive(saved, output, "2019-07-09T00:00:00Z", "2024-05-28T22:00:00Z")


def test_real_manifest_relative_paths_resolve_from_explicit_repository_not_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = _saved_authority(tmp_path, repository_relative_records=True)
    elsewhere = tmp_path / "unrelated-working-directory"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    with patch("market_predictor.swing.datasets.initial_fit_issuer_news.assert_system_memory_available"):
        derived = _derive(source, tmp_path / "derived", "2019-07-09T00:00:00Z", "2024-05-28T22:00:00Z", repository_root=tmp_path)
    assert derived.manifest["source_events"] == derived.manifest["scored_events"] == 1
    assert derived.manifest["source_path_resolution"]["repository_root"] == str(tmp_path)
    replay = load_initial_fit_issuer_inputs(derived.directory, expected_manifest_sha256=derived.manifest_sha256)
    assert replay.manifest_sha256 == derived.manifest_sha256


@pytest.mark.parametrize("violation", ["missing_base", "wrong_base", "foreign_authority", "different_sidecar"])
def test_saved_path_scope_never_falls_back_to_an_existing_file(tmp_path: Path, violation: str) -> None:
    source = _saved_authority(tmp_path, repository_relative_records=True)
    request = json.loads((source.attribution_dir / "_request.json").read_text())
    collection = Path(request["collection_manifest_path"]).parent
    record = json.loads((collection / "_manifest.json").read_text())["artifacts"][0]
    root, base = collection, tmp_path
    if violation == "missing_base":
        with pytest.raises(DataReadinessError, match="explicit repository_root"):
            _metadata(root, record, "events", ["event_id"])
        return
    if violation == "wrong_base":
        base = collection
    elif violation == "foreign_authority":
        root = source.sentiment_dir
    else:
        record["manifest_path"] = str(source.sentiment_dir / "sentiment/unrelated.parquet.manifest.json")
    with pytest.raises((DataReadinessError, FileNotFoundError)):
        _metadata(root, record, "events", ["event_id"], repository_root=base)


def _request_only_interruption(source: SavedIssuerAuthority, output: Path, repository: Path) -> str:
    namespace = "market_predictor.swing.datasets.initial_fit_issuer_news."
    with patch(namespace + "assert_system_memory_available"), patch(namespace + "_metadata", side_effect=FileNotFoundError("first child")):
        with pytest.raises(FileNotFoundError, match="first child"):
            _derive(source, output, "2019-07-09T00:00:00Z", "2024-05-28T22:00:00Z", repository_root=repository)
    assert {path.name for path in output.iterdir()} == {"_request.json"}
    return file_sha256(output / "_request.json")


def test_pinned_request_only_interruption_resumes_without_rewriting_request(tmp_path: Path) -> None:
    source = _saved_authority(tmp_path, repository_relative_records=True)
    output = tmp_path / "derived"
    pin = _request_only_interruption(source, output, tmp_path)
    with patch("market_predictor.swing.datasets.initial_fit_issuer_news.assert_system_memory_available"):
        result = _derive(source, output, "2019-07-09T00:00:00Z", "2024-05-28T22:00:00Z",
            repository_root=tmp_path, expected_existing_request_sha256=pin)
    assert result.manifest["source_events"] == 1
    assert file_sha256(output / "_request.json") == pin


@pytest.mark.parametrize("violation", ["missing_pin", "wrong_pin", "changed_request", "existing_child"])
def test_request_only_resume_refuses_unverified_or_materialized_state(tmp_path: Path, violation: str) -> None:
    source = _saved_authority(tmp_path, repository_relative_records=True)
    output = tmp_path / "derived"
    pin = _request_only_interruption(source, output, tmp_path)
    if violation == "changed_request":
        request = json.loads((output / "_request.json").read_text())
        request["cutoff_utc"] = "2024-05-27T22:00:00+00:00"
        _json(output / "_request.json", request)
        pin = file_sha256(output / "_request.json")
    elif violation == "existing_child":
        (output / "retained-child.parquet").write_bytes(b"synthetic partial evidence; must remain untouched")
    elif violation == "missing_pin":
        pin = None  # type: ignore[assignment]
    else:
        pin = "0" * 64
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    namespace = "market_predictor.swing.datasets.initial_fit_issuer_news."
    with patch(namespace + "assert_system_memory_available"), patch(namespace + "_metadata") as numeric:
        with pytest.raises(DataReadinessError):
            _derive(source, output, "2019-07-09T00:00:00Z", "2024-05-28T22:00:00Z",
                repository_root=tmp_path, expected_existing_request_sha256=pin)
    numeric.assert_not_called()
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_derivation_lease_uses_explicit_repository_root(saved: SavedIssuerAuthority, tmp_path: Path) -> None:
    namespace = "market_predictor.swing.datasets.initial_fit_issuer_news."
    with patch(namespace + "heavy_job_lease", return_value=nullcontext()) as lease, patch(namespace + "_derive"), patch(
        namespace + "assert_system_memory_available"
    ):
        derive_initial_fit_issuer_inputs(source=saved, output_directory=tmp_path / "derived", repository_root=tmp_path)
    assert lease.call_args.kwargs["runtime_dir"] == tmp_path / "data/runtime"


def test_derives_only_initial_fit_scores_and_preserves_unknown_chunks(saved: SavedIssuerAuthority, tmp_path: Path) -> None:
    original_reader = pd.read_parquet
    projections = []

    def metadata_reader(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if "old-sentiment" in str(path):
            assert kwargs.get("columns") is not None
            assert "sentiment_numeric" not in kwargs["columns"]
            projections.append(str(path))
        assert "heldout.parquet" not in str(path)
        return original_reader(path, *args, **kwargs)

    with patch("pandas.read_parquet", side_effect=metadata_reader):
        result = _run(saved, tmp_path / "derived")
    assert projections
    assert result.manifest["source_events"] == result.manifest["scored_events"] == 1
    assert result.manifest["excluded_security_ids"] == []
    assert result.manifest["unavailable_chunks"] == [{"chunk_id": "missing", "reason": "old_attribution_or_sentiment_not_published"}]
    scores, _ = load_canonical_artifact(result.directory / "sentiment/sentiment/crossing.parquet", allow_research=True)
    assert scores.sentiment_numeric.tolist() == [0.4]
    assert scores.event_id.tolist() == ["crossing-0"]
    ledger, _ = load_canonical_artifact(result.directory / "collection/source_collections.parquet", allow_research=True)
    coverage = _coverage_frame(ledger, excluded_security_ids={"security:b"},
                              relation_chunk_ids={"crossing"}, sentiment_chunk_ids={"crossing"})
    missing = coverage.loc[coverage.chunk_id.eq("missing")].iloc[0]
    assert not missing.missingness_known and missing.coverage_state == "coverage_blindspot"
    with pytest.raises(DataReadinessError, match="immutable"):
        _run(saved, result.directory)


def test_source_pin_and_revision_poison_rejected_before_projection(saved: SavedIssuerAuthority, tmp_path: Path) -> None:
    with pytest.raises(DataReadinessError, match="pin differs"):
        _run(replace(saved, attribution_manifest_sha256="f" * 64), tmp_path / "bad-pin")
    path = saved.sentiment_dir / "_request.json"
    request = json.loads(path.read_text())
    request.pop("request_sha256")
    request["model_revision"] = "wrong"
    request_hash = json_sha256(request)
    _json(path, {**request, "request_sha256": request_hash})
    manifest_path = saved.sentiment_dir / "_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["request_sha256"] = request_hash
    _json(manifest_path, manifest)
    with pytest.raises(DataReadinessError, match="exact FinBERT revision"):
        _run(replace(saved, sentiment_manifest_sha256=file_sha256(manifest_path)), tmp_path / "bad-revision")


def test_no_numeric_projection_for_empty_cutoff_selection(tmp_path: Path) -> None:
    path = tmp_path / "scores.parquet"
    frame = pd.DataFrame({"event_id": ["heldout"], "available": pd.to_datetime(["2025-01-01T00:00:00Z"], utc=True),
                          "sentiment_numeric": [float("inf")]})
    frame.to_parquet(path, index=False)
    result = _bounded_rows(path, frame[["event_id", "available"]], clock="available",
                          start=pd.Timestamp("2019-07-09T00:00:00Z"), cutoff=pd.Timestamp("2024-05-28T22:00:00Z"))
    assert result.empty


@pytest.mark.parametrize("poison", [float("inf"), float("nan"), 2.0])
def test_selected_invalid_scores_fail_instead_of_becoming_neutral(poison: float) -> None:
    with pytest.raises(DataReadinessError, match="numeric contract"):
        _validate_selected_scores(pd.DataFrame({"sentiment_numeric": [poison], "sentiment_confidence": [0.9], "relevance": [1.0]}))


def test_selected_relevance_uses_existing_heuristic_range_without_modifying_scores() -> None:
    metadata = SecurityMetadata("security:synthetic", "SYN", "Example Devices Inc.", "Information Technology", "Semiconductor")
    events = pd.DataFrame({"security_id": metadata.security_id, "ticker": metadata.ticker,
        "title": ["Stocks to watch", "SYN revenue", "SYN Example Devices semiconductor earnings"], "summary": "", "text": ""})
    relevance = add_event_relevance(events, metadata).relevance.tolist()
    assert relevance[0] == 0.1 and relevance[-1] == 2.0 and relevance[1] > 1.0
    values = pd.DataFrame({"sentiment_numeric": 0.4, "sentiment_confidence": 0.9, "relevance": [*relevance, 1.15]})
    before = values.copy(deep=True)
    _validate_selected_scores(values)
    pd.testing.assert_frame_equal(values, before)


@pytest.mark.parametrize("relevance", [float("nan"), float("inf"), -float("inf"), -0.1, 0.0, 0.09, 2.0001])
def test_selected_relevance_outside_producer_range_is_rejected(relevance: float) -> None:
    with pytest.raises(DataReadinessError, match="relevance.*numeric contract"):
        _validate_selected_scores(pd.DataFrame({"sentiment_numeric": [0.4], "sentiment_confidence": [0.9], "relevance": [relevance]}))


def test_realistic_saved_relevance_above_one_survives_derivation_exactly(saved: SavedIssuerAuthority, tmp_path: Path) -> None:
    path = saved.sentiment_dir / "sentiment/crossing.parquet"
    frame, manifest = load_canonical_artifact(path, allow_research=True)
    frame.loc[frame.event_id.eq("crossing-0"), "relevance"] = 1.15
    record = _write_frame(frame, path, "event_sentiment_research", manifest["inputs"])
    parent_path = saved.sentiment_dir / "_manifest.json"
    parent = json.loads(parent_path.read_text())
    next(row for row in parent["artifacts"] if row["chunk_id"] == "crossing").update(record)
    _json(parent_path, parent)
    source = replace(saved, sentiment_manifest_sha256=file_sha256(parent_path))
    result = _run(source, tmp_path / "derived-relevance")
    projected, _ = load_canonical_artifact(result.directory / "sentiment/sentiment/crossing.parquet", allow_research=True)
    assert projected.event_id.tolist() == ["crossing-0"]
    assert projected.relevance.tolist() == [1.15]


def test_derived_replay_and_decisions_reject_out_of_bounds(saved: SavedIssuerAuthority, tmp_path: Path) -> None:
    derived = _run(saved, tmp_path / "derived")
    with pytest.raises(DataReadinessError, match="pin differs"):
        load_initial_fit_issuer_inputs(derived.directory, expected_manifest_sha256="f" * 64)
    path = tmp_path / "decisions.parquet"
    _write_frame(pd.DataFrame({"decision_time_utc": pd.to_datetime(["2025-01-01T22:00:00Z"], utc=True)}), path, "decisions", {})
    with pytest.raises(DataReadinessError, match="cross"):
        _verify_decision_bounds(path, derived)
    with pytest.raises(DataReadinessError, match="bounded initial-fit"):
        _derive(saved, tmp_path / "future", "2019-07-09T00:00:00Z", "2024-05-29T22:00:00Z")


def test_derivation_feeds_current_lineage_and_family_publishers(saved: SavedIssuerAuthority, tmp_path: Path) -> None:
    derived = _run(saved, tmp_path / "derived")
    decisions = tmp_path / "decisions.parquet"
    _write_frame(pd.DataFrame({"security_id": ["security:a"], "ticker": ["AAA"], "timeframe": ["1d"],
        "decision_time_utc": pd.to_datetime(["2024-05-28T22:00:00Z"], utc=True),
        "bar_start_utc": pd.to_datetime(["2024-05-28T13:30:00Z"], utc=True),
        "prediction_cutoff_policy_id": ["xnys_1800_america_new_york_v1"]}), decisions, "decisions", {})
    policy = tmp_path / "lineage.toml"
    policy.write_text(_policy_text(), encoding="utf-8")
    namespace = "market_predictor.swing.datasets.initial_fit_issuer_news."
    with patch(namespace + "heavy_job_lease", return_value=nullcontext()), patch(namespace + "assert_system_memory_available"):
        lineage = build_initial_fit_catalyst_lineage(derived_directory=derived.directory,
            expected_manifest_sha256=derived.manifest_sha256, decisions_path=decisions, policy_path=policy, out_dir=tmp_path / "lineage")
        assert lineage["status"] == "complete"
        assert lineage["relation_rows"] == 1
        assert int(lineage["assignment_rows"]) > 0
        verify_completed_catalyst_lineage(tmp_path / "lineage")
        authority = publish_catalyst_decision_authority([tmp_path / "lineage"], tmp_path / "decisions-authority")
        assert len(authority.decisions) == 1
        family = publish_initial_fit_issuer_family(derived_directory=derived.directory,
            expected_manifest_sha256=derived.manifest_sha256, decisions_path=decisions,
            policy_path=Path(__file__).resolve().parents[1] / "configs/swing_event_family_policy.toml",
            output_directory=tmp_path / "family")
        assert not family.events.empty
        assert set(family.events.source_event_id) == {"crossing-0"}
