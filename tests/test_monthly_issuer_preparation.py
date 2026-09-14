"""Synthetic full projection and issuer inputs, including held-out score poison."""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd
import pytest

from market_predictor.canonical.reconciliation import stamp_canonical_decision_ids
from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.catalysts.issuer_events.attribution import ATTRIBUTION_POLICY_SHA256
from market_predictor.catalysts.issuer_events.attribution_history import ATTRIBUTION_SCOPE_POLICY
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import HeavyJobBusyError
from market_predictor.swing.datasets.initial_fit_issuer_news import FINBERT_REVISION, _write_frame
from market_predictor.swing.datasets.issuer_news_preparation import (
    SCHEMA,
    _aligned_score_times,
    _checkpoint,
    _chunk,
    _CompactSink,
    _normalize_compact_clocks,
    _project_source,
    _select,
    _source,
    prepare_initial_fit_monthly_lineages,
)
from market_predictor.swing.datasets.issuer_news_publication import _publish, load_monthly_catalyst_index
from market_predictor.swing.features.catalyst_aggregates import build_swing_ablation_rows
from market_predictor.swing.features.catalyst_decision_authority import load_catalyst_decision_authority
from tests.test_initial_fit_issuer_derivation import _run as derive_fixture
from tests.test_initial_fit_issuer_derivation import saved as saved_fixture
from tests.test_swing_catalyst_lineage import _policy_text, _relations, _sentiments


def _json(path: Path, value: dict) -> str:  # type: ignore[type-arg]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return file_sha256(path)


@pytest.fixture
def prepared_inputs(tmp_path: Path):  # type: ignore[no-untyped-def]
    roots = {name: tmp_path / name for name in ("collection", "attribution", "sentiment")}
    times = pd.to_datetime(["2024-04-30T20:00:00Z", "2024-05-28T20:00:00Z",
        "2024-05-28T21:59:00Z", "2024-05-29T20:00:00Z"], utc=True)
    events = pd.DataFrame({"event_id": ["april", "may", "pending", "future"], "feature_available_at_utc": times,
        "published_at_utc": times, "source_family": "alpaca", "security_id": "security:a", "ticker": "AAA",
        "title": "SYNTHETIC TEST ONLY raises earnings guidance", "summary": "", "text": "",
        "availability_policy": "provider_publication_proxy"})
    event = _write_frame(events, roots["collection"] / "events/main.parquet", "events", {})
    event.update(chunk_id="main", security_id="security:a", ticker="AAA")
    ledger = pd.DataFrame([{"chunk_id": chunk, "collection_id": chunk, "security_id": security, "ticker": ticker,
        "source_family": "alpaca", "status": status, "row_count": count,
        "requested_start_utc": pd.Timestamp("2024-04-01T00:00:00Z"), "requested_end_utc": pd.Timestamp("2024-06-01T00:00:00Z")}
        for chunk, security, ticker, status, count in [("main", "security:a", "AAA", "observed", 4),
            ("empty", "security:zero", "ZERO", "observed_empty", 0),
            ("unknown", "security:unknown", "UNK", "unavailable_saved_evidence", 0)]])
    lr = _write_frame(ledger, roots["collection"] / "ledger.parquet", "source_collections", {})
    shared = {"status": "complete", "production_ready": False, "failed_chunks": {},
        "scope_policy": ATTRIBUTION_SCOPE_POLICY, "source_coverage_admitted": False,
        "excluded_security_ids": [], "coverage_blindspot_security_ids": ["security:unknown"]}
    ch = json_sha256({"synthetic": True})
    cm = {**shared, "request_sha256": ch, "artifacts": [event], "source_collections_path": lr["path"],
        "source_collections_sha256": lr["sha256"]}
    cpin = _json(roots["collection"] / "_manifest.json", cm)
    audit_path = tmp_path / "audit.json"
    audit_pin = _json(audit_path, {**shared, "passed": True, "request_sha256": ch})
    common = {"collection_manifest_sha256": cpin, "collection_audit_sha256": audit_pin, "collection_request_sha256": ch,
        "collection_manifest_path": str(roots["collection"] / "_manifest.json"), "collection_audit_path": str(audit_path),
        "scope_policy": ATTRIBUTION_SCOPE_POLICY, "excluded_security_ids": [], "source_coverage_admitted": False}
    ar = {**common, "attribution_policy_sha256": ATTRIBUTION_POLICY_SHA256}
    sr = {**common, "model_name": "ProsusAI/finbert", "model_revision": FINBERT_REVISION}
    ah, sh = json_sha256(ar), json_sha256(sr)
    _json(roots["attribution"] / "_request.json", {**ar, "request_sha256": ah})
    _json(roots["sentiment"] / "_request.json", {**sr, "request_sha256": sh})
    relation = pd.concat([_relations(time) for time in times], ignore_index=True)
    relation["event_id"] = events.event_id
    relation["relation_id"] = events.event_id + "-relation"
    relation["event_feature_available_at_utc"] = times
    relation["source_security_id"] = relation["target_security_id"] = "security:a"
    relation["source_ticker"] = relation["target_ticker"] = "AAA"
    rr = _write_frame(relation, roots["attribution"] / "relations/main.parquet", "event_security_relations",
        {"event_attribution_request_sha256": ah, "source_event_artifact_sha256": event["sha256"], "chunk_id": "main"})
    rr.update(chunk_id="main", source_event_sha256=event["sha256"])
    score = pd.concat([_sentiments().iloc[:1]] * 4, ignore_index=True)
    score["event_id"] = events.event_id
    score["published_at_utc"] = score["event_available_at_utc"] = times
    score["research_feature_available_at_utc"] = times + pd.Timedelta(minutes=5)
    score["security_id"], score["ticker"] = "security:a", "AAA"
    score["sentiment_model_revision"] = FINBERT_REVISION
    score["sentiment_input_sha256"] = [json_sha256({"synthetic": value}) for value in events.event_id]
    score["sentiment_numeric"] = [0.4, 0.5, float("inf"), float("inf")]
    ss = _write_frame(score, roots["sentiment"] / "sentiment/main.parquet", "event_sentiment_research",
        {"sentiment_request_sha256": sh, "source_event_artifact_sha256": event["sha256"], "chunk_id": "main"})
    ss.update(chunk_id="main", source_event_artifact_sha256=event["sha256"])
    apin = _json(roots["attribution"] / "_manifest.json", {**shared, "request_sha256": ah, "artifacts": [rr]})
    spin = _json(roots["sentiment"] / "_manifest.json", {**shared, "request_sha256": sh, "artifacts": [ss]})
    source = {"kind": "corrected", "collection": {"directory": "collection", "manifest_sha256": cpin},
        "attribution": {"directory": "attribution", "manifest_sha256": apin},
        "sentiment": {"directory": "sentiment", "manifest_sha256": spin}, "audit": {"path": "audit.json", "sha256": audit_pin}}
    partitions = []
    for month, dates in (("2024-04", ["2024-04-30"]), ("2024-05", ["2024-05-01", "2024-05-28"])):
        rows = [{"security_id": security, "ticker": ticker, "timeframe": "1d",
            "decision_time_utc": pd.Timestamp(date + "T22:00:00Z"), "bar_start_utc": pd.Timestamp(date + "T13:30:00Z"),
            "prediction_cutoff_policy_id": "xnys_1800_america_new_york_v1"}
            for date in dates for security, ticker in (("security:a", "AAA"), ("security:zero", "ZERO"), ("security:unknown", "UNK"))]
        partitions.append((month, stamp_canonical_decision_ids(pd.DataFrame(rows))))
    metadata_path = tmp_path / "synthetic-decision-source.json"
    metadata_pin = _json(metadata_path, {"synthetic_test_only": True})
    policy = tmp_path / "lineage.toml"
    policy.write_text(_policy_text(), encoding="utf-8")
    config = {"schema": SCHEMA, "decisions": {"path": metadata_path.name, "sha256": metadata_pin},
        "policy": {"path": "lineage.toml", "sha256": file_sha256(policy)}, "sources": {"corrected": source}}
    config_path = tmp_path / "preparation.json"
    config_pin = _json(config_path, config)

    @contextmanager
    def projection(root, record):  # type: ignore[no-untyped-def]
        assert root == tmp_path and record == config["decisions"]
        yield SimpleNamespace(cohort_sha256="a" * 64, expected_rows=9,
            retained_security_ids=("security:a", "security:zero", "security:unknown"),
            source_files={metadata_path.name: metadata_pin}, partitions=iter((month, frame.copy()) for month, frame in partitions))

    return config_path, config_pin, projection


def _run_preparation(root: Path, inputs, *, pin=None, estimate_only=False):  # type: ignore[no-untyped-def]
    path, digest, projection = inputs
    namespace = "market_predictor.swing.datasets.issuer_news_preparation."
    with patch(namespace + "_decision_projection", side_effect=projection), patch(namespace + "_guard"):
        return prepare_initial_fit_monthly_lineages(root=root, config_path=path, expected_config_sha256=digest,
            output_directory=root / "prepared", expected_existing_manifest_sha256=pin, estimate_only=estimate_only)


def test_empty_stored_chunk_preserves_datetime_availability(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config = json.loads(prepared_inputs[0].read_text())
    source = _source(tmp_path, config["sources"]["corrected"])
    event_pin = ""
    for name, kind in (("collection", "events"), ("attribution", "event_security_relations"),
            ("sentiment", "event_sentiment_research")):
        original = source["records"][name]["main"]
        frame, manifest = load_canonical_artifact(Path(original["path"]), allow_research=True)
        frame = frame.iloc[:0].copy()
        frame["event_id"] = pd.Series([], dtype="float64")
        inputs = dict(manifest["inputs"])
        if name != "collection":
            inputs["source_event_artifact_sha256"] = event_pin
        record = _write_frame(frame, source["roots"][name] / "empty.parquet", kind, inputs)
        record.update(chunk_id="main")
        if name == "collection":
            event_pin = record["sha256"]
        else:
            record["source_event_sha256" if name == "attribution" else "source_event_artifact_sha256"] = event_pin
        source["records"][name]["main"] = record
    events, relations, scores, _, availability = _chunk(source, "main")
    assert availability.empty
    assert isinstance(availability.score_available_at_utc.dtype, pd.DatetimeTZDtype)
    selected = _select(events, relations, scores, pd.Timestamp("2024-04-01T00:00:00Z"),
        pd.Timestamp("2024-05-28T22:00:00Z"))
    assert all(frame.empty for frame in selected[:3])
    assert selected[3] is False


def test_score_clock_alignment_retains_row_identity_duplicates_and_cutoffs() -> None:
    cutoff = pd.Timestamp("2024-05-01T22:00:00Z")
    events = pd.DataFrame({"event_id": ["late", "ready"],
        "feature_available_at_utc": [cutoff, cutoff]}, index=[17, 4])
    scores = pd.DataFrame({"event_id": ["ready", "late"],
        "research_feature_available_at_utc": [cutoff, cutoff + pd.Timedelta(seconds=1)]}, index=[8, 6])
    relations = pd.DataFrame({"event_id": ["ready", "ready", "late", "ready"],
        "feature_available_at_utc": [cutoff, cutoff, cutoff, cutoff + pd.Timedelta(seconds=1)]}, index=[31, 9, 2, 7])
    expected = pd.Series([cutoff + pd.Timedelta(seconds=1), cutoff], index=events.index,
        name="research_feature_available_at_utc")
    pd.testing.assert_series_equal(_aligned_score_times(events.event_id, scores), expected)
    selected_events, selected_relations, selected_scores, pending = _select(events, relations, scores,
        cutoff - pd.Timedelta(days=2), cutoff)
    assert list(selected_events.index) == [4]
    assert list(selected_relations.index) == [31, 9]
    assert list(selected_scores.event_id) == ["ready"]
    assert pending is True
    selected = _select(events, relations.iloc[:0], scores, cutoff - pd.Timedelta(days=2), cutoff)
    assert list(selected[0].event_id) == ["ready"] and selected[1].empty and selected[3] is True


def test_nonempty_stored_chunk_filtered_outside_fit_interval(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config = json.loads(prepared_inputs[0].read_text())
    source = _source(tmp_path, config["sources"]["corrected"])
    with patch("market_predictor.swing.datasets.issuer_news_preparation.FIRST", pd.Timestamp("2025-01-01T00:00:00Z")):
        events, relations, scores, _, availability = _chunk(source, "main")
    assert len(availability) == 4
    selected = _select(events, relations, scores, pd.Timestamp("2024-04-01T00:00:00Z"),
        pd.Timestamp("2024-05-28T22:00:00Z"))
    assert all(frame.empty for frame in selected[:3]) and selected[3] is False


def test_one_numeric_scan_routes_cross_month_and_feeds_existing_monthly_publisher(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    namespace = "market_predictor.swing.datasets.issuer_news_preparation."
    original_reader = pd.read_parquet

    def metadata_only(path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if Path(path).resolve() == (tmp_path / "sentiment/sentiment/main.parquet").resolve():
            assert kwargs.get("columns") is not None and "sentiment_numeric" not in kwargs["columns"]
        return original_reader(path, *args, **kwargs)

    with patch(namespace + "_chunk", wraps=_chunk) as read, patch("pandas.read_parquet", side_effect=metadata_only):
        result = _run_preparation(tmp_path, prepared_inputs)
    assert read.call_count == 1
    assert result["rows"] == 9 and result["months"] == 2
    generation = tmp_path / result["sources"]["corrected"]["directory"]
    manifest = json.loads((generation / "_manifest.json").read_text())
    assert manifest["numeric_source_chunks_read"] == 1
    assert manifest["projected_source_rows"] == 3
    for month, expected in (("2024-04", {"april"}), ("2024-05", {"april", "may"})):
        view = tmp_path / manifest["views"][month]
        scores, _ = load_canonical_artifact(view / "sentiment/sentiment/batch-000000.parquet", allow_research=True)
        assert set(scores.event_id) == expected
        assert scores.sentiment_numeric.between(-1, 1).all()
        assert json.loads((view / "collection/_manifest.json").read_text())["excluded_security_ids"] == []
        if month == "2024-05":
            audit = json.loads((view / "collection/_audit.json").read_text())
            assert audit["unavailable_score_tails"][0]["from_utc"] == "2024-05-28T21:59:00+00:00"
            assert "security:a" not in audit["coverage_blindspot_security_ids"]
    with patch("market_predictor.swing.datasets.issuer_news_publication.assert_system_memory_available"):
        published = _publish(tmp_path, Path(result["publication_config"]), result["publication_config_sha256"], tmp_path / "authorities")
    index = load_monthly_catalyst_index(tmp_path / "authorities", expected_manifest_sha256=published["manifest_sha256"])
    assert index["rows"] == 9
    for record in index["months"].values():
        decisions, _ = load_canonical_artifact(tmp_path / record["decisions"]["path"], allow_research=True)
        decisions["feature_eligible"] = decisions["label_eligible"] = True
        authority = load_catalyst_decision_authority(tmp_path / "authorities" / record["directory"])
        for frame in build_swing_ablation_rows(decisions, authority).values():
            assert len(frame) == record["rows"]
            zero = frame.loc[frame.security_id.eq("security:zero")]
            unknown = frame.loc[frame.security_id.eq("security:unknown")]
            assert zero.catalyst_required_source_complete.all() and zero.event_count_1d.eq(0).all()
            assert not unknown.catalyst_required_source_complete.any() and unknown.event_count_1d.isna().all()
            actor = frame.loc[frame.security_id.eq("security:a")].set_index("decision_time_utc")
            if pd.Timestamp("2024-05-01T22:00:00Z") in actor.index:
                assert actor.loc[pd.Timestamp("2024-05-01T22:00:00Z"), "catalyst_required_source_complete"]
                assert not actor.loc[pd.Timestamp("2024-05-28T22:00:00Z"), "catalyst_required_source_complete"]


def test_completed_preparation_replays_without_numeric_rescan(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    result = _run_preparation(tmp_path, prepared_inputs)
    with patch("market_predictor.swing.datasets.issuer_news_preparation._chunk", side_effect=AssertionError("must not rescan")):
        replay = _run_preparation(tmp_path, prepared_inputs, pin=result["manifest_sha256"])
    assert replay == result
    with pytest.raises(DataReadinessError, match="external manifest/checkpoint"):
        _run_preparation(tmp_path, prepared_inputs)


def test_checkpoint_resume_reuses_once_projected_source(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    def interrupted(output, payload, progress):  # type: ignore[no-untyped-def]
        _checkpoint(output, payload, progress)
        if payload["sources"] and not payload["lineages"]:
            raise RuntimeError("synthetic source checkpoint interruption")

    namespace = "market_predictor.swing.datasets.issuer_news_preparation."
    with patch(namespace + "_checkpoint", side_effect=interrupted), pytest.raises(RuntimeError, match="checkpoint interruption"):
        _run_preparation(tmp_path, prepared_inputs)
    pin = file_sha256(tmp_path / "prepared/_checkpoint.json")
    with patch(namespace + "_chunk", side_effect=AssertionError("source must not be scanned again")):
        result = _run_preparation(tmp_path, prepared_inputs, pin=pin)
    assert result["rows"] == 9 and result["months"] == 2


def test_once_derived_old_archive_is_consumed_without_rederivation(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    old_root = tmp_path / "old-inputs"
    old_root.mkdir()
    saved = saved_fixture.__wrapped__(old_root)
    derived = derive_fixture(saved, tmp_path / "once-derived")
    config_path, _, projection = prepared_inputs
    config = json.loads(config_path.read_text())
    config["sources"]["early"] = {"kind": "derived", "directory": "once-derived", "manifest_sha256": derived.manifest_sha256}
    inputs = config_path, _json(config_path, config), projection
    with patch("market_predictor.swing.datasets.initial_fit_issuer_news._derive", side_effect=AssertionError("no rederivation")):
        result = _run_preparation(tmp_path, inputs)
    assert set(result["sources"]) == {"early", "corrected"}


def test_corrected_same_hash_at_different_source_path_is_not_admitted(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path, _, projection = prepared_inputs
    config = json.loads(path.read_text())
    alias = tmp_path / "other-collection-manifest.json"
    alias.write_bytes((tmp_path / "collection/_manifest.json").read_bytes())
    request_path = tmp_path / "attribution/_request.json"
    request = json.loads(request_path.read_text())
    request.pop("request_sha256")
    request["collection_manifest_path"] = str(alias)
    digest = json_sha256(request)
    _json(request_path, {**request, "request_sha256": digest})
    manifest_path = tmp_path / "attribution/_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["request_sha256"] = digest
    config["sources"]["corrected"]["attribution"]["manifest_sha256"] = _json(manifest_path, manifest)
    inputs = path, _json(path, config), projection
    with patch("market_predictor.swing.datasets.issuer_news_preparation._chunk") as numeric:
        with pytest.raises(DataReadinessError, match="source path association"):
            _run_preparation(tmp_path, inputs)
    numeric.assert_not_called()


def test_busy_projection_lease_stops_before_archive_reads(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path, digest, _ = prepared_inputs

    @contextmanager
    def busy(*args):  # type: ignore[no-untyped-def]
        raise HeavyJobBusyError("synthetic existing heavy owner")
        yield  # pragma: no cover

    namespace = "market_predictor.swing.datasets.issuer_news_preparation."
    with patch(namespace + "_decision_projection", side_effect=busy), patch(namespace + "_source") as source:
        with pytest.raises(HeavyJobBusyError, match="existing heavy owner"):
            prepare_initial_fit_monthly_lineages(root=tmp_path, config_path=path, expected_config_sha256=digest,
                output_directory=tmp_path / "prepared")
    source.assert_not_called()
    assert not (tmp_path / "prepared").exists()


@pytest.mark.parametrize("target", ["original_child", "projected_child", "checkpoint"])
def test_preparation_tamper_rejected(prepared_inputs, tmp_path: Path, target: str) -> None:  # type: ignore[no-untyped-def]
    result = _run_preparation(tmp_path, prepared_inputs)
    generation = tmp_path / result["sources"]["corrected"]["directory"]
    path = {"original_child": tmp_path / "sentiment/sentiment/main.parquet",
        "projected_child": generation / "2024-04/sentiment/sentiment/batch-000000.parquet",
        "checkpoint": tmp_path / "prepared/_manifest.json"}[target]
    path.write_bytes(b"synthetic tamper")
    with pytest.raises(DataReadinessError, match="pin differs"):
        _run_preparation(tmp_path, prepared_inputs, pin=result["manifest_sha256"])


def test_many_logical_chunks_pack_into_bounded_files_and_coverage_only_has_no_event_files(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config = json.loads(prepared_inputs[0].read_text())
    source = _source(tmp_path, config["sources"]["corrected"])
    original = _chunk(source, "main")
    source["records"]["collection"] = {f"chunk-{i}": {} for i in range(23)}
    ledger = source["ledger"].iloc[:1]
    source["ledger"] = pd.concat([ledger.assign(chunk_id=f"chunk-{i}") for i in range(23)], ignore_index=True)
    months = {month: {"first_decision_utc": day + "T22:00:00Z", "last_decision_utc": day + "T22:00:00Z"}
        for month, day in (("2024-04", "2024-04-30"), ("2024-05", "2024-05-01"), ("2024-06", "2024-05-27"))}

    def chunk_reader(_source, chunk):  # type: ignore[no-untyped-def]
        events, relations, scores, pins, availability = original
        frames = [frame.copy() for frame in (events, relations, scores, availability)]
        for frame in frames:
            frame["event_id"] = chunk + ":" + frame.event_id
        frames[1]["relation_id"] = chunk + ":" + frames[1].relation_id
        return frames[0], frames[1], frames[2], pins, frames[3]

    namespace = "market_predictor.swing.datasets.issuer_news_preparation."
    with patch(namespace + "_chunk", side_effect=chunk_reader) as read, patch(namespace + "_guard"), patch(namespace + "BATCH_ROWS", 10):
        result = _project_source(tmp_path, source, tmp_path / "packed", months, "a" * 64, pd.Timedelta(days=2))
    manifest = json.loads((tmp_path / result["directory"] / "_manifest.json").read_text())
    assert read.call_count == 23
    assert manifest["physical_batches"] == 6  # 3 per nonempty month, not 69 chunk-month batches.
    assert len(list((tmp_path / "packed").rglob("batch-*.parquet"))) == 18
    assert not list((tmp_path / "packed/2024-06").rglob("batch-*.parquet"))
    assert not list((tmp_path / "packed").rglob("*.pending"))
    for month in ("2024-04", "2024-05"):
        records = json.loads((tmp_path / "packed" / month / "collection/_manifest.json").read_text())["artifacts"]
        assert [record["rows"] for record in records] == [10, 10, 3]
        for record in records:
            frame, _ = load_canonical_artifact(Path(record["path"]), allow_research=True)
            assert set(frame.preparation_original_chunk_id) == set(record["original_source_children"])
            assert set(frame.preparation_source_child_set_sha256) == {json_sha256(original[3])}


def test_estimate_only_never_projects_numeric_source_or_creates_output(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    with patch("market_predictor.swing.datasets.issuer_news_preparation._chunk", side_effect=AssertionError("numeric read")):
        result = _run_preparation(tmp_path, prepared_inputs, estimate_only=True)
    estimate = result["sources"]["corrected"]
    assert result["status"] == "metadata_only_estimate"
    assert estimate["logical_chunk_overlaps"] == 6
    assert estimate["row_limited_batches"] == 2
    assert estimate["projection_files_estimate"] == 26
    assert estimate["coverage_only_event_parquets"] == 0
    assert not (tmp_path / "prepared").exists()


def test_compact_byte_limit_flushes_without_retaining_cross_month_frames(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config = json.loads(prepared_inputs[0].read_text())
    source = _source(tmp_path, config["sources"]["corrected"])
    events, relations, scores, pins, _ = _chunk(source, "main")
    frames = events.iloc[:1], relations.iloc[:1], scores.iloc[:1]
    sink = _CompactSink(tmp_path / "byte-bounded", "a" * 64)
    namespace = "market_predictor.swing.datasets.issuer_news_preparation."
    try:
        sink.append(frames, "one", pins)
        with patch(namespace + "BATCH_BYTES", sink.size + 1), patch(namespace + "_guard"):
            sink.append(frames, "two", pins)
            sink.finish()
        assert sink.batch == 2 and sink.rows == 0
        assert not sink.writers and not sink.children
        assert [row["rows"] for row in sink.records["events"]] == [1, 1]
    finally:
        sink.close()


def test_context_exit_failure_resumes_from_checkpoint_without_source_reprojection(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path, digest, projection = prepared_inputs

    @contextmanager
    def fail_after_projection(*args):  # type: ignore[no-untyped-def]
        with projection(*args) as value:
            yield value
            raise DataReadinessError("synthetic exit verification failed")

    with pytest.raises(DataReadinessError, match="exit verification"):
        _run_preparation(tmp_path, (path, digest, fail_after_projection))
    assert not (tmp_path / "prepared/_manifest.json").exists()
    assert not list((tmp_path / "prepared").glob(".manifest-*.pending"))
    with patch("market_predictor.swing.datasets.issuer_news_preparation._chunk", side_effect=AssertionError("no numeric rescan")):
        result = _run_preparation(tmp_path, prepared_inputs, pin=file_sha256(tmp_path / "prepared/_checkpoint.json"))
    assert result["rows"] == 9 and (tmp_path / "prepared/_manifest.json").exists()


def test_compact_first_null_clock_cannot_strip_later_utc_or_precision(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config = json.loads(prepared_inputs[0].read_text())
    source = _source(tmp_path, config["sources"]["corrected"])
    events, relations, scores, pins, _ = _chunk(source, "main")
    empty_clock = relations.iloc[:1].copy()
    empty_clock["identity_available_at_utc"] = pd.Series([pd.NaT], dtype="datetime64[ms]")
    known_clock = relations.iloc[:1].copy()
    instant = pd.Timestamp("2024-04-01T00:00:00.000000001Z")
    known_clock["identity_available_at_utc"] = pd.Series([instant], dtype="datetime64[ns, UTC]")
    sink = _CompactSink(tmp_path / "utc-schema", "a" * 64)
    try:
        with patch("market_predictor.swing.datasets.issuer_news_preparation._guard"):
            sink.append((events.iloc[:1], empty_clock, scores.iloc[:1]), "null", pins)
            sink.append((events.iloc[:1], known_clock, scores.iloc[:1]), "known", pins)
            sink.finish()
        stored = pd.read_parquet(sink.records["relations"][0]["path"])
        assert pd.isna(stored.identity_available_at_utc.iloc[0])
        assert stored.identity_available_at_utc.iloc[1] == instant
        assert str(stored.identity_available_at_utc.dtype) == "datetime64[ns, UTC]"
    finally:
        sink.close()


def test_compact_writer_rejects_nonnull_naive_clocks(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    config = json.loads(prepared_inputs[0].read_text())
    source = _source(tmp_path, config["sources"]["corrected"])
    events, relations, scores, pins, _ = _chunk(source, "main")
    relations = relations.iloc[:1].copy()
    relations["identity_available_at_utc"] = pd.Series([pd.Timestamp("2024-04-01")])
    sink = _CompactSink(tmp_path / "invalid-clock", "a" * 64)
    try:
        with pytest.raises(DataReadinessError, match="aware UTC"):
            sink.append((events.iloc[:1], relations, scores.iloc[:1]), "invalid", pins)
        assert not sink.writers
    finally:
        sink.close()


@pytest.mark.parametrize("reverse", [False, True])
def test_compact_clock_validation_cannot_deduplicate_invalid_offset(reverse: bool) -> None:
    values = [pd.Timestamp("2020-01-01T00:00:00Z"), pd.Timestamp("2020-01-01T01:00:00+01:00")]
    frame = pd.DataFrame({"identity_available_at_utc": pd.Series(values[::-1] if reverse else values, dtype=object)})
    with pytest.raises(DataReadinessError, match="aware UTC"):
        _normalize_compact_clocks(frame)
