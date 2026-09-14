"""Independent causal regressions using only small, temporary synthetic archives."""
from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

import market_predictor.swing.datasets.initial_fit_issuer_news as derivation
import market_predictor.swing.datasets.issuer_news_preparation as preparation
from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.catalyst_lineage import _coverage_frame
from market_predictor.swing.features.catalyst_decision_authority import _coverage_eligibility
from tests.test_initial_fit_issuer_derivation import _run as derive_fixture
from tests.test_initial_fit_issuer_derivation import saved as saved
from tests.test_monthly_issuer_authority import _publish_fixture
from tests.test_monthly_issuer_authority import monthly as monthly
from tests.test_monthly_issuer_preparation import _run_preparation
from tests.test_monthly_issuer_preparation import prepared_inputs as prepared_inputs


def _json(path: Path, value: Any) -> str:
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return file_sha256(path)


def _rewrite_corrected(root: Path, inputs: Any, mutation: str) -> Any:
    config_path, _, projection = inputs
    config = json.loads(config_path.read_text())
    source = config["sources"]["corrected"]
    directories = {name: root / source[name]["directory"] for name in ("collection", "attribution", "sentiment")}
    manifests = {name: json.loads((directory / "_manifest.json").read_text()) for name, directory in directories.items()}
    paths = {name: Path(manifest["artifacts"][0]["path"]) for name, manifest in manifests.items()}
    frames = {name: load_canonical_artifact(path, allow_research=True)[0] for name, path in paths.items()}
    if mutation == "intermediate_score_delay":
        event_time = pd.Timestamp("2024-05-01T20:00:00Z")
        for name, columns in (("collection", ("feature_available_at_utc", "published_at_utc")),
                ("attribution", ("feature_available_at_utc", "event_feature_available_at_utc")),
                ("sentiment", ("event_available_at_utc", "published_at_utc"))):
            for column in columns:
                frames[name].loc[frames[name].event_id.eq("may"), column] = event_time
        frames["sentiment"].loc[frames["sentiment"].event_id.eq("may"), "research_feature_available_at_utc"] = (
            pd.Timestamp("2024-05-02T20:05:00Z"))
    elif mutation == "foreign_score_identity":
        frames["sentiment"].loc[frames["sentiment"].event_id.eq("april"), "security_id"] = "security:foreign"
        frames["sentiment"].loc[frames["sentiment"].event_id.eq("april"), "ticker"] = "FOREIGN"
    else:
        raise AssertionError(mutation)
    record = derivation._write_frame(frames["collection"], paths["collection"], "events", {})
    manifests["collection"]["artifacts"][0].update(record)
    source["collection"]["manifest_sha256"] = _json(directories["collection"] / "_manifest.json", manifests["collection"])
    for name, request_key, kind in (("attribution", "event_attribution_request_sha256", "event_security_relations"),
            ("sentiment", "sentiment_request_sha256", "event_sentiment_research")):
        request_path = directories[name] / "_request.json"
        request = json.loads(request_path.read_text())
        request.pop("request_sha256")
        request["collection_manifest_sha256"] = source["collection"]["manifest_sha256"]
        request_hash = json_sha256(request)
        _json(request_path, {**request, "request_sha256": request_hash})
        child = derivation._write_frame(frames[name], paths[name], kind,
            {request_key: request_hash, "chunk_id": "main", "source_event_artifact_sha256": record["sha256"]})
        manifests[name]["artifacts"][0].update(child)
        manifests[name]["artifacts"][0]["source_event_sha256" if name == "attribution"
            else "source_event_artifact_sha256"] = record["sha256"]
        manifests[name]["request_sha256"] = request_hash
        source[name]["manifest_sha256"] = _json(directories[name] / "_manifest.json", manifests[name])
    return config_path, _json(config_path, config), projection


def test_intermediate_unscored_news_is_unknown_not_known_zero(prepared_inputs: Any, tmp_path: Path) -> None:
    inputs = _rewrite_corrected(tmp_path, prepared_inputs, "intermediate_score_delay")
    config = json.loads(inputs[0].read_text())
    source = preparation._source(tmp_path, config["sources"]["corrected"])
    months = {"2024-05": {"first_decision_utc": "2024-05-01T22:00:00Z", "last_decision_utc": "2024-05-28T22:00:00Z"}}
    with patch.object(preparation, "_guard"):
        preparation._project_source(tmp_path, source, tmp_path / "projected", months, "a" * 64, pd.Timedelta(days=3))
    ledger, _ = load_canonical_artifact(tmp_path / "projected/2024-05/collection/source_collections.parquet", allow_research=True)
    coverage = _coverage_frame(ledger, excluded_security_ids=source["blindspots"],
        relation_chunk_ids={"main"}, sentiment_chunk_ids={"main"})
    coverage["completed_at_utc"] = pd.Timestamp("2024-05-29T00:00:00Z")
    decisions = pd.DataFrame({"security_id": ["security:a", "security:zero", "security:unknown"],
        "ticker": ["AAA", "ZERO", "UNK"], "decision_time_utc": pd.Timestamp("2024-05-01T22:00:00Z")})
    known = _coverage_eligibility(decisions, coverage, source_family="alpaca", lookback=pd.Timedelta(days=1),
        require_completion_by_decision=False)
    assert known.iloc[1] and not known.iloc[2], "observed-empty and unobserved controls must stay distinct"
    assert not known.iloc[0], "news already observed but its score is not yet available at this intermediate cutoff"


def test_corrected_score_identity_must_match_source_event(prepared_inputs: Any, tmp_path: Path) -> None:
    inputs = _rewrite_corrected(tmp_path, prepared_inputs, "foreign_score_identity")
    config = json.loads(inputs[0].read_text())
    source = preparation._source(tmp_path, config["sources"]["corrected"])
    with pytest.raises(DataReadinessError, match="identity"):
        preparation._chunk(source, "main")


def test_preparation_context_failure_cannot_leave_complete_manifest(prepared_inputs: Any, tmp_path: Path) -> None:
    config_path, digest, projection = prepared_inputs

    @contextmanager
    def failing_projection(*args: Any) -> Any:
        with projection(*args) as value:
            yield value
            raise DataReadinessError("synthetic final decision-source verification failure")

    with pytest.raises(DataReadinessError, match="final decision-source verification"):
        _run_preparation(tmp_path, (config_path, digest, failing_projection))
    assert not (tmp_path / "prepared/_manifest.json").exists()


def test_saved_derivation_replay_rejects_original_child_tamper(saved: Any, tmp_path: Path) -> None:
    derived = derive_fixture(saved, tmp_path / "derived")
    (saved.sentiment_dir / "sentiment/crossing.parquet").write_bytes(b"test-only original child tamper")
    with pytest.raises(DataReadinessError):
        derivation.load_initial_fit_issuer_inputs(derived.directory, expected_manifest_sha256=derived.manifest_sha256)


def test_saved_derivation_rechecks_original_child_after_numeric_read(saved: Any, tmp_path: Path) -> None:
    original = derivation._bounded_rows

    def tamper(path: Path, *args: Any, **kwargs: Any) -> pd.DataFrame:
        rows = original(path, *args, **kwargs)
        if path == saved.sentiment_dir / "sentiment/crossing.parquet":
            path.write_bytes(b"test-only mutation after numeric read")
        return rows

    with patch.object(derivation, "_bounded_rows", side_effect=tamper), pytest.raises(DataReadinessError):
        derive_fixture(saved, tmp_path / "derived")


def test_saved_derivation_filters_source_event_clocks_before_reading_scores(saved: Any, tmp_path: Path) -> None:
    path = saved.sentiment_dir / "sentiment/crossing.parquet"
    scores, manifest = load_canonical_artifact(path, allow_research=True)
    scores.loc[scores.event_id.eq("crossing-1"), "research_feature_available_at_utc"] = pd.Timestamp("2024-05-28T21:00:00Z")
    record = derivation._write_frame(scores, path, "event_sentiment_research", manifest["inputs"])
    manifest_path = saved.sentiment_dir / "_manifest.json"
    parent = json.loads(manifest_path.read_text())
    next(row for row in parent["artifacts"] if row["chunk_id"] == "crossing").update(record)
    changed = replace(saved, sentiment_manifest_sha256=_json(manifest_path, parent))
    original = derivation._bounded_rows

    def inspect(path: Path, *args: Any, **kwargs: Any) -> pd.DataFrame:
        rows = original(path, *args, **kwargs)
        if "old-sentiment" in path.parts:
            assert "crossing-1" not in set(rows.event_id), "held-out source event score was numerically materialized"
        return rows

    with patch.object(derivation, "_bounded_rows", side_effect=inspect):
        try:
            derive_fixture(changed, tmp_path / "derived")
        except DataReadinessError:
            pass


def test_interrupted_final_index_write_leaves_resumable_checkpoint(monthly: Any, tmp_path: Path) -> None:
    config, _ = monthly
    original = json.dump
    final = tmp_path / "monthly/_manifest.json"

    def interrupt(value: Any, handle: Any, *args: Any, **kwargs: Any) -> None:
        if (isinstance(value, dict) and value.get("schema") == "market_predictor.initial_fit_catalyst_months"
                and value.get("status") == "complete_research_only"):
            handle.write('{"schema":')
            raise OSError("synthetic interrupted final index write")
        original(value, handle, *args, **kwargs)

    with patch.object(json, "dump", side_effect=interrupt), pytest.raises(OSError, match="interrupted final index"):
        _publish_fixture(tmp_path, config)
    assert not final.exists(), "a partial final index shadows the independently pinned resumable checkpoint"
