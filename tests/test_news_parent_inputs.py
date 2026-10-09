"""Synthetic monthly UNIT evidence; no actual parent or reconstruction is run."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.research import news_parent_inputs as parent
from market_predictor.swing.contracts.holding_materialization import SourcePin


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return _hash(path)


def _fixture(root: Path, monkeypatch: pytest.MonkeyPatch, *, future_clock: bool = False,
             duplicate: bool = False) -> tuple[dict[str, Any], pd.DataFrame]:
    monkeypatch.setattr(parent, "_guard", lambda: None)
    (root / "unit.py").write_text("# UNIT executed adapter evidence\n", encoding="utf-8")
    monkeypatch.setattr(parent, "_implementation", lambda root: {"unit.py": _hash(root / "unit.py")})
    names = [f"feature_{position}" for position in range(124)]
    clocks = {name: f"available_at_{name}" for name in names}
    original_parent = {"path": "baseline/_manifest.json", "sha256": "b" * 64}
    original_receipt = {"path": "baseline/receipt.json", "sha256": "c" * 64}
    source_basis = {"basis": "unit_original"}
    sources = {"availability_semantics": "historical_proxy"}
    request = {"schema": "market_predictor.return_relationship_request", "rows": 59,
               "training_eligible": False, "promotion_eligible": False,
               "model_columns": names, "availability_columns": clocks, "parent_publication": original_parent,
               "parent_saved_row_verification": original_receipt, "sources": sources,
               "source_basis": source_basis, "stock_inventory": {"unit": "unchanged ownership"}}
    request_sha = _json(root / "parent/_request.json", request)
    files = {"parent/_request.json": request_sha}
    months: dict[str, Any] = {}
    replay_months: dict[str, Any] = {}
    identifiers: list[str] = []
    first: pd.DataFrame | None = None
    for position, month in enumerate(parent.MONTHS):
        identifier = "decision-0" if duplicate and position == 1 else f"decision-{position}"
        identifiers.append(identifier)
        time = pd.Timestamp(f"{month}-15T21:00:00Z")
        values: dict[str, Any] = {"decision_id": [identifier], "security_id": ["company"], "ticker": ["A"],
                                  "decision_time_utc": [time], "session_date_et": [time.date()],
                                  "feature_profile": ["technical_relationships"], "feature_eligible": [position % 2 == 0],
                                  "cross_section_eligible": [True], "future_return": [None if position == 1 else 0.15],
                                  **{name: pd.Series([0.25], dtype="float32") for name in names},
                                  **{clock: [time + pd.Timedelta(minutes=1) if future_clock else time] for clock in clocks.values()}}
        frame = pd.DataFrame(values)
        folder = root / "parent" / month
        folder.mkdir(parents=True)
        path = folder / "technical_relationships.parquet"
        frame.to_parquet(path, index=False)
        if first is None:
            first = pd.read_parquet(path)
        sha = _hash(path)
        relative = path.relative_to(root).as_posix()
        sidecar = {"schema": "market_data.artifact_manifest.v1", "canonical_schema_version": "market_data.v1",
                   "artifact_type": "swing_return_relationships", "artifact_sha256": sha, "rows": 1,
                   "production_ready": False, "columns": list(frame), "inputs": {"request_sha256": request_sha}}
        sidecar_sha = _json(Path(str(path) + ".manifest.json"), sidecar)
        files.update({relative: sha, relative + ".manifest.json": sidecar_sha})
        digest = json_sha256([identifier])
        child = {"path": f"{month}/technical_relationships.parquet", "sha256": sha, "manifest_sha256": sidecar_sha,
                 "rows": 1, "decision_ids_sha256": digest, "model_columns": names, "availability_columns": clocks}
        months[month] = {"rows": 1, "decision_ids_sha256": digest, "profiles": {"technical_relationships": child}}
        replay_months[month] = {"rows": 1, "decision_ids_sha256": digest, "inherited_predictors_equal": True}
    manifest = {"schema": "market_predictor.return_relationship_publication", "status": "complete_research_only",
                "request_sha256": request_sha, "rows": 59, "months": months, "exclusions_added": [],
                "training_eligible": False, "promotion_eligible": False}
    publication = SourcePin(path="parent/_manifest.json", sha256=_json(root / "parent/_manifest.json", manifest))
    receipt = {"manifest_sha256": publication.sha256, "scope": "published_return_relationship_population_clocks_original_targets",
               "status": "passed", "months": 59, "unique_decisions": 59, "outcome_filtered_rows": 0,
               "matched_profile_population": True, "original_outcome_values_exact": True,
               "training_eligible": False, "promotion_eligible": False}
    receipt_pin = SourcePin(path="parent/receipt.json", sha256=_json(root / "parent/receipt.json", receipt))
    config = {"historical_relationship_publication": publication.model_dump(mode="json"),
              "historical_relationship_receipt": receipt_pin.model_dump(mode="json"),
              "historical_parent_publication": original_parent, "historical_parent_receipt": original_receipt}
    config_pin = SourcePin(path="config.json", sha256=_json(root / "config.json", config))
    files.update({publication.path: publication.sha256, receipt_pin.path: receipt_pin.sha256, config_pin.path: config_pin.sha256})
    replay = {"schema": "market_predictor.original_relationship_replay", "status": "passed_original_snapshot_replay",
              "rows": 59, "decision_ids_sha256": json_sha256(sorted(identifiers)), "differences": [],
              "additions_source_replayed": True, "inherited_predictors_equal": True, "usable_ohlcv_validated": True,
              "source_window_bounds": ["2018-05-29", "2024-05-28"],
              "historical_implementation_disposition": "immutable_provenance_not_executed",
              "config": config_pin.model_dump(mode="json"), "original_sources": sources, "source_basis": source_basis,
              "original_stock_inventory_sha256": json_sha256(request["stock_inventory"]), "months": replay_months,
              "source_files": files, "current_implementation_files": {"old_replay.py": "d" * 64},
              "historical_implementation_files": {"old_producer.py": "e" * 64},
              **dict.fromkeys(("training_eligible", "promotion_eligible", "serving_eligible", "consumer_admission_changed",
                              "provider_transport_admitted", "historical_first_seen_proven", "quarantined_source_inputs_admitted",
                              "saved_rows_rewritten", "targets_read_or_replayed", "baseline_price_features_replayed",
                              "raw_news_aggregates_replayed", "current_archive_equivalence_proven"), False)}
    _set_replay(root, monkeypatch, replay)
    monkeypatch.setattr(parent, "PUBLICATION", publication)
    monkeypatch.setattr(parent, "SAVED_ROW_RECEIPT", receipt_pin)
    monkeypatch.setattr(parent, "EXPECTED_ROWS", 59)
    monkeypatch.setattr(parent, "EXPECTED_IDS", replay["decision_ids_sha256"])
    assert first is not None
    return replay, first


def _set_replay(root: Path, monkeypatch: pytest.MonkeyPatch, replay: dict[str, Any]) -> None:
    monkeypatch.setattr(parent, "ORIGINAL_REPLAY", SourcePin(path="original-replay.json",
                                                             sha256=_json(root / "original-replay.json", replay)))


def test_exact_months_preserved_no_producer_execution(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _, expected = _fixture(tmp_path, monkeypatch)
    with parent.verified_news_parent_inputs(root=tmp_path) as context:
        actual = context.read_month("2019-07")
        pd.testing.assert_frame_equal(actual, expected, check_exact=True)
        assert len(context.model_columns) == 124
        assert len(list(context.iter_months())) == 59
        assert "old_replay.py" not in context.implementation_files
        assert context.historical_implementation_files["original_replay"] == {"old_replay.py": "d" * 64}
        request = context.request
        request["model_columns"][0] = "poison"
        assert context.model_columns[0] == "feature_0"
    with pytest.raises(DataReadinessError, match="closed"):
        context.read_month("2019-07")


@pytest.mark.parametrize("field,value", [("training_eligible", True), ("targets_read_or_replayed", True),
                                         ("current_archive_equivalence_proven", True), ("additions_source_replayed", False),
                                         ("status", "passed"), ("rows", 58), ("decision_ids_sha256", "0" * 64)])
def test_changed_closed_receipt_claim_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: Any) -> None:
    replay, _ = _fixture(tmp_path, monkeypatch)
    replay[field] = value
    _set_replay(tmp_path, monkeypatch, replay)
    with pytest.raises(DataReadinessError, match="replay"):
        with parent.verified_news_parent_inputs(root=tmp_path):
            pytest.fail("poisoned receipt admitted")


def test_no_global_duplicate_decisions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fixture(tmp_path, monkeypatch, duplicate=True)
    with pytest.raises(DataReadinessError, match="global decision ownership"):
        with parent.verified_news_parent_inputs(root=tmp_path):
            pytest.fail("duplicate population admitted")


def test_future_feature_clock_rejected_without_reconstruction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _fixture(tmp_path, monkeypatch, future_clock=True)
    with parent.verified_news_parent_inputs(root=tmp_path) as context:
        with pytest.raises(DataReadinessError, match="future"):
            context.read_month("2019-07")


def test_closed_replay_must_bind_exact_parent_child(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay, _ = _fixture(tmp_path, monkeypatch)
    replay["source_files"].pop("parent/2019-07/technical_relationships.parquet")
    _set_replay(tmp_path, monkeypatch, replay)
    with pytest.raises(DataReadinessError, match="omits exact parent"):
        with parent.verified_news_parent_inputs(root=tmp_path):
            pytest.fail("unbound child admitted")


@pytest.mark.parametrize("path", ["parent/2019-07/technical_relationships.parquet", "unit.py"])
def test_final_recheck_rejects_changed_consumed_bytes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str) -> None:
    _fixture(tmp_path, monkeypatch)
    with pytest.raises(DataReadinessError):
        with parent.verified_news_parent_inputs(root=tmp_path) as context:
            context.read_month("2019-07")
            with (tmp_path / path).open("ab") as stream:
                stream.write(b"changed")


def test_wrong_global_population_hash_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    replay, _ = _fixture(tmp_path, monkeypatch)
    replay["decision_ids_sha256"] = "1" * 64
    monkeypatch.setattr(parent, "EXPECTED_IDS", "1" * 64)
    _set_replay(tmp_path, monkeypatch, replay)
    with pytest.raises(DataReadinessError, match="global population"):
        with parent.verified_news_parent_inputs(root=tmp_path):
            pytest.fail("bad decision hash admitted")
