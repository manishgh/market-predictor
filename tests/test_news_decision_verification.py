"""Synthetic UNIT fixtures only; no operational source or training evidence."""
from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.research import news_decision_publication as publisher
from market_predictor.research import news_decision_verification as owner
from market_predictor.swing.contracts.holding_materialization import SourcePin
from tests.test_news_decision_publication import _fixture
from tests.test_saved_evaluation_configuration import _write
from tests.test_saved_evaluation_configuration import case as configuration_case


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def case(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    source = _fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(owner.parent_owner, "_guard", lambda: None)
    monkeypatch.setattr(owner, "open_news_source_link_index", publisher.open_news_source_link_index)
    monkeypatch.setattr(owner, "open_news_sentiment_copy_evidence", publisher.open_news_sentiment_copy_evidence)
    # Actual retained parent carries additional physical clocks outside its 124
    # model columns. Keep the full publication mapping; expose only model clocks.
    for part in source.parts.values():
        part.rows["non_model_diagnostic"] = pd.Series([1.0, 2.0], dtype="float32")
        part.rows["available_at_non_model_diagnostic"] = part.rows["decision_time_utc"]
    with owner.parent_owner.verified_news_parent_inputs(root=tmp_path) as parent:
        parent.availability_columns = {**parent.availability_columns,
                                       "non_model_diagnostic": "available_at_non_model_diagnostic"}
    raw = _write(tmp_path, "raw_payload.txt", "UNIT original news payload")
    with publisher.open_news_source_link_index(root=tmp_path, index=source.config) as reader:
        assert reader.producer_source_provenance[raw.path] == raw.sha256
    monkeypatch.setattr(owner.runtime, "guard", lambda *args, **kwargs: None)
    monkeypatch.setattr(owner, "heavy_job_lease", lambda *args, **kwargs: nullcontext())
    code = _write(tmp_path, "verifier.py", "# UNIT verifier inventory")
    monkeypatch.setattr(owner, "_implementation", lambda root: {code.path: code.sha256})
    proof = configuration_case.__wrapped__(tmp_path)["evidence"]
    research = _write(tmp_path, "configs/research.toml", "unit_scope = true")
    parent_request = _write(tmp_path, "parent/_request.json", {
        "policy": {"strategy_contract": proof["strategy"]["original_logical"]},
        "source_files": {proof["strategy"]["original_logical"]["path"]: proof["strategy"]["original_logical"]["sha256"],
                         research.path: research.sha256},
    })
    parent = _write(tmp_path, "parent/_manifest.json", {"request_sha256": parent_request.sha256})
    monkeypatch.setattr(owner.parent_owner, "PUBLICATION", parent)
    ready_config = _json(tmp_path / "readiness_config.json")
    ready_config.update(publication=parent.model_dump(mode="json"), research_contract=research.model_dump(mode="json"),
                        published_profile="technical_relationships")
    ready_pin = _write(tmp_path, "readiness_config.json", ready_config)
    ready = _json(tmp_path / "readiness.json")
    ready.update(publication_sha256=parent.sha256, published_profile="technical_relationships")
    ready["source_files"][ready_pin.path] = ready_pin.sha256
    ready_report = _write(tmp_path, "readiness.json", ready)
    request = _json(tmp_path / "run-1/_request.json")
    request["policy"].update(readiness=ready_report.model_dump(mode="json"), readiness_config=ready_pin.model_dump(mode="json"),
                             published_profile="technical_relationships", feature_profile="technical_relationships")
    request["source_files"].update({ready_pin.path: ready_pin.sha256, ready_report.path: ready_report.sha256})
    run_request = _write(tmp_path, "run-1/_request.json", request)
    run = _json(tmp_path / "run-1/_manifest.json")
    run["request_sha256"] = run_request.sha256
    run_pin = _write(tmp_path, "run-1/_manifest.json", run)
    monkeypatch.setattr(owner, "RELATIONSHIP_RUN", run_pin)
    proof["runs"][1] = run_pin.model_dump(mode="json")
    evidence = _write(tmp_path, "proof.json", proof)
    publisher.materialize_news_decisions(root=tmp_path, config=source.config, output=source.output)
    manifest = tmp_path / source.output / "_manifest.json"
    publication = SourcePin(path=manifest.relative_to(tmp_path).as_posix(), sha256=file_sha256(manifest))
    args = dict(root=tmp_path, publication=publication, output=Path("data/reports/news.json"), configuration_evidence=evidence,
                research_contract=research, strategy_contract=SourcePin.model_validate(proof["strategy"]["current"]),
                temporal_contract=SourcePin.model_validate(proof["temporal"]["current"]))
    return dict(root=tmp_path, source=source, args=args, proof=proof, code=code, raw=raw)


def test_complete_replay_and_lightweight_validator_preserve_false_flags_and_contracts(case: dict[str, Any]) -> None:
    progress: list[dict[str, Any]] = []
    report = owner.verify_news_decision_rows(**case["args"], progress=progress.append)
    assert [item["completed_months"] for item in progress] == [1, 2]
    assert report["profiles"][owner.PROFILE] == dict(rows=4, feature_eligible=2, complete_model_rows=0, clock_violations=0)
    assert report["news_additions_source_replayed"] and report["inherited_parent_rows_exact"]
    assert report["baseline_numerical_replayed"] is False
    assert all(report[key] is False for key in publisher.CLOSED)
    assert report["configuration_correspondence"]["new_fit_authorized"] is False
    path = case["root"] / case["args"]["output"]
    receipt = SourcePin(path=case["args"]["output"].as_posix(), sha256=file_sha256(path))
    reads = list(case["source"].reads)
    verified = owner.validate_news_decision_receipt(case["root"], case["args"]["publication"], receipt)
    assert case["source"].reads == reads  # No numerical replay from consumers.
    assert len(verified.model_columns) == 209 and len(verified.months) == 2
    assert len(verified.request["availability_columns"]) == 210
    assert set(verified.availability_columns) == set(verified.model_columns)
    assert verified.contract_pins["strategy_contract"] == case["args"]["strategy_contract"]
    assert verified.source_files[receipt.path] == receipt.sha256
    with pytest.raises(FileExistsError):
        owner.verify_news_decision_rows(**case["args"])


def _reseal_child(case: dict[str, Any], mutate: Any) -> None:
    root, args = case["root"], case["args"]
    manifest_path = root / args["publication"].path
    manifest = _json(manifest_path)
    child = manifest["months"]["2024-01"]
    path = manifest_path.parent / child["path"]
    frame = pd.read_parquet(path)
    mutate(frame)
    frame.to_parquet(path, index=False)
    sidecar = _json(Path(str(path) + ".manifest.json"))
    sidecar["artifact_sha256"] = file_sha256(path)
    sidecar_pin = _write(root, str(path.relative_to(root)) + ".manifest.json", sidecar)
    child.update(sha256=file_sha256(path), manifest_sha256=sidecar_pin.sha256)
    checkpoint_path = manifest_path.parent / "_checkpoint.json"
    checkpoint = _json(checkpoint_path)
    checkpoint["months"] = manifest["months"]
    pin = _write(root, checkpoint_path.relative_to(root).as_posix(), checkpoint)
    manifest["checkpoint_sha256"] = pin.sha256
    args["publication"] = _write(root, manifest_path.relative_to(root).as_posix(), manifest)


@pytest.mark.parametrize("column,value", [
    ("news_direct_earnings_present_3d", 0.0), ("news_selected_version_ids_json", "[]"),
    ("target", 0.9), ("sample_weight", 0.5), ("feature_eligible", False),
])
def test_rehashed_child_changes_fail_full_parity(case: dict[str, Any], column: str, value: Any) -> None:
    _reseal_child(case, lambda frame: frame.__setitem__(column, frame[column].where(frame.index != 0, value)))
    with pytest.raises((DataReadinessError, AssertionError)):
        owner.verify_news_decision_rows(**case["args"])
    assert not (case["root"] / case["args"]["output"]).exists()


def test_resealed_sentiment_resolution_metadata_fails_replay(case: dict[str, Any]) -> None:
    root, args = case["root"], case["args"]
    manifest_path = root / args["publication"].path
    manifest = _json(manifest_path)
    manifest["months"]["2024-01"]["sentiment_resolution"]["all_resolutions_sha256"] = "0" * 64
    checkpoint_path = manifest_path.parent / "_checkpoint.json"
    checkpoint = _json(checkpoint_path)
    checkpoint["months"] = manifest["months"]
    pin = _write(root, checkpoint_path.relative_to(root).as_posix(), checkpoint)
    manifest["checkpoint_sha256"] = pin.sha256
    args["publication"] = _write(root, manifest_path.relative_to(root).as_posix(), manifest)
    with pytest.raises(DataReadinessError, match="sentiment"):
        owner.verify_news_decision_rows(**args)
    assert not (root / args["output"]).exists()


@pytest.mark.parametrize("fault", ["scope", "bool_count", "source_pin", "missing_month", "future_claim"])
def test_resealed_receipt_cannot_change_contract(case: dict[str, Any], fault: str) -> None:
    report = owner.verify_news_decision_rows(**case["args"])
    if fault == "scope":
        report["news_additions_source_replayed"] = False
    elif fault == "bool_count":
        report["monthly"]["2024-01"]["clock_violations"] = False
    elif fault == "source_pin":
        report["source_files"].pop(case["code"].path)
    elif fault == "missing_month":
        report["monthly"].pop("2024-02")
    else:
        report["training_eligible"] = True
    receipt = _write(case["root"], "data/reports/poison.json", report)
    with pytest.raises(DataReadinessError):
        owner.validate_news_decision_receipt(case["root"], case["args"]["publication"], receipt)


def test_source_mutation_during_context_exit_prevents_receipt(case: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    original = owner.open_news_source_link_index

    @contextmanager
    def altered(**kwargs: Any):
        with original(**kwargs) as reader:
            yield reader
        case["source"].source.write_text("UNIT exit mutation", encoding="utf-8")

    monkeypatch.setattr(owner, "open_news_source_link_index", altered)
    with pytest.raises(DataReadinessError):
        owner.verify_news_decision_rows(**case["args"])
    assert not (case["root"] / case["args"]["output"]).exists()


def test_progress_mutation_of_saved_output_prevents_receipt(case: dict[str, Any]) -> None:
    def mutate(event: dict[str, Any]) -> None:
        path = case["root"] / case["source"].output / "2024-01" / f"{owner.PROFILE}.parquet"
        with path.open("ab") as stream:
            stream.write(b"UNIT poison")
    with pytest.raises(DataReadinessError):
        owner.verify_news_decision_rows(**case["args"], progress=mutate)
    assert not (case["root"] / case["args"]["output"]).exists()


def test_configuration_extra_change_rejected_even_with_current_hash_resealed(case: dict[str, Any]) -> None:
    root, args = case["root"], case["args"]
    path = root / args["temporal_contract"].path
    replacement = _write(root, args["temporal_contract"].path, path.read_text().replace("limit = 10", "limit = 11"))
    case["proof"]["temporal"]["current"] = replacement.model_dump(mode="json")
    args["configuration_evidence"] = _write(root, "proof.json", case["proof"])
    args["temporal_contract"] = replacement
    with pytest.raises(DataReadinessError, match="configuration value differs"):
        owner.verify_news_decision_rows(**args)


def test_caller_and_proof_cannot_substitute_retained_temporal_authority(case: dict[str, Any]) -> None:
    case["proof"]["temporal"]["original_logical"]["sha256"] = "a" * 64
    case["proof"]["temporal"]["original_artifact"]["sha256"] = "a" * 64
    case["args"]["configuration_evidence"] = _write(case["root"], "proof.json", case["proof"])
    with pytest.raises(DataReadinessError, match="temporal authority"):
        owner.verify_news_decision_rows(**case["args"])


def test_saved_receipt_consumption_does_not_claim_fresh_raw_replay(case: dict[str, Any]) -> None:
    report = owner.verify_news_decision_rows(**case["args"])
    assert report["source_index_producer_provenance"][case["raw"].path] == case["raw"].sha256
    assert case["raw"].path not in report["source_replay_files"]
    assert case["raw"].path not in report["source_files"]
    (case["root"] / case["raw"].path).write_text("UNIT later raw mutation", encoding="utf-8")
    receipt = SourcePin(path=case["args"]["output"].as_posix(), sha256=file_sha256(case["root"] / case["args"]["output"]))
    verified = owner.validate_news_decision_receipt(case["root"], case["args"]["publication"], receipt)
    assert verified.raw_sources_rechecked is False
    assert verified.consumption_scope == "saved_receipt_consumption"
    assert verified.receipt["news_additions_source_replayed"] is True  # At receipt creation only.


def test_appended_clock_storage_conversion_is_lossless_only() -> None:
    clock = next(iter(publisher.NEWS_CLOCKS.values()))
    expected = pd.DataFrame({clock: pd.to_datetime(["2024-01-10T22:00:00.000001Z"])} )
    saved = expected.copy()
    saved[clock] = saved[clock].astype("datetime64[ms, UTC]")
    with pytest.raises(DataReadinessError, match="precision"):
        owner._compare_rows(saved, expected)


def test_parent_clock_dtype_is_never_normalized() -> None:
    expected = pd.DataFrame({"available_at_parent": pd.to_datetime(["2024-01-10T22:00:00Z"])} )
    saved = expected.copy()
    saved["available_at_parent"] = saved["available_at_parent"].astype("datetime64[ms, UTC]")
    with pytest.raises(DataReadinessError, match="replay"):
        owner._compare_rows(saved, expected)
