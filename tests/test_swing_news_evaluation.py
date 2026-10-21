"""Synthetic news evaluation boundaries; no data publication, training or accounting."""
from __future__ import annotations

import copy
import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.store import file_sha256, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.research import swing_oof_policy_evaluation as oof
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_training import ReturnTrainingPolicy
from market_predictor.swing.contracts.training_readiness import TrainingReadinessPolicy
from market_predictor.swing.training import return_comparison as comparison

REPO = Path(__file__).resolve().parents[1]


def _policy(news: bool) -> dict[str, Any]:
    value = json.loads((REPO / "configs/swing_return_training.json").read_text())
    if news:
        value.update(feature_profile=oof.NEWS_PROFILE, published_profile=oof.NEWS_PROFILE)
    return value


def _requests() -> list[dict[str, Any]]:
    return [{"policy": _policy(news), "feature_names": ["unit"], "source_files": {},
             "versions": {"numpy": "synthetic-identical"}, "folds": [{"number": 1, "train": ["unit"]}],
             "memory_policy": {"maximum_system_used_percent": 91.0 if news else 90.0,
                "minimum_system_free_gib": None, "maximum_process_memory_gib": 5.0, "process_headroom_gib": .75}}
            for news in (False, True)]


def test_news_ceiling_amendment_records_both_exact_policies() -> None:
    requests = _requests()
    result = comparison._paired_requests(requests)
    assert result["news_system_ceiling_amendment_applied"] is True
    assert result["baseline"] == requests[0]["memory_policy"]
    assert result["candidate"] == requests[1]["memory_policy"]
    assert comparison._paired_requests(list(reversed(requests)))["candidate"] == result["baseline"]


@pytest.mark.parametrize("change", ["ceiling", "floor", "process", "headroom", "version", "fold", "weight", "alpha", "type", "non_news"])
def test_news_exception_cannot_waive_other_difference(change: str) -> None:
    requests = _requests()
    candidate = requests[1]
    if change in {"ceiling", "floor", "process", "headroom"}:
        key, value = {"ceiling": ("maximum_system_used_percent", 92.0), "floor": ("minimum_system_free_gib", .75),
                      "process": ("maximum_process_memory_gib", 6.0), "headroom": ("process_headroom_gib", .5)}[change]
        candidate["memory_policy"][key] = value
    elif change == "version":
        candidate["versions"]["numpy"] = "different"
    elif change == "fold":
        candidate["folds"][0]["train"] = ["other"]
    elif change == "weight":
        candidate["policy"]["weighting"] = "per_row"
    elif change == "alpha":
        candidate["policy"]["linear"]["alpha"] = 2.0
    elif change == "type":
        candidate["policy"]["linear"]["alpha"] = 1  # Original saved JSON uses 1.0.
    else:
        candidate["policy"] = _policy(False)
    with pytest.raises(DataReadinessError):
        comparison._paired_requests(requests)


def test_memory_guard_news_delegation_does_not_raise_baseline_ceiling(monkeypatch: pytest.MonkeyPatch) -> None:
    from market_predictor.research import news_runtime_memory
    calls: list[str] = []
    monkeypatch.setattr(news_runtime_memory, "guard", lambda **kwargs: calls.append("news"))
    monkeypatch.setattr(comparison, "assert_peak_memory_budget", lambda **kwargs: None)
    monkeypatch.setattr(comparison, "_guard", lambda: calls.append("comparison-baseline"))
    monkeypatch.setattr(oof, "_guard", lambda: calls.append("oof-baseline"))
    comparison._comparison_guard(False)
    comparison._comparison_guard(True)
    oof._profile_guard("existing_technical")
    oof._profile_guard(oof.NEWS_PROFILE)
    assert calls == ["comparison-baseline", "news", "oof-baseline", "news"]


def _json(root: Path, name: str, value: dict[str, Any]) -> SourcePin:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return SourcePin(path=name, sha256=file_sha256(path))


def test_news_metadata_reader_preserves_hash_json_and_size_bounds(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    payload = b'{"unit_only":true}' + b" " * (8 * 1024**2)
    path.write_bytes(payload)
    pin = SourcePin(path=path.name, sha256=file_sha256(path))
    assert oof._object(tmp_path, pin, {}, news=True) == {"unit_only": True}
    with pytest.raises(DataReadinessError, match="exceeds bound"):
        oof._object(tmp_path, pin, {})
    path.write_bytes(payload + b" ")
    with pytest.raises(DataReadinessError, match="OOF input changed"):
        oof._object(tmp_path, pin, {}, news=True)
    for malformed in (b'{"duplicate":1,"duplicate":2}', b'{"invalid":NaN}'):
        path.write_bytes(malformed + b" " * (8 * 1024**2))
        with pytest.raises(ValueError, match="duplicate key|non-finite"):
            oof._object(tmp_path, SourcePin(path=path.name, sha256=file_sha256(path)), {}, news=True)
    with path.open("wb") as stream:
        stream.truncate(32 * 1024**2 + 1)
    with pytest.raises(DataReadinessError, match="exceeds bound"):
        oof._object(tmp_path, SourcePin(path=path.name, sha256=file_sha256(path)), {}, news=True)


@pytest.fixture
def news_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                  request: pytest.FixtureRequest) -> dict[str, Any]:
    large_metadata = getattr(request, "param", False)

    def metadata_json(name: str, value: dict[str, Any]) -> SourcePin:
        pin = _json(tmp_path, name, value)
        if large_metadata:
            with (tmp_path / name).open("ab") as stream:
                stream.write(b" " * (8 * 1024**2))
        return SourcePin(path=pin.path, sha256=file_sha256(tmp_path / name))

    monkeypatch.setattr(oof, "_profile_guard", lambda profile: None)
    readiness = json.loads((REPO / "configs/swing_training_readiness.json").read_text())
    strategy = load_strategy_contract(REPO / readiness["strategy_contract"]["path"])
    policy = ReturnTrainingPolicy.model_validate(_policy(True))
    names = [*oof.swing_model_feature_columns(contract=strategy, catalyst=False), *oof.RETURN_RELATIONSHIP_COLUMNS, *oof.NEWS_COLUMNS]
    sessions = (date(2019, 7, 9), date(2019, 7, 10))
    cutoff = swing_prediction_cutoffs(pd.Series(sessions))
    clocks = {name: f"available_at_{name}" for name in names}
    physical_clocks = {**clocks, "retained_parent_non_model_diagnostic": "available_at_retained_parent_non_model_diagnostic"}
    # Actual retained parents also carry clocks for physical, non-model fields.
    # Preserve that metadata without admitting another predictor.
    clocks['unused_physical_measurement'] = 'available_at_unused_physical_measurement'
    frame = pd.DataFrame({"decision_id": ["a", "b"], "security_id": "unit", "ticker": "UNIT", "sector": "unit",
        "session_date_et": sessions, "decision_time_utc": cutoff, "feature_eligible": [True, False],
        "daily_bar_count": 250, "primary_benchmark": "XLF",
        "feature_profile": oof.NEWS_PROFILE, **{clock: cutoff - pd.Timedelta(minutes=1) for clock in physical_clocks.values()}})
    # No numerical feature or target columns: evaluation must only read metadata/clocks.
    frame[clocks[names[-1]]] = pd.NaT
    parent_request = metadata_json("data/features/unit/_request.json", {"unit_only": True})
    path = tmp_path / f"data/features/unit/2019-07/{oof.NEWS_PROFILE}.parquet"
    path.parent.mkdir(parents=True)
    frame.to_parquet(path, index=False)
    digest = file_sha256(path)
    sidecar = _json(tmp_path, manifest_path_for(path).relative_to(tmp_path).as_posix(), {
        "artifact_type": "swing_news_decision_features", "artifact_sha256": digest, "rows": 2,
        "production_ready": False, "inputs": {"request_sha256": parent_request.sha256}})
    child = {"path": f"2019-07/{oof.NEWS_PROFILE}.parquet", "sha256": digest, "manifest_sha256": sidecar.sha256,
        "rows": 2, "decision_ids_sha256": json_sha256(["a", "b"]), "model_columns": names, "availability_columns": physical_clocks}
    manifest = {"request_sha256": parent_request.sha256, "months": {"2019-07": child}}
    publication = metadata_json("data/features/unit/_manifest.json", manifest)
    request = {"feature_names": names, "input_rows": 2, "input_decision_ids_sha256": json_sha256(["a", "b"]),
        "source_files": {path.relative_to(tmp_path).as_posix(): digest, sidecar.path: sidecar.sha256}}
    return dict(root=tmp_path, publication=publication, request=request, policy=policy, strategy=strategy,
                sessions=sessions, pins={})


@pytest.mark.parametrize("news_metadata", [False, True], indirect=True)
def test_flat_209_metadata_preserves_population_and_missing_clocks(news_metadata: dict[str, Any]) -> None:
    result = oof._read_parent_metadata(**news_metadata)
    assert len(news_metadata["request"]["feature_names"]) == 209
    assert result.decision_id.tolist() == ["a", "b"]
    assert result.feature_eligible.tolist() == [True, False]
    assert (result.maximum_feature_available_at_utc == result.decision_time_utc - pd.Timedelta(minutes=1)).all()


@pytest.mark.parametrize("change", ["order", "training_pin", "artifact", "future_clock"])
def test_flat_news_metadata_rejects_changed_evidence(news_metadata: dict[str, Any], change: str) -> None:
    args = news_metadata
    root, pub = args["root"], args["publication"]
    manifest = json.loads((root / pub.path).read_text())
    child = manifest["months"]["2019-07"]
    path = (root / pub.path).parent / child["path"]
    if change == "order":
        args["request"]["feature_names"].reverse()
    elif change == "training_pin":
        args["request"]["source_files"][path.relative_to(root).as_posix()] = "a" * 64
    elif change == "artifact":
        path.write_bytes(b"changed")
    else:
        frame = pd.read_parquet(path)
        frame[child["availability_columns"][oof.NEWS_COLUMNS[0]]] = frame.decision_time_utc + pd.Timedelta(seconds=1)
        frame.to_parquet(path, index=False)
        child["sha256"] = file_sha256(path)
        sidecar = json.loads(manifest_path_for(path).read_text())
        sidecar["artifact_sha256"] = child["sha256"]
        sidecar_pin = _json(root, manifest_path_for(path).relative_to(root).as_posix(), sidecar)
        child["manifest_sha256"] = sidecar_pin.sha256
        args["request"]["source_files"].update({path.relative_to(root).as_posix(): child["sha256"], sidecar_pin.path: sidecar_pin.sha256})
        args["publication"] = _json(root, pub.path, manifest)
    with pytest.raises(DataReadinessError):
        oof._read_parent_metadata(**args)


@pytest.mark.parametrize("poison", [None, "receipt", "source", "clock_order"])
def test_news_receipt_consumption_binds_current_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, poison: str | None) -> None:
    config = json.loads((REPO / "configs/swing_training_readiness.json").read_text())
    config["published_profile"] = oof.NEWS_PROFILE
    receipt = _json(tmp_path, "receipt.json", {"source_link_index": {"path": "index.json", "sha256": "c" * 64}})
    config["saved_row_verification"] = receipt.model_dump(mode="json")
    config["maximum_system_used_percent"] = 91.0
    policy = TrainingReadinessPolicy.model_validate(config)
    source = _json(tmp_path, "consumed.json", {"unit": True})
    verified = SimpleNamespace(request={"profile_sha256": "p", "policy": {"source_link_index": {"path": "index.json", "sha256": "c" * 64}}},
        contract_pins={role: getattr(policy, role) for role in ("strategy_contract", "research_contract", "temporal_contract")},
        model_columns=("unit",), availability_columns={"unit": "clock"}, source_files={source.path: source.sha256}, implementation_files={})
    monkeypatch.setattr(oof, "validate_news_decision_receipt", lambda *args: verified)
    report = {"saved_row_verification_sha256": receipt.sha256, "profile_sha256": "p", "model_columns": ["unit"],
        "availability_columns": {"unit": "clock"}, "source_link_index": verified.request["policy"]["source_link_index"],
        "news_additions_source_replayed": True, "inherited_parent_rows_exact": True, "baseline_numerical_replayed": False,
        "training_eligible": False, "serving_eligible": False, "promotion_eligible": False}
    request = {"source_files": {source.path: source.sha256, receipt.path: receipt.sha256}}
    if poison == "receipt":
        report["saved_row_verification_sha256"] = "a" * 64
    elif poison == "source":
        request["source_files"][source.path] = "a" * 64
    elif poison == "clock_order":
        report["availability_columns"] = {"unit": "wrong"}
    if poison:
        with pytest.raises(DataReadinessError):
            oof._news_receipt(tmp_path, policy.publication, request, report, policy, {})
    else:
        result = oof._news_receipt(tmp_path, policy.publication, request, report, policy, {})
        assert result["receipt"] == receipt.model_dump(mode="json") and result["raw_sources_rechecked"] is False


def test_current_roster_replaces_only_untrained_slot() -> None:
    roster = copy.deepcopy(oof.CAMPAIGN_ROSTER)
    assert roster["feature_profiles"] == ["existing_technical", "technical_relationships", oof.NEWS_PROFILE]
    assert len(roster["feature_profiles"]) * len(roster["learner_families"]) == roster["maximum_learned_specifications"] == 6
    assert 6 * len(roster["exit_policies"]) == roster["maximum_model_policy_trials"] == 12
