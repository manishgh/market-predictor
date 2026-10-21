"""Three-row synthetic consumer UNIT tests; source replay is explicitly mocked.

These fixtures are not operational news receipts or permission to fit a model.
Canonical child reads, pin checks, readiness masks and return loading are real.
"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import file_sha256, manifest_path_for, write_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.research import swing_return_inputs as loader
from market_predictor.research import swing_training_readiness as readiness
from market_predictor.research.news_decision_features import NEWS_COLUMNS
from market_predictor.research.news_decision_publication import ARTIFACT_TYPE, CLOSED, PROFILE
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_COLUMNS
from market_predictor.swing.contracts.return_training import ReturnTrainingPolicy
from market_predictor.swing.contracts.training_readiness import TrainingReadinessPolicy
from market_predictor.swing.labels.fixed_horizon_readiness import CONTEXT_COLUMNS, RETURN_COLUMNS
from market_predictor.swing.training.return_validation import date_balanced_weights, security_transfer_mask
from tests.test_swing_training_readiness import _json, _run
from tests.test_swing_training_readiness import publication as publication

REPO = Path(__file__).resolve().parents[1]


def test_news_policy_changes_no_learning_setting() -> None:
    payload = json.loads((REPO / "configs/swing_return_training.json").read_text())
    baseline = ReturnTrainingPolicy.model_validate(payload)
    news = ReturnTrainingPolicy.model_validate({**payload, "feature_profile": PROFILE, "published_profile": PROFILE})
    excluded = {"feature_profile", "published_profile"}
    assert news.model_dump(exclude=excluded) == baseline.model_dump(exclude=excluded)


@pytest.mark.parametrize("feature,published", [(PROFILE, "technical_market"), (PROFILE, "technical_relationships"),
    ("technical_relationships", PROFILE), ("existing_technical", PROFILE), (PROFILE, "catalyst_full")])
def test_news_profile_cannot_substitute_another_publication(feature: str, published: str) -> None:
    payload = json.loads((REPO / "configs/swing_return_training.json").read_text())
    with pytest.raises(ValidationError):
        ReturnTrainingPolicy.model_validate({**payload, "feature_profile": feature, "published_profile": published})


@pytest.mark.parametrize("profile,ceiling,allowed", [(PROFILE, 91, True), (PROFILE, 90, False),
    (PROFILE, 91.01, False), ("technical_market", 90, True), ("technical_market", 91, False),
    ("technical_relationships", 91, False), ("technical_relationships_issuer_reaction", 91, False)])
def test_memory_policy_is_profile_specific(profile: str, ceiling: float, allowed: bool) -> None:
    payload = json.loads((REPO / "configs/swing_training_readiness.json").read_text())
    payload.update(published_profile=profile, maximum_system_used_percent=ceiling)
    if allowed:
        assert TrainingReadinessPolicy.model_validate(payload).maximum_system_used_percent == ceiling
    else:
        with pytest.raises(ValidationError):
            TrainingReadinessPolicy.model_validate(payload)


def test_news_guard_delegates_to_bounded_news_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = json.loads((REPO / "configs/swing_training_readiness.json").read_text())
    policy = TrainingReadinessPolicy.model_validate({**payload, "published_profile": PROFILE,
                                                   "maximum_system_used_percent": 91})
    guard = Mock()
    old = Mock(side_effect=AssertionError("old immediate memory owner called"))
    monkeypatch.setattr(readiness.news_runtime_memory, "guard", guard)
    monkeypatch.setattr(readiness, "assert_memory_budget", old)
    readiness._guard(policy)
    guard.assert_called_once_with(stage="swing news training readiness")
    old.assert_not_called()


@pytest.fixture
def news_unit(publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
              request: pytest.FixtureRequest) -> dict[str, Any]:
    """Adapt tiny generic UNIT rows; mock only the independent source-replay owner."""
    state = publication
    root = state["root"]
    large_metadata = getattr(request, "param", False)

    def metadata_json(path: Path, payload: Any) -> str:
        _json(path, payload)
        if large_metadata:
            with path.open("ab") as stream:
                stream.write(b" " * (8 * 1024**2))
        return file_sha256(path)

    package = root / "src/market_predictor"
    for name in readiness.NEWS_IMPLEMENTATION_PATHS:
        target = package / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "src/market_predictor" / name, target)
    monkeypatch.setattr(loader, "__file__", str(package / "research/swing_return_inputs.py"))
    monkeypatch.setattr(loader, "_guard", lambda policy: None)
    monkeypatch.setattr(loader, "release_process_memory", lambda: None)
    monkeypatch.setattr(loader, "load_temporal_manifest_config", lambda path: SimpleNamespace(
        calendar="XNYS", label_horizon_sessions=10, warmup_sessions=250, initial_fit_end=date(2019, 7, 31)))
    monkeypatch.setattr(loader, "build_temporal_schedule", readiness.build_temporal_schedule)
    frame = pd.read_parquet(state["paths"]["technical_market"])
    base_names = state["manifest"]["months"]["2019-07"]["profiles"]["technical_market"]["model_columns"]
    names = (*base_names, *RETURN_RELATIONSHIP_COLUMNS, *NEWS_COLUMNS)
    clocks = {name: f"available_at_{name}" for name in names}
    added = (*RETURN_RELATIONSHIP_COLUMNS, *NEWS_COLUMNS)
    values = {name: [1.0, 2.0, np.nan] for name in added}
    values.update({name: [0.0, 0.0, np.nan] for name in NEWS_COLUMNS if "source_coverage_known" in name})
    extra = pd.DataFrame({**values, **{clocks[name]: frame.decision_time_utc for name in added}})
    frame = pd.concat((frame, extra), axis=1)
    identities = pd.Series([f"issuer-{number}" for number in range(20)])
    holdout = security_transfer_mask(identities)
    frame["security_id"] = [identities[~holdout].iloc[0], identities[holdout].iloc[0], identities[~holdout].iloc[0]]
    frame["news_projection_status"] = ["observed_retained_documents", "observed_retained_documents", "source_history_unresolved"]
    frame["news_suppressed_story_count"] = [0, 1, 0]
    frame["news_uncertainty_count"] = [0, 0, 1]
    directory = root / "data/features/news-unit"
    contracts = {key: state["policy"][key] for key in ("research_contract", "strategy_contract", "temporal_contract")}
    source_files = {pin["path"]: pin["sha256"] for pin in contracts.values()}
    source_index = {"path": "data/features/synthetic-source-index.json", "sha256": _json(
        root / "data/features/synthetic-source-index.json", {"synthetic_unit_only": True})}
    source_files[source_index["path"]] = source_index["sha256"]
    request = {"schema": "market_predictor.news_decision_publication_request", "profile": PROFILE,
        "source_files": {source_index["path"]: source_index["sha256"]},
        "policy": {"source_link_index": source_index}, "cohort_sha256": "c" * 64,
        "profile_sha256": "a" * 64, "model_columns": list(names), "availability_columns": clocks, **CLOSED}
    request_hash = metadata_json(directory / "_request.json", request)
    path = directory / "2019-07" / f"{PROFILE}.parquet"
    audit = CanonicalAuditReport(checks=(CanonicalAuditCheck(name="synthetic_unit", status="pass", failures=0,
        rows_checked=3, detail="Synthetic UNIT rows, independent source owner mocked."),))
    write_canonical_artifact(frame, path, artifact_type=ARTIFACT_TYPE, audit=audit,
        inputs={"request_sha256": request_hash}, production_ready=False)
    child = {"path": f"2019-07/{PROFILE}.parquet", "sha256": file_sha256(path),
        "manifest_sha256": file_sha256(manifest_path_for(path)), "rows": len(frame),
        "decision_ids_sha256": json_sha256(sorted(frame.decision_id)), "model_columns": list(names), "availability_columns": clocks}
    manifest = {"schema": "market_predictor.news_decision_publication", "status": "complete_research_only",
        "request_sha256": request_hash, "rows": 3, "months": {"2019-07": child}, **CLOSED}
    manifest_path = directory / "_manifest.json"
    manifest_hash = metadata_json(manifest_path, manifest)
    receipt = {"months": 1, "source_files": source_files, "source_link_index": source_index,
        "source_replay_files": {"synthetic_unopened_raw_provenance.json": "b" * 64},
        "profiles": {PROFILE: {"rows": 3, "feature_eligible": 3, "complete_model_rows": 2, "clock_violations": 0}}, **CLOSED}
    receipt_path = root / "data/reports/news-unit-receipt.json"
    receipt_hash = metadata_json(receipt_path, receipt)
    state["policy"].update(published_profile=PROFILE, maximum_system_used_percent=91,
        publication={"path": manifest_path.relative_to(root).as_posix(), "sha256": manifest_hash},
        saved_row_verification={"path": receipt_path.relative_to(root).as_posix(), "sha256": receipt_hash})
    state["config_sha256"] = _json(state["config"], state["policy"])
    verified = SimpleNamespace(request=request, manifest=manifest, source_files=source_files, implementation_files={},
        model_columns=names, availability_columns=clocks, months=manifest["months"], receipt=receipt,
        contract_pins={key: SourcePin.model_validate(pin) for key, pin in contracts.items()})
    verification = Mock(return_value=verified)
    monkeypatch.setattr(readiness, "validate_news_decision_receipt", verification)
    monkeypatch.setattr(loader, "validate_news_decision_receipt", verification)
    state.update(frame=frame, verified=verified, verification=verification, receipt=receipt,
                 manifest=manifest, news_path=path, names=names)
    return state


def _policy(state: dict[str, Any]) -> ReturnTrainingPolicy:
    payload = json.loads((REPO / "configs/swing_return_training.json").read_text())
    payload.update(feature_profile=PROFILE, published_profile=PROFILE,
        readiness={"path": state["output"].relative_to(state["root"]).as_posix(), "sha256": file_sha256(state["output"])},
        readiness_config={"path": state["config"].relative_to(state["root"]).as_posix(), "sha256": state["config_sha256"]})
    return ReturnTrainingPolicy.model_validate(payload)


@pytest.mark.parametrize("news_unit", [False, True], indirect=True)
def test_flat_news_readiness_and_loader_preserve_population_targets_weights(news_unit: dict[str, Any]) -> None:
    report = _run(news_unit)
    data = loader.load_return_inputs(news_unit["root"], _policy(news_unit))
    assert data.features.shape == (3, 209)
    assert tuple(data.features.columns) == news_unit["names"]
    pd.testing.assert_frame_equal(data.features, news_unit["frame"].loc[:, list(news_unit["names"])].astype("float32"))
    columns = [*CONTEXT_COLUMNS, *RETURN_COLUMNS]
    pd.testing.assert_frame_equal(data.metadata.loc[:, columns], news_unit["frame"].loc[:, columns])
    np.testing.assert_array_equal(date_balanced_weights(data.metadata.session_date_et),
                                  date_balanced_weights(news_unit["frame"].session_date_et))
    assert data.metadata.security_holdout.tolist() == [False, True, False]
    assert data.metadata.fixed_horizon_supervision_available.tolist() == [True, True, False]
    assert "synthetic_unopened_raw_provenance.json" not in data.pins
    assert report["raw_sources_rechecked"] is False
    assert report["saved_row_verification_source_provenance"] == news_unit["receipt"]["source_replay_files"]
    assert data.features.iloc[2].isna().all()
    assert report["news_additions_source_replayed"] is True and report["inherited_parent_rows_exact"] is True
    assert report["baseline_numerical_replayed"] is False and "qualification_source_replayed" not in report
    assert all(report[name] is False for name in ("training_eligible", "promotion_eligible", "serving_eligible"))
    coverage = report["news_diagnostics"]["2019-07"]["source_coverage_rows"]
    assert all(counts == {"known": 0, "unknown": 2, "unavailable": 1} for counts in coverage.values())
    assert data.pins[news_unit["policy"]["saved_row_verification"]["path"]] == news_unit["policy"]["saved_row_verification"]["sha256"]


def test_missing_news_receipt_rejects_before_numeric_read(news_unit: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    news_unit["verification"].side_effect = DataReadinessError("synthetic missing exact receipt")
    read = Mock(side_effect=AssertionError("numeric load before receipt"))
    monkeypatch.setattr(readiness, "load_canonical_artifact", read)
    with pytest.raises(DataReadinessError, match="missing exact receipt"):
        _run(news_unit)
    read.assert_not_called()


@pytest.mark.parametrize("field,value", [("source_link_index", {}), ("news_additions_source_replayed", False),
    ("inherited_parent_rows_exact", False), ("profile_sha256", "f" * 64), ("model_columns", []),
    ("training_eligible", True), ("serving_eligible", True), ("news_contract_pins", {}),
    ("raw_sources_rechecked", True), ("saved_row_verification_source_provenance", {})])
def test_news_report_poison_rejects_before_numeric_read(news_unit: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
                                                     field: str, value: Any) -> None:
    report = _run(news_unit)
    _json(news_unit["output"], {**report, field: value})
    policy = _policy(news_unit)
    read = Mock(side_effect=AssertionError("numeric load before evidence check"))
    monkeypatch.setattr(loader, "load_canonical_artifact", read)
    with pytest.raises(DataReadinessError):
        loader.load_return_inputs(news_unit["root"], policy)
    read.assert_not_called()


@pytest.mark.parametrize("field", list(CLOSED))
def test_news_source_flags_must_remain_false(news_unit: dict[str, Any], field: str) -> None:
    news_unit["manifest"][field] = True
    pin = news_unit["policy"]["publication"]
    pin["sha256"] = _json(news_unit["root"] / pin["path"], news_unit["manifest"])
    news_unit["config_sha256"] = _json(news_unit["config"], news_unit["policy"])
    with pytest.raises(DataReadinessError, match="false admission"):
        _run(news_unit)


def _repin_news_manifest(state: dict[str, Any]) -> None:
    pin = state["policy"]["publication"]
    pin["sha256"] = _json(state["root"] / pin["path"], state["manifest"])
    state["config_sha256"] = _json(state["config"], state["policy"])


def test_news_reordered_columns_reject_before_numeric_read(news_unit: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    news_unit["manifest"]["months"]["2019-07"]["model_columns"].reverse()
    _repin_news_manifest(news_unit)
    read = Mock(side_effect=AssertionError("numeric read before column order check"))
    monkeypatch.setattr(readiness, "load_canonical_artifact", read)
    with pytest.raises(DataReadinessError, match="ordered model feature contract"):
        _run(news_unit)
    read.assert_not_called()


def test_news_future_feature_clock_rejects_even_with_rehashed_child(news_unit: dict[str, Any]) -> None:
    path = news_unit["news_path"]
    frame = pd.read_parquet(path)
    clock = f"available_at_{NEWS_COLUMNS[0]}"
    frame.loc[0, clock] = frame.loc[0, "decision_time_utc"] + pd.Timedelta(seconds=1)
    frame.to_parquet(path, index=False)
    sidecar_path = manifest_path_for(path)
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["artifact_sha256"] = file_sha256(path)
    child = news_unit["manifest"]["months"]["2019-07"]
    child.update(sha256=file_sha256(path), manifest_sha256=_json(sidecar_path, sidecar))
    _repin_news_manifest(news_unit)
    with pytest.raises(DataReadinessError, match="clock|future"):
        _run(news_unit)
    assert not news_unit["output"].exists()


def test_changed_news_child_after_readiness_rejects_before_loading(news_unit: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    _run(news_unit)
    policy = _policy(news_unit)
    with news_unit["news_path"].open("ab") as stream:
        stream.write(b"synthetic mutation")
    read = Mock(side_effect=AssertionError("numeric read before source hash check"))
    monkeypatch.setattr(loader, "load_canonical_artifact", read)
    with pytest.raises(DataReadinessError, match="evidence changed"):
        loader.load_return_inputs(news_unit["root"], policy)
    read.assert_not_called()


@pytest.mark.parametrize("contract", ["research_contract", "strategy_contract", "temporal_contract"])
def test_news_requires_exact_independently_proven_contract_pins(
    news_unit: dict[str, Any], monkeypatch: pytest.MonkeyPatch, contract: str,
) -> None:
    proven = news_unit["verified"].contract_pins[contract]
    news_unit["verified"].contract_pins[contract] = proven.model_copy(update={"sha256": "f" * 64})
    read = Mock(side_effect=AssertionError("numeric read before contract provenance"))
    monkeypatch.setattr(readiness, "load_canonical_artifact", read)
    with pytest.raises(DataReadinessError, match="independently verified parent settings"):
        _run(news_unit)
    read.assert_not_called()
