"""Isolated row-verifier tests; metadata admission is explicitly a test double."""
from __future__ import annotations

import json
import shutil
from typing import Any

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import file_sha256, manifest_path_for, write_canonical_artifact
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_COLUMNS
from market_predictor.swing.contracts.return_relationship_publication import VerifiedReturnRelationshipPublication
from market_predictor.swing.contracts.return_training import ReturnTrainingPolicy
from market_predictor.swing.datasets import return_relationship_verification as owner
from tests.test_swing_training_readiness import REPO, _json
from tests.test_swing_training_readiness import publication as publication


@pytest.fixture
def derivative(publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = publication["root"]
    for name in owner.IMPLEMENTATION_PATHS:
        target = root / "src/market_predictor" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "src/market_predictor" / name, target)
    monkeypatch.setattr(owner, "__file__", str(root / "src/market_predictor/swing/datasets/return_relationship_verification.py"))
    monkeypatch.setattr(owner, "_guard", lambda: None)
    monkeypatch.setattr(owner, "release_process_memory", lambda: None)
    monkeypatch.setattr(owner, "_replay_additions", lambda root, path, verified: None)
    parent_path = publication["paths"]["technical_market"]
    parent = pd.read_parquet(parent_path)
    parent["feature_profile"] = "technical_market"
    parent["auxiliary_value"] = [0.0, np.nan, -1.0]
    parent["available_at_auxiliary_value"] = parent.decision_time_utc
    audit = CanonicalAuditReport(checks=(CanonicalAuditCheck(name="fixture", status="pass", failures=0,
        rows_checked=3, detail="Synthetic test only."),))
    parent_path.unlink()
    manifest_path_for(parent_path).unlink()
    write_canonical_artifact(parent, parent_path, artifact_type="swing_research_join", audit=audit,
        inputs={"request_sha256": publication["manifest"]["request_sha256"]}, production_ready=False)
    parent_child = publication["manifest"]["months"]["2019-07"]["profiles"]["technical_market"]
    parent_child.update(sha256=file_sha256(parent_path), manifest_sha256=file_sha256(manifest_path_for(parent_path)))
    parent_child["availability_columns"]["auxiliary_value"] = "available_at_auxiliary_value"
    parent_manifest_path = parent_path.parent.parent / "_manifest.json"
    _json(parent_manifest_path, publication["manifest"])
    frame = parent.copy()
    frame["feature_profile"] = "technical_relationships"
    for name in RETURN_RELATIONSHIP_COLUMNS:
        frame[name] = np.array([0.25, np.nan, -0.5], dtype=np.float32)
    for name in RETURN_RELATIONSHIP_COLUMNS:
        frame[f"available_at_{name}"] = frame.decision_time_utc.where(frame[name].notna())
    for name in RETURN_RELATIONSHIP_COLUMNS:
        frame[f"missing_reason_{name}"] = ["", "missing_required_session_or_value", ""]
    directory = root / "data/features/relationships"
    request = dict(schema="market_predictor.return_relationship_request", profile_sha256="f" * 64)
    request_hash = _json(directory / "_request.json", request)
    path = directory / "2019-07/technical_relationships.parquet"
    write_canonical_artifact(frame, path, artifact_type="swing_return_relationships", audit=audit,
        inputs={"request_sha256": request_hash}, production_ready=False)
    names = (*parent_child["model_columns"], *RETURN_RELATIONSHIP_COLUMNS)
    clocks = {**parent_child["availability_columns"], **{name: f"available_at_{name}" for name in RETURN_RELATIONSHIP_COLUMNS}}
    child = dict(path="2019-07/technical_relationships.parquet", sha256=file_sha256(path),
        manifest_sha256=file_sha256(manifest_path_for(path)), model_columns=list(names), availability_columns=clocks,
        audit=dict(training_eligible=False, promotion_eligible=False, serving_eligible=False))
    months = {"2019-07": dict(rows=3,
        decision_ids_sha256=publication["manifest"]["months"]["2019-07"]["decision_ids_sha256"],
        profiles={"technical_relationships": child})}
    manifest = dict(schema="market_predictor.return_relationship_publication", request_sha256=request_hash,
        rows=3, months=months, status="complete_research_only", training_eligible=False, promotion_eligible=False, serving_eligible=False)
    pin = SourcePin(path="data/features/relationships/_manifest.json", sha256=_json(directory / "_manifest.json", manifest))
    verified = VerifiedReturnRelationshipPublication(request=request, manifest=manifest, source_files={},
        parent_manifest=publication["manifest"], parent_path=parent_manifest_path, model_columns=names,
        availability_columns=clocks, months=months)
    monkeypatch.setattr(owner, "verify_return_relationship_publication", lambda root, publication: verified)
    return dict(root=root, publication=pin, output=root / "data/reports/relationships.json", verified=verified,
        path=path, parent=parent, frame=frame, child=child)


def test_independent_rows_preserve_all_parent_columns(derivative: dict[str, Any]) -> None:
    result = owner.verify_return_relationship_rows(derivative["root"], derivative["publication"], derivative["output"])
    assert result["status"] == "passed" and result["rows"] == 3
    assert result["profiles"]["technical_relationships"]["complete_model_rows"] == 1
    assert result["original_columns_exact"] is True
    assert result["report_sha256"] == file_sha256(derivative["output"])
    assert result["training_eligible"] is result["promotion_eligible"] is result["serving_eligible"] is False
    owner.validate_return_relationship_receipt(derivative["root"], derivative["publication"], result)


@pytest.mark.parametrize("poison", ["auxiliary_value", "available_at_auxiliary_value", "feature_eligible",
    "fixed_horizon_net_return", "row_order", "column_order", "future_clock", "missing_clock", "missing_reason", "dtype"])
def test_row_poison_rejected(derivative: dict[str, Any], poison: str) -> None:
    frame = derivative["frame"].copy()
    name = RETURN_RELATIONSHIP_COLUMNS[0]
    if poison == "row_order":
        frame = frame.iloc[::-1].reset_index(drop=True)
    elif poison == "column_order":
        frame = frame.loc[:, list(reversed(frame.columns))]
    elif poison == "future_clock":
        frame.loc[0, f"available_at_{name}"] += pd.Timedelta(seconds=1)
    elif poison == "missing_clock":
        frame.loc[0, f"available_at_{name}"] = pd.NaT
    elif poison == "missing_reason":
        frame.loc[1, f"missing_reason_{name}"] = ""
    elif poison == "dtype":
        frame[name] = frame[name].astype("float64")
    elif poison == "available_at_auxiliary_value":
        frame.loc[0, poison] -= pd.Timedelta(seconds=1)
    elif poison == "feature_eligible":
        frame.loc[0, poison] = False
    else:
        frame.loc[0, poison] += 1.0
    with pytest.raises(DataReadinessError):
        owner._check_rows(derivative["parent"], frame, derivative["verified"])


def test_changed_child_bytes_rejected_without_receipt(derivative: dict[str, Any]) -> None:
    with derivative["path"].open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(DataReadinessError):
        owner.verify_return_relationship_rows(derivative["root"], derivative["publication"], derivative["output"])
    assert not derivative["output"].exists()


def test_busy_lease_rejects_before_metadata(derivative: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: Any) -> Any:
        raise AssertionError("must not load metadata")
    monkeypatch.setattr(owner, "verify_return_relationship_publication", fail)
    with heavy_job_lease("other-job", runtime_dir=derivative["root"] / "data/runtime"):
        with pytest.raises(HeavyJobBusyError):
            owner.verify_return_relationship_rows(derivative["root"], derivative["publication"], derivative["output"])


def test_current_verifier_change_invalidates_receipt(derivative: dict[str, Any]) -> None:
    receipt = owner.verify_return_relationship_rows(derivative["root"], derivative["publication"], derivative["output"])
    code = derivative["root"] / "src/market_predictor/swing/datasets/return_relationship_verification.py"
    with code.open("ab") as handle:
        handle.write(b"\n# changed\n")
    with pytest.raises(DataReadinessError):
        owner.validate_return_relationship_receipt(derivative["root"], derivative["publication"], receipt)


def test_configured_shared_runtime_lease_is_honored(derivative: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    runtime = derivative["root"] / "data/shared-runtime"
    monkeypatch.setenv("MARKET_PREDICTOR_RUNTIME_DIR", str(runtime))
    with heavy_job_lease("publisher", runtime_dir=runtime):
        with pytest.raises(HeavyJobBusyError):
            owner.verify_return_relationship_rows(derivative["root"], derivative["publication"], derivative["output"])


def test_incomplete_month_inventory_rejected(derivative: dict[str, Any]) -> None:
    derivative["verified"].months.clear()
    with pytest.raises(DataReadinessError, match="month inventory"):
        owner.verify_return_relationship_rows(derivative["root"], derivative["publication"], derivative["output"])
    assert not derivative["output"].exists()


def test_existing_report_is_immutable(derivative: dict[str, Any]) -> None:
    derivative["output"].write_text("protected", encoding="ascii")
    with pytest.raises(FileExistsError):
        owner.verify_return_relationship_rows(derivative["root"], derivative["publication"], derivative["output"])
    assert derivative["output"].read_text() == "protected"


def test_memory_guard_precedes_metadata(derivative: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse() -> None:
        raise MemoryBudgetError("test pressure")
    monkeypatch.setattr(owner, "_guard", refuse)
    with pytest.raises(MemoryBudgetError):
        owner.verify_return_relationship_rows(derivative["root"], derivative["publication"], derivative["output"])
    assert not derivative["output"].exists()


@pytest.mark.parametrize("field", ["scope", "original_columns_exact", "source_files", "model_columns"])
def test_rehashed_receipt_contract_poison_rejected(derivative: dict[str, Any], field: str) -> None:
    receipt = owner.verify_return_relationship_rows(derivative["root"], derivative["publication"], derivative["output"])
    receipt[field] = {} if field == "source_files" else None
    with pytest.raises(DataReadinessError):
        owner.validate_return_relationship_receipt(derivative["root"], derivative["publication"], receipt)


@pytest.mark.parametrize(("feature", "published", "valid"), [
    ("existing_technical", "technical_market", True), ("technical_relationships", "technical_relationships", True),
    ("existing_technical", "technical_relationships", False), ("technical_relationships", "technical_market", False),
    ("arbitrary", "technical_market", False), ("technical_relationships_issuer_reaction", "technical_relationships", False),
])
def test_only_coherent_approved_training_profiles(feature: str, published: str, valid: bool) -> None:
    payload = json.loads((REPO / "configs/swing_return_training.json").read_text())
    payload.update(feature_profile=feature, published_profile=published)
    if valid:
        assert ReturnTrainingPolicy.model_validate(payload).feature_profile == feature
    else:
        with pytest.raises(ValidationError):
            ReturnTrainingPolicy.model_validate(payload)
