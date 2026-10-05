"""Synthetic physical publications; metadata admission is an explicit test double."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import file_sha256, manifest_path_for, write_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.research import issuer_reaction_verification as owner
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.issuer_reaction import REACTION_COLUMNS
from market_predictor.swing.contracts.issuer_reaction_publication import (
    ARTIFACT_TYPE,
    PROFILE,
    IssuerReactionPublicationPolicy,
    VerifiedIssuerReactionPublication,
)
from market_predictor.swing.features.issuer_reaction_profile import build_issuer_reaction_profile
from tests.test_swing_issuer_reaction_profile import bundle as bundle

REPO = Path(__file__).resolve().parents[1]


def _json(path: Path, value: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return file_sha256(path)


def _artifact(path: Path, frame: pd.DataFrame, kind: str, request: str) -> dict[str, Any]:
    for candidate in (path, manifest_path_for(path)):
        if candidate.exists():
            candidate.unlink()
    audit = CanonicalAuditReport(checks=(CanonicalAuditCheck(name="synthetic", status="pass", failures=0,
        rows_checked=len(frame), detail="Synthetic verifier fixture; no source admission."),))
    write_canonical_artifact(frame, path, artifact_type=kind, audit=audit,
                             inputs={"request_sha256": request}, production_ready=False)
    return {"sha256": file_sha256(path), "manifest_sha256": file_sha256(manifest_path_for(path))}


@pytest.fixture
def derivative(tmp_path: Path, bundle: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = tmp_path
    for name in owner.IMPLEMENTATION_PATHS:
        target = root / "src/market_predictor" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "src/market_predictor" / name, target)
    monkeypatch.setattr(owner, "__file__", str(root / "src/market_predictor/research/issuer_reaction_verification.py"))
    monkeypatch.setattr(owner, "current_implementation", lambda root: {})
    monkeypatch.setattr(owner, "_guard", lambda: None)
    monkeypatch.setattr(owner, "release_process_memory", lambda: None)
    bundle["coverage"] = bundle["coverage"].assign(coverage_status="unknown", available_at_utc=pd.NaT)
    parent = bundle["baseline"]
    result = build_issuer_reaction_profile(**bundle)
    month = parent.rows.session_date_et.iloc[0].isoformat()[:7]
    directory = root / "data/features/reactions"
    parent_directory = root / "data/features/relationships"
    parent_path = parent_directory / month / "technical_relationships.parquet"
    parent_request = _json(parent_directory / "_request.json", {"synthetic": True})
    parent_child = {"path": f"{month}/technical_relationships.parquet",
                    **_artifact(parent_path, parent.rows, "swing_return_relationships", parent_request)}
    parent_manifest = {"rows": len(parent.rows), "request_sha256": parent_request, "months": {
        month: {"rows": len(parent.rows), "decision_ids_sha256": json_sha256(sorted(parent.rows.decision_id)),
                "profiles": {"technical_relationships": parent_child}}}}
    parent_pin = SourcePin(path="data/features/relationships/_manifest.json",
                           sha256=_json(parent_directory / "_manifest.json", parent_manifest))
    qualification = SourcePin(path="data/research/qualification/_manifest.json", sha256="a" * 64)
    authority = SourcePin(path="data/research/qualification/_authority.json", sha256=bundle["sources"].event_authority_sha256)
    policy = IssuerReactionPublicationPolicy(schema_version="market_predictor.issuer_reaction_publication_config",
        parent_publication=parent_pin,
        parent_saved_row_verification=SourcePin(path="data/reports/relationships.json", sha256="b" * 64),
        qualification_publication=qualification, qualification_authority=authority)
    request = {"policy": policy.model_dump(mode="json"), "profile_sha256": result.audit["profile_sha256"],
        "sources": bundle["sources"].model_dump(mode="json"), "qualification_publication": qualification.model_dump(mode="json"),
        "historical_implementation_files": {"historical/producer.py": "c" * 64}}
    request_pin = _json(directory / "_request.json", request)
    path = directory / month / f"{PROFILE}.parquet"
    child = {"path": f"{month}/{PROFILE}.parquet", "model_columns": list(result.model_columns),
        "availability_columns": dict(result.availability_columns), **_artifact(path, result.rows, ARTIFACT_TYPE, request_pin)}
    months = {month: {"rows": len(parent.rows), "decision_ids_sha256": json_sha256(sorted(parent.rows.decision_id)),
                      "profiles": {PROFILE: child}}}
    manifest = {"rows": len(parent.rows), "months": months, "request_sha256": request_pin}
    publication = SourcePin(path="data/features/reactions/_manifest.json", sha256=_json(directory / "_manifest.json", manifest))
    sources: dict[str, str] = {publication.path: publication.sha256, parent_pin.path: parent_pin.sha256,
        (directory / "_request.json").relative_to(root).as_posix(): request_pin}
    for artifact, descriptor in ((path, child), (parent_path, parent_child)):
        sources[artifact.relative_to(root).as_posix()] = descriptor["sha256"]
        sources[manifest_path_for(artifact).relative_to(root).as_posix()] = descriptor["manifest_sha256"]
    events_path = root / "data/research/qualified_events.parquet"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    bundle["qualified_events"].to_parquet(events_path, index=False)
    event_pin = events_path.relative_to(root).as_posix()
    sources[event_pin] = file_sha256(events_path)
    verified = VerifiedIssuerReactionPublication(request=request, manifest=manifest, source_files=sources,
        parent_manifest=parent_manifest, parent_path=root / parent_pin.path, model_columns=result.model_columns,
        availability_columns=dict(result.availability_columns), months=months)
    inputs = SimpleNamespace(source_files={event_pin: sources[event_pin]}, sources=bundle["sources"])
    monkeypatch.setattr(owner, "verify_issuer_reaction_publication", lambda root, pin: verified)
    monkeypatch.setattr(owner, "load_reaction_inputs", lambda root, policy: inputs)
    monkeypatch.setattr(owner, "load_parent_month", lambda inputs, key: parent.rows.copy())

    def iterator(root: Path, inputs: Any, key: str) -> Any:
        kwargs = {name: value for name, value in bundle.items() if name != "baseline"}
        kwargs["qualified_events"] = pd.read_parquet(events_path)
        yield "one-source-group", parent, kwargs

    # Only partition assembly is doubled; the source event selector and physical
    # reaction math execute through the actual shared pure projection.
    monkeypatch.setattr(owner, "iter_reaction_inputs", iterator)
    monkeypatch.setattr(owner, "assemble_reaction_month", lambda parent, groups: groups[0])
    return dict(root=root, publication=publication, output=root / "data/reports/reaction.json", verified=verified,
        path=path, parent=parent.rows, frame=result.rows, child=child, bundle=bundle, inputs=inputs,
        sources=sources, event_pin=event_pin, events_path=events_path)


def test_real_projection_replay_and_receipt_preserve_parent(derivative: dict[str, Any]) -> None:
    value = owner.verify_issuer_reaction_rows(derivative["root"], derivative["publication"], derivative["output"])
    assert value["rows"] == value["unique_decisions"] == 1
    assert value["original_outcome_values_exact"] and value["additions_source_replayed"]
    assert value["profiles"][PROFILE]["unknown_coverage_rows"] == 1
    assert value["profiles"][PROFILE]["selected_event_rows"] == 1
    assert all(value[name] is False for name in (
        "training_eligible", "serving_eligible", "promotion_eligible", "baseline_numerical_replayed"))
    assert "historical/producer.py" not in value["source_files"]
    assert value["historical_implementation_files"] == {"historical/producer.py": "c" * 64}
    assert value["report_sha256"] == file_sha256(derivative["output"])
    assert owner.validate_issuer_reaction_receipt(derivative["root"], derivative["publication"], value) is derivative["verified"]


@pytest.mark.parametrize("poison", [
    "parent_value", "target", "eligibility", "parent_clock", "dtype", "future_clock", "reason", "column_order",
])
def test_saved_parent_and_new_column_poison_rejected(derivative: dict[str, Any], poison: str) -> None:
    frame = derivative["frame"].copy()
    name = REACTION_COLUMNS[0]
    if poison == "parent_value":
        frame.loc[0, derivative["verified"].model_columns[0]] += 1
    elif poison == "target":
        frame.loc[0, "spy_fixed_horizon_excess_return"] = 0.0
    elif poison == "eligibility":
        frame.loc[0, "feature_eligible"] = True
    elif poison == "parent_clock":
        frame.loc[0, "parent_clock"] -= pd.Timedelta(seconds=1)
    elif poison == "dtype":
        frame[name] = frame[name].astype("float64")
    elif poison == "future_clock":
        frame.loc[0, f"available_at_{name}"] = frame.loc[0, "decision_time_utc"] + pd.Timedelta(seconds=1)
    elif poison == "reason":
        frame.loc[0, f"missing_reason_{name}"] = "invented_missing"
    else:
        frame = frame.loc[:, list(reversed(frame.columns))]
    with pytest.raises(DataReadinessError):
        owner._check_rows(derivative["parent"], frame, derivative["verified"])


def _replace_child(derivative: dict[str, Any], frame: pd.DataFrame) -> None:
    descriptor = _artifact(derivative["path"], frame, ARTIFACT_TYPE, derivative["verified"].manifest["request_sha256"])
    derivative["child"].update(descriptor)
    for path, key in ((derivative["path"], "sha256"), (manifest_path_for(derivative["path"]), "manifest_sha256")):
        derivative["sources"][path.relative_to(derivative["root"]).as_posix()] = descriptor[key]


@pytest.mark.parametrize("field,value", [(REACTION_COLUMNS[0], np.float32(0.123)), ("selected_event_id", "wrong-event"),
                                         ("reaction_coverage_status", "known")])
def test_hash_valid_saved_addition_or_selection_still_requires_source_replay(
    derivative: dict[str, Any], field: str, value: Any,
) -> None:
    changed = derivative["frame"].copy()
    changed.loc[0, field] = value
    _replace_child(derivative, changed)
    with pytest.raises(DataReadinessError, match="source replay"):
        owner.verify_issuer_reaction_rows(derivative["root"], derivative["publication"], derivative["output"])
    assert not derivative["output"].exists()


def test_known_rejected_revision_cannot_match_saved_old_event(derivative: dict[str, Any]) -> None:
    events = pd.read_parquet(derivative["events_path"])
    revision = events.copy()
    revision["event_version_sha256"] = "d" * 64
    revision["event_available_at_utc"] += pd.Timedelta(minutes=1)
    revision["qualification_status"], revision["event_family"] = "rejected", None
    pd.concat([events, revision], ignore_index=True).to_parquet(derivative["events_path"], index=False)
    digest = file_sha256(derivative["events_path"])
    derivative["sources"][derivative["event_pin"]] = digest
    derivative["inputs"].source_files[derivative["event_pin"]] = digest
    with pytest.raises(DataReadinessError, match="source replay"):
        owner.verify_issuer_reaction_rows(derivative["root"], derivative["publication"], derivative["output"])


@pytest.mark.parametrize("field,value", [("event_authority_sha256", "f" * 64), ("qualification_source_replayed", False),
                                         ("original_columns_exact", 1), ("training_eligible", True),
                                         ("rows", True), ("historical_implementation_files", {})])
def test_receipt_cannot_replace_authority_or_replay_claims(derivative: dict[str, Any], field: str, value: Any) -> None:
    report = owner.verify_issuer_reaction_rows(derivative["root"], derivative["publication"], derivative["output"])
    report[field] = value
    with pytest.raises(DataReadinessError):
        owner.validate_issuer_reaction_receipt(derivative["root"], derivative["publication"], report)


def test_source_mutation_prevents_receipt(derivative: dict[str, Any]) -> None:
    with derivative["events_path"].open("ab") as handle:
        handle.write(b"tamper")
    with pytest.raises(DataReadinessError, match="source changed"):
        owner.verify_issuer_reaction_rows(derivative["root"], derivative["publication"], derivative["output"])
    assert not derivative["output"].exists()


def test_receipt_immutable_and_lease_precedes_input_loading(derivative: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(owner, "verify_issuer_reaction_publication", lambda *args: pytest.fail("read before lease"))
    with heavy_job_lease("synthetic-owner", runtime_dir=derivative["root"] / "data/runtime"):
        with pytest.raises(HeavyJobBusyError):
            owner.verify_issuer_reaction_rows(derivative["root"], derivative["publication"], derivative["output"])
    derivative["output"].parent.mkdir(parents=True, exist_ok=True)
    derivative["output"].write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        owner.verify_issuer_reaction_rows(derivative["root"], derivative["publication"], derivative["output"])
    assert derivative["output"].read_text() == "original"
