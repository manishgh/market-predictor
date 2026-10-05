"""Synthetic unit evidence only; never retained-run or release evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.saved_evaluation_configuration import SavedEvaluationConfigurationEvidence
from market_predictor.swing.datasets import saved_evaluation_configuration as owner


def _write(root: Path, name: str, value: Any) -> SourcePin:
    """Unit-only fixture creation and deliberate resealing of synthetic evidence."""
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value if isinstance(value, str) else json.dumps(value, sort_keys=True), encoding="utf-8")
    return SourcePin(path=name, sha256=file_sha256(path))


def _dump(pin: SourcePin) -> dict[str, Any]:
    return pin.model_dump(mode="json")


@pytest.fixture
def case(tmp_path: Path) -> dict[str, Any]:
    recovered = {}
    for name, changes in (("strategy", owner.STRATEGY_RENAMES), ("temporal", owner.TEMPORAL_RENAMES)):
        old = "\n".join(f'{key} = "{values[0]}"' for key, values in changes.items())
        new = "\n".join(f'{key} = "{values[1]}"' for key, values in changes.items())
        content = '\nlimit = 10\nweights = [1, 2]\nstart = 2019-07-09\n[other]\nactive = true\n'
        original = _write(tmp_path, f"archive/{name}.toml", old + content)
        current = _write(tmp_path, f"configs/{name}.toml", new + content)
        raw = (tmp_path / original.path).read_bytes()
        recovered[name] = {"original_logical": {"path": current.path, "sha256": original.sha256},
            "original_artifact": _dump(original), "current": _dump(current), "git_commit": "a" * 40,
            "git_blob": hashlib.sha1(f"blob {len(raw)}\0".encode() + raw, usedforsecurity=False).hexdigest()}
    placeholder = {"path": "unused.json", "sha256": "b" * 64}
    readiness_policy = {"schema_version": "market_predictor.swing_training_readiness",
        "scope": "initial_fit_fixed_horizon_diagnostics", "maximum_system_used_percent": 90.0,
        "publication": placeholder, "saved_row_verification": placeholder, "research_contract": placeholder,
        "strategy_contract": recovered["strategy"]["original_logical"],
        "temporal_contract": recovered["temporal"]["original_logical"]}
    ready_config = _write(tmp_path, "readiness_config.json", readiness_policy)
    source_files = {ready_config.path: ready_config.sha256,
        **{item["original_logical"]["path"]: item["original_logical"]["sha256"] for item in recovered.values()}}
    ready = _write(tmp_path, "readiness.json", {"schema": "market_predictor.swing_training_readiness",
        "status": "diagnostic_complete", "scope": readiness_policy["scope"], "publication_sha256": "b" * 64,
        "training_eligible": False, "promotion_eligible": False, "source_files": source_files})
    # Only fixed policy settings are reused; all evidence files are synthetic.
    policy = json.loads((Path(__file__).resolve().parents[1] / "configs/swing_return_training.json").read_text())
    policy.update(readiness=_dump(ready), readiness_config=_dump(ready_config))
    request = {"schema": "market_predictor.swing_return_training_request", "policy": policy,
        "source_files": {**source_files, ready.path: ready.sha256}, **dict.fromkeys(owner.FLAGS, False)}
    runs = []
    for index in range(2):
        request_pin = _write(tmp_path, f"run-{index}/_request.json", request | {"unit_fixture": index})
        runs.append(_dump(_write(tmp_path, f"run-{index}/_manifest.json", {
            "schema": "market_predictor.swing_return_training", "status": "complete_research_only",
            "request_sha256": request_pin.sha256, **dict.fromkeys(owner.FLAGS, False)})))
    original = recovered["strategy"]["original_logical"]
    report = _write(tmp_path, "report.json", {"schema": "market_predictor.saved_configuration_differences",
        "rewritten_historical_pins": False, "source_equivalence_proven": False, "training_eligible": False,
        "configurations": {original["path"]: {"historical_sha256": original["sha256"],
            "current_sha256": recovered["strategy"]["current"]["sha256"], "historical_git_revision": "a" * 7,
            "changed_fields": [{"field": key, "historical": values[0], "current": values[1]}
                for key, values in owner.STRATEGY_RENAMES.items()]}}})
    evidence = {"schema_version": "market_predictor.saved_evaluation_configuration",
        "scope": "saved_initial_fit_evaluation_only", "runs": runs, **recovered, "source_configuration_report": _dump(report)}
    return {"root": tmp_path, "evidence": evidence}


def _arguments(case: dict[str, Any], index: int = 0) -> dict[str, Any]:
    evidence = case["evidence"]
    return {"root": case["root"], "evidence_pin": _write(case["root"], "proof.json", evidence),
        "run_manifest": SourcePin.model_validate(evidence["runs"][index]),
        "strategy_pin": SourcePin.model_validate(evidence["strategy"]["current"]),
        "temporal_pin": SourcePin.model_validate(evidence["temporal"]["current"]),
        "original_strategy_pin": SourcePin.model_validate(evidence["strategy"]["original_logical"]),
        "original_temporal_pin": SourcePin.model_validate(evidence["temporal"]["original_logical"]), "pins": {}}


@pytest.mark.parametrize("index", [0, 1])
def test_exact_three_renames_bind_both_runs_without_rewriting_originals(case: Any, index: int) -> None:
    args = _arguments(case, index)
    before = {path: path.read_bytes() for path in case["root"].rglob("*") if path.is_file()}
    result = owner.verify_saved_evaluation_configuration(**args)
    assert result.strategy_path == case["root"] / "configs/strategy.toml"
    assert result.temporal_path == case["root"] / "configs/temporal.toml"
    assert sum(map(len, result.provenance["changed_fields"].values())) == 3
    assert result.provenance["new_fit_authorized"] is False
    assert result.provenance["git_commit_membership_verified"] is False
    assert all(path.read_bytes() == data for path, data in before.items())
    for name in ("strategy", "temporal"):
        item = case["evidence"][name]
        assert args["pins"][item["current"]["path"]] == item["current"]["sha256"]
        assert args["pins"][item["original_artifact"]["path"]] == item["original_logical"]["sha256"]


@pytest.mark.parametrize("old,new", [("limit = 10", "limit = 11"), ("limit = 10", "limit = 10.0"),
    ("limit = 10", "limit = true"), ("[1, 2]", "[2, 1]"), ("[1, 2]", "[1, 2, 3]"),
    ("2019-07-09", "2019-07-10"), ("active = true", "active = false"),
    ("active = true", "replacement = true"), ("temporal_manifest\"", "temporal_manifest.v2\"")])
def test_resealed_current_changes_beyond_exact_renames_reject(case: Any, old: str, new: str) -> None:
    item = case["evidence"]["temporal"]
    path = case["root"] / item["current"]["path"]
    item["current"] = _dump(_write(case["root"], item["current"]["path"], path.read_text().replace(old, new)))
    with pytest.raises(DataReadinessError, match="configuration"):
        owner.verify_saved_evaluation_configuration(**_arguments(case))


@pytest.mark.parametrize("target", ["archive/strategy.toml", "configs/temporal.toml", "run-0/_request.json",
    "readiness.json", "readiness_config.json", "report.json", "proof.json"])
def test_changed_pinned_bytes_reject(case: Any, target: str) -> None:
    args = _arguments(case)
    path = case["root"] / target
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(DataReadinessError):
        owner.verify_saved_evaluation_configuration(**args)


def test_caller_and_evidence_original_pin_cannot_replace_saved_readiness_authority(case: Any) -> None:
    item = case["evidence"]["temporal"]
    raw = (case["root"] / item["original_artifact"]["path"]).read_text() + "\n# resealed substitute\n"
    item["original_artifact"] = _dump(_write(case["root"], item["original_artifact"]["path"], raw))
    item["original_logical"]["sha256"] = item["original_artifact"]["sha256"]
    with pytest.raises(DataReadinessError, match="original saved readiness"):
        owner.verify_saved_evaluation_configuration(**_arguments(case))


@pytest.mark.parametrize("fault", ["unlisted_run", "git_blob", "conflicting_pins", "report_claim"])
def test_independent_scope_blob_and_report_bindings(case: Any, fault: str) -> None:
    if fault == "git_blob":
        case["evidence"]["temporal"]["git_blob"] = "0" * 40
    if fault == "report_claim":
        pin = case["evidence"]["source_configuration_report"]
        report = json.loads((case["root"] / pin["path"]).read_text())
        report["source_equivalence_proven"] = True
        case["evidence"]["source_configuration_report"] = _dump(_write(case["root"], pin["path"], report))
    args = _arguments(case)
    if fault == "unlisted_run":
        args["run_manifest"] = SourcePin(path="other/_manifest.json", sha256="f" * 64)
    if fault == "conflicting_pins":
        args["pins"]["configs/strategy.toml"] = "f" * 64
    with pytest.raises(DataReadinessError):
        owner.verify_saved_evaluation_configuration(**args)


def test_evidence_forbids_extra_fields_and_duplicate_runs(case: Any) -> None:
    evidence = case["evidence"]
    with pytest.raises(ValidationError):
        SavedEvaluationConfigurationEvidence.model_validate_json(json.dumps(evidence | {"waive_hashes": True}))
    with pytest.raises(ValidationError):
        SavedEvaluationConfigurationEvidence.model_validate_json(json.dumps(evidence | {"runs": [evidence["runs"][0]] * 2}))
