from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.swing.training import return_artifacts as owner
from market_predictor.swing.training.return_estimators import fit_return_regressor


@pytest.fixture
def unit(tmp_path: Path) -> tuple[Path, str]:
    parameters = {"alpha": 1., "fit_intercept": True, "solver": "lsqr", "tol": 1e-6, "max_iter": 10000}
    model = fit_return_regressor("regularized_linear_return", pd.DataFrame({"x": [1., 2., 3.]}),
        np.array([.01, .02, .03]), np.ones(3), parameters)
    directory = tmp_path / "model"
    owner.publish_unit(directory, request_sha256="a" * 64,
        unit={"scope": "final_refit", "family": model.family, "feature_names": ["x"], "parameters": parameters},
        model=model, predictions=None, metrics={}, guard=lambda: None)
    return directory, file_sha256(directory / "_manifest.json")


def test_manifest_replacement_between_reads_fails(unit: tuple[Path, str], monkeypatch: pytest.MonkeyPatch) -> None:
    directory, digest = unit
    original = owner.verify_unit
    def replace(*args: Any, **kwargs: Any) -> Any:
        manifest = directory / "_manifest.json"
        manifest.write_bytes(manifest.read_bytes() + b" ")
        return original(*args, **kwargs)
    monkeypatch.setattr(owner, "verify_unit", replace)
    deserialize = Mock()
    monkeypatch.setattr(owner.joblib, "load", deserialize)
    with pytest.raises(DataReadinessError, match="pinned bytes"):
        owner.load_return_model(directory, manifest_sha256=digest)
    deserialize.assert_not_called()


def test_deserialization_uses_exact_verified_bytes(unit: tuple[Path, str], monkeypatch: pytest.MonkeyPatch) -> None:
    directory, digest = unit
    original = owner.joblib.load
    def replace_after_verification(content: io.BytesIO) -> Any:
        assert isinstance(content, io.BytesIO)
        (directory / "model.joblib").write_bytes(b"changed after reading")
        return original(content)
    monkeypatch.setattr(owner.joblib, "load", replace_after_verification)
    result = owner.load_return_model(directory, manifest_sha256=digest)
    assert result.family == "regularized_linear_return"


def test_publication_memory_failure_leaves_no_successful_unit(unit: tuple[Path, str]) -> None:
    directory, digest = unit
    model = owner.load_return_model(directory, manifest_sha256=digest)
    output = directory.parent / "rejected"
    with pytest.raises(MemoryBudgetError):
        owner.publish_unit(output, request_sha256="b" * 64,
            unit={"scope": "final_refit", "family": model.family}, model=model,
            predictions=None, metrics={}, guard=Mock(side_effect=MemoryBudgetError("peak")))
    assert not output.exists()


@pytest.fixture
def completed_run(tmp_path: Path) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    request = {"schema": "market_predictor.swing_return_training_request",
        "policy": {"target": owner.TARGET, "scope": "initial_fit_research_only"},
        "folds": [{"number": number} for number in range(1, 5)], **dict.fromkeys(owner.FLAGS, False)}
    units = {f"{family}/{scope}/fold-{fold}": {} for family in owner.FAMILIES
             for scope in owner.SCOPES for fold in range(1, 5)}
    units.update({f"{family}/final_refit": {} for family in owner.FAMILIES})
    manifest = {"schema": "market_predictor.swing_return_training", "status": "complete_research_only",
        "units": units, **dict.fromkeys(owner.FLAGS, False)}
    return tmp_path, manifest, request


def _write_completed_run(case: tuple[Path, dict[str, Any], dict[str, Any]]) -> tuple[Path, str]:
    directory, manifest, request = case
    (directory / "_request.json").write_text(json.dumps(request), encoding="utf-8")
    manifest["request_sha256"] = file_sha256(directory / "_request.json")
    (directory / "_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return directory, file_sha256(directory / "_manifest.json")


def test_completed_run_metadata_does_not_load_estimators(
    completed_run: tuple[Path, dict[str, Any], dict[str, Any]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory, digest = _write_completed_run(completed_run)
    monkeypatch.setattr(owner.joblib, "load", Mock(side_effect=AssertionError("metadata verification must not load a model")))
    manifest, request = owner.verify_completed_return_run(directory, manifest_sha256=digest)
    assert (manifest, request) == completed_run[1:]


@pytest.mark.parametrize("change", ["scope", "target", "request_admission", "manifest_admission", "status", "missing_fit", "folds"])
def test_completed_run_rejects_pinned_wrong_scope_or_incomplete_evidence(
    completed_run: tuple[Path, dict[str, Any], dict[str, Any]], change: str,
) -> None:
    _, manifest, request = completed_run
    if change == "scope":
        request["policy"]["scope"] = "another_research_experiment"
    elif change == "target":
        request["policy"]["target"] = "another_return"
    elif change == "request_admission":
        request["serving_eligible"] = True
    elif change == "manifest_admission":
        manifest["promotion_eligible"] = True
    elif change == "status":
        manifest["status"] = "incomplete"
    elif change == "missing_fit":
        manifest["units"].pop(next(iter(manifest["units"])))
    else:
        request["folds"].reverse()
    directory, digest = _write_completed_run(completed_run)
    with pytest.raises(DataReadinessError):
        owner.verify_completed_return_run(directory, manifest_sha256=digest)


def test_completed_run_preserves_caller_metadata_bound_and_hash_pins(
    completed_run: tuple[Path, dict[str, Any], dict[str, Any]],
) -> None:
    directory, digest = _write_completed_run(completed_run)
    with pytest.raises(DataReadinessError, match="bounded reader"):
        owner.verify_completed_return_run(directory, manifest_sha256=digest, maximum_metadata_bytes=10)
    with pytest.raises(DataReadinessError, match="pinned bytes"):
        owner.verify_completed_return_run(directory, manifest_sha256="0" * 64)
    request = directory / "_request.json"
    request.write_bytes(request.read_bytes() + b" ")
    with pytest.raises(DataReadinessError, match="pinned bytes"):
        owner.verify_completed_return_run(directory, manifest_sha256=digest)
