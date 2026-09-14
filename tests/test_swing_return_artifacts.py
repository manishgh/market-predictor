from __future__ import annotations

import io
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
