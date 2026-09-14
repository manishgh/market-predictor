"""Small synthetic publications exercise actual canonical reads and pinned admission."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256, manifest_path_for
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.research import swing_return_inputs as owner
from market_predictor.research import swing_training_readiness as audit
from market_predictor.swing.contracts.return_training import ReturnTrainingPolicy
from market_predictor.swing.training.return_validation import security_transfer_mask
from tests.test_swing_training_readiness import _json, _repin, _run
from tests.test_swing_training_readiness import publication as publication

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def admission(publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], ReturnTrainingPolicy]:
    names = pd.Series([f"issuer-{i}" for i in range(20)])
    mask = security_transfer_mask(names)
    ids = [names[~mask].iloc[0], names[mask].iloc[0], names[~mask].iloc[0]]
    for profile, path in publication["paths"].items():
        frame = pd.read_parquet(path)
        frame["security_id"] = ids
        frame.to_parquet(path, index=False)
        sidecar_path = manifest_path_for(path)
        sidecar = json.loads(sidecar_path.read_text())
        sidecar["artifact_sha256"] = file_sha256(path)
        child = publication["manifest"]["months"]["2019-07"]["profiles"][profile]
        child["sha256"] = file_sha256(path)
        child["manifest_sha256"] = _json(sidecar_path, sidecar)
    _repin(publication)
    _run(publication)
    payload = json.loads((REPO / "configs/swing_return_training.json").read_text())
    payload["readiness"] = dict(path=publication["output"].relative_to(publication["root"]).as_posix(),
        sha256=file_sha256(publication["output"]))
    payload["readiness_config"] = dict(path=publication["config"].relative_to(publication["root"]).as_posix(),
        sha256=publication["config_sha256"])
    monkeypatch.setattr(owner, "load_temporal_manifest_config", lambda path: SimpleNamespace(
        calendar="XNYS", label_horizon_sessions=10, warmup_sessions=250, initial_fit_end=date(2019, 7, 31)))
    monkeypatch.setattr(owner, "build_temporal_schedule", audit.build_temporal_schedule)
    monkeypatch.setattr(owner, "_guard", lambda policy: None)
    monkeypatch.setattr(owner, "release_process_memory", lambda: None)
    return publication, ReturnTrainingPolicy.model_validate(payload)


def test_admitted_read_preserves_missing_features_and_outcomes(admission: tuple[dict[str, Any], ReturnTrainingPolicy]) -> None:
    state, policy = admission
    result = owner.load_return_inputs(state["root"], policy)
    assert result.features.shape == (3, 120)
    assert result.features.dtypes.eq(np.dtype("float32")).all()
    assert result.features.iloc[2].isna().all()
    assert result.metadata.fixed_horizon_supervision_available.tolist() == [True, True, False]
    assert result.metadata.feature_eligible.tolist() == [True, True, True]
    assert result.metadata[policy.target].iloc[0] < 0
    assert result.metadata[policy.target].isna().iloc[2]
    assert result.metadata.security_holdout.tolist() == [False, True, False]


def test_replacing_both_child_and_sidecar_cannot_bypass_receipt(admission: tuple[dict[str, Any], ReturnTrainingPolicy],
    monkeypatch: pytest.MonkeyPatch) -> None:
    state, policy = admission
    path = state["paths"]["technical_market"]
    frame = pd.read_parquet(path)
    frame.loc[0, "daily_bar_count"] = 999
    frame.to_parquet(path, index=False)
    sidecar = manifest_path_for(path)
    content = json.loads(sidecar.read_text())
    content["artifact_sha256"] = file_sha256(path)
    _json(sidecar, content)
    read = Mock()
    monkeypatch.setattr(owner, "load_canonical_artifact", read)
    with pytest.raises(DataReadinessError, match="evidence changed"):
        owner.load_return_inputs(state["root"], policy)
    read.assert_not_called()


def test_memory_rejection_precedes_numeric_loading(admission: tuple[dict[str, Any], ReturnTrainingPolicy],
    monkeypatch: pytest.MonkeyPatch) -> None:
    state, policy = admission
    monkeypatch.setattr(owner, "_guard", Mock(side_effect=MemoryBudgetError("limit")))
    read = Mock()
    monkeypatch.setattr(owner, "load_canonical_artifact", read)
    with pytest.raises(MemoryBudgetError):
        owner.load_return_inputs(state["root"], policy)
    read.assert_not_called()


def test_out_of_range_month_fails_before_numeric_loading(admission: tuple[dict[str, Any], ReturnTrainingPolicy],
    monkeypatch: pytest.MonkeyPatch) -> None:
    state, policy = admission
    original = owner.pinned_object
    def injected(path: Path, digest: str) -> Any:
        result = original(path, digest)
        if path.name == "_manifest.json":
            result["months"]["2025-07"] = result["months"]["2019-07"]
        return result
    monkeypatch.setattr(owner, "pinned_object", injected)
    read = Mock()
    monkeypatch.setattr(owner, "load_canonical_artifact", read)
    with pytest.raises(DataReadinessError, match="exact initial-fit months"):
        owner.load_return_inputs(state["root"], policy)
    read.assert_not_called()
