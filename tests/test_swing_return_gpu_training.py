"""GPU selection UNITs; mocked estimators never establish actual CUDA training."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import xgboost
from pydantic import ValidationError

from market_predictor.swing.contracts.return_training import ReturnTrainingPolicy
from market_predictor.swing.training.return_estimators import fit_return_regressor

ROOT = Path(__file__).resolve().parents[1]


def policy() -> dict:
    return json.loads((ROOT / "configs/swing_news_return_training.json").read_text())


def test_cpu_policy_serialization_stays_exact() -> None:
    original = policy()
    assert ReturnTrainingPolicy.model_validate(original).model_dump(mode="json") == original
    assert "device" not in ReturnTrainingPolicy.model_validate(original).boosted.model_dump()


def test_cuda_is_explicit_and_preserves_all_other_policy_values() -> None:
    original = policy()
    proposed = policy()
    proposed["boosted"]["device"] = "cuda"
    parsed = ReturnTrainingPolicy.model_validate(proposed).model_dump(mode="json")
    assert parsed["boosted"].pop("device") == "cuda"
    assert parsed == original


@pytest.mark.parametrize("device", ["cpu", "cuda:1", "gpu", 0, True])
def test_unsupported_device_cannot_change_frozen_training_contract(device: object) -> None:
    proposed = policy()
    proposed["boosted"]["device"] = device
    with pytest.raises(ValidationError):
        ReturnTrainingPolicy.model_validate(proposed)


@pytest.mark.parametrize("actual", ["cpu", "cuda:1", "cuda:0"])
def test_requested_cuda_checks_actual_fitted_device(actual: str, monkeypatch: pytest.MonkeyPatch) -> None:
    estimator = MagicMock()
    estimator.get_booster.return_value.save_config.return_value = json.dumps(
        {"learner": {"generic_param": {"device": actual}}})
    constructor = MagicMock(return_value=estimator)
    monkeypatch.setattr(xgboost, "XGBRegressor", constructor)
    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": True})
    parameters = {**policy()["boosted"], "device": "cuda"}
    features = pd.DataFrame({"signal": [1., 2., 3.]})
    args = ("shallow_boosted_return", features, np.array([.1, .2, .3]), np.ones(3), parameters)
    if actual != "cuda:0":
        with pytest.raises(ValueError, match="CPU fallback refused"):
            fit_return_regressor(*args)
    else:
        fitted = fit_return_regressor(*args)
        assert fitted.parameters == parameters
        assert fitted.scaler is None
        constructor.assert_called_once_with(**parameters)
        np.testing.assert_array_equal(estimator.fit.call_args.args[0], [[1., 0.], [2., 0.], [3., 0.]])
        np.testing.assert_array_equal(estimator.fit.call_args.kwargs["sample_weight"], np.ones(3))


def test_non_cuda_build_refuses_before_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    constructor = MagicMock()
    monkeypatch.setattr(xgboost, "XGBRegressor", constructor)
    monkeypatch.setattr(xgboost, "build_info", lambda: {"USE_CUDA": False})
    with pytest.raises(ValueError, match="CUDA-enabled"):
        fit_return_regressor("shallow_boosted_return", pd.DataFrame({"signal": [1., 2.]}),
            np.array([.1, .2]), np.ones(2), {**policy()["boosted"], "device": "cuda"})
    constructor.assert_not_called()
