"""Small real-estimator tests for train-only return preprocessing."""
from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBRegressor

from market_predictor.swing.training import return_estimators as owner

RIDGE: dict[str, Any] = {
    "alpha": 1.0, "fit_intercept": True, "solver": "lsqr", "tol": 1e-6, "max_iter": 10000,
}
BOOSTED: dict[str, Any] = {
    "objective": "reg:squarederror", "tree_method": "hist", "max_depth": 3,
    "n_estimators": 4, "learning_rate": 0.05, "reg_lambda": 10.0, "reg_alpha": 0.0,
    "min_child_weight": 1.0, "subsample": 1.0, "colsample_bytree": 1.0,
    "n_jobs": 1, "random_state": 42,
}


def training() -> tuple[pd.DataFrame, Any, Any]:
    return (
        pd.DataFrame({"signal": [1.0, np.nan, 5.0, 9.0], "empty": [np.nan] * 4, "complete": [2.0, 4.0, 8.0, 16.0]}),
        np.array([-4.0, 2.0, 6.0, 10.0]),
        np.array([1.0, 1.0, 2.0, 4.0]),
    )


def encoded_training() -> Any:
    return np.array([
        [1, 0, 2, 0, 1, 0], [5, 0, 4, 1, 1, 0],
        [5, 0, 8, 0, 1, 0], [9, 0, 16, 0, 1, 0],
    ], dtype=np.float32)


@pytest.mark.parametrize(("family", "parameters"), [("regularized_linear_return", RIDGE), ("shallow_boosted_return", BOOSTED)])
def test_real_fit_deterministic_and_inputs_unchanged(family: str, parameters: dict[str, Any]) -> None:
    features, target, weights = training()
    features.index = pd.Index([8, 3, 3, 1], name="row")
    original = features.copy(deep=True)
    target_before, weights_before = target.copy(), weights.copy()
    supplied = dict(parameters)
    with patch.object(owner, "threadpool_limits", wraps=threadpool_limits) as limit:
        first = owner.fit_return_regressor(family, features, target, weights, supplied)
        first_prediction = owner.predict_return(first, features)
        assert limit.call_count == 2
        assert all(call.kwargs == {"limits": 1} for call in limit.call_args_list)
    second = owner.fit_return_regressor(family, features, target, weights, supplied)
    assert_array_equal(first_prediction, owner.predict_return(second, features))
    assert_array_equal(first.medians, [5, 0, 6])
    assert first.feature_names == ("signal", "empty", "complete")
    assert first.estimator.n_features_in_ == 6
    assert first.family == family
    assert first.parameters == parameters
    supplied["n_estimators"] = 999
    assert first.parameters == parameters
    pd.testing.assert_frame_equal(features, original)
    assert_array_equal(target, target_before)
    assert_array_equal(weights, weights_before)
    if family == "regularized_linear_return":
        assert isinstance(first.estimator, Ridge)
        assert first.estimator.solver == "lsqr"
        assert (first_prediction > 1).any()  # Returns are not probabilities.
    else:
        assert isinstance(first.estimator, XGBRegressor)
        assert first.scaler is None
        assert first.estimator.get_params()["n_jobs"] == 1


def test_ridge_matches_real_weighted_scaler_and_solver() -> None:
    features, target, weights = training()
    model = owner.fit_return_regressor("regularized_linear_return", features, target, weights, RIDGE)
    expected = encoded_training()
    with threadpool_limits(limits=1):
        scaler = StandardScaler().fit(expected, sample_weight=weights)
        scaled = scaler.transform(expected)
        ridge = Ridge(**RIDGE).fit(scaled, target, sample_weight=weights)
        predictions = ridge.predict(scaled)
    assert_array_equal(model.medians, [5, 0, 6])  # Medians are deliberately unweighted.
    assert_allclose(model.scaler.mean_, np.average(expected, axis=0, weights=weights))
    assert_allclose(model.scaler.var_, scaler.var_)
    assert not np.allclose(model.scaler.mean_, expected.mean(axis=0))
    assert_allclose(model.estimator.coef_, ridge.coef_, rtol=1e-6, atol=1e-6)
    assert_allclose(owner.predict_return(model, features), predictions, rtol=1e-6, atol=1e-6)


def test_booster_matches_real_weighted_fit() -> None:
    features, target, weights = training()
    model = owner.fit_return_regressor("shallow_boosted_return", features, target, weights, BOOSTED)
    with threadpool_limits(limits=1):
        reference = XGBRegressor(**BOOSTED).fit(encoded_training(), target, sample_weight=weights)
        expected = reference.predict(encoded_training())
    assert_array_equal(owner.predict_return(model, features), expected)


@pytest.mark.parametrize(("family", "parameters"), [("regularized_linear_return", RIDGE), ("shallow_boosted_return", BOOSTED)])
def test_heldout_poison_cannot_change_fit(family: str, parameters: dict[str, Any]) -> None:
    features, target, weights = training()
    model = owner.fit_return_regressor(family, features, target, weights, parameters)
    before = owner.predict_return(model, features)
    medians_before = model.medians.copy()
    mean_before = None if model.scaler is None else model.scaler.mean_.copy()
    variance_before = None if model.scaler is None else model.scaler.var_.copy()
    heldout = pd.DataFrame({"signal": [1e12, np.nan], "empty": [-1e12, 7.0], "complete": [np.nan, -1e12]})
    original = heldout.copy(deep=True)
    assert np.isfinite(owner.predict_return(model, heldout)).all()
    pd.testing.assert_frame_equal(heldout, original)
    assert_array_equal(model.medians, medians_before)
    assert_array_equal(owner.predict_return(model, features), before)
    if model.scaler is not None:
        assert_array_equal(model.scaler.mean_, mean_before)
        assert_array_equal(model.scaler.var_, variance_before)


def test_encoding_is_float32_and_has_every_indicator() -> None:
    features, _, _ = training()
    encoded, medians = owner._encode(features, None)
    assert encoded.dtype == np.float32
    assert encoded.nbytes == len(features) * len(features.columns) * 2 * 4
    assert_array_equal(encoded, encoded_training())
    future = pd.DataFrame({"signal": [np.nan], "empty": [42.0], "complete": [np.nan]})
    transformed, _ = owner._encode(future, medians)
    assert_array_equal(transformed, [[5, 42, 6, 1, 0, 1]])


@pytest.mark.parametrize(("family", "parameters"), [("regularized_linear_return", RIDGE), ("shallow_boosted_return", BOOSTED)])
def test_all_missing_features_keep_dimensions(family: str, parameters: dict[str, Any]) -> None:
    features = pd.DataFrame({"a": [np.nan] * 4, "b": [np.nan] * 4})
    model = owner.fit_return_regressor(family, features, np.arange(4.0), np.ones(4), parameters)
    assert_array_equal(model.medians, [0, 0])
    assert model.estimator.n_features_in_ == 4
    assert np.isfinite(owner.predict_return(model, features)).all()


@pytest.mark.parametrize("values", [
    [True, False], ["1", "2"], [1, "2"], [1.0, np.inf], [-np.inf, 1.0],
    [1 + 2j, 2 + 0j], [1e100, 2.0],
    pd.Series([1.0, 2.0], dtype=object), pd.to_datetime(["2020-01-01", "2020-01-02"]),
    pd.Series([1, 2], dtype="category"), pd.Series([True, pd.NA], dtype="boolean"),
])
def test_invalid_feature_values_rejected_on_fit_and_predict(values: Any) -> None:
    valid = pd.DataFrame({"a": [1.0, 2.0]})
    model = owner.fit_return_regressor("regularized_linear_return", valid, np.arange(2.0), np.ones(2), RIDGE)
    invalid = pd.DataFrame({"a": values})
    with pytest.raises(ValueError, match="features must"):
        owner.fit_return_regressor("regularized_linear_return", invalid, np.arange(2.0), np.ones(2), RIDGE)
    with pytest.raises(ValueError, match="features must"):
        owner.predict_return(model, invalid)


@pytest.mark.parametrize("name", ["target", "sample_weight"])
@pytest.mark.parametrize("values", [
    np.array([True] * 4), np.array(["1"] * 4), np.array([1] * 4, dtype=object),
    np.array([1j] * 4), np.array([np.nan] * 4), np.array([np.inf] * 4),
    np.array([-np.inf] * 4), np.ones((4, 1)), np.ones(3), np.array([]), [1.0] * 4,
])
def test_invalid_supervision(name: str, values: Any) -> None:
    features, target, weights = training()
    arguments = {"target": target, "sample_weight": weights}
    arguments[name] = values
    with pytest.raises(ValueError, match=name):
        owner.fit_return_regressor("regularized_linear_return", features, parameters=RIDGE, **arguments)


@pytest.mark.parametrize("weight", [0.0, -1.0])
def test_weights_must_be_positive(weight: float) -> None:
    features, target, weights = training()
    weights[0] = weight
    with pytest.raises(ValueError, match="strictly positive"):
        owner.fit_return_regressor("regularized_linear_return", features, target, weights, RIDGE)


@pytest.mark.parametrize("features", [
    pd.DataFrame(), pd.DataFrame(columns=["a"]), pd.DataFrame(index=[0]),
    pd.DataFrame([[1, 2]], columns=["a", "a"]), pd.DataFrame({0: [1]}), pd.DataFrame({"": [1]}),
])
def test_invalid_feature_structure(features: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="feature"):
        owner.fit_return_regressor("regularized_linear_return", features, np.ones(len(features)), np.ones(len(features)), RIDGE)


@pytest.mark.parametrize("columns", [
    ["complete", "empty", "signal"], ["signal", "empty"],
    ["signal", "empty", "complete", "extra"], ["signal", "signal", "complete"],
])
def test_prediction_requires_exact_ordered_columns(columns: list[str]) -> None:
    features, target, weights = training()
    model = owner.fit_return_regressor("regularized_linear_return", features, target, weights, RIDGE)
    with pytest.raises(ValueError, match="column"):
        owner.predict_return(model, pd.DataFrame(np.ones((2, len(columns))), columns=columns))


def test_nullable_numeric_features_and_single_row() -> None:
    features = pd.DataFrame({"integer": pd.Series([pd.NA], dtype="Int64"), "real": pd.Series([2.0], dtype="Float64")})
    original = features.copy(deep=True)
    model = owner.fit_return_regressor("regularized_linear_return", features, np.array([-2]), np.array([1]), RIDGE)
    assert_array_equal(model.medians, [0, 2])
    assert_allclose(owner.predict_return(model, features), [-2])
    pd.testing.assert_frame_equal(features, original)
    with pytest.raises(ValueError, match="nonempty"):
        owner.predict_return(model, features.iloc[:0])


@pytest.mark.parametrize("family", ["classifier", "ridge", "boosted"])
def test_unknown_family_rejected(family: str) -> None:
    features, target, weights = training()
    with pytest.raises(ValueError, match="family"):
        owner.fit_return_regressor(family, features, target, weights, {})
