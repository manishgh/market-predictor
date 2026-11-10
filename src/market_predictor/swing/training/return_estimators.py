"""Train-only missing-value encoding and bounded, weighted return regressors."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits  # type: ignore[import-untyped]


@dataclass(frozen=True)
class NativeMissingFeature:
    name: str
    finite_training_rows: int
    lower: float | None
    upper: float | None


@dataclass
class FittedReturnRegressor:
    family: str
    feature_names: tuple[str, ...]
    medians: NDArray[np.float64]
    scaler: Any | None
    estimator: Any
    parameters: dict[str, Any]
    native_missing_features: tuple[NativeMissingFeature, ...] = ()


def _feature_names(features: pd.DataFrame) -> tuple[str, ...]:
    if not isinstance(features, pd.DataFrame) or features.empty:
        raise ValueError("features must be a nonempty DataFrame with at least one column")
    names = tuple(features.columns)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError("feature column names must be nonempty strings")
    if len(set(names)) != len(names):
        raise ValueError("feature column names must be unique")
    return names


def _vector(values: NDArray[Any], name: str, rows: int) -> NDArray[np.float64]:
    if not isinstance(values, np.ndarray) or values.ndim != 1 or len(values) != rows:
        raise ValueError(f"{name} must be a one-dimensional ndarray matching feature rows")
    if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
        raise ValueError(f"{name} must contain finite real numbers, not bool or text")
    # Own these buffers: downstream weighted solvers may modify their inputs.
    result = values.astype(np.float64, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} exceeds float64 range")
    if name == "sample_weight" and not (result > 0).all():
        raise ValueError("sample_weight must be strictly positive")
    return result


def _encode(
    features: pd.DataFrame, medians: NDArray[np.float64] | None,
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    rows, columns = features.shape
    learned = np.empty(columns, dtype=np.float64) if medians is None else medians
    if learned.shape != (columns,) or (medians is not None and not np.isfinite(learned).all()):
        raise ValueError("model medians must be finite and match the feature count")
    encoded = np.empty((rows, 2 * columns), dtype=np.float32)
    # Only one original-precision column and its mask are materialized at a time.
    # Indicator columns always follow all value columns, even without fit-time NaNs.
    for index in range(columns):
        column = features.iloc[:, index]
        if getattr(column.dtype, "kind", None) not in ("i", "u", "f"):
            raise ValueError("features must contain real numeric columns, not bool or text")
        values = column.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        missing = np.isnan(values)
        if np.isinf(values).any() or (np.abs(values[~missing]) > np.finfo(np.float32).max).any():
            raise ValueError("features must be finite or NaN and within float32 range")
        if medians is None:
            learned[index] = 0.0 if missing.all() else np.median(values[~missing])
        values[missing] = learned[index]
        encoded[:, index] = values
        encoded[:, columns + index] = missing
    if not np.isfinite(encoded).all():
        raise ValueError("encoded features must be finite float32 values")
    return encoded, learned


def _native_values(column: pd.Series) -> NDArray[np.float64]:
    if getattr(column.dtype, "kind", None) not in ("i", "u", "f"):
        raise ValueError("native-missing features must contain real numeric values")
    values: NDArray[np.float64] = column.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
    if np.isinf(values).any():
        raise ValueError("native-missing features must be finite or NaN")
    return values


def _fit_native_missing(features: pd.DataFrame, columns: tuple[str, ...]) -> tuple[NativeMissingFeature, ...]:
    fitted = []
    for name in columns:
        values = _native_values(features[name])
        observed = values[~np.isnan(values)]
        if observed.size:
            lower, upper = np.quantile(observed, (0.01, 0.99), method="linear")
            fitted.append(NativeMissingFeature(name, int(observed.size), float(lower), float(upper)))
        else:
            fitted.append(NativeMissingFeature(name, 0, None, None))
    return tuple(fitted)


def _encode_model_features(features: pd.DataFrame, medians: NDArray[np.float64] | None,
                           native: tuple[NativeMissingFeature, ...]) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    if not native:
        return _encode(features, medians)
    if len(native) >= len(features.columns) or tuple(features.columns[-len(native):]) != tuple(column.name for column in native):
        raise ValueError("fitted native-missing features must match the ordered suffix")
    legacy, learned = _encode(features.iloc[:, :-len(native)], medians)
    encoded = np.empty((len(features), legacy.shape[1] + len(native)), dtype=np.float32)
    encoded[:, :legacy.shape[1]] = legacy
    for index, fitted in enumerate(native, start=legacy.shape[1]):
        values = _native_values(features[fitted.name])
        lower, upper, count = fitted.lower, fitted.upper, fitted.finite_training_rows
        if type(count) is not int or count < 0:
            raise ValueError("fitted native-missing support must be a nonnegative row count")
        if count == 0 and lower is None and upper is None:
            values.fill(np.nan)
        elif count > 0 and lower is not None and upper is not None and isfinite(lower) and isfinite(upper) and lower <= upper:
            np.clip(values, lower, upper, out=values)
        else:
            raise ValueError("fitted native-missing bounds must agree with training support and be finite and ordered")
        if (np.abs(values[~np.isnan(values)]) > np.finfo(np.float32).max).any():
            raise ValueError("clipped native-missing features exceed float32 range")
        encoded[:, index] = values
    return encoded, learned


def fit_return_regressor(
    family: str,
    features: pd.DataFrame,
    target: NDArray[Any],
    sample_weight: NDArray[Any],
    parameters: dict[str, Any],
    *, native_missing_columns: tuple[str, ...] = (),
) -> FittedReturnRegressor:
    """Fit the parent's fixed learner specification on this training partition only."""
    if family not in ("regularized_linear_return", "shallow_boosted_return"):
        raise ValueError(f"unsupported return regressor family: {family}")
    names = _feature_names(features)
    if native_missing_columns and (family != "shallow_boosted_return"
            or len(native_missing_columns) >= len(names) or names[-len(native_missing_columns):] != native_missing_columns):
        raise ValueError("native-missing features require a boosted regressor and exact ordered suffix")
    labels = _vector(target, "target", len(features))
    weights = _vector(sample_weight, "sample_weight", len(features))
    fitted_parameters = dict(parameters)
    scaler: Any | None = None
    estimator: Any
    if family == "regularized_linear_return":
        estimator = Ridge(**fitted_parameters)
    else:
        from xgboost import XGBRegressor

        estimator = XGBRegressor(**fitted_parameters)
    with threadpool_limits(limits=1):
        native = _fit_native_missing(features, native_missing_columns)
        encoded, medians = _encode_model_features(features, None, native)
        if family == "regularized_linear_return":
            scaler = StandardScaler(copy=False)
            scaler.fit(encoded, sample_weight=weights)
            scaler.transform(encoded, copy=False)
            if not np.isfinite(encoded).all():
                raise ValueError("scaled features must be finite")
        estimator.fit(encoded, labels, sample_weight=weights)
    return FittedReturnRegressor(family, names, medians, scaler, estimator, fitted_parameters, native)


def predict_return(model: FittedReturnRegressor, features: pd.DataFrame) -> NDArray[np.float64]:
    """Predict uncalibrated returns using the exact fitted ordered feature contract."""
    if _feature_names(features) != model.feature_names:
        raise ValueError("prediction features must match the exact fitted column order")
    if model.family not in ("regularized_linear_return", "shallow_boosted_return"):
        raise ValueError(f"unsupported return regressor family: {model.family}")
    if (model.family == "regularized_linear_return") != (model.scaler is not None):
        raise ValueError("fitted scaler does not match the regressor family")
    if model.native_missing_features and model.family != "shallow_boosted_return":
        raise ValueError("fitted native-missing features require a boosted regressor")
    with threadpool_limits(limits=1):
        encoded, _ = _encode_model_features(features, model.medians, model.native_missing_features)
        if model.scaler is not None:
            model.scaler.transform(encoded, copy=False)
            if not np.isfinite(encoded).all():
                raise ValueError("scaled features must be finite")
        prediction = np.asarray(model.estimator.predict(encoded), dtype=np.float64)
    if prediction.shape != (len(features),) or not np.isfinite(prediction).all():
        raise ValueError("return predictions must be a finite vector matching feature rows")
    return prediction
