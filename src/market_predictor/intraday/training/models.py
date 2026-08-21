from __future__ import annotations

from market_predictor.intraday.training.config import IntradayDevelopmentConfig
from market_predictor.intraday.training.config import _CandidateSpec

"""Development-only, cost-aware intraday model training and evaluation."""

import math
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.core.errors import DataReadinessError

MODEL_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_candidate.v1"
EVALUATION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_evaluation.v1"
AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_authority.v1"
FUTURE_EVALUATION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_future_evaluation.v1"
FUTURE_AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_future_authority.v1"
_AUTHORITY_NAME: Final = "_authority.json"
_MANIFEST_NAME: Final = "_manifest.json"
_EVALUATION_NAME: Final = "evaluation.json"
_MODEL_CARD_NAME: Final = "model_card.json"
_CANDIDATE_NAME: Final = "candidate.joblib"
_FUTURE_EVALUATION_NAME: Final = "future_evaluation.json"
_POSITION_LEDGER_NAME: Final = "position_ledger.parquet"
_DAILY_LEDGER_NAME: Final = "daily_ledger.parquet"
_VALIDATION_PREDICTIONS_NAME: Final = "validation_predictions.parquet"


@dataclass(frozen=True, slots=True)
class _FittedPair:
    opportunity_estimator: Any
    downside_estimator: Any
    downside_calibrator: LogisticRegression
    fit_sessions: tuple[str, ...]
    calibration_sessions: tuple[str, ...]


def _fit_opportunity(
    spec: _CandidateSpec,
    features: np.ndarray,
    target: np.ndarray,
    config: IntradayDevelopmentConfig,
) -> Any:
    if spec.family == "ridge_logistic_pair":
        estimator: Any = Pipeline(
            [
                ("scale", StandardScaler(copy=False)),
                ("regressor", Ridge(alpha=float(spec.hyperparameters["alpha"]), solver="cholesky")),
            ]
        )
    elif spec.family == "hgb_pair":
        estimator = HistGradientBoostingRegressor(
            learning_rate=float(spec.hyperparameters["learning_rate"]),
            max_leaf_nodes=int(spec.hyperparameters["max_leaf_nodes"]),
            max_iter=int(spec.hyperparameters["max_iter"]),
            max_bins=int(spec.hyperparameters["max_bins"]),
            random_state=config.random_seed,
        )
    else:
        raise AssertionError(f"unsupported intraday opportunity family: {spec.family}")
    estimator.fit(features, target)
    return estimator


def _fit_downside(
    spec: _CandidateSpec,
    features: np.ndarray,
    target: np.ndarray,
    config: IntradayDevelopmentConfig,
) -> Any:
    if len(np.unique(target)) != 2:
        raise DataReadinessError("downside fit requires both stop-hit classes")
    if spec.family == "ridge_logistic_pair":
        estimator: Any = Pipeline(
            [
                ("scale", StandardScaler(copy=False)),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(spec.hyperparameters["downside_c"]),
                        max_iter=500,
                        random_state=config.random_seed,
                    ),
                ),
            ]
        )
    elif spec.family == "hgb_pair":
        estimator = HistGradientBoostingClassifier(
            learning_rate=float(spec.hyperparameters["learning_rate"]),
            max_leaf_nodes=int(spec.hyperparameters["max_leaf_nodes"]),
            max_iter=int(spec.hyperparameters["max_iter"]),
            max_bins=int(spec.hyperparameters["max_bins"]),
            random_state=config.random_seed,
        )
    else:
        raise AssertionError(f"unsupported intraday downside family: {spec.family}")
    estimator.fit(features, target)
    return estimator


def _fit_pair(
    spec: _CandidateSpec,
    data: pd.DataFrame,
    features: np.ndarray,
    opportunity_target: np.ndarray,
    downside_target: np.ndarray,
    training_sessions: tuple[str, ...],
    config: IntradayDevelopmentConfig,
    *,
    excluded_securities: frozenset[str] = frozenset(),
) -> _FittedPair:
    fit_sessions, calibration_sessions = _split_downside_calibration(data, training_sessions, config)
    training_mask = data["session_date_et"].isin(training_sessions).to_numpy(copy=True)
    if excluded_securities:
        training_mask &= ~data["security_id"].astype(str).isin(excluded_securities).to_numpy()
    downside_fit_mask = training_mask & data["session_date_et"].isin(fit_sessions).to_numpy()
    calibration_mask = training_mask & data["session_date_et"].isin(calibration_sessions).to_numpy()
    opportunity = _fit_opportunity(spec, features[training_mask], opportunity_target[training_mask], config)
    downside = _fit_downside(spec, features[downside_fit_mask], downside_target[downside_fit_mask], config)
    calibration_target = downside_target[calibration_mask]
    if len(np.unique(calibration_target)) != 2:
        raise DataReadinessError("downside calibration requires both stop-hit classes")
    raw = _raw_stop_logit(downside, features[calibration_mask])
    calibrator = LogisticRegression(C=1_000_000.0, max_iter=500, random_state=config.random_seed)
    calibrator.fit(raw.reshape(-1, 1), calibration_target)
    return _FittedPair(
        opportunity_estimator=opportunity,
        downside_estimator=downside,
        downside_calibrator=calibrator,
        fit_sessions=fit_sessions,
        calibration_sessions=calibration_sessions,
    )


def _split_downside_calibration(
    data: pd.DataFrame,
    training_sessions: tuple[str, ...],
    config: IntradayDevelopmentConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    calibration_count = max(
        config.minimum_calibration_sessions,
        math.ceil(len(training_sessions) * config.calibration_fraction),
    )
    calibration_start = len(training_sessions) - calibration_count
    fit_end = calibration_start - config.embargo_sessions
    if fit_end < 20:
        raise DataReadinessError("training history is too short for downside calibration")
    fit_sessions = training_sessions[:fit_end]
    calibration_sessions = training_sessions[calibration_start:]
    fit = data.loc[data["session_date_et"].isin(fit_sessions)]
    calibration = data.loc[data["session_date_et"].isin(calibration_sessions)]
    if fit.empty or calibration.empty or fit["label_available_at_utc"].max() >= calibration["decision_time_utc"].min():
        raise DataReadinessError("downside calibration is not causally purged")
    return fit_sessions, calibration_sessions


def _raw_stop_logit(estimator: Any, features: np.ndarray) -> np.ndarray:
    probability = np.asarray(estimator.predict_proba(features)[:, 1], dtype="float64")
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(probability / (1.0 - probability))


def _predict_pair(fitted: _FittedPair, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    opportunity = np.asarray(fitted.opportunity_estimator.predict(features), dtype="float64")
    raw = _raw_stop_logit(fitted.downside_estimator, features)
    downside = np.asarray(
        fitted.downside_calibrator.predict_proba(raw.reshape(-1, 1))[:, 1],
        dtype="float64",
    )
    if not np.isfinite(opportunity).all() or not np.isfinite(downside).all():
        raise DataReadinessError("intraday paired scores must be finite")
    return opportunity, downside
