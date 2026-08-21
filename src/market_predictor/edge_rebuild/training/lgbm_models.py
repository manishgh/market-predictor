"""Production-grade candidate training for the ten-session edge-rebuild swing strategy."""

from __future__ import annotations

from datetime import date
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.edge_rebuild.swing_features import (
    SWING_FEATURE_PROFILE,
)
from market_predictor.edge_rebuild.training.swing_types import (
    CandidateSpec,
    FittedCandidate,
    SwingTrainingConfig,
    _iso,
)
from market_predictor.edge_rebuild.training.walk_forward import (
    _split_fit_calibration,
)
from market_predictor.v3.errors import DataReadinessError

TRAINING_SCHEMA: Final = "edge_rebuild.swing_training.v5"
MODEL_SCHEMA: Final = "edge_rebuild.swing_candidate.v5"
EVALUATION_SCHEMA: Final = "edge_rebuild.swing_evaluation.v7"
MODEL_CARD_SCHEMA: Final = "edge_rebuild.swing_model_card.v7"
OUTPUT_AUTHORITY_SCHEMA: Final = "edge_rebuild.swing_candidate_authority.v5"
SWING_BASELINE_BUNDLE_PREFIX: Final = "swing_baseline_bundle."
DECISION_START_DATE: Final = date(2019, 7, 9)
HORIZON_SESSIONS: Final = 10
ALLOWED_PROFILES: Final = (
    SWING_FEATURE_PROFILE,
)
# The learned families, per profile and per (rate, depth) point. `dual_hurdle`
# was dropped: it scored 0.452-0.462 AUC on the v12 run -- below chance -- had no
# test covering it, and its four slots pushed the grid past the contract's
# six-candidate experiment budget.
_XGB_GRID: Final = (
    ("xgbranker", "xgboost_ranker"),
    ("xgbregressor", "xgboost_regressor"),
)
_XGB_FAMILIES: Final = len(_XGB_GRID)
_MANIFEST_NAME: Final = "_manifest.json"
_AUTHORITY_NAME: Final = "_authority.json"
_CANDIDATE_NAME: Final = "candidate.joblib"
_EVALUATION_NAME: Final = "evaluation.json"
_MODEL_CARD_NAME: Final = "model_card.json"
_TEXT_COLUMNS: Final = (
    "decision_id",
    "decision_group_id",
    "ticker",
    "security_id",
    "sector",
    "primary_benchmark",
    "market_regime",
)




















































def _fit_candidate(
    spec: CandidateSpec,
    train: pd.DataFrame,
    config: SwingTrainingConfig,
) -> FittedCandidate:
    fit_sessions, calibration_sessions = _split_fit_calibration(train, config)
    fit_mask = train["session_date_et"].isin(fit_sessions)
    calibration_mask = train["session_date_et"].isin(calibration_sessions)
    if (
        train.loc[fit_mask, "target"].nunique() != 2
        or train.loc[calibration_mask, "target"].nunique() != 2
    ):
        raise DataReadinessError("fit and calibration partitions must contain both classes")
    columns = list(spec.feature_columns)

    if spec.estimator_family == "xgboost_ranker":
        train_fit = train.loc[fit_mask].sort_values("decision_group_id")
        x_fit = train_fit[columns].to_numpy(dtype="float32", copy=False)
        y_fit = train_fit["relevance_score"].to_numpy(dtype="float32", copy=False)
        qid_fit = pd.factorize(train_fit["decision_group_id"])[0]
        fit_weight = train_fit.drop_duplicates(
            subset=["decision_group_id"]
        )["ranking_reliability_weight"].to_numpy(dtype="float64", copy=False)
    elif spec.estimator_family == "xgboost_regressor":
        x_fit = train.loc[fit_mask, columns].to_numpy(dtype="float32", copy=False)
        y_fit = train.loc[fit_mask, "barrier_net_return"].to_numpy(dtype="float32", copy=False)
        qid_fit = None
        fit_weight = train.loc[
            fit_mask,
            "ranking_reliability_weight",
        ].to_numpy(dtype="float64", copy=False)
    else:
        x_fit = train.loc[fit_mask, columns].to_numpy(dtype="float32", copy=False)
        y_fit = train.loc[fit_mask, "target"].to_numpy(dtype="int8", copy=False)
        qid_fit = None
        fit_weight = train.loc[
            fit_mask,
            "ranking_reliability_weight",
        ].to_numpy(dtype="float64", copy=False)
    if spec.estimator_family == "logistic":
        estimator: Any = Pipeline(
            [
                (
                    "impute",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                        keep_empty_features=True,
                    ),
                ),
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(spec.hyperparameters["C"]),
                        max_iter=500,
                        random_state=config.random_seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    elif spec.estimator_family == "xgboost_ranker":
        import xgboost as xgb
        estimator = xgb.XGBRanker(
            objective="rank:pairwise",
            learning_rate=float(spec.hyperparameters["learning_rate"]),
            max_depth=int(spec.hyperparameters["max_depth"]),
            n_estimators=int(spec.hyperparameters["n_estimators"]),
            random_state=config.random_seed,
            tree_method="hist",
        )
    elif spec.estimator_family == "xgboost_regressor":
        import xgboost as xgb
        estimator = xgb.XGBRegressor(
            objective=_linex_objective,
            learning_rate=float(spec.hyperparameters["learning_rate"]),
            max_depth=int(spec.hyperparameters["max_depth"]),
            n_estimators=int(spec.hyperparameters["n_estimators"]),
            random_state=config.random_seed,
            tree_method="hist",
            reg_lambda=10.0,
        )
    else:
        raise DataReadinessError(f"unknown swing estimator family: {spec.estimator_family}")
    if spec.estimator_family == "logistic":
        estimator.fit(x_fit, y_fit, model__sample_weight=fit_weight)
    elif spec.estimator_family == "xgboost_ranker":
        estimator.fit(x_fit, y_fit, qid=qid_fit, sample_weight=fit_weight)
    else:
        estimator.fit(x_fit, y_fit, sample_weight=fit_weight)
    del x_fit, y_fit, fit_weight
    raw = _raw_probability(
        estimator,
        train.loc[calibration_mask, columns].to_numpy(dtype="float32", copy=False),
    )
    calibrator = LogisticRegression(
        C=1.0,
        max_iter=300,
        random_state=config.random_seed,
        solver="lbfgs",
    )
    calibrator.fit(
        raw.reshape(-1, 1),
        train.loc[calibration_mask, "target"].to_numpy(dtype="int8", copy=False),
        sample_weight=train.loc[
            calibration_mask,
            "ranking_reliability_weight",
        ].to_numpy(dtype="float64", copy=False),
    )
    return FittedCandidate(
        estimator=estimator,
        calibrator=calibrator,
        feature_columns=spec.feature_columns,
        fit_sessions=len(fit_sessions),
        calibration_sessions=len(calibration_sessions),
        calibration_cutoff_utc=_iso(
            train.loc[calibration_mask, "label_available_at_utc"].max()
        ),
    )


def _linex_objective(
    y_true: np.ndarray, y_pred: np.ndarray, sample_weight: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    residual = y_pred - y_true
    
    # LinEx parameters
    # a > 0 heavily penalizes overestimation (predicted > actual) exponentially
    a = 15.0
    b = 1.0

    # Gradient: b * a * (exp(a * residual) - 1)
    # Hessian: b * a^2 * exp(a * residual)
    
    # Clip residual to prevent overflow in exp
    clipped_residual = np.clip(residual, -1.0, 1.0)
    exp_term = np.exp(a * clipped_residual)
    
    grad = b * a * (exp_term - 1.0)
    hess = b * (a ** 2) * exp_term

    if sample_weight is not None:
        grad *= sample_weight
        hess *= sample_weight
    return grad, hess


def _raw_probability(estimator: Any, features: np.ndarray) -> np.ndarray:
    if hasattr(estimator, "predict_proba"):
        probability = np.asarray(estimator.predict_proba(features)[:, 1], dtype="float64")
    else:
        probability = np.asarray(estimator.predict(features), dtype="float64")
    if not np.isfinite(probability).all():
        raise DataReadinessError("estimator produced non-finite probabilities")
    return probability


def _predict_probability(
    fitted: FittedCandidate,
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    raw = _raw_probability(
        fitted.estimator,
        frame.loc[:, list(feature_columns)].to_numpy(dtype="float32", copy=False),
    )
    calibrated = np.asarray(
        fitted.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1],
        dtype="float64",
    )
    if (
        not np.isfinite(calibrated).all()
        or (calibrated < 0.0).any()
        or (calibrated > 1.0).any()
    ):
        raise DataReadinessError("calibrated probabilities must be finite in [0, 1]")
    return calibrated


































