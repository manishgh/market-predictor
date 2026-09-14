"""Frozen two-specification initial-fit research request."""
from __future__ import annotations

from typing import Literal

from pydantic import Field

from market_predictor.swing.contracts.holding_accounting import HoldingContract
from market_predictor.swing.contracts.holding_materialization import SourcePin


class LinearReturnSettings(HoldingContract):
    alpha: float = Field(ge=1, le=1)
    fit_intercept: Literal[True]
    solver: Literal["lsqr"]
    tol: float = Field(ge=0.000001, le=0.000001)
    max_iter: Literal[10000]


class BoostedReturnSettings(HoldingContract):
    objective: Literal["reg:squarederror"]
    tree_method: Literal["hist"]
    max_depth: Literal[3]
    n_estimators: Literal[200]
    learning_rate: float = Field(ge=0.05, le=0.05)
    reg_lambda: float = Field(ge=10, le=10)
    reg_alpha: float = Field(ge=0, le=0)
    min_child_weight: float = Field(ge=1, le=1)
    subsample: float = Field(ge=1, le=1)
    colsample_bytree: float = Field(ge=1, le=1)
    n_jobs: Literal[1]
    random_state: Literal[42]


class ReturnTrainingPolicy(HoldingContract):
    schema_version: Literal["market_predictor.swing_return_training"]
    scope: Literal["initial_fit_research_only"]
    readiness: SourcePin
    readiness_config: SourcePin
    feature_profile: Literal["existing_technical"]
    published_profile: Literal["technical_market"]
    target: Literal["spy_fixed_horizon_excess_return"]
    folds: Literal[4]
    minimum_train_sessions: Literal[503]
    embargo_sessions: Literal[10]
    holdout_fraction: float = Field(ge=0.2, le=0.2)
    holdout_assignment: Literal["unseeded_sha256_security_id_first_64_bits"]
    missingness: Literal["training_median_all_empty_zero_all_column_indicators"]
    weighting: Literal["equal_date_total_mean_row_one"]
    preprocessing_scope: Literal["separately_fitted_per_fold_and_evaluation_scope"]
    final_fit: Literal["two_models_all_initial_fit_supervision"]
    linear: LinearReturnSettings
    boosted: BoostedReturnSettings
