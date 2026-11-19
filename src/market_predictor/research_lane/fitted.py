"""Walk-forward fitted combinations of centered-rank technical features (fixed settings, no tuning).

Target: centered cross-sectional rank of the research-label forward return among eligible
stocks with a known label on the date. Each scored calendar year is one test fold; its
model is trained only on decision rows t >= the first decision row whose label window ends
strictly before the fold's first session (t + h < first test row), sampled every fifth
session. Every training date carries equal total weight; row weights are 1 / (rows on the
date), rescaled by one constant so the mean row weight is 1 (relative weights unchanged;
this keeps ``min_child_weight`` and the ridge penalty on the scale of row counts).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

from market_predictor.research_lane.predictors import centered_rank

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]

SCORED_YEARS: Final = (2021, 2022, 2023, 2024)
SAMPLE_EVERY_SESSIONS: Final = 5
LEARNERS: Final = ("ridge", "xgboost")
RIDGE_ALPHA: Final = 1000.0
XGBOOST_PARAMETERS: Final[Mapping[str, Any]] = MappingProxyType({
    "n_estimators": 300, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.5,
    "min_child_weight": 200, "reg_lambda": 10, "tree_method": "hist", "n_jobs": 4,
})
XGBOOST_SEEDS: Final = (0, 1, 2)


@dataclass(frozen=True)
class Fold:
    year: int
    test_start: int  # first session row of the scored year
    test_stop: int  # one past the last session row of the scored year


def session_years(sessions: npt.NDArray[np.datetime64]) -> IntArray:
    years: IntArray = sessions.astype("datetime64[Y]").astype(np.int64) + 1970
    return years


def scored_folds(sessions: npt.NDArray[np.datetime64], years: Sequence[int] = SCORED_YEARS) -> tuple[Fold, ...]:
    calendar = session_years(sessions)
    folds = []
    for year in years:
        rows = np.flatnonzero(calendar == year)
        if rows.size == 0:
            raise ValueError(f"no session in scored year {year}")
        folds.append(Fold(year=int(year), test_start=int(rows[0]), test_stop=int(rows[-1]) + 1))
    return tuple(folds)


def training_rows(first_row: int, test_start: int, horizon: int, step: int = SAMPLE_EVERY_SESSIONS) -> IntArray:
    """Every ``step``-th decision row from ``first_row`` whose label exit row t + h is before ``test_start``."""
    rows = np.arange(first_row, test_start, step, dtype=np.int64)
    kept: IntArray = rows[rows + horizon < test_start]
    return kept


def target_ranks(forward: FloatArray, eligible: BoolArray) -> FloatArray:
    """Centered rank of the forward return among eligible stocks with a known label on each date."""
    return centered_rank(forward, eligible)


def design(features: Mapping[str, Float32Array], names: Sequence[str], rows: IntArray, cells: BoolArray) -> Float32Array:
    """Stack the selected cells (row-major over rows x securities) into an observations x features matrix."""
    matrix = np.empty((int(cells.sum()), len(names)), dtype=np.float32)
    for column, name in enumerate(names):
        matrix[:, column] = features[name][rows][cells]
    return matrix


def date_weights(cells: BoolArray) -> FloatArray:
    """1 / (rows on the date), times (total rows / dates) so the mean weight is 1; each date sums to the same total."""
    per_date = cells.sum(axis=1)
    dates = int((per_date > 0).sum())
    total = int(per_date.sum())
    if total == 0:
        return np.zeros(0, dtype=np.float64)
    weights = np.broadcast_to((1.0 / np.maximum(per_date, 1)).reshape(-1, 1), cells.shape)[cells]
    return np.asarray(weights * (total / dates), dtype=np.float64)


@dataclass
class FittedModel:
    learner: str
    predict: Callable[[Float32Array], FloatArray]
    importance: FloatArray
    coefficients: FloatArray | None = None
    intercept: float | None = None


def fit_model(learner: str, x: Float32Array, y: FloatArray, weights: FloatArray) -> FittedModel:
    if learner == "ridge":
        ridge = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        ridge.fit(np.nan_to_num(x.astype(np.float64), nan=0.0), y, sample_weight=weights)
        coefficients = np.asarray(ridge.coef_, dtype=np.float64)

        def predict_ridge(matrix: Float32Array) -> FloatArray:
            return np.asarray(ridge.predict(np.nan_to_num(matrix.astype(np.float64), nan=0.0)), dtype=np.float64)

        return FittedModel(learner, predict_ridge, np.abs(coefficients), coefficients, float(ridge.intercept_))
    if learner == "xgboost":
        models = []
        for seed in XGBOOST_SEEDS:
            model = XGBRegressor(**XGBOOST_PARAMETERS, random_state=seed, importance_type="gain")
            model.fit(x, y, sample_weight=weights)
            models.append(model)
        importance = np.mean([np.asarray(model.feature_importances_, dtype=np.float64) for model in models], axis=0)

        def predict_trees(matrix: Float32Array) -> FloatArray:
            return np.mean([np.asarray(model.predict(matrix), dtype=np.float64) for model in models], axis=0)

        return FittedModel(learner, predict_trees, importance)
    raise ValueError(f"unknown learner {learner}")


@dataclass
class WalkForward:
    scores: FloatArray  # out-of-sample predictions, NaN outside scored years and non-eligible cells
    importance: FloatArray  # per feature, mean over folds
    folds: list[dict[str, Any]] = field(default_factory=list)
    models: list[FittedModel] = field(default_factory=list)


def walk_forward(features: Mapping[str, Float32Array], names: Sequence[str], target: FloatArray, *,
                 eligible: BoolArray, sessions: npt.NDArray[np.datetime64], first_row: int, horizon: int,
                 learner: str, folds: Sequence[Fold], checkpoint: Callable[[str], None] | None = None,
                 keep_models: bool = False) -> WalkForward:
    """Fit one model per fold on purged sampled history and predict every eligible stock in the fold's year."""
    scores = np.full(eligible.shape, np.nan, dtype=np.float64)
    importances: list[FloatArray] = []
    report: list[dict[str, Any]] = []
    kept: list[FittedModel] = []
    for fold in folds:
        rows = training_rows(first_row, fold.test_start, horizon)
        cells = eligible[rows] & np.isfinite(target[rows])
        x = design(features, names, rows, cells)
        y = target[rows][cells].astype(np.float64)
        model = fit_model(learner, x, y, date_weights(cells))
        del x, y
        test = np.arange(fold.test_start, fold.test_stop, dtype=np.int64)
        test_cells = eligible[test]
        block = np.full(test_cells.shape, np.nan, dtype=np.float64)
        block[test_cells] = model.predict(design(features, names, test, test_cells))
        scores[test] = block
        importances.append(model.importance)
        report.append({
            "scored_year": fold.year, "training_dates": int((cells.sum(axis=1) > 0).sum()),
            "training_rows_stock_sessions": int(cells.sum()),
            "first_training_session": str(sessions[rows[0]]) if rows.size else None,
            "last_training_session": str(sessions[rows[-1]]) if rows.size else None,
            "last_training_label_exit_session": str(sessions[rows[-1] + horizon]) if rows.size else None,
            "first_test_session": str(sessions[fold.test_start]), "last_test_session": str(sessions[fold.test_stop - 1]),
            "test_predictions_stock_sessions": int(test_cells.sum()),
        })
        if keep_models:
            kept.append(model)
        if checkpoint is not None:
            checkpoint(f"research lane fitted {learner} h={horizon} fold {fold.year}")
    importance = np.mean(importances, axis=0) if importances else np.zeros(len(names), dtype=np.float64)
    return WalkForward(scores=scores, importance=importance, folds=report, models=kept)
