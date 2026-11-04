"""One raw-score and fitted-calibration path for candidate evaluation and serving."""
from __future__ import annotations

from typing import Any

import numpy as np

from market_predictor.core.errors import DataReadinessError


def candidate_raw_scores(estimator: Any, features: np.ndarray) -> np.ndarray:
    """Keep classifier class-one probabilities and ranker/regressor score units.

    Rank scores and predicted returns are inputs to the already fitted binary
    calibrator. They are never clipped or interpreted as probabilities here.
    """
    if features.ndim != 2 or not len(features):
        raise DataReadinessError("candidate scoring requires a nonempty feature matrix")
    try:
        if callable(getattr(estimator, "predict_proba", None)):
            probabilities = np.asarray(estimator.predict_proba(features), dtype="float64")
            if probabilities.shape != (len(features), 2):
                raise ValueError("binary estimator probability shape differs")
            scores = probabilities[:, 1]
        else:
            scores = np.asarray(estimator.predict(features), dtype="float64")
        if scores.shape != (len(features),) or not np.isfinite(scores).all():
            raise ValueError("estimator scores must be a finite vector matching feature rows")
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise DataReadinessError("candidate estimator produced invalid raw scores") from exc
    return scores


def candidate_calibrated_probabilities(estimator: Any, calibrator: Any, features: np.ndarray) -> np.ndarray:
    """Apply the original fitted calibrator to exactly one raw score per row."""
    scores = candidate_raw_scores(estimator, features)
    try:
        probabilities = np.asarray(calibrator.predict_proba(scores.reshape(-1, 1)), dtype="float64")
        if probabilities.shape != (len(features), 2):
            raise ValueError("binary calibrator probability shape differs")
        calibrated = probabilities[:, 1]
        if (not np.isfinite(calibrated).all() or (calibrated < 0.0).any() or (calibrated > 1.0).any()):
            raise ValueError("calibrated probabilities must be finite in [0, 1]")
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise DataReadinessError("candidate calibrator produced invalid probabilities") from exc
    return calibrated
