from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

CAUSAL_ISOTONIC_METHOD = "isotonic_prior_outer_folds"


@dataclass(frozen=True)
class CausalCalibrationFit:
    calibrator: IsotonicRegression
    method: str
    train_cutoff_utc: pd.Timestamp
    training_rows: int


def fit_prior_isotonic(
    raw_probability: np.ndarray,
    target: np.ndarray,
    label_available_at_utc: pd.Series,
    *,
    before_utc: pd.Timestamp,
    min_rows: int = 100,
) -> CausalCalibrationFit | None:
    probability = np.asarray(raw_probability, dtype=float)
    labels = np.asarray(target, dtype=int)
    availability = pd.to_datetime(label_available_at_utc, utc=True, errors="coerce")
    if len(probability) != len(labels) or len(probability) != len(availability):
        raise ValueError("calibration arrays must have equal lengths")
    cutoff = pd.Timestamp(before_utc)
    cutoff = cutoff.tz_localize("UTC") if cutoff.tzinfo is None else cutoff.tz_convert("UTC")
    eligible = np.isfinite(probability) & availability.notna().to_numpy() & availability.lt(cutoff).to_numpy()
    if int(eligible.sum()) < min_rows or len(np.unique(labels[eligible])) < 2:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(probability[eligible], labels[eligible])
    train_cutoff = availability.iloc[np.flatnonzero(eligible)].max()
    if not train_cutoff < cutoff:
        raise ValueError("calibration training cutoff must be strictly earlier than scoring")
    return CausalCalibrationFit(
        calibrator=calibrator,
        method=CAUSAL_ISOTONIC_METHOD,
        train_cutoff_utc=train_cutoff,
        training_rows=int(eligible.sum()),
    )


def fit_final_isotonic(
    raw_probability: np.ndarray,
    target: np.ndarray,
    *,
    min_rows: int = 100,
) -> IsotonicRegression | None:
    probability = np.asarray(raw_probability, dtype=float)
    labels = np.asarray(target, dtype=int)
    finite = np.isfinite(probability)
    if int(finite.sum()) < min_rows or len(np.unique(labels[finite])) < 2:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(probability[finite], labels[finite])
    return calibrator


def apply_isotonic(calibrator: object, raw_probability: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return np.asarray(np.clip(np.asarray(raw_probability, dtype=float), 0.0, 1.0), dtype=float)
    predictor = cast(IsotonicRegression, calibrator)
    return np.asarray(
        np.clip(predictor.predict(np.asarray(raw_probability, dtype=float)), 0.0, 1.0),
        dtype=float,
    )
