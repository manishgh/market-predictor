"""Chronological research splits, independent security transfer and date weights."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.validation import SessionPurgedWalkForwardSplit


@dataclass(frozen=True)
class ReturnFold:
    number: int
    train: tuple[date, ...]
    embargo: tuple[date, ...]
    score: tuple[date, ...]


def return_folds(sessions: tuple[date, ...], *, count: int, minimum_train: int, embargo: int) -> tuple[ReturnFold, ...]:
    if not sessions or tuple(sorted(set(sessions))) != sessions:
        raise DataReadinessError("return folds require a unique ordered full calendar")
    calendar = pd.DataFrame({"session_date_et": sessions, "decision_group_id": [day.isoformat() for day in sessions]})
    splits = SessionPurgedWalkForwardSplit(n_splits=count, embargo_sessions=embargo,
        min_train_sessions=minimum_train, min_train_rows=1).split(calendar)
    if len(splits) != count:
        raise DataReadinessError("return fold inventory incomplete")
    return tuple(ReturnFold(split.fold + 1, tuple(sessions[i] for i in split.train_indices),
        tuple(sessions[int(split.train_indices[-1]) + 1:int(split.test_indices[0])]),
        tuple(sessions[i] for i in split.test_indices)) for split in splits)


def security_transfer_mask(identities: pd.Series, fraction: float = 0.2) -> np.ndarray:
    if not 0 < fraction < 1 or not identities.map(lambda x: isinstance(x, str) and bool(x) and x.strip() == x).all():
        raise DataReadinessError("security transfer requires canonical identities and fraction")
    threshold = int(fraction * 2**64)
    mapping = {name: int(hashlib.sha256(name.encode()).hexdigest()[:16], 16) < threshold for name in identities.unique()}
    return np.asarray(identities.map(mapping).to_numpy(dtype=bool))


def fit_indices(metadata: pd.DataFrame, train_sessions: tuple[date, ...], cutoff: pd.Timestamp,
    *, transfer: bool) -> np.ndarray:
    eligible = (metadata.session_date_et.isin(train_sessions) & metadata.feature_eligible
        & metadata.fixed_horizon_supervision_available
        & metadata.research_label_mature_at.notna() & metadata.research_label_mature_at.lt(cutoff))
    if transfer:
        eligible &= ~metadata.security_holdout
    indices = np.flatnonzero(eligible.to_numpy())
    if not len(indices):
        raise DataReadinessError("no causally matured research training rows in scope")
    return indices


def date_balanced_weights(sessions: pd.Series) -> np.ndarray:
    if sessions.empty or sessions.isna().any():
        raise DataReadinessError("date-balanced weights require observed sessions")
    counts = sessions.groupby(sessions, sort=False).transform("size")
    return np.asarray((len(sessions) / sessions.nunique() / counts).to_numpy(dtype=np.float64))


def regression_diagnostics(predictions: pd.DataFrame, target: str) -> dict[str, object]:
    scored = predictions.predicted_excess_return.notna()
    known = scored & predictions.fixed_horizon_supervision_available & predictions[target].notna()
    result: dict[str, object] = {"rows": len(predictions), "scored_rows": int(scored.sum()),
        "known_scored_outcomes": int(known.sum()), "unknown_scored_outcomes": int((scored & ~known).sum()),
        "portfolio_evaluated": False, "mse": None, "mae": None, "zero_excess_baseline_mse": None,
        "mean_session_rank_correlation": None, "rank_correlation_sessions": 0}
    if not known.any():
        return result
    observed = predictions.loc[known]
    weights = date_balanced_weights(observed.session_date_et)
    actual = observed[target].to_numpy(dtype=np.float64)
    error = observed.predicted_excess_return.to_numpy(dtype=np.float64) - actual
    correlations = []
    for _, group in observed.groupby("session_date_et", sort=True):
        if len(group) >= 2 and group[target].nunique() > 1 and group.predicted_excess_return.nunique() > 1:
            correlations.append(float(group[target].rank().corr(group.predicted_excess_return.rank())))
    result.update(mse=float(np.average(error**2, weights=weights)), mae=float(np.average(np.abs(error), weights=weights)),
        zero_excess_baseline_mse=float(np.average(actual**2, weights=weights)),
        mean_session_rank_correlation=float(np.mean(correlations)) if correlations else None,
        rank_correlation_sessions=len(correlations))
    return result
