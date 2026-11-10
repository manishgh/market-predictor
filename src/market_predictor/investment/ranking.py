"""Date-relative investment targets and outcome-supported ranking diagnostics."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from market_predictor.core.errors import DataReadinessError
from market_predictor.investment.diagnostics import _auc
from market_predictor.investment.training import HORIZONS, TARGET, investment_selection_diagnostics
from market_predictor.swing.training.return_validation import date_balanced_weights

RANKING_SCORE = "predicted_rank_score"


def within_date_rank_targets(training: pd.DataFrame) -> NDArray[np.float64]:
    """Rank only the caller's exact mature, non-held-out training partition."""
    required = {"decision_id", "security_id", "decision_session", "horizon_sessions", "research_eligible", TARGET}
    if (training.empty or not required.issubset(training) or not training.index.is_unique
            or training.decision_id.isna().any() or training.decision_id.duplicated().any()
            or training[["security_id", "decision_session", "horizon_sessions"]].isna().any().any()
            or training.duplicated(["security_id", "decision_session"]).any()):
        raise DataReadinessError("ranking targets require a nonempty unique training partition")
    if training.horizon_sessions.nunique() != 1 or training.horizon_sessions.iloc[0] not in HORIZONS:
        raise DataReadinessError("ranking targets must separate supported horizons")
    if (training.research_eligible.isna().any() or getattr(training.research_eligible.dtype, "kind", None) != "b"
            or not training.research_eligible.all() or getattr(training[TARGET].dtype, "kind", None) not in ("i", "u", "f")
            or not np.isfinite(training[TARGET].to_numpy(dtype=np.float64)).all()):
        raise DataReadinessError("ranking targets require known finite eligible training outcomes")
    groups = training.groupby("decision_session", sort=False)[TARGET]
    ranks = groups.rank(method="average")
    counts = groups.transform("size")
    values: NDArray[np.float64] = (2.0 * (ranks - 0.5) / counts - 1.0).to_numpy(dtype=np.float64)
    return values


def ranking_diagnostics(frame: pd.DataFrame, score_column: str = RANKING_SCORE) -> dict[str, Any]:
    """Evaluate ordering on label-only date support without treating scores as returns."""
    identities = ["decision_id", "security_id", "decision_session", "horizon_sessions"]
    if (not {score_column, "feature_eligible", *identities}.issubset(frame)
            or not frame.index.is_unique or frame.horizon_sessions.nunique() != 1
            or frame.horizon_sessions.iloc[0] not in HORIZONS
            or frame[identities].isna().any().any()
            or frame.duplicated(["security_id", "decision_session"]).any()):
        raise DataReadinessError("ranking diagnostics require one supported horizon and unique row indices")
    scored = frame[score_column].notna()
    if (frame.feature_eligible.isna().any() or getattr(frame.feature_eligible.dtype, "kind", None) != "b"
            or not np.array_equal(scored.to_numpy(), frame.feature_eligible.to_numpy())
            or getattr(frame[score_column].dtype, "kind", None) not in ("i", "u", "f")):
        raise DataReadinessError("ranking score mask differs from explicit feature eligibility")
    result: dict[str, Any] = investment_selection_diagnostics(frame, score_column)
    known = frame.loc[scored & frame.research_eligible & frame[TARGET].notna()]
    date_rows = []
    correlations: list[float] = []
    rank_dates: list[str] = []
    constants = 0
    for day, declared in frame.groupby("decision_session", sort=True):
        group = known.loc[known.index.intersection(declared.index, sort=False)]
        reason = ("fewer_than_three_known_outcomes" if len(group) < 3 else
            "constant_known_target" if group[TARGET].nunique() < 2 else None)
        constant = bool(len(group) and group[score_column].nunique() == 1)
        correlation = None
        if reason is None:
            correlation = 0.0 if constant else float(group[TARGET].rank().corr(group[score_column].rank()))
            if not np.isfinite(correlation):
                raise DataReadinessError("supported within-date rank correlation must be finite")
            correlations.append(correlation)
            rank_dates.append(str(day))
            constants += int(constant)
        date_rows.append({"decision_session": str(day), "known_rows": len(group), "correlation": correlation,
            "missing_reason": reason, "constant_score": constant})
    weights = date_balanced_weights(known.decision_session) if len(known) else np.empty(0, dtype=np.float64)
    auc = _auc(known, weights, score_column)
    result.update(score_column=score_column, mean_date_rank_correlation=float(np.mean(correlations)) if correlations else None,
        rank_correlation_dates=len(correlations), constant_score_rank_dates=constants, rank_date_support=rank_dates,
        rank_by_date=date_rows, auc=auc,
        auc_date_support=[row["decision_session"] for row in auc["per_date"] if row["missing_reason"] is None],
        portfolio_evaluated=False, untouched_test_evidence=False, confidence_intervals=None,
        overlap_notice="Monthly 63/252-session outcomes overlap; observations are not independent.",
        score_interpretation="ordering_only_not_return_or_probability")
    return result
