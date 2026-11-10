"""Purged retrospective investment splits and matched, non-portfolio diagnostics."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.training.return_validation import date_balanced_weights, security_transfer_mask

HORIZONS = (252, 63)
SCORE_YEARS = (2022, 2023, 2024)
PROFILES = ("technical_relationships", "all_features")
FAMILIES = ("regularized_linear_return", "shallow_boosted_return")
SCOPES = ("temporal", "security_transfer")
TARGET = "target_net_excess_vs_spy"


@dataclass(frozen=True)
class InvestmentFold:
    horizon_sessions: int
    score_year: int
    scope: str
    information_cutoff_utc: pd.Timestamp
    training_maturity_cutoff_utc: pd.Timestamp
    train_indices: np.ndarray
    score_indices: np.ndarray
    prediction_indices: np.ndarray
    missing_reasons: tuple[str, ...]


def investment_fold(metadata: pd.DataFrame, horizon: int, score_year: int,
                    scope: Literal["temporal", "security_transfer"]) -> InvestmentFold:
    """Purge by outcome end, with an embargo measured in exchange sessions, not months.

    Prediction coverage is independent of future outcome availability. Unknown
    outcomes stay in the scoring population and cannot improve selected coverage.
    """
    if horizon not in HORIZONS or score_year not in SCORE_YEARS or scope not in SCOPES:
        raise ValueError("unsupported frozen investment experiment unit")
    dates = pd.to_datetime(metadata.decision_session)
    clocks = pd.to_datetime(metadata.decision_time_utc, utc=True)
    ends = pd.to_datetime(metadata.economic_maturity_at_utc, utc=True)
    horizon_mask = metadata.horizon_sessions.eq(horizon)
    scoring = horizon_mask & dates.dt.year.eq(score_year)
    if not scoring.any():
        raise DataReadinessError("declared scoring year absent from complete investment roster")
    cutoff = clocks.loc[scoring].min()
    calendar = xcals.get_calendar("XNYS")
    session = pd.Timestamp(dates.loc[scoring].min()).normalize()
    boundary = calendar.session_offset(session, -21)
    maturity_cutoff = calendar.session_close(boundary)
    training = horizon_mask & metadata.research_eligible & clocks.lt(cutoff) & ends.lt(maturity_cutoff)
    if scope == "security_transfer":
        held = security_transfer_mask(metadata.security_id)
        training &= ~held
        scoring &= held
    prediction = scoring & metadata.feature_eligible
    known = prediction & metadata.research_eligible
    reasons = []
    if int(training.sum()) < 100:
        reasons.append("fewer_than_100_training_rows")
    if dates.loc[training].dt.to_period("M").nunique() < 12:
        reasons.append("fewer_than_12_training_months")
    if metadata.loc[training, "security_id"].nunique() < 20:
        reasons.append("fewer_than_20_training_securities")
    if int(known.sum()) < 30:
        reasons.append("fewer_than_30_known_scoring_rows")
    if dates.loc[known].dt.to_period("M").nunique() < 3:
        reasons.append("fewer_than_3_known_scoring_months")
    if metadata.loc[known, "security_id"].nunique() < 10:
        reasons.append("fewer_than_10_known_scoring_securities")
    return InvestmentFold(horizon, score_year, scope, cutoff, maturity_cutoff,
        np.flatnonzero(training.to_numpy()), np.flatnonzero(scoring.to_numpy()),
        np.flatnonzero(prediction.to_numpy()), tuple(reasons))


def investment_diagnostics(predictions: pd.DataFrame) -> dict[str, object]:
    """Equal-date errors and fixed-decile cohorts; no compounding overlapping returns."""
    required = {"decision_id", "security_id", "decision_session", "predicted_excess_return", TARGET,
                "stock_net_return", "spy_gross_return", "qqq_gross_return", "sector_gross_return", "research_eligible"}
    if not required.issubset(predictions.columns) or predictions.decision_id.duplicated().any():
        raise DataReadinessError("investment diagnostics need one row per declared decision")
    scored = predictions.predicted_excess_return.notna()
    known = scored & predictions.research_eligible & predictions[TARGET].notna()
    verified = predictions.loc[predictions.research_eligible]
    returns = [TARGET, "stock_net_return", "spy_gross_return", "qqq_gross_return", "sector_gross_return"]
    if (verified[returns].isna().any().any() or not np.isfinite(verified[returns].to_numpy(dtype=float)).all()
            or not np.allclose(verified[TARGET], verified.stock_net_return - verified.spy_gross_return,
                rtol=0.0, atol=1e-12)):
        raise DataReadinessError("verified investment target must reconcile to complete matched holding returns")
    for column in ("predicted_excess_return", TARGET, "stock_net_return", "spy_gross_return",
                   "qqq_gross_return", "sector_gross_return"):
        values = predictions.loc[predictions[column].notna(), column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise DataReadinessError("nonfinite investment evaluation input")
    result: dict[str, object] = {
        "declared_rows": len(predictions), "scored_rows": int(scored.sum()),
        "known_scored_outcomes": int(known.sum()), "unknown_scored_outcomes": int((scored & ~known).sum()),
        "mse": None, "mae": None, "zero_excess_baseline_mse": None, "zero_excess_baseline_mae": None,
        "mean_date_rank_correlation": None, "rank_correlation_dates": 0,
        "portfolio_evaluated": False, "untouched_test_evidence": False,
        "overlap_notice": "63/252-session monthly outcomes overlap; rows and months are not independent samples.",
        "confidence_intervals": None, "annual_compounded_return": None,
    }
    if known.any():
        observed = predictions.loc[known]
        weights = date_balanced_weights(observed.decision_session)
        actual = observed[TARGET].to_numpy(dtype=float)
        error = observed.predicted_excess_return.to_numpy(dtype=float) - actual
        correlations = []
        for _, group in observed.groupby("decision_session", sort=True):
            if len(group) >= 3 and group[TARGET].nunique() > 1 and group.predicted_excess_return.nunique() > 1:
                correlations.append(float(group[TARGET].rank().corr(group.predicted_excess_return.rank())))
        result.update(mse=float(np.average(error ** 2, weights=weights)),
            mae=float(np.average(np.abs(error), weights=weights)),
            zero_excess_baseline_mse=float(np.average(actual ** 2, weights=weights)),
            zero_excess_baseline_mae=float(np.average(np.abs(actual), weights=weights)),
            mean_date_rank_correlation=float(np.mean(correlations)) if correlations else None,
            rank_correlation_dates=len(correlations))
    selected_dates: list[dict[str, object]] = []
    for day, group in predictions.loc[scored].groupby("decision_session", sort=True):
        selected = group.sort_values(["predicted_excess_return", "security_id"],
            ascending=[False, True], kind="stable").head(math.ceil(len(group) / 10))
        supported = selected.loc[selected.research_eligible & selected[TARGET].notna()]
        complete = len(supported) == len(selected)
        detail: dict[str, object] = {
            "decision_session": str(day), "scored_rows": len(group), "selected_rows": len(selected),
            "known_selected_outcomes": len(supported), "complete_selected_outcomes": complete,
            "selected_security_ids": selected.security_id.tolist(),
            "mean_stock_net_return": None, "mean_spy_gross_return": None,
            "mean_net_excess_vs_spy": None, "mean_net_excess_vs_qqq": None, "mean_net_excess_vs_sector": None,
        }
        # Refuse a complete cohort return when any selected outcome is unknown.
        if complete:
            detail.update(mean_stock_net_return=float(supported.stock_net_return.mean()),
                mean_spy_gross_return=float(supported.spy_gross_return.mean()),
                mean_net_excess_vs_spy=float(supported[TARGET].mean()),
                mean_net_excess_vs_qqq=float((supported.stock_net_return - supported.qqq_gross_return).mean()),
                mean_net_excess_vs_sector=float((supported.stock_net_return - supported.sector_gross_return).mean()))
        selected_dates.append(detail)
    result["top_decile_by_date"] = selected_dates
    result["top_decile_complete_dates"] = sum(bool(row["complete_selected_outcomes"]) for row in selected_dates)
    result["top_decile_mean_net_excess_vs_spy"] = (
        float(np.mean([row["mean_net_excess_vs_spy"] for row in selected_dates]))
        if selected_dates and all(row["complete_selected_outcomes"] for row in selected_dates) else None
    )
    return result


def compare_feature_profiles(metrics: dict[str, dict[str, object]]) -> dict[str, object]:
    """Paired feature ablations on identical masks; do not select a winning model."""
    result: dict[str, object] = {}
    for horizon in HORIZONS:
        for family in FAMILIES:
            for scope in SCOPES:
                baseline = metrics.get(f"{horizon}/technical_relationships/{family}/{scope}")
                full = metrics.get(f"{horizon}/all_features/{family}/{scope}")
                if baseline is None or full is None:
                    continue
                counts = ("declared_rows", "scored_rows", "known_scored_outcomes", "unknown_scored_outcomes")
                if any(baseline[name] != full[name] for name in counts):
                    raise DataReadinessError("investment profile comparison populations differ")
                result[f"{horizon}/{family}/{scope}"] = {
                    **{name: full[name] for name in counts},
                    "mse_reduction_vs_technical": (float(str(baseline["mse"])) - float(str(full["mse"]))
                        if baseline["mse"] is not None and full["mse"] is not None else None),
                    "mae_reduction_vs_technical": (float(str(baseline["mae"])) - float(str(full["mae"]))
                        if baseline["mae"] is not None and full["mae"] is not None else None),
                    "model_selected": False,
                }
    return result
