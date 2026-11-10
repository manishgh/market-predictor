"""Matched saved-forecast explanations; no calibration fitting or model selection."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from market_predictor.core.errors import DataReadinessError
from market_predictor.investment.training import TARGET, investment_diagnostics
from market_predictor.swing.training.return_validation import date_balanced_weights

PREDICTION = "predicted_excess_return"
CONTEXT = ("decision_id", "security_id", "decision_session", "horizon_sessions", "sector", "feature_eligible",
    "research_eligible", TARGET, "stock_net_return", "spy_gross_return", "qqq_gross_return", "sector_gross_return")
COVERAGE = ("news_finite_count", "financial_finite_count", "news_coverage_bin", "financial_coverage_bin")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def feature_coverage(features: pd.DataFrame, columns: tuple[str, ...]) -> pd.DataFrame:
    """Finite zero is observed; never infer feature coverage from its value magnitude."""
    _require(len(columns) == 215 and len(set(columns)) == 215 and set(columns).issubset(features),
        "investment diagnostics require the original 215-feature view")
    result = pd.DataFrame(index=features.index)
    for kind, names, complete in (("news", columns[124:209], 85), ("financial", columns[209:], 6)):
        count = np.isfinite(features.loc[:, list(names)].to_numpy(dtype=float)).sum(axis=1)
        result[f"{kind}_finite_count"] = count
        result[f"{kind}_coverage_bin"] = np.where(count == 0, "0", np.where(count == complete, str(complete), f"1-{complete - 1}"))
    return result


def _known(frame: pd.DataFrame) -> pd.DataFrame:
    _require(set((*CONTEXT, PREDICTION)).issubset(frame) and not frame.decision_id.duplicated().any(),
        "investment diagnostic predictions repeat or omit required identities")
    _require(len(frame.horizon_sessions.unique()) == 1, "investment diagnostics must separate horizons")
    for name in (PREDICTION, TARGET):
        values = frame[name].dropna().to_numpy(dtype=float)
        _require(np.isfinite(values).all(), "investment diagnostics received nonfinite forecasts or targets")
    scored = frame[PREDICTION].notna()
    _require(scored.equals(frame.feature_eligible), "saved investment scoring mask differs from feature eligibility")
    return frame.loc[scored & frame.research_eligible & frame[TARGET].notna()].copy()


def _auc(frame: pd.DataFrame, weights: np.ndarray, score_column: str = PREDICTION) -> dict[str, Any]:
    actual = frame[TARGET].to_numpy(dtype=float) > 0
    predicted = frame[score_column].to_numpy(dtype=float)
    positive, negative = int(actual.sum()), int((~actual).sum())
    auc = float(roc_auc_score(actual, predicted, sample_weight=weights)) if positive and negative else None
    dates: list[dict[str, Any]] = []
    for day, group in frame.groupby("decision_session", sort=True):
        labels = group[TARGET].to_numpy(dtype=float) > 0
        both = bool(labels.any() and (~labels).any())
        dates.append({"decision_session": str(day), "known_rows": len(group), "positive_rows": int(labels.sum()),
            "negative_rows": int((~labels).sum()), "auc": float(roc_auc_score(labels, group[score_column])) if both else None,
            "missing_reason": None if both else "one_outcome_class"})
    valid = [row["auc"] for row in dates if row["auc"] is not None]
    return {"label": "strictly_positive_net_excess_vs_spy", "positive_rows": positive, "negative_rows": negative,
        "pooled_equal_date_weight_auc": auc, "pooled_missing_reason": None if auc is not None else "fewer_than_two_outcome_classes",
        "mean_within_date_auc": float(np.mean(valid)) if valid else None,
        "eligible_dates": len(valid), "one_class_dates": len(dates) - len(valid), "known_dates": len(dates), "per_date": dates}


def prediction_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    """Re-use canonical cohort metrics; additions explain their exact known-row mask."""
    canonical = investment_diagnostics(frame)
    observed = _known(frame)
    weights = date_balanced_weights(observed.decision_session) if not observed.empty else np.empty(0, dtype=float)
    auc = _auc(observed, weights)
    scored_dates = int(frame.loc[frame[PREDICTION].notna(), "decision_session"].nunique())
    auc.update({"declared_dates": int(frame.decision_session.nunique()), "scored_dates": scored_dates,
        "scored_dates_without_known_outcomes": scored_dates - int(observed.decision_session.nunique())})
    result: dict[str, Any] = {"canonical_metrics": canonical, "moments": None, "auc": auc,
        "diagnostic_slope_is_not_a_fitted_calibrator": True, "predictions_modified": False}
    if observed.empty:
        return result
    actual, predicted = observed[TARGET].to_numpy(dtype=float), observed[PREDICTION].to_numpy(dtype=float)
    mean_y, mean_p = float(np.average(actual, weights=weights)), float(np.average(predicted, weights=weights))
    var_y = float(np.average((actual - mean_y) ** 2, weights=weights))
    var_p = float(np.average((predicted - mean_p) ** 2, weights=weights))
    cov = float(np.average((actual - mean_y) * (predicted - mean_p), weights=weights))
    error = predicted - actual
    mean_error = mean_p - mean_y
    dates = observed.assign(_p=predicted, _y=actual).groupby("decision_session", sort=False)
    date_p = dates["_p"].transform("mean").to_numpy(dtype=float)
    date_y = dates["_y"].transform("mean").to_numpy(dtype=float)
    date_bias = float(np.average((date_p - date_y) ** 2, weights=weights))
    within = float(np.average(((predicted - date_p) - (actual - date_y)) ** 2, weights=weights))
    mse = float(np.average(error ** 2, weights=weights))
    p2 = float(np.average(predicted ** 2, weights=weights))
    py = float(np.average(predicted * actual, weights=weights))
    slope = cov / var_p if var_p > 0 else None
    result["moments"] = {"known_rows": len(observed), "known_dates": observed.decision_session.nunique(),
        "mean_prediction": mean_p, "mean_target": mean_y, "prediction_std": math.sqrt(var_p), "target_std": math.sqrt(var_y),
        "covariance": cov, "mean_error": mean_error, "squared_bias": mean_error ** 2,
        "error_variance": float(np.average((error - mean_error) ** 2, weights=weights)),
        "date_mean_bias_mse": date_bias, "within_date_centered_mse": within,
        "date_mean_plus_within_residual": mse - date_bias - within,
        "prediction_second_moment": p2, "prediction_target_cross_moment": py,
        "mse_minus_zero_baseline": mse - float(np.average(actual ** 2, weights=weights)),
        "zero_baseline_identity_residual": mse - float(np.average(actual ** 2, weights=weights)) - p2 + 2 * py,
        "diagnostic_target_on_prediction_slope": slope,
        "diagnostic_target_on_prediction_intercept": None if slope is None else mean_y - slope * mean_p}
    return result


def paired_diagnostics(technical: pd.DataFrame, full: pd.DataFrame) -> dict[str, Any]:
    """Keep global date weights through every stratum and tail contribution."""
    left, right = technical.sort_values("decision_id").reset_index(drop=True), full.sort_values("decision_id").reset_index(drop=True)
    _known(left)
    _known(right)
    _require(set(COVERAGE).issubset(left) and set(COVERAGE).issubset(right), "paired investment diagnostics omit feature coverage")
    try:
        pd.testing.assert_frame_equal(left.loc[:, list((*CONTEXT, *COVERAGE))], right.loc[:, list((*CONTEXT, *COVERAGE))],
            check_exact=True)
    except AssertionError as error:
        raise DataReadinessError("paired investment identities, outcomes, coverage or eligibility differ") from error
    _require(left[PREDICTION].notna().equals(right[PREDICTION].notna()), "paired investment forecast masks differ")
    mask = left[PREDICTION].notna() & left.research_eligible & left[TARGET].notna()
    observed = left.loc[mask].copy()
    result: dict[str, Any] = {"declared_rows": len(left), "scored_rows": int(left[PREDICTION].notna().sum()),
        "known_paired_rows": len(observed), "unknown_scored_rows": int((left[PREDICTION].notna() & ~mask).sum()),
        "paired_identity": None, "strata": {}, "tail_concentration": [], "examples_removed": 0,
        "global_date_weights_retained": True, "model_selected": False}
    if observed.empty:
        return result
    weights = date_balanced_weights(observed.decision_session)
    total = float(weights.sum())
    actual = observed[TARGET].to_numpy(dtype=float)
    baseline = observed[PREDICTION].to_numpy(dtype=float)
    changed = right.loc[mask, PREDICTION].to_numpy(dtype=float)
    delta = changed - baseline
    lhs = (changed - actual) ** 2 - (baseline - actual) ** 2
    squared_delta = delta ** 2
    interaction = 2 * delta * (baseline - actual)
    absolute_delta = np.abs(changed - actual) - np.abs(baseline - actual)
    observed["year"] = observed.decision_session.map(lambda day: str(day)[:4])
    left["year"] = left.decision_session.map(lambda day: str(day)[:4])
    left["_scored"], left["_known"] = left[PREDICTION].notna(), mask
    observed["_w"], observed["_delta_mse"] = weights, lhs
    observed["_delta_mae"], observed["_square"], observed["_interaction"] = absolute_delta, squared_delta, interaction
    difference = float(np.dot(weights, lhs) / total)
    result["paired_identity"] = {"full_minus_technical_mse": difference,
        "full_minus_technical_mae": float(np.dot(weights, absolute_delta) / total),
        "squared_prediction_delta": float(np.dot(weights, squared_delta) / total),
        "prediction_delta_error_interaction": float(np.dot(weights, interaction) / total),
        "identity_residual": float(np.dot(weights, lhs - squared_delta - interaction) / total)}

    def contribution(group: pd.DataFrame) -> dict[str, Any]:
        w = group._w.to_numpy(dtype=float)
        return {"known_rows": len(group), "global_weight_fraction": float(w.sum() / total),
            "mse_delta_contribution": float(np.dot(w, group._delta_mse) / total),
            "mae_delta_contribution": float(np.dot(w, group._delta_mae) / total),
            "squared_delta_contribution": float(np.dot(w, group._square) / total),
            "interaction_contribution": float(np.dot(w, group._interaction) / total)}

    for name in ("year", "decision_session", "sector", "news_coverage_bin", "financial_coverage_bin"):
        summaries = []
        for key, population in left.groupby(name, sort=True, dropna=False):
            group = observed.loc[observed.index.intersection(population.index)]
            summaries.append({"value": str(key), "declared_rows": len(population),
                "scored_rows": int(population._scored.sum()),
                "unknown_scored_rows": int((population._scored & ~population._known).sum()), **contribution(group)})
        _require(math.isclose(sum(row["mse_delta_contribution"] for row in summaries), difference, rel_tol=1e-10, abs_tol=1e-12),
            "investment subgroup error contributions do not reconcile")
        result["strata"][name] = summaries
    ordered = observed.assign(_absolute_delta=np.abs(lhs)).sort_values(["_absolute_delta", "decision_id"], ascending=[False, True])
    for fraction in (0.01, 0.05, 0.10):
        group = ordered.head(math.ceil(len(ordered) * fraction))
        result["tail_concentration"].append({"fraction_by_known_row_count": fraction,
            "selection": "absolute_paired_squared_error_difference_descriptive_only", **contribution(group)})
    return result
