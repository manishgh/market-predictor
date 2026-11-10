"""Matched news and financial feature comparisons on retained investment forecasts."""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.investment.diagnostics import paired_diagnostics
from market_predictor.investment.fundamentals import FEATURE_NAMES

ABLATION_PROFILES = ("technical_relationships", "technical_news", "technical_financial", "all_features")
NEW_PROFILES = ABLATION_PROFILES[1:3]
CONDITIONAL_COMPARISONS = {
    "news_given_technical": ("technical_relationships", "technical_news"),
    "financial_given_technical": ("technical_relationships", "technical_financial"),
    "news_given_financial": ("technical_financial", "all_features"),
    "financial_given_news": ("technical_news", "all_features"),
}


def ablation_profiles(model_columns: dict[str, tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    """Project ordered views without changing the admitted dataset's original contract."""
    if set(model_columns) != {ABLATION_PROFILES[0], ABLATION_PROFILES[-1]}:
        raise DataReadinessError("feature comparison requires the two original admitted profiles")
    technical, complete = model_columns[ABLATION_PROFILES[0]], model_columns[ABLATION_PROFILES[-1]]
    if (len(technical) != 124 or len(complete) != 215 or len(set(complete)) != 215
            or complete[:124] != technical or complete[209:] != FEATURE_NAMES):
        raise DataReadinessError("feature comparison original column order or financial contract differs")
    return dict(zip(ABLATION_PROFILES, (technical, complete[:209], technical + FEATURE_NAMES, complete), strict=True))


def factorial_diagnostics(frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    """Report four conditional effects and interaction on one exact four-way population.

    Positive loss differences mean worse forecasts. Existing paired diagnostics own
    row identity, outcome masks, weights and error arithmetic. Year contributions use
    the original aggregate date weights, not independently normalized subgroup weights.
    """
    if set(frames) != set(ABLATION_PROFILES):
        raise DataReadinessError("feature comparison requires exactly four prediction profiles")
    effects: dict[str, dict[str, Any]] = {}
    for name, (reference, candidate) in CONDITIONAL_COMPARISONS.items():
        comparison = paired_diagnostics(frames[reference], frames[candidate])
        identity = comparison.pop("paired_identity")
        comparison["error_decomposition"] = None if identity is None else {
            "mse_difference": identity["full_minus_technical_mse"],
            "mae_difference": identity["full_minus_technical_mae"],
            "squared_prediction_difference": identity["squared_prediction_delta"],
            "prediction_difference_error_interaction": identity["prediction_delta_error_interaction"],
            "identity_residual": identity["identity_residual"],
        }
        effects[name] = {"reference_profile": reference, "candidate_profile": candidate, **comparison}
    first = effects["news_given_technical"]
    counts = ("declared_rows", "scored_rows", "known_paired_rows", "unknown_scored_rows")
    if any(effect[name] != first[name] for effect in effects.values() for name in counts):
        raise DataReadinessError("four-way feature comparison populations differ")
    result: dict[str, Any] = {**{name: first[name] for name in counts}, "conditional_effects": effects,
        "interaction": None, "positive_loss_difference_means": "worse_prediction_error",
        "interpretation": "feature_block_effects_under_fixed_training_pipeline_not_causal_market_effects",
        "examples_removed": 0, "model_selected": False}
    if not first["known_paired_rows"]:
        return result
    interaction: dict[str, Any] = {"year_contributions": []}
    for metric in ("mse", "mae"):
        name = f"{metric}_difference"
        news_difference = (effects["news_given_financial"]["error_decomposition"][name]
            - effects["news_given_technical"]["error_decomposition"][name])
        financial_difference = (effects["financial_given_news"]["error_decomposition"][name]
            - effects["financial_given_technical"]["error_decomposition"][name])
        if not math.isclose(news_difference, financial_difference, rel_tol=1e-10, abs_tol=1e-12):
            raise DataReadinessError("feature interaction does not reconcile across both conditional effects")
        interaction[name] = news_difference
        interaction[f"{metric}_identity_residual"] = news_difference - financial_difference
    years = {name: {row["value"]: row for row in effect["strata"]["year"]} for name, effect in effects.items()}
    if any(set(rows) != set(years["news_given_technical"]) for rows in years.values()):
        raise DataReadinessError("feature interaction year populations differ")
    for year in sorted(years["news_given_technical"]):
        baseline = years["news_given_technical"][year]
        row = {"year": year, **{name: baseline[name] for name in
            ("declared_rows", "scored_rows", "known_rows", "unknown_scored_rows", "global_weight_fraction")}}
        for metric in ("mse", "mae"):
            field = f"{metric}_delta_contribution"
            row[f"{metric}_interaction_contribution"] = (years["news_given_financial"][year][field]
                - years["news_given_technical"][year][field])
        interaction["year_contributions"].append(row)
    for metric in ("mse", "mae"):
        total = sum(row[f"{metric}_interaction_contribution"] for row in interaction["year_contributions"])
        if not math.isclose(total, interaction[f"{metric}_difference"], rel_tol=1e-10, abs_tol=1e-12):
            raise DataReadinessError("feature interaction year contributions do not reconcile")
    result["interaction"] = interaction
    return result
