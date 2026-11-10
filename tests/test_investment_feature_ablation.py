"""Independent four-view loss checks using unequal dates and unknown outcomes."""
from datetime import date

import numpy as np
import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.investment.feature_ablation import ABLATION_PROFILES, ablation_profiles, factorial_diagnostics
from market_predictor.investment.fundamentals import FEATURE_NAMES
from market_predictor.investment.training import TARGET


def frames() -> dict[str, pd.DataFrame]:
    actual = np.array([0.2, -0.1, np.nan, 0.3, 0.1, 0.05, 0.15, np.nan])
    prediction = np.array([0.1, 0.2, 0.3, 0.1, 0.1, 0.4, 0.5, np.nan])
    baseline = pd.DataFrame({"decision_id": [f"decision-{i}" for i in range(8)],
        "security_id": [f"issuer-{i}" for i in range(8)], "decision_session":
            [date(2022, 1, 31)] * 3 + [date(2022, 2, 28)] * 3 + [date(2023, 1, 31)] * 2,
        "horizon_sessions": 252, "sector": ["A", "B"] * 4, "feature_eligible": np.isfinite(prediction),
        "research_eligible": np.isfinite(actual), TARGET: actual, "stock_net_return": actual + 0.05,
        "spy_gross_return": np.where(np.isfinite(actual), 0.05, np.nan),
        "qqq_gross_return": np.where(np.isfinite(actual), 0.07, np.nan),
        "sector_gross_return": np.where(np.isfinite(actual), 0.06, np.nan), "predicted_excess_return": prediction,
        "news_finite_count": [0, 1, 85, 85, 1, 0, 0, 0], "financial_finite_count": [0, 1, 6, 6, 1, 0, 0, 0],
        "news_coverage_bin": ["0", "1-84", "85", "85", "1-84", "0", "0", "0"],
        "financial_coverage_bin": ["0", "1-5", "6", "6", "1-5", "0", "0", "0"]})
    result = {name: baseline.copy(deep=True) for name in ABLATION_PROFILES}
    result["technical_news"]["predicted_excess_return"] += [0.03, -0.01, 0.2, 0.04, 0.05, 0.0, -0.2, 0]
    result["technical_financial"]["predicted_excess_return"] += [-0.1, 0.02, 0.1, 0.01, -0.01, 0.05, -0.1, 0]
    result["all_features"]["predicted_excess_return"] += [0.04, 0.03, 0.5, -0.02, 0.02, 0.03, -0.15, 0]
    return result


def test_profiles_keep_exact_order_and_do_not_mutate_original_contract() -> None:
    complete = tuple(f"column-{i}" for i in range(209)) + FEATURE_NAMES
    original = {"technical_relationships": complete[:124], "all_features": complete}
    before = original.copy()
    projected = ablation_profiles(original)
    assert tuple(projected) == ABLATION_PROFILES
    assert [len(value) for value in projected.values()] == [124, 209, 130, 215]
    assert projected["technical_news"] == complete[:209]
    assert projected["technical_financial"] == complete[:124] + complete[209:]
    assert set(projected["technical_news"]) & set(projected["technical_financial"]) == set(complete[:124])
    assert original == before


@pytest.mark.parametrize("poison", ["prefix", "financial_order", "duplicate", "extra_profile"])
def test_profile_contract_poison_refuses(poison: str) -> None:
    complete = tuple(f"column-{i}" for i in range(209)) + FEATURE_NAMES
    original = {"technical_relationships": complete[:124], "all_features": complete}
    if poison == "prefix":
        original["technical_relationships"] = tuple(reversed(complete[:124]))
    elif poison == "financial_order":
        original["all_features"] = complete[:209] + tuple(reversed(FEATURE_NAMES))
    elif poison == "duplicate":
        original["all_features"] = complete[:208] + (complete[207],) + FEATURE_NAMES
    else:
        original["unexpected"] = complete
    with pytest.raises(DataReadinessError):
        ablation_profiles(original)


def test_four_effects_and_interaction_match_independent_global_date_losses() -> None:
    inputs = frames()
    originals = {name: frame.copy(deep=True) for name, frame in inputs.items()}
    result = factorial_diagnostics(inputs)
    known = inputs["technical_relationships"].research_eligible
    y = inputs["technical_relationships"].loc[known, TARGET].to_numpy()
    # Three dates have two, three and one known rows respectively.
    weights = np.array([1 / 6, 1 / 6, 1 / 9, 1 / 9, 1 / 9, 1 / 3])
    for metric in ("mse", "mae"):
        errors = {name: frame.loc[known, "predicted_excess_return"].to_numpy() - y for name, frame in inputs.items()}
        losses = {name: values ** 2 if metric == "mse" else np.abs(values) for name, values in errors.items()}
        means = {name: float(np.dot(weights, values)) for name, values in losses.items()}
        for effect in result["conditional_effects"].values():
            expected = means[effect["candidate_profile"]] - means[effect["reference_profile"]]
            assert effect["error_decomposition"][f"{metric}_difference"] == pytest.approx(expected)
            assert sum(row[f"{metric}_delta_contribution"] for row in effect["strata"]["year"]) == pytest.approx(expected)
        interaction = losses["all_features"] - losses["technical_news"] - losses["technical_financial"] + losses["technical_relationships"]
        assert result["interaction"][f"{metric}_difference"] == pytest.approx(float(np.dot(weights, interaction)))
        years = {row["year"]: row for row in result["interaction"]["year_contributions"]}
        assert years["2022"][f"{metric}_interaction_contribution"] == pytest.approx(float(np.dot(weights[:-1], interaction[:-1])))
        assert years["2023"][f"{metric}_interaction_contribution"] == pytest.approx(float(weights[-1] * interaction[-1]))
    assert result["known_paired_rows"] == 6 and result["unknown_scored_rows"] == 1
    assert result["scored_rows"] == 7 and result["declared_rows"] == 8
    for name, frame in inputs.items():
        pd.testing.assert_frame_equal(frame, originals[name])


@pytest.mark.parametrize("poison", ["profile", "outcome", "mask", "identity", "coverage"])
def test_four_way_matching_refuses_changed_population(poison: str) -> None:
    inputs = frames()
    changed = inputs["technical_financial"]
    if poison == "profile":
        del inputs["all_features"]
    elif poison == "outcome":
        changed.loc[0, TARGET] += 0.01
    elif poison == "mask":
        changed.loc[0, "research_eligible"] = False
    elif poison == "identity":
        changed.loc[0, "security_id"] = "other-security"
    else:
        changed.loc[0, "news_finite_count"] = 5
    with pytest.raises(DataReadinessError):
        factorial_diagnostics(inputs)


def test_unknown_outcomes_are_not_converted_to_zero_or_removed() -> None:
    inputs = frames()
    for frame in inputs.values():
        frame["research_eligible"] = False
        for name in (TARGET, "stock_net_return", "spy_gross_return", "qqq_gross_return", "sector_gross_return"):
            frame[name] = np.nan
    result = factorial_diagnostics(inputs)
    assert result["interaction"] is None
    assert result["known_paired_rows"] == 0 and result["unknown_scored_rows"] == result["scored_rows"] == 7
    assert result["declared_rows"] == 8
    assert all(effect["error_decomposition"] is None for effect in result["conditional_effects"].values())
