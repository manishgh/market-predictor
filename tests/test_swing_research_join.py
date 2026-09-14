from __future__ import annotations

import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.features.research_join import join_research_features_and_outcomes


def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame({
        "decision_id": ["one", "two"], "security_id": ["issuer-a", "issuer-b"], "ticker": ["AAA", "BBB"],
        "decision_time_utc": pd.to_datetime(["2020-01-02T23:00:00Z"] * 2),
        "momentum": [0.2, None], "news_count": [2.0, None],
        "technical_available": pd.to_datetime(["2020-01-02T21:00:00Z"] * 2),
        "news_available": pd.to_datetime(["2020-01-02T20:00:00Z", None]),
        "future_net_return_10d": [999.0, 999.0],
    })
    outcomes = features.loc[:, ["decision_id", "security_id", "ticker", "decision_time_utc"]].copy()
    outcomes["future_net_return_10d"] = [0.03, None]
    outcomes["label_eligible"] = [True, False]
    outcomes["source_admission_status"] = ["source_bounded_research", "unsupported_corporate_action"]
    return features, outcomes


def _join(features: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    return join_research_features_and_outcomes(features, outcomes,
        feature_columns=("momentum", "news_count"),
        availability_columns={"momentum": "technical_available", "news_count": "news_available"},
        retained_security_ids=frozenset(("issuer-a", "issuer-b")))


def test_exact_join_replaces_no_labels_silently_and_preserves_unavailable_decisions() -> None:
    features, outcomes = _frames()
    result = _join(features, outcomes.iloc[::-1])
    assert result.decision_id.tolist() == ["one", "two"]
    assert result.loc[0, "future_net_return_10d"] == 0.03
    assert pd.isna(result.loc[1, "future_net_return_10d"])
    assert pd.isna(result.loc[1, "news_count"])
    assert not result.loc[1, "label_eligible"]
    assert features.loc[0, "future_net_return_10d"] == 999.0


@pytest.mark.parametrize("key,value", [("security_id", "issuer-c"), ("ticker", "WRONG"),
    ("decision_time_utc", pd.Timestamp("2020-01-02T21:06:00Z"))])
def test_identity_poison_fails(key: str, value: object) -> None:
    features, outcomes = _frames()
    outcomes.loc[0, key] = value
    with pytest.raises(DataReadinessError, match="identity, ticker or cutoff"):
        _join(features, outcomes)


@pytest.mark.parametrize("poison", ["missing", "extra", "duplicate"])
def test_no_inner_join_population_loss(poison: str) -> None:
    features, outcomes = _frames()
    if poison == "missing":
        outcomes = outcomes.iloc[:1]
    elif poison == "extra":
        extra = outcomes.iloc[:1].copy()
        extra["decision_id"], extra["security_id"] = "extra", "issuer-c"
        outcomes = pd.concat((outcomes, extra), ignore_index=True)
    else:
        outcomes = pd.concat((outcomes, outcomes.iloc[:1]), ignore_index=True)
    with pytest.raises(DataReadinessError):
        _join(features, outcomes)


@pytest.mark.parametrize("poison", ["future", "missing_clock", "infinity", "text", "excluded"])
def test_predictor_availability_and_population(poison: str) -> None:
    features, outcomes = _frames()
    if poison == "future":
        features.loc[0, "news_available"] = pd.Timestamp("2020-01-03T10:00:00Z")
    elif poison == "missing_clock":
        features.loc[0, "news_available"] = pd.NaT
    elif poison == "infinity":
        features.loc[0, "momentum"] = float("inf")
    elif poison == "text":
        features["momentum"] = ["invalid", None]
    else:
        features.loc[0, "security_id"] = "excluded"
    with pytest.raises(DataReadinessError):
        _join(features, outcomes)


def test_target_cannot_be_a_predictor() -> None:
    features, outcomes = _frames()
    with pytest.raises(DataReadinessError, match="never outcome"):
        join_research_features_and_outcomes(features, outcomes, feature_columns=("future_net_return_10d",),
            availability_columns={"future_net_return_10d": "technical_available"},
            retained_security_ids=frozenset(("issuer-a", "issuer-b")))


def test_target_poison_does_not_change_predictors() -> None:
    features, outcomes = _frames()
    original = _join(features, outcomes)
    outcomes["future_net_return_10d"] = [1000.0, -1.0]
    result = _join(features, outcomes)
    pd.testing.assert_frame_equal(original[["momentum", "news_count"]], result[["momentum", "news_count"]])


@pytest.mark.parametrize("clock", [12345, "2020-01-02T20:00:00"])
def test_numeric_and_naive_feature_clocks_rejected(clock: object) -> None:
    features, outcomes = _frames()
    features["news_available"] = [clock, None]
    with pytest.raises(DataReadinessError, match="timezone-aware"):
        _join(features, outcomes)


def test_outcome_column_cannot_be_smuggled_as_availability() -> None:
    features, outcomes = _frames()
    features["label_available"] = features["technical_available"]
    with pytest.raises(DataReadinessError, match="outcomes cannot"):
        join_research_features_and_outcomes(features, outcomes, feature_columns=("momentum",),
            availability_columns={"momentum": "label_available"},
            retained_security_ids=frozenset(("issuer-a", "issuer-b")))


@pytest.mark.parametrize("name", ["rank_percentile", "target_price", "ranking_group_size"])
def test_target_derived_fields_cannot_be_predictors(name: str) -> None:
    features, outcomes = _frames()
    features[name] = 1.0
    with pytest.raises(DataReadinessError, match="never outcome"):
        join_research_features_and_outcomes(features, outcomes, feature_columns=(name,),
            availability_columns={name: "technical_available"},
            retained_security_ids=frozenset(("issuer-a", "issuer-b")))
