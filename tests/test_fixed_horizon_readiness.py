from __future__ import annotations

from datetime import date
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.labels.fixed_horizon_readiness import fixed_horizon_readiness


def inputs() -> tuple[pd.DataFrame, dict[str, Any]]:
    days = [date(2023, 1, 3), date(2023, 1, 4), date(2023, 1, 5)]
    calendar = xcals.get_calendar("XNYS")
    maturity = {day: calendar.session_close(calendar.sessions_window(pd.Timestamp(day), 11)[-1]) for day in days}
    frame = pd.DataFrame({"decision_id": ["a", "b", "c"], "security_id": ["issuer"] * 3,
        "ticker": ["ABC"] * 3, "sector": ["technology"] * 3, "session_date_et": days,
        "feature_eligible": [True] * 3, "daily_bar_count": [250] * 3,
        "feature": [1.0, 2.0, np.nan], "stock_source_admitted": [True] * 3,
        "stock_missing_reasons": ["[]"] * 3, "stock_component_id": ["stock-a", "stock-b", "stock-c"],
        "fixed_horizon_gross_return": [-0.10, 0.12, 0.01], "fixed_comparisons_complete": [True] * 3,
        "training_eligible": [False] * 3, "production_eligible": [False] * 3})
    frame["decision_time_utc"] = swing_prediction_cutoffs(frame.session_date_et)
    frame["feature_clock"] = frame.decision_time_utc
    frame["research_label_mature_at"] = frame.session_date_et.map(maturity)
    frame["fixed_horizon_net_return"] = frame.fixed_horizon_gross_return - 0.002
    for role in ("spy", "qqq", "sector"):
        frame[f"{role}_horizon_gross_return"] = 0.02
        frame[f"{role}_fixed_horizon_excess_return"] = frame.fixed_horizon_net_return - 0.02
        frame[f"{role}_component_id"] = [f"{role}-{i}" for i in range(3)]
        frame[f"{role}_missing_reasons"] = "[]"
    kwargs = dict(model_columns=("feature",), availability_columns={"feature": "feature_clock"},
        maturity_by_session=maturity, fit_end=pd.Timestamp("2023-02-01T21:00:00Z"),
        minimum_warmup=250, round_trip_cost=0.002)
    return frame, kwargs


def test_negative_returns_and_missing_features_preserve_separate_masks() -> None:
    frame, kwargs = inputs()
    before = frame.copy(deep=True)
    state = fixed_horizon_readiness(frame, **kwargs)
    assert state.feature_eligible.tolist() == [True, True, True]
    assert state.fixed_horizon_supervision_available.tolist() == [True, True, True]
    assert state.complete_case_supervision.tolist() == [True, True, False]
    pd.testing.assert_frame_equal(frame, before)
    assert not frame.training_eligible.any()


def test_unknown_outcomes_cannot_change_decision_input_eligibility() -> None:
    frame, kwargs = inputs()
    original = fixed_horizon_readiness(frame, **kwargs)
    frame.loc[0, "stock_source_admitted"] = False
    frame.loc[0, "fixed_comparisons_complete"] = False
    frame.loc[0, "stock_missing_reasons"] = '["source_unavailable"]'
    frame.loc[0, "stock_component_id"] = None
    frame.loc[0, "research_label_mature_at"] = pd.NaT
    frame.loc[0, ["fixed_horizon_gross_return", "fixed_horizon_net_return",
        *(f"{role}_fixed_horizon_excess_return" for role in ("spy", "qqq", "sector"))]] = np.nan
    state = fixed_horizon_readiness(frame, **kwargs)
    pd.testing.assert_series_equal(state.decision_inputs_complete, original.decision_inputs_complete)
    assert not state.fixed_horizon_supervision_available.iloc[0]


def test_unknown_benchmark_cannot_be_counted_as_complete_supervision() -> None:
    frame, kwargs = inputs()
    frame.loc[0, "qqq_horizon_gross_return"] = np.nan
    frame.loc[0, "qqq_fixed_horizon_excess_return"] = np.nan
    frame.loc[0, "qqq_missing_reasons"] = '["unavailable"]'
    frame.loc[0, "qqq_component_id"] = None
    frame.loc[0, "fixed_comparisons_complete"] = False
    state = fixed_horizon_readiness(frame, **kwargs)
    assert state.decision_inputs_complete.iloc[0]
    assert not state.fixed_horizon_supervision_available.iloc[0]


@pytest.mark.parametrize(("column", "value", "reason"), [
    ("feature_clock", pd.Timestamp("2023-01-04T23:00:00Z"), "feature clock"),
    ("feature_clock", pd.NaT, "feature clock"),
    ("feature", np.inf, "finite numeric"),
    ("feature", "bad", "finite numeric"),
    ("daily_bar_count", 249, "under-warm"),
    ("daily_bar_count", 250.5, "warm-up"),
    ("feature_eligible", "true", "booleans"),
    ("research_label_mature_at", pd.Timestamp("2023-01-20T21:00:00Z"), "maturity"),
    ("stock_missing_reasons", '["gap"]', "stock admission"),
    ("stock_missing_reasons", "null", "reason arrays"),
    ("stock_missing_reasons", "[", "reason arrays"),
    ("stock_component_id", None, "bound component"),
    ("spy_component_id", None, "bound component"),
    ("fixed_horizon_net_return", -0.104, "cost arithmetic"),
    ("spy_fixed_horizon_excess_return", 0.5, "excess arithmetic"),
    ("sector_horizon_gross_return", np.nan, "benchmark availability"),
    ("fixed_comparisons_complete", False, "comparison flag"),
    ("training_eligible", True, "permission"),
    ("production_eligible", True, "permission"),
    ("decision_time_utc", pd.Timestamp("2023-01-03T22:00:00Z"), "canonical"),
    ("session_date_et", date(2025, 7, 1), "initial-fit"),
])
def test_poisoned_evidence_is_rejected(column: str, value: Any, reason: str) -> None:
    frame, kwargs = inputs()
    frame[column] = frame[column].astype(object)
    frame.loc[0, column] = value
    with pytest.raises(DataReadinessError, match=reason):
        fixed_horizon_readiness(frame, **kwargs)


@pytest.mark.parametrize("clock", ["feature_clock", "research_label_mature_at", "decision_time_utc"])
def test_naive_clocks_rejected(clock: str) -> None:
    frame, kwargs = inputs()
    frame[clock] = frame[clock].dt.tz_localize(None)
    with pytest.raises(DataReadinessError, match="explicit UTC"):
        fixed_horizon_readiness(frame, **kwargs)


def test_label_must_mature_within_fit_boundary() -> None:
    frame, kwargs = inputs()
    kwargs["fit_end"] = pd.Timestamp("2023-01-10T21:00:00Z")
    with pytest.raises(DataReadinessError, match="fit boundary"):
        fixed_horizon_readiness(frame, **kwargs)


def test_outcomes_cannot_be_declared_as_model_features() -> None:
    frame, kwargs = inputs()
    kwargs.update(model_columns=("spy_fixed_horizon_excess_return",),
        availability_columns={"spy_fixed_horizon_excess_return": "feature_clock"})
    with pytest.raises(DataReadinessError, match="outcome"):
        fixed_horizon_readiness(frame, **kwargs)


def test_duplicate_security_session_rejected() -> None:
    frame, kwargs = inputs()
    frame.loc[1, "session_date_et"] = frame.loc[0, "session_date_et"]
    with pytest.raises(DataReadinessError, match="duplicate"):
        fixed_horizon_readiness(frame, **kwargs)


def test_equal_instant_mixed_offset_object_clocks_rejected() -> None:
    frame, kwargs = inputs()
    frame["feature_clock"] = frame.feature_clock.astype(object)
    frame.loc[0, "feature_clock"] = frame.loc[1, "feature_clock"]
    frame.loc[1, "feature_clock"] = pd.Timestamp(frame.loc[0, "feature_clock"]).tz_convert("Europe/Berlin")
    with pytest.raises(DataReadinessError, match="explicit UTC"):
        fixed_horizon_readiness(frame, **kwargs)


def test_benchmark_only_outcome_cannot_cross_fit_boundary() -> None:
    frame, kwargs = inputs()
    frame = frame.iloc[[2]].copy()
    frame["stock_source_admitted"] = False
    frame["fixed_comparisons_complete"] = False
    frame["stock_missing_reasons"] = '["terminal_immature"]'
    frame["stock_component_id"] = None
    frame["research_label_mature_at"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    for name in ("fixed_horizon_gross_return", "fixed_horizon_net_return",
            *(f"{role}_fixed_horizon_excess_return" for role in ("spy", "qqq", "sector"))):
        frame[name] = np.nan
    kwargs["fit_end"] = pd.Timestamp("2023-01-19T21:00:00Z")
    with pytest.raises(DataReadinessError, match="benchmark.*fit boundary"):
        fixed_horizon_readiness(frame, **kwargs)
