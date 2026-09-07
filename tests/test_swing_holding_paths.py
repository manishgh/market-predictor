from __future__ import annotations

from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.contracts import SwingDatasetConfig
from market_predictor.swing.features.panel import _add_barrier_outcomes
from market_predictor.swing.labels import add_exact_swing_labels
from market_predictor.swing.labels.barrier_and_rank import BarrierSpec, apply_triple_barrier
from market_predictor.swing.labels.holding_paths import holding_calendar, outcome_bar_lookup


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range("2024-01-02", "2024-02-01")
    stock = pd.DataFrame({
        "security_id": "issuer:one", "ticker": "ONE", "session_date_et": sessions.date,
        "bar_start_utc": [calendar.session_open(day) for day in sessions],
        "bar_end_utc": [calendar.session_close(day) for day in sessions],
        "available_at_utc": [calendar.session_close(day) + pd.Timedelta(minutes=15) for day in sessions],
        "open": 100.0 + np.arange(len(sessions)) * 0.2,
        "high": 101.0 + np.arange(len(sessions)) * 0.2,
        "low": 99.0 + np.arange(len(sessions)) * 0.2,
        "close": 100.1 + np.arange(len(sessions)) * 0.2, "volume": 1000,
    })
    decisions = stock.iloc[:2].copy()
    decisions["decision_group_id"] = decisions["session_date_et"].astype(str)
    decisions["feature_eligible"] = True
    decisions["primary_benchmark"] = "XLK"
    decisions["membership_effective_to_utc"] = pd.Timestamp("2024-01-04T05:00:00Z")
    decisions["atr_pct_14"] = 2.0 / decisions["close"]
    benchmarks = pd.concat([stock.assign(ticker=ticker) for ticker in ("SPY", "QQQ", "XLK")], ignore_index=True)
    return decisions, stock, benchmarks


def _labels(decisions: pd.DataFrame, stock: pd.DataFrame, benchmarks: pd.DataFrame) -> pd.DataFrame:
    return add_exact_swing_labels(
        decisions, benchmarks, SwingDatasetConfig(horizon_sessions=10, round_trip_cost_bps=20), outcome_bars=stock,
    )


def test_holding_outcomes_continue_after_removal_without_new_decisions() -> None:
    decisions, stock, benchmarks = _inputs()
    original = decisions.copy(deep=True)
    result = _labels(decisions, stock, benchmarks)
    assert len(result) == 2
    assert result["label_eligible"].all()
    assert result["label_window_expected"].all()
    assert result["exit_session_date_et"].tolist() == stock["session_date_et"].iloc[10:12].tolist()
    assert result["future_gross_return_10d"].iloc[0] == pytest.approx(stock["close"].iloc[10] / stock["open"].iloc[1] - 1)
    assert result["future_excess_return_10d_vs_spy"].iloc[0] == pytest.approx(-0.002)
    pd.testing.assert_frame_equal(result[original.columns], original)
    pd.testing.assert_frame_equal(decisions, original)


@pytest.mark.parametrize("offset", [1, 5, 10])
@pytest.mark.parametrize("defect", ["absent", "zero_volume", "invalid_price", "wrong_security"])
def test_incomplete_path_never_skips_to_later_bar(offset: int, defect: str) -> None:
    decisions, stock, benchmarks = _inputs()
    expected_entry = stock["session_date_et"].iloc[1]
    expected_exit = stock["session_date_et"].iloc[10]
    if defect == "absent":
        stock = stock.drop(index=offset)
    elif defect == "zero_volume":
        stock.loc[offset, "volume"] = 0
    elif defect == "invalid_price":
        stock.loc[offset, "close"] = np.nan
    else:
        stock.loc[offset, "security_id"] = "issuer:replacement"
    result = _labels(decisions, stock, benchmarks).iloc[0]
    assert result["entry_session_date_et"] == expected_entry
    assert result["exit_session_date_et"] == expected_exit
    assert result["label_window_expected"]
    assert not result["label_path_exact"]
    assert not result["label_eligible"]
    assert pd.isna(result["future_net_return_10d"])


def test_missing_spy_session_does_not_compress_calendar() -> None:
    decisions, stock, benchmarks = _inputs()
    expected = _labels(decisions, stock, benchmarks)
    benchmarks = benchmarks.loc[~(benchmarks["ticker"].eq("SPY") & benchmarks["session_date_et"].eq(stock["session_date_et"].iloc[5]))]
    result = _labels(decisions, stock, benchmarks)
    pd.testing.assert_frame_equal(result, expected)


def test_exchange_holiday_is_not_an_extra_session() -> None:
    sessions = holding_calendar(date(2024, 1, 12), date(2024, 1, 17))
    assert sessions == (date(2024, 1, 12), date(2024, 1, 16), date(2024, 1, 17))


@pytest.mark.parametrize("defect", ["duplicate", "missing_identity", "naive_time", "wrong_date", "early_availability"])
def test_invalid_source_identity_fails(defect: str) -> None:
    _, stock, _ = _inputs()
    if defect == "duplicate":
        stock = pd.concat([stock, stock.iloc[[0]]])
    elif defect == "missing_identity":
        stock.loc[0, "security_id"] = ""
    elif defect == "naive_time":
        stock["available_at_utc"] = stock["available_at_utc"].dt.tz_localize(None)
    elif defect == "wrong_date":
        stock.loc[0, "session_date_et"] = date(2024, 1, 3)
    else:
        stock.loc[0, "available_at_utc"] = stock.loc[0, "bar_start_utc"]
    with pytest.raises(DataReadinessError):
        outcome_bar_lookup(stock)


def test_future_outcome_poison_cannot_change_features_or_decision_population() -> None:
    decisions, stock, benchmarks = _inputs()
    original = _labels(decisions, stock, benchmarks)
    stock.loc[stock.index >= 2, ["open", "high", "low", "close"]] *= 1.5
    changed = _labels(decisions, stock, benchmarks)
    pd.testing.assert_frame_equal(changed[decisions.columns], original[decisions.columns])
    assert not changed["future_net_return_10d"].equals(original["future_net_return_10d"])


@pytest.mark.parametrize("missing", [False, True])
def test_fixed_and_managed_paths_share_independent_history(missing: bool) -> None:
    decisions, stock, benchmarks = _inputs()
    if missing:
        stock = stock.drop(index=5)
    labelled = _labels(decisions, stock, benchmarks)
    contract = load_strategy_contract(Path(__file__).parents[1] / "configs/edge_rebuild_strategy_contract.toml")
    result = _add_barrier_outcomes(labelled, outcome_bars=stock, benchmark_bars=benchmarks, contract=contract)
    assert len(result) == len(decisions)
    if missing:
        assert result["barrier_net_return"].isna().all()
    else:
        assert result["barrier_holding_sessions"].eq(10).all()
        np.testing.assert_allclose(result["barrier_net_return"], result["future_net_return_10d"])
        assert result["barrier_exit_session_date_et"].tolist() == result["exit_session_date_et"].tolist()


def test_label_availability_includes_late_intermediate_observation() -> None:
    decisions, stock, benchmarks = _inputs()
    late = stock["available_at_utc"].iloc[15]
    stock.loc[4, "available_at_utc"] = late
    result = _labels(decisions, stock, benchmarks)
    assert result["label_available_at_utc"].eq(late).all()


@pytest.mark.parametrize("field", ["bar_start_utc", "bar_end_utc"])
def test_partial_session_bar_cannot_be_labeled_as_full_daily_bar(field: str) -> None:
    _, stock, _ = _inputs()
    stock.loc[1, field] += pd.Timedelta(minutes=30)
    with pytest.raises(DataReadinessError, match="session/availability mismatch"):
        outcome_bar_lookup(stock)


def test_early_close_uses_exchange_schedule() -> None:
    calendar = xcals.get_calendar("XNYS")
    _, stock, _ = _inputs()
    row = stock.iloc[[0]].copy()
    day = date(2024, 11, 29)
    row["session_date_et"] = day
    row["bar_start_utc"] = calendar.session_open(day)
    row["bar_end_utc"] = calendar.session_close(day)
    row["available_at_utc"] = calendar.session_close(day) + pd.Timedelta(minutes=15)
    assert outcome_bar_lookup(row)["outcome_observation_valid"].all()
    row["bar_end_utc"] += pd.Timedelta(hours=3)
    row["available_at_utc"] += pd.Timedelta(hours=3)
    with pytest.raises(DataReadinessError):
        outcome_bar_lookup(row)


def test_direct_barrier_call_does_not_skip_missing_entry() -> None:
    decisions, stock, _ = _inputs()
    bars = stock.drop(index=1).rename(columns={"session_date_et": "session"})
    result = apply_triple_barrier(
        bars, pd.DataFrame({"session": decisions["session_date_et"], "atr": 2.0}),
        spec=BarrierSpec(target_atr_multiple=3, stop_atr_multiple=1.5, horizon_sessions=10),
    )
    assert pd.isna(result.iloc[0]["exit_price"])
    assert pd.notna(result.iloc[1]["exit_price"])


def test_timestamp_barrier_keys_match_date_keys() -> None:
    decisions, stock, _ = _inputs()
    bars = stock.rename(columns={"session_date_et": "session"})
    entries = pd.DataFrame({"session": decisions["session_date_et"], "atr": 2.0})
    spec = BarrierSpec(target_atr_multiple=3, stop_atr_multiple=1.5, horizon_sessions=10)
    expected = apply_triple_barrier(bars, entries, spec=spec)
    bars["session"] = pd.to_datetime(bars["session"])
    entries["session"] = pd.to_datetime(entries["session"])
    observed = apply_triple_barrier(bars, entries, spec=spec)
    assert observed["holding_sessions"].eq(10).all()
    np.testing.assert_allclose(observed["exit_price"], expected["exit_price"])
    assert observed["exit_session"].tolist() == expected["exit_session"].tolist()
