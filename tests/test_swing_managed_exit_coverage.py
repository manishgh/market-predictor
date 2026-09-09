from __future__ import annotations

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import market_predictor.modeling.label_outcomes as label_outcomes
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.contracts import SwingDatasetConfig
from market_predictor.swing.labels import add_exact_swing_labels
from market_predictor.swing.labels.barrier_and_rank import (
    BARRIER_COLUMNS,
    BarrierSpec,
    apply_triple_barrier,
)

SPEC = BarrierSpec(target_atr_multiple=3.0, stop_atr_multiple=1.5, horizon_sessions=10)
OHLC = ["open", "high", "low", "close"]
STOP = (100.0, 101.0, 98.0, 99.0)
TARGET = (100.0, 104.0, 99.0, 103.0)
INVALID_OHLC = [
    (np.nan, 101.0, 99.0, 100.0),
    (100.0, np.inf, 99.0, 100.0),
    (100.0, 101.0, -np.inf, 100.0),
    (100.0, 101.0, 99.0, np.nan),
    (100.0, 101.0, 0.0, 100.0),
    (-1.0, 101.0, 99.0, 100.0),
    (102.0, 101.0, 99.0, 100.0),
    (100.0, 101.0, 99.0, 102.0),
    (98.0, 101.0, 99.0, 100.0),
    (100.0, 101.0, 99.0, 98.0),
    (100.0, 99.0, 101.0, 100.0),
    (100.0, 101.0, 99.0, "invalid"),
]


def _bars() -> pd.DataFrame:
    sessions = xcals.get_calendar("XNYS").sessions_in_range("2024-01-02", "2024-01-19")
    return pd.DataFrame({"session": sessions.strftime("%Y-%m-%d"), **dict.fromkeys(OHLC, 100.0)})


def _resolve(bars: pd.DataFrame, decision: str = "2024-01-02") -> pd.DataFrame:
    return apply_triple_barrier(
        bars, pd.DataFrame({"session": [decision], "atr": [1.0]}), spec=SPEC,
    )


def _assert_unresolved(result: pd.DataFrame) -> None:
    assert len(result) == 1
    assert result.loc[:, list(BARRIER_COLUMNS)].isna().all().all()


@pytest.mark.parametrize(
    ("ohlc", "label", "price", "offset"),
    [
        (STOP, label_outcomes.STOP_HIT, 98.5, 2),
        (TARGET, label_outcomes.TARGET_HIT, 103.0, 2),
        ((96.0, 97.0, 95.0, 96.5), label_outcomes.STOP_HIT, 96.0, 2),
        ((106.0, 107.0, 105.0, 106.5), label_outcomes.TARGET_HIT, 103.0, 2),
        ((105.0, 106.0, 98.0, 100.0), label_outcomes.STOP_HIT, 98.5, 2),
        ((100.0, 101.0, 99.0, 100.5), label_outcomes.TIMEOUT, 100.5, 10),
    ],
)
def test_complete_path_preserves_fill_semantics(
    ohlc: tuple[float, ...], label: int, price: float, offset: int,
) -> None:
    bars = _bars()
    bars.loc[offset, OHLC] = ohlc
    result = _resolve(bars).iloc[0]
    assert result["barrier_label"] == label
    assert result["exit_price"] == pytest.approx(price)
    assert result["exit_session"] == bars.loc[offset, "session"]
    assert result["holding_sessions"] == offset
    assert result["target_price"] == 103.0
    assert result["stop_price"] == 98.5


@pytest.mark.parametrize("fill", [STOP, TARGET])
def test_early_fill_survives_missing_later_session(fill: tuple[float, ...]) -> None:
    bars = _bars()
    bars.loc[2, OHLC] = fill
    expected = _resolve(bars)
    pdt.assert_frame_equal(_resolve(bars.drop(index=3)), expected)


@pytest.mark.parametrize("fill", [STOP, TARGET])
@pytest.mark.parametrize("missing_offset", [1, 2, 3])
def test_missing_observation_before_or_on_fill_blocks_outcome(
    fill: tuple[float, ...], missing_offset: int,
) -> None:
    bars = _bars()
    bars.loc[3, OHLC] = fill
    bars.loc[4, OHLC] = fill
    _assert_unresolved(_resolve(bars.drop(index=missing_offset)))


@pytest.mark.parametrize("fill", [STOP, TARGET])
@pytest.mark.parametrize("offset", [1, 2])
def test_truncated_tail_with_verified_fill_remains_resolved(
    fill: tuple[float, ...], offset: int,
) -> None:
    bars = _bars()
    bars.loc[offset, OHLC] = fill
    pdt.assert_frame_equal(_resolve(bars.iloc[: offset + 1]), _resolve(bars))


@pytest.mark.parametrize("observed_count", [1, 2, 5, 10])
def test_truncated_tail_without_fill_never_invents_timeout(observed_count: int) -> None:
    _assert_unresolved(_resolve(_bars().iloc[:observed_count]))


def test_barrier_after_horizon_does_not_replace_timeout() -> None:
    bars = _bars()
    expected = _resolve(bars)
    bars.loc[11, OHLC] = STOP
    pdt.assert_frame_equal(_resolve(bars), expected)


@pytest.mark.parametrize("fill", [STOP, TARGET])
@pytest.mark.parametrize("poison", INVALID_OHLC)
def test_future_poison_cannot_erase_known_fill(
    fill: tuple[float, ...], poison: tuple[object, ...],
) -> None:
    bars = _bars().astype(dict.fromkeys(OHLC, object))
    bars.loc[2, OHLC] = fill
    expected = _resolve(bars)
    bars.loc[3, OHLC] = poison
    pdt.assert_frame_equal(_resolve(bars), expected)


@pytest.mark.parametrize("offset", [1, 2, 3])
@pytest.mark.parametrize("poison", INVALID_OHLC)
def test_invalid_ohlc_through_actual_exit_blocks_outcome(
    offset: int, poison: tuple[object, ...],
) -> None:
    bars = _bars().astype(dict.fromkeys(OHLC, object))
    bars.loc[3, OHLC] = STOP
    bars.loc[offset, OHLC] = poison
    _assert_unresolved(_resolve(bars))


@pytest.mark.parametrize("field", OHLC)
def test_invalid_exit_bar_cannot_confirm_target(field: str) -> None:
    bars = _bars()
    bars.loc[2, OHLC] = TARGET
    bars.loc[2, field] = np.nan
    _assert_unresolved(_resolve(bars))


@pytest.mark.parametrize("offset", [1, 5, 10])
def test_invalid_ohlc_blocks_timeout(offset: int) -> None:
    bars = _bars()
    bars.loc[offset, "close"] = 102.0
    _assert_unresolved(_resolve(bars))


def test_later_decision_still_requires_its_own_valid_path() -> None:
    bars = _bars()
    bars.loc[2, OHLC] = STOP
    bars.loc[3, "close"] = np.nan
    entries = pd.DataFrame({"session": ["2024-01-02", "2024-01-04"], "atr": 1.0})
    result = apply_triple_barrier(bars, entries, spec=SPEC)
    assert result.loc[0, "barrier_label"] == label_outcomes.STOP_HIT
    _assert_unresolved(result.iloc[[1]])


@pytest.mark.parametrize(
    ("decision", "entry"), [("2024-01-05", "2024-01-08"), ("2024-01-12", "2024-01-16")],
)
def test_next_session_open_skips_weekends_and_exchange_holidays(decision: str, entry: str) -> None:
    bars = pd.DataFrame([
        (decision, 100.0, 999.0, 1.0, 100.0),
        (entry, 50.0, 54.0, 49.0, 53.0),
    ], columns=["session", *OHLC])
    result = _resolve(bars, decision).iloc[0]
    assert result["barrier_label"] == label_outcomes.TARGET_HIT
    assert result["exit_session"] == entry
    assert result["exit_price"] == 53.0
    assert result["holding_sessions"] == 1
    _assert_unresolved(_resolve(bars.iloc[:1], decision))


@pytest.mark.parametrize("decision", ["2024-01-06", "2024-01-15"])
def test_non_exchange_decision_never_enters_next_observed_bar(decision: str) -> None:
    bars = _bars()
    bars.loc[bars["session"].gt(decision), "high"] = 104.0
    _assert_unresolved(_resolve(bars, decision))


def test_non_exchange_observation_is_still_rejected() -> None:
    bars = _bars()
    bars.loc[2, OHLC] = STOP
    bars.loc[3, "session"] = "2024-01-06"
    with pytest.raises(DataReadinessError, match="non-exchange sessions"):
        _resolve(bars)


@pytest.mark.parametrize("defect", ["none", "missing", "invalid"])
def test_fixed_horizon_labels_remain_independent_of_early_managed_exit(defect: str) -> None:
    bars = _bars()
    bars.loc[2, OHLC] = STOP
    calendar = xcals.get_calendar("XNYS")
    stock = bars.rename(columns={"session": "session_date_et"})
    stock["session_date_et"] = pd.to_datetime(stock["session_date_et"]).dt.date
    stock["security_id"] = "issuer:one"
    stock["ticker"] = "ONE"
    stock["volume"] = 1000
    stock["bar_start_utc"] = stock["session_date_et"].map(calendar.session_open)
    stock["bar_end_utc"] = stock["session_date_et"].map(calendar.session_close)
    stock["available_at_utc"] = stock["bar_end_utc"] + pd.Timedelta(minutes=15)
    decisions = stock.iloc[:1].assign(
        feature_eligible=True, primary_benchmark="XLK", decision_group_id="2024-01-02",
    )
    benchmarks = pd.concat([stock.assign(ticker=ticker) for ticker in ("SPY", "QQQ", "XLK")])
    if defect == "missing":
        stock = stock.drop(index=3)
    elif defect == "invalid":
        stock.loc[3, "close"] = np.nan
    fixed = add_exact_swing_labels(
        decisions, benchmarks, SwingDatasetConfig(horizon_sessions=10, round_trip_cost_bps=20),
        outcome_bars=stock,
    ).iloc[0]
    managed = _resolve(stock.rename(columns={"session_date_et": "session"})).iloc[0]
    assert managed["barrier_label"] == label_outcomes.STOP_HIT
    assert managed["exit_price"] == 98.5
    assert fixed["exit_session_date_et"] == pd.Timestamp("2024-01-17").date()
    if defect == "none":
        assert fixed["label_eligible"]
        assert fixed["future_net_return_10d"] == pytest.approx(-0.002)
    else:
        assert not fixed["label_eligible"]
        assert pd.isna(fixed["future_net_return_10d"])
