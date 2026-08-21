from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from market_predictor.edge_rebuild.intraday_bar_features import (
    INTRADAY_BAR_FEATURE_SCHEMA_VERSION,
    INTRADAY_BAR_MODEL_FEATURE_COLUMNS,
    INTRADAY_BAR_MODEL_FEATURES_JSON,
    INTRADAY_BAR_MODEL_FEATURES_SHA256,
    build_causal_intraday_bar_features,
)
from market_predictor.edge_rebuild.intraday_bar_live import (
    build_live_intraday_bar_features,
)
from market_predictor.edge_rebuild.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)
from market_predictor.core.errors import DataReadinessError

ROOT = Path(__file__).resolve().parents[1]


def _contract() -> StrategyContract:
    return load_strategy_contract(ROOT / "configs" / "edge_rebuild_strategy_contract.toml")


def _clock_bars(
    ticker: str,
    *,
    day: str,
    opened: str,
    periods: int,
    minutes: int,
    offset: float = 0.0,
) -> pd.DataFrame:
    starts = pd.date_range(opened, periods=periods, freq=f"{minutes}min")
    sequence = np.arange(periods, dtype="float64")
    opens = 100.0 + offset + sequence * 0.04 + np.sin(sequence / 3.0) * 0.10
    closes = opens + np.sin(sequence / 2.0) * 0.08 + 0.02
    ends = starts + pd.Timedelta(minutes=minutes)
    availability = ends + (pd.Timedelta(minutes=1) if minutes == 5 else pd.Timedelta(0))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "timeframe": f"{minutes}m",
            "bar_start_utc": starts,
            "bar_end_utc": ends,
            "available_at_utc": availability,
            "open": opens,
            "high": np.maximum(opens, closes) + 0.12,
            "low": np.minimum(opens, closes) - 0.12,
            "close": closes,
            "volume": 1_000.0 + sequence * 3.0,
            "source": "alpaca",
            "price_feed": "sip",
            "adjustment": "all",
        }
    )


def _volume_bars(
    ticker: str,
    *,
    day: str,
    opened: str,
    periods: int = 25,
    minute_stride: int = 1,
    offset: float = 0.0,
) -> pd.DataFrame:
    contract = _contract()
    starts = pd.date_range(opened, periods=periods, freq=f"{minute_stride}min")
    sequence = np.arange(periods, dtype="float64")
    opens = 100.0 + offset + sequence * 0.05 + np.sin(sequence / 2.0) * 0.15
    closes = opens + np.cos(sequence / 2.5) * 0.09
    ends = starts + pd.Timedelta(minutes=1)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "session_date_et": date.fromisoformat(day),
            "volume_bar_number": np.arange(1, periods + 1),
            "bar_start_utc": starts,
            "bar_end_utc": ends,
            "available_at_utc": ends,
            "open": opens,
            "high": np.maximum(opens, closes) + 0.10,
            "low": np.minimum(opens, closes) - 0.10,
            "close": closes,
            "volume": 1_000.0 + sequence,
            "volume_threshold": 900.0,
            "volume_overshoot": 100.0 + sequence,
            "relative_volume_at_activation": 2.5,
            "activation_time_utc": pd.Timestamp(opened) + pd.Timedelta(minutes=6),
            "source": "alpaca",
            "price_feed": "sip",
            "adjustment": "all",
            "source_timeframe": "1m",
            "strategy_contract_sha256": contract.sha256(),
        }
    )


def _memberships(tickers: tuple[str, ...], *, available: object = "2020-01-01T00:00:00Z") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": list(tickers),
            "security_id": [f"SEC-{ticker}" for ticker in tickers],
            "sector": ["Information Technology"] * len(tickers),
            "primary_benchmark": ["XLK"] * len(tickers),
            "universe_snapshot_id": ["pit-sp500-1"] * len(tickers),
            "effective_from_utc": [pd.Timestamp("2020-01-01T00:00:00Z")] * len(tickers),
            "effective_to_utc": [pd.NaT] * len(tickers),
            "available_at_utc": [pd.Timestamp(available)] * len(tickers),
        }
    )


def _inputs(
    *,
    tickers: tuple[str, ...] = ("AAA",),
    day: str = "2026-07-08",
    opened: str = "2026-07-08T13:30:00Z",
    five_periods: int = 30,
    minute_periods: int = 150,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    five = pd.concat(
        [
            _clock_bars(ticker, day=day, opened=opened, periods=five_periods, minutes=5, offset=index * 10.0)
            for index, ticker in enumerate(tickers)
        ],
        ignore_index=True,
    )
    stocks = pd.concat(
        [
            _clock_bars(ticker, day=day, opened=opened, periods=minute_periods, minutes=1, offset=index * 10.0)
            for index, ticker in enumerate(tickers)
        ],
        ignore_index=True,
    )
    benchmarks = pd.concat(
        [
            _clock_bars(ticker, day=day, opened=opened, periods=minute_periods, minutes=1, offset=offset)
            for ticker, offset in (("SPY", 20.0), ("QQQ", 30.0), ("XLK", 40.0))
        ],
        ignore_index=True,
    )
    volume = pd.concat(
        [
            _volume_bars(ticker, day=day, opened=opened, minute_stride=index + 1, offset=index * 10.0)
            for index, ticker in enumerate(tickers)
        ],
        ignore_index=True,
    )
    activation_time = pd.Timestamp(opened) + pd.Timedelta(minutes=6)
    activations = pd.DataFrame(
        {
            "ticker": list(tickers),
            "session_date_et": [date.fromisoformat(day)] * len(tickers),
            "activation_time_utc": [activation_time] * len(tickers),
            "median_volume_prior_sessions": [1_000_000.0] * len(tickers),
            "relative_volume_at_activation": [2.5 + index for index in range(len(tickers))],
        }
    )
    return volume, five, stocks, benchmarks, _memberships(tickers), activations


def _build(
    inputs: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame],
) -> pd.DataFrame:
    return build_causal_intraday_bar_features(*inputs, contract=_contract())


def _live_build(
    inputs: tuple[
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
        pd.DataFrame,
    ],
    *,
    cutoff: pd.Timestamp,
) -> None:
    bounded = [frame.copy() for frame in inputs]
    for index in (0, 1, 2, 3):
        bounded[index] = bounded[index].loc[
            pd.to_datetime(
                bounded[index]["bar_end_utc"], utc=True, errors="raise"
            ).le(cutoff)
            & pd.to_datetime(
                bounded[index]["available_at_utc"], utc=True, errors="raise"
            ).le(cutoff)
        ].copy()
    build_live_intraday_bar_features(
        *bounded,
        contract=_contract(),
        as_of_utc=cutoff,
    )


def test_builds_fixed_cohorts_with_ordered_float32_features() -> None:
    result = _build(_inputs(tickers=("AAA", "BBB")))
    eligible = result.loc[result["feature_eligible"]]

    assert not eligible.empty
    assert not result.duplicated(["ticker", "decision_time_utc"]).any()
    assert eligible.groupby("decision_time_utc")["ticker"].nunique().max() == 2
    assert result["feature_available_at_utc"].equals(result["decision_time_utc"])
    assert eligible["source_feature_available_at_utc"].le(eligible["decision_time_utc"]).all()
    assert result["feature_schema_version"].eq(INTRADAY_BAR_FEATURE_SCHEMA_VERSION).all()
    assert result["ordered_feature_names_json"].eq(INTRADAY_BAR_MODEL_FEATURES_JSON).all()
    assert result["ordered_feature_sha256"].eq(INTRADAY_BAR_MODEL_FEATURES_SHA256).all()
    assert "atr_14_5m" in result.columns
    assert np.allclose(
        eligible["five_minute_atr_14_fraction_of_close"],
        eligible["atr_14_5m"] / eligible["stock_context_close"],
    )
    assert tuple(result.loc[:, INTRADAY_BAR_MODEL_FEATURE_COLUMNS].columns) == INTRADAY_BAR_MODEL_FEATURE_COLUMNS
    assert all(dtype == np.dtype("float32") for dtype in result.loc[:, INTRADAY_BAR_MODEL_FEATURE_COLUMNS].dtypes)
    assert not any("macd" in column or "quote" in column or "trade" in column for column in INTRADAY_BAR_MODEL_FEATURE_COLUMNS)


def test_atr_comes_only_from_five_minute_bars() -> None:
    inputs = _inputs()
    expected = _build(inputs)
    changed = tuple(frame.copy() for frame in inputs)
    volume = changed[0]
    volume[["open", "high", "low", "close"]] *= 3.0
    volume["volume"] *= 100.0
    observed = _build(changed)

    pdt.assert_series_equal(
        expected["atr_14_5m"],
        observed["atr_14_5m"],
        check_names=False,
    )
    pdt.assert_series_equal(
        expected["five_minute_atr_14_fraction_of_close"],
        observed["five_minute_atr_14_fraction_of_close"],
        check_names=False,
    )


def test_late_volume_bar_state_waits_for_next_fixed_cohort() -> None:
    inputs = list(_inputs())
    volume = inputs[0]
    twentieth_cutoff = pd.Timestamp("2026-07-08T15:11:00Z")
    volume.loc[volume["volume_bar_number"].eq(25), "available_at_utc"] = twentieth_cutoff + pd.Timedelta(seconds=30)
    result = _build(tuple(inputs))

    at_cutoff = result.loc[result["decision_time_utc"].eq(twentieth_cutoff)].iloc[0]
    next_cutoff = result.loc[result["decision_time_utc"].gt(twentieth_cutoff)].iloc[0]
    assert at_cutoff["volume_bar_number"] == 24
    assert next_cutoff["volume_bar_number"] == 25
    assert at_cutoff["feature_available_at_utc"] == twentieth_cutoff


def test_future_poison_cannot_change_earlier_fixed_rows() -> None:
    inputs = _inputs()
    expected = _build(inputs)
    cutoff = pd.Timestamp("2026-07-08T15:11:00Z")
    poisoned = tuple(frame.copy() for frame in inputs)
    future_five = poisoned[1]["bar_end_utc"].gt(cutoff)
    poisoned[1].loc[future_five, ["open", "high", "low", "close"]] *= 20.0
    future_stock = poisoned[2]["bar_start_utc"].ge(cutoff)
    poisoned[2].loc[future_stock, ["open", "high", "low", "close", "volume"]] *= 20.0
    future_benchmark = poisoned[3]["bar_start_utc"].ge(cutoff)
    poisoned[3].loc[future_benchmark, ["open", "high", "low", "close", "volume"]] *= 20.0

    observed = _build(poisoned)
    columns = [
        "ticker",
        "decision_time_utc",
        "volume_bar_number",
        "source_feature_available_at_utc",
        "feature_eligible",
        *INTRADAY_BAR_MODEL_FEATURE_COLUMNS,
    ]
    pdt.assert_frame_equal(
        expected.loc[expected["decision_time_utc"].le(cutoff), columns].reset_index(drop=True),
        observed.loc[observed["decision_time_utc"].le(cutoff), columns].reset_index(drop=True),
    )


def test_missing_later_five_minute_bar_preserves_grid_and_earlier_rows() -> None:
    inputs = _inputs()
    expected = _build(inputs)
    cutoff = pd.Timestamp("2026-07-08T15:11:00Z")
    missing_start = cutoff - pd.Timedelta(minutes=6)
    incomplete = tuple(frame.copy() for frame in inputs)
    missing_index = incomplete[1].index[incomplete[1]["bar_start_utc"].eq(missing_start)]
    incomplete[1].drop(missing_index, inplace=True)
    observed = _build(incomplete)

    pdt.assert_frame_equal(
        expected.loc[expected["decision_time_utc"].lt(cutoff)].reset_index(drop=True),
        observed.loc[observed["decision_time_utc"].lt(cutoff)].reset_index(drop=True),
    )
    assert len(observed) == len(expected)
    missing = observed.loc[observed["decision_time_utc"].eq(cutoff)].iloc[0]
    assert not missing["feature_eligible"]
    assert missing["feature_ineligible_reason"] == "missing_exact_five_minute_bar"
    assert not missing["five_minute_bar_observed"]
    later = observed.loc[observed["decision_time_utc"].eq(cutoff + pd.Timedelta(minutes=5))].iloc[0]
    assert later["five_minute_bar_observed"]
    assert not later["five_minute_prefix_complete"]
    assert later["feature_ineligible_reason"] == "incomplete_five_minute_session_prefix"
    assert pd.isna(later["atr_14_5m"])


def test_missing_exact_benchmark_context_abstains_without_shifting_cohort() -> None:
    inputs = list(_inputs())
    cutoff = pd.Timestamp("2026-07-08T15:11:00Z")
    target_minute = cutoff - pd.Timedelta(minutes=2)
    benchmarks = inputs[3]
    inputs[3] = benchmarks.loc[
        ~(benchmarks["ticker"].eq("QQQ") & benchmarks["bar_start_utc"].eq(target_minute))
    ].copy()
    result = _build(tuple(inputs))
    row = result.loc[result["decision_time_utc"].eq(cutoff)].iloc[0]

    assert not row["feature_eligible"]
    assert row["feature_ineligible_reason"] == "missing_exact_qqq_one_minute_context"
    assert row["decision_time_utc"] == cutoff
    assert row["feature_available_at_utc"] == cutoff


def test_early_close_uses_exchange_calendar_session_boundary() -> None:
    inputs = _inputs(
        day="2025-11-28",
        opened="2025-11-28T14:30:00Z",
        five_periods=42,
        minute_periods=210,
    )
    result = _build(inputs)

    assert result["session_close_utc"].eq(pd.Timestamp("2025-11-28T18:00:00Z")).all()
    assert result["decision_time_utc"].lt(pd.Timestamp("2025-11-28T18:00:00Z")).all()
    assert result["decision_time_utc"].max() == pd.Timestamp("2025-11-28T17:56:00Z")


def test_membership_unavailable_at_open_abstains_but_keeps_fixed_identity() -> None:
    inputs = list(_inputs())
    inputs[4] = _memberships(("AAA",), available="2026-07-08T13:31:00Z")
    result = _build(tuple(inputs))
    mature = result.loc[
        result["five_minute_bar_observed"] & result["volume_bar_number"].ge(20)
    ]

    assert not mature["feature_eligible"].any()
    assert mature["feature_ineligible_reason"].eq("membership_not_available_at_session_open").all()
    assert mature["security_id"].eq("SEC-AAA").all()
    assert mature["primary_benchmark"].eq("XLK").all()


@pytest.mark.parametrize("column", ["effective_from_utc", "available_at_utc"])
def test_null_membership_causal_timestamp_fails_batch_and_live(
    column: str,
) -> None:
    inputs = list(_inputs())
    memberships = inputs[4].copy()
    memberships.loc[:, column] = pd.NaT
    inputs[4] = memberships
    poisoned = tuple(inputs)

    with pytest.raises(
        DataReadinessError,
        match="membership causal timestamps are incomplete",
    ):
        _build(poisoned)

    with pytest.raises(
        DataReadinessError,
        match="membership causal timestamps are incomplete",
    ):
        _live_build(
            poisoned,
            cutoff=pd.Timestamp("2026-07-08T15:11:00Z"),
        )


def test_future_effective_membership_fails_closed() -> None:
    inputs = list(_inputs())
    memberships = inputs[4].copy()
    memberships.loc[:, "effective_from_utc"] = pd.Timestamp(
        "2026-07-08T13:31:00Z"
    )
    inputs[4] = memberships

    with pytest.raises(DataReadinessError, match="PIT membership is not unique"):
        _build(tuple(inputs))


@pytest.mark.parametrize(
    ("column", "message"),
    [
        ("effective_from_utc", "PIT membership is not unique"),
        ("available_at_utc", "scheduled live intraday bar cohort has no eligible rows"),
    ],
)
def test_late_membership_timestamp_fails_live(
    column: str,
    message: str,
) -> None:
    inputs = list(_inputs())
    memberships = inputs[4].copy()
    memberships.loc[:, column] = pd.Timestamp("2026-07-08T13:31:00Z")
    inputs[4] = memberships

    with pytest.raises(DataReadinessError, match=message):
        _live_build(
            tuple(inputs),
            cutoff=pd.Timestamp("2026-07-08T15:11:00Z"),
        )
