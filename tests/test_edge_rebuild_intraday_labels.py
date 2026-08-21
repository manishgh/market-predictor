from __future__ import annotations

from collections.abc import Callable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from market_predictor.edge_rebuild.intraday_features import FEATURE_SCHEMA_VERSION
from market_predictor.edge_rebuild.intraday_labels import (
    LABEL_SCHEMA_VERSION,
    build_exact_causal_intraday_labels,
)
from market_predictor.edge_rebuild.labeling import RANK_BOTTOM, RANK_TOP, STOP_HIT
from market_predictor.edge_rebuild.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)
from market_predictor.core.errors import DataReadinessError

ROOT = Path(__file__).resolve().parents[1]


def _contract() -> StrategyContract:
    return load_strategy_contract(ROOT / "configs" / "edge_rebuild_strategy_contract.toml")


def _feature(
    *,
    ticker: str = "AAA",
    day: str = "2026-07-08",
    open_: str = "2026-07-08T13:30:00Z",
    available: str = "2026-07-08T14:00:00Z",
    close: str = "2026-07-08T20:00:00Z",
    eligible: bool = True,
) -> pd.DataFrame:
    contract = _contract()
    return pd.DataFrame(
        {
            "ticker": [ticker],
            "security_id": [f"SEC-{ticker}"],
            "session_date_et": [date.fromisoformat(day)],
            "session_open_utc": [pd.Timestamp(open_)],
            "session_close_utc": [pd.Timestamp(close)],
            "volume_bar_number": [20],
            "available_at_utc": [pd.Timestamp(available)],
            "feature_available_at_utc": [pd.Timestamp(available)],
            "feature_eligible": pd.Series([eligible], dtype=bool),
            "feature_ineligible_reason": pd.Series(
                [pd.NA if eligible else "insufficient_completed_volume_bars"],
                dtype="string",
            ),
            "atr_14": [1.0],
            "primary_benchmark": ["XLK"],
            "universe_snapshot_id": ["pit-1"],
            "source": ["alpaca"],
            "price_feed": ["sip"],
            "adjustment": ["all"],
            "source_timeframe": ["1m"],
            "feature_schema_version": [FEATURE_SCHEMA_VERSION],
            "strategy_contract_sha256": [contract.sha256()],
        }
    )


def _bars(
    ticker: str,
    *,
    day: str = "2026-07-08",
    start: str = "2026-07-08T13:30:00Z",
    periods: int = 120,
    final_delta: float = 0.3,
) -> pd.DataFrame:
    starts = pd.date_range(start, periods=periods, freq="1min", tz="UTC")
    opens = np.full(periods, 100.0)
    closes = np.linspace(100.0, 100.0 + final_delta, periods)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "timeframe": "1m",
            "bar_start_utc": starts,
            "bar_end_utc": starts + pd.Timedelta(minutes=1),
            "available_at_utc": starts + pd.Timedelta(minutes=1),
            "open": opens,
            "high": np.maximum(opens, closes) + 0.1,
            "low": np.minimum(opens, closes) - 0.1,
            "close": closes,
            "volume": 1_000.0,
            "source": "alpaca",
            "price_feed": "sip",
            "adjustment": "all",
        }
    )


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = _feature()
    stocks = _bars("AAA")
    benchmarks = pd.concat(
        [
            _bars("SPY", final_delta=0.1),
            _bars("QQQ", final_delta=0.15),
            _bars("XLK", final_delta=0.2),
        ],
        ignore_index=True,
    )
    return features, stocks, benchmarks


def _build(
    features: pd.DataFrame,
    stocks: pd.DataFrame,
    benchmarks: pd.DataFrame,
) -> pd.DataFrame:
    contract = _contract()
    return build_exact_causal_intraday_labels(
        features,
        stocks,
        benchmarks,
        contract=contract,
        strategy_contract_sha256=contract.sha256(),
    )


def test_uses_strictly_next_exact_minute_and_frozen_cost() -> None:
    features, stocks, benchmarks = _inputs()
    entry = pd.Timestamp("2026-07-08T14:01:00Z")
    target_minute = entry + pd.Timedelta(minutes=2)
    stocks.loc[stocks["bar_start_utc"].eq(target_minute), "high"] = 102.1

    result = _build(features, stocks, benchmarks).iloc[0]

    assert result["label_schema_version"] == LABEL_SCHEMA_VERSION
    assert bool(result["label_eligible"])
    assert result["entry_time_utc"] == entry
    assert result["exit_time_utc"] == target_minute
    assert result["label_available_at_utc"] == target_minute + pd.Timedelta(minutes=1)
    assert result["label_outcome"] == "target_first"
    assert result["label_outcome_reason"] == "target_touched_first"
    assert result["gross_return"] == pytest.approx(0.02)
    assert result["cost"] == pytest.approx(0.001)
    assert result["net_return"] == pytest.approx(0.019)
    assert result["spy_excess_return"] == pytest.approx(result["net_return"] - result["spy_return"])
    assert result["qqq_excess_return"] == pytest.approx(result["net_return"] - result["qqq_return"])


def test_same_minute_collision_is_conservatively_stop_first() -> None:
    features, stocks, benchmarks = _inputs()
    entry = pd.Timestamp("2026-07-08T14:01:00Z")
    stocks.loc[stocks["bar_start_utc"].eq(entry), ["high", "low"]] = [
        102.5,
        98.0,
    ]

    result = _build(features, stocks, benchmarks).iloc[0]

    assert result["barrier_label"] == STOP_HIT
    assert result["label_outcome"] == "stop_first"
    assert result["label_outcome_reason"] == "same_minute_collision_stop_first"
    assert result["exit_price"] == pytest.approx(98.5)
    assert result["holding_minutes"] == 1


def test_missing_exact_next_minute_abstains_instead_of_skipping_forward() -> None:
    features, stocks, benchmarks = _inputs()
    missing = pd.Timestamp("2026-07-08T14:01:00Z")
    stocks = stocks[~stocks["bar_start_utc"].eq(missing)]

    result = _build(features, stocks, benchmarks).iloc[0]

    assert not bool(result["label_eligible"])
    assert result["label_ineligible_reason"] == "missing_exact_next_minute"
    assert pd.isna(result["entry_time_utc"])


def test_missing_minute_inside_horizon_abstains() -> None:
    features, stocks, benchmarks = _inputs()
    missing = pd.Timestamp("2026-07-08T14:10:00Z")
    stocks = stocks[~stocks["bar_start_utc"].eq(missing)]

    result = _build(features, stocks, benchmarks).iloc[0]

    assert not bool(result["label_eligible"])
    assert result["label_ineligible_reason"] == "missing_exact_one_minute_path"


def test_missing_qqq_interval_abstains_instead_of_omitting_comparison() -> None:
    features, stocks, benchmarks = _inputs()
    benchmarks = benchmarks.loc[~benchmarks["ticker"].eq("QQQ")].copy()

    result = _build(features, stocks, benchmarks).iloc[0]

    assert not bool(result["label_eligible"])
    assert result["label_ineligible_reason"] == "missing_exact_qqq_interval"


def test_prior_and_post_horizon_poison_cannot_change_label() -> None:
    features, stocks, benchmarks = _inputs()
    expected = _build(features, stocks, benchmarks)
    poisoned = stocks.copy()
    prior = poisoned["bar_start_utc"].lt(pd.Timestamp("2026-07-08T14:00:00Z"))
    future = poisoned["bar_start_utc"].gt(pd.Timestamp("2026-07-08T14:30:00Z"))
    poisoned.loc[prior | future, ["open", "high", "low", "close"]] *= 50.0
    poisoned.loc[prior | future, "volume"] *= 100.0

    observed = _build(features, poisoned, benchmarks)
    columns = [
        "entry_time_utc",
        "exit_time_utc",
        "label_outcome",
        "exit_price",
        "gross_return",
        "net_return",
    ]
    pdt.assert_frame_equal(expected[columns], observed[columns])


def test_early_close_horizon_abstains_and_never_uses_next_session() -> None:
    features = _feature(
        day="2025-11-28",
        open_="2025-11-28T14:30:00Z",
        available="2025-11-28T17:30:00Z",
        close="2025-11-28T18:00:00Z",
    )
    early = _bars(
        "AAA",
        day="2025-11-28",
        start="2025-11-28T14:30:00Z",
        periods=210,
    )
    next_day = _bars(
        "AAA",
        day="2025-12-01",
        start="2025-12-01T14:30:00Z",
        periods=60,
    )
    next_day[["open", "high", "low", "close"]] *= 10.0
    benchmarks = pd.concat(
        [
            _bars("SPY", day="2025-11-28", start="2025-11-28T14:30:00Z", periods=210),
            _bars("QQQ", day="2025-11-28", start="2025-11-28T14:30:00Z", periods=210),
            _bars("XLK", day="2025-11-28", start="2025-11-28T14:30:00Z", periods=210),
        ],
        ignore_index=True,
    )

    result = _build(
        features,
        pd.concat([early, next_day], ignore_index=True),
        benchmarks,
    ).iloc[0]

    assert not bool(result["label_eligible"])
    assert result["label_ineligible_reason"] == "horizon_crosses_session_close"
    assert pd.isna(result["label_outcome"])


def test_contemporaneous_rank_uses_only_same_decision_group() -> None:
    features = pd.concat(
        [_feature(ticker=f"T{index:02d}") for index in range(10)],
        ignore_index=True,
    )
    stocks: list[pd.DataFrame] = []
    for index in range(10):
        frame = _bars(f"T{index:02d}", final_delta=0.0)
        horizon = frame["bar_start_utc"].between(
            pd.Timestamp("2026-07-08T14:01:00Z"),
            pd.Timestamp("2026-07-08T14:30:00Z"),
        )
        delta = -0.9 + index * 0.2
        frame.loc[horizon, "close"] = 100.0 + delta
        frame.loc[horizon, "high"] = np.maximum(frame.loc[horizon, "open"], frame.loc[horizon, "close"]) + 0.05
        frame.loc[horizon, "low"] = np.minimum(frame.loc[horizon, "open"], frame.loc[horizon, "close"]) - 0.05
        stocks.append(frame)
    benchmarks = pd.concat(
        [_bars("SPY"), _bars("QQQ"), _bars("XLK")], ignore_index=True
    )

    result = _build(features, pd.concat(stocks, ignore_index=True), benchmarks)

    assert result["ranking_group_size"].eq(10).all()
    assert int(result["rank_label"].eq(RANK_TOP).sum()) == 2
    assert int(result["rank_label"].eq(RANK_BOTTOM).sum()) == 2


@pytest.mark.parametrize(
    ("input_name", "mutation", "message"),
    [
        ("features", lambda frame: frame.assign(price_feed="iex"), "lineage"),
        ("features", lambda frame: frame.assign(strategy_contract_sha256="bad"), "lineage"),
        ("stocks", lambda frame: frame.assign(adjustment="raw"), "SIP/all"),
        ("stocks", lambda frame: frame.assign(high=frame["low"] - 1.0), "OHLCV"),
        ("benchmarks", lambda frame: frame.assign(volume=np.nan), "OHLCV"),
    ],
)
def test_malformed_or_untrusted_input_fails_closed(
    input_name: str,
    mutation: Callable[[pd.DataFrame], pd.DataFrame],
    message: str,
) -> None:
    features, stocks, benchmarks = _inputs()
    frames = {
        "features": features,
        "stocks": stocks,
        "benchmarks": benchmarks,
    }
    frames[input_name] = mutation(frames[input_name])

    with pytest.raises(DataReadinessError, match=message):
        _build(frames["features"], frames["stocks"], frames["benchmarks"])


def test_feature_ineligible_row_preserves_reason_without_future_evaluation() -> None:
    features, stocks, benchmarks = _inputs()
    features["feature_eligible"] = False
    features["feature_ineligible_reason"] = "insufficient_completed_volume_bars"

    result = _build(features, stocks, benchmarks).iloc[0]

    assert not bool(result["label_eligible"])
    assert result["label_ineligible_reason"] == ("feature_ineligible:insufficient_completed_volume_bars")
    assert pd.isna(result["entry_time_utc"])
