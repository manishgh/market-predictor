from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_predictor.edge_rebuild.intraday_bar_labels import (
    INTRADAY_BAR_LABEL_SCHEMA_VERSION,
    build_exact_intraday_bar_labels,
)
from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.v3.errors import DataReadinessError

CONTRACT_PATH = Path("configs/edge_rebuild_strategy_contract.toml")
DECISION = pd.Timestamp("2026-07-08T14:00:00Z")


def _contract():
    return load_strategy_contract(CONTRACT_PATH)


def _bars(
    ticker: str,
    *,
    decision: pd.Timestamp = DECISION,
    close_delta: float = 0.0,
    availability_delay: int = 0,
) -> pd.DataFrame:
    starts = pd.date_range(decision + pd.Timedelta(minutes=1), periods=30, freq="1min")
    close = np.linspace(100.0, 100.0 + close_delta, len(starts))
    return pd.DataFrame(
        {
            "ticker": ticker,
            "timeframe": "1m",
            "bar_start_utc": starts,
            "bar_end_utc": starts + pd.Timedelta(minutes=1),
            "available_at_utc": starts + pd.Timedelta(minutes=1 + availability_delay),
            "open": np.full(len(starts), 100.0),
            "high": np.maximum(close, 100.0) + 0.1,
            "low": np.minimum(close, 100.0) - 0.1,
            "close": close,
            "source": "alpaca",
            "price_feed": "sip",
            "adjustment": "all",
        }
    )


def _features(
    tickers: list[str],
    *,
    decision: pd.Timestamp = DECISION,
    sectors: list[str] | None = None,
    atr: float = 1.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": tickers,
            "decision_time_utc": decision,
            "feature_available_at_utc": decision,
            "atr_14_5m": atr,
            "primary_benchmark": sectors or ["XLK"] * len(tickers),
        }
    )


def _benchmarks(*, decision: pd.Timestamp = DECISION) -> pd.DataFrame:
    return pd.concat(
        [
            _bars("SPY", decision=decision, close_delta=1.0),
            _bars("QQQ", decision=decision, close_delta=2.0),
            _bars("XLK", decision=decision, close_delta=3.0),
            _bars("XLV", decision=decision, close_delta=-1.0),
        ],
        ignore_index=True,
    )


def _build(
    features: pd.DataFrame,
    stocks: pd.DataFrame,
    benchmarks: pd.DataFrame,
) -> pd.DataFrame:
    contract = _contract()
    return build_exact_intraday_bar_labels(
        features,
        stocks,
        benchmarks,
        contract=contract,
        strategy_contract_sha256=contract.sha256(),
    )


def test_exact_entry_managed_exit_atr_cost_and_benchmark_interval() -> None:
    stocks = _bars("AAA")
    target_time = DECISION + pd.Timedelta(minutes=3)
    stocks.loc[stocks["bar_start_utc"].eq(target_time), "high"] = 102.1
    benchmarks = _benchmarks()
    benchmarks.loc[
        benchmarks["ticker"].eq("XLK") & benchmarks["bar_start_utc"].eq(target_time),
        "available_at_utc",
    ] = target_time + pd.Timedelta(minutes=5)

    row = _build(_features(["AAA"]), stocks, benchmarks).iloc[0]

    assert row["label_schema_version"] == INTRADAY_BAR_LABEL_SCHEMA_VERSION
    assert bool(row["label_eligible"])
    assert row["entry_time_utc"] == DECISION + pd.Timedelta(minutes=1)
    assert row["exit_time_utc"] == target_time
    assert row["target_price"] == 102.0
    assert row["stop_price"] == 98.5
    assert row["exit_price"] == 102.0
    assert row["gross_return"] == pytest.approx(0.02)
    assert row["cost"] == pytest.approx(0.001)
    assert row["net_return"] == pytest.approx(0.019)
    assert row["label_available_at_utc"] == target_time + pd.Timedelta(minutes=5)
    for ticker, column in (("SPY", "spy_return"), ("QQQ", "qqq_return"), ("XLK", "sector_return")):
        frame = benchmarks[benchmarks["ticker"].eq(ticker)].set_index("bar_start_utc")
        expected = frame.loc[target_time, "close"] / frame.loc[DECISION + pd.Timedelta(minutes=1), "open"] - 1.0
        assert row[column] == pytest.approx(expected)


def test_feature_evidence_after_decision_time_is_rejected() -> None:
    features = _features(["AAA"])
    features["feature_available_at_utc"] = DECISION + pd.Timedelta(nanoseconds=1)

    with pytest.raises(
        DataReadinessError,
        match="features contain evidence after decision_time_utc",
    ):
        _build(features, _bars("AAA"), _benchmarks())


def test_rows_outside_exact_path_cannot_poison_label() -> None:
    stocks = _bars("AAA", close_delta=1.0)
    benchmarks = _benchmarks()
    baseline = _build(_features(["AAA"]), stocks, benchmarks)
    poison_times = [DECISION, DECISION + pd.Timedelta(minutes=31)]
    poison = pd.DataFrame(
        {
            "ticker": "AAA",
            "timeframe": "1m",
            "bar_start_utc": poison_times,
            "bar_end_utc": [value + pd.Timedelta(minutes=1) for value in poison_times],
            "available_at_utc": [value + pd.Timedelta(minutes=2) for value in poison_times],
            "open": 10_000.0,
            "high": 20_000.0,
            "low": 9_000.0,
            "close": 15_000.0,
            "source": "alpaca",
            "price_feed": "sip",
            "adjustment": "all",
        }
    )
    poisoned = _build(
        _features(["AAA"]),
        pd.concat([stocks, poison], ignore_index=True),
        benchmarks,
    )
    columns = [
        "entry_time_utc",
        "exit_time_utc",
        "target_price",
        "stop_price",
        "exit_price",
        "net_return",
        "spy_return",
        "qqq_return",
        "sector_return",
    ]
    pd.testing.assert_frame_equal(baseline[columns], poisoned[columns])


def test_missing_stock_or_benchmark_evidence_abstains_only_affected_row() -> None:
    stocks = pd.concat([_bars("AAA"), _bars("BBB")], ignore_index=True)
    missing_time = DECISION + pd.Timedelta(minutes=10)
    stocks = stocks.loc[~(stocks["ticker"].eq("BBB") & stocks["bar_start_utc"].eq(missing_time))]
    result = _build(_features(["AAA", "BBB"]), stocks, _benchmarks()).set_index("ticker")

    assert bool(result.loc["AAA", "label_eligible"])
    assert not bool(result.loc["BBB", "label_eligible"])
    assert result.loc["BBB", "label_ineligible_reason"] == "missing_exact_stock_one_minute_path"
    assert pd.isna(result.loc["BBB", "net_return"])
    assert result.loc["AAA", "net_return"] != 0.0

    benchmarks = _benchmarks()
    benchmarks = benchmarks.loc[~(benchmarks["ticker"].eq("XLV") & benchmarks["bar_start_utc"].eq(DECISION + pd.Timedelta(minutes=10)))]
    result = _build(
        _features(["AAA", "BBB"], sectors=["XLK", "XLV"]),
        pd.concat([_bars("AAA"), _bars("BBB")], ignore_index=True),
        benchmarks,
    ).set_index("ticker")
    assert bool(result.loc["AAA", "label_eligible"])
    assert not bool(result.loc["BBB", "label_eligible"])
    assert result.loc["BBB", "label_ineligible_reason"] == "missing_exact_sector_interval"
    assert pd.isna(result.loc["BBB", "sector_return"])


def test_feature_ineligible_row_is_preserved_without_path_evaluation() -> None:
    features = _features(["AAA", "BBB"])
    features["feature_eligible"] = [True, False]
    features["feature_ineligible_reason"] = [pd.NA, "insufficient_completed_volume_bars"]

    result = _build(features, _bars("AAA"), _benchmarks()).set_index("ticker")

    assert bool(result.loc["AAA", "label_eligible"])
    assert not bool(result.loc["BBB", "label_eligible"])
    assert result.loc["BBB", "label_ineligible_reason"] == ("feature_ineligible:insufficient_completed_volume_bars")
    for column in (
        "entry_time_utc",
        "exit_time_utc",
        "label_available_at_utc",
        "barrier_label",
        "gross_return",
        "net_return",
        "spy_return",
        "qqq_return",
        "sector_return",
    ):
        assert pd.isna(result.loc["BBB", column])


def test_label_availability_uses_slowest_relevant_evidence() -> None:
    stocks = _bars("AAA")
    benchmarks = _benchmarks()
    final = DECISION + pd.Timedelta(minutes=30)
    delayed = final + pd.Timedelta(minutes=7)
    benchmarks.loc[
        benchmarks["ticker"].eq("QQQ") & benchmarks["bar_start_utc"].eq(final),
        "available_at_utc",
    ] = delayed

    row = _build(_features(["AAA"], atr=10.0), stocks, benchmarks).iloc[0]

    assert row["label_outcome"] == "timeout"
    assert row["exit_time_utc"] == final
    assert row["label_available_at_utc"] == delayed


@pytest.mark.parametrize("source", ["stock", "benchmark"])
@pytest.mark.parametrize(
    "column",
    ["bar_start_utc", "bar_end_utc", "available_at_utc"],
)
def test_null_label_path_timestamp_or_availability_fails_closed(
    source: str,
    column: str,
) -> None:
    stocks = _bars("AAA")
    benchmarks = _benchmarks()
    target = stocks if source == "stock" else benchmarks
    target.loc[target.index[0], column] = pd.NaT

    with pytest.raises(
        DataReadinessError,
        match=rf"{source} one-minute bar timestamps or availability are incomplete",
    ):
        _build(_features(["AAA"]), stocks, benchmarks)


def test_rank_labels_use_only_exact_decision_cohort_and_minimum_count() -> None:
    tickers = [f"A{index:02d}" for index in range(10)]
    stocks = []
    for index, ticker in enumerate(tickers):
        stocks.append(_bars(ticker, close_delta=float(index)))
    later = DECISION + pd.Timedelta(hours=1)
    stocks.append(_bars("LATE", decision=later, close_delta=20.0))
    features = pd.concat(
        [
            _features(tickers, atr=10.0),
            _features(["LATE"], decision=later, atr=10.0),
        ],
        ignore_index=True,
    )
    benchmarks = pd.concat([_benchmarks(), _benchmarks(decision=later)], ignore_index=True).drop_duplicates(["ticker", "bar_start_utc"])

    result = _build(features, pd.concat(stocks, ignore_index=True), benchmarks)
    cohort = result[result["decision_time_utc"].eq(DECISION)].sort_values("net_return")
    late = result[result["ticker"].eq("LATE")].iloc[0]

    assert cohort["ranking_group_size"].eq(10).all()
    assert cohort.iloc[:2]["rank_label"].eq(-1).all()
    assert cohort.iloc[-2:]["rank_label"].eq(1).all()
    assert late["ranking_group_size"] == 1
    assert pd.isna(late["rank_label"])
    assert cohort["decision_group_id"].eq(DECISION.isoformat()).all()
