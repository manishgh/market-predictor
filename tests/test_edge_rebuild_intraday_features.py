from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

from market_predictor.edge_rebuild.intraday_features import (
    FEATURE_SCHEMA_VERSION,
    build_causal_intraday_features,
)
from market_predictor.edge_rebuild.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)
from market_predictor.v3.errors import DataReadinessError

ROOT = Path(__file__).resolve().parents[1]


def _contract() -> StrategyContract:
    return load_strategy_contract(ROOT / "configs" / "edge_rebuild_strategy_contract.toml")


def _starts(day: str, rows: int) -> pd.DatetimeIndex:
    return pd.date_range(f"{day} 13:30:00Z", periods=rows, freq="1min")


def _volume_bars(
    *,
    day: str = "2026-07-08",
    rows: int = 25,
    ticker: str = "AAA",
    price_offset: float = 0.0,
) -> pd.DataFrame:
    contract = _contract()
    starts = _starts(day, rows)
    opens = 100.0 + price_offset + np.arange(rows, dtype="float64") * 0.1
    closes = opens + np.sin(np.arange(rows, dtype="float64") / 2.0) * 0.04 + 0.02
    numbers = np.arange(1, rows + 1)
    available = starts + pd.Timedelta(minutes=2)
    activation = starts[0] + pd.Timedelta(minutes=1)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "session_date_et": date.fromisoformat(day),
            "volume_bar_number": numbers,
            "bar_start_utc": starts,
            "bar_end_utc": starts + pd.Timedelta(minutes=1),
            "available_at_utc": available,
            "first_source_minute_utc": starts,
            "last_source_minute_utc": starts,
            "open": opens,
            "high": np.maximum(opens, closes) + 0.1,
            "low": np.minimum(opens, closes) - 0.1,
            "close": closes,
            "volume": 100.0,
            "source_row_count": 1,
            "volume_threshold": 100.0,
            "volume_overshoot": 0.0,
            "activation_time_utc": activation,
            "model_eligible": numbers >= contract.intraday.minimum_warmup_bars,
            "source": "alpaca",
            "price_feed": "sip",
            "adjustment": "all",
            "source_timeframe": "1m",
            "strategy_contract_sha256": contract.sha256(),
        }
    )


def _minute_bars(
    ticker: str,
    *,
    day: str = "2026-07-08",
    rows: int = 25,
    price_offset: float = 0.0,
    stock_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    starts = _starts(day, rows)
    if stock_prices is None:
        opens = 100.0 + price_offset + np.arange(rows, dtype="float64") * 0.1
        closes = opens + np.sin(np.arange(rows, dtype="float64") / 2.0) * 0.04 + 0.02
    else:
        opens = stock_prices["open"].to_numpy(dtype="float64")
        closes = stock_prices["close"].to_numpy(dtype="float64")
    return pd.DataFrame(
        {
            "ticker": ticker,
            "timeframe": "1m",
            "bar_start_utc": starts,
            "bar_end_utc": starts + pd.Timedelta(minutes=1),
            "available_at_utc": starts + pd.Timedelta(minutes=2),
            "open": opens,
            "high": np.maximum(opens, closes) + 0.1,
            "low": np.minimum(opens, closes) - 0.1,
            "close": closes,
            "volume": 1000.0 + np.arange(rows, dtype="float64"),
            "source": "alpaca",
            "price_feed": "sip",
            "adjustment": "all",
        }
    )


def _memberships(*days: str, ticker: str = "AAA") -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for day in days:
        session_open = pd.Timestamp(f"{day} 13:30:00Z")
        records.append(
            {
                "ticker": ticker,
                "session_date_et": date.fromisoformat(day),
                "session_open_utc": session_open,
                "session_close_utc": pd.Timestamp(f"{day} 20:00:00Z"),
                "security_id": f"SEC-{ticker}",
                "sector": "Technology",
                "primary_benchmark": "XLK",
                "universe_snapshot_id": "pit-snapshot-1",
                "effective_from_utc": pd.Timestamp("2020-01-01T00:00:00Z"),
                "effective_to_utc": pd.NaT,
            }
        )
    return pd.DataFrame(records)


def _inputs(
    *,
    day: str = "2026-07-08",
    rows: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    volume = _volume_bars(day=day, rows=rows)
    stock = _minute_bars("AAA", day=day, rows=rows, stock_prices=volume)
    benchmarks = pd.concat(
        [
            _minute_bars("SPY", day=day, rows=rows, price_offset=20.0),
            _minute_bars("QQQ", day=day, rows=rows, price_offset=30.0),
            _minute_bars("XLK", day=day, rows=rows, price_offset=40.0),
        ],
        ignore_index=True,
    )
    return volume, stock, benchmarks, _memberships(day)


def _build(
    volume: pd.DataFrame,
    stock: pd.DataFrame,
    benchmarks: pd.DataFrame,
    memberships: pd.DataFrame,
) -> pd.DataFrame:
    contract = _contract()
    return build_causal_intraday_features(
        volume,
        stock,
        benchmarks,
        memberships,
        contract=contract,
        strategy_contract_sha256=contract.sha256(),
    )


def test_builds_exact_synchronized_context_after_frozen_warmup() -> None:
    result = _build(*_inputs())

    assert result["feature_schema_version"].eq(FEATURE_SCHEMA_VERSION).all()
    assert result["strategy_contract_sha256"].eq(_contract().sha256()).all()
    assert result["universe_snapshot_id"].eq("pit-snapshot-1").all()
    assert not result.iloc[:19]["feature_eligible"].any()
    assert result.iloc[19:]["feature_eligible"].all()

    eligible = result.iloc[19]
    assert eligible["volume_bar_number"] == 20
    assert eligible["feature_context_minute_utc"] == eligible["last_source_minute_utc"]
    for prefix in ("stock_clock", "spy", "qqq", "sector"):
        assert eligible[f"{prefix}_context_available_at_utc"] <= eligible["feature_available_at_utc"]
    assert eligible["stock_clock_context_close"] == pytest.approx(eligible["close"])


def test_appended_and_poisoned_future_rows_cannot_change_feature_prefix() -> None:
    initial = _inputs(rows=25)
    extended = _inputs(rows=30)
    initial_result = _build(*initial)

    extended_volume, extended_stock, extended_benchmarks, memberships = extended
    future_volume = extended_volume["volume_bar_number"].gt(25)
    extended_volume.loc[future_volume, ["open", "high", "low", "close"]] += 10_000.0
    future_stock = extended_stock["bar_start_utc"].gt(initial[1]["bar_start_utc"].max())
    extended_stock.loc[future_stock, ["open", "high", "low", "close"]] = extended_volume.loc[
        future_volume, ["open", "high", "low", "close"]
    ].to_numpy()
    future_benchmark = extended_benchmarks["bar_start_utc"].gt(initial[2]["bar_start_utc"].max())
    extended_benchmarks.loc[future_benchmark, ["open", "high", "low", "close", "volume"]] *= 100.0

    poisoned_result = _build(extended_volume, extended_stock, extended_benchmarks, memberships)
    pdt.assert_frame_equal(
        initial_result.reset_index(drop=True),
        poisoned_result.iloc[: len(initial_result)].reset_index(drop=True),
    )


def test_technical_and_minute_features_reset_at_each_session() -> None:
    first = _inputs(day="2026-07-08")
    second = _inputs(day="2026-07-09")
    result = _build(
        pd.concat([first[0], second[0]], ignore_index=True),
        pd.concat([first[1], second[1]], ignore_index=True),
        pd.concat([first[2], second[2]], ignore_index=True),
        pd.concat([first[3], second[3]], ignore_index=True),
    )

    second_session = result[result["session_date_et"].eq(date(2026, 7, 9))].reset_index(drop=True)
    assert np.isnan(second_session.loc[0, "return_1_bar"])
    assert np.isnan(second_session.loc[0, "rsi_14"])
    assert np.isnan(second_session.loc[0, "stock_clock_return_5m"])
    assert second_session.loc[0, "feature_ineligible_reason"] == "insufficient_completed_volume_bars"
    assert bool(second_session.loc[19, "feature_eligible"])


def test_missing_exact_sector_minute_is_nan_and_not_backfilled() -> None:
    volume, stock, benchmarks, memberships = _inputs()
    missing_minute = volume.loc[20, "last_source_minute_utc"]
    benchmarks = benchmarks[~(benchmarks["ticker"].eq("XLK") & benchmarks["bar_start_utc"].eq(missing_minute))]

    result = _build(volume, stock, benchmarks, memberships)
    row = result[result["volume_bar_number"].eq(21)].iloc[0]

    assert pd.isna(row["sector_context_available_at_utc"])
    assert np.isnan(row["sector_return_1m"])
    assert not bool(row["feature_eligible"])
    assert row["feature_ineligible_reason"] == "missing_exact_sector_minute_context"


def test_feature_availability_waits_for_slowest_exact_context() -> None:
    volume, stock, benchmarks, memberships = _inputs()
    target_minute = volume.loc[19, "last_source_minute_utc"]
    delayed = target_minute + pd.Timedelta(minutes=15)
    benchmarks.loc[
        benchmarks["ticker"].eq("QQQ") & benchmarks["bar_start_utc"].eq(target_minute),
        "available_at_utc",
    ] = delayed

    result = _build(volume, stock, benchmarks, memberships)
    row = result[result["volume_bar_number"].eq(20)].iloc[0]

    assert row["feature_available_at_utc"] == delayed
    assert row["qqq_context_available_at_utc"] == delayed
    assert row["feature_context_minute_utc"] == target_minute


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("wrong_feed", "hash-bound Alpaca SIP/all"),
        ("numbering_gap", "numbering must be contiguous"),
        ("eligibility_spoof", "warm-up lineage"),
        ("inactive_membership", "membership"),
        ("stock_close_mismatch", "exact source one-minute close"),
    ],
)
def test_rejects_invalid_schema_or_lineage(mutation: str, message: str) -> None:
    volume, stock, benchmarks, memberships = _inputs()
    if mutation == "wrong_feed":
        volume.loc[0, "price_feed"] = "iex"
    elif mutation == "numbering_gap":
        volume.loc[10, "volume_bar_number"] = 99
    elif mutation == "eligibility_spoof":
        volume.loc[0, "model_eligible"] = True
    elif mutation == "inactive_membership":
        memberships.loc[0, "effective_from_utc"] = memberships.loc[0, "session_open_utc"] + pd.Timedelta(days=1)
    elif mutation == "stock_close_mismatch":
        stock.loc[20, "close"] += 1.0
        stock.loc[20, "high"] += 1.0
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(DataReadinessError, match=message):
        _build(volume, stock, benchmarks, memberships)


def test_rejects_contract_hash_mismatch() -> None:
    contract = _contract()
    with pytest.raises(DataReadinessError, match="contract hash"):
        build_causal_intraday_features(
            *_inputs(),
            contract=contract,
            strategy_contract_sha256="0" * 64,
        )


def test_rejects_missing_required_schema_column() -> None:
    volume, stock, benchmarks, memberships = _inputs()

    with pytest.raises(DataReadinessError, match="missing columns.*source_row_count"):
        _build(
            volume.drop(columns="source_row_count"),
            stock,
            benchmarks,
            memberships,
        )
