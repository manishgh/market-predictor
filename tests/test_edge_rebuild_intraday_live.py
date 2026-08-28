from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest

import market_predictor.intraday.datasets.live as live_module
from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.datasets.live import (
    INTRADAY_LIVE_AUDIT_COLUMNS,
    INTRADAY_LIVE_SCHEMA_VERSION,
    build_live_intraday_features,
)
from market_predictor.intraday.features.features import (
    CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS,
    build_causal_intraday_features,
)
from market_predictor.modeling.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def _contract() -> StrategyContract:
    return load_strategy_contract(ROOT / "configs" / "edge_rebuild_strategy_contract.toml")


def _starts(rows: int) -> pd.DatetimeIndex:
    return pd.date_range("2026-07-08 13:30:00Z", periods=rows, freq="1min")


def _volume_bars(rows: int = 25) -> pd.DataFrame:
    contract = _contract()
    starts = _starts(rows)
    opens = 100.0 + np.arange(rows, dtype="float64") * 0.1
    closes = opens + np.sin(np.arange(rows, dtype="float64") / 2.0) * 0.04 + 0.02
    numbers = np.arange(1, rows + 1)
    return pd.DataFrame(
        {
            "ticker": "AAA",
            "session_date_et": date(2026, 7, 8),
            "volume_bar_number": numbers,
            "bar_start_utc": starts,
            "bar_end_utc": starts + pd.Timedelta(minutes=1),
            "available_at_utc": starts + pd.Timedelta(minutes=2),
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
            "relative_volume_at_activation": 2.5,
            "activation_time_utc": starts[0] + pd.Timedelta(minutes=1),
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
    rows: int = 25,
    price_offset: float = 0.0,
    stock_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    starts = _starts(rows)
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


def _inputs(
    rows: int = 25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    volume = _volume_bars(rows)
    stock = _minute_bars("AAA", rows=rows, stock_prices=volume)
    benchmarks = pd.concat(
        [
            _minute_bars("SPY", rows=rows, price_offset=20.0),
            _minute_bars("QQQ", rows=rows, price_offset=30.0),
            _minute_bars("XLK", rows=rows, price_offset=40.0),
        ],
        ignore_index=True,
    )
    membership = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "session_date_et": [date(2026, 7, 8)],
            "session_open_utc": [pd.Timestamp("2026-07-08 13:30:00Z")],
            "session_close_utc": [pd.Timestamp("2026-07-08 20:00:00Z")],
            "security_id": ["SEC-AAA"],
            "sector": ["Technology"],
            "primary_benchmark": ["XLK"],
            "universe_snapshot_id": ["pit-snapshot-1"],
            "effective_from_utc": [pd.Timestamp("2020-01-01T00:00:00Z")],
            "effective_to_utc": [pd.NaT],
        }
    )
    return volume, stock, benchmarks, membership


def _live(
    volume: pd.DataFrame,
    stock: pd.DataFrame,
    benchmarks: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    as_of_utc: object = "2026-07-08T13:56:00Z",
):
    contract = _contract()
    return build_live_intraday_features(
        volume,
        stock,
        benchmarks,
        memberships,
        contract=contract,
        strategy_contract_sha256=contract.sha256(),
        as_of_utc=as_of_utc,
    )


def test_live_builder_calls_shared_transform_and_matches_latest_batch_row(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _inputs()
    contract = _contract()
    batch = build_causal_intraday_features(
        *inputs,
        contract=contract,
        strategy_contract_sha256=contract.sha256(),
    )
    calls = 0

    def tracked_build(*args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal calls
        calls += 1
        return build_causal_intraday_features(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(live_module, "build_causal_intraday_features", tracked_build)
    live = _live(*inputs)

    expected = batch.loc[batch["volume_bar_number"].eq(25), CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS].reset_index(drop=True)
    assert calls == 1
    assert tuple(live.model_features.columns) == CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS
    assert all(dtype == np.dtype("float32") for dtype in live.model_features.dtypes)
    pdt.assert_frame_equal(live.model_features, expected)
    assert tuple(live.audit_identity.columns) == INTRADAY_LIVE_AUDIT_COLUMNS
    assert live.audit_identity.loc[0, "volume_bar_number"] == 25
    assert live.audit_identity.loc[0, "live_schema_version"] == INTRADAY_LIVE_SCHEMA_VERSION
    assert live.audit_identity.loc[0, "as_of_utc"] == pd.Timestamp("2026-07-08T13:56:00Z")
    assert len(live.audit_identity.loc[0, "decision_id"]) == 64


def test_future_poisoned_evidence_is_rejected_instead_of_filtered() -> None:
    volume, stock, benchmarks, memberships = _inputs(rows=26)
    future = volume["volume_bar_number"].eq(26)
    volume.loc[future, ["open", "high", "low", "close"]] += 10_000.0
    volume.loc[future, "volume"] = 10_000.0
    volume.loc[future, "volume_threshold"] = 8_000.0
    volume.loc[future, "volume_overshoot"] = 2_000.0
    stock_future = stock["bar_start_utc"].eq(volume.loc[future, "bar_start_utc"].iloc[0])
    stock.loc[stock_future, ["open", "high", "low", "close"]] = volume.loc[
        future, ["open", "high", "low", "close"]
    ].to_numpy()

    with pytest.raises(DataReadinessError, match="after as_of_utc"):
        _live(
            volume,
            stock,
            benchmarks,
            memberships,
            as_of_utc="2026-07-08T13:56:00Z",
        )


def test_stale_sector_benchmark_does_not_fall_back_to_previous_decision() -> None:
    volume, stock, benchmarks, memberships = _inputs()
    latest_minute = volume.loc[volume["volume_bar_number"].eq(25), "last_source_minute_utc"].iloc[0]
    benchmarks = benchmarks.loc[
        ~(benchmarks["ticker"].eq("XLK") & benchmarks["bar_start_utc"].eq(latest_minute))
    ]

    with pytest.raises(DataReadinessError, match="no stale fallback.*missing_exact_sector"):
        _live(volume, stock, benchmarks, memberships)


def test_missing_qqq_benchmark_rejects_exact_live_decision() -> None:
    volume, stock, benchmarks, memberships = _inputs()
    benchmarks = benchmarks.loc[~benchmarks["ticker"].eq("QQQ")]

    with pytest.raises(DataReadinessError, match="missing_exact_qqq"):
        _live(volume, stock, benchmarks, memberships)


def test_rejects_future_membership_evidence_and_naive_cutoff() -> None:
    volume, stock, benchmarks, memberships = _inputs()
    memberships.loc[0, "effective_from_utc"] = pd.Timestamp("2026-07-08T14:00:00Z")
    with pytest.raises(DataReadinessError, match="membership.*after as_of_utc"):
        _live(volume, stock, benchmarks, memberships)

    volume, stock, benchmarks, memberships = _inputs()
    with pytest.raises(DataReadinessError, match="timezone-aware"):
        _live(
            volume,
            stock,
            benchmarks,
            memberships,
            as_of_utc="2026-07-08 13:56:00",
        )


def test_rejects_latest_row_when_it_is_stale_for_requested_cutoff() -> None:
    with pytest.raises(DataReadinessError, match="decision is stale"):
        _live(*_inputs(), as_of_utc="2026-07-08T14:30:00Z")
