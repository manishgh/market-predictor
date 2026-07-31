from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from market_predictor.edge_rebuild.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)
from market_predictor.edge_rebuild.volume_bars import (
    VolumeBarBuildResult,
    build_causal_volume_bars,
)
from market_predictor.v3.errors import DataReadinessError

ROOT = Path(__file__).resolve().parents[1]


def _contract() -> StrategyContract:
    return load_strategy_contract(ROOT / "configs" / "edge_rebuild_strategy_contract.toml")


def _bars(
    *,
    ticker: str = "AAA",
    day: str = "2026-07-08",
    volumes: tuple[float, ...] = (40.0, 70.0, 25.0, 90.0, 10.0),
    timeframe: str = "1m",
) -> pd.DataFrame:
    starts = pd.date_range(f"{day} 13:30:00Z", periods=len(volumes), freq="1min")
    opens = [100.0 + index for index in range(len(volumes))]
    return pd.DataFrame(
        {
            "ticker": ticker,
            "timeframe": timeframe,
            "bar_start_utc": starts,
            "bar_end_utc": starts + pd.Timedelta(minutes=1),
            "available_at_utc": starts + pd.Timedelta(minutes=2),
            "open": opens,
            "high": [value + 1.0 for value in opens],
            "low": [value - 1.0 for value in opens],
            "close": [value + 0.5 for value in opens],
            "volume": volumes,
            "source": "alpaca",
            "price_feed": "sip",
            "adjustment": "all",
        }
    )


def _activations(
    *,
    ticker: str = "AAA",
    day: str = "2026-07-08",
    activation: str = "2026-07-08 13:34:00Z",
    median: float = 7_800.0,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [ticker],
            "session_date_et": [date.fromisoformat(day)],
            "activation_time_utc": [pd.Timestamp(activation)],
            "median_volume_prior_sessions": [median],
        }
    )


def _build(bars: pd.DataFrame, activations: pd.DataFrame) -> VolumeBarBuildResult:
    contract = _contract()
    return build_causal_volume_bars(
        bars,
        activations,
        contract=contract,
        strategy_contract_sha256=contract.sha256(),
    )


def test_builds_fixed_threshold_ohlcv_and_audits_incomplete_remainder() -> None:
    result = _build(_bars(), _activations())

    assert len(result.bars) == 2
    first = result.bars.iloc[0]
    assert first["volume_threshold"] == 100.0
    assert first["volume"] == 110.0
    assert first["volume_overshoot"] == 10.0
    assert first["source_row_count"] == 2
    assert first["open"] == 100.0
    assert first["high"] == 102.0
    assert first["low"] == 99.0
    assert first["close"] == 101.5
    assert first["first_source_minute_utc"] == pd.Timestamp("2026-07-08 13:30Z")
    assert first["last_source_minute_utc"] == pd.Timestamp("2026-07-08 13:31Z")
    assert not bool(first["model_eligible"])
    assert not bool(result.bars.iloc[1]["model_eligible"])

    audit = result.audit.iloc[0]
    assert audit["incomplete_remainder_source_rows"] == 1
    assert audit["incomplete_remainder_volume"] == 10.0
    assert audit["completed_volume_bars"] == 2
    assert audit["model_eligible_volume_bars"] == 0
    assert result.memory["hard_budget_gib"] == 4.0


def test_model_eligibility_requires_twenty_completed_volume_bars() -> None:
    bars = _bars(volumes=(100.0,) * 20)
    result = _build(
        bars,
        _activations(activation="2026-07-08 13:31:00Z"),
    )

    assert len(result.bars) == 20
    assert not bool(result.bars.iloc[18]["model_eligible"])
    assert bool(result.bars.iloc[19]["model_eligible"])


def test_appended_future_bars_cannot_change_completed_prefix() -> None:
    initial = _build(_bars(volumes=(40.0, 70.0, 25.0)), _activations())
    extended = _build(_bars(volumes=(40.0, 70.0, 25.0, 90.0, 100.0)), _activations())

    pdt.assert_frame_equal(
        initial.bars.reset_index(drop=True),
        extended.bars.iloc[: len(initial.bars)].reset_index(drop=True),
    )


def test_sessions_and_tickers_never_mix() -> None:
    bars = pd.concat(
        [
            _bars(ticker="AAA", day="2026-07-08", volumes=(60.0, 50.0)),
            _bars(ticker="AAA", day="2026-07-09", volumes=(70.0, 40.0)),
            _bars(ticker="BBB", day="2026-07-08", volumes=(80.0, 30.0)),
        ],
        ignore_index=True,
    ).sample(frac=1.0, random_state=7)
    activations = pd.concat(
        [
            _activations(day="2026-07-08", activation="2026-07-08 13:31Z"),
            _activations(day="2026-07-09", activation="2026-07-09 13:31Z"),
            _activations(ticker="BBB", activation="2026-07-08 13:31Z"),
        ],
        ignore_index=True,
    )

    result = _build(bars, activations)

    assert len(result.bars) == 3
    assert set(zip(result.bars["ticker"], result.bars["session_date_et"], strict=True)) == {
        ("AAA", date(2026, 7, 8)),
        ("AAA", date(2026, 7, 9)),
        ("BBB", date(2026, 7, 8)),
    }
    assert result.bars["source_row_count"].eq(2).all()


def test_rejects_five_minute_input() -> None:
    with pytest.raises(DataReadinessError, match="SIP/all 1m"):
        _build(_bars(timeframe="5m"), _activations())


def test_threshold_uses_supplied_prior_session_median_only() -> None:
    low = _build(_bars(volumes=(50.0, 50.0)), _activations(median=7_800.0))
    high = _build(_bars(volumes=(50.0, 50.0)), _activations(median=15_600.0))

    assert len(low.bars) == 1
    assert low.bars.iloc[0]["volume_threshold"] == 100.0
    assert high.bars.empty
    assert high.audit.iloc[0]["volume_threshold"] == 200.0
    assert high.audit.iloc[0]["incomplete_remainder_volume"] == 100.0


def test_early_partial_bar_is_omitted_until_threshold_is_reached() -> None:
    result = _build(_bars(volumes=(30.0, 40.0)), _activations(median=7_800.0))

    assert result.bars.empty
    assert result.audit.iloc[0]["incomplete_remainder_source_rows"] == 2
    assert result.audit.iloc[0]["incomplete_remainder_volume"] == 70.0


def test_output_is_reproducible_for_reordered_input() -> None:
    bars = _bars()
    activations = _activations()
    first = _build(bars, activations)
    second = _build(
        bars.sample(frac=1.0, random_state=11),
        activations.sample(frac=1.0, random_state=13),
    )

    pdt.assert_frame_equal(first.bars, second.bars)
    pdt.assert_frame_equal(first.audit, second.audit)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("price_feed", "iex", "SIP/all 1m"),
        ("source", "other", "SIP/all 1m"),
        ("adjustment", "raw", "SIP/all 1m"),
        ("volume", 0.0, "invalid OHLCV"),
        ("bar_start_utc", pd.Timestamp("2026-07-08 13:30:01Z"), "exact-minute"),
    ],
)
def test_rejects_invalid_canonical_identity_or_values(column: str, value: object, message: str) -> None:
    bars = _bars()
    bars.loc[0, column] = value
    if column == "bar_start_utc":
        bars.loc[0, "bar_end_utc"] = pd.Timestamp(value) + pd.Timedelta(minutes=1)

    with pytest.raises(DataReadinessError, match=message):
        _build(bars, _activations())


def test_rejects_contract_hash_mismatch() -> None:
    contract = _contract()
    with pytest.raises(DataReadinessError, match="contract hash"):
        build_causal_volume_bars(
            _bars(),
            _activations(),
            contract=contract,
            strategy_contract_sha256="0" * 64,
        )
