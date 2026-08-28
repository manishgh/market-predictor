from __future__ import annotations

import hashlib
import json
import pickle
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.features.technical_relationships import (
    TechnicalRelationshipSpec,
    add_technical_relationship_features,
    relationship_spec_from_contract,
    technical_relationship_feature_names,
)


def _spec(
    *,
    groups: tuple[str, ...] = ("ticker",),
    suffix: str = "",
) -> TechnicalRelationshipSpec:
    return TechnicalRelationshipSpec(
        group_columns=groups,
        time_column="time",
        pivot_span_bars=2,
        obv_lookback_bars=5,
        efficiency_lookback_bars=5,
        suffix=suffix,
    )


def _bars(
    close: list[float],
    *,
    ticker: str = "AAA",
    session: str = "2024-01-02",
    rsi: list[float] | None = None,
) -> pd.DataFrame:
    values = np.asarray(close, dtype=float)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "session": session,
            "time": pd.date_range(
                "2024-01-02 14:30",
                periods=len(values),
                freq="5min",
                tz="UTC",
            ),
            "high": values + 1.0,
            "low": values - 1.0,
            "close": values,
            "volume": 1_000.0,
            "rsi_14": rsi if rsi is not None else [60.0] * len(values),
        }
    )


def test_rsi_divergence_waits_for_the_second_pivot_confirmation() -> None:
    high = np.asarray(
        [10, 11, 15, 11, 10, 12, 16, 12, 11, 10, 9, 8],
        dtype=float,
    )
    frame = _bars(
        list(high - 1.0),
        rsi=[50, 60, 80, 65, 55, 58, 60, 55, 50, 45, 40, 35],
    )
    frame["high"] = high
    frame["low"] = high - 2.0

    output = add_technical_relationship_features(frame, spec=_spec())
    strength = output["rsi_bearish_divergence_strength"]
    age = output["rsi_bearish_divergence_confirmation_age_bars"]

    assert strength.iloc[:8].isna().all()
    assert strength.iloc[8] == pytest.approx(
        np.log(16.0 / 15.0) * 0.20,
        rel=1e-6,
    )
    assert age.iloc[8] == 0.0
    assert age.iloc[9] == 1.0


def test_bullish_rsi_divergence_uses_two_confirmed_price_lows() -> None:
    low = np.asarray(
        [14, 13, 10, 13, 14, 12, 8, 12, 13, 14, 15, 16],
        dtype=float,
    )
    frame = _bars(
        list(low + 1.0),
        rsi=[50, 40, 20, 35, 45, 38, 40, 45, 50, 55, 60, 65],
    )
    frame["high"] = low + 2.0
    frame["low"] = low

    output = add_technical_relationship_features(frame, spec=_spec())
    strength = output["rsi_bullish_divergence_strength"]
    age = output["rsi_bullish_divergence_confirmation_age_bars"]

    assert strength.iloc[:8].isna().all()
    assert strength.iloc[8] == pytest.approx(
        -np.log(8.0 / 10.0) * 0.20,
        rel=1e-6,
    )
    assert age.iloc[8] == 0.0
    assert age.iloc[10] == 2.0


def test_appending_future_bars_cannot_change_earlier_relationships() -> None:
    frame = _bars(
        [100, 101, 104, 102, 101, 103, 106, 104, 103, 105, 102, 101],
        rsi=[45, 55, 75, 60, 50, 58, 62, 54, 48, 57, 43, 40],
    )

    prefix = add_technical_relationship_features(frame.iloc[:10], spec=_spec())
    full = add_technical_relationship_features(frame, spec=_spec()).iloc[:10]

    pd.testing.assert_frame_equal(
        prefix.loc[:, technical_relationship_feature_names()],
        full.loc[:, technical_relationship_feature_names()],
        check_exact=True,
    )


def test_obv_and_efficiency_ratio_distinguish_trend_from_range() -> None:
    trend = _bars(
        list(np.arange(100.0, 121.0)),
        ticker="TREND",
        rsi=[70.0] * 21,
    )
    range_cycle = (100.0, 101.0, 102.0, 101.0, 100.0)
    ranging = _bars(
        [range_cycle[index % len(range_cycle)] for index in range(21)],
        ticker="RANGE",
        rsi=[70.0] * 21,
    )
    frame = pd.concat([trend, ranging], ignore_index=True)

    output = add_technical_relationship_features(frame, spec=_spec())
    trend_last = output[output["ticker"] == "TREND"].iloc[-1]
    range_last = output[output["ticker"] == "RANGE"].iloc[-1]

    assert trend_last["obv_directional_change_ratio"] == pytest.approx(1.0)
    assert trend_last["price_obv_confirmation"] == pytest.approx(1.0)
    assert trend_last["kaufman_efficiency_ratio"] == pytest.approx(1.0)
    assert trend_last["rsi_trend_alignment"] == pytest.approx(0.4)
    assert trend_last["rsi_range_position"] == pytest.approx(0.0)

    assert range_last["kaufman_efficiency_ratio"] == pytest.approx(0.0)
    assert range_last["rsi_trend_alignment"] == pytest.approx(0.0)
    assert range_last["rsi_range_position"] == pytest.approx(0.4)


def test_intraday_relationship_windows_reset_at_the_session_boundary() -> None:
    first = _bars(list(np.arange(100.0, 106.0)), session="2024-01-02")
    second = _bars(list(np.arange(106.0, 112.0)), session="2024-01-03")
    second["time"] = pd.date_range(
        "2024-01-03 14:30",
        periods=len(second),
        freq="5min",
        tz="UTC",
    )
    frame = pd.concat([first, second], ignore_index=True)

    output = add_technical_relationship_features(
        frame,
        spec=_spec(groups=("ticker", "session"), suffix="_5m"),
    )
    second_rows = output[output["session"] == "2024-01-03"]

    assert second_rows["kaufman_efficiency_ratio_5m"].iloc[:5].isna().all()
    assert second_rows["kaufman_efficiency_ratio_5m"].iloc[5] == pytest.approx(
        1.0
    )


def test_contract_binds_the_shared_relationship_spec() -> None:
    contract = load_strategy_contract(
        Path("configs/edge_rebuild_strategy_contract.toml")
    )

    spec = relationship_spec_from_contract(
        contract,
        group_columns=("ticker", "session_date_et"),
        time_column="bar_end_utc",
        rsi_column="rsi_14_5m",
        suffix="_5m",
    )

    assert spec.pivot_span_bars == 2
    assert spec.obv_lookback_bars == 20
    assert spec.efficiency_lookback_bars == 20
    assert spec.group_columns == ("ticker", "session_date_et")


def test_relationship_contract_owner_and_hashes_are_stable() -> None:
    contract = load_strategy_contract(
        Path("configs/edge_rebuild_strategy_contract.toml")
    )
    spec = relationship_spec_from_contract(
        contract,
        group_columns=("ticker", "session_date_et"),
        time_column="bar_end_utc",
        rsi_column="rsi_14_5m",
        suffix="_5m",
    )
    restored = pickle.loads(pickle.dumps(spec))
    feature_names = technical_relationship_feature_names()

    assert TechnicalRelationshipSpec.__module__ == (
        "market_predictor.swing.features.technical_relationships"
    )
    assert restored == spec
    assert type(restored).__module__ == (
        "market_predictor.swing.features.technical_relationships"
    )
    assert hashlib.sha256(
        json.dumps(feature_names, separators=(",", ":")).encode("utf-8")
    ).hexdigest() == (
        "6fc5f34e633e3be00092da294bc86afd1d155d3898b7faff415497d67770bf38"
    )
    assert hashlib.sha256(
        json.dumps(asdict(spec), sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest() == (
        "9409760785ae9d31b67866e5f5f92cd118f1dd32b3a3c5a473a107b4836890a4"
    )


def test_representative_relationship_output_hash_is_stable() -> None:
    frame = _bars(
        [100, 101, 104, 102, 101, 103, 106, 104, 103, 105, 102, 101],
        rsi=[45, 55, 75, 60, 50, 58, 62, 54, 48, 57, 43, 40],
    )
    output = add_technical_relationship_features(frame, spec=_spec())
    payload = output.loc[:, technical_relationship_feature_names()].to_json(
        orient="split",
        double_precision=15,
    )

    assert hashlib.sha256(payload.encode("utf-8")).hexdigest() == (
        "814c438377415f3255c7fcd2bb16f005243f47c75302e8a5463c173c4845d4ec"
    )


def test_invalid_or_ambiguous_input_fails_closed() -> None:
    frame = _bars([100, 101, 102, 103, 104, 105])
    duplicate = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataReadinessError, match="one bar per group"):
        add_technical_relationship_features(duplicate, spec=_spec())

    invalid = frame.copy()
    invalid.loc[0, "volume"] = -1
    with pytest.raises(DataReadinessError, match="invalid price, volume, or RSI"):
        add_technical_relationship_features(invalid, spec=_spec())

    with pytest.raises(DataReadinessError, match="lack required columns"):
        add_technical_relationship_features(
            frame.drop(columns="rsi_14"),
            spec=_spec(),
        )
