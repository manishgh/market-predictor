"""The relative-volume screen must never read the session it selects for."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from market_predictor.edge_rebuild.intraday_selection import (
    apply_intraday_universe_layers,
    assert_no_current_session_leak,
    session_liquidity,
)
from market_predictor.edge_rebuild.strategy_contract import IntradayUniverseContract
from market_predictor.v3.errors import DataReadinessError

LOOKBACK = 20


def _universe(**overrides: object) -> IntradayUniverseContract:
    payload: dict[str, object] = {
        "scope": "broad_us_point_in_time",
        "index_restricted": False,
        "minimum_average_volume_shares": 1_000_000,
        "average_volume_lookback_sessions": LOOKBACK,
        "minimum_price": 5.0,
        "maximum_price": 500.0,
        "minimum_bar_continuity": 0.95,
        "minimum_relative_volume": 2.0,
        "relative_volume_lookback_sessions": LOOKBACK,
        "relative_volume_excludes_current_session": True,
        "maximum_candidates_per_session": 30,
        "exclude_exchange_traded_products": True,
    }
    payload.update(overrides)
    return IntradayUniverseContract.model_validate(payload)


def _bars(volumes: list[float], *, ticker: str = "AAA", close: float = 25.0) -> pd.DataFrame:
    start = pd.Timestamp("2023-04-10 13:30:00+00:00")
    return pd.DataFrame(
        {
            "ticker": ticker,
            "bar_start_utc": [start + pd.Timedelta(days=offset) for offset in range(len(volumes))],
            "close": close,
            "volume": volumes,
        }
    )


def test_baseline_excludes_the_measured_session() -> None:
    """A spike must not appear in the average it is being compared against."""

    volumes = [1_000_000.0] * LOOKBACK + [50_000_000.0]
    liquidity = session_liquidity(_bars(volumes), lookback_sessions=LOOKBACK)
    spike = liquidity.iloc[LOOKBACK]

    assert spike["session_volume"] == pytest.approx(50_000_000.0)
    # The measured session's own 50M is absent: the mean is exactly the 1M prior.
    assert spike["average_volume_prior_sessions"] == pytest.approx(1_000_000.0)
    assert spike["relative_volume"] == pytest.approx(50.0)
    # An unshifted rolling(20) would place the spike inside its own window and
    # report a materially lower ratio; this pins the leaking value out.
    leaked_mean = float(pd.Series(volumes[-LOOKBACK:]).mean())
    assert spike["average_volume_prior_sessions"] != pytest.approx(leaked_mean)


def test_baseline_needs_a_full_prior_window() -> None:
    """No baseline exists until `lookback` earlier sessions have been observed."""

    liquidity = session_liquidity(_bars([1_000_000.0] * (LOOKBACK + 1)), lookback_sessions=LOOKBACK)

    assert liquidity["average_volume_prior_sessions"].iloc[:LOOKBACK].isna().all()
    assert liquidity["average_volume_prior_sessions"].iloc[LOOKBACK] == pytest.approx(1_000_000.0)
    assert liquidity["baseline_sessions"].iloc[LOOKBACK] == pytest.approx(float(LOOKBACK))


def test_leak_guard_rejects_a_baseline_that_saw_the_current_session() -> None:
    """The guard fails on the classic defect: `rolling(20)` without the shift."""

    volumes = [1_000_000.0] * LOOKBACK + [50_000_000.0]
    liquidity = session_liquidity(_bars(volumes), lookback_sessions=LOOKBACK)
    leaking = liquidity.copy()
    leaking["average_volume_prior_sessions"] = (
        leaking.groupby("ticker", sort=False)["session_volume"]
        .rolling(LOOKBACK, min_periods=LOOKBACK)
        .mean()
        .reset_index(level=0, drop=True)
    )

    with pytest.raises(DataReadinessError, match="prior-session-only replay"):
        assert_no_current_session_leak(leaking, lookback_sessions=LOOKBACK)


def test_leak_guard_rejects_an_over_wide_baseline() -> None:
    """A window spanning more sessions than the lookback has absorbed today."""

    liquidity = session_liquidity(_bars([1_000_000.0] * (LOOKBACK + 1)), lookback_sessions=LOOKBACK)
    liquidity.loc[LOOKBACK, "baseline_sessions"] = float(LOOKBACK + 1)

    with pytest.raises(DataReadinessError, match="more than"):
        assert_no_current_session_leak(liquidity, lookback_sessions=LOOKBACK)


def test_leak_guard_accepts_the_shipped_screen() -> None:
    liquidity = session_liquidity(_bars([1_000_000.0] * 60), lookback_sessions=LOOKBACK)

    assert_no_current_session_leak(liquidity, lookback_sessions=LOOKBACK)


def test_baselines_do_not_bleed_across_symbols() -> None:
    """One symbol's history must never enter another symbol's denominator."""

    quiet = _bars([1_000_000.0] * (LOOKBACK + 1), ticker="AAA")
    loud = _bars([9_000_000.0] * (LOOKBACK + 1), ticker="BBB")
    liquidity = session_liquidity(
        pd.concat([quiet, loud], ignore_index=True),
        lookback_sessions=LOOKBACK,
    )
    final = liquidity.groupby("ticker")["average_volume_prior_sessions"].last()

    assert final["AAA"] == pytest.approx(1_000_000.0)
    assert final["BBB"] == pytest.approx(9_000_000.0)
    assert_no_current_session_leak(liquidity, lookback_sessions=LOOKBACK)


def test_duplicate_symbol_sessions_are_rejected() -> None:
    bars = _bars([1_000_000.0, 1_000_000.0])
    bars.loc[1, "bar_start_utc"] = bars.loc[0, "bar_start_utc"]

    with pytest.raises(DataReadinessError, match="duplicate symbol-session"):
        session_liquidity(bars, lookback_sessions=LOOKBACK)


def test_layer_one_applies_the_volume_floor_and_price_band() -> None:
    below_floor = session_liquidity(
        _bars([100_000.0] * LOOKBACK + [5_000_000.0]),
        lookback_sessions=LOOKBACK,
    )
    out_of_band = session_liquidity(
        _bars([2_000_000.0] * LOOKBACK + [9_000_000.0], close=3.0),
        lookback_sessions=LOOKBACK,
    )

    assert apply_intraday_universe_layers(below_floor, universe=_universe()).empty
    assert apply_intraday_universe_layers(out_of_band, universe=_universe()).empty


def test_layer_two_applies_the_relative_volume_floor() -> None:
    liquidity = session_liquidity(
        _bars([2_000_000.0] * LOOKBACK + [3_000_000.0, 5_000_000.0]),
        lookback_sessions=LOOKBACK,
    )
    selected = apply_intraday_universe_layers(liquidity, universe=_universe())

    # 3M/2M = 1.5 is below the frozen 2.0 floor; only the 5M session survives.
    assert selected["session_volume"].tolist() == [5_000_000.0]


def test_per_session_cap_keeps_the_highest_relative_volume() -> None:
    start = date(2023, 4, 10)
    rows = [
        {
            "ticker": f"T{index:03d}",
            "session_date_et": start + timedelta(days=1),
            "session_volume": 2_000_000.0 * (index + 2),
            "average_volume_prior_sessions": 2_000_000.0,
            "relative_volume": float(index + 2),
            "session_close": 25.0,
            "baseline_sessions": float(LOOKBACK),
        }
        for index in range(45)
    ]
    selected = apply_intraday_universe_layers(pd.DataFrame(rows), universe=_universe())

    assert len(selected) == 30
    assert selected["session_rank"].tolist() == list(range(1, 31))
    assert selected["relative_volume"].max() == pytest.approx(46.0)
    assert selected["relative_volume"].min() == pytest.approx(17.0)
