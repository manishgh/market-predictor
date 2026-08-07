"""Indicators abstain until their window is satisfied.

The absence of this test is what let an `AdvancedIndicatorsStep` recompute
RSI, MACD, the moving averages and the Bollinger bands with ``min_periods=1``
*after* the warm-up history had been dropped, silently overwriting correct
values. Measured on the published v12 panel: 88,999 of 808,684 feature-eligible
rows (11.01%) carried a "200-session average" built from fewer than 200 bars,
all of them in 2019-07-12 -> 2020-04-21, the opening stretch of the training
fold.

A partial window must produce NaN. A fabricated value is worse than a missing
one because the eligibility gates cannot see it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from market_predictor.swing.dataset import _add_technical_features

WINDOWS = {
    "rsi_14": 14,
    "dist_ema_10": 10,
    "dist_ema_20": 20,
    "dist_ema_50": 50,
    "dist_sma_20": 20,
    "dist_sma_50": 50,
    "dist_sma_200": 200,
    "realized_vol_20d": 20,
    "atr_pct_14": 14,
    "bb_pb": 20,
    "bb_upper_dist": 20,
    "bb_lower_dist": 20,
}


def _indicators(frame: pd.DataFrame) -> pd.DataFrame:
    return _add_technical_features(frame, identity_column="security_id")


def _panel(sessions: int, *, closes: np.ndarray | None = None) -> pd.DataFrame:
    rng = np.random.default_rng(20260807)
    if closes is None:
        closes = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, sessions)))
    dates = pd.bdate_range("2019-01-02", periods=sessions)
    return pd.DataFrame(
        {
            "security_id": "TEST",
            "ticker": "TEST",
            "session_date_et": dates.date,
            "open": closes * 0.995,
            "high": closes * 1.01,
            "low": closes * 0.99,
            "close": closes,
            "volume": rng.integers(1_000_000, 5_000_000, sessions),
        }
    )


@pytest.mark.parametrize(("column", "window"), sorted(WINDOWS.items()))
def test_indicator_abstains_until_its_window_is_satisfied(
    column: str, window: int
) -> None:
    out = _indicators(_panel(260))
    assert column in out.columns, f"{column} is no longer produced"

    warmup = out[column].iloc[: window - 1]
    assert warmup.isna().all(), (
        f"{column} emitted {int(warmup.notna().sum())} value(s) from fewer than "
        f"{window} bars; a partial window must abstain"
    )
    assert out[column].iloc[window:].notna().any(), (
        f"{column} never becomes available after {window} bars"
    )


def test_rsi_does_not_fabricate_extremes_on_a_monotonic_series() -> None:
    """A series that only rises has zero average loss; RSI must not read 100."""

    rising = 100.0 * np.cumprod(np.full(60, 1.01))
    out = _indicators(_panel(60, closes=rising))

    rsi = out["rsi_14"].dropna()
    assert not rsi.empty
    assert not np.isinf(rsi).any(), "RSI produced an infinity"
    assert (rsi <= 100.0).all() and (rsi >= 0.0).all()


def test_bollinger_percent_b_is_missing_rather_than_zero_when_undefined() -> None:
    """%B of 0.0 means "price is on the lower band", not "unknown"."""

    flat = np.full(60, 100.0)
    out = _indicators(_panel(60, closes=flat))

    # A flat series has zero dispersion, so the band has no width.
    assert out["bb_pb"].iloc[:19].isna().all()
    assert out["bb_pb"].iloc[20:].isna().all(), (
        "a zero-width band must yield NaN, never a fabricated 0.0"
    )


def test_moving_averages_use_the_full_history_they_are_given() -> None:
    """dist_sma_200 at bar 200 must reflect 200 bars, not a restarted window."""

    out = _indicators(_panel(260))
    close = out["close"]
    expected = close / close.rolling(200, min_periods=200).mean() - 1.0

    pd.testing.assert_series_equal(
        out["dist_sma_200"],
        expected,
        check_names=False,
        rtol=1e-9,
    )
