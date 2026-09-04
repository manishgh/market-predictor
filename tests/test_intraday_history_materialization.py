from __future__ import annotations

import pickle

import pandas as pd
import pytest

import market_predictor.intraday.datasets.history_materialization as history_materialization
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.corpus_integrity import IntegrityThresholds
from market_predictor.intraday.datasets.history_materialization import (
    POSTMARKET,
    PREMARKET,
    REGULAR,
    SessionBounds,
    _quarantine_ticker_defects,
    _resolve_overlapping_sources,
    classify_segments,
    expected_bars_per_session_segment,
    reorganize_intraday_history,
    selected_ticker_sessions,
    session_bounds_for,
)


def test_intraday_history_materialization_has_one_canonical_owner() -> None:
    value = SessionBounds(
        open_at=pd.Timestamp("2024-01-03 14:30:00+00:00"),
        close_at=pd.Timestamp("2024-01-03 21:00:00+00:00"),
    )
    restored = pickle.loads(pickle.dumps(value))
    owner = "market_predictor.intraday.datasets.history_materialization"

    assert SessionBounds.__module__ == owner
    assert type(restored).__module__ == owner
    assert restored == value
    assert history_materialization.SessionBounds is SessionBounds
    for function in (
        reorganize_intraday_history,
        session_bounds_for,
        classify_segments,
        selected_ticker_sessions,
        expected_bars_per_session_segment,
    ):
        assert function.__module__ == owner


def _bars(times_utc: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA"] * len(times_utc),
            "bar_start_utc": pd.to_datetime(pd.Series(times_utc), utc=True),
        }
    )


def test_segments_split_on_the_real_session_open_and_close() -> None:
    bounds = session_bounds_for("2024-01-03", "2024-01-03")
    frame = _bars(
        [
            "2024-01-03 09:00:00+00:00",  # 04:00 ET, pre-market
            "2024-01-03 14:30:00+00:00",  # 09:30 ET, the open itself
            "2024-01-03 20:55:00+00:00",  # 15:55 ET, last regular bar
            "2024-01-03 21:00:00+00:00",  # 16:00 ET, the close itself
            "2024-01-03 23:55:00+00:00",  # 18:55 ET, post-market
        ]
    )

    result = classify_segments(frame, bounds)

    assert list(result["session_segment"]) == [
        PREMARKET,
        REGULAR,
        REGULAR,
        POSTMARKET,
        POSTMARKET,
    ]


def test_early_close_session_is_not_split_by_clock_time() -> None:
    """2024-07-03 closes at 13:00 ET.

    A fixed 09:30-16:00 rule would file three hours of genuine post-market bars
    as regular session, which would leak thin extended-hours volume into a
    regular-session VWAP.
    """

    bounds = session_bounds_for("2024-07-03", "2024-07-03")
    assert bounds["2024-07-03"].close_at == pd.Timestamp("2024-07-03 17:00", tz="UTC")

    frame = _bars(
        [
            "2024-07-03 16:55:00+00:00",  # 12:55 ET, still regular
            "2024-07-03 17:00:00+00:00",  # 13:00 ET, the early close
            "2024-07-03 19:30:00+00:00",  # 15:30 ET, post-market despite the clock
        ]
    )

    result = classify_segments(frame, bounds)

    assert list(result["session_segment"]) == [REGULAR, POSTMARKET, POSTMARKET]


def test_bars_outside_the_window_are_dropped() -> None:
    bounds = session_bounds_for("2024-01-03", "2024-01-03")
    frame = _bars(
        ["2024-01-02 15:00:00+00:00", "2024-01-03 15:00:00+00:00"]
    )

    result = classify_segments(frame, bounds)

    assert len(result) == 1
    assert list(result["session_date_et"]) == ["2024-01-03"]


def test_session_bounds_cover_every_session_in_range() -> None:
    bounds = session_bounds_for("2024-01-02", "2024-01-31")

    assert len(bounds) == 21
    assert all(b.open_at < b.close_at for b in bounds.values())


def _dual_source_bars(price: float, other_price: float | None = None) -> pd.DataFrame:
    """The same bar delivered by two sources, optionally disagreeing."""

    base = {
        "ticker": "FISV",
        "bar_start_utc": pd.Timestamp("2024-01-03 15:00", tz="UTC"),
        "session_date_et": "2024-01-03",
        "session_segment": REGULAR,
        "open": price,
        "high": price,
        "low": price,
        "close": price,
        "volume": 1_000,
    }
    second = dict(base)
    second["history_era"] = "legacy_ohlcv_v1"
    if other_price is not None:
        second |= {"open": other_price, "high": other_price,
                   "low": other_price, "close": other_price}
    return pd.DataFrame([base | {"history_era": "collected"}, second])


def test_identical_bars_from_two_sources_collapse_to_one() -> None:
    """A symbol can be an index member and also be screened in-play."""

    out = _resolve_overlapping_sources(_dual_source_bars(50.0), "FISV")

    assert len(out) == 1
    # The collected copy wins: it carries observed timestamps, not derived ones.
    assert out.iloc[0]["history_era"] == "collected"


def test_sources_disagreeing_on_price_refuse_the_build() -> None:
    """The same instant cannot have traded at two prices."""

    with pytest.raises(DataReadinessError, match="conflicting bars"):
        _resolve_overlapping_sources(_dual_source_bars(50.0, 51.0), "FISV")


def _session_bars(
    ticker: str,
    session: str,
    close: float,
    *,
    distinct_prices: bool = True,
    volume: int = 1_000,
) -> pd.DataFrame:
    start = pd.Timestamp(f"{session} 14:30", tz="UTC")
    times = pd.date_range(start, periods=78, freq="5min")
    offsets = pd.Series(range(78), dtype=float) * (
        0.001 if distinct_prices else 0.0
    )
    closes = close + offsets
    return pd.DataFrame(
        {
            "ticker": ticker,
            "session_date_et": session,
            "session_segment": REGULAR,
            "history_era": "collected",
            "bar_start_utc": times,
            "open": closes,
            "high": closes,
            "low": closes,
            "close": closes,
            "volume": volume,
        }
    )


def test_unprovable_identity_quarantines_the_complete_ticker() -> None:
    bounds = session_bounds_for("2024-01-02", "2024-01-03")
    ordered = pd.concat(
        [
            _session_bars("FI", "2024-01-02", 3.15),
            _session_bars("FI", "2024-01-03", 115.88),
        ],
        ignore_index=True,
    )

    clean, exclusions = _quarantine_ticker_defects(
        ordered,
        "FI",
        expected_bars_per_session_segment(bounds),
        IntegrityThresholds(),
    )

    assert clean.empty
    assert exclusions[0]["scope"] == "ticker"
    assert exclusions[0]["reason"] == "unprovable_identity"


def test_fabricated_session_is_removed_without_losing_other_history() -> None:
    bounds = session_bounds_for("2024-01-02", "2024-01-03")
    ordered = pd.concat(
        [
            _session_bars(
                "STRC",
                "2024-01-02",
                100.0,
                distinct_prices=False,
                volume=0,
            ),
            _session_bars("STRC", "2024-01-03", 101.0),
        ],
        ignore_index=True,
    )

    clean, exclusions = _quarantine_ticker_defects(
        ordered,
        "STRC",
        expected_bars_per_session_segment(bounds),
        IntegrityThresholds(),
    )

    assert set(clean["session_date_et"]) == {"2024-01-03"}
    assert exclusions[0]["scope"] == "ticker_session"
    assert exclusions[0]["session"] == "2024-01-02"
    assert exclusions[0]["reason"] == "fabricated_bars"
