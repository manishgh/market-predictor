from __future__ import annotations

import pandas as pd

from market_predictor.edge_rebuild.history_materialization import (
    POSTMARKET,
    PREMARKET,
    REGULAR,
    classify_segments,
    session_bounds_for,
)


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
